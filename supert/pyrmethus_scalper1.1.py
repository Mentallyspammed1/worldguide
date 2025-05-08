#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v10.1.0 (Reforged Config & Arcane Clarity)
# Conjures high-frequency trades on Bybit Futures with enhanced config, precision, V5 focus, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 10.1.0 (Reforged: Class-based Config, Enhanced Fill Confirm, Standardized SL/TP, Pyrmethus Style).

Features:
- Dual Supertrend strategy with confirmation.
- ATR for volatility measurement and SL/TP calculation.
- **CRITICAL SAFETY UPGRADE:** Implements exchange-native Stop-Loss and Take-Profit
  orders (both using `stopMarket` type) immediately after entry confirmation,
  based on actual fill price. Uses `fetch_order` primarily for faster confirmation.
- **Includes necessary 'triggerDirection' parameter for Bybit V5 API.**
- Optional Volume spike analysis for entry confirmation.
- Optional Order book pressure analysis for entry confirmation.
- **Enhanced Risk Management:**
    - Risk-based position sizing with margin checks.
    - Checks against exchange minimum order amount and cost *before* placing orders.
    - Caps position size based on `MAX_ORDER_USDT_AMOUNT`.
- **Reforged Configuration:** Uses a dedicated `Config` class for better organization and validation.
- Termux SMS alerts for critical events (with Termux:API check).
- Robust error handling and logging with vibrant Neon color support via Colorama.
- Graceful shutdown on KeyboardInterrupt with position closing attempt.
- Stricter position detection logic (targeting Bybit V5 API).
- **Decimal Precision:** Uses Decimal for critical financial calculations.

Disclaimer:
- **EXTREME RISK**: Arcane energies are volatile. Educational purposes ONLY. High-risk. Use at own absolute risk.
- **EXCHANGE-NATIVE SL/TP:** Relies on exchange-native orders. Subject to exchange performance, slippage, API reliability.
- Parameter Sensitivity: Requires significant tuning and testing in the astral plane (testnet).
- API Rate Limits: Monitor usage lest the exchange spirits grow wary.
- Slippage: Market orders are prone to slippage in turbulent ether.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: Requires Termux:API for SMS communication scrolls. Ensure `pkg install termux-api`.
- API Changes: Exchange APIs (like Bybit V5) can change. Ensure CCXT is updated.

**Installation:**
pip install ccxt pandas pandas_ta python-dotenv colorama # termux-api (if using Termux for SMS)
"""

# Standard Library Imports - The Foundational Runes
import contextlib
import logging
import os
import shutil  # For checking command existence
import subprocess  # For Termux API calls
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

# --- Constants ---

# --- String Constants ---
# Dictionary Keys / Internal Representations
SIDE_KEY = "side"
QTY_KEY = "qty"
ENTRY_PRICE_KEY = "entry_price"
INFO_KEY = "info"
SYMBOL_KEY = "symbol"
ID_KEY = "id"
AVG_PRICE_KEY = "avgPrice"  # Bybit V5 raw field preferred
CONTRACTS_KEY = "contracts"  # CCXT unified field
FILLED_KEY = "filled"
COST_KEY = "cost"
AVERAGE_KEY = "average"  # CCXT unified field for fill price
TIMESTAMP_KEY = "timestamp"
LAST_PRICE_KEY = "last"
BIDS_KEY = "bids"
ASKS_KEY = "asks"
SPREAD_KEY = "spread"
BEST_BID_KEY = "best_bid"
BEST_ASK_KEY = "best_ask"
BID_ASK_RATIO_KEY = "bid_ask_ratio"
ATR_KEY = "atr"
VOLUME_MA_KEY = "volume_ma"
LAST_VOLUME_KEY = "last_volume"
VOLUME_RATIO_KEY = "volume_ratio"
STATUS_KEY = "status"
PRICE_KEY = "price"  # Fallback for average price

# Order Sides / Position Sides
SIDE_BUY = "buy"
SIDE_SELL = "sell"
POSITION_SIDE_LONG = "Long"  # Internal representation for long position
POSITION_SIDE_SHORT = "Short"  # Internal representation for short position
POSITION_SIDE_NONE = "None"  # Internal representation for no position / Bybit V5 side 'None'
BYBIT_SIDE_BUY = "Buy"  # Bybit V5 API side
BYBIT_SIDE_SELL = "Sell"  # Bybit V5 API side

# Order Types / Statuses / Params
ORDER_TYPE_MARKET = "market"
ORDER_TYPE_STOP_MARKET = "stopMarket"  # Used for both SL and TP conditional market orders
# ORDER_TYPE_TAKE_PROFIT_MARKET = 'takeProfitMarket' # Deprecated in favor of stopMarket with triggerDirection
ORDER_STATUS_OPEN = "open"
ORDER_STATUS_CLOSED = "closed"
ORDER_STATUS_CANCELED = "canceled"  # Note: CCXT might use 'cancelled' or 'canceled'
ORDER_STATUS_REJECTED = "rejected"
ORDER_STATUS_EXPIRED = "expired"
PARAM_REDUCE_ONLY = "reduce_only"  # CCXT standard param name
PARAM_STOP_PRICE = "stopPrice"  # CCXT standard param name for trigger price
# PARAM_TRIGGER_PRICE = 'triggerPrice' # Often interchangeable with stopPrice in CCXT, prefer stopPrice
PARAM_TRIGGER_DIRECTION = "triggerDirection"  # Bybit V5 specific for conditional orders (1=above, 2=below)
PARAM_CATEGORY = "category"  # Bybit V5 specific for linear/inverse

# Currencies
USDT_SYMBOL = "USDT"

# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL_STR = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_STR, logging.INFO)

# Custom Log Level for Success
SUCCESS_LEVEL = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self, message, *args, **kwargs) -> None:  # type: ignore
    """Adds a 'success' log level method."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Bind the new method to the Logger class
logging.Logger.success = log_success  # type: ignore

# Basic configuration first
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # logging.FileHandler("scalp_bot_v10.1.log"), # Optional: Log to file
        logging.StreamHandler(sys.stdout)  # Log to console
    ],
)
logger: logging.Logger = logging.getLogger(__name__)

# Apply colors if outputting to a TTY (like Termux)
if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
    # Apply Pyrmethus colors
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
else:
    # Avoid color codes if not a TTY
    logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")  # Ensure level name exists without color


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads, validates, and stores configuration parameters with arcane precision."""

    def __init__(self) -> None:
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        valid = True  # Track overall validity

        # --- API Credentials (Required) ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", None, str, required=True, color=Fore.RED)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", None, str, required=True, color=Fore.RED)
        if not self.api_key or not self.api_secret:
            valid = False

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", str, color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", str, color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, int, color=Fore.YELLOW)
        if self.leverage <= 0:
            logger.critical(f"CRITICAL CONFIG: LEVERAGE invalid: {self.leverage}")
            valid = False
        if self.sleep_seconds <= 0:
            logger.warning(f"CONFIG WARNING: SLEEP_SECONDS ({self.sleep_seconds}) invalid. Setting to 1.")
            self.sleep_seconds = 1

        # --- Risk Management (CRITICAL) ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", Decimal("0.005"), Decimal, color=Fore.GREEN
        )
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", Decimal("1.5"), Decimal, color=Fore.GREEN
        )
        self.atr_take_profit_multiplier: Decimal = self._get_env(
            "ATR_TAKE_PROFIT_MULTIPLIER", Decimal("2.0"), Decimal, color=Fore.GREEN
        )
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", Decimal("500.0"), Decimal, color=Fore.GREEN
        )
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", Decimal("1.05"), Decimal, color=Fore.GREEN
        )
        if not (Decimal(0) < self.risk_per_trade_percentage < Decimal(1)):
            logger.critical(f"CRITICAL CONFIG: RISK_PER_TRADE_PERCENTAGE invalid: {self.risk_per_trade_percentage}")
            valid = False
        if self.atr_stop_loss_multiplier <= 0:
            logger.warning(f"CONFIG WARNING: ATR_STOP_LOSS_MULTIPLIER ({self.atr_stop_loss_multiplier}) not positive.")
        if self.atr_take_profit_multiplier <= 0:
            logger.warning(
                f"CONFIG WARNING: ATR_TAKE_PROFIT_MULTIPLIER ({self.atr_take_profit_multiplier}) not positive."
            )
        if self.max_order_usdt_amount <= 0:
            logger.warning(f"CONFIG WARNING: MAX_ORDER_USDT_AMOUNT ({self.max_order_usdt_amount}) not positive.")
        if self.required_margin_buffer < 1:
            logger.warning(
                f"CONFIG WARNING: REQUIRED_MARGIN_BUFFER ({self.required_margin_buffer}) is less than 1. Margin checks might be ineffective."
            )

        # --- Supertrend Indicator Parameters ---
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, int, color=Fore.CYAN)
        self.st_multiplier: float = float(
            self._get_env("ST_MULTIPLIER", Decimal("2.5"), Decimal, color=Fore.CYAN)
        )  # pandas_ta needs float
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, int, color=Fore.CYAN)
        self.confirm_st_multiplier: float = float(
            self._get_env("CONFIRM_ST_MULTIPLIER", Decimal("2.0"), Decimal, color=Fore.CYAN)
        )  # pandas_ta needs float
        if self.st_atr_length <= 0 or self.confirm_st_atr_length <= 0:
            logger.warning("CONFIG WARNING: Supertrend ATR length(s) are zero or negative.")

        # --- Volume Analysis Parameters ---
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env(
            "VOLUME_SPIKE_THRESHOLD", Decimal("1.5"), Decimal, color=Fore.YELLOW
        )
        self.require_volume_spike_for_entry: bool = self._get_env(
            "REQUIRE_VOLUME_SPIKE_FOR_ENTRY", True, bool, color=Fore.YELLOW
        )
        if self.volume_ma_period <= 0:
            logger.warning("CONFIG WARNING: VOLUME_MA_PERIOD is zero or negative.")

        # --- Order Book Analysis Parameters ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, int, color=Fore.YELLOW)
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG", Decimal("1.2"), Decimal, color=Fore.YELLOW
        )
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT", Decimal("0.8"), Decimal, color=Fore.YELLOW
        )
        self.fetch_order_book_per_cycle: bool = self._get_env(
            "FETCH_ORDER_BOOK_PER_CYCLE", False, bool, color=Fore.YELLOW
        )
        self.use_ob_confirm: bool = self._get_env(
            "USE_OB_CONFIRM", True, bool, color=Fore.YELLOW
        )  # Added explicit OB confirmation flag

        # --- ATR Calculation Parameter (for SL/TP) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, int, color=Fore.GREEN)
        if self.atr_calculation_period <= 0:
            logger.warning("CONFIG WARNING: ATR_CALCULATION_PERIOD is zero or negative.")

        # --- Termux SMS Alert Configuration ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", False, bool, color=Fore.MAGENTA)
        self.sms_recipient_number: str | None = self._get_env("SMS_RECIPIENT_NUMBER", None, str, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, int, color=Fore.MAGENTA)
        if self.enable_sms_alerts and not self.sms_recipient_number:
            logger.warning("CONFIG WARNING: SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = self._get_env("RECV_WINDOW", 10000, int, color=Fore.WHITE)
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)  # Ensure sufficient depth fetched
        self.shallow_ob_fetch_depth: int = self._get_env("SHALLOW_OB_FETCH_DEPTH", 5, int, color=Fore.WHITE)

        # --- Internal Constants & Behavior ---
        self.retry_count: int = self._get_env("RETRY_COUNT", 3, int, color=Fore.WHITE)
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", 1, int, color=Fore.WHITE)
        self.api_fetch_limit_buffer: int = self._get_env("API_FETCH_LIMIT_BUFFER", 5, int, color=Fore.WHITE)
        self.position_qty_epsilon: Decimal = self._get_env(
            "POSITION_QTY_EPSILON", Decimal("1e-9"), Decimal, color=Fore.WHITE
        )
        self.post_close_delay_seconds: int = self._get_env("POST_CLOSE_DELAY_SECONDS", 2, int, color=Fore.WHITE)
        self.post_entry_delay_seconds: float = float(
            self._get_env("POST_ENTRY_DELAY_SECONDS", Decimal("1.0"), Decimal, color=Fore.WHITE)
        )
        self.fetch_order_status_retries: int = self._get_env("FETCH_ORDER_STATUS_RETRIES", 5, int, color=Fore.WHITE)
        self.fetch_order_status_delay: float = float(
            self._get_env("FETCH_ORDER_STATUS_DELAY", Decimal("0.5"), Decimal, color=Fore.WHITE)
        )
        self.enable_monitor_sltp: bool = self._get_env(
            "ENABLE_MONITOR_SLTP", False, bool, color=Fore.YELLOW
        )  # Redundant check flag

        # --- Final Validation Check ---
        if not valid:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}--- Configuration validation FAILED. Cannot proceed. ---{Style.RESET_ALL}"
            )
            raise ValueError("Critical configuration validation failed.")
        else:
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}"
            )

    def _get_env(
        self, var_name: str, default: Any, expected_type: type, required: bool = False, color: str = Fore.WHITE
    ) -> Any:
        """Gets an environment variable, casts type (incl. defaults), logs, handles errors.
        Handles str, int, float, bool, and Decimal types.
        """
        value_str = os.getenv(var_name)
        source = "environment" if value_str is not None else "default"
        value_to_process = value_str if value_str is not None else default

        log_val_str = f"'{value_str}'" if source == "environment" else f"(Default: '{default}')"
        logger.debug(f"{color}Summoning {var_name}: {log_val_str}{Style.RESET_ALL}")

        if value_to_process is None:
            if required:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required config '{var_name}' is missing and has no default.{Style.RESET_ALL}"
                )
                # Raise error immediately if required and no value/default
                raise ValueError(f"Required environment variable '{var_name}' not set and no default provided.")
            return None  # Return None if not required and no value/default

        try:
            if expected_type == bool:
                return str(value_to_process).lower() in ("true", "1", "t", "yes", "y")
            elif expected_type == Decimal:
                return Decimal(str(value_to_process))
            else:
                # Handle int, float, str directly
                return expected_type(value_to_process)
        except (ValueError, TypeError, InvalidOperation) as e:
            env_val_disp = f"'{value_str}'" if value_str is not None else "(Not Set)"
            logger.error(
                f"{Fore.RED}Config Error: Invalid type/value for {var_name}={env_val_disp} (Source: {source}). "
                f"Expected {expected_type.__name__}. Error: {e}. Trying default '{default}'...{Style.RESET_ALL}"
            )
            # Try casting the default again if the primary value failed
            if default is None:
                if required:  # Should have been caught above, but defensive check
                    raise ValueError(f"Required env var '{var_name}' failed casting and has no valid default.")
                return None
            try:
                if expected_type == bool:
                    return str(default).lower() in ("true", "1", "t", "yes", "y")
                elif expected_type == Decimal:
                    return Decimal(str(default))
                else:
                    return expected_type(default)
            except (ValueError, TypeError, InvalidOperation) as e_default:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL CONFIG: Default '{default}' for {var_name} is also incompatible "
                    f"with type {expected_type.__name__}. Error: {e_default}. Cannot proceed.{Style.RESET_ALL}"
                )
                raise ValueError(
                    f"Configuration error: Cannot cast value or default for key '{var_name}' to {expected_type.__name__}."
                )


# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()  # Forge the configuration object
except ValueError:
    # Error already logged within Config init or _get_env
    send_sms_alert("[ScalpBot] CRITICAL: Config validation FAILED. Bot stopped.")  # Send SMS on critical config fail
    sys.exit(1)

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
        _termux_sms_command_exists = shutil.which("termux-sms-send") is not None
        if not _termux_sms_command_exists:
            logger.warning(
                f"{Fore.YELLOW}SMS failed: 'termux-sms-send' command not found. Ensure Termux:API is installed (`pkg install termux-api`) and configured.{Style.RESET_ALL}"
            )

    if not _termux_sms_command_exists:
        return False  # Don't proceed if command is missing

    if not CONFIG.sms_recipient_number:
        logger.warning(f"{Fore.YELLOW}SMS alerts enabled, but SMS_RECIPIENT_NUMBER rune is missing.{Style.RESET_ALL}")
        return False

    try:
        # Prepare the command spell
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
            logger.error(
                f"{Fore.RED}SMS whisper failed. RC: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}"
            )
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
    # API keys already checked in Config, but double-check instance variables
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API keys check failed during initialization.{Style.RESET_ALL}"
        )
        return None

    exchange = ccxt.bybit(
        {
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,  # Built-in rate limiting
            "options": {
                "adjustForTimeDifference": True,  # Adjust for clock skew
                "recvWindow": CONFIG.default_recv_window,  # Increase if timestamp errors occur
                "defaultType": "swap",  # Explicitly default to swap markets
                "warnOnFetchOpenOrdersWithoutSymbol": False,  # Suppress common warning
                "brokerId": "Pyrmethus_Scalp_v10.1",  # Optional: Identify the bot
            },
        }
    )
    try:
        # Test connection and authentication by fetching markets and balance
        logger.debug("Loading market structures...")
        exchange.load_markets()
        logger.debug("Fetching balance (tests authentication)...")
        balance = exchange.fetch_balance()  # Throws AuthenticationError on bad keys
        total_usdt = balance.get("total", {}).get(USDT_SYMBOL, "N/A")
        logger.debug(f"Initial balance fetched: {total_usdt} {USDT_SYMBOL}")
        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}"
        )
        send_sms_alert("[ScalpBot] Initialized successfully and authenticated.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Authentication failed: {e}. Check API key/secret and ensure IP whitelist (if used) is correct and API permissions are sufficient.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
        return None
    except ccxt.NetworkError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Network error during initialization: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Bot stopped.")
        return None
    except ccxt.ExchangeError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Exchange error during initialization: {e}. Check Bybit status page or API documentation for details.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[ScalpBot] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error initializing exchange: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")
        return None


# --- Indicator Calculation Functions - Scrying the Market ---
# (Indicator functions remain largely the same as v10.0, focusing on clarity and Decimal usage)
# Added Pyrmethus logging style.
def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: float, prefix: str = ""
) -> pd.DataFrame:  # Uses float multiplier internally for pandas_ta
    """Calculates the Supertrend indicator using the pandas_ta library."""
    required_input_cols = ["high", "low", "close"]
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [
        f"{col_prefix}supertrend",
        f"{col_prefix}trend",
        f"{col_prefix}st_long",
        f"{col_prefix}st_short",
    ]
    st_col_name = f"SUPERT_{length}_{multiplier}"
    st_trend_col = f"SUPERTd_{length}_{multiplier}"
    st_long_col = f"SUPERTl_{length}_{multiplier}"
    st_short_col = f"SUPERTs_{length}_{multiplier}"

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols):
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}Supertrend): Input DataFrame is missing required columns {required_input_cols} or is empty.{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame()

    if len(df) < length:
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}Supertrend): DataFrame length ({len(df)}) is less than ST period ({length}).{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df

    try:
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={multiplier}")
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)

        if st_col_name not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col_name}, {st_trend_col}")

        cols_to_convert = required_input_cols + [st_col_name, st_trend_col, st_long_col, st_short_col]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[required_input_cols].isnull().values.any():
            logger.warning(
                f"{Fore.YELLOW}Scrying ({col_prefix}ST): NaNs found in input data after coercion.{Style.RESET_ALL}"
            )

        # Rename and process
        df.rename(columns={st_col_name: target_cols[0]}, inplace=True)  # Supertrend value
        df.rename(columns={st_trend_col: target_cols[1]}, inplace=True)  # Trend direction (-1, 1)

        # Calculate flip signals
        prev_trend_direction = df[target_cols[1]].shift(1)
        df[target_cols[2]] = (prev_trend_direction == -1) & (df[target_cols[1]] == 1)  # Long flip
        df[target_cols[3]] = (prev_trend_direction == 1) & (df[target_cols[1]] == -1)  # Short flip
        df[[target_cols[2], target_cols[3]]] = df[[target_cols[2], target_cols[3]]].fillna(False).astype(bool)  # type: ignore

        # Clean up intermediate columns
        cols_to_drop = [c for c in df.columns if c.startswith("SUPERT_") and c not in target_cols]
        df.drop(columns=list(set(cols_to_drop)), errors="ignore", inplace=True)

        # Log last candle result
        last_trend_val = df[target_cols[1]].iloc[-1] if not df.empty and pd.notna(df[target_cols[1]].iloc[-1]) else None
        last_trend_str = "Up" if last_trend_val == 1 else "Down" if last_trend_val == -1 else "N/A"
        last_st_val = (
            df[target_cols[0]].iloc[-1] if not df.empty and pd.notna(df[target_cols[0]].iloc[-1]) else float("nan")
        )
        trend_color = Fore.GREEN if last_trend_str == "Up" else Fore.RED if last_trend_str == "Down" else Fore.WHITE
        logger.debug(
            f"Scrying ({col_prefix}ST({length}, {multiplier})): Last Trend={trend_color}{last_trend_str}{Style.RESET_ALL}, Last Value={last_st_val:.4f}"
        )

    except (KeyError, AttributeError, Exception) as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}Supertrend): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA
    return df if df is not None else pd.DataFrame()


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume MA, and checks for volume spikes."""
    results: dict[str, Decimal | None] = {
        ATR_KEY: None,
        VOLUME_MA_KEY: None,
        LAST_VOLUME_KEY: None,
        VOLUME_RATIO_KEY: None,
    }
    required_cols = ["high", "low", "close", "volume"]

    if df is None or df.empty or not all(c in df.columns for c in required_cols):
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): Input DataFrame is missing required columns {required_cols} or is empty.{Style.RESET_ALL}"
        )
        return results
    min_len = max(atr_len, vol_ma_len, 1)
    if len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): DataFrame length ({len(df)}) < required ({min_len}) for ATR({atr_len})/VolMA({vol_ma_len}).{Style.RESET_ALL}"
        )
        return results

    try:
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[required_cols].isnull().values.any():
            logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): NaNs found in input data after coercion.{Style.RESET_ALL}")

        # Calculate ATR
        atr_col = f"ATRr_{atr_len}"
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]):
            try:
                results[ATR_KEY] = Decimal(str(df[atr_col].iloc[-1]))
            except InvalidOperation:
                logger.warning(f"Scrying (ATR): Invalid Decimal value for ATR: {df[atr_col].iloc[-1]}")
        else:
            logger.warning(f"Scrying: Failed to calculate valid ATR({atr_len}).")
        df.drop(columns=[atr_col], errors="ignore", inplace=True)

        # Calculate Volume MA
        volume_ma_col = f"volume_ma_{vol_ma_len}"
        df[volume_ma_col] = df["volume"].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()

        if pd.notna(df[volume_ma_col].iloc[-1]) and pd.notna(df["volume"].iloc[-1]):
            try:
                results[VOLUME_MA_KEY] = Decimal(str(df[volume_ma_col].iloc[-1]))
                results[LAST_VOLUME_KEY] = Decimal(str(df["volume"].iloc[-1]))
            except InvalidOperation:
                logger.warning("Scrying (Vol): Invalid Decimal value for Volume/MA.")

            # Calculate Volume Ratio
            if (
                results[VOLUME_MA_KEY]
                and results[VOLUME_MA_KEY] > CONFIG.position_qty_epsilon
                and results[LAST_VOLUME_KEY]
            ):
                try:
                    results[VOLUME_RATIO_KEY] = (results[LAST_VOLUME_KEY] / results[VOLUME_MA_KEY]).quantize(
                        Decimal("0.01")
                    )
                except InvalidOperation:
                    logger.warning("Scrying (Vol): Invalid Decimal operation for ratio.")
                    results[VOLUME_RATIO_KEY] = None
            else:
                results[VOLUME_RATIO_KEY] = None
                logger.debug(
                    f"Scrying (Vol): Ratio calc skipped (Vol={results[LAST_VOLUME_KEY]}, MA={results[VOLUME_MA_KEY]})"
                )
        else:
            logger.warning(f"Scrying (Vol): Failed calc VolMA({vol_ma_len}) or get last vol.")
        df.drop(columns=[volume_ma_col], errors="ignore", inplace=True)

        # Log results
        atr_str = f"{results[ATR_KEY]:.4f}" if results[ATR_KEY] else "N/A"
        last_vol_val = results.get(LAST_VOLUME_KEY)
        vol_ma_val = results.get(VOLUME_MA_KEY)
        vol_ratio_val = results.get(VOLUME_RATIO_KEY)
        last_vol_str = f"{last_vol_val:.2f}" if last_vol_val else "N/A"
        vol_ma_str = f"{vol_ma_val:.2f}" if vol_ma_val else "N/A"
        vol_ratio_str = f"{vol_ratio_val:.2f}" if vol_ratio_val else "N/A"

        logger.debug(f"Scrying Results: ATR({atr_len}) = {Fore.CYAN}{atr_str}{Style.RESET_ALL}")
        logger.debug(
            f"Scrying Results: Volume: Last={last_vol_str}, MA({vol_ma_len})={vol_ma_str}, Ratio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}"
        )

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)
    return results


def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> dict[str, Decimal | None]:  # Returns Decimal or None
    """Fetches L2 order book and analyzes bid/ask pressure and spread."""
    results: dict[str, Decimal | None] = {
        BID_ASK_RATIO_KEY: None,
        SPREAD_KEY: None,
        BEST_BID_KEY: None,
        BEST_ASK_KEY: None,
    }
    logger.debug(
        f"Order Book Scrying: Fetching L2 for {symbol} (Analyze Depth: {depth}, API Fetch Limit: {fetch_limit})..."
    )

    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)

        if (
            not order_book
            or not isinstance(order_book.get(BIDS_KEY), list)
            or not isinstance(order_book.get(ASKS_KEY), list)
        ):
            logger.warning(
                f"{Fore.YELLOW}Order Book Scrying: Incomplete or invalid data structure received for {symbol}.{Style.RESET_ALL}"
            )
            return results

        bids: list[list[float | str]] = order_book[BIDS_KEY]
        asks: list[list[float | str]] = order_book[ASKS_KEY]

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}Order Book Scrying: Bids or asks list is empty for {symbol}. Bids: {len(bids)}, Asks: {len(asks)}{Style.RESET_ALL}"
            )
            return results

        # Get best bid/ask and calculate spread
        try:
            best_bid_raw = bids[0][0] if len(bids[0]) > 0 else "0.0"
            best_ask_raw = asks[0][0] if len(asks[0]) > 0 else "0.0"
            results[BEST_BID_KEY] = Decimal(str(best_bid_raw))
            results[BEST_ASK_KEY] = Decimal(str(best_ask_raw))

            if results[BEST_BID_KEY] > 0 and results[BEST_ASK_KEY] > 0:
                spread = results[BEST_ASK_KEY] - results[BEST_BID_KEY]
                results[SPREAD_KEY] = spread.quantize(Decimal("0.0001"))  # Adjust precision as needed
                logger.debug(
                    f"OB Scrying: Best Bid={Fore.GREEN}{results[BEST_BID_KEY]:.4f}{Style.RESET_ALL}, Best Ask={Fore.RED}{results[BEST_ASK_KEY]:.4f}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results[SPREAD_KEY]:.4f}{Style.RESET_ALL}"
                )
            else:
                logger.debug("OB Scrying: Could not calculate spread (Bid/Ask zero or invalid).")

        except (IndexError, InvalidOperation, ValueError, TypeError) as e:
            logger.warning(
                f"{Fore.YELLOW}OB Scrying: Error processing best bid/ask/spread for {symbol}: {e}{Style.RESET_ALL}"
            )
            results[BEST_BID_KEY] = None
            results[BEST_ASK_KEY] = None
            results[SPREAD_KEY] = None

        # Calculate cumulative volume within depth
        try:
            bid_volume_sum_raw = sum(Decimal(str(bid[1])) for bid in bids[:depth] if len(bid) > 1)
            ask_volume_sum_raw = sum(Decimal(str(ask[1])) for ask in asks[:depth] if len(ask) > 1)
            bid_volume_sum = bid_volume_sum_raw.quantize(Decimal("0.0001"))
            ask_volume_sum = ask_volume_sum_raw.quantize(Decimal("0.0001"))
            logger.debug(
                f"OB Scrying (Depth {depth}): Cum Bid={Fore.GREEN}{bid_volume_sum:.4f}{Style.RESET_ALL}, Cum Ask={Fore.RED}{ask_volume_sum:.4f}{Style.RESET_ALL}"
            )

            # Calculate Bid/Ask Ratio
            if ask_volume_sum > CONFIG.position_qty_epsilon:
                bid_ask_ratio = (bid_volume_sum / ask_volume_sum).quantize(Decimal("0.01"))
                results[BID_ASK_RATIO_KEY] = bid_ask_ratio
                ratio_color = (
                    Fore.GREEN
                    if bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
                    else (Fore.RED if bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW)
                )
                logger.debug(f"OB Scrying Ratio (Depth {depth}) = {ratio_color}{bid_ask_ratio:.3f}{Style.RESET_ALL}")
            else:
                logger.debug(
                    f"OB Scrying Ratio calculation skipped (Ask volume at depth {depth} is zero or negligible)"
                )

        except (IndexError, InvalidOperation, ValueError, TypeError) as e:
            logger.warning(
                f"{Fore.YELLOW}OB Scrying: Error calculating cumulative volume or ratio for {symbol}: {e}{Style.RESET_ALL}"
            )
            results[BID_ASK_RATIO_KEY] = None

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}OB Scrying: API error fetching order book for {symbol}: {e}{Style.RESET_ALL}")
    except (IndexError, InvalidOperation, ValueError, TypeError) as e:
        logger.warning(f"{Fore.YELLOW}OB Scrying: Error processing OB data for {symbol}: {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}OB Scrying: Unexpected error analyzing order book for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return dict.fromkeys(results)  # Return None dict on error

    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    """Fetches OHLCV data with retries and basic validation."""
    logger.info(f"Data Fetch: Gathering {limit} {timeframe} candles for {symbol}...")
    for attempt in range(CONFIG.retry_count):
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                logger.warning(
                    f"{Fore.YELLOW}Data Fetch: Received empty OHLCV data for {symbol} on attempt {attempt + 1}.{Style.RESET_ALL}"
                )
                if attempt < CONFIG.retry_count - 1:
                    time.sleep(CONFIG.retry_delay_seconds)
                    continue
                else:
                    return None  # Return None after final retry

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)  # Time magic
            df.set_index("timestamp", inplace=True)

            # --- Basic Data Validation ---
            if df.empty:
                logger.warning(
                    f"{Fore.YELLOW}Data Fetch: DataFrame is empty after conversion for {symbol}.{Style.RESET_ALL}"
                )
                return None  # Cannot proceed with empty DataFrame

            # Check for NaNs
            if df.isnull().values.any():
                nan_counts = df.isnull().sum()
                logger.warning(
                    f"{Fore.YELLOW}Data Fetch: Fetched OHLCV data contains NaN values. Counts:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...{Style.RESET_ALL}"
                )
                # Simple imputation: Forward fill NaNs. More sophisticated methods could be used.
                df.ffill(inplace=True)  # type: ignore
                # Check again after filling - if NaNs remain (e.g., at the very beginning), data is unusable
                if df.isnull().values.any():
                    logger.error(
                        f"{Fore.RED}Data Fetch: NaN values remain after forward fill. Cannot proceed with this data batch.{Style.RESET_ALL}"
                    )
                    return None

            logger.debug(f"Data Fetch: Successfully woven {len(df)} OHLCV candles for {symbol}.")
            return df

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: API error fetching OHLCV for {symbol} (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}"
            )
            if attempt < CONFIG.retry_count - 1:
                time.sleep(CONFIG.retry_delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Data Fetch: Failed to fetch OHLCV for {symbol} after {CONFIG.retry_count} attempts.{Style.RESET_ALL}"
                )
                return None
        except Exception as e:
            logger.error(
                f"{Fore.RED}Data Fetch: Unexpected error fetching market data for {symbol}: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            return None  # Return None on unexpected errors

    return None  # Should not be reached if loop completes, but included for safety


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:  # Returns Decimal for qty/price
    """Fetches current position details for Bybit V5 via CCXT."""
    # Default: no active position, using Decimal for precision
    default_pos: dict[str, Any] = {
        SIDE_KEY: POSITION_SIDE_NONE,
        QTY_KEY: Decimal("0.0"),
        ENTRY_PRICE_KEY: Decimal("0.0"),
    }
    ccxt_unified_symbol = symbol
    market_id = None
    market = None

    # Get Market Info
    try:
        market = exchange.market(ccxt_unified_symbol)
        if not market:
            raise KeyError(f"Market info not found for {ccxt_unified_symbol}")
        market_id = market.get(ID_KEY)
        if not market_id:
            raise KeyError(f"Market ID not found in market info for {ccxt_unified_symbol}")
        logger.debug(
            f"Position Check: Fetching position for CCXT symbol '{ccxt_unified_symbol}' (Target Exchange Market ID: '{market_id}')..."
        )
    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(
            f"{Fore.RED}Position Check: Failed get market info/ID for '{ccxt_unified_symbol}': {e}{Style.RESET_ALL}"
        )
        return default_pos
    except Exception as e:
        logger.error(
            f"{Fore.RED}Position Check: Unexpected error getting market info for '{ccxt_unified_symbol}': {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return default_pos

    # Fetch Positions
    try:
        if not exchange.has.get("fetchPositions"):  # type: ignore
            logger.warning(
                f"{Fore.YELLOW}Position Check: Exchange '{exchange.id}' may not support fetchPositions.{Style.RESET_ALL}"
            )
            return default_pos

        # Determine category for V5 API call
        params = {}
        if market and market.get("linear", False):
            params = {PARAM_CATEGORY: "linear"}
        elif market and market.get("inverse", False):
            params = {PARAM_CATEGORY: "inverse"}

        positions = exchange.fetch_positions(symbols=[ccxt_unified_symbol], params=params)
        logger.debug(f"Fetched positions data (raw count: {len(positions)})")

        # Filter positions
        for pos in positions:
            pos_info = pos.get(INFO_KEY, {})
            pos_symbol_raw = pos_info.get(SYMBOL_KEY)

            if pos_symbol_raw != market_id:
                continue  # Skip if not the target symbol

            # Check for One-Way Mode (positionIdx=0)
            position_idx = pos_info.get("positionIdx", -1)
            try:
                position_idx = int(position_idx)
            except (ValueError, TypeError):
                position_idx = -1
            if position_idx != 0:
                continue  # Skip hedge mode positions

            # Check Side (V5: 'Buy', 'Sell', 'None')
            pos_side_v5 = pos_info.get(SIDE_KEY, POSITION_SIDE_NONE)
            determined_side = POSITION_SIDE_NONE
            if pos_side_v5 == BYBIT_SIDE_BUY:
                determined_side = POSITION_SIDE_LONG
            elif pos_side_v5 == BYBIT_SIDE_SELL:
                determined_side = POSITION_SIDE_SHORT
            else:
                continue  # Skip if side is 'None' or unexpected

            # Check Position Size (V5: 'size')
            size_str = pos_info.get("size")
            if size_str is None or size_str == "":
                continue  # Skip if size field missing

            try:
                size = Decimal(str(size_str))
                if abs(size) > CONFIG.position_qty_epsilon:
                    # Found active position! Get entry price.
                    entry_price_str = pos_info.get(AVG_PRICE_KEY) or pos.get(ENTRY_PRICE_KEY)  # Prefer raw V5 avgPrice
                    entry_price = Decimal("0.0")
                    if entry_price_str is not None and entry_price_str != "":
                        try:
                            entry_price = Decimal(str(entry_price_str))
                        except (InvalidOperation, ValueError, TypeError) as price_err:
                            logger.warning(
                                f"Could not parse entry price string: '{entry_price_str}'. Defaulting to 0.0. Error: {price_err}"
                            )
                    qty_abs = abs(size)
                    pos_color = Fore.GREEN if determined_side == POSITION_SIDE_LONG else Fore.RED
                    logger.info(
                        f"{pos_color}Position Check: FOUND Active Position for {market_id}: Side={determined_side}, Qty={qty_abs}, Entry={entry_price:.4f}{Style.RESET_ALL}"
                    )
                    return {SIDE_KEY: determined_side, QTY_KEY: qty_abs, ENTRY_PRICE_KEY: entry_price}
                else:
                    continue  # Size negligible, treat as flat

            except (ValueError, TypeError, InvalidOperation) as e:
                logger.warning(f"{Fore.YELLOW}Position Check: Error parsing size '{size_str}': {e}{Style.RESET_ALL}")
                continue
        # End of loop

        logger.info(
            f"{Fore.BLUE}Position Check: No active position found for {market_id} (One-Way Mode).{Style.RESET_ALL}"
        )
        return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Position Check: API error during fetch_positions for {symbol}: {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Position Check: Unexpected error during position check for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    logger.warning(
        f"{Fore.YELLOW}Position Check: Returning default (No Position) due to error or no active position found for {symbol}.{Style.RESET_ALL}"
    )
    return default_pos


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage, checks market type, retries."""
    logger.info(f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    market_base = "N/A"
    try:
        market = exchange.market(symbol)
        market_base = market.get("base", "N/A")
        if not market.get("contract", False) or market.get("spot", False):
            logger.error(
                f"{Fore.RED}Leverage Conjuring: Cannot set leverage for non-contract market: {symbol}. Market type: {market.get('type')}{Style.RESET_ALL}"
            )
            return False
    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(
            f"{Fore.RED}Leverage Conjuring: Failed to get market info for symbol '{symbol}': {e}{Style.RESET_ALL}"
        )
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}Leverage Conjuring: Unexpected error getting market info for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return False

    for attempt in range(CONFIG.retry_count):
        try:
            # Bybit V5 requires setting buy and sell leverage separately for set_leverage call via CCXT
            params = {"buyLeverage": str(leverage), "sellLeverage": str(leverage)}
            logger.debug(
                f"Leverage Conjuring: Calling exchange.set_leverage({leverage}, '{symbol}', params={params}) (Attempt {attempt + 1}/{CONFIG.retry_count})"
            )
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(
                f"{Fore.GREEN}Leverage Conjuring: Successfully set leverage to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}"
            )
            return True

        except ccxt.ExchangeError as e:
            error_msg_lower = str(e).lower()
            if any(
                p in error_msg_lower
                for p in [
                    "leverage not modified",
                    "same leverage",
                    "no need to modify leverage",
                    "leverage is same as requested",
                ]
            ):
                logger.info(
                    f"{Fore.CYAN}Leverage Conjuring: Leverage for {symbol} already set to {leverage}x (Confirmed by exchange message).{Style.RESET_ALL}"
                )
                return True
            logger.warning(
                f"{Fore.YELLOW}Leverage Conjuring: Exchange resistance on attempt {attempt + 1}/{CONFIG.retry_count} for {symbol}: {e}{Style.RESET_ALL}"
            )
            if attempt < CONFIG.retry_count - 1:
                time.sleep(CONFIG.retry_delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: FAILED after {CONFIG.retry_count} attempts due to exchange error: {e}{Style.RESET_ALL}"
                )
                send_sms_alert(f"[{market_base}] CRITICAL: Leverage set FAILED {symbol}.")
                return False
        except ccxt.NetworkError as e:
            logger.warning(
                f"{Fore.YELLOW}Leverage Conjuring: Network error on attempt {attempt + 1}/{CONFIG.retry_count} for {symbol}: {e}{Style.RESET_ALL}"
            )
            if attempt < CONFIG.retry_count - 1:
                time.sleep(CONFIG.retry_delay_seconds)
            else:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring: FAILED after {CONFIG.retry_count} attempts due to network error.{Style.RESET_ALL}"
                )
                send_sms_alert(f"[{market_base}] CRITICAL: Leverage set FAILED (Network) {symbol}.")
                return False
        except Exception as e:
            logger.error(
                f"{Fore.RED}Leverage Conjuring: Unexpected error setting leverage for {symbol} on attempt {attempt + 1}: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            return False  # Exit immediately on unexpected errors
    return False


def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal"
) -> dict[str, Any] | None:
    """Closes active position via market order with reduce_only, validates first."""
    initial_side = position_to_close.get(SIDE_KEY, POSITION_SIDE_NONE)
    initial_qty = position_to_close.get(QTY_KEY, Decimal("0.0"))
    market_base = symbol.split("/")[0]

    logger.info(
        f"{Fore.YELLOW}Banish Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}"
    )

    # Re-validate Position Before Closing
    logger.debug("Banish Position: Re-validating live position status...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position.get(SIDE_KEY, POSITION_SIDE_NONE)
    live_amount_to_close = live_position.get(QTY_KEY, Decimal("0.0"))  # Absolute value

    if live_position_side == POSITION_SIDE_NONE or live_amount_to_close <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Banish Position: Discrepancy detected. Initial check showed {initial_side}, but live check shows none or negligible qty ({live_amount_to_close:.8f}). Assuming already closed.{Style.RESET_ALL}"
        )
        return None

    # Determine the side of the market order needed to close
    side_to_execute_close = SIDE_SELL if live_position_side == POSITION_SIDE_LONG else SIDE_BUY

    # Place Reduce-Only Market Order
    params = {PARAM_REDUCE_ONLY: True}

    try:
        amount_float = float(live_amount_to_close)
        amount_str = exchange.amount_to_precision(symbol, amount_float)
        amount_float_prec = float(amount_str)

        if amount_float_prec <= float(CONFIG.position_qty_epsilon):
            logger.error(f"{Fore.RED}Banish Pos: Closing amount {amount_str} negligible. Abort.{Style.RESET_ALL}")
            return None

        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}Banish Position: Attempting to CLOSE {live_position_side} position ({reason}): Executing {side_to_execute_close.upper()} MARKET order for {amount_str} {symbol} (Reduce-Only){Style.RESET_ALL}"
        )

        order = exchange.create_market_order(
            symbol=symbol, side=side_to_execute_close, amount=amount_float_prec, params=params
        )

        # Log details immediately
        fill_price_str = "?"
        if order.get(AVERAGE_KEY) is not None:
            with contextlib.suppress(Exception):
                fill_price_str = f"{Decimal(str(order.get(AVERAGE_KEY))):.4f}"
        elif order.get(PRICE_KEY) is not None:
            with contextlib.suppress(Exception):
                fill_price_str = f"{Decimal(str(order.get(PRICE_KEY))):.4f}"
        filled_qty_str = "?"
        if order.get(FILLED_KEY) is not None:
            with contextlib.suppress(Exception):
                filled_qty_str = f"{Decimal(str(order.get(FILLED_KEY))):.8f}"
        order_id_short = str(order.get(ID_KEY, "N/A"))[-6:]
        cost_str = "?"
        if order.get(COST_KEY) is not None:
            with contextlib.suppress(Exception):
                cost_str = f"{Decimal(str(order.get(COST_KEY))):.2f}"

        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Banish Position: CLOSE Order ({reason}) placed successfully for {symbol}. Qty Filled: {filled_qty_str}/{amount_str}, Avg Fill ~{fill_price_str}, Cost: {cost_str} USDT. ID:...{order_id_short}{Style.RESET_ALL}"
        )

        # Send SMS Alert
        sms_msg = f"[{market_base}] BANISHED {live_position_side} {amount_str} @ ~{fill_price_str} ({reason}). ID:...{order_id_short}"
        send_sms_alert(sms_msg)
        return order

    except ccxt.InsufficientFunds as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Insufficient funds error during close attempt: {e}{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insuff funds! Check margin.")
    except ccxt.NetworkError as e:
        logger.error(f"{Fore.RED}Banish Position ({reason}): Network error placing close order: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Network error! Check connection.")
    except ccxt.ExchangeError as e:
        err_str_lower = str(e).lower()
        # Check for specific errors indicating already closed/closing
        if any(
            phrase in err_str_lower
            for phrase in [
                "order would not reduce position size",
                "position is zero",
                "position size is zero",
                "cannot be less than",
            ]
        ):
            logger.warning(
                f"{Fore.YELLOW}Banish Position ({reason}): Exchange indicates order would not reduce size or position is zero. Assuming already closed.{Style.RESET_ALL}"
            )
            return None  # Treat as success/non-actionable
        logger.error(f"{Fore.RED}Banish Position ({reason}): Exchange error placing close order: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): API error ({type(e).__name__}).")
    except (ValueError, TypeError, InvalidOperation) as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Value error during amount processing (Qty: {live_amount_to_close}): {e}{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Value error ({type(e).__name__}).")
    except Exception as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Unexpected error placing close order: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Unexpected error ({type(e).__name__}). Check logs!")

    return None


def calculate_position_size(
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange,
) -> tuple[float | None, float | None]:  # Returns float for create_order
    """Calculates position size based on risk, checks limits."""
    logger.debug(
        f"Risk Calc Input: Equity={equity:.2f}, Risk%={risk_per_trade_pct:.3%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x, Symbol={symbol}"
    )

    # Input Validation
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(f"{Fore.RED}Risk Calc: Invalid entry/SL price (> 0).{Style.RESET_ALL}")
        return None, None
    price_difference_per_unit = abs(entry_price - stop_loss_price)
    if price_difference_per_unit <= CONFIG.position_qty_epsilon:
        logger.error(f"{Fore.RED}Risk Calc: Entry/SL prices too close.{Style.RESET_ALL}")
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(f"{Fore.RED}Risk Calc: Invalid risk %: {risk_per_trade_pct}.{Style.RESET_ALL}")
        return None, None
    if equity <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.2f}{Style.RESET_ALL}")
        return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}{Style.RESET_ALL}")
        return None, None

    # Calculation
    risk_amount_usdt: Decimal = equity * risk_per_trade_pct
    quantity: Decimal = risk_amount_usdt / price_difference_per_unit

    # Apply exchange precision
    try:
        quantity_precise_str = exchange.amount_to_precision(symbol, float(quantity))
        quantity_precise = Decimal(quantity_precise_str)
        logger.debug(f"Risk Calc: Raw Qty={quantity:.18f}, Precise Qty={quantity_precise_str}")
        quantity = quantity_precise
    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Could not apply precision to qty {quantity:.8f} for {symbol}. Using raw. Err: {e}{Style.RESET_ALL}"
        )

    if quantity <= CONFIG.position_qty_epsilon:
        logger.error(f"{Fore.RED}Risk Calc: Calculated quantity ({quantity}) zero/negligible.{Style.RESET_ALL}")
        return None, None

    # Estimate Value and Margin
    position_value_usdt: Decimal = quantity * entry_price
    required_margin_estimate: Decimal = position_value_usdt / Decimal(leverage)
    logger.debug(
        f"Risk Calc Output: RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference_per_unit:.4f} => PreciseQty={quantity:.8f}, EstVal={position_value_usdt:.2f}, EstMargin={required_margin_estimate:.2f}"
    )

    # Exchange Limit Checks
    try:
        market = exchange.market(symbol)
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})
        min_amount = (
            Decimal(str(amount_limits.get("min", "0"))) if amount_limits.get("min") is not None else Decimal("0")
        )
        max_amount = (
            Decimal(str(amount_limits.get("max", "inf"))) if amount_limits.get("max") is not None else Decimal("inf")
        )
        min_cost = Decimal(str(cost_limits.get("min", "0"))) if cost_limits.get("min") is not None else Decimal("0")
        max_cost = Decimal(str(cost_limits.get("max", "inf"))) if cost_limits.get("max") is not None else Decimal("inf")
        logger.debug(
            f"Market Limits for {symbol}: MinAmt={min_amount}, MaxAmt={max_amount}, MinCost={min_cost}, MaxCost={max_cost}"
        )

        if quantity < min_amount:
            logger.error(f"{Fore.RED}Risk Calc: Qty {quantity:.8f} < MinAmt {min_amount:.8f}.{Style.RESET_ALL}")
            return None, None
        if position_value_usdt < min_cost:
            logger.error(
                f"{Fore.RED}Risk Calc: EstVal {position_value_usdt:.2f} < MinCost {min_cost:.2f}.{Style.RESET_ALL}"
            )
            return None, None
        if quantity > max_amount:
            logger.warning(
                f"{Fore.YELLOW}Risk Calc: Qty {quantity:.8f} > MaxAmt {max_amount:.8f}. Capping.{Style.RESET_ALL}"
            )
            quantity = max_amount
            position_value_usdt = quantity * entry_price
            required_margin_estimate = position_value_usdt / Decimal(leverage)
            logger.info(
                f"Risk Calc: Capped Qty={quantity:.8f}, New EstVal={position_value_usdt:.2f}, New EstMargin={required_margin_estimate:.2f}"
            )
        if position_value_usdt > max_cost:
            logger.error(
                f"{Fore.RED}Risk Calc: EstVal {position_value_usdt:.2f} > MaxCost {max_cost:.2f}.{Style.RESET_ALL}"
            )
            return None, None

    except (ccxt.BadSymbol, KeyError) as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Could not fetch market limits for {symbol}: {e}. Skipping checks.{Style.RESET_ALL}"
        )
    except (InvalidOperation, ValueError, TypeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Error parsing market limits for {symbol}: {e}. Skipping checks.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Unexpected error checking market limits: {e}. Skipping checks.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    # Return float for CCXT compatibility
    return float(quantity), float(required_margin_estimate)


def confirm_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str) -> dict[str, Any] | None:
    """Confirms if an order is filled using fetch_order primarily, falling back to fetch_closed_orders.

    Args:
        exchange: Initialized CCXT exchange object.
        order_id: The ID of the order to confirm.
        symbol: Unified CCXT symbol.

    Returns:
        Optional[Dict[str, Any]]: The filled order details, or None if not confirmed filled/failed.
    """
    log_prefix = f"Fill Confirm (ID:...{order_id[-6:]})"
    logger.debug(f"{log_prefix}: Attempting to confirm fill...")
    start_time = time.time()
    confirmed_order = None

    # --- Primary Method: fetch_order ---
    for attempt in range(CONFIG.fetch_order_status_retries):
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get(STATUS_KEY)
            logger.debug(f"{log_prefix}: Attempt {attempt + 1}, fetch_order status: {status}")

            if status == ORDER_STATUS_CLOSED:
                logger.success(f"{log_prefix}: Confirmed FILLED via fetch_order.")
                confirmed_order = order
                break  # Exit loop on success
            elif status in [ORDER_STATUS_CANCELED, ORDER_STATUS_REJECTED, ORDER_STATUS_EXPIRED]:
                logger.error(
                    f"{Fore.RED}{log_prefix}: Order FAILED with status '{status}' via fetch_order.{Style.RESET_ALL}"
                )
                return None  # Order definitively failed

            # If status is 'open' or None, continue retrying

        except ccxt.OrderNotFound:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix}: Order not found via fetch_order (Attempt {attempt + 1}). Might be processing or already closed.{Style.RESET_ALL}"
            )
            # Continue to next attempt, will try fallback later
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix}: API error during fetch_order (Attempt {attempt + 1}): {e}{Style.RESET_ALL}"
            )
            # Continue retrying
        except Exception as e:
            logger.error(f"{Fore.RED}{log_prefix}: Unexpected error during fetch_order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            # Break on unexpected error? Or continue retrying? Let's continue for now.

        # Wait before retrying fetch_order
        if attempt < CONFIG.fetch_order_status_retries - 1:
            time.sleep(CONFIG.fetch_order_status_delay)

    # --- Fallback Method: fetch_closed_orders (if fetch_order didn't confirm) ---
    if not confirmed_order:
        logger.debug(f"{log_prefix}: fetch_order did not confirm 'closed'. Trying fallback: fetch_closed_orders...")
        try:
            # Fetch recent closed orders
            closed_orders = exchange.fetch_closed_orders(symbol, limit=10)  # Fetch a few recent ones
            logger.debug(f"{log_prefix}: Fallback fetched recent closed orders: {[o.get('id') for o in closed_orders]}")
            for order in closed_orders:
                if order.get(ID_KEY) == order_id:
                    status = order.get(STATUS_KEY)
                    if status == ORDER_STATUS_CLOSED:
                        logger.success(f"{log_prefix}: Confirmed FILLED via fetch_closed_orders fallback.")
                        confirmed_order = order
                        break
                    else:
                        # Found the order but it wasn't closed (e.g., canceled)
                        logger.error(
                            f"{Fore.RED}{log_prefix}: Found order in fallback, but status is '{status}'. Order failed.{Style.RESET_ALL}"
                        )
                        return None
            if not confirmed_order:
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix}: Order not found in recent closed orders via fallback.{Style.RESET_ALL}"
                )

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix}: API error during fetch_closed_orders fallback: {e}{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}{log_prefix}: Unexpected error during fetch_closed_orders fallback: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())

    # --- Final Verdict ---
    if confirmed_order:
        return confirmed_order
    else:
        elapsed = time.time() - start_time
        logger.error(
            f"{Fore.RED}{log_prefix}: FAILED to confirm fill for order {order_id} using both methods within timeout ({elapsed:.1f}s).{Style.RESET_ALL}"
        )
        return None


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    quantity: float,
    required_margin: float,  # Floats received from calculate_position_size
    stop_loss_price: Decimal,
    take_profit_price: Decimal,
) -> dict[str, Any] | None:
    """Places market entry, confirms fill, then places SL/TP orders based on actual fill."""
    market_base = symbol.split("/")[0]
    log_prefix = f"Entry Order ({side.upper()})"
    order_id = None
    entry_order = None
    confirmed_order = None  # To store the confirmed filled order

    try:
        # Fetch Balance & Market Info
        logger.debug(f"{log_prefix}: Gathering resources...")
        balance_info = exchange.fetch_balance()
        free_balance = Decimal(str(balance_info.get("free", {}).get(USDT_SYMBOL, "0")))
        market = exchange.market(symbol)
        if not market:
            raise ValueError(f"Market info not found for {symbol}")
        entry_price_estimate = Decimal(str(exchange.fetch_ticker(symbol).get(LAST_PRICE_KEY, "0")))
        if entry_price_estimate <= 0:
            raise ValueError("Could not fetch valid last price for estimate.")

        # Cap Quantity based on MAX_ORDER_USDT_AMOUNT
        estimated_value = Decimal(str(quantity)) * entry_price_estimate
        if estimated_value > CONFIG.max_order_usdt_amount:
            original_quantity = quantity
            # Calculate capped quantity as Decimal first for precision
            capped_qty_decimal = CONFIG.max_order_usdt_amount / entry_price_estimate
            # Apply exchange precision using float conversion temporarily
            quantity_str = exchange.amount_to_precision(symbol, float(capped_qty_decimal))
            quantity = float(quantity_str)  # Final quantity as float for create_order
            # Recalculate estimated margin
            required_margin = float((Decimal(quantity_str) * entry_price_estimate) / Decimal(CONFIG.leverage))
            logger.warning(
                f"{Fore.YELLOW}{log_prefix}: Qty {original_quantity:.8f} (Val ~{estimated_value:.2f}) > Max {CONFIG.max_order_usdt_amount:.2f}. Capping to {quantity_str} (New Est. Margin ~{required_margin:.2f}).{Style.RESET_ALL}"
            )

        # Final Limit Checks (Amount & Cost)
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})
        min_amount = float(amount_limits.get("min", 0)) if amount_limits.get("min") is not None else 0.0
        min_cost = float(cost_limits.get("min", 0)) if cost_limits.get("min") is not None else 0.0
        if quantity < min_amount:
            logger.error(
                f"{Fore.RED}{log_prefix}: Capped qty {quantity:.8f} < MinAmt {min_amount:.8f}. Abort.{Style.RESET_ALL}"
            )
            return None
        estimated_cost_final = quantity * float(entry_price_estimate)
        if estimated_cost_final < min_cost:
            logger.error(
                f"{Fore.RED}{log_prefix}: EstCost {estimated_cost_final:.2f} < MinCost {min_cost:.2f}. Abort.{Style.RESET_ALL}"
            )
            return None

        # Margin Check
        required_margin_with_buffer = Decimal(str(required_margin)) * CONFIG.required_margin_buffer
        logger.debug(
            f"{log_prefix}: Free Balance={free_balance:.2f}, Est. Margin Required (incl. buffer)={required_margin_with_buffer:.2f}"
        )
        if free_balance < required_margin_with_buffer:
            logger.error(
                f"{Fore.RED}{log_prefix}: Insufficient free balance ({free_balance:.2f}) for margin ({required_margin_with_buffer:.2f}). Abort.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] Order REJECTED ({side.upper()}): Insuff. free balance. Need ~{required_margin_with_buffer:.2f}"
            )
            return None

        # Place Market Order
        entry_side_color = Back.GREEN if side == SIDE_BUY else Back.RED
        text_color = Fore.BLACK if side == SIDE_BUY else Fore.WHITE
        logger.warning(
            f"{entry_side_color}{text_color}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {quantity:.8f} {symbol} ***{Style.RESET_ALL}"
        )
        entry_order = exchange.create_market_order(symbol, side, quantity)
        order_id = entry_order.get(ID_KEY)
        if not order_id:
            raise ValueError("Market order placed but no ID returned.")
        logger.success(
            f"{log_prefix}: Market order submitted. ID: ...{order_id[-6:]}. Waiting for fill confirmation..."
        )
        time.sleep(CONFIG.post_entry_delay_seconds)  # Allow time for order processing

        # Confirm Order Fill (using improved function)
        confirmed_order = confirm_order_fill(exchange, order_id, symbol)
        if not confirmed_order:
            logger.error(
                f"{Fore.RED}{log_prefix}: FAILED to confirm fill for order {order_id}. Aborting SL/TP placement.{Style.RESET_ALL}"
            )
            send_sms_alert(f"[{market_base}] Entry fill confirm FAILED: {order_id}")
            # Consider emergency close here if needed
            return None

        # Extract Actual Fill Details
        try:
            actual_filled_qty = Decimal(str(confirmed_order.get(FILLED_KEY, "0")))
            actual_avg_price = Decimal(str(confirmed_order.get(AVERAGE_KEY, "0")))
            if actual_avg_price <= 0 and confirmed_order.get(PRICE_KEY) is not None:
                actual_avg_price = Decimal(str(confirmed_order.get(PRICE_KEY)))
            if actual_filled_qty <= CONFIG.position_qty_epsilon or actual_avg_price <= 0:
                raise ValueError(f"Invalid fill data: Qty={actual_filled_qty}, Price={actual_avg_price}")
            logger.success(
                f"{log_prefix}: Fill Confirmed: Order ID ...{order_id[-6:]}, Filled Qty={actual_filled_qty:.8f}, Avg Price={actual_avg_price:.4f}"
            )
        except (InvalidOperation, ValueError, TypeError, KeyError) as e:
            logger.error(
                f"{Fore.RED}{log_prefix}: Error parsing confirmed fill details for order {order_id}: {e}. Data: {confirmed_order}{Style.RESET_ALL}"
            )
            send_sms_alert(f"[{market_base}] ERROR parsing fill data: {order_id}")
            return None  # Cannot proceed

        # Place SL/TP Orders
        logger.info(
            f"{log_prefix}: Placing SL ({stop_loss_price}) and TP ({take_profit_price}) orders for filled qty {actual_filled_qty:.8f}..."
        )
        sl_tp_success = True
        close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
        sl_tp_amount_float = float(actual_filled_qty)  # Use actual filled quantity

        # SL Order Params
        sl_trigger_direction = 2 if side == SIDE_BUY else 1
        sl_params = {
            PARAM_STOP_PRICE: exchange.price_to_precision(symbol, float(stop_loss_price)),
            PARAM_REDUCE_ONLY: True,
            PARAM_TRIGGER_DIRECTION: sl_trigger_direction,
        }
        # TP Order Params (Using stopMarket type)
        tp_trigger_direction = 1 if side == SIDE_BUY else 2
        tp_params = {
            PARAM_STOP_PRICE: exchange.price_to_precision(symbol, float(take_profit_price)),
            PARAM_REDUCE_ONLY: True,
            PARAM_TRIGGER_DIRECTION: tp_trigger_direction,
        }

        sl_order_id_short, tp_order_id_short = "N/A", "N/A"
        try:
            logger.debug(f"Placing SL order: side={close_side}, amount={sl_tp_amount_float}, params={sl_params}")
            sl_order = exchange.create_order(
                symbol, ORDER_TYPE_STOP_MARKET, close_side, sl_tp_amount_float, params=sl_params
            )
            sl_order_id_short = str(sl_order.get(ID_KEY, "N/A"))[-6:]
            logger.success(
                f"{Fore.GREEN}{log_prefix}: Stop-Loss order placed. ID: ...{sl_order_id_short}{Style.RESET_ALL}"
            )
            time.sleep(0.1)  # Small delay
        except Exception as e:
            sl_tp_success = False
            logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix}: FAILED to place Stop-Loss order: {e}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL: SL order FAILED for {symbol} ({side}): {e}")

        try:
            logger.debug(f"Placing TP order: side={close_side}, amount={sl_tp_amount_float}, params={tp_params}")
            # Use stopMarket for TP as well
            tp_order = exchange.create_order(
                symbol, ORDER_TYPE_STOP_MARKET, close_side, sl_tp_amount_float, params=tp_params
            )
            tp_order_id_short = str(tp_order.get(ID_KEY, "N/A"))[-6:]
            logger.success(
                f"{Fore.GREEN}{log_prefix}: Take-Profit order placed. ID: ...{tp_order_id_short}{Style.RESET_ALL}"
            )
        except Exception as e:
            # TP failure is less critical than SL, but still log as error
            sl_tp_success = False  # Mark overall success as false if TP fails
            logger.error(
                f"{Back.YELLOW}{Fore.BLACK}{log_prefix}: FAILED to place Take-Profit order: {e}{Style.RESET_ALL}"
            )
            send_sms_alert(f"[{market_base}] WARNING: TP order FAILED for {symbol} ({side}): {e}")

        if sl_tp_success:
            logger.info(
                f"{Fore.GREEN}{log_prefix}: Entry order filled and SL/TP orders placed successfully.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] Entered {side.upper()} {actual_filled_qty:.4f} @ {actual_avg_price:.2f}. SL=...{sl_order_id_short}, TP=...{tp_order_id_short}"
            )
            return confirmed_order
        else:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{log_prefix}: Entry filled, but FAILED to place one/both SL/TP orders. MANUAL CHECK REQUIRED!{Style.RESET_ALL}"
            )
            return None  # Indicate failure if SL/TP placement wasn't fully successful

    except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError) as e:
        logger.error(f"{Fore.RED}{log_prefix}: Failed during order placement/check: {e}{Style.RESET_ALL}")
        if entry_order:
            logger.error(f"Entry order details (if placed): {entry_order}")
        send_sms_alert(f"[{market_base}] Entry order FAILED ({side.upper()}): {type(e).__name__}")
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix}: Unexpected error during order ritual: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] UNEXPECTED error during entry ({side.upper()}): {type(e).__name__}")
        return None


# --- Core Trading Logic - The Spell Weaving Cycle ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> None:
    """Main trading logic loop."""
    cycle_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== New Weaving Cycle: {symbol} ({timeframe}) | {cycle_time_str} =========={Style.RESET_ALL}"
    )

    # --- 1. Get Data ---
    required_ohlcv_len = (
        max(CONFIG.st_atr_length, CONFIG.confirm_st_atr_length, CONFIG.volume_ma_period, CONFIG.atr_calculation_period)
        + CONFIG.api_fetch_limit_buffer
    )
    df = fetch_ohlcv(exchange, symbol, timeframe, limit=required_ohlcv_len)
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - unable to fetch valid OHLCV data.{Style.RESET_ALL}")
        return

    order_book_data = None
    # Fetch OB if always required OR if confirmation is enabled (fetch only when needed later)
    if CONFIG.fetch_order_book_per_cycle:
        order_book_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
        if order_book_data is None and CONFIG.use_ob_confirm:  # If mandatory and failed
            logger.warning(
                f"{Fore.YELLOW}Trade Logic: Skipping cycle - failed to get required OB data.{Style.RESET_ALL}"
            )
            return

    # --- 2. Calculate Indicators ---
    logger.debug("Calculating indicators...")
    df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
    df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
    vol_atr_results = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)

    # Check for critical indicator failures
    if "supertrend" not in df.columns or df["supertrend"].isnull().all():
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - primary Supertrend calc failed.{Style.RESET_ALL}")
        return
    if "confirm_supertrend" not in df.columns or df["confirm_supertrend"].isnull().all():
        logger.warning(
            f"{Fore.YELLOW}Trade Logic: Skipping cycle - confirmation Supertrend calc failed.{Style.RESET_ALL}"
        )
        return
    if vol_atr_results is None or vol_atr_results.get(ATR_KEY) is None:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - Volume/ATR calc failed.{Style.RESET_ALL}")
        return

    # --- 3. Extract Latest Indicator Values ---
    try:
        last_candle = df.iloc[-1]
        # Primary Supertrend
        st_trend = last_candle["trend"]  # Current trend direction (-1 or 1)
        st_long_signal = last_candle["st_long"]  # True only on flip candle
        st_short_signal = last_candle["st_short"]  # True only on flip candle
        # Confirmation Supertrend
        confirm_st_trend = last_candle["confirm_trend"]  # Current trend direction (-1 or 1)
        # Volume/ATR
        current_atr = vol_atr_results.get(ATR_KEY)  # Decimal or None
        volume_ratio = vol_atr_results.get(VOLUME_RATIO_KEY)  # Decimal or None
        # Current Price
        current_price = Decimal(str(last_candle["close"]))

        if current_atr is None or pd.isna(st_trend) or pd.isna(confirm_st_trend) or current_price <= 0:
            raise ValueError("Essential indicator values (ATR, ST trends, Price) are None/NaN/invalid.")

    except (IndexError, KeyError, ValueError, InvalidOperation) as e:
        logger.error(f"{Fore.RED}Trade Logic: Error accessing indicator/price data: {e}{Style.RESET_ALL}")
        logger.debug(f"DataFrame tail:\n{df.tail()}")
        return

    # --- 4. Check Current Position ---
    current_position = get_current_position(exchange, symbol)
    position_side = current_position[SIDE_KEY]
    position_qty = current_position[QTY_KEY]
    position_entry_price = current_position[ENTRY_PRICE_KEY]
    pos_color = (
        Fore.GREEN
        if position_side == POSITION_SIDE_LONG
        else (Fore.RED if position_side == POSITION_SIDE_SHORT else Fore.BLUE)
    )
    logger.info(
        f"State | Position: Side={pos_color}{position_side}{Style.RESET_ALL}, Qty={position_qty:.8f}, Entry={position_entry_price:.4f}"
    )
    logger.info(
        f"State | Indicators: Price={current_price:.4f}, ATR={current_atr:.4f}, ST Trend={st_trend}, Confirm ST Trend={confirm_st_trend}, VolRatio={volume_ratio if volume_ratio else 'N/A'}"
    )

    # --- 5. Determine Signals ---
    long_entry_signal = st_long_signal and confirm_st_trend == 1
    short_entry_signal = st_short_signal and confirm_st_trend == -1
    close_long_signal = position_side == POSITION_SIDE_LONG and st_trend == -1
    close_short_signal = position_side == POSITION_SIDE_SHORT and st_trend == 1
    logger.debug(
        f"Signals: EntryLong={long_entry_signal}, EntryShort={short_entry_signal}, CloseLong={close_long_signal}, CloseShort={close_short_signal}"
    )

    # --- 6. Decision Making ---

    # **Exit Logic:** Prioritize closing
    exit_reason = None
    if close_long_signal:
        exit_reason = "ST Exit Long"
    elif close_short_signal:
        exit_reason = "ST Exit Short"

    if exit_reason:
        exit_side_color = Back.YELLOW
        logger.warning(
            f"{exit_side_color}{Fore.BLACK}{Style.BRIGHT}*** TRADE EXIT SIGNAL: Closing {position_side} due to {exit_reason} ***{Style.RESET_ALL}"
        )
        close_position(exchange, symbol, current_position, reason=exit_reason)
        time.sleep(CONFIG.post_close_delay_seconds)  # Pause after closing
        return  # End cycle

    # **Entry Logic:** Only if flat
    if position_side == POSITION_SIDE_NONE:
        selected_side = None
        if long_entry_signal:
            selected_side = SIDE_BUY
        elif short_entry_signal:
            selected_side = SIDE_SELL

        if selected_side:
            logger.info(
                f"{Fore.CYAN}Entry Signal: Potential {selected_side.upper()} entry detected by Supertrend.{Style.RESET_ALL}"
            )

            # Volume Confirmation
            volume_confirmed = True
            if CONFIG.require_volume_spike_for_entry:
                if volume_ratio is None or volume_ratio <= CONFIG.volume_spike_threshold:
                    volume_confirmed = False
                    logger.info(
                        f"Entry REJECTED ({selected_side.upper()}): Volume spike confirmation FAILED (Ratio: {volume_ratio if volume_ratio else 'N/A'} <= Threshold: {CONFIG.volume_spike_threshold})."
                    )
                else:
                    logger.info(
                        f"{Fore.GREEN}Entry Check ({selected_side.upper()}): Volume spike OK (Ratio: {volume_ratio:.2f}).{Style.RESET_ALL}"
                    )

            # Order Book Confirmation
            ob_confirmed = True
            if volume_confirmed and CONFIG.use_ob_confirm:
                if order_book_data is None:  # Fetch only if needed now
                    logger.debug("Fetching OB data for confirmation...")
                    order_book_data = analyze_order_book(
                        exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
                    )

                if order_book_data is None or order_book_data.get(BID_ASK_RATIO_KEY) is None:
                    ob_confirmed = False
                    logger.warning(
                        f"{Fore.YELLOW}Entry SKIPPED ({selected_side.upper()}): OB confirmation FAILED (Could not get valid OB data/ratio).{Style.RESET_ALL}"
                    )
                else:
                    ob_ratio = order_book_data[BID_ASK_RATIO_KEY]
                    ob_ratio_str = f"{ob_ratio:.3f}"
                    if selected_side == SIDE_BUY and ob_ratio < CONFIG.order_book_ratio_threshold_long:
                        ob_confirmed = False
                        logger.info(
                            f"Entry REJECTED ({selected_side.upper()}): OB confirmation FAILED (Ratio: {ob_ratio_str} < Threshold: {CONFIG.order_book_ratio_threshold_long})."
                        )
                    elif selected_side == SIDE_SELL and ob_ratio > CONFIG.order_book_ratio_threshold_short:
                        ob_confirmed = False
                        logger.info(
                            f"Entry REJECTED ({selected_side.upper()}): OB confirmation FAILED (Ratio: {ob_ratio_str} > Threshold: {CONFIG.order_book_ratio_threshold_short})."
                        )
                    else:
                        ob_color = Fore.GREEN if selected_side == SIDE_BUY else Fore.RED
                        logger.info(
                            f"{ob_color}Entry Check ({selected_side.upper()}): OB pressure OK (Ratio: {ob_ratio_str}).{Style.RESET_ALL}"
                        )

            # Proceed if all confirmations pass
            if volume_confirmed and ob_confirmed:
                logger.success(
                    f"{Fore.GREEN}{Style.BRIGHT}Entry CONFIRMED ({selected_side.upper()}): All checks passed. Calculating parameters...{Style.RESET_ALL}"
                )

                # Calculate SL/TP Prices
                try:
                    price_precision_str = exchange.markets[symbol]["precision"]["price"]
                    price_precision = (
                        Decimal(price_precision_str) if price_precision_str else Decimal("0.0001")
                    )  # Fallback precision
                    sl_distance = current_atr * CONFIG.atr_stop_loss_multiplier
                    tp_distance = current_atr * CONFIG.atr_take_profit_multiplier
                    entry_price_est = current_price  # Use last close price as estimate

                    if selected_side == SIDE_BUY:
                        sl_price = entry_price_est - sl_distance
                        tp_price = entry_price_est + tp_distance
                    else:  # SIDE_SELL
                        sl_price = entry_price_est + sl_distance
                        tp_price = entry_price_est - tp_distance

                    if sl_price <= 0 or tp_price <= 0:
                        raise ValueError("SL/TP price zero or negative.")

                    # Quantize SL/TP using market precision
                    sl_price = sl_price.quantize(price_precision, rounding=ROUND_HALF_UP)
                    tp_price = tp_price.quantize(price_precision, rounding=ROUND_HALF_UP)
                    logger.info(
                        f"Calculated SL={sl_price}, TP={tp_price} based on EntryEst={entry_price_est}, ATR={current_atr:.4f}"
                    )

                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Error calculating SL/TP prices: {e}{Style.RESET_ALL}"
                    )
                    return

                # Calculate Position Size
                try:
                    equity = Decimal(str(exchange.fetch_balance().get("total", {}).get(USDT_SYMBOL, "0")))
                    if equity <= 0:
                        raise ValueError("Zero or negative equity.")
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Failed to fetch valid equity: {e}{Style.RESET_ALL}"
                    )
                    return

                quantity_float, margin_est_float = calculate_position_size(
                    equity,
                    CONFIG.risk_per_trade_percentage,
                    entry_price_est,
                    sl_price,
                    CONFIG.leverage,
                    symbol,
                    exchange,
                )

                if quantity_float is not None and margin_est_float is not None:
                    # Place the order
                    place_risked_market_order(
                        exchange, symbol, selected_side, quantity_float, margin_est_float, sl_price, tp_price
                    )
                    return  # End cycle after attempting entry
                else:
                    logger.error(
                        f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Failed to calculate valid position size or margin.{Style.RESET_ALL}"
                    )

    elif position_side != POSITION_SIDE_NONE:
        logger.info(
            f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. No exit signal. Awaiting SL/TP or next signal."
        )
        # Optional: Implement redundant SL/TP monitoring here if needed
        # if CONFIG.enable_monitor_sltp: monitor_and_close_if_needed(...)

    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n")


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    """Main function to run the bot."""
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v10.1.0 Initializing ({start_time_str}) ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}--- Strategy Enchantment: Dual Supertrend ---{Style.RESET_ALL}"
    )  # Strategy is hardcoded in this version
    logger.info(f"{Fore.GREEN}--- Protective Wards: Exchange Native SL/TP ---{Style.RESET_ALL}")

    # Config object already instantiated and validated globally
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK - HANDLE WITH CARE !!! ---{Style.RESET_ALL}"
    )

    # Initialize Exchange
    exchange = initialize_exchange()
    if not exchange:
        logger.critical("Failed to initialize exchange. Spell fizzles.")
        sys.exit(1)

    # Set Leverage
    if not set_leverage(exchange, CONFIG.symbol, CONFIG.leverage):
        logger.critical(f"Failed to set leverage to {CONFIG.leverage}x for {CONFIG.symbol}. Spell cannot bind.")
        sys.exit(1)

    # Log Final Config Summary (using CONFIG object)
    logger.info(f"{Fore.MAGENTA}--- Final Spell Configuration ---{Style.RESET_ALL}")
    logger.info(f"{Fore.WHITE}Symbol: {CONFIG.symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
    logger.info(
        f"  Supertrend Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}"
    )
    logger.info(
        f"{Fore.GREEN}Risk Ward: {CONFIG.risk_per_trade_percentage:.3%}/trade, Max Pos Value: {CONFIG.max_order_usdt_amount:.4f} USDT"
    )
    logger.info(
        f"{Fore.GREEN}SL/TP Wards: SL Mult={CONFIG.atr_stop_loss_multiplier}, TP Mult={CONFIG.atr_take_profit_multiplier} (ATR Period: {CONFIG.atr_calculation_period})"
    )
    logger.info(
        f"{Fore.YELLOW}Volume Filter: {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})"
    )
    logger.info(
        f"{Fore.YELLOW}Order Book Filter: Use Confirm={CONFIG.use_ob_confirm}, Fetch Each Cycle={CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})"
    )
    logger.info(
        f"{Fore.WHITE}Timing: Sleep={CONFIG.sleep_seconds}s, Margin Buffer={CONFIG.required_margin_buffer:.1%}, SMS Alerts={CONFIG.enable_sms_alerts}"
    )
    logger.info(f"{Fore.CYAN}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}")
    logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")

    # --- Main Loop ---
    run_bot = True
    cycle_count = 0
    while run_bot:
        cycle_count += 1
        logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Weaving Start ---{Style.RESET_ALL}")
        try:
            trade_logic(exchange, CONFIG.symbol, CONFIG.interval)  # Pass necessary args
            logger.debug(f"Cycle {cycle_count} complete. Sleeping for {CONFIG.sleep_seconds} seconds...")
            time.sleep(CONFIG.sleep_seconds)

        except KeyboardInterrupt:
            logger.warning(
                f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt detected. Requesting graceful withdrawal...{Style.RESET_ALL}"
            )
            send_sms_alert(f"[ScalpBot] Shutdown initiated for {CONFIG.symbol} (KeyboardInterrupt).")
            run_bot = False  # Signal loop termination

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Authentication Error during main loop: {e}. API keys invalid/revoked? Shutting down NOW.{Style.RESET_ALL}"
            )
            send_sms_alert("[ScalpBot] CRITICAL: Auth Error - SHUTDOWN. Check Keys/Permissions.")
            run_bot = False
        except ccxt.NetworkError as e:
            logger.error(f"{Fore.RED}ERROR: Network error in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
            time.sleep(CONFIG.sleep_seconds * 2)  # Longer delay
        except ccxt.ExchangeError as e:
            logger.error(f"{Fore.RED}ERROR: Exchange error in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            time.sleep(CONFIG.sleep_seconds)
        except Exception as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}FATAL: An unexpected error occurred in the main loop: {e}{Style.RESET_ALL}"
            )
            logger.error(traceback.format_exc())
            send_sms_alert(f"[ScalpBot] FATAL ERROR: {type(e).__name__}. Bot stopped. Check logs!")
            run_bot = False  # Stop on fatal errors

    # --- Graceful Shutdown ---
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Initiating graceful shutdown sequence...{Style.RESET_ALL}")
    try:
        logger.info("Checking for open position to close on exit...")
        # Ensure exchange object is valid before using
        if exchange:
            current_pos = get_current_position(exchange, CONFIG.symbol)
            if current_pos[SIDE_KEY] != POSITION_SIDE_NONE:
                logger.warning(f"Attempting to close {current_pos[SIDE_KEY]} position before exiting...")
                close_position(exchange, symbol=CONFIG.symbol, position_to_close=current_pos, reason="Shutdown")
            else:
                logger.info("No open position found to close.")
        else:
            logger.warning("Exchange object not available for final position check.")
    except Exception as close_err:
        logger.error(f"{Fore.RED}Failed to check/close position during final shutdown: {close_err}{Style.RESET_ALL}")
        send_sms_alert("[ScalpBot] Error during final position close check on shutdown.")

    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
