#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██║   ██║███████╗
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
from typing import Any, Dict, List, Optional, Tuple, Union, Type

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
    print(f"\033[91m\033[1mCRITICAL ERROR: Missing required Python package: '{missing_pkg}'.\033[0m")
    print(f"\033[93mPlease install it using: pip install {missing_pkg}\033[0m")
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)  # Activate Colorama's magic
load_dotenv()  # Load secrets from the hidden .env scroll
# Set Decimal precision high enough for crypto calculations
# Bybit USDT perps typically have price precision up to 4-6 decimals,
# and quantity precision up to 3-8 decimals. 18 should be safe.
getcontext().prec = 18

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
ORDER_STATUS_OPEN = "open"
ORDER_STATUS_CLOSED = "closed"
ORDER_STATUS_CANCELED = "canceled"  # Note: CCXT might use 'cancelled' or 'canceled'
ORDER_STATUS_REJECTED = "rejected"
ORDER_STATUS_EXPIRED = "expired"
PARAM_REDUCE_ONLY = "reduce_only"  # CCXT standard param name
PARAM_STOP_PRICE = "stopPrice"  # CCXT standard param name for trigger price
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


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a 'success' log level method."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Bind the new method to the Logger class
logging.Logger.success = log_success  # type: ignore[attr-defined]

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
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", None, str, required=True, color=Fore.RED)
        self.api_secret: Optional[str] = self._get_env("BYBIT_API_SECRET", None, str, required=True, color=Fore.RED)
        if not self.api_key or not self.api_secret:
            valid = False

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", str, color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", str, color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, int, color=Fore.YELLOW)
        if self.leverage <= 0:
            logger.critical(f"CRITICAL CONFIG: LEVERAGE must be positive, got: {self.leverage}")
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
            logger.critical(
                f"CRITICAL CONFIG: RISK_PER_TRADE_PERCENTAGE must be between 0 and 1 (exclusive), got: {self.risk_per_trade_percentage}"
            )
            valid = False
        if self.atr_stop_loss_multiplier <= 0:
            logger.warning(
                f"CONFIG WARNING: ATR_STOP_LOSS_MULTIPLIER ({self.atr_stop_loss_multiplier}) should be positive."
            )
        if self.atr_take_profit_multiplier <= 0:
            logger.warning(
                f"CONFIG WARNING: ATR_TAKE_PROFIT_MULTIPLIER ({self.atr_take_profit_multiplier}) should be positive."
            )
        if self.max_order_usdt_amount <= 0:
            logger.warning(f"CONFIG WARNING: MAX_ORDER_USDT_AMOUNT ({self.max_order_usdt_amount}) should be positive.")
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
        if self.st_multiplier <= 0 or self.confirm_st_multiplier <= 0:
            logger.warning("CONFIG WARNING: Supertrend multiplier(s) are zero or negative.")

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
        if self.volume_spike_threshold <= 0:
            logger.warning("CONFIG WARNING: VOLUME_SPIKE_THRESHOLD should be positive.")

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
        if self.order_book_depth <= 0:
            logger.warning("CONFIG WARNING: ORDER_BOOK_DEPTH should be positive.")

        # --- ATR Calculation Parameter (for SL/TP) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, int, color=Fore.GREEN)
        if self.atr_calculation_period <= 0:
            logger.warning("CONFIG WARNING: ATR_CALCULATION_PERIOD is zero or negative.")

        # --- Termux SMS Alert Configuration ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", False, bool, color=Fore.MAGENTA)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, str, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, int, color=Fore.MAGENTA)
        if self.enable_sms_alerts and not self.sms_recipient_number:
            logger.warning("CONFIG WARNING: SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")
        if self.sms_timeout_seconds <= 0:
            logger.warning(f"CONFIG WARNING: SMS_TIMEOUT_SECONDS ({self.sms_timeout_seconds}) invalid. Setting to 10.")
            self.sms_timeout_seconds = 10

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = self._get_env("RECV_WINDOW", 10000, int, color=Fore.WHITE)
        # Bybit V5 L2 OB limit can be 1, 50, 200. Fetch 50 if depth <= 50, else 200.
        self.order_book_fetch_limit: int = 50 if self.order_book_depth <= 50 else 200
        self.shallow_ob_fetch_depth: int = self._get_env(
            "SHALLOW_OB_FETCH_DEPTH", 5, int, color=Fore.WHITE
        )  # Not currently used, but kept for potential future use

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
        self, var_name: str, default: Any, expected_type: Type, required: bool = False, color: str = Fore.WHITE
    ) -> Any:
        """Gets an environment variable, casts type (incl. defaults), logs, handles errors.
        Handles str, int, float, bool, and Decimal types.
        """
        value_str = os.getenv(var_name)
        source = "environment" if value_str is not None else "default"
        value_to_process: Any = value_str if value_str is not None else default

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
                # Explicitly check for string representations of True/False
                if isinstance(value_to_process, str):
                    return value_to_process.lower() in ("true", "1", "t", "yes", "y")
                # If not a string, try standard Python truthiness
                return bool(value_to_process)
            elif expected_type == Decimal:
                # Ensure input is string for Decimal constructor for reliable conversion
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
                    if isinstance(default, str):
                        return default.lower() in ("true", "1", "t", "yes", "y")
                    return bool(default)
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
except ValueError as config_err:
    # Error already logged within Config init or _get_env
    # Attempt to send SMS even if config partially failed (if SMS settings were read)
    # Check if CONFIG object exists and has necessary attributes before trying to send SMS
    config_exists = (
        "CONFIG" in globals() and hasattr(CONFIG, "enable_sms_alerts") and hasattr(CONFIG, "sms_recipient_number")
    )
    if config_exists and CONFIG.enable_sms_alerts and CONFIG.sms_recipient_number:
        # Import send_sms_alert here or define it earlier if needed for this edge case
        # For simplicity, assume send_sms_alert is defined before this point
        try:
            # Define send_sms_alert minimally if it's not available yet
            if "send_sms_alert" not in globals():

                def send_sms_alert(message: str) -> bool:
                    logger.warning(f"Attempted emergency SMS, but function not fully loaded: {message}")
                    return False

            send_sms_alert(f"[ScalpBot] CRITICAL: Config validation FAILED: {config_err}. Bot stopped.")
        except Exception as sms_err:  # Catch errors during the emergency SMS itself
            logger.error(f"Failed to send critical config failure SMS: {sms_err}")
    sys.exit(1)

# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: Optional[bool] = None  # Cache check result


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
        command: List[str] = ["termux-sms-send", "-n", CONFIG.sms_recipient_number, message]
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
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """Initializes and returns the CCXT Bybit exchange instance, opening a portal."""
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    # API keys already checked in Config, but double-check instance variables
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API keys check failed during initialization.{Style.RESET_ALL}"
        )
        return None

    try:
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,  # Built-in rate limiting
                "options": {
                    "adjustForTimeDifference": True,  # Adjust for clock skew
                    "recvWindow": CONFIG.default_recv_window,  # Increase if timestamp errors occur
                    "defaultType": "swap",  # Explicitly default to swap markets (linear/inverse determined by symbol)
                    "warnOnFetchOpenOrdersWithoutSymbol": False,  # Suppress common warning
                    "brokerId": "Pyrmethus_Scalp_v10.1",  # Optional: Identify the bot
                    "defaultMarginMode": "isolated",  # Explicitly set default margin mode if desired, though leverage setting might override
                    "createMarketBuyOrderRequiresPrice": False,  # Bybit V5 doesn't require price for market buy
                },
            }
        )
        # Explicitly set API version to v5 if CCXT doesn't default correctly (usually not needed)
        # exchange.set_sandbox_mode(False) # Ensure not in sandbox unless intended

    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to instantiate CCXT Bybit object: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: CCXT Instantiation Error: {type(e).__name__}. Bot stopped.")
        return None

    try:
        # Test connection and authentication by fetching markets and balance
        logger.debug("Loading market structures...")
        exchange.load_markets()
        logger.debug("Fetching balance (tests authentication)...")
        # Specify params for V5 balance fetch if needed (e.g., account type)
        # For USDT perpetual, accountType='CONTRACT' is typical for V5 unified margin,
        # but CCXT might handle this based on the market type. Let's try without first.
        balance = exchange.fetch_balance()  # params={'accountType': 'CONTRACT'} might be needed
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
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: float, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using the pandas_ta library."""
    required_input_cols = ["high", "low", "close"]
    col_prefix = f"{prefix}" if prefix else ""
    # Define target column names clearly
    target_st_val_col = f"{col_prefix}st_value"
    target_st_trend_col = f"{col_prefix}trend"
    target_st_long_flip_col = f"{col_prefix}st_long_flip"
    target_st_short_flip_col = f"{col_prefix}st_short_flip"
    target_cols = [target_st_val_col, target_st_trend_col, target_st_long_flip_col, target_st_short_flip_col]

    # Define expected pandas_ta column names (adjust if pandas_ta version changes output)
    pta_st_col_name = f"SUPERT_{length}_{multiplier}"
    pta_st_trend_col = f"SUPERTd_{length}_{multiplier}"
    pta_st_long_col = f"SUPERTl_{length}_{multiplier}"  # pandas_ta uses 'l' for long band
    pta_st_short_col = f"SUPERTs_{length}_{multiplier}"  # pandas_ta uses 's' for short band
    expected_pta_cols = [pta_st_col_name, pta_st_trend_col, pta_st_long_col, pta_st_short_col]

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols):
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}Supertrend): Input DataFrame is missing required columns {required_input_cols} or is empty.{Style.RESET_ALL}"
        )
        # Ensure target columns exist with NA if input is invalid but not None
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame()

    if len(df) < length:
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}Supertrend): DataFrame length ({len(df)}) is less than ST period ({length}). Filling with NA.{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df

    try:
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={multiplier}")
        # Ensure input columns are numeric
        for col in required_input_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[required_input_cols].isnull().values.any():
            logger.warning(
                f"{Fore.YELLOW}Scrying ({col_prefix}ST): NaNs found in input data before calculation. Results may be affected.{Style.RESET_ALL}"
            )
            # Optionally fill NaNs here if appropriate (e.g., df.ffill(inplace=True))

        # Calculate Supertrend using pandas_ta
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)

        # Check if pandas_ta created the expected columns
        missing_pta_cols = [col for col in expected_pta_cols if col not in df.columns]
        if missing_pta_cols:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {', '.join(missing_pta_cols)}")

        # Convert potentially generated columns to numeric, coercing errors
        for col in expected_pta_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rename and process
        # Check if target columns already exist before renaming (e.g., from previous runs)
        if target_st_val_col in df.columns:
            df.drop(columns=[target_st_val_col], inplace=True)
        if target_st_trend_col in df.columns:
            df.drop(columns=[target_st_trend_col], inplace=True)
        df.rename(columns={pta_st_col_name: target_st_val_col}, inplace=True)  # Supertrend value
        df.rename(columns={pta_st_trend_col: target_st_trend_col}, inplace=True)  # Trend direction (-1, 1)

        # Calculate flip signals based on trend change
        prev_trend_direction = df[target_st_trend_col].shift(1)
        # Long flip: Previous was Down (-1) and Current is Up (1)
        df[target_st_long_flip_col] = (prev_trend_direction == -1) & (df[target_st_trend_col] == 1)
        # Short flip: Previous was Up (1) and Current is Down (-1)
        df[target_st_short_flip_col] = (prev_trend_direction == 1) & (df[target_st_trend_col] == -1)
        # Ensure boolean type and fill NA (especially first row) with False
        df[[target_st_long_flip_col, target_st_short_flip_col]] = (
            df[[target_st_long_flip_col, target_st_short_flip_col]].fillna(False).astype(bool)
        )

        # Clean up intermediate columns generated by pandas_ta
        # Drop the original long/short band columns and any other potential intermediates
        cols_to_drop = [pta_st_long_col, pta_st_short_col] + [
            c for c in df.columns if c.startswith("SUPERT_") and c not in target_cols
        ]
        # Ensure we don't try to drop columns that weren't created or already dropped
        cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
        if cols_to_drop_existing:
            df.drop(columns=list(set(cols_to_drop_existing)), errors="ignore", inplace=True)

        # Log last candle result
        if not df.empty:
            last_trend_val = df[target_st_trend_col].iloc[-1] if pd.notna(df[target_st_trend_col].iloc[-1]) else None
            last_st_val = df[target_st_val_col].iloc[-1] if pd.notna(df[target_st_val_col].iloc[-1]) else float("nan")
            last_trend_str = "Up" if last_trend_val == 1 else "Down" if last_trend_val == -1 else "N/A"
            trend_color = Fore.GREEN if last_trend_str == "Up" else Fore.RED if last_trend_str == "Down" else Fore.WHITE
            logger.debug(
                f"Scrying ({col_prefix}ST({length}, {multiplier})): Last Trend={trend_color}{last_trend_str}{Style.RESET_ALL}, Last Value={last_st_val:.4f}"
            )
        else:
            logger.debug(f"Scrying ({col_prefix}ST): DataFrame became empty during processing.")

    except (KeyError, AttributeError, Exception) as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}Supertrend): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # Ensure target columns exist with NA on error
        for col in target_cols:
            df[col] = pd.NA
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> Dict[str, Optional[Decimal]]:
    """Calculates ATR, Volume MA, and checks for volume spikes."""
    results: Dict[str, Optional[Decimal]] = {
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
    min_len = max(atr_len, vol_ma_len, 1)  # Need at least 1 row for volume, more for indicators
    if len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): DataFrame length ({len(df)}) < required ({min_len}) for ATR({atr_len})/VolMA({vol_ma_len}).{Style.RESET_ALL}"
        )
        return results

    try:
        # Ensure numeric types, coercing errors to NaN
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[required_cols].isnull().values.any():
            logger.warning(
                f"{Fore.YELLOW}Scrying (Vol/ATR): NaNs found in input data after coercion. Results may be inaccurate.{Style.RESET_ALL}"
            )

        # Calculate ATR using pandas_ta
        atr_col = f"ATRr_{atr_len}"  # Default ATR column name from pandas_ta
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]):
            try:
                # Convert to string first for Decimal robustness
                results[ATR_KEY] = Decimal(str(df[atr_col].iloc[-1]))
            except (InvalidOperation, ValueError, TypeError) as e:
                logger.warning(f"Scrying (ATR): Invalid Decimal value for ATR: {df[atr_col].iloc[-1]}. Error: {e}")
        else:
            logger.warning(
                f"Scrying: Failed to calculate valid ATR({atr_len}). Column '{atr_col}' missing or last value is NaN."
            )
        # Clean up ATR column if it exists
        if atr_col in df.columns:
            df.drop(columns=[atr_col], errors="ignore", inplace=True)

        # Calculate Volume MA
        volume_ma_col = f"volume_ma_{vol_ma_len}"
        # Use min_periods=1 to get a value even if window isn't full, but be aware of implications
        df[volume_ma_col] = df["volume"].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()

        if pd.notna(df[volume_ma_col].iloc[-1]) and pd.notna(df["volume"].iloc[-1]):
            try:
                # Convert to string first for Decimal robustness
                results[VOLUME_MA_KEY] = Decimal(str(df[volume_ma_col].iloc[-1]))
                results[LAST_VOLUME_KEY] = Decimal(str(df["volume"].iloc[-1]))
            except (InvalidOperation, ValueError, TypeError) as e:
                logger.warning(
                    f"Scrying (Vol): Invalid Decimal value for Volume/MA. Vol: {df['volume'].iloc[-1]}, MA: {df[volume_ma_col].iloc[-1]}. Error: {e}"
                )

            # Calculate Volume Ratio
            # Ensure both values are valid Decimals and MA is not effectively zero
            if (
                results[VOLUME_MA_KEY] is not None
                and results[VOLUME_MA_KEY] > CONFIG.position_qty_epsilon
                and results[LAST_VOLUME_KEY] is not None
            ):
                try:
                    results[VOLUME_RATIO_KEY] = (results[LAST_VOLUME_KEY] / results[VOLUME_MA_KEY]).quantize(
                        Decimal("0.01")
                    )
                except (InvalidOperation, ZeroDivisionError) as e:  # Catch division by zero explicitly
                    logger.warning(
                        f"Scrying (Vol): Invalid Decimal operation for ratio. LastVol={results[LAST_VOLUME_KEY]}, VolMA={results[VOLUME_MA_KEY]}. Error: {e}"
                    )
                    results[VOLUME_RATIO_KEY] = None
            else:
                results[VOLUME_RATIO_KEY] = None
                logger.debug(
                    f"Scrying (Vol): Ratio calc skipped (LastVol={results.get(LAST_VOLUME_KEY)}, MA={results.get(VOLUME_MA_KEY)})"
                )
        else:
            logger.warning(
                f"Scrying (Vol): Failed calc VolMA({vol_ma_len}) or get last vol. LastVol: {df['volume'].iloc[-1]}, LastMA: {df[volume_ma_col].iloc[-1]}"
            )
        # Clean up volume MA column if it exists
        if volume_ma_col in df.columns:
            df.drop(columns=[volume_ma_col], errors="ignore", inplace=True)

        # Log results
        atr_str = f"{results[ATR_KEY]:.4f}" if results[ATR_KEY] is not None else "N/A"
        last_vol_val = results.get(LAST_VOLUME_KEY)
        vol_ma_val = results.get(VOLUME_MA_KEY)
        vol_ratio_val = results.get(VOLUME_RATIO_KEY)
        last_vol_str = f"{last_vol_val:.2f}" if last_vol_val is not None else "N/A"
        vol_ma_str = f"{vol_ma_val:.2f}" if vol_ma_val is not None else "N/A"
        vol_ratio_str = f"{vol_ratio_val:.2f}" if vol_ratio_val is not None else "N/A"

        logger.debug(f"Scrying Results: ATR({atr_len}) = {Fore.CYAN}{atr_str}{Style.RESET_ALL}")
        logger.debug(
            f"Scrying Results: Volume: Last={last_vol_str}, MA({vol_ma_len})={vol_ma_str}, Ratio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}"
        )

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results}  # Reset results on error
    return results


def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> Dict[str, Optional[Decimal]]:
    """Fetches L2 order book and analyzes bid/ask pressure and spread."""
    results: Dict[str, Optional[Decimal]] = {
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

        bids: List[List[Union[float, str]]] = order_book[BIDS_KEY]
        asks: List[List[Union[float, str]]] = order_book[ASKS_KEY]

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}Order Book Scrying: Bids or asks list is empty for {symbol}. Bids: {len(bids)}, Asks: {len(asks)}{Style.RESET_ALL}"
            )
            return results

        # Get best bid/ask and calculate spread
        try:
            # Ensure there's data at index 0 and the inner list has at least one element (price)
            if len(bids) > 0 and len(bids[0]) > 0 and len(asks) > 0 and len(asks[0]) > 0:
                best_bid_raw = bids[0][0]
                best_ask_raw = asks[0][0]
                # Convert to string first for Decimal robustness
                results[BEST_BID_KEY] = Decimal(str(best_bid_raw))
                results[BEST_ASK_KEY] = Decimal(str(best_ask_raw))

                if results[BEST_BID_KEY] > 0 and results[BEST_ASK_KEY] > 0:
                    spread = results[BEST_ASK_KEY] - results[BEST_BID_KEY]
                    # Determine price precision dynamically if possible
                    price_precision = Decimal("0.0001")  # Default precision
                    try:
                        market = exchange.market(symbol)
                        price_prec_str = market.get("precision", {}).get("price")
                        if price_prec_str:
                            price_precision = Decimal(str(price_prec_str))
                    except Exception as market_err:
                        logger.debug(
                            f"OB Scrying: Could not get market precision for spread calc: {market_err}. Using default."
                        )
                    results[SPREAD_KEY] = spread.quantize(price_precision)
                    logger.debug(
                        f"OB Scrying: Best Bid={Fore.GREEN}{results[BEST_BID_KEY]}{Style.RESET_ALL}, Best Ask={Fore.RED}{results[BEST_ASK_KEY]}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results[SPREAD_KEY]}{Style.RESET_ALL}"
                    )
                else:
                    logger.debug("OB Scrying: Could not calculate spread (Best Bid/Ask zero or invalid).")
            else:
                logger.warning(f"{Fore.YELLOW}OB Scrying: Best bid/ask data missing or incomplete.{Style.RESET_ALL}")

        except (IndexError, InvalidOperation, ValueError, TypeError) as e:
            logger.warning(
                f"{Fore.YELLOW}OB Scrying: Error processing best bid/ask/spread for {symbol}: {e}{Style.RESET_ALL}"
            )
            results[BEST_BID_KEY] = None
            results[BEST_ASK_KEY] = None
            results[SPREAD_KEY] = None

        # Calculate cumulative volume within depth
        try:
            # Ensure inner lists have at least two elements (price, volume)
            # Convert to string first for Decimal robustness
            bid_volume_sum_raw = sum(Decimal(str(bid[1])) for bid in bids[:depth] if len(bid) > 1)
            ask_volume_sum_raw = sum(Decimal(str(ask[1])) for ask in asks[:depth] if len(ask) > 1)
            # Use a reasonable precision for volume sums
            vol_precision = Decimal("0.0001")
            bid_volume_sum = bid_volume_sum_raw.quantize(vol_precision)
            ask_volume_sum = ask_volume_sum_raw.quantize(vol_precision)
            logger.debug(
                f"OB Scrying (Depth {depth}): Cum Bid={Fore.GREEN}{bid_volume_sum}{Style.RESET_ALL}, Cum Ask={Fore.RED}{ask_volume_sum}{Style.RESET_ALL}"
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

        except (IndexError, InvalidOperation, ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(
                f"{Fore.YELLOW}OB Scrying: Error calculating cumulative volume or ratio for {symbol}: {e}{Style.RESET_ALL}"
            )
            results[BID_ASK_RATIO_KEY] = None

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}OB Scrying: API error fetching order book for {symbol}: {e}{Style.RESET_ALL}")
    except (IndexError, InvalidOperation, ValueError, TypeError) as e:
        # Catch potential errors from processing the raw OB data structure
        logger.warning(
            f"{Fore.YELLOW}OB Scrying: Error processing OB data structure for {symbol}: {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(f"{Fore.RED}OB Scrying: Unexpected error analyzing order book for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return {key: None for key in results}  # Return None dict on error

    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
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
            # Convert timestamp to UTC datetime objects
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # --- Basic Data Validation ---
            if df.empty:
                logger.warning(
                    f"{Fore.YELLOW}Data Fetch: DataFrame is empty after conversion for {symbol}.{Style.RESET_ALL}"
                )
                return None  # Cannot proceed with empty DataFrame

            # Ensure correct data types before NaN checks
            # Convert OHLCV columns to numeric, coercing errors to NaN
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Check for NaNs introduced by conversion or present in original data
            if df.isnull().values.any():
                nan_counts = df.isnull().sum()
                logger.warning(
                    f"{Fore.YELLOW}Data Fetch: Fetched OHLCV data contains NaN values after numeric conversion. Counts:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...{Style.RESET_ALL}"
                )
                # Simple imputation: Forward fill NaNs. More sophisticated methods could be used.
                df.ffill(inplace=True)
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
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """Fetches current position details for Bybit V5 via CCXT. Returns Decimal for qty/price."""
    # Default: no active position, using Decimal for precision
    default_pos: Dict[str, Any] = {
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
        if not exchange.has.get("fetchPositions"):
            logger.warning(
                f"{Fore.YELLOW}Position Check: Exchange '{exchange.id}' may not support fetchPositions.{Style.RESET_ALL}"
            )
            # Attempt fetch_position as a fallback? Or just return default? Let's return default for now.
            return default_pos

        # Determine category for V5 API call based on market info (linear/inverse)
        params = {}
        if market and market.get("linear", False):
            params = {PARAM_CATEGORY: "linear"}
        elif market and market.get("inverse", False):
            params = {PARAM_CATEGORY: "inverse"}
        else:
            # If market type unknown, try linear as a default (most common for USDT perps)
            logger.warning(
                f"{Fore.YELLOW}Position Check: Market type for {symbol} unclear, defaulting category to 'linear'.{Style.RESET_ALL}"
            )
            params = {PARAM_CATEGORY: "linear"}

        # Fetch positions for the specific symbol
        # Setting settle coin might also be needed for V5 depending on account type (e.g., USDT for linear)
        # params['settleCoin'] = 'USDT' # Add if needed
        positions = exchange.fetch_positions(symbols=[ccxt_unified_symbol], params=params)
        logger.debug(f"Fetched positions data (raw count: {len(positions)}) for symbol {ccxt_unified_symbol}")

        # Filter positions: Bybit V5 fetchPositions returns multiple entries even for one symbol/mode.
        # We need the one for the correct market_id AND One-Way mode (positionIdx=0).
        active_position_found = None
        for pos in positions:
            pos_info = pos.get(INFO_KEY, {})
            pos_symbol_raw = pos_info.get(SYMBOL_KEY)  # Raw symbol from exchange ('BTCUSDT')

            # Match the raw symbol from the position data with the market ID we expect
            if pos_symbol_raw != market_id:
                # logger.debug(f"Skipping position entry, symbol mismatch: '{pos_symbol_raw}' != '{market_id}'")
                continue

            # Check for One-Way Mode (positionIdx=0) - V5 returns as string
            position_idx_str = pos_info.get("positionIdx", "-1")
            try:
                position_idx = int(position_idx_str)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse positionIdx '{position_idx_str}' for {market_id}. Skipping.")
                continue
            if position_idx != 0:
                # logger.debug(f"Skipping position entry for {market_id}, not One-Way mode (positionIdx={position_idx}).")
                continue  # Skip hedge mode positions

            # If we found the entry for our symbol and One-Way mode, this is the one.
            active_position_found = pos
            break  # Stop searching once the correct entry is found

        # Process the found position (if any)
        if active_position_found:
            pos_info = active_position_found.get(INFO_KEY, {})
            # Check Side (V5: 'Buy', 'Sell', 'None')
            pos_side_v5 = pos_info.get(SIDE_KEY, POSITION_SIDE_NONE)  # Raw side from exchange
            determined_side = POSITION_SIDE_NONE
            if pos_side_v5 == BYBIT_SIDE_BUY:
                determined_side = POSITION_SIDE_LONG
            elif pos_side_v5 == BYBIT_SIDE_SELL:
                determined_side = POSITION_SIDE_SHORT
            # If side is 'None', it implies no position or a flat state for this entry

            # Check Position Size (V5: 'size') - this is the key indicator of an active position
            size_str = pos_info.get("size")
            if size_str is None or size_str == "":
                logger.debug(f"Position Check: Size field missing for {market_id}. Assuming flat.")
                return default_pos  # Treat as flat if size is missing

            try:
                # Convert size string to Decimal
                size = Decimal(str(size_str))
                # Check if size is significantly different from zero using epsilon
                if abs(size) > CONFIG.position_qty_epsilon:
                    # Found active position! Get entry price.
                    # Prefer raw V5 avgPrice, fallback to CCXT unified entryPrice
                    entry_price_str = pos_info.get(AVG_PRICE_KEY) or active_position_found.get(ENTRY_PRICE_KEY)
                    entry_price = Decimal("0.0")
                    if entry_price_str is not None and entry_price_str != "":
                        try:
                            # Convert entry price string to Decimal
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
                    # Size is zero or negligible, treat as flat
                    logger.info(
                        f"{Fore.BLUE}Position Check: Position size for {market_id} is zero/negligible ({size_str}). Treating as flat.{Style.RESET_ALL}"
                    )
                    return default_pos

            except (ValueError, TypeError, InvalidOperation) as e:
                logger.warning(
                    f"{Fore.YELLOW}Position Check: Error parsing size '{size_str}' for {market_id}: {e}{Style.RESET_ALL}"
                )
                return default_pos  # Treat as flat on parsing error
        else:
            # No position entry matched the symbol and One-Way mode criteria
            logger.info(
                f"{Fore.BLUE}Position Check: No active One-Way Mode position found for {market_id}.{Style.RESET_ALL}"
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
    """Sets leverage for Bybit V5, checks market type, retries."""
    logger.info(f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    market_base = "N/A"
    market = None
    try:
        market = exchange.market(symbol)
        if not market:
            raise KeyError(f"Market info not found for {symbol}")
        market_base = market.get("base", "N/A")
        # Check if it's a contract market (swap, futures)
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
            # Bybit V5 requires setting buy and sell leverage separately via set_leverage call in CCXT
            # CCXT handles mapping this to the correct API call structure for Bybit V5
            # The unified method `set_leverage` should handle this abstraction.
            # Bybit V5 also requires specifying margin mode (isolated/cross) when setting leverage.
            # We set 'defaultMarginMode': 'isolated' in initialize_exchange, CCXT should use this.
            logger.debug(
                f"Leverage Conjuring: Calling exchange.set_leverage({leverage}, '{symbol}') (Attempt {attempt + 1}/{CONFIG.retry_count})"
            )
            # Params might be needed if CCXT abstraction fails:
            # params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage), 'marginMode': 0} # 0 for isolated, 1 for cross
            response = exchange.set_leverage(leverage=leverage, symbol=symbol)  # Rely on CCXT abstraction
            logger.success(
                f"{Fore.GREEN}Leverage Conjuring: Successfully set leverage to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}"
            )
            return True

        except ccxt.ExchangeError as e:
            error_msg_lower = str(e).lower()
            # Bybit V5 specific error codes or messages for "leverage not modified"
            # 110044: "Set leverage not modified" (from Bybit docs)
            # Check for common phrases as well
            if "110044" in str(e) or any(
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
            # Fail fast on unexpected errors
            send_sms_alert(f"[{market_base}] CRITICAL: Leverage set FAILED (Unexpected: {type(e).__name__}) {symbol}.")
            return False
    return False


def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal"
) -> Optional[Dict[str, Any]]:
    """Closes active position via market order with reduce_only, validates first."""
    initial_side = position_to_close.get(SIDE_KEY, POSITION_SIDE_NONE)
    initial_qty = position_to_close.get(QTY_KEY, Decimal("0.0"))
    market_base = symbol.split("/")[0] if "/" in symbol else symbol

    logger.info(
        f"{Fore.YELLOW}Banish Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}"
    )

    # Re-validate Position Before Closing
    logger.debug("Banish Position: Re-validating live position status...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position.get(SIDE_KEY, POSITION_SIDE_NONE)
    live_amount_to_close = live_position.get(QTY_KEY, Decimal("0.0"))  # Absolute value from get_current_position

    if live_position_side == POSITION_SIDE_NONE or live_amount_to_close <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Banish Position: Discrepancy detected or position already closed. Initial check showed {initial_side}, but live check shows none or negligible qty ({live_amount_to_close:.8f}). Assuming closed.{Style.RESET_ALL}"
        )
        return None  # Treat as already closed, no action needed

    # Determine the side of the market order needed to close
    side_to_execute_close = SIDE_SELL if live_position_side == POSITION_SIDE_LONG else SIDE_BUY

    # Place Reduce-Only Market Order
    params = {PARAM_REDUCE_ONLY: True}

    try:
        # Convert the Decimal amount to float for CCXT, applying precision first
        # Use amount_to_precision which returns a string, then convert to float
        amount_str = exchange.amount_to_precision(symbol, float(live_amount_to_close))
        amount_float_prec = float(amount_str)

        if amount_float_prec <= float(CONFIG.position_qty_epsilon):  # Check precision-adjusted amount
            logger.error(
                f"{Fore.RED}Banish Pos: Closing amount {amount_str} ({live_amount_to_close}) negligible after precision. Abort.{Style.RESET_ALL}"
            )
            return None

        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}Banish Position: Attempting to CLOSE {live_position_side} position ({reason}): Executing {side_to_execute_close.upper()} MARKET order for {amount_str} {symbol} (Reduce-Only){Style.RESET_ALL}"
        )

        order = exchange.create_market_order(
            symbol=symbol, side=side_to_execute_close, amount=amount_float_prec, params=params
        )

        # Log details immediately using data from the returned order object
        fill_price_str = "?"
        filled_qty_str = "?"
        cost_str = "?"
        order_id_short = str(order.get(ID_KEY, "N/A"))[-6:]

        # Prefer 'average' for fill price, fallback to 'price'
        avg_price_raw = order.get(AVERAGE_KEY) or order.get(PRICE_KEY)
        if avg_price_raw is not None:
            with contextlib.suppress(InvalidOperation, ValueError, TypeError):
                fill_price_str = f"{Decimal(str(avg_price_raw)):.4f}"

        if order.get(FILLED_KEY) is not None:
            with contextlib.suppress(InvalidOperation, ValueError, TypeError):
                filled_qty_str = f"{Decimal(str(order.get(FILLED_KEY))):.8f}"

        if order.get(COST_KEY) is not None:
            with contextlib.suppress(InvalidOperation, ValueError, TypeError):
                cost_str = f"{Decimal(str(order.get(COST_KEY))):.2f}"

        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Banish Position: CLOSE Order ({reason}) placed successfully for {symbol}. Qty Filled: {filled_qty_str}/{amount_str}, Avg Fill ~{fill_price_str}, Cost: {cost_str} USDT. ID:...{order_id_short}{Style.RESET_ALL}"
        )

        # Send SMS Alert
        sms_msg = f"[{market_base}] BANISHED {live_position_side} {amount_str} @ ~{fill_price_str} ({reason}). ID:...{order_id_short}"
        send_sms_alert(sms_msg)
        return order  # Return the order dictionary

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
        # Check for specific errors indicating already closed/closing or order would not reduce
        # Bybit V5 error codes: 110025 ("Position size is zero"), 110043 ("Order would not reduce position size")
        # Also check for common phrases
        if (
            "110025" in str(e)
            or "110043" in str(e)
            or any(
                phrase in err_str_lower
                for phrase in [
                    "order would not reduce position size",
                    "position is zero",
                    "position size is zero",
                    "cannot be less than",
                    "position has been closed",
                ]
            )
        ):
            logger.warning(
                f"{Fore.YELLOW}Banish Position ({reason}): Exchange indicates order would not reduce size or position is zero/closed. Assuming already closed. Error: {e}{Style.RESET_ALL}"
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

    return None  # Return None if closing failed


def calculate_position_size(
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Calculates position size based on risk, checks limits. Returns Decimals."""
    logger.debug(
        f"Risk Calc Input: Equity={equity:.2f}, Risk%={risk_per_trade_pct:.3%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x, Symbol={symbol}"
    )

    # Input Validation
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(
            f"{Fore.RED}Risk Calc: Invalid entry/SL price (must be > 0). Entry={entry_price}, SL={stop_loss_price}{Style.RESET_ALL}"
        )
        return None, None
    price_difference_per_unit = abs(entry_price - stop_loss_price)
    if price_difference_per_unit <= CONFIG.position_qty_epsilon:
        logger.error(
            f"{Fore.RED}Risk Calc: Entry/SL prices are identical or too close to calculate risk ({price_difference_per_unit}). Entry={entry_price}, SL={stop_loss_price}{Style.RESET_ALL}"
        )
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(
            f"{Fore.RED}Risk Calc: Invalid risk percentage: {risk_per_trade_pct:.3%}. Must be between 0 and 1.{Style.RESET_ALL}"
        )
        return None, None
    if equity <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.2f}. Must be positive.{Style.RESET_ALL}")
        return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}. Must be positive.{Style.RESET_ALL}")
        return None, None

    # Calculation
    risk_amount_usdt: Decimal = equity * risk_per_trade_pct
    quantity: Decimal = risk_amount_usdt / price_difference_per_unit

    # Apply exchange precision to quantity
    try:
        # Use amount_to_precision which returns a string, then convert back to Decimal
        quantity_precise_str = exchange.amount_to_precision(symbol, float(quantity))
        quantity_precise = Decimal(quantity_precise_str)
        logger.debug(f"Risk Calc: Raw Qty={quantity:.18f}, Precise Qty={quantity_precise_str}")
        quantity = quantity_precise  # Use the precision-adjusted Decimal value for further checks
    except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Could not apply exchange precision to quantity {quantity:.8f} for {symbol}. Using raw value. Error: {e}{Style.RESET_ALL}"
        )
    except Exception as e:  # Catch any other unexpected error during precision adjustment
        logger.error(
            f"{Fore.RED}Risk Calc: Unexpected error applying precision to quantity {quantity:.8f} for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return None, None  # Fail calculation if precision fails unexpectedly

    if quantity <= CONFIG.position_qty_epsilon:
        logger.error(
            f"{Fore.RED}Risk Calc: Calculated quantity ({quantity}) is zero or negligible after precision adjustment.{Style.RESET_ALL}"
        )
        return None, None

    # Estimate Value and Margin using the precise quantity
    position_value_usdt: Decimal = quantity * entry_price
    required_margin_estimate: Decimal = position_value_usdt / Decimal(leverage)
    logger.debug(
        f"Risk Calc Output: RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference_per_unit:.4f} => PreciseQty={quantity:.8f}, EstVal={position_value_usdt:.2f}, EstMargin={required_margin_estimate:.2f}"
    )

    # Exchange Limit Checks (using precise quantity and estimated value)
    try:
        market = exchange.market(symbol)
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})

        # Safely get min/max limits, converting to Decimal
        def get_limit_decimal(limit_dict: Optional[dict], key: str, default: str) -> Decimal:
            if limit_dict is None:
                return Decimal(default)
            val = limit_dict.get(key)
            if val is not None:
                try:
                    return Decimal(str(val))
                except (InvalidOperation, ValueError, TypeError):
                    return Decimal(default)
            return Decimal(default)

        min_amount = get_limit_decimal(amount_limits, "min", "0")
        max_amount = get_limit_decimal(amount_limits, "max", "inf")
        min_cost = get_limit_decimal(cost_limits, "min", "0")
        max_cost = get_limit_decimal(cost_limits, "max", "inf")

        logger.debug(
            f"Market Limits for {symbol}: MinAmt={min_amount}, MaxAmt={max_amount}, MinCost={min_cost}, MaxCost={max_cost}"
        )

        # Check against limits
        if quantity < min_amount:
            logger.error(
                f"{Fore.RED}Risk Calc: Calculated Qty {quantity:.8f} is less than Min Amount limit {min_amount:.8f}.{Style.RESET_ALL}"
            )
            return None, None
        if position_value_usdt < min_cost:
            logger.error(
                f"{Fore.RED}Risk Calc: Estimated Value {position_value_usdt:.2f} is less than Min Cost limit {min_cost:.2f}.{Style.RESET_ALL}"
            )
            return None, None
        if quantity > max_amount:
            logger.warning(
                f"{Fore.YELLOW}Risk Calc: Calculated Qty {quantity:.8f} exceeds Max Amount limit {max_amount:.8f}. Capping quantity to limit.{Style.RESET_ALL}"
            )
            # Apply precision to the max_amount limit itself before assigning
            max_amount_str = exchange.amount_to_precision(symbol, float(max_amount))
            quantity = Decimal(max_amount_str)  # Cap quantity to the precision-adjusted max limit
            # Recalculate estimated value and margin based on capped quantity
            position_value_usdt = quantity * entry_price
            required_margin_estimate = position_value_usdt / Decimal(leverage)
            logger.info(
                f"Risk Calc: Capped Qty={quantity:.8f}, New EstVal={position_value_usdt:.2f}, New EstMargin={required_margin_estimate:.2f}"
            )
        if position_value_usdt > max_cost:
            # If even the capped quantity's value exceeds max cost, it's an issue
            logger.error(
                f"{Fore.RED}Risk Calc: Estimated Value {position_value_usdt:.2f} (potentially capped) exceeds Max Cost limit {max_cost:.2f}.{Style.RESET_ALL}"
            )
            return None, None

    except (ccxt.BadSymbol, KeyError) as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Could not fetch market limits for {symbol}: {e}. Skipping limit checks.{Style.RESET_ALL}"
        )
    except (InvalidOperation, ValueError, TypeError) as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Error parsing market limits for {symbol}: {e}. Skipping limit checks.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Unexpected error checking market limits: {e}. Skipping limit checks.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    # Return Decimal values for quantity and margin estimate
    return quantity, required_margin_estimate


def confirm_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
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

            # If status is 'open' or None/unknown, continue retrying

        except ccxt.OrderNotFound:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix}: Order not found via fetch_order (Attempt {attempt + 1}). Might be processing or already closed/canceled.{Style.RESET_ALL}"
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
            # Break on unexpected error? Or continue retrying? Let's continue for robustness.

        # Wait before retrying fetch_order
        if attempt < CONFIG.fetch_order_status_retries - 1:
            time.sleep(CONFIG.fetch_order_status_delay)

    # --- Fallback Method: fetch_closed_orders (if fetch_order didn't confirm 'closed') ---
    if not confirmed_order:
        logger.debug(f"{log_prefix}: fetch_order did not confirm 'closed'. Trying fallback: fetch_closed_orders...")
        try:
            # Fetch recent closed orders (increase limit slightly for safety)
            # Add timestamp filter if possible to limit results
            since_timestamp = int((time.time() - 600) * 1000)  # Look back 10 minutes
            closed_orders = exchange.fetch_closed_orders(symbol, limit=20, since=since_timestamp)
            logger.debug(f"{log_prefix}: Fallback fetched recent closed orders: {[o.get('id') for o in closed_orders]}")
            for order in closed_orders:
                if order.get(ID_KEY) == order_id:
                    status = order.get(STATUS_KEY)
                    # We only care if it's 'closed' in the fallback, as 'canceled' etc. should have been caught by fetch_order
                    if status == ORDER_STATUS_CLOSED:
                        logger.success(f"{log_prefix}: Confirmed FILLED via fetch_closed_orders fallback.")
                        confirmed_order = order
                        break
                    else:
                        # Found the order but it wasn't closed (e.g., canceled and fetch_order missed it?)
                        logger.warning(
                            f"{Fore.YELLOW}{log_prefix}: Found order in fallback, but status is '{status}'. Assuming failed/not filled.{Style.RESET_ALL}"
                        )
                        return None  # Treat as failed if found in closed but not 'closed' status
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
        # Final check: Ensure filled amount is positive
        try:
            filled_qty_str = confirmed_order.get(FILLED_KEY, "0")
            filled_qty = Decimal(str(filled_qty_str))
            if filled_qty <= CONFIG.position_qty_epsilon:
                logger.error(
                    f"{Fore.RED}{log_prefix}: Order {order_id} confirmed '{ORDER_STATUS_CLOSED}' but filled quantity is zero/negligible ({filled_qty}). Treating as FAILED.{Style.RESET_ALL}"
                )
                return None
        except (InvalidOperation, ValueError, TypeError):
            logger.error(
                f"{Fore.RED}{log_prefix}: Could not parse filled quantity '{confirmed_order.get(FILLED_KEY)}' from confirmed order {order_id}. Treating as FAILED.{Style.RESET_ALL}"
            )
            return None
        return confirmed_order
    else:
        elapsed = time.time() - start_time
        logger.error(
            f"{Fore.RED}{log_prefix}: FAILED to confirm fill for order {order_id} using both methods within timeout ({elapsed:.1f}s). Assume failure.{Style.RESET_ALL}"
        )
        return None


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    quantity_decimal: Decimal,  # Receive Decimal from calculation
    required_margin_decimal: Decimal,  # Receive Decimal from calculation
    stop_loss_price: Decimal,
    take_profit_price: Decimal,
) -> Optional[Dict[str, Any]]:
    """Places market entry, confirms fill, then places SL/TP orders based on actual fill."""
    market_base = symbol.split("/")[0] if "/" in symbol else symbol
    log_prefix = f"Entry Order ({side.upper()})"
    order_id: Optional[str] = None
    entry_order: Optional[Dict[str, Any]] = None
    confirmed_order: Optional[Dict[str, Any]] = None  # To store the confirmed filled order

    try:
        # Fetch Balance & Market Info
        logger.debug(f"{log_prefix}: Gathering resources...")
        balance_info = exchange.fetch_balance()
        # Use 'free' balance for margin check (ensure it's USDT)
        free_balance_raw = balance_info.get("free", {}).get(USDT_SYMBOL)
        if free_balance_raw is None:
            raise ValueError("Could not fetch free USDT balance.")
        free_balance = Decimal(str(free_balance_raw))

        market = exchange.market(symbol)
        if not market:
            raise ValueError(f"Market info not found for {symbol}")

        # Get last price for estimations (capping, final cost check)
        ticker = exchange.fetch_ticker(symbol)
        last_price_raw = ticker.get(LAST_PRICE_KEY)
        if last_price_raw is None:
            raise ValueError("Could not fetch last price for estimates.")
        entry_price_estimate = Decimal(str(last_price_raw))
        if entry_price_estimate <= 0:
            raise ValueError(f"Fetched invalid last price: {entry_price_estimate}")

        # --- Pre-flight Checks ---
        # Quantity and margin are already Decimals

        # Cap Quantity based on MAX_ORDER_USDT_AMOUNT
        estimated_value = quantity_decimal * entry_price_estimate
        if estimated_value > CONFIG.max_order_usdt_amount:
            original_quantity_str = f"{quantity_decimal:.8f}"
            # Calculate capped quantity as Decimal first for precision
            capped_qty_decimal_raw = CONFIG.max_order_usdt_amount / entry_price_estimate
            # Apply exchange precision using float conversion temporarily
            quantity_str = exchange.amount_to_precision(symbol, float(capped_qty_decimal_raw))
            quantity_decimal = Decimal(quantity_str)  # Update Decimal quantity
            # Recalculate estimated margin based on capped Decimal quantity
            required_margin_decimal = (quantity_decimal * entry_price_estimate) / Decimal(CONFIG.leverage)
            logger.warning(
                f"{Fore.YELLOW}{log_prefix}: Qty {original_quantity_str} (Val ~{estimated_value:.2f}) > Max {CONFIG.max_order_usdt_amount:.2f}. Capping to {quantity_str} (New Est. Margin ~{required_margin_decimal:.2f}).{Style.RESET_ALL}"
            )

        # Final Limit Checks (Amount & Cost) using potentially capped quantity
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})
        min_amount = (
            Decimal(str(amount_limits.get("min", "0"))) if amount_limits.get("min") is not None else Decimal("0")
        )
        min_cost = Decimal(str(cost_limits.get("min", "0"))) if cost_limits.get("min") is not None else Decimal("0")

        if quantity_decimal < min_amount:
            logger.error(
                f"{Fore.RED}{log_prefix}: Final quantity {quantity_decimal:.8f} < Min Amount limit {min_amount:.8f}. Abort.{Style.RESET_ALL}"
            )
            return None
        estimated_cost_final = quantity_decimal * entry_price_estimate
        if estimated_cost_final < min_cost:
            logger.error(
                f"{Fore.RED}{log_prefix}: Final estimated cost {estimated_cost_final:.2f} < Min Cost limit {min_cost:.2f}. Abort.{Style.RESET_ALL}"
            )
            return None

        # Margin Check using potentially recalculated margin estimate
        required_margin_with_buffer = required_margin_decimal * CONFIG.required_margin_buffer
        logger.debug(
            f"{log_prefix}: Free Balance={free_balance:.2f}, Est. Margin Required (incl. buffer)={required_margin_with_buffer:.2f}"
        )
        if free_balance < required_margin_with_buffer:
            logger.error(
                f"{Fore.RED}{log_prefix}: Insufficient free balance ({free_balance:.2f}) for required margin ({required_margin_with_buffer:.2f}). Abort.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] Order REJECTED ({side.upper()}): Insuff. free balance. Need ~{required_margin_with_buffer:.2f}"
            )
            return None

        # --- Place Market Order ---
        entry_side_color = Back.GREEN if side == SIDE_BUY else Back.RED
        text_color = Fore.BLACK if side == SIDE_BUY else Fore.WHITE
        # Convert final Decimal quantity to float for create_market_order
        quantity_float = float(quantity_decimal)
        logger.warning(
            f"{entry_side_color}{text_color}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {quantity_float:.8f} {symbol} ***{Style.RESET_ALL}"
        )
        entry_order = exchange.create_market_order(symbol, side, quantity_float)
        order_id = entry_order.get(ID_KEY)
        if not order_id:
            raise ValueError("Market order placed but no ID returned.")
        logger.success(
            f"{log_prefix}: Market order submitted. ID: ...{order_id[-6:]}. Waiting for fill confirmation..."
        )
        time.sleep(CONFIG.post_entry_delay_seconds)  # Allow time for order processing

        # --- Confirm Order Fill ---
        confirmed_order = confirm_order_fill(exchange, order_id, symbol)
        if not confirmed_order:
            logger.error(
                f"{Fore.RED}{log_prefix}: FAILED to confirm fill for entry order {order_id}. Aborting SL/TP placement.{Style.RESET_ALL}"
            )
            send_sms_alert(f"[{market_base}] CRITICAL: Entry fill confirm FAILED: {order_id}. Manual check needed!")
            # Consider emergency close here? Risky if fill status unknown. Best to alert and stop.
            return None

        # --- Extract Actual Fill Details ---
        try:
            # Use Decimal for accuracy when reading fill details
            actual_filled_qty_str = confirmed_order.get(FILLED_KEY)
            # Prefer 'average' price from CCXT unified response, fallback to raw 'avgPrice' or 'price'
            actual_avg_price_str = (
                confirmed_order.get(AVERAGE_KEY)
                or confirmed_order.get(INFO_KEY, {}).get(AVG_PRICE_KEY)
                or confirmed_order.get(PRICE_KEY)
            )

            if actual_filled_qty_str is None or actual_avg_price_str is None:
                raise ValueError("Missing filled quantity or average price in confirmed order.")

            actual_filled_qty = Decimal(str(actual_filled_qty_str))
            actual_avg_price = Decimal(str(actual_avg_price_str))

            if actual_filled_qty <= CONFIG.position_qty_epsilon or actual_avg_price <= 0:
                raise ValueError(f"Invalid fill data: Qty={actual_filled_qty}, Price={actual_avg_price}")
            logger.success(
                f"{log_prefix}: Fill Confirmed: Order ID ...{order_id[-6:]}, Filled Qty={actual_filled_qty:.8f}, Avg Price={actual_avg_price:.4f}"
            )
        except (InvalidOperation, ValueError, TypeError, KeyError) as e:
            logger.error(
                f"{Fore.RED}{log_prefix}: Error parsing confirmed fill details for order {order_id}: {e}. Data: {confirmed_order}{Style.RESET_ALL}"
            )
            send_sms_alert(f"[{market_base}] CRITICAL ERROR parsing fill data: {order_id}. Manual check needed!")
            return None  # Cannot proceed without valid fill details

        # --- Place SL/TP Orders ---
        logger.info(
            f"{log_prefix}: Placing SL ({stop_loss_price}) and TP ({take_profit_price}) orders for filled qty {actual_filled_qty:.8f}..."
        )
        sl_tp_success = True
        close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
        # Use the actual filled quantity (as float for create_order) for SL/TP orders
        sl_tp_amount_float = float(actual_filled_qty)

        # Apply precision to SL/TP prices before sending
        try:
            sl_price_str = exchange.price_to_precision(symbol, float(stop_loss_price))
            tp_price_str = exchange.price_to_precision(symbol, float(take_profit_price))
        except Exception as e:
            logger.error(
                f"{Fore.RED}{log_prefix}: Failed to apply precision to SL/TP prices: {e}. Aborting SL/TP placement.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] CRITICAL: SL/TP price precision FAILED for {order_id}. Manual check needed!"
            )
            # Attempt to close position if SL/TP cannot be placed due to precision error? Risky. Alerting is safer.
            return None  # Cannot place SL/TP without correct price format

        # SL Order Params (stopMarket)
        sl_trigger_direction = (
            2 if side == SIDE_BUY else 1
        )  # Trigger when price goes BELOW for LONG SL, ABOVE for SHORT SL
        sl_params = {
            PARAM_STOP_PRICE: sl_price_str,
            PARAM_REDUCE_ONLY: True,
            PARAM_TRIGGER_DIRECTION: sl_trigger_direction,  # 1: Mark price > trigger price, 2: Mark price < trigger price
            "tpslMode": "Full",  # Bybit V5: 'Full' or 'Partial'. Assume full position SL/TP.
            "slOrderType": "Market",  # Explicitly state market execution for SL trigger
        }
        # TP Order Params (stopMarket)
        tp_trigger_direction = (
            1 if side == SIDE_BUY else 2
        )  # Trigger when price goes ABOVE for LONG TP, BELOW for SHORT TP
        tp_params = {
            PARAM_STOP_PRICE: tp_price_str,
            PARAM_REDUCE_ONLY: True,
            PARAM_TRIGGER_DIRECTION: tp_trigger_direction,  # 1: Mark price > trigger price, 2: Mark price < trigger price
            "tpslMode": "Full",  # As above
            "tpOrderType": "Market",  # Explicitly state market execution for TP trigger
        }

        sl_order_id_short, tp_order_id_short = "N/A", "N/A"
        sl_order_info, tp_order_info = None, None

        # Place Stop-Loss Order
        try:
            logger.debug(
                f"Placing SL order: symbol={symbol}, type={ORDER_TYPE_STOP_MARKET}, side={close_side}, amount={sl_tp_amount_float}, params={sl_params}"
            )
            sl_order_info = exchange.create_order(
                symbol, ORDER_TYPE_STOP_MARKET, close_side, sl_tp_amount_float, params=sl_params
            )
            sl_order_id_short = str(sl_order_info.get(ID_KEY, "N/A"))[-6:]
            logger.success(
                f"{Fore.GREEN}{log_prefix}: Stop-Loss order placed. ID: ...{sl_order_id_short}{Style.RESET_ALL}"
            )
            time.sleep(0.1)  # Small delay between orders
        except Exception as e:
            sl_tp_success = False
            logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix}: FAILED to place Stop-Loss order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}] CRITICAL: SL order FAILED for {symbol} ({side}) after entry {order_id[-6:]}: {e}"
            )
            # Attempt to close position immediately if SL fails? VERY RISKY. Alert is primary action.
            # Consider adding an emergency close attempt here if desired, but be aware of risks.
            # emergency_close_result = close_position(exchange, symbol, {SIDE_KEY: side, QTY_KEY: actual_filled_qty}, reason="SL_PLACEMENT_FAILED")
            # if emergency_close_result: logger.warning("Emergency close attempted due to SL failure.")
            # else: logger.error("Emergency close FAILED after SL placement failure.")

        # Place Take-Profit Order (only if SL succeeded or if we want TP regardless)
        # Let's place TP even if SL failed, but log the overall failure.
        try:
            logger.debug(
                f"Placing TP order: symbol={symbol}, type={ORDER_TYPE_STOP_MARKET}, side={close_side}, amount={sl_tp_amount_float}, params={tp_params}"
            )
            tp_order_info = exchange.create_order(
                symbol, ORDER_TYPE_STOP_MARKET, close_side, sl_tp_amount_float, params=tp_params
            )
            tp_order_id_short = str(tp_order_info.get(ID_KEY, "N/A"))[-6:]
            logger.success(
                f"{Fore.GREEN}{log_prefix}: Take-Profit order placed. ID: ...{tp_order_id_short}{Style.RESET_ALL}"
            )
        except Exception as e:
            # TP failure is less critical than SL, but still log as error and mark overall failure
            sl_tp_success = False
            logger.error(
                f"{Back.YELLOW}{Fore.BLACK}{log_prefix}: FAILED to place Take-Profit order: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}] WARNING: TP order FAILED for {symbol} ({side}) after entry {order_id[-6:]}: {e}"
            )

        # Final outcome
        if sl_tp_success:
            logger.info(
                f"{Fore.GREEN}{log_prefix}: Entry order filled and SL/TP orders placed successfully.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] Entered {side.upper()} {actual_filled_qty:.4f} @ {actual_avg_price:.2f}. SL=...{sl_order_id_short}, TP=...{tp_order_id_short}"
            )
            # Return the confirmed *entry* order details
            return confirmed_order
        else:
            logger.error(
                f"{Back.RED}{Fore.WHITE}{log_prefix}: Entry filled (ID:...{order_id[-6:]}), but FAILED to place one/both SL/TP orders. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
            )
            # Return None to indicate the overall process including SL/TP setup failed.
            # The position is open but potentially unprotected.
            return None

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
    # Determine required length based on longest indicator period + buffer
    required_ohlcv_len = (
        max(CONFIG.st_atr_length, CONFIG.confirm_st_atr_length, CONFIG.volume_ma_period, CONFIG.atr_calculation_period)
        + CONFIG.api_fetch_limit_buffer
    )
    df = fetch_ohlcv(exchange, symbol, timeframe, limit=required_ohlcv_len)
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - unable to fetch valid OHLCV data.{Style.RESET_ALL}")
        return

    order_book_data: Optional[Dict[str, Optional[Decimal]]] = None
    # Fetch OB if always required OR if confirmation is enabled (fetch only when needed later)
    if CONFIG.fetch_order_book_per_cycle:
        order_book_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
        # If OB analysis is mandatory for entry and it failed, skip cycle
        if order_book_data is None and CONFIG.use_ob_confirm:
            logger.warning(
                f"{Fore.YELLOW}Trade Logic: Skipping cycle - failed to get required OB data (fetch_order_book_per_cycle=True).{Style.RESET_ALL}"
            )
            return

    # --- 2. Calculate Indicators ---
    logger.debug("Calculating indicators...")
    df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
    df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
    vol_atr_results = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)

    # Check for critical indicator calculation failures
    # Check if columns exist and the last value is not NaN/NA
    # Use the renamed column names from calculate_supertrend
    if "trend" not in df.columns or pd.isna(df["trend"].iloc[-1]):
        logger.warning(
            f"{Fore.YELLOW}Trade Logic: Skipping cycle - primary Supertrend trend calculation failed or resulted in NaN.{Style.RESET_ALL}"
        )
        return
    if "confirm_trend" not in df.columns or pd.isna(df["confirm_trend"].iloc[-1]):
        logger.warning(
            f"{Fore.YELLOW}Trade Logic: Skipping cycle - confirmation Supertrend trend calculation failed or resulted in NaN.{Style.RESET_ALL}"
        )
        return
    if vol_atr_results is None or vol_atr_results.get(ATR_KEY) is None:
        logger.warning(
            f"{Fore.YELLOW}Trade Logic: Skipping cycle - Volume/ATR calculation failed or ATR is None.{Style.RESET_ALL}"
        )
        return

    # --- 3. Extract Latest Indicator Values ---
    try:
        last_candle = df.iloc[-1]
        # Primary Supertrend (using renamed columns)
        st_trend = last_candle["trend"]  # Current trend direction (-1 or 1)
        st_long_signal = last_candle["st_long_flip"]  # True only on the candle where trend flipped to long
        st_short_signal = last_candle["st_short_flip"]  # True only on the candle where trend flipped to short
        # Confirmation Supertrend (using renamed columns)
        confirm_st_trend = last_candle["confirm_trend"]  # Current trend direction (-1 or 1)
        # Volume/ATR
        current_atr: Optional[Decimal] = vol_atr_results.get(ATR_KEY)  # Decimal or None
        volume_ratio: Optional[Decimal] = vol_atr_results.get(VOLUME_RATIO_KEY)  # Decimal or None
        # Current Price (use last close)
        current_price = Decimal(str(last_candle["close"]))

        # Validate essential values extracted
        if current_atr is None or pd.isna(st_trend) or pd.isna(confirm_st_trend) or current_price <= 0:
            raise ValueError(
                "Essential indicator values (ATR, ST trends, Price) are None/NaN/invalid after extraction."
            )
        # Flip signals should be boolean, pd.isna check might not be needed if calculation ensures bool
        # FIX: Ensure boolean type check is correct
        if not isinstance(
            st_long_signal, (bool, pd._libs.missing.NAType)
        ):  # Handle potential pandas NA type if fillna fails
            if pd.isna(st_long_signal):
                st_long_signal = False  # Treat NA as False
            elif not isinstance(st_long_signal, bool):  # If still not boolean, raise error
                raise ValueError(
                    f"Supertrend long flip signal (st_long_flip) is not boolean or NA, type: {type(st_long_signal)}"
                )
        if not isinstance(st_short_signal, (bool, pd._libs.missing.NAType)):
            if pd.isna(st_short_signal):
                st_short_signal = False
            elif not isinstance(st_short_signal, bool):
                raise ValueError(
                    f"Supertrend short flip signal (st_short_flip) is not boolean or NA, type: {type(st_short_signal)}"
                )
        # Ensure they are actual booleans after handling potential NA
        st_long_signal = bool(st_long_signal)
        st_short_signal = bool(st_short_signal)

    except (IndexError, KeyError, ValueError, InvalidOperation) as e:
        logger.error(
            f"{Fore.RED}Trade Logic: Error accessing or validating indicator/price data from last candle: {e}{Style.RESET_ALL}"
        )
        logger.debug(f"DataFrame tail:\n{df.tail()}")
        logger.debug(f"Vol/ATR Results: {vol_atr_results}")
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
    vol_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else "N/A"
    logger.info(
        f"State | Indicators: Price={current_price:.4f}, ATR={current_atr:.4f}, ST Trend={st_trend}, Confirm ST Trend={confirm_st_trend}, VolRatio={vol_ratio_str}"
    )

    # --- 5. Determine Signals ---
    # Entry signal: ST flip occurred on this candle AND confirmation ST agrees with the new direction
    long_entry_signal = st_long_signal and confirm_st_trend == 1
    short_entry_signal = st_short_signal and confirm_st_trend == -1
    # Exit signal: Primary ST trend flips against the current position
    close_long_signal = position_side == POSITION_SIDE_LONG and st_trend == -1
    close_short_signal = position_side == POSITION_SIDE_SHORT and st_trend == 1
    logger.debug(
        f"Signals: EntryLong={long_entry_signal}, EntryShort={short_entry_signal}, CloseLong={close_long_signal}, CloseShort={close_short_signal}"
    )

    # --- 6. Decision Making ---

    # **Exit Logic:** Prioritize closing existing positions based on primary ST flip
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
        time.sleep(CONFIG.post_close_delay_seconds)  # Pause after closing attempt
        return  # End cycle after attempting exit

    # **Entry Logic:** Only consider entry if currently flat
    if position_side == POSITION_SIDE_NONE:
        selected_side = None
        if long_entry_signal:
            selected_side = SIDE_BUY
        elif short_entry_signal:
            selected_side = SIDE_SELL

        if selected_side:
            logger.info(
                f"{Fore.CYAN}Entry Signal: Potential {selected_side.upper()} entry detected by Supertrend flip and confirmation.{Style.RESET_ALL}"
            )

            # --- Entry Confirmations ---
            volume_confirmed = True
            if CONFIG.require_volume_spike_for_entry:
                if volume_ratio is None or volume_ratio <= CONFIG.volume_spike_threshold:
                    volume_confirmed = False
                    vol_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else "N/A"
                    logger.info(
                        f"Entry REJECTED ({selected_side.upper()}): Volume spike confirmation FAILED (Ratio: {vol_ratio_str} <= Threshold: {CONFIG.volume_spike_threshold})."
                    )
                else:
                    logger.info(
                        f"{Fore.GREEN}Entry Check ({selected_side.upper()}): Volume spike OK (Ratio: {volume_ratio:.2f}).{Style.RESET_ALL}"
                    )

            ob_confirmed = True
            # Only check OB if volume passed (or volume check disabled) AND OB check is enabled
            if volume_confirmed and CONFIG.use_ob_confirm:
                # Fetch OB data now if not fetched per cycle
                if order_book_data is None:
                    logger.debug("Fetching OB data for confirmation...")
                    order_book_data = analyze_order_book(
                        exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
                    )

                # Check OB results
                if order_book_data is None or order_book_data.get(BID_ASK_RATIO_KEY) is None:
                    ob_confirmed = False
                    logger.warning(
                        f"{Fore.YELLOW}Entry REJECTED ({selected_side.upper()}): OB confirmation FAILED (Could not get valid OB data/ratio).{Style.RESET_ALL}"
                    )
                else:
                    ob_ratio = order_book_data[BID_ASK_RATIO_KEY]  # Should not be None here
                    if ob_ratio is None:  # Defensive check
                        ob_confirmed = False
                        logger.warning(
                            f"{Fore.YELLOW}Entry REJECTED ({selected_side.upper()}): OB Ratio is None unexpectedly.{Style.RESET_ALL}"
                        )
                    else:
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

            # --- Proceed with Entry Calculation if All Confirmations Pass ---
            if volume_confirmed and ob_confirmed:
                logger.success(
                    f"{Fore.GREEN}{Style.BRIGHT}Entry CONFIRMED ({selected_side.upper()}): All checks passed. Calculating parameters...{Style.RESET_ALL}"
                )

                # Calculate SL/TP Prices using current ATR
                try:
                    # Get market price precision for rounding SL/TP
                    market_info = exchange.market(symbol)
                    price_precision_str = market_info.get("precision", {}).get("price")
                    if price_precision_str is None:
                        raise ValueError("Could not determine price precision.")
                    price_precision = Decimal(str(price_precision_str))  # Ensure Decimal

                    sl_distance = current_atr * CONFIG.atr_stop_loss_multiplier
                    tp_distance = current_atr * CONFIG.atr_take_profit_multiplier
                    entry_price_est = current_price  # Use last close price as entry estimate for calculations

                    if selected_side == SIDE_BUY:
                        sl_price_raw = entry_price_est - sl_distance
                        tp_price_raw = entry_price_est + tp_distance
                    else:  # SIDE_SELL
                        sl_price_raw = entry_price_est + sl_distance
                        tp_price_raw = entry_price_est - tp_distance

                    # Ensure SL/TP are not zero or negative
                    if sl_price_raw <= 0 or tp_price_raw <= 0:
                        raise ValueError(
                            f"Calculated SL/TP price is zero or negative (SL={sl_price_raw}, TP={tp_price_raw})."
                        )

                    # Quantize SL/TP using market precision (rounding away from entry for SL, towards for TP?)
                    # Let's use standard rounding (ROUND_HALF_UP) for simplicity first.
                    sl_price = sl_price_raw.quantize(price_precision, rounding=ROUND_HALF_UP)
                    tp_price = tp_price_raw.quantize(price_precision, rounding=ROUND_HALF_UP)

                    # Final check: Ensure SL/TP didn't round to the same value as entry estimate or cross each other
                    if abs(sl_price - entry_price_est) < price_precision / 2:
                        logger.warning(
                            f"SL price {sl_price} too close to entry estimate {entry_price_est} after rounding. Adjusting slightly."
                        )
                        # Adjust SL slightly further away based on side
                        sl_price = (
                            sl_price - price_precision if selected_side == SIDE_BUY else sl_price + price_precision
                        )
                    if abs(tp_price - entry_price_est) < price_precision / 2:
                        logger.warning(
                            f"TP price {tp_price} too close to entry estimate {entry_price_est} after rounding. Cannot proceed with zero TP distance."
                        )
                        raise ValueError("TP price rounded to entry price estimate.")
                    if (selected_side == SIDE_BUY and sl_price >= tp_price) or (
                        selected_side == SIDE_SELL and sl_price <= tp_price
                    ):
                        raise ValueError(
                            f"SL price ({sl_price}) crossed TP price ({tp_price}) after calculation/rounding."
                        )

                    logger.info(
                        f"Calculated SL={sl_price}, TP={tp_price} based on EntryEst={entry_price_est}, ATR={current_atr:.4f}"
                    )

                except (ValueError, InvalidOperation, KeyError, ccxt.BadSymbol) as e:
                    logger.error(
                        f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Error calculating SL/TP prices: {e}{Style.RESET_ALL}"
                    )
                    return  # Stop processing this entry signal

                # Calculate Position Size
                try:
                    balance_info = exchange.fetch_balance()
                    # Use 'total' equity for risk calculation
                    equity_str = balance_info.get("total", {}).get(USDT_SYMBOL)
                    if equity_str is None:
                        raise ValueError("Could not fetch USDT total equity from balance.")
                    equity = Decimal(str(equity_str))
                    if equity <= 0:
                        raise ValueError(f"Zero or negative equity ({equity}).")
                except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, InvalidOperation) as e:
                    logger.error(
                        f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Failed to fetch valid equity: {e}{Style.RESET_ALL}"
                    )
                    return  # Stop processing this entry signal

                # Calculate position size returns Decimals
                quantity_decimal, margin_est_decimal = calculate_position_size(
                    equity,
                    CONFIG.risk_per_trade_percentage,
                    entry_price_est,
                    sl_price,
                    CONFIG.leverage,
                    symbol,
                    exchange,
                )

                if quantity_decimal is not None and margin_est_decimal is not None:
                    # Place the risked market order (which includes SL/TP placement)
                    # Pass Decimal values to the function
                    place_risked_market_order(
                        exchange, symbol, selected_side, quantity_decimal, margin_est_decimal, sl_price, tp_price
                    )
                    # The place_risked_market_order function handles logging success/failure internally.
                    # No return needed here, let the cycle finish.
                else:
                    logger.error(
                        f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Failed to calculate valid position size or margin estimate.{Style.RESET_ALL}"
                    )
                    # Stop processing this entry signal

    elif position_side != POSITION_SIDE_NONE:
        logger.info(
            f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. No exit signal this cycle. Awaiting exchange SL/TP or next signal."
        )
        # No redundant monitoring needed as we rely on exchange-native SL/TP

    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n")


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    """Main function to run the bot."""
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v10.1.0 Initializing ({start_time_str}) ---{Style.RESET_ALL}"
    )
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment: Dual Supertrend ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Protective Wards: Exchange Native SL/TP (stopMarket) ---{Style.RESET_ALL}")

    # Config object already instantiated and validated globally (CONFIG)
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK - HANDLE WITH CARE !!! ---{Style.RESET_ALL}"
    )

    # Initialize Exchange
    exchange = initialize_exchange()
    if not exchange:
        logger.critical("Failed to initialize exchange. Spell fizzles.")
        sys.exit(1)  # Exit if exchange init fails

    # Set Leverage
    if not set_leverage(exchange, CONFIG.symbol, CONFIG.leverage):
        logger.critical(f"Failed to set leverage to {CONFIG.leverage}x for {CONFIG.symbol}. Spell cannot bind.")
        # Attempt to send SMS even if leverage fails
        send_sms_alert(f"[ScalpBot] CRITICAL: Leverage set FAILED for {CONFIG.symbol}. Bot stopped.")
        sys.exit(1)  # Exit if leverage set fails

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
            trade_logic(exchange, CONFIG.symbol, CONFIG.interval)
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
            run_bot = False  # Stop immediately
        except ccxt.NetworkError as e:
            # Log as error but continue running, assuming temporary network issue
            logger.error(f"{Fore.RED}ERROR: Network error in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
            time.sleep(CONFIG.sleep_seconds * 2)  # Longer delay for network issues
        except ccxt.RateLimitExceeded as e:
            logger.warning(
                f"{Fore.YELLOW}WARNING: Rate limit exceeded: {e}. Increasing sleep duration...{Style.RESET_ALL}"
            )
            time.sleep(CONFIG.sleep_seconds * 3)  # Longer sleep after rate limit
        except ccxt.ExchangeNotAvailable as e:
            logger.error(
                f"{Fore.RED}ERROR: Exchange not available: {e}. Retrying after longer delay...{Style.RESET_ALL}"
            )
            send_sms_alert(f"[ScalpBot] WARNING: Exchange Not Available {CONFIG.symbol}. Retrying.")
            time.sleep(CONFIG.sleep_seconds * 5)  # Much longer delay
        except ccxt.ExchangeError as e:
            # Log as error but continue running, assuming temporary exchange issue
            logger.error(f"{Fore.RED}ERROR: Exchange error in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())  # Log full traceback for exchange errors
            time.sleep(CONFIG.sleep_seconds)
        except Exception as e:
            # Catch any other unexpected error, log critically, and stop the bot
            logger.critical(
                f"{Back.RED}{Fore.WHITE}FATAL: An unexpected error occurred in the main loop: {e}{Style.RESET_ALL}"
            )
            logger.critical(traceback.format_exc())
            send_sms_alert(f"[ScalpBot] FATAL ERROR: {type(e).__name__}. Bot stopped. Check logs!")
            run_bot = False  # Stop on fatal unexpected errors

    # --- Graceful Shutdown ---
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Initiating graceful shutdown sequence...{Style.RESET_ALL}")
    try:
        logger.info("Checking for open position to close on exit...")
        # Ensure exchange object is valid and usable before attempting actions
        if exchange and exchange.check_required_credentials():  # Basic check if usable
            current_pos = get_current_position(exchange, CONFIG.symbol)
            if current_pos[SIDE_KEY] != POSITION_SIDE_NONE:
                logger.warning(
                    f"Attempting to close {current_pos[SIDE_KEY]} position ({current_pos[QTY_KEY]:.8f}) before exiting..."
                )
                # Attempt to close, log result but don't prevent shutdown if it fails
                close_result = close_position(
                    exchange, symbol=CONFIG.symbol, position_to_close=current_pos, reason="Shutdown"
                )
                if close_result:
                    logger.info("Position closed successfully during shutdown.")
                else:
                    logger.error(
                        f"{Fore.RED}Failed to close position during shutdown. Manual check required.{Style.RESET_ALL}"
                    )
                    send_sms_alert(f"[ScalpBot] Error closing position {CONFIG.symbol} on shutdown. MANUAL CHECK!")
            else:
                logger.info("No open position found to close.")
        else:
            logger.warning("Exchange object not available or authenticated for final position check.")
    except Exception as close_err:
        # Catch errors during the shutdown close attempt itself
        logger.error(
            f"{Fore.RED}Failed to check/close position during final shutdown sequence: {close_err}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert("[ScalpBot] Error during final position close check on shutdown.")

    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
