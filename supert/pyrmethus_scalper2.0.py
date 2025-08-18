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
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext
import logging
import os
import shutil  # For checking command existence
import subprocess  # For Termux API calls
import sys
import time
import traceback
from typing import Any

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style, init as colorama_init
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
POSITION_SIDE_NONE = (
    "None"  # Internal representation for no position / Bybit V5 side 'None'
)
BYBIT_SIDE_BUY = "Buy"  # Bybit V5 API side
BYBIT_SIDE_SELL = "Sell"  # Bybit V5 API side

# Order Types / Statuses / Params
ORDER_TYPE_MARKET = "market"
ORDER_TYPE_STOP_MARKET = (
    "stopMarket"  # Used for both SL and TP conditional market orders
)
# ORDER_TYPE_TAKE_PROFIT_MARKET = 'takeProfitMarket' # Deprecated in favor of stopMarket with triggerDirection
ORDER_STATUS_OPEN = "open"
ORDER_STATUS_CLOSED = "closed"
ORDER_STATUS_CANCELED = "canceled"  # Note: CCXT might use 'cancelled' or 'canceled'
ORDER_STATUS_REJECTED = "rejected"
ORDER_STATUS_EXPIRED = "expired"
PARAM_REDUCE_ONLY = "reduce_only"  # CCXT standard param name
PARAM_STOP_PRICE = "stopPrice"  # CCXT standard param name for trigger price
# PARAM_TRIGGER_PRICE = 'triggerPrice' # Often interchangeable with stopPrice in CCXT, prefer stopPrice
PARAM_TRIGGER_DIRECTION = (
    "triggerDirection"  # Bybit V5 specific for conditional orders (1=above, 2=below)
)
PARAM_CATEGORY = "category"  # Bybit V5 specific for linear/inverse

# Currencies
USDT_SYMBOL = "USDT"

# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL_STR = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_STR, logging.INFO)

# Custom Log Level for Success
SUCCESS_LEVEL = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
import logging
import sys
import os
import time
import subprocess
import shutil
import traceback
import contextlib
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import ccxt
from colorama import Fore, Style, Back, init

# Initialize Colorama for cross-platform colored output
init(autoreset=True)

# --- Custom Log Level ---
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# --- Global Constants (for readability and consistency) ---
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper() # Default to INFO, can be overridden by env
USDT_SYMBOL = "USDT"

# Position & Order Management Keys
SIDE_KEY = "side"
QTY_KEY = "amount" # Using 'amount' for consistency with CCXT order objects
ENTRY_PRICE_KEY = "entryPrice"
POSITION_SIDE_NONE = "none"
POSITION_SIDE_LONG = "long"
POSITION_SIDE_SHORT = "short"
BYBIT_SIDE_BUY = "Buy" # Bybit V5 specific
BYBIT_SIDE_SELL = "Sell" # Bybit V5 specific
SIDE_BUY = "buy" # CCXT unified
SIDE_SELL = "sell" # CCXT unified

# CCXT Order Statuses
ORDER_STATUS_CLOSED = "closed"
ORDER_STATUS_CANCELED = "canceled"
ORDER_STATUS_REJECTED = "rejected"
ORDER_STATUS_EXPIRED = "expired"

# CCXT Order Types
ORDER_TYPE_STOP_MARKET = "stop_market" # Bybit V5 specific for SL/TP

# CCXT Parameter Keys
PARAM_REDUCE_ONLY = "reduceOnly"
PARAM_STOP_PRICE = "stopPrice"
PARAM_TRIGGER_DIRECTION = "triggerDirection" # Bybit V5 specific
PARAM_CATEGORY = "category" # Bybit V5 specific

# CCXT Order/Position Info Keys
INFO_KEY = "info"
SYMBOL_KEY = "symbol"
AVG_PRICE_KEY = "avgPrice"
ID_KEY = "id"
AVERAGE_KEY = "average"
PRICE_KEY = "price"
FILLED_KEY = "filled"
COST_KEY = "cost"
STATUS_KEY = "status"

# Indicator Keys
ATR_KEY = "atr"
VOLUME_MA_KEY = "volume_ma"
LAST_VOLUME_KEY = "last_volume"
VOLUME_RATIO_KEY = "volume_ratio"
BIDS_KEY = "bids"
ASKS_KEY = "asks"
BID_ASK_RATIO_KEY = "bid_ask_ratio"
SPREAD_KEY = "spread"
BEST_BID_KEY = "best_bid"
BEST_ASK_KEY = "best_ask"
LAST_PRICE_KEY = "last"


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
else:
    # Avoid color codes if not a TTY
    logging.addLevelName(
        SUCCESS_LEVEL, "SUCCESS"
    )  # Ensure level name exists without color


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads, validates, and stores configuration parameters with arcane precision."""

    def __init__(self) -> None:
        logger.info(
            f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}"
        )
        valid = True  # Track overall validity

        # --- API Credentials (Required) ---
        self.api_key: str | None = self._get_env(
            "BYBIT_API_KEY", None, str, required=True, color=Fore.RED
        )
        self.api_secret: str | None = self._get_env(
            "BYBIT_API_SECRET", None, str, required=True, color=Fore.RED
        )
        if not self.api_key or not self.api_secret:
            valid = False

        # --- Trading Parameters ---
        self.symbol: str = self._get_env(
            "SYMBOL", "BTC/USDT:USDT", str, color=Fore.YELLOW
        )
        self.interval: str = self._get_env("INTERVAL", "1m", str, color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, int, color=Fore.YELLOW
        )
        if self.leverage <= 0:
            logger.critical(f"CRITICAL CONFIG: LEVERAGE invalid: {self.leverage}")
            valid = False
        if self.sleep_seconds <= 0:
            logger.warning(
                f"CONFIG WARNING: SLEEP_SECONDS ({self.sleep_seconds}) invalid. Setting to 1."
            )
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
                f"CRITICAL CONFIG: RISK_PER_TRADE_PERCENTAGE invalid: {self.risk_per_trade_percentage}"
            )
            valid = False
        if self.atr_stop_loss_multiplier <= 0:
            logger.warning(
                f"CONFIG WARNING: ATR_STOP_LOSS_MULTIPLIER ({self.atr_stop_loss_multiplier}) not positive."
            )
        if self.atr_take_profit_multiplier <= 0:
            logger.warning(
                f"CONFIG WARNING: ATR_TAKE_PROFIT_MULTIPLIER ({self.atr_take_profit_multiplier}) not positive."
            )
        if self.max_order_usdt_amount <= 0:
            logger.warning(
                f"CONFIG WARNING: MAX_ORDER_USDT_AMOUNT ({self.max_order_usdt_amount}) not positive."
            )
        if self.required_margin_buffer < 1:
            logger.warning(
                f"CONFIG WARNING: REQUIRED_MARGIN_BUFFER ({self.required_margin_buffer}) is less than 1. Margin checks might be ineffective."
            )

        # --- Supertrend Indicator Parameters ---
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 7, int, color=Fore.CYAN
        )
        self.st_multiplier: float = float(
            self._get_env("ST_MULTIPLIER", Decimal("2.5"), Decimal, color=Fore.CYAN)
        )  # pandas_ta needs float
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 5, int, color=Fore.CYAN
        )
        self.confirm_st_multiplier: float = float(
            self._get_env(
                "CONFIRM_ST_MULTIPLIER", Decimal("2.0"), Decimal, color=Fore.CYAN
            )
        )  # pandas_ta needs float
        if self.st_atr_length <= 0 or self.confirm_st_atr_length <= 0:
            logger.warning(
                "CONFIG WARNING: Supertrend ATR length(s) are zero or negative."
            )

        # --- Volume Analysis Parameters ---
        self.volume_ma_period: int = self._get_env(
            "VOLUME_MA_PERIOD", 20, int, color=Fore.YELLOW
        )
        self.volume_spike_threshold: Decimal = self._get_env(
            "VOLUME_SPIKE_THRESHOLD", Decimal("1.5"), Decimal, color=Fore.YELLOW
        )
        self.require_volume_spike_for_entry: bool = self._get_env(
            "REQUIRE_VOLUME_SPIKE_FOR_ENTRY", True, bool, color=Fore.YELLOW
        )
        if self.volume_ma_period <= 0:
            logger.warning("CONFIG WARNING: VOLUME_MA_PERIOD is zero or negative.")

        # --- Order Book Analysis Parameters ---
        self.order_book_depth: int = self._get_env(
            "ORDER_BOOK_DEPTH", 10, int, color=Fore.YELLOW
        )
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG",
            Decimal("1.2"),
            Decimal,
            color=Fore.YELLOW,
        )
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT",
            Decimal("0.8"),
            Decimal,
            color=Fore.YELLOW,
        )
        self.fetch_order_book_per_cycle: bool = self._get_env(
            "FETCH_ORDER_BOOK_PER_CYCLE", False, bool, color=Fore.YELLOW
        )
        self.use_ob_confirm: bool = self._get_env(
            "USE_OB_CONFIRM", True, bool, color=Fore.YELLOW
        )  # Added explicit OB confirmation flag

        # --- ATR Calculation Parameter (for SL/TP) ---
        self.atr_calculation_period: int = self._get_env(
            "ATR_CALCULATION_PERIOD", 14, int, color=Fore.GREEN
        )
        if self.atr_calculation_period <= 0:
            logger.warning(
                "CONFIG WARNING: ATR_CALCULATION_PERIOD is zero or negative."
            )

        # --- Termux SMS Alert Configuration ---
        self.enable_sms_alerts: bool = self._get_env(
            "ENABLE_SMS_ALERTS", False, bool, color=Fore.MAGENTA
        )
        self.sms_recipient_number: str | None = self._get_env(
            "SMS_RECIPIENT_NUMBER", None, str, color=Fore.MAGENTA
        )
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, int, color=Fore.MAGENTA
        )
        if self.enable_sms_alerts and not self.sms_recipient_number:
            logger.warning(
                "CONFIG WARNING: SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set."
            )

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = self._get_env(
            "RECV_WINDOW", 10000, int, color=Fore.WHITE
        )
        self.order_book_fetch_limit: int = max(
            25, self.order_book_depth
        )  # Ensure sufficient depth fetched
