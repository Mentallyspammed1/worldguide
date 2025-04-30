# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
# fmt: off
#  ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗███████╗██╗   ██╗███████╗
#  ██╔══██╗██║   ██║██╔════╝████╗ ████║██╔════╝██╔════╝██║   ██║██╔════╝
#  ██████╔╝██║   ██║███████╗██╔████╔██║███████╗███████╗██║   ██║███████╗
#  ██╔═══╝ ██║   ██║╚════██║██║╚██╔╝██║╚════██║╚════██║██║   ██║╚════██║
#  ██║     ╚██████╔╝███████║██║ ╚═╝ ██║███████║███████║╚██████╔╝███████║
#  ╚═╝      ╚═════╝ ╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝
#                  <> Termux Trading Spell <>
# Pyrmethus v2.3.2 - Precision, V5 API, TP/Exits, Robustness, Rich UI
# fmt: on
# pylint: enable=line-too-long
"""
Pyrmethus - Termux Trading Spell (v2.3.2 - Enhanced)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT, enhanced with Rich for a clearer presentation.

Features:
- Robust configuration loading and validation from .env file.
- Multi-condition signal generation (EMAs, Stochastic, ATR, Trend Filter).
- Position-based Stop Loss, Take Profit, and Trailing Stop Loss management.
- Signal-based exit mechanism (EMA crossover against position).
- Enhanced error handling with retries and specific exception management.
- High-precision calculations using the Decimal type.
- Trade journaling to CSV file.
- Termux notifications for key events.
- Graceful shutdown handling SIGINT/SIGTERM.
- Improved user interface using the Rich library.
"""

# Standard Library Imports (alphabetical)
import copy
import csv
import logging
import os
import platform
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from decimal import (ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP, Decimal,
                     DivisionByZero, InvalidOperation, getcontext)
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports (alphabetical)
try:
    import ccxt
    import numpy as np
    import pandas as pd
    import requests
    from colorama import Back, Fore, Style, init as colorama_init # Use full name to avoid clash
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.logging import RichHandler # Use Rich for logging output
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    # Explicitly list common required packages for the error message
    COMMON_PACKAGES = ['ccxt', 'python-dotenv', 'pandas', 'numpy', 'colorama', 'requests', 'rich']
except ImportError as e:
    # Provide specific guidance for Termux users or general pip install
    colorama_init(autoreset=True) # Initialize colorama for error message
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell:")
    if os.getenv("TERMUX_VERSION"):
         print(f"{Style.BRIGHT}pkg install python python-cryptography openssl fftw libzmq freetype libpng pkg-config && pip install --no-binary :all: numpy pandas && pip install {missing_pkg}{Style.RESET_ALL}")
         print(f"{Fore.YELLOW}Note: Numpy/Pandas often require specific build steps in Termux.")
    else:
         print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
    if os.getenv("TERMUX_VERSION"):
         # Advise separate steps for Termux due to potential build issues
         print(f"{Style.BRIGHT}pkg install python python-cryptography openssl fftw libzmq freetype libpng pkg-config")
         print(f"{Style.BRIGHT}pip install --no-binary :all: numpy pandas")
         print(f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}")
         print(f"{Fore.YELLOW}Note: Installing numpy/pandas/matplotlib via pip in Termux might require building. Consider 'pkg install python-numpy python-pandas' if pip fails.")
    else:
         print(f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}")
    sys.exit(1) # pylint: disable=consider-using-sys-exit

# Initialize Colorama (still useful for specific inline coloring if needed)
colorama_init(autoreset=True)
# Create a Rich Console instance for enhanced terminal output
console = Console()

# Set Decimal precision context for the entire application
# Using 50 provides ample precision for crypto calculations, preventing floating-point errors.
getcontext().prec = 50

# --- Arcane Configuration & Logging Setup ---
logger = logging.getLogger("pyrmethus") # Use a specific name for the logger

# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5
if not hasattr(logging.Logger, 'trade'):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")
    def trade_log(self, message, *args, **kws):
        """Custom logging method for trade-related events."""
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args, **kws)
    logging.Logger.trade = trade_log

# Configure Logging using RichHandler for console output
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

# Ensure handlers are not duplicated if the script is reloaded
if not logger.handlers:
    # Rich Handler for Console Output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False, # Keep it concise for trading logs
        markup=True,     # Enable Rich markup in log messages
        log_time_format="[%Y-%m-%d %H:%M:%S]",
        level=log_level
    )
    # Customize level styles for RichHandler
    rich_handler.level_styles["trade"] = "bold bright_blue" # Style for TRADE level
    logger.addHandler(rich_handler)

    # Optional: Add File Handler if needed (example)
    # file_handler = logging.FileHandler("pyrmethus.log")
    # file_formatter = logging.Formatter(
    #     "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s"
    # )
    # file_handler.setFormatter(file_formatter)
    # logger.addHandler(file_handler)

logger.propagate = False # Prevent duplicate messages if root logger is configured


class TradingConfig:
    """
    Holds the sacred parameters of our spell, loading from environment variables,
    performing validation, and determining API category. Immutable after creation.
    """
    # pylint: disable=too-many-instance-attributes # Config class naturally has many attributes
    def __init__(self):
        """Initializes configuration by loading and validating environment variables."""
        logger.debug("Loading configuration from environment variables...")
        # --- Core Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW)
        self.market_type: str = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW, allowed_values=['linear', 'inverse']).lower() # Removed 'swap' as it's usually linear/inverse
        self.bybit_v5_category: str = self._determine_v5_category()
        self.interval: str = self._get_env("INTERVAL", "1m", Fore.YELLOW)

        # --- Risk & Position Management (Decimal for precision) ---
        self.risk_percentage: Decimal = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5"))
        self.sl_atr_multiplier: Decimal = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0"))
        self.tp_atr_multiplier: Decimal = self._get_env("TP_ATR_MULTIPLIER", "3.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.0"), max_val=Decimal("50.0")) # TP target (0 disables TP)
        self.tsl_activation_atr_multiplier: Decimal = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")) # ATR distance before TSL activates
        self.trailing_stop_percent: Decimal = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.01"), max_val=Decimal("10.0")) # Percentage for TSL trail distance

        # --- Order Trigger Types ---
        self.sl_trigger_by: str = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by: str = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]) # Note: Bybit TSL might only support LastPrice

        # --- Indicator Periods (Integers) ---
        self.trend_ema_period: int = self._get_env("TREND_EMA_PERIOD", "12", Fore.YELLOW, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period: int = self._get_env("FAST_EMA_PERIOD", "9", Fore.YELLOW, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period: int = self._get_env("SLOW_EMA_PERIOD", "21", Fore.YELLOW, cast_type=int, min_val=2, max_val=500)
        self.stoch_period: int = self._get_env("STOCH_PERIOD", "7", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k: int = self._get_env("STOCH_SMOOTH_K", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d: int = self._get_env("STOCH_SMOOTH_D", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.atr_period: int = self._get_env("ATR_PERIOD", "5", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)

        # --- Signal Logic Thresholds (Decimal) ---
        self.stoch_oversold_threshold: Decimal = self._get_env("STOCH_OVERSOLD_THRESHOLD", "30", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold: Decimal = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", "70", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent: Decimal = self._get_env("TREND_FILTER_BUFFER_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5")) # % buffer around trend EMA
        self.atr_move_filter_multiplier: Decimal = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5")) # Min price move as ATR multiple

        # --- Precision & Zero Checks ---
        # Define a small epsilon based roughly on typical minimum order sizes or price fluctuations.
        # This needs careful consideration based on the specific market (BTC vs SHIB).
        # For now, a fixed small value, but ideally derived from market info later.
        self.position_qty_epsilon: Decimal = Decimal("1E-12")
        logger.debug(f"Using fixed position_qty_epsilon for zero checks: {self.position_qty_epsilon:.1E}")

        # --- API Credentials ---
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED)

        # --- Operational Parameters (Integers) ---
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000) # Candles to fetch
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5) # Pause between cycles
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1) # Wait after order before checking
        self.order_check_timeout_seconds: int = self._get_env("ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5) # Max time to wait for order fill status
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10) # Retries for API calls
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool) # Enable/disable trend filter

        # --- Journaling Configuration ---
        self.journal_file_path: str = self._get_env("JOURNAL_FILE_PATH", "pyrmethus_trading_journal.csv", Fore.YELLOW)
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", "True", Fore.YELLOW, cast_type=bool)

        # --- Critical Validations ---
        if not self.api_key or not self.api_secret:
            logger.critical("[bold red]BYBIT_API_KEY or BYBIT_API_SECRET not found in environment. Halting spell.[/]")
            sys.exit(1) # pylint: disable=consider-using-sys-exit

        # --- Post-Load Validations & Warnings ---
        self._validate_config()
        logger.info("[bold green]Configuration loaded and validated successfully.[/]")

    def _determine_v5_category(self) -> str:
        """Determines the Bybit V5 API category based on symbol and market_type."""
        try:
            # Improved parsing: Handles BASE/QUOTE and BASE/QUOTE:SETTLE
            symbol_parts = self.symbol.replace(':', '/').split('/')
            if len(symbol_parts) < 2:
                raise ValueError("Symbol format must be BASE/QUOTE or BASE/QUOTE:SETTLE")

            # base_curr = symbol_parts[0].upper() # Not strictly needed for category logic below
            settle_curr = symbol_parts[-1].upper() # The last part is the settlement currency

            category = ''
            if self.market_type == 'inverse':
                # Inverse contracts are always settled in the base currency (e.g., BTC/USD settled in BTC)
                category = 'inverse'
                # Optional: Add check if settle_curr matches base_curr for inverse
                # if settle_curr != base_curr: logger.warning(...)
            elif self.market_type == 'linear':
                # Linear contracts are settled in the quote currency (e.g., BTC/USDT settled in USDT)
                category = 'linear'
                # Optional: Add check if settle_curr matches quote_curr for linear
                # quote_curr = symbol_parts[1].upper() if len(symbol_parts) > 1 else None
                # if quote_curr and settle_curr != quote_curr: logger.warning(...)
            else:
                raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}'. Use 'linear' or 'inverse'.")

            logger.info(f"Determined Bybit V5 API category: '[bold white]{category}[/]' for symbol '[white]{self.symbol}[/]' (Market Type: [white]{self.market_type}[/])")
            return category

        except (ValueError, IndexError) as e:
            logger.critical(f"[bold red]Could not determine V5 category from symbol '[white]{self.symbol}[/]' and market_type '[white]{self.market_type}[/]': {e}. Halting.[/]", exc_info=True)
            sys.exit(1) # pylint: disable=consider-using-sys-exit

    def _validate_config(self):
        """Performs post-load validation of configuration parameters, logging warnings or critical errors."""
        errors = []
        warnings = []

        if self.fast_ema_period >= self.slow_ema_period:
            errors.append(f"FAST_EMA ({self.fast_ema_period}) must be less than SLOW_EMA ({self.slow_ema_period}).")
        if self.trend_ema_period <= self.slow_ema_period:
             warnings.append(f"TREND_EMA ({self.trend_ema_period}) is <= SLOW_EMA ({self.slow_ema_period}). Trend filter might react slower than exit signals.")
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            errors.append(f"STOCH_OVERSOLD ({self.stoch_oversold_threshold.normalize()}) must be less than STOCH_OVERBOUGHT ({self.stoch_overbought_threshold.normalize()}).")
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            warnings.append(f"TSL_ACTIVATION_ATR_MULT ({self.tsl_activation_atr_multiplier.normalize()}) is less than SL_ATR_MULT ({self.sl_atr_multiplier.normalize()}). Trailing Stop Loss might activate very close to initial Stop Loss.")
        # Check if TP is enabled (non-zero) and if it's less than or equal to SL multiplier (poor risk:reward)
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            warnings.append(f"TP_ATR_MULT ({self.tp_atr_multiplier.normalize()}) is less than or equal to SL_ATR_MULT ({self.sl_atr_multiplier.normalize()}). This suggests a poor Risk:Reward ratio setup.")
        if self.risk_percentage > Decimal("0.1"): # Warn if risk per trade is high
            warnings.append(f"RISK_PERCENTAGE ({self.risk_percentage:.2%}) is high (> 10%). Ensure this is intended.")

        for warning in warnings:
            logger.warning(f"[yellow]Config Warning:[/yellow] {warning}")

        if errors:
            logger.critical("[bold red]Configuration errors detected:[/]")
            for error in errors:
                logger.critical(f"[red]- {error}[/]")
            logger.critical("[bold red]Halting spell due to configuration errors.[/]")
            sys.exit(1) # pylint: disable=consider-using-sys-exit

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def _get_env(self, key: str, default: Any, color: str, cast_type: type = str, min_val: Optional[Union[int, Decimal, float]] = None, max_val: Optional[Union[int, Decimal, float]] = None, allowed_values: Optional[List[str]] = None) -> Any:
        """
        Gets value from environment, casts, validates, logs using Rich markup, and returns the value or default.
        Handles potential errors during casting or validation.
        """
        value_str = os.getenv(key)
        log_value = "[i]****[/i]" if "SECRET" in key.upper() or "KEY" in key.upper() else value_str # Mask secrets in logs
        default_str = "[i]****[/i]" if isinstance(default, str) and ("SECRET" in key.upper() or "KEY" in key.upper()) else str(default)

        using_default = False
        if value_str is None or value_str.strip() == "":
            if default is None and key not in ['BYBIT_API_KEY', 'BYBIT_API_SECRET']: # API keys handled separately
                 logger.critical(f"[bold red]Required config '{key}' not found in environment and no default provided. Halting.[/]")
                 sys.exit(1) # pylint: disable=consider-using-sys-exit
            value_str_for_cast = str(default) if default is not None else None
            using_default = True
            if default is not None:
                 logger.warning(f"[yellow]'{key}': Not set, using default ->[/] [bright_white]'{default_str}'[/]")
            # No else needed here, handled by the check above
        else:
            logger.info(f"Found [bold cyan]'{key}'[/]: [bright_white]'{log_value}'[/]")
            value_str_for_cast = value_str

        if value_str_for_cast is None: # Should only happen if default was None and env var was missing
            logger.critical(f"[bold red]Config '{key}' is missing and has no default value. Halting.[/]")
            sys.exit(1) # pylint: disable=consider-using-sys-exit

        # Attempt casting
        casted_value = None
        cast_success = False
        try:
            if cast_type == bool:
                casted_value = str(value_str_for_cast).strip().lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                casted_value = Decimal(str(value_str_for_cast).strip())
            elif cast_type == int:
                # Cast to Decimal first to handle potential float strings like "10.0"
                casted_value = int(Decimal(str(value_str_for_cast).strip()))
            elif cast_type == float:
                 casted_value = float(str(value_str_for_cast).strip())
            else: # Default to string
                casted_value = str(value_str_for_cast) # No strip here, allow whitespace if intended for string type
            cast_success = True
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"[red]Cast failed for '{key}' (Value: '{value_str_for_cast}', Type: {cast_type.__name__}): {e}.[/]")
            # Fallback to default if casting failed (and default exists)
            if default is not None:
                logger.warning(f"[yellow]Falling back to default value for '{key}':[/] [bright_white]'{default_str}'[/]")
                value_str_for_cast = str(default) # Use default for re-casting
                using_default = True # Mark as using default now
                try: # Re-cast default value
                    if cast_type == bool: casted_value = str(value_str_for_cast).lower() in ['true', '1', 'yes', 'y', 'on']
                    elif cast_type == Decimal: casted_value = Decimal(str(value_str_for_cast))
                    elif cast_type == int: casted_value = int(Decimal(str(value_str_for_cast)))
                    elif cast_type == float: casted_value = float(value_str_for_cast)
                    else: casted_value = cast_type(value_str_for_cast)
                    cast_success = True
                except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                     logger.critical(f"[bold red]Default value '{default_str}' for '{key}' is invalid for type {cast_type.__name__}: {cast_default_err}. Halting.[/]")
                     sys.exit(1) # pylint: disable=consider-using-sys-exit
            else:
                 logger.critical(f"[bold red]Cast failed for required config '{key}' and no default is available. Halting.[/]")
                 sys.exit(1) # pylint: disable=consider-using-sys-exit

        # Perform Validation (only if casting was successful)
        if cast_success:
            validation_failed = False
            error_message = ""

            # 1. Allowed Values Check
            if allowed_values:
                # Case-insensitive comparison for strings
                comp_value = str(casted_value).lower() if isinstance(casted_value, str) else casted_value
                lower_allowed = [str(v).lower() for v in allowed_values]
                if comp_value not in lower_allowed:
                    error_message = f"Value '[white]{casted_value}[/]' not in allowed list: {lower_allowed}."
                    validation_failed = True

            # 2. Min/Max Check (for numeric types)
            if not validation_failed and isinstance(casted_value, (Decimal, int, float)):
                try:
                    # Ensure min/max_val are comparable (cast to Decimal if needed)
                    min_val_comp = Decimal(str(min_val)) if isinstance(casted_value, Decimal) and min_val is not None else min_val
                    max_val_comp = Decimal(str(max_val)) if isinstance(casted_value, Decimal) and max_val is not None else max_val

                    if min_val_comp is not None and casted_value < min_val_comp:
                        error_message = f"Value [white]{casted_value}[/] is less than minimum allowed [white]{min_val}[/]."
                        validation_failed = True
                    if max_val_comp is not None and casted_value > max_val_comp:
                        error_message = f"Value [white]{casted_value}[/] is greater than maximum allowed [white]{max_val}[/]."
                        validation_failed = True
                except (InvalidOperation, TypeError) as e:
                    error_message = f"Min/max validation error for value '{casted_value}': {e}."
                    validation_failed = True # Treat comparison error as validation failure

            # Handle Validation Failure
            if validation_failed:
                logger.error(f"[red]Validation failed for '{key}': {error_message}[/]")
                if default is not None:
                    # If the original value failed validation, and we weren't already using the default, log fallback.
                    if not using_default:
                         logger.warning(f"[yellow]Falling back to default value for '{key}':[/] [bright_white]'{default_str}'[/]")
                    # Re-cast the default value AGAIN to ensure the returned type is correct.
                    # This handles cases where the env var was present but invalid, forcing a fallback.
                    try:
                        if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                        elif cast_type == Decimal: return Decimal(str(default))
                        elif cast_type == int: return int(Decimal(str(default)))
                        elif cast_type == float: return float(default)
                        else: return cast_type(default)
                    except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                        logger.critical(f"[bold red]Default value '{default_str}' for '{key}' is invalid on fallback: {cast_default_err}. Halting.[/]")
                        sys.exit(1) # pylint: disable=consider-using-sys-exit
                else:
                    # No default available, validation failed, must halt.
                    logger.critical(f"[bold red]Validation failed for required config '{key}' and no default is available to fall back on. Halting.[/]")
                    sys.exit(1) # pylint: disable=consider-using-sys-exit

        # If cast and validation (if applicable) passed
        return casted_value


# --- Instantiate Configuration ---
logger.info(f"[bold magenta]Initializing Arcane Configuration v2.3.2...[/]")
load_dotenv() # Load environment variables from .env file
CONFIG = TradingConfig()

# --- Global Variables ---
MARKET_INFO: Optional[Dict[str, Any]] = None # Holds market details like precision, limits
EXCHANGE: Optional[ccxt.Exchange] = None     # CCXT Exchange instance
# Tracks active SL/TP/TSL order IDs associated with a position side (long/short)
# Crucial for managing conditional orders correctly.
order_tracker: Dict[str, Dict[str, Optional[str]]] = {
    "long": {"sl_id": None, "tsl_id": None, "tp_id": None}, # Added tp_id
    "short": {"sl_id": None, "tsl_id": None, "tp_id": None}
}
shutdown_requested: bool = False # Flag for graceful shutdown sequence

# --- Signal Handling for Graceful Shutdown ---
def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) and SIGTERM for graceful shutdown."""
    # pylint: disable=unused-argument, global-statement
    global shutdown_requested
    if not shutdown_requested:
        sig_name = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
        logger.warning(f"\n[bold yellow]Signal {sig_name} received. Initiating graceful shutdown... (Press Ctrl+C again to force exit)[/]")
        shutdown_requested = True
    else:
        logger.warning("[bold red]Force exit requested. Terminating immediately.[/]")
        sys.exit(1) # Force exit on second signal

signal.signal(signal.SIGINT, signal_handler) # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals (e.g., from systemd or kill)

# --- Core Spell Functions ---

# Placeholder for fetch_with_retries - Assumed to exist and handle retries/errors
def fetch_with_retries(fetch_function, *args, max_retries=CONFIG.max_fetch_retries, delay_factor=1.5, **kwargs) -> Any:
    """Attempts a function call with retries on specific ccxt/network errors."""
    retries = 0
    while retries <= max_retries:
        if shutdown_requested:
             logger.warning(f"Shutdown requested during fetch_with_retries for {fetch_function.__name__}. Aborting fetch.")
             return None
        try:
            result = fetch_function(*args, **kwargs)
            # logger.debug(f"Fetch successful: {fetch_function.__name__}")
            return result
        except (
            ccxt.NetworkError,          # Includes DNS lookup errors, connection timeouts/refused, etc.
            ccxt.ExchangeError,         # Base class for exchange-specific errors (like rate limits)
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"[red]Max retries ({max_retries}) exceeded for {fetch_function.__name__}. Last error: {e}[/]", exc_info=True)
                return None # Indicate failure after all retries
            sleep_time = delay_factor ** retries # Exponential backoff
            logger.warning(f"[yellow]Network/Exchange error in {fetch_function.__name__}: {e}. Retrying ({retries}/{max_retries}) in {sleep_time:.2f}s...[/]")
            time.sleep(sleep_time)
        except ccxt.AuthenticationError as e:
            logger.critical(f"[bold red]Authentication Error during {fetch_function.__name__}: {e}. Check API keys. Halting.[/]", exc_info=True)
            # No retries for auth errors - needs user intervention
                  except ccxt.AuthenticationError as e:
            logger.critical(f"[bold red]Authentication Error during {fetch_function.__name__}: {e}. Check API keys. Halting.[/]", exc_info=True)                                                                                 # No retries for auth errors - needs user intervention
            global shutdown_requested  # Declare global first                      shutdown_requested = True # Then assign to trigger shutdown            # Optionally, attempt graceful shutdown immediately here as well
            # graceful_shutdown() # Call cleanup now                               return None # Indicate failure to the caller
        except ccxt.InsufficientFunds as e:
             logger.error(f"[red]Insufficient Funds during {fetch_function.__name__}: {e}[/]. Cannot proceed with this action.")
             # Usually don't retry insufficient funds immediately
             return None # Indicate failure
        except Exception as e: # Catch other unexpected errors during the fetch
            logger.error(f"[red]Unexpected error during {fetch_function.__name__}: {e}[/]", exc_info=True)
            # Decide whether to retry or fail based on the error type if needed
            return None # Indicate failure for unexpected errors
    return None # Should not be reached, but ensures a return path

# Placeholder for initialize_exchange - Assumed to exist
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """Initializes and returns the CCXT exchange instance."""
    global EXCHANGE, MARKET_INFO # Allow modification of globals
    logger.info(f"Initializing Bybit exchange connection (V5 API, Category: {CONFIG.bybit_v5_category})...")
    try:
        exchange = ccxt.bybit({
            'apiKey': CONFIG.api_key,
            'secret': CONFIG.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap', # V5 uses 'swap' for perpetuals
                'adjustForTimeDifference': True,
                'recvWindow': 10000, # Increase recvWindow if timestamp errors occur
                'brokerId': 'PYRMETHUS', # Optional: Identify your bot
                'v5': True, # Explicitly use V5 features if needed by CCXT version
                'category': CONFIG.bybit_v5_category # IMPORTANT: Set category for V5
            }
        })
        # Test connection and load markets
        logger.debug("Loading markets...")
        markets = fetch_with_retries(exchange.load_markets)
        if markets is None:
             logger.critical("[bold red]Failed to load markets. Check connection and API keys.[/]")
             return None
        logger.info(f"Loaded {len(markets)} markets from Bybit.")

        # Verify the specific market exists
        if CONFIG.symbol not in markets:
             # Try finding the market by ID if symbol format differs slightly
             found_by_id = False
             market_id_attempt = CONFIG.symbol.replace('/', '').replace(':','') # e.g., BTCUSDT
             for m_id, m_data in markets.items():
                  if m_id == market_id_attempt and m_data.get('category') == CONFIG.bybit_v5_category:
                      logger.warning(f"[yellow]Symbol '{CONFIG.symbol}' not found directly, but matched ID '{market_id_attempt}'. Using this market.[/]")
                      MARKET_INFO = m_data
                      # Update CONFIG.symbol to the official one if needed? Or just use MARKET_INFO['symbol']
                      # CONFIG.symbol = MARKET_INFO['symbol'] # Be careful with side effects
                      found_by_id = True
                      break
             if not found_by_id:
                 logger.critical(f"[bold red]Symbol '{CONFIG.symbol}' not found in loaded markets for category '{CONFIG.bybit_v5_category}'. Available symbols include: {list(markets.keys())[:10]}... Halting.[/]")
                 return None
        else:
             MARKET_INFO = markets[CONFIG.symbol]

        # Log market details
        logger.info(f"Market Details for [white]{CONFIG.symbol}[/]:")
        logger.info(f"  - Market ID: [white]{MARKET_INFO.get('id')}[/]")
        logger.info(f"  - Type: [white]{MARKET_INFO.get('type')}[/], Settle: [white]{MARKET_INFO.get('settle')}[/]")
        logger.info(f"  - Precision: Price={MARKET_INFO.get('precision', {}).get('price')}, Amount={MARKET_INFO.get('precision', {}).get('amount')}")
        logger.info(f"  - Limits: Min Amount={MARKET_INFO.get('limits', {}).get('amount', {}).get('min')}, Min Cost={MARKET_INFO.get('limits', {}).get('cost', {}).get('min')}")

        # Set precision for Decimal based on market info if possible
        price_precision = MARKET_INFO.get('precision', {}).get('price')
        if price_precision:
            # Determine decimal places from precision (e.g., 0.01 -> 2, 10 -> -1)
            try:
                 price_precision_decimal = Decimal(str(price_precision))
                 if price_precision_decimal > 0:
                      decimal_places = abs(price_precision_decimal.normalize().as_tuple().exponent)
                      # Example: Set rounding context if needed elsewhere, but formatting handles it mostly
                      # getcontext().prec = max(getcontext().prec, decimal_places + 10) # Ensure enough overall precision
                      logger.debug(f"Detected price precision: {price_precision} ({decimal_places} decimal places)")
                 else: # Handle precision like 10, 100 etc.
                      logger.debug(f"Detected price precision: {price_precision}")

            except InvalidOperation:
                 logger.warning(f"Could not parse market price precision '{price_precision}' as Decimal.")

        amount_precision = MARKET_INFO.get('precision', {}).get('amount')
        if amount_precision:
             try:
                amount_precision_decimal = Decimal(str(amount_precision))
                if amount_precision_decimal > 0:
                    decimal_places = abs(amount_precision_decimal.normalize().as_tuple().exponent)
                    logger.debug(f"Detected amount precision: {amount_precision} ({decimal_places} decimal places)")
                    # Set position_qty_epsilon based on amount precision if desired
                    # CONFIG.position_qty_epsilon = amount_precision_decimal / Decimal('10') # e.g., 1/10th of min step
                    # logger.info(f"Adjusted position_qty_epsilon based on market: {CONFIG.position_qty_epsilon}")
                else:
                    logger.debug(f"Detected amount precision: {amount_precision}")
             except InvalidOperation:
                 logger.warning(f"Could not parse market amount precision '{amount_precision}' as Decimal.")


        # Test authentication (fetch balance)
        logger.debug("Testing authenticated endpoint (fetchBalance)...")
        balance_info = fetch_with_retries(exchange.fetch_balance)
        if balance_info is None:
             logger.critical("[bold red]Failed to fetch balance. Check API key permissions and authentication.[/]")
             return None
        logger.info("[bold green]Exchange connection and authentication successful.[/]")
        EXCHANGE = exchange
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"[bold red]Authentication Error: {e}. Please check your API keys and permissions.[/]", exc_info=True)
    except ccxt.ExchangeNotAvailable as e:
        logger.critical(f"[bold red]Exchange Not Available: {e}. Bybit might be down or unreachable.[/]", exc_info=True)
    except ccxt.RequestTimeout as e:
        logger.critical(f"[bold red]Connection Timeout: {e}. Check network connection or increase recvWindow.[/]", exc_info=True)
    except Exception as e:
        logger.critical(f"[bold red]An unexpected error occurred during exchange initialization: {e}[/]", exc_info=True)
    return None

# Placeholder for termux_notify - Assumed to exist
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API, if available."""
    if platform.system() == "Linux" and os.getenv("TERMUX_VERSION"):
        try:
            subprocess.run(['termux-notification', '-t', title, '-c', content], check=True, timeout=5)
            # logger.debug(f"Termux notification sent: '{title}'")
        except FileNotFoundError:
            logger.warning("termux-notification command not found. Install Termux:API package.")
        except subprocess.TimeoutExpired:
            logger.warning("termux-notification command timed out.")
        except subprocess.CalledProcessError as e:
            logger.error(f"termux-notification failed: {e}")
        except Exception as e:
            logger.error(f"Error sending Termux notification: {e}", exc_info=True)
    # else:
        # logger.debug("Not running in Termux or non-Linux system, skipping notification.")

# --- Formatting Helpers ---
def _format_with_fallback(symbol: str, value: Union[Decimal, str, float, int], exchange_method: callable, rounding_mode=None, fallback_precision: Decimal = Decimal("1E-8"), fallback_rounding=ROUND_HALF_EVEN) -> str:
    """
    Formats a value using the exchange's method, falling back to Decimal formatting.
    """
    if EXCHANGE is None or MARKET_INFO is None:
        logger.warning("Exchange not initialized, using fallback formatting.")
        try:
            value_decimal = Decimal(str(value))
            # Simple fallback: format to a reasonable number of decimal places
            return f"{value_decimal.quantize(fallback_precision, rounding=fallback_rounding)}"
        except (InvalidOperation, TypeError):
            return str(value) # Return as string if conversion fails

    try:
        # Ensure value is float for ccxt formatting methods (which often expect float)
        value_float = float(value)
        # Use the provided ccxt formatting method (e.g., exchange.price_to_precision)
        formatted_value = exchange_method(symbol, value_float)
        # logger.debug(f"Formatted {value} using {exchange_method.__name__}: {formatted_value}")
        return formatted_value
    except (TypeError, ValueError, InvalidOperation) as e:
        logger.warning(f"CCXT formatting ({exchange_method.__name__}) failed for value '{value}' ({type(value)}): {e}. Using Decimal fallback.")
    except Exception as e: # Catch potential errors within ccxt formatting
        logger.error(f"Unexpected error during CCXT formatting ({exchange_method.__name__}): {e}. Using Decimal fallback.", exc_info=True)

    # Fallback using Decimal quantize if ccxt method fails
    try:
        value_decimal = Decimal(str(value))
        precision_str = None
        if exchange_method == EXCHANGE.price_to_precision:
            precision_str = MARKET_INFO.get('precision', {}).get('price')
        elif exchange_method == EXCHANGE.amount_to_precision:
            precision_str = MARKET_INFO.get('precision', {}).get('amount')

        if precision_str:
            precision_decimal = Decimal(str(precision_str))
            rounding = rounding_mode if rounding_mode else fallback_rounding
            # logger.debug(f"Fallback formatting {value_decimal} to precision {precision_decimal} with mode {rounding}")
            return f"{value_decimal.quantize(precision_decimal, rounding=rounding)}"
        else:
            # logger.debug(f"Fallback formatting {value_decimal} to default precision {fallback_precision}")
            return f"{value_decimal.quantize(fallback_precision, rounding=fallback_rounding)}"
    except (InvalidOperation, TypeError, KeyError) as e:
         logger.error(f"Fallback Decimal formatting failed for value '{value}': {e}")
         return str(value) # Last resort: return raw string representation

def format_price(symbol: str, price: Union[Decimal, str, float, int]) -> str:
    """Formats a price value according to the market's precision rules."""
    if EXCHANGE:
        # Bybit typically truncates (rounds down) prices for orders, use ROUND_DOWN
        return _format_with_fallback(symbol, price, EXCHANGE.price_to_precision, rounding_mode=ROUND_DOWN)
    return str(price) # Fallback if exchange not ready

def format_amount(symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
    """Formats an amount value according to the market's precision rules."""
    if EXCHANGE:
        # Bybit typically truncates (rounds down) amounts for orders
        return _format_with_fallback(symbol, amount, EXCHANGE.amount_to_precision, rounding_mode=rounding_mode)
    return str(amount) # Fallback if exchange not ready


# Placeholder for fetch_market_data - Assumed to exist
def fetch_market_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches OHLCV data and returns it as a Pandas DataFrame."""
    if not EXCHANGE: logger.error("Cannot fetch market data: Exchange not initialized."); return None
    logger.debug(f"Fetching {limit} candles for {symbol} ({timeframe})...")
    try:
        ohlcv = fetch_with_retries(EXCHANGE.fetch_ohlcv, symbol, timeframe, limit=limit)
        if ohlcv is None: # fetch_with_retries returns None on failure
             logger.error(f"Failed to fetch OHLCV data for {symbol} after retries.")
             return None
        if not ohlcv: # Empty list returned
            logger.warning(f"No OHLCV data returned for {symbol} ({timeframe}).")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal for precision
        for col in ['open', 'high', 'low', 'close', 'volume']:
            # Handle potential None values or non-numeric strings gracefully
            df[col] = df[col].apply(lambda x: Decimal(str(x)) if x is not None else Decimal('NaN'))

        logger.info(f"Fetched {len(df)} candles for {symbol}. Latest: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        # logger.debug(f"DataFrame tail:\n{df.tail(3)}")
        return df

    except Exception as e:
        logger.error(f"[red]Error processing OHLCV data for {symbol}: {e}[/]", exc_info=True)
        return None

# Placeholder for calculate_indicators - Assumed to exist
def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
    """Calculates technical indicators based on the OHLCV DataFrame."""
    if df.empty: logger.warning("Cannot calculate indicators: DataFrame is empty."); return None
    required_length = max(CONFIG.trend_ema_period, CONFIG.slow_ema_period, CONFIG.stoch_period, CONFIG.atr_period) + 5 # Add buffer
    if len(df) < required_length: logger.warning(f"Not enough data ({len(df)}) to calculate all indicators (need ~{required_length}). Results may be inaccurate."); # return None # Optional: Return None if not enough data

    indicators = {}
    try:
        # --- EMAs ---
        # Use adjust=False for standard EMA calculation matching most platforms
        indicators['trend_ema'] = df['close'].ewm(span=CONFIG.trend_ema_period, adjust=False).mean().iloc[-1]
        indicators['fast_ema'] = df['close'].ewm(span=CONFIG.fast_ema_period, adjust=False).mean().iloc[-1]
        indicators['slow_ema'] = df['close'].ewm(span=CONFIG.slow_ema_period, adjust=False).mean().iloc[-1]

        # --- Stochastic Oscillator ---
        low_min = df['low'].rolling(window=CONFIG.stoch_period).min()
        high_max = df['high'].rolling(window=CONFIG.stoch_period).max()
        stoch_k_raw = ((df['close'] - low_min) / (high_max - low_min)) * 100
        # Handle division by zero (high_max == low_min) -> Stoch is 50? or 0? Check platform behavior. Using 50.
        stoch_k_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        stoch_k_raw.fillna(50, inplace=True) # Fill NaNs resulting from division by zero or initial periods

        indicators['stoch_k'] = stoch_k_raw.rolling(window=CONFIG.stoch_smooth_k).mean().iloc[-1]
        indicators['stoch_d'] = indicators['stoch_k'].rolling(window=CONFIG.stoch_smooth_d).mean().iloc[-1] # Use already smoothed %K for %D

        # --- ATR (Average True Range) ---
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Use EMA for ATR calculation (common method)
        indicators['atr'] = tr.ewm(alpha=1/CONFIG.atr_period, adjust=False).mean().iloc[-1]

        # --- Convert final indicator values to Decimal ---
        for key, value in indicators.items():
            if pd.isna(value):
                 logger.warning(f"Indicator '{key}' calculated as NaN. Check data or period settings.")
                 indicators[key] = Decimal('NaN')
            else:
                 try: indicators[key] = Decimal(str(value))
                 except InvalidOperation: logger.error(f"Failed to convert indicator '{key}' value '{value}' to Decimal."); indicators[key] = Decimal('NaN')

        # logger.debug(f"Calculated Indicators: { {k: f'{v:.4f}' if isinstance(v, Decimal) and not v.is_nan() else v for k, v in indicators.items()} }")
        return indicators

    except Exception as e:
        logger.error(f"[red]Error calculating indicators: {e}[/]", exc_info=True)
        return None

# Placeholder for get_current_position - Assumed to exist
def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetches current position details for the given symbol."""
    if not EXCHANGE: logger.error("Cannot get position: Exchange not initialized."); return None
    logger.debug(f"Fetching position for {symbol}...")
    try:
        params = {'category': CONFIG.bybit_v5_category, 'symbol': MARKET_INFO['id']}
        positions_data = fetch_with_retries(EXCHANGE.fetch_positions, params=params) # V5 needs symbol

        if positions_data is None: # fetch_with_retries failed
            logger.error(f"Failed to fetch positions for {symbol} after retries.")
            return None

        position_details = {"long": {"qty": Decimal(0), "entry_price": Decimal(0), "liq_price": Decimal(0), "pnl": Decimal(0)},
                            "short": {"qty": Decimal(0), "entry_price": Decimal(0), "liq_price": Decimal(0), "pnl": Decimal(0)}}

        if not positions_data:
             logger.debug(f"No open positions reported for {symbol}.")
             return position_details # Return zeroed details if no position

        # Bybit V5 fetch_positions usually returns a list, find the matching symbol
        target_pos = None
        for pos in positions_data:
            if pos.get('info', {}).get('symbol') == MARKET_INFO['id']:
                 target_pos = pos
                 break

        if not target_pos:
            logger.debug(f"No position data found specifically for {MARKET_INFO['id']} in the returned list.")
            return position_details

        # --- Process Position Data (V5 Structure) ---
        info = target_pos.get('info', {})
        side = str(info.get('side', 'None')).lower() # 'Buy' -> long, 'Sell' -> short, 'None' -> flat
        size_str = info.get('size', '0')
        entry_price_str = info.get('avgPrice', info.get('entryPrice', '0')) # Prefer avgPrice if available
        liq_price_str = info.get('liqPrice', '0')
        pnl_str = info.get('unrealisedPnl', '0')

        try:
            qty = Decimal(size_str)
            entry_price = Decimal(entry_price_str) if entry_price_str and Decimal(entry_price_str) > 0 else Decimal('NaN')
            liq_price = Decimal(liq_price_str) if liq_price_str and Decimal(liq_price_str) > 0 else Decimal('NaN')
            pnl = Decimal(pnl_str) if pnl_str else Decimal('0')

            if qty.copy_abs() < CONFIG.position_qty_epsilon: # Treat negligible amounts as flat
                 logger.debug(f"Position size {qty} is negligible, considering flat.")
                 return position_details # Return zeroed

            if side == 'buy':
                position_details["long"]["qty"] = qty
                position_details["long"]["entry_price"] = entry_price
                position_details["long"]["liq_price"] = liq_price
                position_details["long"]["pnl"] = pnl
                logger.debug(f"Parsed LONG position: Qty={qty}, Entry={entry_price}, Liq={liq_price}, PNL={pnl}")
            elif side == 'sell':
                 # Bybit reports short quantity as positive, store it as negative internally for consistency?
                 # Or keep positive and rely on the "short" key? Let's keep it positive as reported.
                position_details["short"]["qty"] = qty # Store positive qty
                position_details["short"]["entry_price"] = entry_price
                position_details["short"]["liq_price"] = liq_price
                position_details["short"]["pnl"] = pnl
                logger.debug(f"Parsed SHORT position: Qty={qty}, Entry={entry_price}, Liq={liq_price}, PNL={pnl}")
            else: # Side is 'None' or unexpected value
                 logger.debug(f"Position side reported as '{side}', considering flat.")
                 return position_details # Return zeroed

        except (InvalidOperation, TypeError) as e:
             logger.error(f"[red]Error parsing position data fields: {e}. Data: {info}[/]", exc_info=True)
             return None # Indicate failure to parse

        return position_details

    except Exception as e:
        logger.error(f"[red]An unexpected error occurred fetching/processing position for {symbol}: {e}[/]", exc_info=True)
        return None

# Placeholder for get_balance - Assumed to exist
def get_balance(currency: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Fetches total and available balance for a specific currency (Wallet Balance)."""
    if not EXCHANGE: logger.error("Cannot get balance: Exchange not initialized."); return None, None
    logger.debug(f"Fetching balance for {currency}...")
    try:
        # V5 uses fetchBalance with specific account type parameters if needed,
        # but default usually works for UNIFIED account.
        params = {'accountType': 'UNIFIED'} # Or 'CONTRACT' if using non-unified
        balance_data = fetch_with_retries(EXCHANGE.fetch_balance, params=params)
        if balance_data is None:
            logger.error(f"Failed to fetch balance after retries.")
            return None, None

        # --- Parse Balance Data (V5 Structure) ---
        # The structure can vary. Often need to look into balance_data['info']['result']['list']
        total_equity = None
        available_balance = None

        # Look for Unified Account balance summary first
        if 'info' in balance_data and 'result' in balance_data['info'] and 'list' in balance_data['info']['result']:
             account_list = balance_data['info']['result']['list']
             if account_list:
                 # For UNIFIED, often the first item contains the overall equity
                 unified_summary = account_list[0]
                 if unified_summary.get('accountType') == 'UNIFIED':
                      # totalEquity = total equity of the unified account in USD/USDT
                      # totalWalletBalance = total value of assets held in USD/USDT
                      # totalAvailableBalance = equity available to open new positions in USD/USDT
                      equity_str = unified_summary.get('totalEquity')
                      avail_str = unified_summary.get('totalAvailableBalance')
                      # Sometimes specific coin balance is needed
                      coin_list = unified_summary.get('coin', [])
                      for coin_info in coin_list:
                          if coin_info.get('coin') == currency:
                              # Use coin-specific equity/available if needed, but usually totalEquity is used for risk calc
                              # equity_str = coin_info.get('equity') # Equity specific to this coin
                              # avail_str = coin_info.get('availableToWithdraw') # Check V5 docs for best 'available' field
                              pass # Stick with total equity for now

                      try:
                          if equity_str is not None: total_equity = Decimal(str(equity_str))
                          if avail_str is not None: available_balance = Decimal(str(avail_str))
                          logger.debug(f"Parsed UNIFIED balance: Total Equity={total_equity}, Available={available_balance}")

                      except (InvalidOperation, TypeError) as e:
                          logger.error(f"[red]Error parsing unified balance data fields: {e}. Data: {unified_summary}[/]", exc_info=True)
                          return None, None # Indicate failure

        # Fallback or alternative: Check the standard CCXT balance structure if parsing 'info' fails
        if total_equity is None:
             if currency in balance_data['total']:
                 total_equity = Decimal(str(balance_data['total'][currency]))
                 available_balance = Decimal(str(balance_data['free'][currency]))
                 logger.debug(f"Parsed standard CCXT balance: Total={total_equity}, Available={available_balance}")
             else:
                 logger.warning(f"Currency '{currency}' not found in balance data: {balance_data.get('total', {}).keys()}")
                 return None, None # Currency not found

        if total_equity is None: # If still None after checks
            logger.error(f"Could not determine balance for {currency}. Balance data: {balance_data}")
            return None, None

        return total_equity, available_balance

    except Exception as e:
        logger.error(f"[red]An unexpected error occurred fetching/processing balance for {currency}: {e}[/]", exc_info=True)
        return None, None

# Placeholder for check_order_status - Assumed to exist
def check_order_status(order_id: str, symbol: str, timeout: int) -> Optional[Dict]:
    """Checks the status of an order until filled, canceled, or timeout."""
    if not EXCHANGE: logger.error("Cannot check order status: Exchange not initialized."); return None
    start_time = time.time()
    logger.debug(f"Checking status for order {order_id} (Symbol: {symbol})...")

    while time.time() - start_time < timeout:
        if shutdown_requested: logger.warning("Shutdown requested during order status check."); return None
        try:
            # V5 fetch_order requires market ID
            params = {'category': CONFIG.bybit_v5_category}
            order = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol=symbol, params=params)

            if order is None: # fetch_with_retries failed or order not found immediately
                logger.warning(f"fetch_order failed or order {order_id} not found yet. Retrying check...")
                time.sleep(CONFIG.order_check_delay_seconds) # Wait before next check
                continue

            status = order.get('status')
            logger.debug(f"Order {order_id} Status: {status}, Info: {order.get('info', {}).get('orderStatus')}") # Log raw status too

            if status == 'closed' or status == 'filled': # 'closed' used by some exchanges for fully filled
                logger.info(f"[green]Order {order_id} confirmed Filled/Closed.[/]")
                return order
            elif status == 'canceled' or status == 'rejected' or status == 'expired':
                logger.warning(f"[yellow]Order {order_id} is {status}.[/]")
                return order # Return the order details even if not filled
            elif status == 'open':
                 logger.debug(f"Order {order_id} is still open. Waiting...")
            else: # Partial fills, unknown status
                 logger.info(f"Order {order_id} status: {status}. Continuing check...")

            time.sleep(CONFIG.order_check_delay_seconds) # Wait before next check

        except ccxt.OrderNotFound:
            logger.warning(f"Order {order_id} not found by exchange (may take time to propagate). Retrying check...")
            time.sleep(CONFIG.order_check_delay_seconds * 2) # Longer wait if not found
        except Exception as e:
            logger.error(f"[red]Error checking status for order {order_id}: {e}[/]", exc_info=True)
            # Decide whether to continue checking or return None based on error
            time.sleep(CONFIG.order_check_delay_seconds) # Wait before potentially retrying

    logger.warning(f"[yellow]Timeout ({timeout}s) reached checking status for order {order_id}. Last known status might be inaccurate.[/]")
    # Try one last fetch after timeout
    try:
         params = {'category': CONFIG.bybit_v5_category}
         final_order = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol=symbol, params=params)
         if final_order: logger.info(f"Final status check for {order_id}: {final_order.get('status')}")
         return final_order
    except Exception as e:
         logger.error(f"Error during final status check for {order_id}: {e}")
         return None # Indicate failure or uncertain status


# Placeholder for log_trade_entry_to_journal - Assumed to exist
def log_trade_entry_to_journal(symbol: str, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]) -> None:
    """Logs trade entry details to a CSV journal file."""
    if not CONFIG.enable_journaling: return
    file_exists = os.path.isfile(CONFIG.journal_file_path)
    try:
        with open(CONFIG.journal_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp_utc', 'symbol', 'side', 'quantity', 'entry_price', 'order_id', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader() # Write header only if file is new

            writer.writerow({
                'timestamp_utc': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'side': side.upper(),
                'quantity': f"{qty:.8f}", # Format Decimal to string
                'entry_price': f"{avg_price:.8f}" if avg_price else 'N/A',
                'order_id': order_id if order_id else 'N/A',
                'status': 'ENTRY' # Add a status column
            })
        # logger.debug(f"Logged ENTRY to journal: {side.upper()} {qty} {symbol} @ {avg_price}")
    except IOError as e:
        logger.error(f"[red]Error writing entry to journal file '{CONFIG.journal_file_path}': {e}[/]")
    except Exception as e:
        logger.error(f"[red]Unexpected error writing entry to journal: {e}[/]", exc_info=True)

# Placeholder for _calculate_trade_parameters - Assumed to exist and handle calculations
def _calculate_trade_parameters(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal, total_equity: Decimal) -> Optional[Dict[str, Decimal]]:
    """
    Calculates position size, SL price based on risk % and ATR.
    Returns None if calculation is not possible.
    """
    if not EXCHANGE or not MARKET_INFO: logger.error("Cannot calculate params: Exchange/Market info missing."); return None
    if atr.is_nan() or atr <= 0: logger.error("Cannot calculate params: Invalid ATR value."); return None
    if total_equity.is_nan() or total_equity <= 0: logger.error("Cannot calculate params: Invalid total equity."); return None

    logger.debug(f"Calculating trade parameters: Side={side}, Risk={risk_percentage:.4%}, ATR={atr:.5f}, Equity={total_equity:.2f}")

    try:
        # 1. Get Current Price (use ticker for more recent price)
        ticker = fetch_with_retries(EXCHANGE.fetch_ticker, symbol)
        if not ticker or 'last' not in ticker or ticker['last'] is None:
            logger.error("Failed to fetch current ticker price for calculation.")
            # Fallback: Use last close from OHLCV if needed, but less ideal
            # df = fetch_market_data(symbol, CONFIG.interval, 2) # Fetch minimal data
            # if df is not None and not df.empty: current_price = df['close'].iloc[-1]
            # else: return None
            return None # Require ticker for accurate calculation
        current_price = Decimal(str(ticker['last']))
        logger.debug(f"Current price from ticker: {current_price:.5f}")

        # 2. Calculate Stop Loss Distance & Price
        sl_distance = atr * CONFIG.sl_atr_multiplier
        sl_price = Decimal(0)
        if side == 'buy':
            sl_price = current_price - sl_distance
        elif side == 'sell':
            sl_price = current_price + sl_distance
        else:
            logger.error(f"Invalid side '{side}' for calculation.")
            return None

        # Ensure SL price is positive
        if sl_price <= 0:
             logger.error(f"Calculated SL price ({sl_price:.5f}) is zero or negative. Cannot proceed. Check ATR/Multiplier/Price.")
             return None

        logger.debug(f"Calculated SL distance: {sl_distance:.5f}, Raw SL Price: {sl_price:.5f}")

        # 3. Calculate Risk Amount in Quote Currency
        risk_amount = total_equity * risk_percentage
        logger.debug(f"Risk Amount ({risk_percentage:.2%}) of Equity ({total_equity:.2f}): {risk_amount:.5f} {MARKET_INFO.get('settle', 'USD')}")

        # 4. Calculate Position Size
        price_difference = abs(current_price - sl_price)
        if price_difference <= 0:
            logger.error("Price difference for position size calculation is zero. Cannot divide by zero.")
            return None

        position_size = Decimal('NaN')
        contract_size = Decimal(str(MARKET_INFO.get('contractSize', '1'))) # Default to 1 if not specified
        settle_currency = MARKET_INFO.get('settle', 'N/A').upper()
        base_currency = MARKET_INFO.get('base', 'N/A').upper()
        quote_currency = MARKET_INFO.get('quote', 'N/A').upper()

        # Size calculation depends on whether it's Inverse or Linear
        if CONFIG.bybit_v5_category == 'inverse':
            # Risk Amt (in Quote) / Price Diff (Quote/Base) * Contract Size (Base/Contract) = Size (in Contracts)
            # BUT: Risk is defined on *settlement* currency (Base). Need risk in Base terms.
            # Risk Amt (Base) = Risk Amt (Quote) / Current Price (Quote/Base)
            risk_amount_base = risk_amount / current_price
            # Size (Contracts) = Risk Amt (Base) / (Price Diff (Quote/Base) / Current Price (Quote/Base) * Contract Size (Base)) <-- Check this formula derivation
            # Simpler: Size (Contracts) = Risk Amt (Base) / (SL Distance (Base/Contract))
            # SL Distance in Base = SL Distance (Quote) / Price * Contract Size (Base)
            # Check Bybit Inverse contract sizing carefully. Often Size is in USD equivalent for BTCUSD.
            # Let's assume size is in Contracts (e.g. BTC for BTCUSD)
            # Size (Contracts) = Risk Amt (Base) / (Price Diff (Quote/Contract) / Price (Quote/Base)) <-- Still seems complex
            # Alternative: Size (Contracts) = Risk Amt (Quote) / Price Diff (Quote / Contract)
            # Price Diff (Quote/Contract) = Price Diff (Quote/Base) * Contract Size (Base)
            # position_size = risk_amount / (price_difference * contract_size) # Check units
            # Let's use a simpler, common definition: Size in USD = Risk Amount (USD) / (% Price Move)
            # Size (Contracts) = (Size in USD / Price) / Contract Size? No.
            # Bybit Inverse: Size is number of contracts. Value = Size * ContractSize / Price.
            # Risk Amount (Base) = Value Change = Delta(Size * CS / Price) ~= Size * CS * Delta(1/Price)
            # Risk Amount (Base) ~= Size * CS * (1/SL - 1/Entry)
            # Size = Risk Amount (Base) / (CS * (1/SL - 1/Entry))
            risk_amount_base = risk_amount / current_price # Approx risk in Base currency
            position_size = risk_amount_base / (contract_size * abs(Decimal('1')/sl_price - Decimal('1')/current_price))
            logger.debug(f"Inverse Calc: RiskBase={risk_amount_base:.8f}, CS={contract_size}, PriceDiffInv={abs(Decimal('1')/sl_price - Decimal('1')/current_price):.8f}")


        elif CONFIG.bybit_v5_category == 'linear':
            # Risk Amt (Quote) / Price Diff (Quote/Base) = Size (in Base Currency Units, e.g., BTC for BTC/USDT)
            # Value = Size * Price. Risk = Delta(Size * Price) = Size * Delta(Price)
            # Size (Base) = Risk Amount (Quote) / Price Difference (Quote)
            position_size = risk_amount / price_difference
            # If contract size is not 1 (e.g., 1000 for SHIB1000/USDT), adjust.
            # Check if 'size' for linear refers to base currency or contracts. Usually base currency.
            # If market uses contracts (like 1000SHIB), need to divide by contract size? Check API docs.
            # Assuming position_size calculated is in BASE units (e.g. BTC).
            if contract_size != Decimal('1'):
                 logger.warning(f"Linear market {symbol} has contract size {contract_size}. Position size calculation assumes size is in BASE units. Verify contract specs.")
                 # If size needs to be in contracts: position_size_contracts = position_size / contract_size
                 # Let's assume CCXT handles this via amount formatting based on market info.

        else: # Should not happen due to config validation
            logger.error(f"Unsupported category '{CONFIG.bybit_v5_category}' for size calculation.")
            return None

        if position_size.is_nan() or position_size <= 0:
             logger.error(f"Calculated position size ({position_size}) is invalid.")
             return None

        logger.debug(f"Raw calculated Position Size: {position_size:.8f}")

        # 5. Apply Market Limits and Precision
        min_amount = Decimal(str(MARKET_INFO.get('limits', {}).get('amount', {}).get('min', '0')))
        max_amount = Decimal(str(MARKET_INFO.get('limits', {}).get('amount', {}).get('max', 'inf'))) # Use 'inf' if no max

        # Format amount according to market precision (usually rounding down)
        formatted_size_str = format_amount(symbol, position_size, rounding_mode=ROUND_DOWN)
        final_size = Decimal(formatted_size_str)
        logger.debug(f"Formatted Position Size (Prec): {final_size:.8f}")

        if final_size < min_amount:
            logger.warning(f"[yellow]Calculated size {final_size} is below minimum order size {min_amount}. Cannot place trade based on current risk/ATR.[/]")
            # Option 1: Don't trade. Option 2: Trade minimum size (increases risk %). Let's not trade.
            return None
            # Option 2 implementation:
            # logger.warning(f"Adjusting size to minimum allowed: {min_amount}")
            # final_size = min_amount

        if final_size > max_amount:
            logger.warning(f"[yellow]Calculated size {final_size} exceeds maximum order size {max_amount}. Capping size.[/]")
            final_size = max_amount
            # Reformat capped size
            formatted_size_str = format_amount(symbol, final_size, rounding_mode=ROUND_DOWN)
            final_size = Decimal(formatted_size_str)
            logger.warning(f"Adjusted size to maximum allowed: {final_size}")


        # 6. Format SL Price according to market precision
        formatted_sl_price_str = format_price(symbol, sl_price)
        final_sl_price = Decimal(formatted_sl_price_str)
        logger.debug(f"Formatted SL Price (Prec): {final_sl_price:.5f}")

        # 7. Calculate Take Profit Price (Optional)
        final_tp_price = Decimal('NaN')
        if CONFIG.tp_atr_multiplier > 0:
            tp_distance = atr * CONFIG.tp_atr_multiplier
            tp_price = Decimal(0)
            if side == 'buy':
                tp_price = current_price + tp_distance
            elif side == 'sell':
                tp_price = current_price - tp_distance

            if tp_price > 0:
                 formatted_tp_price_str = format_price(symbol, tp_price)
                 final_tp_price = Decimal(formatted_tp_price_str)
                 logger.debug(f"Calculated TP distance: {tp_distance:.5f}, Raw TP Price: {tp_price:.5f}, Formatted TP: {final_tp_price:.5f}")
            else:
                 logger.warning(f"Calculated TP price ({tp_price:.5f}) is zero or negative. Disabling TP for this trade.")
                 final_tp_price = Decimal('NaN') # Disable TP if invalid

        # 8. Sanity Check: Ensure SL/TP are logical relative to entry for the side
        if side == 'buy':
            if final_sl_price >= current_price: logger.error(f"Buy SL price {final_sl_price} >= current price {current_price}. Invalid."); return None
            if not final_tp_price.is_nan() and final_tp_price <= current_price: logger.error(f"Buy TP price {final_tp_price} <= current price {current_price}. Invalid."); return None
        elif side == 'sell':
            if final_sl_price <= current_price: logger.error(f"Sell SL price {final_sl_price} <= current price {current_price}. Invalid."); return None
            if not final_tp_price.is_nan() and final_tp_price >= current_price: logger.error(f"Sell TP price {final_tp_price} >= current price {current_price}. Invalid."); return None


        # --- Return calculated parameters ---
        params = {
            "position_size": final_size,
            "sl_price": final_sl_price,
            "tp_price": final_tp_price, # Will be NaN if TP is disabled or invalid
            "current_price": current_price, # Include for reference
            "risk_amount": risk_amount, # Include for reference
        }
        logger.info(f"Calculated Trade Params: Size={params['position_size']}, SL={params['sl_price']}" + (f", TP={params['tp_price']}" if not params['tp_price'].is_nan() else ", TP=Disabled"))
        return params

    except DivisionByZero:
        logger.error("[red]Division by zero error during parameter calculation (likely zero price difference).[/]", exc_info=True)
        return None
    except InvalidOperation as e:
         logger.error(f"[red]Decimal operation error during parameter calculation: {e}[/]", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"[red]Unexpected error calculating trade parameters: {e}[/]", exc_info=True)
        return None

# Placeholder for _execute_market_order - Assumed to exist
def _execute_market_order(symbol: str, side: str, qty_decimal: Decimal) -> Optional[Dict]:
    """Executes a market order and waits for confirmation."""
    if not EXCHANGE or not MARKET_INFO: logger.error("Cannot place order: Exchange/Market info missing."); return None
    market_id = MARKET_INFO['id']
    logger.trade(f"[bold]Executing MARKET Order:[/bold] [white]{side.upper()} {qty_decimal} {symbol}[/]")

    try:
        # Format amount for the order
        amount_str = format_amount(symbol, qty_decimal)
        amount_decimal = Decimal(amount_str) # Use the precise formatted decimal value
        logger.debug(f"Formatted order amount: {amount_decimal} (Raw: {qty_decimal})")

        # --- V5 Order Placement ---
        params = {
            'category': CONFIG.bybit_v5_category,
            'reduceOnly': False, # This is an entry order
            'timeInForce': 'ImmediateOrCancel', # Market orders are usually IOC or FOK
            # 'positionIdx': 0 # 0 for one-way mode (default), 1 for Buy hedge, 2 for Sell hedge
        }
        # Optional: Add client order ID for better tracking
        # params['orderLinkId'] = f'pyrmethus_{side}_{int(time.time() * 1000)}'

        order = fetch_with_retries(
            EXCHANGE.create_order,
            symbol=symbol,
            type='market',
            side=side,
            amount=float(amount_decimal), # CCXT often expects float for amount
            params=params
        )

        if order is None or 'id' not in order:
            logger.error(f"[red]Market order placement failed for {side.upper()} {amount_decimal} {symbol}. No order ID returned or fetch failed.[/]")
            termux_notify("Order Fail", f"Market {side.upper()} {symbol} failed (no ID)")
            return None

        order_id = order['id']
        logger.trade(f"[blue]Market order submitted:[/blue] ID: {order_id}, Side: {side.upper()}, Qty: {amount_decimal}")
        termux_notify("Order Submitted", f"Market {side.upper()} {symbol} ID: {order_id[:8]}...")

        # Wait briefly before checking status
        time.sleep(CONFIG.order_check_delay_seconds)

        # Check order status until filled or timeout
        filled_order = check_order_status(order_id, symbol, CONFIG.order_check_timeout_seconds)

        if filled_order and (filled_order.get('status') == 'closed' or filled_order.get('status') == 'filled'):
            filled_qty = Decimal(str(filled_order.get('filled', '0')))
            avg_price = Decimal(str(filled_order.get('average', '0'))) # Use 'average' if available

            # V5 info might have more accurate price in 'avgPrice'
            info_avg_price_str = filled_order.get('info', {}).get('avgPrice')
            if info_avg_price_str and info_avg_price_str != "0":
                 avg_price = Decimal(info_avg_price_str)

            logger.trade(f"[green]Market Order Confirmed Filled:[/green] ID: {order_id}, Filled Qty: {filled_qty}, Avg Price: {avg_price:.5f}")
            termux_notify("Order Filled", f"{side.upper()} {filled_qty} {symbol} @ {avg_price:.2f}")

            # Log to journal
            log_trade_entry_to_journal(symbol, side, filled_qty, avg_price, order_id)

            # Basic sanity check on filled quantity vs requested
            if filled_qty < qty_decimal * Decimal('0.95'): # Allow for some slippage/rounding diff
                logger.warning(f"[yellow]Partial fill detected:[/yellow] Requested {qty_decimal}, Filled {filled_qty}. Check position manually.")

            return filled_order # Return the confirmed order details
        else:
            status = filled_order.get('status', 'Unknown') if filled_order else 'Check Failed'
            logger.error(f"[red]Market order {order_id} did not confirm filled within timeout.[/] Last Status: {status}. MANUAL INTERVENTION REQUIRED.")
            termux_notify("Order Fail", f"Market {side} {symbol} timeout/fail. Status: {status}")
            # Consider attempting to cancel the lingering market order? Risky.
            return None # Indicate failure

    except ccxt.InsufficientFunds as e:
        logger.error(f"[bold red]Insufficient Funds[/] for market order: {e}")
        termux_notify("Order Fail", f"Insufficient funds for {side} {symbol}")
        return None
    except ccxt.ExchangeError as e: # Catch specific exchange errors like order validation issues
        logger.error(f"[red]Exchange error placing market order: {e}[/]", exc_info=True)
        termux_notify("Order Fail", f"Exchange error: {side} {symbol}")
        return None
    except Exception as e:
        logger.error(f"[red]Unexpected error executing market order: {e}[/]", exc_info=True)
        termux_notify("Order Fail", f"Unexpected error: {side} {symbol}")
        return None

# Placeholder for _set_position_stops - Assumed to exist and handle SL/TP placement
def _set_position_stops(symbol: str, position_side: str, sl_price_str: str, tp_price_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Places Stop Loss and Take Profit orders for the current position.
    Uses V5 conditional market orders (Stop Market / Take Profit Market).
    Returns the SL order ID and TP order ID (or None if not placed/failed).
    """
    if not EXCHANGE or not MARKET_INFO: logger.error("Cannot set stops: Exchange/Market info missing."); return None, None
    market_id = MARKET_INFO['id']
    logger.trade(f"[blue]Setting Stops for {position_side.upper()} position:[/blue] SL @ {sl_price_str}" + (f", TP @ {tp_price_str}" if tp_price_str else ""))

    # --- Get Current Position Size ---
    # Re-fetch position to ensure size is accurate *before* placing stops
    current_pos = get_current_position(symbol)
    if current_pos is None:
        logger.error(f"Failed to get current position size before setting stops for {position_side}. Aborting stop placement.")
        return None, None

    pos_data = current_pos.get(position_side, {})
    position_qty = pos_data.get('qty', Decimal('0'))

    if position_qty.copy_abs() < CONFIG.position_qty_epsilon:
        logger.warning(f"No significant {position_side.upper()} position found (Qty: {position_qty}). Cannot set stops.")
        return None, None

    # Determine the side for the closing SL/TP orders
    close_side = 'sell' if position_side == 'long' else 'buy'
    qty_str = format_amount(symbol, position_qty) # Format the quantity
    qty_decimal = Decimal(qty_str)
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    stop_loss_placed = False
    take_profit_placed = False

    # --- Place Stop Loss Order (Conditional Market) ---
    try:
        logger.debug(f"Placing SL Order: Symbol={symbol}, Side={close_side}, Qty={qty_decimal}, TriggerPrice={sl_price_str}")
        sl_params = {
            'category': CONFIG.bybit_v5_category,
            'stopLoss': sl_price_str,         # The trigger price
            'triggerPrice': sl_price_str,     # V5 sometimes uses triggerPrice for stop orders
            'triggerDirection': 1 if close_side == 'sell' else 2, # 1: Below trigger, 2: Above trigger
            'triggerBy': CONFIG.sl_trigger_by, # MarkPrice, LastPrice, IndexPrice
            'reduceOnly': True,               # Must be reduceOnly for SL/TP
            'closeOnTrigger': True,           # Ensure it closes the position
            'orderFilter': 'StopOrder',       # V5 filter for stop orders
            'positionIdx': 0                  # 0 for one-way mode
            # 'tpslMode': 'Partial' # Or 'Full' - if full, TP might cancel SL? Check docs. Assume partial for now.
            # 'slLimitPrice': sl_price_str # For Stop Limit, not Stop Market
        }
        # Use create_stop_order or equivalent unified method if available in CCXT version
        # Or use generic create_order with specific params
        if hasattr(EXCHANGE, 'create_stop_order'):
             sl_order = fetch_with_retries(
                 EXCHANGE.create_stop_order,
                 symbol=symbol,
                 type='market', # Stop Market
                 side=close_side,
                 amount=float(qty_decimal),
                 params=sl_params
             )
        else: # Fallback to create_order with params
             logger.debug("Using create_order for SL with params.")
             sl_order = fetch_with_retries(
                 EXCHANGE.create_order,
                 symbol=symbol,
                 type='stop', # Or 'stop_market' if supported, 'market' with stopLoss param might work
                 side=close_side,
                 amount=float(qty_decimal),
                 # price=None, # Not needed for Market SL
                 stopPrice=float(sl_price_str), # Use stopPrice if type='stop'
                 params=sl_params
             )


        if sl_order and sl_order.get('id'):
            sl_order_id = sl_order['id']
            logger.trade(f"[green]Stop Loss Order Placed:[/green] ID: {sl_order_id}, Trigger: {sl_price_str}, Side: {close_side.upper()}")
            stop_loss_placed = True
            # Update tracker immediately
            order_tracker[position_side]["sl_id"] = sl_order_id
        else:
            logger.error(f"[red]Failed to place Stop Loss order.[/] Response: {sl_order}")
            termux_notify("Stop Order Fail", f"Failed to place SL for {position_side} {symbol}")

    except ccxt.ExchangeError as e:
         # Handle specific errors like "order cost not available" or margin checks
         logger.error(f"[red]Exchange error placing Stop Loss: {e}[/]", exc_info=True)
         termux_notify("Stop Order Fail", f"Exchange error placing SL for {position_side}")
    except Exception as e:
        logger.error(f"[red]Unexpected error placing Stop Loss: {e}[/]", exc_info=True)
        termux_notify("Stop Order Fail", f"Unexpected error placing SL for {position_side}")


    # --- Place Take Profit Order (Conditional Market) ---
    # Only place TP if SL was successful and tp_price_str is valid
    if stop_loss_placed and tp_price_str and Decimal(tp_price_str) > 0:
        try:
            logger.debug(f"Placing TP Order: Symbol={symbol}, Side={close_side}, Qty={qty_decimal}, TriggerPrice={tp_price_str}")
            tp_params = {
                'category': CONFIG.bybit_v5_category,
                'takeProfit': tp_price_str,       # The trigger price
                'triggerPrice': tp_price_str,     # V5 sometimes uses triggerPrice for TP orders
                'triggerDirection': 1 if close_side == 'buy' else 2, # TP trigger direction is opposite to SL
                'triggerBy': CONFIG.sl_trigger_by, # Usually TP uses same trigger basis as SL (e.g., LastPrice)
                'reduceOnly': True,               # Must be reduceOnly
                'closeOnTrigger': True,           # Ensure it closes the position
                'orderFilter': 'StopOrder',       # V5 filter for stop/tp orders (can be same) - Check Bybit Docs, might be TakeProfitOrder
                'positionIdx': 0                  # 0 for one-way mode
                # 'tpslMode': 'Partial'
                # 'tpLimitPrice': tp_price_str # For TP Limit
            }

            # Use create_take_profit_order or equivalent unified method if available
            # Or use generic create_order with specific params
            if hasattr(EXCHANGE, 'create_take_profit_order'):
                 tp_order = fetch_with_retries(
                     EXCHANGE.create_take_profit_order,
                     symbol=symbol,
                     type='market', # Take Profit Market
                     side=close_side,
                     amount=float(qty_decimal),
                     params=tp_params
                 )
            else: # Fallback to create_order with params
                 logger.debug("Using create_order for TP with params.")
                 tp_order = fetch_with_retries(
                     EXCHANGE.create_order,
                     symbol=symbol,
                     type='take_profit', # Or 'take_profit_market', 'market' with takeProfit param
                     side=close_side,
                     amount=float(qty_decimal),
                     # price=None,
                     stopPrice=float(tp_price_str), # Use stopPrice for TP trigger if type='take_profit'
                     params=tp_params
                 )

            if tp_order and tp_order.get('id'):
                tp_order_id = tp_order['id']
                logger.trade(f"[green]Take Profit Order Placed:[/green] ID: {tp_order_id}, Trigger: {tp_price_str}, Side: {close_side.upper()}")
                take_profit_placed = True
                # Update tracker immediately
                order_tracker[position_side]["tp_id"] = tp_order_id
            else:
                logger.error(f"[red]Failed to place Take Profit order.[/] Response: {tp_order}")
                termux_notify("Stop Order Fail", f"Failed to place TP for {position_side} {symbol}")
                # If TP fails, should we cancel the SL? Potentially dangerous. Log warning.
                logger.warning("[yellow]TP order failed after SL was placed. Position only has SL protection.[/]")

        except ccxt.ExchangeError as e:
            logger.error(f"[red]Exchange error placing Take Profit: {e}[/]", exc_info=True)
            termux_notify("Stop Order Fail", f"Exchange error placing TP for {position_side}")
            logger.warning("[yellow]TP order failed after SL was placed. Position only has SL protection.[/]")
        except Exception as e:
            logger.error(f"[red]Unexpected error placing Take Profit: {e}[/]", exc_info=True)
            termux_notify("Stop Order Fail", f"Unexpected error placing TP for {position_side}")
            logger.warning("[yellow]TP order failed after SL was placed. Position only has SL protection.[/]")

    elif not stop_loss_placed:
         logger.error("Skipping TP placement because SL placement failed.")
    elif not tp_price_str or Decimal(tp_price_str) <= 0:
         logger.info("Take Profit is disabled or invalid, skipping TP order placement.")


    # Return the IDs of the orders placed (or None if they failed)
    return sl_order_id, tp_order_id


# Placeholder for _handle_entry_failure - Assumed to exist
def _handle_entry_failure(symbol: str, side: str, filled_qty_attempted: Decimal):
    """Handles the scenario where an entry order failed or timed out."""
    logger.warning(f"[yellow]Handling entry failure for {side.upper()} {symbol}...[/]")
    # Optional: Attempt to close any small residual position that might have partially filled
    # This is risky and complex, requires careful state checking.
    # For now, just log and rely on manual intervention or next cycle's logic.
    logger.critical(f"[bold red]Entry for {side.upper()} {filled_qty_attempted} {symbol} failed or timed out. MANUAL CHECK REQUIRED.[/]")
    termux_notify("Entry Failed", f"Manual check needed for failed {side} entry {symbol}")
    # Potential actions (use with caution):
    # 1. Check position again.
    # 2. If small position exists, try to close it market.
    # 3. Cancel any related conditional orders (though none should exist yet).


# Placeholder for place_risked_market_order - Assumed to exist
def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    """
    Calculates trade parameters, executes a market order, and sets SL/TP stops.
    Returns True if the entire process (entry + stops) is successful, False otherwise.
    """
    if not EXCHANGE or not MARKET_INFO: logger.error("Cannot place risked order: Exchange/Market info missing."); return False

    logger.info(f"Attempting risked market order: Side={side.upper()}, Risk={risk_percentage:.2%}, ATR={atr:.5f}")

    # --- 1. Get Balance ---
    quote_currency = MARKET_INFO.get('settle', 'USDT') # Use settlement currency for balance check
    total_equity, _ = get_balance(quote_currency)
    if total_equity is None or total_equity <= 0:
        logger.error(f"Cannot place order: Failed to get valid total equity for {quote_currency}.")
        return False

    # --- 2. Calculate Trade Parameters ---
    trade_params = _calculate_trade_parameters(symbol, side, risk_percentage, atr, total_equity)
    if not trade_params:
        logger.error("Failed to calculate trade parameters. Aborting entry.")
        return False

    position_size = trade_params['position_size']
    sl_price = trade_params['sl_price']
    tp_price = trade_params['tp_price'] # Can be NaN

    # Format prices for stop orders
    sl_price_str = format_price(symbol, sl_price)
    tp_price_str = format_price(symbol, tp_price) if not tp_price.is_nan() else None

    # --- 3. Execute Market Entry Order ---
    entry_order = _execute_market_order(symbol, side, position_size)

    if not entry_order or entry_order.get('status') not in ['closed', 'filled']:
        logger.error("Market entry order failed or did not fill. Aborting stop placement.")
        _handle_entry_failure(symbol, side, position_size) # Log critical error, notify
        return False

    # --- Entry Successful - Proceed to Set Stops ---
    position_side = "long" if side == "buy" else "short"
    logger.info(f"Entry successful for {position_side}. Proceeding to set stops.")

    # --- 4. Set SL and TP Orders ---
    # Wait a moment for the position update to propagate on the exchange
    time.sleep(CONFIG.order_check_delay_seconds * 1.5)

    sl_id, tp_id = _set_position_stops(symbol, position_side, sl_price_str, tp_price_str)

    # --- 5. Final Verification & Cleanup ---
    if sl_id: # At least SL must be placed successfully
        logger.info(f"[bold green]Risked market order process complete for {position_side.upper()} {symbol}.[/] Entry filled, SL placed (ID: {sl_id})" + (f", TP placed (ID: {tp_id})" if tp_id else ", TP not placed/failed."))
        # Update tracker (already done within _set_position_stops)
        # order_tracker[position_side]["sl_id"] = sl_id
        # order_tracker[position_side]["tp_id"] = tp_id
        return True
    else:
        # CRITICAL: Entry order filled, but SL placement failed!
        logger.critical(f"[bold red]CRITICAL: Entry order filled for {position_side} {symbol}, but FAILED to place Stop Loss![/] MANUAL INTERVENTION URGENTLY REQUIRED.")
        termux_notify("CRITICAL: SL FAILED", f"Entry filled for {position_side} {symbol}, but SL placement FAILED. CLOSE MANUALLY!")
        # Attempt to close the position immediately as a safety measure? Very risky.
        # close_position(symbol, position_side, position_size) # <-- Decide if this safety close is desired
        return False # Indicate overall process failure

# Placeholder for manage_trailing_stop - Assumed to exist
def manage_trailing_stop(symbol: str, position_side: str, position_qty: Decimal, entry_price: Decimal, current_price: Decimal, atr: Decimal) -> None:
    """Manages the Trailing Stop Loss based on configuration."""
    if not EXCHANGE or not MARKET_INFO: logger.error("Cannot manage TSL: Exchange/Market info missing."); return
    if atr.is_nan() or atr <= 0: logger.warning("Cannot manage TSL: Invalid ATR."); return
    if entry_price.is_nan() or entry_price <= 0: logger.warning("Cannot manage TSL: Invalid entry price."); return
    if position_qty.copy_abs() < CONFIG.position_qty_epsilon: logger.debug("No position, skipping TSL management."); return

    market_id = MARKET_INFO['id']
    logger.debug(f"Managing TSL for {position_side.upper()} {symbol} | Entry: {entry_price:.4f}, Current: {current_price:.4f}, ATR: {atr:.5f}")

    # --- Calculate TSL Activation Price ---
    activation_distance = atr * CONFIG.tsl_activation_atr_multiplier
    activation_price = Decimal('NaN')
    if position_side == 'long':
        activation_price = entry_price + activation_distance
    elif position_side == 'short':
        activation_price = entry_price - activation_distance

    logger.debug(f"TSL Activation Price: {activation_price:.4f} (Distance: {activation_distance:.4f})")

    # --- Check if TSL should be active ---
    tsl_should_be_active = False
    if position_side == 'long' and current_price > activation_price:
        tsl_should_be_active = True
    elif position_side == 'short' and current_price < activation_price:
        tsl_should_be_active = True

    # --- Get Current Stop Orders ---
    # Check local tracker first
    active_sl_id = order_tracker[position_side].get("sl_id")
    active_tsl_id = order_tracker[position_side].get("tsl_id")
    # active_tp_id = order_tracker[position_side].get("tp_id") # Keep track of TP too

    logger.debug(f"Tracker state: SL_ID={active_sl_id}, TSL_ID={active_tsl_id}")

    # --- Logic ---
    if tsl_should_be_active:
        logger.info(f"TSL Activation threshold reached for {position_side.upper()}.")

        # Calculate New Trailing Stop Price
        trail_amount = current_price * (CONFIG.trailing_stop_percent / Decimal('100'))
        new_tsl_price = Decimal('NaN')
        if position_side == 'long':
            new_tsl_price = current_price - trail_amount
            # Ensure TSL doesn't move backward if price retraces slightly after activation
            # Requires fetching the current TSL order's trigger price.
        elif position_side == 'short':
            new_tsl_price = current_price + trail_amount

        # Format the new TSL price
        new_tsl_price_str = format_price(symbol, new_tsl_price)
        logger.info(f"Calculated New TSL Price: {new_tsl_price_str} (Trail Amount: {trail_amount:.4f})")

        # --- Actions ---
        # 1. If TSL is already active (we have a tsl_id), modify it.
        if active_tsl_id:
             logger.debug(f"Attempting to modify existing TSL order {active_tsl_id} to trigger at {new_tsl_price_str}")
             try:
                 # V5 Modify Order: Needs order_id or orderLinkId
                 params = {
                     'category': CONFIG.bybit_v5_category,
                     'triggerPrice': new_tsl_price_str, # New trigger price
                     'orderId': active_tsl_id,
                     # Potentially need 'slTriggerBy' or other params if modifying those aspects
                 }
                 # Use modify_order or equivalent unified method
                 modified_order = fetch_with_retries(EXCHANGE.edit_order, id=active_tsl_id, symbol=symbol, type=None, side=None, amount=None, price=None, params=params)
                 # Note: edit_order might require more args depending on CCXT version/exchange implementation

                 if modified_order and modified_order.get('id') == active_tsl_id:
                     logger.trade(f"[blue]Trailing Stop Modified:[/blue] ID: {active_tsl_id}, New Trigger: {new_tsl_price_str}")
                     # No need to update tracker ID, it's the same order
                 else:
                     logger.error(f"[red]Failed to modify TSL order {active_tsl_id}.[/] Response: {modified_order}")
                     # Potential fallback: Cancel old, place new? Risky during volatility.
                     termux_notify("TSL Modify Fail", f"Failed to modify TSL {active_tsl_id}")

             except ccxt.OrderNotFound:
                 logger.warning(f"[yellow]Tracked TSL order {active_tsl_id} not found on exchange. It might have been filled or cancelled.[/] Clearing tracker.")
                 order_tracker[position_side]["tsl_id"] = None
                 # Re-evaluate state in the next cycle or attempt to place new TSL below.
             except ccxt.ExchangeError as e:
                  logger.error(f"[red]Exchange error modifying TSL {active_tsl_id}: {e}[/]", exc_info=True)
                  termux_notify("TSL Modify Fail", f"Exchange error modifying TSL {active_tsl_id}")
             except Exception as e:
                 logger.error(f"[red]Unexpected error modifying TSL {active_tsl_id}: {e}[/]", exc_info=True)
                 termux_notify("TSL Modify Fail", f"Unexpected error modifying TSL {active_tsl_id}")

        # 2. If TSL is not active (no tsl_id), but should be, cancel original SL and place TSL.
        elif active_sl_id: # Check if original SL exists
            logger.info(f"Activating TSL: Cancelling original SL ({active_sl_id}) and placing new TSL order.")

            # Cancel original SL first
            sl_cancelled = _cancel_order(active_sl_id, symbol, "Stop Loss")
            if sl_cancelled:
                order_tracker[position_side]["sl_id"] = None # Clear SL tracker

                # Place the new TSL order (similar to _set_position_stops but using TSL trigger)
                close_side = 'sell' if position_side == 'long' else 'buy'
                qty_str = format_amount(symbol, position_qty)
                qty_decimal = Decimal(qty_str)
                try:
                     # Use the same params structure as _set_position_stops, but with TSL price/config
                     tsl_params = {
                         'category': CONFIG.bybit_v5_category,
                         'stopLoss': new_tsl_price_str, # TSL trigger price
                         'triggerPrice': new_tsl_price_str,
                         'triggerDirection': 1 if close_side == 'sell' else 2,
                         'triggerBy': CONFIG.tsl_trigger_by, # Use TSL trigger setting
                         'reduceOnly': True,
                         'closeOnTrigger': True,
                         'orderFilter': 'StopOrder',
                         'positionIdx': 0
                     }
                     # Use create_stop_order or equivalent
                     tsl_order = fetch_with_retries(
                         EXCHANGE.create_order, # Use generic create_order as fallback
                         symbol=symbol,
                         type='stop', # Assuming 'stop' or 'stop_market'
                         side=close_side,
                         amount=float(qty_decimal),
                         stopPrice=float(new_tsl_price_str),
                         params=tsl_params
                     )

                     if tsl_order and tsl_order.get('id'):
                         new_tsl_id = tsl_order['id']
                         logger.trade(f"[green]Trailing Stop Loss Activated:[/green] ID: {new_tsl_id}, Trigger: {new_tsl_price_str}")
                         order_tracker[position_side]["tsl_id"] = new_tsl_id # Update tracker
                         termux_notify("TSL Activated", f"{position_side} TSL set at {new_tsl_price_str}")
                     else:
                         logger.error(f"[red]Failed to place new TSL order after cancelling SL.[/] Response: {tsl_order}. POSITION MAY HAVE NO STOP.")
                         termux_notify("CRITICAL: TSL FAIL", f"Failed to place TSL for {position_side} after SL cancel. Check position!")
                         # Try to reinstate original SL? Complex. Log critical error.

                except Exception as e:
                     logger.error(f"[red]Error placing new TSL order: {e}[/]", exc_info=True)
                     termux_notify("CRITICAL: TSL FAIL", f"Error placing TSL for {position_side}. Check position!")
            else:
                logger.error(f"[red]Failed to cancel original SL order {active_sl_id}. Cannot activate TSL.[/]")
                termux_notify("TSL Activate Fail", f"Failed to cancel SL {active_sl_id} for TSL.")

        else: # No active SL or TSL found in tracker, but TSL should be active
             logger.warning(f"[yellow]TSL should be active, but no SL/TSL found in tracker for {position_side}. Was SL/TSL placed correctly? Manual check advised.[/]")
             # Optionally, try placing a TSL directly here if confident no other stops exist
             # place_new_tsl(...) # Be cautious with this

    else: # TSL should NOT be active
         logger.debug(f"TSL activation price not reached for {position_side.upper()}. Current TSL active: {'Yes' if active_tsl_id else 'No'}")
         # Optional: If a TSL order exists but shouldn't be active (e.g., price retraced below activation),
         # potentially cancel TSL and reinstate original SL? This adds complexity.
         # For now, we only activate/modify TSL when the activation condition is met.


# Placeholder for _remove_position_stops - Assumed to exist
def _remove_position_stops(symbol: str, position_side: str) -> bool:
    """Cancels active SL, TP, and TSL orders for a given position side based on the tracker."""
    if not EXCHANGE: logger.error("Cannot remove stops: Exchange not initialized."); return False

    sl_id = order_tracker[position_side].get("sl_id")
    tp_id = order_tracker[position_side].get("tp_id")
    tsl_id = order_tracker[position_side].get("tsl_id")
    all_cancelled = True
    cancelled_ids = []

    logger.info(f"Removing conditional orders for {position_side.upper()} {symbol}...")

    if sl_id:
        if _cancel_order(sl_id, symbol, f"{position_side.upper()} Stop Loss"):
            order_tracker[position_side]["sl_id"] = None
            cancelled_ids.append(f"SL({sl_id[:6]})")
        else:
            all_cancelled = False
            logger.error(f"Failed to cancel SL {sl_id}.")

    if tp_id:
        if _cancel_order(tp_id, symbol, f"{position_side.upper()} Take Profit"):
            order_tracker[position_side]["tp_id"] = None
            cancelled_ids.append(f"TP({tp_id[:6]})")
        else:
            all_cancelled = False
            logger.error(f"Failed to cancel TP {tp_id}.")

    if tsl_id:
        if _cancel_order(tsl_id, symbol, f"{position_side.upper()} Trailing Stop"):
            order_tracker[position_side]["tsl_id"] = None
            cancelled_ids.append(f"TSL({tsl_id[:6]})")
        else:
            all_cancelled = False
            logger.error(f"Failed to cancel TSL {tsl_id}.")

    if not sl_id and not tp_id and not tsl_id:
         logger.info(f"No active conditional orders found in tracker for {position_side.upper()}.")
         return True # Nothing to cancel

    if all_cancelled:
        logger.info(f"[green]Successfully cancelled tracked orders:[/green] {', '.join(cancelled_ids)}")
        return True
    else:
        logger.warning(f"[yellow]Attempted to cancel stops, but one or more failed. Cancelled: {', '.join(cancelled_ids)}. Manual check required.[/]")
        termux_notify("Stop Cancel Fail", f"Failed cancel stops for {position_side} {symbol}. Check!")
        return False

# Helper to cancel a single order by ID
def _cancel_order(order_id: str, symbol: str, order_description: str) -> bool:
    """Attempts to cancel a single order by ID."""
    if not EXCHANGE: return False
    logger.debug(f"Attempting to cancel {order_description} order {order_id} for {symbol}...")
    try:
        params = {'category': CONFIG.bybit_v5_category}
        # V5 cancel_order might need market_id in params depending on ccxt version
        # params['symbol'] = MARKET_INFO['id']
        cancel_response = fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=params)
        logger.debug(f"Cancel response for {order_id}: {cancel_response}")

        # Check response structure - V5 success often indicated by matching orderId
        if cancel_response and (cancel_response.get('id') == order_id or cancel_response.get('info', {}).get('orderId') == order_id or cancel_response.get('info',{}).get('retCode') == 0): # V5 retCode 0 is success
             logger.info(f"[green]Successfully cancelled {order_description} order {order_id}.[/]")
             return True
        else:
             # Order might already be cancelled or filled
             logger.warning(f"[yellow]Cancel command for {order_id} sent, but confirmation unclear or failed.[/] Response: {cancel_response}. Checking status...")
             # Check status to be sure
             final_status = check_order_status(order_id, symbol, timeout=5) # Short timeout check
             if final_status and final_status.get('status') in ['canceled', 'closed', 'filled', 'rejected', 'expired']:
                  logger.info(f"Order {order_id} confirmed inactive (Status: {final_status.get('status')}). Assuming cancellation effective.")
                  return True
             else:
                  logger.error(f"[red]Failed to confirm cancellation for {order_description} order {order_id}.[/] Last status: {final_status.get('status', 'Unknown')}")
                  return False

    except ccxt.OrderNotFound:
        logger.warning(f"[yellow]Order {order_id} ({order_description}) not found on exchange. Assuming already inactive.[/]")
        return True # Treat as success if not found
    except ccxt.NetworkError as e:
         logger.error(f"[red]Network error cancelling order {order_id}: {e}[/]")
         return False
    except ccxt.ExchangeError as e:
         logger.error(f"[red]Exchange error cancelling order {order_id}: {e}[/]")
         # Could check for specific error codes indicating it was already filled/cancelled
         return False
    except Exception as e:
        logger.error(f"[red]Unexpected error cancelling order {order_id}: {e}[/]", exc_info=True)
        return False


# Placeholder for close_position - Assumed to exist
def close_position(symbol: str, position_side: str, position_qty: Decimal) -> bool:
    """Closes the specified position using a market order after cancelling stops."""
    if not EXCHANGE or not MARKET_INFO: logger.error("Cannot close position: Exchange/Market info missing."); return False
    if position_qty.copy_abs() < CONFIG.position_qty_epsilon: logger.info(f"No significant {position_side} position to close."); return True

    logger.trade(f"[bold yellow]Attempting to CLOSE {position_side.upper()} position:[/bold] Qty: {position_qty} {symbol}")
    termux_notify("Closing Position", f"Closing {position_side} {symbol} ({position_qty})")

    # 1. Cancel existing SL/TP/TSL orders for this side
    if not _remove_position_stops(symbol, position_side):
        logger.warning("[yellow]Failed to cancel all conditional orders before closing. Proceeding with close attempt, but manual check advised.[/]")
        # Continue even if cancellation fails, as closing the position is priority

    # 2. Determine close order side and quantity
    close_side = 'sell' if position_side == 'long' else 'buy'
    qty_str = format_amount(symbol, position_qty)
    qty_decimal = Decimal(qty_str)

    # 3. Execute Market Close Order
    logger.debug(f"Placing position close order: Side={close_side.upper()}, Qty={qty_decimal}")
    try:
        # V5 requires reduceOnly=True for closing orders placed manually
        params = {
            'category': CONFIG.bybit_v5_category,
            'reduceOnly': True,
            'closeOnTrigger': False, # Not a conditional order
            'timeInForce': 'ImmediateOrCancel' # Market orders
        }
        # Optional: Add client order ID
        # params['orderLinkId'] = f'pyrmethus_close_{position_side}_{int(time.time() * 1000)}'

        close_order = fetch_with_retries(
            EXCHANGE.create_order,
            symbol=symbol,
            type='market',
            side=close_side,
            amount=float(qty_decimal),
            params=params
        )

        if close_order is None or 'id' not in close_order:
            logger.error(f"[red]Position close market order failed for {close_side.upper()} {qty_decimal} {symbol}. No order ID returned or fetch failed.[/]")
            termux_notify("Close Fail", f"Market close {close_side} {symbol} failed (no ID)")
            return False # Indicate close failure

        close_order_id = close_order['id']
        logger.trade(f"[blue]Position close order submitted:[/blue] ID: {close_order_id}, Side: {close_side.upper()}, Qty: {qty_decimal}")

        # Wait and check status
        time.sleep(CONFIG.order_check_delay_seconds)
        filled_close_order = check_order_status(close_order_id, symbol, CONFIG.order_check_timeout_seconds)

        if filled_close_order and (filled_close_order.get('status') == 'closed' or filled_close_order.get('status') == 'filled'):
            filled_qty = Decimal(str(filled_close_order.get('filled', '0')))
            avg_price = Decimal(str(filled_close_order.get('average', '0')))
             # V5 info might have more accurate price in 'avgPrice'
            info_avg_price_str = filled_close_order.get('info', {}).get('avgPrice')
            if info_avg_price_str and info_avg_price_str != "0":
                 avg_price = Decimal(info_avg_price_str)

            logger.trade(f"[green]Position Close Confirmed:[/green] ID: {close_order_id}, Filled Qty: {filled_qty}, Avg Price: {avg_price:.5f}")
            termux_notify("Position Closed", f"{position_side} {symbol} closed @ {avg_price:.2f}")

            # Log close to journal (consider adding PnL if available)
            # log_trade_close_to_journal(...)

            # Clear tracker explicitly after successful close confirmation
            order_tracker[position_side] = {"sl_id": None, "tsl_id": None, "tp_id": None}
            logger.debug(f"Cleared order tracker for {position_side} after confirmed close.")

            return True # Indicate successful close
        else:
            status = filled_close_order.get('status', 'Unknown') if filled_close_order else 'Check Failed'
            logger.error(f"[red]Position close order {close_order_id} did not confirm filled within timeout.[/] Last Status: {status}. MANUAL INTERVENTION REQUIRED.")
            termux_notify("Close Fail", f"Close {close_side} {symbol} timeout/fail. Status: {status}")
            # Don't clear tracker if close failed, stops might still be active or needed
            return False # Indicate close failure

    except ccxt.InsufficientFunds as e: # Should not happen for reduceOnly, but possible glitch
        logger.error(f"[red]Insufficient Funds error during position close (unexpected for reduceOnly): {e}[/]")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"[red]Exchange error closing position: {e}[/]", exc_info=True)
        # Example: If error indicates position is already closed, treat as success?
        # if "position size is zero" in str(e).lower(): return True
        return False
    except Exception as e:
        logger.error(f"[red]Unexpected error closing position: {e}[/]", exc_info=True)
        return False

# Placeholder for generate_signals - Assumed to exist
def generate_signals(df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]]) -> Dict[str, Union[bool, str]]:
    """
    Generates trading signals based on indicator values and configured logic.
    Returns a dictionary: {'long': bool, 'short': bool, 'reason': str}
    """
    signals = {"long": False, "short": False, "reason": "No signal"}
    if not indicators or df_last_candles.empty or len(df_last_candles) < 2:
        signals["reason"] = "Skipped: Insufficient data or indicators"
        logger.debug(signals["reason"])
        return signals

    try:
        # Extract latest values (ensure they are Decimals)
        current_close = df_last_candles['close'].iloc[-1]
        prev_close = df_last_candles['close'].iloc[-2] # Need previous close for some checks
        trend_ema = indicators.get('trend_ema', Decimal('NaN'))
        fast_ema = indicators.get('fast_ema', Decimal('NaN'))
        slow_ema = indicators.get('slow_ema', Decimal('NaN'))
        stoch_k = indicators.get('stoch_k', Decimal('NaN'))
        stoch_d = indicators.get('stoch_d', Decimal('NaN'))
        atr = indicators.get('atr', Decimal('NaN'))

        # Check for NaN values that prevent signal generation
        required_inds = [current_close, prev_close, trend_ema, fast_ema, slow_ema, stoch_k, stoch_d, atr]
        if any(ind.is_nan() for ind in required_inds):
             signals["reason"] = "Skipped: NaN value in required indicators/prices"
             logger.warning(f"{signals['reason']} - Check calculation or data source.")
             return signals

        # --- Define Signal Conditions ---
        reasons = [] # Collect reasons for the signal

        # 1. Trend Filter (Optional)
        is_uptrend = False
        is_downtrend = False
        trend_buffer = trend_ema * (CONFIG.trend_filter_buffer_percent / Decimal('100'))
        if current_close > trend_ema + trend_buffer: is_uptrend = True
        if current_close < trend_ema - trend_buffer: is_downtrend = True
        trend_condition_met = (not CONFIG.trade_only_with_trend) or \
                              (CONFIG.trade_only_with_trend and (is_uptrend or is_downtrend))
        if CONFIG.trade_only_with_trend: reasons.append(f"Trend={'Up' if is_uptrend else 'Down' if is_downtrend else 'Neutral'}")

        # 2. EMA Crossover
        ema_bull_cross = fast_ema > slow_ema
        ema_bear_cross = fast_ema < slow_ema
        if ema_bull_cross: reasons.append("EMA_Bull")
        if ema_bear_cross: reasons.append("EMA_Bear")

        # 3. Stochastic Condition
        stoch_bull_cond = stoch_k > stoch_d and stoch_k < CONFIG.stoch_oversold_threshold
        stoch_bear_cond = stoch_k < stoch_d and stoch_k > CONFIG.stoch_overbought_threshold
        # Alternative: Crossover based
        # stoch_k_prev = indicators.get('stoch_k_prev', Decimal('NaN')) # Need previous K if using crossover
        # stoch_bull_cross = stoch_k > stoch_d and stoch_k_prev < indicators.get('stoch_d_prev')
        if stoch_bull_cond: reasons.append("Stoch_Bull")
        if stoch_bear_cond: reasons.append("Stoch_Bear")

        # 4. ATR Move Filter (Price moved significantly in direction)
        min_move = atr * CONFIG.atr_move_filter_multiplier
        bull_move = (current_close - prev_close) > min_move
        bear_move = (prev_close - current_close) > min_move
        if bull_move: reasons.append("ATR_BullMove")
        if bear_move: reasons.append("ATR_BearMove")

        # --- Combine Conditions for Long Signal ---
        long_signal = ema_bull_cross and stoch_bull_cond and bull_move
        if CONFIG.trade_only_with_trend:
            long_signal = long_signal and is_uptrend

        # --- Combine Conditions for Short Signal ---
        short_signal = ema_bear_cross and stoch_bear_cond and bear_move
        if CONFIG.trade_only_with_trend:
            short_signal = short_signal and is_downtrend

        # --- Final Signal Assignment ---
        if long_signal:
            signals['long'] = True
            signals['reason'] = f"Long Signal: {' & '.join(reasons)}"
            logger.debug(f"Generated LONG signal. Reason: {signals['reason']}")
        elif short_signal:
            signals['short'] = True
            signals['reason'] = f"Short Signal: {' & '.join(reasons)}"
            logger.debug(f"Generated SHORT signal. Reason: {signals['reason']}")
        else:
            signals['reason'] = f"No signal: Conditions not met. State: {' '.join(reasons)}"
            logger.debug(f"No signal generated. Conditions state: {' '.join(reasons)}")

        return signals

    except Exception as e:
        logger.error(f"[red]Error generating signals: {e}[/]", exc_info=True)
        signals["reason"] = "Error during signal generation"
        return signals


# Placeholder for print_status_panel - Updated to use Rich
# pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches
def print_status_panel(
    cycle: int,
    timestamp: Optional[pd.Timestamp],
    price: Optional[Decimal],
    indicators: Optional[Dict[str, Union[Decimal, bool, int]]],
    positions: Optional[Dict[str, Dict[str, Any]]],
    equity: Optional[Decimal],
    signals: Dict[str, Union[bool, str]],
    order_tracker_state: Dict[str, Dict[str, Optional[str]]]
) -> None:
    """Displays the current bot status using a Rich Panel and Table."""

    # --- Prepare Data ---
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else "[dim]N/A[/]"
    price_str = f"{price:.4f}" if price and not price.is_nan() else "[dim]N/A[/]"
    equity_str = f"{equity:.2f} {MARKET_INFO.get('settle', 'USD')}" if equity and not equity.is_nan() else "[dim]N/A[/]"

    # Indicators Table
    ind_table = Table.grid(padding=(0, 1), expand=False)
    ind_table.add_column(style="cyan") # Label
    ind_table.add_column(style="white") # Value
    if indicators:
        # Format Decimals nicely, handle NaN
        f = lambda v: f"{v:.4f}" if isinstance(v, Decimal) and not v.is_nan() else str(v) if not isinstance(v, Decimal) else "[dim]NaN[/]"
        ind_table.add_row("Trend EMA:", f(indicators.get('trend_ema')))
        ind_table.add_row("Fast EMA:", f(indicators.get('fast_ema')))
        ind_table.add_row("Slow EMA:", f(indicators.get('slow_ema')))
        ind_table.add_row("Stoch %K:", f(indicators.get('stoch_k')))
        ind_table.add_row("Stoch %D:", f(indicators.get('stoch_d')))
        ind_table.add_row("ATR:", f(indicators.get('atr')))
    else:
        ind_table.add_row("[dim]Indicators:", "N/A[/]")

    # Position Table
    pos_table = Table(show_header=True, header_style="bold magenta", border_style="dim", expand=False)
    pos_table.add_column("Side", style="dim", width=6)
    pos_table.add_column("Qty", justify="right")
    pos_table.add_column("Entry", justify="right")
    pos_table.add_column("Liq.", justify="right")
    pos_table.add_column("uPNL", justify="right")
    pos_table.add_column("SL ID", justify="center")
    pos_table.add_column("TP ID", justify="center")
    pos_table.add_column("TSL ID", justify="center")

    has_pos = False
    if positions:
        for side, data in positions.items():
            qty = data.get('qty', Decimal('0'))
            if qty.copy_abs() >= CONFIG.position_qty_epsilon:
                has_pos = True
                entry = data.get('entry_price', Decimal('NaN'))
                liq = data.get('liq_price', Decimal('NaN'))
                pnl = data.get('pnl', Decimal('NaN'))
                sl_id = order_tracker_state.get(side, {}).get('sl_id')
                tp_id = order_tracker_state.get(side, {}).get('tp_id')
                tsl_id = order_tracker_state.get(side, {}).get('tsl_id')

                side_style = "bold green" if side == 'long' else "bold red"
                pnl_style = "green" if not pnl.is_nan() and pnl >= 0 else "red" if not pnl.is_nan() else "dim"

                pos_table.add_row(
                    f"[{side_style}]{side.upper()}[/]",
                    f"{qty.normalize()}", # Use normalize for cleaner Decimal output
                    f"{entry:.4f}" if not entry.is_nan() else "[dim]N/A[/]",
                    f"{liq:.4f}" if not liq.is_nan() else "[dim]N/A[/]",
                    f"[{pnl_style}]{pnl:.3f}[/]" if not pnl.is_nan() else "[dim]N/A[/]",
                    f"{sl_id[:6]}.." if sl_id else "[dim]-[/]",
                    f"{tp_id[:6]}.." if tp_id else "[dim]-[/]",
                    f"{tsl_id[:6]}.." if tsl_id else "[dim]-[/]"
                )
    if not has_pos:
         pos_table.add_row("[bold yellow]FLAT[/]", "[dim]-[/]", "[dim]-[/]", "[dim]-[/]", "[dim]-[/]", "[dim]-[/]", "[dim]-[/]", "[dim]-[/]")

    # Signal Status
    sig_text = Text()
    if signals.get('long'): sig_text.append("Long Signal!", style="bold green")
    elif signals.get('short'): sig_text.append("Short Signal!", style="bold red")
    else: sig_text.append("No Signal", style="yellow")
    sig_text.append(f"\n[dim]{signals.get('reason', 'N/A')}[/]", style="dim")


    # --- Assemble Panel ---
    grid = Table.grid(expand=True)
    grid.add_column(justify="left") # Left Column (Indicators, Signal)
    grid.add_column(justify="right") # Right Column (Time, Price, Equity)

    grid.add_row(ind_table, f"[bold]Cycle:[/bold] [bright_white]{cycle}[/]\n[bold]Time:[/bold] [bright_white]{ts_str}[/]")
    grid.add_row(sig_text, f"[bold]Price:[/bold] [bright_white]{price_str}[/]\n[bold]Equity:[/bold] [bright_white]{equity_str}[/]")

    # Combine grid and position table
    main_content = Table.grid(expand=True)
    main_content.add_row(grid)
    main_content.add_row("\n[bold magenta]--- Position ---[/]") # Separator
    main_content.add_row(pos_table)


    panel = Panel(
        main_content,
        title=f"[bold bright_cyan]Pyrmethus Status[/] - {CONFIG.symbol} ({CONFIG.interval})",
        border_style="blue",
        padding=(1, 2) # Add some padding
    )
    console.print(panel)


# --- Main Trading Cycle & Loop ---

# pylint: disable=too-many-locals, too-many-statements, too-many-branches, global-statement
def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell: data fetch, analysis, execution."""
    global order_tracker # Allow modification by sub-functions (e.g., setting stops, TSL)

    logger.info(f"[bold]--- Starting Cycle {cycle_count} ---[/]")
    start_time = time.time()
    cycle_status = "[green]OK[/]" # Tracks if cycle completed without critical fetch/state errors

    # --- 1. Fetch Market Data ---
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error("[red]Halting cycle: Market data fetch failed.[/]")
        cycle_status = "[bold red]FAIL (Data Fetch)[/]"
        end_time = time.time(); logger.info(f"[bold]--- Cycle {cycle_count} {cycle_status} (Duration: {end_time - start_time:.2f}s) ---[/]"); return

    # --- 2. Get Current Price & Timestamp ---
    current_price: Optional[Decimal] = None
    last_timestamp: Optional[pd.Timestamp] = None
    try:
        if not df.empty:
            last_candle = df.iloc[-1]
            current_price = last_candle["close"] # Already Decimal
            last_timestamp = df.index[-1]
            if current_price.is_nan() or current_price <= 0:
                raise ValueError("Invalid latest close price (NaN or <= 0)")
            logger.debug(f"Latest candle: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}")

            # Stale data check
            if EXCHANGE:
                try:
                    now_utc = pd.Timestamp.utcnow().tz_localize('UTC')
                    time_diff = now_utc - last_timestamp
                    interval_seconds = EXCHANGE.parse_timeframe(CONFIG.interval)
                    # Allow up to 1.5 intervals + 60s buffer for lag/API delays
                    max_lag = pd.Timedelta(seconds=interval_seconds * 1.5 + 60)
                    if time_diff > max_lag:
                        logger.warning(f"[yellow]Data may be stale:[/yellow] Last candle is {time_diff.total_seconds():.0f}s old (Max lag: {max_lag.total_seconds():.0f}s).")
                except Exception as e:
                    logger.warning(f"Staleness check failed: {e}") # Non-critical
        else:
             raise ValueError("DataFrame is empty after fetch.")

    except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
        logger.error(f"[red]Halting cycle: Price/Timestamp processing error: {e}[/]", exc_info=True)
        cycle_status = "[bold red]FAIL (Price Proc)[/]"
        end_time = time.time(); logger.info(f"[bold]--- Cycle {cycle_count} {cycle_status} (Duration: {end_time - start_time:.2f}s) ---[/]"); return

    # --- 3. Calculate Indicators ---
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error("[red]Indicator calculation failed. Skipping trade logic for this cycle.[/]")
        cycle_status = "[yellow]WARN (Indicators)[/]"
        # Don't return yet, still print status panel

    current_atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')

    # --- 4. Get Current State (Balance & Positions) ---
    quote_currency = MARKET_INFO.get('settle', 'USDT') if MARKET_INFO else 'USDT'
    total_equity, _ = get_balance(quote_currency) # Use total equity for risk calc
    if total_equity is None or total_equity.is_nan() or total_equity <= 0:
        logger.error(f"[red]Failed fetching valid equity for {quote_currency}. Skipping trade logic.[/]")
        # Don't return yet, print panel with available info
        cycle_status = "[yellow]WARN (Equity Fetch)[/]" if cycle_status == "[green]OK[/]" else cycle_status

    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error("[red]Failed fetching positions. Skipping trade logic.[/]")
        # Don't return yet, print panel with available info
        cycle_status = "[yellow]WARN (Position Fetch)[/]" if cycle_status == "[green]OK[/]" else cycle_status

    # --- Capture State Snapshot for Panel ---
    # Use deepcopy for the tracker as it might be modified during the cycle
    order_tracker_snapshot = copy.deepcopy(order_tracker)
    # Use the fetched positions (or default empty dict) directly
    positions_snapshot = positions if positions is not None else {"long": {}, "short": {}}

    # --- Initialize signals ---
    signals: Dict[str, Union[bool, str]] = {"long": False, "short": False, "reason": "Skipped: Initial State"}

    # --- Main Logic (Only proceed if critical data is available) ---
    can_run_trade_logic = (
        cycle_status == "[green]OK[/]" and # No critical fetch errors so far
        indicators is not None and
        positions is not None and
        total_equity is not None and total_equity > 0 and not total_equity.is_nan() and
        not current_price.is_nan() and
        not current_atr.is_nan() # ATR needed for entries/TSL
    )

    if can_run_trade_logic:
        # Get live state from fetched data
        active_long_pos = positions.get('long', {})
        active_short_pos = positions.get('short', {})
        active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
        active_short_qty = active_short_pos.get('qty', Decimal('0.0'))
        active_long_entry = active_long_pos.get('entry_price', Decimal('NaN'))
        active_short_entry = active_short_pos.get('entry_price', Decimal('NaN'))

        # Use epsilon for position checks
        has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
        has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
        is_flat = not has_long_pos and not has_short_pos
        logger.debug(f"Initial State Check: Flat={is_flat}, Long Qty={active_long_qty.normalize()}, Short Qty={active_short_qty.normalize()}")

        # --- 5. Manage Trailing Stops (If position exists) ---
        if has_long_pos or has_short_pos:
             pos_side = "long" if has_long_pos else "short"
             pos_entry = active_long_entry if has_long_pos else active_short_entry
             pos_qty = active_long_qty if has_long_pos else active_short_qty
             manage_trailing_stop(CONFIG.symbol, pos_side, pos_qty, pos_entry, current_price, current_atr)
        elif is_flat and any(order_tracker[s][k] for s in ["long", "short"] for k in ["sl_id", "tsl_id", "tp_id"]):
            # If flat, but tracker still holds old IDs, clear them.
             logger.info("Position flat, clearing any stale order trackers.")
             order_tracker["long"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
             order_tracker["short"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
             order_tracker_snapshot = copy.deepcopy(order_tracker) # Update snapshot for panel

        # --- Re-fetch position state AFTER TSL management (as TSL might have closed the position) ---
        logger.debug("Re-fetching position state after TSL check...")
        positions_after_tsl = get_current_position(CONFIG.symbol)
        if positions_after_tsl is None:
            logger.error("[red]Failed re-fetching positions after TSL check. State uncertain. Skipping further trade logic.[/]")
            cycle_status = "[yellow]WARN (Pos Re-fetch Fail)[/]"
            can_run_trade_logic = False # Prevent further actions based on potentially stale state
            positions_snapshot = {"long": {"qty": Decimal('NaN')}, "short": {"qty": Decimal('NaN')}} # Indicate unknown state in panel
        else:
            # Update live state variables
            active_long_pos = positions_after_tsl.get('long', {})
            active_short_pos = positions_after_tsl.get('short', {})
            active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
            active_short_qty = active_short_pos.get('qty', Decimal('0.0'))
            has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
            has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
            is_flat = not has_long_pos and not has_short_pos
            logger.debug(f"State After TSL Check: Flat={is_flat}, Long={active_long_qty.normalize()}, Short={active_short_qty.normalize()}")
            # Update snapshots for the panel
            positions_snapshot = positions_after_tsl
            order_tracker_snapshot = copy.deepcopy(order_tracker) # Capture tracker state *after* TSL logic ran

            # If position became flat after TSL, ensure trackers are clear
            if is_flat and any(order_tracker[s][k] for s in ["long", "short"] for k in ["sl_id", "tsl_id", "tp_id"]):
                logger.info("Position became flat after TSL check, clearing trackers.")
                order_tracker["long"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
                order_tracker["short"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
                order_tracker_snapshot = copy.deepcopy(order_tracker) # Update snapshot again

        # --- Proceed only if state re-fetch was successful ---
        if can_run_trade_logic:
             # --- 6. Generate Trading Signals ---
             # Need at least 2 candles for some indicator calculations/comparisons
             if len(df) >= 2:
                  signals = generate_signals(df.iloc[-2:], indicators) # Pass last 2 candles
             else:
                  reason = f"Skipped Signal Gen: Need >=2 candles ({len(df)} found)."
                  signals = {"long": False, "short": False, "reason": reason}
                  logger.warning(reason)


             # --- 7. Check for Signal-Based Exits (Only if position exists) ---
             exit_triggered_by_signal = False
             if not is_flat:
                 fast_ema = indicators.get('fast_ema', Decimal('NaN'))
                 slow_ema = indicators.get('slow_ema', Decimal('NaN'))
                 if not fast_ema.is_nan() and not slow_ema.is_nan():
                     # Exit Long if Fast EMA crosses Below Slow EMA
                     if has_long_pos and fast_ema < slow_ema:
                         logger.trade("[bold yellow]EMA Bearish Cross vs LONG position. Closing position.[/]")
                         if close_position(CONFIG.symbol, "long", active_long_qty):
                              exit_triggered_by_signal = True
                         else: logger.error("[red]Failed to execute EMA-based close for LONG position.[/]")
                     # Exit Short if Fast EMA crosses Above Slow EMA
                     elif has_short_pos and fast_ema > slow_ema:
                         logger.trade("[bold yellow]EMA Bullish Cross vs SHORT position. Closing position.[/]")
                         if close_position(CONFIG.symbol, "short", active_short_qty):
                              exit_triggered_by_signal = True
                         else: logger.error("[red]Failed to execute EMA-based close for SHORT position.[/]")

                 if not exit_triggered_by_signal and (has_long_pos or has_short_pos):
                      logger.debug("No counter-EMA signal for exit.")

             # --- Re-fetch state AGAIN if an exit was triggered by signal ---
             if exit_triggered_by_signal:
                  logger.debug("Re-fetching state after signal exit attempt...")
                  positions_after_exit = get_current_position(CONFIG.symbol)
                  if positions_after_exit is None:
                       logger.error("[red]Failed re-fetching positions after signal exit. State uncertain.[/]")
                       cycle_status = "[yellow]WARN (Pos Re-fetch Fail)[/]"
                       can_run_trade_logic = False # Prevent entry based on uncertain state
                       positions_snapshot = {"long": {"qty": Decimal('NaN')}, "short": {"qty": Decimal('NaN')}}
                  else:
                       # Update live state variables
                       active_long_pos = positions_after_exit.get('long', {})
                       active_short_pos = positions_after_exit.get('short', {})
                       active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
                       active_short_qty = active_short_pos.get('qty', Decimal('0.0'))
                       has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
                       has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
                       is_flat = not has_long_pos and not has_short_pos
                       logger.debug(f"State After Signal Exit: Flat={is_flat}, Long={active_long_qty.normalize()}, Short={active_short_qty.normalize()}")
                       # Update snapshots for the panel
                       positions_snapshot = positions_after_exit
                       order_tracker_snapshot = copy.deepcopy(order_tracker) # Capture tracker state *after* exit logic ran

             # --- 8. Execute Entry Trades (Only if now flat and entry signal exists) ---
             if can_run_trade_logic and is_flat and (signals.get("long") or signals.get("short")):
                 entry_attempted = False
                 entry_successful = False
                 if signals.get("long"):
                     logger.trade(f"[bold green]Long signal confirmed![/] Reason: {signals.get('reason', '')}. Attempting entry...")
                     entry_attempted = True
                     entry_successful = place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr)
                 elif signals.get("short"):
                     logger.trade(f"[bold red]Short signal confirmed![/] Reason: {signals.get('reason', '')}. Attempting entry...")
                     entry_attempted = True
                     entry_successful = place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr)

                 if entry_attempted:
                     if entry_successful:
                         logger.info("[green]Entry process completed successfully.[/]")
                     else:
                         logger.error("[red]Entry process failed.[/]")
                         cycle_status = "[yellow]WARN (Entry Fail)[/]" # Mark cycle with warning on entry fail

                     # Re-fetch state one last time for panel accuracy after entry attempt
                     logger.debug("Re-fetching state after entry attempt for panel...")
                     positions_after_entry = get_current_position(CONFIG.symbol)
                     if positions_after_entry is not None:
                         positions_snapshot = positions_after_entry
                         order_tracker_snapshot = copy.deepcopy(order_tracker) # Get latest tracker state
                     else:
                         logger.warning("Failed re-fetching positions after entry attempt. Panel may be slightly stale.")

             elif is_flat:
                 logger.debug("Position is flat, no entry signal generated.")
             elif not is_flat:
                 logger.debug(f"Position ({'LONG' if has_long_pos else 'SHORT'}) remains open, skipping new entry.")

    else: # Initial critical data fetch or re-fetch failed
         fail_reason = cycle_status.replace('[','').replace(']','').split('(')[-1][:-1] if '(' in cycle_status else "Unknown"
         signals["reason"] = f"Skipped Trade Logic: Critical data missing or fetch failed ({fail_reason})"
         logger.warning(signals["reason"])

    # --- 9. Display Status Panel ---
    print_status_panel(
        cycle=cycle_count,
        timestamp=last_timestamp,
        price=current_price,
        indicators=indicators,
        positions=positions_snapshot, # Use the snapshot taken after last state update
        equity=total_equity,
        signals=signals,
        order_tracker_state=order_tracker_snapshot # Use the snapshot taken after last state update
    )

    end_time = time.time()
    final_status_log = cycle_status if cycle_status != "[green]OK[/]" else "[green]OK[/]"
    logger.info(f"[bold]--- Cycle {cycle_count} {final_status_log} (Duration: {end_time - start_time:.2f}s) ---[/]")


# pylint: disable=too-many-statements, too-many-branches
def graceful_shutdown() -> None:
    """Attempts to cancel all orders and close open positions before exiting."""
    # pylint: disable=global-statement
    global order_tracker
    logger.warning(f"\n[bold yellow]Initiating Graceful Shutdown Sequence...[/]")
    termux_notify("Shutdown Started", f"Closing {CONFIG.symbol}...")

    if EXCHANGE is None or MARKET_INFO is None:
        logger.error("[red]Cannot perform clean shutdown: Exchange or Market Info not initialized.[/]")
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Cannot shutdown cleanly.")
        return

    symbol = CONFIG.symbol
    market_id = MARKET_INFO.get('id')

    # --- 1. Cancel All Cancellable Orders ---
    # Includes open limit orders and conditional (SL/TP/TSL) orders
    try:
        logger.info(f"[cyan]Attempting to cancel all cancellable orders for {symbol} ({market_id})...[/]")
        cancelled_successfully = False

        # Bybit V5 specific cancel all conditional orders first (more specific)
        try:
            logger.debug("Using V5 cancel_all_conditional_orders...")
            params = {'category': CONFIG.bybit_v5_category, 'symbol': market_id}
            # Need method specific to conditional orders if available, e.g., private_post_order_cancel_all
            # Check CCXT implementation details for Bybit V5 conditional cancel all
            # Using generic cancel_all_orders might work if it covers conditional orders
            # Let's assume cancel_all_orders covers them for now, or use specific endpoint if known.
            # Example using a hypothetical specific endpoint:
            # response = fetch_with_retries(EXCHANGE.private_post_v5_order_cancel_all, params=params) # Adjust endpoint name
            # if response and response.get('retCode') == 0:
            #     logger.info("[green]V5 Cancel All Conditional Orders successful via specific endpoint.[/]")
            #     # cancelled_successfully = True # Don't set yet, need regular orders too
            # else:
            #     logger.warning(f"V5 Cancel All Conditional Orders response unclear/failed: {response}")

        except Exception as e:
            logger.warning(f"[yellow]Could not execute specific V5 conditional cancel all (if available): {e}. Falling back to generic cancel.[/]")


        # Use generic cancel_all_orders (should cover limit and potentially conditional in some implementations)
        try:
            logger.debug("Using generic cancel_all_orders...")
            # V5 might require symbol for cancel_all_orders as well
            params={'symbol': symbol, 'category': CONFIG.bybit_v5_category}
            cancel_resp = fetch_with_retries(EXCHANGE.cancel_all_orders, symbol=symbol, params=params)
            logger.info(f"cancel_all_orders response: {cancel_resp}") # Response varies by exchange
            # Assume success if no error, though response parsing is better
            cancelled_successfully = True # Assume it worked if no exception
        except ccxt.NotSupported:
            logger.warning("[yellow]Exchange does not support cancel_all_orders. Manual check required.[/]")
        except Exception as e:
            logger.error(f"[red]cancel_all_orders error: {e}[/]", exc_info=True)

        if cancelled_successfully:
             logger.info("[green]Order cancellation command(s) sent.[/]")
             # Clear local tracker regardless of exact success, as we intend to close positions anyway
             logger.info("Clearing local order tracker.")
             order_tracker["long"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
             order_tracker["short"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
        else:
             logger.error("[bold red]Order cancellation failed or was not fully supported. MANUAL ORDER CHECK REQUIRED.[/]")


    except Exception as e:
        logger.error(f"[bold red]Critical error during order cancellation phase: {e}. MANUAL CHECK REQUIRED.[/]", exc_info=True)

    # Wait a bit for cancellations to process
    logger.info("Waiting briefly after order cancellation attempt...")
    time.sleep(max(CONFIG.order_check_delay_seconds, 3))

    # --- 2. Close Any Remaining Positions ---
    try:
        logger.info(f"[cyan]Checking for lingering positions to close for {symbol}...[/]")
        positions = get_current_position(symbol)
        closed_count = 0
        positions_to_close = {}

        if positions:
            positions_to_close = {
                side: data for side, data in positions.items()
                if data.get('qty') and data['qty'].copy_abs() >= CONFIG.position_qty_epsilon
            }
            if not positions_to_close:
                logger.info(f"[green]No significant open positions found for {symbol}.[/]")
            else:
                logger.warning(f"[yellow]Found {len(positions_to_close)} position(s) requiring closure:[/]")
                for side, pos_data in positions_to_close.items():
                    qty = pos_data.get('qty', Decimal("0.0"))
                    entry = pos_data.get('entry_price', Decimal('NaN'))
                    logger.warning(f"  - {side.upper()}: Qty={qty.normalize()}, Entry={entry.normalize()}")

                # Attempt to close each position
                for side, pos_data in positions_to_close.items():
                     qty = pos_data.get('qty', Decimal("0.0"))
                     if close_position(symbol, side, qty): # Use dedicated close function
                         closed_count += 1
                     else:
                         logger.error(f"[bold red]Market close order submission FAILED for {side.upper()} position.[/] MANUAL INTERVENTION REQUIRED.")

                if closed_count == len(positions_to_close):
                    logger.info(f"[green]All ({closed_count}) detected positions closure orders submitted successfully.[/]")
                    # Final check
                    time.sleep(CONFIG.order_check_delay_seconds * 2)
                    final_pos = get_current_position(symbol)
                    if final_pos and not any(d.get('qty') and d['qty'].copy_abs() >= CONFIG.position_qty_epsilon for d in final_pos.values()):
                         logger.info("[bold green]Final position check confirms FLAT.[/]")
                    else:
                         logger.warning("[yellow]Final position check shows residual position or failed fetch. Manual verification recommended.[/]")
                else:
                    logger.warning(f"[bold yellow]Attempted {len(positions_to_close)} closures, {closed_count} orders submitted.[/] Some closure orders failed. MANUAL VERIFICATION REQUIRED.")
        else:
            logger.error(f"[bold red]Failed to fetch positions during shutdown sequence.[/] Cannot confirm if flat. MANUAL CHECK REQUIRED.")

    except Exception as e:
        logger.error(f"[bold red]Critical error during position closure phase: {e}. MANUAL CHECK REQUIRED.[/]", exc_info=True)

    logger.warning(f"[bold yellow]Graceful Shutdown Sequence Complete. Pyrmethus rests.[/]")
    termux_notify("Shutdown Complete", f"{CONFIG.symbol} shutdown finished. Verify position state.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Initialize Exchange ---
    EXCHANGE = initialize_exchange()
    if EXCHANGE is None:
        console.print("[bold red]Failed to initialize exchange. Cannot continue.[/]")
        sys.exit(1)

    # --- Display Initial Configuration Summary using Rich ---
    console.print(Panel(
        Text.assemble(
            ("Trading Symbol: ", "yellow"), (f"{CONFIG.symbol}", "white"), "\n",
            ("Interval: ", "yellow"), (f"{CONFIG.interval}", "white"), "\n",
            ("Category: ", "yellow"), (f"{CONFIG.bybit_v5_category}", "white"), "\n",
            ("Risk %: ", "yellow"), (f"{CONFIG.risk_percentage:.3%}", "white"), "\n",
            ("SL Mult: ", "yellow"), (f"{CONFIG.sl_atr_multiplier.normalize()}x", "white"), " | ",
            ("TP Mult: ", "yellow"), (f"{CONFIG.tp_atr_multiplier.normalize()}x" if CONFIG.tp_atr_multiplier > 0 else "Disabled", "white"), "\n",
            ("TSL Act Mult: ", "yellow"), (f"{CONFIG.tsl_activation_atr_multiplier.normalize()}x", "white"), " | ",
            ("TSL Trail %: ", "yellow"), (f"{CONFIG.trailing_stop_percent.normalize()}%", "white"), "\n",
            ("Trend Filter: ", "yellow"), (f"{'ON' if CONFIG.trade_only_with_trend else 'OFF'}", "white"), " | ",
            ("ATR Move Filter: ", "yellow"), (f"{CONFIG.atr_move_filter_multiplier.normalize()}x", "white"), "\n",
            ("Journaling: ", "yellow"), (f"{'Enabled' if CONFIG.enable_journaling else 'Disabled'}", "white"),
            (f" ([dim]{CONFIG.journal_file_path}[/])" if CONFIG.enable_journaling else "", "dim")
        ),
        title="[bold bright_cyan]Pyrmethus v2.3.2 Config[/]",
        border_style="magenta",
        padding=(1, 2)
    ))

    console.print("[bold green]Pyrmethus Spell is Active... Monitoring the markets.[/]")
    termux_notify("Pyrmethus Started", f"{CONFIG.symbol} @ {CONFIG.interval} | Risk {CONFIG.risk_percentage:.2%}")

    # --- Main Loop ---
    cycle_count = 0
    while not shutdown_requested:
        cycle_count += 1
        cycle_start_time = time.monotonic() # Use monotonic clock for interval timing
        try:
            trading_spell_cycle(cycle_count)

        except KeyboardInterrupt:
            logger.warning("\n[yellow]Ctrl+C detected in main loop. Initiating shutdown...[/]")
            shutdown_requested = True # Trigger graceful shutdown via signal handler route is preferred

        except ccxt.AuthenticationError as e:
            logger.critical(f"[bold red]CRITICAL AUTHENTICATION ERROR in main loop: {e}. Halting immediately.[/]", exc_info=True)
            termux_notify("CRITICAL ERROR", f"Auth Error - Pyrmethus Halting!")
            shutdown_requested = True # Attempt graceful shutdown, though it might fail
            # sys.exit(1) # Consider immediate exit if shutdown fails

        except ccxt.ExchangeNotAvailable as e:
             logger.error(f"[bold red]Exchange Not Available: {e}. Retrying after longer delay.[/]", exc_info=True)
             termux_notify("Exchange Down", "Exchange unavailable. Retrying.")
             # Longer sleep before next cycle attempt
             sleep_time = CONFIG.loop_sleep_seconds * 5

        except Exception as e: # Catch-all for unexpected errors *within* trading_spell_cycle
            logger.error(f"[bold red]Unhandled exception in main trading cycle (Cycle {cycle_count}): {e}[/]", exc_info=True)
            logger.error("[red]Attempting to continue loop, but caution is advised. Check logs thoroughly.[/]")
            termux_notify("Pyrmethus Error", f"Unhandled exception cycle {cycle_count}. Check logs.")
            # Consider a longer sleep after an unexpected error
            sleep_time = CONFIG.loop_sleep_seconds * 2 # Longer sleep after error

        else: # If cycle completed without exceptions
             sleep_time = CONFIG.loop_sleep_seconds # Normal sleep time

        # --- Interruptible Sleep ---
        if not shutdown_requested:
            # Calculate time spent and adjust sleep duration to maintain interval roughly
            cycle_duration = time.monotonic() - cycle_start_time
            actual_sleep_time = max(0.1, sleep_time - cycle_duration) # Ensure minimum sleep
            logger.debug(f"Cycle {cycle_count} duration: {cycle_duration:.2f}s. Sleeping for {actual_sleep_time:.2f}s...")
            sleep_end_time = time.monotonic() + actual_sleep_time
            try:
                while time.monotonic() < sleep_end_time and not shutdown_requested:
                    time.sleep(0.5) # Check shutdown flag periodically
            except KeyboardInterrupt:
                logger.warning("\n[yellow]Ctrl+C detected during sleep. Initiating shutdown...[/]")
                shutdown_requested = True

    # --- Shutdown Sequence ---
    graceful_shutdown()
    console.print("[bold bright_cyan]Pyrmethus has returned to the ether.[/]")
    sys.exit(0) # Clean exit after shutdown attempt
