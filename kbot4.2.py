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
    # Pyrmethus v2.3.2 - Precision, V5 API, TP/Exits, Robustness
    # fmt: on
    # pylint: enable=line-too-long
    """
    Pyrmethus - Termux Trading Spell (v2.3.2)

    Conjures market insights and executes trades on Bybit Futures using the
    V5 Unified Account API via CCXT.

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
        from colorama import Back, Fore, Style, init
        from dotenv import load_dotenv
        from rich.console import Console # Added Rich
        from rich.panel import Panel     # Added Rich
        from rich.table import Table     # Added Rich
        from rich.text import Text       # Added Rich
        from tabulate import tabulate # Keep for now? Or remove if fully Rich? Remove if Rich replaces panel.
        # Explicitly list common required packages for the error message
        COMMON_PACKAGES = ['ccxt', 'python-dotenv', 'pandas', 'numpy', 'tabulate', 'colorama', 'requests', 'rich'] # Added rich
    except ImportError as e:
        # Provide specific guidance for Termux users or general pip install
        init(autoreset=True) # Initialize colorama for error message
        missing_pkg = e.name
        print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
        print(f"{Fore.YELLOW}To conjure it, cast the following spell:")
        if os.getenv("TERMUX_VERSION"):
             print(f"{Style.BRIGHT}pkg install python && pip install {missing_pkg}{Style.RESET_ALL}")
        else:
             print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
        # Offer to install all common dependencies
        print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
        if os.getenv("TERMUX_VERSION"):
             # pandas, numpy might require system dependencies in Termux. Advise potentially separate steps.
             print(f"{Style.BRIGHT}pkg install python python-pandas python-numpy && pip install {' '.join([p for p in COMMON_PACKAGES if p not in ['pandas', 'numpy']])}{Style.RESET_ALL}")
             print(f"{Fore.YELLOW}Note: pandas and numpy are often installed via pkg in Termux for better compatibility.")
        else:
             print(f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}")
        sys.exit(1) # pylint: disable=consider-using-sys-exit

    # Weave the Colorama magic into the terminal (Rich handles its own colors)
    init(autoreset=True)
    # Create a Rich Console instance
    console = Console()

    # Set Decimal precision context for the entire application
    getcontext().prec = 50 # Ample precision for crypto

    # --- Arcane Configuration & Logging Setup ---
    logger = logging.getLogger(__name__)

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

    # More detailed log format (Rich handles console colors, keep basic format for file logging if added)
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s"
    )
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Rich Handler for Console Output
    # Ensure handlers are not duplicated
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        # Keep basic StreamHandler for now, Rich integration is primarily in print_status_panel
        # For full Rich logging: from rich.logging import RichHandler
        # stream_handler = RichHandler(console=console, show_time=True, show_level=True, show_path=True, markup=True)
        stream_handler = logging.StreamHandler(sys.stdout) # Basic handler
        stream_handler.setFormatter(log_formatter) # Apply basic formatter
        logger.addHandler(stream_handler)

    logger.propagate = False # Prevent duplicate messages


    class TradingConfig:
        """
        Holds the sacred parameters of our spell, loading from environment variables,
        performing validation, and determining API category.
        """
        # pylint: disable=too-many-instance-attributes # Config class naturally has many attributes
        def __init__(self):
            """Initializes configuration by loading and validating environment variables."""
            logger.debug("Loading configuration from environment variables...")
            self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW)
            self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW, allowed_values=['linear', 'inverse', 'swap']).lower()
            self.bybit_v5_category = self._determine_v5_category()

            self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
            # Financial parameters (Decimal)
            self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5"))
            self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0"))
            self.tp_atr_multiplier = self._get_env("TP_ATR_MULTIPLIER", "3.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.0"), max_val=Decimal("50.0")) # TP target (0 disables)
            self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0"))
            self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.01"), max_val=Decimal("10.0"))
            # Trigger types
            self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
            self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])

            # Indicator Periods (int)
            self.trend_ema_period = self._get_env("TREND_EMA_PERIOD", "12", Fore.YELLOW, cast_type=int, min_val=5, max_val=500)
            self.fast_ema_period = self._get_env("FAST_EMA_PERIOD", "9", Fore.YELLOW, cast_type=int, min_val=1, max_val=200)
            self.slow_ema_period = self._get_env("SLOW_EMA_PERIOD", "21", Fore.YELLOW, cast_type=int, min_val=2, max_val=500)
            self.stoch_period = self._get_env("STOCH_PERIOD", "7", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)
            self.stoch_smooth_k = self._get_env("STOCH_SMOOTH_K", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
            self.stoch_smooth_d = self._get_env("STOCH_SMOOTH_D", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
            self.atr_period = self._get_env("ATR_PERIOD", "5", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)

            # Signal Logic Thresholds (Decimal)
            self.stoch_oversold_threshold = self._get_env("STOCH_OVERSOLD_THRESHOLD", "30", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
            self.stoch_overbought_threshold = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", "70", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
            self.trend_filter_buffer_percent = self._get_env("TREND_FILTER_BUFFER_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
            self.atr_move_filter_multiplier = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))

            # Epsilon for zero checks
            self.position_qty_epsilon = Decimal("1E-12")
            logger.debug(f"Using fixed position_qty_epsilon for zero checks: {self.position_qty_epsilon:.1E}")

            # API Keys
            self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
            self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)

            # Operational Parameters (int)
            self.ohlcv_limit = self._get_env("OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000)
            self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5)
            self.order_check_delay_seconds = self._get_env("ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1)
            self.order_check_timeout_seconds = self._get_env("ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5)
            self.max_fetch_retries = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
            self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)

            # Journaling Configuration
            self.journal_file_path = self._get_env("JOURNAL_FILE_PATH", "pyrmethus_trading_journal.csv", Fore.YELLOW)
            self.enable_journaling = self._get_env("ENABLE_JOURNALING", "True", Fore.YELLOW, cast_type=bool)

            if not self.api_key or not self.api_secret:
                logger.critical(f"{Fore.RED+Style.BRIGHT}BYBIT_API_KEY or BYBIT_API_SECRET not found. Halting.")
                # pylint: disable=consider-using-sys-exit
                sys.exit(1)

            # --- Post-Load Validations ---
            self._validate_config()
            logger.debug("Configuration loaded and validated successfully.")

        def _determine_v5_category(self) -> str:
            """Determines the Bybit V5 API category based on symbol and market_type."""
            try:
                 parts = self.symbol.replace(':','/').split('/')
                 if len(parts) < 2: raise ValueError("Symbol format must be BASE/QUOTE[:SETTLE]")
                 base_curr = parts[0].upper(); settle_curr = parts[-1].upper()
                 category = '';
                 if self.market_type == 'inverse': category = 'inverse'
                 elif self.market_type in ['linear', 'swap']: category = 'linear' if settle_curr != base_curr else 'inverse'
                 else: raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}'")
                 if category == 'inverse' and self.market_type != 'inverse': logger.warning(f"Market type '{self.market_type}' with symbol '{self.symbol}' implies inverse. Setting category to 'inverse'.")
                 logger.info(f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}'")
                 return category
            except (ValueError, IndexError) as e:
                logger.critical(f"Could not parse symbol '{self.symbol}' for category: {e}. Halting.", exc_info=True)
                # pylint: disable=consider-using-sys-exit
                sys.exit(1)

        def _validate_config(self):
            """Performs post-load validation of configuration parameters."""
            if self.fast_ema_period >= self.slow_ema_period: logger.critical(f"{Fore.RED+Style.BRIGHT}FAST_EMA ({self.fast_ema_period}) must be < SLOW_EMA ({self.slow_ema_period}). Halting."); sys.exit(1) # pylint: disable=consider-using-sys-exit
            if self.trend_ema_period <= self.slow_ema_period: logger.warning(f"{Fore.YELLOW}TREND_EMA ({self.trend_ema_period}) <= SLOW_EMA ({self.slow_ema_period}). Consider increasing.")
            if self.stoch_oversold_threshold >= self.stoch_overbought_threshold: logger.critical(f"{Fore.RED+Style.BRIGHT}STOCH_OVERSOLD ({self.stoch_oversold_threshold.normalize()}) must be < STOCH_OVERBOUGHT ({self.stoch_overbought_threshold.normalize()}). Halting."); sys.exit(1) # pylint: disable=consider-using-sys-exit
            if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier: logger.warning(f"{Fore.YELLOW}TSL_ACT_MULT ({self.tsl_activation_atr_multiplier.normalize()}) < SL_MULT ({self.sl_atr_multiplier.normalize()}). TSL may activate early.")
            if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier: logger.warning(f"{Fore.YELLOW}TP_MULT ({self.tp_atr_multiplier.normalize()}) <= SL_MULT ({self.sl_atr_multiplier.normalize()}). Poor R:R setup.")

        # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
        def _get_env(self, key: str, default: Any, color: str, cast_type: type = str, min_val: Optional[Union[int, Decimal, float]] = None, max_val: Optional[Union[int, Decimal, float]] = None, allowed_values: Optional[List[str]] = None) -> Any:
            """Gets value from environment, casts, validates, and logs."""
            # ... (Logic remains the same, already cleaned) ...
            value_str = os.getenv(key); log_value = "****" if "SECRET" in key.upper() or "KEY" in key.upper() else value_str
            if value_str is None or value_str.strip() == "":
                if default is None and key not in ['BYBIT_API_KEY', 'BYBIT_API_SECRET']: logger.critical(f"{Fore.RED+Style.BRIGHT}Required config '{key}' not found. Halting."); sys.exit(1) # pylint: disable=consider-using-sys-exit
                value_str_for_cast = str(default) if default is not None else None
                if default is not None: logger.warning(f"{color}Using default for {key}: {default}")
            else: logger.info(f"{color}Summoned {key}: {log_value}"); value_str_for_cast = value_str
            if value_str_for_cast is None and default is None: logger.critical(f"{Fore.RED+Style.BRIGHT}Required config '{key}' missing, no default. Halting."); sys.exit(1) # pylint: disable=consider-using-sys-exit
            casted_value = None
            try:
                if cast_type == bool: casted_value = str(value_str_for_cast).lower() in ['true', '1', 'yes', 'y', 'on']
                elif cast_type == Decimal: casted_value = Decimal(str(value_str_for_cast))
                elif cast_type == int: casted_value = int(Decimal(str(value_str_for_cast)))
                elif cast_type == float: casted_value = float(str(value_str_for_cast))
                else: casted_value = str(value_str_for_cast)
            except (ValueError, TypeError, InvalidOperation) as e:
                logger.error(f"{Fore.RED}Cast failed for {key} ('{value_str_for_cast}' -> {cast_type.__name__}): {e}. Using default '{default}'.")
                try: # Re-cast default
                    if default is None: casted_value = None
                    elif cast_type == bool: casted_value = str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                    elif cast_type == Decimal: casted_value = Decimal(str(default))
                    elif cast_type == int: casted_value = int(Decimal(str(default)))
                    elif cast_type == float: casted_value = float(default)
                    else: casted_value = cast_type(default)
                except (ValueError, TypeError, InvalidOperation) as cast_default_err: logger.critical(f"{Fore.RED+Style.BRIGHT}Default '{default}' for {key} invalid ({cast_type.__name__}): {cast_default_err}. Halting."); sys.exit(1) # pylint: disable=consider-using-sys-exit
            validation_failed = False; error_message = ""
            if allowed_values:
                comp_value = str(casted_value).lower() if isinstance(casted_value, str) else casted_value; lower_allowed = [str(v).lower() for v in allowed_values]
                if comp_value not in lower_allowed: error_message = f"Invalid value '{casted_value}'. Allowed: {allowed_values}."; validation_failed = True
            if not validation_failed and isinstance(casted_value, (Decimal, int, float)):
                try:
                    min_val_comp = Decimal(str(min_val)) if isinstance(casted_value, Decimal) and min_val is not None else min_val; max_val_comp = Decimal(str(max_val)) if isinstance(casted_value, Decimal) and max_val is not None else max_val
                    if min_val_comp is not None and casted_value < min_val_comp: error_message = f"Value {casted_value} < min {min_val}."; validation_failed = True
                    if max_val_comp is not None and casted_value > max_val_comp: error_message = f"Value {casted_value} > max {max_val}."; validation_failed = True
                except (InvalidOperation, TypeError) as e: error_message = f"Min/max validation error: {e}."; validation_failed = True
            if validation_failed:
                logger.error(f"{Fore.RED}Validation failed for {key}: {error_message} Using default: {default}")
                try: # Re-cast default on validation failure
                    if default is None: return None
                    if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                    elif cast_type == Decimal: return Decimal(str(default))
                    elif cast_type == int: return int(Decimal(str(default)))
                    elif cast_type == float: return float(default)
                    return cast_type(default)
                except (ValueError, TypeError, InvalidOperation) as cast_default_err: logger.critical(f"{Fore.RED+Style.BRIGHT}Default '{default}' for {key} invalid on fallback: {cast_default_err}. Halting."); sys.exit(1) # pylint: disable=consider-using-sys-exit
            return casted_value

    # --- Instantiate Configuration ---
    logger.info(f"{Fore.MAGENTA+Style.BRIGHT}Initializing Arcane Configuration v2.3.2...")
    load_dotenv()
    CONFIG = TradingConfig()

    # --- Global Variables ---
    MARKET_INFO: Optional[Dict] = None
    EXCHANGE: Optional[ccxt.Exchange] = None
    order_tracker: Dict[str, Dict[str, Optional[str]]] = { # Tracks SL/TSL marker presence
        "long": {"sl_id": None, "tsl_id": None},
        "short": {"sl_id": None, "tsl_id": None}
    }
    shutdown_requested = False # Flag for graceful shutdown

    # --- Signal Handling for Graceful Shutdown ---
    def signal_handler(sig, frame):
        """Handles SIGINT/SIGTERM for graceful shutdown."""
        # pylint: disable=unused-argument, global-statement
        global shutdown_requested
        if not shutdown_requested:
            sig_name = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
            logger.warning(f"{Fore.YELLOW+Style.BRIGHT}\nSignal {sig_name} received. Initiating graceful shutdown...")
            shutdown_requested = True
        else:
            logger.warning("Shutdown already in progress.")

    signal.signal(signal.SIGINT, signal_handler) # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Termination signal

    # --- Core Spell Functions ---
    # ... (fetch_with_retries - remains the same) ...
    # ... (Exchange Nexus Initialization - remains the same) ...
    # ... (termux_notify - remains the same) ...
    # ... (_format_with_fallback, format_price, format_amount - remains the same) ...
    # ... (fetch_market_data - remains the same) ...
    # ... (calculate_indicators - remains the same) ...
    # ... (get_current_position - remains the same) ...
    # ... (get_balance - remains the same) ...
    # ... (check_order_status - remains the same) ...
    # ... (log_trade_entry_to_journal - remains the same) ...
    # ... (_calculate_trade_parameters - updated earlier) ...
    # ... (_execute_market_order - updated earlier) ...
    # ... (_set_position_stops - updated earlier) ...
    # ... (_handle_entry_failure - remains the same) ...
    # ... (place_risked_market_order - updated earlier) ...
    # ... (manage_trailing_stop - updated earlier) ...
    # ... (_remove_position_stops - added earlier) ...
    # ... (close_position - added earlier) ...
    # ... (generate_signals - remains the same) ...
    # ... (print_status_panel - updated earlier with Rich and TP) ...
    # (Including placeholder stubs for brevity, full functions are in previous responses)
    def fetch_with_retries(fetch_function, *args, **kwargs) -> Any: # Docstring in previous version
        # ... Full logic ...
        pass
    def termux_notify(title: str, content: str) -> None: # Docstring in previous version
        # ... Full logic ...
        pass
    def _format_with_fallback(symbol: str, value: Union[Decimal, str, float, int], exchange_method, rounding_mode=None, fallback_precision: Decimal = Decimal("1E-8"), fallback_rounding=ROUND_HALF_EVEN) -> str: # Docstring in previous version
        # ... Full logic ...
        pass
    def format_price(symbol: str, price: Union[Decimal, str, float, int]) -> str: # Docstring in previous version
        # ... Full logic ...
        pass
    def format_amount(symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str: # Docstring in previous version
        # ... Full logic ...
        pass
    def fetch_market_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]: # Docstring in previous version
        # ... Full logic ...
        pass
    def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]: # Docstring in previous version
        # ... Full logic ...
        pass
    def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Any]]]: # Docstring in previous version
        # ... Full logic ...
        pass
    def get_balance(currency: str) -> Tuple[Optional[Decimal], Optional[Decimal]]: # Docstring in previous version
        # ... Full logic ...
        pass
    def check_order_status(order_id: str, symbol: str, timeout: int) -> Optional[Dict]: # Docstring in previous version
        # ... Full logic ...
        pass
    def log_trade_entry_to_journal(symbol: str, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]) -> None: # Docstring in previous version
        # ... Full logic ...
        pass
    def _calculate_trade_parameters(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal, total_equity: Decimal) -> Optional[Dict]: # Docstring in previous version
        # ... Full logic ...
        pass
    def _execute_market_order(symbol: str, side: str, qty_decimal: Decimal) -> Optional[Dict]: # Docstring in previous version
        # ... Full logic ...
        pass
    def _set_position_stops(symbol: str, position_side: str, sl_price_str: str, tp_price_str: str) -> bool: # Docstring in previous version
        # ... Full logic ...
        pass
    def _handle_entry_failure(symbol: str, side: str, filled_qty_attempted: Decimal): # Docstring in previous version
        # ... Full logic ...
        pass
    def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool: # Docstring in previous version
        # ... Full logic ...
        pass
    def manage_trailing_stop(symbol: str, position_side: str, position_qty: Decimal, entry_price: Decimal, current_price: Decimal, atr: Decimal) -> None: # Docstring in previous version
        # ... Full logic ...
        pass
    def _remove_position_stops(symbol: str, position_side: str) -> bool: # Docstring in previous version
        # ... Full logic ...
        pass
    def close_position(symbol: str, position_side: str, position_qty: Decimal) -> bool: # Docstring in previous version
        # ... Full logic ...
        pass
    def generate_signals(df_last_candles: pd.DataFrame, indicators: Dict[str, Union[Decimal, bool, int]]) -> Dict[str, Union[bool, str]]: # Docstring in previous version
        # ... Full logic ...
        pass
    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches
    def print_status_panel(cycle: int, timestamp: Optional[pd.Timestamp], price: Optional[Decimal], indicators: Optional[Dict[str, Union[Decimal, bool, int]]], positions: Optional[Dict[str, Dict[str, Any]]], equity: Optional[Decimal], signals: Dict[str, Union[bool, str]], order_tracker_state: Dict[str, Dict[str, Optional[str]]]) -> None: # Docstring in previous version
        # ... Full Rich logic ...
        pass

    # --- Main Trading Cycle & Loop ---

    # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    def trading_spell_cycle(cycle_count: int) -> None:
        """Executes one cycle of the trading spell."""
        # pylint: disable=global-statement
        global order_tracker # Allow modification by sub-functions

        logger.info(f"{Fore.MAGENTA+Style.BRIGHT}\n--- Starting Cycle {cycle_count} ---")
        start_time = time.time()
        cycle_success = True # Tracks critical data fetching

        # --- 1. Fetch Market Data ---
        df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
        if df is None or df.empty: logger.error(f"{Fore.RED}Halting cycle: Market data failed."); end_time = time.time(); logger.info(f"{Fore.MAGENTA}--- Cycle {cycle_count} ABORTED (Data Fetch) (Duration: {end_time - start_time:.2f}s) ---"); return

        # --- 2. Get Current Price & Timestamp ---
        current_price: Optional[Decimal] = None; last_timestamp: Optional[pd.Timestamp] = None
        try:
            last_candle = df.iloc[-1]; current_price = Decimal(str(last_candle["close"])); last_timestamp = df.index[-1]
            if pd.isna(last_candle["close"]) or current_price <= 0: raise ValueError("Invalid latest close price")
            logger.debug(f"Latest candle: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}")
            # Stale data check (abbreviated)
            now_utc = pd.Timestamp.utcnow().tz_localize('UTC'); time_diff = now_utc - last_timestamp
            if EXCHANGE:
                 try: interval_s = EXCHANGE.parse_timeframe(CONFIG.interval); lag = pd.Timedelta(seconds=interval_s * 1.5 + 60);
                 if time_diff > lag: logger.warning(f"{Fore.YELLOW}Data may be stale: {time_diff} ago.")
                 except Exception as e: logger.warning(f"Staleness check error: {e}")
        except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e: logger.error(f"{Fore.RED}Halting cycle: Price/Timestamp error: {e}", exc_info=True); end_time = time.time(); logger.info(f"{Fore.MAGENTA}--- Cycle {cycle_count} ABORTED (Price Proc) (Duration: {end_time - start_time:.2f}s) ---"); return

        # --- 3. Calculate Indicators ---
        indicators = calculate_indicators(df)
        if indicators is None: logger.error(f"{Fore.RED}Indicator calculation failed. Skipping trade logic."); cycle_success = False
        current_atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')

        # --- 4. Get Current State (Balance & Positions) ---
        quote_currency = MARKET_INFO.get('settle', 'USDT') if MARKET_INFO else 'USDT'
        _, current_equity = get_balance(quote_currency) # Use equity for risk calc
        if current_equity is None or current_equity.is_nan() or current_equity <= 0: logger.error(f"{Fore.RED}Failed fetching valid equity. Skipping trade logic."); cycle_success = False
        positions = get_current_position(CONFIG.symbol)
        if positions is None: logger.error(f"{Fore.RED}Failed fetching positions. Skipping trade logic."); cycle_success = False

        # --- Capture State Snapshot ---
        order_tracker_snapshot = copy.deepcopy(order_tracker)
        positions_snapshot = positions if positions is not None else {"long": {}, "short": {}}
        final_positions_for_panel = positions_snapshot; final_order_tracker_state_for_panel = order_tracker_snapshot

        # --- Initialize signals ---
        signals: Dict[str, Union[bool, str]] = {"long": False, "short": False, "reason": "Skipped: Initial State"}

        # --- Main Logic (Requires valid initial state fetches) ---
        can_run_trade_logic = cycle_success

        if can_run_trade_logic:
            # Get live state from initial fetch
            active_long_pos = positions.get('long', {}); active_short_pos = positions.get('short', {})
            active_long_qty = active_long_pos.get('qty', Decimal('0.0')); active_short_qty = active_short_pos.get('qty', Decimal('0.0'))
            active_long_entry = active_long_pos.get('entry_price', Decimal('NaN')); active_short_entry = active_short_pos.get('entry_price', Decimal('NaN'))
            has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon; has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
            is_flat = not has_long_pos and not has_short_pos

            # --- 5. Manage Trailing Stops ---
            if (has_long_pos or has_short_pos) and indicators and not current_price.is_nan() and not current_atr.is_nan():
                 pos_side = "long" if has_long_pos else "short"; pos_entry = active_long_entry if has_long_pos else active_short_entry; pos_qty = active_long_qty if has_long_pos else active_short_qty
                 if not pos_entry.is_nan() and pos_entry > 0: manage_trailing_stop(CONFIG.symbol, pos_side, pos_qty, pos_entry, current_price, current_atr)
                 else: logger.warning(f"Cannot manage TSL for {pos_side.upper()}: Invalid entry price ({pos_entry}).")
            elif is_flat and any(order_tracker[s][k] for s in ["long", "short"] for k in ["sl_id", "tsl_id"]): logger.info("Position flat, clearing trackers."); order_tracker["long"].update({"sl_id": None, "tsl_id": None}); order_tracker["short"].update({"sl_id": None, "tsl_id": None}); final_order_tracker_state_for_panel = copy.deepcopy(order_tracker)

            # --- Re-fetch position state AFTER TSL management ---
            logger.debug("Re-fetching position state after TSL check..."); positions_after_tsl = get_current_position(CONFIG.symbol)
            if positions_after_tsl is None: logger.error(f"{Fore.RED}Failed re-fetching positions after TSL check. State uncertain."); cycle_success = False; signals["reason"] = "Skipped: Pos re-fetch failed"
            else: # Update live state variables
                 active_long_pos=positions_after_tsl.get('long',{}); active_short_pos=positions_after_tsl.get('short',{})
                 active_long_qty=active_long_pos.get('qty',Decimal('0.0')); active_short_qty=active_short_pos.get('qty',Decimal('0.0'))
                 has_long_pos=active_long_qty.copy_abs()>=CONFIG.position_qty_epsilon; has_short_pos=active_short_qty.copy_abs()>=CONFIG.position_qty_epsilon
                 is_flat = not has_long_pos and not has_short_pos; logger.debug(f"State After TSL Check: Flat={is_flat}, Long={active_long_qty.normalize()}, Short={active_short_qty.normalize()}")
                 final_positions_for_panel = positions_after_tsl; final_order_tracker_state_for_panel = copy.deepcopy(order_tracker) # Update snapshots
                 if is_flat and any(order_tracker[s][k] for s in ["long", "short"] for k in ["sl_id", "tsl_id"]): logger.info("Position became flat after TSL check, clearing trackers."); order_tracker["long"].update({"sl_id": None, "tsl_id": None}); order_tracker["short"].update({"sl_id": None, "tsl_id": None}); final_order_tracker_state_for_panel = copy.deepcopy(order_tracker)

                 # --- 6. Generate Trading Signals ---
                 can_gen_signals = indicators is not None and not current_price.is_nan() and len(df) >= 2
                 if can_gen_signals: signals = generate_signals(df.iloc[-2:], indicators)
                 else: reason = "Skipped Signal Gen: " + ("Indicators missing. " if indicators is None else "") + (f"Need >=2 candles ({len(df)} found)." if len(df) < 2 else ""); signals = {"long": False, "short": False, "reason": reason.strip()}; logger.warning(signals['reason'])

                 # --- 7. Check for Signal-Based Exits ---
                 exit_triggered = False
                 if can_gen_signals and not is_flat: # Only check exits if signals are valid and position exists
                     fast_ema = indicators.get('fast_ema', Decimal('NaN')); slow_ema = indicators.get('slow_ema', Decimal('NaN'))
                     if not fast_ema.is_nan() and not slow_ema.is_nan():
                         ema_bear_cross = fast_ema < slow_ema; ema_bull_cross = fast_ema > slow_ema
                         if has_long_pos and ema_bear_cross: logger.trade(f"{Fore.YELLOW}EMA Bearish Cross vs LONG. Closing."); exit_triggered = close_position(CONFIG.symbol, "long", active_long_qty)
                         elif has_short_pos and ema_bull_cross: logger.trade(f"{Fore.YELLOW}EMA Bullish Cross vs SHORT. Closing."); exit_triggered = close_position(CONFIG.symbol, "short", active_short_qty)
                     if not exit_triggered and (has_long_pos or has_short_pos): logger.debug("No counter-EMA signal for exit.")

                 # --- Re-fetch state AGAIN if an exit was triggered ---
                 if exit_triggered:
                      logger.debug("Re-fetching state after signal exit attempt..."); positions_after_exit = get_current_position(CONFIG.symbol)
                      if positions_after_exit is None: logger.error(f"{Fore.RED}Failed re-fetching positions after signal exit."); cycle_success = False # Mark failure
                      else: # Update live state variables
                          active_long_pos=positions_after_exit.get('long',{}); active_short_pos=positions_after_exit.get('short',{})
                          active_long_qty=active_long_pos.get('qty',Decimal('0.0')); active_short_qty=active_short_pos.get('qty',Decimal('0.0'))
                          has_long_pos=active_long_qty.copy_abs()>=CONFIG.position_qty_epsilon; has_short_pos=active_short_qty.copy_abs()>=CONFIG.position_qty_epsilon
                          is_flat = not has_long_pos and not has_short_pos; logger.debug(f"State After Signal Exit: Flat={is_flat}, Long={active_long_qty.normalize()}, Short={active_short_qty.normalize()}")
                          final_positions_for_panel = positions_after_exit; final_order_tracker_state_for_panel = copy.deepcopy(order_tracker) # Update snapshots

                 # --- 8. Execute Entry Trades (Only if now flat) ---
                 if is_flat and can_gen_signals and not current_atr.is_nan() and (signals.get("long") or signals.get("short")):
                     entry_attempted = False; entry_successful = False
                     if signals.get("long"): logger.info(f"{Fore.GREEN+Style.BRIGHT}Long signal! {signals.get('reason', '')}. Attempting entry..."); entry_attempted = True; entry_successful = place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr)
                     elif signals.get("short"): logger.info(f"{Fore.RED+Style.BRIGHT}Short signal! {signals.get('reason', '')}. Attempting entry..."); entry_attempted = True; entry_successful = place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr)
                     if entry_attempted:
                          if entry_successful: logger.info(f"{Fore.GREEN}Entry process completed.")
                          else: logger.error(f"{Fore.RED}Entry process failed."); cycle_success = False # Mark cycle failure on entry fail
                          logger.debug("Re-fetching state after entry attempt..."); positions_after_entry = get_current_position(CONFIG.symbol) # Re-fetch for panel accuracy
                          if positions_after_entry is not None: final_positions_for_panel = positions_after_entry; final_order_tracker_state_for_panel = copy.deepcopy(order_tracker)
                          else: logger.warning("Failed re-fetching positions after entry attempt. Panel may be stale.")
                 elif is_flat: logger.debug("Position flat, no entry signal.")
                 elif not is_flat: logger.debug(f"Position ({'LONG' if has_long_pos else 'SHORT'}) remains open, skipping entry.")

        else: # Initial critical data fetch failed
             signals["reason"] = "Skipped: Critical data missing (Equity/Position fetch failed)"; logger.warning(signals["reason"])

        # --- 9. Display Status Panel ---
        print_status_panel(cycle_count, last_timestamp, current_price, indicators, final_positions_for_panel, current_equity, signals, final_order_tracker_state_for_panel)

        end_time = time.time(); status_log = "Complete" if cycle_success else "Completed with WARNINGS/ERRORS"; logger.info(f"{Fore.MAGENTA}--- Cycle {cycle_count} {status_log} (Duration: {end_time - start_time:.2f}s) ---")

    # pylint: disable=too-many-statements
    def graceful_shutdown() -> None: # Docstring in previous version
        # ... (graceful_shutdown logic - updated earlier) ...
        # pylint: disable=global-statement
        global order_tracker; logger.warning(f"{Fore.YELLOW+Style.BRIGHT}\nInitiating Graceful Shutdown Sequence..."); termux_notify("Shutdown", f"Closing {CONFIG.symbol}...");
        if EXCHANGE is None or MARKET_INFO is None: logger.error(f"{Fore.RED}Cannot shutdown cleanly: Exchange/Market Info missing."); termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Cannot shutdown cleanly."); return
        symbol = CONFIG.symbol; market_id = MARKET_INFO.get('id')
        try: # Cancel Orders
            logger.info(f"{Fore.CYAN}Attempting cancel cancellable orders for {symbol}..."); cancelled = False
            if hasattr(EXCHANGE, 'private_post_order_cancel_all'):
                 try: logger.info("Using V5 cancel_all..."); params = {'category': CONFIG.bybit_v5_category, 'symbol': market_id}; response = fetch_with_retries(EXCHANGE.private_post_order_cancel_all, params=params)
                 if response and response.get('retCode') == 0: logger.info(f"{Fore.GREEN}V5 Cancel all successful."); cancelled = True
                 else: logger.warning(f"V5 Cancel all response unclear/failed: {response}")
                 except Exception as e: logger.warning(f"V5 cancel_all exception: {e}")
            if not cancelled:
                 try: logger.info("Attempting generic cancelAllOrders..."); cancel_resp = fetch_with_retries(EXCHANGE.cancel_all_orders, symbol); logger.info(f"cancelAllOrders response: {cancel_resp}")
                 except Exception as e: logger.error(f"{Fore.RED}cancelAllOrders fallback error: {e}", exc_info=True)
            logger.info("Clearing local order tracker."); order_tracker["long"].update({"sl_id": None, "tsl_id": None}); order_tracker["short"].update({"sl_id": None, "tsl_id": None})
        except Exception as e: logger.error(f"{Fore.RED+Style.BRIGHT}Order cancellation phase error: {e}. MANUAL CHECK REQUIRED.", exc_info=True)
        logger.info("Waiting after order cancellation..."); time.sleep(max(CONFIG.order_check_delay_seconds, 3))
        try: # Close Positions
            logger.info(f"{Fore.CYAN}Checking for lingering positions to close..."); positions = get_current_position(symbol); closed_count = 0
            if positions:
                positions_to_close = {s: d for s, d in positions.items() if d.get('qty') and d['qty'].copy_abs() >= CONFIG.position_qty_epsilon}
                if not positions_to_close: logger.info(f"{Fore.GREEN}No significant positions found.")
                else:
                    logger.warning(f"{Fore.YELLOW}Found {len(positions_to_close)} positions requiring closure.")
                    for side, pos_data in positions_to_close.items():
                         qty = pos_data.get('qty', Decimal("0.0"))
                         if close_position(symbol, side, qty): closed_count += 1 # Use dedicated close function
                         else: logger.error(f"Closure failed for {side.upper()}. MANUAL INTERVENTION REQUIRED.")
                    if closed_count == len(positions_to_close): logger.info(f"{Fore.GREEN}All detected positions closed successfully.")
                    else: logger.warning(f"{Fore.YELLOW}Attempted {len(positions_to_close)} closures, {closed_count} orders submitted. MANUAL VERIFICATION REQUIRED.")
            else: logger.error(f"{Fore.RED}Failed fetching positions during shutdown. MANUAL CHECK REQUIRED.")
        except Exception as e: logger.error(f"{Fore.RED+Style.BRIGHT}Position closure phase error: {e}. MANUAL CHECK REQUIRED.", exc_info=True)
        logger.warning(f"{Fore.YELLOW+Style.BRIGHT}Graceful Shutdown Sequence Complete. Pyrmethus rests.")
        termux_notify("Shutdown Complete", f"{CONFIG.symbol} shutdown finished.")

    # --- Main Execution Block ---
    if __name__ == "__main__":
        # Use console.print for Rich formatting at startup
        console.print(f"[bold bright_cyan] summoning Pyrmethus [magenta]v2.3.2[/]...")
        console.print(f"[yellow]Trading Symbol: [white]{CONFIG.symbol}[/] | Interval: [white]{CONFIG.interval}[/] | Category: [white]{CONFIG.bybit_v5_category}[/]")
        console.print(f"[yellow]Risk: [white]{CONFIG.risk_percentage:.3%}[/] | SL: [white]{CONFIG.sl_atr_multiplier.normalize()}x[/] | TP: [white]{CONFIG.tp_atr_multiplier.normalize()}x[/] | TSL Act: [white]{CONFIG.tsl_activation_atr_multiplier.normalize()}x[/] | TSL %: [white]{CONFIG.trailing_stop_percent.normalize()}%[/]")
        console.print(f"[yellow]Trend Filter: [white]{'ON' if CONFIG.trade_only_with_trend else 'OFF'}[/] | ATR Move Filter: [white]{CONFIG.atr_move_filter_multiplier.normalize()}x[/]")
        console.print(f"[yellow]Journaling: [white]{'Enabled' if CONFIG.enable_journaling else 'Disabled'}[/] ([dim]{CONFIG.journal_file_path}[/])")
        termux_notify("Pyrmethus Started", f"{CONFIG.symbol} @ {CONFIG.interval}")

        cycle_count = 0
        while not shutdown_requested:
            cycle_count += 1
            try:
                trading_spell_cycle(cycle_count)
            except KeyboardInterrupt: logger.warning("\nCtrl+C detected in main loop. Shutting down."); shutdown_requested = True
            except ccxt.AuthenticationError as e: logger.critical(f"{Fore.RED+Style.BRIGHT}CRITICAL AUTH ERROR: {e}. Halting."); graceful_shutdown(); sys.exit(1) # pylint: disable=consider-using-sys-exit
            except Exception as e: # Catch-all for unexpected errors *within* trading_spell_cycle
                logger.error(f"{Fore.RED+Style.BRIGHT}Unhandled exception in main loop (Cycle {cycle_count}): {e}", exc_info=True)
                logger.error(f"{Fore.RED}Continuing loop, but caution advised. Check logs.")
                termux_notify("Pyrmethus Error", f"Unhandled exception cycle {cycle_count}.")
                sleep_time = CONFIG.loop_sleep_seconds * 2 # Longer sleep after error
            else: sleep_time = CONFIG.loop_sleep_seconds # Normal sleep

            if shutdown_requested: logger.info("Shutdown requested, breaking main loop."); break
            # Interruptible Sleep
            logger.debug(f"Cycle {cycle_count} finished. Sleeping for {sleep_time} seconds...")
            sleep_end_time = time.time() + sleep_time
            try:
                while time.time() < sleep_end_time and not shutdown_requested: time.sleep(0.5)
            except KeyboardInterrupt: logger.warning("\nCtrl+C during sleep."); shutdown_requested = True

        # Perform Graceful Shutdown
        graceful_shutdown()
        console.print(f"[bold bright_cyan]Pyrmethus has returned to the ether.[/]")
        sys.exit(0) # Clean exit