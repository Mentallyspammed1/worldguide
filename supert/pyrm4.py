Okay, let's analyze the traceback and the latest code (`pyrm.py`, v2.0.1, which seems to be the base before the fix attempt) to create an enhanced and corrected version `v2.4.0`.

**Traceback Analysis:**

1.  **`Critical error during bot setup phase: unsupported operand type(s) for -: 'str' and 'int'`:**
    *   **Location:** Occurs in the `main` function, specifically during the configuration summary logging.
    *   **Cause:** The line `logger.info(f"  {'Margin Buffer:':<25} {((CONFIG.required_margin_buffer - 1) * 100):.1f}%")` is attempting to subtract the integer `1` from `CONFIG.required_margin_buffer`. This error indicates that `CONFIG.required_margin_buffer` was loaded as a *string* (e.g., `"1.05"`) instead of being cast to a `Decimal`.
    *   **Reason:** The `_get_env` function's fallback logic didn't explicitly cast the *default* value if the environment variable was missing or failed parsing. The default `"1.05"` was returned as a string.
    *   **Fix:** Modify `_get_env` to ensure the default value is also cast to the target `cast_type` if it's used.

2.  **`Close Position (Shutdown Request): Failed: AttributeError - 'NoneType' object has no attribute 'get'`:**
    *   **Location:** Occurs in the `close_position` function during `graceful_shutdown`.
    *   **Cause:** The `exchange.create_market_order(...)` call returned `None` instead of the expected order dictionary. The subsequent code then tried to call `.get('id')` (or similar) on this `None` object.
    *   **Reason:** While unusual for CCXT (which typically raises exceptions on failure), returning `None` could happen due to unhandled internal errors, edge cases, or perhaps specific exchange API behavior not fully mapped by CCXT.
    *   **Fix:** Add an explicit check `if order is None:` immediately after the `create_market_order` call and handle it as a failure.

**Enhancement Plan (Incorporating Previous Ideas + New Fixes for v2.4.0):**

1.  **Fix Bugs:** Correct the `_get_env` default value casting and add the `None` check after `create_market_order`.
2.  **Refactor Order Placement:** Keep the refactored `place_risked_market_order` structure from the v2.3.0 plan (using helpers `_calculate_initial_sl_price`, `_validate_and_cap_quantity`, `_check_margin_availability`, `_place_and_confirm_entry_order`, `_place_stop_order`).
3.  **Optimize Indicators:** Implement logic in `trade_logic` to calculate only necessary indicators based on `CONFIG.strategy_name`.
4.  **Improve SL/TSL Failure Handling:**
    *   Make alerts extremely clear if SL/TSL fails after successful entry.
    *   Add a configuration option `EMERGENCY_CLOSE_ON_SL_FAIL` (default `False`) to allow automatic closure if *initial fixed SL placement* fails (TSL failure is less critical if fixed SL succeeded).
5.  **Enhance Config:** Add `SL_TRIGGER_BY`, `TSL_TRIGGER_BY` (already planned). Add range validation to more parameters.
6.  **Robustness:** Rigorous `None`/`NaN` checks. Consistent `ROUND_DOWN` for quantities. Handle `DivisionByZero`.
7.  **State Management:** Add a simple state machine (`IDLE`, `ENTERING`, `IN_POSITION`, `CLOSING`) to prevent conflicting actions.
8.  **Logging:** Add estimated PnL% to state logging. Add entry/exit PnL to `close_position` logs. Use `log_prefix` consistently.
9.  **Clarity:** Use `CONFIG` constants (`SIDE_BUY`, etc.). Expand type hints. Improve docstrings.
10. **Version Bump:** v2.4.0.

**Refactored and Enhanced Code (v2.4.0):**

```python
# --- START OF FILE pyrmethus_scalper_v2.4.0.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.4.0 (Refactored & State Managed)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and improved robustness.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.4.0 (Unified: Selectable Strategies + Precision + Native SL/TSL + Refactoring + State Mgt + Robustness)

Features:
- Multiple strategies selectable via config.
- Optimized Indicator Calculation: Calculates only necessary indicators.
- Enhanced Precision: Uses Decimal with explicit rounding (ROUND_DOWN for quantities).
- Native SL/TSL Placement: Places exchange-native orders post-entry.
- Robust SL/TSL Handling: Clear alerts on failure; Optional emergency close on initial SL fail.
- Refactored Order Placement Logic for clarity.
- ATR-based initial Stop-Loss calculation.
- Optional Volume/Order Book entry confirmations.
- Risk-based position sizing with margin checks.
- Termux SMS alerts for critical events & trades.
- Robust error handling (NaN, DivisionByZero, API errors).
- Graceful shutdown with cleanup attempt.
- Stricter position detection (Bybit V5 API).
- Simple State Management (IDLE, ENTERING, IN_POSITION, CLOSING) to prevent overlaps.
- Estimated PnL logging.

Disclaimer:
- **EXTREME RISK**: Educational purposes ONLY. Use at own absolute risk.
- **NATIVE SL/TSL DEPENDENCE**: Relies on exchange performance & reliability.
- Parameter Sensitivity: Requires significant tuning and testing.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: Requires Termux:API app and package.
- API Changes: Targets Bybit V5 via CCXT; updates may be needed.
"""

# Standard Library Imports
import logging
import os
import sys
import time
import traceback
import subprocess
from typing import Dict, Optional, Any, Tuple, List, Union, Literal
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation, DivisionByZero

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta # type: ignore[import]
    from dotenv import load_dotenv
    from colorama import init as colorama_init, Fore, Style, Back
except ImportError as e:
    missing_pkg = e.name
    print(f"\033[91mMissing essential spell component: \033[1m{missing_pkg}\033[0m")
    print("\033[93mTo conjure it, cast:\033[0m")
    print(f"\033[1m\033[96mpip install {missing_pkg}\033[0m")
    print("\n\033[96mOr install all:\033[0m")
    print("\033[1m\033[96mpip install ccxt pandas pandas_ta python-dotenv colorama\033[0m")
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 28 # Global Decimal precision
DECIMAL_NaN = Decimal('NaN')
DECIMAL_ZERO = Decimal('0')
DECIMAL_ONE = Decimal('1')
DECIMAL_HUNDRED = Decimal('100')
DEFAULT_QUANTIZE_EXP = Decimal('1e-8') # Default fallback quantization

# --- Bot State Enum ---
class BotState:
    IDLE = "Idle"
    ENTERING = "Entering"
    IN_POSITION = "InPosition"
    CLOSING = "Closing"
    ERROR = "Error" # Added state for critical errors

# --- Global State Variable ---
current_bot_state: str = BotState.IDLE

# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables."""
    # Define constants for sides, position status, and triggers
    SIDE_BUY: Literal['buy'] = "buy"
    SIDE_SELL: Literal['sell'] = "sell"
    POS_LONG: Literal['Long'] = "Long"
    POS_SHORT: Literal['Short'] = "Short"
    POS_NONE: Literal['None'] = "None"
    TRIGGER_BY_LAST: Literal['LastPrice'] = "LastPrice"
    TRIGGER_BY_MARK: Literal['MarkPrice'] = "MarkPrice"
    TRIGGER_BY_INDEX: Literal['IndexPrice'] = "IndexPrice"
    VALID_TRIGGER_TYPES: List[str] = [TRIGGER_BY_LAST, TRIGGER_BY_MARK, TRIGGER_BY_INDEX]

    def __init__(self) -> None:
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes (v2.4.0) ---{Style.RESET_ALL}")
        # --- API Credentials ---
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: Optional[str] = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW, min_val=1, max_val=100)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW, min_val=1)
        self.order_retry_delay: int = self._get_env("ORDER_RETRY_DELAY", 2, cast_type=int, color=Fore.YELLOW, min_val=1)

        # --- Strategy Selection ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: List[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies: raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid: {self.valid_strategies}")
        logger.info(f"Selected Strategy: {Fore.CYAN}{self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", Decimal("0.005"), cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.0001"), max_val=Decimal("0.1"))
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", Decimal("1.5"), cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.1"))
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", Decimal("500.0"), cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("1.0"))
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", Decimal("1.05"), cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("1.0"))

        # --- SL/TSL Configuration ---
        self.sl_trigger_by: str = self._get_env("SL_TRIGGER_BY", self.TRIGGER_BY_LAST, color=Fore.GREEN)
        if self.sl_trigger_by not in self.VALID_TRIGGER_TYPES: raise ValueError(f"Invalid SL_TRIGGER_BY: {self.sl_trigger_by}")
        self.tsl_trigger_by: str = self._get_env("TSL_TRIGGER_BY", self.TRIGGER_BY_LAST, color=Fore.GREEN)
        if self.tsl_trigger_by not in self.VALID_TRIGGER_TYPES: raise ValueError(f"Invalid TSL_TRIGGER_BY: {self.tsl_trigger_by}")
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", Decimal("0.005"), cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.0001"), max_val=Decimal("0.1"))
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", Decimal("0.001"), cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.0001"), max_val=Decimal("0.05"))
        self.emergency_close_on_sl_fail: bool = self._get_env("EMERGENCY_CLOSE_ON_SL_FAIL", False, cast_type=bool, color=Fore.RED) # Default to False for safety

        # --- Indicator Parameters (Common) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN, min_val=2)
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW, min_val=2)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", Decimal("1.5"), cast_type=Decimal, color=Fore.YELLOW, min_val=Decimal("0.1"))
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", False, cast_type=bool, color=Fore.YELLOW)

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW, min_val=1, max_val=50)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", Decimal("1.2"), cast_type=Decimal, color=Fore.YELLOW, min_val=DECIMAL_ZERO)
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", Decimal("0.8"), cast_type=Decimal, color=Fore.YELLOW, min_val=DECIMAL_ZERO)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", False, cast_type=bool, color=Fore.YELLOW)

        # --- Strategy Specific Parameters ---
        # Load only parameters relevant to the selected strategy
        self._load_strategy_params()

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", False, cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA, min_val=5)

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = 10000
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW, min_val=5)

        # --- Internal Constants ---
        self.USDT_SYMBOL: str = "USDT"
        self.RETRY_COUNT: int = 3
        self.RETRY_DELAY_SECONDS: int = self.order_retry_delay
        self.API_FETCH_LIMIT_BUFFER: int = 50
        self.POSITION_QTY_EPSILON: Decimal = Decimal("1e-9")
        self.POST_CLOSE_DELAY_SECONDS: int = 3
        self.MARKET_ORDER_FILL_CHECK_INTERVAL: float = 0.5

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned Successfully ---{Style.RESET_ALL}")

    def _load_strategy_params(self):
        """Loads parameters specific to the selected strategy."""
        if self.strategy_name == "DUAL_SUPERTREND":
            self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN, min_val=2)
            self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", Decimal("2.5"), cast_type=Decimal, color=Fore.CYAN, min_val=Decimal("0.1"))
            self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN, min_val=2)
            self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", Decimal("2.0"), cast_type=Decimal, color=Fore.CYAN, min_val=Decimal("0.1"))
        elif self.strategy_name == "STOCHRSI_MOMENTUM":
            self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN, min_val=2)
            self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN, min_val=2)
            self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN, min_val=1)
            self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN, min_val=1)
            self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", Decimal("80.0"), cast_type=Decimal, color=Fore.CYAN, min_val=Decimal("50"), max_val=Decimal("100"))
            self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", Decimal("20.0"), cast_type=Decimal, color=Fore.CYAN, min_val=DECIMAL_ZERO, max_val=Decimal("50"))
            self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN, min_val=1)
        elif self.strategy_name == "EHLERS_FISHER":
            self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN, min_val=2)
            self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN, min_val=1)
        elif self.strategy_name == "EHLERS_MA_CROSS":
            self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN, min_val=2)
            self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN, min_val=3)
            if self.ehlers_fast_period >= self.ehlers_slow_period: raise ValueError("EHLERS_FAST_PERIOD must be less than EHLERS_SLOW_PERIOD")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE, min_val: Optional[Union[int, Decimal]] = None, max_val: Optional[Union[int, Decimal]] = None) -> Any:
        """Fetches env var, casts, validates ranges, logs, handles defaults/errors."""
        value_str = os.getenv(key)
        value: Any = None
        log_source = ""

        if value_str is not None:
            # Attempt casting from environment variable string
            log_source = f"(from env: '{value_str}')"
            try:
                if cast_type == bool: value = value_str.lower() in ['true', '1', 'yes', 'y']
                elif cast_type == Decimal: value = Decimal(value_str)
                elif cast_type is int: value = int(value_str)
                elif cast_type is float: value = float(value_str)
                elif cast_type is str: value = str(value_str)
                else: value = cast_type(value_str) # For other types
            except (ValueError, TypeError, InvalidOperation) as e:
                logger.error(f"{Fore.RED}Invalid type/value for {key}: '{value_str}'. Expected {cast_type.__name__}. Error: {e}. Using default: '{default}'{Style.RESET_ALL}")
                value = None # Set to None to trigger default usage below
        else:
             value = None # Explicitly None if env var not found

        if value is None:
             # Use default value if env var missing or parsing failed
             value = default
             log_source = f"(using default: '{default}')" if default is not None else "(not set, no default)"
             # IMPORTANT: Cast the default value itself if it's being used
             if value is not None:
                 try:
                     if cast_type == bool and isinstance(value, str): value = value.lower() in ['true', '1', 'yes', 'y']
                     elif cast_type == Decimal and not isinstance(value, Decimal): value = Decimal(str(value))
                     elif cast_type is int and not isinstance(value, int): value = int(value)
                     elif cast_type is float and not isinstance(value, float): value = float(value)
                     # String default needs no cast if cast_type is str
                     # Add other default casts if needed
                 except (ValueError, TypeError, InvalidOperation) as e:
                     logger.error(f"{Fore.RED}Failed to cast DEFAULT value '{default}' for {key} to {cast_type.__name__}: {e}. Setting to None.{Style.RESET_ALL}")
                     value = None # Set to None if default casting fails
                     log_source = f"(default cast failed)"


        # Validate required constraint *after* attempting default
        if value is None and required:
            critical_msg = f"CRITICAL: Required env var '{key}' not set or invalid, and no valid default."
            logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{critical_msg}{Style.RESET_ALL}")
            raise ValueError(critical_msg)

        # Validate numeric ranges *after* successful casting (of env var or default)
        if value is not None and cast_type in [int, Decimal, float]:
            is_adjusted = False
            original_value = value
            if min_val is not None and value < min_val: value = min_val; is_adjusted = True
            if max_val is not None and value > max_val: value = max_val; is_adjusted = True
            if is_adjusted: logger.warning(f"{Fore.YELLOW}Value '{original_value}' for {key} out of range [{min_val}, {max_val}]. Adjusted to {value}.{Style.RESET_ALL}")

        # Final log (redact secrets)
        log_display_value = "****" if "SECRET" in key.upper() else value
        logger.debug(f"{color}Config {key}: {log_display_value} {log_source}{Style.RESET_ALL}")
        return value


# --- Logger Setup ---
# (Keep logger setup from v2.3.0)
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger: logging.Logger = logging.getLogger(__name__)
logger.propagate = False
SUCCESS_LEVEL: int = 25; logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs) # type: ignore
logging.Logger.success = log_success # type: ignore[attr-defined]
if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")


# --- Global Objects ---
try:
    CONFIG = Config()
except ValueError as e: logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Configuration Error: {e}{Style.RESET_ALL}"); sys.exit(1)
except Exception as e: logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Unexpected Error initializing configuration: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); sys.exit(1)


# --- Helper & Utility Functions ---
# (Keep refined versions from v2.3.0)
def safe_decimal_conversion(value: Any, default: Decimal = DECIMAL_NaN) -> Decimal:
    if value is None: return default
    try: return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError): logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using {default}"); return default
def format_order_id(order_id: Optional[Union[str, int]]) -> str: return f"...{str(order_id)[-6:]}" if order_id else "N/A"
def get_market_base_currency(symbol: str) -> str:
    try: return symbol.split('/')[0]
    except IndexError: logger.warning(f"Could not parse base currency from symbol: {symbol}"); return symbol
def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal, str]) -> str:
    try: return exchange.price_to_precision(symbol, float(price))
    except Exception as e: logger.error(f"{Fore.RED}FmtPrice Err '{price}': {e}{Style.RESET_ALL}"); try: return str(Decimal(str(price)).normalize()) except: return str(price)
def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal, str], round_direction=ROUND_DOWN) -> str:
    try: return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}FmtAmt Err '{amount}': {e}{Style.RESET_ALL}")
        try:
            prec_str = exchange.markets[symbol].get('precision', {}).get('amount')
            if prec_str: places = Decimal(str(prec_str)).normalize().as_tuple().exponent*-1; return str(Decimal(str(amount)).quantize(Decimal(f'1e-{places}'), rounding=round_direction))
            return str(Decimal(str(amount)).quantize(DEFAULT_QUANTIZE_EXP, rounding=round_direction))
        except: return str(amount)
def send_sms_alert(message: str) -> bool:
    if not CONFIG.enable_sms_alerts: logger.debug("SMS disabled."); return False
    if not CONFIG.sms_recipient_number: logger.warning("SMS enabled, but no recipient number."); return False
    try:
        safe_msg = message.replace('"', "'").replace("`", "'").replace("$", "")
        cmd: List[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, safe_msg]
        logger.info(f"{Fore.MAGENTA}Attempt SMS to {'*'+CONFIG.sms_recipient_number[-4:]} (Timeout: {CONFIG.sms_timeout_seconds}s): \"{safe_msg[:60]}...\"{Style.RESET_ALL}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if res.returncode == 0: logger.success(f"{Fore.MAGENTA}SMS OK.{Style.RESET_ALL}"); return True
        else: logger.error(f"{Fore.RED}SMS Fail RC:{res.returncode}, Err:{res.stderr.strip() or 'N/A'}{Style.RESET_ALL}"); return False
    except FileNotFoundError: logger.error(f"{Fore.RED}SMS Fail: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}"); return False
    except subprocess.TimeoutExpired: logger.error(f"{Fore.RED}SMS Fail: Timeout ({CONFIG.sms_timeout_seconds}s).{Style.RESET_ALL}"); return False
    except Exception as e: logger.error(f"{Fore.RED}SMS Fail: Unexpected: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); return False

# --- Exchange Initialization ---
# (Keep refined version from v2.3.0)
def initialize_exchange() -> Optional[ccxt.Exchange]:
    logger.info(f"{Fore.BLUE}Initializing CCXT Bybit connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret: logger.critical("API Key/Secret missing."); send_sms_alert("[Pyrmethus] CRITICAL: API keys missing."); return None
    try:
        exchange = ccxt.bybit({'apiKey': CONFIG.api_key,'secret': CONFIG.api_secret,'enableRateLimit': True,'options': {'defaultType': 'linear','recvWindow': CONFIG.default_recv_window,'adjustForTimeDifference': True,'verbose': LOGGING_LEVEL <= logging.DEBUG}})
        logger.debug("Loading markets (force)..."); exchange.load_markets(True)
        logger.debug("Initial balance check..."); exchange.fetch_balance()
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session OK (LIVE MODE!).{Style.RESET_ALL}"); send_sms_alert("[Pyrmethus] Initialized & auth OK.")
        return exchange
    except ccxt.AuthenticationError as e: logger.critical(f"Auth failed: {e}. Check keys/IP/perms."); send_sms_alert(f"[Pyrmethus] CRITICAL: Auth FAILED: {e}.")
    except ccxt.NetworkError as e: logger.critical(f"Network error init: {e}. Check connection."); send_sms_alert(f"[Pyrmethus] CRITICAL: Network Error Init: {e}.")
    except ccxt.ExchangeError as e: logger.critical(f"Exchange error init: {e}. Check Bybit status."); send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error Init: {e}.")
    except Exception as e: logger.critical(f"Unexpected init error: {e}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[Pyrmethus] CRITICAL: Unexpected Init Err: {type(e).__name__}.")
    return None


# --- Indicator Calculation Functions ---
# (Keep refined/optimized versions from v2.3.0 plan)
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    col_prefix = f"{prefix}" if prefix else ""; target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    st_col = f"SUPERT_{length}_{float(multiplier)}"; st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"; st_long_col = f"SUPERTl_{length}_{float(multiplier)}"; st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    req_cols = ["high", "low", "close"]; min_len = length + 1
    for col in target_cols: df[col] = pd.NA
    if df is None or df.empty or not all(c in df.columns for c in req_cols) or len(df) < min_len: logger.warning(f"{Fore.YELLOW}Indicator ({col_prefix}ST): Input invalid/short.{Style.RESET_ALL}"); return df
    try:
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if st_col not in df.columns or st_trend_col not in df.columns: raise KeyError(f"Missing raw cols: {st_col},{st_trend_col}")
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion); df[f"{col_prefix}trend"] = (df[st_trend_col] == 1).astype('boolean')
        prev_trend = df[st_trend_col].shift(1); df[f"{col_prefix}st_long"] = ((prev_trend == -1) & (df[st_trend_col] == 1)).astype('boolean'); df[f"{col_prefix}st_short"] = ((prev_trend == 1) & (df[st_trend_col] == -1)).astype('boolean')
        raw_cols = [st_col, st_trend_col, st_long_col, st_short_col]; df.drop(columns=[c for c in raw_cols if c in df.columns], inplace=True)
    except Exception as e: logger.error(f"{Fore.RED}Indicator ({col_prefix}ST) Err: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols] # type: ignore
    return df
def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> Dict[str, Optional[Decimal]]:
    results: Dict[str, Optional[Decimal]] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}; req_cols = ["high", "low", "close", "volume"]; min_len = max(atr_len, vol_ma_len) + 1
    if df is None or df.empty or not all(c in df.columns for c in req_cols) or len(df) < min_len: logger.warning(f"{Fore.YELLOW}Indicator (Vol/ATR): Input invalid/short.{Style.RESET_ALL}"); return results
    try:
        atr_col = f"ATRr_{atr_len}"; df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns: results["atr"] = safe_decimal_conversion(df[atr_col].iloc[-1]); df.drop(columns=[atr_col], errors='ignore', inplace=True)
        vol_ma_col = f'volume_ma_{vol_ma_len}'; df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        results["volume_ma"] = safe_decimal_conversion(df[volume_ma_col].iloc[-1]); results["last_volume"] = safe_decimal_conversion(df['volume'].iloc[-1]); df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)
        if results["volume_ma"] and results["volume_ma"] > CONFIG.POSITION_QTY_EPSILON and results["last_volume"]:
            try: results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except (DivisionByZero, InvalidOperation): results["volume_ratio"] = None
        for k, v in results.items(): # Convert NaN to None
            if isinstance(v, Decimal) and v.is_nan(): results[k] = None
    except Exception as e: logger.error(f"{Fore.RED}Indicator (Vol/ATR) Err: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); results = {k: None for k in results}
    return results
def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']; min_len = max(rsi_len + stoch_len + max(k, d), mom_len) + 5
    for col in target_cols: df[col] = pd.NA
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len: logger.warning(f"{Fore.YELLOW}Indicator (StochRSI/Mom): Input invalid/short.{Style.RESET_ALL}"); return df
    try:
        st_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col, d_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if k_col in st_df.columns: df['stochrsi_k'] = st_df[k_col].apply(safe_decimal_conversion)
        if d_col in st_df.columns: df['stochrsi_d'] = st_df[d_col].apply(safe_decimal_conversion)
        mom_col = f"MOM_{mom_len}"; df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns: df['momentum'] = df[mom_col].apply(safe_decimal_conversion); df.drop(columns=[mom_col], errors='ignore', inplace=True)
    except Exception as e: logger.error(f"{Fore.RED}Indicator (StochRSI/Mom) Err: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols] # type: ignore
    return df
def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    target_cols = ['ehlers_fisher', 'ehlers_signal']; min_len = length + signal + 5
    for col in target_cols: df[col] = pd.NA
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < min_len: logger.warning(f"{Fore.YELLOW}Indicator (EhlersFisher): Input invalid/short.{Style.RESET_ALL}"); return df
    try:
        fish_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col, sig_col = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fish_col in fish_df.columns: df['ehlers_fisher'] = fish_df[fish_col].apply(safe_decimal_conversion)
        if sig_col in fish_df.columns: df['ehlers_signal'] = fish_df[sig_col].apply(safe_decimal_conversion)
    except Exception as e: logger.error(f"{Fore.RED}Indicator (EhlersFisher) Err: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols] # type: ignore
    return df
def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    target_cols = ['fast_ema', 'slow_ema']; min_len = max(fast_len, slow_len) + 5
    for col in target_cols: df[col] = pd.NA
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len: logger.warning(f"{Fore.YELLOW}Indicator (EhlersMA): Input invalid/short.{Style.RESET_ALL}"); return df
    try:
        logger.warning(f"{Fore.YELLOW}Using standard EMA placeholder for Ehlers MAs.{Style.RESET_ALL}")
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)
    except Exception as e: logger.error(f"{Fore.RED}Indicator (EhlersMA Placeholder) Err: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols] # type: ignore
    return df

# --- Order Book Analysis ---
def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> Dict[str, Optional[Decimal]]:
    results: Dict[str, Optional[Decimal]] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}; prefix="OB"
    logger.debug(f"{prefix}: Fetch L2 {symbol} (D:{depth},L:{fetch_limit})...")
    if not exchange.has.get('fetchL2OrderBook'): logger.warning(f"{Fore.YELLOW}{prefix}: Not supported.{Style.RESET_ALL}"); return results
    try:
        ob = exchange.fetch_l2_order_book(symbol, limit=fetch_limit); bids, asks = ob.get('bids', []), ob.get('asks', [])
        if not bids or not asks: logger.warning(f"{Fore.YELLOW}{prefix}: Empty bids/asks {symbol}.{Style.RESET_ALL}"); return results
        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0])>0 else DECIMAL_NaN
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0])>0 else DECIMAL_NaN
        results["best_bid"] = None if best_bid.is_nan() else best_bid; results["best_ask"] = None if best_ask.is_nan() else best_ask
        if results["best_bid"] and results["best_ask"]:
            if results["best_ask"] > results["best_bid"]: results["spread"] = results["best_ask"] - results["best_bid"]; logger.debug(f"{prefix}: Bid={results['best_bid']:.4f}, Ask={results['best_ask']:.4f}, Sprd={results['spread']:.4f}")
            else: logger.warning(f"{Fore.YELLOW}{prefix}: Bid>=Ask. Spread invalid.{Style.RESET_ALL}"); results["spread"] = None
        else: logger.debug(f"{prefix}: Bid={results['best_bid'] or 'N/A'}, Ask={results['best_ask'] or 'N/A'} (Sprd N/A)")
        bid_vol = sum(safe_decimal_conversion(b[1], DECIMAL_ZERO) for b in bids[:depth] if len(b)>1)
        ask_vol = sum(safe_decimal_conversion(a[1], DECIMAL_ZERO) for a in asks[:depth] if len(a)>1)
        logger.debug(f"{prefix}(D{depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")
        if ask_vol > CONFIG.POSITION_QTY_EPSILON:
            try: results["bid_ask_ratio"] = bid_vol / ask_vol
            except: results["bid_ask_ratio"] = None
        else: results["bid_ask_ratio"] = None
        if results["bid_ask_ratio"]: logger.debug(f"{prefix} Ratio: {results['bid_ask_ratio']:.3f}")
        else: logger.debug(f"{prefix} Ratio: N/A (AskVol={ask_vol:.4f})")
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.warning(f"{Fore.YELLOW}{prefix}: API Error {symbol}: {type(e).__name__}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{prefix}: Unexpected {symbol}: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); results = {k: None for k in results}
    return results

# --- Data Fetching ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches and prepares OHLCV data from the exchange."""
    prefix = "Data Fetch"; logger.debug(f"{prefix}: Fetching {limit} OHLCV for {symbol} ({interval})...")
    if not exchange.has.get("fetchOHLCV"): logger.error(f"{Fore.RED}{prefix}: No fetchOHLCV support.{Style.RESET_ALL}"); return None
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv: logger.warning(f"{Fore.YELLOW}{prefix}: No OHLCV data returned.{Style.RESET_ALL}"); return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty: logger.warning(f"{Fore.YELLOW}{prefix}: Empty DataFrame.{Style.RESET_ALL}"); return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True); df.set_index("timestamp", inplace=True)
        num_cols = ["open", "high", "low", "close", "volume"]
        for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        if df.isnull().values.any():
            logger.warning(f"{Fore.YELLOW}{prefix}: OHLCV has NaNs. Ffilling...{Style.RESET_ALL}"); df.ffill(inplace=True)
            if df.isnull().values.any(): logger.warning(f"{Fore.YELLOW}NaNs remain, bfilling...{Style.RESET_ALL}"); df.bfill(inplace=True)
            if df.isnull().values.any(): logger.error(f"{Fore.RED}{prefix}: NaNs persist. Cannot use.{Style.RESET_ALL}"); return None
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in num_cols): logger.error(f"{Fore.RED}{prefix}: Non-numeric data found.{Style.RESET_ALL}"); return None
        logger.debug(f"{prefix}: OK {len(df)} candles. Last: {df.index[-1]}"); return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.warning(f"{Fore.YELLOW}{prefix}: API Error: {type(e).__name__}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{prefix}: Processing Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
    return None

# --- Position & Order Management ---
# (Keep refined versions from v2.3.0 plan)
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """Fetches current position details using Bybit V5 API specifics via CCXT."""
    default_pos: Dict[str, Any] = {'side': CONFIG.POS_NONE, 'qty': DECIMAL_ZERO, 'entry_price': DECIMAL_ZERO, 'liq_price': DECIMAL_ZERO, 'unrealised_pnl': DECIMAL_ZERO}
    market: Optional[Dict] = None; market_id: Optional[str] = None; prefix = "Position Check"
    try: market = exchange.market(symbol); market_id = market['id']
    except Exception as e: logger.error(f"{Fore.RED}{prefix}: Failed get market info '{symbol}': {e}{Style.RESET_ALL}"); return default_pos
    if not market: return default_pos
    try:
        if not exchange.has.get('fetchPositions'): logger.warning(f"{Fore.YELLOW}{prefix}: No fetchPositions support.{Style.RESET_ALL}"); return default_pos
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else 'linear'); params = {'category': category}
        logger.debug(f"{prefix}: Fetching for {symbol} (ID: {market_id}, Cat: {category})...")
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        active_pos_data = None
        for pos in fetched_positions:
            info = pos.get('info', {}); pos_market_id = info.get('symbol'); idx = int(info.get('positionIdx', -1)); side_v5 = info.get('side', 'None').strip(); size_str = info.get('size')
            if pos_market_id == market_id and idx == 0 and side_v5 in ['Buy', 'Sell']:
                size = safe_decimal_conversion(size_str);
                if not size.is_nan() and abs(size) > CONFIG.POSITION_QTY_EPSILON: active_pos_data = pos; break
        if active_pos_data:
            try:
                info = active_pos_data.get('info', {}); size = safe_decimal_conversion(info.get('size')); entry = safe_decimal_conversion(info.get('avgPrice')); liq = safe_decimal_conversion(info.get('liqPrice')); pnl = safe_decimal_conversion(info.get('unrealisedPnl')); side_v5 = info.get('side')
                side = CONFIG.POS_LONG if side_v5 == 'Buy' else CONFIG.POS_SHORT
                if not size.is_nan() and not entry.is_nan() and entry >= 0:
                    pos_qty = abs(size)
                    if pos_qty > CONFIG.POSITION_QTY_EPSILON:
                        liq_price = liq if not liq.is_nan() else DECIMAL_ZERO
                        unreal_pnl = pnl if not pnl.is_nan() else DECIMAL_ZERO
                        pos_details = {'side': side, 'qty': pos_qty, 'entry_price': entry, 'liq_price': liq_price, 'unrealised_pnl': unreal_pnl}
                        logger.info(f"{Fore.YELLOW}{prefix}: Found ACTIVE {side}: Qty={pos_qty:.8f} @ Entry={entry:.4f}, Liq~{liq_price:.4f}, PnL~{unreal_pnl:.4f}{Style.RESET_ALL}")
                        return pos_details
                    else: logger.info(f"{prefix}: Pos size negligible ({pos_qty:.8f}). Flat."); return default_pos
                else: logger.warning(f"{Fore.YELLOW}{prefix}: Invalid size/entry ({size}/{entry}). Flat.{Style.RESET_ALL}"); return default_pos
            except Exception as parse_err: logger.warning(f"{Fore.YELLOW}{prefix}: Error parsing pos data: {parse_err}. Data: {active_pos_data}{Style.RESET_ALL}"); return default_pos
        else: logger.info(f"{prefix}: No active One-Way pos found for {market_id}."); return default_pos
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.warning(f"{Fore.YELLOW}{prefix}: API Error fetching: {type(e).__name__}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{prefix}: Unexpected error fetching: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
    return default_pos
def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol using Bybit V5 specifics via CCXT."""
    prefix = "Leverage Setting"; logger.info(f"{Fore.CYAN}{prefix}: Attempt {leverage}x for {symbol}...{Style.RESET_ALL}")
    try: market = exchange.market(symbol); assert market.get('contract')
    except: logger.error(f"{Fore.RED}{prefix}: Failed get market/not contract '{symbol}'.{Style.RESET_ALL}"); return False
    for attempt in range(CONFIG.RETRY_COUNT):
        try:
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            logger.debug(f"{prefix}: Call set_leverage lev={leverage}, sym={symbol}, params={params}")
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}{prefix}: Set OK {leverage}x for {symbol}. Resp: {str(response)[:100]}...{Style.RESET_ALL}"); return True
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            if "leverage not modified" in err_str or "same as requested" in err_str or "110044" in err_str: logger.info(f"{Fore.CYAN}{prefix}: Already {leverage}x.{Style.RESET_ALL}"); return True
            logger.warning(f"{Fore.YELLOW}{prefix}: Exchange error (Try {attempt+1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
        except Exception as e: logger.warning(f"{Fore.YELLOW}{prefix}: Other error (Try {attempt+1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
        if attempt < CONFIG.RETRY_COUNT - 1: time.sleep(CONFIG.RETRY_DELAY_SECONDS)
        else: logger.error(f"{Fore.RED}{prefix}: Failed after {CONFIG.RETRY_COUNT} attempts.{Style.RESET_ALL}")
    return False
def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal") -> Optional[Dict[str, Any]]:
    """Closes the specified active position by placing a market order with reduceOnly=True."""
    initial_side = position_to_close.get('side', CONFIG.POS_NONE); initial_qty = position_to_close.get('qty', DECIMAL_ZERO); market_base = get_market_base_currency(symbol); prefix = "Close Position"
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{prefix}: Initiated {symbol}. Reason: {reason}. Init State: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")
    logger.debug(f"{prefix}: Re-validating position state..."); live_position = get_current_position(exchange, symbol); live_pos_side = live_position['side']; live_qty = live_position['qty']
    entry_price = position_to_close.get('entry_price', DECIMAL_NaN) # Get entry from initial data for PnL estimate

    if live_pos_side == CONFIG.POS_NONE or live_qty <= CONFIG.POSITION_QTY_EPSILON:
        logger.warning(f"{Fore.YELLOW}{prefix}: Re-validation shows NO active position. Aborting close.{Style.RESET_ALL}")
        if initial_side != CONFIG.POS_NONE: logger.warning(f"{Fore.YELLOW}{prefix}: Discrepancy: Bot thought {initial_side}, exchange None/Zero.{Style.RESET_ALL}")
        return None
    if live_pos_side != initial_side: logger.warning(f"{Fore.YELLOW}{prefix}: Discrepancy! Initial={initial_side}, Live={live_pos_side}. Closing live.{Style.RESET_ALL}")
    close_exec_side = CONFIG.SIDE_SELL if live_pos_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY
    try:
        amount_str = format_amount(exchange, symbol, live_qty); amount_dec = safe_decimal_conversion(amount_str); amount_float = float(amount_dec)
        if amount_dec <= CONFIG.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}{prefix}: Closing amount '{amount_str}' negligible. Aborting.{Style.RESET_ALL}"); return None
        logger.warning(f"{Back.YELLOW}{Fore.BLACK}{prefix}: Attempt CLOSE {live_pos_side} ({reason}): Exec {close_exec_side.upper()} MARKET {amount_str} {symbol} (reduceOnly)...{Style.RESET_ALL}")
        params = {'reduceOnly': True}; order = exchange.create_market_order(symbol=symbol, side=close_exec_side, amount=amount_float, params=params)

        # FIX: Check if order is None before proceeding
        if order is None:
             logger.error(f"{Fore.RED}{prefix} ({reason}): create_market_order returned None! Assume failure.{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}] CRITICAL ERROR Closing ({reason}): Order placement returned None! Check Manually!")
             return None

        order_id = order.get('id'); status = order.get('status', 'unknown'); filled_qty = safe_decimal_conversion(order.get('filled')); fill_price = safe_decimal_conversion(order.get('average')); cost = safe_decimal_conversion(order.get('cost')); fee = safe_decimal_conversion(order.get('fee', {}).get('cost', '0'))
        # Estimate PnL on close
        pnl_est_str = "N/A"
        if not entry_price.is_nan() and entry_price > 0 and not fill_price.is_nan() and fill_price > 0 and not filled_qty.is_nan():
            price_diff = fill_price - entry_price
            if live_pos_side == CONFIG.POS_SHORT: price_diff = -price_diff
            pnl_val = (filled_qty * price_diff) - fee # Simple PnL estimate including fee
            pnl_color = Fore.GREEN if pnl_val >= 0 else Fore.RED
            pnl_est_str = f"{pnl_color}{pnl_val:+.4f} USDT{Style.RESET_ALL}"

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}{prefix}: Order ({reason}) PLACED/FILLED(?). ID:{format_order_id(order_id)} St:{status} Fill:{filled_qty:.8f}@{fill_price:.4f} Cost:{cost:.2f} Fee:{fee:.4f} EstPnL:{pnl_est_str}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] Closed {live_pos_side} {amount_str} @~{fill_price:.4f} ({reason}). EstPnL:{pnl_est_str.split(' ')[0]}. ID:{format_order_id(order_id)}") # Extract value for SMS
        return order
    # ... (rest of error handling as before, including check for None return from create_market_order) ...
    except ccxt.InsufficientFunds as e: logger.error(f"{Fore.RED}{prefix} ({reason}): Insufficient Funds: {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insufficient Funds.")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        if "position size is zero" in err_str or "order would not reduce position size" in err_str or "110025" in err_str or "110045" in err_str: logger.warning(f"{Fore.YELLOW}{prefix} ({reason}): Exchange reports already closed/zero: {e}. Assuming closed.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] Close ({reason}): Already closed/zero."); return None
        else: logger.error(f"{Fore.RED}{prefix} ({reason}): Exchange error: {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Exchange Error: {type(e).__name__}.")
    except Exception as e: logger.error(f"{Fore.RED}{prefix} ({reason}): Failed: {type(e).__name__} - {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}.")
    return None
def calculate_position_size(equity: Decimal,risk_per_trade_pct: Decimal,entry_price: Decimal,stop_loss_price: Decimal,leverage: int,symbol: str,exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Calculates position size (base currency) and estimated margin based on risk."""
    prefix = "Risk Calc"; logger.debug(f"{prefix}: Eq={equity:.4f}, Risk%={(risk_per_trade_pct*100):.3f}%, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")
    if not (entry_price > 0 and stop_loss_price > 0): logger.error(f"{Fore.RED}{prefix} Err: Invalid entry/SL price.{Style.RESET_ALL}"); return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}{prefix} Err: Entry/SL too close (Diff:{price_diff:.8f}).{Style.RESET_ALL}"); return None, None
    if not (0 < risk_per_trade_pct < 1): logger.error(f"{Fore.RED}{prefix} Err: Invalid risk %: {risk_per_trade_pct:.4%}.{Style.RESET_ALL}"); return None, None
    if equity <= 0: logger.error(f"{Fore.RED}{prefix} Err: Invalid equity: {equity:.4f}.{Style.RESET_ALL}"); return None, None
    if leverage <= 0: logger.error(f"{Fore.RED}{prefix} Err: Invalid leverage: {leverage}.{Style.RESET_ALL}"); return None, None
    try:
        risk_amt = equity * risk_per_trade_pct; qty_raw = risk_amt / price_diff
        qty_prec_str = format_amount(exchange, symbol, qty_raw); qty_prec = safe_decimal_conversion(qty_prec_str)
        if qty_prec.is_nan() or qty_prec <= CONFIG.POSITION_QTY_EPSILON: logger.warning(f"{Fore.YELLOW}{prefix} Warn: Qty negligible/zero after prec ({qty_prec:.8f}).{Style.RESET_ALL}"); return None, None
        pos_val = qty_prec * entry_price; req_margin = pos_val / Decimal(leverage)
        logger.debug(f"{prefix} Result: Qty={qty_prec:.8f}, RiskAmt={risk_amt:.4f}, EstVal={pos_val:.4f}, EstMargin={req_margin:.4f}")
        return qty_prec, req_margin
    except Exception as e: logger.error(f"{Fore.RED}{prefix} Err: Calc failed: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); return None, None
def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
    """Waits for a specific order to reach 'closed' (filled) status."""
    start = time.monotonic(); id_short = format_order_id(order_id); prefix = f"Wait Fill ({id_short})"; logger.info(f"{Fore.CYAN}{prefix}: Waiting {symbol} (Timeout:{timeout_seconds}s)...{Style.RESET_ALL}")
    while time.monotonic() - start < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol); status = order.get('status')
            logger.debug(f"{prefix} Status: {status}")
            if status == 'closed': logger.success(f"{Fore.GREEN}{prefix}: Confirmed FILLED.{Style.RESET_ALL}"); return order
            elif status in ['canceled', 'rejected', 'expired']: logger.error(f"{Fore.RED}{prefix}: Failed status '{status}'.{Style.RESET_ALL}"); return None
            time.sleep(CONFIG.MARKET_ORDER_FILL_CHECK_INTERVAL)
        except ccxt.OrderNotFound: elapsed = time.monotonic() - start; logger.warning(f"{Fore.YELLOW}{prefix}: Not found yet ({elapsed:.1f}s). Retrying...{Style.RESET_ALL}"); time.sleep(0.5)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e: elapsed = time.monotonic() - start; logger.warning(f"{Fore.YELLOW}{prefix}: API Error ({elapsed:.1f}s): {type(e).__name__}. Retrying...{Style.RESET_ALL}"); time.sleep(CONFIG.RETRY_DELAY_SECONDS)
        except Exception as e: elapsed = time.monotonic() - start; logger.error(f"{Fore.RED}{prefix}: Unexpected error ({elapsed:.1f}s): {e}. Stop wait.{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); return None
    logger.error(f"{Fore.RED}{prefix}: Timeout ({timeout_seconds}s).{Style.RESET_ALL}")
    try: final_check = exchange.fetch_order(order_id, symbol); logger.warning(f"{prefix}: Final status on timeout: {final_check.get('status')}")
    except Exception as final_e: logger.warning(f"{prefix}: Final status check failed: {final_e}")
    return None

# --- Order Placement Refactored Helpers ---
# (Keep refined versions from v2.3.0 plan)
def _calculate_initial_sl_price(entry_price: Decimal, atr: Decimal, multiplier: Decimal, side: str, min_price: Decimal, symbol: str, exchange: ccxt.Exchange) -> Optional[Decimal]:
    prefix = "Place Order (Calc SL)"; sl_dist = atr * multiplier
    raw_sl = (entry_price - sl_dist) if side == CONFIG.SIDE_BUY else (entry_price + sl_dist)
    if min_price > 0 and raw_sl < min_price: logger.warning(f"{Fore.YELLOW}{prefix}: Raw SL {raw_sl:.4f}<Min {min_price:.4f}. Adjusting.{Style.RESET_ALL}"); raw_sl = min_price
    elif raw_sl <= 0: logger.error(f"{Fore.RED}{prefix}: Calc SL <= 0 ({raw_sl:.4f}). Cannot proceed.{Style.RESET_ALL}"); return None
    sl_str = format_price(exchange, symbol, raw_sl); sl_price = safe_decimal_conversion(sl_str)
    if sl_price.is_nan(): logger.error(f"{Fore.RED}{prefix}: Failed format/convert SL price {sl_str}.{Style.RESET_ALL}"); return None
    logger.info(f"{prefix}: Initial SL Price ~ {sl_price:.4f} (ATR Dist: {sl_dist:.4f})"); return sl_price
def _validate_and_cap_quantity(calc_qty: Decimal, entry_price: Decimal, leverage: int, max_cap_usdt: Decimal, min_qty_limit: Decimal, max_qty_limit: Optional[Decimal], symbol: str, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    prefix = "Place Order (Validate Qty)"; final_qty = calc_qty; req_margin_est = (final_qty * entry_price) / Decimal(leverage)
    pos_val_est = final_qty * entry_price
    if pos_val_est > max_cap_usdt:
        logger.warning(f"{Fore.YELLOW}{prefix}: Est. val {pos_val_est:.2f}>{max_cap_usdt:.2f} Cap. Capping.{Style.RESET_ALL}")
        try: capped_raw = max_cap_usdt / entry_price
        except DivisionByZero: logger.error(f"{prefix}: Cannot cap, zero entry price."); return None, None
        capped_str = format_amount(exchange, symbol, capped_raw); final_qty = safe_decimal_conversion(capped_str)
        if final_qty.is_nan(): logger.error(f"{prefix}: Failed convert capped qty '{capped_str}'."); return None, None
        req_margin_est = (final_qty * entry_price) / Decimal(leverage); logger.info(f"{prefix}: Qty capped {final_qty:.8f}. New EstMargin ~{req_margin_est:.4f}")
    if final_qty <= CONFIG.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}{prefix}: Final qty ({final_qty:.8f}) negligible/zero. Abort.{Style.RESET_ALL}"); return None, None
    if min_qty_limit > 0 and final_qty < min_qty_limit: logger.error(f"{Fore.RED}{prefix}: Final qty {final_qty:.8f} < Min {min_qty_limit:.8f}. Abort.{Style.RESET_ALL}"); return None, None
    if max_qty_limit is not None and final_qty > max_qty_limit:
        logger.warning(f"{Fore.YELLOW}{prefix}: Final qty {final_qty:.8f} > Max {max_qty_limit:.8f}. Capping.{Style.RESET_ALL}")
        final_qty = max_qty_limit; final_qty = safe_decimal_conversion(format_amount(exchange, symbol, final_qty))
        if final_qty.is_nan(): logger.error(f"{prefix}: Failed convert max qty."); return None, None
        req_margin_est = (final_qty * entry_price) / Decimal(leverage)
    return final_qty, req_margin_est
def _check_margin_availability(req_margin_est: Decimal, free_balance: Decimal, buffer: Decimal, side: str, market_base: str) -> bool:
    prefix = "Place Order (Margin Check)"; buffered_req = req_margin_est * buffer
    if free_balance < buffered_req:
        logger.error(f"{Fore.RED}{prefix}: Insufficient FREE margin. Need ~{buffered_req:.4f}, Have {free_balance:.4f}.{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insuff Free Margin (Need ~{buffered_req:.2f})")
        return False
    logger.info(f"{Fore.GREEN}{prefix}: OK. EstMargin={req_margin_est:.4f}, BufferedReq={buffered_req:.4f}, Free={free_balance:.4f}{Style.RESET_ALL}"); return True
def _place_and_confirm_entry_order(exchange: ccxt.Exchange, symbol: str, side: str, quantity: Decimal) -> Optional[Dict[str, Any]]:
    prefix = f"Place Order ({side.upper()} Entry)"; entry_order_id: Optional[str] = None
    try:
        qty_float = float(quantity); logger.warning(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** {prefix}: Placing MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
        entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params={'reduce_only': False})
        # Check if exchange might return None on placement failure
        if entry_order is None: raise ValueError("create_market_order returned None, placement likely failed.")
        entry_order_id = entry_order.get('id');
        if not entry_order_id: raise ValueError("Entry order placement returned no ID.")
        logger.success(f"{Fore.GREEN}{prefix}: Order submitted ID:{format_order_id(entry_order_id)}. Waiting fill...{Style.RESET_ALL}")
        filled_entry = wait_for_order_fill(exchange, entry_order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry:
            logger.error(f"{Fore.RED}{prefix}: Order {format_order_id(entry_order_id)} did NOT fill/fail.{Style.RESET_ALL}")
            try: logger.warning(f"{prefix}: Attempting cancel {format_order_id(entry_order_id)}..."); exchange.cancel_order(entry_order_id, symbol)
            except Exception as cancel_err: logger.warning(f"{prefix}: Cancel failed {format_order_id(entry_order_id)}: {cancel_err}")
            return None
        return filled_entry
    except Exception as e:
        logger.error(f"{Fore.RED}{Style.BRIGHT}{prefix}: FAILED PLACE/CONFIRM ENTRY: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        if entry_order_id:
            try: logger.warning(f"{prefix}: Attempting cancel after failure {format_order_id(entry_order_id)}..."); exchange.cancel_order(entry_order_id, symbol)
            except Exception as cancel_err: logger.warning(f"{prefix}: Cancel failed: {cancel_err}")
        return None
def _place_stop_order(exchange: ccxt.Exchange, symbol: str, pos_side: str, qty: Decimal, stop_price_str: str, trigger_by: str, is_tsl: bool = False, tsl_params: Optional[Dict] = None) -> Tuple[Optional[str], str]:
    order_type = "Trailing SL" if is_tsl else "Fixed SL"; prefix = f"Place Order ({order_type})"; sl_exec_side = CONFIG.SIDE_SELL if pos_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY
    order_id: Optional[str] = None; status_msg: str = "Placement FAILED"; market_base = get_market_base_currency(symbol)
    try:
        qty_str = format_amount(exchange, symbol, qty); qty_float = float(qty_str); stop_price_float = float(stop_price_str)
        params = {'reduceOnly': True, 'triggerBy': trigger_by}
        if is_tsl and tsl_params:
            params['trailingStop'] = tsl_params['trailingStop']; params['activePrice'] = stop_price_float # TSL activation price
            logger.info(f"{Fore.CYAN}{prefix} ({tsl_params['trail_percent_str']:.2%})... Side:{sl_exec_side}, Qty:{qty_float:.8f}, Trail%:{params['trailingStop']}, ActPx:{stop_price_str}{Style.RESET_ALL}")
        else:
            params['stopPrice'] = stop_price_float # Fixed SL trigger price
            logger.info(f"{Fore.CYAN}{prefix}... Side:{sl_exec_side}, Qty:{qty_float:.8f}, StopPx:{stop_price_str}{Style.RESET_ALL}")

        # Use 'stopMarket' type for Bybit V5 via CCXT
        stop_order = exchange.create_order(symbol, 'stopMarket', sl_exec_side, qty_float, params=params)
        if stop_order is None: raise ValueError(f"{order_type} create_order returned None.") # Check None return
        order_id = stop_order.get('id');
        if not order_id: raise ValueError(f"{order_type} order placement returned no ID.")
        order_id_short = format_order_id(order_id)
        if is_tsl: status_msg = f"Placed (ID:{order_id_short}, Trail:{params['trailingStop']}%, ActPx:{stop_price_str})"
        else: status_msg = f"Placed (ID:{order_id_short}, Trigger:{stop_price_str})"
        logger.success(f"{Fore.GREEN}{prefix}: {status_msg}{Style.RESET_ALL}")
        return order_id, status_msg
    except Exception as e:
        status_msg = f"Placement FAILED: {type(e).__name__} - {e}"
        logger.error(f"{Fore.RED}{Style.BRIGHT}{prefix}: {status_msg}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] CRITICAL ERROR placing {order_type}: {type(e).__name__}. Pos may be unprotected!")
        return None, status_msg

# --- Main Order Placement Orchestration ---
def place_risked_market_order(exchange: ccxt.Exchange, symbol: str, side: str, risk_percentage: Decimal, current_atr: Decimal, sl_atr_multiplier: Decimal, leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal, tsl_percent: Decimal, tsl_activation_offset_percent: Decimal) -> Optional[Dict[str, Any]]:
    """Orchestrates placing a market entry order with risk management, SL, and TSL."""
    market_base = get_market_base_currency(symbol); prefix = f"Place Order ({side.upper()})"; logger.info(f"{Fore.BLUE}{Style.BRIGHT}{prefix}: Full process start for {symbol}...{Style.RESET_ALL}")
    if current_atr.is_nan() or current_atr <= 0: logger.error(f"{prefix}: Invalid ATR. Cannot start."); return None
    filled_entry_order: Optional[Dict[str, Any]] = None
    try:
        balance = exchange.fetch_balance(); market = exchange.market(symbol)
        limits = market.get('limits', {}); min_qty = safe_decimal_conversion(limits.get('amount', {}).get('min'), DECIMAL_ZERO); max_qty = safe_decimal_conversion(limits.get('amount', {}).get('max')) if limits.get('amount',{}).get('max') else None; min_price = safe_decimal_conversion(limits.get('price', {}).get('min'), DECIMAL_ZERO)
        usdt_balance = balance.get(CONFIG.USDT_SYMBOL, {}); usdt_total = safe_decimal_conversion(usdt_balance.get('total')); usdt_free = safe_decimal_conversion(usdt_balance.get('free')); usdt_equity = usdt_total if usdt_total > 0 else usdt_free
        if usdt_equity <= 0 or usdt_free < 0: logger.error(f"{prefix}: Invalid balance Eq={usdt_equity}, Free={usdt_free}."); return None

        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        entry_price_est = ob_data.get("best_ask") if side == CONFIG.SIDE_BUY else ob_data.get("best_bid")
        if not entry_price_est: entry_price_est = safe_decimal_conversion(exchange.fetch_ticker(symbol).get('last'))
        if entry_price_est is None or entry_price_est.is_nan() or entry_price_est <= 0: logger.error(f"{prefix}: Failed get valid entry price estimate."); return None
        logger.info(f"{prefix}: Estimated Entry Price ~ {entry_price_est:.4f}")

        initial_sl_price = _calculate_initial_sl_price(entry_price_est, current_atr, sl_atr_multiplier, side, min_price, symbol, exchange)
        if not initial_sl_price: return None

        calc_qty, req_margin_est = calculate_position_size(usdt_equity, risk_percentage, entry_price_est, initial_sl_price, leverage, symbol, exchange)
        if calc_qty is None or req_margin_est is None: return None

        final_qty, final_margin_est = _validate_and_cap_quantity(calc_qty, entry_price_est, leverage, max_order_cap_usdt, min_qty, max_qty, symbol, exchange)
        if final_qty is None or final_margin_est is None: return None

        if not _check_margin_availability(final_margin_est, usdt_free, margin_check_buffer, side, market_base): return None

        # --- Execute Entry ---
        filled_entry_order = _place_and_confirm_entry_order(exchange, symbol, side, final_qty)
        if not filled_entry_order:
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement/confirmation failed.")
            # Double check position just in case
            time.sleep(1); current_pos = get_current_position(exchange, symbol)
            if current_pos['side'] != CONFIG.POS_NONE: logger.error(f"{Back.RED}{Fore.WHITE}{prefix} POS OPENED despite entry failure! Qty:{current_pos['qty']}. Closing!{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CRITICAL: Pos opened on FAILED entry! Closing NOW."); close_position(exchange, symbol, current_pos, reason="Emergency Close - Failed Entry")
            return None

        # --- Post-Entry SL/TSL Placement ---
        avg_fill_price = safe_decimal_conversion(filled_entry_order.get('average')); filled_qty = safe_decimal_conversion(filled_entry_order.get('filled'))
        pos_side = CONFIG.POS_LONG if side == CONFIG.SIDE_BUY else CONFIG.POS_SHORT
        sl_distance = current_atr * sl_atr_multiplier
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.SIDE_BUY else (avg_fill_price + sl_distance)
        if min_price > 0 and actual_sl_price_raw < min_price: actual_sl_price_raw = min_price
        elif actual_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}{Style.BRIGHT}{prefix}: Invalid ACTUAL SL price ({actual_sl_price_raw:.4f}) post-fill!{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL ({actual_sl_price_raw:.4f})! Emergency Closing.")
            if CONFIG.emergency_close_on_sl_fail: close_position(exchange, symbol, {'side': pos_side, 'qty': filled_qty}, reason="Invalid SL Calc Post-Entry")
            else: logger.error(f"{prefix}: Emergency close disabled, MANUAL INTERVENTION NEEDED!")
            # Return filled order but signal SL failed state
            return filled_entry_order # Indicate entry happened, but SL failed

        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)

        # Place Fixed SL
        sl_order_id, sl_status_msg = _place_stop_order(exchange, symbol, pos_side, filled_qty, actual_sl_price_str, CONFIG.sl_trigger_by)
        if not sl_order_id:
            send_sms_alert(f"[{market_base}] CRITICAL ERROR ({side.upper()}): Failed initial SL placement! Pos UNPROTECTED by fixed SL.")
            if CONFIG.emergency_close_on_sl_fail:
                logger.error(f"{Back.RED}{Fore.WHITE}{prefix} Initial SL placement failed! Emergency Closing position...{Style.RESET_ALL}")
                close_position(exchange, symbol, {'side': pos_side, 'qty': filled_qty}, reason="Emergency Close - SL Placement Failed")
                # Even if close fails, return the entry order but state is critical
                return filled_entry_order
            else:
                logger.error(f"{Back.RED}{Fore.WHITE}{prefix} Initial SL placement failed! Emergency close disabled. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
                # Proceed to TSL attempt, but position is high risk

        # Place TSL
        tsl_order_id, tsl_status_msg = "N/A", "Not Configured"
        tsl_trail_val_str = "OFF"
        if tsl_percent > CONFIG.POSITION_QTY_EPSILON:
            try:
                act_offset = avg_fill_price * tsl_activation_offset_percent
                act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.SIDE_BUY else (avg_fill_price - act_offset)
                if min_price > 0 and act_price_raw < min_price: act_price_raw = min_price
                if act_price_raw <= 0: raise ValueError(f"Invalid TSL act price {act_price_raw:.4f}")
                tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
                tsl_trail_val_str = str((tsl_percent * 100).quantize(Decimal("0.01")))
                tsl_params_dict = {'trailingStop': tsl_trail_val_str, 'trail_percent_str': tsl_percent}
                tsl_order_id, tsl_status_msg = _place_stop_order(exchange, symbol, pos_side, filled_qty, tsl_act_price_str, CONFIG.tsl_trigger_by, is_tsl=True, tsl_params=tsl_params_dict)
                if not tsl_order_id: send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement! Relying on fixed SL if placed.")
            except Exception as tsl_err:
                logger.error(f"{Fore.RED}{Style.BRIGHT}{prefix}: FAILED calc/place TSL: {tsl_err}{Style.RESET_ALL}"); tsl_status_msg = f"FAILED: {tsl_err}"
                send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL setup ({type(tsl_err).__name__}).")

        # Final Log & SMS
        logger.info(f"{Back.BLUE}{Fore.WHITE}{prefix} --- ORDER PLACEMENT SUMMARY ---{Style.RESET_ALL}")
        logger.info(f"  Entry: {format_order_id(filled_entry_order.get('id'))} | Filled: {filled_qty:.8f} @ {avg_fill_price:.4f}")
        logger.info(f"  Fixed SL: {sl_status_msg}")
        logger.info(f"  Trailing SL: {tsl_status_msg}")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{'--- END SUMMARY ---':^40}{Style.RESET_ALL}")
        sms_summary = (f"[{market_base}] {side.upper()} {filled_qty:.6f}@{avg_fill_price:.3f}. "
                       f"SL:{('~'+actual_sl_price_str if sl_order_id else 'FAIL')}. "
                       f"TSL:{('%'+tsl_trail_val_str if tsl_order_id else ('FAIL' if tsl_percent > 0 else 'OFF'))}. "
                       f"EID:{format_order_id(filled_entry_order.get('id'))}")
        send_sms_alert(sms_summary)
        return filled_entry_order # Return successful entry order

    except Exception as e: # Catch errors during setup/checks before placing entry
        logger.error(f"{Fore.RED}{Style.BRIGHT}{prefix}: Overall process FAILED: {type(e).__name__} - {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Overall process failed: {type(e).__name__}")
        return None


# --- cancel_open_orders --- (Keep refined version from v2.3.0)
def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    prefix = "Order Cancel"; logger.info(f"{Fore.CYAN}{prefix}: Attempting {symbol} ({reason})...{Style.RESET_ALL}")
    cancelled_count, failed_count = 0, 0; market_base = get_market_base_currency(symbol)
    try:
        if not exchange.has.get('fetchOpenOrders'): logger.warning(f"{Fore.YELLOW}{prefix}: fetchOpenOrders not supported.{Style.RESET_ALL}"); return
        logger.debug(f"{prefix}: Fetching open orders..."); open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders: logger.info(f"{Fore.CYAN}{prefix}: No open orders found.{Style.RESET_ALL}"); return
        logger.warning(f"{Fore.YELLOW}{prefix}: Found {len(open_orders)} open orders. Cancelling...{Style.RESET_ALL}")
        for order in open_orders:
            oid = order.get('id'); info = f"ID:{format_order_id(oid)} ({order.get('type','?')} {order.get('side','?')})"
            if oid:
                try: logger.debug(f"{prefix}: Cancelling {info}"); exchange.cancel_order(oid, symbol); logger.info(f"{Fore.CYAN}{prefix}: OK {info}{Style.RESET_ALL}"); cancelled_count += 1; time.sleep(0.1)
                except ccxt.OrderNotFound: logger.warning(f"{Fore.YELLOW}{prefix}: Not found (closed?): {info}{Style.RESET_ALL}"); cancelled_count += 1
                except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.error(f"{Fore.RED}{prefix}: FAILED {info}: {e}{Style.RESET_ALL}"); failed_count += 1
                except Exception as e: logger.error(f"{Fore.RED}{prefix}: Unexpected Err cancel {info}: {e}{Style.RESET_ALL}"); failed_count += 1
            else: logger.warning(f"{prefix}: Found order without ID: {order}")
        log_level = logging.INFO if failed_count == 0 else logging.WARNING
        logger.log(log_level, f"{Fore.CYAN}{prefix}: Finished. Cancelled/NF: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
        if failed_count > 0: send_sms_alert(f"[{market_base}] WARNING: Failed cancel {failed_count} orders during {reason}.")
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.error(f"{Fore.RED}{prefix}: Failed fetch open orders: {e}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{prefix}: Unexpected error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())

# --- Strategy Signal Generation ---
# (Keep refined version from v2.3.0)
def generate_signals(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit"}; prefix = f"Signal Gen ({strategy_name})"
    if df is None or len(df) < 2: logger.debug(f"{prefix}: Insufficient data length."); return signals
    last, prev = df.iloc[-1], df.iloc[-2]
    try:
        if strategy_name == "DUAL_SUPERTREND":
            st_l, st_s, cf_t = last.get('st_long'), last.get('st_short'), last.get('confirm_trend')
            if pd.notna(st_l) and pd.notna(st_s) and pd.notna(cf_t): # Ensure indicators are valid
                if st_l and cf_t: signals['enter_long'] = True
                if st_s and not cf_t: signals['enter_short'] = True
                if st_s: signals['exit_long'] = True; signals['exit_reason'] = "Primary ST Short Flip"
                if st_l: signals['exit_short'] = True; signals['exit_reason'] = "Primary ST Long Flip"
            else: logger.debug(f"{prefix}: Skipping: missing/invalid Supertrend columns.")
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k, d, m = last.get('stochrsi_k'), last.get('stochrsi_d'), last.get('momentum'); kp, dp = prev.get('stochrsi_k'), prev.get('stochrsi_d')
            if any(v is None or v.is_nan() for v in [k, d, m, kp, dp]): logger.debug(f"{prefix}: Skipping: missing StochRSI/Mom values.")
            else:
                if kp <= dp and k > d and k < CONFIG.stochrsi_oversold and m > CONFIG.POSITION_QTY_EPSILON: signals['enter_long'] = True
                if kp >= dp and k < d and k > CONFIG.stochrsi_overbought and m < -CONFIG.POSITION_QTY_EPSILON: signals['enter_short'] = True
                if kp >= dp and k < d: signals['exit_long'] = True; signals['exit_reason'] = "StochRSI K < D"
                if kp <= dp and k > d: signals['exit_short'] = True; signals['exit_reason'] = "StochRSI K > D"
        elif strategy_name == "EHLERS_FISHER":
            f, s = last.get('ehlers_fisher'), last.get('ehlers_signal'); fp, sp = prev.get('ehlers_fisher'), prev.get('ehlers_signal')
            if any(v is None or v.is_nan() for v in [f, s, fp, sp]): logger.debug(f"{prefix}: Skipping: missing Ehlers Fisher values.")
            else:
                if fp <= sp and f > s: signals['enter_long'] = True
                if fp >= sp and f < s: signals['enter_short'] = True
                if fp >= sp and f < s: signals['exit_long'] = True; signals['exit_reason'] = "Ehlers Fisher < Signal"
                if fp <= sp and f > s: signals['exit_short'] = True; signals['exit_reason'] = "Ehlers Fisher > Signal"
        elif strategy_name == "EHLERS_MA_CROSS":
            fm, sm = last.get('fast_ema'), last.get('slow_ema'); fmp, smp = prev.get('fast_ema'), prev.get('slow_ema')
            if any(v is None or v.is_nan() for v in [fm, sm, fmp, smp]): logger.debug(f"{prefix}: Skipping: missing Ehlers MA (EMA) values.")
            else:
                if fmp <= smp and fm > sm: signals['enter_long'] = True
                if fmp >= smp and fm < sm: signals['enter_short'] = True
                if fmp >= smp and fm < sm: signals['exit_long'] = True; signals['exit_reason'] = "Fast MA < Slow MA"
                if fmp <= smp and fm > sm: signals['exit_short'] = True; signals['exit_reason'] = "Fast MA > Slow MA"
        else: logger.warning(f"{prefix}: Unknown strategy '{strategy_name}'.")
    except KeyError as e: logger.error(f"{Fore.RED}{prefix} Error: Missing indicator col: {e}.{Style.RESET_ALL}"); signals = {k: (v if k == 'exit_reason' else False) for k, v in signals.items()}
    except Exception as e: logger.error(f"{Fore.RED}{prefix} Error: Unexpected: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); signals = {k: (v if k == 'exit_reason' else False) for k, v in signals.items()}
    return signals

# --- Trading Logic ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame, cycle_count: int) -> None:
    """Executes the main trading logic for one cycle."""
    global current_bot_state # Access global state
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    market_base = get_market_base_currency(symbol); prefix = f"[Cycle {cycle_count} | {market_base}]"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== {prefix} Check Start ({CONFIG.strategy_name}) | State: {current_bot_state} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # Prevent actions if in intermediate state from previous cycle
    if current_bot_state in [BotState.ENTERING, BotState.CLOSING]:
        logger.warning(f"{Fore.YELLOW}{prefix} Still in {current_bot_state} state. Waiting for state resolution. Skipping logic.{Style.RESET_ALL}")
        # Optionally, add a check here to see if the state has been stuck for too long
        # If stuck, could transition to ERROR state or attempt recovery.
        return

    try:
        current_bot_state = BotState.IDLE # Reset state at beginning of cycle (if not error)

        # === Data Sufficiency Check ===
        required_rows = max(150, CONFIG.atr_calculation_period*2, CONFIG.volume_ma_period*2, # Common lookbacks
                            # Strategy specific lookbacks (simplified check)
                           (CONFIG.st_atr_length + CONFIG.confirm_st_atr_length)*2 if CONFIG.strategy_name == "DUAL_SUPERTREND" else 0,
                           (CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 10) if CONFIG.strategy_name == "STOCHRSI_MOMENTUM" else 0,
                           (CONFIG.ehlers_fisher_length*2) if CONFIG.strategy_name == "EHLERS_FISHER" else 0,
                           (CONFIG.ehlers_slow_period*2) if CONFIG.strategy_name == "EHLERS_MA_CROSS" else 0
                           ) + CONFIG.API_FETCH_LIMIT_BUFFER
        if df is None or len(df) < required_rows: logger.warning(f"{Fore.YELLOW}{prefix} Insufficient data ({len(df) if df else 0}, need ~{required_rows}). Skipping.{Style.RESET_ALL}"); return

        # === Step 1: Calculate Indicators ===
        logger.debug(f"{prefix} Calculating indicators..."); vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period); current_atr = vol_atr_data.get("atr")
        # OPTIMIZED: Calculate only needed strategy indicators
        if CONFIG.strategy_name == "DUAL_SUPERTREND": df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier); df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        elif CONFIG.strategy_name == "EHLERS_FISHER": df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)

        # === Step 2: Validate Base Requirements ===
        last_candle = df.iloc[-1]; current_price = safe_decimal_conversion(last_candle.get('close'))
        if current_price.is_nan() or current_price <= 0: logger.warning(f"{Fore.YELLOW}{prefix} Last candle close invalid ({current_price}). Skipping.{Style.RESET_ALL}"); return
        can_place_new_order = current_atr is not None
        if not can_place_new_order: logger.warning(f"{Fore.YELLOW}{prefix} Invalid ATR ({current_atr}). New orders disabled.{Style.RESET_ALL}")

        # === Step 3: Get Position & Optional OB Data ===
        position = get_current_position(exchange, symbol)
        position_side = position['side']; position_qty = position['qty']; position_entry_price = position['entry_price']
        ob_data: Optional[Dict[str, Optional[Decimal]]] = None
        if CONFIG.fetch_order_book_per_cycle: ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)

        # === Step 4: Log State ===
        # (Keep state logging from v2.3.0, including PnL estimate)
        atr_str = f"{current_atr:.5f}" if current_atr else 'N/A'; logger.info(f"{prefix} State | Price:{current_price:.4f} ATR:{atr_str}")
        vol_ratio = vol_atr_data.get("volume_ratio"); vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold; vol_ratio_str = f"{vol_ratio:.2f}" if vol_ratio is not None else 'N/A'
        logger.info(f"{prefix} State | Vol Ratio:{vol_ratio_str}, Spike:{vol_spike}(Req:{CONFIG.require_volume_spike_for_entry})")
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None; spread = ob_data.get("spread") if ob_data else None; ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else 'N/A'; spread_str = f"{spread:.4f}" if spread is not None else 'N/A'
        logger.info(f"{prefix} State | OB Ratio:{ob_ratio_str}, Spread:{spread_str}(Fetched:{ob_data is not None})")
        pnl_str = "N/A"
        if position_side != CONFIG.POS_NONE and not position_entry_price.is_nan() and position_entry_price > 0:
            price_diff = current_price - position_entry_price; pnl_mult = DECIMAL_ONE if position_side == CONFIG.POS_LONG else -DECIMAL_ONE
            est_pnl = position_qty * price_diff * pnl_mult; pnl_color = Fore.GREEN if est_pnl >= 0 else Fore.RED
            pnl_pct_str = f"({(est_pnl / (position_qty * position_entry_price / CONFIG.leverage) * 100):+.2f}% ROE)" if position_qty > 0 and position_entry_price > 0 and CONFIG.leverage > 0 else "" # ROE estimate
            pnl_str = f"{pnl_color}{est_pnl:+.4f} USDT{Style.RESET_ALL} {pnl_pct_str}"
        logger.info(f"{prefix} State | Pos: Side={position_side}, Qty={position_qty:.8f}, Entry={position_entry_price:.4f}, Liq~{position.get('liq_price', DECIMAL_ZERO):.4f}, Est.PnL: {pnl_str}")

        # Update bot state based on position
        if position_side != CONFIG.POS_NONE: current_bot_state = BotState.IN_POSITION
        else: current_bot_state = BotState.IDLE

        # === Step 5: Generate Signals ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name) # Pass df directly
        logger.debug(f"{prefix} Signals ({CONFIG.strategy_name}): {strategy_signals}")

        # === Step 6: Handle Exits (If In Position) ===
        if current_bot_state == BotState.IN_POSITION:
            should_exit = (position_side == CONFIG.POS_LONG and strategy_signals['exit_long']) or (position_side == CONFIG.POS_SHORT and strategy_signals['exit_short'])
            if should_exit:
                exit_reason = strategy_signals['exit_reason']
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}{prefix} *** STRATEGY EXIT: Closing {position_side} ({exit_reason}) ***{Style.RESET_ALL}")
                current_bot_state = BotState.CLOSING # Set state before action
                cancel_open_orders(exchange, symbol, f"Pre-{exit_reason} Exit"); time.sleep(0.5)
                close_result = close_position(exchange, symbol, position, reason=exit_reason)
                if close_result: logger.info(f"{prefix} Pausing {CONFIG.POST_CLOSE_DELAY_SECONDS}s post-close..."); time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS)
                else: logger.error(f"{Fore.RED}{prefix} Failed execute close for {position_side} exit! Manual check!{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CRITICAL: Failed CLOSE {position_side} on signal! Check!")
                current_bot_state = BotState.IDLE # Reset state after action attempt
                return
            else: logger.info(f"{prefix} Holding {position_side}. No strategy exit. Waiting SL/TSL."); return

        # === Step 7: Handle Entries (If IDLE) ===
        if current_bot_state == BotState.IDLE:
            if not can_place_new_order: logger.warning(f"{Fore.YELLOW}{prefix} Holding Cash. Cannot enter: Invalid ATR.{Style.RESET_ALL}"); return
            logger.debug(f"{prefix} Holding Cash. Checking entry signals...")
            enter_long = strategy_signals['enter_long']; enter_short = strategy_signals['enter_short']; potential_entry = enter_long or enter_short
            if not potential_entry: logger.info(f"{prefix} Holding Cash. No entry signal."); return

            # --- Confirmations ---
            passes_vol = not CONFIG.require_volume_spike_for_entry or (vol_atr_data.get("volume_ratio") is not None and vol_atr_data["volume_ratio"] > CONFIG.volume_spike_threshold)
            # Fetch OB only once if needed
            if not CONFIG.fetch_order_book_per_cycle and ob_data is None: logger.debug(f"{prefix} Fetching OB for entry confirm..."); ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None; ob_available = bid_ask_ratio is not None
            passes_long_ob = not ob_available or (bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
            passes_short_ob = not ob_available or (bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)

            # --- Combine ---
            final_enter_long = enter_long and passes_vol and passes_long_ob
            final_enter_short = enter_short and passes_vol and passes_short_ob

            # --- Execute ---
            entry_side: Optional[str] = None
            if final_enter_long: entry_side = CONFIG.SIDE_BUY; logger.success(f"{Back.GREEN}{Fore.BLACK}{prefix} *** CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) ***{Style.RESET_ALL}")
            elif final_enter_short: entry_side = CONFIG.SIDE_SELL; logger.success(f"{Back.RED}{Fore.WHITE}{prefix} *** CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) ***{Style.RESET_ALL}")

            if entry_side:
                current_bot_state = BotState.ENTERING # Set state
                cancel_open_orders(exchange, symbol, f"Pre-{entry_side.upper()} Entry"); time.sleep(0.5)
                place_result = place_risked_market_order(exchange, symbol, entry_side, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier, CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer, CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
                if place_result: logger.info(f"{prefix} Entry order process initiated."); # Don't reset state here, next cycle will check position
                else: logger.error(f"{prefix} Entry order process failed."); current_bot_state = BotState.IDLE # Reset if placement failed immediately
                return # End cycle after entry attempt
            else: logger.info(f"{prefix} Holding Cash. Entry signal failed confirmation checks.")

    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{prefix} CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!")
        current_bot_state = BotState.ERROR # Set error state
    finally:
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== {prefix} Cycle Check End | State: {current_bot_state} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown ---
# (Keep refined version from v2.3.0)
def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str]) -> None:
    """Attempts to gracefully shut down the bot."""
    global current_bot_state
    current_bot_state = BotState.CLOSING # Indicate shutdown state
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit...{Style.RESET_ALL}")
    market_base = get_market_base_currency(symbol) if symbol else "Bot"; send_sms_alert(f"[{market_base}] Shutdown initiated. Cleanup...")
    if not exchange or not symbol: logger.warning(f"{Fore.YELLOW}Shutdown: Exchange/Symbol missing. Cannot cleanup.{Style.RESET_ALL}"); return
    try:
        logger.info(f"{Fore.CYAN}Shutdown: Cancelling open orders for {symbol}...{Style.RESET_ALL}"); cancel_open_orders(exchange, symbol, reason="Graceful Shutdown"); time.sleep(1)
        logger.info(f"{Fore.CYAN}Shutdown: Checking active position for {symbol}...{Style.RESET_ALL}"); position = get_current_position(exchange, symbol)
        if position['side'] != CONFIG.POS_NONE and position['qty'] > CONFIG.POSITION_QTY_EPSILON:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} pos Qty: {position['qty']:.8f}. Closing...{Style.RESET_ALL}"); close_result = close_position(exchange, symbol, position, reason="Shutdown Request")
            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.POST_CLOSE_DELAY_SECONDS * 2}s confirm...{Style.RESET_ALL}"); time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS * 2)
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.POS_NONE or final_pos['qty'] <= CONFIG.POSITION_QTY_EPSILON: logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else: logger.error(f"{Back.RED}{Fore.WHITE}Shutdown CRITICAL: FAILED CONFIRM closure. Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CRITICAL ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. CHECK MANUALLY!")
            else: logger.error(f"{Back.RED}{Fore.WHITE}Shutdown CRITICAL: Failed PLACE close order. MANUAL INTERVENTION!{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CRITICAL ERROR: Failed PLACE close order! MANUAL CHECK!")
        else: logger.info(f"{Fore.GREEN}Shutdown: No active position found.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] No active position found on shutdown.")
    except Exception as e: logger.error(f"{Fore.RED}Shutdown Error: Unexpected during cleanup: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] Error during shutdown cleanup: {type(e).__name__}")
    finally: logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Shutdown Sequence Complete ---{Style.RESET_ALL}")

# --- Main Execution ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    global cycle_count, current_bot_state
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}{'--- Pyrmethus Bybit Scalping Spell v2.4.0 Initializing ---':^80}{Style.RESET_ALL}")
    logger.info(f"{'Timestamp:':<20} {start_time_str}")
    logger.info(f"{'Selected Strategy:':<20} {Fore.CYAN}{CONFIG.strategy_name}{Style.RESET_ALL}")
    logger.info(f"{'Risk Management:':<20} {Fore.GREEN}Initial ATR SL + Native TSL{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}{'--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---':^80}{Style.RESET_ALL}")

    exchange: Optional[ccxt.Exchange] = None
    symbol: Optional[str] = None
    market_base: str = "Bot"
    run_bot: bool = True

    try:
        # === Initialize ===
        exchange = initialize_exchange();
        if not exchange: return

        # === Setup Symbol/Leverage ===
        try:
            symbol_to_use = CONFIG.symbol; logger.info(f"Attempting symbol: {symbol_to_use}")
            market = exchange.market(symbol_to_use); symbol = market['symbol']; market_base = get_market_base_currency(symbol)
            if not market.get('contract'): raise ValueError("Not contract market")
            logger.info(f"{Fore.GREEN}Using Symbol: {symbol} (Type:{market.get('type','N/A')}, ID:{market.get('id')}){Style.RESET_ALL}")
            if not set_leverage(exchange, symbol, CONFIG.leverage): raise RuntimeError("Leverage setup failed")
        except Exception as e: logger.critical(f"Symbol/Leverage setup failed '{CONFIG.symbol}': {e}"); send_sms_alert(f"[{market_base or 'Bot'}] CRITICAL: Setup FAILED ({type(e).__name__}). Exit."); return

        # === Log Config Summary ===
        # (Keep logging from v2.3.0, corrected percentage format)
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}{'--- Configuration Summary ---':^80}{Style.RESET_ALL}")
        logger.info(f"  {'Symbol:':<25} {symbol}"); logger.info(f"  {'Interval:':<25} {CONFIG.interval}"); logger.info(f"  {'Leverage:':<25} {CONFIG.leverage}x")
        logger.info(f"  {'Strategy:':<25} {CONFIG.strategy_name}")
        if CONFIG.strategy_name == "DUAL_SUPERTREND": logger.info(f"    Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": logger.info(f"    Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER": logger.info(f"    Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": logger.info(f"    Params: FastMA={CONFIG.ehlers_fast_period}, SlowMA={CONFIG.ehlers_slow_period} (EMA Placeholder)")
        logger.info(f"{Fore.GREEN}  {'Risk % / Trade:':<25} {(CONFIG.risk_per_trade_percentage * 100):.3f}%") # Corrected format
        logger.info(f"  {'Max Pos Value Cap:':<25} {CONFIG.max_order_usdt_amount:.2f} USDT")
        logger.info(f"  {'Initial SL:':<25} {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period}) | Trigger: {CONFIG.sl_trigger_by}")
        logger.info(f"  {'Trailing SL:':<25} {(CONFIG.trailing_stop_percentage * 100):.2f}% | Act Offset: {(CONFIG.trailing_stop_activation_offset_percent * 100):.2f}% | Trigger: {CONFIG.tsl_trigger_by}") # Corrected format
        logger.info(f"  {'SL Fail Action:':<25} {'Emergency Close' if CONFIG.emergency_close_on_sl_fail else 'Alert Only'}")
        logger.info(f"{Fore.YELLOW}  {'Vol Confirm:':<25} {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"  {'OB Confirm:':<25} {CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        logger.info(f"{Fore.WHITE}  {'Loop Sleep:':<25} {CONFIG.sleep_seconds}s")
        logger.info(f"  {'Margin Buffer:':<25} {((CONFIG.required_margin_buffer - DECIMAL_ONE) * DECIMAL_HUNDRED):.1f}%") # Corrected format
        logger.info(f"  {'Fill Timeout:':<25} {CONFIG.order_fill_timeout_seconds}s")
        logger.info(f"{Fore.MAGENTA}  {'SMS Alerts:':<25} {CONFIG.enable_sms_alerts} (To: {'*****' + CONFIG.sms_recipient_number[-4:] if CONFIG.sms_recipient_number else 'N/A'})")
        logger.info(f"{Fore.CYAN}  {'Logging Level:':<25} {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}{'---------------------------------------------------':^80}{Style.RESET_ALL}")

        send_sms_alert(f"[{market_base}] Pyrmethus v2.4 Started. Strat:{CONFIG.strategy_name}|{symbol}|{CONFIG.interval}|{CONFIG.leverage}x. Live!")

        # === Main Trading Loop ===
        logger.info(f"{Fore.GREEN}{Style.BRIGHT}--- Starting Main Trading Loop ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}")
        while run_bot:
            cycle_start = time.monotonic(); cycle_count += 1; log_prefix = f"[Cycle {cycle_count} | {market_base}]"
            logger.debug(f"{Fore.CYAN}--- {log_prefix} Start | State: {current_bot_state} ---{Style.RESET_ALL}")

            # Check if bot is in error state - requires manual intervention or restart
            if current_bot_state == BotState.ERROR:
                logger.critical(f"{Back.RED}{Fore.WHITE}{log_prefix} Bot is in ERROR state. Requires intervention. Halting.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL: Bot in ERROR state. Halting.")
                run_bot = False
                continue # Skip to finally block

            try:
                # --- Data Fetch ---
                # (Keep dynamic data_limit calculation from v2.3.0)
                data_limit = max(150, CONFIG.atr_calculation_period*2, CONFIG.volume_ma_period*2,
                                (CONFIG.st_atr_length*2 + CONFIG.confirm_st_atr_length*2) if CONFIG.strategy_name == "DUAL_SUPERTREND" else 0,
                                (CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 10) if CONFIG.strategy_name == "STOCHRSI_MOMENTUM" else 0,
                                (CONFIG.ehlers_fisher_length*2) if CONFIG.strategy_name == "EHLERS_FISHER" else 0,
                                (CONFIG.ehlers_slow_period*2) if CONFIG.strategy_name == "EHLERS_MA_CROSS" else 0
                                ) + CONFIG.API_FETCH_LIMIT_BUFFER
                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                if df is not None and not df.empty:
                    trade_logic(exchange, symbol, df, cycle_count) # Pass cycle count
                else: logger.warning(f"{Fore.YELLOW}{log_prefix} No valid market data. Skipping logic.{Style.RESET_ALL}")

            # --- Robust Error Handling (Keep from v2.3.0) ---
            except ccxt.RateLimitExceeded as e: logger.warning(f"{Back.YELLOW}{Fore.BLACK}{log_prefix} Rate Limit: {e}. Sleeping {CONFIG.sleep_seconds * 5}s...{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] WARN: Rate limit hit!"); time.sleep(CONFIG.sleep_seconds * 5)
            except ccxt.NetworkError as e: logger.warning(f"{Fore.YELLOW}{log_prefix} Network error: {e}. Retrying.{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds)
            except ccxt.ExchangeNotAvailable as e: logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix} Exchange unavailable: {e}. Sleeping {CONFIG.sleep_seconds * 10}s...{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable!"); time.sleep(CONFIG.sleep_seconds * 10)
            except ccxt.AuthenticationError as e: logger.critical(f"{Back.RED}{Fore.WHITE}{log_prefix} Auth Error: {e}. Stopping!{Style.RESET_ALL}"); run_bot = False; send_sms_alert(f"[{market_base}] CRITICAL: Auth Error! Stopping.")
            except ccxt.ExchangeError as e: logger.error(f"{Fore.RED}{log_prefix} Unhandled Exchange error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] ERROR: Unhandled Exch Err: {type(e).__name__}."); time.sleep(CONFIG.sleep_seconds)
            except Exception as e: logger.exception(f"{Back.RED}{Fore.WHITE}{log_prefix} !!! UNEXPECTED CRITICAL ERROR: {e} !!!{Style.RESET_ALL}"); run_bot = False; current_bot_state = BotState.ERROR; send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping.")

            # --- Loop Delay ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start; sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"{log_prefix} Cycle time: {elapsed:.2f}s. Sleeping: {sleep_dur:.2f}s.")
                if sleep_dur > 0: time.sleep(sleep_dur)

    except KeyboardInterrupt: logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt. Shutting down...{Style.RESET_ALL}"); run_bot = False
    except Exception as e: logger.critical(f"{Back.RED}{Fore.WHITE}Critical setup error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base or 'Bot'}] CRITICAL SETUP ERR: {type(e).__name__}!"); run_bot = False
    finally:
        graceful_shutdown(exchange, symbol)
        final_alert_market = market_base if market_base != "Bot" else "Bot"; send_sms_alert(f"[{final_alert_market}] Pyrmethus v2.4 terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}{'--- Pyrmethus Scalping Spell Deactivated ---':^80}{Style.RESET_ALL}")

if __name__ == "__main__":
    cycle_count = 0 # Initialize cycle count
    try: main()
    except Exception as e: logger.critical(f"{Back.RED}{Fore.WHITE}Unhandled top-level exception: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); sys.exit(1)
    sys.exit(0)

# --- END OF FILE pyrmethus_scalper_v2.4.0.py ---
```

**Summary of Key Changes in v2.4.0:**

1.  **Bug Fix (Traceback 1):** Corrected `_get_env` to properly cast *default* values if they are used, fixing the `str` vs `int` subtraction error during config logging.
2.  **Bug Fix (Traceback 2):** Added an explicit `if order is None:` check after `create_market_order` in `close_position` to handle potential `None` returns gracefully.
3.  **Refactored Order Placement:** Maintained the helper function structure (`_calculate_initial_sl_price`, `_validate_and_cap_quantity`, etc.) for `place_risked_market_order`.
4.  **Optimized Indicators:** `trade_logic` now only calculates indicators relevant to the `CONFIG.strategy_name`, improving efficiency.
5.  **Improved SL/TSL Handling:**
    *   Added `EMERGENCY_CLOSE_ON_SL_FAIL` config option (default `False`).
    *   If enabled and *initial fixed SL placement* fails, the bot will attempt an emergency close.
    *   Alerts are made more critical if protection placement fails.
6.  **Config Enhancements:** Added `SL_TRIGGER_BY` and `TSL_TRIGGER_BY` config options with validation. Added more `min_val`/`max_val` checks for numeric parameters.
7.  **Robustness:** Strengthened checks for `None` and `NaN` throughout indicator calculations and logic. Explicitly handle `DivisionByZero`. Consistent use of `ROUND_DOWN` via `format_amount`.
8.  **State Management:** Introduced `BotState` enum and `current_bot_state` global variable to track `IDLE`, `ENTERING`, `IN_POSITION`, `CLOSING`, `ERROR` states, preventing conflicting actions within `trade_logic`.
9.  **Logging:** Added estimated PnL% calculation and logging. Added PnL estimate to close confirmation logs. Cycle count added to main logic prefix.
10. **Clarity:** Used `CONFIG.SIDE_BUY`, `CONFIG.POS_LONG`, etc., constants. Expanded type hints. Improved docstrings.
11. **Version:** Updated to `v2.4.0`.

This version addresses the reported errors and significantly improves robustness, clarity, and efficiency. Remember to **test thoroughly on testnet** before deploying live.
