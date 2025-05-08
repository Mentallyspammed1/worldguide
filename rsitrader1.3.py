Okay, let's refactor `rsitrader1.2.py` with a focus on structure, enhanced use of dictionaries (and classes/dataclasses which are often superior for structured data in Python), and overall improvements.

**Key Changes:**

1.  **Configuration (`Config` Dataclass):** Replaced global config variables with a dedicated `Config` dataclass. Loading and validation now populate this object.
2.  **Market Details (`MarketDetails` Dataclass):** Stores symbol-specific information like precision, limits, currencies, etc.
3.  **Exchange Interaction (`ExchangeHandler` Class):** Encapsulates all `ccxt` interactions (fetching data, placing/cancelling orders, managing connection, market details). This centralizes API calls and error handling related to the exchange. The retry decorator is applied to its methods.
4.  **Position Management (`PositionManager` Class):** Manages the bot's trading state (active position, SL/TP, order IDs, TSL data). Handles loading/saving state to JSON and provides methods to update/reset the state. This significantly cleans up the main loop.
5.  **Trading Logic (`TradingBot` Class):** Contains the main `run` loop and orchestrates the interaction between `Config`, `ExchangeHandler`, and `PositionManager`. Signal generation and trade execution logic are methods within this class.
6.  **Improved Structure:** Reduced reliance on global variables. Code is more object-oriented and modular.
7.  **Enhanced Type Hinting:** Added more comprehensive type hints using `typing` and introduced `dataclasses`.
8.  **Clearer Logging:** Log messages often indicate the component (e.g., `[ExchangeHandler]`).
9.  **Robustness:** Incorporated fixes like the parenthesis issue in `fetch_positions` handling and refined error management.
10. **Abstraction:** The `ExchangeHandler` attempts to abstract OCO vs. separate order placement, though exchange-specific details *within* the handler might still be needed for live OCO.

```python
# --- START OF REFACTORED FILE rsitrader_enhanced.py ---

import ccxt
import os
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import pandas_ta as ta
import json
import sys
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union, Callable, Final, Type
import functools
from dataclasses import dataclass, field
import traceback # For more detailed error logging

# --- Load Environment Variables FIRST ---
# Explicitly load .env from the script's directory or CWD
try:
    script_dir = Path(__file__).resolve().parent
    dotenv_path = script_dir / '.env'
    print(f"Attempting to load environment variables from: {dotenv_path}")
except NameError:
    script_dir = Path.cwd()
    dotenv_path = script_dir / '.env'
    print(f"Warning: Could not determine script directory reliably. Looking for .env in current working directory: {dotenv_path}")

load_success = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=False)

if not load_success:
    if dotenv_path.exists():
        print(f"Warning: Found .env file at {dotenv_path}, but load_dotenv() reported failure. Check file permissions or content formatting.")
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Environment variables must be set externally.")
else:
    print(f"Successfully processed .env file check at: {dotenv_path}")

# --- Colorama Initialization ---
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    # Define Neon Colors using Bright style
    NEON_GREEN: Final[str] = Fore.GREEN + Style.BRIGHT
    NEON_PINK: Final[str] = Fore.MAGENTA + Style.BRIGHT
    NEON_CYAN: Final[str] = Fore.CYAN + Style.BRIGHT
    NEON_RED: Final[str] = Fore.RED + Style.BRIGHT
    NEON_YELLOW: Final[str] = Fore.YELLOW + Style.BRIGHT
    NEON_BLUE: Final[str] = Fore.BLUE + Style.BRIGHT
    RESET: Final[str] = Style.RESET_ALL
    COLORAMA_AVAILABLE: Final[bool] = True
except ImportError:
    print("Warning: colorama library not found. Neon styling disabled. Consider installing it: `pip install colorama`")
    NEON_GREEN = NEON_PINK = NEON_CYAN = NEON_RED = NEON_YELLOW = NEON_BLUE = RESET = ""
    COLORAMA_AVAILABLE = False

# --- Logging Configuration ---
log_format_base: str = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] [%(funcName)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format_base, datefmt='%Y-%m-%d %H:%M:%S')
logger: logging.Logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_PRICE_PRECISION: Final[int] = 4
DEFAULT_AMOUNT_PRECISION: Final[int] = 8
POSITION_STATE_FILE: Final[str] = 'position_state.json'
CONFIG_FILE: Final[str] = 'config.json'
SIMULATION_ORDER_PREFIX: Final[str] = "sim_"
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded,
    ccxt.RequestTimeout, ccxt.DDoSProtection,
)

# --- Neon Display Functions (Enhanced clarity and consistency) ---
# (Keep existing display functions: print_neon_header, display_error_box, display_warning_box, etc.)
def print_neon_header() -> None:
    """Prints a visually appealing header for the bot."""
    box_width = 70
    print(f"{NEON_CYAN}{'=' * box_width}{RESET}")
    print(f"{NEON_PINK}{Style.BRIGHT}{'Enhanced RSI/OB Trader Neon Bot - v1.3 (Refactored)':^{box_width}}{RESET}")
    print(f"{NEON_CYAN}{'=' * box_width}{RESET}")

def display_error_box(message: str) -> None:
    box_width = 70
    print(f"{NEON_RED}{'!' * box_width}{RESET}")
    print(f"{NEON_RED}! {message.strip():^{box_width-4}} !{RESET}")
    print(f"{NEON_RED}{'!' * box_width}{RESET}")

def display_warning_box(message: str) -> None:
    box_width = 70
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")
    print(f"{NEON_YELLOW}~ {message.strip():^{box_width-4}} ~{RESET}")
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")

# Wrapper functions for logging with color
def log_info(msg: str) -> None: logger.info(f"{NEON_GREEN}{msg}{RESET}")
def log_error(msg: str, exc_info: bool = False) -> None:
    first_line = msg.split('\n', 1)[0]
    display_error_box(first_line)
    logger.error(f"{NEON_RED}{msg}{RESET}", exc_info=exc_info)
def log_warning(msg: str) -> None:
    display_warning_box(msg)
    logger.warning(f"{NEON_YELLOW}{msg}{RESET}")
def log_debug(msg: str) -> None: logger.debug(f"{Fore.WHITE}{msg}{RESET}")

def print_cycle_divider(timestamp: pd.Timestamp) -> None:
    box_width = 70
    print(f"\n{NEON_BLUE}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}Cycle Start: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}{RESET}")
    print(f"{NEON_BLUE}{'=' * box_width}{RESET}")

# (Keep neon_sleep_timer and print_shutdown_message)
def neon_sleep_timer(seconds: int) -> None:
    if not COLORAMA_AVAILABLE or seconds <= 0:
        if seconds > 0: print(f"Sleeping for {seconds} seconds...")
        time.sleep(max(0, seconds))
        return
    interval: float = 0.5
    steps: int = int(seconds / interval)
    for i in range(steps, -1, -1):
        remaining_seconds = max(0, round(i * interval))
        if remaining_seconds <= 5 and i % 2 == 0: color = NEON_RED
        elif remaining_seconds <= 15: color = NEON_YELLOW
        else: color = NEON_CYAN
        print(f"{color}Next cycle in: {remaining_seconds} seconds... {RESET}", end='\r')
        time.sleep(interval)
    print(" " * 50, end='\r')

def print_shutdown_message() -> None:
    box_width = 70
    print(f"\n{NEON_PINK}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}{Style.BRIGHT}{'RSI/OB Trader Bot Stopped - Goodbye!':^{box_width}}{RESET}")
    print(f"{NEON_PINK}{'=' * box_width}{RESET}")

# --- Retry Decorator ---
def retry_api_call(max_retries: int = 3, initial_delay: float = 5.0, backoff_factor: float = 2.0) -> Callable:
    """Decorator for retrying CCXT API calls on specific network/rate limit errors."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries, delay = 0, initial_delay
            last_exception = None
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    retries += 1
                    last_exception = e
                    if retries > max_retries:
                        log_error(f"API call '{func.__name__}' failed after {max_retries} retries. Last error: {type(e).__name__}: {e}", exc_info=False)
                        raise last_exception
                    log_warning(f"API call '{func.__name__}' failed due to {type(e).__name__}. Retrying in {delay:.1f}s ({retries}/{max_retries})...")
                    neon_sleep_timer(int(round(delay)))
                    delay *= backoff_factor
                except Exception as e:
                    log_error(f"Non-retryable error in API call '{func.__name__}': {type(e).__name__}: {e}", exc_info=True)
                    raise
            if last_exception: raise last_exception
            raise RuntimeError(f"API call '{func.__name__}' failed unexpectedly after exhausting retries without raising.")
        return wrapper
    return decorator

# --- Data Structures ---

@dataclass
class Config:
    # Core settings
    exchange_id: str
    symbol: str
    timeframe: str
    risk_percentage: float
    simulation_mode: bool
    data_limit: int
    sleep_interval_seconds: int

    # Indicator settings
    rsi_length: int
    rsi_overbought: int
    rsi_oversold: int
    stoch_k: int
    stoch_d: int
    stoch_smooth_k: int
    stoch_overbought: int
    stoch_oversold: int

    # Order Block settings
    ob_volume_threshold_multiplier: float
    ob_lookback: int

    # Volume confirmation settings
    entry_volume_confirmation_enabled: bool
    entry_volume_ma_length: int = 20  # Default
    entry_volume_multiplier: float = 1.2 # Default

    # SL/TP and Trailing Stop settings
    enable_atr_sl_tp: bool
    enable_trailing_stop: bool
    atr_length: int = 14 # Default, used if needs_atr is True

    # Conditional SL/TP values
    atr_sl_multiplier: float = 0.0 # Set if enable_atr_sl_tp is True
    atr_tp_multiplier: float = 0.0 # Set if enable_atr_sl_tp is True
    stop_loss_percentage: float = 0.0 # Set if enable_atr_sl_tp is False
    take_profit_percentage: float = 0.0 # Set if enable_atr_sl_tp is False

    # Conditional TSL values
    trailing_stop_atr_multiplier: float = 0.0 # Set if enable_trailing_stop is True
    trailing_stop_activation_atr_multiplier: float = 0.0 # Set if enable_trailing_stop is True

    # Retry settings
    retry_max_retries: int = 3
    retry_initial_delay: float = 5.0
    retry_backoff_factor: float = 2.0

    # --- Derived properties ---
    @property
    def needs_atr(self) -> bool:
        return self.enable_atr_sl_tp or self.enable_trailing_stop

    @property
    def rsi_col_name(self) -> str: return f'RSI_{self.rsi_length}'

    @property
    def stoch_k_col_name(self) -> str: return f'STOCHk_{self.stoch_k}_{self.stoch_d}_{self.stoch_smooth_k}'

    @property
    def stoch_d_col_name(self) -> str: return f'STOCHd_{self.stoch_k}_{self.stoch_d}_{self.stoch_smooth_k}'

    @property
    def atr_col_name(self) -> Optional[str]: return f'ATRr_{self.atr_length}' if self.needs_atr else None

    @property
    def vol_ma_col_name(self) -> Optional[str]: return f'VOL_MA_{self.entry_volume_ma_length}' if self.entry_volume_confirmation_enabled else None

    @property
    def stoch_params(self) -> Dict[str, int]:
        return {'k': self.stoch_k, 'd': self.stoch_d, 'smooth_k': self.stoch_smooth_k}


@dataclass
class MarketDetails:
    symbol: str
    price_precision_digits: int = DEFAULT_PRICE_PRECISION
    amount_precision_digits: int = DEFAULT_AMOUNT_PRECISION
    min_tick: float = 1 / (10 ** DEFAULT_PRICE_PRECISION)
    min_amount: Optional[float] = None
    min_cost: Optional[float] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    is_contract_market: bool = False
    # Exchange capabilities relevant to the symbol/market type
    supports_oco: bool = False
    supports_fetch_positions: bool = False
    supports_stop_market: bool = False
    supports_stop_limit: bool = False
    supports_reduce_only: bool = False # Tracks if 'reduceOnly' param is likely needed/supported


# --- Configuration Loading ---
def load_and_validate_config(filename: str = CONFIG_FILE) -> Config:
    """Loads, validates, and returns the configuration as a Config object."""
    log_info(f"Loading configuration from '{filename}'...")
    config_path = Path(filename)
    if not config_path.exists():
        log_error(f"CRITICAL: Configuration file '{filename}' not found. Please create it.")
        sys.exit(1)

    try:
        with config_path.open('r') as f:
            cfg_data: Dict[str, Any] = json.load(f)
        log_info(f"Configuration file '{filename}' loaded successfully. Validating...")

        # --- Validation Logic (similar to before, but adapted for dataclass) ---
        errors: List[str] = []
        def validate(key: str, types: Union[type, Tuple[type, ...]], condition: Optional[Callable[[Any], bool]] = None, required: bool = True):
            if key not in cfg_data:
                if required: errors.append(f"Missing required key: '{key}'")
                return
            val = cfg_data[key]
            if not isinstance(val, types): errors.append(f"Key '{key}': Expected type {types}, but got {type(val).__name__}.")
            elif condition and not condition(val): errors.append(f"Key '{key}' (value: {val}) failed validation condition.")

        # Perform validations using the validate helper...
        # Core settings
        validate("exchange_id", str, lambda x: len(x) > 0)
        validate("symbol", str, lambda x: len(x) > 0)
        validate("timeframe", str, lambda x: len(x) > 0)
        validate("risk_percentage", (float, int), lambda x: 0 < x < 1)
        validate("simulation_mode", bool)
        validate("data_limit", int, lambda x: x > 50)
        validate("sleep_interval_seconds", int, lambda x: x > 0)

        # Indicators
        validate("rsi_length", int, lambda x: x > 0)
        validate("rsi_overbought", int, lambda x: 50 < x <= 100)
        validate("rsi_oversold", int, lambda x: 0 <= x < 50)
        if "rsi_oversold" in cfg_data and "rsi_overbought" in cfg_data and cfg_data.get("rsi_oversold", 0) >= cfg_data.get("rsi_overbought", 100):
             errors.append("rsi_oversold must be strictly less than rsi_overbought")
        validate("stoch_k", int, lambda x: x > 0)
        validate("stoch_d", int, lambda x: x > 0)
        validate("stoch_smooth_k", int, lambda x: x > 0)
        validate("stoch_overbought", int, lambda x: 50 < x <= 100)
        validate("stoch_oversold", int, lambda x: 0 <= x < 50)
        if "stoch_oversold" in cfg_data and "stoch_overbought" in cfg_data and cfg_data.get("stoch_oversold", 0) >= cfg_data.get("stoch_overbought", 100):
             errors.append("stoch_oversold must be strictly less than stoch_overbought")

        # Order Block
        validate("ob_volume_threshold_multiplier", (float, int), lambda x: x > 0)
        validate("ob_lookback", int, lambda x: x > 0)

        # Volume Confirmation
        validate("entry_volume_confirmation_enabled", bool)
        if cfg_data.get("entry_volume_confirmation_enabled"):
            validate("entry_volume_ma_length", int, lambda x: x > 0, required=False) # Use default if missing
            validate("entry_volume_multiplier", (float, int), lambda x: x > 0, required=False) # Use default

        # SL/TP & TSL (conditional validation)
        validate("enable_atr_sl_tp", bool)
        validate("enable_trailing_stop", bool)
        needs_atr_cfg = cfg_data.get("enable_atr_sl_tp", False) or cfg_data.get("enable_trailing_stop", False)
        if needs_atr_cfg: validate("atr_length", int, lambda x: x > 0, required=False) # Use default

        if cfg_data.get("enable_atr_sl_tp"):
            validate("atr_sl_multiplier", (float, int), lambda x: x > 0)
            validate("atr_tp_multiplier", (float, int), lambda x: x > 0)
        else:
            validate("stop_loss_percentage", (float, int), lambda x: x > 0)
            validate("take_profit_percentage", (float, int), lambda x: x > 0)

        if cfg_data.get("enable_trailing_stop"):
             validate("trailing_stop_atr_multiplier", (float, int), lambda x: x > 0)
             validate("trailing_stop_activation_atr_multiplier", (float, int), lambda x: x >= 0)

        # Optional Retry settings
        validate("retry_max_retries", int, lambda x: x >= 0, required=False)
        validate("retry_initial_delay", (float, int), lambda x: x > 0, required=False)
        validate("retry_backoff_factor", (float, int), lambda x: x >= 1, required=False)

        # --- Report Validation Results ---
        if errors:
            error_str = "\n - ".join(errors)
            log_error(f"CRITICAL: Configuration validation failed with {len(errors)} errors:\n - {error_str}")
            sys.exit(1)

        log_info("Configuration validation passed.")

        # --- Create Config Object ---
        # Handle defaults for conditional fields carefully
        config = Config(**cfg_data) # Unpack validated data into the dataclass

        # Post-process conditional defaults if keys were missing but validation passed (due to required=False)
        if config.entry_volume_confirmation_enabled:
            config.entry_volume_ma_length = cfg_data.get("entry_volume_ma_length", Config.entry_volume_ma_length)
            config.entry_volume_multiplier = cfg_data.get("entry_volume_multiplier", Config.entry_volume_multiplier)
        if config.needs_atr:
             config.atr_length = cfg_data.get("atr_length", Config.atr_length)

        # Set conditional SL/TP values based on enable_atr_sl_tp
        if config.enable_atr_sl_tp:
             config.stop_loss_percentage = 0.0
             config.take_profit_percentage = 0.0
        else:
             config.atr_sl_multiplier = 0.0
             config.atr_tp_multiplier = 0.0

        # Set conditional TSL values based on enable_trailing_stop
        if config.enable_trailing_stop:
            pass # Values loaded directly
        else:
             config.trailing_stop_atr_multiplier = 0.0
             config.trailing_stop_activation_atr_multiplier = 0.0

        # Ensure float types where needed
        config.risk_percentage = float(config.risk_percentage)
        if not config.enable_atr_sl_tp:
            config.stop_loss_percentage = float(config.stop_loss_percentage)
            config.take_profit_percentage = float(config.take_profit_percentage)
        if config.enable_atr_sl_tp:
            config.atr_sl_multiplier = float(config.atr_sl_multiplier)
            config.atr_tp_multiplier = float(config.atr_tp_multiplier)
        if config.enable_trailing_stop:
            config.trailing_stop_atr_multiplier = float(config.trailing_stop_atr_multiplier)
            config.trailing_stop_activation_atr_multiplier = float(config.trailing_stop_activation_atr_multiplier)
        config.ob_volume_threshold_multiplier = float(config.ob_volume_threshold_multiplier)
        if config.entry_volume_confirmation_enabled:
             config.entry_volume_multiplier = float(config.entry_volume_multiplier)


        return config

    except json.JSONDecodeError as e:
        log_error(f"CRITICAL: Error decoding JSON from '{filename}'. Check syntax: {e}")
        sys.exit(1)
    except TypeError as e:
         log_error(f"CRITICAL: Mismatch between config file fields and expected Config structure: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        log_error(f"CRITICAL: Unexpected error loading or validating config '{filename}': {e}", exc_info=True)
        sys.exit(1)


# --- Exchange Handler Class ---
class ExchangeHandler:
    """Handles all interactions with the CCXT exchange."""

    def __init__(self, config: Config):
        self.config = config
        self._exchange: Optional[ccxt.Exchange] = None
        self._market_details: Dict[str, MarketDetails] = {}
        self._api_retry_decorator = retry_api_call(
            max_retries=config.retry_max_retries,
            initial_delay=config.retry_initial_delay,
            backoff_factor=config.retry_backoff_factor
        )
        self._connect()
        self._load_market_details(self.config.symbol)

    def _connect(self):
        """Establishes connection to the exchange."""
        log_info(f"Attempting to connect to exchange: {self.config.exchange_id}")
        exchange_id = self.config.exchange_id.lower()
        api_key_env = f"{exchange_id.upper()}_API_KEY"
        secret_key_env = f"{exchange_id.upper()}_SECRET_KEY"
        passphrase_env = f"{exchange_id.upper()}_PASSPHRASE"

        api_key = os.getenv(api_key_env)
        secret = os.getenv(secret_key_env)
        passphrase = os.getenv(passphrase_env)

        if not api_key or not secret:
            log_error(f"CRITICAL: API Key ('{api_key_env}') or Secret ('{secret_key_env}') not found.")
            sys.exit(1)
        if passphrase:
            log_info(f"Passphrase found ('{passphrase_env}').")

        try:
            exchange_class: Type[ccxt.Exchange] = getattr(ccxt, exchange_id)
            symbol_upper = self.config.symbol.upper()
            is_perp = ':' in symbol_upper or 'PERP' in symbol_upper or 'SWAP' in symbol_upper or '-P' in symbol_upper
            market_type_guess = 'swap' if is_perp else 'spot'
            log_info(f"Guessed market type from symbol '{self.config.symbol}': {market_type_guess} (for `defaultType`)")

            exchange_config: Dict[str, Any] = {
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': market_type_guess,
                    'adjustForTimeDifference': True,
                }
            }
            if passphrase:
                exchange_config['password'] = passphrase

            self._exchange = exchange_class(exchange_config)
            self._load_markets_robustly(force_reload=True) # Initial load

            log_info(f"Successfully connected to {self.exchange.name} ({exchange_id}).")
            log_info(f"Using Market Type: {self.exchange.options.get('defaultType')}")
            log_info(f"Exchange supports OCO: {self.exchange.has.get('oco', False)}")
            log_info(f"Exchange supports fetchPositions: {self.exchange.has.get('fetchPositions', False)}")
            log_info(f"Exchange supports createStopMarketOrder: {self.exchange.has.get('createStopMarketOrder', False)}")
            log_info(f"Exchange supports createStopLimitOrder: {self.exchange.has.get('createStopLimitOrder', False)}")

        except AttributeError:
            log_error(f"CRITICAL: Exchange ID '{exchange_id}' is not supported by ccxt.")
            sys.exit(1)
        except ccxt.AuthenticationError as e:
            log_error(f"CRITICAL: Authentication failed for {exchange_id}. Check credentials. Error: {e}")
            sys.exit(1)
        except ccxt.ExchangeError as e: # Catch other ccxt errors during init/load
            log_error(f"CRITICAL: Exchange error during connection/market loading: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            log_error(f"CRITICAL: Unexpected error during exchange setup: {e}", exc_info=True)
            sys.exit(1)

    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            # This should not happen if _connect was successful, but safety check
            log_error("Exchange object is None. Attempting to reconnect...")
            self._connect()
            if self._exchange is None: # Still None after reconnect attempt
                 log_error("CRITICAL: Reconnection failed. Exiting.")
                 sys.exit(1)
        return self._exchange

    @property
    def name(self) -> str:
        return self.exchange.name if self.exchange else "N/A"

    def get_market_details(self, symbol: str) -> Optional[MarketDetails]:
        """Gets cached market details or loads them if needed."""
        details = self._market_details.get(symbol.upper())
        if not details:
            log_warning(f"Market details for {symbol} not cached. Attempting to load.")
            if self._load_market_details(symbol):
                details = self._market_details.get(symbol.upper())
            else:
                log_error(f"Failed to load market details for {symbol}. Cannot proceed with this symbol.")
                return None
        return details

    def _load_markets_robustly(self, force_reload: bool = False):
        """Internal method to load markets with retry logic."""
        @self._api_retry_decorator
        def _load():
            log_info(f"[ExchangeHandler] Loading markets (force_reload={force_reload})...")
            self.exchange.load_markets(reload=force_reload)
            log_info(f"[ExchangeHandler] Markets loaded successfully. Found {len(self.exchange.markets)} markets.")
        _load()

    def _load_market_details(self, symbol: str) -> bool:
        """Loads and caches precision, limits, and capabilities for a symbol."""
        sym_upper = symbol.upper()
        try:
            # Ensure markets are loaded
            if not self.exchange.markets:
                self._load_markets_robustly(force_reload=True)
            if sym_upper not in self.exchange.markets:
                available = list(self.exchange.markets.keys())
                hint = f"Available examples: {available[:10]}" if available else "No markets loaded?"
                log_error(f"[ExchangeHandler] CRITICAL: Symbol '{sym_upper}' not found on {self.name}. {hint}")
                return False
            market_info = self.exchange.markets[sym_upper]

            if self.config.timeframe not in self.exchange.timeframes:
                available_tfs = list(self.exchange.timeframes.keys())
                hint = f"Available: {available_tfs}" if available_tfs else "No timeframes listed?"
                log_error(f"[ExchangeHandler] CRITICAL: Timeframe '{self.config.timeframe}' not supported by {self.name}. {hint}")
                return False

            log_info(f"[ExchangeHandler] Loading market details for {sym_upper}...")

            details = MarketDetails(symbol=sym_upper) # Start with defaults

            # Precision
            price_prec = market_info.get('precision', {}).get('price')
            amount_prec = market_info.get('precision', {}).get('amount')
            if price_prec is not None:
                details.price_precision_digits = int(self.exchange.decimal_to_precision(price_prec, ccxt.ROUND, counting_mode=self.exchange.precisionMode))
                details.min_tick = float(price_prec) # Precision value often IS the tick size
            if amount_prec is not None:
                details.amount_precision_digits = int(self.exchange.decimal_to_precision(amount_prec, ccxt.ROUND, counting_mode=self.exchange.precisionMode))

            # Fallback/Validation for min_tick
            if details.min_tick <= 0:
                 details.min_tick = 1 / (10 ** details.price_precision_digits) if details.price_precision_digits >= 0 else 0.01
                 log_warning(f"Could not determine valid min_tick from market info, using calculated: {details.min_tick}")

            # Limits
            details.min_amount = market_info.get('limits', {}).get('amount', {}).get('min')
            details.min_cost = market_info.get('limits', {}).get('cost', {}).get('min')

            # Currencies & Market Type
            details.base_currency = market_info.get('base')
            details.quote_currency = market_info.get('quote')
            details.is_contract_market = market_info.get('swap', False) or \
                                         market_info.get('future', False) or \
                                         market_info.get('contract', False) or \
                                         market_info.get('type') in ['swap', 'future']

            # Capabilities
            details.supports_oco = self.exchange.has.get('oco', False)
            details.supports_fetch_positions = self.exchange.has.get('fetchPositions', False)
            details.supports_stop_market = self.exchange.has.get('createStopMarketOrder', False) or self.exchange.has.get('createStopOrder', False) # Check generic too
            details.supports_stop_limit = self.exchange.has.get('createStopLimitOrder', False)
            # ReduceOnly support check - rough guess, might need refinement per exchange
            details.supports_reduce_only = details.is_contract_market and \
                                           self.config.exchange_id in ['bybit', 'binance', 'kucoinfutures', 'okx', 'gateio'] # Common ones

            # Log details
            log_info(f"Market Info | Base: {details.base_currency}, Quote: {details.quote_currency}, Contract: {details.is_contract_market}")
            log_info(f"Precision   | Price: {details.price_precision_digits} decimals (Tick: {details.min_tick:.{details.price_precision_digits+2}f}), Amount: {details.amount_precision_digits} decimals")
            if details.min_amount is not None: log_info(f"Limits      | Min Amount: {details.min_amount} {details.base_currency or ''}")
            if details.min_cost is not None: log_info(f"Limits      | Min Cost: {details.min_cost} {details.quote_currency or ''}")
            log_info(f"Capabilities| OCO: {details.supports_oco}, FetchPos: {details.supports_fetch_positions}, StopMrkt: {details.supports_stop_market}, StopLim: {details.supports_stop_limit}, ReduceOnlyParam: {details.supports_reduce_only}")

            self._market_details[sym_upper] = details
            return True

        except KeyError as e:
            log_error(f"[ExchangeHandler] CRITICAL: Error accessing market info for {sym_upper}. Market data might be incomplete. Key: {e}", exc_info=True)
            return False
        except Exception as e:
            log_error(f"[ExchangeHandler] CRITICAL: Error processing market details for {sym_upper}: {e}", exc_info=True)
            return False

    # --- Formatting Helpers ---
    def format_price(self, symbol: str, price: float) -> float:
        details = self.get_market_details(symbol)
        if not details: raise ValueError(f"Cannot format price, market details not found for {symbol}")
        return float(self.exchange.price_to_precision(symbol, price))

    def format_amount(self, symbol: str, amount: float) -> float:
        details = self.get_market_details(symbol)
        if not details: raise ValueError(f"Cannot format amount, market details not found for {symbol}")
        return float(self.exchange.amount_to_precision(symbol, amount))

    # --- API Call Wrappers ---
    @_api_retry_decorator
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        log_debug(f"[ExchangeHandler] Fetching {limit} OHLCV for {symbol} ({timeframe})...")
        try:
            details = self.get_market_details(symbol) # Ensure market exists via details cache
            if not details: return None

            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                log_warning(f"[ExchangeHandler] No OHLCV data returned for {symbol} ({timeframe}).")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)

            if df.empty:
                log_warning(f"[ExchangeHandler] DataFrame empty after cleaning OHLCV for {symbol} ({timeframe}).")
                return None

            log_debug(f"[ExchangeHandler] Successfully fetched and cleaned {len(df)} candles.")
            return df

        except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
            log_error(f"[ExchangeHandler] CCXT error fetching OHLCV for {symbol}: {e}")
            return None
        except Exception as e: # Decorator handles RETRYABLE_EXCEPTIONS
            if not isinstance(e, RETRYABLE_EXCEPTIONS):
                 log_error(f"[ExchangeHandler] Unexpected error fetching OHLCV: {e}", exc_info=True)
            return None

    @_api_retry_decorator
    def fetch_balance(self) -> Dict[str, Any]:
         log_debug("[ExchangeHandler] Fetching balance...")
         balance = self.exchange.fetch_balance()
         log_debug("[ExchangeHandler] Balance fetched.")
         return balance

    @_api_retry_decorator
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        log_debug(f"[ExchangeHandler] Fetching ticker for {symbol}...")
        ticker = self.exchange.fetch_ticker(symbol)
        log_debug(f"[ExchangeHandler] Ticker fetched for {symbol}.")
        return ticker

    @_api_retry_decorator
    def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
         log_debug(f"[ExchangeHandler] Fetching open orders for {symbol}...")
         orders = self.exchange.fetch_open_orders(symbol)
         log_debug(f"[ExchangeHandler] Found {len(orders)} open orders for {symbol}.")
         return orders

    @_api_retry_decorator
    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        details = self.get_market_details(self.config.symbol)
        if not details or not details.supports_fetch_positions:
            log_warning(f"[ExchangeHandler] fetchPositions not supported or market details unavailable for {self.config.symbol}.")
            return []

        symbols_list = [symbol] if symbol else None # Pass symbol if specified
        log_debug(f"[ExchangeHandler] Fetching positions{' for ' + symbol if symbol else ''}...")
        try:
            positions = self.exchange.fetch_positions(symbols=symbols_list)
            log_debug(f"[ExchangeHandler] Fetched {len(positions)} position entries.")
            return positions
        except ccxt.NotSupported:
             log_warning(f"[ExchangeHandler] Exchange reports fetchPositions not supported, despite 'has' flag.")
             return []
        except Exception as e:
             if not isinstance(e, RETRYABLE_EXCEPTIONS):
                  log_error(f"[ExchangeHandler] Error fetching positions: {e}", exc_info=True)
             return [] # Return empty list on error


    @_api_retry_decorator
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if not order_id or not isinstance(order_id, str):
            log_debug(f"[ExchangeHandler] Invalid order ID for cancellation: '{order_id}'.")
            return False

        details = self.get_market_details(symbol)
        if not details: return False # Cannot cancel without details

        log_info(f"[ExchangeHandler] Attempting to cancel order ID {order_id} for {symbol}...")

        if self.config.simulation_mode:
            log_warning(f"SIMULATION: Skipped cancelling order {order_id}.")
            return True

        try:
            log_warning(f"!!! LIVE MODE: Sending cancel request for order {order_id}.")
            self.exchange.cancel_order(order_id, symbol)
            log_info(f"[ExchangeHandler] Cancel request for order {order_id} sent successfully.")
            return True
        except ccxt.OrderNotFound:
            log_info(f"[ExchangeHandler] Order {order_id} not found. Assumed closed/cancelled.")
            return True
        except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
            log_error(f"[ExchangeHandler] Failed to cancel order {order_id}: {type(e).__name__} - {e}")
            return False
        except Exception as e:
             if not isinstance(e, RETRYABLE_EXCEPTIONS):
                  log_error(f"[ExchangeHandler] Unexpected error cancelling order {order_id}: {e}", exc_info=True)
             return False

    @_api_retry_decorator
    def place_market_order(self, symbol: str, side: str, amount: float, reduce_only: bool = False) -> Optional[Dict[str, Any]]:
        details = self.get_market_details(symbol)
        if not details: return None

        if side not in ['buy', 'sell']:
            log_error(f"[ExchangeHandler] Invalid side '{side}' for market order."); return None
        if not isinstance(amount, (float, int)) or amount <= 0:
            log_error(f"[ExchangeHandler] Invalid amount '{amount}' for market order."); return None

        try:
            amount_fmt = self.format_amount(symbol, amount)
            if amount_fmt <= 0:
                log_error(f"Amount {amount} formatted to {amount_fmt}, <= 0."); return None
        except Exception as fmt_e:
            log_error(f"[ExchangeHandler] Failed to format market order amount {amount}: {fmt_e}"); return None

        params = {}
        reduce_only_applied = False
        if reduce_only and details.is_contract_market:
            if details.supports_reduce_only: # Use cached capability
                 params['reduceOnly'] = True
                 reduce_only_applied = True
                 log_debug("[ExchangeHandler] Applying reduceOnly=True parameter.")
            else:
                 log_warning(f"[ExchangeHandler] ReduceOnly requested for {symbol} but capability unclear/unsupported. Order might fail or ignore flag.")
                 # We might still pass it if ccxt handles it implicitly for some exchanges
                 params['reduceOnly'] = True # Attempt anyway? Or remove this line? Safer to remove if unsure.
                 # params.pop('reduceOnly', None) # Safer option if unsure
                 reduce_only_applied = True # Mark as intended

        log_info(f"[ExchangeHandler] Attempting MARKET {side.upper()} order:")
        log_info(f"  Symbol: {symbol}, Amount: {amount_fmt:.{details.amount_precision_digits}f} {details.base_currency or ''}{' (Reduce-Only)' if reduce_only_applied else ''}")

        order_result: Optional[Dict[str, Any]] = None
        if self.config.simulation_mode:
            sim_ts = int(time.time() * 1000)
            sim_id = f"{SIMULATION_ORDER_PREFIX}market_{side}_{sim_ts}"
            sim_price = 0.0
            try:
                ticker = self.fetch_ticker(symbol) # Already decorated
                sim_price = ticker.get('last', ticker.get('close', 0.0))
                if sim_price <= 0: sim_price = ticker.get('bid', 0.0) if side == 'sell' else ticker.get('ask', 0.0)
            except Exception as e: log_warning(f"Sim price fetch failed: {e}")

            sim_cost = amount_fmt * sim_price if sim_price > 0 else 0.0
            order_result = {
                'id': sim_id, 'clientOrderId': sim_id, 'timestamp': sim_ts,
                'datetime': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'closed',
                'symbol': symbol, 'type': 'market', 'timeInForce': 'IOC', 'side': side,
                'price': sim_price, 'average': sim_price, 'amount': amount_fmt,
                'filled': amount_fmt, 'remaining': 0.0, 'cost': sim_cost, 'fee': None,
                'reduceOnly': reduce_only_applied, 'info': {'simulated': True, 'simulated_price': sim_price}
            }
            log_info(f"Simulated market order result (ID: {sim_id}).")
        else:
            try:
                log_warning(f"!!! LIVE MODE: Placing real market order{' (Reduce-Only)' if reduce_only_applied else ''}.")
                order_result = self.exchange.create_market_order(symbol=symbol, side=side, amount=amount_fmt, params=params)
                log_info("[ExchangeHandler] Market order request sent.")
                time.sleep(1.5) # Allow potential fill
            except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                log_error(f"[ExchangeHandler] Failed to place market order: {type(e).__name__} - {e}"); return None
            except Exception as e:
                 if not isinstance(e, RETRYABLE_EXCEPTIONS):
                      log_error(f"[ExchangeHandler] Unexpected error placing market order: {e}", exc_info=True)
                 return None

        # Process result (common for sim/live)
        if order_result:
            o_id = order_result.get('id', 'N/A')
            o_status = order_result.get('status', 'N/A')
            o_avg_price = order_result.get('average', order_result.get('price'))
            o_filled = order_result.get('filled')
            o_cost = order_result.get('cost')
            o_reduce = order_result.get('reduceOnly', params.get('reduceOnly', 'N/A'))

            p_str = f"{o_avg_price:.{details.price_precision_digits}f}" if isinstance(o_avg_price, (float, int)) else "N/A"
            f_str = f"{o_filled:.{details.amount_precision_digits}f}" if isinstance(o_filled, (float, int)) else "N/A"
            c_str = f"{o_cost:.{details.price_precision_digits}f}" if isinstance(o_cost, (float, int)) else "N/A"
            log_info(f"Order Result | ID: {o_id}, Status: {o_status}, Avg Px: {p_str}, Filled: {f_str}, Cost: {c_str}, Reduce: {o_reduce}")

            if not self.config.simulation_mode and o_status == 'open':
                 log_warning(f"Market order {o_id} status 'open'. Might not have filled yet.")
            elif o_status in ['rejected', 'expired']:
                 log_error(f"Market order {o_id} failed with status: {o_status}"); return None
            return order_result
        else:
            log_error("[ExchangeHandler] Market order placement did not return result object."); return None


    def place_protection_orders(self, symbol: str, pos_side: str, qty: float, sl_pr: float, tp_pr: float) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Places SL/TP protection, preferring OCO if supported, falling back to separate.
        Returns (sl_order_id, tp_order_id, oco_order_id).
        """
        details = self.get_market_details(symbol)
        if not details: return None, None, None

        sl_order_id: Optional[str] = None
        tp_order_id: Optional[str] = None
        oco_ref_id: Optional[str] = None

        # --- Input Validation ---
        if pos_side not in ['long', 'short']: log_error("Invalid position side"); return None, None, None
        if not (isinstance(qty, (float, int)) and qty > 0): log_error("Invalid qty"); return None, None, None
        if not (isinstance(sl_pr, (float, int)) and sl_pr > 0 and isinstance(tp_pr, (float, int)) and tp_pr > 0): log_error("Invalid SL/TP price"); return None, None, None
        if (pos_side == 'long' and sl_pr >= tp_pr) or (pos_side == 'short' and sl_pr <= tp_pr): log_error("Invalid SL/TP logic"); return None, None, None

        close_side: str = 'sell' if pos_side == 'long' else 'buy'

        try: # Format Inputs
            qty_fmt = self.format_amount(symbol, qty)
            sl_pr_fmt = self.format_price(symbol, sl_pr)
            tp_pr_fmt = self.format_price(symbol, tp_pr)
            if qty_fmt <= 0: raise ValueError("Qty formatted to zero")
            if (pos_side == 'long' and sl_pr_fmt >= tp_pr_fmt) or (pos_side == 'short' and sl_pr_fmt <= tp_pr_fmt):
                 raise ValueError(f"SL/TP logic invalid after formatting: SL={sl_pr_fmt}, TP={tp_pr_fmt}")
        except Exception as fmt_e:
            log_error(f"[ExchangeHandler] Failed to format SL/TP inputs: {fmt_e}"); return None, None, None

        # --- Common Params ---
        params = {'reduceOnly': True} if details.is_contract_market else {}
        if params: log_debug("[ExchangeHandler] Applying reduceOnly=True to protection orders.")

        # --- Determine SL Type ---
        sl_order_type: str = 'stopMarket' # Default preference
        sl_limit_price: Optional[float] = None
        stop_price_param_name = 'stopPrice' # Common, but might change per exchange/method

        if details.supports_stop_market:
            sl_order_type = 'stopMarket' # Or 'stop' if that's what ccxt uses for the generic one
            log_debug(f"[ExchangeHandler] Selected SL type: {sl_order_type}. Trigger: {sl_pr_fmt}")
        elif details.supports_stop_limit:
            sl_order_type = 'stopLimit'
            limit_offset = details.min_tick * 10
            limit_raw = sl_pr_fmt - limit_offset if close_side == 'sell' else sl_pr_fmt + limit_offset
            # Adjust if offset crossed trigger
            if (close_side == 'sell' and limit_raw >= sl_pr_fmt) or (close_side == 'buy' and limit_raw <= sl_pr_fmt):
                limit_raw = sl_pr_fmt - details.min_tick if close_side == 'sell' else sl_pr_fmt + details.min_tick
            try:
                sl_limit_price = self.format_price(symbol, limit_raw)
            except Exception as fmt_e:
                log_error(f"Failed to format SL limit price {limit_raw}: {fmt_e}"); return None, None, None
            log_warning(f"[ExchangeHandler] Using fallback SL type: {sl_order_type}. Trigger: {sl_pr_fmt}, Limit: {sl_limit_price}.")
        else:
            log_error(f"[ExchangeHandler] Exchange {self.name} does not support Stop Market or Stop Limit orders via ccxt."); return None, None, None

        # --- Attempt OCO Placement ---
        oco_attempted = False
        oco_succeeded = False
        if details.supports_oco:
            oco_attempted = True
            log_info(f"[ExchangeHandler] Attempting OCO {close_side.upper()} placement...")
            log_info(f"  Qty: {qty_fmt}, TP: {tp_pr_fmt}, SL Trigger: {sl_pr_fmt} ({sl_order_type}{f', SL Limit: {sl_limit_price}' if sl_limit_price else ''})")
            if params: log_info(f"  Params: {params}")

            try:
                oco_params = params.copy()
                # --- !!! EXCHANGE-SPECIFIC OCO PARAMS NEEDED HERE !!! ---
                # This structure is a GUESS and likely NEEDS ADJUSTMENT per exchange
                oco_params[stop_price_param_name] = sl_pr_fmt # SL Trigger
                if sl_order_type == 'stopLimit' and sl_limit_price is not None:
                     # Common param name, check exchange docs
                     oco_params['stopLimitPrice'] = sl_limit_price

                # The 'type' for OCO also varies. Common: 'limit' or maybe 'oco'.
                oco_order_type_guess = 'limit' # Needs verification!

                if self.config.simulation_mode:
                    sim_oco_id = f"{SIMULATION_ORDER_PREFIX}oco_{close_side}_{int(time.time())}"
                    log_warning("!!! SIMULATION: OCO order placement skipped.")
                    oco_ref_id = sim_oco_id
                    oco_succeeded = True
                else:
                    # --- !!! VERIFY AND REPLACE THIS CALL FOR YOUR EXCHANGE !!! ---
                    log_warning(f"!!! LIVE MODE: Attempting OCO for {self.name}. Type='{oco_order_type_guess}', Params={oco_params}. VERIFY THIS STRUCTURE!")
                    # raise ccxt.NotSupported(f"Generic OCO structure used for {self.name}. Implement exchange-specific OCO logic or rely on fallback.") # Safer default

                    # --- Example Placeholder Call (NEEDS VERIFICATION) ---
                    # order_result = self.exchange.create_order(
                    #     symbol=symbol, type=oco_order_type_guess, side=close_side,
                    #     amount=qty_fmt, price=tp_pr_fmt, params=oco_params
                    # )
                    # --- Process OCO Result (If placeholder replaced and successful) ---
                    # if order_result and (order_result.get('id') or order_result.get('info')):
                    #     # Extract OCO reference ID (highly exchange dependent)
                    #     oco_ref_id = order_result.get('id')
                    #     if not oco_ref_id and isinstance(order_result.get('info'), dict):
                    #          oco_ref_id = order_result['info'].get('listClientOrderId', order_result['info'].get('orderListId')) # Binance/Bybit common
                    #     if oco_ref_id:
                    #         log_info(f"OCO request processed. Ref ID: {oco_ref_id}, Status: {order_result.get('status','?')}")
                    #         oco_succeeded = True
                    #     else: log_error(f"OCO placed but failed to get Ref ID. Details: {order_result}")
                    # else: log_error(f"OCO placement failed or returned unexpected result: {order_result}")
                    pass # Remove this 'pass' when implementing real OCO call

            except ccxt.NotSupported as e: log_warning(f"OCO not supported/structure incorrect for {self.name}: {e}. Falling back.")
            except (ccxt.InvalidOrder, ccxt.ExchangeError) as e: log_error(f"OCO placement failed: {type(e).__name__} - {e}. Falling back.")
            except Exception as e:
                 if not isinstance(e, RETRYABLE_EXCEPTIONS): log_error(f"Unexpected OCO error: {e}. Falling back.", exc_info=True)

        if oco_succeeded:
            return None, None, oco_ref_id

        # --- Fallback: Place Separate Orders ---
        log_info(f"[ExchangeHandler] Placing separate SL and TP orders{' (OCO fallback)' if oco_attempted else ''}.")
        sl_placed_ok = False
        tp_placed_ok = False

        # 1. Place SL
        try:
            log_info(f"Attempting separate SL ({sl_order_type}): Qty={qty_fmt}, Side={close_side}, Trigger={sl_pr_fmt}{f', Limit={sl_limit_price}' if sl_limit_price else ''}")
            sl_params_sep = params.copy()
            sl_params_sep[stop_price_param_name] = sl_pr_fmt # Stop price usually needed in params too

            if self.config.simulation_mode:
                sl_order_id = f"{SIMULATION_ORDER_PREFIX}sl_{close_side}_{int(time.time())}"
                log_warning(f"SIMULATION: Separate SL skipped. Mock ID: {sl_order_id}"); sl_placed_ok = True
            else:
                log_warning(f"!!! LIVE MODE: Placing separate {sl_order_type} SL order.")
                # Use create_order for flexibility with stop types
                sl_order = self.exchange.create_order(
                    symbol=symbol, type=sl_order_type, side=close_side,
                    amount=qty_fmt, price=sl_limit_price, # Price arg used for limit in stopLimit, ignored/None for stopMarket
                    params=sl_params_sep
                )
                if sl_order and sl_order.get('id'):
                    sl_order_id = sl_order['id']; sl_placed_ok = True
                    log_info(f"Separate SL ({sl_order_type}) placed. ID: {sl_order_id}")
                    time.sleep(0.5)
                else: log_error(f"Separate SL ({sl_order_type}) failed or no ID returned: {sl_order}")
        except (ccxt.InvalidOrder, ccxt.ExchangeError) as e: log_error(f"Error placing separate SL: {type(e).__name__} - {e}")
        except Exception as e:
             if not isinstance(e, RETRYABLE_EXCEPTIONS): log_error(f"Unexpected error placing separate SL: {e}", exc_info=True)

        # 2. Place TP (Limit Order)
        try:
            log_info(f"Attempting separate TP (Limit): Qty={qty_fmt}, Side={close_side}, Price={tp_pr_fmt}")
            tp_params_sep = params.copy()

            if self.config.simulation_mode:
                tp_order_id = f"{SIMULATION_ORDER_PREFIX}tp_{close_side}_{int(time.time())}"
                log_warning(f"SIMULATION: Separate TP skipped. Mock ID: {tp_order_id}"); tp_placed_ok = True
            else:
                log_warning(f"!!! LIVE MODE: Placing separate limit TP order.")
                tp_order = self.exchange.create_limit_order(
                    symbol=symbol, side=close_side, amount=qty_fmt, price=tp_pr_fmt, params=tp_params_sep
                )
                if tp_order and tp_order.get('id'):
                    tp_order_id = tp_order['id']; tp_placed_ok = True
                    log_info(f"Separate TP (Limit) placed. ID: {tp_order_id}")
                    time.sleep(0.5)
                else: log_error(f"Separate TP (Limit) failed or no ID returned: {tp_order}")
        except (ccxt.InvalidOrder, ccxt.ExchangeError) as e: log_error(f"Error placing separate TP: {type(e).__name__} - {e}")
        except Exception as e:
             if not isinstance(e, RETRYABLE_EXCEPTIONS): log_error(f"Unexpected error placing separate TP: {e}", exc_info=True)

        # --- Final Outcome ---
        if not sl_placed_ok and not tp_placed_ok: log_error("Both separate SL & TP placements failed."); return None, None, None
        elif not sl_placed_ok: log_warning("Separate TP placed, but SL FAILED. Partial protection (TP only).")
        elif not tp_placed_ok: log_warning("Separate SL placed, but TP FAILED. Partial protection (SL only).")

        return sl_order_id, tp_order_id, None


# --- Position Manager Class ---
@dataclass
class PositionState:
    """Holds the state of the current trading position."""
    status: Optional[str] = None      # 'long', 'short', or None
    entry_price: Optional[float] = None
    quantity: Optional[float] = None
    order_id: Optional[str] = None      # Initial entry order ID
    stop_loss: Optional[float] = None     # Target SL price
    take_profit: Optional[float] = None   # Target TP price
    entry_time: Optional[pd.Timestamp] = None # UTC timestamp

    # Protection order tracking
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    oco_order_id: Optional[str] = None

    # Trailing Stop Loss tracking
    highest_price_since_entry: Optional[float] = None
    lowest_price_since_entry: Optional[float] = None
    current_trailing_sl_price: Optional[float] = None # Active TSL price

    def is_active(self) -> bool:
        return self.status is not None

    def get_protection_order_ids(self) -> List[str]:
        """Returns a list of active protection order IDs."""
        ids = []
        if self.oco_order_id: ids.append(self.oco_order_id)
        else:
            if self.sl_order_id: ids.append(self.sl_order_id)
            if self.tp_order_id: ids.append(self.tp_order_id)
        return ids

    def to_dict(self) -> Dict[str, Any]:
        """Converts state to a dictionary for saving (handles timestamp)."""
        d = self.__dict__.copy()
        if isinstance(d.get('entry_time'), pd.Timestamp):
            d['entry_time'] = d['entry_time'].isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionState':
        """Creates state from a dictionary (handles timestamp)."""
        # Ensure all keys exist, potentially with None values
        state_data = {k: data.get(k) for k in cls.__annotations__}

        # Convert timestamp string back to Timestamp object
        entry_time_str = state_data.get('entry_time')
        if isinstance(entry_time_str, str):
            try:
                ts = pd.Timestamp(entry_time_str)
                state_data['entry_time'] = ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')
            except ValueError:
                log_warning(f"Invalid timestamp format '{entry_time_str}' in state file. Resetting entry_time.")
                state_data['entry_time'] = None
        elif entry_time_str is not None and not isinstance(entry_time_str, pd.Timestamp):
             log_warning(f"Unexpected type for entry_time '{type(entry_time_str)}' in state file. Resetting.")
             state_data['entry_time'] = None

        # Convert numbers from potential int loading to float where expected
        for key in ['entry_price', 'quantity', 'stop_loss', 'take_profit',
                    'highest_price_since_entry', 'lowest_price_since_entry',
                    'current_trailing_sl_price']:
            if key in state_data and isinstance(state_data[key], int):
                state_data[key] = float(state_data[key])

        return cls(**state_data)

class PositionManager:
    """Manages the bot's trading position state."""

    def __init__(self, config: Config, exchange_handler: ExchangeHandler, state_file: str = POSITION_STATE_FILE):
        self.config = config
        self.exchange_handler = exchange_handler
        self.state_file_path = Path(state_file)
        self.state = PositionState() # Initialize with empty state
        self.load_state()

    def save_state(self):
        """Saves the current position state to the JSON file."""
        log_debug(f"[PositionManager] Saving position state to '{self.state_file_path}'")
        try:
            state_dict = self.state.to_dict()
            with self.state_file_path.open('w') as f:
                json.dump(state_dict, f, indent=4)
            log_debug("[PositionManager] Position state saved successfully.")
        except IOError as e:
            log_error(f"[PositionManager] Error saving state file '{self.state_file_path}': {e}", exc_info=True)
        except Exception as e:
            log_error(f"[PositionManager] Unexpected error saving state: {e}", exc_info=True)

    def load_state(self):
        """Loads and validates position state from the JSON file."""
        log_info(f"[PositionManager] Attempting to load state from '{self.state_file_path}'...")
        if not self.state_file_path.exists():
            log_info(f"[PositionManager] No state file found. Starting fresh.")
            self.state = PositionState()
            return

        try:
            with self.state_file_path.open('r') as f:
                loaded_data = json.load(f)

            # Basic validation: check if it's a dictionary
            if not isinstance(loaded_data, dict):
                log_error(f"State file '{self.state_file_path}' content is not a valid dictionary. Starting fresh.")
                self.state = PositionState()
                # Optionally rename the corrupted file
                try: self.state_file_path.rename(self.state_file_path.with_suffix('.json.corrupted'))
                except OSError: pass
                return

            self.state = PositionState.from_dict(loaded_data)
            log_info("[PositionManager] State file loaded. Validating content...")

            # Additional validation consistency checks
            issues_found = []
            if self.state.is_active():
                if not isinstance(self.state.entry_price, (float, int)) or \
                   not isinstance(self.state.quantity, (float, int)) or \
                   not isinstance(self.state.stop_loss, (float, int)) or \
                   not isinstance(self.state.take_profit, (float, int)):
                    issues_found.append(f"Active position ('{self.state.status}') missing essential numeric data (entry, qty, sl, tp).")
                # Check if TSL data makes sense if active
                if self.config.enable_trailing_stop and (self.state.highest_price_since_entry is None or self.state.lowest_price_since_entry is None):
                     issues_found.append("TSL enabled, but peak price tracking data missing.")

                # Check if protection order IDs are missing when expected
                if not self.state.get_protection_order_ids():
                     issues_found.append("Active position has no tracked protection order IDs (SL/TP/OCO).")

            if issues_found:
                log_warning(f"[PositionManager] Issues found during state validation:\n - " + "\n - ".join(issues_found))
                log_warning("[PositionManager] Consider resetting state or manual verification. Current state kept for now.")
                # Decide if state should be invalidated based on severity
                # Example: If essential data missing, force reset
                if "missing essential numeric data" in issues_found[0]:
                     log_error("Essential position data invalid. Resetting state to None.")
                     self.reset_state()
                     self.save_state() # Save the reset state

            log_info("[PositionManager] Position state loaded and validated.")
            self.display_status() # Show loaded status

        except json.JSONDecodeError as e:
            log_error(f"[PositionManager] Error decoding JSON from '{self.state_file_path}': {e}. Starting fresh.", exc_info=False)
            self.state = PositionState()
        except Exception as e:
            log_error(f"[PositionManager] Unexpected error loading state '{self.state_file_path}': {e}. Starting fresh.", exc_info=True)
            self.state = PositionState()

    def update_entry(self, side: str, entry_price: float, quantity: float, entry_order_id: str, sl_price: float, tp_price: float, sl_order_id: Optional[str], tp_order_id: Optional[str], oco_order_id: Optional[str]):
        """Updates the state when a new position is entered."""
        log_info(f"[PositionManager] Updating state for new {side.upper()} entry.")
        self.state = PositionState(
            status=side,
            entry_price=entry_price,
            quantity=quantity,
            order_id=entry_order_id,
            stop_loss=sl_price,
            take_profit=tp_price,
            entry_time=pd.Timestamp.now(tz='UTC'),
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
            oco_order_id=oco_order_id,
            highest_price_since_entry=entry_price, # Initialize TSL tracking
            lowest_price_since_entry=entry_price,
            current_trailing_sl_price=None
        )
        self.save_state()
        self.display_status()

    def reset_state(self, reason: str = "Unknown"):
        """Resets the position state to default (no active position)."""
        log_info(f"[PositionManager] Resetting position state. Reason: {reason}")
        self.state = PositionState()
        self.save_state()
        self.display_status()

    def display_status(self):
        """Displays the current position status."""
        details = self.exchange_handler.get_market_details(self.config.symbol)
        price_prec = details.price_precision_digits if details else DEFAULT_PRICE_PRECISION
        amount_prec = details.amount_precision_digits if details else DEFAULT_AMOUNT_PRECISION

        state = self.state
        status = state.status
        entry_price = state.entry_price
        quantity = state.quantity
        sl_target = state.stop_loss
        tp_target = state.take_profit
        tsl_active_price = state.current_trailing_sl_price
        entry_time = state.entry_time

        entry_str = f"{entry_price:.{price_prec}f}" if isinstance(entry_price, (float, int)) else "N/A"
        qty_str = f"{quantity:.{amount_prec}f}" if isinstance(quantity, (float, int)) else "N/A"
        sl_str = f"{sl_target:.{price_prec}f}" if isinstance(sl_target, (float, int)) else "N/A"
        tp_str = f"{tp_target:.{price_prec}f}" if isinstance(tp_target, (float, int)) else "N/A"
        tsl_str = f" | Active TSL: {tsl_active_price:.{price_prec}f}" if isinstance(tsl_active_price, (float, int)) else ""
        time_str = f" | Entered: {entry_time.strftime('%Y-%m-%d %H:%M')}" if isinstance(entry_time, pd.Timestamp) else ""

        if status == 'long': status_color, status_text = NEON_GREEN, "LONG"
        elif status == 'short': status_color, status_text = NEON_RED, "SHORT"
        else: status_color, status_text = NEON_CYAN, "None"

        print(f"{status_color}Position Status: {status_text}{RESET} | Entry: {entry_str} | Qty: {qty_str} | Target SL: {sl_str} | Target TP: {tp_str}{tsl_str}{time_str}")

        protection_info = []
        if state.oco_order_id: protection_info.append(f"OCO ID: {state.oco_order_id}")
        else:
            if state.sl_order_id: protection_info.append(f"SL ID: {state.sl_order_id}")
            if state.tp_order_id: protection_info.append(f"TP ID: {state.tp_order_id}")

        if protection_info:
            print(f"    Protection Orders: {', '.join(protection_info)}")
        elif state.is_active():
             print(f"    {NEON_YELLOW}Warning: Position active but no protection order IDs tracked.{RESET}")

    def check_position_and_orders(self) -> bool:
        """
        Checks consistency of local state against exchange orders/positions.
        Resets state if closure detected. Returns True if state was reset.
        """
        if not self.state.is_active():
            return False # Nothing to check

        log_debug(f"[PositionManager] Checking state consistency for active {self.state.status} position...")
        reset_required = False
        closure_reason = "Unknown"
        order_to_cancel: Optional[str] = None
        cancel_label: str = ""
        symbol = self.config.symbol
        details = self.exchange_handler.get_market_details(symbol)
        if not details: return False # Should not happen if active, but safety check

        try:
            # --- 1. Check Tracked Protection Orders ---
            tracked_ids = self.state.get_protection_order_ids()
            protection_order_still_open = False

            if tracked_ids:
                log_debug(f"Fetching open orders for {symbol} to check tracked IDs: {tracked_ids}")
                open_orders = self.exchange_handler.fetch_open_orders(symbol)
                open_order_ids = {o['id'] for o in open_orders if 'id' in o}
                log_debug(f"Found {len(open_order_ids)} open order IDs: {open_order_ids if open_order_ids else 'None'}")

                if self.state.oco_order_id:
                    if self.state.oco_order_id in open_order_ids:
                        protection_order_still_open = True
                        log_debug(f"Tracked OCO {self.state.oco_order_id} appears open.")
                    else:
                        reset_required = True
                        closure_reason = f"Tracked OCO {self.state.oco_order_id} not found open"
                        log_info(f"{NEON_YELLOW}Closure detected: {closure_reason}. Assuming OCO filled/cancelled.{RESET}")
                else: # Separate SL/TP
                    sl_is_open = self.state.sl_order_id in open_order_ids if self.state.sl_order_id else False
                    tp_is_open = self.state.tp_order_id in open_order_ids if self.state.tp_order_id else False
                    log_debug(f"Separate Order Check: SL Open={sl_is_open}, TP Open={tp_is_open}")

                    if self.state.sl_order_id and not sl_is_open and self.state.tp_order_id and not tp_is_open:
                        reset_required = True; closure_reason = f"Neither SL ({self.state.sl_order_id}) nor TP ({self.state.tp_order_id}) found open"
                        log_info(f"{NEON_YELLOW}Closure detected: {closure_reason}.{RESET}")
                    elif self.state.sl_order_id and not sl_is_open:
                        reset_required = True; closure_reason = f"Tracked SL {self.state.sl_order_id} not found open"
                        log_info(f"{NEON_YELLOW}Closure detected: {closure_reason} (SL assumed hit).{RESET}")
                        if tp_is_open: order_to_cancel = self.state.tp_order_id; cancel_label = "TP"
                    elif self.state.tp_order_id and not tp_is_open:
                        reset_required = True; closure_reason = f"Tracked TP {self.state.tp_order_id} not found open"
                        log_info(f"{NEON_GREEN}Closure detected: {closure_reason} (TP assumed hit).{RESET}")
                        if sl_is_open: order_to_cancel = self.state.sl_order_id; cancel_label = "SL"
                    elif sl_is_open or tp_is_open:
                        protection_order_still_open = True
                        log_debug("At least one separate protection order (SL/TP) is still open.")

            # --- 2. Fallback/Confirmation: Check Actual Position (Contracts) ---
            should_check_pos = details.is_contract_market and details.supports_fetch_positions and \
                               (not tracked_ids or not protection_order_still_open)

            if should_check_pos:
                log_info("Checking actual position status via fetchPositions...")
                positions_data = self.exchange_handler.fetch_positions(symbol) # Fetch for specific symbol
                position_found_on_exchange = False
                min_amount_threshold = details.min_amount if details.min_amount is not None else 1e-9

                for pos_info in positions_data:
                    if isinstance(pos_info, dict) and pos_info.get('symbol') == symbol:
                        # --- Refined Position Size Extraction ---
                        size_val = 0 # Default
                        if pos_info.get('contracts') is not None: size_val = pos_info['contracts']
                        elif isinstance(pos_info.get('info'), dict):
                            info_dict = pos_info['info']
                            size_val = info_dict.get('size', info_dict.get('positionAmt', info_dict.get('qty', 0)))
                        # --- End Refined Extraction ---
                        try: current_pos_size = float(size_val) if size_val is not None else 0.0
                        except (ValueError, TypeError): current_pos_size = 0.0

                        pos_side_exch = pos_info.get('side', pos_info.get('info', {}).get('side', 'unknown')).lower()
                        if pos_side_exch == 'unknown' and 'positionAmt' in pos_info.get('info', {}):
                            amt_sign = float(pos_info['info'].get('positionAmt', 0.0))
                            if amt_sign > min_amount_threshold: pos_side_exch = 'long'
                            elif amt_sign < -min_amount_threshold: pos_side_exch = 'short'

                        log_debug(f"fetchPositions found {symbol}: Size={current_pos_size}, Side='{pos_side_exch}'")

                        if abs(current_pos_size) > min_amount_threshold:
                            position_found_on_exchange = True
                            log_info(f"fetchPositions confirms active {pos_side_exch} position (Size: {current_pos_size}).")
                            if self.state.status == pos_side_exch:
                                if reset_required:
                                    log_warning("Position check confirms active state. Overriding closure detected by order check.")
                                    reset_required = False; order_to_cancel = None
                            else:
                                log_error(f"!!! STATE MISMATCH !!! Local: {self.state.status}, Exchange: {pos_side_exch} (Size: {current_pos_size})")
                                reset_required = True; closure_reason = "State Mismatch via fetchPositions"
                                order_to_cancel = self.state.oco_order_id or self.state.sl_order_id or self.state.tp_order_id
                                cancel_label = "SL/TP/OCO due to mismatch"
                            break # Found our symbol

                if not position_found_on_exchange and self.state.is_active():
                    log_info(f"Position check via fetchPositions confirmed no active position for {symbol}.")
                    if not reset_required:
                        reset_required = True; closure_reason = "Position not found via fetchPositions"
                        log_info(f"{NEON_YELLOW}Closure confirmed: {closure_reason}. Resetting.{RESET}")
                        order_to_cancel = self.state.oco_order_id or self.state.sl_order_id or self.state.tp_order_id
                        cancel_label = "SL/TP/OCO as position closed"

            # --- 3. Perform Reset and Cleanup ---
            if reset_required:
                log_info(f"Initiating state reset. Reason: {closure_reason}")
                if order_to_cancel and cancel_label:
                    log_info(f"Attempting cancellation of lingering {cancel_label} order: {order_to_cancel}")
                    cancelled = self.exchange_handler.cancel_order(order_to_cancel, symbol)
                    log_info(f"Cancellation request for {order_to_cancel} {'sent/confirmed' if cancelled else 'failed'}.")
                self.reset_state(closure_reason) # Resets state and saves
                return True # State was reset

            log_debug("Position state appears consistent.")
            return False # State not reset

        except (ccxt.AuthenticationError, ccxt.ExchangeError) as e:
            log_error(f"[PositionManager] Exchange communication error during check: {e}. State unchanged.", exc_info=False)
            return False
        except Exception as e:
             if not isinstance(e, RETRYABLE_EXCEPTIONS):
                 log_error(f"[PositionManager] Unexpected error during consistency check: {e}. State unchanged.", exc_info=True)
             return False


    def update_trailing_stop(self, current_price: float, current_atr: Optional[float]):
        """Checks and updates the trailing stop loss if conditions are met."""
        if not self.config.enable_trailing_stop or not self.state.is_active() or current_atr is None or current_atr <= 0:
            return

        details = self.exchange_handler.get_market_details(self.config.symbol)
        if not details: return

        state = self.state
        pos_side = state.status
        entry_price = state.entry_price
        initial_sl = state.stop_loss # This gets updated by TSL, serves as the *current* SL target
        original_tp = state.take_profit # Original TP target remains
        qty = state.quantity
        current_tsl_price = state.current_trailing_sl_price # The active TSL price (if moved)
        highest_seen = state.highest_price_since_entry
        lowest_seen = state.lowest_price_since_entry

        if None in [entry_price, initial_sl, original_tp, qty, highest_seen, lowest_seen]:
            log_warning("[PositionManager] TSL Check: Missing essential position data."); return

        effective_sl_price = current_tsl_price if current_tsl_price is not None else initial_sl
        log_debug(f"TSL Check ({pos_side}): Price={current_price:.{details.price_precision_digits}f}, Eff SL={effective_sl_price:.{details.price_precision_digits}f}, ATR={current_atr:.{details.price_precision_digits}f}")

        # Update peak prices
        peak_updated = False
        if pos_side == 'long' and current_price > highest_seen:
            state.highest_price_since_entry = current_price; peak_updated = True
            log_debug(f"TSL: New high price: {current_price:.{details.price_precision_digits}f}")
        elif pos_side == 'short' and current_price < lowest_seen:
            state.lowest_price_since_entry = current_price; peak_updated = True
            log_debug(f"TSL: New low price: {current_price:.{details.price_precision_digits}f}")
        # --- Save state if peak price updated ---
        if peak_updated: self.save_state()

        # Check activation and calculate potential new TSL
        tsl_activated = False
        potential_new_tsl: Optional[float] = None
        cfg_act_mult = self.config.trailing_stop_activation_atr_multiplier
        cfg_trail_mult = self.config.trailing_stop_atr_multiplier

        if pos_side == 'long':
            activation_thresh = entry_price + (current_atr * cfg_act_mult)
            if state.highest_price_since_entry > activation_thresh:
                tsl_activated = True
                potential_new_tsl = state.highest_price_since_entry - (current_atr * cfg_trail_mult)
        else: # Short
            activation_thresh = entry_price - (current_atr * cfg_act_mult)
            if state.lowest_price_since_entry < activation_thresh:
                tsl_activated = True
                potential_new_tsl = state.lowest_price_since_entry + (current_atr * cfg_trail_mult)

        if not tsl_activated or potential_new_tsl is None:
            log_debug("TSL not activated or potential SL calculation failed."); return

        # Check if potential new TSL is an improvement and valid
        should_update = False
        new_tsl_fmt: Optional[float] = None
        try:
            new_tsl_fmt = self.exchange_handler.format_price(self.config.symbol, potential_new_tsl)
        except Exception as fmt_e:
            log_error(f"TSL Update: Failed to format potential TSL price {potential_new_tsl}: {fmt_e}"); return

        if pos_side == 'long':
            if new_tsl_fmt > effective_sl_price and new_tsl_fmt < current_price: should_update = True
        else: # Short
            if new_tsl_fmt < effective_sl_price and new_tsl_fmt > current_price: should_update = True

        if not should_update:
             log_debug(f"TSL Update Check: New SL ({new_tsl_fmt:.{details.price_precision_digits}f}) not improved/valid vs Effective SL ({effective_sl_price:.{details.price_precision_digits}f}) & Price ({current_price:.{details.price_precision_digits}f}).")
             return

        # --- Execute TSL Update ---
        log_info(f"{NEON_YELLOW}Trailing Stop Update Triggered! Moving SL to {new_tsl_fmt:.{details.price_precision_digits}f}{RESET}")

        # 1. Cancel Existing Protection
        ids_to_cancel = state.get_protection_order_ids()
        if not ids_to_cancel:
            log_error("TSL CRITICAL: Update triggered, but no protection IDs found to cancel!"); return

        all_cancelled = True
        for order_id in ids_to_cancel:
            log_info(f"TSL: Cancelling existing protection order: {order_id}")
            if not self.exchange_handler.cancel_order(order_id, self.config.symbol):
                log_error(f"TSL CRITICAL: Failed to cancel order {order_id}. Aborting TSL update."); all_cancelled = False; break
        if not all_cancelled: return

        log_info("TSL: Existing protection order(s) cancelled.")
        state.sl_order_id = state.tp_order_id = state.oco_order_id = None # Clear old IDs immediately

        # 2. Place New Protection with Updated SL
        log_info(f"TSL: Placing new protection: SL @ {new_tsl_fmt:.{details.price_precision_digits}f}, Original TP @ {original_tp:.{details.price_precision_digits}f}")
        new_sl_id, new_tp_id, new_oco_id = self.exchange_handler.place_protection_orders(
            symbol=self.config.symbol, pos_side=pos_side, qty=qty,
            sl_pr=new_tsl_fmt, tp_pr=original_tp
        )

        # 3. Update State
        protection_placed = bool(new_oco_id or new_sl_id)
        if protection_placed:
            log_info(f"{NEON_GREEN}TSL Update Successful: New protection placed.{RESET}")
            if new_oco_id: log_info(f"  New OCO Ref ID: {new_oco_id}")
            if new_sl_id: log_info(f"  New SL ID: {new_sl_id}")
            if new_tp_id: log_info(f"  New TP ID: {new_tp_id}")

            # Update state with new IDs and the new effective SL price
            state.sl_order_id = new_sl_id
            state.tp_order_id = new_tp_id
            state.oco_order_id = new_oco_id
            state.stop_loss = new_tsl_fmt  # Update the target SL
            state.current_trailing_sl_price = new_tsl_fmt # Mark as active TSL
            self.save_state()
            self.display_status()
            if not new_oco_id and not new_tp_id and new_sl_id:
                 log_warning("TSL Update Warning: New SL placed, but separate TP failed.")
        else:
            log_error(f"!!! TSL CRITICAL FAILURE !!! Old orders cancelled but FAILED to place new protection.")
            log_error(f"  !!! POSITION '{pos_side.upper()}' IS CURRENTLY UNPROTECTED !!!")
            # State reflects no protection IDs; save this state
            state.stop_loss = new_tsl_fmt # Keep target SL for reference
            state.current_trailing_sl_price = None # No longer active TSL
            self.save_state()
            log_error("State saved reflecting unprotected status. MANUAL INTERVENTION REQUIRED.")
            # Consider emergency close?
            # self.attempt_emergency_close()


    def calculate_position_size(self, entry: float, sl: float) -> Optional[float]:
        """Calculates position size based on risk, balance, and limits."""
        details = self.exchange_handler.get_market_details(self.config.symbol)
        if not details: return None

        if not (0 < self.config.risk_percentage < 1): log_error("Invalid risk percentage."); return None
        if entry <= 0 or sl <= 0: log_error("Invalid entry/SL price for size calc."); return None
        price_diff = abs(entry - sl)
        if price_diff < details.min_tick: log_error("SL too close to entry."); return None

        log_debug(f"Calculating position size: Entry={entry:.{details.price_precision_digits}f}, SL={sl:.{details.price_precision_digits}f}, Risk={self.config.risk_percentage*100:.2f}%")

        try:
            balance_info = self.exchange_handler.fetch_balance()
            quote_curr = details.quote_currency
            available_balance = 0.0
            source_info = "N/A"

            if quote_curr: # Simplified balance extraction - may need refinement per exchange like original code
                available_balance = balance_info.get(quote_curr, {}).get('free', 0.0)
                source_info = f"['{quote_curr}']['free']"
                if available_balance <= 0 and balance_info.get(quote_curr, {}).get('total', 0.0) > 0:
                     available_balance = balance_info[quote_curr]['total']
                     source_info = f"['{quote_curr}']['total'] (free was zero)"
                elif available_balance <= 0 and 'free' in balance_info and isinstance(balance_info['free'], dict):
                     available_balance = balance_info['free'].get(quote_curr, 0.0)
                     source_info = f"['free']['{quote_curr}']"
                # Add more checks if necessary based on `balance_info` structure

            if available_balance <= 0:
                log_error(f"Insufficient available balance ({available_balance}) in {quote_curr}. Source: {source_info}"); return None
            log_info(f"Available balance ({quote_curr}): {available_balance:.{details.price_precision_digits}f} (Source: {source_info})")

            risk_amount = available_balance * self.config.risk_percentage
            quantity_raw = risk_amount / price_diff
            quantity_adj = self.exchange_handler.format_amount(self.config.symbol, quantity_raw)

            if quantity_adj <= 0: log_error("Calculated quantity is zero or less."); return None

            # Check Limits
            if details.min_amount is not None and quantity_adj < details.min_amount:
                log_error(f"Qty {quantity_adj} below min amount {details.min_amount}."); return None
            estimated_cost = quantity_adj * entry
            if details.min_cost is not None and estimated_cost < details.min_cost:
                log_error(f"Cost {estimated_cost} below min cost {details.min_cost}."); return None
            if estimated_cost > available_balance * 0.995: # Buffer
                log_error(f"Cost {estimated_cost} exceeds available balance buffer."); return None

            log_info(f"{NEON_GREEN}Calculated position size: {quantity_adj:.{details.amount_precision_digits}f} {details.base_currency}{RESET}")
            return quantity_adj

        except Exception as e:
             if not isinstance(e, RETRYABLE_EXCEPTIONS):
                  log_error(f"[PositionManager] Unexpected error calculating position size: {e}", exc_info=True)
             return None

    def attempt_emergency_close(self):
        """Attempts to close the current position with a market order."""
        if not self.state.is_active():
            log_info("[PositionManager] No active position for emergency close.")
            return
        if self.config.simulation_mode:
            log_warning("SIMULATION: Emergency close skipped.")
            # Optionally reset state here in simulation?
            # self.reset_state("Simulated Emergency Close")
            return

        log_warning(f"!!! ATTEMPTING EMERGENCY MARKET CLOSE for {self.state.status.upper()} position !!!")
        close_side = 'sell' if self.state.status == 'long' else 'buy'
        qty_to_close = self.state.quantity

        if qty_to_close is None or qty_to_close <= 0:
             log_error("Emergency close failed: Invalid quantity in state.")
             return

        # Cancel existing protection first (best effort)
        ids = self.state.get_protection_order_ids()
        if ids:
            log_warning("Cancelling existing protection orders before emergency close...")
            for oid in ids: self.exchange_handler.cancel_order(oid, self.config.symbol)

        # Place market close order
        emergency_order = self.exchange_handler.place_market_order(
            symbol=self.config.symbol,
            side=close_side,
            amount=qty_to_close,
            reduce_only=True # Ensure it's a closing order
        )

        if emergency_order and emergency_order.get('id'):
             log_warning(f"Emergency market close order placed (ID: {emergency_order['id']}). Status: {emergency_order.get('status')}.")
             # Reset state AFTER placing the close order
             self.reset_state("Emergency Market Close executed")
        else:
             log_error(f"!!! EMERGENCY CLOSE ORDER FAILED TO PLACE !!! MANUAL INTERVENTION REQUIRED !!!")
             # Do NOT reset state if close failed.

# --- Technical Analysis Functions ---
def calculate_technical_indicators(df: Optional[pd.DataFrame], config: Config) -> Optional[pd.DataFrame]:
    """Calculates indicators based on config and appends to DataFrame."""
    if df is None or df.empty: log_warning("Cannot calculate indicators: Input DataFrame is None or empty."); return None
    required_cols = ['high', 'low', 'close']
    if config.entry_volume_confirmation_enabled: required_cols.append('volume')
    if not all(col in df.columns for col in required_cols):
        log_error(f"Cannot calculate indicators: Missing columns: {[c for c in required_cols if c not in df.columns]}"); return None

    log_debug("Calculating technical indicators...")
    try:
        df_ind = df.copy()
        added_cols = []

        # RSI
        df_ind.ta.rsi(length=config.rsi_length, append=True, col_names=(config.rsi_col_name,))
        added_cols.append(config.rsi_col_name)

        # Stochastic
        df_ind.ta.stoch(k=config.stoch_k, d=config.stoch_d, smooth_k=config.stoch_smooth_k, append=True,
                       col_names=(config.stoch_k_col_name, config.stoch_d_col_name))
        added_cols.extend([config.stoch_k_col_name, config.stoch_d_col_name])

        # ATR (Conditional)
        if config.needs_atr and config.atr_col_name:
            df_ind.ta.atr(length=config.atr_length, append=True, col_names=(config.atr_col_name,))
            added_cols.append(config.atr_col_name)

        # Volume MA (Conditional)
        if config.entry_volume_confirmation_enabled and config.vol_ma_col_name and 'volume' in df_ind.columns:
            min_p = max(1, config.entry_volume_ma_length // 2)
            df_ind[config.vol_ma_col_name] = df_ind['volume'].rolling(window=config.entry_volume_ma_length, min_periods=min_p).mean()
            added_cols.append(config.vol_ma_col_name)
        elif config.entry_volume_confirmation_enabled:
            log_warning("Volume MA requested but 'volume' column missing.")

        # Drop NaNs introduced by indicators
        valid_added_cols = [col for col in added_cols if col in df_ind.columns]
        if valid_added_cols:
            initial_rows = len(df_ind)
            df_ind.dropna(subset=valid_added_cols, inplace=True)
            if len(df_ind) < initial_rows: log_debug(f"Dropped {initial_rows - len(df_ind)} rows due to NaN indicators.")
        else: log_warning("No valid indicator columns generated."); return None

        if df_ind.empty: log_warning("DataFrame empty after dropping NaN indicators."); return None
        log_debug(f"Indicator calculation complete. Rows: {len(df_ind)}")
        return df_ind

    except Exception as e:
        log_error(f"Error calculating indicators: {e}", exc_info=True); return None


def identify_potential_order_block(df: pd.DataFrame, config: Config, market_details: MarketDetails) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Identifies recent potential bullish/bearish order blocks."""
    bullish_ob, bearish_ob = None, None
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    price_prec = market_details.price_precision_digits

    if df is None or df.empty or not all(c in df.columns for c in required_cols):
        log_debug("OB Detection: Invalid DataFrame."); return None, None
    if len(df) < config.ob_lookback + 2:
        log_debug(f"OB Detection: Not enough data ({len(df)} rows) for lookback + 2."); return None, None

    try:
        df_completed = df.iloc[:-1]
        avg_volume = 0.0
        min_vol_periods = max(1, config.ob_lookback // 2)
        if len(df_completed) >= min_vol_periods:
             rolling_volume = df_completed['volume'].rolling(window=config.ob_lookback, min_periods=min_vol_periods).mean()
             if not rolling_volume.empty and pd.notna(rolling_volume.iloc[-1]): avg_volume = rolling_volume.iloc[-1]
        if avg_volume <= 0 and len(df_completed) > 0 and pd.notna(df_completed['volume'].mean()):
             avg_volume = df_completed['volume'].mean()

        volume_threshold = avg_volume * config.ob_volume_threshold_multiplier if avg_volume > 0 else float('inf')
        log_debug(f"OB Analysis | Lookback={config.ob_lookback}, AvgVol={avg_volume:.2f}, VolThresh={volume_threshold:.2f}")

        search_depth = min(len(df) - 2, config.ob_lookback * 3)
        search_start_index = len(df) - 2
        search_end_index = max(0, search_start_index - search_depth)

        for i in range(search_start_index, search_end_index -1, -1):
            if i < 1: break
            try:
                imbalance_candle = df.iloc[i]
                ob_candidate_candle = df.iloc[i-1]
            except IndexError: continue # Should not happen with bounds

            if imbalance_candle.isnull().any() or ob_candidate_candle.isnull().any(): continue

            is_high_volume_imb = imbalance_candle['volume'] > volume_threshold
            is_bullish_imb = imbalance_candle['close'] > imbalance_candle['open'] and imbalance_candle['close'] > ob_candidate_candle['high']
            is_bearish_imb = imbalance_candle['close'] < imbalance_candle['open'] and imbalance_candle['close'] < ob_candidate_candle['low']
            ob_is_bearish = ob_candidate_candle['close'] < ob_candidate_candle['open']
            ob_is_bullish = ob_candidate_candle['close'] > ob_candidate_candle['open']
            imb_sweeps_ob_low = imbalance_candle['low'] < ob_candidate_candle['low']
            imb_sweeps_ob_high = imbalance_candle['high'] > ob_candidate_candle['high']

            # Check Bullish OB (Bearish OB candle + Bullish Imbalance)
            if not bullish_ob and ob_is_bearish and is_bullish_imb and is_high_volume_imb and imb_sweeps_ob_low:
                bullish_ob = {'high': ob_candidate_candle['high'], 'low': ob_candidate_candle['low'], 'time': ob_candidate_candle.name, 'type': 'bullish'}
                log_debug(f"Potential Bullish OB @ {bullish_ob['time'].strftime('%H:%M')} (L:{bullish_ob['low']:.{price_prec}f}, H:{bullish_ob['high']:.{price_prec}f})")
                if bearish_ob: break # Found both

            # Check Bearish OB (Bullish OB candle + Bearish Imbalance)
            elif not bearish_ob and ob_is_bullish and is_bearish_imb and is_high_volume_imb and imb_sweeps_ob_high:
                 bearish_ob = {'high': ob_candidate_candle['high'], 'low': ob_candidate_candle['low'], 'time': ob_candidate_candle.name, 'type': 'bearish'}
                 log_debug(f"Potential Bearish OB @ {bearish_ob['time'].strftime('%H:%M')} (L:{bearish_ob['low']:.{price_prec}f}, H:{bearish_ob['high']:.{price_prec}f})")
                 if bullish_ob: break # Found both

        return bullish_ob, bearish_ob
    except Exception as e:
        log_error(f"Error identifying order blocks: {e}", exc_info=True)
        return None, None

def display_market_stats(current_price: float, rsi: float, stoch_k: float, stoch_d: float, atr: Optional[float], details: MarketDetails) -> None:
    """Displays key market indicators."""
    price_prec = details.price_precision_digits
    print(f"{NEON_PINK}--- Market Stats ---{RESET}")
    print(f"{NEON_GREEN}Price:{RESET}  {current_price:.{price_prec}f}")
    print(f"{NEON_CYAN}RSI:{RESET}    {rsi:.2f}")
    print(f"{NEON_YELLOW}StochK:{RESET} {stoch_k:.2f}")
    print(f"{NEON_YELLOW}StochD:{RESET} {stoch_d:.2f}")
    if atr is not None:
        print(f"{NEON_BLUE}ATR:{RESET}    {atr:.{price_prec}f}")
    print(f"{NEON_PINK}--------------------{RESET}")

def display_order_blocks(bullish_ob: Optional[Dict[str, Any]], bearish_ob: Optional[Dict[str, Any]], details: MarketDetails) -> None:
    """Displays identified potential order blocks."""
    price_prec = details.price_precision_digits
    found = False
    if bullish_ob and isinstance(bullish_ob.get('time'), pd.Timestamp):
        print(f"{NEON_GREEN}Bullish OB:{RESET} {bullish_ob['time'].strftime('%H:%M')} | Low: {bullish_ob['low']:.{price_prec}f} | High: {bullish_ob['high']:.{price_prec}f}")
        found = True
    if bearish_ob and isinstance(bearish_ob.get('time'), pd.Timestamp):
        print(f"{NEON_RED}Bearish OB:{RESET} {bearish_ob['time'].strftime('%H:%M')} | Low: {bearish_ob['low']:.{price_prec}f} | High: {bearish_ob['high']:.{price_prec}f}")
        found = True
    if not found:
        print(f"{NEON_BLUE}Order Blocks: None detected recently.{RESET}")

def display_signal(signal_type: str, direction: str, reason: str) -> None:
    color = NEON_GREEN if direction.lower() == 'long' else NEON_RED if direction.lower() == 'short' else NEON_YELLOW
    print(f"{color}{Style.BRIGHT}*** {signal_type.upper()} {direction.upper()} SIGNAL ***{RESET}\n   Reason: {reason}")

# --- Trading Bot Class ---
class TradingBot:
    """The main trading bot logic."""

    def __init__(self):
        print_neon_header()
        self.config = load_and_validate_config()
        self.exchange_handler = ExchangeHandler(self.config)
        self.position_manager = PositionManager(self.config, self.exchange_handler)
        self._validate_live_mode() # Ask for confirmation if live

    def _validate_live_mode(self):
        """Checks for simulation mode and confirms if live."""
        if self.config.simulation_mode:
            log_warning("###########################")
            log_warning("# SIMULATION MODE IS ACTIVE #")
            log_warning("###########################")
        else:
            log_warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log_warning("!!! LIVE TRADING MODE IS ACTIVE !!!")
            log_warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            try:
                if sys.stdin.isatty():
                    confirm = input(f"{NEON_RED}>>> Type 'LIVE' to confirm, Enter/Ctrl+C to exit: {RESET}")
                    if confirm.strip().upper() != "LIVE":
                        log_info("Live trading not confirmed. Exiting.")
                        sys.exit(0)
                    log_info("Live trading confirmed.")
                else:
                    log_warning("Non-interactive mode. Assuming live confirmation. Pausing 5s...")
                    time.sleep(5)
            except EOFError:
                log_error("Cannot get confirmation in non-interactive environment. Exiting.")
                sys.exit(1)

    def _log_config_summary(self):
        """Logs a summary of the trading configuration."""
        cfg = self.config
        details = self.exchange_handler.get_market_details(cfg.symbol)
        log_info("-" * 70)
        log_info(f"Trading Configuration:")
        log_info(f"  Exchange: {self.exchange_handler.name}, Symbol: {cfg.symbol}, TF: {cfg.timeframe}")
        log_info(f"  Market Type: {'Contract' if details.is_contract_market else 'Spot'}")
        log_info(f"  Risk/Trade: {cfg.risk_percentage*100:.2f}%")
        log_info(f"  Simulation Mode: {cfg.simulation_mode}")
        log_info(f"  Cycle Interval: {cfg.sleep_interval_seconds}s, Data Limit: {cfg.data_limit}")
        log_info(f"  SL/TP Mode: {'ATR Based' if cfg.enable_atr_sl_tp else 'Fixed Percentage'}")
        if cfg.enable_atr_sl_tp: log_info(f"    ATR SL: {cfg.atr_sl_multiplier}x, ATR TP: {cfg.atr_tp_multiplier}x (ATR Len: {cfg.atr_length})")
        else: log_info(f"    Fixed SL: {cfg.stop_loss_percentage*100:.2f}%, Fixed TP: {cfg.take_profit_percentage*100:.2f}%")
        log_info(f"  Trailing Stop: {'Enabled' if cfg.enable_trailing_stop else 'Disabled'}")
        if cfg.enable_trailing_stop: log_info(f"    TSL Activate: {cfg.trailing_stop_activation_atr_multiplier}x ATR, Trail: {cfg.trailing_stop_atr_multiplier}x ATR")
        log_info(f"  OB Vol Multiplier: {cfg.ob_volume_threshold_multiplier}x (Lookback: {cfg.ob_lookback})")
        log_info(f"  Entry Volume Confirm: {'Enabled' if cfg.entry_volume_confirmation_enabled else 'Disabled'}")
        if cfg.entry_volume_confirmation_enabled: log_info(f"    Vol MA Len: {cfg.entry_volume_ma_length}, Vol Multiplier: {cfg.entry_volume_multiplier}x")
        log_info(f"  Retry Settings: Max={cfg.retry_max_retries}, Delay={cfg.retry_initial_delay}s, Backoff={cfg.retry_backoff_factor}x")
        log_info("-" * 70)

    def run(self):
        """Starts the main trading loop."""
        self._log_config_summary()
        if not self.config.simulation_mode: log_warning("LIVE TRADING ACTIVE. Press Ctrl+C to stop gracefully.")
        else: log_info("SIMULATION MODE ACTIVE. Press Ctrl+C to stop.")

        while True:
            try:
                cycle_start_time = pd.Timestamp.now(tz='UTC')
                print_cycle_divider(cycle_start_time)

                # === 1. Check Position Consistency ===
                position_was_reset = self.position_manager.check_position_and_orders()
                self.position_manager.display_status() # Always show status
                if position_was_reset:
                    log_info("Position state reset. Proceeding to check new signals.")
                    # Continue to fetch fresh data

                # === 2. Fetch Data ===
                df_ohlcv = self.exchange_handler.fetch_ohlcv(
                    self.config.symbol, self.config.timeframe, self.config.data_limit
                )
                if df_ohlcv is None or df_ohlcv.empty:
                    log_warning("No valid OHLCV data fetched. Waiting.")
                    self._wait_for_next_cycle(cycle_start_time); continue

                # === 3. Calculate Indicators ===
                df_indicators = calculate_technical_indicators(df_ohlcv, self.config)
                if df_indicators is None or df_indicators.empty:
                    log_warning("Indicator calculation failed. Waiting.")
                    self._wait_for_next_cycle(cycle_start_time); continue

                # === 4. Extract Latest Data & Market Details ===
                market_details = self.exchange_handler.get_market_details(self.config.symbol)
                if not market_details: # Should have been loaded on init, but check
                    log_error("Market details unavailable. Cannot proceed this cycle.");
                    self._wait_for_next_cycle(cycle_start_time); continue

                try:
                    latest_candle = df_indicators.iloc[-1]
                    latest_timestamp = latest_candle.name
                    current_price = float(latest_candle['close'])
                    current_high = float(latest_candle['high'])
                    current_low = float(latest_candle['low'])
                    current_volume = float(latest_candle['volume']) if 'volume' in latest_candle else None

                    latest_rsi = float(latest_candle[self.config.rsi_col_name])
                    latest_stoch_k = float(latest_candle[self.config.stoch_k_col_name])
                    latest_stoch_d = float(latest_candle[self.config.stoch_d_col_name])

                    latest_atr: Optional[float] = None
                    if self.config.needs_atr and self.config.atr_col_name and self.config.atr_col_name in latest_candle and pd.notna(latest_candle[self.config.atr_col_name]):
                        atr_val = float(latest_candle[self.config.atr_col_name])
                        if atr_val > 0: latest_atr = atr_val

                    latest_vol_ma: Optional[float] = None
                    if self.config.entry_volume_confirmation_enabled and self.config.vol_ma_col_name and self.config.vol_ma_col_name in latest_candle and pd.notna(latest_candle[self.config.vol_ma_col_name]):
                         vol_ma_val = float(latest_candle[self.config.vol_ma_col_name])
                         if vol_ma_val > 0: latest_vol_ma = vol_ma_val

                    if any(math.isnan(v) for v in [current_price, latest_rsi, latest_stoch_k, latest_stoch_d]):
                        raise ValueError("NaN in essential indicator data.")

                except (KeyError, ValueError, TypeError, IndexError) as e:
                     log_error(f"Error extracting latest data at {latest_timestamp}: {e}. Check columns/calculation.", exc_info=False)
                     self._wait_for_next_cycle(cycle_start_time); continue

                if self.config.needs_atr and latest_atr is None: log_warning(f"ATR needed but invalid/missing at {latest_timestamp}.")
                if self.config.entry_volume_confirmation_enabled and latest_vol_ma is None: log_debug(f"Vol MA needed but invalid/missing at {latest_timestamp}.")

                display_market_stats(current_price, latest_rsi, latest_stoch_k, latest_stoch_d, latest_atr, market_details)

                # === 5. Identify Order Blocks ===
                bullish_ob, bearish_ob = identify_potential_order_block(df_indicators, self.config, market_details)
                display_order_blocks(bullish_ob, bearish_ob, market_details)

                # === 6. Trading Logic ===
                if self.position_manager.state.is_active():
                    # --- Monitor Position & Trail Stop ---
                    log_info(f"Monitoring active {self.position_manager.state.status.upper()} position...")
                    self.position_manager.update_trailing_stop(current_price, latest_atr)
                else:
                    # --- Check for New Entry ---
                    log_info("Checking for entry signals...")
                    self._check_and_execute_entry(
                        current_price=current_price, latest_rsi=latest_rsi,
                        latest_stoch_k=latest_stoch_k, latest_stoch_d=latest_stoch_d,
                        latest_atr=latest_atr, current_volume=current_volume,
                        latest_vol_ma=latest_vol_ma, bullish_ob=bullish_ob, bearish_ob=bearish_ob,
                        market_details=market_details
                    )

                # === 7. Wait for Next Cycle ===
                self._wait_for_next_cycle(cycle_start_time)

            # --- Graceful Shutdown ---
            except KeyboardInterrupt:
                log_info("Shutdown signal (Ctrl+C) received.")
                self.shutdown()
                break
            # --- Loop Error Handling ---
            except (ccxt.RateLimitExceeded, *RETRYABLE_EXCEPTIONS) as e:
                 log_error(f"Recoverable CCXT Error: {type(e).__name__}: {e}. Waiting...", exc_info=False)
                 wait_time = self.config.sleep_interval_seconds + (60 if isinstance(e, ccxt.RateLimitExceeded) else 15)
                 neon_sleep_timer(wait_time)
            except ccxt.AuthenticationError as e:
                 log_error(f"!!! CRITICAL AUTH ERROR: {e} !!! Check credentials. Stopping.", exc_info=False)
                 self.shutdown(save_state=True) # Try to save before stopping
                 break
            except ccxt.ExchangeError as e:
                 log_error(f"Unhandled CCXT Exchange Error: {type(e).__name__}: {e}. Waiting...", exc_info=True)
                 neon_sleep_timer(self.config.sleep_interval_seconds + 30)
            except Exception as e:
                log_error(f"!!! CRITICAL UNEXPECTED LOOP ERROR: {type(e).__name__}: {e} !!!", exc_info=True)
                log_info("Attempting to save state before waiting...")
                try: self.position_manager.save_state()
                except Exception as save_err: log_error(f"Failed to save state during error handling: {save_err}")
                log_warning("Waiting 60s before attempting to continue...")
                neon_sleep_timer(60)

        log_info("Bot run loop finished.")

    def _wait_for_next_cycle(self, cycle_start_time: pd.Timestamp):
        """Calculates and waits for the remainder of the cycle interval."""
        cycle_end_time = pd.Timestamp.now(tz='UTC')
        elapsed_seconds = (cycle_end_time - cycle_start_time).total_seconds()
        wait_time = max(0, self.config.sleep_interval_seconds - elapsed_seconds)
        log_info(f"Cycle completed in {elapsed_seconds:.2f}s.")
        if wait_time > 0:
            log_info(f"Waiting {wait_time:.1f}s until next cycle...")
            neon_sleep_timer(int(round(wait_time)))
        else:
            log_info("Cycle took longer than interval, starting next immediately.")

    def _check_and_execute_entry(self, current_price: float, latest_rsi: float,
                                latest_stoch_k: float, latest_stoch_d: float,
                                latest_atr: Optional[float], current_volume: Optional[float],
                                latest_vol_ma: Optional[float], bullish_ob: Optional[Dict],
                                bearish_ob: Optional[Dict], market_details: MarketDetails):
        """Checks entry conditions and executes trade sequence if met."""

        # --- Evaluate Entry Conditions ---
        cfg = self.config
        volume_ok = True
        if cfg.entry_volume_confirmation_enabled:
            if current_volume is not None and latest_vol_ma is not None:
                volume_ok = current_volume > latest_vol_ma * cfg.entry_volume_multiplier
                log_debug(f"Vol Check: Vol={current_volume:.2f}, MA={latest_vol_ma:.2f}, Thresh={latest_vol_ma * cfg.entry_volume_multiplier:.2f} -> {volume_ok}")
            else: volume_ok = False; log_debug("Vol Check: FAILED (Missing data)")

        base_long = latest_rsi < cfg.rsi_oversold and latest_stoch_k < cfg.stoch_oversold and latest_stoch_d < cfg.stoch_oversold
        base_short = latest_rsi > cfg.rsi_overbought and latest_stoch_k > cfg.stoch_overbought and latest_stoch_d > cfg.stoch_overbought

        ob_ok = False
        ob_reason = ""
        entry_ob = None
        price_prec = market_details.price_precision_digits

        if base_long and bullish_ob and (bullish_ob['low'] <= current_price <= bullish_ob['high']):
            ob_ok = True; entry_ob = bullish_ob
            ob_reason = f"Price ({current_price:.{price_prec}f}) in Bullish OB ({bullish_ob['low']:.{price_prec}f}-{bullish_ob['high']:.{price_prec}f})"
        elif base_short and bearish_ob and (bearish_ob['low'] <= current_price <= bearish_ob['high']):
            ob_ok = True; entry_ob = bearish_ob
            ob_reason = f"Price ({current_price:.{price_prec}f}) in Bearish OB ({bearish_ob['low']:.{price_prec}f}-{bearish_ob['high']:.{price_prec}f})"

        # --- Determine Entry Side & Reason ---
        entry_side: Optional[str] = None
        entry_reason: str = ""
        if base_long and ob_ok and volume_ok and entry_ob:
            entry_side = 'long'
            entry_reason = f"RSI({latest_rsi:.1f})<OS({cfg.rsi_oversold}), Stoch({latest_stoch_k:.1f},{latest_stoch_d:.1f})<OS({cfg.stoch_oversold}), {ob_reason}"
            if cfg.entry_volume_confirmation_enabled: entry_reason += ", Vol Confirmed"
        elif base_short and ob_ok and volume_ok and entry_ob:
            entry_side = 'short'
            entry_reason = f"RSI({latest_rsi:.1f})>OB({cfg.rsi_overbought}), Stoch({latest_stoch_k:.1f},{latest_stoch_d:.1f})>OB({cfg.stoch_overbought}), {ob_reason}"
            if cfg.entry_volume_confirmation_enabled: entry_reason += ", Vol Confirmed"
        elif base_long or base_short:
            block_reasons = []
            if not ob_ok: block_reasons.append("OB Confirm Failed")
            if not volume_ok: block_reasons.append("Vol Confirm Failed")
            log_info(f"Entry blocked for {'LONG' if base_long else 'SHORT'}. Reason(s): {', '.join(block_reasons)}")

        # --- Execute Entry Sequence ---
        if not entry_side or not entry_ob:
             log_info("Entry conditions not met."); return

        display_signal("Entry", entry_side, entry_reason)

        # 1. Calculate Initial SL/TP
        initial_sl: Optional[float] = None; initial_tp: Optional[float] = None
        if cfg.enable_atr_sl_tp:
            if latest_atr:
                sl_dist = latest_atr * cfg.atr_sl_multiplier
                tp_dist = latest_atr * cfg.atr_tp_multiplier
                initial_sl = current_price - sl_dist if entry_side == 'long' else current_price + sl_dist
                initial_tp = current_price + tp_dist if entry_side == 'long' else current_price - tp_dist
            else: log_error("Cannot calc ATR SL/TP: Invalid ATR. Aborting entry."); return
        else: # Fixed %
            sl_mult = 1.0 - cfg.stop_loss_percentage if entry_side == 'long' else 1.0 + cfg.stop_loss_percentage
            tp_mult = 1.0 + cfg.take_profit_percentage if entry_side == 'long' else 1.0 - cfg.take_profit_percentage
            initial_sl = current_price * sl_mult
            initial_tp = current_price * tp_mult
        if initial_sl is None or initial_tp is None: log_error("Failed initial SL/TP calc. Abort."); return
        log_info(f"Initial Calc SL: {initial_sl:.{price_prec}f}, TP: {initial_tp:.{price_prec}f}")

        # 2. Refine SL with OB
        sl_buffer = market_details.min_tick * 5 # Example buffer
        refined_sl = initial_sl
        if entry_side == 'long':
            potential_sl = entry_ob['low'] - sl_buffer
            if potential_sl < initial_sl and potential_sl < current_price:
                refined_sl = potential_sl; log_info(f"Refined SL with Bullish OB: {refined_sl:.{price_prec}f}")
        else: # Short
            potential_sl = entry_ob['high'] + sl_buffer
            if potential_sl > initial_sl and potential_sl > current_price:
                refined_sl = potential_sl; log_info(f"Refined SL with Bearish OB: {refined_sl:.{price_prec}f}")
        final_sl = refined_sl
        final_tp = initial_tp # TP not refined by OB

        # 3. Final Format & Validate SL/TP
        try:
            final_sl_fmt = self.exchange_handler.format_price(cfg.symbol, final_sl)
            final_tp_fmt = self.exchange_handler.format_price(cfg.symbol, final_tp)
            if (entry_side == 'long' and (final_sl_fmt >= current_price or final_tp_fmt <= current_price)) or \
               (entry_side == 'short' and (final_sl_fmt <= current_price or final_tp_fmt >= current_price)):
                raise ValueError(f"Invalid final SL/TP logic: SL={final_sl_fmt}, TP={final_tp_fmt}, Entry={current_price}")
        except Exception as e: log_error(f"Final SL/TP format/validation failed: {e}. Abort."); return
        log_info(f"Final Target Prices: SL={final_sl_fmt:.{price_prec}f}, TP={final_tp_fmt:.{price_prec}f}")

        # 4. Calculate Position Size
        entry_qty = self.position_manager.calculate_position_size(current_price, final_sl_fmt)
        if entry_qty is None or entry_qty <= 0: log_error("Position size calc failed. Abort."); return

        # 5. Place Entry Market Order
        market_side = 'buy' if entry_side == 'long' else 'sell'
        entry_order = self.exchange_handler.place_market_order(cfg.symbol, market_side, entry_qty, reduce_only=False)

        # 6. Process Entry & Place Protection
        if entry_order and entry_order.get('id'):
            entry_id = entry_order['id']
            entry_status = entry_order.get('status')
            actual_entry_px = entry_order.get('average', entry_order.get('price'))
            actual_filled_qty = entry_order.get('filled')

            if actual_entry_px is None or actual_entry_px <= 0: actual_entry_px = current_price # Fallback
            else: actual_entry_px = float(actual_entry_px)
            if actual_filled_qty is None or actual_filled_qty <= 0: actual_filled_qty = entry_qty # Fallback
            else: actual_filled_qty = float(actual_filled_qty)

            log_info(f"Entry order {entry_id} ({entry_status}). Actual Px: ~{actual_entry_px:.{price_prec}f}, Qty: {actual_filled_qty:.{market_details.amount_precision_digits}f}")

            sl_id, tp_id, oco_id = self.exchange_handler.place_protection_orders(
                symbol=cfg.symbol, pos_side=entry_side, qty=actual_filled_qty,
                sl_pr=final_sl_fmt, tp_pr=final_tp_fmt
            )

            # 7. Update State (ONLY if protection placed)
            if oco_id or sl_id:
                self.position_manager.update_entry(
                    side=entry_side, entry_price=actual_entry_px, quantity=actual_filled_qty,
                    entry_order_id=entry_id, sl_price=final_sl_fmt, tp_price=final_tp_fmt,
                    sl_order_id=sl_id, tp_order_id=tp_id, oco_order_id=oco_id
                )
                log_info(f"Successfully opened {entry_side.upper()} position and saved state.")
                if not oco_id and not tp_id and sl_id: log_warning("Entry & SL OK, but separate TP failed.")
            else:
                # --- CRITICAL: Entry filled, protection failed ---
                log_error(f"!!! CRITICAL ENTRY FAILURE !!! Entry {entry_id} filled, but FAILED to place protection.")
                log_error(f"  !!! POSITION ACTIVE BUT UNPROTECTED !!!")
                self.position_manager.attempt_emergency_close() # Attempt market close

        else:
             log_error(f"Entry market order failed or no result ({entry_order}). No position taken.")


    def shutdown(self, save_state: bool = True):
        """Handles graceful shutdown, saving state and cancelling orders."""
        log_info("Initiating shutdown sequence...")
        if save_state:
            try:
                self.position_manager.save_state()
            except Exception as e:
                log_error(f"Error saving state during shutdown: {e}")

        # Cancel open orders only in live mode
        if not self.config.simulation_mode:
            log_warning(f"Attempting to cancel open orders for {self.config.symbol}...")
            try:
                open_orders = self.exchange_handler.fetch_open_orders(self.config.symbol) # Retries handled inside
                if open_orders:
                    log_info(f"Found {len(open_orders)} open orders. Cancelling...")
                    cancelled, failed = 0, 0
                    for order in open_orders:
                        oid = order.get('id')
                        if not oid: continue
                        log_info(f"Cancelling Order ID: {oid}, Side: {order.get('side')}, Type: {order.get('type')}")
                        if self.exchange_handler.cancel_order(oid, self.config.symbol): cancelled += 1
                        else: failed += 1
                        time.sleep(0.3)
                    log_info(f"Exit cancel summary: {cancelled} cancelled/closed, {failed} failed.")
                    if failed > 0: log_error("Some orders failed cancellation. Manual check needed.")
                else: log_info("No open orders found to cancel.")
            except Exception as e:
                log_error(f"Error cancelling orders on exit: {e}", exc_info=True)
                log_error("Manual check of open orders recommended.")
        else:
            log_info("Simulation mode: No real orders to cancel.")

        print_shutdown_message()
        log_info("Bot shutdown sequence complete.")

# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except SystemExit as e:
        log_info(f"Bot exited via SystemExit (Code: {e.code}).")
    except Exception as main_exec_error:
        log_error(f"Critical error during bot initialization or top-level execution:", exc_info=True)
        traceback.print_exc() # Ensure traceback is printed
        sys.exit(1)
    finally:
        logging.shutdown()
        print("Logging shut down.")

# --- END OF REFACTORED FILE ---
```

**How to Use:**

1.  **Save:** Save this code as a new Python file (e.g., `rsitrader_enhanced.py`).
2.  **Dependencies:** Ensure you have the necessary libraries installed (`ccxt`, `pandas`, `pandas_ta`, `python-dotenv`, `colorama`).
3.  **Configuration:** Create/update your `config.json` file with all the required settings (including the new retry settings if you want to override defaults).
4.  **Environment:** Create/update your `.env` file with your exchange API credentials (e.g., `BINANCE_API_KEY=...`, `BINANCE_SECRET_KEY=...`).
5.  **Run:** Execute the script: `python rsitrader_enhanced.py`

This refactored version provides a much more organized and maintainable structure, making it easier to understand, debug, and extend in the future. Remember that the OCO logic within `ExchangeHandler.place_protection_orders` might still need specific adjustments based on the exact exchange you are using if you want reliable live OCO placement.