# rsitrader_enhanced_v1.1.py
# Enhanced version incorporating ATR SL/TP, Trailing Stops, Config File,
# Volume Confirmation, Retry Logic, improved structure, OCO orders (if supported),
# position check fallback, and general robustness improvements.

import ccxt
import os
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import json  # For config, pretty printing order details and saving state
import os.path  # For checking if state file exists
from typing import Optional, Tuple, Dict, Any, List, Callable
import functools  # For retry decorator
import sys  # For exit
import math  # For rounding, checking NaN

# --- Load Environment Variables FIRST ---
# Fix: Load dotenv *before* accessing environment variables
load_dotenv()
print("Attempting to load environment variables from .env file (for API keys)...")

# --- Colorama Initialization and Neon Palette ---
try:
    from colorama import init, Fore, Back, Style

    init(autoreset=True)  # Autoreset ensures styles don't leak

    # Define neon color palette
    NEON_GREEN: str = Fore.GREEN + Style.BRIGHT
    NEON_PINK: str = Fore.MAGENTA + Style.BRIGHT
    NEON_CYAN: str = Fore.CYAN + Style.BRIGHT
    NEON_RED: str = Fore.RED + Style.BRIGHT
    NEON_YELLOW: str = Fore.YELLOW + Style.BRIGHT
    NEON_BLUE: str = Fore.BLUE + Style.BRIGHT
    RESET: str = Style.RESET_ALL  # Although autoreset is on, good practice
    COLORAMA_AVAILABLE: bool = True
except ImportError:
    print("Warning: colorama not found. Neon styling will be disabled. Install with: pip install colorama")
    # Define dummy colors if colorama is not available
    NEON_GREEN = NEON_PINK = NEON_CYAN = NEON_RED = NEON_YELLOW = NEON_BLUE = RESET = ""
    COLORAMA_AVAILABLE = False

# --- Logging Configuration ---
log_format_base: str = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
# Configure root logger - handlers can be added later if needed (e.g., file handler)
logging.basicConfig(level=logging.INFO, format=log_format_base, datefmt="%Y-%m-%d %H:%M:%S")
logger: logging.Logger = logging.getLogger(__name__)  # Get logger for this module

# --- Neon Display Functions ---


def print_neon_header() -> None:
    """Prints a neon-styled header banner."""
    print(f"{NEON_CYAN}{'=' * 70}{RESET}")
    print(f"{NEON_PINK}{Style.BRIGHT}    Enhanced RSI/OB Trader Neon Bot - Configurable v1.1    {RESET}")
    print(f"{NEON_CYAN}{'=' * 70}{RESET}")


def display_error_box(message: str) -> None:
    """Displays an error message in a neon box."""
    box_width = 70
    print(f"{NEON_RED}{'!' * box_width}{RESET}")
    print(f"{NEON_RED}! {message:^{box_width - 4}} !{RESET}")
    print(f"{NEON_RED}{'!' * box_width}{RESET}")


def display_warning_box(message: str) -> None:
    """Displays a warning message in a neon box."""
    box_width = 70
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")
    print(f"{NEON_YELLOW}~ {message:^{box_width - 4}} ~{RESET}")
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")


# Custom logging wrappers with colorama
def log_info(msg: str) -> None:
    """Logs an INFO level message with neon green color."""
    logger.info(f"{NEON_GREEN}{msg}{RESET}")


def log_error(msg: str, exc_info: bool = False) -> None:
    """Logs an ERROR level message with a neon red box and color."""
    first_line = msg.split("\n", 1)[0]
    display_error_box(first_line)
    logger.error(f"{NEON_RED}{msg}{RESET}", exc_info=exc_info)


def log_warning(msg: str) -> None:
    """Logs a WARNING level message with a neon yellow box and color."""
    display_warning_box(msg)
    logger.warning(f"{NEON_YELLOW}{msg}{RESET}")


def log_debug(msg: str) -> None:
    """Logs a DEBUG level message with simple white color."""
    logger.debug(f"{Fore.WHITE}{msg}{RESET}")


def print_cycle_divider(timestamp: pd.Timestamp) -> None:
    """Prints a neon divider for each trading cycle."""
    box_width = 70
    print(f"\n{NEON_BLUE}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}Cycle Start: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}{RESET}")
    print(f"{NEON_BLUE}{'=' * box_width}{RESET}")


def display_position_status(position: Dict[str, Any], price_precision: int = 4, amount_precision: int = 8) -> None:
    """Displays position status with neon colors and formatted values."""
    status: Optional[str] = position.get("status")
    entry_price: Optional[float] = position.get("entry_price")
    quantity: Optional[float] = position.get("quantity")
    sl: Optional[float] = position.get("stop_loss")  # The *target* SL price
    tp: Optional[float] = position.get("take_profit")
    tsl: Optional[float] = position.get("current_trailing_sl_price")  # Active TSL price level

    entry_str: str = f"{entry_price:.{price_precision}f}" if isinstance(entry_price, (float, int)) else "N/A"
    qty_str: str = f"{quantity:.{amount_precision}f}" if isinstance(quantity, (float, int)) else "N/A"
    sl_str: str = f"{sl:.{price_precision}f}" if isinstance(sl, (float, int)) else "N/A"
    tp_str: str = f"{tp:.{price_precision}f}" if isinstance(tp, (float, int)) else "N/A"
    tsl_str: str = (
        f" | Active TSL: {tsl:.{price_precision}f}" if isinstance(tsl, (float, int)) else ""
    )  # Show TSL only if active

    if status == "long":
        color: str = NEON_GREEN
        status_text: str = "LONG"
    elif status == "short":
        color: str = NEON_RED
        status_text: str = "SHORT"
    else:
        color: str = NEON_CYAN
        status_text: str = "None"

    print(
        f"{color}Position Status: {status_text}{RESET} | Entry: {entry_str} | Qty: {qty_str} | Target SL: {sl_str} | Target TP: {tp_str}{tsl_str}"
    )
    sl_id = position.get("sl_order_id")
    tp_id = position.get("tp_order_id")
    oco_id = position.get("oco_order_id")  # Display OCO ID if used
    if oco_id:
        print(f"    OCO Order ID: {oco_id}")
    else:
        if sl_id:
            print(f"    SL Order ID: {sl_id}")
        if tp_id:
            print(f"    TP Order ID: {tp_id}")
    if not oco_id and not sl_id and not tp_id and status is not None:
        print(f"    {NEON_YELLOW}Warning: Position active but no protection order IDs tracked.{RESET}")


def display_market_stats(
    current_price: float, rsi: float, stoch_k: float, stoch_d: float, atr: Optional[float], price_precision: int
) -> None:
    """Displays market stats in a neon-styled panel."""
    print(f"{NEON_PINK}--- Market Stats ---{RESET}")
    print(f"{NEON_GREEN}Price:{RESET}  {current_price:.{price_precision}f}")
    print(f"{NEON_CYAN}RSI:{RESET}    {rsi:.2f}")
    print(f"{NEON_YELLOW}StochK:{RESET} {stoch_k:.2f}")
    print(f"{NEON_YELLOW}StochD:{RESET} {stoch_d:.2f}")
    if atr is not None:
        print(f"{NEON_BLUE}ATR:{RESET}    {atr:.{price_precision}f}")
    print(f"{NEON_PINK}--------------------{RESET}")


def display_order_blocks(
    bullish_ob: Optional[Dict[str, Any]], bearish_ob: Optional[Dict[str, Any]], price_precision: int
) -> None:
    """Displays identified order blocks with neon colors."""
    found: bool = False
    if bullish_ob and isinstance(bullish_ob.get("time"), pd.Timestamp):
        print(
            f"{NEON_GREEN}Bullish OB:{RESET} {bullish_ob['time'].strftime('%H:%M')} | Low: {bullish_ob['low']:.{price_precision}f} | High: {bullish_ob['high']:.{price_precision}f}"
        )
        found = True
    if bearish_ob and isinstance(bearish_ob.get("time"), pd.Timestamp):
        print(
            f"{NEON_RED}Bearish OB:{RESET} {bearish_ob['time'].strftime('%H:%M')} | Low: {bearish_ob['low']:.{price_precision}f} | High: {bearish_ob['high']:.{price_precision}f}"
        )
        found = True
    if not found:
        print(f"{NEON_BLUE}Order Blocks: None detected in recent data.{RESET}")


def display_signal(signal_type: str, direction: str, reason: str) -> None:
    """Displays trading signals with appropriate neon colors."""
    color: str
    if direction.lower() == "long":
        color = NEON_GREEN
    elif direction.lower() == "short":
        color = NEON_RED
    else:  # e.g., Exit, Warning
        color = NEON_YELLOW

    print(f"{color}{Style.BRIGHT}*** {signal_type.upper()} {direction.upper()} SIGNAL ***{RESET}\n   Reason: {reason}")


def neon_sleep_timer(seconds: int) -> None:
    """Displays a neon countdown timer in the console."""
    if not COLORAMA_AVAILABLE or seconds <= 0:
        if seconds > 0:
            print(f"Sleeping for {seconds} seconds...")
            time.sleep(seconds)
        return

    interval: float = 0.5
    steps: int = int(seconds / interval)
    for i in range(steps, -1, -1):
        remaining_seconds: int = max(0, int(i * interval))
        color: str = NEON_RED if remaining_seconds <= 5 and i % 2 == 0 else NEON_YELLOW
        print(f"{color}Next cycle in: {remaining_seconds} seconds... {Style.RESET_ALL}", end="\r")
        time.sleep(interval)
    print(" " * 50, end="\r")  # Clear line after countdown


def print_shutdown_message() -> None:
    """Prints a neon shutdown message."""
    box_width = 70
    print(f"\n{NEON_PINK}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}{Style.BRIGHT}{'RSI/OB Trader Bot Stopped - Goodbye!':^{box_width}}{RESET}")
    print(f"{NEON_PINK}{'=' * box_width}{RESET}")


# --- Constants ---
DEFAULT_PRICE_PRECISION: int = 4
DEFAULT_AMOUNT_PRECISION: int = 8
POSITION_STATE_FILE: str = "position_state.json"
CONFIG_FILE: str = "config.json"

# --- Retry Decorator ---
RETRYABLE_EXCEPTIONS: Tuple[type[ccxt.NetworkError], ...] = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,  # Add DDoSProtection as potentially retryable
)


def retry_api_call(max_retries: int = 3, initial_delay: float = 5.0, backoff_factor: float = 2.0) -> Callable:
    """
    Decorator factory for retrying API calls with exponential backoff.
    Handles specific RETRYABLE_EXCEPTIONS.

    Args:
        max_retries: Maximum number of retries.
        initial_delay: Initial delay in seconds.
        backoff_factor: Multiplier for the delay (e.g., 2 for exponential).

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries: int = 0
            delay: float = initial_delay
            last_exception: Optional[Exception] = None

            while retries <= max_retries:  # Use <= to allow max_retries attempts
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    retries += 1
                    last_exception = e
                    if retries > max_retries:
                        log_error(
                            f"API call '{func.__name__}' failed after {max_retries} retries. Last error: {type(e).__name__}: {e}",
                            exc_info=False,
                        )
                        raise last_exception
                    else:
                        log_warning(
                            f"API call '{func.__name__}' failed with {type(e).__name__}: {e}. Retrying in {delay:.1f}s... (Attempt {retries}/{max_retries})"
                        )
                        try:
                            neon_sleep_timer(int(round(delay)))  # Round delay for timer
                        except NameError:
                            time.sleep(delay)
                        delay *= backoff_factor
                except Exception as e:
                    log_error(
                        f"Non-retryable error in API call '{func.__name__}': {type(e).__name__}: {e}", exc_info=True
                    )
                    raise
            # Should not be reached if max_retries >= 0, but ensures an exception is raised
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"API call '{func.__name__}' failed unexpectedly.")  # Should not happen

        return wrapper

    return decorator


# --- Configuration Loading ---
def load_config(filename: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Loads configuration from JSON file with enhanced validation.
    Exits on critical errors.

    Args:
        filename: Path to the config file.

    Returns:
        Configuration dictionary.

    Raises:
        SystemExit: On critical errors (file not found, JSON error, validation fail).
    """
    log_info(f"Attempting to load configuration from '{filename}'...")
    if not os.path.exists(filename):
        log_error(f"CRITICAL: Configuration file '{filename}' not found.")
        sys.exit(1)

    try:
        with open(filename, "r") as f:
            config_data: Dict[str, Any] = json.load(f)
        log_info(f"Configuration loaded successfully from {filename}")

        # --- Validation ---
        def validate(
            key: str, expected_type: type, condition: Optional[Callable[[Any], bool]] = None, is_required: bool = True
        ):
            """Helper for config validation."""
            if key not in config_data:
                if is_required:
                    raise ValueError(f"Missing required configuration key: '{key}'")
                return  # Not required and not present, skip validation
            value = config_data[key]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Config key '{key}' has incorrect type. Expected {expected_type.__name__}, got {type(value).__name__}."
                )
            if condition and not condition(value):
                raise ValueError(f"Config key '{key}' value ({value}) failed validation condition.")

        # Required Basic Settings
        validate("exchange_id", str, lambda x: len(x) > 0)
        validate("symbol", str, lambda x: len(x) > 0)
        validate("timeframe", str, lambda x: len(x) > 0)  # Could add check against exchange.timeframes later
        validate("risk_percentage", (float, int), lambda x: 0 < x < 1)
        validate("simulation_mode", bool)
        validate("data_limit", int, lambda x: x > 20)  # Need enough data for lookbacks
        validate("sleep_interval_seconds", int, lambda x: x > 0)

        # Indicator Parameters
        validate("rsi_length", int, lambda x: x > 0)
        validate("rsi_overbought", int, lambda x: 50 < x <= 100)
        validate("rsi_oversold", int, lambda x: 0 <= x < 50)
        validate("stoch_k", int, lambda x: x > 0)
        validate("stoch_d", int, lambda x: x > 0)
        validate("stoch_smooth_k", int, lambda x: x > 0)
        validate("stoch_overbought", int, lambda x: 50 < x <= 100)
        validate("stoch_oversold", int, lambda x: 0 <= x < 50)

        # Order Block Parameters
        validate("ob_volume_threshold_multiplier", (float, int), lambda x: x > 0)
        validate("ob_lookback", int, lambda x: x > 0)

        # Volume Confirmation (Optional based on enable flag)
        validate("entry_volume_confirmation_enabled", bool)
        if config_data.get("entry_volume_confirmation_enabled"):
            validate("entry_volume_ma_length", int, lambda x: x > 0)
            validate("entry_volume_multiplier", (float, int), lambda x: x > 0)

        # SL/TP Settings (Conditional)
        validate("enable_atr_sl_tp", bool)
        validate("enable_trailing_stop", bool)
        needs_atr = config_data.get("enable_atr_sl_tp") or config_data.get("enable_trailing_stop")
        if needs_atr:
            validate("atr_length", int, lambda x: x > 0)

        if config_data.get("enable_atr_sl_tp"):
            validate("atr_sl_multiplier", (float, int), lambda x: x > 0)
            validate("atr_tp_multiplier", (float, int), lambda x: x > 0)
        else:  # Fixed % SL/TP used if ATR SL/TP disabled
            validate("stop_loss_percentage", (float, int), lambda x: x > 0)
            validate("take_profit_percentage", (float, int), lambda x: x > 0)

        if config_data.get("enable_trailing_stop"):
            validate("trailing_stop_atr_multiplier", (float, int), lambda x: x > 0)
            validate(
                "trailing_stop_activation_atr_multiplier", (float, int), lambda x: x >= 0
            )  # Activation can be zero

        # Retry Settings (Optional, with defaults)
        validate("retry_max_retries", int, lambda x: x >= 0, is_required=False)
        validate("retry_initial_delay", (float, int), lambda x: x > 0, is_required=False)
        validate("retry_backoff_factor", (float, int), lambda x: x >= 1, is_required=False)

        log_info("Configuration validation passed.")
        return config_data

    except json.JSONDecodeError as e:
        log_error(f"CRITICAL: Error decoding JSON from configuration file '{filename}': {e}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
        log_error(f"CRITICAL: Configuration validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"CRITICAL: An unexpected error occurred loading configuration: {e}", exc_info=True)
        sys.exit(1)


# Load config early
config: Dict[str, Any] = load_config()

# Create the retry decorator instance using config values or defaults
api_retry_decorator: Callable = retry_api_call(
    max_retries=config.get("retry_max_retries", 3),
    initial_delay=config.get("retry_initial_delay", 5.0),
    backoff_factor=config.get("retry_backoff_factor", 2.0),
)

# --- Environment & Exchange Setup ---
print_neon_header()  # Show header after config load

exchange_id: str = config["exchange_id"].lower()
api_key_env_var: str = f"{exchange_id.upper()}_API_KEY"
secret_key_env_var: str = f"{exchange_id.upper()}_SECRET_KEY"
passphrase_env_var: str = f"{exchange_id.upper()}_PASSPHRASE"

api_key: Optional[str] = os.getenv(api_key_env_var)
secret: Optional[str] = os.getenv(secret_key_env_var)
passphrase: Optional[str] = os.getenv(passphrase_env_var)

if not api_key or not secret:
    log_error(
        f"CRITICAL: API Key ('{api_key_env_var}') or Secret ('{secret_key_env_var}') not found in environment/.env file."
    )
    sys.exit(1)

log_info(f"Attempting to connect to exchange: {exchange_id}")
exchange: ccxt.Exchange
try:
    exchange_class: type[ccxt.Exchange] = getattr(ccxt, exchange_id)
    exchange_config: Dict[str, Any] = {
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap" if ":" in config["symbol"] or "SWAP" in config["symbol"].upper() else "spot",
            "adjustForTimeDifference": True,
        },
    }
    if passphrase:
        log_info("Passphrase detected, adding to exchange configuration.")
        exchange_config["password"] = passphrase

    exchange = exchange_class(exchange_config)

    @api_retry_decorator
    def load_markets_with_retry(exch_instance: ccxt.Exchange) -> None:
        """Loads exchange markets with retry logic."""
        log_info("Loading markets...")
        exch_instance.load_markets(reload=True)  # Force reload to get latest info
        log_info("Markets loaded.")

    load_markets_with_retry(exchange)

    log_info(
        f"Successfully connected to {exchange_id}. Markets loaded ({len(exchange.markets)} symbols found). Default type: {exchange.options.get('defaultType')}"
    )
    # Log OCO capability
    if exchange.has.get("oco"):
        log_info(f"Exchange {exchange_id} supports OCO orders.")
    else:
        log_warning(
            f"Exchange {exchange_id} does NOT appear to support OCO orders via ccxt `has['oco']` flag. SL/TP will be placed separately."
        )

except ccxt.AuthenticationError as e:
    log_error(f"Authentication failed connecting to {exchange_id}. Check API Key/Secret/Passphrase. Error: {e}")
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    log_error(f"Exchange {exchange_id} is not available or timed out. Error: {e}")
    sys.exit(1)
except AttributeError:
    log_error(f"Exchange ID '{exchange_id}' not found in ccxt library.")
    sys.exit(1)
except Exception as e:
    if not isinstance(e, RETRYABLE_EXCEPTIONS):
        log_error(f"An unexpected error occurred during exchange initialization or market loading: {e}", exc_info=True)
    sys.exit(1)


# --- Trading Parameters (Derived from validated config) ---
symbol: str = config["symbol"].strip().upper()
timeframe: str = config["timeframe"]

# Validate symbol and timeframe against loaded markets
market_info: Optional[Dict[str, Any]] = None
price_precision_digits: int = DEFAULT_PRICE_PRECISION
amount_precision_digits: int = DEFAULT_AMOUNT_PRECISION
min_tick: float = 1 / (10**DEFAULT_PRICE_PRECISION)
min_amount: Optional[float] = None

try:
    if symbol not in exchange.markets:
        log_error(f"Symbol '{symbol}' from config not found or not supported on {exchange_id}.")
        # Suggest similar symbols if possible? (complex)
        sys.exit(1)

    market_info = exchange.markets[symbol]

    # Validate timeframe
    if timeframe not in exchange.timeframes:
        available_tf = list(exchange.timeframes.keys())
        log_error(f"Timeframe '{timeframe}' not supported by {exchange_id} for {symbol}. Available: {available_tf}")
        sys.exit(1)

    log_info(f"Using trading symbol: {symbol} on timeframe: {timeframe}")

    # Get precision digits robustly
    price_prec = market_info.get("precision", {}).get("price")
    amount_prec = market_info.get("precision", {}).get("amount")
    if price_prec is not None:
        price_precision_digits = int(
            exchange.decimal_to_precision(price_prec, ccxt.ROUND, counting_mode=exchange.precisionMode)
        )
    if amount_prec is not None:
        amount_precision_digits = int(
            exchange.decimal_to_precision(amount_prec, ccxt.ROUND, counting_mode=exchange.precisionMode)
        )

    # Get minimum tick size (price increment)
    if price_prec is not None:
        try:
            min_tick = (
                float(price_prec)
                if isinstance(price_prec, (float, int))
                else float(exchange.price_to_precision(symbol, 10 ** (-price_precision_digits)))
            )
            # Handle potential zero precision
            if min_tick <= 0:
                min_tick = 1 / (10**price_precision_digits) if price_precision_digits >= 0 else 0.01
        except Exception:
            min_tick = 1 / (10**price_precision_digits) if price_precision_digits >= 0 else 0.01

    # Get minimum order amount
    min_amount = market_info.get("limits", {}).get("amount", {}).get("min")

    log_info(
        f"Symbol Precision | Price: {price_precision_digits} decimals (Min Tick: {min_tick:.{price_precision_digits + 2}f}), Amount: {amount_precision_digits} decimals"
    )
    if min_amount:
        log_info(f"Minimum Order Amount: {min_amount} {market_info.get('base', '')}")

except Exception as e:
    log_error(f"An error occurred while validating symbol/timeframe or getting precision: {e}", exc_info=True)
    sys.exit(1)

# Assign validated config values to variables
rsi_length: int = config["rsi_length"]
rsi_overbought: int = config["rsi_overbought"]
rsi_oversold: int = config["rsi_oversold"]
stoch_k: int = config["stoch_k"]
stoch_d: int = config["stoch_d"]
stoch_smooth_k: int = config["stoch_smooth_k"]
stoch_overbought: int = config["stoch_overbought"]
stoch_oversold: int = config["stoch_oversold"]
data_limit: int = config["data_limit"]
sleep_interval_seconds: int = config["sleep_interval_seconds"]
risk_percentage: float = float(config["risk_percentage"])

enable_atr_sl_tp: bool = config["enable_atr_sl_tp"]
enable_trailing_stop: bool = config["enable_trailing_stop"]
atr_length: int = config.get("atr_length", 14)  # Default if not present but needed

needs_atr: bool = enable_atr_sl_tp or enable_trailing_stop

atr_sl_multiplier: float = 0.0
atr_tp_multiplier: float = 0.0
stop_loss_percentage: float = 0.0
take_profit_percentage: float = 0.0

if enable_atr_sl_tp:
    atr_sl_multiplier = float(config["atr_sl_multiplier"])
    atr_tp_multiplier = float(config["atr_tp_multiplier"])
    log_info(f"Using ATR-based Stop Loss ({atr_sl_multiplier}x ATR) and Take Profit ({atr_tp_multiplier}x ATR).")
else:
    stop_loss_percentage = float(config["stop_loss_percentage"])
    take_profit_percentage = float(config["take_profit_percentage"])
    log_info(
        f"Using Fixed Percentage Stop Loss ({stop_loss_percentage * 100:.1f}%) and Take Profit ({take_profit_percentage * 100:.1f}%)."
    )

trailing_stop_atr_multiplier: float = 0.0
trailing_stop_activation_atr_multiplier: float = 0.0

if enable_trailing_stop:
    trailing_stop_atr_multiplier = float(config["trailing_stop_atr_multiplier"])
    trailing_stop_activation_atr_multiplier = float(config["trailing_stop_activation_atr_multiplier"])
    log_info(
        f"Trailing Stop Loss is ENABLED (Activate @ {trailing_stop_activation_atr_multiplier}x ATR profit, Trail @ {trailing_stop_atr_multiplier}x ATR)."
    )
else:
    log_info("Trailing Stop Loss is DISABLED.")

ob_volume_threshold_multiplier: float = float(config["ob_volume_threshold_multiplier"])
ob_lookback: int = config["ob_lookback"]

entry_volume_confirmation_enabled: bool = config["entry_volume_confirmation_enabled"]
entry_volume_ma_length: int = int(config.get("entry_volume_ma_length", 20))
entry_volume_multiplier: float = float(config.get("entry_volume_multiplier", 1.2))
if entry_volume_confirmation_enabled:
    log_info(f"Entry Volume Confirmation: ENABLED (Vol > {entry_volume_multiplier}x MA({entry_volume_ma_length}))")
else:
    log_info("Entry Volume Confirmation: DISABLED")

# --- Simulation Mode ---
SIMULATION_MODE: bool = config["simulation_mode"]
if SIMULATION_MODE:
    log_warning("SIMULATION MODE IS ACTIVE. No real orders will be placed.")
else:
    log_warning("!!! LIVE TRADING MODE IS ACTIVE. REAL ORDERS WILL BE PLACED !!!")
    try:
        if sys.stdin.isatty():
            user_confirm = input(f"{NEON_RED}TYPE 'LIVE' TO CONFIRM or press Enter/Ctrl+C to exit: {RESET}")
            if user_confirm.strip().upper() != "LIVE":
                log_info("Live trading not confirmed. Exiting.")
                sys.exit(0)
            log_info("Live trading confirmed by user.")
        else:
            log_warning("Running in non-interactive mode. Assuming confirmation for LIVE TRADING.")
            log_warning("Pausing for 5 seconds before starting live trading...")
            time.sleep(5)
    except EOFError:
        log_error("Cannot get user confirmation in this environment. Exiting live mode for safety.")
        sys.exit(1)


# --- Position Management State (Includes OCO ID) ---
position_default_structure: Dict[str, Any] = {
    "status": None,  # 'long' | 'short'
    "entry_price": None,  # float
    "quantity": None,  # float
    "order_id": None,  # str (Entry order ID)
    "stop_loss": None,  # float (Target SL price level)
    "take_profit": None,  # float (Target TP price level)
    "entry_time": None,  # pd.Timestamp (UTC)
    "sl_order_id": None,  # str (ID of separate SL order, if not OCO)
    "tp_order_id": None,  # str (ID of separate TP order, if not OCO)
    "oco_order_id": None,  # str (ID of OCO order, if used - might be list ID or SL/TP ID)
    "highest_price_since_entry": None,  # float: For long TSL
    "lowest_price_since_entry": None,  # float: For short TSL
    "current_trailing_sl_price": None,  # float: Active TSL price
}
position: Dict[str, Any] = position_default_structure.copy()


# --- State Saving and Resumption Functions ---
def save_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Saves the current position state to a JSON file."""
    global position
    try:
        state_to_save = position.copy()
        entry_time = state_to_save.get("entry_time")
        if isinstance(entry_time, pd.Timestamp):
            state_to_save["entry_time"] = entry_time.isoformat()

        with open(filename, "w") as f:
            json.dump(state_to_save, f, indent=4)
        log_debug(f"Position state saved successfully to {filename}")

    except (TypeError, IOError, Exception) as e:
        log_error(f"Error saving position state to {filename}: {e}", exc_info=True)


def load_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Loads position state from JSON, validates, and handles timestamp conversion."""
    global position
    log_info(f"Attempting to load position state from '{filename}'...")
    position = position_default_structure.copy()  # Start fresh

    if not os.path.exists(filename):
        log_info(f"No position state file found ({filename}). Starting with empty state.")
        return

    try:
        with open(filename, "r") as f:
            loaded_state: Dict[str, Any] = json.load(f)

        parsed_entry_time: Optional[pd.Timestamp] = None
        entry_time_str: Optional[str] = loaded_state.get("entry_time")
        if entry_time_str and isinstance(entry_time_str, str):
            try:
                ts = pd.Timestamp(entry_time_str)
                parsed_entry_time = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
            except ValueError:
                log_error(f"Could not parse entry_time '{entry_time_str}' from state file. Setting to None.")

        updated_count = 0
        issues = []
        for key, default_value in position_default_structure.items():
            if key in loaded_state:
                loaded_value = loaded_state[key]
                if key == "entry_time":
                    loaded_value = parsed_entry_time  # Use parsed timestamp

                # Type check (allow int -> float)
                expected_type = type(default_value)
                allow_int_for_float = expected_type is float and isinstance(loaded_value, int)
                if (
                    loaded_value is not None
                    and default_value is not None
                    and not isinstance(loaded_value, expected_type)
                    and not allow_int_for_float
                ):
                    issues.append(
                        f"Type mismatch for '{key}': Expected {expected_type.__name__}, Got {type(loaded_value).__name__}"
                    )
                else:
                    position[key] = float(loaded_value) if allow_int_for_float else loaded_value
                    if key != "entry_time" or parsed_entry_time is not None:
                        updated_count += 1
            elif default_value is not None:  # Key missing, but had a non-None default
                issues.append(f"Key missing: '{key}'")

        # Log issues
        if issues:
            log_warning(
                f"Issues loading state file '{filename}' (using defaults for problematic keys): {'; '.join(issues)}"
            )
        extra_keys = set(loaded_state.keys()) - set(position_default_structure.keys())
        if extra_keys:
            log_warning(f"Extra keys found in state file (ignored): {extra_keys}")

        if updated_count > 0:
            log_info(f"Position state loaded successfully from {filename}. Applied {updated_count} values.")
            display_position_status(position, price_precision_digits, amount_precision_digits)
        else:
            log_warning(f"Loaded state file {filename}, but applied 0 valid values. Using default state.")
            position = position_default_structure.copy()

    except json.JSONDecodeError as e:
        log_error(f"Error decoding JSON from state file {filename}: {e}. Starting with empty state.", exc_info=False)
        position = position_default_structure.copy()
    except Exception as e:
        log_error(f"Error loading position state from {filename}: {e}. Starting with empty state.", exc_info=True)
        position = position_default_structure.copy()


# --- Data Fetching Function (Decorated) ---
@api_retry_decorator
def fetch_ohlcv_data(
    exchange_instance: ccxt.Exchange, trading_symbol: str, tf: str, limit_count: int
) -> Optional[pd.DataFrame]:
    """Fetches and cleans OHLCV data, returning a timezone-aware DataFrame."""
    log_debug(f"Fetching {limit_count} candles for {trading_symbol} on {tf} timeframe...")
    try:
        if trading_symbol not in exchange_instance.markets:
            log_error(f"Symbol '{trading_symbol}' not found in locally loaded markets.")
            return None

        ohlcv: List[list] = exchange_instance.fetch_ohlcv(trading_symbol, tf, limit=limit_count)

        if not ohlcv:
            log_warning(f"No OHLCV data returned for {trading_symbol} ({tf}).")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        numeric_cols: List[str] = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        initial_rows: int = len(df)
        df.dropna(subset=numeric_cols, inplace=True)
        rows_dropped: int = initial_rows - len(df)
        if rows_dropped > 0:
            log_debug(f"Dropped {rows_dropped} rows with NaN OHLCV data.")

        if df.empty:
            log_warning(f"DataFrame empty after cleaning NaN OHLCV for {trading_symbol} ({tf}).")
            return None

        log_debug(f"Successfully fetched and processed {len(df)} candles for {trading_symbol}.")
        return df

    except ccxt.BadSymbol as e:
        log_error(f"BadSymbol error fetching OHLCV for {trading_symbol}: {e}.")
        return None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange specific error fetching OHLCV for {trading_symbol}: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected non-retryable error fetching OHLCV: {e}", exc_info=True)
        return None


# --- Enhanced Order Block Identification Function ---
def identify_potential_order_block(
    df: pd.DataFrame, vol_thresh_mult: float, lookback_len: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Identifies recent potential bullish/bearish order blocks."""
    bullish_ob: Optional[Dict[str, Any]] = None
    bearish_ob: Optional[Dict[str, Any]] = None
    required_cols = ["open", "high", "low", "close", "volume"]

    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        log_debug("Input DataFrame missing required columns for OB detection.")
        return None, None
    if len(df) < lookback_len + 2:
        log_debug(f"Not enough data for OB detection (need > {lookback_len + 1} rows).")
        return None, None

    try:
        completed_candles_df = df.iloc[:-1]
        avg_volume = 0.0
        min_periods_vol = max(1, lookback_len // 2)
        if len(completed_candles_df) >= min_periods_vol:
            rolling_vol = (
                completed_candles_df["volume"].rolling(window=lookback_len, min_periods=min_periods_vol).mean()
            )
            if not rolling_vol.empty and pd.notna(rolling_vol.iloc[-1]):
                avg_volume = rolling_vol.iloc[-1]
        if avg_volume == 0 and len(completed_candles_df) > 0:  # Fallback if rolling mean failed
            avg_volume = completed_candles_df["volume"].mean()
            if pd.isna(avg_volume):
                avg_volume = 0.0

        volume_threshold = avg_volume * vol_thresh_mult if avg_volume > 0 else float("inf")
        log_debug(
            f"OB Analysis | Lookback: {lookback_len}, Avg Vol: {avg_volume:.2f}, Threshold Vol: {volume_threshold:.2f}"
        )

        search_depth = min(len(df) - 2, lookback_len * 3)
        start_index = len(df) - 2
        end_index = max(0, start_index - search_depth)

        for i in range(start_index, end_index, -1):
            if i < 1:
                break
            try:
                impulse_candle = df.iloc[i]
                ob_candle = df.iloc[i - 1]
            except IndexError:
                break

            if impulse_candle.isnull().any() or ob_candle.isnull().any():
                continue

            is_high_volume = impulse_candle["volume"] > volume_threshold
            is_bullish_impulse = (
                impulse_candle["close"] > impulse_candle["open"] and impulse_candle["close"] > ob_candle["high"]
            )
            is_bearish_impulse = (
                impulse_candle["close"] < impulse_candle["open"] and impulse_candle["close"] < ob_candle["low"]
            )
            ob_is_bearish = ob_candle["close"] < ob_candle["open"]
            ob_is_bullish = ob_candle["close"] > ob_candle["open"]
            impulse_sweeps_ob_low = impulse_candle["low"] < ob_candle["low"]
            impulse_sweeps_ob_high = impulse_candle["high"] > ob_candle["high"]

            # Bullish OB: Bearish OB candle, high-vol bullish impulse sweeping low
            if not bullish_ob and ob_is_bearish and is_bullish_impulse and is_high_volume and impulse_sweeps_ob_low:
                bullish_ob = {
                    "high": ob_candle["high"],
                    "low": ob_candle["low"],
                    "time": ob_candle.name,
                    "type": "bullish",
                }
                log_debug(f"Potential Bullish OB found at {ob_candle.name.strftime('%Y-%m-%d %H:%M')}")
                if bearish_ob:
                    break  # Found both

            # Bearish OB: Bullish OB candle, high-vol bearish impulse sweeping high
            elif not bearish_ob and ob_is_bullish and is_bearish_impulse and is_high_volume and impulse_sweeps_ob_high:
                bearish_ob = {
                    "high": ob_candle["high"],
                    "low": ob_candle["low"],
                    "time": ob_candle.name,
                    "type": "bearish",
                }
                log_debug(f"Potential Bearish OB found at {ob_candle.name.strftime('%Y-%m-%d %H:%M')}")
                if bullish_ob:
                    break  # Found both

        return bullish_ob, bearish_ob

    except Exception as e:
        log_error(f"Error during order block identification: {e}", exc_info=True)
        return None, None


# --- Indicator Calculation Function ---
def calculate_technical_indicators(
    df: Optional[pd.DataFrame],
    rsi_len: int,
    stoch_params: Dict[str, int],
    calc_atr: bool = False,
    atr_len: int = 14,
    calc_vol_ma: bool = False,
    vol_ma_len: int = 20,
) -> Optional[pd.DataFrame]:
    """Calculates indicators (RSI, Stoch, optional ATR, Vol MA) using pandas_ta."""
    if df is None or df.empty:
        return None
    required_base = ["high", "low", "close"]
    if calc_vol_ma:
        required_base.append("volume")
    if not all(col in df.columns for col in required_base):
        log_error(f"Missing required columns for indicators: Need {required_base}, got {df.columns.tolist()}")
        return None

    log_debug(f"Calculating indicators on DataFrame with {len(df)} rows...")
    try:
        df_processed = df.copy()
        indicator_cols = []

        # RSI
        rsi_col = f"RSI_{rsi_len}"
        df_processed.ta.rsi(length=rsi_len, append=True, col_names=(rsi_col,))
        indicator_cols.append(rsi_col)

        # Stochastic
        stoch_k_col = f"STOCHk_{stoch_params['k']}_{stoch_params['d']}_{stoch_params['smooth_k']}"
        stoch_d_col = f"STOCHd_{stoch_params['k']}_{stoch_params['d']}_{stoch_params['smooth_k']}"
        df_processed.ta.stoch(
            k=stoch_params["k"],
            d=stoch_params["d"],
            smooth_k=stoch_params["smooth_k"],
            append=True,
            col_names=(stoch_k_col, stoch_d_col),
        )
        indicator_cols.extend([stoch_k_col, stoch_d_col])

        # ATR (optional)
        if calc_atr:
            atr_col = f"ATRr_{atr_len}"
            df_processed.ta.atr(length=atr_len, append=True, col_names=(atr_col,))
            indicator_cols.append(atr_col)
            log_debug(f"ATR ({atr_len}) calculated.")

        # Volume MA (optional)
        if calc_vol_ma:
            if "volume" in df_processed.columns:
                vol_ma_col = f"VOL_MA_{vol_ma_len}"
                min_p_vol = max(1, vol_ma_len // 2)
                df_processed[vol_ma_col] = (
                    df_processed["volume"].rolling(window=vol_ma_len, min_periods=min_p_vol).mean()
                )
                indicator_cols.append(vol_ma_col)
                log_debug(f"Volume MA ({vol_ma_len}) calculated.")
            else:
                log_warning("Volume column not found, cannot calculate Volume MA.")

        # Clean NaNs from indicator columns
        initial_len = len(df_processed)
        valid_indicator_cols = [col for col in indicator_cols if col in df_processed.columns]
        if valid_indicator_cols:
            df_processed.dropna(subset=valid_indicator_cols, inplace=True)
        rows_dropped = initial_len - len(df_processed)
        if rows_dropped > 0:
            log_debug(f"Dropped {rows_dropped} rows with NaN indicators.")

        if df_processed.empty:
            log_warning("DataFrame empty after dropping NaN indicators.")
            return None

        log_debug(f"Indicator calculation complete. DataFrame has {len(df_processed)} rows.")
        return df_processed

    except Exception as e:
        log_error(f"Error calculating technical indicators: {e}", exc_info=True)
        return None


# --- Position Sizing Function (Decorated) ---
@api_retry_decorator
def calculate_position_size(
    exchange_instance: ccxt.Exchange,
    trading_symbol: str,
    current_price: float,
    stop_loss_price: float,
    risk_perc: float,
) -> Optional[float]:
    """Calculates position size based on risk, balance, and SL distance."""
    if not (0 < risk_perc < 1):
        log_error(f"Invalid risk percentage: {risk_perc}.")
        return None
    if current_price <= 0 or stop_loss_price <= 0:
        log_error(f"Invalid prices for size calculation: Entry={current_price}, SL={stop_loss_price}")
        return None

    try:
        log_debug(
            f"Calculating position size: Entry={current_price:.{price_precision_digits}f}, SL={stop_loss_price:.{price_precision_digits}f}, Risk={risk_perc * 100:.2f}%"
        )

        market = exchange_instance.market(trading_symbol)
        if not market:
            log_error(f"Market info for {trading_symbol} not found.")
            return None

        base_currency: Optional[str] = market.get("base")
        quote_currency: Optional[str] = market.get("quote")
        is_inverse: bool = market.get("inverse", False)
        float(market.get("contractSize", 1.0))
        limits = market.get("limits", {})
        min_amount_limit = limits.get("amount", {}).get("min")
        min_cost_limit = limits.get("cost", {}).get("min")

        if not base_currency or not quote_currency:
            log_error(f"Could not determine base/quote for {trading_symbol}.")
            return None
        if is_inverse:
            # Stronger Warning for Inverse Contracts
            log_warning(
                f"Calculating size for INVERSE contract ({trading_symbol}). Sizing formula assumes risk is calculated based on quote currency balance and loss per unit is measured in quote currency. VERIFY this matches your exchange's PnL calculation for inverse contracts."
            )

        # Fetch Balance
        balance_params = {}  # Add specific params if needed for contracts on your exchange
        balance = exchange_instance.fetch_balance(params=balance_params)

        # Find Available Balance (Quote Currency) - More Robust Parsing
        available_balance = 0.0
        balance_source_info = "N/A"

        if quote_currency in balance:
            # Standard structure: balance['USDT']['free']
            if isinstance(balance[quote_currency], dict):
                available_balance = float(balance[quote_currency].get("free", 0.0))
                balance_source_info = f"balance['{quote_currency}']['free']"
                if available_balance == 0.0 and balance[quote_currency].get("total", 0.0) > 0:
                    # Fallback to total if free is 0 but total exists (less safe)
                    available_balance = float(balance[quote_currency]["total"])
                    balance_source_info = f"balance['{quote_currency}']['total'] (used as free was 0)"
                    log_warning(f"Using 'total' {quote_currency} balance ({available_balance}), 'free' was 0.")
            # Flat structure: balance['free']['USDT'] (less common for fetch_balance)
            elif "free" in balance and isinstance(balance["free"], dict):
                available_balance = float(balance["free"].get(quote_currency, 0.0))
                balance_source_info = f"balance['free']['{quote_currency}']"

        # Bybit Unified specific check (example, might need adjustment based on API changes)
        elif (
            exchange_instance.id == "bybit"
            and quote_currency == "USDT"
            and "info" in balance
            and isinstance(balance["info"], dict)
            and "result" in balance["info"]
            and isinstance(balance["info"]["result"], dict)
            and "list" in balance["info"]["result"]
        ):
            for wallet_info in balance["info"]["result"]["list"]:
                if (
                    isinstance(wallet_info, dict)
                    and wallet_info.get("accountType") == "UNIFIED"
                    and wallet_info.get("coin") == quote_currency
                ):
                    available_balance = float(
                        wallet_info.get("availableToWithdraw", wallet_info.get("walletBalance", 0.0))
                    )
                    balance_source_info = "Bybit Unified 'availableToWithdraw' or 'walletBalance'"
                    break

        if available_balance <= 0:
            log_error(
                f"No available {quote_currency} balance ({available_balance}) found. Source checked: {balance_source_info}. Full details: {balance.get(quote_currency, balance.get('info', 'N/A'))}"
            )
            return None
        log_info(
            f"Available balance ({quote_currency}): {available_balance:.{price_precision_digits}f} (Source: {balance_source_info})"
        )

        # Calculate Risk Amount & Price Difference
        risk_amount_quote = available_balance * risk_perc
        price_diff = abs(current_price - stop_loss_price)
        if price_diff <= min_tick / 2:
            log_error(
                f"Stop-loss {stop_loss_price:.{price_precision_digits}f} too close to entry {current_price:.{price_precision_digits}f} (Diff: {price_diff:.{price_precision_digits + 4}f} <= Min Tick/2)."
            )
            return None

        # Calculate Quantity (Base Currency)
        quantity_base: float
        if is_inverse:
            # Risk Amount (Quote) / (Loss per Contract in Quote) * Contract Size (Base/Contract)
            # Loss per Contract (Quote) = Price Diff (Quote/Base) * Contract Size (Base/Contract) / Entry Price (Quote/Base) ??? -> Complex.
            # Simpler approach: Quantity (BASE) = Risk Amount (Quote) / Price Diff (Quote / BASE)
            # This gives the amount in BASE currency equivalent to the risk.
            quantity_base = risk_amount_quote / price_diff
            # Note: For inverse, the *number* of contracts might be quantity_base / contract_size if needed,
            # but ccxt create_order amount is usually in base units.
        else:  # Linear
            # Quantity (Base) = Risk Amount (Quote) / Price Diff (Quote / Base)
            quantity_base = risk_amount_quote / price_diff

        # Adjust for Precision
        try:
            quantity_adjusted = float(exchange_instance.amount_to_precision(trading_symbol, quantity_base))
        except ccxt.ExchangeError as e:
            log_warning(f"Could not use exchange.amount_to_precision: {e}. Using manual rounding.")
            # Fallback: Round to precision digits (less reliable)
            scale = 10**amount_precision_digits
            quantity_adjusted = (
                math.floor(quantity_base * scale) / scale
                if quantity_base > 0
                else math.ceil(quantity_base * scale) / scale
            )

        if quantity_adjusted <= 0:
            log_error(f"Calculated adjusted quantity ({quantity_adjusted}) is zero or negative.")
            return None

        # Check Limits
        if min_amount_limit is not None and quantity_adjusted < min_amount_limit:
            log_error(
                f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} < minimum {min_amount_limit} {base_currency}."
            )
            return None

        estimated_cost = quantity_adjusted * current_price
        if min_cost_limit is not None and estimated_cost < min_cost_limit:
            log_error(
                f"Estimated cost {estimated_cost:.{price_precision_digits}f} {quote_currency} < minimum {min_cost_limit} {quote_currency}. Need Qty > {min_cost_limit / current_price:.{amount_precision_digits}f}"
            )
            return None

        # Final sanity check vs balance
        cost_buffer_factor = 0.99
        if estimated_cost > available_balance * cost_buffer_factor:
            log_error(
                f"Estimated cost ({estimated_cost:.{price_precision_digits}f}) exceeds {cost_buffer_factor * 100:.0f}% of available balance ({available_balance:.{price_precision_digits}f}). Reduce risk %."
            )
            return None

        log_info(
            f"{NEON_GREEN}Position size calculated: {quantity_adjusted:.{amount_precision_digits}f} {base_currency}{RESET} (Risking ~{risk_amount_quote:.2f} {quote_currency})"
        )
        return quantity_adjusted

    except ccxt.AuthenticationError as e:
        log_error(f"Authentication error during position size calculation: {e}")
        return None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error during position size calculation: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error calculating position size: {e}", exc_info=True)
        return None


# --- Order Placement Functions ---


@api_retry_decorator
def cancel_order_with_retry(exchange_instance: ccxt.Exchange, order_id: str, trading_symbol: str) -> bool:
    """Attempts to cancel an order by ID, handling simulation and OrderNotFound."""
    if not order_id or not isinstance(order_id, str):
        log_debug("No valid order ID provided to cancel.")
        return False

    log_info(f"Attempting to cancel order ID: {order_id} for {trading_symbol}...")
    if SIMULATION_MODE:
        log_warning(f"SIMULATION: Skipped cancelling order {order_id}.")
        return True
    else:
        log_warning(f"!!! LIVE MODE: Sending cancellation request for order {order_id}.")
        try:
            exchange_instance.cancel_order(order_id, trading_symbol)
            log_info(f"Order {order_id} cancellation request sent successfully.")
            return True
        except ccxt.OrderNotFound:
            log_info(f"Order {order_id} not found (already closed or cancelled).")
            return True  # Desired state achieved
        # Let other ccxt errors propagate up (handled by decorator/caller)


@api_retry_decorator
def place_market_order(
    exchange_instance: ccxt.Exchange, trading_symbol: str, side: str, amount: float, reduce_only: bool = False
) -> Optional[Dict[str, Any]]:
    """Places a market order with retry logic, simulation, and robust error handling."""
    if side not in ["buy", "sell"]:
        log_error(f"Invalid order side: '{side}'.")
        return None
    if not isinstance(amount, (float, int)) or amount <= 0:
        log_error(f"Invalid order amount: {amount}.")
        return None

    try:
        market = exchange_instance.market(trading_symbol)
        base_currency: str = market.get("base", "")
        quote_currency: str = market.get("quote", "")

        amount_formatted = float(exchange_instance.amount_to_precision(trading_symbol, amount))
        if amount_formatted <= 0:
            log_error(f"Amount {amount} became zero or negative ({amount_formatted}) after precision. Cannot order.")
            return None

        log_info(
            f"Attempting to place {side.upper()} market order: {amount_formatted:.{amount_precision_digits}f} {base_currency} on {trading_symbol} {'(ReduceOnly)' if reduce_only else ''}..."
        )

        params = {}
        is_contract_market = market.get("swap") or market.get("future") or market.get("contract")
        if reduce_only and is_contract_market:
            params["reduceOnly"] = True
            log_debug("Applying 'reduceOnly=True' to market order.")
        elif reduce_only:
            log_warning("ReduceOnly requested but market is not contract type. Ignoring param.")

        order: Optional[Dict[str, Any]] = None
        if SIMULATION_MODE:
            log_warning("!!! SIMULATION: Market order placement skipped.")
            sim_price = 0.0
            try:

                @api_retry_decorator  # Decorate inner function too
                def fetch_sim_ticker(exch, sym):
                    return exch.fetch_ticker(sym)

                ticker = fetch_sim_ticker(exchange_instance, trading_symbol)
                sim_price = ticker.get("last", ticker.get("close", 0.0))
            except Exception as ticker_e:
                log_warning(f"Could not fetch ticker for simulation price: {ticker_e}")

            sim_order_id: str = f"sim_market_{pd.Timestamp.now(tz='UTC').isoformat()}"
            sim_cost = amount_formatted * sim_price if sim_price > 0 else 0.0
            order = {
                "id": sim_order_id,
                "clientOrderId": sim_order_id,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": "closed",
                "symbol": trading_symbol,
                "type": "market",
                "side": side,
                "price": sim_price,
                "average": sim_price,
                "amount": amount_formatted,
                "filled": amount_formatted,
                "remaining": 0.0,
                "cost": sim_cost,
                "fee": None,
                "reduceOnly": params.get("reduceOnly", False),
                "info": {"simulated": True},
            }
        else:
            log_warning(
                f"!!! LIVE MODE: Placing real market order {'(ReduceOnly)' if params.get('reduceOnly') else ''}."
            )
            order = exchange_instance.create_market_order(
                symbol=trading_symbol, side=side, amount=amount_formatted, params=params
            )
            time.sleep(1.5)  # Allow exchange processing time

        if order:
            order_id = order.get("id")
            status = order.get("status")
            avg_price = order.get("average", order.get("price"))
            filled = order.get("filled")
            cost = order.get("cost")
            reduce_info = order.get("reduceOnly", params.get("reduceOnly", "N/A"))
            price_str = f"{avg_price:.{price_precision_digits}f}" if isinstance(avg_price, (float, int)) else "N/A"
            filled_str = f"{filled:.{amount_precision_digits}f}" if isinstance(filled, (float, int)) else "N/A"
            cost_str = f"{cost:.{price_precision_digits}f}" if isinstance(cost, (float, int)) else "N/A"
            log_info(
                f"Order Result | ID: {order_id or 'N/A'}, Status: {status or 'N/A'}, Avg Price: {price_str}, Filled: {filled_str} {base_currency}, Cost: {cost_str} {quote_currency}, ReduceOnly: {reduce_info}"
            )
        else:
            log_error("Market order placement did not return a valid order object.")
            return None
        return order

    except ccxt.InsufficientFunds as e:
        log_error(f"Insufficient funds for {side} {amount_formatted:.{amount_precision_digits}f} {trading_symbol}: {e}")
        return None
    except ccxt.InvalidOrder as e:
        log_error(
            f"Invalid market order for {side} {amount_formatted:.{amount_precision_digits}f} {trading_symbol}: {e}"
        )
        return None
    except ccxt.OrderNotFound as e:  # Should be rare for market orders unless rejected instantly
        log_error(f"OrderNotFound error placing market order for {trading_symbol}: {e}")
        return None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error placing {side} market order for {trading_symbol}: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error placing {side} market order: {e}", exc_info=True)
        return None


# --- Place SL/TP Orders (OCO Preferred) ---
@api_retry_decorator  # Retry the whole OCO/separate placement process
def place_sl_tp_orders(
    exchange_instance: ccxt.Exchange,
    trading_symbol: str,
    position_side: str,
    quantity: float,
    sl_price: float,
    tp_price: float,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Places OCO (One-Cancels-the-Other) SL/TP orders if supported, otherwise places
    separate stop-loss and take-profit orders. Uses reduceOnly.

    Args:
        exchange_instance: ccxt exchange instance.
        trading_symbol: Market symbol.
        position_side: 'long' or 'short' (of the main position).
        quantity: Quantity to close (base currency).
        sl_price: Stop-loss trigger price.
        tp_price: Take-profit limit price.

    Returns:
        Tuple (sl_order_id, tp_order_id, oco_order_id/list_id):
        IDs of the placed orders. oco_order_id might be a list ID or one of the order IDs
        depending on the exchange. sl/tp IDs will be None if OCO is used and successful.
        Returns (None, None, None) on failure.
    """
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    oco_ref_id: Optional[str] = None  # ID referencing the OCO group or one of the orders

    # --- Input Validation ---
    if not (
        isinstance(quantity, (float, int))
        and quantity > 0
        and position_side in ["long", "short"]
        and isinstance(sl_price, (float, int))
        and sl_price > 0
        and isinstance(tp_price, (float, int))
        and tp_price > 0
    ):
        log_error(
            f"Invalid inputs for SL/TP placement: Qty={quantity}, Side={position_side}, SL={sl_price}, TP={tp_price}"
        )
        return None, None, None
    # Basic logic check
    if (position_side == "long" and sl_price >= tp_price) or (position_side == "short" and sl_price <= tp_price):
        log_error(f"Invalid SL/TP prices for {position_side.upper()} position: SL={sl_price}, TP={tp_price}")
        return None, None, None

    try:
        market = exchange_instance.market(trading_symbol)
        market.get("base", "")
        close_side = "sell" if position_side == "long" else "buy"

        qty_fmt = float(exchange_instance.amount_to_precision(trading_symbol, quantity))
        sl_price_fmt = float(exchange_instance.price_to_precision(trading_symbol, sl_price))
        tp_price_fmt = float(exchange_instance.price_to_precision(trading_symbol, tp_price))
        if qty_fmt <= 0:
            raise ValueError("Quantity non-positive after formatting.")

        # --- Prepare Common Parameters ---
        is_contract = market.get("swap") or market.get("future") or market.get("contract")
        params = {"reduceOnly": True} if is_contract else {}
        if params:
            log_debug("Applying 'reduceOnly=True' to protection orders.")

        # --- Determine Stop Loss Type ---
        # Prefer stopMarket ('STOP_LOSS' or 'STOP_MARKET' type in createOrder)
        # Fallback stopLimit ('STOP_LOSS_LIMIT')
        sl_order_type: Optional[str] = None
        sl_limit_price: Optional[float] = None  # For stopLimit only

        # Check ccxt unified methods first (may require specific type strings)
        # These type strings vary wildly between exchanges! Examples:
        # Binance: 'STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT'
        # Bybit: 'StopMarket', 'StopLimit' (used with triggerPrice param)
        # OKX: 'conditional' type with algo type like 'oco', 'trigger', 'move_order_stop'
        # Needs careful checking of exchange.has flags and API docs for reliable OCO/Stop implementation.

        # Heuristic based on common capabilities:
        # We aim for a stop-market for SL and a limit for TP.
        stop_market_possible = (
            exchange_instance.has.get("createStopMarketOrder", False)
            or "STOP_MARKET" in exchange_instance.options.get("createOrderTypes", [])
            or exchange_instance.id in ["binance", "bybit"]
        )  # Assume common ones support it via params

        stop_limit_possible = (
            exchange_instance.has.get("createStopLimitOrder", False)
            or "STOP_LOSS_LIMIT" in exchange_instance.options.get("createOrderTypes", [])
            or "StopLimit" in exchange_instance.options.get("createOrderTypes", [])
        )  # e.g., Bybit

        if stop_market_possible:
            sl_order_type = "stopMarket"  # Use generic type, ccxt might translate
            log_debug("Selected SL type: stopMarket (preferred).")
        elif stop_limit_possible:
            sl_order_type = "stopLimit"
            # Configurable stop-limit offset? For now, use small tick-based offset.
            limit_offset_ticks = min_tick * 10  # Consider making this configurable
            sl_limit_raw = (
                sl_price_fmt - limit_offset_ticks if close_side == "sell" else sl_price_fmt + limit_offset_ticks
            )
            sl_limit_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_raw))
            log_warning(
                f"Selected SL type: stopLimit (fallback). Trigger: {sl_price_fmt}, Limit: {sl_limit_price}. Fill not guaranteed."
            )
        else:
            log_error(
                f"Cannot determine supported stop order type (stopMarket/stopLimit) for {exchange_instance.id}. Cannot place SL."
            )
            # Cannot proceed if no stop type found
            return None, None, None

        # --- Attempt OCO Placement (if supported and SL type determined) ---
        use_oco = exchange_instance.has.get("oco", False) and sl_order_type is not None
        if use_oco:
            log_info(
                f"Attempting OCO {close_side.upper()} order: Qty={qty_fmt}, TP Limit={tp_price_fmt}, SL Trigger={sl_price_fmt} ({sl_order_type}) {'(ReduceOnly)' if params else ''}"
            )
            # Construct OCO parameters - HIGHLY EXCHANGE SPECIFIC!
            # Requires careful reading of ccxt docs for the specific exchange's createOrder 'params' for OCO.
            # Example structure (conceptual, WILL NOT WORK universally):
            oco_params = params.copy()
            try:
                # This is a generic attempt, specific exchanges need specific params structure
                # Some might need separate price/stopPrice, others might use listClientOrderId, etc.
                if exchange_instance.id == "binance":  # Example for Binance
                    oco_params.update(
                        {
                            "stopPrice": sl_price_fmt,  # SL trigger price
                            # Binance uses different types for SL/TP within OCO
                            # 'stopLimitPrice': sl_limit_price, # Required only if using STOP_LOSS_LIMIT type
                            # 'stopLimitTimeInForce': 'GTC', # Required for stopLimit
                            # The 'price' field is typically the TP limit price for Binance OCO
                        }
                    )
                    # Determine SL type for Binance OCO: STOP_MARKET or STOP_LOSS_LIMIT
                    # TP type is usually LIMIT_MAKER or just LIMIT

                    # OCO might be placed via create_order with special type 'oco' or specific params
                    # Trying general createOrder with params first
                    if SIMULATION_MODE:
                        log_warning("!!! SIMULATION: OCO order placement skipped.")
                        sim_oco_id = f"sim_oco_{pd.Timestamp.now(tz='UTC').isoformat()}"
                        oco_ref_id = sim_oco_id  # Simulate successful OCO
                        # In OCO, individual IDs might not be returned immediately or easily accessible
                        sl_order_id = None
                        tp_order_id = None
                    else:
                        log_warning("!!! LIVE MODE: Placing real OCO order (Binance structure).")
                        # Note: Binance OCO often placed via specific endpoint or complex params in createOrder
                        # This is a simplified attempt, might need `create_order` with type='oco' or custom method call.
                        # Assuming `create_order` can handle it via params (less likely for OCO)
                        # A more robust approach would use `create_oco_order` if available or structure params exactly as API requires.
                        # For now, let's simulate failure for non-Binance to avoid placing bad orders.
                        # raise ccxt.NotSupported(f"OCO placement structure for {exchange_instance.id} not fully implemented in this example.")

                        # If trying Binance structure via create_order:
                        # order_result = exchange_instance.create_order(
                        #     symbol=trading_symbol,
                        #     type=tp_type_for_oco, # Often OCO uses the TP type? Check docs.
                        #     side=close_side,
                        #     amount=qty_fmt,
                        #     price=tp_price_fmt, # TP price
                        #     params=oco_params # Contains SL price, potentially SL limit price/timeInForce
                        # )
                        # oco_ref_id = order_result.get('id') # Or potentially order_result['info']['listClientOrderId']
                        log_error(
                            "Live OCO placement for Binance needs specific implementation beyond this generic structure. Simulating failure."
                        )
                        raise ccxt.NotSupported("Live OCO for Binance needs specific handling.")

                # Add elif blocks here for other exchanges supporting OCO (e.g., Bybit, OKX) with their specific param structures.
                # elif exchange_instance.id == 'bybit': ...
                # elif exchange_instance.id == 'okx': ...

                else:  # Exchange supports OCO but we haven't implemented its specific structure
                    log_warning(
                        f"Exchange {exchange_instance.id} supports OCO, but specific parameter structure not implemented here. Falling back to separate SL/TP orders."
                    )
                    use_oco = False  # Force fallback to separate orders

                # If OCO attempt was made (simulated or real) and seems successful
                if oco_ref_id:
                    log_info(
                        f"OCO order request processed {'(Simulated)' if SIMULATION_MODE else ''}. Ref ID: {oco_ref_id}"
                    )
                    return None, None, oco_ref_id  # Return OCO ref ID, clear separate IDs

            except ccxt.NotSupported as e:
                log_warning(
                    f"OCO order placement attempt failed (NotSupported or Structure Unknown): {e}. Falling back to separate SL/TP orders."
                )
                use_oco = False  # Ensure fallback
            except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                log_error(f"OCO order placement failed: {e}. Falling back to separate SL/TP orders.")
                use_oco = False
            except Exception as e:
                log_error(
                    f"Unexpected error during OCO placement: {e}. Falling back to separate SL/TP orders.", exc_info=True
                )
                use_oco = False

        # --- Fallback: Place Separate SL and TP Orders ---
        if not use_oco and sl_order_type is not None:
            log_info("Placing separate SL and TP orders...")
            sl_placed = False
            tp_placed = False

            # --- Place Stop-Loss Order ---
            try:
                log_info(
                    f"Attempting SL ({sl_order_type}): {close_side.upper()} {qty_fmt} @ trigger ~{sl_price_fmt} {'Limit ' + str(sl_limit_price) if sl_limit_price else ''} {'(ReduceOnly)' if params else ''}"
                )
                sl_params_sep = params.copy()
                sl_params_sep["stopPrice"] = sl_price_fmt  # Common param for stop orders

                if SIMULATION_MODE:
                    log_warning("!!! SIMULATION: Separate SL order placement skipped.")
                    sl_order_id = f"sim_sl_{pd.Timestamp.now(tz='UTC').isoformat()}"
                    sl_placed = True
                else:
                    log_warning(f"!!! LIVE MODE: Placing real separate {sl_order_type} SL order.")
                    # Use create_order with determined type and params
                    sl_order_result = exchange_instance.create_order(
                        symbol=trading_symbol,
                        type=sl_order_type,
                        side=close_side,
                        amount=qty_fmt,
                        price=sl_limit_price,  # price is limit price for stopLimit, ignored otherwise
                        params=sl_params_sep,
                    )
                    if sl_order_result and sl_order_result.get("id"):
                        sl_order_id = sl_order_result["id"]
                        sl_placed = True
                        log_info(
                            f"Separate SL order request processed. ID: {sl_order_id}, Status: {sl_order_result.get('status', 'N/A')}"
                        )
                        time.sleep(0.75)
                    else:
                        log_error(f"Separate SL order placement failed or did not return ID. Result: {sl_order_result}")

            except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                log_error(f"Failed to place separate SL order: {e}")
            except Exception as e:
                log_error(f"Unexpected error placing separate SL order: {e}", exc_info=True)

            # --- Place Take-Profit Order (only if SL placement seemed okay or to try anyway) ---
            # If SL failed critically, maybe skip TP? Or place TP anyway? Place anyway for now.
            try:
                log_info(
                    f"Attempting TP (limit): {close_side.upper()} {qty_fmt} @ limit {tp_price_fmt} {'(ReduceOnly)' if params else ''}"
                )
                tp_params_sep = params.copy()

                if SIMULATION_MODE:
                    log_warning("!!! SIMULATION: Separate TP order placement skipped.")
                    tp_order_id = f"sim_tp_{pd.Timestamp.now(tz='UTC').isoformat()}"
                    tp_placed = True
                else:
                    log_warning("!!! LIVE MODE: Placing real separate limit TP order.")
                    # Use create_limit_order for clarity
                    tp_order_result = exchange_instance.create_limit_order(
                        symbol=trading_symbol, side=close_side, amount=qty_fmt, price=tp_price_fmt, params=tp_params_sep
                    )
                    if tp_order_result and tp_order_result.get("id"):
                        tp_order_id = tp_order_result["id"]
                        tp_placed = True
                        log_info(
                            f"Separate TP order request processed. ID: {tp_order_id}, Status: {tp_order_result.get('status', 'N/A')}"
                        )
                        time.sleep(0.75)
                    else:
                        log_error(f"Separate TP order placement failed or did not return ID. Result: {tp_order_result}")

            except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                log_error(f"Failed to place separate TP order: {e}")
            except Exception as e:
                log_error(f"Unexpected error placing separate TP order: {e}", exc_info=True)

            # --- Final check for separate orders ---
            if not sl_placed and not tp_placed:
                log_error("Both separate SL and TP order placements failed.")
                # Returning None, None, None indicates total failure
            elif not sl_placed:
                log_warning("Separate TP placed, but SL placement FAILED. POSITION PARTIALLY PROTECTED.")
            elif not tp_placed:
                log_warning("Separate SL placed, but TP placement FAILED.")

            return sl_order_id, tp_order_id, None  # Return separate IDs, OCO ID is None

        # Should not be reached unless logic error
        return None, None, None

    # --- Handle Top-Level Errors (including decorator retries failing) ---
    except ccxt.InsufficientFunds as e:  # Should be rare with reduceOnly
        log_error(f"Insufficient funds error during SL/TP placement (unexpected with reduceOnly): {e}")
        return None, None, None
    except ccxt.InvalidOrder as e:  # Catch validation errors not caught lower down
        log_error(f"Invalid order parameters placing SL/TP: {e}.")
        return None, None, None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange specific error placing SL/TP orders: {e}")
        return None, None, None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error in place_sl_tp_orders: {e}", exc_info=True)
        return None, None, None


# --- Position and Order Check Function (With Position Fetch Fallback) ---
@api_retry_decorator
def check_position_and_orders(exchange_instance: ccxt.Exchange, trading_symbol: str) -> bool:
    """
    Checks consistency between local state and exchange (orders/positions).
    Resets state if position appears closed. Returns True if state was reset.
    """
    global position
    if position["status"] is None:
        log_debug("No active position state to check.")
        return False

    log_debug(f"Checking state for active {position['status']} position on {trading_symbol}...")
    position_reset_required: bool = False
    order_to_cancel_id: Optional[str] = None
    cancel_side_label: str = ""  # 'TP' or 'SL'
    assumed_close_reason: Optional[str] = None  # 'SL', 'TP', 'Position Closed'

    try:
        # --- Check Tracked Orders ---
        sl_order_id: Optional[str] = position.get("sl_order_id")
        tp_order_id: Optional[str] = position.get("tp_order_id")
        oco_order_id: Optional[str] = position.get("oco_order_id")  # Might be list ID or one of the orders

        orders_were_tracked = bool(sl_order_id or tp_order_id or oco_order_id)
        order_check_passed = True  # Assume okay unless an open order is found missing

        if orders_were_tracked:
            # Fetch all open orders for the symbol (retried by decorator)
            open_orders: List[Dict[str, Any]] = exchange_instance.fetch_open_orders(trading_symbol)
            open_order_ids: set[str] = {order["id"] for order in open_orders if "id" in order}
            log_debug(
                f"Found {len(open_orders)} open orders for {trading_symbol}. IDs: {open_order_ids if open_order_ids else 'None'}"
            )

            if oco_order_id:
                # Check if OCO order (or any related order if ID isn't list ID) is still open
                # Exact check depends on how exchange reports OCO status/IDs
                # Simple check: if the tracked OCO ID is in the open orders list
                # This might need refinement based on exchange specifics
                if oco_order_id not in open_order_ids:
                    log_info(
                        f"{NEON_YELLOW}Tracked OCO order {oco_order_id} no longer open. Assuming position closed via SL/TP.{RESET}"
                    )
                    position_reset_required = True
                    assumed_close_reason = "OCO (SL/TP)"
                    # No specific order to cancel if OCO handled it
                    order_to_cancel_id = None
                else:
                    log_debug(f"Tracked OCO order {oco_order_id} appears to be open.")
                    order_check_passed = False  # Position likely still open

            else:  # Separate SL/TP orders
                sl_found_open = sl_order_id in open_order_ids if sl_order_id else False
                tp_found_open = tp_order_id in open_order_ids if tp_order_id else False

                log_debug(f"Tracked SL ({sl_order_id or 'None'}) open: {sl_found_open}")
                log_debug(f"Tracked TP ({tp_order_id or 'None'}) open: {tp_found_open}")

                # If SL was tracked and is NOT open anymore
                if sl_order_id and not sl_found_open:
                    log_info(
                        f"{NEON_YELLOW}Stop-loss order {sl_order_id} no longer open. Assuming position closed via SL.{RESET}"
                    )
                    position_reset_required = True
                    order_to_cancel_id = tp_order_id  # Mark TP for cancellation
                    cancel_side_label = "TP"
                    assumed_close_reason = "SL"
                # If TP was tracked and is NOT open anymore (and SL didn't trigger reset)
                elif tp_order_id and not tp_found_open:
                    log_info(
                        f"{NEON_GREEN}Take-profit order {tp_order_id} no longer open. Assuming position closed via TP.{RESET}"
                    )
                    position_reset_required = True
                    order_to_cancel_id = sl_order_id  # Mark SL for cancellation
                    cancel_side_label = "SL"
                    assumed_close_reason = "TP"
                elif sl_found_open or tp_found_open:
                    # At least one tracked protection order is still open
                    order_check_passed = False
                    log_debug("At least one tracked SL/TP order is still open.")
                # If neither was found open, but they *were* tracked, reset is already handled above.
                # If neither was tracked, order_check_passed remains True.

        # --- Fallback: Check Actual Position (if no orders tracked or orders missing) ---
        # Only applicable for contract markets where positions can be fetched.
        market = exchange_instance.market(trading_symbol)
        is_contract = market.get("swap") or market.get("future") or market.get("contract")
        can_fetch_positions = exchange_instance.has.get("fetchPositions", False)

        # Condition to check position:
        # 1. We think we have a position locally (position['status'] is not None)
        # 2. AND (EITHER no protection orders were tracked OR the order check above suggested closure (order_check_passed = True))
        # 3. AND it's a contract market where we can fetch positions
        if position["status"] is not None and order_check_passed and is_contract and can_fetch_positions:
            log_info("Order check suggests closure or no orders tracked. Verifying actual position status...")
            try:
                # Fetch positions for the specific symbol (some exchanges require symbol list)
                # Use api_retry_decorator for the fetch call
                @api_retry_decorator
                def fetch_positions_safely(exch, sym_list):
                    return exch.fetch_positions(symbols=sym_list)

                positions_data = fetch_positions_safely(exchange_instance, [trading_symbol])

                # Parse fetch_positions response (structure varies by exchange!)
                current_position_size = 0.0
                found_pos = False
                if positions_data and isinstance(positions_data, list):
                    for pos_info in positions_data:
                        if isinstance(pos_info, dict) and pos_info.get("symbol") == trading_symbol:
                            # Look for 'contracts', 'contractSize', 'side', 'unrealizedPnl' etc.
                            # 'contracts' or equivalent holding the size is key.
                            size_key = "contracts"  # Common key, check specific exchange if needed
                            if size_key not in pos_info and "info" in pos_info and isinstance(pos_info["info"], dict):
                                # Check common 'info' fields (e.g., Bybit uses 'size')
                                size_key = pos_info["info"].get("size", size_key)  # Example fallback

                            pos_size_str = pos_info.get(size_key, pos_info.get("info", {}).get(size_key, "0"))
                            try:
                                current_position_size = float(pos_size_str)
                            except (ValueError, TypeError):
                                log_warning(
                                    f"Could not parse position size ('{size_key}': {pos_size_str}) from fetch_positions result."
                                )
                                current_position_size = 0.0  # Assume zero if unparseable

                            # Check side if available to be extra sure
                            pos_side = pos_info.get("side", pos_info.get("info", {}).get("side", "unknown")).lower()
                            log_debug(f"Fetched position data: Size={current_position_size}, Side={pos_side}")

                            # Position exists if size is non-zero
                            if abs(current_position_size) > (
                                min_amount or 1e-9
                            ):  # Check against min_amount or small epsilon
                                # If position exists, but local state thought it was closed or untracked. RESYNC? Complex.
                                # For now, just confirm if it matches local state *direction*.
                                if (
                                    position["status"] == "long" and pos_side == "long" and current_position_size > 0
                                ) or (
                                    position["status"] == "short" and pos_side == "short" and current_position_size > 0
                                ):  # Some exchanges report size as positive for short too
                                    log_warning(
                                        f"Position check: Found active {pos_side} position ({current_position_size}) on exchange, matching local direction. Order check may have been inaccurate. Keeping local state."
                                    )
                                    found_pos = True
                                    # We won't reset state if position matches local direction
                                else:
                                    # Size or direction mismatch! Indicates state inconsistency.
                                    log_error(
                                        f"Position check: Found MISMATCHED position on exchange! Local: {position['status']}, Exchange: {pos_side} Size={current_position_size}. Resetting local state. MANUAL REVIEW RECOMMENDED."
                                    )
                                    position_reset_required = True
                                    assumed_close_reason = "State Mismatch (Position Fetch)"
                                    # Cancel any potentially lingering tracked orders if mismatch found
                                    order_to_cancel_id = sl_order_id or tp_order_id or oco_order_id
                                    cancel_side_label = "SL/TP/OCO"
                            else:
                                log_info(
                                    "Position check: No active position found via fetch_positions, confirming closure."
                                )
                                position_reset_required = True  # Confirmed no position
                                assumed_close_reason = "Position Closed (Fetch)"
                                # Cancel any lingering tracked orders
                                order_to_cancel_id = sl_order_id or tp_order_id or oco_order_id
                                cancel_side_label = "SL/TP/OCO"

                            break  # Stop after finding the position for the symbol
                else:
                    log_warning("Position check: fetch_positions returned empty or invalid data.")
                    # Cannot confirm closure via position fetch, rely on order check result.
                    position_reset_required = (
                        order_check_passed and orders_were_tracked
                    )  # Only reset if order check suggested it

                # If we found an active position matching our direction, override any reset trigger from order check
                if found_pos:
                    position_reset_required = False

            except ccxt.NotSupported:
                log_warning("fetch_positions not supported by exchange. Cannot use position fallback check.")
                # Rely solely on order check result
                position_reset_required = order_check_passed and orders_were_tracked
            except Exception as e:
                log_error(f"Error during fetch_positions check: {e}. Relying on order check.", exc_info=False)
                # Rely solely on order check result
                position_reset_required = order_check_passed and orders_were_tracked

        # --- Perform State Reset and Cancel Leftover Order ---
        if position_reset_required:
            log_info(f"Position closure detected/assumed (Reason: {assumed_close_reason}). Resetting local state.")

            if order_to_cancel_id:
                log_info(f"Attempting to cancel leftover {cancel_side_label} order: {order_to_cancel_id}")
                try:
                    if not cancel_order_with_retry(exchange_instance, order_to_cancel_id, trading_symbol):
                        log_error(
                            f"Failed to confirm cancellation of leftover order {order_to_cancel_id}. Manual check advised."
                        )
                except Exception as e:
                    log_error(
                        f"Error cancelling leftover order {order_to_cancel_id}: {e}. Manual check advised.",
                        exc_info=False,
                    )
            else:
                log_debug("No corresponding order ID was tracked or needed cancellation.")

            log_info("Resetting local position state.")
            position.update(position_default_structure.copy())
            save_position_state()
            display_position_status(position, price_precision_digits, amount_precision_digits)
            return True  # Indicate position state was reset

        log_debug("Position state appears consistent or actively protected.")
        return False  # Position was not reset

    # --- Handle Errors during check ---
    except ccxt.AuthenticationError as e:
        log_error(f"Authentication error checking position/orders: {e}")
        return False  # State uncertain
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error checking position/orders: {e}")
        return False  # State uncertain
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error checking position/orders: {e}", exc_info=True)
        return False  # State uncertain


# --- Trailing Stop Loss Update Function ---
def update_trailing_stop(
    exchange_instance: ccxt.Exchange, trading_symbol: str, current_price: float, last_atr: Optional[float]
) -> None:
    """Checks and updates Trailing Stop Loss if conditions are met."""
    global position
    if position["status"] is None or not enable_trailing_stop:
        return

    # Use OCO ID if available, otherwise SL ID
    active_sl_order_id = position.get("oco_order_id") or position.get("sl_order_id")

    if active_sl_order_id is None:
        log_warning("TSL Check: Cannot trail - No active SL/OCO order ID is tracked.")
        return
    if last_atr is None or last_atr <= 0:
        log_warning(f"TSL Check: Cannot trail - Invalid ATR ({last_atr}).")
        return

    log_debug(
        f"TSL Check: Active {position['status']} position. Current SL/OCO Order: {active_sl_order_id}, Last TSL Price: {position.get('current_trailing_sl_price')}"
    )

    entry_price: Optional[float] = position.get("entry_price")
    initial_sl_price: Optional[float] = position.get("stop_loss")  # Initial target SL
    current_tsl_price: Optional[float] = position.get("current_trailing_sl_price")

    if entry_price is None or initial_sl_price is None:
        log_warning("TSL Check: Missing entry or initial SL price in state.")
        return

    # Determine the SL price to compare against (current effective SL)
    current_effective_sl_price = current_tsl_price if current_tsl_price is not None else initial_sl_price

    # --- Update Peak Prices ---
    if position["status"] == "long":
        highest_seen = position.get("highest_price_since_entry", entry_price)
        if current_price > highest_seen:
            position["highest_price_since_entry"] = current_price
            log_debug(f"TSL: New high for long: {current_price:.{price_precision_digits}f}")
    elif position["status"] == "short":
        lowest_seen = position.get("lowest_price_since_entry", entry_price)
        if current_price < lowest_seen:
            position["lowest_price_since_entry"] = current_price
            log_debug(f"TSL: New low for short: {current_price:.{price_precision_digits}f}")

    # --- Check Activation and Calculate Potential New TSL ---
    tsl_activated = False
    potential_new_tsl_price: Optional[float] = None

    if position["status"] == "long":
        activation_thresh = entry_price + (last_atr * trailing_stop_activation_atr_multiplier)
        current_high = position.get("highest_price_since_entry", entry_price)
        if current_high > activation_thresh:
            tsl_activated = True
            potential_new_tsl_price = current_high - (last_atr * trailing_stop_atr_multiplier)
            log_debug(
                f"Long TSL Active. High ({current_high:.{price_precision_digits}f}) > Threshold ({activation_thresh:.{price_precision_digits}f}). Potential New TSL: {potential_new_tsl_price:.{price_precision_digits}f}"
            )
    elif position["status"] == "short":
        activation_thresh = entry_price - (last_atr * trailing_stop_activation_atr_multiplier)
        current_low = position.get("lowest_price_since_entry", entry_price)
        if current_low < activation_thresh:
            tsl_activated = True
            potential_new_tsl_price = current_low + (last_atr * trailing_stop_atr_multiplier)
            log_debug(
                f"Short TSL Active. Low ({current_low:.{price_precision_digits}f}) < Threshold ({activation_thresh:.{price_precision_digits}f}). Potential New TSL: {potential_new_tsl_price:.{price_precision_digits}f}"
            )

    # --- Check if Potential TSL is Improvement and Valid ---
    should_update_tsl = False
    new_tsl_price_fmt: Optional[float] = None

    if tsl_activated and potential_new_tsl_price is not None:
        try:
            new_tsl_price_fmt = float(exchange_instance.price_to_precision(trading_symbol, potential_new_tsl_price))
        except ccxt.ExchangeError as e:
            log_error(f"TSL Update: Failed to format potential TSL price {potential_new_tsl_price}: {e}")
            return

        # Check if better than current effective SL and valid relative to current price
        if (
            position["status"] == "long"
            and new_tsl_price_fmt > current_effective_sl_price
            and new_tsl_price_fmt < current_price
        ):
            should_update_tsl = True
        elif (
            position["status"] == "short"
            and new_tsl_price_fmt < current_effective_sl_price
            and new_tsl_price_fmt > current_price
        ):
            should_update_tsl = True

        if should_update_tsl:
            log_debug(
                f"TSL Improvement Check OK: New ({new_tsl_price_fmt:.{price_precision_digits}f}) vs Current Eff. SL ({current_effective_sl_price:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f})"
            )
        else:
            log_debug(
                f"TSL Improvement Check Failed: New ({new_tsl_price_fmt:.{price_precision_digits}f}) vs Current Eff. SL ({current_effective_sl_price:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f})"
            )

    # --- Execute TSL Update (Cancel Old, Place New - Handles OCO or Separate) ---
    if should_update_tsl and new_tsl_price_fmt is not None:
        log_info(
            f"{NEON_YELLOW}Trailing Stop Update! Moving SL from ~{current_effective_sl_price:.{price_precision_digits}f} to {new_tsl_price_fmt:.{price_precision_digits}f}{RESET}"
        )
        old_order_id = active_sl_order_id  # ID to cancel (could be OCO or SL)
        original_tp_price = position.get("take_profit")  # Keep original TP target
        current_qty = position.get("quantity")

        if current_qty is None or original_tp_price is None:
            log_error("TSL Update Error: Missing quantity or original TP price in state.")
            return

        try:
            # --- Step 1: Cancel the existing SL/OCO order ---
            log_info(f"TSL Update: Cancelling old SL/OCO order {old_order_id}...")
            cancel_success = cancel_order_with_retry(exchange_instance, old_order_id, trading_symbol)
            if cancel_success:
                log_info(f"TSL Update: Old order {old_order_id} cancellation successful or order already closed.")
                time.sleep(1.0)
            else:
                log_error(
                    f"TSL Update: Failed to cancel old order {old_order_id}. Aborting TSL update. Position may remain protected by old order."
                )
                return

            # --- Step 2: Place the new protection orders (OCO or Separate) ---
            log_info(
                f"TSL Update: Placing new protection orders with SL at {new_tsl_price_fmt:.{price_precision_digits}f} and original TP at {original_tp_price:.{price_precision_digits}f}"
            )

            # Use place_sl_tp_orders which handles OCO preference
            new_sl_id, new_tp_id, new_oco_id = place_sl_tp_orders(
                exchange_instance=exchange_instance,
                trading_symbol=trading_symbol,
                position_side=position["status"],
                quantity=current_qty,
                sl_price=new_tsl_price_fmt,  # Use the new TSL price
                tp_price=original_tp_price,  # Keep the original TP price
            )

            # --- Step 3: Update State based on new order placement ---
            if new_oco_id or new_sl_id:  # Check if at least SL/OCO part was successful
                log_info(
                    f"{NEON_GREEN}TSL Update: New protection orders placed successfully. New SL Trigger: {new_tsl_price_fmt:.{price_precision_digits}f}{RESET}"
                )
                if new_oco_id:
                    log_info(f"  New OCO Ref ID: {new_oco_id}")
                if new_sl_id:
                    log_info(f"  New SL ID: {new_sl_id}")
                if new_tp_id:
                    log_info(f"  New TP ID: {new_tp_id}")

                # Update state with new IDs and the activated TSL price
                position["sl_order_id"] = new_sl_id
                position["tp_order_id"] = new_tp_id
                position["oco_order_id"] = new_oco_id
                position["stop_loss"] = new_tsl_price_fmt  # Update target SL to TSL price
                position["current_trailing_sl_price"] = new_tsl_price_fmt  # Mark TSL as active
                save_position_state()
                display_position_status(position, price_precision_digits, amount_precision_digits)

                if not new_tp_id and not new_oco_id:  # Warn if separate TP failed
                    log_warning("TSL Update: New SL placed, but separate TP placement failed.")
            else:
                # This means placing new orders failed after cancelling the old one
                log_error(
                    f"TSL Update CRITICAL ERROR: Failed to place new protection orders after cancelling old one {old_order_id}. POSITION IS NOW UNPROTECTED."
                )
                position["sl_order_id"] = None
                position["tp_order_id"] = None
                position["oco_order_id"] = None
                position["current_trailing_sl_price"] = None  # TSL no longer active/valid
                # Keep 'stop_loss' as the target TSL price? Or None? Keep target for info.
                save_position_state()  # Save unprotected state

        except Exception as e:
            # Catch errors during cancel/place process
            log_error(
                f"TSL Update CRITICAL ERROR: Error during TSL update process: {e}. POSITION MAY BE UNPROTECTED.",
                exc_info=True,
            )
            # Try to reflect uncertainty in state
            position["sl_order_id"] = None
            position["tp_order_id"] = None
            position["oco_order_id"] = None
            position["current_trailing_sl_price"] = None
            save_position_state()


# --- Main Trading Loop ---
def run_bot():
    """Main trading loop execution."""
    log_info(f"Initializing trading bot for {symbol} on {timeframe}...")
    load_position_state()  # Load state once at start
    log_info(f"Risk per trade: {risk_percentage * 100:.2f}%")
    log_info(f"Check interval: {sleep_interval_seconds}s")
    log_info(f"ATR SL/TP: {enable_atr_sl_tp}, Trailing Stop: {enable_trailing_stop}")
    log_info(f"{NEON_YELLOW}Press Ctrl+C to stop.{RESET}")

    while True:
        try:
            cycle_start_time: pd.Timestamp = pd.Timestamp.now(tz="UTC")
            print_cycle_divider(cycle_start_time)

            # --- Step 1: Check Position Consistency & Handle Closures ---
            position_was_reset = check_position_and_orders(exchange, symbol)
            display_position_status(position, price_precision_digits, amount_precision_digits)  # Display *after* check

            if position_was_reset:
                log_info("Position state reset by check (SL/TP/Closed). Checking for new signals.")
                continue  # Skip rest of cycle, check for entry immediately

            # --- Step 2: Fetch Fresh Market Data ---
            ohlcv_df = fetch_ohlcv_data(exchange, symbol, timeframe, limit_count=data_limit)
            if ohlcv_df is None or ohlcv_df.empty:
                log_warning(f"No valid OHLCV data for {symbol}. Waiting...")
                neon_sleep_timer(sleep_interval_seconds)
                continue

            # --- Step 3: Calculate Technical Indicators ---
            needs_atr_calc = enable_atr_sl_tp or enable_trailing_stop
            needs_vol_ma_calc = entry_volume_confirmation_enabled
            stoch_params_dict = {"k": stoch_k, "d": stoch_d, "smooth_k": stoch_smooth_k}

            df_indicators = calculate_technical_indicators(
                ohlcv_df,
                rsi_length,
                stoch_params_dict,
                calc_atr=needs_atr_calc,
                atr_len=atr_length,
                calc_vol_ma=needs_vol_ma_calc,
                vol_ma_len=entry_volume_ma_length,
            )
            if df_indicators is None or df_indicators.empty:
                log_warning("Indicator calculation failed or empty DataFrame. Waiting...")
                neon_sleep_timer(sleep_interval_seconds)
                continue

            # --- Step 4: Get Latest Data Point & Values ---
            if len(df_indicators) < 1:
                continue  # Should be handled by calc func, but safety check

            latest_data: pd.Series = df_indicators.iloc[-1]
            latest_timestamp: pd.Timestamp = latest_data.name

            # Dynamic column names
            rsi_col = f"RSI_{rsi_length}"
            stoch_k_col = f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}"
            stoch_d_col = f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}"
            atr_col = f"ATRr_{atr_length}" if needs_atr_calc else None
            vol_ma_col = f"VOL_MA_{entry_volume_ma_length}" if needs_vol_ma_calc else None

            # Extract latest values with robust checks
            try:
                current_price = float(latest_data["close"])
                float(latest_data["high"])
                float(latest_data["low"])
                current_volume = float(latest_data["volume"])
                last_rsi = float(latest_data[rsi_col])
                last_stoch_k = float(latest_data[stoch_k_col])
                last_stoch_d = float(latest_data[stoch_d_col])

                last_atr: Optional[float] = None
                if needs_atr_calc and atr_col and atr_col in latest_data and pd.notna(latest_data[atr_col]):
                    atr_val = float(latest_data[atr_col])
                    if atr_val > 0:
                        last_atr = atr_val
                    else:
                        log_warning(f"Invalid ATR value ({atr_val}) <= 0.")
                elif needs_atr_calc:
                    log_warning("ATR calculation required but column/value missing/NaN.")

                last_volume_ma: Optional[float] = None
                if needs_vol_ma_calc and vol_ma_col and vol_ma_col in latest_data and pd.notna(latest_data[vol_ma_col]):
                    vol_ma_val = float(latest_data[vol_ma_col])
                    if vol_ma_val > 0:
                        last_volume_ma = vol_ma_val
                elif needs_vol_ma_calc:
                    log_debug("Volume MA calculation required but column/value missing/NaN.")

                # Check for NaN in essential values
                if any(math.isnan(v) for v in [current_price, last_rsi, last_stoch_k, last_stoch_d]):
                    raise ValueError("Essential indicator value is NaN.")

            except (KeyError, ValueError, TypeError) as e:
                log_error(
                    f"Error extracting latest data or value invalid at {latest_timestamp}: {e}. Data: {latest_data.to_dict()}",
                    exc_info=False,
                )
                neon_sleep_timer(sleep_interval_seconds)
                continue

            display_market_stats(current_price, last_rsi, last_stoch_k, last_stoch_d, last_atr, price_precision_digits)

            # --- Step 5: Identify Order Blocks ---
            bullish_ob, bearish_ob = identify_potential_order_block(
                df_indicators, ob_volume_threshold_multiplier, ob_lookback
            )
            display_order_blocks(bullish_ob, bearish_ob, price_precision_digits)

            # --- Step 6: Apply Trading Logic ---

            # === A. LOGIC IF IN A POSITION ===
            if position["status"] is not None:
                log_info(
                    f"Monitoring {position['status'].upper()} position entered at {position.get('entry_price', 'N/A'):.{price_precision_digits}f}..."
                )

                # --- A.1: Check Trailing Stop Loss ---
                update_trailing_stop(exchange, symbol, current_price, last_atr)

                # --- A.2: Check Fallback Indicator-Based Exits (Optional) ---
                # Example: Exit if RSI goes strongly against position (crosses back over threshold)
                execute_fallback_exit = False
                fallback_reason = ""
                if position["status"] == "long" and last_rsi > rsi_overbought:
                    fallback_reason = f"Fallback Exit: RSI ({last_rsi:.1f}) > Overbought ({rsi_overbought})"
                    execute_fallback_exit = True
                elif position["status"] == "short" and last_rsi < rsi_oversold:
                    fallback_reason = f"Fallback Exit: RSI ({last_rsi:.1f}) < Oversold ({rsi_oversold})"
                    execute_fallback_exit = True

                if execute_fallback_exit:
                    display_signal("Fallback Exit", position["status"], fallback_reason)
                    log_warning(
                        f"Attempting FALLBACK MARKET EXIT for {position['status'].upper()} position: {fallback_reason}"
                    )

                    exit_qty = position.get("quantity")
                    if exit_qty is None or exit_qty <= 0:
                        log_error("Fallback Exit Error: Invalid quantity in state.")
                    else:
                        # 1. Cancel Existing Protection Orders (OCO or Separate)
                        order_id_to_cancel = (
                            position.get("oco_order_id") or position.get("sl_order_id") or position.get("tp_order_id")
                        )
                        # Also try cancelling the other if separate orders were used
                        other_order_id = (
                            position.get("tp_order_id") if position.get("sl_order_id") else position.get("sl_order_id")
                        )

                        cancellation_ok = True
                        if order_id_to_cancel:
                            log_info(
                                f"Cancelling protection order(s) before fallback exit: {order_id_to_cancel} {f', also {other_order_id}' if other_order_id and not position.get('oco_order_id') else ''}..."
                            )
                            try:
                                if not cancel_order_with_retry(exchange, order_id_to_cancel, symbol):
                                    cancellation_ok = False
                                if other_order_id and not position.get(
                                    "oco_order_id"
                                ):  # Cancel second order if separate
                                    if not cancel_order_with_retry(exchange, other_order_id, symbol):
                                        cancellation_ok = False
                            except Exception as cancel_e:
                                log_error(
                                    f"Error cancelling protection orders during fallback: {cancel_e}", exc_info=False
                                )
                                cancellation_ok = False

                        # 2. Place Market Exit Order (Only if cancellation seemed ok)
                        if cancellation_ok:
                            log_info(
                                "Protection orders cancelled or already closed. Proceeding with fallback market exit."
                            )
                            close_side = "sell" if position["status"] == "long" else "buy"
                            fallback_exit_order = place_market_order(
                                exchange, symbol, close_side, exit_qty, reduce_only=True
                            )

                            if fallback_exit_order and fallback_exit_order.get("id"):
                                log_info(
                                    f"Fallback market exit order processed: ID {fallback_exit_order.get('id', 'N/A')}"
                                )
                                log_info("Resetting local state after fallback exit attempt.")
                                position.update(position_default_structure.copy())
                                save_position_state()
                                display_position_status(position, price_precision_digits, amount_precision_digits)
                                log_info("Proceeding to next cycle after fallback exit.")
                                continue  # Go to next cycle
                            else:
                                log_error(
                                    f"Fallback market order placement FAILED. Result: {fallback_exit_order}. POSITION MAY STILL BE OPEN. MANUAL INTERVENTION REQUIRED."
                                )
                        else:
                            log_error(
                                "Failed to confirm cancellation of protection orders. Aborting fallback market exit. MANUAL REVIEW RECOMMENDED."
                            )
                else:
                    # Normal monitoring message if no fallback exit
                    log_debug(f"Monitoring {position['status'].upper()} position. Waiting for SL/TP/TSL.")

            # === B. LOGIC TO CHECK FOR NEW ENTRIES (Only if position is None) ===
            else:
                log_info("No active position. Checking for entry signals...")

                # --- B.1: Volume Confirmation ---
                vol_confirmed = False
                if entry_volume_confirmation_enabled:
                    if current_volume is not None and last_volume_ma is not None and last_volume_ma > 0:
                        vol_thresh = last_volume_ma * entry_volume_multiplier
                        if current_volume > vol_thresh:
                            vol_confirmed = True
                            log_debug(f"Volume Confirmed: Vol ({current_volume:.2f}) > Threshold ({vol_thresh:.2f})")
                        else:
                            log_debug(
                                f"Volume Not Confirmed: Vol ({current_volume:.2f}) <= Threshold ({vol_thresh:.2f})"
                            )
                    else:
                        log_debug("Volume confirmation skipped (missing data).")
                else:
                    vol_confirmed = True  # Always true if disabled

                # --- B.2: Base Signal Conditions (RSI + Stoch) ---
                base_long_signal = (
                    last_rsi < rsi_oversold and last_stoch_k < stoch_oversold and last_stoch_d < stoch_oversold
                )
                base_short_signal = (
                    last_rsi > rsi_overbought and last_stoch_k > stoch_overbought and last_stoch_d > stoch_overbought
                )

                # --- B.3: Order Block Price Proximity ---
                ob_confirmed = False
                ob_reason = ""
                entry_ob: Optional[Dict] = None

                if base_long_signal and bullish_ob:
                    # Price needs to be within or very near the bullish OB
                    entry_zone_low = bullish_ob["low"]
                    entry_zone_high = bullish_ob[
                        "high"
                    ]  # Maybe allow slight penetration above? Keep simple: within OB.
                    if entry_zone_low <= current_price <= entry_zone_high:
                        ob_confirmed = True
                        ob_reason = f"Price ({current_price:.{price_precision_digits}f}) within Bullish OB [{entry_zone_low:.{price_precision_digits}f}-{entry_zone_high:.{price_precision_digits}f}]"
                        entry_ob = bullish_ob
                        log_debug(f"Bullish OB Price Confirmed: {ob_reason}")
                    else:
                        log_debug(f"Price ({current_price:.{price_precision_digits}f}) outside Bullish OB.")
                elif base_short_signal and bearish_ob:
                    entry_zone_low = bearish_ob["low"]
                    entry_zone_high = bearish_ob["high"]
                    if entry_zone_low <= current_price <= entry_zone_high:
                        ob_confirmed = True
                        ob_reason = f"Price ({current_price:.{price_precision_digits}f}) within Bearish OB [{entry_zone_low:.{price_precision_digits}f}-{entry_zone_high:.{price_precision_digits}f}]"
                        entry_ob = bearish_ob
                        log_debug(f"Bearish OB Price Confirmed: {ob_reason}")
                    else:
                        log_debug(f"Price ({current_price:.{price_precision_digits}f}) outside Bearish OB.")

                # --- B.4: Final Entry Signal ---
                # Requires Base Signal AND OB Confirmation AND Volume Confirmation
                side_to_enter: Optional[str] = None
                entry_reason_full = ""

                if base_long_signal and ob_confirmed and vol_confirmed and entry_ob and entry_ob["type"] == "bullish":
                    side_to_enter = "long"
                    entry_reason_full = (
                        f"RSI({last_rsi:.1f})<OS, Stoch({last_stoch_k:.1f},{last_stoch_d:.1f})<OS, {ob_reason}"
                        + (", Vol Confirmed" if entry_volume_confirmation_enabled else "")
                    )
                elif (
                    base_short_signal and ob_confirmed and vol_confirmed and entry_ob and entry_ob["type"] == "bearish"
                ):
                    side_to_enter = "short"
                    entry_reason_full = (
                        f"RSI({last_rsi:.1f})>OB, Stoch({last_stoch_k:.1f},{last_stoch_d:.1f})>OB, {ob_reason}"
                        + (", Vol Confirmed" if entry_volume_confirmation_enabled else "")
                    )
                else:  # Log reason for no entry if close
                    if base_long_signal or base_short_signal:
                        reasons = []
                        if not ob_confirmed:
                            reasons.append("OB Price Invalid")
                        if not vol_confirmed and entry_volume_confirmation_enabled:
                            reasons.append("Volume Low")
                        if reasons:
                            log_debug(f"Entry Blocked: Base signal met but failed checks: {', '.join(reasons)}")

                # --- B.5: Execute Entry Sequence ---
                if side_to_enter and entry_reason_full and entry_ob:
                    display_signal("Entry", side_to_enter, entry_reason_full)

                    # Calculate SL/TP
                    sl_price_calc: Optional[float] = None
                    tp_price_calc: Optional[float] = None

                    if enable_atr_sl_tp:
                        if last_atr:
                            sl_dist = last_atr * atr_sl_multiplier
                            tp_dist = last_atr * atr_tp_multiplier
                            sl_price_calc = (
                                current_price - sl_dist if side_to_enter == "long" else current_price + sl_dist
                            )
                            tp_price_calc = (
                                current_price + tp_dist if side_to_enter == "long" else current_price - tp_dist
                            )
                            log_info(
                                f"Calculated ATR SL/TP: SL={sl_price_calc:.{price_precision_digits}f}, TP={tp_price_calc:.{price_precision_digits}f}"
                            )
                        else:
                            log_error("Cannot calculate ATR SL/TP: Invalid ATR. Skipping entry.")
                            continue
                    else:  # Fixed %
                        sl_mult = 1 - stop_loss_percentage if side_to_enter == "long" else 1 + stop_loss_percentage
                        tp_mult = 1 + take_profit_percentage if side_to_enter == "long" else 1 - take_profit_percentage
                        sl_price_calc = current_price * sl_mult
                        tp_price_calc = current_price * tp_mult
                        log_info(
                            f"Calculated Fixed % SL/TP: SL={sl_price_calc:.{price_precision_digits}f}, TP={tp_price_calc:.{price_precision_digits}f}"
                        )

                    # Refine SL based on Order Block
                    ob_low = entry_ob["low"]
                    ob_high = entry_ob["high"]
                    adj_sl_price: Optional[float] = None
                    sl_buffer_ticks = min_tick * 5  # Add buffer

                    if side_to_enter == "long":
                        potential_ob_sl = ob_low - sl_buffer_ticks
                        if potential_ob_sl > sl_price_calc and potential_ob_sl < current_price:
                            adj_sl_price = potential_ob_sl
                            log_info(
                                f"Adjusting SL tighter based on Bullish OB low: {adj_sl_price:.{price_precision_digits}f}"
                            )
                    else:  # short
                        potential_ob_sl = ob_high + sl_buffer_ticks
                        if potential_ob_sl < sl_price_calc and potential_ob_sl > current_price:
                            adj_sl_price = potential_ob_sl
                            log_info(
                                f"Adjusting SL tighter based on Bearish OB high: {adj_sl_price:.{price_precision_digits}f}"
                            )

                    # Use adjusted SL if valid and calculated
                    final_sl_price = adj_sl_price if adj_sl_price is not None else sl_price_calc
                    final_tp_price = tp_price_calc

                    # Final Validation and Formatting
                    if final_sl_price is None or final_tp_price is None:
                        log_error("SL or TP price calculation failed. Skipping entry.")
                        continue
                    try:
                        final_sl_price_fmt = float(exchange.price_to_precision(symbol, final_sl_price))
                        final_tp_price_fmt = float(exchange.price_to_precision(symbol, final_tp_price))
                    except (ccxt.ExchangeError, TypeError, ValueError) as fmt_e:
                        log_error(f"Failed to format final SL/TP prices: {fmt_e}. Skipping entry.")
                        continue

                    if (
                        side_to_enter == "long"
                        and (final_sl_price_fmt >= current_price or final_tp_price_fmt <= current_price)
                    ) or (
                        side_to_enter == "short"
                        and (final_sl_price_fmt <= current_price or final_tp_price_fmt >= current_price)
                    ):
                        log_error(
                            f"Invalid final SL/TP for {side_to_enter.upper()}: SL={final_sl_price_fmt}, TP={final_tp_price_fmt}, Price={current_price}. Skipping."
                        )
                        continue

                    # Calculate Size
                    entry_qty = calculate_position_size(
                        exchange, symbol, current_price, final_sl_price_fmt, risk_percentage
                    )
                    if entry_qty is None or entry_qty <= 0:
                        log_error("Failed to calculate valid position size. Skipping entry.")
                        continue

                    # Place Entry Order
                    entry_order_side = "buy" if side_to_enter == "long" else "sell"
                    entry_order = place_market_order(exchange, symbol, entry_order_side, entry_qty, reduce_only=False)

                    # Process Entry Result
                    entry_successful = False
                    entry_order_id = None
                    entry_price_actual = current_price
                    filled_qty_actual = entry_qty

                    if entry_order and entry_order.get("id"):
                        entry_order_id = entry_order["id"]
                        status = entry_order.get("status")
                        avg_price = entry_order.get("average")
                        filled = entry_order.get("filled")

                        if avg_price and avg_price > 0:
                            entry_price_actual = float(avg_price)
                        if filled and filled > 0:
                            filled_qty_actual = float(filled)

                        # Consider success if closed or open with filled qty (market orders)
                        if status == "closed" or (status == "open" and filled_qty_actual > 0):
                            entry_successful = True
                            log_info(
                                f"Entry order processed: ID {entry_order_id}, AvgPrice: {entry_price_actual:.{price_precision_digits}f}, Filled: {filled_qty_actual:.{amount_precision_digits}f}"
                            )
                        else:
                            log_error(
                                f"Entry order {entry_order_id} status uncertain ('{status}', Filled: {filled}). Assuming failure."
                            )
                    else:
                        log_error(f"Entry market order placement failed. Result: {entry_order}")

                    # Place SL/TP and Update State ONLY if Entry Successful
                    if entry_successful:
                        log_info(f"Entry successful. Placing protection orders for position {entry_order_id}...")
                        sl_id, tp_id, oco_id = place_sl_tp_orders(
                            exchange, symbol, side_to_enter, filled_qty_actual, final_sl_price_fmt, final_tp_price_fmt
                        )

                        # Update Position State
                        if oco_id or sl_id:  # Check if at least SL/OCO placed
                            position.update(
                                {
                                    "status": side_to_enter,
                                    "entry_price": entry_price_actual,
                                    "quantity": filled_qty_actual,
                                    "order_id": entry_order_id,
                                    "stop_loss": final_sl_price_fmt,
                                    "take_profit": final_tp_price_fmt,
                                    "entry_time": pd.Timestamp.now(tz="UTC"),
                                    "sl_order_id": sl_id,
                                    "tp_order_id": tp_id,
                                    "oco_order_id": oco_id,
                                    "highest_price_since_entry": entry_price_actual,
                                    "lowest_price_since_entry": entry_price_actual,
                                    "current_trailing_sl_price": None,
                                }
                            )
                            save_position_state()
                            display_position_status(position, price_precision_digits, amount_precision_digits)
                            # Warnings if part of protection failed
                            if not oco_id:
                                if not sl_id:
                                    log_error(
                                        "Entry successful, but separate SL placement FAILED. POSITION UNPROTECTED."
                                    )
                                if not tp_id:
                                    log_warning("Entry successful, but separate TP placement FAILED.")
                            log_info(f"Position {entry_order_id} opened and state updated.")
                        else:
                            # Entry was successful, but SL/OCO placement failed! Critical.
                            log_error(
                                "CRITICAL: Entry order filled, but FAILED to place SL/OCO protection orders. Attempting to close position immediately."
                            )
                            # Attempt emergency market close
                            emergency_close_side = "sell" if side_to_enter == "long" else "buy"
                            emergency_close_order = place_market_order(
                                exchange, symbol, emergency_close_side, filled_qty_actual, reduce_only=True
                            )
                            if emergency_close_order and emergency_close_order.get("id"):
                                log_warning(
                                    f"Emergency market close order placed (ID: {emergency_close_order.get('id')}). State not updated to reflect this failed entry."
                                )
                            else:
                                log_error(
                                    "!!! EMERGENCY MARKET CLOSE FAILED !!! MANUAL INTERVENTION REQUIRED IMMEDIATELY !!!"
                                )
                            # Do NOT update position state to 'active' as it's unprotected or being closed.

                    else:  # Entry failed
                        log_error("Entry attempt failed. Position state not updated.")
                        # TODO: Consider cancelling the failed/pending entry order ID if it exists?

                else:  # No entry signal
                    log_info("Entry conditions not met.")

            # --- Step 7: Wait for the next cycle ---
            log_info(
                f"Cycle complete ({pd.Timestamp.now(tz='UTC') - cycle_start_time}). Waiting {sleep_interval_seconds}s..."
            )
            neon_sleep_timer(sleep_interval_seconds)

        # --- Graceful Shutdown ---
        except KeyboardInterrupt:
            log_info("Keyboard interrupt detected. Shutting down...")
            save_position_state()

            if not SIMULATION_MODE:
                log_warning(f"!!! LIVE MODE: Attempting to cancel all open orders for {symbol} on shutdown...")
                try:

                    @api_retry_decorator  # Decorate inner fetch too
                    def fetch_open_orders_exit(exch, sym):
                        return exch.fetch_open_orders(sym)

                    open_orders = fetch_open_orders_exit(exchange, symbol)

                    if not open_orders:
                        log_info("No open orders found to cancel.")
                    else:
                        log_warning(f"Found {len(open_orders)} open orders. Cancelling...")
                        cancelled, failed = 0, 0
                        for order in open_orders:
                            o_id = order.get("id")
                            if not o_id:
                                continue
                            try:
                                log_info(
                                    f"Cancelling order ID: {o_id} ({order.get('type', '?')}/{order.get('side', '?')})..."
                                )
                                if cancel_order_with_retry(exchange, o_id, symbol):
                                    cancelled += 1
                                else:
                                    failed += 1  # cancel_order_with_retry returned False unexpectedly
                                time.sleep(0.3)
                            except Exception as cancel_e:
                                log_error(f"Failed to cancel order {o_id}: {cancel_e}", exc_info=False)
                                failed += 1
                        log_info(f"Shutdown cancellation: Success/Closed={cancelled}, Failed={failed}")
                        if failed > 0:
                            log_error("Manual check of open orders recommended.")

                except Exception as e:
                    log_error(f"Error fetching/cancelling orders on exit: {e}", exc_info=True)
                    log_error("Manual check of open orders strongly recommended.")
            else:
                log_info("Simulation mode: Skipping order cancellation on exit.")

            break  # Exit while loop

        # --- Main Loop Error Handling ---
        except ccxt.RateLimitExceeded as e:
            log_error(f"Rate Limit Exceeded: {e}. Waiting longer...", exc_info=False)
            neon_sleep_timer(sleep_interval_seconds + 60)
        except ccxt.NetworkError as e:
            log_error(f"Network Error: {e}. Relying on retries within next cycle.", exc_info=False)
            neon_sleep_timer(sleep_interval_seconds)
        except ccxt.ExchangeError as e:
            log_error(f"Exchange Error: {e}. Waiting and retrying...", exc_info=False)
            neon_sleep_timer(sleep_interval_seconds)
        except Exception as e:
            log_error(f"!!! CRITICAL UNEXPECTED ERROR in main loop: {type(e).__name__}: {e} !!!", exc_info=True)
            log_info("Attempting to save state and wait 60s before continuing...")
            try:
                save_position_state()
            except Exception as save_e:
                log_error(f"Failed to save state during error: {save_e}")
            neon_sleep_timer(60)

    # --- Bot Exit ---
    print_shutdown_message()
    log_info("Bot shutdown complete.")


# --- Script Execution ---
if __name__ == "__main__":
    try:
        run_bot()
    except SystemExit as e:
        log_info(f"Bot exited via SystemExit (Code: {e.code}).")
        sys.exit(e.code)
    except Exception as main_exec_e:
        log_error(f"A critical error occurred during bot initialization: {main_exec_e}", exc_info=True)
        print_shutdown_message()
        sys.exit(1)
    finally:
        logging.shutdown()
