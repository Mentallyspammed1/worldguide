# trading_bot_neon_enhanced_v2.py
# Enhanced version incorporating ATR SL/TP, Trailing Stops, Config File,
# Volume Confirmation, Retry Logic, and improved structure.

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
    print(f"{NEON_PINK}{Style.BRIGHT}     Enhanced RSI/OB Trader Neon Bot - Configurable v2     {RESET}")
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
    # Show first line in box for prominence, log full message
    first_line = msg.split("\n", 1)[0]
    display_error_box(first_line)
    logger.error(f"{NEON_RED}{msg}{RESET}", exc_info=exc_info)


def log_warning(msg: str) -> None:
    """Logs a WARNING level message with a neon yellow box and color."""
    display_warning_box(msg)
    logger.warning(f"{NEON_YELLOW}{msg}{RESET}")


def log_debug(msg: str) -> None:
    """Logs a DEBUG level message with simple white color."""
    # Use a less prominent color for debug
    logger.debug(f"{Fore.WHITE}{msg}{RESET}")  # Simple white for debug


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
    sl: Optional[float] = position.get("stop_loss")
    tp: Optional[float] = position.get("take_profit")
    tsl: Optional[float] = position.get("current_trailing_sl_price")  # Added TSL display

    entry_str: str = f"{entry_price:.{price_precision}f}" if isinstance(entry_price, (float, int)) else "N/A"
    qty_str: str = f"{quantity:.{amount_precision}f}" if isinstance(quantity, (float, int)) else "N/A"
    sl_str: str = f"{sl:.{price_precision}f}" if isinstance(sl, (float, int)) else "N/A"
    tp_str: str = f"{tp:.{price_precision}f}" if isinstance(tp, (float, int)) else "N/A"
    tsl_str: str = (
        f" | TSL: {tsl:.{price_precision}f}" if isinstance(tsl, (float, int)) else ""
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
        f"{color}Position Status: {status_text}{RESET} | Entry: {entry_str} | Qty: {qty_str} | SL: {sl_str} | TP: {tp_str}{tsl_str}"
    )


def display_market_stats(
    current_price: float, rsi: float, stoch_k: float, stoch_d: float, atr: Optional[float], price_precision: int
) -> None:
    """Displays market stats in a neon-styled panel."""
    print(f"{NEON_PINK}--- Market Stats ---{RESET}")
    print(f"{NEON_GREEN}Price:{RESET}  {current_price:.{price_precision}f}")
    print(f"{NEON_CYAN}RSI:{RESET}    {rsi:.2f}")
    print(f"{NEON_YELLOW}StochK:{RESET} {stoch_k:.2f}")
    print(f"{NEON_YELLOW}StochD:{RESET} {stoch_d:.2f}")
    # Use price_precision for ATR display as it's a price volatility measure
    if atr is not None:
        print(f"{NEON_BLUE}ATR:{RESET}    {atr:.{price_precision}f}")  # Display ATR
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
    if not COLORAMA_AVAILABLE or seconds <= 0:  # Fallback if colorama not installed or zero sleep
        if seconds > 0:
            print(f"Sleeping for {seconds} seconds...")
            time.sleep(seconds)
        return

    interval: float = 0.5  # Update interval for the timer display
    steps: int = int(seconds / interval)
    for i in range(steps, -1, -1):
        remaining_seconds: int = max(0, int(i * interval))  # Ensure non-negative
        # Flashing effect for last 5 seconds
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
POSITION_STATE_FILE: str = "position_state.json"  # Define filename for state persistence
CONFIG_FILE: str = "config.json"

# --- Retry Decorator ---
# Define exceptions that are generally safe to retry for network/temporary issues
RETRYABLE_EXCEPTIONS: Tuple[type[ccxt.NetworkError], ...] = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.RequestTimeout,
    # Consider adding ccxt.DDoSProtection if frequently encountered and temporary
)


def retry_api_call(max_retries: int = 3, initial_delay: float = 5.0, backoff_factor: float = 2.0) -> Callable:
    """
    Decorator factory to create retry decorators for API calls.
    Handles specific RETRYABLE_EXCEPTIONS with exponential backoff.

    Args:
        max_retries: Maximum number of retries before giving up.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Multiplier for the delay after each retry (e.g., 2 for exponential).

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries: int = 0
            delay: float = initial_delay
            last_exception: Optional[Exception] = None

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    retries += 1
                    last_exception = e
                    if retries >= max_retries:
                        log_error(
                            f"API call '{func.__name__}' failed after {max_retries} retries. Last error: {type(e).__name__}: {e}",
                            exc_info=False,
                        )
                        raise last_exception  # Re-raise the last exception to signal failure
                    else:
                        log_warning(
                            f"API call '{func.__name__}' failed with {type(e).__name__}: {e}. Retrying in {delay:.1f}s... (Attempt {retries}/{max_retries})"
                        )
                        # Use neon sleep if available, otherwise time.sleep
                        try:
                            neon_sleep_timer(int(delay))
                        except NameError:  # Fallback if neon_sleep_timer isn't defined (e.g., colorama issue)
                            time.sleep(delay)
                        delay *= backoff_factor  # Exponential backoff
                except Exception as e:
                    # Handle non-retryable exceptions immediately
                    log_error(
                        f"Non-retryable error in API call '{func.__name__}': {type(e).__name__}: {e}", exc_info=True
                    )
                    raise  # Re-raise immediately

            # This should only be reached if max_retries is 0 or logic error
            if last_exception:
                raise last_exception  # Ensure the last error is raised if loop finishes
            else:
                # Should ideally not happen if max_retries > 0
                raise RuntimeError(f"API call '{func.__name__}' failed unexpectedly after retries.")

        return wrapper

    return decorator


# --- Configuration Loading ---
def load_config(filename: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file. Performs basic validation.
    Exits the script if critical errors occur during loading.

    Args:
        filename: The path to the configuration file.

    Returns:
        A dictionary containing the configuration parameters.

    Raises:
        SystemExit: If the config file is missing, invalid, or lacks required keys.
    """
    log_info(f"Attempting to load configuration from '{filename}'...")
    try:
        with open(filename, "r") as f:
            config_data: Dict[str, Any] = json.load(f)
        log_info(f"Configuration loaded successfully from {filename}")

        # --- Basic Validation ---
        required_keys: List[str] = [
            "exchange_id",
            "symbol",
            "timeframe",
            "risk_percentage",
            "simulation_mode",
            "rsi_length",
            "rsi_overbought",
            "rsi_oversold",
            "stoch_k",
            "stoch_d",
            "stoch_smooth_k",
            "stoch_overbought",
            "stoch_oversold",
            "data_limit",
            "sleep_interval_seconds",
            "ob_volume_threshold_multiplier",
            "ob_lookback",
            "entry_volume_confirmation_enabled",
            # Conditional keys are checked later where used
        ]
        missing_keys: List[str] = [key for key in required_keys if key not in config_data]
        if missing_keys:
            log_error(f"CRITICAL: Missing required configuration keys in '{filename}': {missing_keys}")
            sys.exit(1)

        # --- More Specific Validation (Examples - uncomment/expand as needed) ---
        # TODO: Add more comprehensive type and range validation based on requirements
        if not isinstance(config_data["risk_percentage"], (float, int)) or not (0 < config_data["risk_percentage"] < 1):
            log_error(
                f"CRITICAL: 'risk_percentage' ({config_data['risk_percentage']}) must be a number between 0 and 1 (exclusive)."
            )
            sys.exit(1)
        if not isinstance(config_data["sleep_interval_seconds"], int) or config_data["sleep_interval_seconds"] <= 0:
            log_error(
                f"CRITICAL: 'sleep_interval_seconds' ({config_data['sleep_interval_seconds']}) must be a positive integer."
            )
            sys.exit(1)
        # Example for conditional validation
        # if config_data.get("enable_atr_sl_tp"):
        #    if not isinstance(config_data.get("atr_sl_multiplier"), (float, int)) or config_data["atr_sl_multiplier"] <= 0:
        #         log_error("CRITICAL: 'atr_sl_multiplier' must be a positive number when enable_atr_sl_tp is true.")
        #         sys.exit(1)

        return config_data

    except FileNotFoundError:
        log_error(f"CRITICAL: Configuration file '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_error(f"CRITICAL: Error decoding JSON from configuration file '{filename}': {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"CRITICAL: An unexpected error occurred loading configuration: {e}", exc_info=True)
        sys.exit(1)


# Load config early
config: Dict[str, Any] = load_config()

# Create the decorator instance using config values
api_retry_decorator: Callable = retry_api_call(
    max_retries=config.get("retry_max_retries", 3),
    initial_delay=config.get("retry_initial_delay", 5.0),
    backoff_factor=config.get("retry_backoff_factor", 2.0),
)

# --- Environment & Exchange Setup ---
print_neon_header()  # Show header early
load_dotenv()
log_info("Attempting to load environment variables from .env file (for API keys)...")

exchange_id: str = config.get("exchange_id", "bybit").lower()
api_key_env_var: str = f"{exchange_id.upper()}_API_KEY"
secret_key_env_var: str = f"{exchange_id.upper()}_SECRET_KEY"
passphrase_env_var: str = f"{exchange_id.upper()}_PASSPHRASE"  # Some exchanges like kucoin, okx use this

api_key: Optional[str] = os.getenv(api_key_env_var)
secret: Optional[str] = os.getenv(secret_key_env_var)
passphrase: Optional[str] = os.getenv(passphrase_env_var)  # Keep as Optional[str]

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
        "enableRateLimit": True,  # CCXT internal rate limiting
        "options": {
            # Determine defaultType more dynamically
            "defaultType": "swap"
            if ":" in config.get("symbol", "") or "SWAP" in config.get("symbol", "").upper()
            else "spot",
            "adjustForTimeDifference": True,  # Recommended for avoiding timestamp issues
        },
    }
    if passphrase:
        log_info("Passphrase detected, adding to exchange configuration.")
        exchange_config["password"] = passphrase  # CCXT uses 'password' field for passphrase

    exchange = exchange_class(exchange_config)

    # Load markets with retry mechanism applied
    @api_retry_decorator
    def load_markets_with_retry(exch_instance: ccxt.Exchange) -> None:
        """Loads exchange markets with retry logic."""
        log_info("Loading markets...")
        exch_instance.load_markets()
        log_info("Markets loaded.")

    load_markets_with_retry(exchange)

    log_info(
        f"Successfully connected to {exchange_id}. Markets loaded ({len(exchange.markets)} symbols found). Default type: {exchange.options.get('defaultType')}"
    )

except ccxt.AuthenticationError as e:
    log_error(f"Authentication failed connecting to {exchange_id}. Check API Key/Secret/Passphrase. Error: {e}")
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    log_error(f"Exchange {exchange_id} is not available or timed out during initial connection. Error: {e}")
    sys.exit(1)
except AttributeError:
    log_error(f"Exchange ID '{exchange_id}' not found in ccxt library.")
    sys.exit(1)
except Exception as e:
    # Check if it's a retryable error that failed all retries (already logged by decorator)
    if not isinstance(e, RETRYABLE_EXCEPTIONS):
        log_error(f"An unexpected error occurred during exchange initialization or market loading: {e}", exc_info=True)
    sys.exit(1)  # Exit if any error occurs during setup


# --- Trading Parameters (Loaded from config with validation) ---
symbol: str = config.get("symbol", "").strip().upper()
if not symbol:
    log_error("CRITICAL: 'symbol' not specified in config.json")
    sys.exit(1)

# Validate symbol against exchange markets and get precision info
market_info: Optional[Dict[str, Any]] = None
price_precision_digits: int = DEFAULT_PRICE_PRECISION
amount_precision_digits: int = DEFAULT_AMOUNT_PRECISION
min_tick: float = 1 / (10**DEFAULT_PRICE_PRECISION)  # Default fallback

try:
    if symbol not in exchange.markets:
        log_warning(f"Symbol '{symbol}' from config not found or not supported on {exchange_id}.")
        available_symbols: List[str] = list(exchange.markets.keys())
        # Filter for potential swap markets if needed based on symbol format or default type
        if exchange.options.get("defaultType") == "swap":
            available_symbols = [s for s in available_symbols if ":" in s or exchange.markets[s].get("swap")]
        log_info(f"Some available symbols ({len(available_symbols)} total): {available_symbols[:15]}...")
        sys.exit(1)

    log_info(f"Using trading symbol from config: {symbol}")
    # Store precision info globally for convenience
    market_info = exchange.markets[symbol]

    # Get precision digits robustly
    price_precision_digits = int(market_info.get("precision", {}).get("price", DEFAULT_PRICE_PRECISION))
    amount_precision_digits = int(market_info.get("precision", {}).get("amount", DEFAULT_AMOUNT_PRECISION))

    # Calculate min_tick robustly from precision value if possible
    min_tick_value_str: Optional[str] = market_info.get("precision", {}).get("price")
    if min_tick_value_str is not None:
        try:
            min_tick = float(min_tick_value_str)  # Use precision value directly if available
            if min_tick <= 0:  # Handle cases where precision might be 0 or negative exponent string
                min_tick = 1 / (10**price_precision_digits) if price_precision_digits >= 0 else 0.01
        except (ValueError, TypeError):
            log_warning(
                f"Could not parse 'precision.price' value '{min_tick_value_str}' to float. Using default min_tick calculation."
            )
            min_tick = 1 / (10**price_precision_digits) if price_precision_digits >= 0 else 0.01
    else:
        min_tick = 1 / (10**price_precision_digits) if price_precision_digits >= 0 else 0.01  # Fallback

    log_info(
        f"Symbol Precision | Price: {price_precision_digits} decimals (Min Tick: {min_tick:.{price_precision_digits + 2}f}), Amount: {amount_precision_digits} decimals"
    )

except Exception as e:
    log_error(f"An error occurred while validating the symbol or getting precision: {e}", exc_info=True)
    sys.exit(1)

timeframe: str = config.get("timeframe", "1h")
rsi_length: int = int(config["rsi_length"])
rsi_overbought: int = int(config["rsi_overbought"])
rsi_oversold: int = int(config["rsi_oversold"])
stoch_k: int = int(config["stoch_k"])
stoch_d: int = int(config["stoch_d"])
stoch_smooth_k: int = int(config["stoch_smooth_k"])
stoch_overbought: int = int(config["stoch_overbought"])
stoch_oversold: int = int(config["stoch_oversold"])
data_limit: int = int(config["data_limit"])
sleep_interval_seconds: int = int(config["sleep_interval_seconds"])  # Already validated > 0
risk_percentage: float = float(config["risk_percentage"])  # Already validated 0 < risk < 1

# SL/TP and Trailing Stop Parameters (Conditional loading and validation)
enable_atr_sl_tp: bool = config.get("enable_atr_sl_tp", False)
enable_trailing_stop: bool = config.get("enable_trailing_stop", False)
atr_length: int = int(config.get("atr_length", 14))  # Needed if either ATR SL/TP or TSL is enabled

needs_atr: bool = enable_atr_sl_tp or enable_trailing_stop
if needs_atr:
    if not isinstance(atr_length, int) or atr_length <= 0:
        log_error(
            f"CRITICAL: 'atr_length' ({atr_length}) must be a positive integer if ATR SL/TP or Trailing Stop is enabled."
        )
        sys.exit(1)

# Initialize these vars outside the conditional blocks
atr_sl_multiplier: float = 0.0
atr_tp_multiplier: float = 0.0
stop_loss_percentage: float = 0.0
take_profit_percentage: float = 0.0

if enable_atr_sl_tp:
    atr_sl_multiplier = float(config.get("atr_sl_multiplier", 2.0))
    atr_tp_multiplier = float(config.get("atr_tp_multiplier", 3.0))
    log_info(f"Using ATR-based Stop Loss ({atr_sl_multiplier}x ATR) and Take Profit ({atr_tp_multiplier}x ATR).")
    if (
        not isinstance(atr_sl_multiplier, (float, int))
        or atr_sl_multiplier <= 0
        or not isinstance(atr_tp_multiplier, (float, int))
        or atr_tp_multiplier <= 0
    ):
        log_error("CRITICAL: ATR SL/TP multipliers must be positive numbers.")
        sys.exit(1)
else:
    stop_loss_percentage = float(config.get("stop_loss_percentage", 0.02))
    take_profit_percentage = float(config.get("take_profit_percentage", 0.04))
    log_info(
        f"Using Fixed Percentage Stop Loss ({stop_loss_percentage * 100:.1f}%) and Take Profit ({take_profit_percentage * 100:.1f}%)."
    )
    if (
        not isinstance(stop_loss_percentage, (float, int))
        or stop_loss_percentage <= 0
        or not isinstance(take_profit_percentage, (float, int))
        or take_profit_percentage <= 0
    ):
        log_error("CRITICAL: Fixed SL/TP percentages must be positive numbers.")
        sys.exit(1)

# Initialize TSL params outside conditional
trailing_stop_atr_multiplier: float = 0.0
trailing_stop_activation_atr_multiplier: float = 0.0

if enable_trailing_stop:
    trailing_stop_atr_multiplier = float(config.get("trailing_stop_atr_multiplier", 1.5))
    trailing_stop_activation_atr_multiplier = float(config.get("trailing_stop_activation_atr_multiplier", 1.0))
    log_info(
        f"Trailing Stop Loss is ENABLED (Activate @ {trailing_stop_activation_atr_multiplier}x ATR profit, Trail @ {trailing_stop_atr_multiplier}x ATR)."
    )
    if (
        not isinstance(trailing_stop_atr_multiplier, (float, int))
        or trailing_stop_atr_multiplier <= 0
        or not isinstance(trailing_stop_activation_atr_multiplier, (float, int))
        or trailing_stop_activation_atr_multiplier < 0
    ):  # Activation can be 0
        log_error(
            "CRITICAL: Trailing stop ATR multiplier must be positive number, activation multiplier non-negative number."
        )
        sys.exit(1)
else:
    log_info("Trailing Stop Loss is DISABLED.")

# OB Parameters
ob_volume_threshold_multiplier: float = float(config["ob_volume_threshold_multiplier"])
ob_lookback: int = int(config["ob_lookback"])
if not isinstance(ob_lookback, int) or ob_lookback <= 0:
    log_error("CRITICAL: 'ob_lookback' must be a positive integer.")
    sys.exit(1)
if not isinstance(ob_volume_threshold_multiplier, (float, int)) or ob_volume_threshold_multiplier <= 0:
    log_error("CRITICAL: 'ob_volume_threshold_multiplier' must be a positive number.")
    sys.exit(1)

# Volume Confirmation Parameters
entry_volume_confirmation_enabled: bool = config["entry_volume_confirmation_enabled"]
entry_volume_ma_length: int = int(config.get("entry_volume_ma_length", 20))
entry_volume_multiplier: float = float(config.get("entry_volume_multiplier", 1.2))
if entry_volume_confirmation_enabled:
    log_info(f"Entry Volume Confirmation: ENABLED (Vol > {entry_volume_multiplier}x MA({entry_volume_ma_length}))")
    if (
        not isinstance(entry_volume_ma_length, int)
        or entry_volume_ma_length <= 0
        or not isinstance(entry_volume_multiplier, (float, int))
        or entry_volume_multiplier <= 0
    ):
        log_error(
            "CRITICAL: Volume MA length and multiplier must be positive numbers when volume confirmation is enabled."
        )
        sys.exit(1)
else:
    log_info("Entry Volume Confirmation: DISABLED")

# --- Simulation Mode ---
SIMULATION_MODE: bool = config["simulation_mode"]
if SIMULATION_MODE:
    log_warning("SIMULATION MODE IS ACTIVE (set in config.json). No real orders will be placed.")
else:
    log_warning("!!! LIVE TRADING MODE IS ACTIVE (set in config.json). REAL ORDERS WILL BE PLACED. !!!")
    # Add confirmation step for live trading
    try:
        # Ask for confirmation only if running interactively
        if sys.stdin.isatty():
            user_confirm = input(f"{NEON_RED}TYPE 'LIVE' TO CONFIRM LIVE TRADING or press Enter to exit: {RESET}")
            if user_confirm.strip().upper() != "LIVE":
                log_info("Live trading not confirmed. Exiting.")
                sys.exit(0)
            log_info("Live trading confirmed by user.")
        else:
            log_warning("Running in non-interactive mode. Assuming confirmation for LIVE TRADING.")
            # Mandatory delay in non-interactive live mode for safety
            log_warning("Pausing for 5 seconds before starting live trading...")
            time.sleep(5)  # Short pause to allow cancellation if started accidentally

    except EOFError:  # Handle environments where input is not possible (e.g., docker without -it)
        log_error("Cannot get user confirmation in this environment. Exiting live mode for safety.")
        sys.exit(1)


# --- Position Management State (Includes Trailing Stop fields) ---
# Define the structure clearly for type checking and default initialization
position_default_structure: Dict[str, Any] = {
    "status": None,  # None | 'long' | 'short'
    "entry_price": None,  # Optional[float]
    "quantity": None,  # Optional[float]
    "order_id": None,  # Optional[str] (ID of the entry order)
    "stop_loss": None,  # Optional[float] (Initial or last manually set SL price)
    "take_profit": None,  # Optional[float] (Price level)
    "entry_time": None,  # Optional[pd.Timestamp] (timezone-aware UTC)
    "sl_order_id": None,  # Optional[str] (ID of the open SL order)
    "tp_order_id": None,  # Optional[str] (ID of the open TP order)
    # Fields for Trailing Stop Loss
    "highest_price_since_entry": None,  # Optional[float]: For long positions
    "lowest_price_since_entry": None,  # Optional[float]: For short positions
    "current_trailing_sl_price": None,  # Optional[float]: Active trailing SL price
}
# Initialize global position state using a copy of the default structure
position: Dict[str, Any] = position_default_structure.copy()


# --- State Saving and Resumption Functions ---
def save_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """
    Saves the current position state dictionary to a JSON file.
    Converts pandas Timestamp to ISO 8601 string format for serialization.

    Args:
        filename: The path to the JSON file to save the state.
    """
    global position
    try:
        # Create a copy to modify for serialization without affecting the live state object
        state_to_save = position.copy()
        entry_time = state_to_save.get("entry_time")

        # Convert Timestamp to ISO string if it exists
        if isinstance(entry_time, pd.Timestamp):
            # Ensure timezone information is handled correctly (ISO format preserves it)
            state_to_save["entry_time"] = entry_time.isoformat()

        with open(filename, "w") as f:
            json.dump(state_to_save, f, indent=4)
        log_debug(f"Position state saved successfully to {filename}")  # Use debug for less noise

    except TypeError as e:
        log_error(
            f"Serialization error saving position state to {filename}. Check data types. Error: {e}", exc_info=True
        )
    except IOError as e:
        log_error(f"I/O error saving position state to {filename}: {e}", exc_info=True)
    except Exception as e:
        log_error(f"Unexpected error saving position state to {filename}: {e}", exc_info=True)


def load_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """
    Loads position state from a JSON file if it exists.
    Handles missing keys by using defaults from `position_default_structure`.
    Converts saved ISO timestamp strings back to timezone-aware pandas Timestamps (UTC).
    Validates data types against the default structure.

    Args:
        filename: The path to the JSON file to load the state from.
    """
    global position
    log_info(f"Attempting to load position state from '{filename}'...")
    # Start with the default structure to ensure all keys exist
    position = position_default_structure.copy()

    if not os.path.exists(filename):
        log_info(f"No position state file found at {filename}. Starting with default empty state.")
        return  # Exit function, position is already default

    try:
        with open(filename, "r") as f:
            loaded_state: Dict[str, Any] = json.load(f)

        # --- Data Validation and Type Conversion ---
        parsed_entry_time: Optional[pd.Timestamp] = None
        entry_time_str: Optional[str] = loaded_state.get("entry_time")
        if entry_time_str and isinstance(entry_time_str, str):
            try:
                # Attempt parsing ISO format string, which includes timezone if saved correctly
                ts = pd.Timestamp(entry_time_str)
                # Ensure it's timezone-aware and converted to UTC for consistency
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")  # Assume UTC if no tz info found (less ideal)
                else:
                    ts = ts.tz_convert("UTC")  # Convert to UTC if it had other tz info
                parsed_entry_time = ts
            except ValueError:
                log_error(f"Could not parse entry_time '{entry_time_str}' from state file. Setting entry_time to None.")
                # parsed_entry_time remains None

        # --- Update the global `position` dict, respecting the default structure ---
        updated_count = 0
        missing_in_file = []
        extra_in_file = []
        type_mismatches = []

        current_keys = set(position.keys())  # Keys from default structure
        loaded_keys = set(loaded_state.keys())

        for key, default_value in position_default_structure.items():
            if key in loaded_keys:
                loaded_value = loaded_state[key]

                # Special handling for entry_time (use parsed value)
                if key == "entry_time":
                    loaded_value = parsed_entry_time

                # Check type consistency (allow None values)
                # Allow int to be loaded into float fields
                expected_type = type(default_value)
                is_type_match = isinstance(loaded_value, expected_type)
                allow_int_for_float = expected_type is float and isinstance(loaded_value, int)

                if (
                    loaded_value is not None
                    and default_value is not None
                    and not is_type_match
                    and not allow_int_for_float
                ):
                    type_mismatches.append(
                        f"'{key}' (Expected {expected_type.__name__}, Got {type(loaded_value).__name__})"
                    )
                    # Keep the default value from `position` if type mismatch
                else:
                    # Assign loaded value (potentially converting int to float)
                    position[key] = float(loaded_value) if allow_int_for_float else loaded_value
                    if (
                        key != "entry_time" or parsed_entry_time is not None
                    ):  # Count update unless entry_time parse failed
                        updated_count += 1
            else:
                missing_in_file.append(key)
                # Keep the default value already set in `position`

        extra_in_file = list(loaded_keys - current_keys)

        # Log warnings about discrepancies
        if missing_in_file:
            log_warning(f"Keys missing in state file '{filename}' (using defaults): {missing_in_file}")
        if extra_in_file:
            log_warning(f"Extra keys found in state file '{filename}' (ignored): {extra_in_file}")
        if type_mismatches:
            log_warning(f"Type mismatches found in state file (using defaults for these): {type_mismatches}")

        if updated_count > 0 or not missing_in_file:
            log_info(f"Position state loaded successfully from {filename}. Applied {updated_count} values.")
            display_position_status(position, price_precision_digits, amount_precision_digits)  # Display loaded state
        else:
            # This case might happen if the file exists but is empty or completely invalid
            log_warning(
                f"Loaded state file {filename}, but applied 0 values due to missing keys or type mismatches. Using default state."
            )
            position = position_default_structure.copy()  # Ensure reset to default

    except json.JSONDecodeError as e:
        log_error(
            f"Error decoding JSON from state file {filename}: {e}. Starting with default empty state.", exc_info=False
        )
        position = position_default_structure.copy()  # Reset to default
    except Exception as e:
        log_error(
            f"Error loading position state from {filename}: {e}. Starting with default empty state.", exc_info=True
        )
        position = position_default_structure.copy()  # Reset to default


# --- Data Fetching Function (Decorated) ---
@api_retry_decorator
def fetch_ohlcv_data(
    exchange_instance: ccxt.Exchange, trading_symbol: str, tf: str, limit_count: int
) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV data using ccxt, converts to a pandas DataFrame, cleanses,
    sets a timezone-aware UTC index, and returns it. Retries on common network/API errors.

    Args:
        exchange_instance: The initialized ccxt exchange instance.
        trading_symbol: The symbol to fetch data for (e.g., 'BTC/USDT').
        tf: The timeframe string (e.g., '1h', '15m').
        limit_count: The maximum number of candles to fetch.

    Returns:
        A pandas DataFrame with OHLCV data indexed by timestamp (UTC),
        or None if fetching or processing fails after retries.
    """
    log_debug(f"Fetching {limit_count} candles for {trading_symbol} on {tf} timeframe...")
    try:
        # Check if the market exists locally before fetching (faster than API call)
        if trading_symbol not in exchange_instance.markets:
            log_error(
                f"Symbol '{trading_symbol}' not found in locally loaded markets for exchange '{exchange_instance.id}'."
            )
            return None

        # The actual API call is wrapped by the decorator for retries
        ohlcv: List[list] = exchange_instance.fetch_ohlcv(trading_symbol, tf, limit=limit_count)

        if not ohlcv:
            log_warning(
                f"No OHLCV data returned for {trading_symbol} ({tf}). Exchange might be down, symbol inactive, or no recent trades."
            )
            return None

        # Create DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp to datetime and set as index (make timezone-aware UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        # Convert OHLCV columns to numeric, coercing errors to NaN
        numeric_cols: List[str] = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Data Cleansing: Drop rows with NaN in essential OHLCV columns
        initial_rows: int = len(df)
        df.dropna(subset=numeric_cols, inplace=True)
        rows_dropped: int = initial_rows - len(df)
        if rows_dropped > 0:
            log_debug(f"Dropped {rows_dropped} rows with invalid OHLCV data during initial fetch.")

        if df.empty:
            log_warning(f"DataFrame became empty after cleaning NaN OHLCV data for {trading_symbol} ({tf}).")
            return None

        log_debug(f"Successfully fetched and processed {len(df)} candles for {trading_symbol}.")
        return df

    # Handle specific non-retryable CCXT errors after retries have failed
    except ccxt.BadSymbol as e:
        log_error(
            f"BadSymbol error fetching OHLCV for {trading_symbol}: {e}. Check symbol format/availability on {exchange_instance.id}."
        )
        return None
    except ccxt.ExchangeError as e:  # Catch other non-retryable exchange errors
        log_error(f"Exchange specific error fetching OHLCV for {trading_symbol}: {e}")
        return None
    # The retry decorator handles RETRYABLE_EXCEPTIONS and raises the last one if all retries fail.
    # We catch generic Exception here for unexpected issues within this function's logic.
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"An unexpected non-retryable error occurred fetching OHLCV: {e}", exc_info=True)
        # If it was a retryable error that failed all retries, the decorator raised it,
        # and it gets caught here. We return None as the operation ultimately failed.
        return None


# --- Enhanced Order Block Identification Function ---
def identify_potential_order_block(
    df: pd.DataFrame, vol_thresh_mult: float, lookback_len: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Identifies the most recent potential bullish and bearish order blocks based on
    price action and volume relative to a lookback average.

    An order block is simplified as:
    - Bullish OB: The last down-close candle before a strong, high-volume up-move
                  that ideally sweeps the low of the down candle.
    - Bearish OB: The last up-close candle before a strong, high-volume down-move
                  that ideally sweeps the high of the up candle.

    Args:
        df: DataFrame with OHLCV data, indexed by timestamp. Must contain 'volume'.
        vol_thresh_mult: Multiplier for average volume to define "high volume" impulse candle.
        lookback_len: How many *completed* candles back to calculate average volume from.

    Returns:
        A tuple containing:
        - dict: Information about the most recent potential bullish OB found (or None).
                Keys: 'high', 'low', 'time' (Timestamp of OB candle), 'type'.
        - dict: Information about the most recent potential bearish OB found (or None).
                Keys: 'high', 'low', 'time' (Timestamp of OB candle), 'type'.
    """
    bullish_ob: Optional[Dict[str, Any]] = None
    bearish_ob: Optional[Dict[str, Any]] = None

    # Basic validation
    required_cols = ["open", "high", "low", "close", "volume"]
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        log_debug("Input DataFrame missing required columns (OHLCV) for OB detection.")
        return None, None
    if len(df) < lookback_len + 2:  # Need at least lookback + prev_candle + impulse_candle
        log_debug(f"Not enough data for OB detection (need > {lookback_len + 1} rows, got {len(df)}).")
        return None, None

    try:
        # Calculate average volume excluding the last (potentially incomplete) candle
        # Use data up to the second to last candle for stable average calculation
        completed_candles_df = df.iloc[:-1]
        avg_volume = 0.0
        # Ensure enough completed candles for rolling mean, use min_periods for robustness
        if len(completed_candles_df) >= lookback_len:
            # Calculate rolling mean on completed candles, take the last valid value
            rolling_vol = (
                completed_candles_df["volume"]
                .rolling(window=lookback_len, min_periods=max(1, lookback_len // 2))
                .mean()
            )
            if not rolling_vol.empty and pd.notna(rolling_vol.iloc[-1]):
                avg_volume = rolling_vol.iloc[-1]
        elif len(completed_candles_df) > 0:
            # If not enough for lookback, calculate mean of available completed candles
            avg_volume = completed_candles_df["volume"].mean()
            if pd.isna(avg_volume):
                avg_volume = 0.0  # Handle potential NaN mean if all volumes were NaN

        volume_threshold = (
            avg_volume * vol_thresh_mult if avg_volume > 0 else float("inf")
        )  # Avoid threshold=0 or negative threshold
        log_debug(
            f"OB Analysis | Lookback: {lookback_len}, Avg Vol (Completed Candles): {avg_volume:.2f}, Threshold Vol: {volume_threshold:.2f}"
        )

        # Iterate backwards from second-to-last candle to find the most recent OBs
        # Look back up to 'lookback_len' candles or more for the pattern start if needed
        # Limit search depth to avoid excessive computation on large dataframes if needed
        search_depth = min(len(df) - 2, lookback_len * 3)  # Example search depth limit
        start_index = len(df) - 2
        end_index = max(0, start_index - search_depth)  # Ensure index doesn't go below 0

        for i in range(start_index, end_index, -1):  # Iterate backwards
            # Use .iloc for position-based indexing, check index validity
            if i < 1:
                break  # Need at least one previous candle (index i-1)

            try:
                impulse_candle = df.iloc[i]  # The "impulse" or "reversal" candle
                ob_candle = df.iloc[i - 1]  # The candle forming the potential OB zone
            except IndexError:
                log_debug(f"IndexError accessing candles at index {i} or {i - 1}. Stopping OB search.")
                break  # Should not happen with loop bounds, but safety check

            # Check for NaN values in the candles being examined
            if impulse_candle.isnull().any() or ob_candle.isnull().any():
                log_debug(f"Skipping OB check at index {i - 1}-{i} due to NaN values in candles.")
                continue

            # Define conditions
            is_high_volume_impulse = impulse_candle["volume"] > volume_threshold
            # Bullish impulse: Strong close, ideally closing above OB candle's high
            is_bullish_impulse = (
                impulse_candle["close"] > impulse_candle["open"] and impulse_candle["close"] > ob_candle["high"]
            )
            # Bearish impulse: Strong close, ideally closing below OB candle's low
            is_bearish_impulse = (
                impulse_candle["close"] < impulse_candle["open"] and impulse_candle["close"] < ob_candle["low"]
            )
            # OB candle type
            ob_is_bearish = ob_candle["close"] < ob_candle["open"]
            ob_is_bullish = ob_candle["close"] > ob_candle["open"]
            # Sweep condition (impulse candle's range engulfs OB candle's corresponding extreme)
            impulse_sweeps_ob_low = impulse_candle["low"] < ob_candle["low"]
            impulse_sweeps_ob_high = impulse_candle["high"] > ob_candle["high"]

            # --- Potential Bullish OB ---
            # Condition: OB candle is bearish, followed by a high-volume bullish impulse that sweeps the OB low.
            if (
                not bullish_ob  # Find only the most recent one
                and ob_is_bearish
                and is_bullish_impulse
                and is_high_volume_impulse
                and impulse_sweeps_ob_low
            ):
                bullish_ob = {
                    "high": ob_candle["high"],
                    "low": ob_candle["low"],
                    "time": ob_candle.name,  # Timestamp of the bearish candle (the OB)
                    "type": "bullish",
                }
                log_debug(
                    f"Potential Bullish OB identified at {ob_candle.name.strftime('%Y-%m-%d %H:%M')} (Triggered by: {impulse_candle.name.strftime('%H:%M')})"
                )
                # Optimization: If we found both types, stop searching further back
                if bearish_ob:
                    break

            # --- Potential Bearish OB ---
            # Condition: OB candle is bullish, followed by a high-volume bearish impulse that sweeps the OB high.
            elif (
                not bearish_ob  # Find only the most recent one
                and ob_is_bullish
                and is_bearish_impulse
                and is_high_volume_impulse
                and impulse_sweeps_ob_high
            ):
                bearish_ob = {
                    "high": ob_candle["high"],
                    "low": ob_candle["low"],
                    "time": ob_candle.name,  # Timestamp of the bullish candle (the OB)
                    "type": "bearish",
                }
                log_debug(
                    f"Potential Bearish OB identified at {ob_candle.name.strftime('%Y-%m-%d %H:%M')} (Triggered by: {impulse_candle.name.strftime('%H:%M')})"
                )
                # Optimization: If we found both types, stop searching further back
                if bullish_ob:
                    break

        return bullish_ob, bearish_ob

    except Exception as e:
        log_error(f"Error during order block identification: {e}", exc_info=True)
        return None, None


# --- Indicator Calculation Function (Adds ATR and Volume MA conditionally) ---
def calculate_technical_indicators(
    df: Optional[pd.DataFrame],
    rsi_len: int,
    stoch_params: Dict[str, int],  # Explicitly pass params: {'k': k, 'd': d, 'smooth_k': smooth_k}
    calc_atr: bool = False,
    atr_len: int = 14,
    calc_vol_ma: bool = False,
    vol_ma_len: int = 20,
) -> Optional[pd.DataFrame]:
    """
    Calculates technical indicators (RSI, Stochastic) using pandas_ta and appends
    them to the DataFrame. Optionally calculates and appends ATR and Volume MA.

    Args:
        df: Input DataFrame with OHLCV data. Must contain 'high', 'low', 'close', and 'volume' if needed.
        rsi_len: Length for RSI calculation.
        stoch_params: Dictionary with 'k', 'd', 'smooth_k' for Stochastic.
        calc_atr: Boolean flag to calculate ATR.
        atr_len: Length for ATR calculation (if calc_atr is True).
        calc_vol_ma: Boolean flag to calculate Volume Moving Average.
        vol_ma_len: Length for Volume MA calculation (if calc_vol_ma is True).

    Returns:
        The DataFrame with calculated indicators added, or None if input is invalid
        or calculation fails. Rows with NaN values generated by the indicators
        (due to lookback periods) are dropped.
    """
    if df is None or df.empty:
        log_warning("Input DataFrame is None or empty for indicator calculation.")
        return None
    # Ensure required base columns exist
    required_base_cols = ["high", "low", "close"]
    if calc_vol_ma:
        required_base_cols.append("volume")
    if not all(col in df.columns for col in required_base_cols):
        log_error(
            f"Missing required base columns for indicator calculation: Need {required_base_cols}, found {df.columns.tolist()}"
        )
        return None

    log_debug(f"Calculating indicators on DataFrame with {len(df)} rows...")
    original_columns = set(df.columns)
    calculated_indicator_names: List[str] = []  # Keep track of expected new columns

    try:
        # Use a copy to avoid modifying the original DataFrame passed to the function
        df_processed = df.copy()

        # --- Calculate RSI ---
        rsi_col_name = f"RSI_{rsi_len}"
        df_processed.ta.rsi(length=rsi_len, append=True, col_names=(rsi_col_name,))
        calculated_indicator_names.append(rsi_col_name)

        # --- Calculate Stochastic Oscillator ---
        stoch_k_col = f"STOCHk_{stoch_params['k']}_{stoch_params['d']}_{stoch_params['smooth_k']}"
        stoch_d_col = f"STOCHd_{stoch_params['k']}_{stoch_params['d']}_{stoch_params['smooth_k']}"
        # Ensure k, d, smooth_k are passed correctly if using default pandas_ta naming
        df_processed.ta.stoch(
            k=stoch_params["k"],
            d=stoch_params["d"],
            smooth_k=stoch_params["smooth_k"],
            append=True,
            col_names=(stoch_k_col, stoch_d_col),
        )
        calculated_indicator_names.extend([stoch_k_col, stoch_d_col])

        # --- Calculate ATR (conditionally) ---
        atr_col_name: Optional[str] = None
        if calc_atr:
            atr_col_name = f"ATRr_{atr_len}"  # pandas_ta default name for raw ATR
            df_processed.ta.atr(length=atr_len, append=True, col_names=(atr_col_name,))
            log_debug(f"ATR ({atr_len}) calculated.")
            if atr_col_name:
                calculated_indicator_names.append(atr_col_name)

        # --- Calculate Volume MA (conditionally) ---
        vol_ma_col_name: Optional[str] = None
        if calc_vol_ma:
            if "volume" in df_processed.columns:
                vol_ma_col_name = f"VOL_MA_{vol_ma_len}"
                # Use pandas rolling mean directly for more control over min_periods and potential NaNs
                df_processed[vol_ma_col_name] = (
                    df_processed["volume"].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
                )
                log_debug(f"Volume MA ({vol_ma_len}) calculated.")
                if vol_ma_col_name:
                    calculated_indicator_names.append(vol_ma_col_name)
            else:
                log_warning("Volume column not found in DataFrame, cannot calculate Volume MA.")

        # --- Verify Columns and Clean NaNs ---
        new_columns: List[str] = list(set(df_processed.columns) - original_columns)
        # Check if expected columns were actually added by pandas_ta
        missing_calc = [ind_name for ind_name in calculated_indicator_names if ind_name not in new_columns]
        if missing_calc:
            log_warning(f"Some expected indicators might not have been added to DataFrame by pandas_ta: {missing_calc}")

        log_debug(f"Indicators calculated. Columns added: {sorted(new_columns)}")

        initial_len: int = len(df_processed)
        # Drop rows with NaN values specifically in the newly calculated indicator columns
        # This handles the lookback period NaNs without dropping rows with NaNs in original OHLCV data (already handled in fetch)
        valid_indicator_cols_to_check_nan = [col for col in calculated_indicator_names if col in df_processed.columns]
        if valid_indicator_cols_to_check_nan:
            df_processed.dropna(subset=valid_indicator_cols_to_check_nan, inplace=True)
        else:
            log_warning("No valid indicator columns found to check for NaNs after calculation.")

        rows_dropped_nan: int = initial_len - len(df_processed)
        if rows_dropped_nan > 0:
            log_debug(f"Dropped {rows_dropped_nan} rows with NaN values resulting from indicator lookback periods.")

        if df_processed.empty:
            log_warning("DataFrame became empty after dropping NaN rows from indicator calculations.")
            return None

        log_debug(f"Indicator calculation complete. DataFrame now has {len(df_processed)} rows.")
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
    """
    Calculates the position size in the base currency based on available quote
    currency balance, risk percentage, and stop-loss distance.
    Retries balance fetching on common network/API errors.

    Args:
        exchange_instance: Initialized ccxt exchange instance.
        trading_symbol: The trading symbol (e.g., 'BTC/USDT').
        current_price: The current entry price of the asset.
        stop_loss_price: The calculated stop-loss price for the trade.
        risk_perc: The fraction of the available balance to risk (e.g., 0.01 for 1%).

    Returns:
        The calculated position size (float) in the base currency, adjusted for
        market precision and limits, or None if calculation fails or size is invalid.
    """
    if not (0 < risk_perc < 1):
        log_error(f"Invalid risk percentage: {risk_perc}. Must be between 0 and 1.")
        return None
    if current_price <= 0 or stop_loss_price <= 0:
        log_error(f"Invalid prices for size calculation: Current={current_price}, SL={stop_loss_price}")
        return None

    try:
        log_debug(
            f"Calculating position size: Symbol={trading_symbol}, Entry={current_price:.{price_precision_digits}f}, SL={stop_loss_price:.{price_precision_digits}f}, Risk={risk_perc * 100:.2f}%"
        )

        # --- Fetch Market Info ---
        market = exchange_instance.market(trading_symbol)
        if not market:  # Should not happen if symbol validation passed earlier
            log_error(f"Market info for {trading_symbol} not found in `exchange.market()` result.")
            return None

        base_currency: Optional[str] = market.get("base")
        quote_currency: Optional[str] = market.get("quote")
        is_inverse: bool = market.get("inverse", False)
        contract_size: float = float(market.get("contractSize", 1.0))  # Default to 1 if not specified

        if not base_currency or not quote_currency:
            log_error(f"Could not determine base/quote currency for {trading_symbol} from market info.")
            return None
        log_debug(
            f"Market details: Base={base_currency}, Quote={quote_currency}, Inverse={is_inverse}, ContractSize={contract_size}"
        )

        # --- Fetch Account Balance (Retry handled by decorator) ---
        # Determine params needed for fetch_balance (varies by exchange and account type)
        balance_params = {}
        if market.get("swap", False) or market.get("future", False):
            # Example: Some exchanges might need type specified for contracts
            # if exchange_instance.id in ['binance', 'bybit']: balance_params = {'type': 'swap'} # Or 'future'
            pass  # Keep empty by default, relying on ccxt's default behavior

        log_debug(f"Fetching balance with params: {balance_params}")
        balance = exchange_instance.fetch_balance(params=balance_params)

        # --- Find Available Balance in Quote Currency (Robustly) ---
        available_balance = 0.0
        balance_info_to_log = "N/A"  # For logging if balance not found

        # 1. Check standard 'free' structure
        if isinstance(balance.get("free"), dict) and quote_currency in balance["free"]:
            available_balance = float(balance["free"][quote_currency])
            balance_info_to_log = f"balance['free']['{quote_currency}']"
        # 2. Check structure like balance['USDT']['free']
        elif isinstance(balance.get(quote_currency), dict) and "free" in balance[quote_currency]:
            available_balance = float(balance[quote_currency]["free"])
            balance_info_to_log = f"balance['{quote_currency}']['free']"
        # 3. Fallback: Check 'total' if 'free' is zero or missing (less ideal)
        elif (
            isinstance(balance.get(quote_currency), dict)
            and "total" in balance[quote_currency]
            and available_balance == 0.0
        ):
            available_balance = float(balance[quote_currency]["total"])
            balance_info_to_log = f"balance['{quote_currency}']['total'] (using total as free was 0)"
            if available_balance > 0:
                log_warning(f"Using 'total' {quote_currency} balance for calculation, 'free' was zero or unavailable.")
        # 4. Exchange-specific checks (Example for Bybit Unified Margin Account - USDT)
        elif exchange_instance.id == "bybit" and quote_currency == "USDT":
            # Check unified account structure (this might change with API updates)
            if (
                "info" in balance
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
                        # Bybit Unified uses availableToWithdraw or walletBalance - prefer availableToWithdraw
                        available_balance = float(
                            wallet_info.get("availableToWithdraw", wallet_info.get("walletBalance", 0.0))
                        )
                        balance_info_to_log = "Bybit Unified 'availableToWithdraw' or 'walletBalance'"
                        break
            # Check contract account structure (older or non-unified)
            elif (
                isinstance(balance.get(quote_currency), dict) and "available" in balance[quote_currency]
            ):  # Bybit contracts might use 'available'
                available_balance = float(balance[quote_currency]["available"])
                balance_info_to_log = "Bybit Contract 'available'"

        # Add other exchange-specific checks here if needed...

        if available_balance <= 0:
            log_error(
                f"No available {quote_currency} balance ({available_balance}) found for trading. Check funds or balance structure. Balance info checked: {balance_info_to_log}. Full balance details: {balance.get(quote_currency, balance.get('info', 'Not Available'))}"
            )
            return None
        log_info(
            f"Available balance ({quote_currency}): {available_balance:.{price_precision_digits}f} (Source: {balance_info_to_log})"
        )

        # --- Calculate Risk Amount ---
        risk_amount_quote = available_balance * risk_perc
        log_info(
            f"Risk per trade ({risk_perc * 100:.2f}%): {risk_amount_quote:.{price_precision_digits}f} {quote_currency}"
        )

        # --- Calculate Price Difference for SL ---
        price_diff = abs(current_price - stop_loss_price)
        if price_diff <= min_tick / 2:  # Ensure SL is meaningfully far from entry
            log_error(
                f"Stop-loss price {stop_loss_price:.{price_precision_digits}f} is too close to entry price {current_price:.{price_precision_digits}f} (Diff: {price_diff:.{price_precision_digits + 4}f} <= Min Tick/2: {min_tick / 2:.{price_precision_digits + 4}f}). Cannot calculate size."
            )
            return None
        log_debug(f"Price difference for SL: {price_diff:.{price_precision_digits}f}")

        # --- Calculate Quantity in Base Currency ---
        quantity_base: float = 0.0
        if is_inverse:
            # Inverse contracts: Risk is in quote, position size is in base, value fluctuates with price.
            # Quantity (in base) = (Risk Amount in Quote / Entry Price) / (Price Diff / Entry Price) * Contract Size
            # Simplified: Quantity (in base) = (Risk Amount in Quote / Price Diff) * Contract Size
            # Note: This assumes the risk amount is calculated on the quote balance. Check exchange specs.
            log_warning(
                f"Calculating size for INVERSE contract ({trading_symbol}). Ensure risk calculation logic matches exchange specifications."
            )
            if current_price == 0:  # Avoid division by zero
                log_error("Current price is zero, cannot calculate size for inverse contract.")
                return None
            # Quantity in contracts = Risk Amount (Quote) / (Price Diff per Contract in Quote)
            # Price Diff per Contract (Quote) = Price Diff (Quote/Base) * Contract Size (Base/Contract) / Entry Price (Quote/Base) ??? -> This gets complicated.
            # Let's try simpler: Value of 1 Contract in Quote = Contract Size (Base) * Price (Quote/Base)
            # Risk Amount in Contracts = Risk Amount (Quote) / Value of 1 Contract (Quote) ? No.
            # Let's use: Quantity (BASE) = Risk Amount (Quote) / Price Diff (Quote / BASE) -> Gives quantity in BASE units.
            # Then convert to contracts if necessary: Quantity (Contracts) = Quantity (BASE) / Contract Size (BASE)
            quantity_base = risk_amount_quote / price_diff
            log_debug(
                f"Inverse contract intermediate quantity (base units): {quantity_base:.{amount_precision_digits + 4}f}"
            )

        else:  # Linear contracts
            # Linear contracts: Risk is in quote, position size is in base.
            # Quantity (in base) = Risk Amount (Quote) / Price Diff (Quote / Base)
            quantity_base = risk_amount_quote / price_diff
            log_debug(f"Linear contract quantity (base units): {quantity_base:.{amount_precision_digits + 4}f}")

        # Adjust for contract size if it's not 1 base unit per contract
        # The 'amount' field in orders is usually in BASE currency for CCXT, even for contracts,
        # but the exchange might operate in terms of contracts. CCXT usually handles this conversion.
        # We calculate the size in BASE currency here. Let amount_to_precision handle final adjustment.
        # If contract_size != 1.0:
        #    quantity_contracts = quantity_base / contract_size
        #    log_debug(f"Quantity in contracts: {quantity_contracts}")
        #    # Use quantity_base for amount_to_precision as CCXT expects base units
        # else:
        #    log_debug("Contract size is 1, quantity in base units equals quantity in contracts.")

        # --- Adjust for Precision and Limits ---
        try:
            # Use amount_to_precision which should handle rounding according to exchange rules
            quantity_adjusted = float(exchange_instance.amount_to_precision(trading_symbol, quantity_base))
            log_debug(
                f"Quantity adjusted for precision: {quantity_adjusted:.{amount_precision_digits}f} {base_currency}"
            )
        except ccxt.ExchangeError as precision_error:
            log_warning(
                f"Could not use exchange.amount_to_precision: {precision_error}. Using manual rounding (less precise)."
            )
            # Fallback: round manually (less accurate than exchange method)
            quantity_adjusted = round(quantity_base, amount_precision_digits)
            log_debug(f"Quantity manually rounded: {quantity_adjusted:.{amount_precision_digits}f} {base_currency}")

        if quantity_adjusted <= 0:
            log_error(
                f"Calculated adjusted quantity ({quantity_adjusted:.{amount_precision_digits}f}) is zero or negative. Check risk % or balance."
            )
            return None

        # Check exchange limits
        limits = market.get("limits", {})
        min_amount = limits.get("amount", {}).get("min")
        max_amount = limits.get("amount", {}).get("max")
        min_cost = limits.get("cost", {}).get("min")

        log_debug(f"Market Limits: Min Qty={min_amount}, Max Qty={max_amount}, Min Cost={min_cost}")

        if min_amount is not None and quantity_adjusted < min_amount:
            log_error(
                f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} {base_currency} is below minimum required amount {min_amount} {base_currency}."
            )
            return None
        if max_amount is not None and quantity_adjusted > max_amount:
            log_warning(
                f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} exceeds max amount {max_amount}. Capping quantity to max amount."
            )
            # Ensure capped value also meets precision
            quantity_adjusted = float(exchange_instance.amount_to_precision(trading_symbol, max_amount))

        # Check minimum cost if applicable
        estimated_cost = quantity_adjusted * current_price  # Cost in quote currency
        if min_cost is not None and estimated_cost < min_cost:
            log_error(
                f"Estimated order cost ({estimated_cost:.{price_precision_digits}f} {quote_currency}) is below minimum cost {min_cost} {quote_currency}. Required Qty > {min_cost / current_price:.{amount_precision_digits}f} {base_currency}"
            )
            return None

        # Final sanity check: cost vs available balance (leave small buffer)
        cost_buffer_factor = 0.99  # Use 99% of balance as limit for cost check
        if estimated_cost > available_balance * cost_buffer_factor:
            log_error(
                f"Estimated cost ({estimated_cost:.{price_precision_digits}f} {quote_currency}) exceeds {cost_buffer_factor * 100:.0f}% of available balance ({available_balance:.{price_precision_digits}f} {quote_currency}). Reduce risk % or add funds."
            )
            return None

        log_info(
            f"{NEON_GREEN}Position size calculated: {quantity_adjusted:.{amount_precision_digits}f} {base_currency}{RESET} (Risking ~{risk_amount_quote:.2f} {quote_currency})"
        )
        return quantity_adjusted

    # Handle specific non-retryable errors after decorator has handled retries
    except ccxt.AuthenticationError as e:
        log_error(f"Authentication error during position size calculation (fetching balance): {e}")
        return None
    except ccxt.ExchangeError as e:  # Catch other non-retryable exchange errors
        log_error(f"Exchange error during position size calculation: {e}")
        return None
    except Exception as e:
        # Log unexpected errors that weren't caught by specific handlers or the retry decorator
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected non-retryable error calculating position size: {e}", exc_info=True)
        # If it was a retryable error that failed all retries, the decorator raised it.
        return None


# --- Order Placement Functions (Decorated where appropriate) ---


@api_retry_decorator
def cancel_order_with_retry(exchange_instance: ccxt.Exchange, order_id: str, trading_symbol: str) -> bool:
    """
    Attempts to cancel an order by ID using the exchange API.
    Applies retry logic for network/rate limit errors via the decorator.

    Args:
        exchange_instance: Initialized ccxt exchange instance.
        order_id: The ID string of the order to cancel.
        trading_symbol: The symbol the order belongs to (e.g., 'BTC/USDT').

    Returns:
        True if the cancellation request was successfully sent (or simulated),
        False otherwise (e.g., validation error, non-retryable API error).

    Raises:
        ccxt exceptions if cancellation fails after all retries (handled by caller).
        ccxt.OrderNotFound: Can be raised by exchange if order is already gone.
    """
    if not order_id or not isinstance(order_id, str):
        log_debug("No valid order ID provided to cancel.")
        return False  # Nothing to cancel

    log_info(f"Attempting to cancel order ID: {order_id} for {trading_symbol}...")
    if SIMULATION_MODE:
        log_warning(f"SIMULATION: Skipped cancelling order {order_id}.")
        return True  # Simulate success
    else:
        # --- LIVE TRADING ---
        log_warning(f"!!! LIVE MODE: Sending cancellation request for order {order_id}.")
        try:
            # The actual API call is wrapped by the decorator
            exchange_instance.cancel_order(order_id, trading_symbol)
            log_info(f"Order {order_id} cancellation request sent successfully.")
            return True
        except ccxt.OrderNotFound:
            # This is not an error in the context of trying to cancel; it means the order is already gone.
            log_info(f"Order {order_id} not found. Already closed or cancelled.")
            # We still return True because the desired state (order not open) is achieved.
            return True
        # Let other ccxt exceptions (InvalidOrder, AuthenticationError, ExchangeError, or
        # retryable errors after exhaustion) propagate up, handled by the decorator/caller.
        # The decorator will raise if retries fail.


# Note: Placing multiple orders (like SL and TP together) within a single decorated function
# can be problematic if one part fails and the retry attempts everything again, potentially
# creating duplicate orders. It's often safer to place orders individually or use atomic
# OCO (One-Cancels-the-Other) orders if the exchange supports them via a single API call.
# This implementation places market entry, then SL/TP separately. TSL update also cancels/places separately.


@api_retry_decorator
def place_market_order(
    exchange_instance: ccxt.Exchange,
    trading_symbol: str,
    side: str,  # 'buy' or 'sell'
    amount: float,
    reduce_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Places a market order ('buy' or 'sell') with retry logic for network errors.
    Optionally sets reduceOnly parameter for contract markets.

    Args:
        exchange_instance: Initialized ccxt exchange instance.
        trading_symbol: The market symbol (e.g., 'BTC/USDT').
        side: 'buy' or 'sell'.
        amount: The quantity to trade in the base currency.
        reduce_only: If True, sets the reduceOnly parameter (for futures/swaps).

    Returns:
        The order dictionary returned by ccxt upon successful placement or simulation,
        or None if placement fails due to validation, insufficient funds, or
        non-retryable API errors after retries.
    """
    # --- Input Validation ---
    if side not in ["buy", "sell"]:
        log_error(f"Invalid order side: '{side}'. Must be 'buy' or 'sell'.")
        return None
    if not isinstance(amount, (float, int)) or amount <= 0:
        log_error(f"Invalid order amount: {amount}. Must be a positive number.")
        return None

    try:
        market = exchange_instance.market(trading_symbol)
        base_currency: str = market.get("base", "")
        quote_currency: str = market.get("quote", "")

        # Ensure amount conforms to market precision rules BEFORE placing order
        try:
            amount_formatted = float(exchange_instance.amount_to_precision(trading_symbol, amount))
            if amount_formatted <= 0:
                log_error(
                    f"Amount {amount} became zero or negative ({amount_formatted}) after applying precision. Cannot place order."
                )
                return None
        except ccxt.ExchangeError as e:
            log_error(f"Failed to format amount {amount} to precision for {trading_symbol}: {e}")
            return None  # Cannot proceed without correct precision

        log_info(
            f"Attempting to place {side.upper()} market order for {amount_formatted:.{amount_precision_digits}f} {base_currency} on {trading_symbol} {'(ReduceOnly)' if reduce_only else ''}..."
        )

        # --- Prepare Order Parameters ---
        params = {}
        is_contract_market = market.get("swap") or market.get("future") or market.get("contract")
        if reduce_only:
            if is_contract_market:
                # Check if exchange explicitly supports reduceOnly for market orders (some might not)
                # This info isn't standardized in ccxt market structure, often requires trial/error or API docs.
                # Assume it's supported if it's a contract market.
                params["reduceOnly"] = True
                log_debug("Applying 'reduceOnly=True' parameter to market order.")
            else:
                log_warning(
                    "ReduceOnly requested but market type is likely SPOT. Ignoring reduceOnly param for market order."
                )

        # --- Execute Order (Simulation or Live) ---
        order: Optional[Dict[str, Any]] = None
        if SIMULATION_MODE:
            log_warning("!!! SIMULATION: Market order placement skipped.")
            # Create a realistic dummy order response
            sim_price: float = 0.0
            sim_order_id: str = f"sim_market_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
            try:
                # Fetch ticker safely with retry for simulation price
                @api_retry_decorator
                def fetch_ticker_for_sim(exch: ccxt.Exchange, sym: str) -> Dict:
                    """Fetches ticker for simulation price with retry."""
                    return exch.fetch_ticker(sym)

                ticker = fetch_ticker_for_sim(exchange_instance, trading_symbol)
                sim_price = ticker.get("last", ticker.get("close", 0.0))  # Use last or close price
                if sim_price <= 0:
                    log_warning("Could not fetch valid ticker price for simulation, using 0.")
            except Exception as ticker_e:
                # Error already logged by decorator if retries failed
                log_warning(f"Could not fetch ticker for simulation price after retries. Last error: {ticker_e}")

            sim_cost = amount_formatted * sim_price if sim_price > 0 else 0.0
            order = {
                "id": sim_order_id,
                "clientOrderId": sim_order_id,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": "closed",  # Assume market orders fill instantly in simulation
                "symbol": trading_symbol,
                "type": "market",
                "timeInForce": "IOC",
                "side": side,
                "price": sim_price,  # The simulated trigger/request price (market orders don't have limit price)
                "average": sim_price,  # Assume fill at the fetched price
                "amount": amount_formatted,
                "filled": amount_formatted,
                "remaining": 0.0,
                "cost": sim_cost,
                "fee": None,  # Fee simulation could be added if needed
                "reduceOnly": params.get("reduceOnly", False),  # Store if it was requested
                "info": {
                    "simulated": True,
                    "reduceOnly_applied": params.get("reduceOnly", False),
                },  # Add simulation info
            }
        else:
            # --- LIVE TRADING ---
            log_warning(
                f"!!! LIVE MODE: Placing real market order {'(ReduceOnly)' if params.get('reduceOnly') else ''}."
            )
            # The create_market_order call is wrapped by the decorator for retries
            order = exchange_instance.create_market_order(
                symbol=trading_symbol, side=side, amount=amount_formatted, params=params
            )
            # --- END LIVE TRADING ---

        # --- Log Order Result ---
        if order:
            log_info(f"{side.capitalize()} market order request processed {'(Simulated)' if SIMULATION_MODE else ''}.")
            order_id: Optional[str] = order.get("id")
            order_status: Optional[str] = order.get("status")
            # Use 'average' if available (actual filled price), fallback to 'price' (less accurate for market)
            order_price: Optional[float] = order.get("average", order.get("price"))
            order_filled: Optional[float] = order.get("filled")
            order_cost: Optional[float] = order.get("cost")
            reduce_only_info = order.get("reduceOnly", params.get("reduceOnly", "N/A"))  # Check order response too

            price_str: str = (
                f"{order_price:.{price_precision_digits}f}" if isinstance(order_price, (int, float)) else "N/A"
            )
            filled_str: str = (
                f"{order_filled:.{amount_precision_digits}f}" if isinstance(order_filled, (int, float)) else "N/A"
            )
            cost_str: str = (
                f"{order_cost:.{price_precision_digits}f}" if isinstance(order_cost, (int, float)) else "N/A"
            )

            log_info(
                f"Order Result | ID: {order_id or 'N/A'}, Status: {order_status or 'N/A'}, Avg Price: {price_str}, "
                f"Filled: {filled_str} {base_currency}, Cost: {cost_str} {quote_currency}, ReduceOnly: {reduce_only_info}"
            )

            # Add a short delay after placing a live order to allow exchange processing / state update
            if not SIMULATION_MODE:
                time.sleep(1.5)  # Adjust as needed, e.g., 1-3 seconds
        else:
            # This might happen if the API call returns None or empty dict, even after retries
            log_error("Market order placement attempt did not return a valid order object.")
            return None

        return order  # Return the order dictionary (simulated or real)

    # --- Handle Specific Non-Retryable Errors (after decorator retries) ---
    except ccxt.InsufficientFunds as e:
        # This is a critical error indicating a logic flaw or lack of funds.
        log_error(
            f"Insufficient funds for {side} {amount_formatted:.{amount_precision_digits}f} {trading_symbol}. Check balance and risk settings. Error: {e}"
        )
        return None
    except ccxt.InvalidOrder as e:
        # Indicates incorrect parameters, symbol state (e.g., suspended), or order logic violation.
        log_error(
            f"Invalid market order parameters for {side} {amount_formatted:.{amount_precision_digits}f} {trading_symbol}. Check market limits, amount precision, or symbol status. Error: {e}"
        )
        return None
    except ccxt.OrderNotFound as e:  # Can happen if order rejected immediately or right after placement attempt
        log_error(
            f"OrderNotFound error placing {side} {amount_formatted:.{amount_precision_digits}f} {trading_symbol}. Likely rejected by exchange immediately. Error: {e}"
        )
        return None
    except ccxt.ExchangeError as e:  # Catch other specific non-retryable exchange errors
        log_error(f"Exchange specific error placing {side} market order for {trading_symbol}: {e}")
        return None
    except Exception as e:
        # Log unexpected errors that weren't caught by specific handlers or the retry decorator
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected non-retryable error placing {side} market order: {e}", exc_info=True)
        # If it was a retryable error that failed all retries, the decorator raised it. Return None.
        return None


@api_retry_decorator  # Apply retry to the combined SL/TP placement attempt
def place_sl_tp_orders(
    exchange_instance: ccxt.Exchange,
    trading_symbol: str,
    position_side: str,  # 'long' or 'short' (the side of the main position)
    quantity: float,
    sl_price: float,
    tp_price: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Places stop-loss (stop-market preferred, stop-limit fallback) and
    take-profit (limit) orders to protect an existing position.
    Uses reduceOnly for contract markets. Retries the whole placement process
    on network errors (decorator).

    Args:
        exchange_instance: Initialized ccxt exchange instance.
        trading_symbol: The market symbol (e.g., 'BTC/USDT').
        position_side: 'long' or 'short' (the side of the position being protected).
        quantity: The quantity of the position (in base currency) to close.
        sl_price: The trigger price for the stop-loss order.
        tp_price: The limit price for the take-profit order.

    Returns:
        A tuple containing (sl_order_result, tp_order_result).
        Each result is the ccxt order dictionary if placement was successful
        (or simulated), or None if placement failed for that specific order.
        Returns (None, None) if initial validation fails or a critical error occurs
        during the process that prevents attempting placement.

    Note:
        Placing SL and TP separately might be safer in live trading to avoid issues
        if one fails and the retry duplicates the other. This function attempts both
        within one decorated call for simplicity matching original structure, but be aware
        of potential atomicity issues. Consider OCO orders if supported.
    """
    sl_order: Optional[Dict[str, Any]] = None
    tp_order: Optional[Dict[str, Any]] = None

    # --- Input Validation ---
    if not isinstance(quantity, (float, int)) or quantity <= 0:
        log_error(f"Invalid quantity for SL/TP placement: {quantity}. Must be positive number.")
        return None, None
    if position_side not in ["long", "short"]:
        log_error(f"Invalid position_side for SL/TP placement: '{position_side}'. Must be 'long' or 'short'.")
        return None, None
    if (
        not isinstance(sl_price, (float, int))
        or sl_price <= 0
        or not isinstance(tp_price, (float, int))
        or tp_price <= 0
    ):
        log_error(f"Invalid SL ({sl_price}) or TP ({tp_price}) price for placement. Must be positive numbers.")
        return None, None
    # Basic check: SL should be below entry for long, above for short. TP vice-versa.
    # (More robust check against current_price happens before calling this function)
    if position_side == "long" and sl_price >= tp_price:
        log_error(f"Invalid SL/TP for LONG position: SL ({sl_price}) >= TP ({tp_price}).")
        return None, None
    if position_side == "short" and sl_price <= tp_price:
        log_error(f"Invalid SL/TP for SHORT position: SL ({sl_price}) <= TP ({tp_price}).")
        return None, None

    try:
        market = exchange_instance.market(trading_symbol)
        base_currency: str = market.get("base", "")

        # Determine close side and format values
        close_side = "sell" if position_side == "long" else "buy"  # Order side needed to close the position
        try:
            qty_formatted = float(exchange_instance.amount_to_precision(trading_symbol, quantity))
            sl_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, sl_price))
            tp_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, tp_price))
            if qty_formatted <= 0:
                raise ValueError("Quantity became non-positive after precision formatting.")
        except (ccxt.ExchangeError, ValueError) as fmt_e:
            log_error(f"Failed to format quantity or prices to precision for {trading_symbol}: {fmt_e}")
            return None, None

        # --- Prepare Common Parameters ---
        is_contract_market = market.get("swap") or market.get("future") or market.get("contract")
        reduce_only_param = {"reduceOnly": True} if is_contract_market else {}
        if reduce_only_param:
            log_debug("Applying 'reduceOnly=True' to SL/TP orders.")

        # Check exchange capabilities for stop orders via createOrder (more reliable than separate methods)
        # Note: `exchange.has` flags for stop orders (e.g., `has['stopLossPrice']`) are less reliable across exchanges.
        # Checking `exchange.options` or trying `createOrder` with stop params is often better.
        # We'll try to determine preferred stop type.
        stop_loss_order_type: Optional[str] = None
        stop_loss_limit_price: Optional[float] = None  # Only needed for stopLimit

        # Prefer stopMarket if supported via createOrder params
        # (Checking options is heuristic, actual support may vary)
        unified_order_options = exchange_instance.options.get("createOrder", {})
        supports_stop_market_param = "stopPrice" in unified_order_options.get(
            "params", {}
        ) and "market" in unified_order_options.get("types", {})
        supports_stop_limit_param = "stopPrice" in unified_order_options.get(
            "params", {}
        ) and "limit" in unified_order_options.get("types", {})  # Assuming limit type is standard

        # Heuristic check based on common practice: prefer stop-market via create_order
        # This might need adjustment per exchange based on testing or API docs
        if exchange_instance.has.get("createStopMarketOrder", False):  # Check explicit method first
            stop_loss_order_type = "stopMarket"
            log_debug("Exchange has explicit createStopMarketOrder method (will use via createOrder if possible).")
        elif supports_stop_market_param or exchange_instance.id in [
            "binance",
            "bybit",
            "okx",
        ]:  # Assume common exchanges support via params
            stop_loss_order_type = "stopMarket"
            log_debug("Assuming stopMarket type for SL via createOrder params is supported.")
        elif exchange_instance.has.get("createStopLimitOrder", False):  # Fallback to stop-limit method
            stop_loss_order_type = "stopLimit"
            log_debug("Exchange has explicit createStopLimitOrder method (will use via createOrder if possible).")
        elif supports_stop_limit_param:  # Fallback to stop-limit via params
            stop_loss_order_type = "stopLimit"
            log_debug("Assuming stopLimit type for SL via createOrder params is supported.")
        else:
            log_error(
                f"Could not determine a supported stop order type (stopMarket or stopLimit) via createOrder or explicit methods for {exchange_instance.id}. Cannot place automated SL."
            )
            # Cannot proceed with SL placement

        # --- Attempt to Place Stop-Loss Order ---
        sl_params = reduce_only_param.copy()  # Start with common params
        if stop_loss_order_type == "stopMarket":
            sl_params["stopPrice"] = sl_price_formatted
            log_info(
                f"Attempting SL ({stop_loss_order_type}): {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} {base_currency} @ trigger ~{sl_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in sl_params else ''}"
            )
        elif stop_loss_order_type == "stopLimit":
            # Calculate a reasonable limit price for the stop-limit order to increase fill chance
            # Place limit slightly worse than trigger price to avoid missing fill in fast market
            limit_offset = abs(tp_price_formatted - sl_price_formatted) * 0.10  # 10% of TP-SL range as offset? Risky.
            limit_offset = max(min_tick * 5, limit_offset)  # Ensure offset is at least a few ticks, capped by range?
            # Safer: Place limit very close to stop price or slightly beyond
            limit_offset_ticks = min_tick * 10  # Place limit 10 ticks away from trigger
            sl_limit_price_raw = (
                sl_price_formatted - limit_offset_ticks
                if close_side == "sell"
                else sl_price_formatted + limit_offset_ticks
            )
            stop_loss_limit_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price_raw))

            sl_params["stopPrice"] = sl_price_formatted
            sl_params["price"] = stop_loss_limit_price  # The limit price for the order placed after trigger
            log_warning(
                f"Using stopLimit type for SL. Trigger: {sl_price_formatted:.{price_precision_digits}f}, Limit: {stop_loss_limit_price:.{price_precision_digits}f}. Fill is not guaranteed if price gaps past limit."
            )
            log_info(
                f"Attempting SL ({stop_loss_order_type}): {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} {base_currency} @ trigger {sl_price_formatted:.{price_precision_digits}f}, limit {stop_loss_limit_price:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in sl_params else ''}"
            )
        else:
            # SL placement not possible
            pass  # Error already logged

        # Place SL order if type determined
        if stop_loss_order_type:
            if SIMULATION_MODE:
                log_warning("!!! SIMULATION: Stop-loss order placement skipped.")
                sl_order_id = f"sim_sl_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
                sl_order = {
                    "id": sl_order_id,
                    "status": "open",
                    "symbol": trading_symbol,
                    "type": stop_loss_order_type,
                    "side": close_side,
                    "amount": qty_formatted,
                    "price": stop_loss_limit_price,  # Price is limit price for stopLimit, None for stopMarket in some exchanges
                    "stopPrice": sl_params.get("stopPrice"),
                    "average": None,
                    "filled": 0.0,
                    "remaining": qty_formatted,
                    "cost": 0.0,
                    "reduceOnly": sl_params.get("reduceOnly", False),
                    "info": {"simulated": True, "reduceOnly_applied": sl_params.get("reduceOnly", False)},
                }
            else:
                # --- LIVE TRADING ---
                log_warning(f"!!! LIVE MODE: Placing real {stop_loss_order_type} stop-loss order.")
                # The create_order call is wrapped by the main function's decorator for retries
                sl_order = exchange_instance.create_order(
                    symbol=trading_symbol,
                    type=stop_loss_order_type,
                    side=close_side,
                    amount=qty_formatted,
                    price=stop_loss_limit_price,  # Pass limit price only if stopLimit, ignored otherwise
                    params=sl_params,
                )
                log_info(
                    f"Stop-loss order request processed. ID: {sl_order.get('id', 'N/A')}, Status: {sl_order.get('status', 'N/A')}"
                )
                time.sleep(0.75)  # Small delay after live order
                # --- END LIVE TRADING ---

        # --- Attempt to Place Take-Profit Order ---
        tp_params = reduce_only_param.copy()
        tp_order_type = "limit"  # Standard limit order for TP
        log_info(
            f"Attempting TP ({tp_order_type}): {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} {base_currency} @ limit {tp_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in tp_params else ''}"
        )

        # Check if limit orders are supported (should almost always be true)
        if exchange_instance.has.get("createLimitOrder", True):
            if SIMULATION_MODE:
                log_warning("!!! SIMULATION: Take-profit order placement skipped.")
                tp_order_id = f"sim_tp_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
                tp_order = {
                    "id": tp_order_id,
                    "status": "open",
                    "symbol": trading_symbol,
                    "type": tp_order_type,
                    "side": close_side,
                    "amount": qty_formatted,
                    "price": tp_price_formatted,  # Limit price
                    "average": None,
                    "filled": 0.0,
                    "remaining": qty_formatted,
                    "cost": 0.0,
                    "reduceOnly": tp_params.get("reduceOnly", False),
                    "info": {"simulated": True, "reduceOnly_applied": tp_params.get("reduceOnly", False)},
                }
            else:
                # --- LIVE TRADING ---
                log_warning(f"!!! LIVE MODE: Placing real {tp_order_type} take-profit order.")
                # Use create_limit_order for clarity, wrapped by decorator
                tp_order = exchange_instance.create_limit_order(
                    symbol=trading_symbol,
                    side=close_side,
                    amount=qty_formatted,
                    price=tp_price_formatted,
                    params=tp_params,
                )
                log_info(
                    f"Take-profit order request processed. ID: {tp_order.get('id', 'N/A')}, Status: {tp_order.get('status', 'N/A')}"
                )
                time.sleep(0.75)  # Small delay after live order
                # --- END LIVE TRADING ---
        else:
            log_error(
                f"Exchange {exchange_instance.id} does not explicitly support limit orders (highly unlikely). Cannot place take-profit."
            )
            # tp_order remains None

        # --- Final Check and Warnings ---
        sl_ok = sl_order and sl_order.get("id")
        tp_ok = tp_order and tp_order.get("id")

        if sl_ok and not tp_ok and exchange_instance.has.get("createLimitOrder", True):
            log_warning("SL order placed successfully, but TP order placement failed or did not return an ID.")
        elif not sl_ok and tp_ok and stop_loss_order_type:
            log_warning(
                "TP order placed successfully, but SL order placement failed or did not return an ID. POSITION IS UNPROTECTED!"
            )
        elif not sl_ok and not tp_ok and stop_loss_order_type and exchange_instance.has.get("createLimitOrder", True):
            log_error("Both SL and TP order placements failed or did not return IDs.")
        elif not stop_loss_order_type and tp_ok:  # SL type wasn't supported, but TP worked
            log_warning(
                "TP order placed, but SL order type not supported by exchange config/heuristics. POSITION IS UNPROTECTED!"
            )
        elif not stop_loss_order_type and not tp_ok:  # Neither SL type supported nor TP worked
            log_error("SL order type not supported and TP order placement failed.")

        return sl_order, tp_order

    # --- Handle Specific Non-Retryable Errors (after decorator retries) ---
    # These might occur during the placement of either SL or TP
    except ccxt.InvalidOrder as e:
        log_error(f"Invalid order parameters placing SL/TP: {e}. Check limits, precision, or params.")
        # Return whatever might have succeeded before the error
        return sl_order, tp_order
    except ccxt.InsufficientFunds as e:
        # Should not happen with reduceOnly=True, but catch just in case.
        log_error(f"Insufficient funds error during SL/TP placement (unexpected for reduceOnly): {e}")
        return sl_order, tp_order
    except ccxt.ExchangeError as e:  # Catch other non-retryable exchange errors
        log_error(f"Exchange specific error placing SL/TP orders: {e}")
        return sl_order, tp_order  # Return potentially partially successful results
    except Exception as e:
        # Log unexpected errors
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected non-retryable error in place_sl_tp_orders: {e}", exc_info=True)
        # Return None, None to indicate total failure if an exception occurred after retries
        # or an unexpected error happened before any placement.
        return None, None


# --- Position and Order Check Function (Decorated) ---
@api_retry_decorator
def check_position_and_orders(exchange_instance: ccxt.Exchange, trading_symbol: str) -> bool:
    """
    Checks consistency between the bot's tracked position state and open orders on the exchange.
    Specifically, it checks if tracked SL or TP orders are still open.
    If a tracked order (SL or TP) is found to be closed/filled/cancelled, it assumes
    the position associated with it has been closed. It then attempts to cancel the
    *other* corresponding order (TP or SL) and resets the local position state.

    Args:
        exchange_instance: Initialized ccxt exchange instance.
        trading_symbol: The market symbol to check orders for.

    Returns:
        True if the position state was reset due to a detected closure (SL/TP hit),
        False otherwise (position still considered open or no active position).

    Raises:
        ccxt exceptions from `fetch_open_orders` or `cancel_order_with_retry`
        if they fail after retries (handled by caller or main loop).
    """
    global position
    if position["status"] is None:
        log_debug("No active position state to check.")
        return False  # No active position tracked locally

    log_debug(f"Checking open orders vs local state for active {position['status']} position on {trading_symbol}...")
    position_reset_required: bool = False
    order_to_cancel_id: Optional[str] = None
    assumed_close_reason: Optional[str] = None  # 'SL' or 'TP'

    try:
        # Fetch all open orders for the specific symbol (retried by decorator)
        # Using fetch_open_orders is generally preferred over fetching individual order statuses
        # as it's often a single API call.
        open_orders: List[Dict[str, Any]] = exchange_instance.fetch_open_orders(trading_symbol)
        open_order_ids: set[str] = {order["id"] for order in open_orders if "id" in order}
        log_debug(
            f"Found {len(open_orders)} open orders for {trading_symbol}. IDs: {open_order_ids if open_order_ids else 'None'}"
        )

        # Get the SL and TP order IDs tracked in the local state
        sl_order_id: Optional[str] = position.get("sl_order_id")
        tp_order_id: Optional[str] = position.get("tp_order_id")

        # Check if we have tracked IDs - if not, we can't reliably determine closure via orders
        if not sl_order_id and not tp_order_id:
            # This state might occur if SL/TP placement failed initially, or after a restart
            # without restoring order IDs, or if orders were manually managed.
            log_warning(
                f"In {position['status']} position but no SL/TP order IDs are tracked in state. Cannot verify position closure via orders."
            )
            # TODO: Optional Enhancement: If market is contract type and exchange supports fetch_positions,
            # could try fetching actual position size here as a fallback check.
            # Example: if exchange_instance.has.get('fetchPositions'):
            #             try: positions = exchange_instance.fetch_positions([trading_symbol]) ... logic ...
            # For now, return False as we cannot confirm closure based on tracked orders.
            return False

        # --- Determine if a tracked order is no longer open ---
        sl_found_open: bool = sl_order_id in open_order_ids if sl_order_id else False
        tp_found_open: bool = tp_order_id in open_order_ids if tp_order_id else False

        log_debug(f"Tracked SL Order ({sl_order_id or 'None'}) found open: {sl_found_open}")
        log_debug(f"Tracked TP Order ({tp_order_id or 'None'}) found open: {tp_found_open}")

        # Condition for reset: If SL was tracked and is NOT open anymore
        if sl_order_id and not sl_found_open:
            log_info(
                f"{NEON_YELLOW}Stop-loss order {sl_order_id} no longer open. Assuming position closed via SL.{RESET}"
            )
            position_reset_required = True
            order_to_cancel_id = tp_order_id  # Mark the TP order for cancellation
            assumed_close_reason = "SL"

        # Condition for reset: If TP was tracked and is NOT open anymore (and SL wasn't the trigger)
        elif tp_order_id and not tp_found_open:
            log_info(
                f"{NEON_GREEN}Take-profit order {tp_order_id} no longer open. Assuming position closed via TP.{RESET}"
            )
            position_reset_required = True
            order_to_cancel_id = sl_order_id  # Mark the SL order for cancellation
            assumed_close_reason = "TP"

        # --- Perform State Reset and Cancel Leftover Order ---
        if position_reset_required:
            log_info(
                f"Position closure detected (assumed via {assumed_close_reason}). Resetting local state and cancelling leftover order."
            )

            # Attempt to cancel the other order (if it exists and was tracked)
            if order_to_cancel_id:
                log_info(
                    f"Attempting to cancel leftover {'TP' if assumed_close_reason == 'SL' else 'SL'} order: {order_to_cancel_id}"
                )
                try:
                    # Use the decorated cancel helper (handles retries and OrderNotFound gracefully)
                    cancel_success = cancel_order_with_retry(exchange_instance, order_to_cancel_id, trading_symbol)
                    if cancel_success:
                        log_info(
                            f"Leftover order {order_to_cancel_id} cancellation request sent or order already closed."
                        )
                    else:
                        # This indicates a potential issue if cancellation failed unexpectedly after retries
                        log_error(
                            f"Failed to confirm cancellation of leftover order {order_to_cancel_id} after retries. Manual check advised."
                        )
                except Exception as e:
                    # Catch potential exceptions raised by cancel_order_with_retry if retries failed
                    log_error(
                        f"Error occurred trying to cancel leftover order {order_to_cancel_id}: {e}. Manual check advised.",
                        exc_info=False,
                    )
                    # Proceed with state reset, but warn user.
            else:
                log_debug(
                    f"No corresponding {'TP' if assumed_close_reason == 'SL' else 'SL'} order ID was tracked or provided to cancel."
                )

            # Reset the local position state to default values
            log_info("Resetting local position state to default (no position).")
            position.update(position_default_structure.copy())  # Use update with a copy of defaults
            save_position_state()  # Persist the reset state
            display_position_status(
                position, price_precision_digits, amount_precision_digits
            )  # Show updated (empty) status
            return True  # Indicate position state was reset

        # If both tracked orders are still open (or only one was tracked and it's still open), state is consistent
        log_debug("Position state appears consistent with tracked open orders (SL/TP still open or untracked).")
        return False  # Position was not reset

    # --- Handle Specific Non-Retryable Errors from fetch_open_orders ---
    except ccxt.AuthenticationError as e:
        log_error(f"Authentication error checking position/orders: {e}")
        return False  # State uncertain, cannot proceed reliably
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error checking position/orders: {e}")
        return False  # State uncertain
    except Exception as e:
        # Log unexpected errors
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected non-retryable error checking position/orders: {e}", exc_info=True)
        # If retryable error exhausted retries, decorator raised it. Return False as state is uncertain.
        return False


# --- Trailing Stop Loss Update Function ---
def update_trailing_stop(
    exchange_instance: ccxt.Exchange,
    trading_symbol: str,
    current_price: float,
    last_atr: Optional[float],  # Pass Optional ATR value
) -> None:
    """
    Checks if the trailing stop loss (TSL) needs to be updated based on price movement
    relative to entry and ATR. If an update is required and valid, it cancels the
    old SL order and places a new one at the improved TSL price.

    Args:
        exchange_instance: Initialized ccxt exchange instance.
        trading_symbol: The market symbol (e.g., 'BTC/USDT').
        current_price: The current market price.
        last_atr: The last calculated ATR value (or None if not available/applicable).
    """
    global position
    # --- Pre-conditions for Trailing Stop Logic ---
    if position["status"] is None:
        # log_debug("TSL Check: No active position.")
        return
    if not enable_trailing_stop:
        # log_debug("TSL Check: Trailing stop disabled in config.")
        return
    if position.get("sl_order_id") is None:
        # Only warn if TSL is enabled but no SL ID exists (e.g., initial placement failed)
        log_warning("TSL Check: Cannot perform trailing stop update - No active SL order ID is tracked in state.")
        return
    if last_atr is None or last_atr <= 0:
        log_warning(f"TSL Check: Cannot perform trailing stop update - Invalid ATR value ({last_atr}).")
        return

    log_debug(
        f"TSL Check: Active {position['status']} position. Current SL Order: {position['sl_order_id']}, Last Tracked TSL Price: {position.get('current_trailing_sl_price')}"
    )

    # --- Get Required State Variables ---
    entry_price: Optional[float] = position.get("entry_price")
    initial_sl_price: Optional[float] = position.get("stop_loss")  # The SL price *before* any TSL updates
    current_tsl_price: Optional[float] = position.get(
        "current_trailing_sl_price"
    )  # The currently active TSL price, if any

    if entry_price is None or initial_sl_price is None:
        log_warning("TSL Check: Cannot proceed - Entry price or initial SL price missing from position state.")
        return

    # Determine the SL price we are comparing against (the *current* effective SL)
    # If TSL has already activated, use current_tsl_price, otherwise use the initial SL.
    current_effective_sl_price = current_tsl_price if current_tsl_price is not None else initial_sl_price

    # --- Update Peak Prices Seen Since Entry ---
    if position["status"] == "long":
        highest_seen = position.get("highest_price_since_entry", entry_price)  # Initialize with entry if missing
        if current_price > highest_seen:
            position["highest_price_since_entry"] = current_price
            log_debug(f"TSL: New high for long position: {current_price:.{price_precision_digits}f}")
            # Defer saving state until TSL actually moves
    elif position["status"] == "short":
        lowest_seen = position.get("lowest_price_since_entry", entry_price)  # Initialize with entry if missing
        if current_price < lowest_seen:
            position["lowest_price_since_entry"] = current_price
            log_debug(f"TSL: New low for short position: {current_price:.{price_precision_digits}f}")
            # Defer saving state

    # --- Check Activation Threshold ---
    tsl_activated = False
    potential_new_tsl_price: Optional[float] = None

    if position["status"] == "long":
        activation_price_threshold = entry_price + (last_atr * trailing_stop_activation_atr_multiplier)
        current_high = position.get("highest_price_since_entry", entry_price)
        if current_high > activation_price_threshold:
            tsl_activated = True
            # Calculate potential new TSL based on the highest price seen
            potential_new_tsl_price = current_high - (last_atr * trailing_stop_atr_multiplier)
            log_debug(
                f"Long TSL Activation Met. High ({current_high:.{price_precision_digits}f}) > Threshold ({activation_price_threshold:.{price_precision_digits}f}). Potential New TSL: {potential_new_tsl_price:.{price_precision_digits}f}"
            )
        else:
            log_debug(
                f"Long TSL Activation NOT Met. High ({current_high:.{price_precision_digits}f}) <= Threshold ({activation_price_threshold:.{price_precision_digits}f})"
            )

    elif position["status"] == "short":
        activation_price_threshold = entry_price - (last_atr * trailing_stop_activation_atr_multiplier)
        current_low = position.get("lowest_price_since_entry", entry_price)
        if current_low < activation_price_threshold:
            tsl_activated = True
            # Calculate potential new TSL based on the lowest price seen
            potential_new_tsl_price = current_low + (last_atr * trailing_stop_atr_multiplier)
            log_debug(
                f"Short TSL Activation Met. Low ({current_low:.{price_precision_digits}f}) < Threshold ({activation_price_threshold:.{price_precision_digits}f}). Potential New TSL: {potential_new_tsl_price:.{price_precision_digits}f}"
            )
        else:
            log_debug(
                f"Short TSL Activation NOT Met. Low ({current_low:.{price_precision_digits}f}) >= Threshold ({activation_price_threshold:.{price_precision_digits}f})"
            )

    # --- Check if Potential New TSL is an Improvement and Valid ---
    should_update_tsl = False
    new_tsl_price_formatted: Optional[float] = None

    if tsl_activated and potential_new_tsl_price is not None:
        try:
            # Format potential new TSL to market precision for valid comparison and placement
            new_tsl_price_formatted = float(
                exchange_instance.price_to_precision(trading_symbol, potential_new_tsl_price)
            )
        except ccxt.ExchangeError as fmt_e:
            log_error(
                f"TSL Update: Failed to format potential new TSL price {potential_new_tsl_price} to precision: {fmt_e}"
            )
            return  # Cannot proceed without formatted price

        # Check if new TSL is strictly better (higher for long, lower for short) than the current effective SL
        # AND ensure the new TSL is not placed beyond the current market price (would cause immediate stop-out)
        if position["status"] == "long":
            if new_tsl_price_formatted > current_effective_sl_price and new_tsl_price_formatted < current_price:
                should_update_tsl = True
                log_debug(
                    f"Long TSL Improvement Check: New ({new_tsl_price_formatted:.{price_precision_digits}f}) > Current Eff. SL ({current_effective_sl_price:.{price_precision_digits}f}) AND < Price ({current_price:.{price_precision_digits}f}) -> OK"
                )
            else:
                log_debug(
                    f"Long TSL Improvement Check: New ({new_tsl_price_formatted:.{price_precision_digits}f}) vs Current Eff. SL ({current_effective_sl_price:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f}) -> NO UPDATE"
                )

        elif position["status"] == "short":
            if new_tsl_price_formatted < current_effective_sl_price and new_tsl_price_formatted > current_price:
                should_update_tsl = True
                log_debug(
                    f"Short TSL Improvement Check: New ({new_tsl_price_formatted:.{price_precision_digits}f}) < Current Eff. SL ({current_effective_sl_price:.{price_precision_digits}f}) AND > Price ({current_price:.{price_precision_digits}f}) -> OK"
                )
            else:
                log_debug(
                    f"Short TSL Improvement Check: New ({new_tsl_price_formatted:.{price_precision_digits}f}) vs Current Eff. SL ({current_effective_sl_price:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f}) -> NO UPDATE"
                )

    # --- Execute TSL Update (Cancel Old SL, Place New SL) ---
    if should_update_tsl and new_tsl_price_formatted is not None:
        log_info(
            f"{NEON_YELLOW}Trailing Stop Update Triggered! Moving SL from ~{current_effective_sl_price:.{price_precision_digits}f} to {new_tsl_price_formatted:.{price_precision_digits}f}{RESET}"
        )
        old_sl_order_id = position["sl_order_id"]  # Already confirmed not None at function start

        try:
            # --- Step 1: Cancel the existing SL order ---
            log_info(f"TSL Update: Attempting to cancel old SL order {old_sl_order_id}...")
            cancel_success = cancel_order_with_retry(exchange_instance, old_sl_order_id, trading_symbol)
            # cancel_order_with_retry returns True if cancelled or already not found
            if cancel_success:
                log_info(f"TSL Update: Old SL order {old_sl_order_id} cancellation successful or order already closed.")
                time.sleep(1.0)  # Allow exchange time to process cancellation before placing new order
            else:
                # This case means cancellation failed unexpectedly after retries
                log_error(
                    f"TSL Update: Failed to cancel old SL order {old_sl_order_id}. Aborting TSL update. Position remains protected by potentially existing old SL."
                )
                return  # Do not proceed if cancellation failed

            # --- Step 2: Place the new TSL order (as a new stop-loss order) ---
            log_info(
                f"TSL Update: Attempting to place new SL order at {new_tsl_price_formatted:.{price_precision_digits}f}"
            )
            new_sl_order_result: Optional[Dict[str, Any]] = None
            try:
                close_side = "sell" if position["status"] == "long" else "buy"
                current_qty = position.get("quantity")

                if current_qty is None or not isinstance(current_qty, (float, int)) or current_qty <= 0:
                    log_error("TSL Update: Cannot place new TSL order - Invalid quantity in position state.")
                    raise ValueError(
                        "Invalid position quantity for TSL placement."
                    )  # Raise to prevent state update below

                # Re-use the SL/TP placement logic structure for placing the new SL
                # Determine stop type and params again (in case exchange state changed, though unlikely)
                market = exchange_instance.market(trading_symbol)
                is_contract_market = market.get("swap") or market.get("future") or market.get("contract")
                new_sl_reduce_only_param = {"reduceOnly": True} if is_contract_market else {}

                new_sl_order_type: Optional[str] = None
                new_sl_limit_price: Optional[float] = None
                new_sl_params = new_sl_reduce_only_param.copy()

                # Determine preferred stop type (same logic as in place_sl_tp_orders)
                unified_order_options = exchange_instance.options.get("createOrder", {})
                supports_stop_market_param = "stopPrice" in unified_order_options.get(
                    "params", {}
                ) and "market" in unified_order_options.get("types", {})
                supports_stop_limit_param = "stopPrice" in unified_order_options.get(
                    "params", {}
                ) and "limit" in unified_order_options.get("types", {})

                if (
                    exchange_instance.has.get("createStopMarketOrder", False)
                    or supports_stop_market_param
                    or exchange_instance.id in ["binance", "bybit", "okx"]
                ):
                    new_sl_order_type = "stopMarket"
                    new_sl_params["stopPrice"] = new_tsl_price_formatted
                elif exchange_instance.has.get("createStopLimitOrder", False) or supports_stop_limit_param:
                    new_sl_order_type = "stopLimit"
                    # Calculate limit price for stopLimit (similar logic to place_sl_tp_orders)
                    position.get(
                        "take_profit", current_price * (1.05 if position["status"] == "long" else 0.95)
                    )  # Use original TP for range, fallback if missing
                    limit_offset_ticks = min_tick * 10
                    sl_limit_price_raw = (
                        new_tsl_price_formatted - limit_offset_ticks
                        if close_side == "sell"
                        else new_tsl_price_formatted + limit_offset_ticks
                    )
                    new_sl_limit_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price_raw))
                    new_sl_params["stopPrice"] = new_tsl_price_formatted
                    new_sl_params["price"] = new_sl_limit_price
                    log_warning(
                        f"TSL Update: Using stopLimit type. Trigger: {new_tsl_price_formatted:.{price_precision_digits}f}, Limit: {new_sl_limit_price:.{price_precision_digits}f}."
                    )
                else:
                    log_error(
                        f"TSL Update: Exchange {exchange_instance.id} supports neither stopMarket nor stopLimit via createOrder for TSL update. Cannot place new TSL order."
                    )
                    raise ccxt.NotSupported(
                        f"Exchange {exchange_instance.id} does not support required stop order types for TSL update."
                    )

                log_info(f"TSL Update: Attempting direct placement of new {new_sl_order_type} SL order.")

                # --- Place New SL Order (Simulation or Live) ---
                if SIMULATION_MODE:
                    log_warning("!!! SIMULATION: New TSL order placement skipped.")
                    new_sl_order_id = f"sim_tsl_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
                    new_sl_order_result = {
                        "id": new_sl_order_id,
                        "status": "open",
                        "type": new_sl_order_type,
                        "price": new_sl_limit_price,
                        "stopPrice": new_sl_params.get("stopPrice"),
                        "info": {"simulated": True},
                    }
                else:
                    log_warning(f"!!! LIVE MODE: Placing real {new_sl_order_type} TSL order.")

                    # Apply retry decorator specifically to this critical placement call
                    @api_retry_decorator
                    def place_new_tsl_order_live(qty_fmt):
                        """Places the new TSL order with retry."""
                        return exchange_instance.create_order(
                            symbol=trading_symbol,
                            type=new_sl_order_type,
                            side=close_side,
                            amount=qty_fmt,
                            price=new_sl_limit_price,
                            params=new_sl_params,
                        )

                    # Format quantity *just before* placing the live order
                    qty_formatted_live = float(exchange_instance.amount_to_precision(trading_symbol, current_qty))
                    if qty_formatted_live <= 0:
                        raise ValueError("Quantity became non-positive after precision formatting for live TSL order.")
                    new_sl_order_result = place_new_tsl_order_live(qty_formatted_live)

                # --- Step 3: Update State on Successful Placement ---
                if new_sl_order_result and new_sl_order_result.get("id"):
                    new_sl_order_id = new_sl_order_result["id"]
                    log_info(
                        f"{NEON_GREEN}TSL Update: New trailing SL order placed successfully: ID {new_sl_order_id} at {new_tsl_price_formatted:.{price_precision_digits}f}{RESET}"
                    )
                    # Update the position state with the new SL info
                    position["sl_order_id"] = new_sl_order_id
                    # Update 'stop_loss' to reflect the new *active* SL price for consistency
                    position["stop_loss"] = new_tsl_price_formatted
                    # Explicitly track the TSL price to distinguish from initial SL
                    position["current_trailing_sl_price"] = new_tsl_price_formatted
                    save_position_state()  # Save the successful TSL update
                    display_position_status(
                        position, price_precision_digits, amount_precision_digits
                    )  # Show updated status
                else:
                    # This case means placing the new order failed even after cancellation worked
                    log_error(
                        f"TSL Update CRITICAL ERROR: Failed to place new trailing SL order after cancelling old one {old_sl_order_id}. POSITION IS NOW UNPROTECTED."
                    )
                    position["sl_order_id"] = None  # Mark SL as lost
                    position["current_trailing_sl_price"] = None  # TSL is no longer active
                    # Keep 'stop_loss' as the last known TSL target? Or revert to initial? Reverting might be confusing. Keep it as the target.
                    save_position_state()  # Save the state reflecting the lost SL order

            except Exception as place_e:
                # Catch errors during the placement attempt (including retries failing from decorator)
                log_error(
                    f"TSL Update CRITICAL ERROR: Error placing new trailing SL order: {place_e}. POSITION MAY BE UNPROTECTED.",
                    exc_info=True,
                )
                position["sl_order_id"] = None  # Ensure SL state reflects failure
                position["current_trailing_sl_price"] = None
                save_position_state()  # Save the failed state

        except Exception as cancel_e:
            # Catch errors during the cancellation attempt (including retries failing from decorator)
            # Error is logged by cancel_order_with_retry or its decorator if retries fail
            log_error(
                f"TSL Update: Error cancelling old SL order {old_sl_order_id} during TSL update: {cancel_e}. Aborting TSL placement.",
                exc_info=False,
            )
            # Do not proceed to place new SL if cancellation failed unexpectedly. Position remains protected by the old SL (if it still exists).


# --- Main Trading Loop ---
def run_bot():
    """Encapsulates the main trading loop and shutdown logic."""
    log_info(f"Initializing trading bot for {symbol} on {timeframe}...")
    # Load position state ONCE at startup
    load_position_state()
    log_info(f"Risk per trade: {risk_percentage * 100:.2f}%")
    log_info(f"Bot check interval: {sleep_interval_seconds} seconds ({sleep_interval_seconds / 60:.1f} minutes)")
    log_info(f"ATR SL/TP Enabled: {enable_atr_sl_tp}, Trailing Stop Enabled: {enable_trailing_stop}")
    log_info(f"{NEON_YELLOW}Press Ctrl+C to stop the bot gracefully.{RESET}")

    while True:
        try:
            cycle_start_time: pd.Timestamp = pd.Timestamp.now(tz="UTC")
            print_cycle_divider(cycle_start_time)

            # --- Step 1: Check Position Consistency & Handle Closures ---
            # This function checks if tracked SL/TP orders are closed, cancels the counterpart,
            # resets the local state, and returns True if a reset occurred.
            position_was_reset = check_position_and_orders(exchange, symbol)

            # Display status *after* the check, reflecting any resets immediately
            display_position_status(position, price_precision_digits, amount_precision_digits)

            if position_was_reset:
                log_info(
                    "Position state was reset by order check (SL/TP likely hit). Checking for new entry signals immediately."
                )
                # No sleep needed, proceed directly to check for new entries in the next iteration
                continue  # Skip the rest of this cycle's logic

            # --- Step 2: Fetch Fresh Market Data ---
            ohlcv_df: Optional[pd.DataFrame] = fetch_ohlcv_data(exchange, symbol, timeframe, limit_count=data_limit)
            if ohlcv_df is None or ohlcv_df.empty:
                log_warning(f"Could not fetch valid OHLCV data for {symbol}. Waiting for next cycle...")
                neon_sleep_timer(sleep_interval_seconds)
                continue  # Skip to next iteration

            # --- Step 3: Calculate Technical Indicators ---
            # Determine if ATR and Vol MA need calculation based on config
            needs_atr_calc = enable_atr_sl_tp or enable_trailing_stop
            needs_vol_ma_calc = entry_volume_confirmation_enabled
            stoch_params_dict = {"k": stoch_k, "d": stoch_d, "smooth_k": stoch_smooth_k}

            df_with_indicators: Optional[pd.DataFrame] = calculate_technical_indicators(
                ohlcv_df,  # Pass the original fetched data (calc func makes a copy)
                rsi_len=rsi_length,
                stoch_params=stoch_params_dict,
                calc_atr=needs_atr_calc,
                atr_len=atr_length,
                calc_vol_ma=needs_vol_ma_calc,
                vol_ma_len=entry_volume_ma_length,
            )
            if df_with_indicators is None or df_with_indicators.empty:
                log_warning(f"Indicator calculation failed or resulted in empty DataFrame for {symbol}. Waiting...")
                neon_sleep_timer(sleep_interval_seconds)
                continue

            # --- Step 4: Get Latest Data Point and Values ---
            if len(df_with_indicators) < 1:  # Should not happen if indicator calc returned df, but safety check
                log_warning("DataFrame has no rows after indicator calculation. Waiting...")
                neon_sleep_timer(sleep_interval_seconds)
                continue

            latest_data: pd.Series = df_with_indicators.iloc[-1]
            latest_timestamp: pd.Timestamp = latest_data.name  # Get timestamp from index

            # Construct indicator column names dynamically based on config for lookup
            rsi_col_name: str = f"RSI_{rsi_length}"
            stoch_k_col_name: str = f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}"
            stoch_d_col_name: str = f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}"
            atr_col_name: Optional[str] = f"ATRr_{atr_length}" if needs_atr_calc else None
            vol_ma_col_name: Optional[str] = f"VOL_MA_{entry_volume_ma_length}" if needs_vol_ma_calc else None

            # Verify that all required columns exist in the final DataFrame
            required_cols_for_logic: List[str] = [
                "close",
                "high",
                "low",
                "open",
                "volume",
                rsi_col_name,
                stoch_k_col_name,
                stoch_d_col_name,
            ]
            if needs_atr_calc and atr_col_name:
                required_cols_for_logic.append(atr_col_name)
            if needs_vol_ma_calc and vol_ma_col_name:
                required_cols_for_logic.append(vol_ma_col_name)

            missing_cols = [col for col in required_cols_for_logic if col not in df_with_indicators.columns]
            if missing_cols:
                log_error(
                    f"CRITICAL: Required columns missing in DataFrame after indicator calculation: {missing_cols}. Check config/data/indicator logic. Available: {df_with_indicators.columns.tolist()}"
                )
                neon_sleep_timer(sleep_interval_seconds)
                continue

            # Extract latest values, ensuring they are valid numbers
            try:
                current_price: float = float(latest_data["close"])
                float(latest_data["high"])
                float(latest_data["low"])
                current_volume: float = float(latest_data["volume"])
                last_rsi: float = float(latest_data[rsi_col_name])
                last_stoch_k: float = float(latest_data[stoch_k_col_name])
                last_stoch_d: float = float(latest_data[stoch_d_col_name])

                last_atr: Optional[float] = None
                if needs_atr_calc and atr_col_name:
                    atr_val = latest_data.get(atr_col_name)
                    if pd.notna(atr_val) and isinstance(atr_val, (float, int)) and atr_val > 0:
                        last_atr = float(atr_val)
                    else:
                        log_warning(
                            f"ATR value ({atr_val}) is invalid (NaN, zero, or negative) for latest candle {latest_timestamp}. ATR-based logic will be skipped or may fail."
                        )

                last_volume_ma: Optional[float] = None
                if needs_vol_ma_calc and vol_ma_col_name:
                    vol_ma_val = latest_data.get(vol_ma_col_name)
                    if pd.notna(vol_ma_val) and isinstance(vol_ma_val, (float, int)) and vol_ma_val > 0:
                        last_volume_ma = float(vol_ma_val)
                    else:
                        log_debug(
                            f"Volume MA value ({vol_ma_val}) is invalid (NaN, zero, or negative) for latest candle {latest_timestamp}. Volume confirmation might be skipped."
                        )

                # Final check for essential NaNs that would break core logic
                essential_values = [current_price, last_rsi, last_stoch_k, last_stoch_d]
                if any(pd.isna(v) for v in essential_values):
                    raise ValueError("Essential indicator value (Price, RSI, StochK, StochD) is NaN.")

            except (KeyError, ValueError, TypeError) as e:
                log_error(
                    f"Error extracting latest data or essential value is NaN/invalid type at {latest_timestamp}: {e}. Data: {latest_data.to_dict()}",
                    exc_info=False,
                )
                neon_sleep_timer(sleep_interval_seconds)
                continue

            # Display Market Stats panel
            display_market_stats(current_price, last_rsi, last_stoch_k, last_stoch_d, last_atr, price_precision_digits)

            # --- Step 5: Identify Order Blocks ---
            bullish_ob, bearish_ob = identify_potential_order_block(
                df_with_indicators,  # Pass DF with all indicators (though OB only needs OHLCV)
                vol_thresh_mult=ob_volume_threshold_multiplier,
                lookback_len=ob_lookback,
            )
            display_order_blocks(bullish_ob, bearish_ob, price_precision_digits)

            # --- Step 6: Apply Trading Logic ---

            # === A. LOGIC IF ALREADY IN A POSITION ===
            if position["status"] is not None:
                log_info(
                    f"Currently in {position['status'].upper()} position entered at {position.get('entry_price', 'N/A'):.{price_precision_digits}f}. Monitoring..."
                )

                # --- A.1: Check Trailing Stop Loss (runs if enabled) ---
                # This function handles all internal checks (enabled, ATR valid, SL ID exists)
                # and executes cancel/place logic if TSL needs to move.
                update_trailing_stop(exchange, symbol, current_price, last_atr)
                # Note: If TSL update fails to place new SL, position might become unprotected.
                # The function logs errors in that case. check_position_and_orders will handle
                # if the *old* SL gets hit during the failed update attempt.

                # --- A.2: Check for Fallback Indicator-Based Exits (Optional) ---
                # This provides an alternative exit mechanism if RSI moves strongly against the position,
                # acting as a potential fallback if SL/TP orders fail or are too slow.
                execute_fallback_exit = False
                fallback_exit_reason = ""
                if position["status"] == "long" and last_rsi > rsi_overbought:
                    fallback_exit_reason = (
                        f"Fallback Exit Signal: RSI ({last_rsi:.1f}) crossed back above Overbought ({rsi_overbought})"
                    )
                    execute_fallback_exit = True
                elif position["status"] == "short" and last_rsi < rsi_oversold:
                    fallback_exit_reason = (
                        f"Fallback Exit Signal: RSI ({last_rsi:.1f}) crossed back below Oversold ({rsi_oversold})"
                    )
                    execute_fallback_exit = True
                # Add other fallback conditions here if desired (e.g., Stoch cross back over threshold)

                if execute_fallback_exit:
                    display_signal("Fallback Exit", position["status"], fallback_exit_reason)
                    log_warning(
                        f"Attempting to close {position['status'].upper()} position via FALLBACK MARKET ORDER due to indicator signal: {fallback_exit_reason}"
                    )

                    # --- Safely Execute Fallback Market Exit ---
                    # 1. Get necessary info from state
                    sl_id_to_cancel = position.get("sl_order_id")
                    tp_id_to_cancel = position.get("tp_order_id")
                    exit_qty = position.get("quantity")  # Get current quantity from state

                    if exit_qty is None or exit_qty <= 0:
                        log_error("Cannot place fallback exit order: Invalid quantity in position state.")
                        # Continue monitoring, hoping SL/TP works
                    else:
                        # 2. Attempt to cancel existing SL and TP orders *before* sending market exit
                        log_info("Attempting to cancel existing SL/TP orders before fallback market exit...")
                        orders_cancelled_successfully = True  # Assume success initially
                        try:
                            if sl_id_to_cancel:
                                log_debug(f"Cancelling fallback SL: {sl_id_to_cancel}")
                                if not cancel_order_with_retry(exchange, sl_id_to_cancel, symbol):
                                    orders_cancelled_successfully = (
                                        False  # Mark failure if helper returns False (unexpected error)
                                    )
                            if tp_id_to_cancel:
                                log_debug(f"Cancelling fallback TP: {tp_id_to_cancel}")
                                if not cancel_order_with_retry(exchange, tp_id_to_cancel, symbol):
                                    orders_cancelled_successfully = False
                        except Exception as cancel_e:
                            # Catch errors raised by cancel_order_with_retry (e.g., after retries fail)
                            log_error(f"Error cancelling SL/TP during fallback exit: {cancel_e}", exc_info=False)
                            orders_cancelled_successfully = False

                        # 3. Place Market Exit Order ONLY if cancellations were successful/not found
                        if not orders_cancelled_successfully:
                            log_error(
                                "Failed to confirm cancellation of one or both SL/TP orders. Aborting fallback market exit to avoid potential issues (e.g., double close). MANUAL INTERVENTION MAY BE REQUIRED."
                            )
                            # Do not proceed with market exit if cancellations failed
                        else:
                            log_info(
                                "SL/TP orders cancelled successfully or already closed. Proceeding with fallback market exit."
                            )
                            close_side = "sell" if position["status"] == "long" else "buy"
                            # Place the market order to close the position (use reduceOnly=True)
                            fallback_exit_order_result = place_market_order(
                                exchange, symbol, close_side, exit_qty, reduce_only=True
                            )

                            # Check if market order placement was likely successful
                            if fallback_exit_order_result and fallback_exit_order_result.get("id"):
                                log_info(
                                    f"Fallback {position['status']} position close market order placed/processed: ID {fallback_exit_order_result.get('id', 'N/A')}"
                                )
                                # Reset state immediately after *attempting* market exit.
                                # check_position_and_orders will verify closure on the next cycle if needed.
                                log_info(
                                    "Resetting local position state immediately after fallback market exit attempt."
                                )
                                position.update(position_default_structure.copy())
                                save_position_state()
                                display_position_status(position, price_precision_digits, amount_precision_digits)
                                # Go to next cycle immediately after attempting exit
                                log_info("Proceeding to next cycle after fallback exit attempt.")
                                continue  # Skip sleep and go to next iteration
                            else:
                                log_error(
                                    f"Fallback market order placement FAILED for {position['status']} position. Result: {fallback_exit_order_result}. POSITION MIGHT STILL BE OPEN. MANUAL INTERVENTION REQUIRED."
                                )
                                # Do not reset state if market order failed. Hope original SL/TP still active.

                else:
                    # If no fallback exit condition met, log normal monitoring status
                    sl_id_str = position.get("sl_order_id") or "N/A"
                    tp_id_str = position.get("tp_order_id") or "N/A"
                    tsl_price_str = (
                        f" (Active TSL: {position['current_trailing_sl_price']:.{price_precision_digits}f})"
                        if position.get("current_trailing_sl_price")
                        else ""
                    )
                    log_info(
                        f"Monitoring {position['status'].upper()} position. Waiting for SL (ID: {sl_id_str}{tsl_price_str}) / TP (ID: {tp_id_str}) or TSL update."
                    )

            # === B. LOGIC TO CHECK FOR NEW ENTRIES (Only if not currently in a position) ===
            else:  # position['status'] is None
                log_info("No active position. Checking for new entry signals...")

                # --- B.1: Volume Confirmation Check ---
                volume_confirmed = False
                if entry_volume_confirmation_enabled:
                    if current_volume is not None and last_volume_ma is not None and last_volume_ma > 0:
                        volume_threshold = last_volume_ma * entry_volume_multiplier
                        if current_volume > volume_threshold:
                            volume_confirmed = True
                            log_debug(
                                f"Volume confirmed: Current Vol ({current_volume:.2f}) > Threshold ({volume_threshold:.2f} = {last_volume_ma:.2f} * {entry_volume_multiplier})"
                            )
                        else:
                            log_debug(
                                f"Volume NOT confirmed: Current Vol ({current_volume:.2f}) <= Threshold ({volume_threshold:.2f})"
                            )
                    else:
                        log_debug("Volume confirmation enabled but current volume or MA data is missing/invalid.")
                else:
                    volume_confirmed = True  # Always true if disabled in config

                # --- B.2: Base Signal Conditions (RSI + Stoch Oversold/Overbought) ---
                base_long_signal = (
                    last_rsi < rsi_oversold and last_stoch_k < stoch_oversold and last_stoch_d < stoch_oversold
                )  # Added Stoch D check
                base_short_signal = (
                    last_rsi > rsi_overbought and last_stoch_k > stoch_overbought and last_stoch_d > stoch_overbought
                )  # Added Stoch D check
                if base_long_signal:
                    log_debug(
                        f"Base Long Signal Met: RSI={last_rsi:.1f}, StochK={last_stoch_k:.1f}, StochD={last_stoch_d:.1f}"
                    )
                if base_short_signal:
                    log_debug(
                        f"Base Short Signal Met: RSI={last_rsi:.1f}, StochK={last_stoch_k:.1f}, StochD={last_stoch_d:.1f}"
                    )

                # --- B.3: Order Block Price Proximity Check ---
                ob_price_confirmed = False
                ob_reason_part = ""
                active_ob_for_entry: Optional[Dict] = None  # Store the OB used for entry signal

                if base_long_signal and bullish_ob:
                    # Define entry zone: price should be near or within the bullish OB
                    bullish_ob["high"] - bullish_ob["low"]
                    # Allow entry if price is within OB or slightly above (e.g., 10% of range or a few ticks)
                    # More conservative: Require price to be *within* the OB range
                    entry_zone_high = bullish_ob["high"]  # + max(ob_range * 0.10, min_tick * 3) # Allow slightly above?
                    entry_zone_low = bullish_ob["low"]
                    if entry_zone_low <= current_price <= entry_zone_high:
                        ob_price_confirmed = True
                        ob_reason_part = f"Price ({current_price:.{price_precision_digits}f}) within Bullish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"
                        active_ob_for_entry = bullish_ob
                        log_debug(f"Bullish OB Price Confirmed: {ob_reason_part}")
                    else:
                        log_debug(
                            f"Base Long signal met, but Price ({current_price:.{price_precision_digits}f}) outside Bullish OB zone [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]."
                        )
                elif base_long_signal:
                    log_debug("Base Long signal met, but no recent Bullish OB found or used for confirmation.")
                    # Set ob_price_confirmed to True if OB is not required for entry? Or keep False?
                    # Current logic requires OB proximity. If OB optional, change this.
                    # ob_price_confirmed = True # If OB is optional

                # Check short OB only if long signal wasn't met or didn't confirm
                if not ob_price_confirmed and base_short_signal and bearish_ob:
                    # Define entry zone: price should be near or within the bearish OB
                    bearish_ob["high"] - bearish_ob["low"]
                    # Allow entry if price is within OB or slightly below
                    entry_zone_high = bearish_ob["high"]
                    entry_zone_low = bearish_ob["low"]  # - max(ob_range * 0.10, min_tick * 3) # Allow slightly below?
                    if entry_zone_low <= current_price <= entry_zone_high:
                        ob_price_confirmed = True
                        ob_reason_part = f"Price ({current_price:.{price_precision_digits}f}) within Bearish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"
                        active_ob_for_entry = bearish_ob
                        log_debug(f"Bearish OB Price Confirmed: {ob_reason_part}")
                    else:
                        log_debug(
                            f"Base Short signal met, but Price ({current_price:.{price_precision_digits}f}) outside Bearish OB zone [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]."
                        )
                elif not ob_price_confirmed and base_short_signal:
                    log_debug("Base Short signal met, but no recent Bearish OB found or used for confirmation.")
                    # ob_price_confirmed = True # If OB is optional

                # --- B.4: Combine Conditions for Final Entry Signal ---
                # Strategy: Base Signal AND OB Price Confirmation AND Volume Confirmation
                long_entry_signal = (
                    base_long_signal
                    and ob_price_confirmed
                    and volume_confirmed
                    and active_ob_for_entry
                    and active_ob_for_entry["type"] == "bullish"
                )
                short_entry_signal = (
                    base_short_signal
                    and ob_price_confirmed
                    and volume_confirmed
                    and active_ob_for_entry
                    and active_ob_for_entry["type"] == "bearish"
                )

                # Construct detailed reason strings for logging/display
                entry_reason = ""
                side_to_enter: Optional[str] = None

                if long_entry_signal:
                    side_to_enter = "long"  # Use 'long'/'short' for internal logic, 'buy'/'sell' for orders
                    entry_reason = (
                        f"RSI ({last_rsi:.1f} < {rsi_oversold}), "
                        f"Stoch ({last_stoch_k:.1f},{last_stoch_d:.1f} < {stoch_oversold}), "
                        f"{ob_reason_part}" + (", Volume Confirmed" if entry_volume_confirmation_enabled else "")
                    )
                elif short_entry_signal:
                    side_to_enter = "short"
                    entry_reason = (
                        f"RSI ({last_rsi:.1f} > {rsi_overbought}), "
                        f"Stoch ({last_stoch_k:.1f},{last_stoch_d:.1f} > {stoch_overbought}), "
                        f"{ob_reason_part}" + (", Volume Confirmed" if entry_volume_confirmation_enabled else "")
                    )
                # Log reasons for non-entry if signals were close
                elif (
                    base_long_signal
                    and ob_price_confirmed
                    and not volume_confirmed
                    and entry_volume_confirmation_enabled
                ):
                    log_debug("Long entry blocked: Volume not confirmed.")
                elif (
                    base_short_signal
                    and ob_price_confirmed
                    and not volume_confirmed
                    and entry_volume_confirmation_enabled
                ):
                    log_debug("Short entry blocked: Volume not confirmed.")
                elif (base_long_signal or base_short_signal) and not ob_price_confirmed:
                    log_debug(
                        f"Entry blocked: Base signal met, but OB price condition not met (Price: {current_price:.{price_precision_digits}f})."
                    )

                # --- B.5: Execute Entry Sequence ---
                if side_to_enter and entry_reason and active_ob_for_entry:
                    display_signal("Entry", side_to_enter, entry_reason)

                    # --- Calculate SL/TP Prices ---
                    stop_loss_price: Optional[float] = None
                    take_profit_price: Optional[float] = None
                    sl_tp_method = "Unknown"

                    if enable_atr_sl_tp:
                        sl_tp_method = "ATR"
                        if last_atr:  # Ensure last_atr is valid (positive)
                            if side_to_enter == "long":
                                stop_loss_price = current_price - (last_atr * atr_sl_multiplier)
                                take_profit_price = current_price + (last_atr * atr_tp_multiplier)
                            else:  # short
                                stop_loss_price = current_price + (last_atr * atr_sl_multiplier)
                                take_profit_price = current_price - (last_atr * atr_tp_multiplier)
                            log_info(
                                f"Calculated ATR SL/TP: SL={stop_loss_price:.{price_precision_digits}f} ({atr_sl_multiplier}x ATR), TP={take_profit_price:.{price_precision_digits}f} ({atr_tp_multiplier}x ATR)"
                            )
                        else:
                            log_error(
                                f"Cannot calculate ATR SL/TP for {side_to_enter.upper()} entry: Invalid ATR value ({last_atr}). Skipping entry."
                            )
                            continue  # Skip entry attempt this cycle
                    else:  # Fixed percentage
                        sl_tp_method = "Fixed %"
                        if side_to_enter == "long":
                            stop_loss_price = current_price * (1 - stop_loss_percentage)
                            take_profit_price = current_price * (1 + take_profit_percentage)
                        else:  # short
                            stop_loss_price = current_price * (1 + stop_loss_percentage)
                            take_profit_price = current_price * (1 - take_profit_percentage)
                        log_info(
                            f"Calculated Fixed % SL/TP: SL={stop_loss_price:.{price_precision_digits}f} ({stop_loss_percentage * 100:.1f}%), TP={take_profit_price:.{price_precision_digits}f} ({take_profit_percentage * 100:.1f}%)"
                        )

                    # --- Refine SL based on Order Block ---
                    # Place SL just beyond the low/high of the OB that triggered the entry signal
                    # This provides a potentially tighter and more structurally sound stop.
                    ob_low = active_ob_for_entry["low"]
                    ob_high = active_ob_for_entry["high"]
                    adjusted_sl: Optional[float] = None
                    sl_adjustment_buffer_ticks = min_tick * 5  # Buffer in ticks beyond the OB edge

                    if side_to_enter == "long":
                        # Potential OB SL is just below the OB low
                        potential_ob_sl = ob_low - sl_adjustment_buffer_ticks
                        # Use OB SL only if it's TIGHTER (higher) than the calculated SL and still below current price
                        if potential_ob_sl > stop_loss_price and potential_ob_sl < current_price:
                            adjusted_sl = potential_ob_sl
                            log_info(
                                f"Adjusting SL tighter based on Bullish OB low: {adjusted_sl:.{price_precision_digits}f} (Orig: {stop_loss_price:.{price_precision_digits}f})"
                            )
                        else:
                            log_debug(
                                f"Original {sl_tp_method} SL ({stop_loss_price:.{price_precision_digits}f}) kept. OB-based SL ({potential_ob_sl:.{price_precision_digits}f}) was not tighter or invalid."
                            )
                    else:  # short
                        # Potential OB SL is just above the OB high
                        potential_ob_sl = ob_high + sl_adjustment_buffer_ticks
                        # Use OB SL only if it's TIGHTER (lower) than the calculated SL and still above current price
                        if potential_ob_sl < stop_loss_price and potential_ob_sl > current_price:
                            adjusted_sl = potential_ob_sl
                            log_info(
                                f"Adjusting SL tighter based on Bearish OB high: {adjusted_sl:.{price_precision_digits}f} (Orig: {stop_loss_price:.{price_precision_digits}f})"
                            )
                        else:
                            log_debug(
                                f"Original {sl_tp_method} SL ({stop_loss_price:.{price_precision_digits}f}) kept. OB-based SL ({potential_ob_sl:.{price_precision_digits}f}) was not tighter or invalid."
                            )

                    if adjusted_sl is not None:
                        # Format adjusted SL to precision before assigning
                        try:
                            stop_loss_price = float(exchange.price_to_precision(symbol, adjusted_sl))
                            log_debug(f"Using OB-adjusted SL: {stop_loss_price:.{price_precision_digits}f}")
                        except ccxt.ExchangeError as fmt_e:
                            log_error(
                                f"Failed to format OB-adjusted SL {adjusted_sl} to precision: {fmt_e}. Using original SL {stop_loss_price}."
                            )
                            # Fallback to original calculated SL if formatting fails

                    # --- Final SL/TP Validation ---
                    if stop_loss_price is None or take_profit_price is None:
                        log_error(
                            f"SL or TP price calculation failed unexpectedly. Skipping {side_to_enter.upper()} entry."
                        )
                        continue
                    try:
                        # Format prices just before use
                        sl_price_final = float(exchange.price_to_precision(symbol, stop_loss_price))
                        tp_price_final = float(exchange.price_to_precision(symbol, take_profit_price))
                    except ccxt.ExchangeError as fmt_e:
                        log_error(f"Failed to format final SL/TP prices to precision: {fmt_e}. Skipping entry.")
                        continue

                    # Check logical validity: SL must protect entry, TP must be beyond entry
                    if side_to_enter == "long" and (sl_price_final >= current_price or tp_price_final <= current_price):
                        log_error(
                            f"Invalid LONG SL/TP after adjustments: SL {sl_price_final:.{price_precision_digits}f}, TP {tp_price_final:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping entry."
                        )
                        continue
                    if side_to_enter == "short" and (
                        sl_price_final <= current_price or tp_price_final >= current_price
                    ):
                        log_error(
                            f"Invalid SHORT SL/TP after adjustments: SL {sl_price_final:.{price_precision_digits}f}, TP {tp_price_final:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping entry."
                        )
                        continue

                    # --- Calculate Position Size ---
                    # Use the final validated SL price for sizing
                    entry_quantity = calculate_position_size(
                        exchange, symbol, current_price, sl_price_final, risk_percentage
                    )

                    if entry_quantity is None or entry_quantity <= 0:
                        log_error(
                            f"Failed to calculate valid position size (Result: {entry_quantity}). Skipping {side_to_enter.upper()} entry."
                        )
                        continue  # Skip entry attempt

                    # --- Place Entry Market Order ---
                    entry_order_side = "buy" if side_to_enter == "long" else "sell"
                    entry_order_result = place_market_order(
                        exchange, symbol, entry_order_side, entry_quantity, reduce_only=False
                    )  # Entry is never reduceOnly

                    # --- Process Entry Result ---
                    # Check if order was placed and likely filled/accepted
                    # Market orders might return 'open' initially but fill quickly. Check for ID and positive filled amount.
                    entry_successful = False
                    entry_order_id = None
                    entry_price_actual = current_price  # Default to current price if avg not available
                    filled_quantity_actual = entry_quantity  # Default to calculated qty

                    if entry_order_result and entry_order_result.get("id"):
                        entry_order_id = entry_order_result["id"]
                        order_status = entry_order_result.get("status")
                        avg_price = entry_order_result.get("average")
                        filled_qty = entry_order_result.get("filled")

                        if avg_price and isinstance(avg_price, (float, int)) and avg_price > 0:
                            entry_price_actual = avg_price
                        if filled_qty and isinstance(filled_qty, (float, int)) and filled_qty > 0:
                            filled_quantity_actual = filled_qty

                        # Consider entry successful if order has an ID and status is closed or filled > 0
                        # (Some exchanges might return 'open' briefly for market orders)
                        if order_status == "closed" or (order_status == "open" and filled_quantity_actual > 0):
                            entry_successful = True
                            log_info(
                                f"{side_to_enter.upper()} position entry order processed: ID {entry_order_id}, Status: {order_status}, AvgPrice: {entry_price_actual:.{price_precision_digits}f}, FilledQty: {filled_quantity_actual:.{amount_precision_digits}f}"
                            )
                        else:
                            # Order might be pending, rejected, or failed unexpectedly
                            log_error(
                                f"Entry market order {entry_order_id} placed but status is '{order_status}' with filled amount {filled_qty}. Assuming entry failed or pending. State not updated."
                            )
                    else:
                        log_error(
                            f"Failed to place or confirm {side_to_enter.upper()} entry market order. Result: {entry_order_result}"
                        )

                    # --- Place SL/TP Orders and Update State ONLY if Entry Successful ---
                    if entry_successful:
                        log_info(f"Entry successful. Placing SL and TP orders for position ID {entry_order_id}...")
                        # Place SL/TP orders *after* confirming entry fill/placement
                        # Use the actual filled quantity and final SL/TP prices
                        sl_order_result, tp_order_result = place_sl_tp_orders(
                            exchange,
                            symbol,
                            position_side=side_to_enter,  # Pass 'long' or 'short'
                            quantity=filled_quantity_actual,
                            sl_price=sl_price_final,
                            tp_price=tp_price_final,
                        )

                        # --- Update Position State ---
                        sl_order_id_final = sl_order_result.get("id") if sl_order_result else None
                        tp_order_id_final = tp_order_result.get("id") if tp_order_result else None

                        position.update(
                            {
                                "status": side_to_enter,  # 'long' or 'short'
                                "entry_price": entry_price_actual,
                                "quantity": filled_quantity_actual,
                                "order_id": entry_order_id,  # ID of the entry market order
                                "stop_loss": sl_price_final,  # Store the calculated/adjusted SL price
                                "take_profit": tp_price_final,  # Store the calculated TP price
                                "entry_time": pd.Timestamp.now(tz="UTC"),  # Record entry time
                                "sl_order_id": sl_order_id_final,
                                "tp_order_id": tp_order_id_final,
                                # Initialize TSL fields upon entry
                                "highest_price_since_entry": entry_price_actual
                                if side_to_enter == "long"
                                else entry_price_actual,  # Start peak/low at entry
                                "lowest_price_since_entry": entry_price_actual
                                if side_to_enter == "short"
                                else entry_price_actual,
                                "current_trailing_sl_price": None,  # TSL starts inactive
                            }
                        )
                        save_position_state()  # Save the new active position state
                        display_position_status(
                            position, price_precision_digits, amount_precision_digits
                        )  # Show new status

                        # Log warning if SL/TP placement failed after successful entry
                        if not sl_order_id_final:
                            log_warning(
                                "Entry successful, but SL order placement failed or did not return ID. POSITION IS UNPROTECTED!"
                            )
                        if not tp_order_id_final:
                            log_warning("Entry successful, but TP order placement failed or did not return ID.")
                    else:
                        # Entry failed, do not update state
                        log_error(f"Entry attempt for {side_to_enter.upper()} failed. Position state not updated.")
                        # Potential TODO: Should we try to cancel the failed/pending entry order here?
                        # If entry_order_id exists but status wasn't 'closed'/'filled>0'. Risky.

                else:  # No entry signal met
                    log_info("Entry conditions not met this cycle.")

            # --- Step 7: Wait for the next cycle ---
            log_info(
                f"Cycle complete ({pd.Timestamp.now(tz='UTC') - cycle_start_time}). Waiting for {sleep_interval_seconds} seconds..."
            )
            neon_sleep_timer(sleep_interval_seconds)

        # --- Graceful Shutdown Handling ---
        except KeyboardInterrupt:
            log_info("Keyboard interrupt detected (Ctrl+C). Initiating graceful shutdown...")
            save_position_state()  # Save final state before exiting

            # Attempt to cancel all open orders for the symbol if not in simulation mode
            if not SIMULATION_MODE:
                log_warning(f"!!! LIVE MODE: Attempting to cancel all open orders for {symbol} on shutdown...")
                try:
                    # Fetch open orders first (use retry)
                    @api_retry_decorator
                    def fetch_open_orders_on_exit(exch: ccxt.Exchange, sym: str) -> List[Dict]:
                        """Fetches open orders for shutdown cancellation with retry."""
                        log_debug(f"Fetching open orders for {sym} to cancel on exit...")
                        return exch.fetch_open_orders(sym)

                    open_orders = fetch_open_orders_on_exit(exchange, symbol)

                    if not open_orders:
                        log_info("No open orders found for {symbol} to cancel.")
                    else:
                        log_warning(f"Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...")
                        cancelled_count = 0
                        failed_count = 0
                        for order in open_orders:
                            order_id = order.get("id")
                            order_info = f"ID: {order_id} (Type: {order.get('type', 'N/A')}, Side: {order.get('side', 'N/A')}, Price: {order.get('price', 'N/A')})"
                            if not order_id:
                                log_warning(f"Skipping order cancellation - missing ID: {order}")
                                continue
                            try:
                                log_info(f"Cancelling order -> {order_info}...")
                                # Use cancel helper with retry (handles OrderNotFound)
                                if cancel_order_with_retry(exchange, order_id, symbol):
                                    cancelled_count += 1
                                else:
                                    # This means cancel_order_with_retry encountered an unexpected error after retries
                                    failed_count += 1
                                time.sleep(0.3)  # Small delay between cancellations to avoid rate limits
                            except Exception as cancel_e:
                                # Catch errors raised by cancel_order_with_retry (e.g., after retries fail)
                                log_error(f"Failed to cancel order {order_info} on exit: {cancel_e}", exc_info=False)
                                failed_count += 1

                        log_info(
                            f"Shutdown order cancellation attempt complete. Success/Already Closed: {cancelled_count}, Failed: {failed_count}"
                        )
                        if failed_count > 0:
                            log_error(
                                f"{failed_count} orders could not be cancelled automatically. Please check the exchange manually: {symbol}"
                            )

                except Exception as e:
                    # Catch errors during fetch_open_orders_on_exit itself
                    log_error(f"Error fetching or cancelling open orders on exit for {symbol}: {e}", exc_info=True)
                    log_error("Manual check of open orders on the exchange is strongly recommended.")
            else:
                log_info("Simulation mode: Skipping order cancellation on exit.")

            break  # Exit the main while loop

        # --- Robust Error Handling for the Main Loop ---
        except ccxt.RateLimitExceeded as e:
            log_error(
                f"Main loop caught RateLimitExceeded: {e}. Waiting longer ({sleep_interval_seconds + 60}s)...",
                exc_info=False,
            )
            neon_sleep_timer(sleep_interval_seconds + 60)  # Wait longer than usual
        except ccxt.NetworkError as e:
            log_error(
                f"Main loop caught NetworkError: {e}. Waiting {sleep_interval_seconds}s and relying on API call retries within next cycle...",
                exc_info=False,
            )
            neon_sleep_timer(sleep_interval_seconds)  # Rely on retry decorator within API calls for backoff
        except ccxt.ExchangeError as e:  # Catch other potentially recoverable exchange issues
            log_error(
                f"Main loop caught ExchangeError: {e}. Waiting {sleep_interval_seconds}s and retrying...",
                exc_info=False,
            )
            # Could add specific handling for maintenance errors etc. here if needed
            neon_sleep_timer(sleep_interval_seconds)
        except Exception as e:
            # Catch any other unexpected error to prevent the bot from crashing entirely
            log_error(f"CRITICAL UNEXPECTED ERROR in main loop: {type(e).__name__}: {e}", exc_info=True)
            log_info("Attempting to recover by saving state and waiting 60s before next cycle...")
            try:
                save_position_state()  # Attempt to save state on critical error
            except Exception as save_e:
                log_error(f"Failed to save state during critical error handling: {save_e}", exc_info=True)
            neon_sleep_timer(60)  # Wait a bit before trying next cycle

    # --- Bot Exit ---
    print_shutdown_message()
    log_info("Bot shutdown complete.")


# --- Script Execution ---
if __name__ == "__main__":
    try:
        run_bot()
    except SystemExit as e:
        # Catch sys.exit calls for clean termination (e.g., from config errors)
        log_info(f"Bot exited with status code: {e.code}")
        sys.exit(e.code)
    except Exception as main_exec_e:
        # Catch any truly unexpected top-level error during setup before the loop
        log_error(
            f"A critical error occurred outside the main loop during initialization: {main_exec_e}", exc_info=True
        )
        print_shutdown_message()
        sys.exit(1)  # Exit with error code
    finally:
        # Ensure logs are flushed, etc. (if using file handlers)
        logging.shutdown()
