import ccxt
import os
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import json
import sys
import math
from pathlib import Path  # For cleaner path handling
from typing import Optional, Tuple, Dict, Any, List, Union, Callable, Final  # Added Final
import functools

# --- Load Environment Variables FIRST ---
# Explicitly load .env from the script's directory or CWD
try:
    # Try to find the script's directory
    script_dir = Path(__file__).resolve().parent
    dotenv_path = script_dir / ".env"
    print(f"Attempting to load environment variables from: {dotenv_path}")
except NameError:
    # __file__ might not be defined (e.g., interactive mode, frozen executable)
    script_dir = Path.cwd()
    dotenv_path = script_dir / ".env"
    print(
        f"Warning: Could not determine script directory reliably. Looking for .env in current working directory: {dotenv_path}"
    )

# Use override=False to not overwrite existing environment variables set elsewhere
load_success = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=False)

if not load_success:
    if dotenv_path.exists():
        print(
            f"Warning: Found .env file at {dotenv_path}, but load_dotenv() reported failure. Check file permissions or content formatting."
        )
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
    # Define empty strings as fallbacks
    NEON_GREEN = NEON_PINK = NEON_CYAN = NEON_RED = NEON_YELLOW = NEON_BLUE = RESET = ""
    COLORAMA_AVAILABLE = False

# --- Logging Configuration ---
log_format_base: str = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format_base, datefmt="%Y-%m-%d %H:%M:%S")
logger: logging.Logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_PRICE_PRECISION: Final[int] = 4
DEFAULT_AMOUNT_PRECISION: Final[int] = 8
POSITION_STATE_FILE: Final[str] = "position_state.json"
CONFIG_FILE: Final[str] = "config.json"
SIMULATION_ORDER_PREFIX: Final[str] = "sim_"


# --- Neon Display Functions (Enhanced clarity and consistency) ---
def print_neon_header() -> None:
    """Prints a visually appealing header for the bot."""
    box_width = 70
    print(f"{NEON_CYAN}{'=' * box_width}{RESET}")
    print(f"{NEON_PINK}{Style.BRIGHT}{'Enhanced RSI/OB Trader Neon Bot - v1.2':^{box_width}}{RESET}")
    print(f"{NEON_CYAN}{'=' * box_width}{RESET}")


def display_error_box(message: str) -> None:
    """Displays a message in a neon red error box."""
    box_width = 70
    print(f"{NEON_RED}{'!' * box_width}{RESET}")
    print(f"{NEON_RED}! {message.strip():^{box_width - 4}} !{RESET}")
    print(f"{NEON_RED}{'!' * box_width}{RESET}")


def display_warning_box(message: str) -> None:
    """Displays a message in a neon yellow warning box."""
    box_width = 70
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")
    print(f"{NEON_YELLOW}~ {message.strip():^{box_width - 4}} ~{RESET}")
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")


# Wrapper functions for logging with color
def log_info(msg: str) -> None:
    logger.info(f"{NEON_GREEN}{msg}{RESET}")


def log_error(msg: str, exc_info: bool = False) -> None:
    # Display the first line in a box for emphasis
    display_error_box(msg.split("\n", 1)[0])
    logger.error(f"{NEON_RED}{msg}{RESET}", exc_info=exc_info)


def log_warning(msg: str) -> None:
    display_warning_box(msg)
    logger.warning(f"{NEON_YELLOW}{msg}{RESET}")


def log_debug(msg: str) -> None:
    logger.debug(f"{Fore.WHITE}{msg}{RESET}")  # Use standard white for debug


def print_cycle_divider(timestamp: pd.Timestamp) -> None:
    """Prints a divider indicating the start of a new trading cycle."""
    box_width = 70
    print(f"\n{NEON_BLUE}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}Cycle Start: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}{RESET}")
    print(f"{NEON_BLUE}{'=' * box_width}{RESET}")


def display_position_status(position: Dict[str, Any], price_precision: int, amount_precision: int) -> None:
    """Displays enhanced position status, including tracked order IDs and warnings."""
    status = position.get("status")
    entry_price = position.get("entry_price")
    quantity = position.get("quantity")
    sl_target = position.get("stop_loss")  # The target SL price
    tp_target = position.get("take_profit")  # The target TP price
    tsl_active_price = position.get("current_trailing_sl_price")  # Current TSL if active
    entry_time = position.get("entry_time")

    # Format numbers safely, handling None
    entry_str = f"{entry_price:.{price_precision}f}" if isinstance(entry_price, (float, int)) else "N/A"
    qty_str = f"{quantity:.{amount_precision}f}" if isinstance(quantity, (float, int)) else "N/A"
    sl_str = f"{sl_target:.{price_precision}f}" if isinstance(sl_target, (float, int)) else "N/A"
    tp_str = f"{tp_target:.{price_precision}f}" if isinstance(tp_target, (float, int)) else "N/A"
    tsl_str = (
        f" | Active TSL: {tsl_active_price:.{price_precision}f}" if isinstance(tsl_active_price, (float, int)) else ""
    )
    time_str = f" | Entered: {entry_time.strftime('%Y-%m-%d %H:%M')}" if isinstance(entry_time, pd.Timestamp) else ""

    if status == "long":
        status_color, status_text = NEON_GREEN, "LONG"
    elif status == "short":
        status_color, status_text = NEON_RED, "SHORT"
    else:
        status_color, status_text = NEON_CYAN, "None"

    print(
        f"{status_color}Position Status: {status_text}{RESET} | Entry: {entry_str} | Qty: {qty_str} | Target SL: {sl_str} | Target TP: {tp_str}{tsl_str}{time_str}"
    )

    # Display protection order IDs
    sl_id, tp_id, oco_id = position.get("sl_order_id"), position.get("tp_order_id"), position.get("oco_order_id")
    protection_info = []
    if oco_id:
        protection_info.append(f"OCO ID: {oco_id}")
    else:
        if sl_id:
            protection_info.append(f"SL ID: {sl_id}")
        if tp_id:
            protection_info.append(f"TP ID: {tp_id}")

    if protection_info:
        print(f"    Protection Orders: {', '.join(protection_info)}")
    elif status is not None:  # Position is active but no protection tracked
        print(
            f"    {NEON_YELLOW}Warning: Position active but no protection order IDs are currently tracked in state.{RESET}"
        )


def display_market_stats(
    current_price: float, rsi: float, stoch_k: float, stoch_d: float, atr: Optional[float], price_precision: int
) -> None:
    """Displays key market indicators."""
    print(f"{NEON_PINK}--- Market Stats ---{RESET}")
    print(f"{NEON_GREEN}Price:{RESET}  {current_price:.{price_precision}f}")
    print(f"{NEON_CYAN}RSI:{RESET}    {rsi:.2f}")
    print(f"{NEON_YELLOW}StochK:{RESET} {stoch_k:.2f}")
    print(f"{NEON_YELLOW}StochD:{RESET} {stoch_d:.2f}")
    if atr is not None:
        print(f"{NEON_BLUE}ATR:{RESET}    {atr:.{price_precision}f}")  # Use price precision for ATR
    print(f"{NEON_PINK}--------------------{RESET}")


def display_order_blocks(
    bullish_ob: Optional[Dict[str, Any]], bearish_ob: Optional[Dict[str, Any]], price_precision: int
) -> None:
    """Displays identified potential order blocks."""
    found = False
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
    """Displays an entry or exit signal."""
    color = NEON_GREEN if direction.lower() == "long" else NEON_RED if direction.lower() == "short" else NEON_YELLOW
    print(f"{color}{Style.BRIGHT}*** {signal_type.upper()} {direction.upper()} SIGNAL ***{RESET}\n   Reason: {reason}")


def neon_sleep_timer(seconds: int) -> None:
    """Displays a countdown timer with decreasing urgency colors."""
    if not COLORAMA_AVAILABLE or seconds <= 0:
        if seconds > 0:
            print(f"Sleeping for {seconds} seconds...")
            time.sleep(seconds)
        return

    interval: float = 0.5  # Update interval
    steps: int = int(seconds / interval)

    for i in range(steps, -1, -1):
        remaining_seconds = max(0, round(i * interval))
        # Change color based on remaining time
        if remaining_seconds <= 5 and i % 2 == 0:
            color = NEON_RED  # Flashing red
        elif remaining_seconds <= 15:
            color = NEON_YELLOW
        else:
            color = NEON_CYAN

        print(f"{color}Next cycle in: {remaining_seconds} seconds... {RESET}", end="\r")
        time.sleep(interval)

    # Clear the line after countdown finishes
    print(" " * 50, end="\r")


def print_shutdown_message() -> None:
    """Prints a clean shutdown message."""
    box_width = 70
    print(f"\n{NEON_PINK}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}{Style.BRIGHT}{'RSI/OB Trader Bot Stopped - Goodbye!':^{box_width}}{RESET}")
    print(f"{NEON_PINK}{'=' * box_width}{RESET}")


# --- Retry Decorator ---
# Define specific exceptions that warrant a retry
RETRYABLE_EXCEPTIONS: Tuple[Type[ccxt.NetworkError], ...] = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
)


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
                        log_error(
                            f"API call '{func.__name__}' failed after {max_retries} retries. Last error: {type(e).__name__}: {e}",
                            exc_info=False,
                        )
                        raise last_exception  # Re-raise the last caught retryable exception
                    log_warning(
                        f"API call '{func.__name__}' failed due to {type(e).__name__}. Retrying in {delay:.1f}s ({retries}/{max_retries})..."
                    )
                    neon_sleep_timer(int(round(delay)))  # Use the visual timer
                    delay *= backoff_factor
                except Exception as e:
                    # Log and re-raise non-retryable exceptions immediately
                    log_error(
                        f"Non-retryable error in API call '{func.__name__}': {type(e).__name__}: {e}", exc_info=True
                    )
                    raise
            # This part should ideally be unreachable if max_retries >= 0
            if last_exception:
                raise last_exception
            raise RuntimeError(
                f"API call '{func.__name__}' failed unexpectedly after exhausting retries without raising."
            )  # Should not happen

        return wrapper

    return decorator


# --- Configuration Loading ---
def load_config(filename: str = CONFIG_FILE) -> Dict[str, Any]:
    """Loads, validates, and returns the configuration from a JSON file."""
    log_info(f"Loading configuration from '{filename}'...")
    config_path = Path(filename)
    if not config_path.exists():
        log_error(f"CRITICAL: Configuration file '{filename}' not found. Please create it.")
        sys.exit(1)

    try:
        with config_path.open("r") as f:
            cfg: Dict[str, Any] = json.load(f)
        log_info(f"Configuration file '{filename}' loaded successfully.")

        # --- Configuration Validation ---
        errors: List[str] = []

        def validate(
            key: str,
            types: Union[type, Tuple[type, ...]],
            condition: Optional[Callable[[Any], bool]] = None,
            required: bool = True,
        ):
            if key not in cfg:
                if required:
                    errors.append(f"Missing required key: '{key}'")
                return  # Skip further checks if missing and not required
            val = cfg[key]
            if not isinstance(val, types):
                errors.append(f"Key '{key}': Expected type {types}, but got {type(val).__name__}.")
            elif condition and not condition(val):
                errors.append(f"Key '{key}' (value: {val}) failed validation condition.")

        # Core settings
        validate("exchange_id", str, lambda x: len(x) > 0)
        validate("symbol", str, lambda x: len(x) > 0)
        validate("timeframe", str, lambda x: len(x) > 0)
        validate("risk_percentage", (float, int), lambda x: 0 < x < 1, required=True)
        validate("simulation_mode", bool, required=True)
        validate("data_limit", int, lambda x: x > 50)  # Need enough data for lookbacks
        validate("sleep_interval_seconds", int, lambda x: x > 0)

        # Indicator settings
        validate("rsi_length", int, lambda x: x > 0)
        validate("rsi_overbought", int, lambda x: 50 < x <= 100)
        validate("rsi_oversold", int, lambda x: 0 <= x < 50)
        if (
            "rsi_oversold" in cfg
            and "rsi_overbought" in cfg
            and cfg.get("rsi_oversold", 0) >= cfg.get("rsi_overbought", 100)
        ):
            errors.append("rsi_oversold must be strictly less than rsi_overbought")

        validate("stoch_k", int, lambda x: x > 0)
        validate("stoch_d", int, lambda x: x > 0)
        validate("stoch_smooth_k", int, lambda x: x > 0)
        validate("stoch_overbought", int, lambda x: 50 < x <= 100)
        validate("stoch_oversold", int, lambda x: 0 <= x < 50)
        if (
            "stoch_oversold" in cfg
            and "stoch_overbought" in cfg
            and cfg.get("stoch_oversold", 0) >= cfg.get("stoch_overbought", 100)
        ):
            errors.append("stoch_oversold must be strictly less than stoch_overbought")

        # Order Block settings
        validate("ob_volume_threshold_multiplier", (float, int), lambda x: x > 0)
        validate("ob_lookback", int, lambda x: x > 0)

        # Volume confirmation settings (conditional)
        validate("entry_volume_confirmation_enabled", bool)
        if cfg.get("entry_volume_confirmation_enabled"):
            validate("entry_volume_ma_length", int, lambda x: x > 0)
            validate("entry_volume_multiplier", (float, int), lambda x: x > 0)

        # SL/TP and Trailing Stop settings (conditional dependencies)
        validate("enable_atr_sl_tp", bool)
        validate("enable_trailing_stop", bool)
        needs_atr = cfg.get("enable_atr_sl_tp") or cfg.get("enable_trailing_stop")
        if needs_atr:
            validate("atr_length", int, lambda x: x > 0, required=True)  # Required if ATR features used

        if cfg.get("enable_atr_sl_tp"):
            validate("atr_sl_multiplier", (float, int), lambda x: x > 0, required=True)
            validate("atr_tp_multiplier", (float, int), lambda x: x > 0, required=True)
        else:  # Fixed % SL/TP required if ATR SL/TP is disabled
            validate("stop_loss_percentage", (float, int), lambda x: x > 0, required=True)
            validate("take_profit_percentage", (float, int), lambda x: x > 0, required=True)

        if cfg.get("enable_trailing_stop"):
            validate("trailing_stop_atr_multiplier", (float, int), lambda x: x > 0, required=True)
            validate(
                "trailing_stop_activation_atr_multiplier", (float, int), lambda x: x >= 0, required=True
            )  # Can be 0 to activate immediately

        # Optional Retry settings
        validate("retry_max_retries", int, lambda x: x >= 0, required=False)
        validate("retry_initial_delay", (float, int), lambda x: x > 0, required=False)
        validate("retry_backoff_factor", (float, int), lambda x: x >= 1, required=False)

        # --- Report Validation Results ---
        if errors:
            error_str = "\n - ".join(errors)
            log_error(f"CRITICAL: Configuration validation failed with {len(errors)} errors:\n - {error_str}")
            sys.exit(1)
        else:
            log_info("Configuration validation passed.")
            return cfg

    except json.JSONDecodeError as e:
        log_error(
            f"CRITICAL: Error decoding JSON from '{filename}'. Check for syntax errors (e.g., trailing commas): {e}"
        )
        sys.exit(1)
    except Exception as e:
        log_error(f"CRITICAL: Unexpected error loading or validating config file '{filename}': {e}", exc_info=True)
        sys.exit(1)


# --- Load Config & Create Retry Decorator Instance ---
config: Dict[str, Any] = load_config()
# Create the decorator instance using values from config, with defaults
api_retry_decorator: Callable = retry_api_call(
    max_retries=config.get("retry_max_retries", 3),
    initial_delay=config.get("retry_initial_delay", 5.0),
    backoff_factor=config.get("retry_backoff_factor", 2.0),
)

# --- Environment & Exchange Setup ---
print_neon_header()
exchange_id: str = config["exchange_id"].lower()
# Construct environment variable names dynamically based on exchange_id
api_key_env: str = f"{exchange_id.upper()}_API_KEY"
secret_key_env: str = f"{exchange_id.upper()}_SECRET_KEY"
passphrase_env: str = f"{exchange_id.upper()}_PASSPHRASE"  # Common for exchanges like KuCoin, Bybit V3

api_key: Optional[str] = os.getenv(api_key_env)
secret: Optional[str] = os.getenv(secret_key_env)
passphrase: Optional[str] = os.getenv(passphrase_env)  # Will be None if not set

if not api_key or not secret:
    log_error(
        f"CRITICAL: API Key (expected env var: '{api_key_env}') or Secret Key (expected env var: '{secret_key_env}') not found in environment variables or .env file."
    )
    log_error("Please ensure these variables are set correctly.")
    sys.exit(1)
if passphrase:
    log_info(f"Passphrase found (env var: '{passphrase_env}').")

log_info(f"Attempting to connect to exchange: {exchange_id}")
exchange: ccxt.Exchange
try:
    exchange_class: Type[ccxt.Exchange] = getattr(ccxt, exchange_id)

    # Dynamically guess market type based on symbol format for defaultType
    symbol_upper = config["symbol"].upper()
    # More robust check for perpetual swaps/futures
    is_perp = ":" in symbol_upper or "PERP" in symbol_upper or "SWAP" in symbol_upper or "-P" in symbol_upper
    market_type_guess = "swap" if is_perp else "spot"  # Default to spot if not clearly a derivative
    log_info(f"Guessed market type from symbol '{config['symbol']}': {market_type_guess} (used for `defaultType`)")

    # Base configuration for ccxt exchange instance
    exchange_config: Dict[str, Any] = {
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,  # Let ccxt handle basic rate limiting
        "options": {
            "defaultType": market_type_guess,  # Set guessed market type
            "adjustForTimeDifference": True,  # Important for accurate timestamps
        },
    }
    # Add passphrase to config if it exists
    if passphrase:
        exchange_config["password"] = passphrase

    exchange = exchange_class(exchange_config)

    # Robustly load markets *after* instantiation, using the retry decorator
    @api_retry_decorator
    def load_markets_robustly(exch: ccxt.Exchange, force_reload: bool = True) -> None:
        log_info(f"Loading markets (force_reload={force_reload})...")
        exch.load_markets(reload=force_reload)
        log_info(f"Markets loaded successfully. Found {len(exch.markets)} markets.")

    load_markets_robustly(exchange)  # Initial load

    log_info(f"Successfully connected to {exchange.name} ({exchange_id}).")
    log_info(f"Using Market Type: {exchange.options.get('defaultType')}")
    log_info(f"Exchange supports OCO orders: {exchange.has.get('oco', False)}")
    log_info(f"Exchange supports fetchPositions: {exchange.has.get('fetchPositions', False)}")
    log_info(f"Exchange supports createStopMarketOrder: {exchange.has.get('createStopMarketOrder', False)}")
    log_info(f"Exchange supports createStopLimitOrder: {exchange.has.get('createStopLimitOrder', False)}")


except AttributeError:
    log_error(f"CRITICAL: Exchange ID '{exchange_id}' is not supported by ccxt.")
    sys.exit(1)
except ccxt.AuthenticationError as e:
    log_error(
        f"CRITICAL: Authentication failed for {exchange_id}. Check API Key, Secret, and Passphrase (if required). Error: {e}"
    )
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    log_error(f"CRITICAL: Exchange {exchange_id} is currently unavailable. Error: {e}")
    sys.exit(1)
except Exception as e:  # Catch potential errors during load_markets too
    log_error(f"CRITICAL: Unexpected error during exchange setup or market loading: {e}", exc_info=True)
    sys.exit(1)

# --- Trading Parameters & Validation ---
symbol: str = config["symbol"].strip().upper()  # Ensure uppercase and no extra spaces
timeframe: str = config["timeframe"]
market_info: Optional[Dict[str, Any]] = None
price_precision_digits: int = DEFAULT_PRICE_PRECISION
amount_precision_digits: int = DEFAULT_AMOUNT_PRECISION
min_tick: float = 1 / (10**DEFAULT_PRICE_PRECISION)  # Smallest price increment
min_amount: Optional[float] = None  # Smallest order size in base currency
min_cost: Optional[float] = None  # Smallest order value in quote currency
base_currency: Optional[str] = None
quote_currency: Optional[str] = None
is_contract_market: bool = False  # Is it a swap/future?

try:
    # Validate symbol exists on the loaded markets
    if symbol not in exchange.markets:
        available_symbols = list(exchange.markets.keys())
        hint = f"Available examples: {available_symbols[:10]}" if available_symbols else "No markets loaded?"
        log_error(f"CRITICAL: Symbol '{symbol}' not found on {exchange.name}. {hint}")
        sys.exit(1)
    market_info = exchange.markets[symbol]

    # Validate timeframe is supported
    if timeframe not in exchange.timeframes:
        available_tfs = list(exchange.timeframes.keys())
        hint = f"Available timeframes: {available_tfs}" if available_tfs else "No timeframes listed?"
        log_error(f"CRITICAL: Timeframe '{timeframe}' not supported by {exchange.name}. {hint}")
        sys.exit(1)

    log_info(f"Validated Symbol: {symbol}, Timeframe: {timeframe}")

    # Extract Precision
    price_prec = market_info.get("precision", {}).get("price")
    amount_prec = market_info.get("precision", {}).get("amount")
    # Use ccxt's method to determine decimal places based on precision mode
    if price_prec is not None:
        # Use ROUND mode, some exchanges might require TRUNCATE (check precisionMode)
        price_precision_digits = int(
            exchange.decimal_to_precision(price_prec, ccxt.ROUND, counting_mode=exchange.precisionMode)
        )
    if amount_prec is not None:
        amount_precision_digits = int(
            exchange.decimal_to_precision(amount_prec, ccxt.ROUND, counting_mode=exchange.precisionMode)
        )

    # Extract Tick Size (smallest price change)
    if price_prec is not None:
        # Precision value itself often represents the tick size
        min_tick = float(price_prec)
    else:  # Fallback if price precision not provided
        min_tick = 1 / (10**price_precision_digits)
    # Ensure min_tick is positive
    if min_tick <= 0:
        min_tick = 1 / (10**price_precision_digits) if price_precision_digits >= 0 else 0.01  # Safe default
        log_warning(f"Could not determine valid min_tick from market info, using calculated value: {min_tick}")

    # Extract Limits
    min_amount = market_info.get("limits", {}).get("amount", {}).get("min")
    min_cost = market_info.get("limits", {}).get("cost", {}).get("min")

    # Extract Currencies and Market Type
    base_currency = market_info.get("base")
    quote_currency = market_info.get("quote")
    # Check various flags indicating a contract market
    is_contract_market = (
        market_info.get("swap", False)
        or market_info.get("future", False)
        or market_info.get("contract", False)
        or market_info.get("type") in ["swap", "future"]
    )

    if not base_currency or not quote_currency:
        log_warning(
            f"Could not reliably determine base/quote currency for {symbol}. Base: {base_currency}, Quote: {quote_currency}"
        )

    # Log extracted parameters
    log_info(f"Market Info | Base: {base_currency}, Quote: {quote_currency}, Contract Market: {is_contract_market}")
    log_info(
        f"Precision   | Price: {price_precision_digits} decimals (Tick Size: {min_tick:.{price_precision_digits + 2}f}), Amount: {amount_precision_digits} decimals"
    )  # Show tick size better
    if min_amount is not None:
        log_info(f"Limits      | Min Amount: {min_amount} {base_currency or ''}")
    if min_cost is not None:
        log_info(f"Limits      | Min Cost: {min_cost} {quote_currency or ''}")

except KeyError as e:
    log_error(
        f"CRITICAL: Error accessing market info for {symbol}. Market data might be incomplete. Key: {e}", exc_info=True
    )
    sys.exit(1)
except Exception as e:
    log_error(f"CRITICAL: Error processing symbol/timeframe/precision details: {e}", exc_info=True)
    sys.exit(1)

# --- Assign Config Variables (with type safety and defaults) ---
# Indicators
rsi_length: int = config["rsi_length"]
rsi_overbought: int = config["rsi_overbought"]
rsi_oversold: int = config["rsi_oversold"]
stoch_k: int = config["stoch_k"]
stoch_d: int = config["stoch_d"]
stoch_smooth_k: int = config["stoch_smooth_k"]
stoch_overbought: int = config["stoch_overbought"]
stoch_oversold: int = config["stoch_oversold"]

# Core Bot Settings
data_limit: int = config["data_limit"]
sleep_interval_seconds: int = config["sleep_interval_seconds"]
risk_percentage: float = float(config["risk_percentage"])  # Ensure float

# SL/TP & Trailing Stop Settings
enable_atr_sl_tp: bool = config["enable_atr_sl_tp"]
enable_trailing_stop: bool = config["enable_trailing_stop"]
needs_atr: bool = enable_atr_sl_tp or enable_trailing_stop  # Convenience flag
atr_length: int = config.get("atr_length", 14) if needs_atr else 0  # Default ATR length if needed

# Conditional SL/TP multipliers
if enable_atr_sl_tp:
    atr_sl_multiplier: float = float(config["atr_sl_multiplier"])
    atr_tp_multiplier: float = float(config["atr_tp_multiplier"])
    stop_loss_percentage: float = 0.0  # Not used
    take_profit_percentage: float = 0.0  # Not used
else:
    atr_sl_multiplier: float = 0.0  # Not used
    atr_tp_multiplier: float = 0.0  # Not used
    stop_loss_percentage: float = float(config["stop_loss_percentage"])
    take_profit_percentage: float = float(config["take_profit_percentage"])

# Conditional TSL multipliers
if enable_trailing_stop:
    trailing_stop_atr_multiplier: float = float(config["trailing_stop_atr_multiplier"])
    trailing_stop_activation_atr_multiplier: float = float(config["trailing_stop_activation_atr_multiplier"])
else:
    trailing_stop_atr_multiplier: float = 0.0  # Not used
    trailing_stop_activation_atr_multiplier: float = 0.0  # Not used

# Order Block Settings
ob_volume_threshold_multiplier: float = float(config["ob_volume_threshold_multiplier"])
ob_lookback: int = config["ob_lookback"]

# Volume Confirmation Settings
entry_volume_confirmation_enabled: bool = config["entry_volume_confirmation_enabled"]
if entry_volume_confirmation_enabled:
    entry_volume_ma_length: int = int(config.get("entry_volume_ma_length", 20))  # Default MA length
    entry_volume_multiplier: float = float(config.get("entry_volume_multiplier", 1.2))  # Default multiplier
else:
    entry_volume_ma_length: int = 0
    entry_volume_multiplier: float = 0.0

# --- Simulation Mode Check ---
SIMULATION_MODE: bool = config["simulation_mode"]
if SIMULATION_MODE:
    log_warning("###########################")
    log_warning("# SIMULATION MODE IS ACTIVE #")
    log_warning("# No real trades will be executed. #")
    log_warning("###########################")
else:
    log_warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    log_warning("!!! LIVE TRADING MODE IS ACTIVE !!!")
    log_warning("!!! REAL FUNDS WILL BE USED !!!")
    log_warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    try:
        # Check if running in an interactive terminal
        if sys.stdin.isatty():
            confirm = input(
                f"{NEON_RED}>>> Type 'LIVE' to confirm live trading, or press Enter/Ctrl+C to exit: {RESET}"
            )
            if confirm.strip().upper() != "LIVE":
                log_info("Live trading not confirmed. Exiting.")
                sys.exit(0)
            log_info("Live trading confirmed by user.")
        else:
            # Non-interactive environment (e.g., docker, systemd)
            log_warning("Non-interactive mode detected. Assuming confirmation for live trading.")
            log_warning("Pausing for 5 seconds before starting...")
            time.sleep(5)
    except EOFError:
        # Handle case where input is expected but not possible (e.g., piped input)
        log_error("Cannot get user confirmation in this environment. Exiting to prevent accidental live trading.")
        sys.exit(1)

# --- Position State ---
# Define the expected structure and default values for the position state
position_default_structure: Dict[str, Any] = {
    "status": None,  # 'long', 'short', or None
    "entry_price": None,  # float
    "quantity": None,  # float
    "order_id": None,  # str (ID of the initial entry market order)
    "stop_loss": None,  # float (Target SL price)
    "take_profit": None,  # float (Target TP price)
    "entry_time": None,  # pd.Timestamp (UTC)
    "sl_order_id": None,  # str (ID of the separate SL order)
    "tp_order_id": None,  # str (ID of the separate TP order)
    "oco_order_id": None,  # str (ID or reference for the OCO order pair)
    "highest_price_since_entry": None,  # float (For TSL calculation - long)
    "lowest_price_since_entry": None,  # float (For TSL calculation - short)
    "current_trailing_sl_price": None,  # float (The current active TSL price if TSL moved)
}
# Initialize the global position state variable
position: Dict[str, Any] = position_default_structure.copy()


# --- State Management Functions ---
def save_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Saves the current position state dictionary to a JSON file."""
    global position
    try:
        # Create a copy to avoid modifying the global state during serialization
        state_to_save = position.copy()
        # Convert Timestamp to ISO format string for JSON compatibility
        if isinstance(state_to_save.get("entry_time"), pd.Timestamp):
            state_to_save["entry_time"] = state_to_save["entry_time"].isoformat()

        with open(filename, "w") as f:
            json.dump(state_to_save, f, indent=4, default=str)  # Add default=str as fallback
        log_debug(f"Position state successfully saved to '{filename}'")
    except IOError as e:
        log_error(f"Error saving position state to '{filename}': File I/O error - {e}", exc_info=True)
    except TypeError as e:
        log_error(f"Error saving position state to '{filename}': Serialization error - {e}", exc_info=True)
    except Exception as e:
        log_error(f"Unexpected error saving position state: {e}", exc_info=True)


def load_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Loads and robustly validates position state from JSON file."""
    global position
    log_info(f"Attempting to load position state from '{filename}'...")
    # Start with a clean default state
    position = position_default_structure.copy()
    f_path = Path(filename)

    if not f_path.exists():
        log_info(f"No state file found at '{filename}'. Starting with a fresh state.")
        return

    try:
        with f_path.open("r") as f:
            loaded_state: Dict[str, Any] = json.load(f)

        log_info(f"State file '{filename}' found. Validating content...")
        issues_found: List[str] = []
        validated_state: Dict[str, Any] = position_default_structure.copy()  # Use a temporary dict for validation

        # Validate each key from the loaded state against the default structure
        for key, loaded_value in loaded_state.items():
            if key in validated_state:
                default_value = validated_state[key]  # Get default value for type comparison
                expected_type = type(default_value) if default_value is not None else None

                # Handle timestamp deserialization separately
                if key == "entry_time" and isinstance(loaded_value, str):
                    try:
                        ts = pd.Timestamp(loaded_value)
                        # Ensure timezone is UTC, localize if naive
                        validated_state[key] = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
                        continue  # Skip normal type check for timestamp
                    except ValueError:
                        issues_found.append(
                            f"Key '{key}': Invalid timestamp format ('{loaded_value}'). Resetting to None."
                        )
                        validated_state[key] = None
                        continue

                # Handle None values in default structure (allow loaded value if not None)
                if expected_type is None and loaded_value is not None:
                    # Attempt to infer type or just assign if structure allows None
                    validated_state[key] = loaded_value  # Be lenient if default is None
                    log_debug(f"Key '{key}' default is None, loaded value '{loaded_value}' assigned.")
                    continue

                # Check type consistency (allow int to be loaded into float fields)
                if loaded_value is not None and expected_type is not None:
                    allow_int_for_float = expected_type is float and isinstance(loaded_value, int)
                    if not isinstance(loaded_value, expected_type) and not allow_int_for_float:
                        issues_found.append(
                            f"Key '{key}': Type mismatch. Expected {expected_type.__name__}, got {type(loaded_value).__name__}. Resetting to default."
                        )
                        # Keep the default value in validated_state for this key
                        continue
                    # Assign validated value (convert int to float if needed)
                    validated_state[key] = float(loaded_value) if allow_int_for_float else loaded_value

            else:
                issues_found.append(f"Ignoring unknown key '{key}' found in state file.")

        # Check for essential data consistency if a position is supposedly active
        if validated_state.get("status") is not None:
            if (
                not isinstance(validated_state.get("entry_price"), (float, int))
                or not isinstance(validated_state.get("quantity"), (float, int))
                or not isinstance(validated_state.get("stop_loss"), (float, int))
                or not isinstance(validated_state.get("take_profit"), (float, int))
            ):
                issues_found.append(
                    f"Active position ('{validated_state.get('status')}') found, but essential data (entry_price, quantity, stop_loss, take_profit) is missing or invalid. Resetting status to None."
                )
                validated_state["status"] = None  # Invalidate the position if core data is bad

        # Report issues and update global state
        if issues_found:
            log_warning("Issues found during state file validation:\n - " + "\n - ".join(issues_found))
            log_warning("Using validated state with defaults applied for problematic keys.")

        position = validated_state  # Update the global state with the validated data

        if position["status"]:
            log_info("Position state loaded and validated successfully.")
            display_position_status(position, price_precision_digits, amount_precision_digits)
        else:
            log_info("Loaded state file, but no active position found or state was invalidated.")

    except json.JSONDecodeError as e:
        log_error(
            f"Error decoding JSON from state file '{filename}': {e}. Starting with a fresh state.", exc_info=False
        )
        position = position_default_structure.copy()
    except Exception as e:
        log_error(
            f"Unexpected error loading or validating state file '{filename}': {e}. Starting with a fresh state.",
            exc_info=True,
        )
        position = position_default_structure.copy()


# --- Data Fetching & Indicator Calculation ---
@api_retry_decorator
def fetch_ohlcv_data(exch: ccxt.Exchange, sym: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches, cleans, and returns OHLCV data as a Pandas DataFrame."""
    log_debug(f"Fetching {limit} OHLCV candles for {sym} ({tf})...")
    try:
        # Double-check symbol existence before fetching
        if sym not in exch.markets:
            log_error(f"Symbol '{sym}' not found in exchange markets during OHLCV fetch.")
            return None

        ohlcv = exch.fetch_ohlcv(sym, tf, limit=limit)
        if not ohlcv:  # Check if the list is empty
            log_warning(f"No OHLCV data returned from exchange for {sym} ({tf}).")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Convert timestamp to datetime objects (UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # Set timestamp as the index
        df = df.set_index("timestamp")
        # Convert OHLCV columns to numeric, coercing errors (like '--') to NaN
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Drop rows with any NaN in essential OHLCV columns
        df.dropna(subset=numeric_cols, inplace=True)

        if df.empty:
            log_warning(f"DataFrame became empty after cleaning OHLCV data for {sym} ({tf}).")
            return None

        log_debug(f"Successfully fetched and cleaned {len(df)} candles for {sym}.")
        return df

    # Handle specific CCXT errors that might occur during fetch
    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
        log_error(f"CCXT error fetching OHLCV for {sym} ({tf}): {e}")
        return None
    # Catch other unexpected errors (Network errors handled by decorator)
    except Exception as e:
        # Avoid logging retryable exceptions twice if they slip through
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error fetching OHLCV data for {sym}: {e}", exc_info=True)
        return None  # Return None on any error


def calculate_technical_indicators(
    df: Optional[pd.DataFrame],
    rsi_l: int,
    stoch_p: Dict[str, int],  # k, d, smooth_k
    atr_l: int = 14,
    vol_ma_l: int = 20,
    calc_atr: bool = False,
    calc_vol_ma: bool = False,
) -> Optional[pd.DataFrame]:
    """Calculates technical indicators using pandas_ta and appends them to the DataFrame."""
    if df is None or df.empty:
        log_warning("Cannot calculate indicators: Input DataFrame is None or empty.")
        return None

    # Check for required base columns
    required_cols = ["high", "low", "close"]
    if calc_vol_ma:
        required_cols.append("volume")
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        log_error(f"Cannot calculate indicators: Missing required columns: {missing}")
        return None

    log_debug(
        f"Calculating indicators (RSI, Stoch{' + ATR' if calc_atr else ''}{' + VolMA' if calc_vol_ma else ''})..."
    )
    try:
        # Work on a copy to avoid modifying the original DataFrame passed in
        df_ind = df.copy()
        indicator_columns_added = []  # Keep track of columns added

        # --- Always Calculate RSI and Stochastic ---
        # RSI
        rsi_col_name = f"RSI_{rsi_l}"
        df_ind.ta.rsi(length=rsi_l, append=True, col_names=(rsi_col_name,))
        indicator_columns_added.append(rsi_col_name)

        # Stochastic (using explicit column names for clarity)
        stoch_k_col = f"STOCHk_{stoch_p['k']}_{stoch_p['d']}_{stoch_p['smooth_k']}"
        stoch_d_col = f"STOCHd_{stoch_p['k']}_{stoch_p['d']}_{stoch_p['smooth_k']}"
        df_ind.ta.stoch(
            k=stoch_p["k"],
            d=stoch_p["d"],
            smooth_k=stoch_p["smooth_k"],
            append=True,
            col_names=(stoch_k_col, stoch_d_col),
        )
        indicator_columns_added.extend([stoch_k_col, stoch_d_col])

        # --- Conditionally Calculate ATR ---
        if calc_atr:
            atr_col_name = f"ATRr_{atr_l}"  # Use ATRr for range (true range)
            df_ind.ta.atr(length=atr_l, append=True, col_names=(atr_col_name,))
            indicator_columns_added.append(atr_col_name)

        # --- Conditionally Calculate Volume Moving Average ---
        if calc_vol_ma:
            if "volume" in df_ind.columns:
                vol_ma_col_name = f"VOL_MA_{vol_ma_l}"
                # Ensure min_periods is reasonable (e.g., at least half the window)
                min_p = max(1, vol_ma_l // 2)
                df_ind[vol_ma_col_name] = df_ind["volume"].rolling(window=vol_ma_l, min_periods=min_p).mean()
                indicator_columns_added.append(vol_ma_col_name)
            else:
                # This was checked earlier, but log warning if volume somehow disappeared
                log_warning("Volume column not found, cannot calculate Volume MA.")

        # Drop rows with NaN values *only* in the newly added indicator columns
        # This preserves earlier data rows that might be needed for lookbacks
        valid_indicator_cols = [col for col in indicator_columns_added if col in df_ind.columns]
        if valid_indicator_cols:
            initial_rows = len(df_ind)
            df_ind.dropna(subset=valid_indicator_cols, inplace=True)
            rows_after_na = len(df_ind)
            if rows_after_na < initial_rows:
                log_debug(f"Dropped {initial_rows - rows_after_na} rows due to NaN in indicators.")
        else:
            log_warning("No valid indicator columns were generated or found.")
            return None  # Should not happen if RSI/Stoch always calculated

        if df_ind.empty:
            log_warning("DataFrame became empty after dropping rows with NaN indicators.")
            return None

        log_debug(f"Indicator calculation complete. Resulting DataFrame rows: {len(df_ind)}")
        return df_ind

    except Exception as e:
        log_error(f"Error occurred during indicator calculation: {e}", exc_info=True)
        return None


# --- Order Block Identification ---
def identify_potential_order_block(
    df: pd.DataFrame, vol_thresh_mult: float, lookback: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Identifies the most recent potential bullish and bearish order blocks based on
    volume and price action relative to the preceding candle.
    Looks for an imbalance candle following the potential OB candle.
    """
    bullish_ob, bearish_ob = None, None
    required_cols = ["open", "high", "low", "close", "volume"]

    # Validate input DataFrame
    if df is None or df.empty or not all(c in df.columns for c in required_cols):
        log_debug("OB Detection: Input DataFrame invalid or missing required columns.")
        return None, None
    # Need at least lookback + 2 candles (1 for OB, 1 for imbalance, lookback for avg vol)
    if len(df) < lookback + 2:
        log_debug(f"OB Detection: Not enough data ({len(df)} rows) for lookback ({lookback}) + 2 candles.")
        return None, None

    try:
        # Calculate average volume over the lookback period (excluding the latest, possibly incomplete candle)
        df_completed = df.iloc[:-1]  # Use only completed candles for volume average
        avg_volume = 0.0
        min_vol_periods = max(1, lookback // 2)  # Require at least half the lookback period for avg calc
        if len(df_completed) >= min_vol_periods:
            rolling_volume = df_completed["volume"].rolling(window=lookback, min_periods=min_vol_periods).mean()
            if not rolling_volume.empty and pd.notna(rolling_volume.iloc[-1]):
                avg_volume = rolling_volume.iloc[-1]

        # Fallback if rolling mean failed (e.g., too few periods)
        if avg_volume <= 0 and len(df_completed) > 0:
            mean_vol = df_completed["volume"].mean()
            if pd.notna(mean_vol):
                avg_volume = mean_vol

        # Define the volume threshold for the imbalance candle
        volume_threshold = avg_volume * vol_thresh_mult if avg_volume > 0 else float("inf")  # Avoid threshold=0
        log_debug(
            f"OB Analysis | Lookback: {lookback}, Avg Volume (completed candles): {avg_volume:.2f}, Volume Threshold: {volume_threshold:.2f}"
        )

        # Search backwards for potential OBs (limit search depth for performance)
        # Search up to ~3 times the lookback period, but not exceeding available data
        search_depth = min(len(df) - 2, lookback * 3)  # -2 because we need OB candle (i-1) and Imbalance candle (i)
        search_start_index = len(df) - 2  # Start from the second to last candle (potential imbalance candle)
        search_end_index = max(0, search_start_index - search_depth)

        for i in range(search_start_index, search_end_index - 1, -1):  # Iterate backwards
            if i < 1:
                break  # Need index i and i-1

            try:
                imbalance_candle = df.iloc[i]
                ob_candidate_candle = df.iloc[i - 1]
            except IndexError:
                log_warning(f"OB Search: IndexError at index {i} or {i - 1}. Stopping search.")
                break  # Should not happen with loop bounds, but safety check

            # Skip if data is missing in either candle
            if imbalance_candle.isnull().any() or ob_candidate_candle.isnull().any():
                continue

            # --- Define OB Conditions ---
            # 1. Imbalance Candle Volume > Threshold
            is_high_volume_imbalance = imbalance_candle["volume"] > volume_threshold

            # 2. Imbalance Candle moves strongly *away* from OB candle, creating imbalance
            # Bullish Imbalance: Imbalance candle is green, closes above OB high
            is_bullish_imbalance = (
                imbalance_candle["close"] > imbalance_candle["open"]
                and imbalance_candle["close"] > ob_candidate_candle["high"]
            )
            # Bearish Imbalance: Imbalance candle is red, closes below OB low
            is_bearish_imbalance = (
                imbalance_candle["close"] < imbalance_candle["open"]
                and imbalance_candle["close"] < ob_candidate_candle["low"]
            )

            # 3. OB Candidate Candle Type (Opposite of imbalance direction)
            ob_is_bearish = ob_candidate_candle["close"] < ob_candidate_candle["open"]  # Down candle
            ob_is_bullish = ob_candidate_candle["close"] > ob_candidate_candle["open"]  # Up candle

            # 4. Imbalance Candle "Sweeps" Liquidity (optional but common pattern)
            # Bullish: Imbalance low goes below OB low
            imbalance_sweeps_ob_low = imbalance_candle["low"] < ob_candidate_candle["low"]
            # Bearish: Imbalance high goes above OB high
            imbalance_sweeps_ob_high = imbalance_candle["high"] > ob_candidate_candle["high"]

            # --- Check for Bullish OB ---
            # Needs: Bearish OB candle, followed by strong Bullish Imbalance candle (high vol, sweeps low)
            if (
                not bullish_ob
                and ob_is_bearish
                and is_bullish_imbalance
                and is_high_volume_imbalance
                and imbalance_sweeps_ob_low
            ):
                bullish_ob = {
                    "high": ob_candidate_candle["high"],
                    "low": ob_candidate_candle["low"],
                    "time": ob_candidate_candle.name,  # Timestamp of the OB candle
                    "type": "bullish",
                }
                log_debug(
                    f"Potential Bullish OB found at {ob_candidate_candle.name.strftime('%Y-%m-%d %H:%M')} (Low: {bullish_ob['low']:.{price_precision_digits}f}, High: {bullish_ob['high']:.{price_precision_digits}f})"
                )
                # If we found both, we can stop searching (most recent ones)
                if bearish_ob:
                    break

            # --- Check for Bearish OB ---
            # Needs: Bullish OB candle, followed by strong Bearish Imbalance candle (high vol, sweeps high)
            elif (
                not bearish_ob
                and ob_is_bullish
                and is_bearish_imbalance
                and is_high_volume_imbalance
                and imbalance_sweeps_ob_high
            ):
                bearish_ob = {
                    "high": ob_candidate_candle["high"],
                    "low": ob_candidate_candle["low"],
                    "time": ob_candidate_candle.name,  # Timestamp of the OB candle
                    "type": "bearish",
                }
                log_debug(
                    f"Potential Bearish OB found at {ob_candidate_candle.name.strftime('%Y-%m-%d %H:%M')} (Low: {bearish_ob['low']:.{price_precision_digits}f}, High: {bearish_ob['high']:.{price_precision_digits}f})"
                )
                # If we found both, we can stop searching
                if bullish_ob:
                    break

        # Return the found OBs (or None if not found)
        return bullish_ob, bearish_ob

    except Exception as e:
        log_error(f"Error during order block identification: {e}", exc_info=True)
        return None, None  # Return None on error


# --- Position Sizing ---
@api_retry_decorator  # Apply retry logic to balance fetching within this function
def calculate_position_size(exch: ccxt.Exchange, sym: str, entry: float, sl: float, risk_pct: float) -> Optional[float]:
    """
    Calculates the position size in the base currency based on risk percentage,
    available balance, entry price, and stop-loss price.
    Considers exchange limits (min_amount, min_cost).
    """
    # Use globals set earlier during market validation
    global \
        base_currency, \
        quote_currency, \
        is_contract_market, \
        min_amount, \
        min_cost, \
        min_tick, \
        price_precision_digits, \
        amount_precision_digits

    # --- Input Validation ---
    if not (0 < risk_pct < 1):
        log_error(f"Invalid risk percentage: {risk_pct}. Must be between 0 and 1 (exclusive).")
        return None
    if entry <= 0 or sl <= 0:
        log_error(f"Invalid entry or SL price for size calculation: Entry={entry}, SL={sl}. Must be positive.")
        return None
    price_difference = abs(entry - sl)
    # Ensure SL is not too close to entry (avoid division by zero or tiny number)
    # Use min_tick as a minimum distance threshold
    if price_difference < min_tick:
        log_error(
            f"Stop loss ({sl:.{price_precision_digits}f}) is too close to entry price ({entry:.{price_precision_digits}f}). Minimum distance required: {min_tick:.{price_precision_digits}f}."
        )
        return None

    log_debug(
        f"Calculating position size for {sym}: Entry={entry:.{price_precision_digits}f}, SL={sl:.{price_precision_digits}f}, Risk={risk_pct * 100:.2f}%"
    )

    try:
        # --- Fetch Available Balance ---
        log_debug("Fetching account balance...")
        balance_info = exch.fetch_balance()
        available_balance_quote = 0.0
        balance_source_info = "N/A"  # For logging where the balance came from

        # Try common locations for free quote currency balance
        if quote_currency:
            if quote_currency in balance_info and isinstance(balance_info[quote_currency], dict):
                available_balance_quote = float(balance_info[quote_currency].get("free", 0.0))
                balance_source_info = f"['{quote_currency}']['free']"
                # Fallback to 'total' if 'free' is zero but 'total' exists and is positive (might indicate isolated margin?)
                if available_balance_quote == 0.0 and balance_info[quote_currency].get("total", 0.0) > 0:
                    available_balance_quote = float(balance_info[quote_currency]["total"])
                    balance_source_info = f"['{quote_currency}']['total'] (used as 'free' was zero)"
                    log_warning(
                        f"Using 'total' {quote_currency} balance ({available_balance_quote}), as 'free' was zero."
                    )
            elif (
                "free" in balance_info
                and isinstance(balance_info["free"], dict)
                and quote_currency in balance_info["free"]
            ):
                # Some exchanges structure it as balance['free']['USDT']
                available_balance_quote = float(balance_info["free"].get(quote_currency, 0.0))
                balance_source_info = f"['free']['{quote_currency}']"
            elif "info" in balance_info and isinstance(balance_info["info"], dict):
                # Fallback: Check raw 'info' field for common patterns (exchange-specific)
                # Example for Bybit Unified Margin Account (check 'availableToWithdraw') - NEEDS VERIFICATION
                if (
                    "result" in balance_info["info"]
                    and "list" in balance_info["info"]["result"]
                    and balance_info["info"]["result"]["list"]
                ):
                    for asset_info in balance_info["info"]["result"]["list"]:
                        if asset_info.get("coin") == quote_currency:
                            available_balance_quote = float(
                                asset_info.get("availableToWithdraw", 0.0)
                            )  # Or 'walletBalance'? Check Bybit docs
                            balance_source_info = (
                                f"['info']['result']['list'][coin={quote_currency}]['availableToWithdraw'] (Bybit UMA?)"
                            )
                            break
                # Add more exchange-specific checks here if needed based on `balance_info['info']` structure

        if available_balance_quote <= 0:
            log_error(
                f"Insufficient available balance ({available_balance_quote:.{price_precision_digits}f}) in quote currency ({quote_currency}). Source checked: {balance_source_info}."
            )
            log_debug(
                f"Full balance details: {balance_info.get(quote_currency, balance_info.get('info', 'Balance info structure not recognized'))}"
            )
            return None
        log_info(
            f"Available balance ({quote_currency}): {available_balance_quote:.{price_precision_digits}f} (Source: {balance_source_info})"
        )

        # --- Calculate Risk Amount and Quantity ---
        risk_amount_quote = available_balance_quote * risk_pct
        # Quantity = (Amount to Risk) / (Risk per unit of Base Currency)
        quantity_base = risk_amount_quote / price_difference

        # --- Format Quantity to Exchange Precision ---
        try:
            quantity_adjusted = float(exch.amount_to_precision(sym, quantity_base))
        except Exception as fmt_e:
            log_warning(
                f"ccxt amount_to_precision failed ('{fmt_e}'). Using manual rounding to {amount_precision_digits} decimals."
            )
            quantity_adjusted = round(quantity_base, amount_precision_digits)

        if quantity_adjusted <= 0:
            log_error(
                f"Calculated quantity ({quantity_base:.{amount_precision_digits + 4}f}) resulted in zero or negative ({quantity_adjusted}) after adjusting for precision."
            )
            return None

        # --- Check Exchange Limits ---
        # 1. Minimum Amount (Base Currency)
        if min_amount is not None and quantity_adjusted < min_amount:
            log_error(
                f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} {base_currency} is below the minimum required amount of {min_amount} {base_currency}."
            )
            return None

        # 2. Minimum Cost (Quote Currency)
        estimated_cost_quote = quantity_adjusted * entry
        if min_cost is not None and estimated_cost_quote < min_cost:
            log_error(
                f"Estimated order cost {estimated_cost_quote:.{price_precision_digits}f} {quote_currency} is below the minimum required cost of {min_cost} {quote_currency}."
            )
            # Optionally suggest the required quantity:
            required_qty_for_min_cost = min_cost / entry
            log_info(
                f"  (Minimum quantity required to meet min cost at current price: ~{required_qty_for_min_cost:.{amount_precision_digits}f} {base_currency})"
            )
            return None

        # 3. Check vs Available Balance (with a small buffer for fees/slippage)
        cost_buffer_factor = 0.995  # Use 99.5% of balance to allow for fees etc.
        if estimated_cost_quote > available_balance_quote * cost_buffer_factor:
            log_error(
                f"Estimated order cost ({estimated_cost_quote:.{price_precision_digits}f} {quote_currency}) exceeds {cost_buffer_factor * 100:.1f}% of available balance ({available_balance_quote:.{price_precision_digits}f})."
            )
            return None

        # --- Success ---
        log_info(
            f"{NEON_GREEN}Calculated position size: {quantity_adjusted:.{amount_precision_digits}f} {base_currency}{RESET} (Risking approx. {risk_amount_quote:.{price_precision_digits}f} {quote_currency})"
        )
        return quantity_adjusted

    # Handle potential errors during balance fetch or calculations
    except (ccxt.AuthenticationError, ccxt.ExchangeError) as e:
        # These might be caught by the decorator, but handle them here too for clarity
        log_error(f"Error calculating position size due to exchange communication issue: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        if not isinstance(e, RETRYABLE_EXCEPTIONS):  # Avoid double logging
            log_error(f"Unexpected error during position size calculation: {e}", exc_info=True)
        return None


# --- Order Placement ---
@api_retry_decorator
def cancel_order_with_retry(exch: ccxt.Exchange, order_id: str, sym: str) -> bool:
    """
    Attempts to cancel a specific order by ID, handling simulation mode
    and 'OrderNotFound' errors gracefully. Returns True if cancellation
    succeeded or the order was already not found, False otherwise.
    """
    if not order_id or not isinstance(order_id, str):
        log_debug(f"Invalid order ID provided for cancellation: '{order_id}'. Skipping.")
        return False  # Invalid ID cannot be cancelled

    log_info(f"Attempting to cancel order ID {order_id} for symbol {sym}...")

    if SIMULATION_MODE:
        log_warning(f"SIMULATION: Skipped cancelling order {order_id}.")
        return True  # Assume cancellation works in simulation

    try:
        log_warning(f"!!! LIVE MODE: Sending cancel request for order {order_id}.")
        exch.cancel_order(order_id, sym)
        log_info(f"Cancel request for order {order_id} sent successfully.")
        # Note: Confirmation might take time, we assume success if no exception
        return True
    except ccxt.OrderNotFound:
        # This is not an error in our context; the order is already gone.
        log_info(f"Order {order_id} not found on the exchange. Assumed already closed or cancelled.")
        return True
    except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
        # Log specific ccxt errors but return False as cancellation failed
        log_error(f"Failed to cancel order {order_id}: {type(e).__name__} - {e}")
        return False
    # Let Network/RateLimit errors be handled by the retry decorator
    # Catch unexpected errors
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error cancelling order {order_id}: {e}", exc_info=True)
        return False  # Return False on unexpected errors


@api_retry_decorator
def place_market_order(
    exch: ccxt.Exchange,
    sym: str,
    side: str,  # 'buy' or 'sell'
    amount: float,
    reduce_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Places a market order with retry logic, simulation handling, and basic error checking.
    Returns the order dictionary from ccxt if successful, None otherwise.
    """
    global base_currency, quote_currency, amount_precision_digits, price_precision_digits, is_contract_market

    # --- Input Validation ---
    if side not in ["buy", "sell"]:
        log_error(f"Invalid side '{side}' for market order. Must be 'buy' or 'sell'.")
        return None
    if not isinstance(amount, (float, int)) or amount <= 0:
        log_error(f"Invalid amount '{amount}' for market order. Must be a positive number.")
        return None

    # --- Format Amount ---
    try:
        amount_formatted = float(exch.amount_to_precision(sym, amount))
        if amount_formatted <= 0:
            log_error(f"Amount {amount} formatted to {amount_formatted}, which is <= 0. Cannot place order.")
            return None
    except Exception as fmt_e:
        log_error(f"Failed to format order amount {amount} using exchange precision rules: {fmt_e}")
        return None

    # --- Prepare Order Parameters ---
    params = {}
    reduce_only_applied = False
    if reduce_only:
        if is_contract_market:
            # Check if the exchange explicitly supports 'reduceOnly' in params
            # Note: Some exchanges might need it at the top level, some in 'params'
            # This is a common way, but might need adjustment per exchange.
            if exch.id in ["bybit", "binance", "kucoinfutures", "okx", "gateio"]:  # Add others known to use params
                params["reduceOnly"] = True
                reduce_only_applied = True
            else:  # Assume top-level argument might be needed (less common now)
                log_warning(
                    f"Uncertain how {exch.id} handles 'reduceOnly'. Assuming it's a top-level argument or handled implicitly. Check ccxt/exchange docs if closure fails."
                )
                # We'll pass it to create_market_order anyway, ccxt might handle it.
                reduce_only_applied = True  # Mark as intended
        else:
            log_warning("ReduceOnly flag ignored: Market is not identified as a contract market.")

    log_info(f"Attempting MARKET {side.upper()} order:")
    log_info(f"  Symbol: {sym}")
    log_info(f"  Amount: {amount_formatted:.{amount_precision_digits}f} {base_currency or 'Base'}")
    if reduce_only_applied:
        log_info("  Type: Reduce-Only")

    # --- Execute Order (Simulation or Live) ---
    order_result: Optional[Dict[str, Any]] = None
    if SIMULATION_MODE:
        log_warning("!!! SIMULATION: Market order placement skipped.")
        # Create a realistic-looking simulated order response
        sim_timestamp_ms = int(time.time() * 1000)
        sim_order_id = f"{SIMULATION_ORDER_PREFIX}market_{side}_{sim_timestamp_ms}"
        sim_price = 0.0
        try:  # Attempt to fetch current price for simulation realism

            @api_retry_decorator  # Decorate the inner fetch call too
            def fetch_sim_ticker(e: ccxt.Exchange, s: str) -> Dict:
                return e.fetch_ticker(s)

            ticker = fetch_sim_ticker(exch, sym)
            sim_price = ticker.get("last", ticker.get("close", 0.0))  # Use last or close price
            if sim_price <= 0:
                sim_price = ticker.get("bid", 0.0) if side == "sell" else ticker.get("ask", 0.0)  # Fallback to bid/ask
        except Exception as e:
            log_warning(f"Simulation price fetch failed: {e}. Using 0.0 for simulated price.")

        sim_cost = amount_formatted * sim_price if sim_price > 0 else 0.0
        order_result = {
            "id": sim_order_id,
            "clientOrderId": sim_order_id,
            "timestamp": sim_timestamp_ms,
            "datetime": pd.Timestamp.now(tz="UTC").isoformat(),
            "status": "closed",  # Assume market orders fill instantly in simulation
            "symbol": sym,
            "type": "market",
            "timeInForce": "IOC",  # ImmediateOrCancel is typical for market
            "side": side,
            "price": sim_price,  # The simulated execution price
            "average": sim_price,
            "amount": amount_formatted,  # Original requested amount
            "filled": amount_formatted,  # Assume fully filled
            "remaining": 0.0,
            "cost": sim_cost,  # Estimated cost
            "fee": None,  # Fee simulation is complex, omit for now
            "reduceOnly": reduce_only_applied,
            "info": {"simulated": True, "simulated_price": sim_price},  # Add simulation flag
        }
        log_info(f"Simulated market order result generated (ID: {sim_order_id}).")

    else:  # Live Trading Mode
        try:
            log_warning(f"!!! LIVE MODE: Placing real market order{' (Reduce-Only)' if reduce_only_applied else ''}.")
            # Pass reduce_only as a direct argument if applicable, and also in params
            # ccxt aims to abstract this, but being explicit can help
            order_result = exch.create_market_order(
                symbol=sym,
                side=side,
                amount=amount_formatted,
                params=params,  # Pass additional params like reduceOnly here
            )
            log_info("Market order request sent to exchange.")
            # Optional: Add a small delay to allow the order to potentially fill before next steps
            time.sleep(1.5)  # Adjust as needed, helps avoid immediate status checks returning 'open'

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
            # Handle specific, common CCXT order placement errors
            log_error(f"Failed to place market order: {type(e).__name__} - {e}")
            return None  # Explicitly return None on these failures
        # Let Network/RateLimit errors be handled by the retry decorator
        except Exception as e:
            if not isinstance(e, RETRYABLE_EXCEPTIONS):
                log_error(f"Unexpected error placing market order: {e}", exc_info=True)
            return None  # Return None on unexpected errors

    # --- Process Result ---
    if order_result:
        # Log details from the returned order dictionary
        o_id = order_result.get("id", "N/A")
        o_status = order_result.get("status", "N/A")
        o_avg_price = order_result.get("average", order_result.get("price"))  # Use average if available
        o_filled = order_result.get("filled")
        o_cost = order_result.get("cost")
        o_reduce = order_result.get("reduceOnly", params.get("reduceOnly", "N/A"))  # Check result or original param

        # Format numbers safely for logging
        p_str = f"{o_avg_price:.{price_precision_digits}f}" if isinstance(o_avg_price, (float, int)) else "N/A"
        f_str = f"{o_filled:.{amount_precision_digits}f}" if isinstance(o_filled, (float, int)) else "N/A"
        c_str = f"{o_cost:.{price_precision_digits}f}" if isinstance(o_cost, (float, int)) else "N/A"

        log_info(
            f"Order Result | ID: {o_id}, Status: {o_status}, Avg Fill Px: {p_str}, Filled Qty: {f_str} {base_currency or ''}, Cost: {c_str} {quote_currency or ''}, ReduceOnly: {o_reduce}"
        )

        # Basic check: If live order is still 'open', warn user. Market orders should ideally be 'closed' quickly.
        if not SIMULATION_MODE and o_status == "open":
            log_warning(
                f"Market order {o_id} status is 'open'. It might not have filled immediately. Monitoring required."
            )
        elif o_status == "rejected" or o_status == "expired":
            log_error(f"Market order {o_id} failed with status: {o_status}")
            return None  # Treat failure statuses as None result

        return order_result  # Return the full order dict
    else:
        log_error("Market order placement attempt did not return an order object.")
        return None


@api_retry_decorator
def place_sl_tp_orders(
    exch: ccxt.Exchange,
    sym: str,
    pos_side: str,  # 'long' or 'short' (side of the position being protected)
    qty: float,  # Quantity to close (should match position size)
    sl_pr: float,  # Stop Loss trigger price
    tp_pr: float,  # Take Profit limit price
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Places protection orders (Stop Loss and Take Profit) for an existing position.
    Attempts to use OCO (One-Cancels-the-Other) if supported by the exchange.
    Falls back to placing separate SL (Stop Market preferred, Stop Limit fallback)
    and TP (Limit) orders if OCO fails or is not supported.

    Handles simulation mode.
    Returns a tuple: (sl_order_id, tp_order_id, oco_order_id).
    If OCO is used, oco_order_id will be populated, sl/tp likely None.
    If separate orders are used, sl_order_id and/or tp_order_id will be populated.
    Returns (None, None, None) on complete failure.
    """
    global base_currency, min_tick, is_contract_market, amount_precision_digits, price_precision_digits

    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    oco_ref_id: Optional[str] = None  # Use a distinct name for OCO reference

    # --- Input Validation ---
    if pos_side not in ["long", "short"]:
        log_error(f"Invalid position side '{pos_side}' for SL/TP placement.")
        return None, None, None
    if not (isinstance(qty, (float, int)) and qty > 0):
        log_error(f"Invalid quantity '{qty}' for SL/TP placement.")
        return None, None, None
    if not (isinstance(sl_pr, (float, int)) and sl_pr > 0 and isinstance(tp_pr, (float, int)) and tp_pr > 0):
        log_error(f"Invalid SL ({sl_pr}) or TP ({tp_pr}) price for placement.")
        return None, None, None

    # Logical validation: SL should be below entry for long, above for short. TP opposite.
    # Note: This assumes entry price is between SL and TP. We check against the prices themselves.
    if pos_side == "long" and sl_pr >= tp_pr:
        log_error(f"Invalid SL/TP logic for LONG position: SL ({sl_pr}) must be < TP ({tp_pr}).")
        return None, None, None
    if pos_side == "short" and sl_pr <= tp_pr:
        log_error(f"Invalid SL/TP logic for SHORT position: SL ({sl_pr}) must be > TP ({tp_pr}).")
        return None, None, None

    # Determine the side for the closing orders
    close_side: str = "sell" if pos_side == "long" else "buy"

    # --- Format Inputs ---
    try:
        qty_formatted = float(exch.amount_to_precision(sym, qty))
        sl_price_formatted = float(exch.price_to_precision(sym, sl_pr))
        tp_price_formatted = float(exch.price_to_precision(sym, tp_pr))
        if qty_formatted <= 0:
            raise ValueError("Quantity formatted to zero or less.")
        # Re-check logic after formatting, as precision might change order
        if (pos_side == "long" and sl_price_formatted >= tp_price_formatted) or (
            pos_side == "short" and sl_price_formatted <= tp_price_formatted
        ):
            raise ValueError(f"SL/TP logic invalid after formatting: SL={sl_price_formatted}, TP={tp_price_formatted}")

    except ValueError as ve:
        log_error(f"SL/TP input validation failed after formatting: {ve}")
        return None, None, None
    except Exception as fmt_e:
        log_error(f"Failed to format SL/TP quantity or prices using exchange rules: {fmt_e}")
        return None, None, None

    # --- Prepare Common Parameters ---
    # ReduceOnly is crucial for contract markets to ensure these orders only close the position
    params = {"reduceOnly": True} if is_contract_market else {}
    if params:
        log_debug("Applying reduceOnly=True to protection orders.")
    else:
        log_debug("Not applying reduceOnly (either spot market or assuming implicit).")

    # --- Determine SL Order Type (Stop Market preferred) ---
    sl_order_type: str = "stopMarket"  # Default preference
    sl_limit_price: Optional[float] = None  # Only used for stopLimit fallback

    # Check exchange capabilities for stop orders
    has_stop_market = exch.has.get("createStopMarketOrder", False)
    has_stop_limit = exch.has.get("createStopLimitOrder", False)
    # Some exchanges might use createOrder with type 'stop' or 'stop_market'
    has_generic_stop = exch.has.get("createStopOrder", False)

    if has_stop_market or has_generic_stop:
        # Prefer Stop Market if available or if generic stop exists (assume it can act as market)
        sl_order_type = "stopMarket" if has_stop_market else "stop"  # Use specific if known, else generic 'stop'
        log_debug(f"Selected SL order type: {sl_order_type} (preferred). Trigger: {sl_price_formatted}")
    elif has_stop_limit:
        # Fallback to Stop Limit if Stop Market is unavailable
        sl_order_type = "stopLimit"
        # Calculate a sensible limit price slightly beyond the stop trigger to increase fill chance
        limit_offset = min_tick * 10  # Example: 10 ticks away (adjust multiplier as needed)
        limit_price_raw = (
            sl_price_formatted - limit_offset if close_side == "sell" else sl_price_formatted + limit_offset
        )
        # Ensure limit price stays on the correct side of the trigger
        if (close_side == "sell" and limit_price_raw >= sl_price_formatted) or (
            close_side == "buy" and limit_price_raw <= sl_price_formatted
        ):
            limit_price_raw = sl_price_formatted - min_tick if close_side == "sell" else sl_price_formatted + min_tick
            log_warning("Calculated SL limit offset crossed trigger price. Using 1 tick offset.")

        try:  # Format the calculated limit price
            sl_limit_price = float(exch.price_to_precision(sym, limit_price_raw))
        except Exception as fmt_e:
            log_error(f"Failed to format SL limit price {limit_price_raw}: {fmt_e}. Cannot place Stop Limit SL.")
            return None, None, None
        log_warning(
            f"Using fallback SL order type: {sl_order_type}. Trigger: {sl_price_formatted}, Limit: {sl_limit_price}. Fill depends on market liquidity."
        )
    else:
        # No supported stop order type found
        log_error(
            f"Exchange {exch.id} does not appear to support Stop Market or Stop Limit orders via ccxt. Cannot place Stop Loss."
        )
        return None, None, None

    # --- Attempt OCO Placement (if supported) ---
    oco_supported = exch.has.get("oco", False)
    oco_attempted = False
    oco_succeeded = False

    if oco_supported:
        oco_attempted = True
        log_info(f"Attempting OCO {close_side.upper()} order placement:")
        log_info(
            f"  Qty: {qty_formatted}, TP Limit: {tp_price_formatted}, SL Trigger: {sl_price_formatted} ({sl_order_type}{f', SL Limit: {sl_limit_price}' if sl_limit_price else ''})"
        )
        if params:
            log_info(f"  Params: {params}")

        try:
            # --- !!! EXCHANGE-SPECIFIC OCO PARAMETERS ARE CRITICAL HERE !!! ---
            # The structure for OCO orders varies significantly between exchanges.
            # You MUST consult the ccxt documentation/examples for your specific exchange
            # or the exchange's API documentation for the correct parameters.
            # Common parameters might include: 'stopPrice', 'stopLimitPrice', 'listClientOrderId', etc.
            # The 'type' might also need to be specific (e.g., 'limit' with OCO params, or a dedicated 'oco' type).

            # --- Example Generic Structure (HIGHLY LIKELY TO NEED MODIFICATION) ---
            oco_params = params.copy()  # Start with reduceOnly if applicable
            # These are common names, adjust based on your exchange:
            oco_params["stopPrice"] = sl_price_formatted  # SL trigger price
            if sl_order_type == "stopLimit" and sl_limit_price is not None:
                oco_params["stopLimitPrice"] = sl_limit_price  # SL limit price (for STOP_LOSS_LIMIT type)

            # The `type` for create_order for OCO also varies. Common guesses:
            # - 'limit' (where TP price is the main price argument, SL info in params)
            # - Some exchanges might have a dedicated 'oco' type.
            # CONSULT YOUR EXCHANGE'S CCXT IMPLEMENTATION / API DOCS!
            oco_order_type_guess = "limit"  # Default guess - VERIFY THIS FOR YOUR EXCHANGE

            # --- Simulation Handling for OCO ---
            if SIMULATION_MODE:
                log_warning("!!! SIMULATION: OCO order placement skipped.")
                sim_oco_id = f"{SIMULATION_ORDER_PREFIX}oco_{close_side}_{int(time.time())}"
                oco_ref_id = sim_oco_id  # Use the simulated ID as the reference
                oco_succeeded = True
            else:
                # --- !!! Replace below with the CORRECT ccxt call for YOUR exchange !!! ---
                log_warning(f"!!! LIVE MODE: Attempting OCO order using generic structure for {exch.id}.")
                log_warning(
                    f"!!! VERIFY this structure works for {exch.id} before relying on it! Type='{oco_order_type_guess}', Params={oco_params}"
                )

                # Example Placeholder - THIS WILL LIKELY FAIL OR NEED ADJUSTMENT
                # order_result = exch.create_order(
                #     symbol=sym,
                #     type=oco_order_type_guess, # Verify type!
                #     side=close_side,
                #     amount=qty_formatted,
                #     price=tp_price_formatted, # TP price usually the main price argument
                #     params=oco_params        # SL info and reduceOnly in params
                # )

                # --- SAFER APPROACH: Raise NotSupported until specific logic is implemented ---
                # This forces fallback to separate orders, which is safer than using wrong OCO params.
                # Remove this raise once you have verified and implemented the correct OCO structure.
                raise ccxt.NotSupported(
                    f"Generic OCO structure used for {exch.id}. Implement exchange-specific OCO logic in place_sl_tp_orders() or rely on separate orders."
                )

                # --- Process OCO Result (if the raise above is removed) ---
                # if order_result and order_result.get('id'):
                #     # OCO orders might return one ID, or a list in 'info', or require fetching based on a clientOrderID.
                #     # This part is highly exchange-dependent.
                #     oco_ref_id = order_result.get('id') # Simplistic assumption
                #     if not oco_ref_id:
                #          # Try common 'info' locations
                #          if 'info' in order_result and isinstance(order_result['info'], dict):
                #               oco_ref_id = order_result['info'].get('listClientOrderId', order_result['info'].get('orderListId')) # Binance / Bybit examples
                #
                #     if oco_ref_id:
                #         log_info(f"OCO order request processed successfully. Reference ID: {oco_ref_id}, Status: {order_result.get('status','?')}")
                #         oco_succeeded = True
                #     else:
                #         log_error(f"OCO order placed but failed to retrieve a reference ID. Order details: {order_result}")
                #         # Consider attempting cancellation if possible, though difficult without ID
                # else:
                #     log_error(f"OCO order placement failed or did not return expected result: {order_result}")

        except ccxt.NotSupported as e:
            log_warning(
                f"OCO not supported by {exch.id} or structure used is incorrect: {e}. Falling back to separate SL/TP orders."
            )
        except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
            log_error(f"OCO order placement failed: {type(e).__name__} - {e}. Falling back to separate SL/TP orders.")
        # Let network errors be handled by decorator
        except Exception as e:
            if not isinstance(e, RETRYABLE_EXCEPTIONS):
                log_error(f"Unexpected error during OCO placement: {e}. Falling back.", exc_info=True)
            # Fallback will occur naturally after this block if oco_succeeded is False

    # --- Return if OCO Succeeded ---
    if oco_succeeded:
        return None, None, oco_ref_id  # Return only the OCO reference ID

    # --- Fallback: Place Separate Orders ---
    if oco_attempted:  # Only log fallback message if OCO was tried
        log_info("Placing separate SL and TP orders as OCO fallback.")
    else:
        log_info("Placing separate SL and TP orders (OCO not supported or not attempted).")

    sl_placed_ok = False
    tp_placed_ok = False

    # 1. Place Stop Loss Order
    try:
        log_info(f"Attempting separate SL ({sl_order_type}):")
        log_info(f"  Side: {close_side.upper()}, Qty: {qty_formatted}")
        log_info(f"  Trigger Price: {sl_price_formatted}")
        if sl_limit_price:
            log_info(f"  Limit Price: {sl_limit_price}")
        if params:
            log_info(f"  Params: {params}")

        sl_params_separate = params.copy()
        sl_params_separate["stopPrice"] = sl_price_formatted  # Ensure stopPrice is in params

        if SIMULATION_MODE:
            sl_order_id = f"{SIMULATION_ORDER_PREFIX}sl_{close_side}_{int(time.time())}"
            log_warning(f"!!! SIMULATION: Separate SL order skipped. Assigned mock ID: {sl_order_id}")
            sl_placed_ok = True
        else:
            log_warning(f"!!! LIVE MODE: Placing separate {sl_order_type} SL order.")
            sl_order = exch.create_order(
                symbol=sym,
                type=sl_order_type,
                side=close_side,
                amount=qty_formatted,
                price=sl_limit_price,  # Price arg is used for limit price in stopLimit, ignored in stopMarket
                params=sl_params_separate,
            )
            if sl_order and sl_order.get("id"):
                sl_order_id = sl_order["id"]
                sl_placed_ok = True
                log_info(f"Separate SL ({sl_order_type}) order placed successfully. ID: {sl_order_id}")
                time.sleep(0.5)  # Small delay between orders
            else:
                log_error(
                    f"Separate SL ({sl_order_type}) placement failed or did not return an ID. Response: {sl_order}"
                )

    except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
        log_error(f"Error placing separate SL ({sl_order_type}) order: {type(e).__name__} - {e}")
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error placing separate SL ({sl_order_type}): {e}", exc_info=True)

    # 2. Place Take Profit Order (Limit Order)
    try:
        log_info("Attempting separate TP (Limit):")
        log_info(f"  Side: {close_side.upper()}, Qty: {qty_formatted}")
        log_info(f"  Limit Price: {tp_price_formatted}")
        if params:
            log_info(f"  Params: {params}")

        tp_params_separate = params.copy()  # Use reduceOnly if applicable

        if SIMULATION_MODE:
            tp_order_id = f"{SIMULATION_ORDER_PREFIX}tp_{close_side}_{int(time.time())}"
            log_warning(f"!!! SIMULATION: Separate TP order skipped. Assigned mock ID: {tp_order_id}")
            tp_placed_ok = True
        else:
            log_warning("!!! LIVE MODE: Placing separate limit TP order.")
            # Use create_limit_order for clarity
            tp_order = exch.create_limit_order(
                symbol=sym, side=close_side, amount=qty_formatted, price=tp_price_formatted, params=tp_params_separate
            )
            if tp_order and tp_order.get("id"):
                tp_order_id = tp_order["id"]
                tp_placed_ok = True
                log_info(f"Separate TP (Limit) order placed successfully. ID: {tp_order_id}")
                time.sleep(0.5)  # Small delay
            else:
                log_error(f"Separate TP (Limit) placement failed or did not return an ID. Response: {tp_order}")

    except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
        log_error(f"Error placing separate TP (Limit) order: {type(e).__name__} - {e}")
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
            log_error(f"Unexpected error placing separate TP (Limit): {e}", exc_info=True)

    # --- Final Outcome Assessment ---
    if not sl_placed_ok and not tp_placed_ok:
        log_error("Both separate SL and TP order placements failed.")
        return None, None, None  # Complete failure
    elif not sl_placed_ok:
        log_warning("Separate TP order placed, but SL order FAILED. Position is only partially protected (TP only).")
        # Attempt to cancel the successful TP order to avoid inconsistent state? Risky.
        # cancel_order_with_retry(exch, tp_order_id, sym) # Consider this carefully
    elif not tp_placed_ok:
        log_warning("Separate SL order placed, but TP order FAILED. Position is only partially protected (SL only).")
        # Attempt to cancel the successful SL order? Risky.
        # cancel_order_with_retry(exch, sl_order_id, sym) # Consider this carefully

    return sl_order_id, tp_order_id, None  # Return the IDs of the separately placed orders


# --- Position & Order Check ---
@api_retry_decorator  # Apply retry logic to internal fetch_open_orders / fetch_positions calls
def check_position_and_orders(exch: ccxt.Exchange, sym: str) -> bool:
    """
    Checks the consistency of the local position state against open orders
    and potentially the actual position on the exchange (for contract markets).
    Resets the local state if a closure is detected (e.g., SL/TP hit, OCO filled).
    Attempts to cancel any lingering counterpart order if one leg of SL/TP filled.

    Returns:
        bool: True if the local position state was reset due to detected closure, False otherwise.
    """
    global position, is_contract_market, min_amount  # Use globals

    # If no local position is tracked, nothing to check/reset
    if position.get("status") is None:
        return False

    log_debug(f"Checking state consistency for active {position['status']} position on {sym}...")
    reset_required = False
    closure_reason = "Unknown"
    order_to_cancel: Optional[str] = None  # ID of the counterpart order to cancel if one filled
    cancel_label: str = ""  # Label ('SL' or 'TP') for logging cancellation attempt

    try:
        # --- 1. Check Status of Tracked Protection Orders ---
        sl_id = position.get("sl_order_id")
        tp_id = position.get("tp_order_id")
        oco_id = position.get("oco_order_id")
        has_tracked_orders = bool(sl_id or tp_id or oco_id)
        protection_order_still_open = False  # Flag if we find any tracked protection order open

        if has_tracked_orders:
            log_debug(f"Fetching open orders for {sym} to check tracked IDs: SL={sl_id}, TP={tp_id}, OCO={oco_id}")
            open_orders = exch.fetch_open_orders(sym)
            open_order_ids = {o["id"] for o in open_orders if "id" in o}
            log_debug(
                f"Found {len(open_order_ids)} open order IDs for {sym}: {open_order_ids if open_order_ids else 'None'}"
            )

            if oco_id:
                # Check if the OCO reference ID is still present (simplistic check)
                # Note: OCO state checking is complex and exchange-specific. This assumes
                # the main OCO ID remains until one leg fills or it's cancelled.
                if oco_id in open_order_ids:
                    protection_order_still_open = True
                    log_debug(f"Tracked OCO order {oco_id} appears to be open.")
                else:
                    reset_required = True
                    closure_reason = f"Tracked OCO order {oco_id} not found among open orders"
                    log_info(f"{NEON_YELLOW}Closure detected: {closure_reason}. Assuming OCO filled/cancelled.{RESET}")
            else:
                # Check separate SL and TP orders
                sl_is_open = sl_id in open_order_ids if sl_id else False
                tp_is_open = tp_id in open_order_ids if tp_id else False
                log_debug(f"Separate Order Check: SL Open = {sl_is_open}, TP Open = {tp_is_open}")

                if sl_id and not sl_is_open and tp_id and not tp_is_open:
                    # Both SL and TP are gone - position must be closed
                    reset_required = True
                    closure_reason = f"Neither tracked SL ({sl_id}) nor TP ({tp_id}) orders found open"
                    log_info(f"{NEON_YELLOW}Closure detected: {closure_reason}. Assuming position closed.{RESET}")
                elif sl_id and not sl_is_open:
                    # SL is gone, TP might still be open (or ID was None)
                    reset_required = True
                    closure_reason = f"Tracked SL order {sl_id} not found open"
                    log_info(f"{NEON_YELLOW}Closure detected: {closure_reason}. Assuming SL was hit.{RESET}")
                    if tp_is_open:  # If TP is still open, mark it for cancellation
                        order_to_cancel = tp_id
                        cancel_label = "TP"
                elif tp_id and not tp_open:
                    # TP is gone, SL might still be open (or ID was None)
                    reset_required = True
                    closure_reason = f"Tracked TP order {tp_id} not found open"
                    log_info(
                        f"{NEON_GREEN}Closure detected: {closure_reason}. Assuming TP was hit.{RESET}"
                    )  # Usually green event
                    if sl_is_open:  # If SL is still open, mark it for cancellation
                        order_to_cancel = sl_id
                        cancel_label = "SL"
                elif sl_is_open or tp_is_open:
                    # At least one of the tracked separate orders is still open
                    protection_order_still_open = True
                    log_debug("At least one tracked separate protection order (SL/TP) is still open.")

        # --- 2. Fallback/Confirmation: Check Actual Position (Contracts Only) ---
        # When is this check useful?
        # - If no protection orders were tracked initially (e.g., state file corruption, TSL failure).
        # - If the order check above *didn't* find any open protection orders (confirming closure).
        # - Only applicable for contract markets where positions are explicitly tracked.
        # - Requires fetchPositions capability.
        should_check_position = (
            is_contract_market
            and exch.has.get("fetchPositions")
            and (not has_tracked_orders or not protection_order_still_open)
        )

        if should_check_position:
            log_info(
                "Order check suggests closure or orders untracked. Verifying position status via fetchPositions..."
            )
            try:
                # Use explicit retry decorator for safety on this critical check
                @api_retry_decorator
                def fetch_positions_safe(e: ccxt.Exchange, s_list: List[str]) -> List[Dict]:
                    # Some exchanges need symbols, some don't. Passing symbol is safer.
                    return e.fetch_positions(symbols=s_list)

                positions_data = fetch_positions_safe(exch, [sym])
                position_found_on_exchange = False

                if positions_data and isinstance(positions_data, list):
                    for pos_info in positions_data:
                        # Check if this entry corresponds to our symbol
                        if isinstance(pos_info, dict) and pos_info.get("symbol") == sym:
                            # Extract position size robustly (handle variations like 'contracts', 'size', string/float)
                            size_val = pos_info.get(
                                "contracts",  # Common field
                                pos_info.get("info", {}).get(
                                    "size",  # Binance often uses info.size
                                    pos_info.get("info", {}).get(
                                        "positionAmt",  # Binance sometimes info.positionAmt
                                        pos_info.get("info", {}).get(
                                            "qty",  # Bybit often uses info.qty
                                            0,
                                        ),
                                    ),
                                ),
                            )  # Default to 0 if not found
                            try:
                                current_pos_size = float(size_val) if size_val is not None else 0.0
                            except (ValueError, TypeError):
                                log_warning(
                                    f"Could not parse position size ('{size_val}') from fetchPositions data for {sym}. Assuming 0."
                                )
                                current_pos_size = 0.0

                            # Determine position side from data (handle variations)
                            pos_side_on_exchange = pos_info.get(
                                "side",  # Standard field
                                pos_info.get("info", {}).get(
                                    "side",  # Bybit uses info.side
                                    "unknown",
                                ),
                            ).lower()
                            # Binance sometimes uses positionAmt sign instead of 'side' field
                            if (
                                pos_side_on_exchange == "unknown"
                                and "info" in pos_info
                                and "positionAmt" in pos_info["info"]
                            ):
                                pos_amt_sign = float(pos_info["info"].get("positionAmt", 0.0))
                                if pos_amt_sign > 0:
                                    pos_side_on_exchange = "long"
                                elif pos_amt_sign < 0:
                                    pos_side_on_exchange = "short"

                            log_debug(
                                f"fetchPositions found entry for {sym}: Size={current_pos_size}, Side='{pos_side_on_exchange}'"
                            )

                            # Check if size is effectively non-zero (use min_amount or small epsilon)
                            size_threshold = (
                                min_amount if min_amount is not None else 1e-9
                            )  # Use min_amount if known, else small value
                            if abs(current_pos_size) > size_threshold:
                                position_found_on_exchange = True
                                log_info(
                                    f"fetchPositions confirms an active {pos_side_on_exchange} position exists for {sym} (Size: {current_pos_size})."
                                )

                                # --- State Mismatch Check ---
                                if position["status"] == pos_side_on_exchange:
                                    # Exchange position matches local state. This overrides previous reset flags.
                                    if reset_required:
                                        log_warning(
                                            f"Position check confirms active {position['status']} position. Overriding closure detected by order check. State remains active."
                                        )
                                        reset_required = False  # Keep state active
                                        order_to_cancel = None  # Don't cancel orders if position is confirmed active
                                else:
                                    # Mismatch! Local state says one thing, exchange says another. Critical error.
                                    log_error("!!! STATE MISMATCH DETECTED !!!")
                                    log_error(f"  Local State: {position['status']}")
                                    log_error(f"  Exchange Position: {pos_side_on_exchange} (Size: {current_pos_size})")
                                    log_error("  Resetting local state to None. MANUAL REVIEW STRONGLY ADVISED.")
                                    reset_required = True
                                    closure_reason = "State Mismatch detected via fetchPositions"
                                    # Mark any tracked orders for cancellation due to mismatch
                                    order_to_cancel = oco_id or sl_id or tp_id
                                    cancel_label = "SL/TP/OCO due to state mismatch"
                                break  # Found the position for our symbol, no need to check further list items
                        # else: # Debugging: log if symbol doesn't match
                        #     log_debug(f"fetchPositions entry skipped: Symbol '{pos_info.get('symbol')}' != '{sym}'")

                    # After checking all returned positions: if we expected a position but found none active
                    if not position_found_on_exchange and position["status"] is not None:
                        log_info(f"Position check via fetchPositions confirmed no active position found for {sym}.")
                        if not reset_required:  # If order check didn't already flag reset
                            reset_required = True
                            closure_reason = "Position not found via fetchPositions"
                            log_info(f"{NEON_YELLOW}Closure confirmed: {closure_reason}. Resetting state.{RESET}")
                            # Mark any tracked orders for cancellation
                            order_to_cancel = oco_id or sl_id or tp_id
                            cancel_label = "SL/TP/OCO as position closed"

            except ccxt.NotSupported:
                log_warning(
                    "fetchPositions is not supported by this exchange. Cannot use it for position confirmation."
                )
            except Exception as e:
                log_error(
                    f"Error during fetchPositions check: {e}. Relying solely on order status check.", exc_info=False
                )  # Decorator logs full trace if needed

        # --- 3. Perform State Reset and Cleanup if Required ---
        if reset_required:
            log_info(f"Resetting local position state. Reason: {closure_reason}.")

            # Attempt to cancel the counterpart order if identified
            if order_to_cancel and cancel_label:
                log_info(f"Attempting to cancel potentially lingering {cancel_label} order: {order_to_cancel}")
                try:
                    # Use the robust cancel function
                    cancelled = cancel_order_with_retry(exch, order_to_cancel, sym)
                    if cancelled:
                        log_info(
                            f"Cancellation request for lingering order {order_to_cancel} sent or order already closed."
                        )
                    else:
                        log_warning(
                            f"Failed to cancel lingering {cancel_label} order {order_to_cancel}. Manual check might be needed."
                        )
                except Exception as e:
                    # Catch errors during the cancellation attempt itself
                    log_error(
                        f"Error occurred while trying to cancel lingering order {order_to_cancel}: {e}", exc_info=True
                    )

            # Reset the global position state to default
            position.update(position_default_structure.copy())
            save_position_state()  # Persist the reset state
            log_info("Local position state has been reset.")
            display_position_status(
                position, price_precision_digits, amount_precision_digits
            )  # Display the cleared state
            return True  # Indicate that state was reset

        # If we reached here without reset_required being True, the position is assumed active and consistent
        log_debug("Position state appears consistent with tracked orders/position check.")
        return False  # State was not reset

    except (ccxt.AuthenticationError, ccxt.ExchangeError) as e:
        # Handle errors during the initial fetch_open_orders or fetch_positions call
        log_error(
            f"Error checking position/orders due to exchange communication issue: {e}. State remains unchanged.",
            exc_info=False,
        )
        return False  # Cannot determine state, do not reset prematurely
    except Exception as e:
        # Catch any other unexpected errors during the check
        if not isinstance(e, RETRYABLE_EXCEPTIONS):  # Avoid double logging
            log_error(
                f"Unexpected error during position/order consistency check: {e}. State remains unchanged.",
                exc_info=True,
            )
        return False  # Cannot determine state, do not reset


# --- Trailing Stop ---
def update_trailing_stop(exch: ccxt.Exchange, sym: str, current_price: float, current_atr: Optional[float]) -> None:
    """
    Checks if the trailing stop loss needs to be updated based on price movement and ATR.
    If an update is required, it cancels the existing protection order(s) and
    places new ones with the updated stop loss price.
    """
    global position, enable_trailing_stop, trailing_stop_atr_multiplier, trailing_stop_activation_atr_multiplier
    global price_precision_digits, amount_precision_digits  # For logging/formatting

    # --- Pre-checks ---
    if not enable_trailing_stop:
        return  # TSL feature disabled
    if position.get("status") is None:
        return  # No active position
    if current_atr is None or current_atr <= 0:
        log_warning("TSL Check: Cannot update trailing stop, invalid ATR value received.")
        return

    # --- Get Required Position State ---
    pos_side = position["status"]  # 'long' or 'short'
    entry_price = position.get("entry_price")
    initial_sl_price = position.get("stop_loss")  # Original SL target
    original_tp_price = position.get("take_profit")  # Need original TP for replacement
    position_qty = position.get("quantity")
    current_tsl_price = position.get("current_trailing_sl_price")  # The SL price after last TSL update
    highest_seen = position.get("highest_price_since_entry")
    lowest_seen = position.get("lowest_price_since_entry")

    # Check if essential data is present
    if None in [entry_price, initial_sl_price, original_tp_price, position_qty, highest_seen, lowest_seen]:
        log_warning("TSL Check: Missing essential position data (entry, SL, TP, qty, peak prices). Cannot proceed.")
        return

    # Determine the current effective SL price (either initial SL or the last TSL price)
    effective_sl_price = current_tsl_price if current_tsl_price is not None else initial_sl_price
    log_debug(
        f"TSL Check ({pos_side}): Current Price={current_price:.{price_precision_digits}f}, Eff SL={effective_sl_price:.{price_precision_digits}f}, ATR={current_atr:.{price_precision_digits}f}"
    )

    # --- Update Peak Prices Reached Since Entry ---
    if pos_side == "long" and current_price > highest_seen:
        position["highest_price_since_entry"] = current_price
        log_debug(f"TSL: New high price detected: {current_price:.{price_precision_digits}f}")
    elif pos_side == "short" and current_price < lowest_seen:
        position["lowest_price_since_entry"] = current_price
        log_debug(f"TSL: New low price detected: {current_price:.{price_precision_digits}f}")

    # --- Check TSL Activation Threshold ---
    tsl_activated = False
    potential_new_tsl_price: Optional[float] = None
    if pos_side == "long":
        # Activation threshold: Price must move X ATRs above entry
        activation_threshold = entry_price + (current_atr * trailing_stop_activation_atr_multiplier)
        current_high = position["highest_price_since_entry"]  # Use updated high
        if current_high > activation_threshold:
            tsl_activated = True
            # Calculate potential new TSL: Peak Price - Y ATRs
            potential_new_tsl_price = current_high - (current_atr * trailing_stop_atr_multiplier)
            log_debug(
                f"Long TSL potentially active. High ({current_high:.{price_precision_digits}f}) > Threshold ({activation_threshold:.{price_precision_digits}f})."
            )
            log_debug(f"  Potential New TSL Price: {potential_new_tsl_price:.{price_precision_digits}f}")
    else:  # Short position
        # Activation threshold: Price must move X ATRs below entry
        activation_threshold = entry_price - (current_atr * trailing_stop_activation_atr_multiplier)
        current_low = position["lowest_price_since_entry"]  # Use updated low
        if current_low < activation_threshold:
            tsl_activated = True
            # Calculate potential new TSL: Peak Price + Y ATRs
            potential_new_tsl_price = current_low + (current_atr * trailing_stop_atr_multiplier)
            log_debug(
                f"Short TSL potentially active. Low ({current_low:.{price_precision_digits}f}) < Threshold ({activation_threshold:.{price_precision_digits}f})."
            )
            log_debug(f"  Potential New TSL Price: {potential_new_tsl_price:.{price_precision_digits}f}")

    # --- Check if Potential New TSL is an Improvement and Valid ---
    should_update_tsl = False
    new_tsl_price_formatted: Optional[float] = None
    if tsl_activated and potential_new_tsl_price is not None:
        # Format the potential new TSL price to exchange precision
        try:
            new_tsl_price_formatted = float(exch.price_to_precision(sym, potential_new_tsl_price))
        except Exception as fmt_e:
            log_error(f"TSL Update: Failed to format potential new TSL price {potential_new_tsl_price}: {fmt_e}")
            return  # Cannot proceed without formatted price

        # Check if the new TSL is strictly better than the current effective SL
        # AND ensure the new TSL hasn't crossed the current price (which would cause immediate stop out)
        if pos_side == "long":
            if new_tsl_price_formatted > effective_sl_price and new_tsl_price_formatted < current_price:
                should_update_tsl = True
        else:  # Short position
            if new_tsl_price_formatted < effective_sl_price and new_tsl_price_formatted > current_price:
                should_update_tsl = True

        if should_update_tsl:
            log_info(f"{NEON_YELLOW}Trailing Stop Update Triggered!{RESET}")
            log_info(f"  Position: {pos_side.upper()}")
            log_info(f"  Current Price: {current_price:.{price_precision_digits}f}")
            log_info(f"  Current Eff SL: {effective_sl_price:.{price_precision_digits}f}")
            log_info(f"  Potential New TSL: {new_tsl_price_formatted:.{price_precision_digits}f}")
        else:
            log_debug(
                f"TSL Update Condition Check: New TSL ({new_tsl_price_formatted:.{price_precision_digits}f if new_tsl_price_formatted else 'N/A'}) vs Effective SL ({effective_sl_price:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f}). No update needed."
            )

    # --- Execute TSL Update (Cancel Old, Place New) ---
    if should_update_tsl and new_tsl_price_formatted is not None:
        # Identify the existing protection order(s) to cancel
        ids_to_cancel: List[Tuple[str, str]] = []  # List of (ID, Label)
        if position.get("oco_order_id"):
            ids_to_cancel.append((position["oco_order_id"], "OCO"))
        else:
            if position.get("sl_order_id"):
                ids_to_cancel.append((position["sl_order_id"], "SL"))
            if position.get("tp_order_id"):
                ids_to_cancel.append((position["tp_order_id"], "TP"))

        if not ids_to_cancel:
            log_error(
                "TSL CRITICAL: Update triggered, but no existing protection order IDs found in state to cancel. Aborting update."
            )
            # Consider resetting state or trying to close position? Very risky state.
            return

        log_info(f"TSL: Proceeding with update. Moving SL to {new_tsl_price_formatted:.{price_precision_digits}f}")

        # 1. Cancel Existing Protection Order(s)
        all_cancelled_successfully = True
        for order_id, label in ids_to_cancel:
            log_info(f"TSL: Attempting to cancel existing {label} order: {order_id}")
            if not cancel_order_with_retry(exch, order_id, sym):
                log_error(
                    f"TSL CRITICAL: Failed to cancel existing {label} order {order_id}. Aborting TSL update to avoid inconsistent state."
                )
                all_cancelled_successfully = False
                break  # Stop cancellation process if one fails

        if not all_cancelled_successfully:
            # If cancellation failed, the old order might still be active. Do NOT proceed.
            # The position *might* still be protected by the old order.
            log_error("TSL Update Aborted due to cancellation failure.")
            return

        log_info("TSL: Existing protection order(s) cancelled successfully.")
        # Clear old IDs from state immediately after cancellation confirmation
        position["sl_order_id"] = None
        position["tp_order_id"] = None
        position["oco_order_id"] = None

        # 2. Place New Protection Order(s) with Updated SL, Original TP
        log_info(
            f"TSL: Placing new protection orders: SL @ {new_tsl_price_formatted:.{price_precision_digits}f}, Original TP @ {original_tp_price:.{price_precision_digits}f}"
        )
        new_sl_id, new_tp_id, new_oco_id = place_sl_tp_orders(
            exch=exch,
            sym=sym,
            pos_side=pos_side,
            qty=position_qty,
            sl_pr=new_tsl_price_formatted,  # Use the new TSL price
            tp_pr=original_tp_price,  # Keep the original TP target
        )

        # 3. Update State Based on New Order Placement Result
        protection_placed = bool(new_oco_id or new_sl_id)  # Consider success if at least SL or OCO is placed

        if protection_placed:
            log_info(f"{NEON_GREEN}TSL Update Successful: New protection orders placed.{RESET}")
            if new_oco_id:
                log_info(f"  New OCO Ref ID: {new_oco_id}")
            if new_sl_id:
                log_info(f"  New SL ID: {new_sl_id}")
            if new_tp_id:
                log_info(f"  New TP ID: {new_tp_id}")

            # Update the position state with new IDs and the new TSL price
            position.update(
                {
                    "sl_order_id": new_sl_id,
                    "tp_order_id": new_tp_id,
                    "oco_order_id": new_oco_id,
                    "stop_loss": new_tsl_price_formatted,  # Update the target SL
                    "current_trailing_sl_price": new_tsl_price_formatted,  # Mark this as the current TSL
                }
            )
            save_position_state()  # Persist the updated state
            display_position_status(position, price_precision_digits, amount_precision_digits)

            # Warn if only partial protection was placed (e.g., SL ok, but separate TP failed)
            if not new_oco_id and not new_tp_id and new_sl_id:
                log_warning("TSL Update Warning: New SL order placed, but separate TP order failed. Check manually.")
            elif not new_oco_id and not new_sl_id:  # Should not happen if protection_placed is True, but safety check
                log_error("TSL Update Error: Logic error - protection marked as placed but no SL/OCO ID returned.")
        else:
            # --- CRITICAL FAILURE ---
            # Old orders were cancelled, but placing new ones failed. Position is unprotected!
            log_error("!!! TSL CRITICAL FAILURE !!!")
            log_error(f"  Successfully cancelled old protection order(s): {ids_to_cancel}")
            log_error(
                f"  BUT FAILED to place new protection orders (SL={new_tsl_price_formatted}, TP={original_tp_price})."
            )
            log_error(f"  !!! POSITION '{pos_side.upper()}' IS CURRENTLY UNPROTECTED !!!")
            # Update state to reflect unprotected status
            position.update(
                {
                    "sl_order_id": None,
                    "tp_order_id": None,
                    "oco_order_id": None,
                    "stop_loss": new_tsl_price_formatted,  # Keep target SL price for reference
                    "current_trailing_sl_price": None,  # Clear active TSL marker
                    # Keep other position details like entry, qty, time
                }
            )
            save_position_state()  # Save the unprotected state
            log_error("State saved reflecting unprotected status. MANUAL INTERVENTION REQUIRED IMMEDIATELY.")
            # Consider adding an emergency market close attempt here?
            # emergency_close_position(exch, sym, pos_side, position_qty)

    # else: # Debugging: Log if update wasn't needed
    #     if tsl_activated and potential_new_tsl_price is not None:
    #         log_debug("TSL update conditions met, but new SL was not an improvement or was invalid.")


# --- Main Trading Loop ---
def run_bot():
    """Main execution function containing the trading loop."""
    log_info(f"Initializing RSI/OB Trader Bot for {symbol} on {exchange.name} ({timeframe})...")
    load_position_state()  # Load state at startup

    log_info("-" * 70)
    log_info("Trading Configuration:")
    log_info(f"  Symbol: {symbol}, Timeframe: {timeframe}")
    log_info(f"  Risk per trade: {risk_percentage * 100:.2f}%")
    log_info(f"  Simulation Mode: {SIMULATION_MODE}")
    log_info(f"  Cycle Interval: {sleep_interval_seconds} seconds")
    log_info(f"  SL/TP Mode: {'ATR Based' if enable_atr_sl_tp else 'Fixed Percentage'}")
    if enable_atr_sl_tp:
        log_info(f"    ATR SL: {atr_sl_multiplier}x, ATR TP: {atr_tp_multiplier}x (ATR Length: {atr_length})")
    else:
        log_info(f"    Fixed SL: {stop_loss_percentage * 100:.2f}%, Fixed TP: {take_profit_percentage * 100:.2f}%")
    log_info(f"  Trailing Stop Loss: {'Enabled' if enable_trailing_stop else 'Disabled'}")
    if enable_trailing_stop:
        log_info(
            f"    TSL Activation: {trailing_stop_activation_atr_multiplier}x ATR, TSL Distance: {trailing_stop_atr_multiplier}x ATR"
        )
    log_info(f"  Order Block Volume Multiplier: {ob_volume_threshold_multiplier}x Avg Vol")
    log_info(f"  Entry Volume Confirmation: {'Enabled' if entry_volume_confirmation_enabled else 'Disabled'}")
    if entry_volume_confirmation_enabled:
        log_info(f"    Vol MA Length: {entry_volume_ma_length}, Vol Multiplier: {entry_volume_multiplier}x")
    log_info("-" * 70)

    if not SIMULATION_MODE:
        log_warning("LIVE TRADING ACTIVE. Press Ctrl+C to stop gracefully.")
    else:
        log_info("SIMULATION MODE ACTIVE. Press Ctrl+C to stop.")

    # --- Main Loop ---
    while True:
        try:
            cycle_start_time = pd.Timestamp.now(tz="UTC")
            print_cycle_divider(cycle_start_time)

            # === 1. Check Position Consistency / Handle Closure ===
            # This function checks if tracked orders were filled or if fetchPositions indicates closure
            # It resets the local 'position' state if closure is detected.
            position_was_reset = check_position_and_orders(exchange, symbol)

            # Always display current status after check/reset
            display_position_status(position, price_precision_digits, amount_precision_digits)

            if position_was_reset:
                log_info("Position state was reset in this cycle. Proceeding to check for new signals.")
                # Optional: Add a small delay here if needed after a reset
                # time.sleep(1)
                # Continue to the start of the next loop iteration immediately
                # to avoid processing signals based on potentially old data from before the reset.
                # However, fetching fresh data right away is usually desired.
                # Consider if a 'continue' here is better or worse than proceeding with fresh data fetch.
                # Decision: Proceed to fetch fresh data immediately after reset.

            # === 2. Fetch Fresh Market Data ===
            df_ohlcv = fetch_ohlcv_data(exchange, symbol, timeframe, data_limit)
            if df_ohlcv is None or df_ohlcv.empty:
                log_warning("Failed to fetch or process valid OHLCV data. Waiting for next cycle.")
                neon_sleep_timer(sleep_interval_seconds)
                continue  # Skip rest of the cycle

            # === 3. Calculate Technical Indicators ===
            stoch_params = {"k": stoch_k, "d": stoch_d, "smooth_k": stoch_smooth_k}
            df_indicators = calculate_technical_indicators(
                df=df_ohlcv,
                rsi_l=rsi_length,
                stoch_p=stoch_params,
                atr_l=atr_length,
                vol_ma_l=entry_volume_ma_length,
                calc_atr=needs_atr,  # Only calculate if TSL or ATR SL/TP enabled
                calc_vol_ma=entry_volume_confirmation_enabled,  # Only calculate if needed
            )
            if df_indicators is None or df_indicators.empty:
                log_warning("Failed to calculate technical indicators. Waiting for next cycle.")
                neon_sleep_timer(sleep_interval_seconds)
                continue  # Skip rest of the cycle

            # === 4. Extract Latest Data Point ===
            latest_candle = df_indicators.iloc[-1]
            latest_timestamp = latest_candle.name

            # Define expected column names based on config
            rsi_col_name = f"RSI_{rsi_length}"
            stoch_k_col = f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}"
            stoch_d_col = f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}"
            atr_col_name = f"ATRr_{atr_length}" if needs_atr else None
            vol_ma_col_name = f"VOL_MA_{entry_volume_ma_length}" if entry_volume_confirmation_enabled else None

            # Safely extract latest values, checking for existence and validity
            try:
                current_price = float(latest_candle["close"])
                float(latest_candle["high"])  # Needed for OB context
                float(latest_candle["low"])  # Needed for OB context
                current_volume = float(latest_candle["volume"]) if "volume" in latest_candle else None

                latest_rsi = float(latest_candle[rsi_col_name])
                latest_stoch_k = float(latest_candle[stoch_k_col])
                latest_stoch_d = float(latest_candle[stoch_d_col])

                latest_atr: Optional[float] = None
                if (
                    needs_atr
                    and atr_col_name
                    and atr_col_name in latest_candle
                    and pd.notna(latest_candle[atr_col_name])
                ):
                    latest_atr = float(latest_candle[atr_col_name])
                    if latest_atr <= 0:
                        latest_atr = None  # Treat non-positive ATR as invalid

                latest_vol_ma: Optional[float] = None
                if (
                    entry_volume_confirmation_enabled
                    and vol_ma_col_name
                    and vol_ma_col_name in latest_candle
                    and pd.notna(latest_candle[vol_ma_col_name])
                ):
                    latest_vol_ma = float(latest_candle[vol_ma_col_name])
                    if latest_vol_ma <= 0:
                        latest_vol_ma = None  # Treat non-positive MA as invalid

                # Check for NaN in essential values after extraction
                if any(math.isnan(v) for v in [current_price, latest_rsi, latest_stoch_k, latest_stoch_d]):
                    raise ValueError("NaN value found in essential indicator data.")

            except (KeyError, ValueError, TypeError) as e:
                log_error(
                    f"Error extracting latest indicator data at {latest_timestamp}: {e}. Check indicator calculation and column names.",
                    exc_info=False,
                )
                neon_sleep_timer(sleep_interval_seconds)
                continue  # Skip cycle if data is bad

            # Log warnings if needed indicators are missing/invalid
            if needs_atr and latest_atr is None:
                log_warning(
                    f"ATR calculation needed (ATR SL/TP or TSL enabled) but result is invalid or missing for timestamp {latest_timestamp}."
                )
            if entry_volume_confirmation_enabled and latest_vol_ma is None:
                log_debug(
                    f"Volume MA needed for entry confirmation but result is invalid or missing for timestamp {latest_timestamp}."
                )  # Debug level might be sufficient

            # Display current market stats
            display_market_stats(
                current_price, latest_rsi, latest_stoch_k, latest_stoch_d, latest_atr, price_precision_digits
            )

            # === 5. Identify Potential Order Blocks ===
            # Pass the DataFrame with indicators calculated
            bullish_ob, bearish_ob = identify_potential_order_block(
                df=df_indicators, vol_thresh_mult=ob_volume_threshold_multiplier, lookback=ob_lookback
            )
            display_order_blocks(bullish_ob, bearish_ob, price_precision_digits)

            # === 6. Trading Logic Execution ===

            # --- A. If IN a Position: Monitor & Trail ---
            if position.get("status") is not None:
                log_info(
                    f"Monitoring active {position['status'].upper()} position entered at {position.get('entry_price', 'N/A'):.{price_precision_digits}f}..."
                )
                # Check and potentially update the trailing stop loss
                update_trailing_stop(exchange, symbol, current_price, latest_atr)
                # No new entries considered while in a position
                log_debug("Position active. Waiting for SL/TP/TSL or manual intervention.")

            # --- B. If NOT in a Position: Check for Entry Signals ---
            else:
                log_info("No active position. Checking for entry signals...")

                # --- Evaluate Entry Conditions ---
                # 1. Volume Confirmation (if enabled)
                volume_condition_met = True  # Assume true if disabled
                if entry_volume_confirmation_enabled:
                    if current_volume is not None and latest_vol_ma is not None:
                        volume_condition_met = current_volume > latest_vol_ma * entry_volume_multiplier
                        log_debug(
                            f"Volume Check: Current Vol={current_volume:.2f}, Vol MA={latest_vol_ma:.2f}, Threshold={latest_vol_ma * entry_volume_multiplier:.2f} -> {'OK' if volume_condition_met else 'FAILED'}"
                        )
                    else:
                        volume_condition_met = False  # Fail if data missing
                        log_debug("Volume Check: FAILED (Missing volume or Vol MA data)")

                # 2. Base RSI & Stochastic Oversold/Overbought Signals
                base_long_signal = (
                    latest_rsi < rsi_oversold and latest_stoch_k < stoch_oversold and latest_stoch_d < stoch_oversold
                )
                base_short_signal = (
                    latest_rsi > rsi_overbought
                    and latest_stoch_k > stoch_overbought
                    and latest_stoch_d > stoch_overbought
                )
                if base_long_signal:
                    log_debug("Base Signal: LONG (RSI & Stoch Oversold)")
                if base_short_signal:
                    log_debug("Base Signal: SHORT (RSI & Stoch Overbought)")

                # 3. Order Block Confirmation
                ob_confirmation_met = False
                ob_entry_reason = ""
                entry_order_block: Optional[Dict] = None  # Store the OB used for entry/SL refinement

                if base_long_signal and bullish_ob:
                    # Check if current price is within the bullish OB range
                    if bullish_ob["low"] <= current_price <= bullish_ob["high"]:
                        ob_confirmation_met = True
                        entry_order_block = bullish_ob
                        ob_entry_reason = f"Price ({current_price:.{price_precision_digits}f}) entered Bullish OB ({bullish_ob['low']:.{price_precision_digits}f} - {bullish_ob['high']:.{price_precision_digits}f})"
                        log_debug(f"OB Confirmation: LONG Signal OK ({ob_entry_reason})")
                    else:
                        log_debug(
                            f"OB Confirmation: LONG Signal FAILED (Price {current_price:.{price_precision_digits}f} outside Bullish OB)"
                        )
                elif base_short_signal and bearish_ob:
                    # Check if current price is within the bearish OB range
                    if bearish_ob["low"] <= current_price <= bearish_ob["high"]:
                        ob_confirmation_met = True
                        entry_order_block = bearish_ob
                        ob_entry_reason = f"Price ({current_price:.{price_precision_digits}f}) entered Bearish OB ({bearish_ob['low']:.{price_precision_digits}f} - {bearish_ob['high']:.{price_precision_digits}f})"
                        log_debug(f"OB Confirmation: SHORT Signal OK ({ob_entry_reason})")
                    else:
                        log_debug(
                            f"OB Confirmation: SHORT Signal FAILED (Price {current_price:.{price_precision_digits}f} outside Bearish OB)"
                        )
                elif base_long_signal or base_short_signal:
                    # Base signal present, but no corresponding OB found or price not inside
                    log_debug("OB Confirmation: FAILED (No suitable OB found or price outside range)")

                # 4. Final Entry Decision
                entry_side: Optional[str] = None  # 'long' or 'short'
                entry_reason: str = ""

                if base_long_signal and ob_confirmation_met and volume_condition_met and entry_order_block:
                    entry_side = "long"
                    entry_reason = f"RSI({latest_rsi:.1f})<OS({rsi_oversold}), Stoch({latest_stoch_k:.1f},{latest_stoch_d:.1f})<OS({stoch_oversold}), {ob_entry_reason}"
                    if entry_volume_confirmation_enabled:
                        entry_reason += ", Volume Confirmed"
                elif base_short_signal and ob_confirmation_met and volume_condition_met and entry_order_block:
                    entry_side = "short"
                    entry_reason = f"RSI({latest_rsi:.1f})>OB({rsi_overbought}), Stoch({latest_stoch_k:.1f},{latest_stoch_d:.1f})>OB({stoch_overbought}), {ob_entry_reason}"
                    if entry_volume_confirmation_enabled:
                        entry_reason += ", Volume Confirmed"
                elif base_long_signal or base_short_signal:
                    # Log why entry was blocked if base signal was present
                    reasons = []
                    if not ob_confirmation_met:
                        reasons.append("OB Confirmation Failed")
                    if not volume_condition_met:
                        reasons.append("Volume Confirmation Failed")
                    log_info(
                        f"Entry blocked for potential {'LONG' if base_long_signal else 'SHORT'} signal. Reason(s): {', '.join(reasons)}"
                    )

                # --- Execute Entry Sequence if Signal Found ---
                if entry_side and entry_reason and entry_order_block:
                    display_signal("Entry", entry_side, entry_reason)

                    # --- Step 1: Calculate Initial SL and TP ---
                    initial_sl_price: Optional[float] = None
                    initial_tp_price: Optional[float] = None

                    if enable_atr_sl_tp:
                        if latest_atr:
                            sl_distance = latest_atr * atr_sl_multiplier
                            tp_distance = latest_atr * atr_tp_multiplier
                            initial_sl_price = (
                                current_price - sl_distance if entry_side == "long" else current_price + sl_distance
                            )
                            initial_tp_price = (
                                current_price + tp_distance if entry_side == "long" else current_price - tp_distance
                            )
                        else:
                            log_error("Cannot calculate ATR-based SL/TP: Invalid ATR value. Aborting entry.")
                            continue  # Skip to next cycle
                    else:  # Fixed Percentage SL/TP
                        sl_multiplier = (
                            1.0 - stop_loss_percentage if entry_side == "long" else 1.0 + stop_loss_percentage
                        )
                        tp_multiplier = (
                            1.0 + take_profit_percentage if entry_side == "long" else 1.0 - take_profit_percentage
                        )
                        initial_sl_price = current_price * sl_multiplier
                        initial_tp_price = current_price * tp_multiplier

                    if initial_sl_price is None or initial_tp_price is None:
                        log_error("Failed to calculate initial SL/TP prices. Aborting entry.")
                        continue

                    log_info(
                        f"Initial Calculated SL: {initial_sl_price:.{price_precision_digits}f}, TP: {initial_tp_price:.{price_precision_digits}f}"
                    )

                    # --- Step 2: Refine SL based on Order Block ---
                    # Place SL just beyond the low (for long) or high (for short) of the OB
                    # Add a small buffer (e.g., a few ticks)
                    sl_buffer_ticks = 5  # Adjust as needed
                    sl_buffer = min_tick * sl_buffer_ticks
                    refined_sl_price = initial_sl_price  # Default to initial calc

                    ob_low = entry_order_block["low"]
                    ob_high = entry_order_block["high"]

                    if entry_side == "long":
                        potential_refined_sl = ob_low - sl_buffer
                        # Use refined SL only if it's further away (safer) than initial calc AND still below entry
                        if potential_refined_sl < initial_sl_price and potential_refined_sl < current_price:
                            refined_sl_price = potential_refined_sl
                            log_info(
                                f"Refining SL based on Bullish OB low ({ob_low:.{price_precision_digits}f}) to: {refined_sl_price:.{price_precision_digits}f}"
                            )
                    else:  # Short entry
                        potential_refined_sl = ob_high + sl_buffer
                        # Use refined SL only if it's further away (safer) than initial calc AND still above entry
                        if potential_refined_sl > initial_sl_price and potential_refined_sl > current_price:
                            refined_sl_price = potential_refined_sl
                            log_info(
                                f"Refining SL based on Bearish OB high ({ob_high:.{price_precision_digits}f}) to: {refined_sl_price:.{price_precision_digits}f}"
                            )

                    final_sl_price = refined_sl_price
                    final_tp_price = initial_tp_price  # TP usually not refined by OB

                    # --- Step 3: Final SL/TP Validation & Formatting ---
                    try:
                        final_sl_price_fmt = float(exchange.price_to_precision(symbol, final_sl_price))
                        final_tp_price_fmt = float(exchange.price_to_precision(symbol, final_tp_price))

                        # Final logical check after formatting
                        if (
                            entry_side == "long"
                            and (final_sl_price_fmt >= current_price or final_tp_price_fmt <= current_price)
                        ) or (
                            entry_side == "short"
                            and (final_sl_price_fmt <= current_price or final_tp_price_fmt >= current_price)
                        ):
                            raise ValueError(
                                f"Invalid final SL/TP logic after formatting: SL={final_sl_price_fmt}, TP={final_tp_price_fmt}, Entry={current_price}"
                            )

                    except ValueError as ve:
                        log_error(f"Final SL/TP validation failed: {ve}. Aborting entry.")
                        continue
                    except Exception as fmt_e:
                        log_error(
                            f"Failed to format final SL ({final_sl_price}) or TP ({final_tp_price}): {fmt_e}. Aborting entry."
                        )
                        continue

                    log_info(
                        f"Final Target Prices: SL={final_sl_price_fmt:.{price_precision_digits}f}, TP={final_tp_price_fmt:.{price_precision_digits}f}"
                    )

                    # --- Step 4: Calculate Position Size ---
                    entry_quantity = calculate_position_size(
                        exch=exchange,
                        sym=symbol,
                        entry=current_price,  # Use current price for sizing before placing market order
                        sl=final_sl_price_fmt,
                        risk_pct=risk_percentage,
                    )
                    if entry_quantity is None or entry_quantity <= 0:
                        log_error("Position size calculation failed or returned zero/negative. Aborting entry.")
                        continue

                    # --- Step 5: Place Entry Market Order ---
                    market_order_side = "buy" if entry_side == "long" else "sell"
                    entry_order_result = place_market_order(
                        exch=exchange,
                        sym=symbol,
                        side=market_order_side,
                        amount=entry_quantity,
                        reduce_only=False,  # Entry order is never reduceOnly
                    )

                    # --- Step 6: Process Entry Result & Place Protection Orders ---
                    if entry_order_result and entry_order_result.get("id"):
                        entry_order_id = entry_order_result["id"]
                        entry_status = entry_order_result.get("status")
                        # Use actual average fill price if available, else fallback to target entry price
                        actual_entry_price = entry_order_result.get("average", entry_order_result.get("price"))
                        if actual_entry_price is None or actual_entry_price <= 0:
                            log_warning(
                                f"Entry order {entry_order_id} filled, but average price not available. Using initial target price {current_price} for state."
                            )
                            actual_entry_price = current_price
                        else:
                            actual_entry_price = float(actual_entry_price)  # Ensure float

                        # Use actual filled quantity if available, else fallback to requested amount
                        actual_filled_quantity = entry_order_result.get("filled")
                        if actual_filled_quantity is None or actual_filled_quantity <= 0:
                            log_warning(
                                f"Entry order {entry_order_id} filled, but filled quantity not available or zero. Using requested quantity {entry_quantity} for state."
                            )
                            actual_filled_quantity = entry_quantity
                        else:
                            actual_filled_quantity = float(actual_filled_quantity)  # Ensure float

                        log_info(
                            f"Entry market order placed/filled (ID: {entry_order_id}, Status: {entry_status}). Actual Entry Px: ~{actual_entry_price:.{price_precision_digits}f}, Filled Qty: {actual_filled_quantity:.{amount_precision_digits}f}"
                        )

                        # --- Place SL/TP Orders ---
                        log_info(f"Placing protection orders for the new {entry_side.upper()} position...")
                        sl_id, tp_id, oco_id = place_sl_tp_orders(
                            exch=exchange,
                            sym=symbol,
                            pos_side=entry_side,  # Side of the position ('long' or 'short')
                            qty=actual_filled_quantity,  # Use the actual filled quantity
                            sl_pr=final_sl_price_fmt,  # Use the final calculated SL
                            tp_pr=final_tp_price_fmt,  # Use the final calculated TP
                        )

                        # --- Step 7: Update Position State (ONLY if protection placed) ---
                        if oco_id or sl_id:  # Success if at least OCO or separate SL was placed
                            log_info("Protection order(s) placed successfully. Updating position state.")
                            position.update(
                                {
                                    "status": entry_side,
                                    "entry_price": actual_entry_price,
                                    "quantity": actual_filled_quantity,
                                    "order_id": entry_order_id,  # ID of the entry market order
                                    "stop_loss": final_sl_price_fmt,  # Final target SL
                                    "take_profit": final_tp_price_fmt,  # Final target TP
                                    "entry_time": pd.Timestamp.now(tz="UTC"),  # Record entry time
                                    "sl_order_id": sl_id,  # Will be None if OCO used
                                    "tp_order_id": tp_id,  # Will be None if OCO used
                                    "oco_order_id": oco_id,  # Will be None if separate orders used
                                    "highest_price_since_entry": actual_entry_price,  # Initialize for TSL
                                    "lowest_price_since_entry": actual_entry_price,  # Initialize for TSL
                                    "current_trailing_sl_price": None,  # Reset TSL price on new entry
                                }
                            )
                            save_position_state()  # Persist the new active state
                            display_position_status(position, price_precision_digits, amount_precision_digits)
                            log_info(f"Successfully opened {entry_side.upper()} position and saved state.")

                            # Optional warning if separate TP failed
                            if not oco_id and not tp_id and sl_id:
                                log_warning("Entry and SL placement successful, but separate TP order failed.")

                        else:
                            # --- CRITICAL FAILURE: Entry filled, but protection failed ---
                            log_error("!!! CRITICAL ENTRY FAILURE !!!")
                            log_error(
                                f"  Entry order {entry_order_id} filled for {actual_filled_quantity:.{amount_precision_digits}f} {base_currency}"
                            )
                            log_error("  BUT FAILED to place required SL/TP/OCO protection orders.")
                            log_error("  !!! POSITION IS ACTIVE BUT UNPROTECTED !!!")
                            # Attempt Emergency Close
                            log_warning("Attempting to place emergency market close order...")
                            emergency_close_side = "sell" if entry_side == "long" else "buy"
                            emergency_order = place_market_order(
                                exch=exchange,
                                sym=symbol,
                                side=emergency_close_side,
                                amount=actual_filled_quantity,
                                reduce_only=True,  # Ensure it's a closing order
                            )
                            if emergency_order and emergency_order.get("id"):
                                log_warning(
                                    f"Emergency market close order placed (ID: {emergency_order['id']}). Status: {emergency_order.get('status')}. State NOT updated to active."
                                )
                            else:
                                log_error(
                                    "!!! EMERGENCY CLOSE ORDER FAILED TO PLACE !!! MANUAL INTERVENTION REQUIRED IMMEDIATELY FOR UNPROTECTED POSITION !!!"
                                )
                            # Do NOT update the state to active, as the position should ideally be closed now.

                    else:
                        log_error(
                            f"Entry market order placement failed or did not return a valid result ({entry_order_result}). No position taken."
                        )
                        # Optional: Attempt to cancel the potentially failed/partially filled entry order ID here?
                        # if entry_order_result and entry_order_result.get('id'):
                        #     cancel_order_with_retry(exchange, entry_order_result['id'], symbol)

                else:  # No entry signal met conditions
                    log_info("Entry conditions not met in this cycle.")

            # === 7. Wait for Next Cycle ===
            cycle_end_time = pd.Timestamp.now(tz="UTC")
            elapsed_seconds = (cycle_end_time - cycle_start_time).total_seconds()
            wait_time_seconds = max(0, sleep_interval_seconds - elapsed_seconds)

            log_info(f"Cycle completed in {elapsed_seconds:.2f} seconds.")
            if wait_time_seconds > 0:
                log_info(f"Waiting {wait_time_seconds:.1f} seconds until next cycle...")
                neon_sleep_timer(int(round(wait_time_seconds)))
            else:
                log_info("Cycle took longer than interval, starting next cycle immediately.")

        # --- Graceful Shutdown Handling ---
        except KeyboardInterrupt:
            log_info("Shutdown signal (Ctrl+C) received.")
            log_info("Saving final position state...")
            save_position_state()  # Save state before potentially cancelling orders

            # Attempt to cancel open orders only in live mode
            if not SIMULATION_MODE:
                log_warning(f"Attempting to cancel any open orders for {symbol} on {exchange.name}...")
                try:
                    # Use retry decorator for fetching orders on exit
                    @api_retry_decorator
                    def fetch_open_orders_on_exit(e: ccxt.Exchange, s: str) -> List[Dict]:
                        return e.fetch_open_orders(s)

                    open_orders = fetch_open_orders_on_exit(exchange, symbol)
                    if open_orders:
                        log_info(f"Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...")
                        cancelled_count = 0
                        failed_count = 0
                        for order in open_orders:
                            order_id = order.get("id")
                            order_info = f"ID: {order_id}, Side: {order.get('side')}, Type: {order.get('type')}, Price: {order.get('price')}"
                            if not order_id:
                                log_warning(f"Skipping order with no ID: {order_info}")
                                continue
                            log_info(f"Cancelling order -> {order_info}")
                            if cancel_order_with_retry(exchange, order_id, symbol):
                                cancelled_count += 1
                            else:
                                failed_count += 1
                            time.sleep(0.3)  # Small delay between cancel calls

                        log_info(
                            f"Exit cancellation summary: {cancelled_count} cancelled/closed, {failed_count} failed."
                        )
                        if failed_count > 0:
                            log_error("Some orders failed to cancel. Manual check on the exchange is recommended.")
                    else:
                        log_info("No open orders found for {symbol} to cancel.")
                except Exception as e:
                    log_error(f"An error occurred during open order cancellation on exit: {e}", exc_info=True)
                    log_error("Manual check of open orders on the exchange is strongly recommended.")
            else:
                log_info("Simulation mode: No orders to cancel on exit.")

            break  # Exit the main loop

        # --- Main Loop Error Handling ---
        except (
            ccxt.RateLimitExceeded,
            ccxt.NetworkError,
            ccxt.ExchangeNotAvailable,
            ccxt.RequestTimeout,
            ccxt.DDoSProtection,
        ) as e:
            log_error(
                f"Recoverable CCXT Error encountered: {type(e).__name__}: {e}. Waiting and retrying...", exc_info=False
            )
            # Wait longer after rate limit errors
            wait_time = (
                sleep_interval_seconds + 60 if isinstance(e, ccxt.RateLimitExceeded) else sleep_interval_seconds + 15
            )
            log_warning(f"Waiting for {wait_time} seconds before next cycle due to error.")
            neon_sleep_timer(wait_time)
        except ccxt.AuthenticationError as e:
            log_error(
                f"!!! CRITICAL AUTHENTICATION ERROR: {e} !!! Check API credentials. Stopping bot.", exc_info=False
            )
            save_position_state()  # Attempt to save state before exiting
            break  # Stop the bot on auth errors
        except ccxt.ExchangeError as e:
            log_error(f"Unhandled CCXT Exchange Error: {type(e).__name__}: {e}. Waiting...", exc_info=True)
            neon_sleep_timer(sleep_interval_seconds + 30)  # Wait longer for generic exchange errors
        except Exception as e:
            log_error(f"!!! CRITICAL UNEXPECTED LOOP ERROR: {type(e).__name__}: {e} !!!", exc_info=True)
            log_info("Attempting to save state before waiting...")
            try:
                save_position_state()
            except Exception as save_err:
                log_error(f"Failed to save state during critical error handling: {save_err}")
            log_warning("Waiting for 60 seconds before attempting to continue...")
            neon_sleep_timer(60)

    # --- Bot Shutdown ---
    print_shutdown_message()
    log_info("Bot shutdown sequence complete.")


# --- Script Execution Entry Point ---
if __name__ == "__main__":
    try:
        run_bot()
    except SystemExit as e:
        # Catch sys.exit calls (e.g., from config validation, live mode confirmation)
        log_info(f"Bot exited via SystemExit (Code: {e.code}).")
    except Exception as main_exec_error:
        # Catch any critical errors during initial setup before the main loop starts
        log_error(f"Critical error during bot initialization or top-level execution: {main_exec_error}", exc_info=True)
        sys.exit(1)  # Exit with error code
    finally:
        # Ensure logging resources are released cleanly
        logging.shutdown()
        print("Logging shut down.")
