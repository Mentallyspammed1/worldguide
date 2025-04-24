#!/bin/bash
#--------------------------------------------------------------------------
# Pyrmethus Trading Bot - Setup Script for Termux
#
# Description: Creates the directory structure and necessary Python files
#              for the modular trading bot framework.
# Author:      AI Assistant (Enhanced by User)
# Version:     1.1
#--------------------------------------------------------------------------

# --- Configuration ---
BOT_DIR="$HOME/trading-bot"

# --- Colors for Output ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_CYAN='\033[0;36m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_MAGENTA='\033[0;35m'
COLOR_BRIGHT_YELLOW='\033[1;33m'
COLOR_BRIGHT_GREEN='\033[1;32m'
COLOR_BRIGHT_RED='\033[1;31m'
COLOR_BRIGHT_CYAN='\033[1;36m'

# --- Helper Functions ---
echog() { echo -e "${COLOR_GREEN}$1${COLOR_RESET}"; }
echoy() { echo -e "${COLOR_YELLOW}$1${COLOR_RESET}"; }
echoc() { echo -e "${COLOR_CYAN}$1${COLOR_RESET}"; }
echor() { echo -e "${COLOR_RED}$1${COLOR_RESET}"; }
echom() { echo -e "${COLOR_MAGENTA}$1${COLOR_RESET}"; }
echob() { echo -e "${COLOR_BLUE}$1${COLOR_RESET}"; } # Blue for steps/emphasis
echobright() { echo -e "${COLOR_BRIGHT_YELLOW}$1${COLOR_RESET}"; } # Bright Yellow for important notes

# --- Error Handling Function ---
error_exit() {
    echo -e "\n${COLOR_BRIGHT_RED}ERROR: ${1}${COLOR_RESET}" >&2
    exit "${2:-1}" # Default exit code 1
}

# --- Main Script ---
echom "========================================="
echom "=== Pyrmethus Trading Bot Setup Utility ==="
echom "========================================="
echo # Newline for spacing

# 1. Create Directory Structure
echob "Step 1: Creating directory structure..."
mkdir -p "$BOT_DIR/bot_logs"
if [ $? -ne 0 ]; then
    error_exit "Failed to create directory structure at '$BOT_DIR'. Check permissions in '$HOME'."
fi
echog " -> Directory structure created successfully at $BOT_DIR"

# Change into the bot directory; exit if failed
cd "$BOT_DIR" || error_exit "Failed to change directory to '$BOT_DIR'."
echog " -> Changed working directory to $BOT_DIR"
echo # Newline

# 2. Create config.py
echob "Step 2: Generating config.py..."
cat << 'EOF' > config.py
# ~/trading-bot/config.py
import json
import logging
import os
import math # Added for isnan check
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure pydantic is installed: pip install pydantic
from pydantic import BaseModel, validator, Field, ValidationError

# Optional: Import Colorama for enhanced logging feedback if utils.py uses it
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Define dummy Fore/Style if colorama is not installed
    class DummyStyle:
        def __getattr__(self, name): return ""
    Fore = DummyStyle()
    Style = DummyStyle()

# Define default paths relative to the script location or Termux home
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_LOG_DIR = SCRIPT_DIR / "bot_logs"

# --- Configuration Models (using Pydantic for validation) ---

class StrategyParams(BaseModel):
    """Parameters specific to the trading strategy logic."""
    vt_length: int = Field(40, gt=0, description="Volumatic Trend EMA length")
    vt_atr_period: int = Field(200, gt=0, description="Volumatic Trend ATR period")
    vt_vol_ema_length: int = Field(950, gt=0, description="Volume EMA length for normalization")
    vt_atr_multiplier: float = Field(3.0, gt=0, description="ATR multiplier for trend bands")
    ob_source: str = Field("Wicks", description="Order Block source ('Wicks' or 'Body')")
    ph_left: int = Field(10, gt=0, description="Pivot High left lookback")
    ph_right: int = Field(10, gt=0, description="Pivot High right lookback")
    pl_left: int = Field(10, gt=0, description="Pivot Low left lookback")
    pl_right: int = Field(10, gt=0, description="Pivot Low right lookback")
    ob_extend: bool = Field(True, description="Extend Order Blocks until violated (visual aid, not used in logic)")
    ob_max_boxes: int = Field(30, gt=0, description="Maximum active Order Blocks per side")
    ob_entry_proximity_factor: float = Field(1.003, gt=1.0, description="Factor for entry proximity check (e.g., 1.003 = 0.3% range)")
    ob_exit_proximity_factor: float = Field(1.001, gt=1.0, description="Factor for exit proximity check (e.g., 1.001 = 0.1% range)")

    @validator("ob_source")
    def check_ob_source(cls, v):
        if v not in ["Wicks", "Body"]:
            raise ValueError(f"{Fore.RED}ob_source must be 'Wicks' or 'Body'{Style.RESET_ALL}")
        return v

class ProtectionParams(BaseModel):
    """Parameters for risk management and position protection."""
    enable_trailing_stop: bool = Field(False, description="Enable Trailing Stop Loss (Requires reliable SL fetching/modification)")
    trailing_stop_callback_rate: float = Field(0.005, ge=0, lt=1, description="TSL callback rate (e.g., 0.005 = 0.5%)")
    trailing_stop_activation_percentage: float = Field(0.003, ge=0, lt=1, description="TSL activation profit % (e.g., 0.003 = 0.3%)")
    enable_break_even: bool = Field(False, description="Enable Break Even stop adjustment (Requires reliable SL fetching/modification)")
    break_even_trigger_atr_multiple: float = Field(1.0, gt=0, description="ATR multiple to trigger Break Even")
    break_even_offset_ticks: int = Field(2, ge=0, description="Ticks above/below entry for BE stop")
    initial_stop_loss_atr_multiple: float = Field(1.8, gt=0, description="Initial SL distance in ATR multiples from entry")
    initial_take_profit_atr_multiple: float = Field(0.7, gt=0, description="Initial TP distance in ATR multiples from entry")

class BotConfig(BaseModel):
    """Main bot configuration model."""
    api_key: str = Field("", description="Bybit API Key (Set via ENV VAR 'BYBIT_API_KEY' or here)")
    api_secret: str = Field("", description="Bybit API Secret (Set via ENV VAR 'BYBIT_API_SECRET' or here)")
    trading_pairs: List[str] = Field(["BTC/USDT:USDT"], description="List of trading pairs (Format: SYMBOL/QUOTE:SETTLEMENT)")
    interval: str = Field("5", description="Kline interval (e.g., '1', '5', '15', '60', 'D', 'W')")
    retry_delay: int = Field(6, ge=1, description="Delay in seconds between API retry attempts")
    fetch_limit: int = Field(750, gt=100, le=1000, description="Number of historical klines to fetch initially")
    enable_trading: bool = Field(False, description="Master switch to enable live trading actions")
    use_sandbox: bool = Field(True, description="Use Bybit sandbox/testnet environment")
    risk_per_trade: float = Field(0.01, gt=0, le=0.1, description="Fraction of balance to risk per trade (e.g., 0.01 = 1%)")
    leverage: int = Field(10, gt=0, le=100, description="Leverage to use for positions (Ensure this is set on Bybit too)")
    quote_currency: str = Field("USDT", description="Quote currency for balance checks and calculations")
    loop_delay_seconds: int = Field(15, ge=5, description="Minimum delay between main trading cycles")
    position_confirm_delay_seconds: int = Field(8, ge=1, description="Delay after placing order to confirm position fill status")
    log_level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    cancel_orders_on_exit: bool = Field(False, description="Attempt to cancel open orders on graceful shutdown (Use with caution)")
    strategy_params: StrategyParams = Field(default_factory=StrategyParams)
    protection: ProtectionParams = Field(default_factory=ProtectionParams)

    @validator("interval")
    def check_interval(cls, v):
        # Bybit V5 intervals (adapt if needed for other exchanges or updates)
        valid_intervals = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
        if str(v) not in valid_intervals:
            raise ValueError(f"{Fore.RED}Interval '{v}' must be one of {valid_intervals}{Style.RESET_ALL}")
        return str(v)

    @validator("log_level")
    def check_log_level(cls, v):
        level = v.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"{Fore.RED}Invalid log_level: {v}. Use DEBUG, INFO, WARNING, ERROR, or CRITICAL.{Style.RESET_ALL}")
        return level

    @validator("trading_pairs", pre=True, each_item=True)
    def format_trading_pair(cls, v):
        # Enforces SYMBOL/QUOTE:SETTLEMENT format (e.g., BTC/USDT:USDT)
        if isinstance(v, str):
            parts = v.split(':')
            if len(parts) == 2 and '/' in parts[0]:
                symbol_part, settlement = parts
                if settlement: # Ensure settlement is not empty
                    return v.upper() # Standardize to uppercase
            # Attempt auto-correction if only SYMBOL/QUOTE is provided (assume quote=settlement)
            elif ':' not in v and '/' in v:
                 symbol_part, quote_part = v.split('/')
                 if quote_part:
                     corrected = f"{v.upper()}:{quote_part.upper()}"
                     # Optional: Log a warning about the auto-correction
                     # print(f"{Fore.YELLOW}Warning: Corrected trading pair format from '{v}' to '{corrected}'{Style.RESET_ALL}")
                     return corrected
        raise ValueError(f"{Fore.RED}Invalid trading_pairs format: '{v}'. Use SYMBOL/QUOTE:SETTLEMENT (e.g., BTC/USDT:USDT).{Style.RESET_ALL}")

# --- Loading Function ---

def load_config(file_path: Path = DEFAULT_CONFIG_PATH, logger: Optional[logging.Logger] = None) -> BotConfig:
    """
    Loads configuration from a JSON file, validates using Pydantic,
    applies defaults, and overrides with environment variables if present.
    Creates a default config file if it doesn't exist.
    """
    effective_logger = logger if logger else logging.getLogger("config_loader")
    if not logger: # Basic setup if no logger passed (e.g., during initial load)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    effective_logger.info(f"{Fore.CYAN}# Summoning configuration runes from {file_path}...{Style.RESET_ALL}")

    config_data = {}
    config_exists = file_path.exists()

    if config_exists:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
            effective_logger.debug(f"Successfully loaded raw config data from {file_path}")
        except json.JSONDecodeError as e:
            effective_logger.error(f"{Fore.RED}Error decoding JSON from {file_path}: {e}. Using defaults/ENV vars.{Style.RESET_ALL}")
            config_exists = False # Treat as non-existent if decode fails
        except Exception as e:
            effective_logger.error(f"{Fore.RED}Failed to read config file {file_path}: {e}. Using defaults/ENV vars.{Style.RESET_ALL}")
            config_exists = False # Treat as non-existent if read fails
    else:
        effective_logger.warning(f"{Fore.YELLOW}Configuration file not found at {file_path}. Will create default.{Style.RESET_ALL}")

    # --- Override with Environment Variables ---
    # Check for API keys first
    api_key_env = os.getenv("BYBIT_API_KEY")
    api_secret_env = os.getenv("BYBIT_API_SECRET")
    if api_key_env:
        config_data['api_key'] = api_key_env
        effective_logger.info(f"{Fore.YELLOW}Using API Key from environment variable (BYBIT_API_KEY).{Style.RESET_ALL}")
    if api_secret_env:
        config_data['api_secret'] = api_secret_env
        effective_logger.info(f"{Fore.YELLOW}Using API Secret from environment variable (BYBIT_API_SECRET).{Style.RESET_ALL}")

    # Example for overriding another setting (optional)
    # log_level_env = os.getenv("BOT_LOG_LEVEL")
    # if log_level_env:
    #     config_data['log_level'] = log_level_env
    #     effective_logger.info(f"{Fore.YELLOW}Using Log Level '{log_level_env}' from environment variable (BOT_LOG_LEVEL).{Style.RESET_ALL}")

    # --- Validate and Create Final Config Object ---
    try:
        # Use the loaded data (potentially overridden by ENV vars) to create the Pydantic model
        config = BotConfig(**config_data)
        effective_logger.info(f"{Fore.GREEN}Configuration runes validated successfully.{Style.RESET_ALL}")

    except ValidationError as e:
        effective_logger.error(f"{Fore.RED}Configuration validation failed!{Style.RESET_ALL}\n{e}")
        effective_logger.warning(f"{Fore.YELLOW}Falling back to default configuration due to validation errors.{Style.RESET_ALL}")
        # Create a default instance on validation error
        config = BotConfig()
        # Ensure API keys are cleared if validation failed with potentially bad user input
        config.api_key = api_key_env or ""
        config.api_secret = api_secret_env or ""

    # --- Create Default Config File if Needed ---
    # Create if it didn't exist OR if validation failed (to provide a valid template)
    if not config_exists or isinstance(config, BotConfig) and 'ValidationError' in str(locals().get('e', '')):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with file_path.open("w", encoding="utf-8") as f:
                # Get dictionary representation from the *current* config object (default or partially loaded)
                write_data = config.dict()
                # Explicitly set placeholder text for secrets in the generated file
                write_data['api_key'] = "YOUR_API_KEY_HERE_OR_SET_ENV"
                write_data['api_secret'] = "YOUR_API_SECRET_HERE_OR_SET_ENV"
                json.dump(write_data, f, indent=4, ensure_ascii=False)
            effective_logger.warning(f"{Fore.YELLOW}Created/Replaced default config file at {file_path}. "
                                     f"Please review and set API keys!{Style.RESET_ALL}")
        except Exception as e:
             effective_logger.error(f"{Fore.RED}Failed to write default config file {file_path}: {e}{Style.RESET_ALL}")

    # --- Final Check for API Keys ---
    # Check the *final* config object after all loading/defaults/ENV vars
    if not config.api_key or not config.api_secret or \
       "YOUR_API_KEY" in config.api_key or "YOUR_API_SECRET" in config.api_secret:
         effective_logger.warning(
             f"{Fore.YELLOW}API Key or Secret is missing or uses placeholder in the final configuration. "
             f"Trading actions will likely fail. Set via ENV variables or in {file_path}.{Style.RESET_ALL}"
         )

    return config

# Example of usage (typically called from main.py)
if __name__ == "__main__":
    # This block is for testing the load_config function directly
    logging.basicConfig(level=logging.DEBUG)
    test_logger = logging.getLogger("ConfigTest")
    loaded_config = load_config(logger=test_logger)
    print("\n--- Loaded Config ---")
    if loaded_config:
        # Exclude secrets when printing
        print(loaded_config.dict(exclude={'api_key', 'api_secret'}))
        print(f"\nAPI Key Loaded: {'Yes' if loaded_config.api_key and 'YOUR_API_KEY' not in loaded_config.api_key else 'NO (or placeholder)'}")
        print(f"API Secret Loaded: {'Yes' if loaded_config.api_secret and 'YOUR_API_SECRET' not in loaded_config.api_secret else 'NO (or placeholder)'}")
    else:
        print("Config loading failed.")
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate config.py."; fi
echog " -> config.py generated successfully."
echo # Newline

# 3. Create utils.py
echob "Step 3: Generating utils.py..."
cat << 'EOF' > utils.py
# ~/trading-bot/utils.py
import logging
from logging.handlers import RotatingFileHandler
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, Context, setcontext, ROUND_DOWN
import math # Ensure math is imported for isfinite
import sys
import os
from pathlib import Path
from typing import Any, Optional, Dict, Tuple

# Attempt to import Colorama for colored console output
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
    # Define color constants for consistent mystical flair
    RESET = Style.RESET_ALL
    BRIGHT = Style.BRIGHT
    DIM = Style.DIM
    FG_BLACK = Fore.BLACK
    FG_RED = Fore.RED
    FG_GREEN = Fore.GREEN
    FG_YELLOW = Fore.YELLOW
    FG_BLUE = Fore.BLUE
    FG_MAGENTA = Fore.MAGENTA
    FG_CYAN = Fore.CYAN
    FG_WHITE = Fore.WHITE
    BG_RED = Back.RED
    BG_GREEN = Back.GREEN
    BG_YELLOW = Back.YELLOW
except ImportError:
    COLORAMA_AVAILABLE = False
    # Define dummy classes/variables if colorama is not installed
    class DummyStyle:
        def __getattr__(self, name): return ""
    Fore = DummyStyle()
    Style = DummyStyle()
    Back = DummyStyle()
    RESET = ""
    BRIGHT = ""
    DIM = ""
    FG_BLACK = ""
    FG_RED = ""
    FG_GREEN = ""
    FG_YELLOW = ""
    FG_BLUE = ""
    FG_MAGENTA = ""
    FG_CYAN = ""
    FG_WHITE = ""
    BG_RED = ""
    BG_GREEN = ""
    BG_YELLOW = ""
    print("Warning: Colorama not found. Console output will lack colors.", file=sys.stderr)


# Set Decimal context globally if desired (adjust precision as needed)
# Be mindful of potential side effects if other libraries rely on default context.
# context = Context(prec=30, rounding=ROUND_HALF_UP)
# setcontext(context)

# Determine default log directory relative to this utils.py file
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_LOG_DIR = SCRIPT_DIR / "bot_logs"

# --- Custom Color Log Formatter ---
class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds colors to console output if Colorama is available."""

    LEVEL_COLORS = {
        logging.DEBUG: FG_CYAN + DIM,
        logging.INFO: FG_GREEN,
        logging.WARNING: FG_YELLOW,
        logging.ERROR: FG_RED,
        logging.CRITICAL: BG_RED + FG_WHITE + BRIGHT,
    } if COLORAMA_AVAILABLE else {}

    def format(self, record):
        # Base format string without colors
        log_fmt = f"[%(asctime)s] %(levelname)-8s ({FG_BLUE}{record.name}{RESET})%(loc)s %(message)s"
        date_fmt = "%H:%M:%S"

        # Add location info only for errors/critical or debug level
        loc = ""
        if record.levelno >= logging.ERROR or record.levelno == logging.DEBUG:
             loc = f" ({FG_MAGENTA}{record.filename}:{record.lineno}{RESET})"

        # Apply level-specific color if available
        color = self.LEVEL_COLORS.get(record.levelno, RESET) if COLORAMA_AVAILABLE else ""

        # Create a temporary formatter with the base format
        # Note: record.message contains the already formatted message string
        # We need to format the *entire* line structure.
        formatter = logging.Formatter(log_fmt.replace('%(loc)s', loc), datefmt=date_fmt)

        # Let the base Formatter handle the standard formatting
        formatted_line = formatter.format(record)

        # Prepend the color to the entire line
        return f"{color}{formatted_line}{RESET}"


def setup_logger(name: str, level: str = "INFO", log_dir: Path = DEFAULT_LOG_DIR) -> logging.Logger:
    """
    Configures a logger with both rotating file output and colored console output.
    Uses a hierarchical naming scheme (e.g., pyrmethus.exchange).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(f"pyrmethus.{name}") # Use a hierarchical name

    # Prevent adding multiple handlers if logger was already configured (e.g., by another module)
    if logger.hasHandlers():
        # Update level if necessary, but don't add handlers again
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)
        # logger.debug(f"Logger '{name}' already configured. Level set to {level}.")
        return logger

    logger.setLevel(log_level)
    logger.propagate = False # Prevent double logging by the root logger

    try:
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize logger name for filename (replace dots)
        safe_name = name.replace('.', '_')
        log_file = log_dir / f"{safe_name}.log"

        # --- File Handler (Rotating) ---
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] (%(name)s) %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Rotate log file when it reaches 10 MB, keep 5 backups
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level) # File logs at the configured level

        # --- Console Handler (Colored) ---
        console_formatter = ColorFormatter() # Uses custom formatter defined above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        # Console can optionally have a different level (e.g., INFO while file is DEBUG)
        # For simplicity, keep them the same by default.
        console_handler.setLevel(log_level)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"{FG_GREEN}Logger '{name}' conjured. Level: {level}. Output: {log_file}{RESET}")

    except Exception as e:
         # Fallback to basic logging if handler setup fails
         logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
         # Use a distinct name for the fallback logger to avoid conflicts
         logger = logging.getLogger(f"pyrmethus.{name}_fallback")
         logger.error(f"{FG_RED}Failed to set up custom logging for '{name}': {e}. Using basic config.{RESET}")

    return logger


def safe_decimal(value: Any, field_name: str = "value", precision: Optional[int] = None, allow_zero: bool = True, logger: Optional[logging.Logger] = None, context: Optional[Context] = None) -> Optional[Decimal]:
    """
    Safely converts a value to a Decimal object, handling potential errors,
    NaNs, and optionally applying precision quantization.

    Args:
        value: The value to convert (can be string, float, int, etc.).
        field_name: Descriptive name of the value being converted (for logging).
        precision: Number of decimal places to quantize to (optional).
        allow_zero: Whether a Decimal value of zero is considered valid.
        logger: Logger instance for warnings/errors.
        context: Optional Decimal context for conversion.

    Returns:
        A Decimal object if conversion is successful and valid, otherwise None.
    """
    if value is None or value == '':
        # Log quietly at debug level if needed, often expected
        # if logger: logger.debug(f"Input for '{field_name}' is None or empty, returning None.")
        return None

    # Check for NaN/Infinity specifically for floats before converting to string
    if isinstance(value, float) and not math.isfinite(value):
        if logger: logger.debug(f"{FG_YELLOW}Input for '{field_name}' is non-finite float ({value}), returning None.{RESET}")
        return None

    try:
        # Convert to string first to avoid potential float representation issues
        str_value = str(value).strip()
        if not str_value: # Check again after stripping whitespace
             return None

        # Use provided context or the default context for the conversion
        dec_value = Decimal(str_value, context=context)

        # Check if zero is allowed
        if not allow_zero and dec_value.is_zero():
            if logger: logger.debug(f"{FG_YELLOW}Zero value not allowed for '{field_name}'. Input was '{value}'. Returning None.{RESET}")
            return None

        # Quantize if precision is specified and valid
        if precision is not None:
            if not isinstance(precision, int) or precision < 0:
                 if logger: logger.warning(f"{FG_YELLOW}Invalid precision '{precision}' requested for '{field_name}'. Ignoring quantization.{RESET}")
            else:
                # Create quantizer string like '1e-8' for Decimal quantization
                quantizer_str = f"1e-{precision}"
                quantizer = Decimal(quantizer_str)
                # Use quantize method with standard rounding (ROUND_HALF_UP)
                dec_value = dec_value.quantize(quantizer, rounding=ROUND_HALF_UP, context=context)

        return dec_value

    except InvalidOperation:
        # This catches errors during the Decimal(str_value) conversion
        if logger: logger.error(f"{FG_RED}Failed to cast '{value}' (type: {type(value)}) to Decimal for '{field_name}'. Invalid operation.{RESET}")
        return None
    except Exception as e:
        # Catch other potential errors during conversion or quantization
        if logger: logger.error(f"{FG_RED}Unexpected error casting '{value}' to Decimal for '{field_name}': {e}{RESET}")
        return None


def get_market_precision(market_info: Dict) -> Tuple[int, int]:
    """
    Extracts price and amount precision (number of decimal places) from CCXT market info.
    Handles various CCXT formats (integer decimal places, float/string tick sizes).

    Args:
        market_info: The market dictionary from ccxt.load_markets().

    Returns:
        A tuple (price_precision, amount_precision) as integers.
        Defaults to (8, 8) if extraction fails.
    """
    default_precision = 8 # A sensible default if precision cannot be determined

    def parse_ccxt_precision(precision_value: Any) -> int:
        """Helper function to parse a single precision value from CCXT."""
        if precision_value is None:
            return default_precision
        try:
            # Convert to Decimal for robust handling
            p_val = Decimal(str(precision_value))

            if p_val.is_zero(): # Avoid log10(0) if tick size is 0 (unlikely but possible)
                return 0 # Zero precision means integer only

            # Precision modes:
            # 1. Integer >= 1: Usually means number of decimal places (legacy or explicit mode)
            if p_val >= Decimal(1) and p_val.as_tuple().exponent == 0:
                return int(p_val)
            # 2. Float/Decimal > 0 and < 1: Usually means tick size (e.g., 0.001)
            elif Decimal(0) < p_val < Decimal(1):
                # Calculate decimal places from tick size: -log10(tick_size)
                # Ensure the result is rounded correctly to handle potential float issues
                decimal_places = -p_val.log10().to_integral_value(rounding=ROUND_HALF_UP)
                return int(decimal_places)
            # 3. Other cases (e.g., float >= 1 but not integer, negative) are ambiguous
            else:
                 # Log warning about ambiguous value?
                 # print(f"Warning: Ambiguous precision value encountered: {precision_value}")
                 return default_precision
        except (InvalidOperation, ValueError, TypeError, OverflowError):
            # Fallback on any parsing or conversion error
            return default_precision

    try:
        if not market_info or 'precision' not in market_info:
            # print(f"Warning: Market info or precision data missing. Market: {market_info}")
            return default_precision, default_precision

        precision_data = market_info['precision']

        # Prefer 'price' and 'amount' keys as they are more standard
        price_precision_val = precision_data.get('price')
        amount_precision_val = precision_data.get('amount')

        # Fallback to 'quote' and 'base' if 'price'/'amount' are missing
        if price_precision_val is None:
            price_precision_val = precision_data.get('quote')
        if amount_precision_val is None:
            amount_precision_val = precision_data.get('base')

        price_precision = parse_ccxt_precision(price_precision_val)
        amount_precision = parse_ccxt_precision(amount_precision_val)

        return int(price_precision), int(amount_precision)

    except Exception as e:
        # Catch-all for unexpected structure or errors during parsing
        # print(f"Error extracting precision: {e}. Market Info: {market_info}") # Debugging
        return default_precision, default_precision

def format_value(value: Decimal, precision: int, rounding=ROUND_HALF_UP) -> str:
    """
    Formats a Decimal value to a string with a fixed number of decimal places.

    Args:
        value: The Decimal value to format.
        precision: The number of decimal places required.
        rounding: The rounding method to use (e.g., ROUND_HALF_UP, ROUND_DOWN).

    Returns:
        A string representation of the value formatted to the specified precision.
        Returns "InvalidDecimal" if the input cannot be converted to Decimal.
    """
    if not isinstance(value, Decimal):
        try:
            # Attempt conversion if not already Decimal
            value = Decimal(str(value))
        except InvalidOperation:
            # Log this? Depends on context.
            # print(f"Error: Could not convert '{value}' to Decimal for formatting.")
            return "InvalidDecimal" # Indicate error

    if not isinstance(precision, int) or precision < 0:
        precision = 0 # Ensure non-negative integer precision

    # Create the quantizer Decimal (e.g., '0.01' for precision 2)
    quantizer = Decimal('1e-' + str(precision))

    # Quantize the value using the specified rounding method
    quantized_value = value.quantize(quantizer, rounding=rounding)

    # Format the quantized value to ensure the correct number of trailing zeros
    # Use '.f' format specifier for fixed-point notation.
    format_string = "{:.%df}" % precision
    return format_string.format(quantized_value)

# Example usage (for testing)
if __name__ == "__main__":
    test_logger = setup_logger("UtilsTest", level="DEBUG")
    test_logger.debug("Debug message test.")
    test_logger.info("Info message test.")
    test_logger.warning("Warning message test.")
    test_logger.error("Error message test.")
    test_logger.critical("Critical message test.")

    print("\n--- Safe Decimal Tests ---")
    print(f"None -> {safe_decimal(None, 'test_none', logger=test_logger)}")
    print(f"Empty string -> {safe_decimal('', 'test_empty', logger=test_logger)}")
    print(f"NaN float -> {safe_decimal(float('nan'), 'test_nan', logger=test_logger)}")
    print(f"Infinity float -> {safe_decimal(float('inf'), 'test_inf', logger=test_logger)}")
    print(f"String '123.456' -> {safe_decimal('123.456', 'test_str', precision=2, logger=test_logger)}")
    print(f"Float 0.1 + 0.2 -> {safe_decimal(0.1 + 0.2, 'test_float_sum', precision=15, logger=test_logger)}")
    print(f"Integer 100 -> {safe_decimal(100, 'test_int', precision=0, logger=test_logger)}")
    print(f"String ' 789.1 ' -> {safe_decimal(' 789.1 ', 'test_whitespace', precision=3, logger=test_logger)}")
    print(f"String 'abc' -> {safe_decimal('abc', 'test_invalid_str', logger=test_logger)}")
    print(f"Zero (allowed) -> {safe_decimal(0, 'test_zero_allowed', precision=2, logger=test_logger)}")
    print(f"Zero (not allowed) -> {safe_decimal(0, 'test_zero_not_allowed', allow_zero=False, logger=test_logger)}")
    print(f"Small number (quantized) -> {safe_decimal('0.0000001234', 'test_small', precision=8, logger=test_logger)}")
    print(f"Negative number -> {safe_decimal('-50.5', 'test_neg', precision=1, logger=test_logger)}")

    print("\n--- Format Value Tests ---")
    d = Decimal('123.456789')
    print(f"{d} (prec=2) -> {format_value(d, 2)}")
    print(f"{d} (prec=8) -> {format_value(d, 8)}")
    print(f"{d} (prec=0) -> {format_value(d, 0)}")
    d2 = Decimal('123')
    print(f"{d2} (prec=4) -> {format_value(d2, 4)}")
    d3 = Decimal('0.12')
    print(f"{d3} (prec=5, ROUND_DOWN) -> {format_value(d3, 5, rounding=ROUND_DOWN)}")
    print(f"String '99.9' (prec=3) -> {format_value('99.9', 3)}")
    print(f"Float 0.1 (prec=5) -> {format_value(0.1, 5)}") # Test non-Decimal input
    print(f"Invalid 'xyz' (prec=2) -> {format_value('xyz', 2)}")

    print("\n--- Market Precision Tests ---")
    market1 = {'precision': {'price': 2, 'amount': 4}} # Decimal places
    market2 = {'precision': {'price': '0.01', 'amount': '0.0001'}} # Tick sizes
    market3 = {'precision': {'price': '0.5', 'amount': 1}} # Mixed / unusual
    market4 = {'precision': {'quote': 8, 'base': 6}} # Base/Quote keys
    market5 = {'precision': {}} # Empty precision
    market6 = {} # No precision key
    market7 = {'precision': {'price': None, 'amount': None}} # None values
    market8 = {'precision': {'price': '0.00000001', 'amount': '10'}} # Large amount tick (means integer?)

    print(f"Market1 {market1}: {get_market_precision(market1)}")
    print(f"Market2 {market2}: {get_market_precision(market2)}")
    print(f"Market3 {market3}: {get_market_precision(market3)}")
    print(f"Market4 {market4}: {get_market_precision(market4)}")
    print(f"Market5 {market5}: {get_market_precision(market5)}")
    print(f"Market6 {market6}: {get_market_precision(market6)}")
    print(f"Market7 {market7}: {get_market_precision(market7)}")
    print(f"Market8 {market8}: {get_market_precision(market8)}")
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate utils.py."; fi
echog " -> utils.py generated successfully."
echo # Newline

# 4. Create exchange.py
echob "Step 4: Generating exchange.py..."
cat << 'EOF' > exchange.py
# ~/trading-bot/exchange.py
import ccxt
import pandas as pd
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, Context
import time
import logging
from typing import Optional, Dict, List, Tuple, Any

# Ensure ccxt and pandas are installed: pip install ccxt pandas
from .utils import ( setup_logger, safe_decimal, get_market_precision, format_value,
                     FG_RED, FG_GREEN, FG_YELLOW, FG_CYAN, RESET, BG_RED, FG_WHITE, FG_BLUE, BRIGHT )
from .config import BotConfig # For type hinting

# Optional: Define a consistent Decimal context for CCXT interactions if needed
# decimal_context = Context(prec=28) # Adjust precision as required

class ExchangeManager:
    """
    Handles communication with the Bybit exchange (V5 API) via CCXT.
    Includes features like rate limit handling, retries, data parsing,
    order placement with attached SL/TP, and precision management.
    """

    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exchange = self._initialize_exchange()
        self.markets_cache: Dict[str, Dict] = {}
        self.precision_cache: Dict[str, Tuple[int, int]] = {} # Cache: symbol -> (price_prec, amount_prec)
        self.contract_size_cache: Dict[str, Decimal] = {} # Cache: symbol -> contract_size
        self.load_markets() # Load markets immediately on initialization

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes the CCXT Bybit instance with V5 API settings and error handling."""
        self.logger.info(f"{FG_CYAN}# Conjuring connection to Bybit ({'Sandbox' if self.config.use_sandbox else 'Live'})...{RESET}")
        try:
            exchange_params = {
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True, # Crucial for respecting API limits
                'options': {
                    # --- Bybit V5 Specific Options ---
                    'defaultType': 'linear', # Or 'inverse', 'spot'. Affects default category for some calls.
                    # Explicitly use V5 API (though ccxt often defaults to latest)
                    'api-version': 'v5',
                    # Specify Unified account type if applicable (recommended for V5)
                    # 'account': 'unified', # Might be handled by defaultType or category params now
                    # Set broker ID for tracking (optional, replace with your identifier)
                    'brokerId': 'PyrmethusV1',
                    # ---------------------------------
                    'adjustForTimeDifference': True, # Helps with timestamp/signature errors
                }
            }
            exchange = ccxt.bybit(exchange_params)

            # Explicitly set sandbox mode AFTER initialization
            exchange.set_sandbox_mode(self.config.use_sandbox)

            # Test connection by fetching server time (a lightweight call)
            server_time_ms = self._retry_api_call(exchange.fetch_time)
            server_time = pd.to_datetime(server_time_ms, unit='ms', utc=True)
            self.logger.info(f"{FG_GREEN}Exchange connection established. Server time: {server_time}{RESET}")
            return exchange

        except ccxt.AuthenticationError as e:
            self.logger.critical(f"{BG_RED}{FG_WHITE}Authentication Error: Failed to connect to Bybit. Check API keys. Details: {e}{RESET}")
            raise # Critical error, bot cannot proceed without valid keys
        except ccxt.NetworkError as e:
             self.logger.critical(f"{FG_RED}Network Error initializing exchange: {e}. Check internet connectivity and Bybit status.{RESET}")
             raise # Critical if connection fails initially
        except Exception as e:
            # Catch any other unexpected error during initialization
            self.logger.critical(f"{FG_RED}Unexpected error initializing Bybit exchange: {e}{RESET}", exc_info=True)
            raise

    def load_markets(self, force_reload: bool = False):
        """
        Loads or reloads market data from the exchange.
        Focuses on the 'linear' category for USDT-margined contracts by default.
        Caches market info, precision, and contract size.
        """
        if not self.markets_cache or force_reload:
            self.logger.info(f"{FG_CYAN}# Fetching market runes (category: linear)...{RESET}")
            try:
                # Bybit V5: Specify 'category' = 'linear' for USDT perpetuals/futures
                # Use reload=True if force_reload is requested
                loaded_markets = self._retry_api_call(self.exchange.load_markets, reload=force_reload, params={'category': 'linear'})

                if not loaded_markets:
                     # This case might indicate an issue with the API or the category itself
                     self.logger.error(f"{FG_RED}Failed to load markets: Received empty market data for 'linear' category.{RESET}")
                     # Consider trying 'spot' or other categories if needed, or raise an error.
                     # spot_markets = self._retry_api_call(self.exchange.load_markets, reload=force_reload, params={'category': 'spot'})
                     # if spot_markets: self.markets_cache.update(spot_markets) # Merge if needed
                     # else: raise ccxt.ExchangeError("Received empty market data for all attempted categories")
                     raise ccxt.ExchangeError("Received empty market data for 'linear' category. Check Bybit status or symbol category.")

                self.markets_cache = loaded_markets
                self.precision_cache.clear() # Clear caches before repopulating
                self.contract_size_cache.clear()

                # Pre-cache precision and contract size for configured trading pairs
                missing_pairs_in_linear = []
                for pair_info in self.config.trading_pairs:
                    # Format is SYMBOL/QUOTE:SETTLEMENT
                    symbol = pair_info.split(':')[0] # Get symbol like BTC/USDT
                    if symbol in self.markets_cache:
                        # Populate cache by calling the getter methods
                        self.get_precision(symbol)
                        self.get_contract_size(symbol)
                    else:
                        missing_pairs_in_linear.append(symbol)

                if missing_pairs_in_linear:
                    self.logger.warning(
                        f"{FG_YELLOW}Configured pairs not found in loaded 'linear' markets: "
                        f"{', '.join(missing_pairs_in_linear)}. Ensure they are valid Linear Perpetual/Futures symbols on Bybit.{RESET}"
                    )
                    # Optional: Attempt to load 'spot' markets if pairs might be spot
                    # self.logger.info("Attempting to load 'spot' markets for missing pairs...")
                    # try:
                    #     spot_markets = self._retry_api_call(self.exchange.load_markets, reload=force_reload, params={'category': 'spot'})
                    #     self.markets_cache.update(spot_markets) # Merge spot markets into cache
                    #     # Re-check missing pairs against the updated cache...
                    # except Exception as spot_e:
                    #     self.logger.error(f"Failed to load 'spot' markets: {spot_e}")

                self.logger.info(f"{FG_GREEN}Loaded and processed {len(self.markets_cache)} market runes (primarily 'linear').{RESET}")

            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                self.logger.error(f"{FG_RED}Failed to load markets due to exchange/network error: {e}{RESET}")
                # Depending on bot logic, might retry later or raise if critical at this stage
                raise # Often critical for bot operation, re-raise the error
            except Exception as e:
                self.logger.error(f"{FG_RED}Unexpected error loading markets: {e}{RESET}", exc_info=True)
                raise # Re-raise unexpected errors

    def get_market_info(self, symbol: str) -> Optional[Dict]:
        """Retrieves cached market info for a symbol, attempting reload if not found."""
        market = self.markets_cache.get(symbol)
        if not market:
            self.logger.warning(f"{FG_YELLOW}Market info for '{symbol}' not found in cache. Attempting reload.{RESET}")
            try:
                self.load_markets(force_reload=True) # Attempt to refresh market data
                market = self.markets_cache.get(symbol) # Check again after reload
                if not market:
                    self.logger.error(f"{FG_RED}Market info for '{symbol}' still not found after reload. Check symbol validity and category.{RESET}")
                    return None
            except Exception as e:
                 self.logger.error(f"{FG_RED}Failed to reload markets while getting info for '{symbol}': {e}{RESET}")
                 return None
        return market

    def get_precision(self, symbol: str) -> Tuple[int, int]:
        """Gets cached price and amount precision (decimal places) for a symbol."""
        if symbol in self.precision_cache:
            return self.precision_cache[symbol]

        market_info = self.get_market_info(symbol)
        if market_info:
            precision = get_market_precision(market_info) # Use utility function
            self.precision_cache[symbol] = precision
            self.logger.debug(f"Cached precision for {symbol}: Price={precision[0]}, Amount={precision[1]}")
            return precision
        else:
            # Market info unavailable, return default but log warning
            self.logger.warning(f"{FG_YELLOW}Cannot get precision for '{symbol}', market info unavailable. Using defaults (8, 8).{RESET}")
            return (8, 8) # Fallback precision

    def get_contract_size(self, symbol: str) -> Decimal:
        """
        Gets cached contract size for a symbol (important for futures value calculations).
        Defaults to Decimal(1) if not applicable (spot) or not found.
        """
        if symbol in self.contract_size_cache:
            return self.contract_size_cache[symbol]

        market_info = self.get_market_info(symbol)
        contract_size = Decimal(1) # Default for spot or if info missing/invalid

        if market_info:
            # CCXT standard field is 'contractSize'. Also check 'info' for exchange-specific details.
            # Bybit V5 linear: 'contractSize' should be present, often '1' for USDT pairs.
            raw_contract_size = market_info.get('contractSize')
            if raw_contract_size is None:
                 # Look inside 'info' as a fallback (structure might vary)
                 raw_contract_size = market_info.get('info', {}).get('contractSize')

            if raw_contract_size is not None:
                 cs = safe_decimal(raw_contract_size, f"{symbol} contract size", logger=self.logger)
                 # Ensure contract size is positive, otherwise default to 1
                 if cs is not None and cs > 0:
                     contract_size = cs
                 else:
                     self.logger.warning(f"{FG_YELLOW}Invalid or non-positive contract size '{raw_contract_size}' found for {symbol}. Defaulting to 1.{RESET}")
                     contract_size = Decimal(1)
            # else: contract size not found, using default Decimal(1)

        self.contract_size_cache[symbol] = contract_size
        self.logger.debug(f"Cached contract size for {symbol}: {contract_size}")
        return contract_size

    def _retry_api_call(self, func, *args, **kwargs):
        """
        Wrapper for CCXT API calls with retry logic for common transient errors.
        Handles RateLimitExceeded, NetworkError, RequestTimeout, DDoSProtection,
        and specific retryable Bybit V5 ExchangeErrors.
        """
        max_retries = 4 # Total attempts = 1 initial + max_retries
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                # Extract retry_after time if provided by CCXT or headers
                wait_time_sec = self.config.retry_delay # Default wait
                try:
                    # CCXT often puts retry-after in ms in the exception object
                    if hasattr(e, 'retry_after') and e.retry_after:
                        wait_time_sec = max(1, int(e.retry_after / 1000))
                    # Sometimes it's in the error message headers (less reliable parsing)
                    elif isinstance(e.args[0], str) and 'Retry-After' in e.args[0]:
                         header_val = e.args[0].split('Retry-After":')[1].split('}')[0].strip()
                         wait_time_sec = max(1, int(header_val))
                except Exception:
                    pass # Stick to default if parsing fails

                wait_time = max(wait_time_sec, 1) # Ensure minimum 1s wait
                if attempt < max_retries:
                    self.logger.warning(f"{FG_YELLOW}Rate limit exceeded for {func.__name__}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries+1}){RESET}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"{FG_RED}Rate limit exceeded for {func.__name__} after {max_retries+1} attempts. Giving up.{RESET}")
                    raise e # Re-raise after final attempt

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
                last_exception = e
                if attempt < max_retries:
                    self.logger.warning(f"{FG_YELLOW}Network/Exchange unavailable error for {func.__name__}: {e}. Retrying in {self.config.retry_delay}s... (Attempt {attempt+1}/{max_retries+1}){RESET}")
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"{FG_RED}Network/Exchange error for {func.__name__} after {max_retries+1} attempts: {e}. Giving up.{RESET}")
                    raise e # Re-raise after final attempt

            except ccxt.AuthenticationError as e:
                 last_exception = e
                 self.logger.error(f"{BG_RED}{FG_WHITE}Authentication Error during API call {func.__name__}: {e}. Check API keys immediately.{RESET}")
                 raise # Don't retry authentication errors

            except ccxt.ExchangeError as e:
                last_exception = e
                # --- Bybit V5 Specific Retryable Error Codes ---
                # (Refer to Bybit API documentation for the most current list)
                # 10001: Parameter error (usually non-retryable, but log it)
                # 10002: Request expired (potentially retryable if time sync issue)
                # 10004: Sign error (non-retryable, check API keys/secret generation)
                # 10005: Permission denied (non-retryable, check API key permissions)
                # 10006: System busy / Concurrency limit / Service Unavailable (Retryable)
                # 10010: Request frequency too high (Similar to RateLimitExceeded, retryable)
                # 10016: Internal service error (Retryable)
                # 10017: Request not supported (Non-retryable)
                # 10018: Request timeout (Retryable)
                # 10020: Resource limit exceeded (Retryable)
                # 10029: Websocket connection issue (Context-dependent, might retry)
                # 110045: System is upgrading (Retryable after delay)
                # ... other codes related to specific actions (e.g., order placement, funding)

                retryable_bybit_codes = [10002, 10006, 10010, 10016, 10018, 10020, 110045]
                is_retryable = False
                bybit_code = None

                # Extract Bybit 'retCode' from the exception details
                try:
                    # CCXT often stores the raw response info in e.args[0] or e.args[1]
                    error_details = str(e)
                    if 'retCode' in error_details:
                         # Basic parsing, might need refinement based on exact error string format
                         code_str = error_details.split('"retCode":')[1].split(',')[0].strip()
                         if code_str.isdigit():
                             bybit_code = int(code_str)
                             if bybit_code in retryable_bybit_codes:
                                 is_retryable = True
                except Exception:
                    pass # Ignore parsing errors if retCode cannot be extracted

                if is_retryable and attempt < max_retries:
                    self.logger.warning(f"{FG_YELLOW}Retryable Bybit Exchange Error for {func.__name__} (Code: {bybit_code}): {e}. Retrying... (Attempt {attempt+1}/{max_retries+1}){RESET}")
                    time.sleep(self.config.retry_delay)
                else:
                    # Log non-retryable or final attempt error clearly
                    retry_status = "Non-retryable" if not is_retryable else "Final attempt failed"
                    self.logger.error(f"{FG_RED}{retry_status} Bybit Exchange Error for {func.__name__} (Code: {bybit_code}): {e}{RESET}")
                    raise e # Re-raise the error

            except Exception as e:
                # Catch any other unexpected CCXT errors or general exceptions
                last_exception = e
                self.logger.error(f"{FG_RED}Unexpected error during API call {func.__name__}: {e}{RESET}", exc_info=True)
                raise # Re-throw unexpected errors immediately

        # Should not be reached if loop completes, but as a safeguard:
        self.logger.error(f"{FG_RED}Function {func.__name__} failed after {max_retries+1} attempts without specific exception caught in final raise.{RESET}")
        if last_exception:
             raise last_exception # Re-raise the last known exception
        else:
             # Fallback if no exception was stored (highly unlikely)
             raise ccxt.ExchangeError(f"{func.__name__} failed after multiple retries")


    def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetches OHLCV kline data with retries and converts to a pandas DataFrame."""
        self.logger.debug(f"Fetching {limit} klines for {symbol} ({timeframe})...")
        try:
            # Bybit V5 API: Use 'category' = 'linear' in params for USDT contracts
            params = {'category': 'linear'} # Adjust if using inverse or spot
            ohlcv_list = self._retry_api_call(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit, params=params)

            if not ohlcv_list:
                self.logger.warning(f"{FG_YELLOW}Received empty kline data for {symbol} ({timeframe}). Possible reasons: incorrect symbol/timeframe, no data available, or API issue.{RESET}")
                return None

            # Create DataFrame
            df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # --- Data Cleaning and Type Conversion ---
            # Convert timestamp to datetime (UTC) and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            # Convert OHLCV columns to Decimal using safe_decimal for robustness
            price_prec, _ = self.get_precision(symbol)
            # Volume precision might not be explicitly defined by market, use a reasonable default (e.g., 6)
            vol_prec = 6 # Adjust if needed based on typical volumes for the asset

            for col in ['open', 'high', 'low', 'close']:
                # Apply safe_decimal, ensuring necessary precision for price
                df[col] = df[col].apply(lambda x: safe_decimal(x, col, precision=price_prec, allow_zero=False, logger=self.logger))
            # Volume can be zero, allow it. Ensure non-negative.
            df['volume'] = df[col].apply(lambda x: safe_decimal(x, 'volume', precision=vol_prec, allow_zero=True, logger=self.logger))
            df['volume'] = df['volume'].apply(lambda x: max(Decimal(0), x) if x is not None else None) # Ensure volume >= 0


            # Drop rows with NaN/None in essential columns AFTER conversion attempts
            initial_len = len(df)
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            df.dropna(subset=essential_cols, inplace=True)
            cleaned_len = len(df)
            if cleaned_len < initial_len:
                    self.logger.debug(f"Dropped {initial_len - cleaned_len} rows with invalid OHLCV data for {symbol}.")

            if df.empty:
                self.logger.warning(f"{FG_YELLOW}Kline data for {symbol} became empty after cleaning.{RESET}")
                return None

            # Ensure data is sorted by timestamp (usually is, but good practice)
            df.sort_index(inplace=True)

            self.logger.debug(f"Successfully fetched and cleaned {len(df)} klines for {symbol}.")
            return df

        except Exception as e:
             # Catch errors from _retry_api_call or DataFrame processing
             self.logger.error(f"{FG_RED}Failed to fetch or process klines for {symbol}: {e}{RESET}", exc_info=False) # exc_info False if retry wrapper logged it
             return None


    def fetch_balance(self, currency: str) -> Optional[Decimal]:
        """
        Fetches the available balance for a specific currency (e.g., USDT).
        Adapts to Bybit V5 Unified Account structure.
        """
        self.logger.debug(f"Fetching balance for {currency}...")
        try:
            # Bybit V5: Use fetch_balance with specific parameters for Unified account
            # Specify the account type and the coin/currency
            params = {'accountType': 'UNIFIED', 'coin': currency.upper()} # Ensure currency is uppercase
            balance_data = self._retry_api_call(self.exchange.fetch_balance, params=params)

            # --- Parse V5 Unified Account Balance Response ---
            # Expected structure (may vary slightly): info -> result -> list -> [accountInfo]
            if balance_data and 'info' in balance_data and 'result' in balance_data['info']:
                result_list = balance_data['info']['result'].get('list', [])
                if result_list:
                    # Find the account info matching the requested coin
                    account_info = None
                    for item in result_list:
                        if item.get('coin') == currency.upper():
                            account_info = item
                            break

                    if account_info:
                        # 'availableToWithdraw' or 'availableBalance' might be relevant.
                        # 'availableBalance' typically reflects balance usable for margin/trades.
                        available_balance_str = account_info.get('availableBalance')

                        if available_balance_str is not None:
                            # Use high precision for balance, format later if needed for display
                            bal_decimal = safe_decimal(available_balance_str, f"{currency} available balance", precision=10, allow_zero=True, logger=self.logger)
                            if bal_decimal is not None:
                                self.logger.info(f"{FG_GREEN}Available {currency} balance: {bal_decimal}{RESET}")
                                return bal_decimal
                            else:
                                self.logger.warning(f"{FG_YELLOW}Could not convert fetched available balance '{available_balance_str}' to Decimal for {currency}. Treating as 0.{RESET}")
                                return Decimal(0)
                        else:
                            self.logger.warning(f"{FG_YELLOW}Could not find 'availableBalance' field for {currency} in balance response item.{RESET}")
                            self.logger.debug(f"Account info item: {account_info}")
                            return Decimal(0) # Assume zero if field missing
                    else:
                        self.logger.warning(f"{FG_YELLOW}No account info found for coin '{currency.upper()}' in balance response list. Assuming 0 balance.{RESET}")
                        self.logger.debug(f"Balance response list: {result_list}")
                        return Decimal(0)
                else:
                     self.logger.warning(f"{FG_YELLOW}Balance response 'list' is empty for {currency}. Assuming 0 balance.{RESET}")
                     return Decimal(0)

            # --- Fallback: Check standard CCXT 'free' balance ---
            # This might apply if not using Unified or if V5 parsing fails unexpectedly
            elif balance_data and currency.upper() in balance_data and 'free' in balance_data[currency.upper()]:
                 available_balance_str = balance_data[currency.upper()]['free']
                 bal_decimal = safe_decimal(available_balance_str, f"{currency} available balance (fallback)", precision=10, allow_zero=True, logger=self.logger)
                 if bal_decimal is not None:
                      self.logger.info(f"{FG_GREEN}Available {currency} balance (fallback): {bal_decimal}{RESET}")
                      return bal_decimal
                 else:
                      self.logger.warning(f"{FG_YELLOW}Could not convert fallback balance '{available_balance_str}' to Decimal for {currency}. Treating as 0.{RESET}")
                      return Decimal(0)
            else:
                # If response structure is completely unexpected
                self.logger.warning(f"{FG_YELLOW}Unexpected balance response structure received for {currency}. Assuming 0.{RESET}")
                self.logger.debug(f"Full balance response: {balance_data}")
                return Decimal(0) # Assume zero on unexpected structure

        except Exception as e:
            self.logger.error(f"{FG_RED}Failed to fetch or parse balance for {currency}: {e}{RESET}", exc_info=False)
            return None


    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetches the latest ticker information (last price, bid, ask, mark price) for a symbol."""
        self.logger.debug(f"Fetching ticker for {symbol}...")
        try:
            # Bybit V5: Specify category, e.g., 'linear'
            params = {'category': 'linear'} # Adjust category if needed (spot, inverse)
            ticker_data = self._retry_api_call(self.exchange.fetch_ticker, symbol, params=params)

            if ticker_data:
                 price_prec, _ = self.get_precision(symbol)
                 # Define a context for Decimal conversion within this scope if needed
                 # local_context = Context(prec=price_prec + 4) # Extra precision for intermediate calcs

                 # Parse relevant fields, converting to Decimal using safe_decimal
                 parsed_ticker = {
                    'symbol': symbol,
                    'timestamp': ticker_data.get('timestamp'), # ms timestamp from exchange
                    # Use 'last' for last traded price
                    'last': safe_decimal(ticker_data.get('last'), 'last price', precision=price_prec, logger=self.logger),
                    # Use bid/ask for order placement logic (limit orders) or slippage estimation
                    'bid': safe_decimal(ticker_data.get('bid'), 'bid price', precision=price_prec, logger=self.logger),
                    'ask': safe_decimal(ticker_data.get('ask'), 'ask price', precision=price_prec, logger=self.logger),
                    # Mark price is crucial for futures (funding, liquidation)
                    # Bybit V5: Usually found in info['markPrice']
                    'mark': safe_decimal(ticker_data.get('info', {}).get('markPrice'), 'mark price', precision=price_prec, logger=self.logger)
                 }

                 # Check if essential prices were parsed successfully
                 if parsed_ticker['last'] is not None or parsed_ticker['mark'] is not None:
                    # Log the more relevant price (mark for futures, last as fallback)
                    log_price = parsed_ticker['mark'] if parsed_ticker['mark'] else parsed_ticker['last']
                    self.logger.debug(f"Ticker for {symbol}: Mark/Last={log_price}, Bid={parsed_ticker.get('bid', 'N/A')}, Ask={parsed_ticker.get('ask', 'N/A')}")
                    return parsed_ticker
                 else:
                     self.logger.warning(f"{FG_YELLOW}Could not parse essential price (last/mark) from ticker for {symbol}. Ticker data: {ticker_data}{RESET}")
                     return None
            else:
                # fetch_ticker returned None or empty dict
                self.logger.warning(f"{FG_YELLOW}Received empty ticker data for {symbol}.{RESET}")
                return None

        except Exception as e:
            self.logger.error(f"{FG_RED}Failed to fetch or parse ticker for {symbol}: {e}{RESET}", exc_info=False)
            return None

    def fetch_position(self, symbol: str) -> Optional[Dict]:
        """
        Fetches the current open position for a specific symbol using Bybit V5 API structure.
        Returns a dictionary with position details if found, otherwise None.
        """
        self.logger.debug(f"Fetching position for {symbol}...")
        try:
            # Bybit V5: Requires category and optionally symbol for fetching positions
            # Use fetch_positions (plural) and filter, as it's generally more reliable for V5
            params = {'category': 'linear', 'symbol': symbol} # Specify symbol for efficiency
            # Fetch positions for the specific symbol only
            positions_data = self._retry_api_call(self.exchange.fetch_positions, symbols=[symbol], params=params)

            if positions_data:
                # fetch_positions returns a list, even if fetching for a single symbol.
                # We expect only one item matching the symbol, or an empty list if no position.
                for pos in positions_data:
                    # --- Parse V5 Position Data (Key fields based on Bybit docs/ccxt structure) ---
                    pos_symbol = pos.get('symbol') # Standard CCXT key
                    if pos_symbol != symbol: continue # Skip if somehow a different symbol is returned

                    # Size: info -> size (string, in base currency units like BTC)
                    pos_contracts_str = pos.get('info', {}).get('size')
                    pos_contracts = safe_decimal(pos_contracts_str, f"{symbol} position size", allow_zero=True, logger=self.logger)

                    # Ignore positions with zero size (effectively closed)
                    if pos_contracts is None or pos_contracts.is_zero():
                        continue # This entry represents a closed or non-existent position

                    # --- If size is non-zero, parse other details ---
                    price_prec, amount_prec = self.get_precision(symbol)
                    contract_size = self.get_contract_size(symbol) # Needed for value calculation

                    # Entry Price: info -> avgPrice (string)
                    entry_price_str = pos.get('info', {}).get('avgPrice')
                    entry_price = safe_decimal(entry_price_str, f"{symbol} entry price", precision=price_prec, logger=self.logger)

                    # Side: info -> side ('Buy' or 'Sell') -> map to 'long'/'short'
                    side_raw = pos.get('info', {}).get('side')
                    side = 'long' if side_raw == 'Buy' else 'short' if side_raw == 'Sell' else None

                    # Leverage: info -> leverage (string)
                    leverage_str = pos.get('info', {}).get('leverage')
                    leverage = safe_decimal(leverage_str, f"{symbol} leverage", precision=2, logger=self.logger)

                    # Unrealized PNL: info -> unrealisedPnl (string)
                    unrealized_pnl_str = pos.get('info', {}).get('unrealisedPnl')
                    # PNL precision often matches quote currency precision (price precision)
                    unrealized_pnl = safe_decimal(unrealized_pnl_str, f"{symbol} unrealized PNL", precision=price_prec, logger=self.logger)

                    # Mark Price: info -> markPrice (string) - useful for TSL/BE checks
                    mark_price_str = pos.get('info', {}).get('markPrice')
                    mark_price = safe_decimal(mark_price_str, f"{symbol} mark price", precision=price_prec, logger=self.logger)

                    # Position Value (Quote Currency): Calculated = abs(Size * EntryPrice * ContractSize)
                    size_quote = None
                    if entry_price and pos_contracts and contract_size:
                         try:
                             size_quote = abs(pos_contracts * entry_price * contract_size)
                             # Quantize calculated value for consistency? Optional.
                             # size_quote = size_quote.quantize(Decimal('1e-' + str(price_prec)))
                         except InvalidOperation:
                             self.logger.warning(f"Could not calculate quote value for {symbol} position.")

                    # Timestamp: info -> updatedTime (string, ms)
                    timestamp_ms_str = pos.get('info', {}).get('updatedTime')
                    timestamp = int(timestamp_ms_str) if timestamp_ms_str and timestamp_ms_str.isdigit() else None

                    # --- Validate Essential Data ---
                    if entry_price is not None and side in ['long', 'short']:
                        parsed_position = {
                            'symbol': symbol,
                            'side': side,
                            'size_contracts': pos_contracts, # Size in base currency (e.g., BTC)
                            'size_quote': size_quote,         # Approx value in quote currency (e.g., USDT)
                            'entry_price': entry_price,
                            'mark_price': mark_price,         # Current mark price
                            'leverage': leverage,
                            'unrealized_pnl': unrealized_pnl,
                            'timestamp': timestamp,         # Last updated timestamp (ms)
                            # Store raw CCXT position object for potential debugging or extra fields
                            'raw_data': pos # Contains 'info' field with all raw details
                        }
                        self.logger.info(f"{FG_GREEN}Found open position for {symbol}: {side.upper()} {pos_contracts} contracts @ {entry_price}{RESET}")
                        # Since we fetched for a specific symbol, return the first valid one found
                        return parsed_position
                    else:
                        # Log if essential data was missing for a non-zero size entry
                        self.logger.warning(f"{FG_YELLOW}Incomplete position data found for {symbol} (Entry='{entry_price_str}', Side='{side_raw}'). Ignoring this entry.{RESET}")
                        # Continue to check next item in positions_data (though unlikely if fetching by symbol)

                # If loop finishes without returning, no *active* (non-zero size) position found
                self.logger.info(f"No active open position found for {symbol} (checked {len(positions_data)} entries).")
                return None
            else:
                # This means fetch_positions returned an empty list or None.
                self.logger.info(f"No position data returned for {symbol} (fetch_positions was empty or None).")
                return None

        except Exception as e:
            # Catch errors from retry wrapper or parsing logic
            self.logger.error(f"{FG_RED}Failed to fetch or parse position for {symbol}: {e}{RESET}", exc_info=False)
            return None


    def place_order(self, symbol: str, side: str, order_type: str, amount: Decimal, price: Optional[Decimal] = None, params: Dict = {}) -> Optional[Dict]:
        """
        Places an order using Bybit V5 API conventions via CCXT.
        Handles market and limit orders, including attached SL/TP using V5 features.
        Ensures proper formatting of amount and price based on market precision.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            side: 'buy' or 'sell'.
            order_type: 'market', 'limit', etc.
            amount: Order size in base currency (Decimal).
            price: Required price for limit orders (Decimal).
            params: Additional parameters for the CCXT create_order call (e.g., SL/TP).

        Returns:
            The CCXT order dictionary on success, None on failure.
        """
        order_type = order_type.lower()
        side = side.lower()
        if side not in ['buy', 'sell']:
            self.logger.error(f"{FG_RED}Invalid order side '{side}'. Must be 'buy' or 'sell'.{RESET}")
            return None

        price_prec, amount_prec = self.get_precision(symbol)

        # --- Validate and Format Amount ---
        if amount is None or not isinstance(amount, Decimal) or amount <= 0:
             self.logger.error(f"{FG_RED}Invalid order amount: {amount} (Type: {type(amount)}). Must be a positive Decimal.{RESET}")
             return None

        # Format amount according to market precision (amount is in base currency)
        # Crucially, round DOWN the amount to avoid "insufficient funds" errors due to tiny fractions
        formatted_amount_str = format_value(amount, amount_prec, rounding=ROUND_DOWN)
        # Convert back to Decimal for final checks and logging if needed
        formatted_amount_decimal = safe_decimal(formatted_amount_str, "formatted amount", logger=self.logger)

        if formatted_amount_decimal is None or formatted_amount_decimal <= 0:
             self.logger.error(f"{FG_RED}Order amount became zero or invalid after formatting to {amount_prec} decimals: '{formatted_amount_str}'. Original: {amount}. Cannot place order.{RESET}")
             return None

        # --- Validate and Format Price (if applicable) ---
        formatted_price_str = None
        if order_type in ['limit', 'stop_limit', 'take_profit_limit']: # Types requiring price
            if price is None or not isinstance(price, Decimal) or price <= 0:
                self.logger.error(f"{FG_RED}Invalid or missing price ({price}) for {order_type} order.{RESET}")
                return None
            # Standard rounding for price
            formatted_price_str = format_value(price, price_prec, rounding=ROUND_HALF_UP)
            self.logger.debug(f"Formatted price for {symbol} {order_type}: {formatted_price_str}")

        # --- Prepare Log Message ---
        log_price_part = f" at {formatted_price_str}" if formatted_price_str else ""
        # Check for SL/TP info within the 'params' dict for logging
        sl_price = params.get('stopLoss')
        tp_price = params.get('takeProfit')
        log_sl_tp = ""
        if sl_price: log_sl_tp += f" SL={sl_price}"
        if tp_price: log_sl_tp += f" TP={tp_price}"
        log_reduce_only = " (ReduceOnly)" if params.get('reduceOnly') else ""

        self.logger.info(
            f"{FG_BLUE}Attempting to place {side.upper()} {order_type.upper()} order: "
            f"{formatted_amount_str} {symbol}{log_price_part}{log_sl_tp}{log_reduce_only}...{RESET}"
        )
        self.logger.debug(f" -> Raw Amount: {amount}, Formatted Amount: {formatted_amount_str} (Prec: {amount_prec})")
        if formatted_price_str:
            self.logger.debug(f" -> Raw Price: {price}, Formatted Price: {formatted_price_str} (Prec: {price_prec})")
        self.logger.debug(f" -> Additional Params: {params}")

        # --- Ensure Bybit V5 Specific Params are Present ---
        if 'category' not in params:
            params['category'] = 'linear' # Default to linear if not specified
            self.logger.debug(" -> Added default 'category': 'linear' to params.")
        # Handle position mode (One-Way vs Hedge) - Bybit requires positionIdx
        # 0: One-Way Mode
        # 1: Hedge Mode Buy/Long
        # 2: Hedge Mode Sell/Short
        if 'positionIdx' not in params:
            params['positionIdx'] = 0 # Assume One-Way mode by default
            self.logger.debug(" -> Added default 'positionIdx': 0 (One-Way Mode) to params.")

        # --- Place Order via CCXT ---
        try:
            # Pass the precisely formatted string amount and price to CCXT
            # CCXT handles converting these strings to the required format for the API call
            order = self._retry_api_call(
                self.exchange.create_order,
                symbol,
                order_type,
                side,
                formatted_amount_str, # Pass the formatted STRING amount
                formatted_price_str,  # Pass the formatted STRING price or None
                params                # Pass the full parameters dictionary
            )

            # --- Validate Response ---
            if order and order.get('id'):
                order_id = order.get('id')
                # Check status from the response. Bybit V5 might return 'New', 'PartiallyFilled', 'Filled', 'Cancelled', 'Rejected' etc.
                # Use 'info' field for more detailed Bybit-specific status if needed.
                status = order.get('status', order.get('info', {}).get('orderStatus', '')).lower()
                filled_amount = order.get('filled', 0.0) # CCXT usually provides this

                if status == 'rejected':
                     # Try to get rejection reason from Bybit's response in 'info'
                     reason_code = order.get('info', {}).get('rejectCode', 'N/A')
                     reason_msg = order.get('info', {}).get('rejectReason', 'Unknown') # Standard V5 field? Check docs. Or rely on ccxt parsing.
                     self.logger.error(f"{FG_RED}Order placement for {symbol} REJECTED by exchange. ID: {order_id}, Status: {status}, Reason Code: {reason_code}, Message: {reason_msg}{RESET}")
                     self.logger.debug(f"Rejected Order Details: {order}")
                     return None # Treat rejected as failure

                # Log success details
                log_filled_part = f", Filled: {filled_amount}" if filled_amount > 0 else ""
                log_status_part = f", Status: {status}" if status else ""
                self.logger.info(f"{FG_GREEN}{BRIGHT}Order {order_id} placed successfully for {symbol}{log_status_part}{log_filled_part}.{RESET}")
                self.logger.debug(f"Order details: {order}")
                return order # Return the full CCXT order dictionary
            else:
                # If retry_api_call didn't raise but the order structure is invalid/missing ID
                self.logger.error(f"{FG_RED}Order placement attempt for {symbol} did not return a valid order structure or ID.{RESET}")
                self.logger.debug(f"Invalid order response received: {order}")
                return None

        except ccxt.InsufficientFunds as e:
            self.logger.error(f"{BG_RED}{FG_WHITE}Insufficient Funds to place {side} {order_type} order for {formatted_amount_str} {symbol}. Details: {e}{RESET}")
            return None
        except ccxt.InvalidOrder as e:
            # Often indicates parameter issues (e.g., price/amount precision, invalid combination)
            self.logger.error(f"{FG_RED}Invalid Order parameters for {symbol}: {e}. Check amount/price precision, order type, limits, or conflicting params.{RESET}")
            self.logger.debug(f"Failed Order Params: Amount='{formatted_amount_str}', Price='{formatted_price_str}', Extra={params}")
            # Log the raw exception message which might contain Bybit's specific reason code/message
            self.logger.debug(f"Raw CCXT InvalidOrder Exception: {e.args}")
            return None
        except Exception as e:
             # Catches errors from _retry_api_call or other unexpected issues
             # Error was likely logged already by _retry_api_call if it came from there
             if not isinstance(e, (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError)):
                  # Log if it's an unexpected type not handled by retry logic's final raise
                  self.logger.error(f"{FG_RED}Unexpected failure during {side} {order_type} order placement for {symbol}: {e}{RESET}", exc_info=True)
             return None # Return None on failure


    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancels a specific open order by its ID.

        Args:
            order_id: The exchange's ID for the order to cancel.
            symbol: The trading pair symbol associated with the order.

        Returns:
            True if cancellation was likely successful (or order was already gone).
            False if cancellation failed with an error.
        """
        self.logger.info(f"{FG_YELLOW}Attempting to cancel order {order_id} for {symbol}...{RESET}")
        try:
            # Bybit V5 cancel requires category. Symbol is optional in CCXT's base method but recommended/required by Bybit.
            params = {'category': 'linear'} # Match the category of the order being cancelled
            response = self._retry_api_call(self.exchange.cancel_order, order_id, symbol, params=params)

            # --- Analyze Cancellation Response ---
            # Successful cancellation in Bybit V5 often returns the cancelled orderId.
            # CCXT's cancel_order might return None, the order structure, or raise an error.
            # We primarily rely on *not* getting an exception.

            # Check Bybit V5 specific success code if available in 'info'
            if response and isinstance(response, dict) and response.get('info', {}).get('retCode') == 0:
                 cancelled_id = response.get('info', {}).get('result', {}).get('orderId', order_id)
                 self.logger.info(f"{FG_GREEN}Cancel request for order {cancelled_id} successful (retCode 0).{RESET}")
                 return True
            else:
                 # If no error was raised, assume success but log the response for verification
                 self.logger.info(f"{FG_GREEN}Cancel request for order {order_id} likely successful (no error raised during API call).{RESET}")
                 self.logger.debug(f"Cancel response snippet: {str(response)[:150]}...")
                 return True

        except ccxt.OrderNotFound:
             # This is common and usually acceptable - order was already filled, cancelled, or never existed.
             self.logger.warning(f"{FG_YELLOW}Order {order_id} for {symbol} not found. It might be filled, already cancelled, or the ID/symbol/category is incorrect. Treating as success (order is gone).{RESET}")
             return True # Treat as success if it's already gone
        except ccxt.InvalidOrder as e:
             # This might happen if the order is in a final state (filled, rejected, fully cancelled)
             # where cancellation is no longer possible.
             self.logger.warning(f"{FG_YELLOW}Cannot cancel order {order_id} ({symbol}) due to its current state (likely already closed/cancelled): {e}{RESET}")
             return True # Treat as success if it's effectively gone and cannot be cancelled
        except Exception as e:
            # Errors from retry wrapper were likely logged already
            if not isinstance(e, (ccxt.NetworkError, ccxt.ExchangeError)):
                 # Log unexpected errors during cancellation attempt
                 self.logger.error(f"{FG_RED}Failed to cancel order {order_id} for {symbol}: {e}{RESET}", exc_info=True)
            return False # Indicate failure

    def fetch_open_orders(self, symbol: str) -> List[Dict]:
        """Fetches all open orders for a specific symbol (e.g., 'BTC/USDT')."""
        self.logger.debug(f"Fetching open orders for {symbol}...")
        try:
            # Bybit V5: Specify category
            params = {'category': 'linear'}
            open_orders = self._retry_api_call(self.exchange.fetch_open_orders, symbol, params=params)
            self.logger.debug(f"Found {len(open_orders)} open orders for {symbol}.")
            return open_orders
        except Exception as e:
            # Errors likely logged by retry wrapper
            self.logger.error(f"{FG_RED}Failed to fetch open orders for {symbol}: {e}{RESET}", exc_info=False)
            return [] # Return empty list on failure


    # --- Potential Future Enhancements ---

    # def edit_order(self, order_id: str, symbol: str, ...) -> Optional[Dict]:
    #     """ Edits an existing open order (e.g., modify SL/TP price or amount).
    #         Requires careful handling of Bybit V5 parameters (orderId/orderLinkId, etc.)
    #         and checking exchange support via ccxt.exchange.has['editOrder'].
    #     """
    #     if not self.exchange.has.get('editOrder'):
    #          self.logger.error(f"{FG_RED}Exchange {self.exchange.id} does not support editing orders via edit_order. Need to cancel and replace.{RESET}")
    #          return None
    #     # ... implementation for editing ...

    # def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
    #     """ Cancels all open orders, optionally filtered by symbol. """
    #     # ... implementation using fetch_open_orders and cancel_order loop ...

EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate exchange.py."; fi
echog " -> exchange.py generated successfully."
echo # Newline

# 5. Create strategy.py
echob "Step 5: Generating strategy.py..."
cat << 'EOF' > strategy.py
# ~/trading-bot/strategy.py
import pandas as pd
import pandas_ta as pta
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

# Ensure pandas_ta is installed: pip install pandas_ta
from .utils import safe_decimal, FG_YELLOW, FG_RED, FG_CYAN, RESET, FG_GREEN, BRIGHT
from .config import StrategyParams # For type hinting and validation

# Define a structure for Order Blocks (OBs)
@dataclass
class OrderBlock:
    """Represents a Bullish or Bearish Order Block identified on the chart."""
    type: str # 'bull' or 'bear'
    top: Decimal # Highest price point of the OB
    bottom: Decimal # Lowest price point of the OB
    timestamp: pd.Timestamp # Timestamp of the candle *forming* the block
    pivot_price: Decimal # Price of the pivot high/low that *caused* the block formation
    is_mitigated: bool = False # Flag indicating if price has traded through the block
    mitigation_timestamp: Optional[pd.Timestamp] = None # Timestamp when mitigation occurred
    index: Any = None # Can store the original index (timestamp or int) for reference

    def __post_init__(self):
        # Ensure top is always greater than bottom
        if self.bottom >= self.top:
            raise ValueError(f"OrderBlock bottom ({self.bottom}) must be less than top ({self.top})")

    # Define equality based on key attributes for set operations and uniqueness checks
    def __eq__(self, other):
        if not isinstance(other, OrderBlock):
            return NotImplemented
        # Use quantization for robust Decimal comparison, avoiding minor precision issues
        precision = Decimal('1e-8') # Define a suitable comparison precision
        return (self.type == other.type and
                self.top.quantize(precision) == other.top.quantize(precision) and
                self.bottom.quantize(precision) == other.bottom.quantize(precision) and
                self.timestamp == other.timestamp) # Timestamp match is crucial

    def __hash__(self):
        # Hash based on the same immutable attributes used for equality
        precision = Decimal('1e-8')
        return hash((self.type,
                     self.top.quantize(precision),
                     self.bottom.quantize(precision),
                     self.timestamp))

    def __str__(self):
        mit_status = f"Mitigated @ {self.mitigation_timestamp}" if self.is_mitigated else "Active"
        return (f"{self.type.capitalize()} OB @ {self.timestamp}: "
                f"[{self.bottom:.8f} - {self.top:.8f}] ({mit_status})")


# Define a structure for the strategy analysis results
@dataclass
class StrategyAnalysisResults:
    """Holds the results of the strategy analysis for a given symbol and timeframe."""
    symbol: str
    timeframe: str
    last_close: Decimal
    last_high: Decimal # Include high/low for context (e.g., proximity checks)
    last_low: Decimal
    current_trend_up: Optional[bool] # True for Up, False for Down, None if undetermined
    trend_just_changed: bool = False # Flag if the trend changed on the most recent candle
    atr: Optional[Decimal] = None # Latest ATR value
    active_bull_boxes: List[OrderBlock] = field(default_factory=list) # Unmitigated Bull OBs
    active_bear_boxes: List[OrderBlock] = field(default_factory=list) # Unmitigated Bear OBs
    # Optional: Include latest indicator values for debugging or advanced signals
    vt_trend_ema: Optional[Decimal] = None # Latest Volumatic Trend EMA value
    # Optional: Keep the analyzed DataFrame (can consume memory)
    # df_analyzed: Optional[pd.DataFrame] = None


class VolumaticOBStrategy:
    """
    Implements the Volumatic Trend strategy combined with Order Block identification.
    - Calculates Volumatic Trend indicator to determine market trend.
    - Identifies Bullish and Bearish Order Blocks based on price pivots.
    - Checks for Order Block mitigation.
    - Provides analysis results including trend, ATR, and active OBs.
    """

    def __init__(self, params: Dict, logger: logging.Logger):
        """
        Initializes the strategy with parameters validated by the StrategyParams model.
        """
        try:
            # Validate and store strategy parameters using the Pydantic model
            self.params = StrategyParams(**params)
            self.logger = logger
            # Internal state to store all identified OBs (both active and mitigated) per symbol
            # Using sets for efficient management of unique OBs during updates
            self.all_bull_boxes: Dict[str, set[OrderBlock]] = {}
            self.all_bear_boxes: Dict[str, set[OrderBlock]] = {}
            self.logger.info(f"VolumaticOB Strategy initialized with params: {self.params.dict()}")
        except ValidationError as e:
             logger.error(f"{FG_RED}Invalid strategy parameters provided during initialization:\n{e}{RESET}")
             raise ValueError("Invalid strategy parameters") from e
        except Exception as e:
            logger.error(f"{FG_RED}Unexpected error initializing strategy: {e}{RESET}", exc_info=True)
            raise

    def _calculate_volumatic_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Volumatic Trend (VT) indicator and related components.
        Uses pandas_ta for EMA and ATR calculations. Requires OHLCV columns.
        Returns the DataFrame with added VT indicator columns.
        """
        self.logger.debug("Calculating Volumatic Trend indicators...")

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            self.logger.error(f"{FG_RED}Missing required columns in DataFrame for VT calc: {missing}. Have: {df.columns.tolist()}{RESET}")
            return df # Return unchanged df if columns are missing

        # --- Data Preparation ---
        # Ensure OHLCV columns are numeric (float or Decimal acceptable by pandas_ta)
        # Using pd.to_numeric handles potential string inputs
        for col in required_cols:
            # errors='coerce' will turn non-numeric values into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN in essential columns *before* calculations
        initial_len = len(df)
        df.dropna(subset=required_cols, inplace=True)
        if len(df) < initial_len:
             self.logger.debug(f"Dropped {initial_len - len(df)} rows with NaN in OHLCV before VT calc.")

        if df.empty:
            self.logger.warning(f"{FG_YELLOW}DataFrame became empty after dropping NaNs before VT calculation.{RESET}")
            return df

        # Check if enough data remains for the longest lookback period
        min_data_needed = max(self.params.vt_length, self.params.vt_atr_period, self.params.vt_vol_ema_length)
        if len(df) < min_data_needed:
             self.logger.warning(f"{FG_YELLOW}Insufficient data ({len(df)} rows) after NaN drop for VT calculation (longest lookback: {min_data_needed}).{RESET}")
             return df # Not enough data to calculate indicators reliably

        # --- Indicator Calculations using pandas_ta ---
        try:
            # 1. ATR (Average True Range)
            atr_col = f'ATRr_{self.params.vt_atr_period}'
            df.ta.atr(length=self.params.vt_atr_period, append=True)
            if atr_col not in df.columns: raise ValueError("ATR calculation failed (column not found)")

            # 2. Volume EMA (for normalization)
            vol_ema_col = f'VOL_EMA_{self.params.vt_vol_ema_length}' # Custom name
            df[vol_ema_col] = pta.ema(df['volume'], length=self.params.vt_vol_ema_length)
            if df[vol_ema_col].isnull().all(): raise ValueError("Volume EMA calculation failed (all NaNs)")

            # --- Handle initial NaNs from EMA/ATR ---
            # Forward fill NaNs to allow subsequent calculations
            df[atr_col] = df[atr_col].ffill()
            df[vol_ema_col] = df[vol_ema_col].ffill()

            # Replace any remaining NaNs (e.g., at the very beginning) or zeros in Vol EMA
            # Use a tiny number to avoid division by zero in normalization
            df[vol_ema_col].fillna(method='bfill', inplace=True) # Backfill first
            df[vol_ema_col].fillna(1e-9, inplace=True) # Fill remaining start NaNs
            df[vol_ema_col] = df[vol_ema_col].replace(0, 1e-9)

            # Fill NaNs in ATR (usually at the start)
            df[atr_col].fillna(method='bfill', inplace=True) # Backfill first
            df[atr_col].fillna(0, inplace=True) # Fill remaining start NaNs with 0

            # 3. Normalized Volume
            df['normalized_volume'] = df['volume'] / df[vol_ema_col]

            # 4. Volumatic Trend Value (Range * Normalized Volume)
            # Ensure high/low are numeric before calculation
            df['price_range'] = df['high'] - df['low']
            df['vt_value'] = df['price_range'] * df['normalized_volume']

            # 5. EMA of Volumatic Trend Value (The actual VT EMA line)
            vt_ema_col = f'VT_EMA_{self.params.vt_length}' # Custom name
            df[vt_ema_col] = pta.ema(df['vt_value'], length=self.params.vt_length)
            if df[vt_ema_col].isnull().all(): raise ValueError("VT EMA calculation failed (all NaNs)")

            # Fill NaNs in VT EMA
            df[vt_ema_col].fillna(method='bfill', inplace=True)
            df[vt_ema_col].fillna(0, inplace=True)

            # 6. Determine Trend Direction (Raw: 1=Up, -1=Down, 0=Flat)
            # Compare close price with the calculated VT EMA
            df['trend_raw'] = np.where(
                df['close'] > df[vt_ema_col], 1,
                np.where(df['close'] < df[vt_ema_col], -1, 0)
            )

            # 7. Smoothed Trend Direction (Fill zeros with previous valid trend)
            # Replace 0s with NaN, forward fill, then fill remaining start NaNs with 0
            df['vt_trend_direction'] = df['trend_raw'].replace(0, np.nan).ffill().fillna(0).astype(int)

            # 8. Calculate Trend Bands (Optional - for visualization or other signals)
            df['vt_upper_band'] = df[vt_ema_col] + df[atr_col] * self.params.vt_atr_multiplier
            df['vt_lower_band'] = df[vt_ema_col] - df[vt_ema_col] * self.params.vt_atr_multiplier

            self.logger.debug("Volumatic Trend calculation finished.")

        except Exception as e:
             self.logger.error(f"{FG_RED}Error during pandas_ta indicator calculation: {e}{RESET}", exc_info=True)
             # Return df without potentially corrupted indicator columns if error occurred mid-calc
             return df.drop(columns=[atr_col, vol_ema_col, vt_ema_col, 'normalized_volume', 'price_range', 'vt_value', 'trend_raw', 'vt_trend_direction', 'vt_upper_band', 'vt_lower_band'], errors='ignore')

        # --- Convert key calculated columns to Decimal for precision in strategy logic ---
        # Do this *after* all calculations involving floats are complete.
        indicator_precision = 8 # Define desired precision for indicators
        decimal_cols = [atr_col, vt_ema_col, 'vt_upper_band', 'vt_lower_band']
        for col in decimal_cols:
             if col in df.columns:
                 # Apply safe_decimal conversion, allowing zero for ATR/bands
                 df[col] = df[col].apply(lambda x: safe_decimal(x, col, precision=indicator_precision, allow_zero=True, logger=self.logger))
                 # Optional: Check if conversion resulted in Nones and handle if needed
                 # if df[col].isnull().any():
                 #     self.logger.warning(f"Column '{col}' contains None values after Decimal conversion.")

        return df

    def _identify_order_blocks(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Identifies potential Order Blocks (OBs) based on pivot highs/lows.
        Uses pandas_ta to find pivots and then checks the candle *before* the pivot.
        Updates the internal state (self.all_bull_boxes, self.all_bear_boxes) with unique OBs found.
        """
        self.logger.debug(f"Identifying Order Blocks for {symbol}...")

        # --- Ensure necessary columns exist and are Decimal ---
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"{FG_RED}Missing required OHLC columns for OB identification.{RESET}")
            return

        try:
            # Convert OHLC columns to Decimal if they aren't already
            # Use a reasonable precision consistent with price data
            price_prec = 8 # Or fetch dynamically if possible/needed
            for col in required_cols:
                if not df[col].apply(lambda x: isinstance(x, Decimal)).all():
                     self.logger.debug(f"Converting column '{col}' to Decimal for OB calculation.")
                     df[col] = df[col].apply(lambda x: safe_decimal(x, col, precision=price_prec, allow_zero=False, logger=self.logger))
                     # Drop rows where conversion might have failed (resulted in None)
                     df.dropna(subset=[col], inplace=True)

        except Exception as e:
             self.logger.error(f"{FG_RED}Error preparing Decimal columns for OB calculation: {e}{RESET}")
             return
        if df.empty:
             self.logger.warning(f"{FG_YELLOW}DataFrame empty after ensuring Decimal columns for OB calc.{RESET}")
             return

        # --- Calculate Pivots using pandas_ta ---
        ph_col = f'PH_{self.params.ph_left}_{self.params.ph_right}'
        pl_col = f'PL_{self.params.pl_left}_{self.params.pl_right}'
        try:
            # Calculate Pivot Highs on 'high' prices
            df[ph_col] = pta.pivot_high(df['high'], length=None, left=self.params.ph_left, right=self.params.ph_right)
            # Calculate Pivot Lows on 'low' prices
            df[pl_col] = pta.pivot_low(df['low'], length=None, left=self.params.pl_left, right=self.params.pl_right)
            # Forward fill pivot values? No, we need the exact pivot location.
        except Exception as e:
             self.logger.error(f"{FG_RED}Error calculating Pivots using pandas_ta: {e}{RESET}")
             return # Cannot proceed without pivots

        # --- Initialize or Retrieve Existing Box Sets for the Symbol ---
        # Use sets for efficient uniqueness handling based on OrderBlock hash/eq
        current_bull_set = self.all_bull_boxes.setdefault(symbol, set())
        current_bear_set = self.all_bear_boxes.setdefault(symbol, set())
        newly_found_bull_boxes = set()
        newly_found_bear_boxes = set()

        # --- Iterate Through Candles to Find OBs based on Pivots ---
        # Iterate from the pivot lookback period onwards to avoid index errors
        start_index = max(self.params.ph_left, self.params.pl_left)
        if len(df) <= start_index:
            self.logger.debug("Not enough data points to identify pivots/OBs.")
            return

        # Iterate up to the second-to-last row, as pivots depend on future bars (right lookback)
        for i in range(start_index, len(df) - max(1, self.params.ph_right, self.params.pl_right)):
            current_row = df.iloc[i]
            pivot_candle_timestamp = current_row.name # Timestamp of the pivot candle itself

            # --- Check for Bearish OB (formed by Pivot High) ---
            pivot_high_price_raw = current_row[ph_col]
            if pd.notna(pivot_high_price_raw):
                # Pivot High found at index 'i'. The OB is the *last up candle* before this pivot.
                # Look backwards from i-1 to find the most recent candle where close > open.
                ob_candle_index = -1
                for j in range(i - 1, max(-1, i - 1 - self.params.ph_left), -1): # Look back within left window
                    if df.iloc[j]['close'] > df.iloc[j]['open']:
                        ob_candle_index = j
                        break # Found the last up candle

                if ob_candle_index != -1:
                    ob_candle = df.iloc[ob_candle_index]
                    ob_timestamp = ob_candle.name
                    pivot_high_price = safe_decimal(pivot_high_price_raw, "pivot high price", logger=self.logger)
                    if pivot_high_price is None: continue # Skip if pivot price invalid

                    try:
                        if self.params.ob_source == "Wicks":
                            # Bear OB (using wicks): Top = High, Bottom = Open of the up candle
                            ob_top = ob_candle['high']
                            ob_bottom = ob_candle['open']
                        else: # "Body"
                            # Bear OB (using body): Top = Close, Bottom = Open of the up candle
                            ob_top = ob_candle['close']
                            ob_bottom = ob_candle['open']

                        potential_box = OrderBlock(type='bear', top=ob_top, bottom=ob_bottom,
                                                   timestamp=ob_timestamp, pivot_price=pivot_high_price,
                                                   index=ob_timestamp)
                        newly_found_bear_boxes.add(potential_box) # Add to set (handles duplicates)
                    except (ValueError, TypeError, InvalidOperation) as e:
                         self.logger.warning(f"Could not create Bear OB @ {ob_timestamp} from Pivot @ {pivot_candle_timestamp}: {e}")


            # --- Check for Bullish OB (formed by Pivot Low) ---
            pivot_low_price_raw = current_row[pl_col]
            if pd.notna(pivot_low_price_raw):
                 # Pivot Low found at index 'i'. The OB is the *last down candle* before this pivot.
                 # Look backwards from i-1 to find the most recent candle where close < open.
                 ob_candle_index = -1
                 for j in range(i - 1, max(-1, i - 1 - self.params.pl_left), -1): # Look back within left window
                     if df.iloc[j]['close'] < df.iloc[j]['open']:
                         ob_candle_index = j
                         break # Found the last down candle

                 if ob_candle_index != -1:
                     ob_candle = df.iloc[ob_candle_index]
                     ob_timestamp = ob_candle.name
                     pivot_low_price = safe_decimal(pivot_low_price_raw, "pivot low price", logger=self.logger)
                     if pivot_low_price is None: continue # Skip if pivot price invalid

                     try:
                         if self.params.ob_source == "Wicks":
                             # Bull OB (using wicks): Top = Open, Bottom = Low of the down candle
                             ob_top = ob_candle['open']
                             ob_bottom = ob_candle['low']
                         else: # "Body"
                             # Bull OB (using body): Top = Open, Bottom = Close of the down candle
                             ob_top = ob_candle['open']
                             ob_bottom = ob_candle['close']

                         potential_box = OrderBlock(type='bull', top=ob_top, bottom=ob_bottom,
                                                    timestamp=ob_timestamp, pivot_price=pivot_low_price,
                                                    index=ob_timestamp)
                         newly_found_bull_boxes.add(potential_box) # Add to set
                     except (ValueError, TypeError, InvalidOperation) as e:
                          self.logger.warning(f"Could not create Bull OB @ {ob_timestamp} from Pivot @ {pivot_candle_timestamp}: {e}")


        # --- Update Master Sets and Check Mitigation ---
        # Add newly found unique boxes to the master sets for the symbol
        initial_bull_count = len(current_bull_set)
        initial_bear_count = len(current_bear_set)
        current_bull_set.update(newly_found_bull_boxes)
        current_bear_set.update(newly_found_bear_boxes)

        # Check mitigation status for *all* boxes in the updated sets using the full df history
        self._check_mitigation(df, current_bull_set, current_bear_set)

        # Log summary of findings
        new_bull_found = len(current_bull_set) - initial_bull_count
        new_bear_found = len(current_bear_set) - initial_bear_count
        self.logger.debug(f"{symbol}: Found {new_bull_found} new Bull OBs, {new_bear_found} new Bear OBs.")
        self.logger.debug(f"{symbol}: Total unique OBs stored: Bull={len(current_bull_set)}, Bear={len(current_bear_set)}.")


    def _check_mitigation(self, df: pd.DataFrame, bull_boxes: set[OrderBlock], bear_boxes: set[OrderBlock]):
        """
        Checks and updates the mitigation status of Order Blocks based on subsequent price action.
        Modifies the is_mitigated flag and mitigation_timestamp directly on the OrderBlock objects within the sets.
        """
        self.logger.debug("Checking Order Block mitigation...")
        mitigated_count = 0
        # Iterate through potentially unmitigated boxes
        boxes_to_check = [b for b in bull_boxes if not b.is_mitigated] + [b for b in bear_boxes if not b.is_mitigated]

        if not boxes_to_check:
            self.logger.debug("No unmitigated boxes to check.")
            return

        # Ensure DataFrame index is sorted for efficient lookup
        if not df.index.is_monotonic_increasing:
             df = df.sort_index()

        for box in boxes_to_check:
            # Select candles that occurred *strictly after* the OB candle's timestamp
            # Using .loc for potentially faster timestamp-based slicing
            try:
                relevant_candles = df.loc[df.index > box.timestamp]
            except KeyError: # Can happen if box.timestamp is not in index (unlikely but possible)
                self.logger.warning(f"Timestamp {box.timestamp} not found in DataFrame index for mitigation check.")
                continue

            if relevant_candles.empty: continue # No candles after the OB

            # --- Check Bullish OB Mitigation ---
            if box.type == 'bull':
                # Mitigation: Low of a subsequent candle breaks below the OB's bottom
                mitigating_candles = relevant_candles[relevant_candles['low'] < box.bottom]
                if not mitigating_candles.empty:
                    # Find the first candle that caused mitigation
                    first_mitigation = mitigating_candles.iloc[0]
                    box.is_mitigated = True
                    box.mitigation_timestamp = first_mitigation.name
                    mitigated_count += 1
                    # self.logger.debug(f" -> Bull OB @ {box.timestamp} mitigated by {first_mitigation.name} (Low: {first_mitigation['low']:.8f} < Bottom: {box.bottom:.8f})")

            # --- Check Bearish OB Mitigation ---
            elif box.type == 'bear':
                 # Mitigation: High of a subsequent candle breaks above the OB's top
                 mitigating_candles = relevant_candles[relevant_candles['high'] > box.top]
                 if not mitigating_candles.empty:
                     first_mitigation = mitigating_candles.iloc[0]
                     box.is_mitigated = True
                     box.mitigation_timestamp = first_mitigation.name
                     mitigated_count += 1
                     # self.logger.debug(f" -> Bear OB @ {box.timestamp} mitigated by {first_mitigation.name} (High: {first_mitigation['high']:.8f} > Top: {box.top:.8f})")

        if mitigated_count > 0:
            self.logger.debug(f"Mitigation check complete. Updated status for {mitigated_count} Order Blocks.")


    def analyze(self, df_raw: pd.DataFrame, symbol: str, timeframe: str) -> Optional[StrategyAnalysisResults]:
        """
        Performs the full strategy analysis on the given DataFrame:
        1. Calculates Volumatic Trend indicators.
        2. Identifies and updates Order Blocks (including mitigation).
        3. Extracts latest values and active OBs.
        4. Returns results in a StrategyAnalysisResults object.
        """
        self.logger.info(f"{FG_CYAN}--- Starting Strategy Analysis for {symbol} ({timeframe}) ---{RESET}")
        if df_raw is None or df_raw.empty:
            self.logger.warning(f"{FG_YELLOW}Input DataFrame is empty for {symbol}. Cannot analyze.{RESET}")
            return None

        # --- Data Validation and Preparation ---
        # Ensure index is datetime
        if not isinstance(df_raw.index, pd.DatetimeIndex):
             try:
                 df_raw.index = pd.to_datetime(df_raw.index)
                 if not df_raw.index.is_monotonic_increasing:
                     df_raw = df_raw.sort_index() # Ensure chronological order
             except Exception as e:
                  self.logger.error(f"{FG_RED}Failed to convert index to DatetimeIndex or sort for {symbol}: {e}{RESET}")
                  return None

        # Make a copy to avoid modifying the original DataFrame passed to the function
        df = df_raw.copy()

        # Check minimum data length required for calculations BEFORE running them
        # Considers longest lookback for indicators and pivots
        min_len_needed = max(
            self.params.vt_length,
            self.params.vt_atr_period,
            self.params.vt_vol_ema_length,
            self.params.ph_left + self.params.ph_right + 1, # Pivot calc needs buffer
            self.params.pl_left + self.params.pl_right + 1
        ) + 5 # Add a general buffer

        if len(df) < min_len_needed:
            self.logger.warning(f"{FG_YELLOW}Insufficient data points ({len(df)}) for {symbol} strategy calculations (need ~{min_len_needed}).{RESET}")
            return None

        # --- Calculate Indicators ---
        try:
            # Pass the copy to the calculation function
            df_analyzed = self._calculate_volumatic_trend(df.copy())
            if df_analyzed is None or df_analyzed.empty or 'vt_trend_direction' not in df_analyzed.columns:
                 self.logger.error(f"{FG_RED}Volumatic Trend calculation failed or returned invalid data for {symbol}. Cannot proceed with analysis.{RESET}")
                 return None
        except Exception as e:
             self.logger.error(f"{FG_RED}Error during Volumatic Trend calculation step for {symbol}: {e}{RESET}", exc_info=True)
             return None

        # --- Identify Order Blocks ---
        # Use the original df copy (or df_analyzed if VT calc modified OHLC types unintentionally)
        # Ensure the df passed to OB identification has Decimal OHLC
        try:
             self._identify_order_blocks(df, symbol) # Modifies internal state (self.all_..._boxes)
        except Exception as e:
             self.logger.error(f"{FG_RED}Error during Order Block identification step for {symbol}: {e}{RESET}", exc_info=True)
             # Decide if analysis can continue without OBs. For this strategy, OBs are crucial.
             return None

        # --- Extract Results from the Latest Candle ---
        if df_analyzed.empty:
             self.logger.error(f"{FG_RED}Analyzed DataFrame is empty after indicator calculation for {symbol}. Cannot extract results.{RESET}")
             return None

        try:
            last_row = df_analyzed.iloc[-1]
            # Get original OHLC prices from the source Decimal df (important for precision)
            # Ensure 'df' still holds the correct Decimal data
            last_close_orig = df['close'].iloc[-1]
            last_high_orig = df['high'].iloc[-1]
            last_low_orig = df['low'].iloc[-1]

            # Validate these final prices
            last_close = safe_decimal(last_close_orig, "last_close", logger=self.logger)
            last_high = safe_decimal(last_high_orig, "last_high", logger=self.logger)
            last_low = safe_decimal(last_low_orig, "last_low", logger=self.logger)

            if last_close is None or last_high is None or last_low is None:
                self.logger.error(f"{FG_RED}Could not get valid last OHLC Decimal prices for {symbol}. Aborting analysis.{RESET}")
                return None

            # --- Extract Trend ---
            current_trend_val = last_row.get('vt_trend_direction') # Should be int (-1, 0, 1)
            current_trend_up: Optional[bool] = None
            if pd.notna(current_trend_val): # Check it's not NaN
                 current_trend_val = int(current_trend_val) # Ensure integer type
                 if current_trend_val == 1:
                     current_trend_up = True
                 elif current_trend_val == -1:
                     current_trend_up = False
                 # Else (current_trend_val == 0), trend remains None (undetermined/flat)
            else:
                 self.logger.warning(f"{FG_YELLOW}Last trend value is NaN for {symbol}. Trend undetermined.{RESET}")


            # --- Check for Trend Change ---
            trend_just_changed = False
            if len(df_analyzed) > 1:
                prev_trend_val = df_analyzed.iloc[-2].get('vt_trend_direction')
                # Compare only if both previous and current trends are valid and different
                if pd.notna(prev_trend_val) and pd.notna(current_trend_val) and \
                   int(prev_trend_val) != current_trend_val and \
                   int(prev_trend_val) != 0: # Ignore changes *from* flat (0)
                    trend_just_changed = True
                    self.logger.info(f"{FG_YELLOW}{BRIGHT}Trend change detected for {symbol}: {int(prev_trend_val)} -> {current_trend_val}{RESET}")

            # --- Extract ATR ---
            atr_col_name = f'ATRr_{self.params.vt_atr_period}'
            last_atr = last_row.get(atr_col_name)
            # Ensure ATR is Decimal (should be from _calculate_volumatic_trend)
            if not isinstance(last_atr, Decimal):
                 last_atr = safe_decimal(last_atr, "last_atr", precision=8, logger=self.logger) # Convert if needed

            # --- Extract VT EMA ---
            vt_ema_col_name = f'VT_EMA_{self.params.vt_length}'
            vt_ema = last_row.get(vt_ema_col_name)
            if not isinstance(vt_ema, Decimal):
                vt_ema = safe_decimal(vt_ema, 'vt_ema', precision=8, logger=self.logger)

            # --- Get Active Order Blocks ---
            # Filter the master sets for this symbol to get unmitigated boxes
            active_bull_boxes = sorted(
                [b for b in self.all_bull_boxes.get(symbol, set()) if not b.is_mitigated],
                key=lambda b: b.timestamp, reverse=True # Most recent first
            )[:self.params.ob_max_boxes] # Prune to max allowed

            active_bear_boxes = sorted(
                [b for b in self.all_bear_boxes.get(symbol, set()) if not b.is_mitigated],
                key=lambda b: b.timestamp, reverse=True
            )[:self.params.ob_max_boxes]

            # --- Create Result Object ---
            results = StrategyAnalysisResults(
                symbol=symbol,
                timeframe=timeframe,
                last_close=last_close,
                last_high=last_high,
                last_low=last_low,
                current_trend_up=current_trend_up,
                trend_just_changed=trend_just_changed,
                atr=last_atr,
                active_bull_boxes=active_bull_boxes,
                active_bear_boxes=active_bear_boxes,
                vt_trend_ema=vt_ema
                # Optionally include the full DataFrame if needed for debugging
                # df_analyzed=df_analyzed if self.logger.isEnabledFor(logging.DEBUG) else None
            )

            # --- Log Summary ---
            trend_str = "Up" if results.current_trend_up else "Down" if results.current_trend_up is False else "Undetermined"
            atr_str = f"{last_atr:.{8}f}" if last_atr else "N/A"
            self.logger.info(
                f"{FG_GREEN}Strategy Analysis Complete for {symbol}: Trend={trend_str}, "
                f"Close={last_close}, ATR={atr_str}, "
                f"Active Bull OBs={len(active_bull_boxes)}, Active Bear OBs={len(active_bear_boxes)}{RESET}"
            )
            # Log active boxes at DEBUG level as it can be verbose
            if self.logger.isEnabledFor(logging.DEBUG):
                 for box in active_bull_boxes: self.logger.debug(f" -> {box}")
                 for box in active_bear_boxes: self.logger.debug(f" -> {box}")

            return results

        except IndexError:
             self.logger.error(f"{FG_RED}IndexError accessing last row of DataFrame for {symbol}. DataFrame might be empty or too short after processing.{RESET}")
             self.logger.debug(f"Analyzed DF length: {len(df_analyzed)}, Original DF length: {len(df_raw)}")
             return None
        except Exception as e:
             self.logger.error(f"{FG_RED}Error finalizing analysis results for {symbol}: {e}{RESET}", exc_info=True)
             return None
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate strategy.py."; fi
echog " -> strategy.py generated successfully."
echo # Newline

# 6. Create signals.py
echob "Step 6: Generating signals.py..."
cat << 'EOF' > signals.py
# ~/trading-bot/signals.py
import logging
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, List

from .utils import FG_YELLOW, FG_RED, FG_GREEN, FG_BLUE, RESET, BRIGHT
from .config import BotConfig # For type hinting and accessing params
from .strategy import StrategyAnalysisResults, OrderBlock # Import results structure and OB class

class SignalGenerator:
    """
    Generates trading signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD")
    based on the strategy analysis results and the current position state.
    """

    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        # Access strategy and protection parameters directly from the config object
        self.params = config.strategy_params
        self.protection = config.protection

        # Pre-calculate Decimal proximity factors for efficiency and error checking
        try:
            self.entry_factor = Decimal(str(self.params.ob_entry_proximity_factor))
            self.exit_factor = Decimal(str(self.params.ob_exit_proximity_factor))
            if not (self.entry_factor > 1 and self.exit_factor > 1):
                 raise ValueError("Proximity factors must be strictly greater than 1.")
            self.logger.info(f"Signal Generator initialized. Entry Proximity: {self.entry_factor}, Exit Proximity: {self.exit_factor}")
        except (InvalidOperation, ValueError) as e:
             self.logger.error(f"{FG_RED}Invalid proximity factors in config: {e}. Using defaults.{RESET}")
             # Provide safe fallback defaults
             self.entry_factor = Decimal("1.003") # Default 0.3% range tolerance
             self.exit_factor = Decimal("1.001")  # Default 0.1% range tolerance
             self.logger.warning(f"Using default factors - Entry: {self.entry_factor}, Exit: {self.exit_factor}")

    def _is_near_box(self, price: Decimal, box: OrderBlock, factor: Decimal, is_entry_check: bool) -> bool:
        """
        Checks if the given price is 'near' an Order Block, considering its type and proximity factor.

        Args:
            price: The current price (e.g., last close).
            box: The OrderBlock object to check against.
            factor: The proximity factor (Decimal > 1.0) to define the 'near' range.
            is_entry_check: Boolean indicating if this check is for an entry signal (True)
                           or an exit signal (False). Affects which factor might be used internally.

        Returns:
            True if the price is considered near the box, False otherwise.
        """
        if not isinstance(price, Decimal):
            self.logger.warning(f"Invalid price type ({type(price)}) for proximity check.")
            return False
        # Box top/bottom should be Decimal from OrderBlock creation

        try:
            # Determine which factor to use (entry or tighter exit)
            # This logic assumes the passed `factor` is the primary one for the context (entry/exit)
            # but you could override internally if needed.
            effective_factor = factor

            if box.type == 'bull': # Bullish OB (Potential Support)
                # Price needs to be near or inside the box [bottom, top]
                # Check if price is below the top edge extended by the factor, BUT above the bottom edge.
                # Example: Top=100, Factor=1.003 -> Check Top = 100.3
                check_price_high = box.top * effective_factor
                is_near = box.bottom <= price <= check_price_high
                # if is_near: self.logger.debug(f"Price {price} IS near Bull OB [{box.bottom}, {box.top}] (Check Top: {check_price_high})")
                return is_near

            elif box.type == 'bear': # Bearish OB (Potential Resistance)
                # Price needs to be near or inside the box [bottom, top]
                # Check if price is above the bottom edge reduced by the factor, BUT below the top edge.
                # Example: Bottom=100, Factor=1.003 -> Check Low = 100 / 1.003 = 99.7
                check_price_low = box.bottom / effective_factor
                is_near = check_price_low <= price <= box.top
                # if is_near: self.logger.debug(f"Price {price} IS near Bear OB [{box.bottom}, {box.top}] (Check Low: {check_price_low})")
                return is_near

            else:
                self.logger.warning(f"Unknown OrderBlock type encountered: {box.type}")
                return False

        except (TypeError, ValueError, InvalidOperation) as e:
            # Catch potential math errors (e.g., division by zero if factor is invalid)
            self.logger.error(f"{FG_RED}Error during proximity check math: {e}. Price={price}, Box={box}, Factor={factor}{RESET}")
            return False


    def generate_signal(self, results: StrategyAnalysisResults, position: Optional[Dict]) -> str:
        """
        Generates the primary trading signal based on strategy analysis and position status.

        Args:
            results: The StrategyAnalysisResults object from the strategy module.
            position: Current position info dict (from ExchangeManager.fetch_position)
                      or None if no position is open. Expected keys: 'side' ('long'/'short').

        Returns:
            A signal string: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD".
        """
        signal = "HOLD" # Default signal

        # --- Validate Inputs ---
        if results.current_trend_up is None:
            self.logger.debug(f"Cannot generate signal for {results.symbol}: Trend is undetermined.")
            return "HOLD"
        if results.last_close is None or not isinstance(results.last_close, Decimal):
            self.logger.warning(f"{FG_YELLOW}Cannot generate signal for {results.symbol}: Invalid last close price ({results.last_close}).{RESET}")
            return "HOLD"

        last_close = results.last_close
        is_uptrend = results.current_trend_up # True or False

        # --- 1. Check for Exit Signals (if in a position) ---
        if position:
            position_side = position.get('side')
            if not isinstance(position_side, str) or position_side not in ['long', 'short']:
                 self.logger.error(f"{FG_RED}Invalid/missing position side ('{position_side}') for {results.symbol}. Cannot determine exit signal.{RESET}")
                 return "HOLD" # Cannot manage unknown position state

            self.logger.debug(f"Checking exit conditions for {position_side.upper()} position on {results.symbol}...")

            # --- LONG Exit Conditions ---
            if position_side == 'long':
                # Condition L1: Trend flips bearish AND price confirms below VT EMA (more robust)
                if not is_uptrend and results.vt_trend_ema and last_close < results.vt_trend_ema:
                   self.logger.info(f"{FG_YELLOW}{BRIGHT}Signal: EXIT_LONG for {results.symbol} - Trend flipped DOWN & Price < VT EMA ({results.vt_trend_ema:.{8}f}).{RESET}")
                   return "EXIT_LONG"

                # Condition L2: Price nears an active Bearish OB (potential resistance)
                for box in results.active_bear_boxes:
                    # Use the tighter EXIT proximity factor for exits
                    if self._is_near_box(last_close, box, self.exit_factor, is_entry_check=False):
                        self.logger.info(f"{FG_YELLOW}{BRIGHT}Signal: EXIT_LONG for {results.symbol} - Price ({last_close}) near active Bear OB [{box.bottom:.{8}f} - {box.top:.{8f}}].{RESET}")
                        return "EXIT_LONG"

                # If no exit conditions met, HOLD the long position (signal remains "HOLD")

            # --- SHORT Exit Conditions ---
            elif position_side == 'short':
                 # Condition S1: Trend flips bullish AND price confirms above VT EMA
                 if is_uptrend and results.vt_trend_ema and last_close > results.vt_trend_ema:
                    self.logger.info(f"{FG_YELLOW}{BRIGHT}Signal: EXIT_SHORT for {results.symbol} - Trend flipped UP & Price > VT EMA ({results.vt_trend_ema:.{8}f}).{RESET}")
                    return "EXIT_SHORT"

                 # Condition S2: Price nears an active Bullish OB (potential support)
                 for box in results.active_bull_boxes:
                     # Use the tighter EXIT proximity factor
                     if self._is_near_box(last_close, box, self.exit_factor, is_entry_check=False):
                          self.logger.info(f"{FG_YELLOW}{BRIGHT}Signal: EXIT_SHORT for {results.symbol} - Price ({last_close}) near active Bull OB [{box.bottom:.{8}f} - {box.top:.{8f}}].{RESET}")
                          return "EXIT_SHORT"

                 # If no exit conditions met, HOLD the short position (signal remains "HOLD")

        # --- 2. Check for Entry Signals (if NO position) ---
        else: # No position currently open
            self.logger.debug(f"Checking entry conditions for {results.symbol} (No current position)...")

            # --- LONG Entry Condition ---
            # Condition B1: Trend is UP AND Price is near an active Bullish OB
            if is_uptrend:
                for box in results.active_bull_boxes:
                    # Use the potentially wider ENTRY proximity factor
                    if self._is_near_box(last_close, box, self.entry_factor, is_entry_check=True):
                         # Optional Confirmation: Add checks like price bouncing off the OB,
                         # e.g., last candle closed bullishly (close > open) near the box.
                         # if results.last_close > results.last_open:
                         self.logger.info(f"{FG_GREEN}{BRIGHT}Signal: BUY for {results.symbol} - Uptrend & Price ({last_close}) testing active Bull OB [{box.bottom:.{8}f} - {box.top:.{8f}}].{RESET}")
                         signal = "BUY"
                         break # Take the first qualifying BUY signal

            # --- SHORT Entry Condition ---
            # Condition L1: Trend is DOWN AND Price is near an active Bearish OB
            elif not is_uptrend: # Trend is Down
                 for box in results.active_bear_boxes:
                     # Use the ENTRY proximity factor
                     if self._is_near_box(last_close, box, self.entry_factor, is_entry_check=True):
                         # Optional Confirmation: e.g., last candle closed bearishly.
                         # if results.last_close < results.last_open:
                         self.logger.info(f"{FG_RED}{BRIGHT}Signal: SELL for {results.symbol} - Downtrend & Price ({last_close}) testing active Bear OB [{box.bottom:.{8}f} - {box.top:.{8f}}].{RESET}")
                         signal = "SELL"
                         break # Take the first qualifying SELL signal

        # --- Final Logging for HOLD ---
        if signal == "HOLD":
             pos_status = "None"
             if position:
                pos_status = position.get('side', 'Unknown').upper()
             trend_str = "Up" if is_uptrend else "Down"
             self.logger.debug(f"Signal: HOLD for {results.symbol}. Trend: {trend_str}, Close: {last_close}, Position: {pos_status}")

        return signal
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate signals.py."; fi
echog " -> signals.py generated successfully."
echo # Newline

# 7. Create trading.py
echob "Step 7: Generating trading.py..."
cat << 'EOF' > trading.py
# ~/trading-bot/trading.py
import logging
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, Context
from typing import Optional, Dict, Tuple

from .utils import ( safe_decimal, format_value,
                     FG_RED, FG_GREEN, FG_YELLOW, FG_CYAN, FG_BLUE, FG_MAGENTA, RESET, BRIGHT, BG_RED, FG_WHITE )
from .config import BotConfig
from .exchange import ExchangeManager
from .strategy import StrategyAnalysisResults # For accessing ATR, OB levels etc.

class TradeManager:
    """
    Handles the execution logic of trades based on signals:
    - Position sizing based on risk parameters.
    - Calculation of Stop Loss (SL) and Take Profit (TP) levels.
    - Placement of entry orders with attached SL/TP (using Bybit V5 features).
    - Placement of exit orders (market close).
    - Basic framework for position management (Trailing Stop Loss, Break Even - requires further implementation).
    """

    def __init__(self, config: BotConfig, logger: logging.Logger, exchange_manager: ExchangeManager):
        self.config = config
        self.logger = logger
        self.exchange = exchange_manager
        # Access strategy and protection parameters conveniently
        self.params = config.strategy_params
        self.protection = config.protection

        # --- State Tracking (Example) ---
        # Store active SL/TP order IDs if NOT using attached orders (more complex)
        # self.active_orders: Dict[str, Dict[str, Optional[str]]] = {} # e.g., {'BTC/USDT': {'sl_id': '123', 'tp_id': '456'}}
        # Track if BreakEven has been activated for a position
        self.break_even_active: Dict[str, bool] = {} # e.g., {'BTC/USDT': True}
        # Track highest/lowest price seen for Trailing Stop Loss
        self.tsl_watermark: Dict[str, Optional[Decimal]] = {} # e.g., {'BTC/USDT': Decimal('30500.50')}

        self.logger.info(f"Trade Manager initialized. Trading Enabled: {config.enable_trading}")
        if not config.enable_trading:
            self.logger.warning(f"{FG_YELLOW}Trading is currently DISABLED in the configuration.{RESET}")
        if self.protection.enable_trailing_stop:
             self.logger.info(f"Trailing Stop Loss enabled (Activation: {self.protection.trailing_stop_activation_percentage*100}%, Callback: {self.protection.trailing_stop_callback_rate*100}%). Requires SL modification capability.")
        if self.protection.enable_break_even:
             self.logger.info(f"Break Even enabled (Trigger ATR: {self.protection.break_even_trigger_atr_multiple}, Offset Ticks: {self.protection.break_even_offset_ticks}). Requires SL modification capability.")


    def _calculate_position_size(self, symbol: str, balance: Decimal, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """
        Calculates the position size in the base currency (e.g., BTC for BTC/USDT)
        based on the configured risk percentage, balance, entry price, and stop loss price.
        Assumes linear contracts (value calculated in quote currency, size in base currency).

        Args:
            symbol: The trading pair symbol.
            balance: Current available balance in the quote currency (Decimal).
            entry_price: Estimated entry price (Decimal).
            stop_loss_price: Calculated stop loss price (Decimal).

        Returns:
            The calculated position size in base currency (Decimal), or None if calculation fails.
        """
        self.logger.debug(f"Calculating position size for {symbol}...")

        # --- Input Validation ---
        if balance <= 0:
            self.logger.warning(f"{FG_YELLOW}Cannot calculate size: Balance is zero or negative ({balance} {self.config.quote_currency}).{RESET}")
            return None
        if entry_price <= 0 or stop_loss_price <= 0:
             self.logger.warning(f"{FG_YELLOW}Cannot calculate size: Entry ({entry_price}) or SL price ({stop_loss_price}) is zero or negative.{RESET}")
             return None
        # Ensure prices are Decimal (should be, but double-check)
        if not all(isinstance(p, Decimal) for p in [balance, entry_price, stop_loss_price]):
            self.logger.error(f"{FG_RED}Invalid input types for size calculation: Bal={type(balance)}, Entry={type(entry_price)}, SL={type(stop_loss_price)}{RESET}")
            return None

        risk_fraction = safe_decimal(self.config.risk_per_trade, "risk_per_trade", logger=self.logger)
        if risk_fraction is None or risk_fraction <= 0:
            self.logger.error(f"{FG_RED}Invalid risk_per_trade setting ({self.config.risk_per_trade}). Cannot calculate size.{RESET}")
            return None

        # --- Risk Calculation ---
        # Risk amount in quote currency (e.g., USDT)
        risk_amount_quote = balance * risk_fraction
        self.logger.debug(f" -> Risk Amount ({risk_fraction*100}% of {balance:.4f}): {risk_amount_quote:.4f} {self.config.quote_currency}")

        # Price difference per unit of base currency (risk per unit)
        price_diff = abs(entry_price - stop_loss_price)
        if price_diff == 0:
            self.logger.warning(f"{FG_YELLOW}Cannot calculate size: Entry price equals Stop Loss price ({entry_price}). Risk is zero.{RESET}")
            return None
        self.logger.debug(f" -> Price Difference (Entry-SL): {price_diff:.{8}f}")

        # --- Position Size Calculation (Linear Contracts) ---
        # Size (Base Currency) = Risk Amount (Quote Currency) / Price Difference (Quote/Base)
        try:
            position_size_base = risk_amount_quote / price_diff
            self.logger.debug(f" -> Calculated Raw Size (Base): {position_size_base:.{8}f}")
        except (InvalidOperation, ZeroDivisionError) as e:
            self.logger.error(f"{FG_RED}Decimal calculation error during size calculation: {e} (Risk: {risk_amount_quote}, Diff: {price_diff}){RESET}")
            return None

        # --- Apply Market Constraints and Precision ---
        market_info = self.exchange.get_market_info(symbol)
        if not market_info:
            self.logger.error(f"{FG_RED}Cannot finalize position size: Failed to get market info for {symbol}.{RESET}")
            return None

        _, amount_prec = self.exchange.get_precision(symbol)
        # Contract size adjustment (usually 1 for linear USDT, but important for others)
        contract_size = self.exchange.get_contract_size(symbol) # Should be Decimal(1) for BTC/USDT linear
        if contract_size is None or contract_size <= 0: contract_size = Decimal(1) # Safety fallback

        # If order amount needs to be in contracts instead of base currency units (less common for linear):
        # position_size_contracts = position_size_base / contract_size
        # For linear, amount is typically base currency (e.g., BTC)
        position_size_final = position_size_base

        # Apply amount precision (rounding DOWN to avoid over-spending/size issues)
        formatted_size_str = format_value(position_size_final, amount_prec, rounding=ROUND_DOWN)
        position_size_formatted = safe_decimal(formatted_size_str, "formatted size", logger=self.logger)

        if position_size_formatted is None or position_size_formatted <= 0:
             self.logger.warning(f"{FG_YELLOW}Calculated position size is zero or invalid after applying precision ({amount_prec} decimals). Original: {position_size_final:.{amount_prec+2}f}. Cannot place trade.{RESET}")
             return None
        self.logger.debug(f" -> Size after Precision ({amount_prec} dec): {position_size_formatted}")

        # Check against market limits (minimum and maximum order size)
        min_amount, max_amount = None, None
        limits = market_info.get('limits', {}).get('amount', {})
        min_amount_str = limits.get('min')
        max_amount_str = limits.get('max')
        if min_amount_str: min_amount = safe_decimal(min_amount_str, "min order amount", logger=self.logger)
        if max_amount_str: max_amount = safe_decimal(max_amount_str, "max order amount", logger=self.logger)

        # Check minimum
        if min_amount is not None and position_size_formatted < min_amount:
             base_currency = market_info.get('base', 'BASE')
             self.logger.warning(f"{FG_YELLOW}Calculated size {position_size_formatted} {base_currency} is below market minimum {min_amount} {base_currency}. Cannot place trade.{RESET}")
             return None

        # Check maximum and cap if necessary
        if max_amount is not None and position_size_formatted > max_amount:
            self.logger.warning(f"{FG_YELLOW}Calculated size {position_size_formatted} exceeds market maximum {max_amount}. Capping size to maximum.{RESET}")
            # Re-format the max amount with correct precision
            formatted_max_str = format_value(max_amount, amount_prec, rounding=ROUND_DOWN)
            position_size_formatted = safe_decimal(formatted_max_str, "capped max size", logger=self.logger)
            if position_size_formatted is None or position_size_formatted <= 0:
                 self.logger.error(f"{FG_RED}Failed to format capped maximum size. Cannot place trade.{RESET}")
                 return None # Should not happen if max_amount was valid

        # --- Log Final Size ---
        quote_currency = self.config.quote_currency
        base_currency = market_info.get('base', '')
        self.logger.info(f"Final calculated position size for {symbol}: {position_size_formatted} {base_currency}")
        return position_size_formatted


    def _determine_trade_parameters(self, symbol: str, side: str, results: StrategyAnalysisResults) -> Optional[Dict]:
        """
        Calculates estimated Entry Price, Stop Loss (SL), and Take Profit (TP) levels.
        Uses ATR for initial SL/TP placement relative to the estimated entry.

        Args:
            symbol: Trading pair symbol.
            side: 'long' or 'short'.
            results: The StrategyAnalysisResults containing latest close, ATR, etc.

        Returns:
            A dictionary with 'entry_price', 'stop_loss_price', 'take_profit_price' (Decimal or None),
            or None if parameters cannot be determined.
        """
        self.logger.debug(f"Determining trade parameters for {symbol} ({side})...")

        # --- Extract necessary data from results ---
        last_close = results.last_close # Should be Decimal
        atr = results.atr             # Should be Decimal or None
        # Get protection parameters as Decimals
        sl_atr_multiple = safe_decimal(self.protection.initial_stop_loss_atr_multiple, "SL ATR multiple")
        tp_atr_multiple = safe_decimal(self.protection.initial_take_profit_atr_multiple, "TP ATR multiple")

        # --- Input Validation ---
        if last_close is None or last_close <= 0:
             self.logger.warning(f"{FG_YELLOW}Cannot determine parameters: Invalid last close price ({last_close}).{RESET}")
             return None
        if atr is None or atr <= 0:
            self.logger.warning(f"{FG_YELLOW}Cannot determine parameters: Invalid ATR value ({atr}). Check strategy calculation.{RESET}")
            return None
        if sl_atr_multiple is None or sl_atr_multiple <= 0:
             self.logger.error(f"{FG_RED}Invalid Initial Stop Loss ATR multiple ({self.protection.initial_stop_loss_atr_multiple}). Cannot determine parameters.{RESET}")
             return None
        # Allow TP multiple to be zero or negative (effectively disabling initial TP)
        if tp_atr_multiple is not None and tp_atr_multiple <= 0:
            self.logger.info(f"Initial Take Profit ATR multiple ({tp_atr_multiple}) is non-positive. Initial TP disabled.")
            tp_atr_multiple = None # Disable TP calculation

        # --- Get Price Precision ---
        price_prec, _ = self.exchange.get_precision(symbol)
        quantizer = Decimal('1e-' + str(price_prec)) # For rounding prices

        # --- Entry Price Estimation ---
        # For market orders, the best estimate is the last close price.
        # For limit orders, the entry price would be the limit price itself.
        # Assume market entry for this calculation.
        entry_price = last_close
        self.logger.debug(f" -> Estimated Entry Price (Last Close): {entry_price}")

        # --- Stop Loss Calculation ---
        sl_distance = atr * sl_atr_multiple
        self.logger.debug(f" -> SL Distance (ATR * {sl_atr_multiple}): {sl_distance:.{price_prec+2}f}")

        stop_loss_price = None
        if side.lower() == "long":
            stop_loss_price = entry_price - sl_distance
            # Optional Enhancement: Adjust SL based on nearby Bullish OB
            # Find closest active Bull OB below entry:
            # relevant_boxes = [b for b in results.active_bull_boxes if b.bottom < entry_price]
            # if relevant_boxes:
            #     closest_box = min(relevant_boxes, key=lambda b: entry_price - b.top) # Closest top edge
            #     # Place SL slightly below the OB's bottom
            #     ob_sl = closest_box.bottom - (atr * Decimal('0.1')) # e.g., 0.1 * ATR below bottom
            #     # Use the SL that is further away (more conservative)
            #     stop_loss_price = min(stop_loss_price, ob_sl)
            #     self.logger.debug(f" -> Adjusted SL based on Bull OB {closest_box.timestamp}: {stop_loss_price}")

        elif side.lower() == "sell":
            stop_loss_price = entry_price + sl_distance
            # Optional Enhancement: Adjust SL based on nearby Bearish OB
            # Find closest active Bear OB above entry:
            # relevant_boxes = [b for b in results.active_bear_boxes if b.top > entry_price]
            # if relevant_boxes:
            #     closest_box = min(relevant_boxes, key=lambda b: b.bottom - entry_price) # Closest bottom edge
            #     # Place SL slightly above the OB's top
            #     ob_sl = closest_box.top + (atr * Decimal('0.1')) # e.g., 0.1 * ATR above top
            #     # Use the SL that is further away (more conservative)
            #     stop_loss_price = max(stop_loss_price, ob_sl)
            #     self.logger.debug(f" -> Adjusted SL based on Bear OB {closest_box.timestamp}: {stop_loss_price}")

        # Quantize SL price - round *away* from entry to be safer (gives more room)
        sl_rounding = ROUND_DOWN if side.lower() == "long" else ROUND_UP
        stop_loss_price = stop_loss_price.quantize(quantizer, rounding=sl_rounding)
        self.logger.debug(f" -> Calculated SL Price (Quantized): {stop_loss_price}")

        # --- Take Profit Calculation ---
        take_profit_price = None
        if tp_atr_multiple is not None:
            tp_distance = atr * tp_atr_multiple
            self.logger.debug(f" -> TP Distance (ATR * {tp_atr_multiple}): {tp_distance:.{price_prec+2}f}")
            if side.lower() == "long":
                take_profit_price = entry_price + tp_distance
            elif side.lower() == "sell":
                take_profit_price = entry_price - tp_distance

            # Quantize TP price - round *towards* entry (ensures TP is reachable)
            tp_rounding = ROUND_DOWN if side.lower() == "long" else ROUND_UP
            take_profit_price = take_profit_price.quantize(quantizer, rounding=tp_rounding)
            self.logger.debug(f" -> Calculated TP Price (Quantized): {take_profit_price}")

        # --- Final Validation ---
        if stop_loss_price <= 0:
             self.logger.warning(f"{FG_YELLOW}Invalid SL price ({stop_loss_price}) calculated (zero or negative). Cannot proceed.{RESET}")
             return None
        if take_profit_price is not None and take_profit_price <= 0:
             self.logger.warning(f"{FG_YELLOW}Invalid TP price ({take_profit_price}) calculated (zero or negative). Setting TP to None.{RESET}")
             take_profit_price = None
        # Check SL is on the correct side of entry
        if (side.lower() == "long" and stop_loss_price >= entry_price) or \
           (side.lower() == "sell" and stop_loss_price <= entry_price):
            self.logger.error(f"{FG_RED}Stop loss price ({stop_loss_price}) is on the WRONG side of entry ({entry_price}) for {side} trade. Check calculation/ATR. Cannot proceed.{RESET}")
            return None
        # Check TP is on the correct side (if set)
        if take_profit_price is not None and \
           ((side.lower() == "long" and take_profit_price <= entry_price) or \
            (side.lower() == "sell" and take_profit_price >= entry_price)):
             self.logger.warning(f"{FG_YELLOW}Take profit price ({take_profit_price}) is on the WRONG side of entry ({entry_price}). Setting TP to None.{RESET}")
             take_profit_price = None

        # --- Return Parameters ---
        params = {
            "entry_price": entry_price,       # Estimated entry for market orders
            "stop_loss_price": stop_loss_price, # Calculated SL
            "take_profit_price": take_profit_price, # Calculated TP (can be None)
        }
        tp_log = f"{take_profit_price}" if take_profit_price else "None"
        self.logger.info(f"Determined Trade Parameters for {symbol} {side.upper()}: Entry(Est)={entry_price}, SL={stop_loss_price}, TP={tp_log}")
        return params

    def _execute_entry(self, symbol: str, side: str, size: Decimal, sl_price: Decimal, tp_price: Optional[Decimal]) -> Optional[Dict]:
        """
        Executes the entry trade sequence:
        1. Places the main market order.
        2. Attaches SL and TP orders simultaneously using Bybit V5 parameters if possible.
        3. Waits briefly and confirms the position was opened.

        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            size: Position size in base currency (Decimal).
            sl_price: Stop loss price (Decimal).
            tp_price: Take profit price (Decimal or None).

        Returns:
            The CCXT order dictionary for the entry order if successful and confirmed, None otherwise.
        """
        if not self.config.enable_trading:
             self.logger.warning(f"{FG_MAGENTA}Trading Disabled: Would execute {side.upper()} entry for {size} {symbol}.{RESET}")
             return None # Simulate success when disabled

        self.logger.info(f"{FG_MAGENTA}{BRIGHT}=== Executing {side.upper()} Entry for {size} {symbol} ==={RESET}")
        self.logger.info(f" -> Target SL: {sl_price}, Target TP: {tp_price if tp_price else 'None'}")

        # --- Prepare Order Parameters with Attached SL/TP ---
        price_prec, amount_prec = self.exchange.get_precision(symbol)
        # Format SL/TP prices to strings with correct precision for the API params
        sl_price_str = format_value(sl_price, price_prec)
        tp_price_str = format_value(tp_price, price_prec) if tp_price is not None else None

        entry_order_params = {
             'category': 'linear', # Ensure category is set for V5
             'positionIdx': 0,     # Assume One-Way mode
             # --- Attach SL/TP using Bybit V5 parameters ---
             'stopLoss': sl_price_str,
             'slTriggerBy': 'MarkPrice', # Common choice: MarkPrice, LastPrice, IndexPrice
             # 'slOrderType': 'Market', # Default SL trigger type is Market
         }
        if tp_price_str:
             entry_order_params['takeProfit'] = tp_price_str
             entry_order_params['tpTriggerBy'] = 'MarkPrice' # Usually match SL trigger
             # 'tpOrderType': 'Market', # Default TP trigger type is Market

        # Optional: TimeInForce for market orders (e.g., 'IOC' or 'FOK')
        # Bybit default for market usually works fine.
        # entry_order_params['timeInForce'] = 'ImmediateOrCancel'

        # --- Place Entry Market Order ---
        # Pass the Decimal size object; place_order handles final formatting now
        entry_order = self.exchange.place_order(symbol, side, 'market', size, params=entry_order_params)

        if not entry_order or not entry_order.get('id'):
             self.logger.error(f"{FG_RED}Failed to place entry market order for {symbol}. Check exchange logs/status.{RESET}")
             # No order ID means it definitely failed.
             return None

        entry_order_id = entry_order.get('id')
        entry_order_status = entry_order.get('status', 'unknown').lower()
        self.logger.info(f"Entry order {entry_order_id} submitted. Initial Status: '{entry_order_status}'.")

        # --- Position Confirmation ---
        self.logger.info(f"Waiting {self.config.position_confirm_delay_seconds}s for position confirmation...")
        time.sleep(self.config.position_confirm_delay_seconds)

        self.logger.debug(f"Fetching position for {symbol} to confirm entry...")
        position = self.exchange.fetch_position(symbol)
        expected_size_decimal = size # The size we intended to open

        position_confirmed = False
        actual_entry_price = None
        actual_size = None

        if position:
             pos_side = position.get('side')
             pos_size = position.get('size_contracts') # This should be Decimal
             entry_p = position.get('entry_price')     # Should be Decimal

             if pos_side == side.lower() and pos_size is not None and entry_p is not None:
                 # Compare actual size with expected size, allowing for tiny tolerance
                 # Calculate tolerance based on amount precision
                 tolerance = Decimal('1e-' + str(amount_prec)) / 2 # Half a step tolerance
                 size_diff = abs(pos_size - expected_size_decimal)

                 if size_diff <= tolerance:
                     position_confirmed = True
                     actual_entry_price = entry_p
                     actual_size = pos_size
                     self.logger.info(f"{FG_GREEN}Position Confirmed: {side.upper()} {actual_size} {symbol} @ {actual_entry_price}.{RESET}")
                 else:
                     self.logger.warning(f"{FG_YELLOW}Position size mismatch! Expected: ~{expected_size_decimal}, Actual: {pos_size} (Diff: {size_diff}). Tolerance: {tolerance}{RESET}")
                     # Consider this a potential issue - maybe close immediately? Log for now.
                     # self.execute_exit(symbol, pos_side, pos_size, reason="Size Mismatch on Entry")
             else:
                  self.logger.warning(f"{FG_YELLOW}Position side/size/price mismatch or missing after entry attempt. Expected Side: {side}, Pos: {position}{RESET}")
        else:
             self.logger.warning(f"{FG_YELLOW}Position not found for {symbol} after placing order {entry_order_id} and waiting.{RESET}")

        # --- Handle Confirmation Failure ---
        if not position_confirmed:
            self.logger.error(f"{FG_RED}Position confirmation FAILED for {symbol} after placing order {entry_order_id}.")
            # Try to get final status of the entry order to understand why
            try:
                self.logger.debug(f"Fetching final status for order {entry_order_id}...")
                # Use fetch_order which requires the ID
                order_info = self.exchange.exchange.fetch_order(entry_order_id, symbol, params={'category': 'linear'})
                final_status = order_info.get('status', 'unknown').lower()
                filled_amount = order_info.get('filled', 0)
                self.logger.error(f"Entry Order {entry_order_id} Final Status: '{final_status}', Filled: {filled_amount}.")
                # If order was rejected or cancelled, it explains why no position exists.
                # If it's filled/partially filled but position check failed, it's a bigger issue.
                if final_status in ['filled', 'partially_filled'] and not position:
                     self.logger.critical(f"{BG_RED}{FG_WHITE}CRITICAL STATE: Order {entry_order_id} shows filled but position fetch failed. Manual intervention required!{RESET}")
                # Optional: Attempt to cancel if stuck in 'open' or similar non-final state? Risky.
                # if final_status in ['open', 'new']:
                #      self.logger.warning(f"Entry order {entry_order_id} seems stuck. Attempting cancel...")
                #      self.exchange.cancel_order(entry_order_id, symbol)
            except ccxt.OrderNotFound:
                 self.logger.warning(f"Entry order {entry_order_id} not found when checking final status (might have been instantly rejected/filled & cleaned).")
            except Exception as e:
                self.logger.error(f"Could not fetch final status for failed entry order {entry_order_id}: {e}")
            return None # Indicate entry failure

        # --- Post-Confirmation Actions ---
        # Reset management state for the new position
        self.break_even_active.pop(symbol, None)
        self.tsl_watermark.pop(symbol, None)
        # Store the SL price that was *intended* for this position (useful for TSL/BE)
        # self.active_sl_price[symbol] = sl_price # Example state tracking

        self.logger.info(f"{FG_GREEN}Entry sequence complete for {symbol}. Attached SL/TP should be managed by Bybit.{RESET}")
        return entry_order # Return the original entry order details on success


    def execute_exit(self, symbol: str, position_side: str, position_size: Decimal, reason: str = "Signal") -> bool:
        """
        Closes the entire current position using a market order with reduceOnly=True.
        Assumes attached SL/TP orders are automatically cancelled by Bybit when the position is closed.

        Args:
            symbol: Trading pair symbol.
            position_side: 'long' or 'short' (the side of the position being closed).
            position_size: The exact size of the position to close (Decimal, base currency).
            reason: A string explaining why the exit is being executed (for logging).

        Returns:
            True if the close order was placed successfully, False otherwise.
        """
        if not self.config.enable_trading:
            self.logger.warning(f"{FG_MAGENTA}Trading Disabled: Would execute {position_side.upper()} exit for {position_size} {symbol} (Reason: {reason}).{RESET}")
            # Clear state even in simulation mode if needed
            self.break_even_active.pop(symbol, None)
            self.tsl_watermark.pop(symbol, None)
            return True # Simulate success

        self.logger.info(f"{FG_MAGENTA}{BRIGHT}=== Executing {position_side.upper()} Exit for {position_size} {symbol} (Reason: {reason}) ==={RESET}")

        # Determine the side of the closing order (opposite of position side)
        exit_side = "sell" if position_side.lower() == "long" else "buy"

        # --- Prepare Close Order Parameters ---
        # Use reduceOnly=True to ensure the order only closes the existing position
        # and doesn't accidentally open a new one if the size is slightly off.
        close_order_params = {
            'category': 'linear',
            'reduceOnly': True,
            'positionIdx': 0 # Assume One-Way mode
        }

        # --- Place Market Order to Close ---
        # Pass the exact position size (should be Decimal)
        close_order = self.exchange.place_order(
            symbol,
            exit_side,
            'market',
            position_size, # Pass the exact Decimal size to close
            params=close_order_params
        )

        # --- Handle Close Order Result ---
        if close_order and close_order.get('id'):
            self.logger.info(f"{FG_GREEN}Position close order ({close_order.get('id')}) placed successfully for {symbol}. Status: {close_order.get('status', 'unknown')}.{RESET}")
            # Bybit should automatically cancel associated SL/TP orders when a position is closed by a reduceOnly market order.

            # Clear any related internal state tracking for this position
            self.break_even_active.pop(symbol, None)
            self.tsl_watermark.pop(symbol, None)
            # self.active_orders.pop(symbol, None) # Clear if tracking separate SL/TP IDs

            # Optional short delay to allow exchange state to update before next cycle checks position
            time.sleep(self.config.position_confirm_delay_seconds / 4)
            return True
        else:
            # This is a critical failure - the bot tried to close but couldn't.
            self.logger.error(f"{BG_RED}{FG_WHITE}CRITICAL: Failed to place position close order for {symbol}. Manual intervention required! Check position on exchange.{RESET}")
            # Do NOT clear internal state if close failed, as the position might still be open.
            return False


    def _manage_trailing_stop(self, position: Dict, current_mark_price: Decimal):
        """
        Manages the Trailing Stop Loss (TSL) logic.
        NOTE: This is a basic conceptual implementation. Modifying attached SL orders
              on Bybit V5 via API can be complex and might require cancel/replace logic
              if direct modification isn't reliably supported by CCXT or the API for attached orders.

        Args:
            position: The current position dictionary.
            current_mark_price: The current mark price (Decimal), often used for SL triggers.
        """
        if not self.protection.enable_trailing_stop: return
        symbol = position.get('symbol')
        position_side = position.get('side')
        entry_price = position.get('entry_price')
        # We need the currently active SL price to compare against.
        # Fetching this reliably for attached orders is the main challenge.
        # current_sl_price = self._get_current_stop_loss(symbol, position) # Placeholder for complex fetch logic

        # --- Simplified Logic (Assumes we *could* get current SL & modify) ---
        if not all([symbol, position_side, entry_price, isinstance(current_mark_price, Decimal)]):
            self.logger.warning(f"{FG_YELLOW}Missing data for TSL check on {symbol}.{RESET}")
            return

        # --- TSL Activation Check ---
        profit_pct = Decimal(0)
        if position_side == 'long':
            profit_pct = (current_mark_price / entry_price) - 1
        elif position_side == 'short':
            profit_pct = (entry_price / current_mark_price) - 1

        activation_pct = safe_decimal(self.protection.trailing_stop_activation_percentage, "TSL Activation %")
        if activation_pct is None or profit_pct < activation_pct:
            # self.logger.debug(f"TSL for {symbol} not activated. Profit: {profit_pct:.4f}, Activation: {activation_pct}")
            return # Not profitable enough

        # --- Update High/Low Watermark ---
        current_watermark = self.tsl_watermark.get(symbol)
        new_watermark = current_watermark
        if position_side == 'long':
            if new_watermark is None or current_mark_price > new_watermark:
                new_watermark = current_mark_price
        elif position_side == 'short':
            if new_watermark is None or current_mark_price < new_watermark:
                new_watermark = current_mark_price

        if new_watermark != current_watermark:
            self.logger.debug(f"TSL watermark for {symbol} updated to {new_watermark}")
            self.tsl_watermark[symbol] = new_watermark
        elif current_watermark is None:
             # Should not happen if activation is met, but safety check
             self.logger.warning(f"TSL activated but watermark is still None for {symbol}.")
             return


        # --- Calculate New Potential TSL Price ---
        callback_rate = safe_decimal(self.protection.trailing_stop_callback_rate, "TSL Callback Rate")
        if callback_rate is None: return

        potential_tsl_price = None
        if position_side == 'long':
            potential_tsl_price = new_watermark * (Decimal(1) - callback_rate)
        elif position_side == 'short':
            potential_tsl_price = new_watermark * (Decimal(1) + callback_rate)

        # Quantize the potential TSL price
        price_prec, _ = self.exchange.get_precision(symbol)
        quantizer = Decimal('1e-' + str(price_prec))
        sl_rounding = ROUND_DOWN if position_side == 'long' else ROUND_UP # Round away from current price
        potential_tsl_price = potential_tsl_price.quantize(quantizer, rounding=sl_rounding)

        # --- Compare with Current SL and Modify (Requires Implementation) ---
        self.logger.warning(f"{FG_YELLOW}TSL Check for {symbol}: Activation Met (Profit: {profit_pct:.4f}). Potential New SL: {potential_tsl_price}. "
                           f"Fetching current attached SL and modifying it is NOT YET IMPLEMENTED.{RESET}")

        # Pseudocode for required logic:
        # 1. current_sl_price = self.exchange.fetch_current_stop_loss_for_position(symbol) # Complex!
        # 2. if current_sl_price is None: return # Cannot proceed
        # 3. should_update = False
        # 4. if position_side == 'long' and potential_tsl_price > current_sl_price: should_update = True
        # 5. elif position_side == 'short' and potential_tsl_price < current_sl_price: should_update = True
        # 6. if should_update:
        # 7.    self.logger.info(f"{FG_CYAN}TSL Update: Moving SL for {symbol} from {current_sl_price} to {potential_tsl_price}{RESET}")
        # 8.    success = self.exchange.modify_position_stop_loss(symbol, potential_tsl_price) # Complex! (Might need cancel/replace)
        # 9.    if not success: self.logger.error("Failed to modify SL for TSL.")


    def _manage_break_even(self, position: Dict, current_mark_price: Decimal, atr: Optional[Decimal]):
        """
        Manages the Break Even (BE) stop adjustment logic.
        Moves the SL to entry + offset once price reaches a certain ATR multiple in profit.
        NOTE: Requires reliable fetching/modification of the current SL, similar to TSL.
        """
        if not self.protection.enable_break_even: return
        symbol = position.get('symbol')

        # Check if BE already activated for this symbol/position instance
        if self.break_even_active.get(symbol, False):
            # self.logger.debug(f"BE already active for {symbol}.")
            return

        position_side = position.get('side')
        entry_price = position.get('entry_price')
        # current_sl_price = self._get_current_stop_loss(symbol, position) # Placeholder

        if not all([symbol, position_side, entry_price, isinstance(current_mark_price, Decimal), isinstance(atr, Decimal), atr > 0]):
            self.logger.warning(f"{FG_YELLOW}Missing data or invalid ATR for BE check on {symbol}.{RESET}")
            return

        # --- BE Activation Check ---
        trigger_atr_multiple = safe_decimal(self.protection.break_even_trigger_atr_multiple, "BE Trigger Multiple")
        if trigger_atr_multiple is None: return

        trigger_distance = atr * trigger_atr_multiple
        activated = False
        if position_side == 'long' and current_mark_price >= (entry_price + trigger_distance):
            activated = True
        elif position_side == 'short' and current_mark_price <= (entry_price - trigger_distance):
            activated = True

        if not activated:
            # self.logger.debug(f"BE not activated for {symbol}. Mark: {current_mark_price}, Entry: {entry_price}, TriggerDist: {trigger_distance}")
            return

        # --- Calculate New BE Stop Price ---
        # Get tick size for precise offset calculation
        market_info = self.exchange.get_market_info(symbol)
        tick_size = Decimal('1e-8') # Default small tick size
        if market_info and 'precision' in market_info and 'price' in market_info['precision']:
             price_prec_val = market_info['precision']['price']
             try:
                 # Handle tick size specified as string '0.01' or decimal places integer
                 if isinstance(price_prec_val, str) and '.' in price_prec_val:
                     tick_size = Decimal(price_prec_val)
                 elif isinstance(price_prec_val, (int, float)) and price_prec_val >= 0:
                      tick_size = Decimal('1e-' + str(int(price_prec_val)))
                 self.logger.debug(f"Using tick size {tick_size} for BE offset.")
             except Exception:
                 self.logger.warning(f"Could not determine tick size accurately for BE offset. Using default {tick_size}.")
        else:
            self.logger.warning("Market info/precision missing for BE tick size calculation.")


        offset_ticks = self.protection.break_even_offset_ticks
        offset_amount = tick_size * offset_ticks

        new_be_price = None
        if position_side == 'long':
            new_be_price = entry_price + offset_amount
        elif position_side == 'short':
            new_be_price = entry_price - offset_amount

        # Quantize BE price - round slightly *into* profit
        price_prec, _ = self.exchange.get_precision(symbol)
        quantizer = Decimal('1e-' + str(price_prec))
        be_rounding = ROUND_UP if position_side == 'long' else ROUND_DOWN
        new_be_price = new_be_price.quantize(quantizer, rounding=be_rounding)

        # --- Compare with Current SL and Modify (Requires Implementation) ---
        self.logger.warning(f"{FG_YELLOW}Break Even Check for {symbol}: Activation Met. Potential New SL: {new_be_price}. "
                           f"Fetching current attached SL and modifying it is NOT YET IMPLEMENTED.{RESET}")

        # Pseudocode for required logic:
        # 1. current_sl_price = self.exchange.fetch_current_stop_loss_for_position(symbol) # Complex!
        # 2. if current_sl_price is None: return
        # 3. should_update_be = False
        # 4. if position_side == 'long' and new_be_price > current_sl_price: should_update_be = True
        # 5. elif position_side == 'short' and new_be_price < current_sl_price: should_update_be = True
        # 6. if should_update_be:
        # 7.    self.logger.info(f"{FG_CYAN}{BRIGHT}Break Even Triggered: Moving SL for {symbol} from {current_sl_price} to {new_be_price}{RESET}")
        # 8.    success = self.exchange.modify_position_stop_loss(symbol, new_be_price) # Complex!
        # 9.    if success: self.break_even_active[symbol] = True # Mark BE as active only if modification succeeds
        # 10.   else: self.logger.error("Failed to modify SL for Break Even.")
        # 11.else:
        # 12.   self.logger.debug(f"BE triggered for {symbol}, but new price {new_be_price} is not better than current SL {current_sl_price}. No update needed.")

        # Mark BE as conceptually active (even if SL not modified yet) to prevent re-triggering log spam
        self.break_even_active[symbol] = True


    def process_signal(self, symbol: str, signal: str, results: StrategyAnalysisResults, position: Optional[Dict], balance: Decimal):
        """
        Processes the generated signal for a given symbol.
        Handles entry, exit, and calls position management functions.

        Args:
            symbol: The trading pair symbol.
            signal: The signal string ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD").
            results: The StrategyAnalysisResults object.
            position: The current position dictionary (or None).
            balance: Current available quote currency balance (Decimal).
        """
        self.logger.debug(f"Processing signal '{signal}' for {symbol}. Position: {'Yes (' + position['side'] + ')' if position else 'No'}")

        # --- 1. Handle Exit Signals ---
        if position:
            position_side = position.get('side')
            position_size = position.get('size_contracts') # Should be Decimal

            # Validate position data before attempting exit
            if not position_side or not isinstance(position_size, Decimal) or position_size <= 0:
                 self.logger.error(f"{FG_RED}Invalid position data for {symbol}: Side={position_side}, Size={position_size}. Cannot process exit signal.{RESET}")
                 return # Skip processing if position data is corrupt

            if (signal == "EXIT_LONG" and position_side == "long") or \
               (signal == "EXIT_SHORT" and position_side == "short"):
                # Execute exit - function handles trading enabled check
                self.execute_exit(symbol, position_side, position_size, reason="Signal")
                return # Exit processed, end processing for this symbol cycle

        # --- 2. Handle Entry Signals (Only if No Position) ---
        elif not position:
            if signal == "BUY" or signal == "SELL":
                # Determine side based on signal
                entry_side = "long" if signal == "BUY" else "short"

                # Check balance before proceeding (relevant if trading enabled)
                if self.config.enable_trading and (balance is None or balance <= 0):
                    self.logger.warning(f"{FG_YELLOW}Skipping {entry_side} entry for {symbol}: Insufficient or invalid balance ({balance}).{RESET}")
                    return

                # a) Determine SL/TP parameters
                trade_params = self._determine_trade_parameters(symbol, entry_side, results)
                if not trade_params:
                     self.logger.warning(f"{FG_YELLOW}Could not determine valid trade parameters for {symbol} {entry_side}. Skipping entry.{RESET}")
                     return

                # b) Calculate Position Size
                # Pass the balance needed for calculation
                pos_size = self._calculate_position_size(
                    symbol,
                    balance if balance is not None else Decimal(0), # Pass 0 if balance check failed but trading disabled
                    trade_params['entry_price'], # Use estimated entry from params
                    trade_params['stop_loss_price']
                )
                if not pos_size or pos_size <= 0:
                     self.logger.warning(f"{FG_YELLOW}Could not calculate valid position size for {symbol} {entry_side}. Skipping entry.{RESET}")
                     return

                # c) Execute Entry
                # Function handles trading enabled check internally
                self._execute_entry(
                    symbol,
                    entry_side,
                    pos_size, # Pass the calculated Decimal size
                    trade_params['stop_loss_price'],
                    trade_params.get('take_profit_price') # Pass TP price (can be None)
                )
                return # Entry processed, end processing for this symbol cycle

        # --- 3. Handle HOLD Signal ---
        if signal == "HOLD" and position:
            # If holding a position, run management functions
            current_mark_price = position.get('mark_price') # Use mark price for TSL/BE if available
            if not isinstance(current_mark_price, Decimal):
                # Fallback to last close if mark price isn't available in position data
                current_mark_price = results.last_close

            if current_mark_price:
                # Pass the full position dict, current price, and ATR
                self._manage_trailing_stop(position, current_mark_price)
                self._manage_break_even(position, current_mark_price, results.atr)
            else:
                 self.logger.warning(f"Cannot run position management for {symbol}: Missing current mark/close price.")

        # If signal is HOLD and no position, do nothing.
        elif signal == "HOLD" and not position:
             self.logger.debug(f"Holding signal for {symbol} with no open position. No action.")

EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate trading.py."; fi
echog " -> trading.py generated successfully."
echo # Newline

# 8. Create main.py
echob "Step 8: Generating main.py..."
cat << 'EOF' > main.py
# ~/trading-bot/main.py
import time
import logging
import signal
import os
import sys
from pathlib import Path
from typing import Optional, Dict
from decimal import Decimal, getcontext
import ccxt # Import ccxt for specific exceptions

# --- Set Global Decimal Precision (Optional) ---
# Adjust the precision as needed for your calculations.
# Be cautious if libraries used have their own context management.
# getcontext().prec = 18 # Example: 18 digits of precision

# --- Graceful Shutdown Signal Handling ---
_shutdown_requested = False
def handle_shutdown(signum, frame):
    """Signal handler to initiate graceful shutdown."""
    global _shutdown_requested
    # Use print here as logger might be shutting down
    print(f"\n>>> Shutdown signal ({signal.Signals(signum).name}) received. Initiating graceful exit... <<<")
    _shutdown_requested = True

# --- Module Imports ---
# Use try-except for robustness, especially if run in different environments
try:
    from config import load_config, BotConfig, DEFAULT_CONFIG_PATH, DEFAULT_LOG_DIR
    from utils import setup_logger, FG_RED, FG_GREEN, FG_YELLOW, FG_CYAN, FG_MAGENTA, RESET, BRIGHT, BG_RED, FG_WHITE
    from exchange import ExchangeManager
    from strategy import VolumaticOBStrategy, StrategyAnalysisResults
    from signals import SignalGenerator
    from trading import TradeManager
except ImportError as e:
     # Attempt to handle running the script directly within the bot directory
     print(f"ImportWarning: Running main.py directly? Trying local imports: {e}")
     current_dir = Path(__file__).parent.resolve()
     if str(current_dir) not in sys.path:
         sys.path.insert(0, str(current_dir))
     try:
         from config import load_config, BotConfig, DEFAULT_CONFIG_PATH, DEFAULT_LOG_DIR
         from utils import setup_logger, FG_RED, FG_GREEN, FG_YELLOW, FG_CYAN, FG_MAGENTA, RESET, BRIGHT, BG_RED, FG_WHITE
         from exchange import ExchangeManager
         from strategy import VolumaticOBStrategy, StrategyAnalysisResults
         from signals import SignalGenerator
         from trading import TradeManager
         print("Local imports successful.")
     except ImportError as inner_e:
          # Use print as logger isn't set up yet
          print(f"\n{BG_RED}{FG_WHITE}CRITICAL IMPORT ERROR:{RESET}")
          print(f"Failed to import necessary modules even after adding script directory ('{current_dir}') to path.")
          print("Please ensure all required .py files (config, utils, exchange, strategy, signals, trading) exist in the same directory as main.py.")
          print(f"Python Path: {sys.path}")
          print(f"Error details: {inner_e}")
          sys.exit(1)


# --- Main Bot Execution ---
def run_bot():
    """Initializes and runs the main trading bot loop."""
    global _shutdown_requested
    main_logger: Optional[logging.Logger] = None
    config: Optional[BotConfig] = None
    exchange_manager: Optional[ExchangeManager] = None # Keep track for shutdown cleanup

    try:
        # --- 1. Initial Setup (Logging & Config) ---
        # Setup a temporary logger for early messages before config is fully loaded
        temp_log_dir = Path(os.environ.get("HOME", ".")) / "trading-bot-temp-logs"
        temp_logger = setup_logger("init", level="INFO", log_dir=temp_log_dir)
        temp_logger.info("Initializing Pyrmethus Bot...")

        try:
            # Load configuration using the temporary logger
            config = load_config(DEFAULT_CONFIG_PATH, temp_logger)
            if not config: raise ValueError("Configuration loading returned None.")
            # Determine final log directory based on config (or default if needed)
            log_dir = DEFAULT_LOG_DIR # Default from config module
        except Exception as e:
            temp_logger.critical(f"{BG_RED}{FG_WHITE}CRITICAL: Failed to load configuration: {e}{RESET}", exc_info=True)
            print(f"\n{BG_RED}{FG_WHITE}CRITICAL: Configuration loading failed.{RESET}")
            print(f"Check logs in '{temp_log_dir}' and the config file '{DEFAULT_CONFIG_PATH}'. Exiting.")
            sys.exit(1)

        # Setup the main application logger using the loaded configuration level
        main_logger = setup_logger("PyrmethusBot", level=config.log_level, log_dir=log_dir)
        main_logger.info(f"{FG_MAGENTA}{BRIGHT}========================================={RESET}")
        main_logger.info(f"{FG_MAGENTA}{BRIGHT}=== Pyrmethus Trading Bot Initializing ==={RESET}")
        main_logger.info(f"{FG_MAGENTA}{BRIGHT}========================================={RESET}")

        # Log key configuration settings (excluding secrets)
        try:
            loggable_config = config.dict(exclude={'api_key', 'api_secret'})
            main_logger.info(f"Loaded Configuration (excluding secrets): {loggable_config}")
        except Exception:
            main_logger.warning("Could not log full config details.")
        main_logger.info(f"Log Level: {config.log_level}")
        main_logger.info(f"Trading Enabled: {config.enable_trading}")
        main_logger.info(f"Using Sandbox: {config.use_sandbox}")
        main_logger.info(f"Trading Pairs: {config.trading_pairs}")
        main_logger.info(f"Quote Currency: {config.quote_currency}")
        main_logger.info(f"Risk Per Trade: {config.risk_per_trade*100}%")

        # --- CRITICAL Check for API Keys ---
        if not config.api_key or not config.api_secret or \
           "YOUR_API_KEY" in config.api_key or "YOUR_API_SECRET" in config.api_secret:
            main_logger.critical(f"{BG_RED}{FG_WHITE}CRITICAL: API Key or Secret is missing or uses placeholders.{RESET}")
            main_logger.critical(f"Set BYBIT_API_KEY/BYBIT_API_SECRET environment variables or edit '{DEFAULT_CONFIG_PATH}'.")
            main_logger.critical("Exiting due to missing API credentials.")
            sys.exit(1)
        else:
            main_logger.info(f"{FG_GREEN}API Credentials seem to be loaded (format not checked).{RESET}")


        # --- 2. Instantiate Core Components ---
        try:
            main_logger.info("Initializing Exchange Manager...")
            exchange_manager = ExchangeManager(config, main_logger) # Handles connection test inside

            main_logger.info("Initializing Strategy Analyzer...")
            strategy_analyzer = VolumaticOBStrategy(config.strategy_params.dict(), main_logger)

            main_logger.info("Initializing Signal Generator...")
            signal_generator = SignalGenerator(config, main_logger)

            main_logger.info("Initializing Trade Manager...")
            trade_manager = TradeManager(config, main_logger, exchange_manager)

        except (ccxt.AuthenticationError, ccxt.NetworkError) as conn_err:
             main_logger.critical(f"{BG_RED}{FG_WHITE}CRITICAL Exchange Connection Error during Initialization: {conn_err}{RESET}")
             main_logger.critical("Check API keys, network connectivity, and Bybit status. Exiting.")
             sys.exit(1)
        except Exception as e:
            main_logger.critical(f"{BG_RED}{FG_WHITE}CRITICAL Error during Core Component Initialization: {e}{RESET}", exc_info=True)
            main_logger.critical("Could not initialize bot components. Exiting.")
            sys.exit(1)

        # --- 3. Main Trading Loop ---
        main_logger.info(f"{FG_GREEN}{BRIGHT}Initialization Complete. Entering main trading cycle... Press Ctrl+C to Stop.{RESET}")
        cycle_count = 0
        while not _shutdown_requested:
            cycle_count += 1
            cycle_start_time = time.time()
            main_logger.info(f"\n{FG_CYAN}{BRIGHT}--- Starting Trading Cycle {cycle_count} ---{RESET}")

            try:
                # --- Pre-Cycle Checks ---
                # Fetch Balance once per cycle (only if trading enabled)
                current_balance = None
                if config.enable_trading:
                    main_logger.debug(f"Fetching balance for {config.quote_currency}...")
                    current_balance = exchange_manager.fetch_balance(config.quote_currency)
                    if current_balance is None:
                        main_logger.warning(f"{FG_YELLOW}Could not fetch balance for {config.quote_currency}. Trading actions may be skipped or fail.{RESET}")
                        # Optional: Add a longer delay here or skip cycle?
                    elif current_balance <= Decimal(0):
                        main_logger.warning(f"{FG_YELLOW}Balance is zero or negative ({current_balance} {config.quote_currency}). Trading actions blocked.{RESET}")
                        # No point processing pairs if balance is zero. Wait longer.
                        wait_time = max(10, config.loop_delay_seconds * 2)
                        main_logger.info(f"Waiting {wait_time}s due to zero balance...")
                        time.sleep(wait_time)
                        continue # Skip to next cycle iteration
                else:
                    main_logger.debug("Trading disabled, skipping balance check.")


                # --- Process Each Configured Trading Pair ---
                for pair_info in config.trading_pairs:
                    if _shutdown_requested: break # Check flag frequently within the loop

                    try:
                        # pair_info format is SYMBOL/QUOTE:SETTLEMENT (e.g., BTC/USDT:USDT)
                        symbol_part, settlement_currency = pair_info.split(':')
                        # CCXT uses the symbol part (e.g., BTC/USDT)
                        symbol = symbol_part
                        # Validation happened in config loading, assume format is okay here
                    except ValueError:
                         # Should not happen if config validation worked, but safety check
                         main_logger.error(f"{FG_RED}Skipping invalid pair format in loop: '{pair_info}'.{RESET}")
                         continue

                    pair_start_time = time.time()
                    main_logger.info(f"{FG_BLUE}Processing Pair: {symbol} (Settlement: {settlement_currency}){RESET}")

                    # 1. Fetch Market Data (Klines)
                    main_logger.debug(f"Fetching klines for {symbol}...")
                    df_klines = exchange_manager.fetch_klines(symbol, config.interval, config.fetch_limit)
                    if df_klines is None or df_klines.empty:
                        main_logger.warning(f"{FG_YELLOW}No kline data fetched for {symbol}. Skipping analysis for this pair.{RESET}")
                        continue # Skip to next pair

                    # 2. Analyze Strategy
                    main_logger.debug(f"Analyzing strategy for {symbol}...")
                    analysis_results = strategy_analyzer.analyze(df_klines, symbol, config.interval)
                    if not analysis_results:
                        main_logger.warning(f"{FG_YELLOW}Strategy analysis failed or returned no results for {symbol}. Skipping signal generation.{RESET}")
                        continue # Skip to next pair

                    # 3. Fetch Current Position
                    main_logger.debug(f"Fetching position status for {symbol}...")
                    position = exchange_manager.fetch_position(symbol)
                    # position is None if no position, or dict if position exists

                    # 4. Generate Signal
                    main_logger.debug(f"Generating trading signal for {symbol}...")
                    signal = signal_generator.generate_signal(analysis_results, position)
                    main_logger.info(f"Signal Generated for {symbol}: {FG_YELLOW}{BRIGHT}{signal}{RESET}")

                    # 5. Process Signal (Execute Trade/Manage Position)
                    # Pass balance only if trading is enabled and balance is valid
                    effective_balance = current_balance if config.enable_trading and current_balance is not None else Decimal(0)
                    # Check trading enabled status again before processing
                    if config.enable_trading and effective_balance <= 0:
                         main_logger.debug(f"Skipping trade processing for {symbol} due to zero/invalid balance.")
                    else:
                        main_logger.debug(f"Processing signal '{signal}' via Trade Manager for {symbol}...")
                        trade_manager.process_signal(symbol, signal, analysis_results, position, effective_balance)

                    pair_end_time = time.time()
                    main_logger.debug(f"Pair {symbol} processing took {pair_end_time - pair_start_time:.2f}s.")
                    if _shutdown_requested: break # Check flag after processing each pair

                if _shutdown_requested: break # Break outer loop if flagged during pair processing

                # --- Cycle Completion & Delay ---
                cycle_end_time = time.time()
                elapsed_seconds = cycle_end_time - cycle_start_time
                # Calculate wait time, ensuring minimum delay and accounting for processing time
                wait_time = max(5.0, config.loop_delay_seconds - elapsed_seconds) # Min 5s delay
                main_logger.info(f"{FG_CYAN}--- Trading Cycle {cycle_count} Complete (Took {elapsed_seconds:.2f}s). Waiting {wait_time:.1f}s... ---{RESET}")

                # Sleep interruptibly (check flag periodically during sleep)
                sleep_interval = 0.5 # Check every 0.5 seconds
                wake_up_time = time.time() + wait_time
                while time.time() < wake_up_time and not _shutdown_requested:
                    time.sleep(sleep_interval)


            # --- Handle Loop-Level Exceptions ---
            except KeyboardInterrupt:
                main_logger.info("KeyboardInterrupt detected during main loop. Initiating shutdown...")
                _shutdown_requested = True
            except ccxt.AuthenticationError as e:
                main_logger.critical(f"{BG_RED}{FG_WHITE}CRITICAL: Authentication Error during main loop: {e}. Check API keys! Forcing Shutdown.{RESET}")
                _shutdown_requested = True # Trigger shutdown
            except ccxt.NetworkError as e:
                 main_logger.error(f"{FG_RED}Network Error in main loop: {e}. Might be temporary. Waiting longer before next cycle...{RESET}")
                 # Sleep longer but check shutdown flag during sleep
                 sleep_interval = 1
                 total_wait = max(15, config.loop_delay_seconds * 2) # Wait longer on network issues
                 for _ in range(int(total_wait / sleep_interval)):
                      if _shutdown_requested: break
                      time.sleep(sleep_interval)
            except ccxt.ExchangeError as e:
                 # Includes RateLimitExceeded if not caught by retry wrapper, or other exchange issues
                 main_logger.error(f"{FG_RED}Exchange Error in main loop: {e}. Waiting before next cycle...{RESET}")
                 sleep_interval = 1
                 total_wait = max(10, config.loop_delay_seconds * 1.5)
                 for _ in range(int(total_wait / sleep_interval)):
                     if _shutdown_requested: break
                     time.sleep(sleep_interval)
            except Exception as e:
                # Catch any other unexpected error in the main loop
                main_logger.error(f"{BG_RED}{FG_WHITE}Unhandled Error in Main Trading Loop: {e}{RESET}", exc_info=True)
                main_logger.info(f"Attempting to continue after error. Waiting {max(15, config.loop_delay_seconds * 2)} seconds...")
                sleep_interval = 1
                total_wait = max(15, config.loop_delay_seconds * 2)
                for _ in range(int(total_wait / sleep_interval)):
                    if _shutdown_requested: break
                    time.sleep(sleep_interval)

    # --- Handle Initialization Errors ---
    except Exception as init_error:
         # Catch critical errors during initial setup (logging, config, component instantiation)
         final_logger = main_logger if main_logger else temp_logger # Use whichever logger exists
         error_msg = f"CRITICAL INITIALIZATION ERROR: {init_error}"
         print(f"\n{BG_RED}{FG_WHITE}{error_msg}{RESET}") # Print error to console
         if final_logger:
             final_logger.critical(error_msg, exc_info=True)
         # No cleanup needed as components likely didn't fully initialize
         sys.exit(1) # Exit immediately on critical init failure

    # --- Shutdown Sequence ---
    finally:
        shutdown_msg = f"{FG_MAGENTA}{BRIGHT}====================================\n" \
                       f"=== Pyrmethus Bot Shutting Down ===\n" \
                       f"===================================={RESET}"
        print(f"\n{shutdown_msg}") # Use print for final shutdown message

        final_logger = main_logger if main_logger else temp_logger
        if final_logger: final_logger.info(shutdown_msg)

        # --- Optional: Cleanup Actions ---
        if config and exchange_manager and config.enable_trading and config.cancel_orders_on_exit:
            if final_logger: final_logger.info("Attempting to cancel open orders as configured...")
            print("Attempting to cancel open orders...") # Also print
            try:
                cancelled_count = 0
                for pair_info in config.trading_pairs:
                     if _shutdown_requested and cancelled_count > 0:
                          # Allow some time for previous cancels, but don't wait forever if shutdown is urgent
                          if final_logger: final_logger.warning("Shutdown requested during order cancellation, stopping early.")
                          print("Shutdown requested during order cancellation, stopping early.")
                          break
                     symbol = pair_info.split(':')[0]
                     open_orders = exchange_manager.fetch_open_orders(symbol)
                     if final_logger: final_logger.debug(f"Found {len(open_orders)} open orders for {symbol}.")
                     for order in open_orders:
                         try:
                             order_id = order.get('id')
                             if order_id:
                                 if final_logger: final_logger.info(f"Cancelling order {order_id} for {symbol}...")
                                 print(f"Cancelling order {order_id} for {symbol}...")
                                 exchange_manager.cancel_order(order_id, symbol)
                                 cancelled_count += 1
                                 time.sleep(0.3) # Small delay between cancels
                         except Exception as cancel_e:
                             err_msg = f"Error cancelling order {order.get('id', 'N/A')} for {symbol}: {cancel_e}"
                             if final_logger: final_logger.error(err_msg)
                             print(err_msg)
                msg = f"Attempted cancellation of {cancelled_count} orders."
                if final_logger: final_logger.info(msg)
                print(msg)
            except Exception as cancel_all_e:
                  err_msg = f"Error during bulk order cancellation on exit: {cancel_all_e}"
                  if final_logger: final_logger.error(err_msg)
                  print(err_msg)
        elif config and config.cancel_orders_on_exit:
             msg = "Order cancellation on exit skipped (Trading disabled or components not initialized)."
             if final_logger: final_logger.info(msg)
             print(msg)

        # --- Final Log Message & Exit ---
        final_msg = f"{FG_GREEN}{BRIGHT}Pyrmethus Trading Bot has ceased its vigil. Farewell.{RESET}"
        if final_logger:
             final_logger.info(final_msg)
             logging.shutdown() # Flushes all handlers and closes files
        print(final_msg)
        sys.exit(0)


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure the script directory is in the Python path if run directly
    # (Handles the case where modules are in the same directory)
    script_dir = Path(__file__).parent.resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
        print(f"Note: Added script directory '{script_dir}' to sys.path")

    # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM (kill)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Start the bot
    run_bot()
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate main.py."; fi
echog " -> main.py generated successfully."
echo # Newline

# 9. Create config.json template
echob "Step 9: Generating config.json template..."
cat << 'EOF' > config.json
{
    "api_key": "YOUR_API_KEY_HERE_OR_SET_ENV",
    "api_secret": "YOUR_API_SECRET_HERE_OR_SET_ENV",
    "trading_pairs": [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT"
    ],
    "interval": "5",
    "retry_delay": 6,
    "fetch_limit": 750,
    "enable_trading": false,
    "use_sandbox": true,
    "risk_per_trade": 0.01,
    "leverage": 10,
    "quote_currency": "USDT",
    "loop_delay_seconds": 15,
    "position_confirm_delay_seconds": 8,
    "log_level": "INFO",
    "cancel_orders_on_exit": false,
    "strategy_params": {
        "vt_length": 40,
        "vt_atr_period": 200,
        "vt_vol_ema_length": 950,
        "vt_atr_multiplier": 3.0,
        "ob_source": "Wicks",
        "ph_left": 10,
        "ph_right": 10,
        "pl_left": 10,
        "pl_right": 10,
        "ob_extend": true,
        "ob_max_boxes": 30,
        "ob_entry_proximity_factor": 1.003,
        "ob_exit_proximity_factor": 1.001
    },
    "protection": {
        "enable_trailing_stop": false,
        "trailing_stop_callback_rate": 0.005,
        "trailing_stop_activation_percentage": 0.003,
        "enable_break_even": false,
        "break_even_trigger_atr_multiple": 1.0,
        "break_even_offset_ticks": 2,
        "initial_stop_loss_atr_multiple": 1.8,
        "initial_take_profit_atr_multiple": 0.7
    }
}
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate config.json."; fi
echog " -> config.json template generated successfully."
echo # Newline

# 10. Create requirements.txt
echob "Step 10: Generating requirements.txt..."
cat << EOF > requirements.txt
# ~/trading-bot/requirements.txt
# Python package dependencies for the Pyrmethus Trading Bot.
# Use specific versions known to work together or minimum required versions.
# Check compatibility, especially for ccxt with Bybit V5 API and pandas/pandas_ta.

ccxt>=4.1.0        # Min version supporting Bybit V5 reasonably well (check latest releases)
pandas>=1.5.0,<2.0.0 # Lock Pandas to < 2.0 for broader compatibility (esp. with pandas_ta)
numpy>=1.21.0      # Required by pandas
pandas_ta>=0.3.14b0 # Ensure compatibility with chosen pandas version
pydantic>=1.10.0,<2.0.0 # Pydantic v2 has breaking changes, stick to v1.x for now
colorama>=0.4.4    # For colored terminal output in utils.py
requests>=2.28.0   # Often a dependency of ccxt, good to list explicitly

# Optional: Add other libraries if needed (e.g., for plotting, databases)
# matplotlib
# SQLAlchemy
# python-dotenv # If using .env files for API keys
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate requirements.txt."; fi
echog " -> requirements.txt generated successfully."
echo # Newline

# 11. Create .gitignore
echob "Step 11: Generating .gitignore..."
cat << 'EOF' > .gitignore
# ~/trading-bot/.gitignore
# Specifies intentionally untracked files that Git should ignore.

# --- Configuration (Secrets!) ---
# IMPORTANT: Keep your API keys and sensitive settings out of Git.
# Track a template file instead if needed (e.g., config.template.json).
config.json

# --- Environment Files ---
# Virtual environment directories
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
# Environment variable files
*.env

# --- Log Files ---
# Log directories and files generated by the bot
logs/
bot_logs/
*.log
*.log.*

# --- Python Cache and Artifacts ---
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg
*.egg-info/
.eggs/
dist/
build/
develop-eggs/
downloads/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
.installed.cfg
MANIFEST

# --- IDE and Editor Files ---
.idea/
.vscode/
*.project
*.pydevproject
.sublime-workspace
.sublime-project
*.swp
*~
*.tmp

# --- Test and Coverage Artifacts ---
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# --- OS Generated Files ---
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# --- Other ---
# Temporary log directory used by main.py initialization
trading-bot-temp-logs/
EOF
if [ $? -ne 0 ]; then error_exit "Failed to generate .gitignore."; fi
echog " -> .gitignore generated successfully."
echo # Newline

# 12. Final Summary and Instructions
echom "==========================================="
echog " Pyrmethus Bot Setup Complete!"
echog " All necessary files have been generated in:"
echoc "$BOT_DIR"
echom "===========================================\n"

echobright "==> NEXT STEPS <=="
echo -e "${COLOR_BLUE}1. ${COLOR_BRIGHT_CYAN}Navigate to the Bot Directory:${COLOR_RESET}"
echo -e "   ${COLOR_GREEN}cd $BOT_DIR${COLOR_RESET}"
echo
echo -e "${COLOR_BLUE}2. ${COLOR_BRIGHT_CYAN}Configure the Bot (${COLOR_BRIGHT_RED}IMPORTANT!${COLOR_BRIGHT_CYAN}):${COLOR_RESET}"
echo -e "   - Edit the ${COLOR_YELLOW}config.json${COLOR_RESET} file using a text editor (like nano):"
echo -e "     ${COLOR_GREEN}nano config.json${COLOR_RESET}"
echo -e "   - ${COLOR_BRIGHT_RED}CRITICAL:${COLOR_RESET} Replace placeholders for ${COLOR_YELLOW}\"api_key\"${COLOR_RESET} and ${COLOR_YELLOW}\"api_secret\"${COLOR_RESET} with your actual Bybit API credentials."
echo -e "     ${COLOR_BRIGHT_YELLOW}(Alternatively, for better security, set them as environment variables:${COLOR_RESET}"
echo -e "     ${COLOR_CYAN}export BYBIT_API_KEY=\"YOUR_KEY\"${COLOR_RESET}"
echo -e "     ${COLOR_CYAN}export BYBIT_API_SECRET=\"YOUR_SECRET\"${COLOR_RESET}"
echo -e "     ${COLOR_BRIGHT_YELLOW}The script prioritizes environment variables if set.)${COLOR_RESET}"
echo -e "   - Review and customize other settings:"
echo -e "     - ${COLOR_YELLOW}\"trading_pairs\"${COLOR_RESET}: Add or remove the symbols you want to trade."
echo -e "     - ${COLOR_YELLOW}\"risk_per_trade\"${COLOR_RESET}, ${COLOR_YELLOW}\"leverage\"${COLOR_RESET}: Adjust risk settings."
echo -e "     - ${COLOR_YELLOW}\"strategy_params\"${COLOR_RESET}: Fine-tune the strategy indicators."
echo -e "     - ${COLOR_YELLOW}\"protection\"${COLOR_RESET}: Configure SL, TP, TSL, BE (Note: TSL/BE require full implementation)."
echo -e "   - ${COLOR_BRIGHT_YELLOW}Keep ${COLOR_YELLOW}\"use_sandbox\": true${COLOR_BRIGHT_YELLOW} for testing!${COLOR_RESET}"
echo -e "   - Set ${COLOR_YELLOW}\"enable_trading\": true${COLOR_RESET} ONLY when you have tested thoroughly and are ready for live trading."
echo -e "   - Save the file (Ctrl+O, Enter in nano) and exit (Ctrl+X in nano)."
echo
echo -e "${COLOR_BLUE}3. ${COLOR_BRIGHT_CYAN}(Recommended) Setup Python Virtual Environment:${COLOR_RESET}"
echo -e "   ${COLOR_YELLOW}(This isolates dependencies for this project)${COLOR_RESET}"
echo -e "   # Ensure you are in the '$BOT_DIR' directory"
echo -e "   ${COLOR_GREEN}python -m venv venv${COLOR_RESET}"
echo -e "   ${COLOR_GREEN}source venv/bin/activate${COLOR_RESET}  ${COLOR_YELLOW}# Activate the environment (use this command each time you work on the bot)${COLOR_RESET}"
echo -e "   ${COLOR_YELLOW}(To deactivate later, simply type: deactivate)${COLOR_RESET}"
echo
echo -e "${COLOR_BLUE}4. ${COLOR_BRIGHT_CYAN}Install Dependencies:${COLOR_RESET}"
echo -e "   ${COLOR_YELLOW}(Ensure virtual environment is active if you created one)${COLOR_RESET}"
echo -e "   ${COLOR_GREEN}python -m pip install --upgrade pip${COLOR_RESET}   ${COLOR_YELLOW}# Upgrade pip installer${COLOR_RESET}"
echo -e "   ${COLOR_GREEN}python -m pip install -r requirements.txt${COLOR_RESET}"
echo
echo -e "${COLOR_BLUE}5. ${COLOR_BRIGHT_CYAN}Run the Bot:${COLOR_RESET}"
echo -e "   ${COLOR_YELLOW}(Ensure virtual environment is active if used)${COLOR_RESET}"
echo -e "   ${COLOR_GREEN}python main.py${COLOR_RESET}"
echo
echo -e "${COLOR_BLUE}6. ${COLOR_BRIGHT_CYAN}Monitor Output & Logs:${COLOR_RESET}"
echo -e "   - Observe the terminal output for status messages and errors."
echo -e "   - Check the detailed log files created in the ${COLOR_YELLOW}bot_logs/${COLOR_RESET} directory."
echo
echom "${COLOR_BRIGHT_GREEN}May your trades be efficient and profitable! Remember to test extensively in sandbox mode.${COLOR_RESET}"

exit 0