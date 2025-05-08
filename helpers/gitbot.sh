Okay, here is a bash script that will create the directory structure, populate the files with the provided Python code, and initialize a new Git repository.

**Important Notes:**

1.  **Run Safely:** Execute this script in a directory where you want the *new project folder* to be created. It will create a sub-directory named `bybit_evt_strategy`.
2.  **Overwriting:** The script checks if the project directory already exists and exits to prevent accidental overwriting.
3.  **API Keys:** The `.env` file is created with **PLACEHOLDER** keys. You **MUST** edit this file and replace the placeholders with your actual Bybit API key and secret before running the bot.
4.  **Git User:** The script initializes the Git repository but doesn't force the user/email configuration. It includes commented-out commands showing how to set it specifically for this repository if needed. Git usually picks up your global configuration.
5.  **Remote Repository:** The script doesn't automatically create or link to a remote repository (like GitHub). You'll need to do that manually if desired (example command included).

```bash
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
# set -u # Be cautious with set -u if sourcing other scripts

# --- Configuration ---
PROJECT_DIR="bybit_evt_strategy"
GIT_USER_NAME="mentallyspammed1" # Used in comments/instructions
GIT_USER_EMAIL="your_email@example.com" # Replace with actual email

# --- Safety Check ---
if [ -d "$PROJECT_DIR" ]; then
  echo "Error: Directory '$PROJECT_DIR' already exists. Please remove or rename it first."
  exit 1
fi

echo "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Creating Python files..."

# --- Create config_models.py ---
echo "# Creating config_models.py"
cat << 'EOF' > config_models.py
# config_models.py
"""
Pydantic Models for Application Configuration using pydantic-settings.

Loads configuration from environment variables and/or a .env file.
Provides type validation and default values.
"""

import logging
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
    FilePath,
    DirectoryPath,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
import os # For environment variable access during load

# --- Import Enums for type safety ---
# Attempt to import from helpers, provide fallback if needed during setup
try:
    # Ensure bybit_helpers.py exists and is importable when this runs
    # If running this script standalone before dependencies are set up, this might fail.
    # Consider temporary basic types if needed for initial generation.
    from bybit_helpers import (
        PositionIdx,
        Category,
        OrderFilter,
        Side,
        TimeInForce,
        TriggerBy,
        TriggerDirection,
    )
except ImportError:
    print(
        "Warning [config_models]: Could not import Enums from bybit_helpers. Using basic types/Literals as fallback."
    )
    # Define basic types or Literals as placeholders
    PositionIdx = int # Fallback to int
    Category = Literal["linear", "inverse", "spot", "option"]
    OrderFilter = Literal["Order", "StopOrder", "tpslOrder"] # Add others if needed
    Side = Literal["buy", "sell"]
    TimeInForce = Literal["GTC", "IOC", "FOK", "PostOnly"]
    TriggerBy = Literal["LastPrice", "MarkPrice", "IndexPrice"]
    TriggerDirection = Literal[1, 2]

class APIConfig(BaseModel):
    """Configuration for Bybit API Connection and Market Defaults."""
    exchange_id: Literal["bybit"] = "bybit"
    api_key: Optional[str] = Field(None, description="Bybit API Key")
    api_secret: Optional[str] = Field(None, description="Bybit API Secret")
    testnet_mode: bool = Field(True, description="Use Bybit Testnet environment")
    default_recv_window: PositiveInt = Field(
        10000, description="API request validity window in milliseconds"
    )

    # Market & Symbol Defaults
    symbol: str = Field(..., description="Primary trading symbol (e.g., BTC/USDT:USDT)")
    usdt_symbol: str = Field("USDT", description="Quote currency for balance reporting")
    expected_market_type: Literal["swap", "spot", "option", "future"] = Field(
        "swap", description="Expected market type for validation"
    )
    expected_market_logic: Literal["linear", "inverse"] = Field(
        "linear", description="Expected market logic for derivative validation"
    )

    # Retry & Rate Limit Defaults
    retry_count: NonNegativeInt = Field(
        3, description="Default number of retries for API calls"
    )
    retry_delay_seconds: PositiveFloat = Field(
        2.0, description="Default base delay (seconds) for API retries"
    )

    # Fee Rates (Important for accurate calculations)
    maker_fee_rate: Decimal = Field(
        Decimal("0.0002"), description="Maker fee rate (e.g., 0.0002 for 0.02%)"
    )
    taker_fee_rate: Decimal = Field(
        Decimal("0.00055"), description="Taker fee rate (e.g., 0.00055 for 0.055%)"
    )

    # Order Execution Defaults & Helpers
    default_slippage_pct: Decimal = Field(
        Decimal("0.005"),
        gt=0,
        le=0.1,
        description="Default max slippage % for market order checks (e.g., 0.005 for 0.5%)",
    )
    position_qty_epsilon: Decimal = Field(
        Decimal("1e-9"),
        gt=0,
        description="Small value for floating point quantity comparisons",
    )
    shallow_ob_fetch_depth: PositiveInt = Field(
        5, description="Order book depth for quick spread/slippage check"
    )
    order_book_fetch_limit: PositiveInt = Field(
        25, description="Default depth for fetching L2 order book"
    )

    # Position/Side Constants (Internal use)
    pos_none: Literal["NONE"] = "NONE"
    pos_long: Literal["LONG"] = "LONG"
    pos_short: Literal["SHORT"] = "SHORT"
    side_buy: Literal["buy"] = "buy"
    side_sell: Literal["sell"] = "sell"

    @field_validator('api_key', 'api_secret', mode='before') # mode='before' to catch env var directly
    @classmethod
    def check_not_placeholder(cls, v: Optional[str]) -> Optional[str]:
        if v and "PLACEHOLDER" in v.upper():
             print(f"WARNING: API Key/Secret field appears to be a placeholder: '{v[:15]}...'")
        return v

    @field_validator('symbol', mode='before')
    @classmethod
    def check_symbol_format(cls, v: str) -> str:
        if not isinstance(v, str):
             raise ValueError("Symbol must be a string")
        if ":" not in v and "/" not in v:
            raise ValueError(f"Invalid symbol format: '{v}'. Expected format like 'BTC/USDT:USDT' or 'BTC/USDT'.")
        return v.upper()


class IndicatorSettings(BaseModel):
    """Parameters for Technical Indicator Calculations."""
    min_data_periods: PositiveInt = Field(
        100, description="Minimum historical candles needed for calculations"
    )
    # Ehlers Volumetric specific
    evt_length: PositiveInt = Field(
        7, description="Period length for EVT indicator"
    )
    evt_multiplier: PositiveFloat = Field(
        2.5, description="Multiplier for EVT bands calculation"
    )
    # ATR specific (often used for stop loss)
    atr_period: PositiveInt = Field(
        14, description="Period length for ATR indicator"
    )


class AnalysisFlags(BaseModel):
    """Flags to Enable/Disable Specific Indicator Calculations."""
    use_evt: bool = Field(True, description="Enable Ehlers Volumetric Trend calculation")
    use_atr: bool = Field(True, description="Enable ATR calculation")


class StrategyConfig(BaseModel):
    """Core Strategy Behavior and Parameters."""
    name: str = Field(..., description="Name of the strategy instance")
    timeframe: str = Field("15m", description="Candlestick timeframe (e.g., '1m', '5m', '1h')")
    polling_interval_seconds: PositiveInt = Field(
        60, description="Frequency (seconds) to fetch data and check signals"
    )
    leverage: PositiveInt = Field(
        5, description="Desired leverage (check exchange limits)"
    )
    # Use type annotation directly if possible, fallback to int if import failed
    position_idx: PositionIdx | int = Field( # Allow int as fallback
        0, # Default to One-Way (0)
        description="Position mode (0: One-Way, 1: Hedge Buy, 2: Hedge Sell)"
    )
    risk_per_trade: Decimal = Field(
        Decimal("0.01"), gt=0, le=0.1,
        description="Fraction of available balance to risk per trade (e.g., 0.01 for 1%)",
    )
    stop_loss_atr_multiplier: Decimal = Field(
        Decimal("2.0"), gt=0, description="ATR multiplier for stop loss distance"
    )
    indicator_settings: IndicatorSettings = Field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = Field(default_factory=AnalysisFlags)
    EVT_ENABLED: bool = Field(
        True, description="Confirms EVT logic is core (redundant if analysis_flags.use_evt is True)"
    )

    @field_validator('EVT_ENABLED')
    @classmethod
    def check_evt_consistency(cls, v: bool, info) -> bool:
        flags = info.data.get('analysis_flags')
        if v and flags and not flags.use_evt:
            raise ValueError("'EVT_ENABLED' is True but 'analysis_flags.use_evt' is False.")
        if not v and flags and flags.use_evt:
             print("Warning [StrategyConfig]: 'EVT_ENABLED' is False but 'analysis_flags.use_evt' is True.")
        return v

    @field_validator('stop_loss_atr_multiplier')
    @classmethod
    def check_atr_enabled_for_sl(cls, v: Decimal, info) -> Decimal:
         flags = info.data.get('analysis_flags')
         if v > 0 and flags and not flags.use_atr:
             raise ValueError("'stop_loss_atr_multiplier' > 0 requires 'analysis_flags.use_atr' to be True.")
         return v


class LoggingConfig(BaseModel):
    """Configuration for the Logger Setup."""
    logger_name: str = Field("TradingBot", description="Name for the logger instance")
    log_file: Optional[str] = Field("trading_bot.log", description="Path to the log file (None to disable)")
    console_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level for console output"
    )
    file_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        "DEBUG", description="Logging level for file output"
    )
    log_rotation_bytes: NonNegativeInt = Field(
        5 * 1024 * 1024, description="Max log file size in bytes before rotating (0 disables)"
    )
    log_backup_count: NonNegativeInt = Field(
        5, description="Number of backup log files to keep"
    )
    third_party_log_level_str: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "WARNING", description="Log level for noisy third-party libraries (e.g., ccxt)"
    )


class SMSConfig(BaseModel):
    """Configuration for SMS Alerting (e.g., via Termux)."""
    enable_sms_alerts: bool = Field(False, description="Globally enable/disable SMS alerts")
    use_termux_api: bool = Field(False, description="Use Termux:API for SMS")
    sms_recipient_number: Optional[str] = Field(None, description="Recipient phone number")
    sms_timeout_seconds: PositiveInt = Field(30, description="Timeout for Termux API call")
    # Add Twilio fields if needed later
    # use_twilio_api: bool = Field(False, ...)
    # twilio_account_sid: Optional[str] = Field(None, ...)

    @field_validator('enable_sms_alerts')
    @classmethod
    def check_sms_provider_details(cls, v: bool, info) -> bool:
        if v:
            use_termux = info.data.get('use_termux_api')
            recipient = info.data.get('sms_recipient_number')
            # Add check for Twilio if implemented
            if not use_termux: # and not use_twilio:
                raise ValueError("SMS alerts enabled, but 'use_termux_api' is False.")
            if not recipient:
                raise ValueError("SMS alerts enabled, but 'sms_recipient_number' is missing.")
        return v


class AppConfig(BaseSettings):
    """Master Configuration Model."""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_nested_delimiter='__',
        env_prefix='BOT_', # Important: Env vars should be BOT_API_CONFIG__SYMBOL etc.
        case_sensitive=False,
        extra='ignore'
    )

    api_config: APIConfig = Field(default_factory=APIConfig)
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    sms_config: SMSConfig = Field(default_factory=SMSConfig)

def load_config() -> AppConfig:
    """Loads the AppConfig, handling potential validation errors."""
    try:
        print("Loading configuration from environment variables and .env file...")
        # Explicitly pass env_file to load it relative to the current execution context
        # This helps if the script is run from a different directory.
        env_file_path = os.path.join(os.getcwd(), '.env')
        config = AppConfig(_env_file=env_file_path if os.path.exists(env_file_path) else None)

        # Post-load checks/logging
        if config.api_config.api_key and "PLACEHOLDER" in config.api_config.api_key.upper():
             print("WARNING [load_config]: API Key seems to be a placeholder.")
        print("Configuration loaded successfully.")
        return config
    except ValidationError as e:
        print(f"\n{'-'*20} CONFIGURATION VALIDATION FAILED {'-'*20}")
        for error in e.errors():
            loc = " -> ".join(map(str, error['loc'])) if error['loc'] else 'AppConfig'
            print(f"  Field: {loc}")
            print(f"  Error: {error['msg']}")
            # print(f"  Value: {error.get('input')}") # May expose secrets, use cautiously
        print(f"{'-'*60}\n")
        raise SystemExit("Configuration validation failed. Please check your .env file or environment variables.")
    except Exception as e:
        print(f"FATAL: Unexpected error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit("Failed to load configuration.")

# Example of how to load config in main script:
# if __name__ == "__main__":
#     try:
#         app_settings = load_config()
#         print("\nLoaded Config:")
#         print(app_settings.model_dump_json(indent=2)) # Pretty print loaded config
#     except SystemExit as e:
#          print(f"Exiting due to configuration error: {e}")
EOF

# --- Create neon_logger.py ---
echo "# Creating neon_logger.py"
cat << 'EOF' > neon_logger.py
#!/usr/bin/env python
"""Neon Logger Setup (v1.3) - Enhanced Robustness & Features

Provides a function `setup_logger` to configure a Python logger instance with:
- Colorized console output using a "neon" theme via colorama (TTY only).
- Uses a custom Formatter for cleaner color handling.
- Clean, non-colorized file output.
- Optional log file rotation (size-based).
- Extensive log formatting (timestamp, level, function, line, thread).
- Custom SUCCESS log level.
- Configurable log levels via Pydantic model or direct args.
- Option to control verbosity of third-party libraries.
"""

import logging
import logging.handlers
import os
import sys
from typing import Any, Literal, Optional

# --- Import Pydantic model for config type hinting ---
from config_models import LoggingConfig

# --- Attempt to import colorama ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    # Initialize colorama (autoreset=True ensures colors reset after each print)
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Define dummy color objects if colorama is not installed
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""  # Return empty string

    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()
    COLORAMA_AVAILABLE = False
    # Warning printed by setup_logger if needed

# --- Custom Log Level ---
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log method to the Logger instance."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add the method to the Logger class dynamically
if not hasattr(logging.Logger, "success"):
    logging.Logger.success = log_success  # type: ignore[attr-defined]


# --- Neon Color Theme Mapping ---
LOG_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.BLUE,
    SUCCESS_LEVEL: Fore.MAGENTA + Style.BRIGHT, # Make success stand out
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED + Style.BRIGHT,
    logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
}


# --- Custom Formatter for Colored Console Output ---
class ColoredConsoleFormatter(logging.Formatter):
    """A custom logging formatter that adds colors to console output based on log level,
    only if colorama is available and output is a TTY.
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal['%', '{', '$'] = "%",
        validate: bool = True,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate) # type: ignore
        self.use_colors = COLORAMA_AVAILABLE and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        if not COLORAMA_AVAILABLE:
             print("Warning [Logger]: 'colorama' not found. Console logs will be monochrome.")

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record and applies colors to the level name."""
        original_levelname = record.levelname
        color = LOG_LEVEL_COLORS.get(record.levelno, Fore.WHITE)

        if self.use_colors:
            record.levelname = f"{color}{original_levelname:<9}{Style.RESET_ALL}" # Pad levelname within color codes
        else:
            record.levelname = f"{original_levelname:<9}" # Pad without color codes

        formatted_message = super().format(record)
        record.levelname = original_levelname # Restore original levelname

        return formatted_message


# --- Log Format Strings ---
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s %(funcName)s:%(lineno)d] - %(message)s" # Levelname padding handled by formatter
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create formatters (instantiate only once)
console_formatter = ColoredConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


# --- Main Setup Function ---
def setup_logger(
    config: LoggingConfig,
    propagate: bool = False
) -> logging.Logger:
    """Sets up and configures a logger instance based on LoggingConfig.

    Args:
        config: A validated LoggingConfig Pydantic model instance.
        propagate: Whether to propagate messages to the root logger (default False).

    Returns:
        The configured logging.Logger instance.
    """
    try:
        console_level = logging.getLevelName(config.console_level_str)
        file_level = logging.getLevelName(config.file_level_str)
        third_party_log_level = logging.getLevelName(config.third_party_log_level_str)

        if not isinstance(console_level, int): console_level = logging.INFO; print(f"Warning: Invalid console level '{config.console_level_str}'. Defaulting to INFO.", file=sys.stderr)
        if not isinstance(file_level, int): file_level = logging.DEBUG; print(f"Warning: Invalid file level '{config.file_level_str}'. Defaulting to DEBUG.", file=sys.stderr)
        if not isinstance(third_party_log_level, int): third_party_log_level = logging.WARNING; print(f"Warning: Invalid third-party level '{config.third_party_log_level_str}'. Defaulting to WARNING.", file=sys.stderr)

    except Exception as e:
        print(f"FATAL: Error processing log levels from config: {e}. Using defaults.", file=sys.stderr)
        console_level, file_level, third_party_log_level = logging.INFO, logging.DEBUG, logging.WARNING

    logger = logging.getLogger(config.logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate

    if logger.hasHandlers():
        print(f"Logger '{config.logger_name}' already has handlers. Clearing them.", file=sys.stderr)
        for handler in logger.handlers[:]:
            try: handler.close(); logger.removeHandler(handler)
            except Exception as e: print(f"Warning: Error removing handler {handler}: {e}", file=sys.stderr)

    # --- Console Handler ---
    try:
        console_h = logging.StreamHandler(sys.stdout)
        console_h.setLevel(console_level)
        console_h.setFormatter(console_formatter)
        logger.addHandler(console_h)
        print(f"[Logger] Console logging active: Level=[{logging.getLevelName(console_level)}]")
    except Exception as e: print(f"Error setting up console handler: {e}", file=sys.stderr)

    # --- File Handler ---
    if config.log_file:
        try:
            log_file_path = os.path.abspath(config.log_file)
            log_dir = os.path.dirname(log_file_path)
            if log_dir: os.makedirs(log_dir, exist_ok=True)

            if config.log_rotation_bytes > 0 and config.log_backup_count >= 0:
                file_h = logging.handlers.RotatingFileHandler(
                    log_file_path, maxBytes=config.log_rotation_bytes,
                    backupCount=config.log_backup_count, encoding="utf-8")
                log_type, log_details = "Rotating", f"(Max: {config.log_rotation_bytes / 1024 / 1024:.1f} MB, Backups: {config.log_backup_count})"
            else:
                file_h = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
                log_type, log_details = "Basic", "(Rotation disabled)"

            file_h.setLevel(file_level)
            file_h.setFormatter(file_formatter)
            logger.addHandler(file_h)
            print(f"[Logger] {log_type} file logging active: Level=[{logging.getLevelName(file_level)}] File='{log_file_path}' {log_details}")

        except OSError as e: print(f"FATAL Error configuring log file '{config.log_file}': {e}", file=sys.stderr)
        except Exception as e: print(f"Unexpected error setting up file logging: {e}", file=sys.stderr)
    else:
        print("[Logger] File logging disabled.")

    # --- Configure Third-Party Log Levels ---
    if third_party_log_level >= 0:
        noisy_libs = ["ccxt", "ccxt.base", "ccxt.async_support", "urllib3", "requests", "asyncio", "websockets"]
        print(f"[Logger] Setting third-party library log level: [{logging.getLevelName(third_party_log_level)}]")
        for lib_name in noisy_libs:
            try:
                lib_logger = logging.getLogger(lib_name)
                if lib_logger:
                    lib_logger.setLevel(third_party_log_level)
                    lib_logger.propagate = False # Don't let them log to root
            except Exception as e: print(f"Warning: Could not set level for lib '{lib_name}': {e}", file=sys.stderr)
    else:
        print("[Logger] Third-party library log level control disabled.")

    return logger
EOF

# --- Create bybit_utils.py ---
echo "# Creating bybit_utils.py"
cat << 'EOF' > bybit_utils.py
# bybit_utils.py
"""
Utility functions for formatting, safe conversions, and external interactions
like SMS alerts, supporting the Bybit trading bot framework.
"""

import functools
import logging
import subprocess  # For Termux API call
import time
import asyncio
from collections.abc import Callable
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
from typing import Any, TypeVar, Optional, Union

# --- Import Pydantic models for type hinting ---
from config_models import AppConfig, SMSConfig # Use the unified config

# --- Attempt to import CCXT and colorama ---
try:
    import ccxt
    import ccxt.async_support as ccxt_async # Alias for async usage if needed
except ImportError:
    print("FATAL ERROR [bybit_utils]: CCXT library not found.", file=sys.stderr)
    ccxt = None # Set to None to allow checking later

try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

# --- Logger Setup ---
logger = logging.getLogger(__name__)


# --- Utility Functions ---

def safe_decimal_conversion(
    value: Any, default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """Safely convert various inputs to Decimal, returning default or None on failure."""
    if value is None: return default
    try:
        d = Decimal(str(value))
        if d.is_nan() or d.is_infinite(): return default
        return d
    except (ValueError, TypeError, InvalidOperation): return default

def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> Optional[str]:
    """Format a price value according to the market's precision rules."""
    if price is None: return None
    if not ccxt or not exchange: logger.warning("[format_price] CCXT/Exchange unavailable."); return str(price)
    price_dec = safe_decimal_conversion(price)
    if price_dec is None: logger.warning(f"[format_price] Invalid price value '{price}' for {symbol}."); return "Error"
    try:
        return exchange.price_to_precision(symbol, float(price_dec))
    except (AttributeError, KeyError):
        logger.debug(f"[format_price] Market data/method missing for '{symbol}'. Using fallback.")
        market = getattr(exchange, 'market', lambda s: None)(symbol)
        if market and 'precision' in market and 'price' in market['precision']:
            try:
                tick_size = Decimal(str(market['precision']['price']))
                if tick_size <= 0: raise ValueError("Tick size must be positive")
                formatted = price_dec.quantize(tick_size, rounding=ROUND_HALF_UP) # Common rounding for prices
                # Determine number of decimal places from tick size exponent
                decimal_places = abs(tick_size.normalize().as_tuple().exponent)
                return f"{formatted:.{decimal_places}f}"
            except Exception as format_err: logger.error(f"[format_price] Fallback quantize failed: {format_err}"); return f"{price_dec:.8f}"
        else: return f"{price_dec:.8f}"
    except Exception as e: logger.error(f"[format_price] Error formatting price '{price}': {e}", exc_info=False); return "Error"

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> Optional[str]:
    """Format an amount value according to the market's precision rules."""
    if amount is None: return None
    if not ccxt or not exchange: logger.warning("[format_amount] CCXT/Exchange unavailable."); return str(amount)
    amount_dec = safe_decimal_conversion(amount)
    if amount_dec is None: logger.warning(f"[format_amount] Invalid amount value '{amount}' for {symbol}."); return "Error"
    try:
        return exchange.amount_to_precision(symbol, float(amount_dec))
    except (AttributeError, KeyError):
        logger.debug(f"[format_amount] Market data/method missing for '{symbol}'. Using fallback.")
        market = getattr(exchange, 'market', lambda s: None)(symbol)
        if market and 'precision' in market and 'amount' in market['precision']:
             try:
                step_size = Decimal(str(market['precision']['amount']))
                if step_size <= 0: raise ValueError("Step size must be positive")
                formatted = amount_dec.quantize(step_size, rounding=ROUND_DOWN) # Use ROUND_DOWN for amounts
                decimal_places = abs(step_size.normalize().as_tuple().exponent)
                return f"{formatted:.{decimal_places}f}"
             except Exception as format_err: logger.error(f"[format_amount] Fallback quantize failed: {format_err}"); return f"{amount_dec:.8f}"
        else: return f"{amount_dec:.8f}"
    except Exception as e: logger.error(f"[format_amount] Error formatting amount '{amount}': {e}", exc_info=False); return "Error"

def format_order_id(order_id: Optional[str]) -> str:
    """Format an order ID for concise logging (shows first/last parts)."""
    if not order_id: return "N/A"
    try:
        id_str = str(order_id).strip()
        if len(id_str) <= 10: return id_str
        return f"{id_str[:4]}...{id_str[-4:]}"
    except Exception as e: logger.error(f"Error formatting order ID {order_id}: {e}"); return "UNKNOWN"

def send_sms_alert(message: str, sms_config: SMSConfig) -> bool:
    """Send an SMS alert using configured method (Termux or Twilio placeholder)."""
    if not sms_config.enable_sms_alerts: logger.debug(f"SMS suppressed (disabled): {message}"); return True
    recipient = sms_config.sms_recipient_number
    if not recipient: logger.warning("SMS enabled but no recipient configured."); return False

    # --- Termux API Method ---
    if sms_config.use_termux_api:
        timeout = sms_config.sms_timeout_seconds
        try:
            logger.info(f"Attempting Termux SMS to {recipient}...")
            command = ["termux-sms-send", "-n", recipient, message]
            result = subprocess.run(
                command, timeout=timeout, check=True, capture_output=True, text=True)
            logger.info(f"{Fore.GREEN}Termux SMS Sent OK.{Style.RESET_ALL} Output: {result.stdout.strip() or '(None)'}")
            return True
        except FileNotFoundError: logger.error(f"{Fore.RED}Termux 'termux-sms-send' not found.{Style.RESET_ALL}"); return False
        except subprocess.TimeoutExpired: logger.error(f"{Fore.RED}Termux SMS timed out ({timeout}s).{Style.RESET_ALL}"); return False
        except subprocess.CalledProcessError as e: logger.error(f"{Fore.RED}Termux SMS failed (Code:{e.returncode}): {e.stderr.strip() or '(No stderr)'}{Style.RESET_ALL}"); return False
        except Exception as e: logger.critical(f"{Fore.RED}Unexpected Termux SMS error: {e}{Style.RESET_ALL}", exc_info=True); return False

    # --- Twilio API Method (Placeholder) ---
    # elif sms_config.use_twilio_api:
    #     logger.warning("Twilio SMS sending not implemented.")
    #     return False # Not implemented

    else:
        logger.error("SMS enabled, but no provider (Termux/Twilio) configured/active.")
        return False

# --- Retry Decorator Factory ---
T = TypeVar("T")
_DEFAULT_HANDLED_EXCEPTIONS = (
    ccxt.RateLimitExceeded,
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
) if ccxt else () # Empty tuple if ccxt fails import

def retry_api_call(
    max_retries_override: Optional[int] = None,
    initial_delay_override: Optional[float] = None,
    handled_exceptions = _DEFAULT_HANDLED_EXCEPTIONS,
    error_message_prefix: str = "API Call Failed",
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]: # Correct typing for async
    """Decorator factory to retry ASYNC API calls with config."""

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            app_config: Optional[AppConfig] = kwargs.get("app_config")
            if not app_config: app_config = next((arg for arg in args if isinstance(arg, AppConfig)), None)

            if not isinstance(app_config, AppConfig):
                func_name_log = func.__name__
                logger.critical(f"{Back.RED}{Fore.WHITE}FATAL: No AppConfig for @retry_api_call in '{func_name_log}'.{Style.RESET_ALL}")
                raise ValueError(f"AppConfig required for {func_name_log}")

            effective_max_retries = max_retries_override if max_retries_override is not None else app_config.api_config.retry_count
            effective_base_delay = initial_delay_override if initial_delay_override is not None else app_config.api_config.retry_delay_seconds
            func_name = func.__name__
            last_exception = None

            for attempt in range(effective_max_retries + 1):
                try:
                    if attempt > 0: logger.debug(f"Retrying {func_name} (Attempt {attempt+1}/{effective_max_retries+1})")
                    return await func(*args, **kwargs) # Await the async function

                except handled_exceptions as e:
                    last_exception = e
                    if attempt == effective_max_retries:
                        logger.error(f"{Fore.RED}{error_message_prefix}: Max retries ({effective_max_retries+1}) for {func_name}. Last: {type(e).__name__} - {e}{Style.RESET_ALL}")
                        send_sms_alert(f"{error_message_prefix}: Max retries {func_name} ({type(e).__name__})", app_config.sms_config)
                        raise e

                    delay = effective_base_delay
                    log_level, log_color = logging.WARNING, Fore.YELLOW
                    if isinstance(e, ccxt.RateLimitExceeded):
                        delay *= 2 ** attempt # Exponential backoff
                        retry_after = getattr(e, 'retry_after', None)
                        if retry_after: delay = max(delay, float(retry_after)); logger.warning(f"{log_color}Rate limit {func_name}. API suggests retry after {retry_after}s. Using {delay:.2f}s.{Style.RESET_ALL}")
                        else: logger.warning(f"{log_color}Rate limit {func_name}. Retry {attempt+1}/{effective_max_retries+1} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                    elif isinstance(e, (ccxt.NetworkError, ccxt.RequestTimeout)):
                        log_level, log_color = logging.ERROR, Fore.RED
                        logger.log(log_level, f"{log_color}Network/Timeout {func_name}. Retry {attempt+1}/{effective_max_retries+1} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                    elif isinstance(e, ccxt.ExchangeNotAvailable):
                        log_level, log_color = logging.ERROR, Fore.RED
                        delay = max(delay * 2, 10.0) # Longer delay for exchange issues
                        logger.log(log_level, f"{log_color}Exchange unavailable {func_name}. Retry {attempt+1}/{effective_max_retries+1} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                    else: logger.log(log_level, f"{log_color}Handled exception {type(e).__name__} in {func_name}. Retry {attempt+1}/{effective_max_retries+1} after {delay:.2f}s: {e}{Style.RESET_ALL}")

                    await asyncio.sleep(delay)

                except Exception as e:
                    logger.critical(f"{Back.RED}{Fore.WHITE}UNEXPECTED error in {func_name}: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
                    send_sms_alert(f"CRITICAL Error in {func_name}: {type(e).__name__}", app_config.sms_config)
                    raise e

            if last_exception: raise last_exception # Should be unreachable if loop finishes correctly
            raise RuntimeError(f"Retry loop for {func_name} finished unexpectedly.")

        return wrapper
    return decorator


# --- Order Book Analysis (Removed - Redundant with helper function) ---
# If needed, reimplement here ensuring it takes AppConfig
# async def analyze_order_book(...) -> dict[str, Optional[Decimal]]: ...
EOF

# --- Create indicators.py ---
echo "# Creating indicators.py"
cat << 'EOF' > indicators.py
#!/usr/bin/env python
"""Technical Indicators Module (v1.1)

Provides functions to calculate various technical indicators, primarily leveraging the
`pandas_ta` library. Includes standard indicators, pivot points, level calculations,
and custom indicators like Ehlers Volumetric Trend.

Designed to work with pandas DataFrames containing OHLCV data and uses a
configuration dictionary derived from the Pydantic models.
"""

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    print(
        "FATAL ERROR [indicators]: 'pandas_ta' library not found. Please install it: pip install pandas_ta"
    )
    sys.exit(1)

# --- Import Pydantic models for config type hinting ---
from config_models import AppConfig, IndicatorSettings, AnalysisFlags

# --- Setup ---
logger = logging.getLogger(__name__) # Get logger configured in main script

# --- Constants ---
MIN_PERIODS_DEFAULT = 50 # Default minimum data points

# --- Helper Functions ---
# Using pandas_ta and numpy handles most float ops efficiently

# --- Pivot Point Calculations (Standard & Fibonacci) ---
def calculate_standard_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates standard pivot points for the *next* period."""
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]): return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for pivot calc."); # Optional: low, high = high, low
    pivots = {}
    try:
        pivot = (high + low + close) / 3.0
        pivots["PP"] = pivot; pivots["S1"] = (2 * pivot) - high; pivots["R1"] = (2 * pivot) - low
        pivots["S2"] = pivot - (high - low); pivots["R2"] = pivot + (high - low)
        pivots["S3"] = low - 2 * (high - pivot); pivots["R3"] = high + 2 * (pivot - low)
    except Exception as e: logger.error(f"Error standard pivots: {e}", exc_info=False); return {}
    return pivots

def calculate_fib_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates Fibonacci pivot points for the *next* period."""
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]): return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for Fib pivot calc."); # Optional: low, high = high, low
    fib_pivots = {}
    try:
        pivot = (high + low + close) / 3.0; fib_range = high - low
        if abs(fib_range) < 1e-9: fib_pivots["PP"] = pivot; return fib_pivots # Range near zero
        fib_pivots["PP"] = pivot; fib_pivots["S1"] = pivot - (0.382 * fib_range); fib_pivots["R1"] = pivot + (0.382 * fib_range)
        fib_pivots["S2"] = pivot - (0.618 * fib_range); fib_pivots["R2"] = pivot + (0.618 * fib_range)
        fib_pivots["S3"] = pivot - (1.000 * fib_range); fib_pivots["R3"] = pivot + (1.000 * fib_range)
    except Exception as e: logger.error(f"Error Fib pivots: {e}", exc_info=False); return {}
    return fib_pivots

# --- Support / Resistance Level Calculation ---
def calculate_levels(df_period: pd.DataFrame, current_price: Optional[float] = None) -> Dict[str, Any]:
    """Calculates various support/resistance levels based on historical data."""
    levels: Dict[str, Any] = { "support": {}, "resistance": {}, "pivot": None, "fib_retracements": {}, "standard_pivots": {}, "fib_pivots": {} }
    required_cols = ["high", "low", "close"]
    if df_period is None or df_period.empty or not all(col in df_period.columns for col in required_cols): logger.debug("Levels calc skipped: Invalid DF."); return levels
    if len(df_period) < 2: logger.debug("Levels calc skipped: Need >= 2 rows for pivots.");

    try:
        if len(df_period) >= 2:
            prev_high = df_period["high"].iloc[-2]; prev_low = df_period["low"].iloc[-2]; prev_close = df_period["close"].iloc[-2]
            standard_pivots = calculate_standard_pivot_points(prev_high, prev_low, prev_close)
            if standard_pivots: levels["standard_pivots"] = standard_pivots; levels["pivot"] = standard_pivots.get("PP")
            fib_pivots = calculate_fib_pivot_points(prev_high, prev_low, prev_close)
            if fib_pivots: levels["fib_pivots"] = fib_pivots; if levels["pivot"] is None: levels["pivot"] = fib_pivots.get("PP")

        period_high = df_period["high"].max(); period_low = df_period["low"].min(); period_diff = period_high - period_low
        if abs(period_diff) > 1e-9:
            levels["fib_retracements"] = {
                "High": period_high, "Fib 78.6%": period_low + period_diff*0.786, "Fib 61.8%": period_low + period_diff*0.618,
                "Fib 50.0%": period_low + period_diff*0.5, "Fib 38.2%": period_low + period_diff*0.382,
                "Fib 23.6%": period_low + period_diff*0.236, "Low": period_low }

        if current_price is not None and isinstance(current_price, (int, float)):
            all_lvls = { **{f"Std {k}": v for k, v in levels["standard_pivots"].items()}, **{f"FibPiv {k}": v for k, v in levels["fib_pivots"].items() if k != "PP"}, **levels["fib_retracements"] }
            for lbl, val in all_lvls.items():
                if isinstance(val, (int, float)):
                    if val < current_price: levels["support"][lbl] = val
                    elif val > current_price: levels["resistance"][lbl] = val

    except IndexError: logger.warning("IndexError calculating levels (data < 2 rows). Retracements only.")
    except Exception as e: logger.error(f"Error calculating S/R levels: {e}", exc_info=True)

    levels["support"] = dict(sorted(levels["support"].items(), key=lambda item: item[1], reverse=True))
    levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda item: item[1]))
    return levels

# --- Custom Indicator Implementations ---
def calculate_vwma(close: pd.Series, volume: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series) or close.empty or volume.empty or length <= 0 or len(close) < length or len(close) != len(volume): return None
    try:
        pv = close * volume; cumulative_pv = pv.rolling(window=length, min_periods=length).sum()
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()
        vwma = cumulative_pv / cumulative_vol.replace(0, np.nan); vwma.name = f"VWMA_{length}"; return vwma
    except Exception as e: logger.error(f"Error VWMA(len={length}): {e}", exc_info=True); return None

def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
    """Calculates Ehlers Volumetric Trend indicator using VWMA and SuperSmoother."""
    required_cols = ["close", "volume"]
    if not all(col in df.columns for col in required_cols) or df.empty or length <= 1 or multiplier <= 0:
        logger.warning(f"EVT skipped: Invalid input (len={length}, mult={multiplier}).")
        return df
    df_out = df.copy()
    vwma_col, smooth_col = f"vwma_{length}", f"smooth_vwma_{length}"
    trend_col, buy_col, sell_col = f"evt_trend_{length}", f"evt_buy_{length}", f"evt_sell_{length}"
    try:
        vwma = calculate_vwma(df_out["close"], df_out["volume"], length=length)
        if vwma is None or vwma.isnull().all(): raise ValueError(f"VWMA failed for EVT(len={length})")
        df_out[vwma_col] = vwma
        a = np.exp(-1.414 * np.pi / length); b = 2 * a * np.cos(1.414 * np.pi / length)
        c2, c3, c1 = b, -a * a, 1 - c2 - c3
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        vwma_vals = df_out[vwma_col].values
        for i in range(2, len(df_out)):
            if pd.notna(vwma_vals[i]) and pd.notna(vwma_vals[i-1]) and pd.notna(vwma_vals[i-2]):
                 sm1 = smoothed.iloc[i-1] if pd.notna(smoothed.iloc[i-1]) else vwma_vals[i-1]
                 sm2 = smoothed.iloc[i-2] if pd.notna(smoothed.iloc[i-2]) else vwma_vals[i-2]
                 smoothed.iloc[i] = c1 * vwma_vals[i] + c2 * sm1 + c3 * sm2
        df_out[smooth_col] = smoothed
        mult_h, mult_l = 1.0 + multiplier / 100.0, 1.0 - multiplier / 100.0
        shifted_smooth = df_out[smooth_col].shift(1)
        trend = pd.Series(0, index=df_out.index, dtype=int)
        up_cond = (df_out[smooth_col] > shifted_smooth * mult_h); down_cond = (df_out[smooth_col] < shifted_smooth * mult_l)
        trend[up_cond] = 1; trend[down_cond] = -1
        trend = trend.replace(0, np.nan).ffill().fillna(0).astype(int)
        df_out[trend_col] = trend
        trend_shifted = df_out[trend_col].shift(1, fill_value=0)
        df_out[buy_col] = (df_out[trend_col] == 1) & (trend_shifted != 1)
        df_out[sell_col] = (df_out[trend_col] == -1) & (trend_shifted != -1)
        logger.debug(f"EVT(len={length}, mult={multiplier}) calculated.")
        return df_out
    except Exception as e:
        logger.error(f"Error in EVT(len={length}, mult={multiplier}): {e}", exc_info=True)
        for col in [vwma_col, smooth_col, trend_col, buy_col, sell_col]:
            if col not in df.columns: df[col] = np.nan
        return df

# --- Master Indicator Calculation Function ---
def calculate_all_indicators(df: pd.DataFrame, app_config: AppConfig) -> pd.DataFrame:
    """Calculates enabled technical indicators based on AppConfig."""
    func_name = "calculate_all_indicators"
    if df is None or df.empty: logger.error(f"[{func_name}] Input DataFrame is empty."); return pd.DataFrame()
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols): logger.error(f"[{func_name}] Missing required columns: {[c for c in required_cols if c not in df.columns]}."); return df.copy()

    df_out = df.copy()
    settings: IndicatorSettings = app_config.strategy_config.indicator_settings
    flags: AnalysisFlags = app_config.strategy_config.analysis_flags
    min_rows_needed = settings.min_data_periods

    if len(df_out.dropna(subset=required_cols)) < min_rows_needed: logger.warning(f"[{func_name}] Insufficient valid rows ({len(df_out.dropna(subset=required_cols))}) < {min_rows_needed}.")

    logger.debug(f"Calculating indicators with settings: {settings.model_dump()} flags: {flags.model_dump()}")
    try:
        if flags.use_atr:
             if settings.atr_period > 0: df_out.ta.atr(length=settings.atr_period, append=True)
             else: logger.warning("ATR skipped: atr_period <= 0")
        if flags.use_evt:
            df_out = ehlers_volumetric_trend(df_out, settings.evt_length, float(settings.evt_multiplier)) # Ensure multiplier is float
        # Add other pandas_ta indicators based on flags...
        # Example: if flags.use_rsi and settings.rsi_period > 0: df_out.ta.rsi(length=settings.rsi_period, append=True)

        df_out = df_out.loc[:, ~df_out.columns.duplicated()] # Remove potential duplicate columns
    except Exception as e: logger.error(f"[{func_name}] Error during indicator calculation: {e}", exc_info=True)

    logger.debug(f"[{func_name}] Finished. Shape: {df_out.shape}")
    return df_out
EOF

# --- Create bybit_helper_functions.py ---
echo "# Creating bybit_helper_functions.py"
cat << 'EOF' > bybit_helper_functions.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bybit V5 CCXT Helper Functions (v3.4 - Pydantic Config Integrated)

Collection of helper functions for Bybit V5 API using CCXT, now integrating
with Pydantic models defined in `config_models.py` for configuration.
"""

# Standard Library Imports
import logging
import os
import sys
import time
import random
import json
from decimal import Decimal, ROUND_HALF_UP, getcontext, InvalidOperation, DivisionByZero
from typing import (Optional, Dict, List, Tuple, Any, Literal, Union,
                    TypedDict, Callable, Coroutine, TypeVar, Sequence)
from enum import Enum
import asyncio
import math

# --- Import Pydantic models first ---
from config_models import AppConfig, APIConfig # Use the unified Pydantic config

# Third-party Libraries
try:
    import ccxt.async_support as ccxt
    from ccxt.base.errors import (
        ExchangeError, NetworkError, RateLimitExceeded, AuthenticationError,
        OrderNotFound, InvalidOrder, InsufficientFunds, ExchangeNotAvailable,
        NotSupported, OrderImmediatelyFillable, BadSymbol, ArgumentsRequired,
        RequestTimeout
    )
    from ccxt.base.decimal_to_precision import ROUND_UP, ROUND_DOWN
except ImportError:
    print("FATAL ERROR [bybit_helpers]: CCXT library not found.", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    # print("Warning: pandas library not found. OHLCV will use lists.", file=sys.stderr)
    pd = None

try:
    from colorama import Fore, Style, Back, init
    if os.name == 'nt': init(autoreset=True)
    else: init(autoreset=True)
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

try:
    import websockets
    from websockets.exceptions import WebSocketException, ConnectionClosed, ConnectionClosedOK, ConnectionClosedError, InvalidURI
except ImportError:
    # print("Warning: websockets library not found. WebSocket features disabled.", file=sys.stderr)
    websockets = None
    class DummyWebSocketException(Exception): pass
    WebSocketException = ConnectionClosed = ConnectionClosedOK = ConnectionClosedError = InvalidURI = DummyWebSocketException


# --- Configuration & Constants ---
getcontext().prec = 30 # Decimal precision

# Enums (Imported from config_models or defined as fallback there)
from config_models import (
    PositionIdx, Category, OrderFilter, Side, TimeInForce, TriggerBy, TriggerDirection
)

# Define OrderType here if not fully imported from config_models fallback
OrderType = Literal['Limit', 'Market']
StopLossTakeProfitMode = Literal['Full', 'Partial']


# --- Logger Setup ---
logger = logging.getLogger(__name__) # Get logger from main script

# --- Market Cache ---
class MarketCache:
    def __init__(self):
        self._markets: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, Optional[Category]] = {}
        self._lock = asyncio.Lock()

    async def load_markets(self, exchange: ccxt.bybit, reload: bool = False) -> None:
        async with self._lock:
            if not self._markets or reload:
                action = 'Reloading' if reload else 'Loading'; logger.info(f"{Fore.BLUE}[MarketCache] {action} markets...{Style.RESET_ALL}")
                try:
                    all_markets = await exchange.load_markets(reload=reload)
                    if not all_markets: logger.critical(f"{Back.RED}FATAL: Failed load markets - empty.{Style.RESET_ALL}"); self._markets={}; self._categories={}; raise ExchangeError("Failed load markets: empty.")
                    self._markets = all_markets; self._categories.clear(); logger.success(f"{Fore.GREEN}[MarketCache] Loaded {len(self._markets)} markets.{Style.RESET_ALL}")
                except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e: logger.error(f"{Fore.RED}[MarketCache] Network error loading: {e}{Style.RESET_ALL}")
                except ExchangeError as e: logger.error(f"{Fore.RED}[MarketCache] Exchange error loading: {e}{Style.RESET_ALL}", exc_info=False); self._markets={}; self._categories={}; raise
                except Exception as e: logger.critical(f"{Back.RED}[MarketCache] CRITICAL error loading: {e}{Style.RESET_ALL}", exc_info=True); self._markets={}; self._categories={}; raise

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        market_data = self._markets.get(symbol); #if not market_data: logger.debug(f"[MarketCache] Market '{symbol}' not found.")
        return market_data

    def get_category(self, symbol: str) -> Optional[Category]:
        if symbol in self._categories: return self._categories[symbol]
        market = self.get_market(symbol); category: Optional[Category] = None
        if market:
            category_str = _get_v5_category(market)
            if category_str:
                try: category = Category(category_str)
                except ValueError: logger.error(f"[MarketCache] Invalid category '{category_str}' for '{symbol}'."); category = None
        self._categories[symbol] = category # Cache result
        return category

    def get_all_symbols(self) -> List[str]: return list(self._markets.keys())

market_cache = MarketCache()

# --- Utility Functions (Imported from bybit_utils) ---
from bybit_utils import safe_decimal_conversion, format_price, format_amount, format_order_id, send_sms_alert

def _get_v5_category(market: Dict[str, Any]) -> Optional[str]:
    if not market: return None
    if market.get('spot', False): return Category.SPOT.value
    if market.get('option', False): return Category.OPTION.value
    if market.get('linear', False): return Category.LINEAR.value
    if market.get('inverse', False): return Category.INVERSE.value
    market_type = market.get('type'); symbol = market.get('symbol', 'N/A')
    if market_type == 'spot': return Category.SPOT.value
    if market_type == 'option': return Category.OPTION.value
    if market_type in ['swap', 'future']:
        contract_type = str(market.get('info', {}).get('contractType', '')).lower()
        settle_coin = market.get('settle', '').upper()
        if contract_type == 'linear' or settle_coin in ['USDT', 'USDC']: return Category.LINEAR.value
        if contract_type == 'inverse': return Category.INVERSE.value
        base_coin = market.get('base', '').upper()
        if settle_coin == base_coin and settle_coin: return Category.INVERSE.value
        logger.warning(f"[_get_v5_category] Assuming LINEAR for derivative {symbol}.")
        return Category.LINEAR.value
    logger.warning(f"[_get_v5_category] Unknown market type '{market_type}' for {symbol}.")
    return None


# --- Asynchronous Retry Decorator (Imported from bybit_utils) ---
from bybit_utils import retry_api_call

# --- Exchange Initialization & Configuration ---
@retry_api_call() # Uses AppConfig passed into wrapper
async def initialize_bybit(app_config: AppConfig, use_async: bool = True) -> Optional[Union[ccxt.bybit, ccxt.Exchange]]:
    """ Initializes the Bybit CCXT exchange instance using AppConfig. """
    func_name = "initialize_bybit"; api_conf = app_config.api_config
    mode = 'Testnet' if api_conf.testnet_mode else 'Mainnet'; async_mode = "Async" if use_async else "Sync"
    logger.info(f"{Fore.BLUE}[{func_name}] Initializing Bybit V5 ({mode}, {async_mode})...{Style.RESET_ALL}")
    exchange: Optional[Union[ccxt.bybit, ccxt.Exchange]] = None
    try:
        has_keys = bool(api_conf.api_key and api_conf.api_secret)
        if not has_keys: logger.warning(f"{Fore.YELLOW}[{func_name}] API Keys missing. PUBLIC mode.{Style.RESET_ALL}")
        exchange_options = {
            'apiKey': api_conf.api_key, 'secret': api_conf.api_secret, 'enableRateLimit': True,
            'options': {
                'defaultType': api_conf.expected_market_type, 'adjustForTimeDifference': True,
                'recvWindow': api_conf.default_recv_window,
                'brokerId': f"PB_{app_config.strategy_config.name[:10]}", # Short strategy name in ID
            }}
        exchange_class = ccxt.bybit if use_async else getattr(__import__('ccxt'), 'bybit')
        exchange = exchange_class(exchange_options)
        if api_conf.testnet_mode: exchange.set_sandbox_mode(True)
        logger.info(f"[{func_name}] {mode}. Endpoint: {exchange.urls['api']}")
        await market_cache.load_markets(exchange, reload=True)
        if not market_cache.get_market(api_conf.symbol):
            logger.critical(f"{Back.RED}FATAL: Failed load market '{api_conf.symbol}'.{Style.RESET_ALL}"); await exchange.close(); return None
        if has_keys:
            logger.info(f"[{func_name}] Performing auth check (fetch balance)...")
            try: await fetch_usdt_balance(exchange, app_config=app_config); logger.info(f"[{func_name}] Auth check OK.")
            except AuthenticationError as auth_err: logger.critical(f"{Back.RED}CRITICAL: Auth FAILED: {auth_err}.{Style.RESET_ALL}"); send_sms_alert(f"[BybitHelper] CRITICAL: Auth Failed!", app_config.sms_config); await exchange.close(); return None
            except ExchangeError as bal_err: logger.warning(f"{Fore.YELLOW}[{func_name}] Warn during auth check: {bal_err}.{Style.RESET_ALL}")
        else: logger.info(f"[{func_name}] Skipping auth check (no keys).")
        logger.success(f"{Fore.GREEN}[{func_name}] Bybit V5 exchange initialized OK.{Style.RESET_ALL}")
        return exchange
    except AuthenticationError as e: logger.critical(f"{Back.RED}Auth failed: {e}.{Style.RESET_ALL}"); send_sms_alert(f"[BybitHelper] CRITICAL: Auth Failed!", app_config.sms_config)
    except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e: logger.critical(f"{Back.RED}Network error init: {e}.{Style.RESET_ALL}")
    except ExchangeError as e: logger.critical(f"{Back.RED}Exchange error init: {e}{Style.RESET_ALL}", exc_info=False); send_sms_alert(f"[BybitHelper] CRITICAL: Init ExchangeError: {type(e).__name__}", app_config.sms_config)
    except Exception as e: logger.critical(f"{Back.RED}Unexpected error init: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[BybitHelper] CRITICAL: Init Unexpected Error: {type(e).__name__}", app_config.sms_config)
    if exchange: try: logger.info(f"[{func_name}] Closing partial exchange."); await exchange.close(); except Exception: pass
    return None

# --- Account Functions ---
@retry_api_call()
async def set_leverage(exchange: ccxt.bybit, symbol: str, leverage: int, app_config: AppConfig) -> bool:
    """ Sets leverage, using AppConfig. """
    func_name = "set_leverage"; log_prefix = f"[{func_name} ({symbol} -> {leverage}x)]"
    if leverage <= 0: logger.error(f"{Fore.RED}{log_prefix} Leverage > 0 req.{Style.RESET_ALL}"); return False
    category = market_cache.get_category(symbol); if not category or category not in [Category.LINEAR, Category.INVERSE]: logger.error(f"{Fore.RED}{log_prefix} Requires LINEAR/INVERSE. Got: {category}.{Style.RESET_ALL}"); return False
    market = market_cache.get_market(symbol); if not market: logger.error(f"{Fore.RED}{log_prefix} Market {symbol} unavailable.{Style.RESET_ALL}"); return False
    try: # Validate leverage limits
        limits = market.get('limits', {}).get('leverage', {})
        max_lev, min_lev = safe_decimal_conversion(limits.get('max'), 100), safe_decimal_conversion(limits.get('min'), 1)
        if not (min_lev <= leverage <= max_lev): logger.error(f"{Fore.RED}{log_prefix} Lev {leverage}x outside range [{min_lev}x - {max_lev}x].{Style.RESET_ALL}"); return False
    except Exception as e: logger.warning(f"{Fore.YELLOW}{log_prefix} Cannot validate lev limits: {e}.{Style.RESET_ALL}")
    params = {'category': category.value, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
    logger.info(f"{Fore.CYAN}{log_prefix} Sending request...{Style.RESET_ALL}")
    try:
        await exchange.set_leverage(leverage, symbol, params=params)
        logger.success(f"{Fore.GREEN}{log_prefix} Set OK (Implies ISOLATED).{Style.RESET_ALL}"); return True
    except ExchangeError as e: # V5 error codes
        code = getattr(e, 'code', None); error_str = str(e).lower()
        if code == 110043 or "leverage not modified" in error_str: logger.info(f"{Fore.YELLOW}{log_prefix} Already set to {leverage}x.{Style.RESET_ALL}"); return True
        elif code == 110021: logger.error(f"{Fore.RED}{log_prefix} Failed ({code}): {e}. Check hedge mode?{Style.RESET_ALL}"); return False
        else: logger.error(f"{Fore.RED}{log_prefix} ExchangeError: {e}{Style.RESET_ALL}", exc_info=False); return False
    except NetworkError as e: logger.warning(f"{Fore.YELLOW}{log_prefix} Network error: {e}. Retry handled.{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return False

@retry_api_call()
async def fetch_usdt_balance(exchange: ccxt.bybit, app_config: AppConfig) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """ Fetches USDT balance from UNIFIED account using AppConfig. """
    func_name = "fetch_usdt_balance"; log_prefix = f"[{func_name}]"
    usdt = app_config.api_config.usdt_symbol; logger.debug(f"{log_prefix} Fetching UNIFIED balance ({usdt})...")
    try:
        bal = await exchange.fetch_balance(params={'accountType': 'UNIFIED'})
        equity, avail = None, None
        info_list = bal.get('info', {}).get('result', {}).get('list', [])
        if info_list:
            uni = next((a for a in info_list if a.get('accountType') == 'UNIFIED'), None)
            if uni:
                equity = safe_decimal_conversion(uni.get('totalEquity'))
                coin = next((c for c in uni.get('coin', []) if c.get('coin') == usdt), None)
                if coin: avail = safe_decimal_conversion(coin.get('availableToWithdraw') or coin.get('availableBalance'))
        if equity is None and usdt in bal: equity = safe_decimal_conversion(bal[usdt].get('total'))
        if avail is None and usdt in bal: avail = safe_decimal_conversion(bal[usdt].get('free'))
        if equity is None: logger.warning(f"{log_prefix} Cannot get total equity. Default 0.")
        if avail is None: logger.warning(f"{log_prefix} Cannot get available balance. Default 0.")
        final_eq, final_av = max(Decimal("0"), equity or 0), max(Decimal("0"), avail or 0)
        logger.info(f"{Fore.GREEN}{log_prefix} OK - Equity:{final_eq:.4f}, Avail:{final_av:.4f}{Style.RESET_ALL}")
        return final_eq, final_av
    except AuthenticationError as e: logger.error(f"{Fore.RED}{log_prefix} Auth error: {e}{Style.RESET_ALL}"); return None, None
    except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e: logger.warning(f"{Fore.YELLOW}{log_prefix} Network error: {e}. Retry handled.{Style.RESET_ALL}"); raise
    except ExchangeError as e: logger.error(f"{Fore.RED}{log_prefix} Exchange error: {e}{Style.RESET_ALL}", exc_info=False); return None, None
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return None, None

# --- Market Data Functions ---
@retry_api_call()
async def fetch_ohlcv_paginated(
    exchange: ccxt.bybit, symbol: str, timeframe: str, app_config: AppConfig,
    since: Optional[int] = None, limit: Optional[int] = None
) -> Optional[Union[pd.DataFrame, List[list]]]:
    """ Fetches OHLCV data using AppConfig for retry settings. """
    func_name = "fetch_ohlcv_paginated"; log_prefix = f"[{func_name} ({symbol}, {timeframe})]"
    market = market_cache.get_market(symbol); category = market_cache.get_category(symbol)
    if not market or not category: logger.error(f"{Fore.RED}{log_prefix} Invalid market/category.{Style.RESET_ALL}"); return None
    try: tf_ms = exchange.parse_timeframe(timeframe) * 1000
    except Exception as e: logger.error(f"{log_prefix} Invalid timeframe: {e}."); return None
    fetch_limit_req, all_candles, loops, max_loops = 1000, [], 0, 200
    retries_chunk = app_config.api_config.retry_count; delay_chunks = exchange.rateLimit / 1000 if exchange.enableRateLimit else 0.2
    base_retry_delay = app_config.api_config.retry_delay_seconds
    current_since = since; logger.info(f"{Fore.BLUE}{log_prefix} Fetching...{Style.RESET_ALL}")
    params = {'category': category.value}
    try:
        while loops < max_loops:
            loops += 1; current_limit = fetch_limit_req
            if limit is not None: remaining = limit - len(all_candles); if remaining <= 0: logger.info(f"{log_prefix} Limit {limit} reached."); break; current_limit = min(fetch_limit_req, remaining)
            logger.debug(f"{log_prefix} Loop {loops}, Fetch since={current_since} (Limit:{current_limit})...")
            chunk, last_err = None, None
            for attempt in range(retries_chunk + 1): # Retry logic for individual chunks
                try: chunk = await exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=current_limit, params=params); last_err=None; break
                except (NetworkError, RequestTimeout, ExchangeNotAvailable, RateLimitExceeded) as e: last_err=e; if attempt == retries_chunk: logger.error(f"{Fore.RED}{log_prefix} Chunk fail after {retries_chunk+1}: {e}{Style.RESET_ALL}"); break; else: wait = base_retry_delay * (2**attempt) + random.uniform(0, base_retry_delay*0.2); logger.warning(f"{Fore.YELLOW}{log_prefix} Chunk attempt {attempt+1} fail: {type(e).__name__}. Retry in {wait:.2f}s...{Style.RESET_ALL}"); await asyncio.sleep(wait)
                except ExchangeError as e: last_err=e; logger.error(f"{Fore.RED}{log_prefix} ExchangeError chunk: {e}. Abort.{Style.RESET_ALL}"); break
                except Exception as e: last_err=e; logger.error(f"{Fore.RED}{log_prefix} Unexpected err chunk: {e}{Style.RESET_ALL}", exc_info=True); break
            if last_err: logger.error(f"{Fore.RED}{log_prefix} Abort pagination: chunk fail.{Style.RESET_ALL}"); break
            if not chunk: logger.info(f"{log_prefix} No more candles."); break
            if all_candles and chunk[0][0] <= all_candles[-1][0]: chunk = [c for c in chunk if c[0] > all_candles[-1][0]]; if not chunk: logger.debug(f"{log_prefix} All candles duplicated."); break
            all_candles.extend(chunk); last_ts = chunk[-1][0]; first_ts = chunk[0][0]; ts_to_dt = lambda ts: pd.to_datetime(ts, unit='ms').strftime("%H:%M:%S") if pd else str(ts)[-9:-4]
            logger.info(f"{log_prefix} Fetched {len(chunk)} ({ts_to_dt(first_ts)}-{ts_to_dt(last_ts)}). Total: {len(all_candles)}")
            current_since = last_ts + 1; await asyncio.sleep(delay_chunks)
            if len(chunk) < current_limit: logger.info(f"{log_prefix} Received < limit."); break
        logger.info(f"{log_prefix} Total raw collected: {len(all_candles)}")
        if not all_candles: logger.warning(f"{log_prefix} No candles found."); return pd.DataFrame() if pd else []
        if pd: # Process to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True); df.set_index('datetime', inplace=True)
            for col in ['open','high','low','close','volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.loc[~df.index.duplicated(keep='first')]; df.sort_index(inplace=True)
            logger.success(f"{Fore.GREEN}{log_prefix} Processed {len(df)} unique candles (DF).{Style.RESET_ALL}"); return df
        else: # Process as list
            all_candles.sort(key=lambda x: x[0]); unique = []; seen = set()
            for c in all_candles: if c[0] not in seen: unique.append(c); seen.add(c[0])
            logger.success(f"{Fore.GREEN}{log_prefix} Processed {len(unique)} unique candles (List).{Style.RESET_ALL}"); return unique
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded, ExchangeError) as e: logger.error(f"{Fore.RED}{log_prefix} API/Exch error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Unexpected error: {e}{Style.RESET_ALL}", exc_info=True)
    if all_candles: # Return partial on error if some data was fetched
        logger.warning(f"{log_prefix}: Returning partial data ({len(all_candles)}).")
        if pd: try: df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume']); df['datetime']=pd.to_datetime(df['timestamp'],unit='ms', utc=True); df.set_index('datetime',inplace=True); df=df[~df.index.duplicated(keep='first')]; df.sort_index(inplace=True); return df; except Exception: pass
        all_candles.sort(key=lambda x: x[0]); return all_candles
    return None

@retry_api_call()
async def fetch_ticker_validated(exchange: ccxt.bybit, symbol: str, app_config: AppConfig) -> Optional[Dict]:
    """ Fetches ticker, validates timestamp/keys, uses AppConfig. """
    func_name = "fetch_ticker"; log_prefix = f"[{func_name} ({symbol})]"
    logger.debug(f"{log_prefix} Fetching...")
    category = market_cache.get_category(symbol); if not category: logger.error(f"{Fore.RED}{log_prefix} No category.{Style.RESET_ALL}"); return None
    params = {'category': category.value}
    try:
        ticker = await exchange.fetch_ticker(symbol, params=params)
        req_k, com_k = ['symbol', 'last', 'bid', 'ask'], ['timestamp', 'datetime', 'high', 'low', 'quoteVolume']
        miss_k = [k for k in req_k if k not in ticker or ticker[k] is None]; if miss_k: logger.error(f"{Fore.RED}{log_prefix} Ticker miss keys: {miss_k}. Data: {ticker}{Style.RESET_ALL}"); return None
        miss_c = [k for k in com_k if k not in ticker or ticker[k] is None]; if miss_c: logger.debug(f"{log_prefix} Ticker miss common: {miss_c}.") # Debug level for common keys
        ts_ms = ticker.get('timestamp'); now_ms = int(time.time() * 1000); max_age_s, min_age_s = 90, -10; max_diff, min_diff = max_age_s*1000, min_age_s*1000
        ts_log = "TS: N/A"; ts_ok = False
        if ts_ms is None: ts_log = f"{Fore.YELLOW}TS: Miss{Style.RESET_ALL}"
        elif not isinstance(ts_ms, int): ts_log = f"{Fore.YELLOW}TS: Type ({type(ts_ms).__name__}){Style.RESET_ALL}"
        else: diff = now_ms - ts_ms; age = diff / 1000.0; dt_str = ticker.get('datetime', f"{ts_ms}")[-15:] # Shorter timestamp
              if diff > max_diff or diff < min_diff: logger.warning(f"{Fore.YELLOW}{log_prefix} Timestamp ({dt_str}) stale/invalid. Age: {age:.1f}s{Style.RESET_ALL}"); ts_log = f"{Fore.YELLOW}TS: Stale ({age:.1f}s){Style.RESET_ALL}"
              else: ts_log = f"TS OK ({age:.1f}s)"; ts_ok = True
        logger.info(f"{Fore.GREEN}{log_prefix} OK: Last={ticker.get('last')}, Bid={ticker.get('bid')}, Ask={ticker.get('ask')} | {ts_log}{Style.RESET_ALL}")
        return ticker
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e: logger.warning(f"{Fore.YELLOW}{log_prefix} API error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}"); raise
    except AuthenticationError as e: logger.error(f"{Fore.RED}{log_prefix} Auth error: {e}{Style.RESET_ALL}"); return None
    except ExchangeError as e: logger.error(f"{Fore.RED}{log_prefix} Exchange error: {e}{Style.RESET_ALL}", exc_info=False); return None
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return None

# --- Other fetch functions (funding rate, order book, trades) omitted for brevity, adapt as above ---

# --- Order Management Functions (Adapted) ---
@retry_api_call(max_retries=1, initial_delay=0)
async def place_market_order_slippage_check(
    exchange: ccxt.bybit, symbol: str, side: Side, amount: Decimal, app_config: AppConfig,
    max_slippage_pct: Optional[Decimal]=None, is_reduce_only: bool=False,
    time_in_force: TimeInForce=TimeInForce.IOC, client_order_id: Optional[str]=None,
    position_idx: Optional[PositionIdx]=None, reason: str = "Market Order"
) -> Optional[Dict]:
    """ Places market order with spread check, uses AppConfig. """
    func_name = "place_market_order"; action = "ReduceOnly" if is_reduce_only else "Open/Increase"; log_prefix = f"[{func_name} ({symbol}, {side.value}, {amount}, {action}, {reason})]"
    api_conf = app_config.api_config
    if amount <= api_conf.position_qty_epsilon: logger.error(f"{Fore.RED}{log_prefix} Invalid amount: {amount}.{Style.RESET_ALL}"); return None
    category = market_cache.get_category(symbol); market = market_cache.get_market(symbol); if not category or not market: logger.error(f"{Fore.RED}{log_prefix} Invalid cat/market.{Style.RESET_ALL}"); return None
    fmt_amt = format_amount(exchange, symbol, amount); if fmt_amt is None: logger.error(f"{Fore.RED}{log_prefix} Fail format amount.{Style.RESET_ALL}"); return None
    fmt_amt_float = float(fmt_amt); eff_slip = max_slippage_pct if max_slippage_pct is not None else api_conf.default_slippage_pct
    logger.info(f"{Fore.BLUE}{log_prefix} Placing. Amt:{fmt_amt}, TIF:{time_in_force.value}, SpreadChk:{eff_slip:.4%}{Style.RESET_ALL}")
    try: # Spread Check
        ob = await fetch_l2_order_book_validated(exchange, symbol, api_conf.shallow_ob_fetch_depth, app_config)
        if ob and ob.get('bids') and ob.get('asks'):
            bid, ask = safe_decimal_conversion(ob['bids'][0][0]), safe_decimal_conversion(ob['asks'][0][0])
            if bid and ask and bid > 0: mid = (bid + ask) / 2; spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 0; logger.debug(f"{log_prefix} Spread: {spread_pct:.4f}%")
                 if spread_pct > (eff_slip * 100): logger.error(f"{Back.RED}ABORT: Spread {spread_pct:.4f}% > Max {eff_slip*100:.4f}%{Style.RESET_ALL}"); send_sms_alert(f"[{symbol}] MKT ABORT: Spread {spread_pct:.4f}%>{eff_slip*100:.4f}%", app_config.sms_config); return None
            else: logger.warning(f"{Fore.YELLOW}{log_prefix} Invalid bid/ask. Skip spread check.{Style.RESET_ALL}")
        else: logger.warning(f"{Fore.YELLOW}{log_prefix} No shallow OB. Skip spread check.{Style.RESET_ALL}")
    except Exception as ob_err: logger.warning(f"{Fore.YELLOW}{log_prefix} Error spread check: {ob_err}. Proceeding.{Style.RESET_ALL}")
    params: Dict[str, Any] = {'category': category.value, 'reduceOnly': is_reduce_only, 'timeInForce': time_in_force.value}
    params['clientOrderId'] = client_order_id or f"{reason[:5]}_{symbol[:3]}_{side.value}_{int(time.time()*1000)}"[-36:] # Auto/Clean CID
    if position_idx is not None: params['positionIdx'] = position_idx.value if isinstance(position_idx, Enum) else int(position_idx)
    try:
        logger.info(f"{log_prefix} Sending create_market... CID: {params.get('clientOrderId')}")
        order = await exchange.create_market_order(symbol, side.value, fmt_amt_float, params=params)
        oid, status, filled, avg_px = order.get('id'), order.get('status', '?'), safe_decimal_conversion(order.get('filled', 0)), safe_decimal_conversion(order.get('average'))
        log_clr = Fore.GREEN if status in ['closed', 'filled'] else Fore.YELLOW
        logger.success(f"{log_clr}{log_prefix} OK - ID:{format_order_id(oid)}, Stat:{status}, Fill:{format_amount(exchange, symbol, filled)} @ Avg:{format_price(exchange, symbol, avg_px)}{Style.RESET_ALL}")
        if time_in_force in [TimeInForce.IOC, TimeInForce.FOK] and filled < amount * (1 - api_conf.position_qty_epsilon): logger.warning(f"{Fore.YELLOW}{log_prefix} Order {oid} ({time_in_force.value}) PARTIAL FILL ({filled}/{amount}).{Style.RESET_ALL}")
        return order
    except InsufficientFunds as e: logger.error(f"{Back.RED}{log_prefix} FAIL Insufficient Funds: {e}{Style.RESET_ALL}"); send_sms_alert(f"{symbol} MKT FAIL: Insufficient Funds", app_config.sms_config); return None
    except InvalidOrder as e: logger.error(f"{Back.RED}{log_prefix} FAIL Invalid Order: {e}{Style.RESET_ALL}"); return None
    except ExchangeError as e: logger.error(f"{Back.RED}{log_prefix} FAIL Exchange Error: {e}{Style.RESET_ALL}", exc_info=False); return None
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e: logger.error(f"{Back.RED}{log_prefix} FAIL API Error: {type(e).__name__}: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Back.RED}{log_prefix} FAIL Unexpected Error: {e}{Style.RESET_ALL}", exc_info=True); return None

@retry_api_call(max_retries=1)
async def place_limit_order_tif(
    exchange: ccxt.bybit, symbol: str, side: Side, amount: Decimal, price: Decimal, app_config: AppConfig,
    time_in_force: TimeInForce = TimeInForce.GTC, is_reduce_only: bool = False, is_post_only: bool = False,
    client_order_id: Optional[str] = None, position_idx: Optional[PositionIdx]=None, reason: str = "Limit Order"
) -> Optional[Dict]:
    """ Places limit order using AppConfig. """
    func_name="place_limit"; action="Reduce" if is_reduce_only else "Open"; eff_tif=TimeInForce.POST_ONLY if is_post_only else time_in_force; tif_s=eff_tif.value; log_prefix=f"[{func_name} ({symbol}, {side.value}, {amount}@{price}, {action}, TIF:{tif_s}, {reason})]"
    api_conf = app_config.api_config
    if amount <= api_conf.position_qty_epsilon or price <= 0: logger.error(f"{Fore.RED}{log_prefix} Invalid amount/price.{Style.RESET_ALL}"); return None
    if is_post_only and time_in_force != TimeInForce.POST_ONLY: logger.warning(f"{Fore.YELLOW}{log_prefix} PostOnly conflict TIF '{time_in_force.value}'. Using PostOnly.{Style.RESET_ALL}")
    category = market_cache.get_category(symbol); market = market_cache.get_market(symbol); if not category or not market: logger.error(f"{Fore.RED}{log_prefix} Invalid cat/market.{Style.RESET_ALL}"); return None
    fmt_amt = format_amount(exchange, symbol, amount); fmt_px = format_price(exchange, symbol, price); if fmt_amt is None or fmt_px is None: logger.error(f"{Fore.RED}{log_prefix} Fail format amt/px.{Style.RESET_ALL}"); return None
    fmt_amt_fl, fmt_px_fl = float(fmt_amt), float(fmt_px)
    logger.info(f"{Fore.BLUE}{log_prefix} Placing...{Style.RESET_ALL}")
    params: Dict[str, Any] = {'category': category.value, 'reduceOnly': is_reduce_only, 'timeInForce': eff_tif.value}
    params['clientOrderId'] = client_order_id or f"{reason[:5]}_{symbol[:3]}_{side.value}_{int(time.time()*1000)}"[-36:] # Auto/Clean CID
    if position_idx is not None: params['positionIdx'] = position_idx.value if isinstance(position_idx, Enum) else int(position_idx)
    try:
        logger.info(f"{log_prefix} Sending create_limit... CID: {params.get('clientOrderId')}")
        order = await exchange.create_limit_order(symbol, side.value, fmt_amt_fl, fmt_px_fl, params=params)
        oid, status, px = order.get('id'), order.get('status', '?'), safe_decimal_conversion(order.get('price'))
        log_clr = Fore.GREEN if status == 'open' else Fore.YELLOW if status in ['triggered', 'new'] else Fore.RED
        logger.success(f"{log_clr}{log_prefix} OK - ID:{format_order_id(oid)}, Stat:{status}, Px:{format_price(exchange, symbol, px)}, Amt:{format_amount(exchange, symbol, order.get('amount'))}{Style.RESET_ALL}")
        return order
    except OrderImmediatelyFillable as e: if is_post_only: logger.warning(f"{Fore.YELLOW}{log_prefix} PostOnly FAIL (taker): {e}{Style.RESET_ALL}"); return None; else: logger.error(f"{Back.RED}FAIL Unexpected {type(e).__name__}: {e}{Style.RESET_ALL}"); return None
    except InsufficientFunds as e: logger.error(f"{Back.RED}{log_prefix} FAIL Insufficient Funds: {e}{Style.RESET_ALL}"); send_sms_alert(f"{symbol} LMT FAIL: Insuff Funds", app_config.sms_config); return None
    except InvalidOrder as e: logger.error(f"{Back.RED}{log_prefix} FAIL Invalid Order: {e}{Style.RESET_ALL}"); return None
    except ExchangeError as e: logger.error(f"{Back.RED}{log_prefix} FAIL Exchange Error: {e}{Style.RESET_ALL}", exc_info=False); return None
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e: logger.error(f"{Back.RED}{log_prefix} FAIL API Error: {type(e).__name__}: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Back.RED}{log_prefix} FAIL Unexpected Error: {e}{Style.RESET_ALL}", exc_info=True); return None

# --- Batch Orders, Cancel, Fetch Orders, Positions (Omitted for brevity - Adapt similarly) ---
# Ensure they take `app_config: AppConfig` as input

# --- Example adaption for get_current_position_bybit_v5 ---
@retry_api_call()
async def get_current_position_bybit_v5(
    exchange: ccxt.bybit, symbol: str, app_config: AppConfig # Takes AppConfig
) -> Optional[Union[Dict, List[Dict]]]:
    """ Fetches current position using V5 endpoint, uses AppConfig. """
    func_name = "get_current_position_v5"; log_prefix = f"[{func_name} ({symbol})]"
    logger.debug(f"{log_prefix} Fetching V5 position list...")
    category = market_cache.get_category(symbol); if not category or category in [Category.SPOT, Category.OPTION]: logger.debug(f"{log_prefix} Positions N/A for category '{category}'."); return None
    params = {'category': category.value, 'symbol': symbol}
    api_conf = app_config.api_config # Use config
    try:
        response = await exchange.private_get_v5_position_list(params=params)
        ret_code = response.get('retCode'); if ret_code != 0: logger.error(f"{Fore.RED}{log_prefix} API error Code:{ret_code}, Msg:{response.get('retMsg', 'N/A')}{Style.RESET_ALL}"); return None
        pos_list = response.get('result', {}).get('list', [])
        if not pos_list: logger.info(f"{log_prefix} No position entries."); return None
        epsilon = api_conf.position_qty_epsilon
        open_positions_raw = [p for p in pos_list if abs(safe_decimal_conversion(p.get('size', '0'), 0)) > epsilon]
        if not open_positions_raw: logger.info(f"{log_prefix} Position size is zero."); return None
        parsed_positions = []
        for pos_data in open_positions_raw:
            try: # Parse position data (manual enrichment is key for V5)
                if not exchange.markets: await market_cache.load_markets(exchange)
                parsed = exchange.parse_position(pos_data) if hasattr(exchange, 'parse_position') else {}
                parsed['info'] = pos_data; parsed['symbol'] = pos_data.get('symbol'); parsed['contracts'] = safe_decimal_conversion(pos_data.get('size')); parsed['entryPrice'] = safe_decimal_conversion(pos_data.get('avgPrice')); parsed['markPrice'] = safe_decimal_conversion(pos_data.get('markPrice')); parsed['liquidationPrice'] = safe_decimal_conversion(pos_data.get('liqPrice')); parsed['leverage'] = safe_decimal_conversion(pos_data.get('leverage')); parsed['initialMargin'] = safe_decimal_conversion(pos_data.get('positionIM')); parsed['maintenanceMargin'] = safe_decimal_conversion(pos_data.get('positionMM')); parsed['unrealizedPnl'] = safe_decimal_conversion(pos_data.get('unrealisedPnl'))
                v5_side = pos_data.get('side', 'None'); parsed['side'] = 'long' if v5_side == 'Buy' else 'short' if v5_side == 'Sell' else None
                parsed['misc'] = {'positionIdx': pos_data.get('positionIdx'), 'riskId': pos_data.get('riskId'), 'takeProfit': pos_data.get('takeProfit'), 'stopLoss': pos_data.get('stopLoss'), 'tpslMode': pos_data.get('tpslMode'), 'updatedTime': pos_data.get('updatedTime')}
                if 'timestamp' not in parsed and 'updatedTime' in pos_data: parsed['timestamp'] = safe_decimal_conversion(pos_data['updatedTime'])
                if 'datetime' not in parsed and parsed.get('timestamp'): parsed['datetime'] = exchange.iso8601(int(parsed['timestamp']))
                if parsed.get('side') and parsed.get('contracts', 0) > 0: parsed_positions.append(parsed)
            except Exception as parse_err: logger.error(f"{log_prefix} Failed parse pos data: {pos_data}. Err:{parse_err}", exc_info=True)
        if not parsed_positions: logger.info(f"{log_prefix} No valid open positions after parsing."); return None
        if len(parsed_positions) == 1: pos = parsed_positions[0]; idx = pos['misc'].get('positionIdx'); mode = "OneWay" if idx == 0 else "Hedge"; logger.info(f"{Fore.GREEN}{log_prefix} Pos({mode}): Side:{pos.get('side')}, Size:{pos.get('contracts')}, Entry:{pos.get('entryPrice')}{Style.RESET_ALL}"); return pos
        else: logger.info(f"{Fore.GREEN}{log_prefix} {len(parsed_positions)} pos entries(Hedge). List returned.{Style.RESET_ALL}"); [logger.info(f"  - Side:{p.get('side')}, Size:{p.get('contracts')}, Idx:{p['misc'].get('positionIdx')}") for p in parsed_positions]; return parsed_positions
    except AuthenticationError as e: logger.error(f"{Fore.RED}{log_prefix} Auth error: {e}{Style.RESET_ALL}"); return None
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e: logger.warning(f"{Fore.YELLOW}{log_prefix} API error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}"); raise
    except ExchangeError as e: logger.error(f"{Fore.RED}{log_prefix} Exchange error: {e}{Style.RESET_ALL}", exc_info=False); return None
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return None


# --- Native SL/TP (Adapted) ---
async def place_native_stop_loss(
    exchange: ccxt.bybit, symbol: str, side: Side, amount: Decimal, stop_price: Decimal, app_config: AppConfig,
    base_price: Optional[Decimal] = None, trigger_direction: Optional[TriggerDirection] = None,
    is_reduce_only: bool = True, order_type: OrderType = 'Market', limit_price: Optional[Decimal] = None,
    position_idx: PositionIdx = PositionIdx.ONE_WAY if isinstance(PositionIdx, type) else 0, # Handle fallback
    trigger_by: TriggerBy = TriggerBy.MARK if isinstance(TriggerBy, type) else "MarkPrice", # Handle fallback
    stop_loss_type: Literal['StopLoss','TakeProfit'] = 'StopLoss',
    tpsl_mode: StopLossTakeProfitMode = 'Full'
) -> Optional[Dict]:
    """ Places native SL/TP using AppConfig, V5 conditional order. """
    func_name = f"place_native_{stop_loss_type.lower()}"; log_prefix = f"[{func_name} ({symbol}, {side.value}@{stop_price})]"
    logger.info(f"{Fore.CYAN}{log_prefix} Preparing conditional...{Style.RESET_ALL}")
    category = market_cache.get_category(symbol); if not category or category == Category.SPOT: logger.error(f"{Fore.RED}{log_prefix} Invalid cat '{category}'.{Style.RESET_ALL}"); return None
    if trigger_direction is None: trigger_direction = TriggerDirection.FALL if side == Side.SELL else TriggerDirection.RISE; logger.debug(f"{log_prefix} Auto TriggerDir: {trigger_direction.name if isinstance(trigger_direction, Enum) else trigger_direction}")
    if base_price is None:
        ticker = await fetch_ticker_validated(exchange, symbol, app_config)
        trig_by_val = trigger_by.value if isinstance(trigger_by, Enum) else trigger_by
        ref_px_str = ticker.get('info', {}).get(trig_by_val) if ticker else None; base_price = safe_decimal_conversion(ref_px_str)
        if base_price is None and ticker and ticker.get('last'): base_price = safe_decimal_conversion(ticker['last'])
        if base_price is None: logger.error(f"{Fore.RED}{log_prefix} No base_price ({trig_by_val}/Last).{Style.RESET_ALL}"); return None
        logger.debug(f"{log_prefix} Auto base_price: {base_price}")
    fmt_amt = format_amount(exchange, symbol, amount); fmt_sp = format_price(exchange, symbol, stop_price); fmt_bp = format_price(exchange, symbol, base_price); fmt_lp = format_price(exchange, symbol, limit_price) if order_type == 'Limit' else None
    if not all([fmt_amt, fmt_sp, fmt_bp]): logger.error(f"{Fore.RED}{log_prefix} Failed format inputs.{Style.RESET_ALL}"); return None
    if order_type == 'Limit' and not fmt_lp: logger.error(f"{Fore.RED}{log_prefix} Limit price required.{Style.RESET_ALL}"); return None
    pos_idx_val = position_idx.value if isinstance(position_idx, Enum) else int(position_idx)
    trig_by_val = trigger_by.value if isinstance(trigger_by, Enum) else trigger_by
    trig_dir_val = trigger_direction.value if isinstance(trigger_direction, Enum) else int(trigger_direction)

    params = { 'category': category.value, 'symbol': symbol, 'side': side.value.capitalize(), 'orderType': order_type, 'qty': fmt_amt, 'reduceOnly': is_reduce_only, 'positionIdx': pos_idx_val, 'stopOrderType': stop_loss_type, 'triggerPrice': fmt_sp, 'triggerDirection': trig_dir_val, 'triggerBy': trig_by_val, 'tpslMode': tpsl_mode, 'orderFilter': OrderFilter.STOP_ORDER.value if isinstance(OrderFilter, type) else "StopOrder", 'basePrice': fmt_bp }
    if order_type == 'Limit' and fmt_lp: params['price'] = fmt_lp
    logger.info(f"{Fore.CYAN}{log_prefix} Placing conditional via create_order. Params: {params}...{Style.RESET_ALL}")
    try:
        order = await exchange.create_order(symbol, order_type.lower(), side.value, float(fmt_amt), float(fmt_lp) if order_type == 'Limit' and fmt_lp else None, params=params)
        status = order.get('status', '?'); trig_px = order.get('triggerPrice') or order.get('stopPrice'); log_clr = Fore.YELLOW if status in ['untriggered', 'new', 'open'] else Fore.RED
        logger.success(f"{log_clr}{log_prefix} OK. ID:{format_order_id(order.get('id'))}, Stat:{status}, TrigPx:{format_price(exchange, symbol, trig_px)}{Style.RESET_ALL}")
        return order
    except (InvalidOrder, InsufficientFunds, ExchangeError) as e: logger.error(f"{Fore.RED}{log_prefix} Failed: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=False); return None
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return None

# --- WebSocket Functions (Ensure app_config is passed and used) ---
# ... WebSocket functions omitted for brevity, ensure they accept app_config ...

# --- Misc Functions (Adapted) ---
async def validate_market(
    exchange: ccxt.bybit, symbol: str, app_config: AppConfig, # Takes AppConfig
    expected_type: Optional[Literal['swap', 'spot', 'option', 'future']] = None,
    expected_logic: Optional[Literal['linear', 'inverse']] = None,
    check_active: bool = True, require_contract: Optional[bool] = None
) -> Optional[Dict]:
    """ Validates market using AppConfig defaults. """
    func_name = "validate_market"; log_prefix = f"[{func_name} ({symbol})]"; logger.debug(f"{log_prefix} Validating...")
    api_conf = app_config.api_config # Use config
    market = market_cache.get_market(symbol); if not market: logger.error(f"{Fore.RED}{log_prefix} FAIL - Market data not found.{Style.RESET_ALL}"); return None
    is_active = market.get('active', False); if check_active and not is_active: logger.error(f"{Fore.RED}{log_prefix} FAIL - Market inactive.{Style.RESET_ALL}"); return None; elif not is_active: logger.warning(f"{Fore.YELLOW}{log_prefix} Market inactive (check=False).{Style.RESET_ALL}")
    actual_type = market.get('type'); target_type = expected_type if expected_type is not None else api_conf.expected_market_type; if target_type and actual_type != target_type: logger.error(f"{Fore.RED}{log_prefix} FAIL Type mismatch. Exp:{target_type}, Got:{actual_type}.{Style.RESET_ALL}"); return None
    actual_cat = market_cache.get_category(symbol); target_logic = expected_logic if expected_logic is not None else api_conf.expected_market_logic
    if actual_cat and actual_cat in [Category.LINEAR, Category.INVERSE] and target_logic:
         if actual_cat.value != target_logic: logger.error(f"{Fore.RED}{log_prefix} FAIL Logic mismatch. Exp:{target_logic}, Got:{actual_cat.value}.{Style.RESET_ALL}"); return None
    elif target_logic and actual_cat and actual_cat not in [Category.LINEAR, Category.INVERSE]: logger.debug(f"{log_prefix} Logic check skipped: Cat '{actual_cat}' not deriv.")
    is_contract = market.get('contract', False); if require_contract is True and not is_contract: logger.error(f"{Fore.RED}{log_prefix} FAIL Contract required.{Style.RESET_ALL}"); return None; if require_contract is False and is_contract: logger.error(f"{Fore.RED}{log_prefix} FAIL Contract not allowed.{Style.RESET_ALL}"); return None
    logger.info(f"{Fore.GREEN}{log_prefix} PASSED. Type:{actual_type}, Cat:{actual_cat}, Active:{is_active}{Style.RESET_ALL}")
    return market

async def set_position_mode_bybit_v5(
    exchange: ccxt.bybit, app_config: AppConfig, # Takes AppConfig
    mode: Literal['one-way', 'hedge'],
    category: Optional[Category] = None, symbol: Optional[str] = None
) -> bool:
    """ Sets position mode using AppConfig. """
    func_name = "set_position_mode"; log_prefix = f"[{func_name}]"; logger.info(f"{Fore.CYAN}{log_prefix} Setting mode to '{mode}'...{Style.RESET_ALL}")
    target_cat_enum: Optional[Category] = None
    if category: target_cat_enum = category
    elif symbol: target_cat_enum = market_cache.get_category(symbol)
    elif app_config.api_config.symbol: target_cat_enum = market_cache.get_category(app_config.api_config.symbol); logger.debug(f"{log_prefix} Using cat from default sym {app_config.api_config.symbol}.")
    if not target_cat_enum or target_cat_enum not in [Category.LINEAR, Category.INVERSE]: logger.error(f"{Fore.RED}{log_prefix} Invalid cat. Requires LINEAR/INVERSE. Got:{target_cat_enum}{Style.RESET_ALL}"); return False
    target_cat_str = target_cat_enum.value; bybit_mode = 3 if mode == 'hedge' else 0; params = {'category': target_cat_str, 'mode': bybit_mode}
    if symbol: params['symbol'] = symbol # V5 allows setting per symbol
    logger.info(f"{Fore.CYAN}{log_prefix} Requesting mode '{mode}' (Code:{bybit_mode}) for cat '{target_cat_str}'{(' sym '+symbol) if symbol else ''}...{Style.RESET_ALL}")
    try:
        resp = await exchange.private_post_v5_position_switch_mode(params=params); logger.debug(f"{log_prefix} Raw resp: {resp}")
        code = resp.get('retCode'); msg = resp.get('retMsg', '').lower()
        if code == 0: logger.success(f"{Fore.GREEN}{log_prefix} Mode set OK to '{mode}' for {target_cat_str}{(' ('+symbol+')') if symbol else ''}.{Style.RESET_ALL}"); return True
        elif code in [110021, 34036] or "not modified" in msg: logger.info(f"{Fore.CYAN}{log_prefix} Mode already '{mode}' for {target_cat_str}{(' ('+symbol+')') if symbol else ''}.{Style.RESET_ALL}"); return True
        elif code == 110020 or "have position" in msg or "active order" in msg: logger.error(f"{Fore.RED}{log_prefix} Cannot switch: Active positions/orders exist. Msg:{resp.get('retMsg')}{Style.RESET_ALL}"); return False
        else: logger.error(f"{Fore.RED}{log_prefix} Failed set mode. Code:{code}, Msg:{resp.get('retMsg')}{Style.RESET_ALL}"); return False
    except (NetworkError, AuthenticationError, ExchangeError) as e: logger.error(f"{Fore.RED}{log_prefix} API Error: {type(e).__name__} - {e}{Style.RESET_ALL}"); return False
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return False
EOF

# --- Create ehlers_volumetric_strategy.py ---
echo "# Creating ehlers_volumetric_strategy.py"
cat << 'EOF' > ehlers_volumetric_strategy.py
# ehlers_volumetric_strategy.py
# -*- coding: utf-8 -*-
"""
Enhanced Ehlers Volumetric Strategy Implementation.

Utilizes Pydantic models from config_models.py for configuration and
relies on bybit_helpers and bybit_utils for exchange interaction and utilities.
"""

import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import pandas as pd

# --- Import Custom Modules & Helpers ---
try:
    import bybit_helpers as bybit
    from bybit_helpers import (
        Category, Side, TimeInForce, TriggerBy, TriggerDirection, OrderFilter, PositionIdx
    )
    import indicators as ind
    from bybit_utils import (
        format_amount, format_order_id, format_price, safe_decimal_conversion, send_sms_alert
    )
    from config_models import AppConfig, StrategyConfig, APIConfig, SMSConfig # Use Pydantic models
except ImportError as e:
    print(f"\033[91m\033[1mStrategy FATAL: Failed to import modules: {e}\033[0m")
    print("\033[93mEnsure config_models.py, bybit_helpers.py, indicators.py, bybit_utils.py are present.\033[0m")
    exit(1)

# --- Colorama Setup ---
try:
    from colorama import Back, Fore, Style
    C_INFO, C_SUCCESS, C_WARN, C_ERROR = Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.RED + Style.BRIGHT
    C_CRIT, C_DEBUG, C_BOLD, C_RESET = Back.RED + Fore.WHITE + Style.BRIGHT, Fore.MAGENTA, Style.BRIGHT, Style.RESET_ALL
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor(); C_INFO=C_SUCCESS=C_WARN=C_ERROR=C_CRIT=C_DEBUG=C_BOLD=C_RESET=""


class EhlersVolumetricStrategyEnhanced:
    """ Enhanced Ehlers Volumetric Trend (EVT) strategy using Pydantic config. """

    def __init__(self, app_config: AppConfig, logger: logging.Logger):
        """ Initializes the strategy instance with validated configuration. """
        self.app_config = app_config; self.api_config = app_config.api_config
        self.strategy_config = app_config.strategy_config; self.sms_config = app_config.sms_config
        self.logger = logger
        self.logger.info(f"{C_INFO}{C_BOLD}Conjuring {self.strategy_config.name} Strategy...{C_RESET}")

        # --- Essential Strategy Logic Validation ---
        if not self.strategy_config.EVT_ENABLED or not self.strategy_config.analysis_flags.use_evt:
            msg = "EVT strategy requires 'EVT_ENABLED' and 'analysis_flags.use_evt' True."; self.logger.critical(f"{C_CRIT}FATAL: {msg}{C_RESET}"); raise ValueError(msg)
        if self.strategy_config.analysis_flags.use_atr and not self.strategy_config.indicator_settings.atr_period > 0:
             msg = "ATR SL requires 'analysis_flags.use_atr' True and positive 'indicator_settings.atr_period'."; self.logger.critical(f"{C_CRIT}FATAL: {msg}{C_RESET}"); raise ValueError(msg)

        # --- Core Parameters ---
        self.symbol = self.api_config.symbol; self.timeframe = self.strategy_config.timeframe
        self.leverage = self.strategy_config.leverage; self.position_idx = self.strategy_config.position_idx

        # --- Exchange/Market Info ---
        self.exchange: Optional[bybit.ccxt.bybit] = None; self.market_info: Optional[Dict[str, Any]] = None
        self.min_qty = Decimal("1E-8"); self.qty_step = Decimal("1E-8"); self.price_tick = Decimal("1E-8")

        # --- State Variables ---
        self.current_position: Optional[Dict[str, Any]] = None; self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.last_known_price = Decimal("0"); self.available_balance = Decimal("0")
        self.is_running = False; self.stop_loss_order_id: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # --- Indicator Config & Columns ---
        self.evt_length = self.strategy_config.indicator_settings.evt_length
        self.atr_period = self.strategy_config.indicator_settings.atr_period
        self.evt_trend_col = f"evt_trend_{self.evt_length}"; self.evt_buy_col = f"evt_buy_{self.evt_length}"; self.evt_sell_col = f"evt_sell_{self.evt_length}"
        self.atr_col = f"ATRr_{self.atr_period}" # pandas_ta default naming

        self.required_indicators: List[str] = []
        if self.strategy_config.analysis_flags.use_evt: self.required_indicators.extend([self.evt_trend_col, self.evt_buy_col, self.evt_sell_col])
        if self.strategy_config.analysis_flags.use_atr: self.required_indicators.append(self.atr_col)

        pos_idx_name = self.position_idx.name if isinstance(self.position_idx, Enum) else str(self.position_idx)
        pos_idx_val = self.position_idx.value if isinstance(self.position_idx, Enum) else self.position_idx
        self.logger.info(f"{C_INFO}Strategy '{C_BOLD}{self.strategy_config.name}{C_RESET}{C_INFO}' initialized for {C_BOLD}{self.symbol}{C_RESET}{C_INFO} ({self.timeframe}).")
        self.logger.debug(f"{C_DEBUG}Indicators: {self.required_indicators}, PositionMode: {pos_idx_name}({pos_idx_val}), Risk:{self.strategy_config.risk_per_trade:.2%}, SL Mult:{self.strategy_config.stop_loss_atr_multiplier}")
        self.logger.info(f"{C_SUCCESS}Config validated, strategy ready.")


    async def _initialize(self) -> bool:
        """ Initializes exchange connection, market data, leverage, and state. """
        self.logger.info(f"{C_INFO}{C_BOLD}--- Init Ritual ---{C_RESET}")
        try: self.loop = asyncio.get_running_loop()
        except RuntimeError: self.logger.critical(f"{C_CRIT}No event loop.{C_RESET}"); return False
        try:
            self.logger.info(f"{C_DEBUG}Connecting ({'Testnet' if self.api_config.testnet_mode else 'Live'})...")
            self.exchange = await bybit.initialize_bybit(app_config=self.app_config)
            if not self.exchange: self.logger.critical(f"{C_CRIT}Failed connection.{C_RESET}"); return False; self.logger.info(f"{C_SUCCESS}Connected.")
            self.logger.info(f"{C_DEBUG}Gathering market lore {self.symbol}...")
            self.market_info = bybit.market_cache.get_market(self.symbol)
            if not self.market_info: self.logger.critical(f"{C_CRIT}No market lore {self.symbol}.{C_RESET}"); await self._safe_exchange_close(); return False
            self._extract_market_details()
            if self.leverage > 0:
                self.logger.info(f"{C_INFO}Attuning leverage {self.symbol} -> {self.leverage}x...")
                lev_set = await bybit.set_leverage(self.exchange, self.symbol, self.leverage, app_config=self.app_config)
                if not lev_set: self.logger.warning(f"{C_WARN}Failed leverage tune.{C_RESET}")
                else: self.logger.info(f"{C_SUCCESS}Leverage OK: {self.leverage}x.")
            self.logger.info(f"{C_INFO}Scrying initial state...")
            await self._update_state()
            pos_s, pos_q = self.get_position_side(), self.get_position_size()
            pos_q_s = format_amount(self.exchange, self.symbol, pos_q) if self.exchange else str(pos_q)
            self.logger.info(f"{C_INFO}Initial State: Pos={pos_s or 'None'} Qty={pos_q_s}, Orders={len(self.open_orders)}, Bal={self.available_balance:.4f}, Px={self.last_known_price:.4f}")
            self.logger.info(f"{C_INFO}Initial cleanup: banishing orders {self.symbol}...")
            cat = bybit.market_cache.get_category(self.symbol)
            if cat and self.exchange:
                 cancelled = await bybit.cancel_all_orders(self.exchange, app_config=self.app_config, symbol=self.symbol, category=cat, reason="Init Cleanup")
                 if cancelled is not None: self.logger.info(f"{C_SUCCESS}Cleanup banished {cancelled} orders."); await asyncio.sleep(1); await self._update_state()
                 else: self.logger.warning(f"{C_WARN}Cleanup partial.{C_RESET}")
            else: self.logger.error(f"{C_ERROR}Cannot cleanup orders.{C_RESET}")
            self.logger.info(f"{C_SUCCESS}{C_BOLD}--- Init Ritual Complete ---{C_RESET}"); return True
        except Exception as e: self.logger.critical(f"{C_CRIT}Init critical failure: {e}{C_RESET}", exc_info=True); await self._safe_exchange_close(); return False

    def _extract_market_details(self):
        """ Extracts precision and limits from market_info. """
        if not self.market_info or not self.exchange:
            self.logger.error(f"{C_ERROR}Cannot extract market details.{C_RESET}"); self.min_qty,self.qty_step,self.price_tick=Decimal("1E-8"),Decimal("1E-8"),Decimal("1E-8"); return
        self.logger.debug(f"{C_DEBUG}Extracting market runes...")
        try:
            limits = self.market_info.get("limits", {}); amount_lim = limits.get("amount", {})
            prec = self.market_info.get("precision", {})
            min_q_s = amount_lim.get("min"); qty_s_s = prec.get("amount"); px_t_s = prec.get("price")
            tiny = Decimal("1E-8")
            self.min_qty = max(tiny, safe_decimal_conversion(min_q_s, tiny))
            self.qty_step = max(tiny, safe_decimal_conversion(qty_s_s, tiny))
            self.price_tick = max(tiny, safe_decimal_conversion(px_t_s, tiny))
            if self.qty_step <= 0 or self.price_tick <= 0: raise ValueError(f"Invalid step/tick (Qty:{self.qty_step}, Px:{self.price_tick})")
            min_q_f, qty_s_f, px_t_f = (format_amount(self.exchange, self.symbol, self.min_qty) or "N/A",
                                        format_amount(self.exchange, self.symbol, self.qty_step) or "N/A",
                                        format_price(self.exchange, self.symbol, self.price_tick) or "N/A")
            self.logger.info(f"{C_SUCCESS}Market Runes: MinQty={C_BOLD}{min_q_f}{C_RESET}, QtyStep={C_BOLD}{qty_s_f}{C_RESET}, PriceTick={C_BOLD}{px_t_f}{C_RESET}")
        except Exception as e:
            self.logger.error(f"{C_ERROR}Failed parse market runes {self.symbol}: {e}{C_RESET}", exc_info=True)
            self.min_qty,self.qty_step,self.price_tick=Decimal("1E-8"),Decimal("1E-8"),Decimal("1E-8"); self.logger.warning(f"{C_WARN}Using default tiny runes.{C_RESET}")

    async def _update_state(self):
        """ Gathers current state concurrently. """
        self.logger.debug(f"{C_DEBUG}Updating state...")
        if not self.exchange or not self.loop: self.logger.error(f"{C_ERROR}Cannot update state: No exchange/loop.{C_RESET}"); return
        try:
            tasks = { # Pass AppConfig to helpers
                "position": bybit.get_current_position_bybit_v5(self.exchange, self.symbol, self.app_config),
                "orders": self._fetch_all_open_orders(),
                "balance": bybit.fetch_usdt_balance(self.exchange, self.app_config),
                "ticker": bybit.fetch_ticker_validated(self.exchange, self.symbol, self.app_config),
            }
            res = await asyncio.gather(*tasks.values(), return_exceptions=True); results = dict(zip(tasks.keys(), res))

            # Position
            pos_d = results.get("position")
            if isinstance(pos_d, Exception): self.logger.error(f"{C_ERROR}State Fail (Pos): {pos_d}{C_RESET}"); self.current_position = None
            elif isinstance(pos_d, list): self.current_position = next((p for p in pos_d if p.get('misc',{}).get('positionIdx') == self.position_idx.value), None) if isinstance(self.position_idx, Enum) else None # Hedge
            elif isinstance(pos_d, dict): self.current_position = pos_d # One-way
            else: self.current_position = None
            if self.current_position and self.get_position_size() <= self.api_config.position_qty_epsilon: self.current_position = None # Treat zero size as no position

            # Orders
            orders_l = results.get("orders")
            if isinstance(orders_l, Exception): self.logger.error(f"{C_ERROR}State Fail (Orders): {orders_l}{C_RESET}")
            elif isinstance(orders_l, list):
                self.open_orders = {o['id']: o for o in orders_l if o.get('id')}
                if self.stop_loss_order_id and self.stop_loss_order_id not in self.open_orders: self.logger.info(f"{C_INFO}Tracked SL {format_order_id(self.stop_loss_order_id)} gone.{C_RESET}"); self.stop_loss_order_id = None
            else: self.logger.error(f"{C_ERROR}State Fail (Orders): Bad type {type(orders_l)}{C_RESET}")

            # Balance
            bal_t = results.get("balance")
            if isinstance(bal_t, Exception): self.logger.error(f"{C_ERROR}State Fail (Bal): {bal_t}{C_RESET}")
            elif isinstance(bal_t, tuple) and len(bal_t) == 2 and bal_t[1] is not None: self.available_balance = bal_t[1]
            else: self.logger.error(f"{C_ERROR}State Fail (Bal): Bad format {bal_t}{C_RESET}")

            # Ticker
            ticker = results.get("ticker")
            if isinstance(ticker, Exception): self.logger.warning(f"{C_WARN}State Warn (Ticker): {ticker}{C_RESET}")
            elif isinstance(ticker, dict) and ticker.get('last'):
                new_px = safe_decimal_conversion(ticker['last']);
                if new_px and new_px > 0: self.last_known_price = new_px
                else: self.logger.warning(f"{C_WARN}State Warn (Ticker): Invalid last '{ticker['last']}'. Keep {self.last_known_price:.4f}.{C_RESET}")
            else: self.logger.warning(f"{C_WARN}State Warn (Ticker): No 'last'. Keep {self.last_known_price:.4f}.{C_RESET}")

            pos_s, pos_q = self.get_position_side(), self.get_position_size()
            pos_q_s = format_amount(self.exchange, self.symbol, pos_q) if self.exchange else str(pos_q)
            self.logger.debug(f"{C_DEBUG}State: Pos={pos_s or 'None'} Qty={pos_q_s}, Orders={len(self.open_orders)}, Bal={self.available_balance:.2f}, Px={self.last_known_price:.4f}")
        except Exception as e: self.logger.error(f"{C_ERROR}Unexpected state update error: {e}{C_RESET}", exc_info=True)

    def get_position_side(self) -> Optional[str]:
        """ Returns 'LONG', 'SHORT', or None based on current_position state. """
        if not self.current_position: return None
        side_ccxt = self.current_position.get('side')
        return self.api_config.pos_long if side_ccxt == 'long' else self.api_config.pos_short if side_ccxt == 'short' else None

    def get_position_size(self) -> Decimal:
         """ Returns the size of the current position as Decimal, 0 if no position. """
         if not self.current_position: return Decimal(0)
         return safe_decimal_conversion(self.current_position.get('contracts'), Decimal(0))

    async def _fetch_all_open_orders(self) -> List[Dict[str, Any]]:
        """ Fetches all relevant types of open orders for the symbol. """
        if not self.exchange: self.logger.error(f"{C_ERROR}No exchange for orders.{C_RESET}"); return []
        cat = bybit.market_cache.get_category(self.symbol); if not cat: self.logger.error(f"{C_ERROR}No category for orders {self.symbol}.{C_RESET}"); return []
        self.logger.debug(f"{C_DEBUG}Fetching open orders (Limit, Stop)..."); tasks = []
        filters = [ OrderFilter.ORDER, OrderFilter.STOP_ORDER ]
        for f in filters: tasks.append(bybit.fetch_open_orders_filtered(exchange=self.exchange, symbol=self.symbol, category=cat, order_filter=f, app_config=self.app_config))
        results = await asyncio.gather(*tasks, return_exceptions=True); all_orders = []
        for i, res in enumerate(results):
            if isinstance(res, Exception): self.logger.warning(f"{C_WARN}Failed fetch orders filter '{filters[i].value}': {res}{C_RESET}")
            elif isinstance(res, list): self.logger.debug(f"{C_DEBUG}Fetched {len(res)} orders filter '{filters[i].value}'."); all_orders.extend(res)
        unique = {o['id']: o for o in all_orders if o.get('id')}; self.logger.debug(f"{C_DEBUG}Total unique open orders: {len(unique)}")
        return list(unique.values())

    async def _fetch_and_calculate_indicators(self) -> Optional[pd.DataFrame]:
        """ Fetches OHLCV and calculates indicators. """
        self.logger.debug(f"{C_DEBUG}Divining indicators ({self.timeframe})...")
        if not self.exchange or not self.loop: self.logger.error(f"{C_ERROR}No exchange/loop for indicators.{C_RESET}"); return None
        try:
            min_p = self.strategy_config.indicator_settings.min_data_periods; buffer = 50; limit = min_p + buffer
            self.logger.debug(f"{C_DEBUG}Requesting {limit} candles (min:{min_p}) {self.symbol} {self.timeframe}.")
            ohlcv = await bybit.fetch_ohlcv_paginated(exchange=self.exchange, symbol=self.symbol, timeframe=self.timeframe, limit=limit, app_config=self.app_config)
            if ohlcv is None: self.logger.warning(f"{C_WARN}OHLCV None.{C_RESET}"); return None
            df_ohlcv: Optional[pd.DataFrame] = None
            if isinstance(ohlcv, list):
                if not ohlcv: self.logger.warning(f"{C_WARN}OHLCV empty list.{C_RESET}"); return None
                try:
                    df_ohlcv = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                    df_ohlcv['datetime'] = pd.to_datetime(df_ohlcv['timestamp'], unit='ms', utc=True); df_ohlcv.set_index('datetime', inplace=True)
                    for col in ['open','high','low','close','volume']: df_ohlcv[col] = pd.to_numeric(df_ohlcv[col], errors='coerce')
                    df_ohlcv.dropna(subset=['open','high','low','close','volume'], inplace=True)
                except Exception as e: self.logger.error(f"{C_ERROR}Failed convert OHLCV list: {e}{C_RESET}", exc_info=True); return None
            elif isinstance(ohlcv, pd.DataFrame): df_ohlcv = ohlcv; if df_ohlcv.empty: self.logger.warning(f"{C_WARN}OHLCV empty DF.{C_RESET}"); return None
            else: self.logger.error(f"{C_ERROR}Unexpected OHLCV type: {type(ohlcv)}.{C_RESET}"); return None
            if len(df_ohlcv) < min_p: self.logger.warning(f"{C_WARN}Insufficient OHLCV ({len(df_ohlcv)} < {min_p}).{C_RESET}"); return None
            self.logger.debug(f"{C_DEBUG}Calculating indicators...")
            df_ind = await self.loop.run_in_executor(None, ind.calculate_all_indicators, df_ohlcv.copy(), self.app_config)
            if df_ind is None or not isinstance(df_ind, pd.DataFrame) or df_ind.empty: self.logger.error(f"{C_ERROR}Indicator calc failed.{C_RESET}"); return None
            missing = [c for c in self.required_indicators if c not in df_ind.columns]
            if missing: self.logger.error(f"{C_ERROR}Required indicators missing: {missing}.{C_RESET}"); return None
            if df_ind[self.required_indicators].iloc[-1].isna().any(): nans = df_ind[self.required_indicators].iloc[-1].isna(); self.logger.warning(f"{C_WARN}NaNs in latest indicators: {nans[nans].index.tolist()}.{C_RESET}")
            self.logger.debug(f"{C_DEBUG}Indicators OK. Shape: {df_ind.shape}"); return df_ind
        except Exception as e: self.logger.error(f"{C_ERROR}Error indicator divination: {e}{C_RESET}", exc_info=True); return None

    def _check_signals(self, df: pd.DataFrame) -> Tuple[Optional[Side], bool]:
        """ Interprets signals from the latest indicator data (EVT). """
        entry_side: Optional[Side] = None; should_exit: bool = False
        if df is None or df.empty or len(df) < 2: self.logger.debug(f"{C_DEBUG}Signal Check: Invalid DF."); return None, False
        if not self.strategy_config.analysis_flags.use_evt: self.logger.debug(f"{C_DEBUG}Signal Check: EVT disabled."); return None, False
        try: latest, prev = df.iloc[-1], df.iloc[-2]
        except IndexError: self.logger.warning(f"{C_WARN}Signal Check: IndexError."); return None, False
        trend_l, trend_p = latest.get(self.evt_trend_col), prev.get(self.evt_trend_col)
        buy_l, sell_l = latest.get(self.evt_buy_col), latest.get(self.evt_sell_col)
        omens = {self.evt_trend_col: trend_l, f"{self.evt_trend_col}_p": trend_p, self.evt_buy_col: buy_l, self.evt_sell_col: sell_l}
        if any(pd.isna(v) for v in omens.values()): self.logger.debug(f"{C_DEBUG}Signal Check: NaNs in EVT omens: { {k for k,v in omens.items() if pd.isna(v)} }."); return None, False
        try: cur_t, prev_t = int(trend_l), int(trend_p); is_buy, is_sell = bool(buy_l), bool(sell_l)
        except (ValueError, TypeError) as e: self.logger.error(f"{C_ERROR}Signal Check: Cannot interpret EVT types: {e}{C_RESET}"); return None, False
        self.logger.debug(f"{C_DEBUG}Signal Omens: Trend={cur_t}(P:{prev_t}), Buy={is_buy}, Sell={is_sell}")
        pos_side = self.get_position_side(); is_flat = pos_side is None; is_long = pos_side == self.api_config.pos_long; is_short = pos_side == self.api_config.pos_short
        if is_flat: # Entry Logic
            if is_buy and is_sell: self.logger.warning(f"{C_WARN}Signal Check: Ambiguous (Buy&Sell). No entry.{C_RESET}")
            elif is_buy: entry_side = Side.BUY; self.logger.info(f"{C_SUCCESS}{C_BOLD}ENTRY OMEN: BUY (EVT Buy){C_RESET}")
            elif is_sell: entry_side = Side.SELL; self.logger.info(f"{C_ERROR}{C_BOLD}ENTRY OMEN: SELL (EVT Sell){C_RESET}")
        elif not is_flat: # Exit Logic
            if is_long and cur_t == -1 and prev_t != -1: should_exit = True; self.logger.warning(f"{C_WARN}{C_BOLD}EXIT OMEN: Close LONG (EVT flip bearish {prev_t}->{cur_t}){C_RESET}")
            elif is_short and cur_t == 1 and prev_t != 1: should_exit = True; self.logger.warning(f"{C_WARN}{C_BOLD}EXIT OMEN: Close SHORT (EVT flip bullish {prev_t}->{cur_t}){C_RESET}")
        if entry_side: self.logger.debug(f"{C_DEBUG}Signal Result: Entry={entry_side.value}")
        elif should_exit: self.logger.debug(f"{C_DEBUG}Signal Result: Exit=True")
        else: self.logger.debug(f"{C_DEBUG}Signal Result: No Action")
        return entry_side, should_exit

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """ Calculates position size based on risk, SL, balance, constraints. """
        self.logger.debug(f"{C_DEBUG}Calculating position size..."); risk_pct = self.strategy_config.risk_per_trade
        if not self.exchange: self.logger.error(f"{C_ERROR}Size Calc: No exchange.{C_RESET}"); return None
        if entry_price <= 0 or stop_loss_price <= 0: self.logger.error(f"{C_ERROR}Size Calc: Invalid prices (E:{entry_price}, SL:{stop_loss_price}).{C_RESET}"); return None
        if self.available_balance <= 0: self.logger.warning(f"{C_WARN}Size Calc: Bal <= 0 ({self.available_balance}).{C_RESET}"); return None
        if self.qty_step <= 0 or self.price_tick <= 0: self.logger.error(f"{C_ERROR}Size Calc: Invalid market details.{C_RESET}"); return None
        risk_usd = self.available_balance * risk_pct; price_diff = abs(entry_price - stop_loss_price)
        if price_diff < self.price_tick: self.logger.error(f"{C_ERROR}Size Calc: SL dist < tick ({self.price_tick}). SL too tight.{C_RESET}"); return None
        if price_diff == 0: self.logger.error(f"{C_ERROR}Size Calc: Price diff is zero.{C_RESET}"); return None
        size_base = risk_usd / price_diff; self.logger.debug(f"{C_DEBUG}Size Calc - Risk:{risk_usd:.4f}, PxDiff:{price_diff:.4f}, IdealBase:{size_base:.8f}")
        size_adj = (size_base // self.qty_step) * self.qty_step
        if size_adj <= 0: self.logger.warning(f"{C_WARN}Size Calc: Adjusted size zero (Ideal:{size_base:.8f}, Step:{self.qty_step}).{C_RESET}"); return None
        size_final = size_adj
        if size_adj < self.min_qty:
            min_q_f = format_amount(self.exchange, self.symbol, self.min_qty) or "N/A"; adj_f = format_amount(self.exchange, self.symbol, size_adj) or "N/A"
            self.logger.warning(f"{C_WARN}Size Calc: Adjusted ({adj_f}) < min ({min_q_f}).{C_RESET}")
            min_risk = self.min_qty * price_diff; max_ok_risk = risk_usd * Decimal("1.10") # 10% buffer
            if min_risk <= max_ok_risk: self.logger.info(f"{C_INFO}Size Calc: Using min size ({min_q_f}). Risk ~{min_risk:.4f}."); size_final = self.min_qty
            else: self.logger.error(f"{C_ERROR}Size Calc: Min size risk ({min_risk:.4f}) > budget ({risk_usd:.4f}). Cannot trade.{C_RESET}"); return None
        leverage = Decimal(str(self.leverage)) if self.leverage > 0 else 1
        cost_est = (size_final * entry_price) / leverage; margin_buf = Decimal('0.95') # 95% balance usage max
        max_cost = self.available_balance * margin_buf
        if cost_est > max_cost: final_f = format_amount(self.exchange, self.symbol, size_final) or "N/A"; self.logger.error(f"{C_ERROR}Size Calc: Est cost {cost_est:.4f} > {margin_buf*100:.0f}% avail bal ({self.available_balance:.4f}). Size={final_f}. Lev={self.leverage}x.{C_RESET}"); return None
        final_f = format_amount(self.exchange, self.symbol, size_final) or "N/A"
        self.logger.info(f"{C_SUCCESS}Calculated Pos Size: {C_BOLD}{final_f}{C_RESET} (Risk:{risk_usd:.4f}, EstCost:{cost_est:.4f})")
        return size_final

    def _calculate_stop_loss_price(self, df: pd.DataFrame, side: Side, entry_price: Decimal) -> Optional[Decimal]:
        """ Calculates SL price based on ATR, multiplier, entry, precision. """
        self.logger.debug(f"{C_DEBUG}Calculating SL price for {side.value}..."); atr_mult = self.strategy_config.stop_loss_atr_multiplier
        if not self.strategy_config.analysis_flags.use_atr: self.logger.debug(f"{C_DEBUG}SL Calc: ATR disabled."); return None
        if df is None or df.empty: self.logger.warning(f"{C_WARN}SL Calc: Indicator DF missing.{C_RESET}"); return None
        if self.atr_col not in df.columns: self.logger.error(f"{C_ERROR}SL Calc: ATR col '{self.atr_col}' missing.{C_RESET}"); return None
        if not self.exchange: self.logger.error(f"{C_ERROR}SL Calc: No exchange.{C_RESET}"); return None
        if entry_price <= 0: self.logger.error(f"{C_ERROR}SL Calc: Invalid entry px ({entry_price}).{C_RESET}"); return None
        if self.price_tick <= 0: self.logger.error(f"{C_ERROR}SL Calc: Invalid price tick ({self.price_tick}).{C_RESET}"); return None
        try:
            atr_raw = df.iloc[-1].get(self.atr_col);
            if pd.isna(atr_raw): self.logger.warning(f"{C_WARN}SL Calc: Latest ATR NaN.{C_RESET}"); return None
            atr = safe_decimal_conversion(atr_raw);
            if atr is None or atr <= 0: self.logger.warning(f"{C_WARN}SL Calc: Invalid ATR ({atr_raw}).{C_RESET}"); return None
            offset = atr * atr_mult; sl_raw = (entry_price - offset) if side == Side.BUY else (entry_price + offset)
            self.logger.debug(f"{C_DEBUG}SL Calc - Entry:{entry_price:.8f}, ATR:{atr:.8f}, Mult:{atr_mult}, Offset:{offset:.8f}, RawSL:{sl_raw:.8f}")
            sl_fmt_s = format_price(self.exchange, self.symbol, sl_raw);
            if sl_fmt_s is None or sl_fmt_s == "Error": self.logger.error(f"{C_ERROR}SL Calc: Failed format raw SL {sl_raw}.{C_RESET}"); return None
            sl_fmt = safe_decimal_conversion(sl_fmt_s);
            if sl_fmt is None or sl_fmt <= 0: self.logger.error(f"{C_ERROR}SL Calc: Formatted SL invalid/neg: {sl_fmt_s}.{C_RESET}"); return None
            sl_adj = sl_fmt # Sanity check/adjust
            if side == Side.BUY and sl_adj >= entry_price: self.logger.warning(f"{C_WARN}SL Calc: Buy SL ({sl_fmt_s}) >= Entry. Adjust down.{C_RESET}"); sl_adj -= self.price_tick
            elif side == Side.SELL and sl_adj <= entry_price: self.logger.warning(f"{C_WARN}SL Calc: Sell SL ({sl_fmt_s}) <= Entry. Adjust up.{C_RESET}"); sl_adj += self.price_tick
            if sl_adj <= 0: self.logger.error(f"{C_ERROR}SL Calc: Adjusted SL non-positive ({sl_adj:.8f}).{C_RESET}"); return None
            sl_final_s = format_price(self.exchange, self.symbol, sl_adj); # Reformat after adjust
            if sl_final_s is None or sl_final_s == "Error": self.logger.error(f"{C_ERROR}SL Calc: Failed format final SL {sl_adj}.{C_RESET}"); return None
            sl_final = safe_decimal_conversion(sl_final_s)
            if sl_final and sl_final > 0:
                 atr_f = format_price(self.exchange, self.symbol, atr) or "N/A"; entry_f = format_price(self.exchange, self.symbol, entry_price) or "N/A"
                 self.logger.info(f"{C_SUCCESS}Calculated SL: {C_BOLD}{sl_final_s}{C_RESET} (Side:{side.value}, Entry:{entry_f}, ATR:{atr_f}, Mult:{atr_mult})")
                 return sl_final
            else: self.logger.error(f"{C_ERROR}SL Calc: Final SL invalid: {sl_final_s}.{C_RESET}"); return None
        except Exception as e: self.logger.error(f"{C_ERROR}SL Calc: Unexpected error: {e}{C_RESET}", exc_info=True); return None

    async def _place_stop_loss(self, entry_side: Side, qty: Decimal, sl_price: Decimal):
        """ Places native conditional stop loss order. """
        if not self.exchange: self.logger.error(f"{C_ERROR}Place SL: No exchange.{C_RESET}"); return
        if sl_price is None or sl_price <= 0: self.logger.error(f"{C_ERROR}Place SL: Invalid SL price {sl_price}.{C_RESET}"); return
        if qty <= 0: self.logger.error(f"{C_ERROR}Place SL: Invalid qty {qty}.{C_RESET}"); return
        sl_side = Side.SELL if entry_side == Side.BUY else Side.BUY; trig_dir = TriggerDirection.FALLING if sl_side == Side.SELL else TriggerDirection.RISING
        qty_f = format_amount(self.exchange, self.symbol, qty) or "N/A"; sl_f = format_price(self.exchange, self.symbol, sl_price) or "N/A"
        self.logger.info(f"{C_INFO}Placing {sl_side.value.upper()} native SL for {C_BOLD}{qty_f}{C_RESET} @ trigger {C_BOLD}{sl_f}{C_RESET} ({trig_dir.name if isinstance(trig_dir, Enum) else trig_dir})...")
        base_px = self.last_known_price
        if base_px <= 0: self.logger.critical(f"{C_CRIT}Place SL: Invalid base price ({base_px}). Pos UNPROTECTED.{C_RESET}"); await self._trigger_alert(f"URGENT: Invalid basePrice ({base_px}) for SL! Pos UNPROTECTED.", critical=True); await self._emergency_close(f"Invalid basePrice ({base_px}) for SL"); return
        try: # Use dedicated helper, pass AppConfig
            sl_order = await bybit.place_native_stop_loss( exchange=self.exchange, symbol=self.symbol, side=sl_side, amount=qty, stop_price=sl_price, app_config=self.app_config, base_price=base_px, trigger_direction=trig_dir, is_reduce_only=True, order_type='Market', position_idx=self.position_idx, trigger_by=TriggerBy.LAST_PRICE) # Or MARK
            if sl_order and sl_order.get('id'): self.stop_loss_order_id = sl_order['id']; sl_id_s = format_order_id(self.stop_loss_order_id); self.logger.success(f"{C_SUCCESS}{C_BOLD}Native SL placed! ID: {sl_id_s}{C_RESET}")
            else: self.logger.critical(f"{C_CRIT}Place SL: FAILED place native SL! Helper ret: {sl_order}. Pos UNPROTECTED.{C_RESET}"); self.stop_loss_order_id = None; await self._trigger_alert(f"URGENT: Failed SL placement after {entry_side.value} entry! Pos UNPROTECTED.", critical=True); await self._emergency_close("Failed SL Placement")
        except Exception as e: self.logger.critical(f"{C_CRIT}Place SL: Exception: {e}{C_RESET}", exc_info=True); self.stop_loss_order_id = None; await self._trigger_alert(f"EXCEPTION during SL place ({type(e).__name__})! Pos UNPROTECTED.", critical=True); await self._emergency_close(f"Exception ({type(e).__name__}) during SL Place")

    async def _cancel_stop_loss(self) -> bool:
        """ Cancels the currently tracked stop loss order. """
        if not self.exchange: self.logger.error(f"{C_ERROR}Cancel SL: No exchange.{C_RESET}"); return False
        if not self.stop_loss_order_id: self.logger.debug(f"{C_DEBUG}Cancel SL: No active SL tracked."); return True
        sl_id = self.stop_loss_order_id; sl_id_s = format_order_id(sl_id)
        self.logger.info(f"{C_INFO}Attempting banish SL order: {sl_id_s}")
        try:
            cat = bybit.market_cache.get_category(self.symbol); if not cat: self.logger.error(f"{C_ERROR}Cancel SL: No cat {self.symbol}.{C_RESET}"); return False
            success = await bybit.cancel_order( exchange=self.exchange, symbol=self.symbol, order_id=sl_id, app_config=self.app_config, order_filter=OrderFilter.STOP_ORDER) # Specify filter
            if success: self.logger.info(f"{C_SUCCESS}SL {sl_id_s} banished OK (API success/already gone).{C_RESET}"); self.stop_loss_order_id = None; return True
            else: # cancel_order returns False on unexpected API errors
                self.logger.error(f"{C_ERROR}Cancel SL: Failed banish SL {sl_id_s} (API error).{C_RESET}")
                await asyncio.sleep(1); await self._update_state() # Verify
                if self.stop_loss_order_id == sl_id and sl_id in self.open_orders: self.logger.error(f"{C_ERROR}Cancel SL: FAILED and {sl_id_s} still open!{C_RESET}"); return False
                else: self.logger.info(f"{C_INFO}Cancel SL: Re-check confirms {sl_id_s} gone."); if self.stop_loss_order_id == sl_id: self.stop_loss_order_id = None; return True
        except Exception as e: self.logger.error(f"{C_ERROR}Cancel SL: Exception: {e}{C_RESET}", exc_info=True); return False

    async def _manage_position(self, entry_side: Optional[Side], should_exit: bool, df_indicators: pd.DataFrame):
        """ Orchestrates entry and exit based on signals. """
        pos_side = self.get_position_side(); is_flat = pos_side is None
        # Exit Logic
        if should_exit and not is_flat:
            pos_q = self.get_position_size(); qty_f = format_amount(self.exchange, self.symbol, pos_q) if self.exchange else str(pos_q)
            self.logger.warning(f"{C_WARN}{C_BOLD}Exit signal {pos_side} pos ({qty_f}). Closing...{C_RESET}")
            self.logger.info(f"{C_INFO}Exit: Cancelling SL..."); sl_ok = await self._cancel_stop_loss()
            if not sl_ok: self.logger.error(f"{C_ERROR}Exit: Failed cancel SL! Closing anyway.{C_RESET}"); await asyncio.sleep(0.5)
            self.logger.info(f"{C_INFO}Exit: Submitting market close {pos_side} pos ({qty_f})..."); close_order = await bybit.close_position_reduce_only(exchange=self.exchange, symbol=self.symbol, app_config=self.app_config, position_to_close=self.current_position, reason="Strategy Exit (EVT)")
            if close_order: close_id_s = format_order_id(close_order.get('id')); self.logger.success(f"{C_SUCCESS}Exit: Close order {close_id_s} submitted OK.{C_RESET}"); await self._trigger_alert(f"Closed {pos_side} Pos ({qty_f}). Reason: EVT Exit")
            else: self.logger.critical(f"{C_CRIT}Exit: FAILED submit close order! Manual check needed.{C_RESET}"); await self._trigger_alert(f"URGENT: Failed CLOSE order {pos_side} ({qty_f})! Check Manually!", critical=True)
            self.logger.info(f"{C_INFO}Exit: Updating state after close..."); await asyncio.sleep(5); await self._update_state(); return # Prevent entry
        # Entry Logic
        if entry_side is not None and is_flat:
            self.logger.info(f"{C_BOLD}{C_INFO}Entry signal: {entry_side.value.upper()}. Preparing...{C_RESET}")
            non_sl = {oid: o for oid, o in self.open_orders.items() if oid != self.stop_loss_order_id}; # Optional: Clean non-SL orders
            if non_sl: self.logger.warning(f"{C_WARN}Entry: Found {len(non_sl)} unexpected orders. Banishing..."); cat = bybit.market_cache.get_category(self.symbol);
                 if cat and self.exchange: cancelled = await bybit.cancel_all_orders(self.exchange, self.app_config, symbol=self.symbol, category=cat, order_filter=OrderFilter.ORDER, reason="Pre-Entry Clean"); await asyncio.sleep(1); await self._update_state()
            entry_px_est = self.last_known_price; if entry_px_est <= 0: self.logger.error(f"{C_ERROR}Entry: Invalid est price ({entry_px_est}). Abort.{C_RESET}"); return
            sl_px_init = self._calculate_stop_loss_price(df_indicators, entry_side, entry_px_est); if sl_px_init is None: self.logger.error(f"{C_ERROR}Entry: Failed initial SL calc. Abort.{C_RESET}"); return
            qty_order = self._calculate_position_size(entry_px_est, sl_px_init); if qty_order is None or qty_order <= 0: self.logger.error(f"{C_ERROR}Entry: Failed qty calc ({qty_order}). Abort.{C_RESET}"); return
            qty_f = format_amount(self.exchange, self.symbol, qty_order) if self.exchange else str(qty_order)
            self.logger.info(f"{C_INFO}Entry: Placing {entry_side.value.upper()} market entry ({qty_f})..."); entry_order = await bybit.place_market_order_slippage_check(exchange=self.exchange, symbol=self.symbol, side=entry_side, amount=qty_order, app_config=self.app_config, is_reduce_only=False, time_in_force=TimeInForce.IOC, position_idx=self.position_idx, reason="Strategy Entry (EVT)")
            # Handle Entry Result & Place SL
            if entry_order and entry_order.get('status') in ['closed', 'filled']:
                entry_id_s = format_order_id(entry_order.get('id', 'N/A')); self.logger.success(f"{C_SUCCESS}Entry: Market entry {entry_id_s} filled/closed.{C_RESET}")
                fill_px = safe_decimal_conversion(entry_order.get('average')); fill_qty = safe_decimal_conversion(entry_order.get('filled'))
                if not fill_px or not fill_qty or fill_qty <= 0: # Verify fill from state if needed
                    self.logger.warning(f"{C_WARN}Entry: Receipt lacks fill details. Re-fetching state..."); await asyncio.sleep(5); await self._update_state(); expect_side = self.api_config.pos_long if entry_side == Side.BUY else self.api_config.pos_short
                    if self.get_position_side() == expect_side: self.logger.info(f"{C_INFO}Entry: State confirmed."); fill_px = safe_decimal_conversion(self.current_position.get('entryPrice')) or entry_px_est; fill_qty = self.get_position_size() or qty_order
                    else: self.logger.critical(f"{C_CRIT}Entry: FAILED entry confirm via state! Order:{entry_id_s}. Expect:{expect_side}, Got:{self.get_position_side()}. Abort SL.{C_RESET}"); await self._trigger_alert(f"CRITICAL: Failed Entry Confirm! Order:{entry_id_s}.", critical=True); return
                if fill_qty <= 0: self.logger.error(f"{C_ERROR}Entry: Confirmed fill qty zero ({fill_qty}). Abort SL.{C_RESET}"); return
                fill_q_f = format_amount(self.exchange, self.symbol, fill_qty) or "N/A"; fill_p_f = format_price(self.exchange, self.symbol, fill_px) or "N/A"
                self.logger.success(f"{C_SUCCESS}{C_BOLD}Entry Confirmed: Side={entry_side.value}, Qty={fill_q_f}, AvgPx={fill_p_f}{C_RESET}"); await self._trigger_alert(f"Entered {entry_side.value} {fill_q_f} @ {fill_p_f}")
                self.logger.info(f"{C_INFO}Entry: Recalculating SL based on fill {fill_p_f}..."); sl_px_actual = self._calculate_stop_loss_price(df_indicators, entry_side, fill_px)
                if sl_px_actual is None: self.logger.critical(f"{C_CRIT}Entry: FAILED SL RECALC after entry @ {fill_p_f}. POS UNPROTECTED.{C_RESET}"); await self._trigger_alert(f"URGENT: Failed SL RECALC {entry_side.value} ({fill_q_f})! Pos UNPROTECTED.", critical=True); await self._emergency_close(f"Failed SL Recalc after Entry"); return
                await asyncio.sleep(0.5); await self._place_stop_loss(entry_side, fill_qty, sl_px_actual)
                self.logger.info(f"{C_INFO}Entry: Updating state after entry/SL..."); await asyncio.sleep(2); await self._update_state()
            elif entry_order: status = entry_order.get('status','?'); oid_s = format_order_id(entry_order.get('id','N/A')); self.logger.warning(f"{C_WARN}Entry: Order {oid_s} submitted, status '{status}'. Check.{C_RESET}"); await asyncio.sleep(3); await self._update_state(); self.logger.warning(f"{C_WARN}Entry: State after unclear: Pos={self.get_position_side() or 'None'}, Orders={len(self.open_orders)}")
            else: self.logger.error(f"{C_ERROR}Entry: Failed place market order. Abort.{C_RESET}")
        elif not entry_side and not should_exit: self.logger.debug(f"{C_DEBUG}Manage Pos: No signals.")

    async def run_loop(self):
        """ Main strategy execution loop. """
        self.logger.info(f"{C_INFO}{C_BOLD}Initializing strategy...{C_RESET}")
        if not await self._initialize(): self.logger.critical(f"{C_CRIT}Init failed. Shutdown.{C_RESET}"); await self._cleanup(); return
        self.is_running = True
        self.logger.info(f"{C_SUCCESS}{C_BOLD}=== Starting {self.strategy_config.name} Loop ==="); self.logger.info(f"{C_INFO}Symbol:{self.symbol}, TF:{self.timeframe}, Poll:{self.strategy_config.polling_interval_seconds}s")
        self.logger.info(f"{C_INFO}Press CTRL+C for shutdown.")
        while self.is_running:
            try:
                start = time.monotonic(); now_utc = pd.Timestamp.now(tz='UTC'); self.logger.info(f"{C_INFO}{C_BOLD}{'-'*20} Tick ({now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}) {'-'*20}{C_RESET}")
                await self._update_state()
                if self.exchange is None: self.logger.critical(f"{C_CRIT}Connection lost. Stop.{C_RESET}"); await self._trigger_alert("CRITICAL: Conn lost! Stop.", critical=True); self.is_running=False; continue
                df_ind = await self._fetch_and_calculate_indicators()
                if df_ind is not None and not df_ind.empty: entry_s, exit_f = self._check_signals(df_ind); await self._manage_position(entry_s, exit_f, df_ind)
                else: self.logger.warning(f"{C_WARN}Skip manage: indicator failure.{C_RESET}")
                elapsed = time.monotonic()-start; poll = self.strategy_config.polling_interval_seconds; sleep = max(0.1, poll - elapsed)
                self.logger.info(f"{C_INFO}Tick done {elapsed:.2f}s. Sleep {sleep:.2f}s.")
                await asyncio.sleep(sleep)
            except asyncio.CancelledError: self.logger.warning(f"{C_WARN}{C_BOLD}Loop cancelled. Shutdown...{C_RESET}"); self.is_running = False
            except Exception as e: self.logger.critical(f"{C_CRIT}CRITICAL LOOP ERROR: {e}{C_RESET}", exc_info=True); await self._trigger_alert(f"CRITICAL LOOP ERROR: {type(e).__name__}. Check Logs!", critical=True); self.logger.warning(f"{C_WARN}Pause 60s after loop error..."); await asyncio.sleep(60)
        self.logger.info(f"{C_INFO}{C_BOLD}--- Loop Terminated ---{C_RESET}"); await self._cleanup()

    async def stop(self):
        """ Signals the strategy loop to stop gracefully. """
        if self.is_running: self.logger.warning(f"{C_WARN}{C_BOLD}Shutdown signaled. Loop stops post-cycle.{C_RESET}"); self.is_running = False
        else: self.logger.info(f"{C_INFO}Shutdown signaled, but loop not running.")

    async def _cleanup(self):
        """ Performs cleanup: cancels orders, closes connection. """
        self.logger.info(f"{C_INFO}{C_BOLD}--- Cleanup Ritual ---{C_RESET}")
        if self.exchange:
            self.logger.info(f"{C_INFO}Cleanup: Banishing orders {self.symbol}..."); cat = bybit.market_cache.get_category(self.symbol)
            if cat: try: cancelled = await bybit.cancel_all_orders(self.exchange, self.app_config, symbol=self.symbol, category=cat, reason="Shutdown"); self.logger.info(f"{C_SUCCESS}Cleanup: Banished {cancelled} orders.{C_RESET}" if cancelled is not None else f"{C_WARN}Cleanup: cancel_all incomplete.{C_RESET}") ; except Exception as e: self.logger.error(f"{C_ERROR}Cleanup: Error cancel: {e}{C_RESET}", exc_info=True)
            else: self.logger.error(f"{C_ERROR}Cleanup: Cannot cancel, cat unknown.{C_RESET}")
            await self._safe_exchange_close()
        else: self.logger.info(f"{C_INFO}Cleanup: Exchange unavailable.")
        self.logger.info(f"{C_SUCCESS}{C_BOLD}--- Cleanup Complete ---{C_RESET}")

    async def _safe_exchange_close(self):
        """ Safely attempts to close the CCXT exchange connection. """
        ex = self.exchange; self.exchange = None # Clear ref first
        if ex and hasattr(ex, 'close') and getattr(ex, 'closed', True) is False:
            self.logger.info(f"{C_INFO}Closing exchange connection..."); try: await ex.close(); self.logger.info(f"{C_SUCCESS}Exchange closed.") ; except Exception as e: self.logger.error(f"{C_ERROR}Error closing exchange: {e}{C_RESET}", exc_info=True)

    async def _trigger_alert(self, message: str, critical: bool = False):
        """ Sends alert message via configured method. """
        prefix = f"[{self.strategy_config.name}/{self.symbol}] "; full_msg = prefix + message
        log_lvl, log_clr = (logging.CRITICAL, C_CRIT) if critical else (logging.WARNING, C_WARN)
        self.logger.log(log_lvl, f"{log_clr}{C_BOLD}ALERT: {message}{C_RESET}")
        if self.sms_config.enable_sms_alerts and self.loop and not self.loop.is_closed():
            self.logger.debug(f"{C_DEBUG}Dispatching SMS alert...")
            try: await self.loop.run_in_executor(None, send_sms_alert, full_msg, self.sms_config) # Pass SMSConfig part
            except Exception as e: self.logger.error(f"{C_ERROR}Failed dispatch alert: {e}{C_RESET}", exc_info=True)
        elif self.sms_config.enable_sms_alerts: self.logger.warning(f"{C_WARN}Cannot dispatch SMS: loop unavailable.{C_RESET}")

    async def _emergency_close(self, reason: str):
        """ Attempts immediate market close of position in emergency. """
        self.logger.critical(f"{C_CRIT}{C_BOLD}EMERGENCY CLOSE! Reason: {reason}{C_RESET}")
        await self._trigger_alert(f"EMERGENCY Closing! Reason: {reason}", critical=True)
        self.logger.info(f"{C_INFO}Emergency: Quick state update...")
        try: await asyncio.wait_for(self._update_state(), timeout=10.0)
        except asyncio.TimeoutError: self.logger.error(f"{C_ERROR}Emergency: State update timeout! Proceed blind.{C_RESET}")
        except Exception as e: self.logger.error(f"{C_ERROR}Emergency: Error update state ({e}). Proceed blind.{C_RESET}")
        pos_q = self.get_position_size(); pos_s = self.get_position_side()
        if pos_s and pos_q > self.api_config.position_qty_epsilon:
            qty_f = format_amount(self.exchange, self.symbol, pos_q) if self.exchange else str(pos_q)
            self.logger.warning(f"{C_WARN}Emergency: Closing {pos_s} pos ({qty_f})...{C_RESET}")
            await self._cancel_stop_loss(); await asyncio.sleep(0.2) # Best effort SL cancel
            close_order = await bybit.close_position_reduce_only(exchange=self.exchange, symbol=self.symbol, app_config=self.app_config, position_to_close=self.current_position, reason=f"Emergency: {reason}")
            if close_order: close_id_s = format_order_id(close_order.get('id')); self.logger.warning(f"{C_WARN}{C_BOLD}Emergency close order submitted: {close_id_s}. Monitor manually.{C_RESET}")
            else: self.logger.critical(f"{C_CRIT}Emergency: FAILED SUBMIT emergency close! MANUAL INTERVENTION REQUIRED!{C_RESET}"); await self._trigger_alert(f"!!! CRITICAL FAILURE: FAILED submit EMERGENCY CLOSE {pos_s} ({qty_f}) !!! CHECK NOW!", critical=True)
        else: self.logger.info(f"{C_INFO}Emergency close triggered, but no open position found.")
EOF

# --- Create main.py ---
echo "# Creating main.py"
cat << 'EOF' > main.py
# main.py
""" Main entry point for the Bybit Trading Bot. """

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from typing import Optional

# --- Define COLORAMA constants early ---
try:
    from colorama import Back, Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    C_INFO, C_SUCCESS, C_WARN, C_ERROR = Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.RED + Style.BRIGHT
    C_CRIT, C_DEBUG, C_BOLD, C_RESET = Back.RED + Fore.WHITE + Style.BRIGHT, Fore.MAGENTA, Style.BRIGHT, Style.RESET_ALL
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore
    C_INFO=C_SUCCESS=C_WARN=C_ERROR=C_CRIT=C_DEBUG=C_BOLD=C_RESET=""
    print("Warning: 'colorama' not found. Run 'pip install colorama' for vibrant logs.", file=sys.stderr)

# --- Import local modules ---
try:
    from config_models import AppConfig, load_config # Use Pydantic config loader
    from neon_logger import setup_logger
    # --- IMPORT YOUR CHOSEN STRATEGY CLASS HERE ---
    from ehlers_volumetric_strategy import EhlersVolumetricStrategyEnhanced as StrategyClass
    # Example for different strategy:
    # from macd_rsi_strategy import MACDRSIStrategy as StrategyClass
    # --------------------------------------------
    import bybit_helpers as bybit # Needed for final cleanup
except ImportError as e:
    print(f"{C_CRIT}FATAL: Import Error: {e}{C_RESET}", file=sys.stderr)
    print("Ensure config_models.py, neon_logger.py, bybit_helpers.py, bybit_utils.py, indicators.py, and your strategy file exist.", file=sys.stderr)
    traceback.print_exc(); sys.exit(1)
except Exception as e:
    print(f"{C_CRIT}FATAL: Unexpected Import Error: {e}{C_RESET}", file=sys.stderr)
    traceback.print_exc(); sys.exit(1)


# --- Global Variables ---
logger: Optional[logging.Logger] = None
strategy_instance: Optional[StrategyClass] = None
main_task: Optional[asyncio.Task] = None
shutdown_requested = False
app_config: Optional[AppConfig] = None # Store loaded config globally

# --- Signal Handling for Graceful Shutdown ---
async def handle_signal(sig: signal.Signals):
    """ Asynchronously handles shutdown signals. """
    global shutdown_requested, logger, main_task, strategy_instance
    if shutdown_requested: print("\nShutdown already in progress..."); return
    shutdown_requested = True
    signal_name = signal.Signals(sig).name
    print(f"\n{C_WARN}{C_BOLD}>>> Signal {signal_name} ({sig}) received. Initiating graceful shutdown... <<< {C_RESET}")
    if logger: logger.warning(f"Signal {signal_name} received. Initiating shutdown...")

    # 1. Request strategy loop stop (non-blocking)
    if strategy_instance and getattr(strategy_instance, 'is_running', False):
        if logger: logger.info("Requesting strategy loop stop...")
        asyncio.create_task(strategy_instance.stop()) # Fire and forget stop request
        # Allow some time for loop to potentially finish gracefully
        await asyncio.sleep(1.5)

    # 2. Cancel the main task
    if main_task and not main_task.done():
        if logger: logger.info("Cancelling main strategy task...")
        main_task.cancel()
        try: await asyncio.wait_for(main_task, timeout=10.0) # Wait for cancellation
        except asyncio.CancelledError: logger.info("Main task cancelled.")
        except asyncio.TimeoutError: logger.error("Timeout waiting for main task cancellation!")
        except Exception as e: logger.error(f"Error during main task cancellation: {e}")

    # Further cleanup in main()'s finally block

async def main():
    """ Main async function to setup and run the strategy. """
    global logger, strategy_instance, main_task, app_config

    start_time = time.monotonic()
    # --- 1. Load Configuration ---
    try:
        app_config = load_config() # Load config using Pydantic Settings
    except SystemExit: # load_config raises SystemExit on validation failure
        return # Exit gracefully if config failed validation
    except Exception as cfg_err:
        print(f"{C_CRIT}FATAL: Unhandled error loading configuration: {cfg_err}{C_RESET}", file=sys.stderr)
        traceback.print_exc(); sys.exit(1)

    # --- 2. Setup Logger (using loaded config) ---
    try:
        logger = setup_logger(app_config.logging_config) # Pass LoggingConfig part
        logger.info("=" * 60)
        logger.info(f"=== {app_config.logging_config.logger_name} Initializing ===")
        logger.info(f"Strategy: {app_config.strategy_config.name}")
        logger.info(f"Symbol: {app_config.api_config.symbol}")
        logger.info(f"Timeframe: {app_config.strategy_config.timeframe}")
        logger.info(f"Testnet Mode: {app_config.api_config.testnet_mode}")
        logger.info("=" * 60)
    except Exception as e:
        print(f"{C_CRIT}FATAL: Logger setup failed: {e}{C_RESET}", file=sys.stderr)
        traceback.print_exc(); sys.exit(1)

    # --- 3. Instantiate Strategy ---
    try:
        logger.info(f"Instantiating strategy: {StrategyClass.__name__}...")
        # Pass the entire AppConfig object to the strategy
        strategy_instance = StrategyClass(app_config=app_config, logger=logger)
    except ValueError as e: # Catch strategy-specific validation errors
         logger.critical(f"Strategy instantiation failed: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.critical(f"Failed to instantiate strategy: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Run the Strategy Loop ---
    run_success = False
    exchange_instance_ref = None # Keep ref for final cleanup
    try:
        logger.info("Starting strategy execution...")
        # Strategy initialization is now part of run_loop
        main_task = asyncio.create_task(strategy_instance.run_loop())

        # Wait for the task to complete (either normally or via cancellation)
        await main_task

        # Check task outcome
        if main_task.cancelled(): logger.warning("Strategy loop task was cancelled (expected during shutdown)."); run_success = True
        elif main_task.exception(): loop_exception = main_task.exception(); logger.critical(f"Strategy loop task exited with exception: {loop_exception}", exc_info=loop_exception); run_success = False
        else: logger.info("Strategy run_loop finished normally."); run_success = True

    except asyncio.CancelledError: logger.warning("Main execution task cancelled (shutdown)."); run_success = True
    except Exception as e: logger.critical(f"Main execution block error: {e}", exc_info=True); run_success = False
    finally:
        logger.info(f"{C_BOLD}--- Main Execution Finalizing (Run Success: {run_success}) ---{C_RESET}")
        # --- Final Cleanup ---
        # Get exchange reference from strategy *before* cleanup, if it was initialized
        if strategy_instance: exchange_instance_ref = strategy_instance.exchange
        if strategy_instance and hasattr(strategy_instance, '_cleanup'):
             logger.info("Running strategy internal cleanup...")
             try: await strategy_instance._cleanup() # Ensure cleanup is awaited
             except Exception as strategy_cleanup_err: logger.error(f"Strategy cleanup error: {strategy_cleanup_err}", exc_info=True)

        if exchange_instance_ref and hasattr(exchange_instance_ref, 'close') and callable(exchange_instance_ref.close):
             logger.info("Final exchange connection close attempt...")
             try:
                 if not getattr(exchange_instance_ref, 'closed', True): # Check if already closed
                     await exchange_instance_ref.close(); logger.info(f"{C_SUCCESS}Final exchange close OK.{C_RESET}")
                 else: logger.info("Exchange connection already closed.")
             except Exception as final_close_err: logger.error(f"Final exchange close error: {final_close_err}", exc_info=True)
        else: logger.debug("Exchange ref not available or no close method for final cleanup.")

        if not run_success: logger.error("Main execution finished due to errors.")
        end_time = time.monotonic(); total_runtime = end_time - start_time
        logger.info(f"--- Application Shutdown Complete (Total Runtime: {total_runtime:.2f}s) ---")

# --- Script Entry Point ---
if __name__ == "__main__":
    print(f"{C_INFO}Starting Asynchronous Trading Bot...{C_RESET}")
    event_loop: Optional[asyncio.AbstractEventLoop] = None
    try:
        # Use high-level asyncio.run() which handles loop creation and shutdown
        # Setup signal handlers within the main coroutine if possible, or use run_until_complete if needed.
        # Using run() is generally preferred for simplicity.

        async def main_wrapper():
             # Get event loop inside the async context where it's guaranteed to exist
             loop = asyncio.get_running_loop()
             # Register signal handlers
             signals_to_handle = (signal.SIGINT, signal.SIGTERM) # Common shutdown signals
             for s in signals_to_handle:
                 try: loop.add_signal_handler(s, lambda s=s: asyncio.create_task(handle_signal(s)))
                 except NotImplementedError: print(f"Warning: Signal {s.name} handling not supported on this system.", file=sys.stderr)
                 except RuntimeError as e: print(f"Warning: Failed register signal {s.name}: {e}", file=sys.stderr)
             await main() # Run the core application logic

        asyncio.run(main_wrapper())

    except KeyboardInterrupt: print(f"\n{C_WARN}KeyboardInterrupt caught at top level. Exiting.{C_RESET}")
    except SystemExit as e: print(f"SystemExit called with code {e.code}.") # Allow clean exit
    except Exception as top_level_exc:
        print(f"{C_CRIT}FATAL UNHANDLED ERROR: {top_level_exc}{C_RESET}", file=sys.stderr)
        traceback.print_exc()
        if logger: logger.critical("Fatal unhandled error", exc_info=True)
        sys.exit(1)
    finally:
        print(f"{C_SUCCESS}{C_BOLD}--- Script Execution Finished ---{C_RESET}")
EOF

# --- Create .env file ---
echo "# Creating .env placeholder file"
cat << EOF > .env
# --- Bybit API Credentials ---
# ** REPLACE PLACEHOLDERS BELOW WITH YOUR ACTUAL KEYS **
# For Testnet: Get keys from https://testnet.bybit.com/app/user/api-management
# For Mainnet: Get keys from https://www.bybit.com/app/user/api-management
# Ensure keys have permissions for: Orders, Positions (Read & Write for trading)
BOT_API_CONFIG__API_KEY="YOUR_API_KEY_PLACEHOLDER"
BOT_API_CONFIG__API_SECRET="YOUR_API_SECRET_PLACEHOLDER"
BOT_API_CONFIG__TESTNET_MODE=True # Set to False for LIVE trading!

# --- Strategy Configuration ---
# Ensure the strategy name matches the class used in main.py
BOT_STRATEGY_CONFIG__NAME="EhlersVolumetricEnhanced"
BOT_API_CONFIG__SYMBOL="BTC/USDT:USDT" # Trading symbol
BOT_STRATEGY_CONFIG__TIMEFRAME="15m"   # e.g., 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M
BOT_STRATEGY_CONFIG__LEVERAGE=5        # Desired leverage (check symbol limits on Bybit)
BOT_STRATEGY_CONFIG__RISK_PER_TRADE="0.01" # e.g., 0.01 for 1% risk

# Ehlers Volumetric Specific (if needed to override defaults in config_models.py)
# BOT_STRATEGY_CONFIG__INDICATOR_SETTINGS__EVT_LENGTH=7
# BOT_STRATEGY_CONFIG__INDICATOR_SETTINGS__EVT_MULTIPLIER=2.5
# BOT_STRATEGY_CONFIG__INDICATOR_SETTINGS__ATR_PERIOD=14
# BOT_STRATEGY_CONFIG__STOP_LOSS_ATR_MULTIPLIER="2.0"

# --- Logging Configuration ---
BOT_LOGGING_CONFIG__CONSOLE_LEVEL_STR="INFO" # DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
BOT_LOGGING_CONFIG__FILE_LEVEL_STR="DEBUG"
BOT_LOGGING_CONFIG__LOG_FILE="trading_bot.log" # Set to "" or remove line to disable file logging

# --- SMS Alert Configuration (Optional) ---
BOT_SMS_CONFIG__ENABLE_SMS_ALERTS=False # Set to True to enable
BOT_SMS_CONFIG__USE_TERMUX_API=False     # Set to True if using Termux:API
BOT_SMS_CONFIG__SMS_RECIPIENT_NUMBER="" # Your phone number (E.164 format, e.g., +11234567890)
# Add Twilio vars here if implementing Twilio support

EOF

# --- Create .gitignore file ---
echo "# Creating .gitignore"
cat << 'EOF' > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.hypothesis/
.pytest_cache/

# Environments
.env
.venv/
venv/
env/
ENV/
# misc virtual env files
pyvenv.cfg
pip-selfcheck.json

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Jupyter Notebook
.ipynb_checkpoints

# IDE - VSCode
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
*.code-workspace

# IDE - PyCharm
.idea/

# Application logs
*.log
logs/

# Other
*.db
*.sqlite3
*.csv # Maybe exclude specific data files if needed
*.json # Maybe exclude specific config/data files if needed

# Local config overrides if any (e.g., config_local.py)
config_local.py
EOF

# --- Create requirements.txt file ---
echo "# Creating requirements.txt"
cat << 'EOF' > requirements.txt
# Core Libraries
ccxt>=4.1.0 # Ensure a version supporting async and Bybit V5 is used
pandas>=2.0.0 # For data manipulation and indicators
pandas_ta>=0.3.14b # For technical indicators
pydantic>=2.0.0 # For configuration validation
pydantic-settings>=2.0.0 # For loading config from .env

# Optional Libraries
colorama>=0.4.6 # For colored console output
websockets>=11.0.0 # For WebSocket streaming (if used)
python-dotenv>=1.0.0 # For loading .env file

# Add other dependencies if your custom indicators or strategy require them
# e.g., numpy (often installed with pandas)
# e.g., aiohttp (sometimes used by ccxt)
# e.g., twilio (if implementing Twilio SMS)
EOF

# --- Create README.md file ---
echo "# Creating README.md"
cat << 'EOF' > README.md
# Bybit Ehlers Volumetric Strategy Bot

This project implements an automated trading bot for the Bybit exchange (V5 API),
using an enhanced Ehlers Volumetric Trend (EVT) strategy.

## Features

*   Connects to Bybit (Mainnet or Testnet) via CCXT.
*   Uses Pydantic for robust configuration loading and validation (`.env` file).
*   Implements the Ehlers Volumetric Trend indicator.
*   Calculates position size based on risk percentage and ATR stop loss.
*   Places market entry orders and native conditional stop-loss orders.
*   Manages position state and handles entry/exit signals.
*   Structured logging with console (colored) and file output.
*   Modular code structure (config, helpers, utils, indicators, strategy, main).
*   Optional SMS alerting via Termux:API (requires Android/Termux setup).
*   Graceful shutdown handling.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd bybit_evt_strategy
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    *   Copy or rename the `.env` file.
    *   **Edit `.env` and replace `YOUR_API_KEY_PLACEHOLDER` and `YOUR_API_SECRET_PLACEHOLDER` with your actual Bybit API keys.**
    *   Get keys from:
        *   Testnet: <https://testnet.bybit.com/app/user/api-management>
        *   Mainnet: <https://www.bybit.com/app/user/api-management>
    *   Ensure the keys have permissions for `Orders` and `Positions` (Read & Write). Unified Trading Account scope is needed.

5.  **Review Configuration:**
    *   Check other settings in `.env`, such as `BOT_API_CONFIG__SYMBOL`, `BOT_STRATEGY_CONFIG__TIMEFRAME`, `BOT_STRATEGY_CONFIG__LEVERAGE`, `BOT_STRATEGY_CONFIG__RISK_PER_TRADE`.
    *   Set `BOT_API_CONFIG__TESTNET_MODE` to `False` for live trading (default is `True`).
    *   Configure SMS alerts if desired and using Termux.

## Running the Bot

```bash
python main.py
```

The bot will start, initialize the connection, and begin its trading loop. Monitor the console output and the `trading_bot.log` file.

Press `CTRL+C` to initiate a graceful shutdown.

## Disclaimer

Trading cryptocurrencies involves significant risk. This software is provided "as is" without warranty of any kind. Use it at your own risk. Ensure you understand the strategy and the code before deploying with real funds. Past performance is not indicative of future results. **Never risk more than you can afford to lose.**
EOF

# --- Initialize Git Repository ---
echo "Initializing Git repository..."
git init
git add .

# --- First Commit ---
echo "Creating initial commit..."
git commit -m "Initial commit: Setup Ehlers Volumetric Strategy project structure"

echo ""
echo "-----------------------------------------------------"
echo "Project '$PROJECT_DIR' created and Git repo initialized."
echo "-----------------------------------------------------"
echo ""
echo "${Fore.YELLOW}NEXT STEPS:${Style.RESET_ALL}"
echo "1.  ${C_BOLD}cd $PROJECT_DIR${C_RESET}"
echo "2.  ${C_BOLD}Create and activate a Python virtual environment${C_RESET} (e.g., python -m venv venv && source venv/bin/activate)"
echo "3.  ${C_BOLD}Install requirements:${C_RESET} pip install -r requirements.txt"
echo "4.  ${C_ERROR}${C_BOLD}EDIT THE '.env' FILE${C_RESET} and add your actual Bybit API keys!"
echo "5.  Review other settings in '.env' (Symbol, Timeframe, Leverage, Risk, Testnet Mode)."
echo "6.  (Optional) Configure Git user for this repository:"
echo "    git config user.name \"$GIT_USER_NAME\""
echo "    git config user.email \"$GIT_USER_EMAIL\""
echo "7.  (Optional) Add a remote repository (e.g., GitHub):"
echo "    git remote add origin git@github.com:$GIT_USER_NAME/$PROJECT_DIR.git"
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""
echo "Run the bot using: ${C_BOLD}python main.py${C_RESET}"
echo ""

```

**How to Use:**

1.  Save the bash code above into a file named `create_project.sh`.
2.  Make it executable: `chmod +x create_project.sh`.
3.  Run it from the directory where you want the project folder created: `./create_project.sh`.
4.  Follow the "NEXT STEPS" printed by the script.