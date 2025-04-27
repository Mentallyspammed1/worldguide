#!/bin/bash
# Script to create the directory structure, populate files for the Bybit EVT strategy bot,
# and initialize a new Git repository.
# Includes enhanced Python code snippets provided by the user.

# --- Safety Settings ---
# Exit immediately if a command exits with a non-zero status. Crucial for preventing partial setup on error.
set -e
# Treat unset variables as an error when substituting.
# set -u # Be cautious with set -u, especially if sourcing other scripts or relying on potentially unset env vars.

# --- Configuration ---
PROJECT_DIR="bybit_evt_strategy"
# These GIT variables are primarily used for generating instructions in the final output.
GIT_USER_NAME="YourGitHubUsername" # Replace with your GitHub/GitLab username
GIT_USER_EMAIL="your_email@example.com" # Replace with your actual email used for Git

# --- ANSI Color Codes for Output ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_INFO='\033[36m'    # Cyan
C_SUCCESS='\033[32m' # Green
C_WARN='\033[33m'    # Yellow
C_ERROR='\033[91m'   # Bright Red

# --- Pre-flight Checks ---
echo -e "${C_INFO}${C_BOLD}Starting Project Setup: ${PROJECT_DIR}${C_RESET}"

# Check if Git is installed
if ! command -v git &> /dev/null; then
  echo -e "${C_ERROR}Error: 'git' command not found. Please install Git.${C_RESET}"
  exit 1
fi

# Safety Check: Prevent overwriting existing directory
if [ -d "$PROJECT_DIR" ]; then
  echo -e "${C_ERROR}Error: Directory '${PROJECT_DIR}' already exists in the current location.${C_RESET}"
  echo -e "${C_WARN}Please remove or rename the existing directory before running this script.${C_RESET}"
  exit 1
fi

# --- Directory Creation ---
echo -e "${C_INFO}Creating project directory: ${PROJECT_DIR}${C_RESET}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR" # Change into the project directory for subsequent file creation

echo -e "${C_INFO}Creating Python source files...${C_RESET}"

# --- Create config_models.py (Placeholder/Structure based on other files) ---
echo -e "${C_INFO} -> Generating config_models.py${C_RESET}"
cat << 'EOF' > config_models.py
# config_models.py
"""
Pydantic Models for Application Configuration using pydantic-settings.

Loads configuration from environment variables and/or a .env file.
Provides type validation and default values for the trading bot.
"""

import logging
import os
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define basic types or Literals as placeholders - MUST match the expected values
# These are used if direct import from helpers fails during early setup,
# but the helpers should ideally define and export these properly.
PositionIdx = Literal[0, 1, 2] # 0: One-Way, 1: Hedge Buy, 2: Hedge Sell
Category = Literal["linear", "inverse", "spot", "option"]
OrderFilter = Literal["Order", "StopOrder", "tpslOrder", "TakeProfit", "StopLoss"] # Add others as needed
Side = Literal["buy", "sell"] # Use lowercase consistent with strategy/ccxt args
TimeInForce = Literal["GTC", "IOC", "FOK", "PostOnly"]
TriggerBy = Literal["LastPrice", "MarkPrice", "IndexPrice"]
TriggerDirection = Literal[1, 2] # 1: Rise, 2: Fall


class APIConfig(BaseModel):
    """Configuration for Bybit API Connection and Market Defaults."""
    exchange_id: Literal["bybit"] = Field("bybit", description="CCXT Exchange ID")
    api_key: Optional[str] = Field(None, description="Bybit API Key")
    api_secret: Optional[str] = Field(None, description="Bybit API Secret")
    testnet_mode: bool = Field(True, description="Use Bybit Testnet environment")
    default_recv_window: PositiveInt = Field(10000, ge=100, le=60000, description="API request validity window (ms)")

    symbol: str = Field(..., description="Primary trading symbol (e.g., BTC/USDT:USDT)")
    usdt_symbol: str = Field("USDT", description="Quote currency for balance reporting")
    expected_market_type: Literal["swap", "spot", "option", "future"] = Field("swap")
    expected_market_logic: Literal["linear", "inverse"] = Field("linear")

    retry_count: NonNegativeInt = Field(3, description="Default API call retry count")
    retry_delay_seconds: PositiveFloat = Field(2.0, gt=0, description="Default base retry delay (s)")

    maker_fee_rate: Decimal = Field(Decimal("0.0002"), ge=0, description="Maker fee rate")
    taker_fee_rate: Decimal = Field(Decimal("0.00055"), ge=0, description="Taker fee rate")

    default_slippage_pct: Decimal = Field(Decimal("0.005"), gt=0, le=Decimal("0.1"), description="Max slippage % for market orders (0.005 = 0.5%)")
    position_qty_epsilon: Decimal = Field(Decimal("1e-9"), gt=0, description="Small value for quantity comparisons")
    shallow_ob_fetch_depth: PositiveInt = Field(5, ge=1, le=50, description="Order book depth for slippage check")
    order_book_fetch_limit: PositiveInt = Field(25, ge=1, le=1000, description="Default depth for fetching L2 order book")

    pos_none: Literal["NONE"] = "NONE"
    pos_long: Literal["LONG"] = "LONG"
    pos_short: Literal["SHORT"] = "SHORT"
    side_buy: Side = "buy"
    side_sell: Side = "sell"

    @field_validator('api_key', 'api_secret', mode='before')
    @classmethod
    def check_not_placeholder(cls, v: Optional[str], info) -> Optional[str]:
        if v and isinstance(v, str) and "PLACEHOLDER" in v.upper():
             print(f"\033[93mWarning [APIConfig]: Field '{info.field_name}' looks like a placeholder: '{v[:15]}...'\033[0m")
        return v

    @field_validator('symbol', mode='before')
    @classmethod
    def check_and_format_symbol(cls, v: Any) -> str:
        if not isinstance(v, str) or not v: raise ValueError("Symbol must be a non-empty string")
        if ":" not in v and "/" not in v: raise ValueError(f"Invalid symbol format: '{v}'. Expected 'BASE/QUOTE:SETTLE' or 'BASE/QUOTE'.")
        return v.strip().upper()


class IndicatorSettings(BaseModel):
    """Parameters for Technical Indicator Calculations."""
    min_data_periods: PositiveInt = Field(100, ge=20, description="Min candles for indicators")
    evt_length: PositiveInt = Field(7, gt=1, description="EVT indicator length")
    evt_multiplier: PositiveFloat = Field(2.5, gt=0, description="EVT bands multiplier")
    atr_period: PositiveInt = Field(14, gt=0, description="ATR indicator length")


class AnalysisFlags(BaseModel):
    """Flags to Enable/Disable Specific Indicator Calculations or Features."""
    use_evt: bool = Field(True, description="Enable EVT calculation")
    use_atr: bool = Field(True, description="Enable ATR calculation (for SL/TP)")


class StrategyConfig(BaseModel):
    """Core Strategy Behavior and Parameters."""
    name: str = Field("EhlersVolumetricV1", description="Name of the strategy instance")
    timeframe: str = Field("5m", pattern=r"^\d+[mhdMy]$", description="Candlestick timeframe (e.g., '5m')")
    ohlcv_limit: PositiveInt = Field(200, ge=50, description="Number of candles to fetch for indicators")

    leverage: PositiveInt = Field(10, ge=1, description="Desired leverage")
    default_margin_mode: Literal['cross', 'isolated'] = Field('cross') # Requires UTA Pro for isolated usually
    default_position_mode: Literal['one-way', 'hedge'] = Field('one-way')
    risk_per_trade: Decimal = Field(Decimal("0.01"), gt=0, le=Decimal("0.1"), description="Fraction of balance to risk (0.01 = 1%)")

    stop_loss_atr_multiplier: Decimal = Field(Decimal("2.0"), gt=0, description="ATR multiplier for stop loss")
    take_profit_atr_multiplier: Decimal = Field(Decimal("3.0"), gt=0, description="ATR multiplier for take profit")
    place_tpsl_as_limit: bool = Field(True, description="Place TP/SL as reduce-only Limit orders (True) or use native stops (False)")

    loop_delay_seconds: PositiveInt = Field(60, ge=5, description="Frequency (seconds) to fetch data and check signals")

    # Link indicator settings and flags
    indicator_settings: IndicatorSettings = Field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = Field(default_factory=AnalysisFlags)
    strategy_params: Dict[str, Any] = Field({}, description="Strategy-specific parameters dictionary") # Populated later
    strategy_info: Dict[str, Any] = Field({}, description="Strategy identification dictionary") # Populated later

    @field_validator('timeframe')
    @classmethod
    def check_timeframe_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d+[mhdMy]$", v): raise ValueError(f"Invalid timeframe format: '{v}'.")
        return v

    @model_validator(mode='after')
    def check_consistency(self) -> 'StrategyConfig':
        if self.stop_loss_atr_multiplier > 0 and not self.analysis_flags.use_atr:
            raise ValueError("ATR Multiplier > 0 requires use_atr flag to be True.")
        if self.take_profit_atr_multiplier > 0 and not self.analysis_flags.use_atr:
             raise ValueError("TP ATR Multiplier > 0 requires use_atr flag to be True.")
        # Populate helper dicts after validation
        self.strategy_params = {'ehlers_volumetric': {'evt_length': self.indicator_settings.evt_length, 'evt_multiplier': self.indicator_settings.evt_multiplier}}
        self.strategy_info = {'name': 'ehlers_volumetric'} # Match key used in strategy file
        return self


class LoggingConfig(BaseModel):
    """Configuration for the Logger Setup."""
    logger_name: str = Field("TradingBot", description="Name for the logger instance")
    log_file: Optional[str] = Field("trading_bot.log", description="Log file path (None to disable)")
    console_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field("INFO")
    file_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field("DEBUG")
    log_rotation_bytes: NonNegativeInt = Field(5 * 1024 * 1024, description="Max log size (bytes), 0 disables rotation")
    log_backup_count: NonNegativeInt = Field(5, description="Number of backup logs")
    third_party_log_level_str: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("WARNING")

    @field_validator('log_file', mode='before')
    @classmethod
    def validate_log_file(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() == "": return None
        if any(char in v for char in ['<', '>', ':', '"', '|', '?', '*']): raise ValueError(f"Log file path '{v}' contains invalid characters.")
        return v.strip()

class SMSConfig(BaseModel):
    """Configuration for SMS Alerting."""
    enable_sms_alerts: bool = Field(False, description="Globally enable/disable SMS alerts")
    use_termux_api: bool = Field(False, description="Use Termux:API for SMS")
    sms_recipient_number: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{1,14}$", description="Recipient phone number (E.164 format)")
    sms_timeout_seconds: PositiveInt = Field(30, ge=5, le=120, description="Timeout for Termux API call (s)")

    @model_validator(mode='after')
    def check_sms_details(self) -> 'SMSConfig':
        if self.enable_sms_alerts and self.use_termux_api and not self.sms_recipient_number:
            raise ValueError("Termux SMS enabled, but 'sms_recipient_number' is missing.")
        # Add checks for other providers (e.g., Twilio) if implemented
        return self


class AppConfig(BaseSettings):
    """Master Configuration Model."""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_nested_delimiter='__',
        env_prefix='BOT_',
        case_sensitive=False,
        extra='ignore',
        validate_default=True,
    )

    api_config: APIConfig = Field(default_factory=APIConfig)
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    sms_config: SMSConfig = Field(default_factory=SMSConfig)

    # Helper method to convert to the old single-class Config format if needed temporarily
    def to_legacy_config_dict(self) -> dict:
         legacy = {}
         legacy.update(self.api_config.model_dump())
         legacy.update(self.strategy_config.model_dump(exclude={'indicator_settings', 'analysis_flags', 'strategy_params', 'strategy_info'})) # Exclude nested models
         legacy.update(self.logging_config.model_dump())
         legacy.update(self.sms_config.model_dump())
         # Add back nested items needed by old format
         legacy['indicator_settings'] = self.strategy_config.indicator_settings.model_dump()
         legacy['analysis_flags'] = self.strategy_config.analysis_flags.model_dump()
         legacy['strategy_params'] = self.strategy_config.strategy_params
         legacy['strategy'] = self.strategy_config.strategy_info # Renamed from strategy_info
         # Map level strings back if needed by old logger setup
         legacy['LOG_CONSOLE_LEVEL'] = self.logging_config.console_level_str
         legacy['LOG_FILE_LEVEL'] = self.logging_config.file_level_str
         legacy['LOG_FILE_PATH'] = self.logging_config.log_file
         return legacy


def load_config() -> AppConfig:
    """Loads and validates the application configuration."""
    try:
        print(f"\033[36mLoading configuration...\033[0m")
        # Determine .env path relative to CWD where script is run
        env_file_path = os.path.join(os.getcwd(), '.env')

        if os.path.exists(env_file_path):
            print(f"Attempting to load from: {env_file_path}")
            config = AppConfig(_env_file=env_file_path)
        else:
            print(f"'.env' file not found at {env_file_path}. Loading from environment variables only.")
            config = AppConfig()

        if config.api_config.api_key and "PLACEHOLDER" in config.api_config.api_key.upper():
             print("\033[91m\033[1mCRITICAL WARNING: API Key is a placeholder. Bot WILL fail authentication.\033[0m")
        if not config.api_config.testnet_mode:
             print("\033[93m\033[1mWARNING: Testnet mode is DISABLED. Bot will attempt LIVE trading.\033[0m")

        print(f"\033[32mConfiguration loaded successfully.\033[0m")
        return config

    except ValidationError as e:
        print(f"\n{'-'*20}\033[91m CONFIGURATION VALIDATION FAILED \033[0m{'-'*20}")
        # error.loc gives a tuple path, e.g., ('api_config', 'symbol')
        for error in e.errors():
            loc_path = " -> ".join(map(str, error['loc'])) if error['loc'] else 'AppConfig'
            env_var_suggestion = "BOT_" + "__".join(map(str, error['loc'])).upper()
            print(f"  \033[91mField:\033[0m {loc_path}")
            print(f"  \033[91mError:\033[0m {error['msg']}")
            val_display = repr(error.get('input', 'N/A'))
            is_secret = any(s in loc_path.lower() for s in ['key', 'secret', 'token'])
            if is_secret and isinstance(error.get('input'), str): val_display = "'*****'"
            print(f"  \033[91mValue:\033[0m {val_display}")
            print(f"  \033[93mSuggestion:\033[0m Check env var '{env_var_suggestion}' or the field in '.env'.")
            print("-" * 25)
        print(f"{'-'*60}\n")
        raise SystemExit("\033[91mConfiguration validation failed.\033[0m")

    except Exception as e:
        print(f"\033[91m\033[1mFATAL: Unexpected error loading configuration: {e}\033[0m")
        import traceback
        traceback.print_exc()
        raise SystemExit("\033[91mFailed to load configuration.\033[0m")

# Example usage
if __name__ == "__main__":
    print("Running config_models.py directly for testing...")
    try:
        app_settings = load_config()
        print("\n\033[1mLoaded Config (Partial Example):\033[0m")
        print(f"  Symbol: {app_settings.api_config.symbol}")
        print(f"  Timeframe: {app_settings.strategy_config.timeframe}")
        print(f"  Testnet: {app_settings.api_config.testnet_mode}")
        print(f"  Log File: {app_settings.logging_config.log_file}")
        print(f"  Risk %: {app_settings.strategy_config.risk_per_trade * 100:.2f}%")
        # print("\nFull Config (JSON):\n", app_settings.model_dump_json(indent=2)) # Careful with secrets
        print("\n\033[32mConfiguration test successful.\033[0m")
    except SystemExit as e:
         print(f"\n\033[91mExiting due to configuration error: {e}\033[0m")
EOF

# --- Create neon_logger.py (Using provided enhanced version) ---
echo -e "${C_INFO} -> Generating neon_logger.py (v1.2 enhanced)${C_RESET}"
cat << 'EOF' > neon_logger.py
# --- START OF FILE neon_logger.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neon Logger Setup (v1.2) - Enhanced Robustness & Features

Provides a function `setup_logger` to configure a Python logger instance with:
- Colorized console output using a "neon" theme via colorama (TTY only).
- Uses a custom Formatter for cleaner color handling.
- Clean, non-colorized file output.
- Optional log file rotation (size-based).
- Extensive log formatting (timestamp, level, function, line, thread).
- Custom SUCCESS log level.
- Configurable log levels via args or environment variables.
- Option to control verbosity of third-party libraries.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any

# --- Attempt to import colorama ---
try:
    from colorama import Fore, Style, Back, init as colorama_init
    # Initialize colorama (autoreset=True ensures colors reset after each print)
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Define dummy color objects if colorama is not installed
    class DummyColor:
        def __getattr__(self, name: str) -> str: return "" # Return empty string
    Fore = DummyColor(); Back = DummyColor(); Style = DummyColor()
    COLORAMA_AVAILABLE = False
    print("Warning: 'colorama' library not found. Neon console logging disabled.", file=sys.stderr)
    print("         Install using: pip install colorama", file=sys.stderr)

# --- Custom Log Level ---
SUCCESS_LEVEL = 25 # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log method to the Logger instance."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

# Add the method to the Logger class dynamically
if not hasattr(logging.Logger, 'success'):
    logging.Logger.success = log_success # type: ignore[attr-defined]


# --- Neon Color Theme Mapping ---
LOG_LEVEL_COLORS: Dict[int, str] = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.BLUE,
    SUCCESS_LEVEL: Fore.MAGENTA,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
}

# --- Custom Formatter for Colored Console Output ---
class ColoredConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter that adds colors to console output based on log level,
    only if colorama is available and output is a TTY.
    """
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: str = '%', validate: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.use_colors = COLORAMA_AVAILABLE and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record and applies colors to the level name."""
        # Store original levelname before potential modification
        original_levelname = record.levelname
        color = LOG_LEVEL_COLORS.get(record.levelno, Fore.WHITE) # Default to white

        if self.use_colors:
            # Temporarily add color codes to the levelname for formatting
            record.levelname = f"{color}{original_levelname}{Style.RESET_ALL}"

        # Use the parent class's formatting method
        formatted_message = super().format(record)

        # Restore original levelname to prevent colored output in file logs etc.
        record.levelname = original_levelname

        return formatted_message


# --- Log Format Strings ---
# Include thread name for better context in concurrent applications
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s %(funcName)s:%(lineno)d] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create formatters
console_formatter = ColoredConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


# --- Main Setup Function ---
def setup_logger(
    logger_name: str = "AppLogger",
    log_file: Optional[str] = "app.log",
    console_level_str: str = "INFO", # Changed to string input
    file_level_str: str = "DEBUG", # Changed to string input
    log_rotation_bytes: int = 5 * 1024 * 1024, # 5 MB default max size
    log_backup_count: int = 5, # Keep 5 backup files
    propagate: bool = False,
    third_party_log_level_str: str = "WARNING" # Changed to string input
) -> logging.Logger:
    """
    Sets up and configures a logger instance with neon console, clean file output,
    optional rotation, and control over third-party library logging.

    Reads levels as strings and converts them internally.

    Args:
        logger_name: Name for the logger instance.
        log_file: Path to the log file. None disables file logging. Rotation enabled by default.
        console_level_str: Logging level for console output (e.g., "INFO").
        file_level_str: Logging level for file output (e.g., "DEBUG").
        log_rotation_bytes: Max size in bytes before rotating log file. 0 disables rotation.
        log_backup_count: Number of backup log files to keep. Ignored if rotation is disabled.
        propagate: Whether to propagate messages to the root logger (default False).
        third_party_log_level_str: Level for common noisy libraries (e.g., "WARNING").

    Returns:
        The configured logging.Logger instance.
    """
    func_name = "setup_logger" # For internal logging if needed

    # --- Convert string levels to logging constants ---
    try:
        console_level = logging.getLevelName(console_level_str.upper())
        file_level = logging.getLevelName(file_level_str.upper())
        third_party_log_level = logging.getLevelName(third_party_log_level_str.upper())

        if not isinstance(console_level, int):
            print(f"\033[93mWarning [{func_name}]: Invalid console log level string '{console_level_str}'. Using INFO.\033[0m", file=sys.stderr)
            console_level = logging.INFO
        if not isinstance(file_level, int):
            print(f"\033[93mWarning [{func_name}]: Invalid file log level string '{file_level_str}'. Using DEBUG.\033[0m", file=sys.stderr)
            file_level = logging.DEBUG
        if not isinstance(third_party_log_level, int):
             print(f"\033[93mWarning [{func_name}]: Invalid third-party log level string '{third_party_log_level_str}'. Using WARNING.\033[0m", file=sys.stderr)
             third_party_log_level = logging.WARNING

    except Exception as e:
        print(f"\033[91mError [{func_name}]: Failed converting log level strings: {e}. Using defaults.\033[0m", file=sys.stderr)
        console_level, file_level, third_party_log_level = logging.INFO, logging.DEBUG, logging.WARNING

    # --- Get Logger and Set Base Level/Propagation ---
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # Set logger to lowest level to capture all messages for handlers
    logger.propagate = propagate

    # --- Clear Existing Handlers (if re-configuring) ---
    if logger.hasHandlers():
        print(f"\033[94mInfo [{func_name}]: Logger '{logger_name}' already configured. Clearing handlers.\033[0m", file=sys.stderr)
        for handler in logger.handlers[:]: # Iterate a copy
            try:
                handler.close() # Close file handles etc.
                logger.removeHandler(handler)
            except Exception as e:
                 print(f"\033[93mWarning [{func_name}]: Error removing/closing handler: {e}\033[0m", file=sys.stderr)

    # --- Console Handler ---
    if console_level is not None and console_level >= 0:
        try:
            console_h = logging.StreamHandler(sys.stdout)
            console_h.setLevel(console_level)
            console_h.setFormatter(console_formatter) # Use the colored formatter
            logger.addHandler(console_h)
            print(f"\033[94m[{func_name}] Console logging active at level [{logging.getLevelName(console_level)}].\033[0m")
        except Exception as e:
             print(f"\033[91mError [{func_name}] setting up console handler: {e}\033[0m", file=sys.stderr)
    else:
        print(f"\033[94m[{func_name}] Console logging disabled.\033[0m")

    # --- File Handler (with optional rotation) ---
    if log_file:
        try:
            log_file_path = os.path.abspath(log_file) # Use absolute path
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                 os.makedirs(log_dir, exist_ok=True) # Ensure directory exists

            if log_rotation_bytes > 0 and log_backup_count >= 0:
                # Use Rotating File Handler
                file_h = logging.handlers.RotatingFileHandler(
                    log_file_path, # Use absolute path
                    maxBytes=log_rotation_bytes,
                    backupCount=log_backup_count,
                    encoding='utf-8'
                )
                log_type = "Rotating file"
                log_details = f"(Max: {log_rotation_bytes / 1024 / 1024:.1f} MB, Backups: {log_backup_count})"
            else:
                # Use basic File Handler (no rotation)
                file_h = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
                log_type = "Basic file"
                log_details = "(Rotation disabled)"

            file_h.setLevel(file_level)
            file_h.setFormatter(file_formatter) # Use the plain (non-colored) formatter
            logger.addHandler(file_h)
            print(f"\033[94m[{func_name}] {log_type} logging active at level [{logging.getLevelName(file_level)}] to '{log_file_path}' {log_details}.\033[0m")

        except IOError as e:
            print(f"\033[91mFATAL [{func_name}] Error configuring log file '{log_file}': {e}. File logging disabled.\033[0m", file=sys.stderr)
        except Exception as e:
             print(f"\033[91mError [{func_name}] Unexpected error setting up file logging: {e}. File logging disabled.\033[0m", file=sys.stderr)
    else:
        print(f"\033[94m[{func_name}] File logging disabled.\033[0m")

    # --- Configure Third-Party Log Levels ---
    if third_party_log_level is not None and third_party_log_level >= 0:
        noisy_libraries = ["ccxt", "urllib3", "requests", "asyncio", "websockets"] # Add others if needed
        print(f"\033[94m[{func_name}] Setting third-party library log level to [{logging.getLevelName(third_party_log_level)}].\033[0m")
        for lib_name in noisy_libraries:
            try:
                lib_logger = logging.getLogger(lib_name)
                if lib_logger: # Check if logger exists
                    lib_logger.setLevel(third_party_log_level)
                    lib_logger.propagate = False # Stop noisy logs from reaching our handlers
            except Exception as e_lib:
                 # Non-critical error
                 print(f"\033[93mWarning [{func_name}]: Could not set level for lib '{lib_name}': {e_lib}\033[0m", file=sys.stderr)
    else:
         print(f"\033[94m[{func_name}] Third-party library log level control disabled.\033[0m")

    # --- Log Test Messages ---
    # logger.debug("--- Logger Setup Complete (DEBUG Test) ---")
    # logger.info("--- Logger Setup Complete (INFO Test) ---")
    # logger.success("--- Logger Setup Complete (SUCCESS Test) ---")
    # logger.warning("--- Logger Setup Complete (WARNING Test) ---")
    # logger.error("--- Logger Setup Complete (ERROR Test) ---")
    # logger.critical("--- Logger Setup Complete (CRITICAL Test) ---")
    logger.info(f"--- Logger '{logger_name}' Setup Complete ---")

    # Cast to include the 'success' method for type hinting upstream
    return logger # type: ignore

# --- Example Usage ---
if __name__ == "__main__":
    print("-" * 60)
    print("--- Example Neon Logger v1.2 Usage ---")
    print("-" * 60)
    # Example: Set environment variables for testing overrides
    # os.environ["LOG_CONSOLE_LEVEL"] = "DEBUG"
    # os.environ["LOG_FILE_PATH"] = "test_override.log"

    # Basic setup
    logger_instance = setup_logger(
        logger_name="ExampleLogger",
        log_file="example_app.log",
        console_level_str="INFO", # Use string levels
        file_level_str="DEBUG",
        third_party_log_level_str="WARNING"
    )

    # Log messages at different levels
    logger_instance.debug("This is a detailed debug message (might only go to file).")
    logger_instance.info("This is an informational message.")
    logger_instance.success("Operation completed successfully!") # Custom level
    logger_instance.warning("This is a warning message.")
    logger_instance.error("An error occurred during processing.")
    try:
        1 / 0
    except ZeroDivisionError:
        logger_instance.critical("A critical error (division by zero) happened!", exc_info=True)

    # Test third-party level suppression (if ccxt installed)
    try:
        import ccxt
        ccxt_logger = logging.getLogger('ccxt')
        print(f"CCXT logger level: {logging.getLevelName(ccxt_logger.getEffectiveLevel())}")
        ccxt_logger.info("This CCXT INFO message should be suppressed by default.")
        ccxt_logger.warning("This CCXT WARNING message should appear.")
    except ImportError:
        print("CCXT not installed, skipping third-party logger test.")

    print(f"\nCheck console output and log files created ('example_app.log').")
    # Clean up env vars if set for test
    # os.environ.pop("LOG_CONSOLE_LEVEL", None)
    # os.environ.pop("LOG_FILE_PATH", None)

# --- END OF FILE neon_logger.py ---
EOF

# --- Create bybit_utils.py (Using provided enhanced version) ---
echo -e "${C_INFO} -> Generating bybit_utils.py (v1.1 enhanced)${C_RESET}"
cat << 'EOF' > bybit_utils.py
# --- START OF FILE bybit_utils.py ---

import ccxt
from decimal import Decimal, InvalidOperation
import time
import functools
import logging
import subprocess # For Termux API call
import random # For jitter
from typing import Optional, Any, Callable, TypeVar, Dict, List, Tuple, Union

try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore

# Assume logger is configured in the importing scope (e.g., strategy script)
# If not, create a basic placeholder
# Note: In this project structure, logger is set up in main.py, so this might run before config.
# It's better to get the logger by name if possible.
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Add a basic handler if none exist yet
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default to INFO if not configured
    logger.info(f"Placeholder logger initialized for {__name__}. Main logger setup expected.")


# Placeholder TypeVar for Config object (structure defined in importing script)
# Replace with actual import if possible, or define attributes needed by this module
try:
    from config_models import AppConfig # Try importing the main config
    ConfigPlaceholder = AppConfig # Use the real type hint
except ImportError:
    class ConfigPlaceholder: # type: ignore # Fallback definition
         # Define attributes expected by functions in this file
         RETRY_COUNT: int = 3
         RETRY_DELAY_SECONDS: float = 1.0
         ENABLE_SMS_ALERTS: bool = False
         SMS_RECIPIENT_NUMBER: Optional[str] = None
         SMS_TIMEOUT_SECONDS: int = 30
         # Add others if needed by analyze_order_book, etc.

# --- Utility Functions ---

def safe_decimal_conversion(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Convert various inputs to Decimal, returning default or None on failure."""
    if value is None: return default
    try:
        # Handle potential scientific notation strings
        if isinstance(value, str) and 'e' in value.lower():
            return Decimal(value)
        # Convert float to string first for precision
        if isinstance(value, float):
            value = str(value)
        d = Decimal(value)
        if d.is_nan() or d.is_infinite(): return default # Reject NaN/Inf
        return d
    except (ValueError, TypeError, InvalidOperation):
        # logger.warning(f"safe_decimal_conversion failed for value: {value}", exc_info=True) # Optional: log failures
        return default

def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str:
    """Format a price value according to the market's precision rules."""
    price_dec = safe_decimal_conversion(price)
    if price_dec is None: return "N/A"
    try:
        # Use CCXT's method if available
        return exchange.price_to_precision(symbol, float(price_dec))
    except (AttributeError, KeyError, TypeError, ValueError, ccxt.ExchangeError):
        # logger.warning(f"Market data/precision issue for '{symbol}' in format_price. Using fallback.", exc_info=True)
        # Fallback: format to a reasonable number of decimal places
        return f"{price_dec:.8f}"
    except Exception as e:
        logger.critical(f"Error formatting price '{price}' for {symbol}: {e}", exc_info=True)
        return "Error"

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str:
    """Format an amount value according to the market's precision rules."""
    amount_dec = safe_decimal_conversion(amount)
    if amount_dec is None: return "N/A"
    try:
        # Use CCXT's method if available
        return exchange.amount_to_precision(symbol, float(amount_dec))
    except (AttributeError, KeyError, TypeError, ValueError, ccxt.ExchangeError):
        # logger.warning(f"Market data/precision issue for '{symbol}' in format_amount. Using fallback.", exc_info=True)
        # Fallback: format to a reasonable number of decimal places
        return f"{amount_dec:.8f}"
    except Exception as e:
        logger.critical(f"Error formatting amount '{amount}' for {symbol}: {e}", exc_info=True)
        return "Error"

def format_order_id(order_id: Any) -> str:
    """Format an order ID for concise logging (shows last 6 digits)."""
    try:
        id_str = str(order_id).strip() if order_id else ""
        if not id_str: return 'UNKNOWN'
        return "..." + id_str[-6:] if len(id_str) > 6 else id_str
    except Exception as e:
        logger.error(f"Error formatting order ID {order_id}: {e}")
        return 'UNKNOWN'

def send_sms_alert(message: str, config: Optional[ConfigPlaceholder] = None) -> bool:
    """Send an SMS alert using Termux API."""
    if not config: logger.warning("SMS alert config missing."); return False

    # Use attributes directly from ConfigPlaceholder (or the real AppConfig)
    enabled = getattr(config, 'ENABLE_SMS_ALERTS', getattr(config.sms_config, 'enable_sms_alerts', False))
    if not enabled: return True # Return True if disabled, as no action failed

    recipient = getattr(config, 'SMS_RECIPIENT_NUMBER', getattr(config.sms_config, 'sms_recipient_number', None))
    if not recipient:
        logger.warning("SMS alerts enabled but no SMS_RECIPIENT_NUMBER configured.")
        return False

    use_termux = getattr(config, 'use_termux_api', getattr(config.sms_config, 'use_termux_api', False)) # Check if Termux is the method
    if not use_termux:
         logger.warning("SMS enabled but Termux method not selected/configured.")
         return False # Only Termux is implemented here

    timeout = getattr(config, 'SMS_TIMEOUT_SECONDS', getattr(config.sms_config, 'sms_timeout_seconds', 30))

    try:
        logger.info(f"Attempting to send SMS alert via Termux to {recipient}...")
        command = ["termux-sms-send", "-n", recipient, message]
        result = subprocess.run(command, timeout=timeout, check=True, capture_output=True, text=True)
        log_output = result.stdout.strip() if result.stdout else '(No output)'
        logger.info(f"{Fore.GREEN}SMS Alert Sent Successfully via Termux.{Style.RESET_ALL} Output: {log_output}")
        return True
    except FileNotFoundError:
        logger.error(f"{Fore.RED}Termux API command 'termux-sms-send' not found. Is Termux:API installed and configured?{Style.RESET_ALL}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}Termux SMS command timed out after {timeout} seconds.{Style.RESET_ALL}")
        return False
    except subprocess.CalledProcessError as e:
        stderr_log = e.stderr.strip() if e.stderr else '(No stderr)'
        logger.error(f"{Fore.RED}Termux SMS command failed with exit code {e.returncode}.{Style.RESET_ALL}")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Stderr: {stderr_log}")
        return False
    except Exception as e:
        logger.critical(f"{Fore.RED}Unexpected error sending SMS alert via Termux: {e}{Style.RESET_ALL}", exc_info=True)
        return False

# --- Retry Decorator Factory ---
T = TypeVar('T')

def retry_api_call(
    max_retries_override: Optional[int] = None,
    initial_delay_override: Optional[float] = None,
    handled_exceptions=(ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout),
    error_message_prefix: str = "API Call Failed"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory to retry synchronous API calls with configurable settings.
    Requires config object passed to decorated function (positional or kwarg `app_config`).
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Find config object (accept AppConfig or legacy Config)
            config_obj = kwargs.get('app_config') or kwargs.get('config')
            if not config_obj:
                config_obj = next((arg for arg in args if isinstance(arg, (AppConfig, ConfigPlaceholder))), None) # type: ignore

            if not config_obj:
                logger.error(f"{Fore.RED}No config object (AppConfig or compatible) found for retry_api_call in {func.__name__}{Style.RESET_ALL}")
                raise ValueError("Config object required for retry_api_call")

            # Extract retry settings from config (handle both AppConfig and legacy)
            if isinstance(config_obj, AppConfig):
                api_conf = config_obj.api_config
                sms_conf = config_obj.sms_config
                effective_max_retries = max_retries_override if max_retries_override is not None else api_conf.retry_count
                effective_base_delay = initial_delay_override if initial_delay_override is not None else api_conf.retry_delay_seconds
            else: # Assume legacy Config object
                effective_max_retries = max_retries_override if max_retries_override is not None else getattr(config_obj, 'RETRY_COUNT', 3)
                effective_base_delay = initial_delay_override if initial_delay_override is not None else getattr(config_obj, 'RETRY_DELAY_SECONDS', 1.0)
                sms_conf = config_obj # Pass the whole legacy config for SMS

            func_name = func.__name__
            last_exception = None # Store last exception for final raise

            for attempt in range(effective_max_retries + 1):
                try:
                    if attempt > 0: logger.debug(f"Retrying {func_name} (Attempt {attempt + 1}/{effective_max_retries + 1})")
                    return func(*args, **kwargs) # Execute the original synchronous function
                except handled_exceptions as e:
                    last_exception = e
                    if attempt >= effective_max_retries:
                        logger.error(f"{Fore.RED}{error_message_prefix}: Max retries ({effective_max_retries + 1}) reached for {func_name}. Last error: {type(e).__name__} - {e}{Style.RESET_ALL}")
                        # Use the potentially legacy config directly for SMS
                        send_sms_alert(f"{error_message_prefix}: Max retries for {func_name} ({type(e).__name__})", sms_conf)
                        break # Exit loop, will raise last_exception below

                    # Calculate delay with exponential backoff + jitter
                    delay = (effective_base_delay * (2 ** attempt)) + (effective_base_delay * random.uniform(0.1, 0.5))
                    log_level = logging.WARNING
                    log_color = Fore.YELLOW

                    if isinstance(e, ccxt.RateLimitExceeded):
                        log_color = Fore.YELLOW + Style.BRIGHT
                        logger.log(log_level, f"{log_color}Rate limit exceeded in {func_name}. Retry {attempt + 1}/{effective_max_retries + 1} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                    elif isinstance(e, (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout)):
                         log_level = logging.ERROR
                         log_color = Fore.RED
                         logger.log(log_level, f"{log_color}{type(e).__name__} in {func_name}. Retry {attempt + 1}/{effective_max_retries + 1} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                    else:
                         # Generic handled exception
                         logger.log(log_level, f"{log_color}{type(e).__name__} in {func_name}. Retry {attempt + 1}/{effective_max_retries + 1} after {delay:.2f}s: {e}{Style.RESET_ALL}")

                    time.sleep(delay) # Synchronous sleep

                except Exception as e:
                    logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected critical error in {func_name}: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
                    send_sms_alert(f"CRITICAL Error in {func_name}: {type(e).__name__}", sms_conf)
                    raise e # Re-raise unexpected exceptions immediately

            # If loop finished without returning, raise the last handled exception
            if last_exception:
                raise last_exception
            else: # Should not happen unless max_retries is negative, but safeguard
                 raise Exception(f"Failed to execute {func_name} after {effective_max_retries + 1} retries (unknown error)")

        return wrapper
    return decorator


# --- Order Book Analysis ---
# Note: This is synchronous. If used in an async context, run in executor.
@retry_api_call() # Use default retry settings from config
def analyze_order_book(
    exchange: ccxt.Exchange, # Expects synchronous exchange instance
    symbol: str,
    depth: int, # Analysis depth (levels from top)
    fetch_limit: int, # How many levels to fetch initially
    config: ConfigPlaceholder # Accepts legacy or AppConfig
) -> Dict[str, Optional[Decimal]]:
    """Fetches and analyzes the L2 order book (synchronous)."""
    func_name = "analyze_order_book"
    log_prefix = f"[{func_name}({symbol}, Depth:{depth}, Fetch:{fetch_limit})]"
    logger.debug(f"{log_prefix} Analyzing...")

    # Determine config type for accessing settings
    if isinstance(config, AppConfig):
        api_conf = config.api_config
        sms_conf = config.sms_config
    else: # Assume legacy config
        api_conf = config # Use the whole object, hoping attributes match
        sms_conf = config

    analysis_result = {
        'best_bid': None, 'best_ask': None, 'mid_price': None, 'spread': None,
        'spread_pct': None, 'bid_volume_depth': None, 'ask_volume_depth': None,
        'bid_ask_ratio_depth': None, 'timestamp': None
    }

    try:
        logger.debug(f"{log_prefix} Fetching order book data...")
        # Ensure fetch_limit is at least the analysis depth
        effective_fetch_limit = max(depth, fetch_limit)
        # Bybit V5 needs category for fetch_order_book
        category = market_cache.get_category(symbol) # Assumes market cache is populated elsewhere
        if not category: logger.warning(f"{log_prefix} Category unknown for {symbol}. Fetch may fail."); params = {}
        else: params = {'category': category}

        order_book = exchange.fetch_order_book(symbol, limit=effective_fetch_limit, params=params)

        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book:
            raise ValueError("Invalid order book structure received.")

        analysis_result['timestamp'] = order_book.get('timestamp') # Store timestamp if available

        bids_raw = order_book['bids']
        asks_raw = order_book['asks']
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            raise ValueError("Order book 'bids' or 'asks' data is not a list.")
        if not bids_raw or not asks_raw:
            logger.warning(f"{log_prefix} Order book side empty (Bids:{len(bids_raw)}, Asks:{len(asks_raw)}).")
            return analysis_result # Return defaults

        # Process bids and asks, converting to Decimal
        bids: List[Tuple[Decimal, Decimal]] = []
        asks: List[Tuple[Decimal, Decimal]] = []
        for p, a in bids_raw:
            price = safe_decimal_conversion(p); amount = safe_decimal_conversion(a)
            if price and amount and price > 0 and amount >= 0: bids.append((price, amount))
        for p, a in asks_raw:
            price = safe_decimal_conversion(p); amount = safe_decimal_conversion(a)
            if price and amount and price > 0 and amount >= 0: asks.append((price, amount))

        # Ensure lists are sorted correctly by CCXT (bids descending, asks ascending)
        # bids.sort(key=lambda x: x[0], reverse=True) # Optional verification
        # asks.sort(key=lambda x: x[0]) # Optional verification

        if not bids or not asks:
            logger.warning(f"{log_prefix} Order book validated bids/asks empty after conversion.")
            return analysis_result

        best_bid = bids[0][0]; best_ask = asks[0][0]
        analysis_result['best_bid'] = best_bid; analysis_result['best_ask'] = best_ask

        if best_bid >= best_ask:
            logger.error(f"{Fore.RED}{log_prefix} Order book crossed! Bid ({best_bid}) >= Ask ({best_ask}).{Style.RESET_ALL}")
            # Return crossed data for potential handling upstream
            return analysis_result

        analysis_result['mid_price'] = (best_bid + best_ask) / Decimal("2")
        analysis_result['spread'] = best_ask - best_bid
        analysis_result['spread_pct'] = (analysis_result['spread'] / analysis_result['mid_price']) * Decimal("100") if analysis_result['mid_price'] > 0 else Decimal("inf") # Use mid price for stability

        # Calculate volume within the specified depth
        bid_vol_depth = sum(b[1] for b in bids[:depth] if b[1] is not None)
        ask_vol_depth = sum(a[1] for a in asks[:depth] if a[1] is not None)
        analysis_result['bid_volume_depth'] = bid_vol_depth
        analysis_result['ask_volume_depth'] = ask_vol_depth
        analysis_result['bid_ask_ratio_depth'] = (bid_vol_depth / ask_vol_depth) if ask_vol_depth and ask_vol_depth > 0 else None

        ratio_str = f"{analysis_result['bid_ask_ratio_depth']:.2f}" if analysis_result['bid_ask_ratio_depth'] is not None else 'N/A'
        logger.debug(f"{log_prefix} Analysis OK: Spread={analysis_result['spread_pct']:.4f}%, "
                     f"Depth Ratio(d{depth})={ratio_str}")

        return analysis_result

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, TypeError) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Error fetching/analyzing: {e}{Style.RESET_ALL}")
        raise # Re-raise handled exceptions for the decorator
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}{log_prefix} Unexpected error analyzing order book: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"CRITICAL: OB Analysis failed for {symbol}", sms_conf)
        return analysis_result # Return default on critical failure

# --- END OF FILE bybit_utils.py ---
EOF

# --- Create bybit_helper_functions.py (Using provided enhanced v2.9) ---
# Note: This version is SYNCHRONOUS. If async is needed later, it requires significant changes.
echo -e "${C_INFO} -> Generating bybit_helper_functions.py (v2.9 sync enhanced)${C_RESET}"
cat << 'EOF' > bybit_helper_functions.py
# --- START OF FILE bybit_helper_functions.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bybit V5 CCXT Helper Functions (v2.9 - Add cancel/fetch order, Fix Ticker Validation)

This module provides a collection of robust, reusable, and enhanced **synchronous**
helper functions designed for interacting with the Bybit exchange (V5 API)
using the CCXT library.

**Note:** This version is synchronous. For asynchronous operations, use
`ccxt.async_support` and adapt these functions accordingly (e.g., using `await`).

Core Functionality Includes:
- Exchange Initialization: Securely sets up the ccxt.bybit exchange instance.
- Account Configuration: Set leverage, margin mode, position mode.
- Market Data Retrieval: Validated fetchers for tickers, OHLCV, order books, etc.
- Order Management: Place market, limit, stop orders; cancel orders; fetch orders.
- Position Management: Fetch positions, close positions.
- Balance & Margin: Fetch balances, calculate margin estimates.
- Utilities: Market validation.

Key Enhancements in v2.9:
- Added `cancel_order` and `fetch_order` helper functions.
- Fixed `ValueError` in `fetch_ticker_validated` when timestamp is missing.
- Improved exception message clarity in `fetch_ticker_validated`.
- Explicitly imports utilities from `bybit_utils`.

Dependencies:
- `logger`: Pre-configured `logging.Logger` object (from main script).
- `Config`: Configuration class/object (from main script or `config_models`).
- `bybit_utils.py`: Utility functions and retry decorator.
"""

# Standard Library Imports
import logging
import os
import sys
import time
import random # Used in fetch_ohlcv_paginated retry delay jitter
from decimal import Decimal, ROUND_HALF_UP, DivisionByZero, InvalidOperation, getcontext
from typing import Optional, Dict, List, Tuple, Any, Literal, Union, Callable, TypeVar

# Third-party Libraries
try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt", file=sys.stderr)
    sys.exit(1)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas library not found. OHLCV data will be list.", file=sys.stderr)
    pd = None # type: ignore
    PANDAS_AVAILABLE = False
try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    print("Warning: colorama library not found. Logs will not be colored.", file=sys.stderr)
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore
    COLORAMA_AVAILABLE = False

# --- Import Utilities from bybit_utils ---
try:
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert, retry_api_call,
        analyze_order_book # Ensure analyze_order_book is defined in bybit_utils
    )
    print("[bybit_helpers] Successfully imported utilities from bybit_utils.")
except ImportError as e:
    print(f"FATAL ERROR [bybit_helpers]: Failed to import required functions/decorator from bybit_utils.py: {e}", file=sys.stderr)
    print("Ensure bybit_utils.py is in the same directory or accessible via PYTHONPATH.", file=sys.stderr)
    sys.exit(1)
except NameError as e:
    print(f"FATAL ERROR [bybit_helpers]: A required name is not defined in bybit_utils.py: {e}", file=sys.stderr)
    sys.exit(1)

# Set Decimal context precision
getcontext().prec = 28

# --- Logger Placeholder ---
# Actual logger MUST be provided by importing script (e.g., main.py)
# This provides a fallback if the module is imported before logger setup.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d:%(funcName)s} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default level if not configured
    logger.info(f"Placeholder logger initialized for {__name__}. Main logger setup expected.")

# --- Config Placeholder ---
# Actual config object MUST be provided by importing script or passed to functions
try:
    from config_models import AppConfig # Use the main config model
    Config = AppConfig # Use AppConfig as the type hint
except ImportError:
     logger.error("Could not import AppConfig from config_models. Type hinting will use basic 'object'.")
     Config = object # Fallback type hint


# --- Helper Function Implementations ---

def _get_v5_category(market: Dict[str, Any]) -> Optional[Literal['linear', 'inverse', 'spot', 'option']]:
    """Internal helper to determine the Bybit V5 category from a market object."""
    func_name = "_get_v5_category"
    if not market: return None

    # Use the logic from bybit_helpers v3.5 (more robust)
    info = market.get('info', {})
    category_from_info = info.get('category')
    if category_from_info in ['linear', 'inverse', 'spot', 'option']: return category_from_info # type: ignore

    if market.get('spot', False): return 'spot'
    if market.get('option', False): return 'option'
    if market.get('linear', False): return 'linear'
    if market.get('inverse', False): return 'inverse'

    market_type = market.get('type')
    symbol = market.get('symbol', 'N/A')

    if market_type == 'spot': return 'spot'
    if market_type == 'option': return 'option'

    if market_type in ['swap', 'future']:
        contract_type = str(info.get('contractType', '')).lower()
        settle_coin = market.get('settle', '').upper()
        if contract_type == 'linear': return 'linear'
        if contract_type == 'inverse': return 'inverse'
        if settle_coin in ['USDT', 'USDC']: return 'linear'
        if settle_coin and settle_coin == market.get('base', '').upper(): return 'inverse'
        logger.debug(f"[{func_name}] Ambiguous derivative {symbol}. Assuming 'linear'.")
        return 'linear'

    logger.warning(f"[{func_name}] Could not determine V5 category for market: {symbol}, Type: {market_type}")
    return None

# Snippet 1 / Function 1: Initialize Bybit Exchange
@retry_api_call(max_retries_override=3, initial_delay_override=2.0, error_message_prefix="Exchange Init Failed")
def initialize_bybit(config: Config) -> Optional[ccxt.bybit]:
    """Initializes and validates the Bybit CCXT exchange instance using V5 API settings."""
    func_name = "initialize_bybit"
    api_conf = config.api_config # Access nested config
    strat_conf = config.strategy_config

    mode_str = 'Testnet' if api_conf.testnet_mode else 'Mainnet'
    logger.info(f"{Fore.BLUE}[{func_name}] Initializing Bybit V5 ({mode_str}, Sync)...{Style.RESET_ALL}")
    try:
        exchange_class = getattr(ccxt, api_conf.exchange_id)
        exchange = exchange_class({
            'apiKey': api_conf.api_key,
            'secret': api_conf.api_secret,
            'enableRateLimit': True,
            'options': {
                # 'defaultType': api_conf.expected_market_type, # Less critical for V5 if category used
                'adjustForTimeDifference': True,
                'recvWindow': api_conf.default_recv_window,
                'brokerId': f"PB_{strat_conf.name[:10].replace(' ','_')}" # Example broker ID
            }
        })
        if api_conf.testnet_mode:
            logger.info(f"[{func_name}] Enabling Bybit Sandbox (Testnet) mode.")
            exchange.set_sandbox_mode(True)

        logger.debug(f"[{func_name}] Loading markets...")
        exchange.load_markets(reload=True) # Synchronous load
        if not exchange.markets: raise ccxt.ExchangeError(f"[{func_name}] Failed to load markets.")
        logger.debug(f"[{func_name}] Markets loaded successfully ({len(exchange.markets)} symbols).")

        # --- Authentication Check ---
        if api_conf.api_key and api_conf.api_secret:
            logger.debug(f"[{func_name}] Performing initial balance fetch for validation...")
            # Use fetch_usdt_balance which specifies UNIFIED account
            balance_info = fetch_usdt_balance(exchange, config) # Pass the main config
            if balance_info is None:
                 # Error logged by fetch_usdt_balance, raise specific error here
                 raise ccxt.AuthenticationError("Initial balance check failed. Verify API keys and permissions.")
            logger.info(f"[{func_name}] Initial balance check successful.")
        else:
             logger.warning(f"{Fore.YELLOW}[{func_name}] API keys not provided. Skipping auth check.{Style.RESET_ALL}")


        # --- Optional: Set initial margin mode for the primary symbol ---
        try:
            market = exchange.market(api_conf.symbol); category = _get_v5_category(market)
            if category and category in ['linear', 'inverse']:
                logger.debug(f"[{func_name}] Attempting to set initial margin mode '{strat_conf.default_margin_mode}' for {api_conf.symbol} (Cat: {category})...")
                # Note: set_margin_mode might require specific account types or permissions
                exchange.set_margin_mode(
                     marginMode=strat_conf.default_margin_mode,
                     symbol=api_conf.symbol,
                     params={'category': category, 'leverage': strat_conf.leverage} # Can set leverage here too
                )
                logger.info(f"[{func_name}] Initial margin mode potentially set to '{strat_conf.default_margin_mode}' for {api_conf.symbol}.")
            else:
                logger.warning(f"[{func_name}] Cannot determine contract category for {api_conf.symbol}. Skipping initial margin mode set.")
        except (ccxt.NotSupported, ccxt.ExchangeError, ccxt.ArgumentsRequired, ccxt.BadSymbol) as e_margin:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Could not set initial margin mode for {api_conf.symbol}: {e_margin}. Verify account settings/permissions.{Style.RESET_ALL}")

        logger.success(f"{Fore.GREEN}[{func_name}] Bybit exchange initialized successfully. Testnet: {api_conf.testnet_mode}.{Style.RESET_ALL}")
        return exchange

    except (ccxt.AuthenticationError, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
        logger.error(f"{Fore.RED}[{func_name}] Initialization attempt failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        # Decorator handles retries for NetworkError etc., but AuthError is usually fatal here.
        if isinstance(e, ccxt.AuthenticationError):
             send_sms_alert(f"[BybitHelper] CRITICAL: Bybit Auth failed! {type(e).__name__}", config.sms_config)
        raise # Re-raise to be caught by caller or retry decorator
    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error during Bybit initialization: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[BybitHelper] CRITICAL: Bybit init failed! Unexpected: {type(e).__name__}", config.sms_config)
        return None # Return None on unexpected critical failure

# Snippet 2 / Function 2: Set Leverage
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def set_leverage(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """Sets the leverage for a specific symbol on Bybit V5 (Linear/Inverse)."""
    func_name = "set_leverage"; log_prefix = f"[{func_name}({symbol} -> {leverage}x)]"
    logger.info(f"{Fore.CYAN}{log_prefix} Setting leverage...{Style.RESET_ALL}")
    if leverage <= 0: logger.error(f"{Fore.RED}{log_prefix} Leverage must be positive: {leverage}{Style.RESET_ALL}"); return False
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}{log_prefix} Invalid market type for leverage: {symbol} ({category}).{Style.RESET_ALL}"); return False

        # Basic leverage validation against market limits (if available)
        try:
            limits = market.get('limits', {}).get('leverage', {})
            max_lev_str = limits.get('max'); min_lev_str = limits.get('min', '1')
            max_lev = int(safe_decimal_conversion(max_lev_str, Decimal('100'))) # Default max 100 if missing
            min_lev = int(safe_decimal_conversion(min_lev_str, Decimal('1'))) # Default min 1
            if not (min_lev <= leverage <= max_lev):
                 logger.error(f"{Fore.RED}{log_prefix} Invalid leverage {leverage}x. Allowed: {min_lev}x - {max_lev}x.{Style.RESET_ALL}")
                 return False
        except Exception as e_lim: logger.warning(f"{Fore.YELLOW}{log_prefix} Could not validate leverage limits: {e_lim}. Proceeding.{Style.RESET_ALL}")

        params = {'category': category, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
        logger.debug(f"{log_prefix} Calling exchange.set_leverage with params={params}")
        # CCXT's set_leverage for Bybit V5 calls POST /v5/position/set-leverage
        exchange.set_leverage(leverage, symbol, params=params)
        logger.success(f"{Fore.GREEN}{log_prefix} Leverage set/confirmed to {leverage}x (Category: {category}).{Style.RESET_ALL}"); return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower(); err_code = getattr(e, 'code', None)
        # Bybit V5 code 110043: leverage not modified
        if err_code == 110043 or "leverage not modified" in err_str:
            logger.info(f"{Fore.CYAN}{log_prefix} Leverage for {symbol} is already set to {leverage}x.{Style.RESET_ALL}"); return True
        else: logger.error(f"{Fore.RED}{log_prefix} ExchangeError setting leverage: {e}{Style.RESET_ALL}"); return False # Don't raise, return False
    except (ccxt.NetworkError, ccxt.AuthenticationError, ccxt.BadSymbol) as e:
        logger.error(f"{Fore.RED}{log_prefix} API/Symbol error setting leverage: {e}{Style.RESET_ALL}")
        if isinstance(e, (ccxt.NetworkError)): raise e # Allow retry for network errors
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error setting leverage: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{symbol.split('/')[0]}] ERROR: Failed set leverage {leverage}x (Unexpected)", config.sms_config)
        return False

# Snippet 3 / Function 3: Fetch USDT Balance (V5 UNIFIED)
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def fetch_usdt_balance(exchange: ccxt.bybit, config: Config) -> Optional[Tuple[Decimal, Decimal]]:
    """Fetches the USDT balance (Total Equity and Available Balance) using Bybit V5 UNIFIED account logic."""
    func_name = "fetch_usdt_balance"; log_prefix = f"[{func_name}]"
    usdt_symbol = config.api_config.usdt_symbol
    account_type_target = 'UNIFIED' # Hardcode for UTA

    logger.debug(f"{log_prefix} Fetching {account_type_target} account balance ({usdt_symbol})...")
    try:
        # V5 requires specifying accountType
        balance_data = exchange.fetch_balance(params={'accountType': account_type_target})
        # logger.debug(f"Raw balance response: {balance_data}") # Debugging

        # Parse V5 structure: info -> result -> list -> account dict -> coin list
        info = balance_data.get('info', {}); result_list = info.get('result', {}).get('list', [])
        equity, available = None, None

        if not result_list: logger.warning(f"{log_prefix} Balance response list empty or missing."); return None

        unified_info = next((acc for acc in result_list if acc.get('accountType') == account_type_target), None)
        if not unified_info: logger.warning(f"{log_prefix} Account type '{account_type_target}' not found in response."); return None

        equity = safe_decimal_conversion(unified_info.get('totalEquity'), context="Total Equity")
        if equity is None: logger.warning(f"{log_prefix} Failed to parse total equity. Assuming 0.")

        coin_list = unified_info.get('coin', [])
        usdt_info = next((c for c in coin_list if c.get('coin') == usdt_symbol), None)
        if usdt_info:
            # Prioritize 'availableToWithdraw', fallback 'availableBalance'
            avail_str = usdt_info.get('availableToWithdraw') or usdt_info.get('availableBalance')
            available = safe_decimal_conversion(avail_str, context=f"{usdt_symbol} Available Balance")
        else: logger.warning(f"{log_prefix} {usdt_symbol} details not found in UNIFIED coin list.")

        if available is None: logger.warning(f"{log_prefix} Failed to parse available {usdt_symbol} balance. Assuming 0.")

        final_equity = max(Decimal("0.0"), equity or Decimal("0.0"))
        final_available = max(Decimal("0.0"), available or Decimal("0.0"))

        logger.info(f"{Fore.GREEN}{log_prefix} OK - Equity: {final_equity:.4f}, Available {usdt_symbol}: {final_available:.4f}{Style.RESET_ALL}")
        return final_equity, final_available

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Error fetching/parsing balance: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Unexpected error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert("[BybitHelper] CRITICAL: Failed fetch USDT balance!", config.sms_config)
        return None

# Snippet 4 / Function 4: Place Market Order with Slippage Check
@retry_api_call(max_retries_override=1, initial_delay_override=0) # Retry only once for market orders
def place_market_order_slippage_check(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal,
    config: Config, max_slippage_pct_override: Optional[Decimal] = None,
    is_reduce_only: bool = False, client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """Places a market order on Bybit V5 after checking the current spread against a slippage threshold."""
    func_name = "place_market_order_slippage_check"; market_base = symbol.split('/')[0]; action = "CLOSE" if is_reduce_only else "ENTRY"; log_prefix = f"[{func_name}({action} {side.upper()})]"
    api_conf = config.api_config
    effective_max_slippage = max_slippage_pct_override if max_slippage_pct_override is not None else api_conf.default_slippage_pct
    logger.info(f"{Fore.BLUE}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}. Max Slippage: {effective_max_slippage:.4%}, ReduceOnly: {is_reduce_only}{Style.RESET_ALL}")

    if amount <= api_conf.position_qty_epsilon: logger.error(f"{Fore.RED}{log_prefix}: Amount is zero or negative ({amount}). Aborting.{Style.RESET_ALL}"); return None

    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category for {symbol}. Aborting.{Style.RESET_ALL}"); return None

        # --- Slippage Check ---
        if effective_max_slippage > 0:
            logger.debug(f"{log_prefix} Performing pre-order slippage check (Depth: {api_conf.shallow_ob_fetch_depth})...")
            # Use the synchronous analyze_order_book utility
            ob_analysis = analyze_order_book(exchange, symbol, api_conf.shallow_ob_fetch_depth, api_conf.order_book_fetch_limit, config)
            best_ask, best_bid = ob_analysis.get("best_ask"), ob_analysis.get("best_bid")
            if best_bid and best_ask and best_bid > Decimal("0"):
                 mid_price = (best_ask + best_bid) / 2
                 spread_pct = ((best_ask - best_bid) / mid_price) * 100 if mid_price > 0 else Decimal('inf')
                 logger.debug(f"{log_prefix} Current OB: Bid={format_price(exchange, symbol, best_bid)}, Ask={format_price(exchange, symbol, best_ask)}, Spread={spread_pct:.4%}")
                 if spread_pct > effective_max_slippage:
                     logger.error(f"{Fore.RED}{log_prefix}: Aborted due to high spread {spread_pct:.4%} > Max {effective_max_slippage:.4%}.{Style.RESET_ALL}")
                     send_sms_alert(f"[{market_base}] ORDER ABORT ({side.upper()}): High Spread {spread_pct:.4%}", config.sms_config)
                     return None
            else: logger.warning(f"{Fore.YELLOW}{log_prefix}: Could not get valid OB data for slippage check. Proceeding cautiously.{Style.RESET_ALL}")
        else: logger.debug(f"{log_prefix} Slippage check skipped.")

        # --- Prepare and Place Order ---
        amount_str = format_amount(exchange, symbol, amount)
        if amount_str is None or amount_str == "Error": raise ValueError("Failed to format amount.")
        amount_float = float(amount_str)

        params: Dict[str, Any] = {'category': category}
        if is_reduce_only: params['reduceOnly'] = True
        if client_order_id:
            max_coid_len = 36; original_len = len(client_order_id); valid_coid = client_order_id[:max_coid_len]
            params['clientOrderId'] = valid_coid
            if len(valid_coid) < original_len: logger.warning(f"{log_prefix} Client OID truncated: '{valid_coid}' (Orig len: {original_len})")

        bg = Back.GREEN if side == api_conf.side_buy else Back.RED; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}*** PLACING MARKET {side.upper()} {'REDUCE' if is_reduce_only else 'ENTRY'}: {amount_str} {symbol} (Params: {params}) ***{Style.RESET_ALL}")

        order = exchange.create_market_order(symbol, side, amount_float, params=params)

        order_id = order.get('id'); client_oid_resp = order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = order.get('status', '?')
        filled_qty = safe_decimal_conversion(order.get('filled', '0.0')); avg_price = safe_decimal_conversion(order.get('average'))
        cost = safe_decimal_conversion(order.get('cost'))
        fee = order.get('fee', {}) # Fee info might be nested

        logger.success(f"{Fore.GREEN}{log_prefix}: Submitted OK. ID: {format_order_id(order_id)}, ClientOID: {client_oid_resp}, Status: {status}, Filled Qty: {format_amount(exchange, symbol, filled_qty)}, Avg Px: {format_price(exchange, symbol, avg_price)}, Cost: {cost:.4f}, Fee: {fee}{Style.RESET_ALL}")
        return order

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"{Fore.RED}{log_prefix}: API Error: {type(e).__name__} - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): {type(e).__name__}", config.sms_config)
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): Unexpected {type(e).__name__}.", config.sms_config)
        return None

# Snippet 5 / Function 5: Cancel All Open Orders
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def cancel_all_orders(exchange: ccxt.bybit, symbol: str, config: Config, reason: str = "Cleanup", order_filter: Optional[Literal['Order', 'StopOrder', 'tpslOrder']] = None) -> bool:
    """Cancels all open orders matching a filter for a specific symbol on Bybit V5."""
    func_name = "cancel_all_orders"; market_base = symbol.split('/')[0]; log_prefix = f"[{func_name}({symbol}, Filter:{order_filter or 'All'}, Reason:{reason})]"
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting cancellation...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category. Aborting.{Style.RESET_ALL}"); return False

        # V5 cancelAllOrders uses category, symbol (optional), settleCoin (optional), orderFilter (optional)
        params = {'category': category}
        if order_filter: params['orderFilter'] = order_filter
        # if symbol: params['symbol'] = market['id'] # Can specify symbol too

        logger.debug(f"{log_prefix} Calling cancelAllOrders with params: {params}")
        response = exchange.cancel_all_orders(symbol, params=params) # Pass symbol here too
        logger.debug(f"{log_prefix} Raw response: {response}")

        # --- Parse V5 Response ---
        # Response structure: { retCode: 0, retMsg: 'OK', result: { list: [ { orderId: '...', clientOrderId: '...' }, ... ], success: '1'/'0' }, ... }
        result_data = response.get('result', {})
        cancelled_list = result_data.get('list', [])
        success_flag = result_data.get('success') # Might indicate overall success

        if response.get('retCode') == 0:
             if cancelled_list:
                 logger.success(f"{Fore.GREEN}{log_prefix} Cancelled {len(cancelled_list)} orders successfully.{Style.RESET_ALL}")
                 for item in cancelled_list: logger.debug(f"  - Cancelled ID: {format_order_id(item.get('orderId'))}, ClientOID: {item.get('clientOrderId')}")
                 return True
             elif success_flag == '1': # Success flag but empty list
                 logger.info(f"{Fore.CYAN}{log_prefix} No open orders found matching filter to cancel (Success flag received).{Style.RESET_ALL}")
                 return True
             else: # retCode 0 but no list and no clear success flag
                 logger.warning(f"{Fore.YELLOW}{log_prefix} cancelAllOrders returned success code (0) but no order list. Assuming no matching orders.{Style.RESET_ALL}")
                 return True
        else:
             # Handle specific error codes if needed
             # e.g., Code 10001: Parameter error (might indicate bad filter)
             # e.g., Code 10004: No orders found (sometimes returned as error)
             ret_msg = response.get('retMsg', 'Unknown Error')
             if "order not found" in ret_msg.lower() or response.get('retCode') == 10004:
                  logger.info(f"{Fore.CYAN}{log_prefix} No open orders found matching filter to cancel (Error code received).{Style.RESET_ALL}")
                  return True
             else:
                  logger.error(f"{Fore.RED}{log_prefix} Failed. Code: {response.get('retCode')}, Msg: {ret_msg}{Style.RESET_ALL}")
                  return False

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.error(f"{Fore.RED}{log_prefix} API error during cancel all: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error during cancel all: {e}{Style.RESET_ALL}", exc_info=True)
        return False

# --- Added cancel_order function ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def cancel_order(exchange: ccxt.bybit, symbol: str, order_id: str, config: Config, order_filter: Optional[Literal['Order', 'StopOrder', 'tpslOrder']] = None) -> bool:
    """Cancels a single specific order by ID."""
    func_name = "cancel_order"; log_prefix = f"[{func_name}({symbol}, ID:{format_order_id(order_id)}, Filter:{order_filter})]"
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting cancellation...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category. Aborting.{Style.RESET_ALL}"); return False

        params = {'category': category}
        # Bybit V5 cancelOrder might require orderFilter for Stop/TP/SL orders
        if order_filter: params['orderFilter'] = order_filter

        logger.debug(f"{log_prefix} Calling exchange.cancel_order with ID={order_id}, Symbol={symbol}, Params={params}")
        response = exchange.cancel_order(order_id, symbol, params=params)
        logger.debug(f"{log_prefix} Raw response: {response}")

        # Check response, CCXT might raise OrderNotFound on failure
        # If no exception, assume success (CCXT often normalizes this)
        logger.success(f"{Fore.GREEN}{log_prefix} Successfully cancelled order.{Style.RESET_ALL}")
        return True

    except ccxt.OrderNotFound as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Order already gone or not found: {e}{Style.RESET_ALL}")
        return True # Treat as success
    except ccxt.InvalidOrder as e: # e.g., trying to cancel a filled order
         logger.warning(f"{Fore.YELLOW}{log_prefix} Invalid order state for cancellation (likely filled/rejected): {e}{Style.RESET_ALL}")
         return True # Treat as success (already closed/gone)
    except ccxt.ExchangeError as e:
         # Check specific codes if needed
         # e.g., 110001: Order does not exist
         err_code = getattr(e, 'code', None)
         if err_code == 110001:
             logger.warning(f"{Fore.YELLOW}{log_prefix} Order not found (via ExchangeError 110001).{Style.RESET_ALL}")
             return True # Treat as success
         logger.error(f"{Fore.RED}{log_prefix} API error cancelling: {type(e).__name__} - {e}{Style.RESET_ALL}")
         return False # Don't raise, return failure
    except (ccxt.NetworkError) as e:
        logger.error(f"{Fore.RED}{log_prefix} Network error cancelling: {e}{Style.RESET_ALL}")
        raise e # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error cancelling: {e}{Style.RESET_ALL}", exc_info=True)
        return False

# --- Added fetch_order function ---
@retry_api_call(max_retries_override=3, initial_delay_override=0.5)
def fetch_order(exchange: ccxt.bybit, symbol: str, order_id: str, config: Config, order_filter: Optional[Literal['Order', 'StopOrder', 'tpslOrder']] = None) -> Optional[Dict]:
    """Fetches details for a single specific order by ID."""
    func_name = "fetch_order"; log_prefix = f"[{func_name}({symbol}, ID:{format_order_id(order_id)}, Filter:{order_filter})]"
    logger.debug(f"{log_prefix} Attempting fetch...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"); return None

        params = {'category': category}
        # Bybit V5 fetchOrder might need orderFilter for Stop/TP/SL orders
        if order_filter: params['orderFilter'] = order_filter

        logger.debug(f"{log_prefix} Calling exchange.fetch_order with ID={order_id}, Symbol={symbol}, Params={params}")
        order_data = exchange.fetch_order(order_id, symbol, params=params)

        if order_data:
            logger.debug(f"{log_prefix} Order data fetched. Status: {order_data.get('status')}")
            return order_data
        else:
            # CCXT fetch_order usually raises OrderNotFound, so this case is less likely
            logger.warning(f"{Fore.YELLOW}{log_prefix} fetch_order returned no data (but no exception). Order likely not found.{Style.RESET_ALL}")
            return None

    except ccxt.OrderNotFound as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Order not found: {e}{Style.RESET_ALL}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        err_code = getattr(e, 'code', None)
        # Treat specific error codes as OrderNotFound
        if err_code == 110001 or "order does not exist" in str(e).lower():
             logger.warning(f"{Fore.YELLOW}{log_prefix} Order not found (via ExchangeError).{Style.RESET_ALL}")
             return None
        logger.error(f"{Fore.RED}{log_prefix} API error fetching: {type(e).__name__} - {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- fetch_ohlcv_paginated (Synchronous version) ---
def fetch_ohlcv_paginated(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    config: Config,
    since: Optional[int] = None,
    limit_per_req: int = 1000, # Bybit V5 max limit is 1000
    max_total_candles: Optional[int] = None,
) -> Optional[Union[pd.DataFrame, List[list]]]:
    """
    Fetches historical OHLCV data for a symbol using pagination (synchronous).
    """
    func_name = "fetch_ohlcv_paginated"
    log_prefix = f"[{func_name}({symbol}, {timeframe})]"

    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}{log_prefix} Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None

    try:
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        if limit_per_req > 1000:
            logger.warning(f"{log_prefix} Requested limit_per_req ({limit_per_req}) > max (1000). Clamping.")
            limit_per_req = 1000

        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.warning(f"{Fore.YELLOW}{log_prefix} Could not determine category. Assuming 'linear'.{Style.RESET_ALL}")
            category = 'linear' # Default assumption

        params = {'category': category}

        since_str = pd.to_datetime(since, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M') if since else 'Recent'
        limit_str = str(max_total_candles) if max_total_candles else 'All'
        logger.info(f"{Fore.BLUE}{log_prefix} Fetching... Since: {since_str}, Target Limit: {limit_str}{Style.RESET_ALL}")

        all_candles: List[list] = []
        current_since = since
        request_count = 0
        max_requests = float('inf')
        if max_total_candles:
            max_requests = math.ceil(max_total_candles / limit_per_req)

        retry_conf = config.api_config # Get retry settings from config
        retry_delay = retry_conf.retry_delay_seconds
        max_retries = retry_conf.retry_count

        while request_count < max_requests:
            if max_total_candles and len(all_candles) >= max_total_candles:
                logger.info(f"{log_prefix} Reached target limit ({max_total_candles}). Fetch complete.")
                break

            request_count += 1
            fetch_limit = limit_per_req
            if max_total_candles:
                remaining_needed = max_total_candles - len(all_candles)
                fetch_limit = min(limit_per_req, remaining_needed)
                if fetch_limit <= 0: break # Already have enough

            logger.debug(f"{log_prefix} Fetch Chunk #{request_count}: Since={current_since}, Limit={fetch_limit}, Params={params}")

            candles_chunk: Optional[List[list]] = None
            last_fetch_error: Optional[Exception] = None

            # Internal retry loop for fetching this specific chunk
            for attempt in range(max_retries + 1):
                try:
                    candles_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=fetch_limit, params=params)
                    last_fetch_error = None; break # Success
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
                    last_fetch_error = e
                    if attempt >= max_retries: break # Max retries reached for this chunk
                    current_delay = retry_delay * (2 ** attempt) * random.uniform(0.8, 1.2)
                    logger.warning(f"{Fore.YELLOW}{log_prefix} API Error chunk #{request_count} (Try {attempt + 1}/{max_retries + 1}): {e}. Retrying in {current_delay:.2f}s...{Style.RESET_ALL}")
                    time.sleep(current_delay)
                except ccxt.ExchangeError as e: last_fetch_error = e; logger.error(f"{Fore.RED}{log_prefix} ExchangeError chunk #{request_count}: {e}. Aborting chunk.{Style.RESET_ALL}"); break
                except Exception as e: last_fetch_error = e; logger.error(f"{log_prefix} Unexpected fetch chunk #{request_count} err: {e}", exc_info=True); break

            if last_fetch_error:
                logger.error(f"{Fore.RED}{log_prefix} Failed to fetch chunk #{request_count} after {max_retries + 1} attempts. Last Error: {last_fetch_error}{Style.RESET_ALL}")
                logger.warning(f"{log_prefix} Returning potentially incomplete data ({len(all_candles)} candles) due to fetch failure.")
                break # Stop pagination

            if not candles_chunk: logger.debug(f"{log_prefix} No more candles returned (Chunk #{request_count})."); break

            # Filter duplicates if necessary
            if all_candles and candles_chunk[0][0] <= all_candles[-1][0]:
                logger.debug(f"{log_prefix} Overlap detected chunk #{request_count}. Filtering.")
                first_new_ts = all_candles[-1][0] + 1
                candles_chunk = [c for c in candles_chunk if c[0] >= first_new_ts]
                if not candles_chunk: logger.debug(f"{log_prefix} Entire chunk was overlap/duplicate."); continue # Skip to next fetch if needed

            num_fetched = len(candles_chunk)
            logger.debug(f"{log_prefix} Fetched {num_fetched} new candles (Chunk #{request_count}). Total: {len(all_candles) + num_fetched}")
            all_candles.extend(candles_chunk)

            if num_fetched < fetch_limit: logger.debug(f"{log_prefix} Received fewer candles than requested. End of data likely reached."); break

            # Update 'since' for the next request based on the timestamp of the *last* candle received
            current_since = candles_chunk[-1][0] + 1 # Request starting *after* the last received timestamp

            # Add a small delay based on rate limit
            time.sleep(max(0.05, 1.0 / (exchange.rateLimit if exchange.rateLimit and exchange.rateLimit > 0 else 10)))

        # Process final list
        return _process_ohlcv_list(all_candles, func_name, symbol, timeframe, max_total_candles)

    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
         logger.error(f"{Fore.RED}{log_prefix} Initial setup error for OHLCV fetch: {e}{Style.RESET_ALL}")
         return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Unexpected critical error during OHLCV pagination setup: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- _process_ohlcv_list (Helper for fetch_ohlcv_paginated) ---
def _process_ohlcv_list(
    candle_list: List[list], parent_func_name: str, symbol: str, timeframe: str, max_candles: Optional[int] = None
) -> Optional[Union[pd.DataFrame, List[list]]]:
    """Internal helper to convert OHLCV list to validated pandas DataFrame or return list."""
    func_name = f"{parent_func_name}._process_ohlcv_list"
    log_prefix = f"[{func_name}({symbol}, {timeframe})]"

    if not candle_list:
        logger.warning(f"{Fore.YELLOW}{log_prefix} No candles collected. Returning empty.{Style.RESET_ALL}")
        return pd.DataFrame() if PANDAS_AVAILABLE else []

    logger.debug(f"{log_prefix} Processing {len(candle_list)} raw candles...")
    try:
        # Sort and remove duplicates first
        candle_list.sort(key=lambda x: x[0])
        unique_candles_dict = {c[0]: c for c in candle_list}
        unique_candles = list(unique_candles_dict.values())
        if len(unique_candles) < len(candle_list):
             logger.debug(f"{log_prefix} Removed {len(candle_list) - len(unique_candles)} duplicate timestamps.")

        # Trim to max_candles if specified
        if max_candles and len(unique_candles) > max_candles:
             logger.debug(f"{log_prefix} Trimming final list to {max_candles} candles.")
             unique_candles = unique_candles[-max_candles:]

        if not PANDAS_AVAILABLE:
             logger.info(f"{Fore.GREEN}{log_prefix} Processed {len(unique_candles)} unique candles (returning list).{Style.RESET_ALL}")
             return unique_candles

        # Process into DataFrame
        df = pd.DataFrame(unique_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['datetime'], inplace=True) # Drop rows where timestamp conversion failed
        if df.empty: raise ValueError("All timestamp conversions failed or list was empty.")

        df.set_index('datetime', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle NaNs (optional: forward fill or drop)
        nan_counts = df[numeric_cols].isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(f"{Fore.YELLOW}{log_prefix} Found {total_nans} NaNs in numeric columns. Forward filling... (Counts: {nan_counts[nan_counts > 0].to_dict()}){Style.RESET_ALL}")
            df[numeric_cols] = df[numeric_cols].ffill()
            df.dropna(subset=numeric_cols, inplace=True) # Drop any remaining NaNs at the start

        if df.empty: logger.error(f"{Fore.RED}{log_prefix} Processed DataFrame is empty after cleaning.{Style.RESET_ALL}")

        logger.success(f"{Fore.GREEN}{log_prefix} Processed {len(df)} valid candles into DataFrame.{Style.RESET_ALL}")
        return df

    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Error processing OHLCV list: {e}{Style.RESET_ALL}", exc_info=True)
        # Fallback to returning the unique candle list if DataFrame processing fails
        return unique_candles if 'unique_candles' in locals() else candle_list


# --- place_limit_order_tif ---
@retry_api_call(max_retries_override=1, initial_delay_override=0) # Typically don't retry limit orders unless network error
def place_limit_order_tif(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, price: Decimal, config: Config,
    time_in_force: Literal['GTC', 'IOC', 'FOK', 'PostOnly'] = 'GTC', # Use Literal Type
    is_reduce_only: bool = False, is_post_only: bool = False, client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """Places a limit order on Bybit V5 with options for Time-In-Force, Post-Only, and Reduce-Only."""
    func_name = "place_limit_order_tif"; log_prefix = f"[{func_name}({side.upper()})]"
    api_conf = config.api_config
    logger.info(f"{Fore.BLUE}{log_prefix}: Init {format_amount(exchange, symbol, amount)} @ {format_price(exchange, symbol, price)} (TIF:{time_in_force}, Reduce:{is_reduce_only}, Post:{is_post_only})...{Style.RESET_ALL}")

    if amount <= api_conf.position_qty_epsilon or price <= Decimal("0"): logger.error(f"{Fore.RED}{log_prefix}: Invalid amount/price.{Style.RESET_ALL}"); return None

    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"); return None

        amount_str = format_amount(exchange, symbol, amount); price_str = format_price(exchange, symbol, price)
        if any(v is None or v == "Error" for v in [amount_str, price_str]): raise ValueError("Invalid amount/price formatting.")
        amount_float = float(amount_str); price_float = float(price_str)

        params: Dict[str, Any] = {'category': category}
        # Handle TIF and PostOnly flags correctly
        if time_in_force == 'PostOnly':
             params['postOnly'] = True
             params['timeInForce'] = 'GTC' # PostOnly is a flag, TIF is usually GTC
        elif time_in_force in ['GTC', 'IOC', 'FOK']:
             params['timeInForce'] = time_in_force
             if is_post_only: params['postOnly'] = True # Allow separate postOnly flag
        else:
             logger.warning(f"[{func_name}] Unsupported TIF '{time_in_force}'. Using GTC."); params['timeInForce'] = 'GTC'

        if is_reduce_only: params['reduceOnly'] = True

        if client_order_id:
            max_coid_len = 36; original_len = len(client_order_id); valid_coid = client_order_id[:max_coid_len]
            params['clientOrderId'] = valid_coid
            if len(valid_coid) < original_len: logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}' (Orig len: {original_len})")

        logger.info(f"{Fore.CYAN}{log_prefix}: Placing -> Amt:{amount_float}, Px:{price_float}, Params:{params}{Style.RESET_ALL}")
        order = exchange.create_limit_order(symbol, side, amount_float, price_float, params=params)

        order_id = order.get('id'); client_oid_resp = order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = order.get('status', '?'); effective_tif = order.get('timeInForce', params.get('timeInForce', '?')); is_post_only_resp = order.get('postOnly', params.get('postOnly', False))
        logger.success(f"{Fore.GREEN}{log_prefix}: Limit order placed. ID:{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, TIF:{effective_tif}, Post:{is_post_only_resp}{Style.RESET_ALL}")
        return order

    except ccxt.OrderImmediatelyFillable as e:
         # This happens if PostOnly is True and the order would match immediately
         if params.get('postOnly'):
             logger.warning(f"{Fore.YELLOW}{log_prefix}: PostOnly order failed (would fill immediately): {e}{Style.RESET_ALL}")
             return None # Return None as the order was rejected by the exchange
         else: # Should not happen if PostOnly is False
              logger.error(f"{Fore.RED}{log_prefix}: Unexpected OrderImmediatelyFillable without PostOnly: {e}{Style.RESET_ALL}")
              return None
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"{Fore.RED}{log_prefix}: API Error: {type(e).__name__} - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): {type(e).__name__}", config.sms_config)
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): Unexpected {type(e).__name__}.", config.sms_config)
        return None


# --- get_current_position_bybit_v5 ---
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def get_current_position_bybit_v5(exchange: ccxt.bybit, symbol: str, config: Config) -> Dict[str, Any]:
    """Fetches the current position details for a symbol using Bybit V5's fetchPositions logic."""
    func_name = "get_current_position"; log_prefix = f"[{func_name}({symbol}, V5)]"
    api_conf = config.api_config
    default_position: Dict[str, Any] = {'symbol': symbol, 'side': api_conf.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0"), 'liq_price': None, 'mark_price': None, 'pnl_unrealized': None, 'leverage': None, 'info': {}}
    logger.debug(f"{log_prefix} Fetching position...")
    try:
        market = exchange.market(symbol); market_id = market['id']; category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"); return default_position
        if not exchange.has.get('fetchPositions'): logger.error(f"{Fore.RED}{log_prefix} fetchPositions not available.{Style.RESET_ALL}"); return default_position

        # V5 fetchPositions requires category and optionally symbol
        params = {'category': category, 'symbol': market_id}; logger.debug(f"{log_prefix} Calling fetch_positions with params: {params}")
        # Fetch specific symbol for efficiency
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        # logger.debug(f"{log_prefix} Raw positions response: {fetched_positions}")

        active_position_data: Optional[Dict] = None
        # V5 fetchPositions returns a list, find the one matching the symbol and relevant mode (One-Way = index 0)
        for pos in fetched_positions:
            pos_info = pos.get('info', {}); pos_symbol = pos_info.get('symbol'); pos_v5_side = pos_info.get('side', 'None'); pos_size_str = pos_info.get('size'); pos_idx = int(pos_info.get('positionIdx', -1))
            # Match symbol and ensure it's the One-Way position (idx=0) or primary hedge pos if mode allows
            if pos_symbol == market_id and pos_v5_side != 'None' and pos_idx == 0: # Assuming One-Way mode target
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"))
                if pos_size is not None and abs(pos_size) > api_conf.position_qty_epsilon:
                    active_position_data = pos; logger.debug(f"{log_prefix} Found active One-Way (idx 0) position."); break

        if active_position_data:
            try:
                # Parse standardized CCXT fields first, fallback to info dict
                side_std = active_position_data.get('side') # 'long' or 'short'
                contracts = safe_decimal_conversion(active_position_data.get('contracts'))
                entry_price = safe_decimal_conversion(active_position_data.get('entryPrice'))
                mark_price = safe_decimal_conversion(active_position_data.get('markPrice'))
                liq_price = safe_decimal_conversion(active_position_data.get('liquidationPrice'))
                pnl = safe_decimal_conversion(active_position_data.get('unrealizedPnl'))
                leverage = safe_decimal_conversion(active_position_data.get('leverage'))
                info = active_position_data.get('info', {}) # Raw API data

                # Map CCXT side to internal constants
                position_side = api_conf.pos_long if side_std == 'long' else (api_conf.pos_short if side_std == 'short' else api_conf.pos_none)
                quantity = abs(contracts) if contracts is not None else Decimal("0.0")

                # Fallback parsing from info dict if standard fields are missing (less likely with recent CCXT)
                if quantity <= api_conf.position_qty_epsilon:
                     pos_size_info = safe_decimal_conversion(info.get('size'))
                     if pos_size_info is not None: quantity = abs(pos_size_info)
                if entry_price is None: entry_price = safe_decimal_conversion(info.get('avgPrice'))
                if mark_price is None: mark_price = safe_decimal_conversion(info.get('markPrice'))
                if liq_price is None: liq_price = safe_decimal_conversion(info.get('liqPrice'))
                if pnl is None: pnl = safe_decimal_conversion(info.get('unrealisedPnl'))
                if leverage is None: leverage = safe_decimal_conversion(info.get('leverage'))
                if position_side == api_conf.pos_none:
                     side_info = info.get('side') # Bybit uses 'Buy'/'Sell' in info
                     position_side = api_conf.pos_long if side_info == 'Buy' else (api_conf.pos_short if side_info == 'Sell' else api_conf.pos_none)

                # Final check
                if position_side == api_conf.pos_none or quantity <= api_conf.position_qty_epsilon:
                    logger.info(f"{log_prefix} Position found but size/side negligible after parsing."); return default_position

                log_color = Fore.GREEN if position_side == api_conf.pos_long else Fore.RED
                logger.info(f"{log_color}{log_prefix} ACTIVE {position_side} {symbol}: Qty={format_amount(exchange, symbol, quantity)}, Entry={format_price(exchange, symbol, entry_price)}, Mark={format_price(exchange, symbol, mark_price)}, Liq~{format_price(exchange, symbol, liq_price)}, uPNL={format_price(exchange, api_conf.usdt_symbol, pnl)}, Lev={leverage}x{Style.RESET_ALL}")
                return {'symbol': symbol, 'side': position_side, 'qty': quantity, 'entry_price': entry_price, 'liq_price': liq_price, 'mark_price': mark_price, 'pnl_unrealized': pnl, 'leverage': leverage, 'info': info }

            except Exception as parse_err:
                logger.warning(f"{Fore.YELLOW}{log_prefix} Error parsing active pos: {parse_err}. Data: {str(active_position_data)[:300]}{Style.RESET_ALL}"); return default_position
        else:
            logger.info(f"{log_prefix} No active One-Way position found for {symbol}."); return default_position

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API Error fetching pos: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return default_position # Return default on non-network API errors
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching pos: {e}{Style.RESET_ALL}", exc_info=True)
        return default_position


# --- close_position_reduce_only ---
@retry_api_call(max_retries_override=2, initial_delay_override=1)
def close_position_reduce_only(
    exchange: ccxt.bybit, symbol: str, config: Config, position_to_close: Optional[Dict[str, Any]] = None, reason: str = "Signal Close"
) -> Optional[Dict[str, Any]]:
    """Closes the current position for the given symbol using a reduce-only market order."""
    func_name = "close_position_reduce_only"; market_base = symbol.split('/')[0]; log_prefix = f"[{func_name}({symbol}, Reason:{reason})]"
    api_conf = config.api_config
    logger.info(f"{Fore.YELLOW}{log_prefix} Initiating close...{Style.RESET_ALL}")

    # --- Get Position State ---
    live_position_data: Dict[str, Any]
    if position_to_close:
        logger.debug(f"{log_prefix} Using provided position state."); live_position_data = position_to_close
    else:
        logger.debug(f"{log_prefix} Fetching current position state...");
        live_position_data = get_current_position_bybit_v5(exchange, symbol, config) # Fetch fresh state

    live_side = live_position_data.get('side', api_conf.pos_none)
    live_qty = live_position_data.get('qty', Decimal("0.0"))

    if live_side == api_conf.pos_none or live_qty <= api_conf.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}{log_prefix} No active position validated or qty is zero. Aborting close.{Style.RESET_ALL}")
        return None # Indicate no action needed / nothing to close

    # Determine the side needed for the closing order
    close_order_side: Literal['buy', 'sell'] = api_conf.side_sell if live_side == api_conf.pos_long else api_conf.side_buy

    # --- Place Closing Order ---
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: raise ValueError("Cannot determine category for close order.")

        qty_str = format_amount(exchange, symbol, live_qty)
        if qty_str is None or qty_str == "Error": raise ValueError("Failed formatting position quantity.")
        qty_float = float(qty_str)

        # Use place_market_order helper for consistency and checks
        close_order = place_market_order_slippage_check(
            exchange=exchange,
            symbol=symbol,
            side=close_order_side,
            amount=live_qty, # Pass Decimal amount
            config=config,
            is_reduce_only=True,
            client_order_id=f"close_{market_base}_{int(time.time())}"[-36:], # Generate client ID
            reason=f"Close {live_side} ({reason})" # Pass reason for logging inside helper
        )

        if close_order and close_order.get('id'):
            fill_price = safe_decimal_conversion(close_order.get('average')); fill_qty = safe_decimal_conversion(close_order.get('filled', '0.0')); order_id = format_order_id(close_order.get('id'))
            logger.success(f"{Fore.GREEN}{Style.BRIGHT}{log_prefix} Close Order ({reason}) Submitted OK for {symbol}. ID:{order_id}, Filled:{format_amount(exchange, symbol, fill_qty)}/{qty_str}, AvgFill:{format_price(exchange, symbol, fill_price)}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] Closed {live_side} {qty_str} @ ~{format_price(exchange, symbol, fill_price)} ({reason}). ID:{order_id}", config.sms_config)
            return close_order
        else:
            # place_market_order already logs errors
            logger.error(f"{Fore.RED}{log_prefix} Failed to submit close order via helper.{Style.RESET_ALL}")
            # Check if the helper might have returned None due to slippage check failure
            # If so, the position might still be open.
            return None

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e: # Should be less likely if place_market_order used
        logger.error(f"{Fore.RED}{log_prefix} Close Order Error ({reason}): {type(e).__name__} - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): {type(e).__name__}", config.sms_config)
        if isinstance(e, ccxt.NetworkError): raise e
        return None
    except ccxt.ExchangeError as e:
        error_str = str(e).lower(); err_code = getattr(e, 'code', None)
        # Bybit V5 codes indicating position already closed or reduce failed due to size
        # 110025: Position is closed
        # 110045: Order would not reduce position size
        # 30086: order quantity is greater than the remaining position size (UTA?)
        if err_code in [110025, 110045, 30086] or any(code in error_str for code in ["position is closed", "order would not reduce", "position size is zero", "qty is larger than position size"]):
            logger.warning(f"{Fore.YELLOW}{log_prefix} Close Order ({reason}): Exchange indicates already closed/zero or reduce fail: {e}. Assuming closed.{Style.RESET_ALL}")
            return None # Treat as if closed successfully
        else:
            logger.error(f"{Fore.RED}{log_prefix} Close Order ExchangeError ({reason}): {e}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): ExchangeError Code {err_code}", config.sms_config)
            return None
    except (ccxt.NetworkError, ValueError) as e:
        logger.error(f"{Fore.RED}{log_prefix} Close Order Network/Setup Error ({reason}): {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Close Order Unexpected Error ({reason}): {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): Unexpected Error", config.sms_config)
        return None

# --- fetch_funding_rate ---
@retry_api_call()
def fetch_funding_rate(exchange: ccxt.bybit, symbol: str, config: Config) -> Optional[Dict[str, Any]]:
    """Fetches the current funding rate details for a perpetual swap symbol on Bybit V5."""
    func_name = "fetch_funding_rate"; log_prefix = f"[{func_name}({symbol})]"
    logger.debug(f"{log_prefix} Fetching funding rate...")
    try:
        market = exchange.market(symbol)
        if not market.get('swap', False): logger.error(f"{Fore.RED}{log_prefix} Not a swap market.{Style.RESET_ALL}"); return None
        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}{log_prefix} Invalid category '{category}' for funding rate.{Style.RESET_ALL}"); return None

        params = {'category': category}; logger.debug(f"{log_prefix} Calling fetch_funding_rate with params: {params}")
        funding_rate_info = exchange.fetch_funding_rate(symbol, params=params) # Pass symbol here

        # Parse the standardized CCXT response
        processed_fr: Dict[str, Any] = {
            'symbol': funding_rate_info.get('symbol'),
            'fundingRate': safe_decimal_conversion(funding_rate_info.get('fundingRate')),
            'fundingTimestamp': funding_rate_info.get('fundingTimestamp'), # ms timestamp of rate application
            'fundingDatetime': funding_rate_info.get('fundingDatetime'), # ISO8601 string
            'markPrice': safe_decimal_conversion(funding_rate_info.get('markPrice')),
            'indexPrice': safe_decimal_conversion(funding_rate_info.get('indexPrice')),
            'nextFundingTime': funding_rate_info.get('nextFundingTimestamp'), # ms timestamp of next funding
            'nextFundingDatetime': None, # Will be populated below
            'info': funding_rate_info.get('info', {}) # Raw exchange response
        }

        if processed_fr['fundingRate'] is None: logger.warning(f"{log_prefix} Could not parse 'fundingRate'.")
        if processed_fr['nextFundingTime']:
            try: processed_fr['nextFundingDatetime'] = pd.to_datetime(processed_fr['nextFundingTime'], unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S %Z') if PANDAS_AVAILABLE else str(processed_fr['nextFundingTime'])
            except Exception as dt_err: logger.warning(f"{log_prefix} Could not format next funding datetime: {dt_err}")

        rate = processed_fr.get('fundingRate'); next_dt_str = processed_fr.get('nextFundingDatetime', "N/A"); rate_str = f"{rate:.6%}" if rate is not None else "N/A"
        logger.info(f"{log_prefix} Funding Rate: {rate_str}. Next: {next_dt_str}")
        return processed_fr

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API Error fetching funding rate: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching funding rate: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- set_position_mode_bybit_v5 ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def set_position_mode_bybit_v5(exchange: ccxt.bybit, symbol_or_category: str, mode: Literal['one-way', 'hedge'], config: Config) -> bool:
    """Sets the position mode (One-Way or Hedge) for a specific category (Linear/Inverse) on Bybit V5."""
    func_name = "set_position_mode"; log_prefix = f"[{func_name}(Target:{mode})]"
    logger.info(f"{Fore.CYAN}{log_prefix} Setting mode '{mode}' for category of '{symbol_or_category}'...{Style.RESET_ALL}")

    # Map mode string to Bybit API code (0 for One-Way, 3 for Hedge)
    mode_map = {'one-way': '0', 'hedge': '3'}; target_mode_code = mode_map.get(mode.lower())
    if target_mode_code is None: logger.error(f"{Fore.RED}{log_prefix} Invalid mode '{mode}'. Use 'one-way' or 'hedge'.{Style.RESET_ALL}"); return False

    # Determine target category
    target_category: Optional[Literal['linear', 'inverse']] = None
    if symbol_or_category.lower() in ['linear', 'inverse']:
        target_category = symbol_or_category.lower() # type: ignore
    else:
        try:
            market = exchange.market(symbol_or_category); category = _get_v5_category(market);
            if category in ['linear', 'inverse']: target_category = category # type: ignore
        except Exception as e: logger.warning(f"{log_prefix} Could not get market/category for '{symbol_or_category}': {e}")

    if not target_category: logger.error(f"{Fore.RED}{log_prefix} Could not determine contract category (linear/inverse) from '{symbol_or_category}'.{Style.RESET_ALL}"); return False

    logger.debug(f"{log_prefix} Target Category: {target_category}, Mode Code: {target_mode_code} ('{mode}')")

    # --- Call V5 Endpoint ---
    # Requires calling a private endpoint not directly exposed by standard CCXT methods easily.
    # Use `exchange.private_post_v5_position_switch_mode`
    endpoint = 'private_post_v5_position_switch_mode'
    if not hasattr(exchange, endpoint):
        logger.error(f"{Fore.RED}{log_prefix} CCXT version lacks '{endpoint}'. Cannot set mode via V5 API.{Style.RESET_ALL}")
        return False

    params = {'category': target_category, 'mode': target_mode_code}; logger.debug(f"{log_prefix} Calling {endpoint} with params: {params}")
    try:
        response = getattr(exchange, endpoint)(params); logger.debug(f"{log_prefix} Raw V5 endpoint response: {response}")
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', '').lower()

        if ret_code == 0:
            logger.success(f"{Fore.GREEN}{log_prefix} Mode successfully set to '{mode}' for {target_category}.{Style.RESET_ALL}"); return True
        # Code 110021: Already in the target mode (or mode not modified)
        # Code 34036: Already in the target mode (specific to UTA?)
        elif ret_code in [110021, 34036] or "not modified" in ret_msg:
            logger.info(f"{Fore.CYAN}{log_prefix} Mode already set to '{mode}' for {target_category}.{Style.RESET_ALL}"); return True
        # Code 110020: Cannot switch mode with active positions/orders
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg:
            logger.error(f"{Fore.RED}{log_prefix} Cannot switch mode: Active position or orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"); return False
        else:
            # Raise unexpected error codes
            raise ccxt.ExchangeError(f"Bybit API error setting mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e:
        # Handle specific errors like active positions/orders without failing loudly
        if isinstance(e, ccxt.ExchangeError) and "110020" in str(e):
             logger.error(f"{Fore.RED}{log_prefix} Cannot switch mode (active position/orders): {e}{Style.RESET_ALL}")
             return False # Return False clearly
        else:
             logger.warning(f"{Fore.YELLOW}{log_prefix} API Error setting mode: {e}{Style.RESET_ALL}")
             if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)): raise e # Retry network/auth errors
             return False
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error setting mode: {e}{Style.RESET_ALL}", exc_info=True)
        return False


# --- fetch_l2_order_book_validated ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def fetch_l2_order_book_validated(
    exchange: ccxt.bybit, symbol: str, limit: int, config: Config
) -> Optional[Dict[str, List[Tuple[Decimal, Decimal]]]]:
    """Fetches the Level 2 order book for a symbol using Bybit V5 fetchOrderBook and validates the data."""
    func_name = "fetch_l2_order_book"; log_prefix = f"[{func_name}({symbol}, Limit:{limit})]"
    logger.debug(f"{log_prefix} Fetching L2 OB...")
    api_conf = config.api_config

    if not exchange.has.get('fetchOrderBook'): logger.error(f"{Fore.RED}{log_prefix} fetchOrderBook not supported.{Style.RESET_ALL}"); return None

    try:
        market = exchange.market(symbol); category = _get_v5_category(market);
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"); return None
        params = {'category': category}

        # Check and clamp limit according to Bybit V5 category limits
        max_limit_map = {'spot': 200, 'linear': 500, 'inverse': 500, 'option': 25} # Check Bybit docs for current limits
        max_limit = max_limit_map.get(category, 50) # Default fallback
        if limit > max_limit: logger.warning(f"{log_prefix} Clamping limit {limit} to {max_limit} for category '{category}'."); limit = max_limit

        logger.debug(f"{log_prefix} Calling fetchOrderBook with limit={limit}, params={params}")
        order_book = exchange.fetch_order_book(symbol, limit=limit, params=params)

        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book: raise ValueError("Invalid OB structure")
        raw_bids=order_book['bids']; raw_asks=order_book['asks']
        if not isinstance(raw_bids, list) or not isinstance(raw_asks, list): raise ValueError("Bids/Asks not lists")

        # Validate and convert entries to Decimal tuples [price, amount]
        validated_bids: List[Tuple[Decimal, Decimal]] = []; validated_asks: List[Tuple[Decimal, Decimal]] = []; conversion_errors = 0
        for p_str, a_str in raw_bids:
            p = safe_decimal_conversion(p_str); a = safe_decimal_conversion(a_str)
            if not (p and a and p > 0 and a >= 0): conversion_errors += 1; continue
            validated_bids.append((p, a))
        for p_str, a_str in raw_asks:
            p = safe_decimal_conversion(p_str); a = safe_decimal_conversion(a_str)
            if not (p and a and p > 0 and a >= 0): conversion_errors += 1; continue
            validated_asks.append((p, a))

        if conversion_errors > 0: logger.warning(f"{log_prefix} Skipped {conversion_errors} invalid OB entries.")
        if not validated_bids or not validated_asks: logger.warning(f"{log_prefix} Empty validated bids/asks."); # Return potentially empty lists

        # Check for crossed book
        if validated_bids and validated_asks and validated_bids[0][0] >= validated_asks[0][0]:
            logger.error(f"{Fore.RED}{log_prefix} OB crossed: Bid ({validated_bids[0][0]}) >= Ask ({validated_asks[0][0]}).{Style.RESET_ALL}")
            # Return the crossed book for upstream handling

        logger.debug(f"{log_prefix} Processed L2 OB OK. Bids:{len(validated_bids)}, Asks:{len(validated_asks)}")
        # Return validated data in a structure consistent with analysis needs
        return {'symbol': symbol, 'bids': validated_bids, 'asks': validated_asks, 'timestamp': order_book.get('timestamp'), 'datetime': order_book.get('datetime'), 'nonce': order_book.get('nonce')}

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol, ValueError) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API/Validation Error fetching L2 OB: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching L2 OB: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- place_native_stop_loss ---
@retry_api_call(max_retries_override=1, initial_delay_override=0)
def place_native_stop_loss(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, stop_price: Decimal, config: Config,
    trigger_by: Literal['LastPrice', 'MarkPrice', 'IndexPrice'] = 'MarkPrice', client_order_id: Optional[str] = None, position_idx: Literal[0, 1, 2] = 0
) -> Optional[Dict]:
    """Places a native Stop Market order on Bybit V5, intended as a Stop Loss (reduceOnly)."""
    func_name = "place_native_stop_loss"; log_prefix = f"[{func_name}({side.upper()})]"
    api_conf = config.api_config
    logger.info(f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trigger @ {format_price(exchange, symbol, stop_price)} ({trigger_by}), PosIdx:{position_idx}...{Style.RESET_ALL}")

    if amount <= api_conf.position_qty_epsilon or stop_price <= Decimal("0"): logger.error(f"{Fore.RED}{log_prefix}: Invalid amount/stop price.{Style.RESET_ALL}"); return None

    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"); return None

        amount_str = format_amount(exchange, symbol, amount); stop_price_str = format_price(exchange, symbol, stop_price)
        if any(v is None or v == "Error" for v in [amount_str, stop_price_str]): raise ValueError("Invalid amount/price formatting.")
        amount_float = float(amount_str)

        # --- V5 Stop Order Parameters ---
        # Use create_order with type='market' and stop loss params
        params: Dict[str, Any] = {
            'category': category,
            'stopLoss': stop_price_str, # Price level for the stop loss trigger
            'slTriggerBy': trigger_by, # Trigger type (LastPrice, MarkPrice, IndexPrice)
            'reduceOnly': True, # Ensure it only closes position
            'positionIdx': position_idx, # Specify position index (0 for one-way)
            'tpslMode': 'Full', # Assume full position SL unless partial is needed
            'slOrderType': 'Market' # Execute as market order when triggered
            # 'slLimitPrice': '...' # Required if slOrderType='Limit'
        }
        if client_order_id:
            max_coid_len = 36; valid_coid = client_order_id[:max_coid_len]
            params['orderLinkId'] = valid_coid # V5 uses orderLinkId for client ID on conditional orders
            if len(valid_coid) < len(client_order_id): logger.warning(f"{log_prefix} Client OID truncated to orderLinkId: '{valid_coid}'")

        bg = Back.YELLOW; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE Stop Loss (Market exec) -> Qty:{amount_float}, Side:{side}, TriggerPx:{stop_price_str}, TriggerBy:{trigger_by}, Params:{params}{Style.RESET_ALL}")

        # create_order is used for placing conditional orders like stops in V5 via CCXT
        sl_order = exchange.create_order(
             symbol=symbol,
             type='market', # Base order type is Market (triggered)
             side=side, # The side of the order when triggered (e.g., sell for long SL)
             amount=amount_float,
             params=params
        )

        order_id = sl_order.get('id'); client_oid_resp = sl_order.get('info', {}).get('orderLinkId', params.get('orderLinkId', 'N/A')); status = sl_order.get('status', '?')
        returned_stop_price = safe_decimal_conversion(sl_order.get('stopPrice', info.get('stopLoss')), None) # CCXT might use stopPrice
        returned_trigger = sl_order.get('trigger', trigger_by) # CCXT might use trigger

        logger.success(f"{Fore.GREEN}{log_prefix}: Native SL order placed OK. ID:{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trigger:{format_price(exchange, symbol, returned_stop_price)} (by {returned_trigger}){Style.RESET_ALL}")
        return sl_order

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol) as e:
        logger.error(f"{Fore.RED}{log_prefix}: API Error placing SL: {type(e).__name__} - {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Unexpected error placing SL: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{symbol.split('/')[0]}] SL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config.sms_config)
        return None


# --- fetch_open_orders_filtered ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def fetch_open_orders_filtered(
    exchange: ccxt.bybit, symbol: str, config: Config, side: Optional[Literal['buy', 'sell']] = None,
    order_type: Optional[str] = None, order_filter: Optional[Literal['Order', 'StopOrder', 'tpslOrder']] = None
) -> Optional[List[Dict]]:
    """Fetches open orders for a specific symbol on Bybit V5, with optional filtering."""
    func_name = "fetch_open_orders_filtered"; filter_log = f"(Side:{side or 'Any'}, Type:{order_type or 'Any'}, V5Filter:{order_filter or 'Default'})"
    log_prefix = f"[{func_name}({symbol}) {filter_log}]"
    logger.debug(f"{log_prefix} Fetching open orders...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"); return None

        params: Dict[str, Any] = {'category': category}
        # V5 requires orderFilter to fetch conditional orders ('StopOrder' or 'tpslOrder')
        if order_filter: params['orderFilter'] = order_filter
        elif order_type: # Infer filter from type if not provided
            norm_type = order_type.lower().replace('_', '').replace('-', '')
            if any(k in norm_type for k in ['stop', 'trigger', 'take', 'tpsl', 'conditional']): params['orderFilter'] = 'StopOrder'
            else: params['orderFilter'] = 'Order' # Assume standard limit/market
        # else: fetch all types if no filter specified? Bybit default might be 'Order'. Let CCXT handle default.

        logger.debug(f"{log_prefix} Calling fetch_open_orders with symbol={symbol}, params={params}")
        # Pass symbol to fetch for that specific market
        open_orders = exchange.fetch_open_orders(symbol=symbol, params=params)

        if not open_orders: logger.debug(f"{log_prefix} No open orders found matching criteria."); return []

        # --- Client-Side Filtering (if needed beyond API filter) ---
        filtered = open_orders; initial_count = len(filtered)
        if side:
            side_lower = side.lower(); filtered = [o for o in filtered if o.get('side', '').lower() == side_lower]
            logger.debug(f"{log_prefix} Filtered by side='{side}'. Count: {initial_count} -> {len(filtered)}.")
            initial_count = len(filtered) # Update count for next filter log

        if order_type:
            norm_type_filter = order_type.lower().replace('_', '').replace('-', '');
            # Check standard 'type' and potentially 'info.orderType' or conditional types
            def check_type(o):
                o_type = o.get('type', '').lower().replace('_', '').replace('-', '')
                info = o.get('info', {})
                # Check standard type, info type, and conditional type fields
                return (o_type == norm_type_filter or
                        info.get('orderType', '').lower() == norm_type_filter or
                        info.get('stopOrderType', '').lower() == norm_type_filter)

            filtered = [o for o in filtered if check_type(o)]
            logger.debug(f"{log_prefix} Filtered by type='{order_type}'. Count: {initial_count} -> {len(filtered)}.")

        logger.info(f"{log_prefix} Fetched/filtered {len(filtered)} open orders.")
        return filtered

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API Error fetching open orders: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching open orders: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- calculate_margin_requirement ---
def calculate_margin_requirement(
    exchange: ccxt.bybit, symbol: str, amount: Decimal, price: Decimal, leverage: Decimal, config: Config,
    order_side: Literal['buy', 'sell'], is_maker: bool = False
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Calculates the estimated Initial Margin (IM) requirement for placing an order on Bybit V5."""
    func_name = "calculate_margin_requirement"; log_prefix = f"[{func_name}]"
    api_conf = config.api_config
    logger.debug(f"{log_prefix} Calc margin: {order_side} {format_amount(exchange, symbol, amount)} @ {format_price(exchange, symbol, price)}, Lev:{leverage}x, Maker:{is_maker}")

    if amount <= 0 or price <= 0 or leverage <= 0: logger.error(f"{Fore.RED}{log_prefix} Invalid inputs (amount/price/leverage must be > 0).{Style.RESET_ALL}"); return None, None

    try:
        market = exchange.market(symbol); quote_currency = market.get('quote', api_conf.usdt_symbol)
        if not market.get('contract'): logger.error(f"{Fore.RED}{log_prefix} Not a contract symbol: {symbol}. Cannot calculate margin.{Style.RESET_ALL}"); return None, None

        # Calculate Order Value
        # Handle inverse contracts where value might be Base / Price
        is_inverse = market.get('inverse', False)
        position_value: Decimal
        if is_inverse:
             # Value = Amount (in Base currency contracts) / Price
             if price <= 0: raise ValueError("Price must be positive for inverse value calculation.")
             position_value = amount / price # Result is in Quote currency terms
        else:
             # Value = Amount (in Base) * Price
             position_value = amount * price
        logger.debug(f"{log_prefix} Est Order Value: {format_price(exchange, quote_currency, position_value)} {quote_currency}")

        # Initial Margin = Order Value / Leverage
        initial_margin_base = position_value / leverage; logger.debug(f"{log_prefix} Base IM (Value/Lev): {format_price(exchange, quote_currency, initial_margin_base)} {quote_currency}")

        # Estimate Fees (optional, adds buffer)
        fee_rate = api_conf.maker_fee_rate if is_maker else api_conf.taker_fee_rate;
        estimated_fee = position_value * fee_rate; logger.debug(f"{log_prefix} Est Fee ({fee_rate:.4%}): {format_price(exchange, quote_currency, estimated_fee)} {quote_currency}")

        # Total Estimated IM = Base IM + Estimated Fee
        total_initial_margin_estimate = initial_margin_base # + estimated_fee # Decide whether to include fee estimate

        logger.info(f"{log_prefix} Est TOTAL Initial Margin Req: {format_price(exchange, quote_currency, total_initial_margin_estimate)} {quote_currency}")

        # Estimate Maintenance Margin (MM)
        maintenance_margin_estimate: Optional[Decimal] = None
        try:
            # CCXT market structure often has maintenanceMarginRate under 'info' or directly
            mmr_keys = ['maintenanceMarginRate', 'mmr', 'maintMarginRatio'] # Common keys
            mmr_rate_str = None
            market_info = market.get('info', {})
            for key in mmr_keys:
                mmr_rate_str = market_info.get(key) or market.get(key)
                if mmr_rate_str is not None: break

            if mmr_rate_str is not None:
                 mmr_rate = safe_decimal_conversion(mmr_rate_str, context=f"{symbol} MMR")
                 if mmr_rate is not None and mmr_rate >= 0:
                     maintenance_margin_estimate = position_value * mmr_rate
                     logger.debug(f"{log_prefix} Basic MM Estimate (Base MMR {mmr_rate:.4%}): {format_price(exchange, quote_currency, maintenance_margin_estimate)} {quote_currency}")
                 else: logger.debug(f"{log_prefix} Could not parse valid MMR rate from '{mmr_rate_str}'.")
            else: logger.debug(f"{log_prefix} MMR key not found in market info.")
        except Exception as mm_err: logger.warning(f"{log_prefix} Could not estimate MM: {mm_err}")

        return total_initial_margin_estimate, maintenance_margin_estimate

    except (DivisionByZero, KeyError, ValueError) as e: logger.error(f"{Fore.RED}{log_prefix} Calculation error: {e}{Style.RESET_ALL}"); return None, None
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error during margin calculation: {e}{Style.RESET_ALL}", exc_info=True); return None, None


# --- fetch_ticker_validated (Fixed Timestamp/Age Logic) ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def fetch_ticker_validated(
    exchange: ccxt.bybit, symbol: str, config: Config, max_age_seconds: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Fetches the ticker for a symbol from Bybit V5, validates its freshness and key values.
    Returns a dictionary with Decimal values, or None if validation fails or API error occurs.
    """
    func_name = "fetch_ticker_validated"; log_prefix = f"[{func_name}({symbol})]"
    logger.debug(f"{log_prefix} Fetching/Validating ticker...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"); return None
        params = {'category': category}

        logger.debug(f"{log_prefix} Calling fetch_ticker with params: {params}")
        ticker = exchange.fetch_ticker(symbol, params=params)

        # --- Validation ---
        if not ticker: raise ValueError("fetch_ticker returned empty response.")

        timestamp_ms = ticker.get('timestamp')
        if timestamp_ms is None: raise ValueError("Ticker data is missing timestamp.") # Fail early if TS missing

        current_time_ms = time.time() * 1000
        age_seconds = (current_time_ms - timestamp_ms) / 1000.0

        # Check age validity
        if age_seconds > max_age_seconds: raise ValueError(f"Ticker data stale (Age: {age_seconds:.1f}s > Max: {max_age_seconds}s).")
        if age_seconds < -10: # Allow small future drift
             raise ValueError(f"Ticker timestamp ({timestamp_ms}) seems to be in the future (Age: {age_seconds:.1f}s).")

        # Validate and convert key prices to Decimal
        last_price = safe_decimal_conversion(ticker.get('last')); bid_price = safe_decimal_conversion(ticker.get('bid')); ask_price = safe_decimal_conversion(ticker.get('ask'))
        if last_price is None or last_price <= 0: raise ValueError(f"Invalid 'last' price: {ticker.get('last')}")
        if bid_price is None or bid_price <= 0: logger.warning(f"{log_prefix} Invalid/missing 'bid': {ticker.get('bid')}")
        if ask_price is None or ask_price <= 0: logger.warning(f"{log_prefix} Invalid/missing 'ask': {ticker.get('ask')}")

        spread, spread_pct = None, None
        if bid_price and ask_price:
             if bid_price >= ask_price: logger.warning(f"{log_prefix} Bid ({bid_price}) >= Ask ({ask_price}). Using NaN for spread.") # Warn but don't fail
             else:
                 spread = ask_price - bid_price
                 mid_price = (ask_price + bid_price) / 2
                 spread_pct = (spread / mid_price) * 100 if mid_price > 0 else Decimal("inf")
        else: logger.warning(f"{log_prefix} Cannot calculate spread due to missing bid/ask.")

        # Convert other fields to Decimal safely
        validated_ticker = {
            'symbol': ticker.get('symbol', symbol), 'timestamp': timestamp_ms, 'datetime': ticker.get('datetime'),
            'last': last_price, 'bid': bid_price, 'ask': ask_price,
            'bidVolume': safe_decimal_conversion(ticker.get('bidVolume')),
            'askVolume': safe_decimal_conversion(ticker.get('askVolume')),
            'baseVolume': safe_decimal_conversion(ticker.get('baseVolume')),
            'quoteVolume': safe_decimal_conversion(ticker.get('quoteVolume')),
            'high': safe_decimal_conversion(ticker.get('high')),
            'low': safe_decimal_conversion(ticker.get('low')),
            'open': safe_decimal_conversion(ticker.get('open')),
            'close': last_price, # Use last price as close for ticker
            'change': safe_decimal_conversion(ticker.get('change')),
            'percentage': safe_decimal_conversion(ticker.get('percentage')),
            'average': safe_decimal_conversion(ticker.get('average')),
            'vwap': safe_decimal_conversion(ticker.get('vwap')),
            'spread': spread, 'spread_pct': spread_pct,
            'info': ticker.get('info', {})
        }
        logger.debug(f"{log_prefix} Ticker OK: Last={format_price(exchange, symbol, last_price)}, Spread={(spread_pct or Decimal('NaN')):.4f}% (Age:{age_seconds:.1f}s)")
        return validated_ticker

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API/Symbol error fetching ticker: {type(e).__name__} - {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except ValueError as e:
        # Catch validation errors (stale, bad price, missing timestamp)
        logger.warning(f"{Fore.YELLOW}{log_prefix} Ticker validation failed: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected ticker error: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- place_native_trailing_stop ---
@retry_api_call(max_retries_override=1, initial_delay_override=0)
def place_native_trailing_stop(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, trailing_offset: Union[Decimal, str], config: Config,
    activation_price: Optional[Decimal] = None, trigger_by: Literal['LastPrice', 'MarkPrice', 'IndexPrice'] = 'MarkPrice',
    client_order_id: Optional[str] = None, position_idx: Literal[0, 1, 2] = 0
) -> Optional[Dict]:
    """Places a native Trailing Stop Market order on Bybit V5 (reduceOnly)."""
    func_name = "place_native_trailing_stop"; log_prefix = f"[{func_name}({side.upper()})]"
    api_conf = config.api_config
    params: Dict[str, Any] = {}; trail_log_str = "";

    try:
        # --- Validate & Parse Trailing Offset ---
        if isinstance(trailing_offset, str) and trailing_offset.endswith('%'):
            percent_val = safe_decimal_conversion(trailing_offset.rstrip('%'))
            # Bybit V5 percentage trailing stop range (check docs, e.g., 0.1% to 10%)
            min_pct, max_pct = Decimal("0.1"), Decimal("10.0")
            if not (percent_val and min_pct <= percent_val <= max_pct): raise ValueError(f"Percentage trail '{trailing_offset}' out of range ({min_pct}%-{max_pct}%).")
            params['trailingStop'] = str(percent_val.quantize(Decimal("0.01"))) # Format to 2 decimal places for %
            trail_log_str = f"{percent_val}%";
        elif isinstance(trailing_offset, Decimal):
            if trailing_offset <= Decimal("0"): raise ValueError(f"Absolute trail delta must be positive: {trailing_offset}")
            delta_str = format_price(exchange, symbol, trailing_offset) # Use price precision for delta
            if delta_str is None or delta_str == "Error": raise ValueError("Invalid absolute trail delta formatting.")
            params['trailingStop'] = delta_str # V5 uses trailingStop for both % and absolute value
            trail_log_str = f"{delta_str} (abs)"
        else: raise TypeError(f"Invalid trailing_offset type: {type(trailing_offset)}")

        if activation_price is not None and activation_price <= Decimal("0"): raise ValueError("Activation price must be positive.")

        logger.info(f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trail:{trail_log_str}, ActPx:{format_price(exchange, symbol, activation_price) or 'Immediate'}, Trigger:{trigger_by}, PosIdx:{position_idx}{Style.RESET_ALL}")

        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"); return None

        amount_str = format_amount(exchange, symbol, amount)
        if amount_str is None or amount_str == "Error": raise ValueError("Invalid amount formatting.")
        amount_float = float(amount_str)

        activation_price_str = format_price(exchange, symbol, activation_price) if activation_price is not None else None
        if activation_price is not None and (activation_price_str is None or activation_price_str == "Error"): raise ValueError("Invalid activation price formatting.")

        # --- V5 Trailing Stop Parameters ---
        # Use create_order with type='market' and trailing stop params
        params.update({
            'category': category,
            'reduceOnly': True,
            'positionIdx': position_idx,
            'tpslMode': 'Full', # Assume full position TSL
            'triggerBy': trigger_by,
            # 'tsOrderType': 'Market' # This seems redundant if base type is Market
            # Trailing stop value/percentage already in params['trailingStop']
        })
        if activation_price_str is not None: params['activePrice'] = activation_price_str
        if client_order_id:
            max_coid_len = 36; valid_coid = client_order_id[:max_coid_len]
            params['orderLinkId'] = valid_coid # Use orderLinkId for conditional orders
            if len(valid_coid) < len(client_order_id): logger.warning(f"{log_prefix} Client OID truncated to orderLinkId: '{valid_coid}'")

        bg = Back.YELLOW; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE TSL (Market exec) -> Qty:{amount_float}, Side:{side}, Trail:{trail_log_str}, ActPx:{activation_price_str or 'Immediate'}, Params:{params}{Style.RESET_ALL}")

        tsl_order = exchange.create_order(
             symbol=symbol,
             type='market', # Base order type is Market (triggered)
             side=side,
             amount=amount_float,
             params=params
        )

        order_id = tsl_order.get('id'); client_oid_resp = tsl_order.get('info', {}).get('orderLinkId', params.get('orderLinkId', 'N/A')); status = tsl_order.get('status', '?')
        returned_trail = tsl_order.get('info', {}).get('trailingStop'); returned_act = safe_decimal_conversion(tsl_order.get('info', {}).get('activePrice')); returned_trigger = tsl_order.get('trigger', trigger_by)

        logger.success(f"{Fore.GREEN}{log_prefix}: Native TSL order placed OK. ID:{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trail:{returned_trail}, ActPx:{format_price(exchange, symbol, returned_act)}, TriggerBy:{returned_trigger}{Style.RESET_ALL}")
        return tsl_order

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol, ValueError, TypeError) as e:
        logger.error(f"{Fore.RED}{log_prefix}: API/Input Error placing TSL: {type(e).__name__} - {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Unexpected error placing TSL: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{symbol.split('/')[0]}] TSL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config.sms_config)
        return None

# --- fetch_account_info_bybit_v5 ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def fetch_account_info_bybit_v5(exchange: ccxt.bybit, config: Config) -> Optional[Dict[str, Any]]:
    """Fetches general account information from Bybit V5 API (`/v5/account/info`)."""
    func_name = "fetch_account_info"; log_prefix = f"[{func_name}(V5)]"
    logger.debug(f"{log_prefix} Fetching Bybit V5 account info...")
    endpoint = 'private_get_v5_account_info'
    try:
        if hasattr(exchange, endpoint):
            logger.debug(f"{log_prefix} Using {endpoint} endpoint.")
            account_info_raw = getattr(exchange, endpoint)(); logger.debug(f"{log_prefix} Raw Account Info response: {str(account_info_raw)[:400]}...")
            ret_code = account_info_raw.get('retCode'); ret_msg = account_info_raw.get('retMsg')
            if ret_code == 0 and 'result' in account_info_raw:
                result = account_info_raw['result']
                # Parse relevant fields (check Bybit docs for current fields)
                parsed_info = {
                    'unifiedMarginStatus': result.get('unifiedMarginStatus'), # 1: Regular account; 2: UTA Pro; 3: UTA classic; 4: Default margin account; 5: Not upgraded to UTA
                    'marginMode': result.get('marginMode'), # 0: regular margin; 1: portfolio margin (PM)
                    'dcpStatus': result.get('dcpStatus'), # Disconnect-protect status
                    'timeWindow': result.get('timeWindow'),
                    'smtCode': result.get('smtCode'),
                    'isMasterTrader': result.get('isMasterTrader'),
                    'updateTime': result.get('updateTime'),
                    'rawInfo': result # Include raw data
                }
                logger.info(f"{log_prefix} Account Info: UTA Status={parsed_info.get('unifiedMarginStatus', 'N/A')}, MarginMode={parsed_info.get('marginMode', 'N/A')}, DCP Status={parsed_info.get('dcpStatus', 'N/A')}")
                return parsed_info
            else: raise ccxt.ExchangeError(f"Failed fetch/parse account info. Code={ret_code}, Msg='{ret_msg}'")
        else:
            logger.warning(f"{log_prefix} CCXT lacks '{endpoint}'. Using fallback fetch_accounts() (less detail).")
            accounts = exchange.fetch_accounts(); # Standard CCXT method
            if accounts: logger.info(f"{log_prefix} Fallback fetch_accounts(): {str(accounts[0])[:200]}..."); return accounts[0]
            else: logger.error(f"{Fore.RED}{log_prefix} Fallback fetch_accounts() returned no data.{Style.RESET_ALL}"); return None
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API Error fetching account info: {e}{Style.RESET_ALL}")
        if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)): raise e # Allow retry
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching account info: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- validate_market ---
def validate_market(
    exchange: ccxt.bybit, symbol: str, config: Config, check_active: bool = True
) -> Optional[Dict]:
    """Validates if a symbol exists on the exchange, is active, and optionally matches expectations from config."""
    func_name = "validate_market"; log_prefix = f"[{func_name}({symbol})]"
    api_conf = config.api_config
    eff_expected_type = api_conf.expected_market_type
    eff_expected_logic = api_conf.expected_market_logic
    require_contract = eff_expected_type != 'spot' # Require contract unless explicitly spot

    logger.debug(f"{log_prefix} Validating... Checks: Type='{eff_expected_type or 'Any'}', Logic='{eff_expected_logic or 'Any'}', Active={check_active}, Contract={require_contract}")
    try:
        # Load markets if not already loaded (should be done during init)
        if not exchange.markets:
            logger.info(f"{log_prefix} Markets not loaded. Loading...")
            exchange.load_markets(reload=True)
        if not exchange.markets: logger.error(f"{Fore.RED}{log_prefix} Failed to load markets.{Style.RESET_ALL}"); return None

        market = exchange.market(symbol) # Throws BadSymbol if not found
        is_active = market.get('active', False)

        if check_active and not is_active:
             # Inactive markets might still be useful for historical data, treat as warning
             logger.warning(f"{Fore.YELLOW}{log_prefix} Validation Warning: Market is INACTIVE.{Style.RESET_ALL}")
             # return None # Optionally fail validation for inactive markets

        actual_type = market.get('type'); # e.g., 'spot', 'swap'
        if eff_expected_type and actual_type != eff_expected_type:
            logger.error(f"{Fore.RED}{log_prefix} Validation Failed: Type mismatch. Expected '{eff_expected_type}', Got '{actual_type}'.{Style.RESET_ALL}")
            return None

        is_contract = market.get('contract', False);
        if require_contract and not is_contract:
            logger.error(f"{Fore.RED}{log_prefix} Validation Failed: Expected a contract, but market type is '{actual_type}'.{Style.RESET_ALL}")
            return None

        actual_logic_str: Optional[str] = None
        if is_contract:
            actual_logic_str = _get_v5_category(market); # linear/inverse
            if eff_expected_logic and actual_logic_str != eff_expected_logic:
                logger.error(f"{Fore.RED}{log_prefix} Validation Failed: Logic mismatch. Expected '{eff_expected_logic}', Got '{actual_logic_str}'.{Style.RESET_ALL}")
                return None

        logger.info(f"{Fore.GREEN}{log_prefix} Market OK: Type:{actual_type}, Logic:{actual_logic_str or 'N/A'}, Active:{is_active}.{Style.RESET_ALL}");
        return market

    except ccxt.BadSymbol as e: logger.error(f"{Fore.RED}{log_prefix} Validation Failed: Symbol not found. Error: {e}{Style.RESET_ALL}"); return None
    except ccxt.NetworkError as e: logger.error(f"{Fore.RED}{log_prefix} Network error during market validation: {e}{Style.RESET_ALL}"); return None # Network errors are usually critical for validation
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error validating market: {e}{Style.RESET_ALL}", exc_info=True); return None


# --- fetch_recent_trades ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def fetch_recent_trades(
    exchange: ccxt.bybit, symbol: str, config: Config, limit: int = 100, min_size_filter: Optional[Decimal] = None
) -> Optional[List[Dict]]:
    """Fetches recent public trades for a symbol from Bybit V5, validates data."""
    func_name = "fetch_recent_trades"; log_prefix = f"[{func_name}({symbol}, limit={limit})]"
    api_conf = config.api_config
    filter_log = f"(MinSize:{format_amount(exchange, symbol, min_size_filter) if min_size_filter else 'N/A'})"
    logger.debug(f"{log_prefix} Fetching {limit} trades {filter_log}...")

    # Bybit V5 limit is 1000 for public trades
    max_limit = 1000
    if limit > max_limit: logger.warning(f"{log_prefix} Clamping limit {limit} to {max_limit}."); limit = max_limit
    if limit <= 0: logger.warning(f"{log_prefix} Invalid limit {limit}. Using 100."); limit = 100

    try:
        market = exchange.market(symbol); category = _get_v5_category(market);
        if not category: logger.error(f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"); return None
        params = {'category': category}

        logger.debug(f"{log_prefix} Calling fetch_trades with limit={limit}, params={params}")
        trades_raw = exchange.fetch_trades(symbol, limit=limit, params=params)

        if not trades_raw: logger.debug(f"{log_prefix} No recent trades found."); return []

        processed_trades: List[Dict] = []; conversion_errors = 0; filtered_out_count = 0
        for trade in trades_raw:
            try:
                amount = safe_decimal_conversion(trade.get('amount')); price = safe_decimal_conversion(trade.get('price'))
                # Basic validation of core fields
                if not all([trade.get('id'), trade.get('timestamp'), trade.get('side'), price, amount]) or price <= 0 or amount <= 0:
                    conversion_errors += 1; continue

                # Apply size filter
                if min_size_filter is not None and amount < min_size_filter:
                    filtered_out_count += 1; continue

                # Calculate cost if missing or seems incorrect
                cost = safe_decimal_conversion(trade.get('cost'))
                if cost is None or abs(cost - (price * amount)) > api_conf.position_qty_epsilon * price:
                     cost = price * amount # Recalculate

                processed_trades.append({
                    'id': trade.get('id'), 'timestamp': trade.get('timestamp'), 'datetime': trade.get('datetime'),
                    'symbol': trade.get('symbol', symbol), 'side': trade.get('side'),
                    'price': price, 'amount': amount, 'cost': cost,
                    'takerOrMaker': trade.get('takerOrMaker'), 'fee': trade.get('fee'), # Include fee if available
                    'info': trade.get('info', {})
                })
            except Exception as proc_err:
                conversion_errors += 1; logger.warning(f"{Fore.YELLOW}{log_prefix} Error processing single trade: {proc_err}. Data: {trade}{Style.RESET_ALL}")

        if conversion_errors > 0: logger.warning(f"{Fore.YELLOW}{log_prefix} Skipped {conversion_errors} trades due to processing errors.{Style.RESET_ALL}")
        if filtered_out_count > 0: logger.debug(f"{log_prefix} Filtered {filtered_out_count} trades smaller than {min_size_filter}.")

        # Sort by timestamp descending (most recent first) - CCXT usually returns this way
        processed_trades.sort(key=lambda x: x['timestamp'], reverse=True)

        logger.info(f"{log_prefix} Fetched/processed {len(processed_trades)} trades {filter_log}.")
        return processed_trades

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API Error fetching trades: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching trades: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- update_limit_order ---
@retry_api_call(max_retries_override=1, initial_delay_override=0)
def update_limit_order(
    exchange: ccxt.bybit, symbol: str, order_id: str, config: Config, new_amount: Optional[Decimal] = None,
    new_price: Optional[Decimal] = None, new_trigger_price: Optional[Decimal] = None, # For conditional orders
    new_client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """Attempts to modify an existing open limit or conditional order on Bybit V5."""
    func_name = "update_limit_order"; log_prefix = f"[{func_name}(ID:{format_order_id(order_id)})]"
    api_conf = config.api_config

    # Check if anything is actually being changed
    if all(v is None for v in [new_amount, new_price, new_trigger_price, new_client_order_id]):
        logger.warning(f"{log_prefix} No changes provided (amount, price, trigger, client ID). Aborting update."); return None

    # Basic validation of new values
    if new_amount is not None and new_amount <= api_conf.position_qty_epsilon: logger.error(f"{Fore.RED}{log_prefix}: Invalid new amount ({new_amount})."); return None
    if new_price is not None and new_price <= Decimal("0"): logger.error(f"{Fore.RED}{log_prefix}: Invalid new price ({new_price})."); return None
    if new_trigger_price is not None and new_trigger_price <= Decimal("0"): logger.error(f"{Fore.RED}{log_prefix}: Invalid new trigger price ({new_trigger_price})."); return None

    log_changes = []
    if new_amount is not None: log_changes.append(f"Amt:{format_amount(exchange,symbol,new_amount)}")
    if new_price is not None: log_changes.append(f"Px:{format_price(exchange,symbol,new_price)}")
    if new_trigger_price is not None: log_changes.append(f"TrigPx:{format_price(exchange,symbol,new_trigger_price)}")
    if new_client_order_id is not None: log_changes.append("ClientOID")
    logger.info(f"{Fore.CYAN}{log_prefix}: Update {symbol} ({', '.join(log_changes)})...{Style.RESET_ALL}")

    try:
        if not exchange.has.get('editOrder'): logger.error(f"{Fore.RED}{log_prefix}: editOrder not supported by this CCXT version/exchange config.{Style.RESET_ALL}"); return None

        market = exchange.market(symbol); category = _get_v5_category(market);
        if not category: raise ValueError(f"Cannot determine category for {symbol}")

        # --- Prepare Params for edit_order ---
        # Note: edit_order in CCXT might not directly support all V5 amendment features.
        # It often cancels and replaces for complex changes. Check CCXT Bybit implementation.
        # V5 amend endpoint: /v5/order/amend
        # Required: category, symbol, orderId OR orderLinkId
        # Optional: qty, price, triggerPrice, sl/tp settings, orderLinkId
        edit_params: Dict[str, Any] = {'category': category}

        # Format new values if provided
        final_amount_str = format_amount(exchange, symbol, new_amount) if new_amount is not None else None
        final_price_str = format_price(exchange, symbol, new_price) if new_price is not None else None
        final_trigger_str = format_price(exchange, symbol, new_trigger_price) if new_trigger_price is not None else None

        # Add formatted values to params if they were provided
        if final_amount_str and final_amount_str != "Error": edit_params['qty'] = final_amount_str
        if final_price_str and final_price_str != "Error": edit_params['price'] = final_price_str
        if final_trigger_str and final_trigger_str != "Error": edit_params['triggerPrice'] = final_trigger_str

        if new_client_order_id:
            max_coid_len = 36; valid_coid = new_client_order_id[:max_coid_len]
            edit_params['orderLinkId'] = valid_coid # V5 amend uses orderLinkId
            if len(valid_coid) < len(new_client_order_id): logger.warning(f"{log_prefix} New Client OID truncated to orderLinkId: '{valid_coid}'")

        # Fetch current order to get side/type if needed by edit_order (CCXT specific)
        # current_order = fetch_order(exchange, symbol, order_id, config) # Use helper
        # if not current_order: raise ccxt.OrderNotFound(f"{log_prefix} Original order not found, cannot edit.")
        # status = current_order.get('status'); order_type = current_order.get('type')
        # if status != 'open': raise ccxt.InvalidOrder(f"{log_prefix}: Status is '{status}' (not 'open'). Cannot edit.")
        # --- edit_order might not need side/type if ID is sufficient ---

        logger.info(f"{Fore.CYAN}{log_prefix} Submitting update via edit_order. Params: {edit_params}{Style.RESET_ALL}")

        # Use CCXT's edit_order method
        # Pass None for parameters not being changed (amount, price)
        # CCXT might require side/type from original order, check its specific implementation
        updated_order = exchange.edit_order(
             id=order_id,
             symbol=symbol,
             # type=current_order['type'], # May need original type
             # side=current_order['side'], # May need original side
             amount=float(final_amount_str) if final_amount_str else None, # Pass float or None
             price=float(final_price_str) if final_price_str else None, # Pass float or None
             params=edit_params # Pass category and trigger price etc. here
        )

        if updated_order:
             # edit_order might return the amended order OR the cancel/replace new order ID
             new_id = updated_order.get('id', order_id); status_after = updated_order.get('status', '?'); new_client_oid_resp = updated_order.get('info', {}).get('orderLinkId', edit_params.get('orderLinkId', 'N/A'))
             logger.success(f"{Fore.GREEN}{log_prefix} Update OK. NewID:{format_order_id(new_id)}, Status:{status_after}, ClientOID:{new_client_oid_resp}{Style.RESET_ALL}")
             return updated_order
        else:
             # Should not happen if edit_order raises exceptions on failure
             logger.warning(f"{Fore.YELLOW}{log_prefix} edit_order returned no data. Check status manually.{Style.RESET_ALL}")
             return None

    except (ccxt.OrderNotFound, ccxt.InvalidOrder, ccxt.NotSupported, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol, ValueError) as e:
        logger.error(f"{Fore.RED}{log_prefix} Failed update: {type(e).__name__} - {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{log_prefix} Unexpected update error: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- fetch_position_risk_bybit_v5 ---
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def fetch_position_risk_bybit_v5(exchange: ccxt.bybit, symbol: str, config: Config) -> Optional[Dict[str, Any]]:
    """Fetches detailed risk metrics for the current position of a specific symbol using Bybit V5 logic."""
    func_name = "fetch_position_risk"; log_prefix = f"[{func_name}({symbol}, V5)]"
    api_conf = config.api_config
    logger.debug(f"{log_prefix} Fetching position risk...")
    default_risk = { 'symbol': symbol, 'side': api_conf.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0"), 'mark_price': None, 'liq_price': None, 'leverage': None, 'initial_margin': None, 'maint_margin': None, 'unrealized_pnl': None, 'imr': None, 'mmr': None, 'position_value': None, 'risk_limit_value': None, 'info': {} }
    try:
        market = exchange.market(symbol); market_id = market['id']; category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"); return default_risk

        params = {'category': category, 'symbol': market_id}; position_data: Optional[List[Dict]] = None; fetch_method_used = "N/A"

        # Prefer fetchPositionsRisk if available
        if exchange.has.get('fetchPositionsRisk'):
            try:
                logger.debug(f"{log_prefix} Using fetch_positions_risk...");
                position_data = exchange.fetch_positions_risk(symbols=[symbol], params=params)
                fetch_method_used = "fetchPositionsRisk"
            except (ccxt.NotSupported, ccxt.ExchangeError) as e: logger.warning(f"{log_prefix} fetch_positions_risk failed ({type(e).__name__}). Falling back."); position_data = None
        else: logger.debug(f"{log_prefix} fetchPositionsRisk not supported.")

        # Fallback to fetchPositions
        if position_data is None:
             if exchange.has.get('fetchPositions'):
                 logger.debug(f"{log_prefix} Falling back to fetch_positions...");
                 position_data = exchange.fetch_positions(symbols=[symbol], params=params)
                 fetch_method_used = "fetchPositions (Fallback)"
             else: logger.error(f"{Fore.RED}{log_prefix} No position fetch methods available.{Style.RESET_ALL}"); return default_risk

        if position_data is None: logger.error(f"{Fore.RED}{log_prefix} Failed fetch position data ({fetch_method_used}).{Style.RESET_ALL}"); return default_risk

        # Find the active One-Way position (index 0)
        active_pos_risk: Optional[Dict] = None
        for pos in position_data:
            pos_info = pos.get('info', {}); pos_symbol = pos_info.get('symbol'); pos_v5_side = pos_info.get('side', 'None'); pos_size_str = pos_info.get('size'); pos_idx = int(pos_info.get('positionIdx', -1))
            if pos_symbol == market_id and pos_v5_side != 'None' and pos_idx == 0:
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"))
                if pos_size is not None and abs(pos_size) > api_conf.position_qty_epsilon:
                    active_pos_risk = pos; logger.debug(f"{log_prefix} Found active One-Way pos risk data ({fetch_method_used})."); break

        if not active_pos_risk: logger.info(f"{log_prefix} No active One-Way position found."); return default_risk

        # --- Parse Risk Data ---
        # Prioritize standardized CCXT fields, fallback to 'info'
        try:
            info = active_pos_risk.get('info', {})
            size = safe_decimal_conversion(active_pos_risk.get('contracts', info.get('size')))
            entry_price = safe_decimal_conversion(active_pos_risk.get('entryPrice', info.get('avgPrice')))
            mark_price = safe_decimal_conversion(active_pos_risk.get('markPrice', info.get('markPrice')))
            liq_price = safe_decimal_conversion(active_pos_risk.get('liquidationPrice', info.get('liqPrice')))
            leverage = safe_decimal_conversion(active_pos_risk.get('leverage', info.get('leverage')))
            initial_margin = safe_decimal_conversion(active_pos_risk.get('initialMargin', info.get('positionIM'))) # IM for the position
            maint_margin = safe_decimal_conversion(active_pos_risk.get('maintenanceMargin', info.get('positionMM'))) # MM for the position
            pnl = safe_decimal_conversion(active_pos_risk.get('unrealizedPnl', info.get('unrealisedPnl')))
            imr = safe_decimal_conversion(active_pos_risk.get('initialMarginPercentage', info.get('imr'))) # Initial Margin Rate
            mmr = safe_decimal_conversion(active_pos_risk.get('maintenanceMarginPercentage', info.get('mmr'))) # Maintenance Margin Rate
            pos_value = safe_decimal_conversion(active_pos_risk.get('contractsValue', info.get('positionValue'))) # Value of the position
            risk_limit = safe_decimal_conversion(info.get('riskLimitValue')) # Current risk limit tier value

            side_std = active_pos_risk.get('side') # CCXT standard 'long'/'short'
            side_info = info.get('side') # Bybit 'Buy'/'Sell'
            position_side = api_conf.pos_long if side_std == 'long' or side_info == 'Buy' else (api_conf.pos_short if side_std == 'short' or side_info == 'Sell' else api_conf.pos_none)
            quantity = abs(size) if size is not None else Decimal("0.0")

            if position_side == api_conf.pos_none or quantity <= api_conf.position_qty_epsilon:
                 logger.info(f"{log_prefix} Parsed pos {symbol} negligible."); return default_risk

            # --- Log Parsed Risk Info ---
            log_color = Fore.GREEN if position_side == api_conf.pos_long else Fore.RED
            quote_curr = market.get('quote', api_conf.usdt_symbol)
            logger.info(f"{log_color}{log_prefix} Position Risk ({position_side}):{Style.RESET_ALL}")
            logger.info(f"  Qty:{format_amount(exchange, symbol, quantity)}, Entry:{format_price(exchange, symbol, entry_price)}, Mark:{format_price(exchange, symbol, mark_price)}")
            logger.info(f"  Liq:{format_price(exchange, symbol, liq_price)}, Lev:{leverage}x, uPNL:{format_price(exchange, quote_curr, pnl)}")
            logger.info(f"  IM:{format_price(exchange, quote_curr, initial_margin)}, MM:{format_price(exchange, quote_curr, maint_margin)}")
            logger.info(f"  IMR:{imr:.4% if imr else 'N/A'}, MMR:{mmr:.4% if mmr else 'N/A'}, Value:{format_price(exchange, quote_curr, pos_value)}")
            logger.info(f"  RiskLimitValue:{risk_limit or 'N/A'}")

            return {
                'symbol': symbol, 'side': position_side, 'qty': quantity, 'entry_price': entry_price,
                'mark_price': mark_price, 'liq_price': liq_price, 'leverage': leverage,
                'initial_margin': initial_margin, 'maint_margin': maint_margin, 'unrealized_pnl': pnl,
                'imr': imr, 'mmr': mmr, 'position_value': pos_value, 'risk_limit_value': risk_limit, 'info': info
            }
        except Exception as parse_err:
            logger.warning(f"{Fore.YELLOW}{log_prefix} Error parsing pos risk: {parse_err}. Data: {str(active_pos_risk)[:300]}{Style.RESET_ALL}"); return default_risk

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API Error fetching pos risk: {e}{Style.RESET_ALL}")
        if isinstance(e, ccxt.NetworkError): raise e # Allow retry
        return default_risk
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching pos risk: {e}{Style.RESET_ALL}", exc_info=True)
        return default_risk


# --- set_isolated_margin_bybit_v5 ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def set_isolated_margin_bybit_v5(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """Sets margin mode to ISOLATED for a specific symbol on Bybit V5 and sets leverage for it."""
    func_name = "set_isolated_margin"; log_prefix = f"[{func_name}({symbol}, {leverage}x)]"
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting ISOLATED mode...{Style.RESET_ALL}")
    ret_code = -1 # For tracking API response code
    if leverage <= 0: logger.error(f"{Fore.RED}{log_prefix} Leverage must be positive.{Style.RESET_ALL}"); return False
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}{log_prefix} Not a contract symbol ({category}).{Style.RESET_ALL}"); return False

        # --- Attempt via unified set_margin_mode first ---
        try:
            logger.debug(f"{log_prefix} Attempting via unified exchange.set_margin_mode...")
            # Pass category and leverage directly if supported by CCXT method
            exchange.set_margin_mode(marginMode='isolated', symbol=symbol, params={'category': category, 'leverage': leverage})
            logger.success(f"{Fore.GREEN}{log_prefix} Isolated mode & leverage {leverage}x set OK via unified call for {symbol}.{Style.RESET_ALL}")
            return True # Assume success if no exception
        except (ccxt.NotSupported, ccxt.ExchangeError, ccxt.ArgumentsRequired) as e_unified:
             # Log failure and proceed to V5 specific endpoint attempt
             logger.warning(f"{Fore.YELLOW}{log_prefix} Unified set_margin_mode failed: {e_unified}. Trying private V5 endpoint...{Style.RESET_ALL}")

        # --- Fallback to private V5 endpoint ---
        endpoint = 'private_post_v5_position_switch_isolated'
        if not hasattr(exchange, endpoint): logger.error(f"{Fore.RED}{log_prefix} CCXT lacks '{endpoint}'.{Style.RESET_ALL}"); return False

        params_switch = { 'category': category, 'symbol': market['id'], 'tradeMode': 1, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage) }
        logger.debug(f"{log_prefix} Calling {endpoint} with params: {params_switch}")
        response = getattr(exchange, endpoint)(params_switch); logger.debug(f"{log_prefix} Raw V5 switch response: {response}")
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', '').lower();

        if ret_code == 0:
            logger.success(f"{Fore.GREEN}{log_prefix} Switched {symbol} to ISOLATED with {leverage}x leverage via V5.{Style.RESET_ALL}"); return True
        # Code 110026: Margin mode is not modified (already isolated)
        # Code 34036: Already in the target mode (UTA?)
        elif ret_code in [110026, 34036] or "margin mode is not modified" in ret_msg:
            logger.info(f"{Fore.CYAN}{log_prefix} {symbol} already ISOLATED via V5 check. Confirming leverage...{Style.RESET_ALL}")
            # Explicitly call set_leverage again to ensure the leverage value is correct
            leverage_confirm_success = set_leverage(exchange, symbol, leverage, config)
            if leverage_confirm_success:
                 logger.success(f"{Fore.GREEN}{log_prefix} Leverage confirmed/set {leverage}x for ISOLATED {symbol}.{Style.RESET_ALL}"); return True
            else:
                 logger.error(f"{Fore.RED}{log_prefix} Failed leverage confirm/set after ISOLATED check.{Style.RESET_ALL}"); return False
        # Code 110020: Cannot switch mode with active positions/orders
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg:
            logger.error(f"{Fore.RED}{log_prefix} Cannot switch {symbol} to ISOLATED: active pos/orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"); return False
        else:
            # Raise unexpected V5 error codes
            raise ccxt.ExchangeError(f"Bybit API error switching isolated mode (V5): Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.BadSymbol, ValueError) as e:
        # Don't log expected "have position" error loudly again
        if not (isinstance(e, ccxt.ExchangeError) and ret_code == 110020):
            logger.warning(f"{Fore.YELLOW}{log_prefix} API/Input Error setting isolated margin: {e}{Style.RESET_ALL}")
        if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)): raise e # Allow retry
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error setting isolated margin: {e}{Style.RESET_ALL}", exc_info=True)
        return False


# --- Example Standalone Testing Block ---
if __name__ == "__main__":
    print(f"{Fore.YELLOW}{Style.BRIGHT}--- Bybit V5 Helpers Module Standalone Execution ---{Style.RESET_ALL}")
    print("Basic syntax checks only. Depends on external Config, logger, and bybit_utils.py.")
    # List defined functions (excluding internal ones starting with _)
    all_funcs = [name for name, obj in locals().items() if callable(obj) and not name.startswith('_') and name not in ['Config', 'AppConfig']]
    print(f"Found {len(all_funcs)} function definitions.")
    # Example: print(all_funcs)
    print(f"\n{Fore.GREEN}Basic syntax check passed.{Style.RESET_ALL}")
    print(f"Ensure PANDAS_AVAILABLE={PANDAS_AVAILABLE}, CCXT_AVAILABLE={CCXT_AVAILABLE}")

# --- END OF FILE bybit_helper_functions.py ---
EOF

# --- Create indicators.py (Using provided enhanced v1.1) ---
echo -e "${C_INFO} -> Generating indicators.py (v1.1 enhanced)${C_RESET}"
cat << 'EOF' > indicators.py
# --- START OF FILE indicators.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical Indicators Module (v1.1 - Fixed EVT SuperSmoother)

Provides functions to calculate various technical indicators, primarily leveraging the
`pandas_ta` library for efficiency and breadth. Includes standard indicators,
pivot points, and level calculations. Designed to work with pandas DataFrames
containing OHLCV data.

Key Features:
- Wrappers around `pandas_ta` for common indicators.
- Calculation of Standard and Fibonacci Pivot Points.
- Calculation of Support/Resistance levels based on pivots and Fibonacci retracements.
- Custom Ehlers Volumetric Trend (EVT) implementation with SuperSmoother.
- A master function (`calculate_all_indicators`) to compute indicators based on a config.
- Robust error handling and logging.
- Clear type hinting and documentation.

Assumes input DataFrame has columns: 'open', 'high', 'low', 'close', 'volume'
and a datetime index (preferably UTC).
"""

import logging
import sys
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
try:
    import pandas_ta as ta # type: ignore[import]
    PANDAS_TA_AVAILABLE = True
except ImportError:
    print("Error: 'pandas_ta' library not found. Please install it: pip install pandas_ta", file=sys.stderr)
    PANDAS_TA_AVAILABLE = False
    # sys.exit(1) # Optionally exit if pandas_ta is critical


# --- Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Basic setup if run standalone or before main logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default level

# --- Constants ---
MIN_PERIODS_DEFAULT = 50 # Default minimum number of data points for reliable calculations

# --- Pivot Point Calculations ---

def calculate_standard_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates standard pivot points for the *next* period based on HLC."""
    if not all(isinstance(v, (int, float)) and pd.notna(v) for v in [high, low, close]): # Use pd.notna
        logger.warning("Invalid input for standard pivot points (NaN or non-numeric).")
        return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for pivot calc.")
    pivots = {}
    try:
        pivot = (high + low + close) / 3.0; pivots['PP'] = pivot
        pivots['S1'] = (2 * pivot) - high; pivots['R1'] = (2 * pivot) - low
        pivots['S2'] = pivot - (high - low); pivots['R2'] = pivot + (high - low)
        pivots['S3'] = low - 2 * (high - pivot); pivots['R3'] = high + 2 * (pivot - low)
    except Exception as e: logger.error(f"Error calculating standard pivots: {e}", exc_info=True); return {}
    return pivots

def calculate_fib_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates Fibonacci pivot points for the *next* period based on HLC."""
    if not all(isinstance(v, (int, float)) and pd.notna(v) for v in [high, low, close]): # Use pd.notna
        logger.warning("Invalid input for Fibonacci pivot points (NaN or non-numeric)."); return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for Fib pivot calc.")
    fib_pivots = {}
    try:
        pivot = (high + low + close) / 3.0; fib_range = high - low
        if abs(fib_range) < 1e-9: logger.warning("Zero range, cannot calculate Fib Pivots accurately."); fib_pivots['PP'] = pivot; return fib_pivots
        fib_pivots['PP'] = pivot
        fib_pivots['S1'] = pivot - (0.382 * fib_range); fib_pivots['R1'] = pivot + (0.382 * fib_range)
        fib_pivots['S2'] = pivot - (0.618 * fib_range); fib_pivots['R2'] = pivot + (0.618 * fib_range)
        fib_pivots['S3'] = pivot - (1.000 * fib_range); fib_pivots['R3'] = pivot + (1.000 * fib_range)
    except Exception as e: logger.error(f"Error calculating Fib pivots: {e}", exc_info=True); return {}
    return fib_pivots

# --- Support / Resistance Level Calculation ---

def calculate_levels(df_period: pd.DataFrame, current_price: Optional[float] = None) -> Dict[str, Any]:
    """Calculates various support/resistance levels based on historical data."""
    levels: Dict[str, Any] = {"support": {}, "resistance": {}, "pivot": None, "fib_retracements": {}, "standard_pivots": {}, "fib_pivots": {}}
    required_cols = ['high', 'low', 'close']
    if df_period is None or df_period.empty or not all(col in df_period.columns for col in required_cols): logger.warning("Cannot calculate levels: Invalid DataFrame."); return levels
    standard_pivots, fib_pivots = {}, {}

    # Use previous candle's data for pivots
    if len(df_period) >= 2:
        try:
            prev_row = df_period.iloc[-2] # Use second to last row for previous candle HLC
            if not prev_row[required_cols].isnull().any():
                standard_pivots = calculate_standard_pivot_points(prev_row["high"], prev_row["low"], prev_row["close"])
                fib_pivots = calculate_fib_pivot_points(prev_row["high"], prev_row["low"], prev_row["close"])
            else: logger.warning("Previous candle data contains NaN, skipping pivot calculation.")
        except IndexError: logger.warning("IndexError calculating pivots (need >= 2 rows).")
        except Exception as e: logger.error(f"Error calculating pivots: {e}", exc_info=True)
    else: logger.warning("Cannot calculate pivots: Need >= 2 data points.")

    levels["standard_pivots"] = standard_pivots; levels["fib_pivots"] = fib_pivots
    levels["pivot"] = standard_pivots.get('PP') if standard_pivots else fib_pivots.get('PP')

    # Calculate Fib retracements over the whole period
    try:
        period_high = df_period["high"].max(); period_low = df_period["low"].min()
        if pd.notna(period_high) and pd.notna(period_low):
            period_diff = period_high - period_low
            if abs(period_diff) > 1e-9:
                levels["fib_retracements"] = {
                    "High": period_high,
                    "Fib 78.6%": period_low + period_diff*0.786,
                    "Fib 61.8%": period_low + period_diff*0.618,
                    "Fib 50.0%": period_low + period_diff*0.5,
                    "Fib 38.2%": period_low + period_diff*0.382,
                    "Fib 23.6%": period_low + period_diff*0.236,
                    "Low": period_low
                }
            else: logger.debug("Period range near zero, skipping Fib retracements.")
        else: logger.warning("Could not calculate Fib retracements due to NaN in period High/Low.")
    except Exception as e: logger.error(f"Error calculating Fib retracements: {e}", exc_info=True)

    # Classify levels relative to current price or pivot
    try:
        # Use current price if provided, otherwise use calculated pivot point
        cp = float(current_price) if current_price is not None and pd.notna(current_price) else levels.get("pivot")

        if cp is not None and pd.notna(cp):
            all_levels = {**{f"Std {k}": v for k, v in standard_pivots.items() if pd.notna(v)},
                          **{f"Fib {k}": v for k, v in fib_pivots.items() if k != 'PP' and pd.notna(v)},
                          **{k: v for k, v in levels["fib_retracements"].items() if pd.notna(v)}}
            for label, value in all_levels.items():
                if value < cp: levels["support"][label] = value
                elif value > cp: levels["resistance"][label] = value
        else: logger.debug("Cannot classify S/R relative to current price/pivot (price or pivot is None/NaN).")
    except Exception as e: logger.error(f"Error classifying S/R levels: {e}", exc_info=True)

    # Sort levels for readability
    levels["support"] = dict(sorted(levels["support"].items(), key=lambda item: item[1], reverse=True))
    levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda item: item[1]))
    return levels


# --- Custom Indicator Example ---

def calculate_vwma(close: pd.Series, volume: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series): return None
    if close.empty or volume.empty or len(close) != len(volume): return None # Basic validation
    if length <= 0: logger.error(f"VWMA length must be positive: {length}"); return None
    if len(close) < length: logger.debug(f"VWMA data length {len(close)} < period {length}. Result will have NaNs."); # Allow calculation

    try:
        pv = close * volume
        # Use min_periods=length to ensure enough data for the window sum
        cumulative_pv = pv.rolling(window=length, min_periods=length).sum()
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()
        # Avoid division by zero: replace 0 volume with NaN before dividing
        vwma = cumulative_pv / cumulative_vol.replace(0, np.nan)
        vwma.name = f"VWMA_{length}"
        return vwma
    except Exception as e:
        logger.error(f"Error calculating VWMA(length={length}): {e}", exc_info=True)
        return None

def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: Union[float, int]) -> pd.DataFrame:
    """
    Calculate Ehlers Volumetric Trend using VWMA and SuperSmoother filter.
    Adds columns: 'vwma_X', 'smooth_vwma_X', 'evt_trend_X', 'evt_buy_X', 'evt_sell_X'.
    """
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df.empty:
         logger.warning("EVT skipped: Missing columns or empty df.")
         return df
    if length <= 1 or multiplier <= 0:
        logger.warning(f"EVT skipped: Invalid params (len={length}, mult={multiplier}).")
        return df
    # Need at least length + 2 rows for smoother calculation
    if len(df) < length + 2:
        logger.warning(f"EVT skipped: Insufficient data rows ({len(df)}) for length {length}.")
        return df

    df_out = df.copy()
    vwma_col = f'vwma_{length}'
    smooth_col = f'smooth_vwma_{length}'
    trend_col = f'evt_trend_{length}'
    buy_col = f'evt_buy_{length}'
    sell_col = f'evt_sell_{length}'

    try:
        vwma = calculate_vwma(df_out['close'], df_out['volume'], length=length)
        if vwma is None or vwma.isnull().all():
            raise ValueError(f"VWMA calculation failed for EVT (length={length})")
        df_out[vwma_col] = vwma

        # SuperSmoother Filter Calculation (Corrected constants and implementation)
        # Constants based on Ehlers' formula
        arg = 1.414 * np.pi / length # Corrected sqrt(2) approx
        a = np.exp(-arg)
        b = 2 * a * np.cos(arg)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3

        # Initialize smoothed series & apply filter iteratively
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        # Use .values for potentially faster access if DataFrame is large
        vwma_valid = df_out[vwma_col].values

        # Prime the first two values using VWMA itself (simple initialization)
        if len(df_out) > 0 and pd.notna(vwma_valid[0]): smoothed.iloc[0] = vwma_valid[0]
        if len(df_out) > 1 and pd.notna(vwma_valid[1]): smoothed.iloc[1] = vwma_valid[1] # Simple init

        # Iterate starting from the third element (index 2)
        for i in range(2, len(df_out)):
            # Ensure current VWMA and previous smoothed values are valid
            if pd.notna(vwma_valid[i]):
                # Use previous smoothed value if valid, otherwise fallback (careful with fallback choice)
                # Using previous VWMA as fallback might introduce lag/noise
                sm1 = smoothed.iloc[i-1] if pd.notna(smoothed.iloc[i-1]) else vwma_valid[i-1]
                sm2 = smoothed.iloc[i-2] if pd.notna(smoothed.iloc[i-2]) else vwma_valid[i-2]
                # Only calculate if all inputs are valid numbers
                if pd.notna(sm1) and pd.notna(sm2):
                    smoothed.iloc[i] = c1 * vwma_valid[i] + c2 * sm1 + c3 * sm2

        df_out[smooth_col] = smoothed

        # Trend Determination
        mult_h = 1.0 + float(multiplier) / 100.0
        mult_l = 1.0 - float(multiplier) / 100.0
        shifted_smooth = df_out[smooth_col].shift(1)

        # Conditions - compare only where both current and previous smoothed values are valid
        valid_comparison = pd.notna(df_out[smooth_col]) & pd.notna(shifted_smooth)
        up_trend_cond = valid_comparison & (df_out[smooth_col] > shifted_smooth * mult_h)
        down_trend_cond = valid_comparison & (df_out[smooth_col] < shifted_smooth * mult_l)

        # Vectorized trend calculation using forward fill
        trend = pd.Series(np.nan, index=df_out.index, dtype=float) # Start with NaN
        trend[up_trend_cond] = 1.0   # Mark uptrend start
        trend[down_trend_cond] = -1.0 # Mark downtrend start

        # Forward fill the trend signal (1 or -1 persists), fill initial NaNs with 0 (neutral)
        df_out[trend_col] = trend.ffill().fillna(0).astype(int)

        # Buy/Sell Signal Generation (Trend Initiation)
        trend_shifted = df_out[trend_col].shift(1, fill_value=0) # Previous period's trend (fill start with 0)
        df_out[buy_col] = (df_out[trend_col] == 1) & (trend_shifted != 1)  # Trend becomes 1
        df_out[sell_col] = (df_out[trend_col] == -1) & (trend_shifted != -1) # Trend becomes -1

        logger.debug(f"Ehlers Volumetric Trend (len={length}, mult={multiplier}) calculated.")
        return df_out

    except Exception as e:
        logger.error(f"Error in EVT(len={length}, mult={multiplier}): {e}", exc_info=True)
        # Add NaN columns to original df to signal failure, maintain structure
        for col in [vwma_col, smooth_col, trend_col, buy_col, sell_col]:
            if col not in df_out.columns: df_out[col] = np.nan
        return df


# --- Master Indicator Calculation Function ---
def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Calculates enabled technical indicators using pandas_ta and custom functions."""
    if df is None or df.empty: logger.error("Input DataFrame empty."); return pd.DataFrame()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]; logger.error(f"Input DataFrame missing: {missing}."); return df.copy()

    df_out = df.copy()
    settings = config.get("indicator_settings", {})
    flags = config.get("analysis_flags", {})
    min_rows = settings.get("min_data_periods", MIN_PERIODS_DEFAULT)

    # Check for sufficient valid rows AFTER ensuring columns exist
    if len(df_out.dropna(subset=required_cols)) < min_rows:
        logger.warning(f"Insufficient valid rows ({len(df_out.dropna(subset=required_cols))}) < {min_rows}. Results may be NaN/inaccurate.")

    # --- Calculate Standard Indicators (using pandas_ta if available) ---
    if PANDAS_TA_AVAILABLE:
        atr_len = settings.get('atr_period', 14)
        if flags.get("use_atr", False) and atr_len > 0:
            try:
                logger.debug(f"Calculating ATR(length={atr_len})")
                df_out.ta.atr(length=atr_len, append=True) # Appends 'ATRr_X'
            except Exception as e: logger.error(f"Error calculating ATR({atr_len}): {e}", exc_info=False)

        # Add other pandas_ta indicators based on flags here...
        # Example: EMA
        # if flags.get("use_ema"):
        #    try:
        #         ema_s = settings.get('ema_short_period', 12); ema_l = settings.get('ema_long_period', 26)
        #         if ema_s > 0: df_out.ta.ema(length=ema_s, append=True)
        #         if ema_l > 0: df_out.ta.ema(length=ema_l, append=True)
        #    except Exception as e: logger.error(f"Error calculating EMA: {e}", exc_info=False)
    else:
        logger.warning("pandas_ta not available. Skipping standard indicators (ATR, etc.).")


    # --- Calculate Custom Strategy Indicators ---
    strategy_config = config.get('strategy_params', {}).get(config.get('strategy', {}).get('name', '').lower(), {})
    # Ehlers Volumetric Trend (Primary)
    if flags.get("use_evt"): # Generic flag or strategy-specific check
        try:
            # Get params from strategy_config first, fallback to general indicator_settings
            evt_len = strategy_config.get('evt_length', settings.get('evt_length', 7))
            evt_mult = strategy_config.get('evt_multiplier', settings.get('evt_multiplier', 2.5))
            if evt_len > 1 and evt_mult > 0:
                logger.debug(f"Calculating EVT(len={evt_len}, mult={evt_mult})")
                df_out = ehlers_volumetric_trend(df_out, evt_len, float(evt_mult))
            else: logger.warning(f"Invalid parameters for EVT (len={evt_len}, mult={evt_mult}), skipping.")
        except Exception as e: logger.error(f"Error calculating EVT: {e}", exc_info=True)

    # Example: Dual EVT Strategy specific logic (if needed)
    # if config.get('strategy',{}).get('name','').lower() == "dual_ehlers_volumetric":
    #      # ... calculate confirmation EVT ...

    logger.debug(f"Finished calculating indicators. Final DataFrame shape: {df_out.shape}")
    # Optional: remove duplicate columns if any arose
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    return df_out


# --- Example Standalone Usage ---
if __name__ == "__main__":
    print("-" * 60); print("--- Indicator Module Demo (v1.1) ---"); print("-" * 60)
    logger.setLevel(logging.DEBUG) # Set logger to debug for demo

    # Create dummy data (more realistic price movement)
    periods = 200
    np.random.seed(42) # for reproducibility
    returns = np.random.normal(loc=0.0001, scale=0.01, size=periods)
    prices = 55000 * np.exp(np.cumsum(returns)) # Start price 55000

    # Ensure OHLC are consistent
    data = {'timestamp': pd.date_range(start='2023-01-01', periods=periods, freq='H', tz='UTC')}
    df_test = pd.DataFrame(data).set_index('timestamp')
    df_test['open'] = prices[:-1]
    df_test['close'] = prices[1:]
    # Simulate High/Low relative to Open/Close
    high_factor = 1 + np.random.uniform(0, 0.005, periods-1)
    low_factor = 1 - np.random.uniform(0, 0.005, periods-1)
    df_test = df_test.iloc[:-1].copy() # Adjust size to match open/close
    df_test['high'] = df_test[['open', 'close']].max(axis=1) * high_factor
    df_test['low'] = df_test[['open', 'close']].min(axis=1) * low_factor
    # Ensure H >= O, H >= C and L <= O, L <= C
    df_test['high'] = df_test[['open', 'close', 'high']].max(axis=1)
    df_test['low'] = df_test[['open', 'close', 'low']].min(axis=1)
    df_test['volume'] = np.random.uniform(100, 2000, periods-1)

    print(f"Input shape: {df_test.shape}"); print(f"Input head:\n{df_test.head()}"); print(f"Input tail:\n{df_test.tail()}")

    # Example Config
    test_config = {
        "indicator_settings": {
            "min_data_periods": 50,
            "atr_period": 14,
            "evt_length": 7,
            "evt_multiplier": 2.5
        },
        "analysis_flags": {
            "use_atr": True,
            "use_evt": True
        },
        # These mimic the structure expected by the function
        "strategy_params": {'ehlers_volumetric': {'evt_length': 7, 'evt_multiplier': 2.5}},
        "strategy": {'name': 'ehlers_volumetric'}
    }

    df_results = calculate_all_indicators(df_test, test_config)
    print("-" * 60); print(f"Output shape: {df_results.shape}"); print(f"Output tail:\n{df_results.tail()}"); print("-" * 60)
    print(f"Output columns ({len(df_results.columns)}): {df_results.columns.tolist()}"); print("-" * 60)

    # Check for NaNs in the last row of added indicators
    added_cols = df_results.columns.difference(df_test.columns)
    last_row_nans = df_results[added_cols].iloc[-1].isnull().sum()
    print(f"NaNs in last row of added indicators ({len(added_cols)} cols): {last_row_nans}");
    print(f"Last row details:\n{df_results[added_cols].iloc[-1]}")

# --- END OF FILE indicators.py ---
EOF

# --- Create ehlers_volumetric_strategy.py (Using provided enhanced v1.3) ---
echo -e "${C_INFO} -> Generating ehlers_volumetric_strategy.py (v1.3 enhanced)${C_RESET}"
cat << 'EOF' > ehlers_volumetric_strategy.py
# --- START OF FILE ehlers_volumetric_strategy.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ehlers Volumetric Trend Strategy for Bybit V5 (v1.3 - Class, TP, Order Mgmt)

This script implements a trading strategy based on the Ehlers Volumetric Trend
indicator using the Bybit V5 API via CCXT. It leverages custom helper modules
for exchange interaction, indicator calculation, logging, and utilities.

Strategy Logic:
- Uses Ehlers Volumetric Trend (EVT) for primary entry signals.
- Enters LONG on EVT bullish trend initiation.
- Enters SHORT on EVT bearish trend initiation.
- Exits positions when the EVT trend reverses.
- Uses ATR-based stop-loss and take-profit orders (placed as reduce-only limit orders).
- Manages position size based on risk percentage.
- Includes error handling, retries, and rate limit awareness via helper modules.
- Encapsulated within an EhlersStrategy class.
"""

import os
import sys
import time
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP # Import ROUND_UP for TP
from typing import Optional, Dict, Tuple, Any, Literal

# Third-party libraries
try:
    import ccxt
except ImportError: print("FATAL: CCXT library not found.", file=sys.stderr); sys.exit(1)
try:
    import pandas as pd
except ImportError: print("FATAL: pandas library not found.", file=sys.stderr); sys.exit(1)
try:
    from dotenv import load_dotenv
except ImportError: print("Warning: python-dotenv not found. Cannot load .env file.", file=sys.stderr); load_dotenv = lambda: None # Dummy function

# --- Import Colorama for main script logging ---
try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print("Warning: 'colorama' library not found. Main script logs will not be colored.", file=sys.stderr)
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore


# --- Import Custom Modules ---
try:
    from neon_logger import setup_logger
    import bybit_helper_functions as bybit_helpers # Import the module itself
    import indicators as ind
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert # Sync SMS alert here
    )
    # Import config models
    from config_models import AppConfig, APIConfig, StrategyConfig # Import specific models needed
except ImportError as e:
    print(f"FATAL: Error importing helper modules: {e}", file=sys.stderr)
    print("Ensure all .py files (config_models, neon_logger, bybit_utils, bybit_helper_functions, indicators) are present.", file=sys.stderr)
    sys.exit(1)

# --- Load Environment Variables ---
# load_dotenv() # Called in main.py usually, after logger setup

# --- Logger Placeholder ---
# Logger configured in main block
logger: logging.Logger = logging.getLogger(__name__) # Get logger by name

# --- Strategy Class ---
class EhlersStrategy:
    """Encapsulates the Ehlers Volumetric Trend trading strategy logic."""

    def __init__(self, config: AppConfig): # Use AppConfig type hint
        self.app_config = config
        self.config = config.to_legacy_config_dict() # Convert to legacy dict for internal use IF NEEDED
                                                     # Better: Update class to use AppConfig directly
        self.api_config: APIConfig = config.api_config # Direct access
        self.strategy_config: StrategyConfig = config.strategy_config # Direct access

        self.symbol = self.api_config.symbol
        self.timeframe = self.strategy_config.timeframe
        self.exchange: Optional[ccxt.bybit] = None # Will be initialized (Sync version)
        self.bybit_helpers = bybit_helpers # Store module for access
        self.is_initialized = False
        self.is_running = False

        # Position State
        self.current_side: str = self.api_config.pos_none
        self.current_qty: Decimal = Decimal("0.0")
        self.entry_price: Optional[Decimal] = None
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None

        # Market details
        self.min_qty: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.price_tick: Optional[Decimal] = None

        logger.info(f"EhlersStrategy initialized for {self.symbol} on {self.timeframe}.")

    def _initialize(self) -> bool:
        """Connects to the exchange, validates market, sets config, fetches initial state."""
        logger.info(f"{Fore.CYAN}--- Strategy Initialization Phase ---{Style.RESET_ALL}")
        try:
            # Pass the main AppConfig object to helpers
            self.exchange = self.bybit_helpers.initialize_bybit(self.app_config)
            if not self.exchange: return False

            market_details = self.bybit_helpers.validate_market(self.exchange, self.symbol, self.app_config)
            if not market_details: return False
            self._extract_market_details(market_details)

            logger.info(f"Setting leverage for {self.symbol} to {self.strategy_config.leverage}x...")
            # Pass AppConfig
            if not self.bybit_helpers.set_leverage(self.exchange, self.symbol, self.strategy_config.leverage, self.app_config):
                 logger.critical(f"{Back.RED}Failed to set leverage.{Style.RESET_ALL}")
                 return False
            logger.success("Leverage set/confirmed.")

            # Set Position Mode (One-Way) - Optional but recommended for clarity
            pos_mode = self.strategy_config.default_position_mode
            logger.info(f"Attempting to set position mode to '{pos_mode}'...")
            # Pass AppConfig
            mode_set = self.bybit_helpers.set_position_mode_bybit_v5(self.exchange, self.symbol, pos_mode, self.app_config)
            if not mode_set:
                 logger.warning(f"{Fore.YELLOW}Could not explicitly set position mode to '{pos_mode}'. Ensure it's set correctly in Bybit UI.{Style.RESET_ALL}")
            else:
                 logger.info(f"Position mode confirmed/set to '{pos_mode}'.")

            logger.info("Fetching initial account state (position, orders, balance)...")
            # Pass AppConfig
            if not self._update_state():
                 logger.error("Failed to fetch initial state.")
                 return False

            logger.info(f"Initial Position: Side={self.current_side}, Qty={self.current_qty}")

            logger.info("Performing initial cleanup: cancelling existing orders...")
            # Pass AppConfig
            if not self._cancel_open_orders("Initialization Cleanup"):
                 logger.warning("Initial order cancellation failed or encountered issues.")

            self.is_initialized = True
            logger.success(f"{Fore.GREEN}--- Strategy Initialization Complete ---{Style.RESET_ALL}")
            return True

        except Exception as e:
            logger.critical(f"{Back.RED}Critical error during strategy initialization: {e}{Style.RESET_ALL}", exc_info=True)
            # Clean up exchange connection if partially initialized
            if self.exchange and hasattr(self.exchange, 'close'):
                try: self.exchange.close()
                except Exception: pass # Ignore errors during cleanup close
            return False

    def _extract_market_details(self, market: Dict):
        """Extracts and stores relevant market limits and precision."""
        limits = market.get('limits', {})
        precision = market.get('precision', {})

        self.min_qty = safe_decimal_conversion(limits.get('amount', {}).get('min'))
        amount_precision = precision.get('amount') # Number of decimal places for amount
        # Qty step is usually 10^-precision
        self.qty_step = (Decimal('1') / (Decimal('10') ** int(amount_precision))) if amount_precision is not None else None

        price_precision = precision.get('price') # Number of decimal places for price
        # Price tick is usually 10^-precision
        self.price_tick = (Decimal('1') / (Decimal('10') ** int(price_precision))) if price_precision is not None else None

        logger.info(f"Market Details Set: Min Qty={self.min_qty}, Qty Step={self.qty_step}, Price Tick={self.price_tick}")

    def _update_state(self) -> bool:
        """Fetches and updates the current position, balance, and open orders."""
        logger.debug("Updating strategy state...")
        try:
            # Fetch Position - Pass AppConfig
            pos_data = self.bybit_helpers.get_current_position_bybit_v5(self.exchange, self.symbol, self.app_config)
            if pos_data is None: logger.error("Failed to fetch position data."); return False

            self.current_side = pos_data['side']
            self.current_qty = pos_data['qty']
            self.entry_price = pos_data.get('entry_price') # Can be None if no position

            # Fetch Balance - Pass AppConfig
            balance_info = self.bybit_helpers.fetch_usdt_balance(self.exchange, self.app_config)
            if balance_info is None: logger.error("Failed to fetch balance data."); return False
            _, available_balance = balance_info # Unpack equity, available
            logger.info(f"Available Balance: {available_balance:.4f} {self.api_config.usdt_symbol}")

            # If not in position, reset tracked orders
            if self.current_side == self.api_config.pos_none:
                if self.sl_order_id or self.tp_order_id:
                     logger.debug("Not in position, clearing tracked SL/TP order IDs.")
                     self.sl_order_id = None
                     self.tp_order_id = None
            # Optional: If in position, verify tracked SL/TP orders still exist and are open
            # else: self._verify_open_sl_tp()

            logger.debug("State update complete.")
            return True

        except Exception as e:
            logger.error(f"Unexpected error during state update: {e}", exc_info=True)
            return False

    # Optional verification function
    # def _verify_open_sl_tp(self):
    #     """Checks if tracked SL/TP orders are still open."""
    #     if self.sl_order_id:
    #         sl_order = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.sl_order_id, self.app_config, order_filter='StopOrder') # Adjust filter if needed
    #         if not sl_order or sl_order.get('status') != 'open':
    #              logger.warning(f"Tracked SL order {self.sl_order_id} is no longer open/found. Clearing ID.")
    #              self.sl_order_id = None
    #     if self.tp_order_id:
    #          tp_order = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.tp_order_id, self.app_config, order_filter='Order') # Assuming TP is limit
    #          if not tp_order or tp_order.get('status') != 'open':
    #              logger.warning(f"Tracked TP order {self.tp_order_id} is no longer open/found. Clearing ID.")
    #              self.tp_order_id = None


    def _fetch_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Decimal]]:
        """Fetches OHLCV data and the latest ticker price."""
        logger.debug("Fetching market data...")
        # Pass AppConfig
        ohlcv_df = self.bybit_helpers.fetch_ohlcv_paginated(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            config=self.app_config, # Pass main config
            max_total_candles=self.strategy_config.ohlcv_limit
        )
        if ohlcv_df is None or not isinstance(ohlcv_df, pd.DataFrame) or ohlcv_df.empty:
            logger.warning("Could not fetch sufficient OHLCV data.")
            return None, None

        # Pass AppConfig
        ticker = self.bybit_helpers.fetch_ticker_validated(self.exchange, self.symbol, self.app_config)
        if ticker is None:
            logger.warning("Could not fetch valid ticker data.")
            return ohlcv_df, None # Return OHLCV if available, but no price

        current_price = ticker.get('last') # Already Decimal from helper
        if current_price is None:
            logger.warning("Ticker data retrieved but missing 'last' price.")
            return ohlcv_df, None

        logger.debug(f"Data fetched: {len(ohlcv_df)} candles, Last Price: {current_price}")
        return ohlcv_df, current_price

    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates indicators based on the config."""
        if ohlcv_df is None or ohlcv_df.empty: return None
        logger.debug("Calculating indicators...")
        # Prepare config dict expected by indicators module (if it uses dict)
        # Or pass AppConfig directly if indicators module supports it
        indicator_config_dict = {
            "indicator_settings": self.strategy_config.indicator_settings.model_dump(),
            "analysis_flags": self.strategy_config.analysis_flags.model_dump(),
            "strategy_params": self.strategy_config.strategy_params,
            "strategy": self.strategy_config.strategy_info
        }
        # df_with_indicators = ind.calculate_all_indicators(ohlcv_df, self.app_config) # If module accepts AppConfig
        df_with_indicators = ind.calculate_all_indicators(ohlcv_df, indicator_config_dict) # If module expects dict


        # Validate necessary columns exist
        evt_len = self.strategy_config.indicator_settings.evt_length
        atr_len = self.strategy_config.indicator_settings.atr_period
        evt_trend_col = f'evt_trend_{evt_len}'
        atr_col = f'ATRr_{atr_len}' # pandas_ta default name

        if df_with_indicators is None:
            logger.error("Indicator calculation returned None.")
            return None
        if evt_trend_col not in df_with_indicators.columns:
            logger.error(f"Required EVT trend column '{evt_trend_col}' not found after calculation.")
            return None
        if self.strategy_config.analysis_flags.use_atr and atr_col not in df_with_indicators.columns:
             logger.error(f"Required ATR column '{atr_col}' not found after calculation (use_atr is True).")
             return None

        logger.debug("Indicators calculated successfully.")
        return df_with_indicators

    def _generate_signals(self, df_ind: pd.DataFrame) -> Optional[Literal['buy', 'sell']]:
        """Generates trading signals based on the last indicator data point."""
        if df_ind is None or df_ind.empty: return None
        logger.debug("Generating trading signals...")
        try:
            latest = df_ind.iloc[-1]
            evt_len = self.strategy_config.indicator_settings.evt_length
            trend_col = f'evt_trend_{evt_len}'
            buy_col = f'evt_buy_{evt_len}'
            sell_col = f'evt_sell_{evt_len}'

            if not all(col in latest.index and pd.notna(latest[col]) for col in [trend_col, buy_col, sell_col]):
                 logger.warning(f"EVT signal columns missing or NaN in latest data: {latest[[trend_col, buy_col, sell_col]].to_dict()}")
                 return None

            buy_signal = latest[buy_col]
            sell_signal = latest[sell_col]

            # Return 'buy'/'sell' string consistent with helper functions
            if buy_signal:
                logger.info(f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}")
                return self.api_config.side_buy # 'buy'
            elif sell_signal:
                logger.info(f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}")
                return self.api_config.side_sell # 'sell'
            else:
                # Check for exit signal based purely on trend reversal
                current_trend = int(latest[trend_col])
                if self.current_side == self.api_config.pos_long and current_trend == -1:
                     logger.info(f"{Fore.YELLOW}EXIT LONG signal generated (Trend flipped Short).{Style.RESET_ALL}")
                     # Strategy logic handles exit separately, signal generator focuses on entry
                elif self.current_side == self.api_config.pos_short and current_trend == 1:
                     logger.info(f"{Fore.YELLOW}EXIT SHORT signal generated (Trend flipped Long).{Style.RESET_ALL}")
                     # Strategy logic handles exit separately

                logger.debug("No new entry signal generated.")
                return None

        except IndexError:
            logger.warning("IndexError generating signals (DataFrame likely too short).")
            return None
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
            return None

    def _calculate_sl_tp(self, df_ind: pd.DataFrame, side: Literal['buy', 'sell'], entry_price: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates initial stop-loss and take-profit prices."""
        if df_ind is None or df_ind.empty: return None, None
        logger.debug(f"Calculating SL/TP for {side} entry at {entry_price}...")
        try:
            atr_len = self.strategy_config.indicator_settings.atr_period
            atr_col = f'ATRr_{atr_len}'
            if atr_col not in df_ind.columns: logger.error(f"ATR column '{atr_col}' not found."); return None, None

            latest_atr = safe_decimal_conversion(df_ind.iloc[-1][atr_col])
            if latest_atr is None or latest_atr <= Decimal(0):
                logger.warning(f"Invalid ATR value ({latest_atr}) for SL/TP calculation.")
                return None, None # Require valid ATR

            # Stop Loss Calculation
            sl_multiplier = self.strategy_config.stop_loss_atr_multiplier
            sl_offset = latest_atr * sl_multiplier
            stop_loss_price = (entry_price - sl_offset) if side == self.api_config.side_buy else (entry_price + sl_offset)

            # --- Price Tick Adjustment ---
            # Adjust SL/TP to the nearest valid price tick
            if self.price_tick is None:
                 logger.warning("Price tick size unknown. Cannot adjust SL/TP precisely.")
                 sl_price_adjusted = stop_loss_price
            else:
                 # Round SL "away" from entry (more conservative)
                 rounding_mode = ROUND_DOWN if side == self.api_config.side_buy else ROUND_UP
                 sl_price_adjusted = (stop_loss_price / self.price_tick).quantize(Decimal('1'), rounding=rounding_mode) * self.price_tick

            # Ensure SL didn't cross entry after rounding
            if side == self.api_config.side_buy and sl_price_adjusted >= entry_price:
                 sl_price_adjusted = entry_price - self.price_tick if self.price_tick else entry_price * Decimal("0.999")
                 logger.warning(f"Adjusted Buy SL >= entry. Setting SL just below entry: {sl_price_adjusted}")
            elif side == self.api_config.side_sell and sl_price_adjusted <= entry_price:
                 sl_price_adjusted = entry_price + self.price_tick if self.price_tick else entry_price * Decimal("1.001")
                 logger.warning(f"Adjusted Sell SL <= entry. Setting SL just above entry: {sl_price_adjusted}")


            # Take Profit Calculation
            tp_multiplier = self.strategy_config.take_profit_atr_multiplier
            tp_price_adjusted = None
            if tp_multiplier > 0:
                tp_offset = latest_atr * tp_multiplier
                take_profit_price = (entry_price + tp_offset) if side == self.api_config.side_buy else (entry_price - tp_offset)

                # Ensure TP is logical relative to entry BEFORE rounding
                if side == self.api_config.side_buy and take_profit_price <= entry_price:
                    logger.warning(f"Calculated Buy TP ({take_profit_price}) <= entry ({entry_price}). Skipping TP.")
                elif side == self.api_config.side_sell and take_profit_price >= entry_price:
                    logger.warning(f"Calculated Sell TP ({take_profit_price}) >= entry ({entry_price}). Skipping TP.")
                else:
                    # Adjust TP to nearest tick, round "towards" entry (more conservative fill chance)
                    if self.price_tick is None:
                        logger.warning("Price tick size unknown. Cannot adjust TP precisely.")
                        tp_price_adjusted = take_profit_price
                    else:
                        rounding_mode = ROUND_DOWN if side == self.api_config.side_buy else ROUND_UP
                        tp_price_adjusted = (take_profit_price / self.price_tick).quantize(Decimal('1'), rounding=rounding_mode) * self.price_tick

                    # Ensure TP didn't cross entry after rounding
                    if side == self.api_config.side_buy and tp_price_adjusted <= entry_price:
                         tp_price_adjusted = entry_price + self.price_tick if self.price_tick else entry_price * Decimal("1.001")
                         logger.warning(f"Adjusted Buy TP <= entry. Setting TP just above entry: {tp_price_adjusted}")
                    elif side == self.api_config.side_sell and tp_price_adjusted >= entry_price:
                         tp_price_adjusted = entry_price - self.price_tick if self.price_tick else entry_price * Decimal("0.999")
                         logger.warning(f"Adjusted Sell TP >= entry. Setting TP just below entry: {tp_price_adjusted}")
            else:
                logger.info("Take Profit multiplier is zero or less. Skipping TP calculation.")


            logger.info(f"Calculated SL: {format_price(self.exchange, self.symbol, sl_price_adjusted)}, "
                        f"TP: {format_price(self.exchange, self.symbol, tp_price_adjusted) or 'None'} (ATR: {latest_atr:.4f})")
            return sl_price_adjusted, tp_price_adjusted

        except IndexError: logger.warning("IndexError calculating SL/TP."); return None, None
        except Exception as e: logger.error(f"Error calculating SL/TP: {e}", exc_info=True); return None, None

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """Calculates position size based on risk percentage and stop-loss distance."""
        logger.debug("Calculating position size...")
        try:
            # Pass AppConfig
            balance_info = self.bybit_helpers.fetch_usdt_balance(self.exchange, self.app_config)
            if balance_info is None: logger.error("Cannot calc size: Failed fetch balance."); return None
            _, available_balance = balance_info

            if available_balance is None or available_balance <= Decimal("0"):
                logger.error("Cannot calculate position size: Zero or invalid available balance.")
                return None

            risk_amount_usd = available_balance * self.strategy_config.risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)

            if price_diff <= Decimal("0"):
                logger.error(f"Cannot calculate size: Entry price ({entry_price}) and SL price ({stop_loss_price}) invalid or equal.")
                return None

            # Calculate size based on risk amount and price difference per unit
            # For linear contracts (Value = Amount * Price), Size = Risk / PriceDiff
            # For inverse contracts (Value = Amount / Price), need careful derivation
            market = self.exchange.market(self.symbol)
            is_inverse = market.get('inverse', False)

            position_size_base: Decimal
            if is_inverse:
                 # Risk = Size * abs(1/Entry - 1/SL) => Size = Risk / abs(1/Entry - 1/SL)
                 if entry_price <= 0 or stop_loss_price <= 0: raise ValueError("Prices must be positive for inverse size calc.")
                 size_denominator = abs(Decimal(1)/entry_price - Decimal(1)/stop_loss_price)
                 if size_denominator <= 0: raise ValueError("Inverse size denominator is zero.")
                 position_size_base = risk_amount_usd / size_denominator
            else: # Linear contract
                 position_size_base = risk_amount_usd / price_diff

            # Apply market precision/step size constraints
            if self.qty_step is None:
                 logger.warning("Quantity step size unknown, cannot adjust size precisely.")
                 position_size_adjusted = position_size_base # Use raw value
            else:
                 # Round down to the nearest step size increment
                 position_size_adjusted = (position_size_base // self.qty_step) * self.qty_step

            if position_size_adjusted <= Decimal(0):
                 logger.warning(f"Adjusted position size is zero or negative. Step: {self.qty_step}, Orig: {position_size_base}")
                 return None

            if self.min_qty is not None and position_size_adjusted < self.min_qty:
                logger.warning(f"Calculated size ({position_size_adjusted}) < Min Qty ({self.min_qty}). Cannot trade this size.")
                # Option: Round up to min_qty if desired, but this increases risk.
                # position_size_adjusted = self.min_qty
                # logger.warning(f"Adjusting size up to Min Qty ({self.min_qty}). Risk will be higher.")
                return None # Default: Don't trade if calculated size is too small

            logger.info(f"Calculated position size: {format_amount(self.exchange, self.symbol, position_size_adjusted)} "
                        f"(Risk: {risk_amount_usd:.2f} USDT, Balance: {available_balance:.2f} USDT)")
            return position_size_adjusted

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    def _cancel_open_orders(self, reason: str = "Strategy Action") -> bool:
        """Cancels tracked SL and TP orders."""
        cancelled_sl, cancelled_tp = True, True # Assume success if no ID tracked
        all_success = True

        # Determine order filter based on placement type
        sl_filter = 'StopOrder' if not self.strategy_config.place_tpsl_as_limit else 'Order'
        tp_filter = 'Order' # TP is always limit

        if self.sl_order_id:
            logger.info(f"Cancelling existing SL order {format_order_id(self.sl_order_id)} ({reason})...")
            try:
                # Pass AppConfig and filter
                cancelled_sl = self.bybit_helpers.cancel_order(self.exchange, self.symbol, self.sl_order_id, self.app_config, order_filter=sl_filter)
                if cancelled_sl: logger.info("SL order cancelled successfully or already gone.")
                else: logger.warning("Failed attempt to cancel SL order."); all_success = False
            except Exception as e:
                 logger.error(f"Error cancelling SL order {self.sl_order_id}: {e}", exc_info=True)
                 cancelled_sl = False; all_success = False
            finally:
                 self.sl_order_id = None # Always clear tracked ID after attempt

        if self.tp_order_id:
             logger.info(f"Cancelling existing TP order {format_order_id(self.tp_order_id)} ({reason})...")
             try:
                 # Pass AppConfig and filter
                 cancelled_tp = self.bybit_helpers.cancel_order(self.exchange, self.symbol, self.tp_order_id, self.app_config, order_filter=tp_filter)
                 if cancelled_tp: logger.info("TP order cancelled successfully or already gone.")
                 else: logger.warning("Failed attempt to cancel TP order."); all_success = False
             except Exception as e:
                  logger.error(f"Error cancelling TP order {self.tp_order_id}: {e}", exc_info=True)
                  cancelled_tp = False; all_success = False
             finally:
                  self.tp_order_id = None # Always clear tracked ID after attempt

        return all_success # Return overall success

    def _handle_exit(self, df_ind: pd.DataFrame) -> bool:
        """Checks exit conditions and closes the position if necessary."""
        if self.current_side == self.api_config.pos_none: return False # Not in position

        logger.debug("Checking exit conditions...")
        should_exit = False
        exit_reason = ""
        try:
            evt_len = self.strategy_config.indicator_settings.evt_length
            latest_trend = df_ind.iloc[-1].get(f'evt_trend_{evt_len}')

            if latest_trend is not None:
                latest_trend = int(latest_trend) # Ensure integer comparison
                if self.current_side == self.api_config.pos_long and latest_trend == -1:
                    should_exit = True; exit_reason = "EVT Trend flipped Short"
                elif self.current_side == self.api_config.pos_short and latest_trend == 1:
                    should_exit = True; exit_reason = "EVT Trend flipped Long"
            else:
                logger.warning("Cannot determine latest EVT trend for exit check.")

            # --- Add check for SL/TP Hit ---
            # This requires fetching order status, adds latency. Only do if essential.
            # Example (needs refinement based on order types):
            # if not should_exit and self.sl_order_id:
            #    sl_filter = 'StopOrder' if not self.strategy_config.place_tpsl_as_limit else 'Order'
            #    sl_status = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.sl_order_id, self.app_config, order_filter=sl_filter)
            #    if sl_status and sl_status.get
