Okay, let's enhance the introductory text for better clarity, emphasis, and flow, and then integrate it with the complete, improved bash script.

---

### **Enhanced Project Setup Script for Bybit EVT Strategy (v1.1)**

This document provides an enhanced bash script designed to rapidly scaffold a Python project for a trading bot operating on the Bybit exchange (specifically targeting the V5 API). The foundational trading logic implemented is based on the Ehlers Volumetric Trend (EVT) indicator.

**Purpose:** To automate the tedious initial setup tasks, allowing developers to focus quickly on refining the strategy and core bot logic.

**Automated Actions:**

1.  **Directory Structure:** Creates a clean, organized project folder named `bybit_evt_strategy`.
2.  **File Population:** Generates essential Python modules (`config_models.py`, `neon_logger.py`, `bybit_utils.py`, `bybit_helper_functions.py`, `indicators.py`, `ehlers_volumetric_strategy.py`, `main.py`) pre-populated with robust starting code.
3.  **Configuration Files:** Creates necessary configuration scaffolding (`.env` for secrets, `.gitignore` for version control hygiene, `requirements.txt` for dependencies).
4.  **Documentation:** Generates a basic `README.md` to guide further development.
5.  **Version Control:** Initializes a local Git repository and makes an initial commit of the generated structure.

**Prerequisites:**

*   A Unix-like environment (Linux, macOS, WSL on Windows) with the Bash shell.
*   **Git:** Must be installed and accessible in your PATH.
*   **Python 3.x:** Required for *running* the generated Python bot later, not for executing this setup script itself. (Python 3.8+ recommended for some type hinting features used).

**Key Features of the Generated Project Skeleton:**

*   **Robust Configuration:** Leverages Pydantic V2 for defining, validating, and loading application settings from environment variables and a `.env` file, ensuring type safety and clear defaults.
*   **Modular Design:** Promotes maintainability by separating concerns into distinct Python modules (configuration, logging, exchange utilities, API helpers, indicators, strategy logic, main entry point).
*   **Asynchronous Core:** Built around Python's `asyncio` for efficient handling of I/O-bound operations like API calls and potential WebSocket connections.
*   **Strategy Foundation:** Provides a functional, albeit basic, implementation of the Ehlers Volumetric Trend strategy within `ehlers_volumetric_strategy.py`.
*   **Exchange Interaction:** Includes well-structured helper functions in `bybit_helper_functions.py` for interacting with the Bybit V5 API via the `ccxt` library, incorporating automatic retries and error handling.
*   **Customizable Logging:** Uses a dedicated `neon_logger.py` module for setting up flexible, colorized console logging and optional file logging with rotation.
*   **Git Ready:** The project is immediately initialized as a Git repository, ready for tracking changes and collaboration.

**<0xF0><0x9F><0x9A><0xA7> Crucial Steps & Warnings <0xF0><0x9F><0x9A><0xA7>**

1.  <0xE2><0x9A><0xA0><0xEF><0xB8><0x8F> **API Key Security (MOST IMPORTANT):**
    *   The generated `.env` file contains **PLACEHOLDER** API keys (`YOUR_API_KEY_PLACEHOLDER`, `YOUR_API_SECRET_PLACEHOLDER`).
    *   **YOU ABSOLUTELY MUST EDIT `.env` AND REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL BYBIT API KEY AND SECRET** before attempting to run the Python bot.
    *   Generate API keys via the Bybit website (Testnet recommended for initial development).
    *   Ensure keys have permissions for `Orders` and `Positions` (Read & Write) under the Unified Trading Account (UTA) scope for full functionality.
    *   **NEVER COMMIT YOUR ACTUAL API KEYS TO VERSION CONTROL.** The provided `.gitignore` correctly excludes the `.env` file by default – **DO NOT REMOVE THIS EXCLUSION**.

2.  <0xF0><0x9F><0x9A><0xAB> **Execution Location & Overwriting Prevention:**
    *   Execute this setup script (`create_project.sh`) **only** in the parent directory where you want the new `bybit_evt_strategy` project folder to be created.
    *   The script checks if the target project directory (`bybit_evt_strategy`) already exists in the current location. It will **exit** if it does to prevent accidental data loss. Manually remove or rename any conflicting directory before running the script.

3.  <0xF0><0x9F><0xAA><0xB0> **Python Virtual Environment (Highly Recommended):**
    *   After the script completes, navigate into the newly created `bybit_evt_strategy` directory.
    *   **Before installing dependencies**, create and activate a Python virtual environment. This isolates project packages and avoids conflicts. Commands are provided in the script's final output and the generated `README.md`.

4.  <0xF0><0x9F><0x93><0x9D> **Git Configuration:**
    *   The script initializes a *local* Git repository.
    *   It includes commented-out example commands in its final output showing how to configure your Git user name and email *specifically for this repository* if it differs from your global Git configuration. This is important for correct commit attribution.

5.  <0xE2><0x9A><0xA1><0xEF><0xB8><0x8F> **Remote Repository:**
    *   This script does **not** automatically create or link to a remote repository (e.g., on GitHub, GitLab, Bitbucket).
    *   You will need to create a remote repository manually on your preferred platform.
    *   Follow the example commands (provided in the script's final output) to link your local repository to the remote and push the initial commit.

---

### **The Bash Script (`create_project.sh`)**

```bash
#!/bin/bash
# Script to create the directory structure, populate files for the Bybit EVT strategy bot,
# and initialize a new Git repository.
# Version: 1.1

# --- Safety Settings ---
# Exit immediately if a command exits with a non-zero status. Crucial for preventing partial setup on error.
set -e
# Treat unset variables as an error when substituting. Uncomment cautiously if needed, ensure all vars are defined.
# set -u

# --- Configuration ---
PROJECT_DIR="bybit_evt_strategy"
# These GIT variables are primarily used for generating instructions in the final output.
# Replace placeholders here OR update the final instructions manually if preferred.
GIT_USER_NAME="YourGitHubUsername"       # Example: Replace with your actual GitHub/GitLab username
GIT_USER_EMAIL="your_email@example.com"  # Example: Replace with your actual email used for Git commits
GIT_REMOTE_EXAMPLE="git@github.com:${GIT_USER_NAME}/${PROJECT_DIR}.git" # Example SSH URL

# --- ANSI Color Codes for Output ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_INFO='\033[36m'    # Cyan
C_SUCCESS='\033[32m' # Green
C_WARN='\033[33m'    # Yellow
C_ERROR='\033[91m'   # Bright Red
C_DIM='\033[2m'      # Dim text

# --- Helper Function for Progress ---
step_counter=0
total_steps=12 # Approximate number of major steps
progress() {
    step_counter=$((step_counter + 1))
    echo -e "\n${C_INFO}${C_BOLD}[Step ${step_counter}/${total_steps}] $1${C_RESET}"
}

# --- Pre-flight Checks ---
echo -e "${C_INFO}${C_BOLD}🚀 Starting Project Setup: ${PROJECT_DIR}${C_RESET}"
echo -e "${C_DIM}--------------------------------------------------${C_RESET}"

progress "Checking Prerequisites..."

# Check if Git is installed
if ! command -v git &> /dev/null; then
  echo -e "${C_ERROR}❌ Error: 'git' command not found. Please install Git.${C_RESET}"
  exit 1
else
  echo -e "${C_SUCCESS}✅ Git found: $(git --version)${C_RESET}"
fi

# Safety Check: Prevent overwriting existing directory
if [ -d "$PROJECT_DIR" ]; then
  echo -e "${C_ERROR}❌ Error: Directory '${PROJECT_DIR}' already exists in the current location ($(pwd)).${C_RESET}"
  echo -e "${C_WARN}👉 Please remove or rename the existing directory before running this script.${C_RESET}"
  exit 1
else
  echo -e "${C_SUCCESS}✅ Target directory '${PROJECT_DIR}' is available.${C_RESET}"
fi

# --- Directory Creation ---
progress "Creating Project Directory Structure..."
mkdir -p "$PROJECT_DIR"
echo -e "${C_SUCCESS}✅ Created directory: ${PROJECT_DIR}${C_RESET}"
cd "$PROJECT_DIR" # Change into the project directory for subsequent file creation
echo -e "${C_DIM}   -> Changed working directory to: $(pwd)${C_RESET}"


# --- File Generation ---
progress "Generating Python Source Files..."

# --- Create config_models.py ---
echo -e "${C_DIM}   -> Generating config_models.py${C_RESET}"
# Use single quotes around 'EOF' to prevent shell variable expansion inside the heredoc
cat << 'EOF' > config_models.py
# config_models.py
"""
Pydantic Models for Application Configuration using pydantic-settings (v2).

Loads configuration from environment variables and/or a .env file.
Provides type validation, default values, and structure for the trading bot settings.
"""

import logging
import os # For environment variable access during load
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
    FilePath, # Consider if needed later
    DirectoryPath, # Consider if needed later
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Define Enums for Bybit V5 API parameters (improves type safety) ---
# These can be used directly in type hints and validated by Pydantic.

class Category(str, Enum):
    LINEAR = "linear"
    INVERSE = "inverse"
    SPOT = "spot"
    OPTION = "option"

class PositionIdx(int, Enum):
    """ 0: One-Way Mode, 1: Buy side of Hedge Mode, 2: Sell side of Hedge Mode """
    ONE_WAY = 0
    HEDGE_BUY = 1
    HEDGE_SELL = 2

class OrderFilter(str, Enum):
    ORDER = "Order"
    STOP_ORDER = "StopOrder"
    TP_SL_ORDER = "tpslOrder" # Unified TP/SL order type
    TAKE_PROFIT = "TakeProfit" # Can be used in stopOrderType
    STOP_LOSS = "StopLoss"     # Can be used in stopOrderType
    # Add other potential filters if needed

class Side(str, Enum):
    BUY = "Buy"
    SELL = "Sell"

class TimeInForce(str, Enum):
    GTC = "GTC" # GoodTillCancel
    IOC = "IOC" # ImmediateOrCancel
    FOK = "FOK" # FillOrKill
    POST_ONLY = "PostOnly" # For Limit orders only

class TriggerBy(str, Enum):
    LAST_PRICE = "LastPrice"
    MARK_PRICE = "MarkPrice"
    INDEX_PRICE = "IndexPrice"

class TriggerDirection(int, Enum):
    """ Trigger direction for conditional orders: 1: Rise, 2: Fall """
    RISE = 1
    FALL = 2

class OrderType(str, Enum):
    LIMIT = "Limit"
    MARKET = "Market"

class StopLossTakeProfitMode(str, Enum):
    """ Position TP/SL mode: Full takes the entire position size """
    FULL = "Full"
    PARTIAL = "Partial" # Not typically used for bot's TP/SL orders, more for position setting

class AccountType(str, Enum):
    UNIFIED = "UNIFIED"
    CONTRACT = "CONTRACT" # Deprecated/Legacy
    SPOT = "SPOT" # Legacy Spot

# --- Configuration Models ---

class APIConfig(BaseModel):
    """Configuration for Bybit API Connection and Market Defaults."""
    exchange_id: Literal["bybit"] = Field("bybit", description="CCXT Exchange ID (fixed)")
    api_key: Optional[str] = Field(None, description="Bybit API Key (Loaded from env/file)")
    api_secret: Optional[str] = Field(None, description="Bybit API Secret (Loaded from env/file)")
    testnet_mode: bool = Field(True, description="Use Bybit Testnet environment")
    default_recv_window: PositiveInt = Field(
        10000, ge=100, le=60000, description="API request validity window in milliseconds (100-60000)"
    )

    # Market & Symbol Defaults
    symbol: str = Field("BTC/USDT:USDT", description="Primary trading symbol (e.g., BTC/USDT:USDT for linear swap)")
    category: Category = Field(Category.LINEAR, description="Primary symbol category (linear, inverse, spot, option)")
    usdt_symbol: str = Field("USDT", description="Quote currency for balance reporting (usually USDT)")
    # expected_market_type: Literal["swap", "spot", "option", "future"] = Field(
    #     "swap", description="Expected CCXT market type for validation (e.g., 'swap')"
    # ) # Less critical with explicit category
    # expected_market_logic: Literal["linear", "inverse"] = Field(
    #     "linear", description="Expected market logic for derivative validation ('linear' or 'inverse')"
    # ) # Less critical with explicit category

    # Retry & Rate Limit Defaults
    retry_count: NonNegativeInt = Field(
        3, description="Default number of retries for API calls (0 disables retries)"
    )
    retry_delay_seconds: PositiveFloat = Field(
        2.0, gt=0, description="Default base delay (seconds) for API retries (exponential backoff applies)"
    )

    # Fee Rates (Important for accurate calculations, **VERIFY THESE FOR YOUR ACCOUNT TIER**)
    maker_fee_rate: Decimal = Field(
        Decimal("0.0002"), ge=0, description="Maker fee rate (e.g., 0.0002 for 0.02%)"
    )
    taker_fee_rate: Decimal = Field(
        Decimal("0.00055"), ge=0, description="Taker fee rate (e.g., 0.00055 for 0.055%)"
    )

    # Order Execution Defaults & Helpers
    default_slippage_pct: Decimal = Field(
        Decimal("0.005"), # 0.5%
        gt=0,
        le=Decimal("0.1"), # Max 10% sanity check
        description="Default max allowable slippage % for market order pre-flight checks vs OB spread (e.g., 0.005 for 0.5%)",
    )
    position_qty_epsilon: Decimal = Field(
        Decimal("1e-9"), # Small value for comparing position sizes near zero
        gt=0,
        description="Small tolerance value for floating point quantity comparisons (e.g., treating size < epsilon as zero)",
    )
    shallow_ob_fetch_depth: PositiveInt = Field(
        5, ge=1, le=50, description="Order book depth for quick spread/slippage check (e.g., 5)"
    )
    order_book_fetch_limit: PositiveInt = Field(
        25, ge=1, le=1000, description="Default depth for fetching L2 order book (e.g., 25, 50)"
    )

    # Internal Constants (Mapped to Enums for consistency)
    pos_none: Literal["NONE"] = "NONE"
    pos_long: Literal["LONG"] = "LONG"
    pos_short: Literal["SHORT"] = "SHORT"
    side_buy: Side = Side.BUY
    side_sell: Side = Side.SELL

    @field_validator('api_key', 'api_secret', mode='before') # mode='before' to catch env var directly
    @classmethod
    def check_not_placeholder(cls, v: Optional[str], info) -> Optional[str]:
        """Warns if API key/secret looks like a placeholder."""
        placeholder_key = "YOUR_API_KEY_PLACEHOLDER"
        placeholder_secret = "YOUR_API_SECRET_PLACEHOLDER"
        field_name = info.field_name

        if v and isinstance(v, str):
            if field_name == 'api_key' and v == placeholder_key:
                print(f"\033[93mWarning [APIConfig]: Field '{field_name}' is using the default placeholder value. Please update in '.env'.\033[0m")
            elif field_name == 'api_secret' and v == placeholder_secret:
                 print(f"\033[93mWarning [APIConfig]: Field '{field_name}' is using the default placeholder value. Please update in '.env'.\033[0m")
            elif "PLACEHOLDER" in v.upper(): # Catch other potential placeholders
                 print(f"\033[93mWarning [APIConfig]: Field '{field_name}' might contain a placeholder: '{v[:10]}...'\033[0m")
        return v

    @field_validator('symbol', mode='before')
    @classmethod
    def check_and_format_symbol(cls, v: Any) -> str:
        """Validates and standardizes the symbol format."""
        if not isinstance(v, str) or not v:
             raise ValueError("Symbol must be a non-empty string")
        # Basic check for common derivative format (e.g., BTC/USDT:USDT) or spot (BTC/USDT)
        symbol_upper = v.strip().upper()
        if ":" not in symbol_upper and "/" not in symbol_upper:
            raise ValueError(f"Potentially invalid symbol format: '{v}'. Expected format like 'BTC/USDT:USDT' or 'BTC/USDT'.")
        # CCXT handles exact format, but uppercase is standard
        return symbol_upper

    @model_validator(mode='after')
    def check_api_keys_if_not_testnet(self) -> 'APIConfig':
        """Warns if API keys are missing when not in testnet mode."""
        if not self.testnet_mode and (not self.api_key or not self.api_secret or "PLACEHOLDER" in str(self.api_key) or "PLACEHOLDER" in str(self.api_secret)):
            print("\033[91mCRITICAL [APIConfig]: API Key/Secret missing or placeholders detected while Testnet mode is DISABLED. Live trading likely impossible.\033[0m")
            # Consider raising ValueError here if keys are strictly mandatory for any operation
        return self

    @model_validator(mode='after')
    def check_symbol_category_consistency(self) -> 'APIConfig':
        """Basic check for potential symbol/category mismatch."""
        symbol = self.symbol
        category = self.category

        is_spot = "/" in symbol and ":" not in symbol
        is_linear = ":USDT" in symbol or ":USDC" in symbol
        is_inverse = ":" in symbol and not is_linear and not is_spot # Basic guess

        if category == Category.SPOT and not is_spot:
             print(f"\033[93mWarning [APIConfig]: Category is SPOT, but symbol '{symbol}' doesn't look like a typical spot symbol (e.g., BTC/USDT).\033[0m")
        elif category == Category.LINEAR and not is_linear:
             print(f"\033[93mWarning [APIConfig]: Category is LINEAR, but symbol '{symbol}' doesn't look like a typical linear contract (e.g., BTC/USDT:USDT).\033[0m")
        elif category == Category.INVERSE and not is_inverse:
             print(f"\033[93mWarning [APIConfig]: Category is INVERSE, but symbol '{symbol}' doesn't look like a typical inverse contract.\033[0m")
        # Option check is omitted for brevity

        return self


class IndicatorSettings(BaseModel):
    """Parameters for Technical Indicator Calculations."""
    min_data_periods: PositiveInt = Field(
        100, ge=20, description="Minimum historical candles needed for reliable indicator calculations (e.g., 100)"
    )
    # Ehlers Volumetric specific
    evt_length: PositiveInt = Field(
        7, gt=1, description="Period length for EVT indicator (must be > 1)"
    )
    evt_multiplier: PositiveFloat = Field(
        2.5, gt=0, description="Multiplier for EVT bands calculation (must be > 0, represents percentage, e.g., 2.5 means 2.5%)"
    )
    # ATR specific (often used for stop loss)
    atr_period: PositiveInt = Field(
        14, gt=0, description="Period length for ATR indicator (must be > 0)"
    )
    # Add other indicator parameters here if needed
    # rsi_period: PositiveInt = Field(14, ...)
    # macd_fast: PositiveInt = Field(12, ...)


class AnalysisFlags(BaseModel):
    """Flags to Enable/Disable Specific Indicator Calculations or Features."""
    use_evt: bool = Field(True, description="Enable Ehlers Volumetric Trend calculation and signaling")
    use_atr: bool = Field(True, description="Enable ATR calculation (primarily for stop loss)")
    # Add other flags here
    # use_rsi: bool = Field(False, ...)
    # use_macd: bool = Field(False, ...)


class StrategyConfig(BaseModel):
    """Core Strategy Behavior and Parameters."""
    name: str = Field("EVTBot_01", description="Name of the strategy instance (used in logs, potentially broker IDs)")
    timeframe: str = Field("15m", pattern=r"^\d+[mhdMy]$", description="Candlestick timeframe (e.g., '1m', '5m', '1h', '4h', '1d') - must be valid for Bybit")
    polling_interval_seconds: PositiveInt = Field(
        60, ge=5, description="Frequency (seconds) to fetch data and run strategy logic (min 5s recommended)"
    )
    leverage: PositiveInt = Field(
        5, ge=1, le=100, description="Desired leverage for the symbol (check exchange limits, 1 means no leverage). Must be set in Isolated mode."
    )
    # Use the PositionIdx Enum for type safety
    position_idx: PositionIdx = Field(
        PositionIdx.ONE_WAY, # Default to One-Way (0)
        description="Position mode (0: One-Way, 1: Hedge Buy, 2: Hedge Sell). MUST match account setting on Bybit."
    )
    risk_per_trade: Decimal = Field(
        Decimal("0.01"), # 1%
        gt=0,
        le=Decimal("0.1"), # Max 10% risk sanity check
        description="Fraction of available balance to risk per trade (e.g., 0.01 for 1%)",
    )
    stop_loss_atr_multiplier: Decimal = Field(
        Decimal("2.0"),
        gt=0,
        description="ATR multiplier for stop loss distance (must be > 0 if ATR is used for SL)"
    )
    # Nested models for organization
    indicator_settings: IndicatorSettings = Field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = Field(default_factory=AnalysisFlags)

    # Strategy specific flag (can be redundant if analysis_flags.use_evt is the primary control)
    # EVT_ENABLED: bool = Field(
    #     True, description="Confirms EVT logic is the core driver (should match analysis_flags.use_evt)"
    # ) # Redundant with analysis_flags.use_evt

    @field_validator('timeframe')
    @classmethod
    def check_timeframe_format(cls, v: str) -> str:
        # Basic validation, CCXT handles more complex cases during fetch
        import re
        if not re.match(r"^\d+[mhdMy]$", v):
             raise ValueError(f"Invalid timeframe format: '{v}'. Use formats like '1m', '15m', '1h', '4h', '1d'.")
        # TODO: Could add validation against Bybit's allowed timeframes if needed
        return v

    @model_validator(mode='after')
    def check_feature_consistency(self) -> 'StrategyConfig':
        """Ensures related configuration flags are consistent."""
        # Example check: Ensure ATR is enabled if ATR-based SL is configured
        if self.stop_loss_atr_multiplier > 0 and not self.analysis_flags.use_atr:
            raise ValueError("'stop_loss_atr_multiplier' > 0 requires 'analysis_flags.use_atr' to be True.")

        # Ensure position_idx is valid (should be guaranteed by Enum usage)
        if self.position_idx not in PositionIdx:
             raise ValueError(f"Invalid position_idx: {self.position_idx}. Must be 0, 1, or 2.")

        return self


class LoggingConfig(BaseModel):
    """Configuration for the Logger Setup."""
    logger_name: str = Field("TradingBot", description="Name for the logger instance")
    log_file: Optional[str] = Field("trading_bot.log", description="Path to the log file (relative or absolute). Set to None or empty string to disable file logging.")
    # Use standard logging level names (case-insensitive validation)
    console_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level for console output"
    )
    file_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        "DEBUG", description="Logging level for file output (if enabled)"
    )
    log_rotation_bytes: NonNegativeInt = Field(
        5 * 1024 * 1024, # 5 MB
        description="Max log file size in bytes before rotating (0 disables rotation)"
    )
    log_backup_count: NonNegativeInt = Field(
        5, description="Number of backup log files to keep (requires rotation enabled)"
    )
    third_party_log_level_str: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "WARNING", description="Log level for noisy third-party libraries (e.g., ccxt, websockets)"
    )

    @field_validator('log_file', mode='before')
    @classmethod
    def validate_log_file(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() == "":
            return None # Explicitly return None if empty or None
        # Basic check for invalid characters (OS dependent, this is a simple example)
        # Avoid complex validation here, rely on file system errors if path is bad
        # if any(char in v for char in ['<', '>', ':', '"', '|', '?', '*']):
        #      raise ValueError(f"Log file path '{v}' contains potentially invalid characters.")
        return v.strip()

    @field_validator('console_level_str', 'file_level_str', 'third_party_log_level_str', mode='before')
    @classmethod
    def uppercase_log_levels(cls, v: str) -> str:
        """Ensure log level strings are uppercase for Literal matching."""
        return v.upper()


class SMSConfig(BaseModel):
    """Configuration for SMS Alerting (e.g., via Termux or Twilio)."""
    enable_sms_alerts: bool = Field(False, description="Globally enable/disable SMS alerts")

    # Termux Specific
    use_termux_api: bool = Field(False, description="Use Termux:API for sending SMS (requires Termux app setup on Android)")
    sms_recipient_number: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{1,14}$", description="Recipient phone number (E.164 format recommended, e.g., +11234567890)")
    sms_timeout_seconds: PositiveInt = Field(30, ge=5, le=120, description="Timeout for Termux API call (seconds)")

    # Add Twilio fields here if implementing Twilio support
    # use_twilio_api: bool = Field(False, ...)
    # twilio_account_sid: Optional[str] = Field(None, ...)
    # twilio_auth_token: Optional[str] = Field(None, ...)
    # twilio_from_number: Optional[str] = Field(None, ...)

    @model_validator(mode='after')
    def check_sms_provider_details(self) -> 'SMSConfig':
        """Validates that if SMS is enabled, a provider and necessary details are set."""
        if self.enable_sms_alerts:
            provider_configured = False
            if self.use_termux_api:
                if not self.sms_recipient_number:
                    raise ValueError("Termux SMS enabled, but 'sms_recipient_number' is missing.")
                provider_configured = True
            # --- Add check for Twilio if implemented ---
            # elif self.use_twilio_api:
            #     if not all([self.twilio_account_sid, self.twilio_auth_token, self.twilio_from_number, self.sms_recipient_number]):
            #         raise ValueError("Twilio SMS enabled, but required fields (SID, Token, From, Recipient) are missing.")
            #     provider_configured = True

            if not provider_configured:
                raise ValueError("SMS alerts enabled, but no provider (Termux/Twilio) is configured or required details are missing.")
        return self


class AppConfig(BaseSettings):
    """Master Configuration Model integrating all sub-configurations."""
    # Configure Pydantic-Settings behavior (v2 syntax)
    model_config = SettingsConfigDict(
        env_file='.env',                # Load from .env file in the current working directory
        env_file_encoding='utf-8',      # Specify encoding for .env file
        env_nested_delimiter='__',      # Use double underscore for nested env vars (e.g., BOT_API__SYMBOL)
        env_prefix='BOT_',              # Prefix for environment variables (e.g., BOT_API__API_KEY)
        case_sensitive=False,           # Environment variables are typically case-insensitive
        extra='ignore',                 # Ignore extra fields not defined in the models during loading
        validate_default=True,          # Validate default values set in the models
    )

    # Define the nested configuration models using Field for potential default factories
    api: APIConfig = Field(default_factory=APIConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sms: SMSConfig = Field(default_factory=SMSConfig)

    # Add top-level settings if needed, e.g.:
    # app_version: str = "1.1.0"


def load_config() -> AppConfig:
    """
    Loads the application configuration from environment variables and the .env file.

    Handles validation errors and provides informative messages. Exits on critical failure.

    Returns:
        AppConfig: The validated application configuration object.

    Raises:
        SystemExit: If configuration validation fails or a fatal error occurs during loading.
    """
    try:
        print(f"\033[36mLoading configuration...\033[0m")
        # Determine the path to the .env file relative to the current working directory
        # Assumes .env is in the same directory where the main script is run.
        env_file_path = os.path.join(os.getcwd(), '.env')

        if os.path.exists(env_file_path):
            print(f"Attempting to load configuration from: {env_file_path}")
            # Pass the path explicitly if found
            config = AppConfig(_env_file=env_file_path)
        else:
            print(f"'.env' file not found at {env_file_path}. Loading from environment variables only.")
            config = AppConfig() # Still loads from env vars even if file missing

        # Post-load checks/logging (using print as logger might not be ready)
        # Warnings for placeholder keys are now handled by the validator, but a final check is fine.
        if config.api.api_key and "PLACEHOLDER" in config.api.api_key.upper():
             print("\033[91m\033[1mCRITICAL WARNING: API Key appears to be a placeholder. Bot will likely fail authentication.\033[0m")
        if config.api.api_secret and "PLACEHOLDER" in config.api.api_secret.upper():
            print("\033[91m\033[1mCRITICAL WARNING: API Secret appears to be a placeholder. Bot will likely fail authentication.\033[0m")

        # Clear warning if running in testnet mode
        if config.api.testnet_mode:
             print("\033[92mINFO: Testnet mode is ENABLED. Bot will connect to Bybit Testnet.\033[0m")
        else:
             print("\033[93m\033[1mWARNING: Testnet mode is DISABLED. Bot will attempt LIVE trading on Bybit Mainnet.\033[0m")

        print(f"\033[32mConfiguration loaded and validated successfully.\033[0m")
        # Optional: Print loaded config for debugging (be careful with secrets)
        # print("--- Loaded Config (Partial) ---")
        # print(f"  Symbol: {config.api.symbol}")
        # print(f"  Timeframe: {config.strategy.timeframe}")
        # print(f"  Testnet: {config.api.testnet_mode}")
        # print("-----------------------------")
        return config

    except ValidationError as e:
        print(f"\n{'-'*20}\033[91m CONFIGURATION VALIDATION FAILED \033[0m{'-'*20}")
        # Use Pydantic's built-in error formatting for clarity
        print(e)
        # Provide extra guidance
        print(f"\n  \033[93mSuggestions:\033[0m")
        print(f"   - Check your '.env' file for typos or missing values.")
        print(f"   - Ensure environment variables (prefixed with 'BOT_') are set correctly if not using '.env'.")
        print(f"   - Verify data types (e.g., numbers for counts, decimals for rates).")
        print(f"   - For nested settings like 'api.symbol', the env var would be 'BOT_API__SYMBOL'.")
        print(f"{'-'*60}\n")
        raise SystemExit("\033[91mConfiguration validation failed. Please review the errors above and check your settings.\033[0m")

    except Exception as e:
        print(f"\033[91m\033[1mFATAL: Unexpected error loading configuration: {e}\033[0m")
        import traceback
        traceback.print_exc()
        raise SystemExit("\033[91mFailed to load configuration due to an unexpected error.\033[0m")

# Example of how to load config in the main script:
if __name__ == "__main__":
    # This block executes only when config_models.py is run directly
    # Useful for testing the configuration loading independently
    print("Running config_models.py directly for testing...")
    try:
        app_settings = load_config()
        print("\n\033[1mLoaded Config (JSON Representation - Secrets Omitted):\033[0m")
        # Use model_dump_json for Pydantic v2
        # Be cautious about printing secrets - use exclude or custom serializer if needed
        # Example: Exclude sensitive fields
        print(app_settings.model_dump_json(
            indent=2,
            exclude={'api': {'api_key', 'api_secret'}, 'sms': {'twilio_auth_token'}} # Example exclusion
        ))
        print("\n\033[32mConfiguration test successful.\033[0m")
    except SystemExit as e:
         print(f"\n\033[91mExiting due to configuration error during test: {e}\033[0m")
    except Exception as e:
         print(f"\n\033[91mAn unexpected error occurred during the configuration test: {e}\033[0m")
         import traceback
         traceback.print_exc()
EOF
echo -e "${C_SUCCESS}   ✅ Generated config_models.py${C_RESET}"

# --- Create neon_logger.py ---
echo -e "${C_DIM}   -> Generating neon_logger.py${C_RESET}"
cat << 'EOF' > neon_logger.py
#!/usr/bin/env python
"""Neon Logger Setup (v1.5) - Enhanced Robustness & Pydantic Integration

Provides a function `setup_logger` to configure a Python logger instance with:
- Colorized console output using a "neon" theme via colorama (TTY only).
- Uses a custom Formatter for cleaner color handling and padding.
- Clean, non-colorized file output.
- Optional log file rotation (size-based).
- Comprehensive log formatting (timestamp, level, name, function, line, thread).
- Custom SUCCESS log level (25).
- Configuration driven by a Pydantic `LoggingConfig` model from config_models.
- Option to control verbosity of common third-party libraries.
- Improved error handling during setup.
"""

import logging
import logging.handlers
import os
import sys
from typing import Any, Literal, Optional

# --- Import Pydantic model for config type hinting ---
try:
    # Use the specific LoggingConfig model
    from config_models import LoggingConfig
except ImportError:
    print("FATAL [neon_logger]: Could not import LoggingConfig from config_models.py. Ensure file exists and is importable.", file=sys.stderr)
    # Define a fallback simple structure if needed for basic operation, though setup will likely fail later.
    class LoggingConfig: # type: ignore
        logger_name: str = "FallbackLogger"
        log_file: Optional[str] = "fallback.log"
        console_level_str: str = "INFO"
        file_level_str: str = "DEBUG"
        log_rotation_bytes: int = 0
        log_backup_count: int = 0
        third_party_log_level_str: str = "WARNING"
    print("Warning [neon_logger]: Using fallback LoggingConfig structure.", file=sys.stderr)
    # sys.exit(1) # Or exit if LoggingConfig is absolutely essential

# --- Attempt to import colorama ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    # Initialize colorama (autoreset=True ensures colors reset after each print)
    # On Windows, init() is necessary; on Linux/macOS, it might not be strictly required
    # but doesn't hurt. strip=False prevents stripping codes if output is redirected.
    colorama_init(autoreset=True, strip=False)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Define dummy color objects if colorama is not installed
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""  # Return empty string for any attribute access

    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()
    COLORAMA_AVAILABLE = False
    # Warning printed by setup_logger if colors are expected but unavailable

# --- Custom Log Level ---
SUCCESS_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
SUCCESS_LEVEL_NAME = "SUCCESS"
# Check if level already exists (e.g., if module is reloaded)
if not hasattr(logging, SUCCESS_LEVEL_NAME):
    logging.addLevelName(SUCCESS_LEVEL_NUM, SUCCESS_LEVEL_NAME)

# Type hint for the logger method we are adding
if sys.version_info >= (3, 8):
    from typing import Protocol
    class LoggerWithSuccess(logging.Logger, Protocol): # Inherit from Logger for type checker
        def success(self, message: str, *args: Any, **kwargs: Any) -> None: ...
else:
    # Fallback for older Python versions (less precise type checking)
    LoggerWithSuccess = Any # type: ignore

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log method to the Logger instance."""
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

# Add the method to the Logger class dynamically if it doesn't exist
# This avoids potential issues if the script is run multiple times in the same process
if not hasattr(logging.Logger, SUCCESS_LEVEL_NAME.lower()):
    setattr(logging.Logger, SUCCESS_LEVEL_NAME.lower(), log_success)


# --- Neon Color Theme Mapping ---
# Ensure all standard levels and the custom SUCCESS level are included
LOG_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: Fore.CYAN + Style.DIM, # Dim debug messages
    logging.INFO: Fore.BLUE + Style.BRIGHT,
    SUCCESS_LEVEL_NUM: Fore.MAGENTA + Style.BRIGHT, # Make success stand out
    logging.WARNING: Fore.YELLOW + Style.BRIGHT,
    logging.ERROR: Fore.RED + Style.BRIGHT,
    logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
}
DEFAULT_COLOR = Fore.WHITE # Default for levels not explicitly mapped


# --- Custom Formatter for Colored Console Output ---
class ColoredConsoleFormatter(logging.Formatter):
    """A custom logging formatter that adds colors to console output based on log level,
    only if colorama is available and output is detected as a TTY (terminal).
    Handles level name padding correctly within color codes.
    """
    # Define format string components for easier modification
    # Example: %(asctime)s - %(name)s - %(levelname)-9s [%(threadName)s:%(funcName)s:%(lineno)d] - %(message)s
    LOG_FORMAT_BASE = "%(asctime)s - %(name)s - {levelname_placeholder} [%(threadName)s:%(funcName)s:%(lineno)d] - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LEVELNAME_WIDTH = 9 # Width for the padded level name (e.g., "SUCCESS  ")

    def __init__(
        self,
        *, # Force keyword arguments for clarity
        use_colors: Optional[bool] = None, # Allow overriding color detection
        datefmt: Optional[str] = LOG_DATE_FORMAT,
        **kwargs: Any
    ):
        # Determine if colors should be used
        if use_colors is None:
            # Auto-detect: require colorama, stdout to be a TTY, and not explicitly disabled via env var
            is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
            no_color = os.environ.get('NO_COLOR') is not None # Standard env var to disable color
            self.use_colors = COLORAMA_AVAILABLE and is_tty and not no_color
        else:
            self.use_colors = use_colors and COLORAMA_AVAILABLE # Respect override only if colorama exists

        # Dynamically create the format string with or without color placeholders
        # The actual levelname formatting happens in the format() method
        levelname_fmt = "%(levelname)s" # Placeholder to be replaced later
        fmt = self.LOG_FORMAT_BASE.format(levelname_placeholder=levelname_fmt)

        # Initialize the parent Formatter
        super().__init__(fmt=fmt, datefmt=datefmt, style='%', **kwargs) # type: ignore # Pylint/MyPy confusion on style

        if not COLORAMA_AVAILABLE and use_colors is True: # Warn if colors explicitly requested but unavailable
             print("\033[93mWarning [Logger]: Colorama not found, but color usage was explicitly requested. Console logs will be monochrome.\033[0m", file=sys.stderr)
             self.use_colors = False # Ensure colors are off
        elif not self.use_colors and COLORAMA_AVAILABLE:
             # Inform user if colors are available but disabled (e.g., redirected output, NO_COLOR env var)
             reason = "output is not a TTY" if not is_tty else ("'NO_COLOR' env var set" if no_color else "disabled by request")
             if use_colors is None: # Only print if auto-detected off
                print(f"\033[94mInfo [Logger]: Console colors disabled ({reason}). Logs will be monochrome.\033[0m", file=sys.stderr)

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record, applying colors and padding to the level name."""
        # Get the color for the record's level, default if not found
        level_color = LOG_LEVEL_COLORS.get(record.levelno, DEFAULT_COLOR)

        # Store original levelname, apply padding and color (if enabled)
        original_levelname = record.levelname
        # Pad the original levelname to the fixed width
        padded_levelname = f"{original_levelname:<{self.LEVELNAME_WIDTH}}"

        if self.use_colors:
            # Apply color codes around the *padded* level name
            record.levelname = f"{level_color}{padded_levelname}{Style.RESET_ALL}"
        else:
            # Use the padded level name without color codes
            record.levelname = padded_levelname

        # Let the parent class handle the rest of the formatting using the modified record.levelname
        formatted_message = super().format(record)

        # Restore the original levelname on the record object in case it's used elsewhere downstream
        # (though typically not necessary as format() is usually the last step for a handler)
        record.levelname = original_levelname

        return formatted_message


# --- Log Format Strings (for File Handler) ---
# Use a standard format without color codes for files
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)-9s [%(threadName)s:%(funcName)s:%(lineno)d] - %(message)s" # Pad levelname
FILE_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Create Formatters (instantiate only once for efficiency) ---
# Use the custom formatter for the console
console_formatter = ColoredConsoleFormatter()
# Use a standard formatter for the file
file_formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=FILE_LOG_DATE_FORMAT)


# --- Main Setup Function ---
def setup_logger(
    config: LoggingConfig,
    propagate: bool = False # Whether to allow messages to propagate to the root logger
) -> LoggerWithSuccess: # Return type hint includes the custom .success() method
    """
    Sets up and configures a logger instance based on the provided LoggingConfig.

    Args:
        config: A validated LoggingConfig Pydantic model instance.
        propagate: If True, messages logged to this logger will also be passed to
                   handlers of ancestor loggers (usually the root logger). Default is False.

    Returns:
        The configured logging.Logger instance (typed as LoggerWithSuccess).

    Raises:
        ValueError: If log level strings in config are invalid.
        OSError: If file operations fail (e.g., creating directory, opening file).
    """
    # --- Validate Log Levels from Config ---
    try:
        # Use upper() just in case config validator didn't catch case issues
        console_level = logging.getLevelName(config.console_level_str.upper())
        file_level = logging.getLevelName(config.file_level_str.upper())
        third_party_level = logging.getLevelName(config.third_party_log_level_str.upper())

        # Ensure getLevelName returned valid integer levels
        if not isinstance(console_level, int):
            raise ValueError(f"Invalid console log level name: '{config.console_level_str}'")
        if not isinstance(file_level, int):
            raise ValueError(f"Invalid file log level name: '{config.file_level_str}'")
        if not isinstance(third_party_level, int):
            raise ValueError(f"Invalid third-party log level name: '{config.third_party_log_level_str}'")

    except ValueError as e:
        print(f"\033[91mFATAL [Logger]: Invalid log level in configuration: {e}. Please use DEBUG, INFO, SUCCESS, WARNING, ERROR, or CRITICAL.\033[0m", file=sys.stderr)
        raise # Re-raise the error to halt setup

    except Exception as e:
        # Fallback for unexpected errors during level processing
        print(f"\033[91mFATAL [Logger]: Unexpected error processing log levels from config: {e}. Using defaults (INFO, DEBUG, WARNING).\033[0m", file=sys.stderr)
        console_level, file_level, third_party_level = logging.INFO, logging.DEBUG, logging.WARNING
        # Allow continuation with defaults but log the failure

    # --- Get Logger Instance ---
    logger = logging.getLogger(config.logger_name)
    # Set the logger's effective level to the lowest possible (DEBUG)
    # Handlers will then filter based on their individual levels.
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate

    # --- Clear Existing Handlers (optional but recommended for reconfiguration robustness) ---
    if logger.hasHandlers():
        print(f"\033[94mInfo [Logger]: Logger '{config.logger_name}' already has handlers. Clearing them to reconfigure.\033[0m", file=sys.stderr)
        for handler in logger.handlers[:]: # Iterate over a copy
            try:
                # Attempt to flush and close handler before removing
                if hasattr(handler, 'flush'): handler.flush()
                if hasattr(handler, 'close'): handler.close()
                logger.removeHandler(handler)
            except Exception as e_close:
                print(f"\033[93mWarning [Logger]: Error removing/closing handler {handler}: {e_close}\033[0m", file=sys.stderr)

    # --- Console Handler ---
    try:
        console_h = logging.StreamHandler(sys.stdout)
        console_h.setLevel(console_level)
        console_h.setFormatter(console_formatter)
        logger.addHandler(console_h)
        print(f"\033[94m[Logger] Console logging active: Level=[{logging.getLevelName(console_level)}] Colors={'Enabled' if console_formatter.use_colors else 'Disabled'}\033[0m")
    except Exception as e_console:
        print(f"\033[91mError [Logger]: Failed to set up console handler: {e_console}. Console logging might be broken.\033[0m", file=sys.stderr)
        # Continue setup if possible, but log the error

    # --- File Handler (Optional) ---
    if config.log_file:
        try:
            # Ensure log directory exists
            log_file_path = os.path.abspath(config.log_file)
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                # exist_ok=True prevents error if dir already exists
                os.makedirs(log_dir, exist_ok=True)

            # Choose between RotatingFileHandler and FileHandler based on config
            if config.log_rotation_bytes > 0 and config.log_backup_count >= 0:
                # Use RotatingFileHandler if rotation is configured
                file_h = logging.handlers.RotatingFileHandler(
                    filename=log_file_path,
                    maxBytes=config.log_rotation_bytes,
                    backupCount=config.log_backup_count,
                    encoding="utf-8",
                    delay=True # Defer file opening until first log message
                )
                log_type = "Rotating"
                size_mb = config.log_rotation_bytes / (1024*1024)
                log_details = f"(Max: {size_mb:.1f} MB, Backups: {config.log_backup_count})"
            else:
                # Use basic FileHandler if rotation is disabled
                file_h = logging.FileHandler(log_file_path, mode="a", encoding="utf-8", delay=True) # Append mode
                log_type = "Basic"
                log_details = "(Rotation disabled)"

            file_h.setLevel(file_level)
            file_h.setFormatter(file_formatter) # Use the non-colored file formatter
            logger.addHandler(file_h)
            print(f"\033[94m[Logger] {log_type} file logging active: Level=[{logging.getLevelName(file_level)}] File='{log_file_path}' {log_details}\033[0m")

        except OSError as e_os:
            # Handle file system errors (e.g., permission denied) more gracefully
            print(f"\033[91mFATAL [Logger]: OS Error setting up log file '{config.log_file}': {e_os}. File logging disabled. Check path and permissions.\033[0m", file=sys.stderr)
            # Raise the OS error to potentially halt the application if file logging is critical
            raise
        except Exception as e_file:
            # Handle other unexpected errors during file handler setup
            print(f"\033[91mError [Logger]: Unexpected error setting up file logging: {e_file}. File logging disabled.\033[0m", file=sys.stderr)
            # Log the error but allow continuation without file logging
    else:
        print("\033[94m[Logger] File logging disabled by configuration (log_file is empty or None).\033[0m")

    # --- Configure Third-Party Log Levels ---
    # List of common noisy libraries that might need quieting
    noisy_libs = [
        "ccxt", "ccxt.base", "ccxt.async_support", # CCXT core and async
        "urllib3", "requests", # Underlying HTTP libraries often used by ccxt
        "asyncio", # Can be verbose in debug mode
        "websockets", # WebSocket library if used
    ]
    print(f"\033[94m[Logger] Setting third-party library log level to: [{logging.getLevelName(third_party_level)}]\033[0m")
    for lib_name in noisy_libs:
        try:
            lib_logger = logging.getLogger(lib_name)
            if lib_logger:
                # Set level and prevent propagation to avoid double logging or interference
                lib_logger.setLevel(third_party_level)
                lib_logger.propagate = False
        except Exception as e_lib:
            # Non-critical if setting a specific library level fails
            print(f"\033[93mWarning [Logger]: Could not configure log level for library '{lib_name}': {e_lib}\033[0m", file=sys.stderr)

    # Cast the logger to the type hint that includes the .success method
    # This helps type checkers understand the added method.
    return logger # type: ignore

# Example Usage (within main script or for testing)
if __name__ == "__main__":
    print("Running neon_logger.py directly for testing...")
    # Create a dummy config for testing
    # Test with rotation and without
    test_configs = [
        LoggingConfig(
            logger_name="TestLogger_NoFile",
            log_file=None, # Disable file logging
            console_level_str="DEBUG",
            file_level_str="INFO", # Irrelevant if log_file is None
            third_party_log_level_str="ERROR"
        ),
        LoggingConfig(
            logger_name="TestLogger_File",
            log_file="test_logger.log",
            console_level_str="INFO",
            file_level_str="DEBUG",
            log_rotation_bytes=1024 * 50, # Small rotation size (50 KB) for testing
            log_backup_count=2,
            third_party_log_level_str="WARNING"
        )
    ]
    for i, test_config in enumerate(test_configs):
        print(f"\n--- Testing Config {i+1} ({test_config.logger_name}) ---")
        try:
            test_logger: LoggerWithSuccess = setup_logger(test_config) # Use the type hint
            test_logger.debug("This is a debug message.")
            test_logger.info("This is an info message.")
            test_logger.success("This is a success message!") # Use the custom method
            test_logger.warning("This is a warning message.")
            test_logger.error("This is an error message.")
            test_logger.critical("This is a critical message!")

            # Test third-party logger suppression (optional)
            logging.getLogger("ccxt").warning("This ccxt warning should be logged if >= WARNING.")
            logging.getLogger("ccxt").info("This ccxt info should be suppressed if level is WARNING.")

            if test_config.log_file:
                print(f"Test log file created/updated at: {os.path.abspath(test_config.log_file)}")
                # Add more messages to test rotation if configured
                if test_config.log_rotation_bytes > 0:
                     print("Logging extra messages to test rotation...")
                     for _ in range(100):
                         test_logger.debug("Testing log rotation " * 10)
        except Exception as e:
            print(f"\033[91mError during logger test for {test_config.logger_name}: {e}\033[0m")
            import traceback
            traceback.print_exc()
EOF
echo -e "${C_SUCCESS}   ✅ Generated neon_logger.py${C_RESET}"

# --- Create bybit_utils.py ---
echo -e "${C_DIM}   -> Generating bybit_utils.py${C_RESET}"
cat << 'EOF' > bybit_utils.py
# bybit_utils.py
"""
Utility functions supporting the Bybit trading bot framework (v1.1).

Includes:
- Safe data conversions (especially to Decimal).
- Formatting helpers for prices, amounts, order IDs using market precision.
- SMS alerting functionality (currently Termux-based, requires setup).
- Asynchronous retry decorator for robust API calls with exponential backoff.
"""

import asyncio
import functools
import logging
import random # For jitter in retry delay
import subprocess  # For Termux API call
import time
import sys
from collections.abc import Callable, Coroutine
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN, getcontext
from typing import Any, TypeVar, Optional, Union, Dict, Type

# --- Import Pydantic models for type hinting ---
try:
    # Import specific config models needed by this module
    from config_models import AppConfig, SMSConfig, APIConfig
except ImportError:
    print("FATAL ERROR [bybit_utils]: Could not import from config_models. Check file presence.", file=sys.stderr)
    # Define fallback structures or exit if config is critical
    class DummyConfig: pass
    AppConfig = SMSConfig = APIConfig = DummyConfig # type: ignore
    print("Warning [bybit_utils]: Using fallback config structures.", file=sys.stderr)
    # sys.exit(1)

# --- Attempt to import CCXT ---
# Define common CCXT exceptions even if import fails, so retry decorator can compile
# Using simple base Exception as fallback
class DummyExchangeError(Exception): pass
class DummyNetworkError(DummyExchangeError): pass
class DummyRateLimitExceeded(DummyExchangeError): pass
class DummyExchangeNotAvailable(DummyNetworkError): pass
class DummyRequestTimeout(DummyNetworkError): pass
class DummyAuthenticationError(DummyExchangeError): pass
class DummyDDoSProtection(DummyExchangeError): pass # Add DDoSProtection

try:
    import ccxt
    import ccxt.async_support as ccxt_async # Alias for async usage
    from ccxt.base.errors import (
        ExchangeError, NetworkError, RateLimitExceeded, ExchangeNotAvailable,
        RequestTimeout, AuthenticationError, DDoSProtection, # Include DDoSProtection
        # Import others like OrderNotFound, InvalidOrder etc. as needed by callers
    )
    CCXT_AVAILABLE = True
except ImportError:
    print("\033[91mFATAL ERROR [bybit_utils]: CCXT library not found. Install with 'pip install ccxt'\033[0m", file=sys.stderr)
    ccxt = None # Set to None to allow checking later
    ccxt_async = None
    # Assign dummy classes to names expected by retry decorator
    ExchangeError = DummyExchangeError # type: ignore
    NetworkError = DummyNetworkError # type: ignore
    RateLimitExceeded = DummyRateLimitExceeded # type: ignore
    ExchangeNotAvailable = DummyExchangeNotAvailable # type: ignore
    RequestTimeout = DummyRequestTimeout # type: ignore
    AuthenticationError = DummyAuthenticationError # type: ignore
    DDoSProtection = DummyDDoSProtection # type: ignore
    CCXT_AVAILABLE = False
    # Consider sys.exit(1) if CCXT is absolutely essential for this module's core functions

# --- Attempt to import colorama ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore
    COLORAMA_AVAILABLE = False
    # No warning here, handled by logger setup if needed

# --- Logger Setup ---
# Get logger configured in the main script.
# Ensures consistency in logging format and handlers.
logger = logging.getLogger(__name__) # Use __name__ to get logger for this module

# --- Decimal Precision ---
# Set precision for Decimal context (adjust as needed, 28-30 is usually sufficient)
getcontext().prec = 30

# --- Utility Functions ---

def safe_decimal_conversion(
    value: Any, default: Optional[Decimal] = None, context: str = ""
) -> Optional[Decimal]:
    """
    Safely convert various inputs (string, int, float, Decimal) to Decimal.
    Handles None, empty strings, and potential conversion errors gracefully.

    Args:
        value: The value to convert.
        default: The value to return if conversion fails or input is None/empty. Defaults to None.
        context: Optional string describing the source of the value for logging errors.

    Returns:
        The converted Decimal, or the default value. Logs a warning on failure.
    """
    if value is None:
        return default
    # Handle empty strings explicitly
    if isinstance(value, str) and not value.strip():
        return default

    try:
        # Convert float to string first to mitigate potential precision issues inherent in float representation
        if isinstance(value, float):
            # Using format specifier 'f' can sometimes be more reliable than str() for edge cases
            value_str = format(value, '.15f') # Adjust precision as needed
        else:
            value_str = str(value)

        d = Decimal(value_str)

        # Check for NaN (Not a Number) or Infinity, which are valid Decimal states but often unwanted in trading logic
        if d.is_nan() or d.is_infinite():
            log_context = f" in {context}" if context else ""
            logger.warning(f"[safe_decimal] Converted '{value}' to {d}{log_context}. This may indicate an issue. Returning default.")
            return default
        return d
    except (ValueError, TypeError, InvalidOperation) as e:
        log_context = f" in {context}" if context else ""
        logger.warning(f"[safe_decimal] Failed to convert '{value}' (type: {type(value).__name__}) to Decimal{log_context}: {e}. Returning default.")
        return default
    except Exception as e_unexpected:
        # Catch any other unforeseen errors during conversion
        log_context = f" in {context}" if context else ""
        logger.error(f"[safe_decimal] Unexpected error converting '{value}'{log_context}: {e_unexpected}", exc_info=True)
        return default

def format_value_by_market(
    exchange: Optional[ccxt.Exchange],
    symbol: str,
    value_type: Literal['price', 'amount'],
    value: Any
) -> Optional[str]:
    """
    Formats a price or amount value according to the market's precision rules using CCXT.
    Handles potential errors and missing market data gracefully.

    Args:
        exchange: The CCXT exchange instance (must be initialized and markets loaded).
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        value_type: Either 'price' or 'amount'.
        value: The value to format (can be str, int, float, Decimal).

    Returns:
        The formatted value as a string according to market precision,
        or None if the input value is None or conversion fails critically,
        or the raw string representation if CCXT/market data is unavailable.
    """
    if value is None:
        return None
    if not exchange or not CCXT_AVAILABLE:
        logger.warning(f"[format_value] CCXT Exchange instance unavailable. Returning raw string for {value_type} '{value}'.")
        return str(value)

    # Use safe_decimal_conversion to handle input robustly
    value_dec = safe_decimal_conversion(value, context=f"format_{value_type} for {symbol}")
    if value_dec is None:
        # safe_decimal_conversion already logs a warning
        logger.error(f"[format_value] Cannot format invalid {value_type} value '{value}' for {symbol}.")
        return None # Indicate failure clearly if conversion itself failed

    try:
        if value_type == 'price':
            # Use CCXT's built-in method: exchange.price_to_precision()
            # It requires float input, hence the conversion
            return exchange.price_to_precision(symbol, float(value_dec))
        elif value_type == 'amount':
            # Use CCXT's built-in method: exchange.amount_to_precision()
            # It requires float input
            return exchange.amount_to_precision(symbol, float(value_dec))
        else:
            logger.error(f"[format_value] Invalid value_type specified: '{value_type}'. Must be 'price' or 'amount'.")
            return None
    except ccxt.BadSymbol:
        logger.error(f"[format_value] Market symbol '{symbol}' not found in loaded CCXT markets. Cannot format {value_type}.")
        return None
    except ccxt.ExchangeError as e_ccxt:
        # Catch potential errors within CCXT's formatting methods (e.g., missing precision data)
        logger.warning(f"[format_value] CCXT error formatting {value_type} '{value}' for '{symbol}': {e_ccxt}. Falling back to basic string.")
        # Fallback: return a reasonable string representation (e.g., using Decimal's default)
        return str(value_dec)
    except (ValueError, TypeError) as e_conv:
        # Catch errors during the float conversion if value_dec was somehow problematic
        logger.error(f"[format_value] Error converting Decimal to float for CCXT formatting ({value_type} '{value_dec}'): {e_conv}.")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"[format_value] Unexpected error formatting {value_type} '{value}' for {symbol}: {e}", exc_info=True)
        return str(value_dec) # Fallback to basic string

# Convenience wrappers for format_value_by_market
def format_price(exchange: Optional[ccxt.Exchange], symbol: str, price: Any) -> Optional[str]:
    """Formats a price using market precision."""
    return format_value_by_market(exchange, symbol, 'price', price)

def format_amount(exchange: Optional[ccxt.Exchange], symbol: str, amount: Any) -> Optional[str]:
    """Formats an amount using market precision."""
    return format_value_by_market(exchange, symbol, 'amount', amount)


def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """
    Format an order ID for concise logging (shows first/last parts if long).

    Args:
        order_id: The order ID (string or integer).

    Returns:
        A formatted string (e.g., "1234...5678") or "N/A" if None/empty, or "UNKNOWN" on error.
    """
    if order_id is None:
        return "N/A"
    try:
        id_str = str(order_id).strip()
        if not id_str:
            return "N/A"

        # Define length thresholds for abbreviation
        min_len_for_abbrev = 12 # Only abbreviate if longer than this
        prefix_len = 4
        suffix_len = 4

        if len(id_str) <= min_len_for_abbrev:
            return id_str # Return full ID if it's short
        else:
            # Show first and last parts for longer IDs
            return f"{id_str[:prefix_len]}...{id_str[-suffix_len:]}"
    except Exception as e:
        logger.error(f"Error formatting order ID '{order_id}': {e}")
        return "UNKNOWN"

def send_sms_alert(message: str, sms_config: SMSConfig) -> bool:
    """
    Sends an SMS alert using the configured method (currently Termux).
    This is a potentially BLOCKING function due to subprocess.run.
    Use `send_sms_alert_async` wrapper in async code.

    Args:
        message: The text message content.
        sms_config: The validated SMSConfig object containing settings.

    Returns:
        True if the alert was sent successfully (or if alerts are disabled),
        False if sending failed or configuration was invalid.
    """
    # Check if SMS alerts are globally disabled in config
    if not sms_config.enable_sms_alerts:
        logger.debug(f"SMS suppressed (globally disabled): {message[:80]}...")
        return True # Return True as no action was required/failed

    recipient = sms_config.sms_recipient_number
    if not recipient:
        logger.warning("SMS alert requested, but no recipient number configured in SMSConfig.")
        return False

    # --- Termux API Method ---
    if sms_config.use_termux_api:
        timeout = sms_config.sms_timeout_seconds
        try:
            logger.info(f"Attempting Termux SMS to {recipient} (Timeout: {timeout}s)...")
            # Ensure message is treated as a single argument, handle potential quotes/special chars if necessary
            # The list format for command args handles spaces correctly.
            command = ["termux-sms-send", "-n", recipient, message]

            # Using subprocess.run - this will BLOCK the current thread until completion or timeout.
            # If calling from an async context, wrap this call in loop.run_in_executor.
            result = subprocess.run(
                command,
                timeout=timeout,
                check=True,          # Raise CalledProcessError on non-zero exit code
                capture_output=True, # Capture stdout/stderr
                text=True,           # Decode stdout/stderr as text
                encoding='utf-8'     # Explicitly set encoding
            )
            # Log success with output (if any)
            success_msg = f"{Fore.GREEN}Termux SMS Sent OK to {recipient}.{Style.RESET_ALL}"
            output_log = result.stdout.strip()
            if output_log: success_msg += f" Output: {output_log}"
            logger.info(success_msg)
            return True
        except FileNotFoundError:
            logger.error(f"{Fore.RED}Termux command 'termux-sms-send' not found. Is Termux:API installed and accessible in PATH?{Style.RESET_ALL}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"{Fore.RED}Termux SMS timed out after {timeout} seconds for recipient {recipient}.{Style.RESET_ALL}")
            return False
        except subprocess.CalledProcessError as e:
            # Log detailed error including exit code and stderr
            stderr_output = e.stderr.strip() if e.stderr else "(No stderr captured)"
            logger.error(f"{Fore.RED}Termux SMS failed (Exit Code: {e.returncode}) for recipient {recipient}. Error: {stderr_output}{Style.RESET_ALL}")
            return False
        except Exception as e:
            # Catch any other unexpected errors during subprocess execution
            logger.critical(f"{Fore.RED}Unexpected error during Termux SMS execution: {e}{Style.RESET_ALL}", exc_info=True)
            return False

    # --- Twilio API Method (Placeholder) ---
    # elif sms_config.use_twilio_api:
    #     logger.warning("Twilio SMS sending is not implemented in this version.")
    #     # Add Twilio client logic here if implemented
    #     # from twilio.rest import Client
    #     # try:
    #     #     client = Client(sms_config.twilio_account_sid, sms_config.twilio_auth_token)
    #     #     message = client.messages.create(
    #     #         body=message,
    #     #         from_=sms_config.twilio_from_number,
    #     #         to=recipient
    #     #     )
    #     #     logger.info(f"Twilio SMS Sent OK (SID: {message.sid})")
    #     #     return True
    #     # except Exception as e:
    #     #     logger.error(f"Twilio SMS failed: {e}")
    #     #     return False
    #     return False # Return False until implemented

    else:
        # This case should ideally be caught by SMSConfig validation, but double-check
        logger.error("SMS alerts enabled, but no valid provider (Termux/Twilio) is configured or active.")
        return False


async def send_sms_alert_async(message: str, sms_config: SMSConfig):
    """
    Asynchronous wrapper for the potentially blocking `send_sms_alert` function.
    Uses asyncio's `run_in_executor` to avoid blocking the event loop.

    Args:
        message: The text message content.
        sms_config: The validated SMSConfig object containing settings.
    """
    if not sms_config.enable_sms_alerts:
        return # Don't bother scheduling if disabled globally

    try:
        # Get the current running event loop
        loop = asyncio.get_running_loop()

        # Run the blocking function `send_sms_alert` in the loop's default executor (usually ThreadPoolExecutor)
        # functools.partial is used to pass arguments to the function being executed.
        await loop.run_in_executor(
            None, # Use default executor
            functools.partial(send_sms_alert, message, sms_config)
            # Note: The return value (True/False) of send_sms_alert is ignored here,
            #       but errors during its execution within the executor will propagate.
        )
    except RuntimeError as e:
         # Handle cases where there might not be a running loop (less common in typical async apps)
         logger.error(f"Failed to get running event loop for async SMS dispatch: {e}")
    except Exception as e:
        # Catch any other errors during the scheduling or execution in the executor
        logger.error(f"Error dispatching async SMS alert: {e}", exc_info=True)


# --- Asynchronous Retry Decorator Factory ---
# Type variable for the decorated function's return type
T = TypeVar("T")

# Default exceptions to handle for retry (focus on transient network/server issues)
# AuthenticationError is typically NOT retried as it requires user action.
_DEFAULT_RETRY_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    RateLimitExceeded,
    NetworkError, # Includes RequestTimeout, ExchangeNotAvailable, etc.
    DDoSProtection,
    # Add other potentially transient CCXT errors if needed, e.g., some specific ExchangeError subclasses
) if CCXT_AVAILABLE else () # Empty tuple if CCXT failed import

def retry_api_call(
    max_retries_override: Optional[int] = None,
    initial_delay_override: Optional[float] = None,
    handled_exceptions: Tuple[Type[Exception], ...] = _DEFAULT_RETRY_EXCEPTIONS,
    error_message_prefix: str = "API Call Failed",
    # Optional: Add specific exception-delay multipliers if needed:
    # delay_multipliers: Optional[Dict[Type[Exception], float]] = None
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Decorator factory to automatically retry ASYNCHRONOUS API calls with exponential backoff and jitter.

    Requires the decorated async function (or its caller) to pass an `AppConfig` instance
    either as a positional argument or a keyword argument named `app_config`.

    Args:
        max_retries_override: Specific number of retries for this call, overriding config.
        initial_delay_override: Specific initial delay (seconds), overriding config.
        handled_exceptions: Tuple of exception types to catch and retry. Defaults to common
                            transient CCXT network/rate limit/DDoS errors.
        error_message_prefix: String to prefix log messages on failure/retry.

    Returns:
        A decorator that wraps an async function.
    """
    if not handled_exceptions:
        # If CCXT failed or no exceptions provided, log a warning - decorator won't retry.
        logger.warning("[retry_api_call] No handled_exceptions defined (CCXT might be missing or list empty). Decorator will not retry on errors.")

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # --- Find AppConfig instance passed to the decorated function ---
            app_config: Optional[AppConfig] = kwargs.get("app_config")
            if not isinstance(app_config, AppConfig):
                # Search positional arguments if not found in kwargs
                app_config = next((arg for arg in args if isinstance(arg, AppConfig)), None)

            # --- Validate AppConfig Presence ---
            func_name_log = func.__name__ # Get name of the function being decorated for logging
            if not isinstance(app_config, AppConfig):
                # Critical failure if config is missing, as retry parameters depend on it.
                logger.critical(f"{Back.RED}{Fore.WHITE}FATAL: @retry_api_call applied to '{func_name_log}' requires an AppConfig instance to be passed in its arguments (positional or keyword 'app_config').{Style.RESET_ALL}")
                raise ValueError(f"AppConfig instance not provided to decorated function {func_name_log}")

            # Extract API config for easier access to retry parameters
            api_conf: APIConfig = app_config.api

            # Determine effective retry parameters (use overrides if provided, else config defaults)
            effective_max_retries = max_retries_override if max_retries_override is not None else api_conf.retry_count
            effective_base_delay = initial_delay_override if initial_delay_override is not None else api_conf.retry_delay_seconds

            last_exception: Optional[Exception] = None

            # --- Retry Loop: Initial call + number of retries ---
            # Loop runs 'effective_max_retries + 1' times in total.
            for attempt in range(effective_max_retries + 1):
                try:
                    # Log retry attempts (skip logging for the first attempt)
                    if attempt > 0:
                        logger.debug(f"Retrying {func_name_log} (Attempt {attempt + 1}/{effective_max_retries + 1})...")

                    # Execute the wrapped asynchronous function
                    result = await func(*args, **kwargs)
                    # If successful, return the result immediately
                    return result

                # --- Catch specific exceptions designated for retry ---
                except handled_exceptions as e:
                    last_exception = e # Store the exception for potential re-raising later

                    # Check if this was the last allowed attempt
                    if attempt == effective_max_retries:
                        logger.error(f"{Fore.RED}{error_message_prefix}: Max retries ({effective_max_retries + 1}) reached for {func_name_log}. Final error: {type(e).__name__} - {e}{Style.RESET_ALL}")
                        # Trigger alert on final failure for retried exceptions
                        # Run the async alert function without awaiting it here to avoid blocking retry logic
                        asyncio.create_task(send_sms_alert_async(
                            f"ALERT: Max retries failed for API call {func_name_log} ({type(e).__name__})",
                            app_config.sms
                        ))
                        raise e # Re-raise the last exception after exhausting retries

                    # --- Calculate Delay with Exponential Backoff and Jitter ---
                    # Base delay increases exponentially: base * (2^attempt)
                    # Jitter adds a random fraction of the base delay to prevent thundering herd
                    delay = (effective_base_delay * (2 ** attempt)) + (effective_base_delay * random.uniform(0.1, 0.5))

                    # Log specific error types differently for better diagnostics
                    log_level, log_color = logging.WARNING, Fore.YELLOW # Default level for retries
                    specific_msg = ""

                    if isinstance(e, RateLimitExceeded):
                        log_color = Fore.YELLOW + Style.BRIGHT
                        # CCXT might provide a 'retry_after' value in milliseconds
                        retry_after_ms = getattr(e, 'retry_after', None)
                        if retry_after_ms:
                             # Convert ms to seconds, add a small buffer (e.g., 0.5s)
                             suggested_delay_sec = (float(retry_after_ms) / 1000.0) + 0.5
                             delay = max(delay, suggested_delay_sec) # Use the longer delay if API provides one
                             specific_msg = f" API suggests retry after {retry_after_ms}ms."
                        logger.warning(f"{log_color}Rate limit hit in {func_name_log}.{specific_msg} Retrying in {delay:.2f}s... Error: {e}{Style.RESET_ALL}")

                    elif isinstance(e, (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection)):
                        # Treat these as potentially more severe network/server issues
                        log_level, log_color = logging.ERROR, Fore.RED # Use ERROR level for network issues
                        # Ensure a slightly longer minimum delay for network issues
                        delay = max(delay, 3.0)
                        error_type_name = type(e).__name__
                        logger.log(log_level, f"{log_color}{error_type_name} in {func_name_log}. Retrying in {delay:.2f}s... Error: {e}{Style.RESET_ALL}")
                    else:
                        # Generic handled exception (shouldn't happen if handled_exceptions is specific)
                        logger.log(log_level, f"{log_color}Handled exception {type(e).__name__} in {func_name_log}. Retrying in {delay:.2f}s... Error: {e}{Style.RESET_ALL}")

                    # Wait asynchronously before the next attempt
                    await asyncio.sleep(delay)

                # --- Catch other unexpected exceptions ---
                except Exception as e_unhandled:
                    # Log critical errors and re-raise immediately - DO NOT RETRY unhandled exceptions.
                    logger.critical(f"{Back.RED}{Fore.WHITE}UNEXPECTED error in {func_name_log}: {type(e_unhandled).__name__} - {e_unhandled}{Style.RESET_ALL}", exc_info=True) # Include traceback for unexpected errors
                    # Trigger alert immediately for critical, unhandled errors
                    asyncio.create_task(send_sms_alert_async(
                        f"CRITICAL UNEXPECTED ERROR in {func_name_log}: {type(e_unhandled).__name__}",
                        app_config.sms
                    ))
                    raise e_unhandled # Re-raise the unhandled exception, halting the retry process

            # --- Post-Loop Logic (Should only be reached if max_retries is negative, which is invalid) ---
            # This point should theoretically not be reached if the loop logic is correct and max_retries >= 0.
            # If it is reached, it implies the loop finished without returning or raising.
            if last_exception:
                 # This might happen if max_retries was 0 and the first attempt failed with a handled exception.
                 # The loop finishes, but we should still raise the exception caught.
                 logger.error(f"Retry loop for {func_name_log} completed after failure on first attempt (max_retries=0?). Raising last error: {last_exception}")
                 raise last_exception
            else:
                 # This case is highly unlikely if max_retries >= 0
                 msg = f"Retry loop for {func_name_log} finished unexpectedly without success or a recorded error. Max retries: {effective_max_retries}."
                 logger.critical(msg)
                 raise RuntimeError(msg) # Raise a generic runtime error

        return wrapper # Return the decorated function
    return decorator # Return the decorator itself


# --- Example usage of the decorator (illustrative) ---
# @retry_api_call(max_retries_override=2, error_message_prefix="Fetch Balance Failed")
# async def fetch_balance_with_retry(exchange: ccxt_async.Exchange, app_config: AppConfig):
#     # IMPORTANT: The decorated function MUST accept 'app_config' as an argument
#     #            (either positional or keyword) for the decorator to find it.
#     logger.info("Attempting to fetch balance inside decorated function...")
#     # Example: Simulate a potential failure
#     # if random.random() < 0.5:
#     #     raise ccxt.RequestTimeout("Simulated timeout fetching balance")
#     balance = await exchange.fetch_balance()
#     logger.info("Balance fetched successfully inside decorated function.")
#     return balance
EOF
echo -e "${C_SUCCESS}   ✅ Generated bybit_utils.py${C_RESET}"

# --- Create indicators.py ---
echo -e "${C_DIM}   -> Generating indicators.py${C_RESET}"
cat << 'EOF' > indicators.py
#!/usr/bin/env python
"""Technical Indicators Module (v1.3)

Provides functions to calculate various technical indicators using pandas DataFrames
containing OHLCV data. Leverages the `pandas_ta` library for common indicators
and includes custom implementations like Ehlers Volumetric Trend (EVT).

Designed to be driven by configuration passed via an `AppConfig` object.
Handles potential errors and insufficient data gracefully.
"""

import logging
import sys
from typing import Any, Dict, Optional, Tuple, Union # Added Union

import numpy as np
import pandas as pd

# --- Import pandas_ta ---
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    print(
        "\033[91mFATAL ERROR [indicators]: 'pandas_ta' library not found.\033[0m"
        "\033[93mPlease install it: pip install pandas_ta\033[0m", file=sys.stderr
    )
    # Set flag and allow module to load, but calculations requiring it will fail.
    PANDAS_TA_AVAILABLE = False
    ta = None # Set ta to None to allow checks later
    # Consider sys.exit(1) if pandas_ta is absolutely essential for any operation.

# --- Import Pydantic models for config type hinting ---
try:
    # Import specific models needed
    from config_models import AppConfig, IndicatorSettings, AnalysisFlags
except ImportError:
    print("FATAL [indicators]: Could not import from config_models.py. Ensure file exists.", file=sys.stderr)
    # Define fallback structures or exit
    class DummyConfig: pass
    AppConfig = IndicatorSettings = AnalysisFlags = DummyConfig # type: ignore
    print("Warning [indicators]: Using fallback config structures.", file=sys.stderr)
    # sys.exit(1)


# --- Setup ---
logger = logging.getLogger(__name__) # Get logger configured in main script

# --- Constants ---
# Define standard column names expected in input DataFrames
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"
REQUIRED_OHLCV_COLS = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]


# --- Helper Functions ---

def _validate_dataframe(df: pd.DataFrame, min_rows: int, required_cols: list[str]) -> bool:
    """
    Helper to validate DataFrame input for indicator calculations.
    Checks for DataFrame type, emptiness, required columns, and sufficient valid rows.

    Args:
        df: The pandas DataFrame to validate.
        min_rows: The minimum number of non-NaN rows required in the essential columns.
        required_cols: A list of column names that must be present.

    Returns:
        True if the DataFrame is valid and has sufficient data, False otherwise.
    """
    # Use inspect to get caller function name for clearer logs
    try:
        caller_frame = sys._getframe(1)
        func_name = caller_frame.f_code.co_name
    except Exception:
        func_name = "DataFrame Validation" # Fallback name

    log_prefix = f"[{func_name}]"

    if df is None or not isinstance(df, pd.DataFrame):
        logger.error(f"{log_prefix} Input is not a valid pandas DataFrame.")
        return False
    if df.empty:
        logger.error(f"{log_prefix} Input DataFrame is empty.")
        return False

    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"{log_prefix} Missing required columns: {missing_cols}.")
        return False

    # Check for sufficient non-NaN rows in required columns
    # Using dropna on the subset is efficient
    try:
        valid_rows_count = len(df.dropna(subset=required_cols))
    except KeyError:
        # This should technically be caught by the missing_cols check, but as a safeguard:
        logger.error(f"{log_prefix} Error checking NaNs, required columns might be missing despite initial check.")
        return False

    if valid_rows_count < min_rows:
        logger.warning(
            f"{log_prefix} Insufficient valid data rows ({valid_rows_count}) for calculation "
            f"(minimum required: {min_rows}). Results may be inaccurate or contain NaNs."
        )
        # Decide whether to proceed or fail based on strictness.
        # Current behavior: Warn but allow calculation to proceed (returning False would halt).
        # To halt on insufficient data, uncomment the next line:
        # return False
    return True


# --- Pivot Point Calculations (Standard & Fibonacci) ---
# Note: These typically use the *previous* period's OHLC to calculate levels for the *current* or *next* period.
# The main indicator function applies them to the entire DataFrame history if needed.

def calculate_standard_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates standard pivot points based on High, Low, Close values."""
    pivots = {}
    # Basic validation: Ensure inputs are valid numbers
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.debug("Standard Pivots calculation skipped: Invalid H/L/C input (NaN or non-numeric).")
        return pivots
    if low > high:
        # Handle potential data error where low > high
        logger.warning(f"Standard Pivots Warning: Low ({low}) > High ({high}). Check input data. Using absolute difference for range.")
        # Option 1: Use absolute difference (allows calculation)
        range_hl = abs(high - low)
        # Option 2: Swap them (might hide data issues)
        # low, high = high, low
        # range_hl = high - low
        # Option 3: Return empty (safest if data quality is suspect)
        # return {}
    else:
        range_hl = high - low

    try:
        # Calculate Pivot Point (PP)
        pivot = (high + low + close) / 3.0
        pivots["PP"] = round(pivot, 8) # Round for consistency, adjust precision if needed

        # Calculate Support levels
        pivots["S1"] = round((2 * pivot) - high, 8)
        pivots["S2"] = round(pivot - range_hl, 8)
        pivots["S3"] = round(low - 2 * (high - pivot), 8) # Alternative: S3 = S1 - range_hl

        # Calculate Resistance levels
        pivots["R1"] = round((2 * pivot) - low, 8)
        pivots["R2"] = round(pivot + range_hl, 8)
        pivots["R3"] = round(high + 2 * (pivot - low), 8) # Alternative: R3 = R1 + range_hl

    except Exception as e:
        logger.error(f"Error calculating standard pivots: {e}", exc_info=False)
        return {} # Return empty dict on error
    return pivots

def calculate_fib_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates Fibonacci pivot points based on High, Low, Close values."""
    fib_pivots = {}
    # Basic validation
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.debug("Fibonacci Pivots calculation skipped: Invalid H/L/C input.")
        return fib_pivots
    if low > high:
        logger.warning(f"Fibonacci Pivots Warning: Low ({low}) > High ({high}). Using absolute difference for range.")
        fib_range = abs(high - low)
    else:
        fib_range = high - low

    try:
        # Pivot Point (same as standard)
        pivot = (high + low + close) / 3.0
        fib_pivots["PP"] = round(pivot, 8)

        # Handle potential zero range (e.g., if high == low)
        # Use a small tolerance for float comparison
        if abs(fib_range) < 1e-9:
            logger.debug("Fibonacci Pivots: Range is near zero. Calculating only PP.")
            # All S/R levels would collapse onto PP, return only PP
            return {"PP": fib_pivots["PP"]}

        # Fibonacci Levels (Common Ratios: 0.382, 0.618, 1.000)
        # Support Levels
        fib_pivots["FS1"] = round(pivot - (0.382 * fib_range), 8) # Use 'FS' prefix
        fib_pivots["FS2"] = round(pivot - (0.618 * fib_range), 8)
        fib_pivots["FS3"] = round(pivot - (1.000 * fib_range), 8) # Often corresponds to 'low - range_hl'

        # Resistance Levels
        fib_pivots["FR1"] = round(pivot + (0.382 * fib_range), 8) # Use 'FR' prefix
        fib_pivots["FR2"] = round(pivot + (0.618 * fib_range), 8)
        fib_pivots["FR3"] = round(pivot + (1.000 * fib_range), 8) # Often corresponds to 'high + range_hl'

    except Exception as e:
        logger.error(f"Error calculating Fibonacci pivots: {e}", exc_info=False)
        return {}
    return fib_pivots

# --- Support / Resistance Level Calculation (Example using Pivots & Period Range) ---
# This is a simplified example; robust S/R often involves more complex analysis
# (e.g., peak/trough detection, volume profiles, clustering).
def calculate_support_resistance_levels(
    df_period: pd.DataFrame, current_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates various potential support/resistance levels based on historical data
    (e.g., pivots from the last period, Fib retracements over the whole period).

    Note: This is a basic example for demonstration.

    Args:
        df_period: DataFrame representing the historical period (e.g., daily data for daily levels).
                   Requires 'high', 'low', 'close' columns.
        current_price: The current market price, used to classify levels as support or resistance.

    Returns:
        A dictionary containing:
            'support': Dict of levels below current_price {label: value}.
            'resistance': Dict of levels above current_price {label: value}.
            'pivot_points': Dict containing 'standard' and 'fibonacci' pivot sets calculated from the last bar.
            'fib_retracements': Dict of retracement levels based on the period's high/low.
    """
    levels: Dict[str, Any] = {
        "support": {},
        "resistance": {},
        "pivot_points": {"standard": {}, "fibonacci": {}},
        "fib_retracements": {}
    }

    # Validate input DataFrame requires at least 2 rows (one for last period, one for range)
    if not _validate_dataframe(df_period, min_rows=2, required_cols=[COL_HIGH, COL_LOW, COL_CLOSE]):
        logger.debug("S/R Levels calculation skipped: Invalid DataFrame or insufficient rows.")
        return levels

    try:
        # --- Calculate Pivots for the *next* period based on the *last completed* period in the df ---
        # Use iloc[-1] to get the most recent completed bar's data
        last_high = df_period[COL_HIGH].iloc[-1]
        last_low = df_period[COL_LOW].iloc[-1]
        last_close = df_period[COL_CLOSE].iloc[-1]

        # Calculate Standard Pivots
        standard_pivots = calculate_standard_pivot_points(last_high, last_low, last_close)
        if standard_pivots:
            levels["pivot_points"]["standard"] = standard_pivots

        # Calculate Fibonacci Pivots
        fib_pivots = calculate_fib_pivot_points(last_high, last_low, last_close)
        if fib_pivots:
            levels["pivot_points"]["fibonacci"] = fib_pivots

        # --- Calculate Fibonacci Retracements over the *entire* df_period range ---
        period_high = df_period[COL_HIGH].max()
        period_low = df_period[COL_LOW].min()
        period_diff = period_high - period_low

        # Avoid division by zero or meaningless retracements if range is tiny
        if abs(period_diff) > 1e-9: # Use tolerance for float comparison
            levels["fib_retracements"] = {
                # Using percentages for clarity
                "High (100.0%)": round(period_high, 8),
                "Fib 78.6%": round(period_low + period_diff * 0.786, 8), # Common Fib level
                "Fib 61.8%": round(period_low + period_diff * 0.618, 8), # Golden Ratio conjugate
                "Fib 50.0%": round(period_low + period_diff * 0.5, 8),   # Midpoint
                "Fib 38.2%": round(period_low + period_diff * 0.382, 8), # Golden Ratio conjugate
                "Fib 23.6%": round(period_low + period_diff * 0.236, 8), # Common Fib level
                "Low (0.0%)": round(period_low, 8),
            }
        else:
            logger.debug("Fib Retracements skipped: Period range is near zero.")

        # --- Classify All Calculated Levels as Support/Resistance based on Current Price ---
        if current_price is not None and isinstance(current_price, (int, float)) and not np.isnan(current_price):
            # Combine all calculated levels into one dictionary for easier iteration
            # Prefix pivot labels to avoid clashes
            all_potential_levels = {
                **{f"Std Piv {k}": v for k, v in levels["pivot_points"]["standard"].items()},
                **{f"Fib Piv {k}": v for k, v in levels["pivot_points"]["fibonacci"].items()},
                **levels["fib_retracements"]
            }

            for label, value in all_potential_levels.items():
                # Ensure the level value is valid before comparison
                if isinstance(value, (int, float)) and not np.isnan(value):
                    # Use a small tolerance for comparison to avoid issues with floating point equality
                    tolerance = 1e-9
                    if value < current_price - tolerance:
                        levels["support"][label] = value
                    elif value > current_price + tolerance:
                        levels["resistance"][label] = value
                    # else: level is very close to current price, could be treated as neutral or ignored

    except IndexError:
        logger.warning("IndexError calculating S/R levels (likely insufficient data rows). Some levels might be missing.")
    except Exception as e:
        logger.error(f"Error calculating S/R levels: {e}", exc_info=True) # Log full traceback for unexpected errors

    # Sort S/R levels for easier reading/use (optional)
    # Sort support levels descending (highest support first)
    levels["support"] = dict(sorted(levels["support"].items(), key=lambda item: item[1], reverse=True))
    # Sort resistance levels ascending (lowest resistance first)
    levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda item: item[1]))

    return levels


# --- Custom Indicator Implementations ---

def calculate_vwma(close: pd.Series, volume: pd.Series, length: int) -> Optional[pd.Series]:
    """
    Calculates Volume Weighted Moving Average (VWMA).

    Args:
        close: pandas Series of closing prices.
        volume: pandas Series of volume data.
        length: The lookback period for the VWMA.

    Returns:
        A pandas Series containing the VWMA, or None if calculation fails.
    """
    # Input validation
    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series):
        logger.error("VWMA Error: Inputs must be pandas Series.")
        return None
    if close.empty or volume.empty:
        logger.error("VWMA Error: Input Series cannot be empty.")
        return None
    if len(close) != len(volume):
        logger.error(f"VWMA Error: Close ({len(close)}) and Volume ({len(volume)}) Series lengths differ.")
        return None
    if length <= 0:
        logger.error(f"VWMA Error: Invalid length ({length}). Must be positive.")
        return None
    if len(close) < length:
        # Not strictly an error, but result will have leading NaNs
        logger.debug(f"VWMA Debug: Data length ({len(close)}) is less than the specified period ({length}). Result will contain NaNs.")
        # Allow calculation to proceed

    try:
        # Calculate Price * Volume
        pv = close * volume
        # Calculate rolling sum of Price * Volume over the specified length
        # min_periods=length ensures that we only get a value when a full window is available
        cumulative_pv = pv.rolling(window=length, min_periods=length).sum()
        # Calculate rolling sum of Volume over the specified length
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()

        # Calculate VWMA: Sum(Price * Volume) / Sum(Volume)
        # Replace 0 volume in the denominator with NaN to prevent division by zero errors and propagate NaN result naturally
        vwma = cumulative_pv / cumulative_vol.replace(0, np.nan)

        # Assign a descriptive name to the resulting Series
        vwma.name = f"VWMA_{length}"
        return vwma
    except Exception as e:
        logger.error(f"Error calculating VWMA(length={length}): {e}", exc_info=True)
        return None

def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
    """
    Calculates the Ehlers Volumetric Trend (EVT) indicator.

    This indicator uses a Volume Weighted Moving Average (VWMA) smoothed with a
    SuperSmoother filter. Trend direction is determined by comparing the smoothed
    VWMA to its previous value using bands defined by the multiplier.

    Adds columns to the DataFrame:
        - `vwma_{length}`: Raw Volume Weighted Moving Average.
        - `smooth_vwma_{length}`: SuperSmoother applied to VWMA.
        - `evt_trend_{length}`: Trend direction (1 for up, -1 for down, 0 for neutral/transition).
        - `evt_buy_{length}`: Boolean signal, True when trend flips from non-up (0 or -1) to up (1).
        - `evt_sell_{length}`: Boolean signal, True when trend flips from non-down (0 or 1) to down (-1).

    Args:
        df: DataFrame with 'close' and 'volume' columns. Index should preferably be datetime.
        length: The period length for VWMA and SuperSmoother (e.g., 7). Must be > 1.
        multiplier: Multiplier for trend bands (e.g., 2.5 representing 2.5%). Must be > 0.

    Returns:
        The original DataFrame with EVT columns added. Returns original df on critical failure.
    """
    func_name = "ehlers_volumetric_trend"
    # Need enough data for VWMA(length) + 2 periods for SuperSmoother calculation
    min_required_rows = length + 2

    # Validate DataFrame using the helper function
    if not _validate_dataframe(df, min_rows=min_required_rows, required_cols=[COL_CLOSE, COL_VOLUME]):
        logger.warning(f"[{func_name}] Input validation failed or insufficient data. Skipping EVT calculation.")
        return df # Return original DataFrame without EVT columns

    # Validate parameters
    if length <= 1:
        logger.error(f"[{func_name}] Invalid length ({length}). Must be > 1. Skipping calculation.")
        return df
    if multiplier <= 0:
        logger.error(f"[{func_name}] Invalid multiplier ({multiplier}). Must be > 0. Skipping calculation.")
        return df

    # Work on a copy to avoid modifying the original DataFrame passed to the function
    df_out = df.copy()

    # Define column names based on parameters
    vwma_col = f"vwma_{length}"
    smooth_col = f"smooth_vwma_{length}"
    trend_col = f"evt_trend_{length}"
    buy_col = f"evt_buy_{length}"
    sell_col = f"evt_sell_{length}"

    try:
        # --- Step 1: Calculate VWMA ---
        vwma = calculate_vwma(df_out[COL_CLOSE], df_out[COL_VOLUME], length=length)
        if vwma is None or vwma.isnull().all():
            # calculate_vwma handles internal errors, check if result is usable
            raise ValueError(f"VWMA calculation failed or resulted in all NaNs for length {length}.")
        df_out[vwma_col] = vwma

        # Optional: Fill initial NaNs in VWMA if needed for smoother start?
        # Be cautious: This introduces lookahead bias if not handled carefully.
        # Generally better to let NaNs propagate naturally.
        # df_out[vwma_col] = df_out[vwma_col].fillna(method='bfill') # Backfill example

        # --- Step 2: Apply SuperSmoother Filter to VWMA ---
        # SuperSmoother filter implementation based on Ehlers' formula.
        # Constants derived from the filter's transfer function for a given length.
        sqrt2 = np.sqrt(2.0)
        pi = np.pi
        a = np.exp(-sqrt2 * pi / length)
        b = 2 * a * np.cos(sqrt2 * pi / length)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3

        # Initialize the smoothed series with NaNs
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        # Access the VWMA values as a NumPy array for potentially faster iteration
        vwma_vals = df_out[vwma_col].values

        # Iterate to calculate smoothed values (requires previous 2 values)
        # Start from index 2 as the formula needs values at i-1 and i-2
        for i in range(2, len(df_out)):
            # Check if current VWMA and previous two values (needed for filter) are valid numbers
            # Use previous smoothed values if available, otherwise fallback to previous VWMA values
            # This handles the initialization phase of the filter more gracefully.
            if pd.notna(vwma_vals[i]):
                 # Get previous smoothed value (sm1) or fallback to previous VWMA (vwma_vals[i-1])
                 sm1 = smoothed.iloc[i-1] if pd.notna(smoothed.iloc[i-1]) else (vwma_vals[i-1] if pd.notna(vwma_vals[i-1]) else np.nan)
                 # Get second previous smoothed value (sm2) or fallback to second previous VWMA (vwma_vals[i-2])
                 sm2 = smoothed.iloc[i-2] if pd.notna(smoothed.iloc[i-2]) else (vwma_vals[i-2] if pd.notna(vwma_vals[i-2]) else np.nan)

                 # Calculate smoothed value only if all required inputs are valid
                 if pd.notna(sm1) and pd.notna(sm2):
                     smoothed.iloc[i] = c1 * vwma_vals[i] + c2 * sm1 + c3 * sm2

        df_out[smooth_col] = smoothed

        # --- Step 3: Determine Trend based on smoothed VWMA changes relative to bands ---
        # Calculate band factors based on the multiplier (percentage)
        mult_factor_high = 1.0 + (multiplier / 100.0)
        mult_factor_low = 1.0 - (multiplier / 100.0)

        # Get the smoothed value from the previous period
        shifted_smooth = df_out[smooth_col].shift(1)

        # Conditions for trend change:
        # Uptrend starts if smoothed value crosses above the previous value * upper band factor
        trend_up_condition = (df_out[smooth_col] > shifted_smooth * mult_factor_high)
        # Downtrend starts if smoothed value crosses below the previous value * lower band factor
        trend_down_condition = (df_out[smooth_col] < shifted_smooth * mult_factor_low)

        # Initialize trend series (0 = neutral/no trend established yet)
        trend = pd.Series(0, index=df_out.index, dtype=int)
        # Mark potential trend start points
        trend[trend_up_condition] = 1   # Mark as potential uptrend start
        trend[trend_down_condition] = -1 # Mark as potential downtrend start

        # Propagate the trend signal forward until a counter-signal occurs.
        # Replace 0s with NaN before forward-filling so only actual 1/-1 signals propagate.
        # Fill any remaining NaNs at the beginning of the series with 0 (neutral).
        trend = trend.replace(0, np.nan).ffill().fillna(0).astype(int)
        df_out[trend_col] = trend

        # --- Step 4: Generate Buy/Sell Signals based on Trend *Changes* ---
        # Get the trend from the previous period (use fill_value=0 for the first element)
        trend_shifted = df_out[trend_col].shift(1, fill_value=0)

        # Buy signal: Trend flips from non-up (0 or -1) to up (1)
        df_out[buy_col] = (df_out[trend_col] == 1) & (trend_shifted != 1)
        # Sell signal: Trend flips from non-down (0 or 1) to down (-1)
        df_out[sell_col] = (df_out[trend_col] == -1) & (trend_shifted != -1)

        logger.debug(f"[{func_name}] Calculation successful for length={length}, multiplier={multiplier}.")
        return df_out

    except ValueError as ve: # Catch specific value errors raised internally
        logger.error(f"[{func_name}] Value error during calculation (len={length}, mult={multiplier}): {ve}")
        # Add NaN columns to indicate failure but maintain structure if desired
        for col in [vwma_col, smooth_col, trend_col, buy_col, sell_col]:
            if col not in df_out.columns: df_out[col] = np.nan
        return df # Return original df or partially modified one
    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during calculation (len={length}, mult={multiplier}): {e}", exc_info=True)
        # Add NaN columns to indicate failure
        for col in [vwma_col, smooth_col, trend_col, buy_col, sell_col]:
            if col not in df_out.columns: df_out[col] = np.nan
        return df # Return the (partially) modified DataFrame


# --- Master Indicator Calculation Function ---
def calculate_all_indicators(df: pd.DataFrame, app_config: AppConfig) -> pd.DataFrame:
    """
    Calculates all enabled technical indicators based on the AppConfig settings.
    Applies indicators to the provided DataFrame.

    Args:
        df: Input DataFrame with OHLCV data. Index should ideally be DatetimeIndex.
        app_config: The validated AppConfig object containing strategy and indicator settings.

    Returns:
        A DataFrame with the original data and calculated indicator columns added.
        Returns the original DataFrame (possibly empty) on critical failure or if no indicators enabled.
    """
    func_name = "calculate_all_indicators"
    if df is None:
        logger.error(f"[{func_name}] Input DataFrame is None. Cannot calculate indicators.")
        return pd.DataFrame() # Return empty DF on critical input error
    if df.empty:
        logger.warning(f"[{func_name}] Input DataFrame is empty. Returning empty DataFrame.")
        return df # Return the empty DF

    # Validate index type (optional but good practice)
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(f"[{func_name}] DataFrame index is not a DatetimeIndex. Some indicators might rely on time-based indexing.")
        # Optionally attempt conversion:
        # try:
        #     df.index = pd.to_datetime(df.index)
        #     logger.info(f"[{func_name}] Converted DataFrame index to DatetimeIndex.")
        # except Exception as e_conv:
        #     logger.error(f"[{func_name}] Failed to convert index to DatetimeIndex: {e_conv}. Proceeding with original index.")

    # Extract config components safely
    try:
        settings: IndicatorSettings = app_config.strategy.indicator_settings
        flags: AnalysisFlags = app_config.strategy.analysis_flags
        min_rows_needed = settings.min_data_periods
    except AttributeError as e:
        logger.critical(f"[{func_name}] Failed to access configuration from AppConfig (AttributeError: {e}). Cannot proceed.")
        return df # Return original df as config is missing or malformed

    # Validate DataFrame content and length using the helper
    # Use a slightly relaxed min_rows for the initial check, specific indicators will re-validate if needed.
    if not _validate_dataframe(df, min_rows=max(5, min_rows_needed // 2), required_cols=REQUIRED_OHLCV_COLS): # Basic check first
        logger.error(f"[{func_name}] Input DataFrame validation failed basic checks. Indicator calculation aborted.")
        # Depending on strictness, could return df or empty df
        return df # Return original df, allowing caller to handle potentially missing indicators

    # Work on a copy to avoid modifying the original DataFrame passed into the function
    df_out = df.copy()

    logger.debug(f"[{func_name}] Calculating indicators. Available Flags: {flags.model_dump()}, Settings used: {settings.model_dump()}")

    indicators_calculated = [] # Keep track of successfully calculated indicators

    try:
        # --- Use pandas_ta for standard indicators if available and enabled ---
        if PANDAS_TA_AVAILABLE and ta:
            # ATR Calculation
            if flags.use_atr:
                if settings.atr_period > 0:
                    logger.debug(f"Calculating ATR (length={settings.atr_period}) using pandas_ta...")
                    try:
                        # pandas_ta automatically appends column named like 'ATRr_14'
                        df_out.ta.atr(length=settings.atr_period, append=True)
                        # Verify column was added (pandas_ta might fail silently in some cases)
                        atr_col_name = f"ATRr_{settings.atr_period}" # Default name format
                        if atr_col_name in df_out.columns:
                            indicators_calculated.append("ATR")
                            logger.debug(f"ATR calculation successful (column '{atr_col_name}').")
                        else:
                             logger.warning(f"ATR calculation using pandas_ta did not add expected column '{atr_col_name}'.")
                    except Exception as e_ta_atr:
                         logger.error(f"Error calculating ATR with pandas_ta: {e_ta_atr}", exc_info=False)
                else:
                    logger.warning("ATR calculation skipped: atr_period <= 0 in settings.")

            # Add other pandas_ta indicators based on flags:
            # Example: RSI
            # if flags.use_rsi:
            #     if settings.rsi_period > 0:
            #         logger.debug(f"Calculating RSI (length={settings.rsi_period}) using pandas_ta...")
            #         try:
            #             df_out.ta.rsi(length=settings.rsi_period, append=True) # Appends 'RSI_14'
            #             rsi_col_name = f"RSI_{settings.rsi_period}"
            #             if rsi_col_name in df_out.columns: indicators_calculated.append("RSI")
            #             else: logger.warning(f"RSI pandas_ta calc failed to add column '{rsi_col_name}'.")
            #         except Exception as e_ta_rsi: logger.error(f"Error calc RSI pandas_ta: {e_ta_rsi}", exc_info=False)
            #     else: logger.warning("RSI skipped: rsi_period <= 0")

            # Example: MACD
            # if flags.use_macd:
            #      # MACD requires fast, slow, signal periods
            #      if all(p > 0 for p in [settings.macd_fast, settings.macd_slow, settings.macd_signal]):
            #           logger.debug(f"Calculating MACD(f={settings.macd_fast}, s={settings.macd_slow}, g={settings.macd_signal}) using pandas_ta...")
            #           try:
            #                df_out.ta.macd(fast=settings.macd_fast, slow=settings.macd_slow, signal=settings.macd_signal, append=True) # Appends MACD_f_s_g, MACDh_f_s_g, MACDs_f_s_g
            #                macd_col_name = f"MACD_{settings.macd_fast}_{settings.macd_slow}_{settings.macd_signal}"
            #                if macd_col_name in df_out.columns: indicators_calculated.append("MACD")
            #                else: logger.warning(f"MACD pandas_ta calc failed to add column '{macd_col_name}'.")
            #           except Exception as e_ta_macd: logger.error(f"Error calc MACD pandas_ta: {e_ta_macd}", exc_info=False)
            #      else: logger.warning("MACD skipped: Invalid periods (must be > 0).")

        elif flags.use_atr: # Log if pandas_ta needed but unavailable
            logger.warning(f"[{func_name}] pandas_ta library not available, but ATR calculation was requested. Skipping standard indicators.")

        # --- Calculate Custom Indicators ---
        if flags.use_evt:
            logger.debug(f"Calculating Ehlers Volumetric Trend (length={settings.evt_length}, multiplier={settings.evt_multiplier})")
            # Pass the current state of df_out to the function
            df_out = ehlers_volumetric_trend(df_out, settings.evt_length, float(settings.evt_multiplier))
            # Check if EVT columns were added (function might return original df on error)
            evt_trend_col = f"evt_trend_{settings.evt_length}"
            if evt_trend_col in df_out.columns:
                indicators_calculated.append("EVT")
            else:
                logger.warning("EVT calculation did not add expected columns. Check logs for errors.")

        # --- Post-Calculation Processing ---
        # Remove potential duplicate columns if calculations somehow appended existing names (less likely with pandas_ta's naming)
        # df_out = df_out.loc[:, ~df_out.columns.duplicated()] # Keep first occurrence

        # Optional: Log NaN count in final indicator columns for monitoring data quality/indicator stability
        final_cols = df_out.columns.difference(df.columns) # Get newly added columns
        if not final_cols.empty:
            try:
                nan_counts = df_out[final_cols].isnull().sum()
                nan_summary = nan_counts[nan_counts > 0] # Show only columns with NaNs
                if not nan_summary.empty:
                    logger.debug(f"[{func_name}] NaN counts in newly added indicator columns:\n{nan_summary.to_string()}")
                else:
                     logger.debug(f"[{func_name}] No NaNs found in newly added indicator columns.")
            except Exception as e_nan:
                logger.warning(f"[{func_name}] Could not check NaN counts in indicator columns: {e_nan}")

    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during indicator calculation loop: {e}", exc_info=True)
        # Return the DataFrame as it was before the error, possibly partially calculated
        return df_out

    if not indicators_calculated:
        logger.warning(f"[{func_name}] No indicators were calculated (check flags and library availability).")
    else:
        logger.info(f"[{func_name}] Indicators calculated: {', '.join(indicators_calculated)}. DataFrame shape: {df_out.shape}")

    return df_out
EOF
echo -e "${C_SUCCESS}   ✅ Generated indicators.py${C_RESET}"

# --- Create bybit_helper_functions.py ---
echo -e "${C_DIM}   -> Generating bybit_helper_functions.py${C_RESET}"
cat << 'EOF' > bybit_helper_functions.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bybit V5 CCXT Helper Functions (v3.6 - Enhanced Async & Config)

Collection of asynchronous helper functions for interacting with the Bybit V5 API
using the CCXT library (async support). Integrates tightly with Pydantic models
defined in `config_models.py` for configuration and validation. Provides robust
error handling, retries via decorator, market caching, and commonly needed
operations like fetching data, placing/managing orders, and managing positions.
"""

# Standard Library Imports
import asyncio
import json
import logging
import math
import os
import random # Used in retry jitter
import sys
import time
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import (Any, Coroutine, Dict, List, Literal, Optional, Sequence,
                    Tuple, TypeVar, Union, Callable, Type) # Added Type

# --- Import Pydantic models first ---
try:
    # Import necessary models and Enums defined in config_models
    from config_models import (
        APIConfig, AppConfig, PositionIdx, Category, OrderFilter, Side, TimeInForce,
        TriggerBy, TriggerDirection, OrderType, StopLossTakeProfitMode, AccountType
    )
except ImportError:
    print("FATAL [bybit_helpers]: Could not import from config_models.py. Ensure file exists and is importable.", file=sys.stderr)
    class DummyConfig: pass
    # Define dummy fallbacks if import fails (will likely cause runtime errors later)
    AppConfig = APIConfig = DummyConfig # type: ignore
    PositionIdx = Category = OrderFilter = Side = TimeInForce = TriggerBy = TriggerDirection = OrderType = StopLossTakeProfitMode = AccountType = Enum # type: ignore
    print("Warning [bybit_helpers]: Using fallback types for config models and enums.", file=sys.stderr)
    # sys.exit(1)

# Third-party Libraries
try:
    import ccxt
    import ccxt.async_support as ccxt_async # Use async version exclusively
    from ccxt.base.errors import (
        ArgumentsRequired, AuthenticationError, BadSymbol, CancelPending, # Added CancelPending
        DDoSProtection, ExchangeError, ExchangeNotAvailable, InsufficientFunds,
        InvalidNonce, InvalidOrder, NetworkError, NotSupported, OrderImmediatelyFillable,
        OrderNotFound, RateLimitExceeded, RequestTimeout
    )
    # For precise rounding in specific cases (less common now with built-in methods)
    # from ccxt.base.decimal_to_precision import ROUND_UP, ROUND_DOWN as CCXT_ROUND_DOWN

    CCXT_AVAILABLE = True
except ImportError:
    print("\033[91mFATAL ERROR [bybit_helpers]: CCXT library not found. Install with 'pip install ccxt'\033[0m", file=sys.stderr)
    # Define dummy exceptions and classes if CCXT is missing
    class DummyExchangeError(Exception): pass
    class DummyNetworkError(DummyExchangeError): pass
    class DummyRateLimitExceeded(DummyExchangeError): pass
    class DummyAuthenticationError(DummyExchangeError): pass
    class DummyOrderNotFound(DummyExchangeError): pass
    class DummyInvalidOrder(DummyExchangeError): pass
    class DummyInsufficientFunds(DummyExchangeError): pass
    class DummyExchangeNotAvailable(DummyNetworkError): pass
    class DummyRequestTimeout(DummyNetworkError): pass
    class DummyNotSupported(DummyExchangeError): pass
    class DummyOrderImmediatelyFillable(DummyInvalidOrder): pass
    class DummyBadSymbol(DummyExchangeError): pass
    class DummyArgumentsRequired(DummyExchangeError): pass
    class DummyDDoSProtection(DummyExchangeError): pass
    class DummyInvalidNonce(DummyAuthenticationError): pass
    class DummyCancelPending(DummyExchangeError): pass # Add CancelPending

    ccxt = None; ccxt_async = None # type: ignore
    # Assign dummy exceptions to the names used in the code
    ExchangeError = DummyExchangeError; NetworkError = DummyNetworkError # type: ignore
    RateLimitExceeded = DummyRateLimitExceeded; AuthenticationError = DummyAuthenticationError # type: ignore
    OrderNotFound = DummyOrderNotFound; InvalidOrder = DummyInvalidOrder # type: ignore
    InsufficientFunds = DummyInsufficientFunds; ExchangeNotAvailable = DummyExchangeNotAvailable # type: ignore
    RequestTimeout = DummyRequestTimeout; NotSupported = DummyNotSupported # type: ignore
    OrderImmediatelyFillable = DummyOrderImmediatelyFillable; BadSymbol = DummyBadSymbol # type: ignore
    ArgumentsRequired = DummyArgumentsRequired; DDoSProtection = DummyDDoSProtection # type: ignore
    InvalidNonce = DummyInvalidNonce; CancelPending = DummyCancelPending # type: ignore
    # ROUND_UP = 'ROUND_UP'; CCXT_ROUND_DOWN = 'ROUND_DOWN' # Not used directly

    CCXT_AVAILABLE = False
    # sys.exit(1) # Exit if CCXT is absolutely required

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    # Pandas is primarily used for OHLCV formatting, can operate without it
    print("Info [bybit_helpers]: pandas library not found. OHLCV data will be returned as lists of lists.", file=sys.stderr)
    pd = None # type: ignore
    PANDAS_AVAILABLE = False

try:
    from colorama import Fore, Style, Back, init as colorama_init
    # Initialize colorama (required on Windows, safe on others)
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore
    COLORAMA_AVAILABLE = False

# Optional WebSocket support (Import handled within relevant functions if needed)
# WEBSOCKETS_AVAILABLE = False # Flag can be set if websocket functions are added


# --- Configuration & Constants ---
getcontext().prec = 30 # Set global Decimal precision for consistency

# --- Logger Setup ---
logger = logging.getLogger(__name__) # Get logger instance configured in main script

# --- Global Market Cache ---
class MarketCache:
    """
    Simple asynchronous cache for CCXT market data to avoid repeated loading.
    Includes methods to get market details and infer Bybit V5 category.
    """
    def __init__(self):
        self._markets: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, Optional[Category]] = {} # Cache category Enum member
        self._lock = asyncio.Lock() # Ensure thread-safe/async-safe updates
        self._last_load_time: float = 0.0
        self._cache_duration_seconds: int = 3600 # Cache markets for 1 hour by default

    async def load_markets(self, exchange: ccxt_async.Exchange, reload: bool = False) -> bool:
        """
        Loads or reloads market data into the cache from the exchange.

        Args:
            exchange: The initialized async CCXT exchange instance.
            reload: If True, forces a reload even if cache is fresh.

        Returns:
            True if markets were loaded/reloaded successfully, False otherwise.
        """
        current_time = time.monotonic()
        # Check cache validity first without lock for efficiency
        if not reload and self._markets and (current_time - self._last_load_time < self._cache_duration_seconds):
            logger.debug("[MarketCache] Using cached markets (cache is fresh).")
            return True

        # Acquire lock for loading/reloading to prevent race conditions
        async with self._lock:
            # Double-check cache validity *after* acquiring the lock
            if not reload and self._markets and (current_time - self._last_load_time < self._cache_duration_seconds):
                logger.debug("[MarketCache] Using cached markets (checked after lock).")
                return True

            action = 'Reloading' if self._markets else 'Loading'
            logger.info(f"{Fore.BLUE}[MarketCache] {action} markets for {exchange.id}...{Style.RESET_ALL}")
            try:
                # Force reload from exchange API using reload=True
                all_markets = await exchange.load_markets(reload=True)
                if not all_markets:
                    # This indicates a serious issue with the exchange connection or response
                    logger.critical(f"{Back.RED}FATAL [MarketCache]: Failed to load markets - received empty response from {exchange.id}.{Style.RESET_ALL}")
                    self._markets = {} # Clear potentially stale cache
                    self._categories = {}
                    return False

                self._markets = all_markets
                self._categories.clear() # Clear derived category cache as markets have updated
                self._last_load_time = time.monotonic()
                logger.success(f"{Fore.GREEN}[MarketCache] Loaded/Reloaded {len(self._markets)} markets successfully.{Style.RESET_ALL}")
                return True
            except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
                logger.error(f"{Fore.RED}[MarketCache] Network/Exchange error loading markets: {type(e).__name__} - {e}{Style.RESET_ALL}")
                # Don't clear cache on transient errors, maybe it's still usable
                return False # Indicate failure
            except ExchangeError as e:
                logger.error(f"{Fore.RED}[MarketCache] Exchange specific error loading markets: {e}{Style.RESET_ALL}", exc_info=False)
                return False
            except Exception as e:
                logger.critical(f"{Back.RED}[MarketCache] CRITICAL unexpected error loading markets: {e}{Style.RESET_ALL}", exc_info=True)
                # Consider clearing cache on critical failure
                self._markets = {}
                self._categories = {}
                return False

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves market data for a specific symbol from the cache."""
        market_data = self._markets.get(symbol)
        if not market_data:
            # This might happen if load_markets failed or the symbol is invalid
            logger.debug(f"[MarketCache] Market '{symbol}' not found in cache.")
        return market_data

    def get_category(self, symbol: str) -> Optional[Category]:
        """
        Retrieves or infers the Bybit V5 category (Enum: linear, inverse, spot, option) for a symbol.
        Caches the result for subsequent calls.
        """
        # Check cached category first
        if symbol in self._categories:
            return self._categories[symbol]

        market = self.get_market(symbol)
        category_enum: Optional[Category] = None
        if market:
            category_enum = self._infer_v5_category_from_market(market) # Use internal helper
            if category_enum is None:
                # Log details if inference fails
                market_info = market.get('info', {})
                logger.warning(
                    f"[MarketCache] Could not determine V5 category for symbol '{symbol}'. "
                    f"CCXT Type: {market.get('type', 'N/A')}, "
                    f"Info: category='{market_info.get('category', 'N/A')}', "
                    f"contractType='{market_info.get('contractType', 'N/A')}', "
                    f"settleCoin='{market_info.get('settleCoin', market.get('settle', 'N/A'))}'"
                )
        else:
             # Market itself not found
             logger.warning(f"[MarketCache] Cannot determine category for '{symbol}' as market data is not loaded/available.")


        # Cache the result (even if None) to avoid recalculating
        # This assumes market data doesn't change category frequently between reloads
        self._categories[symbol] = category_enum
        return category_enum

    def _infer_v5_category_from_market(self, market: Dict[str, Any]) -> Optional[Category]:
        """Internal helper to infer V5 category from CCXT market structure."""
        if not market: return None

        # Priority 1: Check 'info' field for explicit 'category' (most reliable for V5)
        info = market.get('info', {})
        category_from_info = info.get('category') # e.g., 'linear', 'inverse', 'spot', 'option'
        if category_from_info:
            try:
                return Category(category_from_info.lower()) # Match Enum value
            except ValueError:
                logger.debug(f"[MarketCache] Category '{category_from_info}' from market.info not a valid Category enum value.")
                # Continue to other checks

        # Priority 2: Use CCXT standard market type flags (spot, option, linear, inverse)
        # These flags are usually set correctly by CCXT based on market data.
        if market.get('spot', False): return Category.SPOT
        if market.get('option', False): return Category.OPTION
        if market.get('linear', False): return Category.LINEAR # CCXT standard linear flag
        if market.get('inverse', False): return Category.INVERSE # CCXT standard inverse flag

        # Priority 3: Infer from CCXT 'type' and other fields (less reliable, more guesswork)
        market_type = market.get('type') # e.g., 'spot', 'swap', 'future'
        symbol = market.get('symbol', 'N/A')

        if market_type == Category.SPOT.value: return Category.SPOT # Check against enum value
        if market_type == Category.OPTION.value: return Category.OPTION

        if market_type in ['swap', 'future']:
            # For derivatives, check contract type and settle currency if category wasn't explicit
            contract_type = str(info.get('contractType', '')).lower() # 'Linear', 'Inverse'
            settle_coin = info.get('settleCoin', market.get('settle', '')).upper() # Prefer 'settleCoin' from info

            if contract_type == Category.LINEAR.value: return Category.LINEAR
            if contract_type == Category.INVERSE.value: return Category.INVERSE

            # If contractType missing, guess based on settle coin (common convention)
            if settle_coin in ['USDT', 'USDC', 'USD']: return Category.LINEAR # Common stablecoin collateral
            # Check if settle coin matches the base currency (typical for inverse)
            base_coin = market.get('base', '').upper()
            if settle_coin and base_coin and settle_coin == base_coin: return Category.INVERSE

            # If still unsure, make a default assumption (e.g., linear is more common)
            logger.debug(f"[MarketCache] Ambiguous derivative market '{symbol}' (type='{market_type}', settle='{settle_coin}'). Assuming '{Category.LINEAR.value}' based on common usage.")
            return Category.LINEAR

        # If none of the above matched, we cannot determine the category
        logger.warning(f"[MarketCache] Could not determine V5 category for market '{symbol}' with CCXT type '{market_type}'.")
        return None

    def get_all_symbols(self) -> List[str]:
        """Returns a list of all symbols currently present in the market cache."""
        # Return a copy of the keys to prevent modification of the internal dict
        return list(self._markets.keys())

# Instantiate the global cache (singleton pattern for simplicity)
market_cache = MarketCache()


# --- Utility Function Imports ---
# Import utility functions AFTER defining logger and market_cache, as they might use them
try:
    # Import specific functions needed
    from bybit_utils import (safe_decimal_conversion, format_price, format_amount,
                             format_order_id, send_sms_alert_async, retry_api_call)
except ImportError:
    print("FATAL [bybit_helpers]: Could not import utility functions from bybit_utils.py.", file=sys.stderr)
    # Define dummy functions or exit if utils are critical
    def _dummy_func(*args, **kwargs): logger.error("Util function missing!"); return None
    def _dummy_decorator_factory(*args_dec, **kwargs_dec):
        def decorator(func): return func
        return decorator
    safe_decimal_conversion = format_price = format_amount = format_order_id = _dummy_func # type: ignore
    send_sms_alert_async = _dummy_func # type: ignore # Define dummy async func if needed
    retry_api_call = _dummy_decorator_factory # type: ignore
    print("Warning [bybit_helpers]: Using dummy utility functions.", file=sys.stderr)
    # sys.exit(1)


# --- Exchange Initialization & Configuration ---
# Apply retry logic directly to the initialization function for robustness
# Use slightly longer delays for initialization as it's less frequent
@retry_api_call(max_retries_override=2, initial_delay_override=5.0, error_message_prefix="Exchange Init Failed")
async def initialize_bybit(app_config: AppConfig) -> Optional[ccxt_async.bybit]:
    """
    Initializes and validates the Bybit CCXT exchange instance using AppConfig.

    Handles testnet/mainnet modes, loads markets into the cache, and performs an
    authentication check by fetching balance (if keys are provided).

    Args:
        app_config: The validated AppConfig object.

    Returns:
        An initialized and validated ccxt.async_support.bybit instance, or None on failure.
    """
    func_name = "initialize_bybit"
    api_conf = app_config.api # Convenience alias for API settings

    if not CCXT_AVAILABLE:
        logger.critical(f"{Back.RED}FATAL [{func_name}]: CCXT library not available. Cannot initialize exchange.{Style.RESET_ALL}")
        return None

    mode_str = 'Testnet' if api_conf.testnet_mode else 'Mainnet'
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}[{func_name}] Initializing Bybit V5 ({mode_str}, Async)...{Style.RESET_ALL}")

    exchange: Optional[ccxt_async.bybit] = None # Ensure type hint is for async version

    try:
        # Check for valid API keys (not None and not placeholders)
        has_valid_keys = bool(
            api_conf.api_key and api_conf.api_secret and
            "PLACEHOLDER" not in api_conf.api_key.upper() and
            "PLACEHOLDER" not in api_conf.api_secret.upper()
        )
        if not has_valid_keys:
            logger.warning(f"{Fore.YELLOW}[{func_name}] API Key/Secret missing or placeholders detected. Initializing in PUBLIC data mode only.{Style.RESET_ALL}")

        # CCXT configuration options for Bybit V5
        exchange_options = {
            'apiKey': api_conf.api_key if has_valid_keys else None,
            'secret': api_conf.api_secret if has_valid_keys else None,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                # 'defaultType': 'swap', # Less critical with explicit category in V5 calls
                'adjustForTimeDifference': True, # Adjust clock drift automatically
                'recvWindow': api_conf.default_recv_window,
                # Add Broker ID / Referer code if applicable (check Bybit docs for format)
                # Example using strategy name:
                'brokerId': f"PB_{app_config.strategy.name[:10].replace(' ', '_')}",
                # V5 specific options might go here if needed, but category is usually passed in params
                # 'defaultCategory': api_conf.category.value, # Can set default, but passing per call is safer
            },
            # Consider adding a custom user agent for identification
            # 'headers': {'User-Agent': f'TradingBot/{app_config.app_version}'},
        }

        # Instantiate the async Bybit exchange class from ccxt.async_support
        exchange = ccxt_async.bybit(exchange_options)

        # Set sandbox mode AFTER instantiation if configured
        if api_conf.testnet_mode:
            exchange.set_sandbox_mode(True)
            logger.info(f"[{func_name}] Testnet mode explicitly enabled.")

        logger.info(f"[{func_name}] Base API URL: {exchange.urls['api']}")

        # --- Load Markets (Crucial Step) ---
        # This populates the market_cache
        markets_loaded = await market_cache.load_markets(exchange, reload=True) # Force reload on init
        if not markets_loaded:
            logger.critical(f"{Back.RED}FATAL [{func_name}]: Failed to load markets from Bybit. Cannot proceed.{Style.RESET_ALL}")
            await safe_exchange_close(exchange) # Attempt cleanup
            return None

        # --- Verify Primary Symbol & Category ---
        primary_symbol = api_conf.symbol
        primary_category = api_conf.category # Category from config

        market_data = market_cache.get_market(primary_symbol)
        cached_category = market_cache.get_category(primary_symbol) # Infer from loaded data

        if not market_data:
            logger.critical(f"{Back.RED}FATAL [{func_name}]: Primary symbol '{primary_symbol}' not found in loaded markets.{Style.RESET_ALL}")
            await safe_exchange_close(exchange)
            return None
        else:
             logger.info(f"[{func_name}] Verified primary symbol '{primary_symbol}' exists in loaded markets.")

        # Compare configured category with inferred category
        if cached_category != primary_category:
             logger.warning(f"{Fore.YELLOW}[{func_name}] Category mismatch for '{primary_symbol}': Config='{primary_category.value}', Inferred='{cached_category.value if cached_category else 'None'}'. Using configured category '{primary_category.value}' for operations, but double-check config.{Style.RESET_ALL}")
             # Proceed using the config value, but warn the user of potential mismatch.

        # --- Authentication Check (If Keys Provided) ---
        if has_valid_keys:
            logger.info(f"[{func_name}] Performing authentication check (fetching balance)...")
            try:
                # Use a function that requires authentication, like fetch_balance
                # Pass the app_config object, as required by the retry decorator and the function itself
                balance_info = await fetch_usdt_balance(exchange, app_config=app_config)

                if balance_info is None:
                    # fetch_usdt_balance logs errors internally, but we check return value here
                    # The retry decorator should have handled transient errors. Persistent None suggests auth failure.
                    raise AuthenticationError("fetch_usdt_balance returned None after potential retries, indicating likely authentication or permission issue.")

                # Log success if balance fetch worked (balance_info is tuple (equity, avail) or None)
                equity, avail = balance_info # Unpack tuple
                logger.info(f"[{func_name}] Authentication check OK. Account Equity: {equity:.4f} {api_conf.usdt_symbol}, Available: {avail:.4f} {api_conf.usdt_symbol}")

            except AuthenticationError as auth_err:
                logger.critical(f"{Back.RED}CRITICAL [{func_name}]: Authentication FAILED! Check API key/secret, permissions (UTA?), and IP restrictions. Error: {auth_err}{Style.RESET_ALL}")
                # Send alert for critical auth failure
                asyncio.create_task(send_sms_alert_async(
                    f"[BybitHelper] CRITICAL: Bot Auth Failed during initialization!",
                    app_config.sms
                ))
                await safe_exchange_close(exchange)
                return None
            except (NetworkError, RequestTimeout) as net_err:
                # These should have been handled by the retry decorator on initialize_bybit itself.
                # If they reach here, it means retries failed.
                 logger.critical(f"{Back.RED}FATAL [{func_name}]: Persistent network error during auth check after retries: {net_err}{Style.RESET_ALL}")
                 await safe_exchange_close(exchange)
                 return None
            except ExchangeError as ex_err:
                # Catch other exchange errors during balance fetch that might not be auth related
                logger.critical(f"{Back.RED}CRITICAL [{func_name}]: Exchange error during auth check: {ex_err}. Check permissions or account status.{Style.RESET_ALL}")
                await safe_exchange_close(exchange)
                return None
        else:
            logger.info(f"[{func_name}] Skipping authentication check (no valid API keys provided).")

        # --- Success ---
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}[{func_name}] Bybit V5 exchange initialized and validated successfully ({mode_str}).{Style.RESET_ALL}")
        return exchange

    # --- Exception Handling during Instantiation or Initial Setup ---
    except AuthenticationError as e:
        # This might catch errors during the ccxt_async.bybit() call itself if keys are immediately rejected
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Authentication error during initial exchange setup: {e}.{Style.RESET_ALL}")
        asyncio.create_task(send_sms_alert_async(f"[BybitHelper] CRITICAL: Bot Auth Failed during setup!", app_config.sms))
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        # Catch network errors not handled by retry during the initial setup phase itself
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Network/Exchange availability error during initial setup: {e}.{Style.RESET_ALL}")
    except ExchangeError as e:
        # Catch other CCXT exchange errors during setup
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Exchange error during initial setup: {e}{Style.RESET_ALL}", exc_info=False)
        asyncio.create_task(send_sms_alert_async(f"[BybitHelper] CRITICAL: Init ExchangeError: {type(e).__name__}", app_config.sms))
    except Exception as e:
        # Catch any other unexpected errors during the complex initialization process
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Unexpected error during initialization: {e}{Style.RESET_ALL}", exc_info=True)
        asyncio.create_task(send_sms_alert_async(f"[BybitHelper] CRITICAL: Init Unexpected Error: {type(e).__name__}", app_config.sms))

    # Ensure exchange is closed if initialization failed at any point before returning None
    await safe_exchange_close(exchange)
    return None

async def safe_exchange_close(exchange: Optional[ccxt_async.Exchange]):
    """Safely attempts to close the CCXT exchange connection if it exists and has a close method."""
    if exchange and hasattr(exchange, 'close') and callable(exchange.close):
        try:
            logger.info("[safe_exchange_close] Attempting to close exchange connection...")
            await exchange.close()
            logger.info("[safe_exchange_close] Exchange connection closed.")
        except Exception as e:
            # Log error but don't prevent script termination
            logger.error(f"[safe_exchange_close] Error closing exchange connection: {e}", exc_info=False)


# --- Account Functions ---

@retry_api_call()
async def set_leverage(
    exchange: ccxt_async.bybit, symbol: str, leverage: int, app_config: AppConfig,
    position_idx: Optional[PositionIdx] = None # Allow specifying hedge mode side
) -> bool:
    """
    Sets leverage for a specific symbol (Linear/Inverse contracts) using V5 API.
    Requires appropriate account mode (Unified Trading with Isolated Margin for the symbol, or legacy Contract).

    Args:
        exchange: Initialized Bybit async exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        leverage: The desired integer leverage (e.g., 5 for 5x). Must be > 0.
        app_config: The application configuration object.
        position_idx: Optional. Required if account is in Hedge Mode (1 for Buy, 2 for Sell).
                      If One-Way mode (0), this is ignored by the API but good to pass for clarity.

    Returns:
        True if leverage was set successfully or already matched, False otherwise.
    """
    func_name = "set_leverage"
    log_prefix = f"[{func_name}({symbol} -> {leverage}x)]"

    if leverage <= 0:
        logger.error(f"{Fore.RED}{log_prefix} Leverage must be greater than 0.{Style.RESET_ALL}")
        return False

    # Determine category and validate market existence
    # Use configured category primarily, but fetch market for validation
    category = app_config.api.category # Use configured category
    market = market_cache.get_market(symbol)

    if not market:
        logger.error(f"{Fore.RED}{log_prefix} Market data for '{symbol}' not found. Cannot set leverage.{Style.RESET_ALL}")
        return False
    if category not in [Category.LINEAR, Category.INVERSE]:
        logger.error(f"{Fore.RED}{log_prefix} Leverage can only be set for LINEAR or INVERSE contracts. Configured category: {category.value}.{Style.RESET_ALL}")
        return False

    # Validate leverage against market limits (best effort using cached data)
    try:
        limits = market.get('limits', {}).get('leverage', {})
        max_lev_val = safe_decimal_conversion(limits.get('max'), context=f"{symbol} max leverage")
        min_lev_val = safe_decimal_conversion(limits.get('min', '1'), context=f"{symbol} min leverage") # Assume min 1 if not present

        if max_lev_val is not None and leverage > max_lev_val:
            logger.error(f"{Fore.RED}{log_prefix} Requested leverage {leverage}x exceeds maximum allowed ({max_lev_val}x) for {symbol}.{Style.RESET_ALL}")
            return False
        if min_lev_val is not None and leverage < min_lev_val:
             logger.error(f"{Fore.RED}{log_prefix} Requested leverage {leverage}x is below minimum allowed ({min_lev_val}x) for {symbol}.{Style.RESET_ALL}")
             return False
    except Exception as e_limits:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Could not fully validate leverage limits against market data due to error: {e_limits}. Proceeding cautiously.{Style.RESET_ALL}")

    # Prepare parameters for V5 API call via CCXT's set_leverage override for Bybit
    # Bybit V5 endpoint: POST /v5/position/set-leverage
    # Requires category, symbol, buyLeverage, sellLeverage
    params = {
        'category': category.value,
        'buyLeverage': str(leverage), # API expects string values
        'sellLeverage': str(leverage) # Set same leverage for both sides
        # 'positionIdx' is NOT used by set_leverage endpoint directly. Leverage applies to the symbol in Isolated mode.
        # Hedge mode leverage might be linked to position mode setting itself.
    }
    logger.info(f"{Fore.CYAN}{log_prefix} Sending request to set leverage... Params: {params}{Style.RESET_ALL}")

    try:
        # CCXT's set_leverage method for bybit async handles the specific API call structure
        # It expects leverage as float/int, symbol, and params dictionary
        response = await exchange.set_leverage(float(leverage), symbol, params=params)

        # Check response structure for success indication (if available)
        # Bybit's V5 set_leverage response on success is minimal (retCode 0).
        # CCXT usually returns info from the request if successful, or raises error.
        # We rely on the absence of exceptions as the primary success indicator.
        logger.success(f"{Fore.GREEN}{log_prefix} Leverage set/confirmed successfully for {symbol} (Assumes Isolated Margin mode). Response snippet: {str(response)[:100]}...{Style.RESET_ALL}")
        return True

    except ExchangeError as e:
        # Check specific Bybit V5 error codes or messages for context
        # Error codes documentation: https://bybit-exchange.github.io/docs/v5/error_code
        error_code_str = str(getattr(e, 'code', None)) # Get error code if available
        error_msg = str(e).lower() # Lowercase message for easier matching

        # Code 110043: Leverage not modified (already set to the desired value)
        if error_code_str == '110043' or "leverage not modified" in error_msg:
            logger.info(f"{Fore.YELLOW}{log_prefix} Leverage already set to {leverage}x.{Style.RESET_ALL}")
            return True # Treat as success

        # Code 110025: Position is not sync (can happen during high volatility or connection issues)
        elif error_code_str == '110025' or "position is not sync between риска" in error_msg:
             logger.warning(f"{Fore.YELLOW}{log_prefix} Position sync issue (Code: {error_code_str}). Retrying might resolve. Error: {e}{Style.RESET_ALL}")
             raise e # Re-raise to allow retry decorator to handle it

        # Code 110021: Cannot set leverage under Hedge Mode (leverage might be tied to position mode switch?)
        elif error_code_str == '110021':
             logger.error(f"{Fore.RED}{log_prefix} Failed (Code: {error_code_str}): Cannot set leverage in Hedge Mode via this endpoint. Leverage might be tied to position mode setting. Error: {e}{Style.RESET_ALL}")
             return False # Unlikely to succeed with retry

        # Code 30086: Cross margin mode does not support setting leverage per symbol
        elif error_code_str == '30086' or "set leverage not supported under cross margin mode" in error_msg:
            logger.error(f"{Fore.RED}{log_prefix} Failed (Code: {error_code_str}): Cannot set leverage per symbol in Cross Margin mode. Switch account/symbol to Isolated Margin.{Style.RESET_ALL}")
            return False

        # Add other relevant error codes if encountered (e.g., permission denied)
        # elif error_code_str == '10004': # Parameter error
        #     logger.error(...) return False
        # elif error_code_str == '10016': # Permission denied
        #      logger.error(...) return False

        else:
            # Generic exchange error not specifically handled above
            logger.error(f"{Fore.RED}{log_prefix} ExchangeError setting leverage: Code={error_code_str}, Error={e}{Style.RESET_ALL}", exc_info=False)
            return False
    except (NetworkError, RequestTimeout, DDoSProtection) as e:
        # Network errors will be retried by the decorator, log warning and re-raise
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Server error during leverage setting (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for the decorator to handle retry
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error setting leverage: {e}{Style.RESET_ALL}", exc_info=True)
        return False

@retry_api_call()
async def fetch_usdt_balance(
    exchange: ccxt_async.bybit, app_config: AppConfig
) -> Optional[Tuple[Decimal, Decimal]]:
    """
    Fetches USDT balance details specifically from the UNIFIED trading account on Bybit V5.

    Args:
        exchange: Initialized Bybit async exchange instance.
        app_config: The application configuration object.

    Returns:
        A tuple containing (Total Equity, Available Balance for USDT) as Decimals,
        or None if fetching fails, account type not found, or USDT balance is not present.
    """
    func_name = "fetch_usdt_balance"
    log_prefix = f"[{func_name}]"
    usdt_symbol = app_config.api.usdt_symbol # Get target coin (e.g., USDT) from config
    account_type_target = AccountType.UNIFIED # Explicitly target UNIFIED account

    logger.debug(f"{log_prefix} Fetching {account_type_target.value} account balance ({usdt_symbol})...")
    try:
        # Use fetch_balance with params for V5 Unified Account
        # V5 requires specifying accountType
        params = {'accountType': account_type_target.value}
        balance_data = await exchange.fetch_balance(params=params)

        # --- Parse V5 Response Structure ---
        # Expected structure: { info: { result: { list: [ { accountType: 'UNIFIED', totalEquity: '...', coin: [ { coin: 'USDT', availableToWithdraw: '...' } ] } ] } } }
        info = balance_data.get('info', {})
        result = info.get('result', {})
        account_list = result.get('list', [])

        if not account_list or not isinstance(account_list, list):
            logger.warning(f"{log_prefix} Balance response 'list' is missing, empty, or not a list. Data: {info}")
            return None

        # Find the dictionary for the targeted UNIFIED account type
        unified_account_info = next((acc for acc in account_list if acc.get('accountType') == account_type_target.value), None)
        if not unified_account_info:
            logger.warning(f"{log_prefix} Could not find account details for type '{account_type_target.value}' in response list.")
            return None

        # Extract total equity for the UNIFIED account
        total_equity_str = unified_account_info.get('totalEquity')
        total_equity = safe_decimal_conversion(total_equity_str, default=Decimal("0"), context="Total Equity")
        # Ensure equity is non-negative
        final_equity = max(Decimal("0"), total_equity)

        # Find the specific coin (e.g., USDT) within the account's coin list
        available_balance = None
        coin_list = unified_account_info.get('coin', [])
        if not isinstance(coin_list, list):
             logger.warning(f"{log_prefix} 'coin' list within account info is missing or not a list.")
             coin_list = [] # Treat as empty

        usdt_coin_info = next((coin for coin in coin_list if coin.get('coin') == usdt_symbol), None)

        if usdt_coin_info:
            # Prioritize 'availableToWithdraw' as it reflects actual usable balance
            # Fallback to 'walletBalance' or other fields if needed, but 'availableToWithdraw' is best for trading funds.
            # Note: Bybit V5 uses 'availableToWithdraw', 'availableBalance' might be deprecated or different.
            avail_str = usdt_coin_info.get('availableToWithdraw')
            if avail_str is None:
                # Fallback or alternative check if primary field missing
                avail_str = usdt_coin_info.get('availableBalance') # Check alternative if needed
                if avail_str is not None: logger.debug(f"{log_prefix} Using 'availableBalance' as 'availableToWithdraw' was missing for {usdt_symbol}.")

            available_balance = safe_decimal_conversion(avail_str, default=Decimal("0"), context=f"{usdt_symbol} Available Balance")
        else:
            logger.warning(f"{log_prefix} {usdt_symbol} details not found within the {account_type_target.value} account coin list. Assuming 0 available balance.")
            available_balance = Decimal("0") # Assume zero if coin not listed

        # Ensure available balance is non-negative
        final_available = max(Decimal("0"), available_balance)

        logger.info(f"{Fore.GREEN}{log_prefix} OK - Equity: {final_equity:.4f}, Available {usdt_symbol}: {final_available:.4f}{Style.RESET_ALL}")
        return final_equity, final_available

    except AuthenticationError as e:
        logger.error(f"{Fore.RED}{log_prefix} Authentication error fetching balance: {e}{Style.RESET_ALL}")
        # Don't retry auth errors here, let the caller handle persistent failure.
        # The retry decorator might handle initial attempts if this was called directly.
        return None
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        # Network/Server errors are suitable for retry
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Exchange error fetching balance (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for the decorator to handle retry
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error fetching balance: {e}{Style.RESET_ALL}", exc_info=False)
        return None # Don't retry generic exchange errors unless known to be transient
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- Market Data Functions ---

@retry_api_call()
async def fetch_ohlcv_paginated(
    exchange: ccxt_async.bybit,
    symbol: str,
    timeframe: str,
    app_config: AppConfig,
    since: Optional[int] = None, # Start time timestamp in milliseconds (inclusive)
    limit: Optional[int] = None, # Max number of candles to return across all pages
    max_candles_per_page: int = 1000 # Bybit V5 limit per request
) -> Optional[Union[pd.DataFrame, List[list]]]:
    """
    Fetches OHLCV data for a symbol, handling pagination automatically using Bybit V5's mechanism.

    Sorts results by timestamp ascending and ensures uniqueness.

    Args:
        exchange: Initialized Bybit async exchange instance.
        symbol: The market symbol.
        timeframe: The timeframe string (e.g., '1m', '5m', '1h', '1d').
        app_config: The application configuration object.
        since: Start time timestamp in milliseconds (optional). Fetches most recent if None.
        limit: The maximum total number of candles to fetch across all pages (optional).
               If None, fetches all available since 'since' up to exchange limits.
        max_candles_per_page: Max candles per API request (default 1000 for Bybit V5).

    Returns:
        A pandas DataFrame with OHLCV data (if pandas is available) and a DatetimeIndex (UTC).
        Columns: [open, high, low, close, volume]. Index: datetime.
        OR a list of lists: [[timestamp_ms, open, high, low, close, volume], ...].
        Returns None on critical failure (e.g., invalid symbol).
        Returns partial data if an error occurs mid-pagination.
    """
    func_name = "fetch_ohlcv_paginated"
    log_prefix = f"[{func_name}({symbol}, {timeframe})]"

    if not PANDAS_AVAILABLE:
        logger.info(f"{log_prefix} Pandas not available. OHLCV data will be returned as list of lists.")

    # Determine category for the API call parameters
    category = market_cache.get_category(symbol)
    if not category:
        logger.error(f"{Fore.RED}{log_prefix} Cannot determine V5 category for '{symbol}'. Cannot fetch OHLCV.{Style.RESET_ALL}")
        return None
    # Spot requires special handling in V5 if not using Unified Margin? Assume Unified for now.
    # if category == Category.SPOT: params['market'] = 'spot' # Example if needed

    # Validate timeframe using CCXT's parser (optional but good check)
    try:
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        logger.debug(f"{log_prefix} Parsed timeframe: {timeframe} ({timeframe_ms}ms)")
    except Exception as e_tf:
        logger.error(f"{Fore.RED}{log_prefix} Invalid timeframe string '{timeframe}': {e_tf}.{Style.RESET_ALL}")
        return None

    # Bybit V5 fetchOHLCV requires the 'category' parameter
    params = {'category': category.value}

    all_candles = []
    current_since = since # Start with the provided 'since' timestamp
    total_fetched = 0
    max_pages = 200 # Safety limit to prevent runaway loops

    logger.info(f"{Fore.BLUE}{log_prefix} Fetching OHLCV... Target limit: {limit or 'All available'}, Since: {current_since or 'Most Recent'}{Style.RESET_ALL}")

    try:
        for page_num in range(max_pages):
            # Determine the limit for this specific API call
            fetch_limit_this_page = max_candles_per_page
            if limit is not None:
                remaining_needed = limit - total_fetched
                if remaining_needed <= 0:
                    logger.debug(f"{log_prefix} Target limit of {limit} candles reached.")
                    break # Exit loop if target limit met
                fetch_limit_this_page = min(max_candles_per_page, remaining_needed)

            if fetch_limit_this_page <= 0: # Should not happen if limit logic is correct
                break

            logger.debug(f"{log_prefix} Page {page_num + 1}/{max_pages}, Fetching since={current_since}, limit={fetch_limit_this_page}, params={params}")

            # Fetch one page/chunk of candles using await exchange.fetch_ohlcv
            # The @retry_api_call decorator handles transient errors for this individual call
            candles_chunk = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since, # Use 'since' for pagination start
                limit=fetch_limit_this_page,
                params=params
            )

            if not candles_chunk:
                logger.info(f"{log_prefix} No more candles returned by exchange (or empty chunk received). End of data for range.")
                break # Exit loop if no data returned for this period

            num_in_chunk = len(candles_chunk)
            total_fetched += num_in_chunk
            first_ts_chunk = candles_chunk[0][0]
            last_ts_chunk = candles_chunk[-1][0]

            # Add the fetched chunk to the main list
            all_candles.extend(candles_chunk)

            # Log progress
            ts_to_dt_str = lambda ts: pd.to_datetime(ts, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S') if PANDAS_AVAILABLE else str(ts)
            logger.info(f"{log_prefix} Fetched {num_in_chunk} candles (Page {page_num + 1}. Range: {ts_to_dt_str(first_ts_chunk)} to {ts_to_dt_str(last_ts_chunk)}). Total collected: {total_fetched}")

            # --- Prepare for next iteration ---
            # Bybit V5 pagination: 'since' is inclusive. To get the next chunk,
            # set 'since' to the timestamp of the *last* candle received + 1ms (or timeframe duration).
            # Using last timestamp + 1ms is generally safer.
            current_since = last_ts_chunk + 1

            # Check if the exchange returned fewer candles than requested (usually indicates end of data)
            if num_in_chunk < fetch_limit_this_page:
                logger.info(f"{log_prefix} Received fewer candles ({num_in_chunk}) than requested limit ({fetch_limit_this_page}). Assuming end of available data.")
                break

            # Check again if the overall limit has now been reached after adding the chunk
            if limit is not None and total_fetched >= limit:
                logger.info(f"{log_prefix} Reached or exceeded overall target limit of {limit} candles.")
                break

            # Optional small delay between pages to be courteous to the API
            await asyncio.sleep(max(0.1, exchange.rateLimit / 2000 if exchange.rateLimit > 0 else 0.1)) # e.g., 100ms delay

        if page_num >= max_pages - 1:
             logger.warning(f"{Fore.YELLOW}{log_prefix} Reached maximum pagination limit ({max_pages} pages). Data might be incomplete.{Style.RESET_ALL}")

        # --- Process Collected Data ---
        if not all_candles:
            logger.warning(f"{log_prefix} No OHLCV candles were collected after pagination.")
            return pd.DataFrame() if PANDAS_AVAILABLE else []

        logger.info(f"{log_prefix} Total raw candles collected across all pages: {len(all_candles)}")

        # --- Sort and Remove Duplicates (Crucial) ---
        # Sort by timestamp (first element) ascending
        all_candles.sort(key=lambda x: x[0])

        # Remove duplicates based on timestamp, keeping the first occurrence (due to sort)
        unique_candles_dict = {candle[0]: candle for candle in all_candles}
        unique_candles = list(unique_candles_dict.values())

        num_duplicates = len(all_candles) - len(unique_candles)
        if num_duplicates > 0:
             logger.debug(f"{log_prefix} Removed {num_duplicates} duplicate candle timestamps.")

        # Apply final limit if specified (in case pagination slightly overshot due to chunk sizes)
        if limit is not None and len(unique_candles) > limit:
            logger.debug(f"{log_prefix} Trimming final unique candle list from {len(unique_candles)} to target limit {limit}.")
            # Keep the most recent 'limit' candles after sorting and unique check
            unique_candles = unique_candles[-limit:]

        logger.info(f"{log_prefix} Processed {len(unique_candles)} unique candles.")

        # --- Return as DataFrame or List ---
        if PANDAS_AVAILABLE and pd:
            try:
                df = pd.DataFrame(unique_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                # Convert timestamp to datetime and set as index (UTC)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('datetime')
                # Convert OHLCV columns to numeric types (robustly handles potential non-numeric data)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Optional: Check for and handle NaNs introduced by conversion if necessary
                nan_count = df[['open', 'high', 'low', 'close']].isnull().any(axis=1).sum()
                if nan_count > 0:
                    logger.warning(f"{log_prefix} Found {nan_count} rows with NaN in OHLC columns after numeric conversion. Check source data quality.")
                    # df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Option to drop rows with NaNs
                logger.success(f"{Fore.GREEN}{log_prefix} Returning {len(df)} unique candles as pandas DataFrame.{Style.RESET_ALL}")
                return df
            except Exception as e_df:
                logger.error(f"{log_prefix} Failed to create or process DataFrame from candles: {e_df}. Returning raw list instead.", exc_info=True)
                # Fallback to returning the list if DataFrame processing fails
                return unique_candles
        else:
            # Return list of lists if pandas is not available
            logger.success(f"{Fore.GREEN}{log_prefix} Returning {len(unique_candles)} unique candles as list of lists.{Style.RESET_ALL}")
            return unique_candles

    # --- Error Handling during Pagination ---
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        # These errors were not recovered by the retry decorator on fetch_ohlcv
        logger.error(f"{Fore.RED}{log_prefix} Unrecoverable API error during pagination: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
    except AuthenticationError as e:
         # Should not happen for public data, but catch just in case
         logger.error(f"{Fore.RED}{log_prefix} Authentication error during pagination (unexpected for OHLCV): {e}{Style.RESET_ALL}")
    except BadSymbol as e:
         logger.error(f"{Fore.RED}{log_prefix} Invalid symbol error during pagination: {e}{Style.RESET_ALL}")
         return None # Return None for bad symbol
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Unrecoverable Exchange error during pagination: {e}{Style.RESET_ALL}", exc_info=False)
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error during OHLCV fetching/processing: {e}{Style.RESET_ALL}", exc_info=True)

    # --- Return Partial Data if Available on Error ---
    if all_candles:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Returning PARTIAL data ({len(all_candles)} raw candles) due to error during pagination.{Style.RESET_ALL}")
        # Process the partial data as best as possible (sort, unique, format)
        try:
            all_candles.sort(key=lambda x: x[0])
            unique_candles_dict = {c[0]: c for c in all_candles}
            unique_candles = list(unique_candles_dict.values())
            if limit is not None and len(unique_candles) > limit: unique_candles = unique_candles[-limit:] # Apply limit

            if PANDAS_AVAILABLE and pd:
                 df = pd.DataFrame(unique_candles, columns=['timestamp','open','high','low','close','volume'])
                 df['datetime']=pd.to_datetime(df['timestamp'],unit='ms', utc=True)
                 df = df.set_index('datetime')
                 for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
                 return df
            else: return unique_candles
        except Exception as e_partial:
             logger.error(f"{log_prefix} Error processing partial candle data: {e_partial}. Returning raw list or None.")
             return all_candles # Return raw list as last resort
    else:
        # No data collected and an error occurred
        return None # Indicate complete failure


@retry_api_call()
async def fetch_ticker_validated(exchange: ccxt_async.bybit, symbol: str, app_config: AppConfig) -> Optional[Dict]:
    """
    Fetches the ticker for a symbol using V5 parameters and validates essential fields.

    Args:
        exchange: Initialized Bybit async exchange instance.
        symbol: The market symbol.
        app_config: The application configuration object.

    Returns:
        The validated ticker dictionary in CCXT standard format, or None on failure.
    """
    func_name = "fetch_ticker_validated"
    log_prefix = f"[{func_name}({symbol})]"
    logger.debug(f"{log_prefix} Fetching ticker...")

    category = market_cache.get_category(symbol)
    if not category:
        logger.error(f"{Fore.RED}{log_prefix} Cannot determine V5 category for '{symbol}'. Cannot fetch ticker.{Style.RESET_ALL}")
        return None

    params = {'category': category.value}
    logger.debug(f"{log_prefix} Using params: {params}")

    try:
        # Fetch ticker using CCXT's method with V5 parameters
        ticker = await exchange.fetch_ticker(symbol, params=params)

        # --- Validation ---
        if not ticker or not isinstance(ticker, dict):
            logger.error(f"{Fore.RED}{log_prefix} Received invalid or empty ticker response: {ticker}{Style.RESET_ALL}")
            return None

        # Essential keys for basic trading decisions (ensure they are not None)
        required_keys = ['symbol', 'last', 'bid', 'ask', 'timestamp']
        # Common keys that are useful but might occasionally be None depending on market state/API
        common_keys = ['datetime', 'high', 'low', 'bidVolume', 'askVolume', 'vwap', 'open', 'close', 'previousClose', 'change', 'percentage', 'average', 'baseVolume', 'quoteVolume']

        # Check for missing required keys or None values
        missing_required = [k for k in required_keys if ticker.get(k) is None]
        if missing_required:
            logger.error(f"{Fore.RED}{log_prefix} Ticker response missing required keys or values are None: {missing_required}. Data: {ticker}{Style.RESET_ALL}")
            return None

        # Check for missing common keys (log as debug/warning)
        missing_common = [k for k in common_keys if k not in ticker or ticker.get(k) is None]
        if missing_common:
            logger.debug(f"{log_prefix} Ticker missing some common keys: {missing_common}.")

        # --- Timestamp Validation (Crucial) ---
        ts_ms = ticker.get('timestamp')
        ts_log_msg = "TS: N/A"
        is_stale = False
        if ts_ms is not None and isinstance(ts_ms, int):
            now_ms = int(time.time() * 1000)
            age_ms = now_ms - ts_ms
            # Define acceptable age range (e.g., allow 120 seconds old, 10 seconds future for clock drift)
            max_age_seconds = 120
            min_age_seconds = -10
            max_diff_ms = max_age_seconds * 1000
            min_diff_ms = min_age_seconds * 1000

            age_s_str = f"{(age_ms / 1000.0):.1f}s"
            dt_str = ticker.get('datetime', f"ms:{ts_ms}") # Use ISO datetime string if available

            if age_ms > max_diff_ms or age_ms < min_diff_ms:
                logger.warning(f"{Fore.YELLOW}{log_prefix} Timestamp ({dt_str}) seems stale or invalid. Age: {age_s_str} (Allowed: {min_age_seconds}s to {max_age_seconds}s).{Style.RESET_ALL}")
                ts_log_msg = f"{Fore.YELLOW}TS: Stale ({age_s_str}){Style.RESET_ALL}"
                is_stale = True
                # Decide whether to return None or just warn based on stale TS
                # return None # Stricter approach: reject stale tickers immediately
            else:
                ts_log_msg = f"TS OK ({age_s_str} old)"
        elif ts_ms is None:
             ts_log_msg = f"{Fore.YELLOW}TS: Missing{Style.RESET_ALL}"
             is_stale = True # Treat missing timestamp as stale/unreliable
        else:
             ts_log_msg = f"{Fore.YELLOW}TS: Invalid Type ({type(ts_ms).__name__}){Style.RESET_ALL}"
             is_stale = True # Treat invalid type as stale/unreliable

        # Log summary including price formatting
        last_px_str = format_price(exchange, symbol, ticker.get('last')) or "N/A"
        bid_px_str = format_price(exchange, symbol, ticker.get('bid')) or "N/A"
        ask_px_str = format_price(exchange, symbol, ticker.get('ask')) or "N/A"
        log_color = Fore.YELLOW if is_stale else Fore.GREEN
        logger.info(f"{log_color}{log_prefix} OK: Last={last_px_str}, Bid={bid_px_str}, Ask={ask_px_str} | {ts_log_msg}{Style.RESET_ALL}")

        # Optional: Return None if the ticker is deemed too stale, based on `is_stale` flag
        # if is_stale:
        #     logger.error(f"{log_prefix} Rejecting ticker due to stale or invalid timestamp.")
        #     return None

        return ticker

    except BadSymbol as e:
         logger.error(f"{Fore.RED}{log_prefix} Invalid symbol error fetching ticker: {e}{Style.RESET_ALL}")
         return None # BadSymbol is not typically retryable
    except AuthenticationError as e:
         # Should not happen for public data, but handle defensively
         logger.error(f"{Fore.RED}{log_prefix} Authentication error fetching ticker (unexpected): {e}{Style.RESET_ALL}")
         return None # Auth errors usually require intervention
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        # Network/Server errors will be retried by the decorator
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Exchange error fetching ticker (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for decorator
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error fetching ticker: {e}{Style.RESET_ALL}", exc_info=False)
        return None # Non-transient exchange error
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching ticker: {e}{Style.RESET_ALL}", exc_info=True)
        return None


@retry_api_call()
async def fetch_l2_order_book_validated(
    exchange: ccxt_async.bybit, symbol: str, limit: int, app_config: AppConfig
) -> Optional[Dict[str, Any]]:
    """
    Fetches L2 order book using V5 parameters and performs basic validation.

    Args:
        exchange: Initialized Bybit async exchange instance.
        symbol: The market symbol.
        limit: The number of bids/asks levels to fetch (e.g., 5, 25, 50). Check API limits.
        app_config: The application configuration object.

    Returns:
        The validated L2 order book dictionary (CCXT format: {'bids': [[price, amount], ...], 'asks': [...]}),
        or None on failure.
    """
    func_name = "fetch_l2_order_book"; log_prefix = f"[{func_name}({symbol}, limit={limit})]"
    logger.debug(f"{log_prefix} Fetching L2 Order Book...")

    category = market_cache.get_category(symbol)
    if not category:
        logger.error(f"{Fore.RED}{log_prefix} Cannot determine V5 category for '{symbol}'. Cannot fetch order book.{Style.RESET_ALL}")
        return None

    # Bybit V5 fetchOrderBook requires category
    params = {'category': category.value}

    try:
        # Fetch L2 order book (fetchL2OrderBook is an alias often used)
        ob = await exchange.fetch_l2_order_book(symbol, limit=limit, params=params)

        # --- Basic Validation ---
        if not ob or not isinstance(ob, dict):
            logger.error(f"{Fore.RED}{log_prefix} Invalid or empty order book response: {ob}{Style.RESET_ALL}")
            return None

        bids, asks = ob.get('bids'), ob.get('asks')
        # Check if bids and asks are present and are lists
        if bids is None or asks is None or not isinstance(bids, list) or not isinstance(asks, list):
            logger.error(f"{Fore.RED}{log_prefix} Order book response missing 'bids' or 'asks' list. Data: {ob}{Style.RESET_ALL}")
            return None

        # Check if sides are non-empty (warning if one side is missing, might be valid in thin markets)
        if not bids: logger.warning(f"{Fore.YELLOW}{log_prefix} Order book 'bids' side is empty.{Style.RESET_ALL}")
        if not asks: logger.warning(f"{Fore.YELLOW}{log_prefix} Order book 'asks' side is empty.{Style.RESET_ALL}")
        # Could potentially return None if BOTH sides are empty, depending on requirements

        # Optional deeper validation: Check format [price, amount], check sorting, check bid < ask
        if bids and asks:
            best_bid_price = safe_decimal_conversion(bids[0][0])
            best_ask_price = safe_decimal_conversion(asks[0][0])
            if best_bid_price is None or best_ask_price is None:
                 logger.warning(f"{Fore.YELLOW}{log_prefix} Could not validate bid/ask prices in OB.{Style.RESET_ALL}")
            elif best_bid_price >= best_ask_price:
                 logger.warning(f"{Fore.YELLOW}{log_prefix} Order book crossed or zero spread detected! Best Bid ({best_bid_price}) >= Best Ask ({best_ask_price}).{Style.RESET_ALL}")

            # Check individual entry format (example for first bid/ask)
            if bids and not (isinstance(bids[0], list) and len(bids[0]) == 2): logger.warning(f"{Fore.YELLOW}{log_prefix} First bid entry format seems incorrect.{Style.RESET_ALL}")
            if asks and not (isinstance(asks[0], list) and len(asks[0]) == 2): logger.warning(f"{Fore.YELLOW}{log_prefix} First ask entry format seems incorrect.{Style.RESET_ALL}")

        logger.info(f"{Fore.GREEN}{log_prefix} OK: Fetched L2 OB. Bids={len(bids)}, Asks={len(asks)}{Style.RESET_ALL}")
        return ob

    except BadSymbol as e:
         logger.error(f"{Fore.RED}{log_prefix} Invalid symbol error fetching order book: {e}{Style.RESET_ALL}")
         return None # Not retryable
    except AuthenticationError as e:
         logger.error(f"{Fore.RED}{log_prefix} Authentication error fetching order book (unexpected): {e}{Style.RESET_ALL}")
         return None
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Exchange error fetching OB (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for decorator
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error fetching order book: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching order book: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- Order Management Functions ---

# Apply retry decorator with specific settings for market orders (maybe fewer retries)
@retry_api_call(max_retries_override=1, initial_delay_override=0.5)
async def place_market_order(
    exchange: ccxt_async.bybit, symbol: str, side: Side, amount: Decimal, app_config: AppConfig,
    is_reduce_only: bool = False,
    time_in_force: TimeInForce = TimeInForce.IOC, # IOC or FOK common for market
    client_order_id: Optional[str] = None,
    position_idx: Optional[PositionIdx] = None, # For hedge mode if needed
    reason: str = "Market Order" # For logging and potentially client_order_id generation
) -> Optional[Dict]:
    """
    Places a market order using Bybit V5 parameters. Includes basic validation.
    Consider using `place_market_order_slippage_check` for pre-flight spread check.

    Args:
        exchange: Initialized Bybit async exchange instance.
        symbol: The market symbol.
        side: Side enum/literal ('Buy' or 'Sell').
        amount: The quantity to trade (as Decimal, positive).
        app_config: The application configuration object.
        is_reduce_only: Set to True for closing/reducing positions only.
        time_in_force: Time in force (IOC or FOK recommended for market).
        client_order_id: Custom client order ID (optional, max length applies).
        position_idx: Specify position index (0, 1, or 2) for Hedge Mode. Use config default if None.
        reason: Short description for logging/order ID generation.

    Returns:
        The order dictionary returned by CCXT upon successful placement, or None on failure.
    """
    func_name = "place_market_order"
    action_str = "ReduceOnly" if is_reduce_only else "Open/Increase"
    log_prefix = f"[{func_name}({symbol}, {side.value}, Qty:{amount:.8f}, {action_str}, {reason})]" # Use side.value
    api_conf = app_config.api # Convenience alias

    # --- Input Validation ---
    if amount <= api_conf.position_qty_epsilon: # Use epsilon for near-zero check
        logger.error(f"{Fore.RED}{log_prefix} Invalid order amount ({amount}). Must be significantly greater than 0.{Style.RESET_ALL}")
        return None
    if side not in [Side.BUY, Side.SELL]: # Check against actual enum/literal values
         logger.error(f"{Fore.RED}{log_prefix} Invalid side: {side}. Must be Side.BUY or Side.SELL.{Style.RESET_ALL}"); return None
    if time_in_force not in [TimeInForce.IOC, TimeInForce.FOK]:
         logger.warning(f"{Fore.YELLOW}{log_prefix} Unusual TimeInForce '{time_in_force.value}' for Market order. Using it, but IOC/FOK recommended.{Style.RESET_ALL}")

    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)
    if not market or not category:
        logger.error(f"{Fore.RED}{log_prefix} Market/Category info unavailable for '{symbol}'. Cannot place order.{Style.RESET_ALL}")
        return None

    # Format amount according to market precision (crucial)
    formatted_amount_str = format_amount(exchange, symbol, amount)
    if formatted_amount_str is None: # format_amount returns None on critical error
        logger.error(f"{Fore.RED}{log_prefix} Failed to format order amount {amount} for market precision.{Style.RESET_ALL}")
        return None
    # Convert formatted string back to Decimal for comparison/logging if needed
    formatted_amount = safe_decimal_conversion(formatted_amount_str, default=Decimal("-1"))
    if formatted_amount <= 0:
         logger.error(f"{Fore.RED}{log_prefix} Formatted amount '{formatted_amount_str}' is invalid or zero after formatting.{Style.RESET_ALL}"); return None

    # Determine effective position index (use config default if not provided)
    effective_pos_idx = position_idx if position_idx is not None else app_config.strategy.position_idx

    # --- Prepare Order Parameters for V5 ---
    # Use create_order method with specific params for V5
    params = {
        'category': category.value,
        'timeInForce': time_in_force.value,
        'reduceOnly': is_reduce_only,
        # 'orderLinkId': client_order_id, # Use orderLinkId for custom ID in V5
        # Add positionIdx ONLY if it's relevant (Hedge Mode or explicitly non-zero)
        # Bybit docs suggest it's needed if account is in Hedge Mode (posMode=3)
        # If in One-Way (posMode=0), positionIdx=0 is implied and might not be needed.
        # Check Bybit docs for create_order endpoint specifics based on account mode.
        # Safest is often to include it if non-zero or if hedge mode is possible.
        'positionIdx': effective_pos_idx.value,
        # 'triggerPrice', 'stopLoss', 'takeProfit' etc. are for conditional/TP/SL orders, not basic market orders.
    }
    # Add client_order_id if provided
    if client_order_id:
        # Ensure length constraints are met (Bybit V5: usually 36 chars for orderLinkId)
        max_id_len = 36
        if len(client_order_id) > max_id_len:
            logger.warning(f"{Fore.YELLOW}{log_prefix} Provided client_order_id '{client_order_id}' exceeds max length ({max_id_len}). Truncating.{Style.RESET_ALL}")
            params['orderLinkId'] = client_order_id[:max_id_len]
        else:
            params['orderLinkId'] = client_order_id

    logger.info(f"{Fore.CYAN}{log_prefix} Submitting Market Order... Amount: {formatted_amount_str}, Params: {params}{Style.RESET_ALL}")

    try:
        # Use exchange.create_order() for placing orders
        order_result = await exchange.create_order(
            symbol=symbol,
            type='market', # Explicitly 'market' type
            side=side.value, # Pass 'Buy' or 'Sell' string value
            amount=float(formatted_amount), # CCXT often expects float amount
            price=None, # Market orders don't have a price
            params=params
        )

        # Basic validation of the returned order structure
        if not order_result or not isinstance(order_result, dict) or 'id' not in order_result:
            logger.error(f"{Fore.RED}{log_prefix} Market order submission failed or returned invalid structure: {order_result}{Style.RESET_ALL}")
            # Consider alerting here as submission failure is critical
            asyncio.create_task(send_sms_alert_async(f"ALERT: Market order submit FAIL {symbol} {side.value}", app_config.sms))
            return None

        order_id_fmt = format_order_id(order_result.get('id'))
        status = order_result.get('status', 'unknown')
        filled_qty = safe_decimal_conversion(order_result.get('filled', '0'), default=Decimal(0))
        avg_price_str = format_price(exchange, symbol, order_result.get('average')) or "N/A"

        logger.success(f"{Fore.GREEN}{log_prefix} Market Order Submitted OK. ID: {order_id_fmt}, Status: {status}, Filled: {filled_qty}, AvgPx: {avg_price_str}{Style.RESET_ALL}")
        # NOTE: Market orders might fill instantly (status 'closed') or partially/not at all (status 'open'/'rejected').
        # Further checks on status/filled amount might be needed depending on strategy logic.
        return order_result

    # --- Error Handling ---
    except InsufficientFunds as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix} Insufficient Funds: {e}{Style.RESET_ALL}")
        asyncio.create_task(send_sms_alert_async(f"ALERT: Insufficient funds for {symbol} {side.value} order", app_config.sms))
        return None # Definitely failed
    except InvalidOrder as e:
        # Includes issues like size precision, price limits (though not for market), reduceOnly mismatch etc.
        logger.error(f"{Fore.RED}{log_prefix} Invalid Order parameters: {e}{Style.RESET_ALL}")
        # Check for specific V5 error codes if possible
        # Example: 110007 = qty is incorrect; 110012 = order cost issue; 110040 = reduceOnly issue
        error_code_str = str(getattr(e, 'code', None))
        logger.error(f"{Fore.RED}{log_prefix} Details: Code={error_code_str}. Check amount precision, leverage, reduceOnly flag, position size.{Style.RESET_ALL}")
        return None
    except OrderImmediatelyFillable as e:
         # Should not happen for market orders, but handle defensively
         logger.warning(f"{Fore.YELLOW}{log_prefix} OrderImmediatelyFillable error on Market order (unexpected): {e}{Style.RESET_ALL}")
         raise e # Might indicate a transient issue, allow retry
    except AuthenticationError as e:
         logger.critical(f"{Back.RED}{Fore.WHITE}{log_prefix} Authentication Error placing order: {e}. Check API keys/permissions.{Style.RESET_ALL}")
         asyncio.create_task(send_sms_alert_async(f"CRITICAL: Auth Error placing {symbol} order!", app_config.sms))
         return None # Auth errors are critical
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Server error placing order (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for decorator
    except ExchangeError as e:
        # Catch other specific exchange errors
        logger.error(f"{Fore.RED}{log_prefix} ExchangeError placing order: {e}{Style.RESET_ALL}", exc_info=False)
        # Check for potentially retryable ExchangeErrors if needed, otherwise return None
        # Example: Check for specific error codes indicating temporary issues
        # error_code = getattr(e, 'code', None)
        # if error_code == 'SOME_TRANSIENT_CODE': raise e
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# TODO: Implement place_market_order_slippage_check if needed - involves fetching OB before placing.

@retry_api_call()
async def place_limit_order(
    exchange: ccxt_async.bybit, symbol: str, side: Side, amount: Decimal, price: Decimal, app_config: AppConfig,
    is_reduce_only: bool = False,
    is_post_only: bool = False,
    time_in_force: TimeInForce = TimeInForce.GTC, # GTC default for limit
    client_order_id: Optional[str] = None,
    position_idx: Optional[PositionIdx] = None,
    reason: str = "Limit Order"
) -> Optional[Dict]:
    """
    Places a limit order using Bybit V5 parameters.

    Args:
        exchange: Initialized Bybit async exchange instance.
        symbol: The market symbol.
        side: Side enum/literal ('Buy' or 'Sell').
        amount: The quantity to trade (as Decimal, positive).
        price: The limit price (as Decimal).
        app_config: The application configuration object.
        is_reduce_only: Set to True for closing/reducing positions only.
        is_post_only: If True, ensures the order only adds liquidity (rejected if it would match immediately).
        time_in_force: Time in force (GTC, IOC, FOK, PostOnly). If is_post_only is True, TIF should ideally be PostOnly.
        client_order_id: Custom client order ID (optional).
        position_idx: Specify position index (0, 1, or 2) for Hedge Mode. Use config default if None.
        reason: Short description for logging/order ID generation.

    Returns:
        The order dictionary returned by CCXT upon successful placement, or None on failure.
    """
    func_name = "place_limit_order"
    action_str = "ReduceOnly" if is_reduce_only else "Open/Increase"
    post_only_str = "PostOnly" if is_post_only else ""
    log_prefix = f"[{func_name}({symbol}, {side.value}, Qty:{amount:.8f}, Px:{price:.8f}, {action_str}{post_only_str}, {reason})]"
    api_conf = app_config.api

    # --- Input Validation ---
    if amount <= api_conf.position_qty_epsilon: logger.error(f"{Fore.RED}{log_prefix} Invalid order amount ({amount}).{Style.RESET_ALL}"); return None
    if price <= 0: logger.error(f"{Fore.RED}{log_prefix} Invalid order price ({price}). Must be positive.{Style.RESET_ALL}"); return None
    if side not in [Side.BUY, Side.SELL]: logger.error(f"{Fore.RED}{log_prefix} Invalid side: {side}.{Style.RESET_ALL}"); return None

    # Handle PostOnly logic: If is_post_only is True, timeInForce MUST be PostOnly
    effective_tif = time_in_force
    if is_post_only:
        if time_in_force != TimeInForce.POST_ONLY:
            logger.warning(f"{Fore.YELLOW}{log_prefix} is_post_only=True but TIF is '{time_in_force.value}'. Forcing TIF to PostOnly.{Style.RESET_ALL}")
            effective_tif = TimeInForce.POST_ONLY
    elif time_in_force == TimeInForce.POST_ONLY and not is_post_only:
         logger.warning(f"{Fore.YELLOW}{log_prefix} TIF is PostOnly but is_post_only=False. Behavior might be unexpected. Consider setting is_post_only=True.{Style.RESET_ALL}")
         # Proceed with PostOnly TIF as requested

    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)
    if not market or not category:
        logger.error(f"{Fore.RED}{log_prefix} Market/Category info unavailable for '{symbol}'. Cannot place order.{Style.RESET_ALL}")
        return None

    # Format amount and price according to market precision
    formatted_amount_str = format_amount(exchange, symbol, amount)
    formatted_price_str = format_price(exchange, symbol, price)
    if formatted_amount_str is None or formatted_price_str is None:
        logger.error(f"{Fore.RED}{log_prefix} Failed to format order amount/price for market precision.{Style.RESET_ALL}")
        return None
    formatted_amount = safe_decimal_conversion(formatted_amount_str, default=Decimal("-1"))
    formatted_price = safe_decimal_conversion(formatted_price_str, default=Decimal("-1"))
    if formatted_amount <= 0 or formatted_price <= 0:
         logger.error(f"{Fore.RED}{log_prefix} Formatted amount/price invalid after formatting.{Style.RESET_ALL}"); return None

    # Determine effective position index
    effective_pos_idx = position_idx if position_idx is not None else app_config.strategy.position_idx

    # --- Prepare Order Parameters for V5 ---
    params = {
        'category': category.value,
        'timeInForce': effective_tif.value,
        'reduceOnly': is_reduce_only,
        'positionIdx': effective_pos_idx.value,
        # 'postOnly': is_post_only, # PostOnly is controlled via timeInForce='PostOnly' in V5
    }
    if client_order_id:
        max_id_len = 36
        params['orderLinkId'] = client_order_id[:max_id_len] if len(client_order_id) > max_id_len else client_order_id


    logger.info(f"{Fore.CYAN}{log_prefix} Submitting Limit Order... Amount: {formatted_amount_str}, Price: {formatted_price_str}, TIF: {effective_tif.value}, Params: {params}{Style.RESET_ALL}")

    try:
        order_result = await exchange.create_order(
            symbol=symbol,
            type='limit', # Explicitly 'limit' type
            side=side.value,
            amount=float(formatted_amount), # Convert to float for CCXT
            price=float(formatted_price),   # Convert to float for CCXT
            params=params
        )

        if not order_result or not isinstance(order_result, dict) or 'id' not in order_result:
            logger.error(f"{Fore.RED}{log_prefix} Limit order submission failed or returned invalid structure: {order_result}{Style.RESET_ALL}")
            asyncio.create_task(send_sms_alert_async(f"ALERT: Limit order submit FAIL {symbol} {side.value}", app_config.sms))
            return None

        order_id_fmt = format_order_id(order_result.get('id'))
        status = order_result.get('status', 'unknown') # Should be 'open' or 'rejected' typically
        logger.success(f"{Fore.GREEN}{log_prefix} Limit Order Submitted OK. ID: {order_id_fmt}, Status: {status}{Style.RESET_ALL}")
        return order_result

    # --- Error Handling (Similar to Market Order, with PostOnly considerations) ---
    except InsufficientFunds as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix} Insufficient Funds: {e}{Style.RESET_ALL}")
        asyncio.create_task(send_sms_alert_async(f"ALERT: Insufficient funds for {symbol} {side.value} limit order", app_config.sms))
        return None
    except OrderImmediatelyFillable as e:
         # This specifically happens when TIF is PostOnly and the order would cross the spread
         logger.warning(f"{Fore.YELLOW}{log_prefix} PostOnly Limit Order Rejected (would fill immediately): {e}{Style.RESET_ALL}")
         # This is often expected behavior for PostOnly, not necessarily a critical error.
         return None # Indicate order was not placed
    except InvalidOrder as e:
        logger.error(f"{Fore.RED}{log_prefix} Invalid Order parameters: {e}{Style.RESET_ALL}")
        error_code_str = str(getattr(e, 'code', None))
        logger.error(f"{Fore.RED}{log_prefix} Details: Code={error_code_str}. Check amount/price precision, leverage, reduceOnly flag, etc.{Style.RESET_ALL}")
        return None
    except AuthenticationError as e:
         logger.critical(f"{Back.RED}{Fore.WHITE}{log_prefix} Authentication Error: {e}.{Style.RESET_ALL}")
         asyncio.create_task(send_sms_alert_async(f"CRITICAL: Auth Error placing {symbol} limit order!", app_config.sms))
         return None
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Server error (will be retried): {e}.{Style.RESET_ALL}")
        raise e
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} ExchangeError placing limit order: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error placing limit order: {e}{Style.RESET_ALL}", exc_info=True)
        return None


@retry_api_call()
async def place_stop_order(
    exchange: ccxt_async.bybit, symbol: str, side: Side, order_type: OrderType, amount: Decimal, app_config: AppConfig,
    trigger_price: Decimal,
    stop_loss_price: Optional[Decimal] = None, # For SL orders attached to entry
    take_profit_price: Optional[Decimal] = None, # For TP orders attached to entry
    is_reduce_only: bool = False,
    trigger_by: TriggerBy = TriggerBy.LAST_PRICE,
    trigger_direction: Optional[TriggerDirection] = None, # Required for TP/SL, inferred otherwise
    position_idx: Optional[PositionIdx] = None,
    client_order_id: Optional[str] = None,
    reason: str = "Stop Order"
) -> Optional[Dict]:
    """
    Places a conditional (stop) order using Bybit V5 parameters.
    Can be used for Stop Market, Stop Limit, Take Profit Market, Take Profit Limit.
    Also supports attaching basic TP/SL to the conditional order itself (V5 feature).

    Args:
        exchange: Initialized Bybit async exchange instance.
        symbol: The market symbol.
        side: Side enum/literal ('Buy' or 'Sell'). Determines the direction AFTER trigger.
        order_type: OrderType enum/literal ('Market' or 'Limit'). The type of order placed AFTER trigger.
        amount: The quantity to trade (as Decimal, positive).
        app_config: The application configuration object.
        trigger_price: The price at which the conditional order activates (as Decimal).
        stop_loss_price: Optional. Price for an attached Stop Loss order.
        take_profit_price: Optional. Price for an attached Take Profit order.
        is_reduce_only: Set to True for closing/reducing positions only.
        trigger_by: TriggerBy enum/literal (LastPrice, MarkPrice, IndexPrice). Default LastPrice.
        trigger_direction: TriggerDirection enum/literal (Rise=1, Fall=2). Inferred if possible, required for TP/SL.
        position_idx: Specify position index (0, 1, or 2) for Hedge Mode. Use config default if None.
        client_order_id: Custom client order ID (optional).
        reason: Short description for logging.

    Returns:
        The order dictionary returned by CCXT upon successful placement, or None on failure.
    """
    func_name = "place_stop_order"
    log_prefix = f"[{func_name}({symbol}, {side.value}, {order_type.value}@{trigger_price:.8f}, Qty:{amount:.8f}, {reason})]"
    api_conf = app_config.api

    # --- Input Validation ---
    if amount <= api_conf.position_qty_epsilon: logger.error(f"{Fore.RED}{log_prefix} Invalid order amount ({amount}).{Style.RESET_ALL}"); return None
    if trigger_price <= 0: logger.error(f"{Fore.RED}{log_prefix} Invalid trigger price ({trigger_price}). Must be positive.{Style.RESET_ALL}"); return None
    if order_type == OrderType.LIMIT: logger.error(f"{Fore.RED}{log_prefix} Stop Limit orders not fully implemented in this helper. Use Stop Market.{Style.RESET_ALL}"); return None # TODO: Add price param for limit orders
    if side not in [Side.BUY, Side.SELL]: logger.error(f"{Fore.RED}{log_prefix} Invalid side: {side}.{Style.RESET_ALL}"); return None

    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)
    if not market or not category:
        logger.error(f"{Fore.RED}{log_prefix} Market/Category info unavailable for '{symbol}'. Cannot place order.{Style.RESET_ALL}")
        return None

    # Format amount and prices according to market precision
    formatted_amount_str = format_amount(exchange, symbol, amount)
    formatted_trigger_px_str = format_price(exchange, symbol, trigger_price)
    formatted_sl_px_str = format_price(exchange, symbol, stop_loss_price) if stop_loss_price else None
    formatted_tp_px_str = format_price(exchange, symbol, take_profit_price) if take_profit_price else None

    if formatted_amount_str is None or formatted_trigger_px_str is None:
        logger.error(f"{Fore.RED}{log_prefix} Failed to format amount or trigger price.{Style.RESET_ALL}")
        return None
    if stop_loss_price and formatted_sl_px_str is None: logger.error(f"{Fore.RED}{log_prefix} Failed to format SL price.{Style.RESET_ALL}"); return None
    if take_profit_price and formatted_tp_px_str is None: logger.error(f"{Fore.RED}{log_prefix} Failed to format TP price.{Style.RESET_ALL}"); return None

    formatted_amount = safe_decimal_conversion(formatted_amount_str, default=Decimal("-1"))
    if formatted_amount <= 0: logger.error(f"{Fore.RED}{log_prefix} Formatted amount invalid.{Style.RESET_ALL}"); return None

    # Determine effective position index
    effective_pos_idx = position_idx if position_idx is not None else app_config.strategy.position_idx

    # Infer trigger direction if not provided (basic logic for standard stops)
    effective_trigger_dir = trigger_direction
    if effective_trigger_dir is None:
        # If side is Buy, usually triggered by price rising (e.g., stop buy entry, SL on short)
        # If side is Sell, usually triggered by price falling (e.g., stop sell entry, SL on long)
        # This is a simplification; TP orders need explicit direction.
        if side == Side.BUY: effective_trigger_dir = TriggerDirection.RISE # Price must rise to trigger buy/cover
        else: effective_trigger_dir = TriggerDirection.FALL   # Price must fall to trigger sell/SL
        logger.debug(f"{log_prefix} Inferred triggerDirection={effective_trigger_dir.value} based on side={side.value}.")
    # TODO: Add validation: For TP orders, direction must be opposite to side action. For SL, direction must be same as side action requires loss.


    # --- Prepare Order Parameters for V5 Conditional Order ---
    # Uses create_order with 'stop' flag or specific params for V5 conditional orders
    # Bybit V5 endpoint: POST /v5/order/create
    params = {
        'category': category.value,
        'triggerPrice': formatted_trigger_px_str,
        'triggerDirection': effective_trigger_dir.value, # 1: Rise, 2: Fall
        'triggerBy': trigger_by.value,
        'reduceOnly': is_reduce_only,
        'positionIdx': effective_pos_idx.value,
        # 'orderFilter': 'StopOrder', # Indicate this is a conditional order structure
        # V5 seems to infer conditional from presence of triggerPrice

        # Optional attached TP/SL (ensure prices are formatted strings)
        # Note: These create separate conditional TP/SL orders linked to the main one.
        'tpslMode': StopLossTakeProfitMode.FULL.value, # Usually 'Full' for bot logic
        'slTriggerBy': trigger_by.value, # Use same trigger basis for SL/TP usually
        'tpTriggerBy': trigger_by.value,
    }
    if formatted_sl_px_str: params['stopLoss'] = formatted_sl_px_str
    if formatted_tp_px_str: params['takeProfit'] = formatted_tp_px_str

    # Add client ID if provided
    if client_order_id:
        max_id_len = 36
        params['orderLinkId'] = client_order_id[:max_id_len] if len(client_order_id) > max_id_len else client_order_id

    logger.info(f"{Fore.CYAN}{log_prefix} Submitting Conditional Order... Amount: {formatted_amount_str}, Trigger: {formatted_trigger_px_str}, Type: {order_type.value}, Params: {params}{Style.RESET_ALL}")

    try:
        # Use create_order, CCXT handles mapping to conditional structure for Bybit V5
        order_result = await exchange.create_order(
            symbol=symbol,
            type=order_type.value.lower(), # 'market' or 'limit' (after trigger)
            side=side.value,
            amount=float(formatted_amount),
            price=None, # Price is for limit orders *after* trigger, not used here for market stop
            params=params
        )

        if not order_result or not isinstance(order_result, dict) or 'id' not in order_result:
            logger.error(f"{Fore.RED}{log_prefix} Stop order submission failed or returned invalid structure: {order_result}{Style.RESET_ALL}")
            asyncio.create_task(send_sms_alert_async(f"ALERT: Stop order submit FAIL {symbol} {side.value}", app_config.sms))
            return None

        order_id_fmt = format_order_id(order_result.get('id'))
        status = order_result.get('status', 'unknown') # Should be 'open' (waiting for trigger) or 'rejected'
        logger.success(f"{Fore.GREEN}{log_prefix} Conditional Order Submitted OK. ID: {order_id_fmt}, Status: {status}{Style.RESET_ALL}")
        return order_result

    # --- Error Handling ---
    except InsufficientFunds as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix} Insufficient Funds: {e}{Style.RESET_ALL}")
        asyncio.create_task(send_sms_alert_async(f"ALERT: Insufficient funds for {symbol} stop order", app_config.sms))
        return None
    except InvalidOrder as e:
        logger.error(f"{Fore.RED}{log_prefix} Invalid Order parameters: {e}{Style.RESET_ALL}")
        error_code_str = str(getattr(e, 'code', None))
        # Common V5 codes: 110007 (qty), 110017 (trigger price), 110040 (reduceOnly), 110053 (TP/SL price invalid)
        logger.error(f"{Fore.RED}{log_prefix} Details: Code={error_code_str}. Check amount, prices, trigger direction, reduceOnly, TP/SL values.{Style.RESET_ALL}")
        return None
    except AuthenticationError as e:
         logger.critical(f"{Back.RED}{Fore.WHITE}{log_prefix} Authentication Error: {e}.{Style.RESET_ALL}")
         asyncio.create_task(send_sms_alert_async(f"CRITICAL: Auth Error placing {symbol} stop order!", app_config.sms))
         return None
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Server error (will be retried): {e}.{Style.RESET_ALL}")
        raise e
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} ExchangeError placing stop order: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{Fore.
