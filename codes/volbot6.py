Okay, here is the enhanced version of the log text, incorporating analysis, clearer structure, and explanations for the observed issues.

---

**Enhanced Log Analysis: Volbot Execution Issues (2025-04-24)**

This log details several execution attempts and failures of different versions of the `volbot` script. The primary issues identified are a runtime `AttributeError` during indicator calculation in `volbot2.py` and an initialization failure due to an API/balance fetch error in `volbot4.py`.

**1. Runtime Error in `volbot2.py` (Analysis Phase Failure)**

*   **Timestamp:** 2025-04-24 ~16:27:09 - 16:27:31
*   **Core Error:** `AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?`
*   **Location:** `/data/data/com.termux/files/home/worldguide/codes/volbot2.py`, line **688**, within the `calculate_volatility_levels` function. This function was called by `_calculate_strategy_indicators` at line **1074**.
*   **Cause:** The code at line 688 attempts to call the pandas DataFrame method `.fillna(0)` on an object that is actually a NumPy array. NumPy arrays do not have a `fillna` method (the error suggests `.fill`). This indicates a likely type mismatch or incorrect data handling within the `calculate_volatility_levels` function, where a pandas operation is expected but a NumPy array is present.
*   **Impact:**
    *   This error consistently occurred during the analysis cycle for multiple trading pairs:
        *   `1000000BABYDOGE/USDT:USDT`
        *   `1000000CHEEMS/USDT:USDT`
        *   `1000000MOG/USDT:USDT`
        *   `1000000PEIPEI/USDT:USDT` (Implied, occurred just before shutdown)
    *   **Consequences:**
        *   `WARNING`: "Cannot update latest state: Processed DataFrame is empty..." - The error likely corrupted or emptied the DataFrame being processed.
        *   `ERROR`: "Failed to calculate indicators or update state... Skipping further analysis." - The bot could not proceed with the strategy logic for these symbols due to the calculation failure.
*   **Secondary Warning:** For CHEEMS, MOG, and PEIPEI, a warning about "Insufficient data (1000 points)... recommend min: ~1050" was logged *before* the error occurred. While potentially causing inaccurate results *if* calculations succeeded, the `AttributeError` prevented calculations altogether.
*   **Shutdown:** The `volbot2.py` process was terminated via KeyboardInterrupt (`^C`) at `16:27:38`.
    *   **CRITICAL OPERATIONAL WARNING:** "Trading was enabled. Consider manually checking open positions/orders." This is vital as the abrupt shutdown might leave trades unmanaged.

**(Original Log Snippet - `volbot2.py` Runtime Errors)**
```log
dguide/codes/volbot2.py", line 688, in calculate_volatility_levels
    ).fillna(0)
      ^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?
2025-04-24 16:27:09 - WARNING  - [volbot_1000000BABYDOGE_USDT_USDT] - Cannot update latest state: Processed DataFrame is empty for 1000000BABYDOGE/USDT:USDT.
2025-04-24 16:27:09 - ERROR    - [volbot_1000000BABYDOGE_USDT_USDT] - Failed to calculate indicators or update state for 1000000BABYDOGE/USDT:USDT. Skipping further analysis.
2025-04-24 16:27:09 - INFO     - [volbot_1000000CHEEMS_USDT_USDT] - --- Starting analysis cycle for 1000000CHEEMS/USDT:USDT ---
2025-04-24 16:27:09 - INFO     - [volbot_1000000CHEEMS_USDT_USDT] - Successfully fetched and processed 1000 klines for 1000000CHEEMS/USDT:USDT 3m
2025-04-24 16:27:09 - WARNING  - [volbot_1000000CHEEMS_USDT_USDT] - Insufficient data (1000 points) for 1000000CHEEMS/USDT:USDT to calculate all indicators reliably (recommend min: ~1050). Results may be inaccurate or contain NaNs.
2025-04-24 16:27:09 - INFO     - [volbot_1000000CHEEMS_USDT_USDT] - Calculating Volbot strategy indicators...
2025-04-24 16:27:09 - INFO     - [volbot_1000000CHEEMS_USDT_USDT] - Calculating Volumatic Trend Levels...
2025-04-24 16:27:21 - ERROR    - [volbot_1000000CHEEMS_USDT_USDT] - Error calculating indicators for 1000000CHEEMS/USDT:USDT: 'numpy.ndarray' object has no attribute 'fillna'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/codes/volbot2.py", line 1074, in _calculate_strategy_indicators
    df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/home/worldguide/codes/volbot2.py", line 688, in calculate_volatility_levels
    ).fillna(0)
      ^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?
2025-04-24 16:27:21 - WARNING  - [volbot_1000000CHEEMS_USDT_USDT] - Cannot update latest state: Processed DataFrame is empty for 1000000CHEEMS/USDT:USDT.
2025-04-24 16:27:21 - ERROR    - [volbot_1000000CHEEMS_USDT_USDT] - Failed to calculate indicators or update state for 1000000CHEEMS/USDT:USDT. Skipping further analysis.
2025-04-24 16:27:21 - INFO     - [volbot_1000000MOG_USDT_USDT] - --- Starting analysis cycle for 1000000MOG/USDT:USDT ---
2025-04-24 16:27:21 - INFO     - [volbot_1000000MOG_USDT_USDT] - Successfully fetched and processed 1000 klines for 1000000MOG/USDT:USDT 3m
2025-04-24 16:27:21 - WARNING  - [volbot_1000000MOG_USDT_USDT] - Insufficient data (1000 points) for 1000000MOG/USDT:USDT to calculate all indicators reliably (recommend min: ~1050). Results may be inaccurate or contain NaNs.
2025-04-24 16:27:21 - INFO     - [volbot_1000000MOG_USDT_USDT] - Calculating Volbot strategy indicators...
2025-04-24 16:27:21 - INFO     - [volbot_1000000MOG_USDT_USDT] - Calculating Volumatic Trend Levels...
2025-04-24 16:27:31 - ERROR    - [volbot_1000000MOG_USDT_USDT] - Error calculating indicators for 1000000MOG/USDT:USDT: 'numpy.ndarray' object has no attribute 'fillna'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/codes/volbot2.py", line 1074, in _calculate_strategy_indicators
    df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/home/worldguide/codes/volbot2.py", line 688, in calculate_volatility_levels
    ).fillna(0)
      ^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?
2025-04-24 16:27:31 - WARNING  - [volbot_1000000MOG_USDT_USDT] - Cannot update latest state: Processed DataFrame is empty for 1000000MOG/USDT:USDT.
2025-04-24 16:27:31 - ERROR    - [volbot_1000000MOG_USDT_USDT] - Failed to calculate indicators or update state for 1000000MOG/USDT:USDT. Skipping further analysis.
2025-04-24 16:27:31 - INFO     - [volbot_1000000PEIPEI_USDT_USDT] - --- Starting analysis cycle for 1000000PEIPEI/USDT:USDT ---
2025-04-24 16:27:32 - INFO     - [volbot_1000000PEIPEI_USDT_USDT] - Successfully fetched and processed 1000 klines for 1000000PEIPEI/USDT:USDT 3m
2025-04-24 16:27:32 - WARNING  - [volbot_1000000PEIPEI_USDT_USDT] - Insufficient data (1000 points) for 1000000PEIPEI/USDT:USDT to calculate all indicators reliably (recommend min: ~1050). Results may be inaccurate or contain NaNs.
2025-04-24 16:27:32 - INFO     - [volbot_1000000PEIPEI_USDT_USDT] - Calculating Volbot strategy indicators...
2025-04-24 16:27:32 - INFO     - [volbot_1000000PEIPEI_USDT_USDT] - Calculating Volumatic Trend Levels...
^C2025-04-24 16:27:38 - INFO     - [volbot_init] - KeyboardInterrupt received. Shutting down gracefully...
2025-04-24 16:27:38 - WARNING  - [volbot_init] - Trading was enabled. Consider manually checking open positions/orders.
2025-04-24 16:27:38 - INFO     - [volbot_init] - --- Volbot Shut Down ---
```

---

**2. Subsequent Shell Commands and Errors**

*   **Nano Edit:** The user attempts to edit `volbot3.py` using `nano`. The logged `Error in /data/data/com.termux/files/home/.nanorc... Bad regex` is an issue with the `nano` text editor's configuration file and is **unrelated** to the Volbot script's functionality.
*   **Syntax Error in `volbot5.py`:** An attempt to run `python volbot5.py` fails immediately.
    *   **Error:** `SyntaxError: '[' was never closed`
    *   **Location:** `/data/data/com.termux/files/home/worldguide/codes/volbot5.py`, line **4018**.
    *   **Cause:** A basic Python syntax error where a list literal `[` was opened but never closed with `]`. This is a separate bug in a different version of the script (`volbot5.py`).
*   **Initialization Failure in `volbot4.py`:** An attempt to run `python volbot4.py` fails during the initial setup phase (around `16:28:19 - 16:28:31`).
    *   **Initialization Steps Logged:** The script successfully loads configuration (timezone, interval, quote currency), detects **Live Trading Mode**, loads Bybit markets (2655 symbols).
    *   **Core Error:** Failure to fetch the initial USDT balance from Bybit.
    *   **API Error Detail:** `WARNING - Exchange/Network error fetching balance (type CONTRACT): bybit {"retCode":10001,"retMsg":"accountType only support UNIFIED."...}`. This specific error from the Bybit API indicates that the script attempted to fetch the balance using the `CONTRACT` account type, but Bybit's V5 API requires the `UNIFIED` account type for this operation.
    *   **Consequences:**
        *   `WARNING`: `_parse_balance: Failed to convert parsed balance string ''...` - The balance parsing function received empty data due to the API error.
        *   `ERROR`: "Could not determine available balance for USDT after all attempts."
        *   `ERROR`: "Initial balance fetch failed. This is critical."
        *   The script correctly identifies potential causes (API keys, permissions, IP whitelist, network issues, etc.).
        *   `CRITICAL`: "Exchange initialization failed... Exiting." - The bot cannot proceed without confirming the API connection and balance.

**(Original Log Snippet - Shell & `volbot4/5.py` Errors)**
```log
~/worldguide/codes main* ⇡ 1m 21s ❯ nano volbot3.py   Error in /data/data/com.termux/files/home/.nanorc on line 91: Bad regex "[+\-*/%<>=!&|^~@]=?": Invalid range end

~/worldguide/codes main* ⇡ 9s ❯ python volbot5.py
  File "/data/data/com.termux/files/home/worldguide/codes/volbot5.py", line 4018
    linear_swaps = [
                   ^
SyntaxError: '[' was never closed

~/worldguide/codes main* ⇡ ❯ python volbot4.py
16:28:19 - INFO     - [init] - --- Volbot Initializing ---
16:28:19 - INFO     - [init] - Timestamp: 2025-04-24 16:28:19 CDT
16:28:19 - INFO     - [init] - Config File: /data/data/com.termux/files/home/worldguide/codes/config.json
16:28:19 - INFO     - [init] - Log Directory: /data/data/com.termux/files/home/worldguide/codes/bot_logs
16:28:19 - INFO     - [init] - Quote Currency: USDT
16:28:19 - INFO     - [init] - Trading Enabled: True
16:28:19 - INFO     - [init] - Use Sandbox: False
16:28:19 - INFO     - [init] - Default Interval: 3 (3m)
16:28:19 - INFO     - [init] - Timezone: America/Chicago
16:28:19 - INFO     - [init] - Console Log Level: INFO
16:28:19 - WARNING  - [init] - Exchange bybit initialized. Status: LIVE TRADING MODE
16:28:19 - INFO     - [init] - Loading markets for bybit...
16:28:28 - INFO     - [init] - Markets loaded successfully (2655 symbols).
16:28:28 - INFO     - [init] - Attempting initial balance fetch (USDT) to test API keys and connection...
16:28:30 - WARNING  - [init] - Exchange/Network error fetching balance (type CONTRACT): bybit {"retCode":10001,"retMsg":"accountType only support UNIFIED.","result":{},"retExtInfo":{},"time":1745530111333}. Trying next.
16:28:30 - WARNING  - [init] - _parse_balance: Failed to convert parsed balance string '' from Bybit V5 list ['availableToWithdraw'] (Account: UNIFIED) to Decimal.
16:28:31 - WARNING  - [init] - _parse_balance: Failed to convert parsed balance string '' from Bybit V5 list ['availableToWithdraw'] (Account: UNIFIED) to Decimal.
16:28:31 - WARNING  - [init] - Default balance fetch did not find balance for USDT.
16:28:31 - ERROR    - [init] - Could not determine available balance for USDT after all attempts.
16:28:31 - ERROR    - [init] - Initial balance fetch failed. This is critical.
16:28:31 - ERROR    - [init] - Possible issues:
16:28:31 - ERROR    - [init] - - Invalid API Key/Secret.
16:28:31 - ERROR    - [init] - - Incorrect API permissions (Read required, Trade needed for execution).
16:28:31 - ERROR    - [init] - - IP Whitelist mismatch on Bybit account settings.
16:28:31 - ERROR    - [init] - - Using Live keys on Testnet or vice-versa.
16:28:31 - ERROR    - [init] - - Network/Firewall issues blocking connection to Bybit API.
16:28:31 - ERROR    - [init] - - Exchange API endpoint issues or maintenance.
16:28:31 - CRITICAL - [init] - Exchange initialization failed. Please check API keys, permissions, connection, and logs. Exiting.
```

---

**3. Partial Source Code (`volbot.py`)**

The provided text includes the beginning of a Python script, likely `volbot4.py` or a related version. This section details the initial setup:

*   **Purpose:** Incorporates a "Volumatic Trend + Order Block strategy" using the `ccxt` library for trading. Aims for robustness, error handling, and clarity.
*   **Key Libraries:** `ccxt`, `numpy`, `pandas`, `pandas_ta`, `requests`, `colorama`, `dotenv`, `logging`, `decimal`, `zoneinfo`.
*   **Setup:**
    *   Initializes `colorama` for colored terminal output and sets `Decimal` precision.
    *   Loads API keys (`BYBIT_API_KEY`, `BYBIT_API_SECRET`) from a `.env` file. Includes a critical check if keys are missing.
    *   Defines constants for colors, file paths (`config.json`, `bot_logs`), timezone (`America/Chicago`), API retries, and loop delays.
    *   Maps user-friendly intervals ("1", "3", "5") to `ccxt` timeframes ("1m", "3m", "5m").
    *   Defines default strategy parameters (lengths, sources, etc.) and risk parameters (ATR period, SL/TP multiples).
    *   Includes a robust logger setup (`SensitiveFormatter` to redact API keys, `setup_logger` for file rotation and colored console output).
    *   Implements configuration loading (`load_config`) from `config.json`, creating defaults if the file is missing, ensuring keys exist, and updating global settings like `QUOTE_CURRENCY`, `TIMEZONE`, and `console_log_level`. Includes basic validation for some config values.
    *   Includes `initialize_exchange` function to set up the `ccxt` Bybit instance with API keys, options (rate limiting, timeouts, linear preference), retry logic, sandbox mode handling, market loading, and a critical initial balance fetch test.
    *   Includes data fetching utilities: `safe_decimal` for robust Decimal conversion, `fetch_current_price_ccxt` with fallbacks, and `fetch_klines_ccxt` with retries, DataFrame conversion, and cleaning.
    *   Includes strategy calculation functions: `ema_swma`, `calculate_volatility_levels` (the function failing in `volbot2.py`), `calculate_pivot_order_blocks`, `manage_order_blocks`.
    *   Defines a `TradingAnalyzer` class to encapsulate indicator calculation, state management, market precision handling, signal generation (`generate_trading_signal`), and risk calculation (`calculate_entry_tp_sl`). It uses market info for precision.
    *   Includes trading logic helper functions: `fetch_balance` (handling various account types, specifically Bybit V5), `_parse_balance_from_response`, `get_market_info` (validating and enhancing market data), `calculate_position_size` (considering risk, SL, market limits, contract types), `get_open_position` (fetching and standardizing position data, including SL/TP/TSL), `place_market_order`, `set_position_protection` (handling SL/TP/TSL updates, specifically for Bybit V5).
    *   Defines the main analysis/trading loop function `analyze_and_trade_symbol`.
    *   Includes the `main` execution block for initialization, symbol selection (interactive prompt), and iterating through symbols calling `analyze_and_trade_symbol`.

**(Original Log Snippet - Partial Source Code)**
```python
# volbot.py
# Incorporates Volumatic Trend + Order Block strategy into a trading framework using ccxt.
# Enhanced version with improved structure, error handling, logging, clarity, and robustness.

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo  # Use zoneinfo (Python 3.9+) for timezone handling

# Initialize colorama and set Decimal precision
init(autoreset=True)
getcontext().prec = 28  # Increased precision for complex financial calculations
load_dotenv()

# --- Constants ---
# Color Scheme
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# Strategy-Specific Colors & Log Levels
COLOR_UP = Fore.CYAN + Style.BRIGHT
COLOR_DN = Fore.YELLOW + Style.BRIGHT
COLOR_BULL_BOX = Fore.GREEN
COLOR_BEAR_BOX = Fore.RED
COLOR_CLOSED_BOX = Fore.LIGHTBLACK_EX
COLOR_INFO = Fore.MAGENTA
COLOR_HEADER = Fore.BLUE + Style.BRIGHT
COLOR_WARNING = NEON_YELLOW
COLOR_ERROR = NEON_RED
COLOR_SUCCESS = NEON_GREEN

# API Credentials (Loaded from .env)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use print here as logger might not be set up yet
    print(f"{COLOR_ERROR}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}")
    raise ValueError("API Key/Secret not found in environment variables.")

# File/Directory Configuration
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Time & Retry Configuration
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default timezone string
try:
    # Initialize TIMEZONE with the default; will be updated by load_config if specified
    TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_STR)
except Exception as tz_err:
    print(f"{COLOR_ERROR}CRITICAL: Default timezone '{DEFAULT_TIMEZONE_STR}' invalid or system tzdata missing: {tz_err}. Exiting.{RESET}")
    exit(1) # Cannot proceed without a valid timezone

MAX_API_RETRIES = 3  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Base delay between retries
LOOP_DELAY_SECONDS = 15  # Min time between the end of one cycle and the start of the next

# Interval Configuration
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# API Error Codes for Retry Logic (HTTP status codes)
# Expand with specific exchange error codes if needed (beyond HTTP)
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]

# --- Default Volbot Strategy Parameters (overridden by config.json) ---
DEFAULT_VOLBOT_LENGTH = 40
DEFAULT_VOLBOT_ATR_LENGTH = 200
DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK = 1000
DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE = 100 # Use max volume in lookback
DEFAULT_VOLBOT_OB_SOURCE = "Wicks" # "Wicks" or "Bodys"
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L = 25
DEFAULT_VOLBOT_MAX_BOXES = 50

# Default Risk Management Parameters (overridden by config.json)
DEFAULT_ATR_PERIOD = 14 # Risk Management ATR (for SL/TP/BE)

# Global QUOTE_CURRENCY placeholder, dynamically loaded from config
QUOTE_CURRENCY = "USDT" # Default fallback, updated by load_config

# Default console log level (updated by config)
console_log_level = logging.INFO

# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that redacts sensitive information like API keys/secrets from log messages."""
    REDACTION_STR = "***REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive info."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, self.REDACTION_STR)
        if API_SECRET:
            msg = msg.replace(API_SECRET, self.REDACTION_STR)
        # Add other sensitive patterns if needed (e.g., passwords, specific tokens)
        # msg = re.sub(r'password=\S+', 'password=***REDACTED***', msg)
        return msg

def setup_logger(name_suffix: str) -> logging.Logger:
    """
    Sets up a logger instance with specified suffix, file rotation, and colored console output.
    Prevents adding duplicate handlers and updates console level based on global setting.

    Args:
        name_suffix: A string suffix for the logger name and filename (e.g., symbol or 'init').

    Returns:
        The configured logging.Logger instance.
    """
    global console_log_level # Ensure we use the potentially updated level
    safe_suffix = re.sub(r'[^\w\-]+', '_', name_suffix) # Make suffix filesystem-safe
    logger_name = f"volbot_{safe_suffix}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Check if handlers already exist to prevent duplicates
    if logger.hasHandlers():
        # Update existing console handler level if necessary
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.level != console_log_level:
                    logger.debug(f"Updating console handler level for {logger_name} to {logging.getLevelName(console_log_level)}")
                    handler.setLevel(console_log_level)
        return logger # Logger already configured

    logger.setLevel(logging.DEBUG) # Set root level to DEBUG to allow handlers to filter

    # File Handler (Always DEBUG level for detailed logs)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        # Use print as logger might not be fully functional yet
        print(f"{COLOR_ERROR}Error setting up file logger for {log_filename}: {e}{RESET}")

    # Console Handler (Uses global console_log_level)
    stream_handler = logging.StreamHandler()
    # Define colors for different log levels
    level_colors = {
        logging.DEBUG: NEON_BLUE,
        logging.INFO: NEON_GREEN,
        logging.WARNING: NEON_YELLOW,
        logging.ERROR: NEON_RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    class ColorFormatter(SensitiveFormatter):
        """Custom formatter to add colors and specific formatting to console logs."""
        def format(self, record):
            log_color = level_colors.get(record.levelno, RESET)
            # Format level name with color and padding
            record.levelname = f"{log_color}{record.levelname:<8}{RESET}"
            # Format timestamp with color
            record.asctime = f"{NEON_BLUE}{self.formatTime(record, self.datefmt)}{RESET}"
            # Extract the base logger name part (e.g., 'volbot_BTCUSDT' -> 'BTCUSDT')
            base_name = record.name.split('_', 1)[-1] if '_' in record.name else record.name
            record.name_part = f"{NEON_PURPLE}[{base_name}]{RESET}"
            # Format the final message using the parent's method after modifications
            # Add color reset at the end of the message itself for safety
            record.msg = f"{log_color if record.levelno >= logging.WARNING else ''}{record.getMessage()}{RESET}"
            # Use a modified format string for the final output structure
            formatted_message = super(SensitiveFormatter, self).format(record)
            # Ensure final message has color reset if something went wrong
            return f"{formatted_message}{RESET}"

    # Modified format string for ColorFormatter instance
    stream_formatter = ColorFormatter(
        # Format: Timestamp - LEVEL    - [Name] - Message
        "%(asctime)s - %(levelname)s - %(name_part)s - %(message)s",
        datefmt='%H:%M:%S' # Use shorter time format for console
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_log_level) # Use the global level
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent messages from reaching the root logger
    return logger

# --- Configuration Loading ---
def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file. Creates a default config if the file
    doesn't exist. Ensures all default keys are present, adding missing ones
    with default values and updating the file if necessary. Updates relevant global settings.

    Args:
        filepath: The path to the configuration JSON file.

    Returns:
        A dictionary containing the configuration settings.
    """
    global QUOTE_CURRENCY, TIMEZONE, console_log_level # Allow updating global variables

    default_config = {
        # --- General Bot Settings ---
        "timezone": DEFAULT_TIMEZONE_STR, # Timezone for logging and potentially scheduling (e.g., "Europe/London")
        "interval": "5",              # Default trading interval (string format from VALID_INTERVALS)
        "retry_delay": RETRY_DELAY_SECONDS, # API retry delay in seconds
        "enable_trading": False,      # MASTER SWITCH: Set to true to allow placing real orders. Default: False.
        "use_sandbox": True,          # Use exchange's testnet/sandbox environment. Default: True.
        "risk_per_trade": 0.01,       # Max percentage of account balance to risk per trade (0.01 = 1%)
        "leverage": 10,               # Desired leverage for contract trading (applied if possible)
        "max_concurrent_positions": 1,# Max open positions allowed per symbol by this bot instance (currently informational, logic not fully implemented)
        "quote_currency": "USDT",     # Currency for balance checks and position sizing (MUST match exchange pairs, e.g., USDT for BTC/USDT)
        "console_log_level": "INFO",  # Console logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        # --- Volbot Strategy Settings ---
        "volbot_enabled": True,         # Enable/disable Volbot strategy calculations and signals
        "volbot_length": DEFAULT_VOLBOT_LENGTH, # Main period for Volbot EMAs
        "volbot_atr_length": DEFAULT_VOLBOT_ATR_LENGTH, # ATR period for Volbot dynamic levels
        "volbot_volume_percentile_lookback": DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK, # Lookback for volume normalization
        "volbot_volume_normalization_percentile": DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE, # Percentile (usually 100=max)
        "volbot_ob_source": DEFAULT_VOLBOT_OB_SOURCE, # "Wicks" or "Bodys" for Order Block detection
        "volbot_pivot_left_len_h": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H, # Left bars for Pivot High
        "volbot_pivot_right_len_h": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H,# Right bars for Pivot High
        "volbot_pivot_left_len_l": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L, # Left bars for Pivot Low
        "volbot_pivot_right_len_l": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L,# Right bars for Pivot Low
        "volbot_max_boxes": DEFAULT_VOLBOT_MAX_BOXES, # Max number of active Order Blocks to track
        "volbot_signal_on_trend_flip": True, # Generate BUY/SELL on Volbot trend direction change
        "volbot_signal_on_ob_entry": True,   # Generate BUY/SELL on price entering an Order Block matching trend

        # --- Risk Management Settings ---
        "atr_period": DEFAULT_ATR_PERIOD, # ATR period for SL/TP/BE calculations (Risk Management ATR)
        "stop_loss_multiple": 1.8, # Risk ATR multiple for initial Stop Loss distance
        "take_profit_multiple": 0.7, # Risk ATR multiple for initial Take Profit distance

        # --- Trailing Stop Loss Config (Exchange-based TSL) ---
        "enable_trailing_stop": True, # Attempt to set an exchange-based Trailing Stop Loss on entry
        "trailing_stop_callback_rate": "0.005", # Trail distance as percentage (0.005=0.5%) or price distance (e.g., "50" for $50). String for Bybit API. Check exchange API docs.
        "trailing_stop_activation_percentage": 0.003, # Profit percentage to activate TSL (0.003=0.3%). 0 for immediate activation if supported.

        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even + offset
        "break_even_trigger_atr_multiple": 1.0, # Profit needed (in multiples of Risk ATR) to trigger BE
        "break_even_offset_ticks": 2,           # Number of minimum price ticks to offset BE SL from entry (for fees/slippage)
    }

    config_updated = False
    loaded_config = {}

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            print(f"{COLOR_WARNING}Created default config file: {filepath}{RESET}")
            loaded_config = default_config.copy() # Use a copy
            config_updated = True # File was created
        except IOError as e:
            print(f"{COLOR_ERROR}Error creating default config file {filepath}: {e}. Using built-in default values.{RESET}")
            loaded_config = default_config.copy() # Use defaults if creation fails
    else:
        try:
            with open(filepath, 'r', encoding="utf-8") as f:
                loaded_config_from_file = json.load(f)
            # Ensure all default keys exist, add missing ones recursively
            loaded_config, config_updated = _ensure_config_keys(loaded_config_from_file, default_config)
            if config_updated:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(loaded_config, f_write, indent=4, sort_keys=True)
                    print(f"{COLOR_WARNING}Updated config file '{filepath}' with missing default keys.{RESET}")
                except IOError as e:
                    print(f"{COLOR_ERROR}Error writing updated config file {filepath}: {e}{RESET}")
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            print(f"{COLOR_ERROR}Error loading or parsing config file {filepath}: {e}. Using default config values and attempting to recreate the file.{RESET}")
            # Attempt to recreate default config if loading failed badly
            loaded_config = default_config.copy() # Start with defaults
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4, sort_keys=True)
                print(f"{COLOR_WARNING}Recreated default config file: {filepath}{RESET}")
                config_updated = True
            except IOError as e_create:
                print(f"{COLOR_ERROR}Error creating default config file after load error: {e_create}{RESET}")

    # --- Update global settings based on loaded/default config ---
    # Quote Currency
    new_quote_currency = loaded_config.get("quote_currency", default_config["quote_currency"]).upper()
    if new_quote_currency != QUOTE_CURRENCY:
        print(f"{COLOR_INFO}Setting QUOTE_CURRENCY to: {new_quote_currency}{RESET}")
        QUOTE_CURRENCY = new_quote_currency

    # Console Log Level
    level_name = loaded_config.get("console_log_level", "INFO").upper()
    new_log_level = getattr(logging, level_name, logging.INFO)
    if new_log_level != console_log_level:
        print(f"{COLOR_INFO}Setting console log level to: {level_name}{RESET}")
        console_log_level = new_log_level
        # Note: Existing loggers' console handlers will be updated by setup_logger() if called again.

    # Timezone
    config_tz_str = loaded_config.get("timezone", DEFAULT_TIMEZONE_STR)
    try:
        new_tz = ZoneInfo(config_tz_str)
        # Check if timezone actually changed to avoid unnecessary logging
        if TIMEZONE is None or new_tz.key != TIMEZONE.key:
            print(f"{COLOR_INFO}Setting timezone to: {config_tz_str}{RESET}")
            TIMEZONE = new_tz
    except Exception as tz_err:
        # Use the existing default TIMEZONE if the config value is invalid
        print(f"{COLOR_ERROR}Invalid timezone '{config_tz_str}' in config: {tz_err}. Using previously set default '{TIMEZONE.key}'.{RESET}")
        # Ensure the invalid config value doesn't persist in the returned dict
        loaded_config["timezone"] = TIMEZONE.key

    # --- Validate specific config values ---
    # Validate interval
    if loaded_config.get("interval") not in VALID_INTERVALS:
         print(f"{COLOR_ERROR}Invalid 'interval' in config: '{loaded_config.get('interval')}'. Using default '5'. Valid: {VALID_INTERVALS}{RESET}")
         loaded_config["interval"] = default_config["interval"] # Correct in loaded config

    # Validate OB source
    ob_source = loaded_config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    if ob_source not in ["Wicks", "Bodys"]:
         print(f"{COLOR_ERROR}Invalid 'volbot_ob_source': '{ob_source}'. Using default '{DEFAULT_VOLBOT_OB_SOURCE}'. Valid: ['Wicks', 'Bodys']{RESET}")
         loaded_config["volbot_ob_source"] = default_config["volbot_ob_source"]

    # Validate numeric values (example: risk_per_trade)
    try:
        risk_val = float(loaded_config.get("risk_per_trade", default_config["risk_per_trade"]))
        if not (0 < risk_val < 1):
            raise ValueError("Risk must be between 0 and 1 (exclusive)")
        loaded_config["risk_per_trade"] = risk_val # Store as float
    except (ValueError, TypeError) as e:
         print(f"{COLOR_ERROR}Invalid 'risk_per_trade' value: '{loaded_config.get('risk_per_trade')}'. Error: {e}. Using default {default_config['risk_per_trade']}.{RESET}")
         loaded_config["risk_per_trade"] = default_config["risk_per_trade"]

    # Add validation for other critical numeric or specific string values as needed

    return loaded_config

def _ensure_config_keys(loaded_config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures default keys exist in loaded config. Returns updated config and a flag indicating if changes were made.
    Modifies loaded_config in place.
    """
    updated = False
    for key, default_value in default_config.items():
        if key not in loaded_config:
            loaded_config[key] = default_value
            updated = True
            print(f"{COLOR_INFO}Config: Added missing key '{key}' with default value: {default_value}{RESET}")
        elif isinstance(default_value, dict) and isinstance(loaded_config.get(key), dict):
            # Recurse for nested dictionaries
            nested_updated = _ensure_config_keys(loaded_config[key], default_value)[1] # Only need the boolean flag
            if nested_updated:
                updated = True
        # Optional: Add type checking/validation here if desired, though validation after loading is often clearer
        # elif type(loaded_config.get(key)) != type(default_value):
        #     print(f"{COLOR_WARNING}Config: Type mismatch for key '{key}'. Expected {type(default_value)}, got {type(loaded_config.get(key))}. Using loaded value.{RESET}")
    return loaded_config, updated

# Load configuration globally AFTER functions are defined and initial logging level is set
CONFIG = load_config(CONFIG_FILE)

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object with configuration settings,
    loads markets, and performs critical connection/authentication tests.

    Args:
        logger: The logger instance for initialization steps.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails critically.
    """
    lg = logger
    try:
        exchange_id = 'bybit' # Hardcoded to Bybit for this script
        exchange_class = getattr(ccxt, exchange_id)

        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable ccxt's built-in rate limiter
            'options': {
                'defaultType': 'linear', # Prefer linear contracts (USDT margined) unless overridden
                'adjustForTimeDifference': True, # Auto-sync time with server clock
                # Increased timeouts (milliseconds) for potentially slow networks/API
                'recvWindow': 10000, # Bybit recommended recv_window for API stability
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 15000,
                'fetchOHLCVTimeout': 20000,
                'fetchPositionTimeout': 15000, # Timeout for fetching single position
                'fetchPositionsTimeout': 20000, # Timeout for fetching multiple positions
            },
            # Add custom HTTP adapter for more robust retry logic beyond rate limits
            # Note: ccxt's enableRateLimit handles 429s, this handles network/server issues
            'requests_session': requests.Session() # Provide a session for custom adapters
        }

        # Configure retry strategy for the requests session
        retries = Retry(
            total=MAX_API_RETRIES,
            backoff_factor=0.5, # Shorter backoff factor (e.g., 0.5 -> 0.5s, 1s, 2s)
            status_forcelist=RETRY_ERROR_CODES, # Retry on these HTTP status codes
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"], # Retry on all common methods
        )
        adapter = HTTPAdapter(max_retries=retries)
        exchange_options['requests_session'].mount('https://', adapter)
        exchange_options['requests_session'].mount('http://', adapter)

        exchange = exchange_class(exchange_options)

        # Set Sandbox Mode based on config
        use_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(use_sandbox)
        sandbox_status = f"{COLOR_WARNING}SANDBOX MODE{RESET}" if use_sandbox else f"{COLOR_ERROR}LIVE TRADING MODE{RESET}"
        lg.warning(f"Exchange {exchange.id} (Version: {exchange.version}) initialized. Status: {sandbox_status}")

        # --- Load Markets - Crucial Step ---
        lg.info(f"Loading markets for {exchange.id}...")
        try:
            exchange.load_markets(reload=True) # Force reload on init
            lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols).")
        except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as e:
            lg.critical(f"{COLOR_ERROR}CRITICAL: Failed to load markets: {e}. Check connection, API status, and firewall. Cannot proceed.{RESET}")
            return None # Critical failure if markets can't be loaded

        # --- Test API Connection & Authentication with Balance Fetch ---
        lg.info(f"Attempting initial balance fetch ({QUOTE_CURRENCY}) to test API keys and connection...")
        # Use the robust fetch_balance which handles different account types
        test_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)

        if test_balance is not None:
             # Balance fetch successful, log the balance (can be 0)
             lg.info(f"{COLOR_SUCCESS}API keys and connection successful. Initial available {QUOTE_CURRENCY} balance: {test_balance:.4f}{RESET}")
        else:
            # fetch_balance logs detailed errors, add a summary here for critical failure
            lg.critical(f"{COLOR_ERROR}CRITICAL: Initial balance fetch failed. This is necessary to confirm API keys and permissions.{RESET}")
            lg.error(f"{COLOR_ERROR}Possible issues:{RESET}")
            lg.error(f"{COLOR_ERROR}- Invalid API Key/Secret provided in .env file.{RESET}")
            lg.error(f"{COLOR_ERROR}- Incorrect API permissions set on Bybit (Read required, Trade needed for execution).{RESET}")
            lg.error(f"{COLOR_ERROR}- IP Whitelist mismatch on Bybit account API key settings.{RESET}")
            lg.error(f"{COLOR_ERROR}- Using Live API keys on Testnet environment or vice-versa.{RESET}")
            lg.error(f"{COLOR_ERROR}- Network/Firewall issues blocking connection to Bybit API endpoints.{RESET}")
            lg.error(f"{COLOR_ERROR}- Bybit API endpoint issues, maintenance, or region restrictions.{RESET}")
            lg.error(f"{COLOR_ERROR}- Clock skew between your server and Bybit (check 'adjustForTimeDifference' setting).{RESET}")
            return None # Critical failure if cannot authenticate/connect

        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{COLOR_ERROR}CCXT Authentication Error during initialization: {e}. Check API Key/Secret and permissions.{RESET}")
    except ccxt.ExchangeError as e:
        lg.critical(f"{COLOR_ERROR}CCXT Exchange Error initializing {exchange_id}: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{COLOR_ERROR}CCXT Network Error initializing {exchange_id}: {e}. Check connection/firewall.{RESET}")
    except Exception as e:
        lg.critical(f"{COLOR_ERROR}Unexpected error initializing CCXT exchange {exchange_id}: {e}{RESET}", exc_info=True)
    return None

# --- CCXT Data Fetching ---
def safe_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """
    Safely convert a value to Decimal, handling None, strings, floats, ints,
    and potential InvalidOperation or non-finite values (NaN, Inf).

    Args:
        value: The value to convert.
        default: The value to return if conversion fails or input is invalid.

    Returns:
        The Decimal representation, or the default value.
    """
    if value is None:
        return default
    try:
        # Convert to string first to handle floats accurately and avoid precision issues
        str_value = str(value).strip()
        # Handle empty strings explicitly
        if not str_value:
            return default
        d = Decimal(str_value)
        # Check for NaN or Infinity which are invalid for most financial ops
        if not d.is_finite():
            # Optionally log this occurrence if needed for debugging
            # logging.getLogger(__name__).debug(f"safe_decimal encountered non-finite value: {value}")
            return default
        return d
    except (InvalidOperation, ValueError, TypeError):
        # Optionally log this occurrence if needed for debugging
        # logging.getLogger(__name__).debug(f"safe_decimal failed for value: {value} (type: {type(value)})")
        return default

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using the exchange's ticker,
    with robust fallbacks (last, mid, ask, bid), validation, and Decimal conversion.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        The current price as a Decimal, or None if fetching fails or price is invalid/zero.
    """
    lg = logger
    price: Optional[Decimal] = None
    attempt = 0
    max_attempts = MAX_API_RETRIES + 1

    while attempt < max_attempts:
        attempt += 1
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempt}/{max_attempts})...")
            # Prepare parameters, especially for Bybit V5
            params = {}
            if 'bybit' in exchange.id.lower(): # Add category for Bybit V5
                try:
                    market = exchange.market(symbol) # Assume market loaded
                    if market:
                        # Determine category based on market type (linear preferred)
                        category = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                        params['category'] = category
                        lg.debug(f"Using category='{category}' for {symbol} ticker fetch.")
                    else:
                        lg.warning(f"Market info not found for {symbol} when fetching ticker. Assuming 'linear'.")
                        params['category'] = 'linear'
                except Exception as market_err:
                    lg.warning(f"Error getting market info for ticker params ({symbol}): {market_err}. Assuming 'linear'.")
                    params['category'] = 'linear'

            ticker = exchange.fetch_ticker(symbol, params=params)
            lg.debug(f"Ticker data received for {symbol}: Keys={list(ticker.keys())}")

            # Order of preference for price, using safe_decimal for robustness:
            # 1. 'last' price (most recent trade price)
            last_price = safe_decimal(ticker.get('last'))
            if last_price is not None and last_price > 0:
                price = last_price
                lg.debug(f"Using 'last' price: {price}")
                break # Found valid price

            # 2. Bid/Ask Midpoint (fair estimate if liquid)
            bid = safe_decimal(ticker.get('bid'))
            ask = safe_decimal(ticker.get('ask'))
            if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
                mid_price = (bid + ask) / Decimal('2')
                price = mid_price
                lg.debug(f"Using bid/ask midpoint: {price} (Bid: {bid}, Ask: {ask})")
                break # Found valid price

            # 3. 'ask' price (lowest price seller is willing to accept) - Use if buying
            if ask is not None and ask > 0:
                price = ask
                lg.debug(f"Using 'ask' price (fallback): {price}")
                break # Found valid price

            # 4. 'bid' price (highest price buyer is willing to pay) - Use if selling (last resort)
            if bid is not None and bid > 0:
                price = bid
                lg.debug(f"Using 'bid' price (last resort): {price}")
                break # Found valid price

            # If none of the above yielded a positive price
            lg.warning(f"{COLOR_WARNING}Ticker for {symbol} received, but no valid positive price found (last/mid/ask/bid). Attempt {attempt}/{max_attempts}. Ticker data: {ticker}{RESET}")
            if attempt < max_attempts: time.sleep(RETRY_DELAY_SECONDS)
            # Continue loop to retry

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"Network error fetching price for {symbol} (Attempt {attempt}/{max_attempts}): {e}. Retrying...")
            if attempt < max_attempts: time.sleep(RETRY_DELAY_SECONDS)
            else: lg.error(f"{COLOR_ERROR}Max retries exceeded for network error fetching price: {e}{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * (attempt + 1) # Simple exponential backoff
            lg.warning(f"Rate limit exceeded fetching price for {symbol} (Attempt {attempt}/{max_attempts}). Retrying in {wait_time}s: {e}")
            if attempt < max_attempts: time.sleep(wait_time)
            else: lg.error(f"{COLOR_ERROR}Max retries exceeded after rate limit fetching price: {e}{RESET}")
        except ccxt.ExchangeNotAvailable as e:
            lg.error(f"{COLOR_ERROR}Exchange not available fetching price for {symbol}: {e}{RESET}")
            # Might retry if temporary, but often indicates a larger issue
            if attempt < max_attempts: time.sleep(RETRY_DELAY_SECONDS * 2) # Longer delay
            else: lg.error(f"{COLOR_ERROR}Max retries exceeded, exchange unavailable fetching price: {e}{RESET}")
        except ccxt.BadSymbol as e:
             lg.error(f"{COLOR_ERROR}Invalid symbol {symbol} for fetching ticker: {e}{RESET}")
             return None # Cannot recover from bad symbol
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            if "symbol not found" in err_str or "instrument not found" in err_str or "invalid symbol" in err_str:
                 lg.error(f"{COLOR_ERROR}Symbol {symbol} not found on exchange ticker endpoint: {e}{RESET}")
                 return None # Cannot recover
            lg.warning(f"Exchange error fetching price for {symbol} (Attempt {attempt}/{max_attempts}): {e}. Retrying...")
            if attempt < max_attempts: time.sleep(RETRY_DELAY_SECONDS)
            else: lg.error(f"{COLOR_ERROR}Max retries exceeded for exchange error fetching price: {e}{RESET}")
        except Exception as e:
            lg.error(f"{COLOR_ERROR}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Unexpected error, stop

    # Final validation after loop
    if price is not None and price > 0:
        return price
    else:
        lg.error(f"{COLOR_ERROR}Failed to fetch a valid positive current price for {symbol} after {max_attempts} attempts.{RESET}")
        return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Fetches OHLCV kline data using CCXT with retries, validation, DataFrame conversion,
    robust data cleaning, and Decimal conversion.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        timeframe: CCXT timeframe string (e.g., '5m', '1h').
        limit: Maximum number of klines to fetch.
        logger: Logger instance.

    Returns:
        A pandas DataFrame with OHLCV data indexed by timestamp, or an empty DataFrame on failure.
        Columns 'open', 'high', 'low', 'close', 'volume' are converted to Decimal where possible.
    """
    lg = logger or logging.getLogger(__name__) # Use provided logger or get default
    empty_df = pd.DataFrame()
    try:
        if not exchange.has.get('fetchOHLCV'):
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
            return empty_df

        ohlcv: Optional[List[List[Any]]] = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
                # Add category param for Bybit V5 klines
                params = {}
                if 'bybit' in exchange.id.lower(): # Add category for Bybit V5
                    try:
                        market = exchange.market(symbol) # Assume market loaded
                        if market:
                            category = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                            params['category'] = category
                            lg.debug(f"Using category='{category}' for {symbol} kline fetch.")
                        else:
                            lg.warning(f"Market info not found for {symbol} when fetching klines. Assuming 'linear'.")
                            params['category'] = 'linear'
                    except Exception as market_err:
                        lg.warning(f"Error getting market info for kline params ({symbol}): {market_err}. Assuming 'linear'.")
                        params['category'] = 'linear'

                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)

                if ohlcv and len(ohlcv) > 0:
                    lg.debug(f"Received {len(ohlcv)} klines from API for {symbol} {timeframe}.")
                    break # Success
                else:
                    lg.warning(f"fetch_ohlcv returned empty list for {symbol} {timeframe} (Attempt {attempt+1}). Retrying...")
                    if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error fetching klines for {symbol} {timeframe} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.error(f"{COLOR_ERROR}Max retries exceeded for network error fetching klines: {e}{RESET}")
                    raise e # Raise after max retries
            except ccxt.RateLimitExceeded as e:
                # Extract wait time suggestion from error message if possible
                wait_time_match = re.search(r'try again in (\d+)', str(e), re.IGNORECASE)
                wait_time = int(wait_time_match.group(1)) if wait_time_match else RETRY_DELAY_SECONDS * (attempt + 2) # Exponential backoff
                lg.warning(f"Rate limit exceeded fetching klines for {symbol} {timeframe}. Retrying in {wait_time}s... (Attempt {attempt+1})")
                if attempt < MAX_API_RETRIES: time.sleep(wait_time)
                else: raise e
            except ccxt.BadSymbol as e:
                 lg.error(f"{COLOR_ERROR}Invalid symbol {symbol} for fetching klines: {e}{RESET}")
                 return empty_df # Cannot recover from bad symbol
            except ccxt.ExchangeError as e:
                 err_str = str(e).lower()
                 if "symbol not found" in err_str or "instrument invalid" in err_str or "invalid symbol" in err_str:
                      lg.error(f"{COLOR_ERROR}Symbol {symbol} not found on exchange kline endpoint: {e}{RESET}")
                      return empty_df # Cannot recover
                 lg.warning(f"Exchange error during fetch_ohlcv for {symbol} {timeframe} (Attempt {attempt+1}): {e}. Retrying...")
                 # Retry generic exchange errors cautiously
                 if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                 else: raise e
            except Exception as e:
                lg.error(f"{COLOR_ERROR}Unexpected error during fetch_ohlcv ({symbol} {timeframe}, Attempt {attempt+1}): {e}{RESET}", exc_info=True)
                # Decide whether to retry unexpected errors
                if attempt < MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)
                else: raise e # Raise after max retries

        if not ohlcv:
            lg.warning(f"{COLOR_WARNING}No kline data returned for {symbol} {timeframe} after all retries.{RESET}")
            return empty_df

        # --- Convert to DataFrame ---
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            lg.warning(f"DataFrame conversion resulted in empty DF for {symbol}.")
            return empty_df

        # --- Data Cleaning and Processing ---
        # 1. Convert timestamp to datetime index (UTC)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
            df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
            df.set_index('timestamp', inplace=True)
        except Exception as dt_err:
             lg.error(f"Error converting timestamp to datetime index for {symbol}: {dt_err}")
             return empty_df # Cannot proceed without valid index

        # 2. Convert OHLCV columns to Decimal for precision, handling errors
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            # Apply safe_decimal first for robust conversion, keep NaN on failure
            df[col] = df[col].apply(lambda x: safe_decimal(x, default=np.nan))
            # Ensure the column remains numeric (object type can cause issues with calculations)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        # 3. Drop rows with any NaN in essential OHLC price data
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # 4. Ensure close price is positive (filter out bad data points like zero prices)
        df = df[df['close'] > 0]
        # 5. Fill NaN volume with 0 (often indicates no trades in that period, safe assumption)
        df['volume'].fillna(0, inplace=True)
        # Re-convert volume to Decimal after fillna
        df['volume'] = df['volume'].apply(lambda x: safe_decimal(x, default=Decimal(0)))


        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price or non-positive close for {symbol}.")

        if df.empty:
            lg.warning(f"{COLOR_WARNING}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
            return empty_df

        # 6. Ensure data is sorted chronologically by timestamp index
        df.sort_index(inplace=True)

        # 7. Check for and handle duplicate timestamps (can happen with API glitches)
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].unique()
            lg.warning(f"{COLOR_WARNING}Found {len(duplicates)} duplicate timestamps in kline data for {symbol}. Keeping last entry for each.{RESET}")
            df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.BadSymbol as e:
         lg.error(f"{COLOR_ERROR}Invalid symbol {symbol} used in fetch_klines_ccxt: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error fetching/processing klines for {symbol} after retries: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{COLOR_ERROR}Exchange error processing klines for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return empty_df

# --- Volbot Strategy Calculation Functions ---

def ema_swma(series: pd.Series, length: int, logger: logging.Logger) -> pd.Series:
    """
    Calculates a Smoothed Weighted Moving Average (SWMA), an EMA applied
    to a weighted average of the last 4 values (weights: 1/6, 2/6, 2/6, 1/6).
    Uses pandas_ta.ema with adjust=False for TV-like calculation. Handles NaNs.

    Args:
        series: Input pandas Series (e.g., 'close' prices). Should contain numeric types (float or Decimal).
        length: EMA length parameter.
        logger: Logger instance.

    Returns:
        A pandas Series with SWMA values (as float64), aligned with the input series index.
    """
    lg = logger
    lg.debug(f"Calculating SWMA with length: {length}...")
    required_periods = 4 # Need 4 periods for the weighting

    # Ensure input is numeric (convert Decimals to float for calculation if needed)
    numeric_series = pd.to_numeric(series, errors='coerce')

    if len(numeric_series) < required_periods:
        lg.warning(f"Series length ({len(numeric_series)}) < {required_periods}. SWMA requires {required_periods} periods. Returning standard EMA.")
        # Fallback to standard EMA, use adjust=False for consistency
        ema_result = ta.ema(numeric_series, length=length, adjust=False)
        # Ensure result is a Series even if input is short
        return ema_result if isinstance(ema_result, pd.Series) else pd.Series(ema_result, index=series.index)

    # Calculate the weighted average: (1/6)*P[t] + (2/6)*P[t-1] + (2/6)*P[t-2] + (1/6)*P[t-3]
    # Use fillna(0) temporarily for calculation, handle NaNs properly afterwards
    w0 = numeric_series.fillna(0) / 6
    w1 = numeric_series.shift(1).fillna(0) * 2 / 6
    w2 = numeric_series.shift(2).fillna(0) * 2 / 6
    w3 = numeric_series.shift(3).fillna(0) * 1 / 6 # Corrected weight for P[t-3]

    weighted_series = w0 + w1 + w2 + w3

    # Set initial values (where shifts caused NaNs) back to NaN before EMA
    # Also set back NaN where original series had NaN
    weighted_series[numeric_series.isna()] = np.nan
    weighted_series.iloc[:required_periods-1] = np.nan

    # Calculate EMA on the weighted series
    # Use adjust=False for behavior closer to TradingView's EMA calculation
    # Apply EMA only on valid weighted values to avoid propagating initial NaNs incorrectly
    smoothed_ema = ta.ema(weighted_series.dropna(), length=length, adjust=False)

    # Reindex to match the original series index. This correctly places NaNs.
    result_series = smoothed_ema.reindex(series.index)

    lg.debug(f"SWMA calculation finished. Result length: {len(result_series)}, NaNs: {result_series.isna().sum()}")
    # Result will be float64 due to pandas_ta operations
    return result_series


def calculate_volatility_levels(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Calculates core Volumatic Trend indicators: EMAs, ATR, dynamic levels,
    normalized volume, and cumulative volume metrics. Includes data length checks
    and uses Decimal for price/level calculations where feasible before final conversion.

    Args:
        df: DataFrame with OHLCV data (prices/volume assumed numeric or Decimal).
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        DataFrame with added Volbot strategy columns. Returns input df if insufficient data.
        Calculated columns are typically float64 due to TA library usage.
    """
    lg = logger
    lg.info("Calculating Volumatic Trend Levels...")
    length = config.get("volbot_length", DEFAULT_VOLBOT_LENGTH)
    atr_length = config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH)
    volume_lookback = config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK)
    # vol_norm_perc = config.get("volbot_volume_normalization_percentile", DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE) # Not directly used here

    # Check for sufficient data length for calculations
    min_len = max(length + 3, atr_length, volume_lookback) + 10 # Add buffer
    if len(df) < min_len:
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)} rows) for Volumatic Trend calculation (min ~{min_len}). Skipping.{RESET}")
        # Add placeholder columns to prevent errors later
        placeholder_cols = [
            'ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat',
            'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat',
            'step_up_strat', 'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat',
            'vol_dn_step_strat', 'vol_trend_up_level_strat', 'vol_trend_dn_level_strat',
            'volume_delta_strat', 'volume_total_strat', 'cum_vol_delta_since_change_strat',
            'cum_vol_total_since_change_strat', 'last_trend_change_idx'
        ]
        for col in placeholder_cols:
            if col not in df.columns: df[col] = np.nan
        return df

    df_calc = df.copy() # Work on a copy

    try:
        # Ensure input columns are numeric for TA functions
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
             df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
        df_calc['volume_float'] = pd.to_numeric(df_calc['volume'], errors='coerce').fillna(0) # Use float volume for TA

        # --- Calculate Strategy EMAs and ATR (using float versions for TA Lib) ---
        df_calc['ema1_strat'] = ema_swma(df_calc['close'], length, lg) # SWMA uses float input/output
        df_calc['ema2_strat'] = ta.ema(df_calc['close'], length=length, adjust=False)
        df_calc['atr_strat'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=atr_length)

        # Drop rows where essential indicators couldn't be calculated (e.g., start of series)
        df_calc.dropna(subset=['ema1_strat', 'ema2_strat', 'atr_strat'], inplace=True)
        if df_calc.empty:
             lg.warning("DataFrame empty after dropping initial NaN indicator values. Cannot proceed.")
             return df # Return original df

        # Determine Trend Direction (UP if smoothed EMA > standard EMA)
        df_calc['trend_up_strat'] = (df_calc['ema1_strat'] > df_calc['ema2_strat']).astype('boolean') # Use nullable boolean
        # Identify exact points where trend changes (True on the first bar of the new trend)
        # Handle potential leading NaNs in trend_up_strat if dropna wasn't aggressive enough
        df_calc['trend_changed_strat'] = df_calc['trend_up_strat'].diff().fillna(False)

        # Initialize level columns as float (will hold results from calculations)
        level_cols = ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat',
                      'step_up_strat', 'step_dn_strat']
        for col in level_cols:
            df_calc[col] = np.nan

        # --- Calculate Dynamic Levels Based on Trend Changes (Vectorized) ---
        change_indices = df_calc.index[df_calc['trend_changed_strat']]
        if not change_indices.empty:
            # Get EMA1 and ATR values at the bar *before* the change occurred
            # Ensure we use numeric values for calculations
            ema1_prev = pd.to_numeric(df_calc['ema1_strat'].shift(1), errors='coerce').loc[change_indices]
            atr_prev = pd.to_numeric(df_calc['atr_strat'].shift(1), errors='coerce').loc[change_indices]

            # Create mask for valid calculations (where previous EMA and ATR exist and ATR > 0)
            valid_mask = pd.notna(ema1_prev) & pd.notna(atr_prev) & (atr_prev > 0)
            valid_indices = change_indices[valid_mask]

            if not valid_indices.empty:
                valid_ema1 = ema1_prev[valid_mask]
                valid_atr = atr_prev[valid_mask]

                # Calculate levels using float arithmetic
                upper = valid_ema1 + valid_atr * 3.0
                lower = valid_ema1 - valid_atr * 3.0
                lower_vol = lower + valid_atr * 4.0 # Top of lower vol zone
                upper_vol = upper - valid_atr * 4.0 # Bottom of upper vol zone

                # Calculate step sizes (ensure non-negative)
                step_up = (lower_vol - lower).clip(lower=0.0) / 100.0
                step_dn = (upper - upper_vol).clip(lower=0.0) / 100.0

                # Assign calculated levels to the DataFrame at the valid change indices
                df_calc.loc[valid_indices, 'upper_strat'] = upper
                df_calc.loc[valid_indices, 'lower_strat'] = lower
                df_calc.loc[valid_indices, 'lower_vol_strat'] = lower_vol
                df_calc.loc[valid_indices, 'upper_vol_strat'] = upper_vol
                df_calc.loc[valid_indices, 'step_up_strat'] = step_up
                df_calc.loc[valid_indices, 'step_dn_strat'] = step_dn

        # Forward fill the calculated levels until the next change
        for col in level_cols:
             df_calc[col] = df_calc[col].ffill()

        # --- Calculate Volume Metrics ---
        # Normalized Volume (0-100 based on max volume in rolling lookback window)
        # Use float volume here
        max_vol_lookback = df_calc['volume_float'].rolling(window=volume_lookback, min_periods=max(1, volume_lookback // 10)).max()
        # Avoid division by zero or NaN/inf results
        # --- This is the potential location of the original error if max_vol_lookback somehow became a NumPy array ---
        # --- The fix is to ensure df_calc['volume_float'] / max_vol_lookback results in a pandas Series or compatible type ---
        # --- Using np.where handles potential NaNs and division issues safely and returns a NumPy array ---
        vol_norm_array = np.where(
            pd.notna(max_vol_lookback) & (max_vol_lookback > 1e-9), # Check > small threshold
            (df_calc['volume_float'].fillna(0).to_numpy() / max_vol_lookback.to_numpy() * 100.0), # Ensure NumPy division
            0.0 # Set to 0 if max_vol is invalid or zero
        )
        # --- Assign the NumPy array back to the DataFrame column ---
        df_calc['vol_norm_strat'] = pd.Series(vol_norm_array, index=df_calc.index).clip(0.0, 100.0) # Clip result between 0 and 100

        # Volume-adjusted step amount (handle potential NaNs in steps or norm_vol)
        df_calc['vol_up_step_strat'] = (df_calc['step_up_strat'].fillna(0.0) * df_calc['vol_norm_strat'].fillna(0.0))
        df_calc['vol_dn_step_strat'] = (df_calc['step_dn_strat'].fillna(0.0) * df_calc['vol_norm_strat'].fillna(0.0))

        # Final Volume-Adjusted Trend Levels
        df_calc['vol_trend_up_level_strat'] = df_calc['lower_strat'].fillna(0.0) + df_calc['vol_up_step_strat'].fillna(0.0)
        df_calc['vol_trend_dn_level_strat'] = df_calc['upper_strat'].fillna(0.0) - df_calc['vol_dn_step_strat'].fillna(0.0)

        # --- Cumulative Volume Since Last Trend Change ---
        # Calculate volume delta using original Decimal volume for precision if available
        # Use float prices for comparison, float volume for calculation
        df_calc['volume_delta_float'] = np.where(
            df_calc['close'] > df_calc['open'], df_calc['volume_float'],
            np.where(df_calc['close'] < df_calc['open'], -df_calc['volume_float'], 0.0)
        ).fillna(0.0)
        df_calc['volume_total_float'] = df_calc['volume_float'].fillna(0.0)

        # Create a grouping key based on when the trend changes
        trend_block_group = df_calc['trend_changed_strat'].cumsum()
        # Calculate cumulative sums within each trend block (using float volumes)
        df_calc['cum_vol_delta_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_delta_float'].cumsum()
        df_calc['cum_vol_total_since_change_strat'] = df_calc.groupby(trend_block_group)['volume_total_float'].cumsum()

        # Track index (timestamp) of last trend change for reference
        # Get the timestamp where trend changed, forward fill it
        last_change_ts = df_calc.index.to_series().where(df_calc['trend_changed_strat']).ffill()
        df_calc['last_trend_change_idx'] = last_change_ts # Store the timestamp

        # Optional: Convert key result columns back to Decimal if needed downstream, though often float is fine
        # for col in ['upper_strat', 'lower_strat', ...]:
        #     df_calc[col] = df_calc[col].apply(lambda x: safe_decimal(x, default=np.nan))

        # Remove temporary float volume column
        df_calc.drop(columns=['volume_float', 'volume_delta_float', 'volume_total_float'], inplace=True, errors='ignore')

        lg.info("Volumatic Trend Levels calculation complete.")
        return df_calc

    except AttributeError as e:
         # Catch the specific error observed in the logs
         lg.error(f"{COLOR_ERROR}AttributeError during Volumatic Trend calculation: {e}{RESET}", exc_info=True)
         lg.error(f"This is likely the error from the logs ('{e}' on a numpy array). Check pandas/numpy interactions, especially around division or functions returning arrays.")
         # Return original DataFrame on error, potentially adding NaN columns if needed elsewhere
         return df
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Volumatic Trend calculation: {e}{RESET}", exc_info=True)
        # Return original DataFrame on error, potentially adding NaN columns if needed elsewhere
        return df


def calculate_pivot_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Identifies Pivot High (PH) and Pivot Low (PL) points based on configuration,
    used for Order Block detection. Includes data length checks and uses Decimal comparison.

    Args:
        df: DataFrame with OHLCV data (prices assumed numeric or Decimal).
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        DataFrame with added 'ph_strat' (Pivot High price as Decimal) and
        'pl_strat' (Pivot Low price as Decimal) columns.
        Returns input df with NaN columns if insufficient data.
    """
    lg = logger
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    left_h = config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    left_l = config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    lg.info(f"Calculating Pivot Points (Source: {source}, Left/Right H: {left_h}/{right_h}, L: {left_l}/{right_l})...")

    # Check for sufficient data length for pivot calculation window
    min_len_h = left_h + right_h + 1
    min_len_l = left_l + right_l + 1
    if len(df) < max(min_len_h, min_len_l):
        lg.warning(f"{COLOR_WARNING}Insufficient data ({len(df)} rows) for Pivot calculation (min H ~{min_len_h}, L ~{min_len_l}). Skipping.{RESET}")
        df['ph_strat'] = pd.NA # Use pandas NA for consistency
        df['pl_strat'] = pd.NA
        return df

    df_calc = df.copy()
    try:
        # Select price series based on source config ('high'/'low' for Wicks, 'close'/'open' for Bodys)
        # Ensure columns exist and convert to Decimal for comparison
        high_source_col = 'high' if source == "Wicks" else 'close'
        low_source_col = 'low' if source == "Wicks" else 'open'

        required_cols = [high_source_col, low_source_col]
        for col in required_cols:
             if col not in df_calc.columns:
                  lg.error(f"Missing required column '{col}' for pivot calculation. Skipping.")
                  df_calc['ph_strat'] = pd.NA
                  df_calc['pl_strat'] = pd.NA
                  return df_calc
             # Convert source columns to Decimal for accurate comparison
             df_calc[f'{col}_dec'] = df_calc[col].apply(lambda x: safe_decimal(x, default=pd.NA))

        high_col_dec = df_calc[f'{high_source_col}_dec']
        low_col_dec = df_calc[f'{low_source_col}_dec']

        # Initialize pivot columns with pandas NA (allows storing Decimals later)
        df_calc['ph_strat'] = pd.NA
        df_calc['pl_strat'] = pd.NA

        # --- Calculate Pivot Highs (PH) ---
        # Vectorized approach using rolling windows for finding local max/min
        # Window size includes left, right, and center bar
        ph_window_size = left_h + right_h + 1
        pl_window_size = left_l + right_l + 1

        # Find rolling maximum/minimum using the Decimal series
        # Note: rolling operations on object dtype (Decimal) can be slow. Consider performance on very large datasets.
        # If performance is an issue, converting to float *just* for rolling might be necessary, but risks precision loss.
        # center=True places the result at the center of the window (requires pandas >= 1.3)
        # Shift result to align with the original definition (pivot at index `i` requires looking `right_h` bars ahead)
        try:
            # Rolling max for PH
            # Use float for rolling performance, then compare with original Decimal for accuracy
            high_col_float = pd.to_numeric(high_col_dec, errors='coerce')
            rolling_max_float = high_col_float.rolling(window=ph_window_size, center=False).max().shift(-right_h)
            # Convert rolling max back to Decimal-like object for comparison (handle potential NaNs)
            rolling_max_dec = rolling_max_float.apply(lambda x: safe_decimal(x) if pd.notna(x) else pd.NA)
            # Identify where the original Decimal value equals the Decimal rolling max
            is_ph = (high_col_dec == rolling_max_dec) & high_col_dec.notna() & rolling_max_dec.notna()

            # Additional check for strict inequality (optional, can be slow)
            # If strict pivots are needed, a loop or more complex vectorization might be required.
            # This current method identifies pivots where the bar is >= neighbors.

            df_calc.loc[is_ph, 'ph_strat'] = high_col_dec[is_ph]

            # Rolling min for PL
            low_col_float = pd.to_numeric(low_col_dec, errors='coerce')
            rolling_min_float = low_col_float.rolling(window=pl_window_size, center=False).min().shift(-right_l)
            rolling_min_dec = rolling_min_float.apply(lambda x: safe_decimal(x) if pd.notna(x) else pd.NA)
            is_pl = (low_col_dec == rolling_min_dec) & low_col_dec.notna() & rolling_min_dec.notna()
            df_calc.loc[is_pl, 'pl_strat'] = low_col_dec[is_pl]

        except (TypeError, AttributeError) as e: # Catch issues with rolling or comparisons
             # Handle potential issues if rolling doesn't support Decimal directly or conversion fails
             lg.warning(f"Vectorized pivot calculation with Decimal/Float failed ({e}). Falling back to loop method. Performance may be impacted.")
             # Fallback Loop Implementation (ensure Decimal comparison)
             df_calc['ph_strat'] = pd.NA # Reset before loop
             df_calc['pl_strat'] = pd.NA
             # Ensure iterrows uses Decimal columns
             for i in range(left_h, len(df_calc) - right_h):
                 idx = df_calc.index[i]
                 pivot_val = high_col_dec.loc[idx]
                 if pd.isna(pivot_val): continue
                 # Check left: pivot_val > all left values (strict)
                 left_vals = high_col_dec.iloc[i-left_h : i]
                 if left_vals.isna().any() or not (pivot_val > left_vals[left_vals.notna()]).all(): continue
                 # Check right: pivot_val > all right values (strict)
                 right_vals = high_col_dec.iloc[i+1 : i+right_h+1]
                 if right_vals.isna().any() or not (pivot_val > right_vals[right_vals.notna()]).all(): continue
                 df_calc.loc[idx, 'ph_strat'] = pivot_val # Store Decimal

             for i in range(left_l, len(df_calc) - right_l):
                 idx = df_calc.index[i]
                 pivot_val = low_col_dec.loc[idx]
                 if pd.isna(pivot_val): continue
                 # Check left: pivot_val < all left values (strict)
                 left_vals = low_col_dec.iloc[i-left_l : i]
                 if left_vals.isna().any() or not (pivot_val < left_vals[left_vals.notna()]).all(): continue
                 # Check right: pivot_val < all right values (strict)
                 right_vals = low_col_dec.iloc[i+1 : i+right_l+1]
                 if right_vals.isna().any() or not (pivot_val < right_vals[right_vals.notna()]).all(): continue
                 df_calc.loc[idx, 'pl_strat'] = pivot_val # Store Decimal

        # Clean up temporary Decimal columns
        df_calc.drop(columns=[f'{high_source_col}_dec', f'{low_source_col}_dec'], inplace=True, errors='ignore')

        # Convert final pivot columns to object type to hold Decimals/NAs
        df_calc['ph_strat'] = df_calc['ph_strat'].astype(object)
        df_calc['pl_strat'] = df_calc['pl_strat'].astype(object)


        lg.info(f"Pivot Point calculation complete. Found {df_calc['ph_strat'].notna().sum()} PH, {df_calc['pl_strat'].notna().sum()} PL.")
        return df_calc

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Pivot calculation: {e}{RESET}", exc_info=True)
        # Return original DataFrame with NA columns on error
        df['ph_strat'] = pd.NA
        df['pl_strat'] = pd.NA
        return df


def manage_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Identifies, creates, and manages the state (active/closed/trimmed) of Order Blocks (OBs)
    based on pivot points and subsequent price action. Limits number of active OBs.
    Uses Decimal for price comparisons.

    Args:
        df: DataFrame with OHLCV and pivot columns ('ph_strat', 'pl_strat' - expected to hold Decimal or NA).
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Tuple: (DataFrame with active OB references, list of all tracked bull OBs, list of all tracked bear OBs).
               Returns input df and empty lists if insufficient data or error.
    """
    lg = logger
    lg.info("Managing Order Block Boxes...")
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    # Use pivot right lengths to determine the OB candle relative to the pivot
    ob_candle_offset_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    ob_candle_offset_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    max_boxes = config.get("volbot_max_boxes", DEFAULT_VOLBOT_MAX_BOXES)

    df_calc = df.copy()
    bull_boxes: List[Dict] = [] # Stores all created bull boxes (active or closed)
    bear_boxes: List[Dict] = [] # Stores all created bear boxes
    active_bull_boxes: List[Dict] = [] # Stores currently active bull boxes
    active_bear_boxes: List[Dict] = [] # Stores currently active bear boxes
    box_counter = 0

    # Check if pivot columns exist
    if 'ph_strat' not in df_calc.columns or 'pl_strat' not in df_calc.columns:
        lg.warning(f"{COLOR_WARNING}Pivot columns ('ph_strat', 'pl_strat') not found. Skipping OB management.{RESET}")
        # Add placeholder columns to prevent errors later
        df_calc['active_bull_ob_strat'] = None
        df_calc['active_bear_ob_strat'] = None
        return df_calc, bull_boxes, bear_boxes

    # Initialize columns to store references to the active OB dict for each bar
    # Use object dtype to store dictionaries or None
    df_calc['active_bull_ob_strat'] = pd.Series(dtype='object')
    df_calc['active_bear_ob_strat'] = pd.Series(dtype='object')

    # Convert necessary price columns to Decimal for the loop
    price_cols_to_convert = ['open', 'high', 'low', 'close']
    decimal_col_names = {}
    for col in price_cols_to_convert:
        if col in df_calc.columns:
            dec_col_name = f'{col}_dec'
            df_calc[dec_col_name] = df_calc[col].apply(lambda x: safe_decimal(x, default=pd.NA))
            decimal_col_names[col] = dec_col_name # Store mapping for easy access
        else:
            lg.error(f"Missing required price column '{col}' for OB management. Skipping.")
            return df_calc, [], [] # Cannot proceed

    try:
        # Iterate through each bar to potentially create new OBs and manage existing ones
        for i in range(len(df_calc)):
            current_idx = df_calc.index[i]
            # Use pre-converted Decimal columns
            current_close_dec = df_calc.at[current_idx, decimal_col_names['close']]

            # Skip if current close price is invalid
            if pd.isna(current_close_dec):
                lg.debug(f"Skipping OB management for index {current_idx}: Invalid close price.")
                # Assign None to active OB refs for this bar
                df_calc.at[current_idx, 'active_bull_ob_strat'] = None
                df_calc.at[current_idx, 'active_bear_ob_strat'] = None
                continue

            # --- Manage Existing Active Boxes ---
            # Check for mitigation BEFORE adding new boxes for the current bar
            next_active_bull = []
            active_bull_ref_for_current_bar = None
            for box in active_bull_boxes:
                # Bull OB Mitigation: Close below the OB bottom (use Decimal comparison)
                if current_close_dec < box['bottom']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx # Record mitigation timestamp
                    lg.debug(f"Closed Bullish OB: {box['id']} at {current_idx} (Close {current_close_dec} < Bottom {box['bottom']})")
                    # Don't add to next_active_bull
                else:
                    # Box remains active
                    next_active_bull.append(box)
                    # Check if current price (close) is inside this active box
                    if box['bottom'] <= current_close_dec <= box['top']:
                         # If multiple active boxes contain the price, prioritize the most recent one (last one added in loop).
                         # Or sort active_boxes by 'pivot_idx' if specific priority is needed.
                         active_bull_ref_for_current_bar = box

            active_bull_boxes = next_active_bull # Update the list of active bull boxes

            next_active_bear = []
            active_bear_ref_for_current_bar = None
            for box in active_bear_boxes:
                # Bear OB Mitigation: Close above the OB top
                if current_close_dec > box['top']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx
                    lg.debug(f"Closed Bearish OB: {box['id']} at {current_idx} (Close {current_close_dec} > Top {box['top']})")
                    # Don't add to next_active_bear
                else:
                    # Box remains active
                    next_active_bear.append(box)
                    # Check if current price is inside this active box
                    if box['bottom'] <= current_close_dec <= box['top']:
                        active_bear_ref_for_current_bar = box

            active_bear_boxes = next_active_bear # Update the list of active bear boxes

            # Store the active OB reference (or None) in the DataFrame for this bar *before* creating new ones
            # Use .loc for setting value with object dtype
            df_calc.loc[current_idx, 'active_bull_ob_strat'] = active_bull_ref_for_current_bar
            df_calc.loc[current_idx, 'active_bear_ob_strat'] = active_bear_ref_for_current_bar


            # --- Create New Bearish OB (Based on Pivot Highs) ---
            # Pivot High is stored in 'ph_strat' (Decimal or NA)
            if pd.notna(df_calc.at[current_idx, 'ph_strat']):
                ob_candle_iloc = i - ob_candle_offset_h
                if ob_candle_iloc >= 0:
                    ob_candle_idx = df_calc.index[ob_candle_iloc]
                    # Define Bearish OB range using pre-converted Decimal prices
                    top_p_dec, bottom_p_dec = pd.NA, pd.NA
                    ob_open_dec = df_calc.at[ob_candle_idx, decimal_col_names['open']]
                    ob_close_dec = df_calc.at[ob_candle_idx, decimal_col_names['close']]
                    ob_high_dec = df_calc.at[ob_candle_idx, decimal_col_names['high']]

                    if source == "Bodys": # Bearish Body OB: Open to Close
                        top_p_dec = ob_open_dec
                        bottom_p_dec = ob_close_dec
                    else: # Wicks (Default): High to Close
                        top_p_dec = ob_high_dec
                        bottom_p_dec = ob_close_dec

                    # Validate prices and create box
                    if pd.notna(top_p_dec) and pd.notna(bottom_p_dec):
                        # Ensure top > bottom using Decimal comparison
                        top_price = max(top_p_dec, bottom_p_dec)
                        bottom_price = min(top_p_dec, bottom_p_dec)
                        # Avoid creating zero-height boxes (use a small tolerance if needed)
                        if top_price > bottom_price:
                            box_counter += 1
                            new_box = {
                                'id': f'BearOB_{box_counter}', 'type': 'bear',
                                'start_idx': ob_candle_idx, # Timestamp of the OB candle
                                'pivot_idx': current_idx,   # Timestamp where pivot was confirmed
                                'end_idx': None,           # Timestamp when mitigated (null when active)
                                'top': top_price, 'bottom': bottom_price, 'state': 'active' # Store Decimals
                            }
                            bear_boxes.append(new_box)
                            active_bear_boxes.append(new_box) # Add to active list
                            lg.debug(f"Created Bearish OB: {new_box['id']} (Pivot: {current_idx}, Candle: {ob_candle_idx}, Range: [{bottom_price}, {top_price}])")

            # --- Create New Bullish OB (Based on Pivot Lows) ---
            if pd.notna(df_calc.at[current_idx, 'pl_strat']):
                ob_candle_iloc = i - ob_candle_offset_l
                if ob_candle_iloc >= 0:
                    ob_candle_idx = df_calc.index[ob_candle_iloc]
                    # Define Bullish OB range using pre-converted Decimal prices
                    top_p_dec, bottom_p_dec = pd.NA, pd.NA
                    ob_open_dec = df_calc.at[ob_candle_idx, decimal_col_names['open']]
                    ob_close_dec = df_calc.at[ob_candle_idx, decimal_col_names['close']]
                    ob_low_dec = df_calc.at[ob_candle_idx, decimal_col_names['low']]

                    if source == "Bodys": # Bullish Body OB: Close to Open
                        top_p_dec = ob_close_dec
                        bottom_p_dec = ob_open_dec
                    else: # Wicks (Default): Open to Low
                        top_p_dec = ob_open_dec
                        bottom_p_dec = ob_low_dec

                    # Validate prices and create box
                    if pd.notna(top_p_dec) and pd.notna(bottom_p_dec):
                        top_price = max(top_p_dec, bottom_p_dec)
                        bottom_price = min(top_p_dec, bottom_p_dec)
                        if top_price > bottom_price:
                            box_counter += 1
                            new_box = {
                                'id': f'BullOB_{box_counter}', 'type': 'bull',
                                'start_idx': ob_candle_idx, 'pivot_idx': current_idx,
                                'end_idx': None,
                                'top': top_price, 'bottom': bottom_price, 'state': 'active' # Store Decimals
                            }
                            bull_boxes.append(new_box)
                            active_bull_boxes.append(new_box) # Add to active list
                            lg.debug(f"Created Bullish OB: {new_box['id']} (Pivot: {current_idx}, Candle: {ob_candle_idx}, Range: [{bottom_price}, {top_price}])")

            # --- Limit Number of Active Boxes ---
            # Keep only the 'max_boxes' most recent *active* boxes based on pivot confirmation time
            if len(active_bull_boxes) > max_boxes:
                # Sort by pivot confirmation timestamp (newest first)
                active_bull_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True)
                removed_boxes = active_bull_boxes[max_boxes:]
                active_bull_boxes = active_bull_boxes[:max_boxes]
                for box in removed_boxes:
                    if box['state'] == 'active': # Only mark active boxes as trimmed
                        box['state'] = 'trimmed' # Mark as trimmed, not closed by price
                        lg.debug(f"Trimmed older active Bull OB: {box['id']} (Pivot: {box['pivot_idx']})")

            if len(active_bear_boxes) > max_boxes:
                active_bear_boxes.sort(key=lambda x: x['pivot_idx'], reverse=True)
                removed_boxes = active_bear_boxes[max_boxes:]
                active_bear_boxes = active_bear_boxes[:max_boxes]
                for box in removed_boxes:
                    if box['state'] == 'active':
                        box['state'] = 'trimmed'
                        lg.debug(f"Trimmed older active Bear OB: {box['id']} (Pivot: {box['pivot_idx']})")

        # End of loop

        # Clean up temporary Decimal columns
        for dec_col_name in decimal_col_names.values():
             df_calc.drop(columns=[dec_col_name], inplace=True, errors='ignore')

        num_active_bull = sum(1 for box in active_bull_boxes if box['state'] == 'active')
        num_active_bear = sum(1 for box in active_bear_boxes if box['state'] == 'active')
        lg.info(f"Order Block management complete. Total created: {len(bull_boxes)} Bull, {len(bear_boxes)} Bear. Currently active: {num_active_bull} Bull, {num_active_bear} Bear.")
        # Return the DataFrame and the lists containing *all* boxes (active, closed, trimmed)
        return df_calc, bull_boxes, bear_boxes

    except Exception as e:
        lg.error(f"{COLOR_ERROR}Error during Order Block management: {e}{RESET}", exc_info=True)
        # Return original df state with None columns if error occurs
        df['active_bull_ob_strat'] = None
        df['active_bear_ob_strat'] = None
        return df, [], []


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """
    Analyzes trading data using Volbot strategy, generates signals, and calculates risk metrics.
    Handles market precision and provides utility methods. Uses Decimal for internal calculations.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any], # Pass the fetched market info
    ) -> None:
        """
        Initializes the analyzer with data, config, logger, and market info.
        Calculates indicators upon initialization.

        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame (assumed cleaned with Decimal types where appropriate).
            logger (logging.Logger): Logger instance.
            config (Dict[str, Any]): Configuration dictionary.
            market_info (Dict[str, Any]): Market information dictionary from ccxt.
        """
        self.df_raw = df
        self.df_processed = pd.DataFrame() # Populated by _calculate_indicators
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN") # User-friendly interval
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN") # CCXT format

        # --- Determine precision and tick size from market_info ---
        self.min_tick_size = self._determine_min_tick_size()
        self.price_precision = self._determine_price_precision() # Decimals for price formatting
        self.amount_precision = self._determine_amount_precision() # Decimals for amount formatting
        self.amount_step_size = self._determine_amount_step_size() # The actual step value

        self.logger.debug(f"Analyzer initialized for {self.symbol}: "
                          f"TickSize={self.min_tick_size}, PricePrec={self.price_precision}, "
                          f"AmountPrec={self.amount_precision}, AmountStep={self.amount_step_size}")

        # Strategy state variables
        self.strategy_state: Dict[str, Any] = {} # Stores latest indicator values (using Decimal where appropriate)
        self.latest_active_bull_ob: Optional[Dict] = None # Ref to latest active bull OB dict
        self.latest_active_bear_ob: Optional[Dict] = None # Ref to latest active bear OB dict
        self.all_bull_boxes: List[Dict] = [] # All bull OBs generated
        self.all_bear_boxes: List[Dict] = [] # All bear OBs generated

        # Calculate indicators immediately on initialization
        self._calculate_indicators()
        # Update state with the latest calculated values
        self._update_latest_strategy_state()

    def _determine_min_tick_size(self) -> Decimal:
        """Determine minimum price increment (tick size) from market info as Decimal."""
        lg = self.logger
        try:
            # Prefer precision info 'price' which often *is* the tick size
            price_prec_val = self.market_info.get('precision', {}).get('price')
            tick = safe_decimal(price_prec_val)
            if tick and tick > 0:
                 lg.debug(f"Tick size determined from precision.price: {tick}")
                 return tick

            # Fallback to limits info 'price' 'min' (less common for tick size)
            min_price_limit = self.market_info.get('limits', {}).get('price', {}).get('min')
            tick = safe_decimal(min_price_limit)
            # Check if it looks like a tick size (e.g., 0.01, 0.5) rather than a minimum trading price (e.g., 1000)
            if tick and tick > 0 and tick < 10: # Heuristic: ticks are usually small
                lg.debug(f"Tick size determined from limits.price.min: {tick}")
                return tick

        except Exception as e:
            lg.warning(f"Could not reliably determine tick size for {self.symbol} from market info: {e}. Using fallback.")

        # Absolute fallback (adjust based on typical market, e.g., 0.1 for BTC, 0.01 for ETH)
        # Determine fallback based on typical price magnitude if possible
        last_price = safe_decimal(self.df_raw['close'].iloc[-1]) if not self.df_raw.empty and 'close' in self.df_raw.columns else None
        if last_price:
            if last_price > 1000: default_tick = Decimal('0.1')
            elif last_price > 10: default_tick = Decimal('0.01')
            elif last_price > 0.1: default_tick = Decimal('0.001')
            else: default_tick = Decimal('0.00001')
        else:
            default_tick = Decimal('0.0001') # Generic fallback

        lg.warning(f"Using default/fallback tick size {default_tick} for {self.symbol}.")
        return default_tick

    def _determine_price_precision(self) -> int:
        """Determine decimal places for price formatting based on tick size."""
        try:
            tick_size = self.min_tick_size # Use the already determined tick size
            if tick_size > 0:
                # Calculate decimal places from the tick size
                # normalize() removes trailing zeros, as_tuple().exponent gives power of 10
                return abs(tick_size.normalize().as_tuple().exponent)
        except Exception as e:
            self.logger.warning(f"Could not determine price precision from tick size ({self.min_tick_size}) for {self.symbol}: {e}. Using default.")
        # Default fallback precision
        return 4

    def _determine_amount_step_size(self) -> Decimal:
        """Determine the minimum amount increment (step size) as Decimal."""
        lg = self.logger
        try:
            # Prefer precision.amount if it's float/str (likely the step size)
            amount_prec_val = self.market_info.get('precision', {}).get('amount')
            step_size = safe_decimal(amount_prec_val)
            if step_size and step_size > 0:
                lg.debug(f"Amount step size determined from precision.amount: {step_size}")
                return step_size

            # If precision.amount is integer, assume it's decimal places, calculate step
            if isinstance(amount_prec_val, int) and amount_prec_val >= 0:
                step_size = Decimal('1') / (Decimal('10') ** amount_prec_val)
                lg.debug(f"Amount step size calculated from precision.amount (places={amount_prec_val}): {step_size}")
                return step_size

            # Fallback: Check limits.amount.min if it looks like a step size
            min_amount_limit = self.market_info.get('limits', {}).get('amount', {}).get('min')
            step_size = safe_decimal(min_amount_limit)
            # Heuristic: Step size is usually small, often power of 10
            if step_size and step_size > 0 and step_size <= 1:
                 lg.debug(f"Amount step size determined from limits.amount.min: {step_size}")
                 return step_size

        except Exception as e:
             lg.warning(f"Could not determine amount step size for {self.symbol}: {e}. Using default.")

        # Default fallback step size
        default_step = Decimal('0.00000001') # Common fallback for crypto base amounts (8 decimals)
        lg.warning(f"Using default amount step size {default_step} for {self.symbol}.")
        return default_step

    def _determine_amount_precision(self) -> int:
        """Determine decimal places for amount formatting based on step size."""
        try:
            step_size = self.amount_step_size # Use the already determined step size
            if step_size > 0:
                return abs(step_size.normalize().as_tuple().exponent)
        except Exception as e:
            self.logger.warning(f"Could not determine amount precision from step size ({self.amount_step_size}) for {self.symbol}: {e}. Using default.")
        # Default fallback precision
        return 8

    def _calculate_indicators(self) -> None:
        """Calculates Risk Management ATR and Volbot strategy indicators. Populates df_processed."""
        if self.df_raw.empty:
            self.logger.warning(f"{COLOR_WARNING}Raw DataFrame empty, cannot calculate indicators for {self.symbol}.{RESET}")
            self.df_processed = pd.DataFrame() # Ensure it's empty
            return

        # --- Check Minimum Data Length ---
        buffer = 50 # Add ample buffer for initial NaNs and stability
        min_required_data = buffer # Start with buffer
        try:
             # Add lengths from config dynamically
             min_len_volbot = 0
             if self.config.get("volbot_enabled", True):
                 min_len_volbot = max(
                     self.config.get("volbot_length", DEFAULT_VOLBOT_LENGTH) + 3, # SWMA needs ~3 extra
                     self.config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
                     self.config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
                     self.config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H) + self.config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H) + 1,
                     self.config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L) + self.config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L) + 1
                 )
             min_len_risk = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
             min_required_data = max(min_len_volbot, min_len_risk, buffer) # Ensure at least buffer length
        except Exception as e:
             self.logger.error(f"Error calculating minimum required data length: {e}. Using fallback minimum.")
             min_required_data = 250 # Fallback minimum

        if len(self.df_raw) < min_required_data:
            self.logger.warning(
                f"{COLOR_WARNING}Insufficient data ({len(self.df_raw)} points) for {self.symbol}. "
                f"Need ~{min_required_data} for reliable calculations. Results may be inaccurate or missing.{RESET}"
            )
            # Proceed, but calculations might return NaNs or be unreliable. Functions below should handle this.

        try:
            df_calc = self.df_raw.copy()

            # Ensure price columns are numeric for TA calculations
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
            df_calc.dropna(subset=price_cols, inplace=True) # Drop rows where essential prices are invalid

            if df_calc.empty:
                self.logger.warning(f"DataFrame became empty after coercing/dropping NaN prices for {self.symbol}.")
                self.df_processed = pd.DataFrame()
                return

            # 1. Calculate Risk Management ATR (using float for TA Lib)
            atr_period_risk = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc['atr_risk'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=atr_period_risk)
            self.logger.debug(f"Calculated Risk Management ATR (Length: {atr_period_risk})")

            # 2. Calculate Volbot Strategy Indicators (if enabled)
            if self.config.get("volbot_enabled", True):
                df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
                # Check if the previous calculation failed (returned original df or empty)
                if df_calc is self.df_raw or df_calc.empty:
                    self.logger.error(f"Volumatic Trend calculation failed or returned empty for {self.symbol}. Skipping further strategy calculations.")
                    self.df_processed = pd.DataFrame()
                    return

                df_calc = calculate_pivot_order_blocks(df_calc, self.config, self.logger)
                if df_calc is self.df_raw or df_calc.empty: # Check again after pivots
                    self.logger.error(f"Pivot calculation failed or returned empty for {self.symbol}. Skipping OB management.")
                    self.df_processed = pd.DataFrame()
                    return

                # Pass the modified df_calc to manage_order_blocks
                df_calc, self.all_bull_boxes, self.all_bear_boxes = manage_order_blocks(df_calc, self.config, self.logger)
            else:
                 self.logger.info("Volbot strategy calculation skipped (disabled in config).")
                 # Add NaN/NA placeholders if strategy disabled but columns might be expected later
                 placeholder_cols = [
                    'ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat',
                    'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat',
                    'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat', 'vol_dn_step_strat',
                    'vol_trend_up_level_strat', 'vol_trend_dn_level_strat', 'cum_vol_delta_since_change_strat',
                    'cum_vol_total_since_change_strat', 'last_trend_change_idx',
                    'ph_strat', 'pl_strat', 'active_bull_ob_strat', 'active_bear_ob_strat'
                 ]
                 for col in placeholder_cols:
                     if col not in df_calc.columns:
                         # Use pd.NA for object columns, np.nan for numeric
                         dtype = object if col in ['ph_strat','pl_strat','active_bull_ob_strat','active_bear_ob_strat','last_trend_change_idx'] else np.nan
                         df_calc[col] = dtype if dtype is object else np.nan


            # --- Convert key calculated columns to Decimal for internal state ---
            # Keep results in df_processed mostly as float64 (from TA libs) or object (for pivots/OB refs)
            # Conversion to Decimal happens primarily in _update_latest_strategy_state
            # Exception: Convert Risk ATR to Decimal now as it's used directly in calculations
            if 'atr_risk' in df_calc.columns:
                 df_calc['atr_risk_dec'] = df_calc['atr_risk'].apply(lambda x: safe_decimal(x, default=pd.NA))

            self.df_processed = df_calc
            self.logger.debug(f"Indicator calculations complete for {self.symbol}. Processed DF has {len(self.df_processed)} rows.")

        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Error calculating indicators for {self.symbol}: {e}{RESET}", exc_info=True)
            self.df_processed = pd.DataFrame() # Ensure empty on error

    def _update_latest_strategy_state(self) -> None:
        """
        Updates `strategy_state` dictionary with the latest available values
        from the last row of `df_processed`. Converts relevant values to Decimal.
        """
        self.strategy_state = {} # Reset state each time
        self.latest_active_bull_ob = None
        self.latest_active_bear_ob = None

        if self.df_processed.empty or len(self.df_processed) == 0:
            self.logger.warning(f"Cannot update state: Processed DataFrame is empty for {self.symbol}.")
            return

        try:
            # Get the last row of the processed data
            latest = self.df_processed.iloc[-1]

            if latest.isnull().all():
                self.logger.warning(f"{COLOR_WARNING}Last row of processed DataFrame contains all NaNs for {self.symbol}. Check data source or indicator calculations.{RESET}")
                return

            # List of columns to potentially extract (core + risk + strategy results)
            # Prioritize original/Decimal columns if available, fallback to float results from TA
            all_possible_cols = [
                # Core data (prefer Decimal if available from fetch_klines)
                'open', 'high', 'low', 'close', 'volume',
                # Risk ATR (use the pre-converted Decimal column)
                'atr_risk_dec',
                # Volbot specific (mostly float results from TA, pivots/OBs are object/Decimal)
                'ema1_strat', 'ema2_strat', 'atr_strat', 'trend_up_strat', 'trend_changed_strat',
                'upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat',
                'step_up_strat', 'step_dn_strat', 'vol_norm_strat', 'vol_up_step_strat',
                'vol_dn_step_strat', 'vol_trend_up_level_strat', 'vol_trend_dn_level_strat',
                'cum_vol_delta_since_change_strat', 'cum_vol_total_since_change_strat',
                'last_trend_change_idx', 'ph_strat', 'pl_strat',
                'active_bull_ob_strat', 'active_bear_ob_strat'
            ]

            # Extract values, handling missing columns and converting to Decimal where appropriate
            for col in all_possible_cols:
                col_name_in_state = col.replace('_dec', '') # Use base name in state dict (e.g., 'atr_risk')
                if col in latest.index and pd.notna(latest[col]):
                    value = latest[col]
                    # Handle specific types
                    if col_name_in_state in ['active_bull_ob_strat', 'active_bear_ob_strat']:
                        # These are expected to be dicts or None/NA
                        self.strategy_state[col_name_in_state] = value if isinstance(value, dict) else None
                    elif col_name_in_state in ['trend_up_strat', 'trend_changed_strat']:
                        # Handle boolean/nullable boolean
                        # Check for pandas BooleanNA explicitly before converting
                        if pd.isna(value): self.strategy_state[col_name_in_state] = None
                        else: self.strategy_state[col_name_in_state] = bool(value)
                    elif col_name_in_state == 'last_trend_change_idx':
                        # Store timestamp as is, or None
                        self.strategy_state[col_name_in_state] = value if isinstance(value, pd.Timestamp) else None
                    elif isinstance(value, (Decimal, pd._libs.missing.NAType)):
                         # If already Decimal or NA, store as is (or None if NA)
                         self.strategy_state[col_name_in_state] = value if isinstance(value, Decimal) else None
                    else:
                        # Attempt Decimal conversion for other numeric types (float, int, string representation)
                        decimal_value = safe_decimal(value, default=None)
                        self.strategy_state[col_name_in_state] = decimal_value # Store Decimal or None

            # Ensure 'atr_risk' key exists, even if '_dec' version wasn't present
            if 'atr_risk' not in self.strategy_state and 'atr_risk' in latest.index:
                 self.strategy_state['atr_risk'] = safe_decimal(latest['atr_risk'], default=None)


            # Update latest active OB references separately
            self.latest_active_bull_ob = self.strategy_state.get('active_bull_ob_strat')
            self.latest_active_bear_ob = self.strategy_state.get('active_bear_ob_strat')
            # Add convenience boolean flags
            self.strategy_state['is_in_active_bull_ob'] = self.latest_active_bull_ob is not None
            self.strategy_state['is_in_active_bear_ob'] = self.latest_active_bear_ob is not None

            # Log the updated state compactly, formatting Decimals
            log_state = {}
            price_fmt = f".{self.price_precision}f"
            vol_fmt = ".2f" # Example format for volume/cumulative volume
            atr_fmt = ".5f" # Example format for ATR

            for k, v in self.strategy_state.items():
                 if isinstance(v, Decimal):
                     # Use specific formatting based on key name patterns
                     if any(p in k for p in ['price', 'level', 'strat', 'open', 'high', 'low', 'close', 'tp', 'sl', 'upper', 'lower']): fmt = price_fmt
                     elif 'vol' in k: fmt = vol_fmt
                     elif 'atr' in k: fmt = atr_fmt
                     else: fmt = ".8f" # Generic Decimal format
                     log_state[k] = f"{v:{fmt}}"
                 elif isinstance(v, (bool, pd._libs.missing.NAType)):
                     log_state[k] = str(v) if pd.notna(v) else 'None'
                 elif isinstance(v, pd.Timestamp):
                     log_state[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                 elif v is not None and k not in ['active_bull_ob_strat', 'active_bear_ob_strat']: # Avoid logging full OB dicts here
                     log_state[k] = v # Keep other types as is

            self.logger.debug(f"Latest strategy state updated for {self.symbol}: {log_state}")
            if self.latest_active_bull_ob: self.logger.debug(f"  Latest Active Bull OB: ID={self.latest_active_bull_ob.get('id')}, Range=[{self.latest_active_bull_ob.get('bottom')}, {self.latest_active_bull_ob.get('top')}]")
            if self.latest_active_bear_ob: self.logger.debug(f"  Latest Active Bear OB: ID={self.latest_active_bear_ob.get('id')}, Range=[{self.latest_active_bear_ob.get('bottom')}, {self.latest_active_bear_ob.get('top')}]")

        except IndexError:
            self.logger.error(f"{COLOR_ERROR}Error accessing latest row (index -1) for {self.symbol}. Processed DataFrame might be empty or too short.{RESET}")
        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Unexpected error updating latest strategy state for {self.symbol}: {e}{RESET}", exc_info=True)

    # --- Utility Functions ---
    def get_price_precision(self) -> int:
        """Returns the number of decimal places for price formatting."""
        return self.price_precision

    def get_amount_precision(self) -> int:
        """Returns the number of decimal places for amount formatting."""
        return self.amount_precision

    def get_min_tick_size(self) -> Decimal:
        """Returns the minimum price increment (tick size) as a Decimal."""
        return self.min_tick_size

    def get_amount_step_size(self) -> Decimal:
        """Returns the minimum amount increment (step size) as a Decimal."""
        return self.amount_step_size

    def round_price(self, price: Union[Decimal, float, str, None]) -> Optional[Decimal]:
        """Rounds a given price DOWN for BIDs/SELL SLs, UP for ASKs/BUY SLs to the symbol's minimum tick size."""
        price_decimal = safe_decimal(price)
        min_tick = self.min_tick_size
        if price_decimal is None or min_tick is None or min_tick <= 0:
            self.logger.error(f"Cannot round price: Invalid input price ({price}) or min_tick ({min_tick})")
            return None
        try:
            # Quantize to the tick size. We typically round *towards* the market worse price.
            # However, for general rounding, ROUND_HALF_UP is common.
            # For specific use cases (like SL/TP), the calling function should apply appropriate rounding.
            # Here, we provide a general rounding, usually DOWN for safety unless specified otherwise.
            # Using ROUND_DOWN generally:
            rounded_price = (price_decimal / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
            return rounded_price
        except Exception as e:
             self.logger.error(f"Error rounding price {price_decimal} with tick {min_tick}: {e}")
             return None

    def round_amount(self, amount: Union[Decimal, float, str, None]) -> Optional[Decimal]:
        """Rounds (truncates/floors) a given amount DOWN to the symbol's amount step size."""
        amount_decimal = safe_decimal(amount)
        step_size = self.amount_step_size
        if amount_decimal is None or step_size is None or step_size <= 0:
            self.logger.error(f"Cannot round amount: Invalid input amount ({amount}) or step_size ({step_size})")
            return None

        try:
            # Truncate/floor to the nearest step size
            # Ensure amount_decimal is not negative before flooring if that's a possibility
            if amount_decimal < 0:
                 # Handle negative amounts if necessary (e.g., round towards zero)
                 # For typical position sizing, amount should be positive.
                 self.logger.warning(f"Rounding a negative amount: {amount_decimal}. Behavior might be unexpected.")
                 # Example: Round towards zero for negative
                 # rounded_amount = (amount_decimal / step_size).quantize(Decimal('1'), rounding=ROUND_UP) * step_size
                 # return rounded_amount

            # For positive amounts, floor division works well
            rounded_amount = (amount_decimal // step_size) * step_size
            # Alternative using quantize with ROUND_DOWN
            # rounding_factor = Decimal('1e-' + str(self.amount_precision)) # Assumes step size matches precision places
            # rounded_amount = amount_decimal.quantize(rounding_factor, rounding=ROUND_DOWN)

            return rounded_amount

        except Exception as e:
             self.logger.error(f"Error rounding amount {amount_decimal} with step size {step_size}: {e}")
             return None

    # --- Signal Generation ---
    def generate_trading_signal(self) -> str:
        """
        Generates "BUY", "SELL", or "HOLD" signal based on Volbot rules defined in config.
        Relies on values in `self.strategy_state`.

        Returns:
            str: "BUY", "SELL", or "HOLD".
        """
        signal = "HOLD" # Default signal
        if not self.strategy_state:
            self.logger.debug("Cannot generate signal: Strategy state is empty.")
            return signal
        if not self.config.get("volbot_enabled", True):
            self.logger.debug("Cannot generate signal: Volbot strategy is disabled in config.")
            return signal

        try:
            # Get relevant state values (handle potential None)
            is_trend_up = self.strategy_state.get('trend_up_strat') # Boolean or None
            trend_changed = self.strategy_state.get('trend_changed_strat', False) # Default to False if missing
            is_in_bull_ob = self.strategy_state.get('is_in_active_bull_ob', False)
            is_in_bear_ob = self.strategy_state.get('is_in_active_bear_ob', False)

            # Get signal generation rules from config
            signal_on_flip = self.config.get("volbot_signal_on_trend_flip", True)
            signal_on_ob = self.config.get("volbot_signal_on_ob_entry", True)

            # Check if trend is determined
            if is_trend_up is None:
                self.logger.debug("Volbot signal: HOLD (Trend state could not be determined - likely insufficient data or start of series)")
                return "HOLD"

            trend_str = f"{COLOR_UP}UP{RESET}" if is_trend_up else f"{COLOR_DN}DOWN{RESET}"
            ob_status = ""
            if is_in_bull_ob: ob_status += f" {COLOR_BULL_BOX}InBullOB{RESET}"
            if is_in_bear_ob: ob_status += f" {COLOR_BEAR_BOX}InBearOB{RESET}"
            if not ob_status: ob_status = " NoActiveOB"

            # Rule 1: Trend Flip Signal (Highest Priority if enabled)
            if signal_on_flip and trend_changed:
                signal = "BUY" if is_trend_up else "SELL"
                reason = f"Trend flipped to {trend_str}"
                color = COLOR_UP if is_trend_up else COLOR_DN
                self.logger.info(f"{color}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                return signal

            # Rule 2: Order Block Entry Signal (If no flip and enabled)
            if signal_on_ob:
                if is_trend_up and is_in_bull_ob:
                    signal = "BUY"
                    ob_id = self.latest_active_bull_ob.get('id', 'N/A') if self.latest_active_bull_ob else 'N/A'
                    reason = f"Price in Bull OB '{ob_id}' during {trend_str} Trend"
                    self.logger.info(f"{COLOR_BULL_BOX}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                    return signal
                elif not is_trend_up and is_in_bear_ob:
                    signal = "SELL"
                    ob_id = self.latest_active_bear_ob.get('id', 'N/A') if self.latest_active_bear_ob else 'N/A'
                    reason = f"Price in Bear OB '{ob_id}' during {trend_str} Trend"
                    self.logger.info(f"{COLOR_BEAR_BOX}Volbot Signal: {signal} (Reason: {reason}){RESET}")
                    return signal

            # Rule 3: Default to HOLD if no entry conditions met
            # Log current state for HOLD signal
            self.logger.info(f"Volbot Signal: HOLD (Conditions: Trend={trend_str},{ob_status})")

        except KeyError as e:
             self.logger.error(f"{COLOR_ERROR}Error generating signal: Missing key '{e}' in strategy_state.{RESET}")
             return "HOLD" # Default to HOLD on missing data
        except Exception as e:
             self.logger.error(f"{COLOR_ERROR}Error generating signal: {e}{RESET}", exc_info=True)
             return "HOLD" # Default to HOLD on any other error

        return signal

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential Take Profit (TP) and initial Stop Loss (SL) based on entry price,
        signal, Risk Management ATR, and configuration multiples. Rounds results precisely
        to the market's minimum tick size and validates them. Uses conservative rounding.

        Args:
            entry_price: The potential or actual entry price (Decimal).
            signal: The trading signal ("BUY" or "SELL").

        Returns:
            Tuple (entry_price, take_profit, stop_loss):
                - entry_price (Decimal): The input entry price.
                - take_profit (Optional[Decimal]): Calculated TP price, rounded, or None if invalid.
                - stop_loss (Optional[Decimal]): Calculated SL price, rounded, or None if invalid.
        """
        if signal not in ["BUY", "SELL"]:
            self.logger.debug(f"Cannot calculate TP/SL: Signal is '{signal}'.")
            return entry_price, None, None

        # --- Get Inputs & Validate ---
        atr_val = self.strategy_state.get("atr_risk") # Use Risk Management ATR (already Decimal or None)
        tp_multiple = safe_decimal(self.config.get("take_profit_multiple", 1.0)) # Ensure Decimal
        sl_multiple = safe_decimal(self.config.get("stop_loss_multiple", 1.5)) # Ensure Decimal
        min_tick = self.min_tick_size # Already determined Decimal tick size

        # Validate inputs needed for calculation
        valid_inputs = True
        if not (isinstance(entry_price, Decimal) and entry_price > 0):
            self.logger.error(f"TP/SL Calc Error ({self.symbol}): Invalid entry_price ({entry_price})")
            valid_inputs = False
        if not isinstance(atr_val, Decimal):
            self.logger.error(f"TP/SL Calc Error ({self.symbol}): Risk ATR is not available or invalid ({atr_val}). State: {self.strategy_state.get('atr_risk')}")
            valid_inputs = False
        elif atr_val <= 0:
             # Allow calculation even if ATR is zero/small, but log warning
             self.logger.warning(f"{COLOR_WARNING}TP/SL Calc Warning ({self.symbol}): Risk ATR is zero or negative ({atr_val}). SL/TP offsets will be zero.{RESET}")
             atr_val = Decimal('0') # Proceed with zero offset
        if not (isinstance(tp_multiple, Decimal) and tp_multiple >= 0): # Allow 0 TP multiple
            self.logger.error(f"TP/SL Calc Error ({self.symbol}): Invalid take_profit_multiple ({tp_multiple})")
            valid_inputs = False
        if not (isinstance(sl_multiple, Decimal) and sl_multiple > 0): # SL multiple must be positive
            self.logger.error(f"TP/SL Calc Error ({self.symbol}): Invalid stop_loss_multiple ({sl_multiple})")
            valid_inputs = False
        if not (isinstance(min_tick, Decimal) and min_tick > 0):
            self.logger.error(f"TP/SL Calc Error ({self.symbol}): Invalid min_tick_size ({min_tick})")
            valid_inputs = False

        if not valid_inputs:
            self.logger.warning(f"{COLOR_WARNING}Cannot calculate TP/SL for {self.symbol} due to invalid inputs.{RESET}")
            return entry_price, None, None

        try:
            # --- Calculate Offsets ---
            tp_offset = atr_val * tp_multiple
            sl_offset = atr_val * sl_multiple
            take_profit_raw: Optional[Decimal] = None
            stop_loss_raw: Optional[Decimal] = None

            # --- Calculate Raw Prices ---
            if signal == "BUY":
                take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
            elif signal == "SELL":
                take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset

            # --- Round TP/SL to Tick Size using Conservative Rounding ---
            take_profit: Optional[Decimal] = None
            stop_loss: Optional[Decimal] = None
            price_fmt = f".{self.price_precision}f" # Format for logging

            # Round TP: Down for SELL, UP for BUY (harder to reach, more conservative profit target)
            if take_profit_raw is not None:
                tp_rounding = ROUND_DOWN if signal == "SELL" else ROUND_UP
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=tp_rounding) * min_tick
                self.logger.debug(f"TP Raw={take_profit_raw:{price_fmt}}, Rounded ({tp_rounding})={take_profit:{price_fmt}}")


            # Round SL: UP for SELL, DOWN for BUY (closer to entry, tighter stop, more conservative risk)
            if stop_loss_raw is not None:
                sl_rounding = ROUND_UP if signal == "SELL" else ROUND_DOWN
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=sl_rounding) * min_tick
                self.logger.debug(f"SL Raw={stop_loss_raw:{price_fmt}}, Rounded ({sl_rounding})={stop_loss:{price_fmt}}")


            # --- Validate Rounded Prices ---
            # Ensure SL is strictly on the losing side of entry
            if stop_loss is not None:
                if signal == "BUY" and stop_loss >= entry_price:
                    # Adjust SL down by one tick if rounding put it at or above entry
                    adjusted_sl = stop_loss - min_tick
                    self.logger.warning(f"{COLOR_WARNING}BUY SL ({stop_loss:{price_fmt}}) rounded >= entry ({entry_price:{price_fmt}}). Adjusting SL down by one tick to {adjusted_sl:{price_fmt}}.{RESET}")
                    stop_loss = adjusted_sl
                elif signal == "SELL" and stop_loss <= entry_price:
                    # Adjust SL up by one tick if rounding put it at or below entry
                    adjusted_sl = stop_loss + min_tick
                    self.logger.warning(f"{COLOR_WARNING}SELL SL ({stop_loss:{price_fmt}}) rounded <= entry ({entry_price:{price_fmt}}). Adjusting SL up by one tick to {adjusted_sl:{price_fmt}}.{RESET}")
                    stop_loss = adjusted_sl

            # Ensure TP is strictly on the winning side of entry (if TP multiple > 0)
            if take_profit is not None and tp_multiple > 0:
                if signal == "BUY" and take_profit <= entry_price:
                     # Adjust TP up by one tick if rounding put it at or below entry
                    adjusted_tp = take_profit + min_tick
                    self.logger.warning(f"{COLOR_WARNING}BUY TP ({take_profit:{price_fmt}}) rounded <= entry ({entry_price:{price_fmt}}). Adjusting TP up by one tick to {adjusted_tp:{price_fmt}}.{RESET}")
                    take_profit = adjusted_tp
                elif signal == "SELL" and take_profit >= entry_price:
                     # Adjust TP down by one tick if rounding put it at or below entry
                    adjusted_tp = take_profit - min_tick
                    self.logger.warning(f"{COLOR_WARNING}SELL TP ({take_profit:{price_fmt}}) rounded >= entry ({entry_price:{price_fmt}}). Adjusting TP down by one tick to {adjusted_tp:{price_fmt}}.{RESET}")
                    take_profit = adjusted_tp

            # Ensure final SL/TP are positive prices
            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{COLOR_ERROR}SL calculation for {signal} resulted in zero/negative price ({stop_loss:{price_fmt}}). Cannot set SL.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                # If TP multiple was 0, a zero TP might be intentional (e.g., no TP). Otherwise, it's an error.
                if tp_multiple > 0:
                     self.logger.error(f"{COLOR_ERROR}TP calculation for {signal} resulted in zero/negative price ({take_profit:{price_fmt}}). Cannot set TP.{RESET}")
                     take_profit = None
                else:
                     self.logger.info(f"TP calculation resulted in zero/negative price ({take_profit:{price_fmt}}) but TP multiple was zero. Setting TP to None.")
                     take_profit = None # Treat as no TP


            # Log final results
            tp_str = f"{take_profit:{price_fmt}}" if take_profit else "None"
            sl_str = f"{stop_loss:{price_fmt}}" if stop_loss else "None"
            self.logger.info(f"Calculated TP/SL for {self.symbol} {signal} (Risk ATR={atr_val}): "
                             f"Entry={entry_price:{price_fmt}}, TP={tp_str}, SL={sl_str}")
            return entry_price, take_profit, stop_loss

        except Exception as e:
            self.logger.error(f"{COLOR_ERROR}Unexpected error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the *available* balance for a specific currency, handling various
    account types (Contract, Unified, Spot) and response structures robustly,
    especially for Bybit V5. Uses retries internally via exchange config.

    Args:
        exchange: Initialized ccxt.Exchange object.
        currency: Currency code (e.g., "USDT").
        logger: Logger instance.

    Returns:
        Available balance as Decimal, or None on failure or if balance is zero/negative.
    """
    lg = logger
    balance_info: Optional[Dict] = None
    available_balance: Optional[Decimal] = None

    # Prioritize specific account types relevant for derivatives/spot based on exchange
    # Bybit V5 uses 'CONTRACT', 'UNIFIED', 'SPOT'. Others might use 'future', 'swap', 'trading' etc.
    account_types_to_try = []
    is_bybit = 'bybit' in exchange.id.lower()
    if is_bybit:
        # Unified covers Spot, Linear, Options. Contract for older Inverse. Spot for dedicated spot.
        account_types_to_try = ['UNIFIED', 'CONTRACT', 'SPOT']
    else:
        # Generic order for other exchanges (adjust as needed based on target exchange)
        account_types_to_try = ['swap', 'future', 'contract', 'trading', 'funding', 'spot']

    # Try fetching balance with specific account types first
    for acc_type in account_types_to_try:
        try:
            lg.debug(f"Fetching balance with params={{'accountType': '{acc_type}'}} for {currency}...")
            # Bybit V5 uses 'accountType' in params for fetchBalance
            # Other exchanges might use 'type' or require different params
            params = {}
            if is_bybit:
                params = {'accountType': acc_type}
            # else: add params for other exchanges if needed, e.g., params = {'type': acc_type}

            balance_info = exchange.fetch_balance(params=params)
            # Store the attempted type for parsing logic clarity
            if balance_info: balance_info['params_used'] = params

            # Attempt to parse the balance from this response
            parsed_balance = _parse_balance_from_response(balance_info, currency, lg)
            if parsed_balance is not None and parsed_balance > 0:
                available_balance = parsed_balance
                lg.debug(f"Found positive balance ({available_balance}) in account type '{acc_type}'.")
                break # Found a usable balance, stop searching
            elif parsed_balance is not None: # Found balance but it's zero
                 lg.debug(f"Balance found in '{acc_type}' is zero. Checking next type.")
                 balance_info = None # Reset to try next type cleanly
            else:
                 lg.debug(f"Balance for {currency} not found in '{acc_type}' account type structure. Checking next type.")
                 balance_info = None # Reset

        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            err_str = str(e).lower()
            # Ignore errors indicating the account type isn't supported/valid and try the next one
            # Specifically handle the "accountType only support UNIFIED" error from the log
            if "accounttype only support unified" in err_str or \
               "account type not support" in err_str or \
               "invalid account type" in err_str or \
               "account type invalid" in err_str or \
               "parameter error" in err_str: # Add common param errors
                lg.debug(f"Account type '{acc_type}' or params not supported/invalid for fetch_balance (Msg: {err_str}). Trying next.")
                continue
            else:
                # Log other errors but continue trying other types if possible
                lg.warning(f"Exchange/Network error fetching balance (type {acc_type}): {e}. Trying next.")
                if acc_type == account_types_to_try[-1]: # If last attempt failed with error
                     lg.error(f"Failed to fetch balance for all attempted specific account types due to errors.")
                     # Proceed to default fetch attempt below
                continue
        except Exception as e:
            lg.warning(f"Unexpected error fetching balance (type {acc_type}): {e}. Trying next.")
            continue

    # If no positive balance found with specific types, try default fetch_balance (no params)
    if available_balance is None:
        lg.debug(f"No positive balance found with specific account types. Fetching balance using default parameters for {currency}...")
        try:
            balance_info = exchange.fetch_balance()
            if balance_info: balance_info['params_used'] = {'type': 'default'} # Mark as default fetch
            parsed_balance = _parse_balance_from_response(balance_info, currency, lg)
            if parsed_balance is not None and parsed_balance > 0:
                 available_balance = parsed_balance
            elif parsed_balance is not None:
                 lg.info(f"Default balance fetch returned zero balance for {currency}.")
            else:
                 lg.warning(f"Default balance fetch did not find balance for {currency}.")

        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            lg.error(f"{COLOR_ERROR}Failed to fetch balance using default parameters: {e}{RESET}")
            # Cannot proceed if default fetch fails
            return None
        except Exception as e:
             lg.error(f"{COLOR_ERROR}Unexpected error during default balance fetch: {e}{RESET}", exc_info=True)
             return None

    # --- Final Result ---
    if available_balance is not None and available_balance > 0:
        lg.info(f"Final available {currency} balance: {available_balance:.4f}")
        return available_balance
    elif available_balance is not None and available_balance <= 0:
        lg.warning(f"{COLOR_WARNING}Available balance for {currency} is zero or negative ({available_balance:.4f}).{RESET}")
        return Decimal('0') # Return 0 explicitly for zero balance
    else:
        lg.error(f"{COLOR_ERROR}Could not determine available balance for {currency} after all attempts.{RESET}")
        # Log the last structure checked for debugging
        lg.debug(f"Last balance_info structure checked: {json.dumps(balance_info, indent=2) if balance_info else 'None'}")
        return None

def _parse_balance_from_response(balance_info: Optional[Dict], currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Helper function to parse the *available* balance from various potential structures
    within a ccxt fetch_balance response dictionary. Prioritizes 'free' or 'available' fields.
    Specifically handles Bybit V5 'info.result.list' structure.

    Args:
        balance_info: The dictionary returned by exchange.fetch_balance().
        currency: The currency code (e.g., "USDT").
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if not found or parsing fails.
        Returns Decimal('0') if balance is explicitly found to be zero.
    """
    lg = logger
    if not balance_info:
        lg.debug("_parse_balance: Input balance_info is None.")
        return None

    attempted_params = balance_info.get('params_used', {})
    lg.debug(f"_parse_balance: Attempting to parse for {currency} with params {attempted_params}. Structure keys: {list(balance_info.keys())}")

    available_balance_str: Optional[str] = None
    parse_source = "N/A"
    found_zero = False # Flag if we explicitly find a zero balance

    try:
        # --- Bybit V5 Specific Structure (Highest Priority if detected) ---
        # Bybit V5 often nests the useful data under info.result.list[]
        info_dict = balance_info.get('info', {})
        result_dict = info_dict.get('result', {})
        balance_list = result_dict.get('list')

        if isinstance(balance_list, list):
            lg.debug(f"_parse_balance: Found Bybit V5 'info.result.list' structure with {len(balance_list)} account(s).")
            target_account_type = attempted_params.get('accountType') # e.g., CONTRACT, UNIFIED, SPOT
            parsed_v5_acc_type = 'N/A'

            for account_data in balance_list:
                current_account_type = account_data.get('accountType')
                # Match if a specific type was requested OR if it was a default fetch (check all accounts)
                match_type = (target_account_type is None or current_account_type == target_account_type)

                if match_type and isinstance(account_data.get('coin'), list):
                    lg.debug(f"_parse_balance: Checking coins in V5 account type '{current_account_type}'...")
                    for coin_data in account_data['coin']:
                        if coin_data.get('coin') == currency:
                            # Priority keys for available balance in Bybit V5:
                            # 1. availableToWithdraw / availableBalance (preferred for truly free funds)
                            # 2. walletBalance (can sometimes represent available for spot/unified, less reliable for contract)
                            # 3. equity (includes PnL, not ideal for available)
                            keys_to_check = ['availableToWithdraw', 'availableBalance', 'walletBalance']
                            for key in keys_to_check:
                                free = coin_data.get(key)
                                if free is not None: # Check for existence, even if empty string
                                    available_balance_str_candidate = str(free)
                                    parsed_v5_acc_type = current_account_type or 'Unknown'
                                    parse_source_candidate = f"Bybit V5 list ['{key}'] (Account: {parsed_v5_acc_type})"
                                    lg.debug(f"_parse_balance: Found {currency} balance candidate via {parse_source_candidate}: '{available_balance_str_candidate}'")
                                    # Check if the found value is non-zero after conversion
                                    temp_dec = safe_decimal(available_balance_str_candidate)
                                    if temp_dec is not None and temp_dec > 0:
                                         lg.debug("Candidate balance is positive.")
                                         available_balance_str = available_balance_str_candidate # Accept positive balance
                                         parse_source = parse_source_candidate
                                         break # Found usable positive balance
                                    elif temp_dec is not None and temp_dec == 0 and not found_zero:
                                         lg.debug("Candidate balance is zero.")
                                         # Store the zero balance string only if we haven't found a positive one yet
                                         if available_balance_str is None:
                                             available_balance_str = available_balance_str_candidate
                                             parse_source = parse_source_candidate
                                         found_zero = True # Mark that we found an explicit zero
                                         # Continue checking other keys/accounts
                                    else: # Includes empty string converted to None by safe_decimal
                                         lg.debug(f"Candidate balance is invalid, empty, or negative (Converted: {temp_dec}).")
                                         # Do not update available_balance_str, keep checking
                            if available_balance_str is not None and safe_decimal(available_balance_str, -1) > 0: break # Exit coin loop if positive balance found
                    if available_balance_str is not None and safe_decimal(available_balance_str, -1) > 0: break # Exit account loop if positive balance found
            if available_balance_str is None and not found_zero:
                lg.debug(f"_parse_balance: {currency} not found within Bybit V5 'info.result.list[].coin[]' for requested type '{target_account_type}'.")

        # --- Standard ccxt Structure (if V5 parse failed or not V5 structure AND no positive balance found yet) ---
        if available_balance_str is None or safe_decimal(available_balance_str, -1) <= 0:
            # Only proceed if we haven't found a positive balance via V5 structure
            lg.debug(f"_parse_balance: Checking standard ccxt structures...")
            # 1. Standard ccxt 'free' balance (top-level currency dict)
            if currency in balance_info and isinstance(balance_info.get(currency), dict) and 'free' in balance_info[currency]:
                free_val = balance_info[currency]['free']
                if free_val is not None:
                    temp_str = str(free_val)
                    temp_dec = safe_decimal(temp_str)
                    if temp_dec is not None and temp_dec > 0:
                        available_balance_str = temp_str
                        parse_source = f"standard ['{currency}']['free']"
                        lg.debug(f"_parse_balance: Found positive balance via {parse_source}: {available_balance_str}")
                    elif temp_dec is not None and temp_dec == 0 and not found_zero:
                        # Store zero only if no positive balance found yet
                        if available_balance_str is None:
                             available_balance_str = temp_str
                             parse_source = f"standard ['{currency}']['free']"
                        found_zero = True
                        lg.debug(f"_parse_balance: Found zero balance via {parse_source}: {available_balance_str}")


            # 2. Alternative top-level 'free' dictionary structure (less common)
            # Check only if still no positive balance found
            elif ('free' in balance_info and isinstance(balance_info.get('free'), dict) and currency in balance_info['free']
                  and (available_balance_str is None or safe_decimal(available_balance_str, -1) <= 0)):
                 free_val = balance_info['free'][currency]
                 if free_val is not None:
                    temp_str = str(free_val)
                    temp_dec = safe_decimal(temp_str)
                    if temp_dec is not None and temp_dec > 0:
                        available_balance_str = temp_str
                        parse_source = f"top-level 'free' dict"
                        lg.debug(f"_parse_balance: Found positive balance via {parse_source}: {available_balance_str}")
                    elif temp_dec is not None and temp_dec == 0 and not found_zero:
                        # Store zero only if no positive balance found yet
                        if available_balance_str is None:
                             available_balance_str = temp_str
                             parse_source = f"top-level 'free' dict"
                        found_zero = True
                        lg.debug(f"_parse_balance: Found zero balance via {parse_source}: {available_balance_str}")


        # --- Final Conversion ---
        if available_balance_str is not None:
            final_balance = safe_decimal(available_balance_str)
            if final_balance is not None:
                # Return the balance (can be zero, handled by caller)
                lg.info(f"Parsed available balance for {currency} via {parse_source}: {final_balance}")
                return final_balance
            else:
                # Log the specific warning from the input log
                if parse_source.startswith("Bybit V5"):
                     lg.warning(f"{COLOR_WARNING}_parse_balance: Failed to convert parsed balance string '{available_balance_str}' from {parse_source} to Decimal.{RESET}")
                else:
                     lg.warning(f"{COLOR_WARNING}_parse_balance: Failed to convert final parsed balance string '{available_balance_str}' from {parse_source} to Decimal.{RESET}")
                return None # Conversion failed

        # --- Balance Not Found ---
        if found_zero:
             lg.info(f"Explicitly found zero available balance for {currency} via {parse_source}.")
             return Decimal('0') # Return zero if explicitly found

        lg.debug(f"_parse_balance: Could not find 'free' or 'available' balance field for {currency} in the response.")
        return None # Indicate available balance wasn't found

    except Exception as e:
        lg.error(f"{COLOR_ERROR}_parse_balance: Error parsing balance response: {e}{RESET}", exc_info=True)
        lg.debug(f"Balance info structure during parse error: {json.dumps(balance_info, indent=2) if balance_info else 'None'}")
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Retrieves, validates, and enhances market information for a symbol from ccxt.
    Ensures essential precision and limit info is present and usable.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT' or 'BTC/USDT').
        logger: Logger instance.

    Returns:
        Enhanced market info dictionary, or None if not found or validation fails.
        Adds 'is_contract', 'is_linear', 'is_inverse', 'contract_type'.
    """
    lg = logger
    original_symbol = symbol # Keep original for logging if simplified
    try:
        # Ensure markets are loaded. Load again if symbol not found initially.
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market '{symbol}' not found or markets possibly stale. Reloading markets...")
            try:
                # Use reload=True to force fetching latest market data
                exchange.load_markets(reload=True)
            except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as reload_err:
                 lg.error(f"Failed to reload markets while checking for '{symbol}': {reload_err}")
                 return None # Cannot proceed if reload fails

            # Check again after reload
            if symbol not in exchange.markets:
                # Try simplifying symbol (e.g., BTC/USDT:USDT -> BTC/USDT) if applicable for derivatives
                simplified_symbol = symbol.split(':')[0] if ':' in symbol else None
                if simplified_symbol and simplified_symbol != symbol and simplified_symbol in exchange.markets:
                    lg.warning(f"Original symbol '{original_symbol}' not found, but simplified '{simplified_symbol}' found. Using simplified symbol.")
                    symbol = simplified_symbol # Use the simplified symbol moving forward
                else:
                    lg.error(f"{COLOR_ERROR}Market '{original_symbol}' not found on {exchange.id} even after reloading markets. "
                             f"Check symbol format and availability. Available keys sample: {list(exchange.markets.keys())[:10]}{RESET}")
                    return None

        market = exchange.market(symbol)
        if not market:
            # This case should be rare if the symbol key exists in exchange.markets
            lg.error(f"{COLOR_ERROR}ccxt returned None for market('{symbol}') despite symbol key existing in markets dict.{RESET}")
            return None

        # --- Enhance with derived info for easier access ---
        mkt_copy = market.copy() # Work on a copy to avoid modifying cached market
        mkt_copy['is_contract'] = mkt_copy.get('contract', False) or mkt_copy.get('type') in ['swap', 'future', 'option']
        mkt_copy['is_linear'] = mkt_copy.get('linear', False)
        mkt_copy['is_inverse'] = mkt_copy.get('inverse', False)
        # Determine contract type string
        if mkt_copy['is_linear']: mkt_copy['contract_type'] = 'Linear'
        elif mkt_copy['is_inverse']: mkt_copy['contract_type'] = 'Inverse'
        elif mkt_copy['is_contract']: mkt_copy['contract_type'] = 'Contract (Unknown Type)'
        else: mkt_copy['contract_type'] = 'Spot' # Assume Spot if not contract

        # Ensure basic type info is present
        mkt_type = mkt_copy.get('type', 'unknown')

        # --- Log key details ---
        price_prec_info = mkt_copy.get('precision', {}).get('price', 'N/A')
        amount_prec_info = mkt_copy.get('precision', {}).get('amount', 'N/A')
        min_amount = mkt_copy.get('limits', {}).get('amount', {}).get('min', 'N/A')
        min_cost = mkt_copy.get('limits', {}).get('cost', {}).get('min', 'N/A')
        contract_size = mkt_copy.get('contractSize', 'N/A') # Important for contracts

        lg.debug(f"Market Info {symbol}: Type={mkt_type}, ContractType={mkt_copy['contract_type']}, IsContract={mkt_copy['is_contract']}, "
                 f"PricePrecInfo={price_prec_info}, AmtPrecInfo={amount_prec_info}, MinAmt={min_amount}, MinCost={min_cost}, ContractSize={contract_size}")

        # --- Validate essential info ---
        precision = mkt_copy.get('precision', {})
        limits = mkt_copy.get('limits', {})
        amount_limits = limits.get('amount', {})

        # 1. Price precision/tick size is crucial
        price_tick = safe_decimal(precision.get('price'))
        if price_tick is None or price_tick <= 0:
            lg.error(f"{COLOR_ERROR}Market {symbol} has invalid or missing 'precision.price' (tick size): {precision.get('price')}. Cannot proceed.{RESET}")
            return None

        # 2. Amount precision/step size is crucial
        amount_step = safe_decimal(precision.get('amount'))
        amount_prec_type = type(precision.get('amount'))
        # If it's not an integer (decimal places), it must be a valid step size > 0
        if not isinstance(precision.get('amount'), int) and (amount_step is None or amount_step <= 0):
             lg.error(f"{COLOR_ERROR}Market {symbol} has invalid or missing 'precision.amount' (step size): {precision.get('amount')} (type: {amount_prec_type}). Cannot proceed.{RESET}")
             return None
        # If it *is* an integer, it represents decimal places and must be >= 0
        elif isinstance(precision.get('amount'), int) and precision.get('amount') < 0:
             lg.error(f"{COLOR_ERROR}Market {symbol} has invalid 'precision.amount' (decimal places): {precision.get('amount')}. Cannot proceed.{RESET}")
             return None

        # 3. Minimum amount limit is needed for order validation
        min_amount_val = safe_decimal(amount_limits.get('min'))
        if min_amount_val is None:
             lg.warning(f"{COLOR_WARNING}Market {symbol} lacks valid 'limits.amount.min': {amount_limits.get('min')}. Order size validation might be incomplete. Defaulting min amount check to 0.{RESET}")
             # Store 0 in the copy for downstream checks, but be aware it's a default
             if 'limits' not in mkt_copy: mkt_copy['limits'] = {}
             if 'amount' not in mkt_copy['limits']: mkt_copy['limits']['amount'] = {}
             mkt_copy['limits']['amount']['min'] = Decimal('0')
        elif min_amount_val < 0:
             lg.error(f"{COLOR_ERROR}Market {symbol} has invalid negative 'limits.amount.min': {min_amount_val}. Cannot proceed.{RESET}")
             return None

        # 4. Minimum cost limit (optional but good to check)
        min_cost_val = safe_decimal(limits.get('cost', {}).get('min'))
        if min_cost_val is None:
             lg.debug(f"Market {symbol} lacks 'limits.cost.min'. Minimum cost checks will be skipped.")
             # Ensure it's None in the copy if missing/invalid
             if 'limits' not in mkt_copy: mkt_copy['limits'] = {}
             if 'cost' not in mkt_copy['limits']: mkt_copy['limits']['cost'] = {}
             mkt_copy['limits']['cost']['min'] = None # Explicitly None
        elif min_cost_val < 0:
             lg.warning(f"{COLOR_WARNING}Market {symbol} has invalid negative 'limits.cost.min': {min_cost_val}. Ignoring min cost limit.{RESET}")
             mkt_copy['limits']['cost']['min'] = None # Treat as None

        # 5. Contract Size (crucial for contracts)
        if mkt_copy['is_contract']:
             contract_size_val = safe_decimal(mkt_copy.get('contractSize'))
             if contract_size_val is None or contract_size_val <= 0:
                  lg.error(f"{COLOR_ERROR}Market {symbol} is a contract but has invalid or missing 'contractSize': {mkt_copy.get('contractSize')}. Cannot proceed.{RESET}")
                  return None

        lg.debug(f"Market info for {symbol} validated successfully.")
        return mkt_copy

    except ccxt.BadSymbol as e:
        lg.error(f"{COLOR_ERROR}Symbol '{original_symbol}' is invalid or not supported by {exchange.id}: {e}{RESET}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as e:
        lg.error(f"{COLOR_ERROR}API Error getting market info for {original_symbol}: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error getting market info for {original_symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # Keep as float for percentage calculation
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    analyzer: TradingAnalyzer, # Pass analyzer for rounding methods
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates position size considering balance, risk percentage, SL distance,
    contract type (linear/inverse/spot), and market constraints (min/max amount, cost, precision).
    Uses Decimal for all calculations and rounding.

    Args:
        balance: Available quote currency balance (Decimal).
        risk_per_trade: Risk percentage (e.g., 0.01 for 1%).
        initial_stop_loss_price: Initial SL price (Decimal). Must be different from entry.
        entry_price: Potential entry price (Decimal).
        market_info: Enhanced market info dictionary from get_market_info().
        analyzer: Initialized TradingAnalyzer instance (provides rounding methods).
        logger: Logger instance.

    Returns:
        Calculated and adjusted position size (Decimal) in base currency or contracts,
        rounded DOWN to the market's amount step size, or None on failure.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN')
    quote_currency = market_info.get('quote', 'QUOTE')
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_linear = market_info.get('is_linear', False)
    is_inverse = market_info.get('is_inverse', False)
    contract_type = market_info.get('contract_type', 'Unknown')

    # Determine the unit of the calculated size for logging
    size_unit = base_currency # Default for Spot and Linear Contracts
    if is_inverse:
        size_unit = "Contracts" # Inverse contracts are typically sized in contracts (e.g., USD value)

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or balance <= 0:
        lg.error(f"Pos Sizing Fail ({symbol}): Invalid or non-positive balance ({balance}).")
        return None
    if not (isinstance(risk_per_trade, (float, int)) and 0 < risk_per_trade < 1):
        lg.error(f"Pos Sizing Fail ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1 (exclusive).")
        return None
    if not isinstance(entry_price, Decimal) or entry_price <= 0:
         lg.error(f"Pos Sizing Fail ({symbol}): Invalid or non-positive entry price ({entry_price}).")
         return None
    if not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= 0:
         lg.error(f"Pos Sizing Fail ({symbol}): Invalid or non-positive SL price ({initial_stop_loss_price}).")
         return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Pos Sizing Fail ({symbol}): Stop Loss price cannot be equal to entry price.")
        return None

    # Validate required market info keys (already checked in get_market_info, but double-check)
    if 'precision' not in market_info or 'limits' not in market_info:
         lg.error(f"Pos Sizing Fail ({symbol}): Market info missing 'precision' or 'limits'. Cannot proceed.")
         return None

    try:
        # --- Core Calculation ---
        # Convert risk_per_trade to Decimal for calculation
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_price = abs(entry_price - initial_stop_loss_price)

        if sl_distance_price <= 0: # Should be caught by earlier check
            lg.error(f"Pos Sizing Fail ({symbol}): SL distance is zero or negative ({sl_distance_price}).")
            return None

        # Get contract size (use Decimal, default to 1 if missing/invalid for spot/linear)
        default_contract_size = Decimal('1')
        contract_size_raw = market_info.get('contractSize', default_contract_size) if is_contract else default_contract_size
        contract_size = safe_decimal(contract_size_raw, default_contract_size)
        if contract_size is None or contract_size <= 0:
             lg.warning(f"{COLOR_WARNING}Invalid contract size ({contract_size_raw}) for {symbol} ({contract_type}). Defaulting to {default_contract_size}.{RESET}")
             contract_size = default_contract_size

        calculated_size = Decimal('0')
        risk_per_unit_quote = Decimal('0')

        # Calculate size based on contract type
        if not is_contract or is_linear: # Spot or Linear Contract
            # Size is in Base Currency (e.g., BTC for BTC/USDT)
            # Risk per unit (of base currency) = SL distance * contract size (usually 1 for linear/spot)
            risk_per_unit_quote = sl_distance_price * contract_size
            if risk_per_unit_quote > 0:
                calculated_size = risk_amount_quote / risk_per_unit_quote
                lg.debug(f"Pos Sizing ({contract_type} {symbol}): RiskAmt={risk_amount_quote}, SLDist={sl_distance_price}, CtrSize={contract_size}, RiskPerUnit={risk_per_unit_quote}")
            else:
                 lg.error(f"Pos Sizing Fail ({contract_type} {symbol}): Risk per unit is zero or negative ({risk_per_unit_quote}). Check inputs.")
                 return None
        elif is_inverse: # Inverse Contract
            # Size is in Contracts (e.g., number of USD contracts for BTC/USD inverse)
            # Risk per Contract = (SL distance * Contract Value in Quote) / Entry Price
            # Contract Value is typically the contract_size (e.g., 1 USD)
            if entry_price > 0:
                # Risk per 1 Contract, denominated in the Quote currency
                risk_per_unit_quote = (sl_distance_price * contract_size) / entry_price
                if risk_per_unit_quote > 0:
                    calculated_size = risk_amount_quote / risk_per_unit_quote
                    lg.debug(f"Pos Sizing ({contract_type} {symbol}): RiskAmt={risk_amount_quote}, SLDist={sl_distance_price}, CtrSize={contract_size}, Entry={entry_price}, RiskPerContract={risk_per_unit_quote}")
                else:
                    lg.error(f"Pos Sizing Fail ({contract_type} {symbol}): Risk per contract is zero or negative ({risk_per_unit_quote}). Check inputs.")
                    return None
            else: # Should have been caught by input validation
                 lg.error(f"Pos Sizing Fail ({contract_type} {symbol}): Entry price is zero or negative.")
                 return None
        else: # Unknown contract type?
             lg.error(f"Pos Sizing Fail ({symbol}): Unknown contract type calculation path. Market Info: {market_info}")
             return None

        # --- Log Initial Calculation ---
        price_fmt = f".{analyzer.get_price_precision()}f"
        amount_fmt = f".{analyzer.get_amount_precision()}f"
        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote} {quote_currency}")
        lg.info(f"  Entry={entry_price:{price_fmt}}, SL={initial_stop_loss_price:{price_fmt}}, SLDist={sl_distance_price:{price_fmt}}")
        lg.info(f"  Contract Type: {contract_type}, ContractSize={contract_size}")
        lg.info(f"  Initial Calculated Size = {calculated_size} {size_unit}")

        if calculated_size <= 0:
             lg.error(f"Pos Sizing Fail ({symbol}): Initial calculation resulted in zero or negative size ({calculated_size}).")
             return None

        # --- Apply Market Constraints ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})

        # Get min/max limits (use safe_decimal with defaults, already validated in get_market_info)
        # Ensure min_amount is Decimal, default to 0 if None was stored
        min_amount = safe_decimal(amount_limits.get('min'), Decimal('0'))
        max_amount = safe_decimal(amount_limits.get('max'), Decimal('Infinity'))
        # min_cost can be None if not provided/invalid
        min_cost = safe_decimal(cost_limits.get('min')) # Returns None if invalid/missing
        max_cost = safe_decimal(cost_limits.get('max'), Decimal('Infinity'))

        adjusted_size = calculated_size

        # 1. Clamp size by Min/Max Amount Limits
        original_calc_size_before_amount_limits = adjusted_size
        if adjusted_size < min_amount:
            lg.warning(f"{COLOR_WARNING}Calculated size {adjusted_size} < min amount {min_amount}. Adjusting to min amount.{RESET}")
            adjusted_size = min_amount
        # Check against max_amount AFTER potential min_amount adjustment
        if adjusted_size > max_amount:
            lg.warning(f"{COLOR_WARNING}Size {adjusted_size} > max amount {max_amount}. Capping at max amount.{RESET}")
            adjusted_size = max_amount

        if adjusted_size != original_calc_size_before_amount_limits:
             lg.info(f"  Size after Amount Limits: {adjusted_size} {size_unit}")

        # Recalculate estimated cost based on potentially adjusted size
        current_cost = Decimal('0')
        if is_linear or not is_contract: # Cost = Size * Price * ContractSize (usually 1)
            current_cost = adjusted_size * entry_price * contract_size
        elif is_inverse: # Inverse Cost = Size (in Contracts) * ContractSize (Quote Value per contract)
             current_cost = adjusted_size * contract_size

        lg.debug(f"  Cost Check: Size={adjusted_size}, Est. Cost={current_cost} {quote_currency}, MinCost={min_cost}, MaxCost={max_cost}")

        # 2. Check and Adjust by Min/Max Cost Limits (if they exist and are valid)
        cost_adjusted = False
        original_size_before_cost_limits = adjusted_size

        # Adjust for Min Cost (only if min_cost is valid > 0)
        if min_cost is not None and min_cost > 0 and current_cost < min_cost:
            lg.warning(f"{COLOR_WARNING}Est. cost {current_cost} < min cost {min_cost}. Attempting to increase size.{RESET}")
            required_size_for_min_cost = Decimal('0')
            try:
                if is_linear or not is_contract:
                     denom = entry_price * contract_size
                     if denom <= 0: raise ZeroDivisionError("Entry price or contract size is zero/negative")
                     required_size_for_min_cost = min_cost / denom
                elif is_inverse:
                     if contract_size <= 0: raise ZeroDivisionError("Contract size is zero/negative")
                     required_size_for_min_cost = min_cost / contract_size

                if required_size_for_min_cost <= 0: raise ValueError("Calculated required size is non-positive")
            except (ValueError, InvalidOperation, ZeroDivisionError) as calc_err:
                 lg.error(f"{COLOR_ERROR}Cannot calculate required size for min cost due to error: {calc_err}. Aborting size calculation.{RESET}")
                 return None

            lg.info(f"  Required size for min cost: {required_size_for_min_cost} {size_unit}")

            # Check if required size exceeds max amount or max cost
            if required_size_for_min_cost > max_amount:
                lg.error(f"{COLOR_ERROR}Cannot meet min cost ({min_cost}): Required size {required_size_for_min_cost} exceeds max amount limit ({max_amount}). Aborted.{RESET}")
                return None
            # Calculate cost of the required size
            required_cost = Decimal('0')
            if is_linear or not is_contract: required_cost = required_size_for_min_cost * entry_price * contract_size
            else: required_cost = required_size_for_min_cost * contract_size
            if max_cost is not None and required_cost > max_cost:
                 lg.error(f"{COLOR_ERROR}Cannot meet min cost ({min_cost}): Required size {required_size_for_min_cost} results in cost ({required_cost}) exceeding max cost limit ({max_cost}). Aborted.{RESET}")
                 return None

            # Use the size required for min cost
            adjusted_size = required_size_for_min_cost
            cost_adjusted = True

        # Adjust for Max Cost (only if max_cost is valid and min cost didn't apply/fail)
        elif max_cost is not None and current_cost > max_cost:
            lg.warning(f"{COLOR_WARNING}Est. cost {current_cost} > max cost {max_cost}. Reducing size.{RESET}")
            allowed_size_for_max_cost = Decimal('0')
            try:
                if is_linear or not is_contract:
                    denom = entry_price * contract_size
                    if denom <= 0: raise ZeroDivisionError("Entry price or contract size is zero/negative")
                    allowed_size_for_max_cost = max_cost / denom
                elif is_inverse:
                    if contract_size <= 0: raise ZeroDivisionError("Contract size is zero/negative")
                    allowed_size_for_max_cost = max_cost / contract_size

                if allowed_size_for_max_cost <= 0: raise ValueError("Calculated allowed size is non-positive")
            except (ValueError, InvalidOperation, ZeroDivisionError) as calc_err:
                 lg.error(f"{COLOR_ERROR}Cannot calculate allowed size for max cost due to error: {calc_err}. Aborting size calculation.{RESET}")
                 return None

            lg.info(f"  Allowed size for max cost: {allowed_size_for_max_cost} {size_unit}")

            # Check if allowed size is below min amount
            if allowed_size_for_max_cost < min_amount:
                lg.error(f"{COLOR_ERROR}Cannot meet max cost ({max_cost}): Allowed size {allowed_size_for_max_cost} is below min amount limit ({min_amount}). Aborted.{RESET}")
                return None

            # Use the size allowed by max cost
            adjusted_size = allowed_size_for_max_cost
            cost_adjusted = True

        if cost_adjusted:
            lg.info(f"  Size after Cost Limits: {adjusted_size} {size_unit}")


        # 3. Apply Amount Precision/Step Size (Truncate/Floor using Analyzer's method)
        final_size = analyzer.round_amount(adjusted_size)

        if final_size is None:
            lg.error(f"{COLOR_ERROR}Failed to apply amount precision/step size to {adjusted_size}. Aborting.{RESET}")
            return None

        if final_size != adjusted_size: # Log only if rounding changed the value
             lg.info(f"  Size after Amount Precision (Rounded Down): {final_size} {size_unit}")


        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"{COLOR_ERROR}Position size became zero or negative ({final_size}) after precision/limit adjustments. Aborted.{RESET}")
            return None

        # Final check against min amount (use a small tolerance for potential floating point issues during intermediate steps if they occurred)
        # Tolerance should be very small, smaller than the step size itself.
        # Since we rounded down, it should ideally be >= min_amount.
        tolerance = analyzer.get_amount_step_size() / Decimal('100') # Tiny tolerance
        if final_size < min_amount - tolerance:
            lg.error(f"{COLOR_ERROR}Final size {final_size} after precision is still less than min amount {min_amount}. Aborted.{RESET}")
            return None

        # Final check against min cost if applicable (allow tolerance)
        if min_cost is not None and min_cost > 0:
            final_cost = Decimal('0')
            if is_linear or not is_contract: final_cost = final_size * entry_price * contract_size
            elif is_inverse: final_cost = final_size * contract_size

            # Use cost tolerance (e.g., a fraction of min_cost or absolute small value)
            cost_tolerance = min_cost * Decimal('0.0001') # 0.01% tolerance
            if final_cost < min_cost - cost_tolerance:
                 lg.error(f"{COLOR_ERROR}Final cost {final_cost} after precision is still less than min cost {min_cost}. Aborted.{RESET}")
                 return None

        lg.info(f"{COLOR_SUCCESS}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e:
        lg.error(f"{COLOR_ERROR}Pos Sizing Error ({symbol}): Missing key in market_info: {e}. Check market data validity. Info: {market_info}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Fetches and validates the *single* open position for a specific symbol,
    handling different ccxt methods and response structures (especially Bybit V5).
    Enhances the position dictionary with standardized 'side', SL/TP prices (as Decimals), and TSL info.
    Assumes One-Way mode on the exchange.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        logger: Logger instance.

    Returns:
        Enhanced position dictionary if an active position exists (size != 0),
        or None if no active position is found or an error occurs.
    """
    lg = logger
    position: Optional[Dict] = None
    # Use a small threshold based on typical minimum contract sizes or precision
    # A size smaller than this is considered effectively zero. Adjust if needed.
    size_threshold = Decimal('1e-9')

    try:
        lg.debug(f"Fetching position for symbol: {symbol}")
        params = {}
        market = None
        # Add Bybit V5 category parameter based on market info
        if 'bybit' in exchange.id.lower():
            try:
                market = exchange.market(symbol) # Assume loaded
                if market:
                    category = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse', False) else 'spot'
                    params['category'] = category
                    lg.debug(f"Using params for fetch_position(s): {params}")
                else:
                     lg.warning(f"Market info not found for {symbol} during position check. Assuming 'linear' category.")
                     params['category'] = 'linear'
            except (KeyError, ccxt.BadSymbol) as e:
                 lg.warning(f"Error getting market info for {symbol} during position check ({e}). Assuming 'linear' category.")
                 params['category'] = 'linear'

        # --- Attempt fetching position ---
        fetched_positions_data: List[Dict] = []

        # 1. Try fetch_position (singular) first if supported (more efficient)
        if exchange.has.get('fetchPosition'):
             try:
                 lg.debug(f"Attempting exchange.fetch_position('{symbol}', params={params})")
                 single_pos_data = exchange.fetch_position(symbol, params=params)
                 # fetch_position might return a dict even if no position (size=0)
                 if single_pos_data:
                     fetched_positions_data = [single_pos_data]
                     lg.debug("fetch_position returned data. Will validate size.")
                 else:
                      lg.debug("fetch_position returned None or empty dict.")

             except ccxt.NotSupported as e:
                  lg.debug(f"fetch_position not supported by exchange or for this symbol/params: {e}. Falling back.")
             except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                 # Specific codes/messages indicating "no position"
                 no_pos_codes = ['110025'] # Bybit V5: position idx not exist / position does not exist
                 no_pos_msgs = ['position does not exist', 'no position found', 'position idx not exist', 'position not found']
                 err_str = str(e).lower()
                 err_code = getattr(e, 'code', None)
                 if str(err_code) in no_pos_codes or any(msg in err_str for msg in no_pos_msgs):
                      lg.info(f"No position found for {symbol} via fetch_position (Code/Msg: {e}).")
                      # This is expected, not an error state.
                 else:
                      # Log other errors but fallback to fetch_positions
                      lg.warning(f"fetch_position failed unexpectedly for {symbol}: {e}. Falling back to fetch_positions.")
             except Exception as e:
                  lg.warning(f"Unexpected error in fetch_position for {symbol}: {e}. Falling back to fetch_positions.")

        # 2. Fallback to fetch_positions (plural) if fetchPosition failed, not supported, or returned no active pos data
        if not fetched_positions_data and exchange.has.get('fetchPositions'):
            try:
                 # Try fetching positions for the specific symbol first
                 lg.debug(f"Attempting exchange.fetch_positions(symbols=['{symbol}'], params={params})")
                 # fetch_positions usually returns a list, potentially empty
                 fetched_positions_data = exchange.fetch_positions(symbols=[symbol], params=params)
                 lg.debug(f"fetch_positions returned {len(fetched_positions_data)} entries for {symbol}.")
            except ccxt.ArgumentsRequired:
                 # If exchange requires fetching all positions (less efficient)
                 lg.debug(f"Fetching all positions as exchange requires it...")
                 all_positions = exchange.fetch_positions(params=params) # Pass category here too if needed
                 # Filter for the specific symbol
                 fetched_positions_data = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Filtered {len(fetched_positions_data)} positions for {symbol} from all positions.")
            except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                 # Handle errors indicating no position found within fetch_positions response
                 no_pos_msgs = ['no position found', 'position does not exist', 'position idx not exist']
                 err_str = str(e).lower()
                 if any(msg in err_str for msg in no_pos_msgs):
                     lg.info(f"No position found for {symbol} via fetch_positions (Exchange message: {err_str}).")
                     fetched_positions_data = [] # Ensure list is empty
                 else:
                     lg.error(f"Exchange/Network error during fetch_positions for {symbol}: {e}", exc_info=True)
                     return None # Treat other errors as failure
            except Exception as e:
                 lg.error(f"Unexpected error fetching positions for {symbol}: {e}", exc_info=True)
                 return None

        # --- Find and Validate the Active Position from the fetched list ---
        active_position_raw: Optional[Dict] = None
        if not fetched_positions_data:
            lg.info(f"No position data structures returned by API for {symbol}.")
        else:
            for pos_data in fetched_positions_data:
                # Consolidate size fetching from various possible keys
                size_val = pos_data.get('contracts') # Standard ccxt v1
                if size_val is None: size_val = pos_data.get('contractSize') # Alt standard v1 (less common for pos size)
                info_dict_temp = pos_data.get('info', {})
                if size_val is None: size_val = info_dict_temp.get('size') # Bybit V5 info size
                if size_val is None: size_val = info_dict_temp.get('positionAmt') # Binance info size
                if size_val is None: size_val = info_dict_temp.get('contracts') # Check info dict too
                if size_val is None: size_val = pos_data.get('amount') # Another possible key

                if size_val is None:
                    lg.debug(f"Skipping position entry, missing/null size field: {pos_data}")
                    continue

                position_size = safe_decimal(size_val)
                # Check if size is valid Decimal and significantly different from zero
                if position_size is not None and abs(position_size) > size_threshold:
                    if active_position_raw is not None:
                         lg.warning(f"{COLOR_WARNING}Multiple active position entries found for {symbol}. Using the first one with size {position_size}. Check exchange mode (Hedge vs One-Way) and API response.{RESET}")
                         # If Hedge mode is possible, logic might need adjustment here
                    else:
                        active_position_raw = pos_data.copy() # Work on a copy
                        lg.debug(f"Found candidate active position entry: Size={position_size}")
                        # Assuming One-Way mode, break after finding the first active position
                        break
                else:
                    lg.debug(f"Position entry found but size ({position_size}) is zero or invalid. Skipping.")


        # --- Process and Enhance the Found Active Position ---
        if active_position_raw:
            position = active_position_raw # Use the found raw position
            info_dict = position.get('info', {}) # Standardized info dict

            # --- Standardize Side ('long' or 'short') ---
            side = position.get('side') # Standard ccxt side
            # Get size again reliably for side inference
            size_decimal = safe_decimal(position.get('contracts', info_dict.get('size', '0')), Decimal('0'))

            if side not in ['long', 'short']:
                info_side = info_dict.get('side', '').lower() # Bybit: 'Buy'/'Sell'/'None'
                if info_side == 'buy': side = 'long'
                elif info_side == 'sell': side = 'short'
                elif size_decimal > size_threshold: side = 'long' # Infer from positive size
                elif size_decimal < -size_threshold: side = 'short' # Infer from negative size
                else: side = None # Cannot determine side

            if side is None:
                 lg.warning(f"Position found for {symbol}, but size ({size_decimal}) is near zero or side undetermined. Treating as no active position.")
                 return None

            position['side'] = side # Store standardized side

            # --- Populate Standard CCXT Fields from Info Dict if Missing ---
            # Ensures consistency regardless of which fetch method worked
            def populate_from_info(pos_dict, info_dict, standard_key, info_keys):
                """Helper to populate standard key from potential info keys."""
                if pos_dict.get(standard_key) is None:
                    for info_key in info_keys:
                        if info_dict.get(info_key) is not None:
                            pos_dict[standard_key] = info_dict[info_key]
                            break # Found one

            populate_from_info(position, info_dict, 'entryPrice', ['entryPrice', 'avgPrice'])
            populate_from_info(position, info_dict, 'markPrice', ['markPrice'])
            populate_from_info(position, info_dict, 'liquidationPrice', ['liqPrice'])
            populate_from_info(position, info_dict, 'unrealizedPnl', ['unrealisedPnl', 'unrealizedPnl'])
            populate_from_info(position, info_dict, 'collateral', ['positionIM', 'collateral'])
            populate_from_info(position, info_dict, 'leverage', ['leverage'])
            populate_from_info(position, info_dict, 'contracts', ['size', 'contracts']) # Ensure size is in standard field
            populate_from_info(position, info_dict, 'symbol', ['symbol']) # Ensure symbol is present

            # --- Enhance with SL/TP/TSL from 'info' (Focus on Bybit V5 structure) ---
            # Use market info for accurate price precision if available
            price_prec = 4 # Default
            if market:
                 tick = safe_decimal(market.get('precision', {}).get('price'))
                 if tick and tick > 0: price_prec = abs(tick.normalize().as_tuple().exponent)

            def get_valid_price_from_info(key: str) -> Optional[Decimal]:
                """Safely gets and validates a price field from info dict, returns Decimal."""
                val_str = info_dict.get(key)
                # Treat '0' or '0.0' etc. string as None (meaning not set)
                if isinstance(val_str, str) and safe_decimal(val_str) == Decimal('0'):
                    return None
                val_dec = safe_decimal(val_str)
                return val_dec if val_dec and val_dec > 0 else None

            # Parse SL/TP (Bybit V5: 'stopLoss', 'takeProfit')
            sl_price = get_valid_price_from_info('stopLoss')
            tp_price = get_valid_price_from_info('takeProfit')
            position['stopLossPrice'] = sl_price # Add Decimal or None
            position['takeProfitPrice'] = tp_price # Add Decimal or None

            # Parse TSL (Bybit V5: 'trailingStop' is distance/rate string, 'activePrice' is string trigger)
            tsl_value_str = info_dict.get('trailingStop', '0') # Is a string like "0", "50", "0.005"
            tsl_value_dec = safe_decimal(tsl_value_str)
            # TSL is considered active if the distance value is valid and greater than zero
            tsl_active = tsl_value_dec is not None and tsl_value_dec > 0

            tsl_activation_price = get_valid_price_from_info('activePrice') # Price level for activation

            position['trailingStopLossValue'] = tsl_value_dec if tsl_active else None # Store Decimal distance or None
            position['trailingStopLossActive'] = tsl_active # Boolean flag
            position['trailingStopLossActivationPrice'] = tsl_activation_price # Store Decimal activation price or None

            # --- Final Conversion of key fields to Decimal ---
            # Ensures downstream logic uses Decimals consistently
            decimal_keys = ['entryPrice', 'markPrice', 'liquidationPrice', 'unrealizedPnl', 'collateral', 'leverage', 'contracts']
            for key in decimal_keys:
                 position[key] = safe_decimal(position.get(key)) # Convert to Decimal, becomes None if invalid

            # Log enhanced position details
            price_fmt = f".{price_prec}f"
            details = {
                'Symbol': position.get('symbol', symbol),
                'Side': side.upper(),
                'Size': f"{position.get('contracts')}" if position.get('contracts') else 'N/A',
                'Entry': f"{position.get('entryPrice'):{price_fmt}}" if position.get('entryPrice') else 'N/A',
                'Mark': f"{position.get('markPrice'):{price_fmt}}" if position.get('markPrice') else 'N/A',
                'Liq': f"{position.get('liquidationPrice'):{price_fmt}}" if position.get('liquidationPrice') else 'N/A',
                'uPnL': f"{position.get('unrealizedPnl'):.4f}" if position.get('unrealizedPnl') else 'N/A',
                'Coll': f"{position.get('collateral'):.4f}" if position.get('collateral') else 'N/A',
                'Lev': f"{position.get('leverage'):.1f}x" if position.get('leverage') else 'N/A',
                'TP': f"{tp_price:{price_fmt}}" if tp_price else 'None',
                'SL': f"{sl_price:{price_fmt}}" if sl_price else 'None',
                'TSL': (f"Active (Val={tsl_value_dec}, ActAt={tsl_activation_price:{price_fmt}})"
                        if tsl_active and tsl_activation_price else
                        f"Active (Val={tsl_value_dec})" if tsl_active else 'None'),
            }
            details_str = ', '.join(f"{k}={v}" for k, v in details.items())
            lg.info(f"{COLOR_SUCCESS}Active Position Found: {details_str}{RESET}")
            return position # Return the enhanced dictionary
        else:
            lg.info(f"No active position found for {symbol} after checking API results.")
            return None

    except ccxt.AuthenticationError as e:
        lg.error(f"{COLOR_ERROR}Authentication error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Avoid logging expected "no position" errors again
        no_pos_codes = ['110025']
        no_pos_msgs = ['position does not exist', 'no position found', 'position idx not exist']
        err_str = str(e).lower()
        err_code = getattr(e, 'code', None)
        if not (str(err_code) in no_pos_codes or any(msg in err_str for msg in no_pos_msgs)):
             lg.error(f"{COLOR_ERROR}Exchange error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error checking positions for {symbol}: {e}{RESET}", exc_info=True)

    return None # Return None if any error occurred or no active position found


def place_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str, # 'buy' or 'sell'
    amount: Decimal, # Base currency or contracts (positive Decimal)
    market_info: Dict,
    analyzer: TradingAnalyzer, # Pass analyzer for rounding
    logger: logging.Logger,
    params: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Places a market order with safety checks, correct parameters (incl. Bybit V5),
    amount rounding, and robust error handling.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        side: 'buy' or 'sell'.
        amount: Order size (positive Decimal). Unit depends on market.
        market_info: Enhanced market info dictionary.
        analyzer: Initialized TradingAnalyzer instance (provides rounding).
        logger: Logger instance.
        params: Additional parameters for create_order (e.g., {'reduceOnly': True}).

    Returns:
        The order dictionary from ccxt if successful, or None if failed or trading disabled.
    """
    lg = logger
    if not CONFIG.get("enable_trading", False):
        lg.warning(f"{COLOR_WARNING}TRADING DISABLED. Skipping market {side} order for {amount} {symbol}.{RESET}")
        return None

    if not isinstance(amount, Decimal) or amount <= 0:
        lg.error(f"{COLOR_ERROR}Invalid amount for market order: {amount}. Must be a positive Decimal.{RESET}")
        return None

    # --- Round Amount to Market Precision/Step Size ---
    rounded_amount = analyzer.round_amount(amount)
    if rounded_amount is None or rounded_amount <= 0:
         lg.error(f"{COLOR_ERROR}Amount {amount} became invalid ({rounded_amount}) after rounding for {symbol}. Cannot place order.{RESET}")
         return None
    if rounded_amount != amount:
         size_unit = market_info.get('base', '') if market_info.get('contract_type', 'Spot') == 'Spot' or market_info.get('is_linear') else "Contracts"
         lg.info(f"Market order amount rounded from {amount} to {rounded_amount} {size_unit} for {symbol} precision.")
         amount = rounded_amount # Use the rounded amount

    # --- Prepare Order Parameters ---
    order_params = params.copy() if params else {}
    is_reduce_only = order_params.get('reduceOnly', False)

    # --- Add Exchange Specific Params (Bybit V5 Example) ---
    is_bybit = 'bybit' in exchange.id.lower()
    if is_bybit:
        # Category (Linear/Inverse/Spot)
        if 'category' not in order_params:
            category = 'linear' if market_info.get('is_linear', False) \
                  else 'inverse' if market_info.get('is_inverse', False) \
                  else 'spot' # Default to spot if not linear/inverse
            order_params['category'] = category
            lg.debug(f"Setting Bybit category='{category}' for market order.")

        # Position Mode (Assume One-way if not specified) - Crucial for Bybit V5 derivatives
        # Only relevant for contract types
        if market_info.get('is_contract') and 'positionIdx' not in order_params:
            # 0=One-Way, 1=Buy Hedge, 2=Sell Hedge. This bot assumes One-Way.
            order_params['positionIdx'] = 0
            lg.debug(f"Setting default Bybit positionIdx=0 (One-way mode) for market order.")

        # For market orders, timeInForce is usually implicit, but can be set if needed
        # order_params['timeInForce'] = 'ImmediateOrCancel' # Or 'FillOrKill'

    # Determine order description for logging
    size_unit = market_info.get('base', '') if market_info.get('contract_type', 'Spot') == 'Spot' or market_info.get('is_linear') else "Contracts"
    order_desc = f"{side.upper()} {amount} {size_unit} {symbol} MARKET"
    if is_reduce_only: order_desc += " [REDUCE_ONLY]"
    lg.info(f"Attempting to place order: {order_desc} with params: {order_params}")

    try:
        # Convert Decimal amount to float for ccxt create_order function if required by library version
        # Note: Newer ccxt versions might handle Decimal directly, but float is safer fallback.
        try:
            amount_for_api = float(amount)
        except Exception:
            lg.error(f"Could not convert amount {amount} to float for API call.")
            return None

        # --- Place Order ---
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=amount_for_api,
            params=order_params
        )

        # --- Post-Order Handling ---
        # Market orders fill quickly, but response structure varies.
        if order and isinstance(order, dict):
            order_id = order.get('id', f'simulated_market_fill_{int(time.time())}')
            order_status = order.get('status', 'unknown').lower()

            if not order.get('id'):
                 lg.warning(f"Market order response lacked ID for {order_desc}, using simulated ID: {order_id}")
                 order['id'] = order_id # Add to dict if missing
            if not order.get('status'):
                 # Assume 'closed' (filled) for market orders if status missing, common on some exchanges
                 lg.warning(f"Market order response lacked status for {order_desc} (ID: {order_id}), assuming 'closed' (filled).")
                 order_status = 'closed'
                 order['status'] = order_status # Add to dict if missing

            # Check status (should be 'closed' or 'filled')
            if order_status in ['closed', 'filled']:
                 lg.info(f"{COLOR_SUCCESS}Successfully placed and likely filled market order: {order_desc}. Order ID: {order_id}{RESET}")
            else:
                 # This shouldn't happen often for market orders, but log if it does
                 lg.warning(f"{COLOR_WARNING}Market order placed ({order_desc}, ID: {order_id}), but status is '{order_status}'. Manual check advised.{RESET}")

            lg.debug(f"Market order API response details: {order}")
            return order
        else:
            lg.error(f"{COLOR_ERROR}Market order placement for {order_desc} did not return a valid order dictionary. Response: {order}{RESET}")
            return None


    # --- Error Handling ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{COLOR_ERROR}Insufficient funds for {order_desc}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        # Common reasons: Size too small/large, cost too small, precision issue, reduceOnly mismatch
        err_str = str(e).lower()
        if "order size is invalid" in err_str or "size" in err_str or "amount" in err_str or "precision" in err_str:
             lg.error(f"{COLOR_ERROR}Invalid market order size/precision for {order_desc}: {e}. Check limits/rounding.{RESET}")
        elif "order cost" in err_str or "value" in err_str or "minimum" in err_str or "minnotional" in err_str:
             lg.error(f"{COLOR_ERROR}Market order cost/value below minimum for {order_desc}: {e}. Check minCost/minNotional limit.{RESET}")
        elif "reduce-only" in err_str or "reduce only" in err_str:
             lg.error(f"{COLOR_ERROR}Reduce-only market order conflict for {order_desc}: {e}. Position size mismatch or order would increase position?{RESET}")
        else:
             lg.error(f"{COLOR_ERROR}Invalid market order {order_desc}: {e}. Check parameters and exchange limits.{RESET}")
    except ccxt.ExchangeError as e:
        # Handle other specific exchange errors if known (e.g., leverage issues, risk limits)
        err_str = str(e).lower()
        if "margin check failed" in err_str or "insufficient margin" in err_str:
             lg.error(f"{COLOR_ERROR}Margin check failed placing {order_desc}: {e}. Check available balance and leverage.{RESET}")
        elif "risk limit" in err_str:
             lg.error(f"{COLOR_ERROR}Risk limit exceeded placing {order_desc}: {e}. Check exchange position/risk settings.{RESET}")
        else:
             lg.error(f"{COLOR_ERROR}Exchange error placing {order_desc}: {e}{RESET}", exc_info=True)
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error placing {order_desc}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error placing {order_desc}: {e}{RESET}", exc_info=True)

    return None # Return None if order placement failed


def set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    analyzer: TradingAnalyzer, # Pass analyzer for rounding/precision
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_params: Optional[Dict] = None, # e.g., {'trailingStop': '50', 'activePrice': '...'} String values for Bybit!
    current_position: Optional[Dict] = None, # Pass pre-fetched enhanced position to avoid redundant API call
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Sets or modifies Stop Loss (SL), Take Profit (TP), and Trailing Stop Loss (TSL)
    for an existing position using the appropriate ccxt method or specific API endpoint (e.g., Bybit V5 /v5/position/trading-stop).
    Only sends API request if changes are detected compared to the current position state.
    Handles rounding and parameter formatting (especially strings for Bybit V5).

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: Trading symbol.
        market_info: Enhanced market info dict.
        analyzer: Initialized TradingAnalyzer instance.
        stop_loss_price: Desired SL price (Decimal). Use 0 or None to remove or leave unchanged.
        take_profit_price: Desired TP price (Decimal). Use 0 or None to remove or leave unchanged.
        trailing_stop_params: Dict with TSL parameters specific to the exchange API.
                              For Bybit V5: {'trailingStop': 'distance_str', 'activePrice': 'price_str' (optional)}
                              Use {'trailingStop': '0'} to remove TSL. None leaves it unchanged.
        current_position: Optional pre-fetched and enhanced position dictionary. If None, it will be fetched.
        logger: Logger instance.

    Returns:
        True if protection was set/modified successfully or no change was needed, False otherwise.
    """
    lg = logger or logging.getLogger(__name__)
    if not CONFIG.get("enable_trading", False):
        lg.warning(f"{COLOR_WARNING}TRADING DISABLED. Skipping protection setting for {symbol}.{RESET}")
        return False # Cannot set protection if trading off

    # --- 1. Get Current Position Info (Fetch if not provided) ---
    position = current_position
    if position is None:
        lg.debug(f"Fetching position for {symbol} before setting protection...")
        # Use enhanced fetcher which returns Decimal prices/None and TSL info
        position = get_open_position(exchange, symbol, lg)
        if position is None:
            lg.info(f"No open position found for {symbol}. Cannot set protection (This might be expected after closing).")
            return True # No action needed if no position

    # --- 2. Validate Position State ---
    position_side = position.get('side') # 'long' or 'short'
    position_size = position.get('contracts') # Already Decimal or None from get_open_position
    size_threshold = Decimal('1e-9')

    if not position_side or not isinstance(position_size, Decimal) or abs(position_size) <= size_threshold:
        lg.info(f"Position side/size invalid or near zero for {symbol}. No protection action needed. Pos Info: {position}")
        return True # No active position to protect

    # --- 3. Determine Current Protection State from Enhanced Position Data ---
    # Prices/values are already Decimals or None from get_open_position
    current_sl: Optional[Decimal] = position.get('stopLossPrice')
    current_tp: Optional[Decimal] = position.get('takeProfitPrice')
    current_tsl_val: Optional[Decimal] = position.get('trailingStopLossValue') # The distance/rate value
    current_tsl_active: bool = position.get('trailingStopLossActive', False)
    current_tsl_act_price: Optional[Decimal] = position.get('trailingStopLossActivationPrice')

    lg.debug(f"Current Protection State ({symbol}): SL={current_sl}, TP={current_tp}, "
             f"TSL Active={current_tsl_active} (Val={current_tsl_val}, ActPrice={current_tsl_act_price})")

    # --- 4. Prepare API Parameters and Detect Changes ---
    params = {} # Parameters for the API call
    log_parts = [] # For logging the changes being made
    needs_api_call = False

    # Get precision/tick size for formatting prices and rounding
    price_prec = analyzer.get_price_precision()
    min_tick = analyzer.get_min_tick_size()
    price_fmt = f".{price_prec}f"

    # --- Exchange/API Specific Setup (Bybit V5 Example) ---
    is_bybit = 'bybit' in exchange.id.lower()
    # Check if the specific endpoint exists (more reliable than version string)
    has_trading_stop_endpoint = exchange.has.get('privatePostPositionTradingStop', False) if is_bybit else False

    if is_bybit and has_trading_stop_endpoint:
        # Category needed for v5 position endpoints
        params['category'] = 'linear' if market_info.get('is_linear', True) else 'inverse' if market_info.get('is_inverse', False) else 'spot'
        # Position Index (0 for One-Way, 1/2 for Hedge) - Get from fetched position if possible
        params['positionIdx'] = position.get('info', {}).get('positionIdx', 0) # Default to 0 (One-Way)
        # Bybit V5 /v5/position/trading-stop endpoint requires trigger prices and tpslMode.
        # Use Mark Price trigger as a common default, adjust if needed.
        params['tpTriggerBy'] = 'MarkPrice'
        params['slTriggerBy'] = 'MarkPrice'
        # tpslMode ('Full' or 'Partial') determined later based on which params are set.
        lg.debug(f"Using Bybit V5 /v5/position/trading-stop params: category={params['category']}, positionIdx={params['positionIdx']}")
    elif is_bybit:
         lg.warning("Bybit exchange detected, but /v5/position/trading-stop endpoint seems unavailable in capabilities. Protection setting might fail or use older methods.")
         # Add logic for older Bybit methods if necessary
    # Add setup for other exchanges here if needed


    # --- Process Stop Loss ---
    sl_change_detected = False
    target_sl_str = "0" # Default API value for SL (means no SL)
    if stop_loss_price is not None: # If a new SL value is provided (could be 0 to remove)
        if stop_loss_price <= 0:
            # Request to remove SL
            if current_sl is not None: # Only change if SL currently exists
                sl_change_detected = True
                log_parts.append("SL=Remove")
                target_sl_str = "0"
            # else: No SL exists, and request is to remove -> no change
        else:
            # Request to set/change SL
            # Round the target SL price CONSERVATIVELY (closer to entry)
            sl_rounding = ROUND_UP if position_side == "SELL" else ROUND_DOWN
            rounded_sl = (stop_loss_price / min_tick).quantize(Decimal('1'), rounding=sl_rounding) * min_tick
            if rounded_sl <= 0:
                 lg.error(f"Target SL price {stop_loss_price} resulted in zero/negative rounded SL ({rounded_sl}). Cannot set.")
                 return False # Invalid target SL after rounding

            # Compare rounded target SL with current SL (handle None)
            if rounded_sl != current_sl:
                sl_change_detected = True
                target_sl_str = f"{rounded_sl:{price_fmt}}" # Format as string for API
                log_parts.append(f"SL={target_sl_str}")
            # else: Rounded SL is the same as current SL -> no change

    if sl_change_detected:
        needs_api_call = True
        if is_bybit and has_trading_stop_endpoint: params['stopLoss'] = target_sl_str
        # Add logic for other exchange params if needed
    elif is_bybit and has_trading_stop_endpoint and current_sl is not None:
         # If not changing SL, but other params might change, need to resubmit current SL value for Bybit V5
         params['stopLoss'] = f"{current_sl:{price_fmt}}"
    elif is_bybit and has_trading_stop_endpoint and 'stopLoss' not in params:
         params['stopLoss'] = '0' # Ensure '0' is sent if no SL exists and none requested

    # --- Process Take Profit ---
    tp_change_detected = False
    target_tp_str = "0" # Default API value for TP (means no TP)
    if take_profit_price is not None: # If a new TP value is provided (could be 0 to remove)
        if take_profit_price <= 0:
            # Request to remove TP
            if current_tp is not None: # Only change if TP currently exists
                tp_change_detected = True
                log_parts.append("TP=Remove")
                target_tp_str = "0"
            # else: No TP exists, and request is to remove -> no change
        else:
            # Request to set/change TP
            # Round the target TP price CONSERVATIVELY (further from entry)
            tp_rounding = ROUND_DOWN if position_side == "SELL" else ROUND_UP
            rounded_tp = (take_profit_price / min_tick).quantize(Decimal('1'), rounding=tp_rounding) * min_tick
            if rounded_tp <= 0:
                 lg.error(f"Target TP price {take_profit_price} resulted in zero/negative rounded TP ({rounded_tp}). Cannot set.")
                 return False # Invalid target TP after rounding

            # Compare rounded target TP with current TP (handle None)
            if rounded_tp != current_tp:
                tp_change_detected = True
                target_tp_str = f"{rounded_tp:{price_fmt}}" # Format as string for API
                log_parts.append(f"TP={target_tp_str}")
            # else: Rounded TP is the same as current TP -> no change

    if tp_change_detected:
        needs_api_call = True
        if is_bybit and has_trading_stop_endpoint: params['takeProfit'] = target_tp_str
        # Add logic for other exchanges
    elif is_bybit and has_trading_stop_endpoint and current_tp is not None:
         # Resubmit current TP if needed for Bybit V5 API call
         params['takeProfit'] = f"{current_tp:{price_fmt}}"
    elif is_bybit and has_trading_stop_endpoint and 'takeProfit' not in params:
         params['takeProfit'] = '0' # Ensure '0' is sent if no TP exists and none requested


    # --- Process Trailing Stop ---
    tsl_change_detected = False
    target_tsl_params = {} # Store params specifically for TSL part of API call
    if trailing_stop_params is not None: # If new TSL instructions provided
        # Extract target values (expect strings for Bybit V5)
        target_tsl_dist_str = trailing_stop_params.get('trailingStop') # e.g., "50", "0.005", "0"
        target_tsl_act_str = trailing_stop_params.get('activePrice')   # e.g., "20000.5" or None

        target_tsl_dist_dec = safe_decimal(target_tsl_dist_str) # Convert target distance for comparison

        # Validate target distance string format before using Decimal
        if target_tsl_dist_str is None or target_tsl_dist_dec is None:
             lg.error(f"Invalid trailingStop value provided: '{target_tsl_dist_str}'. Must be a valid number string. Cannot process TSL.")
             # Don't set needs_api_call = True based on invalid input
        else:
            # Check if TSL distance needs changing
            update_tsl_dist = False
            if target_tsl_dist_dec <= 0: # Request to remove TSL
                if current_tsl_active: # Only change if TSL currently active
                    update_tsl_dist = True
                    target_tsl_dist_str = "0" # Ensure removal value is '0' string
                    log_parts.append("TSL=Remove")
                # else: TSL not active, request remove -> no change
            elif target_tsl_dist_dec > 0: # Request to set/change TSL distance
                # Compare Decimal distance values
                if target_tsl_dist_dec != current_tsl_val:
                    update_tsl_dist = True
                    # Validate string format for distance (percentage or price) - basic check
                    if not re.match(r"^\d+(\.\d+)?%?$", target_tsl_dist_str):
                         lg.error(f"Invalid format for trailingStop distance '{target_tsl_dist_str}'. Should be number or percentage string.")
                         update_tsl_dist = False # Do not proceed with invalid format
                    else:
                         log_parts.append(f"TSL_Dist={target_tsl_dist_str}")
                # else: TSL distance is the same -> no change needed for distance

            if update_tsl_dist:
                tsl_change_detected = True
                target_tsl_params['trailingStop'] = target_tsl_dist_str # Use validated string

                # Handle Activation Price only if TSL distance > 0
                if target_tsl_dist_dec > 0:
                    target_tsl_act_dec = safe_decimal(target_tsl_act_str) # Convert target activation for comparison

                    if target_tsl_act_dec is not None and target_tsl_act_dec > 0:
                        # Round activation price
                        act_rounding = ROUND_DOWN if position_side == "SELL" else ROUND_UP # Activate slightly later (conservative)
                        rounded_tsl_act = (target_tsl_act_dec / min_tick).quantize(Decimal('1'), rounding=act_rounding) * min_tick

                        if rounded_tsl_act != current_tsl_act_price:
                            act_price_str = f"{rounded_tsl_act:{price_fmt}}"
                            target_tsl_params['activePrice'] = act_price_str
                            log_parts.append(f"TSL_ActAt={act_price_str}")
                        # else: Activation price is the same -> no change
                    elif target_tsl_act_str is None and current_tsl_act_price is not None:
                         # Request to remove activation price while keeping/setting TSL distance
                         # Bybit V5: Omitting activePrice might reset it, or send "0"? Docs unclear. Omitting is likely safer.
                         target_tsl_params.pop('activePrice', None) # Ensure not sent
                         log_parts.append("TSL_ActAt=Remove")
                         # Note: We need to track if *only* the activation price changed
                         if not update_tsl_dist: # If distance didn't change but activation did
                              tsl_change_detected = True # Need API call even if distance is same
                    # else: No new activation price provided, and none exists -> no change
                else: # Removing TSL (distance <= 0), ensure activePrice is not sent
                     target_tsl_params.pop('activePrice', None)
                     # No need to log removal if TSL itself is being removed

            # else: No change needed for TSL distance

    if tsl_change_detected:
        needs_api_call = True
        if is_bybit and has_trading_stop_endpoint:
            params.update(target_tsl_params) # Add TSL specific params
        # Add logic for other exchanges
    elif is_bybit and has_trading_stop_endpoint and current_tsl_active:
         # Resubmit current TSL if needed for Bybit V5 API call when other params change
         params['trailingStop'] = str(current_tsl_val) # Use current value as string
         if current_tsl_act_price:
              params['activePrice'] = f"{current_tsl_act_price:{price_fmt}}"
    elif is_bybit and has_trading_stop_endpoint and 'trailingStop' not in params:
         params['trailingStop'] = '0' # Ensure '0' is sent if no TSL exists and none requested


    # --- 5. Make API Call Only If Changes Detected ---
    if not needs_api_call:
        lg.info(f"No changes detected for position protection ({symbol}). No API call needed.")
        return True

    # --- Finalize Bybit V5 Parameters ---
    if is_bybit and has_trading_stop_endpoint:
        # Ensure required fields are present with '0' string if not explicitly set/changed
        params.setdefault('stopLoss', '0')
        params.setdefault('takeProfit', '0')
        params.setdefault('trailingStop', '0')
        # activePrice should only be present if explicitly set

        # Determine tpslMode ('Full' or 'Partial') - Crucial for Bybit V5 /v5/position/trading-stop
        # Use 'Partial' if setting TSL, or if setting BOTH TP and SL simultaneously.
        # Use 'Full' if setting ONLY TP or ONLY SL.
        has_tp = safe_decimal(params.get('takeProfit', '0'), 0) > 0
        has_sl = safe_decimal(params.get('stopLoss', '0'), 0) > 0
        has_tsl = safe_decimal(params.get('trailingStop', '0'), 0) > 0

        if has_tsl:
             # If TSL is active or being activated, mode must be Partial
             params['tpslMode'] = 'Partial'
        elif has_tp and has_sl:
             # Setting both TP and SL requires Partial mode
             params['tpslMode'] = 'Partial'
        elif has_tp or has_sl:
             # Setting only TP or only SL uses Full mode
             params['tpslMode'] = 'Full'
        else:
             # Only removing TP/SL/TSL (all target values are '0')
             # Use Partial mode for removal as well, seems safer/more consistent
             params['tpslMode'] = 'Partial'
             lg.debug("Setting tpslMode to Partial as only removing protection or no protection set.")

        lg.debug(f"Final Bybit V5 protection params: {params}")


    # --- Log and Execute ---
    set_desc = f"Set Protection for {symbol} ({position_side.upper()} {position_size}): {', '.join(log_parts)}"
    lg.info(f"Attempting: {set_desc}")

    try:
        response = None
        # --- Select Appropriate API Method ---
        if is_bybit and has_trading_stop_endpoint:
            lg.debug("Using Bybit V5 private_post /v5/position/trading-stop")
            # Ensure all price/value parameters are strings
            for k in ['stopLoss', 'takeProfit', 'trailingStop', 'activePrice']:
                if k in params and not isinstance(params[k], str):
                     lg.warning(f"Parameter '{k}' was not string ({params[k]}), converting.")
                     params[k] = str(params[k])
            response = exchange.private_post('/v5/position/trading-stop', params) # Use direct endpoint call

        # --- Add elif blocks here for other exchanges' specific methods ---
        # elif exchange.id == 'binance' and exchange.has...:
        #     # Binance uses PUT /fapi/v1/positionSide/dual for hedge mode TP/SL
        #     # or POST /fapi/v1/order for separate TP/SL orders linked to position
        #     lg.error("Binance protection setting not implemented.")
        #     return False

        elif exchange.has.get('editPosition'): # Generic check (might work for some, but parameters vary wildly)
             lg.warning("Attempting generic exchange.editPosition (Experimental/Unreliable)")
             # Need to translate params to what editPosition expects (highly variable)
             # This part requires specific implementation per exchange if not Bybit V5
             # Example (likely incorrect): edit_params = {'stopLoss': {...}, 'takeProfit': {...}}
             # response = exchange.edit_position(symbol, params=edit_params)
             lg.error("Generic editPosition logic not implemented. Cannot set protection.")
             return False # Indicate failure until implemented
        else:
            # Check for separate set_stop_loss, set_take_profit methods if needed
            lg.error(f"Protection setting (TP/SL/TSL) via a single unified method not available or implemented for exchange {exchange.id}. Cannot proceed.")
            return False

        # --- Process Response ---
        lg.info(f"{COLOR_SUCCESS}Protection setting API request sent successfully for {symbol}.{RESET}")
        lg.debug(f"Protection setting API response: {response}")

        # --- Response Validation (Bybit V5 Example) ---
        if is_bybit and isinstance(response, dict):
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', '')
            if ret_code == 0:
                lg.debug("API response indicates success (retCode 0).")
                return True # Success
            else:
                # Check for non-critical "errors" that mean success or no change needed
                # Codes based on Bybit V5 documentation and common scenarios:
                # 110043: Set TP/SL orders can only be modified or cancelled (Might mean already set as requested)
                # 34036:  The order is not modified (No change detected by exchange)
                # 110025: position idx not match position mode (Warning if mode seems correct)
                # 110068: The trailing stop loss is not modified
                # 110067: The take profit/stop loss is not modified
                # 110090: tp/sl order maybe executed or cancelled (Order already closed/triggered?)
                # 110017: Request parameter error (Check if this sometimes means "no change")
                # 110072: Trailing stop order is not activated (If trying to modify inactive TSL?)
                non_error_codes = [0, 110043, 34036, 110025, 110067, 110068, 110090, 110072]
                # Messages indicating no real failure
                no_change_msgs = ["not modified", "same tpsl", "order is not modified", "no need to modify", "already closed", "already cancelled", "order not exists or too late to cancel"]
                err_str = ret_msg.lower()

                if ret_code in non_error_codes or any(msg in err_str for msg in no_change_msgs):
                    lg.warning(f"{COLOR_WARNING}Protection setting for {symbol} - Non-critical/Informational response code {ret_code}: '{ret_msg}'. Assuming success or no change required.{RESET}")
                    return True # Treat as success/no action needed
                else:
                    lg.error(f"{COLOR_ERROR}Protection setting failed ({symbol}). API Code: {ret_code}, Msg: {ret_msg}{RESET}")
                    return False
        elif response is not None: # If response exists but format is unknown
            lg.warning("Protection setting response received but format unknown or not dictionary, assuming success based on lack of exception.")
            return True
        else: # No response received (should have been caught by exception)
             lg.error("Protection setting call did not return a response.")
             return False


    # --- Error Handling for API Call ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{COLOR_ERROR}Insufficient funds during protection setting for {symbol}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{COLOR_ERROR}Invalid order parameters setting protection for {symbol}: {e}. Check values relative to price/position/liquidation.{RESET}")
    except ccxt.ExchangeError as e:
        # Handle already logged non-critical errors again just in case exception mapping catches them
        err_code = getattr(e, 'code', None)
        err_str = str(e).lower()
        non_error_codes = [110043, 34036, 110025, 110067, 110068, 110090, 110072]
        no_change_msgs = ["not modified", "same tpsl", "order is not modified", "no need to modify", "already closed", "already cancelled", "order not exists or too late to cancel"]
        if str(err_code) in non_error_codes or any(msg in err_str for msg in no_change_msgs):
             lg.warning(f"{COLOR_WARNING}Protection setting for {symbol} - Caught non-critical error {err_code}: '{err_str}'. Assuming success/no change.{RESET}")
             return True # Treat as success
        lg.error(f"{COLOR_ERROR}Exchange error setting protection for {symbol}: {e}{RESET}", exc_info=True)
    except ccxt.NetworkError as e:
        lg.error(f"{COLOR_ERROR}Network error setting protection for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{COLOR_ERROR}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)

    return False # Return False if any exception occurred


# --- Main Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Performs one full cycle of analysis and potential trading action for a single symbol.
    Fetches data, analyzes, checks position, generates signals, manages risk, and executes trades.

    Args:
        exchange: Initialized ccxt.Exchange object.
        symbol: The trading symbol to analyze (e.g., 'BTC/USDT').
        config: The global configuration dictionary.
        logger: The logger instance specific to this symbol.
    """
    lg = logger
    lg.info(f"---== Analyzing {symbol} ==---")
    cycle_start_time = time.monotonic()

    try:
        # --- 1. Fetch Market Info (Crucial for precision, limits) ---
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info:
            lg.error(f"Skipping cycle for {symbol}: Could not retrieve valid market info.")
            return # Cannot proceed without market info

        # --- 2. Fetch Kline Data ---
        interval_str = config.get("interval", "5") # User interval (e.g., "5")
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(interval_str)
        if not ccxt_timeframe:
            lg.error(f"Invalid interval '{interval_str}' in config for {symbol}. Skipping cycle.")
            return

        # Determine appropriate lookback limit based on indicator needs
        min_required_data = 250 # Fallback
        try:
             # Calculate min required length based on config dynamically
             buffer = 100 # Add ample buffer for initial NaNs and stability
             min_len_volbot = 0
             if config.get("volbot_enabled", True):
                 min_len_volbot = max(
                     config.get("volbot_length", DEFAULT_VOLBOT_LENGTH) + 3, # SWMA
                     config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
                     config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
                     config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H) + config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H) + 1,
                     config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L) + config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L) + 1
                 )
             min_len_risk = config.get("atr_period", DEFAULT_ATR_PERIOD)
             # Ensure minimum is at least the buffer size
             min_required_data = max(min_len_volbot, min_len_risk, 0) + buffer
             lg.debug(f"Calculated min required klines: {min_required_data} (Volbot: {min_len_volbot}, Risk: {min_len_risk}, Buffer: {buffer})")
        except Exception as e:
             lg.error(f"Error calculating min required data length: {e}. Using fallback {min_required_data}.")

        kline_limit = min_required_data
        lg.debug(f"Fetching {kline_limit} klines for {symbol} {ccxt_timeframe}...")
        df_klines = fetch_klines_ccxt(exchange, symbol, ccxt_timeframe, limit=kline_limit, logger=lg)

        # Validate fetched klines
        min_acceptable_klines = 50 # Need a reasonable minimum for *any* analysis
        if df_klines.empty or len(df_klines) < min_acceptable_klines:
            lg.warning(f"{COLOR_WARNING}Insufficient kline data for {symbol} (got {len(df_klines)}, needed ~{min_required_data}, min acceptable {min_acceptable_klines}). Skipping analysis cycle.{RESET}")
            return

        # --- 3. Initialize Analyzer & Calculate Indicators ---
        analyzer = TradingAnalyzer(df=df_klines, logger=lg, config=config, market_info=market_info)
        if analyzer.df_processed.empty or not analyzer.strategy_state:
            lg.error(f"Failed to calculate indicators or update state for {symbol}. Skipping trading logic.")
            return # Cannot proceed without analysis results

        # --- 4. Check Current Position ---
        # Use the enhanced get_open_position which returns None if no active position
        current_position = get_open_position(exchange, symbol, lg) # Returns enhanced dict or None
        has_open_position = current_position is not None
        position_side = current_position.get('side') if has_open_position else None # 'long' or 'short'
        # Use Decimal entry price and size directly from enhanced position dict
        position_entry_price = current_position.get('entryPrice') if has_open_position else None # Decimal or None
        position_size = current_position.get('contracts') if has_open_position else Decimal('0') # Decimal or 0

        # --- 5. Fetch Current Market Price ---
        current_price = fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price is None:
            # Fallback to last close price from klines if ticker fails
            last_close = analyzer.strategy_state.get('close') # Already Decimal or None from state
            if isinstance(last_close, Decimal) and last_close > 0:
                current_price = last_close
                lg.warning(f"{COLOR_WARNING}Ticker price fetch failed for {symbol}. Using last kline close ({current_price}) for checks. Signal/execution might be based on slightly stale data.{RESET}")
            else:
                lg.error(f"{COLOR_ERROR}Failed to get current price for {symbol} from ticker and no valid last close price available. Cannot proceed with trading logic.{RESET}")
                return

        price_fmt = f".{analyzer.get_price_precision()}f" # Use analyzer's precision for logging

        # --- 6. Generate Trading Signal ---
        signal = analyzer.generate_trading_signal() # Returns "BUY", "SELL", or "HOLD"

        # --- 7. Position Management & Trading Logic ---
        trading_enabled = config.get("enable_trading", False)

        # =============================
        # == Scenario 1: IN a Position ==
        # =============================
        # Ensure position_size is Decimal and entry price exists
        if has_open_position and isinstance(position_size, Decimal) and isinstance(position_entry_price, Decimal) and abs(position_size) > Decimal('1e-9'):
            pos_size_fmt = f".{analyzer.get_amount_precision()}f"
            lg.info(f"Managing existing {position_side.upper()} position ({position_size:{pos_size_fmt}} {symbol} @ {position_entry_price:{price_fmt}}). Current Price: {current_price:{price_fmt}}")

            # --- 7a. Check for Exit Signal ---
            # Exit if signal flips against the current position direction
            exit_signal_triggered = (position_side == 'long' and signal == "SELL") or \
                                    (position_side == 'short' and signal == "BUY")

            if exit_signal_triggered:
                reason = f"Opposing signal ({signal}) received while in {position_side.upper()} position"
                color = COLOR_DN if signal == "SELL" else COLOR_UP
                lg.info(f"{color}Exit Signal Triggered: {reason}. Attempting to close position.{RESET}")

                if trading_enabled:
                    close_side = 'sell' if position_side == 'long' else 'buy'
                    # Place market order to close, using reduceOnly flag
                    # Pass the analyzer instance for rounding inside place_market_order
                    close_order = place_market_order(
                        exchange, symbol, close_side, abs(position_size),
                        market_info, analyzer, lg, params={'reduceOnly': True}
                    )
                    if close_order:
                        lg.info(f"{COLOR_SUCCESS}Market order placed to close {position_side} position.{RESET}")
                        # Attempt to cancel associated SL/TP orders (best effort)
                        # Bybit V5 uses set_position_protection with SL/TP=0 to cancel existing orders
                        lg.info(f"Attempting to cancel any existing SL/TP/TSL orders for {symbol}...")
                        time.sleep(1) # Brief pause before cancelling stops
                        try:
                             cancel_success = set_position_protection(
                                 exchange, symbol, market_info, analyzer,
                                 stop_loss_price=Decimal(0), # Signal to remove SL
                                 take_profit_price=Decimal(0), # Signal to remove TP
                                 trailing_stop_params={'trailingStop': '0'}, # Signal to remove TSL
                                 current_position=None, # Force refetch to confirm closure if needed by API
                                 logger=lg
                             )
                             if cancel_success: lg.info(f"Cancellation request for SL/TP/TSL sent successfully.")
                             else: lg.warning(f"{COLOR_WARNING}Could not confirm cancellation of SL/TP/TSL orders for {symbol} after closing.{RESET}")
                        except Exception as cancel_err:
                            lg.warning(f"{COLOR_WARNING}Error attempting to cancel stop orders for {symbol}: {cancel_err}{RESET}")
                    else:
                        lg.error(f"{COLOR_ERROR}Failed to place market order to close {position_side} position. MANUAL INTERVENTION MAY BE REQUIRED!{RESET}")
                    # End cycle for this symbol after attempting closure
                    return
                else:
                    lg.warning(f"{COLOR_WARNING}TRADING DISABLED: Would have closed {position_side} position ({reason}).{RESET}")
                    # Even if trading disabled, stop further management for this cycle if exit triggered
                    return

            # --- 7b. Risk Management (Only if NO exit signal) ---
            else:
                lg.info(f"No exit signal. Performing risk management checks for {position_side} position...")
                # Store potential updates, only call API if something changes
                new_sl_price: Optional[Decimal] = None # Potential new SL (e.g., from BE)
                new_tsl_params: Optional[Dict] = None # Potential new TSL settings (e.g., activation)
                needs_protection_update = False

                # Get current protection state directly from the enhanced position object
                current_sl = current_position.get('stopLossPrice') # Decimal or None
                current_tp = current_position.get('takeProfitPrice') # Decimal or None
                current_tsl_active = current_position.get('trailingStopLossActive', False)
                current_tsl_val = current_position.get('trailingStopLossValue') # Decimal distance or None

                # --- i. Break-Even Logic ---
                enable_be = config.get("enable_break_even", True)
                # Optional: Consider disabling BE if TSL is already active (avoids conflicting SL updates)
                disable_be_if_tsl_active = True # Example: configuration option?
                run_be_check = enable_be and not (disable_be_if_tsl_active and current_tsl_active)

                if run_be_check:
                    risk_atr = analyzer.strategy_state.get("atr_risk") # Risk ATR (already Decimal or None)
                    min_tick = analyzer.get_min_tick_size() # Decimal

                    if isinstance(risk_atr, Decimal) and risk_atr > 0 and min_tick > 0:
                        be_trigger_multiple = safe_decimal(config.get("break_even_trigger_atr_multiple", 1.0), Decimal(1.0))
                        be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                        profit_target_for_be = risk_atr * be_trigger_multiple # Profit needed in price points

                        # Calculate current profit in price points
                        current_profit_points = (current_price - position_entry_price) if position_side == 'long' else (position_entry_price - current_price)

                        lg.debug(f"BE Check: CurrentProfitPts={current_profit_points:{price_fmt}}, TargetProfitPts={profit_target_for_be:{price_fmt}}, ATR={risk_atr}")

                        # Check if profit target reached
                        if current_profit_points >= profit_target_for_be:
                            # Calculate BE price (entry + offset)
                            offset_amount = min_tick * be_offset_ticks
                            be_price_raw = position_entry_price + offset_amount if position_side == 'long' else position_entry_price - offset_amount
                            # Round BE price away from entry to cover costs/slippage
                            be_rounding = ROUND_UP if position_side == 'long' else ROUND_DOWN
                            be_price_rounded = (be_price_raw / min_tick).quantize(Decimal('1'), rounding=be_rounding) * min_tick

                            # Check if this BE SL is actually better than the current SL (or if no SL exists)
                            is_be_sl_better = False
                            if current_sl is None:
                                is_be_sl_better = True # No current SL, so BE is better
                            elif position_side == 'long' and be_price_rounded > current_sl:
                                is_be_sl_better = True
                            elif position_side == 'short' and be_price_rounded < current_sl:
                                is_be_sl_better = True

                            if is_be_sl_better:
                                lg.info(f"{COLOR_SUCCESS}Break-Even Triggered! Profit {current_profit_points:{price_fmt}} >= Target {profit_target_for_be:{price_fmt}}.{RESET}")
                                lg.info(f"  Moving SL from {current_sl:{price_fmt} if current_sl else 'None'} to BE price: {be_price_rounded:{price_fmt}}")
                                new_sl_price = be_price_rounded # Store the potential new SL price
                                needs_protection_update = True
                            else:
                                lg.debug(f"BE triggered, but proposed BE SL {be_price_rounded:{price_fmt}} is not better than current SL {current_sl:{price_fmt} if current_sl else 'None'}. No change.")
                        # else: Profit target not reached for BE
                    elif enable_be: # Log if BE enabled but inputs missing
                        lg.warning(f"Cannot calculate BE for {symbol}: Invalid Risk ATR ({risk_atr}) or Min Tick ({min_tick}).")
                elif enable_be and not run_be_check:
                     lg.debug("BE check skipped because TSL is active.")


                # --- ii. Trailing Stop Activation / Management ---
                # This logic focuses on *activating* the TSL if it's enabled but not yet active.
                # Exchange handles the trailing once activated via set_position_protection.
                enable_tsl = config.get("enable_trailing_stop", True)
                if enable_tsl and not current_tsl_active:
                    # Get TSL parameters from config
                    # Ensure callback rate is string for Bybit API
                    tsl_callback_rate_str = str(config.get("trailing_stop_callback_rate", "0.005"))
                    tsl_activation_perc = safe_decimal(config.get("trailing_stop_activation_percentage", "0.003"), Decimal('0'))

                    # Validate callback rate format (simple check)
                    if not re.match(r"^\d+(\.\d+)?%?$", tsl_callback_rate_str) or safe_decimal(tsl_callback_rate_str.replace('%','')) <= 0:
                         lg.error(f"Invalid TSL callback rate format or value in config: '{tsl_callback_rate_str}'. Cannot activate TSL.")
                    else:
                        activate_tsl = False
                        if tsl_activation_perc <= 0:
                             # Activate immediately if percentage is zero (should have been set on entry)
                             lg.warning(f"{COLOR_WARNING}TSL enabled with <=0 activation threshold, but not active. Attempting to set TSL now.{RESET}")
                             activate_tsl = True
                        else:
                            # Calculate current profit percentage
                            current_profit_perc = Decimal('0')
                            if position_entry_price > 0:
                                current_profit_perc = (current_price / position_entry_price) - 1 if position_side == 'long' else 1 - (current_price / position_entry_price)

                            lg.debug(f"TSL Activation Check: CurrentProfit%={current_profit_perc:.4%}, ActivationThreshold%={tsl_activation_perc:.4%}")
                            if current_profit_perc >= tsl_activation_perc:
                                lg.info(f"{COLOR_SUCCESS}Trailing Stop activation profit threshold reached ({current_profit_perc:.2%} >= {tsl_activation_perc:.2%}).{RESET}")
                                activate_tsl = True

                        # If activation conditions met, prepare TSL params
                        if activate_tsl:
                            lg.info(f"Preparing to activate TSL with distance/rate: {tsl_callback_rate_str}")
                            # For Bybit V5, setting 'trailingStop' with a value > 0 activates it.
                            # 'activePrice' can be used to set a specific trigger price, otherwise Bybit uses its own logic.
                            # We will only include activePrice if the activation percentage > 0.
                            new_tsl_params = {'trailingStop': tsl_callback_rate_str}
                            if tsl_activation_perc > 0:
                                # Calculate activation price based on entry and percentage
                                act_price_raw = position_entry_price * (1 + tsl_activation_perc) if position_side == 'long' else position_entry_price * (1 - tsl_activation_perc)
                                # Round activation price conservatively (activate slightly later)
                                act_rounding = ROUND_UP if position_side == 'long' else ROUND_DOWN
                                rounded_act_price = (act_price_raw / min_tick).quantize(Decimal('1'), rounding=act_rounding) * min_tick
                                if rounded_act_price > 0:
                                     act_price_str = f"{rounded_act_price:{price_fmt}}"
                                     new_tsl_params['activePrice'] = act_price_str
                                     lg.info(f"  Calculated TSL Activation Price: {act_price_str}")
                                else:
                                     lg.warning("Calculated TSL activation price is zero or negative. Omitting activePrice param.")


                            needs_protection_update = True
                            # Note: If BE also triggered, both SL and TSL will be
