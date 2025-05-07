# File: constants.py
import os
from colorama import Fore, Style
from decimal import Decimal # Import Decimal for default periods if needed

# Initialize colorama
Style.RESET_ALL # Ensure RESET_ALL is accessible if used directly
# Note: init(autoreset=True) is called in main script

# --- Neon Color Scheme ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- File/Directory Names ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"

# --- API Retry Settings ---
MAX_API_RETRIES = 3 # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5 # Delay between retries
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable

# --- Time Settings ---
LOOP_DELAY_SECONDS = 10 # Time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY_SECONDS = 8 # Wait time after placing order before confirming position
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default timezone string

# --- Trading Parameters & Defaults ---
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Intervals supported by the bot's logic
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Default periods (can be overridden by config.json)
DEFAULT_INDICATOR_PERIODS = {
    "atr_period": 14,
    "cci_window": 20,
    "williams_r_window": 14,
    "mfi_window": 14,
    "stoch_rsi_window": 14, # Window for Stoch RSI calculation itself
    "stoch_rsi_rsi_window": 12, # Window for underlying RSI in StochRSI
    "stoch_rsi_k": 3, # K period for StochRSI
    "stoch_rsi_d": 3, # D period for StochRSI
    "rsi_period": 14,
    "bollinger_bands_period": 20,
    "bollinger_bands_std_dev": 2.0, # Ensure float for potential use
    "sma_10_window": 10,
    "ema_short_period": 9,
    "ema_long_period": 21,
    "momentum_period": 7,
    "volume_ma_period": 15,
    "fibonacci_window": 50,
    "psar_af": 0.02,
    "psar_max_af": 0.2,
}

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels

# --- Environment Variable Keys ---
# These are fetched in main script or where needed, just listing keys for reference
# API_KEY_ENV = "BYBIT_API_KEY"
# API_SECRET_ENV = "BYBIT_API_SECRET"
# TIMEZONE_ENV = "TIMEZONE"

# Ensure log directory exists (can be called early)
os.makedirs(LOG_DIRECTORY, exist_ok=True)

```

```python
