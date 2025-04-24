# Enhanced version focusing on stop-loss/take-profit mechanisms,
# including break-even logic, MA cross exit condition, correct Bybit V5 API usage,
# multi-symbol support, and improved state management.

import argparse
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List, Union

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# Initialize colorama and set precision
getcontext().prec = 28  # Increased precision for financial calculations
init(autoreset=True)
load_dotenv()

# Neon Color Scheme
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Timezone for logging and display (adjust as needed)
TIMEZONE = ZoneInfo("America/Chicago") # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3 # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5 # Delay between retries
# Intervals supported by the bot's logic. Note: Only certain intervals might have enough liquidity for scalping/intraday.
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable

# Default periods (can be overridden by config.json)
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14 # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 14     # Window for underlying RSI in StochRSI (often same as Stoch RSI window)
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0 # Ensure float
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels
LOOP_DELAY_SECONDS = 15 # Time between the end of one cycle and the start of the next loop for *all* symbols
POSITION_CONFIRM_DELAY = 10 # Seconds to wait after placing order before checking position status
# QUOTE_CURRENCY dynamically loaded from config

# Global dictionary to hold state per symbol (e.g., break_even_triggered)
symbol_states: Dict[str, Dict[str, Any]] = {}


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present."""
    default_config = {
        "symbols": ["BTC/USDT:USDT"], # List of symbols to trade (CCXT format)
        "interval": "5", # Default to 5 minute interval (string format for our logic)
        "loop_delay": LOOP_DELAY_SECONDS, # Delay between full bot cycles (processing all symbols)
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "cci_window": DEFAULT_CCI_WINDOW,
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW,
        "mfi_window": DEFAULT_MFI_WINDOW,
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW,
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF,
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "orderbook_limit": 25, # Depth of orderbook to fetch
        "signal_score_threshold": 1.5, # Score needed to trigger BUY/SELL signal for 'default' weight set
        "scalping_signal_threshold": 2.5, # Separate threshold for 'scalping' weight set
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5, # How much higher volume needs to be than MA
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": False, # SAFETY FIRST: Default to False, enable consciously
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet), disable consciously
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions_total": 1, # Global limit on open positions across all symbols
        "quote_currency": "USDT", # Currency for balance check and sizing
        "position_mode": "One-Way", # "One-Way" or "Hedge" - CRITICAL: Match exchange setting. Bot assumes One-Way by default.
        # --- MA Cross Exit Config ---
        "enable_ma_cross_exit": True, # Enable closing position on adverse EMA cross
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True, # Default to enabling TSL (exchange TSL)
        # Trail distance as a percentage of the entry price (e.g., 0.5%)
        # Bybit API expects absolute distance; this percentage is used for calculation.
        "trailing_stop_callback_rate": 0.005, # Example: 0.5% trail distance relative to entry price
        # Activate TSL when price moves this percentage in profit from entry (e.g., 0.3%)
        # Set to 0 for immediate TSL activation upon entry (API value '0').
        "trailing_stop_activation_percentage": 0.003, # Example: Activate when 0.3% in profit
        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        # Move SL when profit (in price points) reaches X * Current ATR
        "break_even_trigger_atr_multiple": 1.0, # Example: Trigger BE when profit = 1x ATR
        # Place BE SL this many minimum price increments (ticks) beyond entry price
        # E.g., 2 ticks to cover potential commission/slippage on exit
        "break_even_offset_ticks": 2,
         # Set to True to always cancel TSL and revert to Fixed SL when BE is triggered.
         # Set to False to allow TSL to potentially remain active if it's better than BE price.
         # Note: Bybit V5 often automatically cancels TSL when a fixed SL is set, so True is generally safer.
        "break_even_force_fixed_sl": True,
        # --- End Protection Config ---
        "indicators": { # Control which indicators are calculated and contribute to score
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True, # Flag to enable fetching and scoring orderbook data
        },
        "weight_sets": { # Define different weighting strategies
            "scalping": { # Example weighting for a fast scalping strategy
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # A more balanced weighting strategy
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default" # Choose which weight set to use ("default" or "scalping")
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            return default_config # Return default even if creation fails

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added during the update
            if updated_config != config_from_file:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                    print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET}")
                except IOError as e:
                    print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
        # Attempt to create default if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
            print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Type mismatch handling could be added here if needed
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
LOOP_DELAY_SECONDS = int(CONFIG.get("loop_delay", 15)) # Use configured loop delay, default 15
console_log_level = logging.INFO # Default console level, can be changed via args

# --- Logger Setup ---
# Global logger dictionary to manage loggers per symbol
loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Sets up a logger with file and console handlers.
       If is_symbol_logger is True, uses symbol-specific naming and file.
       Otherwise, uses a generic name (e.g., 'main')."""

    if name in loggers:
        # If logger already exists, update console level if needed
        logger = loggers[name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

    # Clean name for filename if it's a symbol
    safe_name = name.replace('/', '_').replace(':', '-') if is_symbol_logger else name
    logger_instance_name = f"livebot_{safe_name}" # Unique internal name for the logger
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_instance_name}.log")
    logger = logging.getLogger(logger_instance_name)

    # Set base logging level to DEBUG to capture everything
    logger.setLevel(logging.DEBUG)

    # File Handler (writes DEBUG and above, includes line numbers)
    try:
        # Ensure the log directory exists
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Add line number to file logs for easier debugging
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger for {log_filename}: {e}{RESET}")


    # Stream Handler (console)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S' # Add timestamp format to console
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console level based on global variable (which can be set by args)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids duplicate outputs)
    logger.propagate = False

    loggers[name] = logger # Store logger instance
    return logger

def get_logger(name: str, is_symbol_logger: bool = False) -> logging.Logger:
    """Retrieves or creates a logger instance."""
    if name not in loggers:
        return setup_logger(name, is_symbol_logger)
    else:
        # Ensure existing logger's console level matches current setting
        logger = loggers[name]
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_log_level)
        return logger

# --- Trading Logic Helper Functions (Including missing ones) ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency, handling Bybit V5 structures."""
    lg = logger
    try:
        balance_info = None
        # For Bybit V5, specify the account type relevant to the market (e.g., CONTRACT, UNIFIED)
        # Try CONTRACT first, then UNIFIED, then SPOT (for spot trading pairs), then default.
        account_types_to_try = ['CONTRACT', 'UNIFIED', 'SPOT'] # Added SPOT for spot symbols

        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency}...")
                params = {'accountType': acc_type}
                # Bybit V5 balance endpoint can filter by coin
                if currency:
                     params['coin'] = currency
                     lg.debug(f"Also adding coin filter: {currency}")

                balance_info = exchange.fetch_balance(params=params)
                # Store the attempted param for later parsing logic
                if balance_info: # Ensure balance_info is not None
                    balance_info['params'] = params # Store params used for this fetch

                # Validate if the currency is present in the fetched data for this account type
                # CCXT standard structure: balance_info[currency]['free']
                if balance_info and currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('free') is not None:
                    lg.debug(f"Found standard ccxt balance for '{currency}' via accountType '{acc_type}'.")
                    break # Found balance info with free balance directly

                # Bybit V5 nested structure check: info -> result -> list -> coin[]
                elif balance_info and 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                     balance_list = balance_info['info']['result']['list']
                     found_in_nested = False
                     for account in balance_list:
                         # Match account type if fetching specific type, otherwise check relevant types
                         if account.get('accountType') == acc_type: # Match type we attempted
                             coin_list = account.get('coin')
                             if isinstance(coin_list, list):
                                 if any(coin_data.get('coin') == currency for coin_data in coin_list):
                                     lg.debug(f"Currency '{currency}' confirmed in nested V5 balance structure (accountType: {acc_type}).")
                                     found_in_nested = True
                                     break # Found in this account's coin list
                         if found_in_nested:
                            break # Break outer loop too
                     if found_in_nested:
                        lg.debug(f"Using balance_info from successful nested check (accountType: {acc_type})")
                        break # Break main loop too

                lg.debug(f"Currency '{currency}' not found in standard or nested structure using accountType '{acc_type}'. Trying next.")
                balance_info = None # Reset to try next type

            except ccxt.ExchangeError as e:
                # Ignore errors indicating the account type doesn't exist or similar API issues for that type
                err_str = str(e).lower()
                bybit_code = getattr(e, 'code', None)

                # Bybit V5 specific account type errors (e.g., not enabled, wrong API key permissions)
                if bybit_code == 130057 or "account type not support" in err_str or "invalid account type" in err_str:
                    lg.debug(f"Account type '{acc_type}' not supported or error fetching: {e}. Trying next.")
                    continue
                # Some V5 errors might return 0 balance list without error code
                # If the list is empty and we requested a specific coin, it might mean 0 balance
                if bybit_code == 0 and balance_info and 'result' in balance_info.get('info', {}) and isinstance(balance_info['info']['result'].get('list'), list) and not balance_info['info']['result']['list'] and 'coin' in params:
                     lg.debug(f"Fetch for {currency} on accountType '{acc_type}' returned empty list (Code 0). Assuming 0 balance for this account type.")
                     # Create a dummy structure to indicate 0 balance for this type if needed later, but for now, just continue
                     continue # Try next type if it's just 0

                else:
                    lg.warning(f"Exchange error fetching balance for account type {acc_type}: {e}. Trying next.")
                    continue # Try next type on other exchange errors too? Maybe safer.
            except Exception as e:
                lg.warning(f"Unexpected error fetching balance for account type {acc_type}: {e}. Trying next.")
                continue

        # --- If specific account types failed or currency not found, try default fetch_balance ---
        # This fetches aggregate balance and CCXT attempts to parse it
        if not balance_info or (currency in balance_info and balance_info[currency].get('free') is None):
            lg.debug(f"Fetching balance using default parameters as fallback for {currency}...")
            try:
                balance_info = exchange.fetch_balance()
                if balance_info:
                    balance_info['params'] = {} # Mark as default fetch
            except Exception as e:
                lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}")
                return None # Critical failure

        # If still no balance info after all attempts
        if not balance_info:
             lg.error(f"{NEON_RED}Failed to fetch any balance information after trying all methods for {currency}.{RESET}")
             return None

        # --- Parse the balance_info for the desired currency ---
        available_balance_str = None

        # 1. Standard CCXT structure: balance_info[currency]['free']
        if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('free') is not None:
            available_balance_str = str(balance_info[currency]['free'])
            lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free']: {available_balance_str}")

        # 2. Bybit V5 structure (often nested): Check 'info' -> 'result' -> 'list'
        # Re-parse regardless of which fetch method succeeded, as the structure can vary
        if available_balance_str is None and 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
            balance_list = balance_info['info']['result']['list']
            # If using specific account type params succeeded, prioritize that account in the list
            successful_acc_type = balance_info.get('params',{}).get('accountType')

            # Iterate through the list of accounts found in the response
            for account in balance_list:
                current_account_type = account.get('accountType')
                # Prioritize the account type that was successfully queried, if known.
                # If default fetch was used, check all account types present in the list.
                if successful_acc_type is None or current_account_type == successful_acc_type:
                    coin_list = account.get('coin')
                    if isinstance(coin_list, list):
                        for coin_data in coin_list:
                            if coin_data.get('coin') == currency:
                                # Prefer Bybit V5 'availableToWithdraw' or 'availableBalance'
                                free = coin_data.get('availableToWithdraw') # For funding/unified/contract
                                if free is None: free = coin_data.get('availableBalance') # For spot
                                # Fallback to walletBalance only if others are missing (might include unrealized PnL)
                                if free is None: free = coin_data.get('walletBalance') # Often total balance

                                if free is not None:
                                    available_balance_str = str(free)
                                    lg.debug(f"Found balance via Bybit V5 nested structure: {available_balance_str} {currency} (Account: {current_account_type or 'N/A'})")
                                    break # Found the currency in this account
                        if available_balance_str is not None: break # Stop searching accounts in the list

            if available_balance_str is None:
                lg.debug(f"{currency} not found within Bybit V5 'info.result.list[].coin[]' structure for relevant account type(s).")


        # 3. Fallback: Check top-level 'free' dictionary if present (older CCXT versions/exchanges)
        if available_balance_str is None and 'free' in balance_info and isinstance(balance_info.get('free'), dict) and currency in balance_info['free'] and balance_info['free'][currency] is not None:
            available_balance_str = str(balance_info['free'][currency])
            lg.debug(f"Found balance via top-level 'free' dictionary: {available_balance_str} {currency}")

        # 4. Last Resort: Check 'total' balance if 'free' is still missing and we found *any* balance info
        if available_balance_str is None and balance_info is not None:
            total_balance = None
            # Check standard ccxt total
            if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('total') is not None:
                total_balance = balance_info[currency]['total']
            # Add check for Bybit nested total if primary failed (already handled in step 2, but double check)
            elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                 balance_list = balance_info['info']['result']['list']
                 successful_acc_type = balance_info.get('params',{}).get('accountType') # Re-check successful type
                 for account in balance_list:
                    current_account_type = account.get('accountType')
                    if successful_acc_type is None or current_account_type == successful_acc_type: # Prioritize successful type or check all
                        coin_list = account.get('coin')
                        if isinstance(coin_list, list):
                            for coin_data in coin_list:
                                if coin_data.get('coin') == currency:
                                    total_balance = coin_data.get('walletBalance') # Use walletBalance as total proxy
                                    if total_balance is not None: break
                            if total_balance is not None: break
                 if total_balance is not None:
                      lg.debug(f"Using 'walletBalance' ({total_balance}) from nested structure as 'total' fallback.")


            if total_balance is not None:
                lg.warning(f"{NEON_YELLOW}Could not determine 'free'/'available' balance for {currency}. Using 'total' balance ({total_balance}) as fallback. This might include collateral/unrealized PnL.{RESET}")
                available_balance_str = str(total_balance)
            else:
                lg.error(f"{NEON_RED}Could not determine any balance for {currency} using total balance fallback. Balance info structure not recognized or currency missing.{RESET}")
                # lg.debug(f"Full balance_info structure: {balance_info}") # Log structure for debugging if needed
                return None

        # --- Convert to Decimal ---
        if available_balance_str is not None:
             try:
                 final_balance = Decimal(available_balance_str)
                 if final_balance >= 0: # Allow zero balance
                     lg.info(f"Available {currency} balance: {final_balance:.4f}")
                     return final_balance
                 else:
                     lg.error(f"Parsed balance for {currency} is negative ({final_balance}). Returning None.")
                     return None
             except (InvalidOperation, ValueError, TypeError) as e:
                 lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
                 return None
        else:
             # This case should be caught by the checks above, but defensive
             lg.error(f"{NEON_RED}Logic error: available_balance_str is None after parsing attempts.{RESET}")
             return None


    except ccxt.AuthenticationError as e:
        lg.error(f"{NEON_RED}Authentication error fetching balance: {e}. Check API key permissions.{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}")
        # Consider if balance fetch should be retried or if failure is critical
        return None
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision, limits, contract type, ensuring markets are loaded."""
    lg = logger
    try:
        # Ensure markets are loaded; reload if symbol is missing or markets seem outdated
        last_load_time = getattr(exchange, 'last_load_markets_timestamp', 0)
        if not exchange.markets or symbol not in exchange.markets or time.time() - last_load_time > 3600: # Reload if > 1 hour old
            lg.info(f"Market info for {symbol} not loaded, symbol missing, or markets outdated, reloading markets...")
            # For Bybit V5, specify category for faster loading if you know the type
            params = {}
            # Simple guess for category based on symbol format
            if 'bybit' in exchange.id.lower():
                 category = 'linear' # Default guess
                 if symbol.endswith(':USDT') or '/USDT' in symbol: category = 'linear'
                 elif symbol.endswith(':USD') or '/USD' in symbol: category = 'inverse'
                 elif '/' not in symbol: category = 'spot' # Assume spot if no '/'
                 # Add other categories if needed (e.g., options)
                 params['category'] = category
                 lg.debug(f"Loading markets with category: {category}")

            exchange.load_markets(reload=True, params=params) # Force reload
            exchange.last_load_markets_timestamp = time.time() # Update timestamp


        # Check again after reloading
        if symbol not in exchange.markets:
            lg.error(f"{NEON_RED}Market {symbol} still not found after reloading markets. Check symbol spelling and availability on {exchange.id}.{RESET}")
            return None

        market = exchange.market(symbol)
        if market:
            # Log key details for confirmation and debugging
            market_type = market.get('type', 'unknown') # spot, swap, future etc.
            contract_type = "Linear" if market.get('linear') else "Inverse" if market.get('inverse') else "N/A"
            # Add contract flag for easier checking later
            market['is_contract'] = market.get('contract', False) or market_type in ['swap', 'future']
            # Add market ID explicitly if not already present
            if 'id' not in market: market['id'] = market.get('info', {}).get('symbol', symbol.replace('/', '').replace(':', '')) # Fallback ID construction

            # --- Derive tick size if missing using TradingAnalyzer helper ---
            # This adds the tick size to the market dict for later use
            # Ensure Analyzer instance is available or create a temporary one
            if market.get('precision', {}).get('tick') is None:
                lg.debug(f"Attempting to derive missing tick size for {symbol} from market info...")
                # Use a temporary analyzer instance just for the helper method
                dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                dummy_df.index.name = 'timestamp'
                try:
                    dummy_config = {"indicators": {}} # minimal config needed by Analyzer init
                    # Pass logger, config, market_info, and dummy state
                    analyzer_temp = TradingAnalyzer(dummy_df, lg, dummy_config, market, {})
                    min_tick = analyzer_temp.get_min_tick_size() # Use the robust helper
                    if min_tick > 0:
                        if 'precision' not in market: market['precision'] = {}
                        # Store as float for consistency
                        market['precision']['tick'] = float(min_tick)
                        lg.debug(f"Added derived min tick size {min_tick} (float:{float(min_tick)}) to market info precision for {symbol}.")
                    else:
                        lg.warning(f"Could not derive a valid positive tick size for {symbol}.")
                except ValueError as ve: # Catch error if market_info was invalid structure
                     lg.warning(f"Could not derive min tick size due to issue with market info structure ({symbol}): {ve}")
                except Exception as e_tick:
                     lg.warning(f"Error deriving min tick size for {symbol}: {e_tick}")


            # --- Log derived info ---
            # Safely get precision values for logging
            precision_dict = market.get('precision', {})
            price_prec_raw = precision_dict.get('price') # Can be integer (decimals) or float/string (tick size)
            amount_prec_raw = precision_dict.get('amount') # Can be integer (decimals) or float/string (step size)
            tick_prec = market.get('precision', {}).get('tick', 'N/A') # Use the potentially derived value

            # Format price_prec and amount_prec for logging based on their type
            price_prec_log = f"{price_prec_raw} (decimals)" if isinstance(price_prec_raw, int) else f"{price_prec_raw} (tick size)" if price_prec_raw is not None else 'N/A'
            amount_prec_log = f"{amount_prec_raw} (decimals)" if isinstance(amount_prec_raw, int) else f"{amount_prec_raw} (step size)" if amount_prec_raw is not None else 'N/A'

            # Safely get limit values for logging
            limits_dict = market.get('limits', {})
            amount_limits = limits_dict.get('amount', {})
            cost_limits = limits_dict.get('cost', {})
            min_amount = amount_limits.get('min', 'N/A')
            max_amount = amount_limits.get('max', 'N/A')
            min_cost = cost_limits.get('min', 'N/A')
            max_cost = cost_limits.get('max', 'N/A')

            lg.info(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract={contract_type}, "
                f"Precision(Price/Amount/Tick): {price_prec_log}/{amount_prec_log}/{tick_prec}, "
                f"Limits(Amount Min/Max): {min_amount}/{max_amount}, "
                f"Limits(Cost Min/Max): {min_cost}/{max_cost}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            return market
        else:
            # Should have been caught by the 'in exchange.markets' check, but safeguard
            lg.error(f"{NEON_RED}Market dictionary not found for {symbol} even after checking exchange.markets.{RESET}")
            return None
    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Symbol '{symbol}' is not supported by {exchange.id} or is incorrectly formatted: {e}{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error loading markets info for {symbol}: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error loading markets info for {symbol}: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


# --- Initialize Exchange (Place before functions that use it) ---
# Moved setup_logger call inside main() after console level is set by args
# Moved initialize_exchange call inside main()

# --- CCXT Data Fetching Functions ---
# (fetch_current_price_ccxt, fetch_klines_ccxt, fetch_orderbook_ccxt definitions go here)
# ... (Definitions are provided above, ensure they are included) ...

# --- Trading Analyzer Class ---
# (TradingAnalyzer class definition goes here)
# ... (Definition is provided above, ensure it is included) ...

# --- Trading Logic Helper Functions ---
# (fetch_balance, get_market_info, calculate_position_size, _manual_amount_rounding,
#  get_open_position, set_leverage_ccxt, place_trade, _set_position_protection,
#  set_trailing_stop_loss definitions go here)
# ... (Definitions are provided above, ensure they are included) ...

# --- Main Analysis and Trading Logic ---
def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    market_info: Dict, # Pass pre-fetched market info
    symbol_state: Dict[str, Any], # Pass mutable state dict for the symbol
    current_balance: Optional[Decimal], # Pass current balance
    all_open_positions: Dict[str, Dict] # Pass dict of all open positions {symbol: position_dict}
) -> None:
    """Analyzes a single symbol and executes/manages trades based on signals and config."""

    lg = logger # Use the symbol-specific logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) ==---")
    cycle_start_time = time.monotonic()

    # Use market_info passed from the main loop
    if not market_info:
        lg.error(f"{NEON_RED}Missing market info for {symbol}. Skipping analysis cycle.{RESET}")
        return

    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
        lg.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe for {symbol}. Skipping.")
        return

    # --- 1. Fetch Data (Kline, Ticker, Orderbook) ---
    # Determine required kline history
    required_kline_history = max([
        int(config.get(p, globals().get(f"DEFAULT_{p.upper()}", 14))) # Get period from config or default const
        for p in ["atr_period", "ema_long_period", "cci_window", "williams_r_window",
                  "mfi_window", "sma_10_window", "rsi_period", "bollinger_bands_period",
                  "volume_ma_period", "fibonacci_window"]
        if isinstance(config.get(p, globals().get(f"DEFAULT_{p.upper()}", 14)), (int, float)) # Ensure it's numeric
    ] + [
        int(config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)) + \
        int(config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)) + \
        max(int(config.get("stoch_rsi_k", DEFAULT_K_WINDOW)), int(config.get("stoch_rsi_d", DEFAULT_D_WINDOW)))
    ]) + 50 # Add a buffer

    kline_limit = max(250, required_kline_history)
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg, market_info=market_info)
    if klines_df.empty or len(klines_df) < max(50, required_kline_history):
        lg.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}, required ~{max(50, required_kline_history)}). Skipping analysis.{RESET}")
        return

    # Fetch current price using market_info
    current_price = fetch_current_price_ccxt(exchange, symbol, lg, market_info=market_info)
    if current_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch current ticker price for {symbol}. Using last close from klines as fallback.{RESET}")
        try:
            last_close_val = klines_df['close'].iloc[-1]
            if pd.notna(last_close_val) and last_close_val > 0:
                current_price = Decimal(str(last_close_val))
                lg.info(f"Using last close price: {current_price}")
            else:
                lg.error(f"{NEON_RED}Last close price ({last_close_val}) is also invalid. Cannot proceed for {symbol}.{RESET}")
                return
        except (IndexError, ValueError, TypeError, InvalidOperation) as e:
            lg.error(f"{NEON_RED}Error getting last close price for {symbol}: {e}. Cannot proceed.{RESET}")
            return

    # Fetch order book if enabled and weighted
    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    if config.get("indicators",{}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", 0))) != 0:
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config.get("orderbook_limit", 25), lg, market_info=market_info)
        if orderbook_data is None:
             lg.warning(f"{NEON_YELLOW}Orderbook fetching enabled but failed for {symbol}. Score will be NaN.{RESET}")


    # --- 2. Analyze Data & Generate Signal ---
    try:
        # Pass persistent symbol_state dict
        analyzer = TradingAnalyzer(
            df=klines_df.copy(),
            logger=lg,
            config=config,
            market_info=market_info,
            symbol_state=symbol_state
        )
    except ValueError as e_analyzer:
        lg.error(f"{NEON_RED}Failed to initialize TradingAnalyzer for {symbol}: {e_analyzer}. Skipping analysis.{RESET}")
        return

    # Generate the trading signal
    signal = analyzer.generate_trading_signal(current_price, orderbook_data)

    # Get necessary data points for potential trade/management
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_atr_float = analyzer.indicator_values.get("ATR")

    if min_tick_size <= 0:
         lg.error(f"{NEON_RED}Invalid minimum tick size ({min_tick_size}) for {symbol}. Cannot calculate trade parameters. Skipping cycle.{RESET}")
         return

    # Calculate potential initial SL/TP (used for sizing if no position exists)
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal)


    # --- 3. Log Analysis Summary ---
    atr_log = f"{current_atr_float:.{price_precision+1}f}" if current_atr_float is not None and pd.notna(current_atr_float) else 'N/A'
    sl_pot_log = f"{sl_potential:.{price_precision}f}" if sl_potential else 'N/A'
    tp_pot_log = f"{tp_potential:.{price_precision}f}" if tp_potential else 'N/A'
    lg.info(f"Analysis: ATR={atr_log}, Potential SL={sl_pot_log}, Potential TP={tp_pot_log}")

    tsl_enabled = config.get('enable_trailing_stop', False)
    be_enabled = config.get('enable_break_even', False)
    ma_exit_enabled = config.get('enable_ma_cross_exit', False)
    lg.info(f"Config: TSL={tsl_enabled}, BE={be_enabled}, MA Cross Exit={ma_exit_enabled}")


    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.info(f"{NEON_YELLOW}Trading is disabled in config. Analysis complete, no trade actions taken for {symbol}.{RESET}")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # Use the pre-fetched position data for this symbol
    open_position = all_open_positions.get(symbol) # Get position for this symbol, will be None if not open

    # Get current count of ALL open positions (across all symbols)
    current_total_open_positions = len(all_open_positions)

    # --- Scenario 1: No Open Position for this Symbol ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            # --- Check Max Position Limit ---
            max_pos_total = config.get("max_concurrent_positions_total", 1)
            if current_total_open_positions >= max_pos_total:
                 lg.info(f"{NEON_YELLOW}Signal {signal} for {symbol}, but max concurrent positions ({max_pos_total}) already reached. Skipping new entry.{RESET}")
                 return

            lg.info(f"*** {signal} Signal & No Open Position: Initiating Trade Sequence for {symbol} ***")

            # --- Pre-Trade Checks & Setup ---
            # Balance check (using balance passed from main loop)
            if current_balance is None or current_balance <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid or zero balance ({current_balance} {QUOTE_CURRENCY}).{RESET}")
                return

            # Check potential SL (needed for sizing)
            if sl_potential is None:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot size trade, Potential Initial Stop Loss calculation failed (ATR invalid?).{RESET}")
                return

            # Set Leverage (only for contracts)
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set/confirm leverage to {leverage}x. Check logs.{RESET}")
                        return
            else:
                 lg.info(f"Leverage setting skipped for {symbol} (Spot market).")


            # Calculate Position Size using potential SL
            position_size = calculate_position_size(
                balance=current_balance, # Use passed balance
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_potential, # Use potential SL for sizing
                entry_price=current_price, # Use current price as estimate
                market_info=market_info,
                exchange=exchange,
                logger=lg
            )

            if position_size is None or position_size <= 0:
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size calculated ({position_size}). Check balance, risk, SL, market limits.{RESET}")
                return

            # --- Place Initial Market Order ---
            lg.info(f"==> Placing {signal} market order for {symbol} | Size: {position_size} <==")
            trade_order = place_trade(
                exchange=exchange,
                symbol=symbol,
                trade_signal=signal,
                position_size=position_size, # Pass Decimal size
                market_info=market_info,
                logger=lg,
                reduce_only=False # Opening trade
            )

            # --- Post-Order: Verify Position and Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                lg.info(f"Order {order_id} placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s for position confirmation...")
                time.sleep(POSITION_CONFIRM_DELAY)

                # Re-fetch position state specifically for this symbol AFTER the delay
                lg.info(f"Attempting to confirm position for {symbol} after order {order_id}...")
                confirmed_position = get_open_position(exchange, symbol, lg, market_info=market_info, position_mode=config.get("position_mode", "One-Way"))

                if confirmed_position:
                    # --- Position Confirmed ---
                    try:
                        # Use enhanced Decimal fields from get_open_position
                        entry_price_actual = confirmed_position.get('entryPriceDecimal')
                        pos_size_actual = confirmed_position.get('contractsDecimal') # Correct key name? Check get_open_position enhancement
                        # If contractsDecimal is not set, try 'contracts' again
                        if pos_size_actual is None:
                            pos_size_actual = Decimal(str(confirmed_position.get('contracts', '0')))


                        valid_entry = False
                        if entry_price_actual and pos_size_actual:
                             min_size_threshold = Decimal('1e-9') # Basic non-zero check
                             if market_info:
                                 try: min_size_threshold = max(min_size_threshold, Decimal(str(market_info['limits']['amount']['min'])) * Decimal('0.01'))
                                 except Exception: pass

                             if entry_price_actual > 0 and abs(pos_size_actual) >= min_size_threshold:
                                 valid_entry = True
                             else:
                                 lg.error(f"Confirmed position has invalid entry price ({entry_price_actual}) or size ({pos_size_actual}).")
                        else:
                            lg.error("Confirmed position missing valid entryPrice or size Decimal information.")


                        if valid_entry:
                            lg.info(f"{NEON_GREEN}Position Confirmed for {symbol}! Actual Entry: ~{entry_price_actual:.{price_precision}f}, Actual Size: {pos_size_actual}{RESET}")

                            # --- Recalculate SL/TP based on ACTUAL entry price ---
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)

                            # --- Set Protection based on Config (TSL or Fixed SL/TP) ---
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting Trailing Stop Loss for {symbol} (TP target: {tp_actual})...")
                                protection_set_success = set_trailing_stop_loss(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=confirmed_position, config=config, logger=lg,
                                    take_profit_price=tp_actual # Pass optional TP
                                )
                            else:
                                lg.info(f"Setting Fixed Stop Loss ({sl_actual}) and Take Profit ({tp_actual}) for {symbol}...")
                                if sl_actual or tp_actual: # Only call if at least one is valid
                                    protection_set_success = _set_position_protection(
                                        exchange=exchange, symbol=symbol, market_info=market_info,
                                        position_info=confirmed_position, logger=lg,
                                        stop_loss_price=sl_actual, # Pass Decimal or None
                                        take_profit_price=tp_actual, # Pass Decimal or None
                                        trailing_stop_distance='0', # Ensure TSL is cancelled
                                        tsl_activation_price='0'
                                    )
                                else:
                                    lg.warning(f"{NEON_YELLOW}Fixed SL/TP calculation based on actual entry failed or resulted in None for {symbol}. No fixed protection set.{RESET}")
                                    protection_set_success = True # Treat as 'success' if no protection needed

                            # --- Final Status Log ---
                            if protection_set_success:
                                lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE for {symbol} ({signal}) ===")
                                # Reset BE state for the new trade
                                analyzer.break_even_triggered = False
                            else:
                                lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/CONFIRM PROTECTION (SL/TP/TSL) for {symbol} ({signal}) ===")
                                lg.warning(f"{NEON_YELLOW}Position is open without automated protection. Manual monitoring/intervention may be required!{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Trade placed, but confirmed position data is invalid. Cannot set protection.{RESET}")
                            lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")

                    except Exception as post_trade_err:
                        lg.error(f"{NEON_RED}Error during post-trade processing (protection setting) for {symbol}: {post_trade_err}{RESET}", exc_info=True)
                        lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")

                else:
                    # --- Position NOT Confirmed ---
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM open position for {symbol} after {POSITION_CONFIRM_DELAY}s delay!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order might have failed to fill, filled partially, rejected, or there's a significant delay/API issue.{RESET}")
                    lg.warning(f"{NEON_YELLOW}No protection will be set. Manual investigation of order {order_id} and position status required!{RESET}")
                    # Optional: Try fetching the order status itself for more info
                    try:
                        order_status_info = exchange.fetch_order(order_id, symbol)
                        lg.info(f"Status of order {order_id}: {order_status_info.get('status', 'Unknown')}, Filled: {order_status_info.get('filled', 'N/A')}")
                    except Exception as fetch_order_err:
                        lg.warning(f"Could not fetch status for order {order_id}: {fetch_order_err}")

            else:
                # place_trade function returned None (order failed)
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED for {symbol} ({signal}). See previous logs for details. ===")
        else:
            # No open position, and signal is HOLD
            lg.info(f"Signal is HOLD and no open position for {symbol}. No trade action taken.")


    # --- Scenario 2: Existing Open Position Found for this Symbol ---
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        # Use enhanced Decimal fields if available, otherwise parse raw
        try: pos_size_dec = open_position.get('contractsDecimal', Decimal(str(open_position.get('contracts', '0'))))
        except: pos_size_dec = Decimal('0')
        try: entry_price_dec = open_position.get('entryPriceDecimal', Decimal(str(open_position.get('entryPrice', '0'))))
        except: entry_price_dec = None

        # Use parsed Decimal versions for protection checks
        current_sl_price_dec = open_position.get('stopLossPriceDecimal') # Might be None or Decimal(0)
        current_tp_price_dec = open_position.get('takeProfitPriceDecimal') # Might be None or Decimal(0)
        tsl_distance_dec = open_position.get('trailingStopLossDistanceDecimal', Decimal(0))
        is_tsl_active = tsl_distance_dec > 0

        # Format for logging
        sl_log_str = f"{current_sl_price_dec:.{price_precision}f}" if current_sl_price_dec else 'N/A'
        tp_log_str = f"{current_tp_price_dec:.{price_precision}f}" if current_tp_price_dec else 'N/A'
        pos_size_log = f"{pos_size_dec:.8f}" if pos_size_dec else 'N/A'
        entry_price_log = f"{entry_price_dec:.{price_precision}f}" if entry_price_dec else 'N/A'

        lg.info(f"Existing {pos_side.upper()} position found for {symbol}. Size: {pos_size_log}, Entry: {entry_price_log}, SL: {sl_log_str}, TP: {tp_log_str}, TSL Active: {is_tsl_active}")

        # Validate essential position info for management
        if pos_side not in ['long', 'short'] or not entry_price_dec or entry_price_dec <= 0 or not pos_size_dec or pos_size_dec <= 0:
             lg.error(f"{NEON_RED}Cannot manage existing position for {symbol}: Invalid side ('{pos_side}'), entry price ('{entry_price_dec}'), or size ('{pos_size_dec}') detected.{RESET}")
             return # Stop management for this symbol if core info is bad


        # --- ** Check Exit Conditions ** ---
        # Priority: 1. Main Signal, 2. MA Cross (if enabled)
        # Note: Exchange SL/TP/TSL trigger independently

        # Condition 1: Main signal opposes the current position direction
        exit_signal_triggered = False
        if (pos_side == 'long' and signal == "SELL") or \
           (pos_side == 'short' and signal == "BUY"):
            exit_signal_triggered = True
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position for {symbol}. ***{RESET}")

        # Condition 2: MA Cross exit (if enabled and not already exiting via signal)
        ma_cross_exit = False
        if not exit_signal_triggered and config.get("enable_ma_cross_exit", False):
            ema_short = analyzer.indicator_values.get("EMA_Short")
            ema_long = analyzer.indicator_values.get("EMA_Long")

            if pd.notna(ema_short) and pd.notna(ema_long):
                # Check for adverse cross based on position side
                if pos_side == 'long' and ema_short < ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bearish): Short EMA ({ema_short:.{price_precision}f}) crossed below Long EMA ({ema_long:.{price_precision}f}). Closing LONG position. ***{RESET}")
                elif pos_side == 'short' and ema_short > ema_long:
                    ma_cross_exit = True
                    lg.warning(f"{NEON_YELLOW}*** MA CROSS EXIT (Bullish): Short EMA ({ema_short:.{price_precision}f}) crossed above Long EMA ({ema_long:.{price_precision}f}). Closing SHORT position. ***{RESET}")
            else:
                lg.warning("MA cross exit check skipped: EMA values not available.")


        # --- Execute Position Close if Exit Condition Met ---
        if exit_signal_triggered or ma_cross_exit:
            lg.info(f"Attempting to close {pos_side} position for {symbol} with a market order...")
            try:
                # Determine the side needed to close the position
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                # Use the Decimal size from the fetched position info
                size_to_close = abs(pos_size_dec)
                if size_to_close <= 0: # Should be caught by earlier check, but defensive
                    raise ValueError(f"Cannot close: Existing position size {pos_size_dec} is zero or invalid.")

                # --- Place Closing Market Order (reduceOnly=True) ---
                close_order = place_trade(
                    exchange=exchange, symbol=symbol, trade_signal=close_side_signal,
                    position_size=size_to_close, market_info=market_info, logger=lg,
                    reduce_only=True # CRITICAL for closing
                )

                if close_order:
                     order_id = close_order.get('id', 'N/A')
                     # Handle the "Position not found" dummy response from place_trade
                     if close_order.get('info',{}).get('reason') == 'Position not found on close attempt':
                          lg.info(f"{NEON_GREEN}Position for {symbol} confirmed already closed (or closing order unnecessary).{RESET}")
                          # Reset state if position is closed
                          analyzer.break_even_triggered = False
                     else:
                          lg.info(f"{NEON_GREEN}Closing order {order_id} placed for {symbol}. Waiting {POSITION_CONFIRM_DELAY}s to verify closure...{RESET}")
                          time.sleep(POSITION_CONFIRM_DELAY)
                          # Verify position is actually closed
                          final_position = get_open_position(exchange, symbol, lg, market_info=market_info, position_mode=config.get("position_mode", "One-Way"))
                          if final_position is None:
                              lg.info(f"{NEON_GREEN}=== POSITION for {symbol} successfully closed. ===")
                              # Reset state after successful closure
                              analyzer.break_even_triggered = False
                          else:
                              lg.error(f"{NEON_RED}*** POSITION CLOSE FAILED for {symbol} after placing reduceOnly order {order_id}. Position still detected: {final_position}{RESET}")
                              lg.warning(f"{NEON_YELLOW}Manual investigation required!{RESET}")
                else:
                    lg.error(f"{NEON_RED}Failed to place closing order for {symbol}. Manual intervention required.{RESET}")

            except (ValueError, InvalidOperation, TypeError) as size_err:
                 lg.error(f"{NEON_RED}Error determining size for closing order ({symbol}): {size_err}. Manual intervention required.{RESET}")
            except Exception as close_err:
                 lg.error(f"{NEON_RED}Unexpected error during position close attempt for {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention required!{RESET}")

        # --- Check Break-Even Condition (Only if NOT Exiting and BE not already triggered for this trade) ---
        elif config.get("enable_break_even", False) and not analyzer.break_even_triggered:
            # Use already validated entry_price_dec and current_atr_float
            if current_atr_float and pd.notna(current_atr_float) and current_atr_float > 0:
                try:
                    current_atr_dec = Decimal(str(current_atr_float))
                    trigger_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    offset_ticks = int(config.get("break_even_offset_ticks", 2))
                    min_tick = min_tick_size # Use min_tick_size fetched earlier

                    profit_target_offset = current_atr_dec * trigger_multiple
                    be_stop_price = None
                    trigger_met = False
                    trigger_price = None

                    if pos_side == 'long':
                        trigger_price = entry_price_dec + profit_target_offset
                        if current_price >= trigger_price:
                            trigger_met = True
                            be_stop_raw = entry_price_dec + (min_tick * offset_ticks)
                            be_stop_price = (be_stop_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                            if be_stop_price <= entry_price_dec:
                                be_stop_price = entry_price_dec + min_tick

                    elif pos_side == 'short':
                        trigger_price = entry_price_dec - profit_target_offset
                        if current_price <= trigger_price:
                            trigger_met = True
                            be_stop_raw = entry_price_dec - (min_tick * offset_ticks)
                            be_stop_price = (be_stop_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                            if be_stop_price >= entry_price_dec:
                                be_stop_price = entry_price_dec - min_tick

                    # --- Trigger Break-Even SL Modification ---
                    if trigger_met:
                        if be_stop_price and be_stop_price > 0:
                            lg.warning(f"{NEON_PURPLE}*** BREAK-EVEN Triggered for {symbol} {pos_side.upper()} position! ***")
                            trigger_price_log = f"{trigger_price:.{price_precision}f}" if trigger_price else "N/A"
                            lg.info(f"  Current Price: {current_price:.{price_precision}f}, Trigger Price: {trigger_price_log}")
                            lg.info(f"  Current SL: {sl_log_str}, Current TP: {tp_log_str}, TSL Active: {is_tsl_active}")
                            lg.info(f"  Target BE SL: {be_stop_price:.{price_precision}f} (Entry {entry_price_log} +/- {offset_ticks} ticks)")

                            # Determine if SL needs update based on current protection and BE target
                            needs_sl_update = True
                            force_fixed_sl_on_be = config.get("break_even_force_fixed_sl", True)

                            if is_tsl_active and not force_fixed_sl_on_be:
                                 # TODO: Add logic to compare current TSL calculated stop vs BE target if needed.
                                 # For now, if not forcing fixed SL, assume we keep TSL if it's active.
                                 lg.warning(f"{NEON_YELLOW}BE triggered, but TSL is active and 'break_even_force_fixed_sl' is False. Keeping TSL active.{RESET}")
                                 needs_sl_update = False
                                 analyzer.break_even_triggered = True # Mark BE logic as evaluated
                            elif current_sl_price_dec and not is_tsl_active: # Fixed SL exists
                                if (pos_side == 'long' and current_sl_price_dec >= be_stop_price) or \
                                   (pos_side == 'short' and current_sl_price_dec <= be_stop_price):
                                     lg.info(f"  Existing Fixed SL ({sl_log_str}) is already at or better than break-even target ({be_stop_price:.{price_precision}f}). No modification needed.")
                                     needs_sl_update = False
                                     analyzer.break_even_triggered = True # Mark BE logic as evaluated

                            # Perform SL update if needed
                            if needs_sl_update:
                                lg.info(f"  Modifying protection: Setting Fixed SL to {be_stop_price:.{price_precision}f}, keeping TP {tp_log_str}, cancelling TSL (if active).")
                                be_success = _set_position_protection(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=open_position, logger=lg,
                                    stop_loss_price=be_stop_price, # New BE SL
                                    take_profit_price=current_tp_price_dec, # Keep existing TP
                                    trailing_stop_distance='0', # Cancel TSL when setting fixed BE SL
                                    tsl_activation_price='0'
                                )
                                if be_success:
                                    lg.info(f"{NEON_GREEN}Break-even stop loss successfully set/updated for {symbol}.{RESET}")
                                    analyzer.break_even_triggered = True # Mark BE as actioned
                                else:
                                    lg.error(f"{NEON_RED}Failed to set break-even stop loss for {symbol}. Original protection may still be active.{RESET}")

                        else: # be_stop_price calculation failed or invalid
                            lg.error(f"{NEON_RED}Break-even triggered, but calculated BE stop price ({be_stop_price}) is invalid. Cannot modify SL.{RESET}")

                except (InvalidOperation, ValueError, TypeError, KeyError, Exception) as be_err:
                    lg.error(f"{NEON_RED}Error during break-even check/calculation for {symbol}: {be_err}{RESET}", exc_info=True)
            else:
                # Log reason if BE check skipped due to missing data
                if not entry_price_dec or entry_price_dec <= 0: lg.debug(f"BE check skipped: Invalid entry price.")
                elif not current_atr_float or not pd.notna(current_atr_float) or current_atr_float <= 0: lg.debug(f"BE check skipped: Invalid ATR.")
                elif not min_tick_size or min_tick_size <= 0: lg.debug(f"BE check skipped: Invalid min tick size.")


        # --- No Exit/BE Condition Met ---
        else: # Handles case where not exiting and (BE disabled OR BE already triggered OR BE condition not met)
             # Check if BE has already been triggered in symbol_state
             if not analyzer.break_even_triggered:
                 lg.info(f"Signal is {signal}. Holding existing {pos_side.upper()} position for {symbol}. Monitoring protection. BE not triggered.")
             else: # BE was triggered previously (state is True)
                  lg.info(f"Signal is {signal}. Holding existing {pos_side.upper()} position for {symbol}. Monitoring protection. BE already triggered.")
             # Optional: Add logic here to adjust TP/SL based on new analysis if desired,
             # being careful not to conflict with TSL or BE logic.

    # --- End of Scenario 2 (Existing Open Position) ---

    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ==---")


# --- Main Function ---
def main():
    """Main function to run the bot."""
    # Determine console log level from arguments
    parser = argparse.ArgumentParser(description="Enhanced Bybit Trading Bot (Whale 2.1)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable DEBUG level logging to console")
    parser.add_argument("-s", "--symbol", help="Trade only a specific symbol (e.g., BTC/USDT:USDT), overrides config list")
    args = parser.parse_args()

    global console_log_level
    if args.debug:
        console_log_level = logging.DEBUG
        print(f"{NEON_YELLOW}Console logging set to DEBUG level.{RESET}")
    else:
        console_log_level = logging.INFO

    # Setup main logger (used for init and overall status)
    main_logger = get_logger("main") # Use get_logger to ensure handler levels are correct
    main_logger.info(f"*** Live Trading Bot Whale 2.1 Initializing ***")
    main_logger.info(f"Using config file: {CONFIG_FILE}")
    main_logger.info(f"Logging to directory: {LOG_DIRECTORY}")
    main_logger.info(f"Configured Timezone: {TIMEZONE}")
    main_logger.info(f"Trading Enabled: {CONFIG.get('enable_trading', False)}")
    main_logger.info(f"Using Sandbox: {CONFIG.get('use_sandbox', True)}")
    main_logger.info(f"Quote Currency: {QUOTE_CURRENCY}")
    main_logger.info(f"Risk Per Trade: {CONFIG.get('risk_per_trade', 0.01):.2%}")
    main_logger.info(f"Leverage: {CONFIG.get('leverage', 1)}x")
    main_logger.info(f"Max Total Positions: {CONFIG.get('max_concurrent_positions_total', 1)}")
    main_logger.info(f"Position Mode (Config): {CONFIG.get('position_mode', 'One-Way')}")


    # Validate interval from config
    if CONFIG.get("interval") not in VALID_INTERVALS:
        main_logger.error(f"{NEON_RED}Invalid 'interval' ({CONFIG.get('interval')}) in {CONFIG_FILE}. Must be one of {VALID_INTERVALS}. Exiting.{RESET}")
        return

    # Initialize exchange
    exchange = initialize_exchange(main_logger)
    if not exchange:
        main_logger.critical(f"{NEON_RED}Failed to initialize exchange. Bot cannot start.{RESET}")
        exit(1) # Exit if exchange fails

    # Determine symbols to trade
    symbols_to_trade = []
    if args.symbol:
         symbols_to_trade = [args.symbol]
         main_logger.info(f"Trading only specified symbol: {args.symbol}")
    else:
         symbols_to_trade = CONFIG.get("symbols", [])
         if not symbols_to_trade:
              main_logger.error(f"{NEON_RED}No symbols configured in {CONFIG_FILE} and no symbol specified via argument. Exiting.{RESET}")
              exit(1) # Exit if no symbols defined
         main_logger.info(f"Trading symbols from config: {', '.join(symbols_to_trade)}")

    # Validate symbols and fetch initial market info
    valid_symbols = []
    market_infos: Dict[str, Dict] = {}
    symbol_loggers: Dict[str, logging.Logger] = {}

    main_logger.info("Validating symbols and fetching market info...")
    for symbol in symbols_to_trade:
        logger = get_logger(symbol, is_symbol_logger=True) # Get/create logger for the symbol
        symbol_loggers[symbol] = logger # Store logger
        market_info = get_market_info(exchange, symbol, logger)
        if market_info:
            valid_symbols.append(symbol)
            market_infos[symbol] = market_info
            # Initialize state for valid symbols
            symbol_states[symbol] = {} # Empty dict to start (break_even_triggered=False implicitly)
        else:
            logger.error(f"Symbol {symbol} is invalid or market info could not be fetched. It will be skipped.")

    if not valid_symbols:
        main_logger.error(f"{NEON_RED}No valid symbols remaining after validation. Exiting.{RESET}")
        exit(1) # Exit if no valid symbols

    main_logger.info(f"Validated symbols: {', '.join(valid_symbols)}")

    # --- Bot Main Loop ---
    main_logger.info(f"Starting main trading loop for {len(valid_symbols)} symbols...")
    while True:
        loop_start_utc = datetime.now(ZoneInfo("UTC"))
        loop_start_local = loop_start_utc.astimezone(TIMEZONE)
        main_logger.debug(f"--- New Main Loop Cycle --- | {loop_start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        try:
            # Fetch global data once per loop cycle (balance, all positions)
            current_balance = fetch_balance(exchange, QUOTE_CURRENCY, main_logger)
            if current_balance is None:
                 main_logger.warning(f"{NEON_YELLOW}Failed to fetch balance for {QUOTE_CURRENCY} this cycle. Size calculations may fail.{RESET}")
                 # Consider skipping this cycle if balance is critical and fetch failed
                 # time.sleep(LOOP_DELAY_SECONDS)
                 # continue

            # Fetch all open positions once
            all_positions_raw = []
            try:
                 # Fetch positions based on category if possible (more efficient for Bybit V5)
                 needs_contract_fetch = any(market_infos[s].get('is_contract') for s in valid_symbols)
                 if needs_contract_fetch:
                     categories_to_fetch = set()
                     for s in valid_symbols:
                         if market_infos[s].get('is_contract'):
                             categories_to_fetch.add('linear' if market_infos[s].get('linear', True) else 'inverse')
                     for category in categories_to_fetch:
                          try:
                              main_logger.debug(f"Fetching {category} positions...")
                              all_positions_raw.extend(exchange.fetch_positions(params={'category': category}))
                          except Exception as e_cat:
                              main_logger.warning(f"Could not fetch {category} positions: {e_cat}")
                 # Spot positions are not typically returned by fetch_positions, handled by balance checks

                 # If fetching by category failed or wasn't applicable, try fetching all (less efficient)
                 if not all_positions_raw and needs_contract_fetch: # Only fetch all if category fetch failed for contracts
                     main_logger.warning("Category position fetch failed or no contracts specified, attempting fetch_positions without category filter...")
                     all_positions_raw = exchange.fetch_positions() # Might fetch only default category or all depending on CCXT/exchange

            except Exception as e_pos:
                main_logger.error(f"{NEON_RED}Failed to fetch open positions this cycle: {e_pos}{RESET}")
                all_positions_raw = [] # Assume none if fetch failed

            # Process raw positions into a dictionary keyed by symbol with enhanced info
            all_open_positions: Dict[str, Dict] = {}
            for pos in all_positions_raw:
                 symbol_key = pos.get('symbol')
                 if symbol_key and symbol_key in valid_symbols:
                     try:
                         pos_size = Decimal(str(pos.get('contracts', pos.get('info',{}).get('size', '0'))))
                         # Use a meaningful threshold (e.g., based on min contract size if available)
                         min_size_threshold = Decimal('1e-9')
                         if market_infos.get(symbol_key):
                             try: min_size_threshold = max(min_size_threshold, Decimal(str(market_infos[symbol_key]['limits']['amount']['min'])) * Decimal('0.01'))
                             except Exception: pass

                         if abs(pos_size) >= min_size_threshold:
                             # Enhance position info (add Decimal versions, etc.)
                             enhanced_pos = pos.copy() # Work on a copy
                             info_dict = enhanced_pos.get('info', {})
                             try: enhanced_pos['entryPriceDecimal'] = Decimal(str(enhanced_pos.get('entryPrice', info_dict.get('avgPrice','0'))))
                             except: enhanced_pos['entryPriceDecimal'] = None
                             try: enhanced_pos['contractsDecimal'] = Decimal(str(enhanced_pos.get('contracts', info_dict.get('size','0'))))
                             except: enhanced_pos['contractsDecimal'] = None
                             # Add other decimal enhancements from get_open_position's logic
                             # SL Price
                             sl_raw = enhanced_pos.get('stopLossPrice', info_dict.get('stopLoss'))
                             enhanced_pos['stopLossPriceDecimal'] = None
                             if sl_raw is not None:
                                try:
                                    sl_dec = Decimal(str(sl_raw))
                                    if sl_dec > 0: enhanced_pos['stopLossPriceDecimal'] = sl_dec
                                except Exception: pass
                             # TP Price
                             tp_raw = enhanced_pos.get('takeProfitPrice', info_dict.get('takeProfit'))
                             enhanced_pos['takeProfitPriceDecimal'] = None
                             if tp_raw is not None:
                                 try:
                                     tp_dec = Decimal(str(tp_raw))
                                     if tp_dec > 0: enhanced_pos['takeProfitPriceDecimal'] = tp_dec
                                 except Exception: pass
                             # TSL Info
                             tsl_dist_raw = info_dict.get('trailingStop', '0')
                             tsl_act_raw = info_dict.get('activePrice', '0')
                             try: enhanced_pos['trailingStopLossDistanceDecimal'] = max(Decimal('0'), Decimal(str(tsl_dist_raw)))
                             except: enhanced_pos['trailingStopLossDistanceDecimal'] = Decimal('0')
                             try: enhanced_pos['tslActivationPriceDecimal'] = max(Decimal('0'), Decimal(str(tsl_act_raw)))
                             except: enhanced_pos['tslActivationPriceDecimal'] = Decimal('0')

                             all_open_positions[symbol_key] = enhanced_pos # Store the enhanced position

                     except (InvalidOperation, ValueError, TypeError) as e:
                         main_logger.warning(f"Could not parse size for potential position {symbol_key}: {e}")
                         pass # Ignore positions with unparseable size

            main_logger.info(f"Total open positions detected: {len(all_open_positions)}")
            if len(all_open_positions) > 0:
                main_logger.info(f"Open positions: {list(all_open_positions.keys())}")


            # --- Iterate Through Symbols ---
            for symbol in valid_symbols:
                symbol_logger = symbol_loggers[symbol]
                symbol_market_info = market_infos[symbol]
                current_symbol_state = symbol_states[symbol] # Get the persistent state dict

                # --- Run analysis and trading logic for the individual symbol ---
                try:
                    analyze_and_trade_symbol(
                        exchange=exchange,
                        symbol=symbol,
                        config=CONFIG,
                        logger=symbol_logger,
                        market_info=symbol_market_info,
                        symbol_state=current_symbol_state,
                        current_balance=current_balance, # Pass overall balance
                        all_open_positions=all_open_positions # Pass all positions dict
                    )
                except Exception as symbol_err:
                     # Catch errors specific to one symbol's processing to prevent crashing the whole bot
                     symbol_logger.error(f"{NEON_RED}Error processing symbol {symbol}: {symbol_err}{RESET}", exc_info=True)
                     symbol_logger.error("Attempting to continue with the next symbol.")

                # Short delay between processing symbols to potentially ease rate limits
                time.sleep(0.5) # Adjust as needed, maybe make configurable

        except KeyboardInterrupt:
            main_logger.info("KeyboardInterrupt received. Attempting graceful shutdown...")
            # Optional: Add logic here to close open positions if desired on exit
            # close_all_open_positions(exchange, all_open_positions, market_infos, main_logger)
            main_logger.info("Shutdown complete.")
            break
        except ccxt.AuthenticationError as e:
            main_logger.critical(f"{NEON_RED}CRITICAL: Authentication Error during main loop: {e}. Bot stopped. Check API keys/permissions.{RESET}")
            break # Stop on authentication errors
        except ccxt.NetworkError as e:
            main_logger.error(f"{NEON_RED}Network Error in main loop: {e}. Retrying after longer delay...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS * 10) # Longer delay for network issues
        except ccxt.RateLimitExceeded as e:
             main_logger.warning(f"{NEON_YELLOW}Rate Limit Exceeded in main loop: {e}. CCXT should handle internally, but consider increasing loop delay if frequent.{RESET}")
             time.sleep(RETRY_DELAY_SECONDS * 2) # Short delay
        except ccxt.ExchangeNotAvailable as e:
            main_logger.error(f"{NEON_RED}Exchange Not Available: {e}. Retrying after significant delay...{RESET}")
            time.sleep(LOOP_DELAY_SECONDS * 5) # Long delay
        except ccxt.ExchangeError as e:
            bybit_code = getattr(e, 'code', None)
            err_str = str(e).lower()
            main_logger.error(f"{NEON_RED}Exchange Error encountered in main loop: {e} (Code: {bybit_code}){RESET}")
            if bybit_code == 10016 or "system maintenance" in err_str:
                main_logger.warning(f"{NEON_YELLOW}Exchange likely in maintenance (Code: {bybit_code}). Waiting longer...{RESET}")
                time.sleep(LOOP_DELAY_SECONDS * 10) # Wait several minutes
            else:
                # For other exchange errors, retry after a moderate delay
                time.sleep(RETRY_DELAY_SECONDS * 3)
        except Exception as e:
            main_logger.error(f"{NEON_RED}An unexpected critical error occurred in the main loop: {e}{RESET}", exc_info=True)
            main_logger.error("Bot encountered a potentially fatal error. For safety, the bot will stop.")
            # Consider sending a notification here (email, Telegram, etc.)
            break # Stop the bot on unexpected errors

        # --- Loop Delay ---
        # Use LOOP_DELAY_SECONDS loaded from config
        main_logger.debug(f"Main loop cycle finished. Sleeping for {LOOP_DELAY_SECONDS} seconds...")
        time.sleep(LOOP_DELAY_SECONDS)

    # --- End of Main Loop ---
    main_logger.info(f"*** Live Trading Bot has stopped. ***")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any uncaught exceptions during initialization or main loop exit
        print(f"{NEON_RED}Unhandled exception reached top level: {e}{RESET}")
        # Attempt to log if possible, otherwise just print
        try:
            # Use the main logger if available
            logger = get_logger("main")
            logger.critical("Unhandled exception caused script termination.", exc_info=True)
        except Exception: # Ignore logging errors during final exception handling
             import traceback
             traceback.print_exc()
    finally:
         print(f"{NEON_CYAN}Bot execution finished.{RESET}")