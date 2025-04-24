```python
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Termux Trading Spell (v2.1.2 - Precision Enhanced & Robust)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved robustness.

import os
import time
import logging
import sys
import subprocess # For termux-toast security
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation, DivisionByZero, ROUND_HALF_UP

# Attempt to import necessary enchantments
try:
    import ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import init, Fore, Style, Back
except ImportError as e:
    # Provide specific guidance for Termux users
    init(autoreset=True) # Initialize colorama for error messages FIRST
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell in your Termux terminal:")
    print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
    print(f"{Style.BRIGHT}pip install ccxt python-dotenv pandas numpy tabulate colorama requests{Style.RESET_ALL}")
    sys.exit(1)

# Weave the Colorama magic into the terminal (initialize after successful imports)
init(autoreset=True)

# Set Decimal precision (adjust if needed, higher precision means more memory/CPU)
# The default precision (usually 28) is often sufficient. Increase only if necessary.
# Example: Increase precision if needed for very small price increments or complex calculations
# getcontext().prec = 50 # Increased precision example

# --- Arcane Configuration ---
print(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.1.2...")

# Summon secrets from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe
# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5  # Between INFO and WARNING
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

def trade(self, message, *args, **kws):
    """Custom logging method for trade-related events."""
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(TRADE_LEVEL_NUM, message, args, **kws)

logging.Logger.trade = trade # Add the custom method to the Logger class

# More detailed log format, includes module and line number for easier debugging
log_formatter = logging.Formatter(
    Fore.CYAN + "%(asctime)s "
    + Style.BRIGHT + "[%(levelname)-8s] " # Padded levelname
    + Fore.WHITE + "(%(filename)s:%(lineno)d) " # Added file/line info
    + Style.RESET_ALL
    + Fore.WHITE + "%(message)s"
)
logger = logging.getLogger(__name__)
# Set level via environment variable or default to INFO
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
# Validate log level string
valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRADE"]
if log_level_str not in valid_log_levels:
    print(f"{Fore.YELLOW}Invalid LOG_LEVEL '{log_level_str}' in .env. Defaulting to INFO.")
    log_level_str = "INFO"

# Map TRADE level string to its numeric value if provided
if log_level_str == "TRADE":
    log_level = TRADE_LEVEL_NUM
else:
    log_level = getattr(logging, log_level_str, logging.INFO)

logger.setLevel(log_level)

# Explicitly use stdout to avoid potential issues in some environments
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
if not logger.hasHandlers():
    logger.addHandler(stream_handler)
# Prevent duplicate messages if the root logger is also configured (common issue)
logger.propagate = False


class TradingConfig:
    """Holds the sacred parameters of our spell, enhanced with precision awareness and validation."""
    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW) # Example: BTC/USDT:USDT (Unified Symbol)
        self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW).lower() # 'linear' or 'inverse'
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.0001"), max_val=Decimal("0.5")) # 0.01% to 50% risk
        self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"))
        self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"))
        # Bybit uses percentage for TSL distance (e.g., 0.5 for 0.5%)
        self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.01"), max_val=Decimal("10.0")) # 0.01% to 10% trail
        self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]) # Usually same as SL

        # Epsilon: Small value for comparing quantities, dynamically determined after market info is loaded.
        # Initialize with a reasonable default Decimal, will be refined based on market step size.
        self.position_qty_epsilon = Decimal("1E-9")

        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000) # Reasonable candle limits
        self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5) # Minimum sleep time
        self.order_check_delay_seconds = self._get_env("ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1)
        self.order_check_timeout_seconds = self._get_env("ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5)
        self.max_fetch_retries = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)
        self.trend_ema_period = self._get_env("TREND_EMA_PERIOD", "20", Fore.YELLOW, cast_type=int, min_val=5, max_val=500) # EMA period validation

        if not self.api_key or not self.api_secret:
            logger.critical(Fore.RED + Style.BRIGHT + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting.")
            sys.exit(1)

        # Validate market type
        if self.market_type not in ['linear', 'inverse']:
             logger.critical(f"{Fore.RED+Style.BRIGHT}Invalid MARKET_TYPE '{self.market_type}'. Must be 'linear' or 'inverse'. Halting.")
             sys.exit(1)
        logger.debug("Configuration loaded successfully.")

    def _get_env(self, key: str, default: Any, color: str, cast_type: type = str,
                 min_val: Optional[Union[int, Decimal]] = None,
                 max_val: Optional[Union[int, Decimal]] = None,
                 allowed_values: Optional[List[str]] = None) -> Any:
        """Gets value from environment, casts, validates, logs, and handles errors robustly."""
        value_str = os.getenv(key)
        is_default = False
        log_value = "****" if "SECRET" in key or "KEY" in key else value_str # Mask secrets

        if value_str is None or value_str.strip() == "": # Treat empty or whitespace-only string as not set
            value_str = None # Ensure it's None if not set
            if default is None:
                # If no default is specified and value isn't set, return None (e.g., for API keys check later)
                logger.debug(f"{color}{key} not set and no default provided.")
                return None
            else:
                value = default
                is_default = True
                logger.warning(f"{color}Using default value for {key}: {default}")
                # Use default value string for casting below
                value_str = str(default)
        else:
            logger.info(f"{color}Summoned {key}: {log_value}")
            value = value_str # Placeholder, will be overwritten by casted value

        # --- Casting ---
        casted_value = None
        try:
            if value_str is None: # Should only happen if default was None and value wasn't set
                casted_value = None
            elif cast_type == bool:
                casted_value = value_str.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                casted_value = Decimal(value_str)
            elif cast_type == int:
                casted_value = int(value_str)
            elif cast_type == float:
                # Generally avoid float for critical values, but allow if needed
                # Consider warning if float is used for price/amount related configs
                logger.warning(f"{Fore.YELLOW}Casting {key} to float. Consider using Decimal for financial precision.")
                casted_value = float(value_str)
            else: # Default is str
                casted_value = str(value_str)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Could not cast {key} ('{value_str}') to {cast_type.__name__}: {e}.")
            if default is not None:
                logger.warning(f"Attempting to use default value: {default}")
                # Attempt to cast the default value itself
                try:
                    if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                    if cast_type == Decimal: return Decimal(str(default))
                    if cast_type == int: return int(default)
                    if cast_type == float: return float(default)
                    return str(default)
                except (ValueError, TypeError, InvalidOperation) as default_e:
                    logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}: {default_e}. Halting.")
                    sys.exit(1)
            else:
                # No default value to fall back on
                logger.critical(f"{Fore.RED+Style.BRIGHT}No valid default provided for {key} after casting failure. Halting.")
                sys.exit(1)

        # --- Validation ---
        # Check if casted_value is None (could happen if original value_str was None and default was None)
        if casted_value is None and default is None:
            # This is acceptable if the config allows optional values (like API keys initially)
            return None

        validation_failed = False
        error_message = ""

        # Allowed values check (for strings like trigger types)
        if allowed_values and casted_value not in allowed_values:
            error_message = f"Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}."
            validation_failed = True

        # Min/Max checks (for numeric types)
        # Ensure comparison happens between compatible types (Decimal vs Decimal, int vs int)
        if not validation_failed and (isinstance(casted_value, (int, float, Decimal))):
            if min_val is not None:
                min_val_comp = Decimal(str(min_val)) if isinstance(casted_value, Decimal) else min_val
                if casted_value < min_val_comp:
                    error_message = f"{key} value {casted_value} is below minimum {min_val}."
                    validation_failed = True
            if max_val is not None:
                max_val_comp = Decimal(str(max_val)) if isinstance(casted_value, Decimal) else max_val
                if casted_value > max_val_comp:
                    error_message = f"{key} value {casted_value} is above maximum {max_val}."
                    validation_failed = True

        if validation_failed:
            logger.error(f"{Fore.RED}{error_message}")
            if default is not None:
                 logger.warning(f"Using default value: {default}")
                 # Re-cast default to ensure correct type is returned (already handled during initial cast failure, but good failsafe)
                 try:
                     if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                     if cast_type == Decimal: return Decimal(str(default))
                     if cast_type == int: return int(default)
                     if cast_type == float: return float(default)
                     return str(default)
                 except (ValueError, TypeError, InvalidOperation):
                     logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting.")
                     sys.exit(1)
            else:
                 logger.critical(f"{Fore.RED+Style.BRIGHT}No valid default provided for {key} after validation failure. Halting.")
                 sys.exit(1)

        return casted_value


CONFIG = TradingConfig()
MARKET_INFO: Optional[Dict] = None # Global to store market details after connection
EXCHANGE: Optional[ccxt.Exchange] = None # Global for the exchange instance

# --- Exchange Nexus Initialization ---
print(Fore.MAGENTA + Style.BRIGHT + "\nEstablishing Nexus with the Exchange v2.1.2...")
try:
    exchange_options = {
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True, # CCXT built-in rate limiter
        "options": {
            'defaultType': 'swap', # More specific for futures/swaps than 'future'
            'defaultSubType': CONFIG.market_type, # 'linear' or 'inverse'
            'adjustForTimeDifference': True, # Auto-sync clock with server
            # Bybit V5 API often requires 'category' for unified endpoints
            'brokerId': 'PyrmethusV212', # Custom identifier for Bybit API tracking
            # Explicitly set category for V5 requests. This is often crucial.
            'v5': {'category': CONFIG.market_type}
        }
    }
    # Log options excluding secrets for debugging
    log_options = exchange_options.copy()
    log_options['apiKey'] = '****'
    log_options['secret'] = '****'
    # Deep copy options to avoid modifying original if nested dicts exist
    import copy
    log_options['options'] = copy.deepcopy(exchange_options['options'])
    logger.debug(f"Initializing CCXT Bybit with options: {log_options}")

    EXCHANGE = ccxt.bybit(exchange_options)

    # Test connectivity and credentials (important!)
    logger.info("Verifying credentials and connection...")
    EXCHANGE.check_required_credentials() # Checks if keys are present/formatted ok
    logger.info("Credentials format check passed.")
    # Fetch time to verify connectivity, API key validity, and clock sync
    server_time = EXCHANGE.fetch_time()
    local_time = EXCHANGE.milliseconds()
    time_diff = abs(server_time - local_time)
    logger.info(f"Exchange time synchronized: {EXCHANGE.iso8601(server_time)} (Difference: {time_diff} ms)")
    if time_diff > 5000: # Warn if clock skew is significant (e.g., > 5 seconds)
        logger.warning(f"{Fore.YELLOW}Significant time difference ({time_diff} ms) between system and exchange. Check system clock synchronization (e.g., using NTP).")

    # Load markets (force reload to ensure fresh data)
    logger.info("Loading market spirits (market data)...")
    EXCHANGE.load_markets(True) # Force reload
    logger.info(Fore.GREEN + Style.BRIGHT + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets).")

    # Verify symbol exists and get market details
    if CONFIG.symbol not in EXCHANGE.markets:
         logger.error(Fore.RED + Style.BRIGHT + f"Symbol {CONFIG.symbol} not found in Bybit {CONFIG.market_type} market spirits.")
         # Suggest available symbols more effectively
         available_symbols = []
         try:
             # Extract quote/settle currency robustly (handles SYMBOL/QUOTE:SETTLE format)
             symbol_parts = CONFIG.symbol.split(':')
             quote_currency = symbol_parts[-1] # USDT from BTC/USDT:USDT
             base_currency = symbol_parts[0].split('/')[0] # BTC from BTC/USDT:USDT
         except IndexError:
             logger.error(f"Could not parse base/quote/settle from SYMBOL '{CONFIG.symbol}'.")
             base_currency, quote_currency = "UNKNOWN", "UNKNOWN"

         # Iterate through loaded markets
         for s, m in EXCHANGE.markets.items():
             # Check if market matches the configured type (linear/inverse) and is active
             is_correct_type = (CONFIG.market_type == 'linear' and m.get('linear')) or \
                               (CONFIG.market_type == 'inverse' and m.get('inverse'))
             # Filter by settle currency for relevance, also check if active and swap type
             if m.get('active') and m.get('swap') and is_correct_type and m.get('settle') == quote_currency:
                 available_symbols.append(s)

         suggestion_limit = 20
         if available_symbols:
             suggestions = ", ".join(sorted(available_symbols)[:suggestion_limit])
             if len(available_symbols) > suggestion_limit:
                 suggestions += "..."
             logger.info(Fore.CYAN + f"Available active {CONFIG.market_type} swap symbols settling in {quote_currency} (sample): " + suggestions)
         else:
             logger.info(Fore.CYAN + f"Could not find any active {CONFIG.market_type} swap symbols settling in {quote_currency} to suggest.")
         sys.exit(1)
    else:
        MARKET_INFO = EXCHANGE.market(CONFIG.symbol)
        logger.info(Fore.CYAN + f"Market spirit for {CONFIG.symbol} acknowledged (ID: {MARKET_INFO.get('id')}).")

        # --- Log key precision and limits using Decimal ---
        # Extract values safely, providing defaults or logging errors
        try:
            # Precision: 'price' (tick size), 'amount' (step size)
            price_precision = MARKET_INFO['precision'].get('price') # Usually tick size as string
            amount_precision = MARKET_INFO['precision'].get('amount') # Usually step size as string
            # Limits: 'amount' (min/max), 'cost' (min/max)
            min_amount = MARKET_INFO['limits']['amount'].get('min') # Minimum order size in base currency/contracts
            max_amount = MARKET_INFO['limits']['amount'].get('max') # Max order size
            min_cost = MARKET_INFO['limits'].get('cost', {}).get('min') # Minimum order value in quote currency
            max_cost = MARKET_INFO['limits'].get('cost', {}).get('max') # Max order value
            # Contract size (value of 1 contract, often in quote currency for linear, base for inverse?) -> Check CCXT/Bybit docs
            contract_size = MARKET_INFO.get('contractSize', '1') # Default to '1' if not present

            # Convert to Decimal for logging and potential use, handle None or invalid values
            price_prec_str = str(price_precision) if price_precision is not None else "N/A"
            amount_prec_str = str(amount_precision) if amount_precision is not None else "N/A"
            min_amount_dec = Decimal(str(min_amount)) if min_amount is not None else Decimal("NaN")
            max_amount_dec = Decimal(str(max_amount)) if max_amount is not None else Decimal("Infinity")
            min_cost_dec = Decimal(str(min_cost)) if min_cost is not None else Decimal("NaN")
            max_cost_dec = Decimal(str(max_cost)) if max_cost is not None else Decimal("Infinity")
            contract_size_dec = Decimal(str(contract_size)) if contract_size is not None else Decimal("NaN")

            logger.debug(f"Market Precision: Price Tick Size={price_prec_str}, Amount Step Size={amount_prec_str}")
            logger.debug(f"Market Limits: Min Amount={min_amount_dec}, Max Amount={max_amount_dec}")
            logger.debug(f"Market Limits: Min Cost={min_cost_dec}, Max Cost={max_cost_dec}")
            logger.debug(f"Contract Size: {contract_size_dec}")

            # --- Dynamically set epsilon based on amount precision (step size) ---
            if amount_precision is not None:
                try:
                    amount_step_dec = Decimal(str(amount_precision))
                    if amount_step_dec > 0:
                        # Use half the step size as epsilon for comparisons.
                        # This helps avoid issues where a position is slightly off the step size.
                        CONFIG.position_qty_epsilon = amount_step_dec / Decimal('2')
                        logger.info(f"Dynamically set position_qty_epsilon based on amount step size ({amount_step_dec}): {CONFIG.position_qty_epsilon:.2E}") # Use scientific notation
                    else:
                        logger.warning(f"Market amount step size '{amount_precision}' is zero or negative. Using default epsilon: {CONFIG.position_qty_epsilon:.2E}")
                except (InvalidOperation, TypeError):
                    logger.warning(f"Could not parse amount step size '{amount_precision}' to Decimal. Using default epsilon: {CONFIG.position_qty_epsilon:.2E}")
            else:
                 logger.warning(f"Market info does not provide amount step size ('precision.amount'). Using default epsilon: {CONFIG.position_qty_epsilon:.2E}")

        except (KeyError, TypeError, InvalidOperation) as e:
             logger.critical(f"{Fore.RED+Style.BRIGHT}Failed to parse critical market info (precision/limits/size) from MARKET_INFO: {e}. Halting.", exc_info=True)
             logger.debug(f"Problematic MARKET_INFO structure: {MARKET_INFO}")
             sys.exit(1)

except ccxt.AuthenticationError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Authentication failed! Check API Key/Secret validity and permissions. Error: {e}")
    sys.exit(1)
except ccxt.NetworkError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Network error connecting to Bybit: {e}. Check internet connection and Bybit status.")
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Bybit exchange is currently unavailable: {e}. Check Bybit status.")
    sys.exit(1)
except ccxt.ExchangeError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Exchange Nexus Error during initialization: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error during Nexus initialization: {e}", exc_info=True)
    sys.exit(1)


# --- Global State Runes ---
# Tracks active SL/TSL order IDs or position-based markers associated with a potential long or short position.
# Reset when a position is closed or a new entry order is successfully placed.
# Uses placeholders like "POS_SL_LONG" if SL/TSL is set on the position directly (common in Bybit V5).
order_tracker: Dict[str, Dict[str, Optional[str]]] = {
    "long": {"sl_id": None, "tsl_id": None},
    "short": {"sl_id": None, "tsl_id": None}
}

# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available) via termux-toast."""
    if not os.getenv("TERMUX_VERSION"):
        logger.debug("Not running in Termux environment. Skipping notification.")
        return

    try:
        # Check if command exists using which (more portable than 'command -v')
        check_cmd = subprocess.run(['which', 'termux-toast'], capture_output=True, text=True, check=False, encoding='utf-8')
        if check_cmd.returncode != 0:
            logger.debug("termux-toast command not found. Skipping notification.")
            return

        # Basic sanitization - focus on preventing shell interpretation issues
        # Replace potentially problematic characters with spaces or remove them
        # Allow basic punctuation like .,:!?() but remove shell command related chars
        safe_title = title.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('<', '').replace('>', '')
        safe_content = content.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('<', '').replace('>', '')

        # Limit length to avoid potential buffer issues or overly long toasts
        max_len = 200 # Increased max length slightly
        full_message = f"{safe_title}: {safe_content}"[:max_len]

        # Use list format for subprocess.run for security
        # Example styling: middle gravity, black text, green background, short duration
        cmd_list = ['termux-toast', '-g', 'middle', '-c', 'black', '-b', 'green', '-d', 'short', '-s', full_message]
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False,
