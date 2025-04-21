```python
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Termux Trading Spell (v2.1.1 - Precision Enhanced & Robust)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved robustness.

import os
import time
import logging
import sys
import subprocess # For termux-toast security
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation, DivisionByZero

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
    init(autoreset=True) # Initialize colorama for error messages
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell in your Termux terminal:")
    print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
    print(f"{Style.BRIGHT}pip install ccxt python-dotenv pandas numpy tabulate colorama requests{Style.RESET_ALL}")
    sys.exit(1)

# Weave the Colorama magic into the terminal
init(autoreset=True)

# Set Decimal precision (adjust if needed, higher precision means more memory/CPU)
# The default precision (usually 28) is often sufficient. Increase only if necessary.
# getcontext().prec = 30 # Example: Increase precision if needed for very small price increments

# --- Arcane Configuration ---
print(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.1.1...")

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
        self.position_qty_epsilon = Decimal("1E-9") # Default tiny Decimal, will be overridden

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
        """Gets value from environment, casts, validates, and logs."""
        value_str = os.getenv(key)
        is_default = False
        log_value = "****" if "SECRET" in key or "KEY" in key else value_str # Mask secrets

        if value_str is None or value_str == "": # Treat empty string as not set
            value = default
            is_default = True
            if default is not None:
                logger.warning(f"{color}Using default value for {key}: {default}")
            # Use default value string for casting below if needed
            value_str = str(default) if default is not None else None
        else:
             logger.info(f"{color}Summoned {key}: {log_value}")

        # Handle case where default is None and no value is set
        if value_str is None:
            if default is None:
                return None
            else:
                # This case should be covered by the is_default logic above, but double check
                logger.warning(f"{color}Value for {key} not found, using default: {default}")
                value = default
                value_str = str(default)

        # --- Casting ---
        casted_value = None
        try:
            if cast_type == bool:
                casted_value = value_str.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                casted_value = Decimal(value_str)
            elif cast_type == int:
                casted_value = int(value_str)
            elif cast_type == float:
                casted_value = float(value_str) # Generally avoid float for critical values, but allow if needed
            else: # Default is str
                casted_value = str(value_str)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Could not cast {key} ('{value_str}') to {cast_type.__name__}: {e}. Using default: {default}")
            # Attempt to cast the default value itself
            try:
                if default is None: return None
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting.")
                sys.exit(1)

        # --- Validation ---
        if casted_value is None: # Should not happen if casting succeeded or defaulted
             logger.critical(f"{Fore.RED+Style.BRIGHT}Failed to obtain a valid value for {key}. Halting.")
             sys.exit(1)

        # Allowed values check (for strings like trigger types)
        if allowed_values and casted_value not in allowed_values:
            logger.error(f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}. Using default: {default}")
            # Return default after logging error
            return default # Assume default is valid

        # Min/Max checks (for numeric types)
        validation_failed = False
        if min_val is not None:
            if (isinstance(casted_value, (int, float)) and casted_value < min_val) or \
               (isinstance(casted_value, Decimal) and casted_value < Decimal(str(min_val))):
                logger.error(f"{Fore.RED}{key} value {casted_value} is below minimum {min_val}. Using default: {default}")
                validation_failed = True
        if max_val is not None:
             if (isinstance(casted_value, (int, float)) and casted_value > max_val) or \
                (isinstance(casted_value, Decimal) and casted_value > Decimal(str(max_val))):
                 logger.error(f"{Fore.RED}{key} value {casted_value} is above maximum {max_val}. Using default: {default}")
                 validation_failed = True

        if validation_failed:
            # Re-cast default to ensure correct type is returned
            try:
                if default is None: return None
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting.")
                sys.exit(1)

        return casted_value


CONFIG = TradingConfig()
MARKET_INFO: Optional[Dict] = None # Global to store market details after connection
EXCHANGE: Optional[ccxt.Exchange] = None # Global for the exchange instance

# --- Exchange Nexus Initialization ---
print(Fore.MAGENTA + Style.BRIGHT + "\nEstablishing Nexus with the Exchange v2.1.1...")
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
            'brokerId': 'PyrmethusV211', # Custom identifier for Bybit API tracking
            'v5': {'category': CONFIG.market_type} # Explicitly set category for V5 requests
        }
    }
    # Log options excluding secrets for debugging
    log_options = exchange_options.copy()
    log_options['apiKey'] = '****'
    log_options['secret'] = '****'
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
        logger.warning(f"{Fore.YELLOW}Significant time difference ({time_diff} ms) between system and exchange. Check system clock synchronization.")

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
             # Extract quote currency robustly (handles SYMBOL/QUOTE:SETTLE format)
             quote_currency = CONFIG.symbol.split(':')[-1] # USDT from BTC/USDT:USDT
             base_currency = CONFIG.symbol.split('/')[0] # BTC from BTC/USDT:USDT
         except IndexError:
             logger.error(f"Could not parse base/quote from SYMBOL '{CONFIG.symbol}'.")
             base_currency, quote_currency = "UNKNOWN", "UNKNOWN"

         for s, m in EXCHANGE.markets.items():
             # Check if market matches the configured type (linear/inverse) and is active
             is_correct_type = (CONFIG.market_type == 'linear' and m.get('linear')) or \
                               (CONFIG.market_type == 'inverse' and m.get('inverse'))
             # Filter by quote currency for relevance, also check if active
             if m.get('active') and is_correct_type and m.get('settle') == quote_currency: # Use 'settle' for futures quote
                 available_symbols.append(s)

         suggestion_limit = 20
         if available_symbols:
             suggestions = ", ".join(sorted(available_symbols)[:suggestion_limit])
             if len(available_symbols) > suggestion_limit:
                 suggestions += "..."
             logger.info(Fore.CYAN + f"Available active {CONFIG.market_type} symbols settling in {quote_currency} (sample): " + suggestions)
         else:
             logger.info(Fore.CYAN + f"Could not find any active {CONFIG.market_type} symbols settling in {quote_currency} to suggest.")
         sys.exit(1)
    else:
        MARKET_INFO = EXCHANGE.market(CONFIG.symbol)
        logger.info(Fore.CYAN + f"Market spirit for {CONFIG.symbol} acknowledged (ID: {MARKET_INFO.get('id')}).")

        # --- Log key precision and limits using Decimal ---
        # Extract values safely, providing defaults or logging errors
        try:
            price_precision = MARKET_INFO['precision']['price'] # Usually number of decimal places or tick size
            amount_precision = MARKET_INFO['precision']['amount'] # Usually number of decimal places or step size
            min_amount = MARKET_INFO['limits']['amount']['min']
            max_amount = MARKET_INFO['limits']['amount'].get('max') # Max might be None
            contract_size = MARKET_INFO.get('contractSize', '1') # Default to '1' if not present
            min_cost = MARKET_INFO['limits'].get('cost', {}).get('min') # Min cost might not exist

            # Convert to Decimal for logging and potential use, handle None
            price_prec_str = str(price_precision) if price_precision is not None else "N/A"
            amount_prec_str = str(amount_precision) if amount_precision is not None else "N/A"
            min_amount_dec = Decimal(str(min_amount)) if min_amount is not None else Decimal("NaN")
            max_amount_dec = Decimal(str(max_amount)) if max_amount is not None else Decimal("Infinity")
            contract_size_dec = Decimal(str(contract_size)) if contract_size is not None else Decimal("NaN")
            min_cost_dec = Decimal(str(min_cost)) if min_cost is not None else Decimal("NaN")

            logger.debug(f"Market Precision: Price Tick/Decimals={price_prec_str}, Amount Step/Decimals={amount_prec_str}")
            logger.debug(f"Market Limits: Min Amount={min_amount_dec}, Max Amount={max_amount_dec}, Min Cost={min_cost_dec}")
            logger.debug(f"Contract Size: {contract_size_dec}")

            # --- Dynamically set epsilon based on amount precision (step size) ---
            # CCXT often provides amount precision as the step size directly
            amount_step_size = MARKET_INFO['precision'].get('amount')
            if amount_step_size is not None:
                try:
                    amount_step_dec = Decimal(str(amount_step_size))
                    # Use half the step size as epsilon for comparisons
                    CONFIG.position_qty_epsilon = amount_step_dec / Decimal('2')
                    logger.info(f"Dynamically set position_qty_epsilon based on amount step size ({amount_step_dec}): {CONFIG.position_qty_epsilon}")
                except (InvalidOperation, TypeError):
                    logger.warning(f"Could not parse amount step size '{amount_step_size}' to Decimal. Using default epsilon: {CONFIG.position_qty_epsilon}")
            else:
                 logger.warning(f"Market info does not provide amount step size ('precision.amount'). Using default epsilon: {CONFIG.position_qty_epsilon}")

        except (KeyError, TypeError, InvalidOperation) as e:
             logger.critical(f"{Fore.RED+Style.BRIGHT}Failed to parse critical market info (precision/limits/size) from MARKET_INFO: {e}. Halting.", exc_info=True)
             logger.debug(f"Problematic MARKET_INFO: {MARKET_INFO}")
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
        check_cmd = subprocess.run(['which', 'termux-toast'], capture_output=True, text=True, check=False)
        if check_cmd.returncode != 0:
            logger.debug("termux-toast command not found. Skipping notification.")
            return

        # Basic sanitization - focus on preventing shell interpretation issues
        # Replace potentially problematic characters with spaces or remove them
        safe_title = title.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('(', '').replace(')', '')
        safe_content = content.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('(', '').replace(')', '')

        # Limit length to avoid potential buffer issues or overly long toasts
        max_len = 150
        full_message = f"{safe_title}: {safe_content}"[:max_len]

        # Use list format for subprocess.run for security
        cmd_list = ['termux-toast', '-g', 'middle', '-c', 'black', '-b', 'green', '-s', full_message] # Example styling
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=5) # Add timeout

        if result.returncode != 0:
            # Log stderr if available
            stderr_msg = result.stderr.strip()
            logger.warning(f"termux-toast command failed with code {result.returncode}" + (f": {stderr_msg}" if stderr_msg else ""))
        # No else needed, success is silent

    except FileNotFoundError:
         logger.debug("termux-toast command not found (FileNotFoundError). Skipping notification.")
    except subprocess.TimeoutExpired:
         logger.warning("termux-toast command timed out. Skipping notification.")
    except Exception as e:
        # Catch other potential exceptions during subprocess execution
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}", exc_info=True)

# --- Precision Casting Spells ---

def format_price(symbol: str, price: Union[float, Decimal, str]) -> str:
    """Formats price according to market precision rules using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format price.")
        # Fallback with a reasonable number of decimal places
        try:
            return f"{Decimal(str(price)):.8f}"
        except Exception:
            return str(price) # Last resort

    try:
        # CCXT's price_to_precision handles rounding/truncation based on market rules (tick size).
        # Ensure input is float as expected by CCXT methods.
        price_float = float(price)
        return EXCHANGE.price_to_precision(symbol, price_float)
    except (AttributeError, KeyError) as e:
         logger.error(f"{Fore.RED}Market info for {symbol} missing precision data: {e}. Using fallback formatting.")
         return f"{Decimal(str(price)):.8f}" # Fallback
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}. Using fallback.")
        try:
            return f"{Decimal(str(price)):.8f}"
        except Exception:
             return str(price)

def format_amount(symbol: str, amount: Union[float, Decimal, str], rounding_mode=ROUND_DOWN) -> str:
    """Formats amount according to market precision rules (step size) using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format amount.")
        # Fallback with a reasonable number of decimal places
        try:
            # Use quantize for fallback if Decimal input
            if isinstance(amount, Decimal):
                 return str(amount.quantize(Decimal("1E-8"), rounding=rounding_mode))
            else:
                 return f"{Decimal(str(amount)):.8f}" # Less precise fallback
        except Exception:
            return str(amount) # Last resort

    try:
        # CCXT's amount_to_precision handles step size and rounding.
        # Map Python Decimal rounding modes to CCXT rounding modes if needed.
        ccxt_rounding_mode = ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND # Basic mapping
        # Ensure input is float as expected by CCXT methods.
        amount_float = float(amount)
        return EXCHANGE.amount_to_precision(symbol, amount_float, rounding_mode=ccxt_rounding_mode)
    except (AttributeError, KeyError) as e:
         logger.error(f"{Fore.RED}Market info for {symbol} missing precision data: {e}. Using fallback formatting.")
         if isinstance(amount, Decimal):
              return str(amount.quantize(Decimal("1E-8"), rounding=rounding_mode))
         else:
              return f"{Decimal(str(amount)):.8f}"
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}. Using fallback.")
        try:
             if isinstance(amount, Decimal):
                  return str(amount.quantize(Decimal("1E-8"), rounding=rounding_mode))
             else:
                  return f"{Decimal(str(amount)):.8f}"
        except Exception:
            return str(amount)

# --- Core Spell Functions ---

def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    """Generic wrapper to fetch data with retries and exponential backoff."""
    global EXCHANGE
    if EXCHANGE is None:
        logger.critical("Exchange object is None, cannot fetch data.")
        return None # Indicate critical failure

    last_exception = None
    # Add category param automatically for V5 if not already present in kwargs['params']
    if 'params' not in kwargs:
        kwargs['params'] = {}
    if 'category' not in kwargs['params'] and hasattr(EXCHANGE, 'options') and 'v5' in EXCHANGE.options and 'category' in EXCHANGE.options['v5']:
         kwargs['params']['category'] = EXCHANGE.options['v5']['category']
         # logger.debug(f"Auto-added category '{kwargs['params']['category']}' to params for {fetch_function.__name__}")

    for attempt in range(CONFIG.max_fetch_retries + 1): # +1 to allow logging final failure
        try:
            # Log the attempt number and function being called at DEBUG level
            logger.debug(f"Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}: Calling {fetch_function.__name__} with args={args}, kwargs={kwargs}")
            result = fetch_function(*args, **kwargs)
            # Optional: Basic validation for common failure patterns (e.g., empty list when data expected)
            # if isinstance(result, list) and not result and fetch_function.__name__ == 'fetch_ohlcv':
            #     logger.warning(f"{fetch_function.__name__} returned empty list, might indicate issue.")
            return result # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = 2 ** attempt # Exponential backoff
            logger.warning(Fore.YELLOW + f"{fetch_function.__name__}: Network issue (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}). Retrying in {wait_time}s... Error: {e}")
            if attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            else:
                logger.error(Fore.RED + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to network issues.")
        except ccxt.ExchangeNotAvailable as e:
             last_exception = e
             logger.error(Fore.RED + f"{fetch_function.__name__}: Exchange not available: {e}. Stopping retries.")
             break # No point retrying if exchange is down
        except ccxt.AuthenticationError as e:
             last_exception = e
             logger.critical(Fore.RED + Style.BRIGHT + f"{fetch_function.__name__}: Authentication error: {e}. Halting script.")
             # Trigger immediate shutdown (or let main loop handle exit)
             # graceful_shutdown() # Be careful calling this from deep within; sys.exit might be safer.
             sys.exit(1) # Critical error, stop immediately
        except ccxt.OrderNotFound as e:
            # Specific handling for OrderNotFound - don't retry, just return None or re-raise
            last_exception = e
            logger.warning(f"{fetch_function.__name__}: Order not found: {e}. Stopping retries for this call.")
            raise e # Re-raise OrderNotFound so calling functions can handle it specifically
        except ccxt.InsufficientFunds as e:
             # Specific handling for InsufficientFunds - usually not retryable
             last_exception = e
             logger.error(f"{fetch_function.__name__}: Insufficient funds: {e}. Stopping retries for this call.")
             raise e # Re-raise InsufficientFunds
        except ccxt.InvalidOrder as e:
             # Specific handling for InvalidOrder - usually not retryable
             last_exception = e
             logger.error(f"{fetch_function.__name__}: Invalid order parameters or state: {e}. Stopping retries for this call.")
             raise e # Re-raise InvalidOrder
        except ccxt.ExchangeError as e:
            # Includes rate limit errors, potentially invalid requests etc.
            last_exception = e
            # Check for specific retryable Bybit error codes if needed (e.g., 10006=timeout, 10016=internal error)
            # Bybit V5 Rate limit codes: 10018 (IP), 10017 (Key), etc.
            error_code = getattr(e, 'code', None) # CCXT might parse the code
            error_message = str(e)
            should_retry = True
            wait_time = 2 * (attempt + 1) # Default backoff

            # Check for common rate limit patterns / codes
            if "Rate limit exceeded" in error_message or error_code in [10017, 10018]:
                 wait_time = 5 * (attempt + 1) # Longer wait for rate limits
                 logger.warning(f"{Fore.YELLOW}{fetch_function.__name__}: Rate limit hit (Code: {error_code}). Retrying in {wait_time}s...")
            # Check for specific non-retryable errors (e.g., invalid parameter codes)
            # Example Bybit V5 codes: 110012 (Invalid symbol), 110007 (Qty too small), 110043 (SL/TP invalid)
            elif error_code in [110012, 110007, 110043, 10001]: # 10001 = Parameter error
                 logger.error(Fore.RED + f"{fetch_function.__name__}: Non-retryable exchange error (Code: {error_code}): {e}. Stopping retries.")
                 should_retry = False
            else:
                 # General exchange error, apply default backoff
                 logger.warning(f"{Fore.YELLOW}{fetch_function.__name__}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}, Code: {error_code}). Retrying in {wait_time}s... Error: {e}")

            if should_retry and attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            elif should_retry: # Final attempt failed
                 logger.error(Fore.RED + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to exchange errors.")
                 break # Exit retry loop
            else: # Non-retryable error encountered
                 break # Exit retry loop

        except Exception as e:
            # Catch-all for unexpected errors
            last_exception = e
            logger.error(Fore.RED + f"{fetch_function.__name__}: Unexpected shadow encountered: {e}", exc_info=True)
            break # Stop on unexpected errors

    # If loop finished without returning, it means all retries failed or a break occurred
    if last_exception:
        logger.error(f"{fetch_function.__name__} ultimately failed. Last exception type: {type(last_exception).__name__}, Message: {last_exception}")
    # Re-raise the last critical exception if it wasn't handled (like OrderNotFound)
    if isinstance(last_exception, (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder)):
        raise last_exception # Propagate specific non-retryable errors

    return None # Indicate failure for retryable errors or unexpected issues

def fetch_market_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data using the retry wrapper and perform validation."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})...")

    if EXCHANGE is None or not hasattr(EXCHANGE, 'fetch_ohlcv'):
         logger.error(Fore.RED + "Exchange object not properly initialized or missing fetch_ohlcv.")
         return None

    # Ensure limit is positive (already validated in config, but double check)
    if limit <= 0:
         logger.error(f"Invalid OHLCV limit requested: {limit}. Using default 100.")
         limit = 100

    ohlcv_data = None
    try:
        ohlcv_data = fetch_with_retries(EXCHANGE.fetch_ohlcv, symbol, timeframe, limit=limit)
    except Exception as e:
        # fetch_with_retries should handle most errors, but catch any unexpected ones here
        logger.error(Fore.RED + f"Unhandled exception during fetch_ohlcv call via fetch_with_retries: {e}", exc_info=True)
        return None

    if ohlcv_data is None:
        # fetch_with_retries already logged the failure reason
        logger.error(Fore.RED + f"Failed to fetch OHLCV data for {symbol}.")
        return None
    if not isinstance(ohlcv_data, list) or not ohlcv_data:
        logger.error(Fore.RED + f"Received empty or invalid OHLCV data type: {type(ohlcv_data)}. Content: {str(ohlcv_data)[:100]}")
        return None

    try:
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp immediately to UTC datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True) # Drop rows where timestamp conversion failed

        # Convert numeric columns, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for NaNs in critical price/volume columns *after* conversion
        initial_len = len(df)
        # Volume can sometimes be NaN or zero, decide if it's critical
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len:
            dropped_count = initial_len - len(df)
            logger.warning(f"Dropped {dropped_count} rows with missing essential price data from OHLCV.")

        if df.empty:
            logger.error(Fore.RED + "DataFrame is empty after processing OHLCV data (all rows dropped?).")
            return None

        df = df.set_index("timestamp")
        # Ensure data is sorted chronologically (fetch_ohlcv usually guarantees this, but verify)
        if not df.index.is_monotonic_increasing:
             logger.warning("OHLCV data was not sorted chronologically. Sorting now.")
             df.sort_index(inplace=True)

        # Check for duplicate timestamps (can indicate data issues)
        if df.index.duplicated().any():
             duplicates = df.index[df.index.duplicated()].unique()
             logger.warning(Fore.YELLOW + f"Duplicate timestamps found in OHLCV data ({len(duplicates)} unique duplicates). Keeping last entry for each.")
             df = df[~df.index.duplicated(keep='last')]

        # Check time difference between last two candles vs expected interval
        if len(df) > 1:
             time_diff = df.index[-1] - df.index[-2]
             try:
                 # Use pandas to parse timeframe string robustly
                 expected_interval_td = pd.Timedelta(EXCHANGE.parse_timeframe(timeframe), unit='s')
                 # Allow some tolerance (e.g., 10% of interval) for minor timing differences
                 tolerance = expected_interval_td * 0.1
                 if abs(time_diff - expected_interval_td) > tolerance:
                      logger.warning(f"Unexpected time gap between last two candles: {time_diff} (expected ~{expected_interval_td})")
             except ValueError:
                 logger.warning(f"Could not parse timeframe '{timeframe}' to calculate expected interval.")

        logger.info(Fore.GREEN + f"Market whispers received ({len(df)} candles). Latest: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return df
    except Exception as e:
        logger.error(Fore.RED + f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
        return None


def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Decimal]]:
    """Calculate technical indicators, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    if df is None or df.empty:
        logger.error(Fore.RED + "Cannot calculate indicators on missing or empty DataFrame.")
        return None
    try:
        # Ensure data is float for TA-Lib / Pandas calculations, convert to Decimal at the end
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # --- Indicator Periods (Consider making these configurable if desired) ---
        fast_ema_period = 8
        slow_ema_period = 12
        confirm_ema_period = 5 # Currently unused in signals, but calculated
        trend_ema_period = CONFIG.trend_ema_period
        stoch_period = 10
        smooth_k = 3
        smooth_d = 3
        atr_period = 10

        # --- Check Data Length Requirements ---
        required_len_ema = max(fast_ema_period, slow_ema_period, trend_ema_period, confirm_ema_period)
        # Stochastic requires period + smooth_k - 1 for %K, then + smooth_d - 1 for %D
        required_len_stoch = stoch_period + smooth_k + smooth_d - 2 # Correct calculation
        # ATR needs atr_period + 1 for initial calculation (TR needs previous close)
        required_len_atr = atr_period + 1 # Minimum for basic TR calc

        min_required_len = max(required_len_ema, required_len_stoch, required_len_atr)

        if len(df) < min_required_len:
             logger.error(f"{Fore.RED}Not enough data ({len(df)}) for all indicators (minimum required: {min_required_len}). Increase OHLCV_LIMIT or wait for more data.")
             return None
        # Add warnings if length is just barely enough for some indicators, as initial values can be less reliable
        elif len(df) < required_len_ema + 5: # Example buffer
             logger.warning(f"Data length ({len(df)}) is close to minimum required for EMAs ({required_len_ema}). Initial values might be less reliable.")
        # Similar warnings for Stoch and ATR if desired

        # --- Calculations using Pandas ---
        fast_ema_series = close.ewm(span=fast_ema_period, adjust=False).mean()
        slow_ema_series = close.ewm(span=slow_ema_period, adjust=False).mean()
        trend_ema_series = close.ewm(span=trend_ema_period, adjust=False).mean()
        confirm_ema_series = close.ewm(span=confirm_ema_period, adjust=False).mean()

        # Stochastic Oscillator %K and %D
        low_min = low.rolling(window=stoch_period).min()
        high_max = high.rolling(window=stoch_period).max()
        # Add epsilon to prevent division by zero if high == low
        stoch_k_raw = 100 * (close - low_min) / (high_max - low_min + 1e-12)
        stoch_k = stoch_k_raw.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()

        # Average True Range (ATR) - Wilder's smoothing matches TradingView standard
        tr_df = pd.DataFrame(index=df.index)
        tr_df["hl"] = high - low
        tr_df["hc"] = (high - close.shift()).abs()
        tr_df["lc"] = (low - close.shift()).abs()
        tr = tr_df[["hl", "hc", "lc"]].max(axis=1)
        # Use ewm with alpha = 1/period for Wilder's smoothing
        atr_series = tr.ewm(alpha=1/atr_period, adjust=False).mean()

        # --- Extract Latest Values & Convert to Decimal ---
        # Define quantizers for consistent decimal places (adjust as needed)
        price_quantizer = Decimal("1E-8") # 8 decimal places for price-like values
        percent_quantizer = Decimal("1E-2") # 2 decimal places for Stoch
        atr_quantizer = Decimal("1E-8") # 8 decimal places for ATR

        # Helper to safely get latest non-NaN value, convert to Decimal, and handle errors
        def get_latest_decimal(series: pd.Series, quantizer: Decimal, name: str, default_val: Decimal = Decimal("NaN")) -> Decimal:
            if series.empty or series.isna().all():
                logger.warning(f"Indicator series '{name}' is empty or all NaN.")
                return default_val
            # Get the last valid (non-NaN) value
            latest_valid_val = series.dropna().iloc[-1] if not series.dropna().empty else None

            if latest_valid_val is None:
                 logger.warning(f"Indicator calculation for '{name}' resulted in NaN or only NaNs.")
                 return default_val
            try:
                # Convert via string for precision, then quantize
                return Decimal(str(latest_valid_val)).quantize(quantizer)
            except (InvalidOperation, TypeError) as e:
                logger.error(f"Could not convert indicator '{name}' value {latest_valid_val} to Decimal: {e}. Returning default.")
                return default_val

        indicators_out = {
            "fast_ema": get_latest_decimal(fast_ema_series, price_quantizer, "fast_ema"),
            "slow_ema": get_latest_decimal(slow_ema_series, price_quantizer, "slow_ema"),
            "trend_ema": get_latest_decimal(trend_ema_series, price_quantizer, "trend_ema"),
            "confirm_ema": get_latest_decimal(confirm_ema_series, price_quantizer, "confirm_ema"),
            "stoch_k": get_latest_decimal(stoch_k, percent_quantizer, "stoch_k", default_val=Decimal("50.00")), # Default neutral
            "stoch_d": get_latest_decimal(stoch_d, percent_quantizer, "stoch_d", default_val=Decimal("50.00")), # Default neutral
            "atr": get_latest_decimal(atr_series, atr_quantizer, "atr", default_val=Decimal("0.0")) # Default zero
        }

        # Check if any crucial indicator calculation failed (returned NaN default)
        critical_indicators = ['fast_ema', 'slow_ema', 'trend_ema', 'stoch_k', 'stoch_d', 'atr']
        failed_indicators = [key for key in critical_indicators if indicators_out[key].is_nan()]

        if failed_indicators:
             logger.error(f"{Fore.RED}One or more critical indicators failed to calculate (NaN): {', '.join(failed_indicators)}")
             return None # Signal failure

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        # logger.debug(f"Latest Indicators: { {k: v for k, v in indicators_out.items()} }") # Log values at debug
        return indicators_out

    except Exception as e:
        logger.error(Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True)
        return None

def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch current positions using retry wrapper, returning quantities and prices as Decimals."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"# Consulting position spirits for {symbol}...")

    if EXCHANGE is None:
         logger.error("Exchange object not available for fetching positions.")
         return None

    # Initialize with Decimal zero for clarity
    pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0"), "liq_price": Decimal("0.0"), "pnl": Decimal("0.0")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0"), "liq_price": Decimal("0.0"), "pnl": Decimal("0.0")}
    }

    positions_data = None
    try:
        # fetch_with_retries handles category param automatically
        positions_data = fetch_with_retries(EXCHANGE.fetch_positions, symbols=[symbol])
    except Exception as e:
        # Handle potential exceptions raised by fetch_with_retries itself (e.g., AuthenticationError)
        logger.error(Fore.RED + f"Unhandled exception during fetch_positions call via fetch_with_retries: {e}", exc_info=True)
        return None # Indicate failure

    if positions_data is None:
         # fetch_with_retries already logged the failure reason
         logger.error(Fore.RED + f"Failed to fetch positions for {symbol}.")
         return None # Indicate failure

    if not isinstance(positions_data, list):
         logger.error(f"Unexpected data type received from fetch_positions: {type(positions_data)}. Expected list. Data: {str(positions_data)[:200]}")
         return None

    if not positions_data:
         logger.info(Fore.BLUE + f"No open positions reported by exchange for {symbol}.")
         return pos_dict # Return the initialized zero dictionary

    # Process the fetched positions
    active_positions_found = 0
    for pos in positions_data:
        # Ensure pos is a dictionary
        if not isinstance(pos, dict):
            logger.warning(f"Skipping non-dictionary item in positions data: {pos}")
            continue

        pos_symbol = pos.get('symbol')
        if pos_symbol != symbol:
            logger.debug(f"Ignoring position data for different symbol: {pos_symbol}")
            continue

        # Use info dictionary for safer access to raw exchange data if needed
        pos_info = pos.get('info', {})
        if not isinstance(pos_info, dict): # Ensure info is a dict
            pos_info = {}

        # Determine side ('long' or 'short') - check unified field first, then 'info'
        side = pos.get("side") # Unified field
        if side not in ["long", "short"]:
            # Fallback for Bybit V5 'info' field if unified 'side' is missing/invalid
            side_raw = pos_info.get("side", "").lower() # e.g., "Buy" or "Sell"
            if side_raw == "buy": side = "long"
            elif side_raw == "sell": side = "short"
            else:
                 logger.warning(f"Could not determine side for position: Info={str(pos_info)[:100]}. Skipping.")
                 continue

        # Get quantity ('contracts' or 'size') - Use unified field first, fallback to info
        contracts_str = pos.get("contracts") # Unified field ('contracts' seems standard)
        if contracts_str is None:
            contracts_str = pos_info.get("size") # Common Bybit V5 field in 'info'

        # Get entry price - Use unified field first, fallback to info
        entry_price_str = pos.get("entryPrice")
        if entry_price_str is None:
            # Check 'avgPrice' (common in V5) or 'entryPrice' in info
            entry_price_str = pos_info.get("avgPrice", pos_info.get("entryPrice"))

        # Get Liq Price and PnL (these are less standardized, rely more on unified fields if available)
        liq_price_str = pos.get("liquidationPrice")
        if liq_price_str is None:
            liq_price_str = pos_info.get("liqPrice")

        pnl_str = pos.get("unrealizedPnl")
        if pnl_str is None:
             # Check Bybit specific info fields
             pnl_str = pos_info.get("unrealisedPnl", pos_info.get("unrealizedPnl"))


        # --- Convert to Decimal and Store ---
        if side in pos_dict and contracts_str is not None:
            try:
                # Convert via string for precision
                contracts = Decimal(str(contracts_str))

                # Use epsilon check for effectively zero positions
                if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                    logger.debug(f"Ignoring effectively zero size {side} position for {symbol} (Qty: {contracts}).")
                    continue

                # Convert other fields, handling potential None or invalid values
                entry_price = Decimal(str(entry_price_str)) if entry_price_str is not None else Decimal("NaN")
                liq_price = Decimal(str(liq_price_str)) if liq_price_str is not None else Decimal("NaN")
                pnl = Decimal(str(pnl_str)) if pnl_str is not None else Decimal("NaN")

                # Assign to the dictionary
                pos_dict[side]["qty"] = contracts
                pos_dict[side]["entry_price"] = entry_price
                pos_dict[side]["liq_price"] = liq_price
                pos_dict[side]["pnl"] = pnl

                # Log with formatted decimals
                entry_log = f"{entry_price:.4f}" if not entry_price.is_nan() else "N/A"
                liq_log = f"{liq_price:.4f}" if not liq_price.is_nan() else "N/A"
                pnl_log = f"{pnl:+.4f}" if not pnl.is_nan() else "N/A"
                logger.info(Fore.YELLOW + f"Found active {side.upper()} position: Qty={contracts}, Entry={entry_log}, Liq≈{liq_log}, PnL≈{pnl_log}")
                active_positions_found += 1

            except (InvalidOperation, TypeError) as e:
                 logger.error(f"Could not parse position data for {side} side: Qty='{contracts_str}', Entry='{entry_price_str}', Liq='{liq_price_str}', PnL='{pnl_str}'. Error: {e}")
                 continue # Skip this potentially corrupt position entry
        elif side not in pos_dict:
            logger.warning(f"Position data found for unknown side '{side}'. Skipping.")

    if active_positions_found == 0:
         logger.info(Fore.BLUE + f"No active non-zero positions found for {symbol} after filtering.")

    logger.info(Fore.GREEN + "Position spirits consulted.")
    return pos_dict


def get_balance(currency: str = "USDT") -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Fetches the free and total balance for a specific currency using retry wrapper, returns Decimals."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching balance.")
        return None, None

    balance_data = None
    try:
        # Bybit V5 fetch_balance might need accountType (UNIFIED/CONTRACT) or coin.
        # CCXT's defaultType/SubType and category *should* handle this, but params might be needed.
        # Let's rely on fetch_with_retries to add category if configured.
        # params = {'accountType': 'UNIFIED'} # Example if needed for specific account types
        balance_data = fetch_with_retries(EXCHANGE.fetch_balance)
    except Exception as e:
        logger.error(Fore.RED + f"Unhandled exception during fetch_balance call via fetch_with_retries: {e}", exc_info=True)
        return None, None

    if balance_data is None:
        # fetch_with_retries already logged the failure
        logger.error(Fore.RED + f"Failed to fetch balance after retries. Cannot assess risk capital.")
        return None, None

    # --- Parse Balance Data ---
    free_balance = Decimal("NaN")
    total_balance = Decimal("NaN") # Represents Equity for futures

    try:
        # CCXT unified structure: balance_data[currency]['free'/'total']
        if currency in balance_data:
            currency_balance = balance_data[currency]
            free_str = currency_balance.get('free')
            total_str = currency_balance.get('total') # 'total' usually represents equity in futures

            if free_str is not None: free_balance = Decimal(str(free_str))
            if total_str is not None: total_balance = Decimal(str(total_str))

        # Alternative structure: balance_data['free'][currency], balance_data['total'][currency]
        elif 'free' in balance_data and isinstance(balance_data['free'], dict) and currency in balance_data['free']:
            free_str = balance_data['free'].get(currency)
            total_str = balance_data.get('total', {}).get(currency) # Total might still be top-level

            if free_str is not None: free_balance = Decimal(str(free_str))
            if total_str is not None: total_balance = Decimal(str(total_str))

        # Fallback: Check 'info' for exchange-specific structure (Bybit V5 example)
        elif 'info' in balance_data and isinstance(balance_data['info'], dict):
            info_data = balance_data['info']
            # V5 structure: result -> list -> account objects
            if 'result' in info_data and isinstance(info_data['result'], dict) and \
               'list' in info_data['result'] and isinstance(info_data['result']['list'], list):
                for account in info_data['result']['list']:
                    if isinstance(account, dict) and account.get('coin') == currency:
                        # Bybit V5 Unified Margin fields (check docs):
                        # 'walletBalance': Total assets in wallet
                        # 'availableToWithdraw': Amount withdrawable
                        # 'equity': Account equity (often the most relevant for risk)
                        # 'availableToBorrow': Margin specific
                        # 'totalPerpUPL': Unrealized PnL
                        equity_str = account.get('equity') # Use equity as 'total'
                        free_str = account.get('availableToWithdraw') # Use availableToWithdraw as 'free'

                        if free_str is not None: free_balance = Decimal(str(free_str))
                        if equity_str is not None: total_balance = Decimal(str(equity_str))
                        logger.debug(f"Parsed Bybit V5 info structure for {currency}: Free={free_balance}, Equity={total_balance}")
                        break # Found the currency account

        # If parsing failed, balances will remain NaN
        if free_balance.is_nan():
             logger.warning(f"Could not find or parse free balance for {currency} in balance data.")
        if total_balance.is_nan():
             logger.warning(f"Could not find or parse total/equity balance for {currency} in balance data.")
             # Critical if equity is needed for risk calc
             logger.error(Fore.RED + "Failed to determine account equity. Cannot proceed safely.")
             return None, None # Return None for equity if it couldn't be found

        # Use 'total' balance (Equity) as the primary value for risk calculation
        equity = total_balance

        logger.info(Fore.GREEN + f"Vault contains {free_balance:.4f} free {currency} (Equity/Total: {equity:.4f}).")
        return free_balance, equity # Return free and total (equity)

    except (InvalidOperation, TypeError, KeyError) as e:
         logger.error(Fore.RED + f"Error parsing balance data for {currency}: {e}. Raw keys: {list(balance_data.keys()) if isinstance(balance_data, dict) else 'N/A'}")
         logger.debug(f"Raw balance data: {balance_data}")
         return None, None # Indicate parsing failure
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected shadow encountered querying vault: {e}", exc_info=True)
        return None, None

def check_order_status(order_id: str, symbol: str, timeout: int = CONFIG.order_check_timeout_seconds) -> Optional[Dict]:
    """Checks order status with retries and timeout. Returns the final order dict or None."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"Verifying final status of order {order_id} for {symbol} (Timeout: {timeout}s)...")
    if EXCHANGE is None:
        logger.error("Exchange object not available for checking order status.")
        return None

    start_time = time.time()
    last_status = 'unknown'
    attempt = 0
    check_interval = 1.5 # seconds between checks

    while time.time() - start_time < timeout:
        attempt += 1
        logger.debug(f"Checking order {order_id}, attempt {attempt}...")
        order_status = None
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            # Category param should be handled automatically by fetch_with_retries
            order_status = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol)

            if order_status and isinstance(order_status, dict):
                last_status = order_status.get('status', 'unknown')
                filled_qty = order_status.get('filled', 0.0)
                logger.info(f"Order {order_id} status check: {last_status}, Filled: {filled_qty}")

                # Check for terminal states (fully filled, canceled, rejected, expired)
                # 'closed' usually means fully filled for market/limit orders.
                if last_status in ['closed', 'canceled', 'rejected', 'expired']:
                     logger.info(f"Order {order_id} reached terminal state: {last_status}.")
                     return order_status # Return the final order dict
                # If 'open' but fully filled (can happen briefly), treat as terminal 'closed'
                elif last_status == 'open' and order_status.get('remaining') == 0.0 and filled_qty > 0.0:
                    logger.info(f"Order {order_id} is 'open' but fully filled. Treating as 'closed'.")
                    order_status['status'] = 'closed' # Update status locally
                    return order_status

            else:
                # fetch_with_retries failed or returned unexpected data
                # Error logged within fetch_with_retries, just note it here
                logger.warning(f"fetch_order call failed or returned invalid data for {order_id}. Continuing check loop.")
                # Continue the loop to retry check_order_status itself

        except ccxt.OrderNotFound:
            # Order is definitively not found. This is a terminal state.
            logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND by exchange.")
            # Return a synthetic dict indicating 'notfound' status? Or None? Let's return None.
            return None # Explicitly indicate not found
        except Exception as e:
            # Catch any other unexpected error during the check itself
            logger.error(f"Unexpected error during order status check loop for {order_id}: {e}", exc_info=True)
            # Decide whether to retry or fail; retrying is part of the loop.

        # Wait before the next check_order_status attempt
        time_elapsed = time.time() - start_time
        if time_elapsed + check_interval < timeout:
            logger.debug(f"Order {order_id} status ({last_status}) not terminal, sleeping {check_interval:.1f}s...")
            time.sleep(check_interval)
            check_interval = min(check_interval * 1.2, 5) # Slightly increase interval up to 5s
        else:
            break # Exit loop if next sleep would exceed timeout

    # --- Timeout Reached ---
    logger.error(Fore.RED + f"Timed out checking status for order {order_id} after {timeout} seconds. Last known status: {last_status}.")
    # Attempt one final fetch outside the loop to get the very last state if possible
    final_check_status = None
    try:
        logger.info(f"Performing final status check for order {order_id} after timeout...")
        final_check_status = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol)
        if final_check_status:
             logger.info(f"Final status after timeout: {final_check_status.get('status', 'unknown')}")
             # Return this final status even if timed out earlier
             return final_check_status
        else:
             logger.error(f"Final status check for order {order_id} also failed.")
             return None # Indicate persistent failure to get status
    except ccxt.OrderNotFound:
        logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND on final check.")
        return None
    except Exception as e:
        logger.error(f"Error during final status check for order {order_id}: {e}")
        return None # Indicate failure


def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    trade_action = f"{side.upper()} Market Entry"
    logger.trade(Style.BRIGHT + f"Attempting {trade_action} for {symbol}...")

    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(Fore.RED + f"{trade_action} failed: Market info or Exchange not available.")
        return False

    # --- Pre-computation & Validation ---
    quote_currency = MARKET_INFO.get('settle', 'USDT') # Use settle currency (e.g., USDT)
    _, total_equity = get_balance(quote_currency)
    if total_equity is None or total_equity <= Decimal("0"):
        logger.error(Fore.RED + f"{trade_action} failed: Invalid or zero account equity ({total_equity}).")
        return False

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.error(Fore.RED + f"{trade_action} failed: Invalid ATR value ({atr}). Check indicator calculation.")
        return False

    # Fetch current ticker price using fetch_ticker with retries
    ticker_data = None
    try:
        ticker_data = fetch_with_retries(EXCHANGE.fetch_ticker, symbol)
    except Exception as e:
         logger.error(Fore.RED + f"{trade_action} failed: Unhandled exception fetching ticker: {e}")
         return False

    if not ticker_data or ticker_data.get("last") is None:
         logger.error(Fore.RED + f"{trade_action} failed: Cannot fetch current ticker price for sizing/SL calculation.")
         # fetch_with_retries should have logged details if it failed
         return False

    try:
        # Use 'last' price as current price estimate, convert to Decimal
        price = Decimal(str(ticker_data["last"]))
        logger.debug(f"Current ticker price: {price:.4f} {quote_currency}")

        # --- Calculate Stop Loss Price ---
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if sl_distance_points <= Decimal(0):
             logger.error(f"{Fore.RED}{trade_action} failed: Stop distance calculation resulted in zero or negative value ({sl_distance_points}). Check ATR ({atr:.6f}) and multiplier ({CONFIG.sl_atr_multiplier}).")
             return False

        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else: # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        sl_price = Decimal(sl_price_formatted_str) # Use the formatted price as Decimal
        logger.debug(f"ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier}, SL Distance Points: {sl_distance_points:.6f}")
        logger.debug(f"Raw SL Price: {sl_price_raw:.6f}, Formatted SL Price for API: {sl_price_formatted_str}")

        # Sanity check SL placement relative to current price
        # Use price tick size for tolerance if available, else small Decimal
        price_tick_size = Decimal(str(MARKET_INFO['precision'].get('price', '0.000001')))
        if side == "buy" and sl_price >= price - price_tick_size:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated SL price ({sl_price}) is too close to or above current price ({price}). Check ATR/multiplier or price feed. Aborting.")
            return False
        if side == "sell" and sl_price <= price + price_tick_size:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated SL price ({sl_price}) is too close to or below current price ({price}). Check ATR/multiplier or price feed. Aborting.")
            return False

        # --- Calculate Position Size ---
        risk_amount_quote = total_equity * risk_percentage
        # Stop distance in quote currency (use absolute difference)
        stop_distance_quote = (price - sl_price).copy_abs()

        if stop_distance_quote <= Decimal("0"):
             logger.error(Fore.RED + f"{trade_action} failed: Stop distance in quote currency is zero or negative ({stop_distance_quote}). Check ATR, multiplier, or market precision. Cannot calculate size.")
             return False

        # Calculate quantity based on contract size and linear/inverse type
        contract_size = Decimal(str(MARKET_INFO.get('contractSize', '1')))
        qty_raw = Decimal('0')

        # --- Sizing Logic ---
        if CONFIG.market_type == 'linear':
            # Linear (e.g., BTC/USDT:USDT): Size is in Base currency (BTC). Value = Size * Price.
            # Risk Amount (Quote) = Qty (Base) * Stop Distance (Quote)
            # Qty (Base) = Risk Amount (Quote) / Stop Distance (Quote)
            qty_raw = risk_amount_quote / stop_distance_quote
            logger.debug(f"Linear Sizing: Qty (Base) = {risk_amount_quote:.4f} {quote_currency} / {stop_distance_quote:.4f} {quote_currency} = {qty_raw}")

        elif CONFIG.market_type == 'inverse':
            # Inverse (e.g., BTC/USD:BTC): Size is in Contracts (often USD value). Value = Size (Contracts) * ContractValue / Price.
            # Risk Amount (Quote = USD) = Qty (Contracts) * Contract Size (Base/Contract) * Stop Distance (Quote)
            # Qty (Contracts) = Risk Amount (Quote) / (Contract Size (Base/Contract) * Stop Distance (Quote))
            # Verify Contract Size interpretation! For Bybit BTC/USD, Contract Size is 1 USD.
            # So, Contract Size (Base/Contract) = 1 / Price (Quote/Base)
            # Risk (Quote) = Qty (Contracts) * (1 / Price) * Stop Distance (Quote)
            # Qty (Contracts) = Risk (Quote) * Price (Quote) / Stop Distance (Quote)
            if price <= Decimal("0"):
                logger.error(Fore.RED + f"{trade_action} failed: Cannot calculate inverse size with zero or negative price.")
                return False
            qty_raw = (risk_amount_quote * price) / stop_distance_quote
            logger.debug(f"Inverse Sizing (Contract Size = {contract_size} {quote_currency}): Qty (Contracts) = ({risk_amount_quote:.4f} * {price:.4f}) / {stop_distance_quote:.4f} = {qty_raw}")

        else:
            logger.error(f"{trade_action} failed: Unsupported market type for sizing: {CONFIG.market_type}")
            return False

        # --- Format and Validate Quantity ---
        # Format quantity according to market precision (ROUND_DOWN to be conservative)
        qty_formatted_str = format_amount(symbol, qty_raw, ROUND_DOWN)
        qty = Decimal(qty_formatted_str)
        logger.debug(f"Risk Amount: {risk_amount_quote:.4f} {quote_currency}, Stop Distance: {stop_distance_quote:.4f} {quote_currency}")
        logger.debug(f"Raw Qty: {qty_raw:.8f}, Formatted Qty (Rounded Down): {qty}")

        # Validate Quantity Against Market Limits
        min_qty_str = str(MARKET_INFO['limits']['amount']['min']) if MARKET_INFO['limits']['amount'].get('min') is not None else "0"
        max_qty_str = str(MARKET_INFO['limits']['amount']['max']) if MARKET_INFO['limits']['amount'].get('max') is not None else None
        min_qty = Decimal(min_qty_str)
        max_qty = Decimal(max_qty_str) if max_qty_str is not None else None

        # Use epsilon for zero check
        if qty < min_qty or qty < CONFIG.position_qty_epsilon:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated quantity ({qty}) is zero or below minimum ({min_qty}). Risk amount ({risk_amount_quote:.4f}), stop distance ({stop_distance_quote:.4f}), or equity might be too small. Cannot place order.")
            return False
        if max_qty is not None and qty > max_qty:
            logger.warning(Fore.YELLOW + f"Calculated quantity {qty} exceeds maximum {max_qty}. Capping order size to {max_qty}.")
            qty = max_qty # Use the Decimal max_qty
            # Re-format capped amount - crucial! Use ROUND_DOWN again.
            qty_formatted_str = format_amount(symbol, qty, ROUND_DOWN)
            qty = Decimal(qty_formatted_str)
            logger.info(f"Re-formatted capped Qty: {qty}")
            # Double check if capped value is now below min (unlikely but possible with large steps)
            if qty < min_qty or qty < CONFIG.position_qty_epsilon:
                 logger.error(Fore.RED + f"{trade_action} failed: Capped quantity ({qty}) is now below minimum ({min_qty}) or zero. Aborting.")
                 return False

        # Validate minimum cost if available
        min_cost_str = str(MARKET_INFO['limits'].get('cost', {}).get('min')) if MARKET_INFO['limits'].get('cost', {}).get('min') is not None else None
        if min_cost_str is not None:
            min_cost = Decimal(min_cost_str)
            estimated_cost = Decimal('0')
            # Estimate cost based on market type (Approximate!)
            try:
                if CONFIG.market_type == 'linear':
                     # Cost = Qty (Base) * Price (Quote/Base) = Quote
                     estimated_cost = qty * price
                elif CONFIG.market_type == 'inverse':
                     # Cost = Qty (Contracts) * Contract Size (Quote/Contract) = Quote
                     # Assuming contract size is in Quote currency (e.g., 1 USD for BTC/USD)
                     estimated_cost = qty * contract_size # Check if contract_size needs conversion if not in quote
                     logger.debug(f"Inverse cost estimation: Qty({qty}) * ContractSize({contract_size}) = {estimated_cost}")
                else:
                     estimated_cost = Decimal('0')

                if estimated_cost < min_cost:
                     logger.error(Fore.RED + f"{trade_action} failed: Estimated order cost/value ({estimated_cost:.4f} {quote_currency}) is below minimum required ({min_cost:.4f} {quote_currency}). Increase risk or equity. Cannot place order.")
                     return False
            except Exception as cost_err:
                 logger.warning(f"Could not estimate order cost: {cost_err}. Skipping min cost check.")

        logger.info(Fore.YELLOW + f"Calculated Order: Side={side.upper()}, Qty={qty}, Entry≈{price:.4f}, SL={sl_price} (ATR={atr:.4f})")

    except (InvalidOperation, TypeError, DivisionByZero, KeyError) as e:
         logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Error during pre-calculation/validation: {e}", exc_info=True)
         return False
    except Exception as e: # Catch any other unexpected errors
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during pre-calculation: {e}", exc_info=True)
        return False

    # --- Cast the Market Order Spell ---
    order = None
    order_id = None
    try:
        logger.trade(f"Submitting {side.upper()} market order for {qty} {symbol}...")
        # fetch_with_retries handles category param
        # CCXT expects float amount
        order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol,
            side=side,
            amount=float(qty)
        )

        if order is None:
            # fetch_with_retries logged the error
            logger.error(Fore.RED + f"{trade_action} failed: Market order placement failed after retries.")
            return False

        logger.debug(f"Market order raw response: {order}")
        order_id = order.get('id')
        if not order_id:
            # Check if order info contains success/error details even without ID
            if isinstance(order.get('info'), dict):
                 ret_code = order['info'].get('retCode')
                 ret_msg = order['info'].get('retMsg')
                 if ret_code == 0: # Bybit V5 success code
                      logger.warning(f"{trade_action}: Market order submitted successfully (retCode 0) but no Order ID returned in standard field. Proceeding cautiously.")
                      # Maybe generate a client order ID if needed? For now, proceed without a trackable ID.
                 else:
                      logger.error(Fore.RED + f"{trade_action} failed: Market order submission failed. Exchange message: {ret_msg} (Code: {ret_code})")
                      return False
            else:
                logger.error(Fore.RED + f"{trade_action} failed: Market order submission failed to return an ID or success info.")
                return False
        else:
             logger.trade(f"Market order submitted: ID {order_id}")

        # --- Verify Order Fill (Crucial Step) ---
        logger.info(f"Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}...")
        time.sleep(CONFIG.order_check_delay_seconds)

        # Use the dedicated check_order_status function
        order_status_data = check_order_status(order_id, symbol, timeout=CONFIG.order_check_timeout_seconds)

        filled_qty = Decimal("0.0")
        average_price = price # Fallback to estimated entry price
        order_final_status = 'unknown'

        if order_status_data and isinstance(order_status_data, dict):
            order_final_status = order_status_data.get('status', 'unknown')
            filled_str = order_status_data.get('filled')
            average_str = order_status_data.get('average') # Average fill price

            if filled_str is not None:
                try: filled_qty = Decimal(str(filled_str))
                except InvalidOperation: logger.error(f"Could not parse filled quantity '{filled_str}' to Decimal.")
            if average_str is not None:
                try:
                    avg_price_decimal = Decimal(str(average_str))
                    if avg_price_decimal > 0: # Use actual fill price only if valid
                         average_price = avg_price_decimal
                except InvalidOperation: logger.error(f"Could not parse average price '{average_str}' to Decimal.")

            logger.debug(f"Order {order_id} status check result: Status='{order_final_status}', Filled='{filled_qty}', AvgPrice='{average_price}'")

            # 'closed' means fully filled for market orders on Bybit
            if order_final_status == 'closed' and filled_qty >= CONFIG.position_qty_epsilon:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Order {order_id} confirmed FILLED: {filled_qty} @ {average_price:.4f}")
            # Handle partial fills (less common for market, but possible during high volatility)
            elif order_final_status == 'open' and filled_qty > CONFIG.position_qty_epsilon:
                 logger.warning(Fore.YELLOW + f"Market Order {order_id} status is 'open' but partially filled ({filled_qty}). This is unusual. Proceeding with filled amount.")
                 # Consider if action needed (e.g., cancel remainder? Bybit usually fills market fully or rejects). Assuming it will resolve.
            elif order_final_status in ['open', 'partially_filled'] and filled_qty < CONFIG.position_qty_epsilon:
                 logger.error(Fore.RED + f"{trade_action} failed: Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty}). Aborting SL placement.")
                 # Attempt to cancel just in case it's stuck
                 try:
                     logger.info(f"Attempting cancellation of stuck/unfilled order {order_id}.")
                     fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol)
                 except Exception as cancel_err: logger.warning(f"Failed to cancel stuck order {order_id}: {cancel_err}")
                 return False
            else: # canceled, rejected, expired, failed, unknown, or closed with zero fill
                 logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Order {order_id} did not fill successfully: Status '{order_final_status}', Filled Qty: {filled_qty}. Aborting SL placement.")
                 # Attempt to cancel if not already in a terminal state (defensive)
                 if order_final_status not in ['canceled', 'rejected', 'expired']:
                     try:
                          logger.info(f"Attempting cancellation of failed/unknown status order {order_id}.")
                          fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol)
                     except Exception: pass # Ignore errors here, main goal failed anyway
                 return False
        else:
             # check_order_status already logged error (e.g., timeout or not found)
             logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Could not determine final status for order {order_id}. Assuming failure. Aborting SL placement.")
             # Attempt to cancel just in case it's stuck somehow
             try:
                  logger.info(f"Attempting cancellation of unknown status order {order_id}.")
                  fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol)
             except Exception: pass
             return False

        # Final check on filled quantity after status check
        if filled_qty < CONFIG.position_qty_epsilon:
             logger.error(Fore.RED + f"{trade_action} failed: Order {order_id} resulted in effectively zero filled quantity ({filled_qty}) after status check. No position opened.")
             return False

        # --- Place Initial Stop-Loss Order (Set on Position for Bybit V5) ---
        position_side = "long" if side == "buy" else "short"
        logger.trade(f"Setting initial SL for new {position_side.upper()} position...")

        # Use the *actual filled quantity* for the SL logic (Bybit sets SL on position, size isn't in the call)
        # Use the SL price calculated earlier, already formatted string
        sl_price_str_for_api = sl_price_formatted_str

        # Define parameters for setting the stop-loss on the position (Bybit V5 specific)
        # We use the `private_post_position_set_trading_stop` implicit method via CCXT
        set_sl_params = {
            'category': CONFIG.market_type, # Required
            'symbol': MARKET_INFO['id'], # Use exchange-specific market ID
            'stopLoss': sl_price_str_for_api, # Trigger price for the stop loss
            'slTriggerBy': CONFIG.sl_trigger_by, # e.g., 'LastPrice', 'MarkPrice'
            'tpslMode': 'Full', # Apply SL to the entire position ('Partial' also possible)
            # 'positionIdx': 0 # For Bybit unified: 0 for one-way mode (default)
            # Note: We don't need quantity here as it applies to the existing position matching symbol/category/side.
        }
        logger.trade(f"Setting Position SL: Trigger={sl_price_str_for_api}, TriggerBy={CONFIG.sl_trigger_by}")
        logger.debug(f"Set SL Params (for setTradingStop): {set_sl_params}")

        sl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods if available
            # Endpoint: POST /v5/position/set-trading-stop
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                sl_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_sl_params)
            else:
                # Fallback: Maybe try via modify_position? Less likely for SL only.
                # Or raise error if specific method missing.
                logger.error(Fore.RED + "Cannot set SL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                # Critical: Position is open without SL. Attempt emergency close.
                raise ccxt.NotSupported("SL setting method not available.")


            logger.debug(f"Set SL raw response: {sl_set_response}")

            # Handle potential failure from fetch_with_retries
            if sl_set_response is None:
                 raise ccxt.ExchangeError("Set SL request failed after retries (fetch_with_retries returned None). Position might be UNPROTECTED.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(sl_set_response.get('info'), dict) and sl_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Stop Loss successfully set directly on the {position_side.upper()} position (Trigger: {sl_price_str_for_api}).")
                # --- Update Global State ---
                # CRITICAL: Clear any previous tracker state for this side
                # Use a placeholder to indicate SL is active on the position
                sl_marker_id = f"POS_SL_{position_side.upper()}"
                order_tracker[position_side] = {"sl_id": sl_marker_id, "tsl_id": None}
                logger.info(f"Updated order tracker: {order_tracker}")

                # Use actual average fill price in notification
                entry_msg = (
                    f"ENTERED {side.upper()} {filled_qty} {symbol.split('/')[0]} @ {average_price:.4f}. "
                    f"Initial SL @ {sl_price_str_for_api}. TSL pending profit threshold."
                )
                logger.trade(Back.GREEN + Fore.BLACK + Style.BRIGHT + entry_msg)
                termux_notify("Trade Entry", f"{side.upper()} {symbol} @ {average_price:.4f}, SL: {sl_price_str_for_api}")
                return True # SUCCESS!

            else:
                 # Extract error message if possible
                 error_msg = "Unknown reason."
                 if isinstance(sl_set_response.get('info'), dict):
                      error_msg = sl_set_response['info'].get('retMsg', error_msg)
                      error_code = sl_set_response['info'].get('retCode')
                      error_msg += f" (Code: {error_code})"
                 raise ccxt.ExchangeError(f"Stop loss setting failed. Exchange message: {error_msg}")

        # --- Handle SL Setting Failures ---
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NotSupported) as e:
             # This is critical - position opened but SL setting failed. Emergency close needed.
             logger.critical(Fore.RED + Style.BRIGHT + f"CRITICAL: Failed to set stop-loss on position after entry: {e}. Position is UNPROTECTED.")
             logger.warning(Fore.YELLOW + "Attempting emergency market closure of unprotected position...")
             try:
                 emergency_close_side = "sell" if position_side == "long" else "buy"
                 # Format filled quantity precisely for closure order
                 close_qty_str = format_amount(symbol, filled_qty, ROUND_DOWN)
                 close_qty_decimal = Decimal(close_qty_str)

                 # Check against minimum quantity again before closing
                 min_qty_close = Decimal(str(MARKET_INFO['limits']['amount']['min']))
                 if close_qty_decimal < min_qty_close:
                      logger.critical(f"{Fore.RED}Emergency closure quantity {close_qty_decimal} is below minimum {min_qty_close}. MANUAL CLOSURE REQUIRED!")
                      termux_notify("EMERGENCY!", f"{symbol} POS UNPROTECTED & < MIN QTY! Close manually!")
                      return False # Indicate failure

                 # Place the emergency closure order
                 emergency_close_order = fetch_with_retries(
                     EXCHANGE.create_market_order,
                     symbol=symbol,
                     side=emergency_close_side,
                     amount=float(close_qty_decimal),
                     params={'reduceOnly': True} # Ensure it only closes
                 )

                 if emergency_close_order and (emergency_close_order.get('id') or emergency_close_order.get('info', {}).get('retCode') == 0):
                     close_id = emergency_close_order.get('id', 'N/A (retCode 0)')
                     logger.trade(Fore.GREEN + f"Emergency closure order placed successfully: ID {close_id}")
                     termux_notify("Closure Attempted", f"{symbol} emergency closure sent.")
                     # Reset tracker state as position *should* be closing (best effort)
                     order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                 else:
                      error_msg = emergency_close_order.get('info', {}).get('retMsg', 'Unknown error') if isinstance(emergency_close_order, dict) else str(emergency_close_order)
                      logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Order placement failed): {error_msg}. MANUAL INTERVENTION REQUIRED!")
                      termux_notify("EMERGENCY!", f"{symbol} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")

             except Exception as close_err:
                 logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Exception during closure): {close_err}. MANUAL INTERVENTION REQUIRED!", exc_info=True)
                 termux_notify("EMERGENCY!", f"{symbol} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                 # Do NOT reset tracker state here, as we don't know the position status

             return False # Signal overall failure of the entry attempt due to SL failure

        except Exception as e:
            logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error setting SL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position may be open without Stop Loss due to unexpected SL setting error. MANUAL INTERVENTION ADVISED.")
            # Consider emergency closure here too? Yes, safer.
            # (Code omitted for brevity, same as above emergency closure block)
            return False

    # --- Handle Initial Market Order Failures ---
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
        # Error placing the initial market order itself
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Exchange error placing market order: {e}")
        # Log specific exchange message if available
        if isinstance(getattr(e, 'args', None), tuple) and len(e.args) > 0 and isinstance(e.args[0], str):
             logger.error(f"Exchange message excerpt: {e.args[0][:500]}")
        return False
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during market order placement: {e}", exc_info=True)
        return False

def manage_trailing_stop(
    symbol: str,
    position_side: str, # 'long' or 'short'
    position_qty: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    atr: Decimal
) -> None:
    """Manages the activation and setting of a trailing stop loss on the position, using Decimal."""
    global order_tracker, EXCHANGE, MARKET_INFO

    logger.debug(f"Checking TSL status for {position_side.upper()} position...")

    if EXCHANGE is None or MARKET_INFO is None:
         logger.error("Exchange or Market Info not available, cannot manage TSL.")
         return

    # --- Initial Checks ---
    if position_qty < CONFIG.position_qty_epsilon or entry_price.is_nan() or entry_price <= Decimal("0"):
        # If position seems closed or invalid, ensure tracker is clear.
        if order_tracker[position_side]["sl_id"] or order_tracker[position_side]["tsl_id"]:
             logger.debug(f"Position {position_side} appears closed or invalid (Qty: {position_qty}, Entry: {entry_price}). Clearing stale order trackers.")
             order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return # No position to manage TSL for

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.warning(Fore.YELLOW + "Cannot evaluate TSL activation: Invalid ATR value.")
        return

    # --- Get Current Tracker State ---
    initial_sl_marker = order_tracker[position_side]["sl_id"] # Could be ID or placeholder "POS_SL_..."
    active_tsl_marker = order_tracker[position_side]["tsl_id"] # Could be ID or placeholder "POS_TSL_..."

    # If TSL is already active (has a marker), assume exchange handles it.
    if active_tsl_marker:
        log_msg = f"{position_side.upper()} TSL ({active_tsl_marker}) is already active. Exchange is managing the trail."
        logger.debug(log_msg)
        # Sanity check: Ensure initial SL marker is None if TSL is active
        if initial_sl_marker:
             logger.warning(f"Inconsistent state: TSL active ({active_tsl_marker}) but initial SL marker ({initial_sl_marker}) is also present. Clearing initial SL marker.")
             order_tracker[position_side]["sl_id"] = None
        return

    # If TSL is not active, check if we *should* activate it.
    # Requires an initial SL marker to be present (we replace SL with TSL).
    if not initial_sl_marker:
        # This can happen if the initial SL setting failed, or if state got corrupted.
        logger.warning(f"Cannot activate TSL for {position_side.upper()}: Initial SL protection marker is missing from tracker. Position might be unprotected.")
        # Consider adding logic here to try and set a regular SL if missing? Or just warn.
        return

    # --- Check TSL Activation Condition ---
    profit = Decimal("NaN")
    try:
        if position_side == "long":
            profit = current_price - entry_price
        else: # short
            profit = entry_price - current_price
    except TypeError: # Handle potential NaN in prices
        logger.warning("Cannot calculate profit for TSL check due to NaN price(s).")
        return

    # Activation threshold in price points
    activation_threshold_points = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(f"{position_side.upper()} Profit: {profit:.4f}, TSL Activation Threshold (Points): {activation_threshold_points:.4f} ({CONFIG.tsl_activation_atr_multiplier} * ATR)")

    # Activate TSL only if profit exceeds the threshold
    if profit > activation_threshold_points:
        logger.trade(Fore.GREEN + Style.BRIGHT + f"Profit threshold reached for {position_side.upper()} position (Profit {profit:.4f} > Threshold {activation_threshold_points:.4f}). Activating TSL.")

        # --- Set Trailing Stop Loss on Position ---
        # Bybit V5 sets TSL directly on the position using specific parameters.
        # We use the same `set_trading_stop` endpoint as the initial SL, but provide TSL params.

        # TSL distance as percentage string (e.g., "0.5" for 0.5%)
        # Ensure correct formatting for the API (string representation)
        trail_percent_str = str(CONFIG.trailing_stop_percent.quantize(Decimal("0.01")))

        # Bybit V5 Parameters for setting TSL on position:
        set_tsl_params = {
            'category': CONFIG.market_type, # Required
            'symbol': MARKET_INFO['id'], # Use exchange-specific market ID
            'trailingStop': trail_percent_str, # Trailing distance percentage (as string)
            'tpslMode': 'Full', # Apply to the whole position
            'slTriggerBy': CONFIG.tsl_trigger_by, # Trigger type for the trail
            # 'activePrice': format_price(symbol, current_price), # Optional: Price to activate the trail immediately. If omitted, Bybit activates when price moves favorably by trail %. Check docs.
            'stopLoss': '0', # IMPORTANT: Set stopLoss to '0' or empty string "" to REMOVE the fixed SL when activating TSL. Check Bybit docs for exact value. Let's try '0'.
            # 'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(f"Setting Position TSL: Trail={trail_percent_str}%, TriggerBy={CONFIG.tsl_trigger_by}, Removing Fixed SL")
        logger.debug(f"Set TSL Params (for setTradingStop): {set_tsl_params}")

        tsl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                tsl_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_tsl_params)
            else:
                logger.error(Fore.RED + "Cannot set TSL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                raise ccxt.NotSupported("TSL setting method not available.")

            logger.debug(f"Set TSL raw response: {tsl_set_response}")

            # Handle potential failure from fetch_with_retries
            if tsl_set_response is None:
                 raise ccxt.ExchangeError("Set TSL request failed after retries (fetch_with_retries returned None). Position might be UNPROTECTED.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(tsl_set_response.get('info'), dict) and tsl_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Trailing Stop Loss successfully activated for {position_side.upper()} position. Trail: {trail_percent_str}%")
                # --- Update Global State ---
                # Set TSL active marker and clear the initial SL marker
                tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                order_tracker[position_side]["tsl_id"] = tsl_marker_id
                order_tracker[position_side]["sl_id"] = None # Remove initial SL marker
                logger.info(f"Updated order tracker: {order_tracker}")
                termux_notify("TSL Activated", f"{position_side.upper()} {symbol} TSL active.")
                return # Success

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                if isinstance(tsl_set_response.get('info'), dict):
                     error_msg = tsl_set_response['info'].get('retMsg', error_msg)
                     error_code = tsl_set_response['info'].get('retCode')
                     error_msg += f" (Code: {error_code})"
                # Check if error was due to trying to remove non-existent SL (might be benign)
                # Example Bybit code: 110025 = SL/TP order not found or completed
                if isinstance(tsl_set_response.get('info'), dict) and tsl_set_response['info'].get('retCode') == 110025:
                     logger.warning(f"TSL activation may have succeeded, but received code 110025 (SL/TP not found/completed) when trying to clear fixed SL. Assuming TSL is active.")
                     # Proceed as if successful, update tracker
                     tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                     order_tracker[position_side]["tsl_id"] = tsl_marker_id
                     order_tracker[position_side]["sl_id"] = None
                     logger.info(f"Updated order tracker (assuming TSL active despite code 110025): {order_tracker}")
                     termux_notify("TSL Activated*", f"{position_side.upper()} {symbol} TSL active (check exchange).")
                     return # Treat as success for now
                else:
                    raise ccxt.ExchangeError(f"Failed to activate trailing stop loss. Exchange message: {error_msg}")

        # --- Handle TSL Setting Failures ---
        except (ccxt.ExchangeError, ccxt.InvalidOrder, ccxt.NotSupported) as e:
            # TSL setting failed. Initial SL marker *might* still be in the tracker.
            # Position might be protected by the initial SL, or might be unprotected if SL setting failed originally.
            logger.error(Fore.RED + Style.BRIGHT + f"Failed to activate TSL: {e}")
            logger.warning(Fore.YELLOW + "Position continues with initial SL (if set) or may be UNPROTECTED if initial SL failed.")
            # Do NOT clear the initial SL marker here. Do not set TSL marker.
            termux_notify("TSL Activation FAILED!", f"{symbol} TSL activation failed. Check logs/position.")
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error activating TSL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position continues with initial SL (if set) or may be UNPROTECTED. MANUAL INTERVENTION ADVISED.")
            termux_notify("TSL Activation FAILED!", f"{symbol} TSL activation failed (unexpected). Check logs/position.")

    else:
        # Profit threshold not met
        sl_status_log = f"({initial_sl_marker})" if initial_sl_marker else "(None!)"
        logger.debug(f"{position_side.upper()} profit ({profit:.4f}) has not crossed TSL activation threshold ({activation_threshold_points:.4f}). Keeping initial SL {sl_status_log}.")


def print_status_panel(
    cycle: int, timestamp: Optional[pd.Timestamp], price: Optional[Decimal], indicators: Optional[Dict[str, Decimal]],
    positions: Optional[Dict[str, Dict[str, Any]]], equity: Optional[Decimal], signals: Dict[str, bool],
    order_tracker_state: Dict[str, Dict[str, Optional[str]]] # Pass tracker state explicitly
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""

    header_color = Fore.MAGENTA + Style.BRIGHT
    section_color = Fore.CYAN
    value_color = Fore.WHITE
    reset_all = Style.RESET_ALL

    print(header_color + "\n" + "=" * 80)
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else f"{Fore.YELLOW}N/A"
    print(f" Cycle: {value_color}{cycle}{header_color} | Timestamp: {value_color}{ts_str}")
    equity_str = f"{equity:.4f} {MARKET_INFO.get('settle', 'Quote')}" if equity is not None and not equity.is_nan() else f"{Fore.YELLOW}N/A"
    print(f" Equity: {Fore.GREEN}{equity_str}" + reset_all)
    print(header_color + "-" * 80)

    # --- Market & Indicators ---
    price_str = f"{price:.4f}" if price is not None and not price.is_nan() else f"{Fore.YELLOW}N/A"
    atr = indicators.get('atr') if indicators else None
    atr_str = f"{atr:.6f}" if atr is not None and not atr.is_nan() else f"{Fore.YELLOW}N/A"
    trend_ema = indicators.get('trend_ema') if indicators else None
    trend_ema_str = f"{trend_ema:.4f}" if trend_ema is not None and not trend_ema.is_nan() else f"{Fore.YELLOW}N/A"

    price_color = Fore.WHITE
    trend_desc = f"{Fore.YELLOW}Trend N/A"
    if price is not None and not price.is_nan() and trend_ema is not None and not trend_ema.is_nan():
        if price > trend_ema: price_color = Fore.GREEN; trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema: price_color = Fore.RED; trend_desc = f"{price_color}(Below Trend)"
        else: price_color = Fore.YELLOW; trend_desc = f"{price_color}(At Trend)"

    stoch_k = indicators.get('stoch_k') if indicators else None
    stoch_d = indicators.get('stoch_d') if indicators else None
    stoch_k_str = f"{stoch_k:.2f}" if stoch_k is not None and not stoch_k.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_d_str = f"{stoch_d:.2f}" if stoch_d is not None and not stoch_d.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_color = Fore.YELLOW
    stoch_desc = f"{Fore.YELLOW}Stoch N/A"
    if stoch_k is not None and not stoch_k.is_nan():
         if stoch_k < Decimal(25): stoch_color = Fore.GREEN; stoch_desc = f"{stoch_color}Oversold (<25)"
         elif stoch_k > Decimal(75): stoch_color = Fore.RED; stoch_desc = f"{stoch_color}Overbought (>75)"
         else: stoch_color = Fore.YELLOW; stoch_desc = f"{stoch_color}Neutral (25-75)"

    fast_ema = indicators.get('fast_ema') if indicators else None
    slow_ema = indicators.get('slow_ema') if indicators else None
    fast_ema_str = f"{fast_ema:.4f}" if fast_ema is not None and not fast_ema.is_nan() else f"{Fore.YELLOW}N/A"
    slow_ema_str = f"{slow_ema:.4f}" if slow_ema is not None and not slow_ema.is_nan() else f"{Fore.YELLOW}N/A"
    ema_cross_color = Fore.WHITE
    ema_desc = f"{Fore.YELLOW}EMA N/A"
    if fast_ema is not None and not fast_ema.is_nan() and slow_ema is not None and not slow_ema.is_nan():
        if fast_ema > slow_ema: ema_cross_color = Fore.GREEN; ema_desc = f"{ema_cross_color}Bullish Cross"
        elif fast_ema < slow_ema: ema_cross_color = Fore.RED; ema_desc = f"{ema_cross_color}Bearish Cross"
        else: ema_cross_color = Fore.YELLOW; ema_desc = f"{Fore.YELLOW}Aligned"

    status_data = [
        [section_color + "Market", value_color + CONFIG.symbol, f"{price_color}{price_str}"],
        [section_color + f"Trend EMA ({CONFIG.trend_ema_period})", f"{value_color}{trend_ema_str}", trend_desc],
        [section_color + "ATR (10)", f"{value_color}{atr_str}", ""],
        [section_color + "EMA Fast/Slow (8/12)", f"{ema_cross_color}{fast_ema_str} / {slow_ema_str}", ema_desc],
        [section_color + "Stoch %K/%D (10,3,3)", f"{stoch_color}{stoch_k_str} / {stoch_d_str}", stoch_desc],
    ]
    print(tabulate(status_data, tablefmt="fancy_grid", colalign=("left", "left", "left")))
    # print(header_color + "-" * 80) # Separator removed, using table grid

    # --- Positions & Orders ---
    pos_avail = positions is not None
    long_pos = positions.get('long', {}) if pos_avail else {}
    short_pos = positions.get('short', {}) if pos_avail else {}

    # Safely get values, handling None or NaN
    long_qty = long_pos.get('qty', Decimal("0.0"))
    short_qty = short_pos.get('qty', Decimal("0.0"))
    long_entry = long_pos.get('entry_price', Decimal("NaN"))
    short_entry = short_pos.get('entry_price', Decimal("NaN"))
    long_pnl = long_pos.get('pnl', Decimal("NaN"))
    short_pnl = short_pos.get('pnl', Decimal("NaN"))
    long_liq = long_pos.get('liq_price', Decimal("NaN"))
    short_liq = short_pos.get('liq_price', Decimal("NaN"))

    # Use the passed tracker state snapshot
    long_sl_marker = order_tracker_state['long']['sl_id']
    long_tsl_marker = order_tracker_state['long']['tsl_id']
    short_sl_marker = order_tracker_state['short']['sl_id']
    short_tsl_marker = order_tracker_state['short']['tsl_id']

    # Determine SL/TSL status strings
    def get_stop_status(sl_marker, tsl_marker):
        if tsl_marker:
            if tsl_marker.startswith("POS_TSL_"): return f"{Fore.GREEN}TSL Active (Pos)"
            else: return f"{Fore.GREEN}TSL Active (ID: ...{tsl_marker[-6:]})" # Should not happen with V5 pos-based TSL
        elif sl_marker:
            if sl_marker.startswith("POS_SL_"): return f"{Fore.YELLOW}SL Active (Pos)"
            else: return f"{Fore.YELLOW}SL Active (ID: ...{sl_marker[-6:]})" # Should not happen with V5 pos-based SL
        else:
            # No marker found in tracker
            return f"{Fore.RED}{Style.BRIGHT}NONE (!)" # Highlight if no stop is tracked

    # Display stop status only if position exists
    long_stop_status = get_stop_status(long_sl_marker, long_tsl_marker) if long_qty >= CONFIG.position_qty_epsilon else f"{value_color}-"
    short_stop_status = get_stop_status(short_sl_marker, short_tsl_marker) if short_qty >= CONFIG.position_qty_epsilon else f"{value_color}-"

    # Format position details, handle potential None or NaN from failed fetch/parsing
    if not pos_avail:
        long_qty_str, short_qty_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_entry_str, short_entry_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_pnl_str, short_pnl_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_liq_str, short_liq_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
    else:
        # Format Decimals nicely, remove trailing zeros for quantity
        long_qty_str = f"{long_qty:.8f}".rstrip('0').rstrip('.') if long_qty != Decimal(0) else "0"
        short_qty_str = f"{short_qty:.8f}".rstrip('0').rstrip('.') if short_qty != Decimal(0) else "0"
        long_entry_str = f"{long_entry:.4f}" if not long_entry.is_nan() else "-"
        short_entry_str = f"{short_entry:.4f}" if not short_entry.is_nan() else "-"
        long_pnl_color = Fore.GREEN if not long_pnl.is_nan() and long_pnl >= 0 else Fore.RED
        short_pnl_color = Fore.GREEN if not short_pnl.is_nan() and short_pnl >= 0 else Fore.RED
        long_pnl_str = f"{long_pnl_color}{long_pnl:+.4f}{value_color}" if long_qty >= CONFIG.position_qty_epsilon and not long_pnl.is_nan() else "-"
        short_pnl_str = f"{short_pnl_color}{short_pnl:+.4f}{value_color}" if short_qty >= CONFIG.position_qty_epsilon and not short_pnl.is_nan() else "-"
        long_liq_str = f"{Fore.RED}{long_liq:.4f}{value_color}" if not long_liq.is_nan() and long_liq > 0 else "-"
        short_liq_str = f"{Fore.RED}{short_liq:.4f}{value_color}" if not short_liq.is_nan() and short_liq > 0 else "-"


    position_data = [
        [section_color + "Status", Fore.GREEN + "LONG", Fore.RED + "SHORT"],
        [section_color + "Quantity", f"{value_color}{long_qty_str}", f"{value_color}{short_qty_str}"],
        [section_color + "Entry Price", f"{value_color}{long_entry_str}", f"{value_color}{short_entry_str}"],
        [section_color + "Unrealized PnL", long_pnl_str, short_pnl_str],
        [section_color + "Liq. Price (Est.)", long_liq_str, short_liq_str],
        [section_color + "Active Stop", long_stop_status, short_stop_status],
    ]
    print(tabulate(position_data, headers="firstrow", tablefmt="fancy_grid", colalign=("left", "left", "left")))
    # print(header_color + "-" * 80) # Separator removed

    # --- Signals ---
    long_signal_color = Fore.GREEN + Style.BRIGHT if signals.get('long', False) else Fore.WHITE
    short_signal_color = Fore.RED + Style.BRIGHT if signals.get('short', False) else Fore.WHITE
    trend_status = f"(Trend Filter: {value_color}{'ON' if CONFIG.trade_only_with_trend else 'OFF'}{header_color})"
    print(f" Signals {trend_status}: Long [{long_signal_color}{str(signals.get('long', False)):<5}{header_color}] | Short [{short_signal_color}{str(signals.get('short', False)):<5}{header_color}]")
    print(header_color + "=" * 80 + reset_all)


def generate_signals(indicators: Optional[Dict[str, Decimal]], current_price: Optional[Decimal]) -> Dict[str, Union[bool, str]]:
    """Generates trading signals based on indicator conditions, using Decimal."""
    long_signal = False
    short_signal = False
    signal_reason = "No signal - Initial State"

    if not indicators:
        logger.warning("Cannot generate signals: indicators dictionary is missing.")
        return {"long": False, "short": False, "reason": "Indicators missing"}
    if current_price is None or current_price.is_nan() or current_price <= Decimal(0):
         logger.warning("Cannot generate signals: current price is missing or invalid.")
         return {"long": False, "short": False, "reason": "Invalid price"}

    try:
        # Use .get with default Decimal('NaN') to handle missing/failed indicators gracefully
        k = indicators.get('stoch_k', Decimal('NaN'))
        # d = indicators.get('stoch_d', Decimal('NaN')) # Available but not used in current logic
        fast_ema = indicators.get('fast_ema', Decimal('NaN'))
        slow_ema = indicators.get('slow_ema', Decimal('NaN'))
        trend_ema = indicators.get('trend_ema', Decimal('NaN'))
        # confirm_ema = indicators.get('confirm_ema', Decimal('NaN')) # Available if needed

        # Check if any required indicator is NaN
        required_indicators = {'stoch_k': k, 'fast_ema': fast_ema, 'slow_ema': slow_ema, 'trend_ema': trend_ema}
        nan_indicators = [name for name, val in required_indicators.items() if val.is_nan()]
        if nan_indicators:
             logger.warning(f"Cannot generate signals: Required indicator(s) are NaN: {', '.join(nan_indicators)}")
             return {"long": False, "short": False, "reason": f"NaN indicator(s): {', '.join(nan_indicators)}"}

        # Define conditions using Decimal comparisons for precision
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        price_above_trend = current_price > trend_ema
        price_below_trend = current_price < trend_ema

        # Stochastic K level conditions
        stoch_oversold = k < Decimal(25)
        stoch_overbought = k > Decimal(75)

        # --- Signal Logic ---
        # Combine conditions clearly
        long_entry_condition = ema_bullish_cross and stoch_oversold
        short_entry_condition = ema_bearish_cross and stoch_overbought

        # Apply trend filter if enabled
        if long_entry_condition:
            if CONFIG.trade_only_with_trend:
                if price_above_trend:
                    long_signal = True
                    signal_reason = "Long: Bullish EMA Cross & Stoch Oversold & Price Above Trend EMA"
                else:
                    signal_reason = "Long Blocked: Price Below Trend EMA (Trend Filter ON)"
            else: # Trend filter off
                long_signal = True
                signal_reason = "Long: Bullish EMA Cross & Stoch Oversold (Trend Filter OFF)"

        elif short_entry_condition:
             if CONFIG.trade_only_with_trend:
                 if price_below_trend:
                     short_signal = True
                     signal_reason = "Short: Bearish EMA Cross & Stoch Overbought & Price Below Trend EMA"
                 else:
                     signal_reason = "Short Blocked: Price Above Trend EMA (Trend Filter ON)"
             else: # Trend filter off
                 short_signal = True
                 signal_reason = "Short: Bearish EMA Cross & Stoch Overbought (Trend Filter OFF)"
        else:
             # Provide more context if no primary condition met
             reason_parts = []
             if not ema_bullish_cross and not ema_bearish_cross: reason_parts.append("No EMA cross")
             if ema_bullish_cross and not stoch_oversold: reason_parts.append("EMA Bull but Stoch not Oversold")
             if ema_bearish_cross and not stoch_overbought: reason_parts.append("EMA Bear but Stoch not Overbought")
             if not reason_parts: reason_parts.append("Conditions not met") # Default if none match
             signal_reason = f"No signal ({', '.join(reason_parts)})"


        # Log the outcome
        if long_signal or short_signal:
             logger.info(f"Signal Generated: {signal_reason}")
        else:
             # Log reason for no signal at debug level unless blocked by trend filter
             if "Blocked" in signal_reason:
                 logger.info(f"Signal Check: {signal_reason}")
             else:
                 logger.debug(f"Signal Check: {signal_reason}")

    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
        return {"long": False, "short": False, "reason": f"Exception: {e}"}

    return {"long": long_signal, "short": short_signal, "reason": signal_reason}


def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell with enhanced precision and logic."""
    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")
    start_time = time.time()
    cycle_success = True # Track if cycle completes without critical errors

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(Fore.RED + "Halting cycle: Market data fetch failed or returned empty.")
        cycle_success = False
        # No status panel if no data
        end_time = time.time()
        logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} Aborted (Duration: {end_time - start_time:.2f}s) ---")
        return # Skip cycle

    # 2. Get Current Price & Timestamp from Data
    current_price: Optional[Decimal] = None
    last_timestamp: Optional[pd.Timestamp] = None
    try:
        # Use close price of the last *completed* candle
        last_candle = df.iloc[-1]
        current_price_float = last_candle["close"]
        if pd.isna(current_price_float):
             raise ValueError("Latest close price is NaN")
        current_price = Decimal(str(current_price_float))
        last_timestamp = df.index[-1] # Already UTC from fetch_market_data
        logger.debug(f"Latest candle: Time={last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price:.4f}")

        # Check for stale data (compare last candle time to current time)
        now_utc = pd.Timestamp.utcnow() # UTC timestamp
        time_diff = now_utc - last_timestamp
        # Allow for interval duration + some buffer (e.g., 1.5 * interval + 60s)
        try:
             interval_seconds = EXCHANGE.parse_timeframe(CONFIG.interval)
             allowed_lag = pd.Timedelta(seconds=interval_seconds * 1.5 + 60)
             if time_diff > allowed_lag:
                  logger.warning(Fore.YELLOW + f"Market data may be stale. Last candle: {last_timestamp.strftime('%H:%M:%S')} ({time_diff} ago). Allowed lag: ~{allowed_lag}")
        except ValueError:
            logger.warning("Could not parse interval to check data staleness.")

    except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
        logger.error(Fore.RED + f"Halting cycle: Failed to get/process current price/timestamp from DataFrame: {e}", exc_info=True)
        cycle_success = False
        # No status panel if price invalid
        end_time = time.time()
        logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} Aborted (Duration: {end_time - start_time:.2f}s) ---")
        return # Skip cycle

    # 3. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(Fore.RED + "Halting cycle: Indicator calculation failed.")
        cycle_success = False
        # Fall through to display panel with available data if possible

    current_atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')

    # 4. Get Current State (Balance & Positions as Decimals)
    # Fetch balance first
    free_balance, current_equity = get_balance(MARKET_INFO.get('settle', 'USDT'))
    if current_equity is None or current_equity.is_nan():
        logger.error(Fore.RED + "Halting cycle: Failed to fetch valid current balance/equity.")
        # Don't proceed without knowing equity for risk calculation
        cycle_success = False
        # Fall through to display panel

    # Fetch positions (crucial state)
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Halting cycle: Failed to fetch current positions.")
        cycle_success = False
        # Fall through to display panel

    # --- Capture State Snapshot for Status Panel & Logic ---
    # Do this *before* potentially modifying state (like TSL management or entry)
    order_tracker_snapshot = {
        "long": order_tracker["long"].copy(),
        "short": order_tracker["short"].copy()
    }
    # Use the fetched positions directly as the snapshot
    positions_snapshot = positions if positions is not None else {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "pnl": Decimal("NaN"), "liq_price": Decimal("NaN")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "pnl": Decimal("NaN"), "liq_price": Decimal("NaN")}
    }

    # --- Logic continues only if critical data is available ---
    if cycle_success and positions is not None and current_equity is not None and not current_equity.is_nan():
        # 5. Manage Trailing Stops (pass Decimals)
        # Use the position data snapshot for consistency within the cycle
        active_long_pos = positions_snapshot.get('long', {})
        active_short_pos = positions_snapshot.get('short', {})
        active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
        active_short_qty = active_short_pos.get('qty', Decimal('0.0'))

        if active_long_qty >= CONFIG.position_qty_epsilon:
            logger.debug("Managing TSL for existing LONG position...")
            manage_trailing_stop(CONFIG.symbol, "long", active_long_qty, active_long_pos.get('entry_price', Decimal('NaN')), current_price, current_atr)
        elif active_short_qty >= CONFIG.position_qty_epsilon:
            logger.debug("Managing TSL for existing SHORT position...")
            manage_trailing_stop(CONFIG.symbol, "short", active_short_qty, active_short_pos.get('entry_price', Decimal('NaN')), current_price, current_atr)
        else:
            # If flat, ensure trackers are clear (belt-and-suspenders check)
            if order_tracker["long"]["sl_id"] or order_tracker["long"]["tsl_id"] or \
               order_tracker["short"]["sl_id"] or order_tracker["short"]["tsl_id"]:
                logger.info("Position is flat, ensuring order trackers are cleared.")
                order_tracker["long"] = {"sl_id": None, "tsl_id": None}
                order_tracker["short"] = {"sl_id": None, "tsl_id": None}
                # Update the snapshot to reflect the clearing for the panel display
                order_tracker_snapshot["long"] = {"sl_id": None, "tsl_id": None}
                order_tracker_snapshot["short"] = {"sl_id": None, "tsl_id": None}

        # 6. Generate Trading Signals (pass Decimals)
        signals_data = generate_signals(indicators, current_price)
        signals = {"long": signals_data["long"], "short": signals_data["short"]} # Extract bools

        # 7. Execute Trades based on Signals
        # Re-fetch positions *after* TSL management and *before* entry decision,
        # in case TSL hit and closed the position during management.
        # This prevents trying to enter a new trade if the position was just closed.
        logger.debug("Re-fetching positions after TSL management, before entry decision...")
        current_positions_after_tsl = get_current_position(CONFIG.symbol)
        if current_positions_after_tsl is None:
            logger.error(Fore.RED + "Halting cycle action: Failed to re-fetch positions after TSL management.")
            cycle_success = False # Mark cycle as failed for logging, but still show panel
        else:
            long_qty_after_tsl = current_positions_after_tsl.get('long', {}).get('qty', Decimal('0.0'))
            short_qty_after_tsl = current_positions_after_tsl.get('short', {}).get('qty', Decimal('0.0'))
            is_flat_after_tsl = long_qty_after_tsl < CONFIG.position_qty_epsilon and short_qty_after_tsl < CONFIG.position_qty_epsilon
            logger.debug(f"Position Status After TSL Check: Flat = {is_flat_after_tsl} (Long Qty: {long_qty_after_tsl}, Short Qty: {short_qty_after_tsl})")

            if is_flat_after_tsl:
                trade_attempted = False
                if signals.get("long"):
                    logger.info(Fore.GREEN + Style.BRIGHT + f"Long signal detected! {signals_data['reason']}. Attempting entry...")
                    trade_attempted = True
                    if place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr):
                         logger.info(f"Long entry successful for cycle {cycle_count}.")
                         # Note: SL/TSL state updated within place_risked_market_order
                    else:
                         logger.error(f"Long entry attempt failed for cycle {cycle_count}.")
                         # Optional: Implement cooldown logic here if needed

                elif signals.get("short"):
                    logger.info(Fore.RED + Style.BRIGHT + f"Short signal detected! {signals_data['reason']}. Attempting entry.")
                    trade_attempted = True
                    if place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr):
                         logger.info(f"Short entry successful for cycle {cycle_count}.")
                         # Note: SL/TSL state updated within place_risked_market_order
                    else:
                         logger.error(f"Short entry attempt failed for cycle {cycle_count}.")
                         # Optional: Implement cooldown logic here if needed

                # If a trade was attempted, main loop sleep handles the pause.

            elif not is_flat_after_tsl:
                 pos_side = "LONG" if long_qty_after_tsl >= CONFIG.position_qty_epsilon else "SHORT"
                 logger.info(f"Position ({pos_side}) already open, skipping new entry signals.")
                 # Future: Add exit logic based on counter-signals or other conditions if desired.
                 # Example: if pos_side == "LONG" and signals.get("short"): close_position("long")
                 # Example: if pos_side == "SHORT" and signals.get("long"): close_position("short")
    else:
        # Cycle failed earlier, skip trade logic
        logger.warning("Skipping trade logic due to earlier critical data fetch failure.")
        signals = {"long": False, "short": False} # Ensure signals are false for panel


    # 8. Display Status Panel (Always display if possible)
    # Use the state captured *before* TSL management and potential trade execution for consistency
    # unless the cycle failed very early.
    print_status_panel(
        cycle_count, last_timestamp, current_price, indicators,
        positions_snapshot, current_equity, signals, order_tracker_snapshot # Use the snapshots
    )

    end_time = time.time()
    status_log = "Complete" if cycle_success else "FAILED (Check Logs)"
    logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} {status_log} (Duration: {end_time - start_time:.2f}s) ---")


def graceful_shutdown() -> None:
    """Dispels active orders and closes open positions gracefully with precision."""
    logger.warning(Fore.YELLOW + Style.BRIGHT + "\nInitiating Graceful Shutdown Sequence...")
    termux_notify("Shutdown", f"Closing orders/positions for {CONFIG.symbol}.")

    global EXCHANGE, MARKET_INFO
    if EXCHANGE is None or MARKET_INFO is None:
        logger.error(Fore.RED + "Exchange object or Market Info not available. Cannot perform clean shutdown.")
        return

    symbol = CONFIG.symbol
    market_id = MARKET_INFO.get('id') # Exchange specific ID

    # 1. Cancel All Open Orders for the Symbol
    # This includes stop loss / take profit orders if they are separate entities
    # (Not typical for Bybit V5 position-based SL/TP, but good practice)
    try:
        logger.info(Fore.CYAN + f"Dispelling all cancellable open orders for {symbol}...")
        # fetch_with_retries handles category param
        # Fetch open orders first to log IDs (best effort)
        open_orders_list = []
        try:
            open_orders_list = fetch_with_retries(EXCHANGE.fetch_open_orders, symbol)
            if open_orders_list:
                 order_ids = [o.get('id', 'N/A') for o in open_orders_list]
                 logger.info(f"Found {len(open_orders_list)} open orders to attempt cancellation: {', '.join(order_ids)}")
            else:
                 logger.info("No cancellable open orders found via fetch_open_orders.")
        except Exception as fetch_err:
             logger.warning(Fore.YELLOW + f"Could not fetch open orders before cancelling: {fetch_err}. Proceeding with cancel all.")

        # Send cancel_all command
        if open_orders_list: # Only cancel if orders were found
            try:
                # Use cancel_all_orders for efficiency if supported and reliable
                # Note: cancel_all_orders might not exist or work reliably for all exchanges/params
                # Fallback: loop through fetched open orders and cancel individually
                if EXCHANGE.has.get('cancelAllOrders'):
                    logger.info("Using cancel_all_orders...")
                    response = fetch_with_retries(EXCHANGE.cancel_all_orders, symbol)
                    logger.info(f"Cancel all orders command sent. Response snippet: {str(response)[:200]}")
                    # Check response for success indicators if possible (Bybit V5 often returns list or success code)
                    if isinstance(response, dict) and response.get('info', {}).get('retCode') == 0:
                        logger.info(Fore.GREEN + "Cancel command successful (retCode 0).")
                    elif isinstance(response, list): # Sometimes returns list of cancelled orders
                        logger.info(Fore.GREEN + "Cancel command likely successful (returned list).")
                    else:
                        logger.warning(Fore.YELLOW + "Cancel all orders command sent, success confirmation unclear.")
                else:
                    logger.info("cancel_all_orders not available/reliable, cancelling individually...")
                    cancelled_count = 0
                    for order in open_orders_list:
                         try:
                              order_id = order['id']
                              logger.debug(f"Cancelling order {order_id}...")
                              fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol)
                              logger.info(f"Cancel request sent for order {order_id}.")
                              cancelled_count += 1
                              time.sleep(0.2) # Small delay between cancels
                         except ccxt.OrderNotFound:
                              logger.warning(f"Order {order_id} already gone when cancelling.")
                         except Exception as ind_cancel_err:
                              logger.error(f"Failed to cancel order {order_id}: {ind_cancel_err}")
                    logger.info(f"Attempted to cancel {cancelled_count}/{len(open_orders_list)} orders individually.")

            except Exception as cancel_err:
                 logger.error(Fore.RED + f"Error sending cancel command(s): {cancel_err}. MANUAL CHECK REQUIRED.")
        else:
             logger.info("Skipping cancellation as no open orders were found.")

        # Clear local tracker regardless, as intent is to have no active tracked orders
        logger.info("Clearing local order tracker state.")
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error during order cancellation phase: {e}. MANUAL CHECK REQUIRED on exchange.", exc_info=True)

    # Add a small delay after cancelling orders before checking/closing positions
    logger.info("Waiting briefly after order cancellation...")
    time.sleep(max(CONFIG.order_check_delay_seconds, 2)) # Wait at least 2 seconds

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function with retries
        positions = get_current_position(symbol)

        closed_count = 0
        if positions:
            try:
                 min_qty_dec = Decimal(str(MARKET_INFO['limits']['amount']['min']))
            except (KeyError, InvalidOperation, TypeError):
                 logger.warning("Could not determine minimum order quantity for closure validation.")
                 min_qty_dec = Decimal("0") # Assume zero if unavailable

            for side, pos_data in positions.items():
                 qty = pos_data.get('qty', Decimal("0.0"))
                 entry_price = pos_data.get('entry_price', Decimal("NaN"))

                 # Check if quantity is significant using epsilon
                 if qty.copy_abs() >= CONFIG.position_qty_epsilon:
                     close_side = "sell" if side == "long" else "buy"
                     logger.warning(Fore.YELLOW + f"Closing {side} position (Qty: {qty}, Entry: {entry_price:.4f if not entry_price.is_nan() else 'N/A'}) with market order...")
                     try:
                         # Format quantity precisely for closure order (use absolute value and round down)
                         close_qty_str = format_amount(symbol, qty.copy_abs(), ROUND_DOWN)
                         close_qty_decimal = Decimal(close_qty_str)

                         # Validate against minimum quantity before attempting closure
                         if close_qty_decimal < min_qty_dec:
                              logger.critical(f"{Fore.RED}Closure quantity {close_qty_decimal} for {side} position is below exchange minimum {min_qty_dec}. MANUAL CLOSURE REQUIRED!")
                              termux_notify("EMERGENCY!", f"{symbol} {side} POS < MIN QTY! Close manually!")
                              continue # Skip trying to close this position

                         # Place the closure market order
                         close_params = {'reduceOnly': True} # Crucial: Only close, don't open new position
                         # fetch_with_retries handles category param
                         close_order = fetch_with_retries(
                             EXCHANGE.create_market_order,
                             symbol=symbol,
                             side=close_side,
                             amount=float(close_qty_decimal), # CCXT needs float
                             params=close_params
                         )

                         # Check response for success
                         if close_order and (close_order.get('id') or close_order.get('info', {}).get('retCode') == 0):
                            close_id = close_order.get('id', 'N/A (retCode 0)')
                            logger.trade(Fore.GREEN + f"Position closure order placed successfully: ID {close_id}")
                            closed_count += 1
                            # Wait briefly to allow fill confirmation before checking next position (if any)
                            time.sleep(max(CONFIG.order_check_delay_seconds, 2))
                            # Optional: Verify closure order status? Might slow shutdown.
                            # closure_status = check_order_status(close_order['id'], symbol) if close_order.get('id') else None
                            # logger.info(f"Closure order {close_id} final status: {closure_status.get('status') if closure_status else 'Unknown/NotChecked'}")
                         else:
                            # Log critical error if closure order placement fails
                            error_msg = close_order.get('info', {}).get('retMsg', 'No ID and no success code.') if isinstance(close_order, dict) else str(close_order)
                            logger.critical(Fore.RED + Style.BRIGHT + f"FAILED TO PLACE closure order for {side} position ({qty}). Response: {error_msg}. MANUAL INTERVENTION REQUIRED!")
                            termux_notify("EMERGENCY!", f"{symbol} {side} POS CLOSURE FAILED! Manual action!")

                     except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                         logger.critical(Fore.RED + Style.BRIGHT + f"FAILED TO CLOSE {side} position ({qty}): {e}. MANUAL INTERVENTION REQUIRED!")
                         termux_notify("EMERGENCY!", f"{symbol} {side} POS CLOSURE FAILED! Manual action!")
                     except Exception as e:
                         logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error closing {side} position: {e}. MANUAL INTERVENTION REQUIRED!", exc_info=True)
                         termux_notify("EMERGENCY!", f"{symbol} {side} POS CLOSURE FAILED! Manual action!")
                 else:
                      logger.debug(f"No significant {side} position found (Qty: {qty}).")

            # Final summary message
            if closed_count > 0:
                 logger.info(Fore.GREEN + f"Successfully placed {closed_count} closure order(s).")
            elif any(p['qty'].copy_abs() >= CONFIG.position_qty_epsilon for p in positions.values()):
                 # This case means positions existed but closure attempts failed or quantities were too small
                 logger.warning(Fore.YELLOW + "Attempted shutdown but closure orders failed or were not possible for all open positions. MANUAL CHECK REQUIRED.")
            else:
                logger.info(Fore.GREEN + "No open positions found requiring closure.")

        elif positions is None:
             # Failure to fetch positions during shutdown is critical
             logger.critical(Fore.RED + Style.BRIGHT + "Could not fetch final positions during shutdown. MANUAL CHECK REQUIRED on exchange!")
             termux_notify("Shutdown Warning!", f"{symbol} Cannot confirm position status. Check exchange!")

    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Error during position closure phase: {e}. Manual check advised.", exc_info=True)

    logger.warning(Fore.YELLOW + Style.BRIGHT + "Graceful Shutdown Sequence Complete.")
    termux_notify("Shutdown Complete", f"{CONFIG.symbol} bot stopped.")


# --- Main Spell Invocation ---
if __name__ == "__main__":
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80)
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Termux Trading Spell Activated (v2.1.1 Precision/Robust) ***")
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL)

    logger.info(f"Initializing Pyrmethus v2.1.1...")
    logger.info(f"Log Level configured to: {log_level_str}")

    # Log key configuration parameters for verification
    logger.info(f"--- Trading Configuration ---")
    logger.info(f"Symbol: {CONFIG.symbol} ({CONFIG.market_type.capitalize()})")
    logger.info(f"Timeframe: {CONFIG.interval}")
    logger.info(f"Risk per trade: {CONFIG.risk_percentage * 100:.2f}%")
    logger.info(f"SL Multiplier: {CONFIG.sl_atr_multiplier} * ATR")
    logger.info(f"TSL Activation: {CONFIG.tsl_activation_atr_multiplier} * ATR Profit")
    logger.info(f"TSL Trail Percent: {CONFIG.trailing_stop_percent}%")
    logger.info(f"Trigger Prices: SL={CONFIG.sl_trigger_by}, TSL={CONFIG.tsl_trigger_by}")
    logger.info(f"Trend Filter EMA({CONFIG.trend_ema_period}): {CONFIG.trade_only_with_trend}")
    logger.info(f"Position Quantity Epsilon: {CONFIG.position_qty_epsilon:.1E}") # Scientific notation
    logger.info(f"Loop Interval: {CONFIG.loop_sleep_seconds}s")
    logger.info(f"OHLCV Limit: {CONFIG.ohlcv_limit}")
    logger.info(f"Fetch Retries: {CONFIG.max_fetch_retries}")
    logger.info(f"-----------------------------")


    # Final check if exchange connection and market info loading succeeded
    if MARKET_INFO and EXCHANGE:
         termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2.1.1)")
         logger.info(Fore.GREEN + Style.BRIGHT + f"Initialization complete. Awaiting market whispers...")
         print(Fore.MAGENTA + "=" * 80 + Style.RESET_ALL) # Separator before first cycle log
    else:
         # Error should have been logged during init, exit was likely called, but double-check.
         logger.critical(Fore.RED + Style.BRIGHT + "Exchange or Market info failed to load during initialization. Cannot start trading loop.")
         sys.exit(1)

    cycle = 0
    try:
        while True:
            cycle += 1
            trading_spell_cycle(cycle)
            logger.info(Fore.BLUE + f"Cycle {cycle} finished. Resting for {CONFIG.loop_sleep_seconds} seconds...")
            time.sleep(CONFIG.loop_sleep_seconds)

    except KeyboardInterrupt:
        logger.warning(Fore.YELLOW + "\nCtrl+C detected! Initiating graceful shutdown...")
        graceful_shutdown()
    except Exception as e:
        # Catch unexpected errors in the main loop
        logger.critical(Fore.RED + Style.BRIGHT + f"\nFATAL RUNTIME ERROR in Main Loop (Cycle {cycle}): {e}", exc_info=True)
        termux_notify("Bot CRASHED!", f"{CONFIG.symbol} FATAL ERROR! Check logs!")
        logger.warning(Fore.YELLOW + "Attempting graceful shutdown after crash...")
        try:
            graceful_shutdown() # Attempt cleanup even on unexpected crash
        except Exception as shutdown_err:
            logger.error(f"Error during crash shutdown: {shutdown_err}", exc_info=True)
        sys.exit(1) # Exit with error code
    finally:
        # Ensure logs are flushed before exit, regardless of how loop ended
        logger.info("Flushing logs...")
        logging.shutdown()
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80)
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Trading Spell Deactivated ***")
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL)

```
