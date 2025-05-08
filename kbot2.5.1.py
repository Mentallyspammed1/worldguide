from colorama import init, Fore, Style

# Initialize Colorama for terminal color magic
init()
print(Fore.CYAN + Style.BRIGHT + "# Summoning the PyrmScalp Banner..." + Style.RESET_ALL)

print(
    Fore.RED
    + Style.BRIGHT
    + r"""
#  ██████╗ ██╗   ██╗██████╗ ███╗   ███╗███████╗ ██████╗ █████╗ ██╗     ██████╗
#  ██╔══██╗╚██╗ ██╔╝██╔══██╗████╗ ████║██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗
#  ██████╔╝ ╚████╔╝ ██████╔╝██╔████╔██║███████╗██║     ███████║██║     ██████╔╝
#  ██╔═══╝   ╚██╔╝  ██╔══██╗██║╚██╔╝██║╚════██║██║     ██╔══██║██║     ██╔═══╝
#  ██║        ██║   ██║  ██║██║ ╚═╝ ██║███████║╚██████╗██║  ██║███████╗██║
#  ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝
"""
    + Style.RESET_ALL
)

print(Fore.GREEN + "# PyrmScalp Banner Manifested!" + Style.RESET_ALL)
# Pyrmethus - Termux Trading Spell (v2.1.3 - Optimized Signals & Precision)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved robustness.

import os
import time
import logging
import sys
import subprocess  # For termux-toast security
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, ROUND_DOWN, InvalidOperation, DivisionByZero
import copy  # For deepcopy of tracker state

# Attempt to import necessary enchantments
try:
    import ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np  # Used by pandas internally, good to list explicitly
    from tabulate import tabulate
    from colorama import init, Fore, Style, Back
    import requests  # Often needed by ccxt, good to suggest installation
    # Explicitly check for ta, although current code uses pandas/numpy for indicators
    # If you switch to the 'ta' library, uncomment and handle import error here.
    # import ta
except ImportError as e:
    # Provide specific guidance for Termux users
    init(autoreset=True)  # Initialize colorama for error messages
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell in your Termux terminal:")
    print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
    print(
        f"{Style.BRIGHT}pip install ccxt python-dotenv pandas numpy tabulate colorama requests{Style.RESET_ALL}"
    )  # Add ta if needed
    sys.exit(1)

# Weave the Colorama magic into the terminal
init(autoreset=True)

# Set Decimal precision (adjust if needed, higher precision means more memory/CPU)
# The default precision (usually 28) is often sufficient for most price/qty calcs.
# It might be necessary to increase if dealing with extremely small values or very high precision instruments.
# getcontext().prec = 50 # Example: Increase precision if needed

# --- Arcane Configuration ---
print(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.1.3...")

# Summon secrets from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe
# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5  # Between INFO and WARNING
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")


def trade_log(self, message, *args, **kws):
    """Custom logging method for trade-related events."""
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(TRADE_LEVEL_NUM, message, args, **kws)


# Add the custom method to the Logger class if it doesn't exist
if not hasattr(logging.Logger, "trade"):
    logging.Logger.trade = trade_log

# More detailed log format, includes module and line number for easier debugging
log_formatter = logging.Formatter(
    Fore.CYAN
    + "%(asctime)s "
    + Style.BRIGHT
    + "[%(levelname)-8s] "  # Padded levelname
    + Fore.WHITE
    + "(%(filename)s:%(lineno)d) "  # Added file/line info
    + Style.RESET_ALL
    + Fore.WHITE
    + "%(message)s"
)
logger = logging.getLogger(__name__)
# Set level via environment variable or default to INFO
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

# Explicitly use stdout to avoid potential issues in some environments
# Ensure handlers are not duplicated if script is reloaded or run multiple times in same process
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

# Prevent duplicate messages if the root logger is also configured (common issue)
logger.propagate = False


class TradingConfig:
    """Holds the sacred parameters of our spell, enhanced with precision awareness and validation."""

    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        # Default symbol format for Bybit V5 Unified is BASE/QUOTE:SETTLE, e.g., BTC/USDT:USDT
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW)
        self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW).lower()  # 'linear' or 'inverse'
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        # Risk as a percentage of total equity (e.g., 0.01 for 1%, 0.001 for 0.1%)
        self.risk_percentage = self._get_env(
            "RISK_PERCENTAGE",
            "0.01",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.00001"),
            max_val=Decimal("0.5"),
        )  # 0.001% to 50% risk
        self.sl_atr_multiplier = self._get_env(
            "SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1")
        )
        # TSL activation threshold in ATR units above entry price
        self.tsl_activation_atr_multiplier = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1")
        )
        # Bybit V5 TSL distance is a percentage (e.g., 0.5 for 0.5%). Ensure value is suitable.
        self.trailing_stop_percent = self._get_env(
            "TRAILING_STOP_PERCENT",
            "0.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.001"),
            max_val=Decimal("10.0"),
        )  # 0.001% to 10% trail
        # Trigger type for SL/TSL orders. Bybit V5 allows LastPrice, MarkPrice, IndexPrice.
        self.sl_trigger_by = self._get_env(
            "SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )
        self.tsl_trigger_by = self._get_env(
            "TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]
        )  # Usually same as SL

        # --- Optimized Indicator Periods (Read from .env with new recommended defaults) ---
        # Based on previous analysis for 3m FARTCOIN/USDT
        self.trend_ema_period = self._get_env(
            "TREND_EMA_PERIOD", "12", Fore.YELLOW, cast_type=int, min_val=5, max_val=500
        )  # Optimized default 12
        self.fast_ema_period = self._get_env(
            "FAST_EMA_PERIOD", "9", Fore.YELLOW, cast_type=int, min_val=1, max_val=200
        )  # Optimized default 9
        self.slow_ema_period = self._get_env(
            "SLOW_EMA_PERIOD", "21", Fore.YELLOW, cast_type=int, min_val=2, max_val=500
        )  # Optimized default 21
        # self.confirm_ema_period = self._get_env("CONFIRM_EMA_PERIOD", "5", Fore.YELLOW, cast_type=int, min_val=1, max_val=100) # Example: if you use a 3rd EMA
        self.stoch_period = self._get_env(
            "STOCH_PERIOD", "7", Fore.YELLOW, cast_type=int, min_val=1, max_val=100
        )  # Optimized default 7
        self.stoch_smooth_k = self._get_env(
            "STOCH_SMOOTH_K", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10
        )  # Optimized default 3
        self.stoch_smooth_d = self._get_env(
            "STOCH_SMOOTH_D", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10
        )  # Optimized default 3
        self.atr_period = self._get_env(
            "ATR_PERIOD", "5", Fore.YELLOW, cast_type=int, min_val=1, max_val=100
        )  # Optimized default 5

        # --- Signal Logic Thresholds (Configurable) ---
        self.stoch_oversold_threshold = self._get_env(
            "STOCH_OVERSOLD_THRESHOLD",
            "30",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("45"),
        )  # Optimized default 30
        self.stoch_overbought_threshold = self._get_env(
            "STOCH_OVERBOUGHT_THRESHOLD",
            "70",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("55"),
            max_val=Decimal("100"),
        )  # Optimized default 70
        # Loosened Trend Filter Threshold (price within X% of Trend EMA)
        self.trend_filter_buffer_percent = self._get_env(
            "TREND_FILTER_BUFFER_PERCENT",
            "0.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )  # Optimized default 0.5
        # ATR Filter Threshold (price move must be > X * ATR)
        self.atr_move_filter_multiplier = self._get_env(
            "ATR_MOVE_FILTER_MULTIPLIER",
            "0.5",
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0"),
            max_val=Decimal("5"),
        )  # Optimized default 0.5

        # Epsilon: Small value for comparing quantities, dynamically determined after market info is loaded.
        self.position_qty_epsilon = Decimal(
            "1E-12"
        )  # Default tiny Decimal, will be overridden based on market precision

        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = self._get_env(
            "OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000
        )  # Reasonable candle limits
        self.loop_sleep_seconds = self._get_env(
            "LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5
        )  # Minimum sleep time
        self.order_check_delay_seconds = self._get_env(
            "ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1
        )
        self.order_check_timeout_seconds = self._get_env(
            "ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5
        )
        self.max_fetch_retries = self._get_env(
            "MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10
        )
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)

        if not self.api_key or not self.api_secret:
            logger.critical(
                Fore.RED + Style.BRIGHT + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting."
            )
            sys.exit(1)

        # Validate market type
        if self.market_type not in ["linear", "inverse"]:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Invalid MARKET_TYPE '{self.market_type}'. Must be 'linear' or 'inverse'. Halting."
            )
            sys.exit(1)

        # Validate EMA periods relative to each other
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}FAST_EMA_PERIOD ({self.fast_ema_period}) must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}). Halting."
            )
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(
                f"{Fore.YELLOW}TREND_EMA_PERIOD ({self.trend_ema_period}) is not significantly longer than SLOW_EMA_PERIOD ({self.slow_ema_period}). Consider increasing TREND_EMA_PERIOD for a smoother trend filter."
            )

        # Validate Stochastic thresholds relative to each other
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold}) must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold}). Halting."
            )
            sys.exit(1)

        logger.debug("Configuration loaded successfully.")

    def _get_env(
        self,
        key: str,
        default: Any,
        color: str,
        cast_type: type = str,
        min_val: Optional[Union[int, Decimal]] = None,
        max_val: Optional[Union[int, Decimal]] = None,
        allowed_values: Optional[List[str]] = None,
    ) -> Any:
        """Gets value from environment, casts, validates, and logs."""
        value_str = os.getenv(key)
        # Mask secrets in logs
        log_value = "****" if "SECRET" in key or "KEY" in key else value_str

        if value_str is None or value_str.strip() == "":  # Treat empty string as not set
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
                value_str = str(default)

        # --- Casting ---
        casted_value = None
        try:
            if cast_type == bool:
                casted_value = value_str.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                casted_value = Decimal(value_str)
            elif cast_type == int:
                casted_value = int(value_str)
            elif cast_type == float:
                casted_value = float(value_str)  # Generally avoid float for critical values, but allow if needed
            else:  # Default is str
                casted_value = str(value_str)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Could not cast {key} ('{value_str}') to {cast_type.__name__}: {e}. Using default: {default}"
            )
            # Attempt to cast the default value itself
            try:
                if default is None:
                    return None
                # Recast default carefully
                if cast_type == bool:
                    return str(default).lower() in ["true", "1", "yes", "y", "on"]
                if cast_type == Decimal:
                    return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting."
                )
                sys.exit(1)

        # --- Validation ---
        if casted_value is None:  # Should not happen if casting succeeded or defaulted
            logger.critical(f"{Fore.RED + Style.BRIGHT}Failed to obtain a valid value for {key}. Halting.")
            sys.exit(1)

        # Allowed values check (for strings like trigger types)
        if allowed_values and casted_value not in allowed_values:
            logger.error(
                f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}. Using default: {default}"
            )
            # Return default after logging error
            return default  # Assume default is valid

        # Min/Max checks (for numeric types - Decimal, int, float)
        validation_failed = False
        try:
            if min_val is not None:
                # Ensure min_val is Decimal if casted_value is Decimal
                min_val_dec = Decimal(str(min_val)) if isinstance(casted_value, Decimal) else min_val
                if casted_value < min_val_dec:
                    logger.error(
                        f"{Fore.RED}{key} value {casted_value} is below minimum {min_val}. Using default: {default}"
                    )
                    validation_failed = True
            if max_val is not None:
                # Ensure max_val is Decimal if casted_value is Decimal
                max_val_dec = Decimal(str(max_val)) if isinstance(casted_value, Decimal) else max_val
                if casted_value > max_val_dec:
                    logger.error(
                        f"{Fore.RED}{key} value {casted_value} is above maximum {max_val}. Using default: {default}"
                    )
                    validation_failed = True
        except InvalidOperation as e:
            logger.error(
                f"{Fore.RED}Error during min/max validation for {key} with value {casted_value} and limits ({min_val}, {max_val}): {e}. Using default: {default}"
            )
            validation_failed = True

        if validation_failed:
            # Re-cast default to ensure correct type is returned
            try:
                if default is None:
                    return None
                if cast_type == bool:
                    return str(default).lower() in ["true", "1", "yes", "y", "on"]
                if cast_type == Decimal:
                    return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(
                    f"{Fore.RED + Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting."
                )
                sys.exit(1)

        return casted_value


CONFIG = TradingConfig()
MARKET_INFO: Optional[Dict] = None  # Global to store market details after connection
EXCHANGE: Optional[ccxt.Exchange] = None  # Global for the exchange instance

# --- Exchange Nexus Initialization ---
print(Fore.MAGENTA + Style.BRIGHT + "\nEstablishing Nexus with the Exchange v2.1.3...")
try:
    exchange_options = {
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True,  # CCXT built-in rate limiter
        "options": {
            "defaultType": "swap",  # More specific for futures/swaps than 'future'
            "defaultSubType": CONFIG.market_type,  # 'linear' or 'inverse'
            "adjustForTimeDifference": True,  # Auto-sync clock with server
            # Bybit V5 API often requires 'category' for unified endpoints
            "brokerId": "PyrmethusV213",  # Custom identifier for Bybit API tracking
            "v5": {"category": CONFIG.market_type},  # Explicitly set category for V5 requests
        },
    }
    # Log options excluding secrets for debugging
    log_options = exchange_options.copy()
    log_options["apiKey"] = "****"
    log_options["secret"] = "****"
    logger.debug(f"Initializing CCXT Bybit with options: {log_options}")

    EXCHANGE = ccxt.bybit(exchange_options)

    # Test connectivity and credentials (important!)
    logger.info("Verifying credentials and connection...")
    EXCHANGE.check_required_credentials()  # Checks if keys are present/formatted ok
    logger.info("Credentials format check passed.")
    # Fetch time to verify connectivity, API key validity, and clock sync
    server_time = EXCHANGE.fetch_time()
    local_time = EXCHANGE.milliseconds()
    time_diff = abs(server_time - local_time)
    logger.info(f"Exchange time synchronized: {EXCHANGE.iso8601(server_time)} (Difference: {time_diff} ms)")
    if time_diff > 5000:  # Warn if clock skew is significant (e.g., > 5 seconds)
        logger.warning(
            f"{Fore.YELLOW}Significant time difference ({time_diff} ms) between system and exchange. Check system clock synchronization."
        )

    # Load markets (force reload to ensure fresh data)
    logger.info("Loading market spirits (market data)...")
    EXCHANGE.load_markets(True)  # Force reload
    logger.info(
        Fore.GREEN
        + Style.BRIGHT
        + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets)."
    )

    # Verify symbol exists and get market details
    if CONFIG.symbol not in EXCHANGE.markets:
        logger.error(
            Fore.RED + Style.BRIGHT + f"Symbol {CONFIG.symbol} not found in Bybit {CONFIG.market_type} market spirits."
        )
        # Suggest available symbols more effectively
        available_symbols = []
        try:
            # Extract quote currency robustly (handles SYMBOL/QUOTE:SETTLE format)
            # For futures, settle currency is often the key identifier in market lists
            settle_currency_candidates = CONFIG.symbol.split(":")  # e.g., ['BTC/USDT', 'USDT']
            settle_currency = settle_currency_candidates[-1] if len(settle_currency_candidates) > 1 else None
            if settle_currency:
                logger.info(f"Searching for symbols settling in {settle_currency}...")
                for s, m in EXCHANGE.markets.items():
                    # Check if market matches the configured type (linear/inverse) and is active
                    is_correct_type = (CONFIG.market_type == "linear" and m.get("linear")) or (
                        CONFIG.market_type == "inverse" and m.get("inverse")
                    )
                    # Filter by settle currency and check if active
                    if m.get("active") and is_correct_type and m.get("settle") == settle_currency:
                        available_symbols.append(s)
            else:
                logger.warning(
                    f"Could not parse settle currency from SYMBOL '{CONFIG.symbol}'. Cannot filter suggestions."
                )
                # Fallback: List all active symbols of the correct type
                for s, m in EXCHANGE.markets.items():
                    is_correct_type = (CONFIG.market_type == "linear" and m.get("linear")) or (
                        CONFIG.market_type == "inverse" and m.get("inverse")
                    )
                    if m.get("active") and is_correct_type:
                        available_symbols.append(s)

        except IndexError:
            logger.error(f"Could not parse base/quote from SYMBOL '{CONFIG.symbol}'.")

        suggestion_limit = 30
        if available_symbols:
            suggestions = ", ".join(sorted(available_symbols)[:suggestion_limit])
            if len(available_symbols) > suggestion_limit:
                suggestions += "..."
            logger.info(Fore.CYAN + f"Available active {CONFIG.market_type} symbols (sample): " + suggestions)
        else:
            logger.info(Fore.CYAN + f"Could not find any active {CONFIG.market_type} symbols to suggest.")
        sys.exit(1)
    else:
        MARKET_INFO = EXCHANGE.market(CONFIG.symbol)
        logger.info(Fore.CYAN + f"Market spirit for {CONFIG.symbol} acknowledged (ID: {MARKET_INFO.get('id')}).")

        # --- Log key precision and limits using Decimal ---
        # Extract values safely, providing defaults or logging errors
        try:
            # precision['price'] might be a tick size (Decimal) or number of decimal places (int)
            price_precision_raw = MARKET_INFO["precision"]["price"]
            # precision['amount'] might be a step size (Decimal) or number of decimal places (int)
            amount_precision_raw = MARKET_INFO["precision"]["amount"]
            min_amount_raw = MARKET_INFO["limits"]["amount"]["min"]
            max_amount_raw = MARKET_INFO["limits"]["amount"].get("max")  # Max might be None
            contract_size_raw = MARKET_INFO.get("contractSize", "1")  # Default to '1' if not present
            min_cost_raw = MARKET_INFO["limits"].get("cost", {}).get("min")  # Min cost might not exist

            # Convert to Decimal for logging and potential use, handle None/N/A
            price_prec_str = str(price_precision_raw) if price_precision_raw is not None else "N/A"
            amount_prec_str = str(amount_precision_raw) if amount_precision_raw is not None else "N/A"
            min_amount_dec = Decimal(str(min_amount_raw)) if min_amount_raw is not None else Decimal("NaN")
            max_amount_dec = (
                Decimal(str(max_amount_raw)) if max_amount_raw is not None else Decimal("Infinity")
            )  # Use Infinity for no max
            contract_size_dec = Decimal(str(contract_size_raw)) if contract_size_raw is not None else Decimal("NaN")
            min_cost_dec = Decimal(str(min_cost_raw)) if min_cost_raw is not None else Decimal("NaN")

            logger.debug(
                f"Market Precision: Price Tick/Decimals={price_prec_str}, Amount Step/Decimals={amount_prec_str}"
            )
            logger.debug(
                f"Market Limits: Min Amount={min_amount_dec}, Max Amount={max_amount_dec}, Min Cost={min_cost_dec}"
            )
            logger.debug(f"Contract Size: {contract_size_dec}")

            # --- Dynamically set epsilon based on amount precision (step size) ---
            # CCXT often provides amount precision as the step size directly
            amount_step_size = MARKET_INFO["precision"].get("amount")
            if amount_step_size is not None:
                try:
                    # Use a very small fraction of the step size as epsilon
                    # e.g., step = 0.001, epsilon = 0.000000000001 (1e-12)
                    # Using 1E-12 directly is generally safe as most step sizes are >= 1e-8
                    # A dynamic approach could be min(Decimal('1E-12'), Decimal(str(amount_step_size)) * Decimal('1E-6'))
                    # For simplicity and general safety with typical crypto precisions, 1E-12 is usually sufficient
                    CONFIG.position_qty_epsilon = Decimal("1E-12")  # A very small, fixed epsilon
                    logger.info(f"Set position_qty_epsilon to a small fixed value: {CONFIG.position_qty_epsilon:.1E}")
                except (InvalidOperation, TypeError):
                    logger.warning(
                        f"Could not parse amount step size '{amount_step_size}'. Using default epsilon: {CONFIG.position_qty_epsilon:.1E}"
                    )
            else:
                logger.warning(
                    f"Market info does not provide amount step size ('precision.amount'). Using default epsilon: {CONFIG.position_qty_epsilon:.1E}"
                )

        except (KeyError, TypeError, InvalidOperation) as e:
            logger.critical(
                f"{Fore.RED + Style.BRIGHT}Failed to parse critical market info (precision/limits/size) from MARKET_INFO: {e}. Halting.",
                exc_info=True,
            )
            logger.debug(f"Problematic MARKET_INFO: {MARKET_INFO}")
            sys.exit(1)

except ccxt.AuthenticationError as e:
    logger.critical(
        Fore.RED + Style.BRIGHT + f"Authentication failed! Check API Key/Secret validity and permissions. Error: {e}"
    )
    sys.exit(1)
except ccxt.NetworkError as e:
    logger.critical(
        Fore.RED + Style.BRIGHT + f"Network error connecting to Bybit: {e}. Check internet connection and Bybit status."
    )
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
# Uses placeholders like "POS_SL_LONG", "POS_TSL_LONG" for Bybit V5 position-based stops.
order_tracker: Dict[str, Dict[str, Optional[str]]] = {
    "long": {"sl_id": None, "tsl_id": None},
    "short": {"sl_id": None, "tsl_id": None},
}


# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available) via termux-toast."""
    if not os.getenv("TERMUX_VERSION"):
        logger.debug("Not running in Termux environment. Skipping notification.")
        return

    try:
        # Check if command exists using which (more portable than 'command -v')
        check_cmd = subprocess.run(["which", "termux-toast"], capture_output=True, text=True, check=False)
        if check_cmd.returncode != 0:
            logger.debug("termux-toast command not found. Skipping notification.")
            return

        # Basic sanitization - focus on preventing shell interpretation issues
        # Replace potentially problematic characters with spaces or remove them
        safe_title = (
            title.replace('"', "'")
            .replace("`", "'")
            .replace("$", "")
            .replace("\\", "")
            .replace(";", "")
            .replace("&", "")
            .replace("|", "")
            .replace("(", "")
            .replace(")", "")
        )
        safe_content = (
            content.replace('"', "'")
            .replace("`", "'")
            .replace("$", "")
            .replace("\\", "")
            .replace(";", "")
            .replace("&", "")
            .replace("|", "")
            .replace("(", "")
            .replace(")", "")
        )

        # Limit length to avoid potential buffer issues or overly long toasts
        max_len = 200  # Increased length slightly
        full_message = f"{safe_title}: {safe_content}"[:max_len]

        # Use list format for subprocess.run for security
        # Example styling: gravity middle, black text on green background, short duration
        cmd_list = ["termux-toast", "-g", "middle", "-c", "black", "-b", "green", "-s", full_message]
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=5)  # Add timeout

        if result.returncode != 0:
            # Log stderr if available
            stderr_msg = result.stderr.strip()
            logger.warning(
                f"termux-toast command failed with code {result.returncode}" + (f": {stderr_msg}" if stderr_msg else "")
            )
        # No else needed, success is silent

    except FileNotFoundError:
        logger.debug("termux-toast command not found (FileNotFoundError). Skipping notification.")
    except subprocess.TimeoutExpired:
        logger.warning("termux-toast command timed out. Skipping notification.")
    except Exception as e:
        # Catch other potential exceptions during subprocess execution
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}", exc_info=True)


# --- Precision Casting Spells ---


def format_price(symbol: str, price: Union[Decimal, str, float, int]) -> str:
    """Formats price according to market precision rules using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format price.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            price_dec = Decimal(str(price))
            return str(price_dec.quantize(Decimal("1E-8")))  # Quantize to 8 decimal places
        except Exception:
            return str(price)  # Last resort

    try:
        # CCXT's price_to_precision handles rounding/truncation based on market rules (tick size).
        # Ensure input is float as expected by CCXT methods.
        price_float = float(price)
        return EXCHANGE.price_to_precision(symbol, price_float)
    except (AttributeError, KeyError, InvalidOperation) as e:
        logger.error(
            f"{Fore.RED}Market info for {symbol} missing precision data or invalid price format: {e}. Using fallback formatting."
        )
        try:
            price_dec = Decimal(str(price))
            return str(price_dec.quantize(Decimal("1E-8")))
        except Exception:
            return str(price)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}. Using fallback.")
        try:
            price_dec = Decimal(str(price))
            return str(price_dec.quantize(Decimal("1E-8")))
        except Exception:
            return str(price)


def format_amount(symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
    """Formats amount according to market precision rules (step size) using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format amount.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            amount_dec = Decimal(str(amount))
            # Use quantize for fallback if Decimal input
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            return str(amount)  # Last resort

    try:
        # CCXT's amount_to_precision handles step size and rounding.
        # Map Python Decimal rounding modes to CCXT rounding modes if needed.
        ccxt_rounding_mode = ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND  # Basic mapping
        # Ensure input is float as expected by CCXT methods.
        amount_float = float(amount)
        return EXCHANGE.amount_to_precision(symbol, amount_float, rounding_mode=ccxt_rounding_mode)
    except (AttributeError, KeyError, InvalidOperation) as e:
        logger.error(
            f"{Fore.RED}Market info for {symbol} missing precision data or invalid amount format: {e}. Using fallback formatting."
        )
        try:
            amount_dec = Decimal(str(amount))
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            return str(amount)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}. Using fallback.")
        try:
            amount_dec = Decimal(str(amount))
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            return str(amount)


# --- Core Spell Functions ---


def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    """Generic wrapper to fetch data with retries and exponential backoff."""
    global EXCHANGE
    if EXCHANGE is None:
        logger.critical("Exchange object is None, cannot fetch data.")
        return None  # Indicate critical failure

    last_exception = None
    # Add category param automatically for V5 if not already present in kwargs['params']
    if "params" not in kwargs:
        kwargs["params"] = {}
    if (
        "category" not in kwargs["params"]
        and hasattr(EXCHANGE, "options")
        and "v5" in EXCHANGE.options
        and "category" in EXCHANGE.options["v5"]
    ):
        kwargs["params"]["category"] = EXCHANGE.options["v5"]["category"]
        # logger.debug(f"Auto-added category '{kwargs['params']['category']}' to params for {fetch_function.__name__}")

    for attempt in range(CONFIG.max_fetch_retries + 1):  # +1 to allow logging final failure
        try:
            # Log the attempt number and function being called at DEBUG level
            # Be cautious not to log sensitive parameters like API keys if they were somehow passed directly
            log_kwargs = {
                k: ("****" if "secret" in str(k).lower() or "key" in str(k).lower() else v) for k, v in kwargs.items()
            }
            logger.debug(
                f"Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}: Calling {fetch_function.__name__} with args={args}, kwargs={log_kwargs}"
            )
            result = fetch_function(*args, **kwargs)
            return result  # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = 2**attempt  # Exponential backoff (1, 2, 4, 8...)
            logger.warning(
                Fore.YELLOW
                + f"{fetch_function.__name__}: Network issue (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}). Retrying in {wait_time}s... Error: {e}"
            )
            if attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            else:
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to network issues."
                )
                # Re-raise the specific network error on final failure? Or return None? Let's return None, caller decides.
                return None  # Indicate failure after retries
        except ccxt.ExchangeNotAvailable as e:
            last_exception = e
            logger.error(Fore.RED + f"{fetch_function.__name__}: Exchange not available: {e}. Stopping retries.")
            # This is usually a hard stop, no point retrying
            # Raise it immediately or return None? Returning None allows caller to decide if it's fatal.
            return None
        except ccxt.AuthenticationError as e:
            last_exception = e
            logger.critical(
                Fore.RED + Style.BRIGHT + f"{fetch_function.__name__}: Authentication error: {e}. Halting script."
            )
            # This is a critical error, the script cannot proceed. Re-raise or sys.exit.
            sys.exit(1)  # Exit immediately on auth failure
        except (
            ccxt.OrderNotFound,
            ccxt.InsufficientFunds,
            ccxt.InvalidOrder,
            ccxt.BadRequest,
            ccxt.PermissionDenied,
        ) as e:
            # These are typically non-retryable errors related to the request parameters or exchange state.
            last_exception = e
            error_type = type(e).__name__
            logger.error(
                Fore.RED
                + f"{fetch_function.__name__}: Non-retryable error ({error_type}): {e}. Stopping retries for this call."
            )
            # Re-raise these specific errors so the caller can handle them appropriately
            raise e
        except ccxt.ExchangeError as e:
            # Includes rate limit errors, potentially invalid requests etc.
            last_exception = e
            # Check for specific retryable Bybit error codes if needed (e.g., 10006=timeout, 10016=internal error)
            # Bybit V5 Rate limit codes: 10018 (IP), 10017 (Key), 10009 (Frequency)
            # Bybit V5 Invalid Parameter codes: 110001, 110003, 110004, 110005, 110006, 110007, 110008, 110011, 110012, 110013, 110014, 110015, 110016, 110017, 110018, 110019, 110020, 110021, 110022, 110023, 110024, 110025, 110026, 110027, 110028, 110029, 110030, 110031, 110032, 110033, 110034, 110035, 110036, 110037, 110038, 110039, 110040, 110041, 110042, 110043, 110044, 110045, 110046, 110047, 110048, 110049, 110050, 110051, 110052, 110053, 110054, 110055, 110056, 110057, 110058, 110059, 110060, 110061, 110062, 110063, 110064, 110065, 110066, 110067, 110068, 110069, 110070, 110071, 110072, 110073, 110074, 110075, 110076, 110077, 110078, 110079, 110080, 110081, 110082, 110083, 110084, 110085, 110086, 110087, 110088, 110089, 110090, 110091, 110092, 110093, 110094, 110095, 110096, 110097, 110098, 110099, 110100
            # General internal errors: 10006, 10016, 10020, 10030
            # Other: 30034 (Position status not normal)
            error_code = getattr(e, "code", None)  # CCXT might parse the code from info dict
            error_message = str(e)
            should_retry = True
            wait_time = 2 * (attempt + 1)  # Default backoff

            # Check for common rate limit patterns / codes
            if "Rate limit exceeded" in error_message or error_code in [10017, 10018, 10009]:
                wait_time = 5 * (attempt + 1)  # Longer wait for rate limits
                logger.warning(
                    f"{Fore.YELLOW}{fetch_function.__name__}: Rate limit hit (Code: {error_code}). Retrying in {wait_time}s... Error: {e}"
                )
            # Check for specific non-retryable errors (e.g., invalid parameter codes)
            # These often start with 11xxxx for Bybit V5
            elif error_code is not None and 110000 <= error_code <= 110100:
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Non-retryable parameter/logic exchange error (Code: {error_code}): {e}. Stopping retries."
                )
                should_retry = False
            # Check for other specific non-retryable codes like insufficient funds, invalid state etc.
            elif error_code in [30034]:
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Non-retryable exchange state error (Code: {error_code}): {e}. Stopping retries."
                )
                should_retry = False
            else:
                # General exchange error, apply default backoff
                logger.warning(
                    f"{Fore.YELLOW}{fetch_function.__name__}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}, Code: {error_code}). Retrying in {wait_time}s... Error: {e}"
                )

            if should_retry and attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            elif should_retry:  # Final attempt failed
                logger.error(
                    Fore.RED
                    + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to exchange errors."
                )
                break  # Exit retry loop
            else:  # Non-retryable error encountered
                break  # Exit retry loop

        except Exception as e:
            # Catch-all for unexpected errors
            last_exception = e
            logger.error(Fore.RED + f"{fetch_function.__name__}: Unexpected shadow encountered: {e}", exc_info=True)
            break  # Stop on unexpected errors

    # If loop finished without returning, it means all retries failed or a break occurred
    # Re-raise the last specific non-retryable exception if it wasn't already (e.g. OrderNotFound, InsufficientFunds)
    if isinstance(
        last_exception,
        (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied),
    ):
        raise last_exception  # Propagate specific non-retryable errors

    # For other failures (Network, ExchangeNotAvailable, general ExchangeError after retries), return None
    if last_exception:
        logger.error(
            f"{fetch_function.__name__} ultimately failed after {CONFIG.max_fetch_retries + 1} attempts or encountered a non-retryable error type not explicitly re-raised."
        )
        return None  # Indicate failure

    # Should not reach here if successful, but defensive return None
    return None


def fetch_market_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data using the retry wrapper and perform validation."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})...")

    if EXCHANGE is None or not hasattr(EXCHANGE, "fetch_ohlcv"):
        logger.error(Fore.RED + "Exchange object not properly initialized or missing fetch_ohlcv.")
        return None

    # Ensure limit is positive (already validated in config, but double check)
    if limit <= 0:
        logger.error(f"Invalid OHLCV limit requested: {limit}. Using default 100.")
        limit = 100

    ohlcv_data = None
    try:
        # fetch_with_retries handles category param automatically
        ohlcv_data = fetch_with_retries(EXCHANGE.fetch_ohlcv, symbol, timeframe, limit=limit)
    except Exception as e:
        # fetch_with_retries should handle most errors, but catch any unexpected ones here
        logger.error(
            Fore.RED + f"Unhandled exception during fetch_ohlcv call via fetch_with_retries: {e}", exc_info=True
        )
        return None

    if ohlcv_data is None:
        # fetch_with_retries already logged the failure reason
        logger.error(Fore.RED + f"Failed to fetch OHLCV data for {symbol}.")
        return None
    if not isinstance(ohlcv_data, list) or not ohlcv_data:
        logger.error(
            Fore.RED
            + f"Received empty or invalid OHLCV data type: {type(ohlcv_data)}. Content: {str(ohlcv_data)[:100]}"
        )
        return None

    try:
        # Use Decimal for numeric columns directly during DataFrame creation where possible
        # However, pandas expects floats for most calculations, so convert back later
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp immediately to UTC datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)  # Drop rows where timestamp conversion failed

        # Convert numeric columns to float first for pandas/numpy compatibility
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaNs in critical price columns *after* conversion
        initial_len = len(df)
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
            logger.warning(
                Fore.YELLOW
                + f"Duplicate timestamps found in OHLCV data ({len(duplicates)} unique duplicates). Keeping last entry for each."
            )
            df = df[~df.index.duplicated(keep="last")]

        # Check time difference between last two candles vs expected interval
        if len(df) > 1:
            time_diff = df.index[-1] - df.index[-2]
            try:
                # Use pandas to parse timeframe string robustly
                expected_interval_td = pd.Timedelta(EXCHANGE.parse_timeframe(timeframe), unit="s")
                # Allow some tolerance (e.g., 20% of interval) for minor timing differences/API lag
                tolerance = expected_interval_td * 0.2
                if abs(time_diff.total_seconds()) > expected_interval_td.total_seconds() + tolerance.total_seconds():
                    logger.warning(
                        f"Unexpected large time gap between last two candles: {time_diff} (expected ~{expected_interval_td})"
                    )
            except ValueError:
                logger.warning(
                    f"Could not parse timeframe '{timeframe}' to calculate expected interval for time gap check."
                )
            except Exception as time_check_e:
                logger.warning(f"Error during time difference check: {time_check_e}")

        logger.info(
            Fore.GREEN
            + f"Market whispers received ({len(df)} candles). Latest: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        return df
    except Exception as e:
        logger.error(Fore.RED + f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
        return None


def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Decimal]]:
    """Calculate technical indicators using CONFIG periods, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    if df is None or df.empty:
        logger.error(Fore.RED + "Cannot calculate indicators on missing or empty DataFrame.")
        return None
    try:
        # Ensure data is float for TA-Lib / Pandas calculations, convert to Decimal at the end
        # Defensive: Make a copy to avoid modifying the original DataFrame if it's used elsewhere
        df_calc = df.copy()
        close = df_calc["close"].astype(float)
        high = df_calc["high"].astype(float)
        low = df_calc["low"].astype(float)

        # --- Check Data Length Requirements ---
        # Ensure enough data for EWMA initial states to stabilize somewhat
        # EWMA needs `span` number of points for the first value *if* adjust=True (default).
        # With adjust=False, the first value is just the first data point, and subsequent values are direct EMA.
        # However, the EMA doesn't represent a true average over the period until `span` points have passed.
        # Stochastic needs `period + smooth_k + smooth_d - 2` data points.
        # ATR needs `period + 1` data points (for the first TR calculation involving previous close).
        required_len_ema_stable = max(CONFIG.fast_ema_period, CONFIG.slow_ema_period, CONFIG.trend_ema_period)
        required_len_stoch = CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 2
        required_len_atr = CONFIG.atr_period + 1

        min_required_len = max(required_len_ema_stable, required_len_stoch, required_len_atr)
        # Add a small buffer to ensure the latest indicator values are not the very first calculated ones
        min_safe_len = min_required_len + max(CONFIG.stoch_smooth_d, 1)  # Add buffer for Stoch smoothing, or just 1

        if len(df_calc) < min_safe_len:
            logger.warning(
                f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for stable indicators (minimum safe: {min_safe_len}). Indicator values might be less reliable. Increase OHLCV_LIMIT or wait for more data."
            )
            # Proceed anyway, but warn

        if len(df_calc) < min_required_len:
            logger.error(
                f"{Fore.RED}Insufficient data ({len(df_calc)}) for core indicator calculations (minimum required: {min_required_len}). Cannot calculate indicators."
            )
            return None  # Critical failure if even minimum isn't met

        # --- Calculations using Pandas and CONFIG periods ---
        fast_ema_series = close.ewm(span=CONFIG.fast_ema_period, adjust=False).mean()
        slow_ema_series = close.ewm(span=CONFIG.slow_ema_period, adjust=False).mean()
        trend_ema_series = close.ewm(span=CONFIG.trend_ema_period, adjust=False).mean()
        # confirm_ema_series = close.ewm(span=CONFIG.confirm_ema_period, adjust=False).mean() # Example if you use a 3rd EMA

        # Stochastic Oscillator %K and %D
        low_min = low.rolling(window=CONFIG.stoch_period).min()
        high_max = high.rolling(window=CONFIG.stoch_period).max()
        # Add epsilon to prevent division by zero if high == low over the period
        stoch_k_raw = 100 * (close - low_min) / (high_max - low_min + 1e-9)  # Use float epsilon for float calc
        stoch_k = stoch_k_raw.rolling(window=CONFIG.stoch_smooth_k).mean()
        stoch_d = stoch_k.rolling(window=CONFIG.stoch_smooth_d).mean()

        # Average True Range (ATR) - Wilder's smoothing matches TradingView standard
        tr_df = pd.DataFrame(index=df_calc.index)
        tr_df["hl"] = high - low
        tr_df["hc"] = (high - close.shift()).abs()
        tr_df["lc"] = (low - close.shift()).abs()
        tr = tr_df[["hl", "hc", "lc"]].max(axis=1)
        # Use ewm with alpha = 1/period for Wilder's smoothing
        atr_series = tr.ewm(alpha=1 / CONFIG.atr_period, adjust=False).mean()

        # --- Extract Latest Values & Convert to Decimal ---
        # Define quantizers for consistent decimal places (adjust as needed)
        # These are for *internal* Decimal representation, not API formatting.
        # Use enough precision to avoid rounding errors before API formatting.
        price_quantizer = Decimal("1E-8")  # 8 decimal places for price-like values
        percent_quantizer = Decimal("1E-2")  # 2 decimal places for Stoch
        atr_quantizer = Decimal("1E-8")  # 8 decimal places for ATR

        # Helper to safely get latest non-NaN value, convert to Decimal, and handle errors
        def get_latest_decimal(
            series: pd.Series, quantizer: Decimal, name: str, default_val: Decimal = Decimal("NaN")
        ) -> Decimal:
            if series.empty or series.isna().all():
                logger.warning(f"Indicator series '{name}' is empty or all NaN.")
                return default_val
            # Get the last valid (non-NaN) value
            latest_valid_val = series.dropna().iloc[-1] if not series.dropna().empty else None

            if latest_valid_val is None:
                # This case should be covered by the initial min_required_len check, but defensive
                logger.warning(f"Indicator calculation for '{name}' resulted in NaN or only NaNs.")
                return default_val
            try:
                # Convert via string for precision, then quantize
                return Decimal(str(latest_valid_val)).quantize(quantizer)
            except (InvalidOperation, TypeError) as e:
                logger.error(
                    f"Could not convert indicator '{name}' value {latest_valid_val} to Decimal: {e}. Returning default."
                )
                return default_val

        # Get latest values
        latest_fast_ema = get_latest_decimal(fast_ema_series, price_quantizer, "fast_ema")
        latest_slow_ema = get_latest_decimal(slow_ema_series, price_quantizer, "slow_ema")
        latest_trend_ema = get_latest_decimal(trend_ema_series, price_quantizer, "trend_ema")
        # latest_confirm_ema = get_latest_decimal(confirm_ema_series, price_quantizer, "confirm_ema") # Example
        latest_stoch_k = get_latest_decimal(
            stoch_k, percent_quantizer, "stoch_k", default_val=Decimal("50.00")
        )  # Default neutral
        latest_stoch_d = get_latest_decimal(
            stoch_d, percent_quantizer, "stoch_d", default_val=Decimal("50.00")
        )  # Default neutral
        latest_atr = get_latest_decimal(atr_series, atr_quantizer, "atr", default_val=Decimal("0.0"))  # Default zero

        # --- Calculate Stochastic Cross Signals (Boolean) ---
        # Requires at least 2 data points for the shift
        stoch_kd_bullish = False
        stoch_kd_bearish = False
        if len(stoch_k) >= 2 and not stoch_k.iloc[-1].is_nan() and not stoch_d.iloc[-1].is_nan():
            try:
                # Get the last two values as Decimals
                stoch_k_last = Decimal(str(stoch_k.iloc[-1]))
                stoch_d_last = Decimal(str(stoch_d.iloc[-1]))
                stoch_k_prev = Decimal(str(stoch_k.iloc[-2]))
                stoch_d_prev = Decimal(str(stoch_d.iloc[-2]))

                # Check for crossover using previous vs current values (Decimal comparison)
                stoch_kd_bullish = (stoch_k_last > stoch_d_last) and (stoch_k_prev <= stoch_d_prev)
                stoch_kd_bearish = (stoch_k_last < stoch_d_last) and (stoch_k_prev >= stoch_d_prev)

                # Ensure crossover happens within the relevant zones for signalling (optional but common)
                # For longs: crossover in oversold zone or just exiting it
                if (
                    stoch_kd_bullish
                    and stoch_k_last > CONFIG.stoch_oversold_threshold
                    and stoch_k_prev > CONFIG.stoch_oversold_threshold
                ):
                    logger.debug(
                        f"Stoch K/D Bullish cross happened above oversold zone ({CONFIG.stoch_oversold_threshold}). Not using for signal."
                    )
                    stoch_kd_bullish = False  # Only signal if cross is *in* or *from* oversold
                # For shorts: crossover in overbought zone or just exiting it
                if (
                    stoch_kd_bearish
                    and stoch_k_last < CONFIG.stoch_overbought_threshold
                    and stoch_k_prev < CONFIG.stoch_overbought_threshold
                ):
                    logger.debug(
                        f"Stoch K/D Bearish cross happened below overbought zone ({CONFIG.stoch_overbought_threshold}). Not using for signal."
                    )
                    stoch_kd_bearish = False  # Only signal if cross is *in* or *from* overbought

            except (InvalidOperation, TypeError) as e:
                logger.warning(f"Error calculating Stoch K/D cross: {e}. Cross signals will be False.")
                stoch_kd_bullish = False
                stoch_kd_bearish = False
        elif len(stoch_k) < 2:
            logger.debug("Not enough data for Stoch K/D cross calculation.")

        indicators_out = {
            "fast_ema": latest_fast_ema,
            "slow_ema": latest_slow_ema,
            "trend_ema": latest_trend_ema,
            # "confirm_ema": latest_confirm_ema, # Example
            "stoch_k": latest_stoch_k,
            "stoch_d": latest_stoch_d,
            "atr": latest_atr,
            "atr_period": CONFIG.atr_period,  # Store period for display
            "stoch_kd_bullish": stoch_kd_bullish,  # Add cross signals
            "stoch_kd_bearish": stoch_kd_bearish,  # Add cross signals
        }

        # Check if any crucial indicator calculation failed (returned NaN default)
        critical_indicators = ["fast_ema", "slow_ema", "trend_ema", "stoch_k", "stoch_d", "atr"]
        failed_indicators = [key for key in critical_indicators if indicators_out[key].is_nan()]

        if failed_indicators:
            logger.error(
                f"{Fore.RED}One or more critical indicators failed to calculate (NaN): {', '.join(failed_indicators)}"
            )
            return None  # Signal failure

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        # logger.debug(f"Latest Indicators: { {k: str(v) for k, v in indicators_out.items()} }") # Log values at debug, convert Decimal to str for clean log
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
        "long": {
            "qty": Decimal("0.0"),
            "entry_price": Decimal("NaN"),
            "liq_price": Decimal("NaN"),
            "pnl": Decimal("NaN"),
        },
        "short": {
            "qty": Decimal("0.0"),
            "entry_price": Decimal("NaN"),
            "liq_price": Decimal("NaN"),
            "pnl": Decimal("NaN"),
        },
    }

    positions_data = None
    try:
        # fetch_with_retries handles category param automatically
        # Note: Bybit V5 fetch_positions might return multiple entries per symbol (e.g., isolated/cross, or different sides).
        # We assume one-way mode and sum up quantities/use average price if necessary, or just process the first entry found for each side.
        # For simplicity, this code assumes one-way mode and picks the first long/short entry.
        positions_data = fetch_with_retries(EXCHANGE.fetch_positions, symbols=[symbol])
    except Exception as e:
        # Handle potential exceptions raised by fetch_with_retries itself (e.g., AuthenticationError, Non-retryable ExchangeError)
        logger.error(
            Fore.RED + f"Unhandled exception during fetch_positions call via fetch_with_retries: {e}", exc_info=True
        )
        return None  # Indicate failure

    if positions_data is None:
        # fetch_with_retries already logged the failure reason
        logger.error(Fore.RED + f"Failed to fetch positions for {symbol}.")
        return None  # Indicate failure

    if not isinstance(positions_data, list):
        logger.error(
            f"Unexpected data type received from fetch_positions: {type(positions_data)}. Expected list. Data: {str(positions_data)[:200]}"
        )
        return None

    if not positions_data:
        logger.info(Fore.BLUE + f"No open positions reported by exchange for {symbol}.")
        return pos_dict  # Return the initialized zero dictionary

    # Process the fetched positions - find the primary long/short position for the symbol
    long_pos_found = False
    short_pos_found = False

    for pos in positions_data:
        # Ensure pos is a dictionary
        if not isinstance(pos, dict):
            logger.warning(f"Skipping non-dictionary item in positions data: {pos}")
            continue

        pos_symbol = pos.get("symbol")
        if pos_symbol != symbol:
            logger.debug(f"Ignoring position data for different symbol: {pos_symbol}")
            continue

        # Use info dictionary for safer access to raw exchange data if needed
        pos_info = pos.get("info", {})
        if not isinstance(pos_info, dict):  # Ensure info is a dict
            pos_info = {}

        # Determine side ('long' or 'short') - check unified field first, then 'info'
        side = pos.get("side")  # Unified field
        if side not in ["long", "short"]:
            # Fallback for Bybit V5 'info' field if unified 'side' is missing/invalid
            side_raw = pos_info.get("side", "").lower()  # e.g., "Buy" or "Sell"
            if side_raw == "buy":
                side = "long"
            elif side_raw == "sell":
                side = "short"
            else:
                logger.warning(f"Could not determine side for position: Info={str(pos_info)[:100]}. Skipping.")
                continue

        # If we already processed this side, skip (assuming one-way mode, first entry per side is sufficient)
        if (side == "long" and long_pos_found) or (side == "short" and short_pos_found):
            logger.debug(
                f"Already processed a {side} position for {symbol}. Skipping subsequent entries for this side."
            )
            continue  # Skip processing this entry

        # Get quantity ('contracts' or 'size') - Use unified field first, fallback to info
        contracts_str = pos.get("contracts")  # Unified field ('contracts' seems standard)
        if contracts_str is None:
            contracts_str = pos_info.get("size")  # Common Bybit V5 field in 'info'

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
                    continue  # Skip processing this entry

                # Convert other fields, handling potential None or invalid values
                entry_price = (
                    Decimal(str(entry_price_str))
                    if entry_price_str is not None and entry_price_str != ""
                    else Decimal("NaN")
                )
                liq_price = (
                    Decimal(str(liq_price_str)) if liq_price_str is not None and liq_price_str != "" else Decimal("NaN")
                )
                pnl = Decimal(str(pnl_str)) if pnl_str is not None and pnl_str != "" else Decimal("NaN")

                # Assign to the dictionary
                pos_dict[side]["qty"] = contracts
                pos_dict[side]["entry_price"] = entry_price
                pos_dict[side]["liq_price"] = liq_price
                pos_dict[side]["pnl"] = pnl

                # Mark side as found
                if side == "long":
                    long_pos_found = True
                else:
                    short_pos_found = True

                # Log with formatted decimals (for display)
                entry_log = f"{entry_price:.4f}" if not entry_price.is_nan() else "N/A"
                liq_log = f"{liq_price:.4f}" if not liq_price.is_nan() else "N/A"
                pnl_log = f"{pnl:+.4f}" if not pnl.is_nan() else "N/A"
                logger.info(
                    Fore.YELLOW
                    + f"Found active {side.upper()} position: Qty={contracts}, Entry={entry_log}, Liq≈{liq_log}, PnL≈{pnl_log}"
                )

            except (InvalidOperation, TypeError) as e:
                logger.error(
                    f"Could not parse position data for {side} side: Qty='{contracts_str}', Entry='{entry_price_str}', Liq='{liq_price_str}', PnL='{pnl_str}'. Error: {e}"
                )
                # Do not continue here, this specific position entry is problematic.
                # The pos_dict[side] will retain its default NaN/0 values.
                continue
        elif side not in pos_dict:
            logger.warning(f"Position data found for unknown side '{side}'. Skipping.")

    if not long_pos_found and not short_pos_found:
        logger.info(Fore.BLUE + f"No active non-zero positions found for {symbol} after filtering.")
    elif long_pos_found and short_pos_found:
        logger.warning(
            Fore.YELLOW
            + f"Both LONG and SHORT positions found for {symbol}. Pyrmethus assumes one-way mode and will manage the first position found for each side. Please ensure your exchange account is configured for one-way trading."
        )

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
        # Example params if needed for specific account types (adjust as per Bybit V5 docs):
        # params = {'accountType': 'UNIFIED'}
        balance_data = fetch_with_retries(EXCHANGE.fetch_balance)
    except Exception as e:
        logger.error(
            Fore.RED + f"Unhandled exception during fetch_balance call via fetch_with_retries: {e}", exc_info=True
        )
        return None, None

    if balance_data is None:
        # fetch_with_retries already logged the failure
        logger.error(Fore.RED + "Failed to fetch balance after retries. Cannot assess risk capital.")
        return None, None

    # --- Parse Balance Data ---
    # Initialize with NaN Decimals to indicate failure to find/parse
    free_balance = Decimal("NaN")
    total_balance = Decimal("NaN")  # Represents Equity for futures/swaps

    try:
        # CCXT unified structure: balance_data[currency]['free'/'total']
        if currency in balance_data and isinstance(balance_data[currency], dict):
            currency_balance = balance_data[currency]
            free_str = currency_balance.get("free")
            total_str = currency_balance.get("total")  # 'total' usually represents equity in futures

            if free_str is not None:
                free_balance = Decimal(str(free_str))
            if total_str is not None:
                total_balance = Decimal(str(total_str))

        # Alternative structure: balance_data['free'][currency], balance_data['total'][currency]
        # Less common for V5, but included for robustness
        elif "free" in balance_data and isinstance(balance_data["free"], dict) and currency in balance_data["free"]:
            free_str = balance_data["free"].get(currency)
            total_str = balance_data.get("total", {}).get(currency)  # Total might still be top-level

            if free_str is not None:
                free_balance = Decimal(str(free_str))
            if total_str is not None:
                total_balance = Decimal(str(total_str))

        # Fallback: Check 'info' for exchange-specific structure (Bybit V5 example)
        # This is the most reliable for Bybit V5 Unified Margin/Contract accounts
        elif "info" in balance_data and isinstance(balance_data["info"], dict):
            info_data = balance_data["info"]
            # V5 structure: result -> list -> account objects
            if (
                "result" in info_data
                and isinstance(info_data["result"], dict)
                and "list" in info_data["result"]
                and isinstance(info_data["result"]["list"], list)
            ):
                for account in info_data["result"]["list"]:
                    # Find the account object for the target currency
                    if isinstance(account, dict) and account.get("coin") == currency:
                        # Bybit V5 Unified Margin fields (check docs):
                        # 'walletBalance': Total assets in wallet
                        # 'availableToWithdraw': Amount withdrawable
                        # 'equity': Account equity (often the most relevant for risk calculation in futures)
                        # 'availableToBorrow': Margin specific
                        # 'totalPerpUPL': Unrealized PnL (already included in equity)
                        equity_str = account.get("equity")  # Use equity as 'total' for risk calculation
                        free_str = account.get("availableToWithdraw")  # Use availableToWithdraw as 'free'

                        if free_str is not None:
                            free_balance = Decimal(str(free_str))
                        if equity_str is not None:
                            total_balance = Decimal(str(equity_str))
                        logger.debug(
                            f"Parsed Bybit V5 info structure for {currency}: Free={free_balance}, Equity={total_balance}"
                        )
                        break  # Found the currency account

        # If parsing failed, balances will remain NaN
        if free_balance.is_nan():
            logger.warning(f"Could not find or parse free balance for {currency} in balance data.")
        if total_balance.is_nan():
            logger.warning(f"Could not find or parse total/equity balance for {currency} in balance data.")
            # Critical if equity is needed for risk calc
            logger.error(Fore.RED + "Failed to determine account equity. Cannot proceed safely.")
            return free_balance, None  # Indicate equity failure specifically

        # Use 'total' balance (Equity) as the primary value for risk calculation
        equity = total_balance

        logger.info(Fore.GREEN + f"Vault contains {free_balance:.4f} free {currency} (Equity/Total: {equity:.4f}).")
        return free_balance, equity  # Return free and total (equity)

    except (InvalidOperation, TypeError, KeyError) as e:
        logger.error(
            Fore.RED
            + f"Error parsing balance data for {currency}: {e}. Raw keys: {list(balance_data.keys()) if isinstance(balance_data, dict) else 'N/A'}"
        )
        logger.debug(f"Raw balance data: {balance_data}")
        return None, None  # Indicate parsing failure
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
    last_status = "unknown"
    attempt = 0
    check_interval = 1.5  # seconds between checks

    while time.time() - start_time < timeout:
        attempt += 1
        logger.debug(f"Checking order {order_id}, attempt {attempt}...")
        order_status_data = None
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            # Category param should be handled automatically by fetch_with_retries
            order_status_data = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol)

            if order_status_data and isinstance(order_status_data, dict):
                last_status = order_status_data.get("status", "unknown")
                filled_qty_raw = order_status_data.get("filled", 0.0)
                filled_qty = Decimal(str(filled_qty_raw))  # Convert to Decimal for accurate comparison

                logger.info(
                    f"Order {order_id} status check: {last_status}, Filled: {filled_qty.normalize()}"
                )  # Use normalize for cleaner log

                # Check for terminal states (fully filled, canceled, rejected, expired)
                # 'closed' usually means fully filled for market/limit orders on Bybit.
                if last_status in ["closed", "canceled", "rejected", "expired"]:
                    logger.info(f"Order {order_id} reached terminal state: {last_status}.")
                    return order_status_data  # Return the final order dict
                # If 'open' but fully filled (can happen briefly), treat as terminal 'closed'
                # Check remaining amount using epsilon
                remaining_qty_raw = order_status_data.get("remaining", 0.0)
                remaining_qty = Decimal(str(remaining_qty_raw))
                if (
                    last_status == "open"
                    and remaining_qty < CONFIG.position_qty_epsilon
                    and filled_qty >= CONFIG.position_qty_epsilon
                ):
                    logger.info(
                        f"Order {order_id} is 'open' but fully filled ({filled_qty.normalize()}). Treating as 'closed'."
                    )
                    order_status_data["status"] = "closed"  # Update status locally for clarity
                    return order_status_data

            else:
                # fetch_with_retries failed or returned unexpected data
                # Error logged within fetch_with_retries, just note it here
                logger.warning(
                    f"fetch_order call failed or returned invalid data for {order_id}. Continuing check loop."
                )
                # Continue the loop to retry check_order_status itself

        except ccxt.OrderNotFound:
            # Order is definitively not found. This is a terminal state indicating it never existed or was fully purged.
            logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND by exchange.")
            return None  # Explicitly indicate not found
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
            # Critical non-retryable errors
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"Authentication/Permission error during order status check for {order_id}: {e}. Halting."
            )
            sys.exit(1)
        except Exception as e:
            # Catch any other unexpected error during the check itself
            logger.error(f"Unexpected error during order status check loop for {order_id}: {e}", exc_info=True)
            # Decide whether to retry or fail; retrying is part of the loop.

        # Wait before the next check_order_status attempt
        time_elapsed = time.time() - start_time
        if time_elapsed + check_interval < timeout:
            logger.debug(f"Order {order_id} status ({last_status}) not terminal, sleeping {check_interval:.1f}s...")
            time.sleep(check_interval)
            check_interval = min(check_interval * 1.2, 5)  # Slightly increase interval up to 5s
        else:
            break  # Exit loop if next sleep would exceed timeout

    # --- Timeout Reached ---
    logger.error(
        Fore.RED
        + f"Timed out checking status for order {order_id} after {timeout} seconds. Last known status: {last_status}."
    )
    # Attempt one final fetch outside the loop to get the very last state if possible
    final_check_status = None
    try:
        logger.info(f"Performing final status check for order {order_id} after timeout...")
        final_check_status = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol)
        if final_check_status:
            logger.info(
                f"Final status after timeout: {final_check_status.get('status', 'unknown')}, Filled: {final_check_status.get('filled', 'N/A')}"
            )
            # Return this final status even if timed out earlier
            return final_check_status
        else:
            logger.error(f"Final status check for order {order_id} also failed.")
            return None  # Indicate persistent failure to get status
    except ccxt.OrderNotFound:
        logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND on final check.")
        return None
    except Exception as e:
        logger.error(f"Error during final status check for order {order_id}: {e}")
        return None  # Indicate failure


def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    trade_action = f"{side.upper()} Market Entry"
    logger.trade(Style.BRIGHT + f"Attempting {trade_action} for {symbol}...")

    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(Fore.RED + f"{trade_action} failed: Market info or Exchange not available.")
        return False

    # --- Pre-computation & Validation ---
    quote_currency = MARKET_INFO.get("settle", "USDT")  # Use settle currency (e.g., USDT)
    _, total_equity = get_balance(quote_currency)  # Fetch balance using the function
    if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
        logger.error(
            Fore.RED
            + f"{trade_action} failed: Invalid, NaN, or zero account equity ({total_equity}). Cannot calculate risk capital."
        )
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
        logger.error(
            Fore.RED
            + f"{trade_action} failed: Cannot fetch current ticker price for sizing/SL calculation. Ticker data: {ticker_data}"
        )
        # fetch_with_retries should have logged details if it failed
        return False

    try:
        # Use 'last' price as current price estimate, convert to Decimal
        price = Decimal(str(ticker_data["last"]))
        if price <= Decimal(0):
            logger.error(
                Fore.RED + f"{trade_action} failed: Fetched current price ({price}) is zero or negative. Aborting."
            )
            return False
        logger.debug(f"Current ticker price: {price:.8f} {quote_currency}")  # Log with high precision for debug

        # --- Calculate Stop Loss Price ---
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if sl_distance_points <= Decimal("0"):  # Use Decimal zero
            logger.error(
                f"{Fore.RED}{trade_action} failed: Stop distance calculation resulted in zero or negative value ({sl_distance_points}). Check ATR ({atr:.6f}) and multiplier ({CONFIG.sl_atr_multiplier})."
            )
            return False

        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else:  # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations/API call
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        # Convert back to Decimal *after* formatting for consistent internal representation
        sl_price = Decimal(sl_price_formatted_str)
        logger.debug(
            f"ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier}, SL Distance Points: {sl_distance_points:.6f}"
        )
        logger.debug(
            f"Raw SL Price: {sl_price_raw:.8f}, Formatted SL Price for API: {sl_price_formatted_str} (Decimal: {sl_price})"
        )

        # Sanity check SL placement relative to current price
        # Use a small multiple of price tick size for tolerance if available, else a tiny Decimal
        try:
            price_precision_info = MARKET_INFO["precision"].get("price")
            # If precision is number of decimals (int), calculate tick size 1 / (10^decimals)
            if isinstance(price_precision_info, int):
                price_tick_size = Decimal(1) / (Decimal(10) ** price_precision_info)
            # If precision is tick size (string or Decimal)
            elif isinstance(price_precision_info, (str, Decimal)):
                price_tick_size = Decimal(str(price_precision_info))
            else:
                price_tick_size = Decimal("1E-8")  # Fallback tiny Decimal
        except Exception:
            price_tick_size = Decimal("1E-8")  # Fallback tiny Decimal

        tolerance_ticks = price_tick_size * Decimal("5")  # Allow a few ticks tolerance

        if side == "buy" and sl_price >= price - tolerance_ticks:
            logger.error(
                Fore.RED
                + f"{trade_action} failed: Calculated SL price ({sl_price}) is too close to or above current price ({price}) [Tolerance: {tolerance_ticks}]. Check ATR/multiplier or market precision. Aborting."
            )
            return False
        if side == "sell" and sl_price <= price + tolerance_ticks:
            logger.error(
                Fore.RED
                + f"{trade_action} failed: Calculated SL price ({sl_price}) is too close to or below current price ({price}) [Tolerance: {tolerance_ticks}]. Check ATR/multiplier or market precision. Aborting."
            )
            return False

        # --- Calculate Position Size ---
        risk_amount_quote = total_equity * risk_percentage
        # Stop distance in quote currency (use absolute difference, ensure Decimals)
        stop_distance_quote = (price - sl_price).copy_abs()

        if stop_distance_quote <= Decimal("0"):
            logger.error(
                Fore.RED
                + f"{trade_action} failed: Stop distance in quote currency is zero or negative ({stop_distance_quote}). Check ATR, multiplier, or market precision. Cannot calculate size."
            )
            return False

        # Calculate quantity based on contract size and linear/inverse type
        contract_size = Decimal(str(MARKET_INFO.get("contractSize", "1")))  # Ensure Decimal
        qty_raw = Decimal("0")

        # --- Sizing Logic ---
        # Bybit uses size in Base currency for Linear (e.g., BTC for BTC/USDT)
        # Bybit uses size in Contracts (which represent USD value for BTC/USD inverse)
        if CONFIG.market_type == "linear":
            # Linear (e.g., BTC/USDT:USDT): Size is in Base currency (BTC). Value = Size * Price.
            # Risk Amount (Quote) = Qty (Base) * Stop Distance (Quote)
            # Qty (Base) = Risk Amount (Quote) / Stop Distance (Quote)
            qty_raw = risk_amount_quote / stop_distance_quote
            logger.debug(
                f"Linear Sizing: Qty (Base) = {risk_amount_quote:.8f} {quote_currency} / {stop_distance_quote:.8f} {quote_currency} = {qty_raw:.8f}"
            )

        elif CONFIG.market_type == "inverse":
            # Inverse (e.g., BTC/USD:BTC): Size is in Contracts. For BTC/USD on Bybit, 1 Contract = 1 USD.
            # Risk (USD) = Qty (Contracts) * Stop Distance (USD per Contract)
            # Stop Distance (USD per Contract) = (Price_entry - SL_Price_entry) * Contract_Size (Base/Contract)
            # Bybit Inverse contract size (Base/Contract) is 1/Price (Quote/Base) for USD pairs.
            # Example BTC/USD: Contract Size = 1/Price (USD/BTC). So 1 Contract = 1/Price (USD/BTC) BTC.
            # This is confusing. A simpler view for Inverse (Bybit V5):
            # Size is in Contracts. 1 Contract = $1 for BTC/USD.
            # Position Value (in Base) = Qty (Contracts) * Contract Size (Base/Contract)
            # Let's re-evaluate the core risk formula:
            # Risk Amount (Quote) = (Entry Price - SL Price) * Position Quantity (in Base)
            # Position Quantity (in Base) = Qty (Contracts) * Contract Size (Base/Contract)
            # Contract Size (Base/Contract) = 1 / Price (Quote/Base) for Bybit Inverse USD pairs
            # So, Risk (Quote) = (Entry Price - SL Price) * Qty (Contracts) * (1 / Price (Quote/Base))
            # Risk (Quote) = Stop Distance (Quote) * Qty (Contracts) / Price (Quote/Base)
            # Qty (Contracts) = Risk (Quote) * Price (Quote/Base) / Stop Distance (Quote)
            # This confirms the formula used previously is correct for Bybit Inverse USD pairs where Contract Size is 1 USD.
            if price <= Decimal("0"):
                logger.error(
                    Fore.RED + f"{trade_action} failed: Cannot calculate inverse size with zero or negative price."
                )
                return False
            qty_raw = (risk_amount_quote * price) / stop_distance_quote
            logger.debug(
                f"Inverse Sizing (Contract Size = {contract_size} {quote_currency}): Qty (Contracts) = ({risk_amount_quote:.8f} * {price:.8f}) / {stop_distance_quote:.8f} = {qty_raw:.8f}"
            )

        else:
            logger.error(f"{trade_action} failed: Unsupported market type for sizing: {CONFIG.market_type}")
            return False

        # --- Format and Validate Quantity ---
        # Format quantity according to market precision (ROUND_DOWN to be conservative)
        qty_formatted_str = format_amount(symbol, qty_raw, ROUND_DOWN)
        qty = Decimal(qty_formatted_str)
        logger.debug(
            f"Risk Amount: {risk_amount_quote:.8f} {quote_currency}, Stop Distance: {stop_distance_quote:.8f} {quote_currency}"
        )
        logger.debug(f"Raw Qty: {qty_raw:.12f}, Formatted Qty (Rounded Down): {qty}")

        # Validate Quantity Against Market Limits
        min_qty_str = (
            str(MARKET_INFO["limits"]["amount"]["min"])
            if MARKET_INFO["limits"]["amount"].get("min") is not None
            else "0"
        )
        max_qty_str = (
            str(MARKET_INFO["limits"]["amount"]["max"])
            if MARKET_INFO["limits"]["amount"].get("max") is not None
            else None
        )
        min_qty = Decimal(min_qty_str)
        # max_qty is infinity if None, otherwise convert to Decimal
        max_qty = Decimal(str(max_qty_str)) if max_qty_str is not None else Decimal("Infinity")

        # Use epsilon for zero check
        if qty < min_qty or qty < CONFIG.position_qty_epsilon:
            logger.error(
                Fore.RED
                + f"{trade_action} failed: Calculated quantity ({qty}) is zero or below minimum ({min_qty}, epsilon {CONFIG.position_qty_epsilon:.1E}). Risk amount ({risk_amount_quote:.4f}), stop distance ({stop_distance_quote:.4f}), or equity might be too small. Cannot place order."
            )
            return False
        if max_qty != Decimal("Infinity") and qty > max_qty:
            logger.warning(
                Fore.YELLOW + f"Calculated quantity {qty} exceeds maximum {max_qty}. Capping order size to {max_qty}."
            )
            qty = max_qty  # Use the Decimal max_qty
            # Re-format capped amount - crucial! Use ROUND_DOWN again.
            qty_formatted_str = format_amount(symbol, qty, ROUND_DOWN)
            qty = Decimal(qty_formatted_str)
            logger.info(f"Re-formatted capped Qty: {qty}")
            # Double check if capped value is now below min (unlikely but possible with large steps)
            if qty < min_qty or qty < CONFIG.position_qty_epsilon:
                logger.error(
                    Fore.RED
                    + f"{trade_action} failed: Capped quantity ({qty}) is now below minimum ({min_qty}) or zero. Aborting."
                )
                return False

        # Validate minimum cost if available
        min_cost_str = (
            str(MARKET_INFO["limits"].get("cost", {}).get("min"))
            if MARKET_INFO["limits"].get("cost", {}).get("min") is not None
            else None
        )
        if min_cost_str is not None:
            min_cost = Decimal(min_cost_str)
            estimated_cost = Decimal("0")
            # Estimate cost based on market type (Approximate!)
            try:
                if CONFIG.market_type == "linear":
                    # Cost = Qty (Base) * Price (Quote/Base) = Quote
                    estimated_cost = qty * price
                elif CONFIG.market_type == "inverse":
                    # Cost = Qty (Contracts) * Contract Size (Quote/Contract) = Quote
                    # Assuming contract size is in Quote currency (e.g., 1 USD for BTC/USD)
                    estimated_cost = qty * contract_size  # Check if contract_size needs conversion if not in quote
                    logger.debug(
                        f"Inverse cost estimation: Qty({qty}) * ContractSize({contract_size}) = {estimated_cost}"
                    )
                else:
                    estimated_cost = Decimal("0")  # Should not happen

                if estimated_cost < min_cost:
                    logger.error(
                        Fore.RED
                        + f"{trade_action} failed: Estimated order cost/value ({estimated_cost:.4f} {quote_currency}) is below minimum required ({min_cost:.4f} {quote_currency}). Increase risk or equity. Cannot place order."
                    )
                    return False
            except Exception as cost_err:
                logger.warning(f"Could not estimate order cost: {cost_err}. Skipping min cost check.")

        logger.info(
            Fore.YELLOW
            + f"Calculated Order: Side={side.upper()}, Qty={qty}, Entry≈{price:.4f}, SL={sl_price} (ATR={atr:.4f})"
        )

    except (InvalidOperation, TypeError, DivisionByZero, KeyError) as e:
        logger.error(
            Fore.RED + Style.BRIGHT + f"{trade_action} failed: Error during pre-calculation/validation: {e}",
            exc_info=True,
        )
        return False
    except Exception as e:  # Catch any other unexpected errors
        logger.error(
            Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during pre-calculation: {e}",
            exc_info=True,
        )
        return False

    # --- Cast the Market Order Spell ---
    order = None
    order_id = None
    filled_qty = Decimal("0.0")  # Initialize filled_qty for later use
    average_price = price  # Initialize average_price for later use

    try:
        logger.trade(f"Submitting {side.upper()} market order for {qty.normalize()} {symbol}...")
        # fetch_with_retries handles category param
        # CCXT expects float amount
        order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol,
            side=side,
            amount=float(qty),  # Explicitly cast Decimal qty to float for CCXT API
        )

        if order is None:
            # fetch_with_retries logged the error
            logger.error(Fore.RED + f"{trade_action} failed: Market order placement failed after retries.")
            return False

        logger.debug(f"Market order raw response: {order}")
        order_id = order.get("id")

        # --- Verify Order Fill (Crucial Step) ---
        # Bybit V5 create_market_order might return retCode=0 without an order ID in the standard field immediately.
        # The actual filled order details might be in 'info'->'result'->'list'.
        # Let's check retCode first if ID is missing, and try to extract details.
        order_status_data = None  # Initialize as None to indicate status check is needed
        if not order_id:
            if isinstance(order.get("info"), dict):
                ret_code = order["info"].get("retCode")
                ret_msg = order["info"].get("retMsg")
                if ret_code == 0:  # Bybit V5 success code
                    logger.warning(
                        f"{trade_action}: Market order submitted successfully (retCode 0) but no Order ID returned in standard field immediately. This is common for V5 market orders. Proceeding to check fill status."
                    )
                    # Try to extract order ID or details from the V5 result list if available
                    if (
                        "result" in order["info"]
                        and isinstance(order["info"]["result"], dict)
                        and "list" in order["info"]["result"]
                        and isinstance(order["info"]["result"]["list"], list)
                        and order["info"]["result"]["list"]
                    ):
                        # For market orders, the list should contain the immediately filled order(s)
                        first_order_info = order["info"]["result"]["list"][0]
                        order_id = first_order_info.get("orderId")
                        # Also capture filled details from response if possible (more accurate than check_order_status if available)
                        filled_qty_from_response_raw = first_order_info.get("cumExecQty", "0")  # Bybit V5 field
                        avg_price_from_response_raw = first_order_info.get("avgPrice", "NaN")  # Bybit V5 field
                        try:
                            filled_qty_from_response = Decimal(str(filled_qty_from_response_raw))
                            avg_price_from_response = Decimal(str(avg_price_from_response_raw))
                        except InvalidOperation:
                            logger.error(
                                f"Could not parse filled qty ({filled_qty_from_response_raw}) or avg price ({avg_price_from_response_raw}) from V5 response."
                            )
                            filled_qty_from_response = Decimal("0")
                            avg_price_from_response = Decimal("NaN")

                        logger.debug(
                            f"Extracted details from V5 response: ID={order_id}, Filled={filled_qty_from_response.normalize()}, AvgPrice={avg_price_from_response}"
                        )
                        # Use extracted details if valid
                        if filled_qty_from_response.copy_abs() >= CONFIG.position_qty_epsilon:
                            filled_qty = filled_qty_from_response
                            average_price = (
                                avg_price_from_response if not avg_price_from_response.is_nan() else price
                            )  # Use estimated price if avgPrice is NaN
                            logger.trade(
                                Fore.GREEN
                                + Style.BRIGHT
                                + f"Market order confirmed FILLED from response: {filled_qty.normalize()} @ {average_price:.4f}"
                            )
                            # Synthesize a CCXT-like dict for consistency if needed later, but primarily use filled_qty/average_price
                            order_status_data = {
                                "status": "closed",
                                "filled": float(filled_qty),
                                "average": float(average_price) if not average_price.is_nan() else None,
                                "id": order_id,
                            }
                            logger.debug("Skipping check_order_status due to immediate fill confirmation in response.")
                            # Proceed to SL placement block immediately below
                        else:
                            logger.warning(
                                f"{trade_action}: Order ID found ({order_id}) but filled quantity from response ({filled_qty_from_response.normalize()}) is zero or negligible. Will proceed with check_order_status."
                            )
                            # Need to proceed with status check if filled qty is zero in response
                            # order_id is set, order_status_data is None -> check_order_status will be called
                    else:
                        logger.warning(
                            f"{trade_action}: Market order submitted (retCode 0) but no Order ID or fill details found in V5 result list. Cannot reliably track status. Aborting."
                        )
                        return False  # Cannot proceed safely without order ID
                else:
                    logger.error(
                        Fore.RED
                        + f"{trade_action} failed: Market order submission failed. Exchange message: {ret_msg} (Code: {ret_code})"
                    )
                    return False
            else:
                logger.error(
                    Fore.RED + f"{trade_action} failed: Market order submission failed to return an ID or success info."
                )
                return False
        else:  # Order ID was found in the standard field
            logger.trade(f"Market order submitted: ID {order_id}")
            # Proceed to verify fill status via check_order_status

        # If order_status_data is None here, it means we got an order_id but no immediate fill confirmation in the response
        if order_status_data is None:
            logger.info(
                f"Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}..."
            )
            time.sleep(CONFIG.order_check_delay_seconds)

            # Use the dedicated check_order_status function
            order_status_data = check_order_status(order_id, symbol, timeout=CONFIG.order_check_timeout_seconds)

        order_final_status = "unknown"
        if order_status_data and isinstance(order_status_data, dict):
            order_final_status = order_status_data.get("status", "unknown")
            filled_str = order_status_data.get("filled")
            average_str = order_status_data.get("average")  # Average fill price

            if filled_str is not None:
                try:
                    filled_qty = Decimal(str(filled_str))
                except InvalidOperation:
                    logger.error(f"Could not parse filled quantity '{filled_str}' to Decimal.")
            if average_str is not None:
                try:
                    avg_price_decimal = Decimal(str(average_str))
                    if avg_price_decimal > 0:  # Use actual fill price only if valid
                        average_price = avg_price_decimal
                except InvalidOperation:
                    logger.error(f"Could not parse average price '{average_str}' to Decimal.")

            logger.debug(
                f"Order {order_id} status check result: Status='{order_final_status}', Filled='{filled_qty.normalize()}', AvgPrice='{average_price:.4f}'"
            )

            # 'closed' means fully filled for market orders on Bybit
            if order_final_status == "closed" and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                logger.trade(
                    Fore.GREEN
                    + Style.BRIGHT
                    + f"Order {order_id} confirmed FILLED: {filled_qty.normalize()} @ {average_price:.4f}"
                )
            # Handle partial fills (less common for market, but possible during high volatility)
            # Bybit V5 market orders typically fill fully or are rejected. If partially filled, something unusual is happening.
            elif (
                order_final_status in ["open", "partially_filled"]
                and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon
            ):
                logger.warning(
                    Fore.YELLOW
                    + f"Market Order {order_id} status is '{order_final_status}' but partially/fully filled ({filled_qty.normalize()}). This is unusual for market orders. Proceeding with filled amount."
                )
                # Assume the filled quantity is the position size and proceed.
            elif (
                order_final_status in ["open", "partially_filled"]
                and filled_qty.copy_abs() < CONFIG.position_qty_epsilon
            ):
                logger.error(
                    Fore.RED
                    + f"{trade_action} failed: Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty.normalize()}). Aborting SL placement."
                )
                # Attempt to cancel just in case it's stuck (defensive)
                try:
                    logger.info(f"Attempting cancellation of stuck/unfilled order {order_id}.")
                    # Bybit V5 cancel_order requires category and symbol
                    cancel_params = {"category": CONFIG.market_type, "symbol": MARKET_INFO["id"]}
                    fetch_with_retries(
                        EXCHANGE.cancel_order, order_id, symbol, params=cancel_params
                    )  # Use fetch_with_retries
                except Exception as cancel_err:
                    logger.warning(f"Failed to cancel stuck order {order_id}: {cancel_err}")
                return False
            else:  # canceled, rejected, expired, failed, unknown, or closed with zero fill
                logger.error(
                    Fore.RED
                    + Style.BRIGHT
                    + f"{trade_action} failed: Order {order_id} did not fill successfully: Status '{order_final_status}', Filled Qty: {filled_qty.normalize()}. Aborting SL placement."
                )
                # Attempt to cancel if not already in a terminal state (defensive)
                if order_final_status not in ["canceled", "rejected", "expired"]:
                    try:
                        logger.info(f"Attempting cancellation of failed/unknown status order {order_id}.")
                        # Bybit V5 cancel_order requires category and symbol
                        cancel_params = {"category": CONFIG.market_type, "symbol": MARKET_INFO["id"]}
                        fetch_with_retries(
                            EXCHANGE.cancel_order, order_id, symbol, params=cancel_params
                        )  # Use fetch_with_retries
                    except Exception:
                        pass  # Ignore errors here, main goal failed anyway
                return False
        else:
            # check_order_status already logged error (e.g., timeout or not found)
            logger.error(
                Fore.RED
                + Style.BRIGHT
                + f"{trade_action} failed: Could not determine final status for order {order_id}. Assuming failure. Aborting SL placement."
            )
            # Attempt to cancel just in case it's stuck somehow (defensive)
            try:
                logger.info(f"Attempting cancellation of unknown status order {order_id}.")
                # If order_id was None earlier, this will fail. check_order_status should handle None ID internally if possible, but better to have ID.
                if order_id:
                    # Bybit V5 cancel_order requires category and symbol
                    cancel_params = {"category": CONFIG.market_type, "symbol": MARKET_INFO["id"]}
                    fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)
                else:
                    logger.warning("Cannot attempt cancellation: No order ID available.")
            except Exception:
                pass
            return False

        # Final check on filled quantity after status check
        if filled_qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.error(
                Fore.RED
                + f"{trade_action} failed: Order {order_id} resulted in effectively zero filled quantity ({filled_qty.normalize()}) after status check. No position opened."
            )
            return False

        # --- Place Initial Stop-Loss Order (Set on Position for Bybit V5) ---
        position_side = "long" if side == "buy" else "short"
        logger.trade(
            f"Setting initial SL for new {position_side.upper()} position (filled qty: {filled_qty.normalize()})..."
        )

        # Use the SL price calculated earlier, already formatted string
        sl_price_str_for_api = sl_price_formatted_str

        # Define parameters for setting the stop-loss on the position (Bybit V5 specific)
        # We use the `private_post_position_set_trading_stop` implicit method via CCXT
        # This endpoint applies to the *entire* position for the symbol/side/category.
        set_sl_params = {
            "category": CONFIG.market_type,  # Required
            "symbol": MARKET_INFO["id"],  # Use exchange-specific market ID
            "stopLoss": sl_price_str_for_api,  # Trigger price for the stop loss
            "slTriggerBy": CONFIG.sl_trigger_by,  # e.g., 'LastPrice', 'MarkPrice'
            "tpslMode": "Full",  # Apply SL/TP/TSL to the entire position ('Partial' also possible)
            # 'positionIdx': 0 # For Bybit unified: 0 for one-way mode (default), 1/2 for hedge mode
            # Note: We don't need quantity here as it applies to the existing position matching symbol/category/side.
            # No need to specify side in params for V5 set-trading-stop, it's determined by the symbol and position context.
            # Wait, the Bybit V5 docs *do* show 'side' as a parameter for set-trading-stop. Let's add it for clarity and correctness.
            "side": "Buy" if position_side == "long" else "Sell",  # Add side parameter
        }
        logger.trade(
            f"Setting Position SL: Trigger={sl_price_str_for_api}, TriggerBy={CONFIG.sl_trigger_by}, Side={set_sl_params['side']}"
        )
        logger.debug(f"Set SL Params (for setTradingStop): {set_sl_params}")

        sl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods if available
            # Endpoint: POST /v5/position/set-trading-stop
            if hasattr(EXCHANGE, "private_post_position_set_trading_stop"):
                sl_set_response = fetch_with_retries(
                    EXCHANGE.private_post_position_set_trading_stop, params=set_sl_params
                )
            else:
                # Fallback: Raise error if specific method missing.
                logger.error(
                    Fore.RED
                    + "Cannot set SL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit."
                )
                # Critical: Position is open without SL. Attempt emergency close.
                raise ccxt.NotSupported("SL setting method not available via CCXT.")

            logger.debug(f"Set SL raw response: {sl_set_response}")

            # Handle potential failure from fetch_with_retries
            if sl_set_response is None:
                # fetch_with_retries already logged the failure
                raise ccxt.ExchangeError("Set SL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(sl_set_response.get("info"), dict) and sl_set_response["info"].get("retCode") == 0:
                logger.trade(
                    Fore.GREEN
                    + Style.BRIGHT
                    + f"Stop Loss successfully set directly on the {position_side.upper()} position (Trigger: {sl_price_str_for_api})."
                )
                # --- Update Global State ---
                # CRITICAL: Clear any previous tracker state for this side (should be clear from check before entry, but defensive)
                # Use a placeholder to indicate SL is active on the position
                sl_marker_id = f"POS_SL_{position_side.upper()}"
                order_tracker[position_side] = {"sl_id": sl_marker_id, "tsl_id": None}
                logger.info(f"Updated order tracker: {order_tracker}")

                # Use actual average fill price in notification
                entry_msg = (
                    f"ENTERED {side.upper()} {filled_qty.normalize()} {symbol.split('/')[0]} @ {average_price:.4f}. "
                    f"Initial SL @ {sl_price_str_for_api}. TSL pending profit threshold."
                )
                logger.trade(Back.GREEN + Fore.BLACK + Style.BRIGHT + entry_msg)
                termux_notify(
                    "Trade Entry", f"{side.upper()} {symbol} @ {average_price:.4f}, SL: {sl_price_str_for_api}"
                )
                return True  # SUCCESS!

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                error_code = None
                if isinstance(sl_set_response.get("info"), dict):
                    error_msg = sl_set_response["info"].get("retMsg", error_msg)
                    error_code = sl_set_response["info"].get("retCode")
                    error_msg += f" (Code: {error_code})"
                raise ccxt.ExchangeError(f"Stop loss setting failed. Exchange message: {error_msg}")

        # --- Handle SL Setting Failures ---
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NotSupported) as e:
            # This is critical - position opened but SL setting failed. Emergency close needed.
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + f"CRITICAL: Failed to set stop-loss on position after entry: {e}. Position is UNPROTECTED."
            )
            logger.warning(Fore.YELLOW + "Attempting emergency market closure of unprotected position...")
            try:
                emergency_close_side = "sell" if position_side == "long" else "buy"
                # Use the *filled quantity* from the successful market order fill check
                # Format filled quantity precisely for closure order
                close_qty_str = format_amount(symbol, filled_qty.copy_abs(), ROUND_DOWN)
                close_qty_decimal = Decimal(close_qty_str)

                # Check against minimum quantity again before closing
                try:
                    min_qty_close = Decimal(str(MARKET_INFO["limits"]["amount"]["min"]))
                except (KeyError, InvalidOperation, TypeError):
                    logger.warning("Could not determine minimum order quantity for emergency closure validation.")
                    min_qty_close = Decimal("0")  # Assume zero if unavailable

                if close_qty_decimal < min_qty_close:
                    logger.critical(
                        f"{Fore.RED}Emergency closure quantity {close_qty_decimal.normalize()} is below minimum {min_qty_close}. MANUAL CLOSURE REQUIRED for {position_side.upper()} position!"
                    )
                    termux_notify(
                        "EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & < MIN QTY! Close manually!"
                    )
                    # Do NOT reset tracker state here, as we don't know the position status for sure.
                    return False  # Indicate failure of the entire entry process

                # Place the emergency closure order
                emergency_close_params = {"reduceOnly": True}  # Ensure it only closes
                # fetch_with_retries handles category param
                emergency_close_order = fetch_with_retries(
                    EXCHANGE.create_market_order,
                    symbol=symbol,
                    side=emergency_close_side,
                    amount=float(close_qty_decimal),  # CCXT needs float
                    params=emergency_close_params,
                )

                if emergency_close_order and (
                    emergency_close_order.get("id") or emergency_close_order.get("info", {}).get("retCode") == 0
                ):
                    close_id = emergency_close_order.get("id", "N/A (retCode 0)")
                    logger.trade(Fore.GREEN + f"Emergency closure order placed successfully: ID {close_id}")
                    termux_notify("Closure Attempted", f"{symbol} emergency closure sent.")
                    # Reset tracker state as position *should* be closing (best effort)
                    order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                else:
                    error_msg = (
                        emergency_close_order.get("info", {}).get("retMsg", "Unknown error")
                        if isinstance(emergency_close_order, dict)
                        else str(emergency_close_order)
                    )
                    logger.critical(
                        Fore.RED
                        + Style.BRIGHT
                        + f"EMERGENCY CLOSURE FAILED (Order placement failed): {error_msg}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!"
                    )
                    termux_notify(
                        "EMERGENCY!",
                        f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!",
                    )
                    # Do NOT reset tracker state here.

            except Exception as close_err:
                logger.critical(
                    Fore.RED
                    + Style.BRIGHT
                    + f"EMERGENCY CLOSURE FAILED (Exception during closure): {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!",
                    exc_info=True,
                )
                termux_notify(
                    "EMERGENCY!",
                    f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!",
                )
                # Do NOT reset tracker state here.

            return False  # Signal overall failure of the entry attempt due to SL failure

        except Exception as e:
            logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error setting SL: {e}", exc_info=True)
            logger.warning(
                Fore.YELLOW
                + Style.BRIGHT
                + "Position may be open without Stop Loss due to unexpected SL setting error. MANUAL INTERVENTION ADVISED."
            )
            # Consider emergency closure here too? Yes, safer. Re-use the emergency closure logic.
            try:
                position_side = "long" if side == "buy" else "short"
                emergency_close_side = "sell" if position_side == "long" else "buy"
                close_qty_str = format_amount(symbol, filled_qty.copy_abs(), ROUND_DOWN)
                close_qty_decimal = Decimal(close_qty_str)
                try:
                    min_qty_close = Decimal(str(MARKET_INFO["limits"]["amount"]["min"]))
                except (KeyError, InvalidOperation, TypeError):
                    logger.warning("Could not determine minimum order quantity for emergency closure validation.")
                    min_qty_close = Decimal("0")  # Assume zero if unavailable

                if close_qty_decimal < min_qty_close:
                    logger.critical(
                        f"{Fore.RED}Emergency closure quantity {close_qty_decimal.normalize()} is below minimum {min_qty_close}. MANUAL CLOSURE REQUIRED for {position_side.upper()} position!"
                    )
                    termux_notify(
                        "EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & < MIN QTY! Close manually!"
                    )
                    return False  # Indicate failure

                emergency_close_params = {"reduceOnly": True}
                emergency_close_order = fetch_with_retries(
                    EXCHANGE.create_market_order,
                    symbol=symbol,
                    side=emergency_close_side,
                    amount=float(close_qty_decimal),
                    params=emergency_close_params,
                )
                if emergency_close_order and (
                    emergency_close_order.get("id") or emergency_close_order.get("info", {}).get("retCode") == 0
                ):
                    close_id = emergency_close_order.get("id", "N/A (retCode 0)")
                    logger.trade(
                        Fore.GREEN
                        + f"Emergency closure order placed successfully after unexpected SL error: ID {close_id}"
                    )
                    termux_notify("Closure Attempted", f"{symbol} emergency closure sent after SL error.")
                    order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                else:
                    error_msg = (
                        emergency_close_order.get("info", {}).get("retMsg", "Unknown error")
                        if isinstance(emergency_close_order, dict)
                        else str(emergency_close_order)
                    )
                    logger.critical(
                        Fore.RED
                        + Style.BRIGHT
                        + f"EMERGENCY CLOSURE FAILED (Order placement failed) after unexpected SL error: {error_msg}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!"
                    )
                    termux_notify(
                        "EMERGENCY!",
                        f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!",
                    )
            except Exception as close_err:
                logger.critical(
                    Fore.RED
                    + Style.BRIGHT
                    + f"EMERGENCY CLOSURE FAILED (Exception during closure) after unexpected SL error: {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!",
                    exc_info=True,
                )
                termux_notify(
                    "EMERGENCY!",
                    f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!",
                )

            return False  # Signal overall failure

    # --- Handle Initial Market Order Failures ---
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied) as e:
        # Error placing the initial market order itself (handled by fetch_with_retries re-raising)
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Exchange error placing market order: {e}")
        # Log specific exchange message if available (CCXT often includes it in the exception string)
        # Example: ccxt.ExchangeError: bybit {"retCode":10001,"retMsg":"invalid order_qty","result":{},"retExtInfo":{},"time":1672816361179}
        # The message usually contains retMsg.
        # if isinstance(getattr(e, 'args', None), tuple) and len(e.args) > 0 and isinstance(e.args[0], str):
        #      logger.error(f"Exchange message excerpt: {e.args[0][:500]}")
        # The exception message itself is usually sufficient.
        return False
    except Exception as e:
        logger.error(
            Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during market order placement: {e}",
            exc_info=True,
        )
        return False


def manage_trailing_stop(
    symbol: str,
    position_side: str,  # 'long' or 'short'
    position_qty: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    atr: Decimal,
) -> None:
    """Manages the activation and setting of a trailing stop loss on the position, using Decimal."""
    global order_tracker, EXCHANGE, MARKET_INFO

    logger.debug(f"Checking TSL status for {position_side.upper()} position...")

    if EXCHANGE is None or MARKET_INFO is None:
        logger.error("Exchange or Market Info not available, cannot manage TSL.")
        return

    # --- Initial Checks ---
    if position_qty.copy_abs() < CONFIG.position_qty_epsilon or entry_price.is_nan() or entry_price <= Decimal("0"):
        # If position seems closed or invalid, ensure tracker is clear.
        if order_tracker[position_side]["sl_id"] or order_tracker[position_side]["tsl_id"]:
            logger.info(
                f"Position {position_side} appears closed or invalid (Qty: {position_qty.normalize()}, Entry: {entry_price}). Clearing stale order trackers."
            )
            order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return  # No position to manage TSL for

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.warning(Fore.YELLOW + "Cannot evaluate TSL activation: Invalid ATR value.")
        return

    # --- Get Current Tracker State ---
    initial_sl_marker = order_tracker[position_side]["sl_id"]  # Could be ID or placeholder "POS_SL_..."
    active_tsl_marker = order_tracker[position_side]["tsl_id"]  # Could be ID or placeholder "POS_TSL_..."

    # If TSL is already active (has a marker), assume exchange handles the trail.
    if active_tsl_marker:
        log_msg = (
            f"{position_side.upper()} TSL ({active_tsl_marker}) is already active. Exchange is managing the trail."
        )
        logger.debug(log_msg)
        # Sanity check: Ensure initial SL marker is None if TSL is active
        if initial_sl_marker:
            logger.warning(
                f"Inconsistent state: TSL active ({active_tsl_marker}) but initial SL marker ({initial_sl_marker}) is also present. Clearing initial SL marker."
            )
            order_tracker[position_side]["sl_id"] = None
        return  # TSL is already active, nothing more to do here

    # If TSL is not active, check if we *should* activate it.
    # Requires an initial SL marker to be present to indicate the position is at least protected by a fixed SL.
    if not initial_sl_marker:
        # This can happen if the initial SL setting failed, or if state got corrupted.
        logger.warning(
            f"Cannot activate TSL for {position_side.upper()}: Initial SL protection marker is missing from tracker. Position might be unprotected or already managed externally."
        )
        # Consider adding logic here to try and set a regular SL if missing? Or just warn.
        return  # Cannot activate TSL if initial SL state is unknown/missing

    # --- Check TSL Activation Condition ---
    profit = Decimal("NaN")
    try:
        if position_side == "long":
            profit = current_price - entry_price
        else:  # short
            profit = entry_price - current_price
    except (TypeError, InvalidOperation):  # Handle potential NaN in prices
        logger.warning("Cannot calculate profit for TSL check due to NaN price(s).")
        return

    # Activation threshold in price points
    activation_threshold_points = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(
        f"{position_side.upper()} Profit: {profit:.8f}, TSL Activation Threshold (Points): {activation_threshold_points:.8f} ({CONFIG.tsl_activation_atr_multiplier} * ATR)"
    )

    # Activate TSL only if profit exceeds the threshold (use Decimal comparison)
    if not profit.is_nan() and profit > activation_threshold_points:
        logger.trade(
            Fore.GREEN
            + Style.BRIGHT
            + f"Profit threshold reached for {position_side.upper()} position (Profit {profit:.4f} > Threshold {activation_threshold_points:.4f}). Activating TSL."
        )

        # --- Set Trailing Stop Loss on Position ---
        # Bybit V5 sets TSL directly on the position using specific parameters.
        # We use the same `set_trading_stop` endpoint as the initial SL, but provide TSL params.

        # TSL distance as percentage string (e.g., "0.5" for 0.5%)
        # Ensure correct formatting for the API (string representation with sufficient precision)
        # Quantize to a reasonable number of decimal places for percentage (e.g., 3-4)
        trail_percent_str = str(CONFIG.trailing_stop_percent.quantize(Decimal("0.001")))  # Format to 3 decimal places

        # Bybit V5 Parameters for setting TSL on position:
        # Endpoint: POST /v5/position/set-trading-stop
        set_tsl_params = {
            "category": CONFIG.market_type,  # Required
            "symbol": MARKET_INFO["id"],  # Use exchange-specific market ID
            "trailingStop": trail_percent_str,  # Trailing distance percentage (as string)
            "tpslMode": "Full",  # Apply to the whole position
            "slTriggerBy": CONFIG.tsl_trigger_by,  # Trigger type for the trail (LastPrice, MarkPrice, IndexPrice)
            # 'activePrice': format_price(symbol, current_price), # Optional: Price to activate the trail immediately. If omitted, Bybit activates when price moves favorably by trail %. Check docs.
            # Recommended: Don't set activePrice here. Let Bybit handle the initial activation based on the trail distance from the best price.
            # To remove the fixed SL when activating TSL, Bybit V5 documentation indicates setting 'stopLoss' to "" (empty string) or '0'.
            # Setting to "" is often safer to explicitly indicate removal.
            "stopLoss": "",  # Remove the fixed SL when activating TSL
            "side": "Buy" if position_side == "long" else "Sell",  # Add side parameter as required by V5 docs
            # 'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(
            f"Setting Position TSL: Trail={trail_percent_str}%, TriggerBy={CONFIG.tsl_trigger_by}, Side={set_tsl_params['side']}, Removing Fixed SL"
        )
        logger.debug(f"Set TSL Params (for setTradingStop): {set_tsl_params}")

        tsl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            if hasattr(EXCHANGE, "private_post_position_set_trading_stop"):
                tsl_set_response = fetch_with_retries(
                    EXCHANGE.private_post_position_set_trading_stop, params=set_tsl_params
                )
            else:
                logger.error(
                    Fore.RED
                    + "Cannot set TSL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit."
                )
                raise ccxt.NotSupported("TSL setting method not available.")

            logger.debug(f"Set TSL raw response: {tsl_set_response}")

            # Handle potential failure from fetch_with_retries
            if tsl_set_response is None:
                # fetch_with_retries already logged the failure
                raise ccxt.ExchangeError("Set TSL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(tsl_set_response.get("info"), dict) and tsl_set_response["info"].get("retCode") == 0:
                logger.trade(
                    Fore.GREEN
                    + Style.BRIGHT
                    + f"Trailing Stop Loss successfully activated for {position_side.upper()} position. Trail: {trail_percent_str}%"
                )
                # --- Update Global State ---
                # Set TSL active marker and clear the initial SL marker
                tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                order_tracker[position_side]["tsl_id"] = tsl_marker_id
                order_tracker[position_side]["sl_id"] = None  # Remove initial SL marker marker from tracker
                logger.info(f"Updated order tracker: {order_tracker}")
                termux_notify("TSL Activated", f"{position_side.upper()} {symbol} TSL active.")
                return  # Success

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                error_code = None
                if isinstance(tsl_set_response.get("info"), dict):
                    error_msg = tsl_set_response["info"].get("retMsg", error_msg)
                    error_code = tsl_set_response["info"].get("retCode")
                    error_msg += f" (Code: {error_code})"
                # Check if error was due to trying to remove non-existent SL (might be benign, e.g., SL already hit)
                # Example Bybit code: 110025 = SL/TP order not found or completed
                if error_code == 110025:
                    logger.warning(
                        "TSL activation may have succeeded, but received code 110025 (SL/TP not found/completed) when trying to clear fixed SL. Assuming TSL is active and fixed SL was already gone."
                    )
                    # Proceed as if successful, update tracker
                    tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                    order_tracker[position_side]["tsl_id"] = tsl_marker_id
                    order_tracker[position_side]["sl_id"] = None
                    logger.info(f"Updated order tracker (assuming TSL active despite code 110025): {order_tracker}")
                    termux_notify("TSL Activated*", f"{position_side.upper()} {symbol} TSL active (check exchange).")
                    return  # Treat as success for now
                else:
                    raise ccxt.ExchangeError(f"Failed to activate trailing stop loss. Exchange message: {error_msg}")

        # --- Handle TSL Setting Failures ---
        except (ccxt.ExchangeError, ccxt.InvalidOrder, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied) as e:
            # TSL setting failed. Initial SL marker *should* still be in the tracker if it was set initially.
            # Position might be protected by the initial SL, or might be unprotected if initial SL failed.
            logger.error(Fore.RED + Style.BRIGHT + f"Failed to activate TSL: {e}")
            logger.warning(
                Fore.YELLOW
                + "Position continues with initial SL (if successfully set) or may be UNPROTECTED if initial SL failed. MANUAL INTERVENTION ADVISED if initial SL state is uncertain."
            )
            # Do NOT clear the initial SL marker here. Do not set TSL marker.
            termux_notify("TSL Activation FAILED!", f"{symbol} TSL activation failed. Check logs/position.")
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error activating TSL: {e}", exc_info=True)
            logger.warning(
                Fore.YELLOW
                + Style.BRIGHT
                + "Position continues with initial SL (if successfully set) or may be UNPROTECTED. MANUAL INTERVENTION ADVISED if initial SL state is uncertain."
            )
            termux_notify(
                "TSL Activation FAILED!", f"{symbol} TSL activation failed (unexpected). Check logs/position."
            )

    else:
        # Profit threshold not met
        sl_status_log = f"({initial_sl_marker})" if initial_sl_marker else "(None!)"
        logger.debug(
            f"{position_side.upper()} profit ({profit:.4f if not profit.is_nan() else 'N/A'}) has not crossed TSL activation threshold ({activation_threshold_points:.4f}). Keeping initial SL {sl_status_log}."
        )


def print_status_panel(
    cycle: int,
    timestamp: Optional[pd.Timestamp],
    price: Optional[Decimal],
    indicators: Optional[Dict[str, Decimal]],
    positions: Optional[Dict[str, Dict[str, Any]]],
    equity: Optional[Decimal],
    signals: Dict[str, Union[bool, str]],
    order_tracker_state: Dict[str, Dict[str, Optional[str]]],  # Pass tracker state snapshot explicitly
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""

    header_color = Fore.MAGENTA + Style.BRIGHT
    section_color = Fore.CYAN
    value_color = Fore.WHITE
    reset_all = Style.RESET_ALL

    print(header_color + "\n" + "=" * 80)
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else f"{Fore.YELLOW}N/A"
    print(f" Cycle: {value_color}{cycle}{header_color} | Timestamp: {value_color}{ts_str}")
    equity_str = (
        f"{equity:.4f} {MARKET_INFO.get('settle', 'Quote')}"
        if equity is not None and not equity.is_nan()
        else f"{Fore.YELLOW}N/A"
    )
    print(f" Equity: {Fore.GREEN}{equity_str}" + reset_all)
    print(header_color + "-" * 80)

    # --- Market & Indicators ---
    # Use .get(..., Decimal('NaN')) for safe access to indicator values
    price_str = f"{price:.4f}" if price is not None and not price.is_nan() else f"{Fore.YELLOW}N/A"
    atr = indicators.get("atr", Decimal("NaN")) if indicators else Decimal("NaN")
    atr_str = f"{atr:.6f}" if not atr.is_nan() else f"{Fore.YELLOW}N/A"
    trend_ema = indicators.get("trend_ema", Decimal("NaN")) if indicators else Decimal("NaN")
    trend_ema_str = f"{trend_ema:.4f}" if not trend_ema.is_nan() else f"{Fore.YELLOW}N/A"

    price_color = Fore.WHITE
    trend_desc = f"{Fore.YELLOW}Trend N/A"
    if price is not None and not price.is_nan() and not trend_ema.is_nan():
        # Use configured buffer for display consistency (approximate)
        trend_buffer_display = trend_ema.copy_abs() * (CONFIG.trend_filter_buffer_percent / Decimal("100"))
        if price > trend_ema + trend_buffer_display:
            price_color = Fore.GREEN
            trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema - trend_buffer_display:
            price_color = Fore.RED
            trend_desc = f"{price_color}(Below Trend)"
        else:
            price_color = Fore.YELLOW
            trend_desc = f"{price_color}(At Trend)"

    stoch_k = indicators.get("stoch_k", Decimal("NaN")) if indicators else Decimal("NaN")
    stoch_d = indicators.get("stoch_d", Decimal("NaN")) if indicators else Decimal("NaN")
    stoch_k_str = f"{stoch_k:.2f}" if not stoch_k.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_d_str = f"{stoch_d:.2f}" if not stoch_d.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_color = Fore.YELLOW
    stoch_desc = f"{Fore.YELLOW}Stoch N/A"
    if not stoch_k.is_nan():
        if stoch_k < CONFIG.stoch_oversold_threshold:
            stoch_color = Fore.GREEN
            stoch_desc = f"{stoch_color}Oversold (<{CONFIG.stoch_oversold_threshold})"
        elif stoch_k > CONFIG.stoch_overbought_threshold:
            stoch_color = Fore.RED
            stoch_desc = f"{stoch_color}Overbought (>{CONFIG.stoch_overbought_threshold})"
        else:
            stoch_color = Fore.YELLOW
            stoch_desc = f"{stoch_color}Neutral ({CONFIG.stoch_oversold_threshold}-{CONFIG.stoch_overbought_threshold})"

    fast_ema = indicators.get("fast_ema", Decimal("NaN")) if indicators else Decimal("NaN")
    slow_ema = indicators.get("slow_ema", Decimal("NaN")) if indicators else Decimal("NaN")
    fast_ema_str = f"{fast_ema:.4f}" if not fast_ema.is_nan() else f"{Fore.YELLOW}N/A"
    slow_ema_str = f"{slow_ema:.4f}" if not slow_ema.is_nan() else f"{Fore.YELLOW}N/A"
    ema_cross_color = Fore.WHITE
    ema_desc = f"{Fore.YELLOW}EMA N/A"
    if not fast_ema.is_nan() and not slow_ema.is_nan():
        if fast_ema > slow_ema:
            ema_cross_color = Fore.GREEN
            ema_desc = f"{ema_cross_color}Bullish"  # Changed from Cross to just Bullish as it's current state, not a historical event
        elif fast_ema < slow_ema:
            ema_cross_color = Fore.RED
            ema_desc = f"{ema_cross_color}Bearish"  # Changed
        else:
            ema_cross_color = Fore.YELLOW
            ema_desc = f"{Fore.YELLOW}Aligned"

    status_data = [
        [section_color + "Market", value_color + CONFIG.symbol, f"{price_color}{price_str}"],
        [section_color + f"Trend EMA ({CONFIG.trend_ema_period})", f"{value_color}{trend_ema_str}", trend_desc],
        [section_color + f"ATR ({CONFIG.atr_period})", f"{value_color}{atr_str}", ""],  # Display ATR period from CONFIG
        [
            section_color + f"EMA Fast/Slow ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})",
            f"{ema_cross_color}{fast_ema_str} / {slow_ema_str}",
            ema_desc,
        ],  # Display EMA periods from CONFIG
        [
            section_color + f"Stoch %K/%D ({CONFIG.stoch_period},{CONFIG.stoch_smooth_k},{CONFIG.stoch_smooth_d})",
            f"{stoch_color}{stoch_k_str} / {stoch_d_str}",
            stoch_desc,
        ],  # Display Stoch periods from CONFIG
    ]
    print(tabulate(status_data, tablefmt="fancy_grid", colalign=("left", "left", "left")))
    # print(header_color + "-" * 80) # Separator removed, using table grid

    # --- Positions & Orders ---
    pos_avail = positions is not None
    long_pos = positions.get("long", {}) if pos_avail else {}
    short_pos = positions.get("short", {}) if pos_avail else {}

    # Safely get values, handling None or NaN Decimals
    long_qty = long_pos.get("qty", Decimal("0.0"))
    short_qty = short_pos.get("qty", Decimal("0.0"))
    long_entry = long_pos.get("entry_price", Decimal("NaN"))
    short_entry = short_pos.get("entry_price", Decimal("NaN"))
    long_pnl = long_pos.get("pnl", Decimal("NaN"))
    short_pnl = short_pos.get("pnl", Decimal("NaN"))
    long_liq = long_pos.get("liq_price", Decimal("NaN"))
    short_liq = short_pos.get("liq_price", Decimal("NaN"))

    # Use the passed tracker state snapshot
    long_sl_marker = order_tracker_state["long"]["sl_id"]
    long_tsl_marker = order_tracker_state["long"]["tsl_id"]
    short_sl_marker = order_tracker_state["short"]["sl_id"]
    short_tsl_marker = order_tracker_state["short"]["tsl_id"]

    # Determine SL/TSL status strings
    def get_stop_status(sl_marker, tsl_marker):
        if tsl_marker:
            if tsl_marker.startswith("POS_TSL_"):
                return f"{Fore.GREEN}TSL Active (Pos)"
            else:
                return f"{Fore.GREEN}TSL Active (ID: ...{tsl_marker[-6:]})"  # Should not happen with V5 pos-based TSL
        elif sl_marker:
            if sl_marker.startswith("POS_SL_"):
                return f"{Fore.YELLOW}SL Active (Pos)"
            else:
                return f"{Fore.YELLOW}SL Active (ID: ...{sl_marker[-6:]})"  # Should not happen with V5 pos-based SL
        else:
            # No marker found in tracker
            return f"{Fore.RED}{Style.BRIGHT}NONE (!)"  # Highlight if no stop is tracked

    # Display stop status only if position exists (using epsilon check)
    long_stop_status = (
        get_stop_status(long_sl_marker, long_tsl_marker)
        if long_qty.copy_abs() >= CONFIG.position_qty_epsilon
        else f"{value_color}-"
    )
    short_stop_status = (
        get_stop_status(short_sl_marker, short_tsl_marker)
        if short_qty.copy_abs() >= CONFIG.position_qty_epsilon
        else f"{value_color}-"
    )

    # Format position details, handle potential None or NaN from failed fetch/parsing
    if not pos_avail:
        long_qty_str, short_qty_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_entry_str, short_entry_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_pnl_str, short_pnl_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_liq_str, short_liq_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
    else:
        # Format Decimals nicely, remove trailing zeros for quantity (more readable)
        long_qty_str = (
            f"{long_qty.normalize()}" if long_qty.copy_abs() >= CONFIG.position_qty_epsilon else "0"
        )  # Use normalize to remove trailing zeros
        short_qty_str = (
            f"{short_qty.normalize()}" if short_qty.copy_abs() >= CONFIG.position_qty_epsilon else "0"
        )  # Use normalize

        long_entry_str = f"{long_entry:.4f}" if not long_entry.is_nan() else "-"
        short_entry_str = f"{short_entry:.4f}" if not short_entry.is_nan() else "-"

        # PnL color based on value, only display if position exists
        long_pnl_color = Fore.GREEN if not long_pnl.is_nan() and long_pnl >= 0 else Fore.RED
        short_pnl_color = Fore.GREEN if not short_pnl.is_nan() and short_pnl >= 0 else Fore.RED
        long_pnl_str = (
            f"{long_pnl_color}{long_pnl:+.4f}{value_color}"
            if long_qty.copy_abs() >= CONFIG.position_qty_epsilon and not long_pnl.is_nan()
            else "-"
        )
        short_pnl_str = (
            f"{short_pnl_color}{short_pnl:+.4f}{value_color}"
            if short_qty.copy_abs() >= CONFIG.position_qty_epsilon and not short_pnl.is_nan()
            else "-"
        )

        # Liq price color (usually red), only display if position exists
        long_liq_str = (
            f"{Fore.RED}{long_liq:.4f}{value_color}"
            if long_qty.copy_abs() >= CONFIG.position_qty_epsilon and not long_liq.is_nan() and long_liq > 0
            else "-"
        )
        short_liq_str = (
            f"{Fore.RED}{short_liq:.4f}{value_color}"
            if short_qty.copy_abs() >= CONFIG.position_qty_epsilon and not short_liq.is_nan() and short_liq > 0
            else "-"
        )

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
    long_signal_status = signals.get("long", False)
    short_signal_status = signals.get("short", False)
    long_signal_color = Fore.GREEN + Style.BRIGHT if long_signal_status else Fore.WHITE
    short_signal_color = Fore.RED + Style.BRIGHT if short_signal_status else Fore.WHITE
    trend_status = f"(Trend Filter: {value_color}{'ON' if CONFIG.trade_only_with_trend else 'OFF'}{header_color})"
    signal_reason_text = signals.get("reason", "N/A")
    print(
        f" Signals {trend_status}: Long [{long_signal_color}{str(long_signal_status).upper():<5}{header_color}] | Short [{short_signal_color}{str(short_signal_status).upper():<5}{header_color}]"
    )  # Use .upper() for bool string
    # Display the signal reason below
    print(f" Reason: {Fore.YELLOW}{signal_reason_text}{Style.RESET_ALL}")
    print(header_color + "=" * 80 + reset_all)


def generate_signals(
    df_last_candles: pd.DataFrame, indicators: Optional[Dict[str, Any]], equity: Optional[Decimal]
) -> Dict[str, Union[bool, str]]:
    """Generates trading signals based on indicator conditions, using Decimal."""
    long_signal = False
    short_signal = False
    signal_reason = "No signal - Initial State"

    if not indicators:
        logger.warning("Cannot generate signals: indicators dictionary is missing.")
        return {"long": False, "short": False, "reason": "Indicators missing"}
    if df_last_candles is None or len(df_last_candles) < 1:
        logger.warning("Cannot generate signals: insufficient candle data.")
        return {"long": False, "short": False, "reason": "Insufficient candle data"}

    try:
        # Get latest candle data
        latest = df_last_candles.iloc[-1]
        current_price = Decimal(str(latest["close"]))
        # Get previous candle data for ATR move check
        previous_close = (
            Decimal(str(df_last_candles.iloc[-2]["close"])) if len(df_last_candles) >= 2 else Decimal("NaN")
        )

        if current_price.is_nan() or current_price <= Decimal(0):
            logger.warning("Cannot generate signals: current price is missing or invalid.")
            return {"long": False, "short": False, "reason": "Invalid price"}

        # Use .get with default Decimal('NaN') or False to handle missing/failed indicators gracefully
        k = indicators.get("stoch_k", Decimal("NaN"))
        # d = indicators.get('stoch_d', Decimal('NaN')) # Available but not used in current logic
        fast_ema = indicators.get("fast_ema", Decimal("NaN"))
        slow_ema = indicators.get("slow_ema", Decimal("NaN"))
        trend_ema = indicators.get("trend_ema", Decimal("NaN"))
        atr = indicators.get("atr", Decimal("NaN"))
        stoch_kd_bullish = indicators.get("stoch_kd_bullish", False)
        stoch_kd_bearish = indicators.get("stoch_kd_bearish", False)

        # Check if any required indicator is NaN
        required_indicators_vals = {
            "stoch_k": k,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "trend_ema": trend_ema,
            "atr": atr,
        }
        nan_indicators = [
            name for name, val in required_indicators_vals.items() if isinstance(val, Decimal) and val.is_nan()
        ]
        if nan_indicators:
            logger.warning(f"Cannot generate signals: Required indicator(s) are NaN: {', '.join(nan_indicators)}")
            return {"long": False, "short": False, "reason": f"NaN indicator(s): {', '.join(nan_indicators)}"}

        # Define conditions using Decimal comparisons for precision and CONFIG thresholds
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema

        # Loosened Trend Filter (price within +/- CONFIG.trend_filter_buffer_percent% of Trend EMA)
        trend_buffer_points = trend_ema.copy_abs() * (CONFIG.trend_filter_buffer_percent / Decimal("100"))
        price_above_trend_loosened = current_price > trend_ema - trend_buffer_points
        price_below_trend_loosened = current_price < trend_ema + trend_buffer_points
        price_strictly_above_trend = current_price > trend_ema  # For logging only
        price_strictly_below_trend = current_price < trend_ema  # For logging only

        # Stochastic K level or K/D cross conditions (Using CONFIG thresholds)
        stoch_oversold = k < CONFIG.stoch_oversold_threshold
        stoch_overbought = k > CONFIG.stoch_overbought_threshold

        # Combined Stochastic Condition: Oversold OR bullish K/D cross (in/from oversold zone - handled in calculate_indicators)
        stoch_long_condition = stoch_oversold or stoch_kd_bullish
        # Combined Stochastic Condition: Overbought OR bearish K/D cross (in/from overbought zone - handled in calculate_indicators)
        stoch_short_condition = stoch_overbought or stoch_kd_bearish

        # ATR Filter: Check if the price move from the previous close is significant
        is_significant_move = False
        if not previous_close.is_nan() and atr > Decimal("0"):
            price_move_points = (current_price - previous_close).copy_abs()
            atr_move_threshold_points = atr * CONFIG.atr_move_filter_multiplier
            is_significant_move = price_move_points > atr_move_threshold_points
            logger.debug(
                f"ATR Move Filter: Price Move ({price_move_points:.6f}) vs Threshold ({atr_move_threshold_points:.6f}). Significant: {is_significant_move}"
            )
        elif previous_close.is_nan():
            logger.debug("ATR Move Filter skipped: Only one candle available.")
            # Decide how to handle this - allow signal if only one candle? Or require filter?
            # Defaulting to False if previous close is NaN is safer - requires enough history.
            is_significant_move = False
        else:  # atr is 0 or NaN
            logger.debug(f"ATR Move Filter skipped: ATR is invalid ({atr:.6f}).")
            is_significant_move = False  # Cannot apply filter if ATR is bad

        # --- Signal Logic ---
        # Combine all conditions
        potential_long = ema_bullish_cross and stoch_long_condition and is_significant_move
        potential_short = ema_bearish_cross and stoch_short_condition and is_significant_move

        # Apply trend filter if enabled
        if potential_long:
            if CONFIG.trade_only_with_trend:
                if price_above_trend_loosened:
                    long_signal = True
                    # Detailed reason string
                    reason_parts = [
                        f"Bullish EMA Cross ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})",
                        f"Stoch Long ({k:.2f}) [Oversold (<{CONFIG.stoch_oversold_threshold}) or Bullish K/D Cross]",
                        f"Price ({current_price:.4f}) near/above Trend EMA ({trend_ema:.4f}) [Buffer {CONFIG.trend_filter_buffer_percent}%]",
                        f"Price move > {CONFIG.atr_move_filter_multiplier}x ATR (ATR {atr:.6f})",
                    ]
                    signal_reason = "Long: " + ", ".join(reason_parts)
                else:
                    # Log detailed rejection reason if trend filter is ON
                    reason_parts = [
                        f"Bullish EMA Cross ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})",
                        f"Stoch Long ({k:.2f}) [Oversold (<{CONFIG.stoch_oversold_threshold}) or Bullish K/D Cross]",
                        f"Price move > {CONFIG.atr_move_filter_multiplier}x ATR (ATR {atr:.6f})",
                    ]
                    trend_reason = f"Price ({current_price:.4f}) not near/above Trend EMA ({trend_ema:.4f}) [Buffer {CONFIG.trend_filter_buffer_percent}%]"
                    signal_reason = (
                        f"Long Blocked: {trend_reason} (Trend Filter ON) | Conditions met (excluding trend): "
                        + ", ".join(reason_parts)
                    )

            else:  # Trend filter off
                long_signal = True
                reason_parts = [
                    f"Bullish EMA Cross ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})",
                    f"Stoch Long ({k:.2f}) [Oversold (<{CONFIG.stoch_oversold_threshold}) or Bullish K/D Cross]",
                    f"Price move > {CONFIG.atr_move_filter_multiplier}x ATR (ATR {atr:.6f})",
                ]
                signal_reason = "Long (Trend Filter OFF): " + ", ".join(reason_parts)

        elif potential_short:
            if CONFIG.trade_only_with_trend:
                if price_below_trend_loosened:
                    short_signal = True
                    # Detailed reason string
                    reason_parts = [
                        f"Bearish EMA Cross ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})",
                        f"Stoch Short ({k:.2f}) [Overbought (>{CONFIG.stoch_overbought_threshold}) or Bearish K/D Cross]",
                        f"Price ({current_price:.4f}) near/below Trend EMA ({trend_ema:.4f}) [Buffer {CONFIG.trend_filter_buffer_percent}%]",
                        f"Price move > {CONFIG.atr_move_filter_multiplier}x ATR (ATR {atr:.6f})",
                    ]
                    signal_reason = "Short: " + ", ".join(reason_parts)
                else:
                    # Log detailed rejection reason if trend filter is ON
                    reason_parts = [
                        f"Bearish EMA Cross ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})",
                        f"Stoch Short ({k:.2f}) [Overbought (>{CONFIG.stoch_overbought_threshold}) or Bearish K/D Cross]",
                        f"Price move > {CONFIG.atr_move_filter_multiplier}x ATR (ATR {atr:.6f})",
                    ]
                    trend_reason = f"Price ({current_price:.4f}) not near/below Trend EMA ({trend_ema:.4f}) [Buffer {CONFIG.trend_filter_buffer_percent}%]"
                    signal_reason = (
                        f"Short Blocked: {trend_reason} (Trend Filter ON) | Conditions met (excluding trend): "
                        + ", ".join(reason_parts)
                    )
            else:  # Trend filter off
                short_signal = True
                reason_parts = [
                    f"Bearish EMA Cross ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})",
                    f"Stoch Short ({k:.2f}) [Overbought (>{CONFIG.stoch_overbought_threshold}) or Bearish K/D Cross]",
                    f"Price move > {CONFIG.atr_move_filter_multiplier}x ATR (ATR {atr:.6f})",
                ]
                signal_reason = "Short (Trend Filter OFF): " + ", ".join(reason_parts)

        else:
            # No signal - build detailed reason why
            reason_parts = []
            # EMA Check
            if ema_bullish_cross:
                reason_parts.append(f"EMA Bullish ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})")
            elif ema_bearish_cross:
                reason_parts.append(f"EMA Bearish ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})")
            else:
                reason_parts.append(f"No EMA Cross ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})")

            # Trend Check
            if CONFIG.trade_only_with_trend:
                if price_strictly_above_trend:
                    reason_parts.append(f"Price ({current_price:.4f}) Above Trend EMA ({trend_ema:.4f})")
                elif price_strictly_below_trend:
                    reason_parts.append(f"Price ({current_price:.4f}) Below Trend EMA ({trend_ema:.4f})")
                else:
                    reason_parts.append(f"Price ({current_price:.4f}) At Trend EMA ({trend_ema:.4f})")
            # else: # Trend filter off - don't add price vs trend info to reason if filter is off

            # Stochastic Check
            if stoch_oversold:
                reason_parts.append(f"Stoch Oversold ({k:.2f} < {CONFIG.stoch_oversold_threshold})")
            elif stoch_overbought:
                reason_parts.append(f"Stoch Overbought ({k:.2f} > {CONFIG.stoch_overbought_threshold})")
            elif stoch_long_condition:
                reason_parts.append(f"Stoch Bullish K/D Cross ({k:.2f})")
            elif stoch_short_condition:
                reason_parts.append(f"Stoch Bearish K/D Cross ({k:.2f})")
            else:
                reason_parts.append(
                    f"Stoch Neutral ({CONFIG.stoch_oversold_threshold}-{CONFIG.stoch_overbought_threshold})"
                )

            # ATR Move Check
            if not is_significant_move:
                if not previous_close.is_nan() and atr > Decimal("0"):
                    price_move_points = (current_price - previous_close).copy_abs()
                    atr_move_threshold_points = atr * CONFIG.atr_move_filter_multiplier
                    reason_parts.append(
                        f"Price move ({price_move_points:.6f}) < {CONFIG.atr_move_filter_multiplier}x ATR ({atr_move_threshold_points:.6f})"
                    )
                else:
                    reason_parts.append("ATR Move Filter Skipped/Failed")

            signal_reason = "No signal (" + ", ".join(reason_parts) + ")"

        # Log the outcome
        if long_signal or short_signal:
            logger.info(Fore.GREEN + Style.BRIGHT + f"Signal Generated: {signal_reason}")
        else:
            # Log reason for no signal at debug level unless blocked by trend filter
            if "Blocked" in signal_reason:
                logger.info(Fore.YELLOW + f"Signal Check: {signal_reason}")
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
    cycle_success = True  # Track if cycle completes without critical errors

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(Fore.RED + "Halting cycle: Market data fetch failed or returned empty.")
        cycle_success = False
        # No status panel if no data to derive price/timestamp from
        end_time = time.time()
        logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} Aborted (Duration: {end_time - start_time:.2f}s) ---")
        return  # Skip cycle

    # 2. Get Current Price & Timestamp from Data (and previous close for ATR filter)
    current_price: Optional[Decimal] = None
    last_timestamp: Optional[pd.Timestamp] = None
    try:
        # Use close price of the last *completed* candle for indicator-based logic
        # Need at least 2 candles if ATR move filter is used, check length
        if len(df) < 2 and CONFIG.atr_move_filter_multiplier > Decimal("0"):
            logger.warning(
                f"{Fore.YELLOW}Insufficient data ({len(df)} candles) for ATR move filter. Need at least 2. Filter will be skipped."
            )
            # Proceed, but signal generation will skip the filter

        last_candle = df.iloc[-1]
        current_price_float = last_candle["close"]
        if pd.isna(current_price_float):
            raise ValueError("Latest close price is NaN")
        current_price = Decimal(str(current_price_float))
        last_timestamp = df.index[-1]  # Already UTC from fetch_market_data
        logger.debug(
            f"Latest candle: Time={last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price:.8f}"
        )  # Log with high precision

        # Check for stale data (compare last candle time to current time)
        now_utc = pd.Timestamp.utcnow()  # UTC timestamp
        time_diff = now_utc - last_timestamp
        # Allow for interval duration + some buffer (e.g., 1.5 * interval + 60s)
        try:
            interval_seconds = EXCHANGE.parse_timeframe(CONFIG.interval)
            allowed_lag = pd.Timedelta(seconds=interval_seconds * 1.5 + 60)
            if time_diff > allowed_lag:
                logger.warning(
                    Fore.YELLOW
                    + f"Market data may be stale. Last candle: {last_timestamp.strftime('%H:%M:%S')} ({time_diff} ago). Allowed lag: ~{allowed_lag}"
                )
        except ValueError:
            logger.warning("Could not parse interval to check data staleness.")

    except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
        logger.error(
            Fore.RED + f"Halting cycle: Failed to get/process current price/timestamp from DataFrame: {e}",
            exc_info=True,
        )
        cycle_success = False
        # No status panel if price invalid
        end_time = time.time()
        logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} Aborted (Duration: {end_time - start_time:.2f}s) ---")
        return  # Skip cycle

    # 3. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(Fore.RED + "Indicator calculation failed. Continuing cycle but skipping trade actions.")
        cycle_success = False  # Mark as failed for logging, but continue to fetch state and show panel

    current_atr = (
        indicators.get("atr", Decimal("NaN")) if indicators else Decimal("NaN")
    )  # Use NaN default if indicators is None

    # 4. Get Current State (Balance & Positions as Decimals)
    # Fetch balance first
    free_balance, current_equity = get_balance(MARKET_INFO.get("settle", "USDT"))
    if current_equity is None or current_equity.is_nan():
        logger.error(
            Fore.RED
            + "Failed to fetch valid current balance/equity. Cannot perform risk calculation or trading actions."
        )
        # Don't proceed with trade actions without knowing equity
        cycle_success = False
        # Fall through to display panel (will show N/A equity)

    # Fetch positions (crucial state)
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Failed to fetch current positions. Cannot manage state or trade.")
        cycle_success = False
        # Fall through to display panel (will show N/A positions)

    # --- Capture State Snapshot for Status Panel & Logic ---
    # Do this *before* potentially modifying state (like TSL management or entry)
    # Use deepcopy for the tracker to ensure the panel shows state before any potential updates in this cycle
    order_tracker_snapshot = copy.deepcopy(order_tracker)
    # Use the fetched positions directly as the snapshot (if fetch succeeded)
    positions_snapshot = (
        positions
        if positions is not None
        else {
            "long": {
                "qty": Decimal("0.0"),
                "entry_price": Decimal("NaN"),
                "pnl": Decimal("NaN"),
                "liq_price": Decimal("NaN"),
            },
            "short": {
                "qty": Decimal("0.0"),
                "entry_price": Decimal("NaN"),
                "pnl": Decimal("NaN"),
                "liq_price": Decimal("NaN"),
            },
        }
    )

    # --- Logic continues only if critical data is available (positions and equity) ---
    # Note: We can still show the panel even if positions/equity fetch failed,
    # but we *cannot* perform trade actions or TSL management safely.
    can_trade_logic = (
        positions is not None
        and current_equity is not None
        and not current_equity.is_nan()
        and current_equity > Decimal("0")
    )

    # Initialize signals dictionary
    signals: Dict[str, Union[bool, str]] = {
        "long": False,
        "short": False,
        "reason": "Skipped due to critical data failure",
    }

    if can_trade_logic:
        # Use the *current* state from `positions` dict (not snapshot) for logic decisions
        active_long_pos = positions.get("long", {})
        active_short_pos = positions.get("short", {})
        active_long_qty = active_long_pos.get("qty", Decimal("0.0"))
        active_short_qty = active_short_pos.get("qty", Decimal("0.0"))

        # Check if already have a significant position in either direction
        has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
        has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
        is_flat = not has_long_pos and not has_short_pos

        # 5. Manage Trailing Stops
        # Only attempt TSL management if indicators and current price are available
        if indicators is not None and not current_price.is_nan() and not current_atr.is_nan():
            if has_long_pos:
                logger.debug("Managing TSL for existing LONG position...")
                manage_trailing_stop(
                    CONFIG.symbol,
                    "long",
                    active_long_qty,
                    active_long_pos.get("entry_price", Decimal("NaN")),
                    current_price,
                    current_atr,
                )
            elif has_short_pos:
                logger.debug("Managing TSL for existing SHORT position...")
                manage_trailing_stop(
                    CONFIG.symbol,
                    "short",
                    active_short_qty,
                    active_short_pos.get("entry_price", Decimal("NaN")),
                    current_price,
                    current_atr,
                )
            else:
                # If flat, ensure trackers are clear (belt-and-suspenders check)
                if (
                    order_tracker["long"]["sl_id"]
                    or order_tracker["long"]["tsl_id"]
                    or order_tracker["short"]["sl_id"]
                    or order_tracker["short"]["tsl_id"]
                ):
                    logger.info("Position is flat, ensuring order trackers are cleared.")
                    order_tracker["long"] = {"sl_id": None, "tsl_id": None}
                    order_tracker["short"] = {"sl_id": None, "tsl_id": None}
                    # Update the snapshot to reflect the clearing for the panel display
                    order_tracker_snapshot["long"] = {"sl_id": None, "tsl_id": None}
                    order_tracker_snapshot["short"] = {"sl_id": None, "tsl_id": None}
        else:
            logger.warning("Skipping TSL management due to missing indicators, invalid price, or invalid ATR.")

        # 6. Generate Trading Signals
        # Signals only generated if indicators and current price are available, AND we have enough data for the ATR filter
        if (
            indicators is not None and not current_price.is_nan() and len(df) >= 2
        ):  # Require at least 2 candles for ATR filter
            signals_data = generate_signals(
                df.iloc[-2:], indicators, current_equity
            )  # Pass last 2 candles, indicators, equity
            signals = {
                "long": signals_data["long"],
                "short": signals_data["short"],
                "reason": signals_data["reason"],
            }  # Keep reason
        elif (
            indicators is not None and not current_price.is_nan() and CONFIG.atr_move_filter_multiplier == Decimal("0")
        ):
            # Allow signal generation if ATR filter is disabled, even with only 1 candle
            logger.warning("Generating signals with only 1 candle, but ATR move filter is OFF.")
            signals_data = generate_signals(
                df.iloc[-1:], indicators, current_equity
            )  # Pass last 1 candle, indicators, equity
            signals = {
                "long": signals_data["long"],
                "short": signals_data["short"],
                "reason": signals_data["reason"],
            }  # Keep reason
        else:
            logger.warning(
                "Skipping signal generation due to missing indicators, invalid price, or insufficient candle data for ATR filter."
            )
            signals = {"long": False, "short": False, "reason": "Skipped due to missing data/insufficient candles"}

        # 7. Execute Trades based on Signals
        # Only attempt entry if currently flat, indicators/ATR are available, and equity is sufficient
        # Also require signal generation to have been successful (ATR filter requires >= 2 candles unless disabled)
        if (
            is_flat
            and signals.get("reason") != "Skipped due to missing data/insufficient candles"
            and indicators is not None
            and not current_atr.is_nan()
        ):  # Equity checked in can_trade_logic
            if signals.get("long"):
                logger.info(
                    Fore.GREEN
                    + Style.BRIGHT
                    + f"Long signal detected! {signals.get('reason', '')}. Attempting entry..."
                )
                # place_risked_market_order handles its own error logging and tracker updates
                if place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr):
                    logger.info(Fore.GREEN + f"Long entry process completed successfully for cycle {cycle_count}.")
                else:
                    logger.error(Fore.RED + f"Long entry process failed for cycle {cycle_count}.")
                    # Optional: Implement cooldown logic here if needed

            elif signals.get("short"):
                logger.info(
                    Fore.RED + Style.BRIGHT + f"Short signal detected! {signals.get('reason', '')}. Attempting entry."
                )
                # place_risked_market_order handles its own error logging and tracker updates
                if place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr):
                    logger.info(Fore.GREEN + f"Short entry process completed successfully for cycle {cycle_count}.")
                else:
                    logger.error(Fore.RED + f"Short entry process failed for cycle {cycle_count}.")
                    # Optional: Implement cooldown logic here if needed

            # If a trade was attempted, main loop sleep handles the pause.

        elif not is_flat:
            pos_side = "LONG" if has_long_pos else "SHORT"
            logger.info(f"Position ({pos_side}) already open, skipping new entry signals.")
            # Future: Add exit logic based on counter-signals or other conditions if desired.
            # Example: if pos_side == "LONG" and signals.get("short"): close_position("long")
            # Example: if pos_side == "SHORT" and signals.get("long"): close_position("short")
        else:
            # This block is hit if can_trade_logic is False, meaning positions/equity fetch failed
            # or if signal generation was skipped due to insufficient data.
            logger.warning(
                "Skipping trade entry logic due to earlier critical data failure (positions/equity) or signal generation skipped."
            )
            # signals will retain its reason from the generation step

    # 8. Display Status Panel (Always display if data allows)
    # Use the state captured *before* TSL management and potential trade execution for consistency
    # unless the cycle failed very early (handled by the initial df check).
    print_status_panel(
        cycle_count,
        last_timestamp,
        current_price,
        indicators,
        positions_snapshot,
        current_equity,
        signals,
        order_tracker_snapshot,  # Use the snapshots
    )

    end_time = time.time()
    status_log = "Complete" if cycle_success else "Completed with WARNINGS/ERRORS"
    logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} {status_log} (Duration: {end_time - start_time:.2f}s) ---")


def graceful_shutdown() -> None:
    """Dispels active orders and closes open positions gracefully with precision."""
    logger.warning(Fore.YELLOW + Style.BRIGHT + "\nInitiating Graceful Shutdown Sequence...")
    termux_notify("Shutdown", f"Closing orders/positions for {CONFIG.symbol}.")

    global EXCHANGE, MARKET_INFO, order_tracker
    if EXCHANGE is None or MARKET_INFO is None:
        logger.error(Fore.RED + "Exchange object or Market Info not available. Cannot perform clean shutdown.")
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Cannot perform clean shutdown - Exchange not ready.")
        return

    symbol = CONFIG.symbol
    MARKET_INFO.get("id")  # Exchange specific ID
    MARKET_INFO.get("settle", "USDT")  # Use settle currency

    # 1. Cancel All Open Orders for the Symbol
    # This includes stop loss / take profit orders if they are separate entities (unlikely for Bybit V5 position stops)
    # and potentially limit orders if they were used for entry/exit (not in this strategy, but good practice).
    try:
        logger.info(Fore.CYAN + f"Dispelling all cancellable open orders for {symbol}...")
        # fetch_with_retries handles category param
        # Fetch open orders first to log IDs (best effort)
        open_orders_list = []
        try:
            # Bybit V5 fetch_open_orders requires category
            fetch_params = {"category": CONFIG.market_type}
            open_orders_list = fetch_with_retries(EXCHANGE.fetch_open_orders, symbol, params=fetch_params)
            if open_orders_list:
                order_ids = [o.get("id", "N/A") for o in open_orders_list]
                logger.info(
                    f"Found {len(open_orders_list)} open orders to attempt cancellation: {', '.join(order_ids)}"
                )
            else:
                logger.info("No cancellable open orders found via fetch_open_orders.")
        except Exception as fetch_err:
            logger.warning(
                Fore.YELLOW
                + f"Could not fetch open orders before cancelling: {fetch_err}. Proceeding with cancel all if available."
            )

        # Send cancel_all command or loop through fetched orders
        if open_orders_list:  # Only attempt cancellation if orders were found
            try:
                # Bybit V5 cancel_all_orders also requires category
                cancel_params = {"category": CONFIG.market_type, "symbol": MARKET_INFO["id"]}
                # Use cancel_all_orders for efficiency if supported and reliable
                # Note: cancel_all_orders might not exist or work reliably for all exchanges/params
                # Fallback: loop through fetched open orders and cancel individually
                # Bybit V5 supports POST /v5/order/cancel-all
                if hasattr(EXCHANGE, "private_post_order_cancel_all"):
                    logger.info(f"Using private_post_order_cancel_all for {symbol}...")
                    response = fetch_with_retries(EXCHANGE.private_post_order_cancel_all, params=cancel_params)
                    logger.debug(f"Cancel all orders raw response: {response}")
                    logger.info(f"Cancel all orders command sent for {symbol}. Checking response...")
                    # Check response for success indicators (Bybit V5 returns retCode)
                    if isinstance(response, dict) and response.get("info", {}).get("retCode") == 0:
                        logger.info(Fore.GREEN + "Cancel all command successful (retCode 0).")
                    else:
                        error_msg = (
                            response.get("info", {}).get("retMsg", "Unknown error")
                            if isinstance(response, dict)
                            else str(response)
                        )
                        logger.warning(
                            Fore.YELLOW
                            + f"Cancel all orders command sent, success confirmation unclear or failed: {error_msg}. MANUAL CHECK REQUIRED."
                        )
                else:
                    # Fallback to individual cancellation if cancel_all is not supported or the specific method isn't found
                    logger.info(
                        "cancel_all_orders method not directly available or reliable, cancelling individually..."
                    )
                    cancelled_count = 0
                    for order in open_orders_list:
                        try:
                            order_id = order["id"]
                            logger.debug(f"Cancelling order {order_id}...")
                            # Bybit V5 cancel_order requires category and symbol
                            individual_cancel_params = {"category": CONFIG.market_type, "symbol": MARKET_INFO["id"]}
                            fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=individual_cancel_params)
                            logger.info(f"Cancel request sent for order {order_id}.")
                            cancelled_count += 1
                            time.sleep(0.2)  # Small delay between cancels
                        except ccxt.OrderNotFound:
                            logger.warning(f"Order {order_id} already gone when attempting cancellation.")
                        except Exception as ind_cancel_err:
                            logger.error(f"Failed to cancel order {order_id}: {ind_cancel_err}")
                    logger.info(f"Attempted to cancel {cancelled_count}/{len(open_orders_list)} orders individually.")

            except Exception as cancel_err:
                logger.error(Fore.RED + f"Error sending cancel command(s): {cancel_err}. MANUAL CHECK REQUIRED.")
        else:
            logger.info("Skipping order cancellation as no open orders were found via fetch_open_orders.")

        # Clear local tracker regardless, as intent is to have no active tracked orders
        logger.info("Clearing local order tracker state.")
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    except Exception as e:
        logger.error(
            Fore.RED
            + Style.BRIGHT
            + f"Unexpected error during order cancellation phase: {e}. MANUAL CHECK REQUIRED on exchange.",
            exc_info=True,
        )

    # Add a small delay after cancelling orders before checking/closing positions
    logger.info("Waiting briefly after order cancellation before checking positions...")
    time.sleep(max(CONFIG.order_check_delay_seconds, 2))  # Wait at least 2 seconds

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function with retries
        positions = get_current_position(symbol)

        closed_count = 0
        if positions:
            try:
                # Get minimum quantity for validation using Decimal
                min_qty_dec = Decimal(str(MARKET_INFO["limits"]["amount"]["min"]))
            except (KeyError, InvalidOperation, TypeError):
                logger.warning("Could not determine minimum order quantity for closure validation.")
                min_qty_dec = Decimal("0")  # Assume zero if unavailable

            # Iterate through fetched positions (not the default pos_dict)
            # Ensure we are iterating over the fetched positions, which might be different from the initial pos_dict state
            fetched_positions_to_process = {}
            if positions is not None:
                # Filter for positions with significant quantity
                for side, pos_data in positions.items():
                    qty = pos_data.get("qty", Decimal("0.0"))
                    if qty.copy_abs() >= CONFIG.position_qty_epsilon:
                        fetched_positions_to_process[side] = pos_data

            if not fetched_positions_to_process:
                logger.info(Fore.GREEN + "No significant open positions found requiring closure.")
            else:
                logger.warning(Fore.YELLOW + f"Found {len(fetched_positions_to_process)} positions requiring closure.")

                for side, pos_data in fetched_positions_to_process.items():
                    qty = pos_data.get("qty", Decimal("0.0"))
                    entry_price = pos_data.get("entry_price", Decimal("NaN"))
                    close_side = "sell" if side == "long" else "buy"
                    logger.warning(
                        Fore.YELLOW
                        + f"Closing {side} position (Qty: {qty.normalize()}, Entry: {entry_price:.4f if not entry_price.is_nan() else 'N/A'}) with market order..."
                    )
                    try:
                        # Format quantity precisely for closure order (use absolute value and round down)
                        close_qty_str = format_amount(symbol, qty.copy_abs(), ROUND_DOWN)
                        close_qty_decimal = Decimal(close_qty_str)

                        # Validate against minimum quantity before attempting closure
                        if close_qty_decimal < min_qty_dec:
                            logger.critical(
                                f"{Fore.RED}Closure quantity {close_qty_decimal.normalize()} for {side} position is below exchange minimum {min_qty_dec}. MANUAL CLOSURE REQUIRED!"
                            )
                            termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS < MIN QTY! Close manually!")
                            continue  # Skip trying to close this position

                        # Place the closure market order
                        close_params = {"reduceOnly": True}  # Crucial: Only close, don't open new position
                        # fetch_with_retries handles category param
                        close_order = fetch_with_retries(
                            EXCHANGE.create_market_order,
                            symbol=symbol,
                            side=close_side,
                            amount=float(close_qty_decimal),  # CCXT needs float
                            params=close_params,
                        )

                        # Check response for success
                        if close_order and (close_order.get("id") or close_order.get("info", {}).get("retCode") == 0):
                            close_id = close_order.get("id", "N/A (retCode 0)")
                            logger.trade(Fore.GREEN + f"Position closure order placed successfully: ID {close_id}")
                            closed_count += 1
                            # Wait briefly to allow fill confirmation before checking next position (if any)
                            time.sleep(max(CONFIG.order_check_delay_seconds, 2))
                            # Optional: Verify closure order status? Might slow shutdown significantly.
                            # For shutdown, placing the order is usually sufficient as market orders fill fast.
                        else:
                            # Log critical error if closure order placement fails
                            error_msg = (
                                close_order.get("info", {}).get("retMsg", "No ID and no success code.")
                                if isinstance(close_order, dict)
                                else str(close_order)
                            )
                            logger.critical(
                                Fore.RED
                                + Style.BRIGHT
                                + f"FAILED TO PLACE closure order for {side} position ({qty.normalize()}): {error_msg}. MANUAL INTERVENTION REQUIRED!"
                            )
                            termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")

                    except (
                        ccxt.InsufficientFunds,
                        ccxt.InvalidOrder,
                        ccxt.ExchangeError,
                        ccxt.BadRequest,
                        ccxt.PermissionDenied,
                    ) as e:
                        logger.critical(
                            Fore.RED
                            + Style.BRIGHT
                            + f"FAILED TO CLOSE {side} position ({qty.normalize()}): {e}. MANUAL INTERVENTION REQUIRED!"
                        )
                        termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")
                    except Exception as e:
                        logger.critical(
                            Fore.RED
                            + Style.BRIGHT
                            + f"Unexpected error closing {side} position: {e}. MANUAL INTERVENTION REQUIRED!",
                            exc_info=True,
                        )
                        termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")

            # Final summary message
            if closed_count == len(fetched_positions_to_process):
                logger.info(
                    Fore.GREEN + f"Successfully placed closure orders for all {closed_count} detected positions."
                )
            elif closed_count > 0:
                logger.warning(
                    Fore.YELLOW
                    + f"Placed closure orders for {closed_count} positions, but {len(fetched_positions_to_process) - closed_count} positions may remain. MANUAL CHECK REQUIRED."
                )
                termux_notify(
                    "Shutdown Warning!",
                    f"{symbol} Manual check needed - {len(fetched_positions_to_process) - closed_count} positions might remain.",
                )
            else:
                logger.warning(
                    Fore.YELLOW
                    + "Attempted shutdown but closure orders failed or were not possible for all open positions. MANUAL CHECK REQUIRED."
                )
                termux_notify("Shutdown Warning!", f"{symbol} Manual check needed - positions might remain.")

        elif positions is None:
            # Failure to fetch positions during shutdown is critical
            logger.critical(
                Fore.RED
                + Style.BRIGHT
                + "Could not fetch final positions during shutdown. MANUAL CHECK REQUIRED on exchange!"
            )
            termux_notify("Shutdown Warning!", f"{symbol} Cannot confirm position status. Check exchange!")

    except Exception as e:
        logger.error(
            Fore.RED + Style.BRIGHT + f"Error during position closure phase: {e}. Manual check advised.", exc_info=True
        )
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Error during position closure. Check logs.")

    logger.warning(Fore.YELLOW + Style.BRIGHT + "Graceful Shutdown Sequence Complete.")
    termux_notify("Shutdown Complete", f"{CONFIG.symbol} bot stopped.")


# --- Main Spell Invocation ---
if __name__ == "__main__":
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80)
    print(
        Back.MAGENTA
        + Fore.WHITE
        + Style.BRIGHT
        + "*** Pyrmethus Termux Trading Spell Activated (v2.1.3 Optimized Signals & Precision) ***"
    )
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL)

    logger.info("Initializing Pyrmethus v2.1.3...")
    logger.info(f"Log Level configured to: {log_level_str}")

    # Log key configuration parameters for verification
    logger.info("--- Trading Configuration ---")
    logger.info(f"Symbol: {CONFIG.symbol} ({CONFIG.market_type.capitalize()})")
    logger.info(f"Timeframe: {CONFIG.interval}")
    logger.info(f"Risk per trade: {CONFIG.risk_percentage * 100:.5f}%")  # Show more precision for risk
    logger.info(f"SL Multiplier: {CONFIG.sl_atr_multiplier}")
    logger.info(f"TSL Activation: {CONFIG.tsl_activation_atr_multiplier} * ATR Profit")
    logger.info(f"TSL Trail Percent: {CONFIG.trailing_stop_percent}%")
    logger.info(f"Trigger Prices: SL={CONFIG.sl_trigger_by}, TSL={CONFIG.tsl_trigger_by}")
    logger.info(f"Trend Filter EMA({CONFIG.trend_ema_period}): {CONFIG.trade_only_with_trend}")
    logger.info(
        f"Indicator Periods: Trend EMA({CONFIG.trend_ema_period}), Fast EMA({CONFIG.fast_ema_period}), Slow EMA({CONFIG.slow_ema_period}), Stoch({CONFIG.stoch_period},{CONFIG.stoch_smooth_k},{CONFIG.stoch_smooth_d}), ATR({CONFIG.atr_period})"
    )
    logger.info(
        f"Signal Thresholds: Stoch OS(<{CONFIG.stoch_oversold_threshold}), Stoch OB(>{CONFIG.stoch_overbought_threshold}), Trend Buffer({CONFIG.trend_filter_buffer_percent}%), ATR Move Filter({CONFIG.atr_move_filter_multiplier}x ATR)"
    )
    logger.info(f"Position Quantity Epsilon: {CONFIG.position_qty_epsilon:.2E}")  # Scientific notation
    logger.info(f"Loop Interval: {CONFIG.loop_sleep_seconds}s")
    logger.info(f"OHLCV Limit: {CONFIG.ohlcv_limit}")
    logger.info(f"Fetch Retries: {CONFIG.max_fetch_retries}")
    logger.info(f"Order Check Timeout: {CONFIG.order_check_timeout_seconds}s")
    logger.info("-----------------------------")

    # Final check if exchange connection and market info loading succeeded
    if MARKET_INFO and EXCHANGE:
        termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2.1.3)")
        logger.info(Fore.GREEN + Style.BRIGHT + "Initialization complete. Awaiting market whispers...")
        print(Fore.MAGENTA + "=" * 80 + Style.RESET_ALL)  # Separator before first cycle log
    else:
        # Error should have been logged during init, exit was likely called, but double-check.
        logger.critical(
            Fore.RED
            + Style.BRIGHT
            + "Exchange or Market info failed to load during initialization. Cannot start trading loop."
        )
        sys.exit(1)

    cycle = 0
    try:
        while True:
            cycle += 1
            try:
                trading_spell_cycle(cycle)
            except Exception as cycle_error:
                # Catch errors *within* a cycle to prevent the whole script from crashing
                logger.error(
                    Fore.RED + Style.BRIGHT + f"Error during trading cycle {cycle}: {cycle_error}", exc_info=True
                )
                termux_notify("Cycle Error!", f"{CONFIG.symbol} Cycle {cycle} failed. Check logs.")
                # Decide if a single cycle failure is fatal. For now, log and continue to the next cycle after sleep.
                # If errors are persistent, fetch_with_retries/other checks should eventually halt.

            logger.info(Fore.BLUE + f"Cycle {cycle} finished. Resting for {CONFIG.loop_sleep_seconds} seconds...")
            time.sleep(CONFIG.loop_sleep_seconds)

    except KeyboardInterrupt:
        logger.warning(Fore.YELLOW + "\nCtrl+C detected! Initiating graceful shutdown...")
        graceful_shutdown()
    except Exception as e:
        # Catch unexpected errors in the main loop *outside* of the trading_spell_cycle call
        logger.critical(
            Fore.RED + Style.BRIGHT + f"\nFATAL RUNTIME ERROR in Main Loop (Cycle {cycle}): {e}", exc_info=True
        )
        termux_notify("Bot CRASHED!", f"{CONFIG.symbol} FATAL ERROR! Check logs!")
        logger.warning(Fore.YELLOW + "Attempting graceful shutdown after crash...")
        try:
            graceful_shutdown()  # Attempt cleanup even on unexpected crash
        except Exception as shutdown_err:
            logger.error(f"Error during crash shutdown: {shutdown_err}", exc_info=True)
        sys.exit(1)  # Exit with error code
    finally:
        # Ensure logs are flushed before exit, regardless of how loop ended
        logger.info("Flushing logs...")
        logging.shutdown()
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80)
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Trading Spell Deactivated ***")
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL)
