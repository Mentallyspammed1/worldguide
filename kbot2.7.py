#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: 2.3
# Changelog:
# - v2.3: Integrated ECC calculation and PnL fetching from separate script.
#         Added ECC scalping as a configurable strategy.
#         Enhanced configuration validation and logging.
#         Improved Decimal usage and precision handling.
#         Refined error handling and retries.
#         Enhanced status panel with more details and color.
#         Improved Termux notification handling.
#         Added graceful shutdown sequence.
# - v2.2: Added Ehlers Cyber Cycle calculation and scalping signals (in original script).
# - v2.1: Added fetch_pnl for unrealized and realized PnL (in original script).
# - v2.0: Fixed error 10004 by including recv_window in signature (in original script).
# - v1.0: Initial version with trailing stop functionality (in original script).
# Pyrmethus - Termux Trading Spell (v2.3 - Integrated ECC & PnL)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved robustness.

import os
import time
import logging
import sys
import subprocess  # For termux-toast security
import copy  # For deepcopy of tracker state
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
    import requests  # Often needed by ccxt, good to suggest installation
except ImportError as e:
    # Provide specific guidance for Termux users
    init(autoreset=True)  # Initialize colorama for error messages
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

# Set Decimal precision (adjust if needed)
# The default precision (usually 28) is often sufficient for most price/qty calcs.
# It might be necessary to increase if dealing with extremely small values or very high precision instruments.
# getcontext().prec = 50 # Example: Increase precision if needed
# By default, the Decimal context is set with sufficient precision (typically 28 digits).

# --- Arcane Configuration ---
print(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.3...")

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
if not hasattr(logging.Logger, 'trade'):
    logging.Logger.trade = trade_log

# More detailed log format, includes module and line number for easier debugging
log_formatter = logging.Formatter(
    Fore.CYAN + "%(asctime)s "
    + Style.BRIGHT + "[%(levelname)-8s] "  # Padded levelname
    + Fore.WHITE + "(%(filename)s:%(lineno)d) "  # Added file/line info
    + Style.RESET_ALL
    + Fore.WHITE + "%(message)s"
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
    stream_handler.setLevel(log_level)  # Set handler level to match logger
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
        self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5"))  # 0.001% to 50% risk
        self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"))
        # TSL activation threshold in ATR units above entry price
        self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"))
        # Bybit V5 TSL distance is a percentage (e.g., 0.5 for 0.5%). Ensure value is suitable.
        self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.001"), max_val=Decimal("10.0"))  # 0.001% to 10% trail
        # Trigger type for SL/TSL orders. Bybit V5 allows LastPrice, MarkPrice, IndexPrice.
        self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])  # Usually same as SL

        # Trading Strategy Selection
        # Supported strategies: 'ecc_scalp', 'ema_stoch'
        self.strategy = self._get_env("STRATEGY", "ecc_scalp", Fore.YELLOW, allowed_values=["ecc_scalp", "ema_stoch"]).lower()

        # ECC Specific Parameters
        self.ecc_alpha = self._get_env("ECC_ALPHA", "0.15", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.01"), max_val=Decimal("0.5"))
        self.ecc_lookback = self._get_env("ECC_LOOKBACK", "10", Fore.YELLOW, cast_type=int, min_val=5)

        # EMA/Stoch Specific Parameters (if strategy is ema_stoch)
        self.ema_fast_period = self._get_env("EMA_FAST_PERIOD", "8", Fore.YELLOW, cast_type=int, min_val=2)
        self.ema_slow_period = self._get_env("EMA_SLOW_PERIOD", "12", Fore.YELLOW, cast_type=int, min_val=2)
        self.stoch_period = self._get_env("STOCH_PERIOD", "10", Fore.YELLOW, cast_type=int, min_val=5)
        self.stoch_smooth_k = self._get_env("STOCH_SMOOTH_K", "3", Fore.YELLOW, cast_type=int, min_val=1)
        self.stoch_smooth_d = self._get_env("STOCH_SMOOTH_D", "3", Fore.YELLOW, cast_type=int, min_val=1)
        self.trend_ema_period = self._get_env("TREND_EMA_PERIOD", "20", Fore.YELLOW, cast_type=int, min_val=5, max_val=500)
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)

        # ATR Period
        self.atr_period = self._get_env("ATR_PERIOD", "10", Fore.YELLOW, cast_type=int, min_val=5)

        # PnL Fetching
        self.fetch_pnl_interval_cycles = self._get_env("FETCH_PNL_INTERVAL_CYCLES", "10", Fore.YELLOW, cast_type=int, min_val=1) # Fetch PnL every N cycles
        self.pnl_lookback_days = self._get_env("PNL_LOOKBACK_DAYS", "7", Fore.YELLOW, cast_type=int, min_val=1) # Lookback for realized PnL history

        # Epsilon: Small value for comparing quantities, dynamically determined after market info is loaded.
        self.position_qty_epsilon = Decimal("1E-12")  # Default tiny Decimal, will be overridden based on market precision

        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000)  # Reasonable candle limits
        self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5)  # Minimum sleep time
        self.order_check_delay_seconds = self._get_env("ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1)
        self.order_check_timeout_seconds = self._get_env("ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5)
        self.max_fetch_retries = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)


        if not self.api_key or not self.api_secret:
            logger.critical(Fore.RED + Style.BRIGHT + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting.")
            sys.exit(1)

        # Validate market type
        if self.market_type not in ['linear', 'inverse']:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Invalid MARKET_TYPE '{self.market_type}'. Must be 'linear' or 'inverse'. Halting.")
            sys.exit(1)

        # Validate strategy specific parameters based on chosen strategy
        if self.strategy == 'ecc_scalp':
            # ECC requires a certain amount of data relative to lookback
            min_ohlcv_for_ecc = self.ecc_lookback + 4 # ECC formula needs data from index 4 onwards relative to prices array start
            if self.ohlcv_limit < min_ohlcv_for_ecc:
                logger.warning(f"{Fore.YELLOW}OHLCV_LIMIT ({self.ohlcv_limit}) might be insufficient for ECC_LOOKBACK ({self.ecc_lookback}). Minimum recommended for stable ECC: {min_ohlcv_for_ecc}. Increasing OHLCV_LIMIT to {min_ohlcv_for_ecc}.")
                self.ohlcv_limit = min_ohlcv_for_ecc # Auto-correct or warn? Let's warn and proceed.
            # ECC also benefits from more data for initial smoothing steps to settle.
            # Let's recommend a slightly higher limit.
            recommended_ohlcv_for_ecc = self.ecc_lookback + 20 # From original ECC script
            if self.ohlcv_limit < recommended_ohlcv_for_ecc:
                logger.warning(f"{Fore.YELLOW}OHLCV_LIMIT ({self.ohlcv_limit}) is less than recommended ({recommended_ohlcv_for_ecc}) for stable ECC. Consider increasing OHLCV_LIMIT.")


        elif self.strategy == 'ema_stoch':
            # EMA/Stoch requires data length relative to periods
            required_len_ema_stable = max(self.ema_fast_period, self.ema_slow_period, self.trend_ema_period)
            required_len_stoch = self.stoch_period + self.stoch_smooth_k + self.stoch_smooth_d - 2
            min_required_len = max(required_len_ema_stable, required_len_stoch)
            min_safe_len = min_required_len + max(self.stoch_smooth_d, 1) # Small buffer

            if self.ohlcv_limit < min_safe_len:
                 logger.warning(f"{Fore.YELLOW}OHLCV_LIMIT ({self.ohlcv_limit}) is less than recommended ({min_safe_len}) for stable EMA/Stoch. Consider increasing OHLCV_LIMIT.")

        # ATR is used by both strategies for SL/TSL
        min_required_atr = self.atr_period + 1
        if self.ohlcv_limit < min_required_atr:
             logger.warning(f"{Fore.YELLOW}OHLCV_LIMIT ({self.ohlcv_limit}) is less than minimum required ({min_required_atr}) for ATR calculation. Consider increasing OHLCV_LIMIT.")


        logger.debug("Configuration loaded successfully.")

    def _get_env(self, key: str, default: Any, color: str, cast_type: type = str,
                 min_val: Optional[Union[int, Decimal]] = None,
                 max_val: Optional[Union[int, Decimal]] = None,
                 allowed_values: Optional[List[str]] = None) -> Any:
        """Gets value from environment, casts, validates, and logs."""
        value_str = os.getenv(key)
        # Mask secrets in logs
        log_value = "****" if "SECRET" in key or "KEY" in key else value_str

        if value_str is None or value_str.strip() == "":  # Treat empty string as not set
            value = default
            if default is not None:
                # Only log default if it's not a secret
                logger.warning(f"{color}Using default value for {key}: {default if 'SECRET' not in key and 'KEY' not in key else '****'}")
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
                # Case-insensitive check for common truthy strings
                casted_value = value_str.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                casted_value = Decimal(value_str)
            elif cast_type == int:
                casted_value = int(value_str)
            elif cast_type == float:
                casted_value = float(value_str)  # Generally avoid float for critical values, but allow if needed
            else:  # Default is str
                casted_value = str(value_str)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Could not cast {key} ('{value_str}') to {cast_type.__name__}: {e}. Using default: {default}")
            # Attempt to cast the default value itself
            try:
                if default is None: return None
                # Recast default carefully
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting.")
                sys.exit(1)

        # --- Validation ---
        if casted_value is None:  # Should not happen if casting succeeded or defaulted
            logger.critical(f"{Fore.RED+Style.BRIGHT}Failed to obtain a valid value for {key}. Halting.")
            sys.exit(1)

        # Allowed values check (for strings like trigger types, strategy)
        if allowed_values and casted_value not in allowed_values:
            logger.error(f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}. Using default: {default}")
            # Return default after logging error
            return default  # Assume default is valid

        # Min/Max checks (for numeric types - Decimal, int, float)
        validation_failed = False
        try:
            # Check if casted_value is a number before comparing
            if isinstance(casted_value, (int, float, Decimal)):
                if min_val is not None:
                    # Ensure min_val is Decimal if casted_value is Decimal for accurate comparison
                    min_val_comp = Decimal(str(min_val)) if isinstance(casted_value, Decimal) else min_val
                    if casted_value < min_val_comp:
                        logger.error(f"{Fore.RED}{key} value {casted_value} is below minimum {min_val}. Using default: {default}")
                        validation_failed = True
                if max_val is not None:
                    # Ensure max_val is Decimal if casted_value is Decimal
                    max_val_comp = Decimal(str(max_val)) if isinstance(casted_value, Decimal) else max_val
                    if casted_value > max_val_comp:
                        logger.error(f"{Fore.RED}{key} value {casted_value} is above maximum {max_val}. Using default: {default}")
                        validation_failed = True
        except InvalidOperation as e:
            logger.error(f"{Fore.RED}Error during min/max validation for {key} with value {casted_value} and limits ({min_val}, {max_val}): {e}. Using default: {default}")
            validation_failed = True
        except TypeError as e:
            logger.warning(f"Skipping min/max validation for non-numeric type {type(casted_value).__name__} for key {key}. Error: {e}")

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
MARKET_INFO: Optional[Dict] = None  # Global to store market details after connection
EXCHANGE: Optional[ccxt.Exchange] = None  # Global for the exchange instance

# --- Exchange Nexus Initialization ---
print(Fore.MAGENTA + Style.BRIGHT + "\nEstablishing Nexus with the Exchange v2.3...")
try:
    exchange_options = {
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True,  # CCXT built-in rate limiter
        "options": {
            'defaultType': 'swap',  # More specific for futures/swaps than 'future'
            'defaultSubType': CONFIG.market_type,  # 'linear' or 'inverse'
            'adjustForTimeDifference': True,  # Auto-sync clock with server
            # Bybit V5 API often requires 'category' for unified endpoints
            'brokerId': 'PyrmethusV23',  # Custom identifier for Bybit API tracking
            'v5': {'category': CONFIG.market_type}  # Explicitly set category for V5 requests
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
    EXCHANGE.check_required_credentials()  # Checks if keys are present/formatted ok
    logger.info("Credentials format check passed.")
    # Fetch time to verify connectivity, API key validity, and clock sync
    server_time = EXCHANGE.fetch_time()
    local_time = EXCHANGE.milliseconds()
    time_diff = abs(server_time - local_time)
    logger.info(f"Exchange time synchronized: {EXCHANGE.iso8601(server_time)} (Difference: {time_diff} ms)")
    if time_diff > 5000:  # Warn if clock skew is significant (e.g., > 5 seconds)
        logger.warning(f"{Fore.YELLOW}Significant time difference ({time_diff} ms) between system and exchange. Check system clock synchronization.")

    # Load markets (force reload to ensure fresh data)
    logger.info("Loading market spirits (market data)...")
    EXCHANGE.load_markets(True)  # Force reload
    logger.info(Fore.GREEN + Style.BRIGHT + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets).")

    # Verify symbol exists and get market details
    if CONFIG.symbol not in EXCHANGE.markets:
        logger.error(Fore.RED + Style.BRIGHT + f"Symbol {CONFIG.symbol} not found in Bybit {CONFIG.market_type} market spirits.")
        # Suggest available symbols more effectively
        available_symbols = []
        try:
            # Extract settle currency robustly (handles SYMBOL/QUOTE:SETTLE format)
            # For futures, settle currency is often the key identifier in market lists
            settle_currency_candidates = CONFIG.symbol.split(':')  # e.g., ['BTC/USDT', 'USDT']
            settle_currency = settle_currency_candidates[-1] if len(settle_currency_candidates) > 1 else None
            if settle_currency:
                logger.info(f"Searching for symbols settling in {settle_currency}...")
                for s, m in EXCHANGE.markets.items():
                    # Check if market matches the configured type (linear/inverse) and is active
                    is_correct_type = (CONFIG.market_type == 'linear' and m.get('linear')) or \
                                      (CONFIG.market_type == 'inverse' and m.get('inverse'))
                    # Filter by settle currency and check if active
                    # Use .get(key, default) for safer access
                    if m.get('active', False) and is_correct_type and m.get('settle') == settle_currency:
                        available_symbols.append(s)
            else:
                logger.warning(f"Could not parse settle currency from SYMBOL '{CONFIG.symbol}'. Cannot filter suggestions.")
                # Fallback: List all active symbols of the correct type
                for s, m in EXCHANGE.markets.items():
                    is_correct_type = (CONFIG.market_type == 'linear' and m.get('linear', False)) or \
                                      (CONFIG.market_type == 'inverse' and m.get('inverse', False))
                    if m.get('active', False) and is_correct_type:
                        available_symbols.append(s)

        except IndexError:
            logger.error(f"Could not parse base/quote from SYMBOL '{CONFIG.symbol}'.")
        except Exception as e:
            logger.error(f"Error suggesting symbols: {e}")

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
            price_precision_raw = MARKET_INFO['precision'].get('price')
            # precision['amount'] might be a step size (Decimal) or number of decimal places (int)
            amount_precision_raw = MARKET_INFO['precision'].get('amount')
            min_amount_raw = MARKET_INFO['limits']['amount'].get('min')
            max_amount_raw = MARKET_INFO['limits']['amount'].get('max')  # Max might be None
            contract_size_raw = MARKET_INFO.get('contractSize', '1')  # Default to '1' if not present
            min_cost_raw = MARKET_INFO['limits'].get('cost', {}).get('min')  # Min cost might not exist

            # Convert to Decimal for logging and potential use, handle None/N/A
            price_prec_str = str(price_precision_raw) if price_precision_raw is not None else "N/A"
            amount_prec_str = str(amount_precision_raw) if amount_precision_raw is not None else "N/A"
            min_amount_dec = Decimal(str(min_amount_raw)) if min_amount_raw is not None else Decimal("NaN")
            max_amount_dec = Decimal(str(max_amount_raw)) if max_amount_raw is not None else Decimal("Infinity")  # Use Infinity for no max
            contract_size_dec = Decimal(str(contract_size_raw)) if contract_size_raw is not None else Decimal("NaN")
            min_cost_dec = Decimal(str(min_cost_raw)) if min_cost_raw is not None else Decimal("NaN")

            logger.debug(f"Market Precision: Price Tick/Decimals={price_prec_str}, Amount Step/Decimals={amount_prec_str}")
            logger.debug(f"Market Limits: Min Amount={min_amount_dec}, Max Amount={max_amount_dec}, Min Cost={min_cost_dec}")
            logger.debug(f"Contract Size: {contract_size_dec}")

            # --- Dynamically set epsilon based on amount precision (step size) ---
            # CCXT often provides amount precision as the step size directly
            amount_step_size = MARKET_INFO['precision'].get('amount')
            if amount_step_size is not None:
                try:
                    # Use a very small fraction of the step size as epsilon
                    # A safe default is a very small number like 1E-12, which is much smaller than typical amount steps.
                    CONFIG.position_qty_epsilon = Decimal("1E-12")  # A very small, fixed epsilon
                    logger.info(f"Set position_qty_epsilon to a small fixed value: {CONFIG.position_qty_epsilon:.1E}")  # Log the chosen epsilon
                except (InvalidOperation, TypeError):
                    logger.warning(f"Could not parse amount step size '{amount_step_size}'. Using default epsilon: {CONFIG.position_qty_epsilon:.1E}")
            else:
                logger.warning(f"Market info does not provide amount step size ('precision.amount'). Using default epsilon: {CONFIG.position_qty_epsilon:.1E}")

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
# Uses placeholders like "POS_SL_LONG", "POS_TSL_LONG" for Bybit V5 position-based stops.
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
        check_cmd = subprocess.run(['which', 'termux-toast'], capture_output=True, text=True, check=False, timeout=3)  # Added timeout for 'which'
        if check_cmd.returncode != 0:
            logger.debug("termux-toast command not found. Skipping notification.")
            return

        # Basic sanitization - focus on preventing shell interpretation issues
        # Replace potentially problematic characters with spaces or remove them
        safe_title = title.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('(', '').replace(')', '').strip()
        safe_content = content.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('(', '').replace(')', '').strip()

        if not safe_title and not safe_content:
            logger.debug("Notification content is empty after sanitization. Skipping.")
            return

        # Limit length to avoid potential buffer issues or overly long toasts
        max_len = 250  # Increased length
        full_message = f"{safe_title}: {safe_content}" if safe_title and safe_content else safe_title if safe_title else safe_content
        full_message = full_message[:max_len].strip()  # Trim and strip whitespace

        if not full_message:
            logger.debug("Notification message is empty after combining and trimming. Skipping.")
            return

        # Use list format for subprocess.run for security
        # Example styling: gravity middle, black text on green background, long duration for detailed messages
        cmd_list = ['termux-toast', '-g', 'middle', '-c', 'black', '-b', 'green', '-s', 'long', full_message]
        # Ensure cmd_list contains only strings
        cmd_list = [str(arg) for arg in cmd_list]

        # Run the command non-blocking or with a short timeout
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=5)  # Add timeout

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

def format_price(symbol: str, price: Union[Decimal, str, float, int]) -> str:
    """Formats price according to market precision rules using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format price.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            price_dec = Decimal(str(price))
            # Use enough decimal places for a fallback, maybe based on typical crypto prices
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))  # Quantize to 8 decimal places
        except Exception:
            logger.error(f"Fallback price formatting failed for {price}.")
            return str(price)  # Last resort

    # Ensure input is Decimal first for internal consistency, then float for CCXT methods
    try:
        price_dec = Decimal(str(price))
    except (InvalidOperation, TypeError):
        logger.error(f"Cannot format price '{price}': Invalid number format.")
        return str(price)  # Return original input if cannot even convert to Decimal

    try:
        # CCXT's price_to_precision handles rounding/truncation based on market rules (tick size).
        # Ensure input is float as expected by CCXT methods.
        price_float = float(price_dec)
        return EXCHANGE.price_to_precision(symbol, price_float)
    except (AttributeError, KeyError, InvalidOperation) as e:
        logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid price format: {e}. Using fallback formatting.")
        try:
            # Fallback with Decimal quantize based on a reasonable number of decimal places
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))
        except Exception:
            logger.error(f"Fallback price formatting failed for {price_dec}.")
            return str(price_dec)  # Last resort (Decimal as string)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price_dec} for {symbol}: {e}. Using fallback.")
        try:
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))
        except Exception:
            logger.error(f"Fallback price formatting failed for {price_dec}.")
            return str(price_dec)

def format_amount(symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
    """Formats amount according to market precision rules (step size) using exchange's method."""
    global MARKET_INFO, EXCHANGE
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format amount.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            amount_dec = Decimal(str(amount))
            # Use quantize for fallback if Decimal input (e.g., 8 decimal places)
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            logger.error(f"Fallback amount formatting failed for {amount}.")
            return str(amount)  # Last resort

    # Ensure input is Decimal first for internal consistency, then float for CCXT methods
    try:
        amount_dec = Decimal(str(amount))
    except (InvalidOperation, TypeError):
        logger.error(f"Cannot format amount '{amount}': Invalid number format.")
        return str(amount)  # Return original input if cannot even convert to Decimal

    try:
        # CCXT's amount_to_precision handles step size and rounding.
        # Map Python Decimal rounding modes to CCXT rounding modes if needed.
        # CCXT primarily supports ROUND (nearest) and TRUNCATE (down).
        ccxt_rounding_mode = ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND  # Basic mapping
        # Ensure input is float as expected by CCXT methods.
        amount_float = float(amount_dec)
        return EXCHANGE.amount_to_precision(symbol, amount_float, rounding_mode=ccxt_rounding_mode)
    except (AttributeError, KeyError, InvalidOperation) as e:
        logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid amount format: {e}. Using fallback formatting.")
        try:
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            logger.error(f"Fallback amount formatting failed for {amount_dec}.")
            return str(amount_dec)  # Last resort (Decimal as string)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount_dec} for {symbol}: {e}. Using fallback.")
        try:
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            logger.error(f"Fallback amount formatting failed for {amount_dec}.")
            return str(amount_dec)

# --- Core Spell Functions ---

def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    """Generic wrapper to fetch data with retries and exponential backoff."""
    global EXCHANGE
    if EXCHANGE is None:
        logger.critical("Exchange object is None, cannot fetch data.")
        return None  # Indicate critical failure

    last_exception = None
    # Add category param automatically for V5 if not already present in kwargs['params']
    # Only attempt this if EXCHANGE and its options structure exist
    if EXCHANGE is not None and hasattr(EXCHANGE, 'options') and 'v5' in EXCHANGE.options and 'category' in EXCHANGE.options['v5']:
        if 'params' not in kwargs:
            kwargs['params'] = {}
        # Check if category is *already* in params before adding
        if 'category' not in kwargs['params']:
            kwargs['params']['category'] = EXCHANGE.options['v5']['category']
            # logger.debug(f"Auto-added category '{kwargs['params']['category']}' to params for {fetch_function.__name__}")

    for attempt in range(CONFIG.max_fetch_retries + 1):  # +1 to allow logging final failure
        try:
            # Log the attempt number and function being called at DEBUG level
            # Be cautious not to log sensitive parameters like API keys if they were somehow passed directly
            log_kwargs = {k: ('****' if 'secret' in str(k).lower() or 'key' in str(k).lower() else v) for k, v in kwargs.items()}
            log_args = tuple('****' if isinstance(arg, str) and ('secret' in arg.lower() or 'key' in arg.lower()) else arg for arg in args)  # Basic sanitization for args too
            logger.debug(f"Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}: Calling {fetch_function.__name__} with args={log_args}, kwargs={log_kwargs}")
            result = fetch_function(*args, **kwargs)
            return result  # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = 2 ** attempt  # Exponential backoff (1, 2, 4, 8...)
            logger.warning(Fore.YELLOW + f"{fetch_function.__name__}: Network issue (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}). Retrying in {wait_time}s... Error: {e}")
            if attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            else:
                logger.error(Fore.RED + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to network issues.")
                break # Exit retry loop
        except ccxt.ExchangeNotAvailable as e:
            last_exception = e
            logger.error(Fore.RED + f"{fetch_function.__name__}: Exchange not available: {e}. Stopping retries.")
            return None # Indicate hard failure
        except ccxt.AuthenticationError as e:
            last_exception = e
            logger.critical(Fore.RED + Style.BRIGHT + f"{fetch_function.__name__}: Authentication error: {e}. Halting script.")
            sys.exit(1)  # Exit immediately on auth failure
        except (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied) as e:
            # These are typically non-retryable errors related to the request parameters or exchange state.
            last_exception = e
            error_type = type(e).__name__
            logger.error(Fore.RED + f"{fetch_function.__name__}: Non-retryable error ({error_type}): {e}. Stopping retries for this call.")
            # Re-raise these specific errors so the caller can handle them appropriately
            raise e
        except ccxt.ExchangeError as e:
            # Includes rate limit errors, potentially invalid requests etc.
            last_exception = e
            # Check for specific retryable Bybit error codes if needed (e.g., 10006=timeout, 10016=internal error)
            # Bybit V5 Rate limit codes: 10018 (IP), 10017 (Key), 10009 (Frequency)
            # Bybit V5 Invalid Parameter codes often start with 11xxxx.
            # General internal errors: 10006, 10016, 10020, 10030
            # Other: 30034 (Position status not normal), 110025 (SL/TP order not found or completed)
            error_code = getattr(e, 'code', None)  # CCXT might parse the code from info dict
            error_message = str(e)
            should_retry = True
            wait_time = 2 * (attempt + 1)  # Default backoff

            # Check for common rate limit patterns / codes
            if "Rate limit exceeded" in error_message or error_code in [10017, 10018, 10009]:
                wait_time = 5 * (attempt + 1)  # Longer wait for rate limits
                logger.warning(f"{Fore.YELLOW}{fetch_function.__name__}: Rate limit hit (Code: {error_code}). Retrying in {wait_time}s... Error: {e}")
            # Check for specific non-retryable errors (e.g., invalid parameter codes, insufficient funds codes)
            # These often start with 11xxxx for Bybit V5
            elif error_code is not None and (110000 <= error_code <= 110100 or error_code in [30034]):
                logger.error(Fore.RED + f"{fetch_function.__name__}: Non-retryable parameter/logic/state exchange error (Code: {error_code}): {e}. Stopping retries.")
                should_retry = False
            else:
                # General exchange error, apply default backoff
                logger.warning(f"{Fore.YELLOW}{fetch_function.__name__}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}, Code: {error_code}). Retrying in {wait_time}s... Error: {e}")

            if should_retry and attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            elif should_retry:  # Final attempt failed
                logger.error(Fore.RED + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to exchange errors.")
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
    if isinstance(last_exception, (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied)):
        raise last_exception  # Propagate specific non-retryable errors

    # For other failures (Network, ExchangeNotAvailable, general ExchangeError after retries), return None
    if last_exception:
        logger.error(f"{fetch_function.__name__} ultimately failed after {CONFIG.max_fetch_retries + 1} attempts or encountered a non-retryable error type not explicitly re-raised.")
        return None  # Indicate failure

    # Should not reach here if successful, but defensive return None
    # If we reached the end of the loop without returning or raising, it implies failure.
    # The function should only return the result on success.
    # So if we are here, it's a failure case.
    return None


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
        # fetch_with_retries handles category param automatically
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
        # Use Decimal for numeric columns directly during DataFrame creation where possible
        # However, pandas expects floats for most calculations, so convert back later
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp immediately to UTC datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True)  # Drop rows where timestamp conversion failed

        # Convert numeric columns to float first for pandas/numpy compatibility
        for col in ["open", "high", "low", "close", "volume"]:
            # Ensure column exists before attempting conversion
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.warning(f"OHLCV data is missing expected column: {col}")

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
            logger.warning(Fore.YELLOW + f"Duplicate timestamps found in OHLCV data ({len(duplicates)} unique duplicates). Keeping last entry for each.")
            df = df[~df.index.duplicated(keep='last')]

        # Check time difference between last two candles vs expected interval
        if len(df) > 1:
            time_diff = df.index[-1] - df.index[-2]
            try:
                # Use pandas to parse timeframe string robustly
                expected_interval_ms = EXCHANGE.parse_timeframe(timeframe) * 1000  # Convert to milliseconds
                expected_interval_td = pd.Timedelta(expected_interval_ms, unit='ms')
                # Allow some tolerance (e.g., 20% of interval) for minor timing differences/API lag
                tolerance = expected_interval_td * 0.2
                if abs(time_diff.total_seconds()) > expected_interval_td.total_seconds() + tolerance.total_seconds():
                    logger.warning(f"Unexpected large time gap between last two candles: {time_diff} (expected ~{expected_interval_td})")
            except ValueError:
                logger.warning(f"Could not parse timeframe '{timeframe}' to calculate expected interval for time gap check.")
            except Exception as time_check_e:
                logger.warning(f"Error during time difference check: {time_check_e}")

        logger.info(Fore.GREEN + f"Market whispers received ({len(df)} candles). Latest: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return df
    except Exception as e:
        logger.error(Fore.RED + f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
        return None

# --- ECC Calculation (from Block 1, adapted) ---
def calculate_ehlers_cyber_cycle(prices: np.ndarray, alpha: Decimal, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Ehlers Cyber Cycle and its trigger line.

    Args:
        prices: Numpy array of closing prices (float).
        alpha: Smoothing factor (Decimal).
        lookback: Number of bars for cycle calculation (int).

    Returns:
        Tuple of (cyber_cycle, trigger_line) arrays (float).
        Arrays will be empty if input prices are insufficient.
    """
    # Ensure enough data for calculation based on lookback + initial smoothing
    # The formula uses prices[i-3], cycle[i-1], cycle[i-2], smooth[i], smooth[i-1], smooth[i-2]
    # Initial smooth requires prices[i-3] for i=2, so needs 3 prior prices + current
    # Initial cycle requires smooth[i-2] for i=4, so needs 2 prior smooth values, each needing 3 prior prices
    # Roughly requires 4 + lookback points for potentially stable results.
    # Let's check against the length needed for the formula's loops to start:
    # Smooth loop starts at i=2, needs prices[i-3] so minimum 3 prices. First smooth value at index 3.
    # Cycle loop starts at i=4, needs smooth[i-2] so needs smooth at index 2.
    # smooth[2] uses prices[2], prices[1], prices[0].
    # cycle[4] uses smooth[4], smooth[3], smooth[2], cycle[3], cycle[2]
    # The cycle formula itself needs `alpha` and previous `cycle` values.
    # A safe minimum length is roughly `lookback + a few initial points`. The original script used `len(prices) < lookback`.
    # Let's enforce a minimum related to the formula structure, say 10-15, and warn for less than lookback + a buffer.
    min_safe_len = max(lookback, 15) # Ensure at least 15 bars for some stability, or lookback if larger.

    if len(prices) < min_safe_len:
        logger.warning(f"Insufficient data ({len(prices)}) for ECC calculation (minimum safe: {min_safe_len}). Returning empty arrays.")
        return np.array([]), np.array([])

    # Convert alpha to float for numpy calculation
    alpha_float = float(alpha)
    alpha_sq = (1 - 0.5 * alpha_float) ** 2
    alpha_minus_1 = 2 * (1 - alpha_float)
    alpha_minus_2 = (1 - alpha_float) ** 2


    # Initialize arrays (use float type)
    smooth = np.zeros(len(prices), dtype=float)
    cycle = np.zeros(len(prices), dtype=float)
    trigger = np.zeros(len(prices), dtype=float)

    # Smooth the price series (from original script logic)
    # This smoothing uses a fixed 4-bar window (i, i-1, i-2, i-3)
    # Loop starts from index 3 to ensure prices[i-3] is valid (0)
    for i in range(3, len(prices)):
        smooth[i] = (prices[i] + 2 * prices[i-1] + 2 * prices[i-2] + prices[i-3]) / 6

    # Calculate Cyber Cycle (from original script logic)
    # Loop starts from index 4 to ensure smooth[i-2] and cycle[i-2] are valid
    for i in range(4, len(prices)):
        # Use the pre-calculated alpha terms
        cycle[i] = alpha_sq * (smooth[i] - 2 * smooth[i-1] + smooth[i-2]) + alpha_minus_1 * cycle[i-1] - alpha_minus_2 * cycle[i-2]
        trigger[i] = cycle[i-1]  # Trigger is 1-period lag of cycle

    # Return the last 'lookback' values
    # Need to check if we have enough calculated values before slicing
    if len(cycle) < lookback:
         logger.warning(f"Not enough ECC values calculated ({len(cycle)}) for requested lookback ({lookback}). Returning all calculated values.")
         return cycle, trigger
    else:
         return cycle[-lookback:], trigger[-lookback:]


def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Decimal]]:
    """Calculate technical indicators, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    if df is None or df.empty:
        logger.error(Fore.RED + "Cannot calculate indicators on missing or empty DataFrame.")
        return None
    try:
        # Ensure data is float for TA-Lib / Pandas calculations, convert to Decimal at the end
        # Defensive: Make a copy to avoid modifying the original DataFrame if it's used elsewhere
        df_calc = df.copy()
        # Ensure necessary columns exist before accessing
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df_calc.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_calc.columns]
            logger.error(f"{Fore.RED}DataFrame is missing required columns for indicator calculation: {missing}")
            return None

        close = df_calc["close"].astype(float)
        high = df_calc["high"].astype(float)
        low = df_calc["low"].astype(float)
        prices_np = close.values # Convert to numpy array for ECC

        # --- Check Data Length Requirements ---
        # ATR needs `period + 1` data points (for the first TR calculation involving previous close).
        min_required_atr = CONFIG.atr_period + 1
        if len(df_calc) < min_required_atr:
            logger.warning(f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for ATR calculation (minimum required: {min_required_atr}). ATR will be NaN.")
            atr_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)  # Not enough data for ATR
        else:
            # Average True Range (ATR) - Wilder's smoothing matches TradingView standard
            tr_df = pd.DataFrame(index=df_calc.index)
            tr_df["hl"] = high - low
            tr_df["hc"] = (high - close.shift(1)).abs() # Shift by 1 for previous close
            tr_df["lc"] = (low - close.shift(1)).abs()
            tr = tr_df[["hl", "hc", "lc"]].max(axis=1)
            # Use ewm with alpha = 1/period for Wilder's smoothing, adjust=False
            atr_series = tr.ewm(alpha=1/float(CONFIG.atr_period), adjust=False).mean()


        # --- Calculate Strategy Specific Indicators ---
        ecc_cycle_series = np.array([])
        ecc_trigger_series = np.array([])
        fast_ema_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        slow_ema_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        trend_ema_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        stoch_k = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        stoch_d = pd.Series([np.nan] * len(df_calc), index=df_calc.index)

        if CONFIG.strategy == 'ecc_scalp':
            # ECC calculation requires numpy array of prices
            ecc_cycle_np, ecc_trigger_np = calculate_ehlers_cyber_cycle(prices_np, CONFIG.ecc_alpha, CONFIG.ecc_lookback)
            # Convert numpy arrays back to pandas Series for consistent handling, aligning with the end of the DataFrame
            if len(ecc_cycle_np) > 0:
                 ecc_cycle_series = pd.Series(ecc_cycle_np, index=df_calc.index[-len(ecc_cycle_np):])
            if len(ecc_trigger_np) > 0:
                 ecc_trigger_series = pd.Series(ecc_trigger_np, index=df_calc.index[-len(ecc_trigger_np):])
            # For ECC strategy, we still calculate trend EMA for the filter
            min_required_trend_ema = CONFIG.trend_ema_period
            if len(df_calc) < min_required_trend_ema:
                 logger.warning(f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for Trend EMA calculation (minimum required: {min_required_trend_ema}). Trend EMA will be NaN.")
            else:
                 # Use adjust=False for EWMA to match standard EMA calculation
                 trend_ema_series = close.ewm(span=CONFIG.trend_ema_period, adjust=False).mean()


        elif CONFIG.strategy == 'ema_stoch':
            # EMA calculation
            min_required_ema = max(CONFIG.ema_fast_period, CONFIG.ema_slow_period)
            if len(df_calc) < min_required_ema:
                logger.warning(f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for Fast/Slow EMA calculation (minimum required: {min_required_ema}). EMAs will be NaN.")
            else:
                 fast_ema_series = close.ewm(span=CONFIG.ema_fast_period, adjust=False).mean()
                 slow_ema_series = close.ewm(span=CONFIG.ema_slow_period, adjust=False).mean()

            # Trend EMA calculation
            min_required_trend_ema = CONFIG.trend_ema_period
            if len(df_calc) < min_required_trend_ema:
                 logger.warning(f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for Trend EMA calculation (minimum required: {min_required_trend_ema}). Trend EMA will be NaN.")
            else:
                 trend_ema_series = close.ewm(span=CONFIG.trend_ema_period, adjust=False).mean()

            # Stochastic Oscillator %K and %D
            # Ensure enough data for rolling window calculation
            min_required_stoch = CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 2
            if len(df_calc) < min_required_stoch:
                 logger.warning(f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for Stochastic calculation (minimum required: {min_required_stoch}). Stoch will be NaN.")
            elif len(df_calc) >= CONFIG.stoch_period: # Period check for %K base
                low_min = low.rolling(window=CONFIG.stoch_period).min()
                high_max = high.rolling(window=CONFIG.stoch_period).max()
                # Add epsilon to prevent division by zero if high == low over the period
                # Use float epsilon for float calculation
                stoch_k_raw = 100 * (close - low_min) / (high_max - low_min + 1e-9)
                # Ensure enough data for smoothing windows
                if len(df_calc) >= CONFIG.stoch_period + CONFIG.stoch_smooth_k - 1: # Smooth K check
                     stoch_k = stoch_k_raw.rolling(window=CONFIG.stoch_smooth_k).mean()
                     if len(df_calc) >= CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 2: # Smooth D check
                          stoch_d = stoch_k.rolling(window=CONFIG.stoch_smooth_d).mean()
                     else:
                          logger.warning(f"{Fore.YELLOW}Not enough data for Stochastic %D calculation.")
                 else:
                      logger.warning(f"{Fore.YELLOW}Not enough data for Stochastic %K smoothing.")
            else:
                 logger.warning(f"{Fore.YELLOW}Not enough data for Stochastic period calculation.")


        # --- Extract Latest Values & Convert to Decimal ---
        # Define quantizers for consistent decimal places (adjust as needed)
        # These are for *internal* Decimal representation, not API formatting.
        # Use enough precision to avoid rounding errors before API formatting.
        price_quantizer = Decimal("1E-8")  # 8 decimal places for price-like values
        percent_quantizer = Decimal("1E-2")  # 2 decimal places for Stoch, ECC
        atr_quantizer = Decimal("1E-8")  # 8 decimal places for ATR

        # Helper to safely get latest non-NaN value, convert to Decimal, and handle errors
        def get_latest_decimal(series: Union[pd.Series, np.ndarray], quantizer: Decimal, name: str, default_val: Decimal = Decimal("NaN")) -> Decimal:
            # Handle numpy arrays passed directly
            if isinstance(series, np.ndarray):
                 # Convert to Series for consistent handling of index/dropna/iloc
                 if len(series) > 0:
                      series = pd.Series(series)
                 else:
                      # logger.warning(f"Indicator series '{name}' (numpy) is empty.")
                      return default_val

            if series.empty or series.isna().all():
                # logger.warning(f"Indicator series '{name}' is empty or all NaN.") # Logged below if crucial
                return default_val
            # Get the last valid (non-NaN) value
            latest_valid_val = series.dropna().iloc[-1] if not series.dropna().empty else None

            if latest_valid_val is None or pd.isna(latest_valid_val):
                # logger.warning(f"Indicator calculation for '{name}' resulted in NaN or only NaNs.") # Logged below if crucial
                return default_val
            try:
                # Convert via string for precision, then quantize
                return Decimal(str(latest_valid_val)).quantize(quantizer, rounding=ROUND_DOWN)  # Use ROUND_DOWN for consistency
            except (InvalidOperation, TypeError) as e:
                logger.error(f"Could not convert indicator '{name}' value {latest_valid_val} to Decimal: {e}. Returning default.")
                return default_val

        indicators_out = {
            "atr": get_latest_decimal(atr_series, atr_quantizer, "atr", default_val=Decimal("0.0")),  # Default zero
            "atr_period": CONFIG.atr_period # Store period for display
        }

        # Add strategy specific indicators
        if CONFIG.strategy == 'ecc_scalp':
             indicators_out["ecc_cycle"] = get_latest_decimal(ecc_cycle_series, percent_quantizer, "ecc_cycle")
             indicators_out["ecc_trigger"] = get_latest_decimal(ecc_trigger_series, percent_quantizer, "ecc_trigger")
             indicators_out["ecc_alpha"] = CONFIG.ecc_alpha # Store params for display
             indicators_out["ecc_lookback"] = CONFIG.ecc_lookback # Store params for display
             indicators_out["trend_ema"] = get_latest_decimal(trend_ema_series, price_quantizer, "trend_ema") # Trend filter EMA

        elif CONFIG.strategy == 'ema_stoch':
             indicators_out["fast_ema"] = get_latest_decimal(fast_ema_series, price_quantizer, "fast_ema")
             indicators_out["slow_ema"] = get_latest_decimal(slow_ema_series, price_quantizer, "slow_ema")
             indicators_out["trend_ema"] = get_latest_decimal(trend_ema_series, price_quantizer, "trend_ema")
             indicators_out["stoch_k"] = get_latest_decimal(stoch_k, percent_quantizer, "stoch_k", default_val=Decimal("50.00"))  # Default neutral
             indicators_out["stoch_d"] = get_latest_decimal(stoch_d, percent_quantizer, "stoch_d", default_val=Decimal("50.00"))  # Default neutral
             indicators_out["ema_fast_period"] = CONFIG.ema_fast_period # Store params for display
             indicators_out["ema_slow_period"] = CONFIG.ema_slow_period # Store params for display
             indicators_out["stoch_period"] = CONFIG.stoch_period # Store params for display
             indicators_out["stoch_smooth_k"] = CONFIG.stoch_smooth_k # Store params for display
             indicators_out["stoch_smooth_d"] = CONFIG.stoch_smooth_d # Store params for display

        # Check if any crucial indicator calculation failed (returned NaN default)
        critical_indicators = ['atr', 'trend_ema'] # ATR and Trend EMA are used by both strategies (Trend EMA for filter)
        if CONFIG.strategy == 'ecc_scalp':
             critical_indicators.extend(['ecc_cycle', 'ecc_trigger'])
        elif CONFIG.strategy == 'ema_stoch':
             critical_indicators.extend(['fast_ema', 'slow_ema', 'stoch_k', 'stoch_d'])

        failed_indicators = [key for key in critical_indicators if indicators_out.get(key, Decimal('NaN')).is_nan()]

        if failed_indicators:
            logger.warning(f"{Fore.YELLOW}One or more critical indicators failed to calculate (NaN): {', '.join(failed_indicators)}. Trade signals/logic might be unreliable.")
            # Do NOT return None here. Return the dict with NaNs. Signal generation and trade logic will handle the NaNs.

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

    # Initialize with Decimal zero/NaN for clarity
    pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN")}
    }

    positions_data = None
    try:
        # fetch_with_retries handles category param automatically
        # Note: Bybit V5 fetch_positions might return multiple entries per symbol (e.g., isolated/cross, or different sides).
        # We assume one-way mode and sum up quantities/use average price if necessary, or just process the first entry found for each side.
        # For simplicity, this code assumes one-way mode and picks the first long/short entry with non-zero size.
        # fetch_positions might return an empty list [], which is valid.
        # Use exchange's internal ID for symbol if available
        symbol_id = MARKET_INFO.get('id') if MARKET_INFO else symbol
        positions_data = fetch_with_retries(EXCHANGE.fetch_positions, symbols=[symbol_id])
    except Exception as e:
        # Handle potential exceptions raised by fetch_with_retries itself (e.g., AuthenticationError, Non-retryable ExchangeError)
        logger.error(Fore.RED + f"Unhandled exception during fetch_positions call via fetch_with_retries: {e}", exc_info=True)
        return None  # Indicate failure

    if positions_data is None:
        # fetch_with_retries already logged the failure reason
        logger.error(Fore.RED + f"Failed to fetch positions for {symbol}.")
        return None  # Indicate failure

    if not isinstance(positions_data, list):
        logger.error(f"Unexpected data type received from fetch_positions: {type(positions_data)}. Expected list. Data: {str(positions_data)[:200]}")
        return None

    if not positions_data:
        logger.info(Fore.BLUE + f"No open positions reported by exchange for {symbol}.")
        return pos_dict  # Return the initialized zero dictionary

    # Process the fetched positions - find the primary long/short position for the symbol
    # In one-way mode, there should be at most one long and one short position per symbol.
    # Aggregate quantities if multiple entries exist for the same side (e.g., if both isolated/cross were returned, though unlikely with V5 category filter).
    aggregated_positions: Dict[str, Dict[str, Decimal]] = {
        "long": {"qty": Decimal("0.0"), "entry_price_sum_qty": Decimal("0.0")},  # Use weighted average sum
        "short": {"qty": Decimal("0.0"), "entry_price_sum_qty": Decimal("0.0")}
    }
    # Store other details from the *first* significant position found for each side
    other_pos_details: Dict[str, Dict[str, Optional[Decimal]]] = {
        "long": {"liq_price": Decimal("NaN"), "pnl": Decimal("NaN")},
        "short": {"liq_price": Decimal("NaN"), "pnl": Decimal("NaN")}
    }
    first_long_found = False
    first_short_found = False

    for pos in positions_data:
        # Ensure pos is a dictionary
        if not isinstance(pos, dict):
            logger.warning(f"Skipping non-dictionary item in positions data: {pos}")
            continue

        pos_symbol = pos.get('symbol')
        # Compare using CCXT symbol format
        if pos_symbol != symbol:
            logger.debug(f"Ignoring position data for different symbol: {pos_symbol}")
            continue

        # Use info dictionary for safer access to raw exchange data if needed
        pos_info = pos.get('info', {})
        if not isinstance(pos_info, dict):  # Ensure info is a dict
            pos_info = {}

        # Determine side ('long' or 'short') - check unified field first, then 'info'
        side = pos.get("side")  # Unified field
        if side not in ["long", "short"]:
            # Fallback for Bybit V5 'info' field if unified 'side' is missing/invalid
            side_raw = pos_info.get("side", "").lower()  # e.g., "Buy" or "Sell"
            if side_raw == "buy": side = "long"
            elif side_raw == "sell": side = "short"
            else:
                logger.warning(f"Could not determine side for position: Info={str(pos_info)[:100]}. Skipping.")
                continue

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

        # --- Convert to Decimal and Aggregate/Store ---
        if side in aggregated_positions and contracts_str is not None:
            try:
                # Convert via string for precision
                contracts = Decimal(str(contracts_str))

                # Use epsilon check for effectively zero positions
                if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                    logger.debug(f"Ignoring effectively zero size {side} position entry for {symbol} (Qty: {contracts.normalize()}).")
                    continue  # Skip processing this entry

                # Aggregate quantity
                aggregated_positions[side]["qty"] += contracts

                # Aggregate for weighted average entry price if entry price is available
                entry_price = Decimal(str(entry_price_str)) if entry_price_str is not None and entry_price_str != '' else Decimal("NaN")
                if not entry_price.is_nan():
                    aggregated_positions[side]["entry_price_sum_qty"] += entry_price * contracts.copy_abs()  # Use absolute qty for sum

                # Store other details from the *first* significant entry found for this side
                if (side == "long" and not first_long_found) or (side == "short" and not first_short_found):
                    liq_price = Decimal(str(liq_price_str)) if liq_price_str is not None and liq_price_str != '' else Decimal("NaN")
                    pnl = Decimal(str(pnl_str)) if pnl_str is not None and pnl_str != '' else Decimal("NaN")
                    other_pos_details[side]["liq_price"] = liq_price
                    other_pos_details[side]["pnl"] = pnl
                    if side == "long": first_long_found = True
                    else: first_short_found = True

                    logger.debug(f"Processing first significant {side.upper()} entry: Qty={contracts.normalize()}, Entry={entry_price}, Liq={liq_price}, PnL={pnl}")

            except (InvalidOperation, TypeError) as e:
                logger.error(f"Could not parse position data for {side} side from entry (Qty:'{contracts_str}', Entry:'{entry_price_str}', Liq:'{liq_price_str}', Pnl:'{pnl_str}'). Error: {e}")
                # Do not continue here, this specific position entry is problematic.
                # The pos_dict[side] will retain its default NaN/0 values.
                continue
        elif side not in aggregated_positions:
            logger.warning(f"Position data found for unknown side '{side}'. Skipping.")

    # --- Finalize Position Dictionary ---
    final_pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN")}
    }

    for side in ["long", "short"]:
        total_qty = aggregated_positions[side]["qty"]
        weighted_sum = aggregated_positions[side]["entry_price_sum_qty"]

        if total_qty.copy_abs() >= CONFIG.position_qty_epsilon:
            final_pos_dict[side]["qty"] = total_qty
            # Calculate weighted average entry price
            if weighted_sum > Decimal("0") and total_qty.copy_abs() > Decimal("0"):
                final_pos_dict[side]["entry_price"] = weighted_sum / total_qty.copy_abs()  # Use absolute qty for division
            else:
                # If sum or total qty is zero/negative (unexpected), entry price is NaN
                final_pos_dict[side]["entry_price"] = Decimal("NaN")

            # Use the stored other details from the first significant position
            final_pos_dict[side]["liq_price"] = other_pos_details[side]["liq_price"]
            final_pos_dict[side]["pnl"] = other_pos_details[side]["pnl"]

            # Log with formatted decimals (for display)
            entry_log = f"{final_pos_dict[side]['entry_price']:.4f}" if not final_pos_dict[side]['entry_price'].is_nan() else "N/A"
            liq_log = f"{final_pos_dict[side]['liq_price']:.4f}" if not final_pos_dict[side]['liq_price'].is_nan() else "N/A"
            pnl_log = f"{final_pos_dict[side]['pnl']:+.4f}" if not final_pos_dict[side]['pnl'].is_nan() else "N/A"
            logger.info(Fore.YELLOW + f"Aggregated active {side.upper()} position: Qty={total_qty.normalize()}, Entry={entry_log}, Liq≈{liq_log}, PnL≈{pnl_log}")

        else:
            logger.info(Fore.BLUE + f"No significant {side.upper()} position found for {symbol}.")

    if final_pos_dict["long"]["qty"].copy_abs() > CONFIG.position_qty_epsilon and \
       final_pos_dict["short"]["qty"].copy_abs() > CONFIG.position_qty_epsilon:
        logger.warning(Fore.YELLOW + f"Both LONG ({final_pos_dict['long']['qty'].normalize()}) and SHORT ({final_pos_dict['short']['qty'].normalize()}) positions found for {symbol}. Pyrmethus assumes one-way mode and will manage these independently. Please ensure your exchange account is configured for one-way trading if this is unexpected.")

    logger.info(Fore.GREEN + "Position spirits consulted.")
    return final_pos_dict


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
        logger.error(Fore.RED + f"Unhandled exception during fetch_balance call via fetch_with_retries: {e}", exc_info=True)
        return None, None

    if balance_data is None:
        # fetch_with_retries already logged the failure
        logger.error(Fore.RED + f"Failed to fetch balance after retries. Cannot assess risk capital.")
        return None, None

    # --- Parse Balance Data ---
    # Initialize with NaN Decimals to indicate failure to find/parse
    free_balance = Decimal("NaN")
    total_balance = Decimal("NaN")  # Represents Equity for futures/swaps

    try:
        # Attempt to parse using standard CCXT structure
        if currency in balance_data and isinstance(balance_data[currency], dict):
            currency_balance = balance_data[currency]
            free_str = currency_balance.get('free')
            total_str = currency_balance.get('total')  # 'total' usually represents equity in futures

            if free_str is not None: free_balance = Decimal(str(free_str))
            if total_str is not None: total_balance = Decimal(str(total_str))

        # Alternative standard CCXT structure (less common for V5)
        elif 'free' in balance_data and isinstance(balance_data['free'], dict) and currency in balance_data['free']:
            free_str = balance_data['free'].get(currency)
            total_str = balance_data.get('total', {}).get(currency)  # Total might still be top-level

            if free_str is not None: free_balance = Decimal(str(free_str))
            if total_str is not None: total_balance = Decimal(str(total_str))

        # Fallback: Check 'info' for exchange-specific structure (Bybit V5 example)
        # This is the most reliable for Bybit V5 Unified Margin/Contract accounts
        # Check if standard parsing yielded NaN for total_balance before trying info fallback
        if total_balance.is_nan() and 'info' in balance_data and isinstance(balance_data['info'], dict):
            info_data = balance_data['info']
            # V5 structure: result -> list -> account objects
            if 'result' in info_data and isinstance(info_data['result'], dict) and \
               'list' in info_data['result'] and isinstance(info_data['result']['list'], list):
                for account in info_data['result']['list']:
                    # Find the account object for the target currency
                    if isinstance(account, dict) and account.get('coin') == currency:
                        # Bybit V5 Unified Margin fields (check docs):
                        # 'walletBalance': Total assets in wallet
                        # 'availableToWithdraw': Amount withdrawable
                        # 'equity': Account equity (often the most relevant for risk calculation in futures)
                        # 'availableToBorrow': Margin specific
                        # 'totalPerpUPL': Unrealized PnL (already included in equity)
                        equity_str = account.get('equity')  # Use equity as 'total' for risk calculation
                        free_str_info = account.get('availableToWithdraw')  # Use availableToWithdraw as 'free'

                        # Overwrite if found in info, as info is often more granular/accurate for V5
                        if free_str_info is not None: free_balance = Decimal(str(free_str_info))
                        if equity_str is not None: total_balance = Decimal(str(equity_str))
                        logger.debug(f"Parsed Bybit V5 info structure for {currency}: Free={free_balance}, Equity={total_balance}")
                        break  # Found the currency account
            else:
                logger.warning("Bybit V5 info structure missing 'result' or 'list'.")

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
        logger.error(Fore.RED + f"Error parsing balance data for {currency}: {e}. Raw keys: {list(balance_data.keys()) if isinstance(balance_data, dict) else 'N/A'}")
        logger.debug(f"Raw balance data: {balance_data}")
        return None, None  # Indicate parsing failure
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected shadow encountered querying vault: {e}", exc_info=True)
        return None, None

# --- PnL Fetching (from Block 1, adapted) ---
def fetch_trade_history_wrapper(symbol: Optional[str] = None, lookback_days: int = CONFIG.pnl_lookback_days) -> Optional[List[Dict]]:
    """
    Fetches trade execution history for realized PnL calculation using retry wrapper.
    Handles pagination. Returns Decimals for numeric fields where possible.
    """
    global EXCHANGE
    logger.debug(f"Fetching trade history for symbol {symbol or 'All'} for last {lookback_days} days...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching trade history.")
        return None

    # Use exchange's internal ID for symbol if available
    symbol_id = MARKET_INFO.get('id') if MARKET_INFO and symbol == CONFIG.symbol else symbol # Only use market_info if fetching the main symbol

    params = {
        "category": CONFIG.market_type,
        "startTime": str(int((time.time() - lookback_days * 86400) * 1000)),
        "limit": "100" # Max limit allowed by Bybit V5 execution list endpoint
    }
    if symbol_id:
        params["symbol"] = symbol_id

    trades = []
    next_page_cursor = None

    while True:
        if next_page_cursor:
             params["cursor"] = next_page_cursor

        # Use fetch_with_retries for the API call
        response = None
        try:
            response = fetch_with_retries(EXCHANGE.private_get_execution_list, params=params) # Use implicit method
        except Exception as e:
            logger.error(Fore.RED + f"Unhandled exception during fetch_trade_history call via fetch_with_retries: {e}", exc_info=True)
            return None # Indicate failure

        if response is None:
            # fetch_with_retries already logged the failure
            logger.error(Fore.RED + f"Failed to fetch trade history after retries for symbol {symbol or 'All'}.")
            return None

        # Check Bybit V5 response structure
        if isinstance(response.get('info'), dict) and response['info'].get('retCode') == 0:
            result = response['info'].get('result', {})
            trade_list = result.get('list', [])
            next_page_cursor = result.get('nextPageCursor')

            # Process and add trades to the list, converting numeric fields to Decimal
            for trade in trade_list:
                 processed_trade = {}
                 # Copy essential fields, converting numeric ones
                 for key in ['symbol', 'side', 'orderId', 'execTime', 'execType', 'price', 'qty', 'execFee', 'feeRate', 'closedPnl', 'realisedPnl']:
                      raw_value = trade.get(key)
                      if raw_value is not None:
                           if key in ['price', 'qty', 'execFee', 'feeRate', 'closedPnl', 'realisedPnl']:
                                try:
                                     processed_trade[key] = Decimal(str(raw_value))
                                except InvalidOperation:
                                     logger.warning(f"Could not parse trade field '{key}' with value '{raw_value}' to Decimal for trade {trade.get('orderId', 'N/A')}. Storing as string.")
                                     processed_trade[key] = str(raw_value) # Store as string if Decimal conversion fails
                           else:
                                processed_trade[key] = raw_value # Keep other fields as they are
                 trades.append(processed_trade)

            if not next_page_cursor:
                break # No more pages

            logger.debug(f"Fetched {len(trade_list)} trades, next cursor: {next_page_cursor[:10] if next_page_cursor else 'None'}. Total fetched so far: {len(trades)}")
            # Optional: add a small delay between paginated calls if rate limits are hit
            time.sleep(0.1)

        else:
            error_msg = response['info'].get('retMsg', 'Unknown error') if isinstance(response.get('info'), dict) else str(response)
            logger.error(Fore.RED + f"Failed to fetch trade history page. Exchange message: {error_msg}. Stopping pagination.")
            return None # Indicate failure

    logger.debug(f"Finished fetching trade history. Total trades: {len(trades)}")
    return trades


def fetch_pnl_wrapper(symbol: Optional[str] = None) -> Optional[Dict]:
    """
    Fetches unrealized PnL from positions and realized PnL from trade history.
    Returns Decimals for PnL values.
    """
    logger.info(Fore.CYAN + f"# Consulting PnL scrolls for {symbol or 'All Symbols'}...")

    result = {"unrealized_pnl": [], "realized_pnl": [], "total_unrealized": Decimal("0.0"), "total_realized": Decimal("0.0")}

    # Fetch Unrealized PnL from Positions
    positions_data = get_current_position(symbol or CONFIG.symbol) # Use dedicated function
    if positions_data is None:
        logger.error(Fore.RED + "Failed to fetch positions for Unrealized PnL.")
        # Continue to fetch realized PnL, but indicate partial failure if needed?
        # For simplicity, return None if positions fetch fails, as it's a core component.
        return None

    # get_current_position already aggregates and converts to Decimal, and logs summary.
    # Just extract the relevant parts if needed, but the function logs what's needed.
    # The main loop will use get_current_position anyway, so maybe this wrapper just needs realized PnL?
    # Let's refactor: fetch_pnl_wrapper only fetches *realized* PnL history. Unrealized PnL is part of `get_current_position`.

    # Fetch Realized PnL from Trade History
    # Fetch history for the specified symbol or all if None
    trades = fetch_trade_history_wrapper(symbol, CONFIG.pnl_lookback_days)
    if trades is None:
        logger.error(Fore.RED + "Failed to fetch trade history for Realized PnL.")
        # Return None if trade history fetch fails, as this wrapper is for PnL summary.
        return None

    # Aggregate realized PnL
    for trade in trades:
        try:
            # Use the Decimal values already parsed by fetch_trade_history_wrapper
            realized_pnl = trade.get('closedPnl', trade.get('realisedPnl', Decimal('0.0'))) # Check both potential keys
            trade_symbol = trade.get('symbol')
            trade_side = trade.get('side')
            trade_exec_time = trade.get('execTime') # Timestamp string/int

            # Ensure realized_pnl is Decimal and not NaN
            if isinstance(realized_pnl, Decimal) and not realized_pnl.is_nan():
                 result["realized_pnl"].append({
                     "symbol": trade_symbol,
                     "side": trade_side,
                     "exec_time": trade_exec_time,
                     "realized_pnl": realized_pnl
                 })
                 result["total_realized"] += realized_pnl
            else:
                 logger.warning(f"Skipping trade with invalid PnL value for {trade_symbol}: {realized_pnl}")

        except Exception as e:
            logger.error(f"Error processing trade for realized PnL {trade.get('symbol', 'Unknown')}: {e}")


    logger.info(f"Realized PnL Summary (Last {CONFIG.pnl_lookback_days} Days): Total Realized = {result['total_realized']:+.4f}")
    # Optionally log individual realized trades if desired, but can be noisy
    # for trade in result["realized_pnl"]:
    #      logger.info(f"  Realized: {trade['symbol']} [{trade['side']}] at {trade['exec_time']}: {trade['realized_pnl']:+.4f}")

    # Note: This function now only returns *realized* PnL details.
    # Unrealized PnL is obtained separately via `get_current_position`.
    # Let's adjust the return structure to reflect this, or combine them here.
    # It's probably better to keep them separate as positions are fetched every cycle anyway.
    # This function will just summarize realized PnL.

    # Let's just return the total realized PnL for the lookback period.
    return {"total_realized": result["total_realized"]}


def check_order_status(order_id: str, symbol: str, timeout: int = CONFIG.order_check_timeout_seconds) -> Optional[Dict]:
    """Checks order status with retries and timeout. Returns the final order dict or None."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"Verifying final status of order {order_id} for {symbol} (Timeout: {timeout}s)...")
    if EXCHANGE is None:
        logger.error("Exchange object not available for checking order status.")
        return None
    if order_id is None:
        logger.warning("Received None order_id to check status. Skipping check.")
        return None  # Cannot check status for a None ID

    start_time = time.time()
    last_status = 'unknown'
    attempt = 0
    check_interval = CONFIG.order_check_delay_seconds  # Start with configured delay

    while time.time() - start_time < timeout:
        attempt += 1
        logger.debug(f"Checking order {order_id}, attempt {attempt}...")
        order_status_data = None
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            # Category param should be handled automatically by fetch_with_retries
            # Bybit V5 fetch_order requires category and symbol (exchange ID)
            fetch_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO.get('id')}  # Use MARKET_INFO['id']
            order_status_data = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=fetch_params)

            if order_status_data and isinstance(order_status_data, dict):
                last_status = order_status_data.get('status', 'unknown')
                # Use Decimal for precision comparison
                filled_qty_raw = order_status_data.get('filled', 0.0)
                try:
                    filled_qty = Decimal(str(filled_qty_raw))
                except (InvalidOperation, TypeError):
                    logger.error(f"Could not parse filled quantity '{filled_qty_raw}' for order {order_id}. Assuming 0.")
                    filled_qty = Decimal('0')

                logger.info(f"Order {order_id} status check: {last_status}, Filled: {filled_qty.normalize()}")  # Use normalize for cleaner log

                # Check for terminal states (fully filled, canceled, rejected, expired)
                # 'closed' usually means fully filled for market/limit orders on Bybit.
                if last_status in ['closed', 'canceled', 'rejected', 'expired']:
                    logger.info(f"Order {order_id} reached terminal state: {last_status}.")
                    return order_status_data  # Return the final order dict
                # If 'open' but fully filled (can happen briefly), treat as terminal 'closed'
                # Check remaining amount using epsilon
                remaining_qty_raw = order_status_data.get('remaining', 0.0)
                try:
                    remaining_qty = Decimal(str(remaining_qty_raw))
                except (InvalidOperation, TypeError):
                    logger.error(f"Could not parse remaining quantity '{remaining_qty_raw}' for order {order_id}. Cannot perform fill check.")
                    remaining_qty = Decimal('NaN')  # Indicate parsing failure

                # Check if filled >= original amount using epsilon (more reliable than remaining == 0)
                original_amount_raw = order_status_data.get('amount', 0.0)
                try:
                    original_amount = Decimal(str(original_amount_raw))
                except (InvalidOperation, TypeError):
                    logger.error(f"Could not parse original amount '{original_amount_raw}' for order {order_id}. Cannot perform fill check.")
                    original_amount = Decimal('NaN')  # Indicate parsing failure

                # Consider order fully filled if filled amount is very close to original amount
                # Use a tolerance based on configured epsilon or a small fraction of original amount
                fill_tolerance = max(CONFIG.position_qty_epsilon, original_amount.copy_abs() * Decimal('1E-6')) if not original_amount.is_nan() else CONFIG.position_qty_epsilon

                # Check if filled quantity is effectively equal to the original amount
                if not original_amount.is_nan() and (original_amount - filled_qty).copy_abs() < fill_tolerance:
                    if last_status == 'open':
                        logger.info(f"Order {order_id} is 'open' but appears fully filled ({filled_qty.normalize()}/{original_amount.normalize()}). Treating as 'closed'.")
                        order_status_data['status'] = 'closed'  # Update status locally for clarity
                        return order_status_data
                    elif last_status in ['partially_filled']:
                        logger.info(f"Order {order_id} is '{last_status}' but appears fully filled ({filled_qty.normalize()}/{original_amount.normalize()}). Treating as 'closed'.")
                        order_status_data['status'] = 'closed'  # Update status locally for clarity
                        return order_status_data

            else:
                # fetch_with_retries failed or returned unexpected data
                # Error logged within fetch_with_retries, just note it here
                logger.warning(f"fetch_order call failed or returned invalid data for {order_id}. Continuing check loop.")
                # Continue the loop to retry check_order_status itself

        except ccxt.OrderNotFound:
            # Order is definitively not found. This is a terminal state indicating it never existed or was fully purged after fill/cancel.
            # For market orders, this often means it filled and was purged quickly.
            logger.info(f"Order {order_id} confirmed NOT FOUND by exchange. Assuming filled/cancelled and purged.")
            # Cannot get fill details, but assume it's gone. For market orders, this often means success.
            # It's better to return None to indicate we couldn't verify the *final* state definitively,
            # unless we can infer fill from context (e.g., it was a market order).
            # For robustness, returning None when status cannot be confirmed is safer.
            return None  # Explicitly indicate not found / unable to verify

        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
            # Critical non-retryable errors
            logger.critical(Fore.RED + Style.BRIGHT + f"Authentication/Permission error during order status check for {order_id}: {e}. Halting.")
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
            check_interval = min(check_interval * 1.2, 5)  # Slightly increase interval up to 5s, max 5s
        else:
            break  # Exit loop if next sleep would exceed timeout

    # --- Timeout Reached ---
    logger.error(Fore.RED + f"Timed out checking status for order {order_id} after {timeout} seconds. Last known status: {last_status}.")
    # Attempt one final fetch outside the loop to get the very last state if possible
    final_check_status = None
    try:
        logger.info(f"Performing final status check for order {order_id} after timeout...")
        # Use fetch_with_retries for the final check too
        fetch_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO.get('id')}
        final_check_status = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=fetch_params)

        if final_check_status and isinstance(final_check_status, dict):
            final_status = final_check_status.get('status', 'unknown')
            final_filled = final_check_status.get('filled', 'N/A')
            logger.info(f"Final status after timeout: {final_status}, Filled: {final_filled}")
            # Return this final status even if timed out earlier
            return final_check_status
        else:
            logger.error(f"Final status check for order {order_id} also failed or returned invalid data.")
            # If the final check also fails, we cannot confirm status.
            return None  # Indicate persistent failure to get status
    except ccxt.OrderNotFound:
        logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND on final check after timeout.")
        return None  # Still cannot confirm final state details
    except Exception as e:
        logger.error(f"Error during final status check for order {order_id}: {e}", exc_info=True)
        return None  # Indicate failure


def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    trade_action = f"{side.upper()} Market Entry"
    logger.trade(Style.BRIGHT + f"Attempting {trade_action} for {symbol}...")

    global MARKET_INFO, EXCHANGE, order_tracker
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(Fore.RED + f"{trade_action} failed: Market info or Exchange not available.")
        return False

    # --- Pre-computation & Validation ---
    quote_currency = MARKET_INFO.get('settle', 'USDT')  # Use settle currency (e.g., USDT)
    free_balance, total_equity = get_balance(quote_currency)  # Fetch balance using the function
    if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
        logger.error(Fore.RED + f"{trade_action} failed: Invalid, NaN, or zero account equity ({total_equity}). Cannot calculate risk capital.")
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
        logger.error(Fore.RED + f"{trade_action} failed: Cannot fetch current ticker price for sizing/SL calculation. Ticker data: {ticker_data}")
        # fetch_with_retries should have logged details if it failed
        return False

    try:
        # Use 'last' price as current price estimate, convert to Decimal
        price = Decimal(str(ticker_data["last"]))
        if price <= Decimal(0):
            logger.error(Fore.RED + f"{trade_action} failed: Fetched current price ({price}) is zero or negative. Aborting.")
            return False
        logger.debug(f"Current ticker price: {price:.8f} {quote_currency}")  # Log with high precision for debug

        # --- Calculate Stop Loss Price ---
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if sl_distance_points <= Decimal("0"):  # Use Decimal zero
            logger.error(f"{Fore.RED}{trade_action} failed: Stop distance calculation resulted in zero or negative value ({sl_distance_points}). Check ATR ({atr:.6f}) and multiplier ({CONFIG.sl_atr_multiplier}).")
            return False

        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else:  # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations/API call
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        # Convert back to Decimal *after* formatting for consistent internal representation
        try:
            sl_price = Decimal(sl_price_formatted_str)
        except InvalidOperation:
            logger.error(Fore.RED + f"{trade_action} failed: Formatted SL price '{sl_price_formatted_str}' is invalid Decimal. Aborting.")
            return False

        logger.debug(f"ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier}, SL Distance Points: {sl_distance_points:.6f}")
        logger.debug(f"Raw SL Price: {sl_price_raw:.8f}, Formatted SL Price for API: {sl_price_formatted_str} (Decimal: {sl_price})")

        # Sanity check SL placement relative to current price
        # Use a small multiple of price tick size for tolerance if available, else a tiny Decimal
        try:
            price_precision_info = MARKET_INFO['precision'].get('price')
            # If precision is number of decimals (int)
            if isinstance(price_precision_info, int):
                price_tick_size = Decimal(1) / (Decimal(10) ** price_precision_info)
            # If precision is tick size (string or Decimal)
            elif isinstance(price_precision_info, (str, Decimal)):
                price_tick_size = Decimal(str(price_precision_info))
            else:
                price_tick_size = Decimal("1E-8")  # Fallback tiny Decimal
        except Exception:
            price_tick_size = Decimal("1E-8")  # Fallback tiny Decimal

        tolerance_ticks = price_tick_size * Decimal('5')  # Allow a few ticks tolerance

        if side == "buy" and sl_price >= price - tolerance_ticks:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated SL price ({sl_price}) is too close to or above current price ({price}) [Tolerance: {tolerance_ticks}]. Check ATR/multiplier or market precision. Aborting.")
            return False
        if side == "sell" and sl_price <= price + tolerance_ticks:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated SL price ({sl_price}) is too close to or below current price ({price}) [Tolerance: {tolerance_ticks}]. Check ATR/multiplier or market precision. Aborting.")
            return False

        # --- Calculate Position Size ---
        risk_amount_quote = total_equity * risk_percentage
        # Stop distance in quote currency (use absolute difference, ensure Decimals)
        stop_distance_quote = (price - sl_price).copy_abs()

        if stop_distance_quote <= Decimal("0"):
            logger.error(Fore.RED + f"{trade_action} failed: Stop distance in quote currency is zero or negative ({stop_distance_quote}). Check ATR, multiplier, or market precision. Cannot calculate size.")
            return False

        # Calculate quantity based on contract size and linear/inverse type
        # Ensure contract_size is a Decimal
        contract_size_dec = Decimal(str(MARKET_INFO.get('contractSize', '1')))  # Ensure Decimal

        qty_raw = Decimal('0')

        # --- Sizing Logic ---
        # Bybit uses size in Base currency for Linear (e.g., BTC for BTC/USDT)
        # Bybit uses size in Contracts (which represent USD value for BTC/USD inverse)
        # Risk Amount (Quote) = (Entry Price (Quote/Base) - SL Price (Quote/Base)) * Qty (Base)
        # Qty (Base) = Risk Amount (Quote) / (Entry Price (Quote/Base) - SL Price (Quote/Base))
        # Qty (Base) = Risk Amount (Quote) / Stop Distance (Quote)
        if CONFIG.market_type == 'linear':
            qty_raw = risk_amount_quote / stop_distance_quote
            logger.debug(f"Linear Sizing: Qty (Base) = {risk_amount_quote:.8f} {quote_currency} / {stop_distance_quote:.8f} {quote_currency} = {qty_raw:.8f}")

        elif CONFIG.market_type == 'inverse':
            # Bybit Inverse (e.g., BTC/USD:BTC): Size is in Contracts. 1 Contract = 1 USD for BTC/USD.
            # Risk Amount (Quote) = (Entry Price (Quote/Base) - SL Price (Quote/Base)) * Qty (Base)
            # Qty (Base) = Qty (Contracts) * Contract Size (Base/Contract)
            # Contract Size (Base/Contract) = 1 / Price (Quote/Base) for BTC/USD
            # Risk (Quote) = (Price - SL Price) * Qty (Contracts) * (1 / Price)
            # Risk (Quote) = Stop Distance (Quote) * Qty (Contracts) / Price (Quote/Base)
            # Qty (Contracts) = Risk (Quote) * Price (Quote/Base) / Stop Distance (Quote)
            if price <= Decimal("0"):
                logger.error(Fore.RED + f"{trade_action} failed: Cannot calculate inverse size with zero or negative price.")
                return False
            qty_raw = (risk_amount_quote * price) / stop_distance_quote
            logger.debug(f"Inverse Sizing (Contract Size = {contract_size_dec} {quote_currency}): Qty (Contracts) = ({risk_amount_quote:.8f} * {price:.8f}) / {stop_distance_quote:.8f} = {qty_raw:.8f}")

        else:
            logger.error(f"{trade_action} failed: Unsupported market type for sizing: {CONFIG.market_type}")
            return False

        # --- Format and Validate Quantity ---
        # Format quantity according to market precision (ROUND_DOWN to be conservative)
        qty_formatted_str = format_amount(symbol, qty_raw, ROUND_DOWN)
        try:
            qty = Decimal(qty_formatted_str)
        except InvalidOperation:
            logger.error(Fore.RED + f"{trade_action} failed: Formatted quantity '{qty_formatted_str}' is invalid Decimal. Aborting.")
            return False

        logger.debug(f"Risk Amount: {risk_amount_quote:.8f} {quote_currency}, Stop Distance: {stop_distance_quote:.8f} {quote_currency}")
        logger.debug(f"Raw Qty: {qty_raw:.12f}, Formatted Qty (Rounded Down): {qty.normalize()}")  # Use normalize for cleaner log

        # Validate Quantity Against Market Limits
        min_qty_str = str(MARKET_INFO['limits']['amount']['min']) if MARKET_INFO['limits']['amount'].get('min') is not None else "0"
        max_qty_str = str(MARKET_INFO['limits']['amount']['max']) if MARKET_INFO['limits']['amount'].get('max') is not None else None
        min_qty = Decimal(min_qty_str)
        # max_qty is infinity if None, otherwise convert to Decimal
        max_qty = Decimal(str(max_qty_str)) if max_qty_str is not None else Decimal('Infinity')

        # Use epsilon for zero check
        if qty < min_qty or qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated quantity ({qty.normalize()}) is zero or below minimum ({min_qty.normalize()}, epsilon {CONFIG.position_qty_epsilon:.1E}). Risk amount ({risk_amount_quote:.4f}), stop distance ({stop_distance_quote:.4f}), or equity might be too small. Cannot place order.")
            return False
        if max_qty != Decimal('Infinity') and qty > max_qty:
            logger.warning(Fore.YELLOW + f"Calculated quantity {qty.normalize()} exceeds maximum {max_qty.normalize()}. Capping order size to {max_qty.normalize()}.")
            qty = max_qty  # Use the Decimal max_qty
            # Re-format capped amount - crucial! Use ROUND_DOWN again.
            qty_formatted_str = format_amount(symbol, qty, ROUND_DOWN)
            try:
                qty = Decimal(qty_formatted_str)
            except InvalidOperation:
                logger.error(Fore.RED + f"{trade_action} failed: Re-formatted capped quantity '{qty_formatted_str}' is invalid Decimal. Aborting.")
                return False

            logger.info(f"Re-formatted capped Qty: {qty.normalize()}")
            # Double check if capped value is now below min (unlikely but possible with large steps)
            if qty < min_qty or qty.copy_abs() < CONFIG.position_qty_epsilon:
                logger.error(Fore.RED + f"{trade_action} failed: Capped quantity ({qty.normalize()}) is now below minimum ({min_qty.normalize()}) or zero. Aborting.")
                return False

        # Validate minimum cost if available
        min_cost_str = str(MARKET_INFO['limits'].get('cost', {}).get('min')) if MARKET_INFO['limits'].get('cost', {}).get('min') is not None else None
        if min_cost_str is not None:
            try:
                min_cost = Decimal(min_cost_str)
                estimated_cost = Decimal('0')
                # Estimate cost based on market type (Approximate!)
                if CONFIG.market_type == 'linear':
                    # Cost = Qty (Base) * Price (Quote/Base) = Quote
                    estimated_cost = qty * price
                elif CONFIG.market_type == 'inverse':
                    # Cost = Qty (Contracts) * Contract Size (Quote/Contract) = Quote
                    # Assuming contract size is in Quote currency (e.g., 1 USD for BTC/USD)
                    # Check if contract_size_dec is valid and positive
                    if not contract_size_dec.is_nan() and contract_size_dec > Decimal('0'):
                        estimated_cost = qty * contract_size_dec
                        logger.debug(f"Inverse cost estimation: Qty({qty.normalize()}) * ContractSize({contract_size_dec.normalize()}) = {estimated_cost.normalize()}")
                    else:
                        logger.warning("Could not determine valid contract size for inverse cost estimation.")
                        estimated_cost = Decimal('NaN')  # Cannot estimate cost
                else:
                    estimated_cost = Decimal('0')  # Should not happen

                # Check if estimated_cost is valid before comparison
                if not estimated_cost.is_nan() and estimated_cost < min_cost:
                    logger.error(Fore.RED + f"{trade_action} failed: Estimated order cost/value ({estimated_cost:.4f} {quote_currency}) is below minimum required ({min_cost:.4f} {quote_currency}). Increase risk or equity. Cannot place order.")
                    return False
                elif estimated_cost.is_nan():
                    logger.warning(Fore.YELLOW + "Estimated order cost could not be determined, skipping min cost check.")

            except (InvalidOperation, TypeError, KeyError) as cost_err:
                logger.warning(f"Could not estimate order cost: {cost_err}. Skipping min cost check.")
            except Exception as cost_err:
                logger.warning(f"Unexpected error during cost estimation: {cost_err}. Skipping min cost check.", exc_info=True)

        logger.info(Fore.YELLOW + f"Calculated Order: Side={side.upper()}, Qty={qty.normalize()}, Entry≈{price:.4f}, SL={sl_price_formatted_str} (ATR={atr:.4f})")

    except (InvalidOperation, TypeError, DivisionByZero, KeyError) as e:
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Error during pre-calculation/validation: {e}", exc_info=True)
        return False
    except Exception as e:  # Catch any other unexpected errors
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during pre-calculation: {e}", exc_info=True)
        return False

    # --- Cast the Market Order Spell ---
    order = None
    order_id = None
    filled_qty = Decimal("0.0")  # Initialize filled_qty for later use
    average_price = price  # Initialize average_price with estimated price, update if actual fill price is available

    try:
        logger.trade(f"Submitting {side.upper()} market order for {qty.normalize()} {symbol}...")
        # fetch_with_retries handles category param
        # CCXT expects float amount
        order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol,
            side=side,
            amount=float(qty)  # Explicitly cast Decimal qty to float for CCXT API
        )

        if order is None:
            # fetch_with_retries logged the error
            logger.error(Fore.RED + f"{trade_action} failed: Market order placement failed after retries.")
            return False

        logger.debug(f"Market order raw response: {order}")

        # --- Verify Order Fill (Crucial Step) ---
        # Bybit V5 create_market_order might return retCode=0 without an order ID in the standard field immediately.
        # The actual filled order details might be in 'info'->'result'->'list'.
        # Let's check retCode first if ID is missing, and try to extract details.

        # Default assumption: need to check status later
        needs_status_check = True
        order_status_data = None  # Initialize as None

        # Try to get order ID and fill details from the initial response first
        if order and isinstance(order, dict):
            order_id = order.get('id')  # Standard CCXT field

            # Check Bybit V5 specific response structure in 'info'
            if 'info' in order and isinstance(order['info'], dict):
                info = order['info']
                ret_code = info.get('retCode')
                ret_msg = info.get('retMsg')

                if ret_code == 0:  # Bybit V5 success code
                    logger.debug(f"Market order initial response retCode 0 ({ret_msg}).")
                    # Try to extract order ID or details from the V5 result list if available
                    if 'result' in info and isinstance(info['result'], dict) and 'list' in info['result'] and isinstance(info['result']['list'], list) and info['result']['list']:
                        # For market orders, the list should contain the immediately filled order(s)
                        first_order_info = info['result']['list'][0]
                        # Prioritize Order ID from the standard field, fallback to info field
                        order_id = order_id or first_order_info.get('orderId')  # Use standard ID if present, otherwise info ID

                        # Also capture filled details from response if possible (more accurate than check_order_status if available)
                        filled_qty_from_response_raw = first_order_info.get('cumExecQty', '0')  # Bybit V5 field
                        avg_price_from_response_raw = first_order_info.get('avgPrice', 'NaN')  # Bybit V5 field
                        order_status_from_response = first_order_info.get('orderStatus')  # Bybit V5 field e.g. "Filled"

                        try:
                            filled_qty_from_response = Decimal(str(filled_qty_from_response_raw))
                            avg_price_from_response = Decimal(str(avg_price_from_response_raw))
                        except InvalidOperation:
                            logger.error(f"Could not parse filled qty ({filled_qty_from_response_raw}) or avg price ({avg_price_from_response_raw}) from V5 response.")
                            filled_qty_from_response = Decimal('0')
                            avg_price_from_response = Decimal('NaN')

                        logger.debug(f"Extracted details from V5 response: ID={order_id}, Status={order_status_from_response}, Filled={filled_qty_from_response.normalize()}, AvgPrice={avg_price_from_response}")

                        # If response indicates "Filled" and filled quantity is significant
                        if order_status_from_response == 'Filled' and filled_qty_from_response.copy_abs() >= CONFIG.position_qty_epsilon:
                            filled_qty = filled_qty_from_response
                            average_price = avg_price_from_response if not avg_price_from_response.is_nan() else price  # Use estimated price if avgPrice is NaN
                            logger.trade(Fore.GREEN + Style.BRIGHT + f"Market order confirmed FILLED from response: {filled_qty.normalize()} @ {average_price:.4f}")
                            # Synthesize a CCXT-like dict for consistency
                            order_status_data = {'status': 'closed', 'filled': float(filled_qty), 'average': float(average_price) if not average_price.is_nan() else None, 'id': order_id}
                            needs_status_check = False  # No need to check status later, already confirmed filled
                        else:
                            logger.warning(f"{trade_action}: Order ID found ({order_id}) but filled quantity from response ({filled_qty_from_response.normalize()}) is zero or negligible, or status is not 'Filled'. Will proceed with check_order_status.")
                            # order_id is set, needs_status_check remains True -> check_order_status will be called
                    else:
                        logger.warning(f"{trade_action}: Market order submitted (retCode 0) but no Order ID or fill details found in V5 result list. Cannot reliably track status immediately.")
                        # needs_status_check remains True, order_id might be None -> check_order_status might fail without ID
                        # If order_id is None here, we cannot proceed safely.
                        if order_id is None:
                            logger.error(Fore.RED + f"{trade_action} failed: Market order submitted but no Order ID was obtained from response. Aborting.")
                            return False
                else:  # Non-zero retCode indicates failure
                    logger.error(Fore.RED + f"{trade_action} failed: Market order submission failed. Exchange message: {ret_msg} (Code: {ret_code})")
                    return False  # Submission failed

            # If we get here and needs_status_check is still True, proceed with status check
            if needs_status_check:
                if order_id is None:
                    logger.error(Fore.RED + f"{trade_action} failed: Market order submission response processed, but no order ID could be identified. Cannot check status. Aborting.")
                    return False  # Cannot proceed safely without order ID

                logger.info(f"Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}...")
                time.sleep(CONFIG.order_check_delay_seconds)

                # Use the dedicated check_order_status function
                order_status_data = check_order_status(order_id, symbol, timeout=CONFIG.order_check_timeout_seconds)

        # Evaluate the result of the status check (either from initial response or fetch_order)
        order_final_status = 'unknown'
        if order_status_data and isinstance(order_status_data, dict):
            order_final_status = order_status_data.get('status', 'unknown')
            filled_str = order_status_data.get('filled')
            average_str = order_status_data.get('average')  # Average fill price

            # Update filled_qty and average_price based on status check result
            if filled_str is not None:
                try: filled_qty = Decimal(str(filled_str))
                except InvalidOperation: logger.error(f"Could not parse filled quantity '{filled_str}' from status check.")
            if average_str is not None:
                try:
                    avg_price_decimal = Decimal(str(average_str))
                    if avg_price_decimal > 0:  # Use actual fill price only if valid
                        average_price = avg_price_decimal
                except InvalidOperation: logger.error(f"Could not parse average price '{average_str}' from status check.")

            logger.debug(f"Order {order_id} status check result: Status='{order_final_status}', Filled='{filled_qty.normalize()}', AvgPrice='{average_price:.4f}'")

            # 'closed' means fully filled for market orders on Bybit
            if order_final_status == 'closed' and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Order {order_id} confirmed FILLED: {filled_qty.normalize()} @ {average_price:.4f}")
            # Handle partial fills (less common for market, but possible during high volatility)
            # Bybit V5 market orders typically fill fully or are rejected. If partially filled, something unusual is happening.
            elif order_final_status in ['open', 'partially_filled'] and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                logger.warning(Fore.YELLOW + f"Market Order {order_id} status is '{order_final_status}' but partially/fully filled ({filled_qty.normalize()}). This is unusual for market orders. Proceeding with filled amount.")
                # Assume the filled quantity is the position size and proceed.
            elif order_final_status in ['open', 'partially_filled'] and filled_qty.copy_abs() < CONFIG.position_qty_epsilon:
                logger.error(Fore.RED + f"{trade_action} failed: Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty.normalize()}). Aborting SL placement.")
                # Attempt to cancel just in case it's stuck (defensive)
                try:
                    logger.info(f"Attempting cancellation of stuck/unfilled order {order_id}.")
                    # Bybit V5 cancel_order requires category and symbol
                    cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                    fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)  # Use fetch_with_retries
                except Exception as cancel_err: logger.warning(f"Failed to cancel stuck order {order_id}: {cancel_err}")
                return False
            else:  # canceled, rejected, expired, failed, unknown, or closed with zero fill
                logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Order {order_id} did not fill successfully: Status '{order_final_status}', Filled Qty: {filled_qty.normalize()}. Aborting SL placement.")
                # Attempt to cancel if not already in a terminal state (defensive)
                if order_final_status not in ['canceled', 'rejected', 'expired']:
                    try:
                        logger.info(f"Attempting cancellation of failed/unknown status order {order_id}.")
                        # Bybit V5 cancel_order requires category and symbol
                        cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                        fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)  # Use fetch_with_retries
                    except Exception: pass  # Ignore errors here, main goal failed anyway
                return False
        else:
            # check_order_status returned None (timeout, not found, or final check failed)
            # If check_order_status returns None, we cannot confirm successful fill. Assume failure.
            logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Could not determine final status for order {order_id} (timeout or not found). Assuming failure. Aborting SL placement.")
            # Attempt to cancel just in case it's stuck somehow (defensive)
            try:
                logger.info(f"Attempting cancellation of unknown status order {order_id}.")
                # If order_id was None earlier, this will fail. check_order_status should handle None ID internally if possible, but better to have ID.
                if order_id:
                    # Bybit V5 cancel_order requires category and symbol
                    cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                    fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)
                else:
                    logger.warning("Cannot attempt cancellation: No order ID available.")
            except Exception: pass
            return False

        # Final check on filled quantity after status check
        if filled_qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.error(Fore.RED + f"{trade_action} failed: Order {order_id} resulted in effectively zero filled quantity ({filled_qty.normalize()}) after status check. No position opened.")
            return False

        # --- Place Initial Stop-Loss Order (Set on Position for Bybit V5) ---
        position_side = "long" if side == "buy" else "short"
        logger.trade(f"Setting initial SL for new {position_side.upper()} position (filled qty: {filled_qty.normalize()})...")

        # Use the SL price calculated earlier, already formatted string
        sl_price_str_for_api = sl_price_formatted_str

        # Define parameters for setting the stop-loss on the position (Bybit V5 specific)
        # We use the `private_post_position_set_trading_stop` implicit method via CCXT
        # This endpoint applies to the *entire* position for the symbol/side/category.
        set_sl_params = {
            'category': CONFIG.market_type,  # Required
            'symbol': MARKET_INFO['id'],  # Use exchange-specific market ID
            'stopLoss': sl_price_str_for_api,  # Trigger price for the stop loss
            'slTriggerBy': CONFIG.sl_trigger_by,  # e.g., 'LastPrice', 'MarkPrice'
            'tpslMode': 'Full',  # Apply SL/TP/TSL to the entire position ('Partial' also possible)
            # 'positionIdx': 0 # For Bybit unified: 0 for one-way mode (default), 1/2 for hedge mode
            # Note: We don't need quantity here as it applies to the existing position matching symbol/category/side.
            # No need to specify side in params for V5 set-trading-stop, it's determined by the symbol and position context.
            # Wait, the Bybit V5 docs *do* show 'side' as a parameter for set-trading-stop. Let's add it for clarity and correctness.
            'side': 'Buy' if position_side == 'long' else 'Sell'  # Add side parameter (Bybit V5 expects "Buy"/"Sell" for side)
        }
        logger.trade(f"Setting Position SL: Trigger={sl_price_str_for_api}, TriggerBy={CONFIG.sl_trigger_by}, Side={set_sl_params['side']}")
        logger.debug(f"Set SL Params (for setTradingStop): {set_sl_params}")

        sl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods if available
            # Endpoint: POST /v5/position/set-trading-stop
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                sl_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_sl_params)
            else:
                # Fallback: Raise error if specific method missing.
                logger.error(Fore.RED + "Cannot set SL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                # Critical: Position is open without SL. Attempt emergency close.
                raise ccxt.NotSupported("SL setting method not available via CCXT.")

            logger.debug(f"Set SL raw response: {sl_set_response}")

            # Handle potential failure from fetch_with_retries
            if sl_set_response is None:
                # fetch_with_retries already logged the failure
                raise ccxt.ExchangeError("Set SL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(sl_set_response.get('info'), dict) and sl_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Stop Loss successfully set directly on the {position_side.upper()} position (Trigger: {sl_price_str_for_api}).")
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
                termux_notify("Trade Entry", f"{side.upper()} {symbol} @ {average_price:.4f}, SL: {sl_price_str_for_api}")
                return True  # SUCCESS!

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                error_code = None
                if isinstance(sl_set_response.get('info'), dict):
                    error_msg = sl_set_response['info'].get('retMsg', error_msg)
                    error_code = sl_set_response['info'].get('retCode')
                    error_msg += f" (Code: {error_code})"
                raise ccxt.ExchangeError(f"Stop loss setting failed. Exchange message: {error_msg}")

        # --- Handle SL Setting Failures ---
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied) as e:
            # This is critical - position opened but SL setting failed. Emergency close needed.
            logger.critical(Fore.RED + Style.BRIGHT + f"CRITICAL: Failed to set stop-loss on position after entry: {e}. Position is UNPROTECTED.")
            logger.warning(Fore.YELLOW + "Attempting emergency market closure of unprotected position...")
            try:
                emergency_close_side = "sell" if position_side == "long" else "buy"
                # Use the *filled quantity* from the successful market order fill check
                # Format filled quantity precisely for closure order
                close_qty_str = format_amount(symbol, filled_qty.copy_abs(), ROUND_DOWN)
                try:
                    close_qty_decimal = Decimal(close_qty_str)
                except InvalidOperation:
                    logger.critical(f"{Fore.RED}Failed to parse closure quantity '{close_qty_str}'. Cannot attempt emergency closure.")
                    termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED! Manual action required!")
                    return False  # Indicate failure of the entire entry process

                # Check against minimum quantity again before closing
                try:
                    min_qty_close = Decimal(str(MARKET_INFO['limits']['amount']['min']))
                except (KeyError, InvalidOperation, TypeError):
                    logger.warning("Could not determine minimum order quantity for emergency closure validation.")
                    min_qty_close = Decimal("0")  # Assume zero if unavailable

                if close_qty_decimal < min_qty_close:
                    logger.critical(f"{Fore.RED}Emergency closure quantity {close_qty_decimal.normalize()} is below minimum {min_qty_close}. MANUAL CLOSURE REQUIRED for {position_side.upper()} position!")
                    termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & < MIN QTY! Close manually!")
                    # Do NOT reset tracker state here, as we don't know the position status for sure.
                    return False  # Indicate failure of the entire entry process

                # Place the emergency closure order
                emergency_close_params = {'reduceOnly': True}  # Ensure it only closes
                # fetch_with_retries handles category param
                emergency_close_order = fetch_with_retries(
                    EXCHANGE.create_market_order,
                    symbol=symbol,
                    side=emergency_close_side,
                    amount=float(close_qty_decimal),  # CCXT needs float
                    params=emergency_close_params
                )

                if emergency_close_order and (emergency_close_order.get('id') or emergency_close_order.get('info', {}).get('retCode') == 0):
                    close_id = emergency_close_order.get('id', 'N/A (retCode 0)')
                    logger.trade(Fore.GREEN + f"Emergency closure order placed successfully: ID {close_id}")
                    termux_notify("Closure Attempted", f"{symbol} emergency closure sent.")
                    # Reset tracker state as position *should* be closing (best effort)
                    order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                else:
                    error_msg = emergency_close_order.get('info', {}).get('retMsg', 'Unknown error') if isinstance(emergency_close_order, dict) else str(emergency_close_order)
                    logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Order placement failed): {error_msg}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!")
                    termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                    # Do NOT reset tracker state here.

            except Exception as close_err:
                logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Exception during closure): {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!", exc_info=True)
                termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                # Do NOT reset tracker state here.

            return False  # Signal overall failure of the entry process due to SL failure

        except Exception as e:
            logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error setting SL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position may be open without Stop Loss due to unexpected SL setting error. MANUAL INTERVENTION ADVISED.")
            # Consider emergency closure here too? Yes, safer. Re-use the emergency closure logic.
            try:
                position_side = "long" if side == "buy" else "short"
                emergency_close_side = "sell" if position_side == "long" else "buy"
                close_qty_str = format_amount(symbol, filled_qty.copy_abs(), ROUND_DOWN)
                try:
                    close_qty_decimal = Decimal(close_qty_str)
                except InvalidOperation:
                    logger.critical(f"{Fore.RED}Failed to parse closure quantity '{close_qty_str}' after unexpected SL error. Cannot attempt emergency closure.")
                    termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED! Manual action required!")
                    return False  # Indicate failure

                try:
                    min_qty_close = Decimal(str(MARKET_INFO['limits']['amount']['min']))
                except (KeyError, InvalidOperation, TypeError):
                    logger.warning("Could not determine minimum order quantity for emergency closure validation.")
                    min_qty_close = Decimal("0")  # Assume zero if unavailable

                if close_qty_decimal < min_qty_close:
                    logger.critical(f"{Fore.RED}Emergency closure quantity {close_qty_decimal.normalize()} is below minimum {min_qty_close}. MANUAL CLOSURE REQUIRED for {position_side.upper()} position!")
                    termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & < MIN QTY! Close manually!")
                    return False  # Indicate failure

                emergency_close_params = {'reduceOnly': True}
                emergency_close_order = fetch_with_retries(
                    EXCHANGE.create_market_order,
                    symbol=symbol,
                    side=emergency_close_side,
                    amount=float(close_qty_decimal),
                    params=emergency_close_params
                )
                if emergency_close_order and (emergency_close_order.get('id') or emergency_close_order.get('info', {}).get('retCode') == 0):
                    close_id = emergency_close_order.get('id', 'N/A (retCode 0)')
                    logger.trade(Fore.GREEN + f"Emergency closure order placed successfully after unexpected SL error: ID {close_id}")
                    termux_notify("Closure Attempted", f"{symbol} emergency closure sent after SL error.")
                    order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                else:
                    error_msg = emergency_close_order.get('info', {}).get('retMsg', 'Unknown error') if isinstance(emergency_close_order, dict) else str(emergency_close_order)
                    logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Order placement failed) after unexpected SL error: {error_msg}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!")
                    termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
            except Exception as close_err:
                logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Exception during closure) after unexpected SL error: {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!", exc_info=True)
                termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")

            return False  # Signal overall failure

    # --- Handle Initial Market Order Failures ---
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied) as e:
        # Error placing the initial market order itself (handled by fetch_with_retries re-raising)
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Exchange error placing market order: {e}")
        # The exception message itself is usually sufficient.
        return False
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during market order placement: {e}", exc_info=True)
        return False


def manage_trailing_stop(
    symbol: str,
    position_side: str,  # 'long' or 'short'
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
    if position_qty.copy_abs() < CONFIG.position_qty_epsilon or entry_price.is_nan() or entry_price <= Decimal("0"):
        # If position seems closed or invalid, ensure tracker is clear.
        if order_tracker[position_side]["sl_id"] or order_tracker[position_side]["tsl_id"]:
            logger.info(f"Position {position_side} appears closed or invalid (Qty: {position_qty.normalize()}, Entry: {entry_price}). Clearing stale order trackers.")
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
        log_msg = f"{position_side.upper()} TSL ({active_tsl_marker}) is already active. Exchange is managing the trail."
        logger.debug(log_msg)
        # Sanity check: Ensure initial SL marker is None if TSL is active
        if initial_sl_marker:
            logger.warning(f"Inconsistent state: TSL active ({active_tsl_marker}) but initial SL marker ({initial_sl_marker}) is also present in tracker. Clearing initial SL marker.")
            order_tracker[position_side]["sl_id"] = None
        return  # TSL is already active, nothing more to do here

    # If TSL is not active, check if we *should* activate it.
    # Requires an initial SL marker to be present to indicate the position is at least protected by a fixed SL.
    if not initial_sl_marker:
        # This can happen if the initial SL setting failed, or if state got corrupted.
        # Note: Bybit V5 allows setting TSL without removing SL in the same call,
        # but clearing the SL when activating TSL is the intended behavior for this strategy.
        # If there's no SL marker here, it implies either SL wasn't set, was hit, or was manually removed.
        # We proceed cautiously: if initial SL marker is missing, we *could* attempt to set TSL,
        # but it might overwrite an existing manual SL/TP.
        # For robustness, let's *only* attempt TSL activation if the initial SL marker is present,
        # assuming the marker indicates the position is protected by the initial SL we set.
        # If the marker is missing, assume the position is either already managed, or unprotected.
        logger.warning(f"Cannot activate TSL for {position_side.upper()}: Initial SL protection marker is missing from tracker ({initial_sl_marker}). Position might be unprotected or already managed externally. Skipping TSL activation.")
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
    logger.debug(f"{position_side.upper()} Profit: {profit:.8f}, TSL Activation Threshold (Points): {activation_threshold_points:.8f} ({CONFIG.tsl_activation_atr_multiplier} * ATR)")

    # Activate TSL only if profit exceeds the threshold (use Decimal comparison)
    # Use a small buffer (e.g., 0.01% of price) for threshold comparison to avoid flickering near the threshold
    # A fixed tiny Decimal might be better than a percentage buffer for volatile pairs
    threshold_buffer = current_price.copy_abs() * Decimal('0.0001') if not current_price.is_nan() else Decimal('0')  # 0.01% of current price as buffer

    if not profit.is_nan() and profit > activation_threshold_points + threshold_buffer:
        logger.trade(Fore.GREEN + Style.BRIGHT + f"Profit threshold reached for {position_side.upper()} position (Profit {profit:.4f} > Threshold {activation_threshold_points:.4f}). Activating TSL.")

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
            'category': CONFIG.market_type,  # Required
            'symbol': MARKET_INFO['id'],  # Use exchange-specific market ID
            'trailingStop': trail_percent_str,  # Trailing distance percentage (as string)
            'tpslMode': 'Full',  # Apply to the whole position
            'slTriggerBy': CONFIG.tsl_trigger_by,  # Trigger type for the trail (LastPrice, MarkPrice, IndexPrice)
            # 'activePrice': format_price(symbol, current_price), # Optional: Price to activate the trail immediately. If omitted, Bybit activates when price moves favorably by trail %. Check docs.
            # Recommended: Don't set activePrice here. Let Bybit handle the initial activation based on the best price.
            # To remove the fixed SL when activating TSL, Bybit V5 documentation indicates setting 'stopLoss' to "" (empty string) or '0'.
            # Setting to "" is often safer to explicitly indicate removal.
            'stopLoss': '',  # Remove the fixed SL when activating TSL
            'side': 'Buy' if position_side == 'long' else 'Sell'  # Add side parameter (Bybit V5 expects "Buy"/"Sell" for side)
            # 'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(f"Setting Position TSL: Trail={trail_percent_str}%, TriggerBy={CONFIG.tsl_trigger_by}, Side={set_tsl_params['side']}, Removing Fixed SL")
        logger.debug(f"Set TSL Params (for setTradingStop): {set_tsl_params}")

        tsl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                tsl_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_tsl_params)
            else:
                logger.error(Fore.RED + "Cannot set TSL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                # Cannot proceed safely, log failure but don't trigger emergency close (position is still protected by initial SL if it was set)
                raise ccxt.NotSupported("TSL setting method not available.")

            logger.debug(f"Set TSL raw response: {tsl_set_response}")

            # Handle potential failure from fetch_with_retries
            if tsl_set_response is None:
                # fetch_with_retries already logged the failure
                raise ccxt.ExchangeError("Set TSL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(tsl_set_response.get('info'), dict) and tsl_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Trailing Stop Loss successfully activated for {position_side.upper()} position. Trail: {trail_percent_str}%")
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
                if isinstance(tsl_set_response.get('info'), dict):
                    error_msg = tsl_set_response['info'].get('retMsg', error_msg)
                    error_code = tsl_set_response['info'].get('retCode')
                    error_msg += f" (Code: {error_code})"
                # Check if error was due to trying to remove non-existent SL (might be benign, e.g., SL already hit)
                # Example Bybit code: 110025 = SL/TP order not found or completed
                if error_code == 110025:
                    logger.warning(f"TSL activation may have succeeded, but received code 110025 (SL/TP not found/completed) when trying to clear fixed SL. Assuming TSL is active and fixed SL was already gone.")
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
            logger.warning(Fore.YELLOW + "Position continues with initial SL (if successfully set) or may be UNPROTECTED if initial SL failed. MANUAL INTERVENTION ADVISED if initial SL state is uncertain.")
            # Do NOT clear the initial SL marker here. Do not set TSL marker.
            termux_notify("TSL Activation FAILED!", f"{symbol} TSL activation failed. Check logs/position.")
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error activating TSL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position continues with initial SL (if successfully set) or may be UNPROTECTED. MANUAL INTERVENTION ADVISED if initial SL state is uncertain.")
            termux_notify("TSL Activation FAILED!", f"{symbol} TSL activation failed (unexpected). Check logs/position.")

    else:
        # Profit threshold not met
        sl_status_log = f"({initial_sl_marker})" if initial_sl_marker else "(None!)"
        logger.debug(f"{position_side.upper()} profit ({profit:.4f}) has not crossed TSL activation threshold ({activation_threshold_points:.4f}). Keeping initial SL {sl_status_log}.")


def print_status_panel(
    cycle: int, timestamp: Optional[pd.Timestamp], price: Optional[Decimal], indicators: Optional[Dict[str, Decimal]],
    positions: Optional[Dict[str, Dict[str, Any]]], equity: Optional[Decimal], signals: Dict[str, Union[bool, str]],
    total_realized_pnl: Optional[Decimal],
    order_tracker_state: Dict[str, Dict[str, Optional[str]]]  # Pass tracker state snapshot explicitly
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
    # Use .get(..., Decimal('NaN')) for safe access to indicator values
    price_str = f"{price:.4f}" if price is not None and not price.is_nan() else f"{Fore.YELLOW}N/A"
    atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')
    atr_str = f"{atr:.6f}" if not atr.is_nan() else f"{Fore.YELLOW}N/A"

    trend_ema = indicators.get('trend_ema', Decimal('NaN')) if indicators else Decimal('NaN')
    trend_ema_str = f"{trend_ema:.4f}" if not trend_ema.is_nan() else f"{Fore.YELLOW}N/A"

    price_color = Fore.WHITE
    trend_desc = f"{Fore.YELLOW}Trend N/A"
    if price is not None and not price.is_nan() and not trend_ema.is_nan():
        # Use a small buffer for display consistency (e.g., 0.01% of price)
        trend_buffer_display = price.copy_abs() * Decimal('0.0001')  # 0.01% of current price
        if price > trend_ema + trend_buffer_display: price_color = Fore.GREEN; trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema - trend_buffer_display: price_color = Fore.RED; trend_desc = f"{price_color}(Below Trend)"
        else: price_color = Fore.YELLOW; trend_desc = f"{price_color}(At Trend)"

    status_data = [
        [section_color + "Market", value_color + CONFIG.symbol, f"{price_color}{price_str}"],
        [section_color + f"Trend EMA ({CONFIG.trend_ema_period})", f"{value_color}{trend_ema_str}", trend_desc],
        [section_color + f"ATR ({indicators.get('atr_period', CONFIG.atr_period)})", f"{value_color}{atr_str}", ""], # Display ATR period if stored in indicators
    ]

    if CONFIG.strategy == 'ecc_scalp':
        ecc_cycle = indicators.get('ecc_cycle', Decimal('NaN')) if indicators else Decimal('NaN')
        ecc_trigger = indicators.get('ecc_trigger', Decimal('NaN')) if indicators else Decimal('NaN')
        ecc_cycle_str = f"{ecc_cycle:.4f}" if not ecc_cycle.is_nan() else f"{Fore.YELLOW}N/A"
        ecc_trigger_str = f"{ecc_trigger:.4f}" if not ecc_trigger.is_nan() else f"{Fore.YELLOW}N/A"
        ecc_color = Fore.WHITE
        ecc_desc = f"{Fore.YELLOW}ECC N/A"
        if not ecc_cycle.is_nan() and not ecc_trigger.is_nan():
             if ecc_cycle > ecc_trigger: ecc_color = Fore.GREEN; ecc_desc = f"{ecc_color}Bullish Cross"
             elif ecc_cycle < ecc_trigger: ecc_color = Fore.RED; ecc_desc = f"{ecc_color}Bearish Cross"
             else: ecc_color = Fore.YELLOW; ecc_desc = f"{Fore.YELLOW}Aligned"

        status_data.extend([
             [section_color + f"ECC ({indicators.get('ecc_lookback', CONFIG.ecc_lookback)}, {indicators.get('ecc_alpha', CONFIG.ecc_alpha)})", f"{ecc_color}{ecc_cycle_str} / {ecc_trigger_str}", ecc_desc],
        ])

    elif CONFIG.strategy == 'ema_stoch':
        fast_ema = indicators.get('fast_ema', Decimal('NaN')) if indicators else Decimal('NaN')
        slow_ema = indicators.get('slow_ema', Decimal('NaN')) if indicators else Decimal('NaN')
        stoch_k = indicators.get('stoch_k', Decimal('NaN')) if indicators else Decimal('NaN')
        stoch_d = indicators.get('stoch_d', Decimal('NaN')) if indicators else Decimal('NaN')

        fast_ema_str = f"{fast_ema:.4f}" if not fast_ema.is_nan() else f"{Fore.YELLOW}N/A"
        slow_ema_str = f"{slow_ema:.4f}" if not slow_ema.is_nan() else f"{Fore.YELLOW}N/A"
        stoch_k_str = f"{stoch_k:.2f}" if not stoch_k.is_nan() else f"{Fore.YELLOW}N/A"
        stoch_d_str = f"{stoch_d:.2f}" if not stoch_d.is_nan() else f"{Fore.YELLOW}N/A"

        ema_cross_color = Fore.WHITE
        ema_desc = f"{Fore.YELLOW}EMA N/A"
        if not fast_ema.is_nan() and not slow_ema.is_nan():
            if fast_ema > slow_ema: ema_cross_color = Fore.GREEN; ema_desc = f"{ema_cross_color}Bullish Cross"
            elif fast_ema < slow_ema: ema_cross_color = Fore.RED; ema_desc = f"{ema_cross_color}Bearish Cross"
            else: ema_cross_color = Fore.YELLOW; ema_desc = f"{Fore.YELLOW}Aligned"

        stoch_color = Fore.YELLOW
        stoch_desc = f"{Fore.YELLOW}Stoch N/A"
        if not stoch_k.is_nan():
            if stoch_k < Decimal(25): stoch_color = Fore.GREEN; stoch_desc = f"{stoch_color}Oversold (<25)"
            elif stoch_k > Decimal(75): stoch_color = Fore.RED; stoch_desc = f"{stoch_color}Overbought (>75)"
            else: stoch_color = Fore.YELLOW; stoch_desc = f"{stoch_color}Neutral (25-75)"

        status_data.extend([
             [section_color + f"EMA Fast/Slow ({indicators.get('ema_fast_period', CONFIG.ema_fast_period)}/{indicators.get('ema_slow_period', CONFIG.ema_slow_period)})", f"{ema_cross_color}{fast_ema_str} / {slow_ema_str}", ema_desc],
             [section_color + f"Stoch %K/%D ({indicators.get('stoch_period', CONFIG.stoch_period)},{indicators.get('stoch_smooth_k', CONFIG.stoch_smooth_k)},{indicators.get('stoch_smooth_d', CONFIG.stoch_smooth_d)})", f"{stoch_color}{stoch_k_str} / {stoch_d_str}", stoch_desc],
        ])

    print(tabulate(status_data, tablefmt="fancy_grid", colalign=("left", "left", "left")))
    # print(header_color + "-" * 80) # Separator removed, using table grid

    # --- Positions & Orders ---
    pos_avail = positions is not None
    long_pos = positions.get('long', {}) if pos_avail else {}
    short_pos = positions.get('short', {}) if pos_avail else {}

    # Safely get values, handling None or NaN Decimals
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
            else: return f"{Fore.GREEN}TSL Active (ID: ...{tsl_marker[-6:]})"  # Should not happen with V5 pos-based TSL
        elif sl_marker:
            if sl_marker.startswith("POS_SL_"): return f"{Fore.YELLOW}SL Active (Pos)"
            else: return f"{Fore.YELLOW}SL Active (ID: ...{sl_marker[-6:]})"  # Should not happen with V5 pos-based SL
        else:
            # No marker found in tracker
            return f"{Fore.RED}{Style.BRIGHT}NONE (!)"  # Highlight if no stop is tracked

    # Display stop status only if position exists (using epsilon check)
    long_stop_status = get_stop_status(long_sl_marker, long_tsl_marker) if long_qty.copy_abs() >= CONFIG.position_qty_epsilon else f"{value_color}-"
    short_stop_status = get_stop_status(short_sl_marker, short_tsl_marker) if short_qty.copy_abs() >= CONFIG.position_qty_epsilon else f"{value_color}-"

    # Format position details, handle potential None or NaN from failed fetch/parsing
    if not pos_avail:
        long_qty_str, short_qty_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_entry_str, short_entry_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_pnl_str, short_pnl_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_liq_str, short_liq_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
    else:
        # Format Decimals nicely, remove trailing zeros for quantity (more readable)
        long_qty_str = f"{long_qty.normalize()}" if long_qty.copy_abs() >= CONFIG.position_qty_epsilon else "0"  # Use normalize to remove trailing zeros
        short_qty_str = f"{short_qty.normalize()}" if short_qty.copy_abs() >= CONFIG.position_qty_epsilon else "0"  # Use normalize

        long_entry_str = f"{long_entry:.4f}" if not long_entry.is_nan() else "-"
        short_entry_str = f"{short_entry:.4f}" if not short_entry.is_nan() else "-"

        # PnL color based on value, only display if position exists
        long_pnl_color = Fore.GREEN if not long_pnl.is_nan() and long_pnl >= 0 else Fore.RED
        short_pnl_color = Fore.GREEN if not short_pnl.is_nan() and short_pnl >= 0 else Fore.RED
        long_pnl_str = f"{long_pnl_color}{long_pnl:+.4f}{value_color}" if long_qty.copy_abs() >= CONFIG.position_qty_epsilon and not long_pnl.is_nan() else "-"
        short_pnl_str = f"{short_pnl_color}{short_pnl:+.4f}{value_color}" if short_qty.copy_abs() >= CONFIG.position_qty_epsilon and not short_pnl.is_nan() else "-"

        # Liq price color (usually red), only display if position exists
        long_liq_str = f"{Fore.RED}{long_liq:.4f}{value_color}" if long_qty.copy_abs() >= CONFIG.position_qty_epsilon and not long_liq.is_nan() and long_liq > 0 else "-"
        short_liq_str = f"{Fore.RED}{short_liq:.4f}{value_color}" if short_qty.copy_abs() >= CONFIG.position_qty_epsilon and not short_liq.is_nan() and short_liq > 0 else "-"

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

    # --- PnL Summary ---
    realized_pnl_str = f"{total_realized_pnl:+.4f}" if total_realized_pnl is not None and not total_realized_pnl.is_nan() else f"{Fore.YELLOW}N/A"
    realized_pnl_color = Fore.GREEN if total_realized_pnl is not None and not total_realized_pnl.is_nan() and total_realized_pnl >= 0 else Fore.RED
    print(f" {section_color}Realized PnL (Last {CONFIG.pnl_lookback_days} Days): {realized_pnl_color}{realized_pnl_str}{reset_all}")
    print(header_color + "-" * 80)

    # --- Signals ---
    long_signal_status = signals.get('long', False)
    short_signal_status = signals.get('short', False)
    long_signal_color = Fore.GREEN + Style.BRIGHT if long_signal_status else Fore.WHITE
    short_signal_color = Fore.RED + Style.BRIGHT if short_signal_status else Fore.WHITE
    trend_status = f"(Trend Filter: {value_color}{'ON' if CONFIG.trade_only_with_trend else 'OFF'}{header_color})"
    signal_reason_text = signals.get('reason', 'N/A')
    print(f" Strategy: {value_color}{CONFIG.strategy.upper()}{header_color} | Signals {trend_status}: Long [{long_signal_color}{str(long_signal_status).upper():<5}{header_color}] | Short [{short_signal_color}{str(short_signal_status).upper():<5}{header_color}]")  # Use .upper() for bool string
    # Display the signal reason below
    print(f" Reason: {Fore.YELLOW}{signal_reason_text}{Style.RESET_ALL}")
    print(header_color + "=" * 80 + reset_all)


def generate_signals(indicators: Optional[Dict[str, Decimal]], current_price: Optional[Decimal], prev_indicators: Optional[Dict[str, Decimal]]) -> Dict[str, Union[bool, str]]:
    """Generates trading signals based on indicator conditions and selected strategy, using Decimal."""
    long_signal = False
    short_signal = False
    signal_reason = f"No signal - {CONFIG.strategy.upper()} Strategy"

    if not indicators:
        logger.warning("Cannot generate signals: indicators dictionary is missing.")
        return {"long": False, "short": False, "reason": "Indicators missing"}
    if not prev_indicators and CONFIG.strategy == 'ecc_scalp':
        # ECC scalp needs previous indicator values for cross detection
        logger.warning("Cannot generate ECC signals: Previous indicators missing.")
        return {"long": False, "short": False, "reason": "Previous indicators missing for ECC"}
    if current_price is None or current_price.is_nan() or current_price <= Decimal(0):
        logger.warning("Cannot generate signals: current price is missing or invalid.")
        return {"long": False, "short": False, "reason": "Invalid price"}

    try:
        # Use .get with default Decimal('NaN') to handle missing/failed indicators gracefully
        trend_ema = indicators.get('trend_ema', Decimal('NaN'))
        prev_trend_ema = prev_indicators.get('trend_ema', Decimal('NaN')) if prev_indicators else Decimal('NaN')


        # Check if any required indicator for the strategy is NaN
        failed_strategy_indicators = []
        if CONFIG.strategy == 'ecc_scalp':
             ecc_cycle = indicators.get('ecc_cycle', Decimal('NaN'))
             ecc_trigger = indicators.get('ecc_trigger', Decimal('NaN'))
             prev_ecc_cycle = prev_indicators.get('ecc_cycle', Decimal('NaN')) if prev_indicators else Decimal('NaN')
             prev_ecc_trigger = prev_indicators.get('ecc_trigger', Decimal('NaN')) if prev_indicators else Decimal('NaN')

             required_for_ecc_signal = {'ecc_cycle': ecc_cycle, 'ecc_trigger': ecc_trigger, 'trend_ema': trend_ema}
             if prev_indicators: # Previous values needed for cross logic
                 required_for_ecc_signal['prev_ecc_cycle'] = prev_ecc_cycle
                 required_for_ecc_signal['prev_ecc_trigger'] = prev_ecc_trigger

             failed_strategy_indicators = [name for name, val in required_for_ecc_signal.items() if val.is_nan()]

             if failed_strategy_indicators:
                  return {"long": False, "short": False, "reason": f"NaN indicator(s) for ECC: {', '.join(failed_strategy_indicators)}"}

             # ECC Signal Logic (Based on original script's scalp_with_ecc)
             # Buy signal: Cycle crosses above Trigger
             ecc_buy_cross = (ecc_cycle > ecc_trigger) and (prev_ecc_cycle <= prev_ecc_trigger)
             # Sell signal: Cycle crosses below Trigger
             ecc_sell_cross = (ecc_cycle < ecc_trigger) and (prev_ecc_cycle >= prev_ecc_trigger)

             long_entry_condition_base = ecc_buy_cross
             short_entry_condition_base = ecc_sell_cross
             base_reason_prefix = "ECC Cross"


        elif CONFIG.strategy == 'ema_stoch':
             k = indicators.get('stoch_k', Decimal('NaN'))
             # d = indicators.get('stoch_d', Decimal('NaN')) # Available but not used in current logic
             fast_ema = indicators.get('fast_ema', Decimal('NaN'))
             slow_ema = indicators.get('slow_ema', Decimal('NaN'))

             required_for_ema_stoch_signal = {'stoch_k': k, 'fast_ema': fast_ema, 'slow_ema': slow_ema, 'trend_ema': trend_ema}
             failed_strategy_indicators = [name for name, val in required_for_ema_stoch_signal.items() if val.is_nan()]

             if failed_strategy_indicators:
                 return {"long": False, "short": False, "reason": f"NaN indicator(s) for EMA/Stoch: {', '.join(failed_strategy_indicators)}"}

             # EMA/Stoch Signal Logic (Based on previous Pyrmethus script)
             ema_bullish_cross = fast_ema > slow_ema
             ema_bearish_cross = fast_ema < slow_ema
             stoch_oversold = k < Decimal(25)
             stoch_overbought = k > Decimal(75)

             long_entry_condition_base = ema_bullish_cross and stoch_oversold
             short_entry_condition_base = ema_bearish_cross and stoch_overbought
             base_reason_prefix = "EMA Cross & Stoch Over"


        else:
             # Should be caught by config validation, but defensive check
             return {"long": False, "short": False, "reason": f"Unsupported Strategy: {CONFIG.strategy}"}

        # --- Apply Trend Filter (Common to both strategies if enabled) ---
        # Use a small buffer (e.g., 0.01% of price) for price vs trend EMA comparison to avoid false signals on tiny fluctuations
        # Calculate buffer only if price is valid
        trend_buffer = current_price.copy_abs() * Decimal('0.0001') if not current_price.is_nan() else Decimal('0')  # 0.01% of current price as buffer

        # Use absolute comparison with buffer
        price_above_trend = current_price > trend_ema + trend_buffer
        price_below_trend = current_price < trend_ema - trend_buffer
        # price_at_trend = abs(current_price - trend_ema) <= trend_buffer # Not explicitly needed for filter logic


        if long_entry_condition_base:
            if CONFIG.trade_only_with_trend:
                if price_above_trend:
                    long_signal = True
                    signal_reason = f"Long: {base_reason_prefix} & Price Above Trend EMA (Trend Filter ON)"
                else:
                    signal_reason = f"Long Blocked: Price Not Above Trend EMA (Trend Filter ON)"
            else:  # Trend filter off
                long_signal = True
                signal_reason = f"Long: {base_reason_prefix} (Trend Filter OFF)"

        elif short_entry_condition_base:
            if CONFIG.trade_only_with_trend:
                if price_below_trend:
                    short_signal = True
                    signal_reason = f"Short: {base_reason_prefix} & Price Below Trend EMA (Trend Filter ON)"
                else:
                    signal_reason = f"Short Blocked: Price Not Below Trend EMA (Trend Filter ON)"
            else:  # Trend filter off
                short_signal = True
                signal_reason = f"Short: {base_reason_prefix} (Trend Filter OFF)"
        else:
            # Provide more context if no primary condition met
            reason_parts = []
            if CONFIG.strategy == 'ecc_scalp':
                 if not ecc_buy_cross and not ecc_sell_cross: reason_parts.append("No ECC cross")
                 elif ecc_buy_cross: reason_parts.append("ECC Bullish Cross (Trend filter blocked)")
                 elif ecc_sell_cross: reason_parts.append("ECC Bearish Cross (Trend filter blocked)")
            elif CONFIG.strategy == 'ema_stoch':
                 if not ema_bullish_cross and not ema_bearish_cross: reason_parts.append("No EMA cross")
                 elif ema_bullish_cross: reason_parts.append("EMA Bullish Cross (Stoch not Oversold)")
                 elif ema_bearish_cross: reason_parts.append("EMA Bearish Cross (Stoch not Overbought)")
                 if not k.is_nan():
                      if stoch_oversold: reason_parts.append("Stoch Oversold")
                      elif stoch_overbought: reason_parts.append("Stoch Overbought")
                      else: reason_parts.append("Stoch Neutral")


            if CONFIG.trade_only_with_trend and not trend_ema.is_nan() and not current_price.is_nan():
                 if price_above_trend: reason_parts.append("Price Above Trend EMA")
                 elif price_below_trend: reason_parts.append("Price Below Trend EMA")
                 else: reason_parts.append("Price At Trend EMA")

            if not reason_parts: reason_parts.append("Conditions not met")  # Default if none match
            signal_reason = f"No signal ({', '.join(reason_parts)})"


        # Log the outcome
        if long_signal or short_signal:
            logger.info(Fore.CYAN + f"Signal Generated: {signal_reason}")
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


def trading_spell_cycle(cycle_count: int, prev_indicators: Optional[Dict[str, Decimal]]) -> Tuple[Optional[Dict[str, Decimal]], bool]:
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
        return None, False # Return None for indicators and indicate failure


    # 2. Get Current Price & Timestamp from Data
    current_price: Optional[Decimal] = None
    last_timestamp: Optional[pd.Timestamp] = None
    try:
        # Use close price of the last *completed* candle for indicator-based logic
        # Ensure there's at least one row
        if not df.empty:
            last_candle = df.iloc[-1]
            current_price_float = last_candle["close"]
            if pd.isna(current_price_float):
                raise ValueError("Latest close price is NaN")
            current_price = Decimal(str(current_price_float))
            last_timestamp = df.index[-1]  # Already UTC from fetch_market_data
            logger.debug(f"Latest candle: Time={last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price:.8f}")  # Log with high precision
        else:
            raise ValueError("DataFrame is empty after processing OHLCV data.")

        # Check for stale data (compare last candle time to current time)
        now_utc = pd.Timestamp.utcnow()  # UTC timestamp
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
        return None, False # Return None for indicators and indicate failure

    # 3. Calculate Indicators (returns Decimals or None)
    indicators = calculate_indicators(df)
    # Note: calculate_indicators now returns dict with NaNs for failed indicators, or None if critical failure (e.g., not enough data)
    if indicators is None:
        logger.error(Fore.RED + "Indicator calculation failed critically (e.g., not enough data). Continuing cycle but skipping trade actions.")
        cycle_success = False  # Mark as failed for logging, but continue to fetch state and show panel
        current_atr = Decimal('NaN')  # Set ATR to NaN if indicators failed critically
    else:
        current_atr = indicators.get('atr', Decimal('NaN'))  # Use NaN default if indicators is None or atr is NaN

    # 4. Get Current State (Balance & Positions as Decimals)
    # Fetch balance first
    quote_currency = MARKET_INFO.get('settle', 'USDT') if MARKET_INFO else 'USDT'  # Fallback currency
    free_balance, current_equity = get_balance(quote_currency)
    if current_equity is None or current_equity.is_nan():
        logger.error(Fore.RED + "Failed to fetch valid current balance/equity. Cannot perform risk calculation or trading actions.")
        # Don't proceed with trade actions without knowing equity
        cycle_success = False
        # Fall through to display panel (will show N/A equity)

    # Fetch positions (crucial state)
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Failed to fetch current positions. Cannot manage state or trade.")
        cycle_success = False
        # Fall through to display panel (will show N/A positions)

    # 5. Fetch Realized PnL Periodically
    total_realized_pnl: Optional[Decimal] = Decimal("NaN")
    if cycle_count == 1 or cycle_count % CONFIG.fetch_pnl_interval_cycles == 0:
         pnl_result = fetch_pnl_wrapper(CONFIG.symbol)
         if pnl_result is not None:
              total_realized_pnl = pnl_result.get('total_realized', Decimal('NaN'))
         else:
              logger.warning("Failed to fetch realized PnL.")
              total_realized_pnl = Decimal('NaN') # Ensure it's NaN if fetch failed
    else:
         logger.debug(f"Skipping realized PnL fetch this cycle ({cycle_count}). Next fetch in {CONFIG.fetch_pnl_interval_cycles - (cycle_count % CONFIG.fetch_pnl_interval_cycles)} cycles.")
         # We don't have the total realized PnL for this cycle unless fetched, so display NaN or '-'


    # --- Capture State Snapshot for Status Panel & Logic ---
    # Do this *before* potentially modifying state (like TSL management or entry)
    # Use deepcopy for the tracker to ensure the panel shows state before any potential updates in this cycle
    order_tracker_snapshot = copy.deepcopy(order_tracker)
    # Use the fetched positions directly as the snapshot (if fetch succeeded)
    positions_snapshot = positions if positions is not None else {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "pnl": Decimal("NaN"), "liq_price": Decimal("NaN")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "pnl": Decimal("NaN"), "liq_price": Decimal("NaN")}
    }

    # --- Logic continues only if critical data is available (positions and equity) ---
    # Note: We can still show the panel even if positions/equity fetch failed,
    # but we *cannot* perform trade actions or TSL management safely.
    can_trade_logic = (
        positions is not None and
        current_equity is not None and
        not current_equity.is_nan() and
        current_equity > Decimal('0')
    )

    # Initialize signals data for the panel even if trade logic is skipped
    signals = {"long": False, "short": False, "reason": "Trade logic skipped"}


    if can_trade_logic:
        # Use the *current* state from `positions` dict (not snapshot) for logic decisions
        active_long_pos = positions.get('long', {})
        active_short_pos = positions.get('short', {})
        active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
        active_short_qty = active_short_pos.get('qty', Decimal('0.0'))

        # Check if already have a significant position in either direction
        has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
        has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
        is_flat = not has_long_pos and not has_short_pos

        # 6. Manage Trailing Stops
        # Only attempt TSL management if indicators and current price are available AND there's a position
        # Ensure ATR is also valid
        if indicators is not None and not current_price.is_nan() and not current_atr.is_nan() and (has_long_pos or has_short_pos):
            if has_long_pos:
                logger.debug("Managing TSL for existing LONG position...")
                # Pass entry price from the fetched position
                long_entry_price = active_long_pos.get('entry_price', Decimal('NaN'))
                manage_trailing_stop(CONFIG.symbol, "long", active_long_qty, long_entry_price, current_price, current_atr)
            elif has_short_pos:
                logger.debug("Managing TSL for existing SHORT position...")
                # Pass entry price from the fetched position
                short_entry_price = active_short_pos.get('entry_price', Decimal('NaN'))
                manage_trailing_stop(CONFIG.symbol, "short", active_short_qty, short_entry_price, current_price, current_atr)
            # If flat, TSL management is skipped implicitly by the check (has_long_pos or has_short_pos)
        elif is_flat:
            # If flat, ensure trackers are clear (belt-and-suspenders check)
            if order_tracker["long"]["sl_id"] or order_tracker["long"]["tsl_id"] or \
               order_tracker["short"]["sl_id"] or order_tracker["short"]["tsl_id"]:
                logger.info("Position is flat, ensuring order trackers are cleared.")
                order_tracker["long"] = {"sl_id": None, "tsl_id": None}
                order_tracker["short"] = {"sl_id": None, "tsl_id": None}
                # Update the snapshot to reflect the clearing for the panel display
                order_tracker_snapshot["long"] = {"sl_id": None, "tsl_id": None}
                order_tracker_snapshot["short"] = {"sl_id": None, "tsl_id": None}
        else:
            logger.warning("Skipping TSL management due to missing indicators, invalid price, invalid ATR, or missing position data.")


        # 7. Generate Trading Signals
        # Signals only generated if indicators and current price are available, AND previous indicators are available for strategies needing crosses
        if indicators is not None and not current_price.is_nan():
            # Pass current AND previous indicators to signal generation
            signals_data = generate_signals(indicators, current_price, prev_indicators)
            signals = {"long": signals_data["long"], "short": signals_data["short"], "reason": signals_data["reason"]} # Keep reason
        else:
            logger.warning("Skipping signal generation due to missing indicators or invalid price.")
            signals = {"long": False, "short": False, "reason": "Skipped due to missing data"}


        # 8. Execute Trades based on Signals
        # Only attempt entry if currently flat, indicators/ATR are available and valid, and equity is sufficient
        if is_flat and indicators is not None and not current_atr.is_nan(): # Equity checked in can_trade_logic
            # Ensure required indicators for signals are not NaN before trading
            signal_indicators_valid = True
            # Define indicators required for trading based on the chosen strategy
            required_for_signal = ['trend_ema'] # Trend EMA is always used for filter
            if CONFIG.strategy == 'ecc_scalp':
                required_for_signal.extend(['ecc_cycle', 'ecc_trigger'])
                if prev_indicators is None:
                     logger.warning("Cannot attempt ECC trade entry: Previous indicators are required for cross detection.")
                     signal_indicators_valid = False
                else:
                     # Also check previous indicator values for NaN if needed for the cross logic
                     required_for_prev_signal = ['ecc_cycle', 'ecc_trigger']
                     for key in required_for_prev_signal:
                          if prev_indicators.get(key, Decimal('NaN')).is_nan():
                               logger.warning(f"Cannot attempt ECC trade entry: Required previous indicator '{key}' is NaN.")
                               signal_indicators_valid = False
                               break # No need to check further indicators

            elif CONFIG.strategy == 'ema_stoch':
                required_for_signal.extend(['fast_ema', 'slow_ema', 'stoch_k']) # Stoch D not strictly needed for signal logic here

            # Check current indicator values for NaN
            if signal_indicators_valid: # Only check current if previous check passed (if applicable)
                 for key in required_for_signal:
                      if indicators.get(key, Decimal('NaN')).is_nan():
                           logger.warning(f"Cannot attempt trade entry: Required current indicator '{key}' is NaN.")
                           signal_indicators_valid = False
                           break  # No need to check further indicators


            if signal_indicators_valid:
                trade_attempted = False
                if signals.get("long"):
                    logger.info(Fore.GREEN + Style.BRIGHT + f"Long signal detected! {signals.get('reason', '')}. Attempting entry...")
                    trade_attempted = True
                    # place_risked_market_order handles its own error logging and tracker updates
                    if place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr):
                        logger.info(f"Long entry process completed successfully for cycle {cycle_count}.")
                        # Re-fetch positions immediately after a successful entry to update state for the panel
                        # This helps ensure the panel reflects the new position quickly
                        logger.debug("Refetching positions after successful entry...")
                        positions = get_current_position(CONFIG.symbol)
                        # Update snapshot for panel
                        positions_snapshot = positions if positions is not None else positions_snapshot  # Use new positions if fetched, else keep old snapshot
                        # The order_tracker is updated inside place_risked_market_order on success.
                    else:
                        logger.error(f"Long entry process failed for cycle {cycle_count}. Check logs.")
                        # Optional: Implement cooldown logic here if needed

                elif signals.get("short"):
                    logger.info(Fore.RED + Style.BRIGHT + f"Short signal detected! {signals.get('reason', '')}. Attempting entry.")
                    trade_attempted = True
                    # place_risked_market_order handles its own error logging and tracker updates
                    if place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr):
                        logger.info(f"Short entry process completed successfully for cycle {cycle_count}.")
                        # Re-fetch positions immediately after a successful entry to update state for the panel
                        logger.debug("Refetching positions after successful entry...")
                        positions = get_current_position(CONFIG.symbol)
                        # Update snapshot for panel
                        positions_snapshot = positions if positions is not None else positions_snapshot  # Use new positions if fetched, else keep old snapshot
                        # The order_tracker is updated inside place_risked_market_order on success.
                    else:
                        logger.error(f"Short entry process failed for cycle {cycle_count}. Check logs.")
                        # Optional: Implement cooldown logic here if needed

                # If a trade was attempted, main loop sleep handles the pause.

            else:
                logger.warning("Skipping trade entry due to invalid indicator values.")

        elif not is_flat:
            pos_side = "LONG" if has_long_pos else "SHORT"
            logger.info(f"Position ({pos_side}) already open, skipping new entry signals.")
            # Future: Add exit logic based on counter-signals or other conditions if desired.
            # Example: if pos_side == "LONG" and signals.get("short"): close_position("long")
            # Example: if pos_side == "SHORT" and signals.get("long"): close_position("short")
        else:
            # This block is hit if can_trade_logic is False, meaning positions/equity fetch failed
            logger.warning("Skipping all trade logic (TSL management, signal generation, entry) due to earlier critical data failure (positions or equity).")
            signals = {"long": False, "short": False, "reason": "Skipped due to critical data failure"}  # Ensure signals are false for panel
            # indicators = None # Keep indicators data for the panel even if trade logic skipped

    else:
        # Cycle failed earlier (positions or equity fetch failed), skip trade logic entirely
        logger.warning("Skipping all trade logic (TSL management, signal generation, entry) due to earlier critical data failure (positions or equity).")
        signals = {"long": False, "short": False, "reason": "Skipped due to critical data failure"}  # Ensure signals are false for panel
        # indicators = None # Keep indicators data for the panel even if trade logic skipped


    # 9. Display Status Panel (Always display if data allows)
    # Use the state captured *before* TSL management and potential trade execution for consistency
    # unless the cycle failed very early (handled by the initial df check).
    # Pass the total_realized_pnl fetched earlier (might be NaN if fetch failed or skipped)
    print_status_panel(
        cycle_count, last_timestamp, current_price, indicators,
        positions_snapshot, current_equity, signals, total_realized_pnl, order_tracker  # Use the *latest* order_tracker state
    )

    end_time = time.time()
    status_log = "Complete" if cycle_success else "Completed with WARNINGS/ERRORS"
    logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} {status_log} (Duration: {end_time - start_time:.2f}s) ---")

    # Return current indicators to be used as prev_indicators in the next cycle
    return indicators, cycle_success


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
    market_id = MARKET_INFO.get('id')  # Exchange specific ID

    # 1. Cancel All Open Orders for the Symbol
    # This includes stop loss / take profit orders if they are separate entities (unlikely for Bybit V5 position stops)
    # and potentially limit orders if they were used for entry/exit (not in this strategy, but good practice).
    try:
        logger.info(Fore.CYAN + f"Dispelling all cancellable open orders for {symbol}...")
        # Bybit V5 fetch_open_orders requires category
        fetch_params = {'category': CONFIG.market_type}
        # Use fetch_with_retries for fetching open orders
        open_orders_list = fetch_with_retries(EXCHANGE.fetch_open_orders, symbol, params=fetch_params)

        if open_orders_list is None:
            logger.warning(Fore.YELLOW + "Failed to fetch open orders during shutdown. Cannot attempt cancellation.")
        elif not open_orders_list:
            logger.info("No cancellable open orders found via fetch_open_orders.")
        else:
            order_ids = [o.get('id', 'N/A') for o in open_orders_list]
            logger.info(f"Found {len(open_orders_list)} open orders to attempt cancellation: {', '.join(order_ids)}")

            try:
                # Bybit V5 cancel_all_orders requires category and symbol
                cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                # Use cancel_all_orders for efficiency if supported and reliable
                # Note: cancel_all_orders might not exist or work reliably for all exchanges/params
                # Fallback: loop through fetched open orders and cancel individually
                # Bybit V5 supports POST /v5/order/cancel-all
                if hasattr(EXCHANGE, 'private_post_order_cancel_all'):
                    logger.info(f"Using private_post_order_cancel_all for {symbol}...")
                    # Use fetch_with_retries for the cancel_all call
                    response = fetch_with_retries(EXCHANGE.private_post_order_cancel_all, params=cancel_params)

                    if response is None:
                        logger.warning(Fore.YELLOW + "Cancel all orders command failed after retries. MANUAL CHECK REQUIRED.")
                    elif isinstance(response, dict) and response.get('info', {}).get('retCode') == 0:
                        logger.info(Fore.GREEN + "Cancel all command successful (retCode 0).")
                    else:
                        error_msg = response.get('info', {}).get('retMsg', 'Unknown error') if isinstance(response, dict) else str(response)
                        logger.warning(Fore.YELLOW + f"Cancel all orders command sent, success confirmation unclear or failed: {error_msg}. MANUAL CHECK REQUIRED.")
                else:
                    # Fallback to individual cancellation if cancel_all is not supported or the specific method isn't found
                    logger.info("cancel_all_orders method not directly available or reliable, cancelling individually...")
                    cancelled_count = 0
                    for order in open_orders_list:
                        try:
                            order_id = order['id']
                            logger.debug(f"Cancelling order {order_id}...")
                            # Bybit V5 cancel_order requires category and symbol
                            individual_cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                            # Use fetch_with_retries for individual cancel
                            cancel_result = fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=individual_cancel_params)
                            if cancel_result:  # CCXT cancel usually returns order dict on success
                                logger.info(f"Cancel request sent for order {order_id}.")
                                cancelled_count += 1
                                time.sleep(0.2)  # Small delay between cancels
                            else:
                                logger.error(f"Failed to cancel order {order_id} after retries.")

                        except ccxt.OrderNotFound:
                            logger.warning(f"Order {order_id} already gone when attempting cancellation.")
                        except Exception as ind_cancel_err:
                            logger.error(f"Failed to cancel order {order_id}: {ind_cancel_err}")
                    logger.info(f"Attempted to cancel {cancelled_count}/{len(open_orders_list)} orders individually.")

            except Exception as cancel_err:
                logger.error(Fore.RED + f"Error sending cancel command(s): {cancel_err}. MANUAL CHECK REQUIRED.")

        # Clear local tracker regardless, as intent is to have no active tracked orders
        logger.info("Clearing local order tracker state.")
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error during order cancellation phase: {e}. MANUAL CHECK REQUIRED on exchange.", exc_info=True)

    # Add a small delay after cancelling orders before checking/closing positions
    logger.info("Waiting briefly after order cancellation before checking positions...")
    time.sleep(max(CONFIG.order_check_delay_seconds, 2))  # Wait at least 2 seconds

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function with retries
        positions = get_current_position(symbol)

        closed_count = 0
        if positions is None:
            # Failure to fetch positions during shutdown is critical
            logger.critical(Fore.RED + Style.BRIGHT + "Could not fetch final positions during shutdown. MANUAL CHECK REQUIRED on exchange!")
            termux_notify("Shutdown Warning!", f"{symbol} Cannot confirm position status. Check exchange!")
        else:  # positions fetch succeeded (might be empty list or dict with zero qtys)
            try:
                # Get minimum quantity for validation using Decimal
                min_qty_dec = Decimal(str(MARKET_INFO['limits']['amount']['min']))
            except (KeyError, InvalidOperation, TypeError):
                logger.warning("Could not determine minimum order quantity for closure validation.")
                min_qty_dec = Decimal("0")  # Assume zero if unavailable

            # Filter for positions with significant quantity
            fetched_positions_to_process = {}
            for side, pos_data in positions.items():
                # Ensure pos_data is a dict and has 'qty' key
                if isinstance(pos_data, dict) and 'qty' in pos_data:
                    qty = pos_data.get('qty', Decimal("0.0"))
                    if isinstance(qty, Decimal) and qty.copy_abs() >= CONFIG.position_qty_epsilon:
                        fetched_positions_to_process[side] = pos_data
                    elif not isinstance(qty, Decimal):
                        logger.warning(f"Position quantity for {side} is not a Decimal ({type(qty).__name__}). Skipping closure.")
                else:
                    logger.warning(f"Position data for {side} is missing or invalid format. Skipping closure.")

            if not fetched_positions_to_process:
                logger.info(Fore.GREEN + "No significant open positions found requiring closure.")
            else:
                logger.warning(Fore.YELLOW + f"Found {len(fetched_positions_to_process)} positions requiring closure.")

                for side, pos_data in fetched_positions_to_process.items():
                    qty = pos_data.get('qty', Decimal("0.0"))
                    entry_price = pos_data.get('entry_price', Decimal("NaN"))
                    close_side = "sell" if side == "long" else "buy"
                    logger.warning(Fore.YELLOW + f"Closing {side} position (Qty: {qty.normalize()}, Entry: {entry_price:.4f if not entry_price.is_nan() else 'N/A'}) with market order...")
                    try:
                        # Format quantity precisely for closure order (use absolute value and round down)
                        close_qty_str = format_amount(symbol, qty.copy_abs(), ROUND_DOWN)
                        try:
                            close_qty_decimal = Decimal(close_qty_str)
                        except InvalidOperation:
                            logger.critical(f"{Fore.RED}Failed to parse closure quantity '{close_qty_str}' for {side} position. Cannot attempt closure.")
                            termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS closure failed (qty parse)! Manual action required!")
                            continue  # Skip trying to close this position

                        # Validate against minimum quantity before attempting closure
                        if close_qty_decimal < min_qty_dec:
                            logger.critical(f"{Fore.RED}Closure quantity {close_qty_decimal.normalize()} for {side} position is below exchange minimum {min_qty_dec.normalize()}. MANUAL CLOSURE REQUIRED!")
                            termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS < MIN QTY! Close manually!")
                            continue  # Skip trying to close this position

                        # Place the closure market order
                        close_params = {'reduceOnly': True}  # Crucial: Only close, don't open new position
                        # fetch_with_retries handles category param
                        close_order = fetch_with_retries(
                            EXCHANGE.create_market_order,
                            symbol=symbol,
                            side=close_side,
                            amount=float(close_qty_decimal),  # CCXT needs float
                            params=close_params
                        )

                        # Check response for success
                        if close_order and (close_order.get('id') or close_order.get('info', {}).get('retCode') == 0):
                            close_id = close_order.get('id', 'N/A (retCode 0)')
                            logger.trade(Fore.GREEN + f"Position closure order placed successfully: ID {close_id}")
                            closed_count += 1
                            # Wait briefly to allow fill confirmation before checking next position (if any)
                            time.sleep(max(CONFIG.order_check_delay_seconds, 2))
                            # Optional: Verify closure order status? Might slow shutdown significantly.
                            # For shutdown, placing the order is usually sufficient as market orders fill fast.
                        else:
                            # Log critical error if closure order placement fails
                            error_msg = close_order.get('info', {}).get('retMsg', 'No ID and no success code.') if isinstance(close_order, dict) else str(close_order)
                            logger.critical(Fore.RED + Style.BRIGHT + f"FAILED TO PLACE closure order for {side} position ({qty.normalize()}): {error_msg}. MANUAL INTERVENTION REQUIRED!")
                            termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")

                    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied) as e:
                        logger.critical(Fore.RED + Style.BRIGHT + f"FAILED TO CLOSE {side} position ({qty.normalize()}): {e}. MANUAL INTERVENTION REQUIRED!")
                        termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")
                    except Exception as e:
                        logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error closing {side} position: {e}. MANUAL INTERVENTION REQUIRED!", exc_info=True)
                        termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")

            # Final summary message
            if closed_count == len(fetched_positions_to_process):
                logger.info(Fore.GREEN + f"Successfully placed closure orders for all {closed_count} detected positions.")
            elif closed_count > 0:
                logger.warning(Fore.YELLOW + f"Placed closure orders for {closed_count} positions, but {len(fetched_positions_to_process) - closed_count} positions may remain. MANUAL CHECK REQUIRED.")
                termux_notify("Shutdown Warning!", f"{symbol} Manual check needed - {len(fetched_positions_to_process) - closed_count} positions might remain.")
            else:
                if len(fetched_positions_to_process) > 0:
                    logger.warning(Fore.YELLOW + "Attempted shutdown but closure orders failed or were not possible for all open positions. MANUAL CHECK REQUIRED.")
                    termux_notify("Shutdown Warning!", f"{symbol} Manual check needed - positions might remain.")

    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Error during position closure phase: {e}. Manual check advised.", exc_info=True)
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Error during position closure. Check logs.")

    logger.warning(Fore.YELLOW + Style.BRIGHT + "Graceful Shutdown Sequence Complete.")
    termux_notify("Shutdown Complete", f"{CONFIG.symbol} bot stopped.")


# --- Main Spell Invocation ---
if __name__ == "__main__":
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80)
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Termux Trading Spell Activated (v2.3 Integrated ECC & PnL) ***")
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL)

    logger.info(f"Initializing Pyrmethus v2.3...")
    logger.info(f"Log Level configured to: {log_level_str}")

    # Log key configuration parameters for verification
    logger.info(f"--- Trading Configuration ---")
    logger.info(f"Symbol: {CONFIG.symbol} ({CONFIG.market_type.capitalize()})")
    logger.info(f"Timeframe: {CONFIG.interval}")
    logger.info(f"Strategy: {CONFIG.strategy.upper()}")
    if CONFIG.strategy == 'ecc_scalp':
         logger.info(f"  ECC Parameters: Alpha={CONFIG.ecc_alpha}, Lookback={CONFIG.ecc_lookback}")
    elif CONFIG.strategy == 'ema_stoch':
         logger.info(f"  EMA Parameters: Fast={CONFIG.ema_fast_period}, Slow={CONFIG.ema_slow_period}")
         logger.info(f"  Stoch Parameters: Period={CONFIG.stoch_period}, Smooth K={CONFIG.stoch_smooth_k}, Smooth D={CONFIG.stoch_smooth_d}")
    logger.info(f"Risk per trade: {CONFIG.risk_percentage * 100:.5f}%")  # Show more precision for risk
    logger.info(f"SL Multiplier: {CONFIG.sl_atr_multiplier} * ATR ({CONFIG.atr_period} periods)")
    logger.info(f"TSL Activation: {CONFIG.tsl_activation_atr_multiplier} * ATR Profit")
    logger.info(f"TSL Trail Percent: {CONFIG.trailing_stop_percent}%")
    logger.info(f"Trigger Prices: SL={CONFIG.sl_trigger_by}, TSL={CONFIG.tsl_trigger_by}")
    logger.info(f"Trend Filter EMA({CONFIG.trend_ema_period}): {CONFIG.trade_only_with_trend}")
    logger.info(f"PNL Fetch Interval: Every {CONFIG.fetch_pnl_interval_cycles} cycles ({CONFIG.pnl_lookback_days} days lookback)")
    logger.info(f"Position Quantity Epsilon: {CONFIG.position_qty_epsilon:.2E}")  # Scientific notation
    logger.info(f"Loop Interval: {CONFIG.loop_sleep_seconds}s")
    logger.info(f"OHLCV Limit: {CONFIG.ohlcv_limit}")
    logger.info(f"Fetch Retries: {CONFIG.max_fetch_retries}")
    logger.info(f"Order Check Timeout: {CONFIG.order_check_timeout_seconds}s")
    logger.info(f"-----------------------------")

    # Final check if exchange connection and market info loading succeeded
    if MARKET_INFO and EXCHANGE:
        termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2.3)")
        logger.info(Fore.GREEN + Style.BRIGHT + f"Initialization complete. Awaiting market whispers...")
        print(Fore.MAGENTA + "=" * 80 + Style.RESET_ALL)  # Separator before first cycle log
    else:
        # Error should have been logged during init, exit was likely called, but double-check.
        logger.critical(Fore.RED + Style.BRIGHT + "Exchange or Market info failed to load during initialization. Cannot start trading loop.")
        sys.exit(1)

    cycle = 0
    prev_indicators: Optional[Dict[str, Decimal]] = None # Store indicators from the previous cycle for cross detection

    try:
        while True:
            cycle += 1
            try:
                # Pass previous indicators and receive current indicators
                current_indicators, cycle_ok = trading_spell_cycle(cycle, prev_indicators)
                # Update prev_indicators for the next cycle if the current cycle was successful
                if cycle_ok and current_indicators is not None:
                     prev_indicators = current_indicators
                elif not cycle_ok:
                     # If the cycle failed critically (e.g., no market data), reset previous indicators
                     # This prevents using potentially stale data if the market fetch failed
                     logger.warning("Cycle failed critically. Resetting previous indicators.")
                     prev_indicators = None
                # If indicators were calculated but had NaNs, current_indicators will reflect that,
                # and signal generation will handle it. We still pass them as prev_indicators.

            except Exception as cycle_error:
                # Catch errors *within* a cycle to prevent the whole script from crashing
                logger.error(Fore.RED + Style.BRIGHT + f"Error during trading cycle {cycle}: {cycle_error}", exc_info=True)
                termux_notify("Cycle Error!", f"{CONFIG.symbol} Cycle {cycle} failed. Check logs.")
                # Decide if a single cycle failure is fatal. For now, log and continue to the next cycle after sleep.
                # If errors are persistent, fetch_with_retries/other checks should eventually halt.
                # If a cycle fails, we should probably reset prev_indicators to be safe, as current data might be inconsistent.
                logger.warning("Cycle failed unexpectedly. Resetting previous indicators.")
                prev_indicators = None


            logger.info(Fore.BLUE + f"Cycle {cycle} finished. Resting for {CONFIG.loop_sleep_seconds} seconds...")
            time.sleep(CONFIG.loop_sleep_seconds)

    except KeyboardInterrupt:
        logger.warning(Fore.YELLOW + "\nCtrl+C detected! Initiating graceful shutdown...")
        graceful_shutdown()
    except Exception as e:
        # Catch unexpected errors in the main loop *outside* of the trading_spell_cycle call
        logger.critical(Fore.RED + Style.BRIGHT + f"\nFATAL RUNTIME ERROR in Main Loop (Cycle {cycle}): {e}", exc_info=True)
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
        # Explicitly close handlers if necessary (StreamHandler usually flushes on exit)
        # for handler in logger.handlers:
        #      handler.flush()
        logging.shutdown()  # This should handle flushing and closing handlers
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80)
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Trading Spell Deactivated ***")
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL)

```
