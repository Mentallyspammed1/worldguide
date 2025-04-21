#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.2.1 (Fortified Configuration & Enhanced Clarity/Robustness)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.2.1 (Unified: Selectable Strategies + Precision + Native SL/TSL + Fortified Config + Pyrmethus Enhancements + Robustness)

Purpose:
Automates scalping strategies on Bybit USDT Perpetual Futures markets. This script is intended
for educational and experimental purposes, demonstrating concepts like API interaction,
indicator calculation, risk management, and automated order placement.

Key Features:
- Strategy Flexibility: Select from multiple trading strategies via configuration:
    - "DUAL_SUPERTREND": Uses two Supertrend indicators for trend confirmation.
    - "STOCHRSI_MOMENTUM": Combines Stochastic RSI for overbought/oversold signals with a Momentum indicator.
    - "EHLERS_FISHER": Implements the Ehlers Fisher Transform for identifying cyclical turning points.
    - "EHLERS_MA_CROSS": Uses Exponential Moving Average crossovers (Note: Placeholder for true Ehlers Super Smoother).
- Enhanced Precision: Leverages Python's `Decimal` type for critical financial calculations, minimizing floating-point inaccuracies.
- Fortified Configuration: Robust loading of settings from environment variables (.env file) with strict type casting and validation for improved reliability.
- Native Stop-Loss & Trailing Stop-Loss: Utilizes Bybit V5 API's exchange-native Stop Loss (fixed, ATR-based) and Trailing Stop Loss capabilities, placed immediately upon position entry for faster reaction times.
- Volatility Adaptation: Employs the Average True Range (ATR) indicator to measure market volatility and dynamically adjust the initial Stop Loss distance.
- Optional Confirmation Filters: Includes optional filters based on Volume Spikes (relative to a moving average) and Order Book Pressure (Bid/Ask volume ratio) to potentially improve entry signal quality.
- Sophisticated Risk Management: Implements risk-based position sizing (percentage of equity per trade), incorporates exchange margin requirements checks with a configurable buffer, and allows setting a maximum position value cap (USDT).
- Termux Integration: Provides optional SMS alerts via Termux:API for critical events like initialization, errors, order placements, and shutdowns. Includes checks for command availability.
- Robust Operation: Features comprehensive error handling for common CCXT exceptions (network issues, authentication failures, rate limits, exchange errors), data validation (NaN handling), and detailed logging with vibrant console colors via Colorama.
- Graceful Shutdown: Designed to handle interruptions (Ctrl+C) or critical errors by attempting to cancel open orders and close any existing positions before exiting.
- Bybit V5 API Focused: Tailored logic for interacting with the Bybit V5 API, particularly regarding position detection (One-Way Mode) and order parameters.

Disclaimer:
- **EXTREME RISK**: Trading cryptocurrencies, especially futures contracts with leverage and automated systems, involves substantial risk of financial loss. This script is provided for EDUCATIONAL PURPOSES ONLY. You could lose your entire investment and potentially more. Use this software entirely at your own risk. The authors and contributors assume NO responsibility for any trading losses.
- **NATIVE SL/TSL RELIANCE**: The bot's protective stop mechanisms rely entirely on Bybit's exchange-native Stop Loss and Trailing Stop Loss order execution. Their performance is subject to exchange conditions, potential slippage during volatile periods, API reliability, order book liquidity, and specific exchange rules. These orders are NOT GUARANTEED to execute at the precise trigger price specified.
- **PARAMETER SENSITIVITY & OPTIMIZATION**: The performance of this bot is highly dependent on the chosen strategy parameters (indicator settings, risk levels, SL/TSL percentages, filter thresholds). These parameters require extensive backtesting, optimization, and forward testing on a TESTNET environment before considering any live deployment. Default parameters are unlikely to be profitable.
- **API RATE LIMITS & BANS**: Excessive API requests can lead to temporary or permanent bans from the exchange. Monitor API usage and adjust script timing (`SLEEP_SECONDS`) accordingly. CCXT's built-in rate limiter is enabled but may not prevent all issues under heavy load.
- **SLIPPAGE**: Market orders, used for entry and potentially for SL/TSL execution by the exchange, are susceptible to slippage. This means the actual execution price may differ from the price observed when the order was placed, especially during high volatility or low liquidity.
- **TEST THOROUGHLY**: **DO NOT RUN THIS SCRIPT WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTING ON A TESTNET OR DEMO ACCOUNT.** Ensure you fully understand every part of the code, its logic, and its potential risks before any live deployment.
- **TERMUX DEPENDENCY**: SMS alert functionality requires a Termux environment on an Android device with the Termux:API package installed (`pkg install termux-api`). Ensure it is correctly installed and configured if you enable SMS alerts.
- **API & LIBRARY UPDATES**: This script targets the Bybit V5 API via the CCXT library. Future updates to the exchange API or the CCXT library may introduce breaking changes that require code modifications. Keep CCXT updated (`pip install -U ccxt`).
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import sys
import time
import traceback
import subprocess
import shlex # For safe command argument parsing
import shutil # For checking command existence (e.g., termux-sms-send)
from typing import Dict, Optional, Any, Tuple, List, Union
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation, DivisionByZero

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta # type: ignore[import] # pandas_ta might lack complete type hints
    from dotenv import load_dotenv
    from colorama import init as colorama_init, Fore, Style, Back
except ImportError as e:
    missing_pkg = e.name
    # Use Colorama's raw codes here as it might not be initialized yet
    print(f"\033[91mMissing essential spell component: \033[1m{missing_pkg}\033[0m") # Bright Red
    print("\033[93mTo conjure it, cast the following spell in your Termux terminal:\033[0m") # Bright Yellow
    print(f"\033[1m\033[96mpip install {missing_pkg}\033[0m") # Bold Bright Cyan
    print("\n\033[96mOr, to ensure all scrolls are present, cast:\033[0m") # Bright Cyan
    print("\033[1m\033[96mpip install ccxt pandas pandas_ta python-dotenv colorama\033[0m")
    print("\033[93mYou may also need to verify dependencies with: \033[1m\033[96mpip list\033[0m")
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True) # Activate Colorama's magic for vibrant logs
load_dotenv() # Load secrets from the hidden .env scroll (if present)
getcontext().prec = 18 # Set Decimal precision for financial exactitude (adjust if needed, 18 is often sufficient)

# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """
    Loads, validates, and stores configuration parameters from environment variables.
    Provides robust type casting and default value handling.
    """
    def __init__(self):
        """Initializes configuration by loading and validating environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED, secret=True)
        self.api_secret: Optional[str] = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED, secret=True)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW) # Target market (CCXT unified format, e.g., 'BTC/USDT:USDT')
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW) # Chart timeframe (e.g., '1m', '5m', '1h')
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW) # Desired leverage multiplier
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW) # Pause between trading cycles (seconds)

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: List[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}")
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}")
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.005 = 0.5% of equity per trade
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN) # Multiplier for ATR to set initial fixed SL distance
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN) # Maximum position value in USDT (overrides risk calc if needed)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN) # e.g., 1.05 = Require 5% more free margin than estimated for order placement

        # --- Trailing Stop Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.005 = 0.5% trailing distance from high/low water mark
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.001 = 0.1% price movement in profit before TSL becomes active

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN) # Primary Supertrend ATR period
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN) # Primary Supertrend ATR multiplier
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN) # Confirmation Supertrend ATR period
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN) # Confirmation Supertrend ATR multiplier
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN) # StochRSI: RSI period
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN) # StochRSI: Stochastic period
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN) # StochRSI: %K smoothing period
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN) # StochRSI: %D smoothing period (signal line)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN) # StochRSI overbought threshold
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN) # StochRSI oversold threshold
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN) # Momentum indicator period
        # Ehlers Fisher Transform
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN) # Fisher Transform calculation period
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN) # Fisher Transform signal line period (1 usually means no separate signal line smoothing)
        # Ehlers MA Cross (Using EMA Placeholder - requires verification/replacement for true Ehlers MA)
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN) # Fast EMA period (placeholder)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN) # Slow EMA period (placeholder)

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        # Volume Analysis
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW) # Moving average period for volume
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW) # Multiplier over MA to consider a 'spike' (e.g., 1.5 = 150% of MA)
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW) # Require volume spike for entry signal?
        # Order Book Analysis
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW) # Number of bid/ask levels to analyze for ratio
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW) # Min Bid/Ask volume ratio for long confirmation
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW) # Max Bid/Ask volume ratio for short confirmation (ratio = Total Bid Vol / Total Ask Vol within depth)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW) # Fetch OB every cycle (more API calls) or only when needed for entry confirmation?

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN) # Period for ATR calculation used in SL

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA) # Enable/disable SMS alerts globally
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA, required=False) # Recipient phone number for alerts (optional)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA) # Max time to wait for SMS command execution (seconds)

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = self._get_env("CCXT_RECV_WINDOW", 10000, cast_type=int, color=Fore.WHITE) # Milliseconds for API request validity (Bybit default 5000, increased for potential latency)
        self.order_book_fetch_limit: int = max(25, self.order_book_depth) # How many levels to fetch (ensure >= depth needed, common limits are 25, 50, 100, 200)
        self.shallow_ob_fetch_depth: int = 5 # Depth for quick price estimates (used in order placement estimate)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW) # Max time to wait for market order fill confirmation (seconds)

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy: str = "buy"       # CCXT standard side for buying
        self.side_sell: str = "sell"     # CCXT standard side for selling
        self.pos_long: str = "Long"      # Internal representation for a long position
        self.pos_short: str = "Short"    # Internal representation for a short position
        self.pos_none: str = "None"      # Internal representation for no position (flat)
        self.usdt_symbol: str = "USDT"   # The stablecoin quote currency symbol used by Bybit
        self.retry_count: int = 3        # Default attempts for certain retryable API calls (e.g., setting leverage)
        self.retry_delay_seconds: int = 2 # Default pause between retries (seconds)
        self.api_fetch_limit_buffer: int = 10 # Extra candles to fetch beyond strict indicator needs, providing a safety margin
        self.position_qty_epsilon: Decimal = Decimal("1e-9") # Small value for float/decimal comparisons involving position size to handle precision issues
        self.post_close_delay_seconds: int = 3 # Brief pause after successfully closing a position (seconds) to allow exchange state to potentially settle

        # --- Post-Initialization Validation ---
        self._validate_parameters()

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    def _validate_parameters(self):
        """Performs basic validation checks on loaded parameters."""
        if self.leverage <= 0:
            raise ValueError("LEVERAGE must be a positive integer.")
        if self.risk_per_trade_percentage <= 0 or self.risk_per_trade_percentage >= 1:
            raise ValueError("RISK_PER_TRADE_PERCENTAGE must be between 0 and 1 (exclusive).")
        if self.atr_stop_loss_multiplier <= 0:
            raise ValueError("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if self.trailing_stop_percentage <= 0 or self.trailing_stop_percentage >= 1:
            raise ValueError("TRAILING_STOP_PERCENTAGE must be between 0 and 1 (exclusive).")
        if self.trailing_stop_activation_offset_percent < 0:
             raise ValueError("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT cannot be negative.")
        if self.max_order_usdt_amount < 0:
             raise ValueError("MAX_ORDER_USDT_AMOUNT cannot be negative.")
        if self.required_margin_buffer < 1:
            raise ValueError("REQUIRED_MARGIN_BUFFER must be >= 1.")
        if self.enable_sms_alerts and not self.sms_recipient_number:
             logger.warning(f"{Fore.YELLOW}SMS alerts enabled (ENABLE_SMS_ALERTS=true) but SMS_RECIPIENT_NUMBER is not set. Alerts will not be sent.{Style.RESET_ALL}")
        # Add more validation as needed for strategy parameters, etc.

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE, secret: bool = False) -> Any:
        """
        Fetches an environment variable, performs robust type casting (including defaults),
        logs the process, handles required variables, and masks secrets in logs.

        Args:
            key: The environment variable name.
            default: The default value to use if the variable is not set.
            cast_type: The target type to cast the value to (e.g., int, Decimal, bool, str).
            required: If True, raises ValueError if the variable is not set and no default is provided.
            color: Colorama Fore color for logging this parameter.
            secret: If True, masks the value in log messages.

        Returns:
            The value from the environment variable or default, cast to the specified type.

        Raises:
            ValueError: If a required variable is missing or if casting fails critically.
        """
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None

        log_value = lambda v: "*******" if secret and v is not None else v

        if value_str is None:
            if required and default is None: # Changed logic: Required is only critical if default is also None
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' not found and no default specified.{Style.RESET_ALL}")
                raise ValueError(f"Required environment variable '{key}' not set and no default provided.")
            elif required and default is not None:
                logger.warning(f"{color}Required rune {key}: Not Set. Using Required Default: '{log_value(default)}'{Style.RESET_ALL}")
                value_to_cast = default
                source = "Required Default"
            elif not required:
                 logger.debug(f"{color}Summoning {key}: Not Set. Using Default: '{log_value(default)}'{Style.RESET_ALL}")
                 value_to_cast = default
                 source = "Default"
        else:
            logger.debug(f"{color}Summoning {key}: Found Env Value: '{log_value(value_str)}'{Style.RESET_ALL}")
            value_to_cast = value_str

        # --- Attempt Casting (applies to both env var value and default value) ---
        if value_to_cast is None:
            # Handles cases where default=None or env var was explicitly empty and default was None
            if required: # This case should now be caught above if default was also None
                 logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' resolved to None unexpectedly.{Style.RESET_ALL}")
                 raise ValueError(f"Required environment variable '{key}' resolved to None.")
            else:
                logger.debug(f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}")
                return None

        final_value: Any = None
        try:
            raw_value_str = str(value_to_cast) # Ensure string representation for casting logic
            if cast_type == bool:
                final_value = raw_value_str.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                # Cast via Decimal to handle potential float strings like "10.0" -> 10 gracefully
                final_value = int(Decimal(raw_value_str))
            elif cast_type == float:
                 final_value = float(raw_value_str)
            elif cast_type == str:
                final_value = raw_value_str # Keep as string
            else:
                logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw value.")
                final_value = value_to_cast # Return original value if type is unknown

        except (ValueError, TypeError, InvalidOperation) as e:
            # Casting failed! Log error and attempt to use default, casting it carefully.
            logger.error(f"{Fore.RED}Invalid type/value for {key}: '{log_value(value_to_cast)}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Attempting to use default '{log_value(default)}'.{Style.RESET_ALL}")
            if default is None:
                 if required:
                     logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast value for required key '{key}' and default is None.{Style.RESET_ALL}")
                     raise ValueError(f"Required env var '{key}' failed casting and has no valid default.")
                 else:
                     logger.warning(f"{Fore.YELLOW}Casting failed for {key}, default is None. Final value: None{Style.RESET_ALL}")
                     return None
            else:
                # Try casting the default value itself
                source = "Default (Fallback)"
                logger.debug(f"Attempting to cast fallback default value '{log_value(default)}' for key '{key}' to {cast_type.__name__}")
                try:
                    default_str = str(default)
                    if cast_type == bool: final_value = default_str.lower() in ['true', '1', 'yes', 'y', 'on']
                    elif cast_type == Decimal: final_value = Decimal(default_str)
                    elif cast_type == int: final_value = int(Decimal(default_str))
                    elif cast_type == float: final_value = float(default_str)
                    elif cast_type == str: final_value = default_str
                    else: final_value = default # Fallback to raw default if type unknown

                    logger.warning(f"{Fore.YELLOW}Successfully used casted default value for {key}: '{log_value(final_value)}'{Style.RESET_ALL}")
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast BOTH provided value ('{log_value(value_to_cast)}') AND default value ('{log_value(default)}') for key '{key}' to {cast_type.__name__}. Error on default: {e_default}{Style.RESET_ALL}")
                    raise ValueError(f"Configuration error: Cannot cast value or default for key '{key}' to {cast_type.__name__}.")

        # Log the final type and value being used
        logger.debug(f"{color}Using final value for {key}: {log_value(final_value)} (Type: {type(final_value).__name__}) (Source: {source}){Style.RESET_ALL}")
        return final_value


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL_STR: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL_MAP: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": 25, # Custom level
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Set default level to INFO if the env var value is invalid
LOGGING_LEVEL: int = LOGGING_LEVEL_MAP.get(LOGGING_LEVEL_STR, logging.INFO)

# Define custom SUCCESS level
SUCCESS_LEVEL: int = LOGGING_LEVEL_MAP["SUCCESS"]
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# Configure basic logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)] # Output to console
)
logger: logging.Logger = logging.getLogger(__name__) # Get the root logger

# Define the success method for the logger instance
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a message with severity 'SUCCESS'."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)
# Add the method to the Logger class (careful with type hinting here)
logging.Logger.success = log_success # type: ignore[attr-defined]

# Apply colors if outputting to a TTY (like Termux console or standard terminal)
if sys.stdout.isatty():
    # Define color mappings for levels
    level_colors = {
        logging.DEBUG: f"{Fore.CYAN}{Style.DIM}",
        logging.INFO: f"{Fore.BLUE}",
        SUCCESS_LEVEL: f"{Fore.MAGENTA}{Style.BRIGHT}",
        logging.WARNING: f"{Fore.YELLOW}{Style.BRIGHT}",
        logging.ERROR: f"{Fore.RED}{Style.BRIGHT}",
        logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}"
    }
    # Apply colors to level names
    for level, color_style in level_colors.items():
        level_name = logging.getLevelName(level)
        logging.addLevelName(level, f"{color_style}{level_name}{Style.RESET_ALL}")
else:
    # If not a TTY, ensure SUCCESS level name is still registered without color codes
    logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config() # Forge the configuration object from environment variables
except ValueError as config_error:
    # Error should have been logged within Config init or _get_env
    logger.critical(f"{Back.RED}{Fore.WHITE}Configuration loading failed. Cannot continue spellcasting. Error: {config_error}{Style.RESET_ALL}")
    # Attempt SMS alert if possible (basic config might be loaded for SMS settings)
    if os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true" and os.getenv("SMS_RECIPIENT_NUMBER"):
        try:
            # Manually construct minimal needed parts for alert
            temp_config_for_sms = type('obj', (object,), {
                'enable_sms_alerts': True,
                'sms_recipient_number': os.getenv("SMS_RECIPIENT_NUMBER"),
                'sms_timeout_seconds': int(os.getenv("SMS_TIMEOUT_SECONDS", "30"))
            })()
            # Need to temporarily assign to CONFIG for send_sms_alert to work
            _original_config = globals().get('CONFIG')
            globals()['CONFIG'] = temp_config_for_sms
            send_sms_alert(f"[Pyrmethus] CRITICAL CONFIG ERROR: {config_error}. Bot failed to start.")
            globals()['CONFIG'] = _original_config # Restore original (likely None)
        except Exception as sms_err:
            logger.error(f"Failed to send SMS alert about config error: {sms_err}")
    sys.exit(1)
except Exception as general_config_error:
    # Catch any other unexpected error during config initialization
    logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected critical error during configuration loading: {general_config_error}{Style.RESET_ALL}")
    logger.debug(traceback.format_exc())
    # Attempt SMS alert similarly
    if os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true" and os.getenv("SMS_RECIPIENT_NUMBER"):
         try:
            temp_config_for_sms = type('obj', (object,), {
                'enable_sms_alerts': True,
                'sms_recipient_number': os.getenv("SMS_RECIPIENT_NUMBER"),
                'sms_timeout_seconds': int(os.getenv("SMS_TIMEOUT_SECONDS", "30"))
            })()
            _original_config = globals().get('CONFIG')
            globals()['CONFIG'] = temp_config_for_sms
            send_sms_alert(f"[Pyrmethus] UNEXPECTED CONFIG ERROR: {type(general_config_error).__name__}. Bot failed.")
            globals()['CONFIG'] = _original_config
         except Exception as sms_err:
            logger.error(f"Failed to send SMS alert about unexpected config error: {sms_err}")
    sys.exit(1)

# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """
    Safely converts a value to a Decimal, handling None and potential errors.

    Args:
        value: The value to convert (can be string, float, int, Decimal, None, etc.).
        default: The Decimal value to return if conversion fails or input is None.

    Returns:
        The converted Decimal value or the default.
    """
    if value is None:
        return default
    try:
        # Using str(value) handles various input types more reliably before Decimal conversion
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as e:
        logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}. Error: {e}")
        return default

def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """Returns the last 6 characters of an order ID for concise logging, or 'N/A'."""
    if order_id:
        order_id_str = str(order_id)
        # Handle potential UUIDs or other long IDs, take last 6 chars
        return f"...{order_id_str[-6:]}" if len(order_id_str) > 6 else order_id_str
    return "N/A"

# --- Precision Formatting - Shaping the Numbers for the Exchange ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal]) -> str:
    """
    Formats a price according to the exchange's market precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        price: The price value (float or Decimal).

    Returns:
        The price formatted as a string according to market precision.
        Returns a normalized Decimal string as fallback on error.
    """
    try:
        # Ensure the market is loaded
        if symbol not in exchange.markets:
            logger.warning(f"Market {symbol} not loaded in format_price. Attempting to load.")
            exchange.load_markets()
        if symbol not in exchange.markets:
             raise ccxt.BadSymbol(f"Market {symbol} could not be loaded for formatting.")

        # CCXT formatting methods typically expect float input
        price_float = float(price)
        formatted_price = exchange.price_to_precision(symbol, price_float)
        # Extra check: Ensure the formatted price isn't zero if the input wasn't,
        # which could happen with extremely small prices and precision rules.
        if Decimal(formatted_price) == 0 and Decimal(str(price)) != 0:
             logger.warning(f"Price formatting resulted in zero for non-zero input {price}. Using Decimal normalize.")
             return str(Decimal(str(price)).normalize()) # Use normalized Decimal string
        return formatted_price
    except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
        logger.error(f"{Fore.RED}Error shaping price {price} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}")
        # Fallback: return a normalized string representation of the Decimal
        return str(Decimal(str(price)).normalize())
    except Exception as e_unexp:
        logger.error(f"{Fore.RED}Unexpected error shaping price {price} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return str(Decimal(str(price)).normalize())


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal]) -> str:
    """
    Formats an amount (quantity) according to the exchange's market precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        amount: The amount value (float or Decimal).

    Returns:
        The amount formatted as a string according to market precision.
        Returns a normalized Decimal string as fallback on error.
    """
    try:
        # Ensure the market is loaded
        if symbol not in exchange.markets:
            logger.warning(f"Market {symbol} not loaded in format_amount. Attempting to load.")
            exchange.load_markets()
        if symbol not in exchange.markets:
             raise ccxt.BadSymbol(f"Market {symbol} could not be loaded for formatting.")

        # CCXT formatting methods typically expect float input
        amount_float = float(amount)
        formatted_amount = exchange.amount_to_precision(symbol, amount_float)
         # Extra check: Ensure the formatted amount isn't zero if the input wasn't
        if Decimal(formatted_amount) == 0 and Decimal(str(amount)) != 0:
             logger.warning(f"Amount formatting resulted in zero for non-zero input {amount}. Using Decimal normalize.")
             return str(Decimal(str(amount)).normalize())
        return formatted_amount
    except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
        logger.error(f"{Fore.RED}Error shaping amount {amount} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}")
        # Fallback: return a normalized string representation of the Decimal
        return str(Decimal(str(amount)).normalize())
    except Exception as e_unexp:
        logger.error(f"{Fore.RED}Unexpected error shaping amount {amount} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return str(Decimal(str(amount)).normalize())

# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: Optional[bool] = None # Cache the result of checking command existence

def send_sms_alert(message: str) -> bool:
    """
    Sends an SMS alert using the 'termux-sms-send' command, if enabled and available.

    Checks for command existence once and caches the result.

    Args:
        message: The text message to send.

    Returns:
        True if the SMS command was executed successfully (return code 0), False otherwise.
    """
    global _termux_sms_command_exists

    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled by configuration.")
        return False

    # Check for command existence only once per script run
    if _termux_sms_command_exists is None:
        termux_command_path = shutil.which('termux-sms-send')
        _termux_sms_command_exists = termux_command_path is not None
        if not _termux_sms_command_exists:
             logger.warning(f"{Fore.YELLOW}SMS alerts enabled, but 'termux-sms-send' command not found in PATH. "
                            f"Ensure Termux:API is installed (`pkg install termux-api`) and PATH is configured correctly.{Style.RESET_ALL}")
        else:
             logger.debug(f"Found 'termux-sms-send' command at: {termux_command_path}")

    if not _termux_sms_command_exists:
        return False # Don't proceed if command is missing

    if not CONFIG.sms_recipient_number:
        # Warning already logged during config validation if number is missing while enabled
        logger.debug("SMS recipient number not configured, cannot send alert.")
        return False

    try:
        # Prepare the command spell. The message should be the last argument(s).
        # No special quoting needed by termux-sms-send usually, it takes the rest as the message.
        command: List[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Dispatching SMS whisper to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}")
        logger.debug(f"Executing command: {' '.join(shlex.quote(arg) for arg in command)}") # Log the command safely

        # Execute the spell via subprocess with timeout and output capture
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,          # Decode stdout/stderr as text
            check=False,        # Don't raise exception on non-zero exit code
            timeout=CONFIG.sms_timeout_seconds
        )

        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}")
            if result.stdout: logger.debug(f"SMS Send stdout: {result.stdout.strip()}")
            return True
        else:
            # Log error details from stderr if available
            error_details = result.stderr.strip() if result.stderr else "No stderr output"
            logger.error(f"{Fore.RED}SMS whisper failed. Return Code: {result.returncode}, Stderr: {error_details}{Style.RESET_ALL}")
            if result.stdout: logger.error(f"SMS Send stdout (on error): {result.stdout.strip()}")
            return False
    except FileNotFoundError:
        # This shouldn't happen due to the check above, but handle defensively
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' command vanished unexpectedly? Ensure Termux:API is installed.{Style.RESET_ALL}")
        _termux_sms_command_exists = False # Update cache
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: Command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected disturbance during dispatch: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False

# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """
    Initializes and returns the CCXT Bybit exchange instance.

    Handles authentication, loads markets, performs basic connectivity checks,
    and configures necessary options for Bybit V5 API interaction.

    Returns:
        A configured CCXT Bybit exchange instance, or None if initialization fails.
    """
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        # This should technically be caught by Config validation, but double-check
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret runes missing. Cannot open portal.{Style.RESET_ALL}")
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing. Spell failed.")
        return None
    try:
        # Forging the connection with Bybit V5 defaults
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True, # Respect the exchange spirits' limits
            "options": {
                "defaultType": "linear", # Assume USDT perpetuals unless symbol specifies otherwise
                "adjustForTimeDifference": True, # Sync client time with server time for request validity
                # V5 API specific options might be added here if needed, but CCXT handles most.
                # Example: Set default category if needed globally (though usually better per-call)
                # 'defaultCategory': 'linear',
            },
            # Explicitly set API version if CCXT default changes or issues arise
            # 'options': {'api-version': 'v5'}, # Uncomment if explicit V5 needed, CCXT usually handles it
            'recvWindow': CONFIG.default_recv_window, # Set custom receive window
        })

        # --- Testnet Configuration ---
        # Uncomment the following line to use Bybit's testnet environment
        # exchange.set_sandbox_mode(True)
        # logger.warning(f"{Back.YELLOW}{Fore.BLACK}!!! TESTNET MODE ACTIVE !!!{Style.RESET_ALL}")

        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True) # Force reload for fresh market data and limits
        logger.debug(f"Loaded {len(exchange.markets)} market structures.")

        # --- Initial Authentication & Connectivity Check ---
        logger.debug("Performing initial balance check for authentication and V5 connectivity...")
        try:
            # Fetch balance using V5 specific parameters to confirm keys and API version access
            exchange.fetch_balance(params={'category': 'linear'}) # Specify category for V5 balance check
            logger.success(f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (Targeting V5 API).{Style.RESET_ALL}")
            # Display warning only if NOT in sandbox mode
            if not getattr(exchange, 'sandbox', False): # Check if sandbox mode is active
                 logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! LIVE SCALPING MODE ACTIVE - EXTREME CAUTION ADVISED !!!{Style.RESET_ALL}")
            send_sms_alert(f"[Pyrmethus/{CONFIG.strategy_name}] Portal opened & authenticated.")
            return exchange
        except ccxt.AuthenticationError as auth_err:
            # Specific handling for auth errors after initial connection
            logger.critical(f"{Back.RED}{Fore.WHITE}Authentication failed during balance check: {auth_err}. Check API keys, permissions, IP whitelist, and account status on Bybit.{Style.RESET_ALL}")
            send_sms_alert(f"[Pyrmethus] CRITICAL: Authentication FAILED ({auth_err}). Spell failed.")
            return None
        except ccxt.ExchangeError as ex_err:
            # Catch V5 specific errors like invalid category if API setup is wrong
            logger.critical(f"{Back.RED}{Fore.WHITE}Exchange error during initial balance check (V5 connectivity issue?): {ex_err}.{Style.RESET_ALL}")
            send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error on Init ({ex_err}). Spell failed.")
            return None

    # --- Broader Error Handling for Initialization ---
    except ccxt.AuthenticationError as e: # Catch auth error during initial setup
        logger.critical(f"{Back.RED}{Fore.WHITE}Authentication failed during initial connection setup: {e}.{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Authentication FAILED on setup ({e}). Spell failed.")
    except ccxt.NetworkError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Network disturbance during portal opening: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Network Error on Init ({e}). Spell failed.")
    except ccxt.ExchangeError as e: # Catch other exchange errors during setup
        logger.critical(f"{Back.RED}{Fore.WHITE}Exchange spirit rejected portal opening: {e}. Check Bybit status, API documentation, or account status.{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error on Init ({e}). Spell failed.")
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected chaos during portal opening: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}. Spell failed.")

    return None # Return None if any initialization step failed

# --- Indicator Calculation Functions - Scrying the Market ---

def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """
    Calculates the Supertrend indicator using pandas_ta.

    Args:
        df: Pandas DataFrame with 'high', 'low', 'close' columns.
        length: The ATR lookback period for the Supertrend calculation.
        multiplier: The ATR multiplier for the Supertrend calculation.
        prefix: A string prefix to add to the resulting column names (e.g., "confirm_").

    Returns:
        The DataFrame with added Supertrend columns:
        - f'{prefix}supertrend': The Supertrend line value (Decimal).
        - f'{prefix}trend': Boolean indicating uptrend (True) or downtrend (False).
        - f'{prefix}st_long': Boolean, True if trend flipped to Long on this candle.
        - f'{prefix}st_short': Boolean, True if trend flipped to Short on this candle.
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    # Construct expected pandas_ta column names (adjust if pandas_ta naming changes)
    # Note: pandas_ta often uses float representation in column names
    multiplier_float_str = str(float(multiplier)).replace('.', '_') # Match potential pandas_ta naming convention
    st_col = f"SUPERT_{length}_{multiplier_float_str}" # Example: SUPERT_7_2_5
    st_trend_col = f"SUPERTd_{length}_{multiplier_float_str}" # Example: SUPERTd_7_2_5
    st_long_col = f"SUPERTl_{length}_{multiplier_float_str}" # Example: SUPERTl_7_2_5
    st_short_col = f"SUPERTs_{length}_{multiplier_float_str}" # Example: SUPERTs_7_2_5
    raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col] # Columns pandas_ta might create

    required_input_cols = ["high", "low", "close"]
    min_len = length + 1 # Minimum data length required for calculation stability

    # Input validation
    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA # Assign NA to expected output columns
        return df

    try:
        # pandas_ta expects float multiplier for calculation
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={float(multiplier)}")
        # Calculate using pandas_ta, appending results to the DataFrame
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # --- Verification and Renaming/Conversion ---
        # Check if pandas_ta created the expected columns (names might vary slightly)
        # Find the actual generated columns matching the pattern
        actual_st_col = next((col for col in df.columns if col.startswith(f"SUPERT_{length}_{multiplier_float_str[:3]}") and not col.endswith(('d','l','s'))), None) # Find main ST line
        actual_trend_col = next((col for col in df.columns if col.startswith(f"SUPERTd_{length}_{multiplier_float_str[:3]}")), None) # Find trend direction
        actual_long_col = next((col for col in df.columns if col.startswith(f"SUPERTl_{length}_{multiplier_float_str[:3]}")), None) # Find long flip signal
        actual_short_col = next((col for col in df.columns if col.startswith(f"SUPERTs_{length}_{multiplier_float_str[:3]}")), None) # Find short flip signal

        if not all([actual_st_col, actual_trend_col, actual_long_col, actual_short_col]):
            # Find which specific columns are missing
            missing_details = f"ST: {actual_st_col is None}, Trend: {actual_trend_col is None}, Long: {actual_long_col is None}, Short: {actual_short_col is None}"
            logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Failed to find all expected output columns from pandas_ta. Missing: {missing_details}. Check pandas_ta version/behavior.{Style.RESET_ALL}")
            # Attempt to clean up any partial columns found
            partial_cols_found = [c for c in [actual_st_col, actual_trend_col, actual_long_col, actual_short_col] if c]
            if partial_cols_found: df.drop(columns=partial_cols_found, errors='ignore', inplace=True)
            for col in target_cols: df[col] = pd.NA # Nullify results
            return df

        # Convert Supertrend value to Decimal, interpret trend and flips
        df[f"{col_prefix}supertrend"] = df[actual_st_col].apply(safe_decimal_conversion)
        # Trend: 1 = Uptrend, -1 = Downtrend. Convert to boolean: True for Up, False for Down.
        df[f"{col_prefix}trend"] = df[actual_trend_col] == 1
        # Flip Signals:
        # st_long_col (SUPERTl): Non-NaN (often 1.0) when trend flips Long.
        # st_short_col (SUPERTs): Non-NaN (often -1.0) when trend flips Short.
        df[f"{col_prefix}st_long"] = df[actual_long_col].notna() # True if flipped Long this candle
        df[f"{col_prefix}st_short"] = df[actual_short_col].notna() # True if flipped Short this candle

        # Clean up raw columns created by pandas_ta
        df.drop(columns=[actual_st_col, actual_trend_col, actual_long_col, actual_short_col], errors='ignore', inplace=True)

        # Log the latest reading for debugging
        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
            signal = 'LONG FLIP' if df[f'{col_prefix}st_long'].iloc[-1] else ('SHORT FLIP' if df[f'{col_prefix}st_short'].iloc[-1] else 'Hold')
            trend_color = Fore.GREEN if last_trend == 'Up' else Fore.RED
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Trend={trend_color}{last_trend}{Style.RESET_ALL}, Val={last_st_val:.4f}, Signal={signal}")
        else:
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except KeyError as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Error accessing column - likely pandas_ta issue, data problem, or naming mismatch: {e}{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df

def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> Dict[str, Optional[Decimal]]:
    """
    Calculates ATR, Volume Simple Moving Average (SMA), and Volume Ratio.

    Args:
        df: Pandas DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len: The lookback period for ATR calculation.
        vol_ma_len: The lookback period for Volume SMA calculation.

    Returns:
        A dictionary containing:
        - 'atr': The latest ATR value (Decimal) or None.
        - 'volume_ma': The latest Volume SMA value (Decimal) or None.
        - 'last_volume': The latest volume value (Decimal) or None.
        - 'volume_ratio': The ratio of last volume to volume SMA (Decimal) or None.
        Returns None values if calculation fails or data is insufficient.
    """
    results: Dict[str, Optional[Decimal]] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1 # Need at least period+1 for reliable calculation

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}). Cannot calculate.{Style.RESET_ALL}")
        return results

    try:
        # Calculate ATR (Average True Range) - Measure of volatility
        logger.debug(f"Scrying (ATR): Calculating with length={atr_len}")
        # Use pandas_ta for ATR calculation
        atr_col = f"ATRr_{atr_len}" # Default pandas_ta name for ATR (raw)
        df.ta.atr(length=atr_len, append=True)

        if atr_col in df.columns:
            # Convert last ATR value to Decimal
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr):
                results["atr"] = safe_decimal_conversion(last_atr)
            # Clean up the raw ATR column added by pandas_ta
            df.drop(columns=[atr_col], errors='ignore', inplace=True)
        else:
             logger.warning(f"ATR column '{atr_col}' not found after calculation. Check pandas_ta behavior.")

        # Calculate Volume Moving Average (SMA) and Ratio - Measure of market energy
        logger.debug(f"Scrying (Volume): Calculating SMA with length={vol_ma_len}")
        volume_ma_col = f'volume_sma_{vol_ma_len}' # Use a distinct name
        # Use pandas rolling mean for SMA of volume
        # min_periods ensures we get a value even if window isn't full at the start
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df['volume'].iloc[-1] # Get the most recent volume bar

        # Convert results to Decimal
        if pd.notna(last_vol_ma): results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol): results["last_volume"] = safe_decimal_conversion(last_vol)

        # Calculate Volume Ratio (Last Volume / Volume MA) safely
        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.position_qty_epsilon and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except (DivisionByZero, InvalidOperation) as ratio_err:
                 logger.warning(f"Division by zero or invalid op encountered calculating volume ratio (Volume MA likely zero/negligible). Error: {ratio_err}")
                 results["volume_ratio"] = None
            except Exception as ratio_err:
                 logger.warning(f"Unexpected error calculating volume ratio: {ratio_err}")
                 results["volume_ratio"] = None
        else:
            results["volume_ratio"] = None # Cannot calculate ratio if MA is zero/negligible or volume is NA

        # Clean up the volume MA column
        if volume_ma_col in df.columns:
            df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)

        # Log calculated results
        atr_str = f"{results['atr']:.5f}" if results['atr'] else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] else 'N/A'
        last_vol_str = f"{results['last_volume']:.2f}" if results['last_volume'] else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results['volume_ratio'] else 'N/A'
        logger.debug(f"Scrying Results: ATR({atr_len})={Fore.CYAN}{atr_str}{Style.RESET_ALL}, "
                     f"LastVol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results} # Nullify all results on error
    return results

def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """
    Calculates StochRSI (%K and %D) and Momentum indicator using pandas_ta.

    Args:
        df: Pandas DataFrame with 'close' column.
        rsi_len: The lookback period for the RSI component of StochRSI.
        stoch_len: The lookback period for the Stochastic component of StochRSI.
        k: The smoothing period for the %K line of StochRSI.
        d: The smoothing period for the %D (signal) line of StochRSI.
        mom_len: The lookback period for the Momentum indicator.

    Returns:
        The DataFrame with added columns:
        - 'stochrsi_k': The StochRSI %K value (Decimal).
        - 'stochrsi_d': The StochRSI %D value (Decimal).
        - 'momentum': The Momentum value (Decimal).
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    # Estimate minimum length: StochRSI needs roughly RSI + Stoch + D periods. Momentum needs its own period.
    min_len_stochrsi = rsi_len + stoch_len + d + 5 # Add buffer
    min_len_mom = mom_len + 1
    min_len = max(min_len_stochrsi, min_len_mom)

    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # Calculate StochRSI using pandas_ta
        logger.debug(f"Scrying (StochRSI): Calculating with RSI={rsi_len}, Stoch={stoch_len}, K={k}, D={d}")
        # Calculate separately first to handle potential column naming issues more easily
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        # Standard pandas_ta column names for StochRSI %K and %D (adjust if needed)
        k_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}"
        d_col = f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"

        # Assign results to main DataFrame and convert to Decimal
        if k_col in stochrsi_df.columns:
            df['stochrsi_k'] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"StochRSI K column '{k_col}' not found after calculation. Check pandas_ta naming.")
            df['stochrsi_k'] = pd.NA
        if d_col in stochrsi_df.columns:
             df['stochrsi_d'] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else:
            logger.warning(f"StochRSI D column '{d_col}' not found after calculation. Check pandas_ta naming.")
            df['stochrsi_d'] = pd.NA

        # Calculate Momentum using pandas_ta
        logger.debug(f"Scrying (Momentum): Calculating with length={mom_len}")
        mom_col = f"MOM_{mom_len}" # Standard pandas_ta name
        df.ta.mom(length=mom_len, append=True)

        if mom_col in df.columns:
            df['momentum'] = df[mom_col].apply(safe_decimal_conversion)
            # Clean up raw momentum column
            df.drop(columns=[mom_col], errors='ignore', inplace=True)
        else:
            logger.warning(f"Momentum column '{mom_col}' not found after calculation. Check pandas_ta naming.")
            df['momentum'] = pd.NA

        # Log latest values for debugging
        k_val = df['stochrsi_k'].iloc[-1]
        d_val = df['stochrsi_d'].iloc[-1]
        mom_val = df['momentum'].iloc[-1]

        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            k_color = Fore.RED if k_val > CONFIG.stochrsi_overbought else (Fore.GREEN if k_val < CONFIG.stochrsi_oversold else Fore.CYAN)
            d_color = Fore.RED if d_val > CONFIG.stochrsi_overbought else (Fore.GREEN if d_val < CONFIG.stochrsi_oversold else Fore.CYAN)
            mom_color = Fore.GREEN if mom_val > CONFIG.position_qty_epsilon else (Fore.RED if mom_val < -CONFIG.position_qty_epsilon else Fore.WHITE)
            logger.debug(f"Scrying (StochRSI/Mom): K={k_color}{k_val:.2f}{Style.RESET_ALL}, D={d_color}{d_val:.2f}{Style.RESET_ALL}, Mom({mom_len})={mom_color}{mom_val:.4f}{Style.RESET_ALL}")
        else:
            logger.debug("Scrying (StochRSI/Mom): Resulted in NA for one or more values on last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df

def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """
    Calculates the Ehlers Fisher Transform indicator using pandas_ta.

    Args:
        df: Pandas DataFrame with 'high', 'low' columns.
        length: The lookback period for the Fisher Transform calculation.
        signal: The smoothing period for the signal line (often 1 for trigger-only).

    Returns:
        The DataFrame with added columns:
        - 'ehlers_fisher': The Fisher Transform value (Decimal).
        - 'ehlers_signal': The Fisher Transform signal line value (Decimal).
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    required_input_cols = ["high", "low"]
    min_len = length + signal + 5 # Add buffer for calculation stability

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        logger.debug(f"Scrying (EhlersFisher): Calculating with length={length}, signal={signal}")
        # Calculate separately first
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        # Standard pandas_ta column names (adjust if needed)
        fish_col = f"FISHERT_{length}_{signal}"
        signal_col = f"FISHERTs_{length}_{signal}"

        # Assign results and convert to Decimal
        if fish_col in fisher_df.columns:
            df['ehlers_fisher'] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else:
             logger.warning(f"Ehlers Fisher column '{fish_col}' not found after calculation. Check pandas_ta naming.")
             df['ehlers_fisher'] = pd.NA
        if signal_col in fisher_df.columns:
            df['ehlers_signal'] = fisher_df[signal_col].apply(safe_decimal_conversion)
        else:
            # If signal=1, pandas_ta might not create a separate signal column, often it's the same as the fisher line
            if signal == 1 and fish_col in fisher_df.columns:
                 logger.debug(f"Ehlers Fisher signal length is 1, using Fisher line '{fish_col}' as signal.")
                 df['ehlers_signal'] = df['ehlers_fisher'] # Use Fisher line itself as signal
            else:
                 logger.warning(f"Ehlers Signal column '{signal_col}' not found after calculation. Check pandas_ta naming.")
                 df['ehlers_signal'] = pd.NA

        # Log latest values for debugging
        fish_val = df['ehlers_fisher'].iloc[-1]
        sig_val = df['ehlers_signal'].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
             logger.debug(f"Scrying (EhlersFisher({length},{signal})): Fisher={Fore.CYAN}{fish_val:.4f}{Style.RESET_ALL}, Signal={Fore.MAGENTA}{sig_val:.4f}{Style.RESET_ALL}")
        else:
             logger.debug("Scrying (EhlersFisher): Resulted in NA for one or more values on last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df

def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """
    Placeholder function using standard Exponential Moving Averages (EMA) instead of
    Ehlers Super Smoother Moving Averages.

    *** WARNING: This is NOT a true Ehlers MA implementation. ***
    For accurate Ehlers MA Cross strategy results, replace this with a proper
    Ehlers Super Smoother filter implementation.

    Args:
        df: Pandas DataFrame with 'close' column.
        fast_len: The period for the fast EMA.
        slow_len: The period for the slow EMA.

    Returns:
        The DataFrame with added columns:
        - 'fast_ema': The fast EMA value (Decimal).
        - 'slow_ema': The slow EMA value (Decimal).
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    target_cols = ['fast_ema', 'slow_ema']
    required_input_cols = ["close"]
    min_len = max(fast_len, slow_len) + 10 # EMA needs buffer for stability

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersMA - EMA Placeholder): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        # *** PYRMETHUS NOTE / WARNING ***
        logger.warning(f"{Fore.YELLOW}{Style.DIM}Scrying (EhlersMA): Using standard EMA as placeholder for Ehlers Super Smoother. "
                       f"This strategy path ('EHLERS_MA_CROSS') will NOT use true Ehlers MAs and may not perform as intended. "
                       f"Verify indicator suitability or implement actual Ehlers Super Smoother.{Style.RESET_ALL}")

        logger.debug(f"Scrying (EhlersMA - EMA Placeholder): Calculating Fast EMA({fast_len}), Slow EMA({slow_len})")
        # Use pandas_ta standard EMA calculation and convert to Decimal
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        # Log latest values for debugging
        fast_val = df['fast_ema'].iloc[-1]
        slow_val = df['slow_ema'].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            cross_color = Fore.GREEN if fast_val > slow_val else Fore.RED
            logger.debug(f"Scrying (EhlersMA({fast_len},{slow_len}) - EMA): Fast={cross_color}{fast_val:.4f}{Style.RESET_ALL}, Slow={cross_color}{slow_val:.4f}{Style.RESET_ALL}")
        else:
             logger.debug("Scrying (EhlersMA - EMA): Resulted in NA for one or more values on last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersMA - EMA): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df

def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> Dict[str, Optional[Decimal]]:
    """
    Fetches and analyzes the L2 order book to calculate bid/ask volume ratio and spread.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        depth: The number of price levels on each side (bids/asks) to include in the volume ratio calculation.
        fetch_limit: The number of price levels to request from the exchange API (should be >= depth).

    Returns:
        A dictionary containing:
        - 'bid_ask_ratio': Ratio of total bid volume to total ask volume within the specified depth (Decimal) or None.
        - 'spread': Difference between best ask and best bid (Decimal) or None.
        - 'best_bid': The highest bid price (Decimal) or None.
        - 'best_ask': The lowest ask price (Decimal) or None.
        Returns None values if analysis fails or data is unavailable.
    """
    results: Dict[str, Optional[Decimal]] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book Scrying: Fetching L2 {symbol} (Depth:{depth}, Request Limit:{fetch_limit})...")

    # Check if the exchange supports fetching L2 order book
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}Order Book Scrying: Exchange '{exchange.id}' does not support fetchL2OrderBook method. Cannot analyze depth.{Style.RESET_ALL}")
        return results

    try:
        # Fetching the order book's current state
        # Bybit V5 might require 'category' param here too, though CCXT might handle it via defaultType
        params = {'category': 'linear'} # Add category for V5 consistency
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit, params=params)
        bids: List[List[Union[float, str]]] = order_book.get('bids', []) # List of [price, amount]
        asks: List[List[Union[float, str]]] = order_book.get('asks', []) # List of [price, amount]

        if not bids or not asks:
             logger.warning(f"{Fore.YELLOW}Order Book Scrying: Empty bids or asks received for {symbol}. Cannot analyze.{Style.RESET_ALL}")
             return results # Return defaults (all None)

        # Extract best bid/ask with Decimal precision
        # Ensure lists are not empty and contain price/amount pairs
        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) >= 1 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) >= 1 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Calculate spread
        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
            logger.debug(f"OB Scrying: Best Bid={Fore.GREEN}{best_bid:.4f}{Style.RESET_ALL}, Best Ask={Fore.RED}{best_ask:.4f}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results['spread']:.4f}{Style.RESET_ALL}")
        else:
            logger.debug(f"OB Scrying: Best Bid={best_bid or 'N/A'}, Best Ask={best_ask or 'N/A'} (Spread calculation skipped due to invalid bid/ask)")

        # Sum total volume within the specified depth using Decimal for precision
        # Ensure list slicing doesn't go out of bounds and elements are valid pairs
        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:min(depth, len(bids))] if len(bid) >= 2)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:min(depth, len(asks))] if len(ask) >= 2)
        logger.debug(f"OB Scrying (Depth {depth}): Total BidVol={Fore.GREEN}{bid_vol:.4f}{Style.RESET_ALL}, Total AskVol={Fore.RED}{ask_vol:.4f}{Style.RESET_ALL}")

        # Calculate Bid/Ask Volume Ratio (Total Bid Volume / Total Ask Volume)
        if ask_vol > CONFIG.position_qty_epsilon: # Avoid division by zero or near-zero
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                # Determine color based on configured thresholds for logging
                ratio_color = Fore.GREEN if results["bid_ask_ratio"] >= CONFIG.order_book_ratio_threshold_long else \
                              (Fore.RED if results["bid_ask_ratio"] <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW)
                logger.debug(f"OB Scrying Ratio (Bids/Asks): {ratio_color}{results['bid_ask_ratio']:.3f}{Style.RESET_ALL}")
            except (DivisionByZero, InvalidOperation, Exception) as e:
                logger.warning(f"{Fore.YELLOW}Error calculating OB ratio: {e}{Style.RESET_ALL}")
                results["bid_ask_ratio"] = None
        else:
            logger.debug(f"OB Scrying Ratio: N/A (Ask volume within depth {depth} is zero or negligible: {ask_vol:.4f})")
            results["bid_ask_ratio"] = None # Set explicitly to None

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except IndexError:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error: Index out of bounds accessing bids/asks for {symbol}. Order book data might be malformed or incomplete.{Style.RESET_ALL}")
    except Exception as e:
        logger.warning(f"{Fore.YELLOW}Unexpected Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    # Ensure results dictionary keys exist even if errors occurred (avoids KeyErrors later)
    results.setdefault("bid_ask_ratio", None)
    results.setdefault("spread", None)
    results.setdefault("best_bid", None)
    results.setdefault("best_ask", None)
    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV data, prepares it as a pandas DataFrame, ensures numeric types,
    and handles missing values (NaNs) robustly.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        interval: The timeframe interval (e.g., '1m', '5m').
        limit: The number of candles to fetch.

    Returns:
        A pandas DataFrame containing the OHLCV data with a datetime index (UTC),
        or None if fetching or processing fails.
    """
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})...")
        # Channeling the data stream from the exchange
        # Bybit V5 might require 'category' param here too, though CCXT might handle it
        params = {'category': 'linear'} # Add category for V5 consistency
        ohlcv: List[List[Union[int, float, str]]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit, params=params)

        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Market might be inactive, symbol incorrect, or API issue.{Style.RESET_ALL}")
            return None

        # Weaving data into a DataFrame structure
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # --- Data Cleaning and Preparation ---
        # Convert timestamp to datetime objects (UTC) and set as index
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
        except Exception as time_e:
             logger.error(f"{Fore.RED}Data Fetch: Error converting timestamp column: {time_e}{Style.RESET_ALL}")
             return None # Cannot proceed without valid timestamps

        # Ensure OHLCV columns are numeric, coercing errors to NaN
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Robust NaN Handling ---
        initial_nan_count = df.isnull().sum().sum()
        if initial_nan_count > 0:
            nan_counts_per_col = df.isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: Found {initial_nan_count} NaN values in OHLCV data after conversion:\n"
                           f"{nan_counts_per_col[nan_counts_per_col > 0]}\nAttempting forward fill (ffill)...{Style.RESET_ALL}")
            df.ffill(inplace=True) # Fill NaNs with the previous valid observation

            # Check if NaNs remain (likely at the beginning of the series if data history is short)
            remaining_nan_count = df.isnull().sum().sum()
            if remaining_nan_count > 0:
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill ({remaining_nan_count}). Attempting backward fill (bfill)...{Style.RESET_ALL}")
                df.bfill(inplace=True) # Fill remaining NaNs with the next valid observation

                # Final check: if NaNs still exist, data is likely too gappy at start/end or completely invalid
                final_nan_count = df.isnull().sum().sum()
                if final_nan_count > 0:
                    logger.error(f"{Fore.RED}Data Fetch: Unfillable NaN values ({final_nan_count}) remain after ffill and bfill. "
                                 f"Data quality insufficient for {symbol}. Columns with NaNs:\n{df.isnull().sum()[df.isnull().sum() > 0]}\nSkipping cycle.{Style.RESET_ALL}")
                    return None # Cannot proceed with unreliable data

        logger.debug(f"Data Fetch: Successfully woven and cleaned {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: Disturbance gathering OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return None # Return None if any error occurred

# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches current position details using Bybit V5 API specifics (`fetchPositions`).
    Assumes One-Way Mode (looks for positionIdx=0).

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').

    Returns:
        A dictionary containing:
        - 'side': Position side ('Long', 'Short', or 'None').
        - 'qty': Position quantity as Decimal (0.0 if flat).
        - 'entry_price': Average entry price as Decimal (0.0 if flat).
        Returns default flat state dictionary on error or if no position found.
    """
    default_pos: Dict[str, Any] = {'side': CONFIG.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market_id: Optional[str] = None
    market: Optional[Dict[str, Any]] = None
    category: Optional[str] = None

    try:
        # Get market details to determine category (linear/inverse) and the exchange's specific ID
        market = exchange.market(symbol)
        market_id = market['id'] # The exchange's specific ID (e.g., BTCUSDT)
        # Determine category based on market properties (linear = USDT margined)
        if market.get('linear'):
            category = 'linear'
        elif market.get('inverse'):
            category = 'inverse'
        else:
            # Fallback or error if category cannot be determined (shouldn't happen for loaded futures markets)
            logger.warning(f"{Fore.YELLOW}Position Check: Could not determine category (linear/inverse) for market '{symbol}'. Assuming 'linear'.{Style.RESET_ALL}")
            category = 'linear' # Default assumption for this bot

    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(f"{Fore.RED}Position Check: Failed to identify market structure for '{symbol}': {e}. Cannot check position.{Style.RESET_ALL}")
        return default_pos
    except Exception as e_market:
        logger.error(f"{Fore.RED}Position Check: Unexpected error getting market info for '{symbol}': {e_market}. Cannot check position.{Style.RESET_ALL}")
        return default_pos

    try:
        # Check if the exchange instance supports fetchPositions (should for Bybit V5 via CCXT)
        if not exchange.has.get('fetchPositions'):
            logger.error(f"{Fore.RED}Position Check: Exchange '{exchange.id}' CCXT instance does not support fetchPositions method. Cannot get V5 position data.{Style.RESET_ALL}")
            # This indicates a potential issue with the CCXT version or exchange setup
            return default_pos

        # Bybit V5 fetchPositions requires 'category' and optionally 'symbol' (market_id)
        params = {'category': category, 'symbol': market_id}
        logger.debug(f"Position Check: Querying V5 positions for {symbol} (MarketID: {market_id}, Category: {category})...")

        # Summon position data from the exchange
        # Pass the specific symbol to filter results if the exchange supports it
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # --- Parse V5 Position Data (One-Way Mode Logic) ---
        # Bybit V5 fetchPositions returns a list. In One-Way mode, the active position
        # for a symbol typically has positionIdx = 0.
        # The 'side' field in the raw 'info' dict indicates 'Buy' (Long) or 'Sell' (Short).
        # A 'size' of "0" or side 'None' indicates no position.
        active_pos_info = None
        for pos in fetched_positions:
            # Access raw info dict provided by CCXT, which contains exchange-specific fields
            pos_info = pos.get('info', {})
            pos_market_id = pos_info.get('symbol')
            # positionIdx indicates hedge mode (1=Buy, 2=Sell) or one-way mode (0)
            position_idx = int(pos_info.get('positionIdx', -1)) # Default to -1 if missing
            pos_side_v5 = pos_info.get('side', 'None') # V5 specific: 'Buy', 'Sell', or 'None'
            size_str = pos_info.get('size', "0") # Position size as a string from the exchange

            # Check if this entry matches our symbol and is the primary One-Way position
            if pos_market_id == market_id and position_idx == 0:
                 # Check if size is non-zero and side indicates an open position
                 size_dec = safe_decimal_conversion(size_str)
                 # Use epsilon comparison for size check
                 if abs(size_dec) > CONFIG.position_qty_epsilon and pos_side_v5 != 'None':
                     logger.debug(f"Found active V5 position candidate: Idx={position_idx}, Side={pos_side_v5}, Size={size_str}")
                     active_pos_info = pos_info # Store the raw info dict of the active position
                     break # Assume only one active position per symbol in One-Way mode

        # --- Process Found Position ---
        if active_pos_info:
            try:
                # Parse details from the active position info using safe Decimal conversion
                size = safe_decimal_conversion(active_pos_info.get('size'))
                # Use 'avgPrice' from V5 info dict for the entry price
                entry_price = safe_decimal_conversion(active_pos_info.get('avgPrice'))
                # Determine internal side representation based on V5 'side' field
                side = CONFIG.pos_long if active_pos_info.get('side') == 'Buy' else CONFIG.pos_short

                # Final validation of parsed data
                if abs(size) <= CONFIG.position_qty_epsilon or entry_price <= 0:
                     logger.warning(f"{Fore.YELLOW}Position Check: Found active V5 pos for {market_id} but parsed invalid size/entry: Size={size}, Entry={entry_price}. Treating as flat.{Style.RESET_ALL}")
                     return default_pos

                pos_color = Fore.GREEN if side == CONFIG.pos_long else Fore.RED
                logger.info(f"{pos_color}Position Check: Found ACTIVE {side} position: Qty={size:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                return {'side': side, 'qty': size, 'entry_price': entry_price}

            except Exception as parse_err:
                 logger.warning(f"{Fore.YELLOW}Position Check: Error parsing details from active V5 position data: {parse_err}. Data: {active_pos_info}. Treating as flat.{Style.RESET_ALL}")
                 return default_pos # Return default on parsing error
        else:
            # No position found matching the criteria (symbol, positionIdx=0, non-zero size)
            logger.info(f"{Fore.BLUE}Position Check: No active One-Way (positionIdx=0) position found for {market_id}. Currently Flat.{Style.RESET_ALL}")
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Position Check: Disturbance querying V5 positions for {symbol}: {type(e).__name__} - {e}. Assuming flat.{Style.RESET_ALL}")
    except Exception as e_pos:
        logger.error(f"{Fore.RED}Position Check: Unexpected error querying V5 positions for {symbol}: {e_pos}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    # Return default (flat) state if any error occurred during the process
    return default_pos

def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """
    Sets the leverage for a given futures symbol using Bybit V5 API specifics.
    Retries on transient errors.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        leverage: The desired integer leverage value.

    Returns:
        True if leverage was set successfully or confirmed already set, False otherwise.
    """
    logger.info(f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    market: Optional[Dict[str, Any]] = None
    category: Optional[str] = None

    try:
        # Get market details to ensure it's a contract market and determine category
        market = exchange.market(symbol)
        if not market.get('contract'):
            logger.error(f"{Fore.RED}Leverage Conjuring Error: Cannot set leverage for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False

        # Determine category for V5 params
        if market.get('linear'): category = 'linear'
        elif market.get('inverse'): category = 'inverse'
        else:
             logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Could not determine category for {symbol}. Assuming 'linear'.{Style.RESET_ALL}")
             category = 'linear'

        # Bybit V5 requires setting buy and sell leverage separately via params.
        # The main `leverage` argument to `set_leverage` might also be needed by CCXT internally.
        params = {
            'category': category,
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage),
        }
        logger.debug(f"Using V5 params for set_leverage: {params}")

    except (ccxt.BadSymbol, KeyError) as e:
         logger.error(f"{Fore.RED}Leverage Conjuring Error: Failed to identify market structure for '{symbol}': {e}.{Style.RESET_ALL}")
         return False
    except Exception as e_market:
         logger.error(f"{Fore.RED}Leverage Conjuring Error: Unexpected error getting market info for '{symbol}': {e_market}.{Style.RESET_ALL}")
         return False

    # --- Attempt Leverage Setting with Retries ---
    for attempt in range(CONFIG.retry_count + 1): # Total attempts = retry_count + 1
        try:
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            # Response format varies, log it for debugging. Success is usually indicated by lack of error.
            logger.success(f"{Fore.GREEN}Leverage Conjuring: Successfully set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True

        except ccxt.ExchangeError as e:
            # Check for common Bybit V5 errors indicating leverage is already set or not modified
            err_str = str(e).lower()
            # Example error codes/messages from Bybit V5 (verify against current docs):
            # 110044: Leverage not modified
            # 110025: Leverage cannot be lower than 1
            # 10001 / "params error": Can occur if leverage is identical
            if "leverage not modified" in err_str or "same leverage" in err_str or "110044" in err_str or ("params error" in err_str and "leverage" in err_str):
                logger.info(f"{Fore.CYAN}Leverage Conjuring: Leverage already set to {leverage}x for {symbol} (or not modified).{Style.RESET_ALL}")
                return True
            elif "cannot be lower than 1" in err_str and leverage < 1:
                 logger.error(f"{Fore.RED}Leverage Conjuring: Invalid leverage value ({leverage}) requested.{Style.RESET_ALL}")
                 return False # Don't retry invalid value

            # Log other exchange errors and decide whether to retry
            logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Exchange resistance (Attempt {attempt+1}/{CONFIG.retry_count+1}): {e}{Style.RESET_ALL}")
            if attempt >= CONFIG.retry_count:
                 logger.error(f"{Fore.RED}Leverage Conjuring: Failed after {CONFIG.retry_count+1} attempts due to ExchangeError.{Style.RESET_ALL}")
                 break # Exit loop after final attempt
            # Use increasing delay for retries
            time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Network/Timeout disturbance (Attempt {attempt+1}/{CONFIG.retry_count+1}): {e}{Style.RESET_ALL}")
            if attempt >= CONFIG.retry_count:
                 logger.error(f"{Fore.RED}Leverage Conjuring: Failed due to network issues after {CONFIG.retry_count+1} attempts.{Style.RESET_ALL}")
                 break
            time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))
        except Exception as e_unexp:
             logger.error(f"{Fore.RED}Leverage Conjuring: Unexpected error (Attempt {attempt+1}/{CONFIG.retry_count+1}): {e_unexp}{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             # Decide if this is retryable or fatal
             if attempt >= CONFIG.retry_count:
                 logger.error(f"{Fore.RED}Leverage Conjuring: Failed due to unexpected error after {CONFIG.retry_count+1} attempts.{Style.RESET_ALL}")
                 break
             time.sleep(CONFIG.retry_delay_seconds * (attempt + 1))

    # If loop completes without returning True, it failed
    return False

def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal") -> Optional[Dict[str, Any]]:
    """
    Closes the specified active position by placing a market order with `reduceOnly=True`.
    Re-validates the position state just before attempting closure.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        position_to_close: A dictionary containing the 'side' and 'qty' of the position believed to be open.
        reason: A string indicating the reason for closing (for logging/alerts).

    Returns:
        The CCXT order dictionary if the close order was successfully placed,
        None if the position was already closed upon re-validation or if the close order placement failed.
    """
    initial_side = position_to_close.get('side', CONFIG.pos_none)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0].split(':')[0] # For concise alerts (e.g., BTC from BTC/USDT:USDT)
    logger.info(f"{Fore.YELLOW}Banish Position Ritual: Initiated for {symbol}. Reason: {reason}. Expected state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # --- Re-validate the position just before closing ---
    logger.debug("Banish Position: Re-validating live position state...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position['side']
    live_amount_to_close = live_position['qty']

    if live_position_side == CONFIG.pos_none or abs(live_amount_to_close) <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position (or negligible size: {live_amount_to_close:.8f}) for {symbol}. Aborting banishment.{Style.RESET_ALL}")
        if initial_side != CONFIG.pos_none:
            # This indicates a potential state discrepancy between the start of the cycle and now
            logger.warning(f"{Fore.YELLOW}Banish Position: State Discrepancy! Expected {initial_side}, but live state is None/Zero.{Style.RESET_ALL}")
        return None # Nothing to close

    # Determine the side of the market order needed to close the position
    side_to_execute_close = CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy

    try:
        # Format amount according to market rules
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_dec = safe_decimal_conversion(amount_str) # Convert formatted string back to Decimal for check
        amount_float = float(amount_dec) # CCXT create order often expects float

        # Check if the amount is valid after formatting (should not be zero if live_amount_to_close was valid)
        if abs(amount_dec) <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Banish Position: Closing amount negligible ({amount_str}) after precision shaping. Cannot close. Manual check advised.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Negligible close amount {amount_str}. MANUAL CHECK!")
            return None

        # --- Execute the Market Close Order ---
        close_color = Back.YELLOW
        logger.warning(f"{close_color}{Fore.BLACK}{Style.BRIGHT}Banish Position: Attempting to CLOSE {live_position_side} ({reason}): "
                       f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}")

        # Bybit V5 requires category, and reduceOnly must be set for closing orders
        market = exchange.market(symbol)
        category = 'linear' if market.get('linear') else 'inverse' # Re-determine category
        params = {
            'category': category,
            'reduceOnly': True # Crucial: ensures this order only reduces/closes the position
        }

        # Place the market order to close the position
        order = exchange.create_market_order(symbol=symbol, side=side_to_execute_close, amount=amount_float, params=params)

        # --- Process Order Placement Result ---
        # Order placed successfully (API call returned), parse response safely
        # Note: 'average' might be None immediately for market orders, 'price' might be 0.
        # Rely on 'filled' and 'cost' primarily. Fetch order later if exact avg price needed.
        fill_price_avg = safe_decimal_conversion(order.get('average')) # May be None initially
        filled_qty = safe_decimal_conversion(order.get('filled')) # Should match amount_dec if fully filled
        cost = safe_decimal_conversion(order.get('cost'))
        order_id_short = format_order_id(order.get('id'))
        status = order.get('status', 'unknown') # Should be 'closed' if fully filled immediately

        # Log success based on filled quantity vs expected quantity (allowing for small tolerance)
        qty_diff = abs(filled_qty - amount_dec)
        if qty_diff < (CONFIG.position_qty_epsilon * Decimal("10")): # Allow slightly larger tolerance for fill matching
            logger.success(f"{Fore.GREEN}{Style.BRIGHT}Banish Position: Order ({reason}) appears FILLED for {symbol}. "
                           f"Filled: {filled_qty:.8f}, AvgFill: {fill_price_avg:.4f}, Cost: {cost:.2f} USDT. ID:{order_id_short}, Status: {status}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] BANISHED {live_position_side} {amount_str} @ ~{fill_price_avg:.4f} ({reason}). ID:{order_id_short}")
            return order # Return the successful order details
        else:
            # Partial fill or zero fill on market close order is unusual but possible (e.g., during extreme volatility/liquidity issues)
            logger.warning(f"{Fore.YELLOW}Banish Position: Order ({reason}) status/fill uncertain. Expected {amount_dec:.8f}, Filled: {filled_qty:.8f} (Diff: {qty_diff:.8e}). "
                           f"ID:{order_id_short}, Status: {status}. Re-checking position state soon is advised.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] WARNING Banishing ({reason}): Fill mismatch? (Exp:{amount_dec:.8f}, Got:{filled_qty:.8f}). ID:{order_id_short}")
            # Return order details, but the caller should ideally re-verify position state later
            return order

    # --- Handle Specific Errors During Order Placement ---
    except ccxt.InsufficientFunds as e:
         logger.error(f"{Fore.RED}Banish Position ({reason}): Insufficient funds error for {symbol}: {e}. This might indicate margin issues, incorrect state, or exchange problems.{Style.RESET_ALL}")
         send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Insufficient Funds! Check account.")
    except ccxt.ExchangeError as e:
        # Check for specific Bybit V5 errors indicating position already closed or issues with reduceOnly
        err_str = str(e).lower()
        # Example error codes/messages (verify against Bybit V5 docs):
        # 110007: Order quantity exceeds position size (reduceOnly)
        # 110015: Reduce-only rule violation / Order would increase position
        # 1100XX: Position size related errors
        # Check for messages indicating the position is already gone or the order wouldn't reduce it
        # Note: Error codes can change, relying on messages might be more robust but less precise.
        already_closed_indicators = [
            "order quantity exceeds open position size", # V5 message
            "position is zero",
            "position size is zero",
            "110007", # V5 code for qty exceeds pos size
        ]
        reduce_only_violation_indicators = [
             "order would not reduce position size", # V5 message?
             "reduce-only rule violation",
             "order would increase position",
             "110015", # V5 code for reduce only violation
        ]

        if any(indicator in err_str for indicator in already_closed_indicators):
             logger.warning(f"{Fore.YELLOW}Banish Position: Exchange indicates position likely already closed or zero size ({e}). Assuming banished.{Style.RESET_ALL}")
             # Consider this 'successful' in the sense that the desired state (flat) is likely achieved.
             return None # Treat as effectively closed (or non-actionable)
        elif any(indicator in err_str for indicator in reduce_only_violation_indicators):
             logger.error(f"{Fore.RED}Banish Position ({reason}): Reduce-Only Violation for {symbol}: {e}. Order likely incorrect or state mismatch.{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Reduce-Only Violation! Check state.")
        else:
             # Log other, unhandled exchange errors
             logger.error(f"{Fore.RED}Banish Position ({reason}): Unhandled Exchange Error for {symbol}: {e}{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Exchange Error: {e}. Check logs.")
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
        logger.error(f"{Fore.RED}Banish Position ({reason}): Network/Timeout Error for {symbol}: {e}. Position state uncertain. Manual check advised.{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Network/Timeout Error. Check position manually.")
    except ValueError as e: # Catches potential issues from format_amount or float conversion
         logger.error(f"{Fore.RED}Banish Position ({reason}): Value Error (likely formatting/conversion) for {symbol}: {e}{Style.RESET_ALL}")
    except Exception as e: # Catch-all for unexpected issues
        logger.error(f"{Fore.RED}Banish Position ({reason}): Unexpected Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR Banishing ({reason}): Unexpected Error {type(e).__name__}. Check logs.")

    # Return None if any error prevented successful order placement or indicated prior closure
    return None

def calculate_position_size(equity: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, stop_loss_price: Decimal,
                            leverage: int, symbol: str, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Calculates the position size based on risk percentage, entry/SL prices, leverage, and market constraints.

    Args:
        equity: Total account equity (USDT) as Decimal.
        risk_per_trade_pct: The fraction of equity to risk (e.g., 0.01 for 1%) as Decimal.
        entry_price: Estimated entry price as Decimal.
        stop_loss_price: Estimated stop loss price as Decimal.
        leverage: The leverage being used (integer).
        symbol: The market symbol.
        exchange: The CCXT exchange instance (for formatting).

    Returns:
        A tuple containing:
        - position_quantity (Decimal): The calculated quantity respecting market precision, or None on failure.
        - estimated_margin_required (Decimal): The estimated margin needed for the position, or None on failure.
    """
    logger.debug(f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")

    # --- Input Validation ---
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid entry/SL price (<= 0). Entry={entry_price}, SL={stop_loss_price}.{Style.RESET_ALL}")
        return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.position_qty_epsilon: # Use epsilon for comparison
        logger.error(f"{Fore.RED}Risk Calc Error: Entry and SL prices are too close ({price_diff:.8f}). Cannot calculate safe size.{Style.RESET_ALL}")
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid risk percentage: {risk_per_trade_pct:.4%}. Must be between 0 and 1 (e.g., 0.01 for 1%).{Style.RESET_ALL}")
        return None, None
    if equity <= 0:
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid equity: {equity:.4f}. Cannot calculate risk.{Style.RESET_ALL}")
        return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc Error: Invalid leverage: {leverage}. Must be > 0.{Style.RESET_ALL}")
        return None, None

    # --- Core Calculation ---
    try:
        # Calculate the maximum USDT amount to risk on this trade
        risk_amount_usdt = equity * risk_per_trade_pct
        # Calculate the risk per unit of the base asset (in USDT)
        risk_per_unit = price_diff
        # Calculate the raw quantity based on risk
        # Quantity = Total Risk Amount (USDT) / Risk per Unit (USDT/BaseAsset) = BaseAsset Quantity
        quantity_raw = risk_amount_usdt / risk_per_unit
        logger.debug(f"Risk Calc: RiskAmt={risk_amount_usdt:.4f} USDT, PriceDiff={price_diff:.4f} USDT/Unit, Raw Qty={quantity_raw:.8f} Units")

    except (DivisionByZero, InvalidOperation) as calc_err:
         logger.error(f"{Fore.RED}Risk Calc Error: Calculation failed (likely division by zero due to price_diff={price_diff}): {calc_err}{Style.RESET_ALL}")
         return None, None
    except Exception as calc_err:
         logger.error(f"{Fore.RED}Risk Calc Error: Unexpected error during raw quantity calculation: {calc_err}{Style.RESET_ALL}")
         return None, None

    # --- Apply Market Precision using CCXT formatter ---
    try:
        # Format the raw quantity according to the market's amount precision rules
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        # Convert the formatted string back to Decimal for internal use
        quantity_precise = Decimal(quantity_precise_str)
        logger.debug(f"Risk Calc: Precise Qty after formatting={quantity_precise:.8f}")
    except Exception as fmt_err:
        logger.warning(f"{Fore.YELLOW}Risk Calc Warning: Failed precision shaping for quantity {quantity_raw:.8f}. Error: {fmt_err}. Attempting fallback quantization.{Style.RESET_ALL}")
        # Fallback: Quantize raw value to a reasonable number of decimal places if formatting fails
        # Determine appropriate decimal places (e.g., 8 or from market info if available)
        try:
             # Attempt to get precision from market info if possible
             amount_precision_digits = exchange.markets[symbol].get('precision', {}).get('amount')
             if amount_precision_digits:
                 quantizer = Decimal('1e-' + str(int(amount_precision_digits))) # e.g., Decimal('1e-3') for 3 digits
                 quantity_precise = quantity_raw.quantize(quantizer, rounding=ROUND_HALF_UP) # Round half up
                 logger.debug(f"Risk Calc: Fallback Quantized Qty (Market Precision {amount_precision_digits})={quantity_precise:.8f}")
             else:
                 # Default fallback if market precision unavailable
                 quantity_precise = quantity_raw.quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP)
                 logger.debug(f"Risk Calc: Fallback Quantized Qty (Default 8 Decimals)={quantity_precise:.8f}")
        except Exception as q_err:
             logger.error(f"{Fore.RED}Risk Calc Error: Failed fallback quantization for quantity {quantity_raw:.8f}: {q_err}{Style.RESET_ALL}")
             return None, None

    # --- Final Checks & Margin Estimation ---
    if abs(quantity_precise) <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Risk Calc Warning: Calculated quantity negligible ({quantity_precise:.8f}) after formatting/quantization. "
                       f"RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}. Cannot place order.{Style.RESET_ALL}")
        return None, None

    # Estimate position value and margin required based on the precise quantity
    try:
        pos_value_usdt = quantity_precise * entry_price
        # Margin = Position Value / Leverage
        required_margin = pos_value_usdt / Decimal(leverage)
        logger.debug(f"Risk Calc Result: Qty={Fore.CYAN}{quantity_precise:.8f}{Style.RESET_ALL}, Est. Pos Value={pos_value_usdt:.4f} USDT, Est. Margin Req.={required_margin:.4f} USDT")
        return quantity_precise, required_margin
    except (DivisionByZero, InvalidOperation, Exception) as margin_err:
        logger.error(f"{Fore.RED}Risk Calc Error: Failed calculating estimated margin: {margin_err}{Style.RESET_ALL}")
        # Return quantity but signal margin calculation failure
        return quantity_precise, None # Allow potential capping logic to proceed if quantity is valid

def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
    """
    Waits for a specific order ID to reach a 'closed' (filled for market/limit) or
    failed ('canceled', 'rejected', 'expired') status by polling `fetch_order`.

    Args:
        exchange: The CCXT exchange instance.
        order_id: The ID of the order to monitor.
        symbol: The market symbol of the order.
        timeout_seconds: Maximum time in seconds to wait for a final status.

    Returns:
        The final order dictionary if a terminal status (closed, canceled, rejected, expired)
        is reached within the timeout, or None if timed out or an error occurred during polling.
    """
    start_time = time.monotonic() # Use monotonic clock for measuring elapsed time
    order_id_short = format_order_id(order_id)
    logger.info(f"{Fore.CYAN}Observing order {order_id_short} ({symbol}) for fill/final status (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")

    while time.monotonic() - start_time < timeout_seconds:
        elapsed_time = time.monotonic() - start_time
        try:
            # Query the order's current status
            # Bybit V5 might need category, though fetch_order might handle it based on symbol
            params = {'category': 'linear'} # Add category for V5 consistency
            order = exchange.fetch_order(order_id, symbol, params=params)
            status = order.get('status')
            filled_qty = safe_decimal_conversion(order.get('filled'))
            logger.debug(f"Order {order_id_short} status: {status}, Filled: {filled_qty:.8f} ({elapsed_time:.1f}s elapsed)")

            # --- Check for Terminal Statuses ---
            if status == 'closed': # 'closed' typically means fully filled for market/limit, or triggered & filled for stop
                logger.success(f"{Fore.GREEN}Order {order_id_short} confirmed FILLED/CLOSED.{Style.RESET_ALL}")
                return order # Success, return final order state
            elif status in ['canceled', 'rejected', 'expired']:
                logger.error(f"{Fore.RED}Order {order_id_short} reached FAILED status: '{status}'.{Style.RESET_ALL}")
                return order # Failed state, return final order state

            # Continue polling if 'open', 'partially_filled' (for limit orders), or None/unknown status

            # --- Polling Delay ---
            # Use a slightly increasing delay to avoid hammering the API
            sleep_interval = min(0.5 + elapsed_time * 0.05, 3.0) # Start at 0.5s, increase slightly, max 3s
            time.sleep(sleep_interval)

        except ccxt.OrderNotFound:
            # This can happen briefly after placing, especially on busy exchanges. Keep trying within timeout.
            logger.warning(f"{Fore.YELLOW}Order {order_id_short} not found yet by exchange ({elapsed_time:.1f}s elapsed). Retrying...{Style.RESET_ALL}")
            # Wait a bit longer if not found initially
            time.sleep(min(1.5 + elapsed_time * 0.1, 4.0))
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            # Handle transient errors during polling
            logger.warning(f"{Fore.YELLOW}Disturbance checking order {order_id_short} ({elapsed_time:.1f}s elapsed): {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}")
            time.sleep(min(2.0 + elapsed_time * 0.1, 5.0)) # Wait longer on error before retrying
        except Exception as e_unexp:
            # Handle unexpected errors during polling
            logger.error(f"{Fore.RED}Unexpected error checking order {order_id_short} ({elapsed_time:.1f}s elapsed): {e_unexp}. Stopping check.{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            return None # Stop checking on unexpected error

    # If the loop finishes without returning, it timed out
    logger.error(f"{Fore.RED}Order {order_id_short} did NOT reach final status within {timeout_seconds}s timeout.{Style.RESET_ALL}")
    # Optionally try fetching one last time outside the loop to get the very latest status
    try:
        params = {'category': 'linear'}
        final_check_order = exchange.fetch_order(order_id, symbol, params=params)
        logger.warning(f"Final status check for timed-out order {order_id_short}: Status='{final_check_order.get('status')}', Filled='{final_check_order.get('filled')}'")
        # Return the final state even if it wasn't a terminal one, indicating timeout
        return final_check_order
    except Exception as final_e:
        logger.error(f"Failed final status check for timed-out order {order_id_short}: {final_e}")
        return None # Timeout failure

def place_risked_market_order(exchange: ccxt.Exchange, symbol: str, side: str,
                            risk_percentage: Decimal, current_atr: Optional[Decimal], sl_atr_multiplier: Decimal,
                            leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal,
                            tsl_percent: Decimal, tsl_activation_offset_percent: Decimal) -> Optional[Dict[str, Any]]:
    """
    Orchestrates the process of placing a market entry order with associated
    exchange-native fixed Stop Loss (ATR-based) and Trailing Stop Loss.

    Steps:
    1. Fetches necessary data (balance, market limits, price estimate).
    2. Calculates initial SL price estimate based on ATR.
    3. Calculates position size based on risk percentage and SL distance.
    4. Applies USDT position value cap if necessary.
    5. Performs margin checks against available balance with a buffer.
    6. Checks against market quantity limits (min/max).
    7. Places the market entry order.
    8. Waits for the entry order to fill.
    9. Calculates the ACTUAL fixed SL price based on the fill price.
    10. Places the exchange-native fixed Stop Loss order (`stopMarket` with `stopPrice`).
    11. Calculates the Trailing SL activation price.
    12. Places the exchange-native Trailing Stop Loss order (`stopMarket` with `trailingStop` & `activePrice`).
    13. Sends confirmation alerts.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        side: The side of the entry order ('buy' or 'sell').
        risk_percentage: Fraction of equity to risk.
        current_atr: The current ATR value (Decimal) for SL calculation.
        sl_atr_multiplier: Multiplier for ATR to determine fixed SL distance.
        leverage: The leverage setting.
        max_order_cap_usdt: Maximum allowed position value in USDT.
        margin_check_buffer: Multiplier for required margin check (e.g., 1.05 for 5% buffer).
        tsl_percent: Trailing stop percentage (e.g., 0.005 for 0.5%).
        tsl_activation_offset_percent: Price movement % required to activate TSL (e.g., 0.001 for 0.1%).

    Returns:
        The filled entry order dictionary (from CCXT) if the entry was successful,
        even if subsequent SL/TSL placement failed (position is open).
        Returns None if the entry process failed critically at any step before fill confirmation.
    """
    market_base = symbol.split('/')[0].split(':')[0] # For concise alerts
    order_side_color = Fore.GREEN if side == CONFIG.side_buy else Fore.RED
    logger.info(f"{order_side_color}{Style.BRIGHT}Place Order Ritual: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}")

    # --- Pre-computation & Validation ---
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Invalid ATR ({current_atr}) provided. Cannot calculate SL distance or place order.{Style.RESET_ALL}")
        return None

    entry_price_estimate: Optional[Decimal] = None
    initial_sl_price_estimate: Optional[Decimal] = None
    final_quantity: Optional[Decimal] = None
    market: Optional[Dict] = None
    category: Optional[str] = None
    min_qty: Optional[Decimal] = None
    max_qty: Optional[Decimal] = None
    min_price: Optional[Decimal] = None
    price_tick_size: Optional[Decimal] = None
    amount_step: Optional[Decimal] = None

    try:
        # === 1. Gather Resources: Balance, Market Info, Limits ===
        logger.debug("Gathering resources: Balance, Market Structure, Limits...")
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else 'inverse' # Determine category for V5 params

        # Fetch available balance using V5 specific parameters
        balance_params = {'category': category}
        balance = exchange.fetch_balance(params=balance_params)

        # Extract market limits and precision
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        precision = market.get('precision', {})

        min_qty_str = amount_limits.get('min')
        max_qty_str = amount_limits.get('max')
        min_price_str = price_limits.get('min')
        # Get precision details (tick sizes) for potential rounding/adjustments
        price_tick_size_str = precision.get('price') # Price granularity
        amount_step_str = precision.get('amount') # Amount granularity

        min_qty = safe_decimal_conversion(min_qty_str) if min_qty_str else None
        max_qty = safe_decimal_conversion(max_qty_str) if max_qty_str else None
        min_price = safe_decimal_conversion(min_price_str) if min_price_str else None
        price_tick_size = safe_decimal_conversion(price_tick_size_str) if price_tick_size_str else None
        amount_step = safe_decimal_conversion(amount_step_str) if amount_step_str else None

        # Extract USDT balance details (use 'total' for equity, 'free' for margin check)
        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        # V5 balance structure might differ slightly, check CCXT unified structure keys
        usdt_total = safe_decimal_conversion(usdt_balance.get('total')) # Total equity (incl. PnL)
        usdt_free = safe_decimal_conversion(usdt_balance.get('free'))   # Available for new orders/margin
        # Use total equity for risk calculation if available and positive, otherwise fall back to free (less ideal but safer than assuming total)
        usdt_equity = usdt_total if usdt_total is not None and usdt_total > 0 else usdt_free

        if usdt_equity is None or usdt_equity <= Decimal("0"):
            logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Zero or Invalid Equity ({usdt_equity}). Cannot calculate risk.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Zero/Invalid Equity ({usdt_equity:.2f})")
            return None
        if usdt_free is None or usdt_free < Decimal("0"): # Free margin shouldn't be negative
            logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Invalid Free Margin ({usdt_free}). Cannot place orders.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Invalid Free Margin ({usdt_free:.2f})")
            return None

        logger.debug(f"Resources: Equity={usdt_equity:.4f}, Free Margin={usdt_free:.4f} {CONFIG.usdt_symbol}")
        if min_qty: logger.debug(f"Market Limits: Min Qty={min_qty:.8f}")
        if max_qty: logger.debug(f"Market Limits: Max Qty={max_qty:.8f}")
        if min_price: logger.debug(f"Market Limits: Min Price={min_price:.8f}")
        if price_tick_size: logger.debug(f"Market Precision: Price Tick={price_tick_size:.8f}")
        if amount_step: logger.debug(f"Market Precision: Amount Step={amount_step:.8f}")

        # === 2. Estimate Entry Price - Peering into the immediate future ===
        logger.debug("Estimating entry price using shallow order book...")
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")

        if side == CONFIG.side_buy and best_ask and best_ask > 0:
            entry_price_estimate = best_ask # Estimate buying at best ask
        elif side == CONFIG.side_sell and best_bid and best_bid > 0:
            entry_price_estimate = best_bid # Estimate selling at best bid
        else:
            # Fallback: Fetch last traded price if OB data is unreliable or missing
            logger.warning(f"{Fore.YELLOW}Shallow OB failed for entry price estimate (Ask:{best_ask}, Bid:{best_bid}). Fetching ticker...{Style.RESET_ALL}")
            try:
                ticker_params = {'category': category} # V5 ticker might need category
                ticker = exchange.fetch_ticker(symbol, params=ticker_params)
                last_price = safe_decimal_conversion(ticker.get('last'))
                if last_price > 0:
                    entry_price_estimate = last_price
                    logger.debug(f"Using ticker last price for estimate: {entry_price_estimate}")
                else:
                    raise ValueError(f"Ticker 'last' price invalid: {last_price}")
            except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
                logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Failed to get valid entry price estimate from OB or Ticker: {e}{Style.RESET_ALL}")
                return None # Cannot proceed without a price estimate
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === 3. Calculate Initial Stop Loss Price (Estimate) - The First Ward ===
        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (entry_price_estimate - sl_distance) if side == CONFIG.side_buy else (entry_price_estimate + sl_distance)

        # Ensure SL price estimate is valid and respects minimum price if applicable
        if initial_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Initial SL price calculation resulted in zero or negative value ({initial_sl_price_raw:.4f}). Cannot proceed.{Style.RESET_ALL}")
            return None
        if min_price is not None and initial_sl_price_raw < min_price:
            logger.warning(f"{Fore.YELLOW}Initial SL price estimate {initial_sl_price_raw:.4f} is below market min price {min_price}. Adjusting SL estimate up to min price.{Style.RESET_ALL}")
            initial_sl_price_raw = min_price # Adjust SL estimate upwards

        # Format the estimated SL price according to market rules BEFORE using it in size calculation
        try:
            initial_sl_price_estimate_str = format_price(exchange, symbol, initial_sl_price_raw)
            initial_sl_price_estimate = Decimal(initial_sl_price_estimate_str)
            if initial_sl_price_estimate <= 0: raise ValueError("Formatted SL price estimate invalid (<=0)")
            # Ensure SL estimate isn't identical to entry estimate after formatting
            if abs(initial_sl_price_estimate - entry_price_estimate) < (price_tick_size or Decimal("1e-8")):
                 logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Formatted initial SL price ({initial_sl_price_estimate}) is too close to entry estimate ({entry_price_estimate}). Increase ATR multiplier or check precision.{Style.RESET_ALL}")
                 return None
            logger.info(f"Calculated Initial SL Price (Estimate) ~ {Fore.YELLOW}{initial_sl_price_estimate:.4f}{Style.RESET_ALL} (Based on ATR: {current_atr:.4f}, Multiplier: {sl_atr_multiplier}, Dist: {sl_distance:.4f})")
        except (ValueError, InvalidOperation) as e:
             logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Failed to format/validate initial SL price estimate '{initial_sl_price_raw:.4f}': {e}{Style.RESET_ALL}")
             return None

        # === 4. Calculate Position Size - Determining the Energy Input ===
        # Use the equity, estimates, leverage, and pass exchange for formatting within the function
        calc_qty, req_margin_est = calculate_position_size(
            equity=usdt_equity,
            risk_per_trade_pct=risk_percentage,
            entry_price=entry_price_estimate,
            stop_loss_price=initial_sl_price_estimate, # Use formatted estimate
            leverage=leverage,
            symbol=symbol,
            exchange=exchange
        )
        if calc_qty is None: # Margin calculation failure is handled separately
            logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Failed risk calculation. Cannot determine position size.{Style.RESET_ALL}")
            return None
        final_quantity = calc_qty # Start with the risk-based quantity

        # === 5. Apply Max Order Cap - Limiting the Power ===
        pos_value_estimate = final_quantity * entry_price_estimate
        logger.debug(f"Estimated position value based on risk calc: {pos_value_estimate:.4f} USDT (Cap: {max_order_cap_usdt:.4f} USDT)")
        if max_order_cap_usdt > 0 and pos_value_estimate > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}Calculated order value {pos_value_estimate:.4f} exceeds Max Cap {max_order_cap_usdt:.4f}. Capping quantity.{Style.RESET_ALL}")
            try:
                # Calculate capped quantity based on max USDT value and estimated entry price
                final_quantity_capped_raw = max_order_cap_usdt / entry_price_estimate
                # Format the capped quantity according to market rules *then* convert back
                final_quantity_str = format_amount(exchange, symbol, final_quantity_capped_raw)
                final_quantity = Decimal(final_quantity_str)
                # Recalculate estimated margin based on the capped quantity
                if final_quantity > 0 and leverage > 0:
                    req_margin_est = (final_quantity * entry_price_estimate / Decimal(leverage))
                    logger.info(f"Quantity capped to: {final_quantity:.8f}, New Est. Margin Req.: {req_margin_est:.4f} USDT")
                else:
                    req_margin_est = None # Cannot estimate margin if capped qty is zero
            except (DivisionByZero, ValueError, InvalidOperation, Exception) as cap_err:
                logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Failed to calculate or format capped quantity: {cap_err}{Style.RESET_ALL}")
                return None

        # === 6. Check Limits & Margin Availability - Final Preparations ===
        if abs(final_quantity) <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Final Quantity negligible ({final_quantity:.8f}) after risk calc/capping. Cannot place order.{Style.RESET_ALL}")
            return None
        # Check against minimum order size
        if min_qty is not None and final_quantity < min_qty:
            logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Final Quantity {final_quantity:.8f} is less than market minimum allowed {min_qty:.8f}. Cannot place order.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Qty {final_quantity:.8f} < Min {min_qty:.8f}")
            return None
        # Check against maximum order size (should be handled by cap, but double-check)
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"{Fore.YELLOW}Final Quantity {final_quantity:.8f} exceeds market maximum {max_qty:.8f}. Adjusting down to max allowed.{Style.RESET_ALL}")
            # Re-format the max quantity according to market rules
            try:
                final_quantity_str = format_amount(exchange, symbol, max_qty)
                final_quantity = Decimal(final_quantity_str)
                # Recalculate margin again if qty changed due to max limit
                if final_quantity > 0 and leverage > 0:
                     req_margin_est = (final_quantity * entry_price_estimate / Decimal(leverage))
                else:
                     req_margin_est = None
            except Exception as max_fmt_err:
                 logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Failed to format max-capped quantity: {max_fmt_err}{Style.RESET_ALL}")
                 return None

        # Final margin calculation based on potentially adjusted final_quantity
        if req_margin_est is None: # Check if margin calc failed at any point
             logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Failed to estimate required margin for quantity {final_quantity:.8f}. Cannot proceed.{Style.RESET_ALL}")
             return None

        req_margin_buffered = req_margin_est * margin_check_buffer # Add safety buffer
        logger.debug(f"Final Margin Check: Need ~{req_margin_est:.4f} (Buffered: {req_margin_buffered:.4f}), Have Free: {usdt_free:.4f}")

        # Check if sufficient free margin is available
        if usdt_free < req_margin_buffered:
            logger.error(f"{Fore.RED}Place Order Error ({side.upper()}): Insufficient FREE margin. Need ~{req_margin_buffered:.4f} (incl. buffer), Have {usdt_free:.4f}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{req_margin_buffered:.2f}, Have {usdt_free:.2f})")
            return None
        logger.info(f"{Fore.GREEN}Final Order Details Pre-Submission: Side={side.upper()}, Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={req_margin_est:.4f}. Margin check OK.{Style.RESET_ALL}")

        # === 7. Place Entry Market Order - Unleashing the Energy ===
        entry_order: Optional[Dict[str, Any]] = None
        order_id: Optional[str] = None
        try:
            qty_float = float(final_quantity) # CCXT expects float for amount
            entry_bg_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
            entry_fg_color = Fore.BLACK # Use black text on green/red background for visibility
            logger.warning(f"{entry_bg_color}{entry_fg_color}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY Order: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")

            # Create the market order - V5 needs category, ensure reduceOnly is False for entry
            entry_params = {
                'category': category,
                'reduceOnly': False
            }
            entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params=entry_params)
            order_id = entry_order.get('id')

            if not order_id:
                # This is highly unexpected and problematic if the order was actually placed
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: Entry order potentially placed but NO Order ID received from exchange! Position state unknown. MANUAL INTERVENTION REQUIRED! Response: {entry_order}{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Entry placed but NO ID received! MANUAL CHECK!")
                # Cannot proceed reliably without the order ID to track fill and place SL/TSL
                return None # Signal critical failure
            logger.success(f"{Fore.GREEN}Market Entry Order submitted. ID: {format_order_id(order_id)}. Awaiting fill confirmation...{Style.RESET_ALL}")

        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {type(e).__name__} - {e}{Style.RESET_ALL}")
            # Check for specific informative errors if possible
            if isinstance(e, ccxt.InsufficientFunds):
                 send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Insufficient Funds on Entry!")
            else:
                 send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}")
            return None # Failed to place entry, cannot proceed
        except Exception as e_place:
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED ERROR Placing Entry Order: {e_place}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Unexpected entry placement error: {type(e_place).__name__}")
            return None

        # === 8. Wait for Entry Fill Confirmation - Observing the Impact ===
        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)

        if not filled_entry:
            # Timeout or unexpected error during wait_for_order_fill
            logger.error(f"{Fore.RED}Entry order {format_order_id(order_id)} fill status UNKNOWN due to timeout or check error. Position state uncertain.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry {format_order_id(order_id)} fill TIMEOUT/ERROR. MANUAL CHECK!")
            # Attempt to cancel the potentially stuck/unfilled order as a safety measure
            try:
                logger.warning(f"Attempting to cancel potentially unconfirmed order {format_order_id(order_id)}")
                cancel_params = {'category': category}
                exchange.cancel_order(order_id, symbol, params=cancel_params)
                logger.info(f"Cancel request sent for order {format_order_id(order_id)}")
            except Exception as cancel_e:
                logger.warning(f"Could not cancel order {format_order_id(order_id)} (may be filled, already gone, or error): {cancel_e}")
            # Even if cancel fails, we cannot proceed to SL/TSL placement without confirmed entry
            return None

        # Check the status of the fetched order
        final_status = filled_entry.get('status')
        if final_status != 'closed':
            # Handle non-filled terminal statuses or unexpected statuses after wait
            logger.error(f"{Fore.RED}Entry order {format_order_id(order_id)} did not confirm filled. Final Status: '{final_status}'. Position state uncertain.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Entry {format_order_id(order_id)} final status '{final_status}'. MANUAL CHECK!")
            # If the order failed (canceled/rejected), no position was opened.
            # If status is still 'open' or something else after timeout, it's problematic.
            # Attempt cancellation again just in case.
            if final_status == 'open':
                 try:
                     logger.warning(f"Attempting to cancel order {format_order_id(order_id)} stuck in '{final_status}' status.")
                     cancel_params = {'category': category}
                     exchange.cancel_order(order_id, symbol, params=cancel_params)
                     logger.info(f"Cancel request sent for stuck order {format_order_id(order_id)}")
                 except Exception as cancel_e:
                     logger.warning(f"Could not cancel stuck order {format_order_id(order_id)}: {cancel_e}")
            return None # Cannot proceed without a confirmed 'closed' entry

        # === 9. Extract Actual Fill Details - Reading the Result ===
        # Use 'average' if available and valid, otherwise calculate from cost/filled if possible
        avg_fill_price = safe_decimal_conversion(filled_entry.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry.get('filled'))
        cost = safe_decimal_conversion(filled_entry.get('cost')) # Total cost in quote currency (USDT)

        # --- Validate Fill Details ---
        # Ensure filled quantity matches expected (within tolerance)
        qty_diff = abs(filled_qty - final_quantity)
        if qty_diff > (CONFIG.position_qty_epsilon * Decimal("10")): # Allow slightly larger tolerance
             logger.warning(f"{Fore.YELLOW}Fill quantity mismatch! Expected: {final_quantity:.8f}, Filled: {filled_qty:.8f} (Diff: {qty_diff:.8e}). Using actual filled quantity for SL/TSL.{Style.RESET_ALL}")
             # Proceed using the actual filled_qty, but log the discrepancy.

        if abs(filled_qty) <= CONFIG.position_qty_epsilon:
            logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill quantity ({filled_qty}) reported for order {format_order_id(order_id)} despite 'closed' status. Position state unknown! MANUAL CHECK REQUIRED!{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill qty {filled_qty} for {format_order_id(order_id)}! MANUAL CHECK!")
            # Return the problematic order details but signal failure state by context
            # The calling logic should handle this as a failure.
            return filled_entry # Return filled order but it's problematic

        # Ensure fill price is valid
        if avg_fill_price <= 0:
             # Try to estimate fill price from cost/qty if average is missing/zero and cost/qty are valid
             if cost > 0 and filled_qty > 0:
                 try:
                     avg_fill_price = cost / filled_qty
                     logger.warning(f"{Fore.YELLOW}Fill price 'average' was missing/zero. Estimated from cost/filled: {avg_fill_price:.4f}{Style.RESET_ALL}")
                 except (DivisionByZero, InvalidOperation):
                      avg_fill_price = Decimal("-1") # Mark as invalid if estimation fails
             else:
                 avg_fill_price = Decimal("-1") # Mark as invalid

        if avg_fill_price <= 0:
             logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill price ({avg_fill_price}) reported or estimated for order {format_order_id(order_id)}. Cannot calculate SL accurately! MANUAL CHECK REQUIRED!{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ORDER FAIL ({side.upper()}): Invalid fill price {avg_fill_price} for {format_order_id(order_id)}! MANUAL CHECK!")
             # Position is likely open but SL cannot be placed reliably. Attempt emergency close.
             logger.warning("Attempting emergency close due to invalid fill price...")
             # Use the confirmed fill quantity but mark position side based on original intent
             emergency_pos_state = {'side': CONFIG.pos_long if side == CONFIG.side_buy else CONFIG.pos_short, 'qty': filled_qty}
             close_position(exchange, symbol, emergency_pos_state, reason="Invalid Fill Price Post-Entry")
             # Return the problematic filled entry order, but the overall process failed.
             return filled_entry

        # Log confirmed entry
        fill_bg_color = Back.GREEN if side == CONFIG.side_buy else Back.RED
        fill_fg_color = Fore.BLACK
        logger.success(f"{fill_bg_color}{fill_fg_color}{Style.BRIGHT}ENTRY CONFIRMED: {format_order_id(order_id)}. Filled: {filled_qty:.8f} @ Avg: {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}")

        # --- SL & TSL Placement (Proceed using actual filled_qty and avg_fill_price) ---
        sl_placed_successfully = False
        tsl_placed_successfully = False
        actual_sl_price_str = "N/A"
        tsl_act_price_str = "N/A"
        sl_order_id_short = "N/A"
        tsl_order_id_short = "N/A"

        # === 10. Calculate ACTUAL Fixed Stop Loss Price - Setting the Ward ===
        # Use the actual average fill price and the original ATR distance
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.side_buy else (avg_fill_price + sl_distance)

        # Apply min price constraint again based on actual fill price
        if actual_sl_price_raw <= 0:
             logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: ACTUAL SL price calculation resulted in zero or negative value ({actual_sl_price_raw:.4f}) based on fill price {avg_fill_price:.4f}. Cannot place Fixed SL!{Style.RESET_ALL}")
             # Position is open without fixed SL protection. TSL might still be placed.
             send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Invalid ACTUAL Fixed SL price ({actual_sl_price_raw:.4f})! Fixed SL FAILED.")
             # Do not attempt emergency close here, let TSL attempt proceed.
        else:
            if min_price is not None and actual_sl_price_raw < min_price:
                 logger.warning(f"{Fore.YELLOW}Actual Fixed SL price {actual_sl_price_raw:.4f} is below market min price {min_price}. Adjusting SL up to min price.{Style.RESET_ALL}")
                 actual_sl_price_raw = min_price # Adjust SL upwards

            # Format the final SL price
            try:
                actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
                actual_sl_price_float = float(actual_sl_price_str) # For CCXT param
                if actual_sl_price_float <= 0: raise ValueError("Formatted actual SL price invalid (<=0)")
                logger.info(f"Calculated ACTUAL Fixed SL Trigger Price: {Fore.YELLOW}{actual_sl_price_str}{Style.RESET_ALL}")

                # === 11. Place Initial Fixed Stop Loss Order - The Static Ward ===
                try:
                    sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy # Opposite side for SL
                    sl_qty_str = format_amount(exchange, symbol, filled_qty) # Use actual filled quantity
                    sl_qty_float = float(sl_qty_str)

                    if abs(sl_qty_float) <= float(CONFIG.position_qty_epsilon):
                         raise ValueError(f"Formatted SL quantity negligible: {sl_qty_float}")

                    logger.info(f"{Fore.CYAN}Weaving Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, TriggerPx: {actual_sl_price_str}{Style.RESET_ALL}")
                    # Bybit V5 stop order params using stopMarket type via CCXT:
                    # 'stopPrice': The trigger price (float)
                    # 'reduceOnly': Must be true for SL/TP orders (boolean)
                    # 'category': Required for V5
                    sl_params = {
                        'category': category,
                        'stopPrice': actual_sl_price_float,
                        'reduceOnly': True,
                        # 'tpslMode': 'Full' # Optional: Specify if partial SL/TP needed (default usually Full)
                        # 'slOrderType': 'Market' # Optional: Bybit default is Market for stopMarket
                    }
                    sl_order = exchange.create_order(symbol, 'stopMarket', sl_side, sl_qty_float, params=sl_params)
                    sl_order_id_short = format_order_id(sl_order.get('id'))
                    logger.success(f"{Fore.GREEN}Initial Fixed SL ward placed. ID: {sl_order_id_short}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}")
                    sl_placed_successfully = True
                except (ValueError, ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
                    logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Initial Fixed SL ward: {e}{Style.RESET_ALL}")
                    # Log specific Bybit errors if possible
                    if isinstance(e, ccxt.ExchangeError): logger.debug(f"Fixed SL Placement Error Details: {e.args}")
                    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed initial FIXED SL placement: {type(e).__name__}")
                    # Continue to TSL placement attempt
                except Exception as e_sl:
                     logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED ERROR placing Initial Fixed SL ward: {e_sl}{Style.RESET_ALL}")
                     logger.debug(traceback.format_exc())
                     send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Unexpected initial Fixed SL error: {type(e_sl).__name__}")

            except (ValueError, InvalidOperation, TypeError) as e_format:
                logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL FAILURE: Failed to format/validate ACTUAL Fixed SL price '{actual_sl_price_raw:.4f}': {e_format}. Cannot place Fixed SL!{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): Failed format/validate ACTUAL Fixed SL price! Fixed SL FAILED.")
                # Fixed SL failed due to formatting, proceed to TSL attempt

        # === 12. Place Trailing Stop Loss Order - The Adaptive Shield ===
        try:
            # Calculate TSL activation price based on actual fill price and offset percentage
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.side_buy else (avg_fill_price - act_offset)

            # Apply min price constraint to activation price
            if act_price_raw <= 0:
                 raise ValueError(f"Invalid TSL activation price calculated (<=0): {act_price_raw:.4f}")

            if min_price is not None and act_price_raw < min_price:
                logger.warning(f"{Fore.YELLOW}TSL activation price {act_price_raw:.4f} is below market min price {min_price}. Adjusting activation up to min price.{Style.RESET_ALL}")
                act_price_raw = min_price

            # Format activation price
            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            if tsl_act_price_float <= 0: raise ValueError("Formatted TSL activation price invalid (<=0)")

            # Prepare TSL parameters for Bybit V5 via CCXT
            tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy # Opposite side
            # Bybit V5 uses 'trailingStop' for percentage distance (e.g., "0.5" for 0.5%)
            # Convert our decimal percentage (e.g., 0.005) to percentage string (e.g., "0.5")
            tsl_trail_value_str = str((tsl_percent * Decimal("100")).normalize()) # .normalize() removes trailing zeros
            tsl_qty_str = format_amount(exchange, symbol, filled_qty) # Use actual filled quantity
            tsl_qty_float = float(tsl_qty_str)

            if abs(tsl_qty_float) <= float(CONFIG.position_qty_epsilon):
                 raise ValueError(f"Formatted TSL quantity negligible: {tsl_qty_float}")

            logger.info(f"{Fore.CYAN}Weaving Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")

            # Bybit V5 TSL params via CCXT using 'stopMarket' type:
            # 'trailingStop': Percentage value as a string (e.g., "0.5" for 0.5%) - This is the distance
            # 'activePrice': Activation trigger price (float)
            # 'reduceOnly': Must be True (boolean)
            # 'category': Required for V5
            tsl_params = {
                'category': category,
                'trailingStop': tsl_trail_value_str, # Distance value as percentage string
                'activePrice': tsl_act_price_float,   # Activation price
                'reduceOnly': True,
                # 'tpslMode': 'Full' # Optional
                # 'slOrderType': 'Market' # Optional
            }
            tsl_order = exchange.create_order(symbol, 'stopMarket', tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id_short = format_order_id(tsl_order.get('id'))
            logger.success(f"{Fore.GREEN}Trailing SL shield placed. ID: {tsl_order_id_short}, Trail%: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")
            tsl_placed_successfully = True

        except (ValueError, ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FAILED to place Trailing SL shield: {e}{Style.RESET_ALL}")
            if isinstance(e, ccxt.ExchangeError): logger.debug(f"TSL Placement Error Details: {e.args}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}")
            # If TSL fails but initial SL was placed, the position is still protected initially.
        except Exception as e_tsl:
             logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}UNEXPECTED ERROR placing Trailing SL shield: {e_tsl}{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR ({side.upper()}): Unexpected TSL error: {type(e_tsl).__name__}")


        # === 13. Final Confirmation & Alert ===
        # Send comprehensive SMS summarizing the outcome
        sl_status = f"~{actual_sl_price_str}(ID:{sl_order_id_short})" if sl_placed_successfully else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        tsl_status = f"{tsl_percent:.2%}@~{tsl_act_price_str}(ID:{tsl_order_id_short})" if tsl_placed_successfully else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        sms_msg = (f"[{market_base}/{CONFIG.strategy_name}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                   f"FixedSL: {sl_status}. "
                   f"TrailingSL: {tsl_status}. "
                   f"EntryID:{format_order_id(order_id)}")
        send_sms_alert(sms_msg)

        if not sl_placed_successfully and not tsl_placed_successfully:
             # Critical situation: Entry confirmed, but NO protective orders placed.
             logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL WARNING: Entry confirmed but BOTH Fixed SL and Trailing SL placement failed! Position is UNPROTECTED. Manual intervention likely required.{Style.RESET_ALL}")
             # Send critical alert (already sent specific failure alerts above, add summary)
             send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ({side.upper()}): BOTH SL & TSL FAILED! POS UNPROTECTED! EntryID:{format_order_id(order_id)}")
             # Consider attempting emergency close again here? Or rely on shutdown handler? For now, just alert critically.

        # Return the details of the successfully filled entry order.
        # The presence of this return value indicates the entry succeeded,
        # even if subsequent SL/TSL steps had issues (logged above).
        return filled_entry

    # --- Handle Errors During Setup Phase (Before Placing Entry Order) ---
    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, InvalidOperation) as e:
        # Catch errors occurring during the setup phase (balance check, price estimate, size calc, margin check)
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Pre-entry process failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Pre-entry setup failed: {type(e).__name__}")
    except Exception as e_overall:
         # Catch any other unexpected errors during the entire process
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Place Order Ritual ({side.upper()}): Unexpected overall failure: {type(e_overall).__name__} - {e_overall}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ORDER FAIL ({side.upper()}): Unexpected overall error: {type(e_overall).__name__}")

    # Return None if the overall process failed before confirming the entry fill
    return None

def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> int:
    """
    Attempts to cancel all open orders for the specified symbol using `cancel_all_orders`.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol for which to cancel orders.
        reason: A string indicating the reason for cancellation (for logging).

    Returns:
        The number of orders reported as cancelled or attempted to cancel.
        Note: `cancel_all_orders` response varies; this often returns 1 on success,
        but the actual number cancelled might differ. Focus is on the attempt.
    """
    logger.info(f"{Fore.CYAN}Order Cleanup Ritual: Initiating for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    cancelled_count: int = 0
    market_base = symbol.split('/')[0].split(':')[0]
    category: Optional[str] = None

    try:
        # Determine category for V5 params
        market = exchange.market(symbol)
        category = 'linear' if market.get('linear') else 'inverse'

        # Check if cancel_all_orders is supported (CCXT might emulate it using fetchOpenOrders + cancelOrder)
        if not exchange.has.get('cancelAllOrders'):
            logger.warning(f"{Fore.YELLOW}Order Cleanup: Exchange '{exchange.id}' CCXT instance does not directly support cancelAllOrders. CCXT might emulate. Proceeding cautiously.{Style.RESET_ALL}")
            # CCXT might still work by fetching open orders and cancelling individually.

        # Bybit V5 cancelAllOrders requires category and optionally symbol
        params = {'category': category}
        logger.warning(f"{Fore.YELLOW}Order Cleanup: Attempting to cancel ALL open orders for {symbol} (Category: {category})...{Style.RESET_ALL}")

        # Use cancel_all_orders if available and preferred for efficiency
        response = exchange.cancel_all_orders(symbol=symbol, params=params)

        # Log the response (structure varies between exchanges)
        logger.info(f"{Fore.CYAN}Order Cleanup: cancel_all_orders request sent for {symbol}. Response: {response}{Style.RESET_ALL}")
        # We can't easily determine the exact number cancelled from the typical response.
        # Assume success if no error is raised. We can increment count conceptually.
        cancelled_count = 1 # Indicate an attempt was made

    except ccxt.ExchangeError as e:
        # Handle potential errors like "no orders to cancel" gracefully
        err_str = str(e).lower()
        # Bybit V5 code 110001 often means "no orders found" or generic parameter error
        if "order does not exist" in err_str or "no orders found" in err_str or "110001" in err_str:
            logger.info(f"{Fore.CYAN}Order Cleanup: No open orders found for {symbol} to cancel (or exchange reported success despite no orders).{Style.RESET_ALL}")
            cancelled_count = 0 # No orders were actually cancelled
        else:
            logger.error(f"{Fore.RED}Order Cleanup: FAILED to cancel orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] WARNING: Failed cancelAllOrders ({reason}): {type(e).__name__}. Check manually.")
            cancelled_count = -1 # Indicate failure
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
        logger.error(f"{Fore.RED}Order Cleanup: Network/Timeout error during cancelAllOrders for {symbol}: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] WARNING: Network/Timeout on cancelAllOrders ({reason}). Check manually.")
        cancelled_count = -1 # Indicate failure
    except Exception as e_cancel:
         logger.error(f"{Fore.RED}Order Cleanup: Unexpected error during cancelAllOrders for {symbol}: {type(e_cancel).__name__} - {e_cancel}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())
         send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] WARNING: Unexpected error on cancelAllOrders ({reason}). Check manually.")
         cancelled_count = -1 # Indicate failure

    # Log summary based on outcome
    if cancelled_count >= 0:
        logger.info(f"{Fore.CYAN}Order Cleanup Ritual Finished for {symbol}. Attempt successful (reported {cancelled_count} actions/attempts).{Style.RESET_ALL}")
    else:
         logger.error(f"{Fore.RED}Order Cleanup Ritual Encountered Errors for {symbol}. Manual check of open orders recommended.{Style.RESET_ALL}")

    # Return conceptual count (1 for attempt, 0 for none found, -1 for error)
    return cancelled_count


# --- Strategy Signal Generation - Interpreting the Omens ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """
    Generates entry and exit signals based on the selected strategy's interpretation
    of indicator values from the provided DataFrame.

    Requires at least 2 rows in the DataFrame for comparing current vs. previous candle values.

    Args:
        df: Pandas DataFrame containing OHLCV data and calculated indicator columns.
        strategy_name: The name of the strategy to apply (must match CONFIG.strategy_name).

    Returns:
        A dictionary containing boolean signals:
        - 'enter_long': True if conditions for a long entry are met.
        - 'enter_short': True if conditions for a short entry are met.
        - 'exit_long': True if conditions to exit a long position are met.
        - 'exit_short': True if conditions to exit a short position are met.
        - 'exit_reason': A string explaining the reason for the exit signal.
    """
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit Signal"}
    required_rows = 2 # Need current and previous candle for comparisons

    if df is None or len(df) < required_rows:
        logger.debug(f"Signal Gen ({strategy_name}): Insufficient data ({len(df) if df is not None else 0} rows, need {required_rows}) for signal generation.")
        return signals # Not enough data for comparisons

    # Access last (most recent closed) and previous candle data safely
    try:
        last = df.iloc[-1] # Latest closed candle's data
        prev = df.iloc[-2] # Previous candle's data
    except IndexError:
         logger.error(f"Signal Gen ({strategy_name}): Error accessing DataFrame rows -1 or -2 (len: {len(df)}). Cannot generate signals.")
         return signals

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Required columns: 'st_long', 'st_short' (primary flips), 'confirm_trend' (confirmation state)
            primary_long_flip = last.get('st_long', False)     # True if primary ST flipped long this candle
            primary_short_flip = last.get('st_short', False)    # True if primary ST flipped short this candle
            confirm_is_up = last.get('confirm_trend', pd.NA) # Boolean: True if confirmation ST is currently up

            # Validate necessary data is available
            if pd.isna(confirm_is_up) or primary_long_flip is None or primary_short_flip is None:
                 logger.warning(f"Signal Gen ({strategy_name}): Skipping due to missing indicator values (PrimaryFlipL={primary_long_flip}, PrimaryFlipS={primary_short_flip}, ConfirmUp={confirm_is_up}).")
                 return signals

            # Enter Long: Primary ST flips long AND Confirmation ST is currently in uptrend
            if primary_long_flip and confirm_is_up is True:
                signals['enter_long'] = True
            # Enter Short: Primary ST flips short AND Confirmation ST is currently in downtrend
            if primary_short_flip and confirm_is_up is False:
                signals['enter_short'] = True
            # Exit Long: Primary ST flips short (exit regardless of confirmation trend)
            if primary_short_flip:
                signals['exit_long'] = True
                signals['exit_reason'] = "Primary ST Flipped Short"
            # Exit Short: Primary ST flips long (exit regardless of confirmation trend)
            if primary_long_flip:
                signals['exit_short'] = True
                signals['exit_reason'] = "Primary ST Flipped Long"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            # Required columns: 'stochrsi_k', 'stochrsi_d', 'momentum'
            k_now = last.get('stochrsi_k', pd.NA)
            d_now = last.get('stochrsi_d', pd.NA)
            mom_now = last.get('momentum', pd.NA)
            k_prev = prev.get('stochrsi_k', pd.NA)
            d_prev = prev.get('stochrsi_d', pd.NA)

            # Validate necessary data is available (Decimal comparisons handle None implicitly via TypeError)
            if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]):
                logger.debug(f"Signal Gen ({strategy_name}): Skipping due to NA values (k={k_now}, d={d_now}, mom={mom_now}, k_prev={k_prev}, d_prev={d_prev})")
                return signals

            # Enter Long: K crosses above D from below oversold, AND Momentum is positive
            if k_prev <= d_prev and k_now > d_now and k_now < CONFIG.stochrsi_oversold and mom_now > CONFIG.position_qty_epsilon:
                signals['enter_long'] = True
            # Enter Short: K crosses below D from above overbought, AND Momentum is negative
            if k_prev >= d_prev and k_now < d_now and k_now > CONFIG.stochrsi_overbought and mom_now < -CONFIG.position_qty_epsilon:
                signals['enter_short'] = True
            # Exit Long: K crosses below D (general exit signal)
            if k_prev >= d_prev and k_now < d_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "StochRSI K crossed below D"
            # Exit Short: K crosses above D (general exit signal)
            if k_prev <= d_prev and k_now > d_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "StochRSI K crossed above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            # Required columns: 'ehlers_fisher', 'ehlers_signal'
            fish_now = last.get('ehlers_fisher', pd.NA)
            sig_now = last.get('ehlers_signal', pd.NA)
            fish_prev = prev.get('ehlers_fisher', pd.NA)
            sig_prev = prev.get('ehlers_signal', pd.NA)

            if any(pd.isna(v) for v in [fish_now, sig_now, fish_prev, sig_prev]):
                logger.debug(f"Signal Gen ({strategy_name}): Skipping due to NA values (fish={fish_now}, sig={sig_now}, fish_prev={fish_prev}, sig_prev={sig_prev})")
                return signals

            # Enter Long: Fisher line crosses above Signal line
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['enter_long'] = True
            # Enter Short: Fisher line crosses below Signal line
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals['enter_short'] = True
            # Exit Long: Fisher crosses below Signal line (same condition as enter short)
            if fish_prev >= sig_prev and fish_now < sig_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "Ehlers Fisher crossed Short"
            # Exit Short: Fisher crosses above Signal line (same condition as enter long)
            if fish_prev <= sig_prev and fish_now > sig_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "Ehlers Fisher crossed Long"

        # --- Ehlers MA Cross Logic (Using EMA Placeholder) ---
        elif strategy_name == "EHLERS_MA_CROSS":
            # Required columns: 'fast_ema', 'slow_ema'
            fast_ma_now = last.get('fast_ema', pd.NA)
            slow_ma_now = last.get('slow_ema', pd.NA)
            fast_ma_prev = prev.get('fast_ema', pd.NA)
            slow_ma_prev = prev.get('slow_ema', pd.NA)

            if any(pd.isna(v) for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]):
                logger.debug(f"Signal Gen ({strategy_name} - EMA): Skipping due to NA values (fast={fast_ma_now}, slow={slow_ma_now}, fast_prev={fast_ma_prev}, slow_prev={slow_ma_prev})")
                return signals

            # Enter Long: Fast EMA crosses above Slow EMA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['enter_long'] = True
            # Enter Short: Fast EMA crosses below Slow EMA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['enter_short'] = True
            # Exit Long: Fast EMA crosses below Slow EMA (same condition as enter short)
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['exit_long'] = True
                signals['exit_reason'] = f"Fast EMA({CONFIG.ehlers_fast_period}) crossed below Slow EMA({CONFIG.ehlers_slow_period})"
            # Exit Short: Fast EMA crosses above Slow EMA (same condition as enter long)
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['exit_short'] = True
                signals['exit_reason'] = f"Fast EMA({CONFIG.ehlers_fast_period}) crossed above Slow EMA({CONFIG.ehlers_slow_period})"

    # --- Error Handling for Signal Generation ---
    except KeyError as e:
        logger.error(f"{Fore.RED}Signal Generation Error ({strategy_name}): Missing expected indicator column in DataFrame: {e}. Check indicator calculation functions and prefixes.{Style.RESET_ALL}")
        # Reset all signals to False on critical error
        signals = {k: False if isinstance(v, bool) else v for k, v in signals.items()}
    except TypeError as e:
         logger.error(f"{Fore.RED}Signal Generation Error ({strategy_name}): Type error during comparison, likely due to unexpected None/NA value: {e}. Check indicator data.{Style.RESET_ALL}")
         logger.debug(f"Last row data: {last.to_dict()}")
         logger.debug(f"Prev row data: {prev.to_dict()}")
         signals = {k: False if isinstance(v, bool) else v for k, v in signals.items()}
    except Exception as e:
        logger.error(f"{Fore.RED}Signal Generation Error ({strategy_name}): Unexpected disturbance during signal evaluation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        signals = {k: False if isinstance(v, bool) else v for k, v in signals.items()}

    # Log generated signals only if a signal is active
    active_signals = {k: v for k, v in signals.items() if isinstance(v, bool) and v}
    if active_signals:
        logger.debug(f"Strategy Signals ({strategy_name}): {active_signals}")
    # else: logger.debug(f"Strategy Signals ({strategy_name}): No active signals.") # Optional: log absence of signals

    return signals


# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """
    Executes the main trading logic for a single cycle based on the provided market data.

    Steps:
    1. Calculates all necessary indicators based on the selected strategy and filters.
    2. Checks for valid data (price, ATR).
    3. Determines the current position state.
    4. Optionally analyzes the order book.
    5. Logs the current market and position state.
    6. Generates trading signals based on the chosen strategy.
    7. If in a position, checks for strategy exit signals. If triggered, cancels existing
       stop orders and attempts to close the position.
    8. If flat (no position), checks for strategy entry signals.
    9. If an entry signal exists, evaluates confirmation filters (Volume, Order Book).
    10. If entry signal is confirmed, cancels any stray orders and attempts to place
        a new market order with calculated risk, native SL, and native TSL.

    Args:
        exchange: The configured CCXT exchange instance.
        symbol: The market symbol being traded.
        df: The pandas DataFrame containing the latest OHLCV and pre-calculated indicator data.
    """
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Weaving Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # --- Basic Data Validation ---
    # Required rows check should happen before calling trade_logic based on indicator needs
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Trade Logic Skipped: Invalid DataFrame received.{Style.RESET_ALL}")
        return

    action_taken_this_cycle: bool = False # Track if an entry/exit order was placed/attempted

    try:
        # === 1. Calculate Necessary Indicators (Ensure they are already on df) ===
        # Indicators should ideally be calculated *before* calling trade_logic or right at the start.
        # Re-calculating them here can be redundant if done in the main loop.
        # Assuming indicators are present from main loop's data prep phase.
        # We still need ATR and Volume analysis results.
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr") # Crucial for SL calculation and entry possibility

        # === 2. Validate Base Requirements - Ensure stable ground ===
        last_candle = df.iloc[-1]
        current_price = safe_decimal_conversion(last_candle.get('close')) # Get latest close price

        if pd.isna(current_price) or current_price <= 0:
            logger.warning(f"{Fore.YELLOW}Trade Logic Skipped: Last candle close price is invalid ({current_price}). Data might be corrupt.{Style.RESET_ALL}")
            return
        # Can we place a new order? Requires a valid ATR for SL calculation.
        can_place_new_order = current_atr is not None and current_atr > Decimal("0")
        if not can_place_new_order:
            # Log warning but allow potential exit logic to proceed
            logger.warning(f"{Fore.YELLOW}Invalid ATR ({current_atr}) calculated. Cannot calculate SL or place NEW entry orders this cycle. Exit logic may still function.{Style.RESET_ALL}")
            # Note: Existing position management (exits based on strategy signals) might still be possible.

        # === 3. Get Current Position & Analyze Order Book (conditionally) ===
        position = get_current_position(exchange, symbol) # Check current market presence
        position_side = position['side']
        position_qty = position['qty']
        position_entry = position['entry_price']

        # Fetch OB data if configured for every cycle, OR if needed later for entry confirmation
        ob_data = None
        if CONFIG.fetch_order_book_per_cycle:
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)

        # === 4. Log Current State - The Oracle Reports ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        is_vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None # Use fetched data if available
        spread = ob_data.get("spread") if ob_data else None

        # Log core state
        atr_str = f"{current_atr:.5f}" if current_atr else f"{Fore.RED}N/A{Style.RESET_ALL}"
        price_color = Fore.GREEN if current_price > df['close'].iloc[-2] else Fore.RED if current_price < df['close'].iloc[-2] else Fore.WHITE
        logger.info(f"State | Price: {price_color}{current_price:.4f}{Style.RESET_ALL}, ATR({CONFIG.atr_calculation_period}): {Fore.MAGENTA}{atr_str}{Style.RESET_ALL}")

        # Log filter states
        vol_ratio_str = f"{vol_ratio:.2f}" if vol_ratio else "N/A"
        vol_spike_str = f"{Fore.GREEN}YES{Style.RESET_ALL}" if is_vol_spike else f"{Fore.RED}NO{Style.RESET_ALL}"
        logger.info(f"State | Volume: Ratio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}, Spike={vol_spike_str} (Threshold={CONFIG.volume_spike_threshold}, RequiredForEntry={CONFIG.require_volume_spike_for_entry})")

        ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else "N/A"
        ob_spread_str = f"{spread:.4f}" if spread else "N/A"
        ob_ratio_color = Fore.WHITE # Default color
        if bid_ask_ratio is not None:
            if bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long: ob_ratio_color = Fore.GREEN
            elif bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short: ob_ratio_color = Fore.RED
            else: ob_ratio_color = Fore.YELLOW
        logger.info(f"State | OrderBook: Ratio(B/A)={ob_ratio_color}{ob_ratio_str}{Style.RESET_ALL} (L >= {CONFIG.order_book_ratio_threshold_long}, S <= {CONFIG.order_book_ratio_threshold_short}), Spread={ob_spread_str} (Fetched This Cycle={ob_data is not None})")

        # Log position state
        pos_color = Fore.GREEN if position_side == CONFIG.pos_long else (Fore.RED if position_side == CONFIG.pos_short else Fore.BLUE)
        logger.info(f"State | Position: Side={pos_color}{position_side}{Style.RESET_ALL}, Qty={position_qty:.8f}, Entry={position_entry:.4f}")

        # === 5. Generate Strategy Signals - Interpret the Omens ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name)

        # === 6. Execute Exit Actions - If the Omens Demand Retreat ===
        # Check if we are currently in a position that matches an exit signal
        should_exit_long = position_side == CONFIG.pos_long and strategy_signals['exit_long']
        should_exit_short = position_side == CONFIG.pos_short and strategy_signals['exit_short']

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals['exit_reason']
            exit_side_color = Back.YELLOW
            logger.warning(f"{exit_side_color}{Fore.BLACK}{Style.BRIGHT}*** STRATEGY EXIT SIGNAL: Attempting to Close {position_side} Position (Reason: {exit_reason}) ***{Style.RESET_ALL}")
            action_taken_this_cycle = True # Mark that we are attempting an action

            # --- Pre-Exit Cleanup: Cancel existing SL/TP orders ---
            # It's crucial to cancel stops BEFORE sending the market close order
            # to avoid potential race conditions, conflicts, or leaving orphaned stops.
            logger.info("Performing pre-exit order cleanup (cancelling SL/TP)...")
            # Use cancel_all_orders for efficiency if supported, targeting the specific symbol
            cancel_result = cancel_open_orders(exchange, symbol, f"Pre-Exit Cleanup ({exit_reason})")
            if cancel_result < 0: # Check if cancellation failed
                 logger.error(f"{Fore.RED}Failed to cancel open orders before exit attempt. Proceeding with caution.{Style.RESET_ALL}")
                 # Decide whether to proceed with close or abort? For now, proceed but log error.

            time.sleep(1.0) # Increased pause after cancel request before placing market close

            # --- Attempt to close the position ---
            close_result_order = close_position(exchange, symbol, position, reason=exit_reason)

            if close_result_order:
                # Position close order placed successfully (API call succeeded)
                logger.info(f"Position close order placed for {position_side}. Pausing briefly...")
                time.sleep(CONFIG.post_close_delay_seconds) # Pause after successful close attempt
            else:
                # Close attempt failed (error logged in close_position, or position was already closed)
                # Check if the reason was 'already closed' based on close_position's return logic
                # If close_position returned None because re-validation showed no position, it's not an error here.
                # If it returned None due to an error placing the order, log it.
                logger.error(f"{Fore.RED}Failed to place position close order for {position_side}, or position was already closed. Check logs from close_position.{Style.RESET_ALL}")
                # Still pause briefly even on failure/no-action to avoid rapid loops if issue persists
                time.sleep(CONFIG.sleep_seconds // 2)

            # --- Exit the current logic cycle ---
            # Regardless of close success/failure, we don't want to evaluate entry signals
            # immediately after an exit signal in the same cycle.
            logger.info("Exiting trade logic cycle after processing exit signal.")
            return # End the current cycle here

        # === 7. Check & Execute Entry Actions (Only if Currently Flat) ===
        if position_side != CONFIG.pos_none:
             # Log current holding status if not exiting
             logger.info(f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. Awaiting Exchange SL/TSL trigger or next Strategy Exit signal.")
             return # Do nothing more this cycle if already in a position

        # --- If flat, check if we can place a new order (requires valid ATR) ---
        if not can_place_new_order:
             logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot evaluate entry: Invalid ATR ({current_atr}) prevents SL calculation.{Style.RESET_ALL}")
             return # Cannot enter without valid ATR for initial SL

        # --- Evaluate Entry Conditions ---
        logger.debug("Position is Flat. Checking strategy entry signals...")
        base_enter_long = strategy_signals['enter_long']
        base_enter_short = strategy_signals['enter_short']
        potential_entry_signal = base_enter_long or base_enter_short

        if not potential_entry_signal:
            logger.info("Holding Cash. No entry signal generated by strategy.")
            return # No base signal, do nothing

        # --- Apply Confirmation Filters (Volume & Order Book) ---
        logger.debug("Potential entry signal found. Evaluating confirmation filters...")

        # Fetch OB data now if not fetched per cycle AND there's a potential entry signal
        if ob_data is None:
            logger.debug("Fetching Order Book data for entry confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None # Update ratio after fetch
            # Re-log OB state if fetched now
            ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else "N/A"
            ob_spread_str = ob_data.get("spread")
            ob_spread_str_fmt = f"{ob_spread_str:.4f}" if ob_spread_str else "N/A"
            ob_ratio_color = Fore.WHITE
            if bid_ask_ratio is not None:
                if bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long: ob_ratio_color = Fore.GREEN
                elif bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short: ob_ratio_color = Fore.RED
                else: ob_ratio_color = Fore.YELLOW
            logger.info(f"State | OrderBook (Post-Fetch): Ratio(B/A)={ob_ratio_color}{ob_ratio_str}{Style.RESET_ALL}, Spread={ob_spread_str_fmt}")


        # Evaluate Order Book Filter
        ob_confirm_long = False
        ob_confirm_short = False
        if bid_ask_ratio is not None:
            ob_confirm_long = bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long
            ob_confirm_short = bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short
        ob_filter_log = f"Filter | OB Confirm Long: {ob_confirm_long} (Ratio: {ob_ratio_str} >= {CONFIG.order_book_ratio_threshold_long}), " \
                        f"OB Confirm Short: {ob_confirm_short} (Ratio: {ob_ratio_str} <= {CONFIG.order_book_ratio_threshold_short})"
        logger.debug(ob_filter_log)

        # Evaluate Volume Filter
        vol_confirm = not CONFIG.require_volume_spike_for_entry or is_vol_spike
        vol_filter_log = f"Filter | Vol Confirm: {vol_confirm} (Spike: {is_vol_spike}, Required: {CONFIG.require_volume_spike_for_entry})"
        logger.debug(vol_filter_log)

        # --- Combine Strategy Signal with Confirmations ---
        # Entry requires base signal AND volume confirmation AND order book confirmation
        final_enter_long = base_enter_long and vol_confirm and ob_confirm_long
        final_enter_short = base_enter_short and vol_confirm and ob_confirm_short

        # Log final entry decision logic breakdown
        if base_enter_long:
             filter_met_long = vol_confirm and ob_confirm_long
             logger.debug(f"Final Entry Check (Long): Strategy={base_enter_long}, Filters OK (Vol={vol_confirm}, OB={ob_confirm_long}) = {filter_met_long} => {Fore.GREEN if final_enter_long else Fore.RED}Enter={final_enter_long}{Style.RESET_ALL}")
        if base_enter_short:
             filter_met_short = vol_confirm and ob_confirm_short
             logger.debug(f"Final Entry Check (Short): Strategy={base_enter_short}, Filters OK (Vol={vol_confirm}, OB={ob_confirm_short}) = {filter_met_short} => {Fore.GREEN if final_enter_short else Fore.RED}Enter={final_enter_short}{Style.RESET_ALL}")

        # --- Execute Entry ---
        entry_side: Optional[str] = None
        if final_enter_long:
            entry_side = CONFIG.side_buy
            entry_bg_color = Back.GREEN
            entry_fg_color = Fore.BLACK
            logger.success(f"{entry_bg_color}{entry_fg_color}{Style.BRIGHT}*** CONFIRMED LONG ENTRY SIGNAL ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
        elif final_enter_short:
            entry_side = CONFIG.side_sell
            entry_bg_color = Back.RED
            entry_fg_color = Fore.BLACK
            logger.success(f"{entry_bg_color}{entry_fg_color}{Style.BRIGHT}*** CONFIRMED SHORT ENTRY SIGNAL ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")

        if entry_side:
            action_taken_this_cycle = True
            # Belt-and-suspenders: Cancel any potential stray orders before entering a new position
            logger.info(f"Performing pre-entry order cleanup ({entry_side.upper()})...")
            cancel_result = cancel_open_orders(exchange, symbol, f"Pre-{entry_side.upper()} Entry Cleanup")
            if cancel_result < 0:
                 logger.error(f"{Fore.RED}Failed pre-entry order cleanup. Attempting entry anyway, but check for stray orders manually.{Style.RESET_ALL}")
            time.sleep(0.5) # Small pause after cancel attempt

            # Place the risked order with SL and TSL
            # Pass all necessary parameters, including the validated current_atr
            place_result_order = place_risked_market_order(
                exchange=exchange, symbol=symbol, side=entry_side,
                risk_percentage=CONFIG.risk_per_trade_percentage,
                current_atr=current_atr, # Pass the validated ATR
                sl_atr_multiplier=CONFIG.atr_stop_loss_multiplier,
                leverage=CONFIG.leverage,
                max_order_cap_usdt=CONFIG.max_order_usdt_amount,
                margin_check_buffer=CONFIG.required_margin_buffer,
                tsl_percent=CONFIG.trailing_stop_percentage,
                tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent
            )

            # Log outcome of placement process
            if place_result_order:
                 logger.info(f"{entry_side.upper()} entry order placement process initiated and returned order details (ID: {format_order_id(place_result_order.get('id'))}). Check logs for SL/TSL status.")
                 # Note: place_risked_market_order returns the entry order dict if entry was filled,
                 # even if SL/TSL placement failed. Failure before fill returns None.
            else:
                 logger.error(f"{Fore.RED}{entry_side.upper()} entry order placement process failed or was aborted. Check previous logs.{Style.RESET_ALL}")
                 # SMS alerts for failures are handled within place_risked_market_order

        else:
             # Log if a base signal existed but filters blocked it
             if potential_entry_signal and not action_taken_this_cycle:
                 reason_blocked = []
                 if base_enter_long and not vol_confirm: reason_blocked.append("Volume")
                 if base_enter_long and not ob_confirm_long: reason_blocked.append("OrderBook(L)")
                 if base_enter_short and not vol_confirm: reason_blocked.append("Volume")
                 if base_enter_short and not ob_confirm_short: reason_blocked.append("OrderBook(S)")
                 logger.info(f"Holding Cash. Strategy signal present but confirmation filters blocked entry (Blocked by: {', '.join(reason_blocked)}).")
             # No logging needed here if potential_entry_signal was false initially

    except Exception as e:
        # Catch-all for unexpected errors within the main logic loop for this cycle
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic cycle: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # Consider sending SMS alert for critical unknown errors in the core logic
        market_base = symbol.split('/')[0].split(':')[0]
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL ERROR in trade_logic cycle: {type(e).__name__}. Check logs!")
    finally:
        # Mark the end of the cycle clearly
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str]) -> None:
    """
    Attempts to perform a graceful shutdown by:
    1. Cancelling all open orders for the target symbol.
    2. Checking for any existing position for the target symbol.
    3. If a position exists, attempting to close it with a market order.
    4. Performing a final position check to confirm closure.

    Args:
        exchange: The active CCXT exchange instance, or None if not initialized.
        symbol: The target market symbol, or None if not set.
    """
    logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing arcane energies gracefully...{Style.RESET_ALL}")
    market_base = symbol.split('/')[0].split(':')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Shutdown initiated. Attempting cleanup...")

    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange portal or symbol not defined. Cannot perform automated cleanup.{Style.RESET_ALL}")
        # Attempt alert even without symbol/exchange details fully known
        send_sms_alert("[Pyrmethus] Shutdown: Exchange/Symbol missing, cleanup skipped.")
        return

    is_sandbox = getattr(exchange, 'sandbox', False)
    mode_str = "TESTNET" if is_sandbox else "LIVE"
    logger.warning(f"Shutdown operating in {mode_str} mode.")

    try:
        # 1. Cancel All Open Orders - Dispel residual intents (SL, TSL, pending entries)
        logger.warning("Shutdown Step 1: Cancelling all open orders for {symbol}...")
        cancel_result = cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        if cancel_result < 0:
             logger.error("Shutdown Step 1: Encountered errors cancelling orders. Proceeding with position check.")
        else:
             logger.info(f"Shutdown Step 1: Cancel order attempt finished.")

        # Wait briefly for cancellations to potentially process on the exchange side
        time.sleep(2.0)

        # 2. Check and Close Existing Position - Banish final market presence
        logger.warning("Shutdown Step 2: Checking for active position to close...")
        # Use get_current_position which is designed for V5
        position = get_current_position(exchange, symbol) # Get final position state

        if position['side'] != CONFIG.pos_none and abs(position['qty']) > CONFIG.position_qty_epsilon:
            pos_color = Fore.GREEN if position['side'] == CONFIG.pos_long else Fore.RED
            logger.warning(f"{Fore.YELLOW}Shutdown Step 2: Active {pos_color}{position['side']}{Style.RESET_ALL} position found (Qty: {position['qty']:.8f}). Attempting market close...{Style.RESET_ALL}")
            close_result_order = close_position(exchange, symbol, position, reason="Graceful Shutdown")

            if close_result_order:
                # Close order placed successfully (or seemed to be)
                wait_time = CONFIG.post_close_delay_seconds * 3 # Wait longer during shutdown
                logger.info(f"{Fore.CYAN}Shutdown Step 2: Close order placed/processed. Waiting {wait_time}s for exchange state update...{Style.RESET_ALL}")
                time.sleep(wait_time)

                # --- Final Confirmation Check ---
                logger.warning("Shutdown Step 3: Final position confirmation check...")
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.pos_none or abs(final_pos['qty']) <= CONFIG.position_qty_epsilon:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown Step 3: Position confirmed CLOSED/FLAT after shutdown attempt.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Position confirmed CLOSED on shutdown ({mode_str}).")
                else:
                    # This is a critical issue - manual intervention likely needed
                    final_pos_color = Fore.GREEN if final_pos['side'] == CONFIG.pos_long else Fore.RED
                    logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN FAILURE ({mode_str}): FAILED TO CONFIRM position closure after waiting! "
                                    f"Final state check shows: {final_pos_color}{final_pos['side']}{Style.RESET_ALL} Qty={final_pos['qty']:.8f}. "
                                    f"MANUAL INTERVENTION REQUIRED on Bybit!{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL SHUTDOWN ERROR ({mode_str}): Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!")
            else:
                # Close order placement failed initially, or position was already gone
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN FAILURE ({mode_str}): Failed to place position close order during shutdown (or position already closed). "
                                f"Final check recommended. MANUAL INTERVENTION MAY BE REQUIRED on Bybit!{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL SHUTDOWN ERROR ({mode_str}): Failed PLACE close order OR already closed. MANUAL CHECK!")
                # Perform one last check just in case
                logger.warning("Performing final position check after close placement failure...")
                final_pos_after_fail = get_current_position(exchange, symbol)
                if final_pos_after_fail['side'] == CONFIG.pos_none or abs(final_pos_after_fail['qty']) <= CONFIG.position_qty_epsilon:
                     logger.info(f"{Fore.GREEN}Shutdown: Final check after failed close attempt shows FLAT state. Likely closed before or by failed attempt.{Style.RESET_ALL}")
                else:
                     logger.critical(f"{Back.RED}{Fore.WHITE}Shutdown: Final check confirms position STILL OPEN after failed close attempt! State: {final_pos_after_fail['side']} Qty={final_pos_after_fail['qty']:.8f}. MANUAL ACTION NEEDED!{Style.RESET_ALL}")

        else:
            # No position found initially in step 2
            logger.info(f"{Fore.GREEN}Shutdown Step 2: No active position found for {symbol}. Clean exit state.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] No active position found on shutdown ({mode_str}).")

    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown Error: Unexpected error during cleanup sequence for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Error during shutdown cleanup ({mode_str}): {type(e).__name__}. Check logs & position manually.")

    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Scalping Spell Shutdown Sequence Complete ({mode_str}) ---{Style.RESET_ALL}")

# --- Main Execution - Igniting the Spell ---
def main() -> None:
    """
    Main function: Initializes components, sets up the market, validates configuration,
    and runs the main trading loop. Handles startup errors and triggers graceful shutdown.
    """
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.2.1 Initializing ({start_time_str}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Protective Wards Activated: Initial ATR-Stop + Exchange Trailing Stop (Bybit V5 Native) ---{Style.RESET_ALL}")
    # Config loading happens before main, errors handled there.

    exchange: Optional[ccxt.Exchange] = None
    symbol_unified: Optional[str] = None # The specific market symbol confirmed by CCXT (e.g., BTC/USDT:USDT)
    run_bot: bool = True # Controls the main trading loop
    cycle_count: int = 0 # Tracks the number of iterations
    market_base: str = "Bot" # Placeholder for SMS alerts before symbol is confirmed

    try:
        # === Initialize Exchange Portal ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Failed to open exchange portal. Spell cannot proceed. Exiting.")
            # SMS alert sent within initialize_exchange on failure
            return # Exit script if exchange init fails

        # === Setup Symbol, Validate Market, and Set Leverage - Focusing the Spell ===
        try:
            # Use symbol from config directly
            symbol_to_use = CONFIG.symbol
            logger.info(f"Attempting to focus spell on symbol: {symbol_to_use}")

            # Validate symbol and get unified representation from CCXT
            # This also loads market data if not already loaded
            market = exchange.market(symbol_to_use)
            symbol_unified = market['symbol'] # Use the precise symbol recognized by CCXT (e.g., BTC/USDT:USDT)
            market_base = symbol_unified.split('/')[0].split(':')[0] # Update for SMS alerts

            # Ensure it's a futures/contract market suitable for leverage and linearity
            market_type = market.get('type', 'unknown') # spot, future, option, swap
            is_contract = market.get('contract', False) # True for futures/swaps
            is_linear = market.get('linear', False) # True if USDT margined
            is_inverse = market.get('inverse', False) # True if coin margined

            logger.info(f"Market Details: Type={market_type}, Contract={is_contract}, Linear={is_linear}, Inverse={is_inverse}")

            if not is_contract:
                raise ValueError(f"Market '{symbol_unified}' (Type: {market_type}) is not a contract/futures/swap market suitable for this bot.")
            # This bot is designed for Linear (USDT-margined) contracts
            if not is_linear:
                 # Log a critical warning but allow proceeding if user insists via config.
                 # Could potentially add a config flag to allow non-linear if desired later.
                 logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL CONFIG WARNING: Market '{symbol_unified}' is not detected as LINEAR (USDT margined). "
                                f"This bot is designed for USDT margined contracts. Risk calculation and PnL may be incorrect. Proceed with extreme caution!{Style.RESET_ALL}")
                 send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL WARNING: Market {symbol_unified} is NOT LINEAR (USDT). Bot logic may fail!")
                 # Consider exiting here if strict linear-only operation is desired:
                 # raise ValueError("Bot configured for Linear contracts only.")

            logger.info(f"{Fore.GREEN}Spell successfully focused on Symbol: {symbol_unified}{Style.RESET_ALL}")

            # Set the desired leverage for the focused symbol
            if not set_leverage(exchange, symbol_unified, CONFIG.leverage):
                # set_leverage logs detailed errors, raise runtime error to stop bot
                raise RuntimeError(f"Leverage conjuring failed for {symbol_unified}. Cannot proceed safely.")

        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Symbol/Leverage setup failed: {e}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL: Symbol/Leverage setup FAILED ({type(e).__name__}). Exiting.")
            run_bot = False # Prevent loop from starting
            return # Exit script
        except Exception as e_setup:
            logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error during spell focus (Symbol/Leverage) setup: {e_setup}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL: Unexpected setup error ({type(e_setup).__name__}). Exiting.")
            run_bot = False # Prevent loop from starting
            return # Exit script

        # === Log Configuration Summary - Reciting the Parameters ===
        logger.info(f"{Fore.MAGENTA}--- Spell Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Symbol: {symbol_unified}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        logger.info(f"{Fore.CYAN}Strategy Path: {CONFIG.strategy_name}")
        # Log relevant strategy parameters for clarity
        if CONFIG.strategy_name == "DUAL_SUPERTREND": logger.info(f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": logger.info(f"  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER": logger.info(f"  Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": logger.info(f"  Params: FastMA(EMA)={CONFIG.ehlers_fast_period}, SlowMA(EMA)={CONFIG.ehlers_slow_period} {Fore.YELLOW}(EMA Placeholder){Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Risk Ward: {CONFIG.risk_per_trade_percentage:.3%} equity/trade, Max Pos Value: {CONFIG.max_order_usdt_amount:.4f} USDT")
        logger.info(f"{Fore.GREEN}Initial SL Ward: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})")
        logger.info(f"{Fore.GREEN}Trailing SL Shield: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"{Fore.YELLOW}Volume Filter: EntryRequiresSpike={CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, SpikeThr={CONFIG.volume_spike_threshold}x)")
        logger.info(f"{Fore.YELLOW}Order Book Filter: FetchPerCycle={CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        # Format margin buffer as percentage for clarity
        margin_buffer_percent = (CONFIG.required_margin_buffer - Decimal(1)) * Decimal(100)
        logger.info(f"{Fore.WHITE}Timing: Sleep={CONFIG.sleep_seconds}s | API: RecvWin={CONFIG.default_recv_window}ms, FillTimeout={CONFIG.order_fill_timeout_seconds}s")
        logger.info(f"{Fore.WHITE}Other: Margin Buffer={margin_buffer_percent:.1f}%, SMS Alerts={CONFIG.enable_sms_alerts}")
        logger.info(f"{Fore.CYAN}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")

        # Send final initialization success alert
        mode_str = "TESTNET" if getattr(exchange, 'sandbox', False) else "LIVE"
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Pyrmethus Bot v2.2.1 Initialized ({mode_str}). Symbol: {symbol_unified}, Strat: {CONFIG.strategy_name}. Starting main loop.")

        # === Main Trading Loop - The Continuous Weaving ===
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Weaving Start ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}")

            # Assert exchange and symbol are valid before proceeding in loop
            assert exchange is not None, "Exchange became None during runtime"
            assert symbol_unified is not None, "Symbol became None during runtime"

            try:
                # --- Calculate required data length dynamically ---
                # Ensure enough data for the longest lookback period of any indicator used across all strategies + buffer
                indicator_periods = [
                    CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
                    CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.momentum_length,
                    CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length,
                    CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
                    CONFIG.atr_calculation_period, CONFIG.volume_ma_period
                ]
                # StochRSI often needs a longer lookback than just its individual component lengths
                stochrsi_lookback = CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + CONFIG.stochrsi_d_period + 10 # Add buffer
                # Base requirement is max of individual periods + buffer for MA calculations etc.
                base_required = max(indicator_periods) + 15
                # Ensure minimum reasonable number of candles + API fetch buffer
                data_limit = max(base_required, stochrsi_lookback, 100) + CONFIG.api_fetch_limit_buffer
                logger.debug(f"Calculated data fetch limit for indicators: {data_limit}")

                # --- Gather fresh market data ---
                df = get_market_data(exchange, symbol_unified, CONFIG.interval, limit=data_limit)

                # --- Process data and execute logic if data is valid ---
                if df is not None and not df.empty and len(df) >= 2: # Need at least 2 rows for signal generation
                    # Pre-calculate all indicators needed by any strategy onto the DataFrame
                    # This avoids recalculating within trade_logic and ensures consistency
                    # Note: This assumes indicator functions modify df inplace or return the modified df
                    df = calculate_supertrend(df.copy(), CONFIG.st_atr_length, CONFIG.st_multiplier) # Use copy to avoid side effects if needed elsewhere
                    df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
                    df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
                    df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
                    df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period) # Uses EMA placeholder

                    # Pass the prepared DataFrame to the core logic function
                    trade_logic(exchange, symbol_unified, df) # Pass the df with all indicators calculated
                elif df is not None and len(df) < 2:
                     logger.warning(f"{Fore.YELLOW}Skipping trade logic: Not enough data rows ({len(df)}) returned after fetch/cleaning for signal generation.{Style.RESET_ALL}")
                else:
                    # Error/Warning logged within get_market_data if fetching/processing failed
                    logger.warning(f"{Fore.YELLOW}Skipping trade logic this cycle due to invalid/missing market data for {symbol_unified}.{Style.RESET_ALL}")

            # --- Robust Error Handling within the Loop ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. The exchange spirits demand patience. Sleeping longer...{Style.RESET_ALL}")
                sleep_time = CONFIG.sleep_seconds * 6 # Sleep much longer
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] WARNING: Rate limit exceeded! Pausing {sleep_time}s.")
                time.sleep(sleep_time)
            except ccxt.NetworkError as e:
                # Transient network issues, usually recoverable
                logger.warning(f"{Fore.YELLOW}Network disturbance in main loop: {e}. Retrying next cycle after delay.{Style.RESET_ALL}")
                # Optional: Add counter for repeated network errors to trigger longer pause or alert
                time.sleep(CONFIG.sleep_seconds * 2) # Slightly longer sleep for network issues
            except ccxt.ExchangeNotAvailable as e:
                # Exchange might be down for maintenance or unavailable (e.g., Cloudflare issues)
                sleep_time = 60 # Wait a significant time
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange Not Available: {e}. Portal temporarily closed. Sleeping {sleep_time}s...{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR: Exchange unavailable ({type(e).__name__})! Pausing {sleep_time}s.")
                time.sleep(sleep_time)
            except ccxt.AuthenticationError as e:
                # API keys might have been revoked, expired, IP changed, or permissions altered
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL: Authentication Error encountered during operation: {e}. API keys invalid or permissions changed. Spell broken! Stopping NOW.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL: Authentication Error during operation! Bot stopping NOW. Check API keys/IP.")
                run_bot = False # Stop the bot immediately
            except ccxt.ExchangeError as e: # Catch other specific exchange errors not handled by more specific handlers
                logger.error(f"{Fore.RED}Unhandled Exchange Error in main loop: {e}{Style.RESET_ALL}")
                logger.debug(f"ExchangeError Details: Type={type(e).__name__}, Args={e.args}")
                # Consider sending SMS for recurring or severe ExchangeErrors
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs.")
                time.sleep(CONFIG.sleep_seconds) # Standard sleep before retrying
            except Exception as e:
                # Catch-all for truly unexpected issues in the main loop or called functions
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL CHAOS in Main Loop (Cycle {cycle_count}): {e} !!! Stopping spell!{Style.RESET_ALL}")
                # logger.exception provides full traceback automatically in default logging setup if level >= ERROR
                logger.debug(traceback.format_exc()) # Explicitly log traceback for clarity
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Bot stopping NOW. Check logs!")
                run_bot = False # Stop the bot on unknown critical errors

            # --- Loop Delay - Controlling the Rhythm ---
            if run_bot:
                cycle_end_time = time.monotonic()
                elapsed = cycle_end_time - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} duration: {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s.")
                if sleep_duration > 0:
                    time.sleep(sleep_duration) # Wait for the configured interval before next cycle

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt detected. User requests withdrawal of arcane energies...{Style.RESET_ALL}")
        run_bot = False # Signal the loop to terminate gracefully after this iteration (if applicable)
    except Exception as startup_error:
        # Catch critical errors during initial setup (before the main loop starts)
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL ERROR during bot startup sequence (before main loop): {startup_error}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # Attempt to send SMS if config allows, otherwise just log
        if CONFIG and CONFIG.enable_sms_alerts and CONFIG.sms_recipient_number:
             # Use market_base which might still be "Bot" if error happened before symbol setup
             send_sms_alert(f"[{market_base}] CRITICAL STARTUP ERROR: {type(startup_error).__name__}. Bot failed to start.")
        run_bot = False # Ensure bot doesn't attempt to run the loop
    finally:
        # --- Graceful Shutdown Sequence ---
        # This will run whether the loop finished normally, was interrupted by Ctrl+C,
        # or hit a critical error that set run_bot=False.
        # Pass the potentially initialized exchange and symbol
        graceful_shutdown(exchange, symbol_unified)
        # Final SMS alert might fail if process termination is abrupt, but attempt it.
        # send_sms_alert(f"[{market_base}/{CONFIG.strategy_name}] Bot process terminated.") # Optional: Alert on final termination
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")

if __name__ == "__main__":
    # Ensure the spell is cast only when invoked directly as a script
    main()