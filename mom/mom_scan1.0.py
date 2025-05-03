#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Momentum Scanner Trading Bot for Bybit (V5 API)

This bot scans multiple symbols based on momentum indicators, Ehlers filters (optional),
and other technical analysis tools to identify potential trading opportunities on Bybit's
Unified Trading Account (primarily linear perpetual contracts). It utilizes both
WebSocket for real-time data and REST API for historical data and order management.

Features:
- Multi-symbol trading.
- Configurable strategy parameters per symbol (via JSON).
- Momentum indicators (EMA/SuperSmoother, ROC, RSI, ADX, Bollinger Bands).
- Ehlers filters (Super Smoother, Instantaneous Trendline - optional).
- Volume analysis (SMA, Percentile).
- Dynamic thresholds based on volatility (ATR).
- ATR-based risk management (position sizing, SL/TP).
- Optional multi-timeframe trend filtering.
- WebSocket integration for low-latency kline updates with robust reconnection.
- REST API fallback and order management.
- Robust error handling and API interaction with specific V5 error code checks.
- Dry run mode for simulation.
- Detailed logging with colors (console) and clean file output (UTF-8 encoded).
- Trade performance metrics tracking and logging (thread-safe).
- Graceful shutdown handling (incl. optional position closing).
- Comprehensive configuration validation on startup.
- Dynamic KLINE_LIMIT calculation based on indicator periods.
- Improved NaN handling and data validation throughout.
- Clearer action logging for trades.
- Enhanced WebSocket management and initial data population.
"""

import os
import sys
import time
import logging
import json
import argparse
import threading
import math
import datetime as dt
from queue import Queue, Empty
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List, Union

# --- Third-party Library Imports ---
import pandas as pd
import numpy as np

# Attempt to import required libraries and provide helpful error messages
try:
    import pandas_ta as ta  # Technical analysis library
except ImportError:
    print("Error: 'pandas_ta' library not found. Please install it: pip install pandas_ta")
    sys.exit(1)
try:
    # Use V5 unified_trading API
    from pybit.unified_trading import HTTP, WebSocket
except ImportError:
    print("Error: 'pybit' library not found or out of date. Please install/upgrade it: pip install -U pybit")
    sys.exit(1)
try:
    from colorama import init as colorama_init, Fore, Style, Back  # Colored console output
except ImportError:
    print("Error: 'colorama' library not found. Please install it: pip install colorama")
    sys.exit(1)
try:
    from dotenv import load_dotenv  # Load environment variables from .env file
except ImportError:
    print("Error: 'python-dotenv' library not found. Please install it: pip install python-dotenv")
    sys.exit(1)

# --- Load Environment Variables ---
# Ensure you have a .env file in the same directory or environment variables set:
# BYBIT_API_KEY=YOUR_API_KEY
# BYBIT_API_SECRET=YOUR_API_SECRET
load_dotenv()

# Initialize Colorama for neon-colored console output (auto-resets style)
colorama_init(autoreset=True)

# --- Constants ---
DEFAULT_CONFIG_FILE = "config.json"
LOG_FILE = "momentum_scanner_bot_enhanced.log"
METRICS_LOG_FILE = "trade_performance.log"
# Bybit V5 Kline limits
MAX_KLINE_LIMIT_PER_REQUEST = 1000 # Maximum klines fetchable in one V5 API call (verify in Bybit docs)
MIN_KLINE_RECORDS_FOR_CALC = 4 # Minimum records needed for more complex calcs like IT (>= 4)
# Used for floating point comparisons (adjust sensitivity as needed)
FLOAT_COMPARISON_TOLERANCE = 1e-9

# --- Logging Setup ---

# Base Logger Configuration (File Handler, no colors in file, UTF-8 encoded)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
# Use 'a' mode to append to log file, change to 'w' to overwrite on each run
log_file_mode = "a"
try:
    file_handler = logging.FileHandler(LOG_FILE, mode=log_file_mode, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
except Exception as log_setup_e:
    print(f"FATAL: Could not configure file logging for {LOG_FILE}: {log_setup_e}")
    sys.exit(1)

# Main application logger
logger = logging.getLogger("MomentumScannerTrader")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)  # Default level, can be overridden by --debug
logger.propagate = False  # Prevent duplicate logs if root logger is configured

# Metrics Logger Configuration (CSV-like format, UTF-8 encoded)
metrics_formatter = logging.Formatter("%(asctime)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
try:
    metrics_handler = logging.FileHandler(METRICS_LOG_FILE, mode="a", encoding='utf-8')
    metrics_handler.setFormatter(metrics_formatter)
except Exception as metrics_log_setup_e:
    print(f"FATAL: Could not configure metrics file logging for {METRICS_LOG_FILE}: {metrics_log_setup_e}")
    sys.exit(1)

# Metrics logger instance
metrics_logger = logging.getLogger("TradeMetrics")
metrics_logger.addHandler(metrics_handler)
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False

# --- Console Logging Function ---

# Neon-Colored Console Output with Symbol Support
SYMBOL_COLORS = {
    # Define specific symbols for distinct colors if desired
    "BTCUSDT": Fore.YELLOW + Style.BRIGHT,
    "ETHUSDT": Fore.BLUE + Style.BRIGHT,
    # Add more specific symbol colors here
    "default": Fore.WHITE, # Default color if symbol not specified or found
}
_symbol_color_cycle = [Fore.CYAN, Fore.MAGENTA, Fore.GREEN, Fore.LIGHTRED_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTMAGENTA_EX]
_symbol_color_map: Dict[str, str] = {}
_color_index = 0

def get_symbol_color(symbol: Optional[str]) -> str:
    """Gets a consistent color for a symbol, cycling through colors if needed."""
    global _color_index
    if not symbol:
        return SYMBOL_COLORS["default"]
    if symbol in SYMBOL_COLORS:
        return SYMBOL_COLORS[symbol]
    if symbol not in _symbol_color_map:
        # Cycle through defined colors for unassigned symbols
        color = _symbol_color_cycle[_color_index % len(_symbol_color_cycle)] + Style.BRIGHT
        _symbol_color_map[symbol] = color
        _color_index += 1
    return _symbol_color_map[symbol]

def log_console(level: int, message: Any, symbol: Optional[str] = None, exc_info: bool = False, *args, **kwargs):
    """
    Logs messages to the console with level-specific colors and optional
    symbol highlighting. Also forwards the message to the file logger.

    Args:
        level: Logging level (e.g., logging.INFO, logging.ERROR).
        message: The message object to log (will be converted to string).
        symbol: Optional symbol name to include with specific color coding.
        exc_info: If True, add exception information to the file log.
        *args: Additional arguments for string formatting (if message is a format string).
        **kwargs: Additional keyword arguments for string formatting.
    """
    # Define neon-like colors for different logging levels
    neon_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN + Style.BRIGHT,
        logging.WARNING: Fore.YELLOW + Style.BRIGHT,
        logging.ERROR: Fore.RED + Style.BRIGHT,
        logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
    }
    color = neon_colors.get(level, Fore.WHITE) # Default to white if level not found
    level_name = logging.getLevelName(level)
    symbol_color = get_symbol_color(symbol)
    symbol_prefix = f"{symbol_color}[{symbol}]{Style.RESET_ALL} " if symbol else ""
    # Construct the colored prefix for console output
    prefix = f"{color}{level_name}:{Style.RESET_ALL} {symbol_prefix}"

    # Ensure message is a string before attempting formatting
    if not isinstance(message, str):
        message_str = str(message)
    else:
        message_str = message

    # Format message safely if args or kwargs are provided
    formatted_message = message_str  # Default to the stringified message
    try:
        # Attempt format only if args or kwargs are present and message was originally a string
        if (args or kwargs) and isinstance(message, str):
            formatted_message = message.format(*args, **kwargs)
    except Exception as fmt_e:
        # Log formatting errors without crashing, include original message
        # Use basic print for this specific error as logging might be involved
        print(f"{Fore.RED}LOG FORMATTING ERROR:{Style.RESET_ALL} {fmt_e} | Original Msg: {message_str}")
        # Also log the error to the file logger for record-keeping
        logger.error(f"Log formatting error: {fmt_e} | Original Msg: {message_str}", exc_info=False)
        # Use the unformatted string message for console output in this error case
        formatted_message = message_str

    # Print formatted (or stringified) message to console
    print(prefix + formatted_message)

    # Log to file (without colors, with symbol if provided, using the potentially formatted message)
    file_message = f"[{symbol}] {formatted_message}" if symbol else formatted_message
    # Determine if exception info should be logged to the file
    # Log exc_info if explicitly requested OR if the level is ERROR or higher
    should_log_exc_info = exc_info or (level >= logging.ERROR)
    logger.log(level, file_message, exc_info=should_log_exc_info)

def log_metrics(message: str):
    """
    Logs trade performance metrics to the metrics file and prints to console with distinct color.

    Args:
        message: The metrics message string (typically comma-separated values).
    """
    try:
        metrics_logger.info(message)
        # Print metrics to console with a distinct color for visibility
        print(f"{Fore.MAGENTA+Style.BRIGHT}METRICS:{Style.RESET_ALL} {message}")
    except Exception as e:
        log_console(logging.ERROR, f"Failed to log metrics: {e}", exc_info=True)


# --- Load API Keys from Environment Variables ---
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

# Critical check: Ensure API keys are loaded before proceeding
if not API_KEY or not API_SECRET:
    # Use log_console for consistency, even at critical level before full logging setup
    log_console(logging.CRITICAL, "BYBIT_API_KEY or BYBIT_API_SECRET not found in environment variables or .env file. Exiting.")
    sys.exit(1)
else:
    log_console(logging.INFO, "API keys loaded successfully.")


# --- Configuration Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the JSON configuration file, performs basic validation, and calculates
    the required KLINE_LIMIT for each symbol based on indicator periods.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        The loaded configuration dictionary.

    Raises:
        SystemExit: If the config file is missing, invalid JSON, or lacks essential structure.
    """
    log_console(logging.INFO, f"Attempting to load configuration file: {config_path}")
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            config = json.load(f)
        log_console(logging.INFO, f"Successfully loaded configuration from {config_path}")

        # --- Basic Structure Validation ---
        if not isinstance(config, dict):
            raise ValueError("Configuration file root must be a JSON object (dictionary).")
        if "BYBIT_CONFIG" not in config or not isinstance(config["BYBIT_CONFIG"], dict):
            raise ValueError("Missing or invalid 'BYBIT_CONFIG' section (must be a dictionary).")
        if "SYMBOLS" not in config["BYBIT_CONFIG"] or not isinstance(config["BYBIT_CONFIG"]["SYMBOLS"], list):
             raise ValueError("'BYBIT_CONFIG.SYMBOLS' must be a list.")
        if not config["BYBIT_CONFIG"]["SYMBOLS"]:
             raise ValueError("'BYBIT_CONFIG.SYMBOLS' list cannot be empty.")
        if "BOT_CONFIG" not in config or not isinstance(config["BOT_CONFIG"], dict):
            raise ValueError("Missing or invalid 'BOT_CONFIG' section (must be a dictionary).")
        if "RISK_CONFIG" not in config or not isinstance(config["RISK_CONFIG"], dict):
             raise ValueError("Missing or invalid 'RISK_CONFIG' section (must be a dictionary).")

        # --- Calculate KLINE_LIMIT per Symbol ---
        bybit_cfg = config["BYBIT_CONFIG"]
        symbols_cfg = bybit_cfg["SYMBOLS"]
        bot_cfg = config["BOT_CONFIG"]
        # Default buffer, ensures enough data for indicator warmup and lookbacks
        kline_limit_buffer = bot_cfg.get("KLINE_LIMIT_BUFFER", 50)
        # Ensure buffer is non-negative integer
        if not isinstance(kline_limit_buffer, int) or kline_limit_buffer < 0:
             log_console(logging.WARNING, f"Invalid KLINE_LIMIT_BUFFER ({kline_limit_buffer}), using default 50.")
             kline_limit_buffer = 50

        for symbol_index, symbol_cfg in enumerate(symbols_cfg):
            if not isinstance(symbol_cfg, dict) or "SYMBOL" not in symbol_cfg or not isinstance(symbol_cfg["SYMBOL"], str):
                raise ValueError(f"Each item in 'SYMBOLS' (index {symbol_index}) must be a dictionary with a valid string 'SYMBOL' key.")

            symbol = symbol_cfg["SYMBOL"]
            strategy_cfg = symbol_cfg.get("STRATEGY_CONFIG", {})
            if not isinstance(strategy_cfg, dict):
                log_console(logging.WARNING, f"Missing or invalid 'STRATEGY_CONFIG' for {symbol}. Using defaults/basic checks.", symbol=symbol)
                strategy_cfg = {}  # Ensure it exists as dict for lookups

            # Collect all periods used by indicators for this symbol
            periods = [
                strategy_cfg.get("ULTRA_FAST_EMA_PERIOD", 5),
                strategy_cfg.get("FAST_EMA_PERIOD", 10),
                strategy_cfg.get("MID_EMA_PERIOD", 30),
                strategy_cfg.get("SLOW_EMA_PERIOD", 89),
                strategy_cfg.get("ROC_PERIOD", 5),
                strategy_cfg.get("ATR_PERIOD_RISK", 14),
                strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14),
                strategy_cfg.get("RSI_PERIOD", 10),
                strategy_cfg.get("VOLUME_SMA_PERIOD", 20),
                strategy_cfg.get("ADX_PERIOD", 14),
                strategy_cfg.get("BBANDS_PERIOD", 20),
                strategy_cfg.get("INSTANTANEOUS_TRENDLINE_PERIOD", 20),
                # Add periods from any other indicators used (ensure they are in STRATEGY_CONFIG)
            ]

            # Consider higher timeframe periods if MTF is enabled
            if strategy_cfg.get("ENABLE_MULTI_TIMEFRAME", False):
                # Assume HTF uses the same periods; adjust if HTF has different settings
                # Double the list to account for HTF potentially needing same max period
                # Note: This is a simplification; HTF might have different period requirements
                periods.extend(periods)

            # Filter out invalid or non-positive periods
            valid_periods = [p for p in periods if isinstance(p, (int, float)) and p > 0]
            if not valid_periods:
                log_console(logging.WARNING, f"No valid positive indicator periods found for {symbol}. Using default max period (200).", symbol=symbol)
                max_period = 200  # Default fallback if no periods found
            else:
                max_period = max(valid_periods)

            # Ensure kline limit is at least the minimum required + buffer
            # Add extra buffer for complex indicators like ADX which require more warmup
            adx_period_val = strategy_cfg.get("ADX_PERIOD", 14)
            adx_warmup_buffer = 0
            if isinstance(adx_period_val, int) and adx_period_val > 0:
                 adx_warmup_buffer = 2 * adx_period_val # ADX needs roughly 2*period for stable values
            else:
                 adx_warmup_buffer = 30 # Default ADX warmup if period invalid/missing

            # Calculate base limit needed
            calculated_limit = int(max_period + kline_limit_buffer + adx_warmup_buffer)
            # Ensure it's at least the minimum required for basic calcs + buffer
            kline_limit = max(MIN_KLINE_RECORDS_FOR_CALC + kline_limit_buffer, calculated_limit)

            # Cap the limit at a reasonable maximum to avoid excessive memory/API usage
            # MAX_KLINE_LIMIT_PER_REQUEST is the API fetch limit; internal limit can be higher for rolling calcs
            internal_max_limit = 2 * MAX_KLINE_LIMIT_PER_REQUEST # Allow storing more than one fetch worth, adjust if needed
            kline_limit = min(kline_limit, internal_max_limit)

            # Store calculated limit within the symbol's config (internal use)
            # Use setdefault to avoid overwriting if already exists (though unlikely here)
            symbol_cfg.setdefault("INTERNAL", {})["KLINE_LIMIT"] = kline_limit
            log_console(logging.DEBUG, f"Calculated KLINE_LIMIT for {symbol}: {kline_limit} (Max Period: {max_period:.0f}, Buffer: {kline_limit_buffer}, ADX Buf: {adx_warmup_buffer})", symbol=symbol)

        # --- Detailed Configuration Validation (Performed later in main block after load) ---
        # This function focuses on loading and basic structural checks + KLINE_LIMIT calc.

        return config

    except FileNotFoundError:
        log_console(logging.CRITICAL, f"Configuration file not found: {config_path}. Please ensure it exists.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_console(logging.CRITICAL, f"Error decoding JSON configuration file {config_path}: {e}")
        sys.exit(1)
    except ValueError as e: # Catch specific validation errors raised above
        log_console(logging.CRITICAL, f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        log_console(logging.CRITICAL, f"Unexpected error loading or performing initial validation of configuration: {e}", exc_info=True)
        sys.exit(1)


# --- Custom Indicator Functions (Ehlers) ---
# Note: These filters are sensitive to data quality and parameter tuning.
# Verify implementation against trusted sources if results seem incorrect.
def super_smoother(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates Ehlers Super Smoother filter for a given series.
    Requires minimum period of 2. Handles NaNs robustly.

    Args:
        series: pandas Series of price data.
        period: Lookback period for the filter (must be an integer >= 2).

    Returns:
        pandas Series with Super Smoother values, or Series of NaNs on error/invalid input.
    """
    if not isinstance(series, pd.Series) or series.empty:
        log_console(logging.DEBUG, "Super Smoother: Input series is invalid or empty.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)
    if not isinstance(period, int) or period < 2:
        log_console(logging.WARNING, f"Super Smoother: Period must be an integer >= 2, got {period}. Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length required for calculation start
    series_cleaned = series.dropna()
    min_len_needed = 2 # Needs current and previous price, and two previous SS values
    if len(series_cleaned) < min_len_needed:
        log_console(logging.DEBUG, f"Super Smoother: Series length ({len(series_cleaned)} non-NaN) is less than required minimum ({min_len_needed}). Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        # Ehlers formula constants
        a1 = math.exp(-math.pi * math.sqrt(2.0) / float(period))
        b1 = 2.0 * a1 * math.cos(math.pi * math.sqrt(2.0) / float(period))
        c1 = -(a1 * a1)
        coeff1 = (1.0 - b1 - c1) / 2.0

        # Ensure float type, ffill cautiously to handle potential leading NaNs
        series_float = series.astype(float).ffill()
        # Initialize output array with NaNs
        ss = np.full(len(series_float), np.nan, dtype=np.float64)

        # Initialize first two values robustly after ffill
        # Check if values exist and are not NaN before assigning
        if len(series_float) > 0 and not np.isnan(series_float.iloc[0]): ss[0] = series_float.iloc[0]
        if len(series_float) > 1 and not np.isnan(series_float.iloc[1]): ss[1] = series_float.iloc[1]

        # Iterative filter calculation loop
        for i in range(2, len(series_float)):
            # Rigorous NaN check for all required inputs at step 'i'
            # Check current price inputs (i, i-1) and previous SS outputs (i-1, i-2)
            if np.isnan(series_float.iloc[i]) or np.isnan(series_float.iloc[i - 1]) or \
               np.isnan(ss[i - 1]) or np.isnan(ss[i - 2]):
                # If any input is NaN, the result for this step remains NaN (as initialized).
                continue # Skip calculation for this step

            # Extract values for calculation clarity
            prev_ss = ss[i - 1]
            prev_prev_ss = ss[i - 2]
            current_price_avg = (series_float.iloc[i] + series_float.iloc[i - 1]) / 2.0
            # Apply the Super Smoother formula
            ss[i] = coeff1 * current_price_avg + b1 * prev_ss + c1 * prev_prev_ss

        # Return result as a pandas Series with the original index
        return pd.Series(ss, index=series.index, dtype=np.float64)
    except Exception as e:
        log_console(logging.ERROR, f"Super Smoother calculation error for period {period}: {e}", exc_info=True)
        # Return NaNs on unexpected error
        return pd.Series(np.nan, index=series.index, dtype=np.float64)


def instantaneous_trendline(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates Ehlers Instantaneous Trendline for a given series.
    Requires minimum period of 4 (based on formula dependencies).
    Uses a fixed alpha as commonly recommended by Ehlers. Handles NaNs robustly.

    Args:
        series: pandas Series of price data.
        period: Lookback period (dominant cycle period, must be an integer >= 4).

    Returns:
        pandas Series with Instantaneous Trendline values, or Series of NaNs on error/invalid input.
    """
    if not isinstance(series, pd.Series) or series.empty:
        log_console(logging.DEBUG, "Instantaneous Trendline: Input series is invalid or empty.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Minimum period requirement based on formula structure
    required_min_period = 4
    if not isinstance(period, int) or period < required_min_period:
        log_console(logging.WARNING, f"Instantaneous Trendline: Period must be an integer >= {required_min_period}, got {period}. Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length needed for the calculation to start
    series_cleaned = series.dropna()
    # The formula at index 'i' depends on prices i, i-1, i-2 and IT values i-1, i-2.
    # The loop starts at i=4, requiring valid data up to index 3 and valid IT up to index 3.
    # Initialization handles up to IT[3], which depends on prices up to index 3.
    min_len_needed = 4
    if len(series_cleaned) < min_len_needed:
        log_console(logging.DEBUG, f"Instantaneous Trendline: Series length ({len(series_cleaned)} non-NaN) is less than required minimum ({min_len_needed}). Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        # Fixed alpha parameter as commonly used in Ehlers' IT formula (adjust if testing variations)
        alpha = 0.07

        # Ensure float type, ffill cautiously
        series_float = series.astype(float).ffill()
        # Initialize output array with NaNs
        it = np.full(len(series_float), np.nan, dtype=np.float64)

        # Initialize first few values robustly based on common Ehlers practice after ffill
        # Check source values are not NaN before using them for initialization
        if len(series_float) > 0 and not np.isnan(series_float.iloc[0]): it[0] = series_float.iloc[0]
        if len(series_float) > 1 and not np.isnan(series_float.iloc[1]): it[1] = series_float.iloc[1]
        # Requires indices 0, 1, 2 to be valid
        if len(series_float) > 2 and not np.any(np.isnan(series_float.iloc[0:3])):
            it[2] = (series_float.iloc[2] + 2.0 * series_float.iloc[1] + series_float.iloc[0]) / 4.0
        # Requires indices 1, 2, 3 to be valid
        if len(series_float) > 3 and not np.any(np.isnan(series_float.iloc[1:4])):
             it[3] = (series_float.iloc[3] + 2.0 * series_float.iloc[2] + series_float.iloc[1]) / 4.0

        # Iterative calculation loop - check for NaNs rigorously
        # Starts from index 4, as it[i] depends on it[i-1] and it[i-2]
        for i in range(4, len(series_float)):
            # Check all required price inputs (i, i-1, i-2) and previous IT outputs (i-1, i-2)
            if np.isnan(series_float.iloc[i]) or np.isnan(series_float.iloc[i - 1]) or \
               np.isnan(series_float.iloc[i - 2]) or np.isnan(it[i - 1]) or np.isnan(it[i - 2]):
                # If any input is NaN, the result for this step remains NaN.
                continue # Skip calculation for this step

            # Extract values for calculation clarity
            price_i = series_float.iloc[i]
            price_im1 = series_float.iloc[i - 1]
            price_im2 = series_float.iloc[i - 2]
            it_im1 = it[i - 1]
            it_im2 = it[i - 2]

            # Ehlers Instantaneous Trendline formula (verify against trusted source if needed):
            it[i] = (alpha - alpha**2 / 4.0) * price_i + \
                    (alpha**2 / 2.0) * price_im1 - \
                    (alpha - 3.0 * alpha**2 / 4.0) * price_im2 + \
                    2.0 * (1.0 - alpha) * it_im1 - \
                    (1.0 - alpha)**2 * it_im2

        # Return result as a pandas Series with the original index
        return pd.Series(it, index=series.index, dtype=np.float64)
    except Exception as e:
        log_console(logging.ERROR, f"Instantaneous Trendline calculation error for period {period}: {e}", exc_info=True)
        # Return NaNs on unexpected error
        return pd.Series(np.nan, index=series.index, dtype=np.float64)


# --- Indicator Calculation ---
def calculate_indicators_momentum(df: pd.DataFrame, strategy_cfg: Dict[str, Any], config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Calculates momentum-based technical indicators, Ehlers filters (optional),
    and generates unified entry/exit signals based on the strategy configuration.
    Performs extensive data validation and NaN handling.

    Args:
        df: DataFrame with OHLCV data. Must have a DatetimeIndex and columns
            ['open', 'high', 'low', 'close', 'volume'], sorted oldest to newest.
        strategy_cfg: Strategy-specific configuration dictionary for the symbol.
        config: The full application configuration dictionary.

    Returns:
        DataFrame with calculated indicators and signals added, or None if
        calculation fails or input data is insufficient. Returns a *copy*.
    """
    # Input Validation
    internal_cfg = strategy_cfg.get("INTERNAL", {})
    # Ensure KLINE_LIMIT is at least MIN_KLINE_RECORDS_FOR_CALC for any calculation
    # Use the value calculated and stored in load_config
    kline_limit = internal_cfg.get("KLINE_LIMIT", MIN_KLINE_RECORDS_FOR_CALC + 50) # Fallback just in case
    required_cols = ["open", "high", "low", "close", "volume"]

    if df is None or df.empty:
        log_console(logging.DEBUG, "Indicator Calc: Input DataFrame is empty.")
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            log_console(logging.DEBUG, "Indicator Calc: DataFrame index is not DatetimeIndex, attempting conversion to UTC.")
            # Convert index to datetime, coercing errors, setting timezone to UTC
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            if not isinstance(df.index, pd.DatetimeIndex):  # Check conversion success
                raise ValueError("Index conversion to DatetimeIndex failed.")
            if df.index.isnull().any():
                log_console(logging.WARNING, "Indicator Calc: Found NaT values in index after conversion. Dropping affected rows.")
                # Drop rows with NaT index or NaN in required columns
                df.dropna(axis=0, how='any', subset=required_cols + [df.index.name], inplace=True)
                if df.empty:
                    log_console(logging.ERROR, "Indicator Calc: DataFrame empty after dropping rows with NaT index or NaN data.")
                    return None
        except Exception as idx_e:
            log_console(logging.ERROR, f"Indicator Calc: Failed to convert DataFrame index to DatetimeIndex: {idx_e}")
            return None
    elif not df.index.tz:
         log_console(logging.DEBUG, "Indicator Calc: DataFrame index lacks timezone. Assuming and setting UTC.")
         df.index = df.index.tz_localize('UTC', ambiguous='infer')


    # Use a copy to avoid modifying the original DataFrame passed to the function
    df_out = df.copy()

    # Attempt numeric conversion for required columns and check length *after* potential NaN drop
    try:
        for col in required_cols:
            if col not in df_out.columns:
                log_console(logging.ERROR, f"Indicator Calc: Missing required column: '{col}'. Cannot proceed.")
                return None  # Critical column missing
            if not pd.api.types.is_numeric_dtype(df_out[col]):
                log_console(logging.DEBUG, f"Indicator Calc: Converting non-numeric column '{col}' to numeric.")
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

        # Drop rows where essential conversions failed (resulted in NaNs in required cols)
        initial_len = len(df_out)
        df_out.dropna(subset=required_cols, inplace=True)
        if len(df_out) < initial_len:
            log_console(logging.WARNING, f"Indicator Calc: Dropped {initial_len - len(df_out)} rows due to NaN values in required columns after numeric conversion.")

        # Check length requirements AFTER cleaning numeric data
        if len(df_out) < kline_limit:
            log_console(logging.DEBUG, f"Indicator Calc: Insufficient valid data points after cleaning ({len(df_out)} < {kline_limit}). Need more historical data.")
            return None
        if df_out.empty:
            log_console(logging.ERROR, "Indicator Calc: DataFrame empty after NaN drop from numeric conversion.")
            return None

    except Exception as e:
        log_console(logging.ERROR, f"Indicator Calc: Failed during data validation/numeric conversion: {e}", exc_info=True)
        return None

    # --- Calculate Indicators ---
    try:
        price_source_key = strategy_cfg.get("PRICE_SOURCE", "close")
        # Calculate common price sources if specified
        if price_source_key == 'hlc3':
            df_out['hlc3'] = (df_out['high'] + df_out['low'] + df_out['close']) / 3.0
            price_source_col = 'hlc3'
        elif price_source_key == 'ohlc4':
             df_out['ohlc4'] = (df_out['open'] + df_out['high'] + df_out['low'] + df_out['close']) / 4.0
             price_source_col = 'ohlc4'
        elif price_source_key in df_out.columns:
            # Use directly if column exists (e.g., 'open', 'high', 'low', 'close')
            price_source_col = price_source_key
        else:
            log_console(logging.ERROR, f"Indicator Calc: Specified price source '{price_source_key}' not found or calculable. Falling back to 'close'.")
            if 'close' not in df_out.columns:
                log_console(logging.ERROR, "Indicator Calc: Fallback 'close' column missing. Cannot proceed.")
                return None # Cannot proceed without a valid price source
            price_source_col = 'close'

        # Check if the selected price source column has valid data
        if price_source_col not in df_out or df_out[price_source_col].isnull().all():
            log_console(logging.ERROR, f"Indicator Calc: Price source column '{price_source_col}' is missing or contains only NaNs.")
            return None

        use_ehlers = strategy_cfg.get("USE_EHLERS_SMOOTHER", False)

        # --- Moving Averages / Smoothers ---
        smoother_params = [
            ("ULTRA_FAST_EMA_PERIOD", "ema_ultra"),
            ("FAST_EMA_PERIOD", "ema_fast"),
            ("MID_EMA_PERIOD", "ema_mid"),
            ("SLOW_EMA_PERIOD", "ema_slow"),
        ]
        smoother_prefix = "ss" if use_ehlers else "ema"
        calculated_ma_cols = [] # Keep track of successfully calculated MA columns

        for period_key, base_col_name in smoother_params:
            length = strategy_cfg.get(period_key)
            col_name = base_col_name.replace("ema", smoother_prefix)
            # Initialize column with NaN first
            df_out[col_name] = np.nan
            if not isinstance(length, int) or length <= 0:
                log_console(logging.ERROR, f"Indicator Calc: Invalid or missing {period_key} in strategy config. Value: {length}. Skipping {col_name}.")
                continue # Skip calculation for this MA

            try:
                if use_ehlers:
                    indicator_series = super_smoother(df_out[price_source_col], length)
                else:
                    indicator_series = ta.ema(df_out[price_source_col], length=length)

                # Check result validity before assigning
                if indicator_series is None or indicator_series.empty or indicator_series.isnull().all():
                    log_console(logging.WARNING, f"Indicator Calc: {col_name} (length {length}) calculation resulted in empty or all NaNs.")
                    # Column already initialized to NaN
                else:
                    df_out[col_name] = indicator_series
                    calculated_ma_cols.append(col_name) # Add to list of successful calcs
            except Exception as calc_e:
                log_console(logging.ERROR, f"Indicator Calc: Error calculating {col_name} (length {length}): {calc_e}", exc_info=False)
                # Column remains NaN

        # --- Instantaneous Trendline ---
        it_period = strategy_cfg.get("INSTANTANEOUS_TRENDLINE_PERIOD", 20)
        df_out["trendline"] = np.nan # Initialize column
        if not isinstance(it_period, int) or it_period < 4: # Check minimum period requirement
            log_console(logging.ERROR, f"Indicator Calc: Invalid INSTANTANEOUS_TRENDLINE_PERIOD ({it_period}). Must be integer >= 4. Skipping calculation.")
        else:
            try:
                it_series = instantaneous_trendline(df_out[price_source_col], it_period)
                if it_series is None or it_series.empty or it_series.isnull().all():
                    log_console(logging.WARNING, "Indicator Calc: Instantaneous Trendline calculation resulted in empty or all NaNs.")
                    # Column already initialized to NaN
                else:
                    df_out["trendline"] = it_series
            except Exception as calc_e:
                log_console(logging.ERROR, f"Indicator Calc: Error calculating Instantaneous Trendline: {calc_e}", exc_info=False)
                # Column already initialized to NaN

        # --- Other Indicators (wrapped calculations) ---
        # Initialize all expected indicator columns to NaN first
        other_indicator_cols = ["roc", "roc_smooth", "atr_risk", "norm_atr", "rsi", "volume_sma", "volume_percentile_75", "adx", "bb_width"]
        for col in other_indicator_cols: df_out[col] = np.nan

        try:
            roc_period = strategy_cfg.get("ROC_PERIOD", 5)
            if roc_period > 0:
                roc_raw = ta.roc(df_out[price_source_col], length=roc_period)
                if roc_raw is not None: df_out["roc"] = roc_raw
                # Fill NaNs before smoothing ROC, use 0.0 as neutral momentum
                roc_filled = df_out["roc"].fillna(0.0)
                roc_smooth_series = ta.sma(roc_filled, length=3)
                if roc_smooth_series is not None: df_out["roc_smooth"] = roc_smooth_series
            else: raise ValueError("ROC_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating ROC: {e}", exc_info=False)
            # Columns remain NaN

        try:
            atr_period_risk = strategy_cfg.get("ATR_PERIOD_RISK", 14)
            if atr_period_risk > 0:
                atr_series = ta.atr(df_out["high"], df_out["low"], df_out["close"], length=atr_period_risk)
                if atr_series is not None:
                    # Use .ffill() as per pandas_ta recommendation, then fill remaining start NaNs with 0
                    df_out["atr_risk"] = atr_series.ffill().fillna(0.0)
                    # Calculate Normalized ATR safely
                    # Avoid division by zero/NaN: replace 0 close with NaN, ffill, then fill remaining start NaNs with 1.0
                    close_safe = df_out["close"].replace(0, np.nan).ffill().fillna(1.0)
                    norm_atr_series = (df_out["atr_risk"] / close_safe) * 100
                    df_out["norm_atr"] = norm_atr_series.ffill().fillna(0.0)
            else: raise ValueError("ATR_PERIOD_RISK must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating ATR: {e}", exc_info=False)
            # Columns remain NaN

        try:
            rsi_period = strategy_cfg.get("RSI_PERIOD", 10)
            if rsi_period > 0:
                rsi_series = ta.rsi(df_out[price_source_col], length=rsi_period)
                if rsi_series is not None: df_out["rsi"] = rsi_series # Handle NaNs later in bulk fill
            else: raise ValueError("RSI_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating RSI: {e}", exc_info=False)
            # Column remains NaN

        try:
            vol_period = strategy_cfg.get("VOLUME_SMA_PERIOD", 20)
            if vol_period > 0:
                vol_sma_series = ta.sma(df_out["volume"], length=vol_period)
                if vol_sma_series is not None: df_out["volume_sma"] = vol_sma_series # Handle NaNs later
                min_periods_vol = max(1, vol_period // 2) # Ensure min_periods is at least 1 for rolling quantile
                # Ensure min_periods for rolling quantile, handle NaNs later
                vol_perc_series = df_out["volume"].rolling(window=vol_period, min_periods=min_periods_vol).quantile(0.75)
                if vol_perc_series is not None: df_out["volume_percentile_75"] = vol_perc_series
            else: raise ValueError("VOLUME_SMA_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating Volume Indicators: {e}", exc_info=False)
            # Columns remain NaN

        try:
            adx_period = strategy_cfg.get("ADX_PERIOD", 14)
            if adx_period > 0:
                adx_df = ta.adx(df_out["high"], df_out["low"], df_out["close"], length=adx_period)
                adx_col_name = f"ADX_{adx_period}"
                if adx_df is not None and not adx_df.empty and adx_col_name in adx_df.columns:
                    df_out["adx"] = adx_df[adx_col_name] # Handle NaNs later
                else:
                    log_console(logging.WARNING, f"Indicator Calc: ADX ({adx_col_name}) calculation failed or returned invalid DataFrame.")
                    # Column remains NaN
            else: raise ValueError("ADX_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating ADX: {e}", exc_info=False)
            # Column remains NaN

        try:
            bb_period = strategy_cfg.get("BBANDS_PERIOD", 20)
            bb_std = strategy_cfg.get("BBANDS_STDDEV", 2.0)
            if bb_period > 0 and bb_std > 0:
                bbands = ta.bbands(df_out["close"], length=bb_period, std=bb_std)
                # Define expected column names from pandas_ta bbands
                bbu_col = f"BBU_{bb_period}_{bb_std}"
                bbl_col = f"BBL_{bb_period}_{bb_std}"
                bbm_col = f"BBM_{bb_period}_{bb_std}"
                if bbands is not None and not bbands.empty and all(c in bbands.columns for c in [bbu_col, bbl_col, bbm_col]):
                    # Calculate BB width safely, handle NaNs and potential division by zero later during fill
                    bbm_safe = bbands[bbm_col].replace(0, np.nan) # Replace 0 with NaN before division
                    df_out["bb_width"] = (bbands[bbu_col] - bbands[bbl_col]) / bbm_safe * 100
                else:
                    log_console(logging.WARNING, "Indicator Calc: Bollinger Bands calculation failed or missing expected columns.")
                    # Column remains NaN
            else: raise ValueError("BBANDS_PERIOD and BBANDS_STDDEV must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating Bollinger Bands: {e}", exc_info=False)
            # Column remains NaN

        # --- Fill NaNs in Key Numeric Columns BEFORE calculating signals ---
        # Forward fill first to propagate last valid values, then fill remaining (start of series) with appropriate defaults.
        # Ensure MA/Smoother columns exist before trying to fill them (use calculated_ma_cols list)
        ma_cols_to_fill = {ma_col: 0.0 for ma_col in calculated_ma_cols} # Default fill for MAs (price=0), adjust if needed

        # Define fill values for other indicators
        numeric_cols_fill_values = {
            "atr_risk": 0.0, "norm_atr": 0.0, "roc": 0.0, "roc_smooth": 0.0,
            "rsi": 50.0, # Neutral RSI
            "adx": 20.0, # Often considered baseline for trend strength
            "bb_width": 0.0, # Default width if calculation failed or at start
            "volume_sma": 0.0, "volume_percentile_75": 0.0,
            # Fill trendline with price source if available and calculated, else 0
            "trendline": df_out[price_source_col] if "trendline" in df_out and not df_out["trendline"].isnull().all() else 0.0,
            **ma_cols_to_fill # Add the dynamically determined MA columns and their fill value
        }

        for col, fill_value in numeric_cols_fill_values.items():
            if col in df_out.columns:
                # Check if fill_value is a Series (like for trendline fallback)
                if isinstance(fill_value, pd.Series):
                    # Ensure indices align if needed, though ffill/fillna usually handle this
                    df_out[col] = df_out[col].ffill().fillna(fill_value)
                else:
                    df_out[col] = df_out[col].ffill().fillna(fill_value)
            # else: Column might be missing due to calculation error, which was already logged. Skip filling.

        # Calculate high_volume signal after filling relevant volume columns
        # Initialize column first
        df_out["high_volume"] = False
        if "volume" in df_out and "volume_percentile_75" in df_out and "volume_sma" in df_out:
            # Ensure volume itself is filled with 0 for comparison where needed
            df_out["high_volume"] = (df_out["volume"].fillna(0.0) > df_out["volume_percentile_75"]) & \
                                   (df_out["volume"].fillna(0.0) > df_out["volume_sma"])
            # Ensure the result is boolean and any remaining NaNs (shouldn't happen) become False
            df_out["high_volume"] = df_out["high_volume"].fillna(False).astype(bool)
        else:
            log_console(logging.WARNING, "Indicator Calc: Volume columns missing or failed calculation, cannot calculate 'high_volume'. Defaulting to False.")
            # Column already initialized to False

        # --- Dynamic Thresholds based on Volatility ---
        atr_period_volatility = strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14)
        volatility_factor = pd.Series(1.0, index=df_out.index) # Default neutral factor

        if "norm_atr" in df_out and atr_period_volatility > 0:
            min_periods_vol_atr = max(1, atr_period_volatility // 2)
            # Calculate rolling mean on norm_atr (already filled), ffill initial NaNs, default remaining to 1.0
            rolling_volatility = df_out["norm_atr"].rolling(window=atr_period_volatility, min_periods=min_periods_vol_atr).mean().ffill().fillna(1.0)
            # Clip volatility factor to prevent extreme adjustments
            volatility_factor = np.clip(rolling_volatility,
                                        strategy_cfg.get("VOLATILITY_FACTOR_MIN", 0.5),
                                        strategy_cfg.get("VOLATILITY_FACTOR_MAX", 2.0))
            # Ensure index alignment if needed (should be fine if calculated on df_out["norm_atr"])
            volatility_factor = pd.Series(volatility_factor, index=df_out.index)
        else:
            log_console(logging.DEBUG,"Indicator Calc: Cannot calculate volatility factor (norm_atr missing or period invalid). Using default factor 1.0.")
            # volatility_factor is already initialized to 1.0 Series

        # Calculate dynamic RSI thresholds
        rsi_base_low = strategy_cfg.get("RSI_LOW_THRESHOLD", 40)
        rsi_base_high = strategy_cfg.get("RSI_HIGH_THRESHOLD", 75)
        rsi_vol_adjust = strategy_cfg.get("RSI_VOLATILITY_ADJUST", 5)
        # Use pd.Series constructor to ensure index alignment and apply clipping
        rsi_low = pd.Series(np.clip(rsi_base_low - (rsi_vol_adjust * volatility_factor), 10, 50), index=df_out.index)
        rsi_high = pd.Series(np.clip(rsi_base_high + (rsi_vol_adjust * volatility_factor), 60, 90), index=df_out.index)

        # Calculate dynamic ROC threshold
        roc_base_threshold = strategy_cfg.get("MOM_THRESHOLD_PERCENT", 0.1)
        # Use pd.Series constructor, take absolute value, apply factor, handle potential NaNs from factor
        roc_threshold = pd.Series(abs(roc_base_threshold) * volatility_factor, index=df_out.index).fillna(abs(roc_base_threshold))

        # --- Trend Definition ---
        # Use the MA column names that were successfully calculated
        fast_ma_col = f"{smoother_prefix}_fast"
        mid_ma_col = f"{smoother_prefix}_mid"
        slow_ma_col = f"{smoother_prefix}_slow"
        trendline_col = "trendline"
        # Initialize trend columns
        df_out["trend_up"] = False
        df_out["trend_down"] = False
        df_out["trend_neutral"] = True

        # Check if required columns for trend definition exist and were calculated
        trend_req_cols = [fast_ma_col, mid_ma_col, slow_ma_col, trendline_col]
        if not all(col in df_out.columns and col in calculated_ma_cols + [trendline_col] for col in trend_req_cols):
            # Check if trendline calculation failed or was skipped
            trendline_ok = "trendline" in df_out.columns and not df_out["trendline"].isnull().all()
            ma_cols_ok = all(col in calculated_ma_cols for col in [fast_ma_col, mid_ma_col, slow_ma_col])
            log_console(logging.WARNING, f"Indicator Calc: Cannot define trend. MA cols OK: {ma_cols_ok}, Trendline OK: {trendline_ok}. Assuming neutral trend.")
        else:
            # NaNs should already be filled in MA/trendline columns
            # Calculate shifted trendline, fill initial shift NaN using backfill then forward fill
            trendline_shift = df_out[trendline_col].shift(1).fillna(method='bfill').fillna(method='ffill')

            # Trend Up Condition: Check MA alignment and trendline slope using tolerance
            trend_up_cond = (
                (df_out[fast_ma_col] > df_out[mid_ma_col]) &
                (df_out[mid_ma_col] > df_out[slow_ma_col]) &
                # Use tolerance for comparing trendline to its shifted value
                (df_out[trendline_col] > trendline_shift * (1 + FLOAT_COMPARISON_TOLERANCE))
            )
            df_out["trend_up"] = trend_up_cond.fillna(False).astype(bool)

            # Trend Down Condition
            trend_down_cond = (
                (df_out[fast_ma_col] < df_out[mid_ma_col]) &
                (df_out[mid_ma_col] < df_out[slow_ma_col]) &
                (df_out[trendline_col] < trendline_shift * (1 - FLOAT_COMPARISON_TOLERANCE))
            )
            df_out["trend_down"] = trend_down_cond.fillna(False).astype(bool)

            # Trend Neutral if neither up nor down
            df_out["trend_neutral"] = ~(df_out["trend_up"] | df_out["trend_down"])

        # --- Entry Signals ---
        adx_threshold = strategy_cfg.get("ADX_THRESHOLD", 20)
        ultra_fast_ma_col = f"{smoother_prefix}_ultra"
        roc_smooth_col = "roc_smooth"
        rsi_col = "rsi"
        high_volume_col = "high_volume"
        adx_col = "adx"
        # Initialize signal columns
        df_out["long_signal"] = False
        df_out["short_signal"] = False
        df_out["long_signal_strength"] = 0.0
        df_out["short_signal_strength"] = 0.0

        # Check if required columns for entry signals exist and were calculated
        entry_req_cols = [
            fast_ma_col, mid_ma_col, ultra_fast_ma_col, roc_smooth_col,
            rsi_col, adx_col, trendline_col, high_volume_col
        ]
        # Check against calculated MA columns + others that should exist
        required_available = all(col in df_out.columns for col in entry_req_cols if col not in calculated_ma_cols) and \
                             all(ma_col in calculated_ma_cols for ma_col in [fast_ma_col, mid_ma_col, ultra_fast_ma_col] if ma_col in entry_req_cols)

        if not required_available:
            missing_cols = [col for col in entry_req_cols if col not in df_out.columns or (col in [fast_ma_col, mid_ma_col, ultra_fast_ma_col] and col not in calculated_ma_cols)]
            log_console(logging.WARNING, f"Indicator Calc: Cannot generate entry signals, required columns missing or failed calculation: {missing_cols}")
        else:
            # Ensure trendline_shift is available (recalculate or use previous one if available)
            if 'trendline_shift' not in locals(): # Check if calculated in trend section
                 trendline_shift = df_out[trendline_col].shift(1).fillna(method='bfill').fillna(method='ffill')

            # --- Long Entry Conditions ---
            # NaNs already handled in source columns by the fill logic above
            long_cond_trend = df_out[fast_ma_col] > df_out[mid_ma_col]
            long_cond_trigger = df_out[ultra_fast_ma_col] > df_out[fast_ma_col]
            # Compare smoothed ROC against the dynamic threshold Series
            long_cond_mom = df_out[roc_smooth_col] > roc_threshold
            # Use dynamic RSI thresholds (Series comparison)
            long_cond_rsi = (df_out[rsi_col] > rsi_low) & (df_out[rsi_col] < rsi_high)
            long_cond_vol = df_out[high_volume_col] # Already boolean
            long_cond_adx = df_out[adx_col] > adx_threshold
            long_cond_itrend = df_out[trendline_col] > trendline_shift * (1 + FLOAT_COMPARISON_TOLERANCE)

            # Combine conditions for final long signal
            df_out["long_signal"] = (
                long_cond_trend & long_cond_trigger & long_cond_mom &
                long_cond_rsi & long_cond_vol & long_cond_adx & long_cond_itrend
            ).fillna(False).astype(bool)

            # --- Short Entry Conditions ---
            short_cond_trend = df_out[fast_ma_col] < df_out[mid_ma_col]
            short_cond_trigger = df_out[ultra_fast_ma_col] < df_out[fast_ma_col]
            # Compare smoothed ROC against negative dynamic threshold
            short_cond_mom = df_out[roc_smooth_col] < -roc_threshold
            # Use dynamic RSI thresholds
            short_cond_rsi = (df_out[rsi_col] > rsi_low) & (df_out[rsi_col] < rsi_high)
            short_cond_vol = df_out[high_volume_col] # Already boolean
            short_cond_adx = df_out[adx_col] > adx_threshold
            short_cond_itrend = df_out[trendline_col] < trendline_shift * (1 - FLOAT_COMPARISON_TOLERANCE)

            # Combine conditions for final short signal
            df_out["short_signal"] = (
                short_cond_trend & short_cond_trigger & short_cond_mom &
                short_cond_rsi & short_cond_vol & short_cond_adx & short_cond_itrend
            ).fillna(False).astype(bool)

            # --- Signal Strength Calculation ---
            # Count how many of the defined conditions are True for each signal type
            long_conditions_list = [long_cond_trend, long_cond_trigger, long_cond_mom, long_cond_rsi, long_cond_vol, long_cond_adx, long_cond_itrend]
            short_conditions_list = [short_cond_trend, short_cond_trigger, short_cond_mom, short_cond_rsi, short_cond_vol, short_cond_adx, short_cond_itrend]
            num_conditions = len(long_conditions_list) # Should be same for both

            if num_conditions > 0:
                # Sum boolean conditions (True=1, False=0) and normalize by total number of conditions
                df_out["long_signal_strength"] = sum(cond.astype(int) for cond in long_conditions_list) / num_conditions
                df_out["short_signal_strength"] = sum(cond.astype(int) for cond in short_conditions_list) / num_conditions
                # Fill any potential NaNs from the calculation (unlikely but possible) with 0.0
                df_out["long_signal_strength"] = df_out["long_signal_strength"].fillna(0.0)
                df_out["short_signal_strength"] = df_out["short_signal_strength"].fillna(0.0)
            # else: Columns initialized to 0.0


        # --- Exit Signals ---
        # Initialize columns
        df_out["exit_long_signal"] = False
        df_out["exit_short_signal"] = False
        # Check required columns for exit signals
        exit_req_cols = [fast_ma_col, mid_ma_col, roc_smooth_col, trendline_col]
        exit_required_available = all(col in df_out.columns for col in exit_req_cols if col not in calculated_ma_cols) and \
                                  all(ma_col in calculated_ma_cols for ma_col in [fast_ma_col, mid_ma_col] if ma_col in exit_req_cols)

        if not exit_required_available:
            missing_cols = [col for col in exit_req_cols if col not in df_out.columns or (col in [fast_ma_col, mid_ma_col] and col not in calculated_ma_cols)]
            log_console(logging.WARNING, f"Indicator Calc: Cannot generate exit signals, required columns missing or failed calculation: {missing_cols}")
        else:
            # Ensure trendline_shift is available
            if 'trendline_shift' not in locals():
                 trendline_shift = df_out[trendline_col].shift(1).fillna(method='bfill').fillna(method='ffill')

            # --- Exit Long Conditions ---
            # Exit if: Fast MA crosses below Mid MA OR Momentum turns negative OR Trendline turns down
            exit_long_cond1 = df_out[fast_ma_col] < df_out[mid_ma_col]
            exit_long_cond2 = df_out[roc_smooth_col] < 0 # Check against zero momentum
            exit_long_cond3 = df_out[trendline_col] < trendline_shift * (1 - FLOAT_COMPARISON_TOLERANCE)
            df_out["exit_long_signal"] = (exit_long_cond1 | exit_long_cond2 | exit_long_cond3).fillna(False).astype(bool)

            # --- Exit Short Conditions ---
            # Exit if: Fast MA crosses above Mid MA OR Momentum turns positive OR Trendline turns up
            exit_short_cond1 = df_out[fast_ma_col] > df_out[mid_ma_col]
            exit_short_cond2 = df_out[roc_smooth_col] > 0 # Check against zero momentum
            exit_short_cond3 = df_out[trendline_col] > trendline_shift * (1 + FLOAT_COMPARISON_TOLERANCE)
            df_out["exit_short_signal"] = (exit_short_cond1 | exit_short_cond2 | exit_short_cond3).fillna(False).astype(bool)

        # --- Final Processing & Validation ---
        # Ensure boolean columns are explicitly boolean type and NaNs are False
        bool_cols = ["long_signal", "short_signal", "exit_long_signal", "exit_short_signal",
                     "trend_up", "trend_down", "trend_neutral", "high_volume"]
        for col in bool_cols:
            if col in df_out.columns:
                # Fill potential NaNs arising from logic gaps before converting type
                df_out[col] = df_out[col].fillna(False).astype(bool)

        # Final check for validity - ensure required columns for trading logic exist and have valid data in last few rows
        final_check_cols = ["close", "atr_risk", "long_signal", "short_signal", "exit_long_signal", "exit_short_signal"]
        if df_out.empty or not all(col in df_out.columns for col in final_check_cols):
            log_console(logging.ERROR, "Indicator Calc: DataFrame empty or missing critical columns after final processing.")
            return None

        # Check last N rows for NaNs in critical columns (if enough rows exist)
        min_rows_for_check = 2 # Check last closed candle and latest candle
        if len(df_out) >= min_rows_for_check:
            if df_out.iloc[-min_rows_for_check:][final_check_cols].isnull().any().any():
                 log_console(logging.ERROR, f"Indicator Calc: Critical columns contain NaN in the last {min_rows_for_check} rows after processing. Check calculations and NaN fills.")
                 # Log the problematic rows for debugging
                 log_console(logging.DEBUG, f"Last {min_rows_for_check} rows tail with potential NaNs:\n{df_out.iloc[-min_rows_for_check:][final_check_cols]}")
                 return None
        elif len(df_out) == 1: # Handle edge case with only one row
             if df_out.iloc[-1:][final_check_cols].isnull().any().any():
                 log_console(logging.ERROR, "Indicator Calc: Critical columns contain NaN in the only row after processing.")
                 return None

        # Return the DataFrame with all calculated indicators and signals
        return df_out

    except Exception as e:
        log_console(logging.ERROR, f"Indicator calculation failed unexpectedly: {e}", exc_info=True)
        return None


# --- Bybit API Interaction Helpers ---
def get_available_balance(session: HTTP, coin: str = "USDT", account_type: str = "UNIFIED") -> float:
    """
    Fetches the available balance for a specific coin from the Bybit wallet using V5 API.
    Handles UNIFIED, CONTRACT, and SPOT account structures robustly.

    Args:
        session: Authenticated Bybit HTTP session object.
        coin: The coin symbol (e.g., "USDT"). Case-sensitive matching Bybit's response.
        account_type: The account type ("UNIFIED", "CONTRACT", "SPOT"). MUST match account structure.

    Returns:
        The available balance as a float, or 0.0 if fetch fails, coin not found,
        balance is zero/negative, or response format is unexpected.
    """
    try:
        # V5 API call uses 'accountType' parameter
        # Ensure account_type is uppercase as expected by the endpoint
        response = session.get_wallet_balance(accountType=account_type.upper(), coin=coin)

        if response and response.get("retCode") == 0:
            result = response.get("result", {})
            list_data = result.get("list", [])

            if not list_data:
                log_console(logging.DEBUG, f"No balance list found for {account_type} account (maybe empty?). Treating balance as 0.")
                return 0.0

            # --- Parsing Logic for V5 'get_wallet_balance' Response ---
            # The structure varies slightly depending on the account type.
            coin_balance_list: List[Dict[str, Any]] = []

            if account_type.upper() == "UNIFIED":
                # For UNIFIED, the coin balances are nested within the first element of 'list'
                account_data = list_data[0] if list_data else {}
                coin_balance_list = account_data.get("coin", [])
                if not isinstance(coin_balance_list, list):
                     log_console(logging.WARNING, f"Expected list for 'coin' in UNIFIED balance response, got {type(coin_balance_list)}. Parsing failed.")
                     return 0.0
            elif account_type.upper() in ["CONTRACT", "SPOT"]:
                 # For CONTRACT/SPOT, 'list' *should* contain the coin balances directly (as list of dicts)
                 if isinstance(list_data, list) and all(isinstance(item, dict) for item in list_data):
                     coin_balance_list = list_data # Assumes direct list of coin dicts
                 else:
                     log_console(logging.WARNING, f"Expected list of dicts for '{account_type}' balance response list, got different structure. Parsing failed.")
                     return 0.0
            else:
                # Handle unknown account types with a warning, attempt UNIFIED structure as fallback
                log_console(logging.WARNING, f"Balance parsing logic not explicitly defined for account type: {account_type}. Attempting UNIFIED structure.")
                account_data = list_data[0] if list_data else {}
                coin_balance_list = account_data.get("coin", [])
                if not isinstance(coin_balance_list, list):
                     log_console(logging.WARNING, f"Fallback parsing for '{account_type}' failed (expected list for 'coin').")
                     return 0.0

            # If after parsing, coin_balance_list is empty
            if not coin_balance_list:
                log_console(logging.WARNING, f"No coin balance details found within balance data structure for {account_type}.")
                return 0.0

            # Iterate through the found coin balances to find the requested coin
            for coin_data in coin_balance_list:
                if isinstance(coin_data, dict) and coin_data.get("coin") == coin:
                    # Prioritize V5 fields representing usable balance:
                    # 'availableToWithdraw' or 'availableBalance' are typically the most relevant.
                    balance_str = coin_data.get("availableToWithdraw")
                    source = "availableToWithdraw"
                    if balance_str is None or balance_str == '':
                        balance_str = coin_data.get("availableBalance") # Often used in Unified/Contract
                        source = "availableBalance"

                    # Fallback to 'walletBalance' (total balance) if others are missing/empty
                    if balance_str is None or balance_str == '':
                        balance_str = coin_data.get("walletBalance") # General total balance
                        source = "walletBalance"
                        log_console(logging.DEBUG, f"'availableToWithdraw'/'availableBalance' empty for {coin}, using '{source}' as fallback.")

                    # Handle case where all relevant balance fields are empty or missing
                    if balance_str is None or balance_str == '':
                        log_console(logging.WARNING, f"Could not find a valid available balance field (availableToWithdraw, availableBalance, walletBalance) for {coin}. Balance is 0.")
                        balance_str = "0" # Treat as zero if no field found

                    # Convert the found balance string to float safely
                    try:
                        balance_float = float(balance_str)
                        # Ensure balance is not negative (can happen with negative PnL in some views)
                        final_balance = max(0.0, balance_float)
                        log_console(logging.DEBUG, f"Found balance for {coin} ({source}): {final_balance:.8f}")
                        return final_balance
                    except (ValueError, TypeError) as e:
                        log_console(logging.ERROR, f"Could not convert balance string '{balance_str}' from field '{source}' to float for {coin}: {e}")
                        return 0.0 # Return 0 if conversion fails

            # If loop completes without finding the coin
            log_console(logging.WARNING, f"Coin '{coin}' not found in the wallet balance details for {account_type} account.")
            return 0.0

        else: # API call failed (retCode != 0 or empty response)
            error_msg = response.get('retMsg', 'Unknown error') if response else "Empty/invalid response"
            error_code = response.get('retCode', 'N/A')
            log_console(logging.ERROR, f"Failed to fetch wallet balance: {error_msg} (Code: {error_code})")
            return 0.0

    except Exception as e:
        log_console(logging.ERROR, f"Exception occurred while fetching wallet balance: {e}", exc_info=True)
        return 0.0


def calculate_position_size_atr(
    balance: float,
    risk_percent: float,
    sl_distance_price: float,
    entry_price: float,
    min_order_qty: float,
    qty_step_float: float,
    qty_precision: int,
    max_position_usdt: Optional[float] = None,
) -> float:
    """
    Calculates the position size based on account balance, risk percentage,
    and ATR-based stop-loss distance. Ensures compliance with minimum order
    quantity and quantity step rules by rounding DOWN to the nearest valid step.

    Args:
        balance: Available trading balance in USDT allocated for this trade/symbol.
        risk_percent: The percentage of the allocated balance to risk per trade (e.g., 1.0 for 1%).
        sl_distance_price: The absolute price difference for the stop loss (must be positive).
        entry_price: The estimated entry price of the trade (must be positive).
        min_order_qty: Minimum order quantity allowed by the instrument (must be positive).
        qty_step_float: The quantity step (increment) as a positive float.
        qty_precision: The number of decimal places for the quantity (non-negative integer).
        max_position_usdt: Optional maximum value of the position in USDT (must be positive if set).

    Returns:
        The calculated position size (quantity), rounded DOWN to the instrument's
        step precision, or 0.0 if calculation is not possible, inputs are invalid,
        or the resulting size is below the minimum required quantity.
    """
    # --- Input Validation ---
    # Check types
    if not all(isinstance(x, (int, float)) for x in [balance, risk_percent, sl_distance_price, entry_price, min_order_qty, qty_step_float]):
        log_console(logging.DEBUG, f"Position Size Calc: Invalid input types. Bal:{type(balance)}, Risk%:{type(risk_percent)}, SLDist:{type(sl_distance_price)}, Entry:{type(entry_price)}, MinQty:{type(min_order_qty)}, Step:{type(qty_step_float)}. Returning size 0.")
        return 0.0
    if not isinstance(qty_precision, int) or qty_precision < 0:
         log_console(logging.ERROR, f"Position Size Calc: Invalid qty_precision ({qty_precision}). Must be non-negative integer. Returning size 0.")
         return 0.0
    # Check positive values for required inputs
    if balance <= FLOAT_COMPARISON_TOLERANCE or risk_percent <= FLOAT_COMPARISON_TOLERANCE or \
       entry_price <= FLOAT_COMPARISON_TOLERANCE or min_order_qty <= FLOAT_COMPARISON_TOLERANCE or \
       qty_step_float <= FLOAT_COMPARISON_TOLERANCE:
        # Use DEBUG as this might happen normally with zero balance or zero risk setting
        log_console(logging.DEBUG, f"Position Size Calc: Non-positive input values detected. Balance={balance:.2f}, Risk%={risk_percent}, Entry={entry_price}, MinQty={min_order_qty}, QtyStep={qty_step_float}. Returning size 0.")
        return 0.0
    # Prevent division by zero or huge sizes from tiny SL distance
    if sl_distance_price <= FLOAT_COMPARISON_TOLERANCE:
        log_console(logging.WARNING, f"Position Size Calc: Stop loss distance ({sl_distance_price}) is zero or too small. Cannot calculate size reliably. Returning 0.")
        return 0.0
    # Validate optional max position value if provided
    if max_position_usdt is not None and (not isinstance(max_position_usdt, (int, float)) or max_position_usdt <= FLOAT_COMPARISON_TOLERANCE):
        log_console(logging.WARNING, f"Position Size Calc: Invalid max_position_usdt ({max_position_usdt}). Ignoring constraint.")
        max_position_usdt = None # Ignore invalid value

    # --- Calculation ---
    # Calculate risk amount in USDT
    risk_amount_usdt = balance * (risk_percent / 100.0)
    log_console(logging.DEBUG, f"Position Size Calc: Risk Amount = {risk_amount_usdt:.4f} USDT")

    # Calculate initial position size based on risk amount and SL distance per unit
    # This assumes linear contracts where PnL = Size * PriceDiff
    try:
        position_size = risk_amount_usdt / sl_distance_price
        log_console(logging.DEBUG, f"Position Size Calc: Initial raw size (Risk/SL): {position_size:.{qty_precision+4}f}")
    except ZeroDivisionError: # Should be caught by check above, but safeguard
        log_console(logging.WARNING, "Position Size Calc: SL distance is effectively zero during calculation. Cannot calculate size.")
        return 0.0

    # --- Apply Constraints ---
    # 1. Max Position Value Constraint (if applicable)
    if max_position_usdt is not None:
        try:
            # Ensure entry price is valid for this calculation
            if entry_price <= FLOAT_COMPARISON_TOLERANCE:
                 log_console(logging.WARNING, "Position Size Calc: Entry price is zero or negative. Cannot apply Max Position USDT constraint.")
            else:
                max_size_by_value = max_position_usdt / entry_price
                if position_size > max_size_by_value:
                    log_console(logging.INFO, f"Position Size Calc: Capping size from {position_size:.{qty_precision}f} to {max_size_by_value:.{qty_precision}f} due to Max Position USDT (${max_position_usdt:.2f})")
                    position_size = max_size_by_value
        except ZeroDivisionError:
             log_console(logging.WARNING, "Position Size Calc: Entry price is zero during Max Position USDT constraint check.")
             # Continue without applying the constraint in this edge case

    # 2. Minimum Order Quantity Constraint (Check BEFORE step rounding)
    # Use a small tolerance related to qty_step to avoid floating point issues right at the minimum boundary
    min_qty_tolerance = qty_step_float * 0.01
    if position_size < min_order_qty - min_qty_tolerance:
        log_console(logging.DEBUG, f"Position Size Calc: Calculated size {position_size:.{qty_precision+4}f} (after potential cap) is below minimum required {min_order_qty}. Returning size 0.")
        return 0.0

    # 3. Quantity Step Constraint (Rounding DOWN)
    try:
        if qty_step_float <= FLOAT_COMPARISON_TOLERANCE:
             # This was checked earlier, but double-check for safety
            log_console(logging.ERROR, f"Position Size Calc: Quantity step ({qty_step_float}) is zero or too small during rounding. Cannot apply step constraint.")
            return 0.0

        # Explicitly round DOWN to the nearest multiple of qty_step_float
        # Use math.floor for clear downward rounding
        # Add a tiny epsilon before division to handle floating point inaccuracies
        # where position_size might be slightly less than a multiple of step_size
        # (e.g., 0.09999999999999999 instead of 0.1 when step is 0.01)
        epsilon = qty_step_float * 1e-9 # Epsilon should be much smaller than step size
        num_steps = math.floor((position_size + epsilon) / qty_step_float)
        position_size_adjusted = num_steps * qty_step_float

        if position_size_adjusted != position_size: # Log only if rounding occurred
             log_console(logging.DEBUG, f"Position Size Calc: Size after step rounding DOWN ({qty_step_float}): {position_size_adjusted:.{qty_precision+4}f}")
        position_size = position_size_adjusted
    except ZeroDivisionError: # Should not happen due to earlier checks
        log_console(logging.ERROR, "Position Size Calc: Quantity step is zero during rounding step. Cannot apply constraint.")
        return 0.0
    except Exception as round_e:
         log_console(logging.ERROR, f"Position Size Calc: Unexpected error during step rounding: {round_e}")
         return 0.0

    # 4. Final check: ensure size is still >= minimum after rounding DOWN
    if position_size < min_order_qty - min_qty_tolerance:
        log_console(logging.INFO, f"Position Size Calc: Final size {position_size:.{qty_precision+4}f} is below minimum {min_order_qty} after rounding down to step. Returning size 0.")
        return 0.0

    # 5. Final rounding to specified precision (mostly cosmetic after step rounding, but ensures adherence)
    # Use standard round for this final step to the required decimal places
    final_size = round(position_size, qty_precision)

    # Ensure final size is not effectively zero after all calculations and rounding
    # Check against min_order_qty again, as rounding could theoretically make it too small
    if final_size < min_order_qty - min_qty_tolerance:
        log_console(logging.DEBUG, f"Position Size Calc: Final calculated size ({final_size}) is below min qty ({min_order_qty}) after final rounding. Returning size 0.")
        return 0.0

    log_console(logging.INFO, f"Position Size Calc: Balance={balance:.2f}, Risk%={risk_percent}, SL Dist={sl_distance_price:.{max(qty_precision, 2)+2}f}, Entry={entry_price:.{max(qty_precision, 2)+2}f} -> Final Size = {final_size:.{qty_precision}f}")
    return final_size


# --- Trade Performance Tracking ---
class TradeMetrics:
    """
    Tracks and logs performance metrics for completed trades in a thread-safe manner.
    """
    def __init__(self, fee_rate: float):
        """
        Initializes the TradeMetrics instance.

        Args:
            fee_rate: The estimated trading fee rate per trade side (e.g., 0.0006 for 0.06%).
                      Should typically be the TAKER fee rate if using market orders.
        """
        self.trades: List[Dict[str, Any]] = []
        # Validate fee rate on initialization
        if not isinstance(fee_rate, (int, float)) or fee_rate < 0:
            log_console(logging.ERROR, f"Metrics Init: Invalid fee rate provided ({fee_rate}). Setting to default 0.0006 (0.06%).")
            self.fee_rate = 0.0006
        else:
            self.fee_rate = fee_rate
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.total_fees_paid = 0.0
        self.last_summary_time: Optional[dt.datetime] = None
        self.lock = threading.Lock() # Lock for thread-safe access to shared metrics data

    def add_trade(self, symbol: str, entry_time: dt.datetime, exit_time: dt.datetime,
                  side: str, entry_price: float, exit_price: float, qty: float, leverage: Union[int, str]):
        """
        Adds a completed trade to the tracker, calculates its P&L and fees, and logs it.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT").
            entry_time: Datetime object of trade entry (UTC recommended).
            exit_time: Datetime object of trade exit (UTC recommended).
            side: "Buy" (for Long) or "Sell" (for Short).
            entry_price: Average entry price (must be positive).
            exit_price: Average exit price (must be positive).
            qty: Quantity traded (absolute value, must be positive).
            leverage: Leverage used for the trade (can be int or string representation of int).
        """
        with self.lock: # Ensure thread safety when modifying shared metrics
            try:
                # --- Input Validation ---
                leverage_int = 1 # Default leverage if conversion fails or invalid
                try:
                    # Handle potential empty string, None, or non-numeric leverage values
                    leverage_val_str = str(leverage).strip()
                    if leverage_val_str:
                        leverage_val = float(leverage_val_str)
                        # Ensure positive integer leverage >= 1
                        leverage_int = max(1, int(leverage_val))
                    else:
                         log_console(logging.WARNING, f"Metrics: Empty leverage value provided for {symbol}. Using 1x.", symbol=symbol)
                except (ValueError, TypeError):
                     log_console(logging.ERROR, f"Metrics: Invalid leverage value '{leverage}' for {symbol}. Using 1x.", symbol=symbol)

                # Validate other inputs robustly
                if not all([
                    isinstance(symbol, str) and symbol,
                    isinstance(entry_time, dt.datetime),
                    isinstance(exit_time, dt.datetime),
                    isinstance(side, str) and side in ["Buy", "Sell"],
                    isinstance(entry_price, (int, float)) and entry_price > FLOAT_COMPARISON_TOLERANCE,
                    isinstance(exit_price, (int, float)) and exit_price > FLOAT_COMPARISON_TOLERANCE,
                    isinstance(qty, (int, float)) and qty > FLOAT_COMPARISON_TOLERANCE,
                    isinstance(leverage_int, int) and leverage_int >= 1
                ]):
                    log_console(logging.ERROR, f"Metrics: Invalid trade data received for {symbol}. Cannot log. Data: "
                                                f"entry_t={entry_time}, exit_t={exit_time}, side={side}, "
                                                f"entry_p={entry_price}, exit_p={exit_price}, qty={qty}, lev='{leverage}'",
                                                symbol=symbol)
                    return # Do not proceed with invalid data

                abs_qty = abs(qty) # Ensure quantity is positive for calculations
                # Calculate price difference based on trade side
                price_diff = exit_price - entry_price if side == "Buy" else entry_price - exit_price
                # Calculate Gross P&L for Linear Contracts (Quantity * Price Difference)
                gross_pnl = price_diff * abs_qty

                # --- Fees Estimation ---
                # Apply fee rate to the notional value traded on both entry and exit
                entry_value = abs_qty * entry_price
                exit_value = abs_qty * exit_price
                # Ensure fee rate is applied correctly (as a percentage)
                fees = (entry_value * self.fee_rate) + (exit_value * self.fee_rate)
                # Calculate Net P&L
                net_pnl = gross_pnl - fees

                # --- Update Aggregate Metrics ---
                is_win = net_pnl > FLOAT_COMPARISON_TOLERANCE # Consider trade a win only if PnL is significantly positive
                self.total_trades += 1
                if is_win: self.wins += 1
                else: self.losses += 1
                self.total_pnl += net_pnl
                self.total_fees_paid += fees

                # Calculate trade duration safely
                trade_duration_sec = max(0.0, (exit_time - entry_time).total_seconds())

                # Define consistent precision for logging financials (adjust based on typical asset values)
                pnl_precision = 4 # e.g., 4 decimal places for USDT P&L
                price_precision = 6 # Adjust based on typical asset tick size (e.g., BTCUSDT)
                qty_precision = 6   # Adjust based on typical asset quantity step (e.g., BTCUSDT)

                # Store trade details in a dictionary
                trade = {
                    "symbol": symbol,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"), # Format time for logging
                    "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": round(trade_duration_sec, 2),
                    "side": side,
                    "entry_price": round(entry_price, price_precision),
                    "exit_price": round(exit_price, price_precision),
                    "qty": round(abs_qty, qty_precision),
                    "leverage": leverage_int,
                    "gross_pnl": round(gross_pnl, pnl_precision),
                    "fees": round(fees, pnl_precision),
                    "net_pnl": round(net_pnl, pnl_precision),
                    "is_win": is_win,
                }
                self.trades.append(trade) # Append to the list of trades

                # Log individual trade metrics in CSV-like format using the dedicated logger
                log_metrics(
                    f"TRADE,{symbol},{trade['entry_time']},{trade['exit_time']},{trade['duration_seconds']},"
                    f"{side},{trade['entry_price']:.{price_precision}f},{trade['exit_price']:.{price_precision}f},"
                    f"{trade['qty']:.{qty_precision}f},{trade['leverage']},{trade['gross_pnl']:.{pnl_precision}f},"
                    f"{trade['fees']:.{pnl_precision}f},{trade['net_pnl']:.{pnl_precision}f},"
                    f"{'Win' if is_win else 'Loss'}"
                )
            except Exception as e:
                # Catch any unexpected errors during metric processing/logging
                log_console(logging.ERROR, f"Error processing or logging trade metrics for {symbol}: {e}", symbol=symbol, exc_info=True)

    def log_summary(self, symbol: Optional[str] = None, force: bool = False, interval: int = 3600):
        """
        Logs summary performance metrics periodically or when forced.

        Args:
            symbol: Optional symbol name for context (usually None for global summary).
            force: If True, log summary regardless of interval.
            interval: Minimum seconds between automatic summary logs.
        """
        with self.lock: # Ensure thread safety for reading metrics data
            current_time = dt.datetime.now(dt.timezone.utc)
            # Determine if logging is needed now
            log_now = force or self.total_trades == 0 # Log if forced or if it's the first time (no trades yet)

            if not log_now and self.last_summary_time:
                time_since_last = (current_time - self.last_summary_time).total_seconds()
                if time_since_last >= interval:
                    log_now = True # Log if interval has passed

            if not log_now:
                return # Exit if no need to log summary now

            if self.total_trades > 0:
                # Calculate summary statistics safely
                win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0
                avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0

                # Calculate Profit Factor (Total Gains / Total Losses)
                profit_factor = 0.0
                total_gains = sum(t['net_pnl'] for t in self.trades if t.get('is_win', False))
                # Sum absolute values of losses
                total_losses = abs(sum(t['net_pnl'] for t in self.trades if not t.get('is_win', False)))

                # Handle division by zero for profit factor
                if total_losses > FLOAT_COMPARISON_TOLERANCE:
                    profit_factor = total_gains / total_losses
                elif total_gains > FLOAT_COMPARISON_TOLERANCE: # Handle case with wins but zero losses
                    profit_factor = float('inf') # Infinite profit factor

                pnl_precision = 4 # Consistent precision for summary P&L
                # Format the summary string
                summary = (
                    f"SUMMARY,{symbol or 'ALL_SYMBOLS'},TotalTrades={self.total_trades},Wins={self.wins},Losses={self.losses},"
                    f"WinRate={win_rate:.2f}%,ProfitFactor={profit_factor:.2f},"
                    f"TotalNetPnL={self.total_pnl:.{pnl_precision}f} USDT,"
                    f"AvgNetPnL={avg_pnl:.{pnl_precision}f} USDT,TotalFees={self.total_fees_paid:.{pnl_precision}f} USDT"
                )
            else:
                # Message if no trades have been recorded yet
                summary = f"SUMMARY,{symbol or 'ALL_SYMBOLS'},No trades executed yet."

            # Log the summary using the dedicated metrics logger
            log_metrics(summary)
            # Update the time of the last summary log
            self.last_summary_time = current_time

    def save_on_shutdown(self):
        """Logs a final summary when the bot is shutting down."""
        with self.lock: # Ensure thread safety
            log_console(logging.INFO, "Generating final trade metrics summary...")
            # Force logging the summary regardless of the interval
            self.log_summary(force=True)


# --- Single Symbol Trading Logic ---
class SymbolTrader:
    """
    Manages all trading activities, state, and API interactions for a single symbol
    based on its specific configuration. Uses Bybit V5 API endpoints.
    """
    def __init__(self, api_key: str, api_secret: str, config: Dict[str, Any], symbol_cfg: Dict[str, Any],
                 dry_run: bool, metrics: TradeMetrics, session: HTTP, category: str):
        """
        Initializes the SymbolTrader instance, fetches instrument info, and performs initial setup.

        Args:
            api_key, api_secret: Bybit API credentials.
            config: Full application configuration dictionary.
            symbol_cfg: Symbol-specific configuration dictionary (includes INTERNAL calculated values).
            dry_run: Global dry-run flag (True = simulation only).
            metrics: Shared TradeMetrics instance for logging performance.
            session: Shared authenticated Bybit HTTP session object (V5).
            category: Trading category (e.g., "linear", "spot") determined by the orchestrator.

        Raises:
            RuntimeError: If critical initialization steps fail (e.g., fetching instrument info).
        """
        self.config = config
        self.symbol_cfg = symbol_cfg
        self.dry_run = dry_run
        self.metrics = metrics
        self.session = session # Use the shared HTTP session
        self.category = category # e.g., "linear", "spot"

        # Extract configuration parameters safely with defaults
        self.symbol: str = symbol_cfg.get("SYMBOL", "MISSING_SYMBOL")
        self.timeframe: str = str(symbol_cfg.get("TIMEFRAME", "15")) # Ensure string
        self.leverage: int = int(symbol_cfg.get("LEVERAGE", 5)) # Ensure int
        self.use_websocket: bool = symbol_cfg.get("USE_WEBSOCKET", True)
        self.strategy_cfg: Dict[str, Any] = symbol_cfg.get("STRATEGY_CONFIG", {})
        self.risk_cfg: Dict[str, Any] = config.get("RISK_CONFIG", {})
        self.bot_cfg: Dict[str, Any] = config.get("BOT_CONFIG", {})
        self.internal_cfg: Dict[str, Any] = symbol_cfg.get("INTERNAL", {}) # Contains calculated KLINE_LIMIT

        # Instrument details - Initialize with sensible defaults, fetched later
        self.min_order_qty: float = 0.000001 # Set a very small default non-zero value
        self.qty_step: str = "0.000001" # String representation from API
        self.qty_step_float: float = 0.000001 # Float representation for calculations
        self.qty_precision: int = 6 # Deduced from qty_step
        self.tick_size: str = "0.01" # String representation from API
        self.tick_size_float: float = 0.01 # Float representation for calculations
        self.price_precision: int = 2 # Deduced from tick_size

        # State variables
        self.last_exit_time: Optional[dt.datetime] = None # Timestamp of last position exit for cooldown
        self.kline_queue: Queue = Queue() # Queue for incoming WebSocket kline updates
        self.kline_df: Optional[pd.DataFrame] = None # Stores the latest OHLCV data with indicators
        self.higher_tf_cache: Optional[bool] = None # Cache for higher timeframe trend result
        self.higher_tf_cache_time: Optional[dt.datetime] = None # Timestamp for HTF cache expiry
        self.current_trade: Optional[Dict[str, Any]] = None # Stores state of the current open position (real or simulated)
        self.order_confirm_retries: int = self.bot_cfg.get("ORDER_CONFIRM_RETRIES", 3)
        self.order_confirm_delay: float = float(self.bot_cfg.get("ORDER_CONFIRM_DELAY_SECONDS", 2.0))
        self.initialization_successful: bool = False # Flag indicating successful init
        self.last_kline_update_time: Optional[dt.datetime] = None # Track last successful data update

        if self.dry_run:
            log_console(logging.WARNING, f"DRY RUN MODE enabled. No real orders will be placed for {self.symbol}.", symbol=self.symbol)

        # --- Perform Initialization Steps ---
        try:
            # 1. Fetch Essential Instrument Information (mandatory)
            if not self._fetch_instrument_info():
                # Raise error if basic info cannot be fetched, trader cannot function
                raise RuntimeError(f"Failed to fetch essential instrument info for {self.symbol}. Cannot initialize trader.")

            # 2. Perform Initial Setup (Leverage/Margin) - Skip for Spot and Dry Run
            if not self.dry_run and self.category != "spot":
                # Leverage/margin setup is not applicable to spot trading
                self._initial_setup()
                # Note: _initial_setup logs warnings but doesn't raise fatal errors if leverage setting fails,
                # allowing continuation if leverage was set previously or user accepts the risk.

            # If all critical steps passed
            self.initialization_successful = True
            log_console(logging.DEBUG, f"Trader for {self.symbol} initialized successfully.", symbol=self.symbol)

        except Exception as e:
            # Catch any exception during initialization steps
            log_console(logging.CRITICAL, f"Initialization FAILED for trader {self.symbol}: {e}", symbol=self.symbol, exc_info=False)
            # Raise a specific error to be caught by the orchestrator, indicating failure
            raise RuntimeError(f"Trader Init Failed for {self.symbol}") from e

    def _fetch_instrument_info(self) -> bool:
        """Fetches and stores instrument details (min qty, step, precision) using V5 API."""
        log_console(logging.DEBUG, f"Fetching instrument info for {self.symbol} (Category: {self.category})...", symbol=self.symbol)
        try:
            # Use V5 endpoint get_instruments_info
            response = self.session.get_instruments_info(category=self.category, symbol=self.symbol)

            if response and response.get("retCode") == 0:
                result_list = response.get("result", {}).get("list", [])
                if result_list and isinstance(result_list[0], dict):
                    instrument = result_list[0] # Assuming the first item is the correct one for the symbol
                    lot_size_filter = instrument.get("lotSizeFilter", {})
                    price_filter = instrument.get("priceFilter", {})

                    # Safely extract and convert filter values
                    try:
                        # Quantity related filters
                        min_qty_str = lot_size_filter.get("minOrderQty", "0")
                        self.min_order_qty = float(min_qty_str)
                        self.qty_step = lot_size_filter.get("qtyStep", "1") # Keep original string
                        self.qty_step_float = float(self.qty_step)

                        # Price related filters
                        self.tick_size = price_filter.get("tickSize", "1") # Keep original string
                        self.tick_size_float = float(self.tick_size)

                        # Calculate precision based on decimal places of the step/tick size strings
                        # Handle cases where step/tick might be integers (e.g., "1")
                        if '.' in self.qty_step and self.qty_step_float > FLOAT_COMPARISON_TOLERANCE:
                            # Count characters after the decimal point
                            decimal_part = self.qty_step.split('.')[-1]
                            self.qty_precision = len(decimal_part)
                        else: self.qty_precision = 0 # Precision is 0 for integer steps

                        if '.' in self.tick_size and self.tick_size_float > FLOAT_COMPARISON_TOLERANCE:
                            decimal_part = self.tick_size.split('.')[-1]
                            self.price_precision = len(decimal_part)
                        else: self.price_precision = 0 # Precision is 0 for integer ticks

                    except (ValueError, TypeError) as conv_e:
                        log_console(logging.ERROR, f"Failed to convert instrument filter values to numbers: {conv_e}. Filters: LotSize={lot_size_filter}, Price={price_filter}", symbol=self.symbol)
                        return False

                    # Validate fetched values (ensure key values are positive)
                    if self.min_order_qty <= FLOAT_COMPARISON_TOLERANCE or self.qty_step_float <= FLOAT_COMPARISON_TOLERANCE or self.tick_size_float <= FLOAT_COMPARISON_TOLERANCE:
                        log_console(logging.ERROR, f"Fetched invalid instrument details (non-positive values): MinQty={self.min_order_qty}, QtyStep={self.qty_step_float}, TickSize={self.tick_size_float}", symbol=self.symbol)
                        return False

                    # Log the successfully fetched and calculated details
                    log_console(
                        logging.INFO,
                        f"Instrument Info Fetched: MinQty={self.min_order_qty}, "
                        f"QtyStep='{self.qty_step}' (Float: {self.qty_step_float:.{max(8, self.qty_precision)}f}, Precision: {self.qty_precision}), "
                        f"TickSize='{self.tick_size}' (Float: {self.tick_size_float:.{max(8, self.price_precision)}f}, Precision: {self.price_precision})",
                        symbol=self.symbol,
                    )
                    return True # Success
                else:
                    log_console(logging.ERROR, f"Instrument info fetch failed: Empty or invalid result list in response for {self.symbol}.", symbol=self.symbol)
                    return False
            else: # API call failed
                error_msg = response.get('retMsg', 'N/A') if response else "Empty response"
                error_code = response.get('retCode', 'N/A')
                log_console(logging.ERROR, f"Instrument info fetch API call failed: {error_msg} (Code: {error_code})", symbol=self.symbol)
                # Specific check for invalid symbol error (e.g., 10001 or relevant message)
                if error_code == 10001 or "invalid symbol" in error_msg.lower():
                     log_console(logging.ERROR, f"----> Is symbol '{self.symbol}' valid for the '{self.category}' category on Bybit?", symbol=self.symbol)
                return False
        except Exception as e:
            log_console(logging.ERROR, f"Exception during instrument info fetch: {e}", symbol=self.symbol, exc_info=True)
            return False

    def _initial_setup(self):
        """Sets leverage using V5 API. Only applicable for non-spot categories (linear/inverse)."""
        # This setup is skipped for SPOT category and in dry run mode.
        if self.category == "spot":
            log_console(logging.DEBUG, "Skipping leverage setup for SPOT category.", symbol=self.symbol)
            return
        if self.dry_run:
            log_console(logging.INFO, "Skipping leverage setting in dry run mode.", symbol=self.symbol)
            return

        log_console(logging.INFO, f"Performing initial setup: Setting leverage to {self.leverage}x for {self.symbol}...", symbol=self.symbol)
        try:
            # V5 API call to set leverage requires string values
            response = self.session.set_leverage(
                category=self.category,
                symbol=self.symbol,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage)
            )
            if response and response.get("retCode") == 0:
                log_console(logging.INFO, f"Leverage successfully set to {self.leverage}x.", symbol=self.symbol)
            else:
                error_msg = response.get('retMsg', 'N/A') if response else "Empty response"
                error_code = response.get('retCode', 'N/A')
                # Handle common non-critical errors gracefully (e.g., leverage already set)
                # V5 Error Code 110043: leverage not modified
                # V5 Error Code 110026: Margin mode related error (can sometimes occur if leverage matches cross margin mode default)
                already_set_codes = [110043, 110026]
                already_set_msgs = ["leverage not modified", "same leverage"]
                if error_code in already_set_codes or any(msg in error_msg.lower() for msg in already_set_msgs):
                    log_console(logging.INFO, f"Leverage already set to {self.leverage}x or not modified (Code: {error_code}, Msg: '{error_msg}'). Assuming OK.", symbol=self.symbol)
                else:
                    # Log other errors as warnings, as trading might still proceed if leverage was set previously.
                    log_console(logging.WARNING, f"Failed to set leverage: {error_msg} (Code: {error_code}). Check API permissions and account settings (e.g., margin mode). Continuing...", symbol=self.symbol)

            # --- Optional: Add Margin Mode Setting ---
            # Example: Set to ISOLATED margin if needed. Requires careful checking of V5 parameters.
            # margin_mode_config = self.symbol_cfg.get("MARGIN_MODE", "ISOLATED").upper() # Or "CROSSED"
            # desired_trade_mode = 0 if margin_mode_config == "ISOLATED" else 1 # 0=ISOLATED, 1=CROSSED
            # try:
            #     log_console(logging.INFO, f"Setting margin mode to {margin_mode_config} ({desired_trade_mode})...", symbol=self.symbol)
            #     response_margin = self.session.switch_margin_mode(
            #          category=self.category, symbol=self.symbol, tradeMode=desired_trade_mode,
            #          buyLeverage=str(self.leverage), sellLeverage=str(self.leverage)
            #     )
            #     if response_margin and response_margin.get("retCode") == 0:
            #          log_console(logging.INFO, f"Margin mode successfully set to {margin_mode_config}.", symbol=self.symbol)
            #     else:
            #          margin_error_msg = response_margin.get('retMsg', 'N/A') if response_margin else "Empty response"
            #          margin_error_code = response_margin.get('retCode', 'N/A')
            #          # Handle "margin mode not modified" (e.g., 110025)
            #          if margin_error_code == 110025 or "margin mode not modified" in margin_error_msg.lower():
            #               log_console(logging.INFO, f"Margin mode already set to {margin_mode_config} or not modified (Code: {margin_error_code}).", symbol=self.symbol)
            #          else:
            #               log_console(logging.WARNING, f"Failed to set margin mode: {margin_error_msg} (Code: {margin_error_code}).", symbol=self.symbol)
            # except Exception as margin_e:
            #     log_console(logging.WARNING, f"Exception setting margin mode: {margin_e}. Continuing...", symbol=self.symbol)

        except Exception as e:
            # Log exceptions during setup as warnings and continue if possible
            log_console(logging.WARNING, f"Exception during initial setup (leverage/margin): {e}. Continuing...", symbol=self.symbol, exc_info=True)

    def _websocket_callback(self, message: Dict[str, Any]):
        """
        Callback specifically for processing V5 WebSocket kline stream messages
        belonging to this SymbolTrader instance. Queues processed data.
        """
        try:
            # Minimal logging here to avoid flooding, focus on processing
            # log_console(logging.DEBUG, f"WS Received Raw for {self.symbol}: {message}", symbol=self.symbol) # Verbose, enable if needed

            # Basic validation of message structure
            if not isinstance(message, dict): return # Ignore non-dict messages

            topic = message.get("topic")
            data_list = message.get("data")

            # Ensure topic and data list exist and are of expected types
            if not topic or not isinstance(data_list, list): return

            # V5 Kline Topic format: kline.{interval}.{symbol}
            # Double-check if the message truly belongs to this trader instance
            # (Dispatcher should handle this, but safety check is good)
            topic_parts = topic.split('.')
            if not (len(topic_parts) == 3 and topic_parts[0] == "kline" and
                    topic_parts[1] == self.timeframe and topic_parts[2] == self.symbol):
                # This message somehow bypassed the dispatcher or is malformed, ignore silently.
                return

            if not data_list: return # Ignore messages with empty data list

            # Process each kline update in the data list
            for kline_raw in data_list:
                if not isinstance(kline_raw, dict): continue # Skip invalid items in data list

                # Check confirmation status (indicates closed candle vs. ongoing)
                is_confirmed = kline_raw.get("confirm", False)
                # Get config setting whether to process only confirmed candles
                process_confirmed_only = self.bot_cfg.get("PROCESS_CONFIRMED_KLINE_ONLY", True)

                # Process based on confirmation status and configuration
                if not process_confirmed_only or (process_confirmed_only and is_confirmed):
                    try:
                        # Extract V5 Kline data fields and convert types robustly
                        timestamp_ms = int(kline_raw["start"]) # Start time of the kline interval
                        # Create a dictionary with standardized keys and correct types
                        kline_processed = {
                            # Convert timestamp to pandas DatetimeIndex, ensuring UTC timezone
                            "timestamp": pd.to_datetime(timestamp_ms, unit="ms", utc=True),
                            "open": float(kline_raw["open"]),
                            "high": float(kline_raw["high"]),
                            "low": float(kline_raw["low"]),
                            "close": float(kline_raw["close"]),
                            "volume": float(kline_raw["volume"]),
                            # Turnover might not always be present, default to 0.0
                            "turnover": float(kline_raw.get("turnover", 0.0)),
                        }
                        # Put the processed kline dictionary into the thread-safe queue
                        self.kline_queue.put(kline_processed)

                        # Debug log confirming item queued (can be noisy)
                        # ts_str = kline_processed['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                        # log_console(logging.DEBUG, f"WS Queued {'Confirmed ' if is_confirmed else ''}Kline @ {ts_str}", symbol=self.symbol)

                    except (KeyError, ValueError, TypeError) as conv_e:
                        log_console(logging.ERROR, f"WebSocket callback: Error converting kline data: {conv_e}. RawKline: {kline_raw}", symbol=self.symbol)
                        # Continue to next kline in list if one fails

        except Exception as e:
            # Catch any unexpected errors within the callback
            log_console(logging.ERROR, f"Unexpected error in WebSocket kline callback for {self.symbol}: {e}", symbol=self.symbol, exc_info=True)

    def _process_kline_queue(self) -> bool:
        """
        Processes kline data received from the WebSocket queue, updates the internal
        DataFrame (`self.kline_df`), and recalculates indicators.

        Returns:
            True if the DataFrame was updated AND indicators were recalculated successfully.
            False otherwise (e.g., queue empty, processing error, indicator calc failure).
        """
        df_updated = False
        recalculated_ok = False
        processed_count = 0
        # Limit processing per call to prevent blocking the main loop for too long if queue grows large
        max_items_per_cycle = 100

        # Process items from the queue until empty or max count reached
        while not self.kline_queue.empty() and processed_count < max_items_per_cycle:
            try:
                # Get kline data (dict) from the queue without blocking
                kline_dict = self.kline_queue.get_nowait()
                processed_count += 1

                # Create a DataFrame from the single kline dictionary
                # The timestamp should already be a DatetimeIndex with UTC timezone from the callback
                new_row = pd.DataFrame([kline_dict]).set_index("timestamp")
                # Verify index is DatetimeIndex and has UTC timezone
                if not isinstance(new_row.index, pd.DatetimeIndex) or not new_row.index.tz:
                     log_console(logging.ERROR, f"WS Process: Invalid index in queued kline data for {self.symbol}. Skipping item.", symbol=self.symbol)
                     continue # Skip this invalid item

                # --- Update Internal DataFrame ---
                if self.kline_df is None or self.kline_df.empty:
                    # Initialize the DataFrame if it's empty
                    log_console(logging.DEBUG, f"Initializing internal kline_df from first WS kline for {self.symbol}.", symbol=self.symbol)
                    self.kline_df = new_row
                    df_updated = True
                else:
                    # Ensure existing DataFrame index is DatetimeIndex and has UTC timezone before merging
                    if not isinstance(self.kline_df.index, pd.DatetimeIndex):
                         log_console(logging.ERROR, f"Internal kline_df has invalid index type! Resetting DF with new WS kline for {self.symbol}.", symbol=self.symbol)
                         self.kline_df = new_row # Re-initialize with the new row
                         df_updated = True
                         continue # Skip to next item or recalc
                    if not self.kline_df.index.tz:
                         log_console(logging.WARNING, f"Internal kline_df index missing timezone. Localizing existing to UTC for {self.symbol}.", symbol=self.symbol)
                         try:
                             self.kline_df.index = self.kline_df.index.tz_localize('UTC', ambiguous='infer')
                         except TypeError as tz_err: # Handle case where index might already be localized incorrectly
                              log_console(logging.ERROR, f"Error localizing existing index to UTC: {tz_err}. Resetting DF.", symbol=self.symbol)
                              self.kline_df = new_row; df_updated = True; continue

                    # Check if the timestamp from the new kline already exists in the DataFrame index
                    idx_to_check = new_row.index[0]
                    if idx_to_check in self.kline_df.index:
                        # --- Update Existing Candle ---
                        # Use DataFrame.update() for efficiency. This overwrites values in self.kline_df
                        # at the matching index with values from new_row. It preserves existing columns
                        # (like indicators) that are not present in new_row.
                        self.kline_df.update(new_row)
                        # Debug log for update (can be noisy)
                        # log_console(logging.DEBUG, f"WS updated existing kline: {idx_to_check.strftime('%Y-%m-%d %H:%M:%S')}", symbol=self.symbol)
                        df_updated = True
                    else:
                        # --- Append New Candle ---
                        # Use pd.concat to append the new row (as a DataFrame)
                        self.kline_df = pd.concat([self.kline_df, new_row])
                        # Ensure index remains sorted after appending (important for indicators)
                        self.kline_df.sort_index(inplace=True)
                        # Debug log for append
                        # log_console(logging.DEBUG, f"WS appended new kline: {idx_to_check.strftime('%Y-%m-%d %H:%M:%S')}", symbol=self.symbol)

                        # --- Trim DataFrame ---
                        # Maintain a reasonable size to prevent excessive memory usage.
                        # Use the calculated KLINE_LIMIT + a buffer from config.
                        max_rows = self.internal_cfg.get("KLINE_LIMIT", 120) + self.bot_cfg.get("KLINE_LIMIT_BUFFER", 50)
                        if len(self.kline_df) > max_rows:
                            # Keep the most recent 'max_rows' candles
                            self.kline_df = self.kline_df.iloc[-max_rows:]
                            # log_console(logging.DEBUG, f"Trimmed kline_df to {len(self.kline_df)} rows.", symbol=self.symbol)
                        df_updated = True

            except Empty:
                # This is expected when the queue runs out of items
                break # Exit the loop
            except Exception as e:
                log_console(logging.ERROR, f"Error processing kline queue item for {self.symbol}: {e}", symbol=self.symbol, exc_info=True)
                # Continue processing other items if possible

        # --- Recalculate Indicators ---
        # Only recalculate if the DataFrame was actually updated and is valid
        if df_updated and self.kline_df is not None and not self.kline_df.empty:
            log_console(logging.DEBUG, f"Recalculating indicators for {self.symbol} after processing {processed_count} WS item(s)...", symbol=self.symbol)
            # Pass a copy to the calculation function to avoid potential side effects on self.kline_df if calculation fails midway
            # Ensure the DataFrame passed has the correct structure expected by the function
            try:
                calculated_df = calculate_indicators_momentum(self.kline_df.copy(), self.strategy_cfg, self.config)
                # Check if calculation was successful and returned a valid DataFrame
                if calculated_df is not None and not calculated_df.empty:
                    # --- Update Internal DataFrame with Indicators ---
                    # Replace internal DF with the one containing new indicators
                    # Ensure the new DF index is also timezone-aware (UTC) - calculation function should handle this
                    if not calculated_df.index.tz:
                         log_console(logging.WARNING, f"Indicator calculation for {self.symbol} returned DF without timezone. Setting UTC.", symbol=self.symbol)
                         calculated_df.index = calculated_df.index.tz_localize('UTC')

                    self.kline_df = calculated_df
                    # Update the timestamp of the last successful update
                    self.last_kline_update_time = dt.datetime.now(dt.timezone.utc)
                    recalculated_ok = True
                    log_console(logging.DEBUG, f"Indicators recalculated successfully for {self.symbol} after WS update.", symbol=self.symbol)
                else:
                    # Calculation function returned None or empty DataFrame
                    log_console(logging.WARNING, f"Indicator recalculation failed after WS update for {self.symbol}. Stale indicators may be used until next successful update.", symbol=self.symbol)
                    # df_updated remains True, but recalculation failed
                    recalculated_ok = False
            except Exception as calc_e:
                 # Catch unexpected errors during the calculation call itself
                 log_console(logging.ERROR, f"Exception during indicator recalculation after WS update for {self.symbol}: {calc_e}", symbol=self.symbol, exc_info=True)
                 recalculated_ok = False

        elif df_updated: # df_updated is True but kline_df somehow became None or empty
            log_console(logging.ERROR, f"Internal kline_df became empty/None after WS processing for {self.symbol}. Cannot recalculate indicators.", symbol=self.symbol)
            recalculated_ok = False
        # else: df_updated is False (queue was empty), no need to recalculate

        # Return True only if DataFrame was updated AND indicators were recalculated successfully
        return df_updated and recalculated_ok

    def get_ohlcv_rest(self, timeframe: Optional[str] = None, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data via REST API (V5), cleans it, validates it,
        and calculates indicators.

        Args:
            timeframe: The timeframe string (e.g., "15", "60", "D"). Defaults to trader's timeframe.
            limit: Number of klines to fetch. Defaults to calculated KLINE_LIMIT + buffer,
                   capped by API maximum per request.

        Returns:
            DataFrame with OHLCV and calculated indicators, or None if fetch/calculation fails.
            The DataFrame index will be a DatetimeIndex localized to UTC.
        """
        tf_to_fetch = timeframe or self.timeframe
        # Determine limit: use provided limit or default based on internal KLINE_LIMIT + buffer
        default_lim = self.internal_cfg.get("KLINE_LIMIT", 120) + 5 # Add small buffer for fetch
        # Ensure limit is positive and does not exceed the API maximum per request
        lim_to_fetch = max(1, min(limit or default_lim, MAX_KLINE_LIMIT_PER_REQUEST))

        log_console(logging.DEBUG, f"Fetching {lim_to_fetch} klines for {self.symbol} (Timeframe: {tf_to_fetch}) via REST API...", symbol=self.symbol)

        try:
            # V5 API call to get klines
            response = self.session.get_kline(
                category=self.category,
                symbol=self.symbol,
                interval=tf_to_fetch,
                limit=lim_to_fetch
            )

            if response and response.get("retCode") == 0:
                kline_list = response.get("result", {}).get("list", [])
                if kline_list:
                    # V5 Kline format: [timestamp_ms_str, open_str, high_str, low_str, close_str, volume_str, turnover_str]
                    df_raw = pd.DataFrame(
                        kline_list,
                        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
                    )
                    # --- Data Cleaning and Preparation ---
                    # V5 returns newest kline first, reverse to have oldest first for TA libraries
                    df = df_raw.iloc[::-1].reset_index(drop=True)

                    # Convert timestamp first, coerce errors to NaT, set UTC timezone
                    try:
                        df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
                    except Exception as ts_e:
                        log_console(logging.ERROR, f"REST KLine: Failed during timestamp conversion for {self.symbol}: {ts_e}", symbol=self.symbol)
                        return None # Cannot proceed without valid timestamps

                    # Drop rows where timestamp conversion failed (resulted in NaT)
                    initial_len_ts = len(df)
                    df.dropna(subset=["timestamp"], inplace=True)
                    if len(df) < initial_len_ts:
                        log_console(logging.WARNING, f"REST KLine: Dropped {initial_len_ts - len(df)} rows with invalid timestamps for {self.symbol}.", symbol=self.symbol)
                    if df.empty:
                        log_console(logging.WARNING, f"REST KLine: DataFrame empty after timestamp conversion/dropna for {self.symbol}.", symbol=self.symbol)
                        return None
                    # Set the valid timestamp column as the DataFrame index
                    df.set_index("timestamp", inplace=True)

                    # Convert OHLCV columns to numeric, coercing errors to NaN
                    ohlcv_cols = ["open", "high", "low", "close", "volume", "turnover"]
                    for col in ohlcv_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                        else:
                             log_console(logging.WARNING, f"REST KLine: Expected column '{col}' missing in response for {self.symbol}. Filling with NaN.", symbol=self.symbol)
                             df[col] = np.nan # Add column with NaNs if missing

                    # Drop rows with NaNs in essential OHLCV columns needed for most indicators
                    essential_cols = ["open", "high", "low", "close", "volume"]
                    initial_len_ohlcv = len(df)
                    df.dropna(subset=essential_cols, inplace=True)
                    if len(df) < initial_len_ohlcv:
                         log_console(logging.WARNING, f"REST KLine: Dropped {initial_len_ohlcv - len(df)} rows with NaNs in essential OHLCV columns for {self.symbol}.", symbol=self.symbol)
                    if df.empty:
                        log_console(logging.WARNING, f"REST KLine: DataFrame empty after dropping rows with NaN OHLCV data for {self.symbol}.", symbol=self.symbol)
                        return None

                    # --- Length Validation ---
                    # Check if enough valid data remains for indicator calculation
                    # Use the calculated internal KLINE_LIMIT required by the strategy
                    min_len_for_calc = self.internal_cfg.get("KLINE_LIMIT", MIN_KLINE_RECORDS_FOR_CALC)
                    if len(df) < min_len_for_calc:
                        log_console(logging.WARNING, f"REST KLine: Not enough valid data ({len(df)} < {min_len_for_calc}) fetched for {self.symbol} {tf_to_fetch} after cleaning. Cannot calculate reliable indicators.", symbol=self.symbol)
                        return None # Return None if insufficient data for reliable indicators

                    log_console(logging.DEBUG, f"REST Fetch for {self.symbol} successful. Cleaned data shape: {df.shape}", symbol=self.symbol)

                    # --- Calculate Indicators ---
                    # Call the main indicator calculation function on the cleaned DataFrame
                    df_with_indicators = calculate_indicators_momentum(df, self.strategy_cfg, self.config)

                    # Check if indicator calculation succeeded
                    if df_with_indicators is None or df_with_indicators.empty:
                        log_console(logging.WARNING, f"REST KLine: Indicator calculation failed for fetched data ({self.symbol} {tf_to_fetch}).", symbol=self.symbol)
                        return None # Return None if indicator calculation fails
                    else:
                        # Update the last update time upon successful fetch and calculation
                        self.last_kline_update_time = dt.datetime.now(dt.timezone.utc)
                        # Return the DataFrame with indicators (index should be UTC DatetimeIndex)
                        return df_with_indicators

                else: # API returned success code but empty kline_list
                    log_console(logging.WARNING, f"REST KLine fetch returned success code but an empty data list for {self.symbol} {tf_to_fetch}.", symbol=self.symbol)
                    return None
            else: # API call failed (retCode != 0 or invalid response)
                error_msg = response.get('retMsg', 'N/A') if response else "Empty/invalid response"
                error_code = response.get('retCode', 'N/A')
                log_console(logging.WARNING, f"REST KLine fetch API call failed for {self.symbol} {tf_to_fetch}: {error_msg} (Code: {error_code})", symbol=self.symbol)
                # Check for specific errors like invalid symbol/timeframe
                if error_code == 10001 or "invalid symbol" in error_msg.lower() or "invalid interval" in error_msg.lower():
                     log_console(logging.ERROR, f"----> Check if symbol '{self.symbol}' and timeframe '{tf_to_fetch}' are valid for category '{self.category}'.", symbol=self.symbol)
                # Handle rate limit errors (e.g., 10006)
                elif error_code == 10006:
                     log_console(logging.WARNING, f"REST KLine fetch rate limited for {self.symbol}. Consider increasing sleep interval or reducing symbols.", symbol=self.symbol)
                return None
        except Exception as e:
            log_console(logging.ERROR, f"Exception during REST KLine fetch or processing for {self.symbol}: {e}", symbol=self.symbol, exc_info=True)
            return None

    def get_ohlcv(self) -> Optional[pd.DataFrame]:
        """
        Provides the latest OHLCV data frame with calculated indicators.
        Prioritizes processing WebSocket updates if enabled and active.
        Falls back to REST API fetch if WebSocket is disabled, inactive,
        data is stale, or initial population is needed.

        Returns:
            A *copy* of the internal DataFrame (`self.kline_df`) with indicators,
            or None if data is unavailable or couldn't be obtained/calculated.
            The returned DataFrame index is a UTC DatetimeIndex.
        """
        ws_processed_and_recalculated = False
        ws_enabled_for_symbol = self.use_websocket

        # 1. Attempt to process WebSocket queue if WS is enabled for this symbol
        if ws_enabled_for_symbol:
            # This function returns True only if DF was updated AND indicators recalculated successfully
            ws_processed_and_recalculated = self._process_kline_queue()

        # 2. Determine if a REST fetch is necessary
        needs_rest_fetch = False
        now_utc = dt.datetime.now(dt.timezone.utc)
        staleness_limit = self.bot_cfg.get("KLINE_STALENESS_SECONDS", 300)

        if self.kline_df is None or self.kline_df.empty:
            # Need REST if internal DataFrame is completely missing
            needs_rest_fetch = True
            log_console(logging.DEBUG, f"Internal kline_df is empty for {self.symbol}, attempting REST fetch.", symbol=self.symbol)
        elif not ws_enabled_for_symbol:
            # If WS is disabled, rely solely on REST. Fetch if data is stale.
            if self.last_kline_update_time is None or \
               (now_utc - self.last_kline_update_time).total_seconds() > staleness_limit:
                needs_rest_fetch = True
                log_console(logging.DEBUG, f"WS disabled and data stale (> {staleness_limit}s) for {self.symbol}, using REST.", symbol=self.symbol)
            else:
                 log_console(logging.DEBUG, f"WS disabled for {self.symbol}, using recent REST data.", symbol=self.symbol)
        elif ws_enabled_for_symbol and not ws_processed_and_recalculated:
            # WS enabled, but processing the queue failed or didn't happen (e.g., empty queue).
            # Check if the existing data (likely from previous REST or WS update) is now stale.
            if self.last_kline_update_time is None or \
               (now_utc - self.last_kline_update_time).total_seconds() > staleness_limit:
                needs_rest_fetch = True
                log_console(logging.WARNING, f"WS queue processing did not yield update or data stale (> {staleness_limit}s) for {self.symbol}. Attempting REST fallback.", symbol=self.symbol)
            # else: WS processing didn't update, but existing data is not yet stale, use it.

        # 3. Perform REST fetch if needed
        if needs_rest_fetch:
            log_console(logging.DEBUG, f"Executing REST fetch for {self.symbol}...", symbol=self.symbol)
            rest_df_with_indicators = self.get_ohlcv_rest() # Uses trader's default timeframe/limit settings

            # Update internal state only if REST fetch was successful
            if rest_df_with_indicators is not None and not rest_df_with_indicators.empty:
                # Ensure the fetched DataFrame index is UTC (should be handled by get_ohlcv_rest)
                if not rest_df_with_indicators.index.tz:
                     log_console(logging.ERROR, f"REST fetch for {self.symbol} returned DF without timezone! Attempting to set UTC.", symbol=self.symbol)
                     rest_df_with_indicators.index = rest_df_with_indicators.index.tz_localize('UTC')

                self.kline_df = rest_df_with_indicators
                # last_kline_update_time is set within get_ohlcv_rest on success
                log_console(logging.DEBUG, f"Updated internal kline_df successfully via REST API for {self.symbol}.", symbol=self.symbol)
            else:
                log_console(logging.WARNING, f"REST API fetch fallback failed for {self.symbol}. Kline data may be missing or stale.", symbol=self.symbol)
                # Do not return here; proceed to return the potentially stale kline_df if it exists.

        # 4. Return a copy of the current internal state
        # Return None if the DataFrame is still None or empty after all attempts
        if self.kline_df is not None and not self.kline_df.empty:
            # Return a deep copy to prevent external modifications affecting internal state
            return self.kline_df.copy()
        else:
            log_console(logging.DEBUG, f"No valid kline data available for {self.symbol} after WS/REST attempts.", symbol=self.symbol)
            return None

    def get_position(self, log_error: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetches current position details for the symbol using V5 API.
        Returns simulated position state in dry run mode.

        Args:
            log_error: If True, log API errors encountered during the fetch.

        Returns:
            A dictionary containing position details if an active position exists,
            or None if no position is open or an error occurred.
            The dictionary structure is standardized for internal use.
            Returns a deepcopy of the simulated position in dry run mode.
        """
        # --- Dry Run Simulation ---
        if self.dry_run:
            # Return a deep copy to prevent modification of the internal state from outside
            # Return None if no simulated trade is currently stored
            return deepcopy(self.current_trade) if self.current_trade else None

        # --- Real Position Fetch (V5 API) ---
        try:
            # Use V5 get_positions endpoint
            response = self.session.get_positions(category=self.category, symbol=self.symbol)

            if response and response.get("retCode") == 0:
                position_list = response.get("result", {}).get("list", [])
                # The list might contain multiple entries (e.g., for different modes or hedges),
                # but typically for linear/unified, there's one entry per symbol with size > 0.
                for pos_data in position_list:
                    if not isinstance(pos_data, dict): continue # Skip invalid entries

                    # Check if the position entry matches the symbol we are interested in
                    if pos_data.get("symbol") == self.symbol:
                        try:
                            pos_side = pos_data.get("side") # V5: "Buy", "Sell", or "None"
                            pos_size_str = pos_data.get("size", "0")
                            pos_size = float(pos_size_str)

                            # --- Check if Position is Active ---
                            # Consider a position active if side is Buy/Sell and size is significant
                            # Use a small threshold relative to min_order_qty or step size to ignore dust
                            min_significant_qty = max(FLOAT_COMPARISON_TOLERANCE, self.min_order_qty * 0.01, self.qty_step_float * 0.1)

                            if pos_side in ["Buy", "Sell"] and abs(pos_size) >= min_significant_qty:
                                # --- Standardize Position Data ---
                                # Convert key fields to appropriate types and store consistently
                                position_standardized = {
                                    'symbol': self.symbol,
                                    'side': pos_side, # "Buy" or "Sell"
                                    'size': pos_size, # Float
                                    'avgPrice': float(pos_data.get('avgPrice', '0')), # Float
                                    'leverage': pos_data.get('leverage', str(self.leverage)), # Store as string (API often returns string)
                                    'liqPrice': float(pos_data.get('liqPrice', '0')), # Float
                                    'markPrice': float(pos_data.get('markPrice', '0')), # Float
                                    'unrealisedPnl': float(pos_data.get('unrealisedPnl', '0')), # Float
                                    'positionValue': float(pos_data.get('positionValue', '0')), # Float
                                    # V5 uses stopLoss / takeProfit fields directly in position data
                                    'stopLoss': pos_data.get('stopLoss', ''), # String price or empty
                                    'takeProfit': pos_data.get('takeProfit', ''), # String price or empty
                                    'positionIdx': pos_data.get('positionIdx', 0), # 0 for one-way, 1/2 for hedge mode Buy/Sell
                                    'createdTime': pos_data.get('createdTime', ''), # Timestamp string (ms)
                                    'updatedTime': pos_data.get('updatedTime', ''), # Timestamp string (ms)
                                    # Add internal entry time if possible
                                    'entry_time': None # Initialize
                                }

                                # --- Attempt to Populate Entry Time ---
                                # 1. Prioritize internal state if it matches symbol/side/approx size
                                if self.current_trade and \
                                   self.current_trade.get("symbol") == self.symbol and \
                                   self.current_trade.get("side") == pos_side and \
                                   math.isclose(abs(self.current_trade.get("size", 0)), abs(pos_size), rel_tol=0.01): # Allow 1% tolerance
                                    position_standardized['entry_time'] = self.current_trade.get('entry_time')
                                # 2. Fallback to API createdTime (convert ms string to datetime UTC)
                                elif position_standardized['createdTime']:
                                    try:
                                        ts_ms = int(position_standardized['createdTime'])
                                        position_standardized['entry_time'] = pd.to_datetime(ts_ms, unit='ms', utc=True)
                                    except (ValueError, TypeError, KeyError):
                                        log_console(logging.DEBUG, f"Could not parse createdTime '{position_standardized['createdTime']}' for position entry time.", symbol=self.symbol)
                                        pass # Ignore parsing errors, entry_time remains None

                                log_console(logging.DEBUG, f"Active position found for {self.symbol}: Side={pos_side}, Size={pos_size}", symbol=self.symbol)
                                # Return the first significant position found for the symbol
                                return position_standardized

                        except (ValueError, TypeError, KeyError) as conv_e:
                            log_console(logging.ERROR, f"Position fetch: Error converting fields for {self.symbol}: {conv_e}. PosData: {pos_data}", symbol=self.symbol)
                            continue # Try next item in list if conversion fails for one entry

                # If loop finishes without finding an active position for the symbol
                log_console(logging.DEBUG, f"No active position found for {self.symbol}.", symbol=self.symbol)
                return None
            else: # API call failed
                if log_error:
                    error_msg = response.get('retMsg', 'N/A') if response else "Empty/invalid response"
                    error_code = response.get('retCode', 'N/A')
                    log_console(logging.ERROR, f"Position fetch API call failed for {self.symbol}: {error_msg} (Code: {error_code})", symbol=self.symbol)
                    # Handle specific errors if needed (e.g., auth errors 10003/10004)
                    if error_code in [10003, 10004]: log_console(logging.CRITICAL, f"Position fetch failed due to AUTH error for {self.symbol}. Check API keys/permissions.", symbol=self.symbol)
                return None
        except Exception as e:
            if log_error:
                log_console(logging.ERROR, f"Exception during position fetch for {self.symbol}: {e}", symbol=self.symbol, exc_info=True)
            return None

    def confirm_order(self, order_id: str, expected_qty: float) -> Tuple[float, Optional[float]]:
        """
        Confirms order status by polling order history (V5 API). Retries until a
        terminal status (Filled, Cancelled, Rejected, etc.) is reached or timeout.

        Args:
            order_id: The ID of the order to confirm.
            expected_qty: The quantity expected to be filled (for logging/comparison).

        Returns:
            Tuple[float, Optional[float]]:
                - Actual cumulative filled quantity (float).
                - Average fill price (float) if available and filled > 0, otherwise None.
            Returns (0.0, None) if order ID is invalid or confirmation fails critically.
        """
        if not order_id:
            log_console(logging.ERROR, "Confirm Order: Invalid or empty Order ID provided.", symbol=self.symbol)
            return 0.0, None

        # --- Dry Run Simulation ---
        if self.dry_run:
            log_console(logging.INFO, f"[DRY RUN] Simulating confirmation for order {order_id} (Expected Qty: {expected_qty})", symbol=self.symbol)
            # Try to get a realistic price from the latest kline data for simulation
            sim_entry_price: Optional[float] = None
            if self.kline_df is not None and not self.kline_df.empty:
                try:
                    # Use the close of the latest available candle
                    sim_entry_price = float(self.kline_df['close'].iloc[-1])
                    # Basic validation of the simulated price
                    if sim_entry_price <= FLOAT_COMPARISON_TOLERANCE: sim_entry_price = None
                except (IndexError, ValueError, TypeError): pass # Ignore errors getting price

            # Fallback if price unavailable or invalid
            if sim_entry_price is None:
                sim_entry_price = 1.0 # Use a dummy price
                log_console(logging.WARNING, f"[DRY RUN] Cannot estimate realistic entry price for {order_id}, using dummy price: {sim_entry_price}.", symbol=self.symbol)

            log_console(logging.DEBUG, f"[DRY RUN] Using simulated avg fill price: {sim_entry_price:.{self.price_precision}f} for order {order_id}", symbol=self.symbol)
            # Assume full fill in dry run for simplicity
            return expected_qty, sim_entry_price

        # --- Real Order Confirmation Loop ---
        last_filled_qty = 0.0
        last_avg_price = None
        last_status = "Unknown"
        terminal_statuses = ["Filled", "Rejected", "Cancelled", "PartiallyFilledCanceled", "Deactivated", "Expired"]
        # V5 Statuses to watch: Created, New, PartiallyFilled, Filled, Cancelled, Rejected, Untriggered (for conditional), Triggered, Deactivated, Active (conditional orders), PartiallyFilledCanceled, Expired

        for attempt in range(self.order_confirm_retries + 1):
            # Delay between attempts (except the first)
            if attempt > 0:
                time.sleep(self.order_confirm_delay)

            log_console(logging.DEBUG, f"Confirming order {order_id} (Attempt {attempt + 1}/{self.order_confirm_retries + 1})...", symbol=self.symbol)

            try:
                # Use get_order_history for V5 - generally more reliable for final status than get_open_orders
                response = self.session.get_order_history(
                    category=self.category,
                    # symbol=self.symbol, # Optional: May speed up lookup but not strictly needed with orderId
                    orderId=order_id,
                    limit=1 # We only need the specific order details
                )

                if response and response.get("retCode") == 0:
                    order_list = response.get("result", {}).get("list", [])
                    if order_list and isinstance(order_list[0], dict):
                        order = order_list[0]
                        status = order.get("orderStatus")
                        last_status = status # Update last known status

                        # Safely extract filled quantity and average price
                        filled_qty = 0.0
                        avg_price = None
                        try:
                            filled_qty_str = order.get("cumExecQty", "0")
                            filled_qty = float(filled_qty_str) if filled_qty_str else 0.0
                            # Only consider avgPrice if the order has actually filled something
                            if filled_qty > FLOAT_COMPARISON_TOLERANCE:
                                avg_price_str = order.get("avgPrice", "0")
                                # Ensure avgPrice is a valid positive number string
                                if avg_price_str:
                                    try:
                                        temp_avg_price = float(avg_price_str)
                                        if temp_avg_price > FLOAT_COMPARISON_TOLERANCE:
                                            avg_price = temp_avg_price
                                    except (ValueError, TypeError): pass # Ignore conversion error for avgPrice

                            # Update last known values
                            last_filled_qty = filled_qty
                            if avg_price is not None: last_avg_price = avg_price

                        except (ValueError, TypeError, KeyError) as e:
                            log_console(logging.ERROR, f"Confirm Order {order_id}: Error parsing qty/price from history: {e}. Data: {order}", symbol=self.symbol)
                            # Continue checking status if possible

                        # --- Check Order Status ---
                        if status == "Filled":
                            log_console(logging.INFO, f"Order {order_id} confirmed: Fully Filled. Qty={filled_qty:.{self.qty_precision}f}, AvgPrice={avg_price or 'N/A'}", symbol=self.symbol)
                            return filled_qty, avg_price # Success, return confirmed values
                        elif status == "PartiallyFilled":
                            log_console(logging.INFO, f"Order {order_id} status '{status}'. Filled: {filled_qty:.{self.qty_precision}f}/{order.get('qty', '?')}. Continuing check...", symbol=self.symbol)
                            # Continue loop to wait for full fill or cancellation
                        elif status in terminal_statuses:
                            # Order reached a final state other than Filled
                            log_console(logging.WARNING, f"Order {order_id} reached terminal status: '{status}'. Final Filled Qty: {filled_qty:.{self.qty_precision}f}", symbol=self.symbol)
                            # Return the quantity that was filled before cancellation/rejection
                            return filled_qty, last_avg_price # May be 0.0
                        elif status in ["New", "Untriggered", "Active", "Created", "Triggered"]:
                            # Order is still pending or active, wait and retry
                            log_console(logging.DEBUG, f"Order {order_id} status '{status}'. Waiting...", symbol=self.symbol)
                        else:
                            # Log unexpected statuses for investigation
                            log_console(logging.WARNING, f"Order {order_id} encountered unhandled status: '{status}'. Continuing check.", symbol=self.symbol)

                    else: # Order not found in history yet
                        log_console(logging.DEBUG, f"Order {order_id} not found in history yet (Attempt {attempt + 1}). Might still be processing or submitted.", symbol=self.symbol)
                        # Consider querying open orders as a secondary check if history is slow? (Adds complexity)
                        # response_open = self.session.get_open_orders(category=self.category, orderId=order_id) ... handle response ...
                else: # API call to get order history failed
                    error_msg = response.get('retMsg', 'N/A') if response else "Empty/invalid response"
                    error_code = response.get('retCode', 'N/A')
                    log_console(logging.WARNING, f"Order history fetch failed for {order_id} (Attempt {attempt + 1}): {error_msg} (Code: {error_code}).", symbol=self.symbol)
                    # Handle specific errors like rate limits (10006) if necessary
                    if error_code == 10006: time.sleep(self.order_confirm_delay * 2) # Longer delay on rate limit

            except Exception as e:
                log_console(logging.ERROR, f"Exception during order confirmation attempt {attempt + 1} for {order_id}: {e}", symbol=self.symbol, exc_info=True)
                # Continue to next attempt if retries remain

        # --- Confirmation Timeout ---
        # Loop finished without reaching a confirmed terminal state
        log_console(logging.ERROR, f"Order {order_id} confirmation TIMEOUT after {self.order_confirm_retries + 1} attempts. Last Status: '{last_status}'.", symbol=self.symbol)
        log_console(logging.ERROR, f"Order {order_id} - Last Known State: FilledQty={last_filled_qty:.{self.qty_precision}f}, AvgPrice={last_avg_price or 'N/A'}. Manual check strongly recommended!", symbol=self.symbol)
        # Return the last known filled quantity and price, even if incomplete or zero, as the best available information
        return last_filled_qty, last_avg_price

    def place_order(self, side: str, qty: float, stop_loss_price: Optional[float] = None,
                    take_profit_price: Optional[float] = None, order_link_id: Optional[str] = None) -> Optional[str]:
        """
        Places a market order with optional Stop Loss and Take Profit using V5 API.
        Confirms execution and updates internal state.

        Args:
            side: "Buy" or "Sell".
            qty: Order quantity (must be positive and meet instrument minimums).
            stop_loss_price: Optional SL price.
            take_profit_price: Optional TP price.
            order_link_id: Optional unique client order ID. If None, one is generated.

        Returns:
            The Bybit order ID (str) if the order is successfully placed and confirmed
            as substantially filled, otherwise None.
        """
        # --- Input Validation ---
        if side not in ["Buy", "Sell"]:
            log_console(logging.ERROR, f"Invalid order side specified: {side}", symbol=self.symbol)
            return None
        if not isinstance(qty, (float, int)) or qty <= FLOAT_COMPARISON_TOLERANCE:
            log_console(logging.ERROR, f"Order quantity must be a positive number, got: {qty}", symbol=self.symbol)
            return None
        # Check against minimum order quantity with a small tolerance
        min_qty_tolerance = self.qty_step_float * 0.01
        if qty < self.min_order_qty - min_qty_tolerance:
            log_console(logging.ERROR, f"Order quantity {qty:.{self.qty_precision}f} is below minimum required {self.min_order_qty}", symbol=self.symbol)
            return None

        # --- Format Parameters for API ---
        # Format quantity string according to instrument precision
        try:
            qty_str = f"{qty:.{self.qty_precision}f}"
        except Exception as fmt_e:
             log_console(logging.ERROR, f"Failed to format quantity {qty} to precision {self.qty_precision}: {fmt_e}", symbol=self.symbol)
             return None

        # Format SL/TP price strings if provided and valid
        sl_str: Optional[str] = None
        if stop_loss_price is not None:
            if not isinstance(stop_loss_price, (float, int)) or stop_loss_price <= FLOAT_COMPARISON_TOLERANCE:
                log_console(logging.WARNING, f"Invalid SL price provided ({stop_loss_price}). Stop Loss will not be set.", symbol=self.symbol)
            else:
                try: sl_str = f"{stop_loss_price:.{self.price_precision}f}"
                except Exception as fmt_e: log_console(logging.ERROR, f"Failed to format SL price {stop_loss_price}: {fmt_e}", symbol=self.symbol)

        tp_str: Optional[str] = None
        if take_profit_price is not None:
            if not isinstance(take_profit_price, (float, int)) or take_profit_price <= FLOAT_COMPARISON_TOLERANCE:
                log_console(logging.WARNING, f"Invalid TP price provided ({take_profit_price}). Take Profit will not be set.", symbol=self.symbol)
            else:
                 try: tp_str = f"{take_profit_price:.{self.price_precision}f}"
                 except Exception as fmt_e: log_console(logging.ERROR, f"Failed to format TP price {take_profit_price}: {fmt_e}", symbol=self.symbol)


        # --- Log Action Intent ---
        log_prefix = f"{Fore.YELLOW+Style.BRIGHT}[DRY RUN]{Style.RESET_ALL} " if self.dry_run else ""
        side_color = Fore.GREEN if side == "Buy" else Fore.RED
        log_msg_console = f"{log_prefix}{side_color}{Style.BRIGHT}{side.upper()} MARKET ORDER:{Style.RESET_ALL} Qty={qty_str}"
        if sl_str: log_msg_console += f", SL={sl_str}"
        if tp_str: log_msg_console += f", TP={tp_str}"
        action_prefix = f"{Fore.MAGENTA+Style.BRIGHT}ACTION:{Style.RESET_ALL} [{self.symbol}] "
        print(action_prefix + log_msg_console) # Print action immediately to console
        # Log to file without dry run prefix
        log_msg_file = log_msg_console.replace(log_prefix, "").strip()
        log_console(logging.INFO, log_msg_file, symbol=self.symbol)

        # --- Dry Run Simulation ---
        if self.dry_run:
            # Generate a unique simulated order ID
            order_id = f"dryrun_{self.symbol}_{side.lower()}_{int(time.time()*1000)}"
            # Estimate entry price from klines for simulation realism
            sim_entry_price = 0.0
            if self.kline_df is not None and not self.kline_df.empty:
                 try: sim_entry_price = float(self.kline_df['close'].iloc[-1])
                 except (IndexError, ValueError, TypeError): pass
            # Fallback to a dummy price if estimation fails
            if sim_entry_price <= FLOAT_COMPARISON_TOLERANCE: sim_entry_price = 1.0

            # Update internal state to simulate the open position
            self.current_trade = {
                "orderId": order_id, "symbol": self.symbol, "side": side,
                "size": qty, # Use requested quantity for simulation
                "avgPrice": sim_entry_price, # Use estimated/dummy price
                "entry_time": dt.datetime.now(dt.timezone.utc), # Record simulation time
                "stopLoss": sl_str, # Store intended SL/TP
                "takeProfit": tp_str,
                "leverage": str(self.leverage), # Store leverage used
                "orderStatus": "Filled", # Simulate filled status immediately
                "positionValue": f"{qty * sim_entry_price:.4f}", # Add estimated value
                "liqPrice": 0.0, # Placeholder
                "markPrice": sim_entry_price, # Placeholder
                "unrealisedPnl": 0.0, # Placeholder
                "positionIdx": 0, # Assuming one-way mode
                "createdTime": str(int(time.time() * 1000)), # Placeholder
                "updatedTime": str(int(time.time() * 1000)), # Placeholder
            }
            log_console(logging.INFO, f"[DRY RUN] Simulated order {order_id} placed. Simulated Entry Price: {sim_entry_price:.{self.price_precision}f}", symbol=self.symbol)
            return order_id # Return the simulated order ID

        # --- Real Order Placement (V5 API) ---
        try:
            # Prepare parameters for the V5 place_order call
            order_params: Dict[str, Any] = {
                "category": self.category, "symbol": self.symbol,
                "side": side, "orderType": "Market", "qty": qty_str,
            }
            # Add SL/TP parameters if they were successfully formatted
            if sl_str:
                order_params["stopLoss"] = sl_str
                # V5 trigger price type: LastPrice, MarkPrice, IndexPrice
                order_params["slTriggerBy"] = self.strategy_cfg.get("SL_TRIGGER_BY", "LastPrice")
            if tp_str:
                order_params["takeProfit"] = tp_str
                order_params["tpTriggerBy"] = self.strategy_cfg.get("TP_TRIGGER_BY", "LastPrice")

            # Set TPSL mode if either SL or TP is set. V5 often requires this for market orders with SL/TP.
            # Default to "Full" position TP/SL. "Partial" allows partial TP/SL setting (more complex).
            if sl_str or tp_str:
                order_params["tpslMode"] = self.strategy_cfg.get("TPSL_MODE", "Full")

            # Generate a unique client order ID if one wasn't provided
            # Helps prevent duplicate orders on retries or network issues
            order_params["orderLinkId"] = order_link_id or f"momscan_{self.symbol}_{side[:1]}_{int(time.time()*1000)}"

            # --- Place the Order ---
            response = self.session.place_order(**order_params)

            # --- Process Placement Response ---
            if response and response.get("retCode") == 0:
                order_id = response.get("result", {}).get("orderId")
                if order_id:
                    log_console(logging.INFO, f"Order {order_id} submitted successfully. Confirming execution...", symbol=self.symbol)
                    # --- Confirm Order Execution ---
                    # This polls the order history to get actual fill details
                    filled_qty, avg_price = self.confirm_order(order_id, qty)

                    # --- Validate Fill Quantity ---
                    # Check if the actual filled quantity is acceptably close to the requested quantity
                    # Define a minimum acceptable fill ratio (e.g., 95%)
                    min_fill_ratio = 0.95
                    # Also consider tolerance based on quantity step size
                    fill_tolerance = self.qty_step_float * 0.5
                    required_fill = max(qty * min_fill_ratio, qty - fill_tolerance) # Need at least this much filled

                    if filled_qty >= required_fill - FLOAT_COMPARISON_TOLERANCE:
                        log_console(logging.INFO, f"Order {order_id} confirmed filled successfully. Actual Qty={filled_qty:.{self.qty_precision}f}, Avg Price={avg_price or 'N/A'}", symbol=self.symbol)
                        # --- Update Internal State with Actual Fill Data ---
                        self.current_trade = {
                            "orderId": order_id, "symbol": self.symbol, "side": side,
                            "size": filled_qty, # Use ACTUAL filled quantity
                            "avgPrice": avg_price if avg_price else 0.0, # Use ACTUAL average price (or 0 if unavailable)
                            "entry_time": dt.datetime.now(dt.timezone.utc), # Record entry time upon confirmation
                            "stopLoss": sl_str, # Store intended SL/TP for reference
                            "takeProfit": tp_str,
                            "leverage": str(self.leverage),
                            "orderStatus": "Filled", # Mark as filled internally
                            # Other fields can be populated by get_position later if needed
                            "positionValue": f"{filled_qty * (avg_price if avg_price else 0.0):.4f}",
                            "liqPrice": 0.0, "markPrice": 0.0, "unrealisedPnl": 0.0, "positionIdx": 0,
                            "createdTime": "", "updatedTime": ""
                        }
                        return order_id # Return the confirmed order ID
                    else:
                        # Fill was insufficient or failed
                        log_console(logging.ERROR, f"Order {order_id} fill confirmation failed or insufficient. Requested Qty={qty:.{self.qty_precision}f}, Filled Qty={filled_qty:.{self.qty_precision}f} (Required >= {required_fill:.{self.qty_precision}f}).", symbol=self.symbol)
                        self.current_trade = None # Clear internal state as the trade didn't establish correctly
                        # --- Optional: Attempt to Cancel ---
                        # If an order was partially filled but insufficiently, you might want to cancel
                        # the remainder to avoid an unintended small position. This is risky if the
                        # cancellation fails or races with further execution. Manual check is safer.
                        # log_console(logging.WARNING, f"Attempting to cancel potentially partially filled order {order_id}...", symbol=symbol)
                        # try:
                        #     cancel_resp = self.session.cancel_order(category=self.category, symbol=self.symbol, orderId=order_id)
                        #     log_console(logging.INFO, f"Cancel response for {order_id}: {cancel_resp}", symbol=symbol)
                        # except Exception as cancel_e:
                        #     log_console(logging.ERROR, f"Error cancelling order {order_id}: {cancel_e}", symbol=symbol)
                        return None # Indicate failure
                else:
                    # API returned success code but no order ID - should be rare
                    log_console(logging.ERROR, f"Order placement API call succeeded (retCode 0) but no Order ID was returned. Response: {response}", symbol=self.symbol)
                    return None
            else: # API call to place order failed (retCode != 0)
                error_msg = response.get('retMsg', 'N/A') if response else "Empty/invalid response"
                error_code = response.get('retCode', 'N/A')
                log_console(logging.ERROR, f"Order placement API call failed: {error_msg} (Code: {error_code})", symbol=self.symbol)
                # --- Provide Context for Common V5 Order Errors ---
                if error_code == 110007: log_console(logging.CRITICAL, "ORDER REJECTED: INSUFFICIENT BALANCE! Check available funds.", symbol=self.symbol)
                elif error_code in [110013, 110014, 110069]: log_console(logging.ERROR,"ORDER REJECTED: Quantity issue (min/max/step). Check instrument info and calculated size.", symbol=self.symbol)
                elif error_code == 110017: log_console(logging.ERROR,"ORDER REJECTED: Risk control triggered by exchange (e.g., large price deviation). Market might be volatile.", symbol=self.symbol)
                elif error_code == 110040: log_console(logging.ERROR,"ORDER REJECTED: Account risk limit exceeded (e.g., max position size allowed).", symbol=self.symbol)
                elif error_code == 110073: log_console(logging.ERROR,f"ORDER REJECTED: TP/SL price invalid (e.g., too close to market, wrong side). Check SL={sl_str}, TP={tp_str}", symbol=self.symbol)
                elif error_code == 110035: log_console(logging.ERROR, f"ORDER REJECTED: TP/SL requires tpslMode parameter for market orders in V5.", symbol=self.symbol)
                elif "reduce-only" in error_msg.lower(): log_console(logging.ERROR, f"ORDER REJECTED: Reduce-only conflict? Unexpected open position might exist.", symbol=self.symbol)
                # Add more specific error codes as encountered
                return None # Indicate failure
        except Exception as e:
            log_console(logging.ERROR, f"Exception occurred during order placement for {self.symbol}: {e}", symbol=self.symbol, exc_info=True)
            return None

    def close_position(self, position_data: Dict[str, Any], exit_reason: str = "Signal", exit_price_est: Optional[float] = None) -> bool:
        """
        Closes an existing position using a market order with reduceOnly=True (V5 API).
        Logs trade metrics upon successful confirmed closure.

        Args:
            position_data: Dictionary containing details of the position to close (obtained from `get_position`).
            exit_reason: String describing why the position is being closed (for logging).
            exit_price_est: Optional estimated exit price (e.g., current market price) used for
                            logging metrics if the actual average exit price cannot be confirmed.

        Returns:
            True if the position was successfully closed (or confirmed already closed), False otherwise.
        """
        # --- Validate Input Position Data ---
        if not isinstance(position_data, dict):
            log_console(logging.WARNING, "Close attempt failed: Invalid position data provided (not a dict).", symbol=self.symbol)
            return False

        # Extract necessary info safely using .get() with defaults
        symbol_from_pos = position_data.get("symbol")
        side = position_data.get("side") # "Buy" or "Sell"
        size = position_data.get("size") # Should be float from get_position
        entry_price = position_data.get("avgPrice") # Should be float
        entry_time = position_data.get("entry_time") # datetime or None
        leverage = position_data.get("leverage", str(self.leverage)) # str

        # Verify symbol matches the trader instance
        if symbol_from_pos != self.symbol:
             log_console(logging.ERROR, f"Position data symbol mismatch ({symbol_from_pos} != {self.symbol}). Cannot close using this trader instance.", symbol=self.symbol)
             return False

        # Validate essential numeric and side data
        if not (side in ["Buy", "Sell"] and isinstance(size, float) and isinstance(entry_price, float) and entry_price > FLOAT_COMPARISON_TOLERANCE):
            log_console(logging.ERROR, f"Invalid position data types for closing {self.symbol}: Side={side}({type(side)}), Size={size}({type(size)}), Entry={entry_price}({type(entry_price)}). Cannot close.", symbol=self.symbol)
            # Clear potentially inconsistent internal state if it matches the symbol
            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                log_console(logging.WARNING,f"Clearing potentially inconsistent internal trade state for {self.symbol} due to invalid position data.", symbol=self.symbol)
                self.current_trade = None
            return False # Cannot proceed with invalid data

        abs_size = abs(size)
        # Check if position size is negligible (dust) - skip closing attempt if so
        # Use a threshold relative to min qty or step size
        min_significant_qty_close = max(FLOAT_COMPARISON_TOLERANCE, self.min_order_qty * 0.01, self.qty_step_float * 0.1)
        if abs_size < min_significant_qty_close:
            log_console(logging.INFO, f"Position size {abs_size:.{self.qty_precision}f} for {self.symbol} is negligible (below {min_significant_qty_close:.{self.qty_precision}f}). Assuming already closed or dust.", symbol=self.symbol)
            # Ensure internal state is cleared if it reflected this tiny position
            if self.current_trade and self.current_trade.get("symbol") == self.symbol: self.current_trade = None
            # Update last exit time to potentially trigger cooldown if configured
            self.last_exit_time = dt.datetime.now(dt.timezone.utc)
            return True # Treat as successful closure

        # Determine the side of the closing order (opposite of position side)
        close_side = "Sell" if side == "Buy" else "Buy"
        # Format quantity string for the API call
        try:
             qty_str = f"{abs_size:.{self.qty_precision}f}"
        except Exception as fmt_e:
             log_console(logging.ERROR, f"Failed to format close quantity {abs_size}: {fmt_e}", symbol=self.symbol)
             return False


        # --- Log Action Intent ---
        log_prefix = f"{Fore.YELLOW+Style.BRIGHT}[DRY RUN]{Style.RESET_ALL} " if self.dry_run else ""
        side_color = Fore.RED if close_side == "Sell" else Fore.GREEN
        log_msg_console = (f"{log_prefix}{side_color}{Style.BRIGHT}CLOSE {side.upper()} POSITION ({close_side} MARKET):"
                           f"{Style.RESET_ALL} Qty={qty_str} | Reason: {exit_reason}")
        action_prefix = f"{Fore.MAGENTA+Style.BRIGHT}ACTION:{Style.RESET_ALL} [{self.symbol}] "
        print(action_prefix + log_msg_console) # Print action immediately
        log_msg_file = log_msg_console.replace(log_prefix, "").strip()
        log_console(logging.INFO, log_msg_file, symbol=self.symbol)


        # --- Dry Run Simulation ---
        if self.dry_run:
            # Check if there's a simulated trade open to close
            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                # Use provided estimate or try to get latest price from klines
                sim_exit_price = exit_price_est
                if sim_exit_price is None or sim_exit_price <= FLOAT_COMPARISON_TOLERANCE:
                    if self.kline_df is not None and not self.kline_df.empty:
                        try: sim_exit_price = float(self.kline_df['close'].iloc[-1])
                        except (IndexError, ValueError, TypeError): pass
                # Fallback to entry price if still no valid exit price (P&L will be inaccurate)
                sim_entry_price_fallback = self.current_trade.get("avgPrice", 1.0)
                if sim_exit_price is None or sim_exit_price <= FLOAT_COMPARISON_TOLERANCE:
                    sim_exit_price = sim_entry_price_fallback if sim_entry_price_fallback > FLOAT_COMPARISON_TOLERANCE else 1.0
                    log_console(logging.WARNING, f"[DRY RUN] Cannot estimate realistic exit price for {self.symbol}, using entry/fallback price ({sim_exit_price:.{self.price_precision}f}) for metrics. P&L may be inaccurate.", symbol=self.symbol)

                log_console(logging.INFO, f"[DRY RUN] Simulating close for {self.symbol} at estimated price: {sim_exit_price:.{self.price_precision}f}", symbol=self.symbol)
                # Get details from the simulated trade state for metrics logging
                entry_time_sim = self.current_trade.get("entry_time") or (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)) # Use stored or fallback entry time
                sim_leverage = self.current_trade.get("leverage", str(self.leverage))
                sim_entry_price = self.current_trade.get("avgPrice", sim_exit_price) # Use stored entry price
                sim_qty = self.current_trade.get("size", abs_size) # Use stored size
                original_side = self.current_trade.get("side", side) # Use stored side

                # --- Log Simulated Trade Metrics ---
                self.metrics.add_trade(
                    symbol=self.symbol, entry_time=entry_time_sim, exit_time=dt.datetime.now(dt.timezone.utc),
                    side=original_side, entry_price=sim_entry_price, exit_price=sim_exit_price,
                    qty=abs(sim_qty), leverage=sim_leverage,
                )
                # Clear the simulated trade state
                self.current_trade = None
                # Update last exit time for cooldown
                self.last_exit_time = dt.datetime.now(dt.timezone.utc)
                return True # Indicate successful simulated closure
            else:
                # Attempted to close in dry run, but no simulated trade was open
                log_console(logging.WARNING, f"[DRY RUN] Close attempt failed: No simulated trade was open for {self.symbol}.", symbol=self.symbol)
                self.current_trade = None # Ensure state is clear just in case
                return False # Indicate failure

        # --- Real Position Close (V5 API) ---
        try:
            # Generate a unique order link ID for the closing order
            close_order_link_id = f"close_{self.symbol}_{exit_reason[:5]}_{int(time.time()*1000)}"
            # Place a market order with reduceOnly=True to ensure it only closes the position
            response = self.session.place_order(
                category=self.category, symbol=self.symbol,
                side=close_side, orderType="Market", qty=qty_str,
                reduceOnly=True, # CRUCIAL: Ensures order only reduces/closes position
                orderLinkId=close_order_link_id,
            )

            # --- Process Close Order Response ---
            if response and response.get("retCode") == 0:
                order_id = response.get("result", {}).get("orderId")
                if order_id:
                    log_console(logging.INFO, f"Position close order {order_id} submitted successfully. Confirming execution...", symbol=self.symbol)
                    # --- Confirm Close Order Execution ---
                    filled_qty, avg_exit_price = self.confirm_order(order_id, abs_size)

                    # --- Validate Close Fill ---
                    # Check if the close order filled sufficiently (e.g., >= 99% of the position size)
                    # This accounts for potential minor discrepancies due to fees or slippage representation
                    min_close_fill_ratio = 0.99
                    fill_tolerance = self.qty_step_float * 0.5 # Tolerance based on step size
                    required_fill = max(abs_size * min_close_fill_ratio, abs_size - fill_tolerance)

                    if filled_qty >= required_fill - FLOAT_COMPARISON_TOLERANCE:
                        log_console(logging.INFO, f"Close order {order_id} confirmed filled successfully. Filled Qty: {filled_qty:.{self.qty_precision}f}, Avg Exit Price: {avg_exit_price or 'N/A'}", symbol=self.symbol)

                        # --- Log Trade Metrics ---
                        exit_time = dt.datetime.now(dt.timezone.utc)
                        self.last_exit_time = exit_time # Set cooldown timer

                        # Use confirmed average exit price if available and valid, otherwise fallback
                        final_exit_price_log = avg_exit_price
                        if final_exit_price_log is None or final_exit_price_log <= FLOAT_COMPARISON_TOLERANCE:
                            # Fallback 1: Use the estimated price passed to the function
                            final_exit_price_log = exit_price_est
                            log_msg = f"using estimated exit price ({exit_price_est})" if exit_price_est else "using entry price as fallback"
                            log_console(logging.WARNING, f"Could not get valid confirmed avg exit price for {self.symbol} metrics, {log_msg}.", symbol=self.symbol)
                            # Fallback 2: Use entry price if estimate is also invalid (P&L will be inaccurate)
                            if final_exit_price_log is None or final_exit_price_log <= FLOAT_COMPARISON_TOLERANCE:
                                final_exit_price_log = entry_price

                        # Ensure entry time is a valid datetime object for metrics
                        entry_time_log = entry_time # From position_data
                        if not isinstance(entry_time_log, dt.datetime):
                            # Attempt to get from internal state as a fallback if API didn't provide it
                            if self.current_trade and self.current_trade.get('symbol') == self.symbol:
                                entry_time_log = self.current_trade.get('entry_time')
                            # Final fallback if still missing (results in inaccurate duration)
                            if not isinstance(entry_time_log, dt.datetime):
                                log_console(logging.WARNING, f"Entry time missing or invalid for {self.symbol} metrics. Using fallback time (exit - 15min).", symbol=self.symbol)
                                entry_time_log = exit_time - dt.timedelta(minutes=15) # Arbitrary fallback

                        # Log the completed trade using the metrics tracker
                        self.metrics.add_trade(
                            symbol=self.symbol, entry_time=entry_time_log, exit_time=exit_time,
                            side=side, entry_price=entry_price, exit_price=final_exit_price_log,
                            # Use actual filled quantity from confirmation for metrics accuracy
                            qty=filled_qty,
                            leverage=leverage,
                        )
                        # Clear internal trade state after successful confirmed close
                        self.current_trade = None
                        return True # Indicate successful closure
                    else:
                        # Close order confirmed but filled quantity is insufficient
                        log_console(logging.ERROR, f"Position close order {order_id} filled qty ({filled_qty:.{self.qty_precision}f}) is less than expected ({required_fill:.{self.qty_precision}f}). Position might be partially closed. Manual check required!", symbol=self.symbol)
                        # Do not clear internal state here, as position might still be partially open.
                        # Consider querying position again (`get_position()`) to update internal state with the remainder?
                        return False # Indicate closure failed or was incomplete
                else:
                    # API returned success code but no order ID for the close order
                    log_console(logging.ERROR, f"Close order placement API call succeeded (retCode 0) but no Order ID returned. Response: {response}", symbol=self.symbol)
                    return False
            else: # API call to place the close order failed (retCode != 0)
                error_msg = response.get('retMsg', 'N/A') if response else "Empty/invalid response"
                error_code = response.get('retCode', 'N/A')

                # --- Handle Specific "Already Closed" Errors ---
                # Check for V5 error codes or messages indicating the position is already closed or size is zero.
                # 110025: Position idx not match position side (can happen if pos closed)
                # 110066: The current position size is zero
                # 3400074: Order quantity exceeded position size * leverage (often means pos size is smaller than expected or 0)
                # 110044: Reduce-only order failed as it would increase position (means no position exists on that side)
                # 110071: reduce only order is rejected (generic reduce only failure, often due to no position)
                pos_already_closed_codes = [110025, 110066, 3400074, 110044, 110071]
                # Check common message fragments too
                already_closed_msg_fragments = [
                    "position size is zero", "order qty exceeded position size",
                    "reduceonly", "reduce-only", "position idx not match",
                    "less than the minimum", # Sometimes indicates closing dust fails
                ]
                is_already_closed = error_code in pos_already_closed_codes or \
                                    any(frag in error_msg.lower() for frag in already_closed_msg_fragments)

                if is_already_closed:
                    # If API indicates position is already closed or size is zero/mismatched
                    log_console(logging.WARNING, f"Close attempt failed for {self.symbol}, but API indicates position is already closed or size issue (Code: {error_code}, Msg: '{error_msg}'). Assuming closed.", symbol=self.symbol)

                    # --- Log Metrics from Internal State (Best Effort) ---
                    # If we had internal state, log the trade based on that, as it likely closed externally (e.g., SL/TP hit, manual close)
                    if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                        exit_time = dt.datetime.now(dt.timezone.utc)
                        # Use estimated exit price or entry price as fallback for metrics
                        final_exit_price = exit_price_est
                        if final_exit_price is None or final_exit_price <= FLOAT_COMPARISON_TOLERANCE:
                            final_exit_price = self.current_trade.get("avgPrice", 0) # Use internal entry price
                        if final_exit_price <= FLOAT_COMPARISON_TOLERANCE: final_exit_price = 1.0 # Absolute fallback

                        entry_time_actual = self.current_trade.get('entry_time') or (exit_time - dt.timedelta(minutes=15)) # Fallback entry time
                        internal_pos_size = self.current_trade.get("size", abs_size) # Use internal size
                        internal_entry_price = self.current_trade.get("avgPrice", entry_price) # Use internal entry
                        internal_side = self.current_trade.get("side", side)
                        internal_leverage = self.current_trade.get("leverage", leverage)

                        log_console(logging.INFO, f"Logging trade metrics from internal state for {self.symbol} as API indicated position already closed/size issue.", symbol=self.symbol)
                        self.metrics.add_trade(
                            symbol=self.symbol, entry_time=entry_time_actual, exit_time=exit_time,
                            side=internal_side, entry_price=internal_entry_price, exit_price=final_exit_price,
                            qty=abs(internal_pos_size), leverage=internal_leverage
                        )
                        self.current_trade = None # Clear internal state as it's confirmed closed externally
                    else:
                         log_console(logging.INFO, f"No internal trade state found for {self.symbol} when API indicated already closed.", symbol=self.symbol)

                    self.last_exit_time = dt.datetime.now(dt.timezone.utc) # Set cooldown timer
                    return True # Treat as successful closure since position is gone
                else:
                    # Genuine failure to place the close order
                    log_console(logging.ERROR, f"Position close order placement failed for {self.symbol}: {error_msg} (Code: {error_code})", symbol=self.symbol)
                    return False # Indicate closure failed
        except Exception as e:
            log_console(logging.ERROR, f"Exception occurred during position close attempt for {self.symbol}: {e}", symbol=self.symbol, exc_info=True)
            return False

    def get_higher_tf_trend(self) -> bool:
        """
        Checks the trend on a higher timeframe (defined in config) using cached data.
        Fetches new data via REST if cache is expired or invalid.

        The definition of a "favorable" trend is customizable within this function.
        Currently checks if Fast MA > Mid MA on the HTF.

        Returns:
            True if the HTF trend is considered favorable for taking trades (or if the
            check is disabled or fails), False if the HTF trend is unfavorable.
        """
        # Check if multi-timeframe filtering is enabled in the strategy config
        if not self.strategy_cfg.get("ENABLE_MULTI_TIMEFRAME", False):
            return True # If disabled, always allow trades (return True)

        # --- Cache Check ---
        cache_ttl_seconds = self.bot_cfg.get("HIGHER_TF_CACHE_SECONDS", 3600) # Default 1 hour cache
        now_utc = dt.datetime.now(dt.timezone.utc)
        # Check if cache exists, is valid (not None), and hasn't expired
        if (self.higher_tf_cache is not None and self.higher_tf_cache_time and
                (now_utc - self.higher_tf_cache_time).total_seconds() < cache_ttl_seconds):
            # Use cached result
            log_console(logging.DEBUG, f"Using cached HTF trend result for {self.symbol}: {'Favorable/Allow' if self.higher_tf_cache else 'Unfavorable'}", symbol=self.symbol)
            return self.higher_tf_cache

        # --- Fetch and Analyze HTF Data ---
        higher_tf = str(self.strategy_cfg.get("HIGHER_TIMEFRAME", "60")) # Get HTF from config
        # Basic validation: Ensure HTF is different from base timeframe
        if higher_tf == self.timeframe:
            log_console(logging.WARNING, f"Higher timeframe ({higher_tf}) is the same as base timeframe ({self.timeframe}) for {self.symbol}. Disabling MTF check.", symbol=self.symbol)
            # Cache as True (allow trades) and update time
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True

        # Fetch slightly more data for HTF to ensure indicator calculation is robust
        # Use the calculated KLINE_LIMIT for the *base* timeframe as a guide, maybe add buffer
        htf_kline_limit = self.internal_cfg.get("KLINE_LIMIT", 120) + 50 # Fetch more candles for HTF
        log_console(logging.INFO, f"Fetching HTF ({higher_tf}) data for {self.symbol} trend analysis...", symbol=self.symbol)

        # Use the REST fetcher which also calculates indicators based on the *same strategy config*
        # Assumption: HTF trend analysis uses the same indicators/periods as the base timeframe.
        # If different indicators/periods are needed for HTF, this logic needs adjustment.
        df_higher = self.get_ohlcv_rest(timeframe=higher_tf, limit=htf_kline_limit)

        # --- Validate Fetched HTF Data ---
        # Need at least 2 rows: one for the last closed candle (-2), one potentially ongoing (-1)
        min_rows_needed_htf = 2
        if df_higher is None or df_higher.empty or len(df_higher) < min_rows_needed_htf:
            log_console(logging.WARNING, f"Could not get sufficient HTF ({higher_tf}) data ({len(df_higher) if df_higher is not None else 0} rows) or indicators failed for {self.symbol}. Allowing trade (fail-open).", symbol=self.symbol)
            # Fail open: Allow trade if HTF check fails to avoid blocking trades due to temporary data issues
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True

        try:
            # --- Determine HTF Trend ---
            # Use the same smoother type (EMA or Ehlers) as the base timeframe for consistency
            use_ehlers_htf = self.strategy_cfg.get("USE_EHLERS_SMOOTHER", False)
            smoother_prefix_htf = "ss" if use_ehlers_htf else "ema"
            fast_ma_col_htf = f"{smoother_prefix_htf}_fast"
            mid_ma_col_htf = f"{smoother_prefix_htf}_mid"

            # Check if the required MA columns exist in the calculated HTF DataFrame
            if fast_ma_col_htf not in df_higher.columns or mid_ma_col_htf not in df_higher.columns:
                log_console(logging.ERROR, f"Required MA cols ({fast_ma_col_htf}, {mid_ma_col_htf}) not found in HTF ({higher_tf}) data after calculation for {self.symbol}. Allowing trade.", symbol=self.symbol)
                self.higher_tf_cache = True; self.higher_tf_cache_time = now_utc
                return True # Fail open

            # Analyze the trend on the second-to-last fully closed candle (-2 index) for stability
            # Accessing iloc[-1] might give incomplete candle data
            candle_to_check = df_higher.iloc[-2]
            fast_ma_htf = candle_to_check.get(fast_ma_col_htf)
            mid_ma_htf = candle_to_check.get(mid_ma_col_htf)

            # Check for NaN values on the specific candle being checked (should be filled by indicator calc, but safeguard)
            if pd.isna(fast_ma_htf) or pd.isna(mid_ma_htf):
                log_console(logging.WARNING, f"HTF ({higher_tf}) MAs are NaN on check candle ({candle_to_check.name}) for {self.symbol}. Allowing trade.", symbol=self.symbol)
                trend_is_favorable = True # Fail open
            else:
                # --- Define Favorable Trend Logic (CUSTOMIZABLE) ---
                # Example: Allow trade only if HTF fast MA > mid MA (suggests uptrend bias)
                # More complex logic could be added: check slope, ADX on HTF, etc.
                trend_is_favorable = fast_ma_htf > mid_ma_htf
                trend_desc = "Bullish (Fast>Mid)" if trend_is_favorable else "Bearish/Neutral (Fast<=Mid)"
                log_console(logging.INFO,
                            f"HTF ({higher_tf}) trend check (@ {candle_to_check.name.strftime('%Y-%m-%d %H:%M')}): "
                            f"{fast_ma_col_htf}={fast_ma_htf:.{self.price_precision}f}, {mid_ma_col_htf}={mid_ma_htf:.{self.price_precision}f}. "
                            f"Status: {trend_desc} -> Allow Trade: {trend_is_favorable}",
                            symbol=self.symbol)

            # --- Update Cache ---
            self.higher_tf_cache = trend_is_favorable
            self.higher_tf_cache_time = now_utc
            return trend_is_favorable

        except IndexError:
            # Should be caught by the length check earlier, but handle just in case
            log_console(logging.WARNING, f"Index error accessing HTF ({higher_tf}) data (need >= {min_rows_needed_htf} rows, have {len(df_higher)}). Allowing trade.", symbol=self.symbol)
            self.higher_tf_cache = True; self.higher_tf_cache_time = now_utc
            return True # Fail open
        except Exception as e:
            log_console(logging.ERROR, f"Error during HTF ({higher_tf}) trend analysis for {self.symbol}: {e}. Allowing trade.", symbol=self.symbol, exc_info=True)
            self.higher_tf_cache = True; self.higher_tf_cache_time = now_utc
            return True # Fail open


# --- Main Trading Bot Orchestrator ---
class MomentumScannerTrader:
    """
    Orchestrates the trading bot operations across multiple symbols, manages API sessions,
    WebSocket connections, and the main trading loop using Bybit V5 API.
    """
    def __init__(self, api_key: str, api_secret: str, config: Dict[str, Any], dry_run: bool = False):
        """
        Initializes the main bot orchestrator, sets up API connections, initializes
        individual symbol traders, and starts the WebSocket monitor if needed.

        Args:
            api_key, api_secret: Bybit API credentials.
            config: The validated application configuration dictionary.
            dry_run: Global dry-run flag.

        Raises:
            ConnectionError: If the initial API connection test fails.
            RuntimeError: If no traders can be initialized or other critical init failures occur.
            ValueError: If configuration structure is invalid (should be caught earlier).
        """
        self.config = config
        self.dry_run = dry_run
        # Extract config sections safely
        self.bybit_cfg: Dict[str, Any] = config.get("BYBIT_CONFIG", {})
        self.risk_cfg: Dict[str, Any] = config.get("RISK_CONFIG", {})
        self.bot_cfg: Dict[str, Any] = config.get("BOT_CONFIG", {})
        # State flags
        self.running: bool = True # Controls the main trading loop
        self.shutdown_requested: bool = False # Signals threads and loops to terminate

        # --- API Setup ---
        self.use_testnet: bool = self.bybit_cfg.get("USE_TESTNET", False)
        self.account_type: str = self.bybit_cfg.get("ACCOUNT_TYPE", "UNIFIED").upper()

        # Determine V5 API category based on account type. This influences which endpoints/streams are used.
        # Common mapping: UNIFIED/CONTRACT -> linear (for USDT perps), SPOT -> spot.
        # Adjust if using Inverse contracts or Options.
        if self.account_type == "SPOT":
            self.category = "spot"
        elif self.account_type in ["UNIFIED", "CONTRACT"]:
            # Defaulting to 'linear' for USDT-margined perpetuals under UNIFIED/CONTRACT.
            # If trading inverse (e.g., BTC-margined), category should be 'inverse'.
            # This assumes the config targets linear contracts primarily.
            self.category = "linear"
            log_console(logging.INFO, f"Using V5 API category '{self.category}' for {self.account_type} account. Ensure configured symbols match this category (e.g., USDT perpetuals).")
        else:
             # Fallback for potentially unsupported account types, warn user.
             log_console(logging.WARNING, f"Account type '{self.account_type}' might not map directly to a standard V5 category. Defaulting to 'linear'. Verify compatibility.")
             self.category = "linear"

        self.api_key = api_key
        self.api_secret = api_secret
        self.session: Optional[HTTP] = None # V5 HTTP session object
        self.ws: Optional[WebSocket] = None # V5 WebSocket session object
        self.ws_active: bool = False # Flag indicating if WS connection is currently active
        self.ws_connection_thread: Optional[threading.Thread] = None # Thread for managing WS connection
        self.ws_lock = threading.Lock() # Lock for thread-safe access to self.ws and self.ws_active

        # --- Traders and Metrics Initialization ---
        self.traders: Dict[str, SymbolTrader] = {} # Dictionary to hold SymbolTrader instances
        # Get fee rate from config for metrics calculation
        fee_rate = self.risk_cfg.get("FEE_RATE", 0.0006) # Default 0.06% taker fee
        self.metrics = TradeMetrics(fee_rate=fee_rate) # Initialize shared metrics tracker

        log_console(logging.INFO, f"Initializing Trader Orchestrator (Testnet: {self.use_testnet}, Account: {self.account_type}, Category: {self.category})...")
        try:
            # --- Setup HTTP Session ---
            # Configure HTTP session with receive window from config (helps with timing issues)
            recv_window = self.bot_cfg.get("API_RECV_WINDOW", 10000) # Default 10 seconds
            self.session = HTTP(
                testnet=self.use_testnet, api_key=self.api_key, api_secret=self.api_secret,
                recv_window=recv_window,
                # Optional: Enable detailed request/response logging from pybit's HTTP client
                # log_requests=args.debug # Requires modification to pybit or custom wrapper
                # logging_level=logging.DEBUG if args.debug else logging.INFO # If pybit supports it directly
            )
            # --- Test API Connection Early ---
            self._test_api_connection() # Raises ConnectionError on failure

            # --- Initialize Individual Symbol Traders ---
            self._initialize_traders() # Raises RuntimeError if no traders succeed

            # --- Start WebSocket Monitor Thread ---
            # Start only if traders were initialized and at least one uses WebSocket
            if self.traders and any(trader.use_websocket for trader in self.traders.values()):
                log_console(logging.INFO, "WebSocket is enabled for one or more symbols. Starting WebSocket connection thread...")
                # Daemon=True ensures thread exits when main program exits
                self.ws_connection_thread = threading.Thread(target=self._start_and_monitor_websocket, name="WebSocketMonitor", daemon=True)
                self.ws_connection_thread.start()
                # Allow some time for the initial connection attempt and subscriptions
                time.sleep(self.bot_cfg.get("WS_STARTUP_DELAY_SECONDS", 5.0))
            else:
                log_console(logging.INFO, "WebSocket is disabled for all symbols or no traders initialized. Bot will use REST API polling only.")

        except (ConnectionError, RuntimeError, ValueError) as init_e:
            # Catch specific critical errors during initialization
            log_console(logging.CRITICAL, f"Bot initialization failed critically: {init_e}", exc_info=False)
            self.running = False; self.shutdown_requested = True # Ensure bot doesn't proceed
            self._cleanup_resources() # Attempt cleanup even on init failure
            raise init_e # Re-raise to stop execution in the main block
        except Exception as e:
            # Catch any other unexpected errors during initialization
            log_console(logging.CRITICAL, f"Unexpected error during bot initialization: {e}", exc_info=True)
            self.running = False; self.shutdown_requested = True
            self._cleanup_resources()
            raise RuntimeError("Unexpected initialization failure") from e

    def _test_api_connection(self):
        """Tests the REST API connection and authentication using V5 get_server_time."""
        if not self.session:
             # This should not happen if called after session init, but safeguard
             raise ConnectionError("HTTP session not initialized before API test.")
        try:
            log_console(logging.DEBUG, "Testing API connection via get_server_time...")
            # V5 endpoint for server time
            server_time_resp = self.session.get_server_time()

            if server_time_resp and server_time_resp.get('retCode') == 0:
                # --- Check Clock Skew ---
                # V5 returns time in nanoseconds string
                time_nano_str = server_time_resp.get('result', {}).get('timeNano', '0')
                server_ts_sec = int(time_nano_str) / 1e9 # Convert ns to seconds
                server_dt = dt.datetime.fromtimestamp(server_ts_sec, tz=dt.timezone.utc)
                local_dt_utc = dt.datetime.now(dt.timezone.utc)
                time_diff_seconds = abs((local_dt_utc - server_dt).total_seconds())
                # Get max allowable skew from config, default to 60 seconds
                max_allowable_skew = self.bot_cfg.get("MAX_CLOCK_SKEW_SECONDS", 60)
                if time_diff_seconds > max_allowable_skew:
                    log_console(logging.WARNING, f"Potential Clock Skew: System clock differs from Bybit server by {time_diff_seconds:.1f} seconds (threshold: {max_allowable_skew}s). Ensure system time is synchronized (e.g., using NTP).")
                else:
                     log_console(logging.DEBUG, f"Clock skew check passed ({time_diff_seconds:.1f}s difference).")

                log_console(logging.INFO, f"REST API connection successful. Server Time: {server_dt.isoformat()}")
            else:
                # API call failed
                error_msg = server_time_resp.get('retMsg', 'N/A') if server_time_resp else "Empty Response"
                error_code = server_time_resp.get('retCode', 'N/A')
                # Check specifically for authentication errors (V5 codes 10003, 10004)
                if error_code in [10003, 10004] or "invalid api key" in error_msg.lower() or "signature" in error_msg.lower():
                     log_console(logging.CRITICAL, f"API Authentication Failed: {error_msg} (Code: {error_code}). Please check your API Key, Secret, and Permissions (ensure V5 Unified Trading access).")
                     raise ConnectionError(f"API Authentication Failed (Code: {error_code})")
                else:
                    # Other API error during test
                    raise ConnectionError(f"API connection test failed: {error_msg} (Code: {error_code})")
        except ConnectionError as ce:
            # Re-raise specific ConnectionError for clarity
            log_console(logging.CRITICAL, f"API connection failed during test: {ce}. Check network, API endpoint, keys, permissions, and system time sync.")
            raise ce
        except Exception as e:
            # Catch any other exceptions during the test
            log_console(logging.CRITICAL, f"API connection test failed unexpectedly: {e}.", exc_info=True)
            raise ConnectionError("Failed API connection test due to unexpected error.") from e

    def _initialize_traders(self):
        """Creates and initializes SymbolTrader instances based on the configuration."""
        symbols_cfg_list: List[Dict[str, Any]] = self.bybit_cfg.get("SYMBOLS", [])
        # Basic check if symbols list exists and is a list (should be validated earlier, but double-check)
        if not isinstance(symbols_cfg_list, list):
            raise ValueError("BYBIT_CONFIG.SYMBOLS is missing or not a list (validation should have caught this).")
        if not symbols_cfg_list:
             raise RuntimeError("BYBIT_CONFIG.SYMBOLS list is empty. No symbols to trade.")

        log_console(logging.INFO, f"Initializing traders for {len(symbols_cfg_list)} configured symbol(s)...")
        successful_traders = 0
        temp_traders: Dict[str, SymbolTrader] = {} # Store successfully initialized traders here

        # Use deepcopy of the symbol config list to avoid modifying the original config dict during iteration,
        # especially as SymbolTrader might store internal data back into its copy.
        for i, symbol_cfg_item in enumerate(deepcopy(symbols_cfg_list)):
            # Validate individual symbol config structure again just in case
            if not isinstance(symbol_cfg_item, dict):
                log_console(logging.ERROR, f"Symbol config entry #{i} is invalid (not a dictionary). Skipping.")
                continue
            symbol = symbol_cfg_item.get("SYMBOL")
            if not symbol or not isinstance(symbol, str):
                log_console(logging.ERROR, f"Symbol config entry #{i} missing or has invalid 'SYMBOL' key (must be a non-empty string). Skipping.")
                continue
            # Check for duplicate symbols in the configuration
            if symbol in temp_traders:
                log_console(logging.WARNING, f"Duplicate symbol '{symbol}' found in configuration. Skipping subsequent entry #{i}.", symbol=symbol)
                continue

            # --- Attempt to Initialize Trader for Symbol ---
            log_console(logging.INFO, f"--- Initializing trader for {symbol} ({i+1}/{len(symbols_cfg_list)}) ---", symbol=symbol)
            try:
                # Ensure HTTP session is available (should be, checked earlier)
                if not self.session:
                     raise RuntimeError("HTTP session is not available for trader initialization.")

                # Create the SymbolTrader instance
                # Pass the shared HTTP session and the determined category
                trader = SymbolTrader(
                    self.api_key, self.api_secret, self.config, symbol_cfg_item,
                    self.dry_run, self.metrics, self.session, self.category
                )
                # SymbolTrader's __init__ raises RuntimeError on critical failure.
                # We also check the internal flag for good measure.
                if trader.initialization_successful:
                    temp_traders[symbol] = trader # Add to the dictionary of active traders
                    successful_traders += 1
                    log_console(logging.INFO, f"--- Trader for {symbol} initialized successfully ---", symbol=symbol)
                else:
                    # This case should ideally be covered by RuntimeError from SymbolTrader, but log defensively.
                    log_console(logging.ERROR, f"Trader initialization reported failure for {symbol} (init flag is False). Skipping.", symbol=symbol)

            except RuntimeError as trader_init_e:
                # Catch specific RuntimeError raised by SymbolTrader on failure
                log_console(logging.ERROR, f"Initialization failed for {symbol}: {trader_init_e}. Skipping this symbol.", symbol=symbol)
            except Exception as e:
                # Catch any other unexpected errors during a specific trader's initialization
                log_console(logging.ERROR, f"Unexpected error occurred while initializing trader for {symbol}: {e}. Skipping this symbol.", symbol=symbol, exc_info=True)
            # Continue to the next symbol even if one fails

        # --- Final Check after Attempting All Symbols ---
        if successful_traders == 0:
            # If no traders could be initialized, the bot cannot run. Raise critical error.
            raise RuntimeError("CRITICAL: No traders could be initialized successfully. Check configuration, API keys/permissions, and logs for individual symbol errors.")

        self.traders = temp_traders # Assign the dictionary of successfully initialized traders
        log_console(logging.INFO, f"Successfully initialized {successful_traders}/{len(symbols_cfg_list)} traders: {', '.join(self.traders.keys())}")

    def _start_and_monitor_websocket(self):
        """
        Manages the WebSocket connection lifecycle: starts, subscribes, monitors,
        and handles reconnections with exponential backoff. Runs in a dedicated thread.
        """
        log_console(logging.INFO, "WebSocket monitoring thread started.")
        # Determine V5 WebSocket channel type based on the trading category
        # Options: linear, inverse, spot, option
        ws_channel_type = self.category.lower()
        if ws_channel_type not in ["linear", "inverse", "spot", "option"]:
             log_console(logging.ERROR, f"Unsupported category '{self.category}' for WebSocket channel_type. Defaulting to 'linear'. Verify config.", symbol=None) # No specific symbol here
             ws_channel_type = "linear" # Default fallback

        # Reconnection delay parameters from config
        retry_delay = float(self.bot_cfg.get("WS_RECONNECT_DELAY_INIT_SECONDS", 10.0))
        max_retry_delay = float(self.bot_cfg.get("WS_RECONNECT_DELAY_MAX_SECONDS", 120.0))

        while not self.shutdown_requested:
            ws_conn_active = False
            # Check current WS status under lock
            with self.ws_lock: ws_conn_active = self.ws is not None and self.ws_active

            if not ws_conn_active:
                # --- Attempt WebSocket Connection ---
                log_console(logging.INFO, f"Attempting WebSocket connection (Channel Type: {ws_channel_type})...")
                try:
                    # Create a new WebSocket instance for each connection attempt using pybit V5 WebSocket
                    temp_ws = WebSocket(
                        testnet=self.use_testnet,
                        channel_type=ws_channel_type, # linear, inverse, spot, option
                        api_key=self.api_key, api_secret=self.api_secret,
                        # Configure ping/pong intervals for connection health checks
                        ping_interval=self.bot_cfg.get("WS_PING_INTERVAL", 20), # Send ping every 20s
                        ping_timeout=self.bot_cfg.get("WS_PING_TIMEOUT", 10), # Expect pong within 10s
                        # Assign callbacks for different WS events
                        on_message=self._handle_ws_message, # Handles non-data messages (auth, sub responses)
                        on_error=self._handle_ws_error, # Handles connection/protocol errors
                        on_close=self._handle_ws_close, # Handles unexpected connection closures
                        on_open=self._handle_ws_open, # Sets ws_active flag on successful open
                        retries=0, # Disable pybit's built-in retries; we manage reconnection manually
                        # Optional: Enable detailed trace logging from the underlying websocket-client library
                        # trace=args.debug # Requires access to args or config flag
                    )

                    # --- Prepare Subscriptions ---
                    # Collect required subscriptions from all initialized traders that use WebSocket
                    subscriptions = []
                    for symbol, trader in self.traders.items():
                        if trader.use_websocket:
                            # V5 kline topic format: kline.{interval}.{symbol}
                            topic = f"kline.{trader.timeframe}.{symbol}"
                            subscriptions.append(topic)
                            log_console(logging.DEBUG, f"Prepared WS subscription: {topic}", symbol=symbol)
                    # TODO: Add other subscriptions if needed (e.g., 'publicTrade.{symbol}', 'order', 'position')

                    # Proceed only if there are subscriptions to make
                    if subscriptions:
                        # Assign the newly created WS object to the instance variable under lock
                        # Do this just before starting interaction to avoid race conditions
                        with self.ws_lock:
                            self.ws = temp_ws

                        # --- Subscribe to Streams ---
                        # Subscribing implicitly starts the WebSocket connection in pybit
                        # The on_open callback will set self.ws_active = True upon success.
                        # Use the dispatcher callback to route messages to the correct trader.
                        log_console(logging.INFO, f"WebSocket subscribing to {len(subscriptions)} streams...")
                        # Subscribe to all topics at once using the dispatcher
                        self.ws.subscribe(subscriptions, callback=self._dispatch_ws_callback)

                        # --- Wait for Connection Establishment ---
                        # Wait for the on_open callback to set ws_active flag, with a timeout
                        wait_time = 0
                        max_wait_connect = self.bot_cfg.get("WS_CONNECT_TIMEOUT_SECONDS", 30)
                        while wait_time < max_wait_connect:
                            with self.ws_lock: ws_established = self.ws_active
                            if ws_established or self.shutdown_requested: break # Exit wait if connected or shutdown
                            time.sleep(1); wait_time += 1

                        if self.shutdown_requested: break # Exit monitor loop if shutdown requested during wait

                        # --- Check Connection Status After Wait ---
                        with self.ws_lock: ws_established = self.ws_active
                        if ws_established:
                            # Connection successful! Reset retry delay and enter monitoring phase.
                            log_console(logging.INFO,"WebSocket connection established successfully. Monitoring activity...")
                            retry_delay = float(self.bot_cfg.get("WS_RECONNECT_DELAY_INIT_SECONDS", 10.0)) # Reset backoff

                            # --- Monitor Active Connection ---
                            # Keep this thread alive while WS is active, periodically checking the flag.
                            # The on_error or on_close callbacks will set ws_active=False if connection drops.
                            while not self.shutdown_requested:
                                with self.ws_lock: ws_still_active = self.ws_active
                                if not ws_still_active:
                                    log_console(logging.INFO, "WebSocket inactive status detected by monitor thread. Triggering reconnection attempt.")
                                    break # Break inner monitoring loop to trigger reconnection logic
                                # Sleep for a configured interval before checking again
                                time.sleep(self.bot_cfg.get("WS_MONITOR_CHECK_INTERVAL_SECONDS", 5.0))
                        else:
                             # Connection timed out
                             log_console(logging.ERROR, f"WebSocket connection failed to establish within {max_wait_connect} seconds. Retrying...")
                             self._close_websocket_connection() # Ensure cleanup before retry attempt
                             # Apply exponential backoff delay before next connection attempt
                             if not self.shutdown_requested: time.sleep(retry_delay)
                             retry_delay = min(max_retry_delay, retry_delay * 1.5) # Increase delay (e.g., 1.5x)

                    else: # No symbols configured for WebSocket
                        log_console(logging.INFO, "No symbols are configured to use WebSocket. WebSocket thread exiting.")
                        break # Exit the main monitoring loop

                except Exception as e:
                    # Catch errors during WS initialization or subscription
                    log_console(logging.ERROR, f"Failed to initialize or start WebSocket connection: {e}. Retrying in {retry_delay:.1f}s.", exc_info=False)
                    self._close_websocket_connection() # Ensure cleanup of potentially partially initialized WS object
                    # Apply exponential backoff delay before next attempt
                    if not self.shutdown_requested: time.sleep(retry_delay)
                    retry_delay = min(max_retry_delay, retry_delay * 1.5) # Increase delay

            else: # WS connection is currently active
                # Sleep for a longer interval while connection is stable
                monitor_interval_stable = max(10.0, self.bot_cfg.get("WS_MONITOR_CHECK_INTERVAL_SECONDS", 5.0))
                time.sleep(monitor_interval_stable)

        # --- End of Monitoring Loop ---
        log_console(logging.INFO, "WebSocket monitoring thread finished.")
        # Ensure final cleanup of the WebSocket connection object on thread exit
        self._close_websocket_connection()

    def _dispatch_ws_callback(self, message: Dict[str, Any]):
        """
        Dispatches incoming WebSocket messages (primarily kline data) to the
        callback function of the correct SymbolTrader instance based on the topic.
        """
        try:
            # V5 messages usually have a 'topic' field for data streams
            topic = message.get("topic")
            if topic:
                # --- Route Kline Messages ---
                # Example topic: kline.15.BTCUSDT
                if topic.startswith("kline."):
                    parts = topic.split('.')
                    # Expecting format: kline.{interval}.{symbol}
                    if len(parts) == 3:
                        symbol = parts[2]
                        # Find the trader instance responsible for this symbol
                        if symbol in self.traders:
                            # Call the specific trader's WebSocket callback method
                            self.traders[symbol]._websocket_callback(message)
                        else:
                            # Received data for a symbol we don't have a trader for (shouldn't happen if subs are correct)
                            log_console(logging.DEBUG, f"WS Dispatch: Received kline data for unsubscribed/inactive symbol: {symbol}", symbol=symbol)
                    else:
                        log_console(logging.WARNING, f"WS Dispatch: Malformed kline topic received: {topic}")
                # --- Route Other Data Streams (Example) ---
                # elif topic.startswith("publicTrade."):
                #     symbol = topic.split('.')[-1]
                #     if symbol in self.traders: self.traders[symbol]._handle_trade_message(message) # Hypothetical handler
                # elif topic == "order": self._handle_order_update(message) # Orchestrator handles private updates
                # elif topic == "position": self._handle_position_update(message) # Orchestrator handles private updates
                else:
                    # Pass other topic messages to the generic handler if needed for logging/debugging
                    log_console(logging.DEBUG, f"WS Dispatch: Passing unhandled topic '{topic}' to generic handler.")
                    self._handle_ws_message(message)
            else:
                # Messages without a topic (like auth/sub responses, pings) are handled by _handle_ws_message
                log_console(logging.DEBUG, f"WS Dispatch: Passing non-topic message to generic handler: {message.get('op', message)}")
                self._handle_ws_message(message)

        except Exception as e:
            log_console(logging.ERROR, f"Error occurred during WebSocket message dispatch: {e}. Message: {message}", exc_info=True)

    def _close_websocket_connection(self):
        """Safely closes the WebSocket connection object and resets related flags."""
        with self.ws_lock: # Ensure thread safety
            ws_instance_to_close = self.ws # Get current instance reference

            # Only proceed if there is a WS object to close
            if ws_instance_to_close:
                log_console(logging.INFO, "Closing WebSocket connection object...")
                # Clear internal reference *before* calling exit() to prevent race conditions
                # where another thread might try to use the object while it's closing.
                self.ws = None
                self.ws_active = False # Mark as inactive immediately

                try:
                    # Use the exit() method provided by pybit's V5 WebSocket class to cleanly shut down
                    ws_instance_to_close.exit()
                    log_console(logging.DEBUG, "WebSocket exit() method called successfully.")
                except Exception as e:
                    # Log errors during closure, but state is already marked as inactive
                    log_console(logging.ERROR, f"Error closing WebSocket object via exit(): {e}", exc_info=False)
            else:
                # If ws object was already None, ensure the active flag is also False
                if self.ws_active:
                     log_console(logging.DEBUG, "WS object was None, ensuring ws_active flag is False.")
                     self.ws_active = False

    # --- WebSocket Event Handlers (Callbacks for pybit WebSocket) ---

    def _handle_ws_open(self, ws_app):
        """Callback executed by pybit when WebSocket connection is successfully opened."""
        log_console(logging.INFO,"WebSocket connection opened successfully.")
        with self.ws_lock: self.ws_active = True # Set flag indicating connection is live
        # Trigger asynchronous population of initial historical data after WS connects
        self._populate_initial_kline_data_async()

    def _populate_initial_kline_data_async(self):
        """Starts a background thread to fetch initial Kline data via REST API
           immediately after a WebSocket connection is established."""
        log_console(logging.DEBUG,"Starting background thread for initial REST kline data population...")
        # Run the fetcher function in a separate daemon thread
        thread = threading.Thread(target=self._fetch_initial_data_worker, name="InitialDataFetcher", daemon=True)
        thread.start()

    def _fetch_initial_data_worker(self):
        """
        Worker thread function that iterates through WS-enabled traders and fetches
        initial historical data using REST if their internal DataFrame is empty.
        """
        # Short delay to allow subscriptions to potentially settle after WS open
        time.sleep(2.0)
        log_console(logging.INFO,"Initial data fetcher thread running...")
        # Iterate over a copy of trader items for thread safety (in case traders are modified elsewhere)
        traders_to_fetch = list(self.traders.items())

        for symbol, trader in traders_to_fetch:
            if self.shutdown_requested: break # Check for shutdown signal periodically

            # Fetch only if WS is enabled for this trader AND its internal kline_df is still empty/None
            # This prevents overwriting data that might have already arrived via WS quickly
            if trader.use_websocket and (trader.kline_df is None or trader.kline_df.empty):
                log_console(logging.DEBUG, f"Fetching initial REST data for WS-enabled symbol {symbol}...", symbol=symbol)
                try:
                    # Use the trader's own method to fetch and calculate indicators via REST
                    initial_df = trader.get_ohlcv_rest() # Fetches, cleans, calculates indicators

                    # Update the trader's internal DataFrame directly if fetch succeeded
                    if initial_df is not None and not initial_df.empty:
                        # Ensure index is UTC (should be handled by get_ohlcv_rest)
                        if not initial_df.index.tz: initial_df.index = initial_df.index.tz_localize('UTC')
                        # Assign the fetched and calculated data to the trader's state
                        trader.kline_df = initial_df
                        # Explicitly set the last update time
                        trader.last_kline_update_time = dt.datetime.now(dt.timezone.utc)
                        log_console(logging.INFO, f"Successfully initialized kline data for {symbol} from REST API.", symbol=symbol)
                    else:
                        log_console(logging.WARNING, f"Initial REST fetch failed for {symbol}. Trader will rely solely on incoming WS updates for data.", symbol=symbol)
                except Exception as e:
                    log_console(logging.ERROR, f"Error fetching initial REST data for {symbol} in worker thread: {e}", symbol=symbol, exc_info=False)

                # Add a small delay between fetches to avoid hitting rate limits aggressively
                time.sleep(self.bot_cfg.get("INITIAL_FETCH_DELAY_PER_SYMBOL_MS", 200) / 1000.0)

        log_console(logging.INFO,"Initial data fetcher thread finished.")

    def _handle_ws_message(self, message):
        """
        Generic WebSocket message handler callback (called by pybit).
        Primarily used here for non-data messages like authentication responses,
        subscription confirmations, and pings/pongs. Data messages are routed
        via `_dispatch_ws_callback`.
        """
        try:
            # Log generic messages at DEBUG level to avoid noise, enable if needed for debugging WS protocol
            # log_console(logging.DEBUG, f"Generic WS Msg Received: {message}")

            if isinstance(message, dict):
                op = message.get("op")
                # --- Handle Authentication Response ---
                if op == "auth":
                    success = message.get("success", False)
                    ret_msg = message.get("ret_msg", "N/A")
                    log_level = logging.INFO if success else logging.ERROR
                    log_console(log_level, f"WebSocket Authentication Status: {'Success' if success else 'Failed'} ({ret_msg})")
                    if not success:
                        # If auth fails, WS is likely unusable, signal monitor to reconnect/retry
                        log_console(logging.ERROR, "WebSocket authentication failed. Connection will be reset.")
                        with self.ws_lock: self.ws_active = False # Trigger reconnect
                # --- Handle Subscription Response ---
                elif op == "subscribe":
                    success = message.get("success", False)
                    # V5 response echoes subscription args or uses ret_msg
                    sub_args = message.get("args", [])
                    topic_info = ", ".join(map(str, sub_args)) if sub_args else message.get("ret_msg", "")
                    req_id = message.get("req_id") # Might contain client request ID
                    if not topic_info and req_id: topic_info = f"ReqID: {req_id}"

                    if success:
                        log_console(logging.DEBUG, f"WebSocket Subscribe Success: {topic_info}")
                    else:
                        # Log subscription failures clearly
                        log_console(logging.ERROR, f"WebSocket Subscribe Failed: {topic_info} - Message: {message.get('ret_msg', 'N/A')}")
                        # Consider if specific subscription failures require action (e.g., retrying sub)
                # --- Handle Ping/Pong (Usually managed internally by pybit) ---
                elif op == "ping":
                    log_console(logging.DEBUG, f"WebSocket Received Ping: {message}")
                    # pybit WebSocket client typically handles sending pongs automatically.
                    # If manual pong is needed (e.g., auto_pong disabled in underlying library), send here:
                    # with self.ws_lock:
                    #     if self.ws: self.ws.send(json.dumps({"op": "pong", "req_id": message.get("req_id")})) # Echo req_id if present
                elif op == "pong":
                     log_console(logging.DEBUG, f"WebSocket Received Pong: {message}")
                # --- Handle Unsubscribe Response ---
                elif op == "unsubscribe":
                     success = message.get("success", False)
                     ret_msg = message.get("ret_msg", "N/A")
                     log_console(logging.DEBUG, f"WebSocket Unsubscribe Status: {'Success' if success else 'Failed'} ({ret_msg})")
                # --- Handle Other Operations ---
                # else:
                #     log_console(logging.DEBUG, f"Unhandled Generic WS Op Received: {op}, Msg: {message}")
            # else: Handle non-dict messages if necessary (e.g., plain text error messages?)
            #     log_console(logging.DEBUG, f"Received non-dict WS message: {message}")

        except Exception as e:
            log_console(logging.ERROR, f"Error handling generic WebSocket message: {e}", exc_info=True)

    def _handle_ws_error(self, ws_app_or_error, error=None):
        """
        Callback executed by pybit on WebSocket errors (e.g., connection drops, protocol errors).
        Signals the monitor thread to attempt reconnection.
        """
        # The signature might vary slightly; try to capture the error message robustly
        err_msg = error if error else ws_app_or_error # error arg is usually present on actual errors
        is_exception = isinstance(err_msg, Exception)
        log_console(logging.ERROR, f"WebSocket Error Encountered: {err_msg}", exc_info=is_exception)
        # --- Trigger Reconnection ---
        # Set the active flag to False. The monitor thread checks this flag
        # and will initiate the reconnection process if it's False.
        with self.ws_lock:
             # Log only if state changes from active to inactive
             if self.ws_active: log_console(logging.INFO, "WS state set to inactive due to error.")
             self.ws_active = False
             # Do NOT set self.ws = None here, let the monitor thread handle cleanup via _close_websocket_connection

    def _handle_ws_close(self, ws_app):
        """
        Callback executed by pybit when the WebSocket connection is closed.
        Differentiates between expected shutdown and unexpected closure.
        Signals the monitor thread to reconnect on unexpected closures.
        """
        # Check if the closure was expected due to bot shutdown request
        if not self.shutdown_requested:
            log_console(logging.WARNING, "WebSocket connection closed unexpectedly.")
            # --- Trigger Reconnection ---
            # Set active flag to False to signal the monitor thread.
            with self.ws_lock:
                # Log only if state changes from active to inactive
                if self.ws_active: log_console(logging.INFO, "WS state set to inactive due to unexpected close.")
                self.ws_active = False
                # Do NOT set self.ws = None here, let the monitor thread handle cleanup
        else:
            # Closure was expected during shutdown process
            log_console(logging.INFO, "WebSocket connection closed (expected during shutdown).")
            # Ensure state is marked inactive even on expected close
            with self.ws_lock: self.ws_active = False

    def run(self):
        """Main trading loop that orchestrates the bot's operations cycle by cycle."""
        # Pre-run check: Ensure traders were initialized
        if not self.traders:
            log_console(logging.ERROR, "No traders were successfully initialized. Bot cannot run. Exiting.")
            self.running = False; self.shutdown_requested = True
            self._cleanup_resources()
            return

        log_console(logging.INFO, f"{Fore.CYAN}{Style.BRIGHT}--- Starting Main Trading Loop ({len(self.traders)} Symbols) ---")
        # Get configuration parameters for the loop
        sleep_interval: float = float(self.bot_cfg.get("SLEEP_INTERVAL_SECONDS", 60.0))
        metrics_interval: int = int(self.bot_cfg.get("METRICS_LOG_INTERVAL_SECONDS", 3600))
        balance_check_interval: int = self.bot_cfg.get("BALANCE_CHECK_INTERVAL_SECONDS", 300)
        last_balance_check_time: float = 0.0
        # Initialize balance variable
        balance: float = 0.0

        # --- Initial Balance Check ---
        if self.dry_run:
            # Use dummy balance for dry run from config
            balance = float(self.risk_cfg.get("DRY_RUN_DUMMY_BALANCE", 10000.0))
            log_console(logging.INFO, f"Using Dry Run Balance: {balance:.2f} USDT")
        else:
            # Fetch real balance on startup
            log_console(logging.INFO, f"Performing initial balance check for {self.account_type} account (Coin: USDT)...")
            balance = get_available_balance(self.session, "USDT", self.account_type)
            if balance > FLOAT_COMPARISON_TOLERANCE:
                log_console(logging.INFO, f"Initial Balance Check ({self.account_type}): {balance:.2f} USDT Available")
                last_balance_check_time = time.time()
            else:
                log_console(logging.WARNING, f"Initial {self.account_type} USDT available balance is zero or could not be fetched. Trades cannot be sized until balance is positive.")

        # --- Main Trading Cycle Loop ---
        while self.running and not self.shutdown_requested:
            try:
                cycle_start_time = time.time()
                current_dt_utc = dt.datetime.now(dt.timezone.utc)
                print(f"\n{Fore.BLUE}{Style.BRIGHT}===== CYCLE START: {current_dt_utc.isoformat(timespec='seconds')} =====")

                # --- Pre-Cycle Checks ---
                # Check WebSocket status if it's supposed to be running
                ws_should_be_active = self.ws_connection_thread and any(t.use_websocket for t in self.traders.values())
                ws_currently_active = False
                with self.ws_lock: ws_currently_active = self.ws_active
                if ws_should_be_active and not ws_currently_active:
                    log_console(logging.WARNING, "WebSocket connection is currently inactive, but should be running. Strategy may rely on REST API polling until WS reconnects.")

                # --- Get Global State (e.g., Account Balance) Periodically ---
                current_time = time.time()
                if not self.dry_run and (current_time - last_balance_check_time > balance_check_interval):
                    log_console(logging.DEBUG, f"Performing periodic balance check (Interval: {balance_check_interval}s)...")
                    fetched_balance = get_available_balance(self.session, "USDT", self.account_type)
                    # Log if balance changed significantly or if it was previously zero/unavailable
                    if abs(fetched_balance - balance) > 0.01 or balance <= FLOAT_COMPARISON_TOLERANCE:
                        log_console(logging.INFO, f"Periodic Balance Check ({self.account_type}): {fetched_balance:.2f} USDT Available")
                    balance = fetched_balance
                    last_balance_check_time = current_time
                    # Warn if balance becomes zero during operation
                    if balance <= FLOAT_COMPARISON_TOLERANCE:
                        log_console(logging.WARNING, f"{self.account_type} USDT balance is zero. No new trades can be sized.")

                # --- Balance Allocation per Trader ---
                num_active_traders = len(self.traders)
                # Allocate balance only if positive and traders exist
                balance_per_trader = (balance / num_active_traders) if num_active_traders > 0 and balance > FLOAT_COMPARISON_TOLERANCE else 0.0

                # Calculate Max Position Size per Trader (Optional)
                max_pos_usdt_config = self.risk_cfg.get("MAX_POSITION_USDT")
                max_pos_usdt_per_trader: Optional[float] = None
                if isinstance(max_pos_usdt_config, (int, float)) and max_pos_usdt_config > FLOAT_COMPARISON_TOLERANCE and num_active_traders > 0:
                    max_pos_usdt_per_trader = max_pos_usdt_config / num_active_traders
                    log_console(logging.DEBUG, f"Allocated Balance/Trader: {balance_per_trader:.2f} USDT, Max Position Value/Trader: {max_pos_usdt_per_trader:.2f} USDT")
                elif balance_per_trader > FLOAT_COMPARISON_TOLERANCE:
                    log_console(logging.DEBUG, f"Allocated Balance/Trader: {balance_per_trader:.2f} USDT, Max Position Value/Trader: Unlimited")

                # --- Iterate Through Configured Symbols ---
                # Use list(self.traders.keys()) for safe iteration if traders could potentially be removed dynamically (though not implemented here)
                for symbol in list(self.traders.keys()):
                    if not self.running or self.shutdown_requested: break # Check shutdown flag frequently

                    trader = self.traders[symbol]
                    symbol_color_prefix = f"{get_symbol_color(symbol)}[{symbol}]{Style.RESET_ALL}"
                    print(f"{symbol_color_prefix} --- Processing Symbol ---")

                    try:
                        # --- Cooldown Check ---
                        # Skip processing if the bot recently exited a position for this symbol
                        if trader.last_exit_time:
                            cooldown_seconds = trader.bot_cfg.get("COOLDOWN_PERIOD_SECONDS", 300)
                            if cooldown_seconds > 0:
                                time_since_exit = (current_dt_utc - trader.last_exit_time).total_seconds()
                                if time_since_exit < cooldown_seconds:
                                    remaining_cooldown = cooldown_seconds - time_since_exit
                                    log_console(logging.INFO, f"Cooldown active for {symbol} ({remaining_cooldown:.0f}s remaining). Skipping cycle.", symbol=symbol)
                                    continue # Skip to the next symbol
                                else:
                                    # Cooldown period has finished, reset the timer
                                    log_console(logging.DEBUG, f"Cooldown finished for {symbol}.", symbol=symbol)
                                    trader.last_exit_time = None

                        # --- Get Latest Data & Indicators ---
                        # This method handles WS/REST switching and indicator calculation internally
                        df_with_indicators = trader.get_ohlcv()

                        # Validate data availability and minimum length for signal generation
                        # Need at least 2 rows for iloc[-2] (last closed) and iloc[-1] (latest) access
                        min_rows_for_signals = 2
                        if df_with_indicators is None or df_with_indicators.empty or len(df_with_indicators) < min_rows_for_signals:
                            log_console(logging.DEBUG, f"Insufficient data ({len(df_with_indicators) if df_with_indicators is not None else 0} rows < {min_rows_for_signals}) or calculation failed for {symbol}. Skipping cycle.", symbol=symbol)
                            continue

                        # --- Access Candle Data Safely ---
                        try:
                            # Use iloc for integer-based indexing, robust to non-sequential DatetimeIndex gaps
                            # [-2] is the last fully closed candle (signals are usually based on closed candles)
                            # [-1] is the latest (potentially incomplete) candle (used for current price estimate)
                            last_closed_candle = df_with_indicators.iloc[-2]
                            latest_candle = df_with_indicators.iloc[-1]
                        except IndexError:
                            # This should be caught by the length check above, but safeguard
                            log_console(logging.WARNING, f"Index error accessing candle data for {symbol} (need >= {min_rows_for_signals} rows, got {len(df_with_indicators)}). Skipping cycle.", symbol=symbol)
                            continue

                        # --- Validate Key Indicator Data ---
                        # Check for NaNs in critical indicators on the candle used for signals
                        critical_indicators = ["close", "atr_risk", "long_signal", "short_signal", "exit_long_signal", "exit_short_signal"]
                        if any(pd.isna(last_closed_candle.get(ind)) for ind in critical_indicators):
                            log_console(logging.WARNING, f"Critical indicator data contains NaN on the last closed candle ({last_closed_candle.name.strftime('%Y-%m-%d %H:%M')}) for {symbol}. Skipping cycle.", symbol=symbol)
                            continue

                        # Get current price estimate (use latest close, fallback to last closed close)
                        current_price = latest_candle.get("close")
                        if pd.isnull(current_price) or current_price <= FLOAT_COMPARISON_TOLERANCE:
                            current_price = last_closed_candle.get("close") # Fallback to previous close
                            if pd.isnull(current_price) or current_price <= FLOAT_COMPARISON_TOLERANCE:
                                log_console(logging.ERROR, f"Cannot determine a valid current price for {symbol}. Skipping cycle.", symbol=symbol)
                                continue # Cannot proceed without a price estimate

                        # --- Position Management ---
                        # Check current position status (real or simulated)
                        position = trader.get_position(log_error=True) # Log errors on position fetch attempts

                        if position:
                            # === Manage Existing Position ===
                            pos_side = position.get("side")
                            pos_size = position.get("size") # float
                            entry_price = position.get("avgPrice") # float

                            # --- Validate Retrieved Position Data ---
                            # Re-check essential fields from the retrieved position data
                            min_significant_qty_check = max(FLOAT_COMPARISON_TOLERANCE, trader.min_order_qty * 0.01, trader.qty_step_float * 0.1)
                            if not (pos_side in ["Buy", "Sell"] and isinstance(pos_size, float) and abs(pos_size) >= min_significant_qty_check and isinstance(entry_price, float) and entry_price > FLOAT_COMPARISON_TOLERANCE):
                                log_console(logging.ERROR, f"Invalid or inconsistent position data retrieved for {symbol}: {position}. Clearing internal state and skipping cycle.", symbol=symbol)
                                trader.current_trade = None # Clear potentially bad internal state
                                position = None # Treat as no position for this cycle
                                continue # Skip to next symbol to avoid acting on bad data

                            pos_size_abs = abs(pos_size)
                            # Log current position details
                            log_console(
                                logging.INFO,
                                f"Active Position Found: Side={pos_side}, Size={pos_size_abs:.{trader.qty_precision}f}, Entry Price={entry_price:.{trader.price_precision}f}",
                                symbol=symbol,
                            )

                            # --- Check Exit Conditions (Based on Last Closed Candle) ---
                            exit_reason: Optional[str] = None
                            exit_long_sig = last_closed_candle.get("exit_long_signal", False)
                            exit_short_sig = last_closed_candle.get("exit_short_signal", False)

                            if pos_side == "Buy" and exit_long_sig:
                                exit_reason = "Strategy Exit Signal (Long)"
                            elif pos_side == "Sell" and exit_short_sig:
                                exit_reason = "Strategy Exit Signal (Short)"

                            # --- Trailing Stop Loss Check (Placeholder) ---
                            # TODO: Implement Trailing Stop Loss logic here if enabled in config
                            # if trader.risk_cfg.get("ENABLE_TRAILING_STOP", False):
                            #    tsl_triggered, tsl_price = check_trailing_stop(position, current_price, trader.risk_cfg)
                            #    if tsl_triggered:
                            #         exit_reason = f"Trailing Stop Hit ({tsl_price:.{trader.price_precision}f})"

                            # --- Execute Close Order if Exit Condition Met ---
                            if exit_reason:
                                log_console(logging.INFO, f"Exit condition triggered for {symbol}: {exit_reason}. Attempting to close position.", symbol=symbol)
                                # Attempt to close the position using the trader method
                                # Pass the retrieved position data and estimated current price for metrics fallback
                                if trader.close_position(position, exit_reason=exit_reason, exit_price_est=current_price):
                                    log_console(logging.INFO, f"Position for {symbol} closed successfully based on signal.", symbol=symbol)
                                    # Cooldown is applied automatically within close_position on success
                                else:
                                    # Log critical error if closure fails after signal was detected
                                    log_console(logging.ERROR, f"CRITICAL: Failed to close position for {symbol} after exit signal '{exit_reason}'. Manual intervention may be required!", symbol=symbol)
                                # Continue to the next symbol after attempting closure (whether successful or not)
                                continue
                            else:
                                # No exit signal, hold the position
                                log_console(logging.DEBUG, f"Holding {pos_side} position for {symbol}. No exit signal detected.", symbol=symbol)

                        else: # No Position Open
                            # === Check Entry Conditions ===
                            log_console(logging.DEBUG, f"No active position for {symbol}. Checking for entry signals...", symbol=symbol)

                            # --- Higher Timeframe Filter ---
                            # Check HTF trend first if enabled in config
                            if trader.strategy_cfg.get("ENABLE_MULTI_TIMEFRAME", False):
                                htf_allows_trade = trader.get_higher_tf_trend()
                                if not htf_allows_trade:
                                    log_console(logging.INFO, f"HTF filter is unfavorable for {symbol}. Skipping entry check.", symbol=symbol)
                                    continue # Skip entry if HTF filter blocks

                            # --- Get Signals and Parameters from Last Closed Candle ---
                            atr_val = last_closed_candle.get("atr_risk", 0.0) # ATR for risk calculation
                            long_signal = last_closed_candle.get("long_signal", False)
                            short_signal = last_closed_candle.get("short_signal", False)
                            long_strength = last_closed_candle.get('long_signal_strength', 0.0)
                            short_strength = last_closed_candle.get('short_signal_strength', 0.0)

                            # Get risk parameters from configuration
                            risk_percent = float(trader.risk_cfg.get("RISK_PER_TRADE_PERCENT", 1.0))
                            sl_mult = float(trader.risk_cfg.get("ATR_MULTIPLIER_SL", 1.5))
                            tp_mult = float(trader.risk_cfg.get("ATR_MULTIPLIER_TP", 2.0))
                            min_strength_threshold = float(trader.strategy_cfg.get("MIN_SIGNAL_STRENGTH", 0.5))

                            # Initialize variables for potential entry
                            entry_side: Optional[str] = None
                            stop_loss_price: Optional[float] = None
                            take_profit_price: Optional[float] = None
                            signal_strength: float = 0.0
                            # Use the latest price as the estimated entry for SL/TP calculation
                            entry_price_est = current_price

                            # --- Determine Entry Side and Calculate SL/TP ---
                            # Check Long Signal
                            if long_signal and long_strength >= min_strength_threshold:
                                entry_side = "Buy"
                                signal_strength = long_strength
                                # Calculate SL based on ATR (only if ATR is valid)
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and sl_mult > FLOAT_COMPARISON_TOLERANCE:
                                     stop_loss_price = entry_price_est - (atr_val * sl_mult)
                                # Calculate TP (optional, only if multiplier > 0 and ATR valid)
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and tp_mult > FLOAT_COMPARISON_TOLERANCE:
                                     take_profit_price = entry_price_est + (atr_val * tp_mult)
                            # Check Short Signal (only if no long signal)
                            elif short_signal and short_strength >= min_strength_threshold:
                                entry_side = "Sell"
                                signal_strength = short_strength
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and sl_mult > FLOAT_COMPARISON_TOLERANCE:
                                     stop_loss_price = entry_price_est + (atr_val * sl_mult)
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and tp_mult > FLOAT_COMPARISON_TOLERANCE:
                                     take_profit_price = entry_price_est - (atr_val * tp_mult)

                            # --- Proceed if Entry Signal Found ---
                            if entry_side:
                                log_console(logging.DEBUG,f"Potential Entry Signal Detected: Side={entry_side}, Strength={signal_strength:.2f}, Est. Entry={entry_price_est:.{trader.price_precision}f}", symbol=symbol)

                                # --- Validate Calculated SL/TP ---
                                if stop_loss_price is None or stop_loss_price <= FLOAT_COMPARISON_TOLERANCE:
                                    log_console(logging.WARNING, f"Invalid Stop Loss calculated ({stop_loss_price}) for {entry_side} entry on {symbol} (ATR={atr_val:.{trader.price_precision}f}, Mult={sl_mult}). Skipping entry.", symbol=symbol)
                                    continue # Skip entry if SL is invalid

                                # Validate TP: ensure it's positive and logical relative to entry/SL
                                if take_profit_price is not None:
                                    if take_profit_price <= FLOAT_COMPARISON_TOLERANCE:
                                        log_console(logging.DEBUG, f"Invalid Take Profit calculated ({take_profit_price}) for {symbol}. TP will not be set.", symbol=symbol)
                                        take_profit_price = None # Do not set invalid TP
                                    # Check if TP is on the wrong side of SL
                                    elif entry_side == "Buy" and take_profit_price <= stop_loss_price:
                                        log_console(logging.WARNING, f"Calculated Long TP ({take_profit_price:.{trader.price_precision}f}) is below or at SL ({stop_loss_price:.{trader.price_precision}f}) for {symbol}. TP will not be set.", symbol=symbol)
                                        take_profit_price = None
                                    elif entry_side == "Sell" and take_profit_price >= stop_loss_price:
                                        log_console(logging.WARNING, f"Calculated Short TP ({take_profit_price:.{trader.price_precision}f}) is above or at SL ({stop_loss_price:.{trader.price_precision}f}) for {symbol}. TP will not be set.", symbol=symbol)
                                        take_profit_price = None

                                # --- Calculate Position Size ---
                                # Ensure SL distance is valid for calculation
                                sl_distance_price_abs = abs(entry_price_est - stop_loss_price)
                                if sl_distance_price_abs <= FLOAT_COMPARISON_TOLERANCE:
                                     log_console(logging.WARNING, f"Stop Loss distance is zero or negative ({sl_distance_price_abs:.{trader.price_precision}f}) for {symbol}. Cannot size trade. Skipping entry.", symbol=symbol)
                                     continue

                                # Calculate size using the dedicated helper function
                                qty_to_trade = calculate_position_size_atr(
                                    balance=balance_per_trader, # Use allocated balance
                                    risk_percent=risk_percent,
                                    sl_distance_price=sl_distance_price_abs,
                                    entry_price=entry_price_est,
                                    min_order_qty=trader.min_order_qty,
                                    qty_step_float=trader.qty_step_float,
                                    qty_precision=trader.qty_precision,
                                    max_position_usdt=max_pos_usdt_per_trader # Pass optional constraint
                                )

                                # --- Place Order if Size is Valid ---
                                if qty_to_trade > FLOAT_COMPARISON_TOLERANCE:
                                    # Log the intended trade details clearly before placing
                                    tp_log_str = f"{take_profit_price:.{trader.price_precision}f}" if take_profit_price else 'None'
                                    sl_log_str = f
