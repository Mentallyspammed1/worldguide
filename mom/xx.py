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
- Momentum indicators (EMA/SuperSmoother, ROC, RSI, ADX).
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
from queue import Queue, Empty                                         from copy import deepcopy
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
except ImportError:                                                        print("Error: 'pybit' library not found or out of date. Please install/upgrade it: pip install -U pybit")
    sys.exit(1)                                                        try:                                                                       from colorama import init as colorama_init, Fore, Style, Back  # Colored console output
except ImportError:                                                        print("Error: 'colorama' library not found. Please install it: pip install colorama")
    sys.exit(1)
try:                                                                       from dotenv import load_dotenv  # Load environment variables from .env file
except ImportError:
    print("Error: 'python-dotenv' library not found. Please install it: pip install python-dotenv")                                               sys.exit(1)

# --- Load Environment Variables ---                                   # Ensure you have a .env file in the same directory or environment variables set:
# BYBIT_API_KEY=YOUR_API_KEY                                           # BYBIT_API_SECRET=YOUR_API_SECRET                                     load_dotenv()
                                                                       # Initialize Colorama for neon-colored console output (auto-resets style)
colorama_init(autoreset=True)                                          
# --- Constants ---                                                    DEFAULT_CONFIG_FILE = "config.json"
LOG_FILE = "momentum_scanner_bot_enhanced.log"                         METRICS_LOG_FILE = "trade_performance.log"
# Bybit V5 Kline limits                                                MAX_KLINE_LIMIT_PER_REQUEST = 1000 # Maximum klines fetchable in one V5 API call (check Bybit docs for exact value)                           MIN_KLINE_RECORDS_FOR_CALC = 2 # Minimum records needed for most basic calculations (e.g., diff)                                              # Used for floating point comparisons                                  FLOAT_COMPARISON_TOLERANCE = 1e-9
                                                                       # --- Logging Setup ---

# Base Logger Configuration (File Handler, no colors in file, UTF-8 encoded)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"                                                       )
# Use 'w' mode to overwrite log file on each run, or 'a' to append     log_file_mode = "a" # Change to 'w' if needed                          file_handler = logging.FileHandler(LOG_FILE, mode=log_file_mode, encoding='utf-8')
file_handler.setFormatter(log_formatter)                                                                                                      # Main application logger
logger = logging.getLogger("MomentumScannerTrader")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)  # Default level, can be overridden by --debug
logger.propagate = False  # Prevent duplicate logs if root logger is configured

# Metrics Logger Configuration (CSV-like format, UTF-8 encoded)
metrics_formatter = logging.Formatter("%(asctime)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S")                                                 metrics_handler = logging.FileHandler(METRICS_LOG_FILE, mode="a", encoding='utf-8')                                                           metrics_handler.setFormatter(metrics_formatter)
                                                                       # Metrics logger instance
metrics_logger = logging.getLogger("TradeMetrics")                     metrics_logger.addHandler(metrics_handler)
metrics_logger.setLevel(logging.INFO)                                  metrics_logger.propagate = False

# --- Console Logging Function ---                                                                                                            # Neon-Colored Console Output with Symbol Support
SYMBOL_COLORS = {
    # Add specific symbols for distinct colors or use a cycle              "DOTUSDT": Fore.CYAN + Style.BRIGHT,
    "TRUMPUSDT": Fore.YELLOW + Style.BRIGHT,
    "BCHUSDT": Fore.MAGENTA + Style.BRIGHT,                                "DOGEUSDT": Fore.BLUE + Style.BRIGHT,
    "default": Fore.WHITE,
}                                                                      _symbol_color_cycle = [Fore.CYAN, Fore.YELLOW, Fore.MAGENTA, Fore.BLUE, Fore.GREEN, Fore.LIGHTRED_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTYELLOW_EX]
_symbol_color_map: Dict[str, str] = {}                                 _color_index = 0

def get_symbol_color(symbol: Optional[str]) -> str:                        """Gets a consistent color for a symbol, cycling through colors if needed."""
    global _color_index                                                    if not symbol:
        return SYMBOL_COLORS["default"]
    if symbol in SYMBOL_COLORS:
        return SYMBOL_COLORS[symbol]                                       if symbol not in _symbol_color_map:                                        color = _symbol_color_cycle[_color_index % len(_symbol_color_cycle)] + Style.BRIGHT                                                           _symbol_color_map[symbol] = color
        _color_index += 1
    return _symbol_color_map[symbol]                                   
def log_console(level: int, message: Any, symbol: Optional[str] = None, exc_info: bool = False, *args, **kwargs):
    """
    Logs messages to the console with level-specific colors and optional
    symbol highlighting. Also forwards the message to the file logger.
                                                                           Args:                                                                      level: Logging level (e.g., logging.INFO, logging.ERROR).
        message: The message object to log (will be converted to string).                                                                             symbol: Optional symbol name to include with specific color coding.                                                                           exc_info: If True, add exception information to the file log.          *args: Additional arguments for string formatting (if message is a format string).                                                            **kwargs: Additional keyword arguments for string formatting.
    """                                                                    neon_colors = {
        logging.DEBUG: Fore.CYAN,                                              logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW + Style.BRIGHT,                           logging.ERROR: Fore.RED + Style.BRIGHT,
        logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,            }
    color = neon_colors.get(level, Fore.WHITE)                             level_name = logging.getLevelName(level)                               symbol_color = get_symbol_color(symbol)
    symbol_prefix = f"{symbol_color}[{symbol}]{Style.RESET_ALL} " if symbol else ""
    prefix = f"{color}{level_name}:{Style.RESET_ALL} {symbol_prefix}"  
    # Ensure message is a string before formatting                         if not isinstance(message, str):
        message_str = str(message)                                         else:
        message_str = message                                          
    # Format message safely if args or kwargs are provided                 formatted_message = message_str  # Default to stringified message
    try:
        # Attempt format only if args or kwargs are present and message was originally a string                                                       if (args or kwargs) and isinstance(message, str):
            formatted_message = message.format(*args, **kwargs)
    except Exception as fmt_e:                                                 # Log formatting errors without crashing, include original message
        print(f"{Fore.RED}LOG FORMATTING ERROR:{Style.RESET_ALL} {fmt_e} | Original Msg: {message_str}")
        logger.error(f"Log formatting error: {fmt_e} | Original Msg: {message_str}", exc_info=False)                                                  # Use the unformatted string message for console output
        formatted_message = message_str
                                                                           # Print formatted (or stringified) message to console
    print(prefix + formatted_message)
                                                                           # Log to file (without colors, with symbol if provided, using formatted message)
    file_message = f"[{symbol}] {formatted_message}" if symbol else formatted_message
    # Include exception info in file log if level is ERROR or higher, or explicitly requested
    should_log_exc_info = exc_info or (level >= logging.ERROR)
    logger.log(level, file_message, exc_info=should_log_exc_info)

def log_metrics(message: str):
    """
    Logs trade performance metrics to the metrics file and prints to console.

    Args:                                                                      message: The metrics message string (should be comma-separated).
    """                                                                    metrics_logger.info(message)
    print(f"{Fore.MAGENTA+Style.BRIGHT}METRICS:{Style.RESET_ALL} {message}")


# --- Load API Keys from Environment Variables ---
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

# Check if API keys are loaded
if not API_KEY or not API_SECRET:
    # Use log_console for consistency, even at critical level before full logging setup
    log_console(logging.CRITICAL, "BYBIT_API_KEY or BYBIT_API_SECRET not found in environment variables or .env file.")
    sys.exit(1)
else:
    log_console(logging.INFO, "API keys loaded successfully.")


# --- Configuration Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the JSON configuration file, validates essential parts, and calculates
    the required KLINE_LIMIT for each symbol based on indicator periods.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        The loaded configuration dictionary.

    Raises:
        SystemExit: If the config file is missing, invalid, or lacks essential keys.
    """
    log_console(logging.INFO, f"Using configuration file: {config_path}")
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            config = json.load(f)
        log_console(logging.INFO, f"Loaded configuration from {config_path}")

        # --- Calculate KLINE_LIMIT per Symbol ---
        bybit_cfg = config.get("BYBIT_CONFIG", {})
        symbols_cfg = bybit_cfg.get("SYMBOLS", [])
        bot_cfg = config.get("BOT_CONFIG", {})
        # Default buffer, ensures enough data for indicator warmup
        kline_limit_buffer = bot_cfg.get("KLINE_LIMIT_BUFFER", 50)     
        if not isinstance(symbols_cfg, list):
            raise ValueError("'BYBIT_CONFIG.SYMBOLS' must be a list.")

        for symbol_cfg in symbols_cfg:
            if not isinstance(symbol_cfg, dict) or "SYMBOL" not in symbol_cfg:
                raise ValueError("Each item in 'SYMBOLS' must be a dictionary with a 'SYMBOL' key.")                                          
            symbol = symbol_cfg["SYMBOL"]
            strategy_cfg = symbol_cfg.get("STRATEGY_CONFIG", {})
            if not isinstance(strategy_cfg, dict):
                log_console(logging.WARNING, f"Missing or invalid 'STRATEGY_CONFIG' for {symbol}. Using defaults.", symbol=symbol)
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
                periods.extend(periods) # Double the list to account for HTF potentially needing same max period

            # Filter out invalid or non-numeric periods
            valid_periods = [p for p in periods if isinstance(p, (int, float)) and p > 0]
            if not valid_periods:
                log_console(logging.WARNING, f"No valid indicator periods found for {symbol}. Using default max period (200).", symbol=symbol)
                max_period = 200  # Default fallback if no periods found
            else:
                max_period = max(valid_periods)

            # Ensure kline limit is at least a minimum value for basic operations + buffer
            # Add extra buffer for complex indicators like ADX which require more warmup
            adx_warmup_buffer = 2 * strategy_cfg.get("ADX_PERIOD", 14) if strategy_cfg.get("ADX_PERIOD", 14) > 0 else 30
            calculated_limit = int(max_period + kline_limit_buffer + adx_warmup_buffer)
            kline_limit = max(MIN_KLINE_RECORDS_FOR_CALC + kline_limit_buffer, calculated_limit)

            # Cap the limit at a reasonable maximum to avoid excessive memory/API usage
            # MAX_KLINE_LIMIT_PER_REQUEST is the API fetch limit, internal limit can be higher for rolling calcs
            internal_max_limit = 2 * MAX_KLINE_LIMIT_PER_REQUEST # Allow storing more than one fetch worth
            kline_limit = min(kline_limit, internal_max_limit)

            # Store calculated limit within the symbol's config (internal use)
            symbol_cfg.setdefault("INTERNAL", {})["KLINE_LIMIT"] = kline_limit
            log_console(logging.DEBUG, f"Calculated KLINE_LIMIT for {symbol}: {kline_limit} (Max Period: {max_period:.0f}, Buffer: {kline_limit_buffer}, ADX Buf: {adx_warmup_buffer})", symbol=symbol)

        # --- Detailed Configuration Validation (Performed later in main block) ---

        return config

    except FileNotFoundError:
        log_console(logging.CRITICAL, f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_console(logging.CRITICAL, f"Error decoding JSON configuration file {config_path}: {e}")
        sys.exit(1)
    except ValueError as e:
        log_console(logging.CRITICAL, f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        log_console(logging.CRITICAL, f"Unexpected error loading configuration: {e}", exc_info=True)
        sys.exit(1)


# --- Custom Indicator Functions (Ehlers) ---
# Note: These are complex filters. Ensure parameters and data quality are good.
# Consider using established libraries if complex filtering is critical.
def super_smoother(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates Ehlers Super Smoother filter for a given series.
    Requires minimum period of 2. Handles NaNs robustly.

    Args:
        series: pandas Series of price data.
        period: Lookback period for the filter (must be >= 2).

    Returns:
        pandas Series with Super Smoother values, or Series of NaNs on error/invalid input.
    """
    if not isinstance(series, pd.Series) or series.empty:
        log_console(logging.DEBUG, "Super Smoother: Input series is invalid or empty.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)
    if not isinstance(period, int) or period < 2:
        log_console(logging.WARNING, f"Super Smoother: Period must be an integer >= 2, got {period}. Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2:
        log_console(logging.DEBUG, f"Super Smoother: Series length ({len(series_cleaned)} non-NaN) is less than required minimum (2). Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        a1 = math.exp(-math.pi * math.sqrt(2.0) / float(period))
        b1 = 2.0 * a1 * math.cos(math.pi * math.sqrt(2.0) / float(period))
        c1 = -(a1 * a1)
        coeff1 = (1.0 - b1 - c1) / 2.0

        # Use float, ffill cautiously to handle initial NaNs if present
        series_float = series.astype(float).ffill()
        ss = np.full(len(series_float), np.nan, dtype=np.float64)

        # Initialize first two values robustly after ffill
        if len(series_float) > 0 and not np.isnan(series_float.iloc[0]): ss[0] = series_float.iloc[0]
        if len(series_float) > 1 and not np.isnan(series_float.iloc[1]): ss[1] = series_float.iloc[1]

        # Filter loop - check for NaNs rigorously
        for i in range(2, len(series_float)):
            # Check all required inputs for NaN before calculation
            # Check current price inputs and previous SS outputs
            if np.isnan(series_float.iloc[i]) or np.isnan(series_float.iloc[i - 1]) or \
               np.isnan(ss[i - 1]) or np.isnan(ss[i - 2]):
                # If any input is NaN, the result for this step is NaN.
                # ss[i] remains NaN as initialized.
                continue

            prev_ss = ss[i - 1]
            prev_prev_ss = ss[i - 2]
            current_price_avg = (series_float.iloc[i] + series_float.iloc[i - 1]) / 2.0
            ss[i] = coeff1 * current_price_avg + b1 * prev_ss + c1 * prev_prev_ss

        return pd.Series(ss, index=series.index, dtype=np.float64)
    except Exception as e:                                                     log_console(logging.ERROR, f"Super Smoother calculation error for period {period}: {e}", exc_info=True)
        return pd.Series(np.nan, index=series.index, dtype=np.float64) 

def instantaneous_trendline(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates Ehlers Instantaneous Trendline for a given series.
    Requires minimum period of 4. Uses a fixed alpha as recommended by Ehlers.
    Handles NaNs robustly.

    Args:
        series: pandas Series of price data.
        period: Lookback period (dominant cycle period, must be >= 4).

    Returns:
        pandas Series with Instantaneous Trendline values, or Series of NaNs on error/invalid input.
    """
    if not isinstance(series, pd.Series) or series.empty:
        log_console(logging.DEBUG, "Instantaneous Trendline: Input series is invalid or empty.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)
    # Ehlers IT generally needs more points for reliable calculation
    required_min_period = 4
    if not isinstance(period, int) or period < required_min_period:
        log_console(logging.WARNING, f"Instantaneous Trendline: Period must be an integer >= {required_min_period}, got {period}. Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length
    series_cleaned = series.dropna()
    min_len_needed = 4 # Based on formula dependencies
    if len(series_cleaned) < min_len_needed:
        log_console(logging.DEBUG, f"Instantaneous Trendline: Series length ({len(series_cleaned)} non-NaN) is less than required minimum ({min_len_needed}). Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        # Fixed alpha as per Ehlers' recommendation (adjust if needed based on source/testing)
        alpha = 0.07

        # Use float, ffill cautiously
        series_float = series.astype(float).ffill()
        it = np.full(len(series_float), np.nan, dtype=np.float64)

        # Initialize first few values robustly based on Ehlers common practice after ffill
        # Check source values are not NaN before using them
        if len(series_float) > 0 and not np.isnan(series_float.iloc[0]): it[0] = series_float.iloc[0]
        if len(series_float) > 1 and not np.isnan(series_float.iloc[1]): it[1] = series_float.iloc[1]
        # Requires indices 0, 1, 2
        if len(series_float) > 2 and not np.any(np.isnan(series_float.iloc[0:3])):
            it[2] = (series_float.iloc[2] + 2.0 * series_float.iloc[1] + series_float.iloc[0]) / 4.0
        # Requires indices 1, 2, 3                                             if len(series_float) > 3 and not np.any(np.isnan(series_float.iloc[1:4])):
             it[3] = (series_float.iloc[3] + 2.0 * series_float.iloc[2] + series_float.iloc[1]) / 4.0

        # Calculation loop - check for NaNs rigorously
        # Starts from index 4, as it[3] depends on it[2] and it[1]
        for i in range(4, len(series_float)):
            # Check all required price inputs and previous IT outputs
            if np.isnan(series_float.iloc[i]) or np.isnan(series_float.iloc[i - 1]) or \
               np.isnan(series_float.iloc[i - 2]) or np.isnan(it[i - 1]) or np.isnan(it[i - 2]):
                # If any input is NaN, the result for this step is NaN.
                # it[i] remains NaN as initialized.
                continue

            price_i = series_float.iloc[i]
            price_im1 = series_float.iloc[i - 1]
            price_im2 = series_float.iloc[i - 2]
            it_im1 = it[i - 1]
            it_im2 = it[i - 2]

            # Formula from Ehlers' papers/code (verify against trusted source if possible):
            it[i] = (alpha - alpha**2 / 4.0) * price_i + \
                    (alpha**2 / 2.0) * price_im1 - \
                    (alpha - 3.0 * alpha**2 / 4.0) * price_im2 + \
                    2.0 * (1.0 - alpha) * it_im1 - \
                    (1.0 - alpha)**2 * it_im2

        return pd.Series(it, index=series.index, dtype=np.float64)
    except Exception as e:
        log_console(logging.ERROR, f"Instantaneous Trendline calculation error for period {period}: {e}", exc_info=True)
        return pd.Series(np.nan, index=series.index, dtype=np.float64)


# --- Indicator Calculation ---
def calculate_indicators_momentum(df: pd.DataFrame, strategy_cfg: Dict[str, Any], config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Calculates momentum-based technical indicators, Ehlers filters (optional),
    and generates unified entry/exit signals based on the strategy configuration.

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
    kline_limit = max(MIN_KLINE_RECORDS_FOR_CALC, internal_cfg.get("KLINE_LIMIT", 120))
    required_cols = ["open", "high", "low", "close", "volume"]

    if df is None or df.empty:
        log_console(logging.DEBUG, "Indicator Calc: Input DataFrame is empty.")
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            log_console(logging.DEBUG, "Indicator Calc: DataFrame index is not DatetimeIndex, attempting conversion.")
            # Convert index to datetime, coercing errors, setting timezone to UTC
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            if not isinstance(df.index, pd.DatetimeIndex):  # Check conversion success
                raise ValueError("Index conversion to DatetimeIndex failed.")
            if df.index.isnull().any():
                log_console(logging.WARNING, "Indicator Calc: Found NaT values in index after conversion. Dropping affected rows.")
                df.dropna(axis=0, how='any', subset=required_cols, inplace=True) # Drop rows with NaT index
                if df.empty:
                    log_console(logging.ERROR, "Indicator Calc: DataFrame empty after dropping rows with NaT index.")
                    return None
        except Exception as idx_e:
            log_console(logging.ERROR, f"Indicator Calc: Failed to convert DataFrame index to DatetimeIndex: {idx_e}")
            return None

    # Use a copy to avoid modifying the original DataFrame passed to the function
    df_out = df.copy()

    # Attempt numeric conversion and check length *after* potential NaN drop
    try:
        for col in required_cols:
            if col not in df_out.columns:
                log_console(logging.ERROR, f"Indicator Calc: Missing required column: '{col}'.")
                return None  # Critical column missing
            if not pd.api.types.is_numeric_dtype(df_out[col]):
                log_console(logging.DEBUG, f"Indicator Calc: Converting non-numeric column '{col}' to numeric.")
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

        # Drop rows where essential conversions failed (resulted in NaNs)
        initial_len = len(df_out)
        df_out.dropna(subset=required_cols, inplace=True)
        if len(df_out) < initial_len:
            log_console(logging.WARNING, f"Indicator Calc: Dropped {initial_len - len(df_out)} rows due to NaN values in required columns after numeric conversion.")

        # Check length requirements AFTER cleaning numeric data
        if len(df_out) < kline_limit:
            log_console(logging.DEBUG, f"Indicator Calc: Insufficient valid data points after NaN drop ({len(df_out)} < {kline_limit}). Need more historical data.")
            return None
        if df_out.empty:
            log_console(logging.ERROR, "Indicator Calc: DataFrame empty after NaN drop from numeric conversion.")
            return None

    except Exception as e:
        log_console(logging.ERROR, f"Indicator Calc: Failed during data validation/numeric conversion: {e}", exc_info=True)
        return None

    try:
        price_source_key = strategy_cfg.get("PRICE_SOURCE", "close")
        # Handle common price source calculations if needed (e.g., hlc3)
        if price_source_key == 'hlc3':
            df_out['hlc3'] = (df_out['high'] + df_out['low'] + df_out['close']) / 3.0
            price_source_col = 'hlc3'
        elif price_source_key == 'ohlc4':
             df_out['ohlc4'] = (df_out['open'] + df_out['high'] + df_out['low'] + df_out['close']) / 4.0
             price_source_col = 'ohlc4'
        elif price_source_key in df_out.columns:
            price_source_col = price_source_key
        else:
            log_console(logging.ERROR, f"Indicator Calc: Specified price source '{price_source_key}' not found or calculable. Falling back to 'close'.")
            if 'close' not in df_out.columns: return None # Cannot proceed without 'close'
            price_source_col = 'close'

        if df_out[price_source_col].isnull().all():                                log_console(logging.ERROR, f"Indicator Calc: Price source column '{price_source_col}' contains only NaNs.")
            return None
                                                                               use_ehlers = strategy_cfg.get("USE_EHLERS_SMOOTHER", False)

        # --- Moving Averages / Smoothers ---                                  smoother_params = [                                                        ("ULTRA_FAST_EMA_PERIOD", "ema_ultra"),
            ("FAST_EMA_PERIOD", "ema_fast"),
            ("MID_EMA_PERIOD", "ema_mid"),
            ("SLOW_EMA_PERIOD", "ema_slow"),
        
        smoother_prefix = "ss" if use_ehlers else "ema"

        for period_key, base_col_name in smoother_params:
            length = strategy_cfg.get(period_key)
            col_name = base_col_name.replace("ema", smoother_prefix)
            if not isinstance(length, int) or length <= 0:
                log_console(logging.ERROR, f"Indicator Calc: Invalid or missing {period_key} in strategy config. Value: {length}. Skipping {col_name}.")
                df_out[col_name] = np.nan
                continue

            try:
                if use_ehlers:
                    indicator_series = super_smoother(df_out[price_source_col], length)
                else:
                    indicator_series = ta.ema(df_out[price_source_col], length=length)

                if indicator_series is None or indicator_series.empty or indicator_series.isnull().all():
                    log_console(logging.WARNING, f"Indicator Calc: {col_name} (length {length}) calculation resulted in empty or all NaNs.")
                    df_out[col_name] = np.nan
                else:
                    df_out[col_name] = indicator_series
            except Exception as calc_e:
                log_console(logging.ERROR, f"Indicator Calc: Error calculating {col_name} (length {length}): {calc_e}", exc_info=False)
                df_out[col_name] = np.nan

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
        try:
            roc_period = strategy_cfg.get("ROC_PERIOD", 5)
            if roc_period > 0:
                df_out["roc"] = ta.roc(df_out[price_source_col], length=roc_period)
                # Fill NaNs before smoothing ROC, use 0 as neutral momentum
                df_out["roc_smooth"] = ta.sma(df_out["roc"].fillna(0.0), length=3)
            else: raise ValueError("ROC_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating ROC: {e}", exc_info=False)
            df_out["roc"] = np.nan; df_out["roc_smooth"] = np.nan

        try:
            atr_period_risk = strategy_cfg.get("ATR_PERIOD_RISK", 14)
            if atr_period_risk > 0:
                atr_series = ta.atr(df_out["high"], df_out["low"], df_out["close"], length=atr_period_risk)
                # Use .ffill() as per pandas_ta recommendation, then fill remaining with 0
                df_out["atr_risk"] = atr_series.ffill().fillna(0.0)
                # Normalized ATR
                # Avoid division by zero/NaN: replace 0 with NaN, ffill, then fill remaining start NaNs with 1
                close_safe = df_out["close"].replace(0, np.nan).ffill().fillna(1.0)
                df_out["norm_atr"] = ((df_out["atr_risk"] / close_safe) * 100).ffill().fillna(0.0)
            else: raise ValueError("ATR_PERIOD_RISK must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating ATR: {e}", exc_info=False)
            df_out["atr_risk"] = np.nan; df_out["norm_atr"] = np.nan

        try:
            rsi_period = strategy_cfg.get("RSI_PERIOD", 10)
            if rsi_period > 0:
                rsi_series = ta.rsi(df_out[price_source_col], length=rsi_period)
                df_out["rsi"] = rsi_series # Handle NaNs later in bulk fill                                                                               else: raise ValueError("RSI_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating RSI: {e}", exc_info=False)
            df_out["rsi"] = np.nan

        try:
            vol_period = strategy_cfg.get("VOLUME_SMA_PERIOD", 20)                 if vol_period > 0:
                df_out["volume_sma"] = ta.sma(df_out["volume"], length=vol_period) # Handle NaNs later
                min_periods_vol = max(1, vol_period // 2) # Ensure min_periods is at least 1                                                                  # Handle NaNs later, ensure min_periods for rolling quantile
                df_out["volume_percentile_75"] = df_out["volume"].rolling(window=vol_period, min_periods=min_periods_vol).quantile(0.75)
            else: raise ValueError("VOLUME_SMA_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating Volume Indicators: {e}", exc_info=False)                                       df_out["volume_sma"] = np.nan; df_out["volume_percentile_75"] = np.nan

        try:
            adx_period = strategy_cfg.get("ADX_PERIOD", 14)
            if adx_period > 0:
                adx_df = ta.adx(df_out["high"], df_out["low"], df_out["close"], length=adx_period)                                                            adx_col_name = f"ADX_{adx_period}"
                if adx_df is not None and not adx_df.empty and adx_col_name in adx_df.columns:
                    df_out["adx"] = adx_df[adx_col_name] # Handle NaNs later                                                                                  else:
                    log_console(logging.WARNING, f"Indicator Calc: ADX ({adx_col_name}) calculation failed or returned invalid DataFrame.")
                    df_out["adx"] = np.nan
            else: raise ValueError("ADX_PERIOD must be > 0")
        except Exception as e:
            log_console(logging.ERROR, f"Indicator Calc: Error calculating ADX: {e}", exc_info=False)
            df_out["adx"] = np.nan                                     
        try:                                                                       bb_period = strategy_cfg.get("BBANDS_PERIOD", 20)
            bb_std = strategy_cfg.get("BBANDS_STDDEV", 2.0)
            if bb_period > 0 and bb_std > 0:
                bbands = ta.bbands(df_out["close"], length=bb_period, std=bb_std)                                                                             bbu_col = f"BBU_{bb_period}_{bb_std}"
                bbl_col = f"BBL_{bb_period}_{bb_std}"
                bbm_col = f"BBM_{bb_period}_{bb_std}"                                  if bbands is not None and not bbands.empty and all(c in bbands.columns for c in [bbu_col, bbl_col, bbm_col]):
                    # Calculate width safely, handle NaNs and potential division by zero later
                    bbm_safe = bbands[bbm_col].replace(0, np.nan) # Replace 0 with NaN before division
                    df_out["bb_width"] = (bbands[bbu_col] - bbands[bbl_col]) / bbm_safe * 100
                else:                                                                      log_console(logging.WARNING, "Indicator Calc: Bollinger Bands calculation failed or missing columns.")                                        df_out["bb_width"] = np.nan                                    else: raise ValueError("BBANDS_PERIOD and BBANDS_STDDEV must be > 0")
        except Exception as e:                                                     log_console(logging.ERROR, f"Indicator Calc: Error calculating Bollinger Bands: {e}", exc_info=False)
            df_out["bb_width"] = np.nan
                                                                               # --- Fill NaNs in Key Numeric Columns BEFORE calculating signals ---
        # Forward fill first, then fill remaining (start of series) with appropriate defaults
        # Ensure MA/Smoother columns exist before trying to fill them          ma_cols_to_fill = {}                                                   for ma in ["ultra", "fast", "mid", "slow"]:
             ma_col = f"{smoother_prefix}_{ma}"
             if ma_col in df_out.columns:
                 ma_cols_to_fill[ma_col] = 0.0 # Default fill for MAs (price=0), adjust if needed

        numeric_cols_to_fill = {                                                   "atr_risk": 0.0, "norm_atr": 0.0, "roc": 0.0, "roc_smooth": 0.0,                                                                              "rsi": 50.0, "adx": 20.0, "bb_width": 0.0, "volume_sma": 0.0,                                                                                 "volume_percentile_75": 0.0,
            # Fill trendline with price if available, else 0
            "trendline": df_out[price_source_col] if price_source_col in df_out else 0.0,                                                                 **ma_cols_to_fill # Add the dynamically determined MA columns                                                                             }

        for col, fill_value in numeric_cols_to_fill.items():
            if col in df_out.columns:
                # Check if fill_value is a Series (like for trendline)                 if isinstance(fill_value, pd.Series):
                    # Align index before filling if needed, though ffill handles it
                    df_out[col] = df_out[col].ffill().fillna(fill_value)
                else:
                    df_out[col] = df_out[col].ffill().fillna(fill_value)
            # else: # Column might be missing due to calc error, already logged

        # Calculate high_volume after filling volume_sma and percentile
        # Fill volume NaNs with 0 for comparison
        if "volume" in df_out and "volume_percentile_75" in df_out and "volume_sma" in df_out:
            df_out["high_volume"] = (df_out["volume"].fillna(0.0) > df_out["volume_percentile_75"]) & \                                                                          (df_out["volume"].fillna(0.0) > df_out["volume_sma"])
            df_out["high_volume"] = df_out["high_volume"].fillna(False).astype(bool) # Ensure boolean
        else:
            log_console(logging.WARNING, "Indicator Calc: Volume columns missing, cannot calculate 'high_volume'.")
            df_out["high_volume"] = False # Default to False

        # --- Dynamic Thresholds based on Volatility ---
        atr_period_volatility = strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14)                                                                         if "norm_atr" in df_out and atr_period_volatility > 0:                     min_periods_vol_atr = max(1, atr_period_volatility // 2)               # Calculate rolling mean on norm_atr (already filled), ffill initial NaNs, default to 1.0                                                     volatility_factor = df_out["norm_atr"].rolling(window=atr_period_volatility, min_periods=min_periods_vol_atr).mean().ffill().fillna(1.0)                                                                             # Clip volatility factor to prevent extreme adjustments                volatility_factor = np.clip(volatility_factor,                                                     strategy_cfg.get("VOLATILITY_FACTOR_MIN", 0.5),                                                                                               strategy_cfg.get("VOLATILITY_FACTOR_MAX", 2.0))                                                               else:                                                                      log_console(logging.DEBUG,"Indicator Calc: Cannot calculate volatility factor (norm_atr missing or period invalid). Using factor 1.0.")                                                                              volatility_factor = pd.Series(1.0, index=df_out.index) # Neutral factor                                                                                                                                          rsi_base_low = strategy_cfg.get("RSI_LOW_THRESHOLD", 40)               rsi_base_high = strategy_cfg.get("RSI_HIGH_THRESHOLD", 75)             # Use pd.Series constructor to ensure index alignment                  rsi_low = pd.Series(np.clip(rsi_base_low - (strategy_cfg.get("RSI_VOLATILITY_ADJUST", 5) * volatility_factor), 10, 50), index=df_out.index)                                                                          rsi_high = pd.Series(np.clip(rsi_base_high + (strategy_cfg.get("RSI_VOLATILITY_ADJUST", 5) * volatility_factor), 60, 90), index=df_out.index)                                                                                                                                               roc_base_threshold = strategy_cfg.get("MOM_THRESHOLD_PERCENT", 0.1)                                                                           # Use pd.Series constructor to ensure index alignment, default to base threshold
        roc_threshold = pd.Series(abs(roc_base_threshold) * volatility_factor, index=df_out.index).fillna(abs(roc_base_threshold))

        # --- Trend Definition ---
        fast_ma_col = f"{smoother_prefix}_fast"                                mid_ma_col = f"{smoother_prefix}_mid"
        slow_ma_col = f"{smoother_prefix}_slow"
        trendline_col = "trendline"                                            #