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
    from colorama import (
        init as colorama_init,
        Fore,
        Style,
        Back,
    )  # Colored console output
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
# Maximum klines fetchable in one V5 API call (check Bybit docs for exact value)
MAX_KLINE_LIMIT_PER_REQUEST = 1000
# Minimum records needed for most basic calculations (e.g., diff)
MIN_KLINE_RECORDS_FOR_CALC = 2
# Used for floating point comparisons
FLOAT_COMPARISON_TOLERANCE = 1e-9

# --- Logging Setup ---

# Base Logger Configuration (File Handler, no colors in file, UTF-8 encoded)
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
# Use 'w' mode to overwrite log file on each run, or 'a' to append
log_file_mode = "a"  # Change to 'w' if needed
file_handler = logging.FileHandler(LOG_FILE, mode=log_file_mode, encoding="utf-8")
file_handler.setFormatter(log_formatter)

# Main application logger
logger = logging.getLogger("MomentumScannerTrader")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)  # Default level, can be overridden by --debug
logger.propagate = False  # Prevent duplicate logs if root logger is configured

# Metrics Logger Configuration (CSV-like format, UTF-8 encoded)
metrics_formatter = logging.Formatter("%(asctime)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
metrics_handler = logging.FileHandler(METRICS_LOG_FILE, mode="a", encoding="utf-8")
metrics_handler.setFormatter(metrics_formatter)

# Metrics logger instance
metrics_logger = logging.getLogger("TradeMetrics")
metrics_logger.addHandler(metrics_handler)
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False

# --- Console Logging Function ---

# Neon-Colored Console Output with Symbol Support
SYMBOL_COLORS = {
    # Add specific symbols for distinct colors or use a cycle
    "DOTUSDT": Fore.CYAN + Style.BRIGHT,
    "TRUMPUSDT": Fore.YELLOW + Style.BRIGHT,
    "BCHUSDT": Fore.MAGENTA + Style.BRIGHT,
    "DOGEUSDT": Fore.BLUE + Style.BRIGHT,
    "default": Fore.WHITE,
}
_symbol_color_cycle = [
    Fore.CYAN,
    Fore.YELLOW,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.GREEN,
    Fore.LIGHTRED_EX,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTYELLOW_EX,
]
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
        color = _symbol_color_cycle[_color_index % len(_symbol_color_cycle)] + Style.BRIGHT
        _symbol_color_map[symbol] = color
        _color_index += 1
    return _symbol_color_map[symbol]


def log_console(
    level: int,
    message: Any,
    symbol: Optional[str] = None,
    exc_info: bool = False,
    *args,
    **kwargs,
):
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
    neon_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW + Style.BRIGHT,
        logging.ERROR: Fore.RED + Style.BRIGHT,
        logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
    }
    color = neon_colors.get(level, Fore.WHITE)
    level_name = logging.getLevelName(level)
    symbol_color = get_symbol_color(symbol)
    symbol_prefix = f"{symbol_color}[{symbol}]{Style.RESET_ALL} " if symbol else ""
    prefix = f"{color}{level_name}:{Style.RESET_ALL} {symbol_prefix}"

    # Ensure message is a string before formatting
    if not isinstance(message, str):
        message_str = str(message)
    else:
        message_str = message

    # Format message safely if args or kwargs are provided
    formatted_message = message_str  # Default to stringified message
    try:
        # Attempt format only if args or kwargs are present and message was originally a string
        if (args or kwargs) and isinstance(message, str):
            formatted_message = message.format(*args, **kwargs)
    except Exception as fmt_e:
        # Log formatting errors without crashing, include original message
        print(f"{Fore.RED}LOG FORMATTING ERROR:{Style.RESET_ALL} {fmt_e} | Original Msg: {message_str}")
        logger.error(
            f"Log formatting error: {fmt_e} | Original Msg: {message_str}",
            exc_info=False,
        )
        # Use the unformatted string message for console output
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

    Args:
        message: The metrics message string (should be comma-separated).
    """
    metrics_logger.info(message)
    print(f"{Fore.MAGENTA + Style.BRIGHT}METRICS:{Style.RESET_ALL} {message}")


# --- Load API Keys from Environment Variables ---
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

# Check if API keys are loaded
if not API_KEY or not API_SECRET:
    # Use log_console for consistency, even at critical level before full logging setup
    log_console(
        logging.CRITICAL,
        "BYBIT_API_KEY or BYBIT_API_SECRET not found in environment variables or .env file.",
    )
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
        with open(config_path, "r", encoding="utf-8") as f:
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
                log_console(
                    logging.WARNING,
                    f"Missing or invalid 'STRATEGY_CONFIG' for {symbol}. Using defaults.",
                    symbol=symbol,
                )
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
                periods.extend(periods)

            # Filter out invalid or non-numeric periods
            valid_periods = [p for p in periods if isinstance(p, (int, float)) and p > 0]
            if not valid_periods:
                log_console(
                    logging.WARNING,
                    f"No valid indicator periods found for {symbol}. Using default max period (200).",
                    symbol=symbol,
                )
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
            # Allow storing more than one fetch worth
            internal_max_limit = 2 * MAX_KLINE_LIMIT_PER_REQUEST
            kline_limit = min(kline_limit, internal_max_limit)

            # Store calculated limit within the symbol's config (internal use)
            symbol_cfg.setdefault("INTERNAL", {})["KLINE_LIMIT"] = kline_limit
            log_console(
                logging.DEBUG,
                f"Calculated KLINE_LIMIT for {symbol}: {kline_limit} (Max Period: {max_period:.0f}, Buffer: {kline_limit_buffer}, ADX Buf: {adx_warmup_buffer})",
                symbol=symbol,
            )

        # --- Detailed Configuration Validation (Performed later in main block) ---

        return config

    except FileNotFoundError:
        log_console(logging.CRITICAL, f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_console(
            logging.CRITICAL,
            f"Error decoding JSON configuration file {config_path}: {e}",
        )
        sys.exit(1)
    except ValueError as e:
        log_console(logging.CRITICAL, f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        log_console(
            logging.CRITICAL,
            f"Unexpected error loading configuration: {e}",
            exc_info=True,
        )
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
        log_console(
            logging.WARNING,
            f"Super Smoother: Period must be an integer >= 2, got {period}. Returning NaNs.",
        )
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2:
        log_console(
            logging.DEBUG,
            f"Super Smoother: Series length ({len(series_cleaned)} non-NaN) is less than required minimum (2). Returning NaNs.",
        )
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
        if len(series_float) > 0 and not np.isnan(series_float.iloc[0]):
            ss[0] = series_float.iloc[0]
        if len(series_float) > 1 and not np.isnan(series_float.iloc[1]):
            ss[1] = series_float.iloc[1]

        # Filter loop - check for NaNs rigorously
        for i in range(2, len(series_float)):
            # Check all required inputs for NaN before calculation
            # Check current price inputs and previous SS outputs
            if (
                np.isnan(series_float.iloc[i])
                or np.isnan(series_float.iloc[i - 1])
                or np.isnan(ss[i - 1])
                or np.isnan(ss[i - 2])
            ):
                # If any input is NaN, the result for this step is NaN.
                # ss[i] remains NaN as initialized.
                continue

            prev_ss = ss[i - 1]
            prev_prev_ss = ss[i - 2]
            current_price_avg = (series_float.iloc[i] + series_float.iloc[i - 1]) / 2.0
            ss[i] = coeff1 * current_price_avg + b1 * prev_ss + c1 * prev_prev_ss

        return pd.Series(ss, index=series.index, dtype=np.float64)
    except Exception as e:
        log_console(
            logging.ERROR,
            f"Super Smoother calculation error for period {period}: {e}",
            exc_info=True,
        )
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
        log_console(
            logging.WARNING,
            f"Instantaneous Trendline: Period must be an integer >= {required_min_period}, got {period}. Returning NaNs.",
        )
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length
    series_cleaned = series.dropna()
    min_len_needed = 4  # Based on formula dependencies
    if len(series_cleaned) < min_len_needed:
        log_console(
            logging.DEBUG,
            f"Instantaneous Trendline: Series length ({len(series_cleaned)} non-NaN) is less than required minimum ({min_len_needed}). Returning NaNs.",
        )
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        # Fixed alpha as per Ehlers' recommendation (adjust if needed based on source/testing)
        alpha = 0.07

        # Use float, ffill cautiously
        series_float = series.astype(float).ffill()
        it = np.full(len(series_float), np.nan, dtype=np.float64)

        # Initialize first few values robustly based on Ehlers common practice after ffill
        # Check source values are not NaN before using them
        if len(series_float) > 0 and not np.isnan(series_float.iloc[0]):
            it[0] = series_float.iloc[0]
        if len(series_float) > 1 and not np.isnan(series_float.iloc[1]):
            it[1] = series_float.iloc[1]
        # Requires indices 0, 1, 2
        if len(series_float) > 2 and not np.any(np.isnan(series_float.iloc[0:3])):
            it[2] = (series_float.iloc[2] + 2.0 * series_float.iloc[1] + series_float.iloc[0]) / 4.0
        # Requires indices 1, 2, 3
        if len(series_float) > 3 and not np.any(np.isnan(series_float.iloc[1:4])):
            it[3] = (series_float.iloc[3] + 2.0 * series_float.iloc[2] + series_float.iloc[1]) / 4.0

        # Calculation loop - check for NaNs rigorously
        # Starts from index 4, as it[3] depends on it[2] and it[1]
        for i in range(4, len(series_float)):
            # Check all required price inputs and previous IT outputs
            if (
                np.isnan(series_float.iloc[i])
                or np.isnan(series_float.iloc[i - 1])
                or np.isnan(series_float.iloc[i - 2])
                or np.isnan(it[i - 1])
                or np.isnan(it[i - 2])
            ):
                # If any input is NaN, the result for this step is NaN.
                # it[i] remains NaN as initialized.
                continue

            price_i = series_float.iloc[i]
            price_im1 = series_float.iloc[i - 1]
            price_im2 = series_float.iloc[i - 2]
            it_im1 = it[i - 1]
            it_im2 = it[i - 2]

            # Formula from Ehlers' papers/code (verify against trusted source if possible):
            it[i] = (
                (alpha - alpha**2 / 4.0) * price_i
                + (alpha**2 / 2.0) * price_im1
                - (alpha - 3.0 * alpha**2 / 4.0) * price_im2
                + 2.0 * (1.0 - alpha) * it_im1
                - (1.0 - alpha) ** 2 * it_im2
            )

        return pd.Series(it, index=series.index, dtype=np.float64)
    except Exception as e:
        log_console(
            logging.ERROR,
            f"Instantaneous Trendline calculation error for period {period}: {e}",
            exc_info=True,
        )
        return pd.Series(np.nan, index=series.index, dtype=np.float64)


# --- Indicator Calculation ---
def calculate_indicators_momentum(
    df: pd.DataFrame, strategy_cfg: Dict[str, Any], config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
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
            log_console(
                logging.DEBUG,
                "Indicator Calc: DataFrame index is not DatetimeIndex, attempting conversion.",
            )
            # Convert index to datetime, coercing errors, setting timezone to UTC
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
            if not isinstance(df.index, pd.DatetimeIndex):  # Check conversion success
                raise ValueError("Index conversion to DatetimeIndex failed.")
            if df.index.isnull().any():
                log_console(
                    logging.WARNING,
                    "Indicator Calc: Found NaT values in index after conversion. Dropping affected rows.",
                )
                df.dropna(axis=0, how="any", subset=required_cols, inplace=True)  # Drop rows with NaT index
                if df.empty:
                    log_console(
                        logging.ERROR,
                        "Indicator Calc: DataFrame empty after dropping rows with NaT index.",
                    )
                    return None
        except Exception as idx_e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Failed to convert DataFrame index to DatetimeIndex: {idx_e}",
            )
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
                log_console(
                    logging.DEBUG,
                    f"Indicator Calc: Converting non-numeric column '{col}' to numeric.",
                )
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

        # Drop rows where essential conversions failed (resulted in NaNs)
        initial_len = len(df_out)
        df_out.dropna(subset=required_cols, inplace=True)
        if len(df_out) < initial_len:
            log_console(
                logging.WARNING,
                f"Indicator Calc: Dropped {initial_len - len(df_out)} rows due to NaN values in required columns after numeric conversion.",
            )

        # Check length requirements AFTER cleaning numeric data
        if len(df_out) < kline_limit:
            log_console(
                logging.DEBUG,
                f"Indicator Calc: Insufficient valid data points after NaN drop ({len(df_out)} < {kline_limit}). Need more historical data.",
            )
            return None
        if df_out.empty:
            log_console(
                logging.ERROR,
                "Indicator Calc: DataFrame empty after NaN drop from numeric conversion.",
            )
            return None

    except Exception as e:
        log_console(
            logging.ERROR,
            f"Indicator Calc: Failed during data validation/numeric conversion: {e}",
            exc_info=True,
        )
        return None

    try:
        price_source_key = strategy_cfg.get("PRICE_SOURCE", "close")
        # Handle common price source calculations if needed (e.g., hlc3)
        if price_source_key == "hlc3":
            df_out["hlc3"] = (df_out["high"] + df_out["low"] + df_out["close"]) / 3.0
            price_source_col = "hlc3"
        elif price_source_key == "ohlc4":
            df_out["ohlc4"] = (df_out["open"] + df_out["high"] + df_out["low"] + df_out["close"]) / 4.0
            price_source_col = "ohlc4"
        elif price_source_key in df_out.columns:
            price_source_col = price_source_key
        else:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Specified price source '{price_source_key}' not found or calculable. Falling back to 'close'.",
            )
            if "close" not in df_out.columns:
                return None  # Cannot proceed without 'close'
            price_source_col = "close"

        if df_out[price_source_col].isnull().all():
            log_console(
                logging.ERROR,
                f"Indicator Calc: Price source column '{price_source_col}' contains only NaNs.",
            )
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

        for period_key, base_col_name in smoother_params:
            length = strategy_cfg.get(period_key)
            col_name = base_col_name.replace("ema", smoother_prefix)
            if not isinstance(length, int) or length <= 0:
                log_console(
                    logging.ERROR,
                    f"Indicator Calc: Invalid or missing {period_key} in strategy config. Value: {length}. Skipping {col_name}.",
                )
                df_out[col_name] = np.nan
                continue

            try:
                if use_ehlers:
                    indicator_series = super_smoother(df_out[price_source_col], length)
                else:
                    indicator_series = ta.ema(df_out[price_source_col], length=length)

                if indicator_series is None or indicator_series.empty or indicator_series.isnull().all():
                    log_console(
                        logging.WARNING,
                        f"Indicator Calc: {col_name} (length {length}) calculation resulted in empty or all NaNs.",
                    )
                    df_out[col_name] = np.nan
                else:
                    df_out[col_name] = indicator_series
            except Exception as calc_e:
                log_console(
                    logging.ERROR,
                    f"Indicator Calc: Error calculating {col_name} (length {length}): {calc_e}",
                    exc_info=False,
                )
                df_out[col_name] = np.nan

        # --- Instantaneous Trendline ---
        it_period = strategy_cfg.get("INSTANTANEOUS_TRENDLINE_PERIOD", 20)
        df_out["trendline"] = np.nan  # Initialize column
        # Check minimum period requirement
        if not isinstance(it_period, int) or it_period < 4:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Invalid INSTANTANEOUS_TRENDLINE_PERIOD ({it_period}). Must be integer >= 4. Skipping calculation.",
            )
        else:
            try:
                it_series = instantaneous_trendline(df_out[price_source_col], it_period)
                if it_series is None or it_series.empty or it_series.isnull().all():
                    log_console(
                        logging.WARNING,
                        "Indicator Calc: Instantaneous Trendline calculation resulted in empty or all NaNs.",
                    )
                    # Column already initialized to NaN
                else:
                    df_out["trendline"] = it_series
            except Exception as calc_e:
                log_console(
                    logging.ERROR,
                    f"Indicator Calc: Error calculating Instantaneous Trendline: {calc_e}",
                    exc_info=False,
                )
                # Column already initialized to NaN

        # --- Other Indicators (wrapped calculations) ---
        try:
            roc_period = strategy_cfg.get("ROC_PERIOD", 5)
            if roc_period > 0:
                df_out["roc"] = ta.roc(df_out[price_source_col], length=roc_period)
                # Fill NaNs before smoothing ROC, use 0 as neutral momentum
                df_out["roc_smooth"] = ta.sma(df_out["roc"].fillna(0.0), length=3)
            else:
                raise ValueError("ROC_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating ROC: {e}",
                exc_info=False,
            )
            df_out["roc"] = np.nan
            df_out["roc_smooth"] = np.nan

        try:
            atr_period_risk = strategy_cfg.get("ATR_PERIOD_RISK", 14)
            if atr_period_risk > 0:
                atr_series = ta.atr(
                    df_out["high"],
                    df_out["low"],
                    df_out["close"],
                    length=atr_period_risk,
                )
                # Use .ffill() as per pandas_ta recommendation, then fill remaining with 0
                df_out["atr_risk"] = atr_series.ffill().fillna(0.0)
                # Normalized ATR
                # Avoid division by zero/NaN: replace 0 with NaN, ffill, then fill remaining start NaNs with 1
                close_safe = df_out["close"].replace(0, np.nan).ffill().fillna(1.0)
                df_out["norm_atr"] = ((df_out["atr_risk"] / close_safe) * 100).ffill().fillna(0.0)
            else:
                raise ValueError("ATR_PERIOD_RISK must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating ATR: {e}",
                exc_info=False,
            )
            df_out["atr_risk"] = np.nan
            df_out["norm_atr"] = np.nan

        try:
            rsi_period = strategy_cfg.get("RSI_PERIOD", 10)
            if rsi_period > 0:
                rsi_series = ta.rsi(df_out[price_source_col], length=rsi_period)
                df_out["rsi"] = rsi_series  # Handle NaNs later in bulk fill
            else:
                raise ValueError("RSI_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating RSI: {e}",
                exc_info=False,
            )
            df_out["rsi"] = np.nan

        try:
            vol_period = strategy_cfg.get("VOLUME_SMA_PERIOD", 20)
            if vol_period > 0:
                df_out["volume_sma"] = ta.sma(df_out["volume"], length=vol_period)  # Handle NaNs later
                # Ensure min_periods is at least 1
                min_periods_vol = max(1, vol_period // 2)
                # Handle NaNs later, ensure min_periods for rolling quantile
                df_out["volume_percentile_75"] = (
                    df_out["volume"].rolling(window=vol_period, min_periods=min_periods_vol).quantile(0.75)
                )
            else:
                raise ValueError("VOLUME_SMA_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating Volume Indicators: {e}",
                exc_info=False,
            )
            df_out["volume_sma"] = np.nan
            df_out["volume_percentile_75"] = np.nan

        try:
            adx_period = strategy_cfg.get("ADX_PERIOD", 14)
            if adx_period > 0:
                adx_df = ta.adx(df_out["high"], df_out["low"], df_out["close"], length=adx_period)
                adx_col_name = f"ADX_{adx_period}"
                if adx_df is not None and not adx_df.empty and adx_col_name in adx_df.columns:
                    df_out["adx"] = adx_df[adx_col_name]  # Handle NaNs later
                else:
                    log_console(
                        logging.WARNING,
                        f"Indicator Calc: ADX ({adx_col_name}) calculation failed or returned invalid DataFrame.",
                    )
                    df_out["adx"] = np.nan
            else:
                raise ValueError("ADX_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating ADX: {e}",
                exc_info=False,
            )
            df_out["adx"] = np.nan

        try:
            bb_period = strategy_cfg.get("BBANDS_PERIOD", 20)
            bb_std = strategy_cfg.get("BBANDS_STDDEV", 2.0)
            if bb_period > 0 and bb_std > 0:
                bbands = ta.bbands(df_out["close"], length=bb_period, std=bb_std)
                bbu_col = f"BBU_{bb_period}_{bb_std}"
                bbl_col = f"BBL_{bb_period}_{bb_std}"
                bbm_col = f"BBM_{bb_period}_{bb_std}"
                if (
                    bbands is not None
                    and not bbands.empty
                    and all(c in bbands.columns for c in [bbu_col, bbl_col, bbm_col])
                ):
                    # Calculate width safely, handle NaNs and potential division by zero later
                    # Replace 0 with NaN before division
                    bbm_safe = bbands[bbm_col].replace(0, np.nan)
                    df_out["bb_width"] = (bbands[bbu_col] - bbands[bbl_col]) / bbm_safe * 100
                else:
                    log_console(
                        logging.WARNING,
                        "Indicator Calc: Bollinger Bands calculation failed or missing columns.",
                    )
                    df_out["bb_width"] = np.nan
            else:
                raise ValueError("BBANDS_PERIOD and BBANDS_STDDEV must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating Bollinger Bands: {e}",
                exc_info=False,
            )
            df_out["bb_width"] = np.nan

        # --- Fill NaNs in Key Numeric Columns BEFORE calculating signals ---
        # Forward fill first, then fill remaining (start of series) with appropriate defaults
        # Ensure MA/Smoother columns exist before trying to fill them
        ma_cols_to_fill = {}
        for ma in ["ultra", "fast", "mid", "slow"]:
            ma_col = f"{smoother_prefix}_{ma}"
            if ma_col in df_out.columns:
                # Default fill for MAs (price=0), adjust if needed
                ma_cols_to_fill[ma_col] = 0.0

        numeric_cols_to_fill = {
            "atr_risk": 0.0,
            "norm_atr": 0.0,
            "roc": 0.0,
            "roc_smooth": 0.0,
            "rsi": 50.0,
            "adx": 20.0,
            "bb_width": 0.0,
            "volume_sma": 0.0,
            "volume_percentile_75": 0.0,
            # Fill trendline with price if available, else 0
            "trendline": (df_out[price_source_col] if price_source_col in df_out else 0.0),
            **ma_cols_to_fill,  # Add the dynamically determined MA columns
        }

        for col, fill_value in numeric_cols_to_fill.items():
            if col in df_out.columns:
                # Check if fill_value is a Series (like for trendline)
                if isinstance(fill_value, pd.Series):
                    # Align index before filling if needed, though ffill handles it
                    df_out[col] = df_out[col].ffill().fillna(fill_value)
                else:
                    df_out[col] = df_out[col].ffill().fillna(fill_value)
            # else: # Column might be missing due to calc error, already logged

        # Calculate high_volume after filling volume_sma and percentile
        # Fill volume NaNs with 0 for comparison
        if "volume" in df_out and "volume_percentile_75" in df_out and "volume_sma" in df_out:
            df_out["high_volume"] = (df_out["volume"].fillna(0.0) > df_out["volume_percentile_75"]) & (
                df_out["volume"].fillna(0.0) > df_out["volume_sma"]
            )
            df_out["high_volume"] = df_out["high_volume"].fillna(False).astype(bool)  # Ensure boolean
        else:
            log_console(
                logging.WARNING,
                "Indicator Calc: Volume columns missing, cannot calculate 'high_volume'.",
            )
            df_out["high_volume"] = False  # Default to False

        # --- Dynamic Thresholds based on Volatility ---
        atr_period_volatility = strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14)
        if "norm_atr" in df_out and atr_period_volatility > 0:
            min_periods_vol_atr = max(1, atr_period_volatility // 2)
            # Calculate rolling mean on norm_atr (already filled), ffill initial NaNs, default to 1.0
            volatility_factor = (
                df_out["norm_atr"]
                .rolling(window=atr_period_volatility, min_periods=min_periods_vol_atr)
                .mean()
                .ffill()
                .fillna(1.0)
            )
            # Clip volatility factor to prevent extreme adjustments
            volatility_factor = np.clip(
                volatility_factor,
                strategy_cfg.get("VOLATILITY_FACTOR_MIN", 0.5),
                strategy_cfg.get("VOLATILITY_FACTOR_MAX", 2.0),
            )
        else:
            log_console(
                logging.DEBUG,
                "Indicator Calc: Cannot calculate volatility factor (norm_atr missing or period invalid). Using factor 1.0.",
            )
            volatility_factor = pd.Series(1.0, index=df_out.index)  # Neutral factor

        rsi_base_low = strategy_cfg.get("RSI_LOW_THRESHOLD", 40)
        rsi_base_high = strategy_cfg.get("RSI_HIGH_THRESHOLD", 75)
        # Use pd.Series constructor to ensure index alignment
        rsi_low = pd.Series(
            np.clip(
                rsi_base_low - (strategy_cfg.get("RSI_VOLATILITY_ADJUST", 5) * volatility_factor),
                10,
                50,
            ),
            index=df_out.index,
        )
        rsi_high = pd.Series(
            np.clip(
                rsi_base_high + (strategy_cfg.get("RSI_VOLATILITY_ADJUST", 5) * volatility_factor),
                60,
                90,
            ),
            index=df_out.index,
        )

        roc_base_threshold = strategy_cfg.get("MOM_THRESHOLD_PERCENT", 0.1)
        # Use pd.Series constructor to ensure index alignment, default to base threshold
        roc_threshold = pd.Series(abs(roc_base_threshold) * volatility_factor, index=df_out.index).fillna(
            abs(roc_base_threshold)
        )

        # --- Trend Definition ---
        fast_ma_col = f"{smoother_prefix}_fast"
        mid_ma_col = f"{smoother_prefix}_mid"
        slow_ma_col = f"{smoother_prefix}_slow"
        trendline_col = "trendline"
        # Initialize trend columns
        df_out["trend_up"] = False
        df_out["trend_down"] = False
        df_out["trend_neutral"] = True

        trend_req_cols = [fast_ma_col, mid_ma_col, slow_ma_col, trendline_col]
        if not all(col in df_out.columns for col in trend_req_cols):
            log_console(
                logging.WARNING,
                f"Indicator Calc: Cannot define trend, required MA/Trendline columns missing: {trend_req_cols}. Assuming neutral trend.",
            )
        else:
            # NaNs already filled in MA/trendline columns
            # Fill initial shift NaN with the first available trendline value using bfill then ffill
            trendline_shift = df_out[trendline_col].shift(1).fillna(method="bfill").fillna(method="ffill")

            trend_up_cond = (
                (df_out[fast_ma_col] > df_out[mid_ma_col])
                & (df_out[mid_ma_col] > df_out[slow_ma_col])
                &
                # Use tolerance for trendline comparison
                (df_out[trendline_col] > trendline_shift * (1 + FLOAT_COMPARISON_TOLERANCE))
            )
            df_out["trend_up"] = trend_up_cond.astype(bool)

            trend_down_cond = (
                (df_out[fast_ma_col] < df_out[mid_ma_col])
                & (df_out[mid_ma_col] < df_out[slow_ma_col])
                & (df_out[trendline_col] < trendline_shift * (1 - FLOAT_COMPARISON_TOLERANCE))
            )
            df_out["trend_down"] = trend_down_cond.astype(bool)

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

        entry_req_cols = [
            fast_ma_col,
            mid_ma_col,
            ultra_fast_ma_col,
            roc_smooth_col,
            rsi_col,
            adx_col,
            trendline_col,
            high_volume_col,
        ]
        if not all(col in df_out.columns for col in entry_req_cols):
            log_console(
                logging.WARNING,
                f"Indicator Calc: Cannot generate entry signals, required columns missing: {entry_req_cols}",
            )
        else:
            # Ensure trendline_shift is available (recalculate or use previous one)
            if "trendline_shift" not in locals():  # Check if calculated in trend section
                trendline_shift = df_out[trendline_col].shift(1).fillna(method="bfill").fillna(method="ffill")

            # Conditions (NaNs already handled in source columns)
            long_cond_trend = df_out[fast_ma_col] > df_out[mid_ma_col]
            long_cond_trigger = df_out[ultra_fast_ma_col] > df_out[fast_ma_col]
            long_cond_mom = df_out[roc_smooth_col] > roc_threshold
            long_cond_rsi = (df_out[rsi_col] > rsi_low) & (df_out[rsi_col] < rsi_high)  # Dynamic RSI thresholds
            long_cond_vol = df_out[high_volume_col]
            long_cond_adx = df_out[adx_col] > adx_threshold
            long_cond_itrend = df_out[trendline_col] > trendline_shift * (1 + FLOAT_COMPARISON_TOLERANCE)

            df_out["long_signal"] = (
                long_cond_trend
                & long_cond_trigger
                & long_cond_mom
                & long_cond_rsi
                & long_cond_vol
                & long_cond_adx
                & long_cond_itrend
            ).astype(bool)

            short_cond_trend = df_out[fast_ma_col] < df_out[mid_ma_col]
            short_cond_trigger = df_out[ultra_fast_ma_col] < df_out[fast_ma_col]
            short_cond_mom = df_out[roc_smooth_col] < -roc_threshold
            short_cond_rsi = (df_out[rsi_col] > rsi_low) & (df_out[rsi_col] < rsi_high)  # Dynamic RSI thresholds
            short_cond_vol = df_out[high_volume_col]
            short_cond_adx = df_out[adx_col] > adx_threshold
            short_cond_itrend = df_out[trendline_col] < trendline_shift * (1 - FLOAT_COMPARISON_TOLERANCE)

            df_out["short_signal"] = (
                short_cond_trend
                & short_cond_trigger
                & short_cond_mom
                & short_cond_rsi
                & short_cond_vol
                & short_cond_adx
                & short_cond_itrend
            ).astype(bool)

            # Signal Strength (Count how many conditions are True)
            long_conditions = [
                long_cond_trend,
                long_cond_trigger,
                long_cond_mom,
                long_cond_rsi,
                long_cond_vol,
                long_cond_adx,
                long_cond_itrend,
            ]
            short_conditions = [
                short_cond_trend,
                short_cond_trigger,
                short_cond_mom,
                short_cond_rsi,
                short_cond_vol,
                short_cond_adx,
                short_cond_itrend,
            ]
            # Ensure division by zero is handled if len is 0 (shouldn't happen here)
            num_conditions = len(long_conditions)
            if num_conditions > 0:
                df_out["long_signal_strength"] = sum(cond.astype(int) for cond in long_conditions) / num_conditions
                df_out["short_signal_strength"] = sum(cond.astype(int) for cond in short_conditions) / num_conditions
            else:
                df_out["long_signal_strength"] = 0.0
                df_out["short_signal_strength"] = 0.0

        # --- Exit Signals ---
        df_out["exit_long_signal"] = False
        df_out["exit_short_signal"] = False
        exit_req_cols = [fast_ma_col, mid_ma_col, roc_smooth_col, trendline_col]
        if not all(col in df_out.columns for col in exit_req_cols):
            log_console(
                logging.WARNING,
                f"Indicator Calc: Cannot generate exit signals, required columns missing: {exit_req_cols}",
            )
        else:
            # Ensure trendline_shift is available
            if "trendline_shift" not in locals():
                trendline_shift = df_out[trendline_col].shift(1).fillna(method="bfill").fillna(method="ffill")

            # Exit Long if: Fast MA crosses below Mid MA OR Momentum turns negative OR Trendline turns down
            exit_long_cond1 = df_out[fast_ma_col] < df_out[mid_ma_col]
            exit_long_cond2 = df_out[roc_smooth_col] < 0  # Check against zero
            exit_long_cond3 = df_out[trendline_col] < trendline_shift * (1 - FLOAT_COMPARISON_TOLERANCE)
            df_out["exit_long_signal"] = (exit_long_cond1 | exit_long_cond2 | exit_long_cond3).astype(bool)

            # Exit Short if: Fast MA crosses above Mid MA OR Momentum turns positive OR Trendline turns up
            exit_short_cond1 = df_out[fast_ma_col] > df_out[mid_ma_col]
            exit_short_cond2 = df_out[roc_smooth_col] > 0  # Check against zero
            exit_short_cond3 = df_out[trendline_col] > trendline_shift * (1 + FLOAT_COMPARISON_TOLERANCE)
            df_out["exit_short_signal"] = (exit_short_cond1 | exit_short_cond2 | exit_short_cond3).astype(bool)

        # --- Final Processing & Validation ---
        # Ensure boolean columns are boolean type
        bool_cols = [
            "long_signal",
            "short_signal",
            "exit_long_signal",
            "exit_short_signal",
            "trend_up",
            "trend_down",
            "trend_neutral",
            "high_volume",
        ]
        for col in bool_cols:
            if col in df_out.columns:
                # Fill potential NaNs from logic gaps before converting
                df_out[col] = df_out[col].fillna(False).astype(bool)

        # Fill NaNs in strength signals (shouldn't occur if sources were filled, but safeguard)
        if "long_signal_strength" in df_out:
            df_out["long_signal_strength"] = df_out["long_signal_strength"].fillna(0.0)
        if "short_signal_strength" in df_out:
            df_out["short_signal_strength"] = df_out["short_signal_strength"].fillna(0.0)

        # Final check for validity - ensure required columns for trading exist and have valid data in last two rows
        final_check_cols = [
            "close",
            "atr_risk",
            "long_signal",
            "short_signal",
            "exit_long_signal",
            "exit_short_signal",
        ]
        if df_out.empty or not all(col in df_out.columns for col in final_check_cols):
            log_console(
                logging.ERROR,
                "Indicator Calc: DataFrame empty or missing critical columns after final processing.",
            )
            return None

        # Check last two rows for NaNs in critical columns (if enough rows exist)
        min_rows_for_check = 2
        if len(df_out) >= min_rows_for_check:
            if df_out.iloc[-min_rows_for_check:][final_check_cols].isnull().any().any():
                log_console(
                    logging.ERROR,
                    f"Indicator Calc: Critical columns contain NaN in the last {min_rows_for_check} rows after processing. Check calculations.",
                )
                # Log problematic rows
                log_console(
                    logging.DEBUG,
                    f"Last {min_rows_for_check} rows tail:\n{df_out.iloc[-min_rows_for_check:]}",
                )
                return None
        elif len(df_out) == 1:  # Handle case with only one row
            if df_out.iloc[-1:][final_check_cols].isnull().any().any():
                log_console(
                    logging.ERROR,
                    "Indicator Calc: Critical columns contain NaN in the only row after processing.",
                )
                return None

        return df_out

    except Exception as e:
        log_console(
            logging.ERROR,
            f"Indicator calculation failed unexpectedly: {e}",
            exc_info=True,
        )
        return None


# --- Bybit API Interaction Helpers ---
def get_available_balance(session: HTTP, coin: str = "USDT", account_type: str = "UNIFIED") -> float:
    """
    Fetches the available balance for a specific coin from the Bybit wallet using V5 API.
    Handles UNIFIED, CONTRACT, and SPOT account structures.

    Args:
        session: Authenticated Bybit HTTP session object.
        coin: The coin symbol (e.g., "USDT").
        account_type: The account type ("UNIFIED", "CONTRACT", "SPOT"). MUST match account structure.

    Returns:
        The available balance as a float, or 0.0 if fetch fails, coin not found, or balance is zero/negative.
    """
    try:
        # V5 API call uses 'accountType' parameter
        balance_info = session.get_wallet_balance(accountType=account_type.upper(), coin=coin)

        if balance_info and balance_info.get("retCode") == 0:
            result = balance_info.get("result", {})
            list_data = result.get("list", [])

            if not list_data:
                log_console(
                    logging.DEBUG,
                    f"No balance list found for {account_type} account (maybe empty?). Treating balance as 0.",
                )
                return 0.0

            # --- Parsing Logic for V5 'get_wallet_balance' ---
            coin_list = []
            # UNIFIED response nests coin list inside the first element of 'list'
            if account_type.upper() == "UNIFIED":
                account_data = list_data[0] if list_data else {}
                coin_list = account_data.get("coin", [])
            # For CONTRACT/SPOT, 'list' *should* contain the coin balances directly
            elif account_type.upper() in ["CONTRACT", "SPOT"]:
                coin_list = list_data  # Assumes direct list of coin dicts
            else:
                log_console(
                    logging.WARNING,
                    f"Balance parsing logic not explicitly defined for account type: {account_type}. Attempting UNIFIED structure.",
                )
                # Fallback to UNIFIED structure attempt
                account_data = list_data[0] if list_data else {}
                coin_list = account_data.get("coin", [])

            if not coin_list:
                log_console(
                    logging.WARNING,
                    f"No coin list found within balance data for {account_type}.",
                )
                return 0.0

            for coin_data in coin_list:
                if isinstance(coin_data, dict) and coin_data.get("coin") == coin:
                    # Prefer 'availableToWithdraw' or 'availableBalance' in V5
                    # These fields represent what's actually usable for trading/withdrawal
                    balance_str = coin_data.get("availableToWithdraw")
                    source = "availableToWithdraw"
                    if balance_str is None or balance_str == "":
                        # Unified/Contract use this
                        balance_str = coin_data.get("availableBalance")
                        source = "availableBalance"

                    # Fallback to walletBalance if others are missing, log a debug message
                    if balance_str is None or balance_str == "":
                        balance_str = coin_data.get("walletBalance")  # General total balance
                        source = "walletBalance"
                        log_console(
                            logging.DEBUG,
                            f"'availableToWithdraw'/'availableBalance' empty for {coin}, using '{source}': {balance_str}",
                        )

                    # Handle if all relevant balance fields are empty/missing
                    if balance_str is None or balance_str == "":
                        log_console(
                            logging.WARNING,
                            f"Could not find a valid available balance field for {coin}. Balance is 0.",
                        )
                        balance_str = "0"

                    try:
                        balance_float = float(balance_str)
                        # Ensure balance is not negative
                        return max(0.0, balance_float)
                    except (ValueError, TypeError) as e:
                        log_console(
                            logging.ERROR,
                            f"Could not convert balance string '{balance_str}' from '{source}' to float for {coin}: {e}",
                        )
                        return 0.0

            log_console(
                logging.WARNING,
                f"Coin {coin} not found in {account_type} wallet balance details.",
            )
            return 0.0

        else:
            error_msg = balance_info.get("retMsg", "Unknown error") if balance_info else "Empty response"
            error_code = balance_info.get("retCode", "N/A")
            log_console(
                logging.ERROR,
                f"Failed to fetch wallet balance: {error_msg} (Code: {error_code})",
            )
            return 0.0

    except Exception as e:
        log_console(logging.ERROR, f"Exception fetching wallet balance: {e}", exc_info=True)
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
    quantity and quantity step rules (rounding DOWN).

    Args:
        balance: Available trading balance in USDT allocated for this trade/symbol.
        risk_percent: The percentage of the allocated balance to risk per trade (e.g., 1.0 for 1%).
        sl_distance_price: The absolute price difference for the stop loss (must be positive).
        entry_price: The estimated entry price of the trade (must be positive).
        min_order_qty: Minimum order quantity allowed by the instrument.
        qty_step_float: The quantity step (increment) as a float.
        qty_precision: The number of decimal places for the quantity.
        max_position_usdt: Optional maximum value of the position in USDT.

    Returns:
        The calculated position size (quantity), rounded DOWN to the instrument's
        step precision, or 0.0 if calculation is not possible or results in zero/negative size.
    """
    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [
            balance,
            risk_percent,
            sl_distance_price,
            entry_price,
            min_order_qty,
            qty_step_float,
        ]
    ):
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Invalid input types. Bal:{type(balance)}, Risk%:{type(risk_percent)}, SLDist:{type(sl_distance_price)}, Entry:{type(entry_price)}, MinQty:{type(min_order_qty)}, Step:{type(qty_step_float)}. Returning size 0.",
        )
        return 0.0
    if not isinstance(qty_precision, int) or qty_precision < 0:
        log_console(
            logging.ERROR,
            f"Position Size Calc: Invalid qty_precision ({qty_precision}). Returning size 0.",
        )
        return 0.0

    if (
        balance <= FLOAT_COMPARISON_TOLERANCE
        or risk_percent <= FLOAT_COMPARISON_TOLERANCE
        or entry_price <= FLOAT_COMPARISON_TOLERANCE
        or min_order_qty <= FLOAT_COMPARISON_TOLERANCE
        or qty_step_float <= FLOAT_COMPARISON_TOLERANCE
    ):
        # Use DEBUG as this might happen normally with zero balance or zero risk setting
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Non-positive input values. Balance={balance:.2f}, Risk%={risk_percent}, Entry={entry_price}, MinQty={min_order_qty}, QtyStep={qty_step_float}. Returning size 0.",
        )
        return 0.0
    # Prevent extremely small SL distance causing huge size or division errors
    if sl_distance_price < FLOAT_COMPARISON_TOLERANCE:
        log_console(
            logging.WARNING,
            f"Position Size Calc: Stop loss distance ({sl_distance_price}) is too small (< {FLOAT_COMPARISON_TOLERANCE}). Cannot calculate size reliably. Returning 0.",
        )
        return 0.0

    # Calculate risk amount in USDT
    risk_amount_usdt = balance * (risk_percent / 100.0)

    # Calculate initial position size based on risk amount and SL distance per unit
    try:
        position_size = risk_amount_usdt / sl_distance_price
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Initial size (Risk/SL): {position_size:.{qty_precision + 4}f}",
        )
    except ZeroDivisionError:  # Should be caught by check above, but safeguard
        log_console(
            logging.WARNING,
            "Position Size Calc: SL distance is effectively zero. Cannot calculate size.",
        )
        return 0.0

    # Apply Constraints
    # 1. Max Position Value Constraint
    if max_position_usdt is not None and max_position_usdt > FLOAT_COMPARISON_TOLERANCE:
        try:
            if entry_price <= FLOAT_COMPARISON_TOLERANCE:
                log_console(
                    logging.WARNING,
                    "Position Size Calc: Entry price is zero or too small. Cannot apply Max Position USDT constraint.",
                )
            else:
                max_size_by_value = max_position_usdt / entry_price
                if position_size > max_size_by_value:
                    log_console(
                        logging.INFO,
                        f"Position Size Calc: Capping size from {position_size:.{qty_precision}f} to {max_size_by_value:.{qty_precision}f} due to Max Position USDT (${max_position_usdt:.2f})",
                    )
                    position_size = max_size_by_value
        except ZeroDivisionError:
            log_console(
                logging.WARNING,
                "Position Size Calc: Entry price is zero. Cannot apply Max Position USDT constraint.",
            )

    # 2. Minimum Order Quantity Constraint (Check BEFORE step rounding)
    # Use a small tolerance related to qty_step to avoid floating point issues right at the minimum
    min_qty_tolerance = qty_step_float * 0.01
    if position_size < min_order_qty - min_qty_tolerance:
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Calculated size {position_size:.{qty_precision + 4}f} is below minimum required {min_order_qty}. Returning size 0.",
        )
        return 0.0

    # 3. Quantity Step Constraint (Rounding DOWN)
    try:
        if qty_step_float <= FLOAT_COMPARISON_TOLERANCE:
            log_console(
                logging.ERROR,
                f"Position Size Calc: Quantity step ({qty_step_float}) is zero or too small. Cannot apply step constraint.",
            )
            # Return 0.0 as we cannot guarantee adherence to step size
            return 0.0
        # Use math.floor for explicit round down, then multiply by step
        # Add a small epsilon before division to handle floating point inaccuracies near multiples of step size
        # e.g., if size is 0.9999999999 and step is 0.1, floor(size/step) might be 9 instead of 10
        # This epsilon should be much smaller than the step size itself
        epsilon = qty_step_float * 1e-9
        position_size_adjusted = math.floor((position_size + epsilon) / qty_step_float) * qty_step_float
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Size after step rounding DOWN ({qty_step_float}): {position_size_adjusted:.{qty_precision + 4}f}",
        )
        position_size = position_size_adjusted
    except ZeroDivisionError:
        log_console(
            logging.ERROR,
            "Position Size Calc: Quantity step is zero during rounding. Cannot apply step constraint.",
        )
        return 0.0

    # 4. Final check: ensure size is still >= minimum after rounding DOWN
    if position_size < min_order_qty - min_qty_tolerance:
        log_console(
            logging.INFO,
            f"Position Size Calc: Final size {position_size:.{qty_precision + 4}f} is below minimum {min_order_qty} after rounding down to step. Returning size 0.",
        )
        return 0.0

    # 5. Final rounding to precision (mostly cosmetic after step rounding, but good practice)
    # Use standard round for this final step
    final_size = round(position_size, qty_precision)

    # Ensure size is not effectively zero after all calculations
    if final_size < min_order_qty - min_qty_tolerance:
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Final calculated size ({final_size}) is zero or negligible relative to min qty. Returning size 0.",
        )
        return 0.0

    log_console(
        logging.INFO,
        f"Position Size Calc: Balance={balance:.2f}, Risk%={risk_percent}, SL Dist={sl_distance_price:.{qty_precision + 2}f}, Entry={entry_price:.{qty_precision + 2}f} -> Final Size = {final_size:.{qty_precision}f}",
    )
    return final_size


# --- Trade Performance Tracking ---
class TradeMetrics:
    """
    Tracks and logs performance metrics for completed trades. Thread-safe.
    """

    def __init__(self, fee_rate: float):
        """
        Initializes the TradeMetrics instance.

        Args:
            fee_rate: The estimated trading fee rate per trade side (e.g., 0.0006 for 0.06%).
                      Should typically be the TAKER fee rate for market orders.
        """
        self.trades: List[Dict[str, Any]] = []
        if not isinstance(fee_rate, (int, float)) or fee_rate < 0:
            log_console(
                logging.ERROR,
                f"Metrics Init: Invalid fee rate ({fee_rate}). Setting to 0.0006 (0.06%).",
            )
            fee_rate = 0.0006
        self.fee_rate = fee_rate
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.total_fees_paid = 0.0
        self.last_summary_time: Optional[dt.datetime] = None
        self.lock = threading.Lock()

    def add_trade(
        self,
        symbol: str,
        entry_time: dt.datetime,
        exit_time: dt.datetime,
        side: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        leverage: Union[int, str],
    ):
        """
        Adds a completed trade to the tracker and calculates its P&L.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT").
            entry_time: Datetime of trade entry.
            exit_time: Datetime of trade exit.
            side: "Buy" (Long) or "Sell" (Short).
            entry_price: Average entry price.
            exit_price: Average exit price.
            qty: Quantity traded (absolute value).
            leverage: Leverage used for the trade (can be int or string from API).
        """
        with self.lock:
            try:
                leverage_int = 1  # Default leverage if conversion fails
                try:
                    # Handle potential empty string or None for leverage
                    leverage_val = float(leverage) if leverage is not None and str(leverage).strip() != "" else 1.0
                    # Ensure positive integer leverage >= 1
                    leverage_int = max(1, int(leverage_val))
                except (ValueError, TypeError):
                    log_console(
                        logging.ERROR,
                        f"Metrics: Invalid leverage value '{leverage}' for {symbol}. Using 1.",
                        symbol=symbol,
                    )

                # Validate inputs more robustly
                if not all(
                    [
                        isinstance(symbol, str) and symbol,
                        isinstance(entry_time, dt.datetime),
                        isinstance(exit_time, dt.datetime),
                        isinstance(side, str) and side in ["Buy", "Sell"],
                        isinstance(entry_price, (int, float)) and entry_price > FLOAT_COMPARISON_TOLERANCE,
                        isinstance(exit_price, (int, float)) and exit_price > FLOAT_COMPARISON_TOLERANCE,
                        isinstance(qty, (int, float)) and qty > FLOAT_COMPARISON_TOLERANCE,
                        isinstance(leverage_int, int) and leverage_int >= 1,
                    ]
                ):
                    log_console(
                        logging.ERROR,
                        f"Metrics: Invalid trade data received for {symbol}. Cannot log. Data: "
                        f"entry_t={entry_time}, exit_t={exit_time}, side={side}, "
                        f"entry_p={entry_price}, exit_p={exit_price}, qty={qty}, lev='{leverage}'",
                        symbol=symbol,
                    )
                    return

                abs_qty = abs(qty)
                price_diff = exit_price - entry_price if side == "Buy" else entry_price - exit_price
                # P&L for Linear Contracts (Quantity * Price Difference)
                gross_pnl = price_diff * abs_qty

                # Fees Estimate (Apply fee rate to traded value on entry AND exit)
                entry_value = abs_qty * entry_price
                exit_value = abs_qty * exit_price
                # Ensure fee rate is applied correctly
                fees = (entry_value * self.fee_rate) + (exit_value * self.fee_rate)
                net_pnl = gross_pnl - fees

                is_win = net_pnl > FLOAT_COMPARISON_TOLERANCE
                self.total_trades += 1
                if is_win:
                    self.wins += 1
                else:
                    self.losses += 1
                self.total_pnl += net_pnl
                self.total_fees_paid += fees

                trade_duration_sec = max(0.0, (exit_time - entry_time).total_seconds())

                # Consistent precision for logging (adjust based on typical asset values)
                pnl_precision = 4
                price_precision = 6  # Adjust based on typical asset precision
                qty_precision = 6  # Adjust based on typical asset precision

                trade = {
                    "symbol": symbol,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
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
                self.trades.append(trade)

                # Log metrics in CSV-like format
                log_metrics(
                    f"TRADE,{symbol},{trade['entry_time']},{trade['exit_time']},{trade['duration_seconds']},"
                    f"{side},{trade['entry_price']:.{price_precision}f},{trade['exit_price']:.{price_precision}f},"
                    f"{trade['qty']:.{qty_precision}f},{trade['leverage']},{trade['gross_pnl']:.{pnl_precision}f},{trade['fees']:.{pnl_precision}f},"
                    f"{trade['net_pnl']:.{pnl_precision}f},{'Win' if is_win else 'Loss'}"
                )
            except Exception as e:
                log_console(
                    logging.ERROR,
                    f"Error processing or logging trade metrics for {symbol}: {e}",
                    symbol=symbol,
                    exc_info=True,
                )

    def log_summary(self, symbol: Optional[str] = None, force: bool = False, interval: int = 3600):
        """Logs summary performance metrics if interval elapsed or forced."""
        with self.lock:
            current_time = dt.datetime.now(dt.timezone.utc)
            log_now = force or self.total_trades == 0  # Log if forced or first time

            if not log_now and self.last_summary_time:
                time_since_last = (current_time - self.last_summary_time).total_seconds()
                if time_since_last >= interval:
                    log_now = True

            if not log_now:
                return

            if self.total_trades > 0:
                win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0
                avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0
                profit_factor = 0.0
                total_gains = sum(t["net_pnl"] for t in self.trades if t["is_win"])
                total_losses = abs(sum(t["net_pnl"] for t in self.trades if not t["is_win"]))
                # Use tolerance for division by zero check
                if total_losses > FLOAT_COMPARISON_TOLERANCE:
                    profit_factor = total_gains / total_losses
                elif total_gains > FLOAT_COMPARISON_TOLERANCE:  # Handle case with wins but no losses
                    profit_factor = float("inf")

                pnl_precision = 4
                summary = (
                    f"SUMMARY,{symbol or 'ALL_SYMBOLS'},TotalTrades={self.total_trades},Wins={self.wins},Losses={self.losses},"
                    f"WinRate={win_rate:.2f}%,ProfitFactor={profit_factor:.2f},"
                    f"TotalNetPnL={self.total_pnl:.{pnl_precision}f} USDT,"
                    f"AvgNetPnL={avg_pnl:.{pnl_precision}f} USDT,TotalFees={self.total_fees_paid:.{pnl_precision}f} USDT"
                )
            else:
                summary = f"SUMMARY,{symbol or 'ALL_SYMBOLS'},No trades executed yet."

            log_metrics(summary)
            self.last_summary_time = current_time

    def save_on_shutdown(self):
        """Logs a final summary when the bot is shutting down."""
        with self.lock:
            log_console(logging.INFO, "Generating final trade metrics summary...")
            self.log_summary(force=True)


# --- Single Symbol Trading Logic ---
class SymbolTrader:
    """
    Manages trading activities for a single symbol based on configuration. Uses Bybit V5 API.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        config: Dict[str, Any],
        symbol_cfg: Dict[str, Any],
        dry_run: bool,
        metrics: TradeMetrics,
        session: HTTP,
        category: str,
    ):
        """
        Initializes the SymbolTrader instance.

        Args:
            api_key, api_secret: Bybit credentials.
            config: Full application configuration.
            symbol_cfg: Symbol-specific configuration (includes INTERNAL).
            dry_run: Global dry-run flag.
            metrics: Shared TradeMetrics instance.
            session: Shared authenticated Bybit HTTP session.
            category: Trading category (e.g., "linear", "spot").

        Raises:
            RuntimeError: If critical initialization steps fail.
        """
        self.config = config
        self.symbol_cfg = symbol_cfg
        self.dry_run = dry_run
        self.metrics = metrics
        self.session = session
        self.category = category  # Passed from main orchestrator

        self.symbol: str = symbol_cfg.get("SYMBOL", "MISSING_SYMBOL")
        self.timeframe: str = str(symbol_cfg.get("TIMEFRAME", "15"))
        self.leverage: int = int(symbol_cfg.get("LEVERAGE", 5))
        self.use_websocket: bool = symbol_cfg.get("USE_WEBSOCKET", True)
        self.strategy_cfg: Dict[str, Any] = symbol_cfg.get("STRATEGY_CONFIG", {})
        self.risk_cfg: Dict[str, Any] = config.get("RISK_CONFIG", {})
        self.bot_cfg: Dict[str, Any] = config.get("BOT_CONFIG", {})
        self.internal_cfg: Dict[str, Any] = symbol_cfg.get("INTERNAL", {})

        # Instrument details - Initialize with defaults
        self.min_order_qty: float = 0.000001  # Set a very small default non-zero value
        self.qty_step: str = "0.000001"
        self.qty_step_float: float = 0.000001
        self.qty_precision: int = 6
        self.tick_size: str = "0.01"
        self.tick_size_float: float = 0.01
        self.price_precision: int = 2

        # State variables
        self.last_exit_time: Optional[dt.datetime] = None
        self.kline_queue: Queue = Queue()
        self.kline_df: Optional[pd.DataFrame] = None
        self.higher_tf_cache: Optional[bool] = None
        self.higher_tf_cache_time: Optional[dt.datetime] = None
        # Real or simulated position state
        self.current_trade: Optional[Dict[str, Any]] = None
        self.order_confirm_retries: int = self.bot_cfg.get("ORDER_CONFIRM_RETRIES", 3)
        self.order_confirm_delay: float = float(self.bot_cfg.get("ORDER_CONFIRM_DELAY_SECONDS", 2.0))
        self.initialization_successful: bool = False
        self.last_kline_update_time: Optional[dt.datetime] = None

        if self.dry_run:
            log_console(
                logging.WARNING,
                "DRY RUN MODE enabled. No real orders will be placed.",
                symbol=self.symbol,
            )

        try:
            if not self._fetch_instrument_info():
                raise RuntimeError(f"Failed to fetch essential instrument info for {self.symbol}.")
            if not self.dry_run and self.category != "spot":  # Leverage/margin setup not applicable to spot
                self._initial_setup()
            self.initialization_successful = True
            log_console(
                logging.DEBUG,
                f"Trader for {self.symbol} initialized successfully.",
                symbol=self.symbol,
            )
        except Exception as e:
            log_console(
                logging.CRITICAL,
                f"Initialization FAILED for trader {self.symbol}: {e}",
                symbol=self.symbol,
                exc_info=False,
            )
            # Raise a specific error to be caught by the orchestrator
            raise RuntimeError(f"Trader Init Failed for {self.symbol}") from e

    def _fetch_instrument_info(self) -> bool:
        """Fetches instrument details (V5 API)."""
        log_console(logging.DEBUG, "Fetching instrument info...", symbol=self.symbol)
        try:
            # Use V5 endpoint get_instruments_info
            info = self.session.get_instruments_info(category=self.category, symbol=self.symbol)

            if info and info.get("retCode") == 0:
                result_list = info.get("result", {}).get("list", [])
                if result_list:
                    instrument = result_list[0]
                    lot_size_filter = instrument.get("lotSizeFilter", {})
                    price_filter = instrument.get("priceFilter", {})

                    try:
                        # Safely convert filter values to floats, handling potential None or empty strings
                        self.min_order_qty = float(lot_size_filter.get("minOrderQty", "0"))
                        self.qty_step = lot_size_filter.get("qtyStep", "1")  # Keep original string
                        self.qty_step_float = float(self.qty_step)
                        self.tick_size = price_filter.get("tickSize", "1")  # Keep original string
                        tick_size_float = float(self.tick_size)
                        self.tick_size_float = tick_size_float  # Store float version

                        # Calculate precision safely based on decimal places of the step/tick size string
                        if "." in self.qty_step and self.qty_step_float > FLOAT_COMPARISON_TOLERANCE:
                            # Count chars after decimal, ignore trailing zeros if needed (usually not for steps)
                            decimal_part = self.qty_step.split(".")[-1]
                            self.qty_precision = len(decimal_part)
                        else:
                            self.qty_precision = 0

                        if "." in self.tick_size and tick_size_float > FLOAT_COMPARISON_TOLERANCE:
                            decimal_part = self.tick_size.split(".")[-1]
                            self.price_precision = len(decimal_part)
                        else:
                            self.price_precision = 0

                    except (ValueError, TypeError) as conv_e:
                        log_console(
                            logging.ERROR,
                            f"Failed to convert instrument filter values: {conv_e}. Filters: {lot_size_filter}, {price_filter}",
                            symbol=self.symbol,
                        )
                        return False

                    # Validate fetched values (ensure they are positive)
                    if (
                        self.min_order_qty <= FLOAT_COMPARISON_TOLERANCE
                        or self.qty_step_float <= FLOAT_COMPARISON_TOLERANCE
                        or tick_size_float <= FLOAT_COMPARISON_TOLERANCE
                    ):
                        log_console(
                            logging.ERROR,
                            f"Fetched invalid instrument details: MinQty={self.min_order_qty}, QtyStep={self.qty_step_float}, TickSize={tick_size_float}",
                            symbol=self.symbol,
                        )
                        return False

                    log_console(
                        logging.INFO,
                        f"Instrument Info: Min Qty={self.min_order_qty}, "
                        f"Qty Step={self.qty_step} (Float: {self.qty_step_float:.{max(8, self.qty_precision)}f}, Precision: {self.qty_precision}), "
                        f"Tick Size={self.tick_size} (Float: {tick_size_float:.{max(8, self.price_precision)}f}, Precision: {self.price_precision})",
                        symbol=self.symbol,
                    )
                    return True
                else:
                    log_console(
                        logging.ERROR,
                        f"Instrument info fetch failed: Empty result list for {self.symbol}.",
                        symbol=self.symbol,
                    )
                    return False
            else:
                error_msg = info.get("retMsg", "N/A") if info else "Empty response"
                error_code = info.get("retCode", "N/A")
                log_console(
                    logging.ERROR,
                    f"Instrument info fetch failed: {error_msg} (Code: {error_code})",
                    symbol=self.symbol,
                )
                # Specific check for invalid symbol error (e.g., 10001 or relevant message)
                if error_code == 10001 or "invalid symbol" in error_msg.lower():
                    log_console(
                        logging.ERROR,
                        f"Check if symbol '{self.symbol}' is valid for the '{self.category}' category on Bybit.",
                        symbol=self.symbol,
                    )
                return False
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Exception during instrument info fetch: {e}",
                symbol=self.symbol,
                exc_info=True,
            )
            return False

    def _initial_setup(self):
        """Sets leverage (V5 API). Only for non-spot categories."""
        if self.category == "spot":
            return  # Skip for spot

        log_console(
            logging.INFO,
            f"Performing initial setup: Setting leverage to {self.leverage}x...",
            symbol=self.symbol,
        )
        if self.dry_run:
            log_console(
                logging.INFO,
                "Skipping leverage setting in dry run mode.",
                symbol=self.symbol,
            )
            return

        try:
            response = self.session.set_leverage(
                category=self.category,
                symbol=self.symbol,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage),
            )
            if response and response.get("retCode") == 0:
                log_console(
                    logging.INFO,
                    f"Leverage successfully set to {self.leverage}x.",
                    symbol=self.symbol,
                )
            else:
                error_msg = response.get("retMsg", "N/A") if response else "Empty response"
                error_code = response.get("retCode", "N/A")
                # V5 Error Code 110043: leverage not modified
                # V5 Error Code 110026: Margin mode related error (can sometimes occur if leverage matches cross margin mode default)
                if (
                    error_code == 110043
                    or "leverage not modified" in error_msg.lower()
                    or "same leverage" in error_msg.lower()
                    or error_code == 110026
                ):
                    log_console(
                        logging.INFO,
                        f"Leverage already set to {self.leverage}x or not modified (Code: {error_code}, Msg: '{error_msg}').",
                        symbol=self.symbol,
                    )
                else:
                    # Log as warning, not critical, as trading might still proceed if leverage was set previously
                    log_console(
                        logging.WARNING,
                        f"Failed to set leverage: {error_msg} (Code: {error_code}). Check API permissions and account settings. Continuing...",
                        symbol=self.symbol,
                    )

            # Add margin mode setting here if needed (requires careful checking of V5 parameters)
            # Example:
            # margin_mode = self.symbol_cfg.get("MARGIN_MODE", "ISOLATED") # Or "CROSSED"
            # if margin_mode.upper() == "ISOLATED":
            #     response_margin = self.session.switch_margin_mode(category=self.category, symbol=self.symbol, tradeMode=0, buyLeverage=str(self.leverage), sellLeverage=str(self.leverage))
            # elif margin_mode.upper() == "CROSSED":
            #     response_margin = self.session.switch_margin_mode(category=self.category, symbol=self.symbol, tradeMode=1, buyLeverage=str(self.leverage), sellLeverage=str(self.leverage))
            # ... handle response_margin ...

        except Exception as e:
            log_console(
                logging.WARNING,
                f"Exception during initial setup (leverage/margin): {e}. Continuing...",
                symbol=self.symbol,
                exc_info=True,
            )

    def _websocket_callback(self, message: Dict[str, Any]):
        """Callback for processing V5 WebSocket kline messages for this specific trader."""
        try:
            # log_console(logging.DEBUG, f"WS Received Raw: {message}", symbol=self.symbol) # Verbose
            if not isinstance(message, dict):
                return

            topic = message.get("topic")
            data_list = message.get("data")

            if not topic or not isinstance(data_list, list):
                return

            # V5 Topic format: kline.{interval}.{symbol}
            # Check if the message belongs to this trader instance
            topic_parts = topic.split(".")
            if not (
                len(topic_parts) == 3
                and topic_parts[0] == "kline"
                and topic_parts[1] == self.timeframe
                and topic_parts[2] == self.symbol
            ):
                # This message is not for this trader instance, ignore silently.
                return

            if not data_list:
                return

            for kline_raw in data_list:
                if not isinstance(kline_raw, dict):
                    continue

                is_confirmed = kline_raw.get("confirm", False)
                process_confirmed_only = self.bot_cfg.get("PROCESS_CONFIRMED_KLINE_ONLY", True)

                # Process only confirmed candles if configured, otherwise process all
                if not process_confirmed_only or (process_confirmed_only and is_confirmed):
                    try:
                        # V5 Kline data fields: start, open, high, low, close, volume, turnover
                        timestamp_ms = int(kline_raw["start"])
                        kline_processed = {
                            # Use UTC timezone explicitly
                            "timestamp": pd.to_datetime(timestamp_ms, unit="ms", utc=True),
                            "open": float(kline_raw["open"]),
                            "high": float(kline_raw["high"]),
                            "low": float(kline_raw["low"]),
                            "close": float(kline_raw["close"]),
                            "volume": float(kline_raw["volume"]),
                            # Turnover might not always be present
                            "turnover": float(kline_raw.get("turnover", 0.0)),
                        }
                        # Put the dictionary into the queue
                        self.kline_queue.put(kline_processed)
                        # ts_str = kline_processed['timestamp'].strftime('%H:%M:%S')
                        # log_console(logging.DEBUG, f"WS Queued {'Confirmed ' if is_confirmed else ''}Kline @ {ts_str}", symbol=self.symbol)
                    except (KeyError, ValueError, TypeError) as conv_e:
                        log_console(
                            logging.ERROR,
                            f"WebSocket callback: Error converting kline data: {conv_e}. RawKline: {kline_raw}",
                            symbol=self.symbol,
                        )

        except Exception as e:
            log_console(
                logging.ERROR,
                f"Unexpected error in WebSocket callback for {self.symbol}: {e}",
                symbol=self.symbol,
                exc_info=True,
            )

    def _process_kline_queue(self) -> bool:
        """
        Processes kline data from WebSocket queue, updates DataFrame, recalculates indicators.

        Returns:
            True if the DataFrame was updated AND indicators were recalculated successfully.
        """
        df_updated = False
        recalculated_ok = False
        processed_count = 0
        max_items_per_cycle = 100  # Limit processing per call to avoid blocking main loop

        while not self.kline_queue.empty() and processed_count < max_items_per_cycle:
            try:
                kline_dict = self.kline_queue.get_nowait()
                processed_count += 1

                # Create a DataFrame from the single kline dictionary, ensure index is UTC
                new_row = pd.DataFrame([kline_dict]).set_index("timestamp")
                if not new_row.index.tz:
                    new_row.index = new_row.index.tz_localize("UTC")

                if self.kline_df is None or self.kline_df.empty:
                    log_console(
                        logging.DEBUG,
                        "kline_df empty, initializing from first WS kline.",
                        symbol=self.symbol,
                    )
                    self.kline_df = new_row
                    df_updated = True
                else:
                    # Ensure existing index is DatetimeIndex and has UTC timezone before proceeding
                    if not isinstance(self.kline_df.index, pd.DatetimeIndex):
                        log_console(
                            logging.ERROR,
                            "Internal kline_df index not DatetimeIndex! Resetting DF.",
                            symbol=self.symbol,
                        )
                        self.kline_df = new_row  # Re-initialize with the new row
                        df_updated = True
                        continue  # Skip to next item or recalc
                    if not self.kline_df.index.tz:
                        log_console(
                            logging.WARNING,
                            "Internal kline_df index missing timezone. Localizing to UTC.",
                            symbol=self.symbol,
                        )
                        self.kline_df.index = self.kline_df.index.tz_localize("UTC", ambiguous="infer")

                    idx_to_check = new_row.index[0]
                    if idx_to_check in self.kline_df.index:
                        # Update existing candle using DataFrame.update for efficiency
                        # This preserves existing columns (like indicators) not present in new_row
                        self.kline_df.update(new_row)
                        # log_console(logging.DEBUG, f"WS updated existing kline: {idx_to_check}", symbol=self.symbol) # Can be noisy
                        df_updated = True
                    else:
                        # Append new candle using pd.concat
                        self.kline_df = pd.concat([self.kline_df, new_row])
                        # Ensure index remains sorted after appending
                        self.kline_df.sort_index(inplace=True)
                        # log_console(logging.DEBUG, f"WS appended new kline: {idx_to_check}", symbol=self.symbol)

                        # Trim DataFrame to maintain reasonable size
                        # Use calculated KLINE_LIMIT + buffer
                        max_rows = self.internal_cfg.get("KLINE_LIMIT", 120) + self.bot_cfg.get(
                            "KLINE_LIMIT_BUFFER", 50
                        )
                        if len(self.kline_df) > max_rows:
                            self.kline_df = self.kline_df.iloc[-max_rows:]
                        df_updated = True

            except Empty:
                break  # Queue is empty
            except Exception as e:
                log_console(
                    logging.ERROR,
                    f"Error processing kline queue item for {self.symbol}: {e}",
                    symbol=self.symbol,
                    exc_info=True,
                )

        # Recalculate indicators only if DF was updated AND is not empty
        if df_updated and self.kline_df is not None and not self.kline_df.empty:
            log_console(
                logging.DEBUG,
                f"Recalculating indicators for {self.symbol} after processing {processed_count} WS item(s)...",
                symbol=self.symbol,
            )
            # Pass a copy to the calculation function to avoid side effects if needed
            # Ensure the DataFrame passed has the correct structure expected by the function
            try:
                calculated_df = calculate_indicators_momentum(self.kline_df.copy(), self.strategy_cfg, self.config)
                if calculated_df is not None and not calculated_df.empty:
                    # Replace internal DF with the one containing new indicators
                    # Ensure the new DF index is also timezone-aware (UTC)
                    if not calculated_df.index.tz:
                        calculated_df.index = calculated_df.index.tz_localize("UTC")
                    self.kline_df = calculated_df
                    self.last_kline_update_time = dt.datetime.now(dt.timezone.utc)
                    recalculated_ok = True
                    log_console(
                        logging.DEBUG,
                        "Indicators recalculated successfully after WS update.",
                        symbol=self.symbol,
                    )
                else:
                    log_console(
                        logging.WARNING,
                        "Indicator recalculation failed after WS update. Stale indicators may be used.",
                        symbol=self.symbol,
                    )
                    # Keep df_updated as True, but recalculation failed
                    recalculated_ok = False
            except Exception as calc_e:
                log_console(
                    logging.ERROR,
                    f"Exception during indicator recalculation after WS update for {self.symbol}: {calc_e}",
                    symbol=self.symbol,
                    exc_info=True,
                )
                recalculated_ok = False

        # df_updated is True but kline_df became None or empty (should not happen ideally)
        elif df_updated:
            log_console(
                logging.ERROR,
                "kline_df became empty/None after WS processing. Cannot recalculate.",
                symbol=self.symbol,
            )
            recalculated_ok = False

        # Return True only if DF updated AND indicators recalculated successfully
        return df_updated and recalculated_ok

    def get_ohlcv_rest(self, timeframe: Optional[str] = None, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV via REST (V5), calculates indicators."""
        tf = timeframe or self.timeframe
        # Use calculated KLINE_LIMIT + small buffer for fetching
        default_lim = self.internal_cfg.get("KLINE_LIMIT", 120) + 5
        # Cap limit at Bybit's max per request and ensure it's positive
        lim = max(1, min(limit or default_lim, MAX_KLINE_LIMIT_PER_REQUEST))

        log_console(
            logging.DEBUG,
            f"Fetching {lim} klines for timeframe {tf} via REST API...",
            symbol=self.symbol,
        )

        try:
            response = self.session.get_kline(category=self.category, symbol=self.symbol, interval=tf, limit=lim)

            if response and response.get("retCode") == 0:
                kline_list = response.get("result", {}).get("list", [])
                if kline_list:
                    # V5 format: [timestamp_ms_str, open_str, high_str, low_str, close_str, volume_str, turnover_str]
                    df = pd.DataFrame(
                        kline_list,
                        columns=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "turnover",
                        ],
                    )
                    # V5 returns newest first, reverse to have oldest first for TA libraries
                    df = df.iloc[::-1].reset_index(drop=True)

                    # Convert and Clean
                    try:
                        # Convert timestamp first, coerce errors to NaT, set UTC timezone
                        df["timestamp"] = pd.to_datetime(
                            pd.to_numeric(df["timestamp"], errors="coerce"),
                            unit="ms",
                            utc=True,
                        )
                    except Exception as ts_e:
                        log_console(
                            logging.ERROR,
                            f"REST KLine: Failed timestamp conversion: {ts_e}",
                            symbol=self.symbol,
                        )
                        return None
                    # Drop rows where timestamp conversion failed
                    df.dropna(subset=["timestamp"], inplace=True)
                    if df.empty:
                        log_console(
                            logging.WARNING,
                            "REST KLine: DataFrame empty after timestamp conversion/dropna.",
                            symbol=self.symbol,
                        )
                        return None
                    df.set_index("timestamp", inplace=True)

                    # Convert OHLCV columns to numeric, coercing errors
                    ohlcv_cols = ["open", "high", "low", "close", "volume", "turnover"]
                    for col in ohlcv_cols:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    # Drop rows with NaNs in essential OHLCV columns
                    essential_cols = ["open", "high", "low", "close", "volume"]
                    initial_len = len(df)
                    df.dropna(subset=essential_cols, inplace=True)
                    if len(df) < initial_len:
                        log_console(
                            logging.WARNING,
                            f"REST KLine: Dropped {initial_len - len(df)} rows with NaNs in OHLCV.",
                            symbol=self.symbol,
                        )
                    if df.empty:
                        log_console(
                            logging.WARNING,
                            "REST KLine: DataFrame empty after OHLCV dropna.",
                            symbol=self.symbol,
                        )
                        return None

                    # Check Minimum Length required for calculations (use calculated internal limit)
                    min_len_for_calc = self.internal_cfg.get("KLINE_LIMIT", MIN_KLINE_RECORDS_FOR_CALC)
                    if len(df) < min_len_for_calc:
                        log_console(
                            logging.WARNING,
                            f"REST KLine: Not enough valid data ({len(df)} < {min_len_for_calc}) for calc.",
                            symbol=self.symbol,
                        )
                        # Return None if not enough data for reliable indicators
                        return None

                    log_console(
                        logging.DEBUG,
                        f"REST Fetch successful. Cleaned data shape: {df.shape}",
                        symbol=self.symbol,
                    )

                    # Calculate Indicators
                    df_with_indicators = calculate_indicators_momentum(df, self.strategy_cfg, self.config)
                    if df_with_indicators is None or df_with_indicators.empty:
                        log_console(
                            logging.WARNING,
                            "REST KLine: Indicator calculation failed.",
                            symbol=self.symbol,
                        )
                        # Return None if indicator calculation fails
                        return None
                    else:
                        self.last_kline_update_time = dt.datetime.now(dt.timezone.utc)
                        return df_with_indicators

                else:  # Empty kline_list from API
                    log_console(
                        logging.WARNING,
                        f"REST KLine fetch returned empty list for {self.symbol} {tf}.",
                        symbol=self.symbol,
                    )
                    return None
            else:  # API error
                error_msg = response.get("retMsg", "N/A") if response else "Empty response"
                error_code = response.get("retCode", "N/A")
                log_console(
                    logging.WARNING,
                    f"REST KLine fetch failed: {error_msg} (Code: {error_code})",
                    symbol=self.symbol,
                )
                if error_code == 10001 or "invalid symbol" in error_msg.lower():
                    log_console(
                        logging.ERROR,
                        f"Check symbol '{self.symbol}' and timeframe '{tf}' validity for category '{self.category}'.",
                        symbol=self.symbol,
                    )
                # Add other specific error handling if needed (e.g., rate limits 10006)
                return None
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Exception during REST KLine fetch for {self.symbol}: {e}",
                symbol=self.symbol,
                exc_info=True,
            )
            return None

    def get_ohlcv(self) -> Optional[pd.DataFrame]:
        """
        Provides latest OHLCV and indicators, prioritizing WebSocket updates.
        Falls back to REST if needed or WS data is stale.

        Returns:
            A *copy* of the internal DataFrame with indicators, or None if unavailable.
        """
        ws_processed_and_recalculated = False
        if self.use_websocket:
            # Process queue & recalculate indicators. Returns True if successful.
            ws_processed_and_recalculated = self._process_kline_queue()

        # Determine if REST fetch is needed
        needs_rest_fetch = False
        if self.kline_df is None or self.kline_df.empty:
            needs_rest_fetch = True
            log_console(
                logging.DEBUG,
                "kline_df is empty, attempting REST fetch.",
                symbol=self.symbol,
            )
        elif not self.use_websocket:
            # Always fetch via REST if WS disabled, but use existing DF if available and recent enough
            staleness_limit = self.bot_cfg.get("KLINE_STALENESS_SECONDS", 300)
            now_utc = dt.datetime.now(dt.timezone.utc)
            if (
                self.last_kline_update_time is None
                or (now_utc - self.last_kline_update_time).total_seconds() > staleness_limit
            ):
                needs_rest_fetch = True
                log_console(
                    logging.DEBUG,
                    "WS disabled and data stale, using REST for kline data.",
                    symbol=self.symbol,
                )
            else:
                log_console(
                    logging.DEBUG,
                    "WS disabled, using recent REST data.",
                    symbol=self.symbol,
                )

        elif self.use_websocket and not ws_processed_and_recalculated:
            # WS enabled, but queue empty OR processing/recalc failed. Check staleness.
            staleness_limit = self.bot_cfg.get("KLINE_STALENESS_SECONDS", 300)
            now_utc = dt.datetime.now(dt.timezone.utc)
            if (
                self.last_kline_update_time is None
                or (now_utc - self.last_kline_update_time).total_seconds() > staleness_limit
            ):
                needs_rest_fetch = True
                log_console(
                    logging.WARNING,
                    f"Kline data appears stale (> {staleness_limit}s ago) or WS recalc failed. Attempting REST.",
                    symbol=self.symbol,
                )

        # Perform REST fetch if needed
        if needs_rest_fetch:
            rest_df_with_indicators = self.get_ohlcv_rest()
            if rest_df_with_indicators is not None and not rest_df_with_indicators.empty:
                # Update internal state, ensuring index is UTC
                if not rest_df_with_indicators.index.tz:
                    rest_df_with_indicators.index = rest_df_with_indicators.index.tz_localize("UTC")
                self.kline_df = rest_df_with_indicators
                log_console(
                    logging.DEBUG,
                    "Updated kline_df successfully via REST API.",
                    symbol=self.symbol,
                )
            else:
                log_console(
                    logging.WARNING,
                    "REST API fetch failed. Kline data may be missing or stale.",
                    symbol=self.symbol,
                )
                # Don't return here, let it return the potentially stale kline_df if it exists

        # Return a copy of the current state to prevent external modification
        return self.kline_df.copy() if self.kline_df is not None else None

    def get_position(self, log_error: bool = True) -> Optional[Dict[str, Any]]:
        """Fetches current position (V5 API), returns simulated state in dry run."""
        if self.dry_run:
            # Return a deep copy to prevent modification of the internal state
            return deepcopy(self.current_trade) if self.current_trade else None

        try:
            # V5 get_positions endpoint
            response = self.session.get_positions(category=self.category, symbol=self.symbol)

            if response and response.get("retCode") == 0:
                position_list = response.get("result", {}).get("list", [])
                for pos in position_list:
                    if not isinstance(pos, dict):
                        continue
                    # Check if the position matches the symbol we are interested in
                    if pos.get("symbol") == self.symbol:
                        try:
                            # "Buy", "Sell", or "None"
                            pos_side = pos.get("side")
                            pos_size_str = pos.get("size", "0")
                            pos_size = float(pos_size_str)

                            # Check if position exists (significant size and valid side)
                            # Use a small fraction of min_order_qty or absolute tolerance
                            min_significant_qty = max(FLOAT_COMPARISON_TOLERANCE, self.min_order_qty * 0.01)

                            if pos_side in ["Buy", "Sell"] and abs(pos_size) >= min_significant_qty:
                                # Convert fields and store consistently
                                pos["size"] = pos_size  # Store as float
                                pos["avgPrice"] = float(pos.get("avgPrice", "0"))
                                pos["side"] = pos_side  # Already string
                                pos["leverage"] = pos.get("leverage", str(self.leverage))  # Store leverage as string
                                pos["liqPrice"] = float(pos.get("liqPrice", "0"))
                                pos["unrealisedPnl"] = float(pos.get("unrealisedPnl", "0"))
                                pos["markPrice"] = float(pos.get("markPrice", "0"))
                                pos["positionValue"] = float(pos.get("positionValue", "0"))
                                pos["stopLoss"] = pos.get("stopLoss", "")  # String or empty
                                pos["takeProfit"] = pos.get("takeProfit", "")  # String or empty

                                # Attempt to add entry time
                                pos["entry_time"] = None  # Initialize
                                # Prioritize internal state if available and matches symbol/side/approx size
                                if (
                                    self.current_trade
                                    and self.current_trade.get("symbol") == self.symbol
                                    and self.current_trade.get("side") == pos_side
                                    and math.isclose(
                                        abs(self.current_trade.get("size", 0)),
                                        abs(pos_size),
                                        rel_tol=0.01,
                                    )
                                ):
                                    pos["entry_time"] = self.current_trade.get("entry_time")
                                # Fallback to API createdTime (convert ms string to datetime)
                                elif pos.get("createdTime"):
                                    try:
                                        pos["entry_time"] = pd.to_datetime(int(pos["createdTime"]), unit="ms", utc=True)
                                    except (ValueError, TypeError, KeyError):
                                        pass  # Ignore parsing errors

                                return pos  # Return the first significant position found for the symbol
                        except (ValueError, TypeError, KeyError) as conv_e:
                            log_console(
                                logging.ERROR,
                                f"Position fetch: Error converting fields: {conv_e}. PosData: {pos}",
                                symbol=self.symbol,
                            )
                            continue  # Try next item in list if conversion fails

                return None  # No active position found for symbol
            else:
                if log_error:
                    error_msg = response.get("retMsg", "N/A") if response else "Empty response"
                    error_code = response.get("retCode", "N/A")
                    log_console(
                        logging.ERROR,
                        f"Position fetch failed: {error_msg} (Code: {error_code})",
                        symbol=self.symbol,
                    )
                return None
        except Exception as e:
            if log_error:
                log_console(
                    logging.ERROR,
                    f"Exception during position fetch for {self.symbol}: {e}",
                    symbol=self.symbol,
                    exc_info=True,
                )
            return None

    def confirm_order(self, order_id: str, expected_qty: float) -> Tuple[float, Optional[float]]:
        """Confirms order status by polling history (V5 API), returns filled qty and avg price."""
        if not order_id:
            log_console(
                logging.ERROR,
                "Confirm Order: Invalid Order ID provided.",
                symbol=self.symbol,
            )
            return 0.0, None

        if self.dry_run:
            log_console(
                logging.INFO,
                f"[DRY RUN] Simulating confirmation for order {order_id}",
                symbol=self.symbol,
            )
            sim_entry_price: Optional[float] = None
            # Try to get a realistic price from latest kline data
            if self.kline_df is not None and not self.kline_df.empty:
                try:
                    sim_entry_price = float(self.kline_df["close"].iloc[-1])
                except (IndexError, ValueError, TypeError):
                    pass
            # Fallback if price unavailable
            if sim_entry_price is None or sim_entry_price <= FLOAT_COMPARISON_TOLERANCE:
                sim_entry_price = 1.0  # Dummy price
                log_console(
                    logging.WARNING,
                    f"[DRY RUN] Cannot estimate entry price for {order_id}, using dummy {sim_entry_price}.",
                    symbol=self.symbol,
                )
            log_console(
                logging.DEBUG,
                f"[DRY RUN] Using simulated avg fill price: {sim_entry_price} for {order_id}",
                symbol=self.symbol,
            )
            # Assume full fill in dry run
            return expected_qty, sim_entry_price

        # Real Order Confirmation
        last_filled_qty = 0.0
        last_avg_price = None
        last_status = "Unknown"

        for attempt in range(self.order_confirm_retries + 1):
            if attempt > 0:
                time.sleep(self.order_confirm_delay)
            log_console(
                logging.DEBUG,
                f"Confirming order {order_id} (Attempt {attempt + 1}/{self.order_confirm_retries + 1})...",
                symbol=self.symbol,
            )

            try:
                # Use get_order_history for V5 - more reliable for final status
                response = self.session.get_order_history(
                    category=self.category,
                    # symbol=self.symbol, # Optional: may speed up lookup but not strictly needed with orderId
                    orderId=order_id,
                    limit=1,  # We only need the specific order
                )

                if response and response.get("retCode") == 0:
                    order_list = response.get("result", {}).get("list", [])
                    if order_list:
                        order = order_list[0]
                        status = order.get("orderStatus")
                        last_status = status
                        filled_qty = 0.0
                        avg_price = None

                        try:
                            filled_qty_str = order.get("cumExecQty", "0")
                            filled_qty = float(filled_qty_str) if filled_qty_str else 0.0
                            avg_price_str = order.get("avgPrice", "0")
                            # Use avgPrice only if valid positive number string and filled qty > 0
                            if avg_price_str and filled_qty > FLOAT_COMPARISON_TOLERANCE:
                                try:
                                    temp_avg_price = float(avg_price_str)
                                    if temp_avg_price > FLOAT_COMPARISON_TOLERANCE:
                                        avg_price = temp_avg_price
                                except (ValueError, TypeError):
                                    pass  # Ignore conversion error for avgPrice

                            last_filled_qty = filled_qty
                            if avg_price is not None:
                                last_avg_price = avg_price

                        except (ValueError, TypeError, KeyError) as e:
                            log_console(
                                logging.ERROR,
                                f"Confirm Order {order_id}: Invalid qty/price format: {e}. Data: {order}",
                                symbol=self.symbol,
                            )

                        # V5 Statuses: Filled, PartiallyFilled, New, Untriggered, Rejected, Cancelled, PartiallyFilledCanceled, Deactivated, Expired, Active, Triggered, Created
                        if status == "Filled":
                            log_console(
                                logging.INFO,
                                f"Order {order_id} confirmed: Fully Filled. Qty={filled_qty:.{self.qty_precision}f}, AvgPrice={avg_price or 'N/A'}",
                                symbol=self.symbol,
                            )
                            return filled_qty, avg_price
                        elif status == "PartiallyFilled":
                            log_console(
                                logging.INFO,
                                f"Order {order_id} Partially Filled ({filled_qty:.{self.qty_precision}f}/{order.get('qty', '?')}). Waiting...",
                                symbol=self.symbol,
                            )
                        elif status in [
                            "New",
                            "Untriggered",
                            "Active",
                            "Created",
                            "Triggered",
                        ]:
                            log_console(
                                logging.DEBUG,
                                f"Order {order_id} status '{status}'. Waiting...",
                                symbol=self.symbol,
                            )
                        elif status in [
                            "Rejected",
                            "Cancelled",
                            "PartiallyFilledCanceled",
                            "Deactivated",
                            "Expired",
                        ]:
                            log_console(
                                logging.WARNING,
                                f"Order {order_id} failed/cancelled: '{status}'. Final Filled: {filled_qty:.{self.qty_precision}f}",
                                symbol=self.symbol,
                            )
                            return (
                                filled_qty,
                                last_avg_price,
                            )  # Return what was filled, even if zero
                        else:
                            log_console(
                                logging.WARNING,
                                f"Order {order_id} unhandled status: '{status}'. Continuing check.",
                                symbol=self.symbol,
                            )

                    else:  # Order not found in history yet
                        log_console(
                            logging.DEBUG,
                            f"Order {order_id} not in history yet (Attempt {attempt + 1}). Might still be 'New' or processing.",
                            symbol=self.symbol,
                        )
                        # Consider querying open orders as well if history is slow? (Can add complexity)
                        # response_open = self.session.get_open_orders(category=self.category, orderId=order_id) ...
                else:  # API call failed
                    error_msg = response.get("retMsg", "N/A") if response else "Empty response"
                    error_code = response.get("retCode", "N/A")
                    log_console(
                        logging.WARNING,
                        f"Order history fetch failed for {order_id} (Attempt {attempt + 1}): {error_msg} (Code: {error_code}).",
                        symbol=self.symbol,
                    )
                    # Handle specific errors like rate limits (10006) if needed
            except Exception as e:
                log_console(
                    logging.ERROR,
                    f"Exception during order confirmation attempt {attempt + 1} for {order_id}: {e}",
                    symbol=self.symbol,
                    exc_info=True,
                )

        # Loop finished without confirmation of 'Filled' or terminal state
        log_console(
            logging.ERROR,
            f"Order {order_id} confirmation TIMEOUT after {self.order_confirm_retries + 1} attempts. Last Status: '{last_status}'.",
            symbol=self.symbol,
        )
        log_console(
            logging.ERROR,
            f"Order {order_id} last known: FilledQty={last_filled_qty:.{self.qty_precision}f}, AvgPrice={last_avg_price or 'N/A'}. Manual check recommended!",
            symbol=self.symbol,
        )
        # Return the last known filled quantity and price, even if incomplete or zero
        return last_filled_qty, last_avg_price

    def place_order(
        self,
        side: str,
        qty: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        order_link_id: Optional[str] = None,
    ) -> Optional[str]:
        """Places market order with optional SL/TP (V5 API), confirms execution."""
        if side not in ["Buy", "Sell"]:
            log_console(logging.ERROR, f"Invalid order side: {side}", symbol=self.symbol)
            return None
        if not isinstance(qty, (float, int)) or qty <= FLOAT_COMPARISON_TOLERANCE:
            log_console(
                logging.ERROR,
                f"Order qty must be positive, got: {qty}",
                symbol=self.symbol,
            )
            return None
        # Check against min_order_qty with tolerance
        min_qty_tolerance = self.qty_step_float * 0.01
        if qty < self.min_order_qty - min_qty_tolerance:
            log_console(
                logging.ERROR,
                f"Order qty {qty:.{self.qty_precision}f} is below minimum {self.min_order_qty}",
                symbol=self.symbol,
            )
            return None

        # Format quantity and price strings according to instrument precision
        qty_str = f"{qty:.{self.qty_precision}f}"
        sl_str: Optional[str] = None
        if stop_loss_price is not None:
            if not isinstance(stop_loss_price, (float, int)) or stop_loss_price <= FLOAT_COMPARISON_TOLERANCE:
                log_console(
                    logging.WARNING,
                    f"Invalid SL price ({stop_loss_price}). SL not set.",
                    symbol=self.symbol,
                )
            else:
                sl_str = f"{stop_loss_price:.{self.price_precision}f}"

        tp_str: Optional[str] = None
        if take_profit_price is not None:
            if not isinstance(take_profit_price, (float, int)) or take_profit_price <= FLOAT_COMPARISON_TOLERANCE:
                log_console(
                    logging.WARNING,
                    f"Invalid TP price ({take_profit_price}). TP not set.",
                    symbol=self.symbol,
                )
            else:
                tp_str = f"{take_profit_price:.{self.price_precision}f}"

        log_prefix = f"{Fore.YELLOW + Style.BRIGHT}[DRY RUN]{Style.RESET_ALL} " if self.dry_run else ""
        side_color = Fore.GREEN if side == "Buy" else Fore.RED
        log_msg = f"{log_prefix}{side_color}{Style.BRIGHT}{side.upper()} MARKET ORDER:{Style.RESET_ALL} Qty={qty_str}"
        if sl_str:
            log_msg += f", SL={sl_str}"
        if tp_str:
            log_msg += f", TP={tp_str}"
        action_prefix = f"{Fore.MAGENTA + Style.BRIGHT}ACTION:{Style.RESET_ALL} [{self.symbol}] "
        print(action_prefix + log_msg)  # Print action immediately
        log_console(logging.INFO, log_msg.replace(log_prefix, "").strip(), symbol=self.symbol)  # Log without prefix

        if self.dry_run:
            # Use a more descriptive dry run order ID
            order_id = f"dryrun_{self.symbol}_{side.lower()}_{int(time.time() * 1000)}"
            sim_entry_price = 0.0
            # Try to get realistic price from klines
            if self.kline_df is not None and not self.kline_df.empty:
                try:
                    sim_entry_price = float(self.kline_df["close"].iloc[-1])
                except (IndexError, ValueError, TypeError):
                    pass
            if sim_entry_price <= FLOAT_COMPARISON_TOLERANCE:
                sim_entry_price = 1.0  # Dummy fallback

            # Update internal state for dry run
            self.current_trade = {
                "orderId": order_id,
                "symbol": self.symbol,
                "side": side,
                "size": qty,
                "avgPrice": sim_entry_price,
                "entry_time": dt.datetime.now(dt.timezone.utc),
                "stopLoss": sl_str,
                "takeProfit": tp_str,
                # Simulate filled status
                "leverage": str(self.leverage),
                "orderStatus": "Filled",
                # Add estimated value
                "positionValue": f"{qty * sim_entry_price:.4f}",
            }
            log_console(
                logging.INFO,
                f"[DRY RUN] Simulated order {order_id}, Sim Entry: {sim_entry_price:.{self.price_precision}f}",
                symbol=self.symbol,
            )
            return order_id

        # --- Real Order Placement (V5) ---
        try:
            order_params: Dict[str, Any] = {
                "category": self.category,
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_str,
            }
            # Add SL/TP parameters if provided and valid
            if sl_str:
                order_params["stopLoss"] = sl_str
                order_params["slTriggerBy"] = self.strategy_cfg.get("SL_TRIGGER_BY", "LastPrice")
            if tp_str:
                order_params["takeProfit"] = tp_str
                order_params["tpTriggerBy"] = self.strategy_cfg.get("TP_TRIGGER_BY", "LastPrice")
            # Set TPSL mode if either SL or TP is set (default to Full position)
            # V5 requires tpslMode when SL/TP is set for market orders
            if sl_str or tp_str:
                order_params["tpslMode"] = self.strategy_cfg.get("TPSL_MODE", "Full")  # "Full" or "Partial"

            # Generate unique order link ID if not provided
            order_params["orderLinkId"] = order_link_id or f"momscan_{self.symbol}_{int(time.time() * 1000)}"

            response = self.session.place_order(**order_params)

            if response and response.get("retCode") == 0:
                order_id = response.get("result", {}).get("orderId")
                if order_id:
                    log_console(
                        logging.INFO,
                        f"Order {order_id} submitted. Confirming execution...",
                        symbol=self.symbol,
                    )
                    # Confirm the order execution and get actual filled qty/price
                    filled_qty, avg_price = self.confirm_order(order_id, qty)

                    # Check if the fill was substantial (e.g., >= 95% of requested)
                    min_fill_ratio = 0.95
                    fill_tolerance = self.qty_step_float * 0.5  # Tolerance based on step size
                    required_fill = max(qty * min_fill_ratio, qty - fill_tolerance)

                    if filled_qty >= required_fill - FLOAT_COMPARISON_TOLERANCE:
                        log_console(
                            logging.INFO,
                            f"Order {order_id} confirmed filled. Qty={filled_qty:.{self.qty_precision}f}, AvgPrice={avg_price or 'N/A'}",
                            symbol=self.symbol,
                        )
                        # Update internal state with actual filled data
                        self.current_trade = {
                            "orderId": order_id,
                            "symbol": self.symbol,
                            "side": side,
                            "size": filled_qty,  # Use actual filled quantity
                            "avgPrice": (avg_price if avg_price else 0.0),  # Use actual average price
                            # Record entry time
                            "entry_time": dt.datetime.now(dt.timezone.utc),
                            "stopLoss": sl_str,
                            "takeProfit": tp_str,  # Store intended SL/TP
                            "leverage": str(self.leverage),
                            "orderStatus": "Filled",
                        }
                        return order_id
                    else:
                        log_console(
                            logging.ERROR,
                            f"Order {order_id} filled qty ({filled_qty:.{self.qty_precision}f}) < required ({required_fill:.{self.qty_precision}f}). Fill failed/incomplete.",
                            symbol=self.symbol,
                        )
                        self.current_trade = None  # Clear internal state if fill failed
                        # Consider attempting to cancel the potentially partially filled order? Risky. Manual check advised.
                        # log_console(logging.WARNING, f"Attempting to cancel potentially partially filled order {order_id}...", symbol=symbol)
                        # try:
                        #     cancel_resp = self.session.cancel_order(category=self.category, symbol=self.symbol, orderId=order_id)
                        #     log_console(logging.INFO, f"Cancel response for {order_id}: {cancel_resp}", symbol=symbol)
                        # except Exception as cancel_e:
                        #     log_console(logging.ERROR, f"Error cancelling order {order_id}: {cancel_e}", symbol=symbol)
                        return None
                else:
                    log_console(
                        logging.ERROR,
                        f"Order placement succeeded (retCode 0) but no Order ID returned. Response: {response}",
                        symbol=self.symbol,
                    )
                    return None
            else:  # API call to place order failed
                error_msg = response.get("retMsg", "N/A") if response else "Empty response"
                error_code = response.get("retCode", "N/A")
                log_console(
                    logging.ERROR,
                    f"Order placement failed: {error_msg} (Code: {error_code})",
                    symbol=self.symbol,
                )
                # Provide context for common V5 error codes
                if error_code == 110007:
                    log_console(
                        logging.CRITICAL,
                        "ORDER REJECTED: INSUFFICIENT BALANCE!",
                        symbol=self.symbol,
                    )
                elif error_code in [110013, 110014]:
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: Quantity issue (min/max/step). Check instrument info.",
                        symbol=self.symbol,
                    )
                elif error_code == 110017:
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: Risk control triggered by exchange (e.g., price deviation).",
                        symbol=self.symbol,
                    )
                elif error_code == 110040:
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: Risk limit exceeded (e.g., max position size).",
                        symbol=self.symbol,
                    )
                elif error_code == 110073:
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: TP/SL price invalid or too close to market.",
                        symbol=self.symbol,
                    )
                elif error_code == 110035:
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: SL/TP requires tpslMode parameter for market orders.",
                        symbol=self.symbol,
                    )
                elif "reduce-only" in error_msg.lower():
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: Reduce-only conflict? Check position.",
                        symbol=self.symbol,
                    )
                return None
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Exception during order placement for {self.symbol}: {e}",
                symbol=self.symbol,
                exc_info=True,
            )
            return None

    def close_position(
        self,
        position_data: Dict[str, Any],
        exit_reason: str = "Signal",
        exit_price_est: Optional[float] = None,
    ) -> bool:
        """Closes position using market order with reduceOnly=True (V5 API), logs metrics."""
        if not isinstance(position_data, dict):
            log_console(
                logging.WARNING,
                "Close attempt failed: No valid position data provided.",
                symbol=self.symbol,
            )
            return False

        # Extract necessary info safely
        side = position_data.get("side")
        size = position_data.get("size")  # Should be float from get_position
        entry_price = position_data.get("avgPrice")  # Should be float
        entry_time = position_data.get("entry_time")  # datetime or None
        leverage = position_data.get("leverage", str(self.leverage))  # str
        symbol_from_pos = position_data.get("symbol")  # Verify symbol matches

        # Validate essential data
        if symbol_from_pos != self.symbol:
            log_console(
                logging.ERROR,
                f"Position data symbol mismatch ({symbol_from_pos} != {self.symbol}). Cannot close.",
                symbol=self.symbol,
            )
            return False
        if not (
            side in ["Buy", "Sell"]
            and isinstance(size, float)
            and abs(size) > FLOAT_COMPARISON_TOLERANCE
            and isinstance(entry_price, float)
            and entry_price > FLOAT_COMPARISON_TOLERANCE
        ):
            log_console(
                logging.ERROR,
                f"Invalid position data for closing: Side={side}, Size={size}, Entry={entry_price}. Cannot close.",
                symbol=self.symbol,
            )
            # Clear potentially inconsistent internal state if it matches the symbol
            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                log_console(
                    logging.WARNING,
                    "Clearing inconsistent internal trade state.",
                    symbol=self.symbol,
                )
                self.current_trade = None
            return False

        abs_size = abs(size)
        # Check if position size is negligible (e.g., dust) - use a threshold relative to min qty or step
        min_significant_qty = max(
            FLOAT_COMPARISON_TOLERANCE,
            self.min_order_qty * 0.01,
            self.qty_step_float * 0.1,
        )
        if abs_size < min_significant_qty:
            log_console(
                logging.INFO,
                f"Position size {abs_size:.{self.qty_precision}f} negligible. Assuming already closed.",
                symbol=self.symbol,
            )
            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                self.current_trade = None
            self.last_exit_time = dt.datetime.now(dt.timezone.utc)
            return True

        close_side = "Sell" if side == "Buy" else "Buy"
        qty_str = f"{abs_size:.{self.qty_precision}f}"

        log_prefix = f"{Fore.YELLOW + Style.BRIGHT}[DRY RUN]{Style.RESET_ALL} " if self.dry_run else ""
        side_color = Fore.RED if close_side == "Sell" else Fore.GREEN
        log_msg = (
            f"{log_prefix}{side_color}{Style.BRIGHT}CLOSE {side.upper()} POSITION ({close_side} MARKET):"
            f"{Style.RESET_ALL} Qty={qty_str} | Reason: {exit_reason}"
        )
        action_prefix = f"{Fore.MAGENTA + Style.BRIGHT}ACTION:{Style.RESET_ALL} [{self.symbol}] "
        print(action_prefix + log_msg)
        log_console(logging.INFO, log_msg.replace(log_prefix, "").strip(), symbol=self.symbol)

        if self.dry_run:
            # Check if there's a simulated trade to close
            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                sim_exit_price = exit_price_est
                # Estimate exit price if not provided
                if sim_exit_price is None or sim_exit_price <= FLOAT_COMPARISON_TOLERANCE:
                    if self.kline_df is not None and not self.kline_df.empty:
                        try:
                            sim_exit_price = float(self.kline_df["close"].iloc[-1])
                        except (IndexError, ValueError, TypeError):
                            pass
                # Fallback to entry price if still no valid exit price
                if sim_exit_price is None or sim_exit_price <= FLOAT_COMPARISON_TOLERANCE:
                    sim_entry_price_fallback = self.current_trade.get("avgPrice", 1.0)
                    sim_exit_price = (
                        sim_entry_price_fallback if sim_entry_price_fallback > FLOAT_COMPARISON_TOLERANCE else 1.0
                    )
                    log_console(
                        logging.WARNING,
                        f"[DRY RUN] Cannot estimate exit price for {self.symbol}, using entry/fallback ({sim_exit_price}). P&L inaccurate.",
                        symbol=self.symbol,
                    )

                log_console(
                    logging.INFO,
                    f"[DRY RUN] Simulating close for {self.symbol} at {sim_exit_price:.{self.price_precision}f}",
                    symbol=self.symbol,
                )
                # Get details from the simulated trade for metrics
                entry_time_sim = self.current_trade.get("entry_time") or (
                    dt.datetime.now(
                        # Fallback entry time
                        dt.timezone.utc
                    )
                    - dt.timedelta(minutes=15)
                )
                sim_leverage = self.current_trade.get("leverage", str(self.leverage))
                sim_entry_price = self.current_trade.get("avgPrice", sim_exit_price)  # Use stored entry price
                sim_qty = self.current_trade.get("size", abs_size)  # Use stored size
                original_side = self.current_trade.get("side", side)  # Use stored side

                # Log the simulated trade metrics
                self.metrics.add_trade(
                    symbol=self.symbol,
                    entry_time=entry_time_sim,
                    exit_time=dt.datetime.now(dt.timezone.utc),
                    side=original_side,
                    entry_price=sim_entry_price,
                    exit_price=sim_exit_price,
                    qty=abs(sim_qty),
                    leverage=sim_leverage,
                )
                # Clear the simulated trade state
                self.current_trade = None
                self.last_exit_time = dt.datetime.now(dt.timezone.utc)
                return True
            else:
                log_console(
                    logging.WARNING,
                    f"[DRY RUN] Close attempt failed: No simulated trade open for {self.symbol}.",
                    symbol=self.symbol,
                )
                self.current_trade = None  # Ensure state is clear
                return False

        # --- Real Position Close (V5) ---
        try:
            close_order_link_id = f"close_{self.symbol}_{int(time.time() * 1000)}"
            response = self.session.place_order(
                category=self.category,
                symbol=self.symbol,
                side=close_side,
                orderType="Market",
                qty=qty_str,
                reduceOnly=True,  # Crucial for closing positions safely
                orderLinkId=close_order_link_id,
            )

            if response and response.get("retCode") == 0:
                order_id = response.get("result", {}).get("orderId")
                if order_id:
                    log_console(
                        logging.INFO,
                        f"Close order {order_id} submitted. Confirming...",
                        symbol=self.symbol,
                    )
                    # Confirm the close order execution
                    filled_qty, avg_exit_price = self.confirm_order(order_id, abs_size)

                    # Check if the close order filled sufficiently (e.g., >= 99% of position size)
                    min_close_fill_ratio = 0.99
                    fill_tolerance = self.qty_step_float * 0.5
                    required_fill = max(abs_size * min_close_fill_ratio, abs_size - fill_tolerance)

                    if filled_qty >= required_fill - FLOAT_COMPARISON_TOLERANCE:
                        log_console(
                            logging.INFO,
                            f"Close order {order_id} confirmed. Filled: {filled_qty:.{self.qty_precision}f}, AvgExit: {avg_exit_price or 'N/A'}",
                            symbol=self.symbol,
                        )

                        # Optional: Query position again to double-check closure (can add delay)
                        # time.sleep(1) # Allow time for position update
                        # final_pos_check = self.get_position(log_error=False)
                        # if final_pos_check is not None: ... log warning if size > dust ...

                        # Log Metrics
                        exit_time = dt.datetime.now(dt.timezone.utc)
                        self.last_exit_time = exit_time
                        # Use confirmed average exit price if available and valid, else fallback
                        final_exit_price_log = (
                            avg_exit_price
                            if avg_exit_price is not None and avg_exit_price > FLOAT_COMPARISON_TOLERANCE
                            else exit_price_est
                        )
                        # If still no valid price, fallback to entry price (P&L will be inaccurate)
                        if final_exit_price_log is None or final_exit_price_log <= FLOAT_COMPARISON_TOLERANCE:
                            log_console(
                                logging.WARNING,
                                f"Cannot determine valid exit price for metrics for {self.symbol}. Using entry {entry_price}.",
                                symbol=self.symbol,
                            )
                            final_exit_price_log = entry_price

                        # Ensure entry time is valid datetime object
                        entry_time_log = entry_time
                        if not isinstance(entry_time_log, dt.datetime):
                            # Try getting from internal state if API didn't provide it
                            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                                entry_time_log = self.current_trade.get("entry_time")
                            # Final fallback if still missing
                            if not isinstance(entry_time_log, dt.datetime):
                                log_console(
                                    logging.WARNING,
                                    f"Entry time missing for metrics for {self.symbol}. Using fallback time.",
                                    symbol=self.symbol,
                                )
                                entry_time_log = exit_time - dt.timedelta(minutes=15)  # Arbitrary fallback

                        self.metrics.add_trade(
                            symbol=self.symbol,
                            entry_time=entry_time_log,
                            exit_time=exit_time,
                            side=side,
                            entry_price=entry_price,
                            exit_price=final_exit_price_log,
                            qty=filled_qty,  # Use actual filled qty for metrics
                            leverage=leverage,
                        )
                        self.current_trade = None  # Clear internal state after successful close
                        return True
                    else:
                        log_console(
                            logging.ERROR,
                            f"Close order {order_id} filled qty ({filled_qty:.{self.qty_precision}f}) < expected ({required_fill:.{self.qty_precision}f}). Manual check required!",
                            symbol=self.symbol,
                        )
                        # Do not clear internal state here, as position might still be partially open
                        # Consider querying position again to update internal state?
                        return False
                else:
                    log_console(
                        logging.ERROR,
                        f"Close order succeeded (retCode 0) but no Order ID returned. {response}",
                        symbol=self.symbol,
                    )
                    return False
            else:  # API call to place close order failed
                error_msg = response.get("retMsg", "N/A") if response else "Empty response"
                error_code = response.get("retCode", "N/A")
                # Check for specific V5 errors indicating position already closed or size mismatch
                # 110025: Position idx not match position side
                # 110066: The current position size is zero
                # 3400074: Order quantity exceeded position size * leverage
                # 110044: Reduce-only order failed as it would increase position
                pos_already_closed_codes = [110025, 110066, 3400074, 110044]
                already_closed_msg_fragments = [
                    "position size is zero",
                    "order qty exceeded position size",
                    "reduceonly",
                    "reduce-only",
                    "position idx not match",
                ]
                already_closed = error_code in pos_already_closed_codes or any(
                    frag in error_msg.lower() for frag in already_closed_msg_fragments
                )

                if already_closed:
                    log_console(
                        logging.WARNING,
                        f"Close attempt failed for {self.symbol}, API indicates already closed or size issue (Code: {error_code}, Msg: '{error_msg}'). Assuming closed.",
                        symbol=self.symbol,
                    )
                    # If we have internal state, log the trade based on that state as it likely closed externally (e.g., SL/TP hit)
                    if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                        exit_time = dt.datetime.now(dt.timezone.utc)
                        # Use estimated exit price or entry price as fallback
                        final_exit_price = (
                            exit_price_est
                            if exit_price_est is not None and exit_price_est > FLOAT_COMPARISON_TOLERANCE
                            else entry_price
                        )
                        entry_time_actual = self.current_trade.get("entry_time") or (
                            exit_time - dt.timedelta(minutes=15)
                        )
                        internal_pos_size = self.current_trade.get("size", abs_size)
                        internal_entry_price = self.current_trade.get("avgPrice", entry_price)
                        internal_side = self.current_trade.get("side", side)
                        internal_leverage = self.current_trade.get("leverage", leverage)
                        log_console(
                            logging.INFO,
                            f"Logging trade metrics from internal state for {self.symbol} (API showed already closed/size issue).",
                            symbol=self.symbol,
                        )
                        self.metrics.add_trade(
                            symbol=self.symbol,
                            entry_time=entry_time_actual,
                            exit_time=exit_time,
                            side=internal_side,
                            entry_price=internal_entry_price,
                            exit_price=final_exit_price,
                            qty=abs(internal_pos_size),
                            leverage=internal_leverage,
                        )
                        self.current_trade = None  # Clear state as it's confirmed closed
                    self.last_exit_time = dt.datetime.now(dt.timezone.utc)  # Set cooldown timer
                    return True  # Treat as successful closure
                else:
                    log_console(
                        logging.ERROR,
                        f"Position close order failed for {self.symbol}: {error_msg} (Code: {error_code})",
                        symbol=self.symbol,
                    )
                    return False
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Exception during position close for {self.symbol}: {e}",
                symbol=self.symbol,
                exc_info=True,
            )
            return False

    def get_higher_tf_trend(self) -> bool:
        """Checks higher timeframe trend (cached). Returns True if favorable or disabled/failed."""
        if not self.strategy_cfg.get("ENABLE_MULTI_TIMEFRAME", False):
            return True  # Allow if disabled

        cache_ttl = self.bot_cfg.get("HIGHER_TF_CACHE_SECONDS", 3600)
        now_utc = dt.datetime.now(dt.timezone.utc)
        # Check cache validity
        if (
            self.higher_tf_cache is not None
            and self.higher_tf_cache_time
            and (now_utc - self.higher_tf_cache_time).total_seconds() < cache_ttl
        ):
            log_console(
                logging.DEBUG,
                f"Using cached HTF trend: {'Favorable/Allow' if self.higher_tf_cache else 'Unfavorable'}",
                symbol=self.symbol,
            )
            return self.higher_tf_cache

        higher_tf = str(self.strategy_cfg.get("HIGHER_TIMEFRAME", "60"))
        if higher_tf == self.timeframe:
            log_console(
                logging.WARNING,
                "HTF same as base timeframe. Disabling MTF check for this symbol.",
                symbol=self.symbol,
            )
            # Cache as True (allow) and update time
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True

        # Fetch slightly more data for HTF to ensure indicator calculation is robust
        # Use the calculated KLINE_LIMIT for the *base* timeframe as a guide, maybe add buffer
        htf_kline_limit = self.internal_cfg.get("KLINE_LIMIT", 120) + 50
        log_console(
            logging.INFO,
            f"Fetching HTF ({higher_tf}) data for trend analysis...",
            symbol=self.symbol,
        )
        # Use the REST fetcher which also calculates indicators
        # Important: Pass the strategy config, assuming HTF uses the same indicator settings
        df_higher = self.get_ohlcv_rest(timeframe=higher_tf, limit=htf_kline_limit)

        # Check if fetch and indicator calculation succeeded and we have enough data
        # Need at least 2 rows for iloc[-2]
        if df_higher is None or df_higher.empty or len(df_higher) < MIN_KLINE_RECORDS_FOR_CALC + 1:
            log_console(
                logging.WARNING,
                f"Could not get sufficient HTF ({higher_tf}) data or indicators failed. Allowing trade (neutral).",
                symbol=self.symbol,
            )
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True  # Fail open (allow trade if HTF check fails)

        try:
            # Determine which MA columns to use based on Ehlers flag (consistent with base TF)
            use_ehlers_htf = self.strategy_cfg.get("USE_EHLERS_SMOOTHER", False)
            smoother_prefix_htf = "ss" if use_ehlers_htf else "ema"
            fast_ma_col = f"{smoother_prefix_htf}_fast"
            mid_ma_col = f"{smoother_prefix_htf}_mid"

            # Check if the required MA columns exist after calculation
            if fast_ma_col not in df_higher.columns or mid_ma_col not in df_higher.columns:
                log_console(
                    logging.ERROR,
                    f"Required MA cols ({fast_ma_col}, {mid_ma_col}) not found in HTF ({higher_tf}) data after calculation. Allowing trade.",
                    symbol=self.symbol,
                )
                self.higher_tf_cache = True
                self.higher_tf_cache_time = now_utc
                return True

            # Check trend on the second-to-last fully closed candle (-2 index) for stability
            candle_to_check = df_higher.iloc[-2]
            fast_ma_prev = candle_to_check.get(fast_ma_col)
            mid_ma_prev = candle_to_check.get(mid_ma_col)

            # Check for NaN values after .get() - should be handled by indicator calc fill, but safeguard
            if pd.isna(fast_ma_prev) or pd.isna(mid_ma_prev):
                log_console(
                    logging.WARNING,
                    f"HTF ({higher_tf}) MAs are NaN on check candle ({candle_to_check.name}). Allowing trade.",
                    symbol=self.symbol,
                )
                trend_is_favorable = True
            else:
                # Define favorable trend logic (CUSTOMIZABLE)
                # Example: Allow trade only if HTF fast MA > mid MA (uptrend)
                # More complex: Consider slope, ADX, etc. on HTF
                trend_is_favorable = fast_ma_prev > mid_ma_prev
                trend_desc = "Bullish (Fast>Mid)" if trend_is_favorable else "Bearish/Neutral (Fast<=Mid)"
                log_console(
                    logging.INFO,
                    f"HTF ({higher_tf}) trend (@ {candle_to_check.name}): "
                    f"{fast_ma_col}={fast_ma_prev:.{self.price_precision}f}, {mid_ma_col}={mid_ma_prev:.{self.price_precision}f}. "
                    f"Status: {trend_desc} -> Allow Trade: {trend_is_favorable}",
                    symbol=self.symbol,
                )

            # Update cache
            self.higher_tf_cache = trend_is_favorable
            self.higher_tf_cache_time = now_utc
            return trend_is_favorable

        except IndexError:
            log_console(
                logging.WARNING,
                f"Index error accessing HTF ({higher_tf}) data (need >= {MIN_KLINE_RECORDS_FOR_CALC + 1} rows). Allowing trade.",
                symbol=self.symbol,
            )
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Error during HTF ({higher_tf}) trend analysis: {e}. Allowing trade.",
                symbol=self.symbol,
                exc_info=True,
            )
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True


# --- Main Trading Bot Orchestrator ---
class MomentumScannerTrader:
    """Orchestrates trading across multiple symbols using Bybit V5 API."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        config: Dict[str, Any],
        dry_run: bool = False,
    ):
        """Initializes the main bot."""
        self.config = config
        self.dry_run = dry_run
        self.bybit_cfg: Dict[str, Any] = config.get("BYBIT_CONFIG", {})
        self.risk_cfg: Dict[str, Any] = config.get("RISK_CONFIG", {})
        self.bot_cfg: Dict[str, Any] = config.get("BOT_CONFIG", {})
        self.running: bool = True
        self.shutdown_requested: bool = False

        # API Setup
        self.use_testnet: bool = self.bybit_cfg.get("USE_TESTNET", False)
        self.account_type: str = self.bybit_cfg.get("ACCOUNT_TYPE", "UNIFIED").upper()
        # Determine V5 category based on account type (adjust if using inverse etc.)
        # Common mapping: UNIFIED/CONTRACT -> linear/inverse, SPOT -> spot
        # Assuming linear for UNIFIED/CONTRACT unless specified otherwise (e.g., by symbol)
        if self.account_type == "SPOT":
            self.category = "spot"
        elif self.account_type in ["UNIFIED", "CONTRACT"]:
            # Default to linear, but could be overridden per symbol if needed
            self.category = "linear"
            log_console(
                logging.INFO,
                f"Using category '{self.category}' for {self.account_type} account. Ensure symbols match (e.g., USDT pairs).",
            )
        else:
            log_console(
                logging.WARNING,
                f"Account type '{self.account_type}' might not map directly to V5 category. Defaulting to 'linear'.",
            )
            self.category = "linear"

        self.api_key = api_key
        self.api_secret = api_secret
        self.session: Optional[HTTP] = None
        self.ws: Optional[WebSocket] = None
        self.ws_active: bool = False
        self.ws_connection_thread: Optional[threading.Thread] = None
        self.ws_lock = threading.Lock()  # Lock for accessing self.ws and self.ws_active

        # Traders and Metrics
        self.traders: Dict[str, SymbolTrader] = {}
        fee_rate = self.risk_cfg.get("FEE_RATE", 0.0006)  # Default 0.06% taker fee
        self.metrics = TradeMetrics(fee_rate=fee_rate)

        log_console(
            logging.INFO,
            f"Initializing Trader (Testnet: {self.use_testnet}, Account: {self.account_type}, Category: {self.category})...",
        )
        try:
            # Configure HTTP session with receive window from config
            recv_window = self.bot_cfg.get("API_RECV_WINDOW", 10000)  # Default 10s
            self.session = HTTP(
                testnet=self.use_testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=recv_window,
                # Optional: Add logging=True to pybit HTTP for detailed request/response logging
                # logging_level=logging.DEBUG if args.debug else logging.INFO
            )
            self._test_api_connection()  # Test REST API connection early
            self._initialize_traders()  # Initialize traders for configured symbols

            # Start WebSocket if needed and traders initialized successfully
            if self.traders and any(trader.use_websocket for trader in self.traders.values()):
                log_console(logging.INFO, "Starting WebSocket connection thread...")
                self.ws_connection_thread = threading.Thread(
                    target=self._start_and_monitor_websocket,
                    name="WebSocketMonitor",
                    daemon=True,
                )
                self.ws_connection_thread.start()
                # Allow some time for the connection attempt and initial subscriptions
                time.sleep(self.bot_cfg.get("WS_STARTUP_DELAY_SECONDS", 5.0))
            else:
                log_console(
                    logging.INFO,
                    "WebSocket disabled or no traders configured for WS. Using REST API only.",
                )

        except (ConnectionError, RuntimeError, ValueError) as init_e:
            log_console(logging.CRITICAL, f"Bot initialization failed: {init_e}", exc_info=False)
            self.running = False
            self.shutdown_requested = True
            self._cleanup_resources()  # Attempt cleanup even on init failure
            raise init_e  # Re-raise to stop execution
        except Exception as e:
            log_console(
                logging.CRITICAL,
                f"Unexpected error during bot initialization: {e}",
                exc_info=True,
            )
            self.running = False
            self.shutdown_requested = True
            self._cleanup_resources()
            raise RuntimeError("Unexpected initialization failure") from e

    def _test_api_connection(self):
        """Tests API connection by fetching server time (V5)."""
        if not self.session:
            raise ConnectionError("HTTP session not initialized.")
        try:
            log_console(logging.DEBUG, "Testing API connection...")
            server_time_resp = self.session.get_server_time()
            if server_time_resp and server_time_resp.get("retCode") == 0:
                time_nano_str = server_time_resp.get("result", {}).get("timeNano", "0")
                # Convert nanoseconds string to seconds float
                server_ts = int(time_nano_str) / 1e9
                server_dt = dt.datetime.fromtimestamp(server_ts, tz=dt.timezone.utc)
                local_dt_utc = dt.datetime.now(dt.timezone.utc)
                # Check clock skew
                time_diff = abs((local_dt_utc - server_dt).total_seconds())
                max_allowable_skew = self.bot_cfg.get("MAX_CLOCK_SKEW_SECONDS", 60)  # Allow config override
                if time_diff > max_allowable_skew:
                    log_console(
                        logging.WARNING,
                        f"System clock differs from server by {time_diff:.1f}s (>{max_allowable_skew}s). Check time synchronization (e.g., NTP).",
                    )
                log_console(
                    logging.INFO,
                    f"API connection successful. Server Time: {server_dt.isoformat()}",
                )
            else:
                error_msg = server_time_resp.get("retMsg", "N/A") if server_time_resp else "Empty Response"
                error_code = server_time_resp.get("retCode", "N/A")
                # Check for auth errors (e.g., 10003, 10004)
                if error_code in [10003, 10004]:
                    log_console(
                        logging.CRITICAL,
                        f"API Authentication Failed: {error_msg} (Code: {error_code}). Check API Key/Secret/Permissions.",
                    )
                raise ConnectionError(f"API test failed: {error_msg} (Code: {error_code})")
        except ConnectionError as ce:
            log_console(
                logging.CRITICAL,
                f"API connection failed: {ce}. Check keys, permissions, network, and time sync.",
            )
            raise ce
        except Exception as e:
            log_console(logging.CRITICAL, f"API test failed unexpectedly: {e}.", exc_info=True)
            raise ConnectionError("Failed API connection test.") from e

    def _initialize_traders(self):
        """Creates and initializes SymbolTrader instances."""
        symbols_cfg_list: List[Dict[str, Any]] = self.bybit_cfg.get("SYMBOLS", [])
        if not isinstance(symbols_cfg_list, list) or not symbols_cfg_list:
            raise ValueError("BYBIT_CONFIG.SYMBOLS is missing, empty, or not a list.")

        log_console(
            logging.INFO,
            f"Initializing traders for {len(symbols_cfg_list)} configured symbol(s)...",
        )
        successful_traders = 0
        temp_traders: Dict[str, SymbolTrader] = {}

        # Use deepcopy to avoid modifying the original config dict during iteration
        for i, symbol_cfg in enumerate(deepcopy(symbols_cfg_list)):
            if not isinstance(symbol_cfg, dict):
                log_console(
                    logging.ERROR,
                    f"Symbol config #{i} invalid (not a dictionary). Skipping.",
                )
                continue
            symbol = symbol_cfg.get("SYMBOL")
            if not symbol or not isinstance(symbol, str):
                log_console(
                    logging.ERROR,
                    f"Symbol config #{i} missing/invalid 'SYMBOL' key. Skipping.",
                )
                continue
            if symbol in temp_traders:
                log_console(
                    logging.WARNING,
                    f"Duplicate symbol '{symbol}' in config. Skipping subsequent entry.",
                    symbol=symbol,
                )
                continue

            log_console(logging.INFO, f"--- Initializing trader for {symbol} ---", symbol=symbol)
            try:
                if not self.session:
                    raise RuntimeError("HTTP session not available for trader initialization.")
                # Pass the shared HTTP session and determined category
                trader = SymbolTrader(
                    self.api_key,
                    self.api_secret,
                    self.config,
                    symbol_cfg,
                    self.dry_run,
                    self.metrics,
                    self.session,
                    self.category,
                )
                # Check if initialization inside SymbolTrader was successful
                if trader.initialization_successful:
                    temp_traders[symbol] = trader
                    successful_traders += 1
                    log_console(
                        logging.INFO,
                        f"--- Trader {symbol} initialized successfully ---",
                        symbol=symbol,
                    )
                else:
                    # Should have been raised as RuntimeError from SymbolTrader, but catch here just in case
                    log_console(
                        logging.ERROR,
                        f"Trader initialization reported failure for {symbol}. Skipping.",
                        symbol=symbol,
                    )

            except RuntimeError as trader_init_e:
                # Catch specific RuntimeError raised by SymbolTrader on failure
                log_console(
                    logging.ERROR,
                    f"Initialization failed for {symbol}: {trader_init_e}. Skipping.",
                    symbol=symbol,
                )
            except Exception as e:
                log_console(
                    logging.ERROR,
                    f"Unexpected error initializing {symbol}: {e}. Skipping.",
                    symbol=symbol,
                    exc_info=True,
                )

        if successful_traders == 0:
            raise RuntimeError("CRITICAL: No traders could be initialized successfully. Check config and logs.")

        self.traders = temp_traders  # Assign successfully initialized traders
        log_console(
            logging.INFO,
            f"Initialized {successful_traders}/{len(symbols_cfg_list)} traders: {', '.join(self.traders.keys())}",
        )

    def _start_and_monitor_websocket(self):
        """Manages WebSocket connection, subscriptions, and reconnections."""
        log_console(logging.INFO, "WebSocket monitoring thread started.")
        # V5 channel type matches category (linear, inverse, spot, option)
        ws_channel_type = self.category.lower()
        if ws_channel_type not in ["linear", "inverse", "spot", "option"]:
            log_console(
                logging.ERROR,
                f"Unsupported category '{self.category}' for WS channel_type. Defaulting to 'linear'.",
            )
            ws_channel_type = "linear"

        retry_delay = self.bot_cfg.get("WS_RECONNECT_DELAY_INIT_SECONDS", 10)
        max_retry_delay = self.bot_cfg.get("WS_RECONNECT_DELAY_MAX_SECONDS", 120)

        while not self.shutdown_requested:
            ws_conn_active = False
            with self.ws_lock:
                ws_conn_active = self.ws is not None and self.ws_active

            if not ws_conn_active:
                log_console(
                    logging.INFO,
                    f"Attempting WS connection (Channel: {ws_channel_type})...",
                )
                try:
                    # Create a new WebSocket instance for each connection attempt
                    temp_ws = WebSocket(
                        testnet=self.use_testnet,
                        channel_type=ws_channel_type,
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        ping_interval=self.bot_cfg.get("WS_PING_INTERVAL", 20),
                        ping_timeout=self.bot_cfg.get("WS_PING_TIMEOUT", 10),
                        on_message=self._handle_ws_message,  # Generic handler for auth/sub responses
                        on_error=self._handle_ws_error,  # Handles connection errors
                        on_close=self._handle_ws_close,  # Handles unexpected closes
                        on_open=self._handle_ws_open,  # Sets ws_active flag
                        retries=0,  # Disable pybit's internal retries, manage manually
                        # Optional: Add trace logging for websocket-client
                        # trace=args.debug
                    )

                    # Prepare subscriptions based on initialized traders
                    subscriptions = []
                    callbacks = {}
                    for symbol, trader in self.traders.items():
                        if trader.use_websocket:
                            # V5 kline topic format
                            topic = f"kline.{trader.timeframe}.{symbol}"
                            subscriptions.append(topic)
                            # Store callback associated with the topic
                            callbacks[topic] = trader._websocket_callback
                            log_console(
                                logging.DEBUG,
                                f"WS prepared subscription: {topic}",
                                symbol=symbol,
                            )

                    if subscriptions:
                        # Assign the WS object just before starting interaction
                        with self.ws_lock:
                            self.ws = temp_ws

                        # Start the WebSocket connection implicitly by subscribing
                        # The on_open callback will set self.ws_active = True
                        # Subscribe to all topics at once
                        log_console(
                            logging.INFO,
                            f"WebSocket subscribing to {len(subscriptions)} streams...",
                        )
                        # Use dispatcher
                        self.ws.subscribe(subscriptions, callback=self._dispatch_ws_callback)

                        # Wait for connection to establish or fail (check ws_active flag)
                        wait_time = 0
                        max_wait = self.bot_cfg.get("WS_CONNECT_TIMEOUT_SECONDS", 30)
                        while wait_time < max_wait:
                            with self.ws_lock:
                                ws_established = self.ws_active
                            if ws_established or self.shutdown_requested:
                                break
                            time.sleep(1)
                            wait_time += 1

                        if self.shutdown_requested:
                            break  # Exit loop if shutdown requested

                        with self.ws_lock:
                            ws_established = self.ws_active
                        if ws_established:
                            log_console(
                                logging.INFO,
                                "WebSocket connection established. Monitor active.",
                            )
                            retry_delay = self.bot_cfg.get("WS_RECONNECT_DELAY_INIT_SECONDS", 10)  # Reset retry delay
                            # Keep monitor thread alive, checking active flag periodically
                            while not self.shutdown_requested:
                                with self.ws_lock:
                                    ws_still_active = self.ws_active
                                if not ws_still_active:
                                    log_console(
                                        logging.INFO,
                                        "WS inactive detected by monitor, triggering reconnect.",
                                    )
                                    break  # Break inner loop to reconnect
                                # Check interval
                                time.sleep(self.bot_cfg.get("WS_MONITOR_CHECK_INTERVAL_SECONDS", 5))
                        else:
                            log_console(
                                logging.ERROR,
                                f"WS connection failed to establish within {max_wait}s. Retrying...",
                            )
                            self._close_websocket_connection()  # Ensure cleanup before retry
                            if not self.shutdown_requested:
                                time.sleep(retry_delay)
                            # Exponential backoff (less aggressive)
                            retry_delay = min(max_retry_delay, retry_delay * 1.5)

                    else:  # No symbols configured for WebSocket
                        log_console(logging.INFO, "No symbols use WebSocket. WS thread exiting.")
                        break  # Exit monitoring loop

                except Exception as e:
                    log_console(
                        logging.ERROR,
                        f"Failed to init/start WS connection: {e}. Retrying in {retry_delay:.1f}s.",
                        exc_info=False,
                    )
                    self._close_websocket_connection()  # Ensure cleanup
                    if not self.shutdown_requested:
                        time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay = min(max_retry_delay, retry_delay * 1.5)

            else:  # WS is active, sleep and check later
                # Check less frequently when stable
                time.sleep(self.bot_cfg.get("WS_MONITOR_CHECK_INTERVAL_SECONDS", 10))

        log_console(logging.INFO, "WebSocket monitoring thread finished.")
        self._close_websocket_connection()  # Ensure cleanup on exit

    def _dispatch_ws_callback(self, message: Dict[str, Any]):
        """Dispatches incoming WebSocket messages to the correct SymbolTrader callback."""
        try:
            topic = message.get("topic")
            if topic:
                # Extract symbol and timeframe from topic: kline.{interval}.{symbol}
                parts = topic.split(".")
                if len(parts) == 3 and parts[0] == "kline":
                    symbol = parts[2]
                    if symbol in self.traders:
                        # Call the specific trader's callback
                        self.traders[symbol]._websocket_callback(message)
                    # else: log silently or add debug log if needed for unexpected symbols
                # else: # Handle other topics if necessary (e.g., position updates)
                #     log_console(logging.DEBUG, f"WS Dispatch: Non-kline topic {topic}")
            # else: Pass to generic handler for non-topic messages (auth, subscribe responses)
            #     self._handle_ws_message(message) # Already handled by pybit's on_message
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Error dispatching WS message: {e}. Message: {message}",
                exc_info=True,
            )

    def _close_websocket_connection(self):
        """Safely closes the WebSocket connection object."""
        with self.ws_lock:
            ws_instance = self.ws  # Get current instance
            if ws_instance:
                log_console(logging.INFO, "Closing WebSocket connection object...")
                # Clear internal reference *before* calling exit to prevent race conditions
                self.ws = None
                self.ws_active = False
                try:
                    # Use the exit() method provided by pybit's WebSocket class
                    ws_instance.exit()
                    log_console(logging.DEBUG, "WebSocket exit() called.")
                except Exception as e:
                    log_console(logging.ERROR, f"Error closing WS object: {e}", exc_info=False)
            else:
                # Ensure flag is false if ws object was already None
                self.ws_active = False

    # --- WebSocket Event Handlers ---
    def _handle_ws_open(self, ws_app):
        """Callback executed when WebSocket connection is successfully opened."""
        log_console(logging.INFO, "WebSocket connection opened successfully.")
        with self.ws_lock:
            self.ws_active = True
        # Populate initial data asynchronously after connection established
        self._populate_initial_kline_data_async()

    def _populate_initial_kline_data_async(self):
        """Starts a background thread to fetch initial Kline data via REST post-WS connection."""
        log_console(logging.DEBUG, "Starting thread for initial REST kline population...")
        thread = threading.Thread(
            target=self._fetch_initial_data_worker,
            name="InitialDataFetcher",
            daemon=True,
        )
        thread.start()

    def _fetch_initial_data_worker(self):
        """Worker thread for fetching initial REST data for WS-enabled traders."""
        time.sleep(2)  # Allow subscriptions to potentially settle
        log_console(logging.INFO, "Initial data fetcher thread running...")
        # Iterate over a copy of trader items for thread safety
        traders_to_fetch = list(self.traders.items())
        for symbol, trader in traders_to_fetch:
            if self.shutdown_requested:
                break  # Check for shutdown signal
            # Fetch only if WS is enabled AND internal DF is still empty/None
            if trader.use_websocket and (trader.kline_df is None or trader.kline_df.empty):
                log_console(
                    logging.DEBUG,
                    f"Fetching initial REST data for {symbol}...",
                    symbol=symbol,
                )
                try:
                    # Use the trader's own method to fetch and calculate indicators
                    initial_df = trader.get_ohlcv_rest()
                    if initial_df is not None and not initial_df.empty:
                        # Update the trader's internal DataFrame directly
                        # Ensure index is UTC
                        if not initial_df.index.tz:
                            initial_df.index = initial_df.index.tz_localize("UTC")
                        trader.kline_df = initial_df
                        trader.last_kline_update_time = dt.datetime.now(dt.timezone.utc)  # Set update time
                        log_console(
                            logging.INFO,
                            f"Initialized kline data for {symbol} from REST.",
                            symbol=symbol,
                        )
                    else:
                        log_console(
                            logging.WARNING,
                            f"Failed initial REST fetch for {symbol}. Will rely solely on WS updates.",
                            symbol=symbol,
                        )
                except Exception as e:
                    log_console(
                        logging.ERROR,
                        f"Error fetching initial REST data for {symbol}: {e}",
                        symbol=symbol,
                        exc_info=False,
                    )
                # Throttle requests slightly between symbols
                time.sleep(self.bot_cfg.get("INITIAL_FETCH_DELAY_PER_SYMBOL_MS", 200) / 1000.0)
        log_console(logging.INFO, "Initial data fetcher thread finished.")

    def _handle_ws_message(self, message):
        """Generic WS message handler (e.g., auth, subscription responses)."""
        # This is called by pybit *in addition* to the specific callback set in subscribe()
        # We primarily use the dispatcher callback now, but keep this for auth/ping etc.
        try:
            # log_console(logging.DEBUG, f"Generic WS Msg: {message}") # Very verbose, enable if needed
            if isinstance(message, dict):
                op = message.get("op")
                # Handle auth response
                if op == "auth":
                    success = message.get("success", False)
                    log_console(
                        logging.INFO,
                        f"WebSocket Auth: {'Success' if success else 'Failed'}",
                    )
                    if not success:
                        log_console(
                            logging.ERROR,
                            f"WS Auth Fail: {message.get('ret_msg', 'N/A')}",
                        )
                # Handle subscription response
                elif op == "subscribe":
                    success = message.get("success", False)
                    # V5 uses 'args' in the request, response might echo these or use ret_msg
                    sub_args = message.get("args", [])
                    topic_info = ", ".join(sub_args) if sub_args else message.get("ret_msg", "")
                    if not topic_info and "req_id" in message:
                        topic_info = f"ReqID: {message['req_id']}"

                    if success:
                        log_console(logging.DEBUG, f"WS Subscribe Success: {topic_info}")
                    else:
                        log_console(
                            logging.ERROR,
                            f"WS Subscribe Fail: {topic_info} - {message.get('ret_msg', 'N/A')}",
                        )
                # Handle ping response (usually handled internally by pybit)
                elif op == "ping":
                    log_console(logging.DEBUG, "WS Received Ping")
                    # Optional: Send pong if auto-pong is disabled (pybit usually handles this)
                    # if self.ws: self.ws.send(json.dumps({"op": "pong"}))
                # else: log_console(logging.DEBUG, f"Unhandled Generic WS Op: {op}, Msg: {message}")

        except Exception as e:
            log_console(logging.ERROR, f"Error handling generic WS message: {e}", exc_info=True)

    def _handle_ws_error(self, ws_app_or_error, error=None):
        """Callback executed on WebSocket error."""
        # The signature can vary depending on the websocket-client version and context
        err_msg = error if error else ws_app_or_error
        is_exception = isinstance(err_msg, Exception)
        log_console(
            logging.ERROR,
            f"WebSocket Error Encountered: {err_msg}",
            exc_info=is_exception,
        )
        # Set active flag to False to trigger reconnection attempt by the monitor thread
        with self.ws_lock:
            self.ws_active = False  # Signal monitor to reconnect

    def _handle_ws_close(self, ws_app):
        """Callback executed when WebSocket connection is closed."""
        # Check if shutdown was requested - if so, this close is expected
        if not self.shutdown_requested:
            log_console(logging.WARNING, "WebSocket connection closed unexpectedly.")
            # Set active flag to False to trigger reconnection by the monitor thread
            with self.ws_lock:
                if self.ws is not None:  # Ensure we only log state change if WS object existed
                    log_console(logging.INFO, "WS state set to inactive due to close.")
                self.ws_active = False
                # Don't set self.ws = None here, _close_websocket_connection handles it
        else:
            log_console(logging.INFO, "WebSocket connection closed (shutdown requested).")
            with self.ws_lock:  # Ensure state is consistent even on expected close
                self.ws_active = False
                # Don't set self.ws = None here

    def run(self):
        """Main trading loop."""
        if not self.traders:
            log_console(logging.ERROR, "No traders initialized. Cannot run.")
            self.running = False
            self.shutdown_requested = True
            self._cleanup_resources()
            return

        log_console(
            logging.INFO,
            f"--- Starting Trading Cycles ({len(self.traders)} Symbols) ---",
        )
        sleep_interval: float = float(self.bot_cfg.get("SLEEP_INTERVAL_SECONDS", 60.0))
        metrics_interval: int = int(self.bot_cfg.get("METRICS_LOG_INTERVAL_SECONDS", 3600))
        last_balance_check_time: float = 0.0
        balance_check_interval: int = self.bot_cfg.get("BALANCE_CHECK_INTERVAL_SECONDS", 300)
        balance: float = 0.0  # Initialize balance

        # Initial Balance Check
        if self.dry_run:
            balance = float(self.risk_cfg.get("DRY_RUN_DUMMY_BALANCE", 10000.0))
            log_console(logging.INFO, f"Using Dry Run Balance: {balance:.2f} USDT")
        else:
            balance = get_available_balance(self.session, "USDT", self.account_type)
            if balance > FLOAT_COMPARISON_TOLERANCE:
                log_console(
                    logging.INFO,
                    f"Initial Balance Check ({self.account_type}): {balance:.2f} USDT Available",
                )
                last_balance_check_time = time.time()
            else:
                log_console(
                    logging.WARNING,
                    f"Initial {self.account_type} USDT balance is zero or fetch failed. Trades cannot be sized.",
                )

        while self.running and not self.shutdown_requested:
            try:
                start_time = time.time()
                current_dt_utc = dt.datetime.now(dt.timezone.utc)
                print(
                    f"\n{Fore.BLUE}{Style.BRIGHT}===== CYCLE START: {current_dt_utc.isoformat(timespec='seconds')} ====="
                )

                # Pre-Cycle Checks
                ws_currently_active = False
                with self.ws_lock:
                    ws_currently_active = self.ws_active
                if (
                    self.ws_connection_thread
                    and not ws_currently_active
                    and any(t.use_websocket for t in self.traders.values())
                ):
                    log_console(
                        logging.WARNING,
                        "WebSocket connection inactive. Strategy relies on REST API polling.",
                    )

                # Get Global State (Balance) - Periodically
                current_time = time.time()
                if not self.dry_run and (current_time - last_balance_check_time > balance_check_interval):
                    fetched_balance = get_available_balance(self.session, "USDT", self.account_type)
                    if abs(fetched_balance - balance) > 0.01:  # Log if changed significantly
                        log_console(
                            logging.INFO,
                            f"Checked Balance ({self.account_type}): {fetched_balance:.2f} USDT Available",
                        )
                    balance = fetched_balance
                    last_balance_check_time = current_time
                    if balance <= FLOAT_COMPARISON_TOLERANCE:
                        log_console(
                            logging.WARNING,
                            f"{self.account_type} USDT balance is zero. No new trades can be sized.",
                        )

                # Balance Allocation per Trader
                num_active_traders = len(self.traders)
                # Allocate balance only if positive
                balance_per_trader = (
                    (balance / num_active_traders)
                    if num_active_traders > 0 and balance > FLOAT_COMPARISON_TOLERANCE
                    else 0.0
                )
                max_pos_usdt_config = self.risk_cfg.get("MAX_POSITION_USDT")
                max_pos_usdt_per_trader: Optional[float] = None
                if (
                    isinstance(max_pos_usdt_config, (int, float))
                    and max_pos_usdt_config > FLOAT_COMPARISON_TOLERANCE
                    and num_active_traders > 0
                ):
                    max_pos_usdt_per_trader = max_pos_usdt_config / num_active_traders
                    log_console(
                        logging.DEBUG,
                        f"Allocated Bal/Trader: {balance_per_trader:.2f}, Max Pos Value/Trader: {max_pos_usdt_per_trader:.2f}",
                    )
                elif balance_per_trader > FLOAT_COMPARISON_TOLERANCE:
                    log_console(
                        logging.DEBUG,
                        f"Allocated Bal/Trader: {balance_per_trader:.2f}, Max Pos Value/Trader: Unlimited",
                    )

                # Iterate Through Symbols (use list keys for safe iteration if traders could be removed)
                for symbol in list(self.traders.keys()):
                    if not self.running or self.shutdown_requested:
                        break  # Check shutdown flag within loop
                    trader = self.traders[symbol]
                    symbol_color_prefix = f"{get_symbol_color(symbol)}[{symbol}]{Style.RESET_ALL}"
                    print(f"{symbol_color_prefix} --- Processing Symbol ---")

                    try:
                        # Cooldown Check: Skip processing if recently exited this symbol
                        if trader.last_exit_time:
                            cooldown_seconds = trader.bot_cfg.get("COOLDOWN_PERIOD_SECONDS", 300)
                            if cooldown_seconds > 0:
                                time_since_exit = (current_dt_utc - trader.last_exit_time).total_seconds()
                                if time_since_exit < cooldown_seconds:
                                    log_console(
                                        logging.INFO,
                                        f"Cooldown active ({cooldown_seconds - time_since_exit:.0f}s left). Skipping cycle.",
                                        symbol=symbol,
                                    )
                                    continue  # Skip to next symbol
                                else:
                                    trader.last_exit_time = None  # Cooldown finished

                        # Get Latest Data (handles WS/REST switching and indicator calculation)
                        df_with_indicators = trader.get_ohlcv()
                        # Need +1 for iloc[-2]
                        if (
                            df_with_indicators is None
                            or df_with_indicators.empty
                            or len(df_with_indicators) < MIN_KLINE_RECORDS_FOR_CALC + 1
                        ):
                            log_console(
                                logging.DEBUG,
                                f"Insufficient data ({len(df_with_indicators) if df_with_indicators is not None else 0} rows) or calculation failed. Skipping cycle.",
                                symbol=symbol,
                            )
                            continue

                        # Access candle data safely using iloc (handles potential index gaps)
                        try:
                            # Use -2 for the last fully closed candle, -1 for the latest (potentially incomplete)
                            last_closed_candle = df_with_indicators.iloc[-2]
                            latest_candle = df_with_indicators.iloc[-1]
                        except IndexError:
                            log_console(
                                logging.WARNING,
                                f"Cannot access candle data (need >= {MIN_KLINE_RECORDS_FOR_CALC + 1} rows, got {len(df_with_indicators)}). Skipping.",
                                symbol=symbol,
                            )
                            continue

                        # Validate key data on the closed candle used for signals
                        critical_indicators = [
                            "close",
                            "atr_risk",
                            "long_signal",
                            "short_signal",
                            "exit_long_signal",
                            "exit_short_signal",
                        ]
                        if any(pd.isna(last_closed_candle.get(ind)) for ind in critical_indicators):
                            log_console(
                                logging.WARNING,
                                f"Critical indicator NaN on closed candle ({last_closed_candle.name}). Skipping cycle.",
                                symbol=symbol,
                            )
                            continue
                        # Get current price (use latest close, fallback to last closed close)
                        current_price = latest_candle.get("close")
                        if pd.isnull(current_price) or current_price <= FLOAT_COMPARISON_TOLERANCE:
                            current_price = last_closed_candle.get("close")  # Fallback
                            if pd.isnull(current_price) or current_price <= FLOAT_COMPARISON_TOLERANCE:
                                log_console(
                                    logging.ERROR,
                                    "Cannot get valid current price. Skipping cycle.",
                                    symbol=symbol,
                                )
                                continue

                        # Position Management: Check current position status
                        # Log errors on position fetch
                        position = trader.get_position(log_error=True)

                        if position:
                            # --- Manage Existing Position ---
                            pos_side = position.get("side")
                            pos_size = position.get("size")  # float
                            entry_price = position.get("avgPrice")  # float

                            # Double-check position data validity
                            min_significant_qty_check = max(
                                FLOAT_COMPARISON_TOLERANCE,
                                trader.min_order_qty * 0.01,
                                trader.qty_step_float * 0.1,
                            )
                            if not (
                                pos_side in ["Buy", "Sell"]
                                and isinstance(pos_size, float)
                                and abs(pos_size) >= min_significant_qty_check
                                and isinstance(entry_price, float)
                                and entry_price > FLOAT_COMPARISON_TOLERANCE
                            ):
                                log_console(
                                    logging.ERROR,
                                    f"Invalid position data retrieved: {position}. Clearing internal state.",
                                    symbol=symbol,
                                )
                                trader.current_trade = None
                                position = None  # Clear state and proceed as if no position
                                # Continue to next symbol to avoid acting on bad data this cycle
                                continue

                            pos_size_abs = abs(pos_size)
                            log_console(
                                logging.INFO,
                                f"Position: {pos_side}, Size: {pos_size_abs:.{trader.qty_precision}f}, Entry: {entry_price:.{trader.price_precision}f}",
                                symbol=symbol,
                            )

                            # Check Exit Conditions based on last closed candle
                            exit_reason: Optional[str] = None
                            exit_long_sig = last_closed_candle.get("exit_long_signal", False)
                            exit_short_sig = last_closed_candle.get("exit_short_signal", False)
                            if pos_side == "Buy" and exit_long_sig:
                                exit_reason = "Strategy Exit (Long)"
                            elif pos_side == "Sell" and exit_short_sig:
                                exit_reason = "Strategy Exit (Short)"

                            # TODO: Implement Trailing Stop Loss logic here if enabled in config
                            # Example placeholder:
                            # if trader.risk_cfg.get("ENABLE_TRAILING_STOP", False):
                            #    tsl_triggered, tsl_price = check_trailing_stop(position, current_price, ...)
                            #    if tsl_triggered: exit_reason = f"Trailing Stop ({tsl_price})"

                            if exit_reason:
                                log_console(
                                    logging.INFO,
                                    f"Exit condition triggered: {exit_reason}. Attempting to close position.",
                                    symbol=symbol,
                                )
                                # Attempt to close the position
                                if trader.close_position(
                                    position,
                                    exit_reason=exit_reason,
                                    exit_price_est=current_price,
                                ):
                                    log_console(
                                        logging.INFO,
                                        "Position closed successfully based on signal.",
                                        symbol=symbol,
                                    )
                                    # Apply cooldown after successful close signal
                                    trader.last_exit_time = dt.datetime.now(dt.timezone.utc)
                                else:
                                    # Critical log if closure fails after signal
                                    log_console(
                                        logging.ERROR,
                                        f"Failed to close position after exit signal '{exit_reason}'. Manual check required!",
                                        symbol=symbol,
                                    )
                                # Continue to next symbol after close attempt (successful or not)
                                continue
                            else:
                                log_console(
                                    logging.DEBUG,
                                    "Holding position. No exit signal detected.",
                                    symbol=symbol,
                                )

                        else:  # No Position Open
                            # --- Check Entry Conditions ---
                            # Check Higher Timeframe Filter first if enabled
                            htf_allows_trade = trader.get_higher_tf_trend()
                            if not htf_allows_trade:
                                log_console(
                                    logging.INFO,
                                    "HTF filter unfavorable. Skipping entry check.",
                                    symbol=symbol,
                                )
                                continue  # Skip entry if HTF filter blocks

                            # Get signals and parameters from the last closed candle
                            atr_val = last_closed_candle.get("atr_risk", 0.0)
                            long_signal = last_closed_candle.get("long_signal", False)
                            short_signal = last_closed_candle.get("short_signal", False)
                            long_strength = last_closed_candle.get("long_signal_strength", 0.0)
                            short_strength = last_closed_candle.get("short_signal_strength", 0.0)

                            # Get risk parameters
                            risk_percent = float(trader.risk_cfg.get("RISK_PER_TRADE_PERCENT", 1.0))
                            sl_mult = float(trader.risk_cfg.get("ATR_MULTIPLIER_SL", 1.5))
                            tp_mult = float(trader.risk_cfg.get("ATR_MULTIPLIER_TP", 2.0))
                            min_strength = float(trader.strategy_cfg.get("MIN_SIGNAL_STRENGTH", 0.5))

                            entry_side: Optional[str] = None
                            stop_loss_price: Optional[float] = None
                            take_profit_price: Optional[float] = None
                            signal_strength: float = 0.0
                            entry_price_est = current_price  # Use latest price for SL/TP calculation

                            # Determine entry side based on signals and strength
                            if long_signal and long_strength >= min_strength:
                                entry_side = "Buy"
                                signal_strength = long_strength
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and sl_mult > FLOAT_COMPARISON_TOLERANCE:
                                    stop_loss_price = entry_price_est - (atr_val * sl_mult)
                                # TP is optional, calculate only if multiplier > 0
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and tp_mult > FLOAT_COMPARISON_TOLERANCE:
                                    take_profit_price = entry_price_est + (atr_val * tp_mult)
                            elif short_signal and short_strength >= min_strength:
                                entry_side = "Sell"
                                signal_strength = short_strength
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and sl_mult > FLOAT_COMPARISON_TOLERANCE:
                                    stop_loss_price = entry_price_est + (atr_val * sl_mult)
                                if atr_val > FLOAT_COMPARISON_TOLERANCE and tp_mult > FLOAT_COMPARISON_TOLERANCE:
                                    take_profit_price = entry_price_est - (atr_val * tp_mult)

                            if entry_side:
                                log_console(
                                    logging.DEBUG,
                                    f"Entry Signal: Side={entry_side}, Str={signal_strength:.2f}, EstEntry={entry_price_est:.{trader.price_precision}f}",
                                    symbol=symbol,
                                )

                                # Validate calculated SL/TP before proceeding
                                if stop_loss_price is None or stop_loss_price <= FLOAT_COMPARISON_TOLERANCE:
                                    log_console(
                                        logging.WARNING,
                                        f"Invalid SL calculated ({stop_loss_price}) from ATR={atr_val}. Skipping entry.",
                                        symbol=symbol,
                                    )
                                    continue
                                # Ensure TP is valid and logical relative to entry (avoid TP inside SL)
                                if take_profit_price is not None:
                                    if take_profit_price <= FLOAT_COMPARISON_TOLERANCE:
                                        log_console(
                                            logging.DEBUG,
                                            f"Invalid TP calculated ({take_profit_price}). TP not set.",
                                            symbol=symbol,
                                        )
                                        take_profit_price = None
                                    elif entry_side == "Buy" and take_profit_price <= stop_loss_price:
                                        log_console(
                                            logging.DEBUG,
                                            f"Long TP ({take_profit_price}) <= SL ({stop_loss_price}). TP not set.",
                                            symbol=symbol,
                                        )
                                        take_profit_price = None
                                    elif entry_side == "Sell" and take_profit_price >= stop_loss_price:
                                        log_console(
                                            logging.DEBUG,
                                            f"Short TP ({take_profit_price}) >= SL ({stop_loss_price}). TP not set.",
                                            symbol=symbol,
                                        )
                                        take_profit_price = None

                                # Calculate Position Size
                                sl_distance_price_abs = abs(entry_price_est - stop_loss_price)
                                if sl_distance_price_abs <= FLOAT_COMPARISON_TOLERANCE:
                                    log_console(
                                        logging.WARNING,
                                        f"SL distance is zero or negative ({sl_distance_price_abs}). Cannot size trade. Skipping.",
                                        symbol=symbol,
                                    )
                                    continue

                                qty_to_trade = calculate_position_size_atr(
                                    balance=balance_per_trader,
                                    risk_percent=risk_percent,
                                    sl_distance_price=sl_distance_price_abs,
                                    entry_price=entry_price_est,
                                    min_order_qty=trader.min_order_qty,
                                    qty_step_float=trader.qty_step_float,
                                    qty_precision=trader.qty_precision,
                                    max_position_usdt=max_pos_usdt_per_trader,
                                )

                                if qty_to_trade > FLOAT_COMPARISON_TOLERANCE:
                                    # Log the intended trade details clearly
                                    tp_log_str = (
                                        f"{take_profit_price:.{trader.price_precision}f}"
                                        if take_profit_price
                                        else "None"
                                    )
                                    sl_log_str = f"{stop_loss_price:.{trader.price_precision}f}"
                                    log_console(
                                        logging.INFO,
                                        f"{entry_side.upper()} ENTRY SIGNAL. Str={signal_strength:.2f}. Size={qty_to_trade:.{trader.qty_precision}f}. SL={sl_log_str}, TP={tp_log_str}",
                                        symbol=symbol,
                                    )
                                    # Place the order
                                    order_id = trader.place_order(
                                        entry_side,
                                        qty_to_trade,
                                        stop_loss_price,
                                        take_profit_price,
                                    )
                                    if order_id:
                                        log_console(
                                            logging.INFO,
                                            f"{entry_side.upper()} order placed successfully (ID: {order_id}).",
                                            symbol=symbol,
                                        )
                                        # Optional: Immediately query position after placing order to confirm internal state?
                                        # time.sleep(1)
                                        # updated_pos = trader.get_position()
                                        # if updated_pos and updated_pos.get('orderId') == trader.current_trade.get('orderId'): ...
                                    else:
                                        log_console(
                                            logging.ERROR,
                                            f"{entry_side.upper()} order placement failed.",
                                            symbol=symbol,
                                        )
                                    # Continue to next symbol after entry attempt
                                    continue
                                else:
                                    log_console(
                                        logging.INFO,
                                        f"{entry_side.upper()} signal detected, but calculated size is zero (Balance={balance_per_trader:.2f}, Risk%={risk_percent}, SLDist={sl_distance_price_abs:.{trader.price_precision}f}). Skipping entry.",
                                        symbol=symbol,
                                    )
                            else:  # No entry signal met criteria
                                log_console(
                                    logging.DEBUG,
                                    "No valid entry signal detected or strength below threshold.",
                                    symbol=symbol,
                                )

                    except Exception as symbol_e:
                        log_console(
                            logging.ERROR,
                            f"Error processing symbol {symbol}: {symbol_e}",
                            symbol=symbol,
                            exc_info=True,
                        )
                        # Continue to the next symbol

                    log_console(
                        logging.DEBUG,
                        f"--- Finished Processing Symbol: {symbol} ---",
                        symbol=symbol,
                    )
                # End of symbol loop

            except KeyboardInterrupt:
                log_console(
                    logging.INFO,
                    "KeyboardInterrupt detected in main loop. Initiating shutdown...",
                )
                self.shutdown_requested = True
                self.running = False
                break  # Exit the main loop
            except Exception as loop_e:
                log_console(
                    logging.CRITICAL,
                    f"CRITICAL ERROR in main trading loop: {loop_e}. Attempting shutdown.",
                    exc_info=True,
                )
                self.shutdown_requested = True
                self.running = False
                break  # Exit the main loop

            # --- Post-Cycle Actions ---
            if self.running and not self.shutdown_requested:
                # Log summary metrics periodically
                self.metrics.log_summary(force=False, interval=metrics_interval)

                # Calculate sleep duration
                end_time = time.time()
                cycle_duration = end_time - start_time
                # Ensure minimum sleep
                sleep_duration = max(0.1, sleep_interval - cycle_duration)
                log_console(
                    logging.DEBUG,
                    f"Cycle duration: {cycle_duration:.2f}s. Sleeping for {sleep_duration:.2f}s...",
                )
                print(f"{Fore.BLUE}{Style.BRIGHT}===== CYCLE END =====")
                try:
                    # Sleep, but allow KeyboardInterrupt to break sleep
                    time.sleep(sleep_duration)
                except KeyboardInterrupt:
                    log_console(
                        logging.INFO,
                        "KeyboardInterrupt detected during sleep. Initiating shutdown...",
                    )
                    self.shutdown_requested = True
                    self.running = False
                    break  # Exit the main loop
        # End of main while loop

        log_console(logging.INFO, "Main trading loop exited.")
        self.shutdown()  # Ensure shutdown sequence is called

    def _cleanup_resources(self):
        """Closes WS connection, intended for use after fatal errors during init or runtime."""
        log_console(logging.INFO, "Cleaning up resources...")
        self._close_websocket_connection()
        # Add any other resource cleanup needed (e.g., closing file handles if not managed by logging)
        log_console(logging.INFO, "Resource cleanup finished.")

    def shutdown(self):
        """Handles graceful shutdown procedures."""
        # Prevent multiple shutdown calls
        if self.shutdown_requested and not self.running:
            log_console(logging.DEBUG, "Shutdown already in progress or completed.")
            return

        log_console(logging.INFO, "--- Initiating Graceful Shutdown ---")
        self.shutdown_requested = True  # Signal all loops/threads to stop
        self.running = False  # Stop main loop if it hasn't already

        # Stop WebSocket thread first
        if self.ws_connection_thread and self.ws_connection_thread.is_alive():
            log_console(logging.INFO, "Waiting for WebSocket thread to stop...")
            # Wait for the thread to finish, with a timeout
            ws_join_timeout = self.bot_cfg.get("WS_SHUTDOWN_TIMEOUT_SECONDS", 15)
            self.ws_connection_thread.join(timeout=ws_join_timeout)
            if self.ws_connection_thread.is_alive():
                log_console(
                    logging.WARNING,
                    f"WebSocket thread join timed out after {ws_join_timeout}s. Forcing WS closure.",
                )
                # Force close WS object if thread is stuck (should be rare)
                self._close_websocket_connection()
            else:
                log_console(logging.INFO, "WebSocket thread stopped.")
        elif self.ws_connection_thread:  # Thread exists but is not alive
            log_console(logging.INFO, "WebSocket thread already stopped.")
            self._close_websocket_connection()  # Ensure WS object is closed/cleaned
        else:
            log_console(logging.INFO, "No WebSocket thread was running.")

        # Optional: Close open positions if configured
        close_on_exit = self.bot_cfg.get("CLOSE_POSITIONS_ON_SHUTDOWN", False)
        if close_on_exit and not self.dry_run:
            log_console(logging.WARNING, "Attempting to close open positions on shutdown...")
            closed_count = 0
            failed_closes = []
            # Iterate over a copy of trader items
            for symbol, trader in list(self.traders.items()):
                try:
                    # Fetch position, retry once with short delay if first attempt fails
                    # Don't flood logs initially
                    position = trader.get_position(log_error=False)
                    if position is None:
                        time.sleep(0.5)
                        position = trader.get_position(log_error=True)

                    if position:
                        log_console(
                            logging.WARNING,
                            f"Closing position for {symbol} due to shutdown...",
                            symbol=symbol,
                        )
                        # Get last price estimate for metrics fallback
                        last_price: Optional[float] = None
                        if trader.kline_df is not None and not trader.kline_df.empty:
                            try:
                                last_price = float(trader.kline_df["close"].iloc[-1])
                            except (IndexError, ValueError, TypeError):
                                pass
                        # Attempt close
                        if trader.close_position(position, exit_reason="Shutdown", exit_price_est=last_price):
                            closed_count += 1
                            # Small delay between close attempts
                            time.sleep(0.5)
                        else:
                            failed_closes.append(symbol)
                            log_console(
                                logging.ERROR,
                                f"Failed to confirm closure for {symbol} during shutdown.",
                                symbol=symbol,
                            )
                except Exception as e:
                    log_console(
                        logging.ERROR,
                        f"Error closing {symbol} during shutdown: {e}",
                        symbol=symbol,
                        exc_info=False,
                    )
                    failed_closes.append(symbol)

            if closed_count > 0:
                log_console(
                    logging.WARNING,
                    f"Submitted close orders for {closed_count} positions.",
                )
            if failed_closes:
                log_console(
                    logging.CRITICAL,
                    f"FAILED TO CLOSE POSITIONS FOR: {', '.join(failed_closes)}. MANUAL ACTION REQUIRED.",
                )
            elif close_on_exit:
                log_console(logging.INFO, "Position closing attempt complete.")

        elif self.dry_run and close_on_exit:
            log_console(
                logging.INFO,
                "Dry run mode: No real positions to close. Logging metrics for simulated trades.",
            )
            # Log metrics for any open simulated trades
            closed_sim_count = 0
            for symbol, trader in list(self.traders.items()):
                if trader.current_trade:
                    log_console(
                        logging.INFO,
                        f"[DRY RUN] Logging final metrics for simulated position {symbol}",
                        symbol=symbol,
                    )
                    # Simulate close at last known price for metrics
                    last_price_sim: Optional[float] = None
                    if trader.kline_df is not None and not trader.kline_df.empty:
                        try:
                            last_price_sim = float(trader.kline_df["close"].iloc[-1])
                        except:
                            pass
                    # Fallback if price unavailable
                    entry_price_sim = trader.current_trade.get("avgPrice", 0)
                    if last_price_sim is None or last_price_sim <= FLOAT_COMPARISON_TOLERANCE:
                        last_price_sim = entry_price_sim if entry_price_sim > FLOAT_COMPARISON_TOLERANCE else 1.0

                    trader.metrics.add_trade(
                        symbol=symbol,
                        entry_time=trader.current_trade.get(
                            "entry_time",
                            dt.datetime.now(
                                # Fallback entry time
                                dt.timezone.utc
                            )
                            - dt.timedelta(minutes=1),
                        ),
                        exit_time=dt.datetime.now(dt.timezone.utc),
                        side=trader.current_trade.get("side", "Buy"),
                        entry_price=entry_price_sim,
                        exit_price=last_price_sim,
                        qty=abs(trader.current_trade.get("size", 0)),
                        leverage=trader.current_trade.get("leverage", "1"),
                    )
                    trader.current_trade = None  # Clear simulated trade
                    closed_sim_count += 1
            if closed_sim_count > 0:
                log_console(
                    logging.INFO,
                    f"[DRY RUN] Logged final metrics for {closed_sim_count} simulated positions.",
                )

        else:  # Close on exit disabled
            log_console(logging.INFO, "Close on shutdown disabled. Positions may remain open.")

        # Save final metrics summary
        try:
            self.metrics.save_on_shutdown()
        except Exception as metrics_e:
            log_console(logging.ERROR, f"Error saving final metrics: {metrics_e}", exc_info=True)

        log_console(logging.INFO, "--- Bot Shutdown Complete ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bybit Enhanced Momentum Scanner Bot (V5 API)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Path to JSON config file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode (simulation only).")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging for console and file.",
    )
    args = parser.parse_args()

    # Set logging level based on --debug flag
    log_level_console = logging.DEBUG if args.debug else logging.INFO
    # Set file logger level too
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    log_console(log_level_console, f"{Fore.CYAN}{Style.BRIGHT}--- Bot Starting ---")
    log_console(log_level_console, f"Using configuration file: {args.config}")
    if args.dry_run:
        # Make dry run warning more prominent
        print(
            f"\n{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT} *** WARNING: DRY RUN MODE ACTIVE - NO REAL TRADES WILL BE PLACED *** {Style.RESET_ALL}\n"
        )
        log_console(logging.WARNING, "*** DRY RUN MODE ACTIVE ***")
    if args.debug:
        log_console(logging.DEBUG, "Debug logging enabled.")

    # --- Load and Validate Configuration ---
    config: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = []
    try:
        # Basic load and KLINE_LIMIT calculation happens here
        config = load_config(args.config)

        # --- Detailed Validation of Loaded Config ---
        log_console(logging.INFO, "Validating configuration sections...")
        required_sections = {
            "BYBIT_CONFIG": dict,
            "RISK_CONFIG": dict,
            "BOT_CONFIG": dict,
        }
        for section, expected_type in required_sections.items():
            if section not in config:
                validation_errors.append(f"Missing required section '{section}'.")
            elif not isinstance(config.get(section), expected_type):
                validation_errors.append(f"Section '{section}' must be a {expected_type.__name__}.")

        if not validation_errors:  # Proceed with detailed checks only if top sections are okay
            bybit_cfg = config.get("BYBIT_CONFIG", {})
            risk_cfg = config.get("RISK_CONFIG", {})
            bot_cfg = config.get("BOT_CONFIG", {})
            symbols = bybit_cfg.get("SYMBOLS", [])

            # Bybit Config Validation
            if not isinstance(bybit_cfg.get("USE_TESTNET", False), bool):
                validation_errors.append("BYBIT_CONFIG: USE_TESTNET must be true or false.")
            account_type = bybit_cfg.get("ACCOUNT_TYPE", "UNIFIED").upper()
            valid_account_types = ["UNIFIED", "CONTRACT", "SPOT"]  # V5 relevant types
            if account_type not in valid_account_types:
                validation_errors.append(
                    f"BYBIT_CONFIG: ACCOUNT_TYPE '{account_type}' invalid (must be one of {valid_account_types})."
                )

            # Risk Config Validation
            risk_percent = risk_cfg.get("RISK_PER_TRADE_PERCENT")
            if not isinstance(risk_percent, (int, float)) or not (FLOAT_COMPARISON_TOLERANCE < risk_percent <= 10):
                validation_errors.append(
                    f"RISK_CONFIG: RISK_PER_TRADE_PERCENT ({risk_percent}) invalid (must be a positive number, e.g., 0.1 to 10)."
                )
            fee_rate = risk_cfg.get("FEE_RATE")
            if not isinstance(fee_rate, (int, float)) or fee_rate < 0:
                validation_errors.append(f"RISK_CONFIG: FEE_RATE ({fee_rate}) must be a non-negative number.")
            max_pos_usdt = risk_cfg.get("MAX_POSITION_USDT")
            if max_pos_usdt is not None and (
                not isinstance(max_pos_usdt, (int, float)) or max_pos_usdt <= FLOAT_COMPARISON_TOLERANCE
            ):
                validation_errors.append(
                    f"RISK_CONFIG: MAX_POSITION_USDT ({max_pos_usdt}) must be a positive number if set."
                )
            if args.dry_run:
                dry_run_balance = risk_cfg.get("DRY_RUN_DUMMY_BALANCE")
                if dry_run_balance is None or (
                    not isinstance(dry_run_balance, (int, float)) or dry_run_balance <= FLOAT_COMPARISON_TOLERANCE
                ):
                    validation_errors.append(
                        f"RISK_CONFIG: DRY_RUN_DUMMY_BALANCE ({dry_run_balance}) must be a positive number when using --dry-run."
                    )
            if (
                not isinstance(risk_cfg.get("ATR_MULTIPLIER_SL", 1.5), (int, float))
                or risk_cfg.get("ATR_MULTIPLIER_SL", 1.5) <= FLOAT_COMPARISON_TOLERANCE
            ):
                validation_errors.append("RISK_CONFIG: ATR_MULTIPLIER_SL must be a positive number.")
            if (
                not isinstance(risk_cfg.get("ATR_MULTIPLIER_TP", 2.0), (int, float))
                or risk_cfg.get("ATR_MULTIPLIER_TP", 2.0) < 0
            ):
                validation_errors.append("RISK_CONFIG: ATR_MULTIPLIER_TP must be a non-negative number (0 to disable).")
            if not isinstance(risk_cfg.get("ENABLE_TRAILING_STOP", False), bool):
                validation_errors.append("RISK_CONFIG: ENABLE_TRAILING_STOP must be true or false.")
            # Add validation for TSL parameters if TSL is enabled (e.g., tsl_atr_mult, tsl_activation_atr)

            # Bot Config Validation (Check numeric types and ranges)
            def validate_numeric(
                cfg: dict,
                key: str,
                req_type: type,
                min_val: Optional[Union[int, float]] = None,
                max_val: Optional[Union[int, float]] = None,
                allow_none: bool = False,
            ):
                val = cfg.get(key)
                if val is None and allow_none:
                    return
                if val is None and not allow_none:
                    validation_errors.append(f"BOT_CONFIG: '{key}' is missing.")
                    return
                if not isinstance(val, req_type):
                    validation_errors.append(f"BOT_CONFIG: '{key}' ({val}) must be type {req_type.__name__}.")
                    return
                if min_val is not None and val < min_val:
                    validation_errors.append(f"BOT_CONFIG: '{key}' ({val}) must be >= {min_val}.")
                if max_val is not None and val > max_val:
                    validation_errors.append(f"BOT_CONFIG: '{key}' ({val}) must be <= {max_val}.")

            validate_numeric(bot_cfg, "SLEEP_INTERVAL_SECONDS", (int, float), min_val=0.1)
            validate_numeric(bot_cfg, "KLINE_LIMIT_BUFFER", int, min_val=0)
            validate_numeric(bot_cfg, "API_RECV_WINDOW", int, min_val=1000, max_val=60000)  # Typical range 1s-60s
            validate_numeric(bot_cfg, "WS_PING_INTERVAL", int, min_val=5)
            validate_numeric(bot_cfg, "WS_PING_TIMEOUT", int, min_val=1)
            validate_numeric(bot_cfg, "WS_STARTUP_DELAY_SECONDS", (int, float), min_val=0)
            validate_numeric(bot_cfg, "ORDER_CONFIRM_RETRIES", int, min_val=0)
            validate_numeric(bot_cfg, "ORDER_CONFIRM_DELAY_SECONDS", (int, float), min_val=0.1)
            if not isinstance(bot_cfg.get("PROCESS_CONFIRMED_KLINE_ONLY", True), bool):
                validation_errors.append("BOT_CONFIG: PROCESS_CONFIRMED_KLINE_ONLY must be true or false.")
            if not isinstance(bot_cfg.get("CLOSE_POSITIONS_ON_SHUTDOWN", False), bool):
                validation_errors.append("BOT_CONFIG: CLOSE_POSITIONS_ON_SHUTDOWN must be true or false.")
            validate_numeric(bot_cfg, "HIGHER_TF_CACHE_SECONDS", int, min_val=0)
            validate_numeric(bot_cfg, "KLINE_STALENESS_SECONDS", int, min_val=1)
            validate_numeric(bot_cfg, "BALANCE_CHECK_INTERVAL_SECONDS", int, min_val=1)
            validate_numeric(bot_cfg, "COOLDOWN_PERIOD_SECONDS", int, min_val=0)
            validate_numeric(bot_cfg, "METRICS_LOG_INTERVAL_SECONDS", int, min_val=1)
            validate_numeric(bot_cfg, "MAX_CLOCK_SKEW_SECONDS", int, min_val=1)
            validate_numeric(bot_cfg, "WS_RECONNECT_DELAY_INIT_SECONDS", (int, float), min_val=1)
            validate_numeric(bot_cfg, "WS_RECONNECT_DELAY_MAX_SECONDS", (int, float), min_val=1)
            validate_numeric(bot_cfg, "WS_CONNECT_TIMEOUT_SECONDS", int, min_val=5)
            validate_numeric(bot_cfg, "WS_MONITOR_CHECK_INTERVAL_SECONDS", (int, float), min_val=1)
            validate_numeric(bot_cfg, "WS_SHUTDOWN_TIMEOUT_SECONDS", int, min_val=1)
            validate_numeric(bot_cfg, "INITIAL_FETCH_DELAY_PER_SYMBOL_MS", int, min_val=0)

            # Symbol Configs Validation
            if not isinstance(symbols, list):
                validation_errors.append("BYBIT_CONFIG: SYMBOLS must be a list.")
            elif not symbols:
                validation_errors.append("BYBIT_CONFIG: SYMBOLS list is empty. Add at least one symbol configuration.")
            else:
                seen_symbols = set()
                # Define valid timeframes based on Bybit V5 API documentation (adjust if needed)
                valid_timeframes = {
                    "1",
                    "3",
                    "5",
                    "15",
                    "30",
                    "60",
                    "120",
                    "240",
                    "360",
                    "720",
                    "D",
                    "W",
                    "M",
                }
                for i, symbol_cfg in enumerate(symbols):
                    if not isinstance(symbol_cfg, dict):
                        validation_errors.append(f"SYMBOLS entry {i}: Must be a dictionary.")
                        continue
                    symbol = symbol_cfg.get("SYMBOL")
                    if not symbol or not isinstance(symbol, str):
                        validation_errors.append(f"SYMBOLS entry {i}: Missing/invalid 'SYMBOL' key (must be a string).")
                        continue
                    if symbol in seen_symbols:
                        validation_errors.append(f"Duplicate symbol found: {symbol}")
                        continue
                    seen_symbols.add(symbol)

                    if "STRATEGY_CONFIG" not in symbol_cfg or not isinstance(symbol_cfg.get("STRATEGY_CONFIG"), dict):
                        validation_errors.append(
                            f"Symbol {symbol}: Missing or invalid 'STRATEGY_CONFIG' (must be a dictionary)."
                        )
                    leverage = symbol_cfg.get("LEVERAGE")
                    # Validate leverage based on account type (spot doesn't use leverage setting this way)
                    if account_type != "SPOT":
                        # Bybit max leverage varies, 100 is a common cap but check specific pairs
                        if not isinstance(leverage, int) or not (1 <= leverage <= 150):
                            validation_errors.append(
                                f"Symbol {symbol}: LEVERAGE ({leverage}) invalid (must be an integer between 1 and 150 for {account_type})."
                            )
                    timeframe = str(symbol_cfg.get("TIMEFRAME", ""))
                    if timeframe not in valid_timeframes:
                        validation_errors.append(
                            f"Symbol {symbol}: Invalid TIMEFRAME '{timeframe}' (must be one of {valid_timeframes})"
                        )
                    if not isinstance(symbol_cfg.get("USE_WEBSOCKET", True), bool):
                        validation_errors.append(f"Symbol {symbol}: USE_WEBSOCKET must be true or false.")

                    # Validate Strategy Config parameters within the symbol
                    strategy_cfg = symbol_cfg.get("STRATEGY_CONFIG", {})
                    if isinstance(strategy_cfg, dict):
                        # Check indicator periods (positive integers)
                        for k, v in strategy_cfg.items():
                            if "PERIOD" in k.upper() and (not isinstance(v, int) or v <= 0):
                                validation_errors.append(
                                    f"Symbol {symbol}: Strategy param '{k}' ({v}) must be a positive integer."
                                )
                            # Check boolean flags
                            if k.upper() in [
                                "USE_EHLERS_SMOOTHER",
                                "ENABLE_MULTI_TIMEFRAME",
                            ] and not isinstance(v, bool):
                                validation_errors.append(
                                    f"Symbol {symbol}: Strategy param '{k}' ({v}) must be true or false."
                                )
                            # Check numeric thresholds (allow floats or ints)
                            num_thresholds = [
                                "MOM_THRESHOLD_PERCENT",
                                "ADX_THRESHOLD",
                                "MIN_SIGNAL_STRENGTH",
                                "RSI_LOW_THRESHOLD",
                                "RSI_HIGH_THRESHOLD",
                                "VOLATILITY_FACTOR_MIN",
                                "VOLATILITY_FACTOR_MAX",
                                "RSI_VOLATILITY_ADJUST",
                                "BBANDS_STDDEV",
                            ]
                            if k.upper() in num_thresholds and not isinstance(v, (int, float)):
                                validation_errors.append(
                                    f"Symbol {symbol}: Strategy param '{k}' ({v}) must be a number."
                                )
                            # Check trigger price types
                            valid_trigger_types = [
                                "LastPrice",
                                "MarkPrice",
                                "IndexPrice",
                            ]
                            if k.upper() in ["SL_TRIGGER_BY", "TP_TRIGGER_BY"] and v not in valid_trigger_types:
                                validation_errors.append(
                                    f"Symbol {symbol}: Strategy param '{k}' ({v}) must be one of {valid_trigger_types}."
                                )
                            # Check TPSL mode
                            if k.upper() == "TPSL_MODE" and v not in [
                                "Full",
                                "Partial",
                            ]:
                                validation_errors.append(
                                    f"Symbol {symbol}: Strategy param 'TPSL_MODE' ({v}) must be 'Full' or 'Partial'."
                                )
                            # Check price source string
                            # Add others if implemented
                            valid_price_sources = [
                                "open",
                                "high",
                                "low",
                                "close",
                                "hlc3",
                                "ohlc4",
                            ]
                            if k.upper() == "PRICE_SOURCE" and (not isinstance(v, str) or v not in valid_price_sources):
                                validation_errors.append(
                                    f"Symbol {symbol}: Strategy param 'PRICE_SOURCE' ({v}) must be a valid string (e.g., {valid_price_sources})."
                                )

                        # Validate Multi-Timeframe settings if enabled
                        if strategy_cfg.get("ENABLE_MULTI_TIMEFRAME"):
                            htf = str(strategy_cfg.get("HIGHER_TIMEFRAME", ""))
                            if htf not in valid_timeframes:
                                validation_errors.append(
                                    f"Symbol {symbol}: Invalid HIGHER_TIMEFRAME '{htf}' (must be one of {valid_timeframes})."
                                )
                            elif htf == timeframe:
                                validation_errors.append(
                                    f"Symbol {symbol}: HIGHER_TIMEFRAME cannot be the same as TIMEFRAME."
                                )

    except SystemExit:  # Raised by load_config on file/JSON errors
        sys.exit(1)
    except Exception as cfg_e:
        log_console(
            logging.CRITICAL,
            f"Unexpected error loading or validating config: {cfg_e}",
            exc_info=True,
        )
        sys.exit(1)

    # Report Validation Results
    if validation_errors:
        log_console(
            logging.CRITICAL,
            f"{Back.RED}{Fore.WHITE} Configuration validation failed with {len(validation_errors)} error(s): {Style.RESET_ALL}",
        )
        for error in validation_errors:
            log_console(logging.CRITICAL, f"- {error}")
        sys.exit(1)
    else:
        log_console(logging.INFO, "Configuration validation passed successfully.")

    # --- Initialize and Run Bot ---
    trader_instance: Optional[MomentumScannerTrader] = None
    exit_code = 0
    try:
        log_console(logging.INFO, "Initializing main trader instance...")
        # Pass the validated config object
        trader_instance = MomentumScannerTrader(API_KEY, API_SECRET, config, args.dry_run)

        # Check if initialization succeeded (trader_instance.running should be True)
        if trader_instance and trader_instance.running:
            trader_instance.run()  # This call blocks until shutdown
        else:
            # This case should ideally be caught by exceptions during init, but safeguard here
            log_console(
                logging.CRITICAL,
                "Trader initialization failed or did not start. Bot cannot run.",
            )
            exit_code = 1

    except KeyboardInterrupt:
        log_console(logging.INFO, "\nKeyboardInterrupt detected. Initiating shutdown...")
        if trader_instance:
            trader_instance.shutdown()  # Trigger graceful shutdown
        else:
            log_console(
                logging.WARNING,
                "Trader instance was not fully initialized before interrupt.",
            )
        exit_code = 0  # Clean exit on user interrupt
    except (ConnectionError, RuntimeError, ValueError) as critical_e:
        # Catch specific critical errors that might occur during runtime or init
        log_console(
            logging.CRITICAL,
            f"Caught critical error: {critical_e}. Bot stopped.",
            exc_info=False,
        )
        if trader_instance:
            trader_instance.shutdown()  # Attempt shutdown
        exit_code = 1
    except Exception as e:
        # Catch any other unexpected exceptions in the main execution flow
        log_console(
            logging.CRITICAL,
            f"Unexpected critical error in main execution: {e}",
            exc_info=True,
        )
        if trader_instance:
            trader_instance.shutdown()  # Attempt shutdown
        exit_code = 1
    finally:
        # This block executes regardless of how the try block exits
        log_console(
            log_level_console,
            f"{Fore.CYAN}{Style.BRIGHT}--- Bot Execution Finished ---",
        )
        # Ensure logging resources are released
        logging.shutdown()
        # Exit with the determined exit code
        sys.exit(exit_code)
