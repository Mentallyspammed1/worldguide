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

try:
    import pandas_ta as ta  # Technical analysis library
except ImportError:
    print(
        "Error: 'pandas_ta' library not found. Please install it: pip install pandas_ta"
    )
    sys.exit(1)
try:
    # Use V5 unified_trading API
    from pybit.unified_trading import HTTP, WebSocket
except ImportError:
    print(
        "Error: 'pybit' library not found or out of date. Please install/upgrade it: pip install -U pybit"
    )
    sys.exit(1)
try:
    from colorama import (
        init as colorama_init,
        Fore,
        Style,
        Back,
    )  # Colored console output
except ImportError:
    print(
        "Error: 'colorama' library not found. Please install it: pip install colorama"
    )
    sys.exit(1)
try:
    from dotenv import load_dotenv  # Load environment variables from .env file
except ImportError:
    print(
        "Error: 'python-dotenv' library not found. Please install it: pip install python-dotenv"
    )
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
MAX_KLINE_LIMIT_PER_REQUEST = 1000  # Maximum klines fetchable in one V5 API call

# --- Logging Setup ---

# Base Logger Configuration (File Handler, no colors in file, UTF-8 encoded)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)

# Main application logger
logger = logging.getLogger("MomentumScannerTrader")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)  # Default level, can be overridden by --debug
logger.propagate = False  # Prevent duplicate logs if root logger is configured

# Metrics Logger Configuration (CSV-like format, UTF-8 encoded)
metrics_formatter = logging.Formatter(
    "%(asctime)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
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
    "BCHUSDT": Fore.MAGENTA + Style.BRIGHT,  # Example
    "DOGEUSDT": Fore.BLUE + Style.BRIGHT,  # Example
    "default": Fore.WHITE,
}
_symbol_color_cycle = [Fore.CYAN, Fore.YELLOW, Fore.MAGENTA, Fore.BLUE, Fore.GREEN]
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
        _symbol_color_map[symbol] = (
            _symbol_color_cycle[_color_index % len(_symbol_color_cycle)] + Style.BRIGHT
        )
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
        print(
            f"{Fore.RED}LOG FORMATTING ERROR:{Style.RESET_ALL} {fmt_e} | Original Msg: {message_str}"
        )
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
        kline_limit_buffer = bot_cfg.get("KLINE_LIMIT_BUFFER", 20)

        if not isinstance(symbols_cfg, list):
            raise ValueError("'BYBIT_CONFIG.SYMBOLS' must be a list.")

        for symbol_cfg in symbols_cfg:
            if not isinstance(symbol_cfg, dict) or "SYMBOL" not in symbol_cfg:
                raise ValueError(
                    "Each item in 'SYMBOLS' must be a dictionary with a 'SYMBOL' key."
                )

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

            # Filter out invalid or non-numeric periods
            valid_periods = [
                p for p in periods if isinstance(p, (int, float)) and p > 0
            ]
            if not valid_periods:
                log_console(
                    logging.WARNING,
                    f"No valid indicator periods found for {symbol}. Using default max period (100).",
                    symbol=symbol,
                )
                max_period = 100  # Default fallback if no periods found
            else:
                max_period = max(valid_periods)

            # Ensure kline limit is at least a minimum value (e.g., 2) for basic operations
            # Add buffer *after* finding max period.
            # The fetch function (`get_ohlcv_rest`) will cap this at MAX_KLINE_LIMIT_PER_REQUEST if needed.
            kline_limit = max(2, int(max_period + kline_limit_buffer))

            # Store calculated limit within the symbol's config (internal use)
            symbol_cfg.setdefault("INTERNAL", {})["KLINE_LIMIT"] = kline_limit
            log_console(
                logging.DEBUG,
                f"Calculated KLINE_LIMIT for {symbol}: {kline_limit} (Max Period: {max_period:.0f})",
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
    Requires minimum period of 2.

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

    # Drop NaNs for length check, but calculate on original series with NaNs handled
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

        # Use float, ffill to handle initial NaNs cautiously
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
            if (
                np.isnan(series_float.iloc[i])
                or np.isnan(series_float.iloc[i - 1])
                or np.isnan(ss[i - 1])
                or np.isnan(ss[i - 2])
            ):
                continue  # Skip this iteration, ss[i] remains NaN

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

    Args:
        series: pandas Series of price data.
        period: Lookback period (dominant cycle period, must be >= 4).

    Returns:
        pandas Series with Instantaneous Trendline values, or Series of NaNs on error/invalid input.
    """
    if not isinstance(series, pd.Series) or series.empty:
        log_console(
            logging.DEBUG, "Instantaneous Trendline: Input series is invalid or empty."
        )
        return pd.Series(np.nan, index=series.index, dtype=np.float64)
    if (
        not isinstance(period, int) or period < 4
    ):  # Ehlers IT generally needs more points
        log_console(
            logging.WARNING,
            f"Instantaneous Trendline: Period must be an integer >= 4, got {period}. Returning NaNs.",
        )
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    series_cleaned = series.dropna()
    if len(series_cleaned) < 4:
        log_console(
            logging.DEBUG,
            f"Instantaneous Trendline: Series length ({len(series_cleaned)} non-NaN) is less than required minimum (4). Returning NaNs.",
        )
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        alpha = 0.07  # Fixed alpha as per Ehlers' recommendation

        series_float = series.astype(float).ffill()
        it = np.full(len(series_float), np.nan, dtype=np.float64)

        # Initialize first few values robustly based on Ehlers common practice after ffill
        # Check source values are not NaN before using them
        if len(series_float) > 0 and not np.isnan(series_float.iloc[0]):
            it[0] = series_float.iloc[0]
        if len(series_float) > 1 and not np.isnan(series_float.iloc[1]):
            it[1] = series_float.iloc[1]
        if len(series_float) > 2 and not np.any(np.isnan(series_float.iloc[0:3])):
            it[2] = (
                series_float.iloc[2] + 2.0 * series_float.iloc[1] + series_float.iloc[0]
            ) / 4.0
        if len(series_float) > 3 and not np.any(np.isnan(series_float.iloc[1:4])):
            it[3] = (
                series_float.iloc[3] + 2.0 * series_float.iloc[2] + series_float.iloc[1]
            ) / 4.0

        # Calculation loop - check for NaNs rigorously
        for i in range(4, len(series_float)):
            if (
                np.isnan(series_float.iloc[i])
                or np.isnan(series_float.iloc[i - 1])
                or np.isnan(series_float.iloc[i - 2])
                or np.isnan(it[i - 1])
                or np.isnan(it[i - 2])
            ):
                continue  # Skip this iteration, it[i] remains NaN

            price_i = series_float.iloc[i]
            price_im1 = series_float.iloc[i - 1]
            price_im2 = series_float.iloc[i - 2]
            it_im1 = it[i - 1]
            it_im2 = it[i - 2]

            # Formula from Ehlers' papers/code:
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
    # Ensure KLINE_LIMIT is at least 2 for any calculation
    kline_limit = max(2, internal_cfg.get("KLINE_LIMIT", 120))
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
            df.index = pd.to_datetime(df.index)
            if not isinstance(df.index, pd.DatetimeIndex):  # Check conversion success
                raise ValueError("Index conversion to DatetimeIndex failed.")
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
                log_console(
                    logging.ERROR, f"Indicator Calc: Missing required column: '{col}'."
                )
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
        price_source = strategy_cfg.get("PRICE_SOURCE", "close")
        if price_source not in df_out.columns or df_out[price_source].isnull().all():
            log_console(
                logging.ERROR,
                f"Indicator Calc: Price source '{price_source}' not found or contains only NaNs.",
            )
            if (
                price_source != "close"
                and "close" in df_out.columns
                and not df_out["close"].isnull().all()
            ):
                log_console(
                    logging.WARNING,
                    "Indicator Calc: Falling back to 'close' as price source.",
                )
                price_source = "close"
            else:
                return None  # Cannot proceed without a valid price source

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
                    f"Indicator Calc: Invalid or missing {period_key} in strategy config. Value: {length}",
                )
                df_out[col_name] = np.nan
                continue

            try:
                if use_ehlers:
                    indicator_series = super_smoother(df_out[price_source], length)
                else:
                    indicator_series = ta.ema(df_out[price_source], length=length)

                if (
                    indicator_series is None
                    or indicator_series.empty
                    or indicator_series.isnull().all()
                ):
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
        df_out["trendline"] = np.nan
        if not isinstance(it_period, int) or it_period < 4:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Invalid INSTANTANEOUS_TRENDLINE_PERIOD ({it_period}). Must be integer >= 4. Skipping calculation.",
            )
        else:
            try:
                it_series = instantaneous_trendline(df_out[price_source], it_period)
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
            df_out["roc"] = ta.roc(df_out[price_source], length=roc_period)
            df_out["roc_smooth"] = ta.sma(
                df_out["roc"].fillna(0), length=3
            )  # Fill NaNs before smoothing
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
            atr_series = ta.atr(
                df_out["high"], df_out["low"], df_out["close"], length=atr_period_risk
            )
            # Use .ffill() as per deprecation warning, then fill remaining with 0
            df_out["atr_risk"] = atr_series.ffill().fillna(0)
            # Normalized ATR
            close_safe = (
                df_out["close"].replace(0, np.nan).ffill().fillna(1)
            )  # Avoid division by zero
            df_out["norm_atr"] = (
                ((df_out["atr_risk"] / close_safe) * 100).ffill().fillna(0)
            )
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
            rsi_series = ta.rsi(df_out[price_source], length=rsi_period)
            df_out["rsi"] = rsi_series  # Handle NaNs later in bulk fill
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating RSI: {e}",
                exc_info=False,
            )
            df_out["rsi"] = np.nan

        try:
            vol_period = strategy_cfg.get("VOLUME_SMA_PERIOD", 20)
            df_out["volume_sma"] = ta.sma(
                df_out["volume"], length=vol_period
            )  # Handle NaNs later
            min_periods_vol = max(1, vol_period // 2)
            df_out["volume_percentile_75"] = (
                df_out["volume"]
                .rolling(window=vol_period, min_periods=min_periods_vol)
                .quantile(0.75)
            )  # Handle NaNs later
            # High volume check will be done after NaN fill
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
            adx_df = ta.adx(
                df_out["high"], df_out["low"], df_out["close"], length=adx_period
            )
            adx_col_name = f"ADX_{adx_period}"
            if (
                adx_df is not None
                and not adx_df.empty
                and adx_col_name in adx_df.columns
            ):
                df_out["adx"] = adx_df[adx_col_name]  # Handle NaNs later
            else:
                log_console(
                    logging.WARNING,
                    f"Indicator Calc: ADX ({adx_col_name}) calculation failed or returned invalid DataFrame.",
                )
                df_out["adx"] = np.nan
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating ADX: {e}",
                exc_info=False,
            )
            df_out["adx"] = np.nan

        try:
            bb_period = strategy_cfg.get("BBANDS_PERIOD", 20)
            bbands = ta.bbands(df_out["close"], length=bb_period, std=2)
            bbu_col = f"BBU_{bb_period}_2.0"
            bbl_col = f"BBL_{bb_period}_2.0"
            bbm_col = f"BBM_{bb_period}_2.0"
            if (
                bbands is not None
                and not bbands.empty
                and all(c in bbands.columns for c in [bbu_col, bbl_col, bbm_col])
            ):
                # Calculate width safely, handle NaNs and potential division by zero later
                bbm_safe = bbands[bbm_col].replace(0, np.nan)
                df_out["bb_width"] = (
                    (bbands[bbu_col] - bbands[bbl_col]) / bbm_safe * 100
                )
            else:
                log_console(
                    logging.WARNING,
                    "Indicator Calc: Bollinger Bands calculation failed or missing columns.",
                )
                df_out["bb_width"] = np.nan
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Indicator Calc: Error calculating Bollinger Bands: {e}",
                exc_info=False,
            )
            df_out["bb_width"] = np.nan

        # --- Fill NaNs in Key Numeric Columns BEFORE calculating signals ---
        # Forward fill first, then fill remaining (start of series) with appropriate defaults
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
            "trendline": 0.0,
        }
        numeric_cols_to_fill.update(
            {f"{smoother_prefix}_{ma}": 0.0 for ma in ["ultra", "fast", "mid", "slow"]}
        )

        for col, fill_value in numeric_cols_to_fill.items():
            if col in df_out.columns:
                # Ensure column exists before filling
                df_out[col] = df_out[col].ffill().fillna(fill_value)
            # else: # Column might be missing due to calc error, already logged
            #     pass

        # Calculate high_volume after filling volume_sma and percentile
        df_out["high_volume"] = (
            df_out["volume"].fillna(0) > df_out["volume_percentile_75"]
        ) & (df_out["volume"].fillna(0) > df_out["volume_sma"])
        df_out["high_volume"] = (
            df_out["high_volume"].fillna(False).astype(bool)
        )  # Ensure boolean

        # --- Dynamic Thresholds based on Volatility ---
        atr_period_volatility = strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14)
        min_periods_vol_atr = max(1, atr_period_volatility // 2)
        volatility_factor = (
            df_out["norm_atr"]
            .rolling(window=atr_period_volatility, min_periods=min_periods_vol_atr)
            .mean()
            .ffill()
            .fillna(1.0)
        )
        volatility_factor = np.clip(volatility_factor, 0.5, 2.0)  # Cap factor

        rsi_base_low = strategy_cfg.get("RSI_LOW_THRESHOLD", 40)
        rsi_base_high = strategy_cfg.get("RSI_HIGH_THRESHOLD", 75)
        rsi_low = pd.Series(
            np.clip(rsi_base_low - (5 * volatility_factor), 10, 50), index=df_out.index
        )
        rsi_high = pd.Series(
            np.clip(rsi_base_high + (5 * volatility_factor), 60, 90), index=df_out.index
        )

        roc_base_threshold = strategy_cfg.get("MOM_THRESHOLD_PERCENT", 0.1)
        roc_threshold = pd.Series(
            abs(roc_base_threshold) * volatility_factor, index=df_out.index
        ).fillna(abs(roc_base_threshold))

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
            trendline_shift = (
                df_out[trendline_col].shift(1).ffill().fillna(df_out[trendline_col])
            )  # Fill initial shift NaN

            trend_up_cond = (
                (df_out[fast_ma_col] > df_out[mid_ma_col])
                & (df_out[mid_ma_col] > df_out[slow_ma_col])
                & (df_out[trendline_col] > trendline_shift)
            )
            df_out["trend_up"] = trend_up_cond.astype(bool)

            trend_down_cond = (
                (df_out[fast_ma_col] < df_out[mid_ma_col])
                & (df_out[mid_ma_col] < df_out[slow_ma_col])
                & (df_out[trendline_col] < trendline_shift)
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
            trendline_shift = (
                df_out[trendline_col].shift(1).ffill().fillna(df_out[trendline_col])
            )

            # Conditions (NaNs already handled in source columns)
            long_cond_trend = df_out[fast_ma_col] > df_out[mid_ma_col]
            long_cond_trigger = df_out[ultra_fast_ma_col] > df_out[fast_ma_col]
            long_cond_mom = df_out[roc_smooth_col] > roc_threshold
            long_cond_rsi = (df_out[rsi_col] > rsi_low) & (df_out[rsi_col] < rsi_high)
            long_cond_vol = df_out[high_volume_col]
            long_cond_adx = df_out[adx_col] > adx_threshold
            long_cond_itrend = df_out[trendline_col] > trendline_shift

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
            short_cond_rsi = (df_out[rsi_col] > rsi_low) & (df_out[rsi_col] < rsi_high)
            short_cond_vol = df_out[high_volume_col]
            short_cond_adx = df_out[adx_col] > adx_threshold
            short_cond_itrend = df_out[trendline_col] < trendline_shift

            df_out["short_signal"] = (
                short_cond_trend
                & short_cond_trigger
                & short_cond_mom
                & short_cond_rsi
                & short_cond_vol
                & short_cond_adx
                & short_cond_itrend
            ).astype(bool)

            # Signal Strength
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
            df_out["long_signal_strength"] = sum(
                cond.astype(int) for cond in long_conditions
            ) / len(long_conditions)
            df_out["short_signal_strength"] = sum(
                cond.astype(int) for cond in short_conditions
            ) / len(short_conditions)

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
            trendline_shift = (
                df_out[trendline_col].shift(1).ffill().fillna(df_out[trendline_col])
            )
            exit_long_cond1 = df_out[fast_ma_col] < df_out[mid_ma_col]
            exit_long_cond2 = df_out[roc_smooth_col] < 0
            exit_long_cond3 = df_out[trendline_col] < trendline_shift
            df_out["exit_long_signal"] = (
                exit_long_cond1 | exit_long_cond2 | exit_long_cond3
            ).astype(bool)

            exit_short_cond1 = df_out[fast_ma_col] > df_out[mid_ma_col]
            exit_short_cond2 = df_out[roc_smooth_col] > 0
            exit_short_cond3 = df_out[trendline_col] > trendline_shift
            df_out["exit_short_signal"] = (
                exit_short_cond1 | exit_short_cond2 | exit_short_cond3
            ).astype(bool)

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
                df_out[col] = df_out[col].astype(bool)

        # Fill NaNs in strength signals (shouldn't occur if sources were filled, but safeguard)
        df_out["long_signal_strength"] = df_out["long_signal_strength"].fillna(0.0)
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
        # Check last two rows for NaNs in critical columns
        if len(df_out) >= 2 and df_out.iloc[-2:][final_check_cols].isnull().any().any():
            log_console(
                logging.ERROR,
                "Indicator Calc: Critical columns contain NaN in the last two rows after processing. Check calculations.",
            )
            log_console(
                logging.DEBUG, f"Last 2 rows tail:\n{df_out.iloc[-2:]}"
            )  # Log problematic rows
            return None
        elif (
            len(df_out) == 1 and df_out.iloc[-1:][final_check_cols].isnull().any().any()
        ):
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
def get_available_balance(
    session: HTTP, coin: str = "USDT", account_type: str = "UNIFIED"
) -> float:
    """
    Fetches the available balance for a specific coin from the Bybit wallet using V5 API.
    Handles UNIFIED account structure correctly.

    Args:
        session: Authenticated Bybit HTTP session object.
        coin: The coin symbol (e.g., "USDT").
        account_type: The account type ("UNIFIED", "CONTRACT", "SPOT"). MUST match account structure.

    Returns:
        The available balance as a float, or 0.0 if fetch fails, coin not found, or balance is zero/negative.
    """
    try:
        balance_info = session.get_wallet_balance(
            accountType=account_type.upper(), coin=coin
        )

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
            # Assumes UNIFIED account structure primarily. Others might differ slightly.
            if account_type.upper() == "UNIFIED":
                account_data = list_data[0] if list_data else {}
                coin_list = account_data.get("coin", [])
            elif account_type.upper() in ["CONTRACT", "SPOT"]:
                # For CONTRACT/SPOT, the 'list' itself might contain the coin balances directly
                coin_list = list_data  # Needs verification for SPOT, but likely similar
            else:
                log_console(
                    logging.WARNING,
                    f"Balance parsing logic not explicitly defined for account type: {account_type}. Attempting UNIFIED structure.",
                )
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
                    balance_str = coin_data.get("availableToWithdraw")
                    source = "availableToWithdraw"
                    if balance_str is None or balance_str == "":
                        balance_str = coin_data.get("availableBalance")
                        source = "availableBalance"

                    # Fallback to walletBalance if others are missing, log a debug message
                    if balance_str is None or balance_str == "":
                        balance_str = coin_data.get("walletBalance")
                        source = "walletBalance"
                        log_console(
                            logging.DEBUG,
                            f"'availableToWithdraw'/'availableBalance' empty for {coin}, using '{source}': {balance_str}",
                        )

                    # Handle if all relevant balance fields are empty/missing
                    if balance_str is None or balance_str == "":
                        log_console(
                            logging.WARNING,
                            f"Could not find a valid balance field for {coin}. Balance is 0.",
                        )
                        balance_str = "0"

                    try:
                        balance_float = float(balance_str)
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
            error_msg = (
                balance_info.get("retMsg", "Unknown error")
                if balance_info
                else "Empty response"
            )
            error_code = balance_info.get("retCode", "N/A")
            log_console(
                logging.ERROR,
                f"Failed to fetch wallet balance: {error_msg} (Code: {error_code})",
            )
            return 0.0

    except Exception as e:
        log_console(
            logging.ERROR, f"Exception fetching wallet balance: {e}", exc_info=True
        )
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
            logging.DEBUG, "Position Size Calc: Invalid input types. Returning size 0."
        )
        return 0.0
    if not isinstance(qty_precision, int) or qty_precision < 0:
        log_console(
            logging.ERROR,
            f"Position Size Calc: Invalid qty_precision ({qty_precision}). Returning size 0.",
        )
        return 0.0

    if (
        balance <= 0
        or risk_percent <= 0
        or entry_price <= 0
        or min_order_qty <= 0
        or qty_step_float <= 0
    ):
        # Use DEBUG as this might happen normally with zero balance
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Invalid input values. Balance={balance:.2f}, Risk%={risk_percent}, Entry={entry_price}, MinQty={min_order_qty}, QtyStep={qty_step_float}. Returning size 0.",
        )
        return 0.0
    # Prevent extremely small SL distance causing huge size or division errors
    # Use a small epsilon based on price scale or a fixed value
    min_sl_value = 1e-9
    if sl_distance_price < min_sl_value:
        log_console(
            logging.WARNING,
            f"Position Size Calc: Stop loss distance ({sl_distance_price}) is too small (< {min_sl_value}). Cannot calculate size reliably. Returning 0.",
        )
        return 0.0

    # Calculate risk amount in USDT
    risk_amount_usdt = balance * (risk_percent / 100.0)

    # Calculate initial position size
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
    if max_position_usdt is not None and max_position_usdt > 0:
        try:
            if entry_price <= 1e-9:
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
    # Use a small tolerance (e.g., 0.001%) to avoid floating point issues right at the minimum
    if position_size < min_order_qty * (1 - 1e-5):
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Calculated size {position_size:.{qty_precision + 4}f} is below minimum required {min_order_qty}. Returning size 0.",
        )
        return 0.0

    # 3. Quantity Step Constraint (Rounding DOWN)
    try:
        if qty_step_float <= 1e-9:
            log_console(
                logging.ERROR,
                f"Position Size Calc: Quantity step ({qty_step_float}) is zero or too small. Cannot apply step constraint.",
            )
            return 0.0
        # Use math.floor for explicit round down, then multiply by step
        position_size_adjusted = (
            math.floor(position_size / qty_step_float) * qty_step_float
        )
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
    if position_size < min_order_qty * (1 - 1e-5):
        log_console(
            logging.INFO,
            f"Position Size Calc: Final size {position_size:.{qty_precision + 4}f} is below minimum {min_order_qty} after rounding down to step. Returning size 0.",
        )
        return 0.0

    # 5. Final rounding to precision (mostly cosmetic after step rounding)
    # Use standard round for this final step
    final_size = round(position_size, qty_precision)

    # Ensure size is not effectively zero after all calculations
    if final_size < min_order_qty * (1 - 1e-5):
        log_console(
            logging.DEBUG,
            f"Position Size Calc: Final calculated size ({final_size}) is zero or negligible relative to min qty. Returning size 0.",
        )
        return 0.0

    log_console(
        logging.INFO,
        f"Position Size Calc: Balance={balance:.2f}, Risk%={risk_percent}, SL Dist={sl_distance_price:.{qty_precision}f}, Entry={entry_price:.{qty_precision}f} -> Final Size = {final_size:.{qty_precision}f}",
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
                leverage_int = int(leverage)  # Convert leverage safely
            except (ValueError, TypeError):
                log_console(
                    logging.ERROR,
                    f"Metrics: Invalid leverage value '{leverage}' for {symbol}. Using 1.",
                    symbol=symbol,
                )
                leverage_int = 1  # Use default 1 if conversion fails

            # Validate inputs more robustly
            if not all(
                [
                    isinstance(symbol, str) and symbol,
                    isinstance(entry_time, dt.datetime),
                    isinstance(exit_time, dt.datetime),
                    isinstance(side, str) and side in ["Buy", "Sell"],
                    isinstance(entry_price, (int, float)) and entry_price > 0,
                    isinstance(exit_price, (int, float)) and exit_price > 0,
                    isinstance(qty, (int, float)) and qty > 0,
                    isinstance(leverage_int, int) and leverage_int > 0,
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

            try:
                abs_qty = abs(qty)
                price_diff = (
                    exit_price - entry_price
                    if side == "Buy"
                    else entry_price - exit_price
                )
                # P&L for Linear Contracts (Quantity * Price Difference)
                gross_pnl = price_diff * abs_qty

                # Fees Estimate (Apply fee rate to traded value on entry AND exit)
                entry_value = abs_qty * entry_price
                exit_value = abs_qty * exit_price
                fees = (entry_value * self.fee_rate) + (exit_value * self.fee_rate)
                net_pnl = gross_pnl - fees

                is_win = net_pnl > 0
                self.total_trades += 1
                if is_win:
                    self.wins += 1
                else:
                    self.losses += 1
                self.total_pnl += net_pnl
                self.total_fees_paid += fees

                trade_duration_sec = max(0.0, (exit_time - entry_time).total_seconds())

                # Consistent precision for logging
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

    def log_summary(
        self, symbol: Optional[str] = None, force: bool = False, interval: int = 3600
    ):
        """Logs summary performance metrics if interval elapsed or forced."""
        with self.lock:
            current_time = dt.datetime.now(dt.timezone.utc)
            log_now = force or self.total_trades == 0

            if not log_now and self.last_summary_time:
                time_since_last = (
                    current_time - self.last_summary_time
                ).total_seconds()
                if time_since_last >= interval:
                    log_now = True

            if not log_now:
                return

            if self.total_trades > 0:
                win_rate = (
                    (self.wins / self.total_trades * 100)
                    if self.total_trades > 0
                    else 0.0
                )
                avg_pnl = (
                    self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0
                )
                profit_factor = 0.0
                total_gains = sum(t["net_pnl"] for t in self.trades if t["is_win"])
                total_losses = abs(
                    sum(t["net_pnl"] for t in self.trades if not t["is_win"])
                )
                if total_losses > 1e-9:
                    profit_factor = total_gains / total_losses
                elif total_gains > 1e-9:
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

        # Instrument details
        self.min_order_qty: float = 0.0
        self.qty_step: str = "1"
        self.qty_step_float: float = 1.0
        self.qty_precision: int = 0
        self.tick_size: str = "1"
        self.price_precision: int = 0

        # State variables
        self.last_exit_time: Optional[dt.datetime] = None
        self.kline_queue: Queue = Queue()
        self.kline_df: Optional[pd.DataFrame] = None
        self.higher_tf_cache: Optional[bool] = None
        self.higher_tf_cache_time: Optional[dt.datetime] = None
        self.current_trade: Optional[Dict[str, Any]] = (
            None  # Real or simulated position state
        )
        self.order_confirm_retries: int = self.bot_cfg.get("ORDER_CONFIRM_RETRIES", 3)
        self.order_confirm_delay: float = float(
            self.bot_cfg.get("ORDER_CONFIRM_DELAY_SECONDS", 2.0)
        )
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
                raise RuntimeError(
                    f"Failed to fetch essential instrument info for {self.symbol}."
                )
            if (
                not self.dry_run and self.category != "spot"
            ):  # Leverage/margin setup not applicable to spot
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
            raise RuntimeError(f"Trader Init Failed: {self.symbol}") from e

    def _fetch_instrument_info(self) -> bool:
        """Fetches instrument details (V5 API)."""
        log_console(logging.DEBUG, "Fetching instrument info...", symbol=self.symbol)
        try:
            info = self.session.get_instruments_info(
                category=self.category, symbol=self.symbol
            )

            if info and info.get("retCode") == 0:
                result_list = info.get("result", {}).get("list", [])
                if result_list:
                    instrument = result_list[0]
                    lot_size_filter = instrument.get("lotSizeFilter", {})
                    price_filter = instrument.get("priceFilter", {})

                    try:
                        self.min_order_qty = float(
                            lot_size_filter.get("minOrderQty", "0")
                        )
                        self.qty_step = lot_size_filter.get("qtyStep", "1")
                        self.qty_step_float = float(self.qty_step)
                        self.tick_size = price_filter.get("tickSize", "1")
                        tick_size_float = float(self.tick_size)

                        # Calculate precision safely
                        self.qty_precision = (
                            len(self.qty_step.split(".")[-1])
                            if "." in self.qty_step
                            and self.qty_step_float > 0
                            and self.qty_step.split(".")[-1] != "0"
                            else 0
                        )
                        self.price_precision = (
                            len(self.tick_size.split(".")[-1])
                            if "." in self.tick_size
                            and tick_size_float > 0
                            and self.tick_size.split(".")[-1] != "0"
                            else 0
                        )

                    except (ValueError, TypeError) as conv_e:
                        log_console(
                            logging.ERROR,
                            f"Failed to convert instrument filter values: {conv_e}. Filters: {lot_size_filter}, {price_filter}",
                            symbol=self.symbol,
                        )
                        return False

                    if (
                        self.min_order_qty <= 0
                        or self.qty_step_float <= 0
                        or tick_size_float <= 0
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
                        f"Qty Step={self.qty_step} (Float: {self.qty_step_float:.{self.qty_precision + 1}f}, Precision: {self.qty_precision}), "
                        f"Tick Size={self.tick_size} (Precision: {self.price_precision})",
                        symbol=self.symbol,
                    )
                    return True
                else:
                    log_console(
                        logging.ERROR,
                        "Instrument info fetch failed: Empty result list.",
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
                error_msg = (
                    response.get("retMsg", "N/A") if response else "Empty response"
                )
                error_code = response.get("retCode", "N/A")
                # V5 Error Code 110043: leverage not modified
                if (
                    error_code == 110043
                    or "leverage not modified" in error_msg.lower()
                    or "same leverage" in error_msg.lower()
                ):
                    log_console(
                        logging.INFO,
                        f"Leverage already set to {self.leverage}x or not modified (Code: {error_code}, Msg: '{error_msg}').",
                        symbol=self.symbol,
                    )
                else:
                    log_console(
                        logging.WARNING,
                        f"Failed to set leverage: {error_msg} (Code: {error_code}).",
                        symbol=self.symbol,
                    )

            # Add margin mode setting here if needed (requires careful checking of V5 parameters)

        except Exception as e:
            log_console(
                logging.WARNING,
                f"Exception during initial setup (leverage/margin): {e}",
                symbol=self.symbol,
                exc_info=True,
            )

    def _websocket_callback(self, message: Dict[str, Any]):
        """Callback for processing V5 WebSocket kline messages."""
        try:
            # log_console(logging.DEBUG, f"WS Received Raw: {message}", symbol=self.symbol) # Verbose
            if not isinstance(message, dict):
                return

            topic = message.get("topic")
            data_list = message.get("data")

            if not topic or not isinstance(data_list, list):
                return

            # V5 Topic format: kline.{interval}.{symbol}
            topic_parts = topic.split(".")
            if not (
                len(topic_parts) == 3
                and topic_parts[0] == "kline"
                and topic_parts[1] == self.timeframe
                and topic_parts[2] == self.symbol
            ):
                return  # Ignore messages for other topics/symbols/timeframes

            if not data_list:
                return

            for kline_raw in data_list:
                if not isinstance(kline_raw, dict):
                    continue

                is_confirmed = kline_raw.get("confirm", False)
                process_confirmed_only = self.bot_cfg.get(
                    "PROCESS_CONFIRMED_KLINE_ONLY", True
                )

                if not process_confirmed_only or (
                    process_confirmed_only and is_confirmed
                ):
                    try:
                        timestamp_ms = int(kline_raw["start"])
                        kline_processed = {
                            "timestamp": pd.to_datetime(
                                timestamp_ms, unit="ms", utc=True
                            ),
                            "open": float(kline_raw["open"]),
                            "high": float(kline_raw["high"]),
                            "low": float(kline_raw["low"]),
                            "close": float(kline_raw["close"]),
                            "volume": float(kline_raw["volume"]),
                            "turnover": float(kline_raw.get("turnover", 0.0)),
                        }
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
                f"Unexpected error in WebSocket callback: {e}",
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
        max_items_per_cycle = 100  # Limit processing per call

        while not self.kline_queue.empty() and processed_count < max_items_per_cycle:
            try:
                kline_dict = self.kline_queue.get_nowait()
                processed_count += 1

                new_row = pd.DataFrame([kline_dict]).set_index("timestamp")

                if self.kline_df is None or self.kline_df.empty:
                    log_console(
                        logging.DEBUG,
                        "kline_df empty, initializing from first WS kline.",
                        symbol=self.symbol,
                    )
                    self.kline_df = new_row
                    df_updated = True
                else:
                    if not isinstance(self.kline_df.index, pd.DatetimeIndex):
                        log_console(
                            logging.ERROR,
                            "Internal kline_df index not DatetimeIndex! Resetting DF.",
                            symbol=self.symbol,
                        )
                        self.kline_df = new_row
                        df_updated = True
                        continue

                    idx_to_check = new_row.index[0]
                    if idx_to_check in self.kline_df.index:
                        # Update existing candle
                        # Use combine_first to keep existing columns (indicators) while updating OHLCV
                        self.kline_df.update(
                            new_row
                        )  # More efficient update for existing rows
                        # log_console(logging.DEBUG, f"WS updated existing kline: {idx_to_check}", symbol=self.symbol) # Can be noisy
                        df_updated = True
                    else:
                        # Append new candle
                        self.kline_df = pd.concat([self.kline_df, new_row])
                        self.kline_df.sort_index(inplace=True)
                        log_console(
                            logging.DEBUG,
                            f"WS appended new kline: {idx_to_check}",
                            symbol=self.symbol,
                        )

                        max_rows = (
                            self.internal_cfg.get("KLINE_LIMIT", 150) + 50
                        )  # Increased buffer
                        if len(self.kline_df) > max_rows:
                            self.kline_df = self.kline_df.iloc[-max_rows:]
                        df_updated = True

            except Empty:
                break
            except Exception as e:
                log_console(
                    logging.ERROR,
                    f"Error processing kline queue item: {e}",
                    symbol=self.symbol,
                    exc_info=True,
                )

        # Recalculate indicators only if DF was updated
        if df_updated:
            log_console(
                logging.DEBUG,
                f"Recalculating indicators after processing {processed_count} WS item(s)...",
                symbol=self.symbol,
            )
            calculated_df = calculate_indicators_momentum(
                self.kline_df, self.strategy_cfg, self.config
            )
            if calculated_df is not None and not calculated_df.empty:
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

        # Return True only if DF updated AND indicators recalculated successfully
        return df_updated and recalculated_ok

    def get_ohlcv_rest(
        self, timeframe: Optional[str] = None, limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV via REST (V5), calculates indicators."""
        tf = timeframe or self.timeframe
        default_lim = self.internal_cfg.get("KLINE_LIMIT", 120) + 5
        # Cap limit at Bybit's max per request
        lim = min(limit or default_lim, MAX_KLINE_LIMIT_PER_REQUEST)

        log_console(
            logging.DEBUG,
            f"Fetching {lim} klines for timeframe {tf} via REST API...",
            symbol=self.symbol,
        )

        try:
            response = self.session.get_kline(
                category=self.category, symbol=self.symbol, interval=tf, limit=lim
            )

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
                    df = df.iloc[::-1].reset_index(drop=True)  # Oldest first

                    # Convert and Clean
                    try:
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
                    df.dropna(subset=["timestamp"], inplace=True)
                    if df.empty:
                        return None
                    df.set_index("timestamp", inplace=True)

                    ohlcv_cols = ["open", "high", "low", "close", "volume", "turnover"]
                    for col in ohlcv_cols:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

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
                        return None

                    # Check Minimum Length
                    min_len_for_calc = self.internal_cfg.get("KLINE_LIMIT", 2)
                    if len(df) < min_len_for_calc:
                        log_console(
                            logging.WARNING,
                            f"REST KLine: Not enough valid data ({len(df)} < {min_len_for_calc}) for calc.",
                            symbol=self.symbol,
                        )
                        return None

                    log_console(
                        logging.DEBUG,
                        f"REST Fetch successful. Cleaned data shape: {df.shape}",
                        symbol=self.symbol,
                    )

                    # Calculate Indicators
                    df_with_indicators = calculate_indicators_momentum(
                        df, self.strategy_cfg, self.config
                    )
                    if df_with_indicators is None or df_with_indicators.empty:
                        log_console(
                            logging.WARNING,
                            "REST KLine: Indicator calculation failed.",
                            symbol=self.symbol,
                        )
                        return None
                    else:
                        self.last_kline_update_time = dt.datetime.now(dt.timezone.utc)
                        return df_with_indicators

                else:  # Empty kline_list
                    log_console(
                        logging.WARNING,
                        "REST KLine fetch returned empty list.",
                        symbol=self.symbol,
                    )
                    return None
            else:  # API error
                error_msg = (
                    response.get("retMsg", "N/A") if response else "Empty response"
                )
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
                return None
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Exception during REST KLine fetch: {e}",
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
            needs_rest_fetch = True  # Always fetch via REST if WS disabled
            log_console(
                logging.DEBUG,
                "WS disabled, using REST for kline data.",
                symbol=self.symbol,
            )
        elif self.use_websocket and not ws_processed_and_recalculated:
            # WS enabled, but queue empty OR processing/recalc failed. Check staleness.
            staleness_limit = self.bot_cfg.get("KLINE_STALENESS_SECONDS", 300)
            now_utc = dt.datetime.now(dt.timezone.utc)
            if (
                self.last_kline_update_time is None
                or (now_utc - self.last_kline_update_time).total_seconds()
                > staleness_limit
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
            if (
                rest_df_with_indicators is not None
                and not rest_df_with_indicators.empty
            ):
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

        # Return a copy of the current state
        return self.kline_df.copy() if self.kline_df is not None else None

    def get_position(self, log_error: bool = True) -> Optional[Dict[str, Any]]:
        """Fetches current position (V5 API), returns simulated state in dry run."""
        if self.dry_run:
            return self.current_trade  # Returns None if no simulated position

        try:
            response = self.session.get_positions(
                category=self.category, symbol=self.symbol
            )

            if response and response.get("retCode") == 0:
                position_list = response.get("result", {}).get("list", [])
                for pos in position_list:
                    if not isinstance(pos, dict):
                        continue
                    if pos.get("symbol") == self.symbol:
                        try:
                            pos_side = pos.get("side")  # "Buy", "Sell", or "None"
                            pos_size_str = pos.get("size", "0")
                            pos_size = float(pos_size_str)

                            # Check if position exists (significant size and valid side)
                            min_significant_qty = self.min_order_qty * 0.01
                            if (
                                pos_side in ["Buy", "Sell"]
                                and abs(pos_size) > min_significant_qty
                            ):
                                pos["size"] = pos_size  # Store as float
                                pos["avgPrice"] = float(pos.get("avgPrice", "0"))
                                pos["side"] = pos_side
                                pos["leverage"] = pos.get(
                                    "leverage", str(self.leverage)
                                )  # Store as string

                                # Add entry time from internal state if available
                                if (
                                    self.current_trade
                                    and self.current_trade.get("symbol") == self.symbol
                                ):
                                    pos["entry_time"] = self.current_trade.get(
                                        "entry_time"
                                    )
                                elif pos.get(
                                    "createdTime"
                                ):  # Try createdTime from API (ms string)
                                    try:
                                        pos["entry_time"] = pd.to_datetime(
                                            int(pos["createdTime"]), unit="ms", utc=True
                                        )
                                    except:
                                        pass  # Ignore parsing errors

                                return pos
                        except (ValueError, TypeError) as conv_e:
                            log_console(
                                logging.ERROR,
                                f"Position fetch: Error converting fields: {conv_e}. PosData: {pos}",
                                symbol=self.symbol,
                            )
                            continue

                return None  # No active position found for symbol
            else:
                if log_error:
                    error_msg = (
                        response.get("retMsg", "N/A") if response else "Empty response"
                    )
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
                    f"Exception during position fetch: {e}",
                    symbol=self.symbol,
                    exc_info=True,
                )
            return None

    def confirm_order(
        self, order_id: str, expected_qty: float
    ) -> Tuple[float, Optional[float]]:
        """Confirms order status by polling history (V5 API), returns filled qty and avg price."""
        if not order_id:
            log_console(
                logging.ERROR, "Confirm Order: Invalid Order ID.", symbol=self.symbol
            )
            return 0.0, None

        if self.dry_run:
            log_console(
                logging.INFO,
                f"[DRY RUN] Simulating confirmation for order {order_id}",
                symbol=self.symbol,
            )
            sim_entry_price: Optional[float] = None
            if self.kline_df is not None and not self.kline_df.empty:
                try:
                    sim_entry_price = float(self.kline_df["close"].iloc[-1])
                except:
                    pass
            if sim_entry_price is None or sim_entry_price <= 0:
                sim_entry_price = 1.0  # Dummy price
                log_console(
                    logging.WARNING,
                    f"[DRY RUN] Cannot estimate entry price, using dummy {sim_entry_price}.",
                    symbol=self.symbol,
                )
            log_console(
                logging.DEBUG,
                f"[DRY RUN] Using simulated avg fill price: {sim_entry_price}",
                symbol=self.symbol,
            )
            return expected_qty, sim_entry_price

        # Real Order Confirmation
        last_filled_qty = 0.0
        last_avg_price = None
        last_status = "Unknown"

        for attempt in range(self.order_confirm_retries + 1):
            if attempt > 0:
                time.sleep(self.order_confirm_delay)
            # else: log_console(logging.DEBUG, f"Confirming order {order_id} (Attempt 1)...", symbol=self.symbol) # Noisy

            try:
                # Use get_order_history for V5
                response = self.session.get_order_history(
                    category=self.category,
                    # symbol=self.symbol, # Optional: may speed up lookup
                    orderId=order_id,
                    limit=1,
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
                            filled_qty = (
                                float(filled_qty_str) if filled_qty_str else 0.0
                            )
                            avg_price_str = order.get("avgPrice", "0")
                            # Use avgPrice only if valid positive number string
                            if avg_price_str and float(avg_price_str) > 1e-9:
                                avg_price = float(avg_price_str)

                            last_filled_qty = filled_qty
                            if avg_price is not None:
                                last_avg_price = avg_price

                        except (ValueError, TypeError) as e:
                            log_console(
                                logging.ERROR,
                                f"Confirm Order {order_id}: Invalid qty/price format: {e}. Data: {order}",
                                symbol=self.symbol,
                            )

                        # V5 Statuses: Filled, PartiallyFilled, New, Untriggered, Rejected, Cancelled, PartiallyFilledCanceled, Deactivated, Expired, Active, Triggered
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
                            return filled_qty, last_avg_price  # Return what was filled
                        else:
                            log_console(
                                logging.WARNING,
                                f"Order {order_id} unhandled status: '{status}'. Continuing check.",
                                symbol=self.symbol,
                            )

                    else:  # Order not found in history yet
                        log_console(
                            logging.DEBUG,
                            f"Order {order_id} not in history yet (Attempt {attempt + 1}).",
                            symbol=self.symbol,
                        )
                else:  # API call failed
                    error_msg = (
                        response.get("retMsg", "N/A") if response else "Empty response"
                    )
                    error_code = response.get("retCode", "N/A")
                    log_console(
                        logging.WARNING,
                        f"Order history fetch failed for {order_id} (Attempt {attempt + 1}): {error_msg} (Code: {error_code}).",
                        symbol=self.symbol,
                    )
            except Exception as e:
                log_console(
                    logging.ERROR,
                    f"Exception during order confirmation attempt {attempt + 1} for {order_id}: {e}",
                    symbol=self.symbol,
                    exc_info=True,
                )

        # Loop finished without confirmation
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
            log_console(
                logging.ERROR, f"Invalid order side: {side}", symbol=self.symbol
            )
            return None
        if not isinstance(qty, (float, int)) or qty <= 0:
            log_console(
                logging.ERROR,
                f"Order qty must be positive, got: {qty}",
                symbol=self.symbol,
            )
            return None
        if qty < self.min_order_qty * (1 - 1e-5):
            log_console(
                logging.ERROR,
                f"Order qty {qty:.{self.qty_precision}f} is below minimum {self.min_order_qty}",
                symbol=self.symbol,
            )
            return None

        qty_str = f"{qty:.{self.qty_precision}f}"
        sl_str = None
        if stop_loss_price is not None:
            if not isinstance(stop_loss_price, (float, int)) or stop_loss_price <= 0:
                log_console(
                    logging.WARNING,
                    f"Invalid SL price ({stop_loss_price}). SL not set.",
                    symbol=self.symbol,
                )
            else:
                sl_str = f"{stop_loss_price:.{self.price_precision}f}"

        tp_str = None
        if take_profit_price is not None:
            if (
                not isinstance(take_profit_price, (float, int))
                or take_profit_price <= 0
            ):
                log_console(
                    logging.WARNING,
                    f"Invalid TP price ({take_profit_price}). TP not set.",
                    symbol=self.symbol,
                )
            else:
                tp_str = f"{take_profit_price:.{self.price_precision}f}"

        log_prefix = (
            f"{Fore.YELLOW + Style.BRIGHT}[DRY RUN]{Style.RESET_ALL} "
            if self.dry_run
            else ""
        )
        side_color = Fore.GREEN if side == "Buy" else Fore.RED
        log_msg = f"{log_prefix}{side_color}{Style.BRIGHT}{side.upper()} MARKET ORDER:{Style.RESET_ALL} Qty={qty_str}"
        if sl_str:
            log_msg += f", SL={sl_str}"
        if tp_str:
            log_msg += f", TP={tp_str}"
        action_prefix = (
            f"{Fore.MAGENTA + Style.BRIGHT}ACTION:{Style.RESET_ALL} [{self.symbol}] "
        )
        print(action_prefix + log_msg)
        log_console(
            logging.INFO, log_msg.replace(log_prefix, "").strip(), symbol=self.symbol
        )

        if self.dry_run:
            order_id = f"dry_run_{side.lower()}_{self.symbol}_{int(time.time() * 1000)}"
            sim_entry_price = 0.0
            if self.kline_df is not None and not self.kline_df.empty:
                try:
                    sim_entry_price = float(self.kline_df["close"].iloc[-1])
                except:
                    pass
            if sim_entry_price <= 0:
                sim_entry_price = 1.0  # Dummy

            self.current_trade = {
                "orderId": order_id,
                "symbol": self.symbol,
                "side": side,
                "size": qty,
                "avgPrice": sim_entry_price,
                "entry_time": dt.datetime.now(dt.timezone.utc),
                "stopLoss": sl_str,
                "takeProfit": tp_str,
                "leverage": str(self.leverage),
                "orderStatus": "Filled",
                "positionValue": f"{qty * sim_entry_price:.4f}",
            }
            log_console(
                logging.INFO,
                f"[DRY RUN] Simulated order {order_id}, Sim Entry: {sim_entry_price:.{self.price_precision}f}",
                symbol=self.symbol,
            )
            return order_id

        # Real Order Placement (V5)
        try:
            order_params: Dict[str, Any] = {
                "category": self.category,
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_str,
            }
            if sl_str:
                order_params["stopLoss"] = sl_str
                order_params["slTriggerBy"] = self.strategy_cfg.get(
                    "SL_TRIGGER_BY", "LastPrice"
                )
            if tp_str:
                order_params["takeProfit"] = tp_str
                order_params["tpTriggerBy"] = self.strategy_cfg.get(
                    "TP_TRIGGER_BY", "LastPrice"
                )
            if sl_str or tp_str:
                order_params["tpslMode"] = self.strategy_cfg.get(
                    "TPSL_MODE", "Full"
                )  # "Full" or "Partial"

            order_params["orderLinkId"] = (
                order_link_id or f"momscan_{self.symbol}_{int(time.time() * 1000)}"
            )

            response = self.session.place_order(**order_params)

            if response and response.get("retCode") == 0:
                order_id = response.get("result", {}).get("orderId")
                if order_id:
                    log_console(
                        logging.INFO,
                        f"Order {order_id} submitted. Confirming execution...",
                        symbol=self.symbol,
                    )
                    filled_qty, avg_price = self.confirm_order(order_id, qty)

                    min_fill_ratio = 0.95  # Require 95% fill
                    if filled_qty >= (qty * min_fill_ratio):
                        log_console(
                            logging.INFO,
                            f"Order {order_id} confirmed filled. Qty={filled_qty:.{self.qty_precision}f}, AvgPrice={avg_price or 'N/A'}",
                            symbol=self.symbol,
                        )
                        self.current_trade = {
                            "orderId": order_id,
                            "symbol": self.symbol,
                            "side": side,
                            "size": filled_qty,
                            "avgPrice": avg_price if avg_price else 0.0,
                            "entry_time": dt.datetime.now(dt.timezone.utc),
                            "stopLoss": sl_str,
                            "takeProfit": tp_str,
                            "leverage": str(self.leverage),
                            "orderStatus": "Filled",
                        }
                        return order_id
                    else:
                        log_console(
                            logging.ERROR,
                            f"Order {order_id} filled qty ({filled_qty:.{self.qty_precision}f}) < required ({qty * min_fill_ratio:.{self.qty_precision}f}). Fill failed/incomplete.",
                            symbol=self.symbol,
                        )
                        self.current_trade = None
                        # Attempt to cancel the potentially partially filled order? Risky. Manual check advised.
                        return None
                else:
                    log_console(
                        logging.ERROR,
                        f"Order placement succeeded (retCode 0) but no Order ID returned. Response: {response}",
                        symbol=self.symbol,
                    )
                    return None
            else:
                error_msg = (
                    response.get("retMsg", "N/A") if response else "Empty response"
                )
                error_code = response.get("retCode", "N/A")
                log_console(
                    logging.ERROR,
                    f"Order placement failed: {error_msg} (Code: {error_code})",
                    symbol=self.symbol,
                )
                # V5 Error Codes: 110007 (balance), 110013/14 (qty), 110017 (risk), 110040 (risk limit), 110073 (tp/sl)
                if error_code == 110007:
                    log_console(
                        logging.CRITICAL,
                        "ORDER REJECTED: INSUFFICIENT BALANCE!",
                        symbol=self.symbol,
                    )
                elif error_code in [110013, 110014]:
                    log_console(
                        logging.ERROR, "ORDER REJECTED: Qty issue.", symbol=self.symbol
                    )
                elif error_code == 110017:
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: Risk control.",
                        symbol=self.symbol,
                    )
                elif error_code == 110040:
                    log_console(
                        logging.ERROR, "ORDER REJECTED: Risk limit.", symbol=self.symbol
                    )
                elif error_code == 110073:
                    log_console(
                        logging.ERROR,
                        "ORDER REJECTED: TP/SL invalid.",
                        symbol=self.symbol,
                    )
                return None
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Exception during order placement: {e}",
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

        side = position_data.get("side")
        size = position_data.get("size")  # float
        entry_price = position_data.get("avgPrice")  # float
        entry_time = position_data.get("entry_time")  # datetime or None
        leverage = position_data.get("leverage", str(self.leverage))  # str

        if not (
            side in ["Buy", "Sell"]
            and isinstance(size, float)
            and abs(size) > 1e-9
            and isinstance(entry_price, float)
            and entry_price > 0
        ):
            log_console(
                logging.ERROR,
                f"Invalid position data for closing: Side={side}, Size={size}, Entry={entry_price}. Cannot close.",
                symbol=self.symbol,
            )
            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                log_console(
                    logging.WARNING,
                    "Clearing inconsistent internal trade state.",
                    symbol=self.symbol,
                )
                self.current_trade = None
            return False

        abs_size = abs(size)
        if abs_size < self.min_order_qty * 0.01:
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

        log_prefix = (
            f"{Fore.YELLOW + Style.BRIGHT}[DRY RUN]{Style.RESET_ALL} "
            if self.dry_run
            else ""
        )
        side_color = Fore.RED if close_side == "Sell" else Fore.GREEN
        log_msg = (
            f"{log_prefix}{side_color}{Style.BRIGHT}CLOSE {side.upper()} POSITION ({close_side} MARKET):"
            f"{Style.RESET_ALL} Qty={qty_str} | Reason: {exit_reason}"
        )
        action_prefix = (
            f"{Fore.MAGENTA + Style.BRIGHT}ACTION:{Style.RESET_ALL} [{self.symbol}] "
        )
        print(action_prefix + log_msg)
        log_console(
            logging.INFO, log_msg.replace(log_prefix, "").strip(), symbol=self.symbol
        )

        if self.dry_run:
            if self.current_trade and self.current_trade.get("symbol") == self.symbol:
                sim_exit_price = exit_price_est
                if sim_exit_price is None or sim_exit_price <= 0:
                    if self.kline_df is not None and not self.kline_df.empty:
                        try:
                            sim_exit_price = float(self.kline_df["close"].iloc[-1])
                        except:
                            pass
                if sim_exit_price is None or sim_exit_price <= 0:
                    sim_exit_price = self.current_trade.get(
                        "avgPrice", 1.0
                    )  # Fallback entry
                    log_console(
                        logging.WARNING,
                        f"[DRY RUN] Cannot estimate exit price, using entry ({sim_exit_price}). P&L inaccurate.",
                        symbol=self.symbol,
                    )

                log_console(
                    logging.INFO,
                    f"[DRY RUN] Simulating close at {sim_exit_price:.{self.price_precision}f}",
                    symbol=self.symbol,
                )
                entry_time_sim = self.current_trade.get("entry_time") or (
                    dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)
                )
                sim_leverage = self.current_trade.get("leverage", str(self.leverage))
                sim_entry_price = self.current_trade.get("avgPrice", sim_exit_price)
                sim_qty = self.current_trade.get("size", abs_size)
                original_side = self.current_trade.get("side", side)

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
                self.current_trade = None
                self.last_exit_time = dt.datetime.now(dt.timezone.utc)
                return True
            else:
                log_console(
                    logging.WARNING,
                    "[DRY RUN] Close attempt failed: No simulated trade open.",
                    symbol=self.symbol,
                )
                self.current_trade = None
                return False

        # Real Position Close (V5)
        try:
            close_order_link_id = f"close_{self.symbol}_{int(time.time() * 1000)}"
            response = self.session.place_order(
                category=self.category,
                symbol=self.symbol,
                side=close_side,
                orderType="Market",
                qty=qty_str,
                reduceOnly=True,
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
                    filled_qty, avg_exit_price = self.confirm_order(order_id, abs_size)

                    min_close_fill_ratio = 0.99
                    if filled_qty >= abs_size * min_close_fill_ratio:
                        log_console(
                            logging.INFO,
                            f"Close order {order_id} confirmed. Filled: {filled_qty:.{self.qty_precision}f}, AvgExit: {avg_exit_price or 'N/A'}",
                            symbol=self.symbol,
                        )

                        # Optional: Query position again to double-check closure
                        final_pos_check = self.get_position(log_error=False)
                        if final_pos_check is not None:
                            final_pos_size = final_pos_check.get("size", 0.0)
                            if abs(final_pos_size) > self.min_order_qty * 0.01:
                                log_console(
                                    logging.WARNING,
                                    f"Position check after close shows remaining size: {final_pos_size:.{self.qty_precision}f}. Close incomplete?",
                                    symbol=self.symbol,
                                )
                                return False  # Indicate potentially incomplete closure

                        # Log Metrics
                        exit_time = dt.datetime.now(dt.timezone.utc)
                        self.last_exit_time = exit_time
                        final_exit_price_log = (
                            avg_exit_price
                            if avg_exit_price is not None and avg_exit_price > 0
                            else exit_price_est
                        )
                        if final_exit_price_log is None or final_exit_price_log <= 0:
                            log_console(
                                logging.WARNING,
                                f"Cannot determine valid exit price for metrics. Using entry {entry_price}.",
                                symbol=self.symbol,
                            )
                            final_exit_price_log = entry_price

                        entry_time_log = entry_time
                        if not isinstance(entry_time_log, dt.datetime):
                            if (
                                self.current_trade
                                and self.current_trade.get("symbol") == self.symbol
                            ):
                                entry_time_log = self.current_trade.get("entry_time")
                            if not isinstance(entry_time_log, dt.datetime):
                                log_console(
                                    logging.WARNING,
                                    "Entry time missing for metrics. Using fallback.",
                                    symbol=self.symbol,
                                )
                                entry_time_log = exit_time - dt.timedelta(minutes=15)

                        self.metrics.add_trade(
                            symbol=self.symbol,
                            entry_time=entry_time_log,
                            exit_time=exit_time,
                            side=side,
                            entry_price=entry_price,
                            exit_price=final_exit_price_log,
                            qty=filled_qty,
                            leverage=leverage,
                        )
                        self.current_trade = None
                        return True
                    else:
                        log_console(
                            logging.ERROR,
                            f"Close order {order_id} filled qty ({filled_qty:.{self.qty_precision}f}) < expected ({abs_size:.{self.qty_precision}f}). Manual check!",
                            symbol=self.symbol,
                        )
                        return False
                else:
                    log_console(
                        logging.ERROR,
                        f"Close order succeeded (retCode 0) but no Order ID returned. {response}",
                        symbol=self.symbol,
                    )
                    return False
            else:
                error_msg = (
                    response.get("retMsg", "N/A") if response else "Empty response"
                )
                error_code = response.get("retCode", "N/A")
                # V5 Error Codes: 110025, 110066, 3400074 often indicate already closed/zero size
                pos_already_closed_codes = [110025, 110066, 3400074]
                already_closed_msg = (
                    "position size is zero" in error_msg.lower()
                    or "order qty exceeded position size" in error_msg.lower()
                    or "reduceonly" in error_msg.lower()
                )

                if error_code in pos_already_closed_codes or already_closed_msg:
                    log_console(
                        logging.WARNING,
                        f"Close attempt failed, API indicates already closed (Code: {error_code}, Msg: '{error_msg}'). Assuming closed.",
                        symbol=self.symbol,
                    )
                    if (
                        self.current_trade
                        and self.current_trade.get("symbol") == self.symbol
                    ):
                        exit_time = dt.datetime.now(dt.timezone.utc)
                        final_exit_price = (
                            exit_price_est
                            if exit_price_est is not None and exit_price_est > 0
                            else entry_price
                        )
                        entry_time_actual = self.current_trade.get("entry_time") or (
                            exit_time - dt.timedelta(minutes=15)
                        )
                        internal_pos_size = self.current_trade.get("size", abs_size)
                        internal_entry_price = self.current_trade.get(
                            "avgPrice", entry_price
                        )
                        internal_side = self.current_trade.get("side", side)
                        internal_leverage = self.current_trade.get("leverage", leverage)
                        log_console(
                            logging.INFO,
                            "Logging trade metrics from internal state (API showed already closed).",
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
                        self.current_trade = None
                    self.last_exit_time = dt.datetime.now(dt.timezone.utc)
                    return True
                else:
                    log_console(
                        logging.ERROR,
                        f"Position close order failed: {error_msg} (Code: {error_code})",
                        symbol=self.symbol,
                    )
                    return False
        except Exception as e:
            log_console(
                logging.ERROR,
                f"Exception during position close: {e}",
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
                "HTF same as base timeframe. Disabling MTF check.",
                symbol=self.symbol,
            )
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True

        htf_kline_limit = self.internal_cfg.get("KLINE_LIMIT", 120) + 20
        log_console(
            logging.INFO,
            f"Fetching HTF ({higher_tf}) data for trend analysis...",
            symbol=self.symbol,
        )
        df_higher = self.get_ohlcv_rest(timeframe=higher_tf, limit=htf_kline_limit)

        if df_higher is None or df_higher.empty or len(df_higher) < 2:
            log_console(
                logging.WARNING,
                f"Could not get HTF ({higher_tf}) data. Allowing trade (neutral).",
                symbol=self.symbol,
            )
            self.higher_tf_cache = True
            self.higher_tf_cache_time = now_utc
            return True  # Fail open

        try:
            use_ehlers_htf = self.strategy_cfg.get("USE_EHLERS_SMOOTHER", False)
            smoother_prefix_htf = "ss" if use_ehlers_htf else "ema"
            fast_ma_col = f"{smoother_prefix_htf}_fast"
            mid_ma_col = f"{smoother_prefix_htf}_mid"

            if (
                fast_ma_col not in df_higher.columns
                or mid_ma_col not in df_higher.columns
            ):
                log_console(
                    logging.ERROR,
                    f"Required MA cols ({fast_ma_col}, {mid_ma_col}) not found in HTF ({higher_tf}) data. Allowing trade.",
                    symbol=self.symbol,
                )
                self.higher_tf_cache = True
                self.higher_tf_cache_time = now_utc
                return True

            # Check trend on second-to-last closed candle (-2 index)
            candle_to_check = df_higher.iloc[-2]
            fast_ma_prev = candle_to_check.get(fast_ma_col)
            mid_ma_prev = candle_to_check.get(mid_ma_col)

            # Check for NaN after .get() - should be handled by indicator calc, but double check
            if pd.isna(fast_ma_prev) or pd.isna(mid_ma_prev):
                log_console(
                    logging.WARNING,
                    f"HTF ({higher_tf}) MAs are NaN on check candle ({candle_to_check.name}). Allowing trade.",
                    symbol=self.symbol,
                )
                trend_is_favorable = True
            else:
                # Allow trade only if HTF trend is bullish (Fast > Mid)
                trend_is_favorable = fast_ma_prev > mid_ma_prev
                trend_desc = (
                    "Bullish (Fast>Mid)"
                    if trend_is_favorable
                    else "Bearish (Fast<=Mid)"
                )
                log_console(
                    logging.INFO,
                    f"HTF ({higher_tf}) trend (@ {candle_to_check.name}): "
                    f"{fast_ma_col}={fast_ma_prev:.{self.price_precision}f}, {mid_ma_col}={mid_ma_prev:.{self.price_precision}f}. "
                    f"Status: {trend_desc} -> Allow Trade: {trend_is_favorable}",
                    symbol=self.symbol,
                )

            self.higher_tf_cache = trend_is_favorable
            self.higher_tf_cache_time = now_utc
            return trend_is_favorable

        except IndexError:
            log_console(
                logging.WARNING,
                f"Index error accessing HTF ({higher_tf}) data (need >= 2 rows). Allowing trade.",
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
        # Determine category based on account type
        if self.account_type == "SPOT":
            self.category = "spot"
        elif self.account_type == "UNIFIED":
            self.category = "linear"  # Default UNIFIED to linear
        elif self.account_type == "CONTRACT":
            self.category = "linear"  # Default CONTRACT to linear (can be inverse too)
        # Add specific logic for inverse if needed based on config or symbol suffix (e.g., BTCUSD)
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
        self.ws_lock = threading.Lock()

        # Traders and Metrics
        self.traders: Dict[str, SymbolTrader] = {}
        fee_rate = self.risk_cfg.get("FEE_RATE", 0.0006)
        self.metrics = TradeMetrics(fee_rate=fee_rate)

        log_console(
            logging.INFO,
            f"Initializing Trader (Testnet: {self.use_testnet}, Account: {self.account_type}, Category: {self.category})...",
        )
        try:
            recv_window = self.bot_cfg.get("API_RECV_WINDOW", 10000)
            self.session = HTTP(
                testnet=self.use_testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=recv_window,
            )
            self._test_api_connection()
            self._initialize_traders()  # Raises error if no traders succeed

            # Start WebSocket if needed and traders initialized
            if self.traders and any(
                trader.use_websocket for trader in self.traders.values()
            ):
                log_console(logging.INFO, "Starting WebSocket connection thread...")
                self.ws_connection_thread = threading.Thread(
                    target=self._start_and_monitor_websocket,
                    name="WebSocketMonitor",
                    daemon=True,
                )
                self.ws_connection_thread.start()
                time.sleep(5)  # Allow time for initial connection attempt
            else:
                log_console(
                    logging.INFO,
                    "WebSocket disabled or no traders. Using REST API only.",
                )

        except (ConnectionError, RuntimeError, ValueError) as init_e:
            log_console(
                logging.CRITICAL, f"Bot initialization failed: {init_e}", exc_info=False
            )
            self.running = False
            self.shutdown_requested = True
            self._cleanup_resources()  # Attempt cleanup
            raise init_e
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
                server_ts = int(time_nano_str) / 1e9
                server_dt = dt.datetime.fromtimestamp(server_ts, tz=dt.timezone.utc)
                local_dt_utc = dt.datetime.now(dt.timezone.utc)
                time_diff = abs((local_dt_utc - server_dt).total_seconds())
                if time_diff > 60:
                    log_console(
                        logging.WARNING,
                        f"System clock differs from server by {time_diff:.1f}s. Check time sync.",
                    )
                log_console(
                    logging.INFO,
                    f"API connection successful. Server Time: {server_dt.isoformat()}",
                )
            else:
                error_msg = (
                    server_time_resp.get("retMsg", "N/A")
                    if server_time_resp
                    else "Empty Response"
                )
                error_code = server_time_resp.get("retCode", "N/A")
                raise ConnectionError(
                    f"API test failed: {error_msg} (Code: {error_code})"
                )
        except ConnectionError as ce:
            log_console(
                logging.CRITICAL,
                f"API connection failed: {ce}. Check keys, permissions, network.",
            )
            raise ce
        except Exception as e:
            log_console(
                logging.CRITICAL, f"API test failed unexpectedly: {e}.", exc_info=True
            )
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

        for i, symbol_cfg in enumerate(deepcopy(symbols_cfg_list)):
            if not isinstance(symbol_cfg, dict):
                log_console(logging.ERROR, f"Symbol config #{i} invalid. Skipping.")
                continue
            symbol = symbol_cfg.get("SYMBOL")
            if not symbol or not isinstance(symbol, str):
                log_console(
                    logging.ERROR,
                    f"Symbol config #{i} missing/invalid 'SYMBOL'. Skipping.",
                )
                continue
            if symbol in temp_traders:
                log_console(
                    logging.WARNING,
                    f"Duplicate symbol {symbol} in config. Skipping.",
                    symbol=symbol,
                )
                continue

            log_console(
                logging.INFO, f"--- Initializing trader for {symbol} ---", symbol=symbol
            )
            try:
                if not self.session:
                    raise RuntimeError("HTTP session not available.")
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
                temp_traders[symbol] = trader
                successful_traders += 1
                log_console(
                    logging.INFO,
                    f"--- Trader {symbol} initialized successfully ---",
                    symbol=symbol,
                )
            except RuntimeError:
                log_console(
                    logging.ERROR,
                    f"Initialization failed for {symbol} (RuntimeError). Skipping.",
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
            raise RuntimeError("No traders could be initialized successfully.")

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
                f"Unsupported category '{self.category}' for WS. Defaulting to 'linear'.",
            )
            ws_channel_type = "linear"

        retry_delay = 10
        max_retry_delay = 120

        while not self.shutdown_requested:
            with self.ws_lock:
                ws_conn_active = self.ws is not None and self.ws_active

            if not ws_conn_active:
                try:
                    log_console(
                        logging.INFO,
                        f"Attempting WS connection (Channel: {ws_channel_type})...",
                    )
                    # Correct V5 WebSocket initialization
                    temp_ws = WebSocket(
                        testnet=self.use_testnet,
                        channel_type=ws_channel_type,
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        ping_interval=self.bot_cfg.get("WS_PING_INTERVAL", 20),
                        ping_timeout=self.bot_cfg.get("WS_PING_TIMEOUT", 10),
                        on_message=self._handle_ws_message,  # Generic handler
                        on_error=self._handle_ws_error,
                        on_close=self._handle_ws_close,
                        on_open=self._handle_ws_open,  # Callback sets ws_active=True
                        retries=0,  # Disable pybit's internal retries
                    )

                    topics_to_subscribe = []
                    for symbol, trader in self.traders.items():
                        if trader.use_websocket:
                            topic = f"kline.{trader.timeframe}.{symbol}"
                            topics_to_subscribe.append(topic)
                            # Pass trader's specific callback for kline data
                            temp_ws.subscribe(
                                topic, callback=trader._websocket_callback
                            )
                            log_console(
                                logging.DEBUG,
                                f"WS prepared subscription: {topic}",
                                symbol=symbol,
                            )

                    if topics_to_subscribe:
                        with self.ws_lock:
                            self.ws = temp_ws  # Assign the WS object
                        # Connection happens implicitly when WebSocket object interacts
                        log_console(
                            logging.INFO,
                            f"WebSocket connection initiated for {len(topics_to_subscribe)} streams.",
                        )
                        retry_delay = 10  # Reset delay on successful initiation attempt

                        # Keep monitor thread alive, check active flag periodically
                        while not self.shutdown_requested:
                            with self.ws_lock:
                                ws_still_active = self.ws_active
                            if not ws_still_active:
                                log_console(
                                    logging.INFO,
                                    "WS inactive detected by monitor, triggering reconnect.",
                                )
                                break  # Break inner loop to reconnect
                            time.sleep(5)  # Check interval
                        # Reconnect attempt happens in the outer loop
                    else:
                        log_console(
                            logging.INFO, "No symbols use WebSocket. WS thread exiting."
                        )
                        break  # Exit monitoring loop

                except Exception as e:
                    log_console(
                        logging.ERROR,
                        f"Failed to init/start WS connection: {e}. Retrying in {retry_delay}s.",
                        exc_info=False,
                    )
                    with self.ws_lock:
                        self.ws = None
                        self.ws_active = False
                    if not self.shutdown_requested:
                        time.sleep(retry_delay)
                    retry_delay = min(max_retry_delay, retry_delay * 2)

            else:  # WS is active, sleep and check later
                time.sleep(10)

        log_console(logging.INFO, "WebSocket monitoring thread finished.")
        self._close_websocket_connection()  # Ensure cleanup

    def _close_websocket_connection(self):
        """Safely closes the WebSocket connection object."""
        with self.ws_lock:
            if self.ws:
                log_console(logging.INFO, "Closing WebSocket connection object...")
                try:
                    self.ws.exit()
                except Exception as e:
                    log_console(
                        logging.ERROR, f"Error closing WS object: {e}", exc_info=False
                    )
                finally:
                    self.ws = None
                    self.ws_active = False
            self.ws_active = False  # Ensure flag is false

    # --- WebSocket Event Handlers ---
    def _handle_ws_open(self, ws_app):
        """WS connection opened."""
        log_console(logging.INFO, "WebSocket connection opened successfully.")
        with self.ws_lock:
            self.ws_active = True
        # Populate initial data after connection established
        self._populate_initial_kline_data_async()

    def _populate_initial_kline_data_async(self):
        """Starts a thread to fetch initial Kline data via REST post-WS connection."""
        log_console(
            logging.DEBUG, "Starting thread for initial REST kline population..."
        )
        thread = threading.Thread(target=self._fetch_initial_data_worker, daemon=True)
        thread.start()

    def _fetch_initial_data_worker(self):
        """Worker thread for fetching initial REST data."""
        time.sleep(2)  # Allow subscriptions to settle
        for symbol, trader in list(self.traders.items()):  # Iterate over copy
            if trader.use_websocket and (
                trader.kline_df is None or trader.kline_df.empty
            ):
                log_console(
                    logging.DEBUG,
                    f"Fetching initial REST data for {symbol}...",
                    symbol=symbol,
                )
                try:
                    initial_df = trader.get_ohlcv_rest()
                    if initial_df is not None and not initial_df.empty:
                        trader.kline_df = initial_df  # Update trader's internal DF
                        log_console(
                            logging.INFO,
                            f"Initialized kline data for {symbol} from REST.",
                            symbol=symbol,
                        )
                    else:
                        log_console(
                            logging.WARNING,
                            f"Failed initial REST fetch for {symbol}. Relying on WS updates.",
                            symbol=symbol,
                        )
                except Exception as e:
                    log_console(
                        logging.ERROR,
                        f"Error fetching initial REST data for {symbol}: {e}",
                        symbol=symbol,
                        exc_info=False,
                    )
                time.sleep(0.5)  # Throttle requests

    def _handle_ws_message(self, message):
        """Generic WS message handler (auth, subs responses)."""
        try:
            # log_console(logging.DEBUG, f"Generic WS Msg: {message}") # Very verbose
            if isinstance(message, dict):
                op = message.get("op")
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
                elif op == "subscribe":
                    success = message.get("success", False)
                    topic = message.get(
                        "ret_msg", message.get("req_id")
                    )  # Topic might be in ret_msg or req_id? Check docs.
                    if success:
                        log_console(logging.DEBUG, f"WS Subscribe Ack: {topic}")
                    else:
                        log_console(
                            logging.ERROR,
                            f"WS Subscribe Fail: {topic} - {message.get('ret_msg', 'N/A')}",
                        )
        except Exception as e:
            log_console(
                logging.ERROR, f"Error handling generic WS message: {e}", exc_info=True
            )

    def _handle_ws_error(self, error):
        """WS error callback."""
        log_console(
            logging.ERROR,
            f"WebSocket Error: {error}",
            exc_info=isinstance(error, Exception),
        )
        with self.ws_lock:
            self.ws_active = False  # Trigger reconnect

    def _handle_ws_close(self, ws_app):
        """WS connection closed callback."""
        log_console(logging.WARNING, "WebSocket connection closed.")
        with self.ws_lock:
            # Ensure state is updated, monitor loop will handle reconnect
            if self.ws_active or self.ws is not None:
                self.ws_active = False
                self.ws = None
                log_console(logging.INFO, "WS state set to inactive on close.")

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
        metrics_interval: int = int(
            self.bot_cfg.get("METRICS_LOG_INTERVAL_SECONDS", 3600)
        )
        last_balance_check_time: float = 0.0
        balance_check_interval: int = self.bot_cfg.get(
            "BALANCE_CHECK_INTERVAL_SECONDS", 300
        )
        balance: float = 0.0

        # Initial Balance Check
        if self.dry_run:
            balance = float(self.risk_cfg.get("DRY_RUN_DUMMY_BALANCE", 10000.0))
            log_console(logging.INFO, f"Using Dry Run Balance: {balance:.2f} USDT")
        else:
            balance = get_available_balance(self.session, "USDT", self.account_type)
            if balance > 0:
                log_console(
                    logging.INFO, f"Initial Balance Check: {balance:.2f} USDT Available"
                )
                last_balance_check_time = time.time()
            else:
                log_console(
                    logging.WARNING,
                    f"Initial {self.account_type} USDT balance zero or fetch failed. Trades cannot be sized.",
                )

        while self.running and not self.shutdown_requested:
            try:
                start_time = time.time()
                current_dt_utc = dt.datetime.now(dt.timezone.utc)
                print(
                    f"\n{Fore.BLUE}{Style.BRIGHT}===== CYCLE START: {current_dt_utc.isoformat()} ====="
                )

                # Pre-Cycle Checks
                with self.ws_lock:
                    ws_currently_active = self.ws_active
                if self.ws_connection_thread and not ws_currently_active:
                    log_console(
                        logging.WARNING,
                        "WebSocket connection inactive. Relying on REST API.",
                    )

                # Get Global State (Balance) - Periodically
                current_time = time.time()
                if not self.dry_run and (
                    current_time - last_balance_check_time > balance_check_interval
                ):
                    fetched_balance = get_available_balance(
                        self.session, "USDT", self.account_type
                    )
                    if abs(fetched_balance - balance) > 0.01:  # Log if changed
                        log_console(
                            logging.INFO,
                            f"Checked Balance: {fetched_balance:.2f} USDT Available",
                        )
                    balance = fetched_balance
                    last_balance_check_time = current_time
                    if balance <= 0:
                        log_console(
                            logging.WARNING,
                            f"{self.account_type} USDT balance zero. No trades can be sized.",
                        )

                # Balance Allocation
                num_active_traders = len(self.traders)
                balance_per_trader = (
                    balance / num_active_traders
                    if num_active_traders > 0 and balance > 0
                    else 0.0
                )
                max_pos_usdt_config = self.risk_cfg.get("MAX_POSITION_USDT")
                max_pos_usdt_per_trader: Optional[float] = None
                if (
                    isinstance(max_pos_usdt_config, (int, float))
                    and max_pos_usdt_config > 0
                    and num_active_traders > 0
                ):
                    max_pos_usdt_per_trader = max_pos_usdt_config / num_active_traders
                    log_console(
                        logging.DEBUG,
                        f"Allocated Bal/Trader: {balance_per_trader:.2f}, Max Pos Value/Trader: {max_pos_usdt_per_trader:.2f}",
                    )
                elif balance_per_trader > 0:
                    log_console(
                        logging.DEBUG,
                        f"Allocated Bal/Trader: {balance_per_trader:.2f}, Max Pos Value/Trader: Unlimited",
                    )

                # Iterate Through Symbols
                for symbol in list(self.traders.keys()):
                    if not self.running or self.shutdown_requested:
                        break
                    trader = self.traders[symbol]
                    symbol_color_prefix = (
                        f"{get_symbol_color(symbol)}[{symbol}]{Style.RESET_ALL}"
                    )
                    print(f"{symbol_color_prefix} --- Processing Symbol ---")

                    try:
                        # Cooldown Check
                        if trader.last_exit_time:
                            cooldown_seconds = trader.bot_cfg.get(
                                "COOLDOWN_PERIOD_SECONDS", 300
                            )
                            if cooldown_seconds > 0:
                                time_since_exit = (
                                    current_dt_utc - trader.last_exit_time
                                ).total_seconds()
                                if time_since_exit < cooldown_seconds:
                                    log_console(
                                        logging.INFO,
                                        f"Cooldown active ({cooldown_seconds - time_since_exit:.0f}s left).",
                                        symbol=symbol,
                                    )
                                    continue

                        # Get Latest Data (handles WS/REST/indicators)
                        df_with_indicators = trader.get_ohlcv()
                        if (
                            df_with_indicators is None
                            or df_with_indicators.empty
                            or len(df_with_indicators) < 2
                        ):
                            log_console(
                                logging.DEBUG,
                                "Insufficient data or calc failed. Skipping cycle.",
                                symbol=symbol,
                            )
                            continue

                        # Access candle data safely
                        try:
                            last_closed_candle = df_with_indicators.iloc[-2]
                            latest_candle = df_with_indicators.iloc[-1]
                        except IndexError:
                            log_console(
                                logging.WARNING,
                                f"Cannot access candle data (need >= 2 rows, got {len(df_with_indicators)}). Skipping.",
                                symbol=symbol,
                            )
                            continue

                        # Validate key data
                        critical_indicators = [
                            "close",
                            "atr_risk",
                            "long_signal",
                            "short_signal",
                            "exit_long_signal",
                            "exit_short_signal",
                        ]
                        if any(
                            pd.isna(last_closed_candle.get(ind))
                            for ind in critical_indicators
                        ):
                            log_console(
                                logging.WARNING,
                                f"Critical indicator NaN on closed candle ({last_closed_candle.name}). Skipping.",
                                symbol=symbol,
                            )
                            continue
                        current_price = latest_candle.get("close")
                        if pd.isnull(current_price) or current_price <= 0:
                            current_price = last_closed_candle.get("close")  # Fallback
                            if pd.isnull(current_price) or current_price <= 0:
                                log_console(
                                    logging.ERROR,
                                    "Cannot get valid current price. Skipping.",
                                    symbol=symbol,
                                )
                                continue

                        # Position Management
                        position = trader.get_position(log_error=True)

                        if position:
                            # --- Existing Position ---
                            pos_side = position.get("side")
                            pos_size = position.get("size")  # float
                            entry_price = position.get("avgPrice")  # float

                            if not (
                                pos_side in ["Buy", "Sell"]
                                and isinstance(pos_size, float)
                                and abs(pos_size) > 1e-9
                                and isinstance(entry_price, float)
                                and entry_price > 0
                            ):
                                log_console(
                                    logging.ERROR,
                                    f"Invalid position data retrieved: {position}. Clearing internal state.",
                                    symbol=symbol,
                                )
                                trader.current_trade = None
                                position = None
                                continue

                            pos_size_abs = abs(pos_size)
                            log_console(
                                logging.INFO,
                                f"Position: {pos_side}, Size: {pos_size_abs:.{trader.qty_precision}f}, Entry: {entry_price:.{trader.price_precision}f}",
                                symbol=symbol,
                            )

                            # Check Exit Conditions
                            exit_reason: Optional[str] = None
                            exit_long_sig = last_closed_candle.get(
                                "exit_long_signal", False
                            )
                            exit_short_sig = last_closed_candle.get(
                                "exit_short_signal", False
                            )
                            if pos_side == "Buy" and exit_long_sig:
                                exit_reason = "Strategy Exit (Long)"
                            elif pos_side == "Sell" and exit_short_sig:
                                exit_reason = "Strategy Exit (Short)"

                            # TODO: Add Trailing Stop logic here if needed (requires state)

                            if exit_reason:
                                log_console(
                                    logging.INFO,
                                    f"Exit condition: {exit_reason}. Closing position.",
                                    symbol=symbol,
                                )
                                if trader.close_position(
                                    position,
                                    exit_reason=exit_reason,
                                    exit_price_est=current_price,
                                ):
                                    log_console(
                                        logging.INFO,
                                        "Position closed successfully.",
                                        symbol=symbol,
                                    )
                                else:
                                    log_console(
                                        logging.ERROR,
                                        f"Failed to close position after exit signal '{exit_reason}'. Manual check!",
                                        symbol=symbol,
                                    )
                                continue  # Process next symbol after close attempt
                            else:
                                log_console(
                                    logging.DEBUG,
                                    "Holding position. No exit signal.",
                                    symbol=symbol,
                                )

                        else:  # No Position Open
                            # --- Check Entry ---
                            htf_allows_trade = trader.get_higher_tf_trend()
                            if not htf_allows_trade:
                                log_console(
                                    logging.INFO,
                                    "HTF filter unfavorable. Skipping entry check.",
                                    symbol=symbol,
                                )
                                continue  # Skip entry if HTF filter blocks

                            # Get signals/params from closed candle
                            atr_val = last_closed_candle.get("atr_risk", 0.0)
                            long_signal = last_closed_candle.get("long_signal", False)
                            short_signal = last_closed_candle.get("short_signal", False)
                            long_strength = last_closed_candle.get(
                                "long_signal_strength", 0.0
                            )
                            short_strength = last_closed_candle.get(
                                "short_signal_strength", 0.0
                            )

                            risk_percent = float(
                                trader.risk_cfg.get("RISK_PER_TRADE_PERCENT", 1.0)
                            )
                            sl_mult = float(
                                trader.risk_cfg.get("ATR_MULTIPLIER_SL", 1.5)
                            )
                            tp_mult = float(
                                trader.risk_cfg.get("ATR_MULTIPLIER_TP", 2.0)
                            )
                            min_strength = float(
                                trader.strategy_cfg.get("MIN_SIGNAL_STRENGTH", 0.5)
                            )

                            entry_side: Optional[str] = None
                            stop_loss_price: Optional[float] = None
                            take_profit_price: Optional[float] = None
                            signal_strength: float = 0.0
                            entry_price_est = current_price

                            if long_signal and long_strength >= min_strength:
                                entry_side = "Buy"
                                signal_strength = long_strength
                                if atr_val > 0 and sl_mult > 0:
                                    stop_loss_price = entry_price_est - (
                                        atr_val * sl_mult
                                    )
                                if atr_val > 0 and tp_mult > 0:
                                    take_profit_price = entry_price_est + (
                                        atr_val * tp_mult
                                    )
                            elif short_signal and short_strength >= min_strength:
                                entry_side = "Sell"
                                signal_strength = short_strength
                                if atr_val > 0 and sl_mult > 0:
                                    stop_loss_price = entry_price_est + (
                                        atr_val * sl_mult
                                    )
                                if atr_val > 0 and tp_mult > 0:
                                    take_profit_price = entry_price_est - (
                                        atr_val * tp_mult
                                    )

                            if entry_side:
                                log_console(
                                    logging.DEBUG,
                                    f"Entry Signal: Side={entry_side}, Str={signal_strength:.2f}, EstEntry={entry_price_est:.{trader.price_precision}f}",
                                    symbol=symbol,
                                )

                                # Validate SL/TP
                                if stop_loss_price is None or stop_loss_price <= 0:
                                    log_console(
                                        logging.WARNING,
                                        f"Invalid SL calculated ({stop_loss_price}) from ATR={atr_val}. Skipping entry.",
                                        symbol=symbol,
                                    )
                                    continue
                                if take_profit_price is not None:
                                    if take_profit_price <= 0:
                                        take_profit_price = None  # Ignore invalid TP
                                    elif (
                                        entry_side == "Buy"
                                        and take_profit_price <= entry_price_est
                                    ):
                                        take_profit_price = None
                                    elif (
                                        entry_side == "Sell"
                                        and take_profit_price >= entry_price_est
                                    ):
                                        take_profit_price = None

                                # Calculate Size
                                sl_distance_price_abs = abs(
                                    entry_price_est - stop_loss_price
                                )
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

                                if qty_to_trade > 0:
                                    log_console(
                                        logging.INFO,
                                        f"{entry_side.upper()} ENTRY SIGNAL. Str={signal_strength:.2f}. Size={qty_to_trade:.{trader.qty_precision}f}. SL={stop_loss_price:.{trader.price_precision}f}, TP={take_profit_price or 'None'}",
                                        symbol=symbol,
                                    )
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
                                    else:
                                        log_console(
                                            logging.ERROR,
                                            f"{entry_side.upper()} order placement failed.",
                                            symbol=symbol,
                                        )
                                    continue  # Process next symbol after entry attempt
                                else:
                                    log_console(
                                        logging.INFO,
                                        f"{entry_side.upper()} signal, but calculated size is zero. Skipping entry.",
                                        symbol=symbol,
                                    )
                            else:  # No entry signal
                                log_console(
                                    logging.DEBUG,
                                    "No valid entry signal detected.",
                                    symbol=symbol,
                                )

                    except Exception as symbol_e:
                        log_console(
                            logging.ERROR,
                            f"Error processing symbol {symbol}: {symbol_e}",
                            symbol=symbol,
                            exc_info=True,
                        )
                        # Continue to next symbol

                    log_console(
                        logging.DEBUG,
                        f"--- Finished Processing Symbol: {symbol} ---",
                        symbol=symbol,
                    )
                # End of symbol loop

            except KeyboardInterrupt:
                log_console(
                    logging.INFO,
                    "KeyboardInterrupt in main loop. Initiating shutdown...",
                )
                self.shutdown_requested = True
                self.running = False
                break
            except Exception as loop_e:
                log_console(
                    logging.CRITICAL,
                    f"CRITICAL ERROR in main loop: {loop_e}. Shutting down.",
                    exc_info=True,
                )
                self.shutdown_requested = True
                self.running = False
                break

            # --- Post-Cycle Actions ---
            if self.running and not self.shutdown_requested:
                self.metrics.log_summary(force=False, interval=metrics_interval)
                end_time = time.time()
                cycle_duration = end_time - start_time
                sleep_duration = max(0.1, sleep_interval - cycle_duration)
                log_console(
                    logging.DEBUG,
                    f"Cycle duration: {cycle_duration:.2f}s. Sleeping for {sleep_duration:.2f}s...",
                )
                print(f"{Fore.BLUE}{Style.BRIGHT}===== CYCLE END =====")
                try:
                    time.sleep(sleep_duration)
                except KeyboardInterrupt:
                    log_console(
                        logging.INFO, "KeyboardInterrupt during sleep. Shutting down..."
                    )
                    self.shutdown_requested = True
                    self.running = False
                    break
        # End of main loop

        log_console(logging.INFO, "Main trading loop exited.")
        self.shutdown()  # Call shutdown sequence

    def _cleanup_resources(self):
        """Closes WS connection, intended for use after fatal errors during init."""
        log_console(logging.INFO, "Cleaning up resources...")
        self._close_websocket_connection()
        log_console(logging.INFO, "Resource cleanup finished.")

    def shutdown(self):
        """Handles graceful shutdown procedures."""
        if self.shutdown_requested and not self.running:
            log_console(logging.DEBUG, "Shutdown already in progress/completed.")
            return

        log_console(logging.INFO, "--- Initiating Graceful Shutdown ---")
        self.shutdown_requested = True
        self.running = False

        # Stop WebSocket thread first
        if self.ws_connection_thread and self.ws_connection_thread.is_alive():
            log_console(logging.INFO, "Waiting for WebSocket thread to stop...")
            self.ws_connection_thread.join(timeout=15)
            if self.ws_connection_thread.is_alive():
                log_console(logging.WARNING, "WebSocket thread join timed out.")
                # Force close WS object if thread stuck
                self._close_websocket_connection()
            else:
                log_console(logging.INFO, "WebSocket thread stopped.")
        elif self.ws_connection_thread:
            log_console(logging.INFO, "WebSocket thread already stopped.")
            self._close_websocket_connection()  # Ensure closed
        else:
            log_console(logging.INFO, "No WebSocket thread running.")

        # Optional: Close open positions
        close_on_exit = self.bot_cfg.get("CLOSE_POSITIONS_ON_SHUTDOWN", False)
        if close_on_exit and not self.dry_run:
            log_console(
                logging.WARNING, "Attempting to close open positions on shutdown..."
            )
            closed_count = 0
            failed_closes = []
            for symbol, trader in list(self.traders.items()):
                try:
                    position = trader.get_position(log_error=False)  # Don't flood logs
                    if position is None:
                        time.sleep(0.5)
                        position = trader.get_position(log_error=True)  # Retry once

                    if position:
                        log_console(
                            logging.WARNING,
                            f"Closing position for {symbol}...",
                            symbol=symbol,
                        )
                        last_price: Optional[float] = None
                        if trader.kline_df is not None and not trader.kline_df.empty:
                            try:
                                last_price = float(trader.kline_df["close"].iloc[-1])
                            except:
                                pass
                        if trader.close_position(
                            position, exit_reason="Shutdown", exit_price_est=last_price
                        ):
                            closed_count += 1
                            time.sleep(0.5)
                        else:
                            failed_closes.append(symbol)
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
                    f"FAILED TO CLOSE: {', '.join(failed_closes)}. MANUAL ACTION REQUIRED.",
                )
            elif close_on_exit:
                log_console(logging.INFO, "Position closing attempt complete.")
        elif self.dry_run:
            log_console(logging.INFO, "Dry run: No real positions to close.")
        else:
            log_console(
                logging.INFO, "Close on shutdown disabled. Positions may remain open."
            )

        # Save final metrics
        try:
            self.metrics.save_on_shutdown()
        except Exception as metrics_e:
            log_console(
                logging.ERROR, f"Error saving final metrics: {metrics_e}", exc_info=True
            )

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
    parser.add_argument(
        "--dry-run", action="store_true", help="Enable dry run mode (simulation only)."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG level logging."
    )
    args = parser.parse_args()

    log_level_console = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    log_console(log_level_console, f"{Fore.CYAN}{Style.BRIGHT}--- Bot Starting ---")
    log_console(log_level_console, f"Using configuration file: {args.config}")
    if args.dry_run:
        print(
            f"\n{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT} *** WARNING: DRY RUN MODE ACTIVE *** {Style.RESET_ALL}\n"
        )
        log_console(logging.WARNING, "*** DRY RUN MODE ACTIVE ***")
    if args.debug:
        log_console(logging.DEBUG, "Debug logging enabled.")

    # --- Load and Validate Configuration ---
    config: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = []
    try:
        config = load_config(args.config)  # Basic load and KLINE_LIMIT calc

        # --- Detailed Validation ---
        log_console(logging.INFO, "Validating configuration sections...")
        required_sections = {
            "BYBIT_CONFIG": dict,
            "RISK_CONFIG": dict,
            "BOT_CONFIG": dict,
        }
        for section, expected_type in required_sections.items():
            if section not in config:
                validation_errors.append(f"Missing section '{section}'.")
            elif not isinstance(config.get(section), expected_type):
                validation_errors.append(
                    f"Section '{section}' must be {expected_type.__name__}."
                )

        if not validation_errors:  # Proceed only if top sections are okay
            bybit_cfg = config.get("BYBIT_CONFIG", {})
            risk_cfg = config.get("RISK_CONFIG", {})
            bot_cfg = config.get("BOT_CONFIG", {})
            symbols = bybit_cfg.get("SYMBOLS", [])

            # Bybit Config
            if not isinstance(bybit_cfg.get("USE_TESTNET", False), bool):
                validation_errors.append(
                    "BYBIT_CONFIG: USE_TESTNET must be true/false."
                )
            account_type = bybit_cfg.get("ACCOUNT_TYPE", "UNIFIED").upper()
            valid_account_types = ["UNIFIED", "CONTRACT", "SPOT"]
            if account_type not in valid_account_types:
                validation_errors.append(
                    f"BYBIT_CONFIG: ACCOUNT_TYPE '{account_type}' invalid ({valid_account_types})."
                )

            # Risk Config
            risk_percent = risk_cfg.get("RISK_PER_TRADE_PERCENT")
            if not isinstance(risk_percent, (int, float)) or not (
                0 < risk_percent <= 10
            ):
                validation_errors.append(
                    f"RISK_CONFIG: RISK_PER_TRADE_PERCENT ({risk_percent}) invalid (0.1-10 recommended)."
                )
            fee_rate = risk_cfg.get("FEE_RATE")
            if not isinstance(fee_rate, (int, float)) or fee_rate < 0:
                validation_errors.append(
                    f"RISK_CONFIG: FEE_RATE ({fee_rate}) must be non-negative."
                )
            max_pos_usdt = risk_cfg.get("MAX_POSITION_USDT")
            if max_pos_usdt is not None and (
                not isinstance(max_pos_usdt, (int, float)) or max_pos_usdt <= 0
            ):
                validation_errors.append(
                    f"RISK_CONFIG: MAX_POSITION_USDT ({max_pos_usdt}) must be positive if set."
                )
            if args.dry_run:
                dry_run_balance = risk_cfg.get("DRY_RUN_DUMMY_BALANCE")
                if dry_run_balance is None or (
                    not isinstance(dry_run_balance, (int, float))
                    or dry_run_balance <= 0
                ):
                    validation_errors.append(
                        f"RISK_CONFIG: DRY_RUN_DUMMY_BALANCE ({dry_run_balance}) must be positive in dry run."
                    )
            if (
                not isinstance(risk_cfg.get("ATR_MULTIPLIER_SL", 1.5), (int, float))
                or risk_cfg.get("ATR_MULTIPLIER_SL", 1.5) <= 0
            ):
                validation_errors.append(
                    "RISK_CONFIG: ATR_MULTIPLIER_SL must be positive."
                )
            if (
                not isinstance(risk_cfg.get("ATR_MULTIPLIER_TP", 2.0), (int, float))
                or risk_cfg.get("ATR_MULTIPLIER_TP", 2.0) < 0
            ):
                validation_errors.append(
                    "RISK_CONFIG: ATR_MULTIPLIER_TP must be non-negative."
                )
            if not isinstance(risk_cfg.get("ENABLE_TRAILING_STOP", False), bool):
                validation_errors.append(
                    "RISK_CONFIG: ENABLE_TRAILING_STOP must be true/false."
                )

            # Bot Config
            if (
                not isinstance(bot_cfg.get("SLEEP_INTERVAL_SECONDS"), (int, float))
                or bot_cfg.get("SLEEP_INTERVAL_SECONDS") <= 0
            ):
                validation_errors.append(
                    "BOT_CONFIG: SLEEP_INTERVAL_SECONDS must be positive."
                )
            if (
                not isinstance(bot_cfg.get("KLINE_LIMIT_BUFFER"), int)
                or bot_cfg.get("KLINE_LIMIT_BUFFER") < 0
            ):
                validation_errors.append(
                    "BOT_CONFIG: KLINE_LIMIT_BUFFER must be non-negative."
                )
            # ... Add checks for all other BOT_CONFIG params used ...
            if (
                not isinstance(bot_cfg.get("ORDER_CONFIRM_RETRIES", 3), int)
                or bot_cfg.get("ORDER_CONFIRM_RETRIES", 3) < 0
            ):
                validation_errors.append(
                    "BOT_CONFIG: ORDER_CONFIRM_RETRIES must be non-negative."
                )
            if (
                not isinstance(
                    bot_cfg.get("ORDER_CONFIRM_DELAY_SECONDS", 2.0), (int, float)
                )
                or bot_cfg.get("ORDER_CONFIRM_DELAY_SECONDS", 2.0) <= 0
            ):
                validation_errors.append(
                    "BOT_CONFIG: ORDER_CONFIRM_DELAY_SECONDS must be positive."
                )
            if not isinstance(bot_cfg.get("PROCESS_CONFIRMED_KLINE_ONLY", True), bool):
                validation_errors.append(
                    "BOT_CONFIG: PROCESS_CONFIRMED_KLINE_ONLY must be true/false."
                )
            if not isinstance(bot_cfg.get("CLOSE_POSITIONS_ON_SHUTDOWN", False), bool):
                validation_errors.append(
                    "BOT_CONFIG: CLOSE_POSITIONS_ON_SHUTDOWN must be true/false."
                )
            if (
                not isinstance(bot_cfg.get("HIGHER_TF_CACHE_SECONDS", 3600), int)
                or bot_cfg.get("HIGHER_TF_CACHE_SECONDS", 3600) < 0
            ):
                validation_errors.append(
                    "BOT_CONFIG: HIGHER_TF_CACHE_SECONDS must be non-negative."
                )
            if (
                not isinstance(bot_cfg.get("KLINE_STALENESS_SECONDS", 300), int)
                or bot_cfg.get("KLINE_STALENESS_SECONDS", 300) <= 0
            ):
                validation_errors.append(
                    "BOT_CONFIG: KLINE_STALENESS_SECONDS must be positive."
                )

            # Symbol Configs
            if not isinstance(symbols, list):
                validation_errors.append("BYBIT_CONFIG: SYMBOLS must be a list.")
            elif not symbols:
                validation_errors.append("BYBIT_CONFIG: SYMBOLS list is empty.")
            else:
                seen_symbols = set()
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
                        validation_errors.append(
                            f"SYMBOLS entry {i}: Must be a dictionary."
                        )
                        continue
                    symbol = symbol_cfg.get("SYMBOL")
                    if not symbol or not isinstance(symbol, str):
                        validation_errors.append(
                            f"SYMBOLS entry {i}: Missing/invalid 'SYMBOL'."
                        )
                        continue
                    if symbol in seen_symbols:
                        validation_errors.append(f"Duplicate symbol: {symbol}")
                        continue
                    seen_symbols.add(symbol)

                    if "STRATEGY_CONFIG" not in symbol_cfg or not isinstance(
                        symbol_cfg.get("STRATEGY_CONFIG"), dict
                    ):
                        validation_errors.append(
                            f"Symbol {symbol}: Missing/invalid 'STRATEGY_CONFIG'."
                        )
                    leverage = symbol_cfg.get("LEVERAGE")
                    if not isinstance(leverage, int) or not (0 < leverage <= 100):
                        validation_errors.append(
                            f"Symbol {symbol}: LEVERAGE ({leverage}) invalid (1-100 integer)."
                        )
                    timeframe = str(symbol_cfg.get("TIMEFRAME", ""))
                    if timeframe not in valid_timeframes:
                        validation_errors.append(
                            f"Symbol {symbol}: Invalid TIMEFRAME '{timeframe}' ({valid_timeframes})"
                        )
                    if not isinstance(symbol_cfg.get("USE_WEBSOCKET", True), bool):
                        validation_errors.append(
                            f"Symbol {symbol}: USE_WEBSOCKET must be true/false."
                        )

                    strategy_cfg = symbol_cfg.get("STRATEGY_CONFIG", {})
                    if isinstance(strategy_cfg, dict):
                        for k, v in strategy_cfg.items():
                            if "PERIOD" in k.upper() and (
                                not isinstance(v, int) or v <= 0
                            ):
                                validation_errors.append(
                                    f"Symbol {symbol}: Param '{k}' ({v}) must be positive integer."
                                )
                            if k.upper() in [
                                "USE_EHLERS_SMOOTHER",
                                "ENABLE_MULTI_TIMEFRAME",
                            ] and not isinstance(v, bool):
                                validation_errors.append(
                                    f"Symbol {symbol}: Param '{k}' ({v}) must be true/false."
                                )
                            if k.upper() in [
                                "MOM_THRESHOLD_PERCENT",
                                "ADX_THRESHOLD",
                                "MIN_SIGNAL_STRENGTH",
                            ] and not isinstance(v, (int, float)):
                                validation_errors.append(
                                    f"Symbol {symbol}: Param '{k}' ({v}) must be a number."
                                )
                            if k.upper() in [
                                "SL_TRIGGER_BY",
                                "TP_TRIGGER_BY",
                            ] and v not in [
                                "LastPrice",
                                "MarkPrice",
                                "IndexPrice",
                            ]:
                                validation_errors.append(
                                    f"Symbol {symbol}: Param '{k}' ({v}) must be LastPrice, MarkPrice, or IndexPrice."
                                )
                        if strategy_cfg.get("ENABLE_MULTI_TIMEFRAME"):
                            htf = str(strategy_cfg.get("HIGHER_TIMEFRAME", ""))
                            if htf not in valid_timeframes:
                                validation_errors.append(
                                    f"Symbol {symbol}: Invalid HIGHER_TIMEFRAME '{htf}'."
                                )
                            elif htf == timeframe:
                                validation_errors.append(
                                    f"Symbol {symbol}: HIGHER_TIMEFRAME cannot be same as TIMEFRAME."
                                )

    except SystemExit:
        sys.exit(1)  # Config load failed and exited
    except Exception as cfg_e:
        log_console(
            logging.CRITICAL,
            f"Unexpected error loading/validating config: {cfg_e}",
            exc_info=True,
        )
        sys.exit(1)

    # Report Validation Results
    if validation_errors:
        log_console(
            logging.CRITICAL,
            f"{Back.RED}{Fore.WHITE}Config validation failed ({len(validation_errors)} errors):",
        )
        for error in validation_errors:
            log_console(logging.CRITICAL, f"- {error}")
        sys.exit(1)
    else:
        log_console(logging.INFO, "Configuration validation passed.")

    # --- Initialize and Run Bot ---
    trader_instance: Optional[MomentumScannerTrader] = None
    exit_code = 0
    try:
        log_console(logging.INFO, "Initializing main trader instance...")
        # Pass validated config
        trader_instance = MomentumScannerTrader(
            API_KEY, API_SECRET, config, args.dry_run
        )

        if trader_instance and trader_instance.running:
            trader_instance.run()  # Blocks until shutdown
        else:
            log_console(
                logging.CRITICAL, "Trader initialization failed. Bot did not run."
            )
            exit_code = 1

    except KeyboardInterrupt:
        log_console(
            logging.INFO, "\nKeyboardInterrupt detected. Initiating shutdown..."
        )
        if trader_instance:
            trader_instance.shutdown()  # Trigger graceful shutdown
        else:
            log_console(
                logging.WARNING,
                "Trader instance not fully initialized before interrupt.",
            )
        exit_code = 0  # Clean exit
    except (ConnectionError, RuntimeError, ValueError) as critical_e:
        log_console(
            logging.CRITICAL,
            f"Caught critical error: {critical_e}. Bot stopped.",
            exc_info=False,
        )
        if trader_instance:
            trader_instance.shutdown()
        exit_code = 1
    except Exception as e:
        log_console(
            logging.CRITICAL,
            f"Unexpected critical error in main execution: {e}",
            exc_info=True,
        )
        if trader_instance:
            trader_instance.shutdown()
        exit_code = 1
    finally:
        log_console(
            log_level_console,
            f"{Fore.CYAN}{Style.BRIGHT}--- Bot Execution Finished ---",
        )
        logging.shutdown()
        sys.exit(exit_code)
