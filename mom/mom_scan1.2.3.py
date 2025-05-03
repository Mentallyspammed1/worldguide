```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Momentum Scanner Trading Bot for Bybit (V5 API) - mom_scan1.2.py

This bot scans multiple symbols based on momentum indicators, Ehlers filters (optional),
and other technical analysis tools to identify potential trading opportunities on Bybit's
Unified Trading Account (primarily linear perpetual contracts). It utilizes both
WebSocket for real-time data and REST API for historical data and order management.

Key Features:
- Multi-Symbol Trading: Monitors and trades multiple configured symbols concurrently.
- Configurable Strategy: Parameters per symbol defined via JSON (periods, thresholds, etc.).
- Momentum Indicators: EMA/SuperSmoother, ROC, RSI, ADX, Bollinger Bands.
- Optional Ehlers Filters: Super Smoother, Instantaneous Trendline (Note: sensitive to tuning).
- Volume Analysis: Volume SMA and rolling percentile for volume surge detection.
- Dynamic Thresholds: Adjusts RSI/ROC thresholds based on market volatility (ATR).
- ATR-Based Risk Management: Position sizing, Stop Loss, and Take Profit calculated using ATR.
- Optional Multi-Timeframe Analysis: Filters trades based on trend confirmation on a higher timeframe.
- WebSocket Integration: Low-latency kline updates with robust reconnection logic and initial data population.
- REST API Fallback: Uses REST API for historical data, initial population, and order management.
- Robust Error Handling: Specific Bybit V5 error code checks and comprehensive exception handling.
- Dry Run Mode: Simulates trading logic without placing real orders.
- Detailed Logging: Colored console output (via Colorama) and clean UTF-8 encoded file output.
- Trade Performance Tracking: Thread-safe metrics logging (P&L, win rate, fees, etc.).
- Graceful Shutdown: Handles SIGINT/SIGTERM signals for clean exit, including optional position closing.
- Comprehensive Configuration Validation: Checks config structure, types, and values on startup.
- Dynamic Kline Limit Calculation: Automatically determines required historical data points based on indicator periods.
- Improved NaN Handling: Rigorous checks and filling of NaN values throughout data processing.
- Clear Action Logging: Distinct logging for trade actions (entry, exit attempts).
- Enhanced WebSocket Management: Dedicated thread for connection monitoring and reconnection.
"""

# --- Standard Library Imports ---
import os
import sys
import time
import logging
import json
import argparse
import threading
import math
import signal
import datetime as dt
from queue import Queue, Empty
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List, Union

# --- Third-party Library Imports ---
# Attempt to import required libraries and provide helpful error messages
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: 'pandas' or 'numpy' library not found. Please install them: pip install pandas numpy")
    sys.exit(1)
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
MAX_KLINE_LIMIT_PER_REQUEST = 1000  # Maximum klines fetchable in one V5 API call
# Minimum records needed for basic calcs like IT (>= 4) - overridden by dynamic calc
MIN_KLINE_RECORDS_FOR_CALC = 10 # Increased minimum for better stability of some indicators
# Used for floating point comparisons (adjust sensitivity as needed)
FLOAT_COMPARISON_TOLERANCE = 1e-9

# --- Logging Setup ---

# Base Logger Configuration (File Handler, no colors in file, UTF-8 encoded)
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
# Use 'a' mode to append to log file, change to 'w' to overwrite on each run
LOG_FILE_MODE = "a"
try:
    FILE_HANDLER = logging.FileHandler(
        LOG_FILE, mode=LOG_FILE_MODE, encoding='utf-8')
    FILE_HANDLER.setFormatter(LOG_FORMATTER)
except Exception as log_setup_e:
    print(
        f"FATAL: Could not configure file logging for {LOG_FILE}: {log_setup_e}")
    sys.exit(1)

# Main application logger
logger = logging.getLogger("MomentumScannerTrader")
logger.addHandler(FILE_HANDLER)
logger.setLevel(logging.INFO)  # Default level, can be overridden by --debug
logger.propagate = False  # Prevent duplicate logs if root logger is configured

# Metrics Logger Configuration (CSV-like format, UTF-8 encoded)
METRICS_FORMATTER = logging.Formatter(
    "%(asctime)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
try:
    METRICS_HANDLER = logging.FileHandler(
        METRICS_LOG_FILE, mode="a", encoding='utf-8')
    METRICS_HANDLER.setFormatter(METRICS_FORMATTER)
    # Add header if the file is new/empty
    if os.path.getsize(METRICS_LOG_FILE) == 0:
        METRICS_HANDLER.stream.write("Timestamp,Type,Symbol,EntryTime,ExitTime,DurationSeconds,Side,EntryPrice,ExitPrice,Qty,Leverage,GrossPnL,Fees,NetPnL,Outcome\n")
        METRICS_HANDLER.stream.flush()
except Exception as metrics_log_setup_e:
    print(
        f"FATAL: Could not configure metrics file logging for {METRICS_LOG_FILE}: {metrics_log_setup_e}")
    sys.exit(1)

# Metrics logger instance
metrics_logger = logging.getLogger("TradeMetrics")
metrics_logger.addHandler(METRICS_HANDLER)
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False

# --- Console Logging Function ---

# Neon-Colored Console Output with Symbol Support
SYMBOL_COLORS = {
    # Define specific symbols for distinct colors if desired
    "BTCUSDT": Fore.YELLOW + Style.BRIGHT,
    "ETHUSDT": Fore.BLUE + Style.BRIGHT,
    # Add more specific symbol colors here
    "default": Fore.WHITE,  # Default color if symbol not specified or found
}
_symbol_color_cycle = [Fore.CYAN, Fore.MAGENTA, Fore.GREEN, Fore.LIGHTRED_EX,
                       Fore.LIGHTBLUE_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTMAGENTA_EX]
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
        color = _symbol_color_cycle[_color_index %
                                    len(_symbol_color_cycle)] + Style.BRIGHT
        _symbol_color_map[symbol] = color
        _color_index += 1
    return _symbol_color_map[symbol]


def log_console(level: int, message: Any, symbol: Optional[str] = None, *args, exc_info: bool = False, **kwargs):
    """
    Logs messages to the console with level-specific colors and optional
    symbol highlighting. Also forwards the message to the file logger.

    Args:
        level: Logging level (e.g., logging.INFO, logging.ERROR).
        message: The message object to log (will be converted to string).
        symbol: Optional symbol name to include with specific color coding.
        *args: Additional arguments for string formatting (if message is a format string).
        exc_info: If True, add exception information to the file log.
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
    # Default to white if level not found
    color = neon_colors.get(level, Fore.WHITE)
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
        print(
            f"{Fore.RED}LOG FORMATTING ERROR:{Style.RESET_ALL} {fmt_e} | Original Msg: {message_str}")
        # Also log the error to the file logger for record-keeping
        logger.error(
            f"Log formatting error: {fmt_e} | Original Msg: {message_str}", exc_info=False)
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
        message: The metrics message string (typically comma-separated values following header).
    """
    try:
        metrics_logger.info(message)
        # Print metrics to console with a distinct color for visibility
        print(f"{Fore.MAGENTA+Style.BRIGHT}METRICS:{Style.RESET_ALL} {message}")
    except Exception as e:
        log_console(logging.ERROR,
                    f"Failed to log metrics: {e}", exc_info=True)


# --- Load API Keys from Environment Variables ---
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

# Critical check: Ensure API keys are loaded before proceeding
if not API_KEY or not API_SECRET:
    # Use log_console for consistency, even at critical level before full logging setup
    log_console(logging.CRITICAL,
                "BYBIT_API_KEY or BYBIT_API_SECRET not found in environment variables or .env file. Exiting.")
    sys.exit(1)
else:
    log_console(logging.INFO, "API keys loaded successfully.")


# --- Configuration Loading & Validation ---
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the JSON configuration file, performs basic structural validation, and calculates
    the required KLINE_LIMIT for each symbol based on indicator periods.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        The loaded configuration dictionary.

    Raises:
        SystemExit: If the config file is missing, invalid JSON, or lacks essential structure.
    """
    log_console(
        logging.INFO, f"Attempting to load configuration file: {config_path}")
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            config = json.load(f)
        log_console(
            logging.INFO, f"Successfully loaded configuration from {config_path}")

        # --- Basic Structure Validation ---
        if not isinstance(config, dict):
            raise ValueError(
                "Configuration file root must be a JSON object (dictionary).")
        if "BYBIT_CONFIG" not in config or not isinstance(config["BYBIT_CONFIG"], dict):
            raise ValueError(
                "Missing or invalid 'BYBIT_CONFIG' section (must be a dictionary).")
        if "SYMBOLS" not in config["BYBIT_CONFIG"] or not isinstance(config["BYBIT_CONFIG"]["SYMBOLS"], list):
            raise ValueError("'BYBIT_CONFIG.SYMBOLS' must be a list.")
        if not config["BYBIT_CONFIG"]["SYMBOLS"]:
            raise ValueError("'BYBIT_CONFIG.SYMBOLS' list cannot be empty.")
        if "BOT_CONFIG" not in config or not isinstance(config["BOT_CONFIG"], dict):
            raise ValueError(
                "Missing or invalid 'BOT_CONFIG' section (must be a dictionary).")
        if "RISK_CONFIG" not in config or not isinstance(config["RISK_CONFIG"], dict):
            raise ValueError(
                "Missing or invalid 'RISK_CONFIG' section (must be a dictionary).")

        # --- Calculate KLINE_LIMIT per Symbol ---
        bybit_cfg = config["BYBIT_CONFIG"]
        symbols_cfg = bybit_cfg["SYMBOLS"]
        bot_cfg = config["BOT_CONFIG"]
        # Default buffer, ensures enough data for indicator warmup and lookbacks
        kline_limit_buffer = bot_cfg.get("KLINE_LIMIT_BUFFER", 50)
        # Ensure buffer is non-negative integer
        if not isinstance(kline_limit_buffer, int) or kline_limit_buffer < 0:
            log_console(
                logging.WARNING, f"Invalid KLINE_LIMIT_BUFFER ({kline_limit_buffer}), using default 50.")
            kline_limit_buffer = 50

        for symbol_index, symbol_cfg in enumerate(symbols_cfg):
            if not isinstance(symbol_cfg, dict) or "SYMBOL" not in symbol_cfg or not isinstance(symbol_cfg["SYMBOL"], str) or not symbol_cfg["SYMBOL"]:
                raise ValueError(
                    f"Each item in 'SYMBOLS' (index {symbol_index}) must be a dictionary with a valid non-empty string 'SYMBOL' key.")

            symbol = symbol_cfg["SYMBOL"]
            strategy_cfg = symbol_cfg.get("STRATEGY_CONFIG", {})
            if not isinstance(strategy_cfg, dict):
                log_console(
                    logging.WARNING, f"Missing or invalid 'STRATEGY_CONFIG' for {symbol}. Using defaults/basic checks.", symbol=symbol)
                strategy_cfg = {}  # Ensure it exists as dict for lookups

            # Collect all periods used by indicators for this symbol
            # Ensure all relevant periods from calculate_indicators_momentum are included
            periods = [
                strategy_cfg.get("ULTRA_FAST_EMA_PERIOD", 5),
                strategy_cfg.get("FAST_EMA_PERIOD", 10),
                strategy_cfg.get("MID_EMA_PERIOD", 30),
                strategy_cfg.get("SLOW_EMA_PERIOD", 89),
                strategy_cfg.get("ROC_PERIOD", 5),
                strategy_cfg.get("ATR_PERIOD_RISK", 14),
                # Used for dynamic thresholds
                strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14),
                strategy_cfg.get("RSI_PERIOD", 10),
                strategy_cfg.get("VOLUME_SMA_PERIOD", 20),
                strategy_cfg.get("ADX_PERIOD", 14),
                strategy_cfg.get("BBANDS_PERIOD", 20),
                strategy_cfg.get("INSTANTANEOUS_TRENDLINE_PERIOD", 20),
                3,  # Fixed 3-period SMA for ROC smoothing
                # Period for rolling volume percentile (same as SMA)
                strategy_cfg.get("VOLUME_SMA_PERIOD", 20),
                # Period for rolling ATR for volatility factor
                strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14),
            ]

            # Consider higher timeframe periods if MTF is enabled
            if strategy_cfg.get("ENABLE_MULTI_TIMEFRAME", False):
                # Assume HTF uses the same periods; adjust if HTF has different settings
                # This is a simplification; HTF might have different period requirements
                # Doubling the list accounts for HTF needing the same max period on its own timeframe.
                # Note: This might overestimate slightly if HTF uses shorter periods, but ensures enough data.
                periods.extend(periods)

            # Filter out invalid or non-positive periods
            valid_periods = [p for p in periods if isinstance(
                p, (int, float)) and p > 0]
            if not valid_periods:
                log_console(
                    logging.WARNING, f"No valid positive indicator periods found for {symbol}. Using default max period (200).", symbol=symbol)
                max_period = 200  # Default fallback if no periods found
            else:
                max_period = max(valid_periods)

            # Ensure kline limit is at least the minimum required + buffer
            # Add extra buffer for complex indicators like ADX which require more warmup
            adx_period_val = strategy_cfg.get("ADX_PERIOD", 14)
            adx_warmup_buffer = 0
            if isinstance(adx_period_val, int) and adx_period_val > 0:
                # ADX needs roughly 2*period for stable values + the period itself
                adx_warmup_buffer = 2 * adx_period_val
            else:
                adx_warmup_buffer = 30  # Default ADX warmup if period invalid/missing

            # Calculate base limit needed
            calculated_limit = int(
                max_period + kline_limit_buffer + adx_warmup_buffer)

            # Ensure it's at least the minimum required for basic calcs like Ehlers IT + buffer
            # Ehlers IT needs at least 4 records. ADX also needs warmup.
            min_records_needed = max(MIN_KLINE_RECORDS_FOR_CALC, adx_period_val +
                                     adx_warmup_buffer if adx_period_val > 0 else MIN_KLINE_RECORDS_FOR_CALC)
            kline_limit = max(min_records_needed +
                              kline_limit_buffer, calculated_limit)

            # Cap the limit at a reasonable maximum to avoid excessive memory/API usage
            # MAX_KLINE_LIMIT_PER_REQUEST is the API fetch limit; internal limit can be higher for rolling calcs
            # Allow storing more than one fetch worth, adjust if needed
            internal_max_limit = int(bot_cfg.get(
                "MAX_INTERNAL_KLINE_ROWS", 2 * MAX_KLINE_LIMIT_PER_REQUEST))
            kline_limit = min(kline_limit, internal_max_limit)

            # Store calculated limit within the symbol's config (internal use)
            # Use setdefault to avoid overwriting if already exists (though unlikely here)
            symbol_cfg.setdefault("INTERNAL", {})["KLINE_LIMIT"] = kline_limit
            log_console(
                logging.DEBUG, f"Calculated KLINE_LIMIT for {symbol}: {kline_limit} (Max Period: {max_period:.0f}, Buffer: {kline_limit_buffer}, ADX Buf: {adx_warmup_buffer}, Min Needed: {min_records_needed}, Capped: {internal_max_limit})", symbol=symbol)

        # --- Comprehensive Configuration Validation (Performed after load) ---
        # Moved validation logic to a dedicated function called in main block.

        return config

    except FileNotFoundError:
        log_console(
            logging.CRITICAL, f"Configuration file not found: {config_path}. Please ensure it exists.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_console(
            logging.CRITICAL, f"Error decoding JSON configuration file {config_path}: {e}")
        sys.exit(1)
    except ValueError as e:  # Catch specific validation errors raised above
        log_console(logging.CRITICAL, f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        log_console(
            logging.CRITICAL, f"Unexpected error loading or performing initial validation of configuration: {e}", exc_info=True)
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Performs detailed validation of the loaded configuration dictionary.

    Args:
        config: The loaded configuration dictionary.

    Returns:
        True if the configuration is valid, False otherwise. Logs specific errors/warnings.
    """
    is_valid = True
    log_console(logging.INFO, "Performing detailed configuration validation...")

    # --- Helper Function for Type/Range Checks ---
    def check_param(cfg_dict: Dict, key: str, expected_type: Union[type, Tuple[type, ...]],
                    min_val: Optional[Union[int, float]] = None, max_val: Optional[Union[int, float]] = None,
                    allowed_values: Optional[List[Any]] = None, context: str = "", is_required: bool = True) -> bool:
        nonlocal is_valid
        value = cfg_dict.get(key)
        param_id = f"{context}.{key}" if context else key

        if value is None:
            if is_required:
                log_console(
                    logging.ERROR, f"Config Validation: Required parameter '{param_id}' is missing.")
                is_valid = False
                return False
            else:
                return True # Optional parameter is missing, which is fine

        if not isinstance(value, expected_type):
            log_console(
                logging.ERROR, f"Config Validation: Parameter '{param_id}' has incorrect type. Expected {expected_type}, got {type(value)}.")
            is_valid = False
            return False

        if isinstance(value, (int, float)):  # Numeric range checks
            if min_val is not None and value < min_val:
                log_console(
                    logging.ERROR, f"Config Validation: Parameter '{param_id}' ({value}) is below minimum allowed value ({min_val}).")
                is_valid = False
                return False
            if max_val is not None and value > max_val:
                log_console(
                    logging.ERROR, f"Config Validation: Parameter '{param_id}' ({value}) is above maximum allowed value ({max_val}).")
                is_valid = False
                return False

        if allowed_values is not None and value not in allowed_values:
            log_console(
                logging.ERROR, f"Config Validation: Parameter '{param_id}' ({value}) is not one of the allowed values: {allowed_values}.")
            is_valid = False
            return False

        # Specific check for string types that shouldn't be empty if required
        if is_required and isinstance(value, str) and not value:
             log_console(
                logging.ERROR, f"Config Validation: Required string parameter '{param_id}' cannot be empty.")
             is_valid = False
             return False

        return True  # Parameter passed validation

    # --- Validate BYBIT_CONFIG ---
    bybit_cfg = config.get("BYBIT_CONFIG", {})
    check_param(bybit_cfg, "USE_TESTNET", bool, context="BYBIT_CONFIG")
    check_param(bybit_cfg, "ACCOUNT_TYPE", str, allowed_values=[
                # Case sensitive per code logic
                "UNIFIED", "CONTRACT", "SPOT"], context="BYBIT_CONFIG")

    symbols_cfg = bybit_cfg.get("SYMBOLS", [])
    if not isinstance(symbols_cfg, list) or not symbols_cfg:
        log_console(
            logging.ERROR, "Config Validation: BYBIT_CONFIG.SYMBOLS must be a non-empty list.")
        is_valid = False
    else:
        allowed_timeframes = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"] # Bybit V5 standard intervals
        for i, symbol_cfg in enumerate(symbols_cfg):
            symbol_context = f"BYBIT_CONFIG.SYMBOLS[{i}]"
            if not isinstance(symbol_cfg, dict):
                log_console(
                    logging.ERROR, f"Config Validation: Item at {symbol_context} must be a dictionary.")
                is_valid = False
                continue
            if not check_param(symbol_cfg, "SYMBOL", str, context=symbol_context):
                continue  # Skip further checks if symbol itself is invalid

            symbol = symbol_cfg["SYMBOL"]
            # Check TIMEFRAME against allowed values
            check_param(symbol_cfg, "TIMEFRAME", str, allowed_values=allowed_timeframes, context=symbol_context)
            check_param(symbol_cfg, "LEVERAGE", (int, float),
                        min_val=1, context=symbol_context)
            check_param(symbol_cfg, "USE_WEBSOCKET",
                        bool, context=symbol_context)

            # Validate STRATEGY_CONFIG within each symbol
            strategy_cfg = symbol_cfg.get("STRATEGY_CONFIG", {})
            strategy_context = f"{symbol_context}({symbol}).STRATEGY_CONFIG"
            if not isinstance(strategy_cfg, dict):
                log_console(
                    logging.ERROR, f"Config Validation: {strategy_context} must be a dictionary.")
                is_valid = False
                continue

            check_param(strategy_cfg, "PRICE_SOURCE", str, allowed_values=[
                        "close", "open", "high", "low", "hlc3", "ohlc4"], context=strategy_context)
            check_param(strategy_cfg, "USE_EHLERS_SMOOTHER",
                        bool, context=strategy_context)
            check_param(strategy_cfg, "ULTRA_FAST_EMA_PERIOD",
                        int, min_val=1, context=strategy_context)
            check_param(strategy_cfg, "FAST_EMA_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "MID_EMA_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "SLOW_EMA_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "ROC_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "ATR_PERIOD_RISK", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "ATR_PERIOD_VOLATILITY",
                        int, min_val=1, context=strategy_context)
            check_param(strategy_cfg, "RSI_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "VOLUME_SMA_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "ADX_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "BBANDS_PERIOD", int,
                        min_val=1, context=strategy_context)
            check_param(strategy_cfg, "BBANDS_STDDEV", (int, float),
                        min_val=0.1, context=strategy_context)
            check_param(strategy_cfg, "INSTANTANEOUS_TRENDLINE_PERIOD", int,
                        min_val=4, context=strategy_context)  # Min period for IT calc
            check_param(strategy_cfg, "RSI_LOW_THRESHOLD", (int, float),
                        min_val=0, max_val=100, context=strategy_context)
            check_param(strategy_cfg, "RSI_HIGH_THRESHOLD", (int, float),
                        min_val=0, max_val=100, context=strategy_context)
            check_param(strategy_cfg, "RSI_VOLATILITY_ADJUST",
                        (int, float), min_val=0, context=strategy_context)
            check_param(strategy_cfg, "MOM_THRESHOLD_PERCENT",
                        (int, float), context=strategy_context)  # Can be negative
            check_param(strategy_cfg, "VOLATILITY_FACTOR_MIN",
                        (int, float), min_val=0.1, context=strategy_context)
            check_param(strategy_cfg, "VOLATILITY_FACTOR_MAX",
                        (int, float), min_val=0.1, context=strategy_context)
            check_param(strategy_cfg, "ADX_THRESHOLD", (int, float),
                        min_val=0, max_val=100, context=strategy_context)
            check_param(strategy_cfg, "MIN_SIGNAL_STRENGTH", (int, float),
                        min_val=0.0, max_val=1.0, context=strategy_context)
            check_param(strategy_cfg, "SL_TRIGGER_BY", str, allowed_values=[
                        "LastPrice", "MarkPrice", "IndexPrice"], context=strategy_context)
            check_param(strategy_cfg, "TP_TRIGGER_BY", str, allowed_values=[
                        "LastPrice", "MarkPrice", "IndexPrice"], context=strategy_context)
            check_param(strategy_cfg, "TPSL_MODE", str, allowed_values=[
                        "Full", "Partial"], context=strategy_context)
            check_param(strategy_cfg, "ENABLE_MULTI_TIMEFRAME", bool, context=strategy_context)

            # Multi-timeframe validation (only if enabled)
            if strategy_cfg.get("ENABLE_MULTI_TIMEFRAME", False):
                # Check HIGHER_TIMEFRAME exists, is allowed, and is different from base TIMEFRAME
                if check_param(strategy_cfg, "HIGHER_TIMEFRAME", str, allowed_values=allowed_timeframes, context=strategy_context):
                    base_tf = symbol_cfg.get("TIMEFRAME")
                    htf = strategy_cfg.get("HIGHER_TIMEFRAME")
                    # Simple check for equality; TODO: Add check for HTF being logically "higher" (e.g., 60 > 15)
                    if base_tf == htf:
                        log_console(
                            logging.ERROR, f"Config Validation: HIGHER_TIMEFRAME ({htf}) cannot be the same as TIMEFRAME ({base_tf}) in {strategy_context}.")
                        is_valid = False

    # --- Validate BOT_CONFIG ---
    bot_cfg = config.get("BOT_CONFIG", {})
    check_param(bot_cfg, "SLEEP_INTERVAL_SECONDS",
                (int, float), min_val=1, context="BOT_CONFIG")
    check_param(bot_cfg, "METRICS_LOG_INTERVAL_SECONDS",
                int, min_val=60, context="BOT_CONFIG")
    check_param(bot_cfg, "BALANCE_CHECK_INTERVAL_SECONDS",
                int, min_val=60, context="BOT_CONFIG")
    check_param(bot_cfg, "KLINE_LIMIT_BUFFER", int,
                min_val=0, context="BOT_CONFIG")
    check_param(bot_cfg, "MAX_INTERNAL_KLINE_ROWS",
                int, min_val=50, context="BOT_CONFIG")
    check_param(bot_cfg, "PROCESS_CONFIRMED_KLINE_ONLY",
                bool, context="BOT_CONFIG")
    check_param(bot_cfg, "ORDER_CONFIRM_RETRIES",
                int, min_val=0, context="BOT_CONFIG")
    check_param(bot_cfg, "ORDER_CONFIRM_DELAY_SECONDS",
                (int, float), min_val=0.1, context="BOT_CONFIG")
    check_param(bot_cfg, "WS_RECONNECT_DELAY_INIT_SECONDS",
                (int, float), min_val=1, context="BOT_CONFIG")
    check_param(bot_cfg, "WS_RECONNECT_DELAY_MAX_SECONDS",
                (int, float), min_val=5, context="BOT_CONFIG")
    check_param(bot_cfg, "WS_STARTUP_DELAY_SECONDS",
                (int, float), min_val=0, context="BOT_CONFIG")
    check_param(bot_cfg, "WS_PING_INTERVAL", int,
                min_val=5, context="BOT_CONFIG")
    check_param(bot_cfg, "WS_PING_TIMEOUT", int,
                min_val=1, context="BOT_CONFIG")
    check_param(bot_cfg, "WS_MONITOR_CHECK_INTERVAL_SECONDS",
                (int, float), min_val=1, context="BOT_CONFIG")
    check_param(bot_cfg, "INITIAL_FETCH_DELAY_PER_SYMBOL_MS",
                int, min_val=0, context="BOT_CONFIG")
    check_param(bot_cfg, "COOLDOWN_PERIOD_SECONDS",
                int, min_val=0, context="BOT_CONFIG")
    check_param(bot_cfg, "CLOSE_POSITIONS_ON_SHUTDOWN",
                bool, context="BOT_CONFIG")
    check_param(bot_cfg, "API_RECV_WINDOW", int, min_val=1000,
                max_val=60000, context="BOT_CONFIG")
    check_param(bot_cfg, "HIGHER_TF_CACHE_SECONDS",
                int, min_val=60, context="BOT_CONFIG")
    check_param(bot_cfg, "MAX_CLOCK_SKEW_SECONDS",
                int, min_val=5, context="BOT_CONFIG")
    check_param(bot_cfg, "KLINE_STALENESS_SECONDS",
                int, min_val=30, context="BOT_CONFIG") # Added validation

    # --- Validate RISK_CONFIG ---
    risk_cfg = config.get("RISK_CONFIG", {})
    check_param(risk_cfg, "RISK_PER_TRADE_PERCENT", (int, float),
                min_val=0.01, max_val=100, context="RISK_CONFIG")
    check_param(risk_cfg, "ATR_MULTIPLIER_SL", (int, float),
                min_val=0.1, context="RISK_CONFIG")
    check_param(risk_cfg, "ATR_MULTIPLIER_TP", (int, float),
                min_val=0.0, context="RISK_CONFIG")  # 0 means no TP based on ATR
    check_param(risk_cfg, "FEE_RATE", (int, float), min_val=0.0,
                max_val=0.1, context="RISK_CONFIG")  # Realistic fee range
    check_param(risk_cfg, "DRY_RUN_DUMMY_BALANCE", (int, float),
                min_val=0.0, context="RISK_CONFIG")
    # Optional max position size - check if present, then validate
    if "MAX_POSITION_USDT" in risk_cfg:
        check_param(risk_cfg, "MAX_POSITION_USDT", (int, float),
                    min_val=1.0, context="RISK_CONFIG", is_required=False)

    # --- Final Summary ---
    if is_valid:
        log_console(logging.INFO, "Configuration validation successful.")
    else:
        log_console(
            logging.CRITICAL, "Configuration validation failed. Please check the errors above.")

    return is_valid

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
        log_console(logging.DEBUG,
                    "Super Smoother: Input series is invalid or empty.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)
    if not isinstance(period, int) or period < 2:
        log_console(
            logging.WARNING, f"Super Smoother: Period must be an integer >= 2, got {period}. Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length required for calculation start
    series_cleaned = series.dropna()
    min_len_needed = 2  # Needs current and previous price, and two previous SS values
    if len(series_cleaned) < min_len_needed:
        log_console(
            logging.DEBUG, f"Super Smoother: Series length ({len(series_cleaned)} non-NaN) is less than required minimum ({min_len_needed}). Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        # Ehlers formula constants
        # Ensure period is float for calculations involving pi
        period_f = float(period)
        a1 = math.exp(-math.pi * math.sqrt(2.0) / period_f)
        b1 = 2.0 * a1 * math.cos(math.pi * math.sqrt(2.0) / period_f)
        c1 = -(a1 * a1)  # Simplified: -a1**2
        coeff1 = (1.0 - b1 - c1) / 2.0

        # Ensure float type, ffill cautiously to handle potential leading NaNs
        # Using .values converts to numpy array for potentially faster iteration
        series_values = series.astype(float).ffill().values
        # Initialize output array with NaNs
        ss = np.full(len(series_values), np.nan, dtype=np.float64)

        # Initialize first two values robustly after ffill
        # Check if values exist and are not NaN before assigning
        if len(series_values) > 0 and not np.isnan(
                series_values[0]):
            ss[0] = series_values[0]
        if len(series_values) > 1 and not np.isnan(
                series_values[1]):
            ss[1] = series_values[1]

        # Iterative filter calculation loop
        for i in range(2, len(series_values)):
            # Rigorous NaN check for all required inputs at step 'i'
            # Check current price inputs (i, i-1) and previous SS outputs (i-1, i-2)
            if np.isnan(series_values[i]) or np.isnan(series_values[i - 1]) or \
               np.isnan(ss[i - 1]) or np.isnan(ss[i - 2]):
                # If any input is NaN, the result for this step remains NaN (as initialized).
                continue  # Skip calculation for this step

            # Extract values for calculation clarity
            prev_ss = ss[i - 1]
            prev_prev_ss = ss[i - 2]
            current_price_avg = (series_values[i] + series_values[i - 1]) / 2.0
            # Apply the Super Smoother formula
            ss[i] = coeff1 * current_price_avg + \
                b1 * prev_ss + c1 * prev_prev_ss

        # Return result as a pandas Series with the original index
        return pd.Series(ss, index=series.index, dtype=np.float64)
    except Exception as e:
        log_console(
            logging.ERROR, f"Super Smoother calculation error for period {period}: {e}", exc_info=True)
        # Return NaNs on unexpected error
        return pd.Series(np.nan, index=series.index, dtype=np.float64)


def instantaneous_trendline(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates Ehlers Instantaneous Trendline for a given series.
    Requires minimum period of 4 (based on formula dependencies).
    Uses a fixed alpha as commonly recommended by Ehlers. Handles NaNs robustly.
    Note: This indicator can be sensitive to noise and parameter tuning.

    Args:
        series: pandas Series of price data.
        period: Lookback period (dominant cycle period, must be an integer >= 4).

    Returns:
        pandas Series with Instantaneous Trendline values, or Series of NaNs on error/invalid input.
    """
    if not isinstance(series, pd.Series) or series.empty:
        log_console(
            logging.DEBUG, "Instantaneous Trendline: Input series is invalid or empty.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Minimum period requirement based on formula structure
    required_min_period = 4
    if not isinstance(period, int) or period < required_min_period:
        log_console(
            logging.WARNING, f"Instantaneous Trendline: Period must be an integer >= {required_min_period}, got {period}. Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    # Check minimum non-NaN length needed for the calculation to start
    series_cleaned = series.dropna()
    # The formula at index 'i' depends on prices i, i-1, i-2 and IT values i-1, i-2.
    # The loop starts at i=4, requiring valid data up to index 3 and valid IT up to index 3.
    # Initialization handles up to IT[3], which depends on prices up to index 3.
    min_len_needed = 4
    if len(series_cleaned) < min_len_needed:
        log_console(
            logging.DEBUG, f"Instantaneous Trendline: Series length ({len(series_cleaned)} non-NaN) is less than required minimum ({min_len_needed}). Returning NaNs.")
        return pd.Series(np.nan, index=series.index, dtype=np.float64)

    try:
        # Fixed alpha parameter as commonly used in Ehlers' IT formula (adjust if testing variations)
        alpha = 0.07
        alpha_sq = alpha * alpha  # Pre-calculate for efficiency

        # Ensure float type, ffill cautiously, use numpy array for iteration
        series_values = series.astype(float).ffill().values
        # Initialize output array with NaNs
        it = np.full(len(series_values), np.nan, dtype=np.float64)

        # Initialize first few values robustly based on common Ehlers practice after ffill
        # Check source values are not NaN before using them for initialization
        if len(series_values) > 0 and not np.isnan(
                series_values[0]):
            it[0] = series_values[0]
        if len(series_values) > 1 and not np.isnan(
                series_values[1]):
            it[1] = series_values[1]
        # Requires indices 0, 1, 2 to be valid
        if len(series_values) > 2 and not np.any(np.isnan(series_values[0:3])):
            it[2] = (series_values[2] + 2.0 *
                     series_values[1] + series_values[0]) / 4.0
        # Requires indices 1, 2, 3 to be valid
        if len(series_values) > 3 and not np.any(np.isnan(series_values[1:4])):
            it[3] = (series_values[3] + 2.0 *
                     series_values[2] + series_values[1]) / 4.0

        # Iterative calculation loop - check for NaNs rigorously
        # Starts from index 4, as it[i] depends on it[i-1] and it[i-2]
        for i in range(4, len(series_values)):
            # Check all required price inputs (i, i-1, i-2) and previous IT outputs (i-1, i-2)
            if np.isnan(series_values[i]) or np.isnan(series_values[i - 1]) or \
               np.isnan(series_values[i - 2]) or np.isnan(it[i - 1]) or np.isnan(it[i - 2]):
                # If any input is NaN, the result for this step remains NaN.
                continue  # Skip calculation for this step

            # Extract values for calculation clarity
            price_i = series_values[i]
            price_im1 = series_values[i - 1]
            price_im2 = series_values[i - 2]
            it_im1 = it[i - 1]
            it_im2 = it[i - 2]
            one_minus_alpha = 1.0 - alpha
            one_minus_alpha_sq = one_minus_alpha * one_minus_alpha

            # Ehlers Instantaneous Trendline formula (verify against trusted source if needed):
            it[i] = (alpha - alpha_sq / 4.0) * price_i + \
                    (alpha_sq / 2.0) * price_im1 - \
                    (alpha - 3.0 * alpha_sq / 4.0) * price_im2 + \
                2.0 * one_minus_alpha * it_im1 - \
                one_minus_alpha_sq * it_im2

        # Return result as a pandas Series with the original index
        return pd.Series(it, index=series.index, dtype=np.float64)
    except Exception as e:
        log_console(
            logging.ERROR, f"Instantaneous Trendline calculation error for period {period}: {e}", exc_info=True)
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
        The index of the returned DataFrame is guaranteed to be a UTC DatetimeIndex.
    """
    # Input Validation
    internal_cfg = strategy_cfg.get("INTERNAL", {})
    # Use the dynamically calculated KLINE_LIMIT stored in the symbol's internal config
    # Fallback to a reasonable default if missing (should not happen if load_config ran correctly)
    kline_limit = internal_cfg.get("KLINE_LIMIT", 150)
    required_cols = ["open", "high", "low", "close", "volume"]

    if df is None or df.empty:
        log_console(logging.DEBUG, "Indicator Calc: Input DataFrame is empty.")
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            log_console(
                logging.DEBUG, "Indicator Calc: DataFrame index is not DatetimeIndex, attempting conversion to UTC.")
            # Convert index to datetime, coercing errors, setting timezone to UTC
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            if not isinstance(df.index, pd.DatetimeIndex):  # Check conversion success
                raise ValueError("Index conversion to DatetimeIndex failed.")
            if df.index.isnull().any():
                log_console(
                    logging.WARNING, "Indicator Calc: Found NaT values in index after conversion. Dropping affected rows.")
                # Drop rows with NaT index or NaN in required columns
                df.dropna(axis=0, how='any', subset=required_cols +
                          ([df.index.name] if df.index.name else []), inplace=True) # Handle unnamed index
                if df.empty:
                    log_console(
                        logging.ERROR, "Indicator Calc: DataFrame empty after dropping rows with NaT index or NaN data.")
                    return None
        except Exception as idx_e:
            log_console(
                logging.ERROR, f"Indicator Calc: Failed to convert DataFrame index to DatetimeIndex: {idx_e}")
            return None
    elif not df.index.tz:
        log_console(
            logging.DEBUG, "Indicator Calc: DataFrame index lacks timezone. Assuming and setting UTC.")
        # Use tz_localize carefully, handle potential errors if already localized
        try:
            df.index = df.index.tz_localize('UTC', ambiguous='infer')
        except TypeError as tz_err:
            log_console(
                logging.WARNING, f"Indicator Calc: Index might already be timezone-aware? Error during tz_localize: {tz_err}. Proceeding.")
            if df.index.tz != dt.timezone.utc:  # If localized but not UTC, convert
                try:
                    df.index = df.index.tz_convert('UTC')
                except Exception as tz_conv_err:
                    log_console(
                        logging.ERROR, f"Indicator Calc: Failed to convert existing timezone to UTC: {tz_conv_err}. Cannot proceed.", exc_info=True)
                    return None
    elif df.index.tz != dt.timezone.utc:
        log_console(
            logging.DEBUG, f"Indicator Calc: DataFrame index has timezone {df.index.tz}, converting to UTC.")
        try:
            df.index = df.index.tz_convert('UTC')
        except Exception as tz_conv_err:
            log_console(
                logging.ERROR, f"Indicator Calc: Failed to convert existing index timezone ({df.index.tz}) to UTC: {tz_conv_err}. Cannot proceed.", exc_info=True)
            return None

    # Use a copy to avoid modifying the original DataFrame passed to the function
    df_out = df.copy()

    # Attempt numeric conversion for required columns and check length *after* potential NaN drop
    try:
        for col in required_cols:
            if col not in df_out.columns:
                log_console(
                    logging.ERROR, f"Indicator Calc: Missing required column: '{col}'. Cannot proceed.")
                return None  # Critical column missing
            if not pd.api.types.is_numeric_dtype(df_out[col]):
                log_console(
                    logging.DEBUG, f"Indicator Calc: Converting non-numeric column '{col}' to numeric.")
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

        # Drop rows where essential conversions failed (resulted in NaNs in required cols)
        initial_len = len(df_out)
        df_out.dropna(subset=required_cols, inplace=True)
        if len(df_out) < initial_len:
            log_console(
                logging.WARNING, f"Indicator Calc: Dropped {initial_len - len(df_out)} rows due to NaN values in required columns after numeric conversion.")

        # Check length requirements AFTER cleaning numeric data, using the calculated KLINE_LIMIT
        if len(df_out) < kline_limit:
            log_console(
                logging.DEBUG, f"Indicator Calc: Insufficient valid data points after cleaning ({len(df_out)} < {kline_limit}). Need more historical data.")
            return None
        if df_out.empty:
            log_console(
                logging.ERROR, "Indicator Calc: DataFrame empty after NaN drop from numeric conversion.")
            return None

    except Exception as e:
        log_console(
            logging.ERROR, f"Indicator Calc: Failed during data validation/numeric conversion: {e}", exc_info=True)
        return None

    # --- Calculate Indicators ---
    try:
        price_source_key = strategy_cfg.get("PRICE_SOURCE", "close")
        # Calculate common price sources if specified
        if price_source_key == 'hlc3':
            df_out['hlc3'] = (
                df_out['high'] + df_out['low'] + df_out['close']) / 3.0
            price_source_col = 'hlc3'
        elif price_source_key == 'ohlc4':
            df_out['ohlc4'] = (df_out['open'] + df_out['high'] +
                               df_out['low'] + df_out['close']) / 4.0
            price_source_col = 'ohlc4'
        elif price_source_key in df_out.columns:
            # Use directly if column exists (e.g., 'open', 'high', 'low', 'close')
            price_source_col = price_source_key
        else:
            log_console(
                logging.ERROR, f"Indicator Calc: Specified price source '{price_source_key}' not found or calculable. Falling back to 'close'.")
            if 'close' not in df_out.columns:
                log_console(
                    logging.ERROR, "Indicator Calc: Fallback 'close' column missing. Cannot proceed.")
                return None  # Cannot proceed without a valid price source
            price_source_col = 'close'

        # Check if the selected price source column has valid data
        if price_source_col not in df_out or df_out[price_source_col].isnull().all():
            log_console(
                logging.ERROR, f"Indicator Calc: Price source column '{price_source_col}' is missing or contains only NaNs.")
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
        calculated_ma_cols = []  # Keep track of successfully calculated MA columns

        for period_key, base_col_name in smoother_params:
            length = strategy_cfg.get(period_key)
            col_name = base_col_name.replace("ema", smoother_prefix)
            # Initialize column with NaN first
            df_out[col_name] = np.nan
            if not isinstance(length, int) or length <= 0:
                log_console(
                    logging.ERROR, f"Indicator Calc: Invalid or missing {period_key} in strategy config. Value: {length}. Skipping {col_name}.")
                continue  # Skip calculation for this MA

            try:
                if use_ehlers:
                    indicator_series = super_smoother(
                        df_out[price_source_col], length)
                else:
                    indicator_series = ta.ema(
                        df_out[price_source_col], length=length)

                # Check result validity before assigning
                if indicator_series is None or indicator_series.empty or indicator_series.isnull().all():
                    log_console(
                        logging.WARNING, f"Indicator Calc: {col_name} (length {length}) calculation resulted in empty or all NaNs.")
                    # Column already initialized to NaN
                else:
                    df_out[col_name] = indicator_series
                    # Add to list of successful calcs
                    calculated_ma_cols.append(col_name)
            except Exception as calc_e:
                log_console(
                    logging.ERROR, f"Indicator Calc: Error calculating {col_name} (length {length}): {calc_e}", exc_info=False)
                # Column remains NaN

        # --- Instantaneous Trendline ---
        it_period = strategy_cfg.get("INSTANTANEOUS_TRENDLINE_PERIOD", 20)
        df_out["trendline"] = np.nan  # Initialize column
        # Check minimum period requirement
        if not isinstance(it_period, int) or it_period < 4:
            log_console(
                logging.ERROR, f"Indicator Calc: Invalid INSTANTANEOUS_TRENDLINE_PERIOD ({it_period}). Must be integer >= 4. Skipping calculation.")
        else:
            try:
                it_series = instantaneous_trendline(
                    df_out[price_source_col], it_period)
                if it_series is None or it_series.empty or it_series.isnull().all():
                    log_console(
                        logging.WARNING, "Indicator Calc: Instantaneous Trendline calculation resulted in empty or all NaNs.")
                    # Column already initialized to NaN
                else:
                    df_out["trendline"] = it_series
            except Exception as calc_e:
                log_console(
                    logging.ERROR, f"Indicator Calc: Error calculating Instantaneous Trendline: {calc_e}", exc_info=False)
                # Column already initialized to NaN

        # --- Other Indicators (wrapped calculations) ---
        # Initialize all expected indicator columns to NaN first
        other_indicator_cols = ["roc", "roc_smooth", "atr_risk", "norm_atr",
                                "rsi", "volume_sma", "volume_percentile_75", "adx", "bb_width"]
        for col in other_indicator_cols:
            df_out[col] = np.nan

        try:
            roc_period = strategy_cfg.get("ROC_PERIOD", 5)
            if roc_period > 0:
                roc_raw = ta.roc(df_out[price_source_col], length=roc_period)
                if roc_raw is not None:
                    df_out["roc"] = roc_raw
                # Fill NaNs before smoothing ROC, use 0.0 as neutral momentum
                roc_filled = df_out["roc"].fillna(0.0)
                # Fixed 3-period smoothing for ROC
                roc_smooth_series = ta.sma(roc_filled, length=3)
                if roc_smooth_series is not None:
                    df_out["roc_smooth"] = roc_smooth_series
            else:
                raise ValueError("ROC_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR, f"Indicator Calc: Error calculating ROC: {e}", exc_info=False)
            # Columns remain NaN

        try:
            atr_period_risk = strategy_cfg.get("ATR_PERIOD_RISK", 14)
            if atr_period_risk > 0:
                atr_series = ta.atr(
                    df_out["high"], df_out["low"], df_out["close"], length=atr_period_risk)
                if atr_series is not None:
                    # Use .ffill() as per pandas_ta recommendation, then fill remaining start NaNs with 0
                    df_out["atr_risk"] = atr_series.ffill().fillna(0.0)
                    # Calculate Normalized ATR safely
                    # Avoid division by zero/NaN: replace 0 close with NaN, ffill, then fill remaining start NaNs with 1.0
                    close_safe = df_out["close"].replace(
                        0, np.nan).ffill().fillna(1.0)
                    norm_atr_series = (df_out["atr_risk"] / close_safe) * 100
                    df_out["norm_atr"] = norm_atr_series.ffill().fillna(0.0)
            else:
                raise ValueError("ATR_PERIOD_RISK must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR, f"Indicator Calc: Error calculating ATR: {e}", exc_info=False)
            # Columns remain NaN

        try:
            rsi_period = strategy_cfg.get("RSI_PERIOD", 10)
            if rsi_period > 0:
                rsi_series = ta.rsi(
                    df_out[price_source_col], length=rsi_period)
                # Handle NaNs later in bulk fill
                if rsi_series is not None:
                    df_out["rsi"] = rsi_series
            else:
                raise ValueError("RSI_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR, f"Indicator Calc: Error calculating RSI: {e}", exc_info=False)
            # Column remains NaN

        try:
            vol_period = strategy_cfg.get("VOLUME_SMA_PERIOD", 20)
            if vol_period > 0:
                vol_sma_series = ta.sma(df_out["volume"], length=vol_period)
                # Handle NaNs later
                if vol_sma_series is not None:
                    df_out["volume_sma"] = vol_sma_series
                # Ensure min_periods is at least 1 for rolling quantile
                min_periods_vol = max(1, vol_period // 2)
                # Ensure min_periods for rolling quantile, handle NaNs later
                vol_perc_series = df_out["volume"].rolling(
                    window=vol_period, min_periods=min_periods_vol).quantile(0.75)
                if vol_perc_series is not None:
                    df_out["volume_percentile_75"] = vol_perc_series
            else:
                raise ValueError("VOLUME_SMA_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR, f"Indicator Calc: Error calculating Volume Indicators: {e}", exc_info=False)
            # Columns remain NaN

        try:
            adx_period = strategy_cfg.get("ADX_PERIOD", 14)
            if adx_period > 0:
                adx_df = ta.adx(df_out["high"], df_out["low"],
                                df_out["close"], length=adx_period)
                # pandas_ta ADX column name convention: ADX_{length}
                adx_col_name = f"ADX_{adx_period}"
                if adx_df is not None and not adx_df.empty and adx_col_name in adx_df.columns:
                    df_out["adx"] = adx_df[adx_col_name]  # Handle NaNs later
                else:
                    log_console(
                        logging.WARNING, f"Indicator Calc: ADX ({adx_col_name}) calculation failed or returned invalid DataFrame.")
                    # Column remains NaN
            else:
                raise ValueError("ADX_PERIOD must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR, f"Indicator Calc: Error calculating ADX: {e}", exc_info=False)
            # Column remains NaN

        try:
            bb_period = strategy_cfg.get("BBANDS_PERIOD", 20)
            bb_std = strategy_cfg.get("BBANDS_STDDEV", 2.0)
            if bb_period > 0 and bb_std > 0:
                bbands = ta.bbands(
                    df_out["close"], length=bb_period, std=bb_std)
                # Define expected column names from pandas_ta bbands (check version if names change)
                # Note: pandas_ta often uses float in col name, format accordingly
                bbu_col = f"BBU_{bb_period}_{bb_std:.1f}"
                bbl_col = f"BBL_{bb_period}_{bb_std:.1f}"
                bbm_col = f"BBM_{bb_period}_{bb_std:.1f}"
                if bbands is not None and not bbands.empty and all(c in bbands.columns for c in [bbu_col, bbl_col, bbm_col]):
                    # Calculate BB width safely, handle NaNs and potential division by zero later during fill
                    # Replace 0 with NaN before division
                    bbm_safe = bbands[bbm_col].replace(0, np.nan)
                    df_out["bb_width"] = (
                        bbands[bbu_col] - bbands[bbl_col]) / bbm_safe * 100
                else:
                    log_console(
                        logging.WARNING, f"Indicator Calc: Bollinger Bands calculation failed or missing expected columns (e.g., {bbu_col}). Check pandas_ta version/output.")
                    # Column remains NaN
            else:
                raise ValueError("BBANDS_PERIOD and BBANDS_STDDEV must be > 0")
        except Exception as e:
            log_console(
                logging.ERROR, f"Indicator Calc: Error calculating Bollinger Bands: {e}", exc_info=False)
            # Column remains NaN

        # --- Fill NaNs in Key Numeric Columns BEFORE calculating signals ---
        # Forward fill first to propagate last valid values, then fill remaining (start of series) with appropriate defaults.
        # Ensure MA/Smoother columns exist before trying to fill them (use calculated_ma_cols list)
        # Default fill for MAs (price=0), adjust if needed
        ma_cols_to_fill = {ma_col: 0.0 for ma_col in calculated_ma_cols}

        # Define fill values for other indicators
        numeric_cols_fill_values = {
            "atr_risk": 0.0, "norm_atr": 0.0, "roc": 0.0, "roc_smooth": 0.0,
            "rsi": 50.0,  # Neutral RSI
            "adx": 20.0,  # Often considered baseline for trend strength
            "bb_width": 0.0,  # Default width if calculation failed or at start
            "volume_sma": 0.0, "volume_percentile_75": 0.0,
            # Fill trendline with price source if available and calculated, else 0
            "trendline": df_out[price_source_col] if "trendline" in df_out and not df_out["trendline"].isnull().all() else 0.0,
            **ma_cols_to_fill  # Add the dynamically determined MA columns and their fill value
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
            df_out["volume"] = df_out["volume"].fillna(
                0.0)  # Fill volume NaNs before comparison
            df_out["high_volume"] = (df_out["volume"] > df_out["volume_percentile_75"]) & \
                (df_out["volume"] > df_out["volume_sma"])
            # Ensure the result is boolean and any remaining NaNs (shouldn't happen) become False
            df_out["high_volume"] = df_out["high_volume"].fillna(
                False).astype(bool)
        else:
            log_console(
                logging.WARNING, "Indicator Calc: Volume columns missing or failed calculation, cannot calculate 'high_volume'. Defaulting to False.")
            # Column already initialized to False

        # --- Dynamic Thresholds based on Volatility ---
        atr_period_volatility = strategy_cfg.get("ATR_PERIOD_VOLATILITY", 14)
        volatility_factor = pd.Series(
            1.0, index=df_out.index)  # Default neutral factor

        if "norm_atr" in df_out and atr_period_volatility > 0:
            min_periods_vol_atr = max(1, atr_period_volatility // 2)
            # Calculate rolling mean on norm_atr (already filled), ffill initial NaNs, default remaining to 1.0
            rolling_volatility = df_out["norm_atr"].rolling(
                window=atr_period_volatility, min_periods=min_periods_vol_atr).mean().ffill().fillna(1.0)
            # Clip volatility factor to prevent extreme adjustments
            volatility_factor = np.clip(rolling_volatility,
                                        strategy_cfg.get(
                                            "VOLATILITY_FACTOR_MIN", 0.5),
                                        strategy_cfg.get("VOLATILITY_FACTOR_MAX", 2.0))
            # Ensure index alignment if needed (should be fine if calculated on df_out["norm_atr"])
            volatility_factor = pd.Series(
                volatility_factor, index=df_out.index)
        else:
            log_console(
                logging.DEBUG, "Indicator Calc: Cannot calculate volatility factor (norm_atr missing or period invalid). Using default factor 1.0.")
            # volatility_factor is already initialized to 1.0 Series

        # Calculate dynamic RSI thresholds
        rsi_base_low = strategy_cfg.get("RSI_LOW_THRESHOLD", 40)
        rsi_base_high = strategy_cfg.get("RSI_HIGH_THRESHOLD", 75)
        rsi_vol_adjust = strategy_cfg.get("RSI_VOLATILITY_ADJUST", 5)
        # Use pd.Series constructor to ensure index alignment and apply clipping
        rsi_low = pd.Series(np.clip(
            rsi_base_low - (rsi_vol_adjust * volatility_factor), 10, 50), index=df_out.index)
        rsi_high = pd.Series(np.clip(
            rsi_base_high + (rsi_vol_adjust * volatility_factor), 60, 90), index=df_out.index)

        # Calculate dynamic ROC threshold
        roc_base_threshold = strategy_cfg.get("MOM_THRESHOLD_PERCENT", 0.1)
        # Use pd.Series constructor, take absolute value, apply factor, handle potential NaNs from factor
        roc_threshold = pd.Series(abs(roc_base_threshold) * volatility_factor,
                                  index=df_out.index).fillna(abs(roc_base_threshold))

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
        # Trendline might not be in calculated_ma_cols, check separately
        trend_ma_cols_ok = all(col in calculated_ma_cols for col in [
                               fast_ma_col, mid_ma_col, slow_ma_col])
        trendline_ok = trendline_col in df_out.columns and not df_out[trendline_col].isnull(
        ).all()

        if not trend_ma_cols_ok or not trendline_ok:
            log_console(
                logging.WARNING, f"Indicator Calc: Cannot define trend. MA cols OK: {trend_ma_cols_ok}, Trendline OK: {trendline_ok}. Assuming neutral trend.")
        else:
            # NaNs should already be filled in MA/trendline columns
            # Calculate shifted trendline, fill initial shift NaN using backfill then forward fill
            trendline_shift = df_out[trendline_col].shift(
                1).fillna(method='bfill').fillna(method='ffill')

            # Trend Up Condition: Check MA alignment and trendline slope using tolerance
            trend_up_cond = (
                (df_out[fast_ma_col] > df_out[mid_ma_col]) &
                (df_out[mid_ma_col] > df_out[slow_ma_col]) &
                # Use tolerance for comparing trendline to its shifted value
                (df_out[trendline_col] > trendline_shift *
                 (1 + FLOAT_COMPARISON_TOLERANCE))
            )
            df_out["trend_up"] = trend_up_cond.fillna(False).astype(bool)

            # Trend Down Condition
            trend_down_cond = (
                (df_out[fast_ma_col] < df_out[mid_ma_col]) &
                (df_out[mid_ma_col] < df_out[slow_ma_col]) &
                (df_out[trendline_col] < trendline_shift *
                 (1 - FLOAT_COMPARISON_TOLERANCE))
            )
            df_out["trend_down"] = trend_down_cond.fillna(False).astype(bool)

            # Trend Neutral if neither up nor down
            df_out["trend_neutral"] = ~(
                df_out["trend_up"] | df_out["trend_down"])

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
        entry_ma_cols_ok = all(ma_col in calculated_ma_cols for ma_col in [
                               fast_ma_col, mid_ma_col, ultra_fast_ma_col] if ma_col in entry_req_cols)
        entry_other_cols_ok = all(col in df_out.columns for col in entry_req_cols if col not in [
                                  fast_ma_col, mid_ma_col, ultra_fast_ma_col])
        required_available = entry_ma_cols_ok and entry_other_cols_ok

        if not required_available:
            missing_cols = [col for col in entry_req_cols if col not in df_out.columns or (
                col in [fast_ma_col, mid_ma_col, ultra_fast_ma_col] and col not in calculated_ma_cols)]
            log_console(
                logging.WARNING, f"Indicator Calc: Cannot generate entry signals, required columns missing or failed calculation: {missing_cols}")
        else:
            # Ensure trendline_shift is available (recalculate or use previous one if available)
            if 'trendline_shift' not in locals():  # Check if calculated in trend section
                if trendline_ok:  # Only recalculate if trendline itself is valid
                    trendline_shift = df_out[trendline_col].shift(
                        1).fillna(method='bfill').fillna(method='ffill')
                else:  # Cannot calculate shift if trendline is invalid
                    log_console(
                        logging.WARNING, "Indicator Calc: Trendline invalid, cannot calculate trendline shift for entry signals.")
                    # Set shift to something that won't trigger signals based on it
                    trendline_shift = pd.Series(0.0, index=df_out.index)

            # --- Long Entry Conditions ---
            # NaNs already handled in source columns by the fill logic above
            long_cond_trend = df_out[fast_ma_col] > df_out[mid_ma_col]
            long_cond_trigger = df_out[ultra_fast_ma_col] > df_out[fast_ma_col]
            # Compare smoothed ROC against the dynamic threshold Series
            long_cond_mom = df_out[roc_smooth_col] > roc_threshold
            # Use dynamic RSI thresholds (Series comparison)
            long_cond_rsi = (df_out[rsi_col] > rsi_low) & (
                df_out[rsi_col] < rsi_high)
            long_cond_vol = df_out[high_volume_col]  # Already boolean
            long_cond_adx = df_out[adx_col] > adx_threshold
            long_cond_itrend = df_out[trendline_col] > trendline_shift * (
                # Only check if trendline valid
                1 + FLOAT_COMPARISON_TOLERANCE) if trendline_ok else False

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
            short_cond_rsi = (df_out[rsi_col] > rsi_low) & (
                df_out[rsi_col] < rsi_high)
            short_cond_vol = df_out[high_volume_col]  # Already boolean
            short_cond_adx = df_out[adx_col] > adx_threshold
            short_cond_itrend = df_out[trendline_col] < trendline_shift * (
                # Only check if trendline valid
                1 - FLOAT_COMPARISON_TOLERANCE) if trendline_ok else False

            # Combine conditions for final short signal
            df_out["short_signal"] = (
                short_cond_trend & short_cond_trigger & short_cond_mom &
                short_cond_rsi & short_cond_vol & short_cond_adx & short_cond_itrend
            ).fillna(False).astype(bool)

            # --- Signal Strength Calculation ---
            # Count how many of the defined conditions are True for each signal type
            long_conditions_list = [long_cond_trend, long_cond_trigger, long_cond_mom,
                                    long_cond_rsi, long_cond_vol, long_cond_adx, long_cond_itrend]
            short_conditions_list = [short_cond_trend, short_cond_trigger, short_cond_mom,
                                     short_cond_rsi, short_cond_vol, short_cond_adx, short_cond_itrend]
            # Should be same for both
            num_conditions = len(long_conditions_list)

            if num_conditions > 0:
                # Sum boolean conditions (True=1, False=0) and normalize by total number of conditions
                df_out["long_signal_strength"] = sum(cond.astype(
                    int) for cond in long_conditions_list) / num_conditions
                df_out["short_signal_strength"] = sum(cond.astype(
                    int) for cond in short_conditions_list) / num_conditions
                # Fill any potential NaNs from the calculation (unlikely but possible) with 0.0
                df_out["long_signal_strength"] = df_out["long_signal_strength"].fillna(
                    0.0)
                df_out["short_signal_strength"] = df_out["short_signal_strength"].fillna(
                    0.0)
            # else: Columns initialized to 0.0

        # --- Exit Signals ---
        # Initialize columns
        df_out["exit_long_signal"] = False
        df_out["exit_short_signal"] = False
        # Check required columns for exit signals
        exit_req_cols = [fast_ma_col, mid_ma_col,
                         roc_smooth_col, trendline_col]
        exit_ma_cols_ok = all(ma_col in calculated_ma_cols for ma_col in [
                              fast_ma_col, mid_ma_col] if ma_col in exit_req_cols)
        exit_other_cols_ok = all(col in df_out.columns for col in exit_req_cols if col not in [
                                 fast_ma_col, mid_ma_col])
        exit_required_available = exit_ma_cols_ok and exit_other_cols_ok

        # --- CORRECTED LOGIC: Check availability BEFORE calculating ---
        if not exit_required_available:
            # Calculate the list of missing columns *inside* the if block
            missing_cols = [
                col for col in exit_req_cols
                if col not in df_out.columns or
                (col in [fast_ma_col, mid_ma_col]
                    and col not in calculated_ma_cols)
            ]
            log_console(
                logging.WARNING, f"Indicator Calc: Cannot generate exit signals, required columns missing or failed calculation: {missing_cols}")
            # Since exit signals cannot be generated, columns remain False (as initialized)
        else:
            # --- Proceed with Exit Signal Calculation ---
            # Ensure trendline_shift is available (recalculate or use previous one if needed)
            if 'trendline_shift' not in locals():
                if trendline_ok:  # Only recalculate if trendline itself is valid
                    trendline_shift = df_out[trendline_col].shift(
                        1).fillna(method='bfill').fillna(method='ffill')
                else:  # Cannot calculate shift if trendline is invalid
                    log_console(
                        logging.WARNING, "Indicator Calc: Trendline invalid, cannot calculate trendline shift for exit signals.")
                    # Use a non-triggering fallback
                    trendline_shift = pd.Series(0.0, index=df_out.index)

            # --- Exit Long Conditions ---
            # Exit if: Fast MA crosses below Mid MA OR Momentum turns negative OR Trendline turns down
            exit_long_cond1 = df_out[fast_ma_col] < df_out[mid_ma_col]
            # Check against zero momentum
            exit_long_cond2 = df_out[roc_smooth_col] < 0
            exit_long_cond3 = df_out[trendline_col] < trendline_shift * \
                (1 - FLOAT_COMPARISON_TOLERANCE) if trendline_ok else False
            df_out["exit_long_signal"] = (
                exit_long_cond1 | exit_long_cond2 | exit_long_cond3).fillna(False).astype(bool)

            # --- Exit Short Conditions ---
            # Exit if: Fast MA crosses above Mid MA OR Momentum turns positive OR Trendline turns up
            exit_short_cond1 = df_out[fast_ma_col] > df_out[mid_ma_col]
            # Check against zero momentum
            exit_short_cond2 = df_out[roc_smooth_col] > 0
            exit_short_cond3 = df_out[trendline_col] > trendline_shift * \
                (1 + FLOAT_COMPARISON_TOLERANCE) if trendline_ok else False
            df_out["exit_short_signal"] = (
                exit_short_cond1 | exit_short_cond2 | exit_short_cond3).fillna(False).astype(bool)

        # --- Final Processing & Validation ---
        # Ensure boolean columns are explicitly boolean type and NaNs are False
        bool_cols = ["long_signal", "short_signal", "exit_long_signal", "exit_short_signal",
                     "trend_up", "trend_down", "trend_neutral", "high_volume"]
        for col in bool_cols:
            if col in df_out.columns:
                # Fill potential NaNs arising from logic gaps before converting type
                df_out[col] = df_out[col].fillna(False).astype(bool)

        # Fill NaNs in strength signals if they exist
        if "long_signal_strength" in df_out:
            df_out["long_signal_strength"] = df_out["long_signal_strength"].fillna(
                0.0)
        if "short_signal_strength" in df_out:
            df_out["short_signal_strength"] = df_out["short_signal_strength"].fillna(
                0.0)

        # Final check for validity - ensure required columns for trading logic exist and have valid data in last few rows
        final_check_cols = ["close", "atr_risk", "long_signal",
                            "short_signal", "exit_long_signal", "exit_short_signal"]
        if df_out.empty or not all(col in df_out.columns for col in final_check_cols):
            log_console(
                logging.ERROR, "Indicator Calc: DataFrame empty or missing critical columns after final processing.")
            return None

        # Check last N rows for NaNs in critical columns (if enough rows exist)
        min_rows_for_check = 2  # Check last closed candle and latest candle
        if len(df_out) >= min_rows_for_check:
            if df_out.iloc[-min_rows_for_check:][final_check_cols].isnull().any().any():
                log_console(
                    logging.ERROR, f"Indicator Calc: Critical columns contain NaN in the last {min_rows_for_check} rows after processing. Check calculations and NaN fills.")
                # Log the problematic rows for debugging
                log_console(
                    logging.DEBUG, f"Last {min_rows_for_check} rows tail with potential NaNs:\n{df_out.iloc[-min_rows_for_check:][final_check_cols]}")
                return None
        elif len(df_out) == 1:  # Handle edge case with only one row
            if df_out.iloc[-1:][final_check_cols].isnull().any().any():
                log_console(
                    logging.ERROR, "Indicator Calc: Critical columns contain NaN in the only row after processing.")
                return None

        # Ensure index is UTC DatetimeIndex before returning (should be handled earlier, but double check)
        if not isinstance(df_out.index, pd.DatetimeIndex) or df_out.index.tz != dt.timezone.utc:
            log_console(
                logging.ERROR, "Indicator Calc: DataFrame index is not UTC DatetimeIndex before return! Attempting final fix.")
            try:
                if isinstance(df_out.index, pd.DatetimeIndex):
                    if df_out.index.tz:
                        df_out.index = df_out.index.tz_convert('UTC')
                    else:
                        df_out.index = df_out.index.tz_localize('UTC')
                else:
                    # Attempt conversion if not already datetime
                    df_out.index = pd.to_datetime(df_out.index, errors='coerce', utc=True)
                    if not isinstance(df_out.index, pd.DatetimeIndex) or df_out.index.isnull().any():
                         raise ValueError("Index conversion failed or resulted in NaT")
            except Exception as final_idx_e:
                 log_console(logging.ERROR, f"Indicator Calc: Cannot fix index before return: {final_idx_e}")
                 return None # Cannot return invalid index


        # Return the DataFrame with all calculated indicators and signals
        return df_out

    except Exception as e:
        log_console(
            logging.ERROR, f"Indicator calculation failed unexpectedly: {e}", exc_info=True)
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
        response = session.get_wallet_balance(
            accountType=account_type.upper(), coin=coin)

        if response and response.get("retCode") == 0:
            result = response.get("result", {})
            list_data = result.get("list", [])

            if not list_data:
                log_console(
                    logging.DEBUG, f"No balance list found for {account_type} account (maybe empty?). Treating balance as 0.")
                return 0.0

            # --- Parsing Logic for V5 'get_wallet_balance' Response ---
            # The structure varies slightly depending on the account type.
            coin_balance_list: List[Dict[str, Any]] = []

            if account_type.upper() == "UNIFIED":
                # For UNIFIED, the coin balances are nested within the first element of 'list'
                account_data = list_data[0] if list_data else {}
                coin_balance_list = account_data.get("coin", [])
                if not isinstance(coin_balance_list, list):
                    log_console(
                        logging.WARNING, f"Expected list for 'coin' in UNIFIED balance response, got {type(coin_balance_list)}. Parsing failed.")
                    return 0.0
            elif account_type.upper() in ["CONTRACT", "SPOT"]:
                # For CONTRACT/SPOT, 'list' *should* contain the coin balances directly (as list of dicts)
                # Spot: list contains dicts per coin. Contract: list contains one dict with 'coin' list inside.
                if account_type.upper() == "SPOT":
                    if isinstance(list_data, list) and all(isinstance(item, dict) for item in list_data):
                        coin_balance_list = list_data  # Assumes direct list of coin dicts for SPOT
                    else:
                        log_console(
                            logging.WARNING, f"Expected list of dicts for 'SPOT' balance response list, got different structure. Parsing failed.")
                        return 0.0
                elif account_type.upper() == "CONTRACT":
                    account_data = list_data[0] if list_data else {}
                    coin_balance_list = account_data.get("coin", [])
                    if not isinstance(coin_balance_list, list):
                        log_console(
                            logging.WARNING, f"Expected list for 'coin' in CONTRACT balance response, got {type(coin_balance_list)}. Parsing failed.")
                        return 0.0
            else:
                # Handle unknown account types with a warning, attempt UNIFIED structure as fallback
                log_console(
                    logging.WARNING, f"Balance parsing logic not explicitly defined for account type: {account_type}. Attempting UNIFIED structure.")
                account_data = list_data[0] if list_data else {}
                coin_balance_list = account_data.get("coin", [])
                if not isinstance(coin_balance_list, list):
                    log_console(
                        logging.WARNING, f"Fallback parsing for '{account_type}' failed (expected list for 'coin').")
                    return 0.0

            # If after parsing, coin_balance_list is empty
            if not coin_balance_list:
                log_console(
                    logging.WARNING, f"No coin balance details found within balance data structure for {account_type}.")
                return 0.0

            # Iterate through the found coin balances to find the requested coin
            for coin_data in coin_balance_list:
                if isinstance(coin_data, dict) and coin_data.get("coin") == coin:
                    # Prioritize V5 fields representing usable balance:
                    # 'availableToWithdraw' or 'availableBalance' are typically the most relevant.
                    # Use availableBalance first as it's generally the spendable balance for trading.
                    balance_str = coin_data.get("availableBalance")
                    source = "availableBalance"
                    if balance_str is None or balance_str == '':
                        # Fallback to withdrawable balance if availableBalance is missing
                        balance_str = coin_data.get("availableToWithdraw")
                        source = "availableToWithdraw"

                    # Fallback to 'walletBalance' (total balance) if others are missing/empty
                    if balance_str is None or balance_str == '':
                        balance_str = coin_data.get(
                            "walletBalance")  # General total balance
                        source = "walletBalance"
                        log_console(
                            logging.DEBUG, f"'availableBalance'/'availableToWithdraw' empty for {coin}, using '{source}' as fallback.")

                    # Handle case where all relevant balance fields are empty or missing
                    if balance_str is None or balance_str == '':
                        log_console(
                            logging.WARNING, f"Could not find a valid available balance field (availableBalance, availableToWithdraw, walletBalance) for {coin}. Balance is 0.")
                        balance_str = "0"  # Treat as zero if no field found

                    # Convert the found balance string to float safely
                    try:
                        balance_float = float(balance_str)
                        # Ensure balance is not negative (can happen with negative PnL in some views)
                        final_balance = max(0.0, balance_float)
                        log_console(
                            logging.DEBUG, f"Found balance for {coin} ({source}): {final_balance:.8f}")
                        return final_balance
                    except (ValueError, TypeError) as e:
                        log_console(
                            logging.ERROR, f"Could not convert balance string '{balance_str}' from field '{source}' to float for {coin}: {e}")
                        return 0.0  # Return 0 if conversion fails

            # If loop completes without finding the coin
            log_console(
                logging.WARNING, f"Coin '{coin}' not found in the wallet balance details for {account_type} account.")
            return 0.0

        else:  # API call failed (retCode != 0 or empty response)
            error_msg = response.get(
                'retMsg', 'Unknown error') if response else "Empty/invalid response"
            error_code = response.get('retCode', 'N/A')
            log_console(
                logging.ERROR, f"Failed to fetch wallet balance: {error_msg} (Code: {error_code})")
            return 0.0

    except Exception as e:
        log_console(
            logging.ERROR, f"Exception occurred while fetching wallet balance: {e}", exc_info=True)
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
        log_console(
            logging.DEBUG, f"Position Size Calc: Invalid input types. Bal:{type(balance)}, Risk%:{type(risk_percent)}, SLDist:{type(sl_distance_price)}, Entry:{type(entry_price)}, MinQty:{type(min_order_qty)}, Step:{type(qty_step_float)}. Returning size 0.")
        return 0.0
    if not isinstance(qty_precision, int) or qty_precision < 0:
        log_console(
            logging.ERROR, f"Position Size Calc: Invalid qty_precision ({qty_precision}). Must be non-negative integer. Returning size 0.")
        return 0.0
    # Check positive values for required inputs
    if balance <= FLOAT_COMPARISON_TOLERANCE or risk_percent <= FLOAT_COMPARISON_TOLERANCE or \
       entry_price <= FLOAT_COMPARISON_TOLERANCE or min_order_qty <= FLOAT_COMPARISON_TOLERANCE or \
       qty_step_float <= FLOAT_COMPARISON_TOLERANCE:
        # Use DEBUG as this might happen normally with zero balance or zero risk setting
        log_console(
            logging.DEBUG, f"Position Size Calc: Non-positive input values detected. Balance={balance:.2f}, Risk%={risk_percent}, Entry={entry_price}, MinQty={min_order_qty}, QtyStep={qty_step_float}. Returning size 0.")
        return 0.0
    # Prevent division by zero or huge sizes from tiny SL distance
    if sl_distance_price <= FLOAT_COMPARISON_TOLERANCE:
        log_console(
            logging.WARNING, f"Position Size Calc: Stop loss distance ({sl_distance_price}) is zero or too small. Cannot calculate size reliably. Returning 0.")
        return 0.0
    # Validate optional max position value if provided
    if max_position_usdt is not None and (not isinstance(max_position_usdt, (int, float)) or max_position_usdt <= FLOAT_COMPARISON_TOLERANCE):
        log_console(
            logging.WARNING, f"Position Size Calc: Invalid max_position_usdt ({max_position_usdt}). Ignoring constraint.")
        max_position_usdt = None  # Ignore invalid value

    # --- Calculation ---
    # Calculate risk amount in USDT
    risk_amount_usdt = balance * (risk_percent / 100.0)
    log_console(
        logging.DEBUG, f"Position Size Calc: Risk Amount = {risk_amount_usdt:.4f} USDT (Balance: {balance:.2f}, Risk%: {risk_percent})")

    # Calculate initial position size based on risk amount and SL distance per unit
    # This assumes linear contracts where PnL = Size * PriceDiff
    try:
        position_size = risk_amount_usdt / sl_distance_price
        log_console(
            logging.DEBUG, f"Position Size Calc: Initial raw size (Risk/SL Dist): {position_size:.{qty_precision+4}f} (SL Dist: {sl_distance_price:.{qty_precision+4}f})")
    except ZeroDivisionError:  # Should be caught by check above, but safeguard
        log_console(
            logging.WARNING, "Position Size Calc: SL distance is effectively zero during calculation. Cannot calculate size.")
        return 0.0

    # --- Apply Constraints ---
    # 1. Max Position Value Constraint (if applicable)
    if max_position_usdt is not None:
        try:
            # Ensure entry price is valid for this calculation
            if entry_price <= FLOAT_COMPARISON_TOLERANCE:
                log_console(
                    logging.WARNING, "Position Size Calc: Entry price is zero or negative. Cannot apply Max Position USDT constraint.")
            else:
                max_size_by_value = max_position_usdt / entry_price
                if position_size > max_size_by_value:
                    log_console(
                        logging.INFO, f"Position Size Calc: Capping size from {position_size:.{qty_precision+2}f} to {max_size_by_value:.{qty_precision+2}f} due to Max Position USDT (${max_position_usdt:.2f})")
                    position_size = max_size_by_value
        except ZeroDivisionError:
            log_console(
                logging.WARNING, "Position Size Calc: Entry price is zero during Max Position USDT constraint check.")
            # Continue without applying the constraint in this edge case

    # 2. Minimum Order Quantity Constraint (Check BEFORE step rounding)
    # Use a small tolerance related to qty_step to avoid floating point issues right at the minimum boundary
    min_qty_tolerance = qty_step_float * 0.01
    if position_size < min_order_qty - min_qty_tolerance:
        log_console(
            logging.DEBUG, f"Position Size Calc: Calculated size {position_size:.{qty_precision+4}f} (after potential cap) is below minimum required {min_order_qty}. Returning size 0.")
        return 0.0

    # 3. Quantity Step Constraint (Rounding DOWN)
    try:
        if qty_step_float <= FLOAT_COMPARISON_TOLERANCE:
            # This was checked earlier, but double-check for safety
            log_console(
                logging.ERROR, f"Position Size Calc: Quantity step ({qty_step_float}) is zero or too small during rounding. Cannot apply step constraint.")
            return 0.0

        # Explicitly round DOWN to the nearest multiple of qty_step_float
        # Use math.floor for clear downward rounding
        # Add a tiny epsilon before division to handle floating point inaccuracies
        # where position_size might be slightly less than a multiple of step_size
        # (e.g., 0.09999999999999999 instead of 0.1 when step is 0.01)
        epsilon = qty_step_float * 1e-9  # Epsilon should be much smaller than step size
        num_steps = math.floor((position_size + epsilon) / qty_step_float)
        position_size_adjusted = num_steps * qty_step_float

        # Ensure the adjusted size is not negative due to extreme floating point issues
        position_size_adjusted = max(0.0, position_size_adjusted)

        # Log only if rounding occurred significantly (more than tolerance)
        if abs(position_size_adjusted - position_size) > FLOAT_COMPARISON_TOLERANCE:
            log_console(
                logging.DEBUG, f"Position Size Calc: Size after step rounding DOWN ({qty_step_float}): {position_size_adjusted:.{qty_precision+4}f}")
        position_size = position_size_adjusted
    except ZeroDivisionError:  # Should not happen due to earlier checks
        log_console(
            logging.ERROR, "Position Size Calc: Quantity step is zero during rounding step. Cannot apply constraint.")
        return 0.0
    except Exception as round_e:
        log_console(
            logging.ERROR, f"Position Size Calc: Unexpected error during step rounding: {round_e}")
        return 0.0

    # 4. Final check: ensure size is still >= minimum after rounding DOWN
    if position_size < min_order_qty - min_qty_tolerance:
        log_console(
            logging.INFO, f"Position Size Calc: Final size {position_size:.{qty_precision+4}f} is below minimum {min_order_qty} after rounding down to step. Returning size 0.")
        return 0.0

    # 5. Final rounding to specified precision (mostly cosmetic after step rounding, but ensures adherence)
    # Use standard round for this final step to the required decimal places
    final_size = round(position_size, qty_precision)

    # Ensure final size is not effectively zero after all calculations and rounding
    # Check against min_order_qty again, as rounding could theoretically make it too small
    if final_size < min_order_qty - min_qty_tolerance:
        log_console(
            logging.DEBUG, f"Position Size Calc: Final calculated size ({final
