#!/data/data/com.termux/files/usr/bin/bash

# Cryptocurrency Trading Bot Setup Script for Termux
# This script sets up the complete trading bot environment on Termux

echo "=== Setting up Cryptocurrency Trading Bot on Termux ==="
echo

# Check for required commands
command -v pkg >/dev/null 2>&1 || { echo >&2 "Termux 'pkg' command not found. Are you running this in Termux?"; exit 1; }
command -v python >/dev/null 2>&1 || { echo >&2 "Python is not installed. Installing now..."; }
command -v pip >/dev/null 2>&1 || { echo >&2 "pip is not installed. Installing now..."; }

# Install required packages
echo "Installing required packages..."
# Update package list and install core dependencies
pkg update -y
pkg install -y python python-pip git libffi openssl nodejs

# Explanation of packages:
# python: The core Python interpreter
# python-pip: The package installer for Python
# git: Required to potentially clone libraries or manage versions
# libffi, openssl: Often required by networking and crypto libraries that Python packages depend on
# nodejs: Required by some ccxt exchange implementations

if [ $? -ne 0 ]; then
    echo "Error installing packages. Exiting."
    exit 1
fi
echo "Required packages installed."
echo

# Create project directory and navigate to it
PROJECT_DIR="$HOME/trading_bot"
echo "Creating project directory: $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR"
if [ $? -ne 0 ]; then
    echo "Error creating project directory. Exiting."
    exit 1
fi
cd "$PROJECT_DIR"
echo "Navigated to $PROJECT_DIR"
echo

# Create required subdirectories
echo "Creating subdirectories..."
mkdir -p bot_logs static/css static/js templates
if [ $? -ne 0 ]; then
    echo "Error creating subdirectories. Exiting."
    exit 1
fi
echo "Subdirectories created."
echo

# Install Python dependencies
echo "Installing Python dependencies..."
# Use --break-system-packages or --user if needed, but Termux generally allows direct install
pip install --upgrade pip
pip install ccxt numpy pandas pandas-ta Flask python-dotenv colorama Chart.js
# Removed flask-sqlalchemy and gunicorn for simplicity in a Termux setup,
# and Chart.js is a JS library, added for clarity in web setup.

if [ $? -ne 0 ]; then
    echo "Error installing Python dependencies. Exiting."
    exit 1
fi
echo "Python dependencies installed."
echo

# Create utils.py
echo "Creating utils.py..."
cat > utils.py << 'EOL'
"""
Utility Functions for Trading Bot

This module contains utility functions for the trading bot:
- API interaction with retry mechanism
- Exchange setup and configuration
- Timeframe parsing and conversion
- Error handling and logging utilities
- Price and quantity rounding
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ccxt
import pandas as pd
import pandas_ta as ta # Keep pandas_ta import here as it's used in calculate_indicators

# Configure logger
logger = logging.getLogger("utils")
logger.setLevel(logging.DEBUG) # Set default level for this module

# Mapping from string timeframes to milliseconds (for CCXT fetch_ohlcv)
# Note: calculate_candle_interval still uses seconds for internal logic
TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000, # Approximation, months vary
}


def parse_timeframe_seconds(timeframe: str) -> int:
    """
    Parse timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
        int: Timeframe in seconds, or 0 if invalid
    """
    return TIMEFRAME_SECONDS.get(timeframe, 0)

def parse_timeframe_ms(timeframe: str) -> int:
     """
     Parse timeframe string to milliseconds (for CCXT).

     Args:
         timeframe: Timeframe string (e.g., '1m', '1h', '1d')

     Returns:
         int: Timeframe in milliseconds, or 0 if invalid
     """
     seconds = parse_timeframe_seconds(timeframe)
     return seconds * 1000


def setup_ccxt_exchange(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    options: Optional[Dict] = None,
    params: Optional[Dict] = None,
    test_mode: bool = False
) -> Optional[ccxt.Exchange]:
    """
    Set up and configure a CCXT exchange instance.

    Args:
        exchange_id: Exchange ID (e.g., 'bybit', 'binance')
        api_key: API key for authentication
        api_secret: API secret for authentication
        options: Exchange-specific options
        params: Additional parameters for exchange initialization
        test_mode: Boolean to enable testnet if supported

    Returns:
        ccxt.Exchange: Configured exchange instance, or None if initialization fails
    """
    # Default config
    config = {
        "apiKey": api_key,
        "secret": api_secret,
        "timeout": 30000,  # 30 seconds timeout
        "enableRateLimit": True,
    }

    # Add options if provided
    if options:
        config["options"] = options

    # Add params if provided
    if params:
        config.update(params) # Use update to merge params

    # Add testnet param if in test mode and supported
    if test_mode and hasattr(ccxt, exchange_id) and getattr(ccxt, exchange_id).has["urls"]["test"]:
         config["urls"] = {"api": getattr(ccxt, exchange_id).urls["test"]}
         logger.info(f"Using testnet for {exchange_id}")
    elif test_mode:
         logger.warning(f"Test mode requested for {exchange_id}, but testnet URL not found or not supported by CCXT.")


    try:
        # Get exchange class
        if not hasattr(ccxt, exchange_id):
            logger.error(f"Unsupported exchange: {exchange_id}")
            return None

        exchange_class = getattr(ccxt, exchange_id)

        # Create exchange instance
        exchange = exchange_class(config)

        # Load markets eagerly
        retry_api_call(exchange.load_markets)

        logger.info(f"CCXT {exchange_id} exchange initialized successfully.")
        return exchange
    except Exception as e:
        logger.error(f"Error initializing {exchange_id} exchange: {e}", exc_info=True) # Log traceback
        return None


def retry_api_call(
    func: Callable,
    *args: Any,
    max_retries: int = 5, # Increased retries
    retry_delay: int = 5, # Initial delay in seconds
    **kwargs: Any
) -> Any:
    """
    Retry an API call with exponential backoff and specific error handling.

    Args:
        func: Function (CCXT method) to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retries
        retry_delay: Initial delay in seconds
        **kwargs: Keyword arguments for the function

    Returns:
        Any: Result of the function call

    Raises:
        Exception: Last encountered exception after all retries are exhausted
    """
    last_exception = None
    retry_count = 0

    while retry_count < max_retries:
        try:
            result = func(*args, **kwargs)
            return result
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
            last_exception = e
            # Rate limit or DDoS protection, use exponential backoff with jitter
            sleep_time = retry_delay * (2 ** retry_count) + (time.random() * retry_delay)
            logger.warning(
                f"Rate limit/DDoS protection: {str(e)}. Retrying in {sleep_time:.1f} seconds... "
                f"({retry_count+1}/{max_retries})"
            )
            time.sleep(sleep_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            # Network error or timeout, use standard exponential backoff with jitter
            sleep_time = retry_delay * (1.5 ** retry_count) + (time.random() * retry_delay * 0.5)
            logger.warning(
                f"Network error/Timeout: {str(e)}. Retrying in {sleep_time:.1f} seconds... "
                f"({retry_count+1}/{max_retries})"
            )
            time.sleep(sleep_time)
        except ccxt.ExchangeError as e:
            last_exception = e
            # Check for specific exchange errors that might require retries
            error_str = str(e).lower()
            if "too many requests" in error_str or "rate limit" in error_str or "busy" in error_str:
                sleep_time = retry_delay * (2 ** retry_count) + (time.random() * retry_delay)
                logger.warning(
                    f"Exchange rate limit/busy: {str(e)}. Retrying in {sleep_time:.1f} seconds... "
                    f"({retry_count+1}/{max_retries})"
                )
                time.sleep(sleep_time)
            elif "temporary" in error_str or "maintenance" in error_str:
                 sleep_time = retry_delay * (3 ** retry_count) # Longer delay for maintenance
                 logger.warning(
                     f"Exchange temporary issue/maintenance: {str(e)}. Retrying in {sleep_time:.1f} seconds... "
                     f"({retry_count+1}/{max_retries})"
                 )
                 time.sleep(sleep_time)
            else:
                # General exchange error, might be permanent - re-raise immediately
                logger.error(f"Permanent Exchange error: {str(e)}", exc_info=True)
                raise e
        except Exception as e:
            # Unexpected error, log and re-raise
            logger.error(f"Unexpected error during API call: {str(e)}", exc_info=True)
            raise e

        retry_count += 1

    # If we get here, we've exhausted retries
    if last_exception:
        logger.error(f"API call failed after {max_retries} retries. Last error: {str(last_exception)}")
        raise last_exception # Re-raise the last exception

    # Should ideally not reach here if max_retries > 0 and loop finishes
    return None


def round_to_precision(
    value: float,
    precision: Union[int, float],
    rounding_mode: str = "down"
) -> float:
    """
    Round a value to the specified precision.

    Args:
        value: The value to round.
        precision: The precision to round to. Can be an integer (decimal places)
                   or a float (tick size).
        rounding_mode: 'down', 'up', or 'nearest'.

    Returns:
        float: The rounded value.
    """
    decimal_value = Decimal(str(value))

    if isinstance(precision, int):
        # Rounding to decimal places
        quantize_value = Decimal('1E-%d' % precision)
        if rounding_mode == "down":
            rounded_value = decimal_value.quantize(quantize_value, rounding=ROUND_DOWN)
        elif rounding_mode == "up":
            rounded_value = decimal_value.quantize(quantize_value, rounding=ROUND_UP)
        else: # nearest
            rounded_value = decimal_value.quantize(quantize_value)
    elif isinstance(precision, float) or isinstance(precision, int):
        # Rounding to tick size
        decimal_precision = Decimal(str(precision))
        if decimal_precision == 0:
             return value # Avoid division by zero

        # Count decimal places in tick size for final float conversion
        precision_decimals = 0
        if '.' in str(precision):
             precision_decimals = len(str(precision).split('.')[-1])

        ticks = decimal_value / decimal_precision

        if rounding_mode == "up":
            rounded_ticks = ticks.quantize(Decimal("1"), rounding=ROUND_UP)
        elif rounding_mode == "down":
            rounded_ticks = ticks.quantize(Decimal("1"), rounding=ROUND_DOWN)
        else:  # nearest
            rounded_ticks = ticks.quantize(Decimal("1"))

        rounded_value = rounded_ticks * decimal_precision

        # Ensure correct decimal places in float conversion
        return float(rounded_value.quantize(Decimal("0." + "0" * precision_decimals) if precision_decimals > 0 else Decimal("1")))
    else:
        logger.warning(f"Unsupported precision type: {type(precision)}. Returning original value.")
        return value

    return float(rounded_value)


def calculate_candle_interval(timeframe: str) -> Tuple[datetime, datetime]:
    """
    Calculate start and end datetime for the current candle interval in UTC.

    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
        Tuple[datetime, datetime]: Start and end datetime in UTC
    """
    now = datetime.utcnow()
    seconds = parse_timeframe_seconds(timeframe)

    if seconds == 0:
        # Invalid timeframe
        logger.warning(f"Invalid timeframe '{timeframe}' for candle interval calculation.")
        return now, now # Return current time as a fallback

    # Special handling for daily, weekly, monthly candles (often UTC midnight aligned)
    if timeframe == "1d":
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
    elif timeframe == "1w":
        # Get current week's Monday (UTC)
        start = now - timedelta(days=now.weekday())
        start = datetime(start.year, start.month, start.day)
        end = start + timedelta(days=7)
    elif timeframe == "1M":
        # Get current month's first day (UTC)
        start = datetime(now.year, now.month, 1)
        # Move to next month's first day
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1)
        else:
            end = datetime(now.year, now.month + 1, 1)
    else:
        # Regular intervals (align to the nearest interval start)
        timestamp_seconds = int(now.timestamp())
        interval_start_timestamp = (timestamp_seconds // seconds) * seconds
        start = datetime.utcfromtimestamp(interval_start_timestamp)
        end = start + timedelta(seconds=seconds)

    return start, end

def format_timestamp(timestamp_ms: int) -> str:
    """
    Formats a millisecond timestamp into a human-readable string.

    Args:
        timestamp_ms: Timestamp in milliseconds.

    Returns:
        str: Formatted datetime string.
    """
    try:
        dt_object = datetime.fromtimestamp(timestamp_ms / 1000)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception as e:
        logger.error(f"Error formatting timestamp {timestamp_ms}: {e}")
        return str(timestamp_ms)

EOL

# Create indicators.py
echo "Creating indicators.py..."
cat > indicators.py << 'EOL'
"""
Technical Indicators Module

This module contains functions for calculating technical indicators
and trading signals based on the configured strategy.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import pandas_ta as ta

# Configure logger
logger = logging.getLogger("indicators")
logger.setLevel(logging.DEBUG) # Set default level for this module

# Default indicator parameters
DEFAULT_RSI_WINDOW = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BB_WINDOW = 20
DEFAULT_BB_STD = 2.0
DEFAULT_EMA_FAST = 8
DEFAULT_EMA_SLOW = 21
DEFAULT_ATR_WINDOW = 14
# Added defaults for CCI and Stoch if they are ever added to calculate_indicators
# DEFAULT_CCI_WINDOW = 20
# DEFAULT_STOCH_WINDOW = 14
# DEFAULT_STOCH_K = 3
# DEFAULT_STOCH_D = 3


def calculate_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate technical indicators based on configuration.

    Args:
        df: DataFrame with OHLCV data (must have 'open', 'high', 'low', 'close', 'volume' columns)
        config: Dictionary with indicator configuration

    Returns:
        pd.DataFrame: DataFrame with added indicator columns. Returns original df if input is invalid.
    """
    if df.empty or not all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
        logger.warning("Input DataFrame is empty or missing OHLCV columns. Skipping indicator calculation.")
        return df

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Calculate RSI if enabled
    rsi_config = config.get("rsi", {})
    if rsi_config.get("enabled", False): # Default to disabled if config missing
        window = rsi_config.get("window", DEFAULT_RSI_WINDOW)
        try:
            # pandas_ta returns a Series or DataFrame
            rsi_data = ta.rsi(df_copy["close"], length=window)
            if rsi_data is not None:
                df_copy["rsi"] = rsi_data
                logger.debug(f"Calculated RSI with window={window}")
            else:
                 logger.warning(f"RSI calculation returned None.")
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=True)

    # Calculate MACD if enabled
    macd_config = config.get("macd", {})
    if macd_config.get("enabled", False): # Default to disabled if config missing
        fast_period = macd_config.get("fast_period", DEFAULT_MACD_FAST)
        slow_period = macd_config.get("slow_period", DEFAULT_MACD_SLOW)
        signal_period = macd_config.get("signal_period", DEFAULT_MACD_SIGNAL)
        try:
            macd = ta.macd(
                df_copy["close"],
                fast=fast_period,
                slow=slow_period,
                signal=signal_period
            )
            # Add MACD components to dataframe
            if macd is not None and not macd.empty:
                # Correct column names based on pandas_ta output format
                macd_col = f"MACD_{fast_period}_{slow_period}_{signal_period}"
                signal_col = f"MACDs_{fast_period}_{slow_period}_{signal_period}"
                hist_col = f"MACDh_{fast_period}_{slow_period}_{signal_period}"

                if macd_col in macd.columns: df_copy["macd"] = macd[macd_col]
                if signal_col in macd.columns: df_copy["macd_signal"] = macd[signal_col]
                if hist_col in macd.columns: df_copy["macd_hist"] = macd[hist_col]

                if "macd" in df_copy.columns: # Check if at least one column was added
                    logger.debug(f"Calculated MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")
                else:
                    logger.warning(f"MACD calculation returned unexpected columns: {macd.columns}")
            else:
                 logger.warning(f"MACD calculation returned None or empty DataFrame.")

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}", exc_info=True)

    # Calculate Bollinger Bands if enabled
    bb_config = config.get("bollinger_bands", {})
    if bb_config.get("enabled", False): # Default to disabled if config missing
        window = bb_config.get("window", DEFAULT_BB_WINDOW)
        std_dev = bb_config.get("std_dev", DEFAULT_BB_STD)
        try:
            bbands = ta.bbands(df_copy["close"], length=window, std=std_dev)
            if bbands is not None and not bbands.empty:
                # Correct column names based on pandas_ta output format
                upper_col = f"BBU_{window}_{std_dev}"
                middle_col = f"BBM_{window}_{std_dev}"
                lower_col = f"BBL_{window}_{std_dev}"
                width_col = f"BBB_{window}_{std_dev}"
                pctb_col = f"BBP_{window}_{std_dev}"

                if upper_col in bbands.columns: df_copy["bb_upper"] = bbands[upper_col]
                if middle_col in bbands.columns: df_copy["bb_middle"] = bbands[middle_col]
                if lower_col in bbands.columns: df_copy["bb_lower"] = bbands[lower_col]
                if width_col in bbands.columns: df_copy["bb_width"] = bbands[width_col]
                if pctb_col in bbands.columns: df_copy["bb_pctb"] = bbands[pctb_col]

                if "bb_upper" in df_copy.columns: # Check if at least one column was added
                    logger.debug(f"Calculated Bollinger Bands with window={window}, std_dev={std_dev}")
                else:
                    logger.warning(f"Bollinger Bands calculation returned unexpected columns: {bbands.columns}")
            else:
                 logger.warning(f"Bollinger Bands calculation returned None or empty DataFrame.")

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}", exc_info=True)

    # Calculate EMA Cross if enabled
    ema_config = config.get("ema_cross", {})
    if ema_config.get("enabled", False): # Default to disabled if config missing
        fast_period = ema_config.get("fast_period", DEFAULT_EMA_FAST)
        slow_period = ema_config.get("slow_period", DEFAULT_EMA_SLOW)
        try:
            df_copy["ema_fast"] = ta.ema(df_copy["close"], length=fast_period)
            df_copy["ema_slow"] = ta.ema(df_copy["close"], length=slow_period)
            # Calculate crossover signals (1: bullish cross, -1: bearish cross, 0: no cross)
            # Using shift(1) to compare current vs previous
            df_copy["ema_cross_signal"] = 0 # Initialize
            # Bullish cross: fast crosses above slow
            df_copy.loc[(df_copy["ema_fast"] > df_copy["ema_slow"]) & (df_copy["ema_fast"].shift(1) <= df_copy["ema_slow"].shift(1)), "ema_cross_signal"] = 1
            # Bearish cross: fast crosses below slow
            df_copy.loc[(df_copy["ema_fast"] < df_copy["ema_slow"]) & (df_copy["ema_fast"].shift(1) >= df_copy["ema_slow"].shift(1)), "ema_cross_signal"] = -1

            # Optional: Also add a column indicating the current state (fast > slow or fast < slow)
            df_copy["ema_direction"] = np.where(
                df_copy["ema_fast"] > df_copy["ema_slow"],
                1,
                np.where(df_copy["ema_fast"] < df_copy["ema_slow"], -1, 0)
            )

            logger.debug(f"Calculated EMA Cross with fast={fast_period}, slow={slow_period}")
        except Exception as e:
            logger.error(f"Error calculating EMA Cross: {e}", exc_info=True)

    # Calculate ATR if enabled (used for risk management, not typically for signal direction)
    atr_config = config.get("atr", {})
    if atr_config.get("enabled", False): # Default to disabled if config missing
        window = atr_config.get("window", DEFAULT_ATR_WINDOW)
        try:
            df_copy["atr"] = ta.atr(
                df_copy["high"],
                df_copy["low"],
                df_copy["close"],
                length=window
            )
            logger.debug(f"Calculated ATR with window={window}")
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=True)

    # Drop rows with NaN values introduced by indicators
    df_copy.dropna(inplace=True)

    return df_copy


def calculate_rsi_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on RSI indicator.
    Signal strength is relative to distance from thresholds.

    Args:
        df: DataFrame with indicator data (must have 'rsi' column)
        config: RSI configuration ({'overbought': 70, 'oversold': 30})

    Returns:
        Tuple[float, str]: Signal strength (0.0 to 1.0) and direction ("long" or "short").
                           Returns (0.0, "") if no valid signal.
    """
    if df.empty or "rsi" not in df.columns:
        return 0.0, ""
    if df["rsi"].iloc[-1] is None or pd.isna(df["rsi"].iloc[-1]):
         return 0.0, ""

    # Get parameters
    rsi = df["rsi"].iloc[-1]
    overbought = config.get("overbought", 70)
    oversold = config.get("oversold", 30)

    # Calculate signal strength and direction
    signal_strength = 0.0
    direction = ""

    if rsi <= oversold:
        # Oversold - potential buy signal. Strength increases as RSI goes lower.
        signal_strength = min(1.0, (oversold - rsi) / oversold) # Scale 0-1
        direction = "long"
    elif rsi >= overbought:
        # Overbought - potential sell signal. Strength increases as RSI goes higher.
        signal_strength = min(1.0, (rsi - overbought) / (100 - overbought)) # Scale 0-1
        direction = "short"
    else:
        # Neutral zone - signal strength is based on proximity to boundaries, but weaker
        midpoint = (overbought + oversold) / 2
        if rsi < midpoint:
            # Below midpoint, weak long signal
            signal_strength = (midpoint - rsi) / (midpoint - oversold) * 0.5 # Max strength 0.5
            direction = "long"
        else:
            # Above midpoint, weak short signal
            signal_strength = (rsi - midpoint) / (overbought - midpoint) * 0.5 # Max strength 0.5
            direction = "short"

    return max(0.0, min(1.0, signal_strength)), direction # Ensure strength is between 0 and 1


def calculate_macd_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on MACD indicator.
    Signal is based on MACD line crossing Signal line (histogram crossing zero).

    Args:
        df: DataFrame with indicator data (must have 'macd_hist' column)
        config: MACD configuration (used for periods, though signal doesn't directly need them)

    Returns:
        Tuple[float, str]: Signal strength (0.0 to 1.0) and direction ("long" or "short").
                           Returns (0.0, "") if no valid signal.
    """
    if df.empty or "macd_hist" not in df.columns or len(df) < 2:
        return 0.0, ""
    if df["macd_hist"].iloc[-1] is None or pd.isna(df["macd_hist"].iloc[-1]) or \
       df["macd_hist"].iloc[-2] is None or pd.isna(df["macd_hist"].iloc[-2]):
         return 0.0, ""

    # Get latest values
    hist = df["macd_hist"].iloc[-1]
    prev_hist = df["macd_hist"].iloc[-2]

    signal_strength = 0.0
    direction = ""

    # Bullish crossover: histogram crosses above zero
    if hist > 0 and prev_hist <= 0:
        signal_strength = min(1.0, abs(hist)) # Strength based on histogram magnitude after cross
        direction = "long"
    # Bearish crossover: histogram crosses below zero
    elif hist < 0 and prev_hist >= 0:
        signal_strength = min(1.0, abs(hist)) # Strength based on histogram magnitude after cross
        direction = "short"
    # Trend following (optional, weaker signal if no recent cross)
    elif hist > 0:
         signal_strength = min(0.5, abs(hist) * 0.5) # Weaker strength for established trend
         direction = "long"
    elif hist < 0:
         signal_strength = min(0.5, abs(hist) * 0.5) # Weaker strength for established trend
         direction = "short"

    return max(0.0, min(1.0, signal_strength)), direction


def calculate_bollinger_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on Bollinger Bands indicator.
    Signal is based on price relative to bands (reversion or breakout).

    Args:
        df: DataFrame with indicator data (must have 'close', 'bb_upper', 'bb_lower')
        config: Bollinger Bands configuration (used for window/std_dev, though signal doesn't directly need them)

    Returns:
        Tuple[float, str]: Signal strength (0.0 to 1.0) and direction ("long" or "short").
                           Returns (0.0, "") if no valid signal.
    """
    if df.empty or "close" not in df.columns or "bb_upper" not in df.columns or "bb_lower" not in df.columns:
        return 0.0, ""
    if df["close"].iloc[-1] is None or pd.isna(df["close"].iloc[-1]) or \
       df["bb_upper"].iloc[-1] is None or pd.isna(df["bb_upper"].iloc[-1]) or \
       df["bb_lower"].iloc[-1] is None or pd.isna(df["bb_lower"].iloc[-1]):
         return 0.0, ""


    # Get latest values
    close = df["close"].iloc[-1]
    upper = df["bb_upper"].iloc[-1]
    lower = df["bb_lower"].iloc[-1]
    middle = df["bb_middle"].iloc[-1] if "bb_middle" in df.columns else (upper + lower) / 2
    width = df["bb_width"].iloc[-1] if "bb_width" in df.columns and not pd.isna(df["bb_width"].iloc[-1]) else (upper - lower) / middle if middle != 0 else 0.01 # Avoid div by zero

    signal_strength = 0.0
    direction = ""

    # Reversion strategy: Buy when price touches/crosses lower band, Sell when price touches/crosses upper band
    if close <= lower:
        # Price at or below lower band - strong buy signal (reversion)
        # Strength increases as price goes further below the band
        signal_strength = min(1.0, (lower - close) / (lower * 0.01)) # Scale based on 1% below lower band
        direction = "long"
    elif close >= upper:
        # Price at or above upper band - strong sell signal (reversion)
        # Strength increases as price goes further above the band
        signal_strength = min(1.0, (close - upper) / (upper * 0.01)) # Scale based on 1% above upper band
        direction = "short"
    # Optional: Trend following strategy (price between middle and bands, moving towards band)
    # This can conflict with reversion, so decide on strategy type (reversion vs trend)
    # Example for trend: if close > middle and rising towards upper, could be trend continuation signal.
    # For simplicity, sticking to reversion based on band touches.

    return max(0.0, min(1.0, signal_strength)), direction


def calculate_ema_cross_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on EMA cross indicator.
    Signal is based on the crossover event.

    Args:
        df: DataFrame with indicator data (must have 'ema_fast', 'ema_slow', 'ema_cross_signal')
        config: EMA cross configuration (used for periods, though signal doesn't directly need them)

    Returns:
        Tuple[float, str]: Signal strength (0.0 to 1.0) and direction ("long" or "short").
                           Returns (0.0, "") if no valid signal.
    """
    if df.empty or "ema_cross_signal" not in df.columns:
        return 0.0, ""
    if df["ema_cross_signal"].iloc[-1] is None or pd.isna(df["ema_cross_signal"].iloc[-1]):
         return 0.0, ""

    # Get latest crossover signal
    cross_signal = df["ema_cross_signal"].iloc[-1]

    signal_strength = 0.0
    direction = ""

    if cross_signal == 1:
        # Bullish crossover - strong buy signal
        signal_strength = 1.0
        direction = "long"
    elif cross_signal == -1:
        # Bearish crossover - strong sell signal
        signal_strength = 1.0
        direction = "short"
    # Note: This signal focuses purely on the *event* of the crossover.
    # If you want strength based on the *distance* between EMAs during a trend,
    # you'd need to add logic using 'ema_fast' and 'ema_slow' and calculate a
    # normalized difference. For this simple cross strategy, 1.0 indicates the event.

    return signal_strength, direction


def calculate_signal(df: pd.DataFrame, indicators_config: Dict,
                     entry_threshold: float = 0.5, exit_threshold: float = 0.3) -> Tuple[float, str]:
    """
    Calculate overall trading signal based on multiple indicators and weights.

    Args:
        df: DataFrame with calculated indicator data
        indicators_config: Configuration for all indicators, including weights and enabled status
        entry_threshold: Signal threshold for entry (0.0 to 1.0)
        exit_threshold: Signal threshold for exit (0.0 to 1.0, absolute value used)

    Returns:
        Tuple[float, str]: Overall signal strength (0.0 to 1.0) and direction ("long", "short", or "").
                           Strength >= threshold implies a tradeable signal in that direction.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Cannot calculate signal.")
        return 0.0, ""

    # Initialize variables for weighted signal calculation
    total_weight = 0.0
    long_signal_sum = 0.0
    short_signal_sum = 0.0

    # Dictionary mapping indicator names to their signal calculation functions
    signal_functions = {
        "rsi": calculate_rsi_signal,
        "macd": calculate_macd_signal,
        "bollinger_bands": calculate_bollinger_signal,
        "ema_cross": calculate_ema_cross_signal,
        # Add other indicators here if you implement them
    }

    # Calculate weighted sum of signals from enabled indicators
    for indicator_name, func in signal_functions.items():
        indicator_config = indicators_config.get(indicator_name, {})
        if indicator_config.get("enabled", False): # Only process if enabled
            weight = float(indicator_config.get("weight", 1.0)) # Default weight 1.0
            if weight > 0: # Only include indicators with positive weight
                try:
                    signal_strength, direction = func(df, indicator_config)
                    if direction == "long":
                        long_signal_sum += signal_strength * weight
                        total_weight += weight
                    elif direction == "short":
                        short_signal_sum += signal_strength * weight
                        total_weight += weight
                    # Signals with "" direction (like ATR) don't contribute to entry/exit direction signal
                except Exception as e:
                    logger.error(f"Error calculating signal for {indicator_name}: {e}", exc_info=True)

    # If no enabled indicators or total weight is zero, return neutral
    if total_weight == 0:
        return 0.0, ""

    # Calculate final signals normalized by total weight
    # Avoid division by zero if somehow total_weight is 0 despite checks
    final_long_signal = long_signal_sum / total_weight if total_weight > 0 else 0.0
    final_short_signal = short_signal_sum / total_weight if total_weight > 0 else 0.0

    # Determine direction based on strongest signal exceeding threshold
    # Consider the difference between long and short signals
    net_signal = final_long_signal - final_short_signal

    overall_strength = abs(net_signal)
    overall_direction = ""

    if net_signal > 0 and overall_strength >= entry_threshold:
        overall_direction = "long"
    elif net_signal < 0 and overall_strength >= entry_threshold:
        overall_direction = "short"

    # Note: This aggregation method sums weighted strengths. An alternative
    # could be requiring a minimum number of indicators to align, or using
    # a decision tree/matrix approach. Summing is a simple weighted average.

    return overall_strength, overall_direction

# Example of how you might use ATR (not for directional signal, but for volatility)
def get_volatility(df: pd.DataFrame, config: Dict) -> float:
    """
    Get volatility measure based on ATR.

    Args:
        df: DataFrame with indicator data (must have 'atr' column)
        config: ATR configuration

    Returns:
        float: Latest ATR value, or 0.0 if not available
    """
    atr_config = config.get("atr", {})
    if atr_config.get("enabled", False) and "atr" in df.columns and not df.empty:
        latest_atr = df["atr"].iloc[-1]
        if latest_atr is not None and pd.notna(latest_atr):
            return float(latest_atr)
    return 0.0
EOL

# Create trading_bot.py
echo "Creating trading_bot.py..."
cat > trading_bot.py << 'EOL'
"""
Trading Bot Core Module

This module implements the main trading bot logic, including:
- Exchange connection and market data retrieval
- Technical analysis and signal generation
- Order execution and management
- Position tracking and risk management
"""

import json
import logging
import os
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP # Import ROUND_UP
from typing import Dict, List, Optional, Tuple, Union, Any

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv # Import dotenv

from indicators import calculate_indicators, calculate_signal, get_volatility # Import get_volatility
from utils import parse_timeframe_ms, setup_ccxt_exchange, retry_api_call, round_to_precision, format_timestamp # Import round_to_precision and format_timestamp

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger = logging.getLogger("trading_bot")
logger.setLevel(logging.DEBUG) # Set default level for this module


class TradingBot:
    """
    Main trading bot class that handles the entire trading logic flow:
    - Loading configuration
    - Connecting to exchange
    - Retrieving market data
    - Analyzing with technical indicators
    - Executing trades based on signals
    - Managing risk and positions
    """

    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the trading bot with the given configuration.

        Args:
            config_file: Path to the configuration file
        """
        self.logger = logger
        self.config_file = config_file
        self.state_file = "bot_state.json"

        # Load configuration and state
        self.config = self.load_config()
        self.state = self.load_state()

        # Get core parameters from config
        self.exchange_id = self.config.get("exchange", "bybit").lower() # Ensure lowercase
        self.symbol = self.config.get("symbol", "BTC/USDT:USDT").upper() # Ensure uppercase
        self.timeframe = self.config.get("timeframe", "15m")
        self.test_mode = self.config.get("test_mode", True)

        # Initialize exchange connection (will load markets internally)
        self.exchange = self.setup_exchange()
        if not self.exchange:
             raise ConnectionError(f"Failed to initialize exchange {self.exchange_id}")

        # Market and position information
        self.market_info = None # Will be fetched in initialize
        self.current_position = self.state.get("positions", {}).get(self.symbol) # Load from state
        self.candles_df = None # Will be fetched in initialize

        # Precision and minimums (updated from market info during initialization)
        self.precision = {
            "price": self.config.get("advanced", {}).get("price_precision", 8), # Default to high precision
            "amount": self.config.get("advanced", {}).get("amount_precision", 8) # Default to high precision
        }
        self.min_amount = self.config.get("advanced", {}).get("min_amount", 0.000001) # Default to small min amount

        # Initialize exchange specifics and fetch initial data
        self.initialize()


    def load_config(self) -> Dict:
        """
        Load configuration from JSON file. Create a default one if it doesn't exist.

        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Configuration file {self.config_file} not found. Creating default.")
            # Create default config if file doesn't exist
            default_config = {
                "exchange": "bybit",
                "symbol": "BTC/USDT:USDT",
                "timeframe": "15m",
                # API keys should ideally be in environment variables or handled securely
                # Placing them here is for demonstration in a simple Termux setup
                # Consider using environment variables via python-dotenv as shown above
                "api_key": os.getenv("API_KEY", ""), # Load from env or use empty string
                "api_secret": os.getenv("API_SECRET", ""), # Load from env or use empty string
                "test_mode": True, # Set to False for live trading
                "risk_management": {
                    "position_size_pct": 1.0, # Percentage of AVAILABLE balance to use for position VALUE
                    "max_risk_per_trade_pct": 1.0, # Percentage of total equity to risk per trade (used with SL)
                    "use_atr_position_sizing": True, # Adjust position size based on ATR stop loss distance
                    "stop_loss": {
                        "enabled": True,
                        "mode": "atr", # "atr" or "fixed_pct"
                        "atr_multiplier": 2.0, # ATR multiplier for ATR mode
                        "fixed_pct": 2.0 # Percentage below entry for fixed_pct mode (for long)
                    },
                    "take_profit": {
                        "enabled": True,
                        "mode": "atr", # "atr" or "fixed_pct"
                        "atr_multiplier": 4.0, # ATR multiplier for ATR mode
                        "fixed_pct": 4.0 # Percentage above entry for fixed_pct mode (for long)
                    },
                    "break_even": {
                        "enabled": True, # Automatically move SL to entry price when profit target hit
                        "activation_pct": 1.0 # Percentage profit required to move SL to break even
                    }
                },
                "strategy": {
                    "entry_threshold": 0.6, # Signal strength required to enter (0.0 to 1.0)
                    "exit_threshold": 0.4, # Signal strength required to exit (0.0 to 1.0)
                    "volume_filter": {
                         "enabled": False,
                         "min_24h_volume_usd": 1000000 # Minimum 24h volume in USD for the symbol
                    },
                    "indicators": {
                        "rsi": {
                            "enabled": True,
                            "window": 14,
                            "overbought": 70,
                            "oversold": 30,
                            "weight": 1.0 # Weight in overall signal calculation
                        },
                        "macd": {
                            "enabled": True,
                            "fast_period": 12,
                            "slow_period": 26,
                            "signal_period": 9,
                            "weight": 1.0
                        },
                        "bollinger_bands": {
                            "enabled": True,
                            "window": 20,
                            "std_dev": 2.0,
                            "weight": 1.0
                        },
                        "ema_cross": {
                            "enabled": True,
                            "fast_period": 8,
                            "slow_period": 21,
                            "use_for_exit": True, # Use EMA cross specifically as an exit signal
                            "weight": 1.0 # Weight in overall signal calculation
                        },
                        "atr": { # ATR is mainly for risk management/position sizing, not directional signal here
                            "enabled": True,
                            "window": 14
                        }
                    }
                },
                "advanced": {
                    "candles_required": 200, # Number of candles to fetch for indicators
                    "price_precision": 8, # Default precision, will try to get from exchange
                    "amount_precision": 8, # Default precision, will try to get from exchange
                    "min_amount": 0.000001, # Default minimum trade amount, will try to get from exchange
                    "max_amount": None # Optional max trade amount
                },
                "loop_interval_seconds": 30 # How often the bot checks for signals (should be less than timeframe)
            }
            try:
                with open(self.config_file, "w") as f:
                    json.dump(default_config, f, indent=4)
                self.logger.info(f"Created default configuration file {self.config_file}")
            except Exception as e:
                self.logger.error(f"Failed to write default config file {self.config_file}: {e}", exc_info=True)
                # Continue with default config in memory
            return default_config
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file {self.config_file}. Please fix it.")
            raise # Re-raise the error

    def load_state(self) -> Dict:
        """
        Load bot state from JSON file.

        Returns:
            Dict: Bot state dictionary
        """
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.logger.info(f"State loaded from {self.state_file}")
            # Ensure necessary keys exist in state
            state.setdefault("positions", {})
            state.setdefault("orders", {})
            state.setdefault("trades", [])
            state.setdefault("last_update", 0)
            state.setdefault("equity_history", []) # Add equity history for charting
            state.setdefault("current_trade", None) # Track details of the current open trade

            # Clean up old orders in state that might not be active anymore (basic approach)
            # More robust would be to fetch open orders from exchange
            if self.symbol in state["positions"]:
                 # If there's an open position but no orders in state, assume SL/TP were filled or cancelled externally
                 if state.get("orders", {}) == {}:
                     self.logger.warning("Found open position in state but no associated orders. Assuming SL/TP were handled externally.")
            else:
                 # If no position, clear any lingering orders for this symbol
                 if self.symbol in state["orders"]:
                      self.logger.info(f"Clearing old orders for {self.symbol} from state as no position found.")
                      del state["orders"][self.symbol]


            return state
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.warning(f"State file {self.state_file} not found or is invalid. Initializing with default state.")
            # Initialize with default state if file doesn't exist or is invalid
            default_state = {
                "positions": {}, # Track open positions {symbol: {side, amount, entry_price, sl_price, tp_price}}
                "orders": {}, # Track active orders {symbol: {sl: {id, price, side}, tp: {id, price, side}}}
                "trades": [], # List of closed trades
                "last_update": 0, # Timestamp of last bot run
                "equity_history": [], # For performance charting {timestamp, balance}
                "current_trade": None # Details of the currently open trade for PnL calculation
            }
            self.save_state(default_state)
            return default_state

    def save_state(self, state: Optional[Dict] = None) -> None:
        """
        Save bot state to JSON file.

        Args:
            state: State dictionary to save (uses self.state if None)
        """
        if state is None:
            state = self.state
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
            self.logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}", exc_info=True)

    def setup_exchange(self) -> Optional[ccxt.Exchange]:
        """
        Set up the CCXT exchange instance.

        Returns:
            ccxt.Exchange: Configured exchange instance
        """
        # Prioritize environment variables, then config file
        api_key = os.getenv("API_KEY", self.config.get("api_key"))
        api_secret = os.getenv("API_SECRET", self.config.get("api_secret"))

        if not api_key or not api_secret:
             self.logger.warning("API_KEY or API_SECRET not found in config or environment variables. Limited functionality (no trading).")
             # Still allow connecting for fetching public data
             api_key = None
             api_secret = None


        # Default options for the exchange
        options = {}
        params = {}

        # Add exchange-specific options
        # Bybit requires 'defaultType' for futures/swap markets
        if self.exchange_id == "bybit":
            options["defaultType"] = "swap" # Or 'future' depending on your market
        # Add other exchange specific options here if needed

        # Create exchange instance with retry mechanism
        exchange = setup_ccxt_exchange(
            self.exchange_id,
            api_key,
            api_secret,
            options=options,
            params=params,
            test_mode=self.test_mode # Pass test_mode flag
        )

        return exchange

    def initialize(self) -> None:
        """Initialize exchange connection and load market data"""
        if not self.exchange:
             self.logger.error("Exchange not initialized. Cannot proceed with initialization.")
             return

        try:
            # Fetch market information for the symbol
            self.market_info = retry_api_call(self.exchange.market, self.symbol)
            if not self.market_info:
                 raise ValueError(f"Could not fetch market info for {self.symbol}")

            self.logger.info(f"Market info loaded for {self.symbol}")

            # Update precision settings from market info if available
            # CCXT market info provides 'precision' and 'limits'
            if "precision" in self.market_info:
                # Price precision can be decimal places or tick size
                if "price" in self.market_info["precision"]:
                    # Use exchange precision if available, otherwise use config/default
                    self.precision["price"] = self.market_info["precision"]["price"]
                    self.logger.debug(f"Using exchange price precision: {self.precision['price']}")
                # Amount precision can be decimal places or step size
                if "amount" in self.market_info["precision"]:
                    self.precision["amount"] = self.market_info["precision"]["amount"]
                    self.logger.debug(f"Using exchange amount precision: {self.precision['amount']}")

            if "limits" in self.market_info and "amount" in self.market_info["limits"]:
                 if "min" in self.market_info["limits"]["amount"]:
                      self.min_amount = self.market_info["limits"]["amount"]["min"]
                      self.logger.debug(f"Using exchange min amount: {self.min_amount}")
                 if "max" in self.market_info["limits"]["amount"]:
                      self.max_amount = self.market_info["limits"]["amount"]["max"]
                      self.logger.debug(f"Using exchange max amount: {self.max_amount}")
                 else:
                      self.max_amount = self.config.get("advanced", {}).get("max_amount") # Use config if no exchange max

            # Fetch current position (already attempted in __init__, but re-confirm)
            self.update_position()

            # Fetch initial candles
            self.update_candles()

            self.logger.info(f"Bot initialized successfully for {self.symbol}")
        except Exception as e:
            self.logger.error(f"Initialization error: {e}", exc_info=True)
            raise # Re-raise to stop the bot if initialization fails


    def update_candles(self) -> None:
        """Fetch and update OHLCV candles data and calculate indicators"""
        if not self.exchange:
             self.logger.error("Exchange not initialized. Cannot update candles.")
             return

        try:
            # Determine how many candles we need for indicators + signal
            # pandas_ta needs enough data points for the longest indicator window + 1 for signal calculation
            indicators_config = self.config.get("strategy", {}).get("indicators", {})
            required_history = 0
            # Find the maximum window size across all enabled indicators
            for indicator_name, ind_config in indicators_config.items():
                 if ind_config.get("enabled", False):
                      if indicator_name == "rsi": required_history = max(required_history, ind_config.get("window", DEFAULT_RSI_WINDOW))
                      elif indicator_name == "macd": required_history = max(required_history, ind_config.get("slow_period", DEFAULT_MACD_SLOW) + ind_config.get("signal_period", DEFAULT_MACD_SIGNAL)) # MACD requires sum of periods
                      elif indicator_name == "bollinger_bands": required_history = max(required_history, ind_config.get("window", DEFAULT_BB_WINDOW))
                      elif indicator_name == "ema_cross": required_history = max(required_history, ind_config.get("slow_period", DEFAULT_EMA_SLOW))
                      elif indicator_name == "atr": required_history = max(required_history, ind_config.get("window", DEFAULT_ATR_WINDOW))
                      # Add other indicators here

            # Add a buffer for safety and ensure enough data for the first calculation point
            # A common rule of thumb is max_window + 20, or just a generous fixed number if unsure
            candles_required = max(required_history + 20, self.config.get("advanced", {}).get("candles_required", 200)) # Use config value or default 200

            self.logger.debug(f"Fetching {candles_required} candles for timeframe {self.timeframe}")

            # Fetch candles with retry mechanism
            # CCXT fetch_ohlcv expects timeframe as string
            ohlcv = retry_api_call(
                self.exchange.fetch_ohlcv,
                self.symbol,
                timeframe=self.timeframe,
                limit=candles_required
            )

            if not ohlcv:
                 self.logger.warning(f"Failed to fetch OHLCV data for {self.symbol} {self.timeframe}")
                 return # Keep the old data or have an empty df

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp (milliseconds) to datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            self.candles_df = df
            self.logger.debug(f"Updated {len(df)} candles for {self.symbol}")

            # Calculate indicators
            self.candles_df = calculate_indicators(
                self.candles_df,
                indicators_config
            )
            self.logger.debug("Indicators calculated.")

        except Exception as e:
            self.logger.error(f"Error updating candles: {e}", exc_info=True)
            # Do not re-raise, allow the bot to continue with old data or no data


    def update_position(self) -> None:
        """Update current position information from exchange"""
        if not self.exchange:
             self.logger.error("Exchange not initialized. Cannot update position.")
             return

        try:
            # Fetch positions for the symbol
            positions = retry_api_call(
                self.exchange.fetch_positions,
                [self.symbol] # Pass symbol list to fetch only relevant positions
            ) or []

            # Find the position for our symbol
            found_position = None
            for position in positions:
                # Check if position belongs to the correct symbol and is not zero size
                if position["symbol"] == self.symbol and float(position["contracts"]) != 0:
                    contracts = float(position["contracts"])
                    found_position = {
                        "side": "long" if contracts > 0 else "short",
                        "amount": abs(contracts),
                        "entry_price": float(position.get("entryPrice", 0)), # Use .get for safety
                        "unrealizedPnl": float(position.get("unrealizedPnl", 0)), # Use .get for safety
                        "liquidation_price": float(position.get("liquidationPrice", 0)), # Use .get for safety
                        "timestamp": int(time.time() * 1000) # Add a timestamp
                    }
                    break # Found the relevant position

            # Update self.current_position and state
            self.current_position = found_position

            if self.current_position:
                 # Update position details in state, preserving SL/TP if they exist
                 # This assumes SL/TP are managed by the bot via orders, not exchange features like OCO/TP_SL orders attached to position
                 if self.symbol not in self.state.get("positions", {}):
                      self.state["positions"][self.symbol] = {} # Initialize if new position

                 self.state["positions"][self.symbol].update({
                     "side": self.current_position["side"],
                     "amount": self.current_position["amount"],
                     "entry_price": self.current_position["entry_price"],
                     "unrealizedPnl": self.current_position["unrealizedPnl"],
                     "liquidation_price": self.current_position["liquidation_price"],
                     "timestamp": self.current_position["timestamp"]
                 })

                 # Ensure SL/TP prices are still in state if position exists (assuming they were set by bot)
                 # If you rely on exchange-managed SL/TP attached to the position, you'd fetch those here
                 if self.state["positions"][self.symbol].get("sl_price") is None:
                     # If position exists but SL price is missing from state, try to set it? Or log warning?
                     # For simplicity, let's assume the bot *must* set the SL order on entry.
                     # If it's missing, it might indicate a problem or external action.
                     self.logger.warning(f"Position {self.symbol} found, but SL price missing from state. Check orders.")

            elif self.symbol in self.state.get("positions", {}):
                # If no position found on exchange but one is in state, clear state
                self.logger.info(f"Position for {self.symbol} not found on exchange, clearing from state.")
                del self.state["positions"][self.symbol]
                # Also clear associated orders and current trade info
                if self.symbol in self.state.get("orders", {}):
                    del self.state["orders"][self.symbol]
                if self.state.get("current_trade") and self.state["current_trade"].get("symbol") == self.symbol:
                     # Move current trade to history if it exists
                     self.logger.info(f"Moving incomplete trade for {self.symbol} to history.")
                     self.state["trades"].append(self.state["current_trade"]) # Append incomplete trade
                     self.state["current_trade"] = None


            self.save_state()
            self.logger.debug(f"Position updated: {self.current_position}")
        except Exception as e:
            self.logger.error(f"Error updating position: {e}", exc_info=True)
            # Do not re-raise, allow the bot to continue


    def calculate_position_size(self, price: float, side: str) -> float:
        """
        Calculate position size in base currency based on risk management settings.

        Args:
            price: Current price for the asset.
            side: The intended trade side ('long' or 'short').

        Returns:
            float: Calculated position size in base currency, rounded to precision.
                   Returns 0.0 if unable to calculate or size is below minimum.
        """
        if not self.exchange or not self.exchange.has["fetchBalance"]:
             self.logger.error("Exchange not initialized or does not support fetchBalance. Cannot calculate position size.")
             return 0.0

        risk_settings = self.config.get("risk_management", {})
        position_size_pct = risk_settings.get("position_size_pct", 1.0) # % of AVAILABLE balance for position VALUE
        max_risk_per_trade_pct = risk_settings.get("max_risk_per_trade_pct", 1.0) # % of TOTAL EQUITY to risk

        try:
            # Fetch account balance
            balance = retry_api_call(self.exchange.fetch_balance)
            if not balance:
                 self.logger.warning("Failed to fetch balance. Cannot calculate position size.")
                 return 0.0

            quote_currency = self.market_info["quote"] # Use market info quote currency
            # Use total or free balance based on preference/exchange behavior
            # 'free' is generally safer as it's available for new trades
            available_balance = float(balance.get(quote_currency, {}).get("free", 0))
            total_equity = float(balance.get(quote_currency, {}).get("total", available_balance)) # Use total if available, else free

            if available_balance <= 0:
                 self.logger.warning(f"Available balance ({quote_currency}) is zero or negative: {available_balance}. Cannot open position.")
                 return 0.0

            # Calculate base position value based on percentage of available balance
            # This determines the maximum capital allocated per trade
            position_value = available_balance * (position_size_pct / 100)

            # Calculate Stop Loss Price for risk calculation
            sl_price = self.calculate_stop_loss(price, side)

            # Adjust position size based on ATR or Fixed SL risk if enabled
            if risk_settings.get("use_atr_position_sizing", True) and sl_price > 0:
                 # Calculate the price distance to the stop loss
                 sl_distance = abs(price - sl_price)

                 if sl_distance > 0:
                     # Calculate the maximum amount we can trade based on max risk per trade
                     # Risk Amount = Total Equity * (Max Risk % / 100)
                     risk_amount = total_equity * (max_risk_per_trade_pct / 100)

                     # Max amount based on risk = Risk Amount / (Price Distance to SL / Price)
                     # Or simply: Max amount = Risk Amount / (Price Distance to SL)
                     # This amount is in the BASE currency terms of the risk amount (e.g., USD value of BTC amount)
                     # So, Max amount (base currency) = Risk Amount (quote currency) / (Price Distance to SL)
                     amount_from_risk = risk_amount / sl_distance

                     # Take the minimum of the percentage-based amount and the risk-adjusted amount
                     # Ensure we don't exceed the capital allocation AND don't risk more than allowed
                     amount = min(position_value / price, amount_from_risk)
                     self.logger.debug(f"ATR Sizing: Position Value: {position_value:.2f} {quote_currency}, Risk Amount: {risk_amount:.2f} {quote_currency}, SL Price: {sl_price}, SL Distance: {sl_distance:.2f}, Amount from Risk: {amount_from_risk:.6f}")

                 else:
                     # SL distance is zero, effectively infinite risk, or SL is at entry (break even)
                     # Fallback to percentage-based sizing if SL distance is invalid for risk calculation
                     self.logger.warning("Calculated SL distance is zero. Falling back to percentage-based sizing.")
                     amount = position_value / price
            else:
                # Use simple percentage-based sizing if ATR sizing disabled or SL not enabled
                amount = position_value / price
                self.logger.debug(f"Percentage Sizing: Position Value: {position_value:.2f} {quote_currency}, Amount: {amount:.6f}")

            # Apply exchange precision and minimum limits
            amount = round_to_precision(amount, self.precision["amount"], "down")

            # Check against minimum and maximum amount allowed by exchange/config
            min_amount = self.min_amount
            max_amount = self.max_amount if self.max_amount is not None else float('inf')

            if amount < min_amount:
                self.logger.warning(
                    f"Calculated position size {amount} is below minimum {min_amount}. "
                    f"Returning 0.0 to prevent small/invalid orders."
                )
                return 0.0 # Do not trade if size is too small
            elif amount > max_amount:
                 self.logger.warning(
                     f"Calculated position size {amount} exceeds maximum {max_amount}. "
                     f"Capping at maximum amount."
                 )
                 amount = round_to_precision(max_amount, self.precision["amount"], "down")

            self.logger.info(f"Calculated trade amount: {amount} {self.market_info['base']}")
            return amount

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}", exc_info=True)
            # Return 0.0 on error to prevent unintended trades
            return 0.0

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price based on configuration.

        Args:
            entry_price: Entry price for the position.
            side: Position side ('long' or 'short').

        Returns:
            float: Stop loss price, rounded to precision. Returns 0.0 if SL is disabled or calculation fails.
        """
        risk_settings = self.config.get("risk_management", {})
        sl_config = risk_settings.get("stop_loss", {})

        if not sl_config.get("enabled", True) or not self.candles_df or self.candles_df.empty:
            return 0.0

        sl_mode = sl_config.get("mode", "atr").lower()
        sl_price = 0.0

        try:
            if sl_mode == "atr" and "atr" in self.candles_df.columns:
                atr = get_volatility(self.candles_df, self.config.get("strategy", {}).get("indicators", {}))
                atr_multiplier = sl_config.get("atr_multiplier", 2.0)

                if atr > 0:
                    if side == "long":
                        sl_price = entry_price - (atr * atr_multiplier)
                    else: # short
                        sl_price = entry_price + (atr * atr_multiplier)
                    self.logger.debug(f"Calculated ATR Stop Loss: ATR={atr:.4f}, Multiplier={atr_multiplier}, Price={sl_price:.4f}")
                else:
                    self.logger.warning("ATR is zero or not available for ATR stop loss mode. SL disabled.")
                    return 0.0 # Cannot calculate ATR SL if ATR is zero

            elif sl_mode == "fixed_pct":
                fixed_pct = sl_config.get("fixed_pct", 2.0)

                if side == "long":
                    sl_price = entry_price * (1 - fixed_pct / 100)
                else: # short
                    sl_price = entry_price * (1 + fixed_pct / 100)
                self.logger.debug(f"Calculated Fixed % Stop Loss: Pct={fixed_pct:.2f}%, Price={sl_price:.4f}")

            else:
                self.logger.warning(f"Unknown stop loss mode '{sl_mode}'. SL disabled.")
                return 0.0 # Unknown mode

            # Ensure SL price is not negative (though highly unlikely with sensible prices)
            sl_price = max(0.0, sl_price)

            # Round to price precision using tick size if available, otherwise decimal places
            price_precision = self.precision["price"]
            sl_price = round_to_precision(sl_price, price_precision, "down" if side == "long" else "up") # Conservative rounding

            # Ensure SL is on the correct side of the entry price
            if (side == "long" and sl_price >= entry_price) or (side == "short" and sl_price <= entry_price):
                 self.logger.warning(f"Calculated SL price {sl_price} is on the wrong side of entry {entry_price} for {side} position. SL disabled.")
                 return 0.0

            return sl_price

        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}", exc_info=True)
            return 0.0


    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price based on configuration.

        Args:
            entry_price: Entry price for the position.
            side: Position side ('long' or 'short').

        Returns:
            float: Take profit price, rounded to precision. Returns 0.0 if TP is disabled or calculation fails.
        """
        risk_settings = self.config.get("risk_management", {})
        tp_config = risk_settings.get("take_profit", {})

        if not tp_config.get("enabled", True) or not self.candles_df or self.candles_df.empty:
            return 0.0

        tp_mode = tp_config.get("mode", "atr").lower()
        tp_price = 0.0

        try:
            if tp_mode == "atr" and "atr" in self.candles_df.columns:
                atr = get_volatility(self.candles_df, self.config.get("strategy", {}).get("indicators", {}))
                atr_multiplier = tp_config.get("atr_multiplier", 4.0)

                if atr > 0:
                    if side == "long":
                        tp_price = entry_price + (atr * atr_multiplier)
                    else: # short
                        tp_price = entry_price - (atr * atr_multiplier)
                    self.logger.debug(f"Calculated ATR Take Profit: ATR={atr:.4f}, Multiplier={atr_multiplier}, Price={tp_price:.4f}")
                else:
                     self.logger.warning("ATR is zero or not available for ATR take profit mode. TP disabled.")
                     return 0.0 # Cannot calculate ATR TP if ATR is zero

            elif tp_mode == "fixed_pct":
                fixed_pct = tp_config.get("fixed_pct", 4.0)

                if side == "long":
                    tp_price = entry_price * (1 + fixed_pct / 100)
                else: # short
                    tp_price = entry_price * (1 - fixed_pct / 100)
                self.logger.debug(f"Calculated Fixed % Take Profit: Pct={fixed_pct:.2f}%, Price={tp_price:.4f}")
            else:
                self.logger.warning(f"Unknown take profit mode '{tp_mode}'. TP disabled.")
                return 0.0 # Unknown mode

            # Ensure TP price is not negative
            tp_price = max(0.0, tp_price)

            # Round to price precision
            price_precision = self.precision["price"]
            tp_price = round_to_precision(tp_price, price_precision, "up" if side == "long" else "down") # Aggressive rounding

            # Ensure TP is on the correct side of the entry price
            if (side == "long" and tp_price <= entry_price) or (side == "short" and tp_price >= entry_price):
                 self.logger.warning(f"Calculated TP price {tp_price} is on the wrong side of entry {entry_price} for {side} position. TP disabled.")
                 return 0.0


            return tp_price

        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}", exc_info=True)
            return 0.0

    def execute_entry(self, side: str) -> Dict:
        """
        Execute an entry order for the given side ('buy' or 'sell').

        Args:
            side: Order side ('buy' for long, 'sell' for short).

        Returns:
            Dict: Order result information.
        """
        if not self.exchange or not self.exchange.has["createMarketOrder"]:
             self.logger.error("Exchange not initialized or does not support market orders. Cannot execute entry.")
             return {"success": False, "message": "Exchange not ready"}
        if self.current_position:
            self.logger.warning("Attempted to enter position, but a position already exists.")
            return {"success": False, "message": "Position already exists"}

        order_side = side.lower() # 'buy' or 'sell'
        position_side = "long" if order_side == "buy" else "short"

        try:
            # Get current market price for size calculation
            ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
            if not ticker or "last" not in ticker:
                 self.logger.error("Failed to fetch ticker price. Cannot execute entry.")
                 return {"success": False, "message": "Failed to fetch ticker price"}

            current_price = ticker["last"]

            # Calculate position size
            amount = self.calculate_position_size(current_price, position_side)

            if amount <= 0:
                 self.logger.warning("Calculated position size is not valid. Skipping entry.")
                 return {"success": False, "message": "Calculated position size is invalid"}

            # Calculate stop loss and take profit prices *before* placing the entry order
            # These will be used to place SL/TP orders *after* entry is filled
            sl_price = self.calculate_stop_loss(current_price, position_side)
            tp_price = self.calculate_take_profit(current_price, position_side)

            self.logger.info(f"Attempting to open {position_side} position: {amount} {self.symbol} at market price (est. {current_price:.4f})")
            if sl_price > 0: self.logger.info(f"Calculated Stop Loss: {sl_price:.4f}")
            if tp_price > 0: self.logger.info(f"Calculated Take Profit: {tp_price:.4f}")


            # Execute market order for entry
            entry_order = retry_api_call(
                self.exchange.create_market_order,
                self.symbol,
                order_side,
                amount
            )

            if not entry_order or "status" not in entry_order:
                 self.logger.error("Entry order creation failed or returned invalid response.")
                 return {"success": False, "message": "Entry order creation failed"}

            self.logger.info(f"Entry order placed: {entry_order['id']} ({entry_order['status']}). Waiting for fill...")

            # Wait briefly and then fetch the order to confirm fill details
            time.sleep(3) # Give the exchange some time to process the market order

            filled_order = retry_api_call(
                self.exchange.fetch_order,
                entry_order["id"],
                self.symbol,
                params={'type': 'market'} # Some exchanges might need type hint
            )

            if not filled_order or filled_order["status"] != "closed":
                # If order isn't closed immediately, it might be partially filled or failed.
                # This simple bot assumes market orders fill quickly.
                # A more advanced bot would handle partial fills and open orders.
                self.logger.warning(f"Entry order {entry_order['id']} not fully filled or status not 'closed': {filled_order.get('status', 'unknown')}. Canceling order and aborting entry.")
                try:
                     retry_api_call(self.exchange.cancel_order, entry_order['id'], self.symbol)
                     self.logger.info(f"Canceled incomplete entry order {entry_order['id']}")
                except Exception as cancel_e:
                     self.logger.error(f"Failed to cancel incomplete entry order {entry_order['id']}: {cancel_e}")

                # Re-fetch position to be sure
                self.update_position()
                return {"success": False, "message": "Entry order not fully filled or failed"}


            # Order is filled. Get actual entry details.
            filled_amount = float(filled_order.get("filled", amount)) # Use filled amount
            actual_entry_price = float(filled_order.get("average", filled_order.get("price", current_price))) # Use average fill price if available

            if filled_amount <= 0:
                 self.logger.error(f"Entry order {entry_order['id']} filled with zero amount. Aborting entry.")
                 self.update_position() # Re-sync position state
                 return {"success": False, "message": "Entry order filled with zero amount"}

            self.logger.info(
                f"Entry order {entry_order['id']} filled. "
                f"Opened {position_side} position of {filled_amount} {self.symbol} at {actual_entry_price:.4f}"
            )

            # Update position state based on the filled order (this might be slightly stale,
            # calling update_position() again later is safer)
            self.state["positions"][self.symbol] = {
                "side": position_side,
                "amount": filled_amount,
                "entry_price": actual_entry_price,
                # SL/TP prices are stored here for state tracking, but actual management is via orders
                "sl_price": sl_price if sl_price > 0 else None,
                "tp_price": tp_price if tp_price > 0 else None,
                "unrealizedPnl": 0.0, # Pnl starts at 0
                "liquidation_price": 0.0, # Will be updated by update_position()
                "timestamp": int(time.time() * 1000)
            }

            # Store current trade details for history tracking
            self.state["current_trade"] = {
                 "symbol": self.symbol,
                 "entry_side": position_side,
                 "entry_price": actual_entry_price,
                 "amount": filled_amount,
                 "timestamp": int(time.time() * 1000),
                 "sl_price": sl_price if sl_price > 0 else None,
                 "tp_price": tp_price if tp_price > 0 else None,
                 "entry_order_id": entry_order["id"]
            }

            # Place SL and TP orders if enabled and calculated
            self.state["orders"][self.symbol] = {} # Initialize orders for this symbol

            if sl_price > 0 and self.exchange.has["createOrder"]:
                try:
                    # Stop Loss order side is opposite of position side
                    sl_order_side = "sell" if position_side == "long" else "buy"
                    sl_type = "stop" # Or 'stop_market', depends on exchange and desired behavior
                    # Some exchanges require a 'stopPrice' param for 'stop' orders
                    sl_params = {"stopPrice": sl_price}
                    # Note: Amount for SL/TP should match the filled amount of the entry order

                    sl_order = retry_api_call(
                        self.exchange.create_order,
                        self.symbol,
                        sl_type,
                        sl_order_side,
                        filled_amount, # Use filled amount
                        price=sl_price, # Limit price for 'stop' order type
                        params=sl_params
                    )
                    self.logger.info(
                        f"Placed Stop Loss order {sl_order.get('id', 'N/A')} ({sl_type}) "
                        f"at {sl_price:.4f} for {filled_amount} {self.symbol}"
                    )
                    self.state["orders"][self.symbol]["sl"] = {
                        "id": sl_order.get("id"),
                        "price": sl_price,
                        "amount": filled_amount,
                        "side": sl_order_side,
                        "type": sl_type
                    }
                except Exception as e:
                    self.logger.error(f"Error placing Stop Loss order: {e}", exc_info=True)
                    self.state["orders"][self.symbol]["sl"] = {"error": str(e)} # Record error in state


            if tp_price > 0 and self.exchange.has["createOrder"]:
                try:
                    # Take Profit order side is opposite of position side
                    tp_order_side = "sell" if position_side == "long" else "buy"
                    tp_type = "limit" # Or 'take_profit', depends on exchange
                    # Some exchanges require 'triggerPrice' for 'take_profit' orders
                    # tp_params = {"triggerPrice": tp_price} # Example for trigger price
                    tp_params = {} # Default empty params

                    tp_order = retry_api_call(
                        self.exchange.create_order,
                        self.symbol,
                        tp_type,
                        tp_order_side,
                        filled_amount, # Use filled amount
                        price=tp_price, # Limit price for 'limit' or target price for 'take_profit'
                        params=tp_params
                    )
                    self.logger.info(
                        f"Placed Take Profit order {tp_order.get('id', 'N/A')} ({tp_type}) "
                        f"at {tp_price:.4f} for {filled_amount} {self.symbol}"
                    )
                    self.state["orders"][self.symbol]["tp"] = {
                        "id": tp_order.get("id"),
                        "price": tp_price,
                        "amount": filled_amount,
                        "side": tp_order_side,
                        "type": tp_type
                    }
                except Exception as e:
                    self.logger.error(f"Error placing Take Profit order: {e}", exc_info=True)
                    self.state["orders"][self.symbol]["tp"] = {"error": str(e)} # Record error in state

            # Final state save after orders are placed/attempted
            self.save_state()

            # Re-fetch position after placing SL/TP orders to ensure state is current
            self.update_position()

            return {
                "success": True,
                "side": position_side,
                "amount": filled_amount,
                "price": actual_entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price
            }

        except Exception as e:
            self.logger.error(f"Error executing entry for {self.symbol}: {e}", exc_info=True)
            # Attempt to cancel any potential open orders if an error occurred mid-process
            try:
                 # This is a heuristic; a more robust solution would track order IDs better
                 open_orders = retry_api_call(self.exchange.fetch_open_orders, self.symbol)
                 for order in open_orders:
                      if order["side"].lower() == order_side and order["amount"] == amount: # Basic matching
                           self.logger.warning(f"Attempting to cancel potential leftover order {order['id']} after failed entry.")
                           retry_api_call(self.exchange.cancel_order, order['id'], self.symbol)
            except Exception as cancel_e:
                 self.logger.error(f"Error during cleanup after failed entry: {cancel_e}")

            # Re-sync position state in case of partial fill before error
            self.update_position()

            return {
                "success": False,
                "message": str(e)
            }


    def execute_exit(self) -> Dict:
        """
        Execute an exit order to close the current position.

        Returns:
            Dict: Order result information.
        """
        if not self.current_position:
            self.logger.warning("Attempted to exit position, but no position exists.")
            return {
                "success": False,
                "message": "No position to exit"
            }

        if not self.exchange or not self.exchange.has["createMarketOrder"]:
             self.logger.error("Exchange not initialized or does not support market orders. Cannot execute exit.")
             return {"success": False, "message": "Exchange not ready"}

        position_side = self.current_position["side"]
        exit_side = "sell" if position_side == "long" else "buy"
        amount_to_close = self.current_position["amount"]
        entry_price = self.current_position["entry_price"]
        entry_timestamp = self.state.get("current_trade", {}).get("timestamp") # Get entry timestamp from state

        self.logger.info(f"Attempting to close {position_side} position of {amount_to_close} {self.symbol}")

        try:
            # Cancel any existing SL/TP orders for this symbol
            if self.symbol in self.state.get("orders", {}):
                orders_to_cancel = self.state["orders"][self.symbol].copy() # Iterate over a copy
                for order_type, order_info in orders_to_cancel.items():
                    order_id = order_info.get("id")
                    if order_id:
                        try:
                            self.logger.info(f"Canceling {order_type.upper()} order {order_id} for {self.symbol}")
                            # CCXT cancel_order requires id and symbol
                            cancel_result = retry_api_call(
                                self.exchange.cancel_order,
                                order_id,
                                self.symbol
                            )
                            self.logger.debug(f"Cancel result for {order_id}: {cancel_result}")
                            # Remove from state after successful cancellation
                            del self.state["orders"][self.symbol][order_type]
                            self.save_state() # Save state after each successful cancellation
                        except ccxt.OrderNotFound:
                            self.logger.warning(f"{order_type.upper()} order {order_id} not found on exchange. Removing from state.")
                            if self.symbol in self.state["orders"] and order_type in self.state["orders"][self.symbol]:
                                del self.state["orders"][self.symbol][order_type]
                                self.save_state()
                        except Exception as e:
                            self.logger.error(f"Error canceling {order_type.upper()} order {order_id}: {e}", exc_info=True)
                            # Keep order in state if cancellation failed, maybe retry later or manually
                    else:
                        self.logger.warning(f"SL/TP order info for {order_type} in state is missing ID.")
                        # Optionally clear invalid order info from state
                        # if self.symbol in self.state["orders"] and order_type in self.state["orders"][self.symbol]:
                        #     del self.state["orders"][self.symbol][order_type]
                        #     self.save_state()

            # Execute market order to close position
            # Some exchanges require 'reduceOnly' or similar parameter to ensure it closes an existing position
            # Check CCXT documentation for exchange-specific params for closing positions
            exit_params = {}
            if hasattr(self.exchange, 'has'):
                 if self.exchange.has.get('reduceMargin'): # Example parameter name
                      exit_params['reduceMargin'] = True
                 if self.exchange.has.get('createOrder'): # Check if createOrder supports reduceOnly
                      # This is an example, check specific exchange documentation
                      if self.exchange_id in ['bybit', 'binance'] and self.exchange.has['createOrder']:
                           exit_params['reduceOnly'] = True


            exit_order = retry_api_call(
                self.exchange.create_market_order,
                self.symbol,
                exit_side,
                amount_to_close,
                params=exit_params
            )

            if not exit_order or "status" not in exit_order:
                 self.logger.error("Exit order creation failed or returned invalid response.")
                 return {"success": False, "message": "Exit order creation failed"}

            self.logger.info(f"Exit order placed: {exit_order['id']} ({exit_order['status']}). Waiting for fill...")

            # Wait briefly and then fetch the order to confirm fill details
            time.sleep(3) # Give the exchange some time to process the market order

            filled_exit_order = retry_api_call(
                self.exchange.fetch_order,
                exit_order["id"],
                self.symbol,
                params={'type': 'market'} # Some exchanges might need type hint
            )

            if not filled_exit_order or filled_exit_order["status"] != "closed":
                 self.logger.warning(f"Exit order {exit_order['id']} not fully filled or status not 'closed': {filled_exit_order.get('status', 'unknown')}. Manual intervention may be required.")
                 # Attempt to re-sync position, but state might be inconsistent
                 self.update_position()
                 return {"success": False, "message": "Exit order not fully filled or failed"}


            # Order is filled. Get actual exit details.
            filled_amount = float(filled_exit_order.get("filled", amount_to_close))
            actual_exit_price = float(filled_exit_order.get("average", filled_exit_order.get("price", 0.0))) # Use average fill price

            if filled_amount <= 0:
                 self.logger.error(f"Exit order {exit_order['id']} filled with zero amount. Position may not be closed.")
                 self.update_position() # Re-sync position state
                 return {"success": False, "message": "Exit order filled with zero amount"}

            if actual_exit_price <= 0:
                 self.logger.error(f"Exit order {exit_order['id']} filled with invalid price {actual_exit_price}. Cannot calculate PnL.")
                 self.update_position() # Re-sync position state
                 return {"success": False, "message": "Exit order filled with invalid price"}


            # Calculate PnL based on entry price from state and actual exit price
            pnl = 0.0
            pnl_pct = 0.0
            if entry_price is not None and entry_price > 0:
                if position_side == "long":
                    pnl = (actual_exit_price - entry_price) * filled_amount
                    pnl_pct = ((actual_exit_price / entry_price) - 1) * 100
                else:  # short
                    pnl = (entry_price - actual_exit_price) * filled_amount
                    pnl_pct = ((entry_price / actual_exit_price) - 1) * 100
            else:
                self.logger.warning("Entry price not found in state. Cannot calculate PnL for this exit.")


            # Add trade to history
            trade_record = {
                 "symbol": self.symbol,
                 "entry_side": position_side,
                 "entry_price": entry_price, # Use entry price from state
                 "amount": filled_amount,
                 "entry_timestamp": entry_timestamp, # Use entry timestamp from state
                 "exit_price": actual_exit_price,
                 "exit_timestamp": int(time.time() * 1000),
                 "pnl": pnl,
                 "pnl_pct": pnl_pct,
                 "exit_order_id": exit_order["id"]
            }
            # Include SL/TP prices from the entry trade if available
            if self.state.get("current_trade"):
                 trade_record["sl_price"] = self.state["current_trade"].get("sl_price")
                 trade_record["tp_price"] = self.state["current_trade"].get("tp_price")

            self.state["trades"].append(trade_record)
            self.logger.info(
                f"Closed {position_side} position of {filled_amount} {self.symbol} "
                f"at {actual_exit_price:.4f} | PnL: {pnl:.2f} ({pnl_pct:.2f}%)"
            )

            # Clear position and related info from state
            if self.symbol in self.state["positions"]:
                del self.state["positions"][self.symbol]
            if self.symbol in self.state.get("orders", {}):
                del self.state["orders"][self.symbol] # Clear orders related to this symbol/position
            self.state["current_trade"] = None # Clear current trade details


            # Update balance and equity history after trade
            try:
                balance = retry_api_call(self.exchange.fetch_balance)
                if balance:
                    quote_currency = self.market_info["quote"]
                    total_equity = float(balance.get(quote_currency, {}).get("total", balance.get(quote_currency, {}).get("free", 0)))
                    self.state["equity_history"].append({
                         "timestamp": int(time.time() * 1000),
                         "equity": total_equity
                    })
            except Exception as balance_e:
                 self.logger.error(f"Error fetching balance after exit: {balance_e}")


            self.save_state()

            # Final position update to confirm closure
            self.update_position()


            return {
                "success": True,
                "amount": filled_amount,
                "price": actual_exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            }

        except Exception as e:
            self.logger.error(f"Error executing exit for {self.symbol}: {e}", exc_info=True)
            # Attempt to cancel any potential open orders if an error occurred mid-process
            try:
                 # This is a heuristic; a more robust solution would track order IDs better
                 open_orders = retry_api_call(self.exchange.fetch_open_orders, self.symbol)
                 for order in open_orders:
                      if order["side"].lower() == exit_side and order["amount"] == amount_to_close: # Basic matching
                           self.logger.warning(f"Attempting to cancel potential leftover order {order['id']} after failed exit.")
                           retry_api_call(self.exchange.cancel_order, order['id'], self.symbol)
            except Exception as cancel_e:
                 self.logger.error(f"Error during cleanup after failed exit: {cancel_e}")

            # Re-sync position state in case of partial fill before error
            self.update_position()

            return {
                "success": False,
                "message": str(e)
            }


    def check_break_even(self) -> None:
        """Check and adjust stop loss to break even if configured"""
        if not self.current_position or not self.exchange or not self.exchange.has["fetchTicker"] or not self.exchange.has["cancelOrder"] or not self.exchange.has["createOrder"]:
            # Need a position, exchange, ticker, cancel, and create order capabilities
            return

        risk_settings = self.config.get("risk_management", {})
        break_even_config = risk_settings.get("break_even", {})

        if not break_even_config.get("enabled", False):
            return # Break even feature is disabled

        # Check if we have an active SL order tracked in state
        if not self.symbol in self.state.get("orders", {}) or "sl" not in self.state["orders"][self.symbol]:
             # No stop loss order currently tracked for this position
             # This could happen if SL was disabled, or if the order was filled/cancelled externally
             # If SL is supposed to be active, this is a warning sign.
             # For break-even, we only act if an SL order exists.
             return

        # Get break even activation percentage
        activation_pct = break_even_config.get("activation_pct", 1.0)
        if activation_pct <= 0:
             self.logger.warning("Break-even activation percentage is zero or negative. Disabling break-even check.")
             return

        # Get current price
        try:
            ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
            if not ticker or "last" not in ticker:
                 self.logger.warning("Failed to fetch ticker price for break-even check.")
                 return
            current_price = ticker["last"]
        except Exception as e:
            self.logger.error(f"Error fetching ticker for break-even check: {e}", exc_info=True)
            return

        # Get position details from state
        position_state = self.state["positions"][self.symbol]
        entry_price = position_state.get("entry_price")
        side = position_state.get("side")
        amount = position_state.get("amount")
        current_sl_price = position_state.get("sl_price") # Get original SL price from state for comparison

        if entry_price is None or amount is None or side is None or current_sl_price is None or current_sl_price <= 0:
             self.logger.warning("Missing position details or original SL price in state. Cannot check break-even.")
             return

        # Calculate current profit percentage
        profit_pct = 0.0
        if entry_price > 0:
            if side == "long":
                profit_pct = ((current_price / entry_price) - 1) * 100
            else:  # short
                profit_pct = ((entry_price / current_price) - 1) * 100
        else:
             self.logger.warning("Entry price is zero or invalid. Cannot check break-even profit.")
             return

        # Calculate the break-even price (entry price + a small buffer for fees/slippage)
        # The buffer direction depends on the trade side
        buffer_pct = 0.05 # 0.05% buffer example
        if side == "long":
            break_even_price = entry_price * (1 + buffer_pct / 100)
        else: # short
            break_even_price = entry_price * (1 - buffer_pct / 100)

        # Round break-even price to precision
        price_precision = self.precision["price"]
        break_even_price = round_to_precision(
             break_even_price,
             price_precision,
             "up" if side == "long" else "down" # Round against the position for safety
        )

        # Check if profit exceeds activation percentage AND current price has passed the break-even price
        # We also check if the current stop loss is already at or past the break-even price
        should_move_to_be = False
        if side == "long":
             if profit_pct >= activation_pct and current_price >= break_even_price and current_sl_price < break_even_price:
                  should_move_to_be = True
         elif side == "short":
             if profit_pct >= activation_pct and current_price <= break_even_price and current_sl_price > break_even_price:
                  should_move_to_be = True

        if should_move_to_be:
             self.logger.info(f"Break-even condition met ({profit_pct:.2f}% profit). Moving SL to {break_even_price:.4f}")

             sl_order_info = self.state["orders"][self.symbol]["sl"]
             sl_order_id = sl_order_info.get("id")
             sl_order_side = sl_order_info.get("side")
             sl_order_type = sl_order_info.get("type", "stop")

             if not sl_order_id:
                  self.logger.error("SL order ID not found in state. Cannot move SL to break-even.")
                  return # Cannot proceed without the order ID

             try:
                 # First, cancel the existing stop loss order
                 self.logger.info(f"Canceling existing SL order {sl_order_id}")
                 cancel_result = retry_api_call(
                     self.exchange.cancel_order,
                     sl_order_id,
                     self.symbol
                 )
                 self.logger.debug(f"Cancel result for {sl_order_id}: {cancel_result}")

                 # Place a new stop loss order at the calculated break-even price
                 # Ensure amount is still the current position amount
                 current_amount = self.current_position.get("amount")
                 if current_amount is None or current_amount <= 0:
                      self.logger.error("Current position amount is invalid. Cannot place new SL order.")
                      # Remove old SL from state as it was cancelled
                      del self.state["orders"][self.symbol]["sl"]
                      self.save_state()
                      return

                 new_sl_params = {"stopPrice": break_even_price} # Example param, check exchange
                 # Use the same type and side as the original SL order
                 new_sl_order = retry_api_call(
                     self.exchange.create_order,
                     self.symbol,
                     sl_order_type, # e.g., 'stop' or 'stop_market'
                     sl_order_side, # 'sell' for long, 'buy' for short
                     current_amount,
                     price=break_even_price, # Limit price for 'stop' order type
                     params=new_sl_params
                 )

                 # Update stop loss order in state
                 self.state["orders"][self.symbol]["sl"] = {
                     "id": new_sl_order.get("id"),
                     "price": break_even_price,
                     "amount": current_amount,
                     "side": sl_order_side,
                     "type": sl_order_type
                 }
                 # Update SL price in position state as well
                 self.state["positions"][self.symbol]["sl_price"] = break_even_price

                 self.save_state()
                 self.logger.info(
                     f"Successfully moved stop loss to break even at {break_even_price:.4f}"
                 )

             except ccxt.OrderNotFound:
                  self.logger.warning(f"Existing SL order {sl_order_id} not found on exchange during break-even check. Assuming it was filled or cancelled externally.")
                  # Clear the old SL from state
                  if self.symbol in self.state["orders"] and "sl" in self.state["orders"][self.symbol]:
                       del self.state["orders"][self.symbol]["sl"]
                       self.save_state()
                  # Do NOT place a new SL if the old one wasn't found (might imply position is already closed)
                  self.update_position() # Re-sync position state

             except Exception as e:
                 self.logger.error(f"Error adjusting stop loss to break even: {e}", exc_info=True)
                 # If placing the new order failed after canceling the old one, state is inconsistent.
                 # Log error and rely on manual intervention or next loop iteration to fix.


    def run_once(self) -> None:
        """
        Execute a single iteration of the trading bot logic.
        """
        self.logger.info(f"--- Running bot cycle for {self.symbol} ({self.timeframe}) ---")
        start_time = time.time()

        try:
            # 1. Update market data (candles & indicators)
            self.update_candles()
            if self.candles_df is None or self.candles_df.empty:
                self.logger.warning("No valid candle data after update. Skipping signal calculation and trading logic.")
                self.state["last_update"] = int(time.time() * 1000)
                self.save_state()
                return # Cannot proceed without data

            # 2. Update current position information
            self.update_position()

            # 3. Check if we should exit an existing position
            if self.current_position:
                self.logger.debug(f"Current Position: {self.current_position['side'].upper()} {self.current_position['amount']} at {self.current_position['entry_price']:.4f}")
                # Check for signal-based exit
                if self.should_exit_trade():
                    self.logger.info("Exit signal detected.")
                    self.execute_exit()
                else:
                    # If not exiting by signal, check if we should move stop loss to break even
                    self.check_break_even()
                    # Note: SL/TP fulfillment is handled by the exchange and detected by update_position()

            # 4. Check if we should enter a new position (only if no position is open)
            else:
                self.logger.debug("No active position.")
                should_enter, side = self.should_enter_trade()
                if should_enter and side in ["long", "short"]:
                    self.logger.info(f"Entry signal detected: {side.upper()}")
                    # Convert side to order side ('buy' or 'sell')
                    order_side = "buy" if side == "long" else "sell"
                    self.execute_entry(order_side)
                else:
                    self.logger.debug(f"No entry signal ({side}, strength too low).")

            # 5. Record equity history
            try:
                if self.exchange and self.exchange.has["fetchBalance"]:
                    balance = retry_api_call(self.exchange.fetch_balance)
                    if balance:
                        quote_currency = self.market_info["quote"]
                        # Use total equity if available, otherwise base balance + unrealized PnL if in position
                        total_equity = float(balance.get(quote_currency, {}).get("total", balance.get(quote_currency, {}).get("free", 0)))
                        if self.current_position and self.current_position.get("unrealizedPnl") is not None:
                             # For futures, total might include unrealized PnL, but adding explicitly for clarity/spot compatibility
                             # This is a simplified approach. Accurate equity tracking is complex.
                             pass # Assume fetch_balance total is sufficient or calculate manually if needed
                        self.state["equity_history"].append({
                             "timestamp": int(time.time() * 1000),
                             "equity": total_equity
                        })
                        # Keep history size reasonable
                        max_history_points = 1000 # Store up to ~1 week of data if running every 10 mins
                        if len(self.state["equity_history"]) > max_history_points:
                             self.state["equity_history"] = self.state["equity_history"][-max_history_points:]

            except Exception as e:
                 self.logger.error(f"Error recording equity history: {e}", exc_info=True)


        except Exception as e:
            self.logger.error(f"Unhandled error during bot cycle: {e}", exc_info=True)

        # 6. Update state with timestamp and save
        self.state["last_update"] = int(time.time() * 1000)
        self.save_state()

        elapsed = time.time() - start_time
        self.logger.info(f"--- Bot cycle completed in {elapsed:.2f} seconds ---")


    def run(self) -> None:
        """
        Run the trading bot in continuous mode.
        """
        self.logger.info(f"Starting trading bot for {self.symbol} on {self.exchange_id}")
        if self.test_mode:
             self.logger.warning("Running in TEST MODE. No real trades will be executed (on testnet).")
        else:
             self.logger.warning("Running in LIVE MODE. Real trades WILL be executed.")

        loop_interval = self.config.get("loop_interval_seconds", 30)
        if loop_interval < 5: # Prevent excessively short intervals
            self.logger.warning(f"Loop interval {loop_interval}s is too short, setting to 5s.")
            loop_interval = 5

        # Wait for the next candle to close if running live? Or run every X seconds?
        # Running every X seconds is simpler for Termux.
        # A more advanced bot might sync its loop to candle close times.

        try:
            while True:
                self.run_once()

                # Calculate time to sleep until the next cycle
                # This simple sleep doesn't align with candle closes
                self.logger.info(f"Sleeping for {loop_interval} seconds until next cycle...")
                time.sleep(loop_interval)

        except KeyboardInterrupt:
            self.logger.info("Trading bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            self.logger.exception(f"Critical unexpected error, bot stopping: {e}")
            # Consider sending a notification here
            sys.exit(1) # Exit with error code


    def should_enter_trade(self) -> Tuple[bool, str]:
        """
        Determine if we should enter a trade based on signals.

        Returns:
            Tuple[bool, str]: (should_enter, side). side is "long", "short", or ""
        """
        # Don't enter if we already have a position
        if self.current_position:
            self.logger.debug("Skipping entry check: Position already exists.")
            return False, ""

        # Ensure we have enough candles for indicators
        strategy = self.config.get("strategy", {})
        indicators_config = strategy.get("indicators", {})
        # Check if candles_df has enough rows after indicator calculation
        # This depends on the longest indicator window + lookback needed for signal logic
        min_candles_for_signal = 30 # Heuristic: Need at least this many candles for most indicators to stabilize
        if self.candles_df is None or len(self.candles_df) < min_candles_for_signal:
             self.logger.warning(f"Not enough candles ({len(self.candles_df) if self.candles_df is not None else 0}) for reliable signals. Skipping entry check.")
             return False, ""

        entry_threshold = strategy.get("entry_threshold", 0.6)
        if entry_threshold <= 0:
             self.logger.warning("Entry threshold is zero or negative. Disabling entries.")
             return False, ""

        # Check volume filter if enabled
        volume_filter = strategy.get("volume_filter", {})
        if volume_filter.get("enabled", False):
            min_volume_usd = volume_filter.get("min_24h_volume_usd", 1000000)
            try:
                ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
                # CCXT ticker provides 'quoteVolume' which is volume in quote currency (often USD/USDT)
                volume_usd = float(ticker.get("quoteVolume", 0)) # Use quoteVolume
                if volume_usd < min_volume_usd:
                    self.logger.debug(
                        f"Volume filter rejected trade: {volume_usd:.2f} < {min_volume_usd:.2f} USD (24h quote volume)"
                    )
                    return False, ""
                else:
                     self.logger.debug(f"Volume filter passed: {volume_usd:.2f} >= {min_volume_usd:.2f} USD")
            except Exception as e:
                self.logger.error(f"Error checking volume filter: {e}", exc_info=True)
                # Decide whether to allow trade on error or reject. Rejecting is safer.
                self.logger.warning("Failed to check volume filter. Skipping entry check.")
                return False, ""

        # Calculate overall signal
        signal_strength, signal_direction = calculate_signal(
            self.candles_df,
            indicators_config,
            entry_threshold=entry_threshold # Pass threshold for logging/debugging in calculate_signal if needed
        )

        # Check if signal is strong enough to enter
        if signal_direction in ["long", "short"] and signal_strength >= entry_threshold:
            self.logger.info(
                f"Entry signal DETECTED: {signal_direction.upper()} with strength {signal_strength:.2f} >= {entry_threshold:.2f}"
            )
            return True, signal_direction
        else:
             self.logger.debug(
                 f"No strong entry signal: direction={signal_direction}, strength={signal_strength:.2f} < {entry_threshold:.2f}"
             )
             return False, ""


    def should_exit_trade(self) -> bool:
        """
        Determine if we should exit the current position based on signals or other conditions.

        Returns:
            bool: True if should exit, False otherwise
        """
        if not self.current_position:
            self.logger.debug("Skipping exit check: No active position.")
            return False

        # Ensure we have enough candles for indicators
        strategy = self.config.get("strategy", {})
        indicators_config = strategy.get("indicators", {})
        min_candles_for_signal = 30 # Same heuristic as entry
        if self.candles_df is None or len(self.candles_df) < min_candles_for_signal:
             self.logger.warning(f"Not enough candles ({len(self.candles_df) if self.candles_df is not None else 0}) for reliable signals. Skipping signal-based exit check.")
             # Still return False, but maybe check other exit conditions? For now, no exit.
             return False

        exit_threshold = strategy.get("exit_threshold", 0.4)
        if exit_threshold <= 0:
             self.logger.warning("Exit threshold is zero or negative. Disabling signal-based exits.")
             signal_based_exit = False # Cannot exit based on signal if threshold is invalid
        else:
            # Calculate overall signal
            signal_strength, signal_direction = calculate_signal(
                self.candles_df,
                indicators_config,
                exit_threshold=abs(exit_threshold) # Pass threshold for logging/debugging
            )

            # Check for signal-based exit (opposite to current position)
            signal_based_exit = False
            if self.current_position["side"] == "long" and signal_direction == "short":
                if signal_strength >= abs(exit_threshold):
                    self.logger.info(
                        f"Signal-based exit DETECTED for long position: short signal strength {signal_strength:.2f} >= {abs(exit_threshold):.2f}"
                    )
                    signal_based_exit = True
            elif self.current_position["side"] == "short" and signal_direction == "long":
                if signal_strength >= abs(exit_threshold):
                    self.logger.info(
                        f"Signal-based exit DETECTED for short position: long signal strength {signal_strength:.2f} >= {abs(exit_threshold):.2f}"
                    )
                    signal_based_exit = True
            else:
                 self.logger.debug(
                     f"No strong opposing signal for exit: direction={signal_direction}, strength={signal_strength:.2f} < {abs(exit_threshold):.2f}"
                 )

        # Check for specific indicator-based exit conditions (e.g., EMA cross)
        indicator_exit = False
        ema_config = indicators_config.get("ema_cross", {})
        if ema_config.get("enabled", False) and ema_config.get("use_for_exit", False):
            if "ema_cross_signal" in self.candles_df.columns and len(self.candles_df) >= 2:
                ema_cross = self.candles_df["ema_cross_signal"].iloc[-1]
                prev_ema_cross = self.candles_df["ema_cross_signal"].iloc[-2]

                if self.current_position["side"] == "long" and ema_cross == -1 and prev_ema_cross != -1:
                    self.logger.info("Indicator-based exit DETECTED for long position: EMA cross turned bearish.")
                    indicator_exit = True
                elif self.current_position["side"] == "short" and ema_cross == 1 and prev_ema_cross != 1:
                    self.logger.info("Indicator-based exit DETECTED for short position: EMA cross turned bullish.")
                    indicator_exit = True
            else:
                 self.logger.warning("EMA cross indicator data not available for exit check.")


        # Return True if *any* exit condition is met
        return signal_based_exit or indicator_exit


    def run_backtest(self) -> Dict:
        """
        Run the trading bot in backtest mode using historical data.

        Returns:
            Dict: Backtest results
        """
        self.logger.info(f"Starting backtest for {self.symbol} on {self.exchange_id} ({self.timeframe})")

        # Fetch historical data for backtest period
        # You might want to add parameters to specify the backtest date range
        # For now, use the number of candles specified in config
        candles_required = self.config.get("advanced", {}).get("candles_required", 200)
        self.logger.info(f"Fetching {candles_required} candles for backtest...")
        try:
            # Use fetch_ohlcv with a limit. For longer backtests, you'd need to fetch in chunks
            ohlcv = retry_api_call(
                self.exchange.fetch_ohlcv,
                self.symbol,
                timeframe=self.timeframe,
                limit=candles_required
            )
            historical_data = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            historical_data["timestamp"] = pd.to_datetime(historical_data["timestamp"], unit="ms")
            historical_data.set_index("timestamp", inplace=True)

            if historical_data.empty:
                self.logger.error("No historical data available for backtest")
                return {"error": "No historical data available"}

            # Calculate indicators on the entire historical dataset
            self.logger.info("Calculating indicators for backtest...")
            indicators_config = self.config.get("strategy", {}).get("indicators", {})
            historical_data = calculate_indicators(historical_data, indicators_config)

            # Drop NaN rows created by indicators BEFORE backtesting
            initial_rows = len(historical_data)
            historical_data.dropna(inplace=True)
            rows_dropped = initial_rows - len(historical_data)
            if rows_dropped > 0:
                 self.logger.warning(f"Dropped {rows_dropped} rows with NaN values after indicator calculation.")


            if historical_data.empty:
                 self.logger.error("No valid data remaining after indicator calculation for backtest.")
                 return {"error": "No valid data after indicator calculation"}


            self.logger.info(f"Running backtest simulation on {len(historical_data)} candles...")

            # Initialize backtest state
            starting_balance = 10000.0 # Start with a fixed virtual balance for backtest
            backtest_state = {
                "trades": [],
                "current_position": None, # Simulate position {side, entry_price, amount, timestamp}
                "equity_curve": [], # List of {"timestamp": ts, "equity": balance}
                "starting_balance": starting_balance,
                "current_balance": starting_balance,
                "peak_balance": starting_balance,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "drawdown_history": [], # List of {"timestamp": ts, "drawdown_pct": pct}
            }

            # Run backtest simulation bar by bar
            results = self._run_backtest_simulation(historical_data, backtest_state)

            self.logger.info("Backtest finished.")
            return results

        except Exception as e:
            self.logger.error(f"Error in backtest: {e}", exc_info=True)
            return {"error": str(e)}

    def _run_backtest_simulation(self, data: pd.DataFrame, backtest_state: Dict) -> Dict:
        """
        Run the actual backtest simulation bar by bar.

        Args:
            data: Historical price data (DataFrame with OHLCV and indicators).
            backtest_state: Initial backtest state dictionary.

        Returns:
            Dict: Updated backtest state including results.
        """
        strategy = self.config.get("strategy", {})
        indicators_config = strategy.get("indicators", {})
        entry_threshold = strategy.get("entry_threshold", 0.6)
        exit_threshold = strategy.get("exit_threshold", 0.4)
        risk_settings = self.config.get("risk_management", {})
        position_size_pct = risk_settings.get("position_size_pct", 1.0)

        # Ensure backtest starts after indicators have enough data
        min_lookback = max([
            indicators_config.get("rsi", {}).get("window", 0) if indicators_config.get("rsi", {}).get("enabled", False) else 0,
            indicators_config.get("macd", {}).get("slow_period", 0) + indicators_config.get("macd", {}).get("signal_period", 0) if indicators_config.get("macd", {}).get("enabled", False) else 0,
            indicators_config.get("bollinger_bands", {}).get("window", 0) if indicators_config.get("bollinger_bands", {}).get("enabled", False) else 0,
            indicators_config.get("ema_cross", {}).get("slow_period", 0) if indicators_config.get("ema_cross", {}).get("enabled", False) else 0,
            indicators_config.get("atr", {}).get("window", 0) if indicators_config.get("atr", {}).get("enabled", False) else 0,
        ]) + 1 # Need at least one extra bar for signal calculation

        if len(data) <= min_lookback:
             self.logger.warning(f"Not enough data for backtest after indicator lookback ({len(data)} <= {min_lookback}). Skipping simulation.")
             return backtest_state # Return initial state

        for i in range(min_lookback, len(data)):
            # Use data up to the current bar (i) to calculate signals for the *next* bar's potential entry/exit
            # Or, more commonly in backtesting, use the *closed* bar (data.iloc[i-1]) to make decision for the *open* of bar i.
            # Let's use the closed bar `i-1` to decide for the current bar `i`.
            # This simulates making a decision right after a candle closes.
            decision_bar_index = i - 1
            current_bar_index = i

            if decision_bar_index < 0: continue # Should not happen with min_lookback check

            # Get data slice ending at the decision bar
            data_slice = data.iloc[:i] # Slice includes data up to i-1
            decision_bar = data.iloc[decision_bar_index]
            current_bar = data.iloc[current_bar_index]

            # --- Check for Exit ---
            if backtest_state["current_position"]:
                position = backtest_state["current_position"]
                should_exit = False

                # Check signal-based exit using data up to the decision bar
                signal_strength, signal_direction = calculate_signal(
                    data_slice, # Use data up to the decision bar
                    indicators_config,
                    exit_threshold=abs(exit_threshold)
                )

                # Check for signal-based exit (opposite to current position)
                if position["side"] == "long" and signal_direction == "short" and signal_strength >= abs(exit_threshold):
                    should_exit = True
                elif position["side"] == "short" and signal_direction == "long" and signal_strength >= abs(exit_threshold):
                    should_exit = True

                # Check for indicator-specific exit (e.g., EMA cross)
                ema_config = indicators_config.get("ema_cross", {})
                if not should_exit and ema_config.get("enabled", False) and ema_config.get("use_for_exit", False):
                    if "ema_cross_signal" in data_slice.columns and len(data_slice) >= 2:
                        ema_cross = data_slice["ema_cross_signal"].iloc[-1]
                        prev_ema_cross = data_slice["ema_cross_signal"].iloc[-2]

                        if position["side"] == "long" and ema_cross == -1 and prev_ema_cross != -1:
                            should_exit = True
                        elif position["side"] == "short" and ema_cross == 1 and prev_ema_cross != 1:
                            should_exit = True


                # Execute exit if needed (simulate filling at the OPEN of the *current* bar)
                if should_exit:
                    # Simulate filling the exit order at the open price of the current bar
                    exit_price = current_bar["open"]
                    entry_price = position["entry_price"]
                    amount = position["amount"]

                    # Calculate PnL
                    if position["side"] == "long":
                        pnl = (exit_price - entry_price) * amount
                        pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price != 0 else 0
                    else:  # short
                        pnl = (entry_price - exit_price) * amount
                        pnl_pct = ((entry_price / exit_price) - 1) * 100 if exit_price != 0 else 0

                    # Update balance
                    backtest_state["current_balance"] += pnl

                    # Record trade
                    trade = {
                        "entry_timestamp": position["timestamp"],
                        "exit_timestamp": current_bar.name.timestamp() * 1000,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "amount": amount,
                        "side": position["side"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "signal" # Indicate exit reason
                    }
                    backtest_state["trades"].append(trade)

                    # Clear position
                    backtest_state["current_position"] = None

                    self.logger.debug(
                        f"Backtest exit ({trade['exit_reason']}): {position['side']} at {exit_price:.4f} "
                        f"| PnL: {pnl:.2f} ({pnl_pct:.2f}%) "
                        f"| Balance: {backtest_state['current_balance']:.2f}"
                    )

            # --- Check for Entry ---
            # Only check for entry if not currently in a position
            if not backtest_state["current_position"]:
                # Check signal-based entry using data up to the decision bar
                should_enter, entry_side = self.should_enter_trade_backtest(data_slice, strategy) # Use a backtest specific check if needed, or reuse should_enter_trade logic


                # Check if signal is strong enough to enter
                if should_enter and entry_side in ["long", "short"]:
                    # Simulate filling the entry order at the open price of the *current* bar
                    entry_price = current_bar["open"]

                    # Calculate position size based on current balance and risk settings
                    # Simplified size calculation for backtest: % of current equity
                    position_value = backtest_state["current_balance"] * (position_size_pct / 100)
                    amount = position_value / entry_price if entry_price > 0 else 0

                    # Simulate calculating SL/TP for tracking (orders are not placed in backtest)
                    sl_price = self.calculate_stop_loss(entry_price, entry_side)
                    tp_price = self.calculate_take_profit(entry_price, entry_side)

                    # Record position
                    if amount > 0:
                        backtest_state["current_position"] = {
                            "side": entry_side,
                            "entry_price": entry_price,
                            "amount": amount,
                            "timestamp": current_bar.name.timestamp() * 1000,
                            "sl_price": sl_price if sl_price > 0 else None,
                            "tp_price": tp_price if tp_price > 0 else None
                        }

                        self.logger.debug(
                            f"Backtest entry: {entry_side} at {entry_price:.4f} "
                            f"| Amount: {amount:.6f} "
                            f"| SL: {sl_price:.4f} "
                            f"| TP: {tp_price:.4f}"
                        )
                    else:
                        self.logger.debug(f"Backtest entry signal ignored: Calculated amount is zero or negative ({amount:.6f})")


            # --- Update Equity and Drawdown ---
            # Calculate current equity including unrealized PnL if in position
            current_equity = backtest_state["current_balance"]
            if backtest_state["current_position"]:
                 position = backtest_state["current_position"]
                 # Estimate unrealized PnL using the close price of the current bar
                 unrealized_pnl = 0.0
                 if position["entry_price"] > 0 and current_bar["close"] > 0:
                      if position["side"] == "long":
                           unrealized_pnl = (current_bar["close"] - position["entry_price"]) * position["amount"]
                      else: # short
                           unrealized_pnl = (position["entry_price"] - current_bar["close"]) * position["amount"]
                 current_equity += unrealized_pnl


            backtest_state["equity_curve"].append({
                "timestamp": current_bar.name.timestamp() * 1000,
                "equity": current_equity
            })

            # Update drawdown metrics using current equity
            if current_equity > backtest_state["peak_balance"]:
                backtest_state["peak_balance"] = current_equity

            current_drawdown = backtest_state["peak_balance"] - current_equity
            current_drawdown_pct = (current_drawdown / backtest_state["peak_balance"]) * 100 if backtest_state["peak_balance"] > 0 else 0

            if current_drawdown > backtest_state["max_drawdown"]:
                backtest_state["max_drawdown"] = current_drawdown
                backtest_state["max_drawdown_pct"] = current_drawdown_pct

            backtest_state["drawdown_history"].append({
                 "timestamp": current_bar.name.timestamp() * 1000,
                 "drawdown_pct": current_drawdown_pct
            })


        # --- End of Simulation ---
        # If the backtest ends with an open position, close it at the last bar's close price
        if backtest_state["current_position"]:
            position = backtest_state["current_position"]
            last_bar = data.iloc[-1]
            exit_price = last_bar["close"] # Simulate closing at the final close price
            entry_price = position["entry_price"]
            amount = position["amount"]

            # Calculate PnL for the final trade
            if position["side"] == "long":
                pnl = (exit_price - entry_price) * amount
                pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price != 0 else 0
            else: # short
                pnl = (entry_price - exit_price) * amount
                pnl_pct = ((entry_price / exit_price) - 1) * 100 if exit_price != 0 else 0

            # Update balance
            backtest_state["current_balance"] += pnl

            # Record the final trade
            trade = {
                "entry_timestamp": position["timestamp"],
                "exit_timestamp": last_bar.name.timestamp() * 1000,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "amount": amount,
                "side": position["side"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": "end_of_backtest" # Indicate reason
            }
            backtest_state["trades"].append(trade)

            backtest_state["current_position"] = None # Clear position


        # --- Calculate Final Backtest Metrics ---
        total_trades = len(backtest_state["trades"])
        profitable_trades = sum(1 for trade in backtest_state["trades"] if trade.get("pnl", 0) > 0)
        losing_trades = total_trades - profitable_trades
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(trade.get("pnl", 0) for trade in backtest_state["trades"])
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0

        winning_pnls = [trade["pnl"] for trade in backtest_state["trades"] if trade.get("pnl", 0) > 0]
        losing_pnls = [trade["pnl"] for trade in backtest_state["trades"] if trade.get("pnl", 0) <= 0] # Include 0 PnL in losing
        avg_winning_pnl = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
        avg_losing_pnl = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0 # This will be a negative or zero number
        profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if sum(losing_pnls) != 0 else float('inf') # Sum of winning PnLs / Sum of losing PnLs (absolute)

        total_return = backtest_state["current_balance"] - backtest_state["starting_balance"]
        total_return_pct = (backtest_state["current_balance"] / backtest_state["starting_balance"] - 1) * 100 if backtest_state["starting_balance"] > 0 else 0

        results = {
            "starting_balance": backtest_state["starting_balance"],
            "final_balance": backtest_state["current_balance"],
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "avg_winning_pnl": avg_winning_pnl,
            "avg_losing_pnl": avg_losing_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": backtest_state["max_drawdown"],
            "max_drawdown_pct": backtest_state["max_drawdown_pct"],
            "trades": backtest_state["trades"],
            "equity_curve": backtest_state["equity_curve"],
            "drawdown_history": backtest_state["drawdown_history"] # Include drawdown history
        }

        self.logger.info(f"Backtest Summary: Trades={total_trades}, Win Rate={win_rate:.2f}%, Total Return={total_return_pct:.2f}%, Max Drawdown={backtest_state['max_drawdown_pct']:.2f}%")

        return results


    def should_enter_trade_backtest(self, data_slice: pd.DataFrame, strategy: Dict) -> Tuple[bool, str]:
        """
        Determine if we should enter a trade based on signals, specifically for backtesting.
        This is a simplified version of should_enter_trade for backtesting.

        Args:
            data_slice: DataFrame slice containing historical data up to the decision bar.
            strategy: Strategy configuration dictionary.

        Returns:
            Tuple[bool, str]: (should_enter, side). side is "long", "short", or ""
        """
        # In backtest, we don't check for existing position here, that's handled by the simulation loop
        # Volume filter check can be included if needed, but adds complexity to backtest data requirements

        indicators_config = strategy.get("indicators", {})
        entry_threshold = strategy.get("entry_threshold", 0.6)

        # Calculate overall signal using the provided data slice
        signal_strength, signal_direction = calculate_signal(
            data_slice,
            indicators_config,
            entry_threshold=entry_threshold
        )

        # Check if signal is strong enough to enter
        if signal_direction in ["long", "short"] and signal_strength >= entry_threshold:
            return True, signal_direction
        else:
            return False, ""

# Add constants or functions here that might be used by both bot and web interface if needed,
# though putting them in utils.py is generally preferred.
EOL

# Create web_interface.py
echo "Creating web_interface.py..."
cat > web_interface.py << 'EOL'
"""
Web Interface for Trading Bot

This module implements a Flask web interface for the trading bot
to display statistics, control the bot, and view trading history.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from flask import Flask, jsonify, render_template, request, session, send_from_directory
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger = logging.getLogger("web_interface")
logger.setLevel(logging.DEBUG) # Set default level for this module

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
# Use a secret key from environment variable or generate one for production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "trading_bot_development_key_change_this")

# File paths (relative to the bot's root directory where main.py is)
CONFIG_FILE = "config.json"
STATE_FILE = "bot_state.json"
LOG_FILE = "bot_logs/trading_bot.log"


def load_json_file(filepath: str, default_data: Any = None) -> Any:
    """Helper to load JSON data from a file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return default_data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filepath}")
        return default_data # Return default or empty on error
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}", exc_info=True)
        return default_data


def save_json_file(filepath: str, data: Any) -> bool:
    """Helper to save JSON data to a file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving file {filepath}: {e}", exc_info=True)
        return False

# Load config and state globally or within request context?
# Loading within request context ensures we get the latest state/config
# but adds file I/O overhead to every request. For a simple bot, this is fine.
# For high traffic, consider caching or a state manager process.

def get_config() -> Dict:
    """Load configuration."""
    # Use a default config structure if file loading fails
    default_config_structure = {
        "exchange": "bybit", "symbol": "BTC/USDT:USDT", "timeframe": "15m",
        "api_key": "", "api_secret": "", "test_mode": True,
        "risk_management": {"position_size_pct": 1.0, "max_risk_per_trade_pct": 1.0, "use_atr_position_sizing": True, "stop_loss": {"enabled": True}, "take_profit": {"enabled": True}, "break_even": {"enabled": False}},
        "strategy": {"entry_threshold": 0.5, "exit_threshold": 0.3, "volume_filter": {"enabled": False}, "indicators": {"rsi": {"enabled": True}, "macd": {"enabled": True}, "bollinger_bands": {"enabled": True}, "ema_cross": {"enabled": True}}},
        "advanced": {"candles_required": 100, "price_precision": 8, "amount_precision": 8, "min_amount": 0.000001},
        "loop_interval_seconds": 15
    }
    loaded_config = load_json_file(CONFIG_FILE, default_data=default_config_structure)
    # Ensure all nested keys exist by merging with defaults if needed
    # This is a basic merge, a deep merge would be better for complex nested structures
    for key, default_value in default_config_structure.items():
         if key not in loaded_config or not isinstance(loaded_config[key], type(default_value)):
              loaded_config[key] = default_value
         elif isinstance(default_value, dict):
              for sub_key, sub_default_value in default_value.items():
                   if sub_key not in loaded_config[key] or not isinstance(loaded_config[key][sub_key], type(sub_default_value)):
                        loaded_config[key][sub_key] = sub_default_value
                   elif isinstance(sub_default_value, dict):
                        for sub_sub_key, sub_sub_default_value in sub_default_value.items():
                             if sub_sub_key not in loaded_config[key][sub_key] or not isinstance(loaded_config[key][sub_key][sub_sub_key], type(sub_sub_default_value)):
                                  loaded_config[key][sub_key][sub_sub_key] = sub_sub_default_value


    return loaded_config


def get_state() -> Dict:
    """Load bot state."""
    default_state = {"positions": {}, "orders": {}, "trades": [], "last_update": 0, "equity_history": [], "current_trade": None}
    loaded_state = load_json_file(STATE_FILE, default_data=default_state)
    # Ensure all default keys exist in the loaded state
    for key, default_value in default_state.items():
         loaded_state.setdefault(key, default_value)

    # Ensure nested dictionaries exist
    loaded_state.setdefault("positions", {})
    loaded_state.setdefault("orders", {})

    return loaded_state

def get_log_content(lines: int = 100) -> List[str]:
    """Reads the last N lines from the log file."""
    try:
        if not os.path.exists(LOG_FILE):
            return ["Log file not found."]
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            # Read lines from the end
            f.seek(0, 2) # Go to the end of the file
            fsize = f.tell()
            f.seek(max(fsize - 1024 * 50, 0), 0) # Go back ~50KB or start
            all_lines = f.readlines()
            return all_lines[-lines:] # Get the last N lines
    except Exception as e:
        logger.error(f"Error reading log file: {e}", exc_info=True)
        return [f"Error reading log file: {e}"]


@app.route("/")
def index():
    """Render main dashboard page"""
    # Redirect directly to dashboard for simplicity in a single-page-app feel
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Render main dashboard page (using index.html as the base)"""
    config = get_config()
    state = get_state()

    # Format timestamps in trades for display
    for trade in state.get("trades", []):
        if "timestamp" in trade:
             trade["entry_time_str"] = datetime.fromtimestamp(trade["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        if "exit_timestamp" in trade:
             trade["exit_time_str"] = datetime.fromtimestamp(trade["exit_timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        # Ensure PnL is float for display
        trade["pnl"] = float(trade.get("pnl", 0))
        trade["pnl_pct"] = float(trade.get("pnl_pct", 0))


    # Format position details for display
    formatted_positions = {}
    for symbol, position in state.get("positions", {}).items():
         formatted_positions[symbol] = {
              "side": position.get("side", "unknown"),
              "amount": float(position.get("amount", 0)),
              "entry_price": float(position.get("entry_price", 0)),
              "unrealizedPnl": float(position.get("unrealizedPnl", 0)),
              "liquidation_price": float(position.get("liquidationPrice", 0)),
              "timestamp_str": datetime.fromtimestamp(position.get("timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S") if position.get("timestamp", 0) > 0 else "N/A",
              "sl_price": float(position.get("sl_price", 0)) if position.get("sl_price") is not None else None,
              "tp_price": float(position.get("tp_price", 0)) if position.get("tp_price") is not None else None,
         }


    return render_template(
        "dashboard.html",
        config=config,
        state={
            "positions": formatted_positions,
            "trades": state.get("trades", []),
            "last_update": state.get("last_update", 0)
        }, # Pass formatted state
        last_update_str=datetime.fromtimestamp(state.get("last_update", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S UTC") if state.get("last_update", 0) > 0 else "Never"
    )


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """API endpoint to get current configuration"""
    return jsonify(get_config())


@app.route("/api/config", methods=["POST"])
def api_update_config():
    """API endpoint to update configuration"""
    try:
        new_config = request.json
        if not new_config:
             return jsonify({"status": "error", "message": "No JSON data received"}), 400

        # Basic validation/sanitization (expand as needed)
        # Ensure numeric values are converted
        if 'risk_management' in new_config:
            rm = new_config['risk_management']
            if 'position_size_pct' in rm: rm['position_size_pct'] = float(rm['position_size_pct'])
            if 'max_risk_per_trade_pct' in rm: rm['max_risk_per_trade_pct'] = float(rm['max_risk_per_trade_pct'])
            if 'stop_loss' in rm:
                sl = rm['stop_loss']
                if 'atr_multiplier' in sl: sl['atr_multiplier'] = float(sl['atr_multiplier'])
                if 'fixed_pct' in sl: sl['fixed_pct'] = float(sl['fixed_pct'])
            if 'take_profit' in rm:
                tp = rm['take_profit']
                if 'atr_multiplier' in tp: tp['atr_multiplier'] = float(tp['atr_multiplier'])
                if 'fixed_pct' in tp: tp['fixed_pct'] = float(tp['fixed_pct'])
            if 'break_even' in rm:
                be = rm['break_even']
                if 'activation_pct' in be: be['activation_pct'] = float(be['activation_pct'])

        if 'strategy' in new_config:
            strat = new_config['strategy']
            if 'entry_threshold' in strat: strat['entry_threshold'] = float(strat['entry_threshold'])
            if 'exit_threshold' in strat: strat['exit_threshold'] = float(strat['exit_threshold'])
            if 'volume_filter' in strat:
                 vf = strat['volume_filter']
                 if 'min_24h_volume_usd' in vf: vf['min_24h_volume_usd'] = float(vf['min_24h_volume_usd'])
            if 'indicators' in strat:
                 inds = strat['indicators']
                 for ind_name, ind_config in inds.items():
                      if isinstance(ind_config, dict):
                           if 'window' in ind_config: ind_config['window'] = int(ind_config['window'])
                           if 'weight' in ind_config: ind_config['weight'] = float(ind_config['weight'])
                           if 'overbought' in ind_config: ind_config['overbought'] = float(ind_config['overbought'])
                           if 'oversold' in ind_config: ind_config['oversold'] = float(ind_config['oversold'])
                           if 'fast_period' in ind_config: ind_config['fast_period'] = int(ind_config['fast_period'])
                           if 'slow_period' in ind_config: ind_config['slow_period'] = int(ind_config['slow_period'])
                           if 'signal_period' in ind_config: ind_config['signal_period'] = int(ind_config['signal_period'])
                           if 'std_dev' in ind_config: ind_config['std_dev'] = float(ind_config['std_dev'])

        if 'advanced' in new_config:
             adv = new_config['advanced']
             if 'candles_required' in adv: adv['candles_required'] = int(adv['candles_required'])
             if 'price_precision' in adv: adv['price_precision'] = int(adv['price_precision'])
             if 'amount_precision' in adv: adv['amount_precision'] = int(adv['amount_precision'])
             if 'min_amount' in adv: adv['min_amount'] = float(adv['min_amount'])
             # max_amount could be float or null

        if 'loop_interval_seconds' in new_config: new_config['loop_interval_seconds'] = int(new_config['loop_interval_seconds'])


        if save_json_file(CONFIG_FILE, new_config):
            logger.info("Configuration updated successfully via API.")
            return jsonify({"status": "success", "message": "Configuration updated"})
        else:
            return jsonify({"status": "error", "message": "Failed to save configuration"}), 500
    except Exception as e:
        logger.error(f"Error updating config via API: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/state", methods=["GET"])
def api_get_state():
    """API endpoint to get current bot state"""
    return jsonify(get_state())

@app.route("/api/log", methods=["GET"])
def api_get_log():
    """API endpoint to get recent log entries"""
    lines = request.args.get("lines", 100, type=int)
    log_content = get_log_content(lines=lines)
    return jsonify(log_content)


@app.route("/api/performance", methods=["GET"])
def api_get_performance():
    """API endpoint to get trading performance statistics and equity curve."""
    state = get_state()
    trades = state.get("trades", [])
    equity_history = state.get("equity_history", [])

    # Calculate performance metrics
    total_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
    losing_trades = total_trades - profitable_trades
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0

    total_pnl = sum(trade.get("pnl", 0) for trade in trades)

    # Calculate drawdown from equity history
    # Assuming equity_history is sorted by timestamp
    peak_equity = state.get("equity_history", [{}])[0].get("equity", 0) if state.get("equity_history") else get_state().get("starting_balance", 10000) # Use starting balance if no history
    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    for record in equity_history:
        equity = record.get("equity", peak_equity) # Use peak if equity is missing
        if equity > peak_equity:
            peak_equity = equity
        current_drawdown = peak_equity - equity
        current_drawdown_pct = (current_drawdown / peak_equity) * 100 if peak_equity > 0 else 0.0

        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
            max_drawdown_pct = current_drawdown_pct


    # Prepare data for charts (equity curve)
    equity_chart_data = [{"timestamp": rec["timestamp"], "equity": rec["equity"]} for rec in equity_history]


    # Prepare data for trades table (already formatted in dashboard route, but useful for API too)
    formatted_trades = []
    for trade in trades:
         formatted_trades.append({
             "entry_time": datetime.fromtimestamp(trade.get("timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S") if trade.get("timestamp", 0) > 0 else "N/A",
             "exit_time": datetime.fromtimestamp(trade.get("exit_timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S") if trade.get("exit_timestamp", 0) > 0 else "N/A",
             "symbol": trade.get("symbol", "N/A"),
             "side": trade.get("entry_side", "N/A"),
             "amount": float(trade.get("amount", 0)),
             "entry_price": float(trade.get("entry_price", 0)),
             "exit_price": float(trade.get("exit_price", 0)),
             "pnl": float(trade.get("pnl", 0)),
             "pnl_pct": float(trade.get("pnl_pct", 0)),
             "exit_reason": trade.get("exit_reason", "N/A")
         })


    return jsonify({
        "total_trades": total_trades,
        "profitable_trades": profitable_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        # Add other stats like best/worst trade, profit factor if needed
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "current_positions_count": len(state.get("positions", {})),
        "equity_curve": equity_chart_data,
        "trades_history": formatted_trades # Include full trade history data
    })

# Optional: Serve static files directly if needed, though Flask does this by default
# @app.route('/static/<path:filename>')
# def static_files(filename):
#     return send_from_directory(app.static_folder, filename)


if __name__ == "__main__":
    # This block is mainly for testing web interface development directly
    # In production/Termux usage, run via main.py which handles logging and arguments
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    app.run(host="0.0.0.0", port=5000, debug=True)
EOL

# Create main.py
echo "Creating main.py..."
cat > main.py << 'EOL'
#!/usr/bin/env python3
"""
Cryptocurrency Trading Bot Main Entry Point

This script serves as the entry point for the trading bot application,
loading configurations and starting either the bot or web interface based
on command-line arguments.
"""

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Ensure bot_logs directory exists before configuring file handler
LOG_DIR = "bot_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging early
LOG_FILE_PATH = os.path.join(LOG_DIR, "trading_bot.log")
logging.basicConfig(
    level=logging.INFO, # Default level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5, # Keep up to 5 backup files
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout), # Log to console
    ],
)

# Set log levels for potentially chatty libraries if needed
# logging.getLogger("ccxt").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("werkzeug").setLevel(logging.WARNING) # Flask server logs


logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG) # Set main logger to DEBUG by default

# Import bot and web modules after initial logging config
try:
    from trading_bot import TradingBot
    from web_interface import app as flask_app # Import Flask app instance
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}. Please ensure dependencies are installed and files exist.", exc_info=True)
    sys.exit(1)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot")
    parser.add_argument(
        "--config", default="config.json", help="Path to configuration file (default: config.json)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging level"
    )
    parser.add_argument(
        "--web", action="store_true", help="Start the web interface"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host address for the web interface (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", default=5000, type=int, help="Port for the web interface (default: 5000)"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run in backtest mode using historical data from the exchange"
    )
    parser.add_argument(
        "--symbol", help="Override trading symbol from config file"
    )
    parser.add_argument(
        "--exchange", help="Override exchange from config file"
    )
    parser.add_argument(
        "--run-once", action="store_true", help="Run a single trading cycle and exit (useful for cron jobs)"
    )
    return parser.parse_args()


def main():
    """Main entry point for the trading bot application"""
    args = parse_arguments()

    # Adjust root logger level if debug flag is set
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Root logging level set to DEBUG")

    # Ensure required directories exist (redundant if done by logging config, but safe)
    # ensure_directories() # Moved to logging config creation

    if args.web:
        # Start web interface
        logger.info(f"Starting web interface on {args.host}:{args.port}")
        # Use Flask's built-in development server. For production, use gunicorn/waitress.
        # In Termux, the built-in server is usually sufficient.
        try:
            flask_app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False) # use_reloader=False to avoid running twice
        except Exception as e:
            logger.critical(f"Failed to start web interface: {e}", exc_info=True)
            sys.exit(1)

    else:
        # Start trading bot (continuous or run-once)
        logger.info("Starting trading bot core logic")
        try:
            bot = TradingBot(config_file=args.config) # Pass config file path
            # Override config with command-line arguments if provided
            if args.symbol:
                bot.config["symbol"] = args.symbol
                bot.symbol = args.symbol # Update bot instance attribute
                logger.info(f"Overriding symbol to {args.symbol}")
            if args.exchange:
                bot.config["exchange"] = args.exchange
                bot.exchange_id = args.exchange # Update bot instance attribute
                logger.info(f"Overriding exchange to {args.exchange}")

            # Re-initialize bot if symbol or exchange was overridden
            if args.symbol or args.exchange:
                 logger.info("Re-initializing bot with overridden parameters...")
                 bot.exchange = bot.setup_exchange() # Re-setup exchange
                 if not bot.exchange:
                      logger.critical("Failed to re-initialize exchange with overridden parameters.")
                      sys.exit(1)
                 bot.initialize() # Re-initialize market data etc.


            if args.backtest:
                logger.info("Running in backtest mode")
                backtest_results = bot.run_backtest()
                logger.info("Backtest Results:")
                # Print backtest results in a human-readable format
                for key, value in backtest_results.items():
                    if key not in ["trades", "equity_curve", "drawdown_history"]: # Exclude large data lists
                        logger.info(f"  {key}: {value}")
                # Optionally save results to a file
                try:
                    results_file = "backtest_results.json"
                    with open(results_file, "w") as f:
                        json.dump(backtest_results, f, indent=4)
                    logger.info(f"Full backtest results saved to {results_file}")
                except Exception as e:
                    logger.error(f"Failed to save backtest results: {e}")

            elif args.run_once:
                 logger.info("Running a single bot cycle")
                 bot.run_once()
                 logger.info("Single bot cycle finished.")

            else:
                logger.info("Running in continuous mode")
                bot.run() # This method contains the main loop

        except ConnectionError as e:
             logger.critical(f"Bot failed to start due to exchange connection error: {e}", exc_info=True)
             sys.exit(1)
        except FileNotFoundError as e:
             logger.critical(f"Bot failed to start: Configuration file not found or accessible: {e}", exc_info=True)
             sys.exit(1)
        except json.JSONDecodeError as e:
             logger.critical(f"Bot failed to start: Invalid JSON in configuration or state file: {e}", exc_info=True)
             sys.exit(1)
        except ImportError as e:
             logger.critical(f"Bot failed to start: Missing dependencies or files: {e}", exc_info=True)
             sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            logger.critical(f"Critical unexpected error, bot stopping: {e}", exc_info=True)
            # Consider sending a notification here
            sys.exit(1) # Exit with error code


if __name__ == "__main__":
    main()
EOL

# Create CSS file
echo "Creating custom.css..."
mkdir -p static/css
cat > static/css/custom.css << 'EOL'
/* Custom styles for the Trading Bot interface */

/* General body and text color adjustments for dark theme */
body {
    color: var(--bs-body-color); /* Use Bootstrap's body color variable */
    background-color: var(--bs-body-bg); /* Use Bootstrap's body background variable */
}

h1, h2, h3, h4, h5, h6 {
    color: var(--bs-heading-color); /* Use Bootstrap's heading color variable */
}

a {
    color: var(--bs-link-color); /* Use Bootstrap's link color variable */
}

a:hover {
    color: var(--bs-link-hover-color); /* Use Bootstrap's link hover color variable */
}


/* Sidebar styles */
.sidebar {
    position: fixed;
    top: 0;
    bottom: 0;
    left: 0;
    z-index: 100;
    padding: 48px 0 0;
    /* box-shadow: inset -1px 0 0 rgba(0, 0, 0, .1); Removed for potential dark theme contrast issues */
    border-right: 1px solid var(--bs-border-color); /* Use Bootstrap border color */
    background-color: var(--bs-body-tertiary-bg); /* Use Bootstrap's tertiary background */
}

.sidebar-sticky {
    position: relative;
    top: 0;
    height: calc(100vh - 48px);
    padding-top: .5rem;
    overflow-x: hidden;
    overflow-y: auto;
}

/* Main content padding to clear fixed sidebar and header */
main {
    padding-top: 1rem; /* Add some padding at the top */
}


/* Position related color coding */
.position-long {
    color: var(--bs-success); /* Use Bootstrap success color */
    font-weight: bold;
}

.position-short {
    color: var(--bs-danger); /* Use Bootstrap danger color */
    font-weight: bold;
}

/* Trade history PnL colors */
.pnl-positive {
    color: var(--bs-success);
    font-weight: bold;
}

.pnl-negative {
    color: var(--bs-danger);
    font-weight: bold;
}


/* Card hover effects */
.card {
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    background-color: var(--bs-body-tertiary-bg); /* Use Bootstrap tertiary background */
    border-color: var(--bs-border-color); /* Use Bootstrap border color */
}

.card:hover {
    transform: translateY(-3px);
    /* box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); */
    box-shadow: 0 0.5rem 1rem rgba(var(--bs-body-color-rgb), 0.15); /* Adjusted for dark theme visibility */
}

.card-header {
    background-color: var(--bs-body-secondary-bg); /* Use Bootstrap secondary background */
    border-bottom: 1px solid var(--bs-border-color);
}


/* Chart container */
.chart-container {
    position: relative;
    height: 250px; /* Adjust height as needed */
    width: 100%;
}

/* Nav link styles */
.nav-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.5rem 1rem;
    color: var(--bs-secondary-color); /* Use Bootstrap secondary color */
    font-weight: 500;
}

.nav-link:hover {
    color: var(--bs-primary); /* Use Bootstrap primary color */
}

.nav-link.active {
    color: var(--bs-primary); /* Use Bootstrap primary color */
    background-color: rgba(var(--bs-primary-rgb), 0.1); /* Subtle background for active link */
    border-radius: 5px; /* Add some rounding */
}

.nav-link svg {
    width: 16px;
    height: 16px;
}

/* Performance stats - Adjust background for dark theme */
.performance-stat {
    border-radius: var(--bs-border-radius); /* Use Bootstrap variable */
    padding: 1rem; /* Adjusted padding */
    margin-bottom: 1rem; /* Adjusted margin */
    border: 1px solid var(--bs-border-color); /* Add border */
}

.performance-positive {
    background-color: rgba(var(--bs-success-rgb), 0.15); /* Use Bootstrap success color with transparency */
}

.performance-negative {
    background-color: rgba(var(--bs-danger-rgb), 0.15); /* Use Bootstrap danger color with transparency */
}

.performance-neutral {
    background-color: rgba(var(--bs-secondary-rgb), 0.1); /* Use Bootstrap secondary color with transparency */
}


/* Table styles */
.table {
    color: var(--bs-body-color); /* Ensure table text color matches body color */
    border-color: var(--bs-border-color); /* Use Bootstrap border color */
}
.table th, .table td {
     border-color: var(--bs-border-color); /* Use Bootstrap border color */
}
.table-striped tbody tr:nth-of-type(odd) {
    background-color: rgba(var(--bs-body-color-rgb), 0.05); /* Subtle stripe for dark theme */
}

.table-hover tbody tr:hover {
    background-color: rgba(var(--bs-primary-rgb), 0.1); /* More visible hover for dark theme */
}

/* Form controls with better contrast */
.form-control, .form-select {
    background-color: var(--bs-body-bg); /* Use Bootstrap body background */
    border-color: var(--bs-border-color); /* Use Bootstrap border color */
    color: var(--bs-body-color); /* Use Bootstrap body color */
}

.form-control:focus, .form-select:focus {
    background-color: var(--bs-body-bg);
    border-color: var(--bs-primary);
    color: var(--bs-body-color);
    box-shadow: 0 0 0 0.25rem rgba(var(--bs-primary-rgb), 0.25); /* Use Bootstrap primary color for focus ring */
}

/* Ensure textareas/inputs for code/config look good */
textarea.form-control {
    font-family: monospace;
    min-height: 200px; /* Give config textarea some height */
}

/* Style for log output area */
#logOutput {
    font-family: monospace;
    white-space: pre-wrap; /* Wrap long lines */
    word-wrap: break-word;
    background-color: var(--bs-body-secondary-bg);
    border: 1px solid var(--bs-border-color);
    padding: 15px;
    border-radius: var(--bs-border-radius);
    max-height: 400px; /* Limit log height */
    overflow-y: auto; /* Add scroll if needed */
    color: var(--bs-secondary-color); /* Log text color */
}

/* Specific log level colors (optional, requires parsing log lines) */
/* Example (requires JS to add classes or server-side rendering): */
/* .log-debug { color: var(--bs-secondary); } */
/* .log-info { color: var(--bs-light); } */
/* .log-warning { color: var(--bs-warning); } */
/* .log-error { color: var(--bs-danger); } */
/* .log-critical { color: var(--bs-danger); font-weight: bold; } */

/* Adjust padding for content when sidebar is present */
@media (min-width: 768px) {
  main {
    margin-left: 210px; /* Match sidebar width */
  }
  .sidebar {
      width: 210px; /* Define sidebar width */
  }
}

/* Add some spacing below main content */
main {
    padding-bottom: 3rem;
}

/* Footer styles */
footer {
    padding-top: 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid var(--bs-border-color);
    color: var(--bs-secondary-color);
}
footer a {
    color: var(--bs-secondary-color);
}
footer a:hover {
     color: var(--bs-light);
}


/* Scrollspy active state adjustments */
body > main > div > h2 {
    scroll-margin-top: 60px; /* Adjust for fixed header/navbar if you add one */
}

EOL

# Create dashboard.js
mkdir -p static/js
echo "Creating dashboard.js..."
cat > static/js/dashboard.js << 'EOL'
// Dashboard.js - Client-side JavaScript for the Trading Bot Dashboard

// DOM elements
let equityChart = null; // Renamed from performanceChart for clarity

// Initialize dashboard components when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
    // Use setInterval to refresh data periodically (e.g., every 15 seconds)
    // This should match or be faster than the bot's loop interval
    const refreshInterval = 15000; // 15 seconds
    refreshData(); // Initial load
    setInterval(refreshData, refreshInterval); // Refresh periodically
});

// Set up event listeners
function setupEventListeners() {
    // Refresh button (manual refresh)
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
             refreshData();
             alert('Data refreshed manually.'); // Provide feedback
        });
    }

    // Save configuration button
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    if (saveConfigBtn) {
        saveConfigBtn.addEventListener('click', saveConfiguration);
    }

    // Toggle config section visibility (optional)
    // const configHeader = document.getElementById('configHeader');
    // if (configHeader) {
    //      configHeader.addEventListener('click', function() {
    //           const configBody = document.getElementById('configCardBody');
    //           if (configBody) {
    //                configBody.classList.toggle('d-none'); // Use Bootstrap d-none class
    //           }
    //      });
    // }
}

// Initialize charts
function initializeCharts() {
    // Equity curve chart
    const ctx = document.getElementById('equityChart'); // Updated ID
    if (ctx) {
        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], // Timestamps
                datasets: [{
                    label: 'Equity',
                    data: [], // Equity values
                    borderColor: 'rgba(75, 192, 192, 1)', // Teal color
                    backgroundColor: 'rgba(75, 192, 192, 0.2)', // Semi-transparent fill
                    borderWidth: 2,
                    tension: 0.1, // Smooth line
                    fill: true,
                    pointRadius: 1, // Smaller points
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time', // Use time scale for timestamps
                        time: {
                            unit: 'hour', // Adjust unit based on timeframe/data density
                            tooltipFormat: 'YYYY-MM-DD HH:mm',
                            displayFormats: {
                                hour: 'MMM D, HH:mm',
                                day: 'MMM D',
                                week: 'MMM D'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        },
                        grid: {
                            color: 'rgba(200, 200, 200, 0.1)' // Light grid lines for dark theme
                        },
                        ticks: {
                            color: 'var(--bs-secondary-color)' // Tick color for dark theme
                        }
                    },
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Equity'
                        },
                         grid: {
                            color: 'rgba(200, 200, 200, 0.1)' // Light grid lines for dark theme
                        },
                         ticks: {
                            color: 'var(--bs-secondary-color)' // Tick color for dark theme
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                         labels: {
                             color: 'var(--bs-secondary-color)' // Legend text color
                         }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                }
            });
    }

    // Add other charts here (e.g., Drawdown chart)
    // Example for Drawdown chart (requires drawdown_history data from API)
    // const drawdownCtx = document.getElementById('drawdownChart');
    // if (drawdownCtx) {
    //      // Initialize Drawdown Chart (similar structure to equityChart)
    //      // Data would be drawdown_history from the performance API
    // }
}

// Refresh dashboard data from API endpoints
function refreshData() {
    console.log("Fetching fresh data...");

    // Get performance data (includes equity curve data)
    fetch('/api/performance')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Performance data received:', data);
            updatePerformanceStats(data);
            updateEquityChart(data.equity_curve); // Update chart with equity data
            updateTradeHistoryTable(data.trades_history); // Update trades table
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
            // Display error on dashboard?
        });

    // Get state data (for current positions and last update time)
    fetch('/api/state')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('State data received:', data);
            updateStateInfo(data);
        })
        .catch(error => {
            console.error('Error fetching state data:', error);
            // Display error on dashboard?
        });

    // Get log data
    fetch('/api/log?lines=200') // Fetch last 200 lines
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
             console.log('Log data received:', data);
             updateLogOutput(data);
        })
        .catch(error => {
             console.error('Error fetching log data:', error);
             // Display error?
        });

    // Optionally fetch config again to ensure UI reflects latest saved config
    // fetch('/api/config')...
}

// Update performance statistics displayed on the page
function updatePerformanceStats(data) {
    // Update summary cards/elements with performance data
    document.getElementById('totalTrades').innerText = data.total_trades;
    document.getElementById('winRate').innerText = formatPercentage(data.win_rate);
    document.getElementById('totalPnl').innerText = formatCurrency(data.total_pnl);
    document.getElementById('maxDrawdownPct').innerText = formatPercentage(data.max_drawdown_pct);

    // Apply color coding to PnL and Drawdown
    const totalPnlElement = document.getElementById('totalPnl');
    if (totalPnlElement) {
         totalPnlElement.classList.remove('pnl-positive', 'pnl-negative');
         if (data.total_pnl > 0) totalPnlElement.classList.add('pnl-positive');
         else if (data.total_pnl < 0) totalPnlElement.classList.add('pnl-negative');
    }
     const maxDrawdownElement = document.getElementById('maxDrawdownPct');
    if (maxDrawdownElement) {
         maxDrawdownElement.classList.remove('pnl-positive', 'pnl-negative');
         if (data.max_drawdown_pct > 0) maxDrawdownElement.classList.add('pnl-negative'); // Drawdown is negative performance
         else if (data.max_drawdown_pct < 0) maxDrawdownElement.classList.add('pnl-positive'); // Negative drawdown is good (not possible here, but for consistency)
    }

}

// Update state information (positions, last update)
function updateStateInfo(data) {
     // Update last update time
     const lastUpdateElement = document.getElementById('lastUpdate');
     if (lastUpdateElement) {
          const timestamp = data.last_update; // This is in milliseconds
          if (timestamp > 0) {
               const date = new Date(timestamp);
               lastUpdateElement.innerText = date.toLocaleString(); // Use local time for display
          } else {
               lastUpdateElement.innerText = 'Never';
          }
     }

     // Update active positions table
     const positionsTableBody = document.getElementById('positionsTableBody');
     if (positionsTableBody) {
          positionsTableBody.innerHTML = ''; // Clear current rows
          const positions = data.positions || {};
          const symbol = document.getElementById('symbol').value; // Get current configured symbol for filtering

          // Filter and display only the position for the configured symbol
          const currentSymbolPosition = positions[symbol];

          if (currentSymbolPosition) {
               const pos = currentSymbolPosition;
               const row = document.createElement('tr');
               row.innerHTML = `
                    <td>${symbol}</td>
                    <td>
                         <span class="badge bg-${pos.side === 'long' ? 'success' : 'danger'}">
                              ${pos.side.toUpperCase()}
                         </span>
                    </td>
                    <td>${pos.amount.toFixed(6)}</td>
                    <td>${pos.entry_price.toFixed(4)}</td>
                    <td class="${pos.unrealizedPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${pos.unrealizedPnl.toFixed(2)}</td>
                    <td>${pos.liquidation_price > 0 ? pos.liquidation_price.toFixed(4) : 'N/A'}</td>
                    <td>${pos.sl_price !== null && pos.sl_price > 0 ? pos.sl_price.toFixed(4) : 'N/A'}</td>
                    <td>${pos.tp_price !== null && pos.tp_price > 0 ? pos.tp_price.toFixed(4) : 'N/A'}</td>
               `;
               positionsTableBody.appendChild(row);
          } else {
               const row = document.createElement('tr');
               row.innerHTML = `<td colspan="8" class="text-center">No active position for ${symbol}</td>`;
               positionsTableBody.appendChild(row);
          }

          // Update active positions count card
          document.getElementById('activePositionsCount').innerText = currentSymbolPosition ? 1 : 0;

     }
}


// Update the Equity Chart with new data
function updateEquityChart(equityData) {
    if (!equityChart || !equityData || equityData.length === 0) {
        return;
    }

    // Sort data by timestamp
    equityData.sort((a, b) => a.timestamp - b.timestamp);

    // Extract timestamps and equity values
    const timestamps = equityData.map(item => new Date(item.timestamp)); // Convert ms timestamp to Date object
    const equities = equityData.map(item => item.equity);

    // Update chart data
    equityChart.data.labels = timestamps;
    equityChart.data.datasets[0].data = equities;

    // Update chart
    equityChart.update();
}

// Update the Trade History Table with new data
function updateTradeHistoryTable(tradesHistory) {
    const tradesTableBody = document.getElementById('tradesTableBody');
    if (!tradesTableBody || !tradesHistory) {
        return;
    }

    // Sort trades by exit timestamp (or entry if exit is missing)
    tradesHistory.sort((a, b) => (b.exit_timestamp || b.timestamp) - (a.exit_timestamp || a.timestamp));

    tradesTableBody.innerHTML = ''; // Clear current rows

    if (tradesHistory.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="9" class="text-center">No trade history</td>`;
        tradesTableBody.appendChild(row);
        return;
    }

    tradesHistory.forEach(trade => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${trade.entry_time}</td>
            <td>${trade.exit_time}</td>
            <td>${trade.symbol}</td>
            <td>
                 <span class="badge bg-${trade.side === 'long' ? 'success' : 'danger'}">
                      ${trade.side.toUpperCase()}
                 </span>
            </td>
            <td>${trade.amount.toFixed(6)}</td>
            <td>${trade.entry_price.toFixed(4)}</td>
            <td>${trade.exit_price > 0 ? trade.exit_price.toFixed(4) : 'N/A'}</td>
            <td class="${trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${trade.pnl.toFixed(2)}</td>
            <td class="${trade.pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">${trade.pnl_pct.toFixed(2)}%</td>
        `;
        tradesTableBody.appendChild(row);
    });
}

// Update the Log Output area
function updateLogOutput(logLines) {
     const logOutputElement = document.getElementById('logOutput');
     if (logOutputElement && logLines) {
          // Join lines and display. Basic styling can be done via CSS.
          // For more advanced log display (color coding levels), you'd need to parse lines.
          logOutputElement.innerText = logLines.join('');
          // Optional: Auto-scroll to bottom
          logOutputElement.scrollTop = logOutputElement.scrollHeight;
     }
}


// Save configuration changes
function saveConfiguration() {
    console.log("Saving configuration...");

    // Get values from the form. Adapt this based on your HTML form structure.
    // This is a simplified example, assuming input IDs match config keys.
    const config = {};

    // Basic Settings
    config.symbol = document.getElementById('symbol').value;
    config.exchange = document.getElementById('exchange').value;
    config.timeframe = document.getElementById('timeframe').value;
    config.test_mode = document.getElementById('test_mode').checked;
    config.loop_interval_seconds = parseInt(document.getElementById('loop_interval_seconds').value, 10);

    // Risk Management
    config.risk_management = {
        position_size_pct: parseFloat(document.getElementById('position_size_pct').value),
        max_risk_per_trade_pct: parseFloat(document.getElementById('max_risk_per_trade_pct').value),
        use_atr_position_sizing: document.getElementById('use_atr_position_sizing').checked,
        stop_loss: {
            enabled: document.getElementById('sl_enabled').checked,
            mode: document.getElementById('sl_mode').value,
            atr_multiplier: parseFloat(document.getElementById('sl_atr_multiplier').value),
            fixed_pct: parseFloat(document.getElementById('sl_fixed_pct').value)
        },
        take_profit: {
            enabled: document.getElementById('tp_enabled').checked,
            mode: document.getElementById('tp_mode').value,
            atr_multiplier: parseFloat(document.getElementById('tp_atr_multiplier').value),
            fixed_pct: parseFloat(document.getElementById('tp_fixed_pct').value)
        },
         break_even: {
             enabled: document.getElementById('be_enabled').checked,
             activation_pct: parseFloat(document.getElementById('be_activation_pct').value)
         }
    };

    // Strategy - Indicators (Enabled/Disabled)
    config.strategy = {
         entry_threshold: parseFloat(document.getElementById('entry_threshold').value),
         exit_threshold: parseFloat(document.getElementById('exit_threshold').value),
         volume_filter: {
             enabled: document.getElementById('volume_filter_enabled').checked,
             min_24h_volume_usd: parseFloat(document.getElementById('min_24h_volume_usd').value)
         },
         indicators: {
             rsi: { enabled: document.getElementById('rsi_enabled').checked, weight: parseFloat(document.getElementById('rsi_weight').value) },
             macd: { enabled: document.getElementById('macd_enabled').checked, weight: parseFloat(document.getElementById('macd_weight').value) },
             bollinger_bands: { enabled: document.getElementById('bollinger_bands_enabled').checked, weight: parseFloat(document.getElementById('bollinger_bands_weight').value) },
             ema_cross: { enabled: document.getElementById('ema_cross_enabled').checked, weight: parseFloat(document.getElementById('ema_cross_weight').value), use_for_exit: document.getElementById('ema_cross_use_for_exit').checked },
             atr: { enabled: document.getElementById('atr_enabled').checked } // ATR config might not need weight for signal calc
         }
         // Add specific indicator parameters if you add inputs for them
         // Example: config.strategy.indicators.rsi.window = parseInt(document.getElementById('rsi_window').value, 10);
    };

    // Advanced Settings
     config.advanced = {
         candles_required: parseInt(document.getElementById('candles_required').value, 10),
         // precision and min_amount might be better fetched from exchange info,
         // but can be set here as defaults if needed.
         // price_precision: parseInt(document.getElementById('price_precision').value, 10),
         // amount_precision: parseInt(document.getElementById('amount_precision').value, 10),
         // min_amount: parseFloat(document.getElementById('min_amount').value),
     };
     // Handle optional max_amount if you add an input for it


    // Send to API
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('Configuration saved successfully');
            // Optional: refresh config form or page after save
            // location.reload();
        } else {
            alert('Error saving configuration: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error saving configuration:', error);
        alert('Error saving configuration');
    });
}

// Utility function to format currency values
function formatCurrency(value) {
    // Use Intl.NumberFormat for better currency formatting
    try {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD', // Assume USD for display, adjust if needed
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    } catch (e) {
        console.error("Error formatting currency:", value, e);
        return value ? value.toFixed(2) : '0.00'; // Fallback
    }
}

// Utility function to format percentage values
function formatPercentage(value) {
     try {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value / 100); // Convert from percentage (e.g., 50 for 50%) to decimal (0.5)
    } catch (e) {
        console.error("Error formatting percentage:", value, e);
        return value ? value.toFixed(2) + '%' : '0.00%'; // Fallback
    }
}

// Optional: Add functions to populate form from current config on page load
// You would fetch the config via API and then fill the form inputs
// This is not implemented in this basic version, assuming manual input for config form.
// A better approach would fetch config on dashboard load and populate the form.
// Example:
// function loadConfigIntoForm() {
//     fetch('/api/config')
//         .then(response => response.json())
//         .then(config => {
//             // Populate form fields based on config object
//             document.getElementById('symbol').value = config.symbol || '';
//             // ... populate other fields
//         })
//         .catch(error => console.error('Error loading config for form:', error));
// }
// Add loadConfigIntoForm() call in DOMContentLoaded or refreshData

EOL

# Create template HTML files
mkdir -p templates
echo "Creating index.html..."
cat > templates/index.html << 'EOL'
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Trading Bot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <link rel="stylesheet" href="/static/css/custom.css">
</head>
<body>
    <div class="container py-4">
        <header class="pb-3 mb-4 border-bottom">
            <div class="d-flex align-items-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-activity me-2 text-primary">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                <h1 class="fs-4 mb-0">Crypto Trading Bot</h1>
            </div>
        </header>

        <div class="p-5 mb-4 bg-body-tertiary rounded-3">
            <div class="container-fluid py-5">
                <h1 class="display-5 fw-bold">Automated Cryptocurrency Trading</h1>
                <p class="col-md-8 fs-4">A Python-based trading bot leveraging technical indicators and risk management to automate trading decisions on cryptocurrency exchanges.</p>
                <a href="/dashboard" class="btn btn-primary btn-lg">Go to Dashboard</a>
            </div>
        </div>

        <div class="row align-items-md-stretch">
            <div class="col-md-6 mb-4">
                <div class="h-100 p-5 bg-body-tertiary border rounded-3">
                    <h2>Key Features</h2>
                    <ul>
                        <li>Configurable multi-indicator strategies</li>
                        <li>Dynamic risk-based position sizing</li>
                        <li>Automated Stop-Loss and Take-Profit</li>
                        <li>Break-Even stop loss automation</li>
                        <li>Support for multiple exchanges (CCXT)</li>
                        <li>Web Dashboard for monitoring & config</li>
                        <li>Backtesting functionality</li>
                    </ul>
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="h-100 p-5 bg-body-tertiary border rounded-3">
                    <h2>Getting Started</h2>
                    <p>1. Run the setup script (you've likely just done this).</p>
                    <p>2. Edit <code>config.json</code> or set environment variables (<code>API_KEY</code>, <code>API_SECRET</code>).</p>
                    <p>3. Start the web interface: <code>python main.py --web</code></p>
                    <p>4. Review/adjust configuration via the dashboard.</p>
                    <p>5. Start the bot core: <code>python main.py</code></p>
                    <p>Monitor activity in the dashboard and bot logs.</p>
                    <a href="/dashboard" class="btn btn-outline-secondary">Go to Dashboard</a>
                </div>
            </div>
        </div>

        <footer class="pt-3 mt-4 text-body-secondary border-top">
            <div class="row">
                <div class="col-md-6">
                    <p>&copy; 2023 Crypto Trading Bot</p>
                </div>
                <div class="col-md-6 text-md-end">
                    <p>
                        <a href="https://github.com/ccxt/ccxt" target="_blank" class="text-decoration-none">CCXT</a> |
                        <a href="https://pandas-ta.readthedocs.io/" target="_blank" class="text-decoration-none">pandas_ta</a> |
                        <a href="https://getbootstrap.com/" target="_blank" class="text-decoration-none">Bootstrap</a> |
                        <a href="https://www.chartjs.org/" target="_blank" class="text-decoration-none">Chart.js</a>
                    </p>
                </div>
            </div>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
</body>
</html>
EOL

echo "Creating dashboard.html..."
cat > templates/dashboard.html << 'EOL'
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <link rel="stylesheet" href="/static/css/custom.css">
</head>
<body data-bs-spy="scroll" data-bs-target="#sidebarMenu" data-bs-root-margin="0px 0px -40%" data-bs-smooth-scroll="true" class="scrollspy-example" tabindex="0">

    <header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
      <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3 fs-6" href="/">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-activity me-2 text-primary">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
          </svg>
          Trading Bot
      </a>
      <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
        <div class="navbar-nav w-100 text-end me-3">
            <div class="nav-item text-nowrap">
                <small class="text-secondary">Last Update: <span id="lastUpdate">{{ last_update_str }}</span></small>
                <button class="btn btn-sm btn-outline-secondary ms-3" id="refreshBtn">Refresh Data</button>
            </div>
        </div>
    </header>


    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-body-tertiary sidebar collapse">
                <div class="position-sticky pt-3 sidebar-sticky">
                    <ul class="nav flex-column">
                        <li class="nav-item mb-2">
                            <a class="nav-link active" href="#overview">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-home">
                                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                                </svg>
                                Overview
                            </a>
                        </li>
                         <li class="nav-item mb-2">
                            <a class="nav-link" href="#performance">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-trending-up">
                                    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline>
                                </svg>
                                Performance
                            </a>
                        </li>
                        <li class="nav-item mb-2">
                            <a class="nav-link" href="#positions">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-dollar-sign">
                                    <line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                                </svg>
                                Positions
                            </a>
                        </li>
                        <li class="nav-item mb-2">
                            <a class="nav-link" href="#trades">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-list">
                                    <line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line>
                                </svg>
                                Trades
                            </a>
                        </li>
                        <li class="nav-item mb-2">
                            <a class="nav-link" href="#config">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings">
                                    <circle cx="12" cy="12" r="3"></circle>
                                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                                </svg>
                                Configuration
                            </a>
                        </li>
                         <li class="nav-item mb-2">
                            <a class="nav-link" href="#logs">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 8 8 9"></polyline>
                                </svg>
                                Logs
                            </a>
                        </li>
                    </ul>
                </div>
            </nav>

            <!-- Main content -->
            <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
                <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
                    <h1 class="h2">Trading Bot Dashboard</h1>
                </div>

                <!-- Overview Cards -->
                <h2 id="overview" class="mt-4 mb-3">Overview</h2>
                <div class="row mb-4">
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title text-secondary">Symbol</h5>
                                <p class="card-text fs-4">{{ config.get('symbol', 'N/A') }}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title text-secondary">Exchange</h5>
                                <p class="card-text fs-4">{{ config.get('exchange', 'N/A') }}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title text-secondary">Active Positions</h5>
                                <p class="card-text fs-4" id="activePositionsCount">{{ state.get('positions')|length }}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                         <div class="card">
                            <div class="card-body">
                                <h5 class="card-title text-secondary">Bot Mode</h5>
                                <p class="card-text fs-4">{{ 'Test Mode' if config.get('test_mode', True) else 'Live Mode' }}</p>
                            </div>
                        </div>
                    </div>
                </div>

                 <!-- Performance Stats -->
                <h2 id="performance" class="mt-4 mb-3">Performance</h2>
                <div class="row mb-4">
                     <div class="col-md-3 mb-3">
                        <div class="performance-stat performance-neutral">
                            <h5 class="text-secondary">Total Trades</h5>
                            <p class="fs-4" id="totalTrades">0</p>
                        </div>
                     </div>
                     <div class="col-md-3 mb-3">
                        <div class="performance-stat performance-neutral">
                            <h5 class="text-secondary">Win Rate</h5>
                            <p class="fs-4" id="winRate">0.00%</p>
                        </div>
                     </div>
                     <div class="col-md-3 mb-3">
                        <div class="performance-stat performance-neutral">
                            <h5 class="text-secondary">Total PnL</h5>
                            <p class="fs-4" id="totalPnl">$0.00</p>
                        </div>
                     </div>
                     <div class="col-md-3 mb-3">
                        <div class="performance-stat performance-neutral">
                            <h5 class="text-secondary">Max Drawdown</h5>
                            <p class="fs-4" id="maxDrawdownPct">0.00%</p>
                        </div>
                     </div>
                </div>

                <!-- Equity Chart -->
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                Equity Curve
                            </div>
                            <div class="card-body">
                                <canvas id="equityChart" class="chart-container"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Active Positions -->
                <h2 id="positions" class="mt-4 mb-3">Active Positions ({{ config.get('symbol', 'N/A') }})</h2>
                <div class="table-responsive mb-4">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Amount</th>
                                <th>Entry Price</th>
                                <th>Unrealized PnL</th>
                                <th>Liquidation Price</th>
                                <th>Stop Loss Price</th>
                                <th>Take Profit Price</th>
                            </tr>
                        </thead>
                        <tbody id="positionsTableBody">
                            {# This will be populated by JavaScript #}
                             <tr>
                                 <td colspan="8" class="text-center">Loading positions...</td>
                             </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Trade History -->
                <h2 id="trades" class="mt-4 mb-3">Trade History</h2>
                <div class="table-responsive mb-4">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>Entry Time</th>
                                <th>Exit Time</th>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Amount</th>
                                <th>Entry Price</th>
                                <th>Exit Price</th>
                                <th>PnL</th>
                                <th>PnL %</th>
                            </tr>
                        </thead>
                        <tbody id="tradesTableBody">
                            {# This will be populated by JavaScript #}
                             <tr>
                                 <td colspan="9" class="text-center">Loading trade history...</td>
                             </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Configuration -->
                <h2 id="config" class="mt-4 mb-3">Configuration</h2>
                <div class="card mb-4">
                    <div class="card-header" id="configHeader">
                        Trading Parameters
                    </div>
                    <div class="card-body" id="configCardBody">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Basic Settings</h5>
                                <div class="mb-3">
                                    <label for="symbol" class="form-label">Symbol</label>
                                    <input type="text" class="form-control" id="symbol" value="{{ config.get('symbol', '') }}">
                                </div>
                                <div class="mb-3">
                                    <label for="exchange" class="form-label">Exchange</label>
                                    <select class="form-select" id="exchange">
                                        <option value="bybit" {% if config.get('exchange', '').lower() == 'bybit' %}selected{% endif %}>Bybit</option>
                                        <option value="binance" {% if config.get('exchange', '').lower() == 'binance' %}selected{% endif %}>Binance</option>
                                        <option value="kucoin" {% if config.get('exchange', '').lower() == 'kucoin' %}selected{% endif %}>KuCoin</option>
                                         {# Add more exchanges as needed #}
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="timeframe" class="form-label">Timeframe</label>
                                    <select class="form-select" id="timeframe">
                                        <option value="1m" {% if config.get('timeframe') == '1m' %}selected{% endif %}>1 minute</option>
                                        <option value="5m" {% if config.get('timeframe') == '5m' %}selected{% endif %}>5 minutes</option>
                                        <option value="15m" {% if config.get('timeframe') == '15m' %}selected{% endif %}>15 minutes</option>
                                        <option value="30m" {% if config.get('timeframe') == '30m' %}selected{% endif %}>30 minutes</option>
                                        <option value="1h" {% if config.get('timeframe') == '1h' %}selected{% endif %}>1 hour</option>
                                        <option value="2h" {% if config.get('timeframe') == '2h' %}selected{% endif %}>2 hours</option>
                                        <option value="4h" {% if config.get('timeframe') == '4h' %}selected{% endif %}>4 hours</option>
                                         <option value="1d" {% if config.get('timeframe') == '1d' %}selected{% endif %}>1 day</option>
                                         {# Add more timeframes as supported by CCXT and strategy #}
                                    </select>
                                </div>
                                 <div class="mb-3 form-check">
                                    <input type="checkbox" class="form-check-input" id="test_mode" {% if config.get('test_mode', True) %}checked{% endif %}>
                                    <label class="form-check-label" for="test_mode">Test Mode</label>
                                    <div class="form-text">Connects to exchange testnet if available.</div>
                                </div>
                                <div class="mb-3">
                                    <label for="loop_interval_seconds" class="form-label">Bot Loop Interval (seconds)</label>
                                    <input type="number" class="form-control" id="loop_interval_seconds" value="{{ config.get('loop_interval_seconds', 30) }}" min="5" step="1">
                                    <div class="form-text">How often the bot checks for signals and updates.</div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h5>Risk Management</h5>
                                <div class="mb-3">
                                    <label for="position_size_pct" class="form-label">Position Size (% of Available Balance)</label>
                                    <input type="number" class="form-control" id="position_size_pct" value="{{ config.get('risk_management', {}).get('position_size_pct', 1.0) }}" min="0.1" max="100" step="0.1">
                                </div>
                                <div class="mb-3">
                                    <label for="max_risk_per_trade_pct" class="form-label">Max Risk Per Trade (% of Total Equity)</label>
                                    <input type="number" class="form-control" id="max_risk_per_trade_pct" value="{{ config.get('risk_management', {}).get('max_risk_per_trade_pct', 1.0) }}" min="0.1" max="100" step="0.1">
                                </div>
                                <div class="mb-3 form-check">
                                    <input type="checkbox" class="form-check-input" id="use_atr_position_sizing" {% if config.get('risk_management', {}).get('use_atr_position_sizing', True) %}checked{% endif %}>
                                    <label class="form-check-label" for="use_atr_position_sizing">Use ATR for Risk-Adjusted Position Sizing</label>
                                    <div class="form-text">Adjusts position size based on stop-loss distance calculated using ATR.</div>
                                </div>

                                <h6>Stop Loss</h6>
                                <div class="mb-3 form-check">
                                     <input type="checkbox" class="form-check-input" id="sl_enabled" {% if config.get('risk_management', {}).get('stop_loss', {}).get('enabled', True) %}checked{% endif %}>
                                     <label class="form-check-label" for="sl_enabled">Enable Stop Loss</label>
                                </div>
                                <div class="mb-3">
                                    <label for="sl_mode" class="form-label">SL Mode</label>
                                    <select class="form-select" id="sl_mode">
                                        <option value="atr" {% if config.get('risk_management', {}).get('stop_loss', {}).get('mode') == 'atr' %}selected{% endif %}>ATR Multiplier</option>
                                        <option value="fixed_pct" {% if config.get('risk_management', {}).get('stop_loss', {}).get('mode') == 'fixed_pct' %}selected{% endif %}>Fixed Percentage</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="sl_atr_multiplier" class="form-label">ATR Multiplier (for ATR mode)</label>
                                    <input type="number" class="form-control" id="sl_atr_multiplier" value="{{ config.get('risk_management', {}).get('stop_loss', {}).get('atr_multiplier', 2.0) }}" min="0.1" step="0.1">
                                </div>
                                 <div class="mb-3">
                                    <label for="sl_fixed_pct" class="form-label">Fixed % (for Fixed % mode)</label>
                                    <input type="number" class="form-control" id="sl_fixed_pct" value="{{ config.get('risk_management', {}).get('stop_loss', {}).get('fixed_pct', 2.0) }}" min="0.1" step="0.1">
                                </div>

                                <h6>Take Profit</h6>
                                 <div class="mb-3 form-check">
                                     <input type="checkbox" class="form-check-input" id="tp_enabled" {% if config.get('risk_management', {}).get('take_profit', {}).get('enabled', True) %}checked{% endif %}>
                                     <label class="form-check-label" for="tp_enabled">Enable Take Profit</label>
                                </div>
                                <div class="mb-3">
                                    <label for="tp_mode" class="form-label">TP Mode</label>
                                    <select class="form-select" id="tp_mode">
                                        <option value="atr" {% if config.get('risk_management', {}).get('take_profit', {}).get('mode') == 'atr' %}selected{% endif %}>ATR Multiplier</option>
                                        <option value="fixed_pct" {% if config.get('risk_management', {}).get('take_profit', {}).get('mode') == 'fixed_pct' %}selected{% endif %}>Fixed Percentage</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="tp_atr_multiplier" class="form-label">ATR Multiplier (for ATR mode)</label>
                                    <input type="number" class="form-control" id="tp_atr_multiplier" value="{{ config.get('risk_management', {}).get('take_profit', {}).get('atr_multiplier', 4.0) }}" min="0.1" step="0.1">
                                </div>
                                 <div class="mb-3">
                                    <label for="tp_fixed_pct" class="form-label">Fixed % (for Fixed % mode)</label>
                                    <input type="number" class="form-control" id="tp_fixed_pct" value="{{ config.get('risk_management', {}).get('take_profit', {}).get('fixed_pct', 4.0) }}" min="0.1" step="0.1">
                                </div>

                                <h6>Break Even</h6>
                                 <div class="mb-3 form-check">
                                     <input type="checkbox" class="form-check-input" id="be_enabled" {% if config.get('risk_management', {}).get('break_even', {}).get('enabled', False) %}checked{% endif %}>
                                     <label class="form-check-label" for="be_enabled">Enable Break Even SL</label>
                                     <div class="form-text">Automatically move SL to entry price when profit target is reached.</div>
                                </div>
                                 <div class="mb-3">
                                    <label for="be_activation_pct" class="form-label">BE Activation Profit (%)</label>
                                    <input type="number" class="form-control" id="be_activation_pct" value="{{ config.get('risk_management', {}).get('break_even', {}).get('activation_pct', 1.0) }}" min="0.1" step="0.1">
                                     <div class="form-text">Percentage profit required to trigger break-even SL.</div>
                                </div>

                            </div> {# end col-md-6 risk #}
                        </div> {# end row #}
                        <hr>
                        <div class="row">
                             <div class="col-md-6">
                                <h5>Strategy Parameters</h5>
                                <div class="mb-3">
                                    <label for="entry_threshold" class="form-label">Entry Signal Threshold</label>
                                    <input type="number" class="form-control" id="entry_threshold" value="{{ config.get('strategy', {}).get('entry_threshold', 0.6) }}" min="0" max="1" step="0.05">
                                    <div class="form-text">Required combined signal strength (0-1) to enter a trade.</div>
                                </div>
                                <div class="mb-3">
                                    <label for="exit_threshold" class="form-label">Exit Signal Threshold</label>
                                    <input type="number" class="form-control" id="exit_threshold" value="{{ config.get('strategy', {}).get('exit_threshold', 0.4) }}" min="0" max="1" step="0.05">
                                    <div class="form-text">Required combined opposing signal strength (0-1) to exit a trade.</div>
                                </div>
                                 <h6>Volume Filter</h6>
                                 <div class="mb-3 form-check">
                                     <input type="checkbox" class="form-check-input" id="volume_filter_enabled" {% if config.get('strategy', {}).get('volume_filter', {}).get('enabled', False) %}checked{% endif %}>
                                     <label class="form-check-label" for="volume_filter_enabled">Enable Volume Filter</label>
                                     <div class="form-text">Only trade symbols above a minimum 24h quote volume.</div>
                                </div>
                                 <div class="mb-3">
                                    <label for="min_24h_volume_usd" class="form-label">Min 24h Volume (USD)</label>
                                    <input type="number" class="form-control" id="min_24h_volume_usd" value="{{ config.get('strategy', {}).get('volume_filter', {}).get('min_24h_volume_usd', 1000000) }}" min="0" step="10000">
                                </div>
                             </div>
                             <div class="col-md-6">
                                <h5>Indicator Settings</h5>
                                <div class="row">
                                     <div class="col-6"><h6>Indicator</h6></div>
                                     <div class="col-3 text-center"><h6>Enabled</h6></div>
                                     <div class="col-3 text-center"><h6>Weight</h6></div>
                                </div>
                                {# RSI #}
                                <div class="row mb-2 align-items-center">
                                     <div class="col-6"><label class="form-check-label" for="rsi_enabled">RSI (14)</label></div>
                                     <div class="col-3 text-center">
                                         <input type="checkbox" class="form-check-input" id="rsi_enabled" {% if config.get('strategy', {}).get('indicators', {}).get('rsi', {}).get('enabled', True) %}checked{% endif %}>
                                     </div>
                                     <div class="col-3 text-center">
                                          <input type="number" class="form-control form-control-sm" id="rsi_weight" value="{{ config.get('strategy', {}).get('indicators', {}).get('rsi', {}).get('weight', 1.0) }}" min="0" step="0.1">
                                     </div>
                                </div>
                                {# MACD #}
                                <div class="row mb-2 align-items-center">
                                     <div class="col-6"><label class="form-check-label" for="macd_enabled">MACD (12, 26, 9)</label></div>
                                     <div class="col-3 text-center">
                                         <input type="checkbox" class="form-check-input" id="macd_enabled" {% if config.get('strategy', {}).get('indicators', {}).get('macd', {}).get('enabled', True) %}checked{% endif %}>
                                     </div>
                                      <div class="col-3 text-center">
                                          <input type="number" class="form-control form-control-sm" id="macd_weight" value="{{ config.get('strategy', {}).get('indicators', {}).get('macd', {}).get('weight', 1.0) }}" min="0" step="0.1">
                                     </div>
                                </div>
                                {# Bollinger Bands #}
                                <div class="row mb-2 align-items-center">
                                     <div class="col-6"><label class="form-check-label" for="bollinger_bands_enabled">Bollinger Bands (20, 2)</label></div>
                                     <div class="col-3 text-center">
                                         <input type="checkbox" class="form-check-input" id="bollinger_bands_enabled" {% if config.get('strategy', {}).get('indicators', {}).get('bollinger_bands', {}).get('enabled', True) %}checked{% endif %}>
                                     </div>
                                      <div class="col-3 text-center">
                                          <input type="number" class="form-control form-control-sm" id="bollinger_bands_weight" value="{{ config.get('strategy', {}).get('indicators', {}).get('bollinger_bands', {}).get('weight', 1.0) }}" min="0" step="0.1">
                                     </div>
                                </div>
                                {# EMA Cross #}
                                <div class="row mb-2 align-items-center">
                                     <div class="col-6">
                                         <label class="form-check-label" for="ema_cross_enabled">EMA Cross (8, 21)</label>
                                         <div class="form-text mt-0">Also use for exit:
                                              <input type="checkbox" class="form-check-input" id="ema_cross_use_for_exit" {% if config.get('strategy', {}).get('indicators', {}).get('ema_cross', {}).get('use_for_exit', True) %}checked{% endif %}>
                                         </div>
                                     </div>
                                     <div class="col-3 text-center">
                                         <input type="checkbox" class="form-check-input" id="ema_cross_enabled" {% if config.get('strategy', {}).get('indicators', {}).get('ema_cross', {}).get('enabled', True) %}checked{% endif %}>
                                     </div>
                                     <div class="col-3 text-center">
                                          <input type="number" class="form-control form-control-sm" id="ema_cross_weight" value="{{ config.get('strategy', {}).get('indicators', {}).get('ema_cross', {}).get('weight', 1.0) }}" min="0" step="0.1">
                                     </div>
                                </div>
                                {# ATR (mainly for risk, can be enabled here) #}
                                <div class="row mb-2 align-items-center">
                                     <div class="col-6"><label class="form-check-label" for="atr_enabled">ATR (14)</label></div>
                                     <div class="col-3 text-center">
                                         <input type="checkbox" class="form-check-input" id="atr_enabled" {% if config.get('strategy', {}).get('indicators', {}).get('atr', {}).get('enabled', True) %}checked{% endif %}>
                                     </div>
                                     <div class="col-3 text-center">N/A</div> {# ATR doesn't contribute directly to weighted signal #}
                                </div>

                                 {# Add specific indicator parameter inputs here if desired #}
                                 <div class="form-text mt-3">Note: Only indicator enabled status and weight are configurable via UI. Other parameters (windows, std_dev, etc.) must be edited directly in <code>config.json</code>.</div>

                             </div> {# end col-md-6 indicators #}
                        </div> {# end row #}

                         <hr>
                         <h5>Advanced Settings</h5>
                         <div class="row">
                              <div class="col-md-6">
                                   <div class="mb-3">
                                        <label for="candles_required" class="form-label">Candles Required</label>
                                        <input type="number" class="form-control" id="candles_required" value="{{ config.get('advanced', {}).get('candles_required', 200) }}" min="50" step="10">
                                        <div class="form-text">Number of historical candles fetched for indicator calculation. Ensure enough for the longest indicator period.</div>
                                   </div>
                                   {# Precision/Min_Amount can be shown but maybe not editable if they come from exchange #}
                                   {#
                                   <div class="mb-3">
                                        <label for="price_precision" class="form-label">Price Precision</label>
                                        <input type="number" class="form-control" id="price_precision" value="{{ config.get('advanced', {}).get('price_precision', 8) }}" min="0" step="1">
                                   </div>
                                    <div class="mb-3">
                                        <label for="amount_precision" class="form-label">Amount Precision</label>
                                        <input type="number" class="form-control" id="amount_precision" value="{{ config.get('advanced', {}).get('amount_precision', 8) }}" min="0" step="1">
                                   </div>
                                    <div class="mb-3">
                                        <label for="min_amount" class="form-label">Minimum Trade Amount</label>
                                        <input type="number" class="form-control" id="min_amount" value="{{ config.get('advanced', {}).get('min_amount', 0.000001) }}" min="0" step="0.000001">
                                   </div>
                                   #}
                              </div>
                         </div>

                        <button type="button" class="btn btn-primary mt-3" id="saveConfigBtn">Save Configuration</button>
                        <div class="form-text mt-2">Restart the bot core (<code>python main.py</code>) for configuration changes to take full effect, especially for symbol, exchange, timeframe, and test mode.</div>
                    </div>
                </div>

                 <!-- Logs -->
                 <h2 id="logs" class="mt-4 mb-3">Logs</h2>
                 <div class="card mb-4">
                      <div class="card-header">
                           Recent Bot Logs
                      </div>
                      <div class="card-body">
                           <pre id="logOutput">Loading logs...</pre>
                      </div>
                 </div>


            </main>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script> {# Use Chart.js UMD bundle #}
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.umd.min.js"></script> {# Use date-fns adapter for time scale #}
    <script src="/static/js/dashboard.js"></script>
</body>
</html>
EOL

# Create default config.json (more detailed default)
echo "Creating config.json..."
cat > config.json << 'EOL'
{
    "exchange": "bybit",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "15m",
    "api_key": "",
    "api_secret": "",
    "test_mode": true,
    "risk_management": {
        "position_size_pct": 1.0,
        "max_risk_per_trade_pct": 1.0,
        "use_atr_position_sizing": true,
        "stop_loss": {
            "enabled": true,
            "mode": "atr",
            "atr_multiplier": 2.0,
            "fixed_pct": 2.0
        },
        "take_profit": {
            "enabled": true,
            "mode": "atr",
            "atr_multiplier": 4.0,
            "fixed_pct": 4.0
        },
        "break_even": {
            "enabled": true,
            "activation_pct": 1.0
        }
    },
    "strategy": {
        "entry_threshold": 0.6,
        "exit_threshold": 0.4,
         "volume_filter": {
             "enabled": false,
             "min_24h_volume_usd": 1000000
         },
        "indicators": {
            "rsi": {
                "enabled": true,
                "window": 14,
                "overbought": 70,
                "oversold": 30,
                "weight": 1.0
            },
            "macd": {
                "enabled": true,
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
                "weight": 1.0
            },
            "bollinger_bands": {
                "enabled": true,
                "window": 20,
                "std_dev": 2.0,
                "weight": 1.0
            },
            "ema_cross": {
                "enabled": true,
                "fast_period": 8,
                "slow_period": 21,
                "use_for_exit": true,
                "weight": 1.0
            },
            "atr": {
                "enabled": true,
                "window": 14
            }
        }
    },
    "advanced": {
        "candles_required": 200,
        "price_precision": 8,
        "amount_precision": 8,
        "min_amount": 0.000001,
        "max_amount": null
    },
    "loop_interval_seconds": 30
}
EOL

# Create empty bot_state.json (default structure)
echo "Creating default bot_state.json..."
cat > bot_state.json << 'EOL'
{
    "positions": {},
    "orders": {},
    "trades": [],
    "last_update": 0,
    "equity_history": [],
    "current_trade": null
}
EOL

# Create .env file placeholder
echo "Creating .env placeholder..."
cat > .env << 'EOL'
# Rename this file to .env and fill in your API keys
# API_KEY=YOUR_API_KEY
# API_SECRET=YOUR_API_SECRET
# FLASK_SECRET_KEY=CHANGE_THIS_TO_A_RANDOM_SECRET
EOL

# Make main.py executable
chmod +x main.py

echo
echo "=== Setup Complete ==="
echo
echo "Project directory created at: $PROJECT_DIR"
echo "Configuration file: $PROJECT_DIR/config.json"
echo "State file: $PROJECT_DIR/bot_state.json"
echo "Log file: $PROJECT_DIR/bot_logs/trading_bot.log"
echo "Web interface files are in $PROJECT_DIR/templates and $PROJECT_DIR/static"
echo

echo "Next Steps:"
echo "1. Navigate to the project directory:"
echo "   cd ~/trading_bot"
echo
echo "2. Edit the config.json file OR the .env file to add your exchange API keys."
echo "   For live trading, change \"test_mode\": true to false in config.json."
echo "   Example using nano:"
echo "   nano config.json"
echo "   nano .env"
echo
echo "3. To start the web interface (recommended first step to check config):"
echo "   python main.py --web"
echo "   Access the dashboard at http://127.0.0.1:5000 (or your Termux IP if accessed remotely)"
echo "   (Use Ctrl+C to stop the web server)"
echo
echo "4. To start the trading bot core logic (runs in the background):"
echo "   python main.py"
echo "   (Use Ctrl+C to stop the bot)"
echo
echo "5. To run a single trading cycle (e.g., for cron):"
echo "   python main.py --run-once"
echo
echo "6. To run a backtest using historical data (uses config settings):"
echo "   python main.py --backtest"
echo
echo "Note: You can run the web interface and the bot core simultaneously in different Termux sessions."
echo "Log output is written to bot_logs/trading_bot.log and shown on the console."
echo
echo "Happy trading!"
