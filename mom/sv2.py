#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.0.0 (Enhanced Safety, Configurable Loop SL/TP)

Features:
- Dual Supertrend strategy with confirmation.
- ATR for volatility measurement and Stop-Loss/Take-Profit calculation.
- Volume spike analysis (optional confirmation).
- Order book pressure analysis (optional confirmation, optimized fetching).
- Risk-based position sizing with margin checks.
- Termux SMS alerts for critical events.
- Robust error handling and logging with color support.
- Graceful shutdown on KeyboardInterrupt with position closing attempt.
- Stricter position detection logic (targeting Bybit V5 API, checks 'side').
- **Configurable Loop-Based SL/TP Check (Default: OFF due to safety concerns).**

Disclaimer:
- **EXTREME RISK**: This software is for educational purposes ONLY. Scalping bots are
  extremely high-risk due to market noise, latency, slippage, fees, and the
  critical need for precise, low-latency execution. Use at your own absolute risk.
- **CRITICAL SL/TP LIMITATION (LOOP-BASED CHECK)**: The optional loop-based
  Stop-Loss/Take-Profit mechanism checks *after* the candle closes. This is
  **HIGHLY INACCURATE AND UNSAFE** for live scalping. Price can move significantly
  beyond calculated levels before the bot reacts in the next cycle. For real
  scalping, you **MUST** implement exchange-native conditional orders
  (e.g., stopMarket, takeProfitMarket using `exchange.create_order` with appropriate
  `stopLossPrice` / `takeProfitPrice` params) placed *immediately* after entry
  confirmation, ideally based on the actual fill price. **This loop check is DISABLED by default.**
- Parameter Sensitivity: Bot performance is highly dependent on parameter tuning.
  Requires significant backtesting and forward testing. Defaults are illustrative.
- API Rate Limits: Frequent API calls can hit exchange rate limits. Monitor usage.
- Slippage: Market orders used for entry/exit are prone to slippage, especially
  during volatile periods.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: SMS alerts require running within Termux on Android with
  Termux:API installed (`pkg install termux-api`) and configured (SMS permissions).
- API Changes: Exchange APIs (like Bybit V3 vs V5) can change. This code targets
  structures observed in V5 via CCXT, but updates may be needed.
"""

# Standard Library Imports
import logging
import os
import sys
import time
import traceback
import subprocess  # For Termux API calls
from typing import Dict, Optional, Any, Tuple, List

# Third-party Libraries
import ccxt
import pandas as pd
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()  # Load variables from .env file into environment variables

# --- Constants ---

# --- API Credentials (Required in .env) ---
BYBIT_API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")

# --- Trading Parameters (Set Defaults, Override in .env) ---
DEFAULT_SYMBOL: str = os.getenv("SYMBOL", "BTC/USDT:USDT")  # CCXT unified symbol format
DEFAULT_INTERVAL: str = os.getenv("INTERVAL", "1m")  # Timeframe (e.g., '1m', '5m')
DEFAULT_LEVERAGE: int = int(os.getenv("LEVERAGE", 10))  # Desired leverage
DEFAULT_SLEEP_SECONDS: int = int(os.getenv("SLEEP_SECONDS", 10))  # Pause between cycles

# --- Risk Management (CRITICAL - TUNE CAREFULLY!) ---
RISK_PER_TRADE_PERCENTAGE: float = float(os.getenv("RISK_PER_TRADE_PERCENTAGE", 0.005))  # 0.5% risk per trade based on equity
ATR_STOP_LOSS_MULTIPLIER: float = float(os.getenv("ATR_STOP_LOSS_MULTIPLIER", 1.5))  # Multiplier for ATR to set SL distance
ATR_TAKE_PROFIT_MULTIPLIER: float = float(os.getenv("ATR_TAKE_PROFIT_MULTIPLIER", 2.0))  # Multiplier for ATR to set TP distance
MAX_ORDER_USDT_AMOUNT: float = float(os.getenv("MAX_ORDER_USDT_AMOUNT", 500.0))  # Maximum position size in USDT value
REQUIRED_MARGIN_BUFFER: float = float(os.getenv("REQUIRED_MARGIN_BUFFER", 1.05))  # Margin check buffer (e.g., 1.05 = requires 5% extra free margin)

# --- Safety Feature (Loop-Based SL/TP - HIGHLY DISCOURAGED) ---
# *** WARNING: Setting this to True enables a DANGEROUS and INACCURATE SL/TP mechanism ***
# *** suitable ONLY for testing or non-latency-sensitive strategies. ***
# *** For real scalping, KEEP THIS FALSE and implement exchange-native conditional orders. ***
ENABLE_LOOP_SLTP_CHECK: bool = os.getenv("ENABLE_LOOP_SLTP_CHECK", "false").lower() == "true" # Default to False

# --- Supertrend Indicator Parameters ---
DEFAULT_ST_ATR_LENGTH: int = int(os.getenv("ST_ATR_LENGTH", 7))  # Primary Supertrend ATR period
DEFAULT_ST_MULTIPLIER: float = float(os.getenv("ST_MULTIPLIER", 2.5))  # Primary Supertrend ATR multiplier
CONFIRM_ST_ATR_LENGTH: int = int(os.getenv("CONFIRM_ST_ATR_LENGTH", 5))  # Confirmation Supertrend ATR period
CONFIRM_ST_MULTIPLIER: float = float(os.getenv("CONFIRM_ST_MULTIPLIER", 2.0))  # Confirmation Supertrend ATR multiplier

# --- Volume Analysis Parameters ---
VOLUME_MA_PERIOD: int = int(os.getenv("VOLUME_MA_PERIOD", 20))  # Moving average period for volume
VOLUME_SPIKE_THRESHOLD: float = float(os.getenv("VOLUME_SPIKE_THRESHOLD", 1.5))  # Volume must be > this * MA for a spike
REQUIRE_VOLUME_SPIKE_FOR_ENTRY: bool = os.getenv("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "true").lower() == "true"  # Require volume spike for entry signal

# --- Order Book Analysis Parameters ---
ORDER_BOOK_DEPTH: int = int(os.getenv("ORDER_BOOK_DEPTH", 10))  # Levels of bids/asks to analyze
ORDER_BOOK_RATIO_THRESHOLD_LONG: float = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_LONG", 1.2))  # Bid/Ask volume ratio must be >= this for long confirmation
ORDER_BOOK_RATIO_THRESHOLD_SHORT: float = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_SHORT", 0.8))  # Bid/Ask volume ratio must be <= this for short confirmation
FETCH_ORDER_BOOK_PER_CYCLE: bool = os.getenv("FETCH_ORDER_BOOK_PER_CYCLE", "false").lower() == "true"  # Fetch OB every cycle (costly) or only on potential signals

# --- ATR Calculation Parameter (for SL/TP) ---
ATR_CALCULATION_PERIOD: int = int(os.getenv("ATR_CALCULATION_PERIOD", 10))  # ATR period used specifically for SL/TP calculation

# --- Termux SMS Alert Configuration (Set in .env) ---
ENABLE_SMS_ALERTS: bool = os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"  # Master switch for SMS alerts
SMS_RECIPIENT_NUMBER: Optional[str] = os.getenv("SMS_RECIPIENT_NUMBER")  # Phone number to send alerts to
SMS_TIMEOUT_SECONDS: int = int(os.getenv("SMS_TIMEOUT_SECONDS", 30))  # Timeout for the termux-sms-send command

# --- CCXT / API Parameters ---
DEFAULT_RECV_WINDOW: int = 15000  # API receive window (milliseconds)
ORDER_BOOK_FETCH_LIMIT: int = max(25, ORDER_BOOK_DEPTH)  # How many OB levels to fetch (API often requires min 25/50/100)
SHALLOW_OB_FETCH_DEPTH: int = 5  # Depth for quick best bid/ask fetch for entry price estimate

# --- Internal Constants ---
SIDE_BUY: str = "buy"
SIDE_SELL: str = "sell"
POSITION_SIDE_LONG: str = "Long"  # Internal representation for long position
POSITION_SIDE_SHORT: str = "Short"  # Internal representation for short position
POSITION_SIDE_NONE: str = "None"  # Internal representation for no position
USDT_SYMBOL: str = "USDT"  # Base currency symbol for balance checks
RETRY_COUNT: int = 3  # Number of retries for certain API calls (e.g., leverage setting)
RETRY_DELAY_SECONDS: int = 1  # Delay between retries
API_FETCH_LIMIT_BUFFER: int = 5  # Fetch slightly more OHLCV data than strictly needed for indicator calculation buffer
POSITION_QTY_EPSILON: float = 1e-9  # Small value for floating point comparisons (e.g., checking if quantity is effectively zero)
POST_CLOSE_DELAY_SECONDS: int = 2  # Brief delay after closing a position before potentially opening a new one


# --- Logger Setup ---
# ---> SET TO DEBUG FOR DETAILED TROUBLESHOOTING <---
# ---> REMEMBER TO SET BACK TO INFO FOR NORMAL OPERATION <---
LOGGING_LEVEL: int = logging.INFO  # Default level
# LOGGING_LEVEL = logging.DEBUG # Uncomment for verbose debugging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # logging.FileHandler("scalp_bot.log"), # Optional: Log to file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)
logger: logging.Logger = logging.getLogger(__name__)

# Custom SUCCESS level and color formatting
SUCCESS_LEVEL: int = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self, message, *args, **kwargs):
    """Adds a 'success' log level method to the Logger class."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

# Bind the new method to the Logger class
logging.Logger.success = log_success  # type: ignore

# ANSI escape codes for colored logging output (adjust based on terminal support)
# Check if the output stream is a TTY (terminal) before adding colors
if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"\033[90m{logging.getLevelName(logging.DEBUG)}\033[0m")  # Gray
    logging.addLevelName(logging.INFO, f"\033[96m{logging.getLevelName(logging.INFO)}\033[0m")  # Cyan
    logging.addLevelName(SUCCESS_LEVEL, f"\033[92m{logging.getLevelName(SUCCESS_LEVEL)}\033[0m")  # Green
    logging.addLevelName(logging.WARNING, f"\033[93m{logging.getLevelName(logging.WARNING)}\033[0m")  # Yellow
    logging.addLevelName(logging.ERROR, f"\033[91m{logging.getLevelName(logging.ERROR)}\033[0m")  # Red
    logging.addLevelName(logging.CRITICAL, f"\033[91m\033[1m{logging.getLevelName(logging.CRITICAL)}\033[0m")  # Bold Red


# --- Termux SMS Alert Function ---
def send_sms_alert(message: str) -> bool:
    """
    Sends an SMS alert using the 'termux-sms-send' command via subprocess.
    Requires Termux and Termux:API with SMS permissions granted.

    Args:
        message (str): The text message content to send.

    Returns:
        bool: True if the command executed successfully (return code 0),
              False otherwise (config disabled, missing number, command error, timeout).
              Note: True only means the command ran successfully, not guaranteed SMS delivery.
    """
    if not ENABLE_SMS_ALERTS:
        logger.debug("SMS alerts disabled via configuration.")
        return False
    if not SMS_RECIPIENT_NUMBER:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set in .env file.")
        return False

    try:
        # Command arguments: send SMS (-n recipient_number) with the message content
        # No complex shell injection risk here with direct message content, shlex not strictly needed
        command: List[str] = ['termux-sms-send', '-n', SMS_RECIPIENT_NUMBER, message]

        logger.info(f"Attempting to send SMS alert to {SMS_RECIPIENT_NUMBER} (Timeout: {SMS_TIMEOUT_SECONDS}s)...")
        logger.debug(f"Executing command: {' '.join(command)}")  # Log the command for debugging

        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode stdout/stderr as text
            check=False,  # Do not raise exception on non-zero exit, check manually
            timeout=SMS_TIMEOUT_SECONDS,  # Set a timeout for the command
        )

        if result.returncode == 0:
            logger.success(f"Termux API command for SMS to {SMS_RECIPIENT_NUMBER} executed successfully.")
            return True
        else:
            # Log detailed error information if the command failed
            logger.error(f"Termux API command 'termux-sms-send' failed. Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Termux API Stderr: {result.stderr.strip()}")
            if result.stdout:  # Sometimes error info might be on stdout
                logger.error(f"Termux API Stdout: {result.stdout.strip()}")
            logger.error("Potential issues: SMS permissions not granted in Android settings for Termux:API, "
                         "Termux:API app not running or installed, incorrect recipient number format, "
                         "Android battery optimization interfering with Termux.")
            return False

    except FileNotFoundError:
        logger.error("Failed to send SMS: 'termux-sms-send' command not found. "
                     "Ensure Termux:API package is installed (`pkg install termux-api`) and the Termux:API app is running.")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Failed to send SMS: 'termux-sms-send' command timed out after {SMS_TIMEOUT_SECONDS} seconds. "
                     "Check Termux:API service status, permissions, and Android battery optimization settings.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending SMS via Termux API: {e}")
        logger.debug(traceback.format_exc())  # Log full traceback for unexpected errors
        return False


# --- Exchange Initialization ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """
    Initializes and returns the CCXT Bybit exchange instance.
    Performs connectivity/authentication checks and sends SMS alerts.

    Returns:
        Optional[ccxt.Exchange]: Initialized exchange object or None on failure.
    """
    logger.info("Initializing CCXT Bybit connection...")
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        logger.critical("CRITICAL: BYBIT_API_KEY and/or BYBIT_API_SECRET missing in .env file.")
        send_sms_alert("[ScalpBot] CRITICAL: API keys missing in .env. Bot stopped.")
        return None

    try:
        exchange = ccxt.bybit(
            {
                "apiKey": BYBIT_API_KEY,
                "secret": BYBIT_API_SECRET,
                "enableRateLimit": True,  # Enable CCXT's built-in rate limiter
                "options": {
                    "defaultType": "linear",  # Crucial: Use 'linear' for USDT Perpetual, 'inverse' for inverse contracts
                    "recvWindow": DEFAULT_RECV_WINDOW,
                    # 'adjustForTimeDifference': True, # Consider if clock skew issues arise
                    # 'verbose': True, # Uncomment for extremely detailed API request/response logging
                },
            }
        )
        # Test connection and authentication by fetching markets and balance
        logger.debug("Loading markets...")
        exchange.load_markets()
        logger.debug("Fetching balance (tests authentication)...")
        exchange.fetch_balance()  # Throws AuthenticationError on bad keys
        logger.info("CCXT Bybit Session Initialized (LIVE FUTURES SCALPING MODE - EXTREME CAUTION!).")
        send_sms_alert("[ScalpBot] Initialized successfully and authenticated.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}. Check API key/secret and ensure IP whitelist (if used) is correct and API permissions are sufficient.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error during initialization: {e}. Check internet connection and Bybit status.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Bot stopped.")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error during initialization: {e}. Check Bybit status page or API documentation for details.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
    except Exception as e:
        logger.critical(f"Failed to initialize exchange due to an unexpected error: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")

    return None


# --- Indicator Calculation ---
def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: float, prefix: str = ""
) -> pd.DataFrame:
    """
    Calculates the Supertrend indicator using the pandas_ta library.

    Args:
        df (pd.DataFrame): Input DataFrame with 'high', 'low', 'close' columns.
        length (int): The ATR lookback period for Supertrend.
        multiplier (float): The ATR multiplier for Supertrend.
        prefix (str): Optional prefix for the generated columns (e.g., "confirm_").

    Returns:
        pd.DataFrame: DataFrame with added Supertrend columns:
                      '{prefix}supertrend' (the Supertrend line value),
                      '{prefix}trend' (boolean, True if trend is up),
                      '{prefix}st_long' (boolean, True on the candle the trend flipped UP),
                      '{prefix}st_short' (boolean, True on the candle the trend flipped DOWN).
                      Returns original DataFrame with NA columns if calculation fails.
    """
    col_prefix = f"{prefix}" if prefix else ""
    # Define the target column names we want to keep
    target_cols = [
        f"{col_prefix}supertrend",
        f"{col_prefix}trend",
        f"{col_prefix}st_long",
        f"{col_prefix}st_short",
    ]
    # Define the raw column names expected from pandas_ta based on its naming convention
    st_col_name = f"SUPERT_{length}_{multiplier}"  # Supertrend line value
    st_trend_col = f"SUPERTd_{length}_{multiplier}"  # Trend direction (-1 down, 1 up)
    st_long_col = f"SUPERTl_{length}_{multiplier}"  # Lower band value (used when trend is up)
    st_short_col = f"SUPERTs_{length}_{multiplier}" # Upper band value (used when trend is down)

    # Input validation
    required_input_cols = ["high", "low", "close"]
    if df is None or df.empty or not all(c in df.columns for c in required_input_cols):
        logger.warning(f"Indicator Calc ({col_prefix}Supertrend): Input DataFrame is missing required columns {required_input_cols} or is empty.")
        for col in target_cols: df[col] = pd.NA  # Add NA columns if they don't exist
        return df
    if len(df) < length:
        logger.warning(f"Indicator Calc ({col_prefix}Supertrend): DataFrame length ({len(df)}) is less than ST period ({length}).")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # Calculate Supertrend using pandas_ta, appending results to the DataFrame
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)

        # Verify that the expected raw columns were created
        if st_col_name not in df.columns or st_trend_col not in df.columns:
             raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col_name}, {st_trend_col}")

        # --- Process and Rename Columns ---
        # 1. Rename the main Supertrend value column
        df[f"{col_prefix}supertrend"] = df[st_col_name]

        # 2. Create a boolean trend column (True if trend is up)
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1

        # 3. Calculate flip signals: Signal is True *only* on the candle where the flip occurs
        prev_trend_direction = df[st_trend_col].shift(1)
        # Long signal: Previous was down (-1) and current is up (1)
        df[f"{col_prefix}st_long"] = (prev_trend_direction == -1) & (df[st_trend_col] == 1)
        # Short signal: Previous was up (1) and current is down (-1)
        df[f"{col_prefix}st_short"] = (prev_trend_direction == 1) & (df[st_trend_col] == -1)

        # --- Clean up intermediate pandas_ta columns ---
        # Identify all columns starting with 'SUPERT_' that are NOT our target columns
        cols_to_drop = [c for c in df.columns if c.startswith("SUPERT_") and c not in target_cols]
        # Also explicitly add the raw trend direction and band value columns if they exist
        cols_to_drop.extend([st_trend_col, st_long_col, st_short_col])
        # Use set to avoid duplicates if target_cols somehow overlapped with raw names, drop safely
        df.drop(columns=list(set(cols_to_drop)), errors='ignore', inplace=True)

        # Log the result of the last candle
        last_trend_str = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
        logger.debug(f"Indicator Calc: Calculated {col_prefix}Supertrend({length}, {multiplier}). "
                     f"Last trend: {last_trend_str}, Last value: {df[f'{col_prefix}supertrend'].iloc[-1]:.4f}")

    except Exception as e:
        logger.error(f"Indicator Calc: Error calculating {col_prefix}Supertrend({length}, {multiplier}): {e}")
        logger.debug(traceback.format_exc())
        # Ensure target columns exist with NA values on error
        for col in target_cols: df[col] = pd.NA
    return df


# --- Volume and ATR Analysis ---
def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> Dict[str, Optional[float]]:
    """
    Calculates ATR, Volume Moving Average, and checks for volume spikes relative to the MA.

    Args:
        df (pd.DataFrame): Input DataFrame with 'high', 'low', 'close', 'volume'.
        atr_len (int): Lookback period for ATR calculation.
        vol_ma_len (int): Lookback period for Volume Moving Average.

    Returns:
        Dict[str, Optional[float]]: Dictionary containing 'atr', 'volume_ma',
                                     'last_volume', 'volume_ratio'. Values are
                                     None if calculation fails or data is insufficient.
    """
    results: Dict[str, Optional[float]] = {
        "atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None
    }
    required_cols = ["high", "low", "close", "volume"]

    # Input validation
    if df is None or df.empty or not all(c in df.columns for c in required_cols):
        logger.warning(f"Indicator Calc (Vol/ATR): Input DataFrame is missing required columns {required_cols} or is empty.")
        return results
    min_len = max(atr_len, vol_ma_len)
    if len(df) < min_len:
         logger.warning(f"Indicator Calc (Vol/ATR): DataFrame length ({len(df)}) is less than required ({min_len}) for ATR({atr_len})/VolMA({vol_ma_len}).")
         return results

    try:
        # Calculate ATR using pandas_ta
        atr_col = f"ATRr_{atr_len}"  # pandas_ta default name for ATR
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]):
            results["atr"] = df[atr_col].iloc[-1]
        else:
            logger.warning(f"Indicator Calc: Failed to calculate valid ATR({atr_len}). Check input data.")
        # Clean up ATR column
        df.drop(columns=[atr_col], errors='ignore', inplace=True)

        # Calculate Volume Moving Average
        volume_ma_col = 'volume_ma'
        # Use min_periods to get MA sooner, but it might be less stable initially
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()

        if pd.notna(df[volume_ma_col].iloc[-1]):
            results["volume_ma"] = df[volume_ma_col].iloc[-1]
            results["last_volume"] = df['volume'].iloc[-1]  # Get the volume of the last candle

            # Calculate Volume Ratio (Last Volume / Volume MA)
            # Check if MA is not None and significantly greater than zero before dividing
            if results["volume_ma"] is not None and results["volume_ma"] > POSITION_QTY_EPSILON and results["last_volume"] is not None:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            else:
                results["volume_ratio"] = None  # Avoid division by zero or near-zero MA
                logger.debug(f"Indicator Calc: Volume ratio calculation skipped (LastVol={results['last_volume']}, VolMA={results['volume_ma']})")
        else:
             logger.warning(f"Indicator Calc: Failed to calculate valid Volume MA({vol_ma_len}). Check input data.")
        # Clean up Volume MA column
        df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)

        # Log results using safe formatting for potentially None values
        atr_val = results.get('atr')
        atr_str = f"{atr_val:.5f}" if atr_val is not None else 'N/A'
        last_vol_val = results.get('last_volume')
        vol_ma_val = results.get('volume_ma')
        vol_ratio_val = results.get('volume_ratio')
        last_vol_str = f"{last_vol_val:.2f}" if last_vol_val is not None else 'N/A'
        vol_ma_str = f"{vol_ma_val:.2f}" if vol_ma_val is not None else 'N/A'
        vol_ratio_str = f"{vol_ratio_val:.2f}" if vol_ratio_val is not None else 'N/A'

        logger.debug(f"Indicator Calc: ATR({atr_len}) = {atr_str}")
        logger.debug(f"Indicator Calc: Volume Analysis: Last={last_vol_str}, MA({vol_ma_len})={vol_ma_str}, Ratio={vol_ratio_str}")

    except Exception as e:
        logger.error(f"Indicator Calc: Error during Volume/ATR analysis: {e}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results} # Reset results to None on error
    return results


# --- Order Book Analysis ---
def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> Dict[str, Optional[float]]:
    """
    Fetches the L2 order book and analyzes bid/ask pressure (cumulative volume ratio)
    and spread within the specified depth.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): The market symbol (e.g., 'BTC/USDT:USDT').
        depth (int): How many price levels deep to calculate cumulative volume for the ratio.
        fetch_limit (int): How many levels to request from the API (must be >= depth,
                           and often subject to API minimums like 25, 50, or 100).

    Returns:
        Dict[str, Optional[float]]: Dictionary with 'bid_ask_ratio', 'spread',
                                     'best_bid', 'best_ask'. Values are None on failure.
    """
    results: Dict[str, Optional[float]] = {
        "bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None
    }
    logger.debug(f"Order Book: Fetching L2 for {symbol} (Analyze Depth: {depth}, API Fetch Limit: {fetch_limit})...")

    # Check if the exchange instance supports fetching L2 order book
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"Order Book: Exchange '{exchange.id}' does not support fetchL2OrderBook method. Skipping analysis.")
        return results
    try:
        # Fetch the order book data
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)

        # Validate the structure of the returned order book data
        if not order_book or not isinstance(order_book.get('bids'), list) or not isinstance(order_book.get('asks'), list):
            logger.warning(f"Order Book: Incomplete or invalid data structure received for {symbol}. OB: {order_book}")
            return results

        bids: List[List[float]] = order_book['bids']  # List of [price, amount]
        asks: List[List[float]] = order_book['asks']  # List of [price, amount]

        # Get best bid/ask and calculate spread
        best_bid = bids[0][0] if bids and len(bids[0]) > 0 else 0.0
        best_ask = asks[0][0] if asks and len(asks[0]) > 0 else 0.0
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        spread = None
        if best_bid > 0 and best_ask > 0:
            spread = best_ask - best_bid
            results["spread"] = spread
            logger.debug(f"Order Book: Best Bid={best_bid:.4f}, Best Ask={best_ask:.4f}, Spread={spread:.4f}")
        else:
            # Log even if spread calculation isn't possible (e.g., one side is empty)
            logger.debug(f"Order Book: Best Bid={best_bid:.4f}, Best Ask={best_ask:.4f} (Spread calculation requires both > 0)")

        # Calculate cumulative volume within the specified analysis depth
        # Sum the 'amount' (index 1) for the top 'depth' levels
        bid_volume_sum = sum(bid[1] for bid in bids[:depth] if len(bid) > 1 and isinstance(bid[1], (int, float)))
        ask_volume_sum = sum(ask[1] for ask in asks[:depth] if len(ask) > 1 and isinstance(ask[1], (int, float)))
        logger.debug(f"Order Book: Analysis (Depth {depth}): Total Bid Vol={bid_volume_sum:.4f}, Total Ask Vol={ask_volume_sum:.4f}")

        # Calculate Bid/Ask Ratio (Total Bid Volume / Total Ask Volume)
        bid_ask_ratio = None
        # Check if ask volume is significantly greater than zero to avoid division issues
        if ask_volume_sum > POSITION_QTY_EPSILON:
            bid_ask_ratio = bid_volume_sum / ask_volume_sum
            results["bid_ask_ratio"] = bid_ask_ratio
            logger.debug(f"Order Book: Bid/Ask Ratio (Depth {depth}) = {bid_ask_ratio:.3f}")
        else:
            logger.debug(f"Order Book: Bid/Ask Ratio calculation skipped (Ask volume at depth {depth} is zero or negligible)")

    except ccxt.NetworkError as e:
        logger.warning(f"Order Book: Network error fetching order book for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Order Book: Exchange error fetching order book for {symbol}: {e}")
    except IndexError:
        logger.warning(f"Order Book: Index error processing bids/asks for {symbol}. Data might be malformed or empty. Bids: {bids[:2]}, Asks: {asks[:2]}")
    except Exception as e:
        logger.error(f"Order Book: Unexpected error analyzing order book for {symbol}: {e}")
        logger.debug(traceback.format_exc())
        # Reset results to None on unexpected error
        results = {key: None for key in results}
    return results


# --- Data Fetching ---
def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int = 100
) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data from the exchange.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol (e.g., 'BTC/USDT:USDT').
        interval (str): Timeframe interval (e.g., '1m', '5m', '1h').
        limit (int): Number of candles to fetch.

    Returns:
        Optional[pd.DataFrame]: DataFrame with OHLCV data, indexed by timestamp (UTC),
                                 or None if fetching fails or data is invalid.
    """
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.")
        return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} (Timeframe: {interval})...")
        # Fetch data: Returns a list of lists [timestamp, open, high, low, close, volume]
        ohlcv: List[List[float]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

        if not ohlcv:
            logger.warning(f"Data Fetch: No OHLCV data returned from exchange for {symbol} ({interval}).")
            return None

        # Convert to pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp to datetime objects (UTC) and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Basic data validation: Check for NaN values
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"Data Fetch: Fetched OHLCV data contains NaN values. Counts:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...")
            # Simple imputation: Forward fill NaNs. More sophisticated methods could be used.
            df.ffill(inplace=True)
            # Check again after filling - if NaNs remain (e.g., at the very beginning), data is unusable
            if df.isnull().values.any():
                 logger.error("Data Fetch: NaN values remain after forward fill. Cannot proceed with this data batch.")
                 return None

        logger.debug(f"Data Fetch: Successfully fetched and processed {len(df)} OHLCV candles for {symbol}.")
        return df

    except ccxt.NetworkError as e:
        logger.warning(f"Data Fetch: Network error fetching OHLCV for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Data Fetch: Exchange error fetching OHLCV for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Data Fetch: Unexpected error fetching market data for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return None


# --- Position & Order Management (REVISED - v1.5.1) ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches the current position details for a given symbol, focusing on
    Bybit V5 API structure via CCXT for robustness against zero-size entries
    and using the 'side' field for direction. Assumes One-Way mode.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol in CCXT unified format (e.g., 'BTC/USDT:USDT').

    Returns:
        Dict[str, Any]: {'side': POSITION_SIDE_*, 'qty': float, 'entry_price': float}.
                        Defaults to side=NONE, qty=0.0, entry_price=0.0 if no *actual*
                        position (size > epsilon and valid side 'Buy'/'Sell') exists
                        for the specified symbol in One-Way mode.
    """
    # Default return value indicating no active position
    default_pos: Dict[str, Any] = {'side': POSITION_SIDE_NONE, 'qty': 0.0, 'entry_price': 0.0}
    ccxt_unified_symbol = symbol
    market_id = None

    # Get the exchange-specific market ID for comparison
    try:
        market = exchange.market(ccxt_unified_symbol)
        market_id = market['id']  # The ID used by the exchange (e.g., 'BTCUSDT')
        logger.debug(f"Position Check: Fetching position for CCXT symbol '{ccxt_unified_symbol}' (Target Exchange Market ID: '{market_id}')...")
    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(f"Position Check: Failed to get market info/ID for '{ccxt_unified_symbol}': {e}")
        return default_pos
    except Exception as e:  # Catch other potential errors during market lookup
        logger.error(f"Position Check: Unexpected error getting market info for '{ccxt_unified_symbol}': {e}")
        logger.debug(traceback.format_exc())
        return default_pos

    try:
        # Check if the exchange supports fetching positions
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"Position Check: Exchange '{exchange.id}' does not support fetchPositions. Cannot reliably get position status.")
            # Attempting fetchBalance might give clues, but it's less direct. Sticking to fetchPositions.
            return default_pos

        # Fetch positions - Bybit V5 often requires 'category' for linear/inverse
        params = {}
        if market.get('linear', False):
            params = {'category': 'linear'}
        elif market.get('inverse', False):
             params = {'category': 'inverse'}
        # If type isn't clear, fetching without category might work or fail depending on exchange default
        logger.debug(f"Position Check: Calling fetchPositions with params: {params}")
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)  # Fetch potentially for multiple symbols if needed, or just one
        # Note: Some CCXT versions/exchanges might return positions for *all* symbols if `symbols` arg is omitted or ignored.
        # We must filter carefully.

        logger.debug(f"Position Check: Raw data from fetch_positions (filtered for symbol '{symbol}'):\n{fetched_positions}")

        # Iterate through the potentially filtered list of positions returned by CCXT
        for pos in fetched_positions:
            logger.debug(f"Position Check: Evaluating raw position entry: {pos}")

            # --- 1. Symbol Check ---
            # CCXT aims to return the unified symbol, but check info.symbol for the raw ID as backup/confirmation
            pos_symbol_unified = pos.get('symbol')  # CCXT unified symbol (e.g., BTC/USDT:USDT)
            pos_symbol_raw = pos.get('info', {}).get('symbol')  # Raw exchange symbol ID (e.g., BTCUSDT)
            logger.debug(f"Position Check: Raw Symbol ID='{pos_symbol_raw}', Unified Symbol='{pos_symbol_unified}'")

            # Match against the target market ID obtained earlier
            if pos_symbol_raw != market_id:
                logger.debug(f"Skipping position entry: Symbol mismatch (Got Raw ID '{pos_symbol_raw}', Unified '{pos_symbol_unified}', Expected Target ID '{market_id}')")
                continue
            logger.debug(f"Symbol '{market_id}' matches target.")

            # --- 2. Position Mode Check (Assume One-Way) ---
            # Bybit V5 uses `positionIdx`: 0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge
            position_idx = pos.get('info', {}).get('positionIdx', 0)  # Default to 0 (One-Way) if not present
            if position_idx != 0:  # Strict check for One-Way mode
                logger.debug(f"Skipping position entry: Position Index is {position_idx} (Expected 0 for One-Way Mode). This bot only supports One-Way mode.")
                continue
            logger.debug(f"Position Index is {position_idx} (OK for One-Way Mode).")

            # --- 3. Strict Side Check (Using Bybit V5 'side' from 'info') ---
            # This is crucial for V5. 'side' indicates the position direction ('Buy', 'Sell') or 'None' if flat.
            pos_side_v5 = pos.get('info', {}).get('side', 'None')  # Expected: 'Buy', 'Sell', or 'None'
            logger.debug(f"Raw position side from info.side: '{pos_side_v5}'")

            determined_side = POSITION_SIDE_NONE
            if pos_side_v5 == 'Buy':
                determined_side = POSITION_SIDE_LONG
            elif pos_side_v5 == 'Sell':
                determined_side = POSITION_SIDE_SHORT

            # *** IMPORTANT: If the API explicitly reports side as 'None', treat as no position, regardless of size ***
            # This handles cases where residual dust might remain but the exchange considers the position closed.
            if determined_side == POSITION_SIDE_NONE:
                logger.debug(f"Skipping position entry: Side reported by API is '{pos_side_v5}'. Not considering it an active directional position.")
                continue
            logger.debug(f"Side '{pos_side_v5}' determined as {determined_side}.")

            # --- 4. Check Position Size (Using Bybit V5 'size' from 'info') ---
            # Use 'size' from raw info preferentially for Bybit V5 as it reflects the actual contract quantity.
            size_str = pos.get('info', {}).get('size')
            # Fallback to CCXT unified 'contracts' field if 'info.size' is missing (less likely for V5)
            if size_str is None or size_str == "":
                size_str = pos.get('contracts')  # CCXT unified field name for quantity

            if size_str is None or size_str == "":  # Check again after fallback
                logger.debug(f"Skipping position entry: Could not find a valid quantity string in 'info.size' or 'contracts'. Data: {pos.get('info')}")
                continue

            # --- 5. Parse and Validate Size ---
            try:
                size = float(size_str)
                logger.debug(f"Parsed quantity: {size} (from string: '{size_str}')")

                # --- Final Check: Is the size significantly non-zero? ---
                if abs(size) > POSITION_QTY_EPSILON:
                    # We have found an active position with a valid side ('Buy' or 'Sell') and a non-negligible size!
                    entry_price_str = pos.get('entryPrice')  # CCXT unified field
                    # Fallback to raw Bybit V5 'avgPrice' if unified 'entryPrice' is missing or invalid
                    try:
                        if entry_price_str is None or entry_price_str == "" or float(entry_price_str) <= 0:
                            entry_price_str = pos.get('info', {}).get('avgPrice')  # Raw field often more reliable
                    except (ValueError, TypeError):
                         entry_price_str = pos.get('info', {}).get('avgPrice') # Fallback on conversion error

                    entry_price = 0.0
                    if entry_price_str is not None and entry_price_str != "":
                         try:
                             entry_price = float(entry_price_str)
                         except (ValueError, TypeError):
                             logger.warning(f"Could not parse entry price string: '{entry_price_str}'. Defaulting to 0.0.")

                    qty_abs = abs(size)  # Return the absolute quantity
                    # Log exactly what is being returned
                    logger.info(f"Position Check: Found ACTIVE position: {determined_side} {qty_abs:.8f} {market_id} @ Entry={entry_price:.4f}")
                    return {'side': determined_side, 'qty': qty_abs, 'entry_price': entry_price}
                else:
                    # Size is zero or negligible, even though side was 'Buy' or 'Sell' (possible API state inconsistency or dust)
                    logger.debug(f"Skipping position entry: Position size {size} is <= epsilon {POSITION_QTY_EPSILON}, despite side being '{pos_side_v5}'. Treating as effectively closed.")
                    continue  # Move to the next entry in fetched_positions

            except (ValueError, TypeError) as e:
                logger.warning(f"Position Check: Error parsing size/quantity for {market_id}. Size string: '{size_str}', Error: {e}. Data: {pos}")
                continue  # Move to the next entry
        # End of loop through fetched_positions

        # If the loop completes without returning, no active position was found for the target symbol/mode
        logger.info(f"Position Check: No active position found for {market_id} (One-Way Mode) after checking {len(fetched_positions)} entries returned by fetch_positions.")
        return default_pos

    # --- Error Handling for fetch_positions call ---
    except ccxt.NetworkError as e:
        logger.warning(f"Position Check: Network error during fetch_positions for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Position Check: Exchange error during fetch_positions for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Position Check: Unexpected error during position check for {symbol}: {e}")
        logger.debug(traceback.format_exc())

    # Return default (no position) if any error occurred during the process
    logger.warning(f"Position Check: Returning default (No Position) due to error or no active position found for {symbol}.")
    return default_pos


# --- Leverage Setting ---
def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """
    Sets the leverage for a given futures symbol with retries and checks.
    Verifies that the symbol is a contract market first.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol in CCXT unified format.
        leverage (int): Desired leverage level (e.g., 10 for 10x).

    Returns:
        bool: True if leverage is successfully set (or was already correct), False otherwise.
    """
    logger.info(f"Leverage Setting: Attempting to set {leverage}x leverage for {symbol}...")
    try:
        market = exchange.market(symbol)
        # --- Market Type Check ---
        # Ensure it's a futures/swap market where leverage is applicable
        # CCXT market structure flags: 'swap', 'future', 'option'. 'contract' is a general flag.
        if not market.get('contract', False) or market.get('spot', False):
            logger.error(f"Leverage Setting: Cannot set leverage for non-contract market: {symbol}. Market type: {market.get('type')}")
            return False
        logger.debug(f"Market info for {symbol}: Type={market.get('type')}, Contract={market.get('contract')}")

    except (ccxt.BadSymbol, KeyError) as e:
         logger.error(f"Leverage Setting: Failed to get market info for symbol '{symbol}': {e}")
         return False
    except Exception as e:
         logger.error(f"Leverage Setting: Unexpected error getting market info for {symbol}: {e}")
         logger.debug(traceback.format_exc())
         return False

    # --- Attempt to Set Leverage with Retries ---
    for attempt in range(RETRY_COUNT):
        try:
            # Use the unified set_leverage method
            # Note: Some exchanges might require additional params (e.g., marginMode: isolated/cross)
            # Bybit usually defaults based on account settings or previous settings for the symbol.
            logger.debug(f"Leverage Setting: Calling exchange.set_leverage({leverage}, '{symbol}') (Attempt {attempt + 1}/{RETRY_COUNT})")
            response = exchange.set_leverage(leverage=leverage, symbol=symbol)  # Named args for clarity
            # Response format varies by exchange, log it for info
            logger.success(f"Leverage Setting: Successfully set leverage to {leverage}x for {symbol}. Response: {response}")
            return True

        except ccxt.ExchangeError as e:
            # Check for common messages indicating leverage is already set or no change needed
            error_msg_lower = str(e).lower()
            # Add more variations if needed based on observed Bybit error messages
            if "leverage not modified" in error_msg_lower or \
               "same leverage" in error_msg_lower or \
               "leverage is not changed" in error_msg_lower:
                logger.info(f"Leverage Setting: Leverage for {symbol} already set to {leverage}x (Confirmed by exchange message).")
                return True

            # Log other exchange errors and retry if attempts remain
            logger.warning(f"Leverage Setting: Exchange error on attempt {attempt + 1}/{RETRY_COUNT} for {symbol}: {e}")
            if attempt < RETRY_COUNT - 1:
                logger.debug(f"Retrying leverage setting after {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"Leverage Setting: Failed to set leverage for {symbol} after {RETRY_COUNT} attempts due to exchange error: {e}")
                # Optionally send SMS on final failure
                # send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL: Leverage set FAILED for {symbol} after retries: {e}")

        except ccxt.NetworkError as e:
             logger.warning(f"Leverage Setting: Network error on attempt {attempt + 1}/{RETRY_COUNT} for {symbol}: {e}")
             if attempt < RETRY_COUNT - 1:
                 time.sleep(RETRY_DELAY_SECONDS)
             else:
                 logger.error(f"Leverage Setting: Failed to set leverage for {symbol} after {RETRY_COUNT} attempts due to network errors.")

        except Exception as e:
            # Catch any other unexpected exceptions during the process
            logger.error(f"Leverage Setting: Unexpected error on attempt {attempt + 1} for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return False  # Exit immediately on unexpected errors

    # If loop finishes without returning True, it failed
    return False


# --- Close Position ---
def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal"
) -> Optional[Dict[str, Any]]:
    """
    Closes the specified active position using a market order with reduce_only=True.
    Includes a pre-close position re-validation step to mitigate race conditions.
    Sends SMS alerts on success or failure.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol (e.g., 'BTC/USDT:USDT').
        position_to_close (Dict[str, Any]): The position dictionary (containing at least
                                             'side' and 'qty') as identified *before*
                                             calling this function. Used for initial info.
        reason (str): Reason for closing (e.g., "SL", "TP", "Reversal", "Shutdown").

    Returns:
        Optional[Dict[str, Any]]: The CCXT order dictionary if the close order was
                                  successfully placed, None otherwise.
    """
    initial_side = position_to_close.get('side', POSITION_SIDE_NONE)
    initial_qty = position_to_close.get('qty', 0.0)
    market_base = symbol.split('/')[0] # For concise SMS alerts (e.g., "BTC")

    logger.info(f"Close Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}")

    # ** Crucial Pre-Close Re-validation Step **
    # Fetch the position state *again* immediately before attempting to close.
    # This helps prevent errors if the position was closed manually or by another
    # process between the main logic check and this function call.
    logger.debug(f"Close Position: Re-validating current position state for {symbol}...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position.get('side')
    live_amount_to_close = live_position.get('qty', 0.0)

    # Check if the live position still exists and matches the expected side (loosely)
    if live_position_side == POSITION_SIDE_NONE or live_amount_to_close <= POSITION_QTY_EPSILON:
        logger.warning(f"Close Position: Re-validation shows NO active position for {symbol} (Live Side: {live_position_side}, Qty: {live_amount_to_close:.8f}). Aborting close attempt.")
        # If the initial state *thought* there was a position, log that discrepancy
        if initial_side != POSITION_SIDE_NONE:
             logger.warning(f"Close Position: Discrepancy detected. Initial check showed {initial_side}, but live check shows none.")
        return None

    # Optional: Check if the live side matches the initial side (can be skipped if just closing whatever is there)
    # if live_position_side != initial_side:
    #     logger.warning(f"Close Position: Re-validation shows position side ({live_position_side}) differs from initial ({initial_side}). Closing the live position anyway.")

    # Use the freshly validated quantity and determine the closing side
    side_to_execute_close = SIDE_SELL if live_position_side == POSITION_SIDE_LONG else SIDE_BUY

    try:
        # Format the amount to the exchange's required precision
        amount_str = exchange.amount_to_precision(symbol, live_amount_to_close)
        amount_float = float(amount_str)

        # Final check on the amount after precision formatting
        if amount_float <= POSITION_QTY_EPSILON:
            logger.error(f"Close Position: Calculated closing amount after precision ({amount_str}) is zero or negligible. Aborting.")
            return None

        logger.warning(f"Close Position: Attempting to CLOSE {live_position_side} position ({reason}): "
                       f"Executing {side_to_execute_close.upper()} MARKET order for {amount_str} {symbol} "
                       f"with params {{'reduce_only': True}}...")

        # --- Execute the Closing Market Order ---
        params = {'reduce_only': True}  # Ensure this order only closes/reduces the position
        order = exchange.create_market_order(
            symbol=symbol,
            side=side_to_execute_close,
            amount=amount_float,
            params=params
        )
        # --- Post-Order Placement ---
        # Log success and key order details
        fill_price_str = f"{order.get('average'):.4f}" if order.get('average') is not None else "Market (Check Fills)"
        filled_qty_str = f"{order.get('filled'):.8f}" if order.get('filled') is not None else "?"
        order_id_short = str(order.get('id', 'N/A'))[-6:] # Get last 6 chars of ID for brevity
        cost_str = f"{order.get('cost'):.2f}" if order.get('cost') is not None else "N/A" # Cost in quote currency (USDT)

        logger.success(f"Close Position: CLOSE Order ({reason}) placed successfully for {symbol}. "
                       f"Qty Filled: {filled_qty_str}/{amount_str}, Avg Fill ~ {fill_price_str}, Cost: {cost_str} USDT. ID:...{order_id_short}")

        # Send SMS Alert on successful order placement
        sms_msg = (f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{fill_price_str} ({reason}). "
                   f"ID:...{order_id_short}")
        send_sms_alert(sms_msg)
        return order  # Return the order details dictionary

    # --- Error Handling for Order Creation ---
    except ccxt.InsufficientFunds as e:
        # This *shouldn't* happen often with reduce_only, but possible in weird margin scenarios
        logger.error(f"Close Position ({reason}): Insufficient funds error during close attempt: {e}")
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insufficient funds! Check margin.")
    except ccxt.NetworkError as e:
        logger.error(f"Close Position ({reason}): Network error placing close order: {e}")
        # SMS might fail here too if network is down
    except ccxt.ExchangeError as e:
        # Log the specific error message from the exchange (e.g., "position is zero", etc.)
        logger.error(f"Close Position ({reason}): Exchange error placing close order: {e}")
        # Check for specific common errors if needed (e.g., position already closed)
        # if "order would not reduce position size" in str(e).lower():
        #     logger.warning(f"Close Position ({reason}): Exchange indicates order would not reduce size. Position likely already closed or closing.")
        #     return None # Treat as potentially already closed
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Exchange error ({type(e).__name__}). Check logs.")
    except ValueError as e:
        # Catch potential errors from amount_to_precision or float conversion
        logger.error(f"Close Position ({reason}): Value error during amount processing (Qty: {live_amount_to_close}): {e}")
    except Exception as e:
        logger.error(f"Close Position ({reason}): Unexpected error placing close order: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Unexpected error ({type(e).__name__}). Check logs.")

    # Return None if any exception occurred during order placement
    return None


# --- Risk Calculation ---
def calculate_position_size(
    equity: float, risk_per_trade_pct: float, entry_price: float, stop_loss_price: float,
    leverage: int, symbol: str, exchange: ccxt.Exchange
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates position size (quantity) based on risk percentage, entry/SL prices,
    and estimates the required initial margin (without buffer).

    Args:
        equity (float): Current account equity (or relevant balance) in USDT.
        risk_per_trade_pct (float): Desired risk percentage (e.g., 0.01 for 1%).
        entry_price (float): Estimated entry price.
        stop_loss_price (float): Calculated stop-loss price.
        leverage (int): Leverage used for the symbol.
        symbol (str): Market symbol (used for precision).
        exchange (ccxt.Exchange): CCXT exchange instance (used for precision).


    Returns:
        Tuple[Optional[float], Optional[float]]: (quantity, required_margin_estimate)
           - quantity (float): The calculated position size in base currency (e.g., BTC).
           - required_margin_estimate (float): Estimated USDT margin needed for the position.
           Returns (None, None) if calculation is invalid or results in zero/negative size.
    """
    logger.debug(f"Risk Calc Input: Equity={equity:.2f}, Risk%={risk_per_trade_pct:.3%}, "
                 f"Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x, Symbol={symbol}")

    # --- Input Validation ---
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error("Risk Calc: Invalid entry or stop-loss price (must be > 0).")
        return None, None
    price_difference_per_unit = abs(entry_price - stop_loss_price)
    if price_difference_per_unit < POSITION_QTY_EPSILON:
        logger.error(f"Risk Calc: Entry ({entry_price}) and stop-loss ({stop_loss_price}) prices are too close or identical. Cannot calculate size.")
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(f"Risk Calc: Invalid risk_per_trade_pct: {risk_per_trade_pct}. Must be between 0 and 1 (exclusive).")
        return None, None
    if equity <= 0:
         logger.error(f"Risk Calc: Invalid equity (<= 0): {equity:.2f}")
         return None, None
    if leverage <= 0:
         logger.error(f"Risk Calc: Invalid leverage (<= 0): {leverage}")
         return None, None

    # --- Calculation ---
    # 1. Calculate Risk Amount in USDT
    risk_amount_usdt = equity * risk_per_trade_pct

    # 2. Calculate Position Size (Quantity)
    # Quantity = Risk Amount / (Price difference per unit)
    quantity = risk_amount_usdt / price_difference_per_unit

    # Apply exchange precision to the calculated quantity *early*
    try:
        quantity_precise_str = exchange.amount_to_precision(symbol, quantity)
        quantity_precise = float(quantity_precise_str)
        logger.debug(f"Risk Calc: Raw Qty={quantity:.8f}, Precise Qty={quantity_precise_str}")
        quantity = quantity_precise  # Use the precise quantity for further calculations
    except Exception as e:
         logger.warning(f"Risk Calc: Could not apply precision to quantity {quantity:.8f} for {symbol}. Using raw value. Error: {e}")


    # 3. Check if calculated quantity is valid
    if quantity <= POSITION_QTY_EPSILON:
        logger.warning(f"Risk Calc: Calculated quantity ({quantity:.8f}) is zero or negligible based on risk parameters. "
                       f"RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference_per_unit:.4f}")
        return None, None

    # 4. Estimate Position Value and Required Margin (Initial Margin)
    position_value_usdt = quantity * entry_price
    # Required Margin = Position Value / Leverage
    required_margin_estimate = position_value_usdt / leverage
    # Note: This is the *initial* margin estimate. Actual margin requirements might
    # include fees, funding rates, and vary slightly based on exact fill price.
    # The margin *buffer* check happens later using free balance.

    logger.debug(f"Risk Calc Output: RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference_per_unit:.4f} "
                 f"=> PreciseQty={quantity:.8f}, EstPosValue={position_value_usdt:.2f}, EstMarginBase={required_margin_estimate:.2f}")
    return quantity, required_margin_estimate


# --- Place Order ---
def place_risked_market_order(
    exchange: ccxt.Exchange, symbol: str, side: str,
    risk_percentage: float, current_atr: Optional[float], sl_atr_multiplier: float,
    leverage: int, max_order_cap_usdt: float, margin_check_buffer: float
) -> Optional[Dict[str, Any]]:
    """
    Calculates position size based on risk & ATR SL, checks balance, limits, and margin,
    then places a market order if all checks pass. Sends SMS alerts.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol (e.g., 'BTC/USDT:USDT').
        side (str): 'buy' or 'sell'.
        risk_percentage (float): Risk per trade as a decimal (e.g., 0.01 for 1%).
        current_atr (Optional[float]): The current ATR value for SL calculation.
        sl_atr_multiplier (float): Multiplier for ATR to determine SL distance.
        leverage (int): Leverage confirmed for the symbol.
        max_order_cap_usdt (float): Maximum allowed position size in USDT value.
        margin_check_buffer (float): Buffer multiplier for checking available margin (e.g., 1.05 for 5% buffer).

    Returns:
        Optional[Dict[str, Any]]: The CCXT order dictionary if the order was successfully placed, None otherwise.
    """
    market_base = symbol.split('/')[0] # For concise SMS alerts
    logger.info(f"Place Order: Initiating {side.upper()} market order placement for {symbol}...")

    # --- Pre-computation Checks ---
    if current_atr is None or current_atr <= 0:
        logger.error(f"Place Order ({side.upper()}): Invalid ATR value ({current_atr}) provided for SL calculation. Cannot place order.")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid ATR provided.")
        return None

    entry_price_estimate: Optional[float] = None
    stop_loss_price: Optional[float] = None
    final_quantity: Optional[float] = None
    final_quantity_str: str = "N/A"

    try:
        # --- 1. Get Balance, Market Info, and Limits ---
        logger.debug("Place Order: Fetching balance and market details...")
        balance = exchange.fetch_balance()
        market = exchange.market(symbol)
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        min_qty = amount_limits.get('min')
        max_qty = amount_limits.get('max')
        qty_precision = market.get('precision', {}).get('amount') # Number of decimal places for quantity
        price_precision = market.get('precision', {}).get('price') # Number of decimal places for price

        # Get available equity/free balance in USDT
        usdt_balance = balance.get(USDT_SYMBOL, {})
        usdt_total = usdt_balance.get('total') # Total equity
        usdt_free = usdt_balance.get('free') # Free margin available

        # Use total equity for risk calculation, but free margin for placement check
        # Fallback logic if 'total' is None or zero
        if usdt_total is None or usdt_total <= 0:
             logger.warning(f"Place Order: USDT 'total' balance is {usdt_total}. Using 'free' balance ({usdt_free}) for equity calculation.")
             usdt_equity = usdt_free if usdt_free is not None else 0.0
        else:
             usdt_equity = usdt_total

        if usdt_equity is None or usdt_equity <= 0:
            logger.error(f"Place Order ({side.upper()}): Cannot determine valid account equity (USDT total/free is zero, negative, or None).")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Zero/Invalid equity.")
            return None
        if usdt_free is None or usdt_free < 0: # Free margin must be available
             logger.error(f"Place Order ({side.upper()}): Invalid free margin (USDT free is {usdt_free}).")
             send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid free margin.")
             return None

        logger.debug(f"Place Order: Account Balance -> Equity={usdt_equity:.2f} USDT, Free Margin={usdt_free:.2f} USDT")
        logger.debug(f"Place Order: Market Limits -> MinQty={min_qty}, MaxQty={max_qty}, QtyPrec={qty_precision}, PricePrec={price_precision}")
        logger.debug(f"Place Order: Market Price Limits -> MinPrice={price_limits.get('min')}, MaxPrice={price_limits.get('max')}")


        # --- 2. Estimate Entry Price ---
        # Use a shallow order book fetch for a slightly more realistic entry price estimate than ticker 'last'
        logger.debug(f"Place Order: Fetching shallow OB (depth {SHALLOW_OB_FETCH_DEPTH}) for entry price estimate...")
        ob_data = analyze_order_book(exchange, symbol, SHALLOW_OB_FETCH_DEPTH, SHALLOW_OB_FETCH_DEPTH)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")

        if side == SIDE_BUY and best_ask and best_ask > 0:
            entry_price_estimate = best_ask
            logger.debug(f"Place Order: Using best ASK {entry_price_estimate:.4f} from shallow OB as BUY entry estimate.")
        elif side == SIDE_SELL and best_bid and best_bid > 0:
            entry_price_estimate = best_bid
            logger.debug(f"Place Order: Using best BID {entry_price_estimate:.4f} from shallow OB as SELL entry estimate.")
        else:
            # Fallback to ticker 'last' price if OB fetch fails or returns invalid prices
            logger.warning("Place Order: Shallow OB fetch failed or returned invalid bid/ask. Falling back to ticker 'last' price for entry estimate.")
            try:
                ticker = exchange.fetch_ticker(symbol)
                entry_price_estimate = ticker.get('last')
                if entry_price_estimate: logger.debug(f"Place Order: Using ticker 'last' price {entry_price_estimate:.4f} as entry estimate.")
            except Exception as ticker_err:
                logger.error(f"Place Order: Failed to fetch ticker price as fallback: {ticker_err}")
                entry_price_estimate = None # Ensure it's None if ticker fails too

        # Validate the estimated entry price
        if entry_price_estimate is None or entry_price_estimate <= 0:
            logger.error(f"Place Order ({side.upper()}): Could not determine a valid entry price estimate (Estimate: {entry_price_estimate}). Cannot place order.")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): No valid entry price estimate.")
            return None

        # --- 3. Calculate Stop Loss Price ---
        stop_loss_distance = current_atr * sl_atr_multiplier
        if side == SIDE_BUY:
            stop_loss_price = entry_price_estimate - stop_loss_distance
        else: # SIDE_SELL
            stop_loss_price = entry_price_estimate + stop_loss_distance

        # Ensure SL price is positive and respects exchange minimum price limit
        min_price_limit = price_limits.get('min')
        if min_price_limit is not None and stop_loss_price < min_price_limit:
            logger.warning(f"Place Order: Calculated SL price {stop_loss_price:.4f} is below minimum limit {min_price_limit}. Adjusting SL to minimum limit.")
            stop_loss_price = min_price_limit
        elif stop_loss_price <= 0:
             logger.warning(f"Place Order: Calculated SL price {stop_loss_price:.4f} is zero or negative. Adjusting SL to a small positive value (e.g., min price limit or tiny fraction).")
             # Adjust to min price limit if available, otherwise a very small number might be risky.
             # For simplicity, let's prevent order if SL becomes invalid. A better approach might use a fixed % SL as fallback.
             if min_price_limit is not None:
                 stop_loss_price = min_price_limit
             else:
                 logger.error(f"Place Order ({side.upper()}): Calculated SL price is invalid ({stop_loss_price:.4f}) and no minimum price limit found. Cannot place order.")
                 send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid SL price calc.")
                 return None


        # Apply exchange price precision to the calculated SL price
        try:
            stop_loss_price_str = exchange.price_to_precision(symbol, stop_loss_price)
            stop_loss_price = float(stop_loss_price_str) # Convert back to float after formatting
            logger.info(f"Place Order: Calculated SL -> EntryEst={entry_price_estimate:.4f}, "
                        f"ATR({current_atr:.5f}) * {sl_atr_multiplier} (Dist={stop_loss_distance:.4f}) => SL Price ~ {stop_loss_price_str}")
        except Exception as e:
            logger.error(f"Place Order: Error applying precision to SL price {stop_loss_price:.4f}: {e}. Aborting.")
            return None


        # --- 4. Calculate Position Size and Estimate Initial Margin ---
        calculated_quantity, required_margin_estimate = calculate_position_size(
            equity=usdt_equity,
            risk_per_trade_pct=risk_percentage,
            entry_price=entry_price_estimate,
            stop_loss_price=stop_loss_price,
            leverage=leverage,
            symbol=symbol,
            exchange=exchange
        )

        if calculated_quantity is None or required_margin_estimate is None:
            logger.error(f"Place Order ({side.upper()}): Failed to calculate valid position size/margin based on risk parameters. Check logs from calculate_position_size.")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Risk calculation failed.")
            return None

        final_quantity = calculated_quantity # Start with the risk-calculated quantity

        # --- 5. Apply Max Order Cap (USDT Value) ---
        order_value_usdt = final_quantity * entry_price_estimate
        if order_value_usdt > max_order_cap_usdt:
            logger.warning(f"Place Order: Initial risk-based order value {order_value_usdt:.2f} USDT exceeds MAX Cap {max_order_cap_usdt:.2f} USDT. Capping quantity.")
            final_quantity = max_order_cap_usdt / entry_price_estimate
            # Recalculate estimated margin based on the capped quantity
            required_margin_estimate = (max_order_cap_usdt / leverage)
            logger.info(f"Place Order: Quantity capped to ~ {final_quantity:.8f} based on Max Cap. New Est. Margin ~ {required_margin_estimate:.2f} USDT.")

        # --- 6. Apply Exchange Precision and Check Min/Max Quantity Limits ---
        try:
            final_quantity_str = exchange.amount_to_precision(symbol, final_quantity)
            final_quantity = float(final_quantity_str)
            logger.info(f"Place Order: Quantity after precision: {final_quantity_str}")
        except Exception as e:
            logger.error(f"Place Order: Error applying precision to final quantity {final_quantity:.8f}: {e}. Aborting.")
            return None

        # Check if final quantity is effectively zero after precision
        if final_quantity <= POSITION_QTY_EPSILON:
            logger.error(f"Place Order ({side.upper()}): Final quantity {final_quantity_str} is zero or negligible after risk calc/capping/precision. Cannot place order.")
            return None # No order to place

        # Check against Min/Max quantity limits from the exchange
        if min_qty is not None and final_quantity < min_qty:
            logger.error(f"Place Order ({side.upper()}): Final Qty {final_quantity_str} is LESS than exchange minimum quantity {min_qty}. Skipping order.")
            # Optionally, could place the minimum qty order, but that deviates from risk calc. Skipping is safer.
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Qty {final_quantity_str} < Min {min_qty}.")
            return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"Place Order ({side.upper()}): Final Qty {final_quantity_str} is GREATER than exchange maximum quantity {max_qty}. Capping to max qty.")
            final_quantity = max_qty
            # Re-apply precision after capping to max_qty
            try:
                final_quantity_str = exchange.amount_to_precision(symbol, final_quantity)
                final_quantity = float(final_quantity_str)
                # Recalculate estimated margin based on max quantity
                required_margin_estimate = ((final_quantity * entry_price_estimate) / leverage)
                logger.info(f"Place Order: Quantity capped to exchange max: {final_quantity_str}. New Est. Margin ~ {required_margin_estimate:.2f} USDT.")
            except Exception as e:
                logger.error(f"Place Order: Error applying precision after capping to max qty {max_qty}: {e}. Aborting.")
                return None

        # Re-calculate final estimated position value and margin after all adjustments
        final_pos_value = final_quantity * entry_price_estimate
        final_req_margin = final_pos_value / leverage
        logger.info(f"Place Order: Final Order Details -> Qty={final_quantity_str}, "
                    f"Est. Value={final_pos_value:.2f} USDT, "
                    f"Est. Req. Margin={final_req_margin:.2f} USDT")

        # --- 7. Check Available Margin (using Free Balance and Buffer) ---
        required_margin_with_buffer = final_req_margin * margin_check_buffer
        if usdt_free < required_margin_with_buffer:
            logger.error(f"Place Order ({side.upper()}): Insufficient FREE margin. "
                         f"Requires ~ {required_margin_with_buffer:.2f} USDT "
                         f"(Base: {final_req_margin:.2f}, Buffer: {margin_check_buffer:.1%}), "
                         f"Available Free: {usdt_free:.2f} USDT.")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{required_margin_with_buffer:.2f}, Have {usdt_free:.2f})")
            return None
        else:
             logger.info(f"Place Order: Margin check passed. Required w/ Buffer: {required_margin_with_buffer:.2f} USDT <= Available Free: {usdt_free:.2f} USDT.")


        # --- 8. Place the Market Order ---
        logger.warning(f"*** Placing {side.upper()} MARKET ORDER: {final_quantity_str} {symbol} ***")
        # Ensure no extra params interfere with basic market order unless intended
        params = {} # Start fresh, add specific params if needed (e.g., timeInForce)
        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=final_quantity,
            params=params
        )

        # --- 9. Post-Order Handling & Logging ---
        # Extract details from the returned order object (structure varies slightly by exchange/ccxt version)
        filled_qty = order.get('filled', 0.0) # Amount actually filled
        avg_fill_price = order.get('average') # Average price if filled
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A')
        order_cost = order.get('cost', 0.0) # Cost in quote currency (USDT)

        fill_price_str = f"{avg_fill_price:.4f}" if avg_fill_price is not None else "N/A"
        filled_qty_str = f"{filled_qty:.8f}" if filled_qty is not None else "?"
        order_id_short = str(order_id)[-6:] # Last 6 chars for brevity
        cost_str = f"{order_cost:.2f}" if order_cost is not None else "N/A"

        logger.success(f"Place Order: {side.upper()} Market Order Placed Successfully! "
                       f"ID:...{order_id_short}, Status: {order_status}, "
                       f"Filled Qty: {filled_qty_str}/{final_quantity_str}, "
                       f"Avg Fill Price: {fill_price_str}, Cost: {cost_str} USDT")

        # Send SMS Alert for successful entry
        sms_msg = (f"[{market_base}] ENTER {side.upper()} {final_quantity_str} @ ~{fill_price_str}. "
                   f"SL ~ {stop_loss_price:.4f}. ID:...{order_id_short}") # Include calculated SL for reference
        send_sms_alert(sms_msg)

        # --- !!! CRITICAL SCALPING TODO !!! ---
        # The calculated SL/TP based on estimated entry is NOT sufficient.
        # You MUST place exchange-native SL/TP orders *after* this market order fills,
        # using the actual `avg_fill_price` and `filled_qty`.
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.critical(f"!!! ACTION REQUIRED: Loop-based SL/TP is active for position ID ...{order_id_short}. !!!")
        logger.critical(f"!!! This is UNSAFE. Implement exchange-native SL/TP orders based on fill price {fill_price_str} !!!")
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Example (Conceptual - Requires specific CCXT implementation for conditional orders):
        if avg_fill_price and filled_qty > 0:
             actual_sl_price = avg_fill_price - stop_loss_distance # Recalculate based on fill
             actual_tp_price = avg_fill_price + tp_distance # Recalculate based on fill
             exchange.create_order(symbol, 'stopMarket', 'sell' if side == 'buy' else 'buy', filled_qty, params={'stopPrice': actual_sl_price, 'reduce_only': True})
             exchange.create_order(symbol, 'takeProfitMarket', 'sell' if side == 'buy' else 'buy', filled_qty, params={'stopPrice': actual_tp_price, 'reduce_only': True})
        # ---------------------------------------

        return order # Return the order dictionary

    # --- Exception Handling for Order Placement Process ---
    except ccxt.InsufficientFunds as e:
        # This might occur if free margin calculation was slightly off or fees pushed it over
        logger.error(f"Place Order ({side.upper()}): Insufficient funds error during order placement: {e}")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient funds reported by exchange!")
    except ccxt.NetworkError as e:
        logger.error(f"Place Order ({side.upper()}): Network error during order placement: {e}")
        # SMS might fail too
    except ccxt.ExchangeError as e:
        # Log specific exchange errors (e.g., invalid order params, margin issues not caught earlier)
        logger.error(f"Place Order ({side.upper()}): Exchange error during order placement: {e}")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Exchange error ({type(e).__name__}). Check logs.")
    except ValueError as e:
        # Catch potential errors from precision formatting or float conversions
         logger.error(f"Place Order ({side.upper()}): Value error during price/amount processing: {e}")
    except Exception as e:
        logger.error(f"Place Order ({side.upper()}): Unexpected critical error during order placement logic: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Unexpected error ({type(e).__name__}). Check logs.")

    # Return None if any exception occurred during the process
    return None


# --- Trading Logic ---
def trade_logic(
    exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame,
    st_len: int, st_mult: float, cf_st_len: int, cf_st_mult: float,
    atr_len: int, vol_ma_len: int,
    risk_pct: float, sl_atr_mult: float, tp_atr_mult: float, leverage: int,
    max_order_cap: float, margin_buffer: float
) -> None:
    """
    Executes the main trading logic for a single cycle.
    1. Calculates indicators (Supertrends, ATR, Volume).
    2. Analyzes Order Book (conditionally).
    3. Gets current position status.
    4. Logs current market state.
    5. Checks loop-based Stop-Loss / Take-Profit (UNSAFE FOR LIVE SCALPING, default OFF).
    6. Checks for entry signals based on dual Supertrend and optional confirmations.
    7. Executes entry or exit actions (closing existing position if reversing).

    *** WARNING: Contains optional loop-based SL/TP check which is unsuitable for live scalping. ***

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): Trading symbol (e.g., 'BTC/USDT:USDT').
        df (pd.DataFrame): DataFrame with fresh OHLCV data.
        st_len, st_mult: Primary Supertrend parameters.
        cf_st_len, cf_st_mult: Confirmation Supertrend parameters.
        atr_len: ATR period for SL/TP calculation.
        vol_ma_len: Volume MA period.
        risk_pct: Risk per trade percentage.
        sl_atr_mult, tp_atr_mult: ATR multipliers for SL/TP.
        leverage: Confirmed leverage for the symbol.
        max_order_cap: Max USDT order value cap.
        margin_buffer: Buffer for margin check before placing order.
    """
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"========== New Check Cycle: {symbol} | Candle Time: {cycle_time_str} ==========")

    # --- Data Validation ---
    # Ensure sufficient data rows for the longest indicator lookback period + buffer
    required_rows = max(st_len, cf_st_len, atr_len, vol_ma_len) + 5 # Add a small buffer
    if df is None or len(df) < required_rows:
        logger.warning(f"Trade Logic: Insufficient data ({len(df) if df is not None else 0} rows, "
                       f"need at least {required_rows} for indicators). Skipping logic cycle.")
        return

    # Initialize variables
    ob_data: Optional[Dict[str, Optional[float]]] = None
    current_atr: Optional[float] = None
    action_taken_this_cycle: bool = False # Flag to track if an order was placed/closed

    try:
        # === 1. Calculate Indicators ===
        logger.debug("Trade Logic: Calculating technical indicators...")
        # Calculate Primary Supertrend
        df = calculate_supertrend(df, st_len, st_mult)
        # Calculate Confirmation Supertrend
        df = calculate_supertrend(df, cf_st_len, cf_st_mult, prefix="confirm_")
        # Calculate ATR (for SL/TP) and Volume metrics
        vol_atr_data = analyze_volume_atr(df, atr_len, vol_ma_len)
        current_atr = vol_atr_data.get("atr")

        # --- Validate Indicator Results ---
        # Check if essential columns exist and the last row has valid values
        required_indicator_cols = ['close', 'supertrend', 'trend', 'st_long', 'st_short',
                                   'confirm_supertrend', 'confirm_trend']
        if not all(col in df.columns for col in required_indicator_cols):
             logger.warning("Trade Logic: One or more required indicator columns missing after calculation. Skipping cycle.")
             return
        if df.iloc[-1][required_indicator_cols].isnull().any():
             logger.warning("Trade Logic: Indicator calculation resulted in NA values for the latest candle. Skipping cycle.")
             logger.debug(f"NA Values in last row:\n{df.iloc[-1][df.iloc[-1].isnull()]}")
             return
        # Validate ATR specifically, as it's crucial for risk management
        if current_atr is None or current_atr <= 0:
             logger.warning(f"Trade Logic: Invalid ATR ({current_atr}) calculated. Cannot determine SL/TP. Skipping cycle.")
             return

        # === Extract Data from the Last Closed Candle ===
        last_candle = df.iloc[-1]
        current_price: float = last_candle['close'] # Price at the close of the last candle
        primary_st_val: float = last_candle['supertrend']
        primary_trend_up: bool = last_candle["trend"]
        primary_long_signal: bool = last_candle["st_long"] # Primary ST flipped Long THIS candle
        primary_short_signal: bool = last_candle["st_short"] # Primary ST flipped Short THIS candle
        confirm_st_val: float = last_candle['confirm_supertrend']
        confirm_trend_up: bool = last_candle["confirm_trend"] # Current state of Confirmation ST trend

        # === 2. Analyze Order Book (Conditionally) ===
        # Fetch OB if configured to do so every cycle OR if a primary signal just occurred (potential entry)
        if FETCH_ORDER_BOOK_PER_CYCLE or primary_long_signal or primary_short_signal:
            logger.debug("Trade Logic: Fetching and analyzing order book...")
            ob_data = analyze_order_book(exchange, symbol, ORDER_BOOK_DEPTH, ORDER_BOOK_FETCH_LIMIT)
        else:
            logger.debug("Trade Logic: Order book fetch skipped (FETCH_PER_CYCLE=False and no primary signal generated this candle).")

        # === 3. Get Current Position Status ===
        # This fetches the live position state from the exchange
        position = get_current_position(exchange, symbol)
        position_side: str = position['side']
        position_qty: float = position['qty']
        entry_price: float = position['entry_price']

        # === 4. Log Current State ===
        # Prepare formatted strings for logging, handling None values
        volume_ratio = vol_atr_data.get("volume_ratio")
        volume_spike = volume_ratio is not None and volume_ratio > VOLUME_SPIKE_THRESHOLD
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread_val = ob_data.get('spread') if ob_data else None

        atr_str = f"{current_atr:.5f}"
        vol_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else 'N/A'
        vol_spike_str = f"{volume_spike}" if volume_ratio is not None else 'N/A'
        bid_ask_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else 'N/A'
        spread_str = f"{spread_val:.4f}" if spread_val is not None else 'N/A'

        logger.info(f"State | Price: {current_price:.4f}, ATR({atr_len}): {atr_str}")
        logger.info(f"State | Volume: Ratio={vol_ratio_str}, Spike={vol_spike_str} (Thr={VOLUME_SPIKE_THRESHOLD:.1f})")
        logger.info(f"State | ST({st_len},{st_mult}): {'Up' if primary_trend_up else 'Down'} ({primary_st_val:.4f}) | Signal: {'LONG' if primary_long_signal else ('SHORT' if primary_short_signal else 'None')}")
        logger.info(f"State | ConfirmST({cf_st_len},{cf_st_mult}): {'Up' if confirm_trend_up else 'Down'} ({confirm_st_val:.4f})")
        logger.info(f"State | OrderBook(D{ORDER_BOOK_DEPTH}): Ratio={bid_ask_ratio_str}, Spread={spread_str}")
        logger.info(f"State | Position: Side={position_side}, Qty={position_qty:.8f}, Entry={entry_price:.4f}")

        # === 5. Check Stop-Loss / Take-Profit (Loop-based - HIGH RISK / INACCURATE / DEFAULT OFF) ===
        if ENABLE_LOOP_SLTP_CHECK and position_side != POSITION_SIDE_NONE and entry_price > 0 and current_atr > 0:
            logger.warning("!!! Trade Logic: Checking active position SL/TP based on LAST CLOSED PRICE (LOOP CHECK - UNSAFE FOR SCALPING) !!!")
            sl_triggered, tp_triggered = False, False
            stop_price, profit_price = 0.0, 0.0
            sl_distance = current_atr * sl_atr_mult
            tp_distance = current_atr * tp_atr_mult

            if position_side == POSITION_SIDE_LONG:
                stop_price = entry_price - sl_distance
                profit_price = entry_price + tp_distance
                logger.debug(f"SL/TP Check (Long): Entry={entry_price:.4f}, CurrentClose={current_price:.4f}, Target SL={stop_price:.4f}, Target TP={profit_price:.4f}")
                if current_price <= stop_price: sl_triggered = True
                if current_price >= profit_price: tp_triggered = True
            elif position_side == POSITION_SIDE_SHORT:
                stop_price = entry_price + sl_distance
                profit_price = entry_price - tp_distance
                logger.debug(f"SL/TP Check (Short): Entry={entry_price:.4f}, CurrentClose={current_price:.4f}, Target SL={stop_price:.4f}, Target TP={profit_price:.4f}")
                if current_price >= stop_price: sl_triggered = True
                if current_price <= profit_price: tp_triggered = True

            # --- Execute Close if SL/TP Triggered by this Inaccurate Loop Check ---
            if sl_triggered:
                reason = f"SL Loop @ {current_price:.4f} (Target ~ {stop_price:.4f})"
                logger.critical(f"*** TRADE EXIT (LOOP CHECK): STOP-LOSS TRIGGERED! ({reason}). Closing {position_side}. ***")
                logger.critical("!!! THIS SL TRIGGER IS BASED ON DELAYED CHECK - ACTUAL LOSS MAY BE SIGNIFICANTLY HIGHER !!!")
                # Pass the 'position' dict we got at the start of the cycle for info, but close_position re-validates
                close_result = close_position(exchange, symbol, position, reason="SL (Loop Check)")
                if close_result: action_taken_this_cycle = True
                # Exit logic for this cycle after attempting close, regardless of success
                # because the condition was met. Prevents trying to enter immediately after SL.
                return
            elif tp_triggered:
                reason = f"TP Loop @ {current_price:.4f} (Target ~ {profit_price:.4f})"
                logger.success(f"*** TRADE EXIT (LOOP CHECK): TAKE-PROFIT TRIGGERED! ({reason}). Closing {position_side}. ***")
                logger.warning("!!! TAKE PROFIT VIA LOOP CHECK MAY MISS BETTER FILLS OR EXPERIENCE SLIPPAGE !!!")
                close_result = close_position(exchange, symbol, position, reason="TP (Loop Check)")
                if close_result: action_taken_this_cycle = True
                # Exit logic for this cycle after attempting close
                return
            else:
                 logger.debug("Trade Logic: Position SL/TP not triggered based on last close price (Loop Check).")
        elif position_side != POSITION_SIDE_NONE:
             # Log if position exists but SL/TP check couldn't run (e.g., bad entry price or ATR or disabled)
             if ENABLE_LOOP_SLTP_CHECK:
                 logger.warning(f"Trade Logic: Position ({position_side}) exists but loop-based SL/TP check skipped (Entry={entry_price}, ATR={current_atr}).")
             else:
                 logger.debug("Trade Logic: Loop-based SL/TP check is disabled (ENABLE_LOOP_SLTP_CHECK=False).")


        # === 6. Check for Entry Signals ===
        # Only proceed if no SL/TP exit was triggered above
        logger.debug("Trade Logic: Checking entry signals...")

        # --- Define Confirmation Conditions ---
        # Order Book Confirmation (only if required and data is available)
        ob_check_required = FETCH_ORDER_BOOK_PER_CYCLE # If True, we need OB data every time
        ob_available = ob_data is not None and bid_ask_ratio is not None
        passes_long_ob = not ob_check_required or (ob_available and bid_ask_ratio >= ORDER_BOOK_RATIO_THRESHOLD_LONG)
        passes_short_ob = not ob_check_required or (ob_available and bid_ask_ratio <= ORDER_BOOK_RATIO_THRESHOLD_SHORT)
        ob_log_str = f"Ratio={bid_ask_ratio_str} (LongThr={ORDER_BOOK_RATIO_THRESHOLD_LONG}, ShortThr={ORDER_BOOK_RATIO_THRESHOLD_SHORT}, Required={ob_check_required})"

        # Volume Confirmation (only if required and data is available)
        volume_check_required = REQUIRE_VOLUME_SPIKE_FOR_ENTRY
        volume_available = volume_ratio is not None # Need ratio to check spike
        passes_volume = not volume_check_required or (volume_available and volume_spike)
        vol_log_str = f"Ratio={vol_ratio_str}, Spike={vol_spike_str} (Required={volume_check_required})"


        # --- Long Entry Condition ---
        enter_long = (
            position_side != POSITION_SIDE_LONG and          # Prevent adding to existing long
            primary_long_signal and                          # Primary ST must have flipped Long THIS candle
            confirm_trend_up and                             # Confirmation ST trend must be Up (agreement)
            passes_long_ob and                               # OB condition met (or not required)
            passes_volume                                    # Volume condition met (or not required)
        )
        logger.debug(f"Entry Check (Long): PrimaryFlip={primary_long_signal}, ConfirmTrendUp={confirm_trend_up}, "
                     f"OB OK={passes_long_ob} [{ob_log_str}], Vol OK={passes_volume} [{vol_log_str}] "
                     f"=> Enter Long = {enter_long}")

        # --- Short Entry Condition ---
        enter_short = (
            position_side != POSITION_SIDE_SHORT and         # Prevent adding to existing short
            primary_short_signal and                         # Primary ST must have flipped Short THIS candle
            not confirm_trend_up and                         # Confirmation ST trend must be Down (agreement)
            passes_short_ob and                              # OB condition met (or not required)
            passes_volume                                    # Volume condition met (or not required)
        )
        logger.debug(f"Entry Check (Short): PrimaryFlip={primary_short_signal}, ConfirmTrendDown={not confirm_trend_up}, "
                      f"OB OK={passes_short_ob} [{ob_log_str}], Vol OK={passes_volume} [{vol_log_str}] "
                      f"=> Enter Short = {enter_short}")


        # === 7. Execute Entry / Reversal Actions ===
        if enter_long:
            logger.success(f"*** TRADE SIGNAL: CONFIRMED LONG ENTRY SIGNAL for {symbol} at ~{current_price:.4f} ***")
            if position_side == POSITION_SIDE_SHORT:
                # --- Reversal: Close Short, then Enter Long ---
                logger.warning("Trade Action: LONG signal received while SHORT. Attempting REVERSAL.")
                # 1. Close the existing short position
                logger.info("Reversal Step 1: Closing existing SHORT position...")
                close_result = close_position(exchange, symbol, position, reason="Reversal to Long")
                action_taken_this_cycle = True # Attempted an action
                if close_result:
                    logger.info(f"Reversal Step 1: Close SHORT order placed. Waiting {POST_CLOSE_DELAY_SECONDS}s...")
                    time.sleep(POST_CLOSE_DELAY_SECONDS) # Allow time for closure to process/API update
                    # 2. Re-check position state AFTER attempting close
                    logger.info("Reversal Step 2: Re-checking position status after close attempt...")
                    current_pos_after_close = get_current_position(exchange, symbol)
                    if current_pos_after_close['side'] == POSITION_SIDE_NONE:
                        logger.success("Reversal Step 2: Confirmed SHORT position closed.")
                        # 3. Enter the new long position
                        logger.info("Reversal Step 3: Proceeding with LONG entry.")
                        place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                    else:
                        logger.error(f"Reversal Step 2: Failed to confirm SHORT position closure after reversal signal (Still showing: {current_pos_after_close['side']}). Skipping LONG entry.")
                else:
                     logger.error("Reversal Step 1: Failed to place order to close SHORT position. Skipping LONG entry.")

            elif position_side == POSITION_SIDE_NONE:
                # --- New Entry: Enter Long ---
                logger.info("Trade Action: Entering NEW LONG position.")
                place_result = place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                if place_result: action_taken_this_cycle = True
            else:
                 # Should not happen if enter_long condition is correct, but as a safeguard:
                 logger.warning("Trade Action: Signal LONG ignored (already long? Check logic).")


        elif enter_short:
            logger.success(f"*** TRADE SIGNAL: CONFIRMED SHORT ENTRY SIGNAL for {symbol} at ~{current_price:.4f} ***")
            if position_side == POSITION_SIDE_LONG:
                # --- Reversal: Close Long, then Enter Short ---
                logger.warning("Trade Action: SHORT signal received while LONG. Attempting REVERSAL.")
                # 1. Close the existing long position
                logger.info("Reversal Step 1: Closing existing LONG position...")
                close_result = close_position(exchange, symbol, position, reason="Reversal to Short")
                action_taken_this_cycle = True # Attempted an action
                if close_result:
                    logger.info(f"Reversal Step 1: Close LONG order placed. Waiting {POST_CLOSE_DELAY_SECONDS}s...")
                    time.sleep(POST_CLOSE_DELAY_SECONDS)
                    # 2. Re-check position state AFTER attempting close
                    logger.info("Reversal Step 2: Re-checking position status after close attempt...")
                    current_pos_after_close = get_current_position(exchange, symbol)
                    if current_pos_after_close['side'] == POSITION_SIDE_NONE:
                        logger.success("Reversal Step 2: Confirmed LONG position closed.")
                        # 3. Enter the new short position
                        logger.info("Reversal Step 3: Proceeding with SHORT entry.")
                        place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                    else:
                        logger.error(f"Reversal Step 2: Failed to confirm LONG position closure after reversal signal (Still showing: {current_pos_after_close['side']}). Skipping SHORT entry.")
                else:
                    logger.error("Reversal Step 1: Failed to place order to close LONG position. Skipping SHORT entry.")

            elif position_side == POSITION_SIDE_NONE:
                # --- New Entry: Enter Short ---
                logger.info("Trade Action: Entering NEW SHORT position.")
                place_result = place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                if place_result: action_taken_this_cycle = True
            else:
                # Safeguard
                 logger.warning("Trade Action: Signal SHORT ignored (already short? Check logic).")

        else:
            # No entry signal this cycle
            if not action_taken_this_cycle: # Only log 'no signal' if no other action (like SL/TP close) happened
                if position_side == POSITION_SIDE_NONE:
                    logger.info("Trade Logic: No confirmed entry signal and no active position. Holding cash.")
                else:
                     logger.info(f"Trade Logic: Holding {position_side} position. No new entry or exit signal this cycle.")

    except Exception as e:
        # Catch-all for unexpected errors within the main logic block
        logger.error(f"Trade Logic: CRITICAL UNEXPECTED ERROR during cycle execution for {symbol}: {e}")
        logger.debug(traceback.format_exc())
        # Consider sending an SMS alert for critical logic failures
        send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!")
    finally:
        # Log end of cycle regardless of outcome
        logger.info(f"========== Cycle Check End: {symbol} ==========\n")


# --- Graceful Shutdown ---
def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str]) -> None:
    """
    Attempts to close any open position for the given symbol before exiting.
    Sends SMS alerts about the shutdown process.
    """
    logger.warning("Shutdown requested. Initiating graceful exit sequence...")
    market_base = symbol.split('/')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown requested. Attempting to close open position if any.")

    if not exchange or not symbol:
        logger.warning("Graceful Shutdown: Exchange not initialized or symbol not set. Cannot check or close position.")
        return

    try:
        logger.info(f"Graceful Shutdown: Checking for active position for {symbol}...")
        # Use the robust position check function
        position = get_current_position(exchange, symbol)

        if position and position.get('side') != POSITION_SIDE_NONE:
            pos_side = position['side']
            pos_qty = position['qty']
            logger.warning(f"Graceful Shutdown: Active {pos_side} position found ({pos_qty:.8f} {symbol}). Attempting to close...")
            # Pass the freshly checked 'position' dict to the close function
            close_result = close_position(exchange, symbol, position, reason="Shutdown")

            if close_result:
                logger.info("Graceful Shutdown: Close order placed. Waiting briefly for exchange processing...")
                # Wait a bit longer on shutdown to increase chance of confirmation
                time.sleep(POST_CLOSE_DELAY_SECONDS * 2)
                # --- Final Confirmation Check ---
                logger.info("Graceful Shutdown: Performing final position check...")
                final_pos_check = get_current_position(exchange, symbol)
                if final_pos_check and final_pos_check.get('side') == POSITION_SIDE_NONE:
                     logger.success("Graceful Shutdown: Position successfully confirmed closed.")
                     send_sms_alert(f"[{market_base}] Position confirmed CLOSED during shutdown.")
                else:
                     final_side = final_pos_check.get('side', 'N/A')
                     final_qty = final_pos_check.get('qty', 0.0)
                     logger.error(f"Graceful Shutdown: FAILED TO CONFIRM position closure after attempting close. Final check showed: {final_side}, Qty: {final_qty:.8f}")
                     send_sms_alert(f"[{market_base}] ERROR: Failed to confirm position closure on shutdown! Final state: {final_side} Qty={final_qty:.8f}. MANUAL CHECK REQUIRED.")
            else:
                 # close_position failed to even place the order
                 logger.error("Graceful Shutdown: Failed to place the close order for the active position. MANUAL INTERVENTION REQUIRED.")
                 send_sms_alert(f"[{market_base}] ERROR: Failed to PLACE close order during shutdown. MANUAL CHECK REQUIRED.")

        else:
            # No active position found during the check
            logger.info(f"Graceful Shutdown: No active position found for {symbol}. No close action needed.")
            send_sms_alert(f"[{market_base}] No active position found during shutdown.")

    except Exception as e:
        # Catch errors during the shutdown process itself (e.g., API errors during final checks)
        logger.error(f"Graceful Shutdown: An error occurred during the shutdown process: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] Error during graceful shutdown sequence: {type(e).__name__}. Check logs.")

    logger.info("--- Scalping Bot Shutdown Complete ---")


# --- Main Execution ---
def main() -> None:
    """
    Main function to initialize the exchange connection, set up parameters,
    and run the main trading loop. Handles initialization errors and graceful shutdown.
    """
    bot_start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"--- Bybit Scalping Bot Initializing v2.0.0 ({bot_start_time_str}) ---")
    logger.warning("--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---")
    logger.warning("--- ENSURE ALL PARAMETERS IN .env ARE CORRECTLY SET & TESTED ---")
    if ENABLE_LOOP_SLTP_CHECK:
        logger.critical("--- *** CRITICAL WARNING: LOOP-BASED SL/TP IS ACTIVE - UNSAFE & INACCURATE FOR LIVE SCALPING *** ---")
        logger.critical("--- *** YOU MUST IMPLEMENT EXCHANGE-NATIVE CONDITIONAL ORDERS FOR SAFE SL/TP MANAGEMENT *** ---")
    else:
        logger.info("--- INFO: Loop-based SL/TP check is DISABLED (Recommended for safety). Use exchange-native orders. ---")


    exchange: Optional[ccxt.Exchange] = None
    symbol: Optional[str] = None # Will hold the validated symbol
    run_bot: bool = True # Flag to control the main loop
    cycle_count: int = 0

    try:
        # === Initialize Exchange ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Bot exiting due to exchange initialization failure.")
            # SMS alert already sent by initialize_exchange on failure
            return # Stop execution

        # === Setup Trading Symbol and Leverage ===
        # Prompt user or use default, then validate
        try:
            raw_symbol_input = input(f"Enter trading symbol (e.g., BTC/USDT:USDT) or press Enter for default [{DEFAULT_SYMBOL}]: ").strip()
            symbol_to_validate = raw_symbol_input or DEFAULT_SYMBOL

            logger.info(f"Validating symbol: {symbol_to_validate}...")
            market = exchange.market(symbol_to_validate) # Throws BadSymbol if invalid
            symbol = market['symbol'] # Use the CCXT canonical symbol format

            # Verify it's a contract market (swap/future)
            if not market.get('contract', False) or market.get('spot', True): # Double check it's not spot
                logger.critical(f"Configuration Error: Symbol '{symbol}' is not a valid SWAP or FUTURES market on {exchange.id}. Type: {market.get('type')}")
                send_sms_alert(f"[ScalpBot] CRITICAL: Symbol {symbol} is not a contract market. Exiting.")
                return
            logger.info(f"Successfully validated symbol: {symbol} (Type: {market.get('type')})")

            # Set leverage for the validated symbol
            if not set_leverage(exchange, symbol, DEFAULT_LEVERAGE):
                logger.critical(f"Leverage setup failed for {symbol} after retries. Cannot proceed safely. Exiting.")
                # SMS alert potentially sent by set_leverage on final failure
                send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL: Leverage setup FAILED for {symbol}. Exiting.")
                return
            logger.info(f"Leverage confirmed/set to {DEFAULT_LEVERAGE}x for {symbol}.")

        except (ccxt.BadSymbol, KeyError) as e:
            logger.critical(f"Configuration Error: Invalid or unsupported symbol '{symbol_to_validate}'. Error: {e}")
            send_sms_alert(f"[ScalpBot] CRITICAL: Invalid symbol '{symbol_to_validate}'. Exiting.")
            return
        except Exception as e:
            logger.critical(f"Configuration Error: Unexpected error during symbol/leverage setup: {e}")
            logger.debug(traceback.format_exc())
            send_sms_alert("[ScalpBot] CRITICAL: Symbol/Leverage setup failed unexpectedly. Exiting.")
            return

        # === Log Configuration Summary ===
        logger.info("--- Scalping Bot Configuration Summary ---")
        logger.info(f"Trading Pair: {symbol}")
        logger.info(f"Timeframe: {DEFAULT_INTERVAL}")
        logger.info(f"Leverage: {DEFAULT_LEVERAGE}x")
        logger.info(f"Risk Per Trade: {RISK_PER_TRADE_PERCENTAGE:.3%}")
        logger.info(f"Max Position Value (Cap): {MAX_ORDER_USDT_AMOUNT:.2f} USDT")
        logger.info(f"SL Calculation: {ATR_STOP_LOSS_MULTIPLIER} * ATR({ATR_CALCULATION_PERIOD})")
        if ENABLE_LOOP_SLTP_CHECK:
            logger.warning(f"TP Calculation: {ATR_TAKE_PROFIT_MULTIPLIER} * ATR({ATR_CALCULATION_PERIOD}) (Loop Check - Unsafe)")
        else:
             logger.info("TP Calculation: Loop Check Disabled")
        logger.info(f"Primary Supertrend: Length={DEFAULT_ST_ATR_LENGTH}, Multiplier={DEFAULT_ST_MULTIPLIER}")
        logger.info(f"Confirmation Supertrend: Length={CONFIRM_ST_ATR_LENGTH}, Multiplier={CONFIRM_ST_MULTIPLIER}")
        logger.info(f"Volume Analysis: MA={VOLUME_MA_PERIOD}, Spike > {VOLUME_SPIKE_THRESHOLD}x MA (Required for Entry: {REQUIRE_VOLUME_SPIKE_FOR_ENTRY})")
        logger.info(f"Order Book Analysis: Depth={ORDER_BOOK_DEPTH}, Long Ratio >= {ORDER_BOOK_RATIO_THRESHOLD_LONG}, Short Ratio <= {ORDER_BOOK_RATIO_THRESHOLD_SHORT}")
        logger.info(f"Fetch Order Book Every Cycle: {FETCH_ORDER_BOOK_PER_CYCLE}")
        logger.info(f"Margin Check Buffer: {REQUIRED_MARGIN_BUFFER:.1%} (Requires {(REQUIRED_MARGIN_BUFFER-1)*100:.1f}% extra free margin)")
        logger.info(f"Check Interval (Sleep): {DEFAULT_SLEEP_SECONDS} seconds")
        logger.info(f"SMS Alerts Enabled: {ENABLE_SMS_ALERTS}, Recipient: {'Set' if SMS_RECIPIENT_NUMBER else 'Not Set'}, Timeout: {SMS_TIMEOUT_SECONDS}s")
        logger.warning("ACTION REQUIRED: Review this configuration carefully. Does it align with your strategy and risk tolerance?")
        if ENABLE_LOOP_SLTP_CHECK:
            logger.critical("RISK WARNING: Loop-based SL/TP check is ACTIVE, INACCURATE & UNSAFE for live scalping.")
        logger.info(f"Current Logging Level: {logging.getLevelName(logger.level)}")
        logger.info("------------------------------------------")
        # Send startup confirmation SMS
        market_base = symbol.split('/')[0]
        sl_tp_status = "Loop-based SL/TP ACTIVE (HIGH RISK)" if ENABLE_LOOP_SLTP_CHECK else "Loop SL/TP Disabled"
        send_sms_alert(f"[ScalpBot] Configured for {symbol} ({DEFAULT_INTERVAL}, {DEFAULT_LEVERAGE}x). {sl_tp_status}. Starting main loop.")

        # === Main Trading Loop ===
        while run_bot:
            cycle_start_time = time.monotonic() # Use monotonic clock for interval timing
            cycle_count += 1
            logger.debug(f"--- Cycle {cycle_count} Start: {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

            try:
                # 1. Fetch Fresh Market Data
                # Determine required data length based on longest indicator period + buffer
                data_limit = max(100, DEFAULT_ST_ATR_LENGTH*2, CONFIRM_ST_ATR_LENGTH*2,
                                 ATR_CALCULATION_PERIOD*2, VOLUME_MA_PERIOD*2) + API_FETCH_LIMIT_BUFFER
                df_market = get_market_data(exchange, symbol, DEFAULT_INTERVAL, limit=data_limit)

                # 2. Execute Trading Logic (only if data is valid)
                if df_market is not None and not df_market.empty:
                    # Pass a copy of the dataframe to prevent accidental modification across cycles
                    trade_logic(
                        exchange=exchange, symbol=symbol, df=df_market.copy(),
                        st_len=DEFAULT_ST_ATR_LENGTH, st_mult=DEFAULT_ST_MULTIPLIER,
                        cf_st_len=CONFIRM_ST_ATR_LENGTH, cf_st_mult=CONFIRM_ST_MULTIPLIER,
                        atr_len=ATR_CALCULATION_PERIOD, vol_ma_len=VOLUME_MA_PERIOD,
                        risk_pct=RISK_PER_TRADE_PERCENTAGE,
                        sl_atr_mult=ATR_STOP_LOSS_MULTIPLIER,
                        tp_atr_mult=ATR_TAKE_PROFIT_MULTIPLIER,
                        leverage=DEFAULT_LEVERAGE,
                        max_order_cap=MAX_ORDER_USDT_AMOUNT,
                        margin_buffer=REQUIRED_MARGIN_BUFFER
                    )
                else:
                    # Log if data fetching failed in this cycle
                    logger.warning(f"Main Loop (Cycle {cycle_count}): No valid market data received for {symbol}. Skipping trade logic for this cycle.")

            # --- Main Loop Exception Handling ---
            # Handle specific CCXT errors gracefully to allow continuation if possible
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"RATE LIMIT EXCEEDED (Cycle {cycle_count}): {e}. Sleeping for {DEFAULT_SLEEP_SECONDS * 5} seconds...")
                send_sms_alert(f"[{market_base}] WARNING: Rate limit exceeded! Check API usage/frequency.")
                time.sleep(DEFAULT_SLEEP_SECONDS * 5) # Longer sleep after rate limit hit
            except ccxt.NetworkError as e:
                # Typically recoverable, log and continue after normal delay
                logger.warning(f"Network error in main loop (Cycle {cycle_count}): {e}. Will retry next cycle.")
                # Let the standard sleep handle the delay
            except ccxt.ExchangeNotAvailable as e:
                # Exchange might be down for maintenance, sleep longer
                logger.error(f"Exchange not available (Cycle {cycle_count}): {e}. Sleeping for {DEFAULT_SLEEP_SECONDS * 10} seconds...")
                send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable! ({e})")
                time.sleep(DEFAULT_SLEEP_SECONDS * 10)
            except ccxt.AuthenticationError as e:
                # Critical: API keys likely invalid or expired. Stop the bot.
                logger.critical(f"Authentication Error encountered in main loop (Cycle {cycle_count}): {e}. API keys may be invalid or permissions revoked. Stopping bot.")
                send_sms_alert(f"[{market_base}] CRITICAL: Authentication Error! Bot stopping NOW.")
                run_bot = False # Stop the main loop permanently
            except ccxt.ExchangeError as e:
                # Catch other specific exchange errors (e.g., invalid order params, margin issues not caught earlier)
                logger.error(f"Unhandled Exchange error in main loop (Cycle {cycle_count}): {e}.")
                logger.debug(traceback.format_exc()) # Log traceback for details
                send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs.")
                # Continue after normal delay, hoping it's temporary
            except Exception as e:
                # Catch any other unexpected errors, log traceback, and stop the bot for safety
                logger.exception(f"!!! UNEXPECTED CRITICAL ERROR in main loop (Cycle {cycle_count}): {e} !!!") # Logs full traceback
                send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR in main loop: {type(e).__name__}! Bot stopping NOW.")
                logger.critical("Stopping bot due to unexpected critical error.")
                run_bot = False # Stop the loop

            # --- Cycle Delay ---
            # Calculate elapsed time and sleep for the remaining duration of the interval
            if run_bot: # Only sleep if the bot is still supposed to be running
                cycle_end_time = time.monotonic()
                elapsed_time = cycle_end_time - cycle_start_time
                sleep_duration = max(0, DEFAULT_SLEEP_SECONDS - elapsed_time)
                logger.debug(f"Cycle {cycle_count} execution time: {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s.")
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        # User pressed Ctrl+C
        logger.warning("Shutdown requested via KeyboardInterrupt (Ctrl+C).")
        run_bot = False # Ensure loop terminates if it was somehow still running

    finally:
        # --- Bot Shutdown Sequence ---
        # This block executes whether the loop finished normally,
        # was stopped by an error, or by KeyboardInterrupt.
        logger.info("--- Initiating Scalping Bot Shutdown Sequence ---")
        # Attempt graceful shutdown (close position) only if exchange/symbol were initialized
        graceful_shutdown(exchange, symbol)

        logger.info("--- Scalping Bot Shutdown Complete ---")
        market_base_final = symbol.split('/')[0] if symbol else "Bot"
        # Send final shutdown message
        send_sms_alert(f"[{market_base_final}] Bot process has terminated.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Print initial information to the console
    print("--- Bybit USDT Futures Scalping Bot v2.0.0 ---")
    print("INFO: Loading configuration from .env file.")
    print("INFO: Ensure API keys (BYBIT_API_KEY, BYBIT_API_SECRET) are set in .env.")
    print("INFO: For SMS alerts, set ENABLE_SMS_ALERTS=true and SMS_RECIPIENT_NUMBER in .env.")
    print("INFO: SMS requires Termux environment with Termux:API installed ('pkg install termux-api') and SMS permission granted.")
    print("\033[93mWARNING: LIVE TRADING IS ACTIVE BY DEFAULT. USE EXTREME CAUTION.\033[0m")
    if ENABLE_LOOP_SLTP_CHECK:
        print("\033[91m\033[1mCRITICAL: LOOP-BASED SL/TP IS ACTIVE - HIGHLY INACCURATE & UNSAFE FOR LIVE SCALPING.\033[0m")
        print("\033[91m\033[1mCRITICAL: IMPLEMENT EXCHANGE-NATIVE CONDITIONAL ORDERS FOR SAFE OPERATION.\033[0m")
    else:
        print("INFO: Loop-based SL/TP check is DISABLED by default (recommended).")
    print(f"INFO: Initial logging level set to: {logging.getLevelName(LOGGING_LEVEL)}")
    print("-" * 40)
    # Start the main execution function
    main()
