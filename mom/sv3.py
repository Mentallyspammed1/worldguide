
2025-05-02 13:37:48 [INFO] Attempting to send SMS alert to 16364866381 (Timeout: 30s)...                                        2025-05-02 13:37:50 [SUCCESS] Termux API command for SMS to 16364866381 executed successfully.                                  2025-05-02 13:37:51 [INFO] ========== New Check Cycle: FARTCOIN/USDT:USDT | Candle Time: 2025-05-02 18:37:00 UTC ==========     2025-05-02 13:37:52 [ERROR] Order Book: Unexpected error analyzing order book for FARTCOIN/USDT:USDT: Invalid format specifier '.0.0001f' for object of type 'float'
2025-05-02 13:37:52 [INFO] Position Check: No active One-Way Mode position found for FARTCOINUSDT after checking 0 entries.     2025-05-02 13:37:52 [ERROR] Trade Logic: CRITICAL UNEXPECTED ERROR during cycle execution for FARTCOIN/USDT:USDT: Invalid format specifier '.0.0001f' for object of type 'float'
2025-05-02 13:37:52 [INFO] Attempting to send SMS alert to 16364866381 (Timeout: 30s)...                                        2025-05-02 13:37:55 [SUCCESS] Termux API command for SMS to 16364866381 executed successfully.                                  2025-05-02 13:37:55 [INFO] ========== Cycle Check End: FARTCOIN/USDT:USDT =========
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.0.1 (Enhanced Safety, Configurable Loop SL/TP, Improved Robustness, V5 API Focus)

Features:
- Dual Supertrend strategy with confirmation.
- ATR for volatility measurement and Stop-Loss/Take-Profit calculation.
- Volume spike analysis (optional confirmation).
- Order book pressure analysis (optional confirmation, optimized fetching).
- Risk-based position sizing with margin checks (Equity/Free Margin).
- Termux SMS alerts for critical events (startup, errors, trades, shutdown).
- Robust error handling (specific CCXT exceptions) and detailed logging with color support.
- Graceful shutdown on KeyboardInterrupt (Ctrl+C) and SIGTERM with position closing attempt and confirmation.
- Stricter position detection logic targeting Bybit V5 API structure via CCXT (checks unified fields first, falls back to 'info', requires 'positionIdx=0').
- User symbol input with validation (must be a linear contract market).
- Leverage setting with validation and retries (explicit V5 params).
- Improved entry price estimation using shallow order book fetch.
- Explicit market type check (linear contract) before setting leverage.
- Pre-close position re-validation to mitigate race conditions.
- Post-reversal position closure confirmation before entering the new trade.
- **Configurable Loop-Based SL/TP Check (Default: OFF due to EXTREME safety concerns).**

Disclaimer:
- **EXTREME RISK**: This software is provided for educational purposes ONLY. Scalping bots are
  inherently high-risk due to market noise, latency, slippage, exchange fees, API limitations,
  and the critical need for precise, low-latency execution. Market conditions can change rapidly.
  **USE AT YOUR OWN ABSOLUTE RISK. THE AUTHORS ASSUME NO LIABILITY FOR ANY FINANCIAL LOSSES.**
- **CRITICAL SL/TP LIMITATION (LOOP-BASED CHECK)**: The optional loop-based
  Stop-Loss/Take-Profit mechanism (`ENABLE_LOOP_SLTP_CHECK = True`) checks position status
  *after* the candle closes based on the *closing price*. This is **EXTREMELY DANGEROUS,
  GROSSLY INACCURATE, AND COMPLETELY UNSUITABLE** for live scalping or any latency-sensitive strategy.
  Price can move significantly beyond calculated SL/TP levels *before* the bot reacts in the *next*
  cycle, leading to potentially massive losses or missed profits. For real scalping, you **MUST**
  implement exchange-native conditional orders (e.g., stopMarket, takeProfitMarket using
  `exchange.create_order` with appropriate `stopLossPrice` / `takeProfitPrice` parameters,
  or Bybit's V5 `setTradingStop` API endpoint) placed *immediately* after entry confirmation,
  ideally based on the actual fill price. **This loop check is DISABLED BY DEFAULT for safety.**
- Parameter Sensitivity: Bot performance is extremely sensitive to parameter tuning (Supertrend lengths/multipliers,
  ATR settings, risk percentage, confirmation thresholds, etc.). Requires significant backtesting,
  forward testing on a testnet, and careful optimization for the specific market and timeframe.
  Defaults provided are illustrative ONLY and likely suboptimal.
- API Rate Limits: Frequent API calls (data fetching, order placement, position checks) can hit
  exchange rate limits, leading to temporary blocks or errors. Monitor usage and adjust `SLEEP_SECONDS`
  or logic frequency if needed. CCXT's built-in rate limiter helps but may not cover all scenarios.
- Slippage: Market orders used for entry/exit are prone to slippage, especially during volatile
  periods or on less liquid pairs. This can significantly impact profitability.
- Test Thoroughly: **DO NOT RUN WITH REAL MONEY WITHOUT EXTENSIVE AND SUCCESSFUL TESTNET/DEMO TESTING.**
  Verify all components (indicators, order logic, risk management, error handling, shutdown) function
  as expected under various simulated market conditions.
- Termux Dependency: SMS alerts require running within Termux on Android with the Termux:API package
  installed (`pkg install termux-api`) and the companion Termux:API app installed and configured
  with SMS permissions granted in Android settings. Functionality depends on Termux and Android specifics.
- API Changes & Compatibility: Exchange APIs (like Bybit V3 vs V5) can change without notice. This code
  targets structures observed in Bybit's V5 API via CCXT as of the last update. Future API changes may
  require code modifications. Ensure your CCXT library is up-to-date. Bot assumes "One-Way" position mode on Bybit.
"""

# Standard Library Imports
import logging
import os
import signal
import sys
import time
import traceback
import subprocess  # For Termux API calls
from typing import Dict, Optional, Any, Tuple, List

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required library. Please install requirements: {e}")
    print("Try: pip install ccxt pandas pandas_ta python-dotenv")
    sys.exit(1)


# --- Configuration Loading ---
load_dotenv()  # Load variables from .env file into environment variables
BOT_VERSION = "2.0.1"

# --- Constants ---

# --- API Credentials (Required in .env) ---
BYBIT_API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")

# --- Trading Parameters (Set Defaults, Override in .env) ---
DEFAULT_SYMBOL: str = os.getenv("SYMBOL", "BTC/USDT:USDT")  # CCXT unified symbol format (e.g., XBT/USDT:USDT, ETH/USDT:USDT)
DEFAULT_INTERVAL: str = os.getenv("INTERVAL", "1m")  # Timeframe (e.g., '1m', '3m', '5m') - Ensure it's supported by Bybit
DEFAULT_LEVERAGE: int = int(os.getenv("LEVERAGE", 10))  # Desired leverage (Ensure it's allowed by Bybit for the symbol)
DEFAULT_SLEEP_SECONDS: int = int(os.getenv("SLEEP_SECONDS", 10))  # Pause between main logic cycles (adjust based on interval and rate limits)

# --- Risk Management (CRITICAL - TUNE CAREFULLY!) ---
RISK_PER_TRADE_PERCENTAGE: float = float(os.getenv("RISK_PER_TRADE_PERCENTAGE", 0.005))  # e.g., 0.005 = 0.5% risk per trade based on account equity
ATR_STOP_LOSS_MULTIPLIER: float = float(os.getenv("ATR_STOP_LOSS_MULTIPLIER", 1.5))  # Multiplier for ATR to set SL distance from entry
ATR_TAKE_PROFIT_MULTIPLIER: float = float(os.getenv("ATR_TAKE_PROFIT_MULTIPLIER", 2.0))  # Multiplier for ATR to set TP distance from entry (Only used if ENABLE_LOOP_SLTP_CHECK=True)
MAX_ORDER_USDT_AMOUNT: float = float(os.getenv("MAX_ORDER_USDT_AMOUNT", 500.0))  # Maximum position size in USDT value (acts as a cap)
REQUIRED_MARGIN_BUFFER: float = float(os.getenv("REQUIRED_MARGIN_BUFFER", 1.10))  # Margin check buffer (e.g., 1.10 = requires 10% extra free margin than estimated)

# --- Safety Feature (Loop-Based SL/TP - HIGHLY DISCOURAGED / UNSAFE) ---
# *** CRITICAL WARNING: Setting this to True enables a DANGEROUS and INACCURATE SL/TP mechanism ***
# *** suitable ONLY for basic testing or strategies NOT requiring precise/timely exits. ***
# *** For any real trading, especially scalping, KEEP THIS FALSE and implement ***
# *** exchange-native conditional orders immediately after entry. ***
ENABLE_LOOP_SLTP_CHECK: bool = os.getenv("ENABLE_LOOP_SLTP_CHECK", "false").lower() == "true" # Default to False for safety

# --- Supertrend Indicator Parameters ---
DEFAULT_ST_ATR_LENGTH: int = int(os.getenv("ST_ATR_LENGTH", 7))  # Primary Supertrend ATR period
DEFAULT_ST_MULTIPLIER: float = float(os.getenv("ST_MULTIPLIER", 2.5))  # Primary Supertrend ATR multiplier
CONFIRM_ST_ATR_LENGTH: int = int(os.getenv("CONFIRM_ST_ATR_LENGTH", 5))  # Confirmation Supertrend ATR period
CONFIRM_ST_MULTIPLIER: float = float(os.getenv("CONFIRM_ST_MULTIPLIER", 2.0))  # Confirmation Supertrend ATR multiplier

# --- Volume Analysis Parameters ---
VOLUME_MA_PERIOD: int = int(os.getenv("VOLUME_MA_PERIOD", 20))  # Moving average period for volume
VOLUME_SPIKE_THRESHOLD: float = float(os.getenv("VOLUME_SPIKE_THRESHOLD", 1.5))  # Volume must be > this * MA for a spike
REQUIRE_VOLUME_SPIKE_FOR_ENTRY: bool = os.getenv("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false").lower() == "true"  # Require volume spike for entry signal (Default: False)

# --- Order Book Analysis Parameters ---
ORDER_BOOK_DEPTH: int = int(os.getenv("ORDER_BOOK_DEPTH", 10))  # Levels of bids/asks to analyze for ratio
ORDER_BOOK_RATIO_THRESHOLD_LONG: float = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_LONG", 1.2))  # Bid/Ask volume ratio must be >= this for long confirmation
ORDER_BOOK_RATIO_THRESHOLD_SHORT: float = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_SHORT", 0.8))  # Bid/Ask volume ratio must be <= this for short confirmation
FETCH_ORDER_BOOK_PER_CYCLE: bool = os.getenv("FETCH_ORDER_BOOK_PER_CYCLE", "false").lower() == "true"  # Fetch OB every cycle (API intensive) or only on potential signals

# --- ATR Calculation Parameter (for SL/TP distance calculation) ---
ATR_CALCULATION_PERIOD: int = int(os.getenv("ATR_CALCULATION_PERIOD", 10))  # ATR period used specifically for SL/TP calculation

# --- Termux SMS Alert Configuration (Set in .env) ---
ENABLE_SMS_ALERTS: bool = os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"  # Master switch for SMS alerts
SMS_RECIPIENT_NUMBER: Optional[str] = os.getenv("SMS_RECIPIENT_NUMBER")  # Phone number to send alerts to
SMS_TIMEOUT_SECONDS: int = int(os.getenv("SMS_TIMEOUT_SECONDS", 30))  # Timeout for the termux-sms-send command

# --- CCXT / API Parameters ---
DEFAULT_RECV_WINDOW: int = 15000  # API receive window (milliseconds), increase if timestamp errors occur
ORDER_BOOK_FETCH_LIMIT: int = max(25, ORDER_BOOK_DEPTH)  # How many OB levels to fetch (API often requires min 25/50/100)
SHALLOW_OB_FETCH_DEPTH: int = 5  # Depth for quick best bid/ask fetch for entry price estimate

# --- Internal Constants ---
SIDE_BUY: str = "buy"
SIDE_SELL: str = "sell"
POSITION_SIDE_LONG: str = "Long"  # Internal representation for long position
POSITION_SIDE_SHORT: str = "Short"  # Internal representation for short position
POSITION_SIDE_NONE: str = "None"  # Internal representation for no position
# Map CCXT unified position side strings to internal representation
POSITION_SIDE_MAP: Dict[str, str] = {"long": POSITION_SIDE_LONG, "short": POSITION_SIDE_SHORT}
USDT_SYMBOL: str = "USDT"  # Base currency symbol for balance checks (ensure matches your account's collateral)
RETRY_COUNT: int = 3  # Number of retries for critical API calls (e.g., leverage setting)
RETRY_DELAY_SECONDS: int = 2  # Delay between retries
API_FETCH_LIMIT_BUFFER: int = 10 # Fetch slightly more OHLCV data than strictly needed for indicator calculation buffer
FLOAT_COMPARISON_EPSILON: float = 1e-9  # Small value for floating point comparisons (e.g., checking if quantity is effectively zero)
POST_CLOSE_DELAY_SECONDS: int = 3  # Brief delay after closing a position before potentially opening a new one (allows exchange state to update)


# --- Logger Setup ---
# ---> SET TO DEBUG FOR DETAILED TROUBLESHOOTING <---
# ---> REMEMBER TO SET BACK TO INFO FOR NORMAL OPERATION <---
LOGGING_LEVEL_STR: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL: int = getattr(logging, LOGGING_LEVEL_STR, logging.INFO) # Default to INFO if invalid level in env

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s", # Simpler format
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
logging.Logger.success = log_success # type: ignore[attr-defined]

# ANSI escape codes for colored logging output (adjust based on terminal support)
# Check if the output stream is a TTY (terminal) before adding colors
if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"\033[90m{logging.getLevelName(logging.DEBUG)}\033[0m")      # Gray
    logging.addLevelName(logging.INFO, f"\033[96m{logging.getLevelName(logging.INFO)}\033[0m")        # Cyan
    logging.addLevelName(SUCCESS_LEVEL, f"\033[92m{logging.getLevelName(SUCCESS_LEVEL)}\033[0m")    # Green
    logging.addLevelName(logging.WARNING, f"\033[93m{logging.getLevelName(logging.WARNING)}\033[0m")  # Yellow
    logging.addLevelName(logging.ERROR, f"\033[91m{logging.getLevelName(logging.ERROR)}\033[0m")      # Red
    logging.addLevelName(logging.CRITICAL, f"\033[91m\033[1m{logging.getLevelName(logging.CRITICAL)}\033[0m") # Bold Red


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
        command: List[str] = ['termux-sms-send', '-n', SMS_RECIPIENT_NUMBER, message]

        logger.info(f"Attempting to send SMS alert to {SMS_RECIPIENT_NUMBER} (Timeout: {SMS_TIMEOUT_SECONDS}s)...")
        logger.debug(f"Executing command: {' '.join(command)}") # Log the command for debugging

        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout/stderr
            text=True,            # Decode stdout/stderr as text
            check=False,          # Do not raise exception on non-zero exit, check manually
            timeout=SMS_TIMEOUT_SECONDS, # Set a timeout for the command
        )

        if result.returncode == 0:
            logger.success(f"Termux API command for SMS to {SMS_RECIPIENT_NUMBER} executed successfully.")
            return True
        else:
            # Log detailed error information if the command failed
            logger.error(f"Termux API command 'termux-sms-send' failed. Return code: {result.returncode}")
            stderr_msg = result.stderr.strip() if result.stderr else "N/A"
            stdout_msg = result.stdout.strip() if result.stdout else "N/A"
            logger.error(f"Termux API Stderr: {stderr_msg}")
            logger.error(f"Termux API Stdout: {stdout_msg}") # Sometimes error info might be on stdout
            logger.error("Potential issues: SMS permissions not granted in Android settings for Termux:API, "
                         "Termux:API app not running or installed, incorrect recipient number format, "
                         "Android battery optimization interfering with Termux, network issues.")
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
        logger.debug(traceback.format_exc()) # Log full traceback for unexpected errors
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
                    # Bybit V5 requires 'category' usually passed in methods,
                    # but setting 'defaultSubType' might help for some unified calls.
                    'defaultSubType': 'linear', # Specify 'linear' for USDT Perpetual
                    'recvWindow': DEFAULT_RECV_WINDOW,
                    # 'adjustForTimeDifference': True, # Consider enabling if clock skew issues arise frequently
                    # 'verbose': True, # Uncomment for extremely detailed API request/response logging (DEBUG ONLY)
                },
            }
        )
        # Test connection and authentication by fetching markets and balance
        logger.debug("Loading markets...")
        exchange.load_markets()
        logger.debug("Fetching balance (tests authentication for linear category)...")
        # Bybit V5 requires 'category' for balance fetch - specifically test linear as it's required
        exchange.fetch_balance(params={'category': 'linear'}) # Throws AuthenticationError on bad keys
        logger.success("CCXT Bybit Session Initialized (LIVE FUTURES SCALPING MODE - EXTREME CAUTION!).")
        send_sms_alert("[ScalpBot] Initialized successfully and authenticated.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}. Check API key/secret, ensure IP whitelist (if used) is correct, and API permissions (read/trade) are sufficient for contract trading.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error during initialization: {e}. Check internet connection, DNS, firewall, and Bybit status page.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Bot stopped.")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error during initialization: {e}. Check Bybit status page or API documentation for details. Could be temporary maintenance or API issue.")
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
                      f'{prefix}supertrend' (the Supertrend line value),
                      f'{prefix}trend' (boolean, True if trend is up),
                      f'{prefix}st_long' (boolean, True on the candle the trend flipped UP),
                      f'{prefix}st_short' (boolean, True on the candle the trend flipped DOWN).
                      Returns original DataFrame with NA columns if calculation fails or input is invalid.
    """
    col_prefix = f"{prefix}" if prefix else ""
    # Define the target column names we want to keep/create
    target_cols = [
        f"{col_prefix}supertrend",
        f"{col_prefix}trend",
        f"{col_prefix}st_long",
        f"{col_prefix}st_short",
    ]
    # Define the raw column names expected from pandas_ta based on its naming convention
    # Note: pandas_ta naming might vary slightly with versions, verify if needed
    st_col_name = f"SUPERT_{length}_{multiplier}"  # Supertrend line value
    st_trend_col = f"SUPERTd_{length}_{multiplier}" # Trend direction (-1 down, 1 up)
    st_long_band_col = f"SUPERTl_{length}_{multiplier}"  # Lower band value (used when trend is up)
    st_short_band_col = f"SUPERTs_{length}_{multiplier}" # Upper band value (used when trend is down)

    # Input validation
    required_input_cols = ["high", "low", "close"]
    if df is None or df.empty or not all(c in df.columns for c in required_input_cols):
        logger.warning(f"Indicator Calc ({col_prefix}Supertrend): Input DataFrame is missing required columns {required_input_cols} or is empty.")
        for col in target_cols: df[col] = pd.NA # Add NA columns if they don't exist
        return df
    # Need enough data for the calculation itself (length) plus one previous candle for the shift
    if len(df) < length + 1:
        logger.warning(f"Indicator Calc ({col_prefix}Supertrend): DataFrame length ({len(df)}) is less than ST period+1 ({length+1}).")
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
        df[f"{col_prefix}trend"] = (df[st_trend_col] == 1)

        # 3. Calculate flip signals: Signal is True *only* on the candle where the flip occurs
        # Ensure the trend column doesn't contain NaNs before shifting/comparing
        # Forward fill any NaNs in raw trend - crucial before shifting
        df[st_trend_col] = df[st_trend_col].ffill()
        prev_trend_direction = df[st_trend_col].shift(1)
        # Long signal: Previous was down (-1) and current is up (1)
        df[f"{col_prefix}st_long"] = (prev_trend_direction == -1) & (df[st_trend_col] == 1)
        # Short signal: Previous was up (1) and current is down (-1)
        df[f"{col_prefix}st_short"] = (prev_trend_direction == 1) & (df[st_trend_col] == -1)

        # --- Clean up intermediate pandas_ta columns ---
        # Identify all columns starting with 'SUPERT_' that are NOT our target columns
        cols_to_drop = [c for c in df.columns if c.startswith("SUPERT_") and c not in target_cols]
        # Also explicitly add the raw trend direction and band value columns if they exist, to be safe
        cols_to_drop.extend([st_trend_col, st_long_band_col, st_short_band_col])
        # Use set to avoid duplicates if target_cols somehow overlapped with raw names, drop safely
        df.drop(columns=list(set(cols_to_drop)), errors='ignore', inplace=True)

        # Log the result of the last candle if data is valid
        if pd.notna(df[f'{col_prefix}trend'].iloc[-1]) and pd.notna(df[f'{col_prefix}supertrend'].iloc[-1]):
            last_trend_str = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
            last_supertrend_val = df[f'{col_prefix}supertrend'].iloc[-1]
            # Assume reasonable precision for logging, adjust if needed
            logger.debug(f"Indicator Calc: Calculated {col_prefix}Supertrend({length}, {multiplier}). "
                         f"Last trend: {last_trend_str}, Last value: {last_supertrend_val:.4f}")
        else:
            logger.warning(f"Indicator Calc ({col_prefix}Supertrend): Last row contains NaN values after calculation.")


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
    min_len = max(atr_len, vol_ma_len) + 1 # Need enough data for calc + potential lookback
    if len(df) < min_len:
         logger.warning(f"Indicator Calc (Vol/ATR): DataFrame length ({len(df)}) is less than required ({min_len}) for ATR({atr_len})/VolMA({vol_ma_len}).")
         return results

    try:
        # Calculate ATR using pandas_ta
        atr_col = f"ATRr_{atr_len}" # pandas_ta default name for ATR
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]):
            # Ensure ATR is positive
            last_atr = df[atr_col].iloc[-1]
            results["atr"] = last_atr if last_atr > FLOAT_COMPARISON_EPSILON else None
            if results["atr"] is None:
                logger.warning(f"Indicator Calc: Calculated ATR({atr_len}) is zero or negative ({last_atr}). Setting ATR to None.")
        else:
            logger.warning(f"Indicator Calc: Failed to calculate valid ATR({atr_len}). Check input data.")
        # Clean up ATR column
        if atr_col in df.columns:
            df.drop(columns=[atr_col], errors='ignore', inplace=True)

        # Calculate Volume Moving Average
        volume_ma_col = 'volume_ma'
        # Use min_periods to get MA sooner, but it might be less stable initially
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()

        if pd.notna(df[volume_ma_col].iloc[-1]):
            results["volume_ma"] = df[volume_ma_col].iloc[-1]
            results["last_volume"] = df['volume'].iloc[-1] # Get the volume of the last candle

            # Calculate Volume Ratio (Last Volume / Volume MA)
            # Check if MA is not None and significantly greater than zero before dividing
            if results["volume_ma"] is not None and results["volume_ma"] > FLOAT_COMPARISON_EPSILON and results["last_volume"] is not None:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            else:
                results["volume_ratio"] = None # Avoid division by zero or near-zero MA
                logger.debug(f"Indicator Calc: Volume ratio calculation skipped (LastVol={results['last_volume']}, VolMA={results['volume_ma']})")
        else:
             logger.warning(f"Indicator Calc: Failed to calculate valid Volume MA({vol_ma_len}). Check input data.")
        # Clean up Volume MA column
        if volume_ma_col in df.columns:
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

    bids: Optional[List[List[float]]] = None # Initialize for error logging
    asks: Optional[List[List[float]]] = None

    try:
        # Fetch the order book data
        # Bybit V5 requires 'category' param for linear contracts
        params = {}
        market = exchange.market(symbol)
        if market.get('linear'):
            params = {'category': 'linear'}
        elif market.get('inverse'):
            params = {'category': 'inverse'} # Add inverse just in case, though bot focuses on linear
        else:
            logger.warning(f"Order Book: Could not determine market category (linear/inverse) for {symbol}. Fetching without category param.")

        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit, params=params)

        # Validate the structure of the returned order book data
        if not order_book or not isinstance(order_book.get('bids'), list) or not isinstance(order_book.get('asks'), list):
            logger.warning(f"Order Book: Incomplete or invalid data structure received for {symbol}. OB: {order_book}")
            return results

        bids = order_book['bids']  # List of [price, amount]
        asks = order_book['asks']  # List of [price, amount]

        # Check if bids/asks lists are non-empty before accessing index 0
        if not bids or not asks:
             logger.warning(f"Order Book: Received empty bids or asks list for {symbol}. Bids: {len(bids)}, Asks: {len(asks)}")
             return results

        # Get best bid/ask and calculate spread
        # Add extra checks for list element structure and validity
        best_bid = bids[0][0] if len(bids[0]) > 0 and isinstance(bids[0][0], (int, float)) else 0.0
        best_ask = asks[0][0] if len(asks[0]) > 0 and isinstance(asks[0][0], (int, float)) else 0.0
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Get market precision for logging
        price_precision_digits = market.get('precision', {}).get('price', 4)
        qty_precision_digits = market.get('precision', {}).get('amount', 8)

        spread = None
        if best_bid > FLOAT_COMPARISON_EPSILON and best_ask > FLOAT_COMPARISON_EPSILON:
            spread = best_ask - best_bid
            results["spread"] = spread
            logger.debug(f"Order Book: Best Bid={best_bid:.{price_precision_digits}f}, Best Ask={best_ask:.{price_precision_digits}f}, Spread={spread:.{price_precision_digits}f}")
        else:
            # Log even if spread calculation isn't possible
            logger.debug(f"Order Book: Best Bid={best_bid:.{price_precision_digits}f}, Best Ask={best_ask:.{price_precision_digits}f} (Spread requires both > 0)")

        # Calculate cumulative volume within the specified analysis depth
        # Sum the 'amount' (index 1) for the top 'depth' levels, adding type checks
        bid_volume_sum = sum(bid[1] for bid in bids[:depth] if len(bid) > 1 and isinstance(bid[1], (int, float)))
        ask_volume_sum = sum(ask[1] for ask in asks[:depth] if len(ask) > 1 and isinstance(ask[1], (int, float)))
        logger.debug(f"Order Book: Analysis (Depth {depth}): Total Bid Vol={bid_volume_sum:.{qty_precision_digits}f}, Total Ask Vol={ask_volume_sum:.{qty_precision_digits}f}")

        # Calculate Bid/Ask Ratio (Total Bid Volume / Total Ask Volume)
        bid_ask_ratio = None
        # Check if ask volume is significantly greater than zero to avoid division issues
        if ask_volume_sum > FLOAT_COMPARISON_EPSILON:
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
        # More specific logging for index error
        logger.warning(f"Order Book: Index error processing bids/asks for {symbol}. Data might be malformed, empty, or have unexpected structure. Bids: {bids[:2] if bids else 'N/A'}, Asks: {asks[:2] if asks else 'N/A'}")
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
                                 or None if fetching fails or data is invalid/insufficient.
    """
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.")
        return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} (Timeframe: {interval})...")
        # Bybit V5 requires 'category' param for linear contracts
        params = {}
        market = exchange.market(symbol)
        if market.get('linear'):
            params = {'category': 'linear'}
        elif market.get('inverse'):
            params = {'category': 'inverse'} # Add inverse just in case
        else:
            logger.warning(f"Data Fetch: Could not determine market category (linear/inverse) for {symbol}. Fetching without category param.")

        ohlcv: List[List[float]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit, params=params)

        if not ohlcv or len(ohlcv) < limit // 2: # Check if we got a reasonable amount of data
            logger.warning(f"Data Fetch: Insufficient or no OHLCV data returned from exchange for {symbol} ({interval}). Received {len(ohlcv) if ohlcv else 0} candles.")
            return None

        # Convert to pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Data Type Conversion and Validation
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            # Convert OHLCV columns to numeric, coercing errors to NaN
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as conversion_err:
             logger.error(f"Data Fetch: Error converting fetched data types for {symbol}: {conversion_err}")
             return None

        df.set_index("timestamp", inplace=True)

        # Basic data validation: Check for NaN values introduced by coercion or gaps
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"Data Fetch: Fetched OHLCV data contains NaN values. Counts:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...")
            # Simple imputation: Forward fill NaNs. More sophisticated methods could be used.
            df.ffill(inplace=True)
            # Check again after filling - if NaNs remain (e.g., at the very beginning), data is unusable
            if df.isnull().values.any():
                 logger.error("Data Fetch: NaN values remain after forward fill, especially at the start. Cannot proceed with this data batch.")
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


# --- Position & Order Management (Revised for Bybit V5 & Robustness) ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches the current position details for a given symbol, specifically tailored
    for Bybit V5 API structure via CCXT. Assumes One-Way mode.
    Prioritizes unified CCXT fields, falls back to 'info' if needed.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol in CCXT unified format (e.g., 'BTC/USDT:USDT').

    Returns:
        Dict[str, Any]: {'side': POSITION_SIDE_*, 'qty': float, 'entry_price': float}.
                        Defaults to side=NONE, qty=0.0, entry_price=0.0 if no *actual*
                        active position (size > epsilon, side is 'long'/'short', positionIdx=0)
                        exists for the specified symbol. Returns default on any error.
    """
    # Default return value indicating no active position
    default_pos: Dict[str, Any] = {'side': POSITION_SIDE_NONE, 'qty': 0.0, 'entry_price': 0.0}
    market = None
    market_id = None

    # Get the exchange-specific market ID for comparison and market type
    try:
        market = exchange.market(symbol)
        market_id = market['id']  # The ID used by the exchange (e.g., 'BTCUSDT')
        logger.debug(f"Position Check: Fetching position for CCXT symbol '{symbol}' (Target Exchange Market ID: '{market_id}')...")
    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(f"Position Check: Failed to get market info/ID for '{symbol}': {e}")
        return default_pos
    except Exception as e:  # Catch other potential errors during market lookup
        logger.error(f"Position Check: Unexpected error getting market info for '{symbol}': {e}")
        logger.debug(traceback.format_exc())
        return default_pos

    try:
        # Check if the exchange supports fetching positions
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"Position Check: Exchange '{exchange.id}' does not support fetchPositions. Cannot reliably get position status.")
            return default_pos

        # Fetch positions - Bybit V5 requires 'category' based on market type
        params = {}
        if market.get('linear', False):
            params = {'category': 'linear'}
        elif market.get('inverse', False):
             params = {'category': 'inverse'}
        else:
             logger.warning(f"Position Check: Market type for {symbol} is neither linear nor inverse? Type: {market.get('type')}. Fetching without category param.")
             # Proceeding without category might work for some exchanges or default, but could fail on Bybit V5.

        logger.debug(f"Position Check: Calling fetchPositions(symbols=None, params={params})")
        # Fetch all positions for the category and filter locally for reliability
        all_positions: List[Dict] = exchange.fetch_positions(symbols=None, params=params)
        logger.debug(f"Position Check: Raw data from fetch_positions (Count: {len(all_positions)})")

        # Iterate through all returned positions and filter for the target symbol and mode
        found_position = None
        for pos in all_positions:
            # logger.debug(f"Position Check: Evaluating raw position entry: {pos}") # Uncomment for extreme debugging

            # --- 1. Symbol Check (using raw market ID from 'info') ---
            # CCXT's unified 'symbol' might not always match perfectly, raw 'symbol' in 'info' is safer for Bybit V5 filtering
            pos_info = pos.get('info', {})
            pos_symbol_raw = pos_info.get('symbol')
            if pos_symbol_raw != market_id:
                continue # Skip positions for other symbols

            # --- 2. Position Mode Check (Assume One-Way via positionIdx=0) ---
            # Bybit V5 uses `positionIdx`: 0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge
            position_idx_raw = pos_info.get('positionIdx', -1) # Default to -1 if not present
            try:
                position_idx = int(position_idx_raw)
            except (ValueError, TypeError):
                 logger.warning(f"Position Check: Could not parse positionIdx '{position_idx_raw}' for {market_id}. Skipping entry.")
                 continue

            if position_idx != 0:  # Strict check for One-Way mode
                logger.debug(f"Skipping position entry for {market_id}: Position Index is {position_idx} (Expected 0 for One-Way Mode). Bot only supports One-Way.")
                continue

            # --- 3. Determine Position Side ---
            # Prioritize CCXT unified 'side' field ('long', 'short', None)
            unified_side = pos.get('side') # Expected: 'long', 'short', or None/empty string
            determined_side = POSITION_SIDE_MAP.get(unified_side, POSITION_SIDE_NONE) if unified_side else POSITION_SIDE_NONE

            # Fallback to Bybit V5 'info.side' ('Buy', 'Sell', 'None') if unified side is unclear
            if determined_side == POSITION_SIDE_NONE:
                pos_side_v5 = pos_info.get('side', 'None')
                if pos_side_v5 == 'Buy':
                    determined_side = POSITION_SIDE_LONG
                elif pos_side_v5 == 'Sell':
                    determined_side = POSITION_SIDE_SHORT
                logger.debug(f"Position Check: Unified side was '{unified_side}', using info.side '{pos_side_v5}' -> {determined_side}")

            # If side is still None after checks, it's not an active directional position
            if determined_side == POSITION_SIDE_NONE:
                logger.debug(f"Skipping position entry for {market_id}: Determined side is None (Unified: '{unified_side}', V5 Info: '{pos_info.get('side')}').")
                continue

            # --- 4. Determine Position Quantity ---
            # Prioritize CCXT unified 'contracts' field (absolute quantity)
            quantity_unified = pos.get('contracts')
            quantity = 0.0

            if quantity_unified is not None:
                try:
                    quantity = float(quantity_unified)
                except (ValueError, TypeError):
                    logger.warning(f"Position Check: Could not parse unified quantity 'contracts': '{quantity_unified}' for {market_id}. Trying fallback.")
                    quantity_unified = None # Force fallback

            # Fallback to Bybit V5 'info.size' if unified quantity is missing/invalid
            if quantity_unified is None:
                size_str_v5 = pos_info.get('size')
                if size_str_v5 is not None and size_str_v5 != "":
                    try:
                        quantity = float(size_str_v5)
                        logger.debug(f"Position Check: Unified quantity missing/invalid, using info.size '{size_str_v5}' -> {quantity}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Position Check: Error parsing fallback quantity 'info.size': '{size_str_v5}' for {market_id}. Error: {e}. Treating as zero.")
                        quantity = 0.0
                else:
                    logger.warning(f"Position Check: Could not find valid quantity in unified 'contracts' or fallback 'info.size' for {market_id}. Treating as zero.")
                    quantity = 0.0

            # --- 5. Final Check: Is the quantity significantly non-zero? ---
            # Use absolute value as quantity should always be positive here after parsing
            if abs(quantity) > FLOAT_COMPARISON_EPSILON:
                # We have found an active One-Way position for our symbol!
                # Get Entry Price: Prioritize unified 'entryPrice'
                entry_price_unified = pos.get('entryPrice')
                entry_price = 0.0

                if entry_price_unified is not None:
                    try:
                        entry_price = float(entry_price_unified)
                        if entry_price <= FLOAT_COMPARISON_EPSILON: entry_price = 0.0 # Treat invalid price as 0
                    except (ValueError, TypeError):
                        logger.warning(f"Position Check: Could not parse unified entry price '{entry_price_unified}'. Trying fallback.")
                        entry_price_unified = None # Force fallback

                # Fallback to Bybit V5 'info.avgPrice' if unified entry price is missing/invalid
                if entry_price_unified is None:
                    avg_price_str_v5 = pos_info.get('avgPrice')
                    if avg_price_str_v5 is not None and avg_price_str_v5 != "":
                        try:
                            entry_price = float(avg_price_str_v5)
                            if entry_price <= FLOAT_COMPARISON_EPSILON: entry_price = 0.0 # Treat invalid price as 0
                            logger.debug(f"Position Check: Unified entry price missing/invalid, using info.avgPrice '{avg_price_str_v5}' -> {entry_price}")
                        except (ValueError, TypeError):
                            logger.warning(f"Position Check: Could not parse fallback entry price 'info.avgPrice': '{avg_price_str_v5}'. Defaulting to 0.0.")
                            entry_price = 0.0

                # Get market precision for logging
                price_precision_digits = market.get('precision', {}).get('price', 4)
                qty_precision_digits = market.get('precision', {}).get('amount', 8)

                # Store the found position details
                found_position = {'side': determined_side, 'qty': abs(quantity), 'entry_price': entry_price}
                logger.info(f"Position Check: Found ACTIVE position: {determined_side} {abs(quantity):.{qty_precision_digits}f} {market_id} @ Entry={entry_price:.{price_precision_digits}f}")
                break # Found the one-way position for the symbol, no need to check further
            else:
                # Quantity is zero or negligible, treat as closed/flat
                logger.debug(f"Skipping position entry for {market_id}: Quantity {quantity:.8f} is <= epsilon {FLOAT_COMPARISON_EPSILON}. Treating as closed.")
                continue # Move to the next entry in all_positions

        # End of loop through all_positions

        # Return the found position or the default if none matched
        if found_position:
            return found_position
        else:
            logger.info(f"Position Check: No active One-Way Mode position found for {market_id} after checking {len(all_positions)} entries.")
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
    Verifies that the symbol is a linear contract market first. Uses Bybit V5 specific params.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol in CCXT unified format.
        leverage (int): Desired leverage level (e.g., 10 for 10x).

    Returns:
        bool: True if leverage is successfully set (or was already correct), False otherwise.
    """
    logger.info(f"Leverage Setting: Attempting to set {leverage}x leverage for {symbol}...")
    market = None
    try:
        market = exchange.market(symbol)
        # --- Market Type Check ---
        # Ensure it's a futures/swap market where leverage is applicable
        # CCXT market structure flags: 'swap', 'future', 'option'. 'contract' is a general flag.
        is_contract = market.get('contract', False)
        is_linear = market.get('linear', False)
        market_type = market.get('type', 'N/A')

        if not is_contract or market.get('spot', False):
            logger.error(f"Leverage Setting: Cannot set leverage for non-contract market: {symbol}. Market type: {market_type}")
            return False
        # Bot specifically requires linear contracts
        if not is_linear:
            logger.error(f"Leverage Setting: Cannot set leverage for non-linear market: {symbol}. Market type: {market_type}. Bot requires USDT-margined contracts.")
            return False

        logger.debug(f"Market info for {symbol}: Type={market_type}, Contract={is_contract}, Linear={is_linear}")

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
            # Use the unified set_leverage method with Bybit V5 specific params
            # V5 requires category and specifying both buy/sell leverage
            # Important: Bybit V5 API often expects leverage values as STRINGS
            params = {
                'category': 'linear', # Explicitly linear
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage)
            }

            logger.debug(f"Leverage Setting: Calling exchange.set_leverage(leverage={leverage}, symbol='{symbol}', params={params}) (Attempt {attempt + 1}/{RETRY_COUNT})")

            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            # Response format varies, log it for info
            logger.success(f"Leverage Setting: Successfully set leverage to {leverage}x for {symbol}. Response: {response}")
            return True

        except ccxt.ExchangeError as e:
            # Check for common messages indicating leverage is already set or no change needed
            error_msg_lower = str(e).lower()
            # Bybit V5 often returns specific codes like 110044 for "leverage not modified"
            if "leverage not modified" in error_msg_lower or \
               "same leverage" in error_msg_lower or \
               "leverage is not changed" in error_msg_lower or \
               "110044" in str(e): # Check for Bybit V5 code
                logger.info(f"Leverage Setting: Leverage for {symbol} already set to {leverage}x (Confirmed by exchange message: {e}).")
                return True

            # Log other exchange errors and retry if attempts remain
            logger.warning(f"Leverage Setting: Exchange error on attempt {attempt + 1}/{RETRY_COUNT} for {symbol}: {e}")
            if attempt < RETRY_COUNT - 1:
                logger.debug(f"Retrying leverage setting after {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"Leverage Setting: Failed to set leverage for {symbol} after {RETRY_COUNT} attempts due to exchange error: {e}")
                send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL: Leverage set FAILED for {symbol} after retries: {type(e).__name__}")

        except ccxt.NetworkError as e:
             logger.warning(f"Leverage Setting: Network error on attempt {attempt + 1}/{RETRY_COUNT} for {symbol}: {e}")
             if attempt < RETRY_COUNT - 1:
                 time.sleep(RETRY_DELAY_SECONDS)
             else:
                 logger.error(f"Leverage Setting: Failed to set leverage for {symbol} after {RETRY_COUNT} attempts due to network errors.")
                 send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL: Leverage set FAILED for {symbol} due to Network Errors.")

        except Exception as e:
            # Catch any other unexpected exceptions during the process
            logger.error(f"Leverage Setting: Unexpected error on attempt {attempt + 1} for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL: Leverage set FAILED for {symbol} due to Unexpected Error: {type(e).__name__}")
            return False # Exit immediately on unexpected errors

    # If loop finishes without returning True, it failed
    return False


# --- Close Position ---
def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal"
) -> Optional[Dict[str, Any]]:
    """
    Closes the specified active position using a market order with reduce_only=True.
    Includes a pre-close position re-validation step using get_current_position.
    Uses Bybit V5 specific params. Sends SMS alerts on success or failure.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol (e.g., 'BTC/USDT:USDT').
        position_to_close (Dict[str, Any]): The position dictionary (containing at least
                                             'side' and 'qty') as identified *before*
                                             calling this function. Used for initial info only.
        reason (str): Reason for closing (e.g., "SL", "TP", "Reversal", "Shutdown").

    Returns:
        Optional[Dict[str, Any]]: The CCXT order dictionary if the close order was
                                  successfully placed, None otherwise.
    """
    initial_side = position_to_close.get('side', POSITION_SIDE_NONE)
    initial_qty = position_to_close.get('qty', 0.0)
    market_base = symbol.split('/')[0] # For concise SMS alerts (e.g., "BTC")

    # Get market precision for logging
    market = None
    qty_precision_digits = 8 # Default
    price_precision_digits = 4 # Default
    try:
        market = exchange.market(symbol)
        qty_precision_digits = market.get('precision', {}).get('amount', 8)
        price_precision_digits = market.get('precision', {}).get('price', 4)
    except Exception as market_err:
        logger.warning(f"Close Position: Could not get market precision for {symbol}: {market_err}. Using defaults.")


    logger.info(f"Close Position: Initiated for {symbol}. Reason: {reason}. Initial state guess: {initial_side} Qty={initial_qty:.{qty_precision_digits}f}")

    # ** Crucial Pre-Close Re-validation Step **
    # Fetch the position state *again* immediately before attempting to close.
    logger.debug(f"Close Position: Re-validating current position state for {symbol} using get_current_position...")
    live_position = get_current_position(exchange, symbol)
    live_position_side = live_position.get('side')
    live_amount_to_close = live_position.get('qty', 0.0)

    # Check if the live position actually exists according to our robust check
    if live_position_side == POSITION_SIDE_NONE or live_amount_to_close <= FLOAT_COMPARISON_EPSILON:
        logger.warning(f"Close Position: Re-validation shows NO active position for {symbol} (Live Side: {live_position_side}, Qty: {live_amount_to_close:.{qty_precision_digits}f}). Aborting close attempt.")
        # If the initial state *thought* there was a position, log that discrepancy
        if initial_side != POSITION_SIDE_NONE:
             logger.warning(f"Close Position: Discrepancy detected. Initial check showed {initial_side}, but live check shows none.")
        # Consider sending an SMS only if a discrepancy was detected? Or maybe not, to avoid noise.
        return None

    # Use the freshly validated quantity and determine the closing side
    side_to_execute_close = SIDE_SELL if live_position_side == POSITION_SIDE_LONG else SIDE_BUY

    try:
        # Format the amount to the exchange's required precision
        amount_str = exchange.amount_to_precision(symbol, live_amount_to_close)
        amount_float = float(amount_str)

        # Final check on the amount after precision formatting
        if amount_float <= FLOAT_COMPARISON_EPSILON:
            logger.error(f"Close Position: Calculated closing amount after precision ({amount_str}) is zero or negligible for {symbol}. Aborting.")
            return None

        logger.warning(f"Close Position: Attempting to CLOSE {live_position_side} position ({reason}): "
                       f"Executing {side_to_execute_close.upper()} MARKET order for {amount_str} {symbol} "
                       f"with params {{'reduce_only': True}}...")

        # --- Execute the Closing Market Order ---
        # Bybit V5 requires category and reduce_only
        params = {
            'reduce_only': True,
            'category': 'linear' # Default assumption for this bot
            }
        # Re-fetch market just in case, though should be available from precision check
        if market is None: market = exchange.market(symbol)
        if market.get('inverse'): params['category'] = 'inverse' # Adjust if somehow inverse
        elif not market.get('linear'): # Defensive check if category couldn't be determined before
             logger.error(f"Close Position: Cannot determine category (linear/inverse) for {symbol}. Aborting close.")
             return None

        order = exchange.create_market_order(
            symbol=symbol,
            side=side_to_execute_close,
            amount=amount_float,
            params=params
        )
        # --- Post-Order Placement ---
        # Log success and key order details
        fill_price = order.get('average')
        filled_qty = order.get('filled')
        order_id = order.get('id', 'N/A')
        order_cost = order.get('cost') # Cost in quote currency (USDT)

        fill_price_str = f"{fill_price:.{price_precision_digits}f}" if fill_price is not None else "Market (Check Fills)"
        filled_qty_str = f"{filled_qty:.{qty_precision_digits}f}" if filled_qty is not None else "?"
        order_id_short = str(order_id)[-6:] # Get last 6 chars of ID for brevity
        cost_str = f"{order_cost:.2f}" if order_cost is not None else "N/A"

        logger.success(f"Close Position: CLOSE Order ({reason}) placed successfully for {symbol}. "
                       f"Qty Filled: {filled_qty_str}/{amount_str}, Avg Fill ~ {fill_price_str}, Cost: {cost_str} USDT. ID:...{order_id_short}")

        # Send SMS Alert on successful order placement
        sms_msg = (f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{fill_price_str} ({reason}). "
                   f"ID:...{order_id_short}")
        send_sms_alert(sms_msg)
        return order # Return the order details dictionary

    # --- Error Handling for Order Creation ---
    except ccxt.InsufficientFunds as e:
        # This *shouldn't* happen often with reduce_only, but possible in weird margin scenarios or if validation failed somehow
        logger.error(f"Close Position ({reason}): Insufficient funds error during close attempt for {symbol}: {e}")
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insufficient funds! Check margin/position state.")
    except ccxt.NetworkError as e:
        logger.error(f"Close Position ({reason}): Network error placing close order for {symbol}: {e}")
        # SMS might fail here too if network is down
    except ccxt.ExchangeError as e:
        # Log the specific error message from the exchange (e.g., "position is zero", order validation errors)
        logger.error(f"Close Position ({reason}): Exchange error placing close order for {symbol}: {e}")
        # Check for specific common errors if needed (e.g., position already closed - might be indicated by certain error codes/messages)
        # Example: Bybit error "30036: The order quantity exceeds the current position quantity" might mean it's already closed or partially closed.
        # Bybit V5 error "110025: Position is closing" or "110043: Position size is zero"
        error_str_lower = str(e).lower()
        if "position" in error_str_lower and ("zero" in error_str_lower or "exceeds" in error_str_lower or "not found" in error_str_lower or "closing" in error_str_lower) or \
           "110025" in str(e) or "110043" in str(e):
             logger.warning(f"Close Position ({reason}): Exchange error suggests position might already be closed or closing ({e}). Assuming closure succeeded or is in progress.")
             # Don't send an error SMS in this specific case, as it might be resolved.
             return None # Treat as potentially already closed or handled
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Exchange error ({type(e).__name__}). Check logs.")
    except ValueError as e:
        # Catch potential errors from amount_to_precision or float conversion
        logger.error(f"Close Position ({reason}): Value error during amount processing for {symbol} (Qty: {live_amount_to_close}): {e}")
    except Exception as e:
        logger.error(f"Close Position ({reason}): Unexpected error placing close order for {symbol}: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Unexpected error ({type(e).__name__}). Check logs.")

    # Return None if any exception occurred during order placement (unless handled above)
    return None


# --- Risk Calculation ---
def calculate_position_size(
    equity: float, risk_per_trade_pct: float, entry_price: float, stop_loss_price: float,
    leverage: int, symbol: str, exchange: ccxt.Exchange
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates position size (quantity) based on risk percentage, entry/SL prices,
    and estimates the required initial margin (without buffer). Applies precision early.

    Args:
        equity (float): Current account equity (or relevant balance) in USDT.
        risk_per_trade_pct (float): Desired risk percentage (e.g., 0.01 for 1%).
        entry_price (float): Estimated entry price.
        stop_loss_price (float): Calculated stop-loss price.
        leverage (int): Leverage used for the symbol.
        symbol (str): Market symbol (used for precision).
        exchange (ccxt.Exchange): CCXT exchange instance (used for precision).

    Returns:
        Tuple[Optional[float], Optional[float]]: (quantity_precise, required_margin_estimate)
           - quantity_precise (float): The calculated position size in base currency (e.g., BTC), adjusted for exchange precision.
           - required_margin_estimate (float): Estimated USDT initial margin needed for the position.
           Returns (None, None) if calculation is invalid or results in zero/negative size.
    """
    # Get market precision for logging
    market = None
    price_precision_digits = 4 # Default
    qty_precision_digits = 8 # Default
    try:
        market = exchange.market(symbol)
        price_precision_digits = market.get('precision', {}).get('price', 4)
        qty_precision_digits = market.get('precision', {}).get('amount', 8)
    except Exception as market_err:
         logger.warning(f"Risk Calc: Could not get market precision for {symbol}: {market_err}. Using defaults.")

    logger.debug(f"Risk Calc Input: Equity={equity:.2f}, Risk%={risk_per_trade_pct:.3%}, "
                 f"Entry={entry_price:.{price_precision_digits}f}, SL={stop_loss_price:.{price_precision_digits}f}, Lev={leverage}x, Symbol={symbol}")

    # --- Input Validation ---
    if not (entry_price > FLOAT_COMPARISON_EPSILON and stop_loss_price > FLOAT_COMPARISON_EPSILON):
        logger.error(f"Risk Calc: Invalid entry ({entry_price}) or stop-loss ({stop_loss_price}) price (must be > ~0).")
        return None, None
    price_difference_per_unit = abs(entry_price - stop_loss_price)
    if price_difference_per_unit < FLOAT_COMPARISON_EPSILON: # Use epsilon for float comparison
        logger.error(f"Risk Calc: Entry ({entry_price:.{price_precision_digits}f}) and stop-loss ({stop_loss_price:.{price_precision_digits}f}) prices are too close or identical. Cannot calculate size.")
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(f"Risk Calc: Invalid risk_per_trade_pct: {risk_per_trade_pct:.3%}. Must be between 0% and 100% (exclusive).")
        return None, None
    if equity <= FLOAT_COMPARISON_EPSILON:
         logger.error(f"Risk Calc: Invalid equity (<= ~0): {equity:.2f}")
         return None, None
    if leverage <= 0:
         logger.error(f"Risk Calc: Invalid leverage (<= 0): {leverage}")
         return None, None

    # --- Calculation ---
    # 1. Calculate Risk Amount in USDT
    risk_amount_usdt = equity * risk_per_trade_pct

    # 2. Calculate Position Size (Quantity)
    # Quantity = Risk Amount / (Price difference per unit)
    quantity_raw = risk_amount_usdt / price_difference_per_unit

    # 3. Apply exchange precision to the calculated quantity *early*
    quantity_precise: Optional[float] = None
    quantity_precise_str: str = "N/A"
    try:
        quantity_precise_str = exchange.amount_to_precision(symbol, quantity_raw)
        quantity_precise = float(quantity_precise_str)
        # Log raw with more digits for comparison
        logger.debug(f"Risk Calc: Raw Qty={quantity_raw:.{qty_precision_digits+2}f}, Precise Qty={quantity_precise_str}")
    except Exception as e:
         logger.error(f"Risk Calc: Could not apply precision to quantity {quantity_raw:.8f} for {symbol}. Error: {e}")
         return None, None # Fail calculation if precision cannot be applied

    # 4. Check if calculated precise quantity is valid (significantly non-zero)
    if quantity_precise is None or quantity_precise <= FLOAT_COMPARISON_EPSILON:
        logger.warning(f"Risk Calc: Calculated quantity ({quantity_precise_str}) is zero or negligible based on risk parameters. "
                       f"RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference_per_unit:.{price_precision_digits}f}")
        return None, None

    # 5. Estimate Position Value and Required Margin (Initial Margin) based on precise quantity
    position_value_usdt = quantity_precise * entry_price
    # Required Margin = Position Value / Leverage
    required_margin_estimate = position_value_usdt / leverage
    # Note: This is the *initial* margin estimate. Actual margin requirements might
    # include fees, funding rates, and vary slightly based on exact fill price.
    # The margin *buffer* check happens later using free balance.

    logger.debug(f"Risk Calc Output: RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference_per_unit:.{price_precision_digits}f} "
                 f"=> PreciseQty={quantity_precise_str}, EstPosValue={position_value_usdt:.2f}, EstMarginBase={required_margin_estimate:.2f}")
    return quantity_precise, required_margin_estimate


# --- Place Order ---
def place_risked_market_order(
    exchange: ccxt.Exchange, symbol: str, side: str,
    risk_percentage: float, current_atr: Optional[float], sl_atr_multiplier: float,
    leverage: int, max_order_cap_usdt: float, margin_check_buffer: float
) -> Optional[Dict[str, Any]]:
    """
    Calculates position size based on risk & ATR SL, checks balance (equity/free),
    limits, and margin buffer, then places a market order if all checks pass.
    Uses shallow OB fetch for entry price estimate. Uses Bybit V5 specific params.
    Sends SMS alerts.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Market symbol (e.g., 'BTC/USDT:USDT').
        side (str): 'buy' or 'sell'.
        risk_percentage (float): Risk per trade as a decimal (e.g., 0.01 for 1%).
        current_atr (Optional[float]): The current ATR value for SL calculation (must be > 0).
        sl_atr_multiplier (float): Multiplier for ATR to determine SL distance.
        leverage (int): Leverage confirmed for the symbol.
        max_order_cap_usdt (float): Maximum allowed position size in USDT value.
        margin_check_buffer (float): Buffer multiplier for checking available margin (e.g., 1.10 for 10% buffer).

    Returns:
        Optional[Dict[str, Any]]: The CCXT order dictionary if the order was successfully placed, None otherwise.
    """
    market_base = symbol.split('/')[0] # For concise SMS alerts
    logger.info(f"Place Order: Initiating {side.upper()} market order placement process for {symbol}...")

    # --- Pre-computation Checks ---
    if current_atr is None or current_atr <= FLOAT_COMPARISON_EPSILON: # Check against epsilon
        logger.error(f"Place Order ({side.upper()}): Invalid ATR value ({current_atr}) provided for SL calculation. Cannot place order.")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid ATR provided.")
        return None

    entry_price_estimate: Optional[float] = None
    stop_loss_price: Optional[float] = None
    final_quantity: Optional[float] = None
    final_quantity_str: str = "N/A"
    stop_loss_price_str: str = "N/A" # For logging/SMS

    try:
        # --- 1. Get Balance, Market Info, and Limits ---
        logger.debug("Place Order: Fetching balance and market details...")
        market = exchange.market(symbol) # Already validated in main, but good practice to have it here
        # Get precision format strings for logging/debugging
        qty_precision_digits = market.get('precision', {}).get('amount', 8)
        price_precision_digits = market.get('precision', {}).get('price', 4)

        # Bybit V5 needs category for balance fetch too
        balance_params = {}
        if market.get('linear'):
            balance_params = {'category': 'linear'}
        elif market.get('inverse'):
            balance_params = {'category': 'inverse'} # Add inverse just in case
        else:
             logger.error(f"Place Order: Cannot determine category (linear/inverse) for balance fetch for {symbol}. Aborting.")
             return None
        balance = exchange.fetch_balance(params=balance_params)

        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        min_qty = amount_limits.get('min')
        max_qty = amount_limits.get('max')
        min_price = price_limits.get('min')
        max_price = price_limits.get('max')

        logger.debug(f"Place Order: Market Limits -> MinQty={min_qty}, MaxQty={max_qty}, QtyPrec={qty_precision_digits}, PricePrec={price_precision_digits}")
        logger.debug(f"Place Order: Market Price Limits -> MinPrice={min_price}, MaxPrice={max_price}")

        # Get available equity/free balance in USDT (adjust symbol if needed)
        # Bybit V5 balance structure might be nested, check 'total', 'free', 'used' for USDT
        usdt_balance = balance.get(USDT_SYMBOL, {})
        # Prefer 'total' for equity, but check 'free' and 'used' as fallback
        usdt_total = usdt_balance.get('total') # Total equity
        usdt_free = usdt_balance.get('free')   # Free margin available
        usdt_used = usdt_balance.get('used')   # Used margin

        usdt_equity = 0.0
        if usdt_total is not None and isinstance(usdt_total, (int, float)) and usdt_total > FLOAT_COMPARISON_EPSILON:
            usdt_equity = float(usdt_total)
        elif usdt_free is not None and usdt_used is not None and isinstance(usdt_free, (int, float)) and isinstance(usdt_used, (int, float)):
            # Fallback: Equity = Free + Used
            usdt_equity = float(usdt_free) + float(usdt_used)
            logger.warning(f"Place Order: USDT 'total' balance is {usdt_total}. Using 'free' ({usdt_free}) + 'used' ({usdt_used}) = {usdt_equity:.2f} for equity calculation.")
        elif usdt_free is not None and isinstance(usdt_free, (int, float)):
             # Less accurate fallback: Equity ~ Free (if position is small or flat)
             usdt_equity = float(usdt_free)
             logger.warning(f"Place Order: USDT 'total' and 'used' balance missing/zero. Using 'free' balance ({usdt_free:.2f}) for equity calculation (less accurate).")
        # else: usdt_equity remains 0.0

        if usdt_equity <= FLOAT_COMPARISON_EPSILON:
            logger.error(f"Place Order ({side.upper()}): Cannot determine valid account equity (USDT <= ~0). Equity: {usdt_equity:.2f}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Zero/Invalid equity.")
            return None
        # Ensure free margin is also a valid float before comparison
        if usdt_free is None or not isinstance(usdt_free, (int, float)) or float(usdt_free) < 0:
             logger.error(f"Place Order ({side.upper()}): Invalid free margin (USDT free is {usdt_free}). Cannot place order.")
             send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid free margin.")
             return None
        usdt_free = float(usdt_free) # Ensure it's a float for calculations

        logger.info(f"Place Order: Account Balance -> Equity={usdt_equity:.2f} USDT, Free Margin={usdt_free:.2f} USDT")


        # --- 2. Estimate Entry Price ---
        # Use a shallow order book fetch for a slightly more realistic entry price estimate than ticker 'last'
        logger.debug(f"Place Order: Fetching shallow OB (depth {SHALLOW_OB_FETCH_DEPTH}) for entry price estimate...")
        # Use larger fetch limit for OB, depth for analysis
        ob_data = analyze_order_book(exchange, symbol, SHALLOW_OB_FETCH_DEPTH, ORDER_BOOK_FETCH_LIMIT)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")

        if side == SIDE_BUY and best_ask and best_ask > FLOAT_COMPARISON_EPSILON:
            entry_price_estimate = best_ask
            logger.debug(f"Place Order: Using best ASK {entry_price_estimate:.{price_precision_digits}f} from shallow OB as BUY entry estimate.")
        elif side == SIDE_SELL and best_bid and best_bid > FLOAT_COMPARISON_EPSILON:
            entry_price_estimate = best_bid
            logger.debug(f"Place Order: Using best BID {entry_price_estimate:.{price_precision_digits}f} from shallow OB as SELL entry estimate.")
        else:
            # Fallback to ticker 'last' price if OB fetch fails or returns invalid prices
            logger.warning("Place Order: Shallow OB fetch failed or returned invalid bid/ask. Falling back to ticker 'last' price for entry estimate.")
            try:
                # Bybit V5 requires category for ticker fetch
                ticker_params = {}
                if market.get('linear'):
                    ticker_params = {'category': 'linear'}
                elif market.get('inverse'):
                    ticker_params = {'category': 'inverse'}
                else:
                    ticker_params = {} # Fallback if category unknown

                ticker = exchange.fetch_ticker(symbol, params=ticker_params)
                last_price = ticker.get('last')
                if last_price and isinstance(last_price, (int, float)) and float(last_price) > FLOAT_COMPARISON_EPSILON:
                    entry_price_estimate = float(last_price)
                    logger.debug(f"Place Order: Using ticker 'last' price {entry_price_estimate:.{price_precision_digits}f} as entry estimate.")
                else:
                    entry_price_estimate = None # Ensure it's None if ticker price is invalid
            except Exception as ticker_err:
                logger.error(f"Place Order: Failed to fetch ticker price as fallback: {ticker_err}")
                entry_price_estimate = None # Ensure it's None if ticker fails too

        # Validate the estimated entry price
        if entry_price_estimate is None or entry_price_estimate <= FLOAT_COMPARISON_EPSILON:
            logger.error(f"Place Order ({side.upper()}): Could not determine a valid entry price estimate (Estimate: {entry_price_estimate}). Cannot place order.")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): No valid entry price estimate.")
            return None

        # --- 3. Calculate Stop Loss Price ---
        stop_loss_distance = current_atr * sl_atr_multiplier
        if side == SIDE_BUY:
            stop_loss_price = entry_price_estimate - stop_loss_distance
        else: # SIDE_SELL
            stop_loss_price = entry_price_estimate + stop_loss_distance

        # Ensure SL price is positive and respects exchange minimum price limit (if available)
        if min_price is not None and isinstance(min_price, (int, float)) and stop_loss_price < float(min_price):
            logger.warning(f"Place Order: Calculated SL price {stop_loss_price:.{price_precision_digits}f} is below minimum limit {min_price}. Adjusting SL to minimum limit.")
            stop_loss_price = float(min_price)
        elif stop_loss_price <= FLOAT_COMPARISON_EPSILON:
             logger.error(f"Place Order ({side.upper()}): Calculated SL price is zero or negative ({stop_loss_price:.{price_precision_digits}f}). Cannot place order with invalid SL.")
             send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid SL price calc (<=0).")
             return None # Prevent order if SL becomes invalid

        # Apply exchange price precision to the calculated SL price
        try:
            stop_loss_price_str = exchange.price_to_precision(symbol, stop_loss_price)
            stop_loss_price = float(stop_loss_price_str) # Convert back to float after formatting
            logger.info(f"Place Order: Calculated SL -> EntryEst={entry_price_estimate:.{price_precision_digits}f}, "
                        f"ATR({current_atr:.5f}) * {sl_atr_multiplier} (Dist={stop_loss_distance:.{price_precision_digits}f}) => SL Price ~ {stop_loss_price_str}")
        except Exception as e:
            logger.error(f"Place Order: Error applying precision to SL price {stop_loss_price:.{price_precision_digits}f}: {e}. Aborting.")
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

        final_quantity = calculated_quantity # Start with the risk-calculated quantity (already precise)

        # --- 5. Apply Max Order Cap (USDT Value) ---
        order_value_usdt = final_quantity * entry_price_estimate
        if order_value_usdt > max_order_cap_usdt:
            logger.warning(f"Place Order: Initial risk-based order value {order_value_usdt:.2f} USDT exceeds MAX Cap {max_order_cap_usdt:.2f} USDT. Capping quantity.")
            capped_quantity_raw = max_order_cap_usdt / entry_price_estimate
            # Re-apply precision to the capped quantity
            try:
                final_quantity_str = exchange.amount_to_precision(symbol, capped_quantity_raw)
                final_quantity = float(final_quantity_str)
                logger.info(f"Place Order: Quantity capped to ~ {final_quantity_str} based on Max Cap.")
                # Recalculate estimated margin based on the capped quantity
                required_margin_estimate = ((final_quantity * entry_price_estimate) / leverage)
            except Exception as e:
                logger.error(f"Place Order: Error applying precision to capped quantity {capped_quantity_raw:.{qty_precision_digits+2}f}: {e}. Aborting.")
                return None
        else:
            # If not capped, still need the string representation of the final quantity
            final_quantity_str = exchange.amount_to_precision(symbol, final_quantity)


        # --- 6. Check Min/Max Quantity Limits ---
        # Check if final quantity is effectively zero after precision/capping
        if final_quantity <= FLOAT_COMPARISON_EPSILON:
            logger.error(f"Place Order ({side.upper()}): Final quantity {final_quantity_str} is zero or negligible after risk calc/capping/precision. Cannot place order.")
            return None # No order to place

        # Check against Min/Max quantity limits from the exchange (convert limits to float for comparison)
        min_qty_float = float(min_qty) if min_qty is not None and isinstance(min_qty, (int, float, str)) else None
        max_qty_float = float(max_qty) if max_qty is not None and isinstance(max_qty, (int, float, str)) else None

        if min_qty_float is not None and final_quantity < min_qty_float:
            logger.error(f"Place Order ({side.upper()}): Final Qty {final_quantity_str} is LESS than exchange minimum quantity {min_qty_float}. Skipping order.")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Qty {final_quantity_str} < Min {min_qty_float}.")
            return None
        if max_qty_float is not None and final_quantity > max_qty_float:
            logger.warning(f"Place Order ({side.upper()}): Final Qty {final_quantity_str} is GREATER than exchange maximum quantity {max_qty_float}. Capping to max qty.")
            final_quantity = max_qty_float
            # Re-apply precision after capping to max_qty
            try:
                final_quantity_str = exchange.amount_to_precision(symbol, final_quantity)
                final_quantity = float(final_quantity_str)
                # Recalculate estimated margin based on max quantity
                required_margin_estimate = ((final_quantity * entry_price_estimate) / leverage)
                logger.info(f"Place Order: Quantity capped to exchange max: {final_quantity_str}. New Est. Margin ~ {required_margin_estimate:.2f} USDT.")
            except Exception as e:
                logger.error(f"Place Order: Error applying precision after capping to max qty {max_qty_float}: {e}. Aborting.")
                return None

        # Re-calculate final estimated position value and margin after all adjustments
        final_pos_value = final_quantity * entry_price_estimate
        final_req_margin = required_margin_estimate # Use the potentially recalculated margin

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
        # Bybit V5 requires category for market orders
        params = {}
        if market.get('linear'):
            params = {'category': 'linear'}
        elif market.get('inverse'):
            params = {'category': 'inverse'}
        else:
             logger.error(f"Place Order: Cannot determine category (linear/inverse) for {symbol}. Aborting order.")
             return None

        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=final_quantity, # Pass the final float quantity
            params=params
        )

        # --- 9. Post-Order Handling & Logging ---
        # Extract details from the returned order object
        filled_qty = order.get('filled') # Amount actually filled (might be float or string)
        avg_fill_price = order.get('average') # Average price if filled
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A')
        order_cost = order.get('cost') # Cost in quote currency (USDT)

        # Safely convert filled_qty and avg_fill_price to float for comparison/logging
        filled_qty_float = 0.0
        if filled_qty is not None:
            try:
                filled_qty_float = float(filled_qty)
            except (ValueError, TypeError): pass # Keep as 0.0 if conversion fails

        avg_fill_price_float = None
        if avg_fill_price is not None:
            try:
                avg_fill_price_float = float(avg_fill_price)
            except (ValueError, TypeError): pass

        fill_price_str = f"{avg_fill_price_float:.{price_precision_digits}f}" if avg_fill_price_float is not None else "N/A"
        filled_qty_str = f"{filled_qty_float:.{qty_precision_digits}f}" if filled_qty_float > FLOAT_COMPARISON_EPSILON else "0.0"
        order_id_short = str(order_id)[-6:] # Last 6 chars for brevity
        cost_str = f"{float(order_cost):.2f}" if order_cost is not None and isinstance(order_cost, (int, float, str)) else "N/A"

        # Check if the order actually filled significantly
        if filled_qty_float > FLOAT_COMPARISON_EPSILON:
            logger.success(f"Place Order: {side.upper()} Market Order Placed Successfully! "
                           f"ID:...{order_id_short}, Status: {order_status}, "
                           f"Filled Qty: {filled_qty_str}/{final_quantity_str}, "
                           f"Avg Fill Price: {fill_price_str}, Cost: {cost_str} USDT")

            # Send SMS Alert for successful entry
            sms_msg = (f"[{market_base}] ENTER {side.upper()} {filled_qty_str} @ ~{fill_price_str}. "
                       f"Calc SL ~ {stop_loss_price_str}. ID:...{order_id_short}") # Include calculated SL for reference
            send_sms_alert(sms_msg)

            # --- !!! CRITICAL SCALPING ACTION REQUIRED: Implement Exchange-Native SL/TP !!! ---
            logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.critical(f"!!! ACTION REQUIRED: Market order {order_id} placed. Loop-based SL/TP is DANGEROUSLY INACCURATE.    !!!")
            logger.critical(f"!!! YOU **MUST** NOW PLACE EXCHANGE-NATIVE STOP-LOSS and/or TAKE-PROFIT ORDERS                     !!!")
            logger.critical(f"!!! based on the actual fill price ({fill_price_str}) and quantity ({filled_qty_str}).             !!!")
            logger.critical("!!! Example using Bybit V5 'Set Trading Stop' (verify params and implementation):                  !!!")
            logger.critical("!!! `exchange.private_post_v5_position_set_trading_stop({'category': 'linear', 'symbol': market['id'], 'stopLoss': 'YOUR_SL_PRICE', 'slTriggerBy': 'LastPrice', 'positionIdx': 0})` !!!")
            logger.critical("!!! Failure to implement reliable SL/TP exposes the position to UNCONTROLLED RISK and potential    !!!")
            logger.critical("!!! liquidation, especially with leverage.                                                         !!!")
            logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Example (Conceptual - Requires specific CCXT implementation for conditional/trigger orders on Bybit V5):
            # if avg_fill_price_float and filled_qty_float > FLOAT_COMPARISON_EPSILON:
            #     try:
            #         # Recalculate SL based on actual fill price
            #         actual_sl_price = avg_fill_price_float - stop_loss_distance if side == SIDE_BUY else avg_fill_price_float + stop_loss_distance
            #         actual_sl_price_str = exchange.price_to_precision(symbol, actual_sl_price)
            #
            #         # --- Bybit V5 Example: Set Stop Loss on Position ---
            #         sl_params = {
            #             'category': 'linear',
            #             'symbol': market['id'], # Use exchange-specific ID
            #             'stopLoss': actual_sl_price_str,
            #             'slTriggerBy': 'LastPrice', # Or 'MarkPrice', 'IndexPrice'
            #             # 'tpslMode': 'Full' or 'Partial' - Affects entire position vs portion
            #             'slOrderType': 'Market', # Or 'Limit'
            #             'positionIdx': 0 # Required for One-Way mode
            #         }
            #         logger.info(f"Attempting to set exchange Stop-Loss via API: SL Price={actual_sl_price_str}")
            #         # Use the appropriate private method (check CCXT/Bybit docs)
            #         # response = exchange.private_post_v5_position_set_trading_stop(sl_params)
            #         # logger.success(f"Exchange SL setting response: {response}")
            #         logger.warning("Conceptual SL placement shown. Verify correct CCXT/Bybit V5 implementation.")
            #
            #         # --- Optionally set Take Profit similarly ---
            #         # ENABLE_EXCHANGE_TP = False # Add a separate config flag?
            #         # if ENABLE_EXCHANGE_TP:
            #         #     tp_atr_multiplier = ATR_TAKE_PROFIT_MULTIPLIER # Use configured TP multiplier
            #         #     tp_distance = current_atr * tp_atr_multiplier
            #         #     actual_tp_price = avg_fill_price_float + tp_distance if side == SIDE_BUY else avg_fill_price_float - tp_distance
            #         #     actual_tp_price_str = exchange.price_to_precision(symbol, actual_tp_price)
            #         #     # Add 'takeProfit' to the params dictionary for set_trading_stop
            #         #     tp_params = sl_params.copy() # Start with SL params
            #         #     tp_params['takeProfit'] = actual_tp_price_str
            #         #     tp_params['tpTriggerBy'] = 'LastPrice' # Or Mark/Index
            #         #     tp_params['tpOrderType'] = 'Market' # Or Limit
            #         #     logger.info(f"Attempting to set exchange Take-Profit via API: TP Price={actual_tp_price_str}")
            #         #     response = exchange.private_post_v5_position_set_trading_stop(tp_params) # Can often set SL and TP together
            #         #     logger.success(f"Exchange TP setting response: {response}")
            #
            #     except Exception as cond_order_err:
            #         logger.error(f"CRITICAL FAILURE: Failed to place exchange-native SL/TP order after entry: {cond_order_err}")
            #         send_sms_alert(f"[{market_base}] CRITICAL! Failed to place SL/TP for order {order_id_short}. MANUAL INTERVENTION NEEDED!")
            # ---------------------------------------

            return order # Return the successful market order dictionary

        else:
            # Order was placed but didn't fill significantly or fill info missing
            logger.error(f"Place Order ({side.upper()}): Market order placed (ID:...{order_id_short}, Status:{order_status}) but reported fill quantity ({filled_qty_str}) is zero or negligible. Check order status manually.")
            send_sms_alert(f"[{market_base}] ORDER WARN ({side.upper()}): Order {order_id_short} placed but fill Qty is {filled_qty_str}. Check manually!")
            return None # Treat as failed placement if no significant fill

    # --- Exception Handling for Order Placement Process ---
    except ccxt.InsufficientFunds as e:
        # This might occur if free margin calculation was slightly off, fees pushed it over, or balance changed rapidly
        logger.error(f"Place Order ({side.upper()}): Insufficient funds error during order placement attempt for {symbol}: {e}")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient funds reported by exchange!")
    except ccxt.NetworkError as e:
        logger.error(f"Place Order ({side.upper()}): Network error during order placement for {symbol}: {e}")
        # SMS might fail too
    except ccxt.ExchangeError as e:
        # Log specific exchange errors (e.g., invalid order params, margin issues not caught earlier, risk limit exceeded)
        logger.error(f"Place Order ({side.upper()}): Exchange error during order placement for {symbol}: {e}")
        # Provide more context in SMS if possible
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Exchange error ({type(e).__name__}). Check logs: {str(e)[:50]}") # Include start of error msg
    except ValueError as e:
        # Catch potential errors from precision formatting or float conversions
         logger.error(f"Place Order ({side.upper()}): Value error during price/amount processing for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Place Order ({side.upper()}): Unexpected critical error during order placement logic for {symbol}: {e}")
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
    3. Gets current position status using robust check.
    4. Logs current market state.
    5. Checks loop-based Stop-Loss / Take-Profit (UNSAFE FOR LIVE SCALPING, default OFF).
    6. Checks for entry signals based on dual Supertrend and optional confirmations (Volume, OB).
    7. Executes entry or exit actions (closing existing position if reversing, with confirmation).

    *** WARNING: Contains optional loop-based SL/TP check which is unsuitable for live scalping. ***

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): Trading symbol (e.g., 'BTC/USDT:USDT').
        df (pd.DataFrame): DataFrame with fresh OHLCV data.
        st_len, st_mult: Primary Supertrend parameters.
        cf_st_len, cf_st_mult: Confirmation Supertrend parameters.
        atr_len: ATR period for SL/TP distance calculation.
        vol_ma_len: Volume MA period.
        risk_pct: Risk per trade percentage.
        sl_atr_mult, tp_atr_mult: ATR multipliers for SL/TP (TP only used by loop check).
        leverage: Confirmed leverage for the symbol.
        max_order_cap: Max USDT order value cap.
        margin_buffer: Buffer for margin check before placing order.
    """
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"========== New Check Cycle: {symbol} | Candle Time: {cycle_time_str} ==========")

    # --- Data Validation ---
    # Ensure sufficient data rows for the longest indicator lookback period + buffer
    # Need +1 for shift(1) in supertrend calc, + buffer
    required_rows = max(st_len, cf_st_len, atr_len, vol_ma_len) + 5
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
        # Calculate ATR (for SL/TP distance) and Volume metrics
        vol_atr_data = analyze_volume_atr(df, atr_len, vol_ma_len)
        current_atr = vol_atr_data.get("atr")

        # --- Validate Indicator Results ---
        # Check if essential columns exist and the last row has valid values
        required_indicator_cols = ['close', 'supertrend', 'trend', 'st_long', 'st_short',
                                   'confirm_supertrend', 'confirm_trend']
        if not all(col in df.columns for col in required_indicator_cols):
             logger.error("Trade Logic: One or more required indicator columns missing after calculation. Skipping cycle.")
             return
        if df.iloc[-1][required_indicator_cols].isnull().any():
             logger.warning("Trade Logic: Indicator calculation resulted in NA/NaN values for the latest candle. Skipping cycle.")
             logger.debug(f"NA Values in last row:\n{df.iloc[-1][df.iloc[-1].isnull()]}")
             return
        # Validate ATR specifically, as it's crucial for risk management (SL calculation)
        if current_atr is None or current_atr <= FLOAT_COMPARISON_EPSILON: # Check against epsilon
             logger.warning(f"Trade Logic: Invalid ATR ({current_atr}) calculated. Cannot determine SL distance. Skipping cycle.")
             return

        # === Extract Data from the Last Closed Candle ===
        last_candle = df.iloc[-1]
        current_price: float = last_candle['close'] # Price at the close of the last candle
        primary_st_val: float = last_candle['supertrend']
        primary_trend_up: bool = last_candle["trend"]
        primary_long_signal: bool = last_candle["st_long"]  # Primary ST flipped Long THIS candle
        primary_short_signal: bool = last_candle["st_short"] # Primary ST flipped Short THIS candle
        confirm_st_val: float = last_candle['confirm_supertrend']
        confirm_trend_up: bool = last_candle["confirm_trend"] # Current state of Confirmation ST trend

        # === 2. Analyze Order Book (Conditionally) ===
        # Fetch OB if configured OR if a primary signal occurred (potential entry confirmation)
        # Only fetch if OB thresholds are active AND (FETCH_PER_CYCLE is True OR a primary signal occurred)
        ob_thresholds_active = ORDER_BOOK_RATIO_THRESHOLD_LONG > 0 or ORDER_BOOK_RATIO_THRESHOLD_SHORT < float('inf') # Check if any OB threshold is meaningful
        needs_ob_check = ob_thresholds_active and (FETCH_ORDER_BOOK_PER_CYCLE or primary_long_signal or primary_short_signal)

        if needs_ob_check:
            logger.debug("Trade Logic: Fetching and analyzing order book...")
            ob_data = analyze_order_book(exchange, symbol, ORDER_BOOK_DEPTH, ORDER_BOOK_FETCH_LIMIT)
        else:
            logger.debug("Trade Logic: Order book fetch skipped (Config/Thresholds/Signal did not require it).")

        # === 3. Get Current Position Status ===
        # This fetches the live position state from the exchange using the robust function
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
        market = None
        price_precision_digits = 4 # Default
        qty_precision_digits = 8 # Default
        try:
            market = exchange.market(symbol)
            price_precision_digits = market.get('precision', {}).get('price', 4)
            qty_precision_digits = market.get('precision', {}).get('amount', 8)
        except Exception as market_err:
             logger.warning(f"Trade Logic: Could not get market precision for {symbol}: {market_err}. Using defaults.")


        atr_str = f"{current_atr:.5f}"
        vol_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else 'N/A'
        vol_spike_str = f"{volume_spike}" if volume_ratio is not None else 'N/A'
        bid_ask_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else 'N/A'
        spread_str = f"{spread_val:.{price_precision_digits}f}" if spread_val is not None else 'N/A'
        primary_st_val_str = f"{primary_st_val:.{price_precision_digits}f}"
        confirm_st_val_str = f"{confirm_st_val:.{price_precision_digits}f}"
        current_price_str = f"{current_price:.{price_precision_digits}f}"
        entry_price_str = f"{entry_price:.{price_precision_digits}f}" if entry_price > FLOAT_COMPARISON_EPSILON else 'N/A'
        position_qty_str = f"{position_qty:.{qty_precision_digits}f}"

        logger.info(f"State | Price: {current_price_str}, ATR({atr_len}): {atr_str}")
        logger.info(f"State | Volume: Ratio={vol_ratio_str}, Spike={vol_spike_str} (Thr={VOLUME_SPIKE_THRESHOLD:.1f}, Req={REQUIRE_VOLUME_SPIKE_FOR_ENTRY})")
        logger.info(f"State | ST({st_len},{st_mult}): {'Up' if primary_trend_up else 'Down'} ({primary_st_val_str}) | Flip: {'LONG' if primary_long_signal else ('SHORT' if primary_short_signal else 'None')}")
        logger.info(f"State | ConfirmST({cf_st_len},{cf_st_mult}): {'Up' if confirm_trend_up else 'Down'} ({confirm_st_val_str})")
        logger.info(f"State | OrderBook(D{ORDER_BOOK_DEPTH}): Ratio={bid_ask_ratio_str} (L>={ORDER_BOOK_RATIO_THRESHOLD_LONG}, S<={ORDER_BOOK_RATIO_THRESHOLD_SHORT}), Spread={spread_str}")
        logger.info(f"State | Position: Side={position_side}, Qty={position_qty_str}, Entry={entry_price_str}")

        # === 5. Check Stop-Loss / Take-Profit (Loop-based - HIGH RISK / INACCURATE / DEFAULT OFF) ===
        if ENABLE_LOOP_SLTP_CHECK and position_side != POSITION_SIDE_NONE and entry_price > FLOAT_COMPARISON_EPSILON and current_atr > FLOAT_COMPARISON_EPSILON:
            logger.warning("!!! Trade Logic: Checking active position SL/TP based on LAST CLOSED PRICE (LOOP CHECK - UNSAFE & INACCURATE FOR SCALPING) !!!")
            sl_triggered, tp_triggered = False, False
            stop_price, profit_price = 0.0, 0.0
            sl_distance = current_atr * sl_atr_mult
            tp_distance = current_atr * tp_atr_mult

            if position_side == POSITION_SIDE_LONG:
                stop_price = entry_price - sl_distance
                profit_price = entry_price + tp_distance
                logger.debug(f"SL/TP Check (Long): Entry={entry_price_str}, CurrentClose={current_price_str}, Target SL={stop_price:.{price_precision_digits}f}, Target TP={profit_price:.{price_precision_digits}f}")
                if current_price <= stop_price: sl_triggered = True
                if current_price >= profit_price: tp_triggered = True
            elif position_side == POSITION_SIDE_SHORT:
                stop_price = entry_price + sl_distance
                profit_price = entry_price - tp_distance
                logger.debug(f"SL/TP Check (Short): Entry={entry_price_str}, CurrentClose={current_price_str}, Target SL={stop_price:.{price_precision_digits}f}, Target TP={profit_price:.{price_precision_digits}f}")
                if current_price >= stop_price: sl_triggered = True
                if current_price <= profit_price: tp_triggered = True

            # --- Execute Close if SL/TP Triggered by this Inaccurate Loop Check ---
            if sl_triggered:
                reason = f"SL Loop @ {current_price_str} (Target ~ {stop_price:.{price_precision_digits}f})"
                logger.critical(f"*** TRADE EXIT (LOOP CHECK): STOP-LOSS TRIGGERED! ({reason}). Closing {position_side}. ***")
                logger.critical("!!! THIS SL TRIGGER IS BASED ON DELAYED CHECK - ACTUAL LOSS MAY BE SIGNIFICANTLY HIGHER !!!")
                # Pass the 'position' dict we got at the start of the cycle for info, but close_position re-validates
                close_result = close_position(exchange, symbol, position, reason="SL (Loop Check)")
                if close_result: action_taken_this_cycle = True
                # Exit logic for this cycle after attempting close, regardless of success
                # because the condition was met. Prevents trying to enter immediately after SL.
                logger.info(f"========== Cycle Check End (After SL Loop Attempt): {symbol} ==========\n")
                return # Stop further logic this cycle
            elif tp_triggered:
                reason = f"TP Loop @ {current_price_str} (Target ~ {profit_price:.{price_precision_digits}f})"
                logger.success(f"*** TRADE EXIT (LOOP CHECK): TAKE-PROFIT TRIGGERED! ({reason}). Closing {position_side}. ***")
                logger.warning("!!! TAKE PROFIT VIA LOOP CHECK MAY MISS BETTER FILLS OR EXPERIENCE SLIPPAGE !!!")
                close_result = close_position(exchange, symbol, position, reason="TP (Loop Check)")
                if close_result: action_taken_this_cycle = True
                # Exit logic for this cycle after attempting close
                logger.info(f"========== Cycle Check End (After TP Loop Attempt): {symbol} ==========\n")
                return # Stop further logic this cycle
            else:
                 logger.debug("Trade Logic: Position SL/TP not triggered based on last close price (Loop Check).")
        elif position_side != POSITION_SIDE_NONE:
             # Log if position exists but SL/TP check couldn't run or is disabled
             if ENABLE_LOOP_SLTP_CHECK:
                 logger.warning(f"Trade Logic: Position ({position_side}) exists but loop-based SL/TP check skipped (Entry={entry_price_str}, ATR={atr_str}).")
             else:
                 logger.info("Trade Logic: Holding position. Loop-based SL/TP check is DISABLED (Recommended). Manage SL/TP via exchange orders.")


        # === 6. Check for Entry Signals ===
        # Only proceed if no SL/TP exit was triggered above and no other action was taken yet
        if not action_taken_this_cycle:
            logger.debug("Trade Logic: Checking entry signals...")

            # --- Define Confirmation Conditions ---
            # Order Book Confirmation (only if thresholds are set and data is available)
            ob_check_active = ORDER_BOOK_RATIO_THRESHOLD_LONG > 0 or ORDER_BOOK_RATIO_THRESHOLD_SHORT < float('inf')
            ob_available = ob_data is not None and bid_ask_ratio is not None
            # Long OB check: Pass if OB disabled (threshold=0), or if OB enabled and ratio meets threshold
            passes_long_ob = (ORDER_BOOK_RATIO_THRESHOLD_LONG <= 0) or \
                             (ob_available and bid_ask_ratio >= ORDER_BOOK_RATIO_THRESHOLD_LONG)
            # Short OB check: Pass if OB disabled (threshold=inf), or if OB enabled and ratio meets threshold
            passes_short_ob = (ORDER_BOOK_RATIO_THRESHOLD_SHORT >= float('inf')) or \
                              (ob_available and bid_ask_ratio <= ORDER_BOOK_RATIO_THRESHOLD_SHORT)
            ob_log_str = f"Ratio={bid_ask_ratio_str} (LThr={ORDER_BOOK_RATIO_THRESHOLD_LONG}, SThr={ORDER_BOOK_RATIO_THRESHOLD_SHORT}, Active={ob_check_active}, Avail={ob_available})"

            # Volume Confirmation (only if required and data is available)
            volume_check_required = REQUIRE_VOLUME_SPIKE_FOR_ENTRY
            volume_available = volume_ratio is not None # Need ratio to check spike
            passes_volume = not volume_check_required or (volume_available and volume_spike)
            vol_log_str = f"Ratio={vol_ratio_str}, Spike={vol_spike_str} (Required={volume_check_required}, Avail={volume_available})"


            # --- Long Entry Condition ---
            enter_long = (
                position_side != POSITION_SIDE_LONG and          # Prevent adding to existing long
                primary_long_signal and                          # Primary ST must have flipped Long THIS candle
                confirm_trend_up and                             # Confirmation ST trend must be Up (agreement)
                passes_long_ob and                               # OB condition met (or not required/disabled/unavailable)
                passes_volume                                    # Volume condition met (or not required/unavailable)
            )
            logger.debug(f"Entry Check (Long): PrimaryFlip={primary_long_signal}, ConfirmTrendUp={confirm_trend_up}, "
                         f"OB OK={passes_long_ob} [{ob_log_str}], Vol OK={passes_volume} [{vol_log_str}] "
                         f"=> Enter Long = {enter_long}")

            # --- Short Entry Condition ---
            enter_short = (
                position_side != POSITION_SIDE_SHORT and         # Prevent adding to existing short
                primary_short_signal and                         # Primary ST must have flipped Short THIS candle
                not confirm_trend_up and                         # Confirmation ST trend must be Down (agreement)
                passes_short_ob and                              # OB condition met (or not required/disabled/unavailable)
                passes_volume                                    # Volume condition met (or not required/unavailable)
            )
            logger.debug(f"Entry Check (Short): PrimaryFlip={primary_short_signal}, ConfirmTrendDown={not confirm_trend_up}, "
                          f"OB OK={passes_short_ob} [{ob_log_str}], Vol OK={passes_volume} [{vol_log_str}] "
                          f"=> Enter Short = {enter_short}")


            # === 7. Execute Entry / Reversal Actions ===
            if enter_long:
                logger.success(f"*** TRADE SIGNAL: CONFIRMED LONG ENTRY SIGNAL for {symbol} at ~{current_price_str} ***")
                if position_side == POSITION_SIDE_SHORT:
                    # --- Reversal: Close Short, then Enter Long ---
                    logger.warning("Trade Action: LONG signal received while SHORT. Attempting REVERSAL.")
                    # 1. Close the existing short position
                    logger.info("Reversal Step 1: Closing existing SHORT position...")
                    # Pass the 'position' dict from the start of the cycle for info, close_position re-validates
                    close_result = close_position(exchange, symbol, position, reason="Reversal to Long")
                    action_taken_this_cycle = True # Attempted an action
                    if close_result:
                        logger.info(f"Reversal Step 1: Close SHORT order placed. Waiting {POST_CLOSE_DELAY_SECONDS}s for settlement...")
                        time.sleep(POST_CLOSE_DELAY_SECONDS) # Allow time for closure to process/API state update
                        # 2. Re-check position state AFTER attempting close to CONFIRM closure
                        logger.info("Reversal Step 2: Re-checking position status after close attempt...")
                        current_pos_after_close = get_current_position(exchange, symbol)
                        if current_pos_after_close['side'] == POSITION_SIDE_NONE:
                            logger.success("Reversal Step 2: Confirmed SHORT position closed.")
                            # 3. Enter the new long position
                            logger.info("Reversal Step 3: Proceeding with LONG entry.")
                            # Pass current_atr which was calculated at the start of the cycle
                            place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                        else:
                            pos_after_close_side = current_pos_after_close['side']
                            pos_after_close_qty = current_pos_after_close['qty']
                            pos_after_close_qty_str = f"{pos_after_close_qty:.{qty_precision_digits}f}"
                            logger.error(f"Reversal Step 2: FAILED TO CONFIRM SHORT position closure after reversal signal (Still showing: {pos_after_close_side} Qty={pos_after_close_qty_str}). Skipping LONG entry this cycle.")
                            send_sms_alert(f"[{symbol.split('/')[0]}] REVERSAL FAIL: Could not confirm SHORT close. LONG entry skipped.")
                    else:
                         logger.error("Reversal Step 1: Failed to place order to close SHORT position. Skipping LONG entry.")
                         send_sms_alert(f"[{symbol.split('/')[0]}] REVERSAL FAIL: Failed to place SHORT close order.")

                elif position_side == POSITION_SIDE_NONE:
                    # --- New Entry: Enter Long ---
                    logger.info("Trade Action: Entering NEW LONG position.")
                    place_result = place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                    if place_result: action_taken_this_cycle = True
                # else: # Already Long, signal ignored due to position_side check in enter_long condition
                     # logger.debug("Trade Action: Signal LONG ignored (already long).")


            elif enter_short:
                logger.success(f"*** TRADE SIGNAL: CONFIRMED SHORT ENTRY SIGNAL for {symbol} at ~{current_price_str} ***")
                if position_side == POSITION_SIDE_LONG:
                    # --- Reversal: Close Long, then Enter Short ---
                    logger.warning("Trade Action: SHORT signal received while LONG. Attempting REVERSAL.")
                    # 1. Close the existing long position
                    logger.info("Reversal Step 1: Closing existing LONG position...")
                    close_result = close_position(exchange, symbol, position, reason="Reversal to Short")
                    action_taken_this_cycle = True # Attempted an action
                    if close_result:
                        logger.info(f"Reversal Step 1: Close LONG order placed. Waiting {POST_CLOSE_DELAY_SECONDS}s for settlement...")
                        time.sleep(POST_CLOSE_DELAY_SECONDS)
                        # 2. Re-check position state AFTER attempting close to CONFIRM closure
                        logger.info("Reversal Step 2: Re-checking position status after close attempt...")
                        current_pos_after_close = get_current_position(exchange, symbol)
                        if current_pos_after_close['side'] == POSITION_SIDE_NONE:
                            logger.success("Reversal Step 2: Confirmed LONG position closed.")
                            # 3. Enter the new short position
                            logger.info("Reversal Step 3: Proceeding with SHORT entry.")
                            place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                        else:
                            pos_after_close_side = current_pos_after_close['side']
                            pos_after_close_qty = current_pos_after_close['qty']
                            pos_after_close_qty_str = f"{pos_after_close_qty:.{qty_precision_digits}f}"
                            logger.error(f"Reversal Step 2: FAILED TO CONFIRM LONG position closure after reversal signal (Still showing: {pos_after_close_side} Qty={pos_after_close_qty_str}). Skipping SHORT entry this cycle.")
                            send_sms_alert(f"[{symbol.split('/')[0]}] REVERSAL FAIL: Could not confirm LONG close. SHORT entry skipped.")
                    else:
                        logger.error("Reversal Step 1: Failed to place order to close LONG position. Skipping SHORT entry.")
                        send_sms_alert(f"[{symbol.split('/')[0]}] REVERSAL FAIL: Failed to place LONG close order.")

                elif position_side == POSITION_SIDE_NONE:
                    # --- New Entry: Enter Short ---
                    logger.info("Trade Action: Entering NEW SHORT position.")
                    place_result = place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage, max_order_cap, margin_buffer)
                    if place_result: action_taken_this_cycle = True
                # else: # Already Short, signal ignored due to position_side check in enter_short condition
                     # logger.debug("Trade Action: Signal SHORT ignored (already short).")

        # --- Log if no action taken ---
        if not action_taken_this_cycle:
            if position_side == POSITION_SIDE_NONE:
                logger.info("Trade Logic: No confirmed entry signal and no active position. Holding cash.")
            else:
                 # If holding a position, remind about manual/exchange SL/TP management
                 logger.info(f"Trade Logic: Holding {position_side} position. No new entry or exit signal this cycle. Ensure exchange SL/TP orders are active if required.")

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
# Flag to prevent multiple shutdown attempts
shutdown_initiated = False
# Global flag to control the main loop, modified by signal handler
run_bot: bool = True

def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str], signum=None, frame=None) -> None:
    """
    Attempts to close any open position for the given symbol using robust checks
    and confirmation before exiting. Sends SMS alerts about the shutdown process.
    Designed to be called by signal handlers or at the end of main loop.
    """
    global shutdown_initiated, run_bot
    if shutdown_initiated:
        logger.warning("Shutdown already in progress, ignoring duplicate request.")
        return
    shutdown_initiated = True
    run_bot = False # Signal main loop to stop

    signal_name = f"Signal {signal.strsignal(signum)}" if signum and isinstance(signum, int) and signum in signal.Signals.__members__.values() else "Process End" # Get signal name if available
    logger.warning(f"Shutdown requested via {signal_name}. Initiating graceful exit sequence...")
    market_base = symbol.split('/')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown requested ({signal_name}). Attempting to close open position if any.")

    if not exchange or not symbol:
        logger.warning("Graceful Shutdown: Exchange not initialized or symbol not set. Cannot check or close position.")
        logger.info("--- Scalping Bot Shutdown Sequence Complete (No Position Check) ---")
        sys.exit(0) # Exit cleanly if nothing to check

    try:
        logger.info(f"Graceful Shutdown: Checking for active position for {symbol} using get_current_position...")
        # Use the robust position check function one last time
        position = get_current_position(exchange, symbol)

        # Get market precision for logging
        market = None
        qty_precision_digits = 8 # Default
        price_precision_digits = 4 # Default
        try:
            market = exchange.market(symbol)
            qty_precision_digits = market.get('precision', {}).get('amount', 8)
            price_precision_digits = market.get('precision', {}).get('price', 4)
        except Exception as market_err:
            logger.warning(f"Graceful Shutdown: Could not get market precision for {symbol}: {market_err}. Using defaults.")


        if position and position.get('side') != POSITION_SIDE_NONE:
            pos_side = position['side']
            pos_qty = position['qty']
            pos_entry = position['entry_price']
            pos_qty_str = f"{pos_qty:.{qty_precision_digits}f}"
            pos_entry_str = f"{pos_entry:.{price_precision_digits}f}" if pos_entry > FLOAT_COMPARISON_EPSILON else "N/A"
            logger.warning(f"Graceful Shutdown: Active {pos_side} position found (Qty={pos_qty_str} {symbol} @ Entry={pos_entry_str}). Attempting to close...")
            # Pass the freshly checked 'position' dict to the close function
            close_result = close_position(exchange, symbol, position, reason="Shutdown")

            if close_result:
                logger.info("Graceful Shutdown: Close order placed. Waiting briefly for exchange processing...")
                # Wait a bit longer on shutdown to increase chance of confirmation
                time.sleep(POST_CLOSE_DELAY_SECONDS * 2) # e.g., 6 seconds
                # --- Final Confirmation Check ---
                logger.info("Graceful Shutdown: Performing final position check...")
                final_pos_check = get_current_position(exchange, symbol)
                if final_pos_check and final_pos_check.get('side') == POSITION_SIDE_NONE:
                     logger.success("Graceful Shutdown: Position successfully confirmed CLOSED.")
                     send_sms_alert(f"[{market_base}] Position confirmed CLOSED during shutdown.")
                else:
                     final_side = final_pos_check.get('side', 'N/A')
                     final_qty = final_pos_check.get('qty', 0.0)
                     final_qty_str = f"{final_qty:.{qty_precision_digits}f}"
                     logger.error(f"Graceful Shutdown: FAILED TO CONFIRM position closure after attempting close. Final check showed: {final_side}, Qty: {final_qty_str}")
                     logger.error("!!! MANUAL INTERVENTION REQUIRED TO CHECK/CLOSE POSITION !!!")
                     send_sms_alert(f"[{market_base}] ERROR: Failed to confirm position closure on shutdown! Final state: {final_side} Qty={final_qty_str}. MANUAL CHECK REQUIRED!")
            else:
                 # close_position failed to even place the order
                 logger.error("Graceful Shutdown: Failed to place the close order for the active position.")
                 logger.error("!!! MANUAL INTERVENTION REQUIRED TO CHECK/CLOSE POSITION !!!")
                 send_sms_alert(f"[{market_base}] ERROR: Failed to PLACE close order during shutdown. MANUAL CHECK REQUIRED!")

        else:
            # No active position found during the check
            logger.info(f"Graceful Shutdown: No active position found for {symbol}. No close action needed.")
            send_sms_alert(f"[{market_base}] No active position found during shutdown.")

    except Exception as e:
        # Catch errors during the shutdown process itself (e.g., API errors during final checks)
        logger.error(f"Graceful Shutdown: An error occurred during the shutdown process: {e}")
        logger.debug(traceback.format_exc())
        logger.error("!!! POTENTIAL UNCLOSED POSITION - MANUAL CHECK REQUIRED !!!")
        send_sms_alert(f"[{market_base}] Error during graceful shutdown sequence: {type(e).__name__}. Check logs & position MANUALLY!")

    logger.info("--- Scalping Bot Shutdown Sequence Complete ---")
    sys.exit(0) # Ensure process exits after shutdown


# --- Main Execution ---
def main() -> None:
    """
    Main function to initialize the exchange connection, set up parameters,
    validate symbol/leverage, run the main trading loop, and handle shutdown.
    """
    global run_bot # Allow modification by signal handler

    bot_start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"--- Bybit Scalping Bot Initializing v{BOT_VERSION} ({bot_start_time_str}) ---")
    logger.warning("--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---")
    logger.warning("--- ENSURE ALL PARAMETERS IN .env ARE CORRECTLY SET & TESTED ---")
    if ENABLE_LOOP_SLTP_CHECK:
        logger.critical("--- *** CRITICAL WARNING: LOOP-BASED SL/TP IS ACTIVE - UNSAFE & INACCURATE FOR LIVE SCALPING *** ---")
        logger.critical("--- *** DISABLE THIS (ENABLE_LOOP_SLTP_CHECK=false) AND IMPLEMENT EXCHANGE-NATIVE CONDITIONAL ORDERS FOR SAFE OPERATION *** ---")
    else:
        logger.info("--- INFO: Loop-based SL/TP check is DISABLED (Recommended for safety). Use exchange-native orders for SL/TP. ---")


    # Initialize exchange and symbol as None initially
    exchange: Optional[ccxt.Exchange] = None
    symbol: Optional[str] = None # Will hold the validated symbol
    market_base: str = "Bot"     # For SMS alerts before symbol is set
    cycle_count: int = 0

    # --- Setup Signal Handlers ---
    # Use lambda to capture the *current* state of exchange and symbol when the signal occurs
    # This relies on exchange and symbol being assigned in the main scope before the loop starts
    def signal_handler(signum, frame):
        graceful_shutdown(exchange, symbol, signum, frame)

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals (e.g., from systemd or kill)

    try:
        # === Initialize Exchange ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Bot exiting due to exchange initialization failure.")
            # SMS alert already sent by initialize_exchange on failure
            return # Stop execution

        # === Setup Trading Symbol and Leverage ===
        # Prompt user or use default, then validate
        symbol_to_validate: Optional[str] = None
        try:
            # Provide default within the prompt itself for clarity
            raw_symbol_input = input(f"Enter trading symbol (e.g., BTC/USDT:USDT) or press Enter for default [{DEFAULT_SYMBOL}]: ").strip()
            # Standardize to upper case only if input was provided
            symbol_to_validate = raw_symbol_input.upper() if raw_symbol_input else DEFAULT_SYMBOL

            logger.info(f"Validating symbol: {symbol_to_validate}...")
            market = exchange.market(symbol_to_validate) # Throws BadSymbol if invalid
            symbol = market['symbol'] # Use the CCXT canonical symbol format
            market_base = symbol.split('/')[0] # Update for SMS alerts

            # Verify it's a contract market (swap/future) and linear (USDT-margined)
            market_type = market.get('type', 'N/A')
            is_contract = market.get('contract', False)
            is_linear = market.get('linear', False) # Check if it's USDT margined (linear)

            if not is_contract or market.get('spot', False): # Double check it's not spot
                logger.critical(f"Configuration Error: Symbol '{symbol}' is not a valid SWAP or FUTURES market on {exchange.id}. Type: {market_type}")
                send_sms_alert(f"[{market_base}] CRITICAL: Symbol {symbol} is not a contract market. Exiting.")
                return
            if not is_linear:
                logger.critical(f"Configuration Error: Symbol '{symbol}' is not a LINEAR (USDT-margined) contract. Type: {market_type}. This bot requires USDT-margined contracts.")
                send_sms_alert(f"[{market_base}] CRITICAL: Symbol {symbol} is not USDT-margined. Exiting.")
                return

            logger.info(f"Successfully validated symbol: {symbol} (Type: {market_type}, Linear: {is_linear})")

            # Set leverage for the validated symbol
            if not set_leverage(exchange, symbol, DEFAULT_LEVERAGE):
                logger.critical(f"Leverage setup failed for {symbol} after retries. Cannot proceed safely. Exiting.")
                # SMS alert potentially sent by set_leverage on final failure
                # Send one here explicitly if set_leverage didn't or might not have
                # Check return value of send_sms_alert to see if it failed too
                if not send_sms_alert(f"[{market_base}] CRITICAL: Leverage setup FAILED for {symbol}. Exiting."):
                     logger.warning("Failed to send critical SMS alert about leverage failure.")
                return
            logger.success(f"Leverage confirmed/set to {DEFAULT_LEVERAGE}x for {symbol}.")

        except (ccxt.BadSymbol, KeyError) as e:
            logger.critical(f"Configuration Error: Invalid or unsupported symbol '{symbol_to_validate}'. Error: {e}")
            send_sms_alert(f"[{market_base if symbol_to_validate else 'Bot'}] CRITICAL: Invalid symbol '{symbol_to_validate}'. Exiting.")
            return
        except EOFError: # Handle case where input stream is closed (e.g., running non-interactively without piping input)
            logger.critical("Configuration Error: Could not read symbol input (EOF). Ensure running interactively or provide symbol via .env.")
            send_sms_alert(f"[{market_base if symbol_to_validate else 'Bot'}] CRITICAL: Failed to read symbol input (EOF). Exiting.")
            return
        except Exception as e:
            logger.critical(f"Configuration Error: Unexpected error during symbol/leverage setup: {e}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base if symbol_to_validate else 'Bot'}] CRITICAL: Symbol/Leverage setup failed unexpectedly. Exiting.")
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
            logger.warning(f"TP Calculation: {ATR_TAKE_PROFIT_MULTIPLIER} * ATR({ATR_CALCULATION_PERIOD}) (Via UNSAFE Loop Check)")
        else:
             logger.info(f"TP Calculation: Loop Check Disabled (TP must be managed via exchange orders)")
        logger.info(f"Primary Supertrend: Length={DEFAULT_ST_ATR_LENGTH}, Multiplier={DEFAULT_ST_MULTIPLIER}")
        logger.info(f"Confirmation Supertrend: Length={CONFIRM_ST_ATR_LENGTH}, Multiplier={CONFIRM_ST_MULTIPLIER}")
        logger.info(f"Volume Analysis: MA={VOLUME_MA_PERIOD}, Spike > {VOLUME_SPIKE_THRESHOLD}x MA (Required for Entry: {REQUIRE_VOLUME_SPIKE_FOR_ENTRY})")
        logger.info(f"Order Book Analysis: Depth={ORDER_BOOK_DEPTH}, Long Ratio >= {ORDER_BOOK_RATIO_THRESHOLD_LONG}, Short Ratio <= {ORDER_BOOK_RATIO_THRESHOLD_SHORT} (Fetch Per Cycle: {FETCH_ORDER_BOOK_PER_CYCLE})")
        logger.info(f"Margin Check Buffer: {REQUIRED_MARGIN_BUFFER:.1%} (Requires {(REQUIRED_MARGIN_BUFFER-1)*100:.1f}% extra free margin)")
        logger.info(f"Check Interval (Sleep): {DEFAULT_SLEEP_SECONDS} seconds")
        logger.info(f"SMS Alerts Enabled: {ENABLE_SMS_ALERTS}, Recipient: {'Set' if SMS_RECIPIENT_NUMBER else 'Not Set'}, Timeout: {SMS_TIMEOUT_SECONDS}s")
        logger.warning("ACTION REQUIRED: Review this configuration carefully. Does it align with your strategy and risk tolerance?")
        if ENABLE_LOOP_SLTP_CHECK:
            logger.critical("RISK WARNING: Loop-based SL/TP check is ACTIVE. This is HIGHLY INACCURATE & UNSAFE for live scalping. DISABLE IT FOR REAL TRADING.")
        logger.info(f"Current Logging Level: {logging.getLevelName(logger.level)}")
        logger.info("------------------------------------------")
        # Send startup confirmation SMS
        sl_tp_status = "Loop SL/TP ACTIVE (UNSAFE!)" if ENABLE_LOOP_SLTP_CHECK else "Loop SL/TP Disabled"
        send_sms_alert(f"[ScalpBot] Configured for {symbol} ({DEFAULT_INTERVAL}, {DEFAULT_LEVERAGE}x). Risk={RISK_PER_TRADE_PERCENTAGE:.2%}. {sl_tp_status}. Starting main loop.")

        # === Main Trading Loop ===
        while run_bot:
            cycle_start_time = time.monotonic() # Use monotonic clock for interval timing
            cycle_count += 1
            logger.debug(f"--- Cycle {cycle_count} Start: {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

            try:
                # 1. Fetch Fresh Market Data
                # Determine required data length based on longest indicator period + buffer
                # Ensure data limit is reasonable, avoid excessively large requests
                data_limit = max(100, DEFAULT_ST_ATR_LENGTH, CONFIRM_ST_ATR_LENGTH,
                                 ATR_CALCULATION_PERIOD, VOLUME_MA_PERIOD) * 2 + API_FETCH_LIMIT_BUFFER
                data_limit = min(data_limit, 1000) # Cap limit to avoid excessive API usage/memory
                logger.debug(f"Data Fetch: Required candles calculation -> Limit={data_limit}")

                # Ensure symbol is not None before proceeding (should be guaranteed after setup)
                if symbol is None:
                    logger.critical("Main Loop: Symbol is None. This should not happen after setup. Stopping.")
                    run_bot = False
                    continue # Skip to next loop iteration (which will exit)

                df_market = get_market_data(exchange, symbol, DEFAULT_INTERVAL, limit=data_limit)

                # 2. Execute Trading Logic (only if data is valid)
                if df_market is not None and not df_market.empty:
                    # Pass a copy of the dataframe to prevent accidental modification across cycles if needed
                    trade_logic(
                        exchange=exchange, symbol=symbol, df=df_market.copy(), # Pass copy for safety
                        st_len=DEFAULT_ST_ATR_LENGTH, st_mult=DEFAULT_ST_MULTIPLIER,
                        cf_st_len=CONFIRM_ST_ATR_LENGTH, cf_st_mult=CONFIRM_ST_MULTIPLIER,
                        atr_len=ATR_CALCULATION_PERIOD, vol_ma_len=VOLUME_MA_PERIOD,
                        risk_pct=RISK_PER_TRADE_PERCENTAGE,
                        sl_atr_mult=ATR_STOP_LOSS_MULTIPLIER,
                        tp_atr_mult=ATR_TAKE_PROFIT_MULTIPLIER, # Only used by loop check
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
                send_sms_alert(f"[{market_base}] WARNING: Rate limit exceeded! Bot sleeping longer.")
                time.sleep(DEFAULT_SLEEP_SECONDS * 5) # Longer sleep after rate limit hit
            except ccxt.NetworkError as e:
                # Typically recoverable, log and continue after normal delay
                logger.warning(f"Network error in main loop (Cycle {cycle_count}): {e}. Will retry next cycle.")
                send_sms_alert(f"[{market_base}] WARNING: Network error encountered: {type(e).__name__}. Retrying...")
                # Let the standard sleep handle the delay, maybe add a small extra delay?
                time.sleep(RETRY_DELAY_SECONDS)
            except ccxt.ExchangeNotAvailable as e:
                # Exchange might be down for maintenance, sleep longer
                logger.error(f"Exchange not available (Cycle {cycle_count}): {e}. Sleeping for {DEFAULT_SLEEP_SECONDS * 10} seconds...")
                send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable! ({e}) Bot sleeping longer.")
                time.sleep(DEFAULT_SLEEP_SECONDS * 10)
            except ccxt.AuthenticationError as e:
                # Critical: API keys likely invalid or expired. Stop the bot.
                logger.critical(f"Authentication Error encountered in main loop (Cycle {cycle_count}): {e}. API keys may be invalid or permissions revoked. Stopping bot.")
                send_sms_alert(f"[{market_base}] CRITICAL: Authentication Error! Bot stopping NOW.")
                run_bot = False # Stop the main loop permanently
            except ccxt.InsufficientFunds as e:
                # Critical margin issue detected outside of normal checks. Stop for safety.
                logger.critical(f"Insufficient Funds Error encountered in main loop (Cycle {cycle_count}): {e}. Potential margin call risk. Stopping bot.")
                send_sms_alert(f"[{market_base}] CRITICAL: Insufficient Funds Error! Potential margin call. Bot stopping NOW.")
                run_bot = False # Stop the loop
            except ccxt.ExchangeError as e:
                # Catch other specific exchange errors (e.g., invalid order params, position errors not caught elsewhere)
                logger.error(f"Unhandled Exchange error in main loop (Cycle {cycle_count}): {e}.")
                logger.debug(traceback.format_exc()) # Log traceback for details
                send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs. Bot continues.")
                # Continue after normal delay, hoping it's temporary or related to a specific failed action handled within logic
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
        # User pressed Ctrl+C - handled by signal handler now
        logger.warning("KeyboardInterrupt caught in main try block (should be handled by signal handler).")
        # Signal handler should have set run_bot = False and called graceful_shutdown

    finally:
        # --- Bot Shutdown Sequence ---
        # This block executes if the loop finishes normally (run_bot becomes False)
        # or is stopped by an error NOT handled by signals.
        # If shutdown was initiated by a signal, graceful_shutdown would have already run and called sys.exit().
        # Check shutdown_initiated flag to avoid running the sequence twice if a signal occurred.
        if not shutdown_initiated:
            logger.info("--- Main loop terminated or error occurred. Initiating Final Shutdown Sequence ---")
            # Attempt graceful shutdown (close position) only if exchange/symbol were initialized
            graceful_shutdown(exchange, symbol) # Call directly if not triggered by signal
        else:
             # This case might be reached if Ctrl+C happened *during* the final block execution,
             # but graceful_shutdown should handle the sys.exit()
             logger.info("--- Shutdown was initiated by signal handler. Exiting. ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Print initial information to the console
    print(f"--- Bybit USDT Futures Scalping Bot v{BOT_VERSION} ---")
    print("INFO: Loading configuration from .env file (if exists).")
    print("INFO: Ensure API keys (BYBIT_API_KEY, BYBIT_API_SECRET) are correctly set in .env.")
    print("INFO: Ensure the selected SYMBOL is a USDT-margined (Linear) contract.")
    print("INFO: For SMS alerts, set ENABLE_SMS_ALERTS=true and SMS_RECIPIENT_NUMBER in .env.")
    print("INFO: SMS requires Termux environment with Termux:API installed ('pkg install termux-api') and SMS permission granted.")
    print("\033[93mWARNING: LIVE TRADING IS ACTIVE. USE EXTREME CAUTION AND TEST THOROUGHLY ON TESTNET FIRST.\033[0m")
    if ENABLE_LOOP_SLTP_CHECK:
        print("\033[91m\033[1mCRITICAL: LOOP-BASED SL/TP IS CURRENTLY ACTIVE IN CONFIGURATION.\033[0m")
        print("\033[91m\033[1mCRITICAL: This feature is HIGHLY INACCURATE & UNSAFE for live scalping.\033[0m")
        print("\033[91m\033[1mCRITICAL: DISABLE IT (ENABLE_LOOP_SLTP_CHECK=false in .env) for real trading and use exchange-native orders.\033[0m")
    else:
        print("INFO: Loop-based SL/TP check is DISABLED by default (recommended for safety).")
    print(f"INFO: Initial logging level set to: {logging.getLevelName(LOGGING_LEVEL)} (Change via LOGGING_LEVEL in .env)")
    print("-" * 60)

    # Start the main execution function
    main()
