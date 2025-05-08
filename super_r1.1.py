Okay, I've enhanced the provided Python script, addressing the `NameError`, significantly improving the Stop Loss (SL) and Take Profit (TP) handling (by adding the necessary structure and strong warnings about implementation), and incorporating various best practices and clarifications.

**Key Changes:**

1.  **`NameError` Fixed:** The `required_rows` calculation logic from `trade_logic` has been correctly placed within the `main` loop *before* fetching data, ensuring `data_limit` is calculated properly.
2.  **SL/TP Implementation Strategy:**
    *   **Emphasis on Exchange-Native Orders:** The `place_risked_market_order` function now includes the *crucial logic skeleton* and **strong warnings** for placing exchange-native Stop Loss and Take Profit orders immediately after the entry order. This is the *only* reliable way to manage SL/TP in HFT/scalping.
    *   **Loop-Based Check Downgraded:** The SL/TP check within the `trade_logic` loop is now explicitly marked as **informational only** and **unreliable for execution**. It will log potential breaches but *does not* trigger position closures. This prevents disastrous reliance on slow loop checks.
    *   **Calculation Integration:** SL and TP prices are calculated within `place_risked_market_order` using the entry estimate and ATR multipliers.
3.  **Improved Risk Calculation & Order Placement:**
    *   `place_risked_market_order` now fetches fresh (shallow) order book data just before placing the order for a better entry price estimate.
    *   Added fallback to ticker 'last' price if OB fetch fails.
    *   Clearer logging of risk calculation steps.
    *   Added a margin buffer check before placing the order.
4.  **Enhanced Logging:**
    *   More detailed debug messages.
    *   Clearer separation of logic cycles in logs.
    *   Improved formatting and added color (as in the original).
    *   Added `traceback.format_exc()` in more exception handlers for better debugging.
5.  **Robustness:**
    *   Added checks for valid ATR values before use.
    *   Improved handling of potential `None` values from API responses (e.g., balance, position details).
    *   More specific exception handling in the main loop (`RateLimitExceeded`, `ExchangeNotAvailable`, `AuthenticationError`).
6.  **Clarity and Comments:**
    *   Improved docstrings and comments throughout the code.
    *   Strengthened disclaimers and warnings about the high risks of scalping and the importance of parameter tuning and testing.
    *   Clarified the purpose of various parameters.
7.  **Dependencies:** Added an explicit check for `pandas_ta` installation at the start.

```python
#!/usr/bin/env python

"""
High-Frequency Trading Bot (Scalping) using Dual Supertrend, ATR, Volume, and Order Book Analysis.

Disclaimer:
- EXTREMELY HIGH RISK: Scalping bots operate in noisy, fast-moving markets.
  Risk of significant losses due to latency, slippage, fees, API issues,
  market volatility, and incorrect parameter tuning is substantial. USE AT YOUR OWN RISK.
- Parameter Sensitivity: Strategy performance is HIGHLY dependent on parameter
  tuning (Supertrend lengths/multipliers, ATR settings, volume thresholds, OB ratios).
  Requires extensive backtesting and optimization for specific market conditions.
  Defaults provided are purely illustrative and likely suboptimal.
- Exchange-Native SL/TP REQUIRED: The loop-based SL/TP check in this script is
  INFORMATIONAL ONLY and DANGEROUSLY UNRELIABLE for execution due to latency.
  You MUST implement exchange-native SL/TP orders immediately after entry.
- Rate Limits: Frequent API calls, especially for order book data, can easily
  exceed exchange rate limits, leading to temporary bans or degraded performance.
  Monitor usage carefully. Adjust FETCH_ORDER_BOOK_PER_CYCLE and SLEEP_SECONDS.
- Slippage: Market orders used for entry/exit are prone to slippage, meaning the
  fill price may differ significantly from the expected price, especially during
  volatility.
- Test Thoroughly: DO NOT RUN THIS BOT WITH REAL CAPITAL WITHOUT EXTENSIVE AND
  SUCCESSFUL TESTING ON A TESTNET OR DEMO ACCOUNT. Validate every component.
- No Guarantees: This script is provided for educational purposes only. There is
  NO guarantee of profit. Trading involves risk, and you can lose your entire investment.
- Code Errors: Bugs may exist. Review and understand the code fully before use.
"""

import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, Tuple, Optional

# Ensure pandas_ta is installed before importing other potentially dependent libraries
try:
    import pandas_ta as ta
except ImportError:
    print("CRITICAL: 'pandas_ta' library not found. Please install it: pip install pandas_ta", file=sys.stderr)
    sys.exit(1)

import ccxt
import pandas as pd
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()

# API Credentials (Fetch from .env file)
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# --- Scalping Specific Configuration ---
DEFAULT_SYMBOL = os.getenv("SYMBOL", "BTC/USDT:USDT")  # CCXT unified format (e.g., BTC/USDT:USDT for Bybit USDT Perpetual)
DEFAULT_INTERVAL = os.getenv("INTERVAL", "1m")  # Scalping timeframe (e.g., 1m, 3m, 5m)
DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", 10))  # CAUTION: High leverage amplifies gains AND losses
DEFAULT_SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", 10))  # Time between logic checks

# Risk Management (CRITICAL - TUNE CAREFULLY!)
RISK_PER_TRADE_PERCENTAGE = float(os.getenv("RISK_PER_TRADE_PERCENTAGE", 0.005))  # e.g., 0.005 = 0.5% of equity risked per trade
ATR_STOP_LOSS_MULTIPLIER = float(os.getenv("ATR_STOP_LOSS_MULTIPLIER", 1.5))  # Stop Loss distance = Multiplier * ATR
ATR_TAKE_PROFIT_MULTIPLIER = float(os.getenv("ATR_TAKE_PROFIT_MULTIPLIER", 2.0))  # Take Profit distance = Multiplier * ATR
MAX_ORDER_USDT_AMOUNT = float(os.getenv("MAX_ORDER_USDT_AMOUNT", 500.0))  # Max position size in USDT value (acts as a safety cap)

# Supertrend Indicator Parameters (Tune for chosen timeframe)
DEFAULT_ST_ATR_LENGTH = int(os.getenv("ST_ATR_LENGTH", 7))  # Primary Supertrend ATR period (shorter for scalping)
DEFAULT_ST_MULTIPLIER = float(os.getenv("ST_MULTIPLIER", 2.5)) # Primary Supertrend multiplier
CONFIRM_ST_ATR_LENGTH = int(os.getenv("CONFIRM_ST_ATR_LENGTH", 5)) # Confirmation Supertrend ATR period (even shorter)
CONFIRM_ST_MULTIPLIER = float(os.getenv("CONFIRM_ST_MULTIPLIER", 2.0)) # Confirmation Supertrend multiplier

# Volume Analysis Parameters
VOLUME_MA_PERIOD = int(os.getenv("VOLUME_MA_PERIOD", 20)) # Moving average period for volume
VOLUME_SPIKE_THRESHOLD = float(os.getenv("VOLUME_SPIKE_THRESHOLD", 1.5))  # Volume must be > X times its MA
REQUIRE_VOLUME_SPIKE_FOR_ENTRY = os.getenv("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "true").lower() == "true" # If True, volume spike is mandatory for entry

# Order Book Analysis Parameters
ORDER_BOOK_DEPTH = int(os.getenv("ORDER_BOOK_DEPTH", 10))  # How many levels deep to analyze bid/ask volume
ORDER_BOOK_RATIO_THRESHOLD_LONG = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_LONG", 1.2))  # Bid volume must be > X times Ask volume for long entry
ORDER_BOOK_RATIO_THRESHOLD_SHORT = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_SHORT", 0.8)) # Bid volume must be < X times Ask volume for short entry (Ask > 1/X * Bid)
FETCH_ORDER_BOOK_PER_CYCLE = os.getenv("FETCH_ORDER_BOOK_PER_CYCLE", "false").lower() == "true"  # <<< If True, fetches OB every cycle (more API calls); if False, fetches only when signal is near

# ATR Calculation Parameter (Used for SL/TP calculation base)
ATR_CALCULATION_PERIOD = int(os.getenv("ATR_CALCULATION_PERIOD", 14)) # Standard ATR period

# CCXT / API Parameters
DEFAULT_RECV_WINDOW = 10000 # Bybit specific: time window for request validity (ms)
ORDER_BOOK_FETCH_LIMIT = max(25, ORDER_BOOK_DEPTH)  # API often requires minimum fetch levels (e.g., 25, 50 for Bybit)
SHALLOW_OB_FETCH_DEPTH = 5  # Depth for quick best bid/ask check before placing order

# --- Constants ---
SIDE_BUY = "buy"
SIDE_SELL = "sell"
POSITION_SIDE_LONG = "Long"
POSITION_SIDE_SHORT = "Short"
POSITION_SIDE_NONE = "None"
QUOTE_CURRENCY = "USDT" # Assuming USDT pairs, adjust if needed
RETRY_COUNT = 3 # Number of retries for certain API calls
RETRY_DELAY_SECONDS = 1 # Short delay between retries

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,  # Default level: INFO. Change to logging.DEBUG for maximum verbosity.
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] # Output logs to console
)
logger = logging.getLogger(__name__)

# Define custom SUCCESS level
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self, message, *args, **kwargs) -> None:
    """Logs a message with SUCCESS level."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

logging.Logger.success = log_success

# Add color to log levels (might not work on all terminals)
logging.addLevelName(logging.DEBUG, f"\033[90m{logging.getLevelName(logging.DEBUG)}\033[0m") # Grey
logging.addLevelName(logging.INFO, f"\033[96m{logging.getLevelName(logging.INFO)}\033[0m") # Cyan
logging.addLevelName(logging.WARNING, f"\033[93m{logging.getLevelName(logging.WARNING)}\033[0m") # Yellow
logging.addLevelName(logging.ERROR, f"\033[91m{logging.getLevelName(logging.ERROR)}\033[0m") # Red
logging.addLevelName(logging.CRITICAL, f"\033[91m\033[1m{logging.getLevelName(logging.CRITICAL)}\033[0m") # Bold Red
logging.addLevelName(SUCCESS_LEVEL, f"\033[92m{logging.getLevelName(SUCCESS_LEVEL)}\033[0m") # Green

# Uncomment below to enable detailed DEBUG logging globally
# logging.getLogger().setLevel(logging.DEBUG)
# logger.debug("DEBUG logging enabled.")

# --- Exchange Initialization ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """
    Initializes and returns the CCXT Bybit exchange instance.
    Tests authentication by fetching balance.

    Returns:
        ccxt.Exchange | None: Initialized exchange instance or None on failure.
    """
    logger.info("Initializing CCXT Bybit exchange...")
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        logger.critical("CRITICAL: BYBIT_API_KEY or BYBIT_API_SECRET not found in .env file. Cannot proceed.")
        return None
    try:
        exchange = ccxt.bybit({
            "apiKey": BYBIT_API_KEY,
            "secret": BYBIT_API_SECRET,
            "enableRateLimit": True,  # Enable built-in rate limiter
            "options": {
                "defaultType": "swap",  # Use swap/perpetual futures
                "recvWindow": DEFAULT_RECV_WINDOW,
                # "adjustForTimeDifference": True, # Generally handled by ccxt now
                # 'enableTimeSync': True, # Usually default/handled by ccxt
            },
        })
        # Test connection and authentication
        logger.debug("Loading markets...")
        exchange.load_markets()
        logger.debug("Testing authentication by fetching balance...")
        # Use fetch_total_balance for USDT perpetuals on Bybit
        balance = exchange.fetch_total_balance()
        logger.debug(f"Balance fetch successful. {QUOTE_CURRENCY} total: {balance.get(QUOTE_CURRENCY, 'N/A')}")
        logger.success("CCXT Bybit Session Initialized Successfully.")
        logger.warning("<<< RUNNING IN LIVE FUTURES SCALPING MODE - EXTREME CAUTION! >>>")
        return exchange
    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed. Check API keys. Error: {e}")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error during exchange initialization: {e}")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange specific error during initialization: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during exchange initialization: {e}")
        logger.debug(traceback.format_exc())
    return None

# --- Indicator Calculation ---
def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: float, prefix: str = ""
) -> pd.DataFrame:
    """
    Calculates the Supertrend indicator using pandas_ta and adds columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
        length (int): The ATR period for Supertrend.
        multiplier (float): The ATR multiplier for Supertrend.
        prefix (str, optional): Prefix for the generated columns (e.g., "confirm_"). Defaults to "".

    Returns:
        pd.DataFrame: DataFrame with added Supertrend columns:
                      '[prefix]supertrend' (the ST line value),
                      '[prefix]trend' (boolean, True if trend is up),
                      '[prefix]st_long' (boolean, True if trend turned up),
                      '[prefix]st_short' (boolean, True if trend turned down').
    """
    col_prefix = f"{prefix}" if prefix else ""
    st_col_name = f"SUPERT_{length}_{multiplier}"
    st_trend_col = f"SUPERTd_{length}_{multiplier}" # Direction column
    st_long_col = f"SUPERTl_{length}_{multiplier}" # Long signal column
    st_short_col = f"SUPERTs_{length}_{multiplier}"# Short signal column

    target_cols = [
        f"{col_prefix}supertrend", f"{col_prefix}trend",
        f"{col_prefix}st_long", f"{col_prefix}st_short"
    ]

    # Check if input DataFrame is valid and has necessary columns
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low", "close"]):
        logger.warning(f"Input DataFrame is invalid or missing required columns for {col_prefix}Supertrend.")
        for col in target_cols: df[col] = pd.NA # Add NA columns if calculation fails
        return df
    if len(df) < length + 1: # Need length + 1 for shift comparison
        logger.warning(f"Insufficient data ({len(df)} rows) for {col_prefix}Supertrend length {length}.")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        logger.debug(f"Calculating {col_prefix}Supertrend (len={length}, mult={multiplier})...")
        # Calculate Supertrend using pandas_ta
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)

        # Verify that pandas_ta created the expected columns
        if st_col_name not in df.columns or st_trend_col not in df.columns \
           or st_long_col not in df.columns or st_short_col not in df.columns:
             raise KeyError(f"pandas_ta did not create expected columns: {st_col_name}, {st_trend_col}, etc.")

        # Rename and derive necessary columns
        df[f"{col_prefix}supertrend"] = df[st_col_name]
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1 # True if trend is up (direction = 1)

        # Pandas_ta >= 0.3.14b directly provides long/short signal columns (SUPERTl, SUPERTs)
        # These signal when the trend *flips* on that bar.
        # Convert the signal columns (which might contain 1.0 or NaN) to boolean
        df[f"{col_prefix}st_long"] = df[st_long_col].notna() & (df[st_long_col] > 0) # Check if not NaN and positive
        df[f"{col_prefix}st_short"] = df[st_short_col].notna() & (df[st_short_col] > 0)

        # Clean up intermediate columns created by pandas_ta
        # We keep the renamed target columns and drop the originals
        cols_to_drop = [st_col_name, st_trend_col, st_long_col, st_short_col]
         # Also drop potential intermediate columns if they exist (e.g., SUPERT_7_3.0_ma, etc.)
        for c in df.columns:
             if c.startswith("SUPERT_") and c not in target_cols:
                 cols_to_drop.append(c)

        df.drop(columns=list(set(cols_to_drop)), errors='ignore', inplace=True) # Use set to avoid duplicates

        # Log last trend value
        last_trend_idx = df[f'{col_prefix}trend'].last_valid_index()
        if last_trend_idx is not None:
            last_trend_val = 'Up' if df.loc[last_trend_idx, f'{col_prefix}trend'] else 'Down'
            logger.debug(f"Calculated {col_prefix}Supertrend. Last trend: {last_trend_val}")
        else:
            logger.debug(f"Calculated {col_prefix}Supertrend. Last trend: N/A (No valid data)")


    except KeyError as ke:
        logger.error(f"Error accessing expected column during {col_prefix}Supertrend calculation: {ke}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    except Exception as e:
        logger.error(f"Error calculating {col_prefix}Supertrend: {e}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Ensure target columns exist even on failure
    return df

# --- Volume and ATR Analysis ---
def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> Dict[str, Optional[float]]:
    """
    Calculates ATR and Volume Moving Average, identifies volume spikes.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len (int): Period for ATR calculation.
        vol_ma_len (int): Period for Volume Moving Average calculation.

    Returns:
        dict: A dictionary containing:
              'atr': Average True Range value for the last period (or None).
              'volume_ma': Volume Moving Average for the last period (or None).
              'last_volume': Actual volume for the last period (or None).
              'volume_ratio': Ratio of last volume to volume MA (or None).
    """
    results: Dict[str, Optional[float]] = {
        "atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None
    }
    required_cols = ["high", "low", "close", "volume"]
    if df is None or df.empty or not all(c in df.columns for c in required_cols):
        logger.warning("Insufficient data or missing columns for Volume/ATR analysis.")
        return results
    # Need at least `period` rows for ATR/MA, plus potentially one more depending on calculation method
    min_rows_required = max(atr_len, vol_ma_len) + 1
    if len(df) < min_rows_required:
        logger.warning(f"Insufficient data rows ({len(df)}) for ATR({atr_len})/VolMA({vol_ma_len}) calculation (Need >= {min_rows_required}).")
        return results

    try:
        # Calculate ATR using pandas_ta
        logger.debug(f"Calculating ATR({atr_len})...")
        df.ta.atr(length=atr_len, append=True)
        atr_col = f"ATRr_{atr_len}"
        if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]):
            results["atr"] = float(df[atr_col].iloc[-1]) # Ensure float
        else:
            logger.warning(f"Failed to calculate or find valid ATR({atr_len}). Column missing or last value NA.")
        # Clean up ATR column (always attempt drop, ignore errors if not found)
        df.drop(columns=[atr_col], errors='ignore', inplace=True)

        # Calculate Volume MA
        logger.debug(f"Calculating Volume MA({vol_ma_len})...")
        volume_ma_col = 'volume_ma'
        # Use min_periods=vol_ma_len to ensure a full window average for the last value
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=vol_ma_len).mean()
        if pd.notna(df[volume_ma_col].iloc[-1]):
            results["volume_ma"] = float(df[volume_ma_col].iloc[-1]) # Ensure float
            results["last_volume"] = float(df['volume'].iloc[-1]) # Ensure float

            # Calculate ratio, avoiding division by zero or near-zero
            if results["volume_ma"] is not None and results["volume_ma"] > 1e-9 and results["last_volume"] is not None:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            else:
                results["volume_ratio"] = None # Undefined ratio
        else:
             logger.warning(f"Failed to calculate or find valid Volume MA({vol_ma_len}). Last value NA (needs {vol_ma_len} periods).")
        # Clean up Volume MA column
        df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)

        # Log calculated values
        atr_val = results.get('atr')
        atr_str = f"{atr_val:.5f}" if atr_val is not None else 'N/A'
        logger.debug(f"ATR({atr_len}) calculation result: {atr_str}")

        last_vol_val = results.get('last_volume')
        vol_ma_val = results.get('volume_ma')
        vol_ratio_val = results.get('volume_ratio')
        last_vol_str = f"{last_vol_val:.2f}" if last_vol_val is not None else 'N/A'
        vol_ma_str = f"{vol_ma_val:.2f}" if vol_ma_val is not None else 'N/A'
        vol_ratio_str = f"{vol_ratio_val:.2f}" if vol_ratio_val is not None else 'N/A'
        logger.debug(f"Volume Analysis: Last={last_vol_str}, MA({vol_ma_len})={vol_ma_str}, Ratio={vol_ratio_str}")

    except Exception as e:
        logger.error(f"Error during Volume/ATR analysis: {e}")
        logger.debug(traceback.format_exc())
        # Reset results on error to avoid using stale/incorrect data
        results = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    return results

# --- Order Book Analysis ---
def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int
) -> Dict[str, Optional[float]]:
    """
    Fetches the L2 order book and analyzes bid/ask pressure and spread.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol (e.g., 'BTC/USDT:USDT').
        depth (int): The number of order book levels (price points) to analyze.

    Returns:
        dict: A dictionary containing:
              'bid_ask_ratio': Ratio of cumulative bid volume to ask volume in the specified depth (or None).
              'spread': Difference between best ask and best bid (or None).
              'best_bid': Highest bid price (or None).
              'best_ask': Lowest ask price (or None).
    """
    results: Dict[str, Optional[float]] = {
        "bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None
    }
    fetch_limit = max(ORDER_BOOK_FETCH_LIMIT, depth) # Use the larger of configured limit or requested depth
    logger.debug(f"Fetching L2 order book for {symbol} (Depth: {depth}, Fetch Limit: {fetch_limit})...")

    # Check exchange capability
    if not exchange.has.get('fetchL2OrderBook') and not exchange.has.get('fetchOrderBook'):
         logger.warning(f"Exchange {exchange.id} does not support fetchL2OrderBook or fetchOrderBook. Skipping order book analysis.")
         return results

    try:
        # Prefer fetchL2OrderBook if available, fallback to fetchOrderBook
        order_book = None
        if exchange.has['fetchL2OrderBook']:
            order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        elif exchange.has['fetchOrderBook']:
             logger.debug("fetchL2OrderBook not available, using fetchOrderBook.")
             order_book = exchange.fetch_order_book(symbol, limit=fetch_limit)
        else:
             # Should have been caught above, but double-check
             logger.warning("No order book fetching method available.")
             return results


        if not order_book or not order_book.get('bids') or not order_book.get('asks'):
            logger.warning(f"Incomplete or empty order book data received for {symbol}.")
            return results

        bids = order_book['bids'] # List of [price, volume]
        asks = order_book['asks'] # List of [price, volume]

        # Get best bid/ask and calculate spread
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        results["best_bid"] = float(best_bid) if best_bid > 0 else None
        results["best_ask"] = float(best_ask) if best_ask > 0 else None

        if results["best_bid"] is not None and results["best_ask"] is not None:
            results["spread"] = results["best_ask"] - results["best_bid"]
            logger.debug(f"Order Book: Best Bid={results['best_bid']:.4f}, Best Ask={results['best_ask']:.4f}, Spread={results['spread']:.4f}")
        else:
            logger.warning(f"Order Book: Best Bid/Ask not available to calculate spread.")
            results["spread"] = None # Ensure spread is None if bid/ask is missing

        # Calculate cumulative volume within the specified depth
        # Ensure values are floats before summing
        bid_volume_sum = sum(float(bid[1]) for bid in bids[:depth] if len(bid) > 1 and bid[1] is not None)
        ask_volume_sum = sum(float(ask[1]) for ask in asks[:depth] if len(ask) > 1 and ask[1] is not None)
        logger.debug(f"Order Book Analysis (Depth {depth}): Total Bid Vol={bid_volume_sum:.4f}, Total Ask Vol={ask_volume_sum:.4f}")

        # Calculate Bid/Ask Ratio, avoid division by zero
        if ask_volume_sum > 1e-9: # Use small threshold to avoid float issues
            results["bid_ask_ratio"] = bid_volume_sum / ask_volume_sum
            logger.debug(f"Order Book Bid/Ask Ratio: {results['bid_ask_ratio']:.3f}")
        else:
            logger.debug("Order Book Bid/Ask Ratio: Undefined (Ask volume is zero or negligible)")
            results["bid_ask_ratio"] = None # Explicitly set to None

    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching order book for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Exchange error fetching order book for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error analyzing order book for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return results

# --- Data Fetching ---
def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int = 100
) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV data from the exchange and returns it as a pandas DataFrame.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol.
        interval (str): The timeframe interval (e.g., '1m', '5m', '1h').
        limit (int): The number of candles to fetch.

    Returns:
        pd.DataFrame | None: DataFrame with OHLCV data indexed by timestamp, or None on failure.
    """
    if not exchange.has["fetchOHLCV"]:
        logger.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return None
    try:
        logger.debug(f"Fetching {limit} OHLCV candles for {symbol} (Interval: {interval})...")
        # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

        if not ohlcv:
            logger.warning(f"No OHLCV data returned for {symbol} ({interval}). Possible issue with symbol or exchange.")
            return None
        if len(ohlcv) < limit:
             logger.warning(f"Received fewer candles ({len(ohlcv)}) than requested ({limit}) for {symbol}.")

        # Convert to pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Convert timestamp to datetime objects (UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Convert other columns to numeric types, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Optional: Check for NaNs introduced by coercion
        if df.isnull().values.any():
            logger.warning(f"NaN values detected in OHLCV data for {symbol} after numeric conversion. Check source data.")
            # Consider dropping rows with NaNs if appropriate, but could shorten data unexpectedly
            # df.dropna(inplace=True)

        logger.debug(f"Successfully fetched {len(df)} OHLCV candles for {symbol}. Last candle time: {df.index[-1] if not df.empty else 'N/A'}")
        return df
    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching OHLCV data for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Exchange error fetching OHLCV data for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching market data for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return None

# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches the current open position details for a given symbol using fetch_position.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol.

    Returns:
        dict: A dictionary containing position details:
              'side': POSITION_SIDE_LONG, POSITION_SIDE_SHORT, or POSITION_SIDE_NONE.
              'qty': Position size (float, 0 if no position).
              'entry_price': Average entry price (float, 0 if no position).
              Returns default dict if error occurs or no position exists.
    """
    default_pos = {'side': POSITION_SIDE_NONE, 'qty': 0.0, 'entry_price': 0.0}
    if not exchange.has.get('fetchPosition'): # Check if fetchPosition is supported
        logger.warning(f"Exchange {exchange.id} does not support fetchPosition. Attempting fetch_positions instead.")
        return get_current_position_fallback(exchange, symbol) # Use fallback if needed

    try:
        logger.debug(f"Fetching current position for {symbol} using fetchPosition...")
        # fetchPosition is generally preferred for single symbol queries if available
        position_info = exchange.fetch_position(symbol)

        # fetchPosition returns a dictionary directly for the position
        if position_info and position_info.get('info'): # Check if structure seems valid
            contracts_val = position_info.get('contracts') # Size in base currency
            side_val = position_info.get('side') # 'long' or 'short'

            # Check for non-zero contracts, handle potential None or string types, use tolerance for float comparison
            if contracts_val is not None and side_val is not None:
                try:
                    qty = float(contracts_val)
                    # Check if position size is meaningfully non-zero (handle potential floating point inaccuracies)
                    if abs(qty) > 1e-9:
                        side = POSITION_SIDE_LONG if side_val == 'long' else POSITION_SIDE_SHORT
                        entry = float(position_info.get('entryPrice') or 0.0) # Handle potential None for entryPrice
                        logger.info(f"Found active position via fetchPosition: {side} {qty:.6f} {symbol} @ Entry={entry:.4f}")
                        return {'side': side, 'qty': qty, 'entry_price': entry}
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse position quantity from fetchPosition for {symbol}: {contracts_val}. Error: {e}")

        # If no meaningful position found
        logger.info(f"No active position found for {symbol} via fetchPosition.")
        return default_pos

    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching position for {symbol}: {e}")
    except ccxt.ExchangeError as e:
         # Some exchanges might throw specific errors if no position exists, treat as 'no position'
        if "position idx not exist" in str(e).lower(): # Example for Bybit
             logger.info(f"No active position found for {symbol} (Exchange reported no position).")
             return default_pos
        logger.warning(f"Exchange error fetching position for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching position for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return default_pos # Return default on any error


def get_current_position_fallback(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fallback: Fetches positions using fetch_positions (potentially less efficient).
    """
    default_pos = {'side': POSITION_SIDE_NONE, 'qty': 0.0, 'entry_price': 0.0}
    logger.debug(f"Using fetch_positions as fallback for {symbol}...")
    try:
        positions = exchange.fetch_positions([symbol]) # Pass symbol in a list

        for pos in positions:
            if pos.get('symbol') != symbol: continue
            contracts_val = pos.get('contracts')
            side_val = pos.get('side')

            if contracts_val is not None and side_val is not None:
                try:
                    qty = float(contracts_val)
                    if abs(qty) > 1e-9:
                        side = POSITION_SIDE_LONG if side_val == 'long' else POSITION_SIDE_SHORT
                        entry = float(pos.get('entryPrice') or 0.0)
                        logger.info(f"Found active position via fetch_positions: {side} {qty:.6f} {symbol} @ Entry={entry:.4f}")
                        return {'side': side, 'qty': qty, 'entry_price': entry}
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse position quantity from fetch_positions for {symbol}: {contracts_val}. Error: {e}")
                    continue

        logger.info(f"No active position found for {symbol} via fetch_positions.")
        return default_pos

    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching position (fallback) for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Exchange error fetching position (fallback) for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching position (fallback) for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return default_pos


# --- Leverage Setting ---
def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """
    Sets the leverage for a futures/swap symbol with retries.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol.
        leverage (int): The desired leverage level.

    Returns:
        bool: True if leverage was set successfully or already set, False otherwise.
    """
    logger.info(f"Attempting to set leverage to {leverage}x for {symbol}...")
    if not exchange.has.get('setLeverage'):
        logger.error(f"Exchange {exchange.id} does not support setLeverage method.")
        return False

    try:
        # First, check if the market is actually a future or swap
        market = exchange.market(symbol)
        if not market.get('swap', False) and not market.get('future', False):
            logger.error(f"Cannot set leverage for non-futures/non-swap market: {symbol}")
            return False
        logger.debug(f"Market type confirmed as swap/future for {symbol}.")
    except (ccxt.BadSymbol, KeyError) as e:
         logger.error(f"Failed to get market info for {symbol} during leverage check: {e}")
         return False # Cannot proceed if market info is unavailable
    except Exception as e:
         logger.error(f"Unexpected error during market check for leverage setting {symbol}: {e}")
         return False

    for attempt in range(RETRY_COUNT):
        try:
            # Attempt to set leverage
            # Some exchanges might require specific params (e.g., buyLeverage, sellLeverage for hedge mode)
            # Assuming one-way mode here.
            params = {} # Add params if needed for your exchange/mode
            response = exchange.set_leverage(leverage, symbol, params=params)
            logger.success(f"Leverage successfully set to {leverage}x for {symbol}. Response: {response}")
            return True
        except ccxt.ExchangeError as e:
            error_message_lower = str(e).lower()
            # Check if the error indicates leverage is already set (common response)
            # Adjust the string based on your exchange's specific message (e.g., Bybit uses "Leverage not modified")
            if "leverage not modified" in error_message_lower or "same leverage" in error_message_lower:
                logger.info(f"Leverage already set to {leverage}x for {symbol} (Exchange confirmed no change needed).")
                return True  # Treat as success

            # Handle other potentially retryable exchange errors
            logger.warning(f"Leverage set attempt {attempt + 1}/{RETRY_COUNT} failed for {symbol}: {e}")
            if attempt < RETRY_COUNT - 1:
                logger.debug(f"Retrying leverage set after {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"Failed to set leverage for {symbol} after {RETRY_COUNT} attempts due to exchange errors.")
                return False # Failed all retries for ExchangeError
        except ccxt.NetworkError as e:
             logger.warning(f"Network error on leverage set attempt {attempt + 1}/{RETRY_COUNT} for {symbol}: {e}")
             if attempt < RETRY_COUNT - 1:
                 logger.debug(f"Retrying leverage set after {RETRY_DELAY_SECONDS}s...")
                 time.sleep(RETRY_DELAY_SECONDS)
             else:
                 logger.error(f"Failed to set leverage for {symbol} after {RETRY_COUNT} attempts due to network errors.")
                 return False # Failed all retries for NetworkError
        except Exception as e:
            # Non-recoverable or unexpected errors
            logger.error(f"Unexpected error setting leverage for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return False  # Exit immediately on unexpected errors

    return False # Should not be reached if logic is correct, but acts as a fallback

# --- Close Position ---
def close_position(
    exchange: ccxt.Exchange, symbol: str, position: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Closes the current open position using a market order with reduce_only flag.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol.
        position (dict): The position dictionary obtained from get_current_position.

    Returns:
        dict | None: The order dictionary returned by CCXT if successful, otherwise None.
    """
    position_side = position.get('side', POSITION_SIDE_NONE)
    position_qty = position.get('qty', 0.0)

    if position_side == POSITION_SIDE_NONE or abs(position_qty) < 1e-9:
        logger.info(f"No active position found for {symbol}, no need to close.")
        return None

    # Determine the side needed to close the position
    side_to_close = SIDE_SELL if position_side == POSITION_SIDE_LONG else SIDE_BUY
    amount_to_close = abs(position_qty) # Always use positive quantity for order amount

    try:
        # Ensure amount is formatted correctly for the exchange
        amount_str = exchange.amount_to_precision(symbol, amount_to_close)
        amount_float = float(amount_str)

        if amount_float <= 0:
            logger.error(f"Calculated closing amount is zero or negative ({amount_str}) for {symbol}. Cannot close.")
            return None

        logger.warning(f"Attempting to CLOSE {position_side} position ({amount_str} {symbol}) "
                       f"via {side_to_close.upper()} MARKET order (reduce_only)...")

        # Set reduce_only parameter to ensure the order only closes the position
        # Check if exchange supports reduce_only in create_market_order params
        params = {}
        if exchange.has.get('reduceOnly'):
             params['reduce_only'] = True
             logger.debug("Using 'reduce_only': True parameter.")
        else:
             logger.warning("Exchange does not explicitly advertise 'reduce_only' support for market orders. Attempting anyway.")
             # Some exchanges might support it implicitly or require it elsewhere
             # Bybit USDT perpetuals support it in params for market orders.
             params['reduce_only'] = True # Keep attempting for Bybit

        # Create and place the market order
        order = exchange.create_market_order(symbol, side_to_close, amount_float, params=params)

        # TODO: Add a loop here to check if the position is actually closed by polling get_current_position
        # or checking order status, as market orders can sometimes fail partially or fully.

        logger.success(f"Position CLOSE order placed successfully for {symbol}. Order ID: {order.get('id', 'N/A')}")
        return order

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds to close position for {symbol}: {e}")
    except ccxt.OrderNotFound as e:
         logger.error(f"OrderNotFound error during closing (might indicate position closed already?): {e}")
         # Could potentially re-check position here
    except ccxt.NetworkError as e:
        logger.error(f"Network error closing position for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error closing position for {symbol}: {e}")
        # Log specific details if available
        logger.debug(f"Exchange error details: {e.args}")
    except Exception as e:
        logger.error(f"Unexpected error closing position for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return None

# --- Risk Calculation ---
def calculate_position_size(
    equity: float, risk_per_trade_pct: float, entry_price: float, stop_loss_price: float, leverage: int, quote_currency: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the position size in base currency units based on risk percentage,
    entry/stop prices, and calculates the required margin.

    Args:
        equity (float): Total account equity in quote currency (e.g., USDT).
        risk_per_trade_pct (float): The fraction of equity to risk (e.g., 0.01 for 1%).
        entry_price (float): The estimated entry price.
        stop_loss_price (float): The calculated stop loss price.
        leverage (int): The leverage being used.
        quote_currency (str): The quote currency (e.g., 'USDT').

    Returns:
        tuple[float | None, float | None]:
            - Position size in base currency (e.g., BTC quantity), or None if invalid.
            - Required margin in quote currency (e.g., USDT), or None if invalid.
    """
    logger.debug(f"Risk Calc Input: Equity={equity:.2f} {quote_currency}, Risk%={risk_per_trade_pct:.3%}, "
                 f"Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Leverage={leverage}x")

    if equity <= 0 or risk_per_trade_pct <= 0 or entry_price <= 0 or stop_loss_price <= 0:
        logger.error("Invalid input for position size calculation (Equity, Risk%, Entry, or SL <= 0).")
        return None, None

    # Ensure entry and stop loss are different
    price_difference = abs(entry_price - stop_loss_price)
    if price_difference < 1e-9: # Avoid division by zero or near-zero
        logger.error(f"Entry price ({entry_price:.4f}) and Stop Loss price ({stop_loss_price:.4f}) are too close or identical.")
        return None, None

    # Amount of quote currency to risk
    risk_amount_quote = equity * risk_per_trade_pct

    # Calculate position size in base currency units
    # Formula: Quantity = RiskAmount / PriceDifferencePerUnit
    quantity_base = risk_amount_quote / price_difference

    if quantity_base <= 0:
        logger.warning(f"Calculated quantity is zero or negative ({quantity_base:.8f}).")
        return None, None

    # Calculate the total value of the position in quote currency
    position_value_quote = quantity_base * entry_price

    # Calculate the required margin based on leverage
    # Add small epsilon to leverage denominator to prevent division by zero if leverage is accidentally 0
    required_margin_quote = position_value_quote / (leverage + 1e-9)

    logger.debug(f"Risk Calc Output: RiskAmt={risk_amount_quote:.2f} {quote_currency}, "
                 f"PriceDiff={price_difference:.4f} "
                 f"=> Quantity={quantity_base:.8f} (Base), "
                 f"Value={position_value_quote:.2f} {quote_currency}, "
                 f"Req. Margin={required_margin_quote:.2f} {quote_currency}")

    return quantity_base, required_margin_quote

# --- Place Order (Improved Entry Price Estimation & Risk Management) ---
def place_risked_market_order(
    exchange: ccxt.Exchange, symbol: str, side: str,
    risk_percentage: float, current_atr: float,
    sl_atr_multiplier: float, tp_atr_multiplier: float, # Added TP multiplier
    leverage: int
) -> Optional[Dict[str, Any]]:
    """
    Calculates position size based on risk/ATR-StopLoss, checks constraints,
    places a market order, and **attempts to place exchange-native SL/TP orders**.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol.
        side (str): 'buy' or 'sell'.
        risk_percentage (float): Risk per trade (e.g., 0.01 for 1%).
        current_atr (float): The current ATR value.
        sl_atr_multiplier (float): Multiplier for ATR to determine SL distance.
        tp_atr_multiplier (float): Multiplier for ATR to determine TP distance.
        leverage (int): Account leverage for the symbol.

    Returns:
        dict | None: The entry market order dictionary from CCXT if successful, otherwise None.
                     Failure to place SL/TP orders after entry will be logged as critical errors
                     but this function still returns the entry order dict if that succeeded.
    """
    logger.info(f"--- Attempting to Place {side.upper()} Market Order for {symbol} with Risk Management ---")

    if current_atr is None or current_atr <= 1e-9: # Check against small positive value
         logger.error(f"Invalid or near-zero ATR value ({current_atr}) provided for risk calculation. Cannot place order.")
         return None

    try:
        # 1. Get Account Balance/Equity
        logger.debug("Fetching account balance...")
        # Use fetch_total_balance for Bybit USDT perpetuals, adjust if needed for other exchanges/asset types
        # Fallback logic could be added here if fetch_total_balance isn't supported
        balance = exchange.fetch_total_balance()
        usdt_equity = balance.get(QUOTE_CURRENCY)

        if usdt_equity is None:
             logger.error(f"Could not determine total equity in {QUOTE_CURRENCY}. Check exchange method or currency name.")
             return None
        if usdt_equity <= 0:
             logger.error(f"Account equity in {QUOTE_CURRENCY} is zero or negative ({usdt_equity}). Cannot place order.")
             return None

        # We also need free balance to check margin availability
        free_balance = exchange.fetch_free_balance()
        usdt_free = free_balance.get(QUOTE_CURRENCY)
        if usdt_free is None:
             logger.error(f"Could not determine free balance in {QUOTE_CURRENCY}. Cannot place order.")
             return None
        # Allow free balance to be zero initially (might use existing position margin)

        logger.debug(f"Account Status: Equity = {usdt_equity:.2f} {QUOTE_CURRENCY}, Free = {usdt_free:.2f} {QUOTE_CURRENCY}")


        # 2. Get Market Info & Estimate Entry Price from Fresh Shallow Order Book
        logger.debug(f"Fetching market info for {symbol}...")
        market = exchange.market(symbol)
        min_qty = market.get('limits', {}).get('amount', {}).get('min')
        price_precision = market.get('precision', {}).get('price')
        amount_precision = market.get('precision', {}).get('amount')
        logger.debug(f"Market Info: Min Qty={min_qty}, Price Precision={price_precision}, Amount Precision={amount_precision}")


        logger.debug("Fetching shallow order book for fresh entry price estimate...")
        # Fetch a small, fresh OB for best bid/ask right before ordering
        ob_data = analyze_order_book(exchange, symbol, SHALLOW_OB_FETCH_DEPTH)
        entry_price_estimate = None

        # Estimate entry: For BUY market, expect to fill near ASK. For SELL market, expect near BID.
        if side == SIDE_BUY and ob_data.get('best_ask'):
            entry_price_estimate = ob_data['best_ask']
            logger.debug(f"Using best ASK {entry_price_estimate:.4f} as BUY entry estimate.")
        elif side == SIDE_SELL and ob_data.get('best_bid'):
            entry_price_estimate = ob_data['best_bid']
            logger.debug(f"Using best BID {entry_price_estimate:.4f} as SELL entry estimate.")
        else:
            # Fallback to ticker 'last' price if OB fetch failed or is incomplete
            logger.warning("Order book fetch for entry estimate failed or incomplete, using ticker 'last' price as fallback.")
            try:
                ticker = exchange.fetch_ticker(symbol)
                entry_price_estimate = ticker.get('last')
                if entry_price_estimate: logger.debug(f"Using ticker 'last' price {entry_price_estimate:.4f} as entry estimate.")
            except Exception as ticker_err:
                logger.error(f"Failed to fetch ticker as fallback: {ticker_err}")

        if entry_price_estimate is None or entry_price_estimate <= 0:
            logger.error(f"Could not determine a valid entry price estimate for {symbol}. Cannot place order.")
            return None

        # 3. Calculate Stop Loss & Take Profit Prices based on Estimated Entry and ATR
        stop_loss_distance = current_atr * sl_atr_multiplier
        take_profit_distance = current_atr * tp_atr_multiplier
        stop_loss_price = 0.0
        take_profit_price = 0.0

        if side == SIDE_BUY:
            stop_loss_price = entry_price_estimate - stop_loss_distance
            take_profit_price = entry_price_estimate + take_profit_distance
        elif side == SIDE_SELL:
            stop_loss_price = entry_price_estimate + stop_loss_distance
            take_profit_price = entry_price_estimate - take_profit_distance

        # Apply price precision to SL/TP prices
        if price_precision is not None:
            stop_loss_price = exchange.price_to_precision(symbol, stop_loss_price)
            take_profit_price = exchange.price_to_precision(symbol, take_profit_price)
            # Convert back to float after precision formatting
            stop_loss_price = float(stop_loss_price)
            take_profit_price = float(take_profit_price)

        # Ensure SL/TP prices are valid (e.g., positive and distinct from entry)
        if stop_loss_price <= 0 or take_profit_price <= 0:
             logger.error(f"Calculated SL ({stop_loss_price:.4f}) or TP ({take_profit_price:.4f}) price is invalid (<= 0). Cannot place order.")
             return None
        if abs(entry_price_estimate - stop_loss_price) < 1e-9 or abs(entry_price_estimate - take_profit_price) < 1e-9:
             logger.error(f"Calculated SL ({stop_loss_price:.4f}) or TP ({take_profit_price:.4f}) price is too close to entry estimate ({entry_price_estimate:.4f}). Cannot place order.")
             return None


        logger.info(f"Calculated SL/TP based on EntryEst={entry_price_estimate:.4f}, "
                    f"ATR({current_atr:.5f}): "
                    f"SL Price ~ {stop_loss_price:.4f} (Dist={stop_loss_distance:.4f}), "
                    f"TP Price ~ {take_profit_price:.4f} (Dist={take_profit_distance:.4f})")

        # 4. Calculate Position Size (Base Quantity) and Required Margin
        quantity_base, required_margin_quote = calculate_position_size(
            usdt_equity, risk_percentage, entry_price_estimate, stop_loss_price, leverage, QUOTE_CURRENCY
        )

        if quantity_base is None or required_margin_quote is None:
            logger.error("Position size calculation failed. Cannot place order.")
            return None # Error logged within calculate_position_size

        # 5. Apply Precision, Check Minimums, and Apply Max Cap
        logger.debug(f"Applying amount precision to calculated quantity: {quantity_base:.8f}")
        precise_qty_str = exchange.amount_to_precision(symbol, quantity_base)
        precise_qty = float(precise_qty_str)
        logger.debug(f"Precise quantity: {precise_qty_str} ({precise_qty})")

        # Check against minimum order size
        if min_qty is not None and precise_qty < min_qty:
            logger.warning(f"Calculated quantity {precise_qty_str} is below the minimum required ({min_qty}) for {symbol}. Skipping order.")
            return None

        # Check against MAX_ORDER_USDT_AMOUNT cap
        order_value_usdt_estimate = precise_qty * entry_price_estimate
        if order_value_usdt_estimate > MAX_ORDER_USDT_AMOUNT:
            logger.warning(f"Risk-calculated order value ({order_value_usdt_estimate:.2f} USDT) exceeds MAX Cap "
                           f"({MAX_ORDER_USDT_AMOUNT:.2f} USDT). Capping quantity.")
            # Calculate capped quantity based on max USDT amount
            capped_qty_base = MAX_ORDER_USDT_AMOUNT / entry_price_estimate
            precise_qty_str = exchange.amount_to_precision(symbol, capped_qty_base)
            precise_qty = float(precise_qty_str)
            logger.info(f"Quantity capped to: {precise_qty_str}")

            # Recalculate required margin for the capped quantity
            required_margin_quote = (precise_qty * entry_price_estimate) / (leverage + 1e-9)
            logger.info(f"Recalculated required margin for capped qty: {required_margin_quote:.2f} USDT")

            # Re-check against minimum order size after capping
            if min_qty is not None and precise_qty < min_qty:
                 logger.warning(f"Capped quantity {precise_qty_str} is still below the minimum required ({min_qty}). Skipping order.")
                 return None

        # Final check for valid quantity
        if precise_qty <= 0:
            logger.error(f"Final calculated order quantity is zero or negative ({precise_qty_str}). Cannot place order.")
            return None

        logger.info(f"Final Order Details: Side={side.upper()}, Quantity={precise_qty_str}, "
                    f"Est. Entry={entry_price_estimate:.4f}, Est. SL={stop_loss_price:.4f}, Est. TP={take_profit_price:.4f} "
                    f"Est. Value={precise_qty * entry_price_estimate:.2f} {QUOTE_CURRENCY}, "
                    f"Est. Req. Margin={required_margin_quote:.2f} {QUOTE_CURRENCY}")

        # 6. Check Available Margin (with a small buffer)
        margin_buffer_factor = 1.05 # Use 5% buffer for margin check
        required_margin_with_buffer = required_margin_quote * margin_buffer_factor
        if usdt_free < required_margin_with_buffer:
            logger.error(f"Insufficient FREE margin. Required with buffer: ~{required_margin_with_buffer:.2f} {QUOTE_CURRENCY}, "
                         f"Available: {usdt_free:.2f} {QUOTE_CURRENCY}. Cannot place order.")
            # Consider implementing logic here to reduce order size if possible and desired,
            # but for now, we just prevent the order.
            return None

        # 7. Place the Market Order
        logger.warning(f"*** PLACING {side.upper()} MARKET ORDER: {precise_qty_str} {symbol} ***")
        entry_order = exchange.create_market_order(symbol, side, precise_qty)

        # Note: Actual fill price might differ from estimate due to slippage.
        filled_qty = entry_order.get('filled') or entry_order.get('amount') # Prefer 'filled' if available
        avg_price = entry_order.get('average') or entry_order.get('price') # Prefer 'average' if available
        order_id = entry_order.get('id', 'N/A')
        order_status = entry_order.get('status', 'unknown')
        logger.success(f"MARKET Order Placement Attempted! ID: {order_id}, Status: {order_status} "
                       f"Size: ~{filled_qty}, Avg Fill Price: ~{avg_price}")

        # ------------------------------------------------------------------
        # !!! CRITICAL: IMPLEMENT EXCHANGE-NATIVE SL/TP ORDERS HERE !!!
        # ------------------------------------------------------------------
        # Immediately after placing the market order, you MUST place
        # corresponding Stop Loss and Take Profit orders on the exchange.
        # DO NOT rely on the bot's main loop to check prices.
        # The exact method depends heavily on the exchange (Bybit in this case).
        # Bybit USDT Perpetual often uses `create_order` with specific `params`.

        # Check if entry order likely filled (adjust status check as needed)
        # This is a basic check; a more robust solution would poll order status.
        if order_status in ['closed', 'filled']: # 'closed' often means fully filled for market orders
            logger.info(f"Entry order {order_id} appears filled. Attempting to place SL/TP orders.")
            close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY # Opposite side for closing

            try:
                # --- Place Stop Loss Order ---
                sl_params = {
                    'stopLoss': stop_loss_price, # The trigger price
                    'triggerDirection': 2 if side == SIDE_BUY else 1, # 1: Mark price < trigger, 2: Mark price > trigger (Check Bybit docs!)
                    'triggerBy': 'MarkPrice', # Or LastPrice, IndexPrice - Check Bybit options
                    'reduce_only': True,
                    'close_on_trigger': True # Ensures it closes the position
                    # Bybit might use 'slTriggerBy', 'stopLossPrice', 'slOrderType': 'Market' etc.
                    # CONSULT CCXT & BYBIT DOCUMENTATION FOR ACCURATE PARAMS for create_order or specific SL/TP methods
                     'position_idx': 0 # For Bybit one-way mode on USDT perpetuals
                }
                logger.info(f"Attempting to place STOP LOSS order: Side={close_side}, Qty={precise_qty_str}, Trigger Price={stop_loss_price:.4f}")
                # Using create_order with type='Stop' or similar might be needed, or specific params as shown above.
                # This is a conceptual example, likely needs adjustment:
                # sl_order = exchange.create_order(symbol, 'stop_market', close_side, precise_qty, price=None, params=sl_params)
                # OR Bybit might use modify_position or a dedicated SL/TP setting function.
                # Example using hypothetical modify_position structure:
                # response_sl = exchange.private_post_position_trading_stop({'symbol': market['id'], 'stopLoss': str(stop_loss_price), 'positionIdx': 0})
                # logger.info(f"Exchange response for SL setting: {response_sl}")

                # >>> Placeholder - Requires actual Bybit API call implementation <<<
                logger.warning("!!! Placeholder SL Order: Actual exchange API call needed here !!!")
                # sl_order = exchange.create_stop_loss_order(symbol, close_side, precise_qty, stop_loss_price, params={'reduce_only': True}) # Fictional method


                # --- Place Take Profit Order ---
                tp_params = {
                     'takeProfit': take_profit_price, # The trigger price
                     'triggerDirection': 1 if side == SIDE_BUY else 2, # Opposite of SL (Check Bybit docs!)
                     'triggerBy': 'MarkPrice', # Or LastPrice, IndexPrice
                     'reduce_only': True,
                     'close_on_trigger': True,
                     # Similar parameter uncertainty as SL - consult docs
                     'position_idx': 0 # For Bybit one-way mode
                }
                logger.info(f"Attempting to place TAKE PROFIT order: Side={close_side}, Qty={precise_qty_str}, Trigger Price={take_profit_price:.4f}")
                # response_tp = exchange.private_post_position_trading_stop({'symbol': market['id'], 'takeProfit': str(take_profit_price), 'positionIdx': 0})
                # logger.info(f"Exchange response for TP setting: {response_tp}")

                # >>> Placeholder - Requires actual Bybit API call implementation <<<
                logger.warning("!!! Placeholder TP Order: Actual exchange API call needed here !!!")
                # tp_order = exchange.create_take_profit_order(symbol, close_side, precise_qty, take_profit_price, params={'reduce_only': True}) # Fictional method

                logger.success(f"SL/TP order placement routines completed (Check logs and exchange interface for confirmation).")

            except ccxt.NetworkError as sltp_ne:
                 logger.critical(f"!!! CRITICAL: Network Error placing SL/TP for order {order_id}: {sltp_ne}. POSITION IS UNPROTECTED !!!")
            except ccxt.ExchangeError as sltp_ee:
                 logger.critical(f"!!! CRITICAL: Exchange Error placing SL/TP for order {order_id}: {sltp_ee}. POSITION IS UNPROTECTED !!!")
            except Exception as sltp_e:
                 logger.critical(f"!!! CRITICAL: Unexpected Error placing SL/TP for order {order_id}: {sltp_e}. POSITION IS UNPROTECTED !!!")
                 logger.debug(traceback.format_exc())
            # **** Add logic here to potentially close the position immediately if SL/TP placement fails ****
            # **** e.g., call close_position(exchange, symbol, get_current_position(exchange, symbol)) ****

        elif order_status in ['open', 'partially_filled']:
             logger.warning(f"Entry order {order_id} is {order_status}. SL/TP placement skipped. Monitor manually or implement order status polling.")
        else:
             logger.error(f"Entry order {order_id} failed or status unknown ({order_status}). SL/TP placement skipped.")
        # ------------------------------------------------------------------

        return entry_order # Return the entry order details regardless of SL/TP success (errors are logged critically)

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds reported by exchange during order placement attempt: {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Network error placing {side.upper()} order for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error placing {side.upper()} order for {symbol}: {e} (Args: {e.args})")
    except Exception as e:
        logger.error(f"Unexpected error placing {side.upper()} order for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return None

# --- Trading Logic ---
def trade_logic(
    exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame,
    st_len: int, st_mult: float, cf_st_len: int, cf_st_mult: float,
    atr_len: int, vol_ma_len: int,
    risk_pct: float, sl_atr_mult: float, tp_atr_mult: float, leverage: int
) -> None:
    """
    Executes the main trading logic for one cycle based on indicators and position state.
    NOTE: This function triggers entries but relies on exchange-native orders for SL/TP.

    Args:
        exchange: Initialized CCXT exchange instance.
        symbol: Market symbol.
        df: DataFrame with OHLCV data.
        st_len, st_mult: Primary Supertrend parameters.
        cf_st_len, cf_st_mult: Confirmation Supertrend parameters.
        atr_len: ATR calculation period.
        vol_ma_len: Volume MA period.
        risk_pct: Risk percentage per trade.
        sl_atr_mult, tp_atr_mult: ATR multipliers for SL/TP (used for entry calculation & logging).
        leverage: Account leverage.
    """
    cycle_start_time = pd.Timestamp.now(tz='UTC')
    logger.info(f"========== New Logic Cycle: {symbol} @ {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ==========")

    # 1. Data Validation
    # Determine minimum required rows based on longest lookback period for indicator calculation + comparison
    required_rows = max(2, st_len + 1, cf_st_len + 1, atr_len + 1, vol_ma_len + 1)
    if df is None or len(df) < required_rows:
        logger.warning(f"Insufficient data ({len(df) if df is not None else 0} rows, need {required_rows}). Skipping logic cycle.")
        return

    # Initialize variables
    ob_data: Optional[Dict[str, float | None]] = None
    current_atr: Optional[float] = None

    try:
        # 2. Calculate Indicators
        logger.debug("Calculating indicators (Supertrends, ATR, Volume)...")
        df = calculate_supertrend(df, st_len, st_mult)
        df = calculate_supertrend(df, cf_st_len, cf_st_mult, prefix="confirm_")
        vol_atr_data = analyze_volume_atr(df, atr_len, vol_ma_len)
        current_atr = vol_atr_data.get("atr")

        # Verify essential indicator calculations succeeded on the *last* row
        required_cols = ['close', 'supertrend', 'trend', 'st_long', 'st_short',
                         'confirm_supertrend', 'confirm_trend']
        if not all(col in df.columns for col in required_cols):
             logger.warning("One or more required indicator columns are missing. Skipping logic.")
             return

        last_row = df.iloc[-1]
        if last_row[required_cols].isnull().any():
             logger.warning("Latest indicator row contains NA values. Skipping logic cycle.")
             # Log which columns are problematic
             for col in required_cols:
                 if pd.isna(last_row[col]): logger.debug(f"Column '{col}' has NA value in last row.")
             return

        # Validate ATR specifically needed for risk management
        if current_atr is None or current_atr <= 1e-9:
            logger.warning(f"ATR calculation failed or resulted in invalid value ({current_atr}). Cannot proceed with risk management/entries.")
            return

        # Extract latest indicator values
        current_price = last_row['close']
        primary_trend_up = last_row["trend"]
        primary_long_signal = last_row["st_long"] # Signal on the bar the trend flipped
        primary_short_signal = last_row["st_short"]# Signal on the bar the trend flipped
        confirm_trend_up = last_row["confirm_trend"]

        # 3. Analyze Order Book (Conditionally)
        # Fetch OB if configured globally OR if a primary signal has just occurred (might confirm with OB)
        should_fetch_ob = FETCH_ORDER_BOOK_PER_CYCLE or primary_long_signal or primary_short_signal
        if should_fetch_ob:
            logger.debug(f"Fetching/Analyzing order book (Reason: FETCH_PER_CYCLE={FETCH_ORDER_BOOK_PER_CYCLE}, Signal={primary_long_signal or primary_short_signal})...")
            ob_data = analyze_order_book(exchange, symbol, ORDER_BOOK_DEPTH)
        else:
            logger.debug("Order book fetch skipped this cycle (no primary signal and FETCH_PER_CYCLE=False).")

        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        # OB considered available if data exists AND ratio was calculable
        ob_available = ob_data is not None and bid_ask_ratio is not None

        # 4. Get Current Position
        position = get_current_position(exchange, symbol) # Use improved function
        position_side = position['side']
        position_qty = position['qty']
        entry_price = position['entry_price']

        # --- Log Current State ---
        primary_trend_str = "Up" if primary_trend_up else "Down"
        confirm_trend_str = "Up" if confirm_trend_up else "Down"
        volume_ratio = vol_atr_data.get("volume_ratio")
        volume_spike = volume_ratio is not None and volume_ratio > VOLUME_SPIKE_THRESHOLD
        volume_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else 'N/A'
        atr_str = f"{current_atr:.5f}" # Already validated non-None/non-zero earlier
        bid_ask_ratio_str = f"{bid_ask_ratio:.3f}" if ob_available else 'N/A'
        spread_val = ob_data.get('spread') if ob_data else None
        spread_str = f"{spread_val:.4f}" if spread_val is not None else 'N/A'

        logger.info(f"Market State: Price={current_price:.4f}, ATR={atr_str}, Vol Ratio={volume_ratio_str} (Spike={volume_spike})")
        logger.info(f"Indicators: Primary ST={last_row['supertrend']:.4f} ({primary_trend_str}, LongSig={primary_long_signal}, ShortSig={primary_short_signal}), "
                    f"Confirm ST={last_row['confirm_supertrend']:.4f} ({confirm_trend_str})")
        if ob_data: # Only log OB details if fetched
            logger.info(f"Order Book: Ratio={bid_ask_ratio_str} (Avail={ob_available}), Spread={spread_str}, Best Bid={ob_data.get('best_bid', 'N/A')}, Best Ask={ob_data.get('best_ask', 'N/A')}")
        logger.info(f"Position State: Side={position_side}, Qty={position_qty:.6f}, Entry={entry_price:.4f}")
        # --- End State Logging ---


        # 5. Check Stop-Loss / Take-Profit (Loop-based - *** INFORMATIONAL ONLY ***)
        # !!! WARNING: This loop-based check is INACCURATE and UNSAFE for scalping. !!!
        # !!! Price can gap past SL/TP levels between checks. USE EXCHANGE-NATIVE ORDERS! !!!
        # This section is kept *only* for logging/debugging awareness, it does NOT trigger closures.
        if position_side != POSITION_SIDE_NONE and entry_price > 0 and current_atr > 0:
            logger.debug("Performing loop-based SL/TP check (Informational Only - RELIES ON EXCHANGE ORDERS)...")
            sl_triggered, tp_triggered = False, False
            # Use the *position's entry price* and the *current ATR* for this informational check
            sl_distance = current_atr * sl_atr_mult
            tp_distance = current_atr * tp_atr_mult
            stop_price, profit_price = 0.0, 0.0

            if position_side == POSITION_SIDE_LONG:
                stop_price = entry_price - sl_distance
                profit_price = entry_price + tp_distance
                logger.debug(f"Loop Check (Long): Entry={entry_price:.4f}, Curr={current_price:.4f}, SL Level ~{stop_price:.4f}, TP Level ~{profit_price:.4f}")
                # Check using current price against calculated levels
                if current_price <= stop_price: sl_triggered = True
                if current_price >= profit_price: tp_triggered = True
            elif position_side == POSITION_SIDE_SHORT:
                stop_price = entry_price + sl_distance
                profit_price = entry_price - tp_distance
                logger.debug(f"Loop Check (Short): Entry={entry_price:.4f}, Curr={current_price:.4f}, SL Level ~{stop_price:.4f}, TP Level ~{profit_price:.4f}")
                # Check using current price against calculated levels
                if current_price >= stop_price: sl_triggered = True
                if current_price <= profit_price: tp_triggered = True

            # Log triggers found by this unreliable method, reminding user that exchange orders handle it.
            if sl_triggered:
                logger.warning(f"*** LOOP-BASED SL DETECTED (INFORMATIONAL) *** Price {current_price:.4f} crossed SL level ~{stop_price:.4f}. Exchange-native order should handle closure.")
                # **** DO NOT CALL close_position() here ****
            elif tp_triggered:
                logger.warning(f"*** LOOP-BASED TP DETECTED (INFORMATIONAL) *** Price {current_price:.4f} crossed TP level ~{profit_price:.4f}. Exchange-native order should handle closure.")
                # **** DO NOT CALL close_position() here ****
            else:
                 logger.debug("Loop-based SL/TP levels not breached in this check.")
        elif position_side != POSITION_SIDE_NONE:
             logger.warning("Active position exists, but entry price or ATR is invalid for loop-based SL/TP check.")
        # --- End Informational SL/TP Check ---

        # 6. Check for Entry Signals
        logger.debug("Checking entry conditions...")

        # Define filter conditions (OB and Volume)
        # If OB wasn't fetched or ratio is None, the filter passes (can't evaluate)
        passes_long_ob = not ob_available or bid_ask_ratio >= ORDER_BOOK_RATIO_THRESHOLD_LONG
        passes_short_ob = not ob_available or bid_ask_ratio <= ORDER_BOOK_RATIO_THRESHOLD_SHORT
        passes_volume = not REQUIRE_VOLUME_SPIKE_FOR_ENTRY or volume_spike

        # Long Entry Condition:
        # - Primary Supertrend generated a LONG signal *on this bar* (trend flipped up)
        # - Confirmation Supertrend is currently in an UP trend
        # - Order Book filter passes (if OB is available)
        # - Volume filter passes (if volume spike is required)
        enter_long = (
            primary_long_signal and confirm_trend_up and passes_long_ob and passes_volume
        )
        logger.debug(f"Long Entry Check: PrimarySig={primary_long_signal}, ConfirmTrendUp={confirm_trend_up}, "
                     f"OB OK={passes_long_ob} (Ratio={bid_ask_ratio_str} vs {ORDER_BOOK_RATIO_THRESHOLD_LONG}, Avail={ob_available}), "
                     f"Vol OK={passes_volume} (Spike={volume_spike}, Required={REQUIRE_VOLUME_SPIKE_FOR_ENTRY}) "
                     f"-> Result={enter_long}")

        # Short Entry Condition:
        # - Primary Supertrend generated a SHORT signal *on this bar* (trend flipped down)
        # - Confirmation Supertrend is currently in a DOWN trend (i.e., NOT up)
        # - Order Book filter passes (if OB is available)
        # - Volume filter passes (if volume spike is required)
        enter_short = (
            primary_short_signal and not confirm_trend_up and passes_short_ob and passes_volume
        )
        logger.debug(f"Short Entry Check: PrimarySig={primary_short_signal}, ConfirmTrendDown={not confirm_trend_up}, "
                     f"OB OK={passes_short_ob} (Ratio={bid_ask_ratio_str} vs {ORDER_BOOK_RATIO_THRESHOLD_SHORT}, Avail={ob_available}), "
                     f"Vol OK={passes_volume} (Spike={volume_spike}, Required={REQUIRE_VOLUME_SPIKE_FOR_ENTRY}) "
                     f"-> Result={enter_short}")


        # 7. Execute Actions
        if position_side == POSITION_SIDE_NONE:
            if enter_long:
                logger.success(f"*** ENTRY SIGNAL: LONG *** for {symbol} at ~{current_price:.4f}. Placing order.")
                # Pass TP multiplier needed for SL/TP order placement inside
                entry_order_result = place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, tp_atr_mult, leverage)
                if entry_order_result:
                    logger.info("Long entry order placed (SL/TP placement attempted within function).")
                else:
                    logger.error("Failed to place LONG entry order.")
            elif enter_short:
                logger.success(f"*** ENTRY SIGNAL: SHORT *** for {symbol} at ~{current_price:.4f}. Placing order.")
                # Pass TP multiplier needed for SL/TP order placement inside
                entry_order_result = place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, tp_atr_mult, leverage)
                if entry_order_result:
                    logger.info("Short entry order placed (SL/TP placement attempted within function).")
                else:
                    logger.error("Failed to place SHORT entry order.")
            else:
                logger.info("No entry signal while flat.")
        elif position_side == POSITION_SIDE_LONG:
            # Check for reversal signal (Short entry signal while Long)
            if enter_short:
                logger.warning(f"*** REVERSAL SIGNAL: SHORT *** while LONG. Closing LONG position first.")
                close_order = close_position(exchange, symbol, position)
                if close_order:
                    logger.info(f"Closed LONG position, attempting SHORT entry...")
                    time.sleep(RETRY_DELAY_SECONDS * 2) # Brief pause before new entry, allow exchange state to settle
                    # Re-fetch position to confirm closure before entering new trade? Optional, adds latency.
                    new_pos = get_current_position(exchange, symbol)
                    if new_pos['side'] == POSITION_SIDE_NONE:
                         logger.info("Position confirmed closed. Placing SHORT entry order.")
                         place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, tp_atr_mult, leverage)
                    else:
                         logger.error("Position did not close successfully after close order. Skipping SHORT entry.")
                else:
                    logger.error("Failed to place close order for LONG position during reversal attempt. Skipping SHORT entry.")
            else:
                logger.info("Maintaining LONG position. No exit or reversal signal.")
        elif position_side == POSITION_SIDE_SHORT:
             # Check for reversal signal (Long entry signal while Short)
            if enter_long:
                logger.warning(f"*** REVERSAL SIGNAL: LONG *** while SHORT. Closing SHORT position first.")
                close_order = close_position(exchange, symbol, position)
                if close_order:
                    logger.info(f"Closed SHORT position, attempting LONG entry...")
                    time.sleep(RETRY_DELAY_SECONDS * 2) # Brief pause before new entry
                    new_pos = get_current_position(exchange, symbol)
                    if new_pos['side'] == POSITION_SIDE_NONE:
                        logger.info("Position confirmed closed. Placing LONG entry order.")
                        place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, tp_atr_mult, leverage)
                    else:
                         logger.error("Position did not close successfully after close order. Skipping LONG entry.")
                else:
                    logger.error("Failed to close SHORT position for reversal attempt. Skipping LONG entry.")
            else:
                logger.info("Maintaining SHORT position. No exit or reversal signal.")

    except Exception as e:
        logger.error(f"!!! CRITICAL ERROR in trade_logic execution: {e} !!!")
        logger.debug(traceback.format_exc()) # Log full traceback for debugging
    finally:
        cycle_end_time = pd.Timestamp.now(tz='UTC')
        duration = (cycle_end_time - cycle_start_time).total_seconds()
        logger.info(f"========== Logic Cycle End: {symbol} (Duration: {duration:.3f}s) ==========\n")


# --- Main Execution ---
def main() -> None:
    """Main function to initialize and run the scalping bot loop."""
    run_start_time = pd.Timestamp.now(tz='UTC')
    logger.info(f"--- HFT Scalping Bot Initializing --- {run_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    logger.warning("--- !!! LIVE FUTURES SCALPING MODE ENABLED - EXTREME RISK INVOLVED !!! ---")
    logger.warning("---          REVIEW ALL PARAMETERS AND TEST THOROUGHLY BEFORE USE          ---")
    logger.critical("--- CRITICAL: ENSURE EXCHANGE-NATIVE SL/TP IMPLEMENTATION IS CORRECT ---")

    exchange = initialize_exchange()
    if not exchange:
        logger.critical("Exchange initialization failed. Bot cannot start.")
        return # Exit if exchange setup fails

    # --- Symbol and Leverage Setup ---
    symbol = ""
    try:
        # Get symbol input or use default
        symbol_input = input(f"Enter trading symbol (e.g., BTC/USDT:USDT) or press Enter for default [{DEFAULT_SYMBOL}]: ").strip().upper()
        if not symbol_input:
            symbol_input = DEFAULT_SYMBOL
            logger.info(f"Using default symbol: {symbol_input}")
        else:
             logger.info(f"User provided symbol: {symbol_input}")

        # Validate symbol and check market type
        logger.info(f"Validating symbol '{symbol_input}' on {exchange.id}...")
        # Ensure markets are loaded before accessing market info
        if not exchange.markets:
            exchange.load_markets()
        market = exchange.market(symbol_input)
        symbol = market['symbol']  # Use the validated, correctly formatted symbol from CCXT
        logger.success(f"Symbol validated: {symbol}")

        if not market.get('swap', False) and not market.get('future', False):
             logger.critical(f"CRITICAL: Symbol {symbol} is not a swap or futures market according to CCXT market data. Leverage cannot be set. Exiting.")
             return

        # Set Leverage
        if not set_leverage(exchange, symbol, DEFAULT_LEVERAGE):
             # Allow continuation even if leverage setting fails, but warn heavily
             logger.critical(f"CRITICAL WARNING: Failed to set leverage to {DEFAULT_LEVERAGE}x for {symbol}. Bot will continue but MAY USE INCORRECT LEVERAGE (e.g., exchange default). Verify on exchange interface!")
             # return # Uncomment this line to exit if leverage setting is mandatory

    except (ccxt.BadSymbol, KeyError) as e:
         logger.critical(f"CRITICAL: Invalid or unsupported symbol '{symbol_input}'. Error: {e}. Exiting.")
         return
    except ccxt.NetworkError as e:
         logger.critical(f"CRITICAL: Network error during symbol/leverage setup: {e}. Exiting.")
         return
    except Exception as e:
        logger.critical(f"CRITICAL: Unexpected error during symbol/leverage setup: {e}. Exiting.")
        logger.debug(traceback.format_exc())
        return

    # --- Configuration Summary ---
    logger.info("--- Scalping Bot Configuration Summary ---")
    logger.info(f" Trading Symbol:        {symbol}")
    logger.info(f" Timeframe:             {DEFAULT_INTERVAL}")
    logger.info(f" Target Leverage:       {DEFAULT_LEVERAGE}x (Verify on exchange!)")
    logger.info(f" Check Interval:        {DEFAULT_SLEEP_SECONDS} seconds")
    logger.info(f" Risk Per Trade:        {RISK_PER_TRADE_PERCENTAGE:.3%}")
    logger.info(f" Max Order Cap (USDT):  {MAX_ORDER_USDT_AMOUNT:.2f}")
    logger.info(f" SL ATR Period:         {ATR_CALCULATION_PERIOD}")
    logger.info(f" SL ATR Multiplier:     {ATR_STOP_LOSS_MULTIPLIER}")
    logger.info(f" TP ATR Multiplier:     {ATR_TAKE_PROFIT_MULTIPLIER}")
    logger.info(f" Primary ST:            Length={DEFAULT_ST_ATR_LENGTH}, Multiplier={DEFAULT_ST_MULTIPLIER}")
    logger.info(f" Confirmation ST:       Length={CONFIRM_ST_ATR_LENGTH}, Multiplier={CONFIRM_ST_MULTIPLIER}")
    logger.info(f" Volume MA Period:      {VOLUME_MA_PERIOD}")
    logger.info(f" Volume Spike Ratio:    > {VOLUME_SPIKE_THRESHOLD:.2f}x MA")
    logger.info(f" Require Volume Spike:  {REQUIRE_VOLUME_SPIKE_FOR_ENTRY}")
    logger.info(f" Order Book Depth:      {ORDER_BOOK_DEPTH}")
    logger.info(f" OB Long Ratio Thresh:  > {ORDER_BOOK_RATIO_THRESHOLD_LONG:.2f}")
    logger.info(f" OB Short Ratio Thresh: < {ORDER_BOOK_RATIO_THRESHOLD_SHORT:.2f}")
    logger.info(f" Fetch OB Every Cycle:  {FETCH_ORDER_BOOK_PER_CYCLE} (Impacts Rate Limits!)")
    logger.warning("--- Ensure configuration aligns with your strategy and risk tolerance. ---")
    logger.critical("--- CRITICAL: Loop-based SL/TP checks are informational ONLY. RELY ON EXCHANGE-NATIVE ORDERS. ---")
    logger.info("---------------------------------------------")
    input("Press Enter to start the trading loop (or CTRL+C to exit)...")
    logger.info("Starting main trading loop... Press CTRL+C to stop gracefully.")


    # --- Calculate required data length ONCE before the loop ---
    # Needs to be large enough for the longest lookback period of any indicator + comparisons
    required_rows_calc = max(
        2, # Base requirement
        DEFAULT_ST_ATR_LENGTH + 1,
        CONFIRM_ST_ATR_LENGTH + 1,
        ATR_CALCULATION_PERIOD + 1,
        VOLUME_MA_PERIOD + 1 # Rolling MA needs window size
    )
    # Fetch more data than strictly required for buffer and stability
    data_limit = max(100, required_rows_calc + 50) # Use calculated rows + buffer
    logger.info(f"Calculated required rows for indicators: {required_rows_calc}. Fetching {data_limit} candles per cycle.")


    # --- Main Trading Loop ---
    while True:
        loop_start_time = time.time()
        try:
            current_time_utc = pd.Timestamp.now(tz='UTC')
            logger.debug(f"--- Top of Loop Cycle at {current_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

            # Fetch latest market data
            df_market = get_market_data(exchange, symbol, DEFAULT_INTERVAL, limit=data_limit)

            # Run trading logic if data is available and sufficient
            if df_market is not None and not df_market.empty and len(df_market) >= required_rows_calc:
                # Pass a copy of the dataframe to prevent modification issues if df_market is reused
                trade_logic(
                    exchange=exchange, symbol=symbol, df=df_market.copy(),
                    st_len=DEFAULT_ST_ATR_LENGTH, st_mult=DEFAULT_ST_MULTIPLIER,
                    cf_st_len=CONFIRM_ST_ATR_LENGTH, cf_st_mult=CONFIRM_ST_MULTIPLIER,
                    atr_len=ATR_CALCULATION_PERIOD, vol_ma_len=VOLUME_MA_PERIOD,
                    risk_pct=RISK_PER_TRADE_PERCENTAGE, sl_atr_mult=ATR_STOP_LOSS_MULTIPLIER,
                    tp_atr_mult=ATR_TAKE_PROFIT_MULTIPLIER, leverage=DEFAULT_LEVERAGE
                )
            elif df_market is None or df_market.empty:
                logger.warning("No market data received in this cycle, skipping trade logic.")
            else: # Data received but not enough rows
                logger.warning(f"Insufficient market data rows received ({len(df_market)} < {required_rows_calc}), skipping trade logic.")


        except KeyboardInterrupt:
            logger.warning(">>> KeyboardInterrupt detected. Attempting graceful shutdown... <<<")
            # --- Graceful Shutdown ---
            logger.info("Fetching final position status...")
            current_pos = get_current_position(exchange, symbol)
            if current_pos['side'] != POSITION_SIDE_NONE:
               logger.warning(f"!!! Open {current_pos['side']} position detected ({current_pos['qty']} @ {current_pos['entry_price']}) !!!")
               close_now = input("Attempt to CLOSE the position with a market order? (yes/NO): ").strip().lower()
               if close_now == 'yes':
                   logger.warning(f"Attempting to close open {current_pos['side']} position before exiting...")
                   close_result = close_position(exchange, symbol, current_pos)
                   if close_result:
                       logger.success("Position close order placed.")
                   else:
                       logger.error("Failed to place position close order. MANUAL INTERVENTION REQUIRED on exchange.")
               else:
                   logger.warning("Position left open. MANUAL INTERVENTION REQUIRED on exchange.")
            else:
                logger.info("No open position detected.")

            # Consider cancelling open orders as well (more complex, requires fetching open orders)
            # logger.info("Attempting to cancel any open limit/stop orders...")
            # try:
            #    exchange.cancel_all_orders(symbol)
            #    logger.info("Cancel all orders request sent (check exchange).")
            # except Exception as cancel_e:
            #    logger.error(f"Failed to cancel orders: {cancel_e}")

            logger.warning(">>> Shutdown initiated. Exiting main loop. <<<")
            break # Exit the main loop

        except ccxt.RateLimitExceeded as e:
             logger.warning(f"!!! RATE LIMIT EXCEEDED: {e}. Consider increasing SLEEP_SECONDS or disabling/tuning FETCH_ORDER_BOOK_PER_CYCLE.")
             sleep_time = DEFAULT_SLEEP_SECONDS * 5 # Sleep much longer after rate limit hit
             logger.warning(f"Sleeping for {sleep_time} seconds due to rate limit...")
             time.sleep(sleep_time)
             continue # Skip rest of the loop and try again

        except ccxt.NetworkError as e:
             # Transient network issues, retry after normal delay
             logger.warning(f"Network error encountered in main loop: {e}. Retrying after sleep...")
             # Continue to sleep logic at the end of the loop

        except ccxt.ExchangeNotAvailable as e:
             # Exchange might be down for maintenance
             logger.error(f"Exchange not available: {e}. Sleeping longer...")
             sleep_time = DEFAULT_SLEEP_SECONDS * 10 # Sleep even longer
             logger.warning(f"Sleeping for {sleep_time} seconds...")
             time.sleep(sleep_time)
             continue # Skip rest of the loop and try again

        except ccxt.AuthenticationError as e:
             logger.critical(f"CRITICAL: Authentication error during loop execution: {e}. API keys might be invalid/expired. Stopping bot.")
             break # Stop the bot on auth errors

        except ccxt.ExchangeError as e:
             # Catch other specific exchange errors if needed, else log generally
             logger.error(f"Unhandled Exchange error encountered in main loop: {e} (Args: {e.args}).")
             logger.debug(traceback.format_exc())
             # Continue to sleep logic, but monitor closely

        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.critical(f"!!! UNEXPECTED CRITICAL ERROR in main loop: {e} !!!")
            logger.debug(traceback.format_exc()) # Log full traceback
            logger.info(f"Attempting to continue after {DEFAULT_SLEEP_SECONDS * 2}s sleep...") # Sleep longer after unexpected errors
            time.sleep(DEFAULT_SLEEP_SECONDS * 2)
            # Continue to sleep logic, hoping it's a temporary issue

        # --- Loop Sleep ---
        # Calculate elapsed time and sleep to maintain the desired check interval
        loop_end_time = time.time()
        elapsed_time = loop_end_time - loop_start_time
        sleep_duration = max(0, DEFAULT_SLEEP_SECONDS - elapsed_time)
        logger.debug(f"Cycle execution time: {elapsed_time:.3f}s. Sleeping for {sleep_duration:.3f}s.")
        if sleep_duration > 0:
             time.sleep(sleep_duration)

    logger.info("--- Scalping Bot Shutdown Complete ---")


if __name__ == "__main__":
    # Initial check already done at the top after imports
    # try:
    #     import pandas_ta
    #     logger.debug(f"pandas_ta version: {pandas_ta.__version__}")
    # except ImportError:
    #     logger.critical("CRITICAL: 'pandas_ta' library not found. Please install it: pip install pandas_ta")
    #     sys.exit(1)

    main()
```

**To Make This Production-Ready (Regarding SL/TP):**

1.  **Identify Bybit's SL/TP Method:** Research the *exact* CCXT method and required parameters for setting Stop Loss and Take Profit on Bybit USDT Perpetual futures *after* a position is opened. This might involve:
    *   Using `exchange.create_order()` with specific `type` (like `'Stop'`, `'MarketIfTouched'`) and `params` (like `stopLossPrice`, `takeProfitPrice`, `triggerPrice`, `reduce_only`, `close_on_trigger`, `positionIdx`).
    *   Using a dedicated method like `exchange.edit_position()` or `exchange.private_post_position_trading_stop()` (using CCXT's implicit API methods - requires careful handling).
    *   Checking the CCXT documentation for Bybit: [https://docs.ccxt.com/en/latest/manual.html#bybit](https://docs.ccxt.com/en/latest/manual.html#bybit) and specifically the section on SL/TP orders.
2.  **Implement the Correct Calls:** Replace the `Placeholder SL Order` and `Placeholder TP Order` sections in `place_risked_market_order` with the *actual*, correct CCXT calls using the parameters identified in step 1.
3.  **Robust Error Handling for SL/TP:** Implement logic to handle failures when placing SL/TP orders. If SL/TP cannot be placed after an entry, the position is unprotected. Consider automatically closing the position immediately as a safety measure if SL/TP placement fails.
4.  **Test Extensively:** Test the SL/TP placement logic thoroughly on the Bybit **testnet** to ensure orders are placed correctly at the right trigger prices and with the `reduce_only` flag set. Verify they trigger as expected.
