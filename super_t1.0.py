    ATR_CALCULATION_PERIOD * 2, VOLUME_MA_PERIOD * 2, required_rows + 5) # Add buffer                                                                                                               ^^^^^^^^^^^^^    NameError: name 'required_rows' is not defined                         2025-04-25 21:37:25,316 [INFO] Attempting to continue after 10s sleep...                                                                      2025-04-25 21:37:35,293 [ERROR] !!! UNEXPECTED CRITICAL ERROR in main loop: name 'required_rows' is not defined !!!                           Traceback (most recent call last):                                       File "/data/data/com.termux/files/home/worldguide/super_t1.0.py", line 1195, in main                                                            ATR_CALCULATION_PERIOD * 2, VOLUME_MA_PERIOD * 2, required_rows + 5) # Add buffer
    fix tp and sl

#!/usr/bin/env python

"""
High-Frequency Trading Bot (Scalping) using Dual Supertrend, ATR, Volume, and Order Book Analysis.

Disclaimer:
- EXTREMELY HIGH RISK: Scalping bots operate in noisy, fast-moving markets.
  Risk of significant losses due to latency, slippage, fees, API issues,
  market volatility, and incorrect parameter tuning is substantial.
- Parameter Sensitivity: Strategy performance is HIGHLY dependent on parameter
  tuning (Supertrend lengths/multipliers, ATR settings, volume thresholds, OB ratios).
  Requires extensive backtesting and optimization for specific market conditions.
  Defaults provided are purely illustrative and likely suboptimal.
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

import ccxt
import pandas as pd
import pandas_ta as ta  # Ensure pandas_ta is installed
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
DEFAULT_RECV_WINDOW = 10000 # Bybit specific: time window for request validity
ORDER_BOOK_FETCH_LIMIT = max(25, ORDER_BOOK_DEPTH)  # API often requires minimum fetch levels (e.g., 25, 50)
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

# Add color to log levels
logging.addLevelName(logging.DEBUG, f"\033[90m{logging.getLevelName(logging.DEBUG)}\033[0m") # Grey
logging.addLevelName(logging.INFO, f"\033[96m{logging.getLevelName(logging.INFO)}\033[0m") # Cyan
logging.addLevelName(logging.WARNING, f"\033[93m{logging.getLevelName(logging.WARNING)}\033[0m") # Yellow
logging.addLevelName(logging.ERROR, f"\033[91m{logging.getLevelName(logging.ERROR)}\033[0m") # Red
logging.addLevelName(logging.CRITICAL, f"\033[91m\033[1m{logging.getLevelName(logging.CRITICAL)}\033[0m") # Bold Red
logging.addLevelName(SUCCESS_LEVEL, f"\033[92m{logging.getLevelName(SUCCESS_LEVEL)}\033[0m") # Green

# Uncomment below to enable detailed DEBUG logging globally
# logging.getLogger().setLevel(logging.DEBUG)

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
            "enableTimeSync": True,   # Enable time synchronization
            "options": {
                "defaultType": "swap",  # Use swap/perpetual futures
                "recvWindow": DEFAULT_RECV_WINDOW,
                 # "adjustForTimeDifference": True, # Already covered by enableTimeSync in recent ccxt
            },
        })
        # Test connection and authentication
        logger.debug("Loading markets...")
        exchange.load_markets()
        logger.debug("Testing authentication by fetching balance...")
        exchange.fetch_balance()
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
    st_long_col = f"SUPERTl_{length}_{multiplier}" # Long signal column (may not be needed)
    st_short_col = f"SUPERTs_{length}_{multiplier}"# Short signal column (may not be needed)

    target_cols = [
        f"{col_prefix}supertrend", f"{col_prefix}trend",
        f"{col_prefix}st_long", f"{col_prefix}st_short"
    ]

    # Check if input DataFrame is valid and has necessary columns
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low", "close"]):
        logger.warning(f"Input DataFrame is invalid or missing required columns for {col_prefix}Supertrend.")
        for col in target_cols: df[col] = pd.NA # Add NA columns if calculation fails
        return df
    if len(df) < length:
        logger.warning(f"Insufficient data ({len(df)} rows) for {col_prefix}Supertrend length {length}.")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        logger.debug(f"Calculating {col_prefix}Supertrend (len={length}, mult={multiplier})...")
        # Calculate Supertrend using pandas_ta
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)

        # Verify that pandas_ta created the expected columns
        if st_col_name not in df.columns or st_trend_col not in df.columns:
             raise KeyError(f"pandas_ta did not create expected columns: {st_col_name}, {st_trend_col}")

        # Rename and derive necessary columns
        df[f"{col_prefix}supertrend"] = df[st_col_name]
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1 # True if trend is up (direction = 1)

        # Detect trend changes
        prev_trend_direction = df[st_trend_col].shift(1)
        # Long signal: previous trend was down (-1) and current trend is up (1)
        df[f"{col_prefix}st_long"] = (prev_trend_direction == -1) & (df[st_trend_col] == 1)
        # Short signal: previous trend was up (1) and current trend is down (-1)
        df[f"{col_prefix}st_short"] = (prev_trend_direction == 1) & (df[st_trend_col] == -1)

        # Clean up intermediate columns created by pandas_ta
        cols_to_drop = [
            c for c in df.columns if c.startswith("SUPERT_") and c not in [st_col_name, st_trend_col, st_long_col, st_short_col]
        ] + [st_col_name, st_trend_col, st_long_col, st_short_col] # Drop original and intermediate ones

        # Ensure target columns are not accidentally dropped if renaming failed
        cols_to_drop = [c for c in cols_to_drop if c not in target_cols]

        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        last_trend_val = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down' if pd.notna(df[f'{col_prefix}trend'].iloc[-1]) else 'N/A'
        logger.debug(f"Calculated {col_prefix}Supertrend. Last trend: {last_trend_val}")

    except KeyError as ke:
        logger.error(f"Error accessing expected column during {col_prefix}Supertrend calculation: {ke}")
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
    if len(df) < max(atr_len, vol_ma_len):
        logger.warning(f"Insufficient data rows ({len(df)}) for ATR({atr_len})/VolMA({vol_ma_len}) calculation.")
        return results

    try:
        # Calculate ATR using pandas_ta
        logger.debug(f"Calculating ATR({atr_len})...")
        df.ta.atr(length=atr_len, append=True)
        atr_col = f"ATRr_{atr_len}"
        if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]):
            results["atr"] = df[atr_col].iloc[-1]
        else:
            logger.warning(f"Failed to calculate or find valid ATR({atr_len}). Column missing or last value NA.")
        df.drop(columns=[atr_col], errors='ignore', inplace=True) # Clean up ATR column

        # Calculate Volume MA
        logger.debug(f"Calculating Volume MA({vol_ma_len})...")
        volume_ma_col = 'volume_ma'
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=vol_ma_len // 2).mean()
        if pd.notna(df[volume_ma_col].iloc[-1]):
            results["volume_ma"] = df[volume_ma_col].iloc[-1]
            results["last_volume"] = df['volume'].iloc[-1]
            # Calculate ratio, avoiding division by zero or near-zero
            if results["volume_ma"] is not None and results["volume_ma"] > 1e-9 and results["last_volume"] is not None:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            else:
                results["volume_ratio"] = None # Undefined ratio
        else:
             logger.warning(f"Failed to calculate or find valid Volume MA({vol_ma_len}). Last value NA.")
        df.drop(columns=[volume_ma_col], errors='ignore', inplace=True) # Clean up Volume MA column

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

    if not exchange.has.get('fetchL2OrderBook'):
         logger.warning(f"Exchange {exchange.id} does not support fetchL2OrderBook. Skipping order book analysis.")
         return results
    try:
        # Fetch order book data
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)

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
            logger.debug(f"Order Book: Best Bid/Ask not available to calculate spread.")

        # Calculate cumulative volume within the specified depth
        bid_volume_sum = sum(float(bid[1]) for bid in bids[:depth])
        ask_volume_sum = sum(float(ask[1]) for ask in asks[:depth])
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

        # Convert to pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Convert timestamp to datetime objects (UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        # Convert other columns to numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        logger.debug(f"Successfully fetched {len(df)} OHLCV candles for {symbol}. Last candle time: {df.index[-1]}")
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
    Fetches the current open position details for a given symbol.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol.

    Returns:
        dict: A dictionary containing position details:
              'side': POSITION_SIDE_LONG, POSITION_SIDE_SHORT, or POSITION_SIDE_NONE.
              'qty': Position size (float, 0 if no position).
              'entry_price': Average entry price (float, 0 if no position).
              Returns default dict if error occurs.
    """
    default_pos = {'side': POSITION_SIDE_NONE, 'qty': 0.0, 'entry_price': 0.0}
    try:
        logger.debug(f"Fetching current position for {symbol}...")
        # Note: fetch_positions might return multiple positions if hedging mode is enabled,
        # but typically returns one for one-way mode or an empty list.
        # Using `params` might be needed for specific exchanges or modes.
        positions = exchange.fetch_positions([symbol]) # Pass symbol in a list

        # Find the relevant position (should usually be only one for a symbol in one-way mode)
        for pos in positions:
            # Ensure 'symbol' key exists and matches
            if pos.get('symbol') != symbol:
                continue

            contracts_val = pos.get('contracts') # Can be None, float, or sometimes string
            side_val = pos.get('side') # 'long' or 'short'

            # Check for non-zero contracts, handle potential None or string types, use tolerance for float comparison
            if contracts_val is not None and side_val is not None:
                try:
                    qty = float(contracts_val)
                    if abs(qty) > 1e-9: # Check if position size is meaningfully non-zero
                        side = POSITION_SIDE_LONG if side_val == 'long' else POSITION_SIDE_SHORT
                        entry = float(pos.get('entryPrice') or 0.0) # Handle potential None for entryPrice
                        logger.info(f"Found active position: {side} {qty:.6f} {symbol} @ Entry={entry:.4f}")
                        return {'side': side, 'qty': qty, 'entry_price': entry}
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse position quantity for {symbol}: {contracts_val}. Error: {e}")
                    continue # Skip this position entry

        # If loop finishes without finding an active position
        logger.info(f"No active position found for {symbol}.")
        return default_pos

    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching position for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Exchange error fetching position for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching position for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return default_pos # Return default on any error

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
            response = exchange.set_leverage(leverage, symbol)
            logger.success(f"Leverage successfully set to {leverage}x for {symbol}. Response: {response}")
            return True
        except ccxt.ExchangeError as e:
            error_message_lower = str(e).lower()
            # Check if the error indicates leverage is already set (common response)
            # Adjust the string based on your exchange's specific message
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
        params = {'reduce_only': True}

        # Create and place the market order
        order = exchange.create_market_order(symbol, side_to_close, amount_float, params=params)

        logger.success(f"Position CLOSE order placed successfully for {symbol}. Order ID: {order.get('id', 'N/A')}")
        return order

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds to close position for {symbol}: {e}")
    except ccxt.OrderNotFound as e:
         logger.error(f"OrderNotFound error during closing (might indicate position closed already?): {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Network error closing position for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error closing position for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error closing position for {symbol}: {e}")
        logger.debug(traceback.format_exc())
    return None

# --- Risk Calculation ---
def calculate_position_size(
    equity: float, risk_per_trade_pct: float, entry_price: float, stop_loss_price: float, leverage: int
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

    Returns:
        tuple[float | None, float | None]:
            - Position size in base currency (e.g., BTC quantity), or None if invalid.
            - Required margin in quote currency (e.g., USDT), or None if invalid.
    """
    logger.debug(f"Risk Calc Input: Equity={equity:.2f}, Risk%={risk_per_trade_pct:.3%}, "
                 f"Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Leverage={leverage}x")

    if equity <= 0 or risk_per_trade_pct <= 0 or entry_price <= 0 or stop_loss_price <= 0:
        logger.error("Invalid input for position size calculation (Equity, Risk%, Entry, or SL <= 0).")
        return None, None

    # Ensure entry and stop loss are different
    price_difference = abs(entry_price - stop_loss_price)
    if price_difference < 1e-9: # Avoid division by zero or near-zero
        logger.error(f"Entry price ({entry_price:.4f}) and Stop Loss price ({stop_loss_price:.4f}) are too close.")
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
    required_margin_quote = position_value_quote / leverage

    logger.debug(f"Risk Calc Output: RiskAmt={risk_amount_quote:.2f} {QUOTE_CURRENCY}, "
                 f"PriceDiff={price_difference:.4f} "
                 f"=> Quantity={quantity_base:.8f} (Base), "
                 f"Value={position_value_quote:.2f} {QUOTE_CURRENCY}, "
                 f"Req. Margin={required_margin_quote:.2f} {QUOTE_CURRENCY}")

    return quantity_base, required_margin_quote

# --- Place Order (Improved Entry Price Estimation & Risk Management) ---
def place_risked_market_order(
    exchange: ccxt.Exchange, symbol: str, side: str,
    risk_percentage: float, current_atr: float, sl_atr_multiplier: float, leverage: int
) -> Optional[Dict[str, Any]]:
    """
    Calculates position size based on risk/ATR-StopLoss, checks constraints,
    and places a market order. Uses fresh order book data for better entry price estimate.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange instance.
        symbol (str): The market symbol.
        side (str): 'buy' or 'sell'.
        risk_percentage (float): Risk per trade (e.g., 0.01 for 1%).
        current_atr (float): The current ATR value.
        sl_atr_multiplier (float): Multiplier for ATR to determine SL distance.
        leverage (int): Account leverage for the symbol.

    Returns:
        dict | None: The order dictionary from CCXT if successful, otherwise None.
    """
    logger.info(f"--- Attempting to Place {side.upper()} Market Order for {symbol} with Risk Management ---")

    if current_atr is None or current_atr <= 0:
         logger.error("Invalid ATR value ({current_atr}) provided for risk calculation. Cannot place order.")
         return None

    try:
        # 1. Get Account Balance/Equity
        logger.debug("Fetching account balance...")
        balance = exchange.fetch_balance()
        # Use 'total' for equity if available, otherwise fallback to 'free' (more conservative)
        usdt_equity = balance.get('total', {}).get(QUOTE_CURRENCY)
        usdt_free = balance.get('free', {}).get(QUOTE_CURRENCY)

        if usdt_equity is None or usdt_equity <= 0:
            if usdt_free is not None and usdt_free > 0:
                logger.warning(f"Could not get total equity, using free balance ({usdt_free:.2f} {QUOTE_CURRENCY}) for risk calculation.")
                usdt_equity = usdt_free
            else:
                logger.error(f"Cannot determine account equity or free balance in {QUOTE_CURRENCY}. Cannot place order.")
                return None
        elif usdt_free is None:
             logger.error(f"Could not determine free balance in {QUOTE_CURRENCY}. Cannot place order.")
             return None

        logger.debug(f"Account Status: Equity = {usdt_equity:.2f} {QUOTE_CURRENCY}, Free = {usdt_free:.2f} {QUOTE_CURRENCY}")

        # 2. Get Market Info & Estimate Entry Price from Fresh Shallow Order Book
        logger.debug(f"Fetching market info for {symbol}...")
        market = exchange.market(symbol)
        min_qty = market.get('limits', {}).get('amount', {}).get('min')
        logger.debug(f"Market min quantity: {min_qty}")

        logger.debug("Fetching shallow order book for fresh entry price estimate...")
        # Fetch a small, fresh OB for best bid/ask right before ordering
        ob_data = analyze_order_book(exchange, symbol, SHALLOW_OB_FETCH_DEPTH)
        entry_price_estimate = None

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

        # 3. Calculate Stop Loss Price based on Estimated Entry and ATR
        stop_loss_distance = current_atr * sl_atr_multiplier
        stop_loss_price = 0.0
        if side == SIDE_BUY:
            stop_loss_price = entry_price_estimate - stop_loss_distance
        elif side == SIDE_SELL:
            stop_loss_price = entry_price_estimate + stop_loss_distance

        # Ensure SL price is valid (e.g., positive)
        if stop_loss_price <= 0:
             logger.error(f"Calculated Stop Loss price ({stop_loss_price:.4f}) is invalid (<= 0). Cannot place order.")
             return None

        logger.info(f"Calculated SL based on EntryEst={entry_price_estimate:.4f}, "
                    f"ATR({current_atr:.5f}) * {sl_atr_multiplier}: "
                    f"SL Price = {stop_loss_price:.4f} (Distance = {stop_loss_distance:.4f})")

        # 4. Calculate Position Size (Base Quantity) and Required Margin
        quantity_base, required_margin_quote = calculate_position_size(
            usdt_equity, risk_percentage, entry_price_estimate, stop_loss_price, leverage
        )

        if quantity_base is None or required_margin_quote is None:
            logger.error("Position size calculation failed. Cannot place order.")
            return None # Error logged within calculate_position_size

        # 5. Apply Precision, Check Minimums, and Apply Max Cap
        logger.debug(f"Applying precision to calculated quantity: {quantity_base:.8f}")
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
            required_margin_quote = (precise_qty * entry_price_estimate) / leverage
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
                    f"Est. Entry={entry_price_estimate:.4f}, Est. SL={stop_loss_price:.4f}, "
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
        order = exchange.create_market_order(symbol, side, precise_qty)

        # Note: Actual fill price might differ from estimate due to slippage.
        filled_qty = order.get('filled') or order.get('amount') # Prefer 'filled' if available
        avg_price = order.get('average') or order.get('price') # Prefer 'average' if available
        order_id = order.get('id', 'N/A')
        logger.success(f"MARKET Order Placement Success! ID: {order_id}, "
                       f"Size: ~{filled_qty}, Avg Fill Price: ~{avg_price}")

        # ------------------------------------------------------------------
        # !!! CRITICAL TODO: IMPLEMENT EXCHANGE-NATIVE SL/TP ORDERS HERE !!!
        # ------------------------------------------------------------------
        # Immediately after placing the market order, you should place
        # corresponding Stop Loss and Take Profit orders on the exchange.
        # Relying on the bot's main loop to check prices is TOO SLOW and
        # UNRELIABLE for scalping due to latency and potential downtime.
        #
        # Example (Conceptual - requires specific CCXT params for your exchange):
        # try:
        #     sl_params = {'stopLossPrice': stop_loss_price, 'reduce_only': True}
        #     tp_params = {'takeProfitPrice': take_profit_price, 'reduce_only': True} # Calculate TP price similarly
        #
        #     # Determine SL/TP side (opposite of entry)
        #     close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
        #
        #     # Place Stop Loss Order (e.g., STOP_MARKET)
        #     logger.info(f"Placing Stop Loss order at {stop_loss_price:.4f}...")
        #     sl_order = exchange.create_order(symbol, 'stop_market', close_side, precise_qty, price=None, params=sl_params)
        #     logger.success(f"Stop Loss order placed. ID: {sl_order.get('id', 'N/A')}")
        #
        #     # Place Take Profit Order (e.g., TAKE_PROFIT_MARKET)
        #     # tp_price = entry_price_estimate + (current_atr * tp_atr_multiplier) if side == SIDE_BUY else entry_price_estimate - (current_atr * tp_atr_multiplier)
        #     # logger.info(f"Placing Take Profit order at {tp_price:.4f}...")
        #     # tp_order = exchange.create_order(symbol, 'take_profit_market', close_side, precise_qty, price=None, params=tp_params)
        #     # logger.success(f"Take Profit order placed. ID: {tp_order.get('id', 'N/A')}")
        #
        # except Exception as sltp_e:
        #      logger.error(f"!!! FAILED TO PLACE SL/TP ORDERS for order ID {order_id}: {sltp_e} !!!")
        #      # CRITICAL: Decide how to handle this failure (e.g., try to close position immediately?)
        # ------------------------------------------------------------------

        return order # Return the entry order details

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds reported by exchange during order placement attempt: {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Network error placing {side.upper()} order for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error placing {side.upper()} order for {symbol}: {e}")
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

    Args:
        exchange: Initialized CCXT exchange instance.
        symbol: Market symbol.
        df: DataFrame with OHLCV data.
        st_len, st_mult: Primary Supertrend parameters.
        cf_st_len, cf_st_mult: Confirmation Supertrend parameters.
        atr_len: ATR calculation period.
        vol_ma_len: Volume MA period.
        risk_pct: Risk percentage per trade.
        sl_atr_mult, tp_atr_mult: ATR multipliers for SL/TP.
        leverage: Account leverage.
    """
    cycle_start_time = pd.Timestamp.now(tz='UTC')
    logger.info(f"========== New Logic Cycle: {symbol} @ {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ==========")

    # 1. Data Validation
    # Determine minimum required rows based on longest lookback period
    required_rows = max(2, st_len, cf_st_len, atr_len, vol_ma_len) + 1 # Need +1 for previous trend comparison
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

        # Verify essential indicator calculations succeeded
        required_cols = ['close', 'supertrend', 'trend', 'st_long', 'st_short',
                         'confirm_supertrend', 'confirm_trend']
        last_row = df.iloc[-1]
        if not all(col in df.columns for col in required_cols) or last_row[required_cols].isnull().any():
             logger.warning("One or more required indicator columns are missing or contain NA values. Skipping logic.")
             # Log which columns are problematic
             for col in required_cols:
                 if col not in df.columns: logger.debug(f"Missing column: {col}")
                 elif pd.isna(last_row[col]): logger.debug(f"Column '{col}' has NA value.")
             return
        if current_atr is None or current_atr <= 0:
            logger.warning(f"ATR calculation failed or resulted in invalid value ({current_atr}). Cannot proceed with risk management.")
            return

        # Extract latest indicator values
        current_price = last_row['close']
        primary_trend_up = last_row["trend"]
        primary_long_signal = last_row["st_long"]
        primary_short_signal = last_row["st_short"]
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
        ob_available = ob_data is not None and bid_ask_ratio is not None

        # 4. Get Current Position
        position = get_current_position(exchange, symbol)
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
        logger.info(f"Indicators: Primary ST={last_row['supertrend']:.4f} ({primary_trend_str}), "
                    f"Confirm ST={last_row['confirm_supertrend']:.4f} ({confirm_trend_str})")
        if ob_data: # Only log OB details if fetched
            logger.info(f"Order Book: Ratio={bid_ask_ratio_str}, Spread={spread_str}")
        logger.info(f"Position State: Side={position_side}, Qty={position_qty:.6f}, Entry={entry_price:.4f}")
        # --- End State Logging ---


        # 5. Check Stop-Loss / Take-Profit (Loop-based - INFORMATIONAL ONLY)
        # !!! WARNING: This loop-based check is INACCURATE and UNSAFE for scalping. !!!
        # !!! Price can gap past SL/TP levels between checks. USE EXCHANGE-NATIVE ORDERS! !!!
        # This section is kept primarily for logging/debugging awareness, not reliable execution.
        if position_side != POSITION_SIDE_NONE and entry_price > 0 and current_atr > 0:
            logger.debug("Performing loop-based SL/TP check (Informational Only)...")
            sl_triggered, tp_triggered = False, False
            stop_price, profit_price = 0.0, 0.0
            sl_distance = current_atr * sl_atr_mult
            tp_distance = current_atr * tp_atr_mult

            if position_side == POSITION_SIDE_LONG:
                stop_price = entry_price - sl_distance
                profit_price = entry_price + tp_distance
                logger.debug(f"Loop Check (Long): Entry={entry_price:.4f}, Curr={current_price:.4f}, SL Level ~{stop_price:.4f}, TP Level ~{profit_price:.4f}")
                if current_price <= stop_price: sl_triggered = True
                if current_price >= profit_price: tp_triggered = True
            elif position_side == POSITION_SIDE_SHORT:
                stop_price = entry_price + sl_distance
                profit_price = entry_price - tp_distance
                logger.debug(f"Loop Check (Short): Entry={entry_price:.4f}, Curr={current_price:.4f}, SL Level ~{stop_price:.4f}, TP Level ~{profit_price:.4f}")
                if current_price >= stop_price: sl_triggered = True
                if current_price <= profit_price: tp_triggered = True

            # Log triggers found by this unreliable method, but DO NOT rely on closing here.
            if sl_triggered:
                logger.warning(f"*** LOOP-BASED SL DETECTED (INFORMATIONAL) *** Price {current_price:.4f} crossed SL level ~{stop_price:.4f}. Exchange order should handle closure.")
                # DO NOT CALL close_position() here based on this check.
            elif tp_triggered:
                logger.warning(f"*** LOOP-BASED TP DETECTED (INFORMATIONAL) *** Price {current_price:.4f} crossed TP level ~{profit_price:.4f}. Exchange order should handle closure.")
                # DO NOT CALL close_position() here based on this check.
            else:
                 logger.debug("Loop-based SL/TP levels not breached in this check.")
        elif position_side != POSITION_SIDE_NONE:
             logger.warning("Active position exists, but entry price or ATR is invalid for loop-based SL/TP check.")

        # 6. Check for Entry Signals (Only if no active position or if reversing)
        logger.debug("Checking entry conditions...")

        # Define filter conditions
        passes_long_ob = not ob_available or (ob_available and bid_ask_ratio >= ORDER_BOOK_RATIO_THRESHOLD_LONG)
        passes_short_ob = not ob_available or (ob_available and bid_ask_ratio <= ORDER_BOOK_RATIO_THRESHOLD_SHORT)
        passes_volume = not REQUIRE_VOLUME_SPIKE_FOR_ENTRY or volume_spike

        # Long Entry Condition: Primary ST turns Long + Confirm ST is Long + OB supports Long (if checked) + Volume supports (if required)
        enter_long = (
            primary_long_signal and confirm_trend_up and passes_long_ob and passes_volume
        )
        logger.debug(f"Long Entry Check: PrimarySig={primary_long_signal}, ConfirmTrendUp={confirm_trend_up}, "
                     f"OB OK={passes_long_ob} (Ratio={bid_ask_ratio_str}, Threshold={ORDER_BOOK_RATIO_THRESHOLD_LONG}), "
                     f"Vol OK={passes_volume} (Spike={volume_spike}, Required={REQUIRE_VOLUME_SPIKE_FOR_ENTRY})")

        # Short Entry Condition: Primary ST turns Short + Confirm ST is Short + OB supports Short (if checked) + Volume supports (if required)
        enter_short = (
            primary_short_signal and not confirm_trend_up and passes_short_ob and passes_volume
        )
        logger.debug(f"Short Entry Check: PrimarySig={primary_short_signal}, ConfirmTrendDown={not confirm_trend_up}, "
                     f"OB OK={passes_short_ob} (Ratio={bid_ask_ratio_str}, Threshold={ORDER_BOOK_RATIO_THRESHOLD_SHORT}), "
                     f"Vol OK={passes_volume} (Spike={volume_spike}, Required={REQUIRE_VOLUME_SPIKE_FOR_ENTRY})")


        # 7. Execute Actions
        if position_side == POSITION_SIDE_NONE:
            if enter_long:
                logger.success(f"*** ENTRY SIGNAL: LONG *** for {symbol} at ~{current_price:.4f}. Placing order.")
                place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage)
                # SL/TP orders should be placed *within* place_risked_market_order
            elif enter_short:
                logger.success(f"*** ENTRY SIGNAL: SHORT *** for {symbol} at ~{current_price:.4f}. Placing order.")
                place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage)
                # SL/TP orders should be placed *within* place_risked_market_order
            else:
                logger.info("No entry signal while flat.")
        elif position_side == POSITION_SIDE_LONG:
            if enter_short: # Reversal signal
                logger.warning(f"*** REVERSAL SIGNAL: SHORT *** while LONG. Closing LONG position first.")
                close_order = close_position(exchange, symbol, position)
                if close_order:
                    logger.info(f"Closed LONG position, attempting SHORT entry...")
                    time.sleep(RETRY_DELAY_SECONDS) # Brief pause before new entry
                    place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage)
                else:
                    logger.error("Failed to close LONG position for reversal. Skipping SHORT entry.")
            else:
                logger.info("Maintaining LONG position. No exit or reversal signal.")
        elif position_side == POSITION_SIDE_SHORT:
            if enter_long: # Reversal signal
                logger.warning(f"*** REVERSAL SIGNAL: LONG *** while SHORT. Closing SHORT position first.")
                close_order = close_position(exchange, symbol, position)
                if close_order:
                    logger.info(f"Closed SHORT position, attempting LONG entry...")
                    time.sleep(RETRY_DELAY_SECONDS) # Brief pause before new entry
                    place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage)
                else:
                    logger.error("Failed to close SHORT position for reversal. Skipping LONG entry.")
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

    exchange = initialize_exchange()
    if not exchange:
        logger.critical("Exchange initialization failed. Bot cannot start.")
        return # Exit if exchange setup fails

    # --- Symbol and Leverage Setup ---
    symbol = ""
    try:
        # Get symbol input or use default
        symbol_input = input(f"Enter trading symbol (e.g., BTC/USDT:USDT) or press Enter for default [{DEFAULT_SYMBOL}]: ").strip()
        if not symbol_input:
            symbol_input = DEFAULT_SYMBOL
            logger.info(f"Using default symbol: {symbol_input}")
        else:
             logger.info(f"User provided symbol: {symbol_input}")

        # Validate symbol and check market type
        logger.info(f"Validating symbol '{symbol_input}' on {exchange.id}...")
        market = exchange.market(symbol_input)
        symbol = market['symbol']  # Use the validated, correctly formatted symbol from CCXT
        logger.success(f"Symbol validated: {symbol}")

        if not market.get('swap', False) and not market.get('future', False):
             logger.critical(f"CRITICAL: Symbol {symbol} is not a swap or futures market. Leverage cannot be set. Exiting.")
             return

        # Set Leverage
        if not set_leverage(exchange, symbol, DEFAULT_LEVERAGE):
             logger.critical(f"CRITICAL: Failed to set leverage to {DEFAULT_LEVERAGE}x for {symbol}. Exiting.")
             return

    except (ccxt.BadSymbol, KeyError) as e:
         logger.critical(f"CRITICAL: Invalid or unsupported symbol '{symbol_input}'. Error: {e}. Exiting.")
         return
    except Exception as e:
        logger.critical(f"CRITICAL: Unexpected error during symbol/leverage setup: {e}. Exiting.")
        logger.debug(traceback.format_exc())
        return

    # --- Configuration Summary ---
    logger.info("--- Scalping Bot Configuration Summary ---")
    logger.info(f" Trading Symbol:        {symbol}")
    logger.info(f" Timeframe:             {DEFAULT_INTERVAL}")
    logger.info(f" Leverage:              {DEFAULT_LEVERAGE}x")
    logger.info(f" Check Interval:        {DEFAULT_SLEEP_SECONDS} seconds")
    logger.info(f" Risk Per Trade:        {RISK_PER_TRADE_PERCENTAGE:.3%}")
    logger.info(f" Max Order Cap (USDT):  {MAX_ORDER_USDT_AMOUNT:.2f}")
    logger.info(f" SL ATR Period:         {ATR_CALCULATION_PERIOD}")
    logger.info(f" SL ATR Multiplier:     {ATR_STOP_LOSS_MULTIPLIER}")
    logger.info(f" TP ATR Multiplier:     {ATR_TAKE_PROFIT_MULTIPLIER} (Note: Loop-based TP check is unreliable)")
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
    logger.warning("--- CRITICAL: Loop-based SL/TP checks are unreliable for scalping. Implement exchange-native orders. ---")
    logger.info("---------------------------------------------")
    logger.info("Starting main trading loop... Press CTRL+C to stop.")

    # --- Main Trading Loop ---
    while True:
        loop_start_time = time.time()
        try:
            logger.debug(f"--- Top of Loop Cycle at {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

            # Fetch latest market data
            # Calculate required data length based on longest indicator lookback + buffer
            data_limit = max(100, DEFAULT_ST_ATR_LENGTH * 2, CONFIRM_ST_ATR_LENGTH * 2,
                             ATR_CALCULATION_PERIOD * 2, VOLUME_MA_PERIOD * 2, required_rows + 5) # Add buffer
            df_market = get_market_data(exchange, symbol, DEFAULT_INTERVAL, limit=data_limit)

            # Run trading logic if data is available
            if df_market is not None and not df_market.empty:
                # Pass a copy of the dataframe to prevent modification issues if df_market is reused
                trade_logic(
                    exchange=exchange, symbol=symbol, df=df_market.copy(),
                    st_len=DEFAULT_ST_ATR_LENGTH, st_mult=DEFAULT_ST_MULTIPLIER,
                    cf_st_len=CONFIRM_ST_ATR_LENGTH, cf_st_mult=CONFIRM_ST_MULTIPLIER,
                    atr_len=ATR_CALCULATION_PERIOD, vol_ma_len=VOLUME_MA_PERIOD,
                    risk_pct=RISK_PER_TRADE_PERCENTAGE, sl_atr_mult=ATR_STOP_LOSS_MULTIPLIER,
                    tp_atr_mult=ATR_TAKE_PROFIT_MULTIPLIER, leverage=DEFAULT_LEVERAGE
                )
            else:
                logger.warning("No market data received in this cycle, skipping trade logic.")

        except KeyboardInterrupt:
            logger.warning(">>> KeyboardInterrupt detected. Shutting down bot... <<<")
            # TODO: Implement graceful shutdown logic here:
            # 1. Attempt to cancel any open (non-SL/TP) orders.
            # 2. Attempt to close any existing open position.
            # Example (conceptual):
            # current_pos = get_current_position(exchange, symbol)
            # if current_pos['side'] != POSITION_SIDE_NONE:
            #    logger.warning(f"Attempting to close open {current_pos['side']} position before exiting...")
            #    close_position(exchange, symbol, current_pos)
            break  # Exit the main loop

        except ccxt.RateLimitExceeded as e:
             logger.warning(f"!!! RATE LIMIT EXCEEDED: {e}. Consider increasing SLEEP_SECONDS or disabling FETCH_ORDER_BOOK_PER_CYCLE.")
             sleep_time = DEFAULT_SLEEP_SECONDS * 5 # Sleep much longer after rate limit hit
             logger.warning(f"Sleeping for {sleep_time} seconds...")
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
             logger.error(f"Unhandled Exchange error encountered in main loop: {e}.")
             logger.debug(traceback.format_exc())
             # Continue to sleep logic

        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.exception(f"!!! UNEXPECTED CRITICAL ERROR in main loop: {e} !!!")
            logger.info(f"Attempting to continue after {DEFAULT_SLEEP_SECONDS}s sleep...")
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
    # Ensure pandas_ta is installed
    try:
        import pandas_ta
        logger.debug(f"pandas_ta version: {pandas_ta.version}")
    except ImportError:
        logger.critical("CRITICAL: 'pandas_ta' library not found. Please install it: pip install pandas_ta")
        sys.exit(1)

    main()