import os
import time
import logging
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import ccxt
from ccxt.base.error import AuthenticationError, ExchangeError, NetworkError, InsufficientFunds
import pandas as pd

# --- Configuration ---
SYMBOL = "DOTUSDT:USDT"
TIMEFRAME = "3m"
LEVERAGE = 25
CANDLE_LIMIT = 100  # Number of candles to fetch for calculations

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout),  # Use sys.stdout for stream handler
    ],
)
logger = logging.getLogger()

# --- Strategy Parameters (Dual Supertrend) ---
STRENGTH1_LENGTH = 3
STRENGTH1_MULTIPLIER = 2.0
STRENGTH2_LENGTH = 15
STRENGTH2_MULTIPLIER = 3.0

# --- Risk Management Parameters ---
MAX_DAILY_RISK_PERCENT = (
    2.0  # 2% daily risk (Note: Daily risk tracking is complex; this is risk *per trade* based on balance)
)
STOP_LOSS_ATR_MULTIPLIER = 2.0  # Use 2x ATR below/above entry for stop-loss
TAKE_PROFIT_PERCENTAGE = 2.5  # 2.5% above/below entry for take-profit

# --- Environment Variables ---
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    logger.error("Bybit API keys not found in environment variables. Please set BYBIT_API_KEY and BYBIT_API_SECRET.")
    sys.exit(1)

# --- Global State ---
current_position: Optional[Dict[str, Any]] = (
    None  # Track active position: {'symbol': str, 'side': str, 'quantity': float, 'entry_price': float, 'stop_loss': float, 'take_profit': float}
)
symbol_info: Optional[Dict] = None  # Cache symbol trading rules


# --- Exchange Initialization ---
def initialize_exchange(api_key: str, api_secret: str) -> ccxt.Exchange:
    """Initializes and configures the Bybit exchange."""
    logger.info("Initializing Bybit exchange...")
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": api_key,
                "apiSecret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",  # For perpetual futures
                    "version": "v5",
                },
            }
        )
        exchange.load_markets()  # Load markets to get symbol info
        logger.info("Exchange initialized successfully.")
        return exchange
    except AuthenticationError:
        logger.error("Failed to initialize exchange: Authentication failed. Check API keys.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize exchange: {type(e).__name__} - {e}")
        sys.exit(1)


# --- Helper Functions ---
def get_symbol_info(exchange: ccxt.Exchange, symbol: str) -> Optional[Dict]:
    """Fetches and caches trading rules for the symbol."""
    global symbol_info
    if symbol_info:
        return symbol_info

    logger.info(f"Fetching market info for {symbol}...")
    try:
        market = exchange.market(symbol)
        symbol_info = {
            "precision_amount": market["precision"]["amount"],
            "precision_price": market["precision"]["price"],
            "min_notional": market["limits"]["cost"]["min"],
            "min_amount": market["limits"]["amount"]["min"],
        }
        logger.info(f"Symbol info fetched: {symbol_info}")
        return symbol_info
    except Exception as e:
        logger.error(f"Failed to fetch symbol info for {symbol}: {type(e).__name__} - {e}")
        return None


def adjust_quantity_to_precision(quantity: float, precision: float) -> float:
    """Adjusts quantity to meet exchange precision."""
    if precision is None:
        return quantity  # No precision info, return as is
    return float(exchange.decimal_to_precision(quantity, ccxt.ROUND, precision))


def adjust_price_to_precision(price: float, precision: float) -> float:
    """Adjusts price to meet exchange precision."""
    if precision is None:
        return price  # No precision info, return as is
    return float(exchange.decimal_to_precision(price, ccxt.ROUND, precision))


# --- Data Fetching ---
def fetch_klines(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches OHLCV data with retry logic."""
    logger.info(f"Fetching {limit} {timeframe} candles for {symbol}...")
    for attempt in range(5):
        try:
            # fetch_ohlcv returns list of lists
            klines = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not klines:
                logger.warning(f"Fetched empty klines for {symbol} {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            # Set timestamp as index for pandas_ta compatibility if needed, though append=True doesn't require it
            # df.set_index('timestamp', inplace=True)
            logger.info(f"Successfully fetched {len(df)} candles.")
            return df
        except NetworkError as e:
            if attempt < 4:
                logger.warning(f"Network error fetching klines (attempt {attempt + 1}/5): {e}")
                time.sleep(exchange.rateLimit / 1000 + 2)  # Wait a bit longer than rate limit
            else:
                logger.error(f"Failed to fetch klines after 5 attempts: {type(e).__name__} - {e}")
        except ExchangeError as e:
            logger.error(f"Exchange error fetching klines: {type(e).__name__} - {e}")
            break  # Don't retry on exchange errors
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching klines: {type(e).__name__} - {e}")
            break  # Don't retry on unexpected errors
    return None


def get_current_price(exchange: ccxt.Exchange, symbol: str) -> Optional[float]:
    """Fetches the current close price for a symbol with retry logic."""
    logger.info(f"Fetching current price for {symbol}...")
    for attempt in range(5):
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker["last"])  # 'last' is typically the most recent trade price
            logger.info(f"Current price for {symbol}: {price}")
            return price
        except NetworkError as e:
            if attempt < 4:
                logger.warning(f"Network error getting current price (attempt {attempt + 1}/5): {e}")
                time.sleep(exchange.rateLimit / 1000 + 2)
            else:
                logger.error(f"Failed to get current price after 5 attempts: {type(e).__name__} - {e}")
        except ExchangeError as e:
            logger.error(f"Exchange error getting current price: {type(e).__name__} - {e}")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting current price: {type(e).__name__} - {e}")
            break
    return None


# --- Indicator Calculations ---
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all required technical indicators."""
    if df.empty:
        logger.warning("DataFrame is empty, cannot calculate indicators.")
        return df

    # Ensure the DataFrame index is suitable for pandas_ta if needed
    # df.set_index('timestamp', inplace=True) # Example if timestamp index required

    try:
        # Calculate Dual Supertrend. pandas_ta appends columns with params.
        # Names will be like 'SUPERT_10_2.0', 'SUPERTd_10_2.0', etc.
        df.ta.supertrend(length=STRENGTH1_LENGTH, multiplier=STRENGTH1_MULTIPLIER, append=True)
        df.ta.supertrend(length=STRENGTH2_LENGTH, multiplier=STRENGTH2_MULTIPLIER, append=True)

        # Calculate ATR (Corrected function name)
        df.ta.atr(length=14, append=True)

        # Add Moving Averages (Optional, not used in current signal logic but good to have)
        # df['ma20'] = df['close'].rolling(window=20).mean()
        # df['ma50'] = df['close'].rolling(window=50).mean()

        # Add RSI (Optional)
        # df['rsi'] = df['close'].ta.rsi(length=14)

        logger.info("Indicators calculated.")
        # Reset index if it was set for pandas_ta
        # df.reset_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {type(e).__name__} - {e}")
        # Return original df or empty df if calculation failed badly
        return df


# --- Strategy Logic ---
def check_entry_signal(df: pd.DataFrame) -> Optional[str]:
    """Checks for buy or sell entry signals based on dual Supertrend."""
    # Need at least enough data for indicators + 1 candle
    min_data_points = max(STRENGTH1_LENGTH, STRENGTH2_LENGTH, 14) + 1  # 14 is for ATR length
    if len(df) < min_data_points:
        logger.info(f"Not enough data ({len(df)} candles) for signal check, need at least {min_data_points}.")
        return None

    # Get latest indicator values. pandas_ta names: SUPERT_<length>_<multiplier> and SUPERTd_<length>_<multiplier>
    st1_direction_col = f"SUPERTd_{STRENGTH1_LENGTH}_{STRENGTH1_MULTIPLIER}"
    st2_direction_col = f"SUPERTd_{STRENGTH2_LENGTH}_{STRENGTH2_MULTIPLIER}"

    # Check if indicator columns exist and last values are not NaN
    if st1_direction_col not in df.columns or st2_direction_col not in df.columns:
        logger.warning(f"Supertrend direction columns not found: {st1_direction_col}, {st2_direction_col}")
        return None

    latest_st1_direction = df[st1_direction_col].iloc[-1]
    latest_st2_direction = df[st2_direction_col].iloc[-1]
    df["close"].iloc[-1]

    if pd.isna(latest_st1_direction) or pd.isna(latest_st2_direction):
        logger.info("Latest indicator values are NaN, cannot check signal.")
        return None

    # Entry conditions: Both Supertrends agree on direction
    # Buy signal: Both Supertrends indicate uptrend (+1)
    if latest_st1_direction > 0 and latest_st2_direction > 0:
        logger.info("Buy signal detected (both Supertrends up)")
        return "buy"

    # Sell signal: Both Supertrends indicate downtrend (-1)
    elif latest_st1_direction < 0 and latest_st2_direction < 0:
        logger.info("Sell signal detected (both Supertrends down)")
        return "sell"

    logger.info("No trading signal detected.")
    return None


def check_exit_signal(df: pd.DataFrame, position_side: str) -> bool:
    """
    Checks for strategic exit signals (e.g., reverse Supertrend signal).
    Note: Primary exits (SL/TP) are handled by exchange orders.
    This is for exiting if the trend reverses according to the strategy.
    """
    min_data_points = max(STRENGTH1_LENGTH, STRENGTH2_LENGTH, 14) + 1
    if len(df) < min_data_points:
        logger.info(f"Not enough data ({len(df)} candles) for exit signal check, need at least {min_data_points}.")
        return False

    st1_direction_col = f"SUPERTd_{STRENGTH1_LENGTH}_{STRENGTH1_MULTIPLIER}"
    st2_direction_col = f"SUPERTd_{STRENGTH2_LENGTH}_{STRENGTH2_MULTIPLIER}"

    if st1_direction_col not in df.columns or st2_direction_col not in df.columns:
        logger.warning(
            f"Supertrend direction columns not found for exit check: {st1_direction_col}, {st2_direction_col}"
        )
        return False

    latest_st1_direction = df[st1_direction_col].iloc[-1]
    latest_st2_direction = df[st2_direction_col].iloc[-1]

    if pd.isna(latest_st1_direction) or pd.isna(latest_st2_direction):
        logger.info("Latest indicator values are NaN, cannot check exit signal.")
        return False

    # Exit if both Supertrends reverse
    if position_side == "buy" and latest_st1_direction < 0 and latest_st2_direction < 0:
        logger.info("Strategic exit signal detected (both Supertrends reversed to down)")
        return True
    elif position_side == "sell" and latest_st1_direction > 0 and latest_st2_direction > 0:
        logger.info("Strategic exit signal detected (both Supertrends reversed to up)")
        return True

    return False


# --- Position Sizing ---
def calculate_position_size(balance: float, entry_price: float, atr: float, side: str, symbol_info: Dict) -> float:
    """
    Calculates position size based on risk percentage and ATR-derived stop loss.
    Returns quantity in base asset (e.g., BTC).
    """
    if atr <= 0:
        logger.warning("ATR is zero or negative, cannot calculate position size.")
        return 0.0
    if balance <= 0:
        logger.warning("Balance is zero or negative, cannot calculate position size.")
        return 0.0
    if entry_price <= 0:
        logger.warning("Entry price is zero or negative, cannot calculate position size.")
        return 0.0

    # Calculate potential stop loss price based on ATR
    atr_stop_distance = atr * STOP_LOSS_ATR_MULTIPLIER
    if side == "buy":
        stop_loss_price = entry_price - atr_stop_distance
    elif side == "sell":
        stop_loss_price = entry_price + atr_stop_distance
    else:
        logger.error(f"Invalid side '{side}' for position size calculation.")
        return 0.0

    # Ensure stop loss price is not too close to entry price (avoids division by zero/tiny stops)
    min_stop_distance = entry_price * 0.001  # e.g., 0.1% of price as minimum stop distance
    if abs(entry_price - stop_loss_price) < min_stop_distance:
        logger.warning(
            f"Calculated stop loss distance ({abs(entry_price - stop_loss_price):.8f}) is too small. Cannot calculate position size."
        )
        return 0.0

    # Calculate the amount of quote currency risked per base unit
    risk_per_base_unit = abs(entry_price - stop_loss_price)  # Risk in USDT per BTC

    # Calculate the total amount of quote currency allowed to risk
    total_risk_amount = balance * (MAX_DAILY_RISK_PERCENT / 100)  # Risk in USDT

    # Calculate quantity in base asset (e.g., BTC)
    calculated_quantity = total_risk_amount / risk_per_base_unit

    # Adjust quantity based on exchange precision and minimum amount
    quantity = adjust_quantity_to_precision(calculated_quantity, symbol_info["precision_amount"])

    # Check against minimum order size (notional value)
    notional_value = quantity * entry_price
    if notional_value < symbol_info["min_notional"]:
        logger.warning(
            f"Calculated notional value ({notional_value:.2f}) is below minimum ({symbol_info['min_notional']:.2f}). Adjusting quantity to meet minimum notional."
        )
        # Calculate minimum quantity based on min_notional and entry price
        min_quantity_notional = symbol_info["min_notional"] / entry_price
        quantity = adjust_quantity_to_precision(
            max(quantity, min_quantity_notional), symbol_info["precision_amount"]
        )  # Take the larger of calculated or min_notional based quantity

    # Final check against minimum amount
    if quantity < symbol_info["min_amount"]:
        logger.warning(
            f"Calculated quantity ({quantity:.8f}) is below minimum amount ({symbol_info['min_amount']:.8f}). Cannot open position."
        )
        return 0.0

    logger.info(f"Position size calculated: {quantity:.8f} {SYMBOL.split('/')[0]} (Risk: {total_risk_amount:.2f} USDT)")
    return quantity


# --- Order Execution ---
def execute_market_order(exchange: ccxt.Exchange, symbol: str, side: str, quantity: float) -> Optional[Dict]:
    """Executes a market order with retry logic."""
    logger.info(f"Attempting to execute {side} market order for {quantity:.8f} {symbol}...")
    if quantity <= 0:
        logger.warning("Order quantity is zero or negative, skipping execution.")
        return None

    for attempt in range(3):  # Fewer retries for order execution
        try:
            order = exchange.create_order(symbol, "market", side, quantity)
            logger.info(f"Market order placed: {order}")
            # Wait for order to be filled (market orders are usually instant but confirmation is good)
            order_id = order["id"]
            time.sleep(2)  # Give exchange a moment
            fetched_order = exchange.fetch_order(order_id, symbol)
            if fetched_order["status"] == "closed":
                logger.info(f"Market order {order_id} filled.")
                return fetched_order
            else:
                logger.warning(f"Market order {order_id} status is {fetched_order['status']} after waiting.")
                # In rare cases, market order might not fill immediately or fully.
                # For simplicity, we assume it fills here. More robust bots check fills.
                return fetched_order  # Return the order even if not fully closed, main logic should handle this

        except InsufficientFunds as e:
            logger.error(f"Insufficient funds to place order: {type(e).__name__} - {e}")
            return None  # No point retrying if funds are low
        except NetworkError as e:
            if attempt < 2:
                logger.warning(f"Network error executing order (attempt {attempt + 1}/3): {e}")
                time.sleep(exchange.rateLimit / 1000 + 2)
            else:
                logger.error(f"Failed to execute order after 3 attempts: {type(e).__name__} - {e}")
        except ExchangeError as e:
            logger.error(f"Exchange error executing order: {type(e).__name__} - {e}")
            # Check for specific errors like minimum order size if precision adjustment failed
            if "minimum quantity" in str(e).lower() or "min_qty" in str(e).lower():
                logger.error("Order failed due to minimum quantity/notional requirement.")
            break  # Don't retry on exchange errors
        except Exception as e:
            logger.error(f"An unexpected error occurred while executing order: {type(e).__name__} - {e}")
            break
    return None


def place_contingent_orders(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    quantity: float,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Places Stop Loss and Take Profit orders for an open position."""
    logger.info(f"Placing contingent SL/TP orders for {symbol} position (entry: {entry_price:.2f})...")

    sl_order = None
    tp_order = None
    order_side = "sell" if side == "buy" else "buy"  # SL/TP side is opposite of position side

    # Adjust prices to precision
    symbol_info = get_symbol_info(exchange, symbol)
    if not symbol_info:
        logger.error("Could not get symbol info for placing contingent orders.")
        return None, None

    sl_price_adj = adjust_price_to_precision(stop_loss_price, symbol_info["precision_price"])
    tp_price_adj = adjust_price_to_precision(take_profit_price, symbol_info["precision_price"])
    quantity_adj = adjust_quantity_to_precision(quantity, symbol_info["precision_amount"])

    logger.info(f"Attempting to place SL: {sl_price_adj:.2f}, TP: {tp_price_adj:.2f} for quantity {quantity_adj:.8f}")

    # Bybit V5 unified margin/trade supports placing SL/TP directly with main order or separately.
    # Placing separately is often more flexible. Use STOP_MARKET for SL and LIMIT for TP.
    # A STOP_MARKET order triggers at the stopPrice and executes as a market order.
    # A TAKE_PROFIT_LIMIT order triggers at the stopPrice (TP price) and executes as a limit order at the limitPrice (TP price).
    # Note: Some exchanges use different order types or parameters for SL/TP. Check CCXT docs or exchange API.

    # Bybit SL: Use STOP_MARKET. stopPrice is the trigger.
    # Bybit TP: Use TAKE_PROFIT_LIMIT or TAKE_PROFIT_MARKET.
    # Let's use TAKE_PROFIT_MARKET for simplicity (triggers at TP price, executes as market).
    # stopPrice for STOP_MARKET (SL) is the SL price.
    # stopPrice for TAKE_PROFIT_MARKET (TP) is the TP price.

    # Place Stop Loss Order
    # Bybit requires 'triggerDirection' for STOP/TAKE_PROFIT orders.
    # For a BUY position (selling to close), SL trigger is price falling (triggerDirection=2)
    # For a SELL position (buying to close), SL trigger is price rising (triggerDirection=1)
    sl_trigger_direction = 2 if side == "buy" else 1

    try:
        # Use create_order with specific type and params
        sl_order = exchange.create_order(
            symbol=symbol,
            type="STOP_MARKET",  # Stop Market order type
            side=order_side,  # Opposite side of position
            amount=quantity_adj,
            price=None,  # Market order has no limit price
            params={
                "stopPrice": sl_price_adj,
                "triggerDirection": sl_trigger_direction,
                "reduceOnly": True,  # Ensure this order only reduces position
                # 'closeOnTrigger': True # Another way to ensure reduce only on Bybit
            },
        )
        logger.info(f"Stop Loss order placed: {sl_order['id']} at stop price {sl_price_adj:.2f}")
    except Exception as e:
        logger.error(f"Failed to place Stop Loss order: {type(e).__name__} - {e}")

    # Place Take Profit Order
    # For a BUY position (selling to close), TP trigger is price rising (triggerDirection=1)
    # For a SELL position (buying to close), TP trigger is price falling (triggerDirection=2)
    tp_trigger_direction = 1 if side == "buy" else 2

    try:
        tp_order = exchange.create_order(
            symbol=symbol,
            type="TAKE_PROFIT_MARKET",  # Take Profit Market order type
            side=order_side,  # Opposite side of position
            amount=quantity_adj,
            price=None,  # Market order has no limit price
            params={
                "stopPrice": tp_price_adj,  # For TP_MARKET, stopPrice is the trigger price
                "triggerDirection": tp_trigger_direction,
                "reduceOnly": True,  # Ensure this order only reduces position
                # 'closeOnTrigger': True
            },
        )
        logger.info(f"Take Profit order placed: {tp_order['id']} at stop price {tp_price_adj:.2f}")
    except Exception as e:
        logger.error(f"Failed to place Take Profit order: {type(e).__name__} - {e}")

    return sl_order, tp_order


def close_position(exchange: ccxt.Exchange, symbol: str, side: str, quantity: float) -> Optional[Dict]:
    """Closes an open position using a market order."""
    logger.info(f"Attempting to close {side} position for {quantity:.8f} {symbol}...")
    close_side = "sell" if side == "buy" else "buy"  # Opposite side to close

    # Adjust quantity to precision
    symbol_info = get_symbol_info(exchange, symbol)
    if not symbol_info:
        logger.error("Could not get symbol info for closing position.")
        return None
    quantity_adj = adjust_quantity_to_precision(quantity, symbol_info["precision_amount"])

    try:
        order = exchange.create_order(
            symbol, "market", close_side, quantity_adj, params={"reduceOnly": True}
        )  # Ensure reduceOnly is set
        logger.info(f"Market order to close position placed: {order}")
        return order
    except Exception as e:
        logger.error(f"Failed to close position: {type(e).__name__} - {e}")
        return None


def cancel_all_orders(exchange: ccxt.Exchange, symbol: str) -> bool:
    """Cancels all open orders for a symbol."""
    logger.info(f"Attempting to cancel all open orders for {symbol}...")
    try:
        # Bybit V5 requires orderCategory and settleCoin for cancel_all_orders
        # Assuming linear perpetuals with USDT settlement
        exchange.cancel_all_orders(symbol, params={"category": "linear", "settleCoin": "USDT"})
        logger.info(f"All open orders for {symbol} cancelled.")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel orders for {symbol}: {type(e).__name__} - {e}")
        return False


def get_open_position(exchange: ccxt.Exchange, symbol: str) -> Optional[Dict]:
    """Fetches details of the current open position for the symbol."""
    logger.info(f"Fetching open positions for {symbol}...")
    try:
        # Bybit V5 fetch_positions returns a list of positions, one for each side (long/short) even if quantity is 0
        positions = exchange.fetch_positions([symbol])
        # Find the position with non-zero quantity
        for position in positions:
            if position["symbol"] == symbol and abs(position["info"].get("size", 0)) > 0:  # Bybit V5 uses 'size' field
                logger.info(
                    f"Found open position: {position['side']} {position['info']['size']} at {position['entryPrice']:.2f}"
                )
                # Map CCXT standard fields to our internal state format
                return {
                    "symbol": position["symbol"],
                    "side": position["side"],
                    "quantity": float(position["info"]["size"]),  # Use 'size' from info dict for V5
                    "entry_price": float(position["entryPrice"]),
                    # Note: SL/TP from exchange are not stored here, rely on exchange to trigger them.
                    # These are just for our internal state representation.
                    "stop_loss": position.get("stopLoss"),  # These might be None if not set via API field
                    "take_profit": position.get("takeProfit"),  # These might be None
                }
        logger.info(f"No open position found for {symbol}.")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch open positions: {type(e).__name__} - {e}")
        return None


# --- Main Bot Logic ---
def main():
    global current_position, symbol_info

    exchange = initialize_exchange(BYBIT_API_KEY, BYBIT_API_SECRET)

    # Get symbol info and set leverage once at the start
    symbol_info = get_symbol_info(exchange, SYMBOL)
    if not symbol_info:
        logger.error("Could not get symbol info, exiting.")
        sys.exit(1)

    # Set leverage (can fail if already set or other issues, log warning but don't exit)
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        logger.info(f"Leverage set to {LEVERAGE} for {SYMBOL}.")
    except Exception as e:
        logger.warning(f"Failed to set leverage: {type(e).__name__} - {e}. Please check manually.")

    # Check for existing position on startup
    current_position = get_open_position(exchange, SYMBOL)
    if current_position:
        logger.info(
            f"Bot started with existing {current_position['side'].upper()} position of {current_position['quantity']:.8f} {SYMBOL} at {current_position['entry_price']:.2f}."
        )
        # You might want to re-place SL/TP here if they weren't persistent or if bot restarted
        # This requires knowing the original SL/TP prices, which aren't stored in the position state fetched by fetch_positions.
        # A more advanced bot would store position details in a database or file.
        # For this script, we'll just acknowledge the position and wait for manual closure or strategic exit.

    # Main trading loop
    while True:
        try:
            logger.info("-" * 30)
            logger.info(f"Starting new cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Fetch data and calculate indicators
            df = fetch_klines(exchange, SYMBOL, TIMEFRAME, CANDLE_LIMIT)
            if df is None or df.empty:
                logger.warning("Failed to fetch klines or DataFrame is empty. Waiting for next cycle.")
                time.sleep(60)  # Wait longer if data fetching failed
                continue

            df = calculate_indicators(df)
            # Ensure indicators were calculated and enough data exists
            min_data_points = max(STRENGTH1_LENGTH, STRENGTH2_LENGTH, 14) + 1
            if len(df) < min_data_points or any(
                col not in df.columns
                for col in [
                    f"SUPERTd_{STRENGTH1_LENGTH}_{STRENGTH1_MULTIPLIER}",
                    f"SUPERTd_{STRENGTH2_LENGTH}_{STRENGTH2_MULTIPLIER}",
                    "ATR_14",
                ]
            ):
                logger.warning("Not enough data or indicators failed to calculate. Waiting for next cycle.")
                time.sleep(60)
                continue

            latest_atr = df["ATR_14"].iloc[-1]
            latest_close_price = df["close"].iloc[-1]  # Get the latest price from the candle data

            # --- Position Management Logic ---
            if current_position is None:
                # No position open, check for entry signal
                signal = check_entry_signal(df)

                if signal:
                    logger.info(f"Entry signal detected: {signal.upper()}")

                    # Get current balance
                    balance = 0
                    try:
                        # Fetch total balance, or free balance for the quote asset
                        # Assuming USDT settled perpetuals
                        balance_data = exchange.fetch_balance()
                        if "USDT" in balance_data and "free" in balance_data["USDT"]:
                            balance = balance_data["USDT"]["free"]
                            logger.info(f"Available USDT balance: {balance:.2f}")
                        else:
                            logger.warning("Could not fetch free USDT balance.")
                            continue  # Skip trade if balance not available
                    except Exception as e:
                        logger.error(f"Failed to fetch balance: {type(e).__name__} - {e}")
                        continue  # Skip trade on balance error

                    # Calculate position size, SL, TP
                    # Use the latest close price from the candle as potential entry price for calculation
                    potential_entry_price = latest_close_price

                    quantity = calculate_position_size(
                        balance=balance,
                        entry_price=potential_entry_price,
                        atr=latest_atr,
                        side=signal,
                        symbol_info=symbol_info,
                    )

                    if quantity > 0:
                        # Execute entry order
                        order = execute_market_order(exchange, SYMBOL, signal, quantity)

                        if order and order["status"] in [
                            "closed",
                            "open",
                        ]:  # 'open' might mean partially filled for market, but we proceed
                            # Get actual entry price and quantity from the executed order
                            # Use info dict as Bybit V5 often has details there
                            executed_qty = float(
                                order.get("filled") or order["info"].get("cumExecQty", 0)
                            )  # Use filled or cumExecQty
                            # Use average fill price if available, otherwise entryPrice from order or initial estimate
                            executed_price = float(
                                order.get("average") or order.get("price") or potential_entry_price
                            )  # average is better if partial fills happen

                            if executed_qty > 0 and executed_price > 0:
                                logger.info(
                                    f"Entry order executed: {executed_qty:.8f} at avg price {executed_price:.2f}"
                                )

                                # Calculate actual SL/TP prices based on the executed entry price
                                actual_atr_stop_distance = latest_atr * STOP_LOSS_ATR_MULTIPLIER
                                actual_stop_loss_price = (
                                    executed_price - actual_atr_stop_distance
                                    if signal == "buy"
                                    else executed_price + actual_atr_stop_distance
                                )
                                actual_take_profit_price = (
                                    executed_price * (1 + TAKE_PROFIT_PERCENTAGE / 100)
                                    if signal == "buy"
                                    else executed_price * (1 - TAKE_PROFIT_PERCENTAGE / 100)
                                )

                                # Adjust SL/TP prices to exchange precision
                                actual_stop_loss_price = adjust_price_to_precision(
                                    actual_stop_loss_price, symbol_info["precision_price"]
                                )
                                actual_take_profit_price = adjust_price_to_precision(
                                    actual_take_profit_price, symbol_info["precision_price"]
                                )

                                logger.info(
                                    f"Calculated SL: {actual_stop_loss_price:.2f}, TP: {actual_take_profit_price:.2f}"
                                )

                                # Update global position state
                                current_position = {
                                    "symbol": SYMBOL,
                                    "side": signal,
                                    "quantity": executed_qty,
                                    "entry_price": executed_price,
                                    "stop_loss": actual_stop_loss_price,
                                    "take_profit": actual_take_profit_price,
                                }
                                logger.info(f"Position opened: {current_position}")

                                # Place contingent SL/TP orders on the exchange
                                sl_order, tp_order = place_contingent_orders(
                                    exchange=exchange,
                                    symbol=SYMBOL,
                                    side=signal,
                                    quantity=executed_qty,
                                    entry_price=executed_price,
                                    stop_loss_price=actual_stop_loss_price,
                                    take_profit_price=actual_take_profit_price,
                                )
                                if sl_order or tp_order:
                                    logger.info("Contingent SL/TP orders placed.")
                                else:
                                    logger.warning("Failed to place contingent SL/TP orders. Position is unprotected!")
                            else:
                                logger.error("Entry order did not report executed quantity or price.")
                        else:
                            logger.warning("Entry order execution failed or did not complete.")
                    else:
                        logger.info("Position size calculation resulted in zero quantity. Skipping trade.")

            else:
                # Position is open, check for strategic exit signal
                logger.info(
                    f"Position currently open: {current_position['side'].upper()} {current_position['quantity']:.8f} at {current_position['entry_price']:.2f}. Monitoring..."
                )

                exit_signal = check_exit_signal(df, current_position["side"])

                if exit_signal:
                    logger.info("Strategic exit signal detected!")
                    # Close the position
                    close_order = close_position(
                        exchange, SYMBOL, current_position["side"], current_position["quantity"]
                    )

                    if close_order:
                        logger.info("Position closing order placed. Cancelling contingent orders...")
                        cancel_all_orders(exchange, SYMBOL)  # Cancel SL/TP orders
                        current_position = None  # Clear position state
                        logger.info("Position closed and state cleared.")
                    else:
                        logger.error("Failed to place position closing order.")
                        # Keep position state as is, maybe retry closing on next cycle or rely on SL/TP

                # Even if no strategic exit, periodically check if position is still open via API
                # (e.g., if SL/TP triggered on exchange)
                # This check can be less frequent.
                if int(time.time()) % 300 == 0:  # Check every 5 minutes
                    logger.info("Performing periodic check for open position status...")
                    actual_position = get_open_position(exchange, SYMBOL)
                    if actual_position is None:
                        logger.info("Periodic check found no open position on exchange. Clearing local state.")
                        current_position = None
                        # Also cancel any leftover orders just in case
                        cancel_all_orders(exchange, SYMBOL)
                    else:
                        # Update local state with actual quantity in case of partial fills or other discrepancies
                        current_position["quantity"] = actual_position["quantity"]
                        current_position["entry_price"] = actual_position[
                            "entry_price"
                        ]  # Entry price might be averaged by exchange

        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Exiting.")
            # Optionally add logic to close positions or cancel orders here
            # if current_position:
            #     logger.info("Attempting to close open position before exiting...")
            #     close_position(exchange, SYMBOL, current_position['side'], current_position['quantity'])
            # cancel_all_orders(exchange, SYMBOL) # Consider cancelling orders
            sys.exit(0)
        except Exception as e:
            logger.error(f"An unhandled error occurred in the main loop: {type(e).__name__} - {e}", exc_info=True)
            # Log the full traceback for unhandled exceptions
            time.sleep(60)  # Wait longer on unexpected errors

        # Wait before the next cycle
        time.sleep(60)  # Wait 60 seconds before fetching new data


if __name__ == "__main__":
    main()
