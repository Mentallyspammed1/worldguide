
ma2.py
  File "/data/data/com.termux/files/home/worldguide/ehlma2.py", line 296
    nonlocal is_fetching_data, is_processing_signal, last_order_book_analysis, position, current_price_global # Allow modification
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: no binding for nonlocal 'is_fetching_data' found
import ccxt.async_support as ccxt # Use async_support for asyncio
import asyncio
import os
import time
import math
import logging
import signal
import sys
from dotenv import load_dotenv

# --- Configuration ---
config = {
    'exchange': 'bybit',  # Or 'binance', 'kucoin', etc. (ensure compatibility)
    'symbol': 'BTC/USDT:USDT',  # Unified symbol format for derivatives, FARTCOIN is not real
    'timeframe': '1m',  # 1-minute candles
    'tradeAmountQuote': 10,  # Amount to trade in Quote currency (e.g., 10 USDT)
    'leverage': 25,  # Set leverage

    # Ehlers Super Smoother Periods
    'fastMaPeriod': 10,
    'slowMaPeriod': 20,

    # ATR Configuration
    'atrPeriod': 14,
    'atrSmoothPeriod': 10,  # Period for smoothing ATR

    # Trailing Stop Configuration
    'trailingStopPercent': 0.5,  # Trailing stop activation percentage (Client-side)

    # Order Book Analysis Configuration
    'orderBookDepth': 50,  # How many levels deep to fetch and analyze
    'imbalanceDepth': 10,  # Levels for simple imbalance calculation
    'imbalanceThreshold': 0.2,  # Threshold for simple imbalance signal
    'weightedImbalanceDepth': 20,  # Levels for weighted imbalance
    'maxSpreadPercent': 0.05,  # Maximum allowed spread % ((ask-bid)/mid) for market orders
    'wallDetectDepth': 20,  # How many levels near BBO to check for walls
    'wallSizeThresholdMultiplier': 5,  # Order size must be X times the average size

    # Order Placement Strategy
    'useLimitOrders': True,  # Try to use limit orders for better entry?
    'limitOrderPriceOffsetTicks': 1,  # How many ticks inside the spread

    # Bot Control
    'maxBufferSize': 200,
    'logLevel': 'info',  # 'debug', 'info', 'warn', 'error'
    'rateLimitBufferMs': 500,
    'mainLoopIntervalMs': 1000, # How often run_cycle is called
}

# --- Logging Utility ---
log_level_map = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR,
}
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('TradingBot')
logger.setLevel(log_level_map.get(config['logLevel'].lower(), logging.INFO))


# --- Indicator Calculations ---
def calculate_ehlers_super_smoother(prices, period):
    if period < 2:
        logger.warning(f"EhlersSuperSmoother period must be >= 2. Received {period}. Returning input prices.")
        return list(prices)
    if len(prices) < 2:
        return [None] * len(prices)

    result = [None] * len(prices)
    a1 = math.exp(-math.sqrt(2) * math.pi / period)
    coeff2 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / period)
    coeff3 = -a1 * a1
    coeff1 = 1 - coeff2 - coeff3

    result[0] = prices[0]
    if len(prices) > 1:
        # Initialize second point carefully
        result[1] = (coeff1 / 2) * (prices[1] + prices[0]) + coeff2 * (result[0] if result[0] is not None else prices[0])

    for i in range(2, len(prices)):
        prev1 = result[i-1] if result[i-1] is not None else prices[i-1]
        prev2 = result[i-2] if result[i-2] is not None else prices[i-2]
        result[i] = (coeff1 * (prices[i] + prices[i - 1]) / 2) + (coeff2 * prev1) + (coeff3 * prev2)
    return result

def calculate_tr(highs, lows, closes):
    if not (highs and lows and closes and len(highs) == len(lows) == len(closes)):
        logger.error('TR Calculation: Input arrays must exist and have the same length.')
        return []
    if not highs:
        return []

    first_tr_val = highs[0] - lows[0] if highs else 0
    tr_values = [max(first_tr_val, 0)]

    for i in range(1, len(highs)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr_values.append(max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ))
    return tr_values

# --- Enhanced Order Book Analysis ---
def analyze_order_book(orderbook, market, cfg):
    if not market or 'precision' not in market:
        logger.error('Market data or precision missing for order book analysis.')
        return None
    if not orderbook or not orderbook.get('bids') or not orderbook.get('asks') or \
       not orderbook['bids'] or not orderbook['asks']:
        logger.debug('Order book data missing or empty for analysis.')
        return None

    best_bid_price = orderbook['bids'][0][0]
    best_ask_price = orderbook['asks'][0][0]

    if best_bid_price <= 0 or best_ask_price <= 0 or best_bid_price >= best_ask_price:
        logger.warning(f"Invalid best bid/ask prices: Bid={best_bid_price}, Ask={best_ask_price}. Skipping analysis.")
        return None

    mid_price = (best_bid_price + best_ask_price) / 2
    spread = best_ask_price - best_bid_price
    spread_percent = (spread / mid_price) * 100

    # Simple Imbalance
    top_bids = orderbook['bids'][:cfg['imbalanceDepth']]
    top_asks = orderbook['asks'][:cfg['imbalanceDepth']]
    top_bid_volume = sum(vol for _, vol in top_bids)
    top_ask_volume = sum(vol for _, vol in top_asks)
    total_top_volume = top_bid_volume + top_ask_volume
    simple_imbalance = (top_bid_volume - top_ask_volume) / total_top_volume if total_top_volume > 0 else 0

    # Weighted Imbalance
    weighted_bid_volume = 0
    weighted_ask_volume = 0
    bids_for_weighted = orderbook['bids'][:cfg['weightedImbalanceDepth']]
    asks_for_weighted = orderbook['asks'][:cfg['weightedImbalanceDepth']]

    for price, volume in bids_for_weighted:
        distance = max(mid_price - price, 1e-9)
        weight = 1 / distance
        weighted_bid_volume += volume * weight
    for price, volume in asks_for_weighted:
        distance = max(price - mid_price, 1e-9)
        weight = 1 / distance
        weighted_ask_volume += volume * weight
    total_weighted_volume = weighted_bid_volume + weighted_ask_volume
    weighted_imbalance = (weighted_bid_volume - weighted_ask_volume) / total_weighted_volume if total_weighted_volume > 0 else 0

    # Wall Detection
    bid_wall_candidates = orderbook['bids'][:cfg['wallDetectDepth']]
    ask_wall_candidates = orderbook['asks'][:cfg['wallDetectDepth']]
    avg_bid_size = sum(vol for _, vol in bid_wall_candidates) / len(bid_wall_candidates) if bid_wall_candidates else 0
    avg_ask_size = sum(vol for _, vol in ask_wall_candidates) / len(ask_wall_candidates) if ask_wall_candidates else 0
    bid_wall_threshold = avg_bid_size * cfg['wallSizeThresholdMultiplier']
    ask_wall_threshold = avg_ask_size * cfg['wallSizeThresholdMultiplier']

    bid_walls = [{'price': price, 'size': vol} for price, vol in bid_wall_candidates if vol >= bid_wall_threshold] if bid_wall_threshold > 0 else []
    ask_walls = [{'price': price, 'size': vol} for price, vol in ask_wall_candidates if vol >= ask_wall_threshold] if ask_wall_threshold > 0 else []

    analysis = {
        'bestBid': best_bid_price,
        'bestAsk': best_ask_price,
        'midPrice': mid_price,
        'spread': spread,
        'spreadPercent': spread_percent,
        'isSpreadTooWide': spread_percent > cfg['maxSpreadPercent'],
        'simpleImbalance': simple_imbalance,
        'weightedImbalance': weighted_imbalance,
        'bidWalls': bid_walls,
        'askWalls': ask_walls,
        'timestamp': orderbook.get('timestamp') or int(time.time() * 1000),
    }

    if config['logLevel'] == 'debug':
        price_prec = market.get('precision', {}).get('price', 2)
        amount_prec = market.get('precision', {}).get('amount', 4)
        logger.debug(f"OB Analysis: Spread={spread:.{price_prec}f} ({spread_percent:.3f}%), "
                     f"SImb={simple_imbalance:.3f}, WImb={weighted_imbalance:.3f}, "
                     f"BidWalls={len(bid_walls)}, AskWalls={len(ask_walls)}")
        if analysis['isSpreadTooWide']:
            logger.debug(f"Spread ({spread_percent:.3f}%) exceeds threshold ({cfg['maxSpreadPercent']}%)")
        if bid_walls:
            logger.debug(f"Bid Walls Found: {', '.join([f'{w['size']:.{amount_prec}f}@{w['price']:.{price_prec}f}' for w in bid_walls])}")
        if ask_walls:
            logger.debug(f"Ask Walls Found: {', '.join([f'{w['size']:.{amount_prec}f}@{w['price']:.{price_prec}f}' for w in ask_walls])}")
    return analysis


# --- Global State (managed by trading_bot) ---
# These will be initialized within trading_bot and passed or accessed via a context object if needed
# For simplicity, using them as globals modified by the main bot function and its helpers.
# This is okay for a single-threaded async script, but for more complex apps, consider a class.
timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []
position = None
is_fetching_data = False
is_processing_signal = False
last_order_book_analysis = None
current_price_global = None # Used for initial price until candles arrive
shutdown_event = asyncio.Event()


# --- Main Trading Bot Logic ---
async def trading_bot():
    global timestamps, opens, highs, lows, closes, volumes, position
    global is_fetching_data, is_processing_signal, last_order_book_analysis, current_price_global

    logger.info('Starting trading bot with enhanced order book analysis...')
    logger.info(f"Configuration: {config}")

    load_dotenv()
    api_key = os.getenv('BYBIT_API_KEY') # Ensure your .env matches this
    secret = os.getenv('BYBIT_SECRET')
    if not api_key or not secret:
        logger.error('API key and secret required in .env file (e.g., BYBIT_API_KEY, BYBIT_SECRET)')
        return

    exchange = None
    try:
        exchange_class = getattr(ccxt, config['exchange'])
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap', # Important for derivatives
                'adjustForTimeDifference': True,
                'recvWindow': 10000, # Example for Bybit/Binance
            }
        })
        await exchange.load_markets()
        logger.info(f"Loaded markets from {config['exchange']}.")

        if exchange.has.get('setLeverage'):
            try:
                # Fetch fresh markets to ensure symbol details are current
                # This might be redundant if load_markets() is comprehensive enough
                # fetched_markets = await exchange.fetch_markets()
                # if any(m['symbol'] == config['symbol'] for m in fetched_markets):
                await exchange.set_leverage(config['leverage'], config['symbol'])
                logger.info(f"Leverage set to {config['leverage']}x for {config['symbol']}")
                # else:
                #    logger.warning(f"Symbol {config['symbol']} not found in fetched markets, cannot set leverage.")
            except ccxt.NetworkError as e:
                 logger.warning(f"Network error when trying to set leverage: {e}. Will proceed.")
            except ccxt.ExchangeError as e: # Catch more specific errors if possible
                logger.warning(f"Could not set leverage (maybe already set or market type mismatch): {e}")
        else:
            logger.warning(f"Exchange {config['exchange']} does not support setting leverage via set_leverage().")

    except Exception as e:
        logger.error(f"Exchange initialization failed: {e}", exc_info=True)
        if exchange: await exchange.close()
        return

    market = exchange.market(config['symbol'])
    if not market:
        logger.error(f"Symbol {config['symbol']} not found on {config['exchange']}.")
        await exchange.close()
        return
    if not market.get('contract', False) and (market.get('type') not in ['swap', 'future']):
         logger.warning(f"{config['symbol']} might not be a contract/derivative market based on CCXT info. Market type: {market.get('type')}")


    amount_precision = market.get('precision', {}).get('amount')
    price_precision = market.get('precision', {}).get('price')
    tick_size = 10**(-price_precision) if price_precision is not None else None

    if amount_precision is None or price_precision is None:
        logger.warning(f"Could not determine precision for {config['symbol']}. Order sizing/pricing might be inaccurate.")
    if tick_size is None and config['useLimitOrders'] and config['limitOrderPriceOffsetTicks'] > 0:
        logger.warning(f"Cannot determine tick size for {config['symbol']}, disabling limit order price offset. Limit orders will be placed at BBO.")
        config['limitOrderPriceOffsetTicks'] = 0

    # --- Initial Data Fetch ---
    try:
        logger.info(f"Fetching initial {config['maxBufferSize']} candles for {config['symbol']}...")
        initial_ohlcv = await exchange.fetch_ohlcv(config['symbol'], config['timeframe'], limit=config['maxBufferSize'])
        for c in initial_ohlcv:
            timestamps.append(c[0]); opens.append(c[1]); highs.append(c[2]); lows.append(c[3]); closes.append(c[4]); volumes.append(c[5])
        if timestamps:
            logger.info(f"Fetched {len(initial_ohlcv)} initial candles. Last candle time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamps[-1]/1000))} UTC")
            current_price_global = closes[-1] if closes else None
        else:
            logger.warning("No initial candles fetched.")
    except Exception as e:
        logger.error(f"Failed to fetch initial OHLCV data: {e}. Starting with WebSocket data only.")


    # --- Main Loop ---
    async def run_cycle():
        nonlocal is_fetching_data, is_processing_signal, last_order_book_analysis, position, current_price_global # Allow modification

        if is_fetching_data or is_processing_signal:
            return

        is_fetching_data = True
        current_price_cycle = closes[-1] if closes else current_price_global # Use global if closes is empty

        try:
            start_time = time.perf_counter()
            
            # Watch OHLCV and OrderBook
            # For watch_ohlcv, limit=1 means we want the latest (possibly partial) candle or the last closed one.
            # Some exchanges might send updates more frequently than timeframe completion.
            # The logic below handles new candles vs. updates to the current candle.
            # The `since` parameter for watch_ohlcv is often used to get candles *after* a certain time.
            # Here, we're just getting the latest update, so `since` might not be needed if `limit=1` gives current.
            # If watch_ohlcv with limit=1 gives an array of 1 candle, it's the current one.
            
            # Fetching one candle at a time via watch_ohlcv can be tricky.
            # It might give you the *last closed* candle if a new one hasn't formed.
            # Or it might give the *current, still-forming* candle.
            # The logic needs to handle both: if timestamp is new -> new candle, if timestamp same -> update current.
            
            # Let's adjust watchOHLCV params for Python. `limit=1` if we expect one candle.
            # `since=timestamps[-1]` if we want updates since the last known candle, but this
            # might miss the current partial candle if the exchange WS sends it without `since`.
            # For simplicity in this conversion, `limit=1` is used, assuming it gives the latest state.
            # The original JS: exchange.watchOHLCV(config.symbol, config.timeframe, undefined, 1)
            # Python: await exchange.watch_ohlcv(config['symbol'], config['timeframe'], limit=1)


            # Using Undefined for since for watch_ohlcv translates to None in python
            # The last parameter '1' for watchOHLCV in JS is limit
            ohlcv_task = exchange.watch_ohlcv(config['symbol'], config['timeframe'], since=None, limit=1)
            orderbook_task = exchange.watch_order_book(config['symbol'], limit=config['orderBookDepth'])
            
            results = await asyncio.gather(ohlcv_task, orderbook_task, return_exceptions=True)
            
            end_time = time.perf_counter()
            # logger.debug(f"Data fetch took: {(end_time - start_time):.3f} s")

            ohlcv_data_or_exc = results[0]
            orderbook_data_or_exc = results[1]

            # --- Process OHLCV ---
            new_candle_received = False
            if isinstance(ohlcv_data_or_exc, Exception):
                logger.warning(f"Failed to fetch OHLCV: {ohlcv_data_or_exc}")
            elif ohlcv_data_or_exc and len(ohlcv_data_or_exc) > 0:
                # watch_ohlcv usually returns a list of candles.
                # If limit=1, it should be a list with one candle [[ts, o, h, l, c, v]]
                latest_candle = ohlcv_data_or_exc[0] # Assuming it's the first element if limit=1
                
                if not timestamps or latest_candle[0] > timestamps[-1]:
                    timestamps.append(latest_candle[0]); opens.append(latest_candle[1]); highs.append(latest_candle[2])
                    lows.append(latest_candle[3]); closes.append(latest_candle[4]); volumes.append(latest_candle[5])
                    new_candle_received = True
                    current_price_cycle = latest_candle[4]
                    logger.debug(f"New candle: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(latest_candle[0]/1000))} C:{latest_candle[4]}")
                elif latest_candle[0] == timestamps[-1]:
                    last_idx = len(timestamps) - 1
                    highs[last_idx] = max(highs[last_idx], latest_candle[2])
                    lows[last_idx] = min(lows[last_idx], latest_candle[3])
                    closes[last_idx] = latest_candle[4]
                    volumes[last_idx] = latest_candle[5] # Or sum if partial volumes
                    current_price_cycle = latest_candle[4]
                    # logger.debug(f"Candle update: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(latest_candle[0]/1000))} C:{latest_candle[4]}")
            
            # --- Process Order Book ---
            if isinstance(orderbook_data_or_exc, Exception):
                logger.warning(f"Failed to fetch OrderBook: {orderbook_data_or_exc}")
                last_order_book_analysis = None # Invalidate old analysis
            else:
                last_order_book_analysis = analyze_order_book(orderbook_data_or_exc, market, config)
                if last_order_book_analysis and current_price_cycle is None:
                    current_price_cycle = last_order_book_analysis['midPrice']
                    logger.debug(f"Using order book mid-price ({current_price_cycle}) as current price.")
            
            current_price_global = current_price_cycle # Update global current price
            is_fetching_data = False

            # --- Buffer Management ---
            while len(closes) > config['maxBufferSize']:
                timestamps.pop(0); opens.pop(0); highs.pop(0); lows.pop(0); closes.pop(0); volumes.pop(0)

            # --- Signal Processing ---
            min_candles_needed = max(config['slowMaPeriod'], config['atrPeriod'] + config['atrSmoothPeriod']) + 2
            if len(closes) >= min_candles_needed and not is_processing_signal and current_price_cycle is not None:
                is_processing_signal = True
                process_start_time = time.perf_counter()
                try:
                    # Calculate Indicators
                    fast_ma_values = calculate_ehlers_super_smoother(closes, config['fastMaPeriod'])
                    slow_ma_values = calculate_ehlers_super_smoother(closes, config['slowMaPeriod'])
                    tr_values = calculate_tr(highs, lows, closes)
                    valid_tr = [v for v in tr_values if isinstance(v, (int, float))]
                    atr_values = calculate_ehlers_super_smoother(valid_tr, config['atrPeriod'])
                    valid_atr = [v for v in atr_values if isinstance(v, (int, float))]
                    smoothed_atr_values = calculate_ehlers_super_smoother(valid_atr, config['atrSmoothPeriod'])

                    last_fast_ma = fast_ma_values[-1] if fast_ma_values and fast_ma_values[-1] is not None else None
                    last_slow_ma = slow_ma_values[-1] if slow_ma_values and slow_ma_values[-1] is not None else None
                    prev_fast_ma = fast_ma_values[-2] if len(fast_ma_values) > 1 and fast_ma_values[-2] is not None else None
                    prev_slow_ma = slow_ma_values[-2] if len(slow_ma_values) > 1 and slow_ma_values[-2] is not None else None
                    # last_atr = atr_values[-1] if atr_values and atr_values[-1] is not None else None # Original ATR from TR
                    last_atr = valid_atr[-1] if valid_atr and valid_atr[-1] is not None else None # Smoothed ATR (first pass)
                    last_smoothed_atr = smoothed_atr_values[-1] if smoothed_atr_values and smoothed_atr_values[-1] is not None else None


                    indicators_valid = all(isinstance(v, (int, float)) for v in [last_fast_ma, last_slow_ma, prev_fast_ma, prev_slow_ma, last_atr, last_smoothed_atr])

                    if not indicators_valid:
                        logger.debug('Indicator calculation resulted in None/invalid values, skipping signal check.')
                        is_processing_signal = False
                        return

                    price_fmt_prec = price_precision if price_precision is not None else 2
                    logger.debug(f"Indicators: Px={current_price_cycle:.{price_fmt_prec}f}, FastMA={last_fast_ma:.{price_fmt_prec}f}, "
                                 f"SlowMA={last_slow_ma:.{price_fmt_prec}f}, ATR={last_atr:.4f}, SmATR={last_smoothed_atr:.4f}")

                    # Position Management & Trading Logic
                    # 1. Update Trailing Stop
                    if position:
                        if position['side'] == 'long':
                            position['highestPrice'] = max(position.get('highestPrice', position['entryPrice']), current_price_cycle)
                            potential_stop = position['highestPrice'] * (1 - config['trailingStopPercent'] / 100)
                            position['stopPrice'] = max(position.get('stopPrice', position['entryPrice'] * (1 - config['trailingStopPercent'] / 100)), potential_stop)
                        elif position['side'] == 'short':
                            position['lowestPrice'] = min(position.get('lowestPrice', position['entryPrice']), current_price_cycle)
                            potential_stop = position['lowestPrice'] * (1 + config['trailingStopPercent'] / 100)
                            position['stopPrice'] = min(position.get('stopPrice', position['entryPrice'] * (1 + config['trailingStopPercent'] / 100)), potential_stop)
                        # logger.debug(f"{position['side']} Pos Update: TrailStop={position['stopPrice']:.{price_fmt_prec}f}")
                    
                    # 2. Check Exit Conditions
                    exit_signal_reason = None
                    if position:
                        stop_price_hit = (position['side'] == 'long' and position.get('stopPrice') and current_price_cycle <= position['stopPrice']) or \
                                         (position['side'] == 'short' and position.get('stopPrice') and current_price_cycle >= position['stopPrice'])
                        ma_cross_exit = (position['side'] == 'long' and last_fast_ma < last_slow_ma and prev_fast_ma >= prev_slow_ma) or \
                                        (position['side'] == 'short' and last_fast_ma > last_slow_ma and prev_fast_ma <= prev_slow_ma)
                        
                        if stop_price_hit: exit_signal_reason = 'Trailing Stop Hit'
                        elif ma_cross_exit: exit_signal_reason = 'MA Crossover Exit'

                        if exit_signal_reason:
                            logger.info(f"Exit Signal ({position['side']}): {exit_signal_reason}. Price: {current_price_cycle:.{price_fmt_prec}f}, Stop: {position.get('stopPrice'):.{price_fmt_prec}f}")
                            try:
                                close_side = 'sell' if position['side'] == 'long' else 'buy'
                                order = await exchange.create_order(config['symbol'], 'market', close_side, position['amount'], params={'reduceOnly': True})
                                logger.info(f"Position closed via Market Order ({position['side']} {position['amount']} {market['base']}). Order ID: {order['id']}")
                                position = None
                            except Exception as e:
                                logger.error(f"Error closing {position['side']} position: {e}", exc_info=True)
                    
                    # 3. Check Entry Conditions
                    if not position and not exit_signal_reason: # No current position and no exit signal processed in this cycle
                        is_bullish_cross = prev_fast_ma <= prev_slow_ma and last_fast_ma > last_slow_ma
                        is_bearish_cross = prev_fast_ma >= prev_slow_ma and last_fast_ma < last_slow_ma
                        is_volatile = last_atr > last_smoothed_atr # ATR > Smoothed ATR

                        entry_signal_reason = None
                        entry_side = None
                        ob_factors_allow_entry = False
                        preferred_order_type = 'market'
                        limit_price = None

                        if last_order_book_analysis:
                            ob = last_order_book_analysis
                            if ob['isSpreadTooWide'] and not config['useLimitOrders']:
                                logger.info(f"Skipping entry: Spread ({ob['spreadPercent']:.3f}%) too wide for market order.")
                            else:
                                if is_bullish_cross and is_volatile:
                                    ob_confirms_buy = ob['simpleImbalance'] > config['imbalanceThreshold'] or ob['weightedImbalance'] > 0.05
                                    no_immediate_ask_wall = not (ob['askWalls'] and ob['askWalls'][0]['price'] <= ob['bestAsk'] + (tick_size or 0) * 3)
                                    if ob_confirms_buy and no_immediate_ask_wall:
                                        entry_signal_reason = 'Long Entry: MA Cross + Volatility + OB Confirm'
                                        entry_side = 'buy'
                                        ob_factors_allow_entry = True
                                        if config['useLimitOrders'] and not ob['isSpreadTooWide'] and tick_size is not None:
                                            preferred_order_type = 'limit'
                                            limit_price = exchange.price_to_precision(config['symbol'], ob['bestBid'] + tick_size * config['limitOrderPriceOffsetTicks'])
                                        elif config['useLimitOrders'] and ob['isSpreadTooWide']:
                                            logger.debug("Spread too wide, falling back to market order for long.")
                                            preferred_order_type = 'market'
                                    else: logger.debug(f"Skipping Long: MA+Vol OK. OB Confirm={ob_confirms_buy}, No Ask Wall={no_immediate_ask_wall}")
                                elif is_bearish_cross and is_volatile:
                                    ob_confirms_sell = ob['simpleImbalance'] < -config['imbalanceThreshold'] or ob['weightedImbalance'] < -0.05
                                    no_immediate_bid_wall = not (ob['bidWalls'] and ob['bidWalls'][0]['price'] >= ob['bestBid'] - (tick_size or 0) * 3)
                                    if ob_confirms_sell and no_immediate_bid_wall:
                                        entry_signal_reason = 'Short Entry: MA Cross + Volatility + OB Confirm'
                                        entry_side = 'sell'
                                        ob_factors_allow_entry = True
                                        if config['useLimitOrders'] and not ob['isSpreadTooWide'] and tick_size is not None:
                                            preferred_order_type = 'limit'
                                            limit_price = exchange.price_to_precision(config['symbol'], ob['bestAsk'] - tick_size * config['limitOrderPriceOffsetTicks'])
                                        elif config['useLimitOrders'] and ob['isSpreadTooWide']:
                                            logger.debug("Spread too wide, falling back to market order for short.")
                                            preferred_order_type = 'market'
                                    else: logger.debug(f"Skipping Short: MA+Vol OK. OB Confirm={ob_confirms_sell}, No Bid Wall={no_immediate_bid_wall}")
                        else:
                            logger.debug("Skipping entry check: No recent order book analysis available.")

                        # Execute Entry Order
                        if entry_signal_reason and entry_side and ob_factors_allow_entry:
                            logger.info(f"Entry Signal: {entry_signal_reason}. Type: {preferred_order_type}{f' @ {limit_price}' if limit_price else ''}")
                            try:
                                amount_in_base_unrounded = config['tradeAmountQuote'] / current_price_cycle
                                amount_in_base = exchange.amount_to_precision(config['symbol'], amount_in_base_unrounded)
                                
                                logger.info(f"Attempting to {entry_side} {amount_in_base} {market['base']} ({config['tradeAmountQuote']} {market['quote']})")

                                if market.get('limits', {}).get('amount', {}).get('min') and float(amount_in_base) < market['limits']['amount']['min']:
                                     logger.error(f"Order amount {amount_in_base} is below market minimum {market['limits']['amount']['min']}. Skipping order.")
                                     raise ccxt.InvalidOrder("Order amount too small")

                                order = None
                                order_params = {} # Add 'timeInForce', 'postOnly' etc. if needed

                                if preferred_order_type == 'limit' and limit_price:
                                    order = await exchange.create_order(config['symbol'], 'limit', entry_side, amount_in_base, limit_price, order_params)
                                    logger.info(f"Limit order placed: {order['side']} {order['amount']} @ {order['price']}. ID: {order['id']}")
                                else: # Market order
                                    order = await exchange.create_order(config['symbol'], 'market', entry_side, amount_in_base, params=order_params)
                                    logger.info(f"Market order placed: {order['side']} {order['amount']}. Avg Price: {order.get('average', 'N/A')}, ID: {order['id']}")
                                
                                # Update Position State
                                entry_price_candidate = None
                                if preferred_order_type == 'limit':
                                    entry_price_candidate = order.get('price')
                                else: # market order
                                    entry_price_candidate = order.get('average')
                                
                                entry_price_actual = entry_price_candidate if entry_price_candidate is not None else current_price_cycle
                                
                                # Handle filled amount. If 'filled' is 0.0 or None, use 'amount' as an estimate for open positions
                                filled_amount = order.get('filled') if order.get('filled', 0.0) > 0 else order.get('amount')
                                
                                if filled_amount and float(filled_amount) > 0:
                                    position = {
                                        'side': 'long' if entry_side == 'buy' else 'short',
                                        'entryPrice': entry_price_actual,
                                        'amount': float(filled_amount), # Ensure it's a float
                                        'highestPrice': entry_price_actual, # For trailing stop
                                        'lowestPrice': entry_price_actual,  # For trailing stop
                                        'stopPrice': None
                                    }
                                    position['stopPrice'] = position['entryPrice'] * (1 - config['trailingStopPercent'] / 100) if position['side'] == 'long' \
                                        else position['entryPrice'] * (1 + config['trailingStopPercent'] / 100)
                                    logger.info(f"Position opened: {position['side']}, Entry: {position['entryPrice']:.{price_fmt_prec}f}, "
                                                f"Amount: {position['amount']}, Initial Stop: {position['stopPrice']:.{price_fmt_prec}f}")
                                else:
                                    logger.warning(f"Order placed (ID: {order['id']}, Type: {preferred_order_type}) but filled amount is 0 or unavailable. Position state not updated.")
                                    # Potentially cancel if it was a limit order not filled, or wait. This example assumes partial/full fill.

                            except ccxt.InsufficientFunds as e:
                                logger.error(f"Insufficient funds for {entry_side} order: {e}", exc_info=True)
                            except ccxt.InvalidOrder as e:
                                logger.error(f"Invalid order parameters for {entry_side} order: {e}", exc_info=True)
                            except Exception as e:
                                logger.error(f"Error placing {entry_side} {preferred_order_type} order: {e}", exc_info=True)
                
                except Exception as processing_error:
                    logger.error(f"Error during signal processing: {processing_error}", exc_info=True)
                finally:
                    process_end_time = time.perf_counter()
                    # logger.debug(f"Signal processing took: {(process_end_time - process_start_time):.3f} s")
                    is_processing_signal = False
            
            elif len(closes) < min_candles_needed:
                # logger.info(f"Waiting for more data... Have {len(closes)}/{min_candles_needed} candles.")
                pass
            elif current_price_cycle is None:
                logger.debug("Waiting for current price data...")


        except ccxt.NetworkError as e:
            logger.warning(f"Network/Exchange issue in main loop: {e}. Retrying after delay...")
            await asyncio.sleep(5) # Longer delay for network issues
        except ccxt.ExchangeNotAvailable as e:
            logger.warning(f"Exchange not available: {e}. Retrying after delay...")
            await asyncio.sleep(10)
        except ccxt.RequestTimeout as e:
            logger.warning(f"Request timed out: {e}. Retrying after delay...")
            await asyncio.sleep(5)
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed! Check API keys. Stopping bot. Error: {e}")
            shutdown_event.set() # Signal shutdown
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded: {e}. Waiting longer...")
            wait_time_ms = (exchange.rate_limit or 1000) * 2 + config['rateLimitBufferMs']
            await asyncio.sleep(wait_time_ms / 1000.0)
        except Exception as e:
            logger.error(f"Error in main trading loop: {e}", exc_info=True)
            await asyncio.sleep(2) # Generic error delay
        finally:
            is_fetching_data = False # Ensure this is reset
            is_processing_signal = False # Ensure this is reset in case of early exit from try block
    
    # --- Start Loop & Graceful Shutdown ---
    logger.info("Starting main execution cycle...")
    
    while not shutdown_event.is_set():
        cycle_start_time = time.perf_counter()
        await run_cycle()
        cycle_end_time = time.perf_counter()
        elapsed_ms = (cycle_end_time - cycle_start_time) * 1000
        sleep_duration_ms = max(0, config['mainLoopIntervalMs'] - elapsed_ms)
        if sleep_duration_ms > 0 :
            await asyncio.sleep(sleep_duration_ms / 1000.0)
    
    # --- Shutdown sequence ---
    logger.info("Shutdown signal received. Finalizing...")
    if config['useLimitOrders']:
        try:
            logger.info("Attempting to cancel open limit orders...")
            open_orders = await exchange.fetch_open_orders(config['symbol'])
            cancelled_count = 0
            for order in open_orders:
                if order['type'] == 'limit':
                    try:
                        await exchange.cancel_order(order['id'], config['symbol'])
                        logger.info(f"Cancelled open limit order {order['id']}")
                        cancelled_count += 1
                        await asyncio.sleep(0.3) # Small delay between cancellations
                    except Exception as cancel_error:
                        logger.error(f"Failed to cancel order {order['id']}: {cancel_error}")
            logger.info(f"Cancelled {cancelled_count} open limit orders.")
        except Exception as e:
            logger.error(f"Error fetching or cancelling open orders during shutdown: {e}")

    if position:
        logger.warning(f"Closing open {position['side']} position via market order before shutdown...")
        try:
            close_side = 'sell' if position['side'] == 'long' else 'buy'
            await exchange.create_order(config['symbol'], 'market', close_side, position['amount'], params={'reduceOnly': True})
            logger.info("Position closed successfully.")
        except Exception as e:
            logger.error(f"EMERGENCY: Failed to close position on shutdown: {e}. Manual intervention may be required!")
    else:
        logger.info("No open position to close.")

    await exchange.close()
    logger.info("Shutdown complete.")


async def main():
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    
    def signal_handler(sig, frame):
        logger.info(f"Signal {signal.Signals(sig).name} received. Initiating shutdown...")
        shutdown_event.set()

    # For Windows, SIGINT is tricky with asyncio. KeyboardInterrupt is more reliable.
    # For POSIX, SIGINT and SIGTERM are common.
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal_handler) # Using signal.signal for simplicity here
            # loop.add_signal_handler(sig, lambda s=sig: shutdown_event.set()) # More asyncio idiomatic
    
    try:
        await trading_bot()
    except KeyboardInterrupt: # Handles Ctrl+C more gracefully if signal handlers are tricky
        logger.info("KeyboardInterrupt received. Shutting down...")
        shutdown_event.set()
        # If trading_bot is already running and awaiting shutdown_event, this will also trigger cleanup.
        # Need to ensure trading_bot() finishes its cleanup if it was interrupted directly.
        # The current structure with shutdown_event should handle this.
    except Exception as e:
        logger.error(f"Unhandled critical error during bot execution: {e}", exc_info=True)
        # Ensure exchange connection is closed if it exists and an error occurs outside trading_bot main try/finally
        # This is complex if `exchange` is not accessible here. Better to handle in trading_bot.
    finally:
        # If shutdown_event was set by KeyboardInterrupt, and trading_bot wasn't awaiting it,
        # we might need to explicitly call a cleanup function if `exchange` was initialized.
        # However, the current `trading_bot` structure includes a finally block for exchange.close()
        # if it exits its main loop due to `shutdown_event`.
        logger.info("Bot process is terminating.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt: # Catch KI if asyncio.run itself is interrupted.
        logger.info("Application terminated by KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Fatal error in asyncio.run: {e}", exc_info=True)
    finally:
        sys.exit(0)


