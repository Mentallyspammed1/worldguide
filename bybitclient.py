#============ FILE: pyrmethus_market_analyzer_v3.1_async_stable.py ============
# ==============================================================================
# 🔥 Pyrmethus's Arcane Market Analyzer v3.1 ASYNC Stable Edition 🔥
# Harnessing WebSockets (ccxt.pro) for real-time updates & asyncio for concurrency!
# Enhanced stability, responsiveness, and interactive limit orders.
# Fixed market fetching, robust cleanup. Use responsibly!
# ==============================================================================
import asyncio
import decimal
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# CCXT Pro for WebSocket and async REST support
import ccxt.pro as ccxtpro
import ccxt # Keep standard ccxt for exceptions

from colorama import Back, Fore, Style, init
from dotenv import load_dotenv

# Initialize Colorama & Decimal Precision
init(autoreset=True)
decimal.getcontext().prec = 50

# Load .env
load_dotenv()
print(f"{Fore.CYAN}{Style.DIM}# Loading ancient scrolls (.env)...{Style.RESET_ALL}")

# ==============================================================================
# Configuration Loading (Robust)
# ==============================================================================
def get_config_value(key: str, default: Any, cast_type: Callable = str) -> Any:
    """Gets value from environment or uses default, casting to specified type."""
    value = os.environ.get(key)
    if value is None: return default
    try:
        if cast_type == bool:
            if value.lower() in ('true', '1', 'yes', 'y'): return True
            if value.lower() in ('false', '0', 'no', 'n'): return False
            raise ValueError(f"Invalid boolean string: {value}")
        if cast_type == decimal.Decimal: return decimal.Decimal(value)
        return cast_type(value)
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        print(f"{Fore.YELLOW}# Config Warning: Invalid value for {key} ('{value}'). Using default: {default}. Error: {e}{Style.RESET_ALL}")
        return default

CONFIG = {
    # --- Core ---
    "API_KEY": get_config_value("BYBIT_API_KEY", None),
    "API_SECRET": get_config_value("BYBIT_API_SECRET", None),
    "VERBOSE_DEBUG": get_config_value("VERBOSE_DEBUG", False, bool),

    # --- Market & Order Book ---
    "SYMBOL": get_config_value("BYBIT_SYMBOL", "BTCUSDT", str).upper(),
    "EXCHANGE_TYPE": get_config_value("BYBIT_EXCHANGE_TYPE", 'linear', str),
    "VOLUME_THRESHOLDS": {
        'high': get_config_value("VOLUME_THRESHOLD_HIGH", decimal.Decimal('10'), decimal.Decimal),
        'medium': get_config_value("VOLUME_THRESHOLD_MEDIUM", decimal.Decimal('2'), decimal.Decimal) },
    "REFRESH_INTERVAL": get_config_value("REFRESH_INTERVAL", 15, int), # Display refresh rate
    "MAX_ORDERBOOK_DEPTH_DISPLAY": get_config_value("MAX_ORDERBOOK_DEPTH_DISPLAY", 35, int),
    "ORDER_FETCH_LIMIT": get_config_value("ORDER_FETCH_LIMIT", 50, int), # WS orderbook depth
    "DEFAULT_EXCHANGE_TYPE": 'linear',
    "CONNECT_TIMEOUT": get_config_value("CONNECT_TIMEOUT", 35000, int),
    "RETRY_DELAY_NETWORK_ERROR": get_config_value("RETRY_DELAY_NETWORK_ERROR", 10, int),
    "RETRY_DELAY_RATE_LIMIT": get_config_value("RETRY_DELAY_RATE_LIMIT", 60, int),

    # --- Indicators ---
    "INDICATOR_TIMEFRAME": get_config_value("INDICATOR_TIMEFRAME", '15m', str),
    "SMA_PERIOD": get_config_value("SMA_PERIOD", 9, int),
    "SMA2_PERIOD": get_config_value("SMA2_PERIOD", 50, int),
    "EMA1_PERIOD": get_config_value("EMA1_PERIOD", 12, int),
    "EMA2_PERIOD": get_config_value("EMA2_PERIOD", 89, int),
    "MOMENTUM_PERIOD": get_config_value("MOMENTUM_PERIOD", 10, int),
    "RSI_PERIOD": get_config_value("RSI_PERIOD", 14, int),
    "STOCH_K_PERIOD": get_config_value("STOCH_K_PERIOD", 14, int),
    "STOCH_D_PERIOD": get_config_value("STOCH_D_PERIOD", 3, int),
    "STOCH_RSI_OVERSOLD": get_config_value("STOCH_RSI_OVERSOLD", decimal.Decimal('20'), decimal.Decimal),
    "STOCH_RSI_OVERBOUGHT": get_config_value("STOCH_RSI_OVERBOUGHT", decimal.Decimal('80'), decimal.Decimal),
    "MIN_OHLCV_CANDLES": max(
        get_config_value("SMA_PERIOD", 9, int), get_config_value("SMA2_PERIOD", 50, int),
        get_config_value("EMA1_PERIOD", 12, int), get_config_value("EMA2_PERIOD", 89, int),
        get_config_value("MOMENTUM_PERIOD", 10, int) + 1,
        get_config_value("RSI_PERIOD", 14, int) + get_config_value("STOCH_K_PERIOD", 14, int) + get_config_value("STOCH_D_PERIOD", 3, int)
    ) + 1, # Ensure enough history for longest lookback + calc

    # --- Display ---
    "PIVOT_TIMEFRAME": get_config_value("PIVOT_TIMEFRAME", '15m', str),
    "PNL_PRECISION": get_config_value("PNL_PRECISION", 2, int),
    "MIN_PRICE_DISPLAY_PRECISION": get_config_value("MIN_PRICE_DISPLAY_PRECISION", 3, int),
    "STOCH_RSI_DISPLAY_PRECISION": get_config_value("STOCH_RSI_DISPLAY_PRECISION", 2, int),
    "VOLUME_DISPLAY_PRECISION": get_config_value("VOLUME_DISPLAY_PRECISION", 2, int),
    "BALANCE_DISPLAY_PRECISION": get_config_value("BALANCE_DISPLAY_PRECISION", 2, int),

    # --- Trading ---
    "FETCH_BALANCE_ASSET": get_config_value("FETCH_BALANCE_ASSET", "USDT", str),
    "DEFAULT_ORDER_TYPE": get_config_value("DEFAULT_ORDER_TYPE", "limit", str).lower(),
    "LIMIT_ORDER_SELECTION_TYPE": get_config_value("LIMIT_ORDER_SELECTION_TYPE", "interactive", str).lower(),
    "POSITION_IDX": get_config_value("BYBIT_POSITION_IDX", 0, int), # For Hedge Mode

    # --- Intervals ---
    "BALANCE_POS_FETCH_INTERVAL": get_config_value("BALANCE_POS_FETCH_INTERVAL", 45, int),
    "OHLCV_FETCH_INTERVAL": get_config_value("OHLCV_FETCH_INTERVAL", 100, int),
}

# Fibonacci Ratios
FIB_RATIOS = { 'r3': decimal.Decimal('1.000'), 'r2': decimal.Decimal('0.618'), 'r1': decimal.Decimal('0.382'),
               's1': decimal.Decimal('0.382'), 's2': decimal.Decimal('0.618'), 's3': decimal.Decimal('1.000') }

# ==============================================================================
# Shared State & Global Exchange Instance
# ==============================================================================
latest_data: Dict[str, Any] = { "ticker": None, "orderbook": None, "balance": None, "positions": [], "indicator_ohlcv": None,
                                "pivot_ohlcv": None, "indicators": {}, "pivots": None, "market_info": None,
                                "last_update_times": {}, "connection_status": {"ws_ticker": "init", "ws_ob": "init", "rest": "init"} }
exchange: Optional[ccxtpro.Exchange] = None # Global exchange instance

# ==============================================================================
# Utility Functions (print_color, verbose_print, termux_toast, format_decimal)
# ==============================================================================
def print_color(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL, end: str = '\n', **kwargs: Any) -> None:
    """Prints colorized text."""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end, **kwargs)

def verbose_print(text: str, color: str = Fore.CYAN, style: str = Style.DIM) -> None:
    """Prints only if VERBOSE_DEBUG is True."""
    if CONFIG.get("VERBOSE_DEBUG", False): print_color(f"# DEBUG: {text}", color=color, style=style)

def termux_toast(message: str, duration: str = "short") -> None:
    """Displays a Termux toast notification."""
    try:
        safe_message = ''.join(c for c in str(message) if c.isalnum() or c in ' .,!?-:+%$/=()[]{}')[:100]
        subprocess.run(['termux-toast', '-d', duration, safe_message], check=True, capture_output=True, timeout=5)
    except FileNotFoundError: pass
    except Exception as e: verbose_print(f"Toast error: {e}")

def format_decimal(value: Optional[Union[decimal.Decimal, str, int, float]], reported_precision: int, min_display_precision: Optional[int] = None) -> str:
    """Formats decimal values for display."""
    if value is None: return "N/A"
    try:
        d_value = decimal.Decimal(str(value)) if not isinstance(value, decimal.Decimal) else value
        display_precision = max(int(reported_precision), 0)
        if min_display_precision is not None: display_precision = max(display_precision, max(int(min_display_precision), 0))
        quantizer = decimal.Decimal('1') / (decimal.Decimal('10') ** display_precision)
        rounded_value = d_value.quantize(quantizer, rounding=decimal.ROUND_HALF_UP)
        formatted_str = "{:f}".format(rounded_value)
        if '.' in formatted_str:
            integer_part, decimal_part = formatted_str.split('.')
            decimal_part = decimal_part[:display_precision].ljust(display_precision, '0')
            formatted_str = f"{integer_part}.{decimal_part}" if display_precision > 0 else integer_part
        elif display_precision > 0: formatted_str += '.' + '0' * display_precision
        return formatted_str
    except Exception as e: verbose_print(f"FormatDecimal Error: {e}"); return str(value)

# ==============================================================================
# Async Market Info Fetcher (Replaces direct fetch_market call)
# ==============================================================================
async def get_market_info(exchange_instance: ccxtpro.Exchange, symbol: str) -> Optional[Dict[str, Any]]:
    """ ASYNCHRONOUSLY Fetches and returns market information. """
    try:
        print_color(f"{Fore.CYAN}# Querying market runes for {symbol} (async)...", style=Style.DIM, end='\r')
        # Use await for load_markets as it's I/O bound
        if not exchange_instance.markets or symbol not in exchange_instance.markets:
            verbose_print(f"Market list needs loading/refresh for {symbol}.")
            await exchange_instance.load_markets(True) # Use await here!
        sys.stdout.write("\033[K")

        if symbol not in exchange_instance.markets:
             print_color(f"Symbol '{symbol}' still not found after async market reload.", color=Fore.RED, style=Style.BRIGHT)
             return None

        market = exchange_instance.market(symbol)
        verbose_print(f"Async market info retrieved for {symbol}")

        # --- Precision/Limits Extraction (same as sync version) ---
        price_prec = 8; amount_prec = 8; min_amount = decimal.Decimal('0')
        try: price_prec = int(decimal.Decimal(str(market.get('precision', {}).get('price', '1e-8'))).log10() * -1)
        except: verbose_print(f"Could not parse price precision for {symbol}, using default {price_prec}")
        try: amount_prec = int(decimal.Decimal(str(market.get('precision', {}).get('amount', '1e-8'))).log10() * -1)
        except: verbose_print(f"Could not parse amount precision for {symbol}, using default {amount_prec}")
        try: min_amount = decimal.Decimal(str(market.get('limits', {}).get('amount', {}).get('min', '0')))
        except: verbose_print(f"Could not parse min amount for {symbol}, using default {min_amount}")

        price_tick_size = decimal.Decimal('1') / (decimal.Decimal('10') ** price_prec) if price_prec >= 0 else decimal.Decimal('1')
        amount_step = decimal.Decimal('1') / (decimal.Decimal('10') ** amount_prec) if amount_prec >= 0 else decimal.Decimal('1')

        return {'price_precision': price_prec, 'amount_precision': amount_prec, 'min_amount': min_amount,
                'price_tick_size': price_tick_size, 'amount_step': amount_step, 'symbol': symbol }
    except ccxt.BadSymbol: sys.stdout.write("\033[K"); print_color(f"Symbol '{symbol}' invalid.", color=Fore.RED, style=Style.BRIGHT); return None
    except ccxt.NetworkError as e: sys.stdout.write("\033[K"); print_color(f"Network error getting market info (async): {e}", color=Fore.YELLOW); return None
    except Exception as e:
        sys.stdout.write("\033[K"); print_color(f"Critical error getting market info (async): {e}", color=Fore.RED)
        traceback.print_exc(); return None

# ==============================================================================
# Indicator Calculation Functions (Same as v2.1)
# ==============================================================================
# calculate_sma, calculate_ema, calculate_momentum, calculate_fib_pivots,
# calculate_rsi_manual, calculate_stoch_rsi_manual
# (Include these functions from v2.1 - no changes)
# ... Functions omitted for brevity ...
def calculate_sma(data: List[Union[str, float, int, decimal.Decimal]], period: int) -> Optional[decimal.Decimal]:
    if not data or len(data) < period: return None
    try: return sum(decimal.Decimal(str(p)) for p in data[-period:]) / decimal.Decimal(period)
    except Exception as e: verbose_print(f"SMA Calc Error: {e}"); return None
def calculate_ema(data: List[Union[str, float, int, decimal.Decimal]], period: int) -> Optional[List[decimal.Decimal]]:
    if not data or len(data) < period: return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data]; ema_values: List[Optional[decimal.Decimal]] = [None] * len(data)
        mult = decimal.Decimal(2) / (decimal.Decimal(period) + 1); sma_init = sum(decimal_data[:period]) / decimal.Decimal(period)
        ema_values[period - 1] = sma_init
        for i in range(period, len(data)):
            if ema_values[i - 1] is None: continue
            ema_values[i] = (decimal_data[i] - ema_values[i - 1]) * mult + ema_values[i - 1]
        return [ema for ema in ema_values if ema is not None]
    except Exception as e: verbose_print(f"EMA Calc Error: {e}"); return None
def calculate_momentum(data: List[Union[str, float, int, decimal.Decimal]], period: int) -> Optional[decimal.Decimal]:
    if not data or len(data) <= period: return None
    try: return decimal.Decimal(str(data[-1])) - decimal.Decimal(str(data[-period - 1]))
    except Exception as e: verbose_print(f"Momentum Calc Error: {e}"); return None
def calculate_fib_pivots(high: Optional[Any], low: Optional[Any], close: Optional[Any]) -> Optional[Dict[str, decimal.Decimal]]:
    if None in [high, low, close]: return None
    try:
        h, l, c = decimal.Decimal(str(high)), decimal.Decimal(str(low)), decimal.Decimal(str(close))
        if h <= 0 or l <= 0 or c <= 0 or h < l: return None
        pp = (h + l + c) / 3; range_hl = max(h - l, decimal.Decimal('0'))
        return {'R3': pp+(range_hl*FIB_RATIOS['r3']), 'R2': pp+(range_hl*FIB_RATIOS['r2']), 'R1': pp+(range_hl*FIB_RATIOS['r1']),
                'PP': pp, 'S1': pp-(range_hl*FIB_RATIOS['s1']), 'S2': pp-(range_hl*FIB_RATIOS['s2']), 'S3': pp-(range_hl*FIB_RATIOS['s3'])}
    except Exception as e: verbose_print(f"Pivot Calc Error: {e}"); return None
def calculate_rsi_manual(close_prices_list: List[Any], period: int = 14) -> Tuple[Optional[List[decimal.Decimal]], Optional[str]]:
    if not close_prices_list or len(close_prices_list) <= period: return None, f"Data short ({len(close_prices_list)}<{period+1})"
    try:
        prices = [decimal.Decimal(str(p)) for p in close_prices_list]; deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        if not deltas: return None, "No changes"
        gains = [d if d > 0 else decimal.Decimal('0') for d in deltas]; losses = [-d if d < 0 else decimal.Decimal('0') for d in deltas]
        if len(gains) < period: return None, f"Deltas short ({len(gains)}<{period})"
        avg_gain, avg_loss = sum(gains[:period]) / decimal.Decimal(period), sum(losses[:period]) / decimal.Decimal(period)
        rsi_values = []; rs = decimal.Decimal('inf') if avg_loss == 0 else avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal('100'))
        for i in range(period, len(gains)):
            avg_gain, avg_loss = (avg_gain*(period-1) + gains[i]) / decimal.Decimal(period), (avg_loss*(period-1) + losses[i]) / decimal.Decimal(period)
            rs = decimal.Decimal('inf') if avg_loss == 0 else avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal('100'))
        return rsi_values, None
    except Exception as e: verbose_print(f"RSI Calc Error: {e}"); return None, str(e)
def calculate_stoch_rsi_manual(rsi_values: List[decimal.Decimal], k_period: int = 14, d_period: int = 3) -> Tuple[Optional[decimal.Decimal], Optional[decimal.Decimal], Optional[str]]:
    if not rsi_values or len(rsi_values) < k_period: return None, None, f"RSI short ({len(rsi_values)}<{k_period})"
    try:
        valid_rsi = [r for r in rsi_values if r is not None and r.is_finite()]
        if len(valid_rsi) < k_period: return None, None, f"Valid RSI short ({len(valid_rsi)}<{k_period})"
        stoch_k_values = []
        for i in range(k_period - 1, len(valid_rsi)):
            window = valid_rsi[i - k_period + 1 : i + 1]; curr, mini, maxi = window[-1], min(window), max(window)
            stoch_k = decimal.Decimal('50') if maxi == mini else max(decimal.Decimal('0'), min(decimal.Decimal('100'), ((curr - mini) / (maxi - mini)) * 100))
            stoch_k_values.append(stoch_k)
        if not stoch_k_values: return None, None, "%K empty"
        latest_k = stoch_k_values[-1]
        if len(stoch_k_values) < d_period: return latest_k, None, f"%K short ({len(stoch_k_values)}<{d_period})"
        latest_d = sum(stoch_k_values[-d_period:]) / decimal.Decimal(d_period)
        return latest_k, latest_d, None
    except Exception as e: verbose_print(f"StochRSI Calc Error: {e}"); return None, None, str(e)

# ==============================================================================
# Data Processing & Analysis (Order Book)
# ==============================================================================
def analyze_orderbook_data(orderbook: Dict, market_info: Dict, config: Dict) -> Optional[Dict]:
    """Analyzes raw order book data."""
    if not orderbook or not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list): return None
    price_prec, amount_prec, vol_disp_prec = market_info['price_precision'], market_info['amount_precision'], config["VOLUME_DISPLAY_PRECISION"]
    vol_thr, display_depth = config["VOLUME_THRESHOLDS"], config["MAX_ORDERBOOK_DEPTH_DISPLAY"]
    analyzed_ob = {'symbol': orderbook.get('symbol', market_info['symbol']), 'timestamp': orderbook.get('datetime', time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(orderbook.get('timestamp', time.time()*1000)/1000))),
                   'asks': [], 'bids': [], 'ask_total_volume_fetched': decimal.Decimal('0'), 'bid_total_volume_fetched': decimal.Decimal('0'),
                   'ask_vwap_fetched': decimal.Decimal('0'), 'bid_vwap_fetched': decimal.Decimal('0'), 'volume_imbalance_ratio_fetched': decimal.Decimal('0'),
                   'cumulative_ask_volume_displayed': decimal.Decimal('0'), 'cumulative_bid_volume_displayed': decimal.Decimal('0')}
    ask_sum = decimal.Decimal('0')
    for i, level in enumerate(orderbook['asks']):
        try: price, volume = decimal.Decimal(str(level[0])), decimal.Decimal(str(level[1]))
        except: continue
        analyzed_ob['ask_total_volume_fetched'] += volume; ask_sum += price * volume
        if i < display_depth:
            analyzed_ob['cumulative_ask_volume_displayed'] += volume; vol_str = format_decimal(volume, amount_prec, vol_disp_prec)
            color, style = (Fore.LIGHTRED_EX, Style.BRIGHT) if volume >= vol_thr['high'] else (Fore.RED, Style.NORMAL) if volume >= vol_thr['medium'] else (Fore.WHITE, Style.NORMAL)
            analyzed_ob['asks'].append({'price': price, 'volume': volume, 'volume_str': vol_str, 'color': color, 'style': style, 'cumulative_volume': format_decimal(analyzed_ob['cumulative_ask_volume_displayed'], amount_prec, vol_disp_prec)})
    bid_sum = decimal.Decimal('0')
    for i, level in enumerate(orderbook['bids']):
        try: price, volume = decimal.Decimal(str(level[0])), decimal.Decimal(str(level[1]))
        except: continue
        analyzed_ob['bid_total_volume_fetched'] += volume; bid_sum += price * volume
        if i < display_depth:
            analyzed_ob['cumulative_bid_volume_displayed'] += volume; vol_str = format_decimal(volume, amount_prec, vol_disp_prec)
            color, style = (Fore.LIGHTGREEN_EX, Style.BRIGHT) if volume >= vol_thr['high'] else (Fore.GREEN, Style.NORMAL) if volume >= vol_thr['medium'] else (Fore.WHITE, Style.NORMAL)
            analyzed_ob['bids'].append({'price': price, 'volume': volume, 'volume_str': vol_str, 'color': color, 'style': style, 'cumulative_volume': format_decimal(analyzed_ob['cumulative_bid_volume_displayed'], amount_prec, vol_disp_prec)})
    ask_tot, bid_tot = analyzed_ob['ask_total_volume_fetched'], analyzed_ob['bid_total_volume_fetched']
    if ask_tot > 0: analyzed_ob['ask_vwap_fetched'], analyzed_ob['volume_imbalance_ratio_fetched'] = ask_sum / ask_tot, bid_tot / ask_tot
    else: analyzed_ob['volume_imbalance_ratio_fetched'] = decimal.Decimal('inf') if bid_tot > 0 else decimal.Decimal('0')
    if bid_tot > 0: analyzed_ob['bid_vwap_fetched'] = bid_sum / bid_tot
    return analyzed_ob

# ==============================================================================
# WebSocket Watcher Tasks
# ==============================================================================
async def watch_ticker(exchange_pro: ccxtpro.Exchange, symbol: str):
    """Watches the ticker stream and updates shared state."""
    verbose_print(f"Starting ticker watcher for {symbol}")
    latest_data["connection_status"]["ws_ticker"] = "connecting"
    while True:
        try:
            ticker = await exchange_pro.watch_ticker(symbol)
            latest_data["ticker"] = ticker
            latest_data["last_update_times"]["ticker"] = time.time()
            if latest_data["connection_status"]["ws_ticker"] != "ok":
                 latest_data["connection_status"]["ws_ticker"] = "ok"; verbose_print(f"Ticker WS connected for {symbol}")
            # verbose_print(f"Ticker: {ticker.get('last')}") # Less noisy ticker log
        except (ccxt.NetworkError, ccxt.RequestTimeout, asyncio.TimeoutError) as e:
            print_color(f"# Ticker WS Net Err: {e}", Fore.YELLOW); latest_data["connection_status"]["ws_ticker"] = "error"; await asyncio.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])
        except ccxt.ExchangeError as e: print_color(f"# Ticker WS Exch Err: {e}", Fore.RED); latest_data["connection_status"]["ws_ticker"] = "error"; await asyncio.sleep(CONFIG["RETRY_DELAY_RATE_LIMIT"])
        except Exception as e:
            print_color(f"# Ticker WS Error: {e}", Fore.RED, style=Style.BRIGHT); latest_data["connection_status"]["ws_ticker"] = "error"; traceback.print_exc(); await asyncio.sleep(30)

async def watch_orderbook(exchange_pro: ccxtpro.Exchange, symbol: str):
    """Watches the order book stream and updates shared state."""
    verbose_print(f"Starting orderbook watcher for {symbol}")
    latest_data["connection_status"]["ws_ob"] = "connecting"
    while True:
        try:
            orderbook = await exchange_pro.watch_order_book(symbol, limit=CONFIG["ORDER_FETCH_LIMIT"])
            market_info = latest_data.get("market_info") # Ensure market_info is ready
            if market_info:
                analyzed_ob = analyze_orderbook_data(orderbook, market_info, CONFIG)
                if analyzed_ob: # Only update if analysis succeeded
                    latest_data["orderbook"] = analyzed_ob
                    latest_data["last_update_times"]["orderbook"] = time.time()
                    if latest_data["connection_status"]["ws_ob"] != "ok":
                        latest_data["connection_status"]["ws_ob"] = "ok"; verbose_print(f"OrderBook WS connected for {symbol}")
                    # verbose_print(f"OB Update: A:{analyzed_ob['asks'][0]['price'] if analyzed_ob['asks'] else 'N/A'} B:{analyzed_ob['bids'][0]['price'] if analyzed_ob['bids'] else 'N/A'}")
            else: await asyncio.sleep(0.5) # Brief pause if market_info not ready
        except (ccxt.NetworkError, ccxt.RequestTimeout, asyncio.TimeoutError) as e:
            print_color(f"# OB WS Net Err: {e}", Fore.YELLOW); latest_data["connection_status"]["ws_ob"] = "error"; await asyncio.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])
        except ccxt.ExchangeError as e: print_color(f"# OB WS Exch Err: {e}", Fore.RED); latest_data["connection_status"]["ws_ob"] = "error"; await asyncio.sleep(CONFIG["RETRY_DELAY_RATE_LIMIT"])
        except Exception as e:
            print_color(f"# OB WS Error: {e}", Fore.RED, style=Style.BRIGHT); latest_data["connection_status"]["ws_ob"] = "error"; traceback.print_exc(); await asyncio.sleep(30)

# ==============================================================================
# Periodic Data Fetching Task (REST via ccxt.pro async methods)
# ==============================================================================
async def fetch_periodic_data(exchange_pro: ccxtpro.Exchange, symbol: str):
    """Periodically fetches balance, positions, and OHLCV using REST."""
    bal_pos_interval = CONFIG["BALANCE_POS_FETCH_INTERVAL"]
    ohlcv_interval = CONFIG["OHLCV_FETCH_INTERVAL"]
    last_ohlcv_fetch_time = 0
    min_ohlcv = CONFIG["MIN_OHLCV_CANDLES"]
    ind_tf, piv_tf = CONFIG['INDICATOR_TIMEFRAME'], CONFIG['PIVOT_TIMEFRAME']

    while True:
        now = time.time()
        fetch_bal_pos = True # Always fetch balance/pos in this cycle
        fetch_ohlcv = (now - last_ohlcv_fetch_time) >= ohlcv_interval

        tasks_to_run = {}
        if fetch_bal_pos:
            tasks_to_run['balance'] = exchange_pro.fetch_balance()
            tasks_to_run['positions'] = exchange_pro.fetch_positions([symbol])
        if fetch_ohlcv:
            history_needed = min_ohlcv + 5
            tasks_to_run['indicator_ohlcv'] = exchange_pro.fetch_ohlcv(symbol, ind_tf, limit=history_needed)
            tasks_to_run['pivot_ohlcv'] = exchange_pro.fetch_ohlcv(symbol, piv_tf, limit=2)

        if not tasks_to_run: # Should not happen with current logic, but safety
            await asyncio.sleep(bal_pos_interval); continue

        verbose_print(f"Running periodic fetches: {list(tasks_to_run.keys())}")
        try:
            results = await asyncio.gather(*tasks_to_run.values(), return_exceptions=True)
            latest_data["connection_status"]["rest"] = "ok" # Assume ok unless exception below

            res_map = dict(zip(tasks_to_run.keys(), results)) # Map results back to keys

            # Process results carefully
            if 'balance' in res_map:
                bal_res = res_map['balance']
                if isinstance(bal_res, dict): latest_data["balance"] = bal_res.get('total', {}).get(CONFIG["FETCH_BALANCE_ASSET"]); latest_data["last_update_times"]["balance"] = time.time()
                elif isinstance(bal_res, Exception): print_color(f"# Err fetch balance: {bal_res}", Fore.YELLOW); latest_data["connection_status"]["rest"] = "error"
            if 'positions' in res_map:
                pos_res = res_map['positions']
                if isinstance(pos_res, list): latest_data["positions"] = [p for p in pos_res if p.get('symbol') == symbol and p.get('contracts') and decimal.Decimal(str(p['contracts'])) != 0]; latest_data["last_update_times"]["positions"] = time.time()
                elif isinstance(pos_res, Exception): print_color(f"# Err fetch positions: {pos_res}", Fore.YELLOW); latest_data["connection_status"]["rest"] = "error"

            # Process OHLCV only if fetched this cycle
            if fetch_ohlcv:
                last_ohlcv_fetch_time = now # Update time even if fetch failed
                ind_res = res_map.get('indicator_ohlcv')
                if isinstance(ind_res, list):
                    if len(ind_res) >= min_ohlcv:
                        latest_data["indicator_ohlcv"] = ind_res; latest_data["last_update_times"]["indicator_ohlcv"] = time.time(); verbose_print(f"Ind OHLCV updated ({len(ind_res)})")
                        await calculate_and_store_indicators() # Recalculate
                    else: print_color(f"# Warn: Insufficient Ind OHLCV ({len(ind_res)}<{min_ohlcv})", Fore.YELLOW); latest_data["indicator_ohlcv"] = None
                elif isinstance(ind_res, Exception): print_color(f"# Err fetch ind OHLCV: {ind_res}", Fore.YELLOW); latest_data["connection_status"]["rest"] = "error"; latest_data["indicator_ohlcv"] = None

                piv_res = res_map.get('pivot_ohlcv')
                if isinstance(piv_res, list) and len(piv_res) > 0:
                    latest_data["pivot_ohlcv"] = piv_res; latest_data["last_update_times"]["pivot_ohlcv"] = time.time(); verbose_print(f"Piv OHLCV updated ({len(piv_res)})")
                    await calculate_and_store_pivots() # Recalculate
                elif isinstance(piv_res, Exception): print_color(f"# Err fetch piv OHLCV: {piv_res}", Fore.YELLOW); latest_data["connection_status"]["rest"] = "error"; latest_data["pivots"] = None
                elif not isinstance(piv_res, Exception): latest_data["pivots"] = None # Clear if fetch returned empty/None

        except Exception as e:
            print_color(f"# Error in periodic gather: {e}", Fore.RED); latest_data["connection_status"]["rest"] = "error"; traceback.print_exc()

        await asyncio.sleep(bal_pos_interval) # Wait for the next balance/pos interval

# ==============================================================================
# Indicator & Pivot Calculation Tasks (Run after data fetch)
# ==============================================================================
async def calculate_and_store_indicators():
    """Calculates all indicators based on stored OHLCV data."""
    verbose_print("Calculating indicators...")
    ohlcv, min_candles = latest_data.get("indicator_ohlcv"), CONFIG["MIN_OHLCV_CANDLES"]
    indicators: Dict[str, Dict] = { k: {'value': None, 'error': None} for k in ['sma1', 'sma2', 'ema1', 'ema2', 'momentum', 'stoch_rsi'] }
    error_msg = None
    if not ohlcv or not isinstance(ohlcv, list) or len(ohlcv) < min_candles: error_msg = f"OHLCV Missing/Short ({len(ohlcv) if ohlcv else 0}<{min_candles})"
    if error_msg:
        for k in indicators: indicators[k]['error'] = error_msg
        latest_data["indicators"] = indicators; verbose_print(f"Indicator calc skipped: {error_msg}"); return
    try:
        close_prices = [c[4] for c in ohlcv if isinstance(c, list) and len(c) >= 5]
        if len(close_prices) < min_candles: raise ValueError("Close price extraction failed/short")

        indicators['sma1']['value'] = calculate_sma(close_prices, CONFIG["SMA_PERIOD"]); # ... (similar value assignments) ...
        indicators['sma2']['value'] = calculate_sma(close_prices, CONFIG["SMA2_PERIOD"])
        ema1_res = calculate_ema(close_prices, CONFIG["EMA1_PERIOD"]); indicators['ema1']['value'] = ema1_res[-1] if ema1_res else None
        ema2_res = calculate_ema(close_prices, CONFIG["EMA2_PERIOD"]); indicators['ema2']['value'] = ema2_res[-1] if ema2_res else None
        indicators['momentum']['value'] = calculate_momentum(close_prices, CONFIG["MOMENTUM_PERIOD"])

        # Mark errors if calc returned None
        for k in ['sma1', 'sma2', 'ema1', 'ema2', 'momentum']:
             if indicators[k]['value'] is None: indicators[k]['error'] = "Calc Fail"

        # StochRSI Chain
        rsi_list, rsi_err = calculate_rsi_manual(close_prices, CONFIG["RSI_PERIOD"])
        if rsi_err: indicators['stoch_rsi']['error'] = f"RSI: {rsi_err}"
        elif rsi_list:
            k, d, stoch_err = calculate_stoch_rsi_manual(rsi_list, CONFIG["STOCH_K_PERIOD"], CONFIG["STOCH_D_PERIOD"])
            indicators['stoch_rsi'].update({'k': k, 'd': d, 'error': stoch_err}) # Update k, d, error
        else: indicators['stoch_rsi']['error'] = "RSI List Empty"

        latest_data["indicators"] = indicators; latest_data["last_update_times"]["indicators"] = time.time(); verbose_print("Indicators calculated.")
    except Exception as e:
         print_color(f"# Indicator Calc Error: {e}", Fore.RED); traceback.print_exc()
         for k in indicators: indicators[k]['error'] = "Calc Exception"; latest_data["indicators"] = indicators

async def calculate_and_store_pivots():
    """Calculates pivots based on stored pivot OHLCV data."""
    verbose_print("Calculating pivots...")
    pivot_ohlcv = latest_data.get("pivot_ohlcv")
    calculated_pivots = None
    if pivot_ohlcv and len(pivot_ohlcv) > 0:
        prev_candle = pivot_ohlcv[0]
        if isinstance(prev_candle, list) and len(prev_candle) >= 5:
             calculated_pivots = calculate_fib_pivots(prev_candle[2], prev_candle[3], prev_candle[4])
             if calculated_pivots: latest_data["pivots"] = calculated_pivots; latest_data["last_update_times"]["pivots"] = time.time(); verbose_print("Pivots calculated.")
             else: latest_data["pivots"] = None; verbose_print("Pivot calculation failed.")
        else: latest_data["pivots"] = None; verbose_print("Invalid prev candle for pivots.")
    else: latest_data["pivots"] = None; verbose_print("No pivot OHLCV data.")

# ==============================================================================
# Display Functions (Adapted for Shared State - Same as v3.0 draft)
# ==============================================================================
# display_combined_analysis_async, display_header, display_ticker_and_trend,
# display_indicators, display_position, display_pivots, display_orderbook,
# display_volume_analysis
# (Include these functions from v3.0 draft - they read `latest_data`)
# ... Functions omitted for brevity ...
def display_combined_analysis_async(shared_data: Dict, market_info: Dict, config: Dict) -> Tuple[Dict[int, decimal.Decimal], Dict[int, decimal.Decimal]]:
    """Orchestrates display using data from the shared state."""
    global exchange
    ticker_info = shared_data.get("ticker"); analyzed_ob = shared_data.get("orderbook"); indicators_info = shared_data.get("indicators", {})
    positions_list = shared_data.get("positions", []); pivots_info = shared_data.get("pivots"); balance_info = shared_data.get("balance")
    position_info_processed = {'has_position': False, 'position': None, 'unrealizedPnl': None}
    if positions_list:
        position_info_processed['has_position'] = True; current_pos = positions_list[0]; position_info_processed['position'] = current_pos
        try: pnl_raw = current_pos.get('unrealizedPnl'); position_info_processed['unrealizedPnl'] = decimal.Decimal(str(pnl_raw)) if pnl_raw is not None else None
        except: position_info_processed['unrealizedPnl'] = None
    ts_ob, ts_tk = shared_data.get("last_update_times", {}).get("orderbook"), shared_data.get("last_update_times", {}).get("ticker")
    timestamp_str = analyzed_ob['timestamp'] if analyzed_ob and analyzed_ob.get('timestamp') else ticker_info['datetime'] if ticker_info and ticker_info.get('datetime') else time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(ts_ob)) + "(OB)" if ts_ob else time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(ts_tk)) + "(Tk)" if ts_tk else time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime()) + "(Now)"
    symbol = market_info['symbol']
    print("\033[H\033[J", end="") # Clear screen
    display_header(symbol, timestamp_str, balance_info, config)
    last_price = display_ticker_and_trend(ticker_info, indicators_info, config, market_info)
    display_indicators(indicators_info, config, market_info, last_price)
    display_position(position_info_processed, ticker_info, market_info, config)
    display_pivots(pivots_info, last_price, market_info, config)
    ask_map, bid_map = display_orderbook(analyzed_ob, market_info, config)
    display_volume_analysis(analyzed_ob, market_info, config)
    stat_str = " | ".join(f"{k.upper()}:{Fore.GREEN if v == 'ok' else Fore.YELLOW if v == 'connecting' else Fore.RED}{v}{Style.RESET_ALL}" for k, v in shared_data.get("connection_status", {}).items())
    print_color(f"--- Status: {stat_str} ---", color=Fore.MAGENTA, style=Style.DIM)
    return ask_map, bid_map
# ... Include the other display functions (header, ticker, indicators etc.) from v3.0 ...
def display_header(symbol: str, timestamp: str, balance_info: Optional[Any], config: Dict) -> None:
    print_color("=" * 85, Fore.CYAN); print_color(f"📜 Pyrmethus Market Vision: {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{timestamp}", Fore.CYAN)
    balance_str, asset, prec = f"{Fore.YELLOW}N/A{Style.RESET_ALL}", config["FETCH_BALANCE_ASSET"], config["BALANCE_DISPLAY_PRECISION"]
    if balance_info is not None:
        try: balance_str = f"{Fore.GREEN}{format_decimal(balance_info, prec, prec)} {asset}{Style.RESET_ALL}"
        except Exception as e: verbose_print(f"Balance Disp Err: {e}"); balance_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"
    print_color(f"💰 Balance ({asset}): {balance_str}"); print_color("-" * 85, Fore.CYAN)
def display_ticker_and_trend(ticker_info: Optional[Dict], indicators_info: Dict, config: Dict, market_info: Dict) -> Optional[decimal.Decimal]:
    price_prec, min_disp_prec = market_info['price_precision'], config["MIN_PRICE_DISPLAY_PRECISION"]; last_price: Optional[decimal.Decimal] = None
    curr_price_str, price_color = f"{Fore.YELLOW}N/A{Style.RESET_ALL}", Fore.WHITE
    if ticker_info and ticker_info.get('last') is not None:
        try:
            last_price = decimal.Decimal(str(ticker_info['last'])); sma1 = indicators_info.get('sma1', {}).get('value')
            if sma1: price_color = Fore.GREEN if last_price > sma1 else Fore.RED if last_price < sma1 else Fore.YELLOW
            curr_price_str = f"{price_color}{Style.BRIGHT}{format_decimal(last_price, price_prec, min_disp_prec)}{Style.RESET_ALL}"
        except Exception as e: verbose_print(f"Ticker Disp Err: {e}"); last_price = None; curr_price_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"
    sma1_val, sma1_err = indicators_info.get('sma1', {}).get('value'), indicators_info.get('sma1', {}).get('error')
    sma1_p, tf = config['SMA_PERIOD'], config['INDICATOR_TIMEFRAME']; trend_str, trend_color = f"SMA({sma1_p}@{tf}): -", Fore.YELLOW
    if sma1_err: trend_str, trend_color = f"SMA({sma1_p}@{tf}): {sma1_err}", Fore.YELLOW
    elif sma1_val and last_price:
        sma1_fmt = format_decimal(sma1_val, price_prec, min_disp_prec); trend_color = Fore.GREEN if last_price > sma1_val else Fore.RED if last_price < sma1_val else Fore.YELLOW
        trend_str = f"{'Above' if last_price > sma1_val else 'Below' if last_price < sma1_val else 'On'} SMA ({sma1_fmt})"
    elif sma1_val: trend_str, trend_color = f"SMA({sma1_p}@{tf}): {format_decimal(sma1_val, price_prec, min_disp_prec)} (No Price)", Fore.WHITE
    else: trend_str = f"SMA({sma1_p}@{tf}): Unavailable"
    print_color(f"  Last Price: {curr_price_str} | {trend_color}{trend_str}{Style.RESET_ALL}"); return last_price
def display_indicators(indicators_info: Dict, config: Dict, market_info: Dict, last_price: Optional[decimal.Decimal]) -> None:
    price_prec, min_disp_prec, stoch_prec, tf = market_info['price_precision'], config["MIN_PRICE_DISPLAY_PRECISION"], config["STOCH_RSI_DISPLAY_PRECISION"], config['INDICATOR_TIMEFRAME']
    line1, line2 = [], []
    sma2_v, sma2_e, sma2_p = indicators_info.get('sma2',{}).get('value'), indicators_info.get('sma2',{}).get('error'), config['SMA2_PERIOD']
    line1.append(f"SMA2({sma2_p}): {Fore.YELLOW}{'Err' if sma2_e else 'N/A' if sma2_v is None else format_decimal(sma2_v, price_prec, min_disp_prec)}{Style.RESET_ALL}")
    ema1_v, ema2_v, ema_e = indicators_info.get('ema1',{}).get('value'), indicators_info.get('ema2',{}).get('value'), indicators_info.get('ema1',{}).get('error') or indicators_info.get('ema2',{}).get('error')
    ema1_p, ema2_p = config['EMA1_PERIOD'], config['EMA2_PERIOD']; ema_str = f"EMA({ema1_p}/{ema2_p}): {Fore.YELLOW}{'Err' if ema_e else 'N/A'}{Style.RESET_ALL}"
    if ema1_v and ema2_v: ema_str = f"EMA({ema1_p}/{ema2_p}): {(Fore.GREEN if ema1_v > ema2_v else Fore.RED if ema1_v < ema2_v else Fore.YELLOW)}{format_decimal(ema1_v, price_prec, min_disp_prec)}/{format_decimal(ema2_v, price_prec, min_disp_prec)}{Style.RESET_ALL}"
    line1.append(ema_str)
    print_color(f"  {' | '.join(line1)}")
    mom_v, mom_e, mom_p = indicators_info.get('momentum',{}).get('value'), indicators_info.get('momentum',{}).get('error'), config['MOMENTUM_PERIOD']
    mom_str = f"Mom({mom_p}): {Fore.YELLOW}{'Err' if mom_e else 'N/A'}{Style.RESET_ALL}"
    if mom_v is not None: mom_str = f"Mom({mom_p}): {(Fore.GREEN if mom_v > 0 else Fore.RED if mom_v < 0 else Fore.YELLOW)}{format_decimal(mom_v, price_prec, min_disp_prec)}{Style.RESET_ALL}"
    line2.append(mom_str)
    st_k, st_d, st_e = indicators_info.get('stoch_rsi',{}).get('k'), indicators_info.get('stoch_rsi',{}).get('d'), indicators_info.get('stoch_rsi',{}).get('error')
    rsi_p, k_p, d_p = config['RSI_PERIOD'], config['STOCH_K_PERIOD'], config['STOCH_D_PERIOD']
    stoch_str = f"StochRSI({rsi_p},{k_p},{d_p}): {Fore.YELLOW}{st_e[:10]+'..' if isinstance(st_e, str) else 'Err' if st_e else 'N/A'}{Style.RESET_ALL}"
    if st_k is not None:
        k_f, d_f = format_decimal(st_k, stoch_prec), format_decimal(st_d, stoch_prec) if st_d is not None else "N/A"
        osold, obought = config['STOCH_RSI_OVERSOLD'], config['STOCH_RSI_OVERBOUGHT']; k_color, signal = Fore.WHITE, ""
        if st_k < osold and (st_d is None or st_d < osold): k_color, signal = Fore.GREEN, "(OS)"
        elif st_k > obought and (st_d is None or st_d > obought): k_color, signal = Fore.RED, "(OB)"
        elif st_d is not None: k_color = Fore.LIGHTGREEN_EX if st_k > st_d else Fore.LIGHTRED_EX if st_k < st_d else Fore.WHITE
        stoch_str = f"StochRSI: {k_color}K={k_f}{Style.RESET_ALL} D={k_color}{d_f}{Style.RESET_ALL} {k_color}{signal}{Style.RESET_ALL}"
    line2.append(stoch_str)
    print_color(f"  {' | '.join(line2)}")
def display_position(position_info: Dict, ticker_info: Optional[Dict], market_info: Dict, config: Dict) -> None:
    pnl_prec, price_prec, amount_prec, min_disp_prec = config["PNL_PRECISION"], market_info['price_precision'], market_info['amount_precision'], config["MIN_PRICE_DISPLAY_PRECISION"]
    pnl_str = f"{Fore.LIGHTBLACK_EX}Position: None{Style.RESET_ALL}"
    if position_info.get('has_position'):
        pos = position_info['position']; side, size_str, entry_str = pos.get('side','N/A').capitalize(), pos.get('contracts','0'), pos.get('entryPrice','0')
        quote = pos.get('quoteAsset', config['FETCH_BALANCE_ASSET']); pnl_val = position_info.get('unrealizedPnl')
        try:
            size, entry = decimal.Decimal(size_str), decimal.Decimal(entry_str) if entry_str else decimal.Decimal('0')
            size_fmt, entry_fmt = format_decimal(size, amount_prec), format_decimal(entry, price_prec, min_disp_prec)
            side_color = Fore.GREEN if side.lower() == 'long' else Fore.RED if side.lower() == 'short' else Fore.WHITE
            if pnl_val is None and ticker_info and ticker_info.get('last') and entry > 0 and size != 0:
                last_p = decimal.Decimal(str(ticker_info['last']))
                pnl_val = (last_p - entry) * size if side.lower() == 'long' else (entry - last_p) * size
            pnl_val_str, pnl_color = ("N/A", Fore.WHITE) if pnl_val is None else (format_decimal(pnl_val, pnl_prec), Fore.GREEN if pnl_val >= 0 else Fore.RED)
            pnl_str = f"Position: {side_color}{side} {size_fmt}{Style.RESET_ALL} @ {Fore.YELLOW}{entry_fmt}{Style.RESET_ALL} | uPNL: {pnl_color}{pnl_val_str} {quote}{Style.RESET_ALL}"
        except Exception as e: verbose_print(f"Pos Disp Err: {e}"); pnl_str = f"{Fore.YELLOW}Position Data Err{Style.RESET_ALL}"
    print_color(f"  {pnl_str}")
def display_pivots(pivots_info: Optional[Dict], last_price: Optional[decimal.Decimal], market_info: Dict, config: Dict) -> None:
    print_color(f"--- Fibonacci Pivots (Prev {config['PIVOT_TIMEFRAME']}) ---", Fore.BLUE)
    if not pivots_info: print_color(f"  {Fore.YELLOW}Pivot data unavailable.{Style.RESET_ALL}"); return
    price_prec, min_disp_prec, width = market_info['price_precision'], config["MIN_PRICE_DISPLAY_PRECISION"], max(12, market_info['price_precision'] + 7)
    levels, lines = ['R3', 'R2', 'R1', 'PP', 'S1', 'S2', 'S3'], {}
    for level in levels:
        value = pivots_info.get(level)
        if value is not None:
            val_str, color = format_decimal(value, price_prec, min_disp_prec), Fore.RED if 'R' in level else Fore.GREEN if 'S' in level else Fore.YELLOW
            hl = ""
            if last_price and value > 0:
                try:
                    if abs(last_price - value) / value < decimal.Decimal('0.001'): hl = Back.LIGHTBLACK_EX + Fore.WHITE + Style.BRIGHT + " *NEAR* " + Style.RESET_ALL
                except: pass
            lines[level] = f"{color}{level}: {val_str.rjust(width)}{Style.RESET_ALL}{hl}"
        else: lines[level] = f"{level}: {'N/A'.rjust(width)}"
    print(f"  {lines.get('R3','')}    {lines.get('S3','')}")
    print(f"  {lines.get('R2','')}    {lines.get('S2','')}")
    print(f"  {lines.get('R1','')}    {lines.get('S1','')}")
    print(f"          {lines.get('PP','')}")
def display_orderbook(analyzed_ob: Optional[Dict], market_info: Dict, config: Dict) -> Tuple[Dict[int, decimal.Decimal], Dict[int, decimal.Decimal]]:
    print_color("--- Order Book Depths ---", Fore.BLUE); ask_map, bid_map = {}, {}
    if not analyzed_ob: print_color(f"  {Fore.YELLOW}Order book data unavailable.{Style.RESET_ALL}"); return ask_map, bid_map
    p_prec, a_prec, min_p_prec, v_disp_prec, depth = market_info['price_precision'], market_info['amount_precision'], config["MIN_PRICE_DISPLAY_PRECISION"], config["VOLUME_DISPLAY_PRECISION"], config["MAX_ORDERBOOK_DEPTH_DISPLAY"]
    idx_w, p_w, v_w, cum_v_w = len(f"[A{depth}]")+1, max(10,p_prec+4), max(10,v_disp_prec+5), max(12,v_disp_prec+7)
    ask_lines, bid_lines = [], []; display_asks = list(reversed(analyzed_ob['asks']))
    for idx, ask in enumerate(display_asks):
        if idx >= depth: break
        idx_str, p_str = f"[A{idx+1}]".ljust(idx_w), format_decimal(ask['price'], p_prec, min_p_prec)
        cum_v_str = f"{Fore.LIGHTBLACK_EX}(Cum:{ask['cumulative_volume']}){Style.RESET_ALL}"
        ask_lines.append(f"{Fore.CYAN}{idx_str}{Style.NORMAL}{Fore.WHITE}{p_str:<{p_w}}{ask['style']}{ask['color']}{ask['volume_str']:<{v_w}}{Style.RESET_ALL} {cum_v_str:<{cum_v_w}}")
        ask_map[idx + 1] = ask['price']
    for idx, bid in enumerate(analyzed_ob['bids']):
        if idx >= depth: break
        idx_str, p_str = f"[B{idx+1}]".ljust(idx_w), format_decimal(bid['price'], p_prec, min_p_prec)
        cum_v_str = f"{Fore.LIGHTBLACK_EX}(Cum:{bid['cumulative_volume']}){Style.RESET_ALL}"
        bid_lines.append(f"{Fore.CYAN}{idx_str}{Style.NORMAL}{Fore.WHITE}{p_str:<{p_w}}{bid['style']}{bid['color']}{bid['volume_str']:<{v_w}}{Style.RESET_ALL} {cum_v_str:<{cum_v_w}}")
        bid_map[idx + 1] = bid['price']
    col_w = idx_w + p_w + v_w + cum_v_w + 3
    print_color(f"{'Asks'.center(col_w)}{'Bids'.center(col_w)}", Fore.LIGHTBLACK_EX); print_color(f"{'-'*col_w} {'-'*col_w}", Fore.LIGHTBLACK_EX)
    max_r = max(len(ask_lines), len(bid_lines))
    for i in range(max_r): print(f"{ask_lines[i] if i < len(ask_lines) else ' '*col_w} {bid_lines[i] if i < len(bid_lines) else ''}")
    best_a, best_b = (display_asks[0]['price'] if display_asks else decimal.Decimal('NaN')), (analyzed_ob['bids'][0]['price'] if analyzed_ob['bids'] else decimal.Decimal('NaN'))
    spread = best_a - best_b if best_a.is_finite() and best_b.is_finite() else decimal.Decimal('NaN')
    spread_str = format_decimal(spread, p_prec, min_p_prec) if spread.is_finite() else "N/A"
    print_color(f"\n--- Spread: {spread_str} ---", Fore.MAGENTA, Style.DIM); return ask_map, bid_map
def display_volume_analysis(analyzed_ob: Optional[Dict], market_info: Dict, config: Dict) -> None:
    if not analyzed_ob: return
    a_prec, p_prec, v_disp_prec, min_p_prec = market_info['amount_precision'], market_info['price_precision'], config["VOLUME_DISPLAY_PRECISION"], config["MIN_PRICE_DISPLAY_PRECISION"]
    print_color("\n--- Volume Analysis (Fetched Depth) ---", Fore.BLUE)
    tot_a, tot_b = analyzed_ob['ask_total_volume_fetched'], analyzed_ob['bid_total_volume_fetched']; cum_a, cum_b = analyzed_ob['cumulative_ask_volume_displayed'], analyzed_ob['cumulative_bid_volume_displayed']
    print_color(f"  Total Ask: {Fore.RED}{format_decimal(tot_a, a_prec, v_disp_prec)}{Style.RESET_ALL} | Total Bid: {Fore.GREEN}{format_decimal(tot_b, a_prec, v_disp_prec)}{Style.RESET_ALL}")
    print_color(f"  Cum Ask(Disp): {Fore.RED}{format_decimal(cum_a, a_prec, v_disp_prec)}{Style.RESET_ALL} | Cum Bid(Disp): {Fore.GREEN}{format_decimal(cum_b, a_prec, v_disp_prec)}{Style.RESET_ALL}")
    imb = analyzed_ob['volume_imbalance_ratio_fetched']; imb_c, imb_s = Fore.WHITE, "N/A"
    if imb.is_infinite(): imb_c, imb_s = Fore.LIGHTGREEN_EX, "Inf"
    elif imb.is_finite():
        imb_s = format_decimal(imb, 2); imb_c = Fore.GREEN if imb > decimal.Decimal('1.5') else Fore.RED if imb < decimal.Decimal('0.67') and not imb.is_zero() else Fore.LIGHTRED_EX if imb.is_zero() and tot_a > 0 else Fore.WHITE
    ask_vwap, bid_vwap = format_decimal(analyzed_ob['ask_vwap_fetched'], p_prec, min_p_prec), format_decimal(analyzed_ob['bid_vwap_fetched'], p_prec, min_p_prec)
    print_color(f"  Imbalance(B/A): {imb_c}{imb_s}{Style.RESET_ALL} | VWAP Ask: {Fore.YELLOW}{ask_vwap}{Style.RESET_ALL} | VWAP Bid: {Fore.YELLOW}{bid_vwap}{Style.RESET_ALL}")
    print_color("--- Pressure Reading ---", Fore.BLUE)
    if imb.is_infinite(): print_color("  Extreme Bid Dominance", Fore.LIGHTYELLOW_EX)
    elif imb.is_zero() and tot_a > 0: print_color("  Extreme Ask Dominance", Fore.LIGHTYELLOW_EX)
    elif imb > decimal.Decimal('1.5'): print_color("  Strong Buy Pressure", Fore.GREEN, Style.BRIGHT)
    elif imb < decimal.Decimal('0.67') and not imb.is_zero(): print_color("  Strong Sell Pressure", Fore.RED, Style.BRIGHT)
    else: print_color("  Volume Relatively Balanced", Fore.WHITE)
    print_color("=" * 85, Fore.CYAN)

# --- Trading Functions (Async Versions - place_market_order_async, place_limit_order_async, place_limit_order_interactive_async) ---
# ... Include these functions from v3.0 draft ...
async def place_market_order_async(exchange_pro: ccxtpro.Exchange, symbol: str, side: str, amount_str: str, market_info: Dict, config: Dict) -> None:
    print_color(f"{Fore.CYAN}# Preparing ASYNC {side.upper()} market order...{Style.RESET_ALL}")
    try:
        amount = decimal.Decimal(amount_str); min_amount, amount_prec, amount_step = market_info.get('min_amount', decimal.Decimal('0')), market_info['amount_precision'], market_info.get('amount_step', decimal.Decimal('0'))
        if amount <= 0: print_color("Amount must be positive.", Fore.YELLOW); return
        if min_amount > 0 and amount < min_amount: print_color(f"Amount < min ({format_decimal(min_amount, amount_prec)}).", Fore.YELLOW); return
        if amount_step > 0 and (amount % amount_step) != 0:
            original_amount=amount; amount = (amount // amount_step) * amount_step; print_color(f"Amount rounded: {original_amount} -> {amount}", Fore.YELLOW)
            if amount <= 0 or (min_amount > 0 and amount < min_amount): print_color("Rounded amount invalid.", Fore.RED); return
        amount_str = format_decimal(amount, amount_prec)
    except Exception as e: print_color(f"Invalid amount: {e}", Fore.YELLOW); return
    side_color = Fore.GREEN if side == 'buy' else Fore.RED
    prompt = f"{Style.BRIGHT}Confirm MARKET {side_color}{side.upper()}{Style.RESET_ALL} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} ({Fore.GREEN}y{Style.RESET_ALL}/{Fore.RED}n{Style.RESET_ALL}): {Style.RESET_ALL}"
    try: confirm = await asyncio.to_thread(input, prompt); confirm = confirm.strip().lower()
    except (EOFError, KeyboardInterrupt): print_color("\nOrder cancelled.", Fore.YELLOW); return
    if confirm == 'y' or confirm == 'yes':
        print_color(f"{Fore.CYAN}# Transmitting ASYNC market order...", style=Style.DIM, end='\r')
        try:
            params = {'positionIdx': config["POSITION_IDX"]} if config.get("POSITION_IDX", 0) != 0 else {}
            order = await exchange_pro.create_market_order(symbol, side, float(amount), params=params)
            sys.stdout.write("\033[K"); oid, avg_p, filled = order.get('id', 'N/A'), order.get('average'), order.get('filled')
            msg = f"✅ Market {side.upper()} [{oid}] Placed!"; details = []
            if filled: details.append(f"Filled: {format_decimal(filled, amount_prec)}")
            if avg_p: details.append(f"Avg Price: {format_decimal(avg_p, market_info['price_precision'])}")
            if details: msg += f" ({', '.join(details)})"
            print_color(f"\n{msg}", Fore.GREEN, Style.BRIGHT); termux_toast(f"{symbol} Market {side.upper()} Placed: {oid}")
        except ccxt.InsufficientFunds as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Funds Err: {e}", Fore.RED); termux_toast(f"{symbol} Mkt Fail: Funds", "long")
        except ccxt.InvalidOrder as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Invalid Order: {e}", Fore.RED); termux_toast(f"{symbol} Mkt Fail: Invalid", "long")
        except ccxt.ExchangeError as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Exch Err: {e}", Fore.RED); termux_toast(f"{symbol} Mkt Fail: ExchErr", "long")
        except Exception as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Place Err: {e}", Fore.RED); termux_toast(f"{symbol} Mkt Fail: Error", "long")
    else: print_color("Market order cancelled.", Fore.YELLOW)
async def place_limit_order_async(exchange_pro: ccxtpro.Exchange, symbol: str, side: str, amount_str: str, price_str: str, market_info: Dict, config: Dict) -> None:
    print_color(f"{Fore.CYAN}# Preparing ASYNC {side.upper()} limit order...{Style.RESET_ALL}")
    try:
        amount, price = decimal.Decimal(amount_str), decimal.Decimal(price_str); min_amount, amount_prec, price_prec = market_info['min_amount'], market_info['amount_precision'], market_info['price_precision']
        amount_step, price_tick = market_info.get('amount_step', decimal.Decimal('0')), market_info.get('price_tick_size', decimal.Decimal('0'))
        if amount <= 0 or price <= 0: print_color("Amount/Price must be positive.", Fore.YELLOW); return
        if min_amount > 0 and amount < min_amount: print_color(f"Amount < min ({format_decimal(min_amount, amount_prec)}).", Fore.YELLOW); return
        if amount_step > 0 and (amount % amount_step) != 0:
            original_amount=amount; amount = (amount // amount_step) * amount_step; print_color(f"Amount rounded: {original_amount} -> {amount}", Fore.YELLOW)
            if amount <= 0 or (min_amount > 0 and amount < min_amount): print_color("Rounded amount invalid.", Fore.RED); return
        if price_tick > 0 and (price % price_tick) != 0:
            original_price=price; price = price.quantize(price_tick, rounding=decimal.ROUND_HALF_UP); print_color(f"Price rounded: {original_price} -> {price}", Fore.YELLOW)
            if price <= 0: print_color("Rounded price invalid.", Fore.RED); return
        amount_str, price_str = format_decimal(amount, amount_prec), format_decimal(price, price_prec)
    except Exception as e: print_color(f"Invalid amount or price: {e}", Fore.YELLOW); return
    side_color = Fore.GREEN if side == 'buy' else Fore.RED
    prompt = f"{Style.BRIGHT}Confirm LIMIT {side_color}{side.upper()}{Style.RESET_ALL} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{price_str}{Style.RESET_ALL} ({Fore.GREEN}y{Style.RESET_ALL}/{Fore.RED}n{Style.RESET_ALL}): {Style.RESET_ALL}"
    try: confirm = await asyncio.to_thread(input, prompt); confirm = confirm.strip().lower()
    except (EOFError, KeyboardInterrupt): print_color("\nLimit order cancelled.", Fore.YELLOW); return
    if confirm == 'y' or confirm == 'yes':
        print_color(f"{Fore.CYAN}# Transmitting ASYNC limit order...", style=Style.DIM, end='\r')
        try:
            params = {'positionIdx': config["POSITION_IDX"]} if config.get("POSITION_IDX", 0) != 0 else {}
            order = await exchange_pro.create_limit_order(symbol, side, float(amount), float(price), params=params)
            sys.stdout.write("\033[K"); oid, lim_p, ord_a = order.get('id', 'N/A'), order.get('price'), order.get('amount')
            msg = f"✅ Limit {side.upper()} [{oid}] Placed!"; details = []
            if ord_a: details.append(f"Amount: {format_decimal(ord_a, amount_prec)}")
            if lim_p: details.append(f"Price: {format_decimal(lim_p, market_info['price_precision'])}")
            if details: msg += f" ({', '.join(details)})"
            print_color(f"\n{msg}", Fore.GREEN, Style.BRIGHT); termux_toast(f"{symbol} Limit {side.upper()} Placed: {oid}")
        except ccxt.InsufficientFunds as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Funds Err: {e}", Fore.RED); termux_toast(f"{symbol} Lim Fail: Funds", "long")
        except ccxt.InvalidOrder as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Invalid Ord: {e}", Fore.RED); termux_toast(f"{symbol} Lim Fail: Invalid", "long")
        except ccxt.ExchangeError as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Exch Err: {e}", Fore.RED); termux_toast(f"{symbol} Lim Fail: ExchErr", "long")
        except Exception as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Async Place Err: {e}", Fore.RED); termux_toast(f"{symbol} Lim Fail: Error", "long")
    else: print_color("Limit order cancelled.", Fore.YELLOW)
async def place_limit_order_interactive_async(exchange_pro: ccxtpro.Exchange, symbol: str, side: str, ask_map: Dict[int, decimal.Decimal], bid_map: Dict[int, decimal.Decimal], market_info: Dict, config: Dict) -> None:
    print_color(f"\n{Fore.BLUE}--- Interactive Limit Order ({side.upper()}) ---{Style.RESET_ALL}")
    target_map, prompt_char = (bid_map, 'B') if side == 'sell' else (ask_map, 'A')
    if not target_map: print_color(f"OB side empty.", Fore.YELLOW); return
    selected_price: Optional[decimal.Decimal] = None
    while selected_price is None:
        try:
            idx_prompt = f"{Style.BRIGHT}Select OB Index ({prompt_char}1-{prompt_char}{len(target_map)}, '{Fore.YELLOW}c{Style.RESET_ALL}' cancel): {Style.RESET_ALL}"
            index_str = await asyncio.to_thread(input, idx_prompt); index_str = index_str.strip().upper()
            if index_str == 'C': print_color("Cancelled.", Fore.YELLOW); return
            if not index_str.startswith(prompt_char) or not index_str[1:].isdigit(): print_color("Invalid format.", Fore.YELLOW); continue
            index = int(index_str[1:]); selected_price = target_map.get(index)
            if selected_price is None: print_color(f"Index {index_str} not found.", Fore.YELLOW)
        except (ValueError, IndexError): print_color("Invalid index.", Fore.YELLOW)
        except (EOFError, KeyboardInterrupt): print_color("\nCancelled.", Fore.YELLOW); return
    price_fmt = format_decimal(selected_price, market_info['price_precision'])
    print_color(f"Selected Price: {Fore.YELLOW}{price_fmt}{Style.RESET_ALL}")
    while True:
        try:
            qty_prompt = f"{Style.BRIGHT}Qty for {side.upper()} @ {price_fmt} ('{Fore.YELLOW}c{Style.RESET_ALL}' cancel): {Style.RESET_ALL}"
            qty_str = await asyncio.to_thread(input, qty_prompt); qty_str = qty_str.strip()
            if qty_str.lower() == 'c': print_color("Cancelled.", Fore.YELLOW); return
            if decimal.Decimal(qty_str) <= 0: print_color("Qty must be positive.", Fore.YELLOW); continue
            break
        except (decimal.InvalidOperation, ValueError): print_color("Invalid quantity.", Fore.YELLOW)
        except (EOFError, KeyboardInterrupt): print_color("\nCancelled.", Fore.YELLOW); return
    await place_limit_order_async(exchange_pro, symbol, side, qty_str, str(selected_price), market_info, config)

# ==============================================================================
# Initial Calculation Trigger
# ==============================================================================
async def fetch_and_recalculate(exchange_pro: ccxtpro.Exchange, symbol: str, config: Dict):
     """Helper to fetch initial/periodic OHLCV and trigger recalculations."""
     verbose_print("Running fetch_and_recalculate...")
     min_ohlcv = config["MIN_OHLCV_CANDLES"]
     ind_tf, piv_tf = config['INDICATOR_TIMEFRAME'], config['PIVOT_TIMEFRAME']
     history_needed = min_ohlcv + 5

     try:
         ind_ohlcv_res, piv_ohlcv_res = await asyncio.gather(
             exchange_pro.fetch_ohlcv(symbol, ind_tf, limit=history_needed),
             exchange_pro.fetch_ohlcv(symbol, piv_tf, limit=2),
             return_exceptions=True
         )
         # Process Indicator OHLCV
         if isinstance(ind_ohlcv_res, list) and len(ind_ohlcv_res) >= min_ohlcv:
             latest_data["indicator_ohlcv"] = ind_ohlcv_res; latest_data["last_update_times"]["indicator_ohlcv"] = time.time()
             await calculate_and_store_indicators()
         elif isinstance(ind_ohlcv_res, Exception): print_color(f"# Recalc Ind OHLCV failed: {ind_ohlcv_res}", Fore.YELLOW)
         else: print_color(f"# Recalc Ind OHLCV insufficient ({len(ind_ohlcv_res if ind_ohlcv_res else [])}<{min_ohlcv})", Fore.YELLOW); latest_data["indicator_ohlcv"] = None; await calculate_and_store_indicators() # Try calc even if short? Or clear indicators? Clear seems safer.
         # Process Pivot OHLCV
         if isinstance(piv_ohlcv_res, list) and len(piv_ohlcv_res) > 0:
             latest_data["pivot_ohlcv"] = piv_ohlcv_res; latest_data["last_update_times"]["pivot_ohlcv"] = time.time()
             await calculate_and_store_pivots()
         elif isinstance(piv_ohlcv_res, Exception): print_color(f"# Recalc Piv OHLCV failed: {piv_ohlcv_res}", Fore.YELLOW); latest_data["pivots"] = None
         else: latest_data["pivots"] = None # Clear if empty/None
     except Exception as e: print_color(f"# Error during fetch/recalc: {e}", Fore.RED)

async def initial_calculations(exchange_pro: ccxtpro.Exchange, symbol: str, config: Dict):
    """Fetches initial data required for calculations."""
    await asyncio.sleep(2) # Initial delay for connections
    print_color("# Performing initial data fetch...", Fore.CYAN, Style.DIM)
    await fetch_and_recalculate(exchange_pro, symbol, config)
    # Fetch initial balance/pos as well
    try:
        bal_res, pos_res = await asyncio.gather(
             exchange_pro.fetch_balance(),
             exchange_pro.fetch_positions([symbol]),
             return_exceptions=True
         )
        if isinstance(bal_res, dict): latest_data["balance"] = bal_res.get('total', {}).get(config["FETCH_BALANCE_ASSET"]); latest_data["last_update_times"]["balance"] = time.time()
        elif isinstance(bal_res, Exception): print_color(f"# Initial Balance fetch failed: {bal_res}", Fore.YELLOW)
        if isinstance(pos_res, list): latest_data["positions"] = [p for p in pos_res if p.get('symbol') == symbol and p.get('contracts') and decimal.Decimal(str(p['contracts'])) != 0]; latest_data["last_update_times"]["positions"] = time.time()
        elif isinstance(pos_res, Exception): print_color(f"# Initial Positions fetch failed: {pos_res}", Fore.YELLOW)
    except Exception as e: print_color(f"# Error during initial bal/pos fetch: {e}", Fore.RED)
    print_color("# Initial data fetch complete.", Fore.CYAN, Style.DIM)

# ==============================================================================
# Main Execution Function (Async)
# ==============================================================================
async def main_async():
    """Async main function: Initializes, starts tasks, runs display/interaction loop."""
    global exchange, latest_data
    print_color("*"*85, Fore.RED, Style.BRIGHT); print_color("   🔥 Pyrmethus Market Analyzer v3.1 ASYNC Stable 🔥", Fore.RED, Style.BRIGHT); print_color("   WebSockets Activated! Use responsibly.", Fore.YELLOW); print_color("*"*85, Fore.RED, Style.BRIGHT); print()

    if not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]: print_color("API Key/Secret missing.", Fore.RED, Style.BRIGHT); return

    print_color(f"{Fore.CYAN}# Binding async spirit to Bybit ({CONFIG['EXCHANGE_TYPE']})...{Style.DIM}")
    exchange = ccxtpro.bybit({'apiKey': CONFIG["API_KEY"], 'secret': CONFIG["API_SECRET"], 'options': {'defaultType': CONFIG["EXCHANGE_TYPE"], 'adjustForTimeDifference': True}, 'enableRateLimit': True, 'newUpdates': True})
    try:
        print_color(f"{Fore.CYAN}# Testing connection...{Style.DIM}", end='\r'); await exchange.fetch_time(); sys.stdout.write("\033[K"); print_color("Async connection ok.", Fore.GREEN)
    except Exception as e: sys.stdout.write("\033[K"); print_color(f"Connection failed: {e}", Fore.RED); await exchange.close(); return

    symbol = CONFIG['SYMBOL']
    print_color(f"{Fore.CYAN}# Verifying symbol: {symbol}{Style.RESET_ALL}")
    market_info = await get_market_info(exchange, symbol) # Use async version
    while not market_info:
        print_color("Failed market info.", Fore.YELLOW)
        try:
            symbol_input = await asyncio.to_thread(input, f"{Style.BRIGHT}Enter Bybit Symbol: {Style.RESET_ALL}")
            symbol_input = symbol_input.strip().upper()
            if not symbol_input: continue
            potential_info = await get_market_info(exchange, symbol_input) # Use async version
            if potential_info: symbol, market_info = symbol_input, potential_info; CONFIG['SYMBOL'] = symbol; print_color(f"Switched to: {Fore.MAGENTA}{symbol}{Style.RESET_ALL}", Fore.CYAN); break
        except (EOFError, KeyboardInterrupt): print_color("\nInterrupted.", Fore.YELLOW); await exchange.close(); return
        except Exception as e: print_color(f"Symbol input error: {e}", Fore.RED); await asyncio.sleep(1)

    latest_data["market_info"] = market_info # Store globally
    print_color(f"Focus: {Fore.MAGENTA}{symbol}{Fore.CYAN} | P:{market_info['price_precision']} A:{market_info['amount_precision']}", Fore.CYAN)
    print_color(f"Trade Mode: {Fore.YELLOW}{CONFIG['DEFAULT_ORDER_TYPE'].capitalize()}{Style.RESET_ALL}" + (f" ({CONFIG['LIMIT_ORDER_SELECTION_TYPE']})" if CONFIG['DEFAULT_ORDER_TYPE']=='limit' else "") + (f" | Hedge Idx:{CONFIG['POSITION_IDX']}" if CONFIG['POSITION_IDX']!=0 else ""))

    # --- Start Background Tasks ---
    tasks = [
        asyncio.create_task(watch_ticker(exchange, symbol)),
        asyncio.create_task(watch_orderbook(exchange, symbol)),
        asyncio.create_task(fetch_periodic_data(exchange, symbol)),
        asyncio.create_task(initial_calculations(exchange, symbol, CONFIG)) # Fetch initial data
    ]
    print_color(f"\n{Fore.CYAN}Starting async analysis. Ctrl+C to exit.{Style.RESET_ALL}")
    ask_map: Dict[int, decimal.Decimal] = {}; bid_map: Dict[int, decimal.Decimal] = {}
    last_error_msg = ""; data_error_streak = 0

    try:
        # --- Main Display & Interaction Loop ---
        while True:
            # Display latest available data
            try:
                ask_map, bid_map = display_combined_analysis_async(latest_data, market_info, CONFIG)
                display_ok = True
            except Exception as display_e:
                print_color(f"\n--- Display Error: {display_e} ---", color=Fore.RED, style=Style.BRIGHT)
                if CONFIG.get("VERBOSE_DEBUG", False): traceback.print_exc()
                display_ok = False # Mark display failure

            # Check connection status for error reporting / delay logic
            fetch_error_occurred = any(v == "error" for v in latest_data["connection_status"].values())

            # --- Handle User Input ---
            action_prompt = f"\n{Style.BRIGHT}{Fore.BLUE}Action (r/b/s/x): {Style.RESET_ALL}"
            try:
                 action = await asyncio.wait_for(asyncio.to_thread(input, action_prompt), timeout=CONFIG["REFRESH_INTERVAL"]) # Wait for input or timeout
                 action = action.strip().lower()
                 side = {'b': 'buy', 'buy': 'buy', 's': 'sell', 'sell': 'sell', 'r': 'refresh', '': 'refresh', 'x': 'exit', 'exit': 'exit'}.get(action, 'unknown')

                 if side in ['buy', 'sell']:
                     order_type = CONFIG['DEFAULT_ORDER_TYPE']
                     if order_type == 'limit':
                         if CONFIG['LIMIT_ORDER_SELECTION_TYPE'] == 'interactive': await place_limit_order_interactive_async(exchange, symbol, side, ask_map, bid_map, market_info, CONFIG)
                         else: # Manual limit
                             price_str = await asyncio.to_thread(input, f"Limit Price ({side.upper()}): "); qty_str = await asyncio.to_thread(input, "Quantity: ")
                             if price_str and qty_str: await place_limit_order_async(exchange, symbol, side, qty_str, price_str, market_info, CONFIG)
                             else: print_color("Price/Qty missing.", Fore.YELLOW)
                     else: # Market order
                         qty_str = await asyncio.to_thread(input, f"{Fore.GREEN if side == 'buy' else Fore.RED}{side.upper()} Qty: {Style.RESET_ALL}")
                         if qty_str: await place_market_order_async(exchange, symbol, side, qty_str, market_info, CONFIG)
                         else: print_color("Quantity missing.", Fore.YELLOW)
                     await asyncio.sleep(1) # Short pause after order attempt

                 elif side == 'refresh': verbose_print("Manual refresh (display updates continuously)")
                 elif side == 'exit': print_color("Dispelling...", Fore.YELLOW); break
                 elif side != 'unknown': print_color("Unknown command.", Fore.YELLOW)

                 # Reset error streak if user interacts successfully
                 data_error_streak = 0

            except asyncio.TimeoutError: # No input, just continue loop for refresh
                 pass
            except (EOFError, KeyboardInterrupt): print_color("\nInput interrupted.", Fore.YELLOW); break

            # --- Delay Logic ---
            if fetch_error_occurred:
                data_error_streak += 1
                wait = min(CONFIG["RETRY_DELAY_NETWORK_ERROR"] * (2 ** min(data_error_streak, 4)), 120)
                print_color(f"Fetch error(s). Waiting {wait}s (Streak: {data_error_streak})...", Fore.YELLOW, Style.DIM)
                await asyncio.sleep(wait)
            # elif display_ok: # Normal refresh handled by input timeout or loop continuation
            #    await asyncio.sleep(0.1) # Small yield if no input/error
            elif not display_ok: # Display error, shorter pause
                 await asyncio.sleep(max(CONFIG["REFRESH_INTERVAL"] // 2, 3))
            else: # No fetch error, no input timeout -> minimal yield
                await asyncio.sleep(0.05)


    except KeyboardInterrupt: print_color("\nBanished by user.", Fore.YELLOW)
    except ccxt.AuthenticationError: print_color("\nAuth Error! Halting.", Fore.RED)
    except Exception as e:
        print_color(f"\n--- Critical Loop Error ---", Fore.RED, Style.BRIGHT); traceback.print_exc(); print_color(f"Error: {e}", Fore.RED); print_color(f"--- End Critical Error ---", Fore.RED, Style.BRIGHT); termux_toast("Pyrmethus CRITICAL Error", "long")
    finally:
        print_color("\nClosing connections & tasks...", Fore.YELLOW)
        for task in tasks: task.cancel()
        try: await asyncio.gather(*tasks, return_exceptions=True) # Wait for tasks to finish cancelling
        except asyncio.CancelledError: verbose_print("Tasks cancelled.")
        if exchange and hasattr(exchange, 'close'):
            try: await exchange.close(); print_color("Exchange connection closed.", Fore.YELLOW)
            except Exception as close_e: print_color(f"Error closing exchange: {close_e}", Fore.RED)
        print_color("Wizard Pyrmethus departs.", Fore.MAGENTA, Style.BRIGHT)

# ==============================================================================
# Script Entry Point
# ==============================================================================
if __name__ == '__main__':
    try: asyncio.run(main_async())
    except KeyboardInterrupt: print_color("\nCtrl+C detected. Shutting down...", Fore.YELLOW)
    except Exception as e: print(f"\nUnhandled top-level exception: {e}"); traceback.print_exc()

