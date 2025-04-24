============ FILE: pyrmethus_market_analyzer_v1.final_interactive.py ============
# ==============================================================================
# 🔥 Pyrmethus's Arcane Market Analyzer v1.FINAL Interactive Edition 🔥
# The Ultimate Termux Script for Market Analysis & Interactive Limit Order Sorcery!
# Woven with CCXT, Colorama, and enhanced with Order Book Bin Selection!
# Use with wisdom and manage risk. Market forces are potent.
# ==============================================================================
import decimal
import os
import subprocess
import sys
import time

import ccxt
from colorama import Back, Fore, Style, init
from dotenv import load_dotenv

# Initialize Colorama for colorful terminal output
init(autoreset=True)
decimal.getcontext().prec = 30  # Set decimal precision for calculations

# Load environment variables from .env file
load_dotenv()
print(f"{Fore.CYAN}{Style.DIM}# Loading ancient scrolls (.env)...{Style.RESET_ALL}")

# ==============================================================================
# Configuration Loading and Defaults
# ==============================================================================
CONFIG = {
    # --- API Keys - Guard these Secrets! ---
    "API_KEY": os.environ.get("BYBIT_API_KEY"),
    "API_SECRET": os.environ.get("BYBIT_API_SECRET"),

    # --- Market and Order Book Configuration ---
    "SYMBOL": os.environ.get("BYBIT_SYMBOL", "BTCUSDT").upper(),
    "EXCHANGE_TYPE": os.environ.get("BYBIT_EXCHANGE_TYPE", 'linear'),
    "VOLUME_THRESHOLDS": {
        'high': decimal.Decimal(os.environ.get("VOLUME_THRESHOLD_HIGH", '10')),
        'medium': decimal.Decimal(os.environ.get("VOLUME_THRESHOLD_MEDIUM", '2'))
    },
    "REFRESH_INTERVAL": int(os.environ.get("REFRESH_INTERVAL", '9')),
    "MAX_ORDERBOOK_DEPTH_DISPLAY": int(os.environ.get("MAX_ORDERBOOK_DEPTH_DISPLAY", '50')),
    "ORDER_FETCH_LIMIT": int(os.environ.get("ORDER_FETCH_LIMIT", '200')),
    "DEFAULT_EXCHANGE_TYPE": 'linear', # Fixed, not user configurable for simplicity
    "CONNECT_TIMEOUT": int(os.environ.get("CONNECT_TIMEOUT", '30000')),
    "RETRY_DELAY_NETWORK_ERROR": int(os.environ.get("RETRY_DELAY_NETWORK_ERROR", '10')),
    "RETRY_DELAY_RATE_LIMIT": int(os.environ.get("RETRY_DELAY_RATE_LIMIT", '60')),

    # --- Technical Indicator Settings ---
    "INDICATOR_TIMEFRAME": os.environ.get("INDICATOR_TIMEFRAME", '15m'),
    "SMA_PERIOD": int(os.environ.get("SMA_PERIOD", '9')),
    "SMA2_PERIOD": int(os.environ.get("SMA2_PERIOD", '20')),
    "EMA1_PERIOD": int(os.environ.get("EMA1_PERIOD", '12')),
    "EMA2_PERIOD": int(os.environ.get("EMA2_PERIOD", '34')),
    "MOMENTUM_PERIOD": int(os.environ.get("MOMENTUM_PERIOD", '10')),
    "RSI_PERIOD": int(os.environ.get("RSI_PERIOD", '14')),
    "STOCH_K_PERIOD": int(os.environ.get("STOCH_K_PERIOD", '14')),
    "STOCH_D_PERIOD": int(os.environ.get("STOCH_D_PERIOD", '3')),
    "STOCH_RSI_OVERSOLD": decimal.Decimal(os.environ.get("STOCH_RSI_OVERSOLD", '20')),
    "STOCH_RSI_OVERBOUGHT": decimal.Decimal(os.environ.get("STOCH_RSI_OVERBOUGHT", '80')),

    # --- Display Preferences ---
    "PIVOT_TIMEFRAME": os.environ.get("PIVOT_TIMEFRAME", '30m'),
    "PNL_PRECISION": int(os.environ.get("PNL_PRECISION", '2')),
    "MIN_PRICE_DISPLAY_PRECISION": int(os.environ.get("MIN_PRICE_DISPLAY_PRECISION", '3')),
    "STOCH_RSI_DISPLAY_PRECISION": int(os.environ.get("STOCH_RSI_DISPLAY_PRECISION", '3')),
    "VOLUME_DISPLAY_PRECISION": int(os.environ.get("VOLUME_DISPLAY_PRECISION", '0')),
    "BALANCE_DISPLAY_PRECISION": int(os.environ.get("BALANCE_DISPLAY_PRECISION", '2')),

    # --- Trading Defaults ---
    "FETCH_BALANCE_ASSET": os.environ.get("FETCH_BALANCE_ASSET", "USDT"),
    "DEFAULT_ORDER_TYPE": os.environ.get("DEFAULT_ORDER_TYPE", "market").lower(), # 'market' or 'limit'
    "LIMIT_ORDER_SELECTION_TYPE": os.environ.get("LIMIT_ORDER_SELECTION_TYPE", "interactive").lower(), # 'interactive' or 'manual'
}

# Fibonacci Ratios for Pivot Point Calculations
FIB_RATIOS = {
    'r3': decimal.Decimal('1.000'), 'r2': decimal.Decimal('0.618'), 'r1': decimal.Decimal('0.382'),
    's1': decimal.Decimal('0.382'), 's2': decimal.Decimal('0.618'), 's3': decimal.Decimal('1.000'),
}

# ==============================================================================
# Utility Functions
# ==============================================================================

def print_color(text, color=Fore.WHITE, style=Style.NORMAL, end='\n', **kwargs):
    """Prints colorized text in the terminal."""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end, **kwargs)

def termux_toast(message, duration="short"):
    """Displays a toast notification on Termux (if termux-api is installed)."""
    try:
        safe_message = ''.join(c for c in str(message) if c.isalnum() or c in ' .,!?-:')[:100]
        subprocess.run(['termux-toast', '-d', duration, safe_message], check=True, capture_output=True, timeout=5)
    except FileNotFoundError:
        print_color("# termux-toast not found. Install termux-api?", color=Fore.YELLOW, style=Style.DIM)
    except Exception as e:
        print_color(f"# Toast error: {e}", color=Fore.YELLOW, style=Style.DIM)

def format_decimal(value, reported_precision, min_display_precision=None):
    """Formats decimal values for display with specified precision."""
    if value is None: return "N/A"
    if not isinstance(value, decimal.Decimal):
        try: value = decimal.Decimal(str(value))
        except: return str(value) # Fallback to string if decimal conversion fails
    try:
        display_precision = int(reported_precision)
        if min_display_precision is not None:
            display_precision = max(display_precision, int(min_display_precision))
        if display_precision < 0: display_precision = 0

        quantizer = decimal.Decimal('1') / (decimal.Decimal('10') ** display_precision)
        # Use ROUND_HALF_UP for general rounding, adjust if specific rounding rules needed
        rounded_value = value.quantize(quantizer, rounding=decimal.ROUND_HALF_UP)
        formatted_str = str(rounded_value.normalize()) # normalize removes trailing zeros

        # Ensure minimum decimal places are shown, especially if rounded to whole number or fewer places than min_display
        if '.' not in formatted_str and display_precision > 0:
            formatted_str += '.' + '0' * display_precision
        elif '.' in formatted_str:
            integer_part, decimal_part = formatted_str.split('.')
            if len(decimal_part) < display_precision:
                formatted_str += '0' * (display_precision - len(decimal_part))
        return formatted_str
    except Exception as e:
        print_color(f"# FormatDecimal Error ({value}, P:{reported_precision}): {e}", color=Fore.YELLOW, style=Style.DIM)
        return str(value)

def get_market_info(exchange, symbol):
    """Fetches and returns market information (precision, limits) from the exchange."""
    try:
        print_color(f"{Fore.CYAN}# Querying market runes for {symbol}...", style=Style.DIM, end='\r')
        if not exchange.markets or symbol not in exchange.markets:
            print_color(f"{Fore.CYAN}# Summoning market list...", style=Style.DIM, end='\r')
            exchange.load_markets(True)
        sys.stdout.write("\033[K")
        market = exchange.market(symbol)
        sys.stdout.write("\033[K")

        price_prec_raw = market.get('precision', {}).get('price')
        amount_prec_raw = market.get('precision', {}).get('amount')
        min_amount_raw = market.get('limits', {}).get('amount', {}).get('min')

        # Convert precision to number of decimal places
        try: price_prec = int(decimal.Decimal(str(price_prec_raw)).log10() * -1) if price_prec_raw is not None else 8
        except: price_prec = 8
        try: amount_prec = int(decimal.Decimal(str(amount_prec_raw)).log10() * -1) if amount_prec_raw is not None else 8
        except: amount_prec = 8
        try: min_amount = decimal.Decimal(str(min_amount_raw)) if min_amount_raw is not None else decimal.Decimal('0')
        except: min_amount = decimal.Decimal('0')

        # Calculate tick sizes from precision (assuming standard decimal structure)
        price_tick_size = decimal.Decimal('1') / (decimal.Decimal('10') ** price_prec) if price_prec >= 0 else decimal.Decimal('1')
        amount_step = decimal.Decimal('1') / (decimal.Decimal('10') ** amount_prec) if amount_prec >= 0 else decimal.Decimal('1')

        return {
            'price_precision': price_prec, 'amount_precision': amount_prec,
            'min_amount': min_amount, 'price_tick_size': price_tick_size, 'amount_step': amount_step, 'symbol': symbol
        }
    except ccxt.BadSymbol:
        sys.stdout.write("\033[K")
        print_color(f"Symbol '{symbol}' is not found on the exchange.", color=Fore.RED, style=Style.BRIGHT)
        return None
    except ccxt.NetworkError as e:
        sys.stdout.write("\033[K")
        print_color(f"Network error fetching market info: {e}", color=Fore.YELLOW)
        return None # Allow retry
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"Error fetching market info for {symbol}: {e}", color=Fore.RED)
        return None # Indicate potentially fatal error for this symbol

# ==============================================================================
# Indicator Calculation Functions
# ==============================================================================

def calculate_sma(data, period):
    """Calculates Simple Moving Average."""
    if not data or len(data) < period: return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data[-period:]]
        return sum(decimal_data) / decimal.Decimal(period)
    except Exception as e:
        print_color(f"# SMA Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

def calculate_ema(data, period):
    """Calculates Exponential Moving Average."""
    if not data or len(data) < period: return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data]
        ema_values = [None] * len(decimal_data)
        multiplier = decimal.Decimal(2) / (decimal.Decimal(period) + 1)
        sma_initial = sum(decimal_data[:period]) / decimal.Decimal(period)
        ema_values[period - 1] = sma_initial
        for i in range(period, len(decimal_data)):
            ema_values[i] = (decimal_data[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        return [ema for ema in ema_values if ema is not None]
    except Exception as e:
        print_color(f"# EMA Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

def calculate_momentum(data, period):
    """Calculates Momentum indicator."""
    if not data or len(data) <= period: return None
    try:
        current_price = decimal.Decimal(str(data[-1]))
        past_price = decimal.Decimal(str(data[-period - 1]))
        return current_price - past_price
    except Exception as e:
        print_color(f"# Momentum Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

def calculate_fib_pivots(high, low, close):
    """Calculates Fibonacci Pivot Points."""
    if None in [high, low, close]: return None
    try:
        h, l, c = decimal.Decimal(str(high)), decimal.Decimal(str(low)), decimal.Decimal(str(close))
        if h <= 0 or l <= 0 or c <= 0 or h < l: return None # Basic validation
        pp = (h + l + c) / 3
        range_hl = h - l
        return {
            'R3': pp + (range_hl * FIB_RATIOS['r3']), 'R2': pp + (range_hl * FIB_RATIOS['r2']),
            'R1': pp + (range_hl * FIB_RATIOS['r1']), 'PP': pp,
            'S1': pp - (range_hl * FIB_RATIOS['s1']), 'S2': pp - (range_hl * FIB_RATIOS['s2']),
            'S3': pp - (range_hl * FIB_RATIOS['s3'])
        }
    except Exception as e:
        print_color(f"# Pivot Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

def calculate_rsi_manual(close_prices_list, period=14):
    """Calculates RSI manually using Wilder's Smoothing Method."""
    if not close_prices_list or len(close_prices_list) <= period: return None, "Not enough data"
    try:
        prices = [decimal.Decimal(str(p)) for p in close_prices_list]
        deltas = [prices[i]-prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else decimal.Decimal('0') for d in deltas]
        losses = [-d if d < 0 else decimal.Decimal('0') for d in deltas]
        if len(gains) < period: return None, "Insufficient deltas"

        avg_gain = sum(gains[:period]) / decimal.Decimal(period)
        avg_loss = sum(losses[:period]) / decimal.Decimal(period)

        rsi_values = [decimal.Decimal('NaN')] * period
        if avg_loss == 0: rs = decimal.Decimal('inf')
        else: rs = avg_gain / avg_loss
        first_rsi = 100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal('100')
        rsi_values.append(first_rsi)

        for i in range(period, len(gains)):
            avg_gain = (avg_gain*(period-1) + gains[i]) / decimal.Decimal(period)
            avg_loss = (avg_loss*(period-1) + losses[i]) / decimal.Decimal(period)
            if avg_loss == 0: rs = decimal.Decimal('inf')
            else: rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal('100')
            rsi_values.append(rsi)
        return [r for r in rsi_values if not r.is_nan()], None
    except Exception as e:
        print_color(f"# RSI Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None, str(e)

def calculate_stoch_rsi_manual(rsi_values, k_period=14, d_period=3):
    """Calculates Stochastic RSI %K and %D manually from RSI values."""
    if not rsi_values or len(rsi_values) < k_period: return None, None, "RSI values too short"
    try:
        valid_rsi = [r for r in rsi_values if r is not None and r.is_finite()]
        if len(valid_rsi) < k_period: return None, None, "Valid RSI values too short"

        stoch_k_values = []
        for i in range(k_period - 1, len(valid_rsi)):
            rsi_window = valid_rsi[i - k_period + 1 : i + 1]
            current_rsi = rsi_window[-1]
            min_rsi = min(rsi_window)
            max_rsi = max(rsi_window)
            if max_rsi == min_rsi: stoch_k = decimal.Decimal('50') # Neutral if range is zero
            else: stoch_k = ((current_rsi - min_rsi) / (max_rsi - min_rsi)) * 100
            stoch_k_values.append(stoch_k)

        if not stoch_k_values: return None, None, "%K calculation failed"
        if len(stoch_k_values) < d_period: return stoch_k_values[-1], None, "%D cannot be calculated yet"

        stoch_d_values = []
        for i in range(d_period - 1, len(stoch_k_values)):
            k_window = stoch_k_values[i - d_period + 1 : i + 1]
            stoch_d = sum(k_window) / decimal.Decimal(d_period)
            stoch_d_values.append(stoch_d)

        latest_k = stoch_k_values[-1] if stoch_k_values else None
        latest_d = stoch_d_values[-1] if stoch_d_values else None
        return latest_k, latest_d, None
    except Exception as e:
        print_color(f"# StochRSI Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None, None, str(e)

# ==============================================================================
# Data Fetching & Processing Functions
# ==============================================================================

def fetch_market_data(exchange, symbol, config):
    """Fetches all required market data in a consolidated manner."""
    results = {"ticker": None, "indicator_ohlcv": None, "pivot_ohlcv": None, "positions": [], "balance": None}
    error_occurred = False
    rate_limit_wait = config["RETRY_DELAY_RATE_LIMIT"]
    network_wait = config["RETRY_DELAY_NETWORK_ERROR"]

    indicator_history_needed = max(
        config['SMA_PERIOD'], config['SMA2_PERIOD'], config['EMA1_PERIOD'], config['EMA2_PERIOD'],
        config['MOMENTUM_PERIOD'] + 1, config['RSI_PERIOD'] + config['STOCH_K_PERIOD'] + config['STOCH_D_PERIOD']
    ) + 5 # Add a buffer

    api_calls = [
        {"func": exchange.fetch_ticker, "args": [symbol], "desc": "ticker"},
        {"func": exchange.fetch_ohlcv, "args": [symbol, config['INDICATOR_TIMEFRAME'], None, indicator_history_needed], "desc": "Indicator OHLCV"},
        {"func": exchange.fetch_ohlcv, "args": [symbol, config['PIVOT_TIMEFRAME'], None, 2], "desc": "Pivot OHLCV"}, # Need previous candle
        {"func": exchange.fetch_positions, "args": [[symbol]], "desc": "positions"},
        {"func": exchange.fetch_balance, "args": [], "desc": "balance"},
    ]

    print_color(f"{Fore.CYAN}# Contacting exchange spirits...", style=Style.DIM, end='\r')
    for call in api_calls:
        try:
            data = call["func"](*call["args"])
            if call["desc"] == "positions":
                # Filter for positions with non-zero contracts for the specific symbol
                results[call["desc"]] = [p for p in data if p.get('symbol') == symbol and p.get('contracts') is not None and decimal.Decimal(str(p.get('contracts','0'))) != 0]
            elif call["desc"] == "balance":
                results[call["desc"]] = data.get('total', {}).get(config["FETCH_BALANCE_ASSET"])
            else:
                results[call["desc"]] = data
            time.sleep(exchange.rateLimit / 1000) # Respect basic rate limit

        except ccxt.RateLimitExceeded:
            print_color(f"Rate Limit ({call['desc']}). Pausing {rate_limit_wait}s.", color=Fore.YELLOW, style=Style.DIM)
            time.sleep(rate_limit_wait)
            error_occurred = True; break # Stop fetching for this cycle
        except ccxt.NetworkError:
            print_color(f"Network Error ({call['desc']}). Pausing {network_wait}s.", color=Fore.YELLOW, style=Style.DIM)
            time.sleep(network_wait)
            error_occurred = True # Mark error, but allow other calls to try
        except ccxt.AuthenticationError as e:
            print_color(f"Authentication Error ({call['desc']}). Check API Keys!", color=Fore.RED, style=Style.BRIGHT)
            error_occurred = True; raise e # Fatal error, stop the script
        except Exception as e:
            # Be more specific about non-critical errors vs critical ones
            if call['desc'] in ["ticker", "Indicator OHLCV", "Pivot OHLCV"]:
                 print_color(f"Error fetching {call['desc']}: {e}", color=Fore.RED, style=Style.DIM)
                 results[call["desc"]] = None # Ensure failed critical fetches result in None
            else: # For positions/balance, a failure is less critical for display loop
                 print_color(f"Warning fetching {call['desc']}: {e}", color=Fore.YELLOW, style=Style.DIM)
            error_occurred = True # Mark that *an* error occurred

    sys.stdout.write("\033[K") # Clear the "Contacting..." message
    return results, error_occurred

def analyze_orderbook_volume(exchange, symbol, market_info, config):
    """Fetches and analyzes the order book volume."""
    print_color(f"{Fore.CYAN}# Summoning order book spirits...", style=Style.DIM, end='\r')
    try:
        orderbook = exchange.fetch_order_book(symbol, limit=config["ORDER_FETCH_LIMIT"])
        sys.stdout.write("\033[K")
    except ccxt.RateLimitExceeded:
        sys.stdout.write("\033[K"); print_color("Rate Limit (OB). Pausing...", color=Fore.YELLOW); time.sleep(config["RETRY_DELAY_RATE_LIMIT"]); return None, True
    except ccxt.NetworkError:
        sys.stdout.write("\033[K"); print_color("Network Error (OB). Retrying...", color=Fore.YELLOW); time.sleep(config["RETRY_DELAY_NETWORK_ERROR"]); return None, True
    except Exception as e:
        sys.stdout.write("\033[K"); print_color(f"Unexpected Error (OB): {e}", color=Fore.RED); return None, False # Non-retryable error

    if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
        print_color(f"Order book for {symbol} is empty or unavailable.", color=Fore.YELLOW); return None, False

    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    vol_disp_prec = config["VOLUME_DISPLAY_PRECISION"]
    volume_thresholds = config["VOLUME_THRESHOLDS"]

    analyzed_orderbook = {
        'symbol': symbol, 'timestamp': exchange.iso8601(exchange.milliseconds()),
        'asks': [], 'bids': [], # Store processed levels here
        'ask_total_volume': decimal.Decimal('0'), 'bid_total_volume': decimal.Decimal('0'), # Total vol in fetched levels
        'ask_weighted_price': decimal.Decimal('0'), 'bid_weighted_price': decimal.Decimal('0'),
        'volume_imbalance_ratio': decimal.Decimal('0'),
        'cumulative_ask_volume_displayed': decimal.Decimal('0'), # Cumulative vol within display depth
        'cumulative_bid_volume_displayed': decimal.Decimal('0')
    }

    ask_vol_price_sum = decimal.Decimal('0')
    for i, ask in enumerate(orderbook['asks']):
        try: price, volume = decimal.Decimal(str(ask[0])), decimal.Decimal(str(ask[1]))
        except: continue # Skip malformed level

        analyzed_orderbook['ask_total_volume'] += volume
        ask_vol_price_sum += price * volume

        # Process only up to display depth
        if i < config["MAX_ORDERBOOK_DEPTH_DISPLAY"]:
            analyzed_orderbook['cumulative_ask_volume_displayed'] += volume
            volume_str = format_decimal(volume, amount_prec, vol_disp_prec)
            highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
            if volume >= volume_thresholds.get('high', decimal.Decimal('inf')): highlight_color, highlight_style = Fore.LIGHTRED_EX, Style.BRIGHT
            elif volume >= volume_thresholds.get('medium', decimal.Decimal('0')): highlight_color, highlight_style = Fore.RED, Style.NORMAL
            analyzed_orderbook['asks'].append({
                'price': price, 'volume': volume, 'volume_str': volume_str, 'color': highlight_color, 'style': highlight_style,
                'cumulative_volume': format_decimal(analyzed_orderbook['cumulative_ask_volume_displayed'], amount_prec, vol_disp_prec)
            })

    bid_vol_price_sum = decimal.Decimal('0')
    for i, bid in enumerate(orderbook['bids']):
        try: price, volume = decimal.Decimal(str(bid[0])), decimal.Decimal(str(bid[1]))
        except: continue # Skip malformed level

        analyzed_orderbook['bid_total_volume'] += volume
        bid_vol_price_sum += price * volume

        # Process only up to display depth
        if i < config["MAX_ORDERBOOK_DEPTH_DISPLAY"]:
            analyzed_orderbook['cumulative_bid_volume_displayed'] += volume
            volume_str = format_decimal(volume, amount_prec, vol_disp_prec)
            highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
            if volume >= volume_thresholds.get('high', decimal.Decimal('inf')): highlight_color, highlight_style = Fore.LIGHTGREEN_EX, Style.BRIGHT
            elif volume >= volume_thresholds.get('medium', decimal.Decimal('0')): highlight_color, highlight_style = Fore.GREEN, Style.NORMAL
            analyzed_orderbook['bids'].append({
                'price': price, 'volume': volume, 'volume_str': volume_str, 'color': highlight_color, 'style': highlight_style,
                'cumulative_volume': format_decimal(analyzed_orderbook['cumulative_bid_volume_displayed'], amount_prec, vol_disp_prec)
            })

    # Calculate VWAP and Imbalance based on fetched depth
    ask_total_vol = analyzed_orderbook['ask_total_volume']
    bid_total_vol = analyzed_orderbook['bid_total_volume']
    if ask_total_vol > 0:
        analyzed_orderbook['ask_weighted_price'] = ask_vol_price_sum / ask_total_vol
        analyzed_orderbook['volume_imbalance_ratio'] = bid_total_vol / ask_total_vol
    else: # Avoid division by zero
        analyzed_orderbook['ask_weighted_price'] = decimal.Decimal('0')
        analyzed_orderbook['volume_imbalance_ratio'] = decimal.Decimal('inf') if bid_total_vol > 0 else decimal.Decimal('0')

    if bid_total_vol > 0:
        analyzed_orderbook['bid_weighted_price'] = bid_vol_price_sum / bid_total_vol
    else:
        analyzed_orderbook['bid_weighted_price'] = decimal.Decimal('0')

    return analyzed_orderbook, False # Success

# ==============================================================================
# Display Functions
# ==============================================================================

def display_header(symbol, timestamp, balance_info, config):
    """Displays the main header section with symbol, time, and balance."""
    print_color("=" * 80, color=Fore.CYAN)
    print_color(f"📜 Pyrmethus's Market Vision: {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{timestamp}", color=Fore.CYAN)
    balance_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    asset = config["FETCH_BALANCE_ASSET"]
    bal_disp_prec = config["BALANCE_DISPLAY_PRECISION"]
    if balance_info is not None:
        try: balance_str = f"{Fore.GREEN}{format_decimal(decimal.Decimal(str(balance_info)), bal_disp_prec, bal_disp_prec)} {asset}{Style.RESET_ALL}"
        except: balance_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"
    print_color(f"💰 Available Balance ({asset}): {balance_str}")
    print_color("-" * 80, color=Fore.CYAN)

def display_ticker_and_trend(ticker_info, indicators_info, config, market_info):
    """Displays the last price and primary trend indicator (SMA1)."""
    price_prec = market_info['price_precision']
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    last_price, current_price_str, price_color = None, f"{Fore.YELLOW}N/A{Style.RESET_ALL}", Fore.WHITE

    if ticker_info and ticker_info.get('last') is not None:
        try:
            last_price = decimal.Decimal(str(ticker_info['last']))
            sma1 = indicators_info.get('sma1', {}).get('value')
            if sma1:
                if last_price > sma1: price_color = Fore.GREEN
                elif last_price < sma1: price_color = Fore.RED
                else: price_color = Fore.YELLOW
            current_price_str = f"{price_color}{Style.BRIGHT}{format_decimal(last_price, price_prec, min_disp_prec)}{Style.RESET_ALL}"
        except: last_price = None; current_price_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"

    sma1_val = indicators_info.get('sma1', {}).get('value')
    sma1_error = indicators_info.get('sma1', {}).get('error')
    sma1_period = config['SMA_PERIOD']
    tf = config['INDICATOR_TIMEFRAME']
    trend_str, trend_color = f"Trend ({sma1_period}@{tf}): -", Fore.YELLOW

    if sma1_error: trend_str, trend_color = f"Trend ({sma1_period}@{tf}): SMA Error", Fore.YELLOW
    elif sma1_val and last_price:
        sma1_str_fmt = format_decimal(sma1_val, price_prec, min_disp_prec)
        if last_price > sma1_val: trend_str, trend_color = f"Above {sma1_period}@{tf} SMA ({sma1_str_fmt})", Fore.GREEN
        elif last_price < sma1_val: trend_str, trend_color = f"Below {sma1_period}@{tf} SMA ({sma1_str_fmt})", Fore.RED
        else: trend_str, trend_color = f"On {sma1_period}@{tf} SMA ({sma1_str_fmt})", Fore.YELLOW
    else: trend_str = f"Trend ({sma1_period}@{tf}): SMA unavailable"

    print_color(f"  Last Price: {current_price_str} | {trend_color}{trend_str}{Style.RESET_ALL}")
    return last_price # Return parsed last price for other displays

def display_indicators(indicators_info, config, market_info, last_price):
    """Displays secondary indicators like SMA2, EMAs, Momentum, Stoch RSI."""
    price_prec = market_info['price_precision']
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    stoch_disp_prec = config["STOCH_RSI_DISPLAY_PRECISION"]
    tf = config['INDICATOR_TIMEFRAME']

    # SMA2 Display
    sma2_val = indicators_info.get('sma2', {}).get('value')
    sma2_error = indicators_info.get('sma2', {}).get('error')
    sma2_period = config['SMA2_PERIOD']
    sma2_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    sma2_trend_str, sma2_trend_color = f"SMA ({sma2_period}@{tf}): -", Fore.YELLOW
    if sma2_error: sma2_trend_str, sma2_trend_color = f"SMA ({sma2_period}@{tf}): Error", Fore.YELLOW
    elif sma2_val:
        sma2_str_fmt = format_decimal(sma2_val, price_prec, min_disp_prec)
        sma2_str = f"{Fore.YELLOW}{sma2_str_fmt}{Style.RESET_ALL}"
        if last_price:
            if last_price > sma2_val: sma2_trend_str, sma2_trend_color = f"Above {sma2_period}@{tf} SMA", Fore.GREEN
            elif last_price < sma2_val: sma2_trend_str, sma2_trend_color = f"Below {sma2_period}@{tf} SMA", Fore.RED
            else: sma2_trend_str, sma2_trend_color = f"On {sma2_period}@{tf} SMA", Fore.YELLOW
        else: sma2_trend_str, sma2_trend_color = f"SMA ({sma2_period}@{tf}): OK", Fore.WHITE
    else: sma2_trend_str = f"SMA ({sma2_period}@{tf}): unavailable"
    print_color(f"  SMA ({sma2_period}@{tf}): {sma2_str} | {sma2_trend_color}{sma2_trend_str}{Style.RESET_ALL}")

    # EMA Display
    ema1_val = indicators_info.get('ema1', {}).get('value')
    ema2_val = indicators_info.get('ema2', {}).get('value')
    ema_error = indicators_info.get('ema1', {}).get('error') or indicators_info.get('ema2', {}).get('error')
    ema1_period, ema2_period = config['EMA1_PERIOD'], config['EMA2_PERIOD']
    ema1_str, ema2_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}", f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    ema_trend_str, ema_trend_color, ema_trend_signal = f"EMA ({ema1_period}/{ema2_period}@{tf}): -", Fore.YELLOW, ""
    if ema_error: ema_trend_str, ema_trend_color = f"EMA ({ema1_period}/{ema2_period}@{tf}): Error", Fore.YELLOW
    elif ema1_val and ema2_val:
        ema1_str_fmt = format_decimal(ema1_val, price_prec, min_disp_prec)
        ema2_str_fmt = format_decimal(ema2_val, price_prec, min_disp_prec)
        ema1_str, ema2_str = f"{Fore.YELLOW}{ema1_str_fmt}{Style.RESET_ALL}", f"{Fore.YELLOW}{ema2_str_fmt}{Style.RESET_ALL}"
        if last_price:
            if ema1_val > ema2_val and last_price > ema1_val: ema_trend_color, ema_trend_signal = Fore.GREEN, "(Bullish)"
            elif ema1_val < ema2_val and last_price < ema1_val: ema_trend_color, ema_trend_signal = Fore.RED, "(Bearish)"
            elif ema1_val > ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTGREEN_EX, "(Neutral-Bull)"
            elif ema1_val < ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTRED_EX, "(Neutral-Bear)"
            else: ema_trend_color, ema_trend_signal = Fore.YELLOW, "(Neutral)"
        else: # No price context
            if ema1_val > ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTGREEN_EX, "(Crossed Up)"
            elif ema1_val < ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTRED_EX, "(Crossed Down)"
            else: ema_trend_color, ema_trend_signal = Fore.YELLOW, "(Neutral)"
        ema_trend_str = f"EMA ({ema1_period}/{ema2_period}@{tf})"
    else: ema_trend_str = f"EMA ({ema1_period}/{ema2_period}@{tf}): unavailable"
    print_color(f"  EMAs: {ema1_str} ({ema1_period}), {ema2_str} ({ema2_period}) | {ema_trend_color}{ema_trend_str} {ema_trend_signal}{Style.RESET_ALL}")

    # Momentum Display
    mom_val = indicators_info.get('momentum', {}).get('value')
    mom_error = indicators_info.get('momentum', {}).get('error')
    mom_period = config['MOMENTUM_PERIOD']
    momentum_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    momentum_trend_str, momentum_color = f"Momentum ({mom_period}@{tf}): -", Fore.YELLOW
    if mom_error: momentum_trend_str, momentum_color = f"Momentum ({mom_period}@{tf}): Error", Fore.YELLOW
    elif mom_val is not None:
        momentum_str_fmt = format_decimal(mom_val, price_prec, min_disp_prec)
        momentum_str = f"{Fore.YELLOW}{momentum_str_fmt}{Style.RESET_ALL}"
        if mom_val > 0: momentum_color, momentum_trend_str = Fore.GREEN, f"Momentum ({mom_period}@{tf}): Positive"
        elif mom_val < 0: momentum_color, momentum_trend_str = Fore.RED, f"Momentum ({mom_period}@{tf}): Negative"
        else: momentum_color, momentum_trend_str = Fore.YELLOW, f"Momentum ({mom_period}@{tf}): Neutral"
    else: momentum_trend_str = f"Momentum ({mom_period}@{tf}): unavailable"
    print_color(f"  {momentum_color}{momentum_trend_str}: {momentum_str}{Style.RESET_ALL}")

    # Stoch RSI Display
    stoch_k_val = indicators_info.get('stoch_rsi', {}).get('k')
    stoch_d_val = indicators_info.get('stoch_rsi', {}).get('d')
    stoch_error = indicators_info.get('stoch_rsi', {}).get('error')
    rsi_p, k_p, d_p = config['RSI_PERIOD'], config['STOCH_K_PERIOD'], config['STOCH_D_PERIOD']
    stoch_k_str, stoch_d_str, stoch_color, stoch_signal = "N/A", "N/A", Fore.WHITE, ""
    if stoch_error:
        stoch_k_str, stoch_d_str = "Error", "Error"; stoch_color = Fore.YELLOW
        stoch_signal = f"({stoch_error})" if isinstance(stoch_error, str) else "(Error)"
    elif stoch_k_val is not None:
        stoch_k_str = format_decimal(stoch_k_val, stoch_disp_prec)
        stoch_d_str = format_decimal(stoch_d_val, stoch_disp_prec) if stoch_d_val is not None else "N/A"
        oversold_lim = config['STOCH_RSI_OVERSOLD']
        overbought_lim = config['STOCH_RSI_OVERBOUGHT']
        is_oversold = stoch_k_val < oversold_lim and (stoch_d_val is None or stoch_d_val < oversold_lim)
        is_overbought = stoch_k_val > overbought_lim and (stoch_d_val is None or stoch_d_val > overbought_lim)
        if is_oversold: stoch_color, stoch_signal = Fore.GREEN, "(Oversold)"
        elif is_overbought: stoch_color, stoch_signal = Fore.RED, "(Overbought)"
        elif stoch_d_val: # Check cross only if D exists
            if stoch_k_val > stoch_d_val: stoch_color = Fore.LIGHTGREEN_EX
            elif stoch_k_val < stoch_d_val: stoch_color = Fore.LIGHTRED_EX
            else: stoch_color = Fore.WHITE # Neutral color if K exists but no cross or D doesn't exist
    print_color(f"  Stoch RSI ({rsi_p},{k_p},{d_p}@{tf}): {stoch_color}%K={stoch_k_str}, %D={stoch_d_str} {stoch_signal}{Style.RESET_ALL}")

def display_position(position_info, ticker_info, market_info, config):
    """Displays current position information, calculating PNL if needed."""
    pnl_prec = config["PNL_PRECISION"]
    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    pnl_str = f"{Fore.LIGHTBLACK_EX}Position: None or Fetch Failed{Style.RESET_ALL}"

    if position_info.get('has_position'):
        pos = position_info['position']
        side = pos.get('side', 'N/A').capitalize()
        size_str = pos.get('contracts', '0')
        entry_price_str = pos.get('entryPrice', '0')
        quote_asset = pos.get('quoteAsset', config['FETCH_BALANCE_ASSET'])
        pnl_val = position_info.get('unrealizedPnl') # Already Decimal or None from processing step

        try:
            size = decimal.Decimal(size_str)
            entry_price = decimal.Decimal(entry_price_str)
            size_fmt = format_decimal(size, amount_prec)
            entry_fmt = format_decimal(entry_price, price_prec, min_disp_prec)
            side_color = Fore.GREEN if side.lower() == 'long' else Fore.RED if side.lower() == 'short' else Fore.WHITE

            # Calculate PNL if not directly provided or parsing failed
            if pnl_val is None and ticker_info and ticker_info.get('last') is not None and entry_price > 0 and size != 0:
                last_price_for_pnl = decimal.Decimal(str(ticker_info['last']))
                if side.lower() == 'long': pnl_val = (last_price_for_pnl - entry_price) * size
                else: pnl_val = (entry_price - last_price_for_pnl) * size

            # Format PNL for display
            pnl_val_str, pnl_color = "N/A", Fore.WHITE
            if pnl_val is not None:
                pnl_val_str = format_decimal(pnl_val, pnl_prec)
                pnl_color = Fore.GREEN if pnl_val > 0 else Fore.RED if pnl_val < 0 else Fore.WHITE

            pnl_str = (f"Position: {side_color}{side} {size_fmt}{Style.RESET_ALL} | "
                       f"Entry: {Fore.YELLOW}{entry_fmt}{Style.RESET_ALL} | "
                       f"uPNL: {pnl_color}{pnl_val_str} {quote_asset}{Style.RESET_ALL}")

        except Exception as e:
            pnl_str = f"{Fore.YELLOW}Position: Error parsing data ({e}){Style.RESET_ALL}"

    print_color(f"  {pnl_str}")

def display_pivots(pivots_info, last_price, market_info, config):
    """Displays Fibonacci Pivot Points."""
    print_color("--- Fibonacci Pivots (Prev Day) ---", color=Fore.BLUE)
    price_prec = market_info['price_precision']
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    pivot_width = max(10, price_prec + 6) # Dynamic width

    if pivots_info:
        pivot_lines = {}
        levels = ['R3', 'R2', 'R1', 'PP', 'S1', 'S2', 'S3']
        for level in levels:
            value = pivots_info.get(level)
            if value is not None:
                level_str = f"{level}:".ljust(4)
                value_str_aligned = format_decimal(value, price_prec, min_disp_prec).rjust(pivot_width)
                level_color = Fore.RED if 'R' in level else Fore.GREEN if 'S' in level else Fore.YELLOW
                highlight = ""
                # Highlight if price is near pivot
                if last_price:
                    try:
                        diff_ratio = abs(last_price - value) / last_price if last_price else decimal.Decimal('inf')
                        if diff_ratio < decimal.Decimal('0.001'): # Highlight within 0.1%
                            highlight = Back.LIGHTBLACK_EX + Fore.WHITE + Style.BRIGHT + " *NEAR* " + Style.RESET_ALL
                    except: pass # Ignore highlight errors
                pivot_lines[level] = f"  {level_color}{level_str}{value_str_aligned}{Style.RESET_ALL}{highlight}"
            else:
                pivot_lines[level] = f"  {level}:".ljust(6) + f"{'N/A':>{pivot_width}}"

        # Print in structured layout
        print(f"{pivot_lines.get('R3','')}")
        print(f"{pivot_lines.get('R2','').ljust(pivot_width + 15)} {pivot_lines.get('S3','')}") # Align R2/S3 roughly
        print(f"{pivot_lines.get('R1','').ljust(pivot_width + 15)} {pivot_lines.get('S2','')}") # Align R1/S2 roughly
        print(f"{pivot_lines.get('PP','').ljust(pivot_width + 15)} {pivot_lines.get('S1','')}") # Align PP/S1 roughly

    else:
        print_color(f"  {Fore.YELLOW}Pivot data unavailable.{Style.RESET_ALL}")

def display_orderbook(analyzed_orderbook, market_info, config):
    """Displays the analyzed order book with indices for selection."""
    print_color("--- Order Book Depths ---", color=Fore.BLUE)
    if not analyzed_orderbook:
        print_color(f"  {Fore.YELLOW}Order book data unavailable.{Style.RESET_ALL}")
        return

    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    vol_disp_prec = config["VOLUME_DISPLAY_PRECISION"]

    # Calculate dynamic column widths
    idx_width = 4 # Width for index like "[A1]"
    price_width = max(10, price_prec + 4)
    volume_width = max(12, vol_disp_prec + 6)
    cum_volume_width = max(15, vol_disp_prec + 10) # Width for "(Cum: ...)" part

    ask_lines, bid_lines = [], []
    ask_index_map, bid_index_map = {}, {} # To map simple index to price

    # Prepare Ask lines (reversed display: lowest ask near spread)
    # Enumerate asks in reverse for display indexing (A1 is best ask)
    # asks stored are lowest price first, so reverse the list first
    # then enumerate the reversed list
    display_asks = list(reversed(analyzed_orderbook['asks']))
    for idx, ask in enumerate(display_asks):
        ask_idx_str = f"[A{idx+1}]".ljust(idx_width)
        price_str = format_decimal(ask['price'], price_prec, min_disp_prec)
        cum_vol_str = f"{Fore.LIGHTBLACK_EX}(Cum: {ask['cumulative_volume']}){Style.RESET_ALL}"
        ask_lines.append(
            f"{Fore.CYAN}{ask_idx_str}{Style.RESET_ALL}"
            f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}"
            f"{ask['style']}{ask['color']}{ask['volume_str']:<{volume_width}} "
            f"{cum_vol_str}"
        )
        ask_index_map[idx + 1] = ask['price'] # Map A1, A2... to price

    # Prepare Bid lines (B1 is best bid)
    for idx, bid in enumerate(analyzed_orderbook['bids']):
        bid_idx_str = f"[B{idx+1}]".ljust(idx_width)
        price_str = format_decimal(bid['price'], price_prec, min_disp_prec)
        cum_vol_str = f"{Fore.LIGHTBLACK_EX}(Cum: {bid['cumulative_volume']}){Style.RESET_ALL}"
        bid_lines.append(
            f"{Fore.CYAN}{bid_idx_str}{Style.RESET_ALL}"
            f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}"
            f"{bid['style']}{bid['color']}{bid['volume_str']:<{volume_width}} "
            f"{cum_vol_str}"
        )
        bid_index_map[idx + 1] = bid['price'] # Map B1, B2... to price

    # Print header
    col_width_total = idx_width + price_width + volume_width + len("(Cum: )") + cum_volume_width # Approximate total width
    print_color(f"{'Asks':^{col_width_total}}{'Bids':^{col_width_total}}", color=Fore.LIGHTBLACK_EX)
    print_color(f"{'-'*col_width_total:<{col_width_total}} {'-'*col_width_total:<{col_width_total}}", color=Fore.LIGHTBLACK_EX)

    # Print rows side-by-side
    max_rows = max(len(ask_lines), len(bid_lines))
    for i in range(max_rows):
        ask_part = ask_lines[i] if i < len(ask_lines) else ''
        bid_part = bid_lines[i] if i < len(bid_lines) else ''
        print(f"{ask_part:<{col_width_total}}  {bid_part}") # Adjust padding if needed

    # Calculate and display spread
    best_ask = display_asks[0]['price'] if display_asks else decimal.Decimal('NaN')
    best_bid = analyzed_orderbook['bids'][0]['price'] if analyzed_orderbook['bids'] else decimal.Decimal('NaN')
    spread = best_ask - best_bid if best_ask.is_finite() and best_bid.is_finite() else decimal.Decimal('NaN')
    spread_str = format_decimal(spread, price_prec, min_disp_prec) if spread.is_finite() else "N/A"
    print_color(f"\n--- Spread: {spread_str} ---", color=Fore.MAGENTA, style=Style.DIM)

    # Return the index maps for interactive selection
    return ask_index_map, bid_index_map

def display_volume_analysis(analyzed_orderbook, market_info, config):
    """Displays summary volume analysis."""
    if not analyzed_orderbook: return
    amount_prec = market_info['amount_precision']
    price_prec = market_info['price_precision']
    vol_disp_prec = config["VOLUME_DISPLAY_PRECISION"]
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]

    print_color("\n--- Volume Analysis (Fetched Depth) ---", color=Fore.BLUE)
    total_ask_vol = analyzed_orderbook['ask_total_volume']
    total_bid_vol = analyzed_orderbook['bid_total_volume']
    cum_ask_vol_disp = analyzed_orderbook['cumulative_ask_volume_displayed']
    cum_bid_vol_disp = analyzed_orderbook['cumulative_bid_volume_displayed']

    print_color(f"  Total Ask Vol: {Fore.RED}{format_decimal(total_ask_vol, amount_prec, vol_disp_prec)}{Style.RESET_ALL} | Total Bid Vol : {Fore.GREEN}{format_decimal(total_bid_vol, amount_prec, vol_disp_prec)}{Style.RESET_ALL}")
    print_color(f"  Cumul Ask (Disp): {Fore.RED}{format_decimal(cum_ask_vol_disp, amount_prec, vol_disp_prec)}{Style.RESET_ALL} | Cumul Bid (Disp): {Fore.GREEN}{format_decimal(cum_bid_vol_disp, amount_prec, vol_disp_prec)}{Style.RESET_ALL}")

    imbalance_ratio = analyzed_orderbook['volume_imbalance_ratio']
    imbalance_color = Fore.WHITE
    imbalance_str = "N/A"
    if imbalance_ratio.is_infinite(): imbalance_color, imbalance_str = Fore.LIGHTGREEN_EX, "Inf"
    elif imbalance_ratio.is_finite():
        imbalance_str = format_decimal(imbalance_ratio, 2)
        if imbalance_ratio > decimal.Decimal('1.5'): imbalance_color = Fore.GREEN
        elif imbalance_ratio < decimal.Decimal('0.67') and not imbalance_ratio.is_zero(): imbalance_color = Fore.RED
        elif imbalance_ratio.is_zero() and total_ask_vol > 0: imbalance_color = Fore.LIGHTRED_EX # Zero bids

    ask_vwap_str = format_decimal(analyzed_orderbook['ask_weighted_price'], price_prec, min_disp_prec)
    bid_vwap_str = format_decimal(analyzed_orderbook['bid_weighted_price'], price_prec, min_disp_prec)

    print_color(f"  Imbalance (B/A): {imbalance_color}{imbalance_str}{Style.RESET_ALL} | Ask VWAP: {Fore.YELLOW}{ask_vwap_str}{Style.RESET_ALL} | Bid VWAP: {Fore.YELLOW}{bid_vwap_str}{Style.RESET_ALL}")

    # Pressure Reading
    print_color("--- Pressure Reading ---", color=Fore.BLUE)
    if imbalance_ratio.is_infinite(): print_color("  Extreme Bid Dominance", color=Fore.LIGHTYELLOW_EX)
    elif imbalance_ratio.is_zero() and total_ask_vol > 0: print_color("  Extreme Ask Dominance", color=Fore.LIGHTYELLOW_EX)
    elif imbalance_ratio > decimal.Decimal('1.5'): print_color("  Strong Buy Pressure", color=Fore.GREEN, style=Style.BRIGHT)
    elif imbalance_ratio < decimal.Decimal('0.67'): print_color("  Strong Sell Pressure", color=Fore.RED, style=Style.BRIGHT)
    else: print_color("  Volume Relatively Balanced", color=Fore.WHITE)
    print_color("=" * 80, color=Fore.CYAN)

def display_combined_analysis(analysis_data, market_info, config):
    """Orchestrates the display of all analyzed data sections."""
    analyzed_orderbook = analysis_data['orderbook']
    ticker_info = analysis_data['ticker']
    indicators_info = analysis_data['indicators']
    position_info = analysis_data['position']
    pivots_info = analysis_data['pivots']
    balance_info = analysis_data['balance']
    timestamp = analysis_data.get('timestamp', exchange.iso8601(exchange.milliseconds())) # Use OB timestamp or current

    symbol = market_info['symbol']
    # print("\033[H\033[J", end="") # Optional: Clear screen

    display_header(symbol, timestamp, balance_info, config)
    last_price = display_ticker_and_trend(ticker_info, indicators_info, config, market_info)
    display_indicators(indicators_info, config, market_info, last_price)
    display_position(position_info, ticker_info, market_info, config) # Pass ticker for PNL calc fallback
    display_pivots(pivots_info, last_price, market_info, config) # Pass price for highlighting
    ask_map, bid_map = display_orderbook(analyzed_orderbook, market_info, config) # Handles None OB internally
    display_volume_analysis(analyzed_orderbook, market_info, config) # Handles None OB internally

    return ask_map, bid_map # Return maps for interactive limit order placement

# ==============================================================================
# Trading Spell Functions
# ==============================================================================

def place_market_order(exchange, symbol, side, amount_str, market_info):
    """Places a market order with validation and confirmation."""
    print_color(f"{Fore.CYAN}# Preparing {side.upper()} market order spell...{Style.RESET_ALL}")
    try:
        amount = decimal.Decimal(amount_str)
        min_amount = market_info.get('min_amount', decimal.Decimal('0'))
        amount_prec = market_info['amount_precision']
        amount_step = market_info.get('amount_step', decimal.Decimal('1') / (decimal.Decimal('10') ** amount_prec))

        if amount <= 0:
            print_color("Amount must be positive.", color=Fore.YELLOW); return

        if min_amount > 0 and amount < min_amount:
            min_amount_fmt = format_decimal(min_amount, amount_prec)
            print_color(f"Amount {amount_str} below minimum ({min_amount_fmt}).", color=Fore.YELLOW); return

        # Round amount to nearest valid step (downwards)
        if amount_step > 0 and (amount % amount_step) != 0:
            rounded_amount = (amount // amount_step) * amount_step
            if min_amount > 0 and rounded_amount < min_amount:
                print_color(f"Rounding to {rounded_amount} is below minimum.", color=Fore.YELLOW); return
            confirm_round = input(f"{Fore.YELLOW}Round amount to {format_decimal(rounded_amount, amount_prec)}? (yes/no): {Style.RESET_ALL}").lower()
            if confirm_round != 'yes':
                print_color("Order cancelled.", color=Fore.YELLOW); return
            amount = rounded_amount
        amount_str = format_decimal(amount, amount_prec) # Use formatted rounded amount

    except (decimal.InvalidOperation, ValueError) as e:
        print_color(f"Invalid amount: {e}", color=Fore.YELLOW); return
    except KeyError as e:
        print_color(f"Market info error: {e}", color=Fore.YELLOW); return

    side_color = Fore.GREEN if side == 'buy' else Fore.RED
    prompt = (f"{Style.BRIGHT}Confirm MARKET {side_color}{side.upper()}{Style.RESET_ALL} order: "
              f"{Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} "
              f"({Fore.GREEN}yes{Style.RESET_ALL}/{Fore.RED}no{Style.RESET_ALL}): {Style.RESET_ALL}")
    try: confirm = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt): print_color("\nOrder cancelled.", color=Fore.YELLOW); return

    if confirm == 'yes':
        print_color(f"{Fore.CYAN}# Transmitting market order...", style=Style.DIM, end='\r')
        try:
            amount_float = float(amount) # CCXT often expects floats
            params = {} # Add exchange-specific params if needed (e.g., positionIdx for Bybit hedge mode)
            order = exchange.create_market_order(symbol, side, amount_float, params=params)
            sys.stdout.write("\033[K")
            order_id = order.get('id', 'N/A')
            avg_price = order.get('average')
            filled = order.get('filled')
            confirmation_msg = f"✅ Market {side.upper()} [{order_id}] Placed!"
            details = []
            if filled: details.append(f"Filled: {format_decimal(filled, amount_prec)}")
            if avg_price: details.append(f"Avg Price: {format_decimal(avg_price, market_info['price_precision'])}")
            if details: confirmation_msg += f" ({', '.join(details)})"
            print_color(f"\n{confirmation_msg}", color=Fore.GREEN, style=Style.BRIGHT)
            termux_toast(f"{symbol} Market {side.upper()} Placed: {order_id}")
        except ccxt.InsufficientFunds as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Insufficient Funds: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Market Order Failed: Insufficient Funds", duration="long")
        except ccxt.ExchangeError as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Exchange Error: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Market Order Failed: ExchangeError", duration="long")
        except Exception as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Placement Error: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Market Order Failed: PlacementError", duration="long")
    else: print_color("Market order cancelled.", color=Fore.YELLOW)

def place_limit_order(exchange, symbol, side, amount_str, price_str, market_info):
    """Places a limit order with validation and confirmation."""
    print_color(f"{Fore.CYAN}# Preparing {side.upper()} limit order spell...{Style.RESET_ALL}")
    try:
        amount = decimal.Decimal(amount_str)
        price = decimal.Decimal(price_str)
        min_amount = market_info['min_amount']
        amount_prec = market_info['amount_precision']
        price_prec = market_info['price_precision']
        amount_step = market_info['amount_step']
        price_tick_size = market_info['price_tick_size']

        if amount <= 0 or price <= 0:
            print_color("Amount and Price must be positive.", color=Fore.YELLOW); return

        if min_amount > 0 and amount < min_amount:
            print_color(f"Amount below minimum ({format_decimal(min_amount, amount_prec)}).", color=Fore.YELLOW); return

        # Round amount DOWN to nearest valid step
        if amount_step > 0 and (amount % amount_step) != 0:
            original_amount = amount
            amount = (amount // amount_step) * amount_step
            print_color(f"Amount rounded down from {original_amount} to {amount} due to step size {amount_step}", color=Fore.YELLOW)
            if min_amount > 0 and amount < min_amount:
                print_color(f"Rounded amount {amount} is below minimum.", color=Fore.RED); return

        # Round price to nearest valid tick size (adjust rounding mode if needed)
        if price_tick_size > 0 and (price % price_tick_size) != 0:
            original_price = price
            # Adjust rounding based on side? Often use floor for bids, ceil for asks, or just nearest.
            # Using quantize with ROUND_HALF_UP as a default. Be careful with this.
            price = price.quantize(price_tick_size, rounding=decimal.ROUND_HALF_UP)
            print_color(f"Price rounded from {original_price} to {price} due to tick size {price_tick_size}", color=Fore.YELLOW)

        amount_str = format_decimal(amount, amount_prec)
        price_str = format_decimal(price, price_prec)

    except (decimal.InvalidOperation, ValueError) as e:
        print_color(f"Invalid amount or price: {e}", color=Fore.YELLOW); return
    except KeyError as e:
        print_color(f"Market info error: {e}", color=Fore.YELLOW); return

    side_color = Fore.GREEN if side == 'buy' else Fore.RED
    prompt = (f"{Style.BRIGHT}Confirm LIMIT {side_color}{side.upper()}{Style.RESET_ALL} order: "
              f"{Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{price_str}{Style.RESET_ALL} "
              f"({Fore.GREEN}yes{Style.RESET_ALL}/{Fore.RED}no{Style.RESET_ALL}): {Style.RESET_ALL}")
    try: confirm = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt): print_color("\nLimit order cancelled.", color=Fore.YELLOW); return

    if confirm == 'yes':
        print_color(f"{Fore.CYAN}# Transmitting limit order...", style=Style.DIM, end='\r')
        try:
            amount_float = float(amount)
            price_float = float(price)
            params = {} # Add specific params if needed
            order = exchange.create_limit_order(symbol, side, amount_float, price_float, params=params)
            sys.stdout.write("\033[K")
            order_id = order.get('id', 'N/A')
            limit_price = order.get('price')
            order_amount = order.get('amount')
            confirmation_msg = f"✅ Limit {side.upper()} [{order_id}] Placed!"
            details = []
            if order_amount: details.append(f"Amount: {format_decimal(order_amount, amount_prec)}")
            if limit_price: details.append(f"Price: {format_decimal(limit_price, market_info['price_precision'])}")
            if details: confirmation_msg += f" ({', '.join(details)})"
            print_color(f"\n{confirmation_msg}", color=Fore.GREEN, style=Style.BRIGHT)
            termux_toast(f"{symbol} Limit {side.upper()} Placed: {order_id}")
        except ccxt.InsufficientFunds as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Insufficient Funds: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Limit Order Failed: Insufficient Funds", duration="long")
        except ccxt.InvalidOrder as e: # Catch rounding/tick size issues
             sys.stdout.write("\033[K"); print_color(f"\n❌ Invalid Order (Tick/Step Error?): {e}", color=Fore.RED, style=Style.BRIGHT)
             termux_toast(f"{symbol} Limit Order Failed: Invalid Order", duration="long")
        except ccxt.ExchangeError as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Exchange Error: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Limit Order Failed: ExchangeError", duration="long")
        except Exception as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Placement Error: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Limit Order Failed: PlacementError", duration="long")
    else: print_color("Limit order cancelled.", color=Fore.YELLOW)

def place_limit_order_interactive(exchange, symbol, side, ask_map, bid_map, market_info):
    """Handles interactive limit order placement by selecting from the order book."""
    print_color(f"\n{Fore.BLUE}--- Interactive Limit Order ({side.upper()}) ---{Style.RESET_ALL}")

    target_map = ask_map if side == 'sell' else bid_map # Sell into bids, Buy from asks
    prompt_char = 'B' if side == 'buy' else 'A' # Prompt for Bid index if buying, Ask index if selling

    if not target_map:
        print_color("Order book side is empty, cannot select price.", color=Fore.YELLOW)
        return

    while True:
        try:
            index_str = input(f"{Style.BRIGHT}Select Order Book Index ({prompt_char}1, {prompt_char}2, ...) or '{Fore.YELLOW}cancel{Style.RESET_ALL}': {Style.RESET_ALL}").strip().upper()
            if index_str == 'CANCEL':
                print_color("Interactive limit order cancelled.", color=Fore.YELLOW)
                return

            if not index_str.startswith(prompt_char):
                 print_color(f"Invalid format. Use '{prompt_char}' followed by number (e.g., {prompt_char}1).", color=Fore.YELLOW)
                 continue

            index = int(index_str[1:]) # Get number after 'A' or 'B'
            selected_price = target_map.get(index)

            if selected_price is None:
                print_color(f"Index {index_str} not found in the displayed order book.", color=Fore.YELLOW)
            else:
                price_str = format_decimal(selected_price, market_info['price_precision'])
                print_color(f"Selected Price: {Fore.YELLOW}{price_str}{Style.RESET_ALL}")
                break # Valid index selected

        except (ValueError, IndexError):
            print_color("Invalid index number.", color=Fore.YELLOW)
        except (EOFError, KeyboardInterrupt):
            print_color("\nInteractive limit order cancelled by user.", color=Fore.YELLOW)
            return

    # Get quantity after selecting price
    while True:
        try:
            qty_str = input(f"{Style.BRIGHT}Enter Quantity for {side.upper()} at {price_str}: {Style.RESET_ALL}").strip()
            # Basic validation before passing to the main function
            if not qty_str or decimal.Decimal(qty_str) <= 0:
                 print_color("Quantity must be a positive number.", color=Fore.YELLOW)
                 continue
            break # Got a potentially valid quantity string
        except (decimal.InvalidOperation, ValueError):
             print_color("Invalid quantity format.", color=Fore.YELLOW)
        except (EOFError, KeyboardInterrupt):
            print_color("\nInteractive limit order cancelled by user.", color=Fore.YELLOW)
            return

    # Call the main limit order function with selected price and quantity
    place_limit_order(exchange, symbol, side, qty_str, str(selected_price), market_info)


# ==============================================================================
# Main Analysis Loop & Orchestration
# ==============================================================================

def run_analysis_cycle(exchange, symbol, market_info, config):
    """Performs one cycle of fetching, processing, and displaying data."""
    print_color(f"{Fore.CYAN}# Beginning analysis cycle...", style=Style.DIM, end='\r')

    # --- Fetch Data ---
    fetched_data, data_error = fetch_market_data(exchange, symbol, config)
    analyzed_orderbook, orderbook_error = analyze_orderbook_volume(exchange, symbol, market_info, config)
    sys.stdout.write("\033[K")

    if data_error and not any(fd for fd in [fetched_data.get(k) for k in ["ticker", "indicator_ohlcv", "pivot_ohlcv"]] if fd):
        print_color(f"{Fore.YELLOW}Critical data fetch failed, skipping analysis this cycle.{Style.RESET_ALL}")
        return False, None, None # Indicate failure, return None for maps

    # --- Process Indicators ---
    print_color(f"{Fore.CYAN}# Weaving indicator patterns...", style=Style.DIM, end='\r')
    indicators_info = { # Store value and error status separately
        'sma1': {'value': None, 'error': False}, 'sma2': {'value': None, 'error': False},
        'ema1': {'value': None, 'error': False}, 'ema2': {'value': None, 'error': False},
        'momentum': {'value': None, 'error': False},
        'stoch_rsi': {'k': None, 'd': None, 'error': None} # Store k, d, and potential error message
    }
    indicator_ohlcv = fetched_data.get("indicator_ohlcv")
    if indicator_ohlcv:
        close_prices = [candle[4] for candle in indicator_ohlcv] # Index 4 is close price
        indicators_info['sma1']['value'] = calculate_sma(close_prices, config["SMA_PERIOD"])
        indicators_info['sma2']['value'] = calculate_sma(close_prices, config["SMA2_PERIOD"])
        ema1_res = calculate_ema(close_prices, config["EMA1_PERIOD"])
        ema2_res = calculate_ema(close_prices, config["EMA2_PERIOD"])
        indicators_info['ema1']['value'] = ema1_res[-1] if ema1_res else None
        indicators_info['ema2']['value'] = ema2_res[-1] if ema2_res else None
        indicators_info['momentum']['value'] = calculate_momentum(close_prices, config["MOMENTUM_PERIOD"])
        rsi_list, rsi_error = calculate_rsi_manual(close_prices, config["RSI_PERIOD"])
        if rsi_error: indicators_info['stoch_rsi']['error'] = f"RSI Error: {rsi_error}"
        elif rsi_list:
            k, d, err = calculate_stoch_rsi_manual(rsi_list, config["STOCH_K_PERIOD"], config["STOCH_D_PERIOD"])
            indicators_info['stoch_rsi']['k'], indicators_info['stoch_rsi']['d'], indicators_info['stoch_rsi']['error'] = k, d, err
        else: indicators_info['stoch_rsi']['error'] = "RSI list empty/failed"
        # Mark errors if calculation returned None
        for k in ['sma1', 'sma2', 'ema1', 'ema2', 'momentum']:
             if indicators_info[k]['value'] is None: indicators_info[k]['error'] = True
    else: # OHLCV data missing - mark all indicators as errored
        for k in indicators_info: indicators_info[k]['error'] = True if k != 'stoch_rsi' else "OHLCV Missing"

    # --- Process Pivots ---
    pivots_info = None
    pivot_ohlcv = fetched_data.get("pivot_ohlcv")
    if pivot_ohlcv and len(pivot_ohlcv) > 0:
        prev_candle = pivot_ohlcv[0] # Use the first (oldest) candle fetched for previous period
        p_high, p_low, p_close = prev_candle[2], prev_candle[3], prev_candle[4]
        pivots_info = calculate_fib_pivots(p_high, p_low, p_close)

    # --- Process Positions ---
    position_info = {'has_position': False, 'position': None, 'unrealizedPnl': None}
    position_data = fetched_data.get("positions") # Already filtered in fetch
    if position_data:
        current_pos = position_data[0] # Assume only one position per symbol for linear/inverse
        position_info['has_position'] = True
        position_info['position'] = current_pos
        try:
            pnl_raw = current_pos.get('unrealizedPnl')
            if pnl_raw is not None:
                position_info['unrealizedPnl'] = decimal.Decimal(str(pnl_raw))
        except Exception as pnl_e:
            print_color(f"# PNL parsing warning: {pnl_e}", color=Fore.YELLOW, style=Style.DIM)
            position_info['unrealizedPnl'] = None # Mark as unavailable

    sys.stdout.write("\033[K") # Clear indicator message

    # --- Aggregate Data for Display ---
    analysis_data = {
        'ticker': fetched_data.get('ticker'),
        'indicators': indicators_info,
        'pivots': pivots_info,
        'position': position_info,
        'balance': fetched_data.get('balance'),
        'orderbook': analyzed_orderbook, # Can be None
        'timestamp': analyzed_orderbook['timestamp'] if analyzed_orderbook else exchange.iso8601(exchange.milliseconds())
    }

    # --- Display ---
    ask_map, bid_map = display_combined_analysis(analysis_data, market_info, config)

    # Return True if cycle ran without critical fetch errors, plus the maps
    return not data_error or any(fetched_data.values()), ask_map, bid_map

# ==============================================================================
# Main Execution Function
# ==============================================================================

def main():
    """The main summoning ritual."""
    print_color("*" * 80, color=Fore.RED, style=Style.BRIGHT)
    print_color("   🔥 Pyrmethus's Termux Market Analyzer Activated 🔥", color=Fore.RED, style=Style.BRIGHT)
    print_color("   Use with wisdom. Market forces are potent and volatile.", color=Fore.YELLOW)
    print_color("   MARKET ORDERS CARRY SLIPPAGE RISK. LIMIT ORDERS MAY NOT FILL.", color=Fore.YELLOW)
    print_color("   YOU ARE THE MASTER OF YOUR FATE (AND YOUR TRADES).", color=Fore.YELLOW)
    print_color("*" * 80, color=Fore.RED, style=Style.BRIGHT); print("\n")

    if not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]:
        print_color("API Key/Secret scrolls are missing in .env.", color=Fore.RED, style=Style.BRIGHT)
        print_color("Ensure .env contains BYBIT_API_KEY and BYBIT_API_SECRET.", color=Fore.YELLOW)
        return

    print_color(f"{Fore.CYAN}# Binding to Bybit ({CONFIG['EXCHANGE_TYPE']}) exchange spirit...{Style.RESET_ALL}", style=Style.DIM)
    global exchange # Make exchange global to be accessible in display timestamp fallback
    try:
        exchange = ccxt.bybit({
            'apiKey': CONFIG["API_KEY"],
            'secret': CONFIG["API_SECRET"],
            'options': {
                'defaultType': CONFIG["EXCHANGE_TYPE"], # 'linear', 'inverse'
                'adjustForTimeDifference': True, # Crucial for signature timing
            },
            'timeout': CONFIG["CONNECT_TIMEOUT"],
            'enableRateLimit': True # Let ccxt handle basic rate limiting
        })
        print_color(f"{Fore.CYAN}# Testing connection conduit...{Style.RESET_ALL}", style=Style.DIM, end='\r')
        exchange.fetch_time() # Lightweight call to test connection/auth
        sys.stdout.write("\033[K")
        print_color("Connection established. Exchange spirit is responsive.", color=Fore.GREEN)
    except ccxt.AuthenticationError:
        sys.stdout.write("\033[K")
        print_color("Authentication failed! Check API Key/Secret.", color=Fore.RED, style=Style.BRIGHT)
        return
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"Failed to connect to exchange: {e}", color=Fore.RED, style=Style.BRIGHT)
        return

    # --- Symbol Selection ---
    symbol = CONFIG['SYMBOL'] # Get symbol from config
    print_color(f"{Fore.CYAN}# Attempting to use symbol from config: {symbol}{Style.RESET_ALL}")
    market_info = get_market_info(exchange, symbol)
    while not market_info:
        print_color("Failed to get market info for configured symbol.", color=Fore.YELLOW)
        try:
            symbol_input = input(f"{Style.BRIGHT}{Fore.BLUE}Enter Bybit Futures symbol (e.g., BTCUSDT): {Style.RESET_ALL}").strip().upper()
            if not symbol_input: continue
            market_info = get_market_info(exchange, symbol_input)
            if market_info:
                symbol = symbol_input # Update symbol if valid input provided
        except (EOFError, KeyboardInterrupt):
            print_color("\nSummoning interrupted.", color=Fore.YELLOW); return
        except Exception as e:
            print_color(f"Error during symbol selection: {e}", color=Fore.RED)
            time.sleep(1)

    print_color(f"Focusing vision on: {Fore.MAGENTA}{symbol}{Fore.CYAN} | "
                f"Price Precision: {market_info['price_precision']}, Amount Precision: {market_info['amount_precision']}", color=Fore.CYAN)

    # --- Main Loop ---
    print_color(f"\n{Fore.CYAN}Starting continuous analysis for {symbol}. Press Ctrl+C to banish.{Style.RESET_ALL}")
    last_critical_error_msg = ""
    ask_map, bid_map = {}, {} # Initialize order book maps

    while True:
        cycle_successful = False
        try:
            # Run the analysis cycle
            cycle_successful, ask_map, bid_map = run_analysis_cycle(exchange, symbol, market_info, CONFIG)

            # --- Action Prompt ---
            if cycle_successful:
                action_prompt = f"\n{Style.BRIGHT}{Fore.BLUE}Action ({Fore.CYAN}refresh{Fore.BLUE}/{Fore.GREEN}buy{Fore.BLUE}/{Fore.RED}sell{Fore.BLUE}/{Fore.YELLOW}exit{Fore.BLUE}): {Style.RESET_ALL}"
                action = input(action_prompt).strip().lower()

                if action in ['buy', 'sell']:
                    order_type = CONFIG['DEFAULT_ORDER_TYPE']
                    if order_type == 'limit':
                        selection_type = CONFIG['LIMIT_ORDER_SELECTION_TYPE']
                        if selection_type == 'interactive':
                            # Pass the current ask/bid maps
                            place_limit_order_interactive(exchange, symbol, action, ask_map, bid_map, market_info)
                        else: # Manual limit order price input
                            price_str = input(f"{Style.BRIGHT}Enter Limit Price for {action.upper()}: {Style.RESET_ALL}").strip()
                            qty_str = input(f"{Style.BRIGHT}Enter Quantity: {Style.RESET_ALL}").strip()
                            if price_str and qty_str:
                                place_limit_order(exchange, symbol, action, qty_str, price_str, market_info)
                            else: print_color("Price and Quantity required for limit order.", color=Fore.YELLOW)
                    else: # Market order
                        qty_str = input(f"{Style.BRIGHT}{Fore.GREEN if action == 'buy' else Fore.RED}{action.upper()} Quantity: {Style.RESET_ALL}").strip()
                        if qty_str:
                            place_market_order(exchange, symbol, action, qty_str, market_info)
                        else: print_color("Quantity required for market order.", color=Fore.YELLOW)
                    time.sleep(2) # Pause after order attempt

                elif action == 'refresh' or action == '':
                    print_color("Refreshing...", color=Fore.CYAN, style=Style.DIM)
                elif action == 'exit':
                    print_color("Dispelling the vision...", color=Fore.YELLOW); break
                else:
                    print_color("Unknown command whispered.", color=Fore.YELLOW)
                    time.sleep(1)

            # Pause before next cycle (if successful or not critically failed)
            print_color(f"{Fore.CYAN}# Pausing for {CONFIG['REFRESH_INTERVAL']}s...{Style.RESET_ALL}", style=Style.DIM, end='\r')
            time.sleep(CONFIG["REFRESH_INTERVAL"])
            sys.stdout.write("\033[K")

        except KeyboardInterrupt:
            print_color("\nBanished by user input.", color=Fore.YELLOW); break
        except ccxt.AuthenticationError: # Catch auth errors during loop
            print_color("\nAuthentication Error! Halting the ritual.", color=Fore.RED, style=Style.BRIGHT)
            break
        except Exception as e:
            # Catch-all for unexpected errors in the main loop
            current_error_msg = str(e)
            print_color(".", color=Fore.RED, end='') # Indicate loop error briefly
            # Avoid spamming the same error repeatedly
            if current_error_msg != last_critical_error_msg:
                print_color(f"\nCritical Loop Error: {e}", color=Fore.RED, style=Style.BRIGHT)
                termux_toast(f"Pyrmethus Error: {current_error_msg[:50]}", duration="long")
                last_critical_error_msg = current_error_msg
            # Wait longer after a critical loop error
            try: time.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"] * 2) # Use a relevant delay
            except Exception: time.sleep(20) # Fallback sleep


# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == '__main__':
    try:
        main()
    finally:
        # Final farewell message
        print_color("\nWizard Pyrmethus departs. May your analysis illuminate the path!", color=Fore.MAGENTA, style=Style.BRIGHT)
