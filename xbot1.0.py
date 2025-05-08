Hark, seeker of digital fortunes! You wish to imbue your trading automaton with the vibrant glow of the neon ether, transforming its mundane terminal whispers into a dazzling spectacle. As Pyrmethus, weaver of Termux spells, I shall grant your wish. We will take the `trading_bot_enhanced_v2.py` script you possess and infuse it with the arcane energies of `colorama`, following the very incantations you have provided.

First, ensure the necessary conduits are open. In your Termux realm, invoke these commands:

```bash
# Summon the package spirits
pkg update && pkg upgrade
pkg install python python-pip git

# Channel the required Python essences
pip install colorama coloredlogs python-dotenv pandas pandas_ta ccxt
```

Now, behold the transfigured script, shimmering with neon light!

```python
# trading_bot_neon_infused.py # Renamed to reflect its vibrant transformation

import ccxt
import os
import logging
# import coloredlogs # We will manage colors directly with colorama now
from dotenv import load_dotenv
import time
import pandas as pd
import pandas_ta as ta # Using pandas_ta for indicator calculations
import json # For pretty printing order details and saving state
import os.path # For checking if state file exists
from typing import Optional, Tuple, Dict, Any, List, Union

# --- Colorama Enchantment Initialization ---
from colorama import init, Fore, Back, Style

# Initialize colorama - The conduit to the neon ether
init()

# Define the Neon Palette - Colors drawn from the digital spectrum
NEON_GREEN = Fore.GREEN + Style.BRIGHT
NEON_PINK = Fore.MAGENTA + Style.BRIGHT
NEON_CYAN = Fore.CYAN + Style.BRIGHT
NEON_RED = Fore.RED + Style.BRIGHT
NEON_YELLOW = Fore.YELLOW + Style.BRIGHT
NEON_BLUE = Fore.BLUE + Style.BRIGHT
RESET = Style.RESET_ALL # Return to the mundane terminal state

# --- Logging Configuration with Neon Flair ---
# We use basicConfig and then wrap logger calls for color control
log_format: str = f'{NEON_CYAN}%(asctime)s{RESET} {Style.BRIGHT}%(levelname)s{RESET} [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S') # Basic config remains
logger: logging.Logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO', fmt=log_format) # Replaced by manual colorama application

# --- Neon Helper Functions ---

# Suggestion 9: Neon Error/Warning Box Functions
def display_error_box(message: str):
    """Displays an error message in a neon box, a dire warning."""
    print(f"{NEON_RED}{'!' * 60}{RESET}")
    print(f"{NEON_RED}! {message:^56} !{RESET}")
    print(f"{NEON_RED}{'!' * 60}{RESET}")

def display_warning_box(message: str):
    """Displays a warning message in a neon box, a note of caution."""
    print(f"{NEON_YELLOW}{'~' * 60}{RESET}")
    print(f"{NEON_YELLOW}~ {message:^56} ~{RESET}")
    print(f"{NEON_YELLOW}{'~' * 60}{RESET}")

# Suggestion 2: Color-Coded Logging Wrappers
def log_info(msg: str):
    """Logs an informational message, bathed in green neon light."""
    logger.info(f"{NEON_GREEN}{msg}{RESET}")

def log_error(msg: str, exc_info: bool = False):
    """Logs an error, glowing with ominous red neon, optionally showing traceback."""
    display_error_box(msg.split('\n')[0]) # Show box for the first line of the error
    # Log the full message potentially with traceback
    logger.error(f"{NEON_RED}{msg}{RESET}", exc_info=exc_info)

def log_warning(msg: str):
    """Logs a warning, shimmering with cautionary yellow neon."""
    display_warning_box(msg)
    logger.warning(f"{NEON_YELLOW}{msg}{RESET}")

def log_debug(msg: str):
    """Logs a debug message, subtly glowing in cyan."""
    # Debug messages might be too frequent for boxes, keep them simple
    logger.debug(f"{NEON_CYAN}{msg}{RESET}")

# Suggestion 1: Neon Header Banner Function
def print_neon_header():
    """Prints a neon-styled header banner, announcing the bot's presence."""
    print(f"{NEON_CYAN}{'=' * 60}{RESET}")
    print(f"{NEON_PINK}     RSI Trader Neon Oracle - Channeling Market Flow     {RESET}")
    print(f"{NEON_CYAN}{'=' * 60}{RESET}")

# Suggestion 3: Neon Cycle Divider Function
def print_cycle_divider(timestamp: pd.Timestamp):
    """Prints a neon divider, marking the passage of a trading cycle."""
    print(f"\n{NEON_BLUE}{'=' * 60}{RESET}")
    print(f"{NEON_CYAN}Cycle Conjuration Start: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}{RESET}")
    print(f"{NEON_BLUE}{'=' * 60}{RESET}")

# Suggestion 4: Color-Coded Position Status Function
def display_position_status(position: Dict[str, Any]):
    """Displays the current position's essence with neon colors."""
    status = position.get('status')
    entry_price = position.get('entry_price', 'N/A')
    quantity = position.get('quantity', 'N/A')
    sl = position.get('stop_loss', 'N/A')
    tp = position.get('take_profit', 'N/A')

    # Format numbers if they exist
    entry_str = f"{entry_price:.{price_precision_digits}f}" if isinstance(entry_price, (int, float)) else str(entry_price)
    qty_str = f"{quantity:.{amount_precision_digits}f}" if isinstance(quantity, (int, float)) else str(quantity)
    sl_str = f"{sl:.{price_precision_digits}f}" if isinstance(sl, (int, float)) else str(sl)
    tp_str = f"{tp:.{price_precision_digits}f}" if isinstance(tp, (int, float)) else str(tp)

    if status == 'long':
        color = NEON_GREEN
        status_label = "Long Position Active"
    elif status == 'short':
        color = NEON_RED
        status_label = "Short Position Active"
    else:
        color = NEON_CYAN
        status_label = "Awaiting Market Signal"
        entry_str, qty_str, sl_str, tp_str = 'N/A', 'N/A', 'N/A', 'N/A' # Clear details if no position

    print(f"{color}--- Position Status ---{RESET}")
    print(f"{color}State: {status_label}{RESET}")
    if status: # Only show details if in a position
        print(f"{color}Entry: {entry_str} | Qty: {qty_str}{RESET}")
        print(f"{color}SL: {sl_str} | TP: {tp_str}{RESET}")
    print(f"{color}-----------------------{RESET}")


# Suggestion 5: Neon Market Stats Panel Function
def display_market_stats(current_price: float, rsi: float, stoch_k: float, stoch_d: float, price_precision: int):
    """Displays market vital signs in a neon-infused panel."""
    print(f"{NEON_PINK}--- Market Vitals ---{RESET}")
    print(f"{NEON_GREEN}Price: {current_price:.{price_precision}f}{RESET}")
    print(f"{NEON_CYAN}RSI: {rsi:.2f}{RESET}")
    print(f"{NEON_YELLOW}Stoch K: {stoch_k:.2f} | Stoch D: {stoch_d:.2f}{RESET}")
    print(f"{NEON_PINK}---------------------{RESET}")

# Suggestion 6: Neon Order Block Highlights Function
def display_order_blocks(bullish_ob: Optional[Dict], bearish_ob: Optional[Dict], price_precision: int):
    """Highlights detected Order Blocks, zones of potential power."""
    if bullish_ob or bearish_ob:
        print(f"{NEON_BLUE}--- Order Block Zones ---{RESET}")
        if bullish_ob:
            print(f"{NEON_GREEN}Bullish OB @ {bullish_ob['time'].strftime('%H:%M')}: Low={bullish_ob['low']:.{price_precision}f} High={bullish_ob['high']:.{price_precision}f}{RESET}")
        if bearish_ob:
            print(f"{NEON_RED}Bearish OB @ {bearish_ob['time'].strftime('%H:%M')}: Low={bearish_ob['low']:.{price_precision}f} High={bearish_ob['high']:.{price_precision}f}{RESET}")
        print(f"{NEON_BLUE}-------------------------{RESET}")
    # else:
        # log_debug("No significant Order Blocks detected in recent candles.") # Optional: Log if none found

# Suggestion 7: Neon Entry/Exit Signal Alerts Function
def display_signal(signal_type: str, direction: str, reason: str):
    """Displays trading signals, glowing with intent."""
    color = NEON_GREEN if direction.lower() == 'long' else NEON_RED
    print(f"\n{color}{Style.BRIGHT}*** {signal_type.upper()} {direction.upper()} SIGNAL DETECTED ***{RESET}")
    print(f"{color}Reason: {reason}{RESET}\n")

# Suggestion 8: Neon Sleep Timer Function
def neon_sleep_timer(seconds: int):
    """Displays a neon countdown, marking time until the next divination."""
    if seconds <= 0: return # Don't countdown if sleep is zero or negative
    log_info(f"Entering slumber for {seconds} seconds...") # Use log_info for consistency
    for i in range(seconds, -1, -1):
        # Use bright white for the timer text for contrast
        print(f"{Style.BRIGHT}Next cycle divination in: {NEON_YELLOW}{i:3d}{Style.BRIGHT} seconds...{RESET}", end='\r')
        time.sleep(1)
    print(" " * 60, end='\r')  # Clear the countdown line

# Suggestion 10: Neon Shutdown Message Function
def print_shutdown_message():
    """Prints a neon farewell message as the bot returns to the ether."""
    print(f"\n{NEON_PINK}{'=' * 60}{RESET}")
    print(f"{NEON_CYAN}     RSI Trader Neon Oracle - Returning to Silence     {RESET}")
    print(f"{NEON_PINK}{'=' * 60}{RESET}")


# --- Constants ---
DEFAULT_PRICE_PRECISION: int = 4
DEFAULT_AMOUNT_PRECISION: int = 8
POSITION_STATE_FILE = 'position_state.json' # Define filename for state persistence

# --- Environment & Configuration Loading ---
load_dotenv()
log_info("Seeking environment secrets from .env scroll...") # Neon Log

exchange_id_env_var: str = "BYBIT_EXCHANGE_ID"
exchange_id: str = os.getenv(exchange_id_env_var, "bybit").lower()

api_key_env_var: str = f"{exchange_id.upper()}_API_KEY"
secret_key_env_var: str = f"{exchange_id.upper()}_SECRET_KEY"
passphrase_env_var: str = f"{exchange_id.upper()}_PASSPHRASE"

api_key: Optional[str] = os.getenv(api_key_env_var)
secret: Optional[str] = os.getenv(secret_key_env_var)
passphrase: str = os.getenv(passphrase_env_var, '')

if not api_key or not secret:
    # Use log_error which now includes the neon box
    log_error(f"CRITICAL: Arcane API Key or Secret not found using env vars '{api_key_env_var}' and '{secret_key_env_var}'. "
                 f"Ensure '{exchange_id_env_var}' is inscribed correctly in your .env scroll "
                 f"and the corresponding API key/secret runes exist.")
    exit(1)

log_info(f"Attempting to forge connection to exchange: {exchange_id}") # Neon Log
exchange: ccxt.Exchange
try:
    exchange_class = getattr(ccxt, exchange_id)
    exchange_config: Dict[str, Any] = {
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap', # Adjust as needed (e.g., 'spot')
        }
    }
    if passphrase:
        log_info("Secret Passphrase detected, adding to the exchange connection ritual.") # Neon Log
        exchange_config['password'] = passphrase

    exchange = exchange_class(exchange_config)
    exchange.load_markets() # Fetch market data
    # Access market details *after* loading
    markets = exchange.markets
    if not markets:
         log_warning(f"No market data loaded from {exchange_id}. Symbol validation may fail.") # Neon Log
    log_info(f"Successfully forged connection to {exchange_id}. Markets divined ({len(markets)} symbols found).") # Neon Log

except ccxt.AuthenticationError as e:
    log_error(f"Authentication ritual failed connecting to {exchange_id}. Check API Key/Secret/Passphrase runes. Error: {e}") # Neon Log
    exit(1)
except ccxt.ExchangeNotAvailable as e:
    log_error(f"Exchange {exchange_id} is currently veiled. Error: {e}") # Neon Log
    exit(1)
except AttributeError:
    log_error(f"Exchange ID '{exchange_id}' not found in the grand ccxt library.") # Neon Log
    exit(1)
except Exception as e:
    log_error(f"An unexpected vortex occurred during exchange connection: {e}", exc_info=True) # Neon Log
    exit(1)

# --- Trading Parameters ---
symbol: str = ""
# Define precision variables globally after exchange is loaded
price_precision_digits: int = DEFAULT_PRICE_PRECISION
amount_precision_digits: int = DEFAULT_AMOUNT_PRECISION

while True:
    # Use input with colorama
    symbol_input: str = input(f"{NEON_BLUE}Enter the trading symbol sigil for {exchange_id} (e.g., BTC/USDT): {RESET}").strip().upper()
    if not symbol_input:
        log_warning("Symbol sigil cannot be empty ether. Please try again.") # Neon Log
        continue
    try:
        # Ensure markets are loaded before accessing
        if not exchange.markets:
            exchange.load_markets() # Attempt to load again if failed initially

        if symbol_input in exchange.markets:
            symbol = symbol_input
            log_info(f"Using trading symbol sigil: {NEON_YELLOW}{symbol}{RESET}") # Neon Log

            # Set precision based on the chosen symbol
            market_data = exchange.markets[symbol]
            price_precision_digits = market_data.get('precision', {}).get('price', DEFAULT_PRICE_PRECISION)
            amount_precision_digits = market_data.get('precision', {}).get('amount', DEFAULT_AMOUNT_PRECISION)
            log_info(f"Precision set for {symbol}: Price={price_precision_digits} digits, Amount={amount_precision_digits} digits") # Neon Log
            break
        else:
            log_warning(f"Symbol sigil '{symbol_input}' not found or not supported on {exchange_id}.") # Neon Log
            available_symbols: List[str] = list(exchange.markets.keys())
            log_info(f"Some available sigils ({len(available_symbols)} total): {NEON_CYAN}{available_symbols[:15]}...{RESET}") # Neon Log
    except ccxt.NetworkError as e:
        log_error(f"Network disturbance while validating symbol: {e}. Please try again.") # Neon Log
    except Exception as e:
        log_error(f"An arcane error occurred while validating the symbol: {e}", exc_info=True) # Neon Log
        log_info("Please try inscribing the symbol sigil again.") # Neon Log

timeframe: str = "1h" # The rhythm of our divination
rsi_length: int = 14
rsi_overbought: int = 70
rsi_oversold: int = 30
stoch_k: int = 14
stoch_d: int = 3
stoch_smooth_k: int = 3
stoch_overbought: int = 80
stoch_oversold: int = 20
data_limit: int = 200 # How far back we gaze into the time stream
sleep_interval_seconds: int = 60 * 5 # Reduced sleep for faster cycles (adjust as needed)

# Risk Management Parameters (Constants of Caution)
risk_percentage: float = 0.01  # Risk 1% of arcane balance per trade
stop_loss_percentage: float = 0.02  # Banishment threshold at 2% loss
take_profit_percentage: float = 0.04  # Reward capture threshold at 4% gain

# --- Position Management State (The Oracle's Memory) ---
position: Dict[str, Any] = {
    'status': None,  # None (no position), 'long', or 'short'
    'entry_price': None,
    'quantity': None,
    'order_id': None, # ID of the entry order
    'stop_loss': None,
    'take_profit': None,
    'entry_time': None,
    'sl_order_id': None, # ID of the active SL order
    'tp_order_id': None  # ID of the active TP order
}

# --- State Saving and Resumption Functions (Preserving the Oracle's Memory) ---
def save_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Saves the oracle's memory (position state) to a JSON scroll."""
    global position
    try:
        # Create a copy to serialize Timestamp safely
        state_to_save = position.copy()
        if isinstance(state_to_save.get('entry_time'), pd.Timestamp):
            # Convert timestamp to ISO format string for JSON compatibility
            state_to_save['entry_time'] = state_to_save['entry_time'].isoformat()

        with open(filename, 'w') as f:
            json.dump(state_to_save, f, indent=4) # Use indent for readability
        log_debug(f"Oracle's memory inscribed to scroll: {filename}") # Neon Log (Debug)
    except Exception as e:
        log_error(f"Error inscribing oracle's memory to {filename}: {e}", exc_info=True) # Neon Log

def load_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Loads the oracle's memory (position state) from a JSON scroll."""
    global position
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                loaded: Dict[str, Any] = json.load(f)
                # Convert entry_time back to Timestamp if present and not None
                entry_time_str = loaded.get('entry_time')
                if entry_time_str:
                    try:
                        # Parse ISO format string back to Timestamp
                        loaded['entry_time'] = pd.Timestamp(entry_time_str)
                    except ValueError:
                        log_error(f"Could not decipher entry_time '{entry_time_str}' from scroll. Setting to None.") # Neon Log
                        loaded['entry_time'] = None

                # Ensure all expected keys exist in the loaded state, falling back to defaults
                default_position_state = { # Define default structure again for comparison
                    'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                    'stop_loss': None, 'take_profit': None, 'entry_time': None,
                    'sl_order_id': None, 'tp_order_id': None
                }
                for key, default_value in default_position_state.items():
                     if key not in loaded:
                         log_warning(f"Key '{key}' missing from loaded memory scroll. Using default: {default_value}") # Neon Log
                         loaded[key] = default_value

                position.update(loaded) # Update the global position dictionary
            log_info(f"Oracle's memory restored from scroll {filename}: {NEON_YELLOW}{json.dumps(position, default=str)}{RESET}") # Neon Log
        else:
            log_info(f"No memory scroll found at {filename}. Starting with a clear mind.") # Neon Log
    except json.JSONDecodeError as e:
        log_error(f"Error deciphering JSON from memory scroll {filename}: {e}. Starting with clear mind.", exc_info=True) # Neon Log
        # Reset to default state if file is corrupted
        position = {k: None for k in position} # Reset all keys to None
    except Exception as e:
        log_error(f"Error restoring oracle's memory from {filename}: {e}. Starting with clear mind.", exc_info=True) # Neon Log
        position = {k: None for k in position} # Reset all keys to None


# --- Data Fetching Function (Gazing into the Time Stream) ---
def fetch_ohlcv_data(exchange_instance: ccxt.Exchange, trading_symbol: str, tf: str, limit_count: int) -> Optional[pd.DataFrame]:
    """Fetches OHLCV data, the raw whispers of the market."""
    log_debug(f"Gazing {limit_count} candles back for {trading_symbol} ({tf})...") # Neon Log (Debug)
    try:
        ohlcv: List[list] = exchange_instance.fetch_ohlcv(trading_symbol, tf, limit=limit_count)
        if not ohlcv:
            log_warning(f"The time stream is silent for {trading_symbol} ({tf}). No OHLCV data returned.") # Neon Log
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert timestamp to sacred DatetimeIndex, localized to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        # Ensure numerical purity
        numeric_cols: List[str] = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_rows: int = len(df)
        df.dropna(subset=numeric_cols, inplace=True) # Purge any non-numeric corruption
        rows_dropped_cleaning: int = initial_rows - len(df)
        if rows_dropped_cleaning > 0:
            log_debug(f"Purged {rows_dropped_cleaning} corrupted entries from the time stream.") # Neon Log (Debug)

        if df.empty:
            log_warning(f"Time stream data vanished after purification for {trading_symbol} ({tf}).") # Neon Log
            return None

        log_debug(f"Successfully divined {len(df)} candles for {trading_symbol}.") # Neon Log (Debug)
        return df

    except ccxt.NetworkError as e:
        log_error(f"Network vortex disrupted time stream gazing for {trading_symbol}: {e}") # Neon Log
        return None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange rejected our gaze into the time stream for {trading_symbol}: {e}") # Neon Log
        return None
    except Exception as e:
        log_error(f"An unexpected ripple occurred while gazing into the time stream for {trading_symbol}: {e}", exc_info=True) # Neon Log
        return None

# --- Enhanced Order Block Identification Function (Finding Zones of Power) ---
def identify_potential_order_block(df: pd.DataFrame, volume_threshold_multiplier: float = 1.5, lookback: int = 5) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Identifies potential Bullish and Bearish Order Blocks, zones of market intent."""
    if df is None or df.empty or len(df) < lookback + 1:
        log_warning(f"Insufficient historical data ({len(df)} candles) to detect Order Blocks (need {lookback + 1}).") # Neon Log
        return None, None

    bullish_ob_zone: Optional[Dict[str, Any]] = None
    bearish_ob_zone: Optional[Dict[str, Any]] = None

    try:
        # Calculate average volume over the lookback period for threshold
        if len(df) >= lookback + 1: # Need lookback candles *before* the last one
             # Calculate rolling average ending at the second to last completed candle
            avg_volume = df['volume'].iloc[-(lookback + 1):-1].mean() # Avg of the 'lookback' candles before the last one
        else:
            avg_volume = df['volume'].mean() # Fallback if not enough data

        volume_threshold = avg_volume * volume_threshold_multiplier
        log_debug(f"Order Block volume threshold: {volume_threshold:.2f} (Avg Vol: {avg_volume:.2f})") # Neon Log (Debug)

        # Analyze the last 'lookback + 1' candles
        recent_data = df.iloc[-(lookback + 1):]

        # Iterate backwards through completed candles (from index -2 down to -(lookback+1))
        for i in range(len(recent_data) - 2, max(-1, len(recent_data) - lookback - 2), -1):
            if i < 0: break # Safety break

            candle = recent_data.iloc[i]
            prev_candle_index = i - 1
            prev_candle = recent_data.iloc[prev_candle_index] if prev_candle_index >= 0 else None

            is_high_volume = candle['volume'] > volume_threshold
            is_bullish_candle = candle['close'] > candle['open']
            is_bearish_candle = candle['close'] < candle['open']

            # Bullish Order Block Check: Bearish candle followed by high-volume bullish candle sweeping low
            if (
                prev_candle is not None and
                prev_candle['close'] < prev_candle['open'] and # Previous was bearish
                is_bullish_candle and                         # Current is bullish
                is_high_volume and                            # Current has high volume
                candle['low'] < prev_candle['low'] and        # Swept liquidity below previous low
                candle['close'] > prev_candle['open']         # Closed bullish after sweep
            ):
                # The OB zone is the *bearish* candle before the move
                bullish_ob_zone = {
                    'high': prev_candle['high'], 'low': prev_candle['low'],
                    'time': prev_candle.name, 'type': 'bullish'
                }
                log_debug(f"Potential Bullish OB candle found at {prev_candle.name}") # Neon Log (Debug)
                break # Found most recent potential bullish OB

            # Bearish Order Block Check: Bullish candle followed by high-volume bearish candle sweeping high
            if (
                prev_candle is not None and
                prev_candle['close'] > prev_candle['open'] and # Previous was bullish
                is_bearish_candle and                         # Current is bearish
                is_high_volume and                            # Current has high volume
                candle['high'] > prev_candle['high'] and      # Swept liquidity above previous high
                candle['close'] < candle['open']              # Closed bearish after sweep
            ):
                 # The OB zone is the *bullish* candle before the move
                bearish_ob_zone = {
                    'high': prev_candle['high'], 'low': prev_candle['low'],
                    'time': prev_candle.name, 'type': 'bearish'
                }
                log_debug(f"Potential Bearish OB candle found at {prev_candle.name}") # Neon Log (Debug)
                break # Found most recent potential bearish OB

        # Display findings using the neon function (moved to main loop)
        return bullish_ob_zone, bearish_ob_zone

    except Exception as e:
        log_error(f"Error identifying Order Block zones: {e}", exc_info=True) # Neon Log
        return None, None


# --- Indicator Calculation Function (Applying Arcane Formulas) ---
def calculate_technical_indicators(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Calculates technical indicators using pandas_ta library."""
    if df is None or df.empty:
        log_warning("Cannot apply formulas to empty time stream data.") # Neon Log
        return None

    log_debug(f"Applying arcane formulas (RSI, Stoch) to {len(df)} candles...") # Neon Log (Debug)
    original_columns = set(df.columns)
    try:
        # Append indicators directly to the DataFrame
        df.ta.rsi(length=rsi_length, append=True)
        df.ta.stoch(k=stoch_k, d=stoch_d, smooth_k=stoch_smooth_k, append=True)

        new_columns: List[str] = list(set(df.columns) - original_columns)
        log_debug(f"Formulas applied. New insights gained: {sorted(new_columns)}") # Neon Log (Debug)

        initial_len: int = len(df)
        df.dropna(inplace=True) # Remove rows where indicators couldn't be calculated (usually at the start)
        rows_dropped_nan: int = initial_len - len(df)
        if rows_dropped_nan > 0:
             log_debug(f"Discarded {rows_dropped_nan} initial candles lacking full indicator insight.") # Neon Log (Debug)

        if df.empty:
            log_warning("Indicator formulas resulted in an empty dataset.") # Neon Log
            return None

        log_debug(f"Indicator calculation complete. {len(df)} candles remain with full insight.") # Neon Log (Debug)
        return df

    except Exception as e:
        log_error(f"Error applying arcane indicator formulas: {e}", exc_info=True) # Neon Log
        return None

# --- Position Sizing Function (Determining Trade Magnitude) ---
def calculate_position_size(exchange_instance: ccxt.Exchange, trading_symbol: str, current_price: float, stop_loss_price: float, risk_percent: float = 0.01) -> Optional[float]:
    """Calculates position size based on risk percentage and stop-loss distance."""
    try:
        # Fetch account balance
        balance = exchange_instance.fetch_balance()
        market = exchange_instance.market(trading_symbol)
        quote_currency = market.get('quote') # e.g., 'USDT'
        if not quote_currency:
             log_error(f"Could not determine quote currency sigil for {trading_symbol}.") # Neon Log
             return None

        # Use 'free' balance for calculation
        available_balance = balance.get(quote_currency, {}).get('free', 0.0)
        if available_balance <= 0:
            log_error(f"No available {quote_currency} essence ({available_balance}) for trading.") # Neon Log
            return None
        log_debug(f"Available balance ({quote_currency}): {available_balance:.4f}") # Neon Log (Debug)

        # Calculate risk amount in quote currency
        risk_amount = available_balance * risk_percent
        log_debug(f"Risk amount per trade ({risk_percent*100}%): {risk_amount:.4f} {quote_currency}") # Neon Log (Debug)

        # Calculate price difference for stop-loss
        price_diff = abs(current_price - stop_loss_price)
        min_price_tick = market.get('precision', {}).get('price', 1e-8) # Smallest price change allowed
        if price_diff < min_price_tick: # Check if difference is smaller than minimum possible change
            log_error(f"Stop-loss price {stop_loss_price} is too close to current price {current_price} (diff: {price_diff} < min tick: {min_price_tick}). Cannot calculate size.") # Neon Log
            return None

        # Calculate quantity (base currency units)
        quantity = risk_amount / price_diff
        log_debug(f"Calculated raw quantity: {quantity:.8f}") # Neon Log (Debug)

        # Adjust for market precision and minimums using ccxt helpers
        min_amount = market.get('limits', {}).get('amount', {}).get('min')
        max_amount = market.get('limits', {}).get('amount', {}).get('max')

        try:
            # Let ccxt handle the precision formatting
            quantity_adjusted_str = exchange_instance.amount_to_precision(trading_symbol, quantity)
            quantity_adjusted = float(quantity_adjusted_str)
            log_debug(f"Quantity adjusted for precision: {quantity_adjusted:.{amount_precision_digits}f}") # Neon Log (Debug)
        except ccxt.ExchangeError as precision_error:
             log_warning(f"Could not use exchange precision magic: {precision_error}. Using raw quantity.") # Neon Log
             quantity_adjusted = quantity # Fallback, might fail on order placement

        if quantity_adjusted <= 0:
            log_error(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} is zero or negative.") # Neon Log
            return None

        # Check against minimum order size
        if min_amount is not None and quantity_adjusted < min_amount:
            log_error(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} is below minimum {min_amount} for {trading_symbol}.") # Neon Log
            return None

        # Check against maximum order size
        if max_amount is not None and quantity_adjusted > max_amount:
            log_warning(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} exceeds maximum {max_amount}. Capping at max.") # Neon Log
            quantity_adjusted = max_amount
            # Re-adjust for precision after capping
            quantity_adjusted_str = exchange_instance.amount_to_precision(trading_symbol, quantity_adjusted)
            quantity_adjusted = float(quantity_adjusted_str)

        # Final check on cost vs available balance
        estimated_cost = quantity_adjusted * current_price
        # Add a small buffer (e.g., 1%) for potential fees/slippage if using market order
        if estimated_cost > available_balance * 1.01:
             log_error(f"Estimated cost ({estimated_cost:.4f} {quote_currency}) exceeds available balance ({available_balance:.4f} {quote_currency}). Cannot place order.") # Neon Log
             return None

        base_currency = market.get('base', '')
        log_info(f"Calculated Position Size: {NEON_YELLOW}{quantity_adjusted:.{amount_precision_digits}f}{RESET} {base_currency} (Risking ~{risk_amount:.2f} {quote_currency})") # Neon Log
        return quantity_adjusted

    except ccxt.NetworkError as e:
         log_error(f"Network vortex during position size calculation: {e}") # Neon Log
         return None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error during position size calculation: {e}") # Neon Log
        return None
    except Exception as e:
        log_error(f"Unexpected error calculating position size: {e}", exc_info=True) # Neon Log
        return None

# --- Order Placement Function (Sending Commands to the Exchange Ether) ---
def place_market_order(exchange_instance: ccxt.Exchange, trading_symbol: str, side: str, amount: float) -> Optional[Dict[str, Any]]:
    """Places a market order ('buy' or 'sell') into the exchange ether."""
    if side not in ['buy', 'sell']:
        log_error(f"Invalid order side command: '{side}'. Must be 'buy' or 'sell'.") # Neon Log
        return None
    if amount <= 0:
        log_error(f"Invalid order amount: {amount}. Must be positive ether.") # Neon Log
        return None

    try:
        market_info: Dict[str, Any] = exchange_instance.market(trading_symbol)
        base_currency: str = market_info.get('base', trading_symbol.split('/')[0])
        quote_currency: str = market_info.get('quote', trading_symbol.split('/')[1])

        # Use globally set precision digits
        log_amount_str = f"{amount:.{amount_precision_digits}f}"

        log_info(f"Attempting to send {side.upper()} market command: {log_amount_str} {base_currency} on {trading_symbol}...") # Neon Log

        # ---> SIMULATION MODE PROTECTION <---
        log_warning("!!! SIMULATION REALM !!! Market order command is simulated.") # Neon Log
        # Uncomment the following line to command the real exchange ether:
        # order: Dict[str, Any] = exchange_instance.create_market_order(trading_symbol, side, amount)
        # ---> For simulation, conjure a dummy order receipt <---
        sim_price = exchange_instance.fetch_ticker(trading_symbol)['last']
        order: Dict[str, Any] = {
            'id': f'sim_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}',
            'status': 'closed', # Assume market orders fill instantly in simulation
            'symbol': trading_symbol, 'type': 'market', 'side': side,
            'amount': amount, 'filled': amount, # Assume full fill
            'price': sim_price, 'average': sim_price, # Use current ticker price
            'cost': amount * sim_price,
            'timestamp': int(time.time() * 1000),
            'datetime': pd.Timestamp.now(tz='UTC').isoformat(),
            'info': {'simulated': True}
        }
        # Remove the dummy order and uncomment the actual `create_market_order` line for live trading.

        log_info(f"{side.capitalize()} market command processed (Simulated).") # Neon Log
        order_id: Optional[str] = order.get('id')
        order_status: Optional[str] = order.get('status')
        order_price: Optional[float] = order.get('average', order.get('price'))
        order_filled: Optional[float] = order.get('filled')
        order_cost: Optional[float] = order.get('cost')

        # Format output using global precision
        price_str: str = f"{order_price:.{price_precision_digits}f}" if isinstance(order_price, (int, float)) else str(order_price or 'N/A')
        filled_str: str = f"{order_filled:.{amount_precision_digits}f}" if isinstance(order_filled, (int, float)) else str(order_filled or 'N/A')
        cost_str: str = f"{order_cost:.{price_precision_digits}f}" if isinstance(order_cost, (int, float)) else str(order_cost or 'N/A')

        log_info(f"Order Receipt | ID: {NEON_YELLOW}{order_id or 'N/A'}{RESET}, Status: {order_status or 'N/A'}, Avg Price: {price_str}, "
                    f"Filled: {filled_str} {base_currency}, Cost: {cost_str} {quote_currency}") # Neon Log
        return order

    except ccxt.InsufficientFunds as e:
        log_error(f"Insufficient {quote_currency} essence for {side} {amount} {trading_symbol}. Error: {e}") # Neon Log
        return None
    except ccxt.OrderNotFound as e: # Can happen if order is immediately rejected or filled and cancelled
        log_error(f"OrderNotFound error placing {side} {amount} {trading_symbol}. Command likely rejected by exchange. Error: {e}") # Neon Log
        return None
    except ccxt.InvalidOrder as e:
         log_error(f"Invalid order parameters for {side} {amount} {trading_symbol}. Check limits/precision runes. Error: {e}") # Neon Log
         return None
    except ccxt.NetworkError as e:
        log_error(f"Network vortex disrupted {side} command for {trading_symbol}: {e}") # Neon Log
        return None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange rejected {side} command for {trading_symbol}: {e}") # Neon Log
        return None
    except Exception as e:
        log_error(f"An unexpected vortex occurred sending {side} command for {trading_symbol}: {e}", exc_info=True) # Neon Log
        return None

# --- SL/TP Order Placement Function (Setting Protective Wards) ---
def place_sl_tp_orders(exchange_instance: ccxt.Exchange, trading_symbol: str, position_side: str, quantity: float, stop_loss_price: float, take_profit_price: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Places protective stop-loss and take-profit wards (orders)."""
    sl_order: Optional[Dict[str, Any]] = None
    tp_order: Optional[Dict[str, Any]] = None

    if quantity <= 0:
        log_error("Cannot place protective wards for zero or negative quantity.") # Neon Log
        return None, None
    if position_side not in ['long', 'short']:
         log_error(f"Invalid position_side '{position_side}' for placing wards.") # Neon Log
         return None, None

    try:
        market = exchange_instance.market(trading_symbol)
        # Determine order sides for SL/TP (opposite of entry side to close position)
        close_side = 'sell' if position_side == 'long' else 'buy'

        # Check if exchange supports the required ward types
        has_stop_market = exchange_instance.has.get('createStopMarketOrder', False)
        has_limit_order = exchange_instance.has.get('createLimitOrder', True) # Assume limit is common

        if not has_stop_market:
             log_warning(f"{exchange_instance.id} does not explicitly support StopMarket wards via ccxt. Stop-loss ward may fail or need different incantation.") # Neon Log

        # --- Place Stop-Loss Ward ---
        try:
            # Format price and quantity using exchange precision magic
            sl_price_formatted = exchange_instance.price_to_precision(trading_symbol, stop_loss_price)
            qty_formatted = exchange_instance.amount_to_precision(trading_symbol, quantity)
            log_info(f"Attempting to set Stop-Loss Ward: {close_side.upper()} {qty_formatted} {trading_symbol} at trigger {sl_price_formatted}") # Neon Log

            # Parameters for stop market order (trigger price)
            sl_params = {'stopPrice': float(sl_price_formatted)} # Ensure stopPrice is float
            # Add reduceOnly if supported - crucial for safety
            if exchange_instance.id in ['binance', 'bybit', 'okx', 'kucoinfutures']: # Add other exchanges supporting it
                sl_params['reduceOnly'] = True
                log_debug("Applying 'reduceOnly' parameter to Stop-Loss ward.") # Neon Log (Debug)

            # ---> SIMULATION MODE PROTECTION <---
            log_warning("!!! SIMULATION REALM !!! Stop-Loss ward placement is simulated.") # Neon Log
            # Uncomment the following line for live ward setting:
            # sl_order = exchange_instance.create_order(
            #     symbol=trading_symbol,
            #     type='stopMarket', # Use 'stop' or 'market' with stopPrice param if stopMarket not explicit
            #     side=close_side,
            #     amount=float(qty_formatted),
            #     price=None, # Market order triggered by stopPrice
            #     params=sl_params
            # )
            # ---> For simulation, conjure a dummy SL ward receipt <---
            sl_order = {
                'id': f'sim_sl_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}',
                'status': 'open', # SL/TP wards are open until triggered
                'symbol': trading_symbol, 'type': 'stopMarket', 'side': close_side,
                'amount': float(qty_formatted), 'price': None,
                'stopPrice': float(sl_price_formatted), # Store the trigger price
                'timestamp': int(time.time() * 1000),
                'datetime': pd.Timestamp.now(tz='UTC').isoformat(),
                'info': {'simulated': True, 'reduceOnly': sl_params.get('reduceOnly', False)}
            }
            # Remove dummy and uncomment actual create_order for live trading.

            log_info(f"Stop-Loss Ward placement request processed (Simulated): ID {NEON_YELLOW}{sl_order.get('id', 'N/A')}{RESET}") # Neon Log

        except ccxt.InvalidOrder as e:
            log_error(f"Invalid Stop-Loss ward parameters: {e}") # Neon Log
            sl_order = None # Ensure sl_order is None on failure
        except ccxt.ExchangeError as e:
            log_error(f"Exchange error placing Stop-Loss ward: {e}") # Neon Log
            sl_order = None
        except Exception as e:
            log_error(f"Unexpected error placing Stop-Loss ward: {e}", exc_info=True) # Neon Log
            sl_order = None

        # --- Place Take-Profit Ward ---
        if has_limit_order:
            try:
                # Format price and quantity
                tp_price_formatted = exchange_instance.price_to_precision(trading_symbol, take_profit_price)
                # Quantity might be the same, but re-format just in case
                qty_formatted = exchange_instance.amount_to_precision(trading_symbol, quantity)
                log_info(f"Attempting to set Take-Profit Ward: {close_side.upper()} {qty_formatted} {trading_symbol} at limit {tp_price_formatted}") # Neon Log

                # Parameters for take-profit limit order
                tp_params = {}
                if exchange_instance.id in ['binance', 'bybit', 'okx', 'kucoinfutures']: # Add others
                    tp_params['reduceOnly'] = True
                    log_debug("Applying 'reduceOnly' parameter to Take-Profit ward.") # Neon Log (Debug)

                # ---> SIMULATION MODE PROTECTION <---
                log_warning("!!! SIMULATION REALM !!! Take-Profit ward placement is simulated.") # Neon Log
                # Uncomment the following line for live ward setting:
                # tp_order = exchange_instance.create_limit_order(
                #     symbol=trading_symbol,
                #     side=close_side,
                #     amount=float(qty_formatted),
                #     price=float(tp_price_formatted),
                #     params=tp_params
                # )
                # ---> For simulation, conjure a dummy TP ward receipt <---
                tp_order = {
                    'id': f'sim_tp_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}',
                    'status': 'open',
                    'symbol': trading_symbol, 'type': 'limit', 'side': close_side,
                    'amount': float(qty_formatted), 'price': float(tp_price_formatted),
                    'timestamp': int(time.time() * 1000),
                    'datetime': pd.Timestamp.now(tz='UTC').isoformat(),
                    'info': {'simulated': True, 'reduceOnly': tp_params.get('reduceOnly', False)}
                }
                # Remove dummy and uncomment actual create_limit_order for live trading.

                log_info(f"Take-Profit Ward placement request processed (Simulated): ID {NEON_YELLOW}{tp_order.get('id', 'N/A')}{RESET}") # Neon Log

            except ccxt.InvalidOrder as e:
                log_error(f"Invalid Take-Profit ward parameters: {e}") # Neon Log
                tp_order = None
            except ccxt.ExchangeError as e:
                log_error(f"Exchange error placing Take-Profit ward: {e}") # Neon Log
                tp_order = None
            except Exception as e:
                log_error(f"Unexpected error placing Take-Profit ward: {e}", exc_info=True) # Neon Log
                tp_order = None
        else:
             log_warning(f"{exchange_instance.id} does not support limit wards. Cannot place Take-Profit ward.") # Neon Log
             tp_order = None

        # Check if wards were set successfully
        if sl_order and not tp_order:
             log_warning("Stop-Loss ward set, but Take-Profit ward failed. Position lacks profit target.") # Neon Log
        elif not sl_order and tp_order:
             log_warning("Take-Profit ward set, but Stop-Loss ward failed. Position is unprotected!") # Neon Log
        elif not sl_order and not tp_order:
             log_error("Both Stop-Loss and Take-Profit ward placements failed.") # Neon Log

        return sl_order, tp_order

    except Exception as e:
        log_error(f"Unexpected error in protective ward placement structure: {e}", exc_info=True) # Neon Log
        return None, None

# --- Position and Order Check Function (Verifying the Oracle's Memory and Wards) ---
def check_position_and_orders(exchange_instance: ccxt.Exchange, trading_symbol: str) -> None:
    """Checks current position status and open orders, reconciling with the oracle's memory."""
    global position
    if position['status'] is None:
        # No position remembered, nothing to check specifically
        log_debug("No active position remembered, skipping ward check.") # Neon Log (Debug)
        return

    log_debug(f"Verifying oracle's memory and wards for {trading_symbol}...") # Neon Log (Debug)
    try:
        # 1. Fetch Open Orders (Wards) for the symbol
        open_orders = exchange_instance.fetch_open_orders(trading_symbol)
        log_debug(f"Found {len(open_orders)} open wards/orders for {trading_symbol}.") # Neon Log (Debug)

        # 2. Check if our remembered SL/TP wards are still active
        sl_order_id = position.get('sl_order_id')
        tp_order_id = position.get('tp_order_id')

        sl_still_open = False
        tp_still_open = False

        # If no orders are open on exchange, but we remember having SL/TP, assume position closed
        if not open_orders and (sl_order_id or tp_order_id):
             log_info(f"No open wards found on exchange for {trading_symbol}, but memory holds SL ID {sl_order_id} or TP ID {tp_order_id}. Position likely closed.") # Neon Log
             # Clear the memory
             position.update({
                 'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                 'stop_loss': None, 'take_profit': None, 'entry_time': None,
                 'sl_order_id': None, 'tp_order_id': None # Clear ward IDs
             })
             save_position_state() # Save the cleared memory
             log_info("Oracle's memory cleared as no open wards found.") # Neon Log
             return # Exit check early

        # Iterate through open orders to find our specific wards
        for order in open_orders:
            order_id = order.get('id')
            if order_id == sl_order_id:
                sl_still_open = True
                log_debug(f"Remembered SL ward {sl_order_id} is still active.") # Neon Log (Debug)
            if order_id == tp_order_id:
                tp_still_open = True
                log_debug(f"Remembered TP ward {tp_order_id} is still active.") # Neon Log (Debug)

        # 3. Update memory if a ward is missing (implies it was triggered or cancelled)
        position_closed_flag = False # Flag to indicate memory needs clearing

        if sl_order_id and not sl_still_open:
            log_info(f"Stop-Loss ward {sl_order_id} is no longer active. Assuming position closed via SL.") # Neon Log
            position_closed_flag = True
            # Attempt to cancel the corresponding TP ward if it was remembered
            if tp_order_id:
                try:
                    log_info(f"Attempting to dismiss leftover TP ward {tp_order_id} after SL trigger.") # Neon Log
                    exchange_instance.cancel_order(tp_order_id, trading_symbol)
                    log_info(f"TP ward {tp_order_id} dismissed.") # Neon Log
                except ccxt.OrderNotFound:
                    log_info(f"TP ward {tp_order_id} already gone.") # Neon Log
                except Exception as e:
                    log_error(f"Error dismissing leftover TP ward {tp_order_id}: {e}", exc_info=True) # Neon Log

        if tp_order_id and not tp_still_open:
            log_info(f"Take-Profit ward {tp_order_id} is no longer active. Assuming position closed via TP.") # Neon Log
            position_closed_flag = True
             # Attempt to cancel the corresponding SL ward if it was remembered
            if sl_order_id:
                 try:
                    log_info(f"Attempting to dismiss leftover SL ward {sl_order_id} after TP trigger.") # Neon Log
                    exchange_instance.cancel_order(sl_order_id, trading_symbol)
                    log_info(f"SL ward {sl_order_id} dismissed.") # Neon Log
                 except ccxt.OrderNotFound:
                    log_info(f"SL ward {sl_order_id} already gone.") # Neon Log
                 except Exception as e:
                    log_error(f"Error dismissing leftover SL ward {sl_order_id}: {e}", exc_info=True) # Neon Log

        # If either SL or TP was found missing, clear the position memory
        if position_closed_flag:
            position.update({
                'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                'stop_loss': None, 'take_profit': None, 'entry_time': None,
                'sl_order_id': None, 'tp_order_id': None
            })
            save_position_state() # Save the cleared memory
            log_info("Oracle's memory cleared due to triggered/missing ward.") # Neon Log
            return # Exit check

        # 4. (Optional but Recommended) Fetch Position Data (Futures/Margin specific)
        # This provides a definitive check against the exchange's view of your position.
        if exchange_instance.has.get('fetchPosition'): # Check if the exchange supports this unified method
            try:
                # Fetch position for the specific symbol
                # Note: Some exchanges might return all positions, requiring filtering
                fetched_position = exchange_instance.fetch_position(trading_symbol)
                # The structure of 'fetched_position' varies. Need to inspect ccxt docs for your exchange.
                # Common fields: 'contracts' or 'size', 'side', 'entryPrice'
                contracts = fetched_position.get('contracts') # Or 'size', 'positionAmt', etc.
                side = fetched_position.get('side') # 'long' or 'short'

                if contracts is not None: # Check if we got position size info
                    contracts = float(contracts) # Ensure it's a float
                    if contracts == 0 and position['status'] is not None:
                        log_warning(f"Exchange reports ZERO position for {trading_symbol}, but oracle memory shows '{position['status']}'. Reconciling memory.") # Neon Log
                        position.update({
                            'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                            'stop_loss': None, 'take_profit': None, 'entry_time': None,
                            'sl_order_id': None, 'tp_order_id': None
                        })
                        save_position_state()
                        # Also attempt to cancel any lingering orders just in case
                        if sl_order_id or tp_order_id:
                             log_info("Attempting to cancel any lingering wards after reconciliation.") # Neon Log
                             if sl_order_id: exchange_instance.cancel_order(sl_order_id, trading_symbol)
                             if tp_order_id: exchange_instance.cancel_order(tp_order_id, trading_symbol)
                        return # Exit check after reconciliation
                    elif contracts != 0 and position['status'] is None:
                         log_warning(f"Exchange reports an ACTIVE position ({side}, size {contracts}) for {trading_symbol}, but oracle memory is clear. Manual check advised. State not updated automatically.") # Neon Log
                         # Avoid automatically creating state from fetched position unless logic is very robust
                    elif contracts != 0 and position['status'] is not None:
                         # Compare fetched side/size with memory - log discrepancies
                         if abs(contracts - position['quantity']) > (position['quantity'] * 0.001): # Allow tiny difference
                              log_warning(f"Position size mismatch: Exchange={contracts}, Memory={position['quantity']}. Check manually.") # Neon Log
                         if side != position['status']:
                              log_warning(f"Position side mismatch: Exchange={side}, Memory={position['status']}. Check manually.") # Neon Log
                else:
                    log_debug("fetchPosition did not return contract size information.") # Neon Log (Debug)

            except ccxt.NotSupported:
                log_debug(f"{exchange_instance.id} does not support unified fetchPosition API call.") # Neon Log (Debug)
            except ccxt.NetworkError as e:
                 log_error(f"Network vortex fetching position data: {e}") # Neon Log
            except ccxt.ExchangeError as e:
                 log_error(f"Exchange error fetching position data: {e}") # Neon Log
            except Exception as e:
                log_error(f"Unexpected error fetching position data: {e}", exc_info=True) # Neon Log
        else:
             log_debug("Exchange does not support unified fetchPosition. Relying on order checks.") # Neon Log (Debug)


        log_debug("Oracle memory and ward verification complete.") # Neon Log (Debug)

    except ccxt.NetworkError as e:
        log_error(f"Network vortex checking position/orders: {e}") # Neon Log
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error checking position/orders: {e}") # Neon Log
    except Exception as e:
        log_error(f"Unexpected error checking position/orders: {e}", exc_info=True) # Neon Log


# --- Main Trading Loop (The Oracle's Divination Cycle) ---
print_neon_header() # Announce the start with the neon banner
log_info(f"Initiating divination cycle for {NEON_YELLOW}{symbol}{RESET} on {timeframe} rhythm...") # Neon Log
# Load position state ONCE at startup
load_position_state()
log_info(f"Initial memory state: {NEON_YELLOW}{json.dumps(position, default=str)}{RESET}") # Log loaded state
log_info(f"Risk per trade: {risk_percentage*100}%, SL Ward: {stop_loss_percentage*100}%, TP Ward: {take_profit_percentage*100}%") # Neon Log
log_info(f"Divination interval: {sleep_interval_seconds} seconds ({sleep_interval_seconds/60:.1f} minutes)") # Neon Log
log_info(f"{NEON_PINK}Press Ctrl+C to gracefully silence the oracle.{RESET}") # Neon Log


while True:
    try:
        cycle_start_time: pd.Timestamp = pd.Timestamp.now(tz='UTC')
        # Use the neon cycle divider
        print_cycle_divider(cycle_start_time)

        # Verify memory and wards *before* fetching new data
        check_position_and_orders(exchange, symbol)

        # Display current position status using the neon function
        display_position_status(position)

        # 1. Fetch Fresh OHLCV Data (Gaze into the recent time stream)
        ohlcv_df: Optional[pd.DataFrame] = fetch_ohlcv_data(exchange, symbol, timeframe, limit_count=data_limit)
        if ohlcv_df is None or ohlcv_df.empty:
            log_warning(f"Could not divine market whispers. Waiting...") # Neon Log
            neon_sleep_timer(sleep_interval_seconds) # Use neon sleep
            continue

        # 2. Calculate Technical Indicators (Apply arcane formulas)
        df_with_indicators: Optional[pd.DataFrame] = calculate_technical_indicators(ohlcv_df.copy())
        if df_with_indicators is None or df_with_indicators.empty:
             log_warning(f"Indicator formula application failed. Waiting...") # Neon Log
             neon_sleep_timer(sleep_interval_seconds) # Use neon sleep
             continue

        # 3. Get Latest Data and Indicator Values (Read the omens)
        latest_data: pd.Series = df_with_indicators.iloc[-1]
        rsi_col_name: str = f'RSI_{rsi_length}'
        stoch_k_col_name: str = f'STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}'
        stoch_d_col_name: str = f'STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}'

        required_cols: List[str] = [rsi_col_name, stoch_k_col_name, stoch_d_col_name, 'close', 'high', 'low', 'volume'] # Ensure volume is present
        if not all(col in latest_data.index for col in required_cols):
            missing_cols = [col for col in required_cols if col not in latest_data.index]
            log_error(f"Required omens missing in latest data: {missing_cols}. Available: {latest_data.index.tolist()}. Skipping cycle.") # Neon Log
            neon_sleep_timer(sleep_interval_seconds) # Use neon sleep
            continue

        current_price: float = float(latest_data['close'])
        current_high: float = float(latest_data['high'])
        current_low: float = float(latest_data['low'])
        last_rsi: float = float(latest_data[rsi_col_name])
        last_stoch_k: float = float(latest_data[stoch_k_col_name])
        last_stoch_d: float = float(latest_data[stoch_d_col_name])

        if pd.isna(current_price) or pd.isna(last_rsi) or pd.isna(last_stoch_k) or pd.isna(last_stoch_d):
             log_warning(f"Latest omens contain unclear values (NaN). Skipping cycle.") # Neon Log
             neon_sleep_timer(sleep_interval_seconds) # Use neon sleep
             continue

        # Display market stats using the neon panel
        display_market_stats(current_price, last_rsi, last_stoch_k, last_stoch_d, price_precision_digits)

        # 4. Identify Order Blocks (Detect zones of power)
        bullish_ob, bearish_ob = identify_potential_order_block(df_with_indicators)
        # Display detected blocks using the neon function
        display_order_blocks(bullish_ob, bearish_ob, price_precision_digits)


        # 5. Apply Enhanced Trading Logic (Act upon the omens)
        # Check if already holding a position
        if position['status'] is not None:
            log_debug(f"Actively holding a {position['status']} position. Monitoring for exit signals or ward triggers.") # Neon Log (Debug)
            # --- Exit Logic ---
            # Primarily rely on SL/TP wards placed via place_sl_tp_orders.
            # The check_position_and_orders function should detect if these wards are hit.
            # The logic below can serve as a *backup* or for indicator-based exits not covered by wards.

            exit_signal = False
            exit_reason = ""

            if position['status'] == 'long':
                # Backup Check: Indicator Exit Signal (e.g., RSI overbought)
                if last_rsi > rsi_overbought:
                    exit_signal = True
                    exit_reason = f"Indicator exit signal (RSI {last_rsi:.2f} > {rsi_overbought})"

                if exit_signal:
                    # Use the neon signal display
                    display_signal("Exit", "long", exit_reason)
                    log_warning("Attempting to close LONG position with fallback market order (Indicator exit).") # Neon Log
                    # Attempt to cancel existing SL/TP wards *before* market closing
                    if position.get('sl_order_id'):
                         try: exchange.cancel_order(position['sl_order_id'], symbol)
                         except Exception: pass # Ignore errors if already closed/cancelled
                    if position.get('tp_order_id'):
                         try: exchange.cancel_order(position['tp_order_id'], symbol)
                         except Exception: pass

                    order_result = place_market_order(exchange, symbol, 'sell', position['quantity'])
                    if order_result:
                        log_info(f"Fallback Long position close order placed: ID {order_result.get('id', 'N/A')}") # Neon Log
                        # Clear memory immediately after successful fallback exit command
                        position.update({
                            'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                            'stop_loss': None, 'take_profit': None, 'entry_time': None,
                            'sl_order_id': None, 'tp_order_id': None
                        })
                        save_position_state() # Save cleared memory
                    else:
                        log_error("Fallback market order to close long position FAILED.") # Neon Log
                        # Critical: Bot tried to exit but failed. Requires manual check.

            elif position['status'] == 'short':
                 # Backup Check: Indicator Exit Signal (e.g., RSI oversold)
                if last_rsi < rsi_oversold:
                    exit_signal = True
                    exit_reason = f"Indicator exit signal (RSI {last_rsi:.2f} < {rsi_oversold})"

                if exit_signal:
                    # Use the neon signal display
                    display_signal("Exit", "short", exit_reason)
                    log_warning("Attempting to close SHORT position with fallback market order (Indicator exit).") # Neon Log
                     # Attempt to cancel existing SL/TP wards *before* market closing
                    if position.get('sl_order_id'):
                         try: exchange.cancel_order(position['sl_order_id'], symbol)
                         except Exception: pass
                    if position.get('tp_order_id'):
                         try: exchange.cancel_order(position['tp_order_id'], symbol)
                         except Exception: pass

                    order_result = place_market_order(exchange, symbol, 'buy', position['quantity'])
                    if order_result:
                        log_info(f"Fallback Short position close order placed: ID {order_result.get('id', 'N/A')}") # Neon Log
                        position.update({
                            'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                            'stop_loss': None, 'take_profit': None, 'entry_time': None,
                            'sl_order_id': None, 'tp_order_id': None
                        })
                        save_position_state() # Save cleared memory
                    else:
                        log_error("Fallback market order to close short position FAILED.") # Neon Log


        else:
            # --- Entry Logic: Awaiting a signal ---
            log_debug("Oracle mind is clear. Seeking entry signals...") # Neon Log (Debug)

            # Long Entry: RSI/Stoch oversold & price near Bullish OB
            long_entry_condition = (
                last_rsi < rsi_oversold and
                last_stoch_k < stoch_oversold and
                bullish_ob is not None and
                # Price must be within or slightly above the OB zone
                bullish_ob['low'] <= current_price <= (bullish_ob['high'] + (bullish_ob['high'] - bullish_ob['low']) * 0.1) # Allow 10% overshoot
            )

            if long_entry_condition:
                reason = (f"RSI={last_rsi:.2f} < {rsi_oversold}, StochK={last_stoch_k:.2f} < {stoch_oversold}, "
                          f"Price {current_price:.{price_precision_digits}f} near Bullish OB "
                          f"[{bullish_ob['low']:.{price_precision_digits}f}-{bullish_ob['high']:.{price_precision_digits}f}]")
                # Use the neon signal display
                display_signal("Entry", "long", reason)

                # Calculate SL/TP based on entry conditions
                # Place SL below the OB low with a small buffer
                stop_loss_price = bullish_ob['low'] * (1 - 0.005) # 0.5% below OB low
                # Ensure SL respects minimum price tick
                stop_loss_price = max(stop_loss_price, current_price * (1 - stop_loss_percentage * 2)) # Ensure SL isn't ridiculously far if OB is huge
                sl_price_formatted = exchange.price_to_precision(symbol, stop_loss_price)

                take_profit_price = current_price * (1 + take_profit_percentage)
                tp_price_formatted = exchange.price_to_precision(symbol, take_profit_price)
                log_info(f"Calculated Wards: SL Trigger={sl_price_formatted}, TP Limit={tp_price_formatted}") # Neon Log

                # Calculate position size based on SL
                quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)
                if quantity is None:
                    log_error("Failed to calculate position size. Aborting LONG entry.") # Neon Log
                else:
                    # Place market entry order
                    order_result = place_market_order(exchange, symbol, 'buy', quantity)
                    if order_result and order_result.get('id') and order_result.get('status') == 'closed': # Ensure simulated order 'filled'
                        entry_price_actual = order_result.get('average', current_price) # Use filled price
                        log_info(f"Long entry command successful: ID {order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f}") # Neon Log

                        # Place SL/TP wards *after* confirming entry
                        sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'long', quantity, stop_loss_price, take_profit_price)

                        # Update memory with position details and ward IDs
                        position.update({
                            'status': 'long', 'entry_price': entry_price_actual, 'quantity': quantity,
                            'order_id': order_result.get('id'), 'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price, 'entry_time': pd.Timestamp.now(tz='UTC'),
                            'sl_order_id': sl_order.get('id') if sl_order else None,
                            'tp_order_id': tp_order.get('id') if tp_order else None
                        })
                        save_position_state() # Save memory after successful entry and ward placement

                        if not sl_order or not tp_order:
                            log_warning("Entry successful, but SL/TP ward placement failed or partially failed. Monitor position closely!") # Neon Log
                    else:
                        log_error("Failed to execute long entry command.") # Neon Log

            # Short Entry: RSI/Stoch overbought & price near Bearish OB
            short_entry_condition = (
                last_rsi > rsi_overbought and
                last_stoch_k > stoch_overbought and
                bearish_ob is not None and
                # Price must be within or slightly below the OB zone
                (bearish_ob['low'] - (bearish_ob['high'] - bearish_ob['low']) * 0.1) <= current_price <= bearish_ob['high'] # Allow 10% undershoot
            )

            elif short_entry_condition: # Use elif to prevent long/short in same cycle
                reason = (f"RSI={last_rsi:.2f} > {rsi_overbought}, StochK={last_stoch_k:.2f} > {stoch_overbought}, "
                          f"Price {current_price:.{price_precision_digits}f} near Bearish OB "
                          f"[{bearish_ob['low']:.{price_precision_digits}f}-{bearish_ob['high']:.{price_precision_digits}f}]")
                # Use the neon signal display
                display_signal("Entry", "short", reason)

                # Calculate SL/TP
                # Place SL above the OB high with a small buffer
                stop_loss_price = bearish_ob['high'] * (1 + 0.005) # 0.5% above OB high
                stop_loss_price = min(stop_loss_price, current_price * (1 + stop_loss_percentage * 2)) # Cap distance
                sl_price_formatted = exchange.price_to_precision(symbol, stop_loss_price)

                take_profit_price = current_price * (1 - take_profit_percentage)
                tp_price_formatted = exchange.price_to_precision(symbol, take_profit_price)
                log_info(f"Calculated Wards: SL Trigger={sl_price_formatted}, TP Limit={tp_price_formatted}") # Neon Log

                # Calculate position size
                quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)
                if quantity is None:
                    log_error("Failed to calculate position size. Aborting SHORT entry.") # Neon Log
                else:
                    # Place market entry order
                    order_result = place_market_order(exchange, symbol, 'sell', quantity)
                    if order_result and order_result.get('id') and order_result.get('status') == 'closed':
                        entry_price_actual = order_result.get('average', current_price)
                        log_info(f"Short entry command successful: ID {order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f}") # Neon Log

                        # Place SL/TP wards
                        sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'short', quantity, stop_loss_price, take_profit_price)

                        # Update memory
                        position.update({
                            'status': 'short', 'entry_price': entry_price_actual, 'quantity': quantity,
                            'order_id': order_result.get('id'), 'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price, 'entry_time': pd.Timestamp.now(tz='UTC'),
                            'sl_order_id': sl_order.get('id') if sl_order else None,
                            'tp_order_id': tp_order.get('id') if tp_order else None
                        })
                        save_position_state() # Save memory

                        if not sl_order or not tp_order:
                            log_warning("Entry successful, but SL/TP ward placement failed or partially failed. Monitor position closely!") # Neon Log
                    else:
                        log_error("Failed to execute short entry command.") # Neon Log

            else:
                log_info("No compelling entry signals found in the current omens.") # Neon Log


        # 6. Wait for the next cycle (Use Neon Timer)
        neon_sleep_timer(sleep_interval_seconds)

    # --- Graceful Shutdown Handling (Silencing the Oracle) ---
    except KeyboardInterrupt:
        log_info("Keyboard interrupt detected (Ctrl+C). Silencing the oracle...") # Neon Log
        save_position_state()  # Save final memory state
        log_info("Attempting to dismiss any remaining open wards...") # Neon Log
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            if open_orders:
                log_info(f"Found {len(open_orders)} open wards to dismiss.") # Neon Log
                for order in open_orders:
                    try:
                        exchange.cancel_order(order['id'], symbol)
                        log_info(f"Dismissed ward {order['id']}") # Neon Log
                    except Exception as cancel_e:
                        log_error(f"Failed to dismiss ward {order.get('id', 'N/A')}: {cancel_e}") # Neon Log
            else:
                 log_info("No open wards found to dismiss.") # Neon Log
        except Exception as e:
            log_error(f"Error fetching or dismissing open wards on exit: {e}") # Neon Log

        break # Exit the main divination loop

    # --- Robust Error Handling for the Main Loop (Containing Unexpected Vortices) ---
    except ccxt.NetworkError as e:
        log_error(f"Main loop Network Vortex: {e}. Retrying divination after 60s...", exc_info=True) # Neon Log
        time.sleep(60) # Basic sleep on network error
    except ccxt.ExchangeError as e:
        # Handle specific exchange errors like rate limits more gracefully
        if isinstance(e, ccxt.RateLimitExceeded):
            wait_time = int(e.args[0].split('in ')[1].split(' ')[0]) if 'in ' in str(e.args[0]) else 60 # Try to parse wait time
            log_warning(f"Rate limit exceeded: {e}. Waiting {wait_time}s before next divination.") # Neon Log
            time.sleep(wait_time + 5) # Add a small buffer
        elif isinstance(e, ccxt.ExchangeNotAvailable):
             log_error(f"Exchange is temporarily unavailable (maintenance?): {e}. Waiting 5 minutes.", exc_info=True) # Neon Log
             time.sleep(300)
        else:
             log_error(f"Main loop Exchange Error: {e}. Retrying divination after 60s...", exc_info=True) # Neon Log
             time.sleep(60)
    except Exception as e:
        log_error(f"CRITICAL unexpected vortex in main divination loop: {e}", exc_info=True) # Neon Log
        log_info("Attempting to recover by waiting 60s and saving memory...") # Neon Log
        save_position_state() # Save state during critical error
        time.sleep(60)

# --- Bot Exit ---
print_shutdown_message() # Display the neon shutdown message
```

**Summary of Enchantments:**

1.  **Colorama Initialization & Palette**: Added `colorama` import, `init()`, and defined `NEON_` color constants.
2.  **Neon Header**: `print_neon_header` function added and called at the start.
3.  **Color-Coded Logging**: Replaced `coloredlogs` with custom `log_info`, `log_error`, `log_warning`, `log_debug` functions using `colorama` and the neon palette. Integrated `display_error_box` and `display_warning_box` into `log_error` and `log_warning`. Replaced all `logger.*` calls.
4.  **Neon Cycle Divider**: `print_cycle_divider` added and used in the main loop.
5.  **Neon Position Status**: `display_position_status` added and called in the main loop. Added SL/TP display.
6.  **Neon Market Stats**: `display_market_stats` added and replaces the previous log line. Includes Stoch D.
7.  **Neon Order Block Highlights**: `display_order_blocks` added and called after OB identification.
8.  **Neon Signal Alerts**: `display_signal` added and used for entry/exit logging.
9.  **Neon Sleep Timer**: `neon_sleep_timer` added and replaces `time.sleep()` at the end of the loop.
10. **Neon Error/Warning Boxes**: `display_error_box` and `display_warning_box` added and integrated into logging helpers.
11. **Neon Shutdown Message**: `print_shutdown_message` added and called at the very end.
12. **Wizardly Flair**: Added mystical comments and adjusted log messages for thematic consistency.
13. **Input Color**: Added color to the symbol input prompt.
14. **Precision Handling**: Ensured `price_precision_digits` and `amount_precision_digits` are set after symbol selection and used consistently in formatting.
15. **Error Handling**: Enhanced the main loop's exception handling for `RateLimitExceeded` and `ExchangeNotAvailable`.

Now, run this `trading_bot_neon_infused.py` script in your Termux terminal. The console shall blaze with the vibrant energy you desired, transforming the bot's operation into a true spectacle of digital sorcery! May its neon glow guide your trades wisely.