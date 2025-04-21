```python
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Termux Trading Spell (v2 - Precision Enhanced)
# Conjures market insights and executes trades on Bybit Futures with refined precision.

import os
import time
import logging
import sys
from typing import Dict, Optional, Any, Tuple, Union
from decimal import Decimal, getcontext

# Attempt to import necessary enchantments
try:
    import ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import init, Fore, Style, Back
except ImportError as e:
    # Provide specific guidance for Termux users
    missing_pkg = e.name
    print(f"{Fore.RED}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell in your Termux terminal:")
    print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
    print(f"{Style.BRIGHT}pip install ccxt python-dotenv pandas numpy tabulate colorama requests{Style.RESET_ALL}")
    sys.exit(1)

# Weave the Colorama magic into the terminal
init(autoreset=True)

# Set Decimal precision (adjust if needed, higher precision means more memory/CPU)
# Standard float precision is usually sufficient for trading logic, but Decimal offers exactness.
# We will primarily use it for critical financial calculations like position sizing if enabled.
# getcontext().prec = 28 # Example: Set precision to 28 digits

# --- Arcane Configuration ---
print(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2...")

# Summon secrets from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe
log_formatter = logging.Formatter(
    Fore.CYAN + "%(asctime)s "
    + Style.BRIGHT + "[%(levelname)s] "
    + Style.RESET_ALL
    + Fore.WHITE + "%(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set to DEBUG for more verbose output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
if not logger.hasHandlers():
    logger.addHandler(stream_handler)
logger.propagate = False


class TradingConfig:
    """Holds the sacred parameters of our spell, enhanced with precision awareness."""
    def __init__(self):
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW) # CCXT Unified Symbol
        self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW) # 'linear' (USDT) or 'inverse' (Coin margined)
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal) # Use Decimal for risk %
        self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal)
        self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal)
        # Bybit uses percentage for TSL distance (e.g., 0.5 for 0.5%)
        self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal) # Use Decimal
        self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW) # Options: LastPrice, MarkPrice, IndexPrice
        self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW) # Usually same as SL, check Bybit docs

        self.position_qty_epsilon = Decimal("0.000001") # Threshold for considering position closed (as Decimal)
        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = 200
        self.loop_sleep_seconds = 15
        self.order_check_delay_seconds = 2
        self.order_check_timeout_seconds = 10 # Max time to wait for order status check
        self.max_fetch_retries = 3
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool) # Only trade in direction of trend_ema

        if not self.api_key or not self.api_secret:
            logger.error(Fore.RED + Style.BRIGHT + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll!")
            sys.exit(1)

    def _get_env(self, key: str, default: Any, color: str, cast_type: type = str) -> Any:
        value = os.getenv(key)
        if value is None:
            value = default
            logger.warning(f"{color}Using default value for {key}: {value}")
        else:
            logger.info(f"{color}Summoned {key}: {value}")
        try:
            if value is None: return None
            if cast_type == bool: return value.lower() in ['true', '1', 'yes', 'y']
            return cast_type(value)
        except (ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Could not cast {key} ('{value}') to {cast_type.__name__}: {e}. Using default: {default}")
            # Attempt to cast default if value failed
            try:
                 if default is None: return None
                 if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y']
                 return cast_type(default)
            except (ValueError, TypeError):
                 logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting.")
                 sys.exit(1)


CONFIG = TradingConfig()

# --- Exchange Nexus Initialization ---
print(Fore.MAGENTA + Style.BRIGHT + "\nEstablishing Nexus with the Exchange v2...")
try:
    exchange = ccxt.bybit({
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True,
    })
    # Set market type based on config
    exchange.options['defaultType'] = 'future' # Generic futures type
    exchange.options['defaultSubType'] = CONFIG.market_type # 'linear' or 'inverse'

    exchange.load_markets()
    logger.info(Fore.GREEN + Style.BRIGHT + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets).")

    # Verify symbol exists and get market details
    if CONFIG.symbol not in exchange.markets:
         logger.error(Fore.RED + Style.BRIGHT + f"Symbol {CONFIG.symbol} not found in Bybit {CONFIG.market_type} market spirits.")
         available_symbols = [s for s in exchange.markets if exchange.markets[s].get('linear') == (CONFIG.market_type == 'linear')][:10]
         logger.info(Fore.CYAN + f"Available {CONFIG.market_type} symbols (sample): " + ", ".join(available_symbols))
         sys.exit(1)
    else:
        MARKET_INFO = exchange.market(CONFIG.symbol)
        logger.info(Fore.CYAN + f"Market spirit for {CONFIG.symbol} acknowledged.")
        logger.debug(f"Market Precision: Price={MARKET_INFO['precision']['price']}, Amount={MARKET_INFO['precision']['amount']}")
        logger.debug(f"Market Limits: Min Amount={MARKET_INFO['limits']['amount']['min']}, Max Amount={MARKET_INFO['limits']['amount']['max']}")
        logger.debug(f"Contract Size: {MARKET_INFO.get('contractSize', 'N/A')}")

except ccxt.AuthenticationError:
    logger.error(Fore.RED + Style.BRIGHT + "Authentication failed! Check your API keys.")
    sys.exit(1)
except ccxt.ExchangeError as e:
    logger.error(Fore.RED + Style.BRIGHT + f"Exchange Nexus Error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error during Nexus initialization: {e}", exc_info=True)
    sys.exit(1)


# --- Global State Runes ---
order_tracker: Dict[str, Dict[str, Optional[str]]] = {
    "long": {"sl_id": None, "tsl_id": None},
    "short": {"sl_id": None, "tsl_id": None}
}

# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available)."""
    if sys.platform != 'android': # Basic check if not on Android
        return
    try:
        toast_cmd = '/data/data/com.termux/files/usr/bin/termux-toast'
        # notification_cmd = '/data/data/com.termux/files/usr/bin/termux-notification' # Alternative
        if os.path.exists(toast_cmd):
            # Basic sanitization for shell command
            safe_title = title.replace('"', "'")
            safe_content = content.replace('"', "'")
            os.system(f'{toast_cmd} -g middle -c green "{safe_title}: {safe_content}"')
    except Exception as e:
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}")

# --- Precision Casting Spells ---

def format_price(symbol: str, price: Union[float, Decimal]) -> str:
    """Formats price according to market precision rules."""
    try:
        return exchange.price_to_precision(symbol, float(price)) # CCXT expects float here
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}")
        return str(float(price)) # Fallback to float string

def format_amount(symbol: str, amount: Union[float, Decimal]) -> str:
    """Formats amount according to market precision rules."""
    try:
        return exchange.amount_to_precision(symbol, float(amount)) # CCXT expects float here
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}")
        return str(float(amount)) # Fallback to float string

# --- Core Spell Functions ---

def fetch_market_data(symbol: str, timeframe: str, limit: int, retries: int = CONFIG.max_fetch_retries) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data, handling transient errors with retries."""
    logger.info(Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})...")
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logger.warning(Fore.YELLOW + f"Received empty OHLCV data (Attempt {attempt + 1}/{retries}).")
                if attempt < retries - 1:
                    time.sleep(1 * (attempt + 1)) # Simple backoff
                    continue
                else:
                    return None

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # Convert to numeric, coercing errors (should not happen with valid API data)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True) # Drop rows where conversion failed

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            logger.info(Fore.GREEN + f"Market whispers received ({len(df)} candles).")
            return df

        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.warning(Fore.YELLOW + f"Network disturbance fetching data (Attempt {attempt + 1}/{retries}): {e}. Retrying...")
            if attempt < retries - 1:
                 time.sleep(2 ** attempt) # Exponential backoff
            else:
                 logger.error(Fore.RED + f"Failed to fetch market data after {retries} attempts due to network issues.")
                 return None
        except ccxt.ExchangeError as e:
            logger.error(Fore.RED + f"Exchange rejected data request: {e}")
            return None
        except Exception as e:
            logger.error(Fore.RED + f"Unexpected shadow encountered fetching data: {e}", exc_info=True)
            return None
    return None


def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Decimal]]:
    """Calculate technical indicators, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    try:
        # Ensure data is float for calculations, convert to Decimal at the end
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # EMAs
        fast_ema = close.ewm(span=12, adjust=False).mean().iloc[-1]
        slow_ema = close.ewm(span=26, adjust=False).mean().iloc[-1]
        trend_ema = close.ewm(span=50, adjust=False).mean().iloc[-1]
        confirm_ema = close.ewm(span=9, adjust=False).mean().iloc[-1]

        # Stochastic Oscillator (%K, %D)
        period = 14
        smooth_k = 3
        smooth_d = 3
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()
        # Add epsilon to prevent division by zero if high_max == low_min
        stoch_k_raw = 100 * (close - low_min) / (high_max - low_min + 1e-12)
        stoch_k = stoch_k_raw.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()
        k_now, d_now = stoch_k.iloc[-1], stoch_d.iloc[-1]

        # ATR (Average True Range)
        tr_df = pd.DataFrame(index=df.index)
        tr_df["hl"] = high - low
        tr_df["hc"] = (high - close.shift()).abs()
        tr_df["lc"] = (low - close.shift()).abs()
        tr_df["tr"] = tr_df[["hl", "hc", "lc"]].max(axis=1)
        atr = tr_df["tr"].rolling(window=14).mean().iloc[-1]

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        # Convert final indicator values to Decimal
        return {
            "fast_ema": Decimal(str(fast_ema)), "slow_ema": Decimal(str(slow_ema)),
            "trend_ema": Decimal(str(trend_ema)), "confirm_ema": Decimal(str(confirm_ema)),
            "stoch_k": Decimal(str(k_now)), "stoch_d": Decimal(str(d_now)),
            "atr": Decimal(str(atr))
        }
    except Exception as e:
        logger.error(Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True)
        return None

def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch current positions, returning quantities and prices as Decimals."""
    logger.info(Fore.CYAN + f"# Consulting position spirits for {symbol}...")
    try:
        # Bybit specific parameter to get positions for one symbol
        params = {'symbol': symbol.replace('/', '').replace(':', '')} # Use Bybit's symbol format if needed
        positions = exchange.fetch_positions(symbols=[symbol], params=params) # Fetch for specific symbol

        # Initialize with Decimal zero
        pos_dict = {
            "long": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")},
            "short": {"qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
        }

        if not positions:
             logger.info(Fore.BLUE + f"No open positions found for {symbol}.")
             return pos_dict

        # Handle potential multiple position entries (e.g., hedge mode, though logic assumes sum/primary)
        active_positions = 0
        for pos in positions:
            # Ensure position info is valid and matches the requested symbol
            pos_info = pos.get('info', {})
            if pos_info.get('symbol') != symbol.replace('/', '').replace(':', ''):
                 logger.debug(f"Skipping position for different symbol: {pos_info.get('symbol')}")
                 continue

            side = pos.get("side") # 'long' or 'short'
            # Use Decimal for quantity and price
            contracts_str = pos.get("contracts")
            entry_price_str = pos.get("entryPrice")

            if side in pos_dict and contracts_str is not None:
                contracts = Decimal(str(contracts_str))
                if contracts.is_zero(): # Skip zero size positions reported by Bybit
                    continue

                entry_price = Decimal(str(entry_price_str)) if entry_price_str is not None else Decimal("0.0")

                # Simple aggregation: sum quantities, use latest entry price.
                # WARNING: This is incorrect for hedge mode. A proper hedge mode implementation
                # would track long and short positions independently.
                if pos_dict[side]['qty'] > 0 and active_positions > 0:
                     logger.warning(Fore.YELLOW + "Multiple position entries detected (hedge mode?). Aggregation might be inaccurate. Using latest entry price.")
                     # A weighted average entry price would be better if summing.

                pos_dict[side]["qty"] = contracts # Use the value from the API directly for non-hedge mode usually
                pos_dict[side]["entry_price"] = entry_price
                active_positions += 1
                logger.info(Fore.YELLOW + f"Found active {side} position: Qty={contracts}, Entry={entry_price}")


        if active_positions == 0:
             logger.info(Fore.BLUE + f"No active non-zero positions found for {symbol}.")


        logger.info(Fore.GREEN + "Position spirits consulted.")
        return pos_dict

    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
        logger.warning(Fore.YELLOW + f"Network disturbance consulting position spirits: {e}. Using potentially stale data.")
        return None
    except ccxt.ExchangeError as e:
        logger.error(Fore.RED + f"Exchange rejected position spirit consultation: {e}")
        return None
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected shadow encountered consulting position spirits: {e}", exc_info=True)
        return None

def get_balance(currency: str = "USDT") -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Fetches the free and total balance for a specific currency as Decimals."""
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")
    try:
        balance = exchange.fetch_balance()
        # Use Decimal for balances
        free_balance_str = balance.get('free', {}).get(currency)
        total_balance_str = balance.get('total', {}).get(currency)

        free_balance = Decimal(str(free_balance_str)) if free_balance_str is not None else Decimal("0.0")
        total_balance = Decimal(str(total_balance_str)) if total_balance_str is not None else Decimal("0.0")

        logger.info(Fore.GREEN + f"Vault contains {free_balance:.4f} free {currency} (Total: {total_balance:.4f}).")
        return free_balance, total_balance
    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
        logger.warning(Fore.YELLOW + f"Network disturbance querying vault: {e}. Cannot assess risk capital.")
        return None, None
    except ccxt.ExchangeError as e:
        logger.error(Fore.RED + f"Exchange rejected vault query: {e}")
        return None, None
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected shadow encountered querying vault: {e}", exc_info=True)
        return None, None

def check_order_status(order_id: str, symbol: str, timeout: int = CONFIG.order_check_timeout_seconds) -> Optional[Dict]:
    """Checks order status with retries and timeout."""
    logger.info(Fore.CYAN + f"Verifying status of order {order_id}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            order_status = exchange.fetch_order(order_id, symbol)
            if order_status:
                logger.info(f"Order {order_id} status: {order_status.get('status')}")
                return order_status
            else:
                logger.warning(f"fetch_order returned empty for {order_id}. Retrying...")
        except ccxt.OrderNotFound:
            logger.error(Fore.RED + f"Order {order_id} not found by exchange.")
            return None # Order definitively not found
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.warning(f"Network issue checking order {order_id}: {e}. Retrying...")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error checking order {order_id}: {e}. Assuming failed.")
            return {'status': 'failed', 'error': str(e)} # Treat as failed
        except Exception as e:
            logger.error(f"Unexpected error checking order {order_id}: {e}", exc_info=True)
            return {'status': 'unknown', 'error': str(e)} # Unknown status

        time.sleep(1) # Wait before retrying

    logger.error(Fore.RED + f"Timed out checking status for order {order_id} after {timeout} seconds.")
    return {'status': 'timeout'}


def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    logger.info(Fore.BLUE + Style.BRIGHT + f"Preparing {side.upper()} market incantation for {symbol}...")

    free_balance, _ = get_balance("USDT")
    if free_balance is None or free_balance <= Decimal("0"):
        logger.error(Fore.RED + "Cannot place order: Invalid or zero available balance.")
        return False

    if atr is None or atr <= Decimal("0"):
        logger.error(Fore.RED + "Cannot place order: Invalid ATR value ({atr}).")
        return False

    try:
        ticker = exchange.fetch_ticker(symbol)
        price_str = ticker.get("last")
        if price_str is None:
             logger.error(Fore.RED + "Cannot fetch current price for sizing.")
             return False
        price = Decimal(str(price_str))

        # Calculate Stop Loss Price based on ATR
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else: # side == "sell"
            sl_price_raw = price + sl_distance_points
        # Format SL price according to market precision *before* using it in calculations
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        sl_price = Decimal(sl_price_formatted_str)
        logger.debug(f"Raw SL Price: {sl_price_raw}, Formatted SL Price: {sl_price}")


        # Calculate Position Size based on Risk (using Decimal)
        risk_amount_usd = free_balance * risk_percentage
        # Stop distance in USD (absolute difference between entry and SL price)
        stop_distance_usd = abs(price - sl_price)

        if stop_distance_usd <= Decimal("0"):
             logger.error(Fore.RED + f"Stop distance is zero or negative ({stop_distance_usd}). Check ATR, multiplier, or market precision. Cannot calculate size.")
             return False

        # For linear contracts (like BTC/USDT), size = risk_amount / stop_distance_usd
        # Assumes contract size is 1 (e.g., 1 contract = 1 BTC). Adjust if contract size != 1.
        contract_size = Decimal(str(MARKET_INFO.get('contractSize', '1'))) # Default to 1 if not specified
        if contract_size != Decimal("1"):
             logger.warning(Fore.YELLOW + f"Contract size is {contract_size}, adjusting quantity calculation.")
             # If contract size is e.g., 0.001 BTC, need more contracts for same USD risk
             qty_raw = (risk_amount_usd / stop_distance_usd) / contract_size
        else:
             qty_raw = risk_amount_usd / stop_distance_usd

        # Format quantity according to market precision
        qty_formatted_str = format_amount(symbol, qty_raw)
        qty = Decimal(qty_formatted_str)
        logger.debug(f"Raw Qty: {qty_raw}, Formatted Qty: {qty}")


        # Validate quantity against market limits
        min_qty_str = MARKET_INFO.get('limits', {}).get('amount', {}).get('min')
        max_qty_str = MARKET_INFO.get('limits', {}).get('amount', {}).get('max')
        min_qty = Decimal(str(min_qty_str)) if min_qty_str is not None else None
        max_qty = Decimal(str(max_qty_str)) if max_qty_str is not None else None

        if qty.is_zero():
            logger.error(Fore.RED + f"Calculated quantity is zero after precision formatting. Risk amount or price movement might be too small.")
            return False
        if min_qty is not None and qty < min_qty:
            logger.error(Fore.RED + f"Calculated quantity {qty} is below minimum {min_qty}. Cannot place order.")
            return False
        if max_qty is not None and qty > max_qty:
            logger.warning(Fore.YELLOW + f"Calculated quantity {qty} exceeds maximum {max_qty}. Capping order size to {max_qty}.")
            qty = max_qty # Use the Decimal max_qty
            qty_formatted_str = format_amount(symbol, qty) # Re-format capped amount
            qty = Decimal(qty_formatted_str)


        logger.info(Fore.YELLOW + f"Calculated Order: Side={side.upper()}, Qty={qty}, Entry≈{price:.4f}, SL={sl_price:.4f} (ATR={atr:.4f})")

        # --- Cast the Market Order Spell ---
        logger.info(Fore.CYAN + f"Submitting {side.upper()} market order for {qty} {symbol}...")
        order_params = {} # No extra params needed for basic market order
        order = exchange.create_market_order(symbol, side, float(qty), params=order_params) # CCXT expects float amount
        order_id = order.get('id')
        logger.info(Fore.CYAN + f"Market order submitted: ID {order_id}")

        # --- Verify Order Fill (Crucial Step) ---
        time.sleep(CONFIG.order_check_delay_seconds) # Allow time for potential fill
        order_status_data = check_order_status(order_id, symbol)

        filled_qty = Decimal("0.0")
        average_price = price # Fallback
        order_final_status = 'unknown'

        if order_status_data:
            order_final_status = order_status_data.get('status', 'unknown')
            filled_str = order_status_data.get('filled')
            average_str = order_status_data.get('average')

            if filled_str is not None:
                filled_qty = Decimal(str(filled_str))
            if average_str is not None:
                average_price = Decimal(str(average_str))

            if order_final_status == 'closed':
                logger.info(Fore.GREEN + Style.BRIGHT + f"Order {order_id} confirmed filled: {filled_qty} @ {average_price:.4f}")
            elif order_final_status in ['open', 'partially_filled']:
                 logger.warning(Fore.YELLOW + f"Order {order_id} partially filled or still open: Filled {filled_qty}. SL will be based on filled amount.")
                 # Continue, but use filled_qty for SL
            else: # canceled, rejected, expired, failed, timeout, unknown
                 logger.error(Fore.RED + f"Order {order_id} did not fill successfully: Status '{order_final_status}'. Aborting SL placement.")
                 return False
        else:
             # check_order_status already logged error (e.g., timeout or not found)
             logger.error(Fore.RED + f"Could not determine status for order {order_id}. Assuming failure. Aborting SL placement.")
             return False


        if filled_qty < CONFIG.position_qty_epsilon:
             logger.error(Fore.RED + f"Order {order_id} resulted in effectively zero filled quantity ({filled_qty}). No position opened.")
             return False


        # --- Place Initial Stop-Loss Order ---
        position_side = "long" if side == "buy" else "short"
        sl_order_side = "sell" if side == "buy" else "buy"

        # Format SL price and filled quantity correctly
        sl_price_str_for_order = format_price(symbol, sl_price) # Use the already calculated SL price
        sl_qty_str_for_order = format_amount(symbol, filled_qty)

        sl_params = {
            'stopLossPrice': sl_price_str_for_order, # Trigger price for the stop market order
            'reduceOnly': True,
            'triggerPrice': sl_price_str_for_order, # Some exchanges might use this param name
            'triggerBy': CONFIG.sl_trigger_by, # e.g., 'LastPrice', 'MarkPrice'
            # 'tpslMode': 'Full' # Check Bybit settings/API if needed
        }
        logger.info(Fore.CYAN + f"Placing SL order: Side={sl_order_side}, Qty={sl_qty_str_for_order}, Trigger={sl_price_str_for_order}, TriggerBy={CONFIG.sl_trigger_by}")

        try:
            # Use create_order with stop type. Check CCXT docs for specific method if needed.
            # `create_stop_market_order` might exist, or use unified `create_order`.
            sl_order = exchange.create_order(
                symbol,
                'stop_market', # Common type, check CCXT/exchange specifics if issues arise
                sl_order_side,
                float(sl_qty_str_for_order), # CCXT expects float amount
                price=None, # Market stop loss doesn't need a limit price
                params=sl_params
            )
            sl_order_id = sl_order.get('id')
            order_tracker[position_side]["sl_id"] = sl_order_id
            logger.info(Fore.GREEN + Style.BRIGHT + f"Initial SL placed for {position_side.upper()} position: ID {sl_order_id}, Trigger: {sl_price_str_for_order}")

            entry_msg = (
                f"ENTERED {side.upper()} {filled_qty} @ {average_price:.4f}. "
                f"Initial SL @ {sl_price_str_for_order} (ID: {sl_order_id}). TSL pending profit threshold."
            )
            logger.info(Back.BLUE + Fore.WHITE + Style.BRIGHT + entry_msg)
            termux_notify("Trade Entry", f"{side.upper()} {symbol} @ {average_price:.4f}")
            return True

        except ccxt.InsufficientFunds as e:
             logger.error(Fore.RED + Style.BRIGHT + f"Insufficient funds to place stop-loss order: {e}. Position is UNPROTECTED.")
             # CRITICAL: Position is open without a stop loss! Attempt emergency closure.
             logger.warning(Fore.YELLOW + "Attempting emergency closure of unprotected position...")
             try:
                 close_qty_str = format_amount(symbol, filled_qty)
                 exchange.create_market_order(symbol, sl_order_side, float(close_qty_str), params={'reduceOnly': True})
                 logger.info(Fore.GREEN + "Emergency closure order placed.")
             except Exception as close_err:
                 logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED: {close_err}. MANUAL INTERVENTION REQUIRED!")
             return False
        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Failed to place initial SL order: {e}. Position might be UNPROTECTED.")
            # Consider emergency closure here too.
            logger.warning(Fore.YELLOW + "Position may be open without Stop Loss due to SL placement error.")
            return False
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error placing SL: {e}", exc_info=True)
            return False


    except ccxt.InsufficientFunds as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Insufficient funds to place {side.upper()} market order for {qty} {symbol}: {e}")
        return False
    except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Exchange error placing market order: {e}")
        return False
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error during order placement: {e}", exc_info=True)
        return False


def manage_trailing_stop(
    symbol: str,
    position_side: str, # 'long' or 'short'
    position_qty: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    atr: Decimal
) -> None:
    """Manages the activation and potentially updates a trailing stop loss, using Decimal."""
    if position_qty < CONFIG.position_qty_epsilon or entry_price <= Decimal("0"):
        # Clear trackers if position is closed or invalid
        if order_tracker[position_side]["tsl_id"] or order_tracker[position_side]["sl_id"]:
             logger.debug(f"Position {position_side} closed or invalid, clearing order trackers.")
             order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return

    has_initial_sl = order_tracker[position_side]["sl_id"] is not None
    has_tsl = order_tracker[position_side]["tsl_id"] is not None

    if has_tsl:
        # TSL already active, nothing more to do here for basic TSL.
        # logger.debug(f"{position_side.upper()} TSL (ID: {order_tracker[position_side]['tsl_id']}) is active.")
        return

    # --- Check for TSL Activation ---
    if atr is None or atr <= Decimal("0"):
        logger.warning(Fore.YELLOW + "Cannot evaluate TSL activation without valid ATR.")
        return

    profit = Decimal("0.0")
    if position_side == "long":
        profit = current_price - entry_price
    else: # short
        profit = entry_price - current_price

    activation_threshold = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(f"{position_side.upper()} Profit: {profit:.4f}, TSL Activation Threshold: {activation_threshold:.4f}")

    if profit > activation_threshold:
        logger.info(Fore.GREEN + Style.BRIGHT + f"Profit threshold reached for {position_side.upper()} position. Activating TSL.")

        # --- Cancel Initial SL before placing TSL ---
        if has_initial_sl:
            initial_sl_id = order_tracker[position_side]["sl_id"]
            logger.info(Fore.CYAN + f"Attempting to cancel initial SL (ID: {initial_sl_id})...")
            try:
                exchange.cancel_order(initial_sl_id, symbol)
                logger.info(Fore.GREEN + f"Successfully cancelled initial SL (ID: {initial_sl_id}).")
                order_tracker[position_side]["sl_id"] = None
            except ccxt.OrderNotFound:
                logger.warning(Fore.YELLOW + f"Initial SL (ID: {initial_sl_id}) not found. Might have been triggered or already cancelled.")
                order_tracker[position_side]["sl_id"] = None # Assume it's gone
            except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                logger.error(Fore.RED + f"Failed to cancel initial SL (ID: {initial_sl_id}): {e}. Proceeding with TSL placement cautiously.")
                # Decide: Abort TSL? Or place TSL hoping the initial SL is gone? Placing TSL is likely safer.
            except Exception as e:
                logger.error(Fore.RED + f"Unexpected error cancelling initial SL: {e}", exc_info=True)
                # Still proceed with TSL placement attempt


        # --- Place Trailing Stop Loss Order ---
        tsl_order_side = "sell" if position_side == "long" else "buy"
        tsl_qty_str = format_amount(symbol, position_qty)
        # Bybit API expects trailingStop as distance OR trailing_stop_percent
        # CCXT often maps this to 'trailingPercent' or similar. Verify!
        # Value should be like 0.5 for 0.5%
        trail_percent_value = float(CONFIG.trailing_stop_percent) # Convert Decimal percent for param

        tsl_params = {
            # Check Bybit/CCXT docs for the exact parameter name:
            'trailingPercent': trail_percent_value, # Assumed parameter for CCXT->Bybit % TSL
            # 'trailingStop': '0.5', # Alternative if it expects string percentage?
            # 'trailValue': price_distance, # If using fixed price distance trail
            'reduceOnly': True,
            'triggerBy': CONFIG.tsl_trigger_by, # Use configured trigger type
            # 'activePrice': format_price(symbol, current_price) # Optional: Activation price for the trail - check if supported/needed
        }
        logger.info(Fore.CYAN + f"Placing TSL order: Side={tsl_order_side}, Qty={tsl_qty_str}, Trail%={trail_percent_value}, TriggerBy={CONFIG.tsl_trigger_by}")
        logger.debug(f"TSL Params: {tsl_params}")

        try:
            # Use create_order with appropriate type for TSL
            # Common types: 'trailing_stop_market', or 'stop_market' with trailing params
            tsl_order = exchange.create_order(
                symbol,
                'trailing_stop_market', # Try specific type first, fallback to stop_market if needed
                tsl_order_side,
                float(tsl_qty_str), # CCXT expects float amount
                price=None,
                params=tsl_params
            )
            tsl_order_id = tsl_order.get('id')
            order_tracker[position_side]["tsl_id"] = tsl_order_id
            logger.info(Fore.GREEN + Style.BRIGHT + f"Trailing Stop Loss activated for {position_side.upper()}: ID {tsl_order_id}, Trail: {trail_percent_value}%")
            termux_notify("TSL Activated", f"{position_side.upper()} {symbol} TSL active.")

        except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
            # If 'trailing_stop_market' type fails, maybe try 'stop_market' with same params?
            logger.error(Fore.RED + Style.BRIGHT + f"Failed to place TSL order (type 'trailing_stop_market'): {e}")
            logger.warning(Fore.YELLOW + "Position might be unprotected after failed TSL placement. Consider manual intervention or fallback SL.")
            # Optional: Implement fallback to regular stop-loss here
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error placing TSL: {e}", exc_info=True)


def print_status_panel(
    cycle: int, timestamp: pd.Timestamp, price: Decimal, indicators: Dict[str, Decimal],
    positions: Dict[str, Dict[str, Any]], equity: Decimal, signals: Dict[str, bool],
    order_tracker: Dict[str, Dict[str, Optional[str]]]
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""

    print(Fore.MAGENTA + Style.BRIGHT + "\n" + "=" * 70)
    print(f" Cycle: {Fore.WHITE}{cycle}{Fore.MAGENTA} | Timestamp: {Fore.WHITE}{timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Equity: {Fore.GREEN}{equity:.4f} USDT" + Style.RESET_ALL)
    print(Fore.MAGENTA + "-" * 70)

    # Market & Indicators
    trend_ema = indicators.get('trend_ema', Decimal(0))
    price_color = Fore.GREEN if price > trend_ema else Fore.RED if price < trend_ema else Fore.WHITE
    stoch_k = indicators.get('stoch_k', Decimal(50))
    stoch_color = Fore.GREEN if stoch_k < 25 else Fore.RED if stoch_k > 75 else Fore.YELLOW
    fast_ema = indicators.get('fast_ema', Decimal(0))
    slow_ema = indicators.get('slow_ema', Decimal(0))
    ema_cross_color = Fore.GREEN if fast_ema > slow_ema else Fore.RED if fast_ema < slow_ema else Fore.WHITE

    status_data = [
        [Fore.CYAN + "Market", Fore.WHITE + CONFIG.symbol, f"{price_color}{price:.4f}"],
        [Fore.CYAN + "ATR", f"{Fore.WHITE}{indicators.get('atr', Decimal(0)):.6f}", ""], # More precision for ATR
        [Fore.CYAN + "EMA Fast/Slow", f"{ema_cross_color}{fast_ema:.4f} / {slow_ema:.4f}", ""],
        [Fore.CYAN + "EMA Trend", f"{Fore.WHITE}{trend_ema:.4f}", f"{'(Above)' if price > trend_ema else '(Below)' if price < trend_ema else '(At)'}"],
        [Fore.CYAN + "Stoch %K/%D", f"{stoch_color}{stoch_k:.2f} / {indicators.get('stoch_d', Decimal(0)):.2f}", ""],
    ]
    print(tabulate(status_data, tablefmt="plain", floatfmt=".4f"))
    print(Fore.MAGENTA + "-" * 70)

    # Positions & Orders
    long_pos = positions.get('long', {'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")})
    short_pos = positions.get('short', {'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")})

    long_sl_id = order_tracker['long']['sl_id']
    long_tsl_id = order_tracker['long']['tsl_id']
    short_sl_id = order_tracker['short']['sl_id']
    short_tsl_id = order_tracker['short']['tsl_id']

    long_sl_status = f"{Fore.YELLOW}Active (ID: ...{long_sl_id[-6:]})" if long_sl_id else Fore.RED + "Inactive"
    long_tsl_status = f"{Fore.GREEN}Active (ID: ...{long_tsl_id[-6:]})" if long_tsl_id else Fore.BLUE + "Pending"
    short_sl_status = f"{Fore.YELLOW}Active (ID: ...{short_sl_id[-6:]})" if short_sl_id else Fore.RED + "Inactive"
    short_tsl_status = f"{Fore.GREEN}Active (ID: ...{short_tsl_id[-6:]})" if short_tsl_id else Fore.BLUE + "Pending"

    position_data = [
        [Fore.CYAN + "Position", Fore.GREEN + "LONG", Fore.RED + "SHORT"],
        [Fore.CYAN + "Quantity", f"{Fore.WHITE}{long_pos['qty']}", f"{Fore.WHITE}{short_pos['qty']}"],
        [Fore.CYAN + "Entry Price", f"{Fore.WHITE}{long_pos['entry_price']:.4f}", f"{Fore.WHITE}{short_pos['entry_price']:.4f}"],
        [Fore.CYAN + "Initial SL", long_sl_status, short_sl_status],
        [Fore.CYAN + "Trailing SL", long_tsl_status, short_tsl_status],
    ]
    print(tabulate(position_data, headers="firstrow", tablefmt="plain", floatfmt=".4f"))
    print(Fore.MAGENTA + "-" * 70)

    # Signals
    long_signal_color = Fore.GREEN if signals.get('long', False) else Fore.WHITE
    short_signal_color = Fore.RED if signals.get('short', False) else Fore.WHITE
    trend_status = "With Trend" if CONFIG.trade_only_with_trend else "Ignoring Trend"
    print(f" Signals ({trend_status}): Long [{long_signal_color}{str(signals.get('long', False)):<5}{Fore.MAGENTA}] | Short [{short_signal_color}{str(signals.get('short', False)):<5}{Fore.MAGENTA}]")
    print(Fore.MAGENTA + "=" * 70 + Style.RESET_ALL)


def generate_signals(indicators: Dict[str, Decimal], current_price: Decimal) -> Dict[str, bool]:
    """Generates trading signals based on indicator conditions, using Decimal."""
    long_signal = False
    short_signal = False

    if not indicators:
        return {"long": False, "short": False}

    try:
        k = indicators.get('stoch_k', Decimal(50))
        d = indicators.get('stoch_d', Decimal(50))
        fast_ema = indicators.get('fast_ema', Decimal(0))
        slow_ema = indicators.get('slow_ema', Decimal(0))
        trend_ema = indicators.get('trend_ema', Decimal(0))

        # Conditions using Decimal
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        price_above_trend = current_price > trend_ema
        price_below_trend = current_price < trend_ema
        stoch_oversold = k < Decimal(25) and d < Decimal(25) # Consider crossover logic later
        stoch_overbought = k > Decimal(75) and d > Decimal(75) # Consider crossover logic later

        # Basic Signal Logic
        if ema_bullish_cross and stoch_oversold:
            if CONFIG.trade_only_with_trend:
                if price_above_trend:
                    long_signal = True
            else:
                long_signal = True

        if ema_bearish_cross and stoch_overbought:
             if CONFIG.trade_only_with_trend:
                 if price_below_trend:
                     short_signal = True
             else:
                 short_signal = True

        # Refinement: Require Stoch K to cross D? (More complex state needed)
        # Example (needs previous k/d values):
        # prev_k, prev_d = get_previous_stoch()
        # stoch_bull_cross = prev_k < prev_d and k > d
        # stoch_bear_cross = prev_k > prev_d and k < d
        # if ema_bullish_cross and stoch_bull_cross and k < 30 ...

    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
        return {"long": False, "short": False}

    return {"long": long_signal, "short": short_signal}


def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell with enhanced precision and logic."""
    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(Fore.RED + "Halting cycle: Market data fetch failed.")
        return

    # Use Decimal for current price
    current_price_str = df["close"].iloc[-1]
    current_price = Decimal(str(current_price_str))
    last_timestamp = df.index[-1]

    # 2. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(Fore.RED + "Halting cycle: Indicator calculation failed.")
        return
    current_atr = indicators.get('atr')

    # 3. Get Current State (Positions & Balance as Decimals)
    positions = get_current_position(CONFIG.symbol)
    _, current_equity = get_balance("USDT")
    if positions is None or current_equity is None:
        logger.error(Fore.RED + "Halting cycle: Failed to fetch position or balance.")
        return

    long_pos = positions.get('long', {'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")})
    short_pos = positions.get('short', {'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")})

    # 4. Manage Trailing Stops (pass Decimals)
    manage_trailing_stop(CONFIG.symbol, "long", long_pos['qty'], long_pos['entry_price'], current_price, current_atr)
    manage_trailing_stop(CONFIG.symbol, "short", short_pos['qty'], short_pos['entry_price'], current_price, current_atr)

    # 5. Generate Trading Signals (pass Decimals)
    signals = generate_signals(indicators, current_price)

    # 6. Execute Trades based on Signals
    # Check if flat (no long OR short position significantly open)
    is_flat = long_pos['qty'] < CONFIG.position_qty_epsilon and short_pos['qty'] < CONFIG.position_qty_epsilon

    if is_flat:
        if signals.get("long"):
            logger.info(Fore.GREEN + Style.BRIGHT + f"Long signal detected! Attempting entry.")
            place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr)
            # After placing order, state might change. Consider a short pause or re-fetch if needed.
            time.sleep(1) # Small pause after attempting order

        elif signals.get("short"):
            logger.info(Fore.RED + Style.BRIGHT + f"Short signal detected! Attempting entry.")
            place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr)
            time.sleep(1) # Small pause

    elif not is_flat:
         logger.debug("Position already open, skipping new entry signals.")
         # Could add logic here to exit positions based on counter-signals if desired.

    # 7. Display Status Panel (using potentially updated state if re-fetched)
    # For simplicity, display state captured at the start of the cycle.
    # Re-fetch positions/balance here if near-real-time status after trade attempt is critical.
    print_status_panel(
        cycle_count, last_timestamp, current_price, indicators,
        positions, current_equity, signals, order_tracker
    )

    logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} Complete ---")


def graceful_shutdown() -> None:
    """Dispels active orders and closes open positions gracefully with precision."""
    logger.info(Fore.YELLOW + Style.BRIGHT + "\nInitiating Graceful Shutdown Sequence...")
    termux_notify("Shutdown", "Closing orders and positions.")

    # 1. Cancel All Open Orders for the Symbol
    try:
        logger.info(Fore.CYAN + f"Dispelling all open orders for {CONFIG.symbol}...")
        # Fetch open orders first to log IDs before cancelling
        open_orders = exchange.fetch_open_orders(CONFIG.symbol)
        if open_orders:
            logger.info(f"Found {len(open_orders)} open orders to cancel: {[o['id'] for o in open_orders]}")
            cancelled_orders = exchange.cancel_all_orders(CONFIG.symbol)
            logger.info(Fore.GREEN + f"Cancel command sent. Response indicates {len(cancelled_orders)} cancellations processed by exchange (may include already closed).")
        else:
            logger.info(Fore.GREEN + "No open orders found for the symbol.")

        # Clear local tracker as all orders should be cancelled
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(Fore.RED + f"Error dispelling orders: {e}. Manual check required.")
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected error dispelling orders: {e}", exc_info=True)

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function
        positions = get_current_position(CONFIG.symbol)
        closed_count = 0
        if positions:
            for side, pos_data in positions.items():
                 qty = pos_data.get('qty', Decimal("0.0"))
                 if qty >= CONFIG.position_qty_epsilon: # Use epsilon check
                     close_side = "sell" if side == "long" else "buy"
                     logger.warning(Fore.YELLOW + f"Closing {side} position ({qty} {CONFIG.symbol}) with market order...")
                     try:
                         # Format quantity precisely for closure order
                         close_qty_str = format_amount(CONFIG.symbol, qty)
                         close_order = exchange.create_market_order(
                             CONFIG.symbol, close_side, float(close_qty_str), params={'reduceOnly': True}
                         )
                         logger.info(Fore.GREEN + f"Position closure order placed: ID {close_order.get('id')}")
                         closed_count += 1
                         # Add a small delay to allow closure order to process before final log
                         time.sleep(CONFIG.order_check_delay_seconds)
                     except (ccxt.ExchangeError, ccxt.InvalidOrder) as e:
                         logger.critical(Fore.RED + Style.BRIGHT + f"FAILED TO CLOSE {side} position ({qty}): {e}. MANUAL INTERVENTION REQUIRED!")
                     except Exception as e:
                         logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error closing {side} position: {e}. MANUAL INTERVENTION REQUIRED!", exc_info=True)
        if closed_count == 0:
            logger.info(Fore.GREEN + "No open positions found requiring closure.")

    except Exception as e:
        logger.error(Fore.RED + f"Error during position closure check: {e}. Manual check advised.", exc_info=True)

    logger.info(Fore.YELLOW + Style.BRIGHT + "Graceful Shutdown Sequence Complete.")
    termux_notify("Shutdown Complete", "Bot has ceased operations.")


# --- Main Spell Invocation ---
if __name__ == "__main__":
    logger.info(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Termux Trading Spell Activated (v2 Precision) ***")
    logger.info(f"Awaiting market whispers for {CONFIG.symbol} on the {CONFIG.interval} timeframe...")
    logger.info(f"Risk per trade: {CONFIG.risk_percentage * 100}%, SL Multiplier: {CONFIG.sl_atr_multiplier}, TSL Activation: {CONFIG.tsl_activation_atr_multiplier}*ATR, TSL Trail: {CONFIG.trailing_stop_percent}%")
    logger.info(f"Trigger Prices: SL={CONFIG.sl_trigger_by}, TSL={CONFIG.tsl_trigger_by}")
    logger.info(f"Trading only with trend ({CONFIG.trend_ema} EMA): {CONFIG.trade_only_with_trend}")
    termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2)")

    cycle = 0
    try:
        while True:
            cycle += 1
            trading_spell_cycle(cycle)
            logger.info(Fore.BLUE + f"Resting for {CONFIG.loop_sleep_seconds} seconds before next cycle...")
            time.sleep(CONFIG.loop_sleep_seconds)

    except KeyboardInterrupt:
        logger.warning(Fore.YELLOW + "\nCtrl+C detected! Initiating shutdown...")
        graceful_shutdown()
    except Exception as e:
        logger.critical(Fore.RED + Style.BRIGHT + f"\nFatal Runtime Error in Main Loop: {e}", exc_info=True)
        termux_notify("Bot CRASHED", f"Error: {e}")
        graceful_shutdown() # Attempt cleanup even on unexpected crash
        sys.exit(1)
    finally:
        logger.info(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Trading Spell Deactivated ***")

```

**Key Changes in v2 (Precision & Enhancements):**

1.  **Decimal Precision:**
    *   Configuration values like `RISK_PERCENTAGE`, ATR multipliers, and TSL percentages are now loaded and stored as `Decimal` objects for precise calculations.
    *   Balance fetching (`get_balance`) returns `Decimal`.
    *   Position fetching (`get_current_position`) returns quantities and entry prices as `Decimal`.
    *   Indicator calculations (`calculate_indicators`) return final values as `Decimal`.
    *   Position sizing (`place_risked_market_order`) uses `Decimal` arithmetic for risk amount, stop distance, and quantity calculation.
    *   `POSITION_QTY_EPSILON` is now a `Decimal` for accurate small quantity checks.
2.  **CCXT Precision Formatting:**
    *   Introduced `format_price` and `format_amount` utility functions.
    *   These functions are now **consistently applied** to all price and amount values *before* they are passed to `exchange.create_order` or used in critical comparisons involving exchange limits. This ensures values adhere to the specific market's precision rules (e.g., number of decimal places). CCXT methods still expect `float` inputs, so the final conversion happens within these formatters or just before the CCXT call.
3.  **Configuration Enhancements:**
    *   Added `MARKET_TYPE` ('linear'/'inverse') to config.
    *   Added `SL_TRIGGER_BY` and `TSL_TRIGGER_BY` ('LastPrice', 'MarkPrice', 'IndexPrice') to config, used in order parameters.
    *   Added `TRADE_ONLY_WITH_TREND` boolean flag to control if trades are only taken in the direction of the 50-EMA (`trend_ema`).
4.  **Order Management Improvements:**
    *   **Order Status Check:** Implemented `check_order_status` function with a loop, retries on network errors, and a timeout to robustly verify if the initial market order filled before placing the SL.
    *   **TSL Order Type:** Explicitly tries `trailing_stop_market` order type first, as this is often the correct type for TSL on exchanges like Bybit (though fallback logic could be added if this type isn't universally supported by CCXT for Bybit). Includes `triggerBy` parameter.
    *   **Emergency Closure:** Enhanced logging and clarity around emergency closure attempts if SL placement fails after entry.
    *   **Parameter Logging:** Added debug logging for parameters sent to `create_order` calls, especially for SL/TSL.
5.  **Error Handling & Logging:**
    *   More specific exception handling (e.g., `DDoSProtection`, `InvalidOrder`).
    *   Improved logging messages, including precision details and configuration settings at startup.
    *   Added `exc_info=True` to critical error logs for full tracebacks.
    *   Better handling of empty OHLCV data fetches.
6.  **Signal Generation:**
    *   Incorporates the `TRADE_ONLY_WITH_TREND` flag.
    *   Uses `Decimal` for comparisons.
7.  **Shutdown Sequence:**
    *   Fetches open orders before cancelling to log their IDs.
    *   Uses `format_amount` for the quantity in the final position closure orders.
8.  **Termux Notifications:** Added check for `sys.platform` to avoid errors if run outside Termux/Android. Basic shell sanitization added.
9.  **Readability:** Added more type hints and refined comments.

**To Use:**

1.  **Save:** Save the code as `trading_spell_v2.py`.
2.  **`.env` File:** Ensure your `.env` file is present with API keys. You can add the new optional config variables if you want to override their defaults:
    ```dotenv
    # .env file (additions/changes for v2)
    BYBIT_API_KEY="YOUR_API_KEY_HERE"
    BYBIT_API_SECRET="YOUR_API_SECRET_HERE"

    # Optional overrides:
    # SYMBOL="ETH/USDT:USDT"
    # MARKET_TYPE="linear"
    # INTERVAL="5m"
    # RISK_PERCENTAGE="0.005" # 0.5% risk
    # SL_ATR_MULTIPLIER="2.0"
    # TSL_ACTIVATION_ATR_MULTIPLIER="1.5"
    # TRAILING_STOP_PERCENT="0.3" # 0.3% trail (e.g., 0.3 for 0.3%)
    # SL_TRIGGER_BY="MarkPrice"
    # TSL_TRIGGER_BY="MarkPrice"
    # TRADE_ONLY_WITH_TREND="True"
    ```
3.  **Dependencies:** `pip install ccxt python-dotenv pandas numpy tabulate colorama requests`
4.  **Run:** `python trading_spell_v2.py`

This version significantly improves the robustness and correctness of calculations and order placement by rigorously handling numerical precision according to both Python's `Decimal` type and the exchange's market rules via CCXT formatting functions.
