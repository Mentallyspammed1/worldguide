import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal, getcontext
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

import ccxt
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

getcontext().prec = 10
init(autoreset=True)
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago")
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
RESET = Style.RESET_ALL

os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return msg.replace(API_KEY, "***").replace(API_SECRET, "***")


def load_config(filepath: str) -> dict:
    default_config = {
        "interval": "15",
        "analysis_interval": 30,
        "retry_delay": 5,
        "momentum_period": 10,
        "momentum_ma_short": 12,
        "momentum_ma_long": 26,
        "volume_ma_period": 20,
        "atr_period": 14,
        "trend_strength_threshold": 0.4,
        "sideways_atr_multiplier": 1.5,
        "indicators": {
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "divergence": True,
            "stoch_rsi": True,
            "rsi": True,
            "macd": False,
            "bollinger_bands": True,
        },
        "weight_sets": {
            "low_volatility": {
                "ema_alignment": 0.4,
                "momentum": 0.3,
                "volume_confirmation": 0.2,
                "divergence": 0.1,
                "stoch_rsi": 0.7,
                "rsi": 0.0,
                "macd": 0.0,
                "bollinger_bands": 0.0,
            }
        },
        "rsi_period": 14,
        "bollinger_bands_period": 20,
        "bollinger_bands_std_dev": 2,
        "orderbook_limit": 100,
        "orderbook_cluster_threshold": 1000,
    }

    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config


CONFIG = load_config(CONFIG_FILE)


def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.5,
        status_forcelist=RETRY_ERROR_CODES,
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def setup_logger(symbol: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(
        SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        SensitiveFormatter(
            NEON_BLUE + "%(asctime)s" + RESET + " - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(stream_handler)

    return logger


def bybit_request(
    method: str,
    endpoint: str,
    params: dict | None = None,
    logger: logging.Logger | None = None,
) -> dict | None:
    session = create_session()
    try:
        params = params or {}
        timestamp = str(int(datetime.now(TIMEZONE).timestamp() * 1000))
        signature_params = params.copy()
        signature_params["timestamp"] = timestamp
        param_str = "&".join(
            f"{key}={value}" for key, value in sorted(signature_params.items())
        )
        signature = hmac.new(
            API_SECRET.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json",
        }
        url = f"{BASE_URL}{endpoint}"
        request_kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "timeout": 10,
        }
        if method == "GET":
            request_kwargs["params"] = params
        elif method == "POST":
            request_kwargs["json"] = params

        response = session.request(**request_kwargs)
        response.raise_for_status()
        json_response = response.json()
        if json_response and json_response.get("retCode") == 0:
            return json_response
        else:
            if logger:
                logger.error(
                    f"{NEON_RED}Bybit API error: {json_response.get('retCode')} - {json_response.get('retMsg')}{RESET}"
                )
            return None

    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f"{NEON_RED}API request failed: {e}{RESET}")
        return None


def fetch_current_price(symbol: str, logger: logging.Logger) -> Decimal | None:
    endpoint = "/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    response = bybit_request("GET", endpoint, params, logger)
    if not response or response.get("retCode") != 0 or not response.get("result"):
        logger.error(f"{NEON_RED}Failed to fetch ticker data: {response}{RESET}")
        return None
    tickers = response["result"].get("list", [])
    for ticker in tickers:
        if ticker.get("symbol") == symbol:
            last_price_str = ticker.get("lastPrice")
            if not last_price_str:
                logger.error(f"{NEON_RED}No lastPrice in ticker data{RESET}")
                return None
            try:
                return Decimal(last_price_str)
            except Exception as e:
                logger.error(f"{NEON_RED}Error parsing last price: {e}{RESET}")
                return None
    logger.error(f"{NEON_RED}Symbol {symbol} not found in ticker data{RESET}")
    return None


def fetch_klines(
    symbol: str, interval: str, limit: int = 200, logger: logging.Logger = None
) -> list:  # Changed return type annotation
    try:
        endpoint = "/v5/market/kline"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "category": "linear",
        }
        response = bybit_request("GET", endpoint, params, logger)
        if (
            response
            and response.get("retCode") == 0
            and response.get("result")
            and response["result"].get("list")
        ):
            data = response["result"]["list"]
            # Minimal processing without pandas
            processed_data = []
            columns = ["start_time", "open", "high", "low", "close", "volume", "turnover"]
            for row in data:
                entry = {}
                for i, col in enumerate(columns):
                    if i < len(row):
                        try:
                            if col == 'start_time':
                                entry[col] = int(row[i])  # Keep as timestamp
                            else:
                                entry[col] = float(row[i])  # Convert numeric fields to float
                        except (ValueError, TypeError):
                            entry[col] = None  # Handle potential conversion errors
                    else:
                        entry[col] = None  # Handle missing columns
                processed_data.append(entry)

            # Ensure required columns exist, even if None
            required_cols = {"close", "high", "low", "volume"}
            if not processed_data or not all(col in processed_data[0] for col in required_cols):
                 if logger:
                     logger.error(f"{NEON_RED}Kline data missing required columns after processing.{RESET}")
                 return []

            return processed_data  # Return list of dictionaries

        if logger:
            logger.error(f"{NEON_RED}Failed to fetch klines: {response}{RESET}")
        return []
    except (requests.exceptions.RequestException, KeyError, ValueError, TypeError) as e:
        if logger:
            logger.exception(f"{NEON_RED}Error fetching klines: {e}{RESET}")
        return []


def fetch_orderbook(symbol: str, limit: int, logger: logging.Logger) -> dict | None:
    retry_count = 0
    exchange = ccxt.bybit()
    while retry_count <= MAX_API_RETRIES:
        try:
            orderbook_data = exchange.fetch_order_book(symbol, limit=limit)
            if orderbook_data:
                # Convert prices and sizes to floats if needed (ccxt usually does this)
                orderbook_data['bids'] = [[float(p), float(s)] for p, s in orderbook_data['bids']]
                orderbook_data['asks'] = [[float(p), float(s)] for p, s in orderbook_data['asks']]
                return orderbook_data
            else:
                logger.error(
                    f"{NEON_RED}Failed to fetch orderbook data from ccxt (empty response). Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}"
                )
                time.sleep(RETRY_DELAY_SECONDS)
                retry_count += 1
        except ccxt.ExchangeError as e:
            if "orderbook_limit" in str(e).lower():
                logger.warning(
                    f"{NEON_YELLOW}ccxt ExchangeError: orderbook_limit issue. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}"
                )
            else:
                logger.error(
                    f"{NEON_RED}ccxt ExchangeError fetching orderbook: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}"
                )
            time.sleep(RETRY_DELAY_SECONDS)
            retry_count += 1
        except ccxt.NetworkError as e:
            logger.error(
                f"{NEON_RED}ccxt NetworkError fetching orderbook: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}"
            )
            time.sleep(RETRY_DELAY_SECONDS)
            retry_count += 1
        except Exception as e:
            logger.exception(
                f"{NEON_RED}Unexpected error fetching orderbook with ccxt: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}"
            )
            time.sleep(RETRY_DELAY_SECONDS)
            retry_count += 1

    logger.error(
        f"{NEON_RED}Max retries reached for orderbook fetch using ccxt. Aborting.{RESET}"
    )
    return None

# --- Helper functions for list-based calculations ---


def rolling_mean(data: list[float], window: int) -> list[float | None]:
    if not data or window <= 0 or window > len(data):
        return [None] * len(data)
    [0.0] * len(data)
    means = [None] * len(data)
    current_sum = 0.0
    for i, x in enumerate(data):
        current_sum += x
        if i >= window:
            current_sum -= data[i - window]
        if i >= window - 1:
            means[i] = current_sum / window
    return means


def rolling_std(data: list[float], window: int) -> list[float | None]:
    if not data or window <= 1 or window > len(data):
        return [None] * len(data)
    means = rolling_mean(data, window)
    stds = [None] * len(data)
    current_sum_sq = 0.0
    for i, x in enumerate(data):
        current_sum_sq += x * x
        if i >= window:
            current_sum_sq -= data[i - window] * data[i - window]
        if i >= window - 1 and means[i] is not None:
            variance = (current_sum_sq / window) - (means[i] ** 2)
            # Handle potential floating point inaccuracies leading to small negative variance
            stds[i] = max(0, variance) ** 0.5
    return stds


def rolling_max(data: list[float], window: int) -> list[float | None]:
    if not data or window <= 0 or window > len(data):
        return [None] * len(data)
    max_vals = [None] * len(data)
    from collections import deque
    dq = deque()
    for i, x in enumerate(data):
        while dq and data[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - window:
            dq.popleft()
        if i >= window - 1:
            max_vals[i] = data[dq[0]]
    return max_vals


def rolling_min(data: list[float], window: int) -> list[float | None]:
    if not data or window <= 0 or window > len(data):
        return [None] * len(data)
    min_vals = [None] * len(data)
    from collections import deque
    dq = deque()
    for i, x in enumerate(data):
        while dq and data[dq[-1]] >= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - window:
            dq.popleft()
        if i >= window - 1:
            min_vals[i] = data[dq[0]]
    return min_vals


def ewm_mean(data: list[float], span: int, adjust: bool = False) -> list[float | None]:
    if not data or span <= 0:
        return [None] * len(data)
    alpha = 2 / (span + 1)
    ewma = [None] * len(data)
    if not adjust:
        ewma[0] = data[0]
        for i in range(1, len(data)):
             ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i - 1]
    else:  # Adjust=True is more complex without pandas/numpy built-ins
         # Simplified approach for adjust=False as it's common
        ewma[0] = data[0]
        for i in range(1, len(data)):
             ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i - 1]

    return ewma


def list_diff(data: list[float], n: int = 1) -> list[float | None]:
    if n <= 0 or n >= len(data):
        return [None] * len(data)
    diffs = [None] * n
    for i in range(n, len(data)):
        diffs.append(data[i] - data[i - n])
    return diffs


def list_cumsum(data: list[float | None]) -> list[float | None]:
    cumsum = [None] * len(data)
    current_sum = 0.0
    valid_start = False
    for i, x in enumerate(data):
        if x is not None:
           if not valid_start:
               current_sum = x
               valid_start = True
           else:
               current_sum += x
           cumsum[i] = current_sum
        else:
            cumsum[i] = None  # Propagate None if input is None
    return cumsum

# --- TradingAnalyzer rewrite without pandas ---


class TradingAnalyzer:
    def __init__(
        self,
        kline_data: list[dict],  # Expect list of dicts now
        logger: logging.Logger,
        config: dict,
        symbol: str,
        interval: str,
    ) -> None:
        self.data = kline_data
        self.logger = logger
        self.levels = {}
        self.fib_levels = {}
        self.config = config
        self.signal = None
        self.weight_sets = config["weight_sets"]
        self.user_defined_weights = self.weight_sets["low_volatility"]
        self.symbol = symbol
        self.interval = interval
        # Extract columns into lists for easier processing
        self.close = [d.get('close') for d in self.data]
        self.high = [d.get('high') for d in self.data]
        self.low = [d.get('low') for d in self.data]
        self.volume = [d.get('volume') for d in self.data]
        self.start_time = [d.get('start_time') for d in self.data]

    def _get_last_valid(self, data_list):
        """Helper to get the last non-None value."""
        for x in reversed(data_list):
            if x is not None:
                return x
        return None

    def _get_tail(self, data_list, n=3):
        """Helper to get the last n non-None values."""
        valid_values = [x for x in data_list if x is not None]
        return valid_values[-n:] if len(valid_values) >= n else [None] * n

    def calculate_sma(self, window: int) -> list[float | None]:
        if not all(isinstance(x, (int, float)) for x in self.close if x is not None):
             self.logger.error(f"{NEON_RED}Invalid data type in 'close' for SMA.{RESET}")
             return [None] * len(self.close)
        return rolling_mean(self.close, window)

    def calculate_momentum(self, period: int = 10) -> list[float | None]:
         if not self.close or len(self.close) <= period:
             return [None] * len(self.close)
         momentum = [None] * period
         for i in range(period, len(self.close)):
             prev_close = self.close[i - period]
             curr_close = self.close[i]
             if prev_close is not None and curr_close is not None and prev_close != 0:
                 mom = ((curr_close - prev_close) / prev_close) * 100
                 momentum.append(mom)
             else:
                 momentum.append(None)
         return momentum

    def calculate_cci(self, window: int = 20, constant: float = 0.015) -> list[float | None]:
         if len(self.close) < window: return [None] * len(self.close)
         typical_price = [((h + l + c) / 3) if h is not None and l is not None and c is not None else None
                          for h, l, c in zip(self.high, self.low, self.close, strict=False)]
         if not any(tp is not None for tp in typical_price): return [None] * len(self.close)

         sma_typical_price = rolling_mean(typical_price, window)
         mean_deviation = [None] * len(typical_price)
         cci = [None] * len(typical_price)

         for i in range(window - 1, len(typical_price)):
             window_data = [tp for tp in typical_price[i - window + 1:i + 1] if tp is not None]
             if len(window_data) == window and sma_typical_price[i] is not None:  # Ensure full window and valid mean
                dev_sum = sum(abs(x - sma_typical_price[i]) for x in window_data)
                mean_deviation[i] = dev_sum / window
                if mean_deviation[i] is not None and mean_deviation[i] != 0:
                    cci_val = (typical_price[i] - sma_typical_price[i]) / (constant * mean_deviation[i])
                    cci[i] = cci_val
         return cci

    def calculate_williams_r(self, window: int = 14) -> list[float | None]:
         if len(self.close) < window: return [None] * len(self.close)
         highest_high = rolling_max(self.high, window)
         lowest_low = rolling_min(self.low, window)
         wr = [None] * len(self.close)
         for i in range(window - 1, len(self.close)):
             hh = highest_high[i]
             ll = lowest_low[i]
             cl = self.close[i]
             if hh is not None and ll is not None and cl is not None and (hh - ll) != 0:
                 wr[i] = (hh - cl) / (hh - ll) * -100
         return wr

    def calculate_mfi(self, window: int = 14) -> list[float | None]:
         if len(self.close) < window + 1: return [None] * len(self.close)
         typical_price = [((h + l + c) / 3) if h is not None and l is not None and c is not None else None
                          for h, l, c in zip(self.high, self.low, self.close, strict=False)]
         if not any(tp is not None for tp in typical_price): return [None] * len(self.close)

         raw_money_flow = [(tp * v) if tp is not None and v is not None else None
                           for tp, v in zip(typical_price, self.volume, strict=False)]

         positive_flow = [0.0] * len(typical_price)
         negative_flow = [0.0] * len(typical_price)

         for i in range(1, len(typical_price)):
             if typical_price[i] is not None and typical_price[i - 1] is not None and raw_money_flow[i] is not None:
                 if typical_price[i] > typical_price[i - 1]:
                     positive_flow[i] = raw_money_flow[i]
                 elif typical_price[i] < typical_price[i - 1]:
                     negative_flow[i] = raw_money_flow[i]
                 # else: both remain 0

         positive_mf_sum = rolling_mean([p or 0 for p in positive_flow], window)  # Treat None as 0 for sum
         negative_mf_sum = rolling_mean([n or 0 for n in negative_flow], window)  # Treat None as 0 for sum

         mfi = [None] * len(typical_price)
         for i in range(window, len(typical_price)):
              pos_sum = positive_mf_sum[i] * window if positive_mf_sum[i] is not None else 0  # rolling_mean gives avg, multiply by window
              neg_sum = negative_mf_sum[i] * window if negative_mf_sum[i] is not None else 0

              if neg_sum != 0:
                   money_ratio = pos_sum / neg_sum
                   mfi[i] = 100 - (100 / (1 + money_ratio))
              elif pos_sum != 0:  # Handle case where negative flow is zero but positive is not
                   mfi[i] = 100.0
              else:  # Both sums are zero
                   mfi[i] = 50.0  # Or some other neutral value, depends on definition

         return mfi

    def calculate_fibonacci_retracement(
        self, high: float, low: float, current_price: float
    ) -> dict[str, float]:
        try:
            diff = high - low
            if diff == 0:
                return {}
            fib_levels = {
                "Fib 23.6%": high - diff * 0.236,
                "Fib 38.2%": high - diff * 0.382,
                "Fib 50.0%": high - diff * 0.5,
                "Fib 61.8%": high - diff * 0.618,
                "Fib 78.6%": high - diff * 0.786,
                "Fib 88.6%": high - diff * 0.886,
                "Fib 94.1%": high - diff * 0.941,
            }
            self.levels = {"Support": {}, "Resistance": {}}
            for label, value in fib_levels.items():
                if value < current_price:
                    self.levels["Support"][label] = value
                elif value > current_price:
                    self.levels["Resistance"][label] = value
            self.fib_levels = fib_levels
            return self.fib_levels
        except ZeroDivisionError:
            self.logger.error(
                f"{NEON_RED}Fibonacci calculation error: Division by zero.{RESET}"
            )
            return {}
        except Exception as e:
            self.logger.exception(f"{NEON_RED}Fibonacci calculation error: {e}{RESET}")
            return {}

    def calculate_pivot_points(self, high: float, low: float, close: float) -> None:
        try:
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            self.levels.update({
                "pivot": pivot,
                "r1": r1,
                "s1": s1,
                "r2": r2,
                "s2": s2,
                "r3": r3,
                "s3": s3,
            })
        except Exception as e:
            self.logger.exception(
                f"{NEON_RED}Pivot point calculation error: {e}{RESET}"
            )
            self.levels = {}  # Reset levels on error

    def find_nearest_levels(
        self, current_price: float, num_levels: int = 5
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        try:
            support_levels = []
            resistance_levels = []

            def process_level(label, value) -> None:
                 # Ensure value is a valid number before comparison
                if isinstance(value, (int, float, Decimal)):
                    value_float = float(value)  # Convert Decimal to float for comparison
                    if value_float < current_price:
                        support_levels.append((label, value_float))
                    elif value_float > current_price:
                        resistance_levels.append((label, value_float))

            # Iterate through potentially nested levels dictionary
            for label, value in self.levels.items():
                if isinstance(value, dict):
                    for sub_label, sub_value in value.items():
                        process_level(f"{label} ({sub_label})", sub_value)
                else:
                     process_level(label, value)

            # Sort by absolute difference from current price
            support_levels.sort(key=lambda x: abs(x[1] - current_price))
            resistance_levels.sort(key=lambda x: abs(x[1] - current_price))

            nearest_supports = support_levels[:num_levels]
            nearest_resistances = resistance_levels[:num_levels]
            return nearest_supports, nearest_resistances
        except (KeyError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Error finding nearest levels: {e}{RESET}")
            return [], []
        except Exception as e:
            self.logger.exception(
                f"{NEON_RED}Unexpected error finding nearest levels: {e}{RESET}"
            )
            return [], []

    def calculate_atr(self, window: int = 20) -> list[float | None]:
         if len(self.close) < window + 1: return [None] * len(self.close)

         tr = [None] * len(self.close)
         for i in range(1, len(self.close)):
             h, l, pc = self.high[i], self.low[i], self.close[i - 1]
             if h is not None and l is not None and pc is not None:
                 tr1 = h - l
                 tr2 = abs(h - pc)
                 tr3 = abs(l - pc)
                 tr[i] = max(tr1, tr2, tr3)

         # Simple Moving Average of TR for ATR
         atr = rolling_mean(tr, window)  # Use helper for rolling mean
         return atr

    def calculate_rsi(self, window: int = 14) -> list[float | None]:
        if len(self.close) <= window: return [None] * len(self.close)
        deltas = list_diff(self.close, 1)

        gains = [max(d, 0) if d is not None else 0 for d in deltas]
        losses = [abs(min(d, 0)) if d is not None else 0 for d in deltas]

        avg_gain = rolling_mean(gains, window)  # Use simple moving average initially
        avg_loss = rolling_mean(losses, window)  # Use simple moving average initially

        # Smoothed average (like Wilder's) - more accurate RSI
        # Initialize first valid avg_gain/avg_loss
        first_valid_idx = -1
        for i in range(window, len(self.close)):
            if avg_gain[i] is not None and avg_loss[i] is not None:
                first_valid_idx = i
                break

        if first_valid_idx != -1:
             # Apply smoothing for subsequent periods
             for i in range(first_valid_idx + 1, len(self.close)):
                 current_gain = gains[i] if gains[i] is not None else 0
                 current_loss = losses[i] if losses[i] is not None else 0
                 prev_avg_gain = avg_gain[i - 1] if avg_gain[i - 1] is not None else 0
                 prev_avg_loss = avg_loss[i - 1] if avg_loss[i - 1] is not None else 0

                 avg_gain[i] = ((prev_avg_gain * (window - 1)) + current_gain) / window
                 avg_loss[i] = ((prev_avg_loss * (window - 1)) + current_loss) / window

        rsi = [None] * len(self.close)
        for i in range(window, len(self.close)):
            ag = avg_gain[i]
            al = avg_loss[i]
            if ag is not None and al is not None:
                if al == 0:
                    rsi[i] = 100.0
                else:
                    rs = ag / al
                    rsi[i] = 100 - (100 / (1 + rs))
        return rsi

    def calculate_stoch_rsi(
        self,
        rsi_window: int = 14,
        stoch_window: int = 12,
        k_window: int = 3,
        d_window: int = 3,
    ) -> dict[str, list[float | None]]:
        if len(self.close) < rsi_window + stoch_window + max(k_window, d_window):
             return {"stoch_rsi": [], "k": [], "d": []}

        rsi = self.calculate_rsi(window=rsi_window)
        if not any(r is not None for r in rsi):
             return {"stoch_rsi": [], "k": [], "d": []}

        min_rsi = rolling_min(rsi, stoch_window)
        max_rsi = rolling_max(rsi, stoch_window)

        stoch_rsi = [None] * len(rsi)
        for i in range(len(rsi)):
             if rsi[i] is not None and min_rsi[i] is not None and max_rsi[i] is not None:
                 denominator = max_rsi[i] - min_rsi[i]
                 if denominator != 0:
                     stoch_rsi[i] = (rsi[i] - min_rsi[i]) / denominator
                 else:  # Handle case where max = min (flat RSI)
                      stoch_rsi[i] = 0.0  # Or 0.5, depending on convention

        # Calculate %K (SMA of StochRSI)
        k_line = rolling_mean(stoch_rsi, k_window)

        # Calculate %D (SMA of %K)
        d_line = rolling_mean(k_line, d_window)

        return {"stoch_rsi": stoch_rsi, "k": k_line, "d": d_line}

    def calculate_momentum_ma(self) -> dict:
        momentum = list_diff(self.close, self.config["momentum_period"])
        momentum_ma_short = rolling_mean(momentum, self.config["momentum_ma_short"])
        momentum_ma_long = rolling_mean(momentum, self.config["momentum_ma_long"])
        volume_ma = rolling_mean(self.volume, self.config["volume_ma_period"])

        return {
            "momentum": momentum,
            "momentum_ma_short": momentum_ma_short,
            "momentum_ma_long": momentum_ma_long,
            "volume_ma": volume_ma,
        }

    def calculate_macd(self) -> dict[str, list[float | None]]:
        if len(self.close) < 26:  # Need enough data for longest EMA
            return {"macd": [], "signal": [], "histogram": []}

        ema_short = ewm_mean(self.close, span=12, adjust=False)
        ema_long = ewm_mean(self.close, span=26, adjust=False)

        macd = [None] * len(self.close)
        for i in range(len(self.close)):
            if ema_short[i] is not None and ema_long[i] is not None:
                macd[i] = ema_short[i] - ema_long[i]

        signal = ewm_mean(macd, span=9, adjust=False)

        histogram = [None] * len(self.close)
        for i in range(len(self.close)):
             if macd[i] is not None and signal[i] is not None:
                  histogram[i] = macd[i] - signal[i]

        return {"macd": macd, "signal": signal, "histogram": histogram}

    # MACD Divergence detection is complex without libraries like pandas/numpy for slicing/comparison easily.
    # Skipping a direct list-based equivalent for brevity unless specifically required.
    def detect_macd_divergence(self, macd_data: dict) -> str | None:
         prices = self.close
         macd_histogram = macd_data.get("histogram", [])

         if len(prices) < 3 or len(macd_histogram) < 3:
             return None

         # Get last two valid values (handling potential Nones)
         last_price = self._get_last_valid(prices)
         prev_price = None
         for p in reversed(prices[:-1]):
             if p is not None:
                 prev_price = p
                 break

         last_hist = self._get_last_valid(macd_histogram)
         prev_hist = None
         for h in reversed(macd_histogram[:-1]):
              if h is not None:
                  prev_hist = h
                  break

         if None in [last_price, prev_price, last_hist, prev_hist]:
              return None  # Not enough valid recent data

         # Bullish Divergence: Lower low in price, higher low in histogram
         # Simplified: Check if price fell AND histogram rose
         if prev_price > last_price and prev_hist < last_hist:
              # Add more checks (e.g., ensure histogram was negative) for robustness
              return "bullish"
         # Bearish Divergence: Higher high in price, lower high in histogram
         # Simplified: Check if price rose AND histogram fell
         elif prev_price < last_price and prev_hist > last_hist:
              # Add more checks (e.g., ensure histogram was positive) for robustness
              return "bearish"

         return None

    def calculate_ema(self, window: int) -> list[float | None]:
        if not all(isinstance(x, (int, float)) for x in self.close if x is not None):
            self.logger.error(f"{NEON_RED}Invalid data type in 'close' for EMA.{RESET}")
            return [None] * len(self.close)
        return ewm_mean(self.close, span=window, adjust=False)

    def determine_trend_momentum(self) -> dict:
        if len(self.close) < self.config["momentum_ma_long"]:  # Need enough data for longest MA
            return {"trend": "Insufficient Data", "strength": 0}

        atr_list = self.calculate_atr()
        last_atr = self._get_last_valid(atr_list)

        if last_atr is None or last_atr == 0:
            self.logger.warning(f"{NEON_YELLOW}ATR is zero or unavailable, cannot calculate trend strength.{RESET}")
            return {"trend": "Neutral", "strength": 0}

        momentum_data = self.calculate_momentum_ma()
        last_short_ma = self._get_last_valid(momentum_data["momentum_ma_short"])
        last_long_ma = self._get_last_valid(momentum_data["momentum_ma_long"])

        if last_short_ma is None or last_long_ma is None:
            return {"trend": "Insufficient Data", "strength": 0}

        if last_short_ma > last_long_ma:
            trend = "Uptrend"
        elif last_short_ma < last_long_ma:
            trend = "Downtrend"
        else:
            trend = "Neutral"

        trend_strength = abs(last_short_ma - last_long_ma) / last_atr if last_atr else 0
        return {"trend": trend, "strength": trend_strength}

    def calculate_adx(self, window: int = 14) -> float | None:
        if len(self.close) < window + 1: return None  # Need at least window+1 periods

        high, low, close = self.high, self.low, self.close

        tr = [None] * len(close)
        pdm = [0.0] * len(close)  # +DM
        ndm = [0.0] * len(close)  # -DM

        for i in range(1, len(close)):
            h, l, pc = high[i], low[i], close[i - 1]
            ph, pl = high[i - 1], low[i - 1]  # Previous high/low

            if h is not None and l is not None and pc is not None:
                tr1 = h - l
                tr2 = abs(h - pc)
                tr3 = abs(l - pc)
                tr[i] = max(tr1, tr2, tr3)

            if h is not None and ph is not None and l is not None and pl is not None:
                up_move = h - ph
                down_move = pl - l

                if up_move > down_move and up_move > 0:
                    pdm[i] = up_move
                if down_move > up_move and down_move > 0:
                    ndm[i] = down_move

        # Use Wilder's smoothing (similar to EMA)
        def wilder_smooth(data, period):
            smoothed = [None] * len(data)
            first_sum = sum(x for x in data[1:period + 1] if x is not None)  # Sum first 'period' values (skip index 0)
            if first_sum is None: return smoothed  # Cannot start if initial data is missing

            smoothed[period] = first_sum  # Initial value is just the sum
            for i in range(period + 1, len(data)):
                 prev_smooth = smoothed[i - 1]
                 current_val = data[i] if data[i] is not None else 0
                 if prev_smooth is not None:
                      smoothed[i] = prev_smooth - (prev_smooth / period) + current_val
                 else:  # If previous calculation failed, cannot continue
                     return smoothed
            return smoothed

        atr_smoothed = wilder_smooth(tr, window)
        pdm_smoothed = wilder_smooth(pdm, window)
        ndm_smoothed = wilder_smooth(ndm, window)

        pdi = [None] * len(close)  # +DI
        ndi = [None] * len(close)  # -DI
        dx = [None] * len(close)

        for i in range(window, len(close)):
             atr_s = atr_smoothed[i]
             pdm_s = pdm_smoothed[i]
             ndm_s = ndm_smoothed[i]

             if atr_s is not None and atr_s != 0 and pdm_s is not None and ndm_s is not None:
                 pdi[i] = 100 * (pdm_s / atr_s)
                 ndi[i] = 100 * (ndm_s / atr_s)
                 di_sum = pdi[i] + ndi[i]
                 if di_sum != 0:
                     dx[i] = 100 * (abs(pdi[i] - ndi[i]) / di_sum)

        # ADX is the smoothed average of DX
        adx_smoothed = wilder_smooth(dx, window)

        return self._get_last_valid(adx_smoothed)

    def calculate_obv(self) -> list[float | None]:
        if len(self.close) < 2: return [None] * len(self.close)
        obv = [0.0] * len(self.close)  # Start OBV at 0 or first volume? Usually 0.
        if self.volume[0] is not None:  # Initialize based on convention
            obv[0] = 0  # Or self.volume[0] or None - starting point varies. Let's use 0.

        for i in range(1, len(self.close)):
            c, pc, v = self.close[i], self.close[i - 1], self.volume[i]
            prev_obv = obv[i - 1]

            if c is not None and pc is not None and v is not None and prev_obv is not None:
                if c > pc:
                    obv[i] = prev_obv + v
                elif c < pc:
                    obv[i] = prev_obv - v
                else:
                    obv[i] = prev_obv
            elif prev_obv is not None:  # Propagate previous OBV if current data is missing
                 obv[i] = prev_obv
            else:  # If previous OBV is None, cannot calculate current
                 obv[i] = None

        return obv

    def calculate_adi(self) -> list[float | None]:
         if len(self.close) < 1: return []
         adi = [None] * len(self.close)
         current_adi = 0.0
         initialized = False

         for i in range(len(self.close)):
             c, h, l, v = self.close[i], self.high[i], self.low[i], self.volume[i]

             if None in [c, h, l, v]:
                 adi[i] = current_adi if initialized else None
                 continue  # Skip calculation if data is missing

             denominator = h - l
             if denominator == 0:  # If high == low, multiplier is 0 or undefined
                 mfm = 0.0  # Money Flow Multiplier
             else:
                 mfm = ((c - l) - (h - c)) / denominator

             mfv = mfm * v  # Money Flow Volume

             if not initialized:
                 current_adi = mfv
                 initialized = True
             else:
                 current_adi += mfv
             adi[i] = current_adi

         return adi

    def calculate_psar(self, acceleration=0.02, max_acceleration=0.2) -> list[float | None]:
        if len(self.close) < 2: return [None] * len(self.close)

        psar_list = [None] * len(self.close)
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = acceleration
        ep = self.high[0]  # Extreme point for uptrend
        sar = self.low[0]  # Initial SAR often set to low[0] for uptrend start

        if self.close[1] < self.close[0]:  # If second candle closes lower, assume initial downtrend
             trend = -1
             ep = self.low[0]
             sar = self.high[0]

        psar_list[0] = sar  # Set initial SAR

        for i in range(1, len(self.close)):
             prev_sar = psar_list[i - 1]
             if prev_sar is None: continue  # Cannot calculate if previous SAR missing

             current_high = self.high[i]
             current_low = self.low[i]
             if current_high is None or current_low is None:
                 psar_list[i] = prev_sar  # Propagate SAR if data missing
                 continue

             if trend == 1:  # --- Uptrend Calculation ---
                 sar = prev_sar + af * (ep - prev_sar)
                 # Ensure SAR does not move above the previous two lows
                 sar = min(sar, self.low[i - 1] if self.low[i - 1] is not None else sar)
                 if i > 1:
                     sar = min(sar, self.low[i - 2] if self.low[i - 2] is not None else sar)

                 # Check for trend reversal
                 if current_low < sar:
                     trend = -1  # Switch to downtrend
                     sar = ep  # New SAR is the old EP
                     ep = current_low  # New EP is the current low
                     af = acceleration  # Reset AF
                 else:  # Continue uptrend
                     # Update EP if new high is made
                     if current_high > ep:
                         ep = current_high
                         af = min(af + acceleration, max_acceleration)

             else:  # --- Downtrend Calculation ---
                 sar = prev_sar + af * (ep - prev_sar)
                  # Ensure SAR does not move below the previous two highs
                 sar = max(sar, self.high[i - 1] if self.high[i - 1] is not None else sar)
                 if i > 1:
                     sar = max(sar, self.high[i - 2] if self.high[i - 2] is not None else sar)

                 # Check for trend reversal
                 if current_high > sar:
                     trend = 1  # Switch to uptrend
                     sar = ep  # New SAR is the old EP
                     ep = current_high  # New EP is the current high
                     af = acceleration  # Reset AF
                 else:  # Continue downtrend
                     # Update EP if new low is made
                     if current_low < ep:
                         ep = current_low
                         af = min(af + acceleration, max_acceleration)

             psar_list[i] = sar

        return psar_list

    def calculate_fve(self) -> list[float | None]:  # Force Index (smoothed is more common)
        if len(self.close) < 2: return [None] * len(self.close)
        force = [None] * len(self.close)
        for i in range(1, len(self.close)):
            c, pc, v = self.close[i], self.close[i - 1], self.volume[i]
            if c is not None and pc is not None and v is not None:
                force[i] = (c - pc) * v
        # Often an EMA of the raw force index is used.
        # Example: return ewm_mean(force, span=13) # 13-period EMA of Force Index
        # Returning raw force index cumulative sum here based on original name 'fve'
        return list_cumsum(force)

    def calculate_bollinger_bands(
        self, window: int = 40, num_std_dev: float = 2.0
    ) -> dict[str, list[float | None]]:
        if len(self.close) < window:
            return {"upper_band": [], "middle_band": [], "lower_band": []}

        middle_band = rolling_mean(self.close, window)
        rolling_sd = rolling_std(self.close, window)

        upper_band = [None] * len(self.close)
        lower_band = [None] * len(self.close)

        for i in range(window - 1, len(self.close)):
            mb = middle_band[i]
            sd = rolling_sd[i]
            if mb is not None and sd is not None:
                upper_band[i] = mb + (sd * num_std_dev)
                lower_band[i] = mb - (sd * num_std_dev)

        return {
            "upper_band": upper_band,
            "middle_band": middle_band,
            "lower_band": lower_band,
        }

    def analyze_orderbook_levels(self, orderbook: dict, current_price: Decimal) -> str:
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return f"{NEON_YELLOW}Orderbook data not available or incomplete.{RESET}"

        # orderbook['bids'] and ['asks'] are lists of [price, size]
        bids = orderbook['bids']  # Already should be floats from fetch_orderbook
        asks = orderbook['asks']

        analysis_output = ""
        float(current_price)

        def check_cluster_at_level(
            level_name,
            level_price,
            bids_data,
            asks_data,
            threshold=self.config["orderbook_cluster_threshold"],
            price_range_factor=0.0005  # 0.05% range around the level
        ) -> str | None:
             if not isinstance(level_price, (float, int)): return None  # Skip if level_price is invalid

             price_range = level_price * price_range_factor
             min_p, max_p = level_price - price_range, level_price + price_range

             bid_volume_at_level = sum(size for price, size in bids_data if min_p <= price <= max_p)
             ask_volume_at_level = sum(size for price, size in asks_data if min_p <= price <= max_p)

             if bid_volume_at_level > threshold:
                 return f"Significant bid volume ({bid_volume_at_level:.0f}) near {level_name} ${level_price:.2f}."
             if ask_volume_at_level > threshold:
                 return f"Significant ask volume ({ask_volume_at_level:.0f}) near {level_name} ${level_price:.2f}."
             return None

        # Check Fibonacci Levels
        for level_type, levels_dict in self.levels.items():  # e.g., level_type = 'Support' or 'Resistance'
             if isinstance(levels_dict, dict):
                for level_name, level_value in levels_dict.items():  # e.g., level_name = 'Fib 23.6%'
                    cluster_analysis = check_cluster_at_level(
                        f"{level_type} {level_name}", level_value, bids, asks
                    )
                    if cluster_analysis:
                         color = NEON_GREEN if level_type == "Support" else NEON_RED
                         analysis_output += f"  {color}{cluster_analysis}{RESET}\n"

        # Check Pivot Points
        for level_name, level_price in self.levels.items():
             if level_name not in ["Support", "Resistance"] and isinstance(level_price, (float, int, Decimal)):
                cluster_analysis = check_cluster_at_level(
                    f"Pivot {level_name}", float(level_price), bids, asks
                )
                if cluster_analysis:
                    analysis_output += f"  {NEON_BLUE}{cluster_analysis}{RESET}\n"

        if not analysis_output:
            return "  No significant orderbook clusters detected near Fibonacci/Pivot levels."
        return analysis_output.strip()

    def analyze(self, current_price: Decimal, timestamp: str) -> None:
        if not self.data:
            self.logger.error(f"{NEON_RED}No kline data available for analysis.{RESET}")
            return

        # Ensure data is usable
        valid_highs = [h for h in self.high if h is not None]
        valid_lows = [l for l in self.low if l is not None]
        last_close = self._get_last_valid(self.close)

        if not valid_highs or not valid_lows or last_close is None:
             self.logger.error(f"{NEON_RED}Insufficient valid high/low/close data for analysis.{RESET}")
             return

        high = max(valid_highs)
        low = min(valid_lows)
        close = last_close

        self.calculate_fibonacci_retracement(high, low, float(current_price))
        self.calculate_pivot_points(high, low, close)
        nearest_supports, nearest_resistances = self.find_nearest_levels(
            float(current_price)
        )

        trend_data = self.determine_trend_momentum()
        trend = trend_data.get("trend", "Unknown")
        strength = trend_data.get("strength", 0)
        atr = self.calculate_atr()
        obv = self.calculate_obv()
        rsi_20 = self.calculate_rsi(window=20)
        rsi_100 = self.calculate_rsi(window=100)
        mfi = self.calculate_mfi()
        cci = self.calculate_cci()
        wr = self.calculate_williams_r()
        adx = self.calculate_adx()  # Returns a single float or None
        adi = self.calculate_adi()
        sma10 = self.calculate_sma(10)
        psar = self.calculate_psar()
        fve = self.calculate_fve()
        macd_data = self.calculate_macd()
        bollinger_bands_data = self.calculate_bollinger_bands(
            window=self.config["bollinger_bands_period"],
            num_std_dev=self.config["bollinger_bands_std_dev"],
        )

        orderbook_data = fetch_orderbook(
            self.symbol, self.config["orderbook_limit"], self.logger
        )
        orderbook_analysis_str = self.analyze_orderbook_levels(
            orderbook_data, current_price
        )

        # Get last 3 values for indicators, handling None
        def get_last_n(data_list, n=3):
            valid_vals = [v for v in data_list if v is not None]
            padding = [None] * (n - len(valid_vals))
            return padding + valid_vals[-n:]

        last_3_close = get_last_n(self.close)
        last_3_vol = get_last_n(self.volume)

        indicator_values = {
            "obv": get_last_n(obv),
            "rsi_20": get_last_n(rsi_20),
            "rsi_100": get_last_n(rsi_100),
            "mfi": get_last_n(mfi),
            "cci": get_last_n(cci),
            "wr": get_last_n(wr),
            "adx": [adx] * 3,  # ADX is single value, repeat for consistency
            "adi": get_last_n(adi),
            "mom": [trend_data] * 3,  # Trend data is a dict, repeat
            "sma": get_last_n(sma10, n=1),  # Only need last SMA
            "psar": get_last_n(psar),
            "fve": get_last_n(fve),
            "macd": [  # Get last 3 tuples of (macd, signal, hist)
                 list(get_last_n(macd_data.get(k, []), n=3)) for k in ["macd", "signal", "histogram"]
            ],
            "bollinger_bands": [  # Get last 3 tuples of (upper, middle, lower)
                 list(get_last_n(bollinger_bands_data.get(k, []), n=3)) for k in ["upper_band", "middle_band", "lower_band"]
             ]
        }

        output = f"""
{NEON_BLUE}Exchange:{RESET} Bybit
{NEON_BLUE}Symbol:{RESET} {self.symbol}
{NEON_BLUE}Interval:{RESET} {self.interval}
{NEON_BLUE}Timestamp:{RESET} {timestamp}
{NEON_BLUE}Price:{RESET}   {'N/A' if last_3_close[0] is None else f'{last_3_close[0]:.2f}'} | {'N/A' if last_3_close[1] is None else f'{last_3_close[1]:.2f}'} | {'N/A' if last_3_close[2] is None else f'{last_3_close[2]:.2f}'}
{NEON_BLUE}Vol:{RESET}   {'N/A' if last_3_vol[0] is None else f'{last_3_vol[0]:,}'} | {'N/A' if last_3_vol[1] is None else f'{last_3_vol[1]:,}'} | {'N/A' if last_3_vol[2] is None else f'{last_3_vol[2]:,}'}
{NEON_BLUE}Current Price:{RESET} {current_price:.2f}
{NEON_BLUE}ATR:{RESET} {self._get_last_valid(atr):.4f}
{NEON_BLUE}Trend:{RESET} {trend} (Strength: {strength:.2f})
"""
        # RSI Cross Check
        rsi_20_val = indicator_values["rsi_20"][-1]
        rsi_100_val = indicator_values["rsi_100"][-1]
        prev_rsi_20 = indicator_values["rsi_20"][-2]
        prev_rsi_100 = indicator_values["rsi_100"][-2]

        rsi_cross = f"{NEON_YELLOW}RSI 20/100:{RESET} No recent cross or insufficient data"
        if None not in [rsi_20_val, rsi_100_val, prev_rsi_20, prev_rsi_100]:
            if rsi_20_val > rsi_100_val and prev_rsi_20 <= prev_rsi_100:
                rsi_cross = f"{NEON_GREEN}RSI 20/100:{RESET} RSI 20 crossed ABOVE RSI 100"
            elif rsi_20_val < rsi_100_val and prev_rsi_20 >= prev_rsi_100:
                rsi_cross = f"{NEON_RED}RSI 20/100:{RESET} RSI 20 crossed BELOW RSI 100"

        output += rsi_cross + "\n"

        # --- Interpret Indicators ---
        # (interpret_indicator function needs slight modification for new data structure)
        for indicator_name, values in indicator_values.items():
             interpretation = self.interpret_indicator(self.logger, indicator_name, values, last_close)  # Pass last_close if needed
             if interpretation:
                 output += interpretation + "\n"

        output += f"""
{NEON_BLUE}Support and Resistance Levels:{RESET}
"""
        # Use the float values returned by find_nearest_levels
        for s_label, s_value in nearest_supports:
            output += f"S: {s_label} ${s_value:.3f}\n"
        for r_label, r_value in nearest_resistances:
            output += f"R: {r_label} ${r_value:.3f}\n"

        output += f"""
{NEON_BLUE}Orderbook Analysis near Fibonacci/Pivot Levels:{RESET}
{orderbook_analysis_str}
"""

        self.logger.info(output)

    def interpret_indicator(
        self, logger: logging.Logger, indicator_name: str, values: list | dict, last_close: float | None
    ) -> str | None:
        if values is None:
            return f"{indicator_name.upper()}: No data available."

        # Helper to get the last valid numeric value from a list
        def get_last_numeric(value_list):
            if not isinstance(value_list, list): return None
            for x in reversed(value_list):
                if isinstance(x, (int, float)):
                    return x
            return None

        # Helper to get the second to last valid numeric value
        def get_second_last_numeric(value_list):
             if not isinstance(value_list, list): return None
             count = 0
             second_last = None
             for x in reversed(value_list):
                 if isinstance(x, (int, float)):
                     count += 1
                     if count == 2:
                         second_last = x
                         break
             return second_last

        try:
            last_val = None
            prev_val = None

            # Handle different structures of 'values'
            if indicator_name in ["macd", "bollinger_bands"]:
                 # These have nested lists, specific handling below
                 pass
            elif indicator_name == "mom":
                 # This is a list containing the trend_data dict
                 if values and isinstance(values[0], dict):
                     last_val = values[0]  # The dict itself is the 'value'
                 else:
                     return f"{indicator_name.upper()}: Trend data missing."
            elif isinstance(values, list):
                last_val = get_last_numeric(values)
                prev_val = get_second_last_numeric(values)
            else:  # Should not happen based on analyze structure, but handle anyway
                 return f"{indicator_name.upper()}: Unexpected data structure."

            # --- Indicator Interpretation Logic ---
            if indicator_name == "rsi_20" or indicator_name == "rsi_100":
                 if last_val is None: return f"{indicator_name.upper()}: No data."
                 if last_val > 70: return f"{NEON_RED}{indicator_name.upper()}:{RESET} Overbought ({last_val:.2f})"
                 if last_val < 30: return f"{NEON_GREEN}{indicator_name.upper()}:{RESET} Oversold ({last_val:.2f})"
                 return f"{NEON_YELLOW}{indicator_name.upper()}:{RESET} Neutral ({last_val:.2f})"

            elif indicator_name == "mfi":
                if last_val is None: return "MFI: No data."
                if last_val > 80: return f"{NEON_RED}MFI:{RESET} Overbought ({last_val:.2f})"
                if last_val < 20: return f"{NEON_GREEN}MFI:{RESET} Oversold ({last_val:.2f})"
                return f"{NEON_YELLOW}MFI:{RESET} Neutral ({last_val:.2f})"

            elif indicator_name == "cci":
                 if last_val is None: return "CCI: No data."
                 if last_val > 100: return f"{NEON_RED}CCI:{RESET} Overbought ({last_val:.2f})"
                 if last_val < -100: return f"{NEON_GREEN}CCI:{RESET} Oversold ({last_val:.2f})"
                 return f"{NEON_YELLOW}CCI:{RESET} Neutral ({last_val:.2f})"

            elif indicator_name == "wr":
                 if last_val is None: return "Williams %R: No data."
                 if last_val < -80: return f"{NEON_GREEN}Williams %R:{RESET} Oversold ({last_val:.2f})"
                 if last_val > -20: return f"{NEON_RED}Williams %R:{RESET} Overbought ({last_val:.2f})"
                 return f"{NEON_YELLOW}Williams %R:{RESET} Neutral ({last_val:.2f})"

            elif indicator_name == "adx":
                 # ADX value is directly available in last_val here (repeated 3 times in list)
                 adx_val = get_last_numeric(values)  # Get the actual ADX value
                 if adx_val is None: return "ADX: No data."
                 if adx_val > 25: return f"{NEON_GREEN}ADX:{RESET} Trending ({adx_val:.2f})"
                 return f"{NEON_YELLOW}ADX:{RESET} Ranging ({adx_val:.2f})"

            elif indicator_name == "obv":
                if last_val is None or prev_val is None: return "OBV: Insufficient data."
                if last_val > prev_val: return f"{NEON_BLUE}OBV:{RESET} Bullish"
                if last_val < prev_val: return f"{NEON_BLUE}OBV:{RESET} Bearish"
                return f"{NEON_BLUE}OBV:{RESET} Neutral"

            elif indicator_name == "adi":
                if last_val is None or prev_val is None: return "ADI: Insufficient data."
                if last_val > prev_val: return f"{NEON_BLUE}ADI:{RESET} Accumulation"
                if last_val < prev_val: return f"{NEON_BLUE}ADI:{RESET} Distribution"
                return f"{NEON_BLUE}ADI:{RESET} Neutral"

            elif indicator_name == "mom":
                 trend = last_val.get("trend", "N/A")
                 strength = last_val.get("strength", 0)
                 return f"{NEON_PURPLE}Momentum:{RESET} {trend} (Strength: {strength:.2f})"

            elif indicator_name == "sma":
                 if last_val is None: return "SMA (10): No data."
                 return f"{NEON_YELLOW}SMA (10):{RESET} {last_val:.2f}"

            elif indicator_name == "psar":
                 if last_val is None: return "PSAR: No data."
                 # Compare PSAR to price for trend indication
                 trend_indicator = ""
                 if last_close is not None:
                     if last_close > last_val: trend_indicator = f" ({NEON_GREEN}Price Above - Bullish{RESET})"
                     elif last_close < last_val: trend_indicator = f" ({NEON_RED}Price Below - Bearish{RESET})"
                 return f"{NEON_BLUE}PSAR:{RESET} {last_val:.4f}{trend_indicator}"

            elif indicator_name == "fve":
                 if last_val is None: return "FVE: No data."
                 return f"{NEON_BLUE}FVE:{RESET} {last_val:,.0f} (Last Value)"  # Raw cumulative force

            elif indicator_name == "macd":
                 # values = [ [macd_n-2, macd_n-1, macd_n], [sig_n-2, sig_n-1, sig_n], [hist_n-2, hist_n-1, hist_n] ]
                 if len(values) != 3 or not all(isinstance(v, list) for v in values):
                      return f"{NEON_RED}MACD:{RESET} Data structure issue."

                 last_macd = get_last_numeric(values[0])
                 last_signal = get_last_numeric(values[1])
                 last_hist = get_last_numeric(values[2])

                 if None in [last_macd, last_signal, last_hist]:
                      return f"{NEON_RED}MACD:{RESET} Calculation issue or insufficient data."

                 cross_info = ""
                 # Basic Crossover Check
                 prev_macd = get_second_last_numeric(values[0])
                 prev_signal = get_second_last_numeric(values[1])
                 if prev_macd is not None and prev_signal is not None:
                      if prev_macd <= prev_signal and last_macd > last_signal:
                           cross_info = f" ({NEON_GREEN}Bullish Cross{RESET})"
                      elif prev_macd >= prev_signal and last_macd < last_signal:
                           cross_info = f" ({NEON_RED}Bearish Cross{RESET})"

                 return f"{NEON_GREEN}MACD:{RESET} MACD={last_macd:.2f}, Signal={last_signal:.2f}, Hist={last_hist:.2f}{cross_info}"

            elif indicator_name == "bollinger_bands":
                 # values = [ [ub_n-2, ub_n-1, ub_n], [mb_n-2, mb_n-1, mb_n], [lb_n-2, lb_n-1, lb_n] ]
                 if len(values) != 3 or not all(isinstance(v, list) for v in values):
                      return f"{NEON_RED}Bollinger Bands:{RESET} Data structure issue."

                 upper_band = get_last_numeric(values[0])
                 middle_band = get_last_numeric(values[1])
                 lower_band = get_last_numeric(values[2])

                 if None in [upper_band, middle_band, lower_band, last_close]:
                      return f"{NEON_RED}Bollinger Bands:{RESET} Calculation issue or insufficient data."

                 if last_close > upper_band:
                     return f"{NEON_RED}Bollinger Bands:{RESET} Price above Upper Band ({upper_band:.2f})"
                 elif last_close < lower_band:
                     return f"{NEON_GREEN}Bollinger Bands:{RESET} Price below Lower Band ({lower_band:.2f})"
                 else:
                     return f"{NEON_YELLOW}Bollinger Bands:{RESET} Price within Bands (Upper={upper_band:.2f}, Middle={middle_band:.2f}, Lower={lower_band:.2f})"
            else:
                return None  # Unknown indicator
        except (TypeError, IndexError, KeyError) as e:
            logger.error(f"Error interpreting {indicator_name}: {e}")
            return f"{indicator_name.upper()}: Interpretation error."
        except Exception as e:
            logger.error(f"Unexpected error interpreting {indicator_name}: {e}")
            return f"{indicator_name.upper()}: Unexpected error."


def main() -> None:
    symbol = ""
    while True:
        symbol = (
            input(f"{NEON_BLUE}Enter trading symbol (e.g., BTCUSDT): {RESET}")
            .upper()
            .strip()
        )
        if symbol:
            break

    interval = ""
    while True:
        interval = input(
            f"{NEON_BLUE}Enter timeframe (e.g., {', '.join(VALID_INTERVALS)}): {RESET}"
        ).strip()
        if not interval:
            interval = CONFIG["interval"]
            break
        if interval in VALID_INTERVALS:
            break
        else:
             pass

    logger = setup_logger(symbol)
    analysis_interval = CONFIG["analysis_interval"]
    retry_delay = CONFIG["retry_delay"]

    while True:
        try:
            current_price = fetch_current_price(symbol, logger)
            if current_price is None:
                logger.error(
                    f"{NEON_RED}Failed to fetch current price. Retrying in {retry_delay} seconds...{RESET}"
                )
                time.sleep(retry_delay)
                continue

            # Fetch klines as list of dictionaries
            kline_list = fetch_klines(symbol, interval, logger=logger, limit=200)  # Fetch enough for calculations
            if not kline_list:
                logger.error(
                    f"{NEON_RED}Failed to fetch kline data or data is empty. Retrying in {retry_delay} seconds...{RESET}"
                )
                time.sleep(retry_delay)
                continue

            analyzer = TradingAnalyzer(kline_list, logger, CONFIG, symbol, interval)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analyzer.analyze(current_price, timestamp)

            # Wait before next analysis cycle
            logger.info(f"{NEON_BLUE}Waiting {analysis_interval} seconds for next analysis...{RESET}")
            time.sleep(analysis_interval)

        except requests.exceptions.RequestException as e:
            logger.error(
                f"{NEON_RED}Network error: {e}. Retrying in {retry_delay} seconds...{RESET}"
            )
            time.sleep(retry_delay)
        except KeyboardInterrupt:
            logger.info(f"{NEON_YELLOW}Analysis stopped by user.{RESET}")
            break
        except Exception as e:
            logger.exception(  # Use exception to log traceback
                f"{NEON_RED}An unexpected error occurred in main loop: {e}. Retrying in {retry_delay} seconds...{RESET}"
            )
            time.sleep(retry_delay)


if __name__ == "__main__":
    main()
