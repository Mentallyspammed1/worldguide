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
import numpy as np

# import pandas as pd # Removed pandas import
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
) -> list[dict]:  # Changed return type from pd.DataFrame
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
            # Convert raw list of lists to list of dictionaries
            columns = ["start_time", "open", "high", "low", "close", "volume", "turnover"]
            kline_data = []
            for row in data:
                kline_dict = {}
                for i, col_name in enumerate(columns):
                    if i < len(row):
                         try:
                             if col_name == "start_time":
                                 # Convert timestamp string to integer then to datetime object
                                 # Keep as ms timestamp string for now, conversion done later
                                 kline_dict[col_name] = int(row[i])
                             else:
                                 kline_dict[col_name] = float(row[i]) if row[i] else 0.0  # Handle empty turnover etc.
                         except (ValueError, TypeError) as e:
                             if logger:
                                 logger.warning(f"Could not convert {col_name}={row[i]} to number: {e}. Setting to 0.")
                             kline_dict[col_name] = 0.0 if col_name != "start_time" else 0
                    else:
                        kline_dict[col_name] = 0.0  # Assign default if column missing in row
                kline_data.append(kline_dict)

            # Ensure essential keys exist even if API changes response format slightly
            for kline in kline_data:
                for key in ["start_time", "open", "high", "low", "close", "volume"]:
                    if key not in kline:
                        if logger:
                            logger.warning(f"Key '{key}' missing in kline data, adding default 0.")
                        kline[key] = 0.0 if key != "start_time" else 0

            return kline_data

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
            if orderbook_data and 'bids' in orderbook_data and 'asks' in orderbook_data:
                # Convert price/size lists to float
                bids = [[float(p), float(s)] for p, s in orderbook_data['bids']]
                asks = [[float(p), float(s)] for p, s in orderbook_data['asks']]
                return {'bids': bids, 'asks': asks}
            else:
                logger.error(
                    f"{NEON_RED}Failed to fetch valid orderbook data from ccxt (empty/malformed response). Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}"
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

# --- Helper Functions for Manual Calculations ---


def calculate_sma_manual(data: list[float], window: int) -> list[float]:
    if not data or window <= 0 or len(data) < window:
        return [np.nan] * len(data)
    sma = [np.nan] * (window - 1)
    for i in range(window - 1, len(data)):
        sma.append(sum(data[i - window + 1 : i + 1]) / window)
    return sma


def calculate_ema_manual(data: list[float], window: int) -> list[float]:
    if not data or window <= 0:
        return [np.nan] * len(data)
    ema = [np.nan] * len(data)
    if len(data) >= window:
        ema[window - 1] = sum(data[:window]) / window  # Start with SMA
        multiplier = 2 / (window + 1)
        for i in range(window, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema


def calculate_rsi_manual(closes: list[float], window: int = 14) -> list[float]:
    if len(closes) < window + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = [np.nan] * len(closes)
    avg_loss = [np.nan] * len(closes)
    rsi = [np.nan] * len(closes)

    if len(gains) >= window:
        avg_gain[window] = sum(gains[:window]) / window
        avg_loss[window] = sum(losses[:window]) / window

        for i in range(window + 1, len(closes)):
            avg_gain[i] = ((avg_gain[i - 1] * (window - 1)) + gains[i - 1]) / window
            avg_loss[i] = ((avg_loss[i - 1] * (window - 1)) + losses[i - 1]) / window

        for i in range(window, len(closes)):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_macd_manual(closes: list[float], fast_period=12, slow_period=26, signal_period=9) -> dict:
    ema_fast = calculate_ema_manual(closes, fast_period)
    ema_slow = calculate_ema_manual(closes, slow_period)

    macd_line = [np.nan] * len(closes)
    for i in range(len(closes)):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = calculate_ema_manual(macd_line, signal_period)

    histogram = [np.nan] * len(closes)
    for i in range(len(closes)):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def calculate_stoch_rsi_manual(closes: list[float], rsi_window=14, stoch_window=12, k_window=3, d_window=3) -> dict:
    rsi = calculate_rsi_manual(closes, rsi_window)

    stoch_rsi = [np.nan] * len(rsi)
    if len(rsi) >= stoch_window:
        for i in range(stoch_window - 1, len(rsi)):
            period_rsi = [r for r in rsi[i - stoch_window + 1 : i + 1] if not np.isnan(r)]
            if period_rsi:
                min_rsi = min(period_rsi)
                max_rsi = max(period_rsi)
                if max_rsi - min_rsi != 0 and not np.isnan(rsi[i]):
                     stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi)
                elif not np.isnan(rsi[i]):  # Handle case where max == min
                    stoch_rsi[i] = 0.0  # Or some other default, e.g., 0.5?

    k_line = calculate_sma_manual(stoch_rsi, k_window)  # SMA of StochRSI
    d_line = calculate_sma_manual(k_line, d_window)    # SMA of K line

    return {"stoch_rsi": stoch_rsi, "k": k_line, "d": d_line}


def calculate_bollinger_bands_manual(closes: list[float], window: int = 20, num_std_dev: float = 2.0) -> dict:
    if len(closes) < window:
        return {"upper_band": [np.nan] * len(closes), "middle_band": [np.nan] * len(closes), "lower_band": [np.nan] * len(closes)}

    middle_band = calculate_sma_manual(closes, window)
    upper_band = [np.nan] * len(closes)
    lower_band = [np.nan] * len(closes)

    for i in range(window - 1, len(closes)):
        period_closes = closes[i - window + 1 : i + 1]
        std_dev = np.std(period_closes)
        if not np.isnan(middle_band[i]):
             upper_band[i] = middle_band[i] + (std_dev * num_std_dev)
             lower_band[i] = middle_band[i] - (std_dev * num_std_dev)

    return {"upper_band": upper_band, "middle_band": middle_band, "lower_band": lower_band}


def calculate_atr_manual(klines: list[dict], window: int = 14) -> list[float]:
    if len(klines) < window + 1:
        return [np.nan] * len(klines)

    tr_values = [np.nan]  # TR needs previous close, so first is nan
    for i in range(1, len(klines)):
        high = klines[i]['high']
        low = klines[i]['low']
        prev_close = klines[i - 1]['close']

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

    # Calculate ATR using simple moving average for simplicity
    atr = calculate_sma_manual(tr_values, window)
    return atr

# --- End Helper Functions ---


class TradingAnalyzer:
    def __init__(
        self,
        klines: list[dict],  # Changed from df to klines list
        logger: logging.Logger,
        config: dict,
        symbol: str,
        interval: str,
    ) -> None:
        self.klines = klines  # Store raw kline list
        self.logger = logger
        self.levels = {}
        self.fib_levels = {}
        self.config = config
        self.signal = None
        self.weight_sets = config["weight_sets"]
        self.user_defined_weights = self.weight_sets["low_volatility"]
        self.symbol = symbol
        self.interval = interval

        # Extract data lists for calculations
        self.closes = [k.get('close', np.nan) for k in self.klines]
        self.highs = [k.get('high', np.nan) for k in self.klines]
        self.lows = [k.get('low', np.nan) for k in self.klines]
        self.volumes = [k.get('volume', np.nan) for k in self.klines]

    # --- Calculation methods using manual helpers ---
    def calculate_sma(self, window: int) -> list[float]:
         return calculate_sma_manual(self.closes, window)

    def calculate_momentum(self, period: int = 10) -> list[float]:
        if len(self.closes) < period + 1:
            return [np.nan] * len(self.closes)
        momentum = [np.nan] * period
        for i in range(period, len(self.closes)):
            prev_close = self.closes[i - period]
            if prev_close != 0:
                mom = ((self.closes[i] - prev_close) / prev_close) * 100
            else:
                mom = np.nan  # Avoid division by zero
            momentum.append(mom)
        return momentum

    def calculate_cci(self, window: int = 20, constant: float = 0.015) -> list[float]:
        if len(self.closes) < window:
            return [np.nan] * len(self.closes)

        typical_prices = [(h + l + c) / 3 for h, l, c in zip(self.highs, self.lows, self.closes, strict=False)]
        sma_typical = calculate_sma_manual(typical_prices, window)

        cci = [np.nan] * len(typical_prices)
        for i in range(window - 1, len(typical_prices)):
            period_tp = typical_prices[i - window + 1 : i + 1]
            mean_tp = sma_typical[i]
            if not np.isnan(mean_tp):
                 mean_deviation = sum(abs(tp - mean_tp) for tp in period_tp) / window
                 if constant * mean_deviation != 0:
                     cci[i] = (typical_prices[i] - mean_tp) / (constant * mean_deviation)
        return cci

    def calculate_williams_r(self, window: int = 14) -> list[float]:
        if len(self.closes) < window:
            return [np.nan] * len(self.closes)

        wr = [np.nan] * (window - 1)
        for i in range(window - 1, len(self.closes)):
            period_highs = self.highs[i - window + 1 : i + 1]
            period_lows = self.lows[i - window + 1 : i + 1]
            highest_high = max(period_highs)
            lowest_low = min(period_lows)
            current_close = self.closes[i]

            if highest_high - lowest_low != 0:
                val = (highest_high - current_close) / (highest_high - lowest_low) * -100
                wr.append(val)
            else:
                wr.append(np.nan)  # Avoid division by zero
        return wr

    def calculate_mfi(self, window: int = 14) -> list[float]:
        if len(self.closes) < window + 1:
            return [np.nan] * len(self.closes)

        typical_prices = [(h + l + c) / 3 for h, l, c in zip(self.highs, self.lows, self.closes, strict=False)]
        raw_money_flow = [tp * v for tp, v in zip(typical_prices, self.volumes, strict=False)]

        positive_flow = [0.0] * len(typical_prices)
        negative_flow = [0.0] * len(typical_prices)

        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i - 1]:
                positive_flow[i] = raw_money_flow[i]
            elif typical_prices[i] < typical_prices[i - 1]:
                negative_flow[i] = raw_money_flow[i]
            # If equal, both remain 0

        mfi = [np.nan] * len(typical_prices)
        for i in range(window, len(typical_prices)):
            sum_pos_flow = sum(positive_flow[i - window + 1 : i + 1])
            sum_neg_flow = sum(negative_flow[i - window + 1 : i + 1])

            if sum_neg_flow == 0:
                # Handle division by zero - if no negative flow, MFI is 100
                 mfi[i] = 100.0
            else:
                money_ratio = sum_pos_flow / sum_neg_flow
                mfi[i] = 100 - (100 / (1 + money_ratio))
        return mfi

    def calculate_fibonacci_retracement(
        self, high: float, low: float, current_price: float
    ) -> dict[str, float]:
        try:
            diff = high - low
            if diff == 0:
                self.logger.warning(f"{NEON_YELLOW}Fibonacci calculation warning: High equals Low ({high}).{RESET}")
                return {}
            fib_levels = {
                "Fib 23.6%": high - diff * 0.236,
                "Fib 38.2%": high - diff * 0.382,
                "Fib 50.0%": high - diff * 0.5,
                "Fib 61.8%": high - diff * 0.618,
                "Fib 78.6%": high - diff * 0.786,
                "Fib 88.6%": high - diff * 0.886,  # Less common but added
                "Fib 94.1%": high - diff * 0.941,  # Less common but added
            }
            self.levels = {"Support": {}, "Resistance": {}}  # Reset S/R levels specific to Fib
            for label, value in fib_levels.items():
                if value < current_price:
                    self.levels["Support"][label] = value
                elif value > current_price:
                    self.levels["Resistance"][label] = value
            self.fib_levels = fib_levels  # Store all calculated levels
            return self.fib_levels
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
            # Update self.levels, ensuring not to overwrite Fib S/R dicts
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
            # Don't reset self.levels here, Fib levels might still be valid

    def find_nearest_levels(
        self, current_price: float, num_levels: int = 5
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        try:
            support_levels = []
            resistance_levels = []

            def process_level(label, value) -> None:
                 # Ensure value is a number before comparison
                if isinstance(value, (int, float, Decimal)) and not np.isnan(value):
                    if value < current_price:
                        support_levels.append((label, float(value)))
                    elif value > current_price:
                        resistance_levels.append((label, float(value)))

            # Process Fibonacci levels
            for label, value in self.levels.get("Support", {}).items():
                process_level(label, value)  # Already categorized by Fib calc
            for label, value in self.levels.get("Resistance", {}).items():
                process_level(label, value)  # Already categorized by Fib calc

            # Process Pivot Point levels
            for label, value in self.levels.items():
                 if label not in ["Support", "Resistance"]:  # Avoid reprocessing Fib dicts
                     process_level(label.upper(), value)  # Process top-level PP values

            # Sort by distance and take the nearest N
            support_levels.sort(key=lambda x: abs(x[1] - current_price))
            resistance_levels.sort(key=lambda x: abs(x[1] - current_price))

            nearest_supports = support_levels[:num_levels]
            nearest_resistances = resistance_levels[:num_levels]

            # Sort by price for cleaner output
            nearest_supports.sort(key=lambda x: x[1], reverse=True)  # Highest support first
            nearest_resistances.sort(key=lambda x: x[1])  # Lowest resistance first

            return nearest_supports, nearest_resistances
        except (KeyError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Error finding nearest levels: {e}{RESET}")
            return [], []
        except Exception as e:
            self.logger.exception(
                f"{NEON_RED}Unexpected error finding nearest levels: {e}{RESET}"
            )
            return [], []

    def calculate_atr(self, window: int = 14) -> list[float]:  # Default changed from 20
         return calculate_atr_manual(self.klines, window)

    def calculate_rsi(self, window: int = 14) -> list[float]:
        return calculate_rsi_manual(self.closes, window)

    def calculate_stoch_rsi(
        self,
        rsi_window: int = 14,
        stoch_window: int = 12,
        k_window: int = 3,
        d_window: int = 3,
    ) -> dict:
        return calculate_stoch_rsi_manual(self.closes, rsi_window, stoch_window, k_window, d_window)

    def calculate_momentum_ma(self) -> dict:
        momentum = self.calculate_momentum(self.config["momentum_period"])
        momentum_ma_short = calculate_sma_manual(momentum, self.config["momentum_ma_short"])
        momentum_ma_long = calculate_sma_manual(momentum, self.config["momentum_ma_long"])
        volume_ma = calculate_sma_manual(self.volumes, self.config["volume_ma_period"])
        return {
            "momentum": momentum,
            "momentum_ma_short": momentum_ma_short,
            "momentum_ma_long": momentum_ma_long,
            "volume_ma": volume_ma
        }

    def calculate_macd(self) -> dict:
        return calculate_macd_manual(self.closes)

    def detect_macd_divergence(self, macd_data: dict) -> str | None:
        if not macd_data or len(self.closes) < 30 or len(macd_data.get("histogram", [])) < 2:
            return None

        prices = self.closes
        macd_histogram = macd_data["histogram"]

        # Ensure we have valid numbers for comparison
        if np.isnan(prices[-1]) or np.isnan(prices[-2]) or \
           np.isnan(macd_histogram[-1]) or np.isnan(macd_histogram[-2]):
            return None

        # Bullish Divergence: Price makes lower low, Histogram makes higher low
        if (prices[-1] < prices[-2]) and (macd_histogram[-1] > macd_histogram[-2]):
            return "bullish"
        # Bearish Divergence: Price makes higher high, Histogram makes lower high
        elif (prices[-1] > prices[-2]) and (macd_histogram[-1] < macd_histogram[-2]):
            return "bearish"

        return None

    def calculate_ema(self, window: int) -> list[float]:
        return calculate_ema_manual(self.closes, window)

    def determine_trend_momentum(self, mom_ma_data: dict, atr_values: list[float]) -> dict:
         if len(self.closes) < self.config["momentum_ma_long"] or not atr_values or np.isnan(atr_values[-1]):
             return {"trend": "Insufficient Data", "strength": 0}

         mom_short = mom_ma_data["momentum_ma_short"][-1]
         mom_long = mom_ma_data["momentum_ma_long"][-1]
         last_atr = atr_values[-1]

         if np.isnan(mom_short) or np.isnan(mom_long):
              return {"trend": "Insufficient Data", "strength": 0}

         if mom_short > mom_long:
             trend = "Uptrend"
         elif mom_short < mom_long:
             trend = "Downtrend"
         else:
             trend = "Neutral"

         if last_atr == 0:
            self.logger.warning(
                 f"{NEON_YELLOW}ATR is zero, cannot calculate trend strength.{RESET}"
            )
            trend_strength = 0
         else:
            trend_strength = abs(mom_short - mom_long) / last_atr

         return {"trend": trend, "strength": trend_strength}

    def calculate_adx(self, window: int = 14) -> list[float]:
        if len(self.klines) < window + 1:
            return [np.nan] * len(self.klines)

        tr_values = [np.nan]
        plus_dm = [np.nan]
        minus_dm = [np.nan]

        for i in range(1, len(self.klines)):
            h, l, _c = self.klines[i]['high'], self.klines[i]['low'], self.klines[i]['close']
            ph, pl = self.klines[i - 1]['high'], self.klines[i - 1]['low'], self.klines[i - 1]['close']

            tr = max(h - l, abs(h - ph), abs(l - pl))
            tr_values.append(tr)

            move_up = h - ph
            move_down = pl - l

            pdm = move_up if move_up > move_down and move_up > 0 else 0
            mdm = move_down if move_down > move_up and move_down > 0 else 0
            plus_dm.append(pdm)
            minus_dm.append(mdm)

        # Use EMA for smoothing TR, +DM, -DM (common practice)
        atr = calculate_ema_manual(tr_values, window)
        smoothed_plus_dm = calculate_ema_manual(plus_dm, window)
        smoothed_minus_dm = calculate_ema_manual(minus_dm, window)

        plus_di = [np.nan] * len(self.klines)
        minus_di = [np.nan] * len(self.klines)
        dx = [np.nan] * len(self.klines)

        for i in range(window, len(self.klines)):
             if not np.isnan(atr[i]) and atr[i] != 0:
                 if not np.isnan(smoothed_plus_dm[i]):
                     plus_di[i] = 100 * (smoothed_plus_dm[i] / atr[i])
                 if not np.isnan(smoothed_minus_dm[i]):
                     minus_di[i] = 100 * (smoothed_minus_dm[i] / atr[i])

             if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                 di_sum = plus_di[i] + minus_di[i]
                 if di_sum != 0:
                     dx[i] = 100 * (abs(plus_di[i] - minus_di[i]) / di_sum)

        # ADX is the EMA of DX
        adx = calculate_ema_manual(dx, window)
        return adx

    def calculate_obv(self) -> list[float]:
        if len(self.closes) < 2:
            return [np.nan] * len(self.closes)

        obv = [0.0] * len(self.closes)  # Start OBV at 0 or use first volume? Let's use 0.
        for i in range(1, len(self.closes)):
            if self.closes[i] > self.closes[i - 1]:
                obv[i] = obv[i - 1] + self.volumes[i]
            elif self.closes[i] < self.closes[i - 1]:
                obv[i] = obv[i - 1] - self.volumes[i]
            else:
                obv[i] = obv[i - 1]  # No change if price is the same
        return obv

    def calculate_adi(self) -> list[float]:
        if len(self.closes) < 1:
            return []

        adi = [0.0] * len(self.closes)
        for i in range(len(self.closes)):
            high, low, close, volume = self.highs[i], self.lows[i], self.closes[i], self.volumes[i]
            if high - low != 0:
                mfm = ((close - low) - (high - close)) / (high - low)
                mfv = mfm * volume
            else:
                mfv = 0  # Avoid division by zero if high == low

            adi[i] = (adi[i - 1] if i > 0 else 0) + mfv
        return adi

    def calculate_psar(self, acceleration=0.02, max_acceleration=0.2) -> list[float]:
        if len(self.klines) < 2:
            return [np.nan] * len(self.klines)

        psar_values = [np.nan] * len(self.klines)
        trend = 1  # Initial trend assumed up
        ep = self.highs[0]  # Initial extreme point
        af = acceleration  # Initial acceleration factor
        psar_values[0] = self.lows[0]  # Initial PSAR often set to low

        for i in range(1, len(self.klines)):
            prev_psar = psar_values[i - 1]
            current_high = self.highs[i]
            current_low = self.lows[i]
            prev_high = self.highs[i - 1]  # Needed for EP update logic
            prev_low = self.lows[i - 1]   # Needed for EP update logic

            if trend == 1:  # Uptrend
                sar = prev_psar + af * (ep - prev_psar)
                # Ensure SAR does not move above the previous or current low
                sar = min(sar, prev_low, current_low)

                if current_low < sar:  # Trend reverses to down
                    trend = -1
                    sar = ep  # New SAR becomes the extreme point of the previous trend
                    ep = current_low  # New extreme point is the low of the reversal candle
                    af = acceleration  # Reset AF
                else:  # Trend continues up
                    if current_high > ep:  # New extreme point reached
                        ep = current_high
                        af = min(af + acceleration, max_acceleration)

            else:  # Downtrend (trend == -1)
                sar = prev_psar + af * (ep - prev_psar)
                 # Ensure SAR does not move below the previous or current high
                sar = max(sar, prev_high, current_high)

                if current_high > sar:  # Trend reverses to up
                    trend = 1
                    sar = ep  # New SAR becomes the extreme point of the previous trend
                    ep = current_high  # New extreme point is the high of the reversal candle
                    af = acceleration  # Reset AF
                else:  # Trend continues down
                    if current_low < ep:  # New extreme point reached
                        ep = current_low
                        af = min(af + acceleration, max_acceleration)

            psar_values[i] = sar

        return psar_values

    def calculate_fve(self) -> list[float]:  # Force Index (often smoothed with EMA)
        if len(self.closes) < 2:
            return [np.nan] * len(self.closes)
        force = [np.nan]  # First value needs previous close
        for i in range(1, len(self.closes)):
            force.append((self.closes[i] - self.closes[i - 1]) * self.volumes[i])

        # Often smoothed, e.g., EMA(13) of force index
        # Returning raw force index cumulative sum for simplicity as requested by name 'fve'
        fve = [0.0] * len(force)
        for i in range(1, len(force)):
            if not np.isnan(force[i]):
                 fve[i] = fve[i - 1] + force[i]
            else:
                 fve[i] = fve[i - 1]  # Carry forward if force couldn't be calculated
        return fve

    def calculate_bollinger_bands(
        self, window: int = 20, num_std_dev: float = 2.0  # Default window changed from 40
    ) -> dict:
         return calculate_bollinger_bands_manual(self.closes, window, num_std_dev)

    def analyze_orderbook_levels(self, orderbook: dict | None, current_price: Decimal) -> str:
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return f"{NEON_YELLOW}Orderbook data not available or invalid.{RESET}"

        # Orderbook data should already be floats from fetch_orderbook
        bids = orderbook['bids']  # List of [price, size]
        asks = orderbook['asks']  # List of [price, size]
        float(current_price)

        analysis_output = ""
        threshold = float(self.config.get("orderbook_cluster_threshold", 1000))  # Use float threshold

        def check_cluster_at_level(
            level_name: str,
            level_price: float,
            bids_data: list[list[float]],
            asks_data: list[list[float]],
            cluster_threshold: float,
            price_tolerance_percent: float = 0.05  # 0.05% tolerance around the level
        ) -> str | None:
            tolerance = level_price * (price_tolerance_percent / 100.0)
            min_price = level_price - tolerance
            max_price = level_price + tolerance

            bid_volume_at_level = sum(size for price, size in bids_data if min_price <= price <= max_price)
            ask_volume_at_level = sum(size for price, size in asks_data if min_price <= price <= max_price)

            if bid_volume_at_level > cluster_threshold:
                return f"Significant bid volume ({bid_volume_at_level:.2f}) near {level_name} @ {level_price:.4f}."
            if ask_volume_at_level > cluster_threshold:
                return f"Significant ask volume ({ask_volume_at_level:.2f}) near {level_name} @ {level_price:.4f}."
            return None

        # Check Fibonacci Levels
        for level_type, levels_dict in self.levels.items():
             if level_type in ["Support", "Resistance"] and isinstance(levels_dict, dict):
                 for label, price in levels_dict.items():
                     if isinstance(price, (float, int, Decimal)):
                         cluster_analysis = check_cluster_at_level(
                             f"{label}", float(price), bids, asks, threshold
                         )
                         if cluster_analysis:
                             color = NEON_GREEN if level_type == "Support" else NEON_RED
                             analysis_output += f"  {color}{cluster_analysis}{RESET}\n"

        # Check Pivot Point Levels
        for label, price in self.levels.items():
            if label not in ["Support", "Resistance"] and isinstance(price, (float, int, Decimal)):
                 cluster_analysis = check_cluster_at_level(
                     f"Pivot {label.upper()}", float(price), bids, asks, threshold
                 )
                 if cluster_analysis:
                      analysis_output += f"  {NEON_BLUE}{cluster_analysis}{RESET}\n"

        if not analysis_output:
            return "  No significant orderbook clusters detected near Fibonacci/Pivot levels."
        return analysis_output.strip()

    def analyze(self, current_price: Decimal, timestamp: str) -> None:
        if not self.klines:
            self.logger.error(f"{NEON_RED}No kline data available for analysis.{RESET}")
            return

        # Ensure we have enough data for calculations requiring lookback
        min_data_needed = max(
            self.config.get("momentum_ma_long", 26),
            self.config.get("bollinger_bands_period", 20),
            self.config.get("atr_period", 14) + 1,  # ATR needs lookback+1
            self.config.get("rsi_period", 14) + 1,  # RSI needs lookback+1
            100 + 1  # For RSI 100
        )
        if len(self.klines) < min_data_needed:
             self.logger.warning(f"{NEON_YELLOW}Insufficient kline data ({len(self.klines)} points) for full analysis (need >= {min_data_needed}). Results may be incomplete.{RESET}")
             # Continue analysis with available data, indicators will return NaNs

        # --- Calculate All Indicators ---
        closes = self.closes
        highs = self.highs
        lows = self.lows

        # Use recent data for Fib/Pivots if available
        recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs) if highs else 0
        recent_low = min(lows[-20:]) if len(lows) >= 20 else min(lows) if lows else 0
        last_close = closes[-1] if closes else 0

        self.calculate_fibonacci_retracement(recent_high, recent_low, float(current_price))
        if recent_high and recent_low and last_close:
             self.calculate_pivot_points(recent_high, recent_low, last_close)
        else:
             self.logger.warning(f"{NEON_YELLOW}Cannot calculate Pivot Points due to missing H/L/C data.{RESET}")

        nearest_supports, nearest_resistances = self.find_nearest_levels(float(current_price))

        mom_ma_data = self.calculate_momentum_ma()
        atr = self.calculate_atr(self.config["atr_period"])
        trend_data = self.determine_trend_momentum(mom_ma_data, atr)
        trend = trend_data.get("trend", "Unknown")
        strength = trend_data.get("strength", 0)

        obv = self.calculate_obv()
        rsi_20 = self.calculate_rsi(window=20)  # Keep separate RSI calculations
        rsi_100 = self.calculate_rsi(window=100)
        rsi_conf = self.calculate_rsi(window=self.config["rsi_period"])  # RSI based on config
        mfi = self.calculate_mfi()
        cci = self.calculate_cci()
        wr = self.calculate_williams_r()
        adx = self.calculate_adx()
        adi = self.calculate_adi()
        sma10 = self.calculate_sma(10)
        psar = self.calculate_psar()
        fve = self.calculate_fve()
        macd_df = self.calculate_macd()
        bollinger_bands_df = self.calculate_bollinger_bands(
            window=self.config["bollinger_bands_period"],
            num_std_dev=self.config["bollinger_bands_std_dev"],
        )
        stoch_rsi_df = self.calculate_stoch_rsi()

        orderbook_data = fetch_orderbook(
            self.symbol, self.config["orderbook_limit"], self.logger
        )
        orderbook_analysis_str = self.analyze_orderbook_levels(
            orderbook_data, current_price
        )

        # --- Get last 3 values (or fewer if data is short) ---
        def get_last_n(data_list: list, n: int = 3):
            if not data_list:
                return [np.nan] * n
            return data_list[-n:]

        # Structure data for output
        indicator_values = {
             "obv": get_last_n(obv),
             "rsi_20": get_last_n(rsi_20),
             "rsi_100": get_last_n(rsi_100),
             "rsi_conf": get_last_n(rsi_conf),  # Configured RSI
             "mfi": get_last_n(mfi),
             "cci": get_last_n(cci),
             "wr": get_last_n(wr),
             "adx": get_last_n(adx, n=1),  # ADX is usually a single value
             "adi": get_last_n(adi),
             "mom": [trend_data] * 3,  # Replicate for consistency, interpretation uses latest
             "sma10": get_last_n(sma10, n=1),  # Just need latest SMA
             "psar": get_last_n(psar),
             "fve": get_last_n(fve),
             "macd": [get_last_n(macd_df['macd']), get_last_n(macd_df['signal']), get_last_n(macd_df['histogram'])],
             "bollinger_bands": [get_last_n(bollinger_bands_df['upper_band']), get_last_n(bollinger_bands_df['middle_band']), get_last_n(bollinger_bands_df['lower_band'])],
             "stoch_rsi": [get_last_n(stoch_rsi_df['stoch_rsi']), get_last_n(stoch_rsi_df['k']), get_last_n(stoch_rsi_df['d'])],
        }

        # --- Format Output ---
        last_3_closes = get_last_n(self.closes)
        last_3_volumes = get_last_n(self.volumes)

        output = f"""
{NEON_BLUE}Exchange:{RESET} Bybit
{NEON_BLUE}Symbol:{RESET} {self.symbol}
{NEON_BLUE}Interval:{RESET} {self.interval}
{NEON_BLUE}Timestamp:{RESET} {timestamp}
{NEON_BLUE}Price:{RESET}   {last_3_closes[0]:.4f} | {last_3_closes[1]:.4f} | {last_3_closes[2]:.4f}
{NEON_BLUE}Vol:{RESET}     {last_3_volumes[0]:,.0f} | {last_3_volumes[1]:,.0f} | {last_3_volumes[2]:,.0f}
{NEON_BLUE}Current Price:{RESET} {current_price:.4f}
{NEON_BLUE}ATR ({self.config['atr_period']}):{RESET} {get_last_n(atr, n=1)[0]:.4f}
{NEON_BLUE}Trend:{RESET} {trend} (Strength: {strength:.2f})
"""
        rsi_20_val = indicator_values["rsi_20"][-1]
        rsi_100_val = indicator_values["rsi_100"][-1]
        rsi_20_prev = indicator_values["rsi_20"][-2]
        rsi_100_prev = indicator_values["rsi_100"][-2]

        if not np.isnan(rsi_20_val) and not np.isnan(rsi_100_val) and \
           not np.isnan(rsi_20_prev) and not np.isnan(rsi_100_prev):
             if rsi_20_val > rsi_100_val and rsi_20_prev <= rsi_100_prev:
                 rsi_cross = f"{NEON_GREEN}RSI 20/100:{RESET} RSI 20 crossed ABOVE RSI 100"
             elif rsi_20_val < rsi_100_val and rsi_20_prev >= rsi_100_prev:
                 rsi_cross = f"{NEON_RED}RSI 20/100:{RESET} RSI 20 crossed BELOW RSI 100"
             else:
                 rsi_cross = f"{NEON_YELLOW}RSI 20/100:{RESET} No recent cross"
        else:
             rsi_cross = f"{NEON_YELLOW}RSI 20/100:{RESET} Insufficient data for cross check"

        output += rsi_cross + "\n"

        # Display indicators
        for indicator_name in indicator_values:
             interpretation = self.interpret_indicator(self.logger, indicator_name, indicator_values[indicator_name])
             if interpretation:  # Only print if interpretation exists
                 output += interpretation + "\n"

        # Add MACD Divergence check
        macd_divergence = self.detect_macd_divergence(macd_df)
        if macd_divergence == "bullish":
            output += f"{NEON_GREEN}MACD Divergence:{RESET} Potential Bullish Divergence Detected\n"
        elif macd_divergence == "bearish":
            output += f"{NEON_RED}MACD Divergence:{RESET} Potential Bearish Divergence Detected\n"

        output += f"""
{NEON_BLUE}Support and Resistance Levels:{RESET}
"""
        for s_label, s_price in nearest_supports:
            output += f"S: {s_label} ${s_price:.4f}\n"
        for r_label, r_price in nearest_resistances:
            output += f"R: {r_label} ${r_price:.4f}\n"

        output += f"""
{NEON_BLUE}Orderbook Analysis near Fibonacci/Pivot Levels:{RESET}
{orderbook_analysis_str}
"""

        self.logger.info(output)

    def interpret_indicator(
        self, logger: logging.Logger, indicator_name: str, values: list  # Can contain lists or floats
    ) -> str | None:
        try:
            # --- Oscillators ---
            if indicator_name == "rsi_20":
                val = values[-1]
                if np.isnan(val): return f"{NEON_YELLOW}RSI 20:{RESET} Not Available"
                if val > 70: return f"{NEON_RED}RSI 20:{RESET} Overbought ({val:.2f})"
                if val < 30: return f"{NEON_GREEN}RSI 20:{RESET} Oversold ({val:.2f})"
                return f"{NEON_YELLOW}RSI 20:{RESET} Neutral ({val:.2f})"
            elif indicator_name == "rsi_100":
                val = values[-1]
                if np.isnan(val): return f"{NEON_YELLOW}RSI 100:{RESET} Not Available"
                if val > 70: return f"{NEON_RED}RSI 100:{RESET} Overbought ({val:.2f})"
                if val < 30: return f"{NEON_GREEN}RSI 100:{RESET} Oversold ({val:.2f})"
                return f"{NEON_YELLOW}RSI 100:{RESET} Neutral ({val:.2f})"
            elif indicator_name == "rsi_conf":  # Using config period
                val = values[-1]
                if np.isnan(val): return f"{NEON_YELLOW}RSI ({self.config['rsi_period']}):{RESET} Not Available"
                if val > 70: return f"{NEON_RED}RSI ({self.config['rsi_period']}):{RESET} Overbought ({val:.2f})"
                if val < 30: return f"{NEON_GREEN}RSI ({self.config['rsi_period']}):{RESET} Oversold ({val:.2f})"
                return f"{NEON_YELLOW}RSI ({self.config['rsi_period']}):{RESET} Neutral ({val:.2f})"
            elif indicator_name == "mfi":
                 val = values[-1]
                 if np.isnan(val): return f"{NEON_YELLOW}MFI:{RESET} Not Available"
                 if val > 80: return f"{NEON_RED}MFI:{RESET} Overbought ({val:.2f})"
                 if val < 20: return f"{NEON_GREEN}MFI:{RESET} Oversold ({val:.2f})"
                 return f"{NEON_YELLOW}MFI:{RESET} Neutral ({val:.2f})"
            elif indicator_name == "cci":
                val = values[-1]
                if np.isnan(val): return f"{NEON_YELLOW}CCI:{RESET} Not Available"
                if val > 100: return f"{NEON_RED}CCI:{RESET} Overbought ({val:.2f})"
                if val < -100: return f"{NEON_GREEN}CCI:{RESET} Oversold ({val:.2f})"
                return f"{NEON_YELLOW}CCI:{RESET} Neutral ({val:.2f})"
            elif indicator_name == "wr":
                 val = values[-1]
                 if np.isnan(val): return f"{NEON_YELLOW}Williams %R:{RESET} Not Available"
                 if val < -80: return f"{NEON_GREEN}Williams %R:{RESET} Oversold ({val:.2f})"
                 if val > -20: return f"{NEON_RED}Williams %R:{RESET} Overbought ({val:.2f})"
                 return f"{NEON_YELLOW}Williams %R:{RESET} Neutral ({val:.2f})"
            elif indicator_name == "stoch_rsi":
                k_val, d_val = values[1][-1], values[2][-1]
                if np.isnan(k_val) or np.isnan(d_val): return f"{NEON_YELLOW}Stoch RSI:{RESET} Not Available"
                cross = ""
                if not np.isnan(values[1][-2]) and not np.isnan(values[2][-2]):
                    if k_val > d_val and values[1][-2] <= values[2][-2]: cross = f" {NEON_GREEN}(K/D Bull Cross){RESET}"
                    if k_val < d_val and values[1][-2] >= values[2][-2]: cross = f" {NEON_RED}(K/D Bear Cross){RESET}"
                if k_val > 80 and d_val > 80: return f"{NEON_RED}Stoch RSI:{RESET} Overbought (K:{k_val:.2f}, D:{d_val:.2f}){cross}"
                if k_val < 20 and d_val < 20: return f"{NEON_GREEN}Stoch RSI:{RESET} Oversold (K:{k_val:.2f}, D:{d_val:.2f}){cross}"
                return f"{NEON_YELLOW}Stoch RSI:{RESET} Neutral (K:{k_val:.2f}, D:{d_val:.2f}){cross}"

            # --- Trend/Momentum ---
            elif indicator_name == "adx":
                val = values[0]  # ADX is single value
                if np.isnan(val): return f"{NEON_YELLOW}ADX:{RESET} Not Available"
                if val > 25: return f"{NEON_GREEN}ADX:{RESET} Trending ({val:.2f})"
                return f"{NEON_YELLOW}ADX:{RESET} Ranging/Weak Trend ({val:.2f})"
            elif indicator_name == "mom":  # Already calculated trend/strength
                trend = values[0]["trend"]
                strength = values[0]["strength"]
                return f"{NEON_PURPLE}Momentum:{RESET} {trend} (Strength: {strength:.2f})"
            elif indicator_name == "macd":
                 macd_line, signal_line, histogram = values[0][-1], values[1][-1], values[2][-1]
                 if np.isnan(macd_line) or np.isnan(signal_line) or np.isnan(histogram): return f"{NEON_YELLOW}MACD:{RESET} Not Available"
                 cross = ""
                 macd_prev, signal_prev = values[0][-2], values[1][-2]
                 if not np.isnan(macd_prev) and not np.isnan(signal_prev):
                      if macd_line > signal_line and macd_prev <= signal_prev: cross = f" {NEON_GREEN}(Bull Cross){RESET}"
                      if macd_line < signal_line and macd_prev >= signal_prev: cross = f" {NEON_RED}(Bear Cross){RESET}"
                 hist_color = NEON_GREEN if histogram > 0 else NEON_RED if histogram < 0 else NEON_YELLOW
                 return f"{NEON_BLUE}MACD:{RESET} Line={macd_line:.4f}, Signal={signal_line:.4f}, Hist={hist_color}{histogram:.4f}{RESET}{cross}"

            # --- Volume ---
            elif indicator_name == "obv":
                val, prev_val = values[-1], values[-2]
                if np.isnan(val) or np.isnan(prev_val): return f"{NEON_YELLOW}OBV:{RESET} Not Available"
                trend = NEON_GREEN + "Rising" if val > prev_val else NEON_RED + "Falling" if val < prev_val else NEON_YELLOW + "Flat"
                return f"{NEON_BLUE}OBV:{RESET} {trend}{RESET} ({val:,.0f})"
            elif indicator_name == "adi":
                val, prev_val = values[-1], values[-2]
                if np.isnan(val) or np.isnan(prev_val): return f"{NEON_YELLOW}ADI:{RESET} Not Available"
                trend = NEON_GREEN + "Accumulation" if val > prev_val else NEON_RED + "Distribution" if val < prev_val else NEON_YELLOW + "Neutral"
                return f"{NEON_BLUE}ADI:{RESET} {trend}{RESET} ({val:,.0f})"
            elif indicator_name == "fve":  # Force Index Value (Cumulative)
                val = values[-1]
                if np.isnan(val): return f"{NEON_YELLOW}FVE:{RESET} Not Available"
                return f"{NEON_BLUE}FVE:{RESET} {val:,.0f}"

            # --- Volatility / Bands ---
            elif indicator_name == "bollinger_bands":
                upper, middle, lower = values[0][-1], values[1][-1], values[2][-1]
                if np.isnan(upper) or np.isnan(middle) or np.isnan(lower): return f"{NEON_YELLOW}Bollinger Bands:{RESET} Not Available"
                last_close = self.closes[-1]
                if np.isnan(last_close): return f"{NEON_YELLOW}Bollinger Bands:{RESET} Price Not Available"

                band_width = ((upper - lower) / middle * 100) if middle != 0 else 0

                status = NEON_YELLOW + "Within Bands"
                if last_close > upper: status = NEON_RED + "Above Upper Band"
                if last_close < lower: status = NEON_GREEN + "Below Lower Band"
                return f"{NEON_BLUE}Bollinger Bands:{RESET} {status}{RESET} (U:{upper:.4f}, M:{middle:.4f}, L:{lower:.4f}, Width:{band_width:.2f}%)"

             # --- Other ---
            elif indicator_name == "psar":
                val = values[-1]
                last_close = self.closes[-1]
                if np.isnan(val) or np.isnan(last_close): return f"{NEON_YELLOW}PSAR:{RESET} Not Available"
                trend = NEON_GREEN + "Below Price (Uptrend)" if val < last_close else NEON_RED + "Above Price (Downtrend)"
                return f"{NEON_BLUE}PSAR:{RESET} {trend}{RESET} ({val:.4f})"
            elif indicator_name == "sma10":
                 val = values[0]  # Only care about the last value
                 if np.isnan(val): return f"{NEON_YELLOW}SMA (10):{RESET} Not Available"
                 return f"{NEON_YELLOW}SMA (10):{RESET} {val:.4f}"

            else:
                # This case should ideally not be reached if all indicators are handled
                logger.warning(f"Indicator interpretation not defined for: {indicator_name}")
                return None  # Return None for unhandled indicators

        except (TypeError, IndexError, KeyError) as e:
            logger.error(f"Error interpreting {indicator_name}: {e}. Values: {values}")
            return f"{indicator_name.upper()}: Interpretation error."
        except Exception as e:
            logger.exception(f"Unexpected error interpreting {indicator_name}: {e}")
            return f"{indicator_name.upper()}: Unexpected interpretation error."


def main() -> None:
    symbol = ""
    while True:
        try:
            symbol = input(f"{NEON_BLUE}Enter trading symbol (e.g., BTCUSDT): {RESET}").upper().strip()
            if symbol:
                # Basic validation: Check if it looks like a symbol pair
                if len(symbol) > 4 and any(c.isalpha() for c in symbol) and any(c.isdigit() or c.isalpha() for c in symbol):
                     break
                else:
                     pass
            else:
                 pass
        except EOFError:
             return

    interval = ""
    while True:
        try:
            interval_input = input(
                f"{NEON_BLUE}Enter timeframe (e.g., {', '.join(VALID_INTERVALS)}) [Default: {CONFIG.get('interval', '15')}]: {RESET}"
            ).strip()
            if not interval_input:
                interval = CONFIG.get("interval", "15")  # Use default from config or fallback
                break
            if interval_input in VALID_INTERVALS:
                interval = interval_input
                break
            else:
                 pass
        except EOFError:
             return

    logger = setup_logger(symbol)
    analysis_interval = CONFIG.get("analysis_interval", 30)  # Default 30s
    retry_delay = CONFIG.get("retry_delay", 5)  # Default 5s

    logger.info(f"Starting analysis for {symbol} on {interval} interval.")
    logger.info(f"Analysis refresh rate: {analysis_interval} seconds.")
    logger.info(f"Using configuration from: {CONFIG_FILE}")
    # Log key config parameters
    logger.info(f"Key Indicator Periods: RSI={CONFIG.get('rsi_period')}, BB={CONFIG.get('bollinger_bands_period')}, Mom={CONFIG.get('momentum_period')}, ATR={CONFIG.get('atr_period')}")

    while True:
        try:
            fetch_start_time = time.time()
            current_price = fetch_current_price(symbol, logger)
            if current_price is None:
                logger.error(
                    f"{NEON_RED}Failed to fetch current price. Retrying in {retry_delay} seconds...{RESET}"
                )
                time.sleep(retry_delay)
                continue

            # Determine limit needed based on longest lookback required
            # Example: Need 100 for RSI 100 + 1 for diff + some buffer = ~150-200
            kline_limit = max(200, CONFIG.get('rsi_period', 14) + 100, CONFIG.get('bollinger_bands_period', 20) + 50)  # Robust limit

            kline_data = fetch_klines(symbol, interval, limit=kline_limit, logger=logger)
            fetch_end_time = time.time()
            logger.info(f"Data fetch took {fetch_end_time - fetch_start_time:.2f} seconds.")

            if not kline_data:
                logger.error(
                    f"{NEON_RED}Failed to fetch sufficient kline data. Retrying in {retry_delay} seconds...{RESET}"
                )
                time.sleep(retry_delay)
                continue

            analyzer = TradingAnalyzer(kline_data, logger, CONFIG, symbol, interval)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")  # Add timezone
            analysis_start_time = time.time()
            analyzer.analyze(current_price, timestamp)
            analysis_end_time = time.time()
            logger.info(f"Analysis took {analysis_end_time - analysis_start_time:.2f} seconds.")

            # Calculate sleep time considering fetch and analysis time
            elapsed_time = time.time() - fetch_start_time
            sleep_duration = max(0, analysis_interval - elapsed_time)
            logger.info(f"Sleeping for {sleep_duration:.2f} seconds...")
            time.sleep(sleep_duration)

        except requests.exceptions.RequestException as e:
            logger.error(
                f"{NEON_RED}Network error: {e}. Retrying in {retry_delay} seconds...{RESET}"
            )
            time.sleep(retry_delay)
        except KeyboardInterrupt:
            logger.info(f"{NEON_YELLOW}Analysis stopped by user.{RESET}")
            break
        except Exception as e:
            logger.exception(
                f"{NEON_RED}An unexpected error occurred in the main loop: {e}. Retrying in {retry_delay} seconds...{RESET}"
            )
            time.sleep(retry_delay)


if __name__ == "__main__":
    main()
