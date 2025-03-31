
import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hmac
import hashlib
import time
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Union, Optional, Any
from colorama import init, Fore, Style
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext, InvalidOperation
import json
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ccxt
import ccxt.async_support as ccxt_async
import asyncio
import pandas_ta as ta
import traceback


init(autoreset=True)
load_dotenv()

getcontext().prec = 18

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
CCXT_URLS = {
    'test': 'https://api-testnet.bybit.com',
    'prod': 'https://api.bybit.com',
}
API_ENV = os.getenv("API_ENVIRONMENT", "prod").lower()
CCXT_BASE_URL = CCXT_URLS.get(API_ENV, CCXT_URLS['prod'])

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

DEFAULT_TIMEZONE = "America/Chicago"
try:
    TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", DEFAULT_TIMEZONE))
except Exception:
    print(f"{Fore.YELLOW}Warning: Could not load timezone '{os.getenv('TIMEZONE', DEFAULT_TIMEZONE)}'. Using UTC.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")

MAX_API_RETRIES = 5
RETRY_DELAY_SECONDS = 5
CCXT_TIMEOUT_MS = 20000
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M"
}
REVERSE_CCXT_INTERVAL_MAP = {v: k for k, v in CCXT_INTERVAL_MAP.items()}

NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
RESET = Style.RESET_ALL


def load_config(filepath: str) -> dict:
    default_config = {
        "analysis_interval_seconds": 30,
        "kline_limit": 200,
        "indicator_settings": {
            "default_interval": "15",
            "momentum_period": 10,
            "volume_ma_period": 20,
            "atr_period": 14,
            "rsi_period": 14,
            "stoch_rsi_period": 14,
            "stoch_k_period": 3,
            "stoch_d_period": 3,
            "cci_period": 20,
            "williams_r_period": 14,
            "mfi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_bands_period": 20,
            "bollinger_bands_std_dev": 2,
            "ema_short_period": 12,
            "ema_long_period": 26,
            "sma_short_period": 10,
            "sma_long_period": 50,
            "adx_period": 14,
            "psar_step": 0.02,
            "psar_max_step": 0.2,
        },
        "analysis_flags": {
            "ema_alignment": True,
            "momentum_crossover": False, # Requires more logic
            "volume_confirmation": True,
            "rsi_divergence": False,
            "macd_divergence": True,
            "stoch_rsi_cross": True,
            "rsi_threshold": True,
            "mfi_threshold": True,
            "cci_threshold": True,
            "williams_r_threshold": True,
            "macd_cross": True,
            "bollinger_bands_break": True,
            "adx_trend_strength": True,
            "obv_trend": True,
            "adi_trend": True,
            "psar_flip": True
        },
        "thresholds": {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "mfi_overbought": 80,
            "mfi_oversold": 20,
            "cci_overbought": 100,
            "cci_oversold": -100,
            "williams_r_overbought": -20,
            "williams_r_oversold": -80,
            "adx_trending": 25,
        },
        "orderbook_settings":{
            "limit": 50,
            "cluster_threshold_usd": 10000,
            "cluster_proximity_pct": 0.1
        },
        "logging": {
            "level": "INFO",
            "rotation_max_bytes": 10 * 1024 * 1024,
            "rotation_backup_count": 5
        }
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, 'w', encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            print(f"{NEON_YELLOW}Created new config file '{filepath}' with defaults.{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file: {e}{RESET}")
            print(f"{NEON_YELLOW}Loading internal defaults.{RESET}")
            return default_config

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            user_config = json.load(f)

            def merge_dicts(default: dict, user: dict) -> dict:
                merged = default.copy()
                for key, value in user.items():
                    if isinstance(value, dict) and isinstance(merged.get(key), dict):
                        merged[key] = merge_dicts(merged[key], value)
                    else:
                        merged[key] = value
                return merged

            merged_config = merge_dicts(default_config, user_config)
            return merged_config
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        print(f"{NEON_RED}Error loading/parsing config file '{filepath}': {e}{RESET}")
        print(f"{NEON_YELLOW}Loading internal defaults.{RESET}")
        return default_config

CONFIG = load_config(CONFIG_FILE)


class SensitiveFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(symbol: str) -> logging.Logger:
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.log")
    logger = logging.getLogger(symbol)
    log_level = getattr(logging, CONFIG.get('logging', {}).get('level', 'INFO').upper(), logging.INFO)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        max_bytes = int(CONFIG.get('logging', {}).get('rotation_max_bytes', 10*1024*1024))
        backup_count = int(CONFIG.get('logging', {}).get('rotation_backup_count', 5))

        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Failed to set up file logger: {e}{RESET}")

    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        NEON_BLUE + "%(asctime)s" + RESET + " [%(levelname)s] " + NEON_YELLOW + f"{symbol}" + RESET + " - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    return logger

logger: Optional[logging.Logger] = None


async def async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class BybitCCXTClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str, logger_instance: logging.Logger):
        self.logger = logger_instance
        self.exchange = ccxt_async.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', # Assuming linear contracts (USDT pairs)
                'adjustForTimeDifference': True,
            },
            'urls': {'api': base_url},
            'timeout': CCXT_TIMEOUT_MS,
        })
        self.markets = None

    async def initialize_markets(self, retries=3):
        for attempt in range(retries):
            try:
                self.markets = await self.exchange.load_markets()
                self.logger.info(f"Successfully loaded {len(self.markets)} markets from {self.exchange.name}.")
                return True
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                 self.logger.warning(f"Failed to load markets (Attempt {attempt + 1}/{retries}): {e}. Retrying...")
                 await async_sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            except ccxt.AuthenticationError as e:
                 self.logger.error(f"{NEON_RED}Authentication Error loading markets: {e}. Check API credentials.{RESET}")
                 return False
            except Exception as e:
                self.logger.exception(f"Unexpected error loading markets (Attempt {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    return False
                 await async_sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        return False

    def is_valid_symbol(self, symbol: str) -> bool:
        if self.markets is None:
            self.logger.warning("Markets not loaded, cannot validate symbol.")
            return False # Or True, depending on desired strictness before load
        return symbol in self.markets

    def get_symbol_details(self, symbol: str) -> Optional[dict]:
         if not self.is_valid_symbol(symbol):
             return None
         return self.markets.get(symbol)

    def get_market_category(self, symbol: str) -> str:
        details = self.get_symbol_details(symbol)
        if details:
            if details.get('linear'): return 'linear'
            if details.get('inverse'): return 'inverse'
            if details.get('spot'): return 'spot'
        # Default guess based on quote currency
        if symbol.endswith('/USDT'): return 'linear'
        if symbol.endswith('/USD'): return 'inverse'
        return 'spot' # Default fallback

    async def close(self):
        if self.exchange:
            try:
                await self.exchange.close()
                self.logger.info("Closed ccxt exchange connection.")
            except Exception as e:
                 self.logger.error(f"Error closing ccxt connection: {e}")

    async def fetch_with_retry(self, method_name: str, *args, **kwargs) -> Optional[Any]:
        retries = MAX_API_RETRIES
        current_delay = RETRY_DELAY_SECONDS
        for attempt in range(retries):
            try:
                method = getattr(self.exchange, method_name)
                result = await method(*args, **kwargs)
                return result
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
                self.logger.warning(f"{NEON_YELLOW}Network/Timeout/Rate Limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}")
            except ccxt.RateLimitExceeded as e:
                 self.logger.warning(f"{NEON_YELLOW}Explicit Rate limit exceeded calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}")
            except ccxt.ExchangeError as e:
                error_str = str(e).lower()
                if 'too many visits' in error_str or 'system busy' in error_str or 'service unavailable' in error_str:
                     self.logger.warning(f"{NEON_YELLOW}Server busy/rate limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}")
                else:
                    self.logger.error(f"{NEON_RED}Non-retryable ccxt ExchangeError calling {method_name}: {e}{RESET}")
                    return None
            except Exception as e:
                 self.logger.exception(f"{NEON_RED}Unexpected error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}\nArgs: {args}\nKwargs: {kwargs}{RESET}")

            if attempt < retries - 1:
                await async_sleep(current_delay)
                current_delay = min(current_delay * 1.5, 60)
            else:
                self.logger.error(f"{NEON_RED}Max retries reached for {method_name}.{RESET}")
                return None
        return None

    async def fetch_ticker(self, symbol: str) -> Optional[dict]:
        self.logger.debug(f"Fetching ticker for {symbol}")
        category = self.get_market_category(symbol)
        tickers = await self.fetch_with_retry('fetch_tickers', symbols=[symbol], params={'category': category})
        if tickers and symbol in tickers:
            return tickers[symbol]
        self.logger.error(f"Could not fetch ticker for {symbol} (category: {category})")
        return None

    async def fetch_current_price(self, symbol: str) -> Optional[Decimal]:
        ticker = await self.fetch_ticker(symbol)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            try:
                return Decimal(str(ticker['last']))
            except (InvalidOperation, TypeError) as e:
                self.logger.error(f"Error converting last price '{ticker['last']}' to Decimal for {symbol}: {e}")
                return None
        self.logger.warning(f"Last price not found in ticker data for {symbol}")
        return None

    async def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            self.logger.error(f"Invalid timeframe '{timeframe}' provided.")
            return pd.DataFrame()

        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching {limit} klines for {symbol} with interval {ccxt_timeframe} (category: {category})")
        klines = await self.fetch_with_retry('fetch_ohlcv', symbol, timeframe=ccxt_timeframe, limit=limit, params={'category': category})

        if klines is None or len(klines) == 0:
            self.logger.warning(f"No kline data returned for {symbol} interval {ccxt_timeframe}")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                 self.logger.warning(f"Kline data for {symbol} was empty initially.")
                 return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(TIMEZONE)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            initial_rows = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                 self.logger.warning(f"Dropped {dropped_rows} rows with NaN values from kline data for {symbol}.")

            if df.empty:
                 self.logger.warning(f"Kline data for {symbol} is empty after cleaning.")
                 return pd.DataFrame()

            self.logger.debug(f"Successfully fetched and processed {len(df)} klines for {symbol}.")
            return df.sort_values(by='timestamp').reset_index(drop=True)

        except Exception as e:
            self.logger.exception(f"Error processing kline data for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_orderbook(self, symbol: str, limit: int) -> Optional[dict]:
        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching order book for {symbol} with limit {limit} (category: {category})")
        orderbook = await self.fetch_with_retry('fetch_order_book', symbol, limit=limit, params={'category': category})
        if orderbook:
            if 'bids' in orderbook and 'asks' in orderbook:
                self.logger.debug(f"Fetched order book for {symbol} with {len(orderbook['bids'])} bids and {len(orderbook['asks'])} asks.")
                return orderbook
            else:
                self.logger.warning(f"Fetched order book data for {symbol} is missing bids or asks.")
                return None
        else:
            self.logger.warning(f"Failed to fetch order book for {symbol} after retries.")
            return None


class TradingAnalyzer:
    def __init__(self, config: dict, logger_instance: logging.Logger, symbol: str):
        self.config = config
        self.logger = logger_instance
        self.symbol = symbol
        self.indicator_settings = config.get("indicator_settings", {})
        self.analysis_flags = config.get("analysis_flags", {})
        self.thresholds = config.get("thresholds", {})
        self.orderbook_settings = config.get("orderbook_settings", {})

        # Dynamically generate expected column names based on config
        self._generate_column_names()

    def _generate_column_names(self):
        """Generates expected indicator column names based on config."""
        self.col_names = {}
        is_ = self.indicator_settings
        self.col_names['sma_short'] = f"SMA_{is_.get('sma_short_period', 10)}"
        self.col_names['sma_long'] = f"SMA_{is_.get('sma_long_period', 50)}"
        self.col_names['ema_short'] = f"EMA_{is_.get('ema_short_period', 12)}"
        self.col_names['ema_long'] = f"EMA_{is_.get('ema_long_period', 26)}"
        self.col_names['rsi'] = f"RSI_{is_.get('rsi_period', 14)}"
        self.col_names['stochrsi_k'] = f"STOCHRSIk_{is_.get('stoch_rsi_period', 14)}_{is_.get('rsi_period', 14)}_{is_.get('stoch_k_period', 3)}_{is_.get('stoch_d_period', 3)}"
        self.col_names['stochrsi_d'] = f"STOCHRSId_{is_.get('stoch_rsi_period', 14)}_{is_.get('rsi_period', 14)}_{is_.get('stoch_k_period', 3)}_{is_.get('stoch_d_period', 3)}"
        self.col_names['macd_line'] = f"MACD_{is_.get('macd_fast', 12)}_{is_.get('macd_slow', 26)}_{is_.get('macd_signal', 9)}"
        self.col_names['macd_signal'] = f"MACDs_{is_.get('macd_fast', 12)}_{is_.get('macd_slow', 26)}_{is_.get('macd_signal', 9)}"
        self.col_names['macd_hist'] = f"MACDh_{is_.get('macd_fast', 12)}_{is_.get('macd_slow', 26)}_{is_.get('macd_signal', 9)}"
        self.col_names['bb_upper'] = f"BBU_{is_.get('bollinger_bands_period', 20)}_{float(is_.get('bollinger_bands_std_dev', 2))}" # Ensure std dev is float for name matching
        self.col_names['bb_lower'] = f"BBL_{is_.get('bollinger_bands_period', 20)}_{float(is_.get('bollinger_bands_std_dev', 2))}"
        self.col_names['bb_mid'] = f"BBM_{is_.get('bollinger_bands_period', 20)}_{float(is_.get('bollinger_bands_std_dev', 2))}"
        self.col_names['atr'] = f"ATRr_{is_.get('atr_period', 14)}" # pandas_ta uses ATRr for True Range average
        self.col_names['cci'] = f"CCI_{is_.get('cci_period', 20)}_0.015" # Default c=0.015 in pandas_ta
        self.col_names['willr'] = f"WILLR_{is_.get('williams_r_period', 14)}"
        self.col_names['mfi'] = f"MFI_{is_.get('mfi_period', 14)}"
        self.col_names['adx'] = f"ADX_{is_.get('adx_period', 14)}"
        self.col_names['dmp'] = f"DMP_{is_.get('adx_period', 14)}"
        self.col_names['dmn'] = f"DMN_{is_.get('adx_period', 14)}"
        self.col_names['obv'] = "OBV"
        self.col_names['adosc'] = "ADOSC" # Accumulation/Distribution Oscillator
        psar_step = is_.get('psar_step', 0.02)
        psar_max = is_.get('psar_max_step', 0.2)
        self.col_names['psar_long'] = f"PSARl_{psar_step}_{psar_max}" # PSAR when price is above (potential support)
        self.col_names['psar_short'] = f"PSARs_{psar_step}_{psar_max}" # PSAR when price is below (potential resistance)
        self.col_names['psar_af'] = f"PSARaf_{psar_step}_{psar_max}"
        self.col_names['psar_rev'] = f"PSARr_{psar_step}_{psar_max}"

        self.col_names['mom'] = f"MOM_{is_.get('momentum_period', 10)}"
        self.col_names['vol_ma'] = f"VOL_MA_{is_.get('volume_ma_period', 20)}"

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("Cannot calculate indicators on empty DataFrame.")
            return df
        if len(df) < self.indicator_settings.get('ema_long_period', 26): # Basic check for sufficient data
             self.logger.warning(f"Insufficient data points ({len(df)}) for some indicators. Results may be inaccurate.")


        df = df.copy()
        is_ = self.indicator_settings

        strategy_ta = [
            {"kind": "sma", "length": is_.get("sma_short_period", 10)},
            {"kind": "sma", "length": is_.get("sma_long_period", 50)},
            {"kind": "ema", "length": is_.get("ema_short_period", 12)},
            {"kind": "ema", "length": is_.get("ema_long_period", 26)},
            {"kind": "rsi", "length": is_.get("rsi_period", 14)},
            {
                "kind": "stochrsi", "length": is_.get("stoch_rsi_period", 14),
                "rsi_length": is_.get("rsi_period", 14), "k": is_.get("stoch_k_period", 3),
                "d": is_.get("stoch_d_period", 3)
            },
            {
                "kind": "macd", "fast": is_.get("macd_fast", 12),
                "slow": is_.get("macd_slow", 26), "signal": is_.get("macd_signal", 9)
            },
            {
                "kind": "bbands", "length": is_.get("bollinger_bands_period", 20),
                "std": is_.get("bollinger_bands_std_dev", 2)
            },
            {"kind": "atr", "length": is_.get("atr_period", 14)},
            {"kind": "cci", "length": is_.get("cci_period", 20)},
            {"kind": "willr", "length": is_.get("williams_r_period", 14)},
            {"kind": "mfi", "length": is_.get("mfi_period", 14)},
            {"kind": "adx", "length": is_.get("adx_period", 14)},
            {"kind": "obv"},
            {"kind": "adosc"},
            {
                "kind": "psar", "step": is_.get("psar_step", 0.02),
                "max_step": is_.get("psar_max_step", 0.2)
            },
        ]

        strategy = ta.Strategy(
            name="TradingAnalysis",
            description="Comprehensive TA using pandas_ta",
            ta=strategy_ta
        )

        try:
           df.ta.strategy(strategy, timed=False)
           self.logger.debug(f"Calculated pandas_ta indicators for {self.symbol}.")
        except Exception as e:
            self.logger.exception(f"Error calculating indicators using pandas_ta for {self.symbol}: {e}")
            # Don't return yet, try custom calcs

        # Custom calculations
        try:
            mom_period = is_.get("momentum_period", 10)
            if mom_period > 0 and 'close' in df.columns:
                df[self.col_names['mom']] = ta.mom(df['close'], length=mom_period)

            vol_ma_period = is_.get("volume_ma_period", 20)
            if vol_ma_period > 0 and 'volume' in df.columns:
                 df[self.col_names['vol_ma']] = ta.sma(df['volume'], length=vol_ma_period)
        except Exception as e:
            self.logger.exception(f"Error calculating custom indicators for {self.symbol}: {e}")

        # Fill potentially generated PSAR long/short columns if one is all NaN
        psar_l = self.col_names['psar_long']
        psar_s = self.col_names['psar_short']
        if psar_l in df.columns and psar_s in df.columns:
            if df[psar_l].isnull().all():
                df[psar_l] = df[psar_s]
            elif df[psar_s].isnull().all():
                 df[psar_s] = df[psar_l]

        return df


    def _calculate_levels(self, df: pd.DataFrame, current_price: Decimal) -> dict:
        levels = {"support": {}, "resistance": {}, "pivot": None}
        if df.empty or len(df) < 2 :
            self.logger.warning("Insufficient data for level calculation.")
            return levels
        # Use .iloc[-1] for most recent, handle potential index issues
        try:
            high = df["high"].max()
            low = df["low"].min()
            close = df["close"].iloc[-1]
            open_ = df["open"].iloc[-1] # Use current period open for some PPs if desired

            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                 self.logger.warning("NaN values found in OHLC data, cannot calculate levels accurately.")
                 return levels

            high_f = float(high)
            low_f = float(low)
            close_f = float(close)
            current_price_f = float(current_price)

            diff = high_f - low_f
            if diff > 1e-9:
                fib_levels = {
                    "Fib 23.6%": high_f - diff * 0.236, "Fib 38.2%": high_f - diff * 0.382,
                    "Fib 50.0%": high_f - diff * 0.5, "Fib 61.8%": high_f - diff * 0.618,
                    "Fib 78.6%": high_f - diff * 0.786,
                }
                for label, value in fib_levels.items():
                    try:
                        value_dec = Decimal(str(value))
                        if value_dec < current_price: levels["support"][label] = value_dec
                        else: levels["resistance"][label] = value_dec
                    except InvalidOperation: continue # Skip invalid levels

            try:
                pivot = (high_f + low_f + close_f) / 3
                levels["pivot"] = Decimal(str(pivot))
                pivot_levels = {
                    "R1": (2 * pivot) - low_f, "S1": (2 * pivot) - high_f,
                    "R2": pivot + (high_f - low_f), "S2": pivot - (high_f - low_f),
                    "R3": high_f + 2 * (pivot - low_f), "S3": low_f - 2 * (high_f - pivot),
                }
                for label, value in pivot_levels.items():
                     try:
                         value_dec = Decimal(str(value))
                         if value_dec < current_price: levels["support"][label] = value_dec
                         else: levels["resistance"][label] = value_dec
                     except InvalidOperation: continue
            except ZeroDivisionError:
                self.logger.warning("Zero division encountered calculating pivot points.")
            except InvalidOperation as e:
                self.logger.error(f"Invalid operation during pivot calculation: {e}")


        except (TypeError, ValueError, InvalidOperation, IndexError) as e:
            self.logger.error(f"Error calculating levels for {self.symbol}: {e}")
        except Exception as e:
             self.logger.exception(f"Unexpected error calculating levels for {self.symbol}: {e}")

        return levels

    def _analyze_orderbook(self, orderbook: dict, current_price: Decimal, levels: dict) -> dict:
        analysis = {"clusters": [], "pressure": "Neutral", "total_bid_usd": Decimal(0), "total_ask_usd": Decimal(0)}
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook or not orderbook['bids'] or not orderbook['asks']:
            self.logger.debug("Orderbook data incomplete or unavailable for analysis.")
            return analysis

        try:
            bids = pd.DataFrame(orderbook.get('bids', []), columns=['price', 'size'], dtype=str)
            asks = pd.DataFrame(orderbook.get('asks', []), columns=['price', 'size'], dtype=str)

            if bids.empty and asks.empty:
                return analysis

            # Convert, coerce errors, fill NaNs with 0
             bids['price'] = pd.to_numeric(bids['price'], errors='coerce').fillna(0).apply(Decimal)
             bids['size'] = pd.to_numeric(bids['size'], errors='coerce').fillna(0).apply(Decimal)
             asks['price'] = pd.to_numeric(asks['price'], errors='coerce').fillna(0).apply(Decimal)
             asks['size'] = pd.to_numeric(asks['size'], errors='coerce').fillna(0).apply(Decimal)


            bids = bids[bids['price'] > 0] # Remove zero prices if any survived
            asks = asks[asks['price'] > 0]

            if bids.empty and asks.empty:
                return analysis

            bids['value_usd'] = bids['price'] * bids['size']
            asks['value_usd'] = asks['price'] * asks['size']

            total_bid_value = bids['value_usd'].sum()
            total_ask_value = asks['value_usd'].sum()
            analysis["total_bid_usd"] = total_bid_value
            analysis["total_ask_usd"] = total_ask_value

            total_value = total_bid_value + total_ask_value
            if total_value > 0:
                bid_ask_ratio = total_bid_value / total_value
                if bid_ask_ratio > 0.6: analysis["pressure"] = f"{NEON_GREEN}High Buy Pressure{RESET}"
                elif bid_ask_ratio < 0.4: analysis["pressure"] = f"{NEON_RED}High Sell Pressure{RESET}"
                else: analysis["pressure"] = f"{NEON_YELLOW}Neutral Pressure{RESET}"

            cluster_threshold = Decimal(str(self.orderbook_settings.get("cluster_threshold_usd", 10000)))
            proximity_pct = Decimal(str(self.orderbook_settings.get("cluster_proximity_pct", 0.1))) / 100

            all_levels = {**levels.get("support", {}), **levels.get("resistance", {})}
            if levels.get("pivot"): all_levels["Pivot"] = levels["pivot"]

            processed_clusters = set()

            for name, level_price in all_levels.items():
                if not isinstance(level_price, Decimal) or level_price <= 0: continue

                min_price = level_price * (Decimal(1) - proximity_pct)
                max_price = level_price * (Decimal(1) + proximity_pct)

                # Bid clusters (support)
                bids_near = bids[(bids['price'] >= min_price) & (bids['price'] <= max_price)]
                bid_cluster_value = bids_near['value_usd'].sum()
                if bid_cluster_value >= cluster_threshold:
                    cluster_id = f"B_{name}_{level_price:.4f}"
                    if cluster_id not in processed_clusters:
                        analysis["clusters"].append({
                            "type": "Support", "level_name": name, "level_price": level_price,
                            "cluster_value_usd": bid_cluster_value, "price_range": (min_price, max_price)
                        })
                        processed_clusters.add(cluster_id)

                # Ask clusters (resistance)
                asks_near = asks[(asks['price'] >= min_price) & (asks['price'] <= max_price)]
                ask_cluster_value = asks_near['value_usd'].sum()
                if ask_cluster_value >= cluster_threshold:
                     cluster_id = f"A_{name}_{level_price:.4f}"
                     if cluster_id not in processed_clusters:
                        analysis["clusters"].append({
                            "type": "Resistance", "level_name": name, "level_price": level_price,
                            "cluster_value_usd": ask_cluster_value, "price_range": (min_price, max_price)
                        })
                        processed_clusters.add(cluster_id)

        except (KeyError, ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Error analyzing orderbook for {self.symbol}: {e}")
            self.logger.debug(traceback.format_exc())
        except Exception as e:
            self.logger.exception(f"Unexpected error analyzing orderbook for {self.symbol}: {e}")

        return analysis

    # --- Interpretation Helpers ---
    def _format_val(self, value, precision=2):
        if pd.isna(value): return "N/A"
        try:
            # Use Decimal formatting for precision if it's a Decimal
            if isinstance(value, Decimal):
                # Dynamically adjust precision for very small numbers if needed
                 # return f"{value:.{precision}f}" # Standard f-string formatting
                 # Using quantize for proper Decimal rounding
                 quantizer = Decimal('1e-' + str(precision))
                 return str(value.quantize(quantizer))

            # Fallback for float/int
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError, InvalidOperation):
            return str(value)

    def _get_val(self, row, key, default=np.nan):
        return row.get(key, default)


    def _interpret_trend(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

         # EMA Alignment
        if flags.get("ema_alignment") and cols['ema_short'] in last_row and cols['ema_long'] in last_row:
            ema_short = self._get_val(last_row, cols['ema_short'])
            ema_long = self._get_val(last_row, cols['ema_long'])
            if not pd.isna(ema_short) and not pd.isna(ema_long):
                curr_state = "Neutral"
                color = NEON_YELLOW
                if ema_short > ema_long: curr_state, color = "Bullish", NEON_GREEN
                elif ema_short < ema_long: curr_state, color = "Bearish", NEON_RED
                signals["ema_trend"] = curr_state
                summary.append(f"{color}EMA Alignment: {curr_state}{RESET} (S:{self._format_val(ema_short)} {'><'[ema_short > ema_long]} L:{self._format_val(ema_long)})")

        # ADX Trend Strength
        if flags.get("adx_trend_strength") and cols['adx'] in last_row:
             adx = self._get_val(last_row, cols['adx'])
             dmp = self._get_val(last_row, cols.get('dmp'), 0) # DMP column might not exist if ADX fails
             dmn = self._get_val(last_row, cols.get('dmn'), 0)
             trend_threshold = self.thresholds.get("adx_trending", 25)
             if not pd.isna(adx) and not pd.isna(dmp) and not pd.isna(dmn):
                adx_str = f"ADX ({self._format_val(adx)}): "
                if adx >= trend_threshold:
                    if dmp > dmn:
                        adx_str += f"{NEON_GREEN}Strong Uptrend (+DI > -DI){RESET}"
                        signals["adx_trend"] = "Strong Bullish"
                    else:
                        adx_str += f"{NEON_RED}Strong Downtrend (-DI > +DI){RESET}"
                        signals["adx_trend"] = "Strong Bearish"
                else:
                     adx_str += f"{NEON_YELLOW}Weak/Ranging Trend{RESET}"
                     signals["adx_trend"] = "Ranging"
                summary.append(adx_str)
             else:
                  signals["adx_trend"] = "N/A"

        # PSAR Flip
        if flags.get("psar_flip"):
            psar_val = np.nan
            # PSAR value is in 'long' column if price > psar, 'short' if price < psar
            psar_l_val = self._get_val(last_row, cols['psar_long'])
            psar_s_val = self._get_val(last_row, cols['psar_short'])

            if not pd.isna(psar_l_val): psar_val = psar_l_val
            elif not pd.isna(psar_s_val): psar_val = psar_s_val

            close_val = self._get_val(last_row, 'close')

            if not pd.isna(psar_val) and not pd.isna(close_val):
                 is_bullish_psar = close_val > psar_val
                 signals["psar_trend"] = "Bullish" if is_bullish_psar else "Bearish"

                 # Check for flip using the reversal column 'PSARr'
                 psar_rev = self._get_val(last_row, cols.get('psar_rev')) # May not exist if calculation failed
                 flipped = not pd.isna(psar_rev) and psar_rev == 1 # Check if reversal happened on this candle

                 psar_status = f"PSAR ({self._format_val(psar_val, precision=4)}): "
                 if is_bullish_psar:
                     psar_status += f"{NEON_GREEN}Price Above (Bullish){RESET}"
                     if flipped:
                         psar_status += f" {NEON_GREEN}(Just Flipped Bullish){RESET}"
                         signals["psar_signal"] = "Flip Bullish"
                 else:
                     psar_status += f"{NEON_RED}Price Below (Bearish){RESET}"
                     if flipped:
                         psar_status += f" {NEON_RED}(Just Flipped Bearish){RESET}"
                         signals["psar_signal"] = "Flip Bearish"

                 summary.append(psar_status)
            else:
                 signals["psar_trend"] = "N/A"

        return summary, signals

    def _interpret_oscillators(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        thresh = self.thresholds
        cols = self.col_names

        # RSI
        if flags.get("rsi_threshold") and cols['rsi'] in last_row:
            rsi = self._get_val(last_row, cols['rsi'])
            ob = thresh.get("rsi_overbought", 70)
            os = thresh.get("rsi_oversold", 30)
            if not pd.isna(rsi):
                rsi_status = f"RSI ({self._format_val(rsi)}): "
                level = "Neutral"
                color = NEON_YELLOW
                if rsi >= ob: level, color = "Overbought", NEON_RED
                elif rsi <= os: level, color = "Oversold", NEON_GREEN
                signals["rsi_level"] = level
                summary.append(rsi_status + f"{color}{level}{RESET}")
            else: signals["rsi_level"] = "N/A"

        # MFI
        if flags.get("mfi_threshold") and cols['mfi'] in last_row:
            mfi = self._get_val(last_row, cols['mfi'])
            ob = thresh.get("mfi_overbought", 80)
            os = thresh.get("mfi_oversold", 20)
            if not pd.isna(mfi):
                mfi_status = f"MFI ({self._format_val(mfi)}): "
                level = "Neutral"
                color = NEON_YELLOW
                if mfi >= ob: level, color = "Overbought", NEON_RED
                elif mfi <= os: level, color = "Oversold", NEON_GREEN
                signals["mfi_level"] = level
                summary.append(mfi_status + f"{color}{level}{RESET}")
            else: signals["mfi_level"] = "N/A"


        # CCI
        if flags.get("cci_threshold") and cols['cci'] in last_row:
             cci = self._get_val(last_row, cols['cci'])
             ob = thresh.get("cci_overbought", 100)
             os = thresh.get("cci_oversold", -100)
             if not pd.isna(cci):
                 cci_status = f"CCI ({self._format_val(cci)}): "
                 level, color = "Neutral", NEON_YELLOW
                 if cci >= ob: level, color = "Overbought", NEON_RED
                 elif cci <= os: level, color = "Oversold", NEON_GREEN
                 signals["cci_level"] = level
                 summary.append(cci_status + f"{color}{level} Zone{RESET}")
             else: signals["cci_level"] = "N/A"


        # Williams %R
        if flags.get("williams_r_threshold") and cols['willr'] in last_row:
             wr = self._get_val(last_row, cols['willr'])
             ob = thresh.get("williams_r_overbought", -20)
             os = thresh.get("williams_r_oversold", -80)
             if not pd.isna(wr):
                 wr_status = f"Williams %R ({self._format_val(wr)}): "
                 level, color = "Neutral", NEON_YELLOW
                 if wr >= ob: level, color = "Overbought", NEON_RED # Higher value is OB
                 elif wr <= os: level, color = "Oversold", NEON_GREEN # Lower value is OS
                 signals["wr_level"] = level
                 summary.append(wr_status + f"{color}{level}{RESET}")
             else: signals["wr_level"] = "N/A"

        # StochRSI Cross
        if flags.get("stoch_rsi_cross") and cols['stochrsi_k'] in last_row and cols['stochrsi_d'] in last_row:
            k_now = self._get_val(last_row, cols['stochrsi_k'])
            d_now = self._get_val(last_row, cols['stochrsi_d'])
            k_prev = self._get_val(prev_row, cols['stochrsi_k'])
            d_prev = self._get_val(prev_row, cols['stochrsi_d'])

            if not pd.isna(k_now) and not pd.isna(d_now) and not pd.isna(k_prev) and not pd.isna(d_prev):
                stoch_status = f"StochRSI (K:{self._format_val(k_now)}, D:{self._format_val(d_now)}): "
                cross = "Neutral"
                color = NEON_YELLOW
                if k_now > d_now and k_prev <= d_prev: cross, color = "Bullish", NEON_GREEN
                elif k_now < d_now and k_prev >= d_prev: cross, color = "Bearish", NEON_RED
                elif k_now > d_now: cross, color = "Bullish", NEON_GREEN # Indicate current state if no cross
                elif k_now < d_now: cross, color = "Bearish", NEON_RED # Indicate current state if no cross

                signals["stochrsi_cross"] = cross
                cross_text = f"{cross} Cross" if 'Cross' in stoch_status else cross
                summary.append(stoch_status + f"{color}{cross_text}{RESET}" if cross != "Neutral" else stoch_status + f"{NEON_YELLOW}No recent cross{RESET}")

            else: signals["stochrsi_cross"] = "N/A"

        return summary, signals

    def _interpret_macd(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

        # MACD Cross
        if flags.get("macd_cross") and cols['macd_line'] in last_row and cols['macd_signal'] in last_row:
            line_now = self._get_val(last_row, cols['macd_line'])
            sig_now = self._get_val(last_row, cols['macd_signal'])
            line_prev = self._get_val(prev_row, cols['macd_line'])
            sig_prev = self._get_val(prev_row, cols['macd_signal'])

            if not pd.isna(line_now) and not pd.isna(sig_now) and not pd.isna(line_prev) and not pd.isna(sig_prev):
                macd_status = f"MACD (L:{self._format_val(line_now, 4)}, S:{self._format_val(sig_now, 4)}): "
                signal = "Neutral"
                color = NEON_YELLOW
                if line_now > sig_now and line_prev <= sig_prev: signal, color = "Bullish", NEON_GREEN
                elif line_now < sig_now and line_prev >= sig_prev: signal, color = "Bearish", NEON_RED
                elif line_now > sig_now: signal = "Above Signal" # Current state
                elif line_now < sig_now: signal = "Below Signal" # Current state

                signals["macd_cross"] = signal
                cross_text = f"{signal} Cross" if signal in ["Bullish", "Bearish"] else signal
                text_color = NEON_GREEN if "Bullish" in signal or "Above" in signal else NEON_RED if "Bearish" in signal or "Below" in signal else NEON_YELLOW
                summary.append(macd_status + f"{text_color}{cross_text}{RESET}")

            else: signals["macd_cross"] = "N/A"

        # MACD Divergence (Basic 2-point check)
        # WARNING: Simple check, prone to false signals. Real divergence needs pattern recognition.
        if flags.get("macd_divergence") and cols['macd_hist'] in last_row and 'close' in last_row:
             hist_now = self._get_val(last_row, cols['macd_hist'])
             hist_prev = self._get_val(prev_row, cols['macd_hist'])
             price_now = self._get_val(last_row, 'close')
             price_prev = self._get_val(prev_row, 'close')

             if not pd.isna(hist_now) and not pd.isna(hist_prev) and not pd.isna(price_now) and not pd.isna(price_prev):
                 divergence = "None"
                 # Bullish: Lower low in price, higher low in histogram (Must cross zero or be below zero)
                 if price_now < price_prev and hist_now > hist_prev and (hist_prev < 0 or hist_now < 0):
                     divergence = "Bullish"
                     summary.append(f"{NEON_GREEN}Potential Bullish MACD Divergence{RESET}")
                 # Bearish: Higher high in price, lower high in histogram (Must cross zero or be above zero)
                 elif price_now > price_prev and hist_now < hist_prev and (hist_prev > 0 or hist_now > 0):
                     divergence = "Bearish"
                     summary.append(f"{NEON_RED}Potential Bearish MACD Divergence{RESET}")
                 signals["macd_divergence"] = divergence
             else:
                  signals["macd_divergence"] = "N/A"

        return summary, signals

    def _interpret_bbands(self, last_row: pd.Series) -> Tuple[List[str], Dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

        if flags.get("bollinger_bands_break") and cols['bb_upper'] in last_row and cols['bb_lower'] in last_row:
             upper = self._get_val(last_row, cols['bb_upper'])
             lower = self._get_val(last_row, cols['bb_lower'])
             middle = self._get_val(last_row, cols['bb_mid'])
             close_val = self._get_val(last_row, 'close')

             if not pd.isna(upper) and not pd.isna(lower) and not pd.isna(middle) and not pd.isna(close_val):
                bb_status = f"BB (U:{self._format_val(upper)}, M:{self._format_val(middle)}, L:{self._format_val(lower)}): "
                signal = "Within Bands"
                color = NEON_YELLOW
                if close_val > upper:
                    signal, color = "Breakout Upper", NEON_RED
                    bb_status += f"{color}Price Above Upper Band{RESET}"
                elif close_val < lower:
                     signal, color = "Breakdown Lower", NEON_GREEN
                     bb_status += f"{color}Price Below Lower Band{RESET}"
                else:
                     bb_status += f"{color}Price Within Bands{RESET}"
                signals["bbands_signal"] = signal
                summary.append(bb_status)
             else:
                 signals["bbands_signal"] = "N/A"

        return summary, signals

    def _interpret_volume(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names

        # Volume MA Comparison
        if flags.get("volume_confirmation") and cols['vol_ma'] in last_row:
            volume = self._get_val(last_row, 'volume')
            vol_ma = self._get_val(last_row, cols['vol_ma'])
            if not pd.isna(volume) and not pd.isna(vol_ma) and vol_ma > 0:
                vol_status = f"Volume ({self._format_val(volume,0)} vs MA:{self._format_val(vol_ma,0)}): "
                level = "Average"
                if volume > vol_ma * Decimal("1.5"): level = "High" # Increased threshold
                elif volume < vol_ma * Decimal("0.7"): level = "Low"
                signals["volume_level"] = level
                summary.append(vol_status + f"{NEON_PURPLE}{level} Volume{RESET}")
            else: signals["volume_level"] = "N/A"

        # OBV Trend
        if flags.get("obv_trend") and cols['obv'] in last_row:
             obv_now = self._get_val(last_row, cols['obv'])
             obv_prev = self._get_val(prev_row, cols['obv'])
             if not pd.isna(obv_now) and not pd.isna(obv_prev):
                 trend = "Flat"
                 color = NEON_YELLOW
                 if obv_now > obv_prev: trend, color = "Increasing", NEON_GREEN
                 elif obv_now < obv_prev: trend, color = "Decreasing", NEON_RED
                 signals["obv_trend"] = trend
                 summary.append(f"OBV Trend: {color}{trend}{RESET}")
             else: signals["obv_trend"] = "N/A"

        # ADOSC Trend (A/D Oscillator)
        if flags.get("adi_trend") and cols['adosc'] in last_row:
             adi_now = self._get_val(last_row, cols['adosc'])
             adi_prev = self._get_val(prev_row, cols['adosc'])
             if not pd.isna(adi_now) and not pd.isna(adi_prev):
                 trend = "Flat"
                 color = NEON_YELLOW
                 if adi_now > adi_prev and adi_now > 0: trend, color = "Accumulation", NEON_GREEN # Basic Accumulation/Distribution interpretation
                 elif adi_now < adi_prev and adi_now < 0: trend, color = "Distribution", NEON_RED
                 elif adi_now > adi_prev: trend, color = "Increasing", NEON_GREEN # Generic trend if not clearly A/D
                 elif adi_now < adi_prev: trend, color = "Decreasing", NEON_RED
                 signals["adi_trend"] = trend
                 summary.append(f"A/D Osc Trend: {color}{trend}{RESET}")
             else: signals["adi_trend"] = "N/A"

        return summary, signals


    def _interpret_levels_orderbook(self, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> List[str]:
        summary = []
        summary.append(f"{NEON_BLUE}--- Levels & Orderbook ---{RESET}")

        if levels:
             nearest_supports = sorted(levels.get("support", {}).items(), key=lambda item: abs(item[1] - current_price))[:3]
             nearest_resistances = sorted(levels.get("resistance", {}).items(), key=lambda item: abs(item[1] - current_price))[:3]
             pivot = levels.get("pivot")
             if pivot: summary.append(f"Pivot Point: ${self._format_val(pivot, 4)}")
             if nearest_supports: summary.append("Nearest Support:")
             for name, price in nearest_supports: summary.append(f"  > {name}: ${self._format_val(price, 4)}")
             if nearest_resistances: summary.append("Nearest Resistance:")
             for name, price in nearest_resistances: summary.append(f"  > {name}: ${self._format_val(price, 4)}")

        if orderbook_analysis:
             pressure = orderbook_analysis.get('pressure', 'N/A')
             total_bid = orderbook_analysis.get('total_bid_usd', Decimal(0))
             total_ask = orderbook_analysis.get('total_ask_usd', Decimal(0))
             limit = self.orderbook_settings.get('limit', 50)

             summary.append(f"OB Pressure (Top {limit}): {pressure}")
             summary.append(f"OB Value (Bids): ${self._format_val(total_bid, 0)}")
             summary.append(f"OB Value (Asks): ${self._format_val(total_ask, 0)}")

             clusters = sorted(orderbook_analysis.get("clusters", []), key=lambda x: x['cluster_value_usd'], reverse=True)
             if clusters:
                 summary.append(f"{NEON_PURPLE}Significant OB Clusters (Top 5):{RESET}")
                 for cluster in clusters[:5]:
                      color = NEON_GREEN if cluster["type"] == "Support" else NEON_RED
                      level_price_f = self._format_val(cluster['level_price'], 4)
                      cluster_val_f = self._format_val(cluster['cluster_value_usd'], 0)
                      summary.append(f"  {color}{cluster['type']} near {cluster['level_name']} (${level_price_f}) - Value: ${cluster_val_f}{RESET}")
             else:
                summary.append(f"{NEON_YELLOW}No significant OB clusters found.{RESET}")
        else:
             summary.append(f"{NEON_YELLOW}Orderbook analysis unavailable.{RESET}")

        return summary

    def _interpret_analysis(self, df: pd.DataFrame, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> dict:
        interpretation = {"summary": [], "signals": {}}
        if df.empty or len(df) < 2:
            self.logger.warning(f"Insufficient data ({len(df)} rows) for interpretation on {self.symbol}.")
            interpretation["summary"].append(f"{NEON_RED}Insufficient data for analysis.{RESET}")
            return interpretation

        try:
             # Ensure index is sequential for iloc[-1] and [-2]
             df = df.reset_index(drop=True)
             last_row = df.iloc[-1]
             prev_row = df.iloc[-2] if len(df) >= 2 else last_row

             trend_summary, trend_signals = self._interpret_trend(last_row, prev_row)
             osc_summary, osc_signals = self._interpret_oscillators(last_row, prev_row)
             macd_summary, macd_signals = self._interpret_macd(last_row, prev_row)
             bb_summary, bb_signals = self._interpret_bbands(last_row)
             vol_summary, vol_signals = self._interpret_volume(last_row, prev_row)
             level_summary = self._interpret_levels_orderbook(current_price, levels, orderbook_analysis)

             interpretation["summary"].extend(trend_summary)
             interpretation["summary"].extend(osc_summary)
             interpretation["summary"].extend(macd_summary)
             interpretation["summary"].extend(bb_summary)
             interpretation["summary"].extend(vol_summary)
             interpretation["summary"].extend(level_summary)

             interpretation["signals"].update(trend_signals)
             interpretation["signals"].update(osc_signals)
             interpretation["signals"].update(macd_signals)
             interpretation["signals"].update(bb_signals)
             interpretation["signals"].update(vol_signals)
             # Level/OB signals could be added here (e.g., "PriceNearSupportCluster")

        except IndexError:
             self.logger.error(f"IndexError during interpretation for {self.symbol}. DataFrame length: {len(df)}")
             interpretation["summary"].append(f"{NEON_RED}Error accessing data for interpretation.{RESET}")
        except Exception as e:
             self.logger.exception(f"Unexpected error during analysis interpretation for {self.symbol}: {e}")
             interpretation["summary"].append(f"{NEON_RED}Error during interpretation: {e}{RESET}")


        return interpretation


    def analyze(self, df_klines: pd.DataFrame, current_price: Decimal, orderbook: Optional[dict]) -> dict:
        """Main analysis function."""
        analysis_result = {
            "symbol": self.symbol,
            "timestamp": datetime.now(TIMEZONE).isoformat(),
            "current_price": self._format_val(current_price, 4),
            "kline_interval": "N/A",
            "levels": {},
            "orderbook_analysis": {},
            "interpretation": {"summary": [f"{NEON_RED}Analysis could not be performed.{RESET}"], "signals": {}},
            "raw_indicators": {}
        }

        if df_klines.empty:
            self.logger.error(f"Kline data is empty for {self.symbol}, cannot perform analysis.")
            return analysis_result
        elif len(df_klines) < 2:
             self.logger.warning(f"Kline data has only {len(df_klines)} row(s) for {self.symbol}, analysis may be incomplete.")
             analysis_result["interpretation"]["summary"] = [f"{NEON_YELLOW}Warning: Insufficient kline data for full analysis.{RESET}"]


        try:
             # Infer Interval
            if len(df_klines) >= 2:
                time_diff = df_klines['timestamp'].diff().min()
                analysis_result["kline_interval"] = str(time_diff) if pd.notna(time_diff) else "N/A"


            df_with_indicators = self._calculate_indicators(df_klines)
            if df_with_indicators.empty:
                 return analysis_result

            if not df_with_indicators.empty:
                last_row_indicators = df_with_indicators.iloc[-1].to_dict()
                analysis_result["raw_indicators"] = {
                    k: self._format_val(v, 4) if isinstance(v, (Decimal, float, np.floating)) else
                       (v.isoformat() if isinstance(v, pd.Timestamp) else None if pd.isna(v) else v)
                    for k, v in last_row_indicators.items()
                }


            levels = self._calculate_levels(df_with_indicators, current_price)
            analysis_result["levels"] = {
                 k: {name: self._format_val(price, 4) for name, price in v.items()} if isinstance(v, dict)
                   else (self._format_val(v, 4) if isinstance(v, Decimal) else v)
                 for k, v in levels.items()
            }


            orderbook_analysis_raw = self._analyze_orderbook(orderbook, current_price, levels)
             analysis_result["orderbook_analysis"] = {
                "pressure": orderbook_analysis_raw.get("pressure", "N/A"),
                "total_bid_usd": self._format_val(orderbook_analysis_raw.get("total_bid_usd", Decimal(0)), 0),
                "total_ask_usd": self._format_val(orderbook_analysis_raw.get("total_ask_usd", Decimal(0)), 0),
                "clusters": [
                    {
                        "type": c.get("type"), "level_name": c.get("level_name"),
                        "level_price": self._format_val(c.get("level_price"), 4),
                        "cluster_value_usd": self._format_val(c.get("cluster_value_usd"), 0),
                        "price_range": (self._format_val(c.get("price_range", (np.nan, np.nan))[0], 4), self._format_val(c.get("price_range", (np.nan, np.nan))[1], 4))

                    } for c in orderbook_analysis_raw.get("clusters", [])
                ]
            }


            interpretation = self._interpret_analysis(df_with_indicators, current_price, levels, orderbook_analysis_raw)
            analysis_result["interpretation"] = interpretation

        except Exception as e:
            self.logger.exception(f"Critical error during analysis pipeline for {self.symbol}: {e}")
            analysis_result["interpretation"]["summary"] = [f"{NEON_RED}Critical Analysis Error: {e}{RESET}"]

        return analysis_result

    def format_analysis_output(self, analysis_result: dict) -> str:
        symbol = analysis_result.get('symbol', 'N/A')
        timestamp_str = analysis_result.get('timestamp', 'N/A')
        try:
            # Attempt to parse and format timestamp nicely
            dt_obj = datetime.fromisoformat(timestamp_str).astimezone(TIMEZONE)
            ts_formatted = dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")
        except (ValueError, TypeError):
            ts_formatted = timestamp_str # Fallback to raw string

        price = analysis_result.get('current_price', 'N/A')
        interval = analysis_result.get('kline_interval', 'N/A')


        output = f"\n--- Analysis Report for {symbol} --- ({ts_formatted})\n"
        output += f"{NEON_BLUE}Interval:{RESET} {interval}    {NEON_BLUE}Current Price:{RESET} ${price}\n"

        interpretation = analysis_result.get("interpretation", {})
        summary = interpretation.get("summary", [])

        if summary:
             output += "\n" + "\n".join(summary) + "\n"
        else:
             output += f"{NEON_YELLOW}No interpretation summary available.{RESET}\n"

        return output


async def run_analysis_loop(symbol: str, interval_config: str, client: BybitCCXTClient, analyzer: TradingAnalyzer, logger_instance: logging.Logger):
    analysis_interval_sec = CONFIG.get("analysis_interval_seconds", 30)
    kline_limit = CONFIG.get("kline_limit", 200)
    orderbook_limit = analyzer.orderbook_settings.get("limit", 50)

    if analysis_interval_sec < 10:
        logger_instance.warning(f"Analysis interval ({analysis_interval_sec}s) is very short. Ensure system and API can handle the load.")

    while True:
        start_time = time.monotonic()
        logger_instance.debug(f"--- Starting Analysis Cycle for {symbol} ---")

        analysis_result = None # Define outside try block

        try:
            tasks = {
                "price": asyncio.create_task(client.fetch_current_price(symbol)),
                "klines": asyncio.create_task(client.fetch_klines(symbol, interval_config, kline_limit)),
                "orderbook": asyncio.create_task(client.fetch_orderbook(symbol, orderbook_limit)),
            }

            await asyncio.wait([tasks["price"], tasks["klines"]], return_when=asyncio.ALL_COMPLETED)

            current_price = tasks["price"].result()
            df_klines = tasks["klines"].result()

            if current_price is None or df_klines.empty:
                logger_instance.error(f"Failed to fetch critical data (price or klines) for {symbol}. Skipping analysis cycle.")
                await async_sleep(RETRY_DELAY_SECONDS) # Wait before retrying fetch
                continue

            # Await orderbook with timeout
            try:
                orderbook = await asyncio.wait_for(tasks["orderbook"], timeout=15.0)
            except asyncio.TimeoutError:
                 logger_instance.warning(f"Fetching orderbook for {symbol} timed out. Proceeding without it.")
                 orderbook = None
                 tasks["orderbook"].cancel() # Attempt to cancel the lingering task
            except Exception as ob_err:
                logger_instance.error(f"Error fetching or processing orderbook task for {symbol}: {ob_err}")
                orderbook = None


            analysis_result = analyzer.analyze(df_klines, current_price, orderbook)

            # Output and Logging
            output_string = analyzer.format_analysis_output(analysis_result)
            logger_instance.info(output_string) # Log the formatted output

            # Log structured JSON if debug level
            if logger_instance.isEnabledFor(logging.DEBUG):
                try:
                    # Remove color codes for JSON logging
                    clean_summary = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in analysis_result.get("interpretation", {}).get("summary", [])]
                    log_data = analysis_result.copy()
                    log_data["interpretation"]["summary"] = clean_summary
                    logger_instance.debug(json.dumps(log_data, default=str, indent=2))
                except Exception as json_err:
                     logger_instance.error(f"Error converting analysis result to JSON for debug logging: {json_err}")


        except ccxt.AuthenticationError as e:
            logger_instance.critical(f"{NEON_RED}Authentication Error: {e}. Check API Key/Secret. Stopping.{RESET}")
            break
        except ccxt.InvalidNonce as e:
             logger_instance.critical(f"{NEON_RED}Invalid Nonce Error: {e}. Check system time sync. Stopping.{RESET}")
             break
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
             logger_instance.error(f"{NEON_RED}Order-related ccxt error: {e}. Check parameters or funds.{RESET}")
             # Continue analysis loop
        except Exception as e:
            logger_instance.exception(f"{NEON_RED}An unexpected error occurred in the main analysis loop for {symbol}: {e}{RESET}")
            # Log stack trace

        # Sleep management
        elapsed_time = time.monotonic() - start_time
        sleep_duration = max(0, analysis_interval_sec - elapsed_time)
        logger_instance.debug(f"Cycle took {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s.")
        if elapsed_time > analysis_interval_sec:
             logger_instance.warning(f"Analysis cycle took longer ({elapsed_time:.2f}s) than configured interval ({analysis_interval_sec}s).")

        await async_sleep(sleep_duration)

    logger_instance.info(f"Analysis loop stopped for {symbol}.")


async def main():
    global logger

    # Initialize temporary client for market loading
    temp_logger = setup_logger("INIT")
    client = BybitCCXTClient(API_KEY, API_SECRET, CCXT_BASE_URL, temp_logger)
    markets_loaded = await client.initialize_markets()

    if not markets_loaded:
        temp_logger.critical(f"{NEON_RED}Failed to load markets. Cannot proceed. Exiting.{RESET}")
        await client.close()
        logging.shutdown()
        return

    valid_symbols = list(client.markets.keys()) if client.markets else []
    await client.close() # Close temp client

    symbol = ""
    while True:
        symbol_input = input(f"{NEON_BLUE}Enter trading symbol (e.g., BTC/USDT): {RESET}").upper().strip()
        if not symbol_input: continue

        # Standardize potential variations (e.g., BTCUSDT to BTC/USDT)
        if "/" not in symbol_input and len(symbol_input) > 4: # Basic guess
             potential_base = symbol_input[:-4] if symbol_input.endswith('USDT') else symbol_input[:-3] if symbol_input.endswith(('USD', 'EUR', 'BTC')) else None
             potential_quote = symbol_input[len(potential_base):] if potential_base else None
             if potential_base and potential_quote:
                 formatted_symbol = f"{potential_base}/{potential_quote}"
                 if formatted_symbol in valid_symbols:
                    print(f"{NEON_YELLOW}Assuming you mean: {formatted_symbol}{RESET}")
                    symbol_input = formatted_symbol
                 else:
                    # Check reverse? USDT/BTC rarely exists
                    pass

        if symbol_input in valid_symbols:
            symbol = symbol_input
            break
        else:
            print(f"{NEON_RED}Invalid or unknown symbol: '{symbol_input}'. {RESET}")
            # Suggest similar symbols? (more complex)
            find_similar = [s for s in valid_symbols if symbol_input.split('/')[0] in s] if '/' in symbol_input else [s for s in valid_symbols if symbol_input in s]
            if find_similar:
                 print(f"{NEON_YELLOW}Did you mean one of these? {', '.join(find_similar[:5])}{RESET}")


    interval_key = ""
    default_interval = CONFIG.get("indicator_settings", {}).get("default_interval", "15")
    while True:
        interval_input = input(f"{NEON_BLUE}Enter timeframe [{', '.join(VALID_INTERVALS)}] (default: {default_interval}): {RESET}").strip()
        if not interval_input:
             interval_key = default_interval
             print(f"{NEON_YELLOW}Using default interval: {interval_key}{RESET}")
             break
        if interval_input in VALID_INTERVALS:
            interval_key = interval_input
            break
        if interval_input in REVERSE_CCXT_INTERVAL_MAP:
            interval_key = REVERSE_CCXT_INTERVAL_MAP[interval_input]
            print(f"{NEON_YELLOW}Using interval {interval_key} (mapped from {interval_input}){RESET}")
            break
        print(f"{NEON_RED}Invalid interval. Choose from: {', '.join(VALID_INTERVALS)} or compatible (e.g., 1h, 4h, 1d){RESET}")


    logger = setup_logger(symbol)
    logger.info("Logger initialized.")

    client = BybitCCXTClient(API_KEY, API_SECRET, CCXT_BASE_URL, logger)
     # No need to load markets again if we trust the first load worked
    client.markets = {s: m for s, m in temp_logger.exchange.markets.items()} if temp_logger.exchange.markets else None

    analyzer = TradingAnalyzer(CONFIG, logger, symbol)

    ccxt_interval = CCXT_INTERVAL_MAP.get(interval_key, "N/A")
    market_type = client.get_market_category(symbol)

    logger.info(f"Starting analysis for {NEON_PURPLE}{symbol}{RESET} ({market_type}) on interval {NEON_PURPLE}{interval_key}{RESET} ({ccxt_interval}).")
    logger.info(f"Using API Environment: {NEON_YELLOW}{API_ENV.upper()}{RESET} ({client.exchange.urls['api']})")
    logger.info(f"Analysis loop interval: {CONFIG.get('analysis_interval_seconds', 30)} seconds.")
    logger.info(f"Kline fetch limit: {CONFIG.get('kline_limit', 200)} candles.")


    try:
        await run_analysis_loop(symbol, interval_key, client, analyzer, logger)
    except KeyboardInterrupt:
        logger.info(f"{NEON_YELLOW}Analysis stopped by user.{RESET}")
    except Exception as e:
         logger.critical(f"{NEON_RED}Critical error during main execution: {e}{RESET}", exc_info=True)
    finally:
        logger.info("Shutting down...")
        await client.close()
        logger.info("Application finished.")
        # Flush handlers and shutdown logging system
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()

if __name__ == "__main__":
    # Add basic regex import for cleaning log output
    import re
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{NEON_YELLOW}Process interrupted by user. Exiting gracefully.{RESET}")
    except Exception as e:
        print(f"\n{NEON_RED}A critical top-level error occurred: {e}{RESET}")
        traceback.print_exc()
