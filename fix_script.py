import asyncio
import json
import logging
import os
import re  # Added import
import time
import traceback
from datetime import datetime
from decimal import Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv

init(autoreset=True)
load_dotenv()

getcontext().prec = 18

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
CCXT_URLS = {
    "test": "https://api-testnet.bybit.com",
    "prod": "https://api.bybit.com",
}
API_ENV = os.getenv("API_ENVIRONMENT", "prod").lower()
CCXT_BASE_URL = CCXT_URLS.get(API_ENV, CCXT_URLS["prod"])

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

DEFAULT_TIMEZONE = "America/Chicago"
try:
    TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", DEFAULT_TIMEZONE))
except Exception:
    TIMEZONE = ZoneInfo("UTC")

MAX_API_RETRIES = 5
RETRY_DELAY_SECONDS = 5
CCXT_TIMEOUT_MS = 20000
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "360": "6h",
    "720": "12h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
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
            "momentum_crossover": False,  # Requires more logic
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
            "psar_flip": True,
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
        "orderbook_settings": {"limit": 50, "cluster_threshold_usd": 10000, "cluster_proximity_pct": 0.1},
        "logging": {"level": "INFO", "rotation_max_bytes": 10 * 1024 * 1024, "rotation_backup_count": 5},
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            return default_config
        except OSError:
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
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
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
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
    log_level = getattr(logging, CONFIG.get("logging", {}).get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        max_bytes = int(CONFIG.get("logging", {}).get("rotation_max_bytes", 10 * 1024 * 1024))
        backup_count = int(CONFIG.get("logging", {}).get("rotation_backup_count", 5))

        file_handler = RotatingFileHandler(log_filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        NEON_BLUE + "%(asctime)s" + RESET + " [%(levelname)s] " + NEON_YELLOW + f"{symbol}" + RESET + " - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    return logger


logger: logging.Logger | None = None


async def async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class BybitCCXTClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str, logger_instance: logging.Logger) -> None:
        self.logger = logger_instance
        self.exchange = ccxt_async.bybit(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",  # Assuming linear contracts (USDT pairs)
                    "adjustForTimeDifference": True,
                },
                "urls": {"api": base_url},
                "timeout": CCXT_TIMEOUT_MS,
            }
        )
        self.markets = None
        self._temp_exchange_for_init = None  # Added for market loading fix

    async def initialize_markets(self, retries=3) -> bool:
        # Create a temporary exchange instance ONLY for market loading
        # This avoids potential issues with the main instance during startup
        self._temp_exchange_for_init = ccxt_async.bybit(
            {
                "enableRateLimit": True,
                "options": {"adjustForTimeDifference": True},
                "urls": {"api": self.exchange.urls["api"]},
                "timeout": CCXT_TIMEOUT_MS,
            }
        )
        for attempt in range(retries):
            try:
                self.markets = await self._temp_exchange_for_init.load_markets()
                self.logger.info(
                    f"Successfully loaded {len(self.markets)} markets from {self._temp_exchange_for_init.name}."
                )
                await self._temp_exchange_for_init.close()  # Close the temp exchange
                self._temp_exchange_for_init = None
                return True
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                self.logger.warning(f"Failed to load markets (Attempt {attempt + 1}/{retries}): {e}. Retrying...")
                await async_sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            except ccxt.AuthenticationError as e:
                # Auth error shouldn't happen here unless Bybit changed API for public endpoints
                self.logger.error(f"{NEON_RED}Authentication Error loading markets (unexpected): {e}.{RESET}")
                await self._temp_exchange_for_init.close()
                self._temp_exchange_for_init = None
                return False
            except Exception as e:
                self.logger.exception(f"Unexpected error loading markets (Attempt {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    await self._temp_exchange_for_init.close()
                    self._temp_exchange_for_init = None
                    return False
                await async_sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        # Ensure temp exchange is closed even if loop finishes without returning
        if self._temp_exchange_for_init:
            await self._temp_exchange_for_init.close()
            self._temp_exchange_for_init = None
        return False

    def is_valid_symbol(self, symbol: str) -> bool:
        if self.markets is None:
            self.logger.warning("Markets not loaded, cannot validate symbol.")
            return False  # Or True, depending on desired strictness before load
        return symbol in self.markets

    def get_symbol_details(self, symbol: str) -> dict | None:
        if not self.is_valid_symbol(symbol):
            return None
        return self.markets.get(symbol)

    def get_market_category(self, symbol: str) -> str:
        details = self.get_symbol_details(symbol)
        if details:
            if details.get("linear"):
                return "linear"
            if details.get("inverse"):
                return "inverse"
            if details.get("spot"):
                return "spot"
            # Fallback check based on type if linear/inverse/spot missing
            market_type = details.get("type")
            if market_type == "swap":
                return "linear" if details.get("quote", "").upper() == "USDT" else "inverse"
            elif market_type == "spot":
                return "spot"

        # Default guess based on quote currency if details are unhelpful
        self.logger.debug(
            f"Could not determine category from market details for {symbol}, guessing based on quote currency."
        )
        if symbol.endswith("/USDT"):
            return "linear"
        if symbol.endswith("/USD"):
            return "inverse"
        return "spot"  # Default fallback

    async def close(self) -> None:
        if self._temp_exchange_for_init:  # Ensure temp exchange is closed if close() is called early
            try:
                await self._temp_exchange_for_init.close()
                self.logger.info("Closed temporary ccxt exchange connection used for init.")
            except Exception as e:
                self.logger.error(f"Error closing temporary ccxt connection: {e}")
            self._temp_exchange_for_init = None

        if self.exchange:
            try:
                await self.exchange.close()
                self.logger.info("Closed main ccxt exchange connection.")
            except Exception as e:
                self.logger.error(f"Error closing main ccxt connection: {e}")

    async def fetch_with_retry(self, method_name: str, *args, **kwargs) -> Any | None:
        retries = MAX_API_RETRIES
        current_delay = RETRY_DELAY_SECONDS
        for attempt in range(retries):
            try:
                method = getattr(self.exchange, method_name)
                result = await method(*args, **kwargs)
                return result
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
                self.logger.warning(
                    f"{NEON_YELLOW}Network/Timeout/Rate Limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}"
                )
            except ccxt.RateLimitExceeded as e:
                self.logger.warning(
                    f"{NEON_YELLOW}Explicit Rate limit exceeded calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}"
                )
            except ccxt.ExchangeError as e:
                error_str = str(e).lower()
                # Added more specific Bybit error codes/messages if known
                # e.g., if '10001' means parameter error, '10006' session expired etc.
                if (
                    "too many visits" in error_str
                    or "system busy" in error_str
                    or "service unavailable" in error_str
                    or "api concurrency limit" in error_str
                ):
                    self.logger.warning(
                        f"{NEON_YELLOW}Server busy/rate limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}"
                    )
                else:
                    self.logger.error(f"{NEON_RED}Non-retryable ccxt ExchangeError calling {method_name}: {e}{RESET}")
                    return None
            except Exception as e:
                self.logger.exception(
                    f"{NEON_RED}Unexpected error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}\nArgs: {args}\nKwargs: {kwargs}{RESET}"
                )

            if attempt < retries - 1:
                await async_sleep(current_delay)
                current_delay = min(current_delay * 1.5, 60)  # Cap delay
            else:
                self.logger.error(f"{NEON_RED}Max retries reached for {method_name}.{RESET}")
                return None
        return None

    async def fetch_ticker(self, symbol: str) -> dict | None:
        self.logger.debug(f"Fetching ticker for {symbol}")
        category = self.get_market_category(symbol)
        # Bybit often requires category in params for fetch_ticker/fetch_tickers
        tickers = await self.fetch_with_retry("fetch_tickers", symbols=[symbol], params={"category": category})
        if tickers and symbol in tickers:
            return tickers[symbol]
        self.logger.error(f"Could not fetch ticker for {symbol} (category: {category})")
        return None

    async def fetch_current_price(self, symbol: str) -> Decimal | None:
        ticker = await self.fetch_ticker(symbol)
        if ticker and "last" in ticker and ticker["last"] is not None:
            try:
                return Decimal(str(ticker["last"]))
            except (InvalidOperation, TypeError) as e:
                self.logger.error(f"Error converting last price '{ticker['last']}' to Decimal for {symbol}: {e}")
                return None
        self.logger.warning(f"Last price not found or null in ticker data for {symbol}")
        return None

    async def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            self.logger.error(f"Invalid timeframe '{timeframe}' provided.")
            return pd.DataFrame()

        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching {limit} klines for {symbol} with interval {ccxt_timeframe} (category: {category})")
        # Pass category explicitly as Bybit API requires it for OHLCV
        klines = await self.fetch_with_retry(
            "fetch_ohlcv", symbol, timeframe=ccxt_timeframe, limit=limit, params={"category": category}
        )

        if klines is None or len(klines) == 0:
            self.logger.warning(f"No kline data returned for {symbol} interval {ccxt_timeframe}")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
            if df.empty:
                self.logger.warning(f"Kline data for {symbol} was empty initially.")
                return pd.DataFrame()

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(TIMEZONE)

            # Convert to Decimal after basic numeric conversion for better precision handling later
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Keep as float for pandas_ta, convert to Decimal where needed for calculations/output
                # df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)

            initial_rows = len(df)
            df.dropna(
                subset=["open", "high", "low", "close", "volume"], inplace=True
            )  # Drop rows where essential values are missing
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                self.logger.warning(
                    f"Dropped {dropped_rows} rows with NaN in essential OHLCV columns from kline data for {symbol}."
                )

            if df.empty:
                self.logger.warning(f"Kline data for {symbol} is empty after cleaning.")
                return pd.DataFrame()

            self.logger.debug(f"Successfully fetched and processed {len(df)} klines for {symbol}.")
            # Ensure sorting by timestamp
            return df.sort_values(by="timestamp").reset_index(drop=True)

        except Exception as e:
            self.logger.exception(f"Error processing kline data for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_orderbook(self, symbol: str, limit: int) -> dict | None:
        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching order book for {symbol} with limit {limit} (category: {category})")
        # Pass category explicitly as Bybit API requires it
        orderbook = await self.fetch_with_retry("fetch_order_book", symbol, limit=limit, params={"category": category})
        if orderbook:
            # Check if bids/asks are present and lists (can be None)
            if isinstance(orderbook.get("bids"), list) and isinstance(orderbook.get("asks"), list):
                self.logger.debug(
                    f"Fetched order book for {symbol} with {len(orderbook['bids'])} bids and {len(orderbook['asks'])} asks."
                )
                return orderbook
            else:
                self.logger.warning(f"Fetched order book data for {symbol} has missing or invalid bids/asks structure.")
                return None
        else:
            self.logger.warning(f"Failed to fetch order book for {symbol} after retries.")
            return None


class TradingAnalyzer:
    def __init__(self, config: dict, logger_instance: logging.Logger, symbol: str) -> None:
        self.config = config
        self.logger = logger_instance
        self.symbol = symbol
        self.indicator_settings = config.get("indicator_settings", {})
        self.analysis_flags = config.get("analysis_flags", {})
        self.thresholds = config.get("thresholds", {})
        self.orderbook_settings = config.get("orderbook_settings", {})

        # Dynamically generate expected column names based on config
        self._generate_column_names()

    def _generate_column_names(self) -> None:
        """Generates expected indicator column names based on config."""
        self.col_names = {}
        is_ = self.indicator_settings
        # Define defaults directly in get() for robustness
        self.col_names["sma_short"] = f"SMA_{is_.get('sma_short_period', 10)}"
        self.col_names["sma_long"] = f"SMA_{is_.get('sma_long_period', 50)}"
        self.col_names["ema_short"] = f"EMA_{is_.get('ema_short_period', 12)}"
        self.col_names["ema_long"] = f"EMA_{is_.get('ema_long_period', 26)}"
        self.col_names["rsi"] = f"RSI_{is_.get('rsi_period', 14)}"
        self.col_names["stochrsi_k"] = (
            f"STOCHRSIk_{is_.get('stoch_rsi_period', 14)}_{is_.get('rsi_period', 14)}_{is_.get('stoch_k_period', 3)}_{is_.get('stoch_d_period', 3)}"
        )
        self.col_names["stochrsi_d"] = (
            f"STOCHRSId_{is_.get('stoch_rsi_period', 14)}_{is_.get('rsi_period', 14)}_{is_.get('stoch_k_period', 3)}_{is_.get('stoch_d_period', 3)}"
        )
        self.col_names["macd_line"] = (
            f"MACD_{is_.get('macd_fast', 12)}_{is_.get('macd_slow', 26)}_{is_.get('macd_signal', 9)}"
        )
        self.col_names["macd_signal"] = (
            f"MACDs_{is_.get('macd_fast', 12)}_{is_.get('macd_slow', 26)}_{is_.get('macd_signal', 9)}"
        )
        self.col_names["macd_hist"] = (
            f"MACDh_{is_.get('macd_fast', 12)}_{is_.get('macd_slow', 26)}_{is_.get('macd_signal', 9)}"
        )
        # Ensure std dev is float for name matching, handle potential non-numeric config values
        try:
            bb_std = float(is_.get("bollinger_bands_std_dev", 2.0))
        except (ValueError, TypeError):
            bb_std = 2.0
            self.logger.warning(f"Invalid 'bollinger_bands_std_dev' in config, using default {bb_std}")
        self.col_names["bb_upper"] = f"BBU_{is_.get('bollinger_bands_period', 20)}_{bb_std}"
        self.col_names["bb_lower"] = f"BBL_{is_.get('bollinger_bands_period', 20)}_{bb_std}"
        self.col_names["bb_mid"] = f"BBM_{is_.get('bollinger_bands_period', 20)}_{bb_std}"
        self.col_names["atr"] = f"ATRr_{is_.get('atr_period', 14)}"  # pandas_ta uses ATRr for True Range average
        self.col_names["cci"] = f"CCI_{is_.get('cci_period', 20)}_0.015"  # Default c=0.015 in pandas_ta
        self.col_names["willr"] = f"WILLR_{is_.get('williams_r_period', 14)}"
        self.col_names["mfi"] = f"MFI_{is_.get('mfi_period', 14)}"
        self.col_names["adx"] = f"ADX_{is_.get('adx_period', 14)}"
        self.col_names["dmp"] = f"DMP_{is_.get('adx_period', 14)}"  # +DI
        self.col_names["dmn"] = f"DMN_{is_.get('adx_period', 14)}"  # -DI
        self.col_names["obv"] = "OBV"
        # Use ADOSC (Accum/Dist Oscillator) instead of ADI line for better signal generation
        self.col_names["adosc"] = "ADOSC"  # pandas-ta standard name
        psar_step = is_.get("psar_step", 0.02)
        psar_max = is_.get("psar_max_step", 0.2)
        self.col_names["psar_long"] = f"PSARl_{psar_step}_{psar_max}"  # PSAR when price is above (potential support)
        self.col_names["psar_short"] = (
            f"PSARs_{psar_step}_{psar_max}"  # PSAR when price is below (potential resistance)
        )
        self.col_names["psar_af"] = f"PSARaf_{psar_step}_{psar_max}"  # Acceleration Factor
        self.col_names["psar_rev"] = f"PSARr_{psar_step}_{psar_max}"  # Reversal point (1=reversal, 0=no)

        self.col_names["mom"] = f"MOM_{is_.get('momentum_period', 10)}"
        # Add a volume MA column name based on pandas_ta default (if not specified)
        self.col_names["vol_ma"] = f"SMA_{is_.get('volume_ma_period', 20)}"  # Assuming SMA for volume MA

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("Cannot calculate indicators on empty DataFrame.")
            return df

        min_data_points_needed = max(
            self.indicator_settings.get("sma_long_period", 50),
            self.indicator_settings.get("ema_long_period", 26),
            self.indicator_settings.get("macd_slow", 26)
            + self.indicator_settings.get("macd_signal", 9),  # MACD needs slow+signal
            self.indicator_settings.get("bollinger_bands_period", 20),
            self.indicator_settings.get("adx_period", 14) * 2,  # ADX needs more data
            self.indicator_settings.get("mfi_period", 14),
            self.indicator_settings.get("volume_ma_period", 20),
            # Add other long periods
        )

        if len(df) < min_data_points_needed:
            self.logger.warning(
                f"Insufficient data points ({len(df)} < {min_data_points_needed} required) for some indicators. Results may be inaccurate or contain NaNs."
            )

        # Ensure columns are numeric floats for pandas_ta
        df_numeric = df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_numeric.columns:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")
        df_numeric.dropna(
            subset=["open", "high", "low", "close"], inplace=True
        )  # Drop rows if essential prices are NaN
        if df_numeric.empty:
            self.logger.error("DataFrame became empty after converting price columns to numeric.")
            return pd.DataFrame()

        is_ = self.indicator_settings

        # Define the strategy using a list of dictionaries
        strategy_ta = [
            # Moving Averages
            {"kind": "sma", "length": is_.get("sma_short_period", 10)},
            {"kind": "sma", "length": is_.get("sma_long_period", 50)},
            {"kind": "ema", "length": is_.get("ema_short_period", 12)},
            {"kind": "ema", "length": is_.get("ema_long_period", 26)},
            # Oscillators
            {"kind": "rsi", "length": is_.get("rsi_period", 14)},
            {
                "kind": "stochrsi",
                "length": is_.get("stoch_rsi_period", 14),
                "rsi_length": is_.get("rsi_period", 14),
                "k": is_.get("stoch_k_period", 3),
                "d": is_.get("stoch_d_period", 3),
            },
            {"kind": "cci", "length": is_.get("cci_period", 20)},  # Uses default c=0.015
            {"kind": "willr", "length": is_.get("williams_r_period", 14)},
            {"kind": "mfi", "length": is_.get("mfi_period", 14)},
            # MACD
            {
                "kind": "macd",
                "fast": is_.get("macd_fast", 12),
                "slow": is_.get("macd_slow", 26),
                "signal": is_.get("macd_signal", 9),
            },
            # Volatility / Bands
            {
                "kind": "bbands",
                "length": is_.get("bollinger_bands_period", 20),
                "std": float(is_.get("bollinger_bands_std_dev", 2.0)),  # Ensure float
            },
            {"kind": "atr", "length": is_.get("atr_period", 14)},  # Generates 'ATRr_14'
            # Trend
            {"kind": "adx", "length": is_.get("adx_period", 14)},  # Generates ADX, DMP, DMN
            {
                "kind": "psar",
                "step": is_.get("psar_step", 0.02),
                "max_step": is_.get("psar_max_step", 0.2),
            },  # Generates PSARl, PSARs, PSARaf, PSARr
            # Volume
            {"kind": "obv"},
            {"kind": "adosc"},  # Accumulation/Distribution Oscillator
            # Custom/Separate calculations if needed
            {"kind": "mom", "length": is_.get("momentum_period", 10)},
            {
                "kind": "sma",
                "close": "volume",
                "length": is_.get("volume_ma_period", 20),
                "prefix": "VOL",
            },  # Calculate Volume SMA
        ]

        # Create the Strategy object
        custom_strategy = ta.Strategy(
            name="TradingAnalysisStrategy", description="Comprehensive TA Strategy using pandas_ta", ta=strategy_ta
        )

        try:
            # Apply the strategy - this appends columns to df_numeric
            df_numeric.ta.strategy(custom_strategy, timed=False)
            self.logger.debug(f"Calculated pandas_ta indicators for {self.symbol}.")

            # Rename the volume SMA column to match our expected name convention
            pandas_ta_vol_sma_name = f"VOL_SMA_{is_.get('volume_ma_period', 20)}"
            if pandas_ta_vol_sma_name in df_numeric.columns and self.col_names["vol_ma"] != pandas_ta_vol_sma_name:
                df_numeric.rename(columns={pandas_ta_vol_sma_name: self.col_names["vol_ma"]}, inplace=True)
                self.logger.debug(
                    f"Renamed volume MA column from {pandas_ta_vol_sma_name} to {self.col_names['vol_ma']}"
                )

        except Exception as e:
            self.logger.exception(
                f"Error calculating indicators using pandas_ta for {self.symbol}: {e}\n{traceback.format_exc()}"
            )
            # Return the original DataFrame without indicators if calculation fails
            return df  # Or return df_numeric? Check implications. Let's return original for safety.

        # Combine original DataFrame (with DatetimeIndex, potentially Decimals)
        # with the new indicator columns from df_numeric
        # We need to ensure the index aligns perfectly.
        # Since we potentially dropped rows in df_numeric, use the index from df_numeric
        # Add indicator columns back to original df based on matching index
        indicator_cols = [col for col in df_numeric.columns if col not in df.columns]
        # Use join to merge based on index
        df_with_indicators = df.join(df_numeric[indicator_cols])

        # PSAR Handling: Combine PSARl and PSARs into a single 'psar' value if needed
        # The separate l/s columns are actually more useful for interpretation
        psar_l_col = self.col_names.get("psar_long")
        psar_s_col = self.col_names.get("psar_short")
        if psar_l_col in df_with_indicators.columns and psar_s_col in df_with_indicators.columns:
            # Example: Create a single 'psar_value' column
            # df_with_indicators['psar_value'] = df_with_indicators[psar_l_col].fillna(df_with_indicators[psar_s_col])
            # We will use the l/s columns directly in interpretation instead.
            pass

        # Final check for required columns after calculation and merge
        missing_cols = [name for name in self.col_names.values() if name not in df_with_indicators.columns]
        if missing_cols:
            self.logger.warning(
                f"Some indicator columns are missing after calculation for {self.symbol}: {missing_cols}"
            )

        return df_with_indicators

    def _calculate_levels(self, df: pd.DataFrame, current_price: Decimal) -> dict:
        levels = {"support": {}, "resistance": {}, "pivot": None}
        if df.empty or len(df) < 2:
            self.logger.warning("Insufficient data for level calculation.")
            return levels

        try:
            # Use the full period's data for more stable levels
            high_period = df["high"].max()
            low_period = df["low"].min()
            # Use the last candle's close for pivot calculation as is standard
            close_last = df["close"].iloc[-1]

            # Ensure we have valid numbers before proceeding
            if pd.isna(high_period) or pd.isna(low_period) or pd.isna(close_last):
                self.logger.warning("NaN values found in essential OHLC data for level calculation.")
                return levels

            # Convert to float for calculations, then back to Decimal for storage
            high_f = float(high_period)
            low_f = float(low_period)
            close_f = float(close_last)
            float(current_price)  # Already Decimal, convert for consistency

            # Fibonacci Retracement Levels (based on period high/low)
            diff = high_f - low_f
            if diff > 1e-9:  # Avoid division by zero or tiny differences
                fib_levels = {
                    "Fib 23.6%": high_f - diff * 0.236,
                    "Fib 38.2%": high_f - diff * 0.382,
                    "Fib 50.0%": high_f - diff * 0.5,
                    "Fib 61.8%": high_f - diff * 0.618,
                    "Fib 78.6%": high_f - diff * 0.786,
                }
                for label, value in fib_levels.items():
                    try:
                        value_dec = Decimal(str(value))
                        # Check against current price to classify as support or resistance
                        if value_dec < current_price:
                            levels["support"][label] = value_dec
                        else:
                            levels["resistance"][label] = value_dec
                    except InvalidOperation:
                        self.logger.warning(f"Could not convert Fibonacci level '{label}' ({value}) to Decimal.")
                        continue

            # Classic Pivot Points (based on last candle's HLC)
            # For more stability, could use period's HLC, but standard uses last candle
            # Let's stick to standard: use last candle's H, L, C
            high_last = df["high"].iloc[-1]
            low_last = df["low"].iloc[-1]
            # close_last is already defined

            if pd.isna(high_last) or pd.isna(low_last):
                self.logger.warning("NaN values in last candle's H/L, cannot calculate standard pivot points.")
            else:
                high_lf = float(high_last)
                low_lf = float(low_last)
                # close_f is already defined

                try:
                    pivot = (high_lf + low_lf + close_f) / 3
                    levels["pivot"] = Decimal(str(pivot))
                    pivot_levels = {
                        "S1": (2 * pivot) - high_lf,
                        "R1": (2 * pivot) - low_lf,
                        "S2": pivot - (high_lf - low_lf),
                        "R2": pivot + (high_lf - low_lf),
                        "S3": low_lf - 2 * (high_lf - pivot),
                        "R3": high_lf + 2 * (pivot - low_lf),
                    }
                    for label, value in pivot_levels.items():
                        try:
                            value_dec = Decimal(str(value))
                            # Classify based on current price
                            if value_dec < current_price:
                                levels["support"]["PP " + label] = value_dec  # Prefix to distinguish
                            else:
                                levels["resistance"]["PP " + label] = value_dec
                        except InvalidOperation:
                            self.logger.warning(f"Could not convert Pivot level '{label}' ({value}) to Decimal.")
                            continue
                except ZeroDivisionError:
                    self.logger.warning("Zero division encountered calculating pivot points (likely identical H/L/C).")
                except InvalidOperation as e:
                    self.logger.error(f"Invalid operation during pivot calculation: {e}")

        except (TypeError, ValueError, InvalidOperation, IndexError) as e:
            self.logger.error(f"Error calculating levels for {self.symbol}: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error calculating levels for {self.symbol}: {e}")

        return levels

    def _analyze_orderbook(self, orderbook: dict, current_price: Decimal, levels: dict) -> dict:
        analysis = {"clusters": [], "pressure": "Neutral", "total_bid_usd": Decimal(0), "total_ask_usd": Decimal(0)}
        if (
            not orderbook
            or not isinstance(orderbook.get("bids"), list)
            or not isinstance(orderbook.get("asks"), list)
            or not orderbook["bids"]
            or not orderbook["asks"]
        ):
            self.logger.debug("Orderbook data incomplete or unavailable for analysis.")
            return analysis

        try:
            # Convert lists directly to Decimal, handle potential errors
            bids_raw = [
                [Decimal(str(p)), Decimal(str(s))]
                for p, s in orderbook.get("bids", [])
                if p is not None and s is not None
            ]
            asks_raw = [
                [Decimal(str(p)), Decimal(str(s))]
                for p, s in orderbook.get("asks", [])
                if p is not None and s is not None
            ]

            if not bids_raw and not asks_raw:
                return analysis

            bids = pd.DataFrame(bids_raw, columns=["price", "size"])
            asks = pd.DataFrame(asks_raw, columns=["price", "size"])

            # Ensure price and size are positive
            bids = bids[(bids["price"] > 0) & (bids["size"] > 0)]
            asks = asks[(asks["price"] > 0) & (asks["size"] > 0)]

            if bids.empty and asks.empty:
                return analysis

            bids["value_usd"] = bids["price"] * bids["size"]
            asks["value_usd"] = asks["price"] * asks["size"]

            total_bid_value = bids["value_usd"].sum()
            total_ask_value = asks["value_usd"].sum()
            analysis["total_bid_usd"] = total_bid_value
            analysis["total_ask_usd"] = total_ask_value

            # Calculate Buy/Sell Pressure Ratio
            total_value = total_bid_value + total_ask_value
            if total_value > 0:
                # Ratio based on value within the fetched book depth
                bid_ask_ratio = total_bid_value / total_value
                if bid_ask_ratio > Decimal("0.6"):
                    analysis["pressure"] = f"{NEON_GREEN}High Buy Pressure{RESET}"
                elif bid_ask_ratio < Decimal("0.4"):
                    analysis["pressure"] = f"{NEON_RED}High Sell Pressure{RESET}"
                else:
                    analysis["pressure"] = f"{NEON_YELLOW}Neutral Pressure{RESET}"
            else:
                analysis["pressure"] = f"{NEON_YELLOW}Neutral Pressure (No Value){RESET}"

            # Cluster Analysis
            cluster_threshold = Decimal(str(self.orderbook_settings.get("cluster_threshold_usd", 10000)))
            proximity_pct = (
                Decimal(str(self.orderbook_settings.get("cluster_proximity_pct", 0.1))) / 100
            )  # Convert percentage

            # Combine support and resistance levels for checking clusters
            all_levels = {**levels.get("support", {}), **levels.get("resistance", {})}
            if levels.get("pivot"):
                all_levels["Pivot"] = levels["pivot"]

            processed_clusters = set()  # Avoid reporting same cluster multiple times if levels overlap

            for name, level_price in all_levels.items():
                if not isinstance(level_price, Decimal) or level_price <= 0:
                    continue

                # Define price range around the level
                min_price = level_price * (Decimal(1) - proximity_pct)
                max_price = level_price * (Decimal(1) + proximity_pct)

                # Find bids (potential support clusters) near this level
                bids_near = bids[(bids["price"] >= min_price) & (bids["price"] <= max_price)]
                bid_cluster_value = bids_near["value_usd"].sum()
                if bid_cluster_value >= cluster_threshold:
                    # Unique ID for this cluster instance
                    cluster_id = f"B_{name}_{level_price:.8f}"  # Use more precision for ID
                    if cluster_id not in processed_clusters:
                        analysis["clusters"].append(
                            {
                                "type": "Support",
                                "level_name": name,
                                "level_price": level_price,
                                "cluster_value_usd": bid_cluster_value,
                                "price_range": (min_price, max_price),
                                "avg_price": bids_near["price"].mean()
                                if not bids_near.empty
                                else level_price,  # Add avg price
                            }
                        )
                        processed_clusters.add(cluster_id)

                # Find asks (potential resistance clusters) near this level
                asks_near = asks[(asks["price"] >= min_price) & (asks["price"] <= max_price)]
                ask_cluster_value = asks_near["value_usd"].sum()
                if ask_cluster_value >= cluster_threshold:
                    cluster_id = f"A_{name}_{level_price:.8f}"
                    if cluster_id not in processed_clusters:
                        analysis["clusters"].append(
                            {
                                "type": "Resistance",
                                "level_name": name,
                                "level_price": level_price,
                                "cluster_value_usd": ask_cluster_value,
                                "price_range": (min_price, max_price),
                                "avg_price": asks_near["price"].mean() if not asks_near.empty else level_price,
                            }
                        )
                        processed_clusters.add(cluster_id)

        except (KeyError, ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Error analyzing orderbook for {self.symbol}: {e}")
            self.logger.debug(traceback.format_exc())  # Log full traceback on debug level
        except Exception as e:
            self.logger.exception(f"Unexpected error analyzing orderbook for {self.symbol}: {e}")

        return analysis

    # --- Interpretation Helpers ---
    def _format_val(self, value, precision=2):
        if pd.isna(value) or value is None:
            return "N/A"
        try:
            # Use Decimal formatting for precision
            if isinstance(value, Decimal):
                # Handle very small or very large numbers if necessary
                if abs(value) < Decimal("1e-8") and value != 0:  # Scientific notation for very small
                    return f"{value:.{precision}e}"
                # Use quantize for correct Decimal rounding
                quantizer = Decimal("1e-" + str(precision))
                return str(value.quantize(quantizer, rounding="ROUND_HALF_UP"))
            elif isinstance(value, (float, np.floating)):
                if abs(value) < 1e-8 and value != 0.0:
                    return f"{value:.{precision}e}"
                return f"{value:.{precision}f}"
            elif isinstance(value, (int, np.integer)):
                return str(value)  # No precision needed for integers
            else:
                # Attempt to convert to float as a fallback
                return f"{float(value):.{precision}f}"
        except (ValueError, TypeError, InvalidOperation):
            return str(value)  # Return as string if formatting fails

    def _get_val(self, row: pd.Series, key: str, default=np.nan):
        """Safely get value from Series, return default if key missing or value is NaN."""
        if key not in row:
            # Log missing key only once per run? Or just debug level?
            # self.logger.debug(f"Indicator key '{key}' not found in DataFrame row.")
            return default
        val = row[key]
        # Check for NaN explicitly (pd.isna handles numpy NaN, None, etc.)
        return val if pd.notna(val) else default

    def _interpret_trend(self, last_row: pd.Series, prev_row: pd.Series) -> tuple[list[str], dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names
        price_precision = 4  # Default precision for price-like values

        # EMA Alignment
        if flags.get("ema_alignment"):
            ema_short = self._get_val(last_row, cols.get("ema_short"))
            ema_long = self._get_val(last_row, cols.get("ema_long"))
            if pd.notna(ema_short) and pd.notna(ema_long):
                curr_state = "Neutral"
                color = NEON_YELLOW
                if ema_short > ema_long:
                    curr_state, color = "Bullish", NEON_GREEN
                elif ema_short < ema_long:
                    curr_state, color = "Bearish", NEON_RED
                signals["ema_trend"] = curr_state
                summary.append(
                    f"{color}EMA Alignment: {curr_state}{RESET} (S:{self._format_val(ema_short, price_precision)} {'><'[ema_short > ema_long]} L:{self._format_val(ema_long, price_precision)})"
                )
            else:
                signals["ema_trend"] = "N/A"

        # ADX Trend Strength
        if flags.get("adx_trend_strength"):
            adx = self._get_val(last_row, cols.get("adx"))
            dmp = self._get_val(last_row, cols.get("dmp"), 0)  # Default to 0 if missing
            dmn = self._get_val(last_row, cols.get("dmn"), 0)  # Default to 0 if missing
            trend_threshold = self.thresholds.get("adx_trending", 25)
            if pd.notna(adx):
                adx_str = f"ADX ({self._format_val(adx, 1)}): "
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
            psar_l_val = self._get_val(last_row, cols.get("psar_long"))
            psar_s_val = self._get_val(last_row, cols.get("psar_short"))
            close_val = self._get_val(last_row, "close")
            psar_rev = self._get_val(last_row, cols.get("psar_rev"))  # Reversal signal

            current_psar_val = np.nan
            is_bullish_psar = False

            # Determine current PSAR value and trend direction
            if pd.notna(psar_l_val):  # Price was above PSAR
                current_psar_val = psar_l_val
                is_bullish_psar = True
            elif pd.notna(psar_s_val):  # Price was below PSAR
                current_psar_val = psar_s_val
                is_bullish_psar = False

            if pd.notna(current_psar_val) and pd.notna(close_val):
                signals["psar_trend"] = "Bullish" if is_bullish_psar else "Bearish"
                flipped = pd.notna(psar_rev) and psar_rev == 1  # Check if reversal happened on this candle

                psar_status = f"PSAR ({self._format_val(current_psar_val, price_precision)}): "
                if is_bullish_psar:
                    psar_status += f"{NEON_GREEN}Price Above (Support){RESET}"
                    if flipped:
                        psar_status += f" {NEON_GREEN}(Just Flipped Bullish){RESET}"
                        signals["psar_signal"] = "Flip Bullish"
                else:  # is_bearish_psar
                    psar_status += f"{NEON_RED}Price Below (Resistance){RESET}"
                    if flipped:
                        psar_status += f" {NEON_RED}(Just Flipped Bearish){RESET}"
                        signals["psar_signal"] = "Flip Bearish"
                if not flipped:
                    signals["psar_signal"] = "Hold"  # No flip on this candle

                summary.append(psar_status)
            else:
                signals["psar_trend"] = "N/A"
                signals["psar_signal"] = "N/A"

        return summary, signals

    def _interpret_oscillators(self, last_row: pd.Series, prev_row: pd.Series) -> tuple[list[str], dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        thresh = self.thresholds
        cols = self.col_names
        osc_precision = 1  # Standard precision for oscillators

        # RSI
        if flags.get("rsi_threshold"):
            rsi = self._get_val(last_row, cols.get("rsi"))
            ob = thresh.get("rsi_overbought", 70)
            os = thresh.get("rsi_oversold", 30)
            if pd.notna(rsi):
                rsi_status = f"RSI ({self._format_val(rsi, osc_precision)}): "
                level = "Neutral"
                color = NEON_YELLOW
                if rsi >= ob:
                    level, color = "Overbought", NEON_RED
                elif rsi <= os:
                    level, color = "Oversold", NEON_GREEN
                signals["rsi_level"] = level
                summary.append(rsi_status + f"{color}{level}{RESET}")
            else:
                signals["rsi_level"] = "N/A"

        # MFI
        if flags.get("mfi_threshold"):
            mfi = self._get_val(last_row, cols.get("mfi"))
            ob = thresh.get("mfi_overbought", 80)
            os = thresh.get("mfi_oversold", 20)
            if pd.notna(mfi):
                mfi_status = f"MFI ({self._format_val(mfi, osc_precision)}): "
                level = "Neutral"
                color = NEON_YELLOW
                if mfi >= ob:
                    level, color = "Overbought", NEON_RED
                elif mfi <= os:
                    level, color = "Oversold", NEON_GREEN
                signals["mfi_level"] = level
                summary.append(mfi_status + f"{color}{level}{RESET}")
            else:
                signals["mfi_level"] = "N/A"

        # CCI
        if flags.get("cci_threshold"):
            cci = self._get_val(last_row, cols.get("cci"))
            ob = thresh.get("cci_overbought", 100)
            os = thresh.get("cci_oversold", -100)
            if pd.notna(cci):
                cci_status = f"CCI ({self._format_val(cci, osc_precision)}): "
                level, color = "Neutral", NEON_YELLOW
                if cci >= ob:
                    level, color = "Overbought", NEON_RED
                elif cci <= os:
                    level, color = "Oversold", NEON_GREEN
                signals["cci_level"] = level
                summary.append(cci_status + f"{color}{level} Zone{RESET}")
            else:
                signals["cci_level"] = "N/A"

        # Williams %R
        if flags.get("williams_r_threshold"):
            wr = self._get_val(last_row, cols.get("willr"))
            ob = thresh.get("williams_r_overbought", -20)  # Typically -20
            os = thresh.get("williams_r_oversold", -80)  # Typically -80
            if pd.notna(wr):
                wr_status = f"Williams %R ({self._format_val(wr, osc_precision)}): "
                level, color = "Neutral", NEON_YELLOW
                if wr >= ob:
                    level, color = "Overbought", NEON_RED  # Closer to 0 is OB
                elif wr <= os:
                    level, color = "Oversold", NEON_GREEN  # Closer to -100 is OS
                signals["wr_level"] = level
                summary.append(wr_status + f"{color}{level}{RESET}")
            else:
                signals["wr_level"] = "N/A"

        # StochRSI Cross
        if flags.get("stoch_rsi_cross"):
            k_now = self._get_val(last_row, cols.get("stochrsi_k"))
            d_now = self._get_val(last_row, cols.get("stochrsi_d"))
            k_prev = self._get_val(prev_row, cols.get("stochrsi_k"))
            d_prev = self._get_val(prev_row, cols.get("stochrsi_d"))

            if pd.notna(k_now) and pd.notna(d_now) and pd.notna(k_prev) and pd.notna(d_prev):
                stoch_status = f"StochRSI (K:{self._format_val(k_now, osc_precision)}, D:{self._format_val(d_now, osc_precision)}): "
                current_state = "Neutral"
                cross_signal = "Hold"  # Default if no cross
                color = NEON_YELLOW

                # Determine current state (K relative to D)
                if k_now > d_now:
                    current_state = "Bullish"
                elif k_now < d_now:
                    current_state = "Bearish"

                # Check for crosses
                if k_now > d_now and k_prev <= d_prev:  # Bullish cross
                    cross_signal = "Bullish Cross"
                    color = NEON_GREEN
                elif k_now < d_now and k_prev >= d_prev:  # Bearish cross
                    cross_signal = "Bearish Cross"
                    color = NEON_RED
                else:  # No cross, just show current state
                    if current_state == "Bullish":
                        color = NEON_GREEN
                    elif current_state == "Bearish":
                        color = NEON_RED
                    cross_signal = f"{current_state} State"  # Report state, not cross

                signals["stochrsi_cross"] = cross_signal  # Store the signal (cross or state)
                summary.append(stoch_status + f"{color}{cross_signal}{RESET}")
            else:
                signals["stochrsi_cross"] = "N/A"

        return summary, signals

    def _interpret_macd(self, last_row: pd.Series, prev_row: pd.Series) -> tuple[list[str], dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names
        macd_precision = 5  # Higher precision for MACD values

        # MACD Cross
        if flags.get("macd_cross"):
            line_now = self._get_val(last_row, cols.get("macd_line"))
            sig_now = self._get_val(last_row, cols.get("macd_signal"))
            line_prev = self._get_val(prev_row, cols.get("macd_line"))
            sig_prev = self._get_val(prev_row, cols.get("macd_signal"))

            if pd.notna(line_now) and pd.notna(sig_now) and pd.notna(line_prev) and pd.notna(sig_prev):
                macd_status = f"MACD (L:{self._format_val(line_now, macd_precision)}, S:{self._format_val(sig_now, macd_precision)}): "
                signal = "Hold"  # Default signal
                current_state = "Neutral"
                color = NEON_YELLOW

                # Determine current state
                if line_now > sig_now:
                    current_state = "Bullish"
                elif line_now < sig_now:
                    current_state = "Bearish"

                # Check for cross
                if current_state == "Bullish" and line_prev <= sig_prev:
                    signal = "Bullish Cross"
                    color = NEON_GREEN
                elif current_state == "Bearish" and line_prev >= sig_prev:
                    signal = "Bearish Cross"
                    color = NEON_RED
                else:  # No cross, report current state
                    signal = f"{current_state} State"
                    if current_state == "Bullish":
                        color = NEON_GREEN
                    elif current_state == "Bearish":
                        color = NEON_RED

                signals["macd_cross"] = signal  # Store cross or state signal
                summary.append(macd_status + f"{color}{signal}{RESET}")

            else:
                signals["macd_cross"] = "N/A"

        # MACD Divergence (Simple 2-point check - USE WITH CAUTION)
        if flags.get("macd_divergence"):
            hist_now = self._get_val(last_row, cols.get("macd_hist"))
            hist_prev = self._get_val(prev_row, cols.get("macd_hist"))
            price_now = self._get_val(last_row, "close")
            price_prev = self._get_val(prev_row, "close")

            if pd.notna(hist_now) and pd.notna(hist_prev) and pd.notna(price_now) and pd.notna(price_prev):
                divergence = "None"
                # Potential Bullish Divergence: Lower low in price, higher low in histogram (often below zero line)
                if price_now < price_prev and hist_now > hist_prev:  # (and hist_prev < 0 potentially)
                    divergence = "Bullish"
                    summary.append(f"{NEON_GREEN}Potential Bullish MACD Divergence{RESET}")
                # Potential Bearish Divergence: Higher high in price, lower high in histogram (often above zero line)
                elif price_now > price_prev and hist_now < hist_prev:  # (and hist_prev > 0 potentially)
                    divergence = "Bearish"
                    summary.append(f"{NEON_RED}Potential Bearish MACD Divergence{RESET}")
                signals["macd_divergence"] = divergence
            else:
                signals["macd_divergence"] = "N/A"

        return summary, signals

    def _interpret_bbands(self, last_row: pd.Series) -> tuple[list[str], dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names
        price_precision = 4

        if flags.get("bollinger_bands_break"):
            upper = self._get_val(last_row, cols.get("bb_upper"))
            lower = self._get_val(last_row, cols.get("bb_lower"))
            middle = self._get_val(last_row, cols.get("bb_mid"))
            close_val = self._get_val(last_row, "close")

            if pd.notna(upper) and pd.notna(lower) and pd.notna(middle) and pd.notna(close_val):
                bb_status = f"BB (U:{self._format_val(upper, price_precision)}, M:{self._format_val(middle, price_precision)}, L:{self._format_val(lower, price_precision)}): "
                signal = "Within Bands"
                color = NEON_YELLOW
                if close_val > upper:
                    signal, color = "Breakout Upper", NEON_RED  # Potential reversal/overbought
                    bb_status += f"{color}Price Above Upper Band{RESET}"
                elif close_val < lower:
                    signal, color = "Breakdown Lower", NEON_GREEN  # Potential reversal/oversold
                    bb_status += f"{color}Price Below Lower Band{RESET}"
                else:
                    # Optionally check proximity to middle band
                    signal = "Above Mid" if close_val > middle else "Below Mid"
                    bb_status += f"{NEON_YELLOW}Price {signal}{RESET}"
                signals["bbands_signal"] = signal
                summary.append(bb_status)
            else:
                signals["bbands_signal"] = "N/A"

        return summary, signals

    def _interpret_volume(self, last_row: pd.Series, prev_row: pd.Series) -> tuple[list[str], dict[str, str]]:
        summary, signals = [], {}
        flags = self.analysis_flags
        cols = self.col_names
        vol_precision = 0  # No decimals for volume typically

        # Volume MA Comparison
        if flags.get("volume_confirmation"):
            volume = self._get_val(last_row, "volume")
            vol_ma = self._get_val(last_row, cols.get("vol_ma"))  # Uses generated vol_ma name
            if (
                pd.notna(volume) and pd.notna(vol_ma) and vol_ma > 0
            ):  # Check vol_ma > 0 to avoid division errors implicitly
                vol_status = f"Volume ({self._format_val(volume, vol_precision)} vs MA:{self._format_val(vol_ma, vol_precision)}): "
                level = "Average"
                color = NEON_YELLOW
                # Use Decimal for comparison thresholds if inputs were Decimal
                try:
                    vol_dec = Decimal(str(volume))
                    vol_ma_dec = Decimal(str(vol_ma))
                    high_thresh = vol_ma_dec * Decimal("1.5")  # e.g., 50% above MA
                    low_thresh = vol_ma_dec * Decimal("0.7")  # e.g., 30% below MA

                    if vol_dec > high_thresh:
                        level, color = "High", NEON_GREEN
                    elif vol_dec < low_thresh:
                        level, color = "Low", NEON_RED
                except (InvalidOperation, TypeError):
                    # Fallback to float comparison if Decimal conversion fails
                    if volume > vol_ma * 1.5:
                        level, color = "High", NEON_GREEN
                    elif volume < vol_ma * 0.7:
                        level, color = "Low", NEON_RED

                signals["volume_level"] = level
                summary.append(vol_status + f"{color}{level} Volume{RESET}")
            else:
                signals["volume_level"] = "N/A"

        # OBV Trend
        if flags.get("obv_trend"):
            obv_now = self._get_val(last_row, cols.get("obv"))
            obv_prev = self._get_val(prev_row, cols.get("obv"))
            if pd.notna(obv_now) and pd.notna(obv_prev):
                trend = "Flat"
                color = NEON_YELLOW
                if obv_now > obv_prev:
                    trend, color = "Increasing", NEON_GREEN  # Confirming uptrend?
                elif obv_now < obv_prev:
                    trend, color = "Decreasing", NEON_RED  # Confirming downtrend?
                signals["obv_trend"] = trend
                summary.append(f"OBV Trend: {color}{trend}{RESET}")
            else:
                signals["obv_trend"] = "N/A"

        # ADOSC Trend (A/D Oscillator)
        if flags.get("adi_trend"):  # Flag name might be confusing, it's ADOSC
            adosc_now = self._get_val(last_row, cols.get("adosc"))
            adosc_prev = self._get_val(prev_row, cols.get("adosc"))
            if pd.notna(adosc_now) and pd.notna(adosc_prev):
                trend = "Flat"
                color = NEON_YELLOW
                # Simple interpretation: Positive & rising = Accumulation, Negative & falling = Distribution
                if adosc_now > 0 and adosc_now > adosc_prev:
                    trend, color = "Accumulation", NEON_GREEN
                elif adosc_now < 0 and adosc_now < adosc_prev:
                    trend, color = "Distribution", NEON_RED
                elif adosc_now > adosc_prev:
                    trend, color = "Increasing", NEON_GREEN  # Generic rise
                elif adosc_now < adosc_prev:
                    trend, color = "Decreasing", NEON_RED  # Generic fall
                signals["adosc_trend"] = trend  # Changed key from adi_trend
                summary.append(f"A/D Osc Trend: {color}{trend}{RESET} ({self._format_val(adosc_now, 2)})")
            else:
                signals["adosc_trend"] = "N/A"

        return summary, signals

    def _interpret_levels_orderbook(self, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> list[str]:
        summary = []
        price_precision = 4  # Precision for displaying levels
        summary.append(f"{NEON_BLUE}--- Levels & Orderbook ---{RESET}")

        current_price_float = float(current_price)  # For sorting distance

        if levels:
            # Sort levels by proximity to current price
            supports = sorted(
                levels.get("support", {}).items(), key=lambda item: abs(float(item[1]) - current_price_float)
            )
            resistances = sorted(
                levels.get("resistance", {}).items(), key=lambda item: abs(float(item[1]) - current_price_float)
            )
            pivot = levels.get("pivot")

            if pivot:
                summary.append(f"Pivot Point: ${self._format_val(pivot, price_precision)}")

            if supports:
                summary.append("Nearest Support:")
            for name, price in supports[:3]:  # Show top 3 nearest
                summary.append(f"  > {name}: ${self._format_val(price, price_precision)}")

            if resistances:
                summary.append("Nearest Resistance:")
            for name, price in resistances[:3]:  # Show top 3 nearest
                summary.append(f"  > {name}: ${self._format_val(price, price_precision)}")
        else:
            summary.append(f"{NEON_YELLOW}Levels calculation unavailable.{RESET}")

        if (
            orderbook_analysis
            and orderbook_analysis.get("total_bid_usd", 0) + orderbook_analysis.get("total_ask_usd", 0) > 0
        ):
            pressure = orderbook_analysis.get("pressure", "N/A")
            total_bid = orderbook_analysis.get("total_bid_usd", Decimal(0))
            total_ask = orderbook_analysis.get("total_ask_usd", Decimal(0))
            limit = self.orderbook_settings.get("limit", 50)

            summary.append(f"OB Pressure (Top {limit}): {pressure}")
            summary.append(f"OB Value (Bids): ${self._format_val(total_bid, 0)}")
            summary.append(f"OB Value (Asks): ${self._format_val(total_ask, 0)}")

            # Sort clusters by value, descending
            clusters = sorted(
                orderbook_analysis.get("clusters", []),
                key=lambda x: x.get("cluster_value_usd", Decimal(0)),
                reverse=True,
            )
            if clusters:
                summary.append(f"{NEON_PURPLE}Significant OB Clusters (Top 5 by Value):{RESET}")
                for cluster in clusters[:5]:
                    color = (
                        NEON_GREEN
                        if cluster.get("type") == "Support"
                        else NEON_RED
                        if cluster.get("type") == "Resistance"
                        else NEON_YELLOW
                    )
                    level_price_f = self._format_val(cluster.get("level_price"), price_precision)
                    cluster_val_f = self._format_val(cluster.get("cluster_value_usd"), 0)
                    avg_price_f = self._format_val(
                        cluster.get("avg_price"), price_precision
                    )  # Show avg price of cluster
                    summary.append(
                        f"  {color}{cluster.get('type')} near {cluster.get('level_name')} (${level_price_f}) - Value: ${cluster_val_f} (Avg Price: ${avg_price_f}){RESET}"
                    )
            else:
                summary.append(f"{NEON_YELLOW}No significant OB clusters found.{RESET}")
        else:
            summary.append(f"{NEON_YELLOW}Orderbook analysis unavailable or empty.{RESET}")

        return summary

    def _interpret_analysis(
        self, df: pd.DataFrame, current_price: Decimal, levels: dict, orderbook_analysis: dict
    ) -> dict:
        interpretation = {"summary": [], "signals": {}}
        if df.empty or len(df) < 2:
            # Check if we already logged this warning in analyze()
            if len(df) == 1:
                self.logger.debug(f"Interpretation skipped for {self.symbol} due to only 1 row of data.")
            else:
                self.logger.warning(f"Insufficient data ({len(df)} rows) for interpretation on {self.symbol}.")
            interpretation["summary"].append(f"{NEON_RED}Insufficient data for full interpretation.{RESET}")
            return interpretation

        try:
            # Ensure index is sequential if not already (should be from fetch_klines)
            df = df.reset_index(drop=True)
            last_row = df.iloc[-1]
            # Ensure prev_row exists and is valid
            prev_row = df.iloc[-2] if len(df) >= 2 else pd.Series(dtype="object")  # Empty series if only 1 row

            # --- Generate Interpretation Sections ---
            trend_summary, trend_signals = self._interpret_trend(last_row, prev_row)
            osc_summary, osc_signals = self._interpret_oscillators(last_row, prev_row)
            macd_summary, macd_signals = self._interpret_macd(last_row, prev_row)
            bb_summary, bb_signals = self._interpret_bbands(last_row)
            vol_summary, vol_signals = self._interpret_volume(last_row, prev_row)
            level_summary = self._interpret_levels_orderbook(current_price, levels, orderbook_analysis)

            # --- Combine Summaries ---
            interpretation["summary"].append(f"{NEON_BLUE}--- Trend & Momentum ---{RESET}")
            interpretation["summary"].extend(trend_summary)
            interpretation["summary"].extend(macd_summary)  # Group MACD with trend

            interpretation["summary"].append(f"{NEON_BLUE}--- Oscillators & Overbought/Oversold ---{RESET}")
            interpretation["summary"].extend(osc_summary)

            interpretation["summary"].append(f"{NEON_BLUE}--- Volatility & Volume ---{RESET}")
            interpretation["summary"].extend(bb_summary)
            interpretation["summary"].extend(vol_summary)

            interpretation["summary"].extend(level_summary)  # Levels/OB added last

            # --- Combine Signals ---
            interpretation["signals"].update(trend_signals)
            interpretation["signals"].update(osc_signals)
            interpretation["signals"].update(macd_signals)
            interpretation["signals"].update(bb_signals)
            interpretation["signals"].update(vol_signals)
            # Add Level/OB signals if defined (e.g., "PriceNearSupportCluster": True)

        except IndexError:
            self.logger.error(f"IndexError during interpretation for {self.symbol}. DataFrame length: {len(df)}")
            interpretation["summary"].append(f"{NEON_RED}Error accessing data for interpretation.{RESET}")
        except Exception as e:
            self.logger.exception(f"Unexpected error during analysis interpretation for {self.symbol}: {e}")
            interpretation["summary"].append(f"{NEON_RED}Error during interpretation: {e}{RESET}")

        return interpretation

    def analyze(self, df_klines: pd.DataFrame, current_price: Decimal, orderbook: dict | None) -> dict:
        """Main analysis function."""
        analysis_result = {
            "symbol": self.symbol,
            "timestamp": datetime.now(TIMEZONE).isoformat(),
            "current_price": self._format_val(current_price, 4),  # Format early
            "kline_interval": "N/A",
            "levels": {},
            "orderbook_analysis": {},
            "interpretation": {"summary": [f"{NEON_RED}Analysis could not be performed.{RESET}"], "signals": {}},
            "raw_indicators": {},
        }

        if df_klines.empty:
            self.logger.error(f"Kline data is empty for {self.symbol}, cannot perform analysis.")
            # Keep default error message in analysis_result
            return analysis_result
        elif len(df_klines) < 2:
            # Log warning but proceed with analysis as much as possible
            self.logger.warning(
                f"Kline data has only {len(df_klines)} row(s) for {self.symbol}, analysis may be incomplete."
            )
            # Update the summary to reflect this warning, but don't overwrite potential results later
            analysis_result["interpretation"]["summary"] = [
                f"{NEON_YELLOW}Warning: Insufficient kline data ({len(df_klines)} rows) for full analysis.{RESET}"
            ]

        try:
            # Infer Interval from timestamps if possible
            if len(df_klines) >= 2:
                # Calculate the difference between consecutive timestamps
                time_diffs = df_klines["timestamp"].diff()
                # Find the most common time difference (mode) as the interval
                # Drop NA from the first row's diff
                common_diff = time_diffs.dropna().mode()
                if not common_diff.empty:
                    interval_td = common_diff[0]
                    # Convert timedelta to a readable string (e.g., '0 days 01:00:00')
                    analysis_result["kline_interval"] = str(interval_td)  # Or format more nicely
                    # Optional: Map back to Bybit interval string if needed
                    # seconds = interval_td.total_seconds() / 60 # Interval in minutes
                    # interval_key = REVERSE_CCXT_INTERVAL_MAP.get(...) # Match based on seconds/minutes
                else:
                    analysis_result["kline_interval"] = "Variable/Unknown"

            # --- Calculate Indicators ---
            df_with_indicators = self._calculate_indicators(df_klines)
            if df_with_indicators.empty:
                self.logger.error(f"DataFrame empty after indicator calculation for {self.symbol}.")
                # analysis_result already has default error message
                return analysis_result  # Return early if indicators fail critically

            # Store raw indicator values from the last row
            if not df_with_indicators.empty:
                last_row_indicators = df_with_indicators.iloc[-1]
                analysis_result["raw_indicators"] = {
                    k: self._format_val(v, 5)
                    if isinstance(v, (Decimal, float, np.floating))
                    # Use higher precision for raw data
                    else (v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else str(v) if pd.notna(v) else None)
                    for k, v in last_row_indicators.items()
                    if k != "timestamp"  # Exclude timestamp from raw indicators dict
                }

            # --- Calculate Levels ---
            levels = self._calculate_levels(df_with_indicators, current_price)
            # Format levels for output
            analysis_result["levels"] = {
                k: {name: self._format_val(price, 4) for name, price in v.items()}
                if isinstance(v, dict)
                else (self._format_val(v, 4) if isinstance(v, Decimal) else v)
                for k, v in levels.items()
            }

            # --- Analyze Orderbook ---
            orderbook_analysis_raw = self._analyze_orderbook(orderbook, current_price, levels)
            # Format orderbook analysis for output
            analysis_result["orderbook_analysis"] = {
                "pressure": orderbook_analysis_raw.get("pressure", "N/A"),
                "total_bid_usd": self._format_val(orderbook_analysis_raw.get("total_bid_usd", Decimal(0)), 0),
                "total_ask_usd": self._format_val(orderbook_analysis_raw.get("total_ask_usd", Decimal(0)), 0),
                "clusters": [
                    {
                        "type": c.get("type"),
                        "level_name": c.get("level_name"),
                        "level_price": self._format_val(c.get("level_price"), 4),
                        "cluster_value_usd": self._format_val(c.get("cluster_value_usd"), 0),
                        "avg_price": self._format_val(c.get("avg_price"), 4),
                        # Keep raw range for potential internal use, format later if needed
                        # "price_range": (self._format_val(c.get("price_range", (np.nan, np.nan))[0], 4), self._format_val(c.get("price_range", (np.nan, np.nan))[1], 4))
                    }
                    for c in orderbook_analysis_raw.get("clusters", [])
                ],
            }

            # --- Interpret Results ---
            # Pass the raw levels and OB analysis dicts to interpretation
            interpretation = self._interpret_analysis(df_with_indicators, current_price, levels, orderbook_analysis_raw)
            # Overwrite the initial warning/error message if interpretation was successful
            if interpretation["summary"]:
                analysis_result["interpretation"] = interpretation
            else:
                # Keep the initial warning/error message if interpretation failed
                pass

        except Exception as e:
            self.logger.exception(f"Critical error during analysis pipeline for {self.symbol}: {e}")
            analysis_result["interpretation"]["summary"] = [f"{NEON_RED}Critical Analysis Error: {e}{RESET}"]
            # Keep other potentially calculated fields like price, interval etc.

        return analysis_result

    def format_analysis_output(self, analysis_result: dict) -> str:
        symbol = analysis_result.get("symbol", "N/A")
        timestamp_str = analysis_result.get("timestamp", "N/A")
        try:
            # Attempt to parse and format timestamp nicely
            dt_obj = datetime.fromisoformat(timestamp_str).astimezone(TIMEZONE)
            ts_formatted = dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")
        except (ValueError, TypeError):
            ts_formatted = timestamp_str  # Fallback to raw string

        price = analysis_result.get("current_price", "N/A")
        interval = analysis_result.get("kline_interval", "N/A")

        output = f"\n--- Analysis Report for {NEON_PURPLE}{symbol}{RESET} --- ({ts_formatted})\n"
        output += f"{NEON_BLUE}Interval:{RESET} {interval}    {NEON_BLUE}Current Price:{RESET} ${price}\n"

        interpretation = analysis_result.get("interpretation", {})
        summary = interpretation.get("summary", [])  # This is already formatted with colors

        if summary:
            # Join the formatted lines from the summary list
            output += "\n" + "\n".join(summary) + "\n"
        else:
            output += f"\n{NEON_YELLOW}No interpretation summary available.{RESET}\n"

        # Optionally add raw signal values if needed for quick glance
        # signals = interpretation.get("signals", {})
        # if signals:
        #     output += f"\n{NEON_BLUE}--- Signals ---{RESET}\n"
        #     output += json.dumps(signals, indent=2) + "\n"

        return output


async def run_analysis_loop(
    symbol: str,
    interval_config: str,
    client: BybitCCXTClient,
    analyzer: TradingAnalyzer,
    logger_instance: logging.Logger,
) -> None:
    analysis_interval_sec = CONFIG.get("analysis_interval_seconds", 30)
    kline_limit = CONFIG.get("kline_limit", 200)
    orderbook_limit = analyzer.orderbook_settings.get("limit", 50)

    if analysis_interval_sec < 10:
        logger_instance.warning(
            f"Analysis interval ({analysis_interval_sec}s) is very short. Ensure system and API rate limits can handle the load."
        )

    while True:
        start_time = time.monotonic()
        logger_instance.debug(f"--- Starting Analysis Cycle for {symbol} ---")

        analysis_result = None  # Define outside try block

        try:
            # --- Fetch Data Concurrently ---
            # Create tasks for fetching data
            price_task = asyncio.create_task(client.fetch_current_price(symbol))
            klines_task = asyncio.create_task(client.fetch_klines(symbol, interval_config, kline_limit))
            orderbook_task = asyncio.create_task(client.fetch_orderbook(symbol, orderbook_limit))

            # Await mandatory data first (price & klines)
            await asyncio.gather(price_task, klines_task)

            current_price = price_task.result()
            df_klines = klines_task.result()

            # Check if mandatory data was fetched successfully
            if current_price is None:
                logger_instance.error(f"Failed to fetch current price for {symbol}. Skipping analysis cycle.")
                await async_sleep(RETRY_DELAY_SECONDS)  # Wait before retrying fetch
                continue
            if df_klines.empty:
                logger_instance.error(f"Failed to fetch kline data for {symbol}. Skipping analysis cycle.")
                await async_sleep(RETRY_DELAY_SECONDS)  # Wait before retrying fetch
                continue

            # Await orderbook (optional data) with a timeout
            orderbook = None  # Default to None
            try:
                # Wait for the orderbook task for a limited time
                orderbook = await asyncio.wait_for(orderbook_task, timeout=15.0)  # 15 second timeout
            except TimeoutError:
                logger_instance.warning(f"Fetching orderbook for {symbol} timed out. Proceeding without it.")
                # Attempt to cancel the lingering task (best effort)
                try:
                    orderbook_task.cancel()
                    await orderbook_task  # Allow cancellation to propagate
                except asyncio.CancelledError:
                    logger_instance.debug("Orderbook task successfully cancelled after timeout.")
                except Exception as cancel_err:
                    logger_instance.error(f"Error trying to cancel orderbook task: {cancel_err}")
            except Exception as ob_err:
                logger_instance.error(f"Error fetching or processing orderbook task for {symbol}: {ob_err}")
                # Ensure task is handled if it errored before timeout
                if not orderbook_task.done():
                    try:
                        orderbook_task.cancel()
                        await orderbook_task
                    except:
                        pass  # Ignore errors during cleanup

            # --- Perform Analysis ---
            analysis_result = analyzer.analyze(df_klines, current_price, orderbook)

            # --- Output and Logging ---
            # Use the dedicated formatting function
            output_string = analyzer.format_analysis_output(analysis_result)

            # Log the human-readable formatted output to INFO level
            # Remove color codes for file logging if desired, or keep them if terminal supports
            # For simplicity, log with color codes for now.
            logger_instance.info(output_string)

            # Log structured JSON data at DEBUG level
            if logger_instance.isEnabledFor(logging.DEBUG):
                try:
                    # Create a deep copy to modify for logging without affecting original
                    log_data = json.loads(
                        json.dumps(analysis_result, default=str)
                    )  # Convert Decimals etc. to strings first

                    # Remove color codes from the summary within the copied data
                    if "interpretation" in log_data and "summary" in log_data["interpretation"]:
                        log_data["interpretation"]["summary"] = [
                            re.sub(r"\x1b\[[0-9;]*m", "", line) for line in log_data["interpretation"]["summary"]
                        ]
                    # Remove color codes from orderbook pressure
                    if "orderbook_analysis" in log_data and "pressure" in log_data["orderbook_analysis"]:
                        log_data["orderbook_analysis"]["pressure"] = re.sub(
                            r"\x1b\[[0-9;]*m", "", log_data["orderbook_analysis"]["pressure"]
                        )

                    logger_instance.debug(f"Analysis Result (JSON):\n{json.dumps(log_data, indent=2)}")
                except Exception as json_err:
                    logger_instance.error(f"Error preparing analysis result for JSON debug logging: {json_err}")

        except ccxt.AuthenticationError as e:
            logger_instance.critical(
                f"{NEON_RED}Authentication Error: {e}. Check API Key/Secret. Stopping analysis loop for {symbol}.{RESET}"
            )
            break  # Stop the loop for this symbol on auth errors
        except ccxt.InvalidNonce as e:
            logger_instance.critical(
                f"{NEON_RED}Invalid Nonce Error: {e}. Check system time sync. Stopping analysis loop for {symbol}.{RESET}"
            )
            break  # Stop the loop - likely requires manual intervention
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            # These errors are usually related to trading actions, not analysis fetching
            # Log them but continue the analysis loop.
            logger_instance.error(f"{NEON_RED}Trading-related ccxt error occurred: {e}. Continuing analysis.{RESET}")
        except Exception as e:
            logger_instance.exception(
                f"{NEON_RED}An unexpected error occurred in the main analysis loop for {symbol}: {e}{RESET}"
            )
            # Log the full stack trace automatically via .exception()

        # --- Sleep Management ---
        end_time = time.monotonic()
        elapsed_time = end_time - start_time
        sleep_duration = max(0.1, analysis_interval_sec - elapsed_time)  # Ensure minimum sleep of 0.1s
        logger_instance.debug(
            f"Analysis cycle for {symbol} took {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s."
        )

        if elapsed_time > analysis_interval_sec:
            logger_instance.warning(
                f"Analysis cycle for {symbol} took longer ({elapsed_time:.2f}s) than configured interval ({analysis_interval_sec}s). Consider increasing interval or checking performance."
            )

        await async_sleep(sleep_duration)
        # --- End of Loop ---

    logger_instance.info(f"Analysis loop stopped for {symbol}.")


async def main() -> None:
    global logger  # Allow main to potentially set the logger after symbol is known

    # 1. Initial Setup (Temporary Logger, Market Loading)
    temp_logger = setup_logger("INIT")  # Logger for initialization phase
    client = BybitCCXTClient(API_KEY, API_SECRET, CCXT_BASE_URL, temp_logger)

    temp_logger.info("Attempting to load markets...")
    markets_loaded = await client.initialize_markets()

    valid_symbols = []
    if markets_loaded:
        if client.markets:
            valid_symbols = list(client.markets.keys())
            temp_logger.info(f"Loaded {len(valid_symbols)} market symbols.")
        else:
            temp_logger.error("Market loading reported success but client.markets is empty.")
            markets_loaded = False  # Treat as failure
    else:
        temp_logger.critical(
            f"{NEON_RED}Failed to load markets after multiple retries. Cannot proceed. Exiting.{RESET}"
        )
        await client.close()  # Close the client even on failure
        logging.shutdown()
        return

    # No need to keep the initial client instance open just for market data
    initial_market_data = client.markets  # Store the loaded market data
    await client.close()
    temp_logger.info("Initial client closed after loading markets.")

    # 2. User Input: Symbol
    symbol = ""
    while True:
        try:
            symbol_input = input(f"{NEON_BLUE}Enter trading symbol (e.g., BTC/USDT): {RESET}").upper().strip()
            if not symbol_input:
                continue

            # Simple standardization (BTCUSDT -> BTC/USDT) - adjust logic as needed
            formatted_symbol = symbol_input
            if "/" not in symbol_input:
                # Guess common quote currencies
                for quote in ["USDT", "USD", "BTC", "ETH", "EUR"]:
                    if symbol_input.endswith(quote) and len(symbol_input) > len(quote):
                        base = symbol_input[: -len(quote)]
                        potential_symbol = f"{base}/{quote}"
                        if potential_symbol in valid_symbols:
                            formatted_symbol = potential_symbol
                            break  # Found a match

            # Validate against loaded symbols
            if formatted_symbol in valid_symbols:
                symbol = formatted_symbol
                break
            else:
                # Suggest similar symbols (basic substring search)
                find_similar = [s for s in valid_symbols if symbol_input in s or formatted_symbol.split("/")[0] in s]
                if find_similar:
                    ", ".join(find_similar[:5])  # Limit suggestions
        except EOFError:
            return
        except KeyboardInterrupt:
            return

    # 3. User Input: Interval
    interval_key = ""
    default_interval = CONFIG.get("indicator_settings", {}).get("default_interval", "15")
    while True:
        try:
            interval_input_raw = input(
                f"{NEON_BLUE}Enter timeframe [{', '.join(VALID_INTERVALS)}] (default: {default_interval}): {RESET}"
            ).strip()
            interval_input = interval_input_raw  # Keep raw for potential mapping

            if not interval_input:
                interval_key = default_interval
                break

            # Check if input is directly in VALID_INTERVALS (Bybit internal format)
            if interval_input in VALID_INTERVALS:
                interval_key = interval_input
                break
            # Check if input is in CCXT format (e.g., '1h', '1d') and map back
            elif interval_input in REVERSE_CCXT_INTERVAL_MAP:
                interval_key = REVERSE_CCXT_INTERVAL_MAP[interval_input]
                break
            else:
                pass
        except EOFError:
            return
        except KeyboardInterrupt:
            return

    # 4. Final Initialization (Symbol-Specific Logger, Re-create Client, Analyzer)
    logger = setup_logger(symbol)  # Setup logger specific to the chosen symbol
    logger.info(f"Logger initialized for symbol {symbol}.")

    # Re-create the client, now with the proper symbol-specific logger
    client = BybitCCXTClient(API_KEY, API_SECRET, CCXT_BASE_URL, logger)
    # Assign the previously loaded market data to the new client instance
    client.markets = initial_market_data
    logger.info(f"Bybit client re-initialized for {symbol}.")

    analyzer = TradingAnalyzer(CONFIG, logger, symbol)
    logger.info(f"Trading analyzer initialized for {symbol}.")

    ccxt_interval = CCXT_INTERVAL_MAP.get(interval_key, "N/A")
    market_type = client.get_market_category(symbol)

    logger.info(
        f"Starting analysis for {NEON_PURPLE}{symbol}{RESET} (Category: {market_type}) on interval {NEON_PURPLE}{interval_key}{RESET} (CCXT: {ccxt_interval})."
    )
    logger.info(f"Using API Environment: {NEON_YELLOW}{API_ENV.upper()}{RESET} ({client.exchange.urls['api']})")
    logger.info(f"Analysis loop interval: {CONFIG.get('analysis_interval_seconds', 30)} seconds.")
    logger.info(f"Kline fetch limit: {CONFIG.get('kline_limit', 200)} candles.")

    # 5. Run the Main Loop
    try:
        await run_analysis_loop(symbol, interval_key, client, analyzer, logger)
    except KeyboardInterrupt:
        logger.info(f"{NEON_YELLOW}Analysis loop interrupted by user.{RESET}")
    except Exception as e:
        # This catches errors that might escape the loop's internal handling
        logger.critical(
            f"{NEON_RED}A critical error occurred outside the main analysis loop: {e}{RESET}", exc_info=True
        )
    finally:
        # 6. Cleanup
        logger.info("Shutting down application components...")
        await client.close()  # Ensure client connection is closed
        logger.info("Application finished.")
        # Explicitly close logger handlers and shutdown logging
        if logger:
            handlers = logger.handlers[:]
            for handler in handlers:
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except Exception:
                    pass
        logging.shutdown()


if __name__ == "__main__":
    # Main execution block
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch interruption if it happens before async loop starts or during user input
        pass
    except Exception:
        # Catch any other unexpected top-level errors
        traceback.print_exc()
    finally:
        # Ensure style reset happens on exit
        pass
