import asyncio
import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import pandas_ta as ta  # Import pandas_ta
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Configuration & Constants ---

# Load environment variables
load_dotenv()

# Set decimal precision
getcontext().prec = 18  # Increased precision for crypto

# Initialize Colorama
init(autoreset=True)

# API Credentials
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

# API Configuration
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
# Use ccxt standard URLs
CCXT_URLS = {
    "test": "https://api-testnet.bybit.com",
    "prod": "https://api.bybit.com",
}
# Select 'prod' or 'test'
API_ENV = os.getenv("API_ENVIRONMENT", "prod").lower()
CCXT_BASE_URL = CCXT_URLS.get(API_ENV, CCXT_URLS["prod"])


# File/Directory Configuration
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Timezone
DEFAULT_TIMEZONE = "America/Chicago"
try:
    TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", DEFAULT_TIMEZONE))
except Exception:
    print(
        f"{Fore.YELLOW}Warning: Could not load timezone '{os.getenv('TIMEZONE', DEFAULT_TIMEZONE)}'. Using UTC.{Style.RESET_ALL}"
    )
    TIMEZONE = ZoneInfo("UTC")

# Retry Configuration
MAX_API_RETRIES = 5
RETRY_DELAY_SECONDS = 5
CCXT_TIMEOUT_MS = 20000  # 20 seconds
VALID_INTERVALS = [
    "1",
    "3",
    "5",
    "15",
    "30",
    "60",
    "120",
    "240",
    "360",
    "720",
    "D",
    "W",
    "M",
]
# Map ccxt intervals to display/config intervals if needed, Bybit uses minutes directly mostly
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

# Color Constants
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
RESET = Style.RESET_ALL

# --- Configuration Loading ---


def load_config(filepath: str) -> dict:
    default_config = {
        "analysis_interval_seconds": 30,
        "kline_limit": 200,
        "indicator_settings": {
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
            "momentum_crossover": True,
            "volume_confirmation": True,
            "rsi_divergence": False,  # Requires more complex state tracking
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
        "orderbook_settings": {
            "limit": 50,  # Fetch fewer levels, more frequently
            "cluster_threshold_usd": 10000,  # Define cluster size in quote currency (USD for USDT pairs)
            "cluster_proximity_pct": 0.1,  # Check for clusters within 0.1% of S/R levels
        },
        "logging": {
            "level": "INFO",
            "rotation_max_bytes": 10 * 1024 * 1024,  # 10 MB
            "rotation_backup_count": 5,
        },
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            print(
                f"{NEON_YELLOW}Created new config file '{filepath}' with defaults.{RESET}"
            )
            return default_config
        except OSError as e:
            print(f"{NEON_RED}Error creating default config file: {e}{RESET}")
            print(f"{NEON_YELLOW}Loading internal defaults.{RESET}")
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            user_config = json.load(f)
            # Simple merge (user config overrides defaults) - consider deep merge for nested dicts if needed
            merged_config = default_config.copy()
            merged_config.update(user_config)
            # Validate structure for key areas
            for key in [
                "indicator_settings",
                "analysis_flags",
                "thresholds",
                "orderbook_settings",
                "logging",
            ]:
                if key in user_config and isinstance(user_config[key], dict):
                    merged_config[key] = {**default_config[key], **user_config[key]}
            return merged_config
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        print(f"{NEON_RED}Error loading/parsing config file '{filepath}': {e}{RESET}")
        print(f"{NEON_YELLOW}Loading internal defaults.{RESET}")
        return default_config


CONFIG = load_config(CONFIG_FILE)

# --- Logging Setup ---


class SensitiveFormatter(logging.Formatter):
    """Masks sensitive information in log records."""

    def format(self, record):
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***")
        return msg


def setup_logger(symbol: str) -> logging.Logger:
    log_filename = os.path.join(
        LOG_DIRECTORY,
        f"{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.log",
    )
    logger = logging.getLogger(symbol)
    log_level = getattr(logging, CONFIG["logging"]["level"].upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    try:
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=CONFIG["logging"]["rotation_max_bytes"],
            backupCount=CONFIG["logging"]["rotation_backup_count"],
            encoding="utf-8",
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Failed to set up file logger: {e}{RESET}")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        NEON_BLUE
        + "%(asctime)s"
        + RESET
        + " [%(levelname)s] "
        + NEON_YELLOW
        + f"{symbol}"
        + RESET
        + " - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(log_level)  # Ensure console also respects level
    logger.addHandler(stream_handler)

    return logger


# Global logger instance (initialized later in main)
logger: logging.Logger | None = None

# --- Async Utilities ---


async def async_sleep(seconds: float) -> None:
    """Asynchronous sleep function."""
    await asyncio.sleep(seconds)


# --- API Interaction (ccxt) ---


class BybitCCXTClient:
    """Wrapper for ccxt Bybit client with retry logic."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        logger_instance: logging.Logger,
    ):
        self.logger = logger_instance
        self.exchange = ccxt_async.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,  # Enable ccxt's built-in rate limiter
            "options": {
                "defaultType": "linear",  # Or 'inverse' or 'spot'
                "adjustForTimeDifference": True,  # Auto-sync time
            },
            "urls": {"api": base_url},
            "timeout": CCXT_TIMEOUT_MS,
        })

    async def close(self):
        """Close the underlying exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self.logger.info("Closed ccxt exchange connection.")

    async def fetch_with_retry(self, method_name: str, *args, **kwargs) -> Any | None:
        """Calls a ccxt method with retry logic."""
        retries = MAX_API_RETRIES
        current_delay = RETRY_DELAY_SECONDS
        for attempt in range(retries):
            try:
                method = getattr(self.exchange, method_name)
                result = await method(*args, **kwargs)
                return result
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
            ) as e:
                self.logger.warning(
                    f"{NEON_YELLOW}Network/Timeout error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}"
                )
            except ccxt.ExchangeError as e:
                # Specific handling for common retryable errors
                if (
                    "Too Many Visits" in str(e)
                    or "rate limit" in str(e).lower()
                    or isinstance(e, ccxt.RateLimitExceeded)
                ):
                    self.logger.warning(
                        f"{NEON_YELLOW}Rate limit exceeded calling {method_name} (Attempt {attempt + 1}/{retries}). Retrying in {current_delay}s...{RESET}"
                    )
                elif "System busy" in str(e) or "service unavailable" in str(e).lower():
                    self.logger.warning(
                        f"{NEON_YELLOW}Server busy error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay}s...{RESET}"
                    )
                else:
                    self.logger.error(
                        f"{NEON_RED}Non-retryable ccxt ExchangeError calling {method_name}: {e}{RESET}"
                    )
                    return None  # Don't retry for other exchange errors
            except Exception as e:
                self.logger.exception(
                    f"{NEON_RED}Unexpected error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}{RESET}"
                )
                # Optional: Decide if you want to retry on *any* exception

            if attempt < retries - 1:
                await async_sleep(current_delay)
                current_delay = min(
                    current_delay * 1.5, 30
                )  # Exponential backoff with cap
            else:
                self.logger.error(
                    f"{NEON_RED}Max retries reached for {method_name}.{RESET}"
                )
                return None
        return None  # Should not be reached, but adding for safety

    async def fetch_ticker(self, symbol: str) -> dict | None:
        """Fetches ticker information for a symbol."""
        self.logger.debug(f"Fetching ticker for {symbol}")
        # Bybit v5 uses fetch_tickers for multiple, or specific symbol
        tickers = await self.fetch_with_retry(
            "fetch_tickers", symbols=[symbol], params={"category": "linear"}
        )
        if tickers and symbol in tickers:
            return tickers[symbol]
        self.logger.error(f"Could not fetch ticker for {symbol}")
        return None

    async def fetch_current_price(self, symbol: str) -> Decimal | None:
        """Fetches the last traded price for a symbol."""
        ticker = await self.fetch_ticker(symbol)
        if ticker and "last" in ticker and ticker["last"] is not None:
            try:
                return Decimal(str(ticker["last"]))
            except (InvalidOperation, TypeError) as e:
                self.logger.error(
                    f"Error converting last price '{ticker['last']}' to Decimal: {e}"
                )
                return None
        self.logger.warning(f"Last price not found in ticker data for {symbol}")
        return None

    async def fetch_klines(
        self, symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """Fetches OHLCV kline data."""
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            self.logger.error(f"Invalid timeframe '{timeframe}' provided.")
            return pd.DataFrame()

        self.logger.debug(
            f"Fetching {limit} klines for {symbol} with interval {ccxt_timeframe}"
        )
        klines = await self.fetch_with_retry(
            "fetch_ohlcv",
            symbol,
            timeframe=ccxt_timeframe,
            limit=limit,
            params={"category": "linear"},
        )

        if klines is None or len(klines) == 0:
            self.logger.warning(
                f"No kline data returned for {symbol} interval {ccxt_timeframe}"
            )
            return pd.DataFrame()

        try:
            df = pd.DataFrame(
                klines, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            # Convert timestamp to datetime (ccxt returns ms)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms", utc=True
            ).dt.tz_convert(TIMEZONE)
            # Convert OHLCV to numeric, coercing errors
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(
                subset=["open", "high", "low", "close", "volume"], inplace=True
            )  # Drop rows with NaN in essential columns

            if df.empty:
                self.logger.warning(
                    f"Kline data for {symbol} was empty after cleaning."
                )
                return pd.DataFrame()

            # Optional: Convert to Decimal if high precision needed later, but slows down pandas_ta
            # for col in ['open', 'high', 'low', 'close']:
            #     df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
            # df['volume'] = df['volume'].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)

            self.logger.debug(f"Successfully fetched and processed {len(df)} klines.")
            return df.sort_values(by="timestamp").reset_index(
                drop=True
            )  # Ensure sorted

        except Exception as e:
            self.logger.exception(f"Error processing kline data: {e}")
            return pd.DataFrame()

    async def fetch_orderbook(self, symbol: str, limit: int) -> dict | None:
        """Fetches order book data."""
        self.logger.debug(f"Fetching order book for {symbol} with limit {limit}")
        # Bybit v5 requires category, ccxt handles this via params
        orderbook = await self.fetch_with_retry(
            "fetch_order_book", symbol, limit=limit, params={"category": "linear"}
        )
        if orderbook:
            # Basic validation
            if "bids" in orderbook and "asks" in orderbook:
                self.logger.debug(
                    f"Fetched order book with {len(orderbook['bids'])} bids and {len(orderbook['asks'])} asks."
                )
                return orderbook
            else:
                self.logger.warning("Fetched order book data is missing bids or asks.")
                return None
        else:
            self.logger.warning(
                f"Failed to fetch order book for {symbol} after retries."
            )
            return None


# --- Trading Analysis ---


class TradingAnalyzer:
    """Performs technical analysis on kline data."""

    def __init__(self, config: dict, logger_instance: logging.Logger, symbol: str):
        self.config = config
        self.logger = logger_instance
        self.symbol = symbol
        self.indicator_settings = config.get("indicator_settings", {})
        self.analysis_flags = config.get("analysis_flags", {})
        self.thresholds = config.get("thresholds", {})
        self.orderbook_settings = config.get("orderbook_settings", {})

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all enabled indicators using pandas_ta."""
        if df.empty:
            self.logger.warning("Cannot calculate indicators on empty DataFrame.")
            return df

        df = df.copy()  # Work on a copy

        # Create a Strategy using pandas_ta
        strategy = ta.Strategy(
            name="TradingAnalysis",
            description="Comprehensive TA using pandas_ta",
            ta=[
                {
                    "kind": "sma",
                    "length": self.indicator_settings.get("sma_short_period", 10),
                },
                {
                    "kind": "sma",
                    "length": self.indicator_settings.get("sma_long_period", 50),
                },
                {
                    "kind": "ema",
                    "length": self.indicator_settings.get("ema_short_period", 12),
                },
                {
                    "kind": "ema",
                    "length": self.indicator_settings.get("ema_long_period", 26),
                },
                {
                    "kind": "rsi",
                    "length": self.indicator_settings.get("rsi_period", 14),
                },
                {
                    "kind": "stochrsi",
                    "length": self.indicator_settings.get("stoch_rsi_period", 14),
                    "rsi_length": self.indicator_settings.get(
                        "rsi_period", 14
                    ),  # Needs base RSI length
                    "k": self.indicator_settings.get("stoch_k_period", 3),
                    "d": self.indicator_settings.get("stoch_d_period", 3),
                },
                {
                    "kind": "macd",
                    "fast": self.indicator_settings.get("macd_fast", 12),
                    "slow": self.indicator_settings.get("macd_slow", 26),
                    "signal": self.indicator_settings.get("macd_signal", 9),
                },
                {
                    "kind": "bbands",
                    "length": self.indicator_settings.get("bollinger_bands_period", 20),
                    "std": self.indicator_settings.get("bollinger_bands_std_dev", 2),
                },
                {
                    "kind": "atr",
                    "length": self.indicator_settings.get("atr_period", 14),
                },
                {
                    "kind": "cci",
                    "length": self.indicator_settings.get("cci_period", 20),
                },
                {
                    "kind": "willr",
                    "length": self.indicator_settings.get("williams_r_period", 14),
                },  # Williams %R
                {
                    "kind": "mfi",
                    "length": self.indicator_settings.get("mfi_period", 14),
                },
                {
                    "kind": "adx",
                    "length": self.indicator_settings.get("adx_period", 14),
                },
                {"kind": "obv"},  # On Balance Volume
                {
                    "kind": "adosc"
                },  # Accumulation/Distribution Oscillator (similar concept to ADI)
                {
                    "kind": "psar",
                    "step": self.indicator_settings.get("psar_step", 0.02),
                    "max_step": self.indicator_settings.get("psar_max_step", 0.2),
                },
                # Add others as needed, checking pandas_ta documentation for kind names and parameters
            ],
        )

        try:
            # Run the strategy
            df.ta.strategy(strategy, timed=False)  # Set timed=True to benchmark
            self.logger.debug(f"Calculated indicators: {list(df.columns)}")
        except Exception as e:
            self.logger.exception(f"Error calculating indicators using pandas_ta: {e}")
            # Return df possibly partially populated or original if error was early

        # Custom calculations if needed (e.g., Momentum MA crossover not directly in pandas_ta)
        mom_period = self.indicator_settings.get("momentum_period", 10)
        if mom_period > 0 and "close" in df.columns:
            df[f"MOM_{mom_period}"] = df["close"].diff(mom_period)
            # Example: Momentum MA Crossover
            # mom_ma_short = self.indicator_settings.get("momentum_ma_short", 12)
            # mom_ma_long = self.indicator_settings.get("momentum_ma_long", 26)
            # if mom_ma_short > 0:
            #     df[f'MOM_MA_{mom_ma_short}'] = df[f'MOM_{mom_period}'].rolling(window=mom_ma_short).mean()
            # if mom_ma_long > 0:
            #     df[f'MOM_MA_{mom_ma_long}'] = df[f'MOM_{mom_period}'].rolling(window=mom_ma_long).mean()

        vol_ma_period = self.indicator_settings.get("volume_ma_period", 20)
        if vol_ma_period > 0 and "volume" in df.columns:
            df[f"VOL_MA_{vol_ma_period}"] = (
                df["volume"].rolling(window=vol_ma_period).mean()
            )

        return df

    def _calculate_levels(self, df: pd.DataFrame, current_price: Decimal) -> dict:
        """Calculates support and resistance levels (Fibonacci, Pivots)."""
        levels = {"support": {}, "resistance": {}, "pivot": None}
        if df.empty or len(df) < 2:
            self.logger.warning("Insufficient data for level calculation.")
            return levels

        high = df["high"].max()
        low = df["low"].min()
        close = df["close"].iloc[-1]  # Use latest close from data frame for pivots

        try:
            # Ensure numeric types for calculations
            high_f = float(high)
            low_f = float(low)
            close_f = float(close)
            current_price_f = float(current_price)

            # Fibonacci Retracement
            diff = high_f - low_f
            if diff > 1e-9:  # Check for non-zero difference
                fib_levels = {
                    "Fib 23.6%": high_f - diff * 0.236,
                    "Fib 38.2%": high_f - diff * 0.382,
                    "Fib 50.0%": high_f - diff * 0.5,
                    "Fib 61.8%": high_f - diff * 0.618,
                    "Fib 78.6%": high_f - diff * 0.786,
                    # Add more if needed: 88.6%, etc.
                }
                for label, value in fib_levels.items():
                    value_dec = Decimal(str(value))
                    if value_dec < current_price:
                        levels["support"][label] = value_dec
                    else:
                        levels["resistance"][label] = value_dec

            # Standard Pivot Points (using Previous High, Low, Close might be more traditional)
            # Using max/min/last close of the period for simplicity here
            pivot = (high_f + low_f + close_f) / 3
            levels["pivot"] = Decimal(str(pivot))
            pivot_levels = {
                "R1": (2 * pivot) - low_f,
                "S1": (2 * pivot) - high_f,
                "R2": pivot + (high_f - low_f),
                "S2": pivot - (high_f - low_f),
                "R3": high_f + 2 * (pivot - low_f),
                "S3": low_f - 2 * (high_f - pivot),
            }
            for label, value in pivot_levels.items():
                value_dec = Decimal(str(value))
                if value_dec < current_price:
                    levels["support"][label] = value_dec
                else:
                    levels["resistance"][label] = value_dec

        except (TypeError, ValueError, InvalidOperation) as e:
            self.logger.error(f"Error calculating levels: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error calculating levels: {e}")

        return levels

    def _analyze_orderbook(
        self, orderbook: dict, current_price: Decimal, levels: dict
    ) -> dict:
        """Analyzes order book for clusters near support/resistance levels."""
        analysis = {
            "clusters": [],
            "pressure": "Neutral",
            "total_bid_volume": Decimal(0),
            "total_ask_volume": Decimal(0),
        }
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            self.logger.warning(
                "Orderbook data incomplete or unavailable for analysis."
            )
            return analysis

        try:
            bids = pd.DataFrame(
                orderbook["bids"], columns=["price", "size"], dtype=str
            )  # Load as string first
            asks = pd.DataFrame(orderbook["asks"], columns=["price", "size"], dtype=str)

            # Convert to Decimal, handle potential errors
            bids["price"] = bids["price"].apply(
                lambda x: Decimal(x) if x else Decimal(0)
            )
            bids["size"] = bids["size"].apply(lambda x: Decimal(x) if x else Decimal(0))
            asks["price"] = asks["price"].apply(
                lambda x: Decimal(x) if x else Decimal(0)
            )
            asks["size"] = asks["size"].apply(lambda x: Decimal(x) if x else Decimal(0))

            # Calculate total volume (size * price ~ quote volume estimate)
            bids["value_usd"] = bids["price"] * bids["size"]
            asks["value_usd"] = asks["price"] * asks["size"]

            total_bid_value = bids["value_usd"].sum()
            total_ask_value = asks["value_usd"].sum()
            analysis["total_bid_volume"] = (
                total_bid_value  # Store as value (closer to USD)
            )
            analysis["total_ask_volume"] = total_ask_value

            # Pressure Calculation (Simple ratio)
            if total_bid_value + total_ask_value > 0:
                bid_ask_ratio = total_bid_value / (total_bid_value + total_ask_value)
                if bid_ask_ratio > 0.6:
                    analysis["pressure"] = "High Buy Pressure"
                elif bid_ask_ratio < 0.4:
                    analysis["pressure"] = "High Sell Pressure"
                else:
                    analysis["pressure"] = "Neutral"

            # Cluster Analysis near S/R
            cluster_threshold = Decimal(
                str(self.orderbook_settings.get("cluster_threshold_usd", 10000))
            )
            proximity_pct = (
                Decimal(str(self.orderbook_settings.get("cluster_proximity_pct", 0.1)))
                / 100
            )

            all_levels = {**levels.get("support", {}), **levels.get("resistance", {})}
            if levels.get("pivot"):
                all_levels["Pivot"] = levels["pivot"]

            for name, level_price in all_levels.items():
                if not isinstance(level_price, Decimal):
                    continue  # Skip if level wasn't calculated correctly

                min_price = level_price * (Decimal(1) - proximity_pct)
                max_price = level_price * (Decimal(1) + proximity_pct)

                # Check bid clusters below/at level (potential support)
                bids_near_level = bids[
                    (bids["price"] >= min_price) & (bids["price"] <= max_price)
                ]
                bid_cluster_value = bids_near_level["value_usd"].sum()
                if bid_cluster_value >= cluster_threshold:
                    analysis["clusters"].append({
                        "type": "Support Cluster (Bids)",
                        "level_name": name,
                        "level_price": f"{level_price:.4f}",
                        "cluster_value_usd": f"{bid_cluster_value:,.0f}",
                        "price_range": f"{min_price:.4f}-{max_price:.4f}",
                    })

                # Check ask clusters above/at level (potential resistance)
                asks_near_level = asks[
                    (asks["price"] >= min_price) & (asks["price"] <= max_price)
                ]
                ask_cluster_value = asks_near_level["value_usd"].sum()
                if ask_cluster_value >= cluster_threshold:
                    analysis["clusters"].append({
                        "type": "Resistance Cluster (Asks)",
                        "level_name": name,
                        "level_price": f"{level_price:.4f}",
                        "cluster_value_usd": f"{ask_cluster_value:,.0f}",
                        "price_range": f"{min_price:.4f}-{max_price:.4f}",
                    })

        except (KeyError, ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Error analyzing orderbook: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error analyzing orderbook: {e}")

        return analysis

    def _interpret_analysis(
        self,
        df: pd.DataFrame,
        current_price: Decimal,
        levels: dict,
        orderbook_analysis: dict,
    ) -> dict:
        """Interprets calculated indicators and levels into signals."""
        interpretation = {"summary": [], "signals": {}}
        if df.empty or len(df) < 2:
            self.logger.warning("Insufficient data for interpretation.")
            return interpretation

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) >= 2 else last_row  # Handle short dataframes

        # --- Indicator Interpretation Helpers ---
        def get_val(row, key, default=np.nan):
            return row.get(key, default)

        def format_val(value, precision=2):
            if pd.isna(value):
                return "N/A"
            try:
                return f"{float(value):.{precision}f}"
            except (ValueError, TypeError):
                return str(value)  # Fallback for non-numeric

        # --- Trend & Momentum ---
        ema_short_col = f"EMA_{self.indicator_settings.get('ema_short_period', 12)}"
        ema_long_col = f"EMA_{self.indicator_settings.get('ema_long_period', 26)}"
        sma_short_col = f"SMA_{self.indicator_settings.get('sma_short_period', 10)}"
        sma_long_col = f"SMA_{self.indicator_settings.get('sma_long_period', 50)}"
        adx_col = f"ADX_{self.indicator_settings.get('adx_period', 14)}"
        dmp_col = f"DMP_{self.indicator_settings.get('adx_period', 14)}"  # ADX +DI
        dmn_col = f"DMN_{self.indicator_settings.get('adx_period', 14)}"  # ADX -DI
        psar_col_up = "PSARl_0.02_0.2"  # Default PSAR col name from pandas_ta
        psar_col_down = "PSARs_0.02_0.2"  # Need to check exact names pandas_ta creates

        # Correct PSAR column identification
        psar_col = None
        if psar_col_up in df.columns and last_row[psar_col_up] is not np.nan:
            psar_col = psar_col_up
        elif psar_col_down in df.columns and last_row[psar_col_down] is not np.nan:
            psar_col = psar_col_down
        else:
            # Find any PSAR column if defaults changed/failed
            for col in df.columns:
                if col.startswith("PSAR"):
                    psar_col = col
                    break

        if (
            self.analysis_flags.get("ema_alignment")
            and ema_short_col in df.columns
            and ema_long_col in df.columns
        ):
            ema_short = get_val(last_row, ema_short_col)
            ema_long = get_val(last_row, ema_long_col)
            if not pd.isna(ema_short) and not pd.isna(ema_long):
                if ema_short > ema_long:
                    interpretation["summary"].append(
                        f"{NEON_GREEN}EMA Bullish Alignment{RESET} ({format_val(ema_short)} > {format_val(ema_long)})"
                    )
                    interpretation["signals"]["ema_trend"] = "Bullish"
                elif ema_short < ema_long:
                    interpretation["summary"].append(
                        f"{NEON_RED}EMA Bearish Alignment{RESET} ({format_val(ema_short)} < {format_val(ema_long)})"
                    )
                    interpretation["signals"]["ema_trend"] = "Bearish"
                else:
                    interpretation["summary"].append(
                        f"{NEON_YELLOW}EMA Neutral{RESET} ({format_val(ema_short)} = {format_val(ema_long)})"
                    )
                    interpretation["signals"]["ema_trend"] = "Neutral"

        if self.analysis_flags.get("adx_trend_strength") and adx_col in df.columns:
            adx = get_val(last_row, adx_col)
            dmp = get_val(last_row, dmp_col, 0)  # Default to 0 if NaN for comparison
            dmn = get_val(last_row, dmn_col, 0)
            trend_threshold = self.thresholds.get("adx_trending", 25)
            if not pd.isna(adx):
                trend_status = f"ADX ({format_val(adx)}): "
                if adx >= trend_threshold:
                    if dmp > dmn:
                        trend_status += f"{NEON_GREEN}Strong Uptrend (+DI > -DI){RESET}"
                        interpretation["signals"]["adx_trend"] = "Strong Bullish"
                    else:
                        trend_status += f"{NEON_RED}Strong Downtrend (-DI > +DI){RESET}"
                        interpretation["signals"]["adx_trend"] = "Strong Bearish"
                else:
                    trend_status += f"{NEON_YELLOW}Weak/Ranging Trend{RESET}"
                    interpretation["signals"]["adx_trend"] = "Ranging"
                interpretation["summary"].append(trend_status)

        if self.analysis_flags.get("psar_flip") and psar_col:
            psar_val = get_val(last_row, psar_col)
            prev_psar_val = get_val(prev_row, psar_col)
            close_val = get_val(last_row, "close")
            prev_close_val = get_val(prev_row, "close")

            if not pd.isna(psar_val) and not pd.isna(close_val):
                psar_status = f"PSAR ({format_val(psar_val, precision=4)}): "
                is_bullish_psar = close_val > psar_val
                was_bullish_psar = (
                    not pd.isna(prev_psar_val)
                    and not pd.isna(prev_close_val)
                    and prev_close_val > prev_psar_val
                )

                if is_bullish_psar:
                    psar_status += f"{NEON_GREEN}Price above PSAR (Bullish){RESET}"
                    interpretation["signals"]["psar_trend"] = "Bullish"
                    if not was_bullish_psar:  # Check for flip
                        psar_status += f" {NEON_GREEN}(Just Flipped Bullish){RESET}"
                        interpretation["signals"]["psar_signal"] = "Flip Bullish"
                else:
                    psar_status += f"{NEON_RED}Price below PSAR (Bearish){RESET}"
                    interpretation["signals"]["psar_trend"] = "Bearish"
                    if was_bullish_psar:  # Check for flip
                        psar_status += f" {NEON_RED}(Just Flipped Bearish){RESET}"
                        interpretation["signals"]["psar_signal"] = "Flip Bearish"
                interpretation["summary"].append(psar_status)

        # --- Oscillators ---
        rsi_col = f"RSI_{self.indicator_settings.get('rsi_period', 14)}"
        mfi_col = f"MFI_{self.indicator_settings.get('mfi_period', 14)}"
        cci_col = f"CCI_{self.indicator_settings.get('cci_period', 20)}_0.015"  # pandas_ta default CCI name
        wr_col = f"WILLR_{self.indicator_settings.get('williams_r_period', 14)}"
        stochrsi_k_col = f"STOCHRSIk_{self.indicator_settings.get('stoch_rsi_period', 14)}_{self.indicator_settings.get('rsi_period', 14)}_{self.indicator_settings.get('stoch_k_period', 3)}_{self.indicator_settings.get('stoch_d_period', 3)}"
        stochrsi_d_col = f"STOCHRSId_{self.indicator_settings.get('stoch_rsi_period', 14)}_{self.indicator_settings.get('rsi_period', 14)}_{self.indicator_settings.get('stoch_k_period', 3)}_{self.indicator_settings.get('stoch_d_period', 3)}"

        if self.analysis_flags.get("rsi_threshold") and rsi_col in df.columns:
            rsi = get_val(last_row, rsi_col)
            ob = self.thresholds.get("rsi_overbought", 70)
            os = self.thresholds.get("rsi_oversold", 30)
            if not pd.isna(rsi):
                rsi_status = f"RSI ({format_val(rsi)}): "
                if rsi >= ob:
                    rsi_status += f"{NEON_RED}Overbought{RESET}"
                    interpretation["signals"]["rsi_level"] = "Overbought"
                elif rsi <= os:
                    rsi_status += f"{NEON_GREEN}Oversold{RESET}"
                    interpretation["signals"]["rsi_level"] = "Oversold"
                else:
                    rsi_status += f"{NEON_YELLOW}Neutral{RESET}"
                    interpretation["signals"]["rsi_level"] = "Neutral"
                interpretation["summary"].append(rsi_status)

        # ... (Add similar interpretations for MFI, CCI, Williams %R using their thresholds) ...
        if self.analysis_flags.get("mfi_threshold") and mfi_col in df.columns:
            mfi = get_val(last_row, mfi_col)
            ob = self.thresholds.get("mfi_overbought", 80)
            os = self.thresholds.get("mfi_oversold", 20)
            if not pd.isna(mfi):
                mfi_status = f"MFI ({format_val(mfi)}): "
                if mfi >= ob:
                    mfi_status += f"{NEON_RED}Overbought{RESET}"
                    interpretation["signals"]["mfi_level"] = "Overbought"
                elif mfi <= os:
                    mfi_status += f"{NEON_GREEN}Oversold{RESET}"
                    interpretation["signals"]["mfi_level"] = "Oversold"
                else:
                    mfi_status += f"{NEON_YELLOW}Neutral{RESET}"
                    interpretation["signals"]["mfi_level"] = "Neutral"
                interpretation["summary"].append(mfi_status)

        if self.analysis_flags.get("cci_threshold") and cci_col in df.columns:
            cci = get_val(last_row, cci_col)
            ob = self.thresholds.get("cci_overbought", 100)
            os = self.thresholds.get("cci_oversold", -100)
            if not pd.isna(cci):
                cci_status = f"CCI ({format_val(cci)}): "
                if cci >= ob:
                    cci_status += f"{NEON_RED}Overbought Zone{RESET}"
                    interpretation["signals"]["cci_level"] = "Overbought"
                elif cci <= os:
                    cci_status += f"{NEON_GREEN}Oversold Zone{RESET}"
                    interpretation["signals"]["cci_level"] = "Oversold"
                else:
                    cci_status += f"{NEON_YELLOW}Neutral Zone{RESET}"
                    interpretation["signals"]["cci_level"] = "Neutral"
                interpretation["summary"].append(cci_status)

        if self.analysis_flags.get("williams_r_threshold") and wr_col in df.columns:
            wr = get_val(last_row, wr_col)
            ob = self.thresholds.get("williams_r_overbought", -20)
            os = self.thresholds.get("williams_r_oversold", -80)
            if not pd.isna(wr):
                wr_status = f"Williams %R ({format_val(wr)}): "
                if wr >= ob:  # Note: WR is inverse, higher is overbought
                    wr_status += f"{NEON_RED}Overbought{RESET}"
                    interpretation["signals"]["wr_level"] = "Overbought"
                elif wr <= os:  # Lower is oversold
                    wr_status += f"{NEON_GREEN}Oversold{RESET}"
                    interpretation["signals"]["wr_level"] = "Oversold"
                else:
                    wr_status += f"{NEON_YELLOW}Neutral{RESET}"
                    interpretation["signals"]["wr_level"] = "Neutral"
                interpretation["summary"].append(wr_status)

        # StochRSI Cross
        if (
            self.analysis_flags.get("stoch_rsi_cross")
            and stochrsi_k_col in df.columns
            and stochrsi_d_col in df.columns
        ):
            k_now = get_val(last_row, stochrsi_k_col)
            d_now = get_val(last_row, stochrsi_d_col)
            k_prev = get_val(prev_row, stochrsi_k_col)
            d_prev = get_val(prev_row, stochrsi_d_col)

            if (
                not pd.isna(k_now)
                and not pd.isna(d_now)
                and not pd.isna(k_prev)
                and not pd.isna(d_prev)
            ):
                stoch_status = (
                    f"StochRSI (K:{format_val(k_now)}, D:{format_val(d_now)}): "
                )
                if k_now > d_now and k_prev <= d_prev:  # Bullish cross
                    stoch_status += f"{NEON_GREEN}Bullish Cross произошло{RESET}"
                    interpretation["signals"]["stochrsi_cross"] = "Bullish"
                elif k_now < d_now and k_prev >= d_prev:  # Bearish cross
                    stoch_status += f"{NEON_RED}Bearish Cross произошло{RESET}"
                    interpretation["signals"]["stochrsi_cross"] = "Bearish"
                else:  # No recent cross
                    stoch_status += f"{NEON_YELLOW}No recent cross{RESET}"
                    interpretation["signals"]["stochrsi_cross"] = "Neutral"
                interpretation["summary"].append(stoch_status)

        # --- MACD ---
        macd_line_col = f"MACD_{self.indicator_settings.get('macd_fast', 12)}_{self.indicator_settings.get('macd_slow', 26)}_{self.indicator_settings.get('macd_signal', 9)}"
        macd_signal_col = f"MACDs_{self.indicator_settings.get('macd_fast', 12)}_{self.indicator_settings.get('macd_slow', 26)}_{self.indicator_settings.get('macd_signal', 9)}"
        macd_hist_col = f"MACDh_{self.indicator_settings.get('macd_fast', 12)}_{self.indicator_settings.get('macd_slow', 26)}_{self.indicator_settings.get('macd_signal', 9)}"

        # MACD Cross
        if (
            self.analysis_flags.get("macd_cross")
            and macd_line_col in df.columns
            and macd_signal_col in df.columns
        ):
            line_now = get_val(last_row, macd_line_col)
            sig_now = get_val(last_row, macd_signal_col)
            line_prev = get_val(prev_row, macd_line_col)
            sig_prev = get_val(prev_row, macd_signal_col)

            if (
                not pd.isna(line_now)
                and not pd.isna(sig_now)
                and not pd.isna(line_prev)
                and not pd.isna(sig_prev)
            ):
                macd_status = (
                    f"MACD (L:{format_val(line_now)}, S:{format_val(sig_now)}): "
                )
                if line_now > sig_now and line_prev <= sig_prev:  # Bullish cross
                    macd_status += f"{NEON_GREEN}Bullish Cross (Line > Signal){RESET}"
                    interpretation["signals"]["macd_cross"] = "Bullish"
                elif line_now < sig_now and line_prev >= sig_prev:  # Bearish cross
                    macd_status += f"{NEON_RED}Bearish Cross (Line < Signal){RESET}"
                    interpretation["signals"]["macd_cross"] = "Bearish"
                else:
                    macd_status += f"{NEON_YELLOW}{'Above Signal' if line_now > sig_now else 'Below Signal'}{RESET}"
                    interpretation["signals"]["macd_cross"] = (
                        "Above Signal" if line_now > sig_now else "Below Signal"
                    )
                interpretation["summary"].append(macd_status)

        # MACD Divergence (Simple check on last 2 points histogram vs price)
        if self.analysis_flags.get("macd_divergence") and macd_hist_col in df.columns:
            hist_now = get_val(last_row, macd_hist_col)
            hist_prev = get_val(prev_row, macd_hist_col)
            price_now = get_val(last_row, "close")
            price_prev = get_val(prev_row, "close")

            if (
                not pd.isna(hist_now)
                and not pd.isna(hist_prev)
                and not pd.isna(price_now)
                and not pd.isna(price_prev)
            ):
                # Bullish Divergence: Lower low in price, higher low in histogram
                if price_now < price_prev and hist_now > hist_prev:
                    interpretation["summary"].append(
                        f"{NEON_GREEN}Possible Bullish MACD Divergence{RESET}"
                    )
                    interpretation["signals"]["macd_divergence"] = "Bullish"
                # Bearish Divergence: Higher high in price, lower high in histogram
                elif price_now > price_prev and hist_now < hist_prev:
                    interpretation["summary"].append(
                        f"{NEON_RED}Possible Bearish MACD Divergence{RESET}"
                    )
                    interpretation["signals"]["macd_divergence"] = "Bearish"
                else:
                    interpretation["signals"]["macd_divergence"] = "None"

        # --- Bollinger Bands ---
        bb_upper_col = f"BBU_{self.indicator_settings.get('bollinger_bands_period', 20)}_{self.indicator_settings.get('bollinger_bands_std_dev', 2)}.0"
        bb_lower_col = f"BBL_{self.indicator_settings.get('bollinger_bands_period', 20)}_{self.indicator_settings.get('bollinger_bands_std_dev', 2)}.0"
        bb_mid_col = f"BBM_{self.indicator_settings.get('bollinger_bands_period', 20)}_{self.indicator_settings.get('bollinger_bands_std_dev', 2)}.0"  # Middle band (SMA)

        if (
            self.analysis_flags.get("bollinger_bands_break")
            and bb_upper_col in df.columns
            and bb_lower_col in df.columns
        ):
            upper = get_val(last_row, bb_upper_col)
            lower = get_val(last_row, bb_lower_col)
            middle = get_val(last_row, bb_mid_col)
            close_val = get_val(last_row, "close")

            if (
                not pd.isna(upper)
                and not pd.isna(lower)
                and not pd.isna(middle)
                and not pd.isna(close_val)
            ):
                bb_status = f"BB (U:{format_val(upper)}, M:{format_val(middle)}, L:{format_val(lower)}): "
                if close_val > upper:
                    bb_status += f"{NEON_RED}Price Above Upper Band (Potential Reversal/Overbought){RESET}"
                    interpretation["signals"]["bbands_signal"] = "Breakout Upper"
                elif close_val < lower:
                    bb_status += f"{NEON_GREEN}Price Below Lower Band (Potential Reversal/Oversold){RESET}"
                    interpretation["signals"]["bbands_signal"] = "Breakdown Lower"
                else:
                    bb_status += f"{NEON_YELLOW}Price within Bands{RESET}"
                    interpretation["signals"]["bbands_signal"] = "Within Bands"
                interpretation["summary"].append(bb_status)

        # --- Volume Indicators ---
        obv_col = "OBV"
        adi_col = "ADOSC"  # Use AD Oscillator from pandas_ta
        vol_ma_col = f"VOL_MA_{self.indicator_settings.get('volume_ma_period', 20)}"

        if self.analysis_flags.get("volume_confirmation") and vol_ma_col in df.columns:
            volume = get_val(last_row, "volume")
            vol_ma = get_val(last_row, vol_ma_col)
            if not pd.isna(volume) and not pd.isna(vol_ma):
                vol_status = (
                    f"Volume ({format_val(volume, 0)} vs MA {format_val(vol_ma, 0)}): "
                )
                if volume > vol_ma * Decimal("1.2"):  # Example: 20% above MA
                    vol_status += f"{NEON_PURPLE}High Volume{RESET}"
                    interpretation["signals"]["volume_level"] = "High"
                elif volume < vol_ma * Decimal("0.8"):
                    vol_status += f"{NEON_PURPLE}Low Volume{RESET}"
                    interpretation["signals"]["volume_level"] = "Low"
                else:
                    vol_status += f"{NEON_PURPLE}Average Volume{RESET}"
                    interpretation["signals"]["volume_level"] = "Average"
                interpretation["summary"].append(vol_status)

        if self.analysis_flags.get("obv_trend") and obv_col in df.columns:
            obv_now = get_val(last_row, obv_col)
            obv_prev = get_val(prev_row, obv_col)
            if not pd.isna(obv_now) and not pd.isna(obv_prev):
                obv_status = "OBV Trend: "
                if obv_now > obv_prev:
                    obv_status += (
                        f"{NEON_GREEN}Increasing (Bullish Confirmation){RESET}"
                    )
                    interpretation["signals"]["obv_trend"] = "Increasing"
                elif obv_now < obv_prev:
                    obv_status += f"{NEON_RED}Decreasing (Bearish Confirmation){RESET}"
                    interpretation["signals"]["obv_trend"] = "Decreasing"
                else:
                    obv_status += f"{NEON_YELLOW}Flat{RESET}"
                    interpretation["signals"]["obv_trend"] = "Flat"
                interpretation["summary"].append(obv_status)

        if self.analysis_flags.get("adi_trend") and adi_col in df.columns:
            adi_now = get_val(last_row, adi_col)
            adi_prev = get_val(prev_row, adi_col)
            if not pd.isna(adi_now) and not pd.isna(adi_prev):
                adi_status = "A/D Osc Trend: "
                if adi_now > adi_prev:
                    adi_status += f"{NEON_GREEN}Increasing (Accumulation){RESET}"
                    interpretation["signals"]["adi_trend"] = "Increasing"
                elif adi_now < adi_prev:
                    adi_status += f"{NEON_RED}Decreasing (Distribution){RESET}"
                    interpretation["signals"]["adi_trend"] = "Decreasing"
                else:
                    adi_status += f"{NEON_YELLOW}Flat{RESET}"
                    interpretation["signals"]["adi_trend"] = "Flat"
                interpretation["summary"].append(adi_status)

        # --- Levels & Orderbook ---
        interpretation["summary"].append(
            f"{NEON_BLUE}--- Levels & Orderbook ---{RESET}"
        )
        if levels:
            nearest_supports = sorted(
                levels.get("support", {}).items(),
                key=lambda item: abs(item[1] - current_price),
            )[:3]
            nearest_resistances = sorted(
                levels.get("resistance", {}).items(),
                key=lambda item: abs(item[1] - current_price),
            )[:3]
            if levels.get("pivot"):
                interpretation["summary"].append(f"Pivot Point: ${levels['pivot']:.4f}")
            for name, price in nearest_supports:
                interpretation["summary"].append(f"Support ({name}): ${price:.4f}")
            for name, price in nearest_resistances:
                interpretation["summary"].append(f"Resistance ({name}): ${price:.4f}")

        if orderbook_analysis:
            interpretation["summary"].append(
                f"Orderbook Pressure: {orderbook_analysis.get('pressure', 'N/A')}"
            )
            interpretation["summary"].append(
                f"Total Bid Value (Limit {self.orderbook_settings.get('limit', 50)}): ${orderbook_analysis.get('total_bid_volume', Decimal(0)):,.0f}"
            )
            interpretation["summary"].append(
                f"Total Ask Value (Limit {self.orderbook_settings.get('limit', 50)}): ${orderbook_analysis.get('total_ask_volume', Decimal(0)):,.0f}"
            )
            clusters = orderbook_analysis.get("clusters", [])
            if clusters:
                interpretation["summary"].append(
                    f"{NEON_PURPLE}Significant Orderbook Clusters:{RESET}"
                )
                for cluster in clusters[:5]:  # Show top 5 clusters
                    color = NEON_GREEN if "Support" in cluster["type"] else NEON_RED
                    interpretation["summary"].append(
                        f"  {color}{cluster['type']} near {cluster['level_name']} ({cluster['level_price']}) - Value: ${cluster['cluster_value_usd']}{RESET}"
                    )
            else:
                interpretation["summary"].append(
                    f"{NEON_YELLOW}No significant clusters found near S/R levels.{RESET}"
                )

        return interpretation

    def analyze(
        self, df_klines: pd.DataFrame, current_price: Decimal, orderbook: dict | None
    ) -> dict:
        """Main analysis function."""
        analysis_result = {
            "symbol": self.symbol,
            "timestamp": datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "current_price": f"{current_price:.4f}",
            "kline_interval": df_klines["timestamp"].diff().min()
            if not df_klines.empty
            else "N/A",  # Infer interval
            "levels": {},
            "orderbook_analysis": {},
            "interpretation": {
                "summary": ["Analysis could not be performed."],
                "signals": {},
            },
            "raw_indicators": {},  # Store last row of indicators
        }

        if df_klines.empty:
            self.logger.error("Kline data is empty, cannot perform analysis.")
            return analysis_result

        # 1. Calculate Indicators
        df_with_indicators = self._calculate_indicators(df_klines)
        if df_with_indicators.empty:
            return analysis_result  # Calculation failed

        # Store last row of indicators
        if not df_with_indicators.empty:
            indicator_values = df_with_indicators.iloc[-1].to_dict()
            # Convert numpy types to standard python types for JSON serialization/logging
            for k, v in indicator_values.items():
                if isinstance(v, (np.integer, np.int64)):
                    indicator_values[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    indicator_values[k] = float(v) if not np.isnan(v) else None
                elif isinstance(v, pd.Timestamp):
                    indicator_values[k] = v.isoformat()
                elif pd.isna(v):
                    indicator_values[k] = None
            analysis_result["raw_indicators"] = indicator_values

        # 2. Calculate Support/Resistance Levels
        levels = self._calculate_levels(df_with_indicators, current_price)
        analysis_result["levels"] = {
            k: {name: f"{price:.4f}" for name, price in v.items()}
            if isinstance(v, dict)
            else (f"{v:.4f}" if isinstance(v, Decimal) else v)
            for k, v in levels.items()
        }  # Format for output

        # 3. Analyze Orderbook
        orderbook_analysis = self._analyze_orderbook(orderbook, current_price, levels)
        # Format Decimal for output
        if "total_bid_volume" in orderbook_analysis:
            orderbook_analysis["total_bid_volume"] = (
                f"{orderbook_analysis['total_bid_volume']:,.0f}"
            )
        if "total_ask_volume" in orderbook_analysis:
            orderbook_analysis["total_ask_volume"] = (
                f"{orderbook_analysis['total_ask_volume']:,.0f}"
            )
        analysis_result["orderbook_analysis"] = orderbook_analysis

        # 4. Interpret Results
        interpretation = self._interpret_analysis(
            df_with_indicators, current_price, levels, orderbook_analysis
        )
        analysis_result["interpretation"] = interpretation

        return analysis_result

    def format_analysis_output(self, analysis_result: dict) -> str:
        """Formats the analysis result dictionary into a readable string."""
        output = f"\n--- Analysis Report for {analysis_result['symbol']} --- ({analysis_result['timestamp']})\n"
        output += (
            f"{NEON_BLUE}Current Price:{RESET} ${analysis_result['current_price']}\n"
        )

        interpretation = analysis_result.get("interpretation", {})
        summary = interpretation.get("summary", [])

        if summary:
            output += "\n" + "\n".join(summary) + "\n"
        else:
            output += f"{NEON_YELLOW}No interpretation summary available.{RESET}\n"

        # Optionally add raw indicator values if needed
        # raw = analysis_result.get("raw_indicators", {})
        # if raw:
        #     output += f"\n{NEON_BLUE}--- Latest Indicator Values ---{RESET}\n"
        #     # Format and print selected raw values for debugging/info
        #     rsi_col = f"RSI_{self.indicator_settings.get('rsi_period', 14)}"
        #     if rsi_col in raw and raw[rsi_col] is not None:
        #          output += f"RSI: {raw[rsi_col]:.2f}\n"
        #     macd_line_col = f"MACD_{self.indicator_settings.get('macd_fast', 12)}_{self.indicator_settings.get('macd_slow', 26)}_{self.indicator_settings.get('macd_signal', 9)}"
        #     if macd_line_col in raw and raw[macd_line_col] is not None:
        #         output += f"MACD Line: {raw[macd_line_col]:.4f}\n"
        #     # Add others as needed

        return output


# --- Main Application Logic ---


async def run_analysis_loop(
    symbol: str,
    interval: str,
    client: BybitCCXTClient,
    analyzer: TradingAnalyzer,
    logger_instance: logging.Logger,
):
    """The main loop for fetching data and running analysis."""
    analysis_interval_sec = CONFIG.get("analysis_interval_seconds", 30)
    kline_limit = CONFIG.get("kline_limit", 200)
    orderbook_limit = analyzer.orderbook_settings.get("limit", 50)

    while True:
        start_time = time.monotonic()
        logger_instance.info(f"{NEON_BLUE}--- Starting Analysis Cycle ---{RESET}")

        try:
            # Fetch data concurrently
            tasks = {
                "price": asyncio.create_task(client.fetch_current_price(symbol)),
                "klines": asyncio.create_task(
                    client.fetch_klines(symbol, interval, kline_limit)
                ),
                "orderbook": asyncio.create_task(
                    client.fetch_orderbook(symbol, orderbook_limit)
                ),
            }

            # Wait for price and klines first, as they are essential
            await asyncio.wait(
                [tasks["price"], tasks["klines"]], return_when=asyncio.ALL_COMPLETED
            )

            current_price = tasks["price"].result()
            df_klines = tasks["klines"].result()

            if current_price is None:
                logger_instance.error(
                    "Failed to fetch current price. Skipping analysis cycle."
                )
                await async_sleep(RETRY_DELAY_SECONDS)
                continue
            if df_klines.empty:
                logger_instance.error(
                    "Failed to fetch kline data. Skipping analysis cycle."
                )
                await async_sleep(RETRY_DELAY_SECONDS)
                continue

            # Wait for orderbook (less critical, analysis can proceed without it)
            await asyncio.wait(
                [tasks["orderbook"]], return_when=asyncio.FIRST_COMPLETED, timeout=10
            )  # Shorter timeout for less critical data
            orderbook = (
                tasks["orderbook"].result() if tasks["orderbook"].done() else None
            )
            if orderbook is None and not tasks["orderbook"].done():
                logger_instance.warning(
                    "Fetching orderbook timed out or failed, proceeding without it."
                )
                tasks["orderbook"].cancel()  # Cancel if timed out

            # Perform analysis
            analysis_result = analyzer.analyze(df_klines, current_price, orderbook)

            # Log and print results
            output_string = analyzer.format_analysis_output(analysis_result)
            logger_instance.info(f"Analysis Complete:\n{output_string}")
            # Optionally print to console without logger prefix
            # print(output_string)

            # Log structured JSON result (optional)
            try:
                # Convert Decimals in signals/levels for JSON logging if not done earlier
                # logger_instance.debug(json.dumps(analysis_result, default=str)) # default=str handles Decimals
                pass
            except Exception as json_err:
                logger_instance.error(
                    f"Error converting analysis result to JSON for logging: {json_err}"
                )

        except ccxt.AuthenticationError as e:
            logger_instance.error(
                f"{NEON_RED}Authentication Error: Check API Key/Secret. Stopping. - {e}{RESET}"
            )
            break  # Stop on auth errors
        except ccxt.InvalidNonce as e:
            logger_instance.error(
                f"{NEON_RED}Invalid Nonce Error: Check system time sync. Stopping. - {e}{RESET}"
            )
            break
        except ccxt.InsufficientFunds as e:
            logger_instance.error(
                f"{NEON_RED}Insufficient Funds: {e}{RESET}"
            )  # Informational if placing orders
            # Continue analysis loop
        except ccxt.InvalidOrder as e:
            logger_instance.error(
                f"{NEON_RED}Invalid Order parameters: {e}{RESET}"
            )  # Informational if placing orders
            # Continue analysis loop
        except Exception as e:
            logger_instance.exception(
                f"{NEON_RED}An unexpected error occurred in the main loop: {e}{RESET}"
            )

        # Calculate sleep time
        elapsed_time = time.monotonic() - start_time
        sleep_duration = max(0, analysis_interval_sec - elapsed_time)
        logger_instance.info(
            f"Cycle took {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s."
        )
        await async_sleep(sleep_duration)


async def main():
    global logger  # Allow modification of the global logger variable

    symbol = ""
    while True:
        symbol_input = (
            input(f"{NEON_BLUE}Enter trading symbol (e.g., BTC/USDT): {RESET}")
            .upper()
            .strip()
        )
        # Basic validation: check for '/' and non-empty parts
        if "/" in symbol_input and all(
            part.strip() for part in symbol_input.split("/")
        ):
            symbol = symbol_input
            break
        else:
            print(
                f"{NEON_RED}Invalid symbol format. Use format like 'BTC/USDT'.{RESET}"
            )

    interval = ""
    while True:
        interval_input = input(
            f"{NEON_BLUE}Enter timeframe [{', '.join(VALID_INTERVALS)}]: {RESET}"
        ).strip()
        if not interval_input:
            # Find default interval from config, potentially mapping back from ccxt if needed
            default_interval_minutes = CONFIG.get("indicator_settings", {}).get(
                "default_interval", "15"
            )  # Example default
            interval = default_interval_minutes
            print(
                f"{NEON_YELLOW}No interval provided. Using default: {interval}{RESET}"
            )
            break
        if interval_input in VALID_INTERVALS:
            interval = interval_input
            break
        # Allow ccxt intervals like '1h' as input too
        if interval_input in REVERSE_CCXT_INTERVAL_MAP:
            interval = REVERSE_CCXT_INTERVAL_MAP[interval_input]
            print(
                f"{NEON_YELLOW}Using interval {interval} (mapped from {interval_input}){RESET}"
            )
            break

        print(
            f"{NEON_RED}Invalid interval. Choose from: {', '.join(VALID_INTERVALS)}{RESET}"
        )

    # Setup logger after getting symbol
    logger = setup_logger(symbol)

    # Initialize API Client and Analyzer
    client = BybitCCXTClient(API_KEY, API_SECRET, CCXT_BASE_URL, logger)
    analyzer = TradingAnalyzer(CONFIG, logger, symbol)

    logger.info(
        f"Starting analysis for {symbol} on {interval} ({CCXT_INTERVAL_MAP.get(interval)}) interval."
    )
    logger.info(
        f"Using API Environment: {API_ENV.upper()} ({client.exchange.urls['api']})"
    )
    logger.info(
        f"Analysis loop interval: {CONFIG.get('analysis_interval_seconds', 30)} seconds."
    )

    try:
        await run_analysis_loop(symbol, interval, client, analyzer, logger)
    except KeyboardInterrupt:
        logger.info(f"{NEON_YELLOW}Analysis stopped by user.{RESET}")
    except Exception as e:
        logger.exception(f"{NEON_RED}Critical error during execution: {e}{RESET}")
    finally:
        await client.close()
        logger.info("Application finished.")
        logging.shutdown()  # Ensure logs are flushed


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{NEON_YELLOW}Process interrupted by user. Exiting.{RESET}")
    except Exception as e:
        # Catch potential errors during asyncio setup/shutdown itself
        print(f"\n{NEON_RED}A critical error occurred: {e}{RESET}")
