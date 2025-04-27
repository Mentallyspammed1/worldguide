# sxs.py
# Enhanced and Upgraded Scalping Bot Framework
# Derived from xrscalper.py, focusing on robust execution, error handling,
# advanced position management (BE, TSL), and Bybit V5 compatibility.

# Standard Library Imports
import json
import logging
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo

# Third-Party Imports
import ccxt
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialization ---
init(autoreset=True)
load_dotenv()

# Set Decimal precision for financial calculations
getcontext().prec = 36

# --- Neon Color Scheme ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')
    logging.critical(f"{NEON_RED}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}")
    raise ValueError("API keys not set.")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

DEFAULT_TIMEZONE = ZoneInfo("America/Chicago")
TIMEZONE = DEFAULT_TIMEZONE

MAX_API_RETRIES = 5
RETRY_DELAY_SECONDS = 7
RETRYABLE_HTTP_CODES = [429, 500, 502, 503, 504]

VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_PERIOD = 20
DEFAULT_WILLIAMS_R_PERIOD = 14
DEFAULT_MFI_PERIOD = 14
DEFAULT_STOCH_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_K_PERIOD = 3
DEFAULT_STOCH_RSI_D_PERIOD = 3
DEFAULT_RSI_PERIOD = 14
DEFAULT_BBANDS_PERIOD = 20
DEFAULT_BBANDS_STDDEV = 2.0
DEFAULT_SMA10_PERIOD = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_PERIOD = 50
DEFAULT_PSAR_STEP = 0.02
DEFAULT_PSAR_MAX_STEP = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
LOOP_DELAY_SECONDS = 10
POSITION_CONFIRM_DELAY_SECONDS = 10

config: dict[str, Any] = {}
default_config: dict[str, Any] = {}


# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Custom formatter to redact sensitive information from logs."""
    _patterns = {}
    _sensitive_keys = {"API_KEY": API_KEY, "API_SECRET": API_SECRET}

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for key_name, key_value in self._sensitive_keys.items():
            if key_value:
                if key_value not in self._patterns:
                    self._patterns[key_value] = f"***{key_name}***"
                if key_value in msg:
                    msg = msg.replace(key_value, self._patterns[key_value])
        return msg


def load_config(filepath: str) -> dict[str, Any]:
    """Loads and validates configuration from a JSON file."""
    global TIMEZONE, default_config
    default_config = {
        "symbol": "BTC/USDT:USDT",
        "interval": "5",
        "retry_delay": RETRY_DELAY_SECONDS,
        "max_api_retries": MAX_API_RETRIES,
        "enable_trading": False,
        "use_sandbox": True,
        "max_concurrent_positions": 1,
        "quote_currency": "USDT",
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "loop_delay_seconds": LOOP_DELAY_SECONDS,
        "timezone": "America/Chicago",
        "risk_per_trade": 0.01,
        "leverage": 20,
        "stop_loss_multiple": 1.8,
        "take_profit_multiple": 0.7,
        "entry_order_type": "market",
        "limit_order_offset_buy": 0.0005,
        "limit_order_offset_sell": 0.0005,
        "enable_trailing_stop": True,
        "trailing_stop_callback_rate": 0.005,
        "trailing_stop_activation_percentage": 0.003,
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,
        "break_even_offset_ticks": 2,
        "time_based_exit_minutes": None,
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_PERIOD,
        "bollinger_bands_period": DEFAULT_BBANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BBANDS_STDDEV,
        "cci_period": DEFAULT_CCI_PERIOD,
        "williams_r_period": DEFAULT_WILLIAMS_R_PERIOD,
        "mfi_period": DEFAULT_MFI_PERIOD,
        "stoch_rsi_period": DEFAULT_STOCH_RSI_PERIOD,
        "stoch_rsi_rsi_period": DEFAULT_STOCH_RSI_RSI_PERIOD,
        "stoch_rsi_k_period": DEFAULT_STOCH_RSI_K_PERIOD,
        "stoch_rsi_d_period": DEFAULT_STOCH_RSI_D_PERIOD,
        "psar_step": DEFAULT_PSAR_STEP,
        "psar_max_step": DEFAULT_PSAR_MAX_STEP,
        "sma_10_period": DEFAULT_SMA10_PERIOD,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_period": DEFAULT_FIB_PERIOD,
        "orderbook_limit": 25,
        "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "volume_confirmation_multiplier": 1.5,
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        "weight_sets": {
            "scalping": {
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": {
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default"
    }

    current_config = default_config.copy()
    needs_saving = False

    if os.path.exists(filepath):
        try:
            with open(filepath, encoding='utf-8') as f:
                loaded_config = json.load(f)
            current_config = _merge_configs(loaded_config, default_config)
            if current_config != loaded_config:
                needs_saving = True
        except Exception:
            needs_saving = True
    else:
        needs_saving = True

    def validate_param(key: str, validation_func: callable, error_msg: str) -> bool:
        is_valid = False
        value = current_config.get(key)
        default = default_config.get(key)
        try:
            if key in current_config and validation_func(value):
                is_valid = True
            else:
                current_config[key] = default
                nonlocal needs_saving
                needs_saving = True
        except Exception:
            current_config[key] = default
            needs_saving = True
        return is_valid

    validate_param("symbol", lambda v: isinstance(v, str) and v.strip(), "Invalid symbol '{value}' for '{key}'. Reset to '{default}'.")
    validate_param("interval", lambda v: v in VALID_INTERVALS, "Invalid interval '{value}' for '{key}'. Reset to '{default}'.")
    validate_param("entry_order_type", lambda v: v in ["market", "limit"], "Invalid order type '{value}' for '{key}'. Reset to '{default}'.")
    validate_param("quote_currency", lambda v: isinstance(v, str) and len(v) >= 3 and v.isupper(), "Invalid quote '{value}' for '{key}'. Reset to '{default}'.")

    try:
        TIMEZONE = ZoneInfo(current_config["timezone"])
    except Exception:
        current_config["timezone"] = default_config["timezone"]
        TIMEZONE = ZoneInfo(default_config["timezone"])
        needs_saving = True

    numeric_params = {
        "risk_per_trade": (0, 1, False, False, False),
        "leverage": (1, 1000, True, True, True),
        "stop_loss_multiple": (0, float('inf'), False, True, False),
        "take_profit_multiple": (0, float('inf'), False, True, False),
        "trailing_stop_callback_rate": (0, 1, False, False, False),
        "trailing_stop_activation_percentage": (0, 1, True, False, False),
        "break_even_trigger_atr_multiple": (0, float('inf'), False, True, False),
        "break_even_offset_ticks": (0, 1000, True, True, True),
        "signal_score_threshold": (0, float('inf'), False, True, False),
        "atr_period": (2, 1000, True, True, True),
        "ema_short_period": (1, 1000, True, True, True),
        "ema_long_period": (1, 1000, True, True, True),
        "rsi_period": (2, 1000, True, True, True),
        "bollinger_bands_period": (2, 1000, True, True, True),
        "bollinger_bands_std_dev": (0, 10, False, True, False),
        "cci_period": (2, 1000, True, True, True),
        "williams_r_period": (2, 1000, True, True, True),
        "mfi_period": (2, 1000, True, True, True),
        "stoch_rsi_period": (2, 1000, True, True, True),
        "stoch_rsi_rsi_period": (2, 1000, True, True, True),
        "stoch_rsi_k_period": (1, 1000, True, True, True),
        "stoch_rsi_d_period": (1, 1000, True, True, True),
        "psar_step": (0, 1, False, True, False),
        "psar_max_step": (0, 1, False, True, False),
        "sma_10_period": (1, 1000, True, True, True),
        "momentum_period": (1, 1000, True, True, True),
        "volume_ma_period": (1, 1000, True, True, True),
        "fibonacci_period": (2, 1000, True, True, True),
        "orderbook_limit": (1, 200, True, True, True),
        "position_confirm_delay_seconds": (0, 120, True, True, False),
        "loop_delay_seconds": (1, 300, True, True, False),
        "stoch_rsi_oversold_threshold": (0, 100, True, False, False),
        "stoch_rsi_overbought_threshold": (0, 100, False, True, False),
        "volume_confirmation_multiplier": (0, float('inf'), False, True, False),
        "limit_order_offset_buy": (0, 0.1, True, False, False),
        "limit_order_offset_sell": (0, 0.1, True, False, False),
        "retry_delay": (1, 120, True, True, False),
        "max_api_retries": (0, 10, True, True, True),
        "max_concurrent_positions": (1, 10, True, True, True),
    }
    for key, (min_val, max_val, allow_min, allow_max, is_int) in numeric_params.items():
        value = current_config.get(key)
        default = default_config.get(key)
        try:
            val_dec = Decimal(str(value))
            if not val_dec.is_finite(): raise ValueError
            lower_ok = (val_dec >= min_val) if allow_min else (val_dec > min_val)
            upper_ok = (val_dec <= max_val) if allow_max else (val_dec < max_val)
            if lower_ok and upper_ok:
                current_config[key] = int(val_dec) if is_int else float(val_dec)
            else:
                raise ValueError
        except Exception:
            current_config[key] = default
            needs_saving = True

    if needs_saving:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(current_config, f, indent=4, sort_keys=True)
        except Exception:
            pass

    return current_config


def _merge_configs(loaded_config: dict, default_config: dict) -> dict:
    """Recursively merges loaded config with defaults."""
    merged = default_config.copy()
    for key, value in loaded_config.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = _merge_configs(value, merged[key])
        else:
            merged[key] = value
    return merged


# --- Logging Setup ---
def setup_logger(name: str, config: dict[str, Any] = None, level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    config = config or {}

    # File handler (UTC)
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_formatter = SensitiveFormatter("%(asctime)s.%(msecs)03dZ %(levelname)-8s [%(name)s:%(lineno)d] %(message)s", datefmt='%Y-%m-%dT%H:%M:%S')
    file_formatter.converter = time.gmtime
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Console handler (local time)
    console_tz = ZoneInfo(config.get("timezone", "America/Chicago")) if config else TIMEZONE
    stream_handler = logging.StreamHandler()
    console_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} {NEON_YELLOW}%(levelname)-8s{RESET} {NEON_PURPLE}[%(name)s]{RESET} %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
    console_formatter.converter = lambda *args: datetime.now(console_tz).timetuple()
    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


# --- CCXT Exchange Setup ---
def initialize_exchange(config: dict[str, Any], logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object."""
    lg = logger
    try:
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'rateLimit': 150,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 25000,
                'fetchOHLCVTimeout': 20000,
                'fetchOrderBookTimeout': 15000,
                'setLeverageTimeout': 20000,
                'fetchMyTradesTimeout': 20000,
                'fetchClosedOrdersTimeout': 25000,
            }
        })

        if config.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}SANDBOX MODE{RESET}")
            exchange.set_sandbox_mode(True)
            if 'testnet' not in str(exchange.urls.get('api', '')).lower():
                exchange.urls['api'] = 'https://api-testnet.bybit.com'
                lg.info(f"Manually set to Testnet: {exchange.urls['api']}")
        else:
            lg.info(f"{NEON_GREEN}LIVE MODE{RESET}")
            if 'testnet' in str(exchange.urls.get('api', '')).lower():
                exchange.urls['api'] = 'https://api.bybit.com'
                lg.info(f"Reset to Production: {exchange.urls['api']}")

        lg.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets(reload=True)
        if config["symbol"] not in exchange.markets:
            lg.critical(f"{NEON_RED}Symbol '{config['symbol']}' not found.{RESET}")
            return None

        balance = fetch_balance(exchange, config["quote_currency"], lg)
        if balance is None:
            lg.critical(f"{NEON_RED}Balance fetch failed.{RESET}")
            return None

        lg.info(f"{NEON_GREEN}Exchange initialized. Balance: {balance:.4f} {config['quote_currency']}{RESET}")
        return exchange
    except Exception as e:
        lg.critical(f"{NEON_RED}Exchange init failed: {e}{RESET}", exc_info=True)
        return None


# --- Helper Functions ---
def safe_api_call(func: callable, logger: logging.Logger, *args, **kwargs) -> Any:
    """Wraps API calls with retry logic."""
    lg = logger
    max_retries = config.get("max_api_retries", MAX_API_RETRIES)
    base_delay = config.get("retry_delay", RETRY_DELAY_SECONDS)

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
            if attempt == max_retries:
                lg.error(f"{NEON_RED}API call failed after {max_retries} retries: {e}{RESET}")
                return None
            delay = base_delay * (2 ** attempt)
            lg.warning(f"Retryable error (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            lg.error(f"{NEON_RED}API call error: {e}{RESET}", exc_info=True)
            return None


def fetch_balance(exchange: ccxt.Exchange, quote_currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches available balance with retries."""
    lg = logger
    try:
        balance_data = safe_api_call(exchange.fetch_balance, lg)
        if balance_data:
            free_balance = Decimal(str(balance_data.get('free', {}).get(quote_currency, 0)))
            return free_balance if free_balance.is_finite() else None
    except Exception as e:
        lg.error(f"Balance fetch failed: {e}", exc_info=True)
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV data."""
    lg = logger
    try:
        klines = safe_api_call(exchange.fetch_ohlcv, lg, symbol, timeframe, limit=limit)
        if klines:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(lambda x: x.astype(str).map(Decimal))
            return df
    except Exception as e:
        lg.error(f"Klines fetch failed: {e}", exc_info=True)
    return pd.DataFrame()


def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetches current ticker price."""
    lg = logger
    try:
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol)
        if ticker and 'last' in ticker:
            return Decimal(str(ticker['last']))
    except Exception as e:
        lg.error(f"Price fetch failed: {e}", exc_info=True)
    return None


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> dict | None:
    """Fetches order book data."""
    lg = logger
    try:
        orderbook = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit)
        if orderbook:
            return {'bids': [[Decimal(str(p)), Decimal(str(v))] for p, v in orderbook['bids']],
                    'asks': [[Decimal(str(p)), Decimal(str(v))] for p, v in orderbook['asks']]}
    except Exception as e:
        lg.error(f"Orderbook fetch failed: {e}", exc_info=True)
    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Retrieves market info with enhanced validation."""
    lg = logger
    try:
        if not exchange.markets:
            exchange.load_markets(reload=True)
        market = exchange.market(symbol)
        is_contract = market.get('contract', False)
        is_linear = market.get('linear', True)
        return {
            'id': market['id'],
            'precision': market['precision'],
            'limits': market['limits'],
            'contractSize': market.get('contractSize', 1),
            'is_contract': is_contract,
            'is_linear': is_linear,
            'base': market.get('base', ''),
        }
    except Exception as e:
        lg.error(f"Market info fetch failed for {symbol}: {e}", exc_info=True)
        return None


# --- Trading Analysis ---
class TradingAnalyzer:
    """Analyzes market data and generates trading signals."""
    def __init__(self, klines_df: pd.DataFrame, logger: logging.Logger, config: dict[str, Any], market_info: dict) -> None:
        self.df = klines_df
        self.lg = logger
        self.config = config
        self.market_info = market_info
        self.indicator_values: dict[str, Decimal] = {}
        self.calculate_indicators()

    def get_min_tick_size(self) -> Decimal:
        """Returns minimum price tick size."""
        tick = Decimal(str(self.market_info['precision'].get('price', '0.0001')))
        return tick if tick.is_finite() and tick > 0 else Decimal('0.0001')

    def get_min_amount_step(self) -> Decimal:
        """Returns minimum amount step size."""
        step = Decimal(str(self.market_info['precision'].get('amount', '0.001')))
        return step if step.is_finite() and step > 0 else Decimal('0.001')

    def get_price_precision(self) -> int:
        """Returns price precision as number of decimal places."""
        tick = self.get_min_tick_size()
        return abs(tick.as_tuple().exponent)

    def get_amount_precision_places(self) -> int:
        """Returns amount precision as number of decimal places."""
        step = self.get_min_amount_step()
        return abs(step.as_tuple().exponent)

    def calculate_indicators(self) -> None:
        """Calculates enabled technical indicators."""
        indicators = self.config.get("indicators", {})

        if indicators.get("ema_alignment", False):
            self._calculate_ema()
        if indicators.get("momentum", False):
            self._calculate_momentum()
        if indicators.get("volume_confirmation", False):
            self._calculate_volume_ma()
        if indicators.get("stoch_rsi", False):
            self._calculate_stoch_rsi()
        if indicators.get("rsi", False):
            self._calculate_rsi()
        if indicators.get("bollinger_bands", False):
            self._calculate_bbands()
        if indicators.get("vwap", False):
            self._calculate_vwap()
        if indicators.get("cci", False):
            self._calculate_cci()
        if indicators.get("wr", False):
            self._calculate_williams_r()
        if indicators.get("psar", False):
            self._calculate_psar()
        if indicators.get("sma_10", False):
            self._calculate_sma()
        if indicators.get("mfi", False):
            self._calculate_mfi()
        if indicators.get("atr", True):  # ATR always calculated for risk management
            self._calculate_atr()

    def _calculate_ema(self) -> None:
        df = self.df
        short_ema = ta.ema(df['close'], length=self.config["ema_short_period"])
        long_ema = ta.ema(df['close'], length=self.config["ema_long_period"])
        if short_ema is not None and long_ema is not None:
            self.indicator_values["EMA_short"] = Decimal(str(short_ema.iloc[-1]))
            self.indicator_values["EMA_long"] = Decimal(str(long_ema.iloc[-1]))

    def _calculate_momentum(self) -> None:
        df = self.df
        mom = ta.momentum(df['close'], length=self.config["momentum_period"])
        if mom is not None:
            self.indicator_values["Momentum"] = Decimal(str(mom.iloc[-1]))

    def _calculate_volume_ma(self) -> None:
        df = self.df
        vol_ma = ta.sma(df['volume'], length=self.config["volume_ma_period"])
        if vol_ma is not None:
            self.indicator_values["Volume_MA"] = Decimal(str(vol_ma.iloc[-1]))
            self.indicator_values["Volume"] = df['volume'].iloc[-1]

    def _calculate_stoch_rsi(self) -> None:
        df = self.df
        stochrsi = ta.stochrsi(df['close'], length=self.config["stoch_rsi_period"],
                               rsi_length=self.config["stoch_rsi_rsi_period"],
                               k=self.config["stoch_rsi_k_period"],
                               d=self.config["stoch_rsi_d_period"])
        if stochrsi is not None and 'STOCHRSI' in stochrsi:
            self.indicator_values["StochRSI"] = Decimal(str(stochrsi['STOCHRSI'].iloc[-1]))

    def _calculate_rsi(self) -> None:
        df = self.df
        rsi = ta.rsi(df['close'], length=self.config["rsi_period"])
        if rsi is not None:
            self.indicator_values["RSI"] = Decimal(str(rsi.iloc[-1]))

    def _calculate_bbands(self) -> None:
        df = self.df
        bbands = ta.bbands(df['close'], length=self.config["bollinger_bands_period"],
                           std=self.config["bollinger_bands_std_dev"])
        if bbands is not None:
            self.indicator_values["BB_upper"] = Decimal(str(bbands['BBU'].iloc[-1]))
            self.indicator_values["BB_lower"] = Decimal(str(bbands['BBL'].iloc[-1]))
            self.indicator_values["BB_middle"] = Decimal(str(bbands['BBM'].iloc[-1]))

    def _calculate_vwap(self) -> None:
        df = self.df
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        if vwap is not None:
            self.indicator_values["VWAP"] = Decimal(str(vwap.iloc[-1]))

    def _calculate_cci(self) -> None:
        df = self.df
        cci = ta.cci(df['high'], df['low'], df['close'], length=self.config["cci_period"])
        if cci is not None:
            self.indicator_values["CCI"] = Decimal(str(cci.iloc[-1]))

    def _calculate_williams_r(self) -> None:
        df = self.df
        wr = ta.willr(df['high'], df['low'], df['close'], length=self.config["williams_r_period"])
        if wr is not None:
            self.indicator_values["Williams_R"] = Decimal(str(wr.iloc[-1]))

    def _calculate_psar(self) -> None:
        df = self.df
        psar = ta.psar(df['high'], df['low'], af=self.config["psar_step"], max_af=self.config["psar_max_step"])
        if psar is not None and 'PSARl' in psar:
            latest_psar = psar['PSARl'].iloc[-1] if not pd.isna(psar['PSARl'].iloc[-1]) else psar['PSARs'].iloc[-1]
            self.indicator_values["PSAR"] = Decimal(str(latest_psar))

    def _calculate_sma(self) -> None:
        df = self.df
        sma = ta.sma(df['close'], length=self.config["sma_10_period"])
        if sma is not None:
            self.indicator_values["SMA_10"] = Decimal(str(sma.iloc[-1]))

    def _calculate_mfi(self) -> None:
        df = self.df
        mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=self.config["mfi_period"])
        if mfi is not None:
            self.indicator_values["MFI"] = Decimal(str(mfi.iloc[-1]))

    def _calculate_atr(self) -> None:
        df = self.df
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.config["atr_period"])
        if atr is not None:
            self.indicator_values["ATR"] = Decimal(str(atr.iloc[-1]))

    def _check_ema_alignment(self) -> float:
        short_ema = self.indicator_values.get("EMA_short")
        long_ema = self.indicator_values.get("EMA_long")
        if short_ema and long_ema:
            return 1.0 if short_ema > long_ema else -1.0 if short_ema < long_ema else 0.0
        return 0.0

    def _check_momentum(self) -> float:
        mom = self.indicator_values.get("Momentum")
        if mom:
            return 1.0 if mom > 0 else -1.0 if mom < 0 else 0.0
        return 0.0

    def _check_volume_confirmation(self) -> float:
        vol = self.indicator_values.get("Volume")
        vol_ma = self.indicator_values.get("Volume_MA")
        if vol and vol_ma:
            multiplier = Decimal(str(self.config["volume_confirmation_multiplier"]))
            return 1.0 if vol > vol_ma * multiplier else -1.0 if vol < vol_ma else 0.0
        return 0.0

    def _check_stoch_rsi(self) -> float:
        stoch_rsi = self.indicator_values.get("StochRSI")
        if stoch_rsi:
            oversold = self.config["stoch_rsi_oversold_threshold"]
            overbought = self.config["stoch_rsi_overbought_threshold"]
            return 1.0 if stoch_rsi < oversold else -1.0 if stoch_rsi > overbought else 0.0
        return 0.0

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI")
        if rsi:
            return 1.0 if rsi < 30 else -1.0 if rsi > 70 else 0.0
        return 0.0

    def _check_bollinger_bands(self, current_price: Decimal) -> float:
        upper = self.indicator_values.get("BB_upper")
        lower = self.indicator_values.get("BB_lower")
        if upper and lower and current_price:
            return 1.0 if current_price < lower else -1.0 if current_price > upper else 0.0
        return 0.0

    def _check_vwap(self, current_price: Decimal) -> float:
        vwap = self.indicator_values.get("VWAP")
        if vwap and current_price:
            return 1.0 if current_price > vwap else -1.0 if current_price < vwap else 0.0
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI")
        if cci:
            return 1.0 if cci < -100 else -1.0 if cci > 100 else 0.0
        return 0.0

    def _check_williams_r(self) -> float:
        wr = self.indicator_values.get("Williams_R")
        if wr:
            return 1.0 if wr < -80 else -1.0 if wr > -20 else 0.0
        return 0.0

    def _check_psar(self, current_price: Decimal) -> float:
        psar = self.indicator_values.get("PSAR")
        if psar and current_price:
            return 1.0 if current_price > psar else -1.0 if current_price < psar else 0.0
        return 0.0

    def _check_sma(self, current_price: Decimal) -> float:
        sma = self.indicator_values.get("SMA_10")
        if sma and current_price:
            return 1.0 if current_price > sma else -1.0 if current_price < sma else 0.0
        return 0.0

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI")
        if mfi:
            return 1.0 if mfi < 20 else -1.0 if mfi > 80 else 0.0
        return 0.0

    def _check_orderbook(self, orderbook_data: dict | None, current_price: Decimal) -> float:
        if orderbook_data and current_price:
            bid_vol = sum(v for p, v in orderbook_data['bids'] if p >= current_price * Decimal('0.995'))
            ask_vol = sum(v for p, v in orderbook_data['asks'] if p <= current_price * Decimal('1.005'))
            return 1.0 if bid_vol > ask_vol else -1.0 if ask_vol > bid_vol else 0.0
        return 0.0

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: dict | None) -> str:
        """Generates a trading signal based on weighted indicator scores."""
        weights = self.config["weight_sets"].get(self.config["active_weight_set"], {})
        scores = {}
        if self.config["indicators"].get("ema_alignment"):
            scores["ema_alignment"] = self._check_ema_alignment()
        if self.config["indicators"].get("momentum"):
            scores["momentum"] = self._check_momentum()
        if self.config["indicators"].get("volume_confirmation"):
            scores["volume_confirmation"] = self._check_volume_confirmation()
        if self.config["indicators"].get("stoch_rsi"):
            scores["stoch_rsi"] = self._check_stoch_rsi()
        if self.config["indicators"].get("rsi"):
            scores["rsi"] = self._check_rsi()
        if self.config["indicators"].get("bollinger_bands"):
            scores["bollinger_bands"] = self._check_bollinger_bands(current_price)
        if self.config["indicators"].get("vwap"):
            scores["vwap"] = self._check_vwap(current_price)
        if self.config["indicators"].get("cci"):
            scores["cci"] = self._check_cci()
        if self.config["indicators"].get("wr"):
            scores["wr"] = self._check_williams_r()
        if self.config["indicators"].get("psar"):
            scores["psar"] = self._check_psar(current_price)
        if self.config["indicators"].get("sma_10"):
            scores["sma_10"] = self._check_sma(current_price)
        if self.config["indicators"].get("mfi"):
            scores["mfi"] = self._check_mfi()
        if self.config["indicators"].get("orderbook"):
            scores["orderbook"] = self._check_orderbook(orderbook_data, current_price)

        total_score = sum(float(weights.get(k, 0)) * v for k, v in scores.items() if weights.get(k, 0) != 0)
        threshold = self.config["signal_score_threshold"]
        signal = "BUY" if total_score >= threshold else "SELL" if total_score <= -threshold else "HOLD"
        if signal != "HOLD":
            self.lg.info(f"Signal: {signal}, Score: {total_score:.2f}, Components: {', '.join(f'{k}={v:.1f}' for k, v in scores.items())}")
        return signal

    def calculate_entry_tp_sl(self, entry_price: Decimal, signal: str) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates entry, take-profit, and stop-loss prices."""
        atr = self.indicator_values.get("ATR")
        if not atr or not entry_price.is_finite() or entry_price <= 0:
            return entry_price, None, None

        tick = self.get_min_tick_size()
        sl_mult = Decimal(str(self.config["stop_loss_multiple"]))
        tp_mult = Decimal(str(self.config["take_profit_multiple"]))

        if signal == "BUY":
            sl = (entry_price - atr * sl_mult).quantize(tick, ROUND_DOWN)
            tp = (entry_price + atr * tp_mult).quantize(tick, ROUND_UP)
        elif signal == "SELL":
            sl = (entry_price + atr * sl_mult).quantize(tick, ROUND_UP)
            tp = (entry_price - atr * tp_mult).quantize(tick, ROUND_DOWN)
        else:
            return entry_price, None, None

        return entry_price, tp if tp > 0 else None, sl if sl > 0 else None


def calculate_position_size(
    balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal,
    entry_price: Decimal, market_info: dict, exchange: ccxt.Exchange, logger: logging.Logger
) -> Decimal | None:
    """Calculates position size based on risk management."""
    lg = logger
    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance = abs(entry_price - initial_stop_loss_price)
        if sl_distance <= 0:
            lg.error("SL distance invalid.")
            return None

        contract_size = Decimal(str(market_info.get('contractSize', '1')))
        risk_per_contract = sl_distance * contract_size
        size = risk_amount_quote / risk_per_contract

        min_amount = Decimal(str(market_info['limits']['amount']['min']))
        max_amount = Decimal(str(market_info['limits']['amount']['max']))
        step = Decimal(str(market_info['precision']['amount']))

        size = max(min_amount, min(max_amount, size))
        size = (size // step) * step

        if size <= 0 or not size.is_finite():
            lg.error(f"Invalid size: {size}")
            return None

        lg.info(f"Position size: {size}")
        return size
    except Exception as e:
        lg.error(f"Size calc error: {e}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Fetches open position details."""
    lg = logger
    try:
        positions = safe_api_call(exchange.fetch_positions, lg, symbols=[symbol])
        if not positions:
            return None

        for pos in positions:
            size = Decimal(str(pos.get('contracts', pos.get('info', {}).get('size', '0'))))
            if abs(size) > Decimal('1e-9'):
                pos['contractsDecimal'] = size
                pos['side'] = 'long' if size > 0 else 'short'
                pos['entryPriceDecimal'] = Decimal(str(pos.get('entryPrice', '0')))
                pos['unrealizedPnlDecimal'] = Decimal(str(pos.get('unrealizedPnl', '0')))
                info = pos.get('info', {})
                pos['stopLossPriceDecimal'] = Decimal(str(info.get('stopLoss', '0'))) if info.get('stopLoss') else None
                pos['takeProfitPriceDecimal'] = Decimal(str(info.get('takeProfit', '0'))) if info.get('takeProfit') else None
                pos['trailingStopLossValueDecimal'] = Decimal(str(info.get('trailingStop', '0'))) if info.get('trailingStop') else None
                lg.info(f"Active {pos['side']} position: {size}")
                return pos
        return None
    except Exception as e:
        lg.error(f"Position fetch error: {e}", exc_info=True)
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: dict, logger: logging.Logger) -> bool:
    """Sets leverage for the symbol."""
    lg = logger
    if not market_info.get('is_contract', False):
        lg.debug("Not a contract market.")
        return True
    try:
        params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)} if exchange.id == 'bybit' else {}
        safe_api_call(exchange.set_leverage, lg, leverage, symbol, params=params)
        lg.info(f"Leverage set to {leverage}x")
        return True
    except Exception as e:
        lg.error(f"Leverage set failed: {e}", exc_info=True)
        return False


def place_trade(
    exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal,
    market_info: dict, logger: logging.Logger, order_type: str = 'market',
    limit_price: Decimal | None = None, reduce_only: bool = False
) -> dict | None:
    """Places a trade order."""
    lg = logger
    side = 'buy' if trade_signal == "BUY" else 'sell'
    try:
        amount = float(position_size)
        params = {'reduceOnly': reduce_only}
        if exchange.id == 'bybit':
            params['positionIdx'] = 0
        order = safe_api_call(exchange.create_order, lg, symbol=symbol, type=order_type, side=side,
                              amount=amount, price=float(limit_price) if limit_price else None, params=params)
        if order:
            lg.info(f"Order placed: {order.get('id')}")
            return order
    except Exception as e:
        lg.error(f"Trade placement failed: {e}", exc_info=True)
    return None


def _set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict,
    logger: logging.Logger, stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None, trailing_stop_distance: Decimal | None = None,
    tsl_activation_price: Decimal | None = None
) -> bool:
    """Sets position protection (SL/TP/TSL) for Bybit V5."""
    lg = logger
    if 'bybit' not in exchange.id.lower():
        lg.error("Protection only for Bybit.")
        return False
    try:
        params = {'category': 'linear', 'symbol': market_info['id'], 'tpslMode': 'Full',
                  'positionIdx': 0}
        if trailing_stop_distance and tsl_activation_price:
            params['trailingStop'] = str(trailing_stop_distance)
            params['activePrice'] = str(tsl_activation_price)
        if stop_loss_price:
            params['stopLoss'] = str(stop_loss_price)
        if take_profit_price:
            params['takeProfit'] = str(take_profit_price)

        response = safe_api_call(exchange.private_post, lg, '/v5/position/set-trading-stop', params=params)
        if response and response.get('retCode') == 0:
            lg.info("Protection set successfully.")
            return True
        lg.error(f"Protection set failed: {response}")
        return False
    except Exception as e:
        lg.error(f"Protection error: {e}", exc_info=True)
        return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict,
    config: dict[str, Any], logger: logging.Logger, take_profit_price: Decimal | None = None
) -> bool:
    """Sets a trailing stop loss."""
    lg = logger
    if not config.get("enable_trailing_stop"):
        return False
    try:
        entry = position_info['entryPriceDecimal']
        side = position_info['side']
        cb_rate = Decimal(str(config["trailing_stop_callback_rate"]))
        act_pct = Decimal(str(config["trailing_stop_activation_percentage"]))

        analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info)
        tick = analyzer.get_min_tick_size()
        act_price = (entry + entry * act_pct if side == 'long' else entry - entry * act_pct).quantize(tick)
        dist = (act_price * cb_rate).quantize(tick)

        return _set_position_protection(exchange, symbol, market_info, position_info, lg,
                                        None, take_profit_price, dist, act_price)
    except Exception as e:
        lg.error(f"TSL error: {e}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger) -> None:
    """Executes one cycle of analysis and trading."""
    lg = logger
    lg.info(f"--- Cycle Start: {symbol} ({config['interval']}) ---")
    try:
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info:
            raise ValueError("Market info missing.")

        klines_df = fetch_klines_ccxt(exchange, symbol, CCXT_INTERVAL_MAP[config["interval"]], 500, lg)
        if klines_df.empty:
            raise ValueError("Klines data missing.")

        current_price = fetch_current_price_ccxt(exchange, symbol, lg) or klines_df['close'].iloc[-1]
        orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg) if config["indicators"].get("orderbook") else None

        analyzer = TradingAnalyzer(klines_df, lg, config, market_info)
        signal = analyzer.generate_trading_signal(current_price, orderbook_data)
        _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal)

        lg.info(f"Price: {current_price}, ATR: {analyzer.indicator_values.get('ATR', 'N/A')}, Signal: {signal}")

        if not config.get("enable_trading"):
            lg.debug("Trading disabled.")
            return

        position = get_open_position(exchange, symbol, lg)
        if not position:
            if signal in ["BUY", "SELL"]:
                balance = fetch_balance(exchange, config["quote_currency"], lg)
                if not balance:
                    raise ValueError("Balance fetch failed.")
                size = calculate_position_size(balance, config["risk_per_trade"], sl_calc, current_price, market_info, exchange, lg)
                if size:
                    set_leverage_ccxt(exchange, symbol, config["leverage"], market_info, lg)
                    order = place_trade(exchange, symbol, signal, size, market_info, lg, config["entry_order_type"])
                    if order:
                        time.sleep(config["position_confirm_delay_seconds"])
                        new_pos = get_open_position(exchange, symbol, lg)
                        if new_pos:
                            set_trailing_stop_loss(exchange, symbol, market_info, new_pos, config, lg, tp_calc)
        else:
            side = position['side']
            if (side == 'long' and signal == "SELL") or (side == 'short' and signal == "BUY"):
                close_sig = "SELL" if side == 'long' else "BUY"
                place_trade(exchange, symbol, close_sig, abs(position['contractsDecimal']), market_info, lg, 'market', reduce_only=True)
            elif config["enable_break_even"] and not position.get('trailingStopLossValueDecimal'):
                atr = analyzer.indicator_values.get("ATR")
                profit = (current_price - position['entryPriceDecimal']) / atr if side == 'long' else (position['entryPriceDecimal'] - current_price) / atr
                if profit >= Decimal(str(config["break_even_trigger_atr_multiple"])):
                    tick = analyzer.get_min_tick_size()
                    be_sl = (position['entryPriceDecimal'] + tick * config["break_even_offset_ticks"]).quantize(tick) if side == 'long' else \
                            (position['entryPriceDecimal'] - tick * config["break_even_offset_ticks"]).quantize(tick)
                    _set_position_protection(exchange, symbol, market_info, position, lg, be_sl)

    except Exception as e:
        lg.error(f"Cycle error: {e}", exc_info=True)


def main() -> None:
    """Main bot execution loop."""
    global config
    logger = setup_logger("ScalpXRX_Init")
    logger.info("Starting ScalpXRX Bot")

    try:
        config = load_config(CONFIG_FILE)
        symbol = config["symbol"]
        main_logger = setup_logger(f"ScalpXRX_{symbol.replace('/', '_').replace(':', '-')}", config)

        exchange = initialize_exchange(config, main_logger)
        if not exchange:
            main_logger.critical("Exchange init failed.")
            return

        while True:
            analyze_and_trade_symbol(exchange, symbol, config, main_logger)
            time.sleep(config["loop_delay_seconds"])
    except KeyboardInterrupt:
        logger.info("Shutdown via interrupt.")
    except Exception as e:
        logger.critical(f"Startup error: {e}", exc_info=True)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    config = load_config(CONFIG_FILE)
    main()
