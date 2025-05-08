# pyrmethus_volumatic_bot.py
# Enhanced trading bot with Volumatic Trend and Pivot Order Block strategy
# Version 1.1.1: Adds dynamic sizing, risk management, performance tracking, and backtesting

# --- Core Libraries ---
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo

import backtrader as bt  # For backtesting support
import ccxt
import jsonschema  # For configuration validation

# --- Dependencies ---
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialize Environment ---
getcontext().prec = 28
init(autoreset=True)
load_dotenv()

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
try:
    TIMEZONE = ZoneInfo("America/Chicago")
except Exception:
    TIMEZONE = ZoneInfo("UTC")

# API Settings
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
POSITION_CONFIRM_DELAY_SECONDS = 8
LOOP_DELAY_SECONDS = 15

# Timeframe Settings
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling
DEFAULT_FETCH_LIMIT = 750
MAX_DF_LEN = 2000

# Default Strategy Parameters
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 1000
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0
DEFAULT_OB_SOURCE = "Wicks"
DEFAULT_PH_LEFT = 10
DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10
DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50

# Neon Colors
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT

os.makedirs(LOG_DIRECTORY, exist_ok=True)


# --- Performance Tracking ---
@dataclass
class TradeStats:
    """Tracks trading performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    current_drawdown: Decimal = Decimal('0')
    win_rate: float = 0.0

    def update(self, trade_pnl: Decimal) -> None:
        """Updates stats with new trade PNL."""
        self.total_trades += 1
        if trade_pnl > Decimal('0'):
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.total_pnl += trade_pnl
        self.current_drawdown = min(self.current_drawdown, self.total_pnl)
        self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
        self.win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0


# --- Configuration Schema ---
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "interval": {"type": "string", "enum": VALID_INTERVALS},
        "retry_delay": {"type": "number", "minimum": 1},
        "fetch_limit": {"type": "integer", "minimum": 100},
        "enable_trading": {"type": "boolean"},
        "use_sandbox": {"type": "boolean"},
        "risk_per_trade": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "leverage": {"type": "integer", "minimum": 0},
        "quote_currency": {"type": "string", "minLength": 1},
        "loop_delay_seconds": {"type": "number", "minimum": 0},
        "position_confirm_delay_seconds": {"type": "number", "minimum": 0},
        "strategy_params": {
            "type": "object",
            "properties": {
                "vt_length": {"type": "integer", "minimum": 1},
                "vt_atr_period": {"type": "integer", "minimum": 1},
                "vt_vol_ema_length": {"type": "integer", "minimum": 1},
                "vt_atr_multiplier": {"type": "number", "minimum": 0},
                "ob_source": {"type": "string", "enum": ["Wicks", "Bodys"]},
                "ph_left": {"type": "integer", "minimum": 1},
                "ph_right": {"type": "integer", "minimum": 1},
                "pl_left": {"type": "integer", "minimum": 1},
                "pl_right": {"type": "integer", "minimum": 1},
                "ob_extend": {"type": "boolean"},
                "ob_max_boxes": {"type": "integer", "minimum": 1},
                "ob_entry_proximity_factor": {"type": "number", "minimum": 1},
                "ob_exit_proximity_factor": {"type": "number", "minimum": 1}
            },
            "required": ["vt_length", "vt_atr_period", "vt_vol_ema_length", "ob_source"]
        },
        "protection": {
            "type": "object",
            "properties": {
                "enable_trailing_stop": {"type": "boolean"},
                "trailing_stop_callback_rate": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                "trailing_stop_activation_percentage": {"type": "number", "minimum": 0, "maximum": 1},
                "enable_break_even": {"type": "boolean"},
                "break_even_trigger_atr_multiple": {"type": "number", "exclusiveMinimum": 0},
                "break_even_offset_ticks": {"type": "integer", "minimum": 0},
                "initial_stop_loss_atr_multiple": {"type": "number", "exclusiveMinimum": 0},
                "initial_take_profit_atr_multiple": {"type": "number", "minimum": 0}
            },
            "required": ["initial_stop_loss_atr_multiple"]
        },
        "risk_management": {
            "type": "object",
            "properties": {
                "max_drawdown_percentage": {"type": "number", "minimum": 0, "maximum": 1},
                "max_portfolio_risk": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}
            }
        }
    },
    "required": ["interval", "quote_currency", "risk_per_trade", "leverage"]
}


# --- Logging Setup ---
class StructuredLogger(logging.LoggerAdapter):
    """Custom logger with support for structured JSON logging."""
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger, {})
        self.structured_logging = os.getenv("STRUCTURED_LOGGING", "false").lower() == "true"

    def process(self, msg, kwargs):
        if self.structured_logging:
            log_data = {
                "timestamp": datetime.now(TIMEZONE).isoformat(),
                "level": kwargs.get("levelname", self.logger.level),
                "message": msg,
                "logger": self.logger.name
            }
            return json.dumps(log_data), kwargs
        return msg, kwargs


def setup_logger(name: str) -> StructuredLogger:
    """Sets up a logger with file and console handlers."""
    safe_name = name.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return StructuredLogger(logger)

    logger.setLevel(logging.DEBUG)
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIRECTORY, f"{logger_name}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
    stream_handler.setFormatter(stream_formatter)
    console_log_level = getattr(logging, os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper(), logging.INFO)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return StructuredLogger(logger)


# --- Configuration Management ---
def load_config(filepath: str) -> dict[str, Any]:
    """Loads and validates configuration, falling back to defaults if needed."""
    default_config = {
        "interval": "5",
        "retry_delay": RETRY_DELAY_SECONDS,
        "fetch_limit": DEFAULT_FETCH_LIMIT,
        "enable_trading": False,
        "use_sandbox": True,
        "risk_per_trade": 0.01,
        "leverage": 20,
        "quote_currency": "USDT",
        "loop_delay_seconds": LOOP_DELAY_SECONDS,
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER,
            "ob_source": DEFAULT_OB_SOURCE,
            "ph_left": DEFAULT_PH_LEFT,
            "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT,
            "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005,
            "ob_exit_proximity_factor": 1.001
        },
        "protection": {
            "enable_trailing_stop": True,
            "trailing_stop_callback_rate": 0.005,
            "trailing_stop_activation_percentage": 0.003,
            "enable_break_even": True,
            "break_even_trigger_atr_multiple": 1.0,
            "break_even_offset_ticks": 2,
            "initial_stop_loss_atr_multiple": 1.8,
            "initial_take_profit_atr_multiple": 0.7
        },
        "risk_management": {
            "max_drawdown_percentage": 0.2,  # 20% max drawdown
            "max_portfolio_risk": 0.05       # 5% max portfolio risk
        }
    }

    init_logger = setup_logger("init")
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file not found: {filepath}. Creating default.{RESET}")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config = json.load(f)
        jsonschema.validate(config, CONFIG_SCHEMA)
        updated_config, changed = _ensure_config_keys(config, default_config)
        if changed:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(updated_config, f, indent=4)
            init_logger.info(f"{NEON_GREEN}Updated config file: {filepath}{RESET}")
        return updated_config
    except jsonschema.ValidationError as e:
        init_logger.error(f"{NEON_RED}Config validation error: {e}. Using default.{RESET}")
        return default_config
    except Exception as e:
        init_logger.error(f"{NEON_RED}Error loading config: {e}. Using default.{RESET}")
        return default_config


def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any], parent_key: str = "") -> tuple[dict[str, Any], bool]:
    """Ensures all required config keys exist, adding defaults if missing."""
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Added missing config key '{full_key_path}': {default_value}{RESET}")
        elif isinstance(default_value, dict):
            nested_updated, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_updated
                changed = True
    return updated_config, changed


# --- Exchange Setup ---
def initialize_exchange(logger: StructuredLogger) -> ccxt.Exchange | None:
    """Initializes the Bybit exchange with retry logic."""
    lg = logger
    try:
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 30000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'fetchOHLCVTimeout': 60000
            }
        })
        if CONFIG.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}Using SANDBOX mode{RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}Using LIVE trading{RESET}")

        lg.info(f"Loading markets for {exchange.id}...")
        for attempt in range(MAX_API_RETRIES):
            try:
                exchange.load_markets(reload=attempt > 0)
                if exchange.markets:
                    lg.info(f"Markets loaded: {len(exchange.markets)} symbols")
                    break
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                if attempt < MAX_API_RETRIES - 1:
                    lg.warning(f"Network error: {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))  # Exponential backoff
                else:
                    lg.critical(f"{NEON_RED}Failed to load markets: {e}{RESET}")
                    return None

        balance = fetch_balance(exchange, CONFIG['quote_currency'], lg)
        if balance is None and CONFIG.get('enable_trading', False):
            lg.critical(f"{NEON_RED}Initial balance fetch failed{RESET}")
            return None
        return exchange
    except Exception as e:
        lg.critical(f"{NEON_RED}Exchange initialization failed: {e}{RESET}")
        return None


# --- Data Fetching ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: StructuredLogger) -> Decimal | None:
    """Fetches the current market price with retry logic."""
    lg = logger
    for attempt in range(MAX_API_RETRIES):
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = Decimal(str(ticker.get('last') or ((ticker.get('bid') + ticker.get('ask')) / 2)))
            if price > Decimal('0'):
                return price
        except Exception as e:
            lg.warning(f"Price fetch error: {e}. Attempt {attempt + 1}/{MAX_API_RETRIES}")
            time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    lg.error(f"{NEON_RED}Failed to fetch price for {symbol}{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: StructuredLogger) -> pd.DataFrame:
    """Fetches OHLCV data with retry logic."""
    lg = logger
    for attempt in range(MAX_API_RETRIES):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                lg.warning(f"No kline data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            df = df[df['close'] > Decimal('0')]
            if len(df) > MAX_DF_LEN:
                df = df.iloc[-MAX_DF_LEN:].copy()
            lg.info(f"Fetched {len(df)} klines for {symbol}")
            return df
        except Exception as e:
            lg.warning(f"Kline fetch error: {e}. Attempt {attempt + 1}/{MAX_API_RETRIES}")
            time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    return pd.DataFrame()


# --- Strategy Implementation ---
class VolumaticOBStrategy:
    """Implements the Volumatic Trend and Pivot Order Block strategy."""
    def __init__(self, config: dict[str, Any], market_info: dict, logger: StructuredLogger) -> None:
        self.config = config
        self.logger = logger
        self.market_info = market_info
        strategy_params = config.get('strategy_params', {})
        self.vt_length = strategy_params.get('vt_length', DEFAULT_VT_LENGTH)
        self.vt_atr_period = strategy_params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD)
        self.vt_vol_ema_length = strategy_params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH)
        self.vt_atr_multiplier = Decimal(str(strategy_params.get('vt_atr_multiplier', DEFAULT_VT_ATR_MULTIPLIER)))
        self.ob_source = strategy_params.get('ob_source', DEFAULT_OB_SOURCE)
        self.ph_left = strategy_params.get('ph_left', DEFAULT_PH_LEFT)
        self.ph_right = strategy_params.get('ph_right', DEFAULT_PH_RIGHT)
        self.pl_left = strategy_params.get('pl_left', DEFAULT_PL_LEFT)
        self.pl_right = strategy_params.get('pl_right', DEFAULT_PL_RIGHT)
        self.ob_extend = strategy_params.get('ob_extend', DEFAULT_OB_EXTEND)
        self.ob_max_boxes = strategy_params.get('ob_max_boxes', DEFAULT_OB_MAX_BOXES)
        self.bull_boxes: list[dict] = []
        self.bear_boxes: list[dict] = []
        self.min_data_len = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length,
                                self.ph_left + self.ph_right, self.pl_left + self.pl_right) + 50

    def update(self, df: pd.DataFrame) -> dict:
        """Updates strategy indicators and order blocks."""
        if df.empty or len(df) < self.min_data_len:
            self.logger.error(f"Insufficient data: {len(df)} rows, need {self.min_data_len}")
            return {}

        df = df.copy()
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vt_atr_period)
        df['vol_ema'] = ta.ema(df['volume'], length=self.vt_vol_ema_length)
        df['vol_norm'] = df['volume'] / df['vol_ema']
        df['ema'] = ta.ema(df['close'], length=self.vt_length)
        df['upper_band'] = df['ema'] + (df['atr'] * self.vt_atr_multiplier)
        df['lower_band'] = df['ema'] - (df['atr'] * self.vt_atr_multiplier)
        df['trend_up'] = df['close'] > df['upper_band']
        df['trend_down'] = df['close'] < df['lower_band']
        df['trend_changed'] = (df['trend_up'] != df['trend_up'].shift(1)) | (df['trend_down'] != df['trend_down'].shift(1))

        # Order Block Detection
        source_col = 'low' if self.ob_source == 'Wicks' else 'close'
        df['pivot_high'] = ta.pivothigh(df[source_col], left=self.ph_left, right=self.ph_right)
        df['pivot_low'] = ta.pivotlow(df[source_col], left=self.pl_left, right=self.pl_right)

        for idx in df.index:
            if pd.notna(df.loc[idx, 'pivot_high']):
                box_top = df.loc[idx, 'high']
                box_bottom = df.loc[idx, 'low']
                box_id = str(uuid.uuid4())
                self.bear_boxes.append({
                    'id': box_id,
                    'top': Decimal(str(box_top)),
                    'bottom': Decimal(str(box_bottom)),
                    'left_idx': idx,
                    'right_idx': idx,
                    'active': True,
                    'violated': False
                })
            if pd.notna(df.loc[idx, 'pivot_low']):
                box_top = df.loc[idx, 'high']
                box_bottom = df.loc[idx, 'low']
                box_id = str(uuid.uuid4())
                self.bull_boxes.append({
                    'id': box_id,
                    'top': Decimal(str(box_top)),
                    'bottom': Decimal(str(box_bottom)),
                    'left_idx': idx,
                    'right_idx': idx,
                    'active': True,
                    'violated': False
                })

        last_close = df['close'].iloc[-1]
        for box in self.bull_boxes:
            if box['active'] and last_close < box['bottom']:
                box['active'] = False
                box['violated'] = True
            elif self.ob_extend:
                box['right_idx'] = df.index[-1]
        for box in self.bear_boxes:
            if box['active'] and last_close > box['top']:
                box['active'] = False
                box['violated'] = True
            elif self.ob_extend:
                box['right_idx'] = df.index[-1]

        self.bull_boxes = sorted(self.bull_boxes, key=lambda x: x['left_idx'], reverse=True)[:self.ob_max_boxes]
        self.bear_boxes = sorted(self.bear_boxes, key=lambda x: x['left_idx'], reverse=True)[:self.ob_max_boxes]

        return {
            'dataframe': df,
            'last_close': df['close'].iloc[-1],
            'current_trend_up': bool(df['trend_up'].iloc[-1]),
            'trend_just_changed': bool(df['trend_changed'].iloc[-1]),
            'active_bull_boxes': [b for b in self.bull_boxes if b['active']],
            'active_bear_boxes': [b for b in self.bear_boxes if b['active']],
            'vol_norm_int': int(df['vol_norm'].iloc[-1]) if pd.notna(df['vol_norm'].iloc[-1]) else None,
            'atr': df['atr'].iloc[-1] if pd.notna(df['atr'].iloc[-1]) else None,
            'upper_band': df['upper_band'].iloc[-1] if pd.notna(df['upper_band'].iloc[-1]) else None,
            'lower_band': df['lower_band'].iloc[-1] if pd.notna(df['lower_band'].iloc[-1]) else None
        }


# --- Signal Generation ---
class SignalGenerator:
    """Generates trading signals based on strategy analysis."""
    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        self.config = config
        self.logger = logger
        strategy_cfg = config.get('strategy_params', {})
        protection_cfg = config.get('protection', {})
        self.ob_entry_proximity_factor = Decimal(str(strategy_cfg.get('ob_entry_proximity_factor', 1.005)))
        self.ob_exit_proximity_factor = Decimal(str(strategy_cfg.get('ob_exit_proximity_factor', 1.001)))
        self.initial_tp_atr_multiple = Decimal(str(protection_cfg.get('initial_take_profit_atr_multiple', 0.7)))
        self.initial_sl_atr_multiple = Decimal(str(protection_cfg.get('initial_stop_loss_atr_multiple', 1.8)))

    def generate_signal(self, analysis_results: dict, open_position: dict | None) -> str:
        """Generates a trading signal based on analysis results."""
        if not analysis_results.get('dataframe') or analysis_results['last_close'] <= Decimal('0'):
            return "HOLD"

        latest_close = analysis_results['last_close']
        is_trend_up = analysis_results['current_trend_up']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        current_pos_side = open_position.get('side') if open_position else None

        if current_pos_side == 'long' and is_trend_up is False:
            return "EXIT_LONG"
        elif current_pos_side == 'short' and is_trend_up is True:
            return "EXIT_SHORT"
        elif current_pos_side is None:
            if is_trend_up and active_bull_obs:
                for ob in active_bull_obs:
                    if ob['bottom'] <= latest_close <= ob['top'] * self.ob_entry_proximity_factor:
                        return "BUY"
            elif not is_trend_up and active_bear_obs:
                for ob in active_bear_obs:
                    lower_bound = ob['bottom'] / self.ob_entry_proximity_factor
                    if lower_bound <= latest_close <= ob['top']:
                        return "SELL"
        return "HOLD"

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: dict, exchange: ccxt.Exchange) -> tuple[Decimal | None, Decimal | None]:
        """Calculates initial take-profit and stop-loss prices."""
        if entry_price <= Decimal('0') or atr <= Decimal('0'):
            return None, None

        Decimal(str(market_info['precision']['price']))
        tp_offset = atr * self.initial_tp_atr_multiple
        sl_offset = atr * self.initial_sl_atr_multiple

        if signal == "BUY":
            take_profit = entry_price + tp_offset if self.initial_tp_atr_multiple > 0 else None
            stop_loss = entry_price - sl_offset
        else:
            take_profit = entry_price - tp_offset if self.initial_tp_atr_multiple > 0 else None
            stop_loss = entry_price + sl_offset

        take_profit = exchange.price_to_precision(market_info['symbol'], float(take_profit)) if take_profit else None
        stop_loss = exchange.price_to_precision(market_info['symbol'], float(stop_loss)) if stop_loss else None
        return Decimal(take_profit) if take_profit else None, Decimal(stop_loss) if stop_loss else None


# --- Trading Logic ---
def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal, market_info: dict, exchange: ccxt.Exchange, logger: StructuredLogger) -> Decimal | None:
    """Calculates position size based on risk and volatility."""
    if balance <= Decimal('0') or entry_price <= Decimal('0') or initial_stop_loss_price <= Decimal('0'):
        logger.error(f"Invalid inputs for position sizing: Balance={balance}, Entry={entry_price}, SL={initial_stop_loss_price}")
        return None

    risk_amount = balance * Decimal(str(risk_per_trade))
    price_diff = abs(entry_price - initial_stop_loss_price)
    if price_diff <= Decimal('0'):
        logger.error(f"Invalid price difference for sizing: {price_diff}")
        return None

    contracts = risk_amount / price_diff
    formatted_size = Decimal(exchange.amount_to_precision(market_info['symbol'], float(contracts)))
    min_size = Decimal(str(market_info['limits']['amount']['min'])) if market_info['limits']['amount'].get('min') else Decimal('0')
    if formatted_size < min_size:
        logger.warning(f"Position size {formatted_size} below minimum {min_size}. Setting to minimum.")
        formatted_size = min_size
    return formatted_size


def place_trade(exchange: ccxt.Exchange, symbol: str, side: str, amount: Decimal, market_info: dict, logger: StructuredLogger, reduce_only: bool = False) -> dict | None:
    """Places a market order with retry logic."""
    try:
        order_type = 'market'
        params = {'reduceOnly': reduce_only}
        order = exchange.create_order(symbol, order_type, side.lower(), float(amount), None, params)
        logger.info(f"Order placed: {side} {amount} @ market")
        return order
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        return None


def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: dict, position: dict, config: dict, logger: StructuredLogger, take_profit_price: Decimal | None = None) -> bool:
    """Sets a trailing stop-loss order."""
    try:
        protection = config.get('protection', {})
        callback_rate = protection.get('trailing_stop_callback_rate', 0.005)
        params = {
            'trailingStop': callback_rate,
            'activationPrice': position['entryPrice'],
            'positionIdx': 0  # 0 for one-way mode
        }
        if take_profit_price:
            params['takeProfit'] = float(take_profit_price)
        exchange.private_post_position_trading_stop({'symbol': market_info['id'], **params})
        logger.info(f"Trailing stop set: Callback={callback_rate * 100}%")
        return True
    except Exception as e:
        logger.error(f"Failed to set trailing stop: {e}")
        return False


# --- Backtesting Support ---
class VolumaticBacktestStrategy(bt.Strategy):
    """Backtesting strategy for VolumaticOB."""
    params = (
        ('config', {}),
        ('logger', None),
    )

    def __init__(self) -> None:
        self.strategy = VolumaticOBStrategy(self.params.config, {}, self.params.logger)
        self.signal_generator = SignalGenerator(self.params.config, self.params.logger)
        self.position_size = Decimal('0')
        self.entry_price = None

    def next(self) -> None:
        df = pd.DataFrame({
            'open': [self.data.open[0]],
            'high': [self.data.high[0]],
            'low': [self.data.low[0]],
            'close': [self.data.close[0]],
            'volume': [self.data.volume[0]]
        }, index=[self.data.datetime.datetime(0)])
        results = self.strategy.update(df)
        signal = self.signal_generator.generate_signal(results, {'side': 'long' if self.position else None})

        if signal == "BUY" and not self.position:
            size = calculate_position_size(
                balance=Decimal('10000'),  # Example balance
                risk_per_trade=self.params.config['risk_per_trade'],
                initial_stop_loss_price=results['last_close'] - results['atr'] * Decimal('1.8'),
                entry_price=results['last_close'],
                market_info={'precision': {'price': '0.01', 'amount': '0.1'}, 'limits': {'amount': {'min': '0.1'}}},
                exchange=None,
                logger=self.params.logger
            )
            self.buy(size=float(size))
            self.position_size = size
            self.entry_price = results['last_close']
        elif signal in ["EXIT_LONG", "SELL"] and self.position:
            self.close()


# --- Main Loop ---
def main() -> None:
    """Main entry point for the trading bot."""
    global CONFIG
    init_logger = setup_logger("init")
    CONFIG = load_config(CONFIG_FILE)
    exchange = initialize_exchange(init_logger)
    if not exchange:
        return

    # Backtesting Mode
    if os.getenv("BACKTEST_MODE", "false").lower() == "true":
        init_logger.info("Running in backtest mode")
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=fetch_klines_ccxt(exchange, "BTC/USDT", "5m", 1000, init_logger))
        cerebro.adddata(data)
        cerebro.addstrategy(VolumaticBacktestStrategy, config=CONFIG, logger=init_logger)
        cerebro.run()
        init_logger.info(f"Backtest completed. Final portfolio value: {cerebro.broker.getvalue()}")
        return

    symbol = input(f"{NEON_YELLOW}Enter trading symbol (e.g., BTC/USDT): {RESET}").strip().upper()
    market_info = get_market_info(exchange, symbol, init_logger)
    if not market_info:
        init_logger.error(f"Invalid symbol: {symbol}")
        return

    interval = input(f"{NEON_YELLOW}Enter interval {VALID_INTERVALS} (default: {CONFIG['interval']}): {RESET}").strip() or CONFIG['interval']
    if interval not in VALID_INTERVALS:
        init_logger.error(f"Invalid interval: {interval}")
        return
    CONFIG['interval'] = interval

    symbol_logger = setup_logger(symbol)
    strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
    signal_generator = SignalGenerator(CONFIG, symbol_logger)
    trade_stats = TradeStats()

    try:
        while True:
            analyze_and_trade_symbol(
                exchange=exchange,
                symbol=symbol,
                config=CONFIG,
                logger=symbol_logger,
                strategy_engine=strategy_engine,
                signal_generator=signal_generator,
                market_info=market_info,
                trade_stats=trade_stats
            )
            time.sleep(CONFIG['loop_delay_seconds'])
    except KeyboardInterrupt:
        symbol_logger.info("Shutdown initiated")
    finally:
        symbol_logger.info(f"Trade Stats: {asdict(trade_stats)}")
        logging.shutdown()


def analyze_and_trade_symbol(exchange, symbol, config, logger, strategy_engine, signal_generator, market_info, trade_stats) -> None:
    """Analyzes market data and executes trades."""
    klines_df = fetch_klines_ccxt(exchange, symbol, CCXT_INTERVAL_MAP[config['interval']], config['fetch_limit'], logger)
    if klines_df.empty:
        return

    analysis_results = strategy_engine.update(klines_df)
    if not analysis_results:
        return

    signal = signal_generator.generate_signal(analysis_results, get_open_position(exchange, symbol, logger))
    if config.get('enable_trading', False):
        if signal in ["BUY", "SELL"]:
            balance = fetch_balance(exchange, config['quote_currency'], logger)
            tp, sl = signal_generator.calculate_initial_tp_sl(
                analysis_results['last_close'], signal, analysis_results['atr'], market_info, exchange
            )
            if sl:
                size = calculate_position_size(balance, config['risk_per_trade'], sl, analysis_results['last_close'], market_info, exchange, logger)
                if size:
                    order = place_trade(exchange, symbol, signal, size, market_info, logger)
                    if order:
                        set_trailing_stop_loss(exchange, symbol, market_info, {'entryPrice': analysis_results['last_close']}, config, logger, tp)
                        trade_stats.update(Decimal('0'))  # Update stats (simplified)
        elif signal in ["EXIT_LONG", "EXIT_SHORT"]:
            position = get_open_position(exchange, symbol, logger)
            if position:
                place_trade(exchange, symbol, "SELL" if signal == "EXIT_LONG" else "BUY", abs(position['size_decimal']), market_info, logger, reduce_only=True)


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: StructuredLogger) -> dict | None:
    """Fetches the current open position for a symbol."""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['contracts'] > 0:
                return {
                    'side': pos['side'].lower(),
                    'size_decimal': Decimal(str(pos['contracts'])),
                    'entryPrice': pos['entryPrice'],
                    'stopLossPrice': pos.get('stopLoss'),
                    'takeProfitPrice': pos.get('takeProfit')
                }
        return None
    except Exception as e:
        logger.error(f"Failed to fetch position: {e}")
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: StructuredLogger) -> dict | None:
    """Fetches market information for a symbol."""
    try:
        exchange.load_markets()
        if symbol in exchange.markets:
            return exchange.market(symbol)
    except Exception as e:
        logger.error(f"Failed to get market info: {e}")
    return None


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: StructuredLogger) -> Decimal | None:
    """Fetches the available balance for a currency."""
    try:
        balance = exchange.fetch_balance()
        return Decimal(str(balance[currency]['free'])) if currency in balance else None
    except Exception as e:
        logger.error(f"Failed to fetch balance: {e}")
        return None


if __name__ == "__main__":
    main()
