```python
# -*- coding: utf-8 -*-
# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) and multiple enhancement snippets.

# Standard Library Imports
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List, TypedDict, Union
from zoneinfo import ZoneInfo # Preferred over pytz for standard library timezone handling

# Third-party Imports
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv
from pydantic import (BaseModel, Field, ValidationError, validator) # Use Pydantic v1 validator
from tenacity import (retry, stop_after_attempt, wait_exponential,
                      retry_if_exception_type, retry_any) # Added retry_any

# --- Initialize Environment and Settings ---
getcontext().prec = 28  # Set precision for Decimal calculations
init(autoreset=True)    # Initialize Colorama for colored terminal output
load_dotenv()           # Load environment variables from .env file

# --- Constants ---
# API Credentials (Loaded from .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file")

# File Paths and Directories
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"

# Timezone Configuration (Adjust as needed)
# Use IANA timezone names: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
DEFAULT_TIMEZONE_STR = "America/Chicago"
try:
    TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_STR)
except Exception:
    print(f"{Fore.YELLOW}Warning: Could not load timezone '{DEFAULT_TIMEZONE_STR}'. Using UTC.{Style.RESET_ALL}")
    TIMEZONE = timezone.utc

# API Interaction Settings
MAX_API_RETRIES = 5  # Increased retries for more resilience
RETRY_DELAY_SECONDS = 5
RETRY_WAIT_MULTIPLIER = 1.5 # Exponential backoff factor
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 20
# CCXT exceptions considered retryable
RETRYABLE_CCXT_EXCEPTIONS = (
    ccxt.RequestTimeout,
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
)

# Bot Operation Settings
LOOP_DELAY_SECONDS = 15  # Time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY_SECONDS = 8 # Wait after placing order before confirming position state
MAX_DF_LEN = 2000        # Maximum length of DataFrame to keep in memory

# Supported Intervals and Mapping for CCXT
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_INTERVAL = "5" # Default if not in config

# Default Quote Currency (Overridden by config.json)
DEFAULT_QUOTE_CURRENCY = "USDT"

# --- Default Parameters (Incorporating Snippet Defaults) ---
# Strategy/Indicator Parameters
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 1060 # Lookback window for volume percentile rank
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0
DEFAULT_OB_SOURCE = "Wicks"
DEFAULT_PH_LEFT = 10
DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10
DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50
DEFAULT_OB_ENTRY_PROXIMITY_FACTOR = 1.005
DEFAULT_OB_EXIT_PROXIMITY_FACTOR = 1.001
DEFAULT_VOLUME_THRESHOLD = 75        # [S1] Min normalized volume % (Percentile Rank)
DEFAULT_ADX_THRESHOLD = 25           # [S3] Min ADX value for trend strength
DEFAULT_RSI_CONFIRM = True           # [S7] Use RSI filter
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_RSI_OVERBOUGHT = 70
DEFAULT_DYNAMIC_OB_PROXIMITY = True  # [S8] Adjust OB proximity based on volatility
DEFAULT_FIB_TP_LEVELS = False        # [S10] Use Fibonacci levels for TP
DEFAULT_CANDLESTICK_CONFIRM = True   # [S11] Use candlestick pattern filter
DEFAULT_BREAKOUT_CONFIRM = False     # [S13] Use breakout entry logic
DEFAULT_VWAP_CONFIRM = True          # [S15] Use VWAP filter
DEFAULT_VALIDATE_OB_VOLUME = True    # [S17] Check volume on OB candle
DEFAULT_ADAPTIVE_OB_SIZING = True    # [S18] Adjust OB size based on ATR
DEFAULT_MULTI_TF_CONFIRM = False     # [S19] Use higher timeframe confirmation
DEFAULT_HIGHER_TF = "60"             # [S19] Higher timeframe interval
DEFAULT_OB_CONFLUENCE = 1            # [S21] Min number of OBs to hit for entry
DEFAULT_MOMENTUM_CONFIRM = True      # [S23] Use ROC momentum filter
DEFAULT_ROC_LENGTH = 10              # [S23] ROC lookback period
DEFAULT_OB_EXPIRY_ATR_PERIODS = 50.0 # [S25] OB expiry based on ATR periods (None to disable)

# Protection Parameters
DEFAULT_ENABLE_TRAILING_STOP = True
DEFAULT_TSL_CALLBACK_RATE = 0.005
DEFAULT_TSL_ACTIVATION_PERCENTAGE = 0.003
DEFAULT_ENABLE_BREAK_EVEN = True
DEFAULT_BE_TRIGGER_ATR_MULTIPLE = 1.0
DEFAULT_BE_OFFSET_TICKS = 2
DEFAULT_INITIAL_SL_ATR_MULTIPLE = 1.8
DEFAULT_INITIAL_TP_ATR_MULTIPLE = 0.7
DEFAULT_DYNAMIC_ATR_MULTIPLIER = True # [S2] Adjust TP/SL multiples based on ATR percentile
DEFAULT_VOLATILITY_SL_ADJUST = True   # [S5] Widen SL based on ATR std dev
DEFAULT_TSL_ACTIVATION_DELAY_ATR = 0.5 # [S9] Min profit in ATR before TSL activates
DEFAULT_USE_ATR_TRAILING_STOP = True  # [S12] Use ATR for TSL distance
DEFAULT_TSL_ATR_MULTIPLE = 1.5        # [S12] ATR multiple for TSL distance
DEFAULT_MIN_RR_RATIO = 1.5            # [S14] Minimum Risk/Reward ratio filter
DEFAULT_DYNAMIC_BE_TRIGGER = True     # [S16] Adjust BE trigger based on trade duration
DEFAULT_DYNAMIC_TSL_CALLBACK = True   # [S22] Adjust TSL distance based on profit
DEFAULT_SL_TRAIL_TO_OB = False        # [S24] Trail SL to nearest valid OB boundary
DEFAULT_PARTIAL_TP_LEVELS = [         # [S4] Default partial TP structure
    {'multiple': 0.7, 'percentage': 0.5},
    {'multiple': 1.5, 'percentage': 0.5}
]

# Risk & General Parameters
DEFAULT_FETCH_LIMIT = 750
DEFAULT_ATR_POSITION_SIZING = True    # [S20] Adjust position size based on ATR volatility
DEFAULT_MAX_HOLDING_MINUTES = 240     # [S6] Max time to hold position (None to disable)
DEFAULT_ORDER_TYPE = "market"

# Neon Color Scheme for Logging
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Global Variables ---
market_info_cache: Dict[str, Dict] = {}
CONFIG: Dict[str, Any] = {} # Loaded later
QUOTE_CURRENCY: str = DEFAULT_QUOTE_CURRENCY # Updated from config
config_mtime: float = 0.0 # Last modification time of config file

# --- Configuration Validation with Pydantic v1 Syntax ---

class TakeProfitLevel(BaseModel):
    """[S4] Defines a single level for partial take profit."""
    multiple: float = Field(..., gt=0, description="[S4] ATR multiple for this TP level (relative to base TP multiple)")
    percentage: float = Field(..., gt=0, le=1.0, description="[S4] Percentage of the position to close at this level (0.0 to 1.0)")

class StrategyParams(BaseModel):
    """Strategy-specific parameters incorporating enhancements."""
    vt_length: int = Field(DEFAULT_VT_LENGTH, gt=0, description="Length for Volumatic Trend Moving Average")
    vt_atr_period: int = Field(DEFAULT_VT_ATR_PERIOD, gt=0, description="ATR period for Volumatic Trend bands & general ATR")
    vt_vol_ema_length: int = Field(DEFAULT_VT_VOL_EMA_LENGTH, gt=0, description="Lookback period for Volume Normalization (Percentile Rank Window)")
    vt_atr_multiplier: float = Field(DEFAULT_VT_ATR_MULTIPLIER, gt=0, description="ATR multiplier for Volumatic Trend upper/lower bands")
    vt_step_atr_multiplier: float = Field(DEFAULT_VT_STEP_ATR_MULTIPLIER, ge=0, description="ATR multiplier for Volumatic volume step visualization (if used)")
    ob_source: str = Field(DEFAULT_OB_SOURCE, pattern="^(Wicks|Bodys)$", description="Source for OB pivots ('Wicks' or 'Bodys')")
    ph_left: int = Field(DEFAULT_PH_LEFT, gt=0, description="Left lookback for Pivot High")
    ph_right: int = Field(DEFAULT_PH_RIGHT, gt=0, description="Right lookback for Pivot High")
    pl_left: int = Field(DEFAULT_PL_LEFT, gt=0, description="Left lookback for Pivot Low")
    pl_right: int = Field(DEFAULT_PL_RIGHT, gt=0, description="Right lookback for Pivot Low")
    ob_extend: bool = Field(DEFAULT_OB_EXTEND, description="Extend OB boxes to the right until violated")
    ob_max_boxes: int = Field(DEFAULT_OB_MAX_BOXES, gt=0, description="Max number of active OBs to track")
    ob_entry_proximity_factor: float = Field(DEFAULT_OB_ENTRY_PROXIMITY_FACTOR, ge=1.0, description="Factor to extend OB range for entry signal (e.g., 1.005 = 0.5% beyond edge)")
    ob_exit_proximity_factor: float = Field(DEFAULT_OB_EXIT_PROXIMITY_FACTOR, ge=1.0, description="Factor to shrink OB range for exit signal (e.g., 1.001 means exit 0.1% before edge)")
    volume_threshold: int = Field(DEFAULT_VOLUME_THRESHOLD, ge=0, le=100, description="[S1] Min normalized volume percentile rank for entry confirmation")
    adx_threshold: int = Field(DEFAULT_ADX_THRESHOLD, ge=0, description="[S3] Min ADX value for trend strength confirmation")
    rsi_confirm: bool = Field(DEFAULT_RSI_CONFIRM, description="[S7] Enable RSI overbought/oversold filter for entries")
    rsi_oversold: int = Field(DEFAULT_RSI_OVERSOLD, ge=0, lt=100, description="[S7] RSI oversold level")
    rsi_overbought: int = Field(DEFAULT_RSI_OVERBOUGHT, ge=0, lt=100, description="[S7] RSI overbought level")
    dynamic_ob_proximity: bool = Field(DEFAULT_DYNAMIC_OB_PROXIMITY, description="[S8] Adjust OB entry proximity based on ATR volatility")
    fib_tp_levels: bool = Field(DEFAULT_FIB_TP_LEVELS, description="[S10] Use Fibonacci retracement levels for TP instead of ATR multiples")
    candlestick_confirm: bool = Field(DEFAULT_CANDLESTICK_CONFIRM, description="[S11] Enable candlestick pattern filter (e.g., Engulfing)")
    breakout_confirm: bool = Field(DEFAULT_BREAKOUT_CONFIRM, description="[S13] Use OB breakout logic for entries instead of OB zone entry")
    vwap_confirm: bool = Field(DEFAULT_VWAP_CONFIRM, description="[S15] Enable VWAP filter (price must be above VWAP for long, below for short)")
    validate_ob_volume: bool = Field(DEFAULT_VALIDATE_OB_VOLUME, description="[S17] Require OB formation candle to have volume above a threshold")
    adaptive_ob_sizing: bool = Field(DEFAULT_ADAPTIVE_OB_SIZING, description="[S18] Adjust OB box size slightly based on current ATR")
    multi_tf_confirm: bool = Field(DEFAULT_MULTI_TF_CONFIRM, description="[S19] Enable higher timeframe trend confirmation")
    higher_tf: str = Field(DEFAULT_HIGHER_TF, description="[S19] Higher timeframe interval (e.g., '60', '240')")
    ob_confluence: int = Field(DEFAULT_OB_CONFLUENCE, ge=1, description="[S21] Minimum number of active OBs price must touch for entry signal")
    momentum_confirm: bool = Field(DEFAULT_MOMENTUM_CONFIRM, description="[S23] Enable momentum filter using ROC")
    roc_length: int = Field(DEFAULT_ROC_LENGTH, gt=0, description="[S23] Lookback period for ROC indicator")
    ob_expiry_atr_periods: Optional[float] = Field(DEFAULT_OB_EXPIRY_ATR_PERIODS, ge=0, description="[S25] Expire OBs after this many ATR-adjusted periods (None to disable)")

    @validator('rsi_oversold')
    def check_rsi_levels(cls, v, values):
        if 'rsi_overbought' in values and v >= values['rsi_overbought']:
            raise ValueError('rsi_oversold must be less than rsi_overbought')
        return v

    @validator('higher_tf')
    def check_higher_tf(cls, v: str, values):
        if values.get('multi_tf_confirm') and v not in VALID_INTERVALS:
             raise ValueError(f"higher_tf '{v}' must be one of {VALID_INTERVALS} if multi_tf_confirm is true")
        return v

class ProtectionConfig(BaseModel):
    """Position protection parameters (SL, TP, BE, TSL) incorporating enhancements."""
    enable_trailing_stop: bool = Field(DEFAULT_ENABLE_TRAILING_STOP, description="Enable Trailing Stop Loss")
    trailing_stop_callback_rate: float = Field(DEFAULT_TSL_CALLBACK_RATE, ge=0, description="[If ATR TSL Disabled] TSL distance as a percentage of entry price (e.g., 0.005 = 0.5%)")
    trailing_stop_activation_percentage: float = Field(DEFAULT_TSL_ACTIVATION_PERCENTAGE, ge=0, description="[If ATR TSL Disabled] Profit percentage to activate TSL")
    enable_break_even: bool = Field(DEFAULT_ENABLE_BREAK_EVEN, description="Enable moving Stop Loss to Break Even")
    break_even_trigger_atr_multiple: float = Field(DEFAULT_BE_TRIGGER_ATR_MULTIPLE, ge=0, description="Profit in ATR multiples required to trigger Break Even")
    break_even_offset_ticks: int = Field(DEFAULT_BE_OFFSET_TICKS, ge=0, description="Number of price ticks above/below entry price for Break Even SL")
    initial_stop_loss_atr_multiple: float = Field(DEFAULT_INITIAL_SL_ATR_MULTIPLE, gt=0, description="Base ATR multiple for initial Stop Loss distance")
    initial_take_profit_atr_multiple: float = Field(DEFAULT_INITIAL_TP_ATR_MULTIPLE, gt=0, description="Base ATR multiple for Take Profit (used for first partial TP or full TP)")
    dynamic_atr_multiplier: bool = Field(DEFAULT_DYNAMIC_ATR_MULTIPLIER, description="[S2] Adjust initial TP/SL ATR multiples based on ATR percentile")
    volatility_sl_adjust: bool = Field(DEFAULT_VOLATILITY_SL_ADJUST, description="[S5] Widen initial SL based on recent ATR standard deviation")
    tsl_activation_delay_atr: float = Field(DEFAULT_TSL_ACTIVATION_DELAY_ATR, ge=0, description="[S9] Minimum profit in ATR multiples before TSL becomes active")
    use_atr_trailing_stop: bool = Field(DEFAULT_USE_ATR_TRAILING_STOP, description="[S12] Use ATR to calculate TSL distance instead of callback rate")
    trailing_stop_atr_multiple: float = Field(DEFAULT_TSL_ATR_MULTIPLE, gt=0, description="[S12] ATR multiple for TSL distance")
    min_rr_ratio: float = Field(DEFAULT_MIN_RR_RATIO, ge=0, description="[S14] Minimum required Risk/Reward ratio (Reward/Risk) for trade entry")
    dynamic_be_trigger: bool = Field(DEFAULT_DYNAMIC_BE_TRIGGER, description="[S16] Adjust BE trigger threshold based on trade holding time")
    dynamic_tsl_callback: bool = Field(DEFAULT_DYNAMIC_TSL_CALLBACK, description="[S22] Dynamically adjust TSL distance based on current profit")
    sl_trail_to_ob: bool = Field(DEFAULT_SL_TRAIL_TO_OB, description="[S24] Trail SL to the boundary of the nearest valid Order Block")
    partial_tp_levels: Optional[List[TakeProfitLevel]] = Field(default_factory=lambda: DEFAULT_PARTIAL_TP_LEVELS.copy(), description="[S4] List of partial take profit levels (multiple, percentage). Set to null or empty list to disable.")

    @validator('partial_tp_levels')
    def check_partial_tp_percentages(cls, v: Optional[List[TakeProfitLevel]]):
        if v:
            try:
                total_percentage = sum(Decimal(str(level.percentage)) for level in v)
                if abs(total_percentage - Decimal('1.0')) > Decimal('0.01'):
                    raise ValueError(f"Partial TP percentages must sum to 1.0 (current sum: {total_percentage:.4f})")
            except InvalidOperation:
                 raise ValueError("Invalid numeric value found in partial TP percentages.")
        return v

class Config(BaseModel):
    """Main bot configuration incorporating enhancements."""
    interval: str = Field(DEFAULT_INTERVAL, description=f"Trading interval, must be one of {VALID_INTERVALS}")
    retry_delay: int = Field(RETRY_DELAY_SECONDS, gt=0, description="Base delay in seconds between API retry attempts")
    fetch_limit: int = Field(DEFAULT_FETCH_LIMIT, gt=50, le=1500, description="Number of historical klines to fetch initially")
    orderbook_limit: int = Field(25, gt=0, le=200, description="Depth of order book to fetch (if needed)")
    enable_trading: bool = Field(False, description="Master switch to enable/disable placing real trades")
    use_sandbox: bool = Field(True, description="Use exchange's testnet/sandbox environment")
    risk_per_trade: float = Field(0.01, gt=0, lt=1.0, description="Fraction of available balance to risk per trade (e.g., 0.01 = 1%)")
    leverage: int = Field(20, gt=0, description="Leverage to use for contract trading")
    max_concurrent_positions: int = Field(1, ge=1, description="Maximum number of positions to hold concurrently (per symbol)")
    quote_currency: str = Field(DEFAULT_QUOTE_CURRENCY, min_length=2, description="Quote currency for trading pairs (e.g., USDT, USD)")
    loop_delay_seconds: int = Field(LOOP_DELAY_SECONDS, ge=1, description="Delay in seconds between trading cycles")
    position_confirm_delay_seconds: int = Field(POSITION_CONFIRM_DELAY_SECONDS, ge=1, description="Delay after placing order before checking position status")
    strategy_params: StrategyParams = Field(default_factory=StrategyParams, description="Strategy-specific parameters")
    protection: ProtectionConfig = Field(default_factory=ProtectionConfig, description="Position protection settings (SL/TP/BE/TSL)")
    max_holding_minutes: Optional[int] = Field(DEFAULT_MAX_HOLDING_MINUTES, ge=0, description="[S6] Maximum time in minutes to hold a position (None to disable)")
    atr_position_sizing: bool = Field(DEFAULT_ATR_POSITION_SIZING, description="[S20] Adjust position size based on current ATR volatility")
    order_type: str = Field(DEFAULT_ORDER_TYPE, pattern="^(market|limit)$", description="Order type for entries ('market' or 'limit')")

    @validator('interval')
    def check_interval(cls, v: str):
        if v not in VALID_INTERVALS:
            raise ValueError(f"Interval must be one of {VALID_INTERVALS}")
        return v

# --- Configuration Loading ---

class SensitiveFormatter(logging.Formatter):
    """Formatter that removes sensitive information from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Sets up a logger with specified name and level."""
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.setLevel(level)
        return logger
    logger.setLevel(level)
    logger.propagate = False
    try:
        file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger

base_logger = setup_logger("PyrmethusBot_Base")

def load_config(filepath: str) -> Dict[str, Any]:
    """Loads, validates, and potentially creates the configuration file."""
    global config_mtime
    default_config_model = Config()
    # Use model_dump() for Pydantic v2+, dict() for v1
    # default_config_dict = default_config_model.model_dump(mode='json') # Pydantic v2
    default_config_dict = default_config_model.dict() # Pydantic v1

    if not os.path.exists(filepath):
        base_logger.warning(f"Config file not found at '{filepath}'. Creating default config.")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config_dict, f, indent=4)
            base_logger.info(f"Created default config file: {filepath}")
            config_mtime = os.path.getmtime(filepath)
            return default_config_dict
        except IOError as e:
            base_logger.error(f"Error creating default config file '{filepath}': {e}. Using internal defaults.")
            config_mtime = time.time()
            return default_config_dict

    try:
        current_mtime = os.path.getmtime(filepath)
        with open(filepath, 'r', encoding="utf-8") as f:
            config_from_file = json.load(f)
        validated_config = Config(**config_from_file)
        # validated_config_dict = validated_config.model_dump(mode='json') # Pydantic v2
        validated_config_dict = validated_config.dict() # Pydantic v1
        base_logger.info("Configuration loaded and validated successfully.")
        config_mtime = current_mtime
        return validated_config_dict
    except (json.JSONDecodeError, ValidationError) as e:
        base_logger.error(f"Config error in '{filepath}': {e}. ", exc_info=False)
        base_logger.warning("Attempting to use default config and overwrite the invalid file.")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config_dict, f, indent=4)
            base_logger.info(f"Overwrote '{filepath}' with default configuration.")
            config_mtime = os.path.getmtime(filepath)
            return default_config_dict
        except IOError as e_write:
            base_logger.error(f"Error overwriting config file '{filepath}': {e_write}. Using internal defaults.")
            config_mtime = time.time()
            return default_config_dict
    except FileNotFoundError:
         base_logger.error(f"Config file '{filepath}' disappeared unexpectedly. Using internal defaults.")
         config_mtime = time.time()
         return default_config_dict
    except Exception as e:
        base_logger.critical(f"Unexpected error loading config '{filepath}': {e}", exc_info=True)
        base_logger.warning("Using internal default configuration.")
        config_mtime = time.time()
        return default_config_dict

# Load initial configuration
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", DEFAULT_QUOTE_CURRENCY)

# --- CCXT Exchange Setup ---

def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object."""
    global CONFIG
    logger.info("Initializing CCXT Bybit exchange...")
    try:
        exchange_params = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 20000,
                'fetchOHLCVTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                # 'brokerId': 'PyrmethusV1', # Optional
            }
        }
        exchange = ccxt.bybit(exchange_params)

        if CONFIG.get('use_sandbox', True):
            logger.warning(f"{NEON_YELLOW}SANDBOX MODE ENABLED (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            logger.warning(f"{BRIGHT}{NEON_RED}LIVE TRADING MODE ENABLED{RESET}")

        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets(True)
        logger.info(f"Markets loaded successfully for {exchange.id}.")

        logger.info("Fetching initial balance to verify connection...")
        account_type = 'CONTRACT' # Adjust if needed
        balance = exchange.fetch_balance(params={'accountType': account_type})
        available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
        logger.info(f"{NEON_GREEN}Exchange initialized successfully. Account Type: {account_type}, Available {QUOTE_CURRENCY}: {available_quote}{RESET}")
        return exchange
    except (ccxt.AuthenticationError, ccxt.ExchangeError) as e:
        logger.critical(f"{NEON_RED}Exchange Initialization Failed: {type(e).__name__} - {e}{RESET}", exc_info=False)
        return None
    except ccxt.NetworkError as e:
        logger.critical(f"{NEON_RED}Network error during exchange initialization: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        logger.critical(f"{NEON_RED}An unexpected error occurred during exchange initialization: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching Helpers with Retries ---

def _should_retry_ccxt(exception: BaseException) -> bool:
    """Determines if a CCXT exception is retryable. (Custom logic for tenacity)"""
    if isinstance(exception, RETRYABLE_CCXT_EXCEPTIONS):
        base_logger.warning(f"Retryable CCXT error encountered: {type(exception).__name__}. Retrying...")
        return True
    if isinstance(exception, ccxt.ExchangeError):
        error_str = str(exception).lower()
        if "server error" in error_str or "busy" in error_str or "try again later" in error_str or "request timeout" in error_str or "temporarily unavailable" in error_str:
             base_logger.warning(f"Retryable ExchangeError encountered: {exception}. Retrying...")
             return True
    base_logger.debug(f"Non-retryable error encountered: {type(exception).__name__}")
    return False

# Define retry decorator using retry_any to combine conditions
ccxt_retry_decorator = retry(
    stop=stop_after_attempt(MAX_API_RETRIES),
    wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_any(retry_if_exception_type(RETRYABLE_CCXT_EXCEPTIONS), _should_retry_ccxt), # Combine standard types and custom logic
    before_sleep=lambda retry_state: base_logger.info(f"Retrying API call (attempt {retry_state.attempt_number})...")
)

@ccxt_retry_decorator
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the current market price for a symbol using fetch_ticker."""
    logger.debug(f"Fetching current price for {symbol}...")
    ticker = {} # Initialize ticker
    try:
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker.get('last')
        if last_price is not None and last_price > 0:
            price = Decimal(str(last_price))
            logger.debug(f"Fetched last price for {symbol}: {price}")
            return price
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid_price = (Decimal(str(bid)) + Decimal(str(ask))) / Decimal('2')
            logger.debug(f"Fetched mid price (bid/ask) for {symbol}: {mid_price}")
            return mid_price
        close_price = ticker.get('close')
        if close_price is not None and close_price > 0:
             price = Decimal(str(close_price))
             logger.debug(f"Fetched close price for {symbol}: {price}")
             return price
        logger.warning(f"Could not determine a valid price (last, mid, close) for {symbol}. Ticker data: {ticker}")
        return None
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"CCXT error fetching price for {symbol}: {e}")
        raise
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error converting price data for {symbol}: {e}. Ticker: {ticker}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error fetching price for {symbol}: {e}", exc_info=True)
        raise

def optimize_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Optimizes DataFrame memory usage by downcasting numeric types."""
    if df.empty: return df
    try:
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and df[col].dtype != 'float32':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        if 'volume' in df.columns and df['volume'].dtype != 'float32':
            max_vol = df['volume'].max()
            if pd.notna(max_vol) and max_vol < np.finfo(np.float32).max:
                 df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float32')
            else:
                 df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float64')
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        # logger.debug(f"DataFrame memory optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not optimize DataFrame memory usage: {e}")
    return df

@ccxt_retry_decorator
def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV kline data from the exchange with retries."""
    logger.debug(f"Fetching {limit} klines for {symbol} ({timeframe})...")
    ohlcv = []
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            logger.warning(f"No kline data returned from exchange for {symbol} ({timeframe}).")
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        except ValueError as e:
            logger.error(f"Error converting timestamp for {symbol}: {e}. First few timestamps: {df['timestamp'].head().tolist()}")
            return pd.DataFrame()
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
             logger.warning(f"DataFrame empty after timestamp conversion for {symbol} ({timeframe}).")
             return df
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > 0]
        df.sort_index(inplace=True)
        df = optimize_dataframe(df.copy(), logger)
        if df.empty:
            logger.warning(f"DataFrame empty after processing for {symbol} ({timeframe}). Check fetched data quality.")
            return df
        if len(df) > MAX_DF_LEN:
             df = df.iloc[-MAX_DF_LEN:]
        logger.info(f"Fetched and processed {len(df)} klines for {symbol} ({timeframe}). Last timestamp: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'}")
        return df
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"CCXT error fetching klines for {symbol} ({timeframe}): {e}")
        raise
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error processing kline data for {symbol} ({timeframe}): {e}. Data sample: {ohlcv[:5] if ohlcv else 'N/A'}")
         return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching klines for {symbol} ({timeframe}): {e}", exc_info=True)
        raise

def safe_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely converts a value to Decimal, returning default on error."""
    if value is None: return default
    try:
        if isinstance(value, str) and 'e' in value.lower(): pass
        elif isinstance(value, float) and (np.isinf(value) or np.isnan(value)): return default
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError): return default

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information, using cache if available."""
    if symbol in market_info_cache:
        return market_info_cache[symbol]
    logger.debug(f"Fetching market info for {symbol} from exchange...")
    market = None
    try:
        market = exchange.market(symbol)
        if not market:
            logger.error(f"Market info not found for symbol '{symbol}' on {exchange.id}.")
            return None
        is_contract = market.get('contract', False) or market.get('type') in ['swap', 'future'] or market.get('linear', False) or market.get('inverse', False)
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        cost_limits = limits.get('cost', {})
        precision = market.get('precision', {})
        market_details = {
            'id': market.get('id'), 'symbol': market.get('symbol'), 'base': market.get('base'), 'quote': market.get('quote'),
            'active': market.get('active', False), 'type': market.get('type'), 'linear': market.get('linear'), 'inverse': market.get('inverse'),
            'contract': market.get('contract', False), 'contractSize': safe_decimal(market.get('contractSize', '1'), default=Decimal('1')),
            'is_contract': is_contract,
            'limits': {'amount': {'min': safe_decimal(amount_limits.get('min')), 'max': safe_decimal(amount_limits.get('max'))},
                       'price': {'min': safe_decimal(price_limits.get('min')), 'max': safe_decimal(price_limits.get('max'))},
                       'cost': {'min': safe_decimal(cost_limits.get('min')), 'max': safe_decimal(cost_limits.get('max'))}},
            'precision': {'amount': safe_decimal(precision.get('amount')), 'price': safe_decimal(precision.get('price'))},
            'taker': safe_decimal(market.get('taker'), default=Decimal('0.0006')),
            'maker': safe_decimal(market.get('maker'), default=Decimal('0.0001')),
            'info': market.get('info', {})
        }
        if market_details['precision']['amount'] is None or market_details['precision']['amount'] <= 0:
             logger.warning(f"Amount precision (tick size) for {symbol} is invalid or missing: {market_details['precision']['amount']}. Sizing/formatting may fail.")
        if market_details['precision']['price'] is None or market_details['precision']['price'] <= 0:
             logger.warning(f"Price precision (tick size) for {symbol} is invalid or missing: {market_details['precision']['price']}. Order placement/formatting may fail.")
        if market_details['limits']['amount']['min'] is None:
             logger.warning(f"Minimum amount limit for {symbol} is missing. Sizing might use 0.")
        market_info_cache[symbol] = market_details
        logger.info(f"Cached market info for {symbol}: Type={market_details['type']}, Contract={market_details['is_contract']}, Active={market_details['active']}")
        logger.debug(f"Limits: {market_details['limits']}")
        logger.debug(f"Precision (Tick Sizes): {market_details['precision']}")
        return market_details
    except ccxt.BadSymbol:
        logger.error(f"Symbol '{symbol}' not found on {exchange.id}.")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
         logger.error(f"CCXT error getting market info for {symbol}: {e}")
         return None
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error processing market data for {symbol}: {e}. Market data: {market if market else 'N/A'}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error getting market info for {symbol}: {e}", exc_info=True)
        return None

@ccxt_retry_decorator
def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency."""
    logger.debug(f"Fetching available balance for {currency}...")
    balance_info = {}
    try:
        params = {'accountType': 'CONTRACT'} # Adjust as needed
        balance_info = exchange.fetch_balance(params=params)
        free_balance = balance_info.get('free', {}).get(currency)
        currency_data = balance_info.get(currency, {})
        available = currency_data.get('free')
        total_balance = balance_info.get('total', {}).get(currency)
        balance_value = None
        if free_balance is not None: balance_value = free_balance
        elif available is not None: balance_value = available
        elif total_balance is not None:
             logger.warning(f"Using 'total' balance for {currency} as 'free'/'available' not found.")
             balance_value = total_balance
        else:
             logger.error(f"Could not find balance information for {currency} in response.")
             logger.debug(f"Full Balance Info: {balance_info}")
             return None
        balance = safe_decimal(balance_value)
        if balance is None:
            logger.error(f"Failed to convert balance value '{balance_value}' to Decimal for {currency}.")
            return None
        if balance < 0:
             logger.warning(f"Reported available balance for {currency} is negative ({balance}). Treating as 0.")
             return Decimal('0')
        logger.info(f"Available {currency} balance: {balance:.4f}")
        return balance
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"CCXT error fetching balance for {currency}: {e}")
        raise
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error converting balance data for {currency}: {e}. Balance Info: {balance_info}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error fetching balance for {currency}: {e}", exc_info=True)
        raise

@ccxt_retry_decorator
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Fetches the open position for a specific symbol using fetch_positions."""
    logger.debug(f"Checking for open position for {symbol}...")
    positions = []
    try:
        positions = exchange.fetch_positions([symbol])
        size_tolerance = Decimal('1e-9')
        open_positions = []
        for pos in positions:
            size_contracts = safe_decimal(pos.get('contracts'), default=Decimal('0'))
            size_info = safe_decimal(pos.get('info', {}).get('size'), default=Decimal('0'))
            pos_value = safe_decimal(pos.get('info', {}).get('positionValue'), default=Decimal('0'))
            if abs(size_contracts) > size_tolerance or abs(size_info) > size_tolerance or abs(pos_value) > size_tolerance:
                 open_positions.append(pos)
        if not open_positions:
            logger.info(f"No active position found for {symbol}.")
            return None
        if len(open_positions) > 1:
            logger.warning(f"Multiple ({len(open_positions)}) open positions found for {symbol}. Using first. Ensure One-Way mode.")
            for i, p in enumerate(open_positions): logger.debug(f"Position {i+1} Info: {p.get('info')}")
        pos = open_positions[0]
        pos_info = pos.get('info', {})
        size = safe_decimal(pos.get('contracts')) or safe_decimal(pos_info.get('size'))
        if size is None:
            logger.error(f"Could not determine position size for {symbol}. Position data: {pos}")
            return None
        entry_price = safe_decimal(pos.get('entryPrice')) or safe_decimal(pos_info.get('avgPrice')) or safe_decimal(pos_info.get('entryPrice'))
        if entry_price is None or entry_price <= 0:
            logger.error(f"Could not determine valid entry price for {symbol}. Position data: {pos}")
            entry_price = Decimal('0')
        side = 'long' if size > 0 else 'short'
        mark_price = safe_decimal(pos.get('markPrice')) or safe_decimal(pos_info.get('markPrice'))
        liq_price = safe_decimal(pos.get('liquidationPrice')) or safe_decimal(pos_info.get('liqPrice'))
        unrealized_pnl = safe_decimal(pos.get('unrealizedPnl')) or safe_decimal(pos_info.get('unrealisedPnl'))
        leverage = safe_decimal(pos.get('leverage')) or safe_decimal(pos_info.get('leverage'))
        timestamp_ms = pos.get('timestamp')
        datetime_str = pos.get('datetime')
        created_time_ms_str = pos_info.get('createdTime')
        entry_timestamp = None
        if created_time_ms_str:
            try: entry_timestamp = int(created_time_ms_str)
            except (ValueError, TypeError): logger.warning(f"Could not parse position createdTime string: {created_time_ms_str}")
        elif timestamp_ms: entry_timestamp = timestamp_ms
        elif datetime_str:
             try:
                 dt_obj = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                 entry_timestamp = int(dt_obj.timestamp() * 1000)
             except (TypeError, ValueError): logger.warning(f"Could not parse position datetime string: {datetime_str}")
        sl_price = safe_decimal(pos_info.get('stopLoss')) or safe_decimal(pos.get('stopLossPrice')) or safe_decimal(pos_info.get('slPrice')) or safe_decimal(pos_info.get('stop_loss'))
        tp_price = safe_decimal(pos_info.get('takeProfit')) or safe_decimal(pos.get('takeProfitPrice')) or safe_decimal(pos_info.get('tpPrice')) or safe_decimal(pos_info.get('take_profit'))
        tsl_distance = safe_decimal(pos_info.get('trailingStop'))
        tsl_activation_price = safe_decimal(pos_info.get('activePrice'))
        position_details = {
            'symbol': pos.get('symbol', symbol), 'side': side, 'contracts': size, 'entryPrice': entry_price,
            'markPrice': mark_price, 'liquidationPrice': liq_price, 'unrealizedPnl': unrealized_pnl, 'leverage': leverage,
            'entryTimestamp': entry_timestamp,
            'stopLossPrice': sl_price if sl_price and sl_price > 0 else None,
            'takeProfitPrice': tp_price if tp_price and tp_price > 0 else None,
            'tslDistance': tsl_distance if tsl_distance and tsl_distance > 0 else None,
            'tslActivationPrice': tsl_activation_price if tsl_activation_price and tsl_activation_price > 0 else None,
            'info': pos_info
        }
        log_sl = f"{position_details['stopLossPrice']:.4f}" if position_details['stopLossPrice'] else 'None'
        log_tp = f"{position_details['takeProfitPrice']:.4f}" if position_details['takeProfitPrice'] else 'None'
        log_tsl_dist = f"{position_details['tslDistance']:.4f}" if position_details['tslDistance'] else 'None'
        log_tsl_act = f"{position_details['tslActivationPrice']:.4f}" if position_details['tslActivationPrice'] else 'None'
        logger.info(f"Active {position_details['side'].upper()} position found for {symbol}: Size={position_details['contracts']}, Entry={position_details['entryPrice']:.4f}, SL={log_sl}, TP={log_tp}, TSL Dist={log_tsl_dist}, TSL Act={log_tsl_act}")
        logger.debug(f"Full position details: {position_details}")
        return position_details
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"CCXT error fetching position for {symbol}: {e}")
        raise
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error processing position data for {symbol}: {e}. Positions data: {positions}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error fetching position for {symbol}: {e}", exc_info=True)
        raise

@ccxt_retry_decorator
def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a contract symbol."""
    if not market_info.get('is_contract', False):
        logger.debug(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True
    if leverage <= 0:
        logger.error(f"Invalid leverage value ({leverage}) for {symbol}. Must be positive.")
        return False
    logger.debug(f"Setting leverage to {leverage}x for {symbol}...")
    response = None
    try:
        params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
        if 'category' not in params: params['category'] = 'linear' if market_info.get('linear', True) else 'inverse'
        logger.debug(f"Calling set_leverage for {symbol} with leverage={leverage}, params={params}")
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        logger.info(f"Leverage set to {leverage}x for {symbol}.")
        logger.debug(f"Leverage Response: {response}")
        return True
    except ccxt.ExchangeError as e:
        error_str = str(e).lower()
        if "leverage not modified" in error_str or "leverage same" in error_str or "leverage not change" in error_str or "110025" in str(e) or "30086" in str(e):
            logger.info(f"Leverage for {symbol} already set to {leverage}x or no change needed.")
            return True
        elif "max leverage" in error_str or "leverage too high" in error_str:
             logger.error(f"Failed to set leverage for {symbol}: Requested leverage {leverage}x exceeds maximum allowed. Error: {e}")
             return False
        else:
            logger.error(f"Exchange error setting leverage for {symbol} to {leverage}x: {e}")
            logger.debug(f"Leverage setting request params: leverage={leverage}, symbol={symbol}, params={params if 'params' in locals() else 'N/A'}")
            logger.debug(f"Leverage setting response (if any): {response}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error setting leverage for {symbol} to {leverage}x: {e}", exc_info=True)
        raise

def format_value_by_tick_size(value: Decimal, tick_size: Decimal, rounding_mode: str = ROUND_DOWN) -> Decimal:
    """Formats a Decimal value according to a specific tick size using quantization."""
    if tick_size <= 0: raise ValueError("Tick size must be positive")
    quantized_value = (value / tick_size).quantize(Decimal('1'), rounding=rounding_mode) * tick_size
    return quantized_value

def format_value_for_exchange(value: Decimal, precision_type: str, market_info: Dict, logger: logging.Logger, rounding_mode: str = ROUND_DOWN) -> Optional[Decimal]:
    """Formats a Decimal value according to exchange precision (tick size)."""
    symbol = market_info.get('symbol', 'UNKNOWN')
    precision_data = market_info.get('precision')
    if precision_data is None:
        logger.error(f"Precision info missing for {symbol}. Cannot format {precision_type}.")
        return None
    tick_size = precision_data.get(precision_type)
    if tick_size is None or not isinstance(tick_size, Decimal) or tick_size <= 0:
        logger.error(f"Invalid or missing '{precision_type}' precision (tick size) for {symbol}: {tick_size}. Cannot format value.")
        return None
    try:
        formatted_value = format_value_by_tick_size(value, tick_size, rounding_mode)
        # logger.debug(f"Formatted {precision_type} for {symbol}: Original={value}, Tick={tick_size}, Rounded={formatted_value} (Mode: {rounding_mode})")
        return formatted_value
    except (ValueError, InvalidOperation, TypeError) as e:
        logger.error(f"Error formatting {precision_type} for {symbol} using manual quantization: {e}. Value: {value}, Tick Size: {tick_size}")
        return None
    except Exception as e:
         logger.error(f"Unexpected error during formatting for {symbol}: {e}", exc_info=True)
         return None

def calculate_position_size(
    balance: Decimal, risk_per_trade: float, entry_price: Decimal, stop_loss_price: Decimal,
    market_info: Dict, config: Dict[str, Any], current_atr: Optional[Decimal], logger: logging.Logger
) -> Optional[Decimal]:
    """Calculates the position size in contracts/base currency units."""
    symbol = market_info.get('symbol', 'UNKNOWN')
    lg = logger
    if balance is None or balance <= 0: lg.error(f"Invalid balance ({balance}) for sizing {symbol}."); return None
    if not (0 < risk_per_trade < 1): lg.error(f"Invalid risk_per_trade ({risk_per_trade}) for sizing {symbol}."); return None
    if stop_loss_price is None or stop_loss_price <= 0: lg.error(f"Invalid stop_loss_price ({stop_loss_price}) for sizing {symbol}."); return None
    if entry_price is None or entry_price <= 0: lg.error(f"Invalid entry_price ({entry_price}) for sizing {symbol}."); return None
    if entry_price == stop_loss_price: lg.error(f"Entry and SL price cannot be the same ({entry_price}) for {symbol}."); return None

    risk_per_contract_quote = Decimal('0')
    raw_size = Decimal('0')
    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_price = abs(entry_price - stop_loss_price)
        if sl_distance_price <= 0: lg.error(f"SL distance is zero or negative for {symbol}."); return None

        contract_size_base = market_info.get('contractSize', Decimal('1'))
        if contract_size_base <= 0: lg.error(f"Invalid contract size ({contract_size_base}) for {symbol}."); return None

        risk_per_contract_quote = sl_distance_price * contract_size_base
        if risk_per_contract_quote <= 0: lg.error(f"Calculated risk per contract is zero/negative ({risk_per_contract_quote}) for {symbol}."); return None

        # [S20] ATR-Based Position Sizing Adjustment
        atr_risk_factor = Decimal('1.0')
        if config.get('atr_position_sizing', False) and current_atr is not None and current_atr > 0 and entry_price > 0:
            relative_atr = current_atr / entry_price
            if relative_atr > Decimal('0.02'): atr_risk_factor = Decimal('0.9')
            elif relative_atr < Decimal('0.005'): atr_risk_factor = Decimal('1.1')
            lg.info(f"[S20] ATR Position Sizing: Relative ATR={relative_atr:.4f}, Risk Factor={atr_risk_factor:.2f}")
            risk_amount_quote *= atr_risk_factor # Adjust risk amount based on vol

        raw_size = risk_amount_quote / risk_per_contract_quote

        min_amount = market_info.get('limits', {}).get('amount', {}).get('min')
        max_amount = market_info.get('limits', {}).get('amount', {}).get('max')

        if min_amount is not None and raw_size < min_amount:
            lg.warning(f"Calculated size {raw_size:.8f} < min ({min_amount}) for {symbol}. Adjusting to min.")
            raw_size = min_amount
            actual_risk = raw_size * risk_per_contract_quote
            actual_risk_percent = (actual_risk / balance) * 100 if balance > 0 else Decimal('0')
            lg.warning(f"Actual risk for min size: {actual_risk:.4f} {QUOTE_CURRENCY} ({actual_risk_percent:.2f}%)")
            max_allowed_risk = balance * Decimal(str(risk_per_trade)) * Decimal('1.5')
            if actual_risk > max_allowed_risk:
                 lg.error(f"Risk with min size ({actual_risk:.4f}) exceeds target risk ({risk_amount_quote:.4f}). Aborting trade.")
                 return None

        if max_amount is not None and raw_size > max_amount:
             lg.warning(f"Calculated size {raw_size:.8f} > max ({max_amount}) for {symbol}. Capping at max.")
             raw_size = max_amount

        formatted_size = format_value_for_exchange(raw_size, 'amount', market_info, lg, ROUND_DOWN)

        if formatted_size is None: lg.error(f"Failed to format size {raw_size:.8f} for {symbol}."); return None
        if formatted_size <= 0: lg.error(f"Formatted size zero/negative ({formatted_size}) for {symbol}. Raw: {raw_size:.8f}, Min: {min_amount}"); return None

        if min_amount is not None and formatted_size < min_amount:
             lg.error(f"Formatted size {formatted_size} < min {min_amount} for {symbol} after format. Check precision.")
             formatted_min = format_value_for_exchange(min_amount, 'amount', market_info, lg, ROUND_DOWN)
             if formatted_min and formatted_min == min_amount:
                  lg.warning(f"Using min order size {min_amount} as calc size too small after format.")
                  formatted_size = min_amount
                  actual_risk = formatted_size * risk_per_contract_quote
                  actual_risk_percent = (actual_risk / balance) * 100 if balance > 0 else Decimal('0')
                  lg.warning(f"Final risk check for min size: {actual_risk:.4f} {QUOTE_CURRENCY} ({actual_risk_percent:.2f}%)")
                  max_allowed_risk = balance * Decimal(str(risk_per_trade)) * Decimal('1.5')
                  if actual_risk > max_allowed_risk:
                      lg.error(f"Risk with min size ({actual_risk:.4f}) still exceeds target risk ({risk_amount_quote:.4f}). Aborting trade.")
                      return None
             else:
                  lg.error(f"Cannot determine valid size meeting min requirements ({min_amount}). Min size formatting failed ({formatted_min}).")
                  return None

        estimated_cost = formatted_size * entry_price
        min_cost = market_info.get('limits', {}).get('cost', {}).get('min')
        max_cost = market_info.get('limits', {}).get('cost', {}).get('max')

        if min_cost is not None and estimated_cost < min_cost: lg.error(f"Estimated cost ({estimated_cost:.4f}) < min cost ({min_cost}). Cannot trade."); return None
        if max_cost is not None and estimated_cost > max_cost: lg.error(f"Estimated cost ({estimated_cost:.4f}) > max cost ({max_cost}). Cannot trade."); return None

        lg.info(f"Calculated Position Size for {symbol}: {formatted_size} {market_info.get('base', 'Units')}")
        lg.info(f" -> Target Risk: {risk_amount_quote:.4f} {QUOTE_CURRENCY} (Factor: {atr_risk_factor:.2f})")
        lg.info(f" -> SL Distance: {sl_distance_price:.4f}")
        lg.info(f" -> Estimated Cost: {estimated_cost:.4f} {QUOTE_CURRENCY}")
        return formatted_size
    except ZeroDivisionError: lg.error(f"Division by zero during size calc for {symbol}. Risk per contract: {risk_per_contract_quote}"); return None
    except (InvalidOperation, TypeError, ValueError) as e: lg.error(f"Decimal/conversion error during size calc for {symbol}: {e}"); return None
    except Exception as e: lg.error(f"Unexpected error calculating size for {symbol}: {e}", exc_info=True); return None

@ccxt_retry_decorator
def place_trade(
    exchange: ccxt.Exchange, symbol: str, signal: str, position_size: Decimal,
    market_info: Dict, config: Dict[str, Any], logger: logging.Logger,
    reduce_only: bool = False, limit_price: Optional[Decimal] = None
) -> Optional[Dict]:
    """Places a market or limit order with retries and proper parameters."""
    lg = logger
    side = 'buy' if signal == "BUY" else 'sell'
    order_type = 'market'
    if not reduce_only: order_type = config.get('order_type', 'market')
    if limit_price is not None: order_type = 'limit'

    order = None
    try:
        amount_decimal = position_size
        try: amount_float = float(amount_decimal)
        except (TypeError, ValueError): lg.error(f"Invalid size type for float conversion: {amount_decimal}"); return None
        if amount_float <= 0: lg.error(f"Invalid size {amount_float} ({amount_decimal}) for {symbol}."); return None

        params = {'category': 'linear' if market_info.get('linear', True) else 'inverse', 'positionIdx': 0}
        if reduce_only:
            params['reduceOnly'] = True
            params['timeInForce'] = 'IOC' if order_type == 'market' else 'GTC'
        else: params['timeInForce'] = 'GTC'

        log_price = f"Price={limit_price}" if order_type == 'limit' else "Market Price"
        lg.info(f"Placing {order_type.upper()} {side.upper()} order: {amount_float} {market_info.get('base', '')} on {symbol}. {log_price}. ReduceOnly={reduce_only}. Params={params}")

        if order_type == 'limit':
            if limit_price is None or limit_price <= 0: lg.error(f"Limit price missing/invalid for limit order on {symbol}."); return None
            limit_rounding = ROUND_DOWN if side == 'buy' else ROUND_UP
            formatted_limit_price = format_value_for_exchange(limit_price, 'price', market_info, lg, limit_rounding)
            if formatted_limit_price is None: lg.error(f"Failed to format limit price {limit_price} for {symbol}."); return None
            try: price_float = float(formatted_limit_price)
            except (TypeError, ValueError): lg.error(f"Invalid limit price type for float conversion: {formatted_limit_price}"); return None
            lg.info(f"Formatted Limit Price: {price_float} ({formatted_limit_price})")
            order = exchange.create_order(symbol=symbol, type='limit', side=side, amount=amount_float, price=price_float, params=params)
        else: # Market order
            order = exchange.create_order(symbol=symbol, type='market', side=side, amount=amount_float, price=None, params=params)

        if order:
            order_id = order.get('id')
            order_status = order.get('status', 'unknown')
            filled_amount = order.get('filled', 0.0)
            avg_fill_price = order.get('average')
            log_color = NEON_GREEN if order_status in ['closed', 'filled'] else NEON_YELLOW if order_status == 'open' else NEON_RED
            lg.info(f"{log_color}{order_type.capitalize()} {side} order {order_status} for {symbol}: ID={order_id}, Amount={order.get('amount')}, Filled={filled_amount}, AvgPrice={avg_fill_price or 'N/A'}{RESET}")
            lg.debug(f"Full Order Response: {order}")
            return order
        else:
            lg.error(f"Failed to place {order_type} order for {symbol}. Exchange response empty/invalid.")
            return None
    except ccxt.InsufficientFunds as e: lg.error(f"{NEON_RED}Insufficient funds for {side} order {symbol}: {e}{RESET}"); fetch_balance(exchange, QUOTE_CURRENCY, lg); return None
    except ccxt.InvalidOrder as e: lg.error(f"{NEON_RED}Invalid order params for {symbol}: {e}{RESET}"); lg.error(f" -> Size: {position_size}, Price: {limit_price}, Type: {order_type}, Reduce: {reduce_only}"); return None
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        lg.error(f"CCXT error placing {side} order for {symbol}: {e}")
        if "margin" in str(e).lower() or "insufficient balance" in str(e).lower(): lg.warning("Order failed possibly due to margin/balance.")
        elif "order cost" in str(e).lower(): lg.warning("Order failed possibly due to cost limits.")
        lg.debug(f"Failed Order Details: Size={position_size}, Price={limit_price}, Type={order_type}, Reduce={reduce_only}")
        lg.debug(f"Failed Order Response: {order}")
        raise
    except (InvalidOperation, TypeError, ValueError) as e: lg.error(f"Data conversion error placing trade for {symbol}: {e}"); return None
    except Exception as e: lg.error(f"Unexpected error placing {side} order for {symbol}: {e}", exc_info=True); raise

@ccxt_retry_decorator
def _set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None,
    position_idx: int = 0
) -> bool:
    """Sets or updates SL, TP, and/or TSL for an existing position using Bybit's API."""
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol}: Not a contract market.")
        return False

    category = 'linear' if market_info.get('linear', True) else 'inverse'
    params = {'category': category, 'symbol': market_info['id'], 'positionIdx': position_idx}
    log_parts = []
    something_to_set = False
    last_price_known = market_info.get('last_price') # Use last known price for better rounding heuristics

    # Format and add protection levels (setting to '0' cancels on Bybit)
    if stop_loss_price is not None:
        if stop_loss_price > 0:
            # Round SL away from potential position
            sl_rounding = ROUND_DOWN if last_price_known and stop_loss_price < last_price_known else ROUND_UP
            sl_formatted = format_value_for_exchange(stop_loss_price, 'price', market_info, lg, sl_rounding)
            if sl_formatted: params['stopLoss'] = str(sl_formatted); log_parts.append(f"SL={sl_formatted}"); something_to_set = True
            else: lg.error(f"Invalid SL price format for {symbol}: {stop_loss_price}.")
        else: params['stopLoss'] = '0'; log_parts.append("SL=Cancel"); something_to_set = True

    if take_profit_price is not None:
        if take_profit_price > 0:
            # Round TP towards potential position target
            tp_rounding = ROUND_UP if last_price_known and take_profit_price > last_price_known else ROUND_DOWN
            tp_formatted = format_value_for_exchange(take_profit_price, 'price', market_info, lg, tp_rounding)
            if tp_formatted: params['takeProfit'] = str(tp_formatted); log_parts.append(f"TP={tp_formatted}"); something_to_set = True
            else: lg.error(f"Invalid TP price format for {symbol}: {take_profit_price}.")
        else: params['takeProfit'] = '0'; log_parts.append("TP=Cancel"); something_to_set = True

    if trailing_stop_distance is not None:
        if trailing_stop_distance > 0:
            # Format distance using price precision, usually round down conservatively
            tsl_dist_formatted = format_value_for_exchange(trailing_stop_distance, 'price', market_info, lg, ROUND_DOWN)
            if tsl_dist_formatted and tsl_dist_formatted > 0:
                params['trailingStop'] = str(tsl_dist_formatted)
                log_parts.append(f"TSL_Dist={tsl_dist_formatted}")
                something_to_set = True
                if tsl_activation_price is not None and tsl_activation_price > 0:
                    act_rounding = ROUND_UP if last_price_known and tsl_activation_price > last_price_known else ROUND_DOWN
                    tsl_act_formatted = format_value_for_exchange(tsl_activation_price, 'price', market_info, lg, act_rounding)
                    if tsl_act_formatted: params['activePrice'] = str(tsl_act_formatted); log_parts.append(f"TSL_Act={tsl_act_formatted}")
                    else: lg.warning(f"Invalid TSL activation price format for {symbol}: {tsl_activation_price}. Setting TSL dist without activation.")
            else: lg.error(f"Invalid TSL distance format/zero value for {symbol}: {trailing_stop_distance}. Cannot set TSL.")
        else: params['trailingStop'] = '0'; log_parts.append("TSL=Cancel"); something_to_set = True

    if not something_to_set:
        lg.info(f"No valid protection levels provided or format failed for {symbol}. No API call.")
        return False

    response = None
    try:
        lg.info(f"Setting/Updating protection for {symbol}: {', '.join(log_parts)}")
        lg.debug(f"Calling v5/position/trading-stop with params: {params}")
        method_name_v5 = 'private_post_v5_position_trading_stop'
        method_name_unified = 'private_post_position_trading_stop'
        if hasattr(exchange, method_name_v5): response = getattr(exchange, method_name_v5)(params)
        elif hasattr(exchange, method_name_unified): response = getattr(exchange, method_name_unified)(params)
        else: lg.error(f"Cannot find CCXT method ({method_name_v5} or {method_name_unified}) for Bybit set trading stop."); return False

        lg.debug(f"Set protection response for {symbol}: {response}")
        if isinstance(response, dict) and response.get('retCode') == 0:
             lg.info(f"{NEON_GREEN}Protection successfully set/updated for {symbol}.{RESET}")
             return True
        else:
             error_code = response.get('retCode', 'N/A'); error_msg = response.get('retMsg', 'Unknown error')
             lg.error(f"Failed to set protection for {symbol}. Exchange: Code={error_code}, Msg='{error_msg}'")
             lg.debug(f"Failed Params: {params}")
             # Add more specific error code handling if needed
             return False
    except ccxt.InvalidOrder as e: lg.error(f"Invalid order/params setting protection for {symbol}: {e}"); lg.debug(f"Params: {params}"); return False
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        lg.error(f"CCXT error setting protection for {symbol}: {e}")
        lg.debug(f"Params: {params}"); lg.debug(f"Response: {response}")
        if "order not found or too late" in str(e).lower() or "position status is not normal" in str(e).lower():
             lg.warning("Protection setting failed likely due to concurrent position close/fill.")
             return False
        raise
    except Exception as e: lg.error(f"Unexpected error setting protection for {symbol}: {e}", exc_info=True); lg.debug(f"Params: {params}"); raise

def set_trailing_stop_loss(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict,
    config: Dict[str, Any], logger: logging.Logger, current_atr: Optional[Decimal], current_price: Optional[Decimal]
) -> bool:
    """Calculates TSL parameters and calls _set_position_protection."""
    lg = logger
    protection_cfg = config.get("protection", {})
    if not protection_cfg.get("enable_trailing_stop", False): lg.debug("TSL disabled in config."); return False

    side = position_info.get('side')
    entry_price = position_info.get('entryPrice')
    if side not in ['long', 'short'] or entry_price is None or entry_price <= 0: lg.error(f"Invalid position info for TSL: Side={side}, Entry={entry_price}"); return False
    if current_price is None or current_price <= 0: lg.warning("Cannot calculate TSL params: Invalid current price."); return False

    # [S9] Check TSL Activation Delay
    tsl_activation_delay_atr = safe_decimal(protection_cfg.get("tsl_activation_delay_atr", 0.5), default=Decimal('0.5'))
    if tsl_activation_delay_atr > 0:
        if current_atr is None or current_atr <= 0: lg.warning("Cannot check TSL activation delay: ATR unavailable.")
        else:
            profit = (current_price - entry_price) if side == 'long' else (entry_price - current_price)
            required_profit_for_activation = current_atr * tsl_activation_delay_atr
            if profit < required_profit_for_activation:
                lg.info(f"[S9] TSL activation delayed: Profit {profit:.4f} < Required {required_profit_for_activation:.4f}")
                return False # Not enough profit yet

    distance: Optional[Decimal] = None
    activation_price: Optional[Decimal] = None

    # [S12] ATR-Based Trailing Stop
    use_atr_tsl = protection_cfg.get("use_atr_trailing_stop", True)
    if use_atr_tsl and current_atr is not None and current_atr > 0:
        trailing_atr_multiple = safe_decimal(protection_cfg.get("trailing_stop_atr_multiple", 1.5), default=Decimal('1.5'))
        distance = current_atr * trailing_atr_multiple
        activation_pct = safe_decimal(protection_cfg.get("trailing_stop_activation_percentage", 0.003), default=Decimal('0.003'))
        activation_offset = entry_price * activation_pct
        activation_price = entry_price + activation_offset if side == 'long' else entry_price - activation_offset
        lg.info(f"[S12] Using ATR TSL: Dist={distance:.4f}, Act={activation_price:.4f}")
    else: # Fixed Percentage Callback Rate TSL
        if use_atr_tsl: lg.warning("ATR TSL enabled but ATR failed. Falling back to percentage TSL.")
        callback_rate = safe_decimal(protection_cfg.get("trailing_stop_callback_rate", 0.005), default=Decimal('0.005'))
        activation_pct = safe_decimal(protection_cfg.get("trailing_stop_activation_percentage", 0.003), default=Decimal('0.003'))
        distance = current_price * callback_rate # Base distance on current price for reactivity
        activation_offset = entry_price * activation_pct
        activation_price = entry_price + activation_offset if side == 'long' else entry_price - activation_offset
        lg.info(f"Using Percentage TSL: Dist={distance:.4f}, Act={activation_price:.4f}")

    # [S22] Dynamic TSL Distance Adjustment
    if protection_cfg.get("dynamic_tsl_callback", True) and distance is not None and current_atr is not None and current_atr > 0:
        profit = (current_price - entry_price) if side == 'long' else (entry_price - current_price)
        profit_atr = profit / current_atr if current_atr > 0 else Decimal('0')
        if profit_atr > Decimal('2.0'): # Example: Tighten if profit > 2x ATR
            tighten_factor = Decimal('0.75')
            original_distance = distance
            distance *= tighten_factor
            lg.info(f"[S22] Dynamic TSL: Profit {profit_atr:.2f}x ATR > 2.0. Tightening TSL dist from {original_distance:.4f} to {distance:.4f}")

    if distance is None or distance <= 0: lg.error(f"Calculated invalid TSL distance ({distance}). Cannot set TSL."); return False
    if activation_price is None or activation_price <= 0: lg.error(f"Calculated invalid TSL activation price ({activation_price}). Cannot set TSL."); return False

    # Validate & Adjust TSL activation price vs current price
    min_tick = market_info.get('precision', {}).get('price', Decimal('0.0001'))
    if min_tick is None or min_tick <= 0: lg.warning(f"Cannot validate TSL activation price: Invalid price tick size ({min_tick}).")
    elif side == 'long' and activation_price <= current_price:
        lg.warning(f"TSL Act Price ({activation_price:.4f}) for LONG <= current ({current_price:.4f}). Adjusting slightly above.")
        activation_price = current_price + min_tick
    elif side == 'short' and activation_price >= current_price:
        lg.warning(f"TSL Act Price ({activation_price:.4f}) for SHORT >= current ({current_price:.4f}). Adjusting slightly below.")
        activation_price = current_price - min_tick

    # Preserve existing SL/TP when setting TSL
    current_sl = position_info.get('stopLossPrice')
    current_tp = position_info.get('takeProfitPrice')
    position_idx = position_info.get('info', {}).get('positionIdx', 0)

    lg.debug(f"Calling _set_position_protection with: SL={current_sl}, TP={current_tp}, TSL_Dist={distance}, TSL_Act={activation_price}")
    market_info['last_price'] = current_price # Update last known price for rounding
    return _set_position_protection(
        exchange, symbol, market_info, lg,
        stop_loss_price=current_sl, take_profit_price=current_tp,
        trailing_stop_distance=distance, tsl_activation_price=activation_price,
        position_idx=position_idx
    )

# --- Strategy Implementation ---

class HigherTimeframeAnalysis(TypedDict):
    """[S19] Structure for higher timeframe analysis results."""
    current_trend_up: Optional[bool]

class StrategyAnalysisResults(TypedDict):
    """Structure for returning analysis results from the strategy."""
    dataframe: pd.DataFrame
    last_close: Optional[Decimal]
    current_trend_up: Optional[bool]
    trend_just_changed: bool
    active_bull_boxes: List[Dict]
    active_bear_boxes: List[Dict]
    vol_norm_int: Optional[int] # [S1]
    atr: Optional[Decimal]
    upper_band: Optional[Decimal]
    lower_band: Optional[Decimal]
    adx: Optional[Decimal]           # [S3]
    rsi: Optional[Decimal]           # [S7]
    vwap: Optional[Decimal]          # [S15]
    roc: Optional[Decimal]           # [S23]
    higher_tf_analysis: Optional[HigherTimeframeAnalysis] # [S19]

class VolumaticOBStrategy:
    """Implements the Volumatic Trend + Pivot Order Block strategy with enhancements."""
    def __init__(self, config: Dict[str, Any], market_info: Dict, logger: logging.Logger, exchange: Optional[ccxt.Exchange] = None):
        self.config = config
        self.logger = logger
        self.market_info = market_info
        self.exchange = exchange # [S19] Required for Multi-TF Confirm
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        try:
            self.strategy_params = StrategyParams(**config.get("strategy_params", {}))
            self.protection_config = ProtectionConfig(**config.get("protection", {}))
        except ValidationError as e:
             self.logger.error(f"Strategy/Protection Pydantic validation failed: {e}. Using defaults.")
             self.strategy_params = StrategyParams()
             self.protection_config = ProtectionConfig()
        self.bull_boxes: List[Dict[str, Any]] = []
        self.bear_boxes: List[Dict[str, Any]] = []
        self.min_data_len = self._calculate_min_data_length()
        self.logger.debug(f"Minimum data length required: {self.min_data_len}")

    def _calculate_min_data_length(self) -> int:
        """Determines minimum candles required based on indicator lookbacks."""
        lookbacks = [
            self.strategy_params.vt_length, self.strategy_params.vt_atr_period,
            self.strategy_params.vt_vol_ema_length, # [S1] Volume lookback
            self.strategy_params.ph_left + self.strategy_params.ph_right,
            self.strategy_params.pl_left + self.strategy_params.pl_right,
            14, # [S3] ADX default lookback
            14, # [S7] RSI default lookback
            self.strategy_params.roc_length, # [S23] ROC lookback
            20, # pandas_ta default VWAP window
            100, # [S2] Window for ATR percentile
            20   # [S5] Window for ATR std dev
        ]
        pivot_buffer = max(self.strategy_params.ph_right, self.strategy_params.pl_right) + 5
        buffer = max(50, pivot_buffer)
        return max(lookbacks) + buffer

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all necessary indicators."""
        if df.empty or len(df) < 5: self.logger.warning("DF too small for indicators."); return df
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols): self.logger.error(f"Missing required columns: {required_cols}"); return df
        df_out = df.copy()

        try: # Core Volumatic
            df_out['atr'] = ta.atr(df_out['high'], df_out['low'], df_out['close'], length=self.strategy_params.vt_atr_period, fillna=False)
            df_out['vt_ema'] = ta.ema(df_out['close'], length=self.strategy_params.vt_length, fillna=False)
            df_out['trend_up'] = (df_out['close'].shift(1) >= df_out['vt_ema'].shift(1))
            df_out['trend_up'].fillna(method='bfill', inplace=True)
            df_out['trend_up'] = df_out['trend_up'].astype(bool)
            df_out['trend_changed'] = (df_out['trend_up'] != df_out['trend_up'].shift(1))
            df_out['trend_changed'].iloc[0] = False; df_out['trend_changed'].fillna(False, inplace=True)
            df_out['ema_at_change'] = np.where(df_out['trend_changed'], df_out['vt_ema'], np.nan)
            df_out['atr_at_change'] = np.where(df_out['trend_changed'], df_out['atr'], np.nan)
            df_out['ema_for_bands'] = df_out['ema_at_change'].ffill()
            df_out['atr_for_bands'] = df_out['atr_at_change'].ffill()
            atr_multiplier = safe_decimal(self.strategy_params.vt_atr_multiplier, Decimal('3.0'))
            ema_numeric = pd.to_numeric(df_out['ema_for_bands'], errors='coerce')
            atr_numeric = pd.to_numeric(df_out['atr_for_bands'], errors='coerce')
            df_out['upper_band'] = ema_numeric + (atr_numeric * atr_multiplier)
            df_out['lower_band'] = ema_numeric - (atr_numeric * atr_multiplier)
            # [S1] Volume Percentile Rank
            vol_norm_len = self.strategy_params.vt_vol_ema_length
            df_out['vol_norm'] = df_out['volume'].rolling(window=vol_norm_len, min_periods=max(1, vol_norm_len // 2)).rank(pct=True) * 100
            df_out['vol_norm'].fillna(0, inplace=True)
            df_out['vol_norm_int'] = df_out['vol_norm'].round().astype(int)
        except Exception as e: self.logger.error(f"Error calculating core Volumatic indicators: {e}", exc_info=True); return df

        try: # Enhancement Indicators
            # [S3] ADX
            adx_df = ta.adx(df_out['high'], df_out['low'], df_out['close'], length=14, fillna=False)
            df_out['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else np.nan
            # [S7] RSI
            df_out['rsi'] = ta.rsi(df_out['close'], length=14, fillna=False)
            # [S11] Candlestick Patterns
            if self.strategy_params.candlestick_confirm:
                try:
                    eng_df = ta.cdl_pattern(df_out['open'], df_out['high'], df_out['low'], df_out['close'], name="engulfing")
                    df_out['bull_engulfing'] = (eng_df['CDL_ENGULFING'] == 100) if eng_df is not None and 'CDL_ENGULFING' in eng_df.columns else False
                    df_out['bear_engulfing'] = (eng_df['CDL_ENGULFING'] == -100) if eng_df is not None and 'CDL_ENGULFING' in eng_df.columns else False
                    # Add other patterns here if needed
                except Exception as candle_e: self.logger.warning(f"[S11] Candlestick pattern calc failed: {candle_e}"); df_out['bull_engulfing'] = False; df_out['bear_engulfing'] = False
            # [S15] VWAP
            df_out['vwap'] = ta.vwap(df_out['high'], df_out['low'], df_out['close'], df_out['volume'], fillna=False)
            # [S23] ROC
            df_out['roc'] = ta.roc(df_out['close'], length=self.strategy_params.roc_length, fillna=False)
        except Exception as e: self.logger.warning(f"Error calculating enhancement indicators: {e}", exc_info=True)

        # Final Cleanup: Drop initial NaNs from required indicators
        required_indicators = ['vt_ema', 'atr', 'upper_band', 'lower_band', 'trend_up', 'vol_norm_int'] # Core + S1
        if self.strategy_params.adx_threshold > 0: required_indicators.append('adx')          # S3
        if self.strategy_params.rsi_confirm: required_indicators.append('rsi')                # S7
        if self.strategy_params.vwap_confirm: required_indicators.append('vwap')             # S15
        if self.strategy_params.momentum_confirm: required_indicators.append('roc')          # S23
        # S11 (candlestick) is checked directly later if enabled

        available_required = [ind for ind in required_indicators if ind in df_out.columns]
        if not available_required: self.logger.error("No key indicators calculated."); return df
        initial_len = len(df_out)
        df_out.dropna(subset=available_required, how='any', inplace=True)
        if initial_len - len(df_out) > 0: self.logger.debug(f"Dropped {initial_len - len(df_out)} initial NaN rows.")
        if df_out.empty: self.logger.warning("DataFrame empty after dropping NaNs.")
        return df_out

    def _identify_order_blocks(self, df: pd.DataFrame):
        """Identifies Pivot High/Low based Order Blocks and manages their state."""
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            self.logger.warning("Insufficient data/columns for OB calculation."); return
        min_pivot_len = max(self.strategy_params.ph_left + self.strategy_params.ph_right, self.strategy_params.pl_left + self.strategy_params.pl_right) + 2
        if len(df) < min_pivot_len: self.logger.warning(f"Not enough data ({len(df)}) for pivot calc (need ~{min_pivot_len})."); return

        try:
            high_series, low_series = (df['high'], df['low']) if self.strategy_params.ob_source == "Wicks" else (df[['open', 'close']].max(axis=1), df[['open', 'close']].min(axis=1))
        except KeyError: self.logger.error("Missing open/close for 'Bodys' OB source."); return

        try: # Calculate Pivots
            df['ph_signal'] = ta.pivot(high_series, left=self.strategy_params.ph_left, right=self.strategy_params.ph_right, high_low='high', fillna=False)
            df['pl_signal'] = ta.pivot(low_series, left=self.strategy_params.pl_left, right=self.strategy_params.pl_right, high_low='low', fillna=False)
        except Exception as e: self.logger.error(f"Error calculating pivots: {e}"); return

        potential_pivot_range = max(self.strategy_params.ph_right, self.strategy_params.pl_right) + 5
        start_iloc = max(0, len(df) - potential_pivot_range)
        recent_df = df.iloc[start_iloc:]
        current_atr = safe_decimal(df['atr'].iloc[-1]) if 'atr' in df.columns and not df['atr'].empty and pd.notna(df['atr'].iloc[-1]) else Decimal('0')
        new_bear_indices, new_bull_indices = set(), set()

        # Identify New OBs
        for signal_idx_dt in recent_df.index:
            is_ph_signal = pd.notna(recent_df.loc[signal_idx_dt, 'ph_signal'])
            is_pl_signal = pd.notna(recent_df.loc[signal_idx_dt, 'pl_signal'])

            # Bearish OB from Pivot High
            if is_ph_signal:
                try:
                    signal_idx_loc = df.index.get_loc(signal_idx_dt)
                    pivot_candle_loc = signal_idx_loc - self.strategy_params.ph_right
                    if pivot_candle_loc >= 0:
                        pivot_candle_idx_dt = df.index[pivot_candle_loc]
                        if pivot_candle_idx_dt not in new_bear_indices and not any(b['left_idx'] == pivot_candle_idx_dt for b in self.bear_boxes):
                            ob_candle = df.loc[pivot_candle_idx_dt]
                            ob_vol_norm = int(ob_candle.get('vol_norm_int', 0))
                            # [S17] Validate OB Volume
                            if self.strategy_params.validate_ob_volume and (ob_vol_norm is None or ob_vol_norm < (self.strategy_params.volume_threshold * 0.5)):
                                self.logger.debug(f"[S17] Skipping Bear OB {pivot_candle_idx_dt}: Low volume ({ob_vol_norm})")
                                continue
                            box_top = safe_decimal(ob_candle['high']) if self.strategy_params.ob_source == "Wicks" else safe_decimal(max(ob_candle['open'], ob_candle['close']))
                            box_bottom = safe_decimal(ob_candle['low']) if self.strategy_params.ob_source == "Wicks" else safe_decimal(min(ob_candle['open'], ob_candle['close']))
                            if box_top is None or box_bottom is None or box_top <= box_bottom: continue
                            # [S18] Adaptive OB Sizing
                            if self.strategy_params.adaptive_ob_sizing and current_atr > 0:
                                atr_adj = current_atr * Decimal('0.1'); box_top += atr_adj; box_bottom -= atr_adj
                            new_box = {'id': f"bear_{pivot_candle_idx_dt.strftime('%Y%m%d%H%M')}", 'type': 'bear', 'left_idx': pivot_candle_idx_dt, 'right_idx': df.index[-1],
                                       'top': box_top, 'bottom': box_bottom, 'active': True, 'violated': False, 'expired': False,
                                       'volume': safe_decimal(ob_candle.get('volume')), 'vol_norm': ob_vol_norm}
                            self.bear_boxes.append(new_box); new_bear_indices.add(pivot_candle_idx_dt)
                            self.logger.debug(f"New Bear OB: {new_box['id']} [{new_box['bottom']:.4f}, {new_box['top']:.4f}]")
                except (IndexError, KeyError) as e: self.logger.error(f"Error processing Bear OB signal {signal_idx_dt}: {e}")

            # Bullish OB from Pivot Low
            if is_pl_signal:
                try:
                    signal_idx_loc = df.index.get_loc(signal_idx_dt)
                    pivot_candle_loc = signal_idx_loc - self.strategy_params.pl_right
                    if pivot_candle_loc >= 0:
                        pivot_candle_idx_dt = df.index[pivot_candle_loc]
                        if pivot_candle_idx_dt not in new_bull_indices and not any(b['left_idx'] == pivot_candle_idx_dt for b in self.bull_boxes):
                            ob_candle = df.loc[pivot_candle_idx_dt]
                            ob_vol_norm = int(ob_candle.get('vol_norm_int', 0))
                             # [S17] Validate OB Volume
                            if self.strategy_params.validate_ob_volume and (ob_vol_norm is None or ob_vol_norm < (self.strategy_params.volume_threshold * 0.5)):
                                self.logger.debug(f"[S17] Skipping Bull OB {pivot_candle_idx_dt}: Low volume ({ob_vol_norm})")
                                continue
                            box_top = safe_decimal(ob_candle['high']) if self.strategy_params.ob_source == "Wicks" else safe_decimal(max(ob_candle['open'], ob_candle['close']))
                            box_bottom = safe_decimal(ob_candle['low']) if self.strategy_params.ob_source == "Wicks" else safe_decimal(min(ob_candle['open'], ob_candle['close']))
                            if box_top is None or box_bottom is None or box_top <= box_bottom: continue
                            # [S18] Adaptive OB Sizing
                            if self.strategy_params.adaptive_ob_sizing and current_atr > 0:
                                atr_adj = current_atr * Decimal('0.1'); box_top += atr_adj; box_bottom -= atr_adj
                            new_box = {'id': f"bull_{pivot_candle_idx_dt.strftime('%Y%m%d%H%M')}", 'type': 'bull', 'left_idx': pivot_candle_idx_dt, 'right_idx': df.index[-1],
                                       'top': box_top, 'bottom': box_bottom, 'active': True, 'violated': False, 'expired': False,
                                       'volume': safe_decimal(ob_candle.get('volume')), 'vol_norm': ob_vol_norm}
                            self.bull_boxes.append(new_box); new_bull_indices.add(pivot_candle_idx_dt)
                            self.logger.debug(f"New Bull OB: {new_box['id']} [{new_box['bottom']:.4f}, {new_box['top']:.4f}]")
                except (IndexError, KeyError) as e: self.logger.error(f"Error processing Bull OB signal {signal_idx_dt}: {e}")

        # Manage Existing Boxes (Violation, Extension, Expiry)
        last_close = safe_decimal(df['close'].iloc[-1]); last_high = safe_decimal(df['high'].iloc[-1])
        last_low = safe_decimal(df['low'].iloc[-1]); last_bar_idx = df.index[-1]
        if last_close is None or last_high is None or last_low is None: self.logger.warning("Last candle data missing."); return

        # [S25] ATR-Based OB Expiry Setup
        expiry_atr_periods = self.strategy_params.ob_expiry_atr_periods
        mean_atr, interval_minutes, can_calculate_expiry = Decimal('0'), 0, False
        if expiry_atr_periods is not None and expiry_atr_periods > 0:
             atr_series = df['atr'].dropna()
             if 'atr' in df.columns and len(atr_series) > 50: mean_atr = safe_decimal(atr_series.iloc[-200:].mean(), default=Decimal('0')) # Long average ATR
             else: mean_atr = Decimal('0')
             try: interval_str = self.config.get("interval", DEFAULT_INTERVAL); interval_minutes = int(interval_str) if interval_str.isdigit() else {'D': 1440, 'W': 10080, 'M': 43200}.get(interval_str, 0)
             except: interval_minutes = 0
             if interval_minutes > 0 and mean_atr > 0 and current_atr > 0: can_calculate_expiry = True
             else: self.logger.debug("[S25] Cannot calculate expiry (interval/mean ATR/current ATR invalid).")

        # Update Boxes
        for box_list in [self.bull_boxes, self.bear_boxes]:
            for box in box_list:
                if box['active']:
                    is_bull = box['type'] == 'bull'
                    # Violation Check
                    if (is_bull and last_close < box['bottom']) or (not is_bull and last_close > box['top']):
                        box.update({'active': False, 'violated': True, 'right_idx': last_bar_idx})
                        self.logger.debug(f"{box['type'].capitalize()} OB {box['id']} violated at {last_close:.4f}")
                        continue
                    # Extension
                    if self.strategy_params.ob_extend: box['right_idx'] = last_bar_idx
                    # [S25] Expiry Check
                    if can_calculate_expiry:
                        try:
                            age_timedelta = last_bar_idx - box['left_idx']
                            age_bars = age_timedelta.total_seconds() / (interval_minutes * 60)
                            vol_ratio = (mean_atr / current_atr) if current_atr > 0 else Decimal('1')
                            vol_ratio_clamped = max(Decimal('0.5'), min(vol_ratio, Decimal('2.0')))
                            expiry_threshold_bars = Decimal(str(expiry_atr_periods)) * vol_ratio_clamped
                            expiry_threshold_bars = max(Decimal('10'), expiry_threshold_bars) # Min expiry
                            if Decimal(str(age_bars)) > expiry_threshold_bars:
                                box.update({'active': False, 'expired': True, 'right_idx': last_bar_idx})
                                self.logger.debug(f"[S25] {box['type'].capitalize()} OB {box['id']} expired after {age_bars:.1f} bars (Threshold: {expiry_threshold_bars:.1f}, VolRatio: {vol_ratio_clamped:.2f})")
                        except Exception as e: self.logger.warning(f"[S25] Error calculating expiry for OB {box['id']}: {e}")

        # Prune OB Lists
        sort_key = lambda b: (b.get('active', False), not b.get('expired', False), b.get('left_idx', datetime.min.replace(tzinfo=timezone.utc)))
        for box_list_ref in [self.bull_boxes, self.bear_boxes]:
            box_list_ref.sort(key=sort_key, reverse=True)
        max_active = self.strategy_params.ob_max_boxes; max_inactive = max_active // 2
        active_bull = [b for b in self.bull_boxes if b.get('active')]; inactive_bull = [b for b in self.bull_boxes if not b.get('active')]
        active_bear = [b for b in self.bear_boxes if b.get('active')]; inactive_bear = [b for b in self.bear_boxes if not b.get('active')]
        self.bull_boxes = active_bull[:max_active] + inactive_bull[:max_inactive]
        self.bear_boxes = active_bear[:max_active] + inactive_bear[:max_inactive]
        self.logger.info(f"OB Update: Active Bull={len(active_bull)}, Active Bear={len(active_bear)} (Total Kept: Bull={len(self.bull_boxes)}, Bear={len(self.bear_boxes)})")

    def _get_higher_tf_analysis(self) -> Optional[HigherTimeframeAnalysis]:
        """[S19] Performs basic trend analysis on a higher timeframe."""
        if not self.strategy_params.multi_tf_confirm or self.exchange is None: return None
        higher_tf_interval_str = self.strategy_params.higher_tf
        if higher_tf_interval_str not in CCXT_INTERVAL_MAP: self.logger.warning(f"[S19] Invalid higher TF '{higher_tf_interval_str}'. Skipping."); return None
        base_interval_str = self.config.get("interval", DEFAULT_INTERVAL)
        if CCXT_INTERVAL_MAP.get(higher_tf_interval_str) == CCXT_INTERVAL_MAP.get(base_interval_str): self.logger.warning(f"[S19] Higher TF same as base. Skipping."); return None

        ccxt_higher_tf = CCXT_INTERVAL_MAP[higher_tf_interval_str]
        self.logger.debug(f"[S19] Performing higher timeframe analysis ({ccxt_higher_tf})...")
        try:
            htf_fetch_limit = self.strategy_params.vt_length + 50
            higher_df = fetch_klines_ccxt(self.exchange, self.symbol, ccxt_higher_tf, limit=htf_fetch_limit, logger=self.logger)
            if higher_df.empty or len(higher_df) < self.strategy_params.vt_length: self.logger.warning(f"[S19] Insufficient HTF data ({len(higher_df)}) for {ccxt_higher_tf}."); return None

            htf_ema = ta.ema(higher_df['close'], length=self.strategy_params.vt_length, fillna=False)
            if htf_ema is None or htf_ema.isna().all(): self.logger.warning(f"[S19] Could not calculate EMA on HTF {ccxt_higher_tf}."); return None

            last_htf_close = safe_decimal(higher_df['close'].iloc[-1])
            last_valid_htf_ema = htf_ema.dropna().iloc[-1] if not htf_ema.dropna().empty else None
            last_htf_ema = safe_decimal(last_valid_htf_ema)
            if last_htf_close is None or last_htf_ema is None: self.logger.warning(f"[S19] Missing close/EMA on last HTF {ccxt_higher_tf}."); return None

            htf_trend_up = last_htf_close >= last_htf_ema
            trend_str = 'UP' if htf_trend_up else 'DOWN'
            self.logger.info(f"[S19] Higher timeframe ({ccxt_higher_tf}) trend: {trend_str} (Close={last_htf_close:.4f}, EMA={last_htf_ema:.4f})")
            return HigherTimeframeAnalysis(current_trend_up=htf_trend_up)
        except Exception as e: self.logger.error(f"[S19] Error during HTF ({ccxt_higher_tf}) analysis: {e}", exc_info=True); return None

    def update(self, df: pd.DataFrame) -> Optional[StrategyAnalysisResults]:
        """Updates strategy state and returns analysis results."""
        start_time = time.monotonic()
        self.logger.debug(f"Starting strategy update for {self.symbol} with {len(df)} candles. Last: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'}")
        if df.empty or len(df) < self.min_data_len: self.logger.warning(f"Insufficient data: {len(df)} rows < required {self.min_data_len}"); return None

        df_with_indicators = self._calculate_indicators(df)
        if df_with_indicators.empty: self.logger.error("DF empty after indicators."); return None
        self._identify_order_blocks(df_with_indicators) # Updates self.bull/bear_boxes
        higher_tf_results = self._get_higher_tf_analysis() # [S19]

        try:
            if df_with_indicators.empty: self.logger.error("DF empty post-processing."); return None
            last_row = df_with_indicators.iloc[-1]
            last_close = safe_decimal(last_row.get('close'))
            current_trend_up = last_row.get('trend_up'); current_trend_up = None if pd.isna(current_trend_up) else bool(current_trend_up)
            trend_just_changed = bool(last_row.get('trend_changed', False))
            vol_norm_val = last_row.get('vol_norm_int'); vol_norm_int = int(vol_norm_val) if pd.notna(vol_norm_val) else None # S1
            atr = safe_decimal(last_row.get('atr'))
            upper_band = safe_decimal(last_row.get('upper_band')); lower_band = safe_decimal(last_row.get('lower_band'))
            adx = safe_decimal(last_row.get('adx'))        # S3
            rsi = safe_decimal(last_row.get('rsi'))        # S7
            vwap = safe_decimal(last_row.get('vwap'))      # S15
            roc = safe_decimal(last_row.get('roc'))        # S23

            if last_close is None: self.logger.error("Last close missing."); return None
            active_bull_obs = [b for b in self.bull_boxes if b.get('active')]
            active_bear_obs = [b for b in self.bear_boxes if b.get('active')]

            results = StrategyAnalysisResults(
                dataframe=df_with_indicators, last_close=last_close, current_trend_up=current_trend_up,
                trend_just_changed=trend_just_changed, active_bull_boxes=active_bull_obs, active_bear_boxes=active_bear_obs,
                vol_norm_int=vol_norm_int, atr=atr, upper_band=upper_band, lower_band=lower_band,
                adx=adx, rsi=rsi, vwap=vwap, roc=roc, higher_tf_analysis=higher_tf_results
            )
            self.logger.debug(f"Strategy update finished in {time.monotonic() - start_time:.3f}s.")
            return results
        except IndexError: self.logger.error("Could not access last row of DF (empty?)."); return None
        except Exception as e: self.logger.error(f"Error preparing strategy results: {e}", exc_info=True); return None

# --- Signal Generation ---

class SignalGenerator:
    """Generates trading signals based on strategy analysis results and configuration."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        try:
            self.strategy_params = StrategyParams(**config.get("strategy_params", {}))
            self.protection_config = ProtectionConfig(**config.get("protection", {}))
        except ValidationError as e: self.logger.error(f"SignalGenerator Pydantic validation failed: {e}. Using defaults."); self.strategy_params = StrategyParams(); self.protection_config = ProtectionConfig()
        self.base_ob_entry_proximity_factor = safe_decimal(self.strategy_params.ob_entry_proximity_factor, default=Decimal('1.005'))
        self.base_ob_exit_proximity_factor = safe_decimal(self.strategy_params.ob_exit_proximity_factor, default=Decimal('1.001'))

    def _check_entry_confluence(self, analysis_results: StrategyAnalysisResults) -> bool:
        """Checks all configured confluence filters for a potential entry."""
        latest_close = analysis_results.get('last_close')
        is_trend_up = analysis_results.get('current_trend_up')
        if latest_close is None or latest_close <= 0: self.logger.debug("Confluence Fail: Invalid close."); return False
        if is_trend_up is None: self.logger.debug("Confluence Fail: Trend unknown."); return False

        # [S1] Volume Confirmation
        if self.strategy_params.volume_threshold > 0:
            vol_norm = analysis_results.get('vol_norm_int')
            if vol_norm is None or vol_norm < self.strategy_params.volume_threshold: self.logger.debug(f"[S1] Confluence Fail: Low Vol ({vol_norm} < {self.strategy_params.volume_threshold})"); return False
        # [S3] Trend Strength Filter (ADX)
        if self.strategy_params.adx_threshold > 0:
            adx = analysis_results.get('adx')
            if adx is None or adx < self.strategy_params.adx_threshold: self.logger.debug(f"[S3] Confluence Fail: Weak Trend (ADX {adx:.1f} < {self.strategy_params.adx_threshold})"); return False
        # [S7] RSI Confirmation
        if self.strategy_params.rsi_confirm:
            rsi = analysis_results.get('rsi')
            if rsi is None: self.logger.debug("[S7] Confluence Fail: RSI unavailable"); return False
            if (is_trend_up and rsi >= self.strategy_params.rsi_overbought) or (not is_trend_up and rsi <= self.strategy_params.rsi_oversold): self.logger.debug(f"[S7] Confluence Fail: RSI Extreme ({rsi:.1f}) against trend."); return False
        # [S11] Candlestick Confirmation
        if self.strategy_params.candlestick_confirm:
            df = analysis_results.get('dataframe')
            if df is None or df.empty or not ('bull_engulfing' in df.columns and 'bear_engulfing' in df.columns): self.logger.debug("[S11] Confluence Fail: Candle data unavailable."); return False
            pattern_found = df['bull_engulfing'].iloc[-1] if is_trend_up else df['bear_engulfing'].iloc[-1]
            if not pattern_found: self.logger.debug("[S11] Confluence Fail: No confirming candle pattern."); return False
        # [S15] VWAP Confirmation
        if self.strategy_params.vwap_confirm:
            vwap = analysis_results.get('vwap')
            if vwap is None: self.logger.debug("[S15] Confluence Fail: VWAP unavailable"); return False
            if (is_trend_up and latest_close < vwap) or (not is_trend_up and latest_close > vwap): self.logger.debug(f"[S15] Confluence Fail: Price vs VWAP (P={latest_close:.4f}, V={vwap:.4f})"); return False
        # [S19] Multi-Timeframe Confirmation
        if self.strategy_params.multi_tf_confirm:
            htf_analysis = analysis_results.get('higher_tf_analysis')
            if htf_analysis is None or htf_analysis.get('current_trend_up') is None: self.logger.debug("[S19] Confluence Fail: Higher TF trend unknown."); return False
            if is_trend_up != htf_analysis['current_trend_up']: self.logger.debug(f"[S19] Confluence Fail: Base trend vs HTF trend mismatch."); return False
        # [S23] Momentum Confirmation (ROC)
        if self.strategy_params.momentum_confirm:
             roc = analysis_results.get('roc')
             if roc is None: self.logger.debug("[S23] Confluence Fail: ROC unavailable"); return False
             if (is_trend_up and roc <= 0) or (not is_trend_up and roc >= 0): self.logger.debug(f"[S23] Confluence Fail: Momentum ROC ({roc:.2f}) against trend."); return False

        self.logger.debug("All active entry confluence filters passed.")
        return True

    def _check_ob_entry(self, analysis_results: StrategyAnalysisResults) -> str:
        """Checks for Order Block based entry signals (Zone Entry or Breakout)."""
        signal = "HOLD"
        latest_close = analysis_results.get('last_close'); is_trend_up = analysis_results.get('current_trend_up')
        active_bull_obs = analysis_results.get('active_bull_boxes', []); active_bear_obs = analysis_results.get('active_bear_boxes', [])
        current_atr = analysis_results.get('atr'); df = analysis_results.get('dataframe')
        if latest_close is None or latest_close <= 0: return signal
        if is_trend_up is None: return signal

        # [S8] Calculate Dynamic OB Proximity Factor
        ob_entry_proximity = self.base_ob_entry_proximity_factor
        if self.strategy_params.dynamic_ob_proximity and current_atr is not None and current_atr > 0 and latest_close > 0:
            relative_atr = current_atr / latest_close
            dynamic_factor = Decimal('1') + (relative_atr * Decimal('2')) # Example: scale by 2x rel ATR
            clamped_factor = max(Decimal('1.0'), min(dynamic_factor, Decimal('1.5'))) # Clamp 1.0x - 1.5x
            ob_entry_proximity = self.base_ob_entry_proximity_factor * clamped_factor
            # self.logger.debug(f"[S8] Dynamic OB Entry Proximity: {ob_entry_proximity:.5f}")

        # [S13] Breakout Confirmation Logic
        if self.strategy_params.breakout_confirm:
            if df is None or len(df) < 2: return "HOLD"
            prev_close = safe_decimal(df['close'].iloc[-2])
            if prev_close is None: return "HOLD"
            if is_trend_up:
                relevant_bear_obs = sorted([ob for ob in active_bear_obs if ob.get('top') and ob['top'] > latest_close], key=lambda x: x['top'])
                if relevant_bear_obs:
                    nearest_bear_ob = relevant_bear_obs[0]
                    if nearest_bear_ob.get('top') and latest_close > nearest_bear_ob['top'] and prev_close <= nearest_bear_ob['top']:
                         self.logger.info(f"[S13] BUY signal (Breakout): Broke Bear OB {nearest_bear_ob.get('id','N/A')} Top ({nearest_bear_ob['top']:.4f})")
                         signal = "BUY"
            else: # Bearish Trend
                relevant_bull_obs = sorted([ob for ob in active_bull_obs if ob.get('bottom') and ob['bottom'] < latest_close], key=lambda x: x['bottom'], reverse=True)
                if relevant_bull_obs:
                     nearest_bull_ob = relevant_bull_obs[0]
                     if nearest_bull_ob.get('bottom') and latest_close < nearest_bull_ob['bottom'] and prev_close >= nearest_bull_ob['bottom']:
                          self.logger.info(f"[S13] SELL signal (Breakout): Broke Bull OB {nearest_bull_ob.get('id','N/A')} Bottom ({nearest_bull_ob['bottom']:.4f})")
                          signal = "SELL"
        # Default: Zone Entry Logic
        else:
            ob_hit_count = 0
            if is_trend_up:
                for ob in active_bull_obs:
                    ob_bottom, ob_top = ob.get('bottom'), ob.get('top')
                    if ob_bottom is None or ob_top is None: continue
                    entry_top_boundary = ob_top * ob_entry_proximity # Allow entry slightly above top
                    if ob_bottom <= latest_close <= entry_top_boundary:
                        self.logger.debug(f"Price entered Bull OB {ob.get('id','N/A')} [{ob_bottom:.4f}, {ob_top:.4f}] (ProxTop: {entry_top_boundary:.4f})")
                        ob_hit_count += 1; signal = "BUY" # Simplification: Take first hit if confluence allows
                        break # Only need one hit if confluence is 1, exit loop early
            else: # Bearish Trend
                for ob in active_bear_obs:
                    ob_bottom, ob_top = ob.get('bottom'), ob.get('top')
                    if ob_bottom is None or ob_top is None: continue
                    entry_bottom_boundary = ob_bottom * (Decimal("2") - ob_entry_proximity) # Allow entry slightly below bottom
                    if entry_bottom_boundary <= latest_close <= ob_top:
                        self.logger.debug(f"Price entered Bear OB {ob.get('id','N/A')} [{ob_bottom:.4f}, {ob_top:.4f}] (ProxBot: {entry_bottom_boundary:.4f})")
                        ob_hit_count += 1; signal = "SELL"
                        break

            # [S21] OB Confluence Check
            if signal != "HOLD" and ob_hit_count < self.strategy_params.ob_confluence:
                 self.logger.debug(f"[S21] Signal Filtered: OB Confluence not met ({ob_hit_count} < {self.strategy_params.ob_confluence})")
                 signal = "HOLD"
        return signal

    def _check_exit_conditions(self, analysis_results: StrategyAnalysisResults, open_position: Dict) -> str:
        """Checks for exit conditions based on trend reversal or OB proximity."""
        signal = "HOLD"
        latest_close = analysis_results.get('last_close'); is_trend_up = analysis_results.get('current_trend_up')
        active_bull_obs = analysis_results.get('active_bull_boxes', []); active_bear_obs = analysis_results.get('active_bear_boxes', [])
        current_pos_side = open_position.get('side')
        if latest_close is None or latest_close <= 0: return signal

        # Exit 1: Trend Reversal
        if (current_pos_side == 'long' and is_trend_up is False) or (current_pos_side == 'short' and is_trend_up is True):
            self.logger.info(f"{NEON_YELLOW}Exit Signal: Trend changed against open {current_pos_side} position.{RESET}")
            return f"EXIT_{current_pos_side.upper()}"

        # Exit 2: Price Approaching Opposing OB
        ob_exit_proximity = self.base_ob_exit_proximity_factor # Use base factor for exits for simplicity
        if current_pos_side == 'long':
            relevant_bear_obs = sorted([b for b in active_bear_obs if b.get('bottom') and b['bottom'] > latest_close], key=lambda x: x['bottom'])
            if relevant_bear_obs:
                nearest_bear_ob = relevant_bear_obs[0]; ob_bottom = nearest_bear_ob.get('bottom')
                if ob_bottom:
                    exit_boundary = ob_bottom * (Decimal("2") - ob_exit_proximity) # e.g., 0.999 * bottom
                    if latest_close >= exit_boundary:
                        self.logger.info(f"{NEON_YELLOW}Exit Signal (Long): Price {latest_close:.4f} approaching Bear OB {nearest_bear_ob.get('id','N/A')} Bottom ({ob_bottom:.4f}) prox ({exit_boundary:.4f}).{RESET}")
                        return "EXIT_LONG"
        elif current_pos_side == 'short':
            relevant_bull_obs = sorted([b for b in active_bull_obs if b.get('top') and b['top'] < latest_close], key=lambda x: x['top'], reverse=True)
            if relevant_bull_obs:
                nearest_bull_ob = relevant_bull_obs[0]; ob_top = nearest_bull_ob.get('top')
                if ob_top:
                    exit_boundary = ob_top * ob_exit_proximity # e.g., 1.001 * top
                    if latest_close <= exit_boundary:
                        self.logger.info(f"{NEON_YELLOW}Exit Signal (Short): Price {latest_close:.4f} approaching Bull OB {nearest_bull_ob.get('id','N/A')} Top ({ob_top:.4f}) prox ({exit_boundary:.4f}).{RESET}")
                        return "EXIT_SHORT"
        return signal

    def generate_signal(self, analysis_results: Optional[StrategyAnalysisResults], open_position: Optional[Dict]) -> str:
        """Generates the final BUY/SELL/HOLD/EXIT signal."""
        if analysis_results is None: self.logger.warning("Cannot generate signal: Analysis missing."); return "HOLD"
        if analysis_results.get('current_trend_up') is None: self.logger.warning("Cannot generate signal: Trend unknown."); return "HOLD"

        if open_position: # Check Exits First
            exit_signal = self._check_exit_conditions(analysis_results, open_position)
            if exit_signal != "HOLD": return exit_signal
            else: self.logger.debug("Holding existing position (no exit signal)."); return "HOLD"
        else: # Check Entries
            passes_confluence = self._check_entry_confluence(analysis_results)
            if not passes_confluence: return "HOLD"
            entry_signal = self._check_ob_entry(analysis_results)
            if entry_signal != "HOLD":
                self.logger.info(f"{BRIGHT}{NEON_GREEN if entry_signal == 'BUY' else NEON_RED}Entry Signal Generated: {entry_signal}{RESET}")
                return entry_signal
            self.logger.debug("No entry signal (Trend/OB conditions not met after confluence).")
            return "HOLD"

    def calculate_initial_tp_sl(
        self, entry_price: Decimal, signal: str, analysis_results: StrategyAnalysisResults, market_info: Dict
    ) -> Tuple[Optional[List[Dict]], Optional[Decimal]]:
        """Calculates initial TP levels and SL price based on config and analysis."""
        self.logger.debug(f"Calculating initial TP/SL for {signal} entry at {entry_price:.4f}")
        atr = analysis_results.get('atr'); df = analysis_results.get('dataframe')
        if signal not in ["BUY", "SELL"]: self.logger.error("Invalid signal for TP/SL calc."); return None, None
        if atr is None or atr <= 0: self.logger.warning("ATR invalid for TP/SL calc."); return None, None
        if entry_price <= 0: self.logger.error("Invalid entry price for TP/SL calc."); return None, None

        base_tp_multiple = safe_decimal(self.protection_config.initial_take_profit_atr_multiple, default=Decimal('0.7'))
        base_sl_multiple = safe_decimal(self.protection_config.initial_stop_loss_atr_multiple, default=Decimal('1.8'))
        self.logger.debug(f"Base Multipliers: TP={base_tp_multiple}x, SL={base_sl_multiple}x ATR")

        tp_multiple, sl_multiple = base_tp_multiple, base_sl_multiple

        # [S2] Dynamic ATR Multiplier (Percentile)
        if self.protection_config.dynamic_atr_multiplier and df is not None and 'atr' in df.columns:
            try:
                atr_series = df['atr'].dropna(); window = 100; min_periods = window // 2
                if len(atr_series) >= min_periods:
                    q25_series = atr_series.rolling(window=window, min_periods=min_periods).quantile(0.25)
                    q75_series = atr_series.rolling(window=window, min_periods=min_periods).quantile(0.75)
                    if not q25_series.empty and not q75_series.empty:
                        atr_perc_25, atr_perc_75 = safe_decimal(q25_series.iloc[-1]), safe_decimal(q75_series.iloc[-1])
                        if atr_perc_25 and atr_perc_75:
                            factor = Decimal('1.0')
                            if atr > atr_perc_75 * Decimal('1.1'): factor = Decimal('1.2'); state = "High Vol"
                            elif atr < atr_perc_25 * Decimal('0.9'): factor = Decimal('0.8'); state = "Low Vol"
                            else: state = "Normal Vol"
                            if factor != Decimal('1.0'):
                                tp_multiple *= factor; sl_multiple *= factor
                                self.logger.info(f"[S2] Dynamic TP/SL: {state} (ATR {atr:.4f} vs P25 {atr_perc_25:.4f}/P75 {atr_perc_75:.4f}). Factor: {factor:.2f}")
            except Exception as e: self.logger.warning(f"[S2] Could not calculate ATR percentile: {e}")

        # [S5] Volatility-Based SL Adjustment (Std Dev / Coeff Var)
        if self.protection_config.volatility_sl_adjust and df is not None and 'atr' in df.columns:
            try:
                window = 20; min_periods = window // 2
                recent_atr = df['atr'].iloc[-window:].dropna()
                if len(recent_atr) >= min_periods:
                    atr_std, atr_mean = safe_decimal(recent_atr.std()), safe_decimal(recent_atr.mean())
                    if atr_mean and atr_mean > 0 and atr_std:
                        coeff_of_var = atr_std / atr_mean
                        if coeff_of_var > Decimal('0.3'): # High relative std dev
                            sl_adjust_factor = Decimal('1.2'); original_sl = sl_multiple; sl_multiple *= sl_adjust_factor
                            self.logger.info(f"[S5] Volatility SL Adjust: High ATR CoeffVar ({coeff_of_var:.2f} > 0.3). SL Multiple {original_sl:.2f} -> {sl_multiple:.2f}")
            except Exception as e: self.logger.warning(f"[S5] Could not calculate ATR std dev: {e}")

        self.logger.debug(f"Adjusted Multipliers: TP={tp_multiple:.2f}x, SL={sl_multiple:.2f}x ATR")

        sl_offset = atr * sl_multiple
        sl_price_raw = entry_price - sl_offset if signal == "BUY" else entry_price + sl_offset
        if sl_price_raw <= 0 or sl_price_raw == entry_price: self.logger.error(f"Invalid raw SL ({sl_price_raw})."); return None, None

        tp_levels_calculated: List[Dict] = []
        use_fib_tp = self.strategy_params.fib_tp_levels
        use_partial_tp = isinstance(self.protection_config.partial_tp_levels, list) and self.protection_config.partial_tp_levels

        # [S10] Fibonacci TP Levels (Alternative)
        if use_fib_tp and df is not None:
            self.logger.debug("[S10] Calculating Fibonacci TP levels.")
            try:
                window = 50; min_periods = window // 2 # Longer window for swings
                if len(df) < min_periods: raise ValueError("Not enough data for Fib swing")
                recent_data = df.iloc[-window:]
                swing_high, swing_low = safe_decimal(recent_data['high'].max()), safe_decimal(recent_data['low'].min())
                if not (swing_high and swing_low and swing_high > swing_low): raise ValueError("Invalid swing points")
                price_range = swing_high - swing_low
                fib_levels = [Decimal('0.618'), Decimal('1.0'), Decimal('1.618'), Decimal('2.0')] # Example levels
                if not fib_levels: raise ValueError("No Fib levels")
                percentage_per_level = Decimal('1.0') / Decimal(len(fib_levels))
                for level in fib_levels:
                    tp_price_raw = (entry_price + price_range * level) if signal == "BUY" else (entry_price - price_range * level)
                    if (signal == "BUY" and tp_price_raw > entry_price) or (signal == "SELL" and tp_price_raw < entry_price):
                        tp_levels_calculated.append({'price': tp_price_raw, 'percentage': percentage_per_level})
                    else: self.logger.warning(f"[S10] Skipping Fib TP {level}: Raw price {tp_price_raw} invalid.")
                if not tp_levels_calculated: self.logger.warning("[S10] No valid Fib TP levels calculated.")
            except Exception as e: self.logger.warning(f"[S10] Error calculating Fib TPs: {e}. Falling back to ATR TP."); tp_levels_calculated = []

        # [S4] Partial TP Levels (If Fib not used or failed)
        if not tp_levels_calculated and use_partial_tp:
            self.logger.debug("[S4] Calculating Partial TP levels based on config.")
            remaining_percentage = Decimal('1.0'); total_allocated = Decimal('0.0')
            for i, level in enumerate(self.protection_config.partial_tp_levels):
                try:
                    level_multiple = tp_multiple * safe_decimal(level.multiple, default=Decimal('1.0')) # Use adjusted base TP multiple
                    level_percentage = safe_decimal(level.percentage, default=Decimal('0.0'))
                    actual_percentage = min(level_percentage, remaining_percentage)
                    if actual_percentage <= 0: continue
                    tp_offset = atr * level_multiple
                    tp_price_raw = entry_price + tp_offset if signal == "BUY" else entry_price - tp_offset
                    if (signal == "BUY" and tp_price_raw > entry_price) or (signal == "SELL" and tp_price_raw < entry_price):
                        tp_levels_calculated.append({'price': tp_price_raw, 'percentage': actual_percentage})
                        remaining_percentage -= actual_percentage; total_allocated += actual_percentage
                    else: self.logger.warning(f"[S4] Skipping partial TP {i+1}: Raw price {tp_price_raw} invalid.")
                    if remaining_percentage <= Decimal('0.001'): break
                except Exception as e: self.logger.error(f"[S4] Error processing partial TP {i+1} ({level}): {e}")
            if remaining_percentage > Decimal('0.01') and tp_levels_calculated:
                 self.logger.warning(f"[S4] Partial TPs sum to {total_allocated:.4f}. Allocating remaining {remaining_percentage:.2%} to last.")
                 tp_levels_calculated[-1]['percentage'] += remaining_percentage

        # Default: Single Full TP level (If Fib/Partial not used or failed)
        if not tp_levels_calculated:
            self.logger.debug("Calculating single Full TP level.")
            tp_offset = atr * tp_multiple
            tp_price_raw = entry_price + tp_offset if signal == "BUY" else entry_price - tp_offset
            if (signal == "BUY" and tp_price_raw > entry_price) or (signal == "SELL" and tp_price_raw < entry_price):
                tp_levels_calculated.append({'price': tp_price_raw, 'percentage': Decimal('1.0')})
            else: self.logger.warning(f"Invalid single raw TP ({tp_price_raw}). No TP will be set.")

        if not tp_levels_calculated: self.logger.error("No valid TP levels calculated."); return None, None

        # [S14] Risk-Reward Ratio Filter
        first_tp_price = tp_levels_calculated[0].get('price')
        if first_tp_price is None: self.logger.error("First TP price missing for RR calc."); return None, None
        risk_amount = abs(entry_price - sl_price_raw); reward_amount = abs(first_tp_price - entry_price)
        min_rr = safe_decimal(self.protection_config.min_rr_ratio, default=Decimal('1.5'))
        if risk_amount <= 0: self.logger.error(f"Cannot calc RR: Risk zero/negative (E={entry_price}, SL={sl_price_raw})."); return None, None
        if reward_amount <= 0: self.logger.warning(f"Reward zero/negative (E={entry_price}, TP1={first_tp_price})."); return None, None
        rr_ratio = reward_amount / risk_amount
        if rr_ratio < min_rr:
            self.logger.warning(f"[S14] Trade Filtered: R:R Ratio low ({rr_ratio:.2f} < {min_rr}). Risk={risk_amount:.4f}, Reward={reward_amount:.4f}")
            return None, None

        self.logger.info(f"Calculated Raw Protections: SL={sl_price_raw:.4f}, Min R:R={rr_ratio:.2f} (>= {min_rr})")
        for i, tp in enumerate(tp_levels_calculated):
             tp_p = tp.get('price'); tp_pct = tp.get('percentage')
             log_p = f"{tp_p:.4f}" if isinstance(tp_p, Decimal) else 'N/A'
             log_pct = f"{tp_pct:.0%}" if isinstance(tp_pct, Decimal) else 'N/A'
             self.logger.info(f" -> Raw TP {i+1}: Price={log_p}, Percentage={log_pct}")
        return tp_levels_calculated, sl_price_raw


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(
    exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger,
    strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator
) -> None:
    """Performs one cycle of analysis and trading logic for a single symbol."""
    lg = logger
    lg.info(f"---== Cycle Start: Analyzing {symbol} ({config.get('interval', 'N/A')}) ==---")
    cycle_start_time = time.monotonic()

    try:
        # 1. Get Market Info & Validate
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info or not market_info.get('active', False): lg.error(f"Market inactive/error for {symbol}. Skipping."); return
        if not market_info.get('precision', {}).get('price') or not market_info.get('precision', {}).get('amount'): lg.error(f"Price/amount precision missing for {symbol}."); return
        if market_info.get('limits', {}).get('amount', {}).get('min') is None: lg.warning(f"Min amount limit missing for {symbol}.")

        # 2. Fetch Data
        ccxt_interval = CCXT_INTERVAL_MAP.get(config.get("interval"))
        if not ccxt_interval: lg.error(f"Invalid interval '{config.get('interval')}' mapped to None."); return
        fetch_limit = max(config.get("fetch_limit", DEFAULT_FETCH_LIMIT), strategy_engine.min_data_len + 50)
        klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, fetch_limit, lg)
        if klines_df.empty or len(klines_df) < strategy_engine.min_data_len: lg.warning(f"Insufficient klines ({len(klines_df)} < {strategy_engine.min_data_len}). Skipping."); return

        # 3. Run Strategy Analysis
        analysis_results = strategy_engine.update(klines_df)
        if analysis_results is None: lg.error("Strategy analysis failed. Skipping."); return
        latest_close = analysis_results.get('last_close'); current_atr = analysis_results.get('atr')
        if latest_close is None or latest_close <= 0: lg.error(f"Invalid last close ({latest_close}). Skipping."); return
        market_info['last_price'] = latest_close # Add for protection formatting heuristics

        lg.debug(f"Analysis Complete: Close={latest_close:.4f}, ATR={current_atr:.4f if current_atr else 'N/A'}, TrendUp={analysis_results.get('current_trend_up')}")

        # 4. Check Existing Position
        open_position = get_open_position(exchange, symbol, lg)

        # 5. Generate Signal
        signal = signal_generator.generate_signal(analysis_results, open_position)
        lg.info(f"Generated Signal for {symbol}: {signal}")

        # --- 6. Trading Execution ---
        if not config.get("enable_trading", False):
            lg.warning("Trading disabled. Skipping execution.")
            if open_position is None and signal in ["BUY", "SELL"]: lg.info(f"[DISABLED] Would attempt {signal} entry.")
            elif open_position and signal in ["EXIT_LONG", "EXIT_SHORT"]: lg.info(f"[DISABLED] Would attempt closing {open_position['side']}.")
            return

        # ================= ENTRY LOGIC ========================
        if open_position is None and signal in ["BUY", "SELL"]:
            lg.info(f"{BRIGHT}Attempting {signal} entry for {symbol} at ~{latest_close:.4f}...{RESET}")
            # TODO: Add proper check for max_concurrent_positions if trading multiple symbols

            # Calculate TP/SL (includes R:R check - S14)
            tp_levels_raw, initial_sl_raw = signal_generator.calculate_initial_tp_sl(latest_close, signal, analysis_results, market_info)
            if initial_sl_raw is None or not tp_levels_raw: lg.error(f"Trade aborted: Invalid TP/SL or R:R fail. SL={initial_sl_raw}, TP={tp_levels_raw}"); return

            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= Decimal('0'): lg.error(f"Trade aborted: Invalid balance ({balance})."); return

            if market_info.get('is_contract', False):
                if not set_leverage_ccxt(exchange, symbol, config.get("leverage", 1), market_info, lg): lg.error("Trade aborted: Failed to set leverage."); return

            # Calculate Size (includes S20 ATR sizing if enabled)
            position_size = calculate_position_size(balance, config["risk_per_trade"], latest_close, initial_sl_raw, market_info, config, current_atr, lg)
            if position_size is None or position_size <= 0: lg.error("Trade aborted: Invalid size."); return

            # Place Entry Order
            limit_entry_price_raw = None
            if config.get('order_type') == 'limit':
                 price_tick = market_info['precision'].get('price')
                 if not price_tick or price_tick <= 0: lg.error("Cannot calc limit entry: Invalid tick size."); return
                 offset = price_tick * Decimal('2'); limit_entry_price_raw = latest_close - offset if signal == "BUY" else latest_close + offset
                 lg.info(f"Calculated Raw Limit Entry Price: {limit_entry_price_raw:.4f}")

            trade_order = place_trade(exchange, symbol, signal, position_size, market_info, config, lg, reduce_only=False, limit_price=limit_entry_price_raw)

            # Post-Entry Processing
            if trade_order and trade_order.get('id'):
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order {trade_order.get('id')} placed. Waiting {confirm_delay}s for confirmation...")
                time.sleep(confirm_delay)
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    lg.info(f"{NEON_GREEN}{signal} position confirmed for {symbol}.{RESET}")
                    entry_price_actual = confirmed_position.get('entryPrice') or latest_close # Fallback
                    if entry_price_actual != latest_close: lg.info(f"Recalculating protection for actual entry: {entry_price_actual:.4f}")

                    # Recalculate TP/SL based on actual entry price
                    tp_levels_final_raw, sl_final_raw = signal_generator.calculate_initial_tp_sl(entry_price_actual, signal, analysis_results, market_info)

                    if sl_final_raw is None or not tp_levels_final_raw: lg.error("Failed to recalculate valid SL/TP post-entry! Manual monitoring needed.")
                    else:
                        first_tp_raw = tp_levels_final_raw[0]['price'] if tp_levels_final_raw else None
                        pos_idx = confirmed_position.get('info', {}).get('positionIdx', 0)
                        protection_params = {'stop_loss_price': sl_final_raw, 'take_profit_price': first_tp_raw, 'position_idx': pos_idx}

                        # Add initial TSL params if enabled
                        if config.get("protection", {}).get("enable_trailing_stop", False):
                            lg.info("Calculating initial TSL params...")
                            use_atr_tsl = config.get("protection", {}).get("use_atr_trailing_stop", True)
                            tsl_dist, tsl_act = None, None
                            if use_atr_tsl and current_atr and current_atr > 0:
                                tsl_mult = safe_decimal(config.get("protection", {}).get("trailing_stop_atr_multiple", 1.5))
                                tsl_dist = current_atr * tsl_mult
                            else:
                                tsl_rate = safe_decimal(config.get("protection", {}).get("trailing_stop_callback_rate", 0.005))
                                tsl_dist = entry_price_actual * tsl_rate
                            tsl_act_pct = safe_decimal(config.get("protection", {}).get("trailing_stop_activation_percentage", 0.003))
                            tsl_act_offset = entry_price_actual * tsl_act_pct
                            tsl_act = entry_price_actual + tsl_act_offset if signal == "BUY" else entry_price_actual - tsl_act_offset
                            if tsl_dist and tsl_dist > 0:
                                protection_params['trailing_stop_distance'] = tsl_dist
                                if tsl_act and tsl_act > 0: protection_params['tsl_activation_price'] = tsl_act
                                else: lg.warning("Invalid TSL activation price calc.")
                            else: lg.warning("Invalid TSL distance calc.")

                        # Set initial protections
                        lg.info(f"Setting initial protection: SL={sl_final_raw:.4f}, TP={first_tp_raw:.4f if first_tp_raw else 'N/A'}, TSL_Dist={protection_params.get('trailing_stop_distance')}, TSL_Act={protection_params.get('tsl_activation_price')}")
                        protection_success = _set_position_protection(exchange, symbol, market_info, lg, **protection_params)
                        if not protection_success: lg.warning(f"{NEON_YELLOW}Failed initial protection set for {symbol}. Manual monitoring advised.{RESET}")

                        # [S4] Place Partial TP Limit Orders (Beyond first TP)
                        if protection_success and len(tp_levels_final_raw) > 1 and config.get("protection", {}).get("partial_tp_levels"):
                            lg.info("[S4] Attempting partial TP limit orders...")
                            total_pos_size = abs(confirmed_position.get('contracts', position_size))
                            for i, tp_level in enumerate(tp_levels_final_raw[1:], start=1):
                                tp_price_raw = tp_level.get('price'); level_percentage = safe_decimal(tp_level.get('percentage'), default=Decimal('0.0'))
                                if tp_price_raw is None or level_percentage <= 0: lg.warning(f"[S4] Skipping partial TP {i+1}: Invalid data."); continue
                                size_for_this_level = total_pos_size * level_percentage
                                if size_for_this_level > 0:
                                    tp_side = "SELL" if signal == "BUY" else "BUY"
                                    lg.info(f"[S4] Placing Partial TP {i+1}: Side={tp_side}, Size%={level_percentage:.1%}, Price={tp_price_raw:.4f}")
                                    formatted_tp_size = format_value_for_exchange(size_for_this_level, 'amount', market_info, lg, ROUND_DOWN)
                                    if formatted_tp_size is None or formatted_tp_size <= 0: lg.warning(f"[S4] Skipping partial TP {i+1}: Formatted size invalid."); continue
                                    try:
                                        tp_order = place_trade(exchange, symbol, tp_side, formatted_tp_size, market_info, config, lg, reduce_only=True, limit_price=tp_price_raw)
                                        if tp_order and tp_order.get('id'): lg.info(f"[S4] Partial TP Order {i+1} placed: ID {tp_order.get('id')}")
                                        else: lg.error(f"[S4] Failed partial TP order {i+1} (API response issue).")
                                    except Exception as e: lg.error(f"[S4] Exception placing partial TP {i+1}: {e}", exc_info=True)
                                else: lg.warning(f"[S4] Skipping partial TP {i+1}: Calc size zero.")
                else: lg.error(f"Failed confirmation for {symbol} after order {trade_order.get('id')}. Manual check needed.")
            else: lg.error(f"Order placement failed for {signal} {symbol}. No order returned.")

        # ================= EXIT & MANAGEMENT LOGIC ===================
        elif open_position and signal in ["HOLD", "EXIT_LONG", "EXIT_SHORT"]:
            pos_side = open_position['side']; entry_price = open_position.get('entryPrice')
            pos_size = open_position.get('contracts'); pos_idx = open_position.get('info', {}).get('positionIdx', 0)
            if entry_price is None or pos_size is None: lg.error(f"Cannot manage {symbol}: Missing pos info."); return

            lg.debug(f"Managing open {pos_side.upper()} position. Entry: {entry_price:.4f}, Size: {pos_size}")

            # --- Strategy Exit Signal ---
            if (pos_side == 'long' and signal == "EXIT_LONG") or (pos_side == 'short' and signal == "EXIT_SHORT"):
                lg.info(f"{NEON_YELLOW}Strategy exit signal '{signal}'. Closing position...{RESET}")
                close_signal = "SELL" if pos_side == 'long' else "BUY"; size_to_close = abs(pos_size)
                formatted_close_size = format_value_for_exchange(size_to_close, 'amount', market_info, lg, ROUND_DOWN)
                if formatted_close_size and formatted_close_size > 0:
                    close_order = place_trade(exchange, symbol, close_signal, formatted_close_size, market_info, config, lg, reduce_only=True, limit_price=None)
                    if close_order: lg.info("Position close order placed."); # TODO: Cancel open TP orders
                    else: lg.error("Failed placing close order!")
                else: lg.error(f"Failed formatting close size {size_to_close}.")
                return # Exit cycle after close attempt

            # [S6] Max Holding Time Exit
            max_holding_minutes = config.get('max_holding_minutes'); entry_timestamp_ms = open_position.get('entryTimestamp')
            if max_holding_minutes is not None and max_holding_minutes > 0 and entry_timestamp_ms:
                try:
                    entry_time = datetime.fromtimestamp(entry_timestamp_ms / 1000, tz=timezone.utc)
                    holding_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
                    if holding_minutes > max_holding_minutes:
                        lg.warning(f"{NEON_YELLOW}[S6] Held {holding_minutes:.1f}m > max {max_holding_minutes}m. Closing position.{RESET}")
                        close_signal = "SELL" if pos_side == 'long' else "BUY"; size_to_close = abs(pos_size)
                        formatted_close_size = format_value_for_exchange(size_to_close, 'amount', market_info, lg, ROUND_DOWN)
                        if formatted_close_size and formatted_close_size > 0:
                             close_order = place_trade(exchange, symbol, close_signal, formatted_close_size, market_info, config, lg, reduce_only=True, limit_price=None)
                             if close_order: lg.info("[S6] Time-based close order placed.")
                             else: lg.error("[S6] Failed placing time-based close order.")
                        else: lg.error(f"[S6] Failed formatting time-based close size {size_to_close}.")
                        return
                except Exception as time_e: lg.error(f"[S6] Error calculating holding time: {time_e}")

            # --- Manage Protections (BE, TSL, OB Trailing) ---
            protection_cfg = config.get("protection", {})
            current_sl = open_position.get('stopLossPrice'); current_tp = open_position.get('takeProfitPrice')
            current_tsl_dist = open_position.get('tslDistance'); current_tsl_active_price = open_position.get('tslActivationPrice')

            # Break-Even Logic
            if protection_cfg.get("enable_break_even", False) and current_atr is not None and current_atr > 0:
                profit = (latest_close - entry_price) if pos_side == 'long' else (entry_price - latest_close)
                trigger_atr_multiple = safe_decimal(protection_cfg.get("break_even_trigger_atr_multiple", 1.0), default=Decimal('1.0'))
                be_trigger_threshold = current_atr * trigger_atr_multiple

                # [S16] Dynamic Break-Even Trigger
                if protection_cfg.get("dynamic_be_trigger", True) and entry_timestamp_ms:
                    try:
                        holding_minutes = (datetime.now(timezone.utc) - datetime.fromtimestamp(entry_timestamp_ms / 1000, tz=timezone.utc)).total_seconds() / 60
                        if holding_minutes > 60: # Example: trigger faster after 60 min
                            adj_factor = Decimal('0.7'); dynamic_thresh = be_trigger_threshold * adj_factor
                            lg.debug(f"[S16] Dynamic BE: Holding > 60m. Trigger {be_trigger_threshold:.4f} -> {dynamic_thresh:.4f}")
                            be_trigger_threshold = dynamic_thresh
                    except Exception as dyn_be_e: lg.warning(f"[S16] Could not apply dynamic BE: {dyn_be_e}")

                if profit >= be_trigger_threshold:
                    min_tick = market_info['precision'].get('price')
                    if not min_tick or min_tick <= 0: lg.warning("Cannot calc BE SL: Invalid tick size.")
                    else:
                        offset_ticks = protection_cfg.get("break_even_offset_ticks", 2)
                        be_offset = min_tick * offset_ticks
                        be_sl_price_raw = entry_price + be_offset if pos_side == 'long' else entry_price - be_offset
                        sl_needs_update = current_sl is None or \
                                          (pos_side == 'long' and be_sl_price_raw > current_sl) or \
                                          (pos_side == 'short' and be_sl_price_raw < current_sl)
                        if sl_needs_update:
                            lg.info(f"{NEON_GREEN}Profit ({profit:.4f}) >= BE Trigger ({be_trigger_threshold:.4f}). Moving SL to BE target: {be_sl_price_raw:.4f}{RESET}")
                            update_success = _set_position_protection(exchange, symbol, market_info, lg, stop_loss_price=be_sl_price_raw, take_profit_price=current_tp,
                                                                     trailing_stop_distance=current_tsl_dist, tsl_activation_price=current_tsl_active_price, position_idx=pos_idx)
                            if not update_success: lg.warning("Failed updating SL to Break-Even.")

            # Trailing Stop Activation / Update
            if protection_cfg.get("enable_trailing_stop", False):
                lg.debug("Checking TSL status/updates...")
                set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, current_atr, latest_close)
                # Re-fetch position to check if TSL is now active for OB trailing logic
                pos_after_tsl_check = get_open_position(exchange, symbol, lg)
                is_tsl_now_active = pos_after_tsl_check and pos_after_tsl_check.get('tslDistance') is not None and pos_after_tsl_check.get('tslDistance') > 0
            else:
                is_tsl_now_active = False
                pos_after_tsl_check = open_position # Use original if TSL disabled

            # [S24] SL Trailing to OB Boundaries (if TSL not active)
            if protection_cfg.get("sl_trail_to_ob", False) and not is_tsl_now_active and pos_after_tsl_check:
                new_sl_target_raw = None; nearest_protective_ob = None
                active_bull_obs = analysis_results.get('active_bull_boxes', []); active_bear_obs = analysis_results.get('active_bear_boxes', [])
                min_tick = market_info['precision'].get('price')

                if pos_side == 'long':
                    protective_obs = [b for b in active_bull_obs if b.get('bottom') and b['bottom'] < latest_close and b['bottom'] > entry_price]
                    if protective_obs: nearest_protective_ob = max(protective_obs, key=lambda x: x['bottom']); ob_boundary = nearest_protective_ob.get('bottom')
                    if nearest_protective_ob and min_tick and ob_boundary: new_sl_target_raw = ob_boundary - min_tick
                else: # Short
                    protective_obs = [b for b in active_bear_obs if b.get('top') and b['top'] > latest_close and b['top'] < entry_price]
                    if protective_obs: nearest_protective_ob = min(protective_obs, key=lambda x: x['top']); ob_boundary = nearest_protective_ob.get('top')
                    if nearest_protective_ob and min_tick and ob_boundary: new_sl_target_raw = ob_boundary + min_tick

                if new_sl_target_raw and nearest_protective_ob:
                    current_sl_check = pos_after_tsl_check.get('stopLossPrice')
                    sl_needs_update = current_sl_check is None or \
                                      (pos_side == 'long' and new_sl_target_raw > current_sl_check) or \
                                      (pos_side == 'short' and new_sl_target_raw < current_sl_check)
                    if sl_needs_update:
                        lg.info(f"{NEON_GREEN}[S24] Trailing SL to OB {nearest_protective_ob.get('id','N/A')} boundary: Target = {new_sl_target_raw:.4f}{RESET}")
                        update_success = _set_position_protection(exchange, symbol, market_info, lg, stop_loss_price=new_sl_target_raw,
                                                                 take_profit_price=pos_after_tsl_check.get('takeProfitPrice'), # Preserve TP
                                                                 trailing_stop_distance=Decimal('0'), tsl_activation_price=Decimal('0'), # Cancel TSL
                                                                 position_idx=pos_idx)
                        if not update_success: lg.warning("[S24] Failed updating SL to OB boundary.")

        # --- End of Cycle ---
        lg.info(f"---== Cycle End: {symbol} ({time.monotonic() - cycle_start_time:.2f}s) ==---")

    except ccxt.RateLimitExceeded as e: lg.warning(f"Rate limit exceeded {symbol}: {e}."); time.sleep(RETRY_DELAY_SECONDS * 2)
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e: lg.error(f"Network Error {symbol}: {e}.")
    except ccxt.AuthenticationError as e: lg.critical(f"Auth Error {symbol}: {e}. Stopping."); raise SystemExit("Auth Error")
    except ccxt.ExchangeError as e:
        lg.error(f"Unhandled Exchange Error {symbol}: {e}.")
        if "margin" in str(e).lower() or "liquidation" in str(e).lower(): lg.critical(f"Potential critical exchange issue: {e}")
    except Exception as e: lg.critical(f"!!! UNHANDLED EXCEPTION {symbol}: {e} !!!", exc_info=True)

def check_config_reload(logger: logging.Logger) -> bool:
    """Checks if the config file has been modified and reloads if necessary."""
    global CONFIG, config_mtime, QUOTE_CURRENCY
    reloaded = False
    try:
        if not os.path.exists(CONFIG_FILE):
             if config_mtime > 0: logger.error(f"Config {CONFIG_FILE} not found. Using previous config."); config_mtime = 0
             return False
        current_mtime = os.path.getmtime(CONFIG_FILE)
        if current_mtime > config_mtime + 1.0:
            logger.warning(f"{NEON_YELLOW}Config file '{CONFIG_FILE}' changed. Reloading...{RESET}")
            new_config = load_config(CONFIG_FILE)
            if new_config:
                 CONFIG = new_config; QUOTE_CURRENCY = CONFIG.get("quote_currency", DEFAULT_QUOTE_CURRENCY)
                 try: config_mtime = os.path.getmtime(CONFIG_FILE) # Use actual mtime after successful load
                 except OSError: config_mtime = time.time() # Fallback if file disappears during reload
                 logger.info("Configuration reloaded successfully.")
                 reloaded = True
            else: logger.error("Failed reloading config. Continuing with previous.")
    except FileNotFoundError: logger.error(f"Config {CONFIG_FILE} not found during reload."); config_mtime = 0
    except Exception as e: logger.error(f"Error during config reload check: {e}", exc_info=True)
    return reloaded

def validate_symbol(symbol_input: str, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[str]:
    """Validates symbol format and existence on the exchange."""
    try:
        symbol_standard = symbol_input.strip().upper().replace('-', '/')
        if '/' not in symbol_standard or len(symbol_standard.split('/')) != 2:
            logger.error(f"Invalid symbol format: '{symbol_input}'. Use BASE/QUOTE."); return None
        logger.debug(f"Validating symbol '{symbol_standard}' on {exchange.id}...")
        market_info = get_market_info(exchange, symbol_standard, logger) # Uses cache
        if market_info:
            ccxt_symbol = market_info.get('symbol')
            if market_info.get('active'): logger.info(f"Symbol {ccxt_symbol} validated."); return ccxt_symbol
            else: logger.error(f"Symbol {ccxt_symbol} inactive on {exchange.id}."); return None
        else: logger.error(f"Symbol '{symbol_standard}' not found/load failed from {exchange.id}."); return None
    except Exception as e: logger.error(f"Error validating symbol '{symbol_input}': {e}", exc_info=True); return None

def validate_interval(interval_input: str, logger: logging.Logger) -> Optional[str]:
    """Validates the selected interval."""
    interval = interval_input.strip()
    if interval in VALID_INTERVALS: logger.info(f"Interval '{interval}' validated."); return interval
    else: logger.error(f"Invalid interval: '{interval}'. Must be one of {VALID_INTERVALS}"); return None

def main() -> None:
    """Main function to initialize and run the bot."""
    global CONFIG, QUOTE_CURRENCY
    init_logger = setup_logger("BotInit")
    init_logger.info(f"{BRIGHT}{NEON_GREEN}--- Initializing Pyrmethus Volumatic Bot (Enhanced) ---{RESET}")
    init_logger.info(f"Timestamp: {datetime.now(TIMEZONE)}")
    init_logger.info(f"Config: {os.path.abspath(CONFIG_FILE)}, Trading: {CONFIG.get('enable_trading')}, Sandbox: {CONFIG.get('use_sandbox')}")

    if CONFIG.get("enable_trading") and not CONFIG.get("use_sandbox"):
        init_logger.warning(f"{BRIGHT}{NEON_RED}--- LIVE TRADING ENABLED ---{RESET}")
        try:
            print(f"{NEON_YELLOW}Disclaimer: Trading involves risks. API Key: {API_KEY[:5]}...{API_KEY[-4:] if API_KEY else 'N/A'}")
            if input(f"Type 'confirm-live' to proceed: {RESET}").lower() != 'confirm-live': print("Exiting."); return
            init_logger.info("Live trading confirmed.")
        except (KeyboardInterrupt, EOFError): print("Exiting."); return

    exchange = initialize_exchange(init_logger)
    if not exchange: print("Exiting due to exchange init failure."); return

    target_symbol: Optional[str] = None
    while not target_symbol:
        try:
            symbol_input = input(f"Enter symbol (e.g., BTC/USDT): ").strip()
            if symbol_input: target_symbol = validate_symbol(symbol_input, exchange, init_logger)
        except (KeyboardInterrupt, EOFError): print("Exiting."); return

    selected_interval = CONFIG.get('interval', DEFAULT_INTERVAL)
    while True:
        try:
            interval_input = input(f"Enter interval [{'/'.join(VALID_INTERVALS)}] (Default: {selected_interval}): ").strip()
            if not interval_input: validated_interval = selected_interval; init_logger.info(f"Using default interval: {validated_interval}"); break
            else:
                 validated_interval = validate_interval(interval_input, init_logger)
                 if validated_interval:
                     if validated_interval != CONFIG.get('interval'): init_logger.info(f"Using interval '{validated_interval}' for session."); CONFIG["interval"] = validated_interval
                     break
        except (KeyboardInterrupt, EOFError): print("Exiting."); return

    symbol_logger_name = f"Trader_{target_symbol.replace('/', '_')}"
    symbol_logger = setup_logger(symbol_logger_name)
    symbol_logger.info(f"Initializing strategy for {target_symbol} on {CONFIG['interval']}...")

    market_info = get_market_info(exchange, target_symbol, symbol_logger)
    if not market_info: symbol_logger.critical(f"Failed getting market info for {target_symbol}. Exiting."); print(f"Exiting: Market info failed for {target_symbol}."); return

    strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger, exchange)
    signal_generator = SignalGenerator(CONFIG, symbol_logger)

    symbol_logger.info(f"{BRIGHT}{NEON_GREEN}--- Starting Trading Loop for {target_symbol} ---{RESET}")
    try:
        while True:
            loop_start_timestamp = time.time()
            if check_config_reload(symbol_logger):
                symbol_logger.info("Re-initializing engines due to config reload...")
                market_info = get_market_info(exchange, target_symbol, symbol_logger)
                if not market_info: symbol_logger.error("Market info failed post-reload. Skipping."); time.sleep(CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)); continue
                strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger, exchange)
                signal_generator = SignalGenerator(CONFIG, symbol_logger)
                symbol_logger.info("Re-initialization complete.")

            analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger, strategy_engine, signal_generator)

            elapsed_time = time.time() - loop_start_timestamp
            loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_time = max(0, loop_delay - elapsed_time)
            symbol_logger.debug(f"Loop took {elapsed_time:.2f}s. Sleeping {sleep_time:.2f}s.")
            if sleep_time > 0: time.sleep(sleep_time)
    except KeyboardInterrupt: symbol_logger.info("Keyboard interrupt. Shutting down...")
    except SystemExit as e: symbol_logger.critical(f"System exit: {e}")
    except Exception as e: symbol_logger.critical(f"Critical unhandled exception in main loop: {e}", exc_info=True)
    finally:
        print(f"\n{NEON_PURPLE}--- Bot Shutdown Sequence ---{RESET}")
        symbol_logger.info("--- Bot Shutdown Sequence ---")
        if CONFIG.get("enable_trading") and exchange and market_info:
            try:
                symbol_logger.warning("Attempting shutdown cleanup...")
                open_pos = get_open_position(exchange, target_symbol, symbol_logger)
                if open_pos and open_pos.get('contracts'):
                    pos_side = open_pos['side']; pos_size = abs(open_pos['contracts'])
                    lg.warning(f"Closing open {pos_side} position {target_symbol} (Size: {pos_size})...")
                    close_sig = "SELL" if pos_side == 'long' else "BUY"
                    close_size_formatted = format_value_for_exchange(pos_size, 'amount', market_info, lg, ROUND_DOWN)
                    if close_size_formatted and close_size_formatted > 0: place_trade(exchange, target_symbol, close_sig, close_size_formatted, market_info, CONFIG, lg, reduce_only=True)
                    else: lg.error("Could not format shutdown close size.")
                    time.sleep(2)
                lg.info(f"Cancelling open orders for {target_symbol}...")
                cancel_response = exchange.cancel_all_orders(target_symbol)
                lg.debug(f"Cancel orders response: {cancel_response}")
            except Exception as shutdown_e: symbol_logger.error(f"Error during shutdown cleanup: {shutdown_e}", exc_info=True)
        if exchange and hasattr(exchange, 'close'):
            try: exchange.close(); symbol_logger.info("Exchange connection closed.")
            except Exception as e: symbol_logger.error(f"Error closing exchange: {e}")
        logging.shutdown()
        print(f"{NEON_PURPLE}--- Bot Shutdown Complete ---{RESET}")

if __name__ == "__main__":
    main()
```

**Summary of Changes and Integrated Snippets:**

1.  **Configuration (Pydantic Models & Defaults):**
    *   Added fields to `StrategyParams` for: `volume_threshold` (S1), `adx_threshold` (S3), `rsi_confirm`, `rsi_oversold`, `rsi_overbought` (S7), `dynamic_ob_proximity` (S8), `fib_tp_levels` (S10), `candlestick_confirm` (S11), `breakout_confirm` (S13), `vwap_confirm` (S15), `validate_ob_volume` (S17), `adaptive_ob_sizing` (S18), `multi_tf_confirm`, `higher_tf` (S19), `ob_confluence` (S21), `momentum_confirm`, `roc_length` (S23), `ob_expiry_atr_periods` (S25).
    *   Added fields to `ProtectionConfig` for: `dynamic_atr_multiplier` (S2), `volatility_sl_adjust` (S5), `tsl_activation_delay_atr` (S9), `use_atr_trailing_stop`, `trailing_stop_atr_multiple` (S12), `min_rr_ratio` (S14), `dynamic_be_trigger` (S16), `dynamic_tsl_callback` (S22), `sl_trail_to_ob` (S24).
    *   Updated `ProtectionConfig.partial_tp_levels` to use `TakeProfitLevel` model (S4).
    *   Added fields to main `Config` for: `max_holding_minutes` (S6), `atr_position_sizing` (S20).
    *   Updated default constants list to include defaults for all new parameters.

2.  **Indicator Calculation (`_calculate_indicators`):**
    *   Integrated calculation for Volume Percentile Rank (`vol_norm_int`) (S1).
    *   Added ADX calculation (S3).
    *   Added RSI calculation (S7).
    *   Added basic candlestick pattern detection (Engulfing) using `pandas_ta` (S11).
    *   Added VWAP calculation using `pandas_ta` (S15).
    *   Added ROC calculation (S23).
    *   Updated `_calculate_min_data_length` to account for new lookbacks.
    *   Improved NaN dropping logic based on *required* available indicators.

3.  **Order Block Logic (`_identify_order_blocks`):**
    *   Added check for OB formation volume (`validate_ob_volume`) (S17).
    *   Added adaptive OB sizing based on ATR (`adaptive_ob_sizing`) (S18).
    *   Added OB expiry logic based on ATR periods (`ob_expiry_atr_periods`) (S25).

4.  **Signal Generation (`SignalGenerator`):**
    *   **`_check_entry_confluence`:** Added checks for all relevant filter snippets (S1, S3, S7, S11, S15, S19, S23).
    *   **`_check_ob_entry`:** Added logic for dynamic OB entry proximity (S8) and OB breakout confirmation (S13). Implemented OB confluence check (S21).
    *   **`_check_exit_conditions`:** Added exit logic based on price approaching opposing OBs.
    *   **`calculate_initial_tp_sl`:**
        *   Implemented dynamic ATR multiplier adjustment (S2).
        *   Implemented volatility-based SL adjustment (S5).
        *   Added logic for Fibonacci TP levels (S10).
        *   Refined partial TP logic (S4) to work with adjusted base multiples and handle alternatives.
        *   Implemented Risk/Reward Ratio filter (S14).

5.  **Position Sizing (`calculate_position_size`):**
    *   Added ATR-based position sizing adjustment factor (S20).

6.  **Position Management (`analyze_and_trade_symbol` & Helpers):**
    *   Added Max Holding Time check and exit (S6).
    *   **`set_trailing_stop_loss`:** Implemented TSL activation delay (S9) and dynamic TSL distance adjustment (S22). Code already handled ATR-based TSL (S12) via config.
    *   **`analyze_and_trade_symbol`:**
        *   Implemented Dynamic Break-Even trigger (S16).
        *   Implemented SL Trailing to OB boundary (S24), ensuring it doesn't conflict directly with active TSL.
        *   Added logic to place subsequent partial TP orders (beyond the first) as separate reduce-only limit orders (S4).

7.  **Error Handling & Retries:**
    *   Fixed the `tenacity` retry decorator combination using `retry_any`.
    *   Added more specific logging for snippet actions (e.g., `[S14] Trade Filtered...`).
    *   Improved handling of potential `None` values or calculation failures in various functions.

8.  **General:**
    *   Passed `exchange` object to `VolumaticOBStrategy` for HTF analysis (S19).
    *   Passed `current_atr` and `current_price` where needed for dynamic calculations.
    *   Updated docstrings and comments to reflect changes.

This enhanced version integrates all 25 snippets, significantly increasing the complexity and configurability of the bot. Thorough testing in sandbox mode is crucial before deploying with real funds.