
Could not fetch initial balance (potential connection issue or account type mismatch): bybit {"retCode":10001,"retMsg":"accountType only support UNIFIED.","result":{},"retExtInfo":{},"time":1745561386358}       2025-04-25 01:09:46 - INFO     - [BotInit] - Exchange object created, but balance check failed. Proceeding cautiously.
Enter symbol to trade (e.g., BTC/USDT): DOT/USDT:USDT           2025-04-25 01:09:58 - INFO     - [BotInit] - Cached market info for DOT/USDT:USDT: Type=swap, Contract=True, Active=True        2025-04-25 01:09:58 - INFO     - [BotInit] - Symbol DOT/USDT:USDT validated successfully.                                       Enter interval [1/3/5/15/30/60/120/240/D/W/M] (Press Enter for config value: 3): 3
2025-04-25 01:10:03 - INFO     - [BotInit] - Interval '3' validated.                                                            2025-04-25 01:10:03 - INFO     - [Trader_DOT_USDT:USDT] - Initializing strategy for DOT/USDT:USDT on 3 interval...
2025-04-25 01:10:03 - INFO     - [Trader_DOT_USDT:USDT] - --- Starting Trading Loop for DOT/USDT:USDT ---                       2025-04-25 01:10:03 - INFO     - [Trader_DOT_USDT:USDT] - ---== Cycle Start: Analyzing DOT/USDT:USDT (3) ==---
2025-04-25 01:10:03 - INFO     - [Trader_DOT_USDT:USDT] - Fetched and processed 1000 klines for DOT/USDT:USDT (3m). Last timestamp: 2025-04-25 06:09:00 UTC                                     2025-04-25 01:10:03 - WARNING  - [Trader_DOT_USDT:USDT] - Insufficient kline data after fetch: 1000 rows, require 1110. Skipping cycle.
2025-04-25 01:10:18 - INFO     - [Trader_DOT_USDT:USDT] - ---== Cycle Start: Analyzing DOT/USDT:USDT (3) ==---
2025-04-25 01:10:18 - INFO     - [Trader_DOT_USDT:USDT] - Fetched and processed 1000 klines for DOT/USDT:USDT (3m). Last timestamp: 2025-04-25 06:09:00 UTC
2025-04-25 01:10:18 - WARNING  - [Trader_DOT_USDT:USDT] - Insufficient kline data after fetch: 1000 rows, require 1110. Skipping cycle.
2025-04-25 01:10:33 - INFO     - [Trader_DOT_USDT:USDT] - ---== Cycle Start: Analyzing DOT/USDT:USDT (3) ==---
2025-04-25 01:10:33 - INFO     - [Trader_DOT_USDT:USDT] - Fetched and processed 1000 klines for DOT/USDT:USDT (3m). Last timestamp: 2025-04-25 06:09:00 UTC
2025-04-25 01:10:33 - WARNING  - [Trader_DOT_USDT:USDT] - Insufficient kline data after fetch: 1000 rows, require 1110. Skipping cycle.                                                         2025-04-25 01:10:48 - INFO     - [Trader_DOT_USDT:USDT] - ---== Cycle Start: Analyzing DOT/USDT:USDT (3) ==---                  2025-04-25 01:10:48 - INFO     - [Trader_DOT_USDT:USDT] - Fetched and processed 1000 klines for DOT/USDT:USDT (3m). Last timestamp: 2025-04-25 06:09:00 UTC
2025-04-25 01:10:48 - WARNING  - [Trader_DOT_USDT:USDT] - Insufficient kline data after fetch: 1000 rows, require 1110. Skipping cycle.                                                         2025-04-25 01:11:03 - INFO     - [Trader_DOT_USDT:USDT] - ---== Cycle Start: Analyzing DOT/USDT:USDT (3) ==---                  2025-04-25 01:11:03 - INFO     - [Trader_DOT_USDT:USDT] - Fetched and processed 1000 klines for DOT/USDT:USDT (3m). Last timestamp: 2025-04-25 06:09:00 UTC
2025-04-25 01:11:03 - WARNING  - [Trader_DOT_USDT:USDT] - Insufficient kline data after fetch: 1000 rows, require 1110. Skipping cycle.                                                         2025-04-25 01:11:18 - INFO     - [Trader_DOT_USDT:USDT] - ---== Cycle Start: Analyzing DOT/USDT:USDT (3) ==---                  2025-04-25 01:11:18 - INFO     - [Trader_DOT_USDT:USDT] - Fetched and processed 1000 klines for DOT/USDT:USDT (3m). Last timestamp: 2025-04-25 06:09:00 UTC                                     2025-04-25 01:11:18 - WARNING  - [Trader_DOT_USDT:USD
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
                      retry_if_exception_type, retry_if_exception, retry_any) # Added retry_if_exception, retry_any

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

# Default Strategy/Indicator Parameters (Overridden by config.json)
# Volumatic Trend Params
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 1060 # Note: Original implementation used EMA length, percentile rank is likely better
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0 # Used for visualization only
# Order Block Params
DEFAULT_OB_SOURCE = "Wicks"  # "Wicks" or "Bodys"
DEFAULT_PH_LEFT = 10
DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10
DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50
DEFAULT_OB_ENTRY_PROXIMITY_FACTOR = 1.005 # e.g., 1.005 = 0.5% beyond OB edge
DEFAULT_OB_EXIT_PROXIMITY_FACTOR = 1.001 # e.g., 1.001 = 0.1% before OB edge
# Snippet Defaults
DEFAULT_VOLUME_THRESHOLD = 75 # Snippet 1: Min normalized volume % (Percentile Rank)
DEFAULT_ADX_THRESHOLD = 25 # Snippet 3: Min ADX value for trend strength
DEFAULT_RSI_CONFIRM = True # Snippet 7: Use RSI filter
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_RSI_OVERBOUGHT = 70
DEFAULT_DYNAMIC_OB_PROXIMITY = True # Snippet 8: Adjust OB proximity based on volatility
DEFAULT_FIB_TP_LEVELS = False # Snippet 10: Use Fibonacci levels for TP
DEFAULT_CANDLESTICK_CONFIRM = True # Snippet 11: Use candlestick pattern filter
DEFAULT_BREAKOUT_CONFIRM = False # Snippet 13: Use breakout entry logic
DEFAULT_VWAP_CONFIRM = True # Snippet 15: Use VWAP filter
DEFAULT_VALIDATE_OB_VOLUME = True # Snippet 17: Check volume on OB candle
DEFAULT_ADAPTIVE_OB_SIZING = True # Snippet 18: Adjust OB size based on ATR
DEFAULT_MULTI_TF_CONFIRM = False # Snippet 19: Use higher timeframe confirmation
DEFAULT_HIGHER_TF = "60" # Snippet 19: Higher timeframe interval
DEFAULT_OB_CONFLUENCE = 1 # Snippet 21: Min number of OBs to hit for entry
DEFAULT_MOMENTUM_CONFIRM = True # Snippet 23: Use ROC momentum filter
DEFAULT_ROC_LENGTH = 10 # Snippet 23: ROC lookback period
DEFAULT_OB_EXPIRY_ATR_PERIODS = 50.0 # Snippet 25: OB expiry based on ATR periods (None to disable)

# Default Protection Parameters (Overridden by config.json)
DEFAULT_ENABLE_TRAILING_STOP = True
DEFAULT_TSL_CALLBACK_RATE = 0.005 # 0.5% - Used if ATR TSL disabled
DEFAULT_TSL_ACTIVATION_PERCENTAGE = 0.003 # 0.3% - Used if ATR TSL disabled
DEFAULT_ENABLE_BREAK_EVEN = True
DEFAULT_BE_TRIGGER_ATR_MULTIPLE = 1.0 # Profit in ATR multiples to trigger BE
DEFAULT_BE_OFFSET_TICKS = 2 # How many ticks above/below entry for BE SL
DEFAULT_INITIAL_SL_ATR_MULTIPLE = 1.8
DEFAULT_INITIAL_TP_ATR_MULTIPLE = 0.7 # Base TP multiple (used for first partial or full TP)
DEFAULT_DYNAMIC_ATR_MULTIPLIER = True # Snippet 2: Adjust TP/SL multiples based on ATR percentile
DEFAULT_VOLATILITY_SL_ADJUST = True # Snippet 5: Widen SL based on ATR std dev
DEFAULT_TSL_ACTIVATION_DELAY_ATR = 0.5 # Snippet 9: Min profit in ATR before TSL activates (0 to disable)
DEFAULT_USE_ATR_TRAILING_STOP = True # Snippet 12: Use ATR for TSL distance
DEFAULT_TSL_ATR_MULTIPLE = 1.5 # Snippet 12: ATR multiple for TSL distance
DEFAULT_MIN_RR_RATIO = 1.5 # Snippet 14: Minimum Risk/Reward ratio filter
DEFAULT_DYNAMIC_BE_TRIGGER = True # Snippet 16: Adjust BE trigger based on trade duration
DEFAULT_DYNAMIC_TSL_CALLBACK = True # Snippet 22: Adjust TSL distance based on profit
DEFAULT_SL_TRAIL_TO_OB = False # Snippet 24: Trail SL to nearest valid OB boundary
DEFAULT_PARTIAL_TP_LEVELS = [ # Snippet 4: Default partial TP structure
    {'multiple': 0.7, 'percentage': 0.5}, # 50% size at 0.7x base TP multiple
    {'multiple': 1.5, 'percentage': 0.5}  # Remaining 50% size at 1.5x base TP multiple
]
DEFAULT_ATR_POSITION_SIZING = True # Snippet 20: Adjust position size based on ATR volatility
DEFAULT_MAX_HOLDING_MINUTES = 240 # Snippet 6: Max time to hold position (None to disable)
DEFAULT_ORDER_TYPE = "market" # "market" or "limit"

# Fetch limit for initial historical data
DEFAULT_FETCH_LIMIT = 750

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
    """Defines a single level for partial take profit."""
    multiple: float = Field(..., gt=0, description="ATR multiple for this TP level (relative to base TP multiple)")
    percentage: float = Field(..., gt=0, le=1.0, description="Percentage of the position to close at this level (0.0 to 1.0)")

class StrategyParams(BaseModel):
    """Strategy-specific parameters."""
    vt_length: int = Field(DEFAULT_VT_LENGTH, gt=0, description="Length for Volumatic Trend Moving Average")
    vt_atr_period: int = Field(DEFAULT_VT_ATR_PERIOD, gt=0, description="ATR period for Volumatic Trend bands")
    vt_vol_ema_length: int = Field(DEFAULT_VT_VOL_EMA_LENGTH, gt=0, description="Lookback period for Volume Normalization (Percentile Rank Window)")
    vt_atr_multiplier: float = Field(DEFAULT_VT_ATR_MULTIPLIER, gt=0, description="ATR multiplier for Volumatic Trend upper/lower bands")
    vt_step_atr_multiplier: float = Field(DEFAULT_VT_STEP_ATR_MULTIPLIER, ge=0, description="ATR multiplier for Volumatic volume step visualization (if used)")
    ob_source: str = Field(DEFAULT_OB_SOURCE, regex="^(Wicks|Bodys)$", description="Source for OB pivots ('Wicks' or 'Bodys')") # Use regex for pattern
    ph_left: int = Field(DEFAULT_PH_LEFT, gt=0, description="Left lookback for Pivot High")
    ph_right: int = Field(DEFAULT_PH_RIGHT, gt=0, description="Right lookback for Pivot High")
    pl_left: int = Field(DEFAULT_PL_LEFT, gt=0, description="Left lookback for Pivot Low")
    pl_right: int = Field(DEFAULT_PL_RIGHT, gt=0, description="Right lookback for Pivot Low")
    ob_extend: bool = Field(DEFAULT_OB_EXTEND, description="Extend OB boxes to the right until violated")
    ob_max_boxes: int = Field(DEFAULT_OB_MAX_BOXES, gt=0, description="Max number of active OBs to track")
    ob_entry_proximity_factor: float = Field(DEFAULT_OB_ENTRY_PROXIMITY_FACTOR, ge=1.0, description="Factor to extend OB range for entry signal (e.g., 1.005 = 0.5% beyond edge)")
    ob_exit_proximity_factor: float = Field(DEFAULT_OB_EXIT_PROXIMITY_FACTOR, ge=1.0, description="Factor to shrink OB range for exit signal (e.g., 1.001 means exit 0.1% before edge)")
    volume_threshold: int = Field(DEFAULT_VOLUME_THRESHOLD, ge=0, le=100, description="[S1] Min normalized volume percentile rank for entry confirmation") # Adjusted range 0-100
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
    def check_rsi_levels(cls, v, values): # Pydantic v1: v is the field value, values is a dict of other fields
        """Ensure RSI oversold is less than overbought."""
        if 'rsi_overbought' in values and v >= values['rsi_overbought']:
            raise ValueError('rsi_oversold must be less than rsi_overbought')
        return v

    @validator('higher_tf')
    def check_higher_tf(cls, v: str, values): # Pydantic v1 signature
        """Ensure higher_tf is a valid interval if multi_tf_confirm is True."""
        if values.get('multi_tf_confirm') and v not in VALID_INTERVALS:
             raise ValueError(f"higher_tf '{v}' must be one of {VALID_INTERVALS} if multi_tf_confirm is true")
        return v

class ProtectionConfig(BaseModel):
    """Position protection parameters (SL, TP, BE, TSL)."""
    enable_trailing_stop: bool = Field(DEFAULT_ENABLE_TRAILING_STOP, description="Enable Trailing Stop Loss")
    trailing_stop_callback_rate: float = Field(DEFAULT_TSL_CALLBACK_RATE, ge=0, description="[If ATR TSL Disabled] TSL distance as a percentage of entry price (e.g., 0.005 = 0.5%)")
    trailing_stop_activation_percentage: float = Field(DEFAULT_TSL_ACTIVATION_PERCENTAGE, ge=0, description="[If ATR TSL Disabled] Profit percentage to activate TSL")
    enable_break_even: bool = Field(DEFAULT_ENABLE_BREAK_EVEN, description="Enable moving Stop Loss to Break Even")
    break_even_trigger_atr_multiple: float = Field(DEFAULT_BE_TRIGGER_ATR_MULTIPLE, ge=0, description="Profit in ATR multiples required to trigger Break Even")
    break_even_offset_ticks: int = Field(DEFAULT_BE_OFFSET_TICKS, ge=0, description="Number of price ticks above/below entry price for Break Even SL")
    initial_stop_loss_atr_multiple: float = Field(DEFAULT_INITIAL_SL_ATR_MULTIPLE, gt=0, description="ATR multiple for initial Stop Loss distance")
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
    partial_tp_levels: Optional[List[TakeProfitLevel]] = Field(default_factory=lambda: DEFAULT_PARTIAL_TP_LEVELS.copy(), description="[S4] List of partial take profit levels (multiple, percentage). Set to null or empty list to disable.") # Use copy for default factory

    @validator('partial_tp_levels')
    def check_partial_tp_percentages(cls, v: Optional[List[TakeProfitLevel]]): # Pydantic v1 signature
        """Validate that partial TP percentages sum to approximately 1.0 if defined."""
        if v: # If list is not None and not empty
            try:
                total_percentage = sum(Decimal(str(level.percentage)) for level in v)
                # Use Decimal comparison with tolerance
                if abs(total_percentage - Decimal('1.0')) > Decimal('0.01'):
                    # Consider raising warning instead of error? Or auto-normalize?
                    # For now, raise error to force user correction.
                    raise ValueError(f"Partial TP percentages must sum to 1.0 (current sum: {total_percentage:.4f})")
            except InvalidOperation:
                 raise ValueError("Invalid numeric value found in partial TP percentages.")
        return v

class Config(BaseModel):
    """Main bot configuration."""
    interval: str = Field(DEFAULT_INTERVAL, description=f"Trading interval, must be one of {VALID_INTERVALS}")
    retry_delay: int = Field(RETRY_DELAY_SECONDS, gt=0, description="Base delay in seconds between API retry attempts")
    fetch_limit: int = Field(DEFAULT_FETCH_LIMIT, gt=50, le=1500, description="Number of historical klines to fetch initially") # Bybit max is 1500 for some endpoints
    orderbook_limit: int = Field(25, gt=0, le=200, description="Depth of order book to fetch (if needed, currently not used)") # Bybit max is 200
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
    order_type: str = Field(DEFAULT_ORDER_TYPE, regex="^(market|limit)$", description="Order type for entries ('market' or 'limit')")

    @validator('interval')
    def check_interval(cls, v: str): # Pydantic v1 signature
        """Validate interval against supported list."""
        if v not in VALID_INTERVALS:
            raise ValueError(f"Interval must be one of {VALID_INTERVALS}")
        return v

# --- Configuration Loading ---

class SensitiveFormatter(logging.Formatter):
    """Formatter that removes sensitive information from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        """Redacts API keys/secrets from the log message."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Sets up a logger with specified name and level.

    Args:
        name: The name for the logger.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured logger instance.
    """
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        # Optional: Clear existing handlers if configuration needs reset
        # for handler in logger.handlers[:]:
        #     logger.removeHandler(handler)
        # Ensure level is set correctly even if handlers exist
        logger.setLevel(level)
        return logger # Return existing logger if already configured

    logger.setLevel(level)
    logger.propagate = False # Prevent propagation to root logger

    # File Handler (DEBUG level)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # Console Handler (INFO level, colored)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO) # Log INFO and above to console
    logger.addHandler(stream_handler)

    return logger

# Initialize a base logger for early messages
base_logger = setup_logger("PyrmethusBot_Base")

def load_config(filepath: str) -> Dict[str, Any]:
    """Loads, validates, and potentially creates the configuration file.

    Args:
        filepath: Path to the configuration JSON file.

    Returns:
        A dictionary representing the validated configuration.
    """
    global config_mtime # Allow updating the global mtime
    default_config_model = Config()
    default_config_dict = default_config_model.dict() # Use .dict() for Pydantic v1

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
            config_mtime = time.time() # Set mtime to now to prevent immediate reload attempt
            return default_config_dict

    try:
        current_mtime = os.path.getmtime(filepath)
        with open(filepath, 'r', encoding="utf-8") as f:
            config_from_file = json.load(f)

        # Validate the loaded config using Pydantic
        # Pydantic automatically handles merging defaults for missing keys
        validated_config = Config(**config_from_file)
        validated_config_dict = validated_config.dict() # Use .dict() for Pydantic v1
        base_logger.info("Configuration loaded and validated successfully.")
        config_mtime = current_mtime
        return validated_config_dict

    except (json.JSONDecodeError, ValidationError) as e:
        base_logger.error(f"Config error in '{filepath}': {e}. ", exc_info=False) # Don't need full traceback for common validation errors
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
    """Initializes the CCXT Bybit exchange object.

    Args:
        logger: The logger instance to use.

    Returns:
        Initialized CCXT exchange object or None if initialization fails.
    """
    global CONFIG # Ensure access to the global config dictionary
    logger.info("Initializing CCXT Bybit exchange...")
    try:
        exchange_params = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable built-in rate limiting
            'options': {
                'defaultType': 'linear',        # Assume linear contracts (USDT margined)
                'adjustForTimeDifference': True,# Auto-adjust for time skew
                'recvWindow': 10000,           # Increase recvWindow for potentially slow networks
                # Timeouts in milliseconds
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 20000,
                'fetchOHLCVTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                # Bybit specific options (check current API docs)
                # 'brokerId': 'PyrmethusV1', # Optional: Identify bot via Broker ID if supported by Bybit
            }
        }
        exchange = ccxt.bybit(exchange_params)

        if CONFIG.get('use_sandbox', True):
            logger.warning(f"{NEON_YELLOW}SANDBOX MODE ENABLED (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            logger.warning(f"{BRIGHT}{NEON_RED}LIVE TRADING MODE ENABLED{RESET}")

        # Test connection and load markets
        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets(True) # Force reload markets on init
        logger.info(f"Markets loaded successfully for {exchange.id}.")

        # Test authentication by fetching balance
        logger.info("Fetching initial balance to verify connection...")
        # Specify account type for Bybit Unified Trading Account (UTA) if applicable
        # Possible values: CONTRACT, SPOT, UNIFIED, FUND
        # Defaulting to CONTRACT for linear perpetuals
        account_type = 'CONTRACT' # Adjust if using Spot or Unified directly
        try:
            balance = exchange.fetch_balance(params={'accountType': account_type})
            available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
            logger.info(f"{NEON_GREEN}Exchange initialized successfully. "
                        f"Account Type: {account_type}, "
                        f"Available {QUOTE_CURRENCY}: {available_quote}{RESET}")
        except ccxt.AuthenticationError as auth_err:
             logger.critical(f"{NEON_RED}Authentication failed: {auth_err}. Check API Key/Secret.{RESET}")
             return None
        except Exception as balance_err:
             logger.warning(f"{NEON_YELLOW}Could not fetch initial balance (potential connection issue or account type mismatch): {balance_err}{RESET}")
             logger.info(f"{NEON_GREEN}Exchange object created, but balance check failed. Proceeding cautiously.{RESET}")


        return exchange

    except (ccxt.AuthenticationError, ccxt.ExchangeError) as e:
        logger.critical(f"{NEON_RED}Exchange Initialization Failed: {type(e).__name__} - {e}{RESET}", exc_info=False) # Don't need full trace for auth error
        return None
    except ccxt.NetworkError as e:
        logger.critical(f"{NEON_RED}Network error during exchange initialization: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        logger.critical(f"{NEON_RED}An unexpected error occurred during exchange initialization: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching Helpers with Retries ---

def _should_retry_ccxt(exception: BaseException) -> bool:
    """Determines if a CCXT exception is retryable based on custom logic. (Used by decorator)"""
    if isinstance(exception, RETRYABLE_CCXT_EXCEPTIONS):
        # This check is technically redundant if using retry_any with retry_if_exception_type,
        # but kept here for clarity or if used standalone.
        base_logger.warning(f"Retryable CCXT error encountered (standard type): {type(exception).__name__}. Retrying...")
        return True
    # Handle specific ExchangeError cases that might be retryable (e.g., temporary server errors)
    if isinstance(exception, ccxt.ExchangeError):
        # Check if the error message indicates a temporary issue (customize as needed)
        error_str = str(exception).lower()
        # Add more keywords based on observed temporary errors
        # Example keywords: "server error", "busy", "try again later", "request timeout", "service unavailable", "internal error"
        retry_keywords = ["server error", "busy", "try again later", "request timeout", "service unavailable", "internal error"]
        if any(keyword in error_str for keyword in retry_keywords):
             base_logger.warning(f"Retryable ExchangeError encountered (keyword match): {exception}. Retrying...")
             return True
        # Example Bybit specific retryable error codes (check API docs)
        # e.g., 10006: Request Timeout, 10002: Internal Error, etc.
        # if "10006" in str(exception) or "10002" in str(exception):
        #      base_logger.warning(f"Retryable ExchangeError encountered (Bybit code match): {exception}. Retrying...")
        #      return True
    base_logger.debug(f"Non-retryable error encountered: {type(exception).__name__} - {exception}")
    return False

# Define retry decorator using the custom function and standard types
# *** FIX APPLIED HERE ***
retry_condition = retry_any(
    retry_if_exception_type(RETRYABLE_CCXT_EXCEPTIONS), # Retry on standard CCXT errors
    retry_if_exception(_should_retry_ccxt)              # Retry based on custom logic in the function
)

ccxt_retry_decorator = retry(
    stop=stop_after_attempt(MAX_API_RETRIES),
    wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_condition, # Use the combined condition
    reraise=True # Reraise the exception after retries are exhausted
)

@ccxt_retry_decorator
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the current market price for a symbol using fetch_ticker.

    Handles potential missing 'last' price by falling back to bid/ask midpoint.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol (e.g., 'BTC/USDT').
        logger: The logger instance.

    Returns:
        The current price as a Decimal, or None if fetching fails.
    """
    logger.debug(f"Fetching current price for {symbol}...")
    ticker = None # Initialize
    try:
        # Use fetch_ticker for the most recent price info
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker.get('last')

        # Prefer 'last' price if available and valid
        if last_price is not None and last_price > 0:
            price = Decimal(str(last_price))
            logger.debug(f"Fetched last price for {symbol}: {price}")
            return price

        # Fallback to mid-price (average of bid and ask)
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            bid_decimal = Decimal(str(bid))
            ask_decimal = Decimal(str(ask))
            mid_price = (bid_decimal + ask_decimal) / Decimal('2')
            logger.debug(f"Fetched mid price (bid/ask) for {symbol}: {mid_price}")
            return mid_price

        # Fallback to close price if others fail (might be slightly delayed)
        close_price = ticker.get('close')
        if close_price is not None and close_price > 0:
             price = Decimal(str(close_price))
             logger.debug(f"Fetched close price for {symbol}: {price}")
             return price

        logger.warning(f"Could not determine a valid price (last, mid, close) for {symbol}. Ticker data: {ticker}")
        return None

    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"CCXT error fetching price for {symbol}: {e}")
        raise # Reraise to allow tenacity to handle retries
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error converting price data for {symbol}: {e}. Ticker: {ticker if ticker else 'N/A'}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error fetching price for {symbol}: {e}", exc_info=True)
        raise # Reraise for potential retry or main loop handling

def optimize_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Optimizes DataFrame memory usage by downcasting numeric types.

    Args:
        df: The pandas DataFrame to optimize.
        logger: The logger instance.

    Returns:
        The optimized DataFrame.
    """
    if df.empty:
        return df
    try:
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and df[col].dtype != 'float32':
                # Downcast to float32, handling potential non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        if 'volume' in df.columns and df['volume'].dtype != 'float32':
            # Volume might need float64 for large values, but check range first
            max_vol = df['volume'].max()
            if pd.notna(max_vol) and max_vol < np.finfo(np.float32).max:
                 df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float32')
            else:
                 df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float64') # Keep float64 if needed
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if abs(start_mem - end_mem) > 0.1: # Only log if significant change
            logger.debug(f"DataFrame memory optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not optimize DataFrame memory usage: {e}")
    return df

@ccxt_retry_decorator
def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV kline data from the exchange with retries.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        timeframe: The CCXT timeframe string (e.g., '1m', '1h', '1d').
        limit: The maximum number of klines to fetch.
        logger: The logger instance.

    Returns:
        A pandas DataFrame containing the OHLCV data, sorted chronologically,
        or an empty DataFrame if fetching fails or no data is returned.
    """
    logger.debug(f"Fetching {limit} klines for {symbol} ({timeframe})...")
    ohlcv = [] # Initialize ohlcv list
    try:
        # fetch_ohlcv(symbol, timeframe, since, limit, params)
        # 'since' is typically not needed when just getting the latest 'limit' candles
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        if not ohlcv:
            logger.warning(f"No kline data returned from exchange for {symbol} ({timeframe}).")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to datetime objects (UTC) and set as index
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        except ValueError as e:
            logger.error(f"Error converting timestamp for {symbol}: {e}. First few timestamps: {df['timestamp'].head().tolist()}")
            return pd.DataFrame() # Return empty if timestamps are invalid

        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
        if df.empty:
             logger.warning(f"DataFrame empty after timestamp conversion for {symbol} ({timeframe}).")
             return df
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Data Cleaning
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Remove rows with NaN prices
        # Allow zero volume candles, but check price validity
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        df.sort_index(inplace=True) # Ensure data is chronologically sorted

        # Optimize memory usage
        df = optimize_dataframe(df.copy(), logger) # Operate on a copy

        if df.empty:
            logger.warning(f"DataFrame empty after processing for {symbol} ({timeframe}). Check fetched data quality.")
            return df

        # Trim DataFrame to maximum length
        if len(df) > MAX_DF_LEN:
             df = df.iloc[-MAX_DF_LEN:]

        logger.info(f"Fetched and processed {len(df)} klines for {symbol} ({timeframe}). "
                    f"Last timestamp: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'}")
        return df

    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"CCXT error fetching klines for {symbol} ({timeframe}): {e}")
        raise # Reraise for tenacity retry
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error processing kline data for {symbol} ({timeframe}): {e}. Data sample: {ohlcv[:5] if ohlcv else 'N/A'}")
         return pd.DataFrame() # Return empty on data processing errors
    except Exception as e:
        logger.error(f"Unexpected error fetching klines for {symbol} ({timeframe}): {e}", exc_info=True)
        raise # Reraise for potential retry

def safe_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely converts a value to Decimal, returning default on error."""
    if value is None:
        return default
    try:
        # Handle potential scientific notation strings
        if isinstance(value, str) and 'e' in value.lower():
            # Let Decimal handle scientific notation directly
            pass
        elif isinstance(value, float) and (np.isinf(value) or np.isnan(value)):
             # base_logger.debug(f"Attempted to convert invalid float ({value}) to Decimal.")
             return default # Handle infinity/NaN floats
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        # base_logger.debug(f"Failed to convert '{value}' (type: {type(value)}) to Decimal.", exc_info=True) # Optional: Debug log failed conversions
        return default

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information, using cache if available.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        logger: The logger instance.

    Returns:
        A dictionary containing market details, or None if not found or error occurs.
    """
    if symbol in market_info_cache:
        # logger.debug(f"Using cached market info for {symbol}")
        return market_info_cache[symbol]

    logger.debug(f"Fetching market info for {symbol} from exchange...")
    market = None # Initialize market variable
    try:
        # Exchange might have loaded markets already, but force can be safer if structure changes
        # Consider only forcing reload periodically, not every time cache is missed.
        # exchange.load_markets(True) # Optional: Force reload
        market = exchange.market(symbol)
        if not market:
            logger.error(f"Market info not found for symbol '{symbol}' on {exchange.id}.")
            return None

        # Determine contract type reliably
        is_contract = market.get('contract', False) or \
                      market.get('type') in ['swap', 'future'] or \
                      market.get('linear', False) or \
                      market.get('inverse', False)

        # Extract essential details, using safe_decimal for numeric conversions
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        cost_limits = limits.get('cost', {})
        precision = market.get('precision', {})

        market_details = {
            'id': market.get('id'),
            'symbol': market.get('symbol'),
            'base': market.get('base'),
            'quote': market.get('quote'),
            'active': market.get('active', False),
            'type': market.get('type'), # e.g., 'spot', 'swap', 'future'
            'linear': market.get('linear'),
            'inverse': market.get('inverse'),
            'contract': market.get('contract', False),
            'contractSize': safe_decimal(market.get('contractSize', '1'), default=Decimal('1')), # Default to 1 if missing/invalid
            'is_contract': is_contract,
            'limits': {
                'amount': {
                    'min': safe_decimal(amount_limits.get('min')), # Allow None if not present
                    'max': safe_decimal(amount_limits.get('max')),
                },
                 'price': {
                    'min': safe_decimal(price_limits.get('min')),
                    'max': safe_decimal(price_limits.get('max')),
                },
                 'cost': {
                    'min': safe_decimal(cost_limits.get('min')),
                    'max': safe_decimal(cost_limits.get('max')),
                }
            },
            'precision': {
                # Precision indicates the tick size (step size)
                'amount': safe_decimal(precision.get('amount')), # Tick size for amount
                'price': safe_decimal(precision.get('price')),      # Tick size for price
            },
            # Use safe_decimal for fees, default to reasonable values if missing
            'taker': safe_decimal(market.get('taker'), default=Decimal('0.0006')),
            'maker': safe_decimal(market.get('maker'), default=Decimal('0.0001')),
            'info': market.get('info', {}) # Raw market info from exchange
        }

        # Validate crucial precision/limits after conversion
        if market_details['precision']['amount'] is None or market_details['precision']['amount'] <= 0:
             logger.warning(f"Amount precision (tick size) for {symbol} is invalid or missing: {market_details['precision']['amount']}. Sizing/formatting may fail.")
             # Decide if this is critical - maybe return None? For now, just warn.
             # return None
        if market_details['precision']['price'] is None or market_details['precision']['price'] <= 0:
             logger.warning(f"Price precision (tick size) for {symbol} is invalid or missing: {market_details['precision']['price']}. Order placement/formatting may fail.")
             # return None
        if market_details['limits']['amount']['min'] is None:
             logger.warning(f"Minimum amount limit for {symbol} is missing. Sizing might use 0 or fail if calculated size is small.")
             # Let sizing handle potential None.

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
         # Don't cache on error
         return None
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error processing market data for {symbol}: {e}. Market data: {market if market else 'N/A'}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error getting market info for {symbol}: {e}", exc_info=True)
        return None

@ccxt_retry_decorator
def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency.

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency code (e.g., 'USDT').
        logger: The logger instance.

    Returns:
        The available balance as a Decimal, or None if fetching fails.
    """
    logger.debug(f"Fetching available balance for {currency}...")
    balance_info = {} # Initialize
    try:
        # Adjust params for Bybit account types if necessary
        # Common params: {'type': 'CONTRACT', 'accountType': 'CONTRACT'} for derivatives
        # Or {'type': 'SPOT', 'accountType': 'SPOT'} for spot
        # Or {'accountType': 'UNIFIED'} for UTA
        params = {'accountType': 'CONTRACT'} # Defaulting to contract, adjust as needed based on bot's target
        balance_info = exchange.fetch_balance(params=params)

        # Navigate the balance structure (can vary slightly between exchanges/accounts)
        # Prefer 'free' balance, fallback to specific currency entry's 'free'
        free_balance = balance_info.get('free', {}).get(currency)
        currency_data = balance_info.get(currency, {})
        available = currency_data.get('free') # Check currency-specific 'free'
        total_balance = balance_info.get('total', {}).get(currency) # Use total as last resort

        balance_value = None
        if free_balance is not None:
            balance_value = free_balance
        elif available is not None:
             balance_value = available
        elif total_balance is not None:
             logger.warning(f"Using 'total' balance for {currency} as 'free'/'available' not found directly.")
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
        raise # Reraise for retry
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error converting balance data for {currency}: {e}. Balance Info: {balance_info}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error fetching balance for {currency}: {e}", exc_info=True)
        raise # Reraise for potential retry

@ccxt_retry_decorator
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Fetches the open position for a specific symbol using fetch_positions.

    Standardizes the output format using safe_decimal.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        logger: The logger instance.

    Returns:
        A dictionary containing standardized position details, or None if no
        position exists or an error occurs.
    """
    logger.debug(f"Checking for open position for {symbol}...")
    positions = [] # Initialize
    try:
        # fetch_positions is generally better for multiple positions/unified accounts
        # It returns a list, even if only one symbol is requested.
        # Specify symbol(s) to potentially reduce response size/time
        positions = exchange.fetch_positions([symbol])

        # Filter for positions with a non-negligible size.
        # Check common fields for position size ('contracts', 'info.size', 'info.positionValue')
        # Use a small tolerance for floating point comparisons
        size_tolerance = Decimal('1e-9') # Adjust based on minimum contract size if needed
        open_positions = []
        for pos in positions:
            # Check if the symbol matches the requested symbol (case-insensitive for safety)
            if pos.get('symbol','').upper() != symbol.upper():
                 continue

            # Use safe_decimal for all numeric conversions
            size_contracts = safe_decimal(pos.get('contracts'), default=Decimal('0'))
            size_info = safe_decimal(pos.get('info', {}).get('size'), default=Decimal('0'))
            pos_value = safe_decimal(pos.get('info', {}).get('positionValue'), default=Decimal('0')) # Bybit sometimes uses this

            # Check if any size representation indicates an open position
            if abs(size_contracts) > size_tolerance or \
               abs(size_info) > size_tolerance or \
               abs(pos_value) > size_tolerance:
                 open_positions.append(pos)

        if not open_positions:
            logger.info(f"No active position found for {symbol}.")
            return None

        # Assuming the bot manages only one position per symbol in One-Way mode
        if len(open_positions) > 1:
            logger.warning(f"Multiple ({len(open_positions)}) open positions found for {symbol}. Using the first one found. Ensure exchange is in One-Way mode if managing single positions.")
            # Log details of all positions for debugging
            for i, p in enumerate(open_positions):
                 size_info = p.get('info', {}).get('size')
                 entry_info = p.get('info', {}).get('avgPrice')
                 logger.debug(f"Position {i+1} Info: Size={size_info}, Entry={entry_info}") # Log key raw info

        pos = open_positions[0]
        pos_info = pos.get('info', {}) # Raw info from the exchange

        # --- Standardize Position Details Extraction ---
        # Size: Prefer 'contracts', fallback to 'info.size'
        size = safe_decimal(pos.get('contracts'))
        if size is None:
            size = safe_decimal(pos_info.get('size'))
        if size is None: # If still None after checking both, treat as error
            logger.error(f"Could not determine position size for {symbol}. Position data: {pos}")
            return None # Cannot proceed without size

        # Entry Price: Prefer 'entryPrice', fallback to 'info.avgPrice' or 'info.entryPrice'
        entry_price = safe_decimal(pos.get('entryPrice'))
        if entry_price is None or entry_price <= 0:
            entry_price = safe_decimal(pos_info.get('avgPrice')) # Bybit often uses avgPrice
        if entry_price is None or entry_price <= 0:
            entry_price = safe_decimal(pos_info.get('entryPrice'))
        if entry_price is None or entry_price <= 0:
            logger.error(f"Could not determine valid entry price for {symbol}. Position data: {pos}")
            entry_price = Decimal('0') # Set to 0 as placeholder, but logged error

        # Side: Determined by the sign of the size
        side = 'long' if size > 0 else 'short'

        # Mark Price: Prefer 'markPrice', fallback to 'info.markPrice'
        mark_price = safe_decimal(pos.get('markPrice'))
        if mark_price is None:
            mark_price = safe_decimal(pos_info.get('markPrice'))

        # Liquidation Price: Prefer 'liquidationPrice', fallback to 'info.liqPrice'
        liq_price = safe_decimal(pos.get('liquidationPrice'))
        if liq_price is None:
            liq_price = safe_decimal(pos_info.get('liqPrice'))

        # Unrealized PnL: Prefer 'unrealizedPnl', fallback to 'info.unrealisedPnl'
        unrealized_pnl = safe_decimal(pos.get('unrealizedPnl'))
        if unrealized_pnl is None:
            unrealized_pnl = safe_decimal(pos_info.get('unrealisedPnl')) # Note spelling variation

        # Leverage: Prefer 'leverage', fallback to 'info.leverage'
        leverage = safe_decimal(pos.get('leverage'))
        if leverage is None:
            leverage = safe_decimal(pos_info.get('leverage'))

        # Timestamp: Check 'timestamp', 'datetime', or 'info' fields
        timestamp_ms = pos.get('timestamp')
        datetime_str = pos.get('datetime')
        # Bybit v5 specific creation time in milliseconds (string format)
        created_time_ms_str = pos_info.get('createdTime')

        entry_timestamp = None
        if created_time_ms_str:
            try:
                entry_timestamp = int(created_time_ms_str) # Prefer specific creation time if available
            except (ValueError, TypeError):
                 logger.warning(f"Could not parse position createdTime string: {created_time_ms_str}")
        elif timestamp_ms:
             entry_timestamp = timestamp_ms
        elif datetime_str:
             try:
                 # Attempt to parse ISO 8601 format
                 dt_obj = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                 entry_timestamp = int(dt_obj.timestamp() * 1000)
             except (TypeError, ValueError):
                 logger.warning(f"Could not parse position datetime string: {datetime_str}")

        # Protection Levels (Extract from 'info' as standard fields are often missing/inconsistent)
        # Use safe_decimal and check multiple possible keys, convert '0' or invalid to None
        sl_price = safe_decimal(pos_info.get('stopLoss')) or \
                   safe_decimal(pos.get('stopLossPrice')) or \
                   safe_decimal(pos_info.get('slPrice')) or \
                   safe_decimal(pos_info.get('stop_loss'))
        sl_price = sl_price if sl_price and sl_price > 0 else None

        tp_price = safe_decimal(pos_info.get('takeProfit')) or \
                   safe_decimal(pos.get('takeProfitPrice')) or \
                   safe_decimal(pos_info.get('tpPrice')) or \
                   safe_decimal(pos_info.get('take_profit'))
        tp_price = tp_price if tp_price and tp_price > 0 else None

        # Bybit 'trading-stop' response: 'trailingStop' is distance, 'activePrice' is activation
        tsl_distance = safe_decimal(pos_info.get('trailingStop')) # Usually the trail distance/amount
        tsl_distance = tsl_distance if tsl_distance and tsl_distance > 0 else None

        tsl_activation_price = safe_decimal(pos_info.get('activePrice')) # Activation price for Bybit TSL
        tsl_activation_price = tsl_activation_price if tsl_activation_price and tsl_activation_price > 0 else None

        position_details = {
            'symbol': pos.get('symbol', symbol), # Use input symbol as fallback
            'side': side,
            'contracts': size,
            'entryPrice': entry_price,
            'markPrice': mark_price, # Can be None
            'liquidationPrice': liq_price, # Can be None
            'unrealizedPnl': unrealized_pnl, # Can be None
            'leverage': leverage, # Can be None
            'entryTimestamp': entry_timestamp, # Milliseconds since epoch, can be None
            # Protection levels (None if not set or invalid)
            'stopLossPrice': sl_price,
            'takeProfitPrice': tp_price,
            'tslDistance': tsl_distance,
            'tslActivationPrice': tsl_activation_price,
            'info': pos_info # Keep raw info for debugging or specific needs
        }

        # Log concise summary
        log_sl = f"{position_details['stopLossPrice']:.4f}" if position_details['stopLossPrice'] else 'None'
        log_tp = f"{position_details['takeProfitPrice']:.4f}" if position_details['takeProfitPrice'] else 'None'
        log_tsl_dist = f"{position_details['tslDistance']:.4f}" if position_details['tslDistance'] else 'None'
        log_tsl_act = f"{position_details['tslActivationPrice']:.4f}" if position_details['tslActivationPrice'] else 'None'

        logger.info(f"Active {position_details['side'].upper()} position found for {symbol}: "
                    f"Size={position_details['contracts']}, Entry={position_details['entryPrice']:.4f}, "
                    f"SL={log_sl}, TP={log_tp}, "
                    f"TSL Dist={log_tsl_dist}, TSL Act={log_tsl_act}")
        logger.debug(f"Full position details: {position_details}") # Log full details at debug level
        return position_details

    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"CCXT error fetching position for {symbol}: {e}")
        raise # Reraise for retry
    except (InvalidOperation, TypeError, ValueError) as e:
         logger.error(f"Error processing position data for {symbol}: {e}. Positions data: {positions}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error fetching position for {symbol}: {e}", exc_info=True)
        raise # Reraise for potential retry


@ccxt_retry_decorator
def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a contract symbol using the standard CCXT method.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        leverage: The desired leverage (integer > 0).
        market_info: The market information dictionary.
        logger: The logger instance.

    Returns:
        True if leverage was set successfully or was already correct, False otherwise.
    """
    if not market_info.get('is_contract', False):
        logger.debug(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True
    if leverage <= 0:
        logger.error(f"Invalid leverage value ({leverage}) for {symbol}. Must be positive.")
        return False

    logger.debug(f"Setting leverage to {leverage}x for {symbol}...")
    response = None # Initialize response
    try:
        # Standard CCXT method: set_leverage(symbol, leverage, params)
        # Bybit V5 API requires buyLeverage and sellLeverage to be the same for cross margin,
        # and often requires them explicitly for isolated margin.
        # CCXT's set_leverage usually handles this via params.
        # Ensure we pass leverage for both buy and sell if needed.
        params = {
            'buyLeverage': str(leverage), # API expects string
            'sellLeverage': str(leverage)
        }
        # CCXT might automatically add category based on market type, but being explicit can help
        if 'category' not in params:
            params['category'] = 'linear' if market_info.get('linear', True) else 'inverse'

        logger.debug(f"Calling set_leverage for {symbol} with leverage={leverage}, params={params}")
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        logger.info(f"Leverage set to {leverage}x for {symbol}.")
        logger.debug(f"Leverage Response: {response}")
        return True
    except ccxt.ExchangeError as e:
        # Check for common "already set" or "no change" messages
        error_str = str(e).lower()
        # Bybit error codes: 110025=Leverage not modified; 30086=Same leverage
        # Check specific codes and general messages
        if "leverage not modified" in error_str or \
           "leverage same" in error_str or \
           "leverage not change" in error_str or \
           "110025" in str(e) or \
           "30086" in str(e):
            logger.info(f"Leverage for {symbol} already set to {leverage}x or no change needed.")
            return True # Treat as success if already set correctly
        elif "max leverage" in error_str or "leverage too high" in error_str:
             logger.error(f"Failed to set leverage for {symbol}: Requested leverage {leverage}x exceeds maximum allowed. Error: {e}")
             return False
        else:
            logger.error(f"Exchange error setting leverage for {symbol} to {leverage}x: {e}")
            logger.debug(f"Leverage setting request params: leverage={leverage}, symbol={symbol}, params={params if 'params' in locals() else 'N/A'}")
            logger.debug(f"Leverage setting response (if any): {response}")
            raise # Reraise potentially retryable errors
    except Exception as e:
        logger.error(f"Unexpected error setting leverage for {symbol} to {leverage}x: {e}", exc_info=True)
        raise # Reraise


def format_value_by_tick_size(
    value: Decimal,
    tick_size: Decimal,
    rounding_mode: str = ROUND_DOWN # Default rounding mode
) -> Decimal:
    """Formats a Decimal value according to a specific tick size using quantization.

    Args:
        value: The Decimal value to format.
        tick_size: The required step size (precision).
        rounding_mode: The rounding mode (e.g., ROUND_DOWN, ROUND_UP, ROUND_HALF_UP).

    Returns:
        The formatted Decimal value.

    Raises:
        ValueError: If tick_size is zero or negative.
        InvalidOperation: If quantization fails.
    """
    if not isinstance(value, Decimal):
        raise TypeError(f"Value must be a Decimal, got {type(value)}")
    if not isinstance(tick_size, Decimal):
         raise TypeError(f"Tick size must be a Decimal, got {type(tick_size)}")
    if tick_size <= 0:
        raise ValueError("Tick size must be positive")
    # Use quantize: (value / tick_size).quantize(Decimal('1'), rounding) * tick_size
    # This correctly handles rounding to the nearest multiple of tick_size.
    # Example: value=123.456, tick_size=0.01, ROUND_DOWN
    # (123.456 / 0.01) = 12345.6
    # quantize(Decimal('1'), ROUND_DOWN) -> 12345
    # 12345 * 0.01 = 123.45
    quantized_value = (value / tick_size).quantize(Decimal('1'), rounding=rounding_mode) * tick_size
    return quantized_value

def format_value_for_exchange(
    value: Union[Decimal, float, str], # Accept various inputs
    precision_type: str, # 'amount' or 'price'
    market_info: Dict,
    logger: logging.Logger,
    rounding_mode: str = ROUND_DOWN # Default rounding mode
) -> Optional[Decimal]:
    """Formats a value according to exchange precision (tick size).

    Converts input to Decimal before formatting.

    Args:
        value: The value to format (Decimal, float, or string).
        precision_type: 'amount' or 'price'.
        market_info: The market information dictionary containing precision tick sizes.
        logger: Logger instance.
        rounding_mode: The rounding mode (e.g., ROUND_DOWN, ROUND_UP).

    Returns:
        The formatted Decimal value, or None if precision info is missing or invalid.
    """
    symbol = market_info.get('symbol', 'UNKNOWN')
    value_decimal = safe_decimal(value) # Convert input to Decimal first
    if value_decimal is None:
        logger.error(f"Cannot format value: Input '{value}' could not be converted to Decimal for {symbol}.")
        return None

    precision_data = market_info.get('precision')
    if precision_data is None:
        logger.error(f"Precision info missing for {symbol}. Cannot format {precision_type}.")
        return None

    tick_size = precision_data.get(precision_type)
    if tick_size is None or not isinstance(tick_size, Decimal) or tick_size <= 0:
        logger.error(f"Invalid or missing '{precision_type}' precision (tick size) for {symbol}: {tick_size}. Cannot format value.")
        return None

    try:
        # Manual quantization using tick size is generally the most reliable method
        formatted_value = format_value_by_tick_size(value_decimal, tick_size, rounding_mode)
        # logger.debug(f"Formatted {precision_type} for {symbol}: Original={value_decimal}, Tick={tick_size}, Rounded={formatted_value} (Mode: {rounding_mode})")
        return formatted_value

    except (ValueError, InvalidOperation, TypeError) as e:
        logger.error(f"Error formatting {precision_type} for {symbol} using manual quantization: {e}. Value: {value_decimal}, Tick Size: {tick_size}")
        return None
    except Exception as e:
         logger.error(f"Unexpected error during formatting for {symbol}: {e}", exc_info=True)
         return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    market_info: Dict,
    config: Dict[str, Any],
    current_atr: Optional[Decimal],
    logger: logging.Logger
) -> Optional[Decimal]:
    """Calculates the position size in contracts/base currency units.

    Adjusts based on risk percentage, stop loss distance, contract size,
    market limits/precision, and optionally ATR volatility.

    Args:
        balance: Available quote currency balance.
        risk_per_trade: Fraction of balance to risk (0.0 to 1.0).
        entry_price: Planned entry price.
        stop_loss_price: Planned stop loss price.
        market_info: Market information dictionary.
        config: Bot configuration dictionary.
        current_atr: Current ATR value (optional, for ATR sizing).
        logger: Logger instance.

    Returns:
        The calculated position size as a Decimal (formatted), or None if calculation fails.
    """
    symbol = market_info.get('symbol', 'UNKNOWN')
    lg = logger

    # --- Input Validation ---
    if balance is None or balance <= 0:
        lg.error(f"Invalid balance ({balance}) for position sizing {symbol}.")
        return None
    if not (0 < risk_per_trade < 1):
        lg.error(f"Invalid risk_per_trade ({risk_per_trade}) for position sizing {symbol}.")
        return None
    if stop_loss_price is None or stop_loss_price <= 0:
        lg.error(f"Invalid stop_loss_price ({stop_loss_price}) for position sizing {symbol}.")
        return None
    if entry_price is None or entry_price <= 0:
        lg.error(f"Invalid entry_price ({entry_price}) for position sizing {symbol}.")
        return None
    if entry_price == stop_loss_price:
         lg.error(f"Entry price and stop loss price cannot be the same ({entry_price}) for {symbol}.")
         return None

    risk_per_contract_quote = Decimal('0') # Initialize
    raw_size = Decimal('0')
    try:
        # --- Risk Amount Calculation ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_price = abs(entry_price - stop_loss_price)
        if sl_distance_price <= 0:
            lg.error(f"Stop loss distance is zero or negative for {symbol}. Cannot calculate size.")
            return None

        # --- Contract Value Calculation ---
        # Assuming LINEAR contracts based on defaultType='linear'
        contract_size_base = market_info.get('contractSize', Decimal('1')) # Size of one contract in base currency (e.g., 1 for BTC/USDT)
        if contract_size_base is None or contract_size_base <= 0:
            lg.error(f"Invalid contract size ({contract_size_base}) for {symbol}. Cannot calculate size.")
            return None

        # For linear, risk per contract = SL distance * contract_size_base
        risk_per_contract_quote = sl_distance_price * contract_size_base

        if risk_per_contract_quote <= 0:
             lg.error(f"Calculated risk per contract is zero or negative ({risk_per_contract_quote}) for {symbol}.")
             return None

        # --- ATR-Based Risk Adjustment (Optional - Snippet 20) ---
        atr_risk_factor = Decimal('1.0')
        if config.get('atr_position_sizing', False) and current_atr is not None and current_atr > 0:
            # Adjust risk amount based on volatility (relative ATR)
            # Higher ATR -> smaller size, Lower ATR -> larger size for same $ risk.
            relative_atr = current_atr / entry_price if entry_price > 0 else Decimal('0')
            # Example: Reduce size by 10% if relative ATR is > 2%, increase if < 0.5%
            if relative_atr > Decimal('0.02'): # High vol condition
                atr_risk_factor = Decimal('0.9') # Reduce size by 10%
            elif relative_atr < Decimal('0.005'): # Low vol condition
                atr_risk_factor = Decimal('1.1') # Increase size by 10%
            lg.info(f"ATR Position Sizing: Relative ATR={relative_atr:.4f}, Risk Factor={atr_risk_factor:.2f}")
            # Apply factor to the risk amount
            risk_amount_quote *= atr_risk_factor

        # --- Calculate Raw Size ---
        # Size (in contracts/base units) = Total Risk Amount (Quote) / Risk per Contract (Quote)
        raw_size = risk_amount_quote / risk_per_contract_quote

        # --- Apply Limits and Precision ---
        min_amount = market_info.get('limits', {}).get('amount', {}).get('min')
        max_amount = market_info.get('limits', {}).get('amount', {}).get('max')

        # Apply min amount limit *before* precision formatting
        if min_amount is not None and raw_size < min_amount:
            lg.warning(f"Calculated size {raw_size:.8f} is below minimum ({min_amount}) for {symbol}. Adjusting to minimum.")
            raw_size = min_amount
            # Recalculate risk if size is forced to minimum
            actual_risk = raw_size * risk_per_contract_quote
            actual_risk_percent = (actual_risk / balance) * 100 if balance > 0 else Decimal('0')
            lg.warning(f"Actual risk for minimum size: {actual_risk:.4f} {QUOTE_CURRENCY} ({actual_risk_percent:.2f}% of balance)")
            # Check if minimum size risk exceeds allowed risk significantly (e.g., > 1.5x target)
            max_allowed_risk = balance * Decimal(str(risk_per_trade)) * Decimal('1.5')
            if actual_risk > max_allowed_risk:
                 lg.error(f"Risk with minimum size ({actual_risk:.4f}) significantly exceeds target risk ({risk_amount_quote:.4f}). Aborting trade.")
                 return None

        # Apply max amount limit
        if max_amount is not None and raw_size > max_amount:
             lg.warning(f"Calculated size {raw_size:.8f} exceeds maximum ({max_amount}) for {symbol}. Capping at maximum.")
             raw_size = max_amount

        # Apply amount precision (rounding DOWN to be conservative with risk/size)
        formatted_size = format_value_for_exchange(raw_size, 'amount', market_info, lg, ROUND_DOWN)

        if formatted_size is None:
            lg.error(f"Failed to format position size {raw_size:.8f} according to precision rules for {symbol}.")
            return None
        if formatted_size <= 0:
            lg.error(f"Formatted position size is zero or negative ({formatted_size}) for {symbol} after limits/precision. Raw size: {raw_size:.8f}, Min amount: {min_amount}")
            return None

        # Final check against minimum *after* formatting
        if min_amount is not None and formatted_size < min_amount:
             lg.error(f"Formatted position size {formatted_size} is below minimum {min_amount} for {symbol}, even after initial adjustment. Check precision/limits.")
             # Try formatting the minimum size itself to see if it's usable
             formatted_min = format_value_for_exchange(min_amount, 'amount', market_info, lg, ROUND_DOWN)
             if formatted_min and formatted_min == min_amount:
                  lg.warning(f"Using minimum order size {min_amount} as calculated size was too small after formatting.")
                  formatted_size = min_amount
                  # Re-check risk for this minimum size
                  actual_risk = formatted_size * risk_per_contract_quote
                  actual_risk_percent = (actual_risk / balance) * 100 if balance > 0 else Decimal('0')
                  lg.warning(f"Final risk check for minimum size: {actual_risk:.4f} {QUOTE_CURRENCY} ({actual_risk_percent:.2f}% of balance)")
                  max_allowed_risk = balance * Decimal(str(risk_per_trade)) * Decimal('1.5')
                  if actual_risk > max_allowed_risk:
                      lg.error(f"Risk with minimum size ({actual_risk:.4f}) still exceeds target risk ({risk_amount_quote:.4f}) after formatting. Aborting trade.")
                      return None
             else:
                  lg.error(f"Cannot determine a valid size meeting minimum requirements ({min_amount}). Minimum size itself may be invalid after formatting ({formatted_min}).")
                  return None

        # --- Calculate Cost ---
        # Cost = Size * Entry Price (for linear)
        estimated_cost = formatted_size * entry_price
        min_cost = market_info.get('limits', {}).get('cost', {}).get('min')
        max_cost = market_info.get('limits', {}).get('cost', {}).get('max')

        if min_cost is not None and estimated_cost < min_cost:
            lg.error(f"Estimated cost ({estimated_cost:.4f} {QUOTE_CURRENCY}) for size {formatted_size} is below minimum cost ({min_cost}). Cannot place trade.")
            return None
        if max_cost is not None and estimated_cost > max_cost:
             lg.error(f"Estimated cost ({estimated_cost:.4f} {QUOTE_CURRENCY}) for size {formatted_size} exceeds maximum cost ({max_cost}). Cannot place trade.")
             return None

        lg.info(f"Calculated Position Size for {symbol}: {formatted_size} {market_info.get('base', 'Units')}")
        lg.info(f" -> Target Risk Amount: {risk_amount_quote:.4f} {QUOTE_CURRENCY}")
        lg.info(f" -> Stop Distance: {sl_distance_price:.4f}")
        lg.info(f" -> ATR Factor: {atr_risk_factor:.2f}")
        lg.info(f" -> Estimated Cost: {estimated_cost:.4f} {QUOTE_CURRENCY}")

        return formatted_size

    except ZeroDivisionError:
        lg.error(f"Division by zero encountered during size calculation for {symbol}. "
                 f"Risk per contract: {risk_per_contract_quote}")
        return None
    except (InvalidOperation, TypeError, ValueError) as e:
         lg.error(f"Decimal or conversion error during size calculation for {symbol}: {e}")
         lg.debug(f"Inputs: balance={balance}, risk%={risk_per_trade}, entry={entry_price}, sl={stop_loss_price}, atr={current_atr}")
         return None
    except Exception as e:
        lg.error(f"Unexpected error calculating position size for {symbol}: {e}", exc_info=True)
        return None

@ccxt_retry_decorator
def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    signal: str, # "BUY" or "SELL"
    position_size: Decimal, # Expects already formatted Decimal
    market_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    reduce_only: bool = False,
    limit_price: Optional[Decimal] = None # Pass *raw* limit price if order_type is 'limit'
) -> Optional[Dict]:
    """Places a market or limit order with retries and proper parameters.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        signal: "BUY" (for long entry or short close) or "SELL" (for short entry or long close).
        position_size: The size of the order (already formatted Decimal).
        market_info: Market information dictionary.
        config: Bot configuration dictionary.
        logger: Logger instance.
        reduce_only: True if the order should only reduce an existing position.
        limit_price: The *raw* (unformatted) price for a limit order. Formatting happens here.

    Returns:
        The order dictionary returned by CCXT, or None if placing fails.
    """
    lg = logger
    side = 'buy' if signal == "BUY" else 'sell'
    # Determine order type: use config for entries, default to market for reduceOnly unless limit_price is given
    order_type = 'market'
    if not reduce_only: # Entry order
        order_type = config.get('order_type', 'market')
    # Allow limit order for reduceOnly if limit_price is explicitly provided
    if limit_price is not None:
         order_type = 'limit'

    order = None # Initialize order variable
    try:
        # Validate position size
        if position_size is None or position_size <= 0:
            lg.error(f"Invalid position size {position_size} for {symbol}. Cannot place trade.")
            return None
        # Convert formatted Decimal size to float for CCXT
        try:
             amount_float = float(position_size)
        except (TypeError, ValueError):
             lg.error(f"Invalid position size type for float conversion: {position_size} ({type(position_size)})")
             return None

        # --- Prepare Order Parameters (Bybit V5 specific) ---
        # Ref: https://bybit-exchange.github.io/docs/v5/order/create-order
        params = {
            'category': 'linear' if market_info.get('linear', True) else 'inverse',
            # positionIdx: 0=One-Way Mode, 1=Buy Hedge Mode, 2=Sell Hedge Mode
            # Assuming One-Way Mode. Bot logic needs adjustment for Hedge Mode.
            'positionIdx': 0,
        }
        if reduce_only:
            params['reduceOnly'] = True
            # For Market ReduceOnly, IOC is often required/recommended by exchanges
            if order_type == 'market':
                params['timeInForce'] = 'IOC' # ImmediateOrCancel
            # For Limit ReduceOnly, GTC is usually fine, but check exchange rules
            elif order_type == 'limit':
                 params['timeInForce'] = 'GTC' # GoodTilCanceled
        else: # Entry orders
             # Default TimeInForce for entries (GTC is common)
             params['timeInForce'] = 'GTC' # GoodTilCanceled
             # Optionally add PostOnly for limit entries if desired (maker only)
             # if order_type == 'limit': params['postOnly'] = True

        # Log the attempt clearly
        log_price_str = f"Price={limit_price}" if order_type == 'limit' and limit_price else "Market Price"
        lg.info(f"Placing {order_type.upper()} {side.upper()} order: {amount_float} {market_info.get('base', '')} on {symbol}. {log_price_str}. ReduceOnly={reduce_only}. Params={params}")

        # --- Create Order ---
        if order_type == 'limit':
            if limit_price is None or limit_price <= 0:
                lg.error(f"Limit price missing or invalid for limit order on {symbol}.")
                return None

            # Format limit price according to precision rules
            # Rounding for limit orders:
            # BUY: Round DOWN (more likely to fill, less aggressive)
            # SELL: Round UP (more likely to fill, less aggressive)
            # This aims for better fill probability by placing slightly inside the spread
            limit_rounding = ROUND_DOWN if side == 'buy' else ROUND_UP
            formatted_limit_price = format_value_for_exchange(limit_price, 'price', market_info, lg, limit_rounding)
            if formatted_limit_price is None:
                 lg.error(f"Failed to format limit price {limit_price} for {symbol}.")
                 return None
            try:
                 price_float = float(formatted_limit_price)
            except (TypeError, ValueError):
                 lg.error(f"Invalid formatted limit price type for float conversion: {formatted_limit_price}")
                 return None

            lg.info(f"Formatted Limit Price: {price_float} ({formatted_limit_price})")

            order = exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount_float,
                price=price_float,
                params=params
            )

        else: # Market order
            # Price parameter is ignored for market orders
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount_float,
                price=None, # No price for market orders
                params=params
            )

        # --- Process Response ---
        if order:
            order_id = order.get('id')
            order_status = order.get('status', 'unknown')
            filled_amount = order.get('filled', 0.0)
            avg_fill_price = order.get('average')

            # Determine log color based on status
            log_color = NEON_GREEN if order_status in ['closed', 'filled'] else NEON_YELLOW if order_status == 'open' else NEON_RED

            lg.info(f"{log_color}{order_type.capitalize()} {side} order {order_status} for {symbol}: "
                    f"ID={order_id}, Amount={order.get('amount')}, Filled={filled_amount}, AvgPrice={avg_fill_price or 'N/A'}{RESET}")
            lg.debug(f"Full Order Response: {order}")
            # Note: Market orders might fill instantly (status 'closed') or take time. Limit orders will likely be 'open'.
            return order
        else:
            # This case shouldn't happen often with CCXT, usually throws exception on failure
            lg.error(f"Failed to place {order_type} order for {symbol}. Exchange response was empty or invalid.")
            return None

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} order for {symbol}: {e}{RESET}")
        # Optionally fetch balance again here to confirm
        fetch_balance(exchange, QUOTE_CURRENCY, lg)
        return None
    except ccxt.InvalidOrder as e:
         lg.error(f"{NEON_RED}Invalid order parameters for {symbol}: {e}{RESET}")
         lg.error(f" -> Size: {position_size}, Price: {limit_price}, OrderType: {order_type}, ReduceOnly: {reduce_only}")
         lg.debug(f" -> Market Info Limits: {market_info.get('limits')}")
         lg.debug(f" -> Market Info Precision: {market_info.get('precision')}")
         return None # Don't retry invalid orders usually
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        lg.error(f"CCXT error placing {side} order for {symbol}: {e}")
        # Log details that might help diagnose (e.g., rate limits, margin checks)
        error_str_lower = str(e).lower()
        if "margin" in error_str_lower or "insufficient balance" in error_str_lower:
             lg.warning("Order failed possibly due to margin/balance requirements. Check leverage and available balance.")
        elif "order cost" in error_str_lower or "min order value" in error_str_lower:
             lg.warning("Order failed possibly due to cost limits. Check position size and price against market limits.")
        elif "order price" in error_str_lower or "invalid price" in error_str_lower:
            lg.warning("Order failed possibly due to price limits or formatting. Check limit price against market limits and precision.")
        # Log the order details that failed
        lg.debug(f"Failed Order Details: Size={position_size}, Price={limit_price}, Type={order_type}, Reduce={reduce_only}")
        lg.debug(f"Failed Order Response (if available in exception): {order}")
        raise # Reraise potentially retryable errors
    except (InvalidOperation, TypeError, ValueError) as e:
         lg.error(f"Data conversion error during trade placement for {symbol}: {e}")
         return None
    except Exception as e:
        lg.error(f"Unexpected error placing {side} order for {symbol}: {e}", exc_info=True)
        raise # Reraise

@ccxt_retry_decorator
def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None, # Expects raw Decimal
    take_profit_price: Optional[Decimal] = None, # Expects raw Decimal
    trailing_stop_distance: Optional[Decimal] = None, # Expects raw Decimal distance
    tsl_activation_price: Optional[Decimal] = None, # Expects raw Decimal
    position_idx: int = 0 # 0 for One-Way, 1/2 for Hedge
) -> bool:
    """Sets or updates SL, TP, and/or TSL for an existing position using Bybit's API.

    Formats the raw Decimal prices/distances according to market precision before sending.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol.
        market_info: Market information dictionary (must include 'last_price' for rounding hints).
        logger: Logger instance.
        stop_loss_price: The desired raw stop loss price (Decimal).
        take_profit_price: The desired raw take profit price (Decimal).
        trailing_stop_distance: The desired raw trailing stop distance (Decimal).
        tsl_activation_price: The desired raw TSL activation price (Decimal).
        position_idx: Position index (0 for One-Way, 1 for Buy Hedge, 2 for Sell Hedge).

    Returns:
        True if the protection was set successfully, False otherwise.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol}: Not a contract market.")
        return False # Indicate not applicable, not necessarily failure

    # Prepare parameters for Bybit's v5/position/trading-stop endpoint
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    params = {
        'category': category,
        'symbol': market_info['id'], # Use the exchange-specific ID
        'positionIdx': position_idx,
        # --- Optional parameters (Defaults usually OK, but specify if needed) ---
        # 'tpslMode': 'Full', # 'Full' or 'Partial'. Default is 'Full' (controls if TP/SL apply to whole position)
        # 'tpTriggerBy': 'LastPrice', # 'LastPrice', 'MarkPrice', 'IndexPrice' (Default: LastPrice)
        # 'slTriggerBy': 'LastPrice', # 'LastPrice', 'MarkPrice', 'IndexPrice' (Default: LastPrice)
        # 'tpOrderType': 'Market', # 'Market' or 'Limit' (Default: Market)
        # 'slOrderType': 'Market', # 'Market' or 'Limit' (Default: Market)
    }
    log_parts = []
    something_to_set = False
    current_price = market_info.get('last_price') # Used for rounding hints

    # Format and add protection levels to params if provided
    # Setting a value to '0' or "" often cancels it on Bybit.
    if stop_loss_price is not None:
        if stop_loss_price > 0:
            # Use appropriate rounding: AWAY from position for SL
            # Heuristic: if current price known, round away from it, else use entry/default
            sl_rounding = ROUND_DOWN if current_price and stop_loss_price < current_price else ROUND_UP
            sl_formatted = format_value_for_exchange(stop_loss_price, 'price', market_info, lg, sl_rounding)
            if sl_formatted:
                params['stopLoss'] = str(sl_formatted) # API expects string
                log_parts.append(f"SL={sl_formatted}")
                something_to_set = True
            else:
                lg.error(f"Invalid SL price formatting for {symbol}: {stop_loss_price}. Cannot set SL.")
                # Decide whether to fail or continue without SL
                # return False # Fail early if formatting fails
        else: # Cancel SL if price is 0 or less (or None was passed implicitly meaning cancel)
            params['stopLoss'] = '0'
            log_parts.append("SL=Cancel")
            something_to_set = True


    if take_profit_price is not None:
        if take_profit_price > 0:
            # Use appropriate rounding: TOWARDS target for TP
            tp_rounding = ROUND_UP if current_price and take_profit_price > current_price else ROUND_DOWN
            tp_formatted = format_value_for_exchange(take_profit_price, 'price', market_info, lg, tp_rounding)
            if tp_formatted:
                params['takeProfit'] = str(tp_formatted)
                log_parts.append(f"TP={tp_formatted}")
                something_to_set = True
            else:
                lg.error(f"Invalid TP price formatting for {symbol}: {take_profit_price}. Cannot set TP.")
        else: # Cancel TP
             params['takeProfit'] = '0'
             log_parts.append("TP=Cancel")
             something_to_set = True


    # Trailing Stop requires distance (and optionally activation price)
    # Note: Bybit's API behavior for setting TSL can be tricky.
    # Setting just 'trailingStop' might require 'activePrice' too.
    # Setting 'trailingStop' to '0' cancels it.
    if trailing_stop_distance is not None:
        if trailing_stop_distance > 0:
            # Format distance (usually treated like a price difference, use price precision/tick size)
            # Rounding for distance: ROUND_DOWN is often safer (smaller distance)
            tsl_dist_formatted = format_value_for_exchange(trailing_stop_distance, 'price', market_info, lg, ROUND_DOWN) # Use price tick size

            if tsl_dist_formatted and tsl_dist_formatted > 0:
                params['trailingStop'] = str(tsl_dist_formatted)
                log_parts.append(f"TSL_Dist={tsl_dist_formatted}")
                something_to_set = True

                # Add activation price if provided and valid
                if tsl_activation_price is not None and tsl_activation_price > 0:
                    # Round activation price appropriately
                    act_rounding = ROUND_UP if current_price and tsl_activation_price > current_price else ROUND_DOWN
                    tsl_act_formatted = format_value_for_exchange(tsl_activation_price, 'price', market_info, lg, act_rounding)
                    if tsl_act_formatted:
                         params['activePrice'] = str(tsl_act_formatted)
                         log_parts.append(f"TSL_Act={tsl_act_formatted}")
                    else:
                         lg.warning(f"Invalid TSL activation price formatting for {symbol}: {tsl_activation_price}. Setting TSL distance without activation price.")
                         # Remove activation price if formatting failed but distance is valid
                         if 'activePrice' in params: del params['activePrice']
                # else:
                     # lg.debug("TSL distance set without specific activation price (exchange might use default activation).")

            else:
                lg.error(f"Invalid TSL distance formatting or zero value for {symbol}: Raw={trailing_stop_distance}, Formatted={tsl_dist_formatted}. Cannot set TSL.")
                # Remove TSL params if formatting failed
                if 'trailingStop' in params: del params['trailingStop']
                if 'activePrice' in params: del params['activePrice']
        else: # Cancel TSL
             params['trailingStop'] = '0'
             # Bybit might require activePrice also set to 0 to cancel if it was previously set
             params['activePrice'] = '0'
             log_parts.append("TSL=Cancel")
             something_to_set = True


    # Only call API if there's something valid to set
    if not something_to_set:
        lg.info(f"No valid protection levels provided or formatting failed for {symbol}. No API call made.")
        return False # Indicate nothing was set due to invalid inputs

    response = None # Initialize response variable
    try:
        lg.info(f"Setting/Updating protection for {symbol}: {', '.join(log_parts)}")
        lg.debug(f"Calling v5/position/trading-stop with params: {params}")

        # Use the specific CCXT private method if available, otherwise construct the call
        # Check if the specific method exists (safer against CCXT updates)
        method_name_v5 = 'private_post_v5_position_trading_stop'
        method_name_unified = 'private_post_position_trading_stop' # Older/alternative name

        if hasattr(exchange, method_name_v5):
            response = getattr(exchange, method_name_v5)(params)
        elif hasattr(exchange, method_name_unified):
             response = getattr(exchange, method_name_unified)(params)
        else:
             # Fallback: Manually build and send the request if method not exposed (more fragile)
             # path = 'v5/position/trading-stop' # Check Bybit docs for correct path
             # response = exchange.request(path, 'private', 'POST', params) # Use generic request
             lg.error(f"Could not find appropriate CCXT method ({method_name_v5} or {method_name_unified}) for Bybit's set trading stop. Update CCXT or check API mapping.")
             return False

        lg.debug(f"Set protection response for {symbol}: {response}") # Log raw response for debugging

        # Basic check on response (Bybit v5 usually returns retCode=0 on success)
        if isinstance(response, dict) and response.get('retCode') == 0:
             lg.info(f"{NEON_GREEN}Protection successfully set/updated for {symbol}.{RESET}")
             return True
        else:
             # Log the error message from the exchange response
             error_code = response.get('retCode', 'N/A')
             error_msg = response.get('retMsg', 'Unknown error')
             lg.error(f"Failed to set protection for {symbol}. Exchange Response: Code={error_code}, Msg='{error_msg}'")
             lg.debug(f"Failed Params: {params}")
             # Specific error handling based on Bybit error codes:
             # https://bybit-exchange.github.io/docs/v5/error_code
             if error_code == 110044: # TP price error
                 lg.error("Take Profit price condition not met (e.g., too close to current price or invalid).")
             elif error_code == 110045: # SL price error
                 lg.error("Stop Loss price condition not met (e.g., too close to current price or invalid).")
             elif error_code == 110059: # TSL activation price error
                  lg.error("Trailing Stop activation price condition not met (e.g., invalid relative to current price).")
             elif error_code == 110043: # Position idx error
                  lg.error("Position index error. Check if Hedge Mode is active and positionIdx is correct.")
             # Add more specific codes as needed
             return False

    except ccxt.InvalidOrder as e:
         # This might catch formatting/parameter errors not caught by Bybit codes
         lg.error(f"Invalid order/parameters when setting protection for {symbol}: {e}")
         lg.debug(f"Protection Params: {params}")
         lg.debug(f"Protection Response: {response}")
         return False # Typically not retryable
    except (ccxt.ExchangeError, ccxt.NetworkError) as e:
        lg.error(f"CCXT error setting protection for {symbol}: {e}")
        lg.debug(f"Protection Params: {params}")
        lg.debug(f"Protection Response: {response}")
        # Check for specific retryable conditions within the error message if needed
        if "order not found or too late" in str(e).lower() or "position status is not normal" in str(e).lower():
             lg.warning("Protection setting failed likely because position closed or order filled concurrently.")
             return False # Don't retry if position state changed
        raise # Reraise potentially retryable errors
    except Exception as e:
        lg.error(f"Unexpected error setting protection for {symbol}: {e}", exc_info=True)
        lg.debug(f"Protection Params: {params}")
        lg.debug(f"Protection Response: {response}")
        raise # Reraise


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    current_atr: Optional[Decimal],
    current_price: Optional[Decimal] # Pass current price for calculations
) -> bool:
    """Calculates TSL parameters and calls _set_position_protection.

    Handles ATR-based TSL, fixed callback TSL, activation delays, and dynamic adjustments.

    Args:
        exchange: CCXT exchange instance.
        symbol: Market symbol.
        market_info: Market information.
        position_info: Current open position details dictionary.
        config: Bot configuration.
        logger: Logger instance.
        current_atr: Current ATR value.
        current_price: Current market price.

    Returns:
        True if TSL (and potentially TP/SL) was successfully set/updated, False otherwise.
    """
    lg = logger
    protection_cfg = config.get("protection", {})
    if not protection_cfg.get("enable_trailing_stop", False):
        lg.debug("Trailing stop is disabled in config.")
        return False # Return False as TSL wasn't set (though not an error)

    side = position_info.get('side')
    entry_price = position_info.get('entryPrice')

    if side not in ['long', 'short'] or entry_price is None or entry_price <= 0:
        lg.error(f"Invalid position info for TSL setting: Side={side}, Entry={entry_price}")
        return False
    if current_price is None or current_price <= 0:
         lg.warning("Cannot calculate TSL parameters without a valid current price.")
         return False

    # --- Check TSL Activation Delay (Snippet 9) ---
    tsl_activation_delay_atr = safe_decimal(protection_cfg.get("tsl_activation_delay_atr", 0.5), default=Decimal('0.5'))
    if tsl_activation_delay_atr > 0:
        if current_atr is None or current_atr <= 0:
             lg.warning("Cannot check TSL activation delay: ATR is unavailable.")
             # Proceed without delay check or return False? Proceed cautiously.
        else:
            profit = (current_price - entry_price) if side == 'long' else (entry_price - current_price)
            required_profit_for_activation = current_atr * tsl_activation_delay_atr
            if profit < required_profit_for_activation:
                lg.info(f"TSL activation delayed for {symbol}: Profit {profit:.4f} < Required {required_profit_for_activation:.4f} ({tsl_activation_delay_atr}x ATR)")
                return False # Not enough profit yet, TSL not set this cycle

    # --- Calculate TSL Distance and Activation Price (Raw) ---
    distance_raw: Optional[Decimal] = None
    activation_price_raw: Optional[Decimal] = None

    # Snippet 12: ATR-Based Trailing Stop
    use_atr_tsl = protection_cfg.get("use_atr_trailing_stop", True)
    if use_atr_tsl and current_atr is not None and current_atr > 0:
        trailing_atr_multiple = safe_decimal(protection_cfg.get("trailing_stop_atr_multiple", 1.5), default=Decimal('1.5'))
        distance_raw = current_atr * trailing_atr_multiple

        # Activation Price: Use percentage from config for activation trigger point
        activation_pct = safe_decimal(protection_cfg.get("trailing_stop_activation_percentage", 0.003), default=Decimal('0.003'))
        activation_offset = entry_price * activation_pct
        activation_price_raw = entry_price + activation_offset if side == 'long' else entry_price - activation_offset
        lg.info(f"Using ATR-based TSL for {symbol}: Raw Dist={distance_raw:.4f} ({trailing_atr_multiple}x ATR), Raw Act Price={activation_price_raw:.4f} ({activation_pct*100}%)")

    else: # Fixed Percentage Callback Rate TSL
        if use_atr_tsl: # Log if ATR TSL was desired but ATR failed
             lg.warning("ATR TSL enabled but ATR unavailable. Falling back to percentage-based TSL.")
        callback_rate = safe_decimal(protection_cfg.get("trailing_stop_callback_rate", 0.005), default=Decimal('0.005'))
        activation_pct = safe_decimal(protection_cfg.get("trailing_stop_activation_percentage", 0.003), default=Decimal('0.003'))

        # Bybit 'trailingStop' expects distance in price points. Calculate from rate.
        # Using current price might be more reactive than entry price for distance calc.
        distance_raw = current_price * callback_rate
        activation_offset = entry_price * activation_pct
        activation_price_raw = entry_price + activation_offset if side == 'long' else entry_price - activation_offset
        lg.info(f"Using Percentage-based TSL for {symbol}: Raw Dist={distance_raw:.4f} ({callback_rate*100}% of current price), Raw Act Price={activation_price_raw:.4f} ({activation_pct*100}% of entry)")

    # --- Dynamic TSL Callback Adjustment (Snippet 22) ---
    if protection_cfg.get("dynamic_tsl_callback", True) and distance_raw is not None and current_atr is not None and current_atr > 0:
        profit = (current_price - entry_price) if side == 'long' else (entry_price - current_price)
        profit_atr = profit / current_atr if current_atr > 0 else Decimal('0')
        # Example: Tighten TSL distance if profit > 2x ATR
        if profit_atr > Decimal('2.0'):
            tighten_factor = Decimal('0.75') # Tighten distance by 25%
            original_distance = distance_raw
            distance_raw *= tighten_factor
            lg.info(f"Dynamic TSL: Profit {profit_atr:.2f}x ATR > 2.0. Tightening Raw TSL distance from {original_distance:.4f} to {distance_raw:.4f}")
        # Example: Loosen slightly if profit barely positive? (More complex)


    # --- Validation and Final Checks ---
    if distance_raw is None or distance_raw <= 0:
        lg.error(f"Calculated invalid raw TSL distance ({distance_raw}) for {symbol}. Cannot set TSL.")
        return False
    if activation_price_raw is None or activation_price_raw <= 0:
         lg.error(f"Calculated invalid raw TSL activation price ({activation_price_raw}) for {symbol}. Cannot set TSL.")
         return False

    # Ensure TSL activation price is valid relative to current price (required by Bybit)
    # Activation price must be better than the current price for the direction (more profit needed).
    # Add a small buffer using tick size.
    min_tick = market_info.get('precision', {}).get('price', Decimal('0.0001'))
    if min_tick is None or min_tick <= 0: # Safety check for tick size
         lg.warning(f"Cannot validate TSL activation price vs current price: Invalid price tick size ({min_tick}).")
    elif side == 'long' and activation_price_raw <= current_price:
        lg.warning(f"Raw TSL Activation Price ({activation_price_raw:.4f}) for LONG is not above current price ({current_price:.4f}). Adjusting slightly above current.")
        activation_price_raw = current_price + min_tick # Set slightly above current
    elif side == 'short' and activation_price_raw >= current_price:
        lg.warning(f"Raw TSL Activation Price ({activation_price_raw:.4f}) for SHORT is not below current price ({current_price:.4f}). Adjusting slightly below current.")
        activation_price_raw = current_price - min_tick # Set slightly below current

    # --- Call the Protection Setting Function ---
    # Preserve existing SL/TP if they exist and aren't being explicitly overridden
    # Fetch them from the position_info dictionary passed in
    current_sl_raw = position_info.get('stopLossPrice')
    current_tp_raw = position_info.get('takeProfitPrice')
    position_idx = position_info.get('info', {}).get('positionIdx', 0)

    lg.debug(f"Calling _set_position_protection with: SL={current_sl_raw}, TP={current_tp_raw}, TSL_Dist={distance_raw}, TSL_Act={activation_price_raw}")
    # Pass current market price to _set_position_protection for better rounding heuristics
    market_info['last_price'] = current_price
    return _set_position_protection(
        exchange, symbol, market_info, lg,
        stop_loss_price=current_sl_raw, # Preserve existing SL (raw)
        take_profit_price=current_tp_raw, # Preserve existing TP (raw)
        trailing_stop_distance=distance_raw, # Pass raw distance
        tsl_activation_price=activation_price_raw, # Pass raw activation price
        position_idx=position_idx
    )


# --- Strategy Implementation ---

class HigherTimeframeAnalysis(TypedDict):
    """Structure for higher timeframe analysis results."""
    current_trend_up: Optional[bool]
    # Add other relevant HTF info if needed (e.g., HTF OBs, HTF RSI)

class StrategyAnalysisResults(TypedDict):
    """Structure for returning analysis results from the strategy."""
    dataframe: pd.DataFrame
    last_close: Optional[Decimal]
    last_high: Optional[Decimal] # Added for potential use
    last_low: Optional[Decimal]  # Added for potential use
    current_trend_up: Optional[bool] # True for Up, False for Down, None for Unknown
    trend_just_changed: bool
    active_bull_boxes: List[Dict]
    active_bear_boxes: List[Dict]
    vol_norm_int: Optional[int] # Volume Percentile Rank (0-100)
    atr: Optional[Decimal]
    upper_band: Optional[Decimal] # Volumatic Upper Band
    lower_band: Optional[Decimal] # Volumatic Lower Band
    # Fields for enhancement snippets
    adx: Optional[Decimal]
    rsi: Optional[Decimal]
    vwap: Optional[Decimal]
    roc: Optional[Decimal]
    higher_tf_analysis: Optional[HigherTimeframeAnalysis] # For Snippet 19

class VolumaticOBStrategy:
    """Implements the Volumatic Trend + Pivot Order Block strategy with enhancements.

    Attributes:
        config (Dict): The bot's configuration dictionary.
        logger (logging.Logger): Logger instance for this strategy.
        market_info (Dict): Market information for the symbol.
        exchange (Optional[ccxt.Exchange]): CCXT exchange instance (for HTF analysis).
        strategy_params (StrategyParams): Pydantic model with strategy parameters.
        protection_config (ProtectionConfig): Pydantic model with protection parameters.
        bull_boxes (List[Dict]): List of identified bullish order blocks.
        bear_boxes (List[Dict]): List of identified bearish order blocks.
        min_data_len (int): Minimum number of klines needed for calculations.
    """
    def __init__(self, config: Dict[str, Any], market_info: Dict, logger: logging.Logger, exchange: Optional[ccxt.Exchange] = None):
        self.config = config
        self.logger = logger
        self.market_info = market_info
        self.exchange = exchange # Required for Snippet 19 (Multi-TF Confirm)
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')

        try:
            self.strategy_params = StrategyParams(**config.get("strategy_params", {}))
            self.protection_config = ProtectionConfig(**config.get("protection", {}))
        except ValidationError as e:
             self.logger.error(f"Strategy/Protection Pydantic validation failed: {e}. Using defaults.")
             # Fallback to default models if config section is malformed
             self.strategy_params = StrategyParams()
             self.protection_config = ProtectionConfig()

        # Cache OBs - these persist across calls to update()
        self.bull_boxes: List[Dict[str, Any]] = []
        self.bear_boxes: List[Dict[str, Any]] = []

        # Calculate minimum data length needed based on longest lookback period across all indicators
        self.min_data_len = self._calculate_min_data_length()
        self.logger.debug(f"Minimum data length required for indicators: {self.min_data_len}")

    def _calculate_min_data_length(self) -> int:
        """Determines the minimum number of candles required based on indicator lookbacks."""
        lookbacks = [
            self.strategy_params.vt_length,
            self.strategy_params.vt_atr_period,
            self.strategy_params.vt_vol_ema_length, # Window for volume percentile rank
            self.strategy_params.ph_left + self.strategy_params.ph_right, # Pivot lookback combines left/right
            self.strategy_params.pl_left + self.strategy_params.pl_right,
            14, # Default ADX lookback
            14, # Default RSI lookback
            self.strategy_params.roc_length, # ROC lookback
            20, # Rolling window for VWAP (pandas_ta default)
            100, # Rolling window for ATR percentile (Snippet 2) - Adjust if needed
            20,  # Rolling window for ATR std dev (Snippet 5) - Adjust if needed
            int(self.strategy_params.ob_expiry_atr_periods or 0) + 200 # If expiry used, need data for mean ATR calc
            # Add lookbacks from other indicators if used
        ]
        # Buffer for stability and calculations needing prior data points
        pivot_buffer = max(self.strategy_params.ph_right, self.strategy_params.pl_right) + 5
        buffer = max(50, pivot_buffer)
        return max(lookbacks) + buffer

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all necessary indicators and adds them to the DataFrame.

        Args:
            df: Input DataFrame with OHLCV data.

        Returns:
            DataFrame with calculated indicators, or original df if input is insufficient.
        """
        if df.empty or len(df) < 5: # Need at least a few rows for basic calculations
            self.logger.warning("DataFrame too small for indicator calculation.")
            return df

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Missing required columns in DataFrame for indicator calculation. Need: {required_cols}")
            return df

        df_out = df.copy() # Work on a copy

        # --- Core Volumatic Trend ---
        try:
            # Calculate ATR first as it's used in bands
            df_out['atr'] = ta.atr(df_out['high'], df_out['low'], df_out['close'], length=self.strategy_params.vt_atr_period, fillna=False)

            # Trend Definition: Price vs EMA
            df_out['vt_ema'] = ta.ema(df_out['close'], length=self.strategy_params.vt_length, fillna=False)
            # Trend confirmed on the *next* bar's open (use shifted comparison)
            # True if previous close was >= previous EMA
            df_out['trend_up'] = (df_out['close'].shift(1) >= df_out['vt_ema'].shift(1))
            # Handle initial NaNs in trend: backfill seems reasonable
            df_out['trend_up'].fillna(method='bfill', inplace=True)
            # Ensure boolean type after fillna
            df_out['trend_up'] = df_out['trend_up'].astype(bool)


            # Detect trend changes (change happens on the current bar compared to the previous)
            df_out['trend_changed'] = (df_out['trend_up'] != df_out['trend_up'].shift(1))
            # First row's trend_changed should be False (no previous trend to compare)
            if not df_out.empty:
                df_out['trend_changed'].iloc[0] = False
            df_out['trend_changed'].fillna(False, inplace=True)


            # Volumatic Bands (Based on EMA and ATR *at the time of the trend change*)
            # Find the EMA and ATR values on the bars *where the trend changed*
            df_out['ema_at_change'] = np.where(df_out['trend_changed'], df_out['vt_ema'], np.nan)
            df_out['atr_at_change'] = np.where(df_out['trend_changed'], df_out['atr'], np.nan)
            # Forward fill these values to apply them until the next trend change
            df_out['ema_for_bands'] = df_out['ema_at_change'].ffill()
            df_out['atr_for_bands'] = df_out['atr_at_change'].ffill()

            # Calculate bands using the forward-filled values and safe_decimal
            atr_multiplier = safe_decimal(self.strategy_params.vt_atr_multiplier, Decimal('3.0'))
            # Ensure ema_for_bands and atr_for_bands are numeric before calculation
            ema_numeric = pd.to_numeric(df_out['ema_for_bands'], errors='coerce')
            atr_numeric = pd.to_numeric(df_out['atr_for_bands'], errors='coerce')
            df_out['upper_band'] = ema_numeric + (atr_numeric * atr_multiplier)
            df_out['lower_band'] = ema_numeric - (atr_numeric * atr_multiplier)


            # Volume Normalization (Percentile Rank over lookback period)
            vol_norm_len = self.strategy_params.vt_vol_ema_length
            # Calculate rolling percentile of volume (rank pct=True gives percentile rank 0-1)
            df_out['vol_norm'] = df_out['volume'].rolling(window=vol_norm_len, min_periods=max(1, vol_norm_len // 2)).rank(pct=True) * 100
            df_out['vol_norm'].fillna(0, inplace=True) # Fill initial NaNs with 0 percentile
            df_out['vol_norm_int'] = df_out['vol_norm'].round().astype(int) # Convert to integer 0-100

        except Exception as e:
             self.logger.error(f"Error calculating core Volumatic indicators: {e}", exc_info=True)
             # Return original df if core indicators fail
             return df

        # --- Additional Indicators for Snippets ---
        try:
            # Snippet 3: ADX
            adx_df = ta.adx(df_out['high'], df_out['low'], df_out['close'], length=14, fillna=False) # Use default length 14
            if adx_df is not None and not adx_df.empty and 'ADX_14' in adx_df.columns:
                df_out['adx'] = adx_df['ADX_14']
            else:
                df_out['adx'] = np.nan
                self.logger.debug("ADX calculation failed or returned empty.")

            # Snippet 7: RSI
            df_out['rsi'] = ta.rsi(df_out['close'], length=14, fillna=False) # Use default length 14

            # Snippet 11: Candlestick Patterns
            if self.strategy_params.candlestick_confirm:
                # Using pandas_ta for pattern recognition
                try:
                    # Calculate Bullish/Bearish Engulfing
                    eng_df = ta.cdl_pattern(df_out['open'], df_out['high'], df_out['low'], df_out['close'], name="engulfing")
                    if eng_df is not None and 'CDL_ENGULFING' in eng_df.columns:
                        df_out['bull_engulfing'] = eng_df['CDL_ENGULFING'] == 100
                        df_out['bear_engulfing'] = eng_df['CDL_ENGULFING'] == -100
                    else:
                        df_out['bull_engulfing'] = False
                        df_out['bear_engulfing'] = False
                    # Add other patterns as needed: ta.cdl_pattern(..., name="doji"), ta.cdl_hammer(), etc.
                except Exception as candle_e:
                    self.logger.warning(f"Could not calculate candlestick patterns: {candle_e}")
                    df_out['bull_engulfing'] = False
                    df_out['bear_engulfing'] = False

            # Snippet 15: VWAP
            # Simple rolling VWAP using pandas_ta:
            df_out['vwap'] = ta.vwap(df_out['high'], df_out['low'], df_out['close'], df_out['volume'], fillna=False)

            # Snippet 23: ROC
            df_out['roc'] = ta.roc(df_out['close'], length=self.strategy_params.roc_length, fillna=False)

        except Exception as e:
             self.logger.warning(f"Error calculating enhancement indicators: {e}", exc_info=True)
             # Continue with potentially missing indicators

        # --- Final Cleanup ---
        # Drop initial rows where key indicators are NaN due to lookback periods
        required_indicators = ['vt_ema', 'atr', 'upper_band', 'lower_band', 'trend_up']
        # Add snippet indicators if they are required by the logic activated in config
        if self.strategy_params.adx_threshold > 0: required_indicators.append('adx')
        if self.strategy_params.rsi_confirm: required_indicators.append('rsi')
        if self.strategy_params.vwap_confirm: required_indicators.append('vwap')
        if self.strategy_params.momentum_confirm: required_indicators.append('roc')

        # Check which required indicators actually exist in the dataframe
        available_required = [ind for ind in required_indicators if ind in df_out.columns]

        if not available_required:
             self.logger.error("No key indicators could be calculated. Returning original data.")
             return df

        # Drop rows if *any* of the essential available indicators are NaN
        initial_len = len(df_out)
        df_out.dropna(subset=available_required, how='any', inplace=True) # Drop if any required is NaN
        dropped_rows = initial_len - len(df_out)
        if dropped_rows > 0:
            self.logger.debug(f"Dropped {dropped_rows} initial rows due to NaN in required indicators.")

        if df_out.empty:
             self.logger.warning("DataFrame became empty after dropping NaN indicator rows.")

        return df_out


    def _identify_order_blocks(self, df: pd.DataFrame):
        """Identifies Pivot High/Low based Order Blocks and manages their state.

        Updates the internal self.bull_boxes and self.bear_boxes lists.

        Args:
            df: DataFrame with OHLCV and indicator data (including ATR).
        """
        if df.empty or 'high' not in df.columns or 'low' not in df.columns:
            self.logger.warning("Insufficient data or columns for OB calculation.")
            return

        min_pivot_len = max(self.strategy_params.ph_left + self.strategy_params.ph_right,
                            self.strategy_params.pl_left + self.strategy_params.pl_right) + 2
        if len(df) < min_pivot_len:
            self.logger.warning(f"Not enough data ({len(df)}) for pivot/OB calculation (requires ~{min_pivot_len}).")
            return

        # Determine source series for pivots based on config
        try:
            if self.strategy_params.ob_source == "Wicks":
                high_series = df['high']
                low_series = df['low']
            elif self.strategy_params.ob_source == "Bodys":
                high_series = df[['open', 'close']].max(axis=1)
                low_series = df[['open', 'close']].min(axis=1)
            else:
                self.logger.error(f"Invalid ob_source '{self.strategy_params.ob_source}'. Defaulting to Wicks.")
                high_series = df['high']
                low_series = df['low']
        except KeyError:
             self.logger.error("Missing open/close columns for 'Bodys' OB source.")
             return

        # --- Calculate Pivot Points using pandas_ta ---
        # `ta.pivot` identifies the candle *after* the pivot is confirmed (right lookback bars later)
        # The value returned is the price of the actual pivot candle.
        try:
            # Use fillna=False to avoid forward filling pivot signals incorrectly
            df['ph_signal'] = ta.pivot(high_series, left=self.strategy_params.ph_left, right=self.strategy_params.ph_right, high_low='high', fillna=False)
            df['pl_signal'] = ta.pivot(low_series, left=self.strategy_params.pl_left, right=self.strategy_params.pl_right, high_low='low', fillna=False)
        except Exception as e:
            self.logger.error(f"Error calculating pivots: {e}", exc_info=True)
            return # Cannot proceed without pivots

        # --- Iterate through recent potential pivots to identify new OBs ---
        # Look back enough candles to cover the right lookback period + buffer
        potential_pivot_range = max(self.strategy_params.ph_right, self.strategy_params.pl_right) + 5
        # Use .iloc for positional slicing robustness, handle case where df is shorter
        start_iloc = max(0, len(df) - potential_pivot_range)
        recent_df = df.iloc[start_iloc:]

        current_atr = safe_decimal(df['atr'].iloc[-1]) if 'atr' in df.columns and not df['atr'].empty and pd.notna(df['atr'].iloc[-1]) else Decimal('0')

        new_bear_indices = set() # Track newly added OBs in this cycle
        new_bull_indices = set()

        for signal_idx_dt in recent_df.index:
            # Check if a pivot signal exists at this index
            is_ph_signal = pd.notna(recent_df.loc[signal_idx_dt, 'ph_signal'])
            is_pl_signal = pd.notna(recent_df.loc[signal_idx_dt, 'pl_signal'])

            # --- Bearish OB (from Pivot High) ---
            if is_ph_signal:
                # The actual pivot candle index is `ph_right` bars *before* the signal index
                try:
                    signal_idx_loc = df.index.get_loc(signal_idx_dt)
                    pivot_candle_loc = signal_idx_loc - self.strategy_params.ph_right
                    if pivot_candle_loc >= 0:
                        pivot_candle_idx_dt = df.index[pivot_candle_loc]
                        # Check if this OB pivot index is already processed or exists
                        if pivot_candle_idx_dt not in new_bear_indices and not any(b['left_idx'] == pivot_candle_idx_dt for b in self.bear_boxes):
                            ob_candle = df.loc[pivot_candle_idx_dt]

                            # Snippet 17: Validate OB based on Volume
                            ob_vol_norm = int(ob_candle.get('vol_norm_int', 0)) # Ensure int
                            vol_threshold = self.strategy_params.volume_threshold
                            # Use a lower threshold for OB validation than entry signal? e.g., 50%
                            min_ob_vol = vol_threshold * 0.5
                            if self.strategy_params.validate_ob_volume and (ob_vol_norm is None or ob_vol_norm < min_ob_vol):
                                self.logger.debug(f"Skipping Bear OB at {pivot_candle_idx_dt}: Low volume ({ob_vol_norm} < {min_ob_vol:.0f})")
                                continue

                            # Determine OB boundaries (High/Low or Open/Close of the pivot candle)
                            box_top = safe_decimal(ob_candle['high'])
                            box_bottom = safe_decimal(ob_candle['low'])
                            if self.strategy_params.ob_source == "Bodys":
                                 box_top = safe_decimal(max(ob_candle['open'], ob_candle['close']))
                                 box_bottom = safe_decimal(min(ob_candle['open'], ob_candle['close']))

                            if box_top is None or box_bottom is None or box_top <= box_bottom:
                                 self.logger.warning(f"Invalid candle data for Bear OB at {pivot_candle_idx_dt}: Top={box_top}, Bottom={box_bottom}")
                                 continue

                            # Snippet 18: Adaptive OB Sizing
                            if self.strategy_params.adaptive_ob_sizing and current_atr > 0:
                                atr_adjust = current_atr * Decimal('0.1') # Example: adjust by 10% of ATR
                                box_top += atr_adjust
                                box_bottom -= atr_adjust # Widen the box slightly

                            new_box = {
                                'id': f"bear_{pivot_candle_idx_dt.strftime('%Y%m%d%H%M')}",
                                'type': 'bear',
                                'left_idx': pivot_candle_idx_dt, # Timestamp of the pivot candle
                                'right_idx': df.index[-1],     # Initial right index is current candle
                                'top': box_top,
                                'bottom': box_bottom,
                                'active': True,
                                'violated': False,
                                'expired': False, # Add expired flag
                                'volume': safe_decimal(ob_candle.get('volume')),
                                'vol_norm': ob_vol_norm
                            }
                            self.bear_boxes.append(new_box)
                            new_bear_indices.add(pivot_candle_idx_dt)
                            self.logger.debug(f"New Bear OB created: ID={new_box['id']}, Range=[{new_box['bottom']:.4f}, {new_box['top']:.4f}]")
                except (IndexError, KeyError) as e:
                     self.logger.error(f"Error processing potential Bear OB signal at {signal_idx_dt}: {e}")

            # --- Bullish OB (from Pivot Low) ---
            if is_pl_signal:
                try:
                    signal_idx_loc = df.index.get_loc(signal_idx_dt)
                    pivot_candle_loc = signal_idx_loc - self.strategy_params.pl_right
                    if pivot_candle_loc >= 0:
                        pivot_candle_idx_dt = df.index[pivot_candle_loc]
                        if pivot_candle_idx_dt not in new_bull_indices and not any(b['left_idx'] == pivot_candle_idx_dt for b in self.bull_boxes):
                            ob_candle = df.loc[pivot_candle_idx_dt]

                            # Snippet 17: Validate OB based on Volume
                            ob_vol_norm = int(ob_candle.get('vol_norm_int', 0)) # Ensure int
                            vol_threshold = self.strategy_params.volume_threshold
                            min_ob_vol = vol_threshold * 0.5
                            if self.strategy_params.validate_ob_volume and (ob_vol_norm is None or ob_vol_norm < min_ob_vol):
                                self.logger.debug(f"Skipping Bull OB at {pivot_candle_idx_dt}: Low volume ({ob_vol_norm} < {min_ob_vol:.0f})")
                                continue

                            # Determine OB boundaries
                            box_top = safe_decimal(ob_candle['high'])
                            box_bottom = safe_decimal(ob_candle['low'])
                            if self.strategy_params.ob_source == "Bodys":
                                box_top = safe_decimal(max(ob_candle['open'], ob_candle['close']))
                                box_bottom = safe_decimal(min(ob_candle['open'], ob_candle['close']))

                            if box_top is None or box_bottom is None or box_top <= box_bottom:
                                 self.logger.warning(f"Invalid candle data for Bull OB at {pivot_candle_idx_dt}: Top={box_top}, Bottom={box_bottom}")
                                 continue

                            # Snippet 18: Adaptive OB Sizing
                            if self.strategy_params.adaptive_ob_sizing and current_atr > 0:
                                atr_adjust = current_atr * Decimal('0.1')
                                box_top += atr_adjust
                                box_bottom -= atr_adjust

                            new_box = {
                                'id': f"bull_{pivot_candle_idx_dt.strftime('%Y%m%d%H%M')}",
                                'type': 'bull',
                                'left_idx': pivot_candle_idx_dt,
                                'right_idx': df.index[-1],
                                'top': box_top,
                                'bottom': box_bottom,
                                'active': True,
                                'violated': False,
                                'expired': False,
                                'volume': safe_decimal(ob_candle.get('volume')),
                                'vol_norm': ob_vol_norm
                            }
                            self.bull_boxes.append(new_box)
                            new_bull_indices.add(pivot_candle_idx_dt)
                            self.logger.debug(f"New Bull OB created: ID={new_box['id']}, Range=[{new_box['bottom']:.4f}, {new_box['top']:.4f}]")
                except (IndexError, KeyError) as e:
                     self.logger.error(f"Error processing potential Bull OB signal at {signal_idx_dt}: {e}")

        # --- Manage Existing Boxes (Violation, Extension, Expiry) ---
        last_close = safe_decimal(df['close'].iloc[-1])
        last_high = safe_decimal(df['high'].iloc[-1])
        last_low = safe_decimal(df['low'].iloc[-1])
        last_bar_idx = df.index[-1]

        if last_close is None or last_high is None or last_low is None:
             self.logger.warning("Last candle data missing, cannot update OB states.")
             return

        # Snippet 25: ATR-Based OB Expiry Setup
        expiry_atr_periods = self.strategy_params.ob_expiry_atr_periods
        mean_atr = Decimal('0')
        interval_minutes = 0
        can_calculate_expiry = False
        if expiry_atr_periods is not None and expiry_atr_periods > 0:
             # Calculate mean ATR over a longer period for stability
             long_lookback = max(200, len(df) // 2) # e.g., 200 periods or half the data
             atr_series = df['atr'].dropna()
             if 'atr' in df.columns and len(atr_series) > long_lookback // 2:
                  mean_atr = safe_decimal(atr_series.iloc[-long_lookback:].mean(), default=Decimal('0'))
             else:
                  mean_atr = Decimal('0')

             # Get interval in minutes (handle potential errors)
             try:
                 interval_str = self.config.get("interval", DEFAULT_INTERVAL)
                 if interval_str.isdigit():
                     interval_minutes = int(interval_str)
                 elif interval_str == 'D': interval_minutes = 1440
                 elif interval_str == 'W': interval_minutes = 10080
                 elif interval_str == 'M': interval_minutes = 43200 # Approximation
                 else: interval_minutes = 0 # Handle unexpected interval strings
             except (ValueError, TypeError): interval_minutes = 0

             if interval_minutes <= 0:
                  self.logger.warning("Could not determine interval in minutes for OB expiry calculation. Disabling expiry check.")
             elif mean_atr <= 0:
                  self.logger.warning("Could not calculate mean ATR for OB expiry. Disabling expiry check.")
             elif current_atr is None or current_atr <= 0: # Check current ATR too
                  self.logger.warning("Current ATR is zero or invalid. Disabling expiry check.")
             else:
                 can_calculate_expiry = True # All components needed are valid


        # Update Bullish Boxes
        for box in self.bull_boxes:
            if box['active']:
                # Violation Check: Close below the bottom
                if last_close < box['bottom']:
                    box.update({'active': False, 'violated': True, 'right_idx': last_bar_idx})
                    self.logger.debug(f"Bull OB {box['id']} violated at {last_close:.4f} (Bottom: {box['bottom']:.4f})")
                    continue # Stop processing this box if violated

                # Extension
                if self.strategy_params.ob_extend:
                    box['right_idx'] = last_bar_idx

                # Snippet 25: ATR-Based OB Expiry Check
                if can_calculate_expiry:
                    try:
                        # Ensure both indices are timezone-aware or naive consistently
                        if last_bar_idx.tzinfo is None and box['left_idx'].tzinfo is not None:
                             left_idx_naive = box['left_idx'].replace(tzinfo=None)
                             age_timedelta = last_bar_idx - left_idx_naive
                        elif last_bar_idx.tzinfo is not None and box['left_idx'].tzinfo is None:
                             left_idx_aware = box['left_idx'].replace(tzinfo=last_bar_idx.tzinfo)
                             age_timedelta = last_bar_idx - left_idx_aware
                        else: # Both same (aware or naive)
                             age_timedelta = last_bar_idx - box['left_idx']

                        age_bars = age_timedelta.total_seconds() / (interval_minutes * 60)

                        # Adjust expiry threshold based on current ATR relative to mean ATR
                        # Higher current vol -> shorter expiry, Lower current vol -> longer expiry
                        vol_ratio = mean_atr / current_atr # If current < mean, ratio > 1 (longer expiry)
                        # Clamp ratio to avoid extreme expiry times (e.g., 0.5x to 2x)
                        vol_ratio_clamped = max(Decimal('0.5'), min(vol_ratio, Decimal('2.0')))
                        expiry_threshold_bars = Decimal(str(expiry_atr_periods)) * vol_ratio_clamped
                        # Ensure a minimum reasonable expiry time in bars
                        min_expiry_bars = 10
                        expiry_threshold_bars = max(Decimal(str(min_expiry_bars)), expiry_threshold_bars)

                        if Decimal(str(age_bars)) > expiry_threshold_bars:
                            box.update({'active': False, 'expired': True, 'right_idx': last_bar_idx})
                            self.logger.debug(f"Bull OB {box['id']} expired after {age_bars:.1f} bars "
                                              f"(Threshold: {expiry_threshold_bars:.1f} ATR-adj bars, VolRatio: {vol_ratio_clamped:.2f})")
                    except Exception as e:
                         self.logger.warning(f"Error calculating expiry for Bull OB {box['id']}: {e}")


        # Update Bearish Boxes
        for box in self.bear_boxes:
            if box['active']:
                # Violation Check: Close above the top
                if last_close > box['top']:
                    box.update({'active': False, 'violated': True, 'right_idx': last_bar_idx})
                    self.logger.debug(f"Bear OB {box['id']} violated at {last_close:.4f} (Top: {box['top']:.4f})")
                    continue # Stop processing this box if violated

                # Extension
                if self.strategy_params.ob_extend:
                    box['right_idx'] = last_bar_idx

                # Snippet 25: ATR-Based OB Expiry Check
                if can_calculate_expiry:
                     try:
                        if last_bar_idx.tzinfo is None and box['left_idx'].tzinfo is not None:
                             left_idx_naive = box['left_idx'].replace(tzinfo=None)
                             age_timedelta = last_bar_idx - left_idx_naive
                        elif last_bar_idx.tzinfo is not None and box['left_idx'].tzinfo is None:
                             left_idx_aware = box['left_idx'].replace(tzinfo=last_bar_idx.tzinfo)
                             age_timedelta = last_bar_idx - left_idx_aware
                        else: # Both same (aware or naive)
                             age_timedelta = last_bar_idx - box['left_idx']

                        age_bars = age_timedelta.total_seconds() / (interval_minutes * 60)
                        vol_ratio = mean_atr / current_atr
                        vol_ratio_clamped = max(Decimal('0.5'), min(vol_ratio, Decimal('2.0')))
                        expiry_threshold_bars = Decimal(str(expiry_atr_periods)) * vol_ratio_clamped
                        min_expiry_bars = 10
                        expiry_threshold_bars = max(Decimal(str(min_expiry_bars)), expiry_threshold_bars)

                        if Decimal(str(age_bars)) > expiry_threshold_bars:
                            box.update({'active': False, 'expired': True, 'right_idx': last_bar_idx})
                            self.logger.debug(f"Bear OB {box['id']} expired after {age_bars:.1f} bars "
                                              f"(Threshold: {expiry_threshold_bars:.1f} ATR-adj bars, VolRatio: {vol_ratio_clamped:.2f})")
                     except Exception as e:
                          self.logger.warning(f"Error calculating expiry for Bear OB {box['id']}: {e}")


        # --- Prune OB Lists ---
        # Keep a limited number of active and inactive (violated/expired) boxes
        # Sort criteria: Active first, Non-expired first, then Newest (by left_idx)
        default_time = datetime.min.replace(tzinfo=timezone.utc) # Ensure timezone consistent default
        sort_key = lambda b: (b.get('active', False), not b.get('expired', False), b.get('left_idx', default_time))

        self.bull_boxes.sort(key=sort_key, reverse=True)
        self.bear_boxes.sort(key=sort_key, reverse=True)

        max_active = self.strategy_params.ob_max_boxes
        max_inactive = max_active // 2 # Keep fewer inactive boxes

        # Separate active and inactive after sorting
        active_bull = [b for b in self.bull_boxes if b.get('active')]
        inactive_bull = [b for b in self.bull_boxes if not b.get('active')]
        active_bear = [b for b in self.bear_boxes if b.get('active')]
        inactive_bear = [b for b in self.bear_boxes if not b.get('active')]

        # Recombine keeping limits
        self.bull_boxes = active_bull[:max_active] + inactive_bull[:max_inactive]
        self.bear_boxes = active_bear[:max_active] + inactive_bear[:max_inactive]

        # Use len(active_bull/bear) for logging active counts before pruning
        self.logger.info(f"OB Update: Active Bull={len(active_bull)}, Active Bear={len(active_bear)} "
                         f"(Total Kept: Bull={len(self.bull_boxes)}, Bear={len(self.bear_boxes)})")


    def _get_higher_tf_analysis(self) -> Optional[HigherTimeframeAnalysis]:
        """Performs basic trend analysis on a higher timeframe (Snippet 19).

        Returns:
            HigherTimeframeAnalysis dictionary or None if disabled, error, or insufficient data.
        """
        if not self.strategy_params.multi_tf_confirm or self.exchange is None:
            return None

        higher_tf_interval_str = self.strategy_params.higher_tf
        if higher_tf_interval_str not in CCXT_INTERVAL_MAP:
            self.logger.warning(f"Invalid higher timeframe '{higher_tf_interval_str}' specified in config. Skipping HTF analysis.")
            return None

        # Avoid infinite recursion if HTF is same as base interval
        base_interval_str = self.config.get("interval", DEFAULT_INTERVAL)
        if CCXT_INTERVAL_MAP.get(higher_tf_interval_str) == CCXT_INTERVAL_MAP.get(base_interval_str):
            self.logger.warning(f"Higher timeframe ({higher_tf_interval_str}) is the same as base interval ({base_interval_str}). Skipping HTF analysis.")
            return None


        ccxt_higher_tf = CCXT_INTERVAL_MAP[higher_tf_interval_str]
        self.logger.debug(f"Performing higher timeframe analysis ({ccxt_higher_tf})...")
        try:
            # Fetch limited data for HTF trend check (e.g., enough for EMA)
            htf_fetch_limit = self.strategy_params.vt_length + 50 # Need enough for EMA calculation + buffer
            higher_df = fetch_klines_ccxt(self.exchange, self.symbol, ccxt_higher_tf, limit=htf_fetch_limit, logger=self.logger)

            if higher_df.empty or len(higher_df) < self.strategy_params.vt_length:
                self.logger.warning(f"Insufficient kline data ({len(higher_df)}) for higher timeframe {ccxt_higher_tf} trend analysis.")
                return None

            # Perform basic trend analysis (e.g., price vs EMA)
            htf_ema = ta.ema(higher_df['close'], length=self.strategy_params.vt_length, fillna=False)
            if htf_ema is None or htf_ema.empty or htf_ema.isna().all():
                 self.logger.warning(f"Could not calculate EMA on higher timeframe {ccxt_higher_tf} data.")
                 return None

            # Get last valid values safely
            last_htf_close = safe_decimal(higher_df['close'].iloc[-1])
            # Find last non-NaN EMA value
            last_valid_htf_ema = htf_ema.dropna().iloc[-1] if not htf_ema.dropna().empty else None
            last_htf_ema = safe_decimal(last_valid_htf_ema)


            if last_htf_close is None or last_htf_ema is None:
                 self.logger.warning(f"Missing close or EMA value on last HTF ({ccxt_higher_tf}) candle.")
                 return None

            htf_trend_up = last_htf_close >= last_htf_ema # Current close vs EMA
            trend_str = 'UP' if htf_trend_up else 'DOWN'
            self.logger.info(f"Higher timeframe ({ccxt_higher_tf}) trend: {trend_str} (Close={last_htf_close:.4f}, EMA={last_htf_ema:.4f})")

            return HigherTimeframeAnalysis(current_trend_up=htf_trend_up)

        except Exception as e:
            self.logger.error(f"Error during higher timeframe ({ccxt_higher_tf}) analysis for {self.symbol}: {e}", exc_info=True)
            return None


    def update(self, df: pd.DataFrame) -> Optional[StrategyAnalysisResults]:
        """Updates the strategy state with new data and returns analysis results.

        Args:
            df: The latest DataFrame with OHLCV data.

        Returns:
            A StrategyAnalysisResults dictionary containing the results,
            or None if the update fails critically.
        """
        start_time = time.monotonic()
        self.logger.debug(f"Starting strategy update for {self.symbol} with {len(df)} candles. Last candle: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else 'N/A'}")

        if df.empty or len(df) < self.min_data_len:
            self.logger.warning(f"Insufficient data for strategy update: {len(df)} rows, require {self.min_data_len}")
            return None # Indicate failure to update

        # 1. Calculate Indicators
        df_with_indicators = self._calculate_indicators(df)
        if df_with_indicators.empty:
            self.logger.error("DataFrame is empty after indicator calculation. Cannot proceed.")
            return None

        # 2. Identify and Manage Order Blocks
        # This modifies self.bull_boxes and self.bear_boxes directly
        self._identify_order_blocks(df_with_indicators)

        # 3. Perform Higher Timeframe Analysis (Optional)
        higher_tf_results = self._get_higher_tf_analysis()

        # 4. Extract Last Row Data and Prepare Results
        try:
            if df_with_indicators.empty:
                 self.logger.error("DataFrame became empty during processing. Cannot extract last row.")
                 return None

            last_row = df_with_indicators.iloc[-1]

            # Safely extract values using .get() and safe_decimal
            last_close = safe_decimal(last_row.get('close'))
            last_high = safe_decimal(last_row.get('high'))
            last_low = safe_decimal(last_row.get('low'))
            current_trend_up = last_row.get('trend_up') # Already boolean or NaN
            if pd.isna(current_trend_up): current_trend_up = None # Convert NaN to None
            trend_just_changed = bool(last_row.get('trend_changed', False))
            vol_norm_val = last_row.get('vol_norm_int')
            vol_norm_int = int(vol_norm_val) if pd.notna(vol_norm_val) else None
            atr = safe_decimal(last_row.get('atr'))
            upper_band = safe_decimal(last_row.get('upper_band'))
            lower_band = safe_decimal(last_row.get('lower_band'))
            adx = safe_decimal(last_row.get('adx'))
            rsi = safe_decimal(last_row.get('rsi'))
            vwap = safe_decimal(last_row.get('vwap'))
            roc = safe_decimal(last_row.get('roc'))

            if last_close is None:
                self.logger.error("Last close price is missing after indicator calculation.")
                return None # Critical data missing

            # Filter OBs to return only active ones in the result
            active_bull_obs = [b for b in self.bull_boxes if b.get('active')]
            active_bear_obs = [b for b in self.bear_boxes if b.get('active')]


            results = StrategyAnalysisResults(
                dataframe=df_with_indicators, # Return the processed dataframe
                last_close=last_close,
                last_high=last_high,
                last_low=last_low,
                current_trend_up=current_trend_up,
                trend_just_changed=trend_just_changed,
                active_bull_boxes=active_bull_obs, # Return only active boxes
                active_bear_boxes=active_bear_obs,
                vol_norm_int=vol_norm_int,
                atr=atr,
                upper_band=upper_band,
                lower_band=lower_band,
                adx=adx,
                rsi=rsi,
                vwap=vwap,
                roc=roc,
                higher_tf_analysis=higher_tf_results
            )

            end_time = time.monotonic()
            self.logger.debug(f"Strategy update finished in {end_time - start_time:.3f} seconds.")
            return results

        except IndexError:
             self.logger.error("Could not access last row of DataFrame after indicator calculation (DataFrame might be empty).")
             return None
        except Exception as e:
             self.logger.error(f"Unexpected error preparing strategy results: {e}", exc_info=True)
             return None


# --- Signal Generation ---

class SignalGenerator:
    """Generates trading signals based on strategy analysis results and configuration."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        try:
            self.strategy_params = StrategyParams(**config.get("strategy_params", {}))
            self.protection_config = ProtectionConfig(**config.get("protection", {}))
        except ValidationError as e:
            self.logger.error(f"SignalGenerator Pydantic validation failed: {e}. Using defaults.")
            self.strategy_params = StrategyParams()
            self.protection_config = ProtectionConfig()

        # Cache factors for quick access using safe_decimal
        self.base_ob_entry_proximity_factor = safe_decimal(self.strategy_params.ob_entry_proximity_factor, default=Decimal('1.005'))
        self.base_ob_exit_proximity_factor = safe_decimal(self.strategy_params.ob_exit_proximity_factor, default=Decimal('1.001'))


    def _check_entry_confluence(self, analysis_results: StrategyAnalysisResults) -> bool:
        """Checks all configured confluence filters for a potential entry.

        Args:
            analysis_results: The results from the strategy update.

        Returns:
            True if all active confluence checks pass, False otherwise.
        """
        # Get data safely
        latest_close = analysis_results.get('last_close')
        is_trend_up = analysis_results.get('current_trend_up') # Main timeframe trend

        if latest_close is None or latest_close <= 0:
            self.logger.debug("Entry Filtered: Invalid latest close price.")
            return False
        if is_trend_up is None:
            self.logger.debug("Entry Filtered: Trend is unknown.")
            return False

        # --- Apply Filters Sequentially ---
        # Snippet 1: Volume Confirmation
        if self.strategy_params.volume_threshold > 0:
            vol_norm = analysis_results.get('vol_norm_int')
            if vol_norm is None or vol_norm < self.strategy_params.volume_threshold:
                self.logger.debug(f"Entry Filtered: Low Volume ({vol_norm} < {self.strategy_params.volume_threshold})")
                return False

        # Snippet 3: Trend Strength Filter (ADX)
        if self.strategy_params.adx_threshold > 0:
            adx = analysis_results.get('adx')
            if adx is None or adx < self.strategy_params.adx_threshold:
                self.logger.debug(f"Entry Filtered: Weak Trend (ADX {adx:.1f} < {self.strategy_params.adx_threshold})")
                return False

        # Snippet 7: RSI Confirmation
        if self.strategy_params.rsi_confirm:
            rsi = analysis_results.get('rsi')
            if rsi is None:
                 self.logger.debug("Entry Filtered: RSI unavailable")
                 return False
            # Filter if RSI is in extreme zone *against* the trend
            if is_trend_up and rsi >= self.strategy_params.rsi_overbought:
                 self.logger.debug(f"Entry Filtered: RSI Overbought ({rsi:.1f} >= {self.strategy_params.rsi_overbought}) for Long")
                 return False
            if not is_trend_up and rsi <= self.strategy_params.rsi_oversold:
                 self.logger.debug(f"Entry Filtered: RSI Oversold ({rsi:.1f} <= {self.strategy_params.rsi_oversold}) for Short")
                 return False

        # Snippet 11: Candlestick Confirmation
        if self.strategy_params.candlestick_confirm:
            df = analysis_results.get('dataframe')
            if df is None or df.empty or len(df) < 1: # Need at least one row
                 self.logger.debug("Entry Filtered: DataFrame unavailable for candlestick check.")
                 return False # Cannot confirm
            # Check if columns exist (might not if calculation failed)
            bull_col_exists = 'bull_engulfing' in df.columns
            bear_col_exists = 'bear_engulfing' in df.columns
            # Use simple engulfing check for now, check last confirmed candle
            last_candle_index = -1 # Check the most recent candle
            if is_trend_up:
                if not bull_col_exists or not df['bull_engulfing'].iloc[last_candle_index]:
                    self.logger.debug("Entry Filtered: No confirming Bullish Candlestick Pattern (e.g., Engulfing) on last candle.")
                    return False # Hard filter
            elif not is_trend_up:
                 if not bear_col_exists or not df['bear_engulfing'].iloc[last_candle_index]:
                    self.logger.debug("Entry Filtered: No confirming Bearish Candlestick Pattern (e.g., Engulfing) on last candle.")
                    return False # Hard filter

        # Snippet 15: VWAP Confirmation
        if self.strategy_params.vwap_confirm:
            vwap = analysis_results.get('vwap')
            if vwap is None:
                 self.logger.debug("Entry Filtered: VWAP unavailable")
                 return False
            if is_trend_up and latest_close < vwap:
                self.logger.debug(f"Entry Filtered: Price ({latest_close:.4f}) below VWAP ({vwap:.4f}) for Long")
                return False
            if not is_trend_up and latest_close > vwap:
                self.logger.debug(f"Entry Filtered: Price ({latest_close:.4f}) above VWAP ({vwap:.4f}) for Short")
                return False

        # Snippet 19: Multi-Timeframe Confirmation
        if self.strategy_params.multi_tf_confirm:
            htf_analysis = analysis_results.get('higher_tf_analysis')
            if htf_analysis is None or htf_analysis.get('current_trend_up') is None:
                self.logger.debug("Entry Filtered: Higher TF trend unknown or analysis failed")
                return False
            htf_trend_up = htf_analysis['current_trend_up']
            if is_trend_up and not htf_trend_up:
                self.logger.debug(f"Entry Filtered: Base trend UP, but Higher TF trend DOWN")
                return False
            if not is_trend_up and htf_trend_up:
                self.logger.debug(f"Entry Filtered: Base trend DOWN, but Higher TF trend UP")
                return False

        # Snippet 23: Momentum Confirmation (ROC)
        if self.strategy_params.momentum_confirm:
             roc = analysis_results.get('roc')
             if roc is None:
                  self.logger.debug("Entry Filtered: ROC unavailable")
                  return False
             # Require positive ROC for long, negative for short
             if is_trend_up and roc <= 0:
                  self.logger.debug(f"Entry Filtered: Negative/Zero Momentum (ROC {roc:.2f}) for Long")
                  return False
             if not is_trend_up and roc >= 0:
                  self.logger.debug(f"Entry Filtered: Positive/Zero Momentum (ROC {roc:.2f}) for Short")
                  return False

        # If all active filters passed
        self.logger.debug("All active entry confluence filters passed.")
        return True


    def _check_ob_entry(self, analysis_results: StrategyAnalysisResults) -> str:
        """Checks for Order Block based entry signals (Zone Entry or Breakout)."""
        signal = "HOLD"
        # Safe data extraction
        latest_close = analysis_results.get('last_close')
        is_trend_up = analysis_results.get('current_trend_up')
        active_bull_obs = analysis_results.get('active_bull_boxes', [])
        active_bear_obs = analysis_results.get('active_bear_boxes', [])
        current_atr = analysis_results.get('atr')
        df = analysis_results.get('dataframe')

        if latest_close is None or latest_close <= 0: return signal
        if is_trend_up is None: return signal # Trend unknown

        # --- Calculate Dynamic Proximity Factor (Snippet 8) ---
        ob_entry_proximity = self.base_ob_entry_proximity_factor
        if self.strategy_params.dynamic_ob_proximity and current_atr is not None and current_atr > 0:
            relative_atr = current_atr / latest_close if latest_close > 0 else Decimal('0')
            # Increase proximity range slightly in higher vol (more room for entry)
            # Scale factor: 1 + (relative_atr * scaling_intensity)
            # Example: Intensity = 2 -> RelATR=0.01 gives 1.02 factor, RelATR=0.03 gives 1.06 factor
            dynamic_factor = Decimal('1') + (relative_atr * Decimal('2'))
            # Clamp factor (e.g., 1.0x to 1.5x of base factor)
            clamped_factor = max(Decimal('1.0'), min(dynamic_factor, Decimal('1.5')))
            ob_entry_proximity = self.base_ob_entry_proximity_factor * clamped_factor
            # self.logger.debug(f"Using dynamic OB entry proximity factor: {ob_entry_proximity:.5f} (Base: {self.base_ob_entry_proximity_factor}, ATR-Adj: {clamped_factor:.2f})")


        # --- Entry Logic: Breakout or Zone ---
        # Snippet 13: Breakout Confirmation Logic
        if self.strategy_params.breakout_confirm:
            if df is None or len(df) < 2: return "HOLD" # Need previous bar
            prev_close = safe_decimal(df['close'].iloc[-2])
            if prev_close is None: return "HOLD"

            if is_trend_up:
                # Look for breakout *above* the nearest *active* Bear OB (break of resistance)
                # Filter relevant OBs first
                relevant_bear_obs = [ob for ob in active_bear_obs if ob.get('top') is not None and ob['top'] > latest_close]
                if relevant_bear_obs:
                    # Sort by OB top price (ascending) to find the nearest one above
                    nearest_bear_ob = min(relevant_bear_obs, key=lambda x: x['top'])
                    ob_top_price = nearest_bear_ob['top']
                    if latest_close > ob_top_price and prev_close <= ob_top_price:
                         self.logger.info(f"BUY signal (Breakout): Price broke above Bear OB {nearest_bear_ob.get('id','N/A')} Top ({ob_top_price:.4f})")
                         signal = "BUY"

            else: # Bearish Trend
                # Look for breakout *below* the nearest *active* Bull OB (break of support)
                relevant_bull_obs = [ob for ob in active_bull_obs if ob.get('bottom') is not None and ob['bottom'] < latest_close]
                if relevant_bull_obs:
                     # Sort by OB bottom price (descending) to find the nearest one below
                     nearest_bull_ob = max(relevant_bull_obs, key=lambda x: x['bottom'])
                     ob_bottom_price = nearest_bull_ob['bottom']
                     if latest_close < ob_bottom_price and prev_close >= ob_bottom_price:
                          self.logger.info(f"SELL signal (Breakout): Price broke below Bull OB {nearest_bull_ob.get('id','N/A')} Bottom ({ob_bottom_price:.4f})")
                          signal = "SELL"

        # Default Entry Logic: Price enters relevant OB zone
        else:
            ob_hit_count = 0
            if is_trend_up:
                for ob in active_bull_obs:
                    # Ensure OB has valid boundaries
                    ob_bottom = ob.get('bottom')
                    ob_top = ob.get('top')
                    if ob_bottom is None or ob_top is None: continue

                    # Allow entry slightly above the top edge based on proximity factor
                    entry_top_boundary = ob_top * ob_entry_proximity
                    if ob_bottom <= latest_close <= entry_top_boundary:
                        self.logger.debug(f"Price ({latest_close:.4f}) entered Bull OB {ob.get('id','N/A')} "
                                          f"Range=[{ob_bottom:.4f}, {ob_top:.4f}] "
                                          f"(Proximity Top: {entry_top_boundary:.4f})")
                        ob_hit_count += 1
                        if signal == "HOLD": # Take first valid OB entry
                            signal = "BUY"
            else: # Bearish Trend
                for ob in active_bear_obs:
                    ob_bottom = ob.get('bottom')
                    ob_top = ob.get('top')
                    if ob_bottom is None or ob_top is None: continue

                    # Allow entry slightly below the bottom edge
                    # Proximity factor > 1, so (2 - factor) gives value < 1
                    entry_bottom_boundary = ob_bottom * (Decimal("2") - ob_entry_proximity) # e.g., if factor=1.005, multiplier is 0.995
                    if entry_bottom_boundary <= latest_close <= ob_top:
                        self.logger.debug(f"Price ({latest_close:.4f}) entered Bear OB {ob.get('id','N/A')} "
                                          f"Range=[{ob_bottom:.4f}, {ob_top:.4f}] "
                                          f"(Proximity Bottom: {entry_bottom_boundary:.4f})")
                        ob_hit_count += 1
                        if signal == "HOLD":
                            signal = "SELL"

            # Snippet 21: OB Confluence Check
            if signal != "HOLD" and ob_hit_count < self.strategy_params.ob_confluence:
                 self.logger.debug(f"Signal Filtered: OB Confluence not met ({ob_hit_count} hits < {self.strategy_params.ob_confluence} required)")
                 signal = "HOLD"

        return signal

    def _check_exit_conditions(self, analysis_results: StrategyAnalysisResults, open_position: Dict) -> str:
        """Checks for exit conditions based on trend reversal or OB proximity."""
        signal = "HOLD"
        # Safe data extraction
        latest_close = analysis_results.get('last_close')
        is_trend_up = analysis_results.get('current_trend_up')
        trend_just_changed = analysis_results.get('trend_just_changed', False)
        active_bull_obs = analysis_results.get('active_bull_boxes', [])
        active_bear_obs = analysis_results.get('active_bear_boxes', [])
        current_pos_side = open_position.get('side')

        if latest_close is None or latest_close <= 0: return signal

        # --- Exit Condition 1: Trend Reversal ---
        # Exit if trend *confirmed* against position (i.e., is_trend_up has flipped)
        if current_pos_side == 'long' and is_trend_up is False: # Trend is now DOWN
            self.logger.info(f"{NEON_YELLOW}Exit Signal: Trend changed to DOWN for open LONG position.{RESET}")
            return "EXIT_LONG"
        if current_pos_side == 'short' and is_trend_up is True: # Trend is now UP
            self.logger.info(f"{NEON_YELLOW}Exit Signal: Trend changed to UP for open SHORT position.{RESET}")
            return "EXIT_SHORT"

        # --- Exit Condition 2: Price Approaching Opposing OB ---
        # Calculate dynamic exit proximity (could be different from entry)
        ob_exit_proximity = self.base_ob_exit_proximity_factor
        # Potentially add dynamic adjustment for exit proximity too (e.g., based on ATR or PnL)

        if current_pos_side == 'long':
            # Find nearest *active* Bear OB above current price
            relevant_bear_obs = [b for b in active_bear_obs if b.get('bottom') is not None and b['bottom'] > latest_close]
            if relevant_bear_obs:
                nearest_bear_ob = min(relevant_bear_obs, key=lambda x: x['bottom']) # Nearest bottom above current price
                ob_bottom = nearest_bear_ob['bottom']
                # Exit if price gets within proximity factor of the bottom edge
                exit_boundary = ob_bottom * (Decimal("2") - ob_exit_proximity) # e.g., factor 1.001 -> multiplier 0.999
                if latest_close >= exit_boundary:
                    self.logger.info(f"{NEON_YELLOW}Exit Signal (Long): Price ({latest_close:.4f}) approaching Bear OB {nearest_bear_ob.get('id','N/A')} "
                                     f"Bottom ({ob_bottom:.4f}) with proximity ({exit_boundary:.4f}).{RESET}")
                    return "EXIT_LONG"

        elif current_pos_side == 'short':
            # Find nearest *active* Bull OB below current price
            relevant_bull_obs = [b for b in active_bull_obs if b.get('top') is not None and b['top'] < latest_close]
            if relevant_bull_obs:
                nearest_bull_ob = max(relevant_bull_obs, key=lambda x: x['top']) # Nearest top below current price
                ob_top = nearest_bull_ob['top']
                # Exit if price gets within proximity factor of the top edge
                exit_boundary = ob_top * ob_exit_proximity # e.g., factor 1.001 -> multiplier 1.001
                if latest_close <= exit_boundary:
                    self.logger.info(f"{NEON_YELLOW}Exit Signal (Short): Price ({latest_close:.4f}) approaching Bull OB {nearest_bull_ob.get('id','N/A')} "
                                     f"Top ({ob_top:.4f}) with proximity ({exit_boundary:.4f}).{RESET}")
                    return "EXIT_SHORT"

        return signal # No exit condition met

    def generate_signal(self, analysis_results: Optional[StrategyAnalysisResults], open_position: Optional[Dict]) -> str:
        """Generates the final BUY/SELL/HOLD/EXIT signal.

        Args:
            analysis_results: The results from the strategy update, or None if update failed.
            open_position: Dictionary with details of the current open position, or None.

        Returns:
            The generated signal string: "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT".
        """
        if analysis_results is None:
            self.logger.warning("Cannot generate signal: Strategy analysis results are missing.")
            return "HOLD"
        if analysis_results.get('current_trend_up') is None:
             self.logger.warning("Cannot generate signal: Trend is unknown.")
             return "HOLD"

        # --- Check Exit Conditions First if Position is Open ---
        if open_position:
            exit_signal = self._check_exit_conditions(analysis_results, open_position)
            if exit_signal != "HOLD":
                return exit_signal
            else:
                # No exit signal, just hold the current position
                self.logger.debug("Holding existing position based on exit condition checks.")
                return "HOLD"

        # --- No Open Position: Check Entry Conditions ---
        else:
            # Check confluence filters first
            passes_confluence = self._check_entry_confluence(analysis_results)
            if not passes_confluence:
                # self.logger.debug("Entry signal suppressed by confluence filters.") # Already logged in _check_entry_confluence
                return "HOLD"

            # Check OB entry conditions (Zone or Breakout)
            entry_signal = self._check_ob_entry(analysis_results)
            if entry_signal != "HOLD":
                self.logger.info(f"{BRIGHT}{NEON_GREEN if entry_signal == 'BUY' else NEON_RED}Entry Signal Generated: {entry_signal}{RESET}")
                return entry_signal

            # No entry signal triggered
            self.logger.debug("No entry signal generated (Trend/OB conditions not met).")
            return "HOLD"

    def calculate_initial_tp_sl(
        self,
        entry_price: Decimal,
        signal: str, # "BUY" or "SELL"
        analysis_results: StrategyAnalysisResults,
        market_info: Dict
    ) -> Tuple[Optional[List[Dict]], Optional[Decimal]]:
        """Calculates initial TP levels and SL price based on configuration and analysis.

        Args:
            entry_price: The price at which the position was (or will be) entered.
            signal: "BUY" or "SELL".
            analysis_results: Strategy analysis results containing ATR, DataFrame, etc.
            market_info: Market information dictionary.

        Returns:
            A tuple containing:
            - List of TP level dictionaries [{'price': Decimal, 'percentage': Decimal}, ...], or None if no valid TP.
            - The calculated SL price as a Decimal (unformatted), or None if calculation fails or RR filter fails.
        """
        self.logger.debug(f"Calculating initial TP/SL for {signal} entry at {entry_price:.4f}")
        atr = analysis_results.get('atr')
        df = analysis_results.get('dataframe') # Needed for dynamic adjustments

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            self.logger.error("Invalid signal provided for TP/SL calculation.")
            return None, None
        if atr is None or atr <= 0:
            self.logger.warning("ATR is missing or invalid. Cannot calculate ATR-based TP/SL.")
            return None, None
        if entry_price <= 0:
            self.logger.error("Invalid entry price provided for TP/SL calculation.")
            return None, None
        # Don't fail if df is None, just skip dynamic adjustments

        # --- Determine Base Multipliers ---
        base_tp_multiple = safe_decimal(self.protection_config.initial_take_profit_atr_multiple, default=Decimal('0.7'))
        base_sl_multiple = safe_decimal(self.protection_config.initial_stop_loss_atr_multiple, default=Decimal('1.8'))
        self.logger.debug(f"Base Multipliers: TP={base_tp_multiple}x ATR, SL={base_sl_multiple}x ATR")


        # --- Dynamic Adjustments to Multipliers ---
        tp_multiple = base_tp_multiple
        sl_multiple = base_sl_multiple

        # Snippet 2: Dynamic ATR Multiplier based on ATR percentile
        if self.protection_config.dynamic_atr_multiplier and df is not None and 'atr' in df.columns:
            try:
                atr_series = df['atr'].dropna()
                window = 100 # Lookback for percentile
                min_periods = window // 2
                if len(atr_series) >= min_periods:
                    # Calculate rolling quantiles safely
                    q25_series = atr_series.rolling(window=window, min_periods=min_periods).quantile(0.25)
                    q75_series = atr_series.rolling(window=window, min_periods=min_periods).quantile(0.75)

                    if not q25_series.empty and not q75_series.empty and not q25_series.isna().all() and not q75_series.isna().all():
                        atr_perc_25 = safe_decimal(q25_series.iloc[-1])
                        atr_perc_75 = safe_decimal(q75_series.iloc[-1])

                        if atr_perc_25 and atr_perc_75: # Ensure percentiles calculated
                            # Check ATR relative to percentiles
                            if atr > atr_perc_75 * Decimal('1.1'): # High volatility (ATR > ~75th percentile + buffer)
                                factor = Decimal('1.2') # Widen TP/SL range
                                tp_multiple *= factor
                                sl_multiple *= factor
                                self.logger.info(f"Dynamic TP/SL: High Volatility (ATR {atr:.4f} > 75th Perc {atr_perc_75:.4f}). Factor: {factor:.2f}")
                            elif atr < atr_perc_25 * Decimal('0.9'): # Low volatility (ATR < ~25th percentile - buffer)
                                factor = Decimal('0.8') # Tighten TP/SL range
                                tp_multiple *= factor
                                sl_multiple *= factor
                                self.logger.info(f"Dynamic TP/SL: Low Volatility (ATR {atr:.4f} < 25th Perc {atr_perc_25:.4f}). Factor: {factor:.2f}")
            except Exception as e:
                self.logger.warning(f"Could not calculate ATR percentile for dynamic TP/SL: {e}")

        # Snippet 5: Volatility-Based SL Adjustment (based on ATR std dev / Coeff of Variation)
        if self.protection_config.volatility_sl_adjust and df is not None and 'atr' in df.columns:
            try:
                window = 20 # Look at std dev over last 20 periods
                min_periods = window // 2
                recent_atr = df['atr'].iloc[-window:].dropna()
                if len(recent_atr) >= min_periods:
                    atr_std = safe_decimal(recent_atr.std())
                    atr_mean = safe_decimal(recent_atr.mean())
                    if atr_mean is not None and atr_mean > 0 and atr_std is not None:
                        coeff_of_var = atr_std / atr_mean # Coefficient of Variation
                        # If volatility of ATR is high, widen SL further
                        if coeff_of_var > Decimal('0.3'): # Example threshold: If std dev > 30% of mean
                            sl_adjust_factor = Decimal('1.2') # Widen SL by 20%
                            original_sl_multiple = sl_multiple
                            sl_multiple *= sl_adjust_factor
                            self.logger.info(f"Volatility SL Adjust: High ATR CoeffVar ({coeff_of_var:.2f} > 0.3). SL Multiple increased from {original_sl_multiple:.2f} to {sl_multiple:.2f}")
            except Exception as e:
                self.logger.warning(f"Could not calculate ATR std dev for SL adjustment: {e}")

        self.logger.debug(f"Adjusted Multipliers: TP={tp_multiple:.2f}x ATR, SL={sl_multiple:.2f}x ATR")

        # --- Calculate Stop Loss Price (Raw) ---
        sl_offset = atr * sl_multiple
        sl_price_raw = entry_price - sl_offset if signal == "BUY" else entry_price + sl_offset

        # Ensure SL is not zero or negative, and different from entry
        if sl_price_raw <= 0 or sl_price_raw == entry_price:
            self.logger.error(f"Calculated invalid raw SL price ({sl_price_raw}) <= 0 or equals entry price. Aborting calculation.")
            return None, None

        # --- Calculate Take Profit Levels (Raw) ---
        tp_levels_calculated: List[Dict] = []

        # Snippet 4: Partial Take-Profit Levels
        use_partial_tp = isinstance(self.protection_config.partial_tp_levels, list) and self.protection_config.partial_tp_levels
        use_fib_tp = self.strategy_params.fib_tp_levels

        if use_partial_tp and not use_fib_tp:
            self.logger.debug("Calculating Partial TP levels based on config.")
            remaining_percentage = Decimal('1.0')
            total_allocated = Decimal('0.0')
            for i, level_model in enumerate(self.protection_config.partial_tp_levels):
                try:
                    # Access validated data from the Pydantic model instance
                    level_multiple = safe_decimal(level_model.multiple, default=Decimal('1.0'))
                    level_percentage = safe_decimal(level_model.percentage, default=Decimal('0.0'))

                    # Use dynamically adjusted base TP multiple
                    effective_tp_multiple = tp_multiple * level_multiple

                    # Ensure percentage doesn't exceed remaining (handle potential overshoot)
                    actual_percentage = min(level_percentage, remaining_percentage)
                    if actual_percentage <= 0: continue # Skip if no remaining percentage or invalid level percentage

                    tp_offset = atr * effective_tp_multiple # Use effective multiple
                    tp_price_raw = entry_price + tp_offset if signal == "BUY" else entry_price - tp_offset

                    # Ensure TP is valid (better than entry)
                    if (signal == "BUY" and tp_price_raw > entry_price) or \
                       (signal == "SELL" and tp_price_raw < entry_price):
                        tp_levels_calculated.append({'price': tp_price_raw, 'percentage': actual_percentage})
                        remaining_percentage -= actual_percentage
                        total_allocated += actual_percentage
                        self.logger.debug(f"Calculated Raw Partial TP {i+1}: Price={tp_price_raw:.4f}, Size%={actual_percentage:.1%}")
                    else:
                        self.logger.warning(f"Skipping partial TP level {i+1}: Calculated raw price {tp_price_raw} not valid relative to entry {entry_price}.")

                    if remaining_percentage <= Decimal('0.001'): # Allow for small rounding errors
                         break # Stop if 100% allocated
                except Exception as e:
                    self.logger.error(f"Error processing partial TP level {i+1} ({level_model}): {e}")

            # If loop finishes but percentage remains, allocate rest to last level?
            if remaining_percentage > Decimal('0.01') and tp_levels_calculated:
                 self.logger.warning(f"Partial TP percentages sum to {total_allocated:.4f} (< 1.0). Allocating remaining {remaining_percentage:.2%} to last level.")
                 tp_levels_calculated[-1]['percentage'] += remaining_percentage


        # Snippet 10: Fibonacci-Based TP Levels (Alternative)
        elif use_fib_tp and df is not None:
            self.logger.debug("Calculating Fibonacci TP levels.")
            try:
                window = 30 # Increased window for potentially better swings
                min_periods = window // 2
                if len(df) < min_periods: raise ValueError(f"Not enough data ({len(df)}) for Fib swing (need {min_periods})")
                recent_data = df.iloc[-window:]

                swing_high = safe_decimal(recent_data['high'].max())
                swing_low = safe_decimal(recent_data['low'].min())

                if swing_high is None or swing_low is None or swing_high <= swing_low:
                     raise ValueError(f"Invalid swing points for Fib: High={swing_high}, Low={swing_low}")

                price_range = swing_high - swing_low
                if price_range <= 0:
                     raise ValueError(f"Invalid price range for Fib: {price_range}")

                # Common Fib extension levels for TP (relative to the range)
                fib_levels = [Decimal('0.618'), Decimal('1.0'), Decimal('1.618')] # Example levels
                if not fib_levels: raise ValueError("No Fib levels defined")
                percentage_per_level = Decimal('1.0') / Decimal(len(fib_levels)) # Equal percentage per level

                for level in fib_levels:
                    if signal == "BUY":
                        # Projecting range upwards from entry or swing high? Let's use entry.
                        tp_price_raw = entry_price + price_range * level
                        if tp_price_raw > entry_price:
                            tp_levels_calculated.append({'price': tp_price_raw, 'percentage': percentage_per_level})
                        else: self.logger.warning(f"Skipping Fib TP level {level}: Raw price {tp_price_raw} not above entry {entry_price}.")
                    else: # SELL
                        # Projecting range downwards from entry
                        tp_price_raw = entry_price - price_range * level
                        if tp_price_raw < entry_price:
                            tp_levels_calculated.append({'price': tp_price_raw, 'percentage': percentage_per_level})
                        else: self.logger.warning(f"Skipping Fib TP level {level}: Raw price {tp_price_raw} not below entry {entry_price}.")

                if not tp_levels_calculated:
                     self.logger.warning("Could not calculate any valid Fibonacci TP levels.")

            except Exception as e:
                self.logger.warning(f"Error calculating Fibonacci TP levels: {e}. Falling back to single ATR TP.")
                tp_levels_calculated = [] # Ensure list is cleared if Fib calculation fails


        # Default: Single Full TP level if partial/Fib not used or failed
        if not tp_levels_calculated:
            self.logger.debug("Calculating single Full TP level.")
            tp_offset = atr * tp_multiple # Use dynamically adjusted multiple
            tp_price_raw = entry_price + tp_offset if signal == "BUY" else entry_price - tp_offset

            if (signal == "BUY" and tp_price_raw > entry_price) or \
               (signal == "SELL" and tp_price_raw < entry_price):
                tp_levels_calculated.append({'price': tp_price_raw, 'percentage': Decimal('1.0')})
                self.logger.debug(f"Calculated Raw Single TP: Price={tp_price_raw:.4f}")
            else:
                self.logger.warning(f"Calculated invalid single raw TP price ({tp_price_raw}). No TP will be set.")


        # --- Final Validation and Risk/Reward Check ---
        if not tp_levels_calculated:
            self.logger.error("No valid TP levels could be calculated.")
            return None, None # Must have at least one TP

        # Snippet 14: Risk-Reward Ratio Filter
        # Use the first TP level for RR calculation
        first_tp_price = tp_levels_calculated[0].get('price')
        if first_tp_price is None:
             self.logger.error("First TP level has no price. Cannot calculate RR.")
             return None, None

        risk_amount = abs(entry_price - sl_price_raw)
        reward_amount = abs(first_tp_price - entry_price)
        min_rr = safe_decimal(self.protection_config.min_rr_ratio, default=Decimal('1.5'))

        if risk_amount <= 0:
            self.logger.error(f"Cannot calculate RR: Risk is zero or negative (Entry={entry_price}, Raw SL={sl_price_raw}).")
            return None, None
        if reward_amount <= 0:
             self.logger.warning(f"Potential reward is zero or negative (Entry={entry_price}, First Raw TP={first_tp_price}). Check TP calculation.")
             # Decide if this should filter the trade or just warn. Filtering seems safer.
             return None, None # Filter if reward is non-positive

        rr_ratio = reward_amount / risk_amount
        if rr_ratio < min_rr:
            self.logger.warning(f"Trade Filtered: Risk/Reward Ratio too low ({rr_ratio:.2f} < {min_rr}). "
                                f"Risk={risk_amount:.4f}, Reward={reward_amount:.4f} (First TP={first_tp_price:.4f}), SL={sl_price_raw:.4f}")
            return None, None # Filter trade based on RR

        self.logger.info(f"Calculated Raw Protections: SL={sl_price_raw:.4f}, Min R:R={rr_ratio:.2f} (>= {min_rr})")
        for i, tp in enumerate(tp_levels_calculated):
             tp_p = tp.get('price', 'N/A')
             tp_pct = tp.get('percentage', 'N/A')
             if isinstance(tp_p, Decimal): tp_p = f"{tp_p:.4f}"
             if isinstance(tp_pct, Decimal): tp_pct = f"{tp_pct:.0%}"
             self.logger.info(f" -> Raw TP Level {i+1}: Price={tp_p}, Percentage={tp_pct}")

        # Return raw prices, formatting happens just before API call
        return tp_levels_calculated, sl_price_raw


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    strategy_engine: VolumaticOBStrategy,
    signal_generator: SignalGenerator
) -> None:
    """Performs one cycle of analysis and trading logic for a single symbol.

    Fetches data, runs strategy, generates signals, manages positions and protections.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol to trade.
        config: The current bot configuration.
        logger: Logger instance for this symbol.
        strategy_engine: Instantiated strategy calculation engine.
        signal_generator: Instantiated signal generation engine.
    """
    lg = logger
    lg.info(f"---== Cycle Start: Analyzing {symbol} ({config.get('interval', 'N/A')}) ==---")
    cycle_start_time = time.monotonic()

    try:
        # --- 1. Get Market Info (Cached) ---
        market_info = get_market_info(exchange, symbol, lg)
        if not market_info or not market_info.get('active', False):
            lg.error(f"Market info error or market inactive for {symbol}. Skipping cycle.")
            return
        # Ensure necessary precision/limits are present (crucial ones)
        if not market_info.get('precision', {}).get('price') or not market_info.get('precision', {}).get('amount'):
             lg.error(f"Price or amount precision missing for {symbol} in market info. Cannot proceed.")
             return
        if not market_info.get('limits', {}).get('amount', {}).get('min'):
             lg.warning(f"Minimum amount limit missing for {symbol}. Sizing might fail if calculated size is very small.")
             # Allow proceeding but be aware sizing might hit issues later

        # --- 2. Fetch Data ---
        ccxt_interval = CCXT_INTERVAL_MAP.get(config.get("interval"))
        if not ccxt_interval:
            lg.error(f"Invalid interval '{config.get('interval')}' mapped to None. Skipping cycle.")
            return
        # Fetch slightly more than minimum needed, let strategy engine handle trimming/dropna
        fetch_limit = max(config.get("fetch_limit", DEFAULT_FETCH_LIMIT), strategy_engine.min_data_len + 50)
        klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, fetch_limit, lg)

        if klines_df.empty or len(klines_df) < strategy_engine.min_data_len:
            lg.warning(f"Insufficient kline data after fetch: {len(klines_df)} rows, require {strategy_engine.min_data_len}. Skipping cycle.")
            return

        # --- 3. Run Strategy Analysis ---
        analysis_results = strategy_engine.update(klines_df)
        if analysis_results is None:
            lg.error("Strategy analysis failed or returned None. Skipping cycle.")
            return

        latest_close = analysis_results.get('last_close')
        current_atr = analysis_results.get('atr')
        if latest_close is None or latest_close <= 0:
             lg.error(f"Invalid last close price ({latest_close}) after analysis. Skipping cycle.")
             return
        # Add latest close to market_info for use in protection formatting heuristics
        market_info['last_price'] = latest_close

        lg.debug(f"Analysis Complete: Last Close={latest_close:.4f}, "
                 f"ATR={current_atr:.4f if current_atr else 'N/A'}, "
                 f"Trend Up={analysis_results.get('current_trend_up')}, "
                 f"Trend Changed={analysis_results.get('trend_just_changed')}")

        # --- 4. Check Existing Position ---
        open_position = get_open_position(exchange, symbol, lg)

        # --- 5. Generate Signal ---
        signal = signal_generator.generate_signal(analysis_results, open_position)
        lg.info(f"Generated Signal for {symbol}: {signal}")

        # --- 6. Trading Execution ---
        if not config.get("enable_trading", False):
            lg.warning("Trading is disabled in config. Skipping execution logic.")
            # Log potential actions if trading were enabled
            if open_position is None and signal in ["BUY", "SELL"]:
                 lg.info(f"[TRADING DISABLED] Would attempt {signal} entry.")
            elif open_position and signal in ["EXIT_LONG", "EXIT_SHORT"]:
                 lg.info(f"[TRADING DISABLED] Would attempt to close {open_position['side']} position.")
            return # Skip all order placement and management

        # ========================================
        # === ENTRY LOGIC ========================
        # ========================================
        if open_position is None and signal in ["BUY", "SELL"]:
            lg.info(f"{BRIGHT}Attempting {signal} entry for {symbol} at ~{latest_close:.4f}...{RESET}")

            # Check Max Concurrent Positions (simple check, assumes only this bot trades)
            # For multi-symbol or multi-instance, need a shared state mechanism
            # A more robust check would involve fetching all positions, which can be slow.
            # This simple check assumes only one position per symbol is managed by *this instance*.
            # active_positions = exchange.fetch_positions() # Fetch all positions (can be slow)
            # active_symbols = {p['symbol'] for p in active_positions if abs(safe_decimal(p.get('contracts', 0))) > 1e-9}
            # if len(active_symbols) >= config.get('max_concurrent_positions', 1):
            #      lg.warning(f"Max concurrent positions ({config.get('max_concurrent_positions', 1)}) reached. Skipping entry for {symbol}.")
            #      return

            # --- Pre-computation for Entry ---
            # Calculate potential TP/SL *before* calculating size to check RR ratio
            # Use latest_close as the hypothetical entry price for this calculation
            tp_levels_raw, initial_sl_raw = signal_generator.calculate_initial_tp_sl(
                latest_close, signal, analysis_results, market_info
            )

            if initial_sl_raw is None or not tp_levels_raw:
                lg.error(f"Trade aborted: Invalid initial TP/SL calculated or R:R filter failed. Raw SL={initial_sl_raw}, Raw TP Levels={tp_levels_raw}")
                return # Abort entry

            # Fetch balance
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= Decimal('0'): # Check balance > 0
                lg.error(f"Trade aborted: Invalid or zero balance ({balance}) for {QUOTE_CURRENCY}.")
                return

            # Set Leverage (only for contracts)
            if market_info.get('is_contract', False):
                leverage_set = set_leverage_ccxt(exchange, symbol, config.get("leverage", 1), market_info, lg)
                if not leverage_set:
                    lg.error("Trade aborted: Failed to set leverage.")
                    return

            # Calculate Position Size using raw SL for accurate risk calc
            position_size = calculate_position_size(
                balance=balance,
                risk_per_trade=config["risk_per_trade"],
                entry_price=latest_close, # Use current price as planned entry for sizing
                stop_loss_price=initial_sl_raw, # Use raw SL for calculation
                market_info=market_info,
                config=config,
                current_atr=current_atr,
                logger=lg
            )

            if position_size is None or position_size <= 0:
                lg.error("Trade aborted: Invalid position size calculated.")
                return

            # --- Place Entry Order ---
            limit_entry_price_raw = None
            if config.get('order_type') == 'limit':
                 # Calculate a suitable limit price (e.g., slightly better than current)
                 price_tick = market_info['precision'].get('price')
                 if price_tick is None or price_tick <= 0:
                      lg.error("Cannot calculate limit entry price: Invalid price tick size. Aborting limit entry.")
                      return # Cannot place limit order without tick size
                 # Place limit order 2 ticks away from current price towards the desired direction
                 offset = price_tick * Decimal('2') # Example: 2 ticks away
                 limit_entry_price_raw = latest_close - offset if signal == "BUY" else latest_close + offset
                 lg.info(f"Calculated Raw Limit Entry Price: {limit_entry_price_raw:.4f}")
                 # Formatting happens inside place_trade

            trade_order = place_trade(
                exchange=exchange,
                symbol=symbol,
                signal=signal, # "BUY" or "SELL"
                position_size=position_size, # Already formatted Decimal
                market_info=market_info,
                config=config,
                logger=lg,
                reduce_only=False,
                limit_price=limit_entry_price_raw # Pass raw limit price (or None for market)
            )

            # --- Post-Entry Processing ---
            if trade_order and trade_order.get('id'):
                # Wait briefly for order to potentially fill and position state to update
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order placed (ID: {trade_order.get('id')}). Waiting {confirm_delay}s for position confirmation...")
                time.sleep(confirm_delay)

                # Confirm position opened
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    lg.info(f"{NEON_GREEN}{signal} position successfully opened/confirmed for {symbol}.{RESET}")
                    entry_price_actual = confirmed_position.get('entryPrice')
                    if entry_price_actual is None or entry_price_actual <= 0:
                        lg.warning(f"Could not get actual entry price from confirmed position. Using planned entry {latest_close:.4f} for protection calculation.")
                        entry_price_actual = latest_close # Fallback

                    # Recalculate TP/SL based on *actual* entry price for potentially better accuracy
                    lg.info(f"Recalculating protection based on actual entry price: {entry_price_actual:.4f}")
                    tp_levels_final_raw, sl_final_raw = signal_generator.calculate_initial_tp_sl(
                        entry_price_actual, signal, analysis_results, market_info
                    )

                    if sl_final_raw is None or not tp_levels_final_raw:
                        lg.error("Failed to recalculate valid SL/TP after entry based on actual price. Manual monitoring required!")
                    else:
                        # --- Set Initial Protections (SL, TP, TSL) ---
                        protection_success = False
                        # Use first TP level for initial setting if multiple exist
                        first_tp_raw = tp_levels_final_raw[0]['price'] if tp_levels_final_raw else None
                        pos_idx = confirmed_position.get('info', {}).get('positionIdx', 0)

                        # Attempt to set protections via _set_position_protection
                        # Pass raw values, formatting happens inside the function
                        protection_params = {
                            'stop_loss_price': sl_final_raw,
                            'take_profit_price': first_tp_raw, # Set first TP initially
                            'position_idx': pos_idx
                        }

                        # Add TSL params if enabled
                        if config.get("protection", {}).get("enable_trailing_stop", False):
                            lg.info("Calculating initial Trailing Stop Loss parameters...")
                            # Calculate TSL distance and activation based on current state (use actual entry)
                            use_atr_tsl = config.get("protection", {}).get("use_atr_trailing_stop", True)
                            tsl_dist_raw = None
                            tsl_act_raw = None

                            if use_atr_tsl and current_atr and current_atr > 0:
                                tsl_mult = safe_decimal(config.get("protection", {}).get("trailing_stop_atr_multiple", 1.5))
                                tsl_dist_raw = current_atr * tsl_mult
                            else:
                                tsl_rate = safe_decimal(config.get("protection", {}).get("trailing_stop_callback_rate", 0.005))
                                tsl_dist_raw = entry_price_actual * tsl_rate # Base distance on actual entry

                            tsl_act_pct = safe_decimal(config.get("protection", {}).get("trailing_stop_activation_percentage", 0.003))
                            tsl_act_offset = entry_price_actual * tsl_act_pct
                            tsl_act_raw = entry_price_actual + tsl_act_offset if signal == "BUY" else entry_price_actual - tsl_act_offset

                            if tsl_dist_raw and tsl_dist_raw > 0:
                                protection_params['trailing_stop_distance'] = tsl_dist_raw
                                if tsl_act_raw and tsl_act_raw > 0:
                                     protection_params['tsl_activation_price'] = tsl_act_raw
                                else:
                                     lg.warning("Calculated invalid TSL activation price, TSL might not activate correctly.")
                            else:
                                 lg.warning("Calculated invalid TSL distance, TSL will not be set.")


                        lg.info(f"Attempting to set initial protection: SL={sl_final_raw:.4f}, TP={first_tp_raw:.4f if first_tp_raw else 'N/A'}, "
                                f"TSL_Dist={protection_params.get('trailing_stop_distance')}, TSL_Act={protection_params.get('tsl_activation_price')}")
                        # Add current price to market_info for rounding hint
                        market_info['last_price'] = latest_close
                        protection_success = _set_position_protection(
                            exchange, symbol, market_info, lg, **protection_params
                        )

                        if not protection_success:
                            lg.warning(f"{NEON_YELLOW}Failed to set initial position protection (SL/TP/TSL) after entry for {symbol}. Manual monitoring advised.{RESET}")

                        # Snippet 4: Place Partial TP Limit Orders (If applicable and initial protection set)
                        # Only place orders for levels *beyond* the first one (which was set via trading-stop)
                        if protection_success and len(tp_levels_final_raw) > 1 and config.get("protection", {}).get("partial_tp_levels"):
                            lg.info("Attempting to place partial TP limit orders (for levels beyond the first)...")
                            total_pos_size = abs(confirmed_position.get('contracts', position_size)) # Use confirmed size
                            placed_percentage_sum = tp_levels_final_raw[0].get('percentage', Decimal('0.0')) # Start with % of first TP

                            # Skip the first TP level as it was set via main protection call
                            for i, tp_level in enumerate(tp_levels_final_raw[1:], start=1):
                                tp_price_raw = tp_level.get('price')
                                level_percentage = safe_decimal(tp_level.get('percentage'), default=Decimal('0.0'))

                                if tp_price_raw is None or level_percentage <= 0:
                                    lg.warning(f"Skipping partial TP level {i+1}: Invalid price or percentage.")
                                    continue

                                # Calculate size for this TP level based on its percentage
                                size_for_this_level = total_pos_size * level_percentage

                                # Check if size is valid and format
                                if size_for_this_level > 0:
                                    formatted_tp_size = format_value_for_exchange(size_for_this_level, 'amount', market_info, lg, ROUND_DOWN)
                                    if formatted_tp_size is None or formatted_tp_size <= 0:
                                         lg.warning(f"Skipping partial TP {i+1}: Formatted size is zero or invalid ({formatted_tp_size}). Raw calc: {size_for_this_level}")
                                         continue

                                    tp_side = "SELL" if signal == "BUY" else "BUY"
                                    lg.info(f"Placing Partial TP {i+1}: Side={tp_side}, Size={formatted_tp_size}, Price={tp_price_raw:.4f}")
                                    try:
                                        # Place reduce-only limit order for the partial TP
                                        tp_order = place_trade(
                                            exchange=exchange, symbol=symbol, signal=tp_side,
                                            position_size=formatted_tp_size, # Pass formatted size
                                            market_info=market_info, config=config, logger=lg,
                                            reduce_only=True, limit_price=tp_price_raw # Pass raw price
                                        )
                                        if tp_order and tp_order.get('id'):
                                            lg.info(f"Partial TP Order {i+1} placed: ID {tp_order.get('id')}")
                                            placed_percentage_sum += level_percentage
                                        else:
                                             lg.error(f"Failed to place partial TP order {i+1} (API response issue).")
                                    except Exception as e:
                                        lg.error(f"Exception placing partial TP order {i+1}: {e}", exc_info=True)
                                else:
                                    lg.warning(f"Skipping partial TP {i+1}: Calculated size is zero ({size_for_this_level}).")
                            lg.info(f"Finished placing partial TP orders. Total percentage covered by orders (incl. first): {placed_percentage_sum:.1%}")
                else:
                    lg.error(f"Failed to confirm position opening for {symbol} after placing order {trade_order.get('id')}. Possible fill delay or order issue. Manual check required.")
                    # Consider attempting to cancel the unconfirmed order? Risky if filled. Needs careful logic.
                    # try:
                    #    lg.warning(f"Attempting to cancel potentially unfilled order {trade_order.get('id')}")
                    #    exchange.cancel_order(trade_order.get('id'), symbol)
                    # except Exception as cancel_e:
                    #    lg.error(f"Failed to cancel order {trade_order.get('id')}: {cancel_e}")
            else:
                 lg.error(f"Order placement failed for {signal} {symbol}. No order returned or ID missing.")


        # ========================================
        # === EXIT & MANAGEMENT LOGIC ==========
        # ========================================
        elif open_position and signal in ["HOLD", "EXIT_LONG", "EXIT_SHORT"]:
            pos_side = open_position['side']
            entry_price = open_position.get('entryPrice') # Should exist if position is open
            pos_size = open_position.get('contracts') # Should exist
            pos_idx = open_position.get('info', {}).get('positionIdx', 0)

            if entry_price is None or pos_size is None:
                 lg.error(f"Cannot manage position {symbol}: Missing entry price or size in position info.")
                 return # Cannot proceed

            lg.debug(f"Managing open {pos_side.upper()} position. Entry: {entry_price:.4f}, Size: {pos_size}")

            # --- Handle Strategy Exit Signal ---
            should_exit = (pos_side == 'long' and signal == "EXIT_LONG") or \
                          (pos_side == 'short' and signal == "EXIT_SHORT")
            if should_exit:
                lg.info(f"{NEON_YELLOW}Strategy exit signal '{signal}' received for {pos_side} position. Closing position...{RESET}")
                close_signal = "SELL" if pos_side == 'long' else "BUY"
                size_to_close = abs(pos_size) # Close the entire position

                # Place market order to close
                # Format size just before placing order
                formatted_close_size = format_value_for_exchange(size_to_close, 'amount', market_info, lg, ROUND_DOWN)
                if formatted_close_size is None or formatted_close_size <= 0:
                     lg.error(f"Failed to format close size {size_to_close}. Cannot place close order.")
                else:
                    close_order = place_trade(
                        exchange, symbol, close_signal, formatted_close_size, market_info, config, lg,
                        reduce_only=True, limit_price=None # Use market order for exit
                    )
                    if close_order:
                        lg.info(f"Position closure order placed for {symbol}.")
                        # Consider cancelling remaining open TP orders if partial TPs were used
                        # cancel_open_tp_orders(exchange, symbol, lg) # Needs implementation
                    else:
                        lg.error(f"Failed to place position closure order for {symbol}. Manual intervention likely required!")
                return # Exit cycle after placing close order

            # --- Handle Time-Based Exit (Snippet 6) ---
            max_holding_minutes = config.get('max_holding_minutes')
            entry_timestamp_ms = open_position.get('entryTimestamp')
            if max_holding_minutes is not None and max_holding_minutes > 0 and entry_timestamp_ms:
                try:
                    entry_time = datetime.fromtimestamp(entry_timestamp_ms / 1000, tz=timezone.utc)
                    # Ensure current_time is timezone-aware (use UTC for comparison)
                    current_time_utc = datetime.now(timezone.utc)
                    holding_duration = current_time_utc - entry_time
                    holding_minutes = holding_duration.total_seconds() / 60

                    if holding_minutes > max_holding_minutes:
                        lg.warning(f"{NEON_YELLOW}Position held for {holding_minutes:.1f} minutes, exceeding max {max_holding_minutes}m. Closing position.{RESET}")
                        close_signal = "SELL" if pos_side == 'long' else "BUY"
                        size_to_close = abs(pos_size)
                        formatted_close_size = format_value_for_exchange(size_to_close, 'amount', market_info, lg, ROUND_DOWN)
                        if formatted_close_size is None or formatted_close_size <= 0:
                            lg.error(f"Failed to format time-based close size {size_to_close}.")
                        else:
                            close_order = place_trade(exchange, symbol, close_signal, formatted_close_size, market_info, config, lg, reduce_only=True, limit_price=None)
                            if close_order: lg.info(f"Time-based position closure order placed for {symbol}.")
                            else: lg.error(f"Failed to place time-based closure order for {symbol}.")
                        return # Exit cycle
                except Exception as time_e:
                     lg.error(f"Error calculating holding time: {time_e}")


            # --- Manage Protections (BE, TSL, OB Trailing) ---
            protection_cfg = config.get("protection", {})
            current_sl_raw = open_position.get('stopLossPrice') # Raw from position info
            current_tp_raw = open_position.get('takeProfitPrice') # Raw from position info
            current_tsl_dist_raw = open_position.get('tslDistance') # Raw from position info
            current_tsl_active_price_raw = open_position.get('tslActivationPrice') # Raw from position info
            is_tsl_set = current_tsl_dist_raw is not None and current_tsl_dist_raw > 0 # Check if TSL parameters are set

            # --- Break-Even Logic ---
            if protection_cfg.get("enable_break_even", False) and current_atr is not None and current_atr > 0:
                profit = (latest_close - entry_price) if pos_side == 'long' else (entry_price - latest_close)
                trigger_atr_multiple = safe_decimal(protection_cfg.get("break_even_trigger_atr_multiple", 1.0), default=Decimal('1.0'))
                be_trigger_threshold = current_atr * trigger_atr_multiple

                # Snippet 16: Dynamic Break-Even Trigger based on duration
                if protection_cfg.get("dynamic_be_trigger", True) and entry_timestamp_ms:
                    try:
                        entry_time = datetime.fromtimestamp(entry_timestamp_ms / 1000, tz=timezone.utc)
                        holding_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
                        # Example: Trigger BE faster (lower threshold) after 60 minutes
                        if holding_minutes > 60:
                            adjustment_factor = Decimal('0.7') # Reduce threshold by 30%
                            dynamic_threshold = be_trigger_threshold * adjustment_factor
                            lg.debug(f"Dynamic BE: Holding > 60m. Adjusted trigger threshold from {be_trigger_threshold:.4f} to {dynamic_threshold:.4f}")
                            be_trigger_threshold = dynamic_threshold
                    except Exception as dyn_be_e:
                         lg.warning(f"Could not apply dynamic BE trigger adjustment: {dyn_be_e}")

                # Check if profit meets threshold
                if profit >= be_trigger_threshold:
                    min_tick = market_info['precision'].get('price')
                    if min_tick is None or min_tick <= 0:
                         lg.warning("Cannot calculate BE SL price: Invalid price tick size.")
                    else:
                        offset_ticks = protection_cfg.get("break_even_offset_ticks", 2)
                        be_offset = min_tick * offset_ticks
                        # Calculate desired BE SL price (raw)
                        be_sl_price_raw = entry_price + be_offset if pos_side == 'long' else entry_price - be_offset

                        # Check if current SL needs updating to BE (compare raw prices before formatting)
                        sl_needs_update_to_be = False
                        if current_sl_raw is None: # No SL exists yet
                            sl_needs_update_to_be = True
                        elif pos_side == 'long' and be_sl_price_raw > current_sl_raw: # Move SL up to BE
                            sl_needs_update_to_be = True
                        elif pos_side == 'short' and be_sl_price_raw < current_sl_raw: # Move SL down to BE
                            sl_needs_update_to_be = True

                        if sl_needs_update_to_be:
                            lg.info(f"{NEON_GREEN}Profit ({profit:.4f}) reached BE threshold ({be_trigger_threshold:.4f}). Moving SL to Break-Even target: {be_sl_price_raw:.4f}{RESET}")
                            # Set *only* the SL, attempt to preserve existing TP/TSL
                            # Pass existing TP/TSL values (raw) to _set_position_protection to *attempt* preservation.
                            market_info['last_price'] = latest_close # Update hint
                            update_success = _set_position_protection(
                                exchange, symbol, market_info, lg,
                                stop_loss_price=be_sl_price_raw, # Pass raw price
                                take_profit_price=current_tp_raw, # Try to keep existing TP (raw)
                                trailing_stop_distance=current_tsl_dist_raw, # Try to keep existing TSL (raw)
                                tsl_activation_price=current_tsl_active_price_raw, # Raw
                                position_idx=pos_idx
                            )
                            if not update_success:
                                 lg.warning("Failed to update SL to Break-Even via API.")
                            # If BE moved SL, the existing TSL (if any) on the exchange might get cancelled or behave unexpectedly.
                            # The next TSL check might reset it if conditions are met again.


            # --- Trailing Stop Activation / Update ---
            if protection_cfg.get("enable_trailing_stop", False):
                # Check if TSL should be activated or potentially updated (dynamic TSL)
                # Call set_trailing_stop_loss which handles activation logic and dynamic updates if configured
                lg.debug("Checking Trailing Stop Loss status/updates...")
                # Pass the original open_position dictionary, as TSL function reads current protections from it
                tsl_set_attempt = set_trailing_stop_loss(
                    exchange, symbol, market_info, open_position, config, lg,
                    current_atr, latest_close
                )
                # The function returns False if TSL is disabled, not activated yet due to delay, or if API call failed
                # We don't need to do much with the return value here unless we want specific logs
                if tsl_set_attempt:
                     lg.debug("TSL update/activation successful or already active.")
                     # If TSL was successfully set/updated, fetch position state again to reflect changes
                     open_position = get_open_position(exchange, symbol, lg)
                     if open_position:
                          current_sl_raw = open_position.get('stopLossPrice')
                          is_tsl_set = open_position.get('tslDistance') is not None and open_position['tslDistance'] > 0
                     else:
                          lg.warning("Position disappeared after TSL update attempt.")
                          return # Position closed, exit cycle


            # --- SL Trailing to OB Boundaries (Snippet 24) ---
            # This logic runs if enabled AND if TSL is NOT currently active (to avoid conflicts)
            # Check the 'is_tsl_set' flag which was updated after the TSL check/update attempt
            if protection_cfg.get("sl_trail_to_ob", False) and not is_tsl_set:
                new_sl_target_raw = None
                nearest_protective_ob = None
                active_bull_obs = analysis_results.get('active_bull_boxes', [])
                active_bear_obs = analysis_results.get('active_bear_boxes', [])

                if pos_side == 'long':
                    # Find highest Bull OB bottom below current price but above entry (acting as support)
                    protective_obs = [b for b in active_bull_obs if b.get('bottom') is not None and b['bottom'] < latest_close and b['bottom'] > entry_price]
                    if protective_obs:
                        nearest_protective_ob = max(protective_obs, key=lambda x: x['bottom'])
                        ob_bottom = nearest_protective_ob.get('bottom')
                        # Target SL slightly below the OB bottom for safety
                        min_tick = market_info['precision'].get('price')
                        if min_tick and ob_bottom:
                            new_sl_target_raw = ob_bottom - min_tick

                else: # Short position
                    # Find lowest Bear OB top above current price but below entry (acting as resistance)
                    protective_obs = [b for b in active_bear_obs if b.get('top') is not None and b['top'] > latest_close and b['top'] < entry_price]
                    if protective_obs:
                        nearest_protective_ob = min(protective_obs, key=lambda x: x['top'])
                        ob_top = nearest_protective_ob.get('top')
                        # Target SL slightly above the OB top
                        min_tick = market_info['precision'].get('price')
                        if min_tick and ob_top:
                            new_sl_target_raw = ob_top + min_tick

                # Check if the new SL target is better than the current SL
                if new_sl_target_raw and nearest_protective_ob:
                    # Use the potentially updated current_sl_raw from after TSL check
                    sl_needs_update_to_ob = False
                    if current_sl_raw is None:
                        sl_needs_update_to_ob = True
                    elif pos_side == 'long' and new_sl_target_raw > current_sl_raw:
                        sl_needs_update_to_ob = True
                    elif pos_side == 'short' and new_sl_target_raw < current_sl_raw:
                        sl_needs_update_to_ob = True

                    if sl_needs_update_to_ob:
                        lg.info(f"{NEON_GREEN}Trailing SL to OB {nearest_protective_ob.get('id','N/A')} boundary: New SL Target = {new_sl_target_raw:.4f}{RESET}")
                        # Update SL, attempting to preserve TP (TSL should be inactive)
                        market_info['last_price'] = latest_close # Update hint
                        update_success = _set_position_protection(
                            exchange, symbol, market_info, lg,
                            stop_loss_price=new_sl_target_raw, # Pass raw
                            take_profit_price=open_position.get('takeProfitPrice'), # Preserve current TP (raw)
                            # Explicitly cancel TSL params if trying to set SL via OB trail
                            trailing_stop_distance=Decimal('0'),
                            tsl_activation_price=Decimal('0'),
                            position_idx=pos_idx
                        )
                        if not update_success:
                            lg.warning("Failed to update SL to OB boundary via API.")


        # --- End of Cycle ---
        lg.info(f"---== Cycle End: {symbol} ({time.monotonic() - cycle_start_time:.2f}s) ==---")

    except ccxt.RateLimitExceeded as e:
        lg.warning(f"Rate limit exceeded during cycle for {symbol}: {e}. Consider increasing loop_delay_seconds.")
        time.sleep(RETRY_DELAY_SECONDS * 2) # Extra sleep on rate limit
    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
        lg.error(f"Network Error during cycle for {symbol}: {e}. Loop will continue after delay.")
        # Retry is handled by decorators on individual API calls, but log cycle interruption
    except ccxt.AuthenticationError as e:
         lg.critical(f"Authentication Error for {symbol}: {e}. Check API keys. Stopping bot.")
         raise SystemExit("Authentication Error") # Stop the bot
    except ccxt.ExchangeError as e:
        # Log specific exchange errors that might indicate bigger issues
        lg.error(f"Unhandled Exchange Error during cycle for {symbol}: {e}. Loop will continue.")
        # Example: Check for margin errors, liquidation engine errors, etc.
        error_str_lower = str(e).lower()
        if "margin" in error_str_lower or "liquidation" in error_str_lower or "reduce-only" in error_str_lower or "position size" in error_str_lower:
            lg.critical(f"Potential critical exchange issue detected affecting position state: {e}")
            # Consider pausing or stopping the bot based on error severity, or trying to resync state
    except Exception as e:
        lg.critical(f"!!! UNHANDLED EXCEPTION in trading cycle for {symbol}: {e} !!!", exc_info=True)
        # Consider adding a mechanism to pause or exit on repeated critical errors


def check_config_reload(logger: logging.Logger) -> bool:
    """Checks if the config file has been modified and reloads if necessary.

    Args:
        logger: The logger instance.

    Returns:
        True if the config was reloaded, False otherwise.
    """
    global CONFIG, config_mtime, QUOTE_CURRENCY
    reloaded = False
    try:
        if not os.path.exists(CONFIG_FILE):
             # If file disappeared after initial load, log error but continue with current config
             if config_mtime > 0: # Only log if it existed before
                  logger.error(f"Config file {CONFIG_FILE} not found during reload check. Continuing with previously loaded config.")
                  config_mtime = 0 # Reset mtime to avoid repeated checks until file reappears
             return False

        current_mtime = os.path.getmtime(CONFIG_FILE)
        # Use a small tolerance for modification time comparison
        if current_mtime > config_mtime + 1.0: # Check if modified more than 1 sec ago
            logger.warning(f"{NEON_YELLOW}Config file '{CONFIG_FILE}' changed. Reloading...{RESET}")
            new_config = load_config(CONFIG_FILE) # load_config handles validation and logging
            if new_config: # Check if loading succeeded
                 CONFIG = new_config
                 QUOTE_CURRENCY = CONFIG.get("quote_currency", DEFAULT_QUOTE_CURRENCY)
                 # Use the actual mtime from the file we just loaded
                 config_mtime = os.path.getmtime(CONFIG_FILE)
                 logger.info("Configuration reloaded successfully.")
                 # NOTE: Strategy/Signal engines might need re-initialization if their params changed significantly.
                 # For simplicity, this example re-initializes them in the main loop after reload.
                 reloaded = True
            else:
                 logger.error("Failed to reload configuration. Continuing with previous settings.")
                 # Keep old mtime to avoid constant reload attempts if file remains invalid
    except FileNotFoundError:
         # Should be handled by the os.path.exists check above, but handle defensively
         logger.error(f"Config file {CONFIG_FILE} not found unexpectedly during reload check.")
         config_mtime = 0
    except Exception as e:
         logger.error(f"Error during config reload check: {e}", exc_info=True)

    return reloaded

def validate_symbol(symbol_input: str, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[str]:
    """Validates symbol format and existence on the exchange.

    Args:
        symbol_input: The symbol string entered by the user.
        exchange: Initialized CCXT exchange object.
        logger: Logger instance.

    Returns:
        The standardized symbol string from CCXT if valid and active, otherwise None.
    """
    try:
        # Basic cleaning: uppercase, remove whitespace
        symbol_cleaned = symbol_input.strip().upper()
        # Standardize separator if needed (e.g., handle both BTC-USDT and BTC/USDT)
        symbol_standard = symbol_cleaned.replace('-', '/')

        # Check basic format
        if '/' not in symbol_standard or len(symbol_standard.split('/')) != 2:
            logger.error(f"Invalid symbol format: '{symbol_input}'. Use BASE/QUOTE format (e.g., BTC/USDT).")
            return None

        # Check against exchange markets (uses get_market_info which handles caching)
        logger.debug(f"Validating symbol '{symbol_standard}' on {exchange.id}...")
        market_info = get_market_info(exchange, symbol_standard, logger)

        if market_info:
            # Use the symbol exactly as returned by CCXT/market_info
            ccxt_symbol = market_info.get('symbol')
            if market_info.get('active'):
                logger.info(f"Symbol {ccxt_symbol} validated successfully.")
                return ccxt_symbol # Return the standardized symbol from CCXT
            else:
                logger.error(f"Symbol {ccxt_symbol} found but is not active on {exchange.id}.")
                return None
        else:
            # get_market_info already logged the error (not found or API error)
            logger.error(f"Symbol '{symbol_standard}' not found or failed to load from {exchange.id}.")
            return None
    except Exception as e:
        logger.error(f"Unexpected error validating symbol '{symbol_input}': {e}", exc_info=True)
        return None


def validate_interval(interval_input: str, logger: logging.Logger) -> Optional[str]:
    """Validates the selected interval against the allowed list.

    Args:
        interval_input: The interval string entered by the user.
        logger: Logger instance.

    Returns:
        The validated interval string if valid, otherwise None.
    """
    interval = interval_input.strip()
    if interval in VALID_INTERVALS:
        logger.info(f"Interval '{interval}' validated.")
        return interval
    else:
        logger.error(f"Invalid interval: '{interval}'. Must be one of {VALID_INTERVALS}")
        return None

def main() -> None:
    """Main function to initialize and run the bot."""
    global CONFIG, QUOTE_CURRENCY, config_mtime # Allow main to update global config if needed

    # Use a dedicated logger for initialization phase
    init_logger = setup_logger("BotInit")
    init_logger.info(f"{BRIGHT}{NEON_GREEN}--- Initializing Pyrmethus Volumatic Bot ---{RESET}")
    init_logger.info(f"Timestamp: {datetime.now(TIMEZONE)}")
    init_logger.info(f"Using config file: {os.path.abspath(CONFIG_FILE)}")
    init_logger.info(f"Trading Enabled: {CONFIG.get('enable_trading')}")
    init_logger.info(f"Using Sandbox: {CONFIG.get('use_sandbox')}")
    config_mtime = os.path.getmtime(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else 0 # Initialize mtime

    # --- Live Trading Confirmation ---
    if CONFIG.get("enable_trading") and not CONFIG.get("use_sandbox"):
        init_logger.warning(f"{BRIGHT}{NEON_RED}--- LIVE TRADING IS ENABLED ON REAL ACCOUNT ---{RESET}")
        try:
            print(f"{NEON_YELLOW}Disclaimer: Trading involves risks. Ensure you understand the strategy and risks.")
            print(f"The bot will trade using API Key starting with: {API_KEY[:5]}...{API_KEY[-4:] if API_KEY else 'N/A'}")
            confirmation = input(f"Type 'confirm-live' to proceed with live trading, or anything else to abort: {RESET}").lower()
            if confirmation != 'confirm-live':
                init_logger.info("Live trading aborted by user confirmation.")
                print("Exiting.")
                return
            init_logger.info("Live trading confirmed by user.")
        except (KeyboardInterrupt, EOFError):
            init_logger.info("User interrupted confirmation. Exiting.")
            print("Exiting.")
            return

    # --- Initialize Exchange ---
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical("Failed to initialize exchange. Exiting.")
        print("Exiting due to exchange initialization failure.")
        return

    # --- Symbol and Interval Selection ---
    target_symbol: Optional[str] = None
    while not target_symbol:
        try:
            symbol_input = input(f"Enter symbol to trade (e.g., BTC/USDT): ").strip()
            if not symbol_input: continue
            target_symbol = validate_symbol(symbol_input, exchange, init_logger)
        except (KeyboardInterrupt, EOFError):
            init_logger.info("User interrupted symbol selection. Exiting.")
            print("Exiting.")
            return

    # --- Interval Selection / Confirmation ---
    selected_interval = CONFIG.get('interval', DEFAULT_INTERVAL) # Get default/loaded interval
    while True:
        try:
            interval_prompt = f"Enter interval [{'/'.join(VALID_INTERVALS)}] (Press Enter for config value: {selected_interval}): "
            interval_input = input(interval_prompt).strip()
            if not interval_input: # User pressed Enter, use config value
                 validated_interval = selected_interval
                 init_logger.info(f"Using interval from config: {validated_interval}")
                 break
            else: # User entered something, validate it
                 validated_interval = validate_interval(interval_input, init_logger)
                 if validated_interval:
                     # Update global config if changed via input for this session
                     if validated_interval != CONFIG.get('interval'):
                          init_logger.info(f"Using interval '{validated_interval}' for this session (overriding config value '{CONFIG.get('interval')}').")
                          CONFIG["interval"] = validated_interval # Update runtime config
                     selected_interval = validated_interval # Update for future loops if needed
                     break
                 # If validation failed, loop continues
        except (KeyboardInterrupt, EOFError):
            init_logger.info("User interrupted interval selection. Exiting.")
            print("Exiting.")
            return

    # --- Initialize Strategy Components ---
    # Sanitize symbol for logger name (replace '/')
    symbol_logger_name = f"Trader_{target_symbol.replace('/', '_')}"
    symbol_logger = setup_logger(symbol_logger_name) # Get logger specific to the symbol
    symbol_logger.info(f"Initializing strategy for {target_symbol} on {selected_interval} interval...")

    # Re-fetch/confirm market info for the chosen symbol
    market_info = get_market_info(exchange, target_symbol, symbol_logger)
    if not market_info:
        symbol_logger.critical(f"Failed to get market info for selected symbol {target_symbol} after selection. Exiting.")
        print(f"Exiting due to market info failure for {target_symbol}.")
        return

    # Initialize Strategy and Signal Engines with the final config and market info
    strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger, exchange)
    signal_generator = SignalGenerator(CONFIG, symbol_logger)

    # --- Main Trading Loop ---
    symbol_logger.info(f"{BRIGHT}{NEON_GREEN}--- Starting Trading Loop for {target_symbol} ---{RESET}")
    try:
        while True:
            loop_start_timestamp = time.time()

            # --- Configuration Reload Check ---
            if check_config_reload(symbol_logger):
                # Re-initialize engines if config was reloaded? Essential if params changed.
                symbol_logger.info("Re-initializing strategy and signal generator due to config reload...")
                # Re-fetch market info as limits/precision might change (less likely, but possible)
                # Use the newly selected symbol and potentially updated config interval
                current_target_symbol = target_symbol # Keep the symbol selected at start
                current_interval = CONFIG['interval'] # Use reloaded interval
                market_info = get_market_info(exchange, current_target_symbol, symbol_logger)
                if not market_info:
                     symbol_logger.error(f"Failed to get market info for {current_target_symbol} after config reload. Skipping cycle.")
                     # Delay before next cycle to avoid rapid failures
                     time.sleep(CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS))
                     continue
                # Pass the new CONFIG object
                strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger, exchange)
                signal_generator = SignalGenerator(CONFIG, symbol_logger)
                symbol_logger.info("Re-initialization complete.")


            # --- Core Analysis and Trading ---
            # analyze_and_trade_symbol now contains the main logic for one cycle
            analyze_and_trade_symbol(
                exchange,
                target_symbol, # Use the symbol selected at the start
                CONFIG,        # Use the potentially reloaded config
                symbol_logger,
                strategy_engine,
                signal_generator
            )

            # --- Loop Delay ---
            elapsed_time = time.time() - loop_start_timestamp
            loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_time = max(0, loop_delay - elapsed_time)
            symbol_logger.debug(f"Main loop cycle took {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s (Loop Delay: {loop_delay}s).")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except SystemExit as e:
         symbol_logger.critical(f"System exit requested: {e}")
    except Exception as e:
         symbol_logger.critical(f"Critical unhandled exception in main loop: {e}", exc_info=True)
    finally:
        print(f"\n{NEON_PURPLE}--- Bot Shutdown Sequence Initiated ---{RESET}")
        symbol_logger.info("--- Bot Shutdown Sequence Initiated ---")

        # Optional: Attempt to close open positions or cancel orders on shutdown
        # This requires careful implementation and error handling
        if CONFIG.get("enable_trading") and exchange and market_info: # Check if resources are available
            try:
                symbol_logger.warning("Attempting to close open positions/orders on shutdown...")
                # Fetch position one last time
                open_pos = get_open_position(exchange, target_symbol, symbol_logger)
                if open_pos:
                    pos_side = open_pos['side']
                    pos_size = open_pos.get('contracts')
                    if pos_size:
                        symbol_logger.warning(f"Closing open {pos_side} position for {target_symbol} (Size: {pos_size})...")
                        close_sig = "SELL" if pos_side == 'long' else "BUY"
                        close_size_formatted = format_value_for_exchange(abs(pos_size), 'amount', market_info, symbol_logger, ROUND_DOWN)
                        if close_size_formatted and close_size_formatted > 0:
                             place_trade(exchange, target_symbol, close_sig, close_size_formatted, market_info, CONFIG, symbol_logger, reduce_only=True)
                             time.sleep(2) # Brief pause for order processing
                        else:
                             symbol_logger.error("Could not format close size during shutdown.")
                    else:
                        symbol_logger.warning("Position found but size is invalid, cannot close.")
                else:
                     symbol_logger.info(f"No open position found for {target_symbol} during shutdown.")

                # Cancel remaining open orders (TPs, potentially pending entries)
                symbol_logger.info(f"Cancelling any remaining open orders for {target_symbol}...")
                # open_orders = exchange.fetch_open_orders(target_symbol) # Can be slow, use carefully
                # Simplified: cancel all orders for the symbol
                try:
                    cancel_response = exchange.cancel_all_orders(target_symbol)
                    symbol_logger.debug(f"Cancel all orders response: {cancel_response}")
                    symbol_logger.info("Open orders cancellation attempt completed.")
                except ccxt.ExchangeError as cancel_err:
                     symbol_logger.error(f"Error cancelling orders during shutdown: {cancel_err}")

            except Exception as shutdown_e:
                symbol_logger.error(f"Error during shutdown cleanup: {shutdown_e}", exc_info=True)

        if exchange and hasattr(exchange, 'close'):
            try:
                exchange.close()
                symbol_logger.info("Exchange connection closed.")
            except Exception as e:
                symbol_logger.error(f"Error closing exchange connection: {e}")

        logging.shutdown()
        print(f"{NEON_PURPLE}--- Bot Shutdown Complete ---{RESET}")

if __name__ == "__main__":
    # Standard entry point execution
    main()
