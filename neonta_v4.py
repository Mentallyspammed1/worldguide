

    asyncio.run(main())
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 678, in run_until_complete
    self.run_forever()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 645, in run_forever
    self._run_once()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 1999, in _run_once
    handle._run()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 2536, in main
    main_logger.info(f"Orderbook Limit: {CONFIG.orderbook_settings.limit} levels")
Message: 'Orderbook Limit: 50 levels'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.12/logging/__init__.py", line 1160, in emit
    msg = self.format(record)
          ^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/logging/__init__.py", line 999, in format
    return fmt.format(record)
           ^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 365, in format
    formatter = logging.Formatter(log_fmt, self.datefmt, self.style)
                                                         ^^^^^^^^^^
AttributeError: 'ColorStreamFormatter' object has no attribute 'style'. Did you mean: '_style'?
Call stack:
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 2588, in <module>
    asyncio.run(main())
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 678, in run_until_complete
    self.run_forever()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 645, in run_forever
    self._run_once()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 1999, in _run_once
    handle._run()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 2537, in main
    main_logger.info(f"Timezone: {APP_TIMEZONE}")
Message: 'Timezone: America/Chicago'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.12/logging/__init__.py", line 1160, in emit
    msg = self.format(record)
          ^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/logging/__init__.py", line 999, in format
    return fmt.format(record)
           ^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 365, in format
    formatter = logging.Formatter(log_fmt, self.datefmt, self.style)
                                                         ^^^^^^^^^^
AttributeError: 'ColorStreamFormatter' object has no attribute 'style'. Did you mean: '_style'?
Call stack:
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 2588, in <module>
    asyncio.run(main())
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 678, in run_until_complete
    self.run_forever()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 645, in run_forever
    self._run_once()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 1999, in _run_once
    handle._run()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 2257, in run_analysis_loop
    logger_instance.info(f"Starting analysis loop for {symbol} with interval {interval_config}...")
Message: 'Starting analysis loop for TRUMP/USDT with interval 5...'
Arguments: ()
/data/data/com.termux/files/usr/lib/python3.12/multiprocessing/pool.py:48: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '[ 30863.76743333 116768.70643333  73285.9556      16078.98333333
  29496.35466667  48945.37773333  94515.72176667 123993.275
  71800.974      217233.06366667  75737.6062     157477.7144
 341667.73666667  23412.7712      90534.909      102656.84066667
 111171.60216667  33738.189       40792.14626667  18284.17666667
  25715.053       21972.5532      71493.12753333  32384.583
  71928.13046667  67600.6752      43356.4705      13659.0624
  25268.0521      26490.1362      21803.10933333  48580.28486667
  14179.91666667  12154.765       52602.5664      22578.62656667
  78584.35613333  45580.58266667 368367.89093333 522643.1648
 570697.85483333  97827.9768      70724.03516667 252925.8389
 343949.1346     109646.524      101047.0912     263812.11166667
 197421.9616     236303.5232     124613.87233333  80119.29986667
  93888.29233333  37846.3308      60345.1612     438909.231
 181142.52        92759.04346667 177797.88693333  33949.012
  53198.748       84331.69246667 174616.84266667 143190.5748
  67562.32923333 158750.87953333  67651.03833333  21836.8026
  17412.10613333  34508.5         32485.9393      65043.8934
  36947.69866667  64467.34        46254.9475      35855.85
  65893.232       72213.94206667  72745.58206667  59084.27983333
  39312.375       24004.97533333  89388.46716667  38583.7045
  45147.71        57347.1846      45974.44413333  51325.67786667
  75082.37453333 521695.40366667  38324.176       25733.05933333
  52953.55466667  50769.05856667 133806.3364     274149.01563333
  48557.30966667 116163.9024     172496.92883333]' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  return list(map(*args))
/data/data/com.termux/files/usr/lib/python3.12/multiprocessing/pool.py:48: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '[ 20060.3416      31227.26186667  32181.0983      50126.648
  36416.8768      69124.9176     119885.23306667  30624.9039
  48099.0275      39890.95356667  92865.08546667  27979.9213
  37261.02453333  97366.184       19262.2785      44714.0657
  46731.618      116472.07        44080.00013333  41401.2084
  36281.2404      22615.23383333  31802.0632      72119.5203
  50523.2573      50943.122       16518.535       14897.6685
   5917.26196667   9883.192       30832.34        26229.97433333
  96883.5685       8149.46366667  65236.6136      66922.85896667
 264319.78906667  62658.41316667  66530.82146667  83448.2497
  47976.72        42413.59466667 129670.3713     204050.25
 138846.0815      73535.14666667  83745.6594      69040.09966667
  56668.0898      31015.1969      68711.3426      33767.12173333
  97522.047       67959.355      113273.556       29630.0898
 137025.6832      38573.021       63946.666       45951.16783333
 168385.448       44624.60263333  31132.01753333 127226.47
  89492.97096667  86815.0784      75406.5109     101350.9048
 390458.34583333 235979.9        130718.40716667  40713.10493333
  96439.0152      28384.82746667  61425.4233      52867.4965
  78078.0344      39816.57983333  57754.6684      49940.332
  34455.57413333  35677.8184      35635.7391      52893.16926667
 192244.7289      87284.48373333 255021.67386667  34773.6195
  76669.977       40351.3472      49940.72293333 178342.30626667
  18087.54326667 150093.048       44189.97053333  37100.5856    ]' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  return list(map(*args))
/data/data/com.termux/files/home/worldguide/neonta_v4.py:1437: DeprecationWarning: In future, it will be an error for 'np.bool' scalars to be interpreted as an index
  details = f"S:{format_decimal(ema_short)} {'><'[ema_short > ema_long]} L:{format_decimal(ema_long)}"
--- Logging error ---
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.12/logging/__init__.py", line 1160, in emit
    msg = self.format(record)
          ^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/logging/__init__.py", line 999, in format
    return fmt.format(record)
           ^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 365, in format
    formatter = logging.Formatter(log_fmt, self.datefmt, self.style)
                                                         ^^^^^^^^^^
AttributeError: 'ColorStreamFormatter' object has no attribute 'style'. Did you mean: '_style'?
Call stack:
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 2588, in <module>
    asyncio.run(main())
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 678, in run_until_complete
    self.run_forever()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 645, in run_forever
    self._run_once()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/base_events.py", line 1999, in _run_once
    handle._run()
  File "/data/data/com.termux/files/usr/lib/python3.12/asyncio/events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "/data/data/com.termux/files/home/worldguide/neonta_v4.py", line 2329, in run_analysis_loop
    logger_instance.info(output_string)
Message: '\n--- Analysis Report for \x1b[35mTRUMP/USDT\x1b[0m --- (2025-05-01 04:09:46 CDT)\n\x1b[36mInterval:\x1b[0m 0 days 00:05:00    \x1b[36mCurrent Price:\x1b[0m $13.1700\nEMA Align (12/26): \x1b[92mBullish\x1b[0m (S:13.16 < L:13.15)\nADX Trend (14) (12.17): \x1b[33mRanging\x1b[0m (ADX < 25)\nPSAR (13.0976): \x1b[92mBullish\x1b[0m\nRSI (14) (52.28): \x1b[33mNeutral\x1b[0m\nMFI (14) (62.10): \x1b[33mNeutral\x1b[0m\nCCI (20) (80.63): \x1b[33mNeutral\x1b[0m\nWilliams %R (14) (-35.71): \x1b[33mNeutral\x1b[0m\nStochRSI (3/3) (K:74.70 D:83.06): \x1b[91mBearish\x1b[0m (Crossed Down)\nMACD (12/26/9) (L:0.0104 S:0.0056): \x1b[92mAbove Signal\x1b[0m (Line > Signal)\nBBands (20/2.0) (13.17): \x1b[33mWithin Bands\x1b[0m (L:13.08 M:13.14 U:13.21)\nVolume vs MA(20) (2816): \x1b[35mLow Volume\x1b[0m (Vol:2816 MA:7167)\nOBV Trend (219458): \x1b[33mFlat\x1b[0m\nA/D Osc Trend: \x1b[33mN/A\x1b[0m\n\x1b[36m\n--- Levels & Orderbook ---\x1b[0m\nPivot Point: $13.0700\nNearest Support:\n  > Fib 38.2% (Retrace Down): $13.1286\n  > Fib 50.0% (Retrace Down): $13.0200\n  > Fib 61.8% (Retrace Down): $12.9114\nNearest Resistance:\n  > Fib 23.6% (Retrace Down): $13.2629\n  > Period High: $13.4800\n  > R1: $13.5800\n\nOB Pressure (Top 50): \x1b[33mNeutral Pressure\x1b[0m\nOB Value (Bids): $916614\nOB Value (Asks): $763740\n\x1b[35mSignificant OB Clusters (Top 5):\x1b[0m\n\x1b[92m  Support near Pivot ($13.0700) - Value: $114843\x1b[0m\n\x1b[92m  Support near Fib 38.2% (Retrace Down) ($13.1286) - Value: $104736\x1b[0m\n\x1b[91m  Resistance near Fib 23.6% (Retrace Down) ($13.2629) - Value: $71663\x1b[0m\n\x1b[92m  Support near Fib 50.0% (Retrace Down) ($13.0200) - Value: $32166\x1b[0m\n\x1b[92m  Support near Fib 61.8% (Retrace Down) ($12.9114) - Value: $25123\x1b[0m\n'
Arguments: ()

# -*- coding: utf-8 -*-
"""
Neonta v3: Cryptocurrency Technical Analysis Bot

This script performs technical analysis on cryptocurrency pairs using data
fetched from the Bybit exchange via the ccxt library. It calculates various
technical indicators, identifies potential support/resistance levels, analyzes
order book data, and provides an interpretation of the market state.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, getcontext
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.exceptions import RequestException

# --- Initialization ---
init(autoreset=True)  # Initialize colorama
load_dotenv()         # Load environment variables from .env file
getcontext().prec = 18  # Set global Decimal precision

# --- Constants ---
CONFIG_FILE_NAME = "config.json"
LOG_DIRECTORY_NAME = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default if not set or invalid
MAX_API_RETRIES = 5
INITIAL_RETRY_DELAY_SECONDS = 5.0
MAX_RETRY_DELAY_SECONDS = 60.0
CCXT_TIMEOUT_MS = 20000 # Milliseconds for CCXT requests
MAX_JITTER_FACTOR = 0.2 # Max jitter for sleep intervals (e.g., 0.1 = 10%)

# Bybit API Configuration (Ensure these are set in your .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file")

API_ENV = os.getenv("API_ENVIRONMENT", "prod").lower() # 'prod' or 'test'
IS_TESTNET = API_ENV == 'test'

# Timezone Configuration
try:
    APP_TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR))
except ZoneInfoNotFoundError:
    print(f"{Fore.YELLOW}Warning: Timezone '{os.getenv('TIMEZONE', DEFAULT_TIMEZONE_STR)}' not found. Using UTC.{Style.RESET_ALL}")
    APP_TIMEZONE = ZoneInfo("UTC")

# Paths
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = BASE_DIR / CONFIG_FILE_NAME
LOG_DIRECTORY = BASE_DIR / LOG_DIRECTORY_NAME
LOG_DIRECTORY.mkdir(exist_ok=True) # Ensure log directory exists

# Timeframes (Mapping user input to CCXT intervals)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M"
}
REVERSE_CCXT_INTERVAL_MAP = {v: k for k, v in CCXT_INTERVAL_MAP.items()}

# Color Constants Enum
class Color(Enum):
    """Enum for storing colorama color codes."""
    GREEN = Fore.LIGHTGREEN_EX
    BLUE = Fore.CYAN
    PURPLE = Fore.MAGENTA
    YELLOW = Fore.YELLOW
    RED = Fore.LIGHTRED_EX
    RESET = Style.RESET_ALL

    @staticmethod
    def format(text: str, color: 'Color') -> str:
        """Formats text with the specified color."""
        return f"{color.value}{text}{Color.RESET.value}"

# Signal States Enum
class SignalState(Enum):
    """Enum representing various analysis signal states."""
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    STRONG_BULLISH = "Strong Bullish"
    STRONG_BEARISH = "Strong Bearish"
    RANGING = "Ranging"
    OVERBOUGHT = "Overbought"
    OVERSOLD = "Oversold"
    ABOVE_SIGNAL = "Above Signal"
    BELOW_SIGNAL = "Below Signal"
    BREAKOUT_UPPER = "Breakout Upper"
    BREAKDOWN_LOWER = "Breakdown Lower"
    WITHIN_BANDS = "Within Bands"
    HIGH_VOLUME = "High Volume"
    LOW_VOLUME = "Low Volume"
    AVERAGE_VOLUME = "Average Volume"
    INCREASING = "Increasing"
    DECREASING = "Decreasing"
    FLAT = "Flat"
    ACCUMULATION = "Accumulation"
    DISTRIBUTION = "Distribution"
    FLIP_BULLISH = "Flip Bullish"
    FLIP_BEARISH = "Flip Bearish"
    NONE = "None" # Explicitly no signal detected
    NA = "N/A"   # Data or calculation unavailable

# --- Configuration Loading ---
@dataclass
class IndicatorSettings:
    """Settings for technical indicators."""
    default_interval: str = "15"
    momentum_period: int = 10
    volume_ma_period: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    stoch_rsi_period: int = 14
    stoch_k_period: int = 3
    stoch_d_period: int = 3
    cci_period: int = 20
    williams_r_period: int = 14
    mfi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_bands_period: int = 20
    bollinger_bands_std_dev: float = 2.0
    ema_short_period: int = 12
    ema_long_period: int = 26
    sma_short_period: int = 10
    sma_long_period: int = 50
    adx_period: int = 14
    psar_step: float = 0.02
    psar_max_step: float = 0.2

@dataclass
class AnalysisFlags:
    """Flags to enable/disable specific analysis checks."""
    ema_alignment: bool = True
    momentum_crossover: bool = False # Requires more complex logic, disabled by default
    volume_confirmation: bool = True
    rsi_divergence: bool = False # Basic check implemented, disabled by default (prone to false signals)
    macd_divergence: bool = True # Basic check implemented
    stoch_rsi_cross: bool = True
    rsi_threshold: bool = True
    mfi_threshold: bool = True
    cci_threshold: bool = True
    williams_r_threshold: bool = True
    macd_cross: bool = True
    bollinger_bands_break: bool = True
    adx_trend_strength: bool = True
    obv_trend: bool = True
    adi_trend: bool = True # Uses ADOSC (A/D Oscillator)
    psar_flip: bool = True

@dataclass
class Thresholds:
    """Threshold values for oscillator indicators."""
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    mfi_overbought: int = 80
    mfi_oversold: int = 20
    cci_overbought: int = 100
    cci_oversold: int = -100
    williams_r_overbought: int = -20 # Note: Higher value is Overbought for Williams %R
    williams_r_oversold: int = -80  # Note: Lower value is Oversold for Williams %R
    adx_trending: int = 25

@dataclass
class OrderbookSettings:
    """Settings for order book analysis."""
    limit: int = 50 # Number of bids/asks levels to fetch
    cluster_threshold_usd: int = 10000 # Minimum USD value to consider a cluster significant
    cluster_proximity_pct: float = 0.1 # Proximity percentage around levels to check for clusters (e.g., 0.1 = +/- 0.1%)

@dataclass
class LoggingSettings:
    """Configuration for logging."""
    level: str = "INFO"
    rotation_max_bytes: int = 10 * 1024 * 1024 # 10 MB
    rotation_backup_count: int = 5

@dataclass
class AppConfig:
    """Main application configuration structure."""
    analysis_interval_seconds: int = 30
    kline_limit: int = 200 # Number of candles to fetch for analysis
    indicator_settings: IndicatorSettings = field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = field(default_factory=AnalysisFlags)
    thresholds: Thresholds = field(default_factory=Thresholds)
    orderbook_settings: OrderbookSettings = field(default_factory=OrderbookSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def _dataclass_to_dict(cls, dc_instance) -> dict:
        """Recursively converts a dataclass instance to a dictionary."""
        if not hasattr(dc_instance, "__dataclass_fields__"):
            return dc_instance
        result = {}
        for f in dc_instance.__dataclass_fields__:
            value = getattr(dc_instance, f)
            if hasattr(value, "__dataclass_fields__"):
                result[f] = cls._dataclass_to_dict(value)
            elif isinstance(value, list):
                 result[f] = [cls._dataclass_to_dict(i) for i in value]
            else:
                result[f] = value
        return result

    @classmethod
    def _merge_dicts(cls, default: dict, user: dict) -> dict:
        """Recursively merges user dict into default dict, only updating existing keys."""
        merged = default.copy()
        for key, user_value in user.items():
            if key in merged:
                default_value = merged[key]
                if isinstance(user_value, dict) and isinstance(default_value, dict):
                    merged[key] = cls._merge_dicts(default_value, user_value)
                else:
                    # Overwrite default value with user value if key exists
                    merged[key] = user_value
            # else: Ignore keys from user config that are not in the default structure
        return merged

    @classmethod
    def _dict_to_dataclass(cls, data_class, data_dict):
        """Converts a dictionary to a nested dataclass structure, respecting defaults."""
        field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
        init_args = {}
        for name, type_hint in field_types.items():
            if name in data_dict:
                value = data_dict[name]
                # If the field type is a dataclass, recursively convert the sub-dict
                if hasattr(type_hint, '__dataclass_fields__') and isinstance(value, dict):
                    init_args[name] = cls._dict_to_dataclass(type_hint, value)
                else:
                    # Attempt type conversion if necessary (e.g., str to int/float)
                    try:
                        if isinstance(getattr(data_class(), name, None), (int, float)) and isinstance(value, (str, int, float)):
                             init_args[name] = type_hint(value)
                        else:
                             init_args[name] = value
                    except (ValueError, TypeError):
                         print(Color.format(f"Warning: Could not convert config value '{value}' for key '{name}' to type {type_hint}. Using value as is.", Color.YELLOW))
                         init_args[name] = value # Use value as is if conversion fails
            # else: Rely on default_factory or default value defined in dataclass
        try:
            return data_class(**init_args)
        except TypeError as e:
             print(Color.format(f"Error creating dataclass {data_class.__name__} from config: {e}", Color.RED))
             print(Color.format("Using default values for this section.", Color.YELLOW))
             return data_class() # Return default instance on error

    @classmethod
    def load(cls, filepath: Path) -> 'AppConfig':
        """Loads configuration from a JSON file, merging with defaults."""
        default_config_obj = cls()
        default_config_dict = cls._dataclass_to_dict(default_config_obj)

        if not filepath.exists():
            print(Color.format(f"Config file '{filepath}' not found.", Color.YELLOW))
            try:
                with filepath.open('w', encoding="utf-8") as f:
                    json.dump(default_config_dict, f, indent=2, ensure_ascii=False)
                print(Color.format(f"Created new config file '{filepath}' with default settings.", Color.GREEN))
                return default_config_obj # Return the default dataclass object
            except IOError as e:
                print(Color.format(f"Error creating default config file '{filepath}': {e}", Color.RED))
                print(Color.format("Loading internal defaults.", Color.YELLOW))
                return default_config_obj

        try:
            with filepath.open("r", encoding="utf-8") as f:
                user_config = json.load(f)

            if not isinstance(user_config, dict):
                raise TypeError("Config file does not contain a valid JSON object.")

            merged_config_dict = cls._merge_dicts(default_config_dict, user_config)
            # Convert the final merged dict back into the nested dataclass structure
            loaded_config = cls._dict_to_dataclass(cls, merged_config_dict)
            print(Color.format(f"Successfully loaded configuration from '{filepath}'.", Color.GREEN))
            return loaded_config

        except (FileNotFoundError, json.JSONDecodeError, TypeError, IOError) as e:
            print(Color.format(f"Error loading/parsing config file '{filepath}': {e}", Color.RED))
            print(Color.format("Loading internal defaults.", Color.YELLOW))
            return default_config_obj
        except Exception as e:
            print(Color.format(f"Unexpected error loading config: {e}", Color.RED))
            traceback.print_exc()
            print(Color.format("Loading internal defaults.", Color.YELLOW))
            return default_config_obj

CONFIG = AppConfig.load(CONFIG_FILE_PATH)

# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that masks API key/secret and removes color codes for file logging."""
    _color_code_regex = re.compile(r'\x1b\[[0-9;]*m')

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, masking secrets and removing color."""
        msg = super().format(record)
        # Mask secrets
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        # Remove color codes
        msg_no_color = self._color_code_regex.sub('', msg)
        return msg_no_color

class ColorStreamFormatter(logging.Formatter):
    """Formatter that adds colors for stream (console) output."""
    _level_color_map = {
        logging.DEBUG: Color.PURPLE.value,
        logging.INFO: Color.GREEN.value,
        logging.WARNING: Color.YELLOW.value,
        logging.ERROR: Color.RED.value,
        logging.CRITICAL: Color.RED.value + Style.BRIGHT,
    }
    _asctime_color = Color.BLUE.value
    _symbol_color = Color.YELLOW.value
    _reset_color = Color.RESET.value

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style='%', symbol: str = "GENERAL"):
        # We don't use the fmt argument directly, format is constructed in format()
        super().__init__(fmt=None, datefmt=datefmt, style=style)
        self.symbol = symbol

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with appropriate colors."""
        level_color = self._level_color_map.get(record.levelno, self._reset_color)

        # Create a temporary formatter for this record to handle standard fields
        log_fmt = (
            f"{self._asctime_color}%(asctime)s{self._reset_color} "
            f"[{level_color}%(levelname)-8s{self._reset_color}] "
            f"{self._symbol_color}{self.symbol}{self._reset_color} - "
            f"%(message)s"
        )
        # Use a standard Formatter instance internally for actual formatting
        # This avoids recalculating the format string parts constantly
        formatter = logging.Formatter(log_fmt, self.datefmt, self.style)
        return formatter.format(record)

def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger instance for a specific symbol with file and stream handlers."""
    log_filename = LOG_DIRECTORY / f"{symbol.replace('/', '_')}_{datetime.now(APP_TIMEZONE).strftime('%Y%m%d')}.log"
    logger = logging.getLogger(symbol)

    # Determine log level from config
    log_level_str = CONFIG.logging.level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (with rotation and sensitive data masking)
    try:
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=CONFIG.logging.rotation_max_bytes,
            backupCount=CONFIG.logging.rotation_backup_count,
            encoding='utf-8'
        )
        # Use ISO 8601 format for file logs
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt='%Y-%m-%dT%H:%M:%S%z')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file by default, logger level controls overall
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        print(Color.format(f"Error setting up file logger for {symbol}: {e}. File logging disabled.", Color.RED))
    except Exception as e:
        print(Color.format(f"Unexpected error setting up file logger for {symbol}: {e}", Color.RED))
        traceback.print_exc()

    # Stream Handler (with colors)
    stream_handler = logging.StreamHandler()
    stream_formatter = ColorStreamFormatter(
        datefmt='%Y-%m-%d %H:%M:%S', # Use a more readable date format for console
        symbol=symbol
    )
    stream_handler.setFormatter(stream_formatter)
    # Stream handler level should respect the overall logger level set from config
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    # Prevent propagation to root logger if it has handlers (avoids duplicate console logs)
    logger.propagate = False

    return logger

# --- Utility Functions ---
async def async_sleep_with_jitter(seconds: float, max_jitter_factor: float = MAX_JITTER_FACTOR) -> None:
    """Asynchronous sleep with random jitter to avoid thundering herd."""
    if seconds <= 0:
        return
    jitter = np.random.uniform(0, seconds * max_jitter_factor)
    await asyncio.sleep(seconds + jitter)

def format_decimal(value: Optional[Union[Decimal, float, int, str]], precision: int = 2) -> str:
    """
    Safely formats a numeric value (or string representation) as a Decimal string
    with specified precision. Handles None, NaN, and conversion errors gracefully.
    """
    if value is None or pd.isna(value):
        return "N/A"
    try:
        # Convert to string first to handle floats accurately
        decimal_value = Decimal(str(value))
        # Use quantize for proper Decimal rounding based on precision
        quantizer = Decimal('1e-' + str(precision))
        return str(decimal_value.quantize(quantizer))
    except (InvalidOperation, ValueError, TypeError):
        # Fallback for values that cannot be converted to Decimal
        try:
            # Attempt to format as float if Decimal fails
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError):
            return str(value) # Last resort: simple string conversion

# --- CCXT Client ---
class BybitCCXTClient:
    """
    Asynchronous CCXT client specifically for Bybit V5 API with robust error
    handling, retry logic, and market loading/management.
    """
    def __init__(self, api_key: str, api_secret: str, is_testnet: bool, logger_instance: logging.Logger):
        """
        Initializes the Bybit CCXT client.

        Args:
            api_key: The Bybit API key.
            api_secret: The Bybit API secret.
            is_testnet: Boolean indicating whether to use the testnet environment.
            logger_instance: The logger instance to use for logging.
        """
        self.logger = logger_instance
        self.is_testnet = is_testnet
        self._exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True, # Enable built-in rate limiter
            'options': {
                'defaultType': 'linear', # Default to linear contracts (USDT margined)
                'adjustForTimeDifference': True, # Auto-sync time with server
                'brokerId': 'NEONTA', # Optional: Set a broker ID
                'recvWindow': 10000, # Increase recvWindow slightly (default 5000ms)
                # Explicitly set testnet mode using standard ccxt option
                'testnet': self.is_testnet,
            },
            'timeout': CCXT_TIMEOUT_MS,
        }
        # Note: We are NOT overriding 'urls' here. CCXT should handle testnet/mainnet
        # based on the 'testnet' option. Overriding 'urls' caused the original error.

        try:
            self.exchange = ccxt_async.bybit(self._exchange_config)
            self.logger.info(f"CCXT client initialized for Bybit {'Testnet' if self.is_testnet else 'Mainnet'}.")
            self.logger.debug(f"Using Base URL: {self.exchange.urls['api']}") # Log the URL ccxt selected
        except Exception as e:
            self.logger.exception(f"Failed to initialize CCXT Bybit instance: {e}")
            raise  # Re-raise the exception to prevent starting with a broken client

        self.markets: Optional[Dict[str, Any]] = None
        self.market_categories: Dict[str, str] = {} # Cache for market categories (linear, inverse, spot)

    async def initialize_markets(self, retries: int = MAX_API_RETRIES) -> bool:
        """
        Loads market data from the exchange with retry logic.

        Args:
            retries: Maximum number of attempts to load markets.

        Returns:
            True if markets were loaded successfully, False otherwise.
        """
        self.logger.info("Attempting to load markets...")
        current_delay = INITIAL_RETRY_DELAY_SECONDS
        for attempt in range(retries):
            try:
                # load_markets(reload=True) might be needed if called multiple times
                self.markets = await self.exchange.load_markets()
                if not self.markets:
                    # Raise an error if load_markets returns None or empty dict
                    raise ccxt.ExchangeError("load_markets returned None or an empty dictionary.")

                self.logger.info(f"Successfully loaded {len(self.markets)} markets from {self.exchange.name}.")
                self._cache_market_categories() # Populate category cache
                return True
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, RequestException) as e:
                self.logger.warning(f"Network/Timeout error loading markets (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s...")
            except ccxt.AuthenticationError as e:
                self.logger.error(Color.format(f"Authentication Error loading markets: {e}. Please check API credentials.", Color.RED))
                return False # Authentication errors are fatal, no point retrying
            except ccxt.ExchangeError as e:
                 # Catch other specific ccxt errors if needed
                 self.logger.error(f"CCXT ExchangeError loading markets (Attempt {attempt + 1}/{retries}): {e}")
            except Exception as e:
                # Catch any other unexpected errors during market loading
                self.logger.exception(f"Unexpected error loading markets (Attempt {attempt + 1}/{retries}): {e}")
                # Log the original traceback that caused the issue in the user's code
                if isinstance(e, TypeError) and "string indices must be integers" in str(e):
                     self.logger.error("Caught the original TypeError related to URL structure. Ensure CCXT initialization is correct.")

            if attempt < retries - 1:
                await async_sleep_with_jitter(current_delay)
                current_delay = min(current_delay * 1.5, MAX_RETRY_DELAY_SECONDS) # Exponential backoff
            else:
                self.logger.error(Color.format(f"Failed to load markets after {retries} attempts.", Color.RED))

        return False

    def _cache_market_categories(self) -> None:
        """Pre-calculates and caches market categories (linear, inverse, spot) after loading markets."""
        if not self.markets:
            self.logger.warning("Cannot cache market categories: Markets not loaded.")
            return

        self.market_categories = {}
        count = 0
        for symbol, details in self.markets.items():
            # Determine category based on CCXT market properties
            category = 'spot' # Default assumption
            market_type = details.get('type', 'spot') # spot, linear, inverse

            if market_type == 'linear': category = 'linear'
            elif market_type == 'inverse': category = 'inverse'
            elif market_type == 'spot': category = 'spot'
            else:
                # Fallback guess based on quote currency if type is missing/unexpected
                quote = details.get('quote')
                if quote == 'USDT': category = 'linear'
                elif quote == 'USD': category = 'inverse'
                elif quote: category = 'spot' # Assume spot for other quotes
                self.logger.debug(f"Guessed category '{category}' for symbol {symbol} based on quote currency '{quote}'. Market type was '{market_type}'.")

            self.market_categories[symbol] = category
            count += 1
        self.logger.info(f"Cached categories for {count} markets.")


    def is_valid_symbol(self, symbol: str) -> bool:
        """Checks if the symbol exists in the loaded markets."""
        if self.markets is None:
            self.logger.warning("Markets not loaded, cannot validate symbol.")
            # Return False to be strict, prevents operations on potentially invalid symbols
            return False
        return symbol in self.markets

    def get_symbol_details(self, symbol: str) -> Optional[dict]:
        """
        Gets market details for a specific symbol from the loaded markets.

        Args:
            symbol: The market symbol (e.g., 'BTC/USDT').

        Returns:
            A dictionary containing market details, or None if the symbol is invalid
            or markets are not loaded.
        """
        if not self.is_valid_symbol(symbol):
            self.logger.warning(f"Attempted to get details for invalid or unloaded symbol: {symbol}")
            return None
        # self.markets is confirmed not None by is_valid_symbol if it returns True
        return self.markets.get(symbol) # type: ignore

    def get_market_category(self, symbol: str) -> str:
        """
        Gets the market category ('linear', 'inverse', 'spot') for a symbol,
        using the cache if available.

        Args:
            symbol: The market symbol.

        Returns:
            The category string ('linear', 'inverse', 'spot'). Defaults to 'spot'
            if unable to determine.
        """
        if symbol in self.market_categories:
            return self.market_categories[symbol]

        # Fallback if called before cache is populated or for an unknown symbol
        self.logger.warning(f"Category for symbol {symbol} not found in cache. Attempting dynamic check.")
        details = self.get_symbol_details(symbol) # Checks validity again
        if details:
            market_type = details.get('type', 'spot')
            if market_type == 'linear': return 'linear'
            if market_type == 'inverse': return 'inverse'
            if market_type == 'spot': return 'spot'
            # Fallback guess based on quote
            quote = details.get('quote')
            if quote == 'USDT': return 'linear'
            if quote == 'USD': return 'inverse'
            if quote: return 'spot'

        # Final fallback guess if details are missing or uninformative
        self.logger.warning(f"Could not reliably determine category for {symbol}, defaulting to 'spot'.")
        return 'spot'

    async def close(self) -> None:
        """Closes the underlying ccxt exchange connection gracefully."""
        if self.exchange:
            try:
                await self.exchange.close()
                self.logger.info("Closed CCXT exchange connection.")
            except Exception as e:
                self.logger.error(f"Error closing CCXT connection: {e}")

    async def fetch_with_retry(self, method_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
        """
        Generic fetch method with retry logic for common transient API errors.

        Args:
            method_name: The name of the ccxt exchange method to call (e.g., 'fetch_ticker').
            *args: Positional arguments for the ccxt method.
            **kwargs: Keyword arguments for the ccxt method.

        Returns:
            The result from the ccxt method call, or None if it fails after retries.
        """
        retries = MAX_API_RETRIES
        current_delay = INITIAL_RETRY_DELAY_SECONDS
        last_exception: Optional[Exception] = None

        for attempt in range(retries):
            try:
                method = getattr(self.exchange, method_name)
                result = await method(*args, **kwargs)
                # Optional: Add basic validation here if needed (e.g., check if result is None)
                return result
            # Specific, common transient errors first
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, RequestException) as e:
                log_msg = f"Network/Timeout/DDoS error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                self.logger.warning(Color.format(log_msg, Color.YELLOW))
                last_exception = e
            except ccxt.RateLimitExceeded as e:
                log_msg = f"Rate limit exceeded calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                self.logger.warning(Color.format(log_msg, Color.YELLOW))
                last_exception = e
            # Broader ExchangeError, check for potentially retryable Bybit messages
            except ccxt.ExchangeError as e:
                error_str = str(e).lower()
                # Bybit specific error codes/messages that might indicate temporary issues
                # Ref: https://bybit-exchange.github.io/docs/v5/error_code
                retryable_bybit_errors = [
                    'too many visits', 'system busy', 'service unavailable', 'ip rate limit',
                    '10006', # recv_window error (sometimes transient)
                    '10010', # request timeout
                    '10016', # service unavailable
                    '10018', # system busy
                    '10002', # request expired
                ]
                if any(err_code in error_str for err_code in retryable_bybit_errors):
                    log_msg = f"Retryable server/rate limit error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}. Retrying in {current_delay:.1f}s..."
                    self.logger.warning(Color.format(log_msg, Color.YELLOW))
                    last_exception = e
                else:
                    # Non-retryable ExchangeError (e.g., invalid symbol, insufficient funds)
                    self.logger.error(Color.format(f"Non-retryable CCXT ExchangeError calling {method_name}: {e}", Color.RED))
                    self.logger.debug(f"Args: {args}, Kwargs: {kwargs}")
                    return None # Do not retry these errors
            # Catch AuthenticationError separately as it's fatal
            except ccxt.AuthenticationError as e:
                 self.logger.error(Color.format(f"Authentication Error calling {method_name}: {e}. Check API credentials.", Color.RED))
                 return None # Fatal error
            # Catch all other unexpected exceptions
            except Exception as e:
                self.logger.exception(Color.format(f"Unexpected error calling {method_name} (Attempt {attempt + 1}/{retries}): {e}", Color.RED))
                self.logger.debug(f"Args: {args}, Kwargs: {kwargs}")
                last_exception = e
                # Decide whether to retry unexpected errors or not. Retrying is safer for potentially transient issues.

            # If an exception occurred and we haven't returned yet, wait and retry
            if attempt < retries - 1:
                await async_sleep_with_jitter(current_delay)
                current_delay = min(current_delay * 1.5, MAX_RETRY_DELAY_SECONDS) # Exponential backoff
            else:
                self.logger.error(Color.format(f"Max retries ({retries}) reached for {method_name}. Last error: {last_exception}", Color.RED))
                return None # Max retries exceeded

        return None # Should technically not be reached, but ensures a return path

    async def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Fetches ticker information for a symbol using the retry mechanism."""
        self.logger.debug(f"Fetching ticker for {symbol}")
        if not self.is_valid_symbol(symbol):
            self.logger.error(f"Cannot fetch ticker: Invalid symbol '{symbol}'.")
            return None

        category = self.get_market_category(symbol)
        params = {'category': category}
        # Use fetch_tickers, often more reliable even for single symbols
        # Note: fetch_tickers requires a list/tuple of symbols
        tickers = await self.fetch_with_retry('fetch_tickers', symbols=[symbol], params=params)

        if tickers and isinstance(tickers, dict) and symbol in tickers:
            return tickers[symbol]
        else:
            self.logger.error(f"Could not fetch ticker for {symbol} (category: {category}). Response: {tickers}")
            return None

    async def fetch_current_price(self, symbol: str) -> Optional[Decimal]:
        """Fetches the last traded price for a symbol and returns it as a Decimal."""
        ticker = await self.fetch_ticker(symbol)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            try:
                # Convert to string first for accurate Decimal conversion
                price_str = str(ticker['last'])
                return Decimal(price_str)
            except (InvalidOperation, TypeError, ValueError) as e:
                self.logger.error(f"Error converting last price '{ticker['last']}' to Decimal for {symbol}: {e}")
                return None
        else:
            self.logger.warning(f"Last price not found or is null in ticker data for {symbol}.")
            return None

    async def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetches OHLCV (kline) data for a symbol and timeframe, returning a processed DataFrame.

        Args:
            symbol: The market symbol.
            timeframe: The user-friendly timeframe string (e.g., "15", "1h").
            limit: The maximum number of klines to fetch.

        Returns:
            A pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            where 'timestamp' is timezone-aware (APP_TIMEZONE) and OHLCV are Decimals.
            Returns an empty DataFrame on failure or if no data is returned.
        """
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            self.logger.error(f"Invalid timeframe '{timeframe}' provided. Valid options: {VALID_INTERVALS}")
            return pd.DataFrame()
        if not self.is_valid_symbol(symbol):
             self.logger.error(f"Cannot fetch klines: Invalid symbol '{symbol}'.")
             return pd.DataFrame()

        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching {limit} klines for {symbol} | Interval: {ccxt_timeframe} | Category: {category}")
        params = {'category': category}
        # fetch_ohlcv(symbol, timeframe, since, limit, params)
        klines = await self.fetch_with_retry('fetch_ohlcv', symbol, timeframe=ccxt_timeframe, limit=limit, params=params)

        if klines is None or not isinstance(klines, list) or len(klines) == 0:
            self.logger.warning(f"No kline data returned for {symbol} interval {ccxt_timeframe}.")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                self.logger.warning(f"Kline data for {symbol} resulted in an empty DataFrame initially.")
                return pd.DataFrame()

            # Convert timestamp to datetime and set timezone
            # Errors='coerce' will turn unparseable timestamps into NaT
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            # Drop rows where timestamp conversion failed
            df.dropna(subset=['timestamp'], inplace=True)
            if not df.empty:
                 df['timestamp'] = df['timestamp'].dt.tz_convert(APP_TIMEZONE)
            else:
                 self.logger.warning(f"Kline data for {symbol} had no valid timestamps after conversion.")
                 return pd.DataFrame()


            # Convert OHLCV columns to Decimal for precision, coercing errors
            for col in ['open', 'high', 'low', 'close', 'volume']:
                # Convert to string first, then Decimal. Use pd.NA for unconvertible values.
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else pd.NA)
                # Coerce remaining errors (like pd.NA) to None or keep as is depending on downstream needs
                # For calculations, dropping rows with NA in essential columns is often best.

            initial_rows = len(df)
            # Drop rows with NA/NaN in essential OHLCV columns after conversion attempt
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            df.dropna(subset=essential_cols, inplace=True)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                self.logger.warning(f"Dropped {dropped_rows} rows with missing essential OHLCV data from klines for {symbol}.")

            if df.empty:
                self.logger.warning(f"Kline data for {symbol} is empty after cleaning missing values.")
                return pd.DataFrame()

            # Ensure data types are correct (Decimal for OHLCV, datetime for timestamp)
            # This might be redundant after the apply step but serves as a check
            for col in essential_cols:
                 if not all(isinstance(x, Decimal) for x in df[col]):
                      self.logger.warning(f"Column '{col}' contains non-Decimal values after processing.")
                      # Attempt one more conversion or handle appropriately
                      df[col] = pd.to_numeric(df[col], errors='coerce') # Fallback to numeric if Decimal failed broadly


            # Sort by timestamp just in case API returns them out of order
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            self.logger.debug(f"Successfully fetched and processed {len(df)} klines for {symbol}.")
            return df

        except (ValueError, TypeError, KeyError, InvalidOperation) as e:
            self.logger.exception(f"Error processing kline data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
             self.logger.exception(f"Unexpected error processing kline data for {symbol}: {e}")
             return pd.DataFrame()


    async def fetch_orderbook(self, symbol: str, limit: int) -> Optional[dict]:
        """
        Fetches the order book for a symbol using the retry mechanism.

        Args:
            symbol: The market symbol.
            limit: The maximum number of price levels to fetch for bids and asks.

        Returns:
            A dictionary representing the order book (with 'bids', 'asks' keys),
            or None if fetching fails or the response is invalid.
        """
        if not self.is_valid_symbol(symbol):
             self.logger.error(f"Cannot fetch orderbook: Invalid symbol '{symbol}'.")
             return None

        category = self.get_market_category(symbol)
        self.logger.debug(f"Fetching order book for {symbol} | Limit: {limit} | Category: {category}")
        params = {'category': category}
        # fetch_order_book(symbol, limit, params)
        orderbook = await self.fetch_with_retry('fetch_order_book', symbol, limit=limit, params=params)

        if orderbook and isinstance(orderbook, dict):
            # Basic validation: Check for presence and list type of bids/asks
            if ('bids' in orderbook and isinstance(orderbook['bids'], list) and
                    'asks' in orderbook and isinstance(orderbook['asks'], list)):
                # Optional: Deeper validation - check if bids/asks contain [price, size] pairs
                # Example: all(isinstance(level, list) and len(level) == 2 for level in orderbook['bids'])
                self.logger.debug(f"Fetched order book for {symbol} with {len(orderbook['bids'])} bids and {len(orderbook['asks'])} asks.")
                return orderbook
            else:
                self.logger.warning(f"Fetched order book data for {symbol} is missing bids/asks or has an unexpected format: {orderbook}")
                return None
        else:
            self.logger.warning(f"Failed to fetch order book for {symbol} or received invalid data after retries.")
            return None


# --- Trading Analyzer ---
class TradingAnalyzer:
    """
    Performs technical analysis using kline data and interprets the results.
    Optionally incorporates order book analysis.
    """
    def __init__(self, config: AppConfig, logger_instance: logging.Logger, symbol: str):
        """
        Initializes the TradingAnalyzer.

        Args:
            config: The application configuration object.
            logger_instance: The logger instance to use.
            symbol: The trading symbol being analyzed.
        """
        self.config = config
        self.logger = logger_instance
        self.symbol = symbol
        # Convenience accessors for config sections
        self.indicator_settings = config.indicator_settings
        self.analysis_flags = config.analysis_flags
        self.thresholds = config.thresholds
        self.orderbook_settings = config.orderbook_settings

        # Dynamically generate expected indicator column names based on config
        self._generate_column_names()

    def _generate_column_names(self) -> None:
        """Generates expected pandas_ta indicator column names based on config settings."""
        self.col_names: Dict[str, str] = {}
        is_ = self.indicator_settings # Alias for brevity

        # Helper to format Bollinger Bands std dev for column name consistency
        def fmt_bb_std(std: Union[float, int, Decimal]) -> str:
            try:
                # Format to one decimal place, consistent with pandas_ta default naming
                return f"{Decimal(str(std)):.1f}"
            except (InvalidOperation, TypeError):
                return str(std) # Fallback

        # Standard Indicators
        self.col_names['sma_short'] = f"SMA_{is_.sma_short_period}"
        self.col_names['sma_long'] = f"SMA_{is_.sma_long_period}"
        self.col_names['ema_short'] = f"EMA_{is_.ema_short_period}"
        self.col_names['ema_long'] = f"EMA_{is_.ema_long_period}"
        self.col_names['rsi'] = f"RSI_{is_.rsi_period}"
        self.col_names['stochrsi_k'] = f"STOCHRSIk_{is_.stoch_rsi_period}_{is_.rsi_period}_{is_.stoch_k_period}_{is_.stoch_d_period}"
        self.col_names['stochrsi_d'] = f"STOCHRSId_{is_.stoch_rsi_period}_{is_.rsi_period}_{is_.stoch_k_period}_{is_.stoch_d_period}"
        self.col_names['macd_line'] = f"MACD_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
        self.col_names['macd_signal'] = f"MACDs_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
        self.col_names['macd_hist'] = f"MACDh_{is_.macd_fast}_{is_.macd_slow}_{is_.macd_signal}"
        self.col_names['bb_upper'] = f"BBU_{is_.bollinger_bands_period}_{fmt_bb_std(is_.bollinger_bands_std_dev)}"
        self.col_names['bb_lower'] = f"BBL_{is_.bollinger_bands_period}_{fmt_bb_std(is_.bollinger_bands_std_dev)}"
        self.col_names['bb_mid'] = f"BBM_{is_.bollinger_bands_period}_{fmt_bb_std(is_.bollinger_bands_std_dev)}"
        self.col_names['atr'] = f"ATRr_{is_.atr_period}" # pandas_ta uses ATRr for the 'True Range average' variant
        # pandas_ta CCI includes the constant in the name by default (0.015)
        self.col_names['cci'] = f"CCI_{is_.cci_period}_0.015"
        self.col_names['willr'] = f"WILLR_{is_.williams_r_period}"
        self.col_names['mfi'] = f"MFI_{is_.mfi_period}"
        self.col_names['adx'] = f"ADX_{is_.adx_period}"
        self.col_names['dmp'] = f"DMP_{is_.adx_period}" # +DI component of ADX
        self.col_names['dmn'] = f"DMN_{is_.adx_period}" # -DI component of ADX
        self.col_names['obv'] = "OBV"
        self.col_names['adosc'] = "ADOSC" # Accumulation/Distribution Oscillator (preferred over raw AD line)
        # PSAR columns generated by pandas_ta
        psar_step_str = str(is_.psar_step)
        psar_max_str = str(is_.psar_max_step)
        self.col_names['psar_long'] = f"PSARl_{psar_step_str}_{psar_max_str}"
        self.col_names['psar_short'] = f"PSARs_{psar_step_str}_{psar_max_str}"
        self.col_names['psar_af'] = f"PSARaf_{psar_step_str}_{psar_max_str}" # Acceleration Factor
        self.col_names['psar_rev'] = f"PSARr_{psar_step_str}_{psar_max_str}" # Reversal signal (1 for reversal)

        # Custom calculated indicators (if not directly covered by pandas_ta strategy)
        self.col_names['mom'] = f"MOM_{is_.momentum_period}"
        # Volume MA needs special handling in strategy definition
        self.col_names['vol_ma'] = f"VOL_MA_{is_.volume_ma_period}"

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators using pandas_ta strategy.

        Args:
            df: Input DataFrame with OHLCV data (expects Decimal types).

        Returns:
            DataFrame with calculated indicator columns appended, or the original
            DataFrame if calculation fails or input is unsuitable.
        """
        if df.empty:
            self.logger.warning("Cannot calculate indicators: Input DataFrame is empty.")
            return df
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
             self.logger.error("Cannot calculate indicators: Input DataFrame missing required OHLCV columns.")
             return df

        # Ensure OHLCV columns are numeric (float or Decimal) for pandas_ta
        # pandas_ta generally works better with floats. Convert Decimal to float for calculation.
        # Keep a copy of the original df if needed elsewhere with Decimals.
        df_calc = df.copy()
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            # Coerce to numeric (float64), turning errors into NaN
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

        # Drop rows if essential numeric columns became NaN after conversion
        initial_rows = len(df_calc)
        df_calc.dropna(subset=numeric_cols, inplace=True)
        if len(df_calc) < initial_rows:
             self.logger.warning(f"Dropped {initial_rows - len(df_calc)} rows due to non-numeric OHLCV data during indicator calculation prep.")

        if df_calc.empty:
             self.logger.error("DataFrame became empty after converting OHLCV to numeric. Cannot calculate indicators.")
             return df # Return original unmodified df

        # Check if enough data points exist for the longest required period
        is_ = self.indicator_settings
        # Estimate max period needed (consider MACD slow+signal, ADX needs more data)
        min_data_needed = max(
            is_.sma_long_period, is_.ema_long_period,
            is_.macd_slow + is_.macd_signal, # MACD needs signal periods beyond slow EMA
            is_.bollinger_bands_period,
            is_.adx_period * 2, # ADX often needs 2x period for smoothing
            is_.stoch_rsi_period + is_.rsi_period, # StochRSI needs underlying RSI data
            is_.volume_ma_period, is_.atr_period, is_.cci_period, is_.williams_r_period, is_.mfi_period,
            is_.momentum_period
        )
        if len(df_calc) < min_data_needed:
            self.logger.warning(f"Insufficient data points ({len(df_calc)} < {min_data_needed} estimated needed) for some indicators. Results may be inaccurate or contain NaNs.")
        elif len(df_calc) < 2:
             self.logger.warning("Only one data point available. Most indicators cannot be calculated.")
             # Return original df as calculations will fail or be meaningless
             return df


        # Define the strategy using pandas_ta structure
        # Ensure all periods are positive integers/floats
        strategy_ta = [
            {"kind": "sma", "length": is_.sma_short_period} if is_.sma_short_period > 0 else None,
            {"kind": "sma", "length": is_.sma_long_period} if is_.sma_long_period > 0 else None,
            {"kind": "ema", "length": is_.ema_short_period} if is_.ema_short_period > 0 else None,
            {"kind": "ema", "length": is_.ema_long_period} if is_.ema_long_period > 0 else None,
            {"kind": "rsi", "length": is_.rsi_period} if is_.rsi_period > 0 else None,
            {"kind": "stochrsi", "length": is_.stoch_rsi_period, "rsi_length": is_.rsi_period, "k": is_.stoch_k_period, "d": is_.stoch_d_period} if all(p > 0 for p in [is_.stoch_rsi_period, is_.rsi_period, is_.stoch_k_period, is_.stoch_d_period]) else None,
            {"kind": "macd", "fast": is_.macd_fast, "slow": is_.macd_slow, "signal": is_.macd_signal} if all(p > 0 for p in [is_.macd_fast, is_.macd_slow, is_.macd_signal]) else None,
            {"kind": "bbands", "length": is_.bollinger_bands_period, "std": float(is_.bollinger_bands_std_dev)} if is_.bollinger_bands_period > 0 and is_.bollinger_bands_std_dev > 0 else None,
            {"kind": "atr", "length": is_.atr_period} if is_.atr_period > 0 else None,
            {"kind": "cci", "length": is_.cci_period} if is_.cci_period > 0 else None,
            {"kind": "willr", "length": is_.williams_r_period} if is_.williams_r_period > 0 else None,
            {"kind": "mfi", "length": is_.mfi_period} if is_.mfi_period > 0 else None,
            {"kind": "adx", "length": is_.adx_period} if is_.adx_period > 0 else None,
            {"kind": "obv"}, # OBV doesn't have a length parameter in the same way
            {"kind": "adosc"}, # AD Oscillator
            {"kind": "psar", "step": is_.psar_step, "max_step": is_.psar_max_step} if is_.psar_step > 0 and is_.psar_max_step > 0 else None,
            {"kind": "mom", "length": is_.momentum_period} if is_.momentum_period > 0 else None,
            # Calculate volume MA using 'sma' kind on 'volume' column
            {"kind": "sma", "close": "volume", "length": is_.volume_ma_period, "prefix": "VOL_MA"} if is_.volume_ma_period > 0 else None,
        ]

        # Filter out None entries (from invalid periods)
        valid_strategy_ta = [item for item in strategy_ta if item is not None]

        if not valid_strategy_ta:
             self.logger.error("No valid indicators configured or all periods are invalid. Skipping calculation.")
             return df # Return original df

        # Create the pandas_ta strategy object
        strategy = ta.Strategy(
            name="NeontaAnalysis",
            description="Comprehensive TA using pandas_ta",
            ta=valid_strategy_ta
        )

        try:
            # Apply the strategy to the DataFrame (modifies df_calc in place)
            df_calc.ta.strategy(strategy, timed=False) # timed=True adds performance logging

            # Rename the volume MA column generated by pandas_ta to our expected name
            # Default name generated by prefix is like "VOL_MA_SMA_20"
            vol_ma_generated_name = f"VOL_MA_SMA_{is_.volume_ma_period}"
            if vol_ma_generated_name in df_calc.columns and self.col_names['vol_ma'] not in df_calc.columns:
                df_calc.rename(columns={vol_ma_generated_name: self.col_names['vol_ma']}, inplace=True)
            elif vol_ma_generated_name not in df_calc.columns and is_.volume_ma_period > 0:
                 self.logger.warning(f"Expected volume MA column '{vol_ma_generated_name}' not found after calculation.")


            self.logger.debug(f"Calculated pandas_ta indicators for {self.symbol}.")

            # Optional: Fill potentially generated PSAR long/short columns if one is all NaN
            # This handles cases where PSAR might only generate one column initially
            psar_l_col = self.col_names.get('psar_long')
            psar_s_col = self.col_names.get('psar_short')
            if psar_l_col and psar_s_col and psar_l_col in df_calc.columns and psar_s_col in df_calc.columns:
                if df_calc[psar_l_col].isnull().all() and not df_calc[psar_s_col].isnull().all():
                    df_calc[psar_l_col] = df_calc[psar_s_col]
                    self.logger.debug(f"Filled NaN PSAR long column from short column for {self.symbol}")
                elif df_calc[psar_s_col].isnull().all() and not df_calc[psar_l_col].isnull().all():
                    df_calc[psar_s_col] = df_calc[psar_l_col]
                    self.logger.debug(f"Filled NaN PSAR short column from long column for {self.symbol}")

            # Merge indicator columns (now floats) back into the original DataFrame (with Decimals)
            # This preserves the Decimal precision of OHLCV data.
            indicator_cols = [col for col in df_calc.columns if col not in df.columns]
            result_df = pd.concat([df, df_calc[indicator_cols]], axis=1)

            # Convert indicator columns back to Decimal if desired (adds overhead)
            # Keeping them as floats is generally fine for interpretation.
            # for col in indicator_cols:
            #     if col in result_df.columns:
            #          result_df[col] = result_df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)

            return result_df

        except Exception as e:
            self.logger.exception(f"Error calculating indicators using pandas_ta strategy for {self.symbol}: {e}")
            # Return the original DataFrame without indicators on failure
            return df

    def _calculate_levels(self, df: pd.DataFrame, current_price: Decimal) -> dict:
        """
        Calculates potential support, resistance, and pivot levels based on historical data.

        Args:
            df: DataFrame containing OHLCV data (expects Decimal types).
            current_price: The current market price as a Decimal.

        Returns:
            A dictionary containing 'support', 'resistance', and 'pivot' levels.
            Levels are stored as {level_name: Decimal(price)}.
        """
        levels = {"support": {}, "resistance": {}, "pivot": None}
        if df.empty or len(df) < 2:
            self.logger.warning("Insufficient data for level calculation (need at least 2 rows).")
            return levels
        if not isinstance(current_price, Decimal):
             self.logger.error("Invalid current_price type for level calculation. Expected Decimal.")
             return levels

        try:
            # Use the full period available in the DataFrame for more robust levels
            # Ensure values are Decimals before calculation
            high = df["high"].max()
            low = df["low"].min()
            # Use the most recent close for pivot calculation
            close = df["close"].iloc[-1]

            if not all(isinstance(v, Decimal) for v in [high, low, close]) or pd.isna(high) or pd.isna(low) or pd.isna(close):
                self.logger.warning("NaN or non-Decimal values found in OHLC data. Cannot calculate levels accurately.")
                return levels

            # --- Fibonacci Retracement Levels ---
            diff = high - low
            # Check for zero or negligible difference to avoid errors/meaningless levels
            if diff > Decimal("1e-12"): # Use a small threshold for Decimal comparison
                # Standard Fibonacci levels
                fib_ratios = [Decimal("0.236"), Decimal("0.382"), Decimal("0.5"), Decimal("0.618"), Decimal("0.786")]
                fib_levels = {}
                # Calculate levels relative to the high/low range
                for ratio in fib_ratios:
                    # Level based on retracement from high
                    level_down = high - diff * ratio
                    fib_levels[f"Fib {ratio*100:.1f}% (Retrace Down)"] = level_down
                    # Level based on retracement from low (extension concept)
                    # level_up = low + diff * ratio # Less common for basic S/R
                    # fib_levels[f"Fib {ratio*100:.1f}% (Retrace Up)"] = level_up

                # Add High and Low as natural S/R
                fib_levels["Period High"] = high
                fib_levels["Period Low"] = low

                # Classify Fibonacci levels as support or resistance based on current price
                for label, value in fib_levels.items():
                    if value < current_price:
                        levels["support"][label] = value
                    elif value > current_price:
                        levels["resistance"][label] = value
                    # else: Level is exactly the current price, could be either? Ignore for now.
            else:
                 self.logger.debug("Price range (High - Low) is too small for Fibonacci calculation.")


            # --- Pivot Points (Classical Method) ---
            try:
                pivot = (high + low + close) / Decimal(3)
                levels["pivot"] = pivot

                # Calculate classical support and resistance levels based on the pivot
                r1 = (Decimal(2) * pivot) - low
                s1 = (Decimal(2) * pivot) - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)
                r3 = high + Decimal(2) * (pivot - low)
                s3 = low - Decimal(2) * (high - pivot)

                pivot_levels = {"R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}

                # Classify pivot levels as support or resistance
                for label, value in pivot_levels.items():
                    if value < current_price:
                        levels["support"][label] = value
                    elif value > current_price:
                        levels["resistance"][label] = value

            except (InvalidOperation, ArithmeticError) as e:
                self.logger.error(f"Error during pivot point calculation: {e}")

        except (TypeError, ValueError, InvalidOperation, IndexError, KeyError) as e:
            self.logger.error(f"Error calculating levels for {self.symbol}: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error calculating levels for {self.symbol}: {e}")

        # Sort levels by price for cleaner output later (optional)
        levels["support"] = dict(sorted(levels["support"].items(), key=lambda item: item[1], reverse=True)) # Highest support first
        levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda item: item[1])) # Lowest resistance first

        return levels

    def _analyze_orderbook(self, orderbook: Optional[dict], current_price: Decimal, levels: dict) -> dict:
        """
        Analyzes order book data for buy/sell pressure and identifies significant
        clusters of orders near calculated support/resistance levels.

        Args:
            orderbook: The fetched order book dictionary (containing 'bids' and 'asks').
            current_price: The current market price as a Decimal.
            levels: Dictionary containing calculated 'support' and 'resistance' levels.

        Returns:
            A dictionary containing analysis results:
            - 'pressure': SignalState indicating overall buy/sell pressure.
            - 'total_bid_usd': Total USD value of bids in the fetched book.
            - 'total_ask_usd': Total USD value of asks in the fetched book.
            - 'clusters': A list of dictionaries, each describing a significant order cluster found.
        """
        analysis: Dict[str, Any] = {
            "clusters": [],
            "pressure": SignalState.NA.value, # Default to N/A
            "total_bid_usd": Decimal(0),
            "total_ask_usd": Decimal(0)
        }
        if not orderbook or not isinstance(orderbook.get('bids'), list) or not isinstance(orderbook.get('asks'), list):
            self.logger.debug("Orderbook data incomplete or unavailable for analysis.")
            return analysis
        if not isinstance(current_price, Decimal):
             self.logger.error("Invalid current_price type for orderbook analysis. Expected Decimal.")
             return analysis

        try:
            # Convert bids and asks to DataFrames for easier processing
            # Ensure data is converted to Decimal, handle errors gracefully
            def to_decimal_df(data: List[List[Union[str, float, int]]], columns: List[str]) -> pd.DataFrame:
                if not data: return pd.DataFrame(columns=columns)
                try:
                    df = pd.DataFrame(data, columns=columns)
                    for col in columns:
                        # Convert to string first, then Decimal. Coerce errors to NaN/NaT.
                        df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else pd.NA)
                        # Convert pd.NA to None or handle as needed. Here, drop rows with NA price/size.
                    df.dropna(subset=columns, inplace=True)
                    return df
                except (ValueError, TypeError, InvalidOperation) as e:
                     self.logger.error(f"Error converting orderbook data to Decimal DataFrame: {e}")
                     return pd.DataFrame(columns=columns) # Return empty df on error

            bids_df = to_decimal_df(orderbook.get('bids', []), ['price', 'size'])
            asks_df = to_decimal_df(orderbook.get('asks', []), ['price', 'size'])

            # Filter out zero prices/sizes that might indicate data issues
            bids_df = bids_df[(bids_df['price'] > Decimal(0)) & (bids_df['size'] > Decimal(0))]
            asks_df = asks_df[(asks_df['price'] > Decimal(0)) & (asks_df['size'] > Decimal(0))]

            if bids_df.empty and asks_df.empty:
                self.logger.debug("Orderbook is empty after cleaning zero/invalid values.")
                analysis["pressure"] = SignalState.NEUTRAL.value # Indicate neutral if empty
                return analysis

            # Calculate USD value for each level and total pressure
            if not bids_df.empty:
                bids_df['value_usd'] = bids_df['price'] * bids_df['size']
                analysis["total_bid_usd"] = bids_df['value_usd'].sum()
            if not asks_df.empty:
                asks_df['value_usd'] = asks_df['price'] * asks_df['size']
                analysis["total_ask_usd"] = asks_df['value_usd'].sum()

            total_value = analysis["total_bid_usd"] + analysis["total_ask_usd"]
            if total_value > Decimal(0):
                # Simple pressure calculation based on total value ratio
                bid_ask_ratio = analysis["total_bid_usd"] / total_value
                # Define thresholds for high pressure (e.g., > 60% or < 40%)
                high_pressure_threshold = Decimal("0.6")
                low_pressure_threshold = Decimal("0.4")
                if bid_ask_ratio > high_pressure_threshold:
                    analysis["pressure"] = Color.format("High Buy Pressure", Color.GREEN)
                elif bid_ask_ratio < low_pressure_threshold:
                    analysis["pressure"] = Color.format("High Sell Pressure", Color.RED)
                else:
                    analysis["pressure"] = Color.format("Neutral Pressure", Color.YELLOW)
            else:
                 analysis["pressure"] = SignalState.NEUTRAL.value # Neutral if total value is zero

            # --- Cluster Analysis ---
            cluster_threshold_usd = Decimal(str(self.orderbook_settings.cluster_threshold_usd))
            # Convert proximity percentage to a Decimal factor
            proximity_factor = Decimal(str(self.orderbook_settings.cluster_proximity_pct)) / Decimal(100)

            # Combine all calculated support, resistance, and pivot levels for checking
            all_levels_to_check: Dict[str, Decimal] = {}
            all_levels_to_check.update(levels.get("support", {}))
            all_levels_to_check.update(levels.get("resistance", {}))
            if levels.get("pivot") is not None and isinstance(levels["pivot"], Decimal):
                all_levels_to_check["Pivot"] = levels["pivot"]

            processed_clusters = set() # Track processed levels to avoid duplicates if S/R overlap

            for name, level_price in all_levels_to_check.items():
                if not isinstance(level_price, Decimal) or level_price <= 0: continue

                # Define the price range around the level to check for clusters
                price_delta = level_price * proximity_factor
                min_price = level_price - price_delta
                max_price = level_price + price_delta

                # Check for significant bid clusters (potential support confirmation) near the level
                if not bids_df.empty:
                    bids_near_level = bids_df[(bids_df['price'] >= min_price) & (bids_df['price'] <= max_price)]
                    bid_cluster_value_usd = bids_near_level['value_usd'].sum()

                    if bid_cluster_value_usd >= cluster_threshold_usd:
                        # Use level price for uniqueness check
                        cluster_id = f"BID_{level_price:.8f}" # Use sufficient precision for ID
                        if cluster_id not in processed_clusters:
                            analysis["clusters"].append({
                                "type": "Support", # Bid cluster implies potential support
                                "level_name": name,
                                "level_price": level_price,
                                "cluster_value_usd": bid_cluster_value_usd,
                                "price_range": (min_price, max_price) # Store the range checked
                            })
                            processed_clusters.add(cluster_id)

                # Check for significant ask clusters (potential resistance confirmation) near the level
                if not asks_df.empty:
                    asks_near_level = asks_df[(asks_df['price'] >= min_price) & (asks_df['price'] <= max_price)]
                    ask_cluster_value_usd = asks_near_level['value_usd'].sum()

                    if ask_cluster_value_usd >= cluster_threshold_usd:
                        cluster_id = f"ASK_{level_price:.8f}"
                        if cluster_id not in processed_clusters:
                            analysis["clusters"].append({
                                "type": "Resistance", # Ask cluster implies potential resistance
                                "level_name": name,
                                "level_price": level_price,
                                "cluster_value_usd": ask_cluster_value_usd,
                                "price_range": (min_price, max_price)
                            })
                            processed_clusters.add(cluster_id)

        except (KeyError, ValueError, TypeError, InvalidOperation, AttributeError) as e:
            self.logger.error(f"Error analyzing orderbook for {self.symbol}: {e}")
            self.logger.debug(traceback.format_exc()) # Log stack trace for debugging OB issues
        except Exception as e:
            # Catch any unexpected errors during order book analysis
            self.logger.exception(f"Unexpected error analyzing orderbook for {self.symbol}: {e}")

        # Sort identified clusters by their USD value (descending) for prominence
        analysis["clusters"] = sorted(analysis.get("clusters", []), key=lambda x: x['cluster_value_usd'], reverse=True)

        return analysis

    # --- Interpretation Helpers ---

    def _get_val(self, row: pd.Series, key: str, default: Any = None) -> Any:
        """
        Safely gets a value from a Pandas Series (representing a DataFrame row),
        handling missing keys and NaN/None values.

        Args:
            row: The pandas Series (row).
            key: The key (column name) to retrieve.
            default: The value to return if the key is missing or the value is NaN/None.

        Returns:
            The value from the Series, or the default value.
        """
        if key not in row:
            # Log only once per key maybe? Or at debug level.
            # self.logger.debug(f"Indicator key '{key}' not found in DataFrame row.")
            return default
        val = row[key]
        # Check for pandas NA, numpy NaN, or Python None
        return default if pd.isna(val) else val

    def _format_signal(self, label: str, value: Any, signal: SignalState, precision: int = 2, details: str = "") -> str:
        """
        Formats a single line of the analysis summary, applying color based on the signal state.

        Args:
            label: The label for the indicator/signal (e.g., "RSI", "MACD Cross").
            value: The numeric value of the indicator (or relevant info).
            signal: The SignalState enum member representing the interpretation.
            precision: The decimal precision for formatting the value.
            details: Additional context or information string to append.

        Returns:
            A formatted string ready for printing, including color codes.
        """
        # Format the numeric value using the utility function
        value_str = format_decimal(value, precision) if value is not None else ""

        # Map SignalState to Color
        color_map = {
            SignalState.BULLISH: Color.GREEN, SignalState.STRONG_BULLISH: Color.GREEN,
            SignalState.OVERSOLD: Color.GREEN, SignalState.INCREASING: Color.GREEN,
            SignalState.ACCUMULATION: Color.GREEN, SignalState.FLIP_BULLISH: Color.GREEN,
            SignalState.BREAKDOWN_LOWER: Color.GREEN, # Price breaking below BB lower (potential reversal/buy?) - Color debatable
            SignalState.ABOVE_SIGNAL: Color.GREEN,

            SignalState.BEARISH: Color.RED, SignalState.STRONG_BEARISH: Color.RED,
            SignalState.OVERBOUGHT: Color.RED, SignalState.DECREASING: Color.RED,
            SignalState.DISTRIBUTION: Color.RED, SignalState.FLIP_BEARISH: Color.RED,
            SignalState.BREAKOUT_UPPER: Color.RED, # Price breaking above BB upper (potential reversal/sell?) - Color debatable
            SignalState.BELOW_SIGNAL: Color.RED,

            SignalState.NEUTRAL: Color.YELLOW, SignalState.RANGING: Color.YELLOW,
            SignalState.WITHIN_BANDS: Color.YELLOW, SignalState.AVERAGE_VOLUME: Color.YELLOW,
            SignalState.FLAT: Color.YELLOW, SignalState.NONE: Color.YELLOW,

            SignalState.HIGH_VOLUME: Color.PURPLE, SignalState.LOW_VOLUME: Color.PURPLE,

            SignalState.NA: Color.YELLOW, # Default color for N/A or unmapped states
        }
        color = color_map.get(signal, Color.YELLOW) # Default to yellow if signal not in map

        # Get the string value from the SignalState enum
        signal_text = signal.value if isinstance(signal, SignalState) else str(signal)

        # Construct the output string
        label_part = f"{label}"
        value_part = f" ({value_str})" if value_str and value_str != "N/A" else ""
        signal_part = f": {Color.format(signal_text, color)}"
        details_part = f" ({details})" if details else ""

        return f"{label_part}{value_part}{signal_part}{details_part}"

    def _interpret_trend(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets trend indicators (EMAs, ADX, PSAR)."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names

        # --- EMA Alignment ---
        signal_key = "ema_trend"
        signals[signal_key] = SignalState.NA # Default
        if flags.ema_alignment:
            ema_short = self._get_val(last_row, cols.get('ema_short'))
            ema_long = self._get_val(last_row, cols.get('ema_long'))
            label = f"EMA Align ({self.indicator_settings.ema_short_period}/{self.indicator_settings.ema_long_period})"
            signal = SignalState.NA
            details = ""
            value = None # No single value for alignment

            if ema_short is not None and ema_long is not None:
                signal = SignalState.NEUTRAL
                if ema_short > ema_long: signal = SignalState.BULLISH
                elif ema_short < ema_long: signal = SignalState.BEARISH
                details = f"S:{format_decimal(ema_short)} {'><'[ema_short > ema_long]} L:{format_decimal(ema_long)}"
                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        # --- ADX Trend Strength ---
        signal_key = "adx_trend"
        signals[signal_key] = SignalState.NA # Default
        if flags.adx_trend_strength:
            adx = self._get_val(last_row, cols.get('adx'))
            dmp = self._get_val(last_row, cols.get('dmp'), default=Decimal(0)) # +DI
            dmn = self._get_val(last_row, cols.get('dmn'), default=Decimal(0)) # -DI
            label = f"ADX Trend ({self.indicator_settings.adx_period})"
            signal = SignalState.NA
            details = ""
            value = adx

            if adx is not None and dmp is not None and dmn is not None:
                # Ensure comparison is between Decimals
                trend_threshold = Decimal(str(self.thresholds.adx_trending))
                adx_decimal = Decimal(str(adx)) # Convert adx value to Decimal for comparison

                if adx_decimal >= trend_threshold:
                    if dmp > dmn:
                        signal = SignalState.STRONG_BULLISH
                        details = f"+DI ({format_decimal(dmp)}) > -DI ({format_decimal(dmn)})"
                    else:
                        signal = SignalState.STRONG_BEARISH
                        details = f"-DI ({format_decimal(dmn)}) > +DI ({format_decimal(dmp)})"
                else:
                    signal = SignalState.RANGING
                    details = f"ADX < {trend_threshold}"
                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        # --- PSAR Flip ---
        signal_key_trend = "psar_trend"
        signal_key_signal = "psar_signal"
        signals[signal_key_trend] = SignalState.NA # Default trend
        signals[signal_key_signal] = SignalState.NA # Default signal (flip or trend)
        if flags.psar_flip:
            psar_l = self._get_val(last_row, cols.get('psar_long'))
            psar_s = self._get_val(last_row, cols.get('psar_short'))
            close_val = self._get_val(last_row, 'close')
            # PSARr column indicates a reversal occurred on this candle (value is 1 if reversal)
            psar_rev_signal = self._get_val(last_row, cols.get('psar_rev'))

            label = "PSAR"
            signal = SignalState.NA
            trend_signal = SignalState.NA
            details = ""
            psar_display_val = None # The PSAR value to display

            if close_val is not None:
                # Determine current trend based on price vs PSAR levels
                # Prefer the non-NaN PSAR value for trend determination
                if psar_l is not None and psar_s is None: # Only long PSAR available
                    psar_display_val = psar_l
                    trend_signal = SignalState.BULLISH if close_val > psar_l else SignalState.BEARISH
                elif psar_s is not None and psar_l is None: # Only short PSAR available
                    psar_display_val = psar_s
                    trend_signal = SignalState.BEARISH if close_val < psar_s else SignalState.BULLISH
                elif psar_l is not None and psar_s is not None: # Both available (shouldn't happen with pandas_ta PSAR)
                     # Heuristic: If close > long PSAR, assume bullish trend, use long PSAR value
                     if close_val > psar_l:
                         psar_display_val = psar_l
                         trend_signal = SignalState.BULLISH
                     # If close < short PSAR, assume bearish trend, use short PSAR value
                     elif close_val < psar_s:
                          psar_display_val = psar_s
                          trend_signal = SignalState.BEARISH
                     else: # Price is between the two? This case is unlikely with standard PSAR. Default to neutral?
                          psar_display_val = (psar_l + psar_s) / 2 # Or pick one?
                          trend_signal = SignalState.NEUTRAL # Ambiguous state
                # else: Both psar_l and psar_s are None, cannot determine trend

            if trend_signal != SignalState.NA:
                signals[signal_key_trend] = trend_signal # Store the determined trend
                signal = trend_signal # Base signal is the current trend

                # Check if a reversal occurred on this candle using PSARr column
                if psar_rev_signal == 1:
                    # If reversal signal is 1, the trend just flipped
                    signal = SignalState.FLIP_BULLISH if trend_signal == SignalState.BULLISH else SignalState.FLIP_BEARISH
                    details = "Just Flipped!"
                signals[signal_key_signal] = signal # Store the final signal (flip or trend)

            summary_lines.append(self._format_signal(label, psar_display_val, signal, precision=4, details=details))

        return summary_lines, signals

    def _interpret_oscillators(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets oscillator indicators (RSI, MFI, CCI, Williams %R, StochRSI)."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        thresh = self.thresholds
        cols = self.col_names

        # --- RSI Level ---
        signal_key = "rsi_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.rsi_threshold:
            rsi = self._get_val(last_row, cols.get('rsi'))
            label = f"RSI ({self.indicator_settings.rsi_period})"
            signal = SignalState.NA
            if rsi is not None:
                # Ensure comparisons use Decimal
                rsi_val = Decimal(str(rsi))
                ob_thresh = Decimal(str(thresh.rsi_overbought))
                os_thresh = Decimal(str(thresh.rsi_oversold))
                signal = SignalState.NEUTRAL
                if rsi_val >= ob_thresh: signal = SignalState.OVERBOUGHT
                elif rsi_val <= os_thresh: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, rsi, signal))

        # --- MFI Level ---
        signal_key = "mfi_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.mfi_threshold:
            mfi = self._get_val(last_row, cols.get('mfi'))
            label = f"MFI ({self.indicator_settings.mfi_period})"
            signal = SignalState.NA
            if mfi is not None:
                mfi_val = Decimal(str(mfi))
                ob_thresh = Decimal(str(thresh.mfi_overbought))
                os_thresh = Decimal(str(thresh.mfi_oversold))
                signal = SignalState.NEUTRAL
                if mfi_val >= ob_thresh: signal = SignalState.OVERBOUGHT
                elif mfi_val <= os_thresh: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, mfi, signal))

        # --- CCI Level ---
        signal_key = "cci_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.cci_threshold:
            cci = self._get_val(last_row, cols.get('cci'))
            label = f"CCI ({self.indicator_settings.cci_period})"
            signal = SignalState.NA
            if cci is not None:
                cci_val = Decimal(str(cci))
                ob_thresh = Decimal(str(thresh.cci_overbought))
                os_thresh = Decimal(str(thresh.cci_oversold))
                signal = SignalState.NEUTRAL
                if cci_val >= ob_thresh: signal = SignalState.OVERBOUGHT
                elif cci_val <= os_thresh: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, cci, signal))

        # --- Williams %R Level ---
        signal_key = "wr_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.williams_r_threshold:
            wr = self._get_val(last_row, cols.get('willr'))
            label = f"Williams %R ({self.indicator_settings.williams_r_period})"
            signal = SignalState.NA
            if wr is not None:
                wr_val = Decimal(str(wr))
                # Note: Williams %R thresholds are negative
                ob_thresh = Decimal(str(thresh.williams_r_overbought)) # e.g., -20
                os_thresh = Decimal(str(thresh.williams_r_oversold)) # e.g., -80
                signal = SignalState.NEUTRAL
                # Overbought is when value is *above* the OB threshold (closer to 0)
                if wr_val >= ob_thresh: signal = SignalState.OVERBOUGHT
                # Oversold is when value is *below* the OS threshold (closer to -100)
                elif wr_val <= os_thresh: signal = SignalState.OVERSOLD
                signals[signal_key] = signal
            summary_lines.append(self._format_signal(label, wr, signal))

        # --- StochRSI Cross ---
        signal_key = "stochrsi_cross"
        signals[signal_key] = SignalState.NA # Default
        if flags.stoch_rsi_cross:
            k_now = self._get_val(last_row, cols.get('stochrsi_k'))
            d_now = self._get_val(last_row, cols.get('stochrsi_d'))
            k_prev = self._get_val(prev_row, cols.get('stochrsi_k')) # Need previous row for cross
            d_prev = self._get_val(prev_row, cols.get('stochrsi_d'))
            label = f"StochRSI ({self.indicator_settings.stoch_k_period}/{self.indicator_settings.stoch_d_period})"
            signal = SignalState.NA
            details = ""
            value = f"K:{format_decimal(k_now)} D:{format_decimal(d_now)}"

            if k_now is not None and d_now is not None and k_prev is not None and d_prev is not None:
                # Convert to Decimal for comparison robustness
                k_now_d, d_now_d = Decimal(str(k_now)), Decimal(str(d_now))
                k_prev_d, d_prev_d = Decimal(str(k_prev)), Decimal(str(d_prev))

                # Check for bullish crossover: K crosses above D
                crossed_bullish = k_now_d > d_now_d and k_prev_d <= d_prev_d
                # Check for bearish crossover: K crosses below D
                crossed_bearish = k_now_d < d_now_d and k_prev_d >= d_prev_d

                if crossed_bullish:
                    signal = SignalState.BULLISH
                    details = "Crossed Up"
                elif crossed_bearish:
                    signal = SignalState.BEARISH
                    details = "Crossed Down"
                # If no cross, indicate the current state (K above/below D)
                elif k_now_d > d_now_d:
                    signal = SignalState.ABOVE_SIGNAL # K is above D
                    details = "K > D"
                elif k_now_d < d_now_d:
                    signal = SignalState.BELOW_SIGNAL # K is below D
                    details = "K < D"
                else:
                    signal = SignalState.NEUTRAL # K == D
                    details = "K == D"
                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        return summary_lines, signals

    def _interpret_macd(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets MACD line/signal cross and basic divergence."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names

        # --- MACD Cross ---
        signal_key = "macd_cross"
        signals[signal_key] = SignalState.NA # Default
        if flags.macd_cross:
            line_now = self._get_val(last_row, cols.get('macd_line'))
            sig_now = self._get_val(last_row, cols.get('macd_signal'))
            line_prev = self._get_val(prev_row, cols.get('macd_line'))
            sig_prev = self._get_val(prev_row, cols.get('macd_signal'))
            label = f"MACD ({self.indicator_settings.macd_fast}/{self.indicator_settings.macd_slow}/{self.indicator_settings.macd_signal})"
            signal = SignalState.NA
            details = ""
            value = f"L:{format_decimal(line_now, 4)} S:{format_decimal(sig_now, 4)}"

            if line_now is not None and sig_now is not None and line_prev is not None and sig_prev is not None:
                line_now_d, sig_now_d = Decimal(str(line_now)), Decimal(str(sig_now))
                line_prev_d, sig_prev_d = Decimal(str(line_prev)), Decimal(str(sig_prev))

                crossed_bullish = line_now_d > sig_now_d and line_prev_d <= sig_prev_d
                crossed_bearish = line_now_d < sig_now_d and line_prev_d >= sig_prev_d

                if crossed_bullish:
                    signal = SignalState.BULLISH
                    details = "Crossed Up"
                elif crossed_bearish:
                    signal = SignalState.BEARISH
                    details = "Crossed Down"
                elif line_now_d > sig_now_d:
                    signal = SignalState.ABOVE_SIGNAL
                    details = "Line > Signal"
                elif line_now_d < sig_now_d:
                    signal = SignalState.BELOW_SIGNAL
                    details = "Line < Signal"
                else:
                    signal = SignalState.NEUTRAL
                    details = "Line == Signal"
                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, precision=4, details=details)) # Use higher precision for MACD

        # --- MACD Divergence (Basic 2-point check) ---
        # WARNING: This is a very simplified check and prone to false signals.
        # Real divergence analysis requires pattern recognition over multiple pivots.
        signal_key = "macd_divergence"
        signals[signal_key] = SignalState.NA # Default
        if flags.macd_divergence:
            hist_now = self._get_val(last_row, cols.get('macd_hist'))
            hist_prev = self._get_val(prev_row, cols.get('macd_hist'))
            price_now = self._get_val(last_row, 'close')
            price_prev = self._get_val(prev_row, 'close')
            signal = SignalState.NA # Default to NA if data missing

            if hist_now is not None and hist_prev is not None and price_now is not None and price_prev is not None:
                hist_now_d, hist_prev_d = Decimal(str(hist_now)), Decimal(str(hist_prev))
                price_now_d, price_prev_d = Decimal(str(price_now)), Decimal(str(price_prev))
                zero = Decimal(0)
                signal = SignalState.NONE # Default to None (no divergence detected)

                # Basic Bullish Divergence: Lower low in price, higher low in histogram (near/below zero)
                if price_now_d < price_prev_d and hist_now_d > hist_prev_d and (hist_prev_d < zero or hist_now_d < zero):
                    signal = SignalState.BULLISH
                    summary_lines.append(Color.format("Potential Bullish MACD Divergence", Color.GREEN))

                # Basic Bearish Divergence: Higher high in price, lower high in histogram (near/above zero)
                elif price_now_d > price_prev_d and hist_now_d < hist_prev_d and (hist_prev_d > zero or hist_now_d > zero):
                    signal = SignalState.BEARISH
                    summary_lines.append(Color.format("Potential Bearish MACD Divergence", Color.RED))

                signals[signal_key] = signal
            # No separate summary line added unless divergence is detected

        return summary_lines, signals

    def _interpret_bbands(self, last_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets Bollinger Bands breakouts/position."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names

        # --- Bollinger Bands Break/Position ---
        signal_key = "bbands_signal"
        signals[signal_key] = SignalState.NA # Default
        if flags.bollinger_bands_break:
            upper = self._get_val(last_row, cols.get('bb_upper'))
            lower = self._get_val(last_row, cols.get('bb_lower'))
            middle = self._get_val(last_row, cols.get('bb_mid'))
            close_val = self._get_val(last_row, 'close')
            label = f"BBands ({self.indicator_settings.bollinger_bands_period}/{self.indicator_settings.bollinger_bands_std_dev})"
            signal = SignalState.NA
            details = ""
            value = close_val # Display current close price relative to bands

            if upper is not None and lower is not None and middle is not None and close_val is not None:
                upper_d, lower_d, middle_d, close_d = Decimal(str(upper)), Decimal(str(lower)), Decimal(str(middle)), Decimal(str(close_val))
                signal = SignalState.WITHIN_BANDS # Default assumption
                details = f"L:{format_decimal(lower_d)} M:{format_decimal(middle_d)} U:{format_decimal(upper_d)}"

                if close_d > upper_d:
                    signal = SignalState.BREAKOUT_UPPER
                    details += " (Price > Upper)"
                elif close_d < lower_d:
                    signal = SignalState.BREAKDOWN_LOWER
                    details += " (Price < Lower)"
                # Optional: Check position relative to middle band
                # elif close_d > middle_d: details += " (Above Middle)"
                # elif close_d < middle_d: details += " (Below Middle)"

                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, details=details))

        return summary_lines, signals

    def _interpret_volume(self, last_row: pd.Series, prev_row: pd.Series) -> Tuple[List[str], Dict[str, SignalState]]:
        """Interprets volume levels and volume-based indicators (OBV, ADOSC)."""
        summary_lines: List[str] = []
        signals: Dict[str, SignalState] = {}
        flags = self.analysis_flags
        cols = self.col_names

        # --- Volume Level vs MA ---
        signal_key = "volume_level"
        signals[signal_key] = SignalState.NA # Default
        if flags.volume_confirmation:
            volume = self._get_val(last_row, 'volume')
            vol_ma = self._get_val(last_row, cols.get('vol_ma'))
            label = f"Volume vs MA({self.indicator_settings.volume_ma_period})"
            signal = SignalState.NA
            details = ""
            value = volume # Display current volume

            if volume is not None and vol_ma is not None:
                vol_d, vol_ma_d = Decimal(str(volume)), Decimal(str(vol_ma))
                if vol_ma_d > Decimal(0): # Avoid division by zero
                    signal = SignalState.AVERAGE_VOLUME # Default
                    # Define thresholds relative to MA (e.g., 1.5x for high, 0.7x for low)
                    high_thresh_factor = Decimal("1.5")
                    low_thresh_factor = Decimal("0.7")

                    if vol_d > vol_ma_d * high_thresh_factor:
                        signal = SignalState.HIGH_VOLUME
                    elif vol_d < vol_ma_d * low_thresh_factor:
                        signal = SignalState.LOW_VOLUME

                    details = f"Vol:{format_decimal(vol_d, 0)} MA:{format_decimal(vol_ma_d, 0)}"
                    signals[signal_key] = signal
                else:
                    signal = SignalState.LOW_VOLUME # If MA is zero, volume is low or zero
                    details = f"Vol:{format_decimal(vol_d, 0)} MA: 0"
                    signals[signal_key] = signal


            summary_lines.append(self._format_signal(label, value, signal, precision=0, details=details))

        # --- OBV Trend ---
        signal_key = "obv_trend"
        signals[signal_key] = SignalState.NA # Default
        if flags.obv_trend:
            obv_now = self._get_val(last_row, cols.get('obv'))
            obv_prev = self._get_val(prev_row, cols.get('obv'))
            label = "OBV Trend"
            signal = SignalState.NA
            value = obv_now

            if obv_now is not None and obv_prev is not None:
                obv_now_d, obv_prev_d = Decimal(str(obv_now)), Decimal(str(obv_prev))
                signal = SignalState.FLAT
                if obv_now_d > obv_prev_d: signal = SignalState.INCREASING
                elif obv_now_d < obv_prev_d: signal = SignalState.DECREASING
                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, precision=0))

        # --- A/D Oscillator Trend (ADOSC) ---
        signal_key = "adi_trend" # Keep key name generic for Accum/Dist
        signals[signal_key] = SignalState.NA # Default
        if flags.adi_trend:
            adosc_now = self._get_val(last_row, cols.get('adosc'))
            adosc_prev = self._get_val(prev_row, cols.get('adosc'))
            label = "A/D Osc Trend"
            signal = SignalState.NA
            value = adosc_now

            if adosc_now is not None and adosc_prev is not None:
                adosc_now_d, adosc_prev_d = Decimal(str(adosc_now)), Decimal(str(adosc_prev))
                zero = Decimal(0)
                signal = SignalState.FLAT # Default trend

                # Accumulation: Generally positive and rising ADOSC
                if adosc_now_d > zero and adosc_now_d > adosc_prev_d:
                    signal = SignalState.ACCUMULATION
                # Distribution: Generally negative and falling ADOSC
                elif adosc_now_d < zero and adosc_now_d < adosc_prev_d:
                    signal = SignalState.DISTRIBUTION
                # If not clearly A/D, indicate simple direction
                elif adosc_now_d > adosc_prev_d:
                    signal = SignalState.INCREASING
                elif adosc_now_d < adosc_prev_d:
                    signal = SignalState.DECREASING

                signals[signal_key] = signal

            summary_lines.append(self._format_signal(label, value, signal, precision=0))

        return summary_lines, signals

    def _interpret_levels_orderbook(self, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> List[str]:
        """Formats the calculated levels and orderbook analysis into summary lines."""
        summary_lines = []
        summary_lines.append(Color.format("\n--- Levels & Orderbook ---", Color.BLUE))

        # --- Levels Summary ---
        pivot = levels.get("pivot")
        support_levels = levels.get("support", {})
        resistance_levels = levels.get("resistance", {})

        if pivot:
            summary_lines.append(f"Pivot Point: ${format_decimal(pivot, 4)}")
        else:
             summary_lines.append("Pivot Point: N/A")

        # Show nearest levels (e.g., top 3 closest)
        def get_nearest(level_dict: dict, price: Decimal, count: int) -> List[Tuple[str, Decimal]]:
            if not level_dict: return []
            return sorted(level_dict.items(), key=lambda item: abs(item[1] - price))[:count]

        nearest_supports = get_nearest(support_levels, current_price, 3)
        nearest_resistances = get_nearest(resistance_levels, current_price, 3)

        if nearest_supports:
            summary_lines.append("Nearest Support:")
            for name, price in nearest_supports:
                summary_lines.append(f"  > {name}: ${format_decimal(price, 4)}")
        else:
             summary_lines.append("Nearest Support: None Calculated")

        if nearest_resistances:
            summary_lines.append("Nearest Resistance:")
            for name, price in nearest_resistances:
                summary_lines.append(f"  > {name}: ${format_decimal(price, 4)}")
        else:
            summary_lines.append("Nearest Resistance: None Calculated")

        if not nearest_supports and not nearest_resistances and not pivot:
            summary_lines.append(Color.format("No significant levels calculated.", Color.YELLOW))

        # --- Orderbook Summary ---
        summary_lines.append("") # Add a blank line
        ob_limit = self.orderbook_settings.limit
        total_bid_usd = orderbook_analysis.get('total_bid_usd', Decimal(0))
        total_ask_usd = orderbook_analysis.get('total_ask_usd', Decimal(0))

        if total_bid_usd + total_ask_usd > Decimal(0):
            # Pressure is already formatted with color in _analyze_orderbook
            pressure_str = orderbook_analysis.get('pressure', SignalState.NA.value)
            summary_lines.append(f"OB Pressure (Top {ob_limit}): {pressure_str}")
            summary_lines.append(f"OB Value (Bids): ${format_decimal(total_bid_usd, 0)}")
            summary_lines.append(f"OB Value (Asks): ${format_decimal(total_ask_usd, 0)}")

            clusters = orderbook_analysis.get("clusters", [])
            if clusters:
                # Display top N clusters (e.g., 5)
                max_clusters_to_show = 5
                summary_lines.append(Color.format(f"Significant OB Clusters (Top {max_clusters_to_show}):", Color.PURPLE))
                for cluster in clusters[:max_clusters_to_show]:
                    cluster_type = cluster.get('type', 'N/A')
                    color = Color.GREEN if cluster_type == "Support" else Color.RED if cluster_type == "Resistance" else Color.YELLOW
                    level_name = cluster.get('level_name', 'N/A')
                    level_price_f = format_decimal(cluster.get('level_price'), 4)
                    cluster_val_f = format_decimal(cluster.get('cluster_value_usd'), 0)
                    summary_lines.append(Color.format(f"  {cluster_type} near {level_name} (${level_price_f}) - Value: ${cluster_val_f}", color))
            else:
                summary_lines.append(Color.format("No significant OB clusters found near levels.", Color.YELLOW))
        else:
            summary_lines.append(Color.format(f"Orderbook analysis unavailable or empty (Top {ob_limit}).", Color.YELLOW))

        return summary_lines

    def _interpret_analysis(self, df: pd.DataFrame, current_price: Decimal, levels: dict, orderbook_analysis: dict) -> dict:
        """
        Combines all interpretation steps into a final summary and signal dictionary.

        Args:
            df: DataFrame with OHLCV and calculated indicator data.
            current_price: Current market price.
            levels: Calculated support/resistance/pivot levels.
            orderbook_analysis: Results from order book analysis.

        Returns:
            A dictionary containing:
            - 'summary': A list of formatted strings for the analysis report.
            - 'signals': A dictionary mapping signal keys (e.g., 'ema_trend') to
                         their corresponding SignalState enum *values* (strings).
        """
        interpretation: Dict[str, Any] = {"summary": [], "signals": {}}

        # Define all expected signal keys to ensure they exist in the output, even if NA
        all_signal_keys = [
            "ema_trend", "adx_trend", "psar_trend", "psar_signal",
            "rsi_level", "mfi_level", "cci_level", "wr_level", "stochrsi_cross",
            "macd_cross", "macd_divergence", "bbands_signal",
            "volume_level", "obv_trend", "adi_trend"
        ]
        # Initialize all signals to NA
        interpretation["signals"] = {key: SignalState.NA for key in all_signal_keys}


        if df.empty or len(df) < 2:
            self.logger.warning(f"Insufficient data ({len(df)} rows) for full interpretation on {self.symbol}.")
            interpretation["summary"].append(Color.format("Insufficient data for full analysis interpretation.", Color.YELLOW))
            # Keep signals as NA initialized above
            return interpretation
        if not isinstance(current_price, Decimal):
             self.logger.error("Invalid current_price type for interpretation. Expected Decimal.")
             interpretation["summary"].append(Color.format("Invalid current price for interpretation.", Color.RED))
             return interpretation


        try:
            # Ensure index is sequential for iloc[-1] and [-2] access
            # Use .copy() to avoid SettingWithCopyWarning if df comes directly from calculation
            df_indexed = df.copy().reset_index(drop=True)
            last_row = df_indexed.iloc[-1]
            # Use last_row as prev_row if only one row exists after indicator calculation (handles edge case)
            prev_row = df_indexed.iloc[-2] if len(df_indexed) >= 2 else last_row

            # --- Run Interpretation Sections ---
            trend_summary, trend_signals = self._interpret_trend(last_row, prev_row)
            osc_summary, osc_signals = self._interpret_oscillators(last_row, prev_row)
            macd_summary, macd_signals = self._interpret_macd(last_row, prev_row)
            bb_summary, bb_signals = self._interpret_bbands(last_row)
            vol_summary, vol_signals = self._interpret_volume(last_row, prev_row)
            level_ob_summary = self._interpret_levels_orderbook(current_price, levels, orderbook_analysis)

            # --- Combine Results ---
            interpretation["summary"].extend(trend_summary)
            interpretation["summary"].extend(osc_summary)
            interpretation["summary"].extend(macd_summary)
            interpretation["summary"].extend(bb_summary)
            interpretation["summary"].extend(vol_summary)
            interpretation["summary"].extend(level_ob_summary)

            # Combine all signal dictionaries, overwriting defaults
            all_signals = {
                **trend_signals, **osc_signals, **macd_signals,
                **bb_signals, **vol_signals
            }

            # Update the interpretation dict, converting Enum members to their string values
            for key, signal_enum in all_signals.items():
                 if key in interpretation["signals"]: # Ensure we only update expected keys
                     interpretation["signals"][key] = signal_enum.value if isinstance(signal_enum, SignalState) else SignalState.NA.value
                 else:
                      self.logger.warning(f"Generated signal key '{key}' not in expected list. Ignoring.")


        except IndexError as e:
            self.logger.error(f"IndexError during interpretation for {self.symbol}. DataFrame length: {len(df)}. Error: {e}")
            interpretation["summary"].append(Color.format("Error accessing data rows for interpretation.", Color.RED))
            # Reset signals to NA on error
            interpretation["signals"] = {key: SignalState.NA.value for key in all_signal_keys}
        except KeyError as e:
             self.logger.error(f"KeyError during interpretation for {self.symbol}. Missing indicator column? Error: {e}")
             interpretation["summary"].append(Color.format(f"Error accessing indicator data ({e}) for interpretation.", Color.RED))
             interpretation["signals"] = {key: SignalState.NA.value for key in all_signal_keys}
        except Exception as e:
            self.logger.exception(f"Unexpected error during analysis interpretation for {self.symbol}: {e}")
            interpretation["summary"].append(Color.format(f"Unexpected error during interpretation: {e}", Color.RED))
            interpretation["signals"] = {key: SignalState.NA.value for key in all_signal_keys}

        return interpretation

    def analyze(self, df_klines: pd.DataFrame, current_price: Optional[Decimal], orderbook: Optional[dict]) -> dict:
        """
        Main analysis function orchestrating data processing, indicator calculation,
        level calculation, order book analysis, and interpretation.

        Args:
            df_klines: DataFrame containing OHLCV kline data (expects Decimals).
            current_price: The current market price as a Decimal, or None if unavailable.
            orderbook: The fetched order book dictionary, or None if unavailable.

        Returns:
            A dictionary containing the comprehensive analysis result, including raw
            indicator values, levels, order book insights, and interpretation summary/signals.
        """
        # Initialize the result structure
        analysis_result: Dict[str, Any] = {
            "symbol": self.symbol,
            "timestamp": datetime.now(APP_TIMEZONE).isoformat(),
            "current_price": "N/A",
            "kline_interval": "N/A", # Will be inferred from data
            "levels": {"support": {}, "resistance": {}, "pivot": None},
            "orderbook_analysis": {"pressure": SignalState.NA.value, "total_bid_usd": "N/A", "total_ask_usd": "N/A", "clusters": []},
            "interpretation": {"summary": [Color.format("Analysis could not be performed.", Color.RED)], "signals": {}},
            "raw_indicators": {} # Store last row's indicator values
        }

        # --- Pre-checks ---
        if current_price is None:
             self.logger.error(f"Current price is unavailable for {self.symbol}. Analysis will be incomplete.")
             analysis_result["interpretation"]["summary"] = [Color.format("Current price unavailable, analysis incomplete.", Color.RED)]
             # Allow analysis to proceed with indicators/levels, but interpretation might be limited
        else:
             analysis_result["current_price"] = format_decimal(current_price, 4) # Format price early

        if df_klines.empty:
            self.logger.error(f"Kline data is empty for {self.symbol}. Cannot perform analysis.")
            # Keep the initial error message in summary
            return analysis_result
        if len(df_klines) < 2:
            self.logger.warning(f"Kline data has only {len(df_klines)} row(s) for {self.symbol}. Analysis requires at least 2 rows for comparisons; results may be incomplete.")
            analysis_result["interpretation"]["summary"] = [Color.format("Warning: Insufficient kline data (< 2 rows) for full analysis.", Color.YELLOW)]
            # Allow analysis to proceed, but interpretation relying on prev_row will be affected


        try:
            # --- Infer Kline Interval ---
            if len(df_klines) >= 2 and 'timestamp' in df_klines.columns:
                # Calculate time difference between consecutive timestamps
                time_diffs = df_klines['timestamp'].diff()
                # Find the most common time difference (mode) as the interval
                # Use dropna() in case the first diff is NaT
                mode_diff = time_diffs.dropna().mode()
                if not mode_diff.empty:
                    # Format timedelta nicely (e.g., '0 days 00:15:00')
                    analysis_result["kline_interval"] = str(mode_diff[0])
                else:
                    # Handle cases with only one diff or variable intervals
                    median_diff = time_diffs.median()
                    if pd.notna(median_diff):
                         analysis_result["kline_interval"] = f"~{str(median_diff)} (Median)"
                    else:
                         analysis_result["kline_interval"] = "Variable/Unknown"
            elif len(df_klines) == 1:
                 analysis_result["kline_interval"] = "Single Candle"


            # --- 1. Calculate Indicators ---
            # Pass the kline df (with Decimals)
            df_with_indicators = self._calculate_indicators(df_klines)
            if df_with_indicators.empty or len(df_with_indicators) < len(df_klines):
                self.logger.error(f"Indicator calculation failed or significantly reduced data for {self.symbol}.")
                # Update summary if it hasn't been set to a more specific error yet
                if "Analysis could not be performed" in analysis_result["interpretation"]["summary"][0]:
                     analysis_result["interpretation"]["summary"] = [Color.format("Indicator calculation failed.", Color.RED)]
                return analysis_result # Stop analysis if indicators fail critically

            # Store raw indicator values from the *last* row for inspection/debugging
            if not df_with_indicators.empty:
                last_row_indicators = df_with_indicators.iloc[-1].to_dict()
                analysis_result["raw_indicators"] = {
                    # Format numbers, handle timestamps, convert others to string safely
                    k: format_decimal(v, 4) if isinstance(v, (Decimal, float, np.floating)) else
                       (v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else
                        (str(v) if pd.notna(v) and not isinstance(v, (dict, list)) else None)) # Avoid complex types
                    for k, v in last_row_indicators.items()
                    # Only include keys that are likely indicators (heuristic: check col_names) or core data
                    if k in self.col_names.values() or k in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                }


            # --- 2. Calculate Levels ---
            # Ensure current_price is Decimal before passing
            levels = {}
            if isinstance(current_price, Decimal):
                # Pass the DataFrame *with* indicators, as some level calcs might use them (though current impl doesn't)
                levels = self._calculate_levels(df_with_indicators, current_price)
                # Format level prices for the result dictionary
                analysis_result["levels"] = {
                    "support": {name: format_decimal(price, 4) for name, price in levels.get("support", {}).items()},
                    "resistance": {name: format_decimal(price, 4) for name, price in levels.get("resistance", {}).items()},
                    "pivot": format_decimal(levels.get("pivot"), 4) if levels.get("pivot") is not None else "N/A"
                }
            else:
                 analysis_result["levels"]["pivot"] = "N/A (No Price)"
                 self.logger.warning("Skipping level calculation due to missing current price.")


            # --- 3. Analyze Orderbook ---
            orderbook_analysis_raw = {}
            if orderbook and isinstance(current_price, Decimal):
                 orderbook_analysis_raw = self._analyze_orderbook(orderbook, current_price, levels)
                 # Format orderbook analysis results for output dictionary
                 analysis_result["orderbook_analysis"] = {
                    "pressure": orderbook_analysis_raw.get("pressure", SignalState.NA.value), # Pressure is already formatted
                    "total_bid_usd": format_decimal(orderbook_analysis_raw.get("total_bid_usd", Decimal(0)), 0),
                    "total_ask_usd": format_decimal(orderbook_analysis_raw.get("total_ask_usd", Decimal(0)), 0),
                    "clusters": [
                        {
                            "type": c.get("type", "N/A"),
                            "level_name": c.get("level_name", "N/A"),
                            "level_price": format_decimal(c.get("level_price"), 4),
                            "cluster_value_usd": format_decimal(c.get("cluster_value_usd"), 0),
                            # Format the tuple elements for price range
                            "price_range": (
                                format_decimal(c.get("price_range", (None, None))[0], 4),
                                format_decimal(c.get("price_range", (None, None))[1], 4)
                            )
                        } for c in orderbook_analysis_raw.get("clusters", [])
                    ]
                 }
            else:
                 self.logger.debug("Skipping orderbook analysis due to missing orderbook data or current price.")
                 # Keep default NA values in analysis_result["orderbook_analysis"]


            # --- 4. Interpret Results ---
            # Ensure current_price is Decimal before passing
            if isinstance(current_price, Decimal):
                 interpretation = self._interpret_analysis(df_with_indicators, current_price, levels, orderbook_analysis_raw)
                 analysis_result["interpretation"] = interpretation
            else:
                 # If price is missing, interpretation might be limited. Run it anyway?
                 # Or just provide a message indicating limited interpretation.
                 self.logger.warning("Running interpretation without current price. Results may be limited.")
                 interpretation = self._interpret_analysis(df_with_indicators, Decimal(0), levels, orderbook_analysis_raw) # Pass dummy price?
                 analysis_result["interpretation"] = interpretation
                 analysis_result["interpretation"]["summary"].insert(0, Color.format("Warning: Interpretation performed without current price.", Color.YELLOW))


        except Exception as e:
            self.logger.exception(f"Critical error during analysis pipeline for {self.symbol}: {e}")
            analysis_result["interpretation"]["summary"] = [Color.format(f"Critical Analysis Pipeline Error: {e}", Color.RED)]
            # Ensure signals are marked as NA in case of pipeline error
            analysis_result["interpretation"]["signals"] = {
                k: SignalState.NA.value for k in analysis_result["interpretation"].get("signals", {})
            }

        return analysis_result

    def format_analysis_output(self, analysis_result: dict) -> str:
        """
        Formats the analysis result dictionary into a human-readable string for logging/display.

        Args:
            analysis_result: The dictionary returned by the analyze() method.

        Returns:
            A formatted string summarizing the analysis.
        """
        symbol = analysis_result.get('symbol', 'N/A')
        timestamp_str = analysis_result.get('timestamp', 'N/A')
        try:
            # Attempt to parse and format timestamp nicely in the application's timezone
            dt_obj = datetime.fromisoformat(timestamp_str).astimezone(APP_TIMEZONE)
            ts_formatted = dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")
        except (ValueError, TypeError):
            ts_formatted = timestamp_str # Fallback to raw ISO string if parsing fails

        price = analysis_result.get('current_price', 'N/A')
        # Use the inferred interval from the analysis result
        interval = analysis_result.get('kline_interval', 'N/A')

        # --- Header ---
        header = f"\n--- Analysis Report for {Color.format(symbol, Color.PURPLE)} --- ({ts_formatted})\n"
        info_line = f"{Color.format('Interval:', Color.BLUE)} {interval}    {Color.format('Current Price:', Color.BLUE)} ${price}\n"

        # --- Interpretation Summary ---
        interpretation = analysis_result.get("interpretation", {})
        summary_lines = interpretation.get("summary", []) # These lines already contain color formatting

        if summary_lines:
            # Join the formatted lines from the interpretation steps
            interpretation_block = "\n".join(summary_lines) + "\n"
        else:
            interpretation_block = Color.format("No interpretation summary available.", Color.YELLOW) + "\n"

        # --- Combine Parts ---
        output = header + info_line + interpretation_block
        return output


# --- Main Application Logic ---
async def run_analysis_loop(symbol: str, interval_config: str, client: BybitCCXTClient, analyzer: TradingAnalyzer, logger_instance: logging.Logger) -> None:
    """
    The main asynchronous loop that periodically fetches data, runs analysis,
    and logs the results for a specific symbol.

    Args:
        symbol: The trading symbol to analyze.
        interval_config: The user-friendly timeframe string (e.g., "15", "1h").
        client: The initialized BybitCCXTClient instance.
        analyzer: The initialized TradingAnalyzer instance.
        logger_instance: The logger instance for this symbol.
    """
    analysis_interval_sec = float(CONFIG.analysis_interval_seconds)
    kline_limit = CONFIG.kline_limit
    orderbook_limit = analyzer.orderbook_settings.limit

    if analysis_interval_sec < 10:
        logger_instance.warning(f"Analysis interval ({analysis_interval_sec}s) is very short. Ensure system and API rate limits can handle the load.")
    if not interval_config or interval_config not in VALID_INTERVALS:
         logger_instance.critical(f"Invalid interval '{interval_config}' passed to analysis loop. Stopping.")
         return

    logger_instance.info(f"Starting analysis loop for {symbol} with interval {interval_config}...")

    while True:
        cycle_start_time = time.monotonic()
        logger_instance.debug(f"--- Starting Analysis Cycle for {symbol} ---")

        analysis_result: Optional[dict] = None # Initialize for the cycle

        try:
            # --- Fetch Data Concurrently ---
            # Create tasks for fetching data
            price_task = asyncio.create_task(client.fetch_current_price(symbol))
            klines_task = asyncio.create_task(client.fetch_klines(symbol, interval_config, kline_limit))
            # Only fetch orderbook if limit is positive
            orderbook_task = None
            if orderbook_limit > 0:
                orderbook_task = asyncio.create_task(client.fetch_orderbook(symbol, orderbook_limit))
            else:
                 logger_instance.debug("Orderbook fetching disabled (limit <= 0).")


            # --- Wait for Critical Data (Price and Klines) ---
            # Use gather to wait for both, allowing exceptions to propagate immediately
            try:
                results = await asyncio.gather(price_task, klines_task)
                current_price: Optional[Decimal] = results[0]
                df_klines: pd.DataFrame = results[1]
            except Exception as gather_err:
                 logger_instance.error(f"Error fetching critical data (price/klines) for {symbol}: {gather_err}")
                 # Cancel the orderbook task if it's still running
                 if orderbook_task and not orderbook_task.done():
                     orderbook_task.cancel()
                 await async_sleep_with_jitter(INITIAL_RETRY_DELAY_SECONDS) # Wait before retrying cycle
                 continue # Skip rest of the cycle


            # --- Validate Critical Data ---
            if current_price is None:
                logger_instance.warning(f"Failed to fetch current price for {symbol}. Analysis may be limited.")
                # Continue analysis, but price-dependent parts will be affected
            if df_klines is None or df_klines.empty:
                logger_instance.error(f"Failed to fetch valid kline data for {symbol}. Skipping analysis cycle.")
                if orderbook_task and not orderbook_task.done(): orderbook_task.cancel()
                await async_sleep_with_jitter(INITIAL_RETRY_DELAY_SECONDS)
                continue

            # --- Fetch Orderbook (if enabled) ---
            orderbook: Optional[dict] = None
            if orderbook_task:
                try:
                    # Wait for the already running task with a reasonable timeout
                    orderbook = await asyncio.wait_for(orderbook_task, timeout=15.0) # 15-second timeout for OB fetch
                except asyncio.TimeoutError:
                    logger_instance.warning(f"Fetching orderbook for {symbol} timed out. Proceeding without it.")
                    # Task is automatically cancelled by wait_for on timeout
                except asyncio.CancelledError:
                     logger_instance.debug("Orderbook task was cancelled.") # Expected if critical data failed
                except Exception as ob_err:
                    logger_instance.error(f"Error fetching or processing orderbook task result for {symbol}: {ob_err}")
                    # Ensure task is cancelled if it errored but didn't finish cleanly
                    if not orderbook_task.done(): orderbook_task.cancel()


            # --- Perform Analysis ---
            analysis_result = analyzer.analyze(df_klines, current_price, orderbook)


            # --- Output and Logging ---
            if analysis_result:
                output_string = analyzer.format_analysis_output(analysis_result)
                # Log the formatted output (includes colors for console via ColorStreamFormatter)
                # The file logger's SensitiveFormatter will strip colors.
                logger_instance.info(output_string)

                # Log structured JSON at DEBUG level (without colors)
                if logger_instance.isEnabledFor(logging.DEBUG):
                    try:
                        # Create a copy for logging to avoid modifying the original result
                        log_data = analysis_result.copy()
                        # Remove color codes from summary and pressure for clean JSON log
                        if 'interpretation' in log_data and 'summary' in log_data['interpretation']:
                            log_data['interpretation']['summary'] = [
                                SensitiveFormatter._color_code_regex.sub('', line)
                                for line in log_data['interpretation']['summary']
                            ]
                        if 'orderbook_analysis' in log_data and 'pressure' in log_data['orderbook_analysis']:
                            log_data['orderbook_analysis']['pressure'] = SensitiveFormatter._color_code_regex.sub(
                                '', log_data['orderbook_analysis']['pressure']
                            )

                        # Convert Decimals to strings for JSON compatibility
                        def decimal_default(obj):
                            if isinstance(obj, Decimal):
                                return str(obj)
                            # Let default JSON encoder handle others, or raise TypeError
                            return json.JSONEncoder().encode(obj) # Re-raise error for non-serializable

                        logger_instance.debug(f"Analysis Result JSON:\n{json.dumps(log_data, indent=2, default=decimal_default)}")

                    except TypeError as json_err:
                        logger_instance.error(f"Error serializing analysis result for JSON debug logging: {json_err}")
                    except Exception as log_json_err:
                         logger_instance.error(f"Unexpected error preparing analysis result for JSON debug logging: {log_json_err}")
            else:
                 logger_instance.error("Analysis result was None. Skipping output.")


        # --- Specific Exception Handling for the Loop ---
        except ccxt.AuthenticationError as e:
            logger_instance.critical(Color.format(f"Authentication Error during analysis cycle: {e}. Check API Key/Secret. Stopping analysis for {symbol}.", Color.RED))
            break # Stop the loop for this symbol on auth error
        except ccxt.InvalidNonce as e:
            logger_instance.critical(Color.format(f"Invalid Nonce Error: {e}. Check system time sync with server. Stopping analysis for {symbol}.", Color.RED))
            break # Stop the loop on nonce error (usually requires intervention)
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            # These might occur if trading logic were added. Log as error but continue analysis.
            logger_instance.error(Color.format(f"Order-related CCXT error encountered: {e}. Check parameters or funds if trading.", Color.RED))
            # Continue the loop for analysis purposes
        except asyncio.CancelledError:
            logger_instance.info(f"Analysis task for {symbol} was cancelled.")
            break # Exit the loop cleanly if cancelled
        except Exception as e:
            logger_instance.exception(Color.format(f"An unexpected error occurred in the main analysis loop for {symbol}: {e}", Color.RED))
            # Add a longer sleep after unexpected errors to prevent rapid failure loops
            await async_sleep_with_jitter(10.0)

        # --- Sleep Management ---
        cycle_end_time = time.monotonic()
        elapsed_time = cycle_end_time - cycle_start_time
        sleep_duration = max(0, analysis_interval_sec - elapsed_time)

        logger_instance.debug(f"Analysis cycle for {symbol} took {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s.")

        if elapsed_time > analysis_interval_sec:
            logger_instance.warning(
                f"Analysis cycle duration ({elapsed_time:.2f}s) exceeded configured interval ({analysis_interval_sec}s). "
                f"Consider increasing the interval or optimizing analysis."
            )

        await async_sleep_with_jitter(sleep_duration) # Use sleep with jitter

    logger_instance.info(f"Analysis loop stopped for {symbol}.")


async def main() -> None:
    """
    Main entry point for the application. Sets up logging, initializes the CCXT client,
    prompts user for symbol/interval, and starts the analysis loop. Handles graceful shutdown.
    """
    # Setup a temporary logger for initial setup steps
    init_logger = setup_logger("INIT")
    init_logger.info("Application starting...")
    init_logger.info(f"Using configuration from: {CONFIG_FILE_PATH}")
    init_logger.info(f"Log directory: {LOG_DIRECTORY}")
    init_logger.info(f"Timezone: {APP_TIMEZONE}")
    init_logger.info(f"API Environment: {'Testnet' if IS_TESTNET else 'Mainnet'}")

    # --- Initialize Client and Load Markets ---
    # Use a temporary client instance just for loading markets initially
    temp_client = BybitCCXTClient(API_KEY, API_SECRET, IS_TESTNET, init_logger)
    markets_loaded = await temp_client.initialize_markets()

    if not markets_loaded or not temp_client.markets:
        init_logger.critical(Color.format("Failed to load markets during initialization. Cannot proceed. Exiting.", Color.RED))
        await temp_client.close()
        logging.shutdown()
        return

    # Get valid symbols and keep market data to pass to the main client
    valid_symbols = list(temp_client.markets.keys())
    loaded_markets_data = temp_client.markets
    loaded_market_categories = temp_client.market_categories # Pass cached categories too
    await temp_client.close() # Close the temporary client connection
    init_logger.info("Market data loaded successfully.")


    # --- User Input for Symbol and Interval ---
    selected_symbol = ""
    while True:
        try:
            print(Color.format("\nAvailable markets loaded. Please enter the symbol to analyze.", Color.BLUE))
            symbol_input = input(Color.format("Enter trading symbol (e.g., BTC/USDT): ", Color.YELLOW)).strip().upper()
            if not symbol_input: continue # Ask again if empty input

            # Attempt to standardize format (e.g., BTCUSDT -> BTC/USDT)
            potential_symbol = symbol_input
            if "/" not in symbol_input and len(symbol_input) > 3: # Basic check
                # Common quote currencies
                quotes = ['USDT', 'USD', 'USDC', 'BTC', 'ETH', 'EUR', 'GBP', 'DAI', 'BUSD']
                for quote in quotes:
                    if symbol_input.endswith(quote):
                        base = symbol_input[:-len(quote)]
                        formatted = f"{base}/{quote}"
                        if formatted in valid_symbols:
                            print(Color.format(f"Assuming symbol: {formatted}", Color.CYAN))
                            potential_symbol = formatted
                            break # Found a potential match

            # Validate the (potentially formatted) symbol against loaded markets
            if potential_symbol in valid_symbols:
                selected_symbol = potential_symbol
                print(Color.format(f"Selected symbol: {selected_symbol}", Color.GREEN))
                break
            else:
                print(Color.format(f"Invalid or unsupported symbol: '{symbol_input}'.", Color.RED))
                # Suggest similar symbols based on simple substring matching
                find_similar = [s for s in valid_symbols if symbol_input in s.replace('/', '')][:10] # Limit suggestions
                if find_similar:
                    print(Color.format(f"Did you mean one of these? {', '.join(find_similar)}", Color.YELLOW))

        except EOFError:
            print(Color.format("\nInput stream closed. Exiting.", Color.YELLOW))
            logging.shutdown()
            return
        except KeyboardInterrupt:
             print(Color.format("\nOperation cancelled by user during input.", Color.YELLOW))
             logging.shutdown()
             return


    selected_interval_key = ""
    default_interval = CONFIG.indicator_settings.default_interval
    while True:
        try:
            print(Color.format(f"\nEnter analysis timeframe.", Color.BLUE))
            interval_prompt = (f"Available: [{', '.join(VALID_INTERVALS)}]\n"
                               f"Enter timeframe (default: {default_interval}): ")
            interval_input = input(Color.format(interval_prompt, Color.YELLOW)).strip()

            if not interval_input:
                selected_interval_key = default_interval
                print(Color.format(f"Using default interval: {selected_interval_key}", Color.CYAN))
                break

            # Check if input is directly in our valid keys ('1', '15', 'D', etc.)
            if interval_input in VALID_INTERVALS:
                selected_interval_key = interval_input
                print(Color.format(f"Selected interval: {selected_interval_key}", Color.GREEN))
                break

            # Check if input is a standard CCXT interval ('1m', '1h', '1d', etc.)
            if interval_input in REVERSE_CCXT_INTERVAL_MAP:
                selected_interval_key = REVERSE_CCXT_INTERVAL_MAP[interval_input]
                print(Color.format(f"Using interval {selected_interval_key} (mapped from {interval_input})", Color.CYAN))
                break

            print(Color.format(f"Invalid interval '{interval_input}'. Please choose from the list or use standard CCXT format (e.g., 1h, 4h, 1d).", Color.RED))

        except EOFError:
            print(Color.format("\nInput stream closed. Exiting.", Color.YELLOW))
            logging.shutdown()
            return
        except KeyboardInterrupt:
             print(Color.format("\nOperation cancelled by user during input.", Color.YELLOW))
             logging.shutdown()
             return

    # --- Setup Main Components for the Selected Symbol ---
    main_logger = setup_logger(selected_symbol) # Setup logger specific to the chosen symbol
    main_logger.info(f"Logger initialized for symbol {selected_symbol}.")

    # Create the main client instance, passing the already loaded market data
    client = BybitCCXTClient(API_KEY, API_SECRET, IS_TESTNET, main_logger)
    client.markets = loaded_markets_data # Assign pre-loaded markets
    client.market_categories = loaded_market_categories # Assign pre-cached categories
    main_logger.info(f"CCXT Client initialized for {selected_symbol}. Markets assigned.")

    analyzer = TradingAnalyzer(CONFIG, main_logger, selected_symbol)
    main_logger.info("Trading Analyzer initialized.")

    # Log final setup details
    ccxt_interval = CCXT_INTERVAL_MAP.get(selected_interval_key, "N/A")
    market_type = client.get_market_category(selected_symbol)
    main_logger.info(f"--- Starting Analysis ---")
    main_logger.info(f"Symbol: {Color.format(selected_symbol, Color.PURPLE)} ({market_type.capitalize()})")
    main_logger.info(f"Interval: {Color.format(selected_interval_key, Color.PURPLE)} (CCXT: {ccxt_interval})")
    main_logger.info(f"API Env: {Color.format(API_ENV.upper(), Color.YELLOW)} (URL: {client.exchange.urls['api']})")
    main_logger.info(f"Loop Interval: {CONFIG.analysis_interval_seconds} seconds")
    main_logger.info(f"Kline Limit: {CONFIG.kline_limit} candles")
    main_logger.info(f"Orderbook Limit: {CONFIG.orderbook_settings.limit} levels")
    main_logger.info(f"Timezone: {APP_TIMEZONE}")


    # --- Run Analysis Loop and Handle Shutdown ---
    main_task = None
    try:
        # Create and run the main analysis task
        main_task = asyncio.create_task(
            run_analysis_loop(selected_symbol, selected_interval_key, client, analyzer, main_logger)
        )
        await main_task # Wait for the loop to complete or be cancelled

    except KeyboardInterrupt:
        main_logger.info(Color.format("\nCtrl+C detected. Stopping analysis loop...", Color.YELLOW))
        if main_task and not main_task.done():
            main_task.cancel()
            # Wait briefly for cancellation to propagate
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        main_logger.info(Color.format("Main analysis task was cancelled.", Color.YELLOW))
        # Expected during shutdown
    except Exception as e:
        main_logger.critical(Color.format(f"A critical error occurred during main execution: {e}", Color.RED), exc_info=True)
        if main_task and not main_task.done():
            main_task.cancel()
            await asyncio.sleep(0.1)
    finally:
        main_logger.info("Initiating shutdown sequence...")
        # Ensure the client connection is closed
        await client.close()
        main_logger.info("Application finished.")

        # Flush and close logger handlers gracefully
        # Get handlers associated with this specific logger instance
        handlers = main_logger.handlers[:]
        for handler in handlers:
            try:
                handler.flush()
                handler.close()
                main_logger.removeHandler(handler)
            except Exception as e:
                # Use print as logger might be unreliable during shutdown
                print(f"Error closing handler {handler}: {e}")

        # Attempt to shutdown the entire logging system
        logging.shutdown()


if __name__ == "__main__":
    try:
        # Run the main asynchronous function
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C if it occurs *before* the main async loop starts
        # or *after* it exits but before the script terminates.
        print(f"\n{Color.YELLOW.value}Process interrupted by user. Exiting gracefully.{Color.RESET.value}")
    except Exception as e:
        # Catch any other unexpected top-level errors during script execution
        print(f"\n{Color.RED.value}A critical top-level error occurred: {e}{Color.RESET.value}")
        traceback.print_exc() # Print detailed traceback for top-level errors
