```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: 2.8
# Changelog:
# - v2.8: Integrated 25 enhancement suggestions for improved robustness, usability, and maintainability.
# - v2.3: Integrated ECC calculation and PnL fetching from separate script.
#         Added ECC scalping as a configurable strategy alongside EMA/Stoch.
#         Enhanced configuration validation and logging.
#         Improved Decimal usage and precision handling across all calculations and API interactions.
#         Refined error handling, retries, and exception handling for robustness.
#         Enhanced status panel with more details, color, and strategy-specific info.
#         Improved Termux notification handling for security.
#         Added graceful shutdown sequence to cancel orders and close positions.
#         Added Weighted Average Entry Price calculation for positions.
#         Improved Bybit V5 API interaction handling (category, implicit methods, response parsing).
#         Added checks for sufficient OHLCV data for indicator calculation.
#         Added epsilon checks for zero quantities/amounts.
# - v2.2: Added Ehlers Cyber Cycle calculation and scalping signals (in original script).
# - v2.1: Added fetch_pnl for unrealized and realized PnL (in original script).
# - v2.0: Fixed error 10004 by including recv_window in signature (in original script).
# - v1.0: Initial version with trailing stop functionality (in original script).
# Pyrmethus - Termux Trading Spell (v2.8 Enhanced)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved robustness.

import os
import time
import logging
import sys
import subprocess  # For termux-toast security
import copy  # For deepcopy of tracker state
import json  # <xaiArtifact artifact_id="c2a3767e-3119-420d-92a6-05533a61856c" title="Persist State to File" contentType="python">Snippet 9: Persist State</xaiArtifact>
import smtplib  # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
from email.mime.text import MIMEText  # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
import threading  # <xaiArtifact artifact_id="08d15550-8b52-440d-a39c-565e19f2d93d" title="Implement Health Check Endpoint" contentType="python">Snippet 14: Health Check Endpoint</xaiArtifact>
from http.server import HTTPServer, BaseHTTPRequestHandler  # <xaiArtifact artifact_id="08d15550-8b52-440d-a39c-565e19f2d93d" title="Implement Health Check Endpoint" contentType="python">Snippet 14: Health Check Endpoint</xaiArtifact>
import unittest # <xaiArtifact artifact_id="b4d1a1c5-c586-48b2-9948-b7a8c56e29e7" title="Add Unit Tests" contentType="python">Snippet 23: Unit Tests</xaiArtifact>
from cachetools import TTLCache # <xaiArtifact artifact_id="061781c2-1141-436d-a49a-47006d3c4902" title="Optimize OHLCV Fetch with Cache" contentType="python">Snippet 11: OHLCV Cache</xaiArtifact>
from datetime import datetime # <xaiArtifact artifact_id="8486574b-a355-4c77-b738-a598e5207a78" title="Add Blackout Periods" contentType="python">Snippet 21: Blackout Periods</xaiArtifact>

from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation, DivisionByZero, DecimalException # Added DecimalException

# Attempt to import necessary enchantments
try:
    import ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import init, Fore, Style, Back
    import requests  # Often needed by ccxt, good to suggest installation
except ImportError as e:
    # Provide specific guidance for Termux users
    init(autoreset=True)  # Initialize colorama for error messages
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell in your Termux terminal:")
    print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
    print(f"{Style.BRIGHT}pip install ccxt python-dotenv pandas numpy tabulate colorama requests cachetools") # Added cachetools
    sys.exit(1)

# Weave the Colorama magic into the terminal
init(autoreset=True)

# Set Decimal precision (adjust if needed)
# The default precision (usually 28) is often sufficient for most price/qty calcs.
# It might be necessary to increase if dealing with extremely small values or very high precision instruments.
# getcontext().prec = 50 # Example: Increase precision if needed
# By default, the Decimal context is set with sufficient precision (typically 28 digits).

# --- Arcane Configuration ---
print(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.8...")

# Summon secrets from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe
# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5  # Between INFO and WARNING
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

def trade_log(self, message, *args, **kws):
    """Custom logging method for trade-related events."""
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(TRADE_LEVEL_NUM, message, args, **kws)

# Add the custom method to the Logger class if it doesn't exist
if not hasattr(logging.Logger, 'trade'):
    logging.Logger.trade = trade_log

# More detailed log format, includes module and line number for easier debugging
log_formatter = logging.Formatter(
    Fore.CYAN + "%(asctime)s "
    + Style.BRIGHT + "[%(levelname)-8s] "  # Padded levelname
    + Fore.WHITE + "(%(filename)s:%(lineno)d) "  # Added file/line info
    + Style.RESET_ALL
    + Fore.WHITE + "%(message)s"
)
logger = logging.getLogger(__name__)

# <xaiArtifact artifact_id="36b43938-1767-4c6c-9805-319220c7b853" title="Add Configurable Logging Levels per Module" contentType="python">Snippet 24: Configurable Logging Levels</xaiArtifact>
# Set levels via environment variable or default
module_log_levels = {
    'ccxt': os.getenv("CCXT_LOG_LEVEL", "WARNING").upper(),
    'bot': os.getenv("BOT_LOG_LEVEL", "INFO").upper() # Assuming main bot logger is named 'bot' implicitly or uses __name__
}
# Set level for CCXT logger
logging.getLogger('ccxt').setLevel(getattr(logging, module_log_levels['ccxt'], logging.WARNING))
# Set level for the main bot logger (__name__)
logger.setLevel(getattr(logging, module_log_levels['bot'], logging.INFO))
# </xaiArtifact>

# Explicitly use stdout to avoid potential issues in some environments
# Ensure handlers are not duplicated if script is reloaded or run multiple times in same process
# Remove existing handlers if any before adding the desired ones
if logger.hasHandlers():
    logger.handlers.clear()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
# Set handler level to match logger level (or lower)
stream_handler.setLevel(logger.level)
logger.addHandler(stream_handler)

# Prevent duplicate messages if the root logger is also configured (common issue)
logger.propagate = False

# <xaiArtifact artifact_id="4386574b-a355-4c77-b738-a598e5207a78" title="Enhance Logging with Structured JSON Output" contentType="python">Snippet 3: JSON Logging</xaiArtifact>
# Add JSON handler for easier log parsing (optional, can be toggled via env var)
# NOTE: Adding a second handler will result in duplicate log messages, one formatted, one JSON.
# A more robust solution would replace the default handler or use a different logger name.
# Integrating as per snippet instruction, which adds a second handler.
if os.getenv("ENABLE_JSON_LOGGING", "False").lower() in ['true', '1', 'yes']:
    json_handler = logging.StreamHandler(sys.stdout)
    # Custom JSON formatter that safely handles messages that might contain quotes or special chars
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "module": record.filename,
                "line": record.lineno,
                # Escape the message string to be safe within JSON
                "message": record.getMessage().replace('"', '\\"').replace('\\', '\\\\')
            }
            # Add exc_info, stack_info if available
            if record.exc_info:
                 log_record['exc_info'] = self.formatException(record.exc_info)
            if record.stack_info:
                 log_record['stack_info'] = self.formatStack(record.stack_info)

            return json.dumps(log_record)

    json_handler.setFormatter(JsonFormatter())
    json_handler.setLevel(logger.level) # Match logger level
    logger.addHandler(json_handler)
    logger.info("JSON logging enabled.")
# </xaiArtifact>


class TradingConfig:
    """Holds the sacred parameters of our spell, enhanced with precision awareness and validation."""

    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        # Default symbol format for Bybit V5 Unified is BASE/QUOTE:SETTLE, e.g., BTC/USDT:USDT
        # <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
        self.symbols = self._get_env("SYMBOLS", "BTC/USDT:USDT", Fore.YELLOW).split(",")
        if not self.symbols:
             logger.critical(Fore.RED + Style.BRIGHT + "No symbols configured! Set SYMBOLS in .env. Halting.")
             sys.exit(1)
        # For now, keep CONFIG.symbol as the first symbol for backward compatibility with single-symbol logic
        # This needs refactoring for full multi-symbol support where each cycle processes one symbol.
        self.symbol = self.symbols[0] # Use the first symbol as the default for single-symbol logic
        # </xaiArtifact>

        self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW, allowed_values=["linear", "inverse"]).lower()
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        # Risk as a percentage of total equity (e.g., 0.01 for 1%, 0.001 for 0.1%)
        self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5"))  # 0.001% to 50% risk
        self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"))
        # TSL activation threshold in ATR units above entry price
        self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"))
        # Bybit V5 TSL distance is a percentage (e.g., 0.5 for 0.5%). Ensure value is suitable.
        self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.001"), max_val=Decimal("10.0"))  # 0.001% to 10% trail
        # Trigger type for SL/TSL orders. Bybit V5 allows LastPrice, MarkPrice, IndexPrice.
        self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])  # Usually same as SL

        # Trading Strategy Selection
        # Supported strategies: 'ecc_scalp', 'ema_stoch'
        self.strategy = self._get_env("STRATEGY", "ecc_scalp", Fore.YELLOW, allowed_values=["ecc_scalp", "ema_stoch"]).lower()

        # ECC Specific Parameters
        self.ecc_alpha = self._get_env("ECC_ALPHA", "0.15", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.01"), max_val=Decimal("0.5"))
        self.ecc_lookback = self._get_env("ECC_LOOKBACK", "10", Fore.YELLOW, cast_type=int, min_val=5)

        # EMA/Stoch Specific Parameters (if strategy is ema_stoch)
        # Using defaults that require slightly more data than the minimum checks below,
        # but are common settings. OHLCV_LIMIT should be increased if these are larger.
        self.ema_fast_period = self._get_env("EMA_FAST_PERIOD", "8", Fore.YELLOW, cast_type=int, min_val=2)
        self.ema_slow_period = self._get_env("EMA_SLOW_PERIOD", "12", Fore.YELLOW, cast_type=int, min_val=2)
        self.stoch_period = self._get_env("STOCH_PERIOD", "10", Fore.YELLOW, cast_type=int, min_val=5)
        self.stoch_smooth_k = self._get_env("STOCH_SMOOTH_K", "3", Fore.YELLOW, cast_type=int, min_val=1)
        self.stoch_smooth_d = self._get_env("STOCH_SMOOTH_D", "3", Fore.YELLOW, cast_type=int, min_val=1)
        self.trend_ema_period = self._get_env("TREND_EMA_PERIOD", "20", Fore.YELLOW, cast_type=int, min_val=5, max_val=500)
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)

        # ATR Period
        self.atr_period = self._get_env("ATR_PERIOD", "10", Fore.YELLOW, cast_type=int, min_val=5)

        # PnL Fetching
        self.fetch_pnl_interval_cycles = self._get_env("FETCH_PNL_INTERVAL_CYCLES", "10", Fore.YELLOW, cast_type=int, min_val=1) # Fetch PnL every N cycles
        self.pnl_lookback_days = self._get_env("PNL_LOOKBACK_DAYS", "7", Fore.YELLOW, cast_type=int, min_val=1) # Lookback for realized PnL history

        # Epsilon: Small value for comparing quantities, dynamically determined after market info is loaded.
        self.position_qty_epsilon = Decimal("1E-12")  # Default tiny Decimal, will be overridden based on market precision

        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        # Minimum OHLCV limit required for indicator calculations
        # ECC needs lookback + initial smoothing (roughly 4 + lookback for formula start)
        # EMA needs max(fast, slow, trend)
        # Stoch needs period + smooth_k + smooth_d - 2
        # ATR needs period + 1
        # Take max of all requirements for the chosen strategy and ATR
        min_ohlcv_required_atr = self.atr_period + 1 # Minimum for ATR

        min_ohlcv_required_indicators = min_ohlcv_required_atr

        if self.strategy == 'ecc_scalp':
            min_ohlcv_required_indicators = max(min_ohlcv_required_indicators, self.ecc_lookback + 5, self.trend_ema_period) # ECC formula needs ~5 + lookback, trend EMA
        elif self.strategy == 'ema_stoch':
             min_ohlcv_required_indicators = max(min_ohlcv_required_indicators,
                                                 max(self.ema_fast_period, self.ema_slow_period, self.trend_ema_period),
                                                 self.stoch_period + self.stoch_smooth_k + self.stoch_smooth_d - 2) # Stoch needs period + smooth_k + smooth_d - 2

        # Add a buffer for safer calculation stability (e.g., 10-20 bars)
        recommended_ohlcv_buffer = 20
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", str(min_ohlcv_required_indicators + recommended_ohlcv_buffer), Fore.YELLOW, cast_type=int, min_val=max(50, min_ohlcv_required_indicators + 5)) # Ensure minimum is sensible

        if self.ohlcv_limit < min_ohlcv_required_indicators:
            logger.warning(f"{Fore.YELLOW}OHLCV_LIMIT ({self.ohlcv_limit}) is less than the minimum required ({min_ohlcv_required_indicators}) for stable indicator calculations. Consider increasing OHLCV_LIMIT.")
        elif self.ohlcv_limit < min_ohlcv_required_indicators + recommended_ohlcv_buffer:
            logger.info(f"{Fore.CYAN}OHLCV_LIMIT ({self.ohlcv_limit}) is less than recommended ({min_ohlcv_required_indicators + recommended_ohlcv_buffer}) for indicator stability. Consider increasing OHLCV_LIMIT.")


        self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5)  # Minimum sleep time
        self.order_check_delay_seconds = self._get_env("ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1)
        self.order_check_timeout_seconds = self._get_env("ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5)
        self.max_fetch_retries = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)

        # <xaiArtifact artifact_id="e2d1c1a2-1a2b-3c4d-5e6f-7a8b9c0d1e2f" title="Add Configuration for Maximum Position Size" contentType="python">Snippet 1: Max Position Size</xaiArtifact>
        self.max_position_percentage = self._get_env(
            "MAX_POSITION_PERCENTAGE", "0.5", Fore.YELLOW,
            cast_type=Decimal, min_val=Decimal("0.01"), max_val=Decimal("1.0")
        )  # Max 50% of equity
        # </xaiArtifact>

        # <xaiArtifact artifact_id="f3d2c1b0-2b1a-4d3c-6e5f-8a9b0c1d2e3f" title="Implement Cooldown Period After Trades" contentType="python">Snippet 2: Cooldown Period</xaiArtifact>
        self.trade_cooldown_seconds = self._get_env(
            "TRADE_COOLDOWN_SECONDS", "300", Fore.YELLOW,
            cast_type=int, min_val=0
        )
        # </xaiArtifact>

        # <xaiArtifact artifact_id="a1b2c3d4-e5f6-7890-1234-567890abcdef" title="Add Circuit Breaker for Consecutive Losses" contentType="python">Snippet 4: Circuit Breaker</xaiArtifact>
        self.max_consecutive_losses = self._get_env(
            "MAX_CONSECUTIVE_LOSSES", "3", Fore.YELLOW,
            cast_type=int, min_val=1
        )
        # </xaiArtifact>

        # <xaiArtifact artifact_id="d9f0e1c2-a3b4-5c6d-7e8f-9a0b1c2d3e4f" title="Add Market Volatility Filter" contentType="python">Snippet 8: Market Volatility Filter</xaiArtifact>
        self.max_atr_volatility = self._get_env(
            "MAX_ATR_VOLATILITY", "0.05", Fore.YELLOW,
            cast_type=Decimal, min_val=Decimal("0.001")
        )
        # </xaiArtifact>

        # <xaiArtifact artifact_id="f8e9d0c1-a2b3-4c5d-6e7f-8a9b0c1d2e3a" title="Add Support for Take-Profit Orders" contentType="python">Snippet 6: Take-Profit Orders</xaiArtifact>
        self.tp_atr_multiplier = self._get_env(
            "TP_ATR_MULTIPLIER", "2.0", Fore.YELLOW,
            cast_type=Decimal, min_val=Decimal("0.1")
        )
        # </xaiArtifact>

        # <xaiArtifact artifact_id="c3a4b5c6-d7e8-9f01-2345-67890abcdeff" title="Add Leverage Configuration" contentType="python">Snippet 13: Leverage Configuration</xaiArtifact>
        self.leverage = self._get_env(
            "LEVERAGE", "10", Fore.YELLOW,
            cast_type=int, min_val=1, max_val=100
        )
        # </xaiArtifact>

        # <xaiArtifact artifact_id="7b8c9d0e-1f2e-3d4c-5b6a-7f8e9d0c1b2a" title="Add Funding Rate Check" contentType="python">Snippet 17: Funding Rate Check</xaiArtifact>
        self.max_funding_rate = self._get_env(
            "MAX_FUNDING_RATE", "0.001", Fore.YELLOW,
            cast_type=Decimal, min_val=Decimal("0")
        )
        # </xaiArtifact>

        # <xaiArtifact artifact_id="1f2e3d4c-5b6a-7f8e-9d0c-1b2a3c4d5e6f" title="Add Position Age Timeout" contentType="python">Snippet 18: Position Age Timeout</xaiArtifact>
        self.max_position_age_hours = self._get_env(
            "MAX_POSITION_AGE_HOURS", "24", Fore.YELLOW,
            cast_type=int, min_val=1
        )
        # </xaiArtifact>

        # <xaiArtifact artifact_id="e9f0a1b2-c3d4-5e6f-7a8b-9c0d1e2f3a4b" title="Add Blackout Periods" contentType="python">Snippet 21: Blackout Periods</xaiArtifact>
        # e.g., "00:00-01:00,12:00-13:00" (UTC times)
        self.blackout_hours_str = self._get_env("BLACKOUT_HOURS", "", Fore.YELLOW)
        self.blackout_hours = []
        if self.blackout_hours_str:
            for period_str in self.blackout_hours_str.split(","):
                try:
                    start_time_str, end_time_str = period_str.strip().split("-")
                    start_time = datetime.strptime(start_time_str, "%H:%M").time()
                    end_time = datetime.strptime(end_time_str, "%H:%M").time()
                    self.blackout_hours.append((start_time, end_time))
                except (ValueError, IndexError):
                    logger.error(f"{Fore.RED}Invalid BLACKOUT_HOURS format: '{period_str}'. Use HH:MM-HH:MM.")
                    self.blackout_hours = [] # Clear list if any period is invalid
                    break # Stop processing invalid config

        # </xaiArtifact>

        # <xaiArtifact artifact_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890" title="Add Dry Run Mode" contentType="python">Snippet 25: Dry Run Mode</xaiArtifact>
        self.dry_run = self._get_env(
            "DRY_RUN", "False", Fore.YELLOW,
            cast_type=bool
        )
        # </xaiArtifact>


        if not self.api_key or not self.api_secret:
            logger.critical(Fore.RED + Style.BRIGHT + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting.")
            sys.exit(1)

        logger.debug("Configuration loaded successfully.")

    def _get_env(self, key: str, default: Any, color: str, cast_type: type = str,
                 min_val: Optional[Union[int, Decimal]] = None,
                 max_val: Optional[Union[int, Decimal]] = None,
                 allowed_values: Optional[List[str]] = None) -> Any:
        """Gets value from environment, casts, validates, and logs."""
        value_str = os.getenv(key)
        # Mask secrets in logs
        log_value = "****" if "SECRET" in key or "KEY" in key else value_str

        is_default = False
        if value_str is None or value_str.strip() == "":  # Treat empty string as not set
            value = default
            is_default = True
            if default is not None:
                # Only log default if it's not a secret
                logger.warning(f"{color}Using default value for {key}: {default if 'SECRET' not in key and 'KEY' not in key else '****'}")
            # Use default value string for casting below if needed
            value_str = str(default) if default is not None else None
        else:
            logger.info(f"{color}Summoned {key}: {log_value}")

        # Handle case where default is None and no value is set
        if value_str is None:
            if default is None:
                # If no value set and no default, return None (handled by required checks later)
                return None
            else:
                # This case should be covered by the is_default logic above, but double check
                logger.warning(f"{color}Value for {key} not found, using default: {default}")
                value = default
                value_str = str(default)

        # --- Casting ---
        casted_value = None
        try:
            if cast_type == bool:
                # Case-insensitive check for common truthy strings
                casted_value = value_str.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                casted_value = Decimal(value_str)
            elif cast_type == int:
                casted_value = int(value_str)
            elif cast_type == float:
                casted_value = float(value_str)  # Generally avoid float for critical values, but allow if needed
            else:  # Default is str
                casted_value = str(value_str)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Could not cast {key} ('{value_str}') to {cast_type.__name__}: {e}. Using default: {default}")
            # Attempt to cast the default value itself
            try:
                if default is None: return None
                # Recast default carefully
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting.")
                sys.exit(1)

        # --- Validation ---
        if casted_value is None:  # Should not happen if casting succeeded or defaulted
            logger.critical(f"{Fore.RED+Style.BRIGHT}Failed to obtain a valid value for {key}. Halting.")
            sys.exit(1)

        # Allowed values check (for strings like trigger types, strategy)
        if allowed_values and casted_value not in allowed_values:
            logger.error(f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}. Using default: {default}")
            # Return default after logging error
            try:
                if default is None: return None # Default is None, return None
                # Re-cast default to ensure correct type is returned
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} cannot be cast to {cast_type.__name__} for validation fallback. Halting.")
                sys.exit(1)


        # Min/Max checks (for numeric types - Decimal, int, float)
        validation_failed = False
        try:
            # Check if casted_value is a number before comparing
            if isinstance(casted_value, (int, float, Decimal)):
                if min_val is not None:
                    # Ensure min_val is Decimal if casted_value is Decimal for accurate comparison
                    min_val_comp = Decimal(str(min_val)) if isinstance(casted_value, Decimal) else min_val
                    if casted_value < min_val_comp:
                        logger.error(f"{Fore.RED}{key} value {casted_value} is below minimum {min_val}. Using default: {default}")
                        validation_failed = True
                if max_val is not None:
                    # Ensure max_val is Decimal if casted_value is Decimal
                    max_val_comp = Decimal(str(max_val)) if isinstance(casted_value, Decimal) else max_val
                    if casted_value > max_val_comp:
                        logger.error(f"{Fore.RED}{key} value {casted_value} is above maximum {max_val}. Using default: {default}")
                        validation_failed = True
        except InvalidOperation as e:
            logger.error(f"{Fore.RED}Error during min/max validation for {key} with value {casted_value} and limits ({min_val}, {max_val}): {e}. Using default: {default}")
            validation_failed = True
        except TypeError as e:
            logger.warning(f"Skipping min/max validation for non-numeric type {type(casted_value).__name__} for key {key}. Error: {e}")


        if validation_failed:
            # Re-cast default to ensure correct type is returned
            try:
                if default is None: return None
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation):
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}. Halting.")
                sys.exit(1)

        return casted_value


CONFIG = TradingConfig()
# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# MARKET_INFO is now a dictionary of market infos keyed by symbol
MARKET_INFOS: Dict[str, Dict] = {}
# </xaiArtifact>
EXCHANGE: Optional[ccxt.Exchange] = None  # Global for the exchange instance

# --- Exchange Nexus Initialization ---
print(Fore.MAGENTA + Style.BRIGHT + "\nEstablishing Nexus with the Exchange v2.8...")
try:
    exchange_options = {
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True,  # CCXT built-in rate limiter
        "options": {
            'defaultType': 'swap',  # More specific for futures/swaps than 'future'
            'defaultSubType': CONFIG.market_type,  # 'linear' or 'inverse'
            'adjustForTimeDifference': True,  # Auto-sync clock with server
            # Bybit V5 API often requires 'category' for unified endpoints
            'brokerId': 'PyrmethusV28',  # Custom identifier for Bybit API tracking
            'v5': {'category': CONFIG.market_type}  # Explicitly set category for V5 requests
        }
    }
    # Log options excluding secrets for debugging
    log_options = exchange_options.copy()
    log_options['apiKey'] = '****'
    log_options['secret'] = '****'
    logger.debug(f"Initializing CCXT Bybit with options: {log_options}")

    EXCHANGE = ccxt.bybit(exchange_options)

    # Test connectivity and credentials (important!)
    logger.info("Verifying credentials and connection...")
    EXCHANGE.check_required_credentials()  # Checks if keys are present/formatted ok
    logger.info("Credentials format check passed.")
    # Fetch time to verify connectivity, API key validity, and clock sync
    server_time = EXCHANGE.fetch_time()
    local_time = EXCHANGE.milliseconds()
    time_diff = abs(server_time - local_time)
    logger.info(f"Exchange time synchronized: {EXCHANGE.iso8601(server_time)} (Difference: {time_diff} ms)")
    if time_diff > 5000:  # Warn if clock skew is significant (e.g., > 5 seconds)
        logger.warning(f"{Fore.YELLOW}Significant time difference ({time_diff} ms) between system and exchange. Check system clock synchronization.")

    # Load markets (force reload to ensure fresh data)
    logger.info("Loading market spirits (market data)...")
    EXCHANGE.load_markets(True)  # Force reload
    logger.info(Fore.GREEN + Style.BRIGHT + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets).")

    # <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
    # Verify all configured symbols exist and get market details
    for symbol in CONFIG.symbols:
        if symbol not in EXCHANGE.markets:
            logger.error(Fore.RED + Style.BRIGHT + f"Symbol {symbol} not found in Bybit {CONFIG.market_type} market spirits.")
            # Suggest available symbols more effectively (logic adapted from original single-symbol check)
            available_symbols = []
            try:
                # Extract settle currency robustly (handles SYMBOL/QUOTE:SETTLE format)
                settle_currency_candidates = symbol.split(':')
                settle_currency = settle_currency_candidates[-1] if len(settle_currency_candidates) > 1 else None
                if settle_currency:
                    logger.info(f"Searching for active symbols settling in {settle_currency} for market type '{CONFIG.market_type}'...")
                    for s, m in EXCHANGE.markets.items():
                        is_correct_type = (CONFIG.market_type == 'linear' and m.get('linear', False)) or \
                                          (CONFIG.market_type == 'inverse' and m.get('inverse', False))
                        if m.get('active', False) and is_correct_type and m.get('settle') == settle_currency:
                            available_symbols.append(s)
                else:
                    logger.warning(f"Could not parse settle currency from SYMBOL '{symbol}'. Cannot filter suggestions.")
                    logger.info(f"Listing all active symbols for market type '{CONFIG.market_type}'...")
                    for s, m in EXCHANGE.markets.items():
                        is_correct_type = (CONFIG.market_type == 'linear' and m.get('linear', False)) or \
                                          (CONFIG.market_type == 'inverse' and m.get('inverse', False))
                        if m.get('active', False) and is_correct_type:
                            available_symbols.append(s)

            except Exception as e:
                logger.error(f"Error suggesting symbols: {e}")

            suggestion_limit = 30
            if available_symbols:
                suggestions = ", ".join(sorted(available_symbols)[:suggestion_limit])
                if len(available_symbols) > suggestion_limit:
                    suggestions += "..."
                logger.info(Fore.CYAN + f"Available active {CONFIG.market_type} symbols (sample): " + suggestions)
            else:
                logger.info(Fore.CYAN + f"Could not find any active {CONFIG.market_type} symbols to suggest.")
            sys.exit(1) # Exit if any configured symbol is invalid
        else:
            MARKET_INFOS[symbol] = EXCHANGE.market(symbol)
            logger.info(Fore.CYAN + f"Market spirit for {symbol} acknowledged (ID: {MARKET_INFOS[symbol].get('id')}).")

            # --- Log key precision and limits using Decimal (per symbol) ---
            try:
                market_info = MARKET_INFOS[symbol]
                price_precision_raw = market_info['precision'].get('price')
                amount_precision_raw = market_info['precision'].get('amount')
                min_amount_raw = market_info['limits']['amount'].get('min')
                max_amount_raw = market_info['limits']['amount'].get('max')
                contract_size_raw = market_info.get('contractSize', '1')
                min_cost_raw = market_info['limits'].get('cost', {}).get('min')

                price_prec_str = str(price_precision_raw) if price_precision_raw is not None else "N/A"
                amount_prec_str = str(amount_precision_raw) if amount_precision_raw is not None else "N/A"
                min_amount_dec = Decimal(str(min_amount_raw)) if min_amount_raw is not None else Decimal("NaN")
                max_amount_dec = Decimal(str(max_amount_raw)) if max_amount_raw is not None else Decimal("Infinity")
                contract_size_dec = Decimal(str(contract_size_raw)) if contract_size_raw is not None else Decimal("NaN")
                min_cost_dec = Decimal(str(min_cost_raw)) if min_cost_raw is not None else Decimal("NaN")

                logger.debug(f"[{symbol}] Market Precision: Price Tick/Decimals={price_prec_str}, Amount Step/Decimals={amount_prec_str}")
                logger.debug(f"[{symbol}] Market Limits: Min Amount={min_amount_dec.normalize()}, Max Amount={max_amount_dec.normalize()}, Min Cost={min_cost_dec.normalize()}")
                logger.debug(f"[{symbol}] Contract Size: {contract_size_dec.normalize()}")

                # --- Dynamically set epsilon based on amount precision (step size) ---
                amount_step_size = market_info['precision'].get('amount')
                if amount_step_size is not None:
                    try:
                        # A fixed tiny Decimal like 1E-12 is generally safe and simpler.
                        # Note: CONFIG.position_qty_epsilon is global. If symbols have wildly different precisions,
                        # this might need to be stored per symbol or adjusted per trade.
                        # For simplicity now, it's a global minimum threshold.
                        # We'll use the smallest step size across all symbols or a fixed small value.
                        # Let's just keep the fixed small value set in CONFIG init.
                         pass # Keep the fixed CONFIG.position_qty_epsilon for now.
                    except Exception:
                         logger.warning(f"[{symbol}] Could not parse amount step size '{amount_step_size}'. Using global default epsilon.")
                else:
                     logger.warning(f"[{symbol}] Market info does not provide amount step size ('precision.amount'). Using global default epsilon.")


            except (KeyError, TypeError, InvalidOperation) as e:
                logger.critical(f"{Fore.RED+Style.BRIGHT}[{symbol}] Failed to parse critical market info (precision/limits/size) from MARKET_INFO: {e}. Halting.", exc_info=True)
                logger.debug(f"Problematic MARKET_INFO for {symbol}: {MARKET_INFOS[symbol]}")
                sys.exit(1)

    # Set leverage for all symbols
    for symbol in CONFIG.symbols:
        try:
            logger.info(f"Setting leverage to {CONFIG.leverage}x for {symbol}")
            # Use fetch_with_retries for setting leverage
            # Bybit V5 set_leverage requires symbol and category
            set_leverage_params = {'category': CONFIG.market_type}
            # CCXT's set_leverage method handles the symbol and leverage value
            leverage_response = fetch_with_retries(EXCHANGE.set_leverage, CONFIG.leverage, symbol, params=set_leverage_params)

            if leverage_response is None:
                logger.error(Fore.RED + f"Failed to set leverage for {symbol} after retries.")
                # Decide if this is critical. Usually, it is.
                # Continue for now, but trading might fail later.
                termux_notify("Leverage Failed!", f"Failed to set leverage for {symbol}. Check logs.")
            elif isinstance(leverage_response, dict) and leverage_response.get('info', {}).get('retCode') == 0:
                 logger.info(Fore.GREEN + f"Set leverage successfully for {symbol}.")
            else:
                 error_msg = leverage_response.get('info', {}).get('retMsg', 'Unknown error') if isinstance(leverage_response, dict) else str(leverage_response)
                 error_code = leverage_response.get('info', {}).get('retCode', 'N/A') if isinstance(leverage_response, dict) else 'N/A'
                 logger.error(Fore.RED + f"Failed to set leverage for {symbol}. Exchange message: {error_msg} (Code: {error_code}). MANUAL CHECK REQUIRED!")
                 termux_notify("Leverage Failed!", f"Failed to set leverage for {symbol}. Msg: {error_msg}. Check logs.")

        except Exception as e:
             logger.error(Fore.RED + f"Unexpected error setting leverage for {symbol}: {e}", exc_info=True)
             termux_notify("Leverage Error!", f"Error setting leverage for {symbol}. Check logs.")

    # </xaiArtifact>

except ccxt.AuthenticationError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Authentication failed! Check API Key/Secret validity and permissions. Error: {e}")
    send_email("Bot Critical Error", f"Authentication failed: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
    sys.exit(1)
except ccxt.NetworkError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Network error connecting to Bybit: {e}. Check internet connection and Bybit status.")
    send_email("Bot Critical Error", f"Network error: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Bybit exchange is currently unavailable: {e}. Check Bybit status.")
    send_email("Bot Critical Error", f"Exchange not available: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
    sys.exit(1)
except ccxt.ExchangeError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Exchange Nexus Error during initialization: {e}", exc_info=True)
    send_email("Bot Critical Error", f"Exchange error during init: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
    sys.exit(1)
except Exception as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error during Nexus initialization: {e}", exc_info=True)
    send_email("Bot Critical Error", f"Unexpected error during init: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
    sys.exit(1)


# --- Global State Runes ---
# Tracks active SL/TSL order IDs or position-based markers associated with a potential long or short position.
# Reset when a position is closed or a new entry order is successfully placed.
# Uses placeholders like "POS_SL_LONG", "POS_TSL_LONG" for Bybit V5 position-based stops.
# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# order_tracker is now keyed by symbol
order_tracker: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {} # {symbol: {"long": {"sl_id": None, "tsl_id": None, "tp_id": None}, "short": {...}}}
# </xaiArtifact>

# <xaiArtifact artifact_id="a1b2c3d4-e5f6-7890-1234-567890abcdef" title="Add Circuit Breaker for Consecutive Losses" contentType="python">Snippet 4: Circuit Breaker</xaiArtifact>
# consecutive_losses is now keyed by symbol
consecutive_losses: Dict[str, int] = {}
# </xaiArtifact>

# <xaiArtifact artifact_id="f3d2c1b0-2b1a-4d3c-6e5f-8a9b0c1d2e3f" title="Implement Cooldown Period After Trades" contentType="python">Snippet 2: Cooldown Period</xaiArtifact>
# last_trade_time is now keyed by symbol
last_trade_time: Dict[str, float] = {}
# </xaiArtifact>

# <xaiArtifact artifact_id="03f4e5d6-a7b8-9c0d-1e2f-3a4b5c6d7e8f" title="Add Performance Metrics" contentType="python">Snippet 20: Performance Metrics</xaiArtifact>
# trades list for performance tracking (basic entry recording)
# NOTE: This snippet only records entry. A full PnL calculation needs exit tracking.
trades: List[Dict] = []
# </xaiArtifact>


# <xaiArtifact artifact_id="9d0e1f2a-3b4c-5d6e-7f8a-9b0c1d2e3f4a" title="Implement Cooldown Period After Trades" contentType="python">Snippet 2: Cooldown Period</xaiArtifact>
# Function to check if cooldown is active for a symbol
def is_cooldown_active(symbol: str) -> bool:
    global last_trade_time
    # Use .get() with default 0 to handle symbols not yet in the dict
    last_time = last_trade_time.get(symbol, 0)
    current_time = time.time()
    if current_time - last_time < CONFIG.trade_cooldown_seconds:
        remaining = CONFIG.trade_cooldown_seconds - (current_time - last_time)
        logger.info(f"[{symbol}] Cooldown active, skipping trade entry for {remaining:.0f}s.")
        return True
    return False

# Function to update cooldown time after a trade
def update_cooldown_time(symbol: str) -> None:
    global last_trade_time
    last_trade_time[symbol] = time.time()
    logger.debug(f"[{symbol}] Cooldown timer updated.")
# </xaiArtifact>


# <xaiArtifact artifact_id="c2a3767e-3119-420d-92a6-05533a61856c" title="Persist State to File" contentType="python">Snippet 9: Persist State</xaiArtifact>
STATE_FILE = "bot_state.json"

def save_state() -> None:
    """Saves the order_tracker and other relevant state to a JSON file."""
    global order_tracker, last_trade_time, consecutive_losses
    state = {
        "order_tracker": order_tracker,
        "last_trade_time": last_trade_time,
        "consecutive_losses": consecutive_losses,
        # Add other state variables here if needed (e.g., trades list, though that might grow large)
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        logger.debug("Saved bot state.")
    except Exception as e:
        logger.error(f"Failed to save state to {STATE_FILE}: {e}")

def load_state() -> None:
    """Loads the order_tracker and other relevant state from a JSON file."""
    global order_tracker, last_trade_time, consecutive_losses
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            # Load states, providing defaults if keys are missing (e.g., adding new state variables)
            order_tracker = state.get("order_tracker", {})
            last_trade_time = state.get("last_trade_time", {})
            consecutive_losses = state.get("consecutive_losses", {})
            # Load other states here

        logger.debug("Loaded bot state.")
        # Ensure dictionaries are initialized for all configured symbols if not present in state
        for symbol in CONFIG.symbols:
             if symbol not in order_tracker:
                  order_tracker[symbol] = {"long": {"sl_id": None, "tsl_id": None, "tp_id": None}, "short": {"sl_id": None, "tsl_id": None, "tp_id": None}} # Added tp_id
             # Ensure inner structure exists if loading older state file format
             if "long" not in order_tracker[symbol]: order_tracker[symbol]["long"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
             if "short" not in order_tracker[symbol]: order_tracker[symbol]["short"] = {"sl_id": None, "tsl_id": None, "tp_id": None}
             # Ensure tp_id exists if loading older state file format
             if "tp_id" not in order_tracker[symbol]["long"]: order_tracker[symbol]["long"]["tp_id"] = None
             if "tp_id" not in order_tracker[symbol]["short"]: order_tracker[symbol]["short"]["tp_id"] = None


             if symbol not in last_trade_time:
                  last_trade_time[symbol] = 0.0 # Use float 0 for time
             if symbol not in consecutive_losses:
                  consecutive_losses[symbol] = 0

        logger.info(f"Initial order tracker state: {order_tracker}")

    except FileNotFoundError:
        logger.debug(f"No state file '{STATE_FILE}' found, starting fresh.")
        # Initialize empty states for all symbols when starting fresh
        for symbol in CONFIG.symbols:
             order_tracker[symbol] = {"long": {"sl_id": None, "tsl_id": None, "tp_id": None}, "short": {"sl_id": None, "tsl_id": None, "tp_id": None}} # Added tp_id
             last_trade_time[symbol] = 0.0
             consecutive_losses[symbol] = 0
        logger.info(f"Initialized order tracker state: {order_tracker}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from state file '{STATE_FILE}'. File might be corrupted. Starting fresh.")
        # Initialize empty states when JSON is invalid
        order_tracker = {}
        last_trade_time = {}
        consecutive_losses = {}
        for symbol in CONFIG.symbols:
             order_tracker[symbol] = {"long": {"sl_id": None, "tsl_id": None, "tp_id": None}, "short": {"sl_id": None, "tsl_id": None, "tp_id": None}} # Added tp_id
             last_trade_time[symbol] = 0.0
             consecutive_losses[symbol] = 0
        logger.info(f"Initialized order tracker state after JSON error: {order_tracker}")

    except Exception as e:
        logger.error(f"An unexpected error occurred loading state from {STATE_FILE}: {e}. Starting fresh.", exc_info=True)
        # Initialize empty states on unexpected error
        order_tracker = {}
        last_trade_time = {}
        consecutive_losses = {}
        for symbol in CONFIG.symbols:
             order_tracker[symbol] = {"long": {"sl_id": None, "tsl_id": None, "tp_id": None}, "short": {"sl_id": None, "tsl_id": None, "tp_id": None}} # Added tp_id
             last_trade_time[symbol] = 0.0
             consecutive_losses[symbol] = 0
        logger.info(f"Initialized order tracker state after unexpected load error: {order_tracker}")
# </xaiArtifact>


# <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
def send_email(subject: str, body: str) -> None:
    """Sends a notification via email using configured SMTP settings."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT", "587")
    smtp_sender = os.getenv("SMTP_SENDER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    alert_recipient = os.getenv("ALERT_EMAIL")

    if not all([smtp_server, smtp_sender, smtp_password, alert_recipient]):
        logger.warning(Fore.YELLOW + "SMTP configuration incomplete (SMTP_SERVER, SMTP_SENDER, SMTP_PASSWORD, or ALERT_EMAIL missing). Skipping email notification.")
        return

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        logger.error(Fore.RED + f"Invalid SMTP_PORT configured: '{smtp_port_str}'. Skipping email.")
        return

    msg = MIMEText(body)
    msg['Subject'] = f"Pyrmethus Bot Alert - {subject}"
    msg['From'] = smtp_sender
    msg['To'] = alert_recipient

    try:
        logger.debug(f"Attempting to send email to {alert_recipient}...")
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(smtp_sender, smtp_password)
            server.send_message(msg)
        logger.info(Fore.GREEN + "Email notification sent successfully.")
    except Exception as e:
        logger.error(Fore.RED + f"Failed to send email notification: {e}", exc_info=True)
# </xaiArtifact>


# <xaiArtifact artifact_id="47586970-0123-4567-89ab-cdef01234567" title="Improve Error Handling for Decimal Conversions" contentType="python">Snippet 7: Decimal Conversions</xaiArtifact>
def safe_decimal(value: Any, default: Decimal = Decimal('0')) -> Decimal:
    """Safely converts value to Decimal with fallback."""
    if value is None:
        return default
    try:
        # Convert via string to preserve precision
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, DecimalException) as e: # Added DecimalException
        logger.warning(f"Failed to convert {value} (type {type(value).__name__}) to Decimal. Using default: {default}. Error: {e}")
        return default
    except Exception as e:
         logger.error(f"Unexpected error converting {value} to Decimal: {e}", exc_info=True)
         return default
# </xaiArtifact>


# <xaiArtifact artifact_id="08d15550-8b52-440d-a39c-565e19f2d93d" title="Implement Health Check Endpoint" contentType="python">Snippet 14: Health Check Endpoint</xaiArtifact>
class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Pyrmethus Bot Healthy")
    # Suppress logging from the HTTP server itself
    def log_message(self, format, *args):
        pass # Do nothing to avoid spamming logs

def start_health_server() -> None:
    """Starts a simple HTTP server in a separate thread for health checks."""
    try:
        # Use 0.0.0.0 to listen on all interfaces if needed, but localhost is safer default
        server_address = ('localhost', 8080)
        httpd = HTTPServer(server_address, HealthCheckHandler)
        # Run the server in a daemon thread so it doesn't prevent script exit
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        logger.info(Fore.GREEN + f"Health check server started on {server_address[0]}:{server_address[1]}")
    except Exception as e:
        logger.error(Fore.RED + f"Failed to start health check server: {e}", exc_info=True)
# </xaiArtifact>


# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available) via termux-toast."""
    if not os.getenv("TERMUX_VERSION"):
        # logger.debug("Not running in Termux environment. Skipping notification.") # Too noisy
        return

    try:
        # Check if command exists using which (more portable than 'command -v')
        # Added timeout for 'which' to prevent hangs
        check_cmd = subprocess.run(['which', 'termux-toast'], capture_output=True, text=True, check=False, timeout=3)
        if check_cmd.returncode != 0:
            logger.debug("termux-toast command not found. Skipping notification.")
            return

        # Basic sanitization - focus on preventing shell interpretation issues
        # Replace potentially problematic characters with spaces or remove them
        safe_title = title.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('(', '').replace(')', '').strip()
        safe_content = content.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '').replace('&', '').replace('|', '').replace('(', '').replace(')', '').strip()

        if not safe_title and not safe_content:
            logger.debug("Notification content is empty after sanitization. Skipping.")
            return

        # Limit length to avoid potential buffer issues or overly long toasts
        max_len = 250  # Increased length
        full_message = f"{safe_title}: {safe_content}" if safe_title and safe_content else safe_title if safe_title else safe_content
        full_message = full_message[:max_len].strip()  # Trim and strip whitespace

        if not full_message:
            logger.debug("Notification message is empty after combining and trimming. Skipping.")
            return

        # Use list format for subprocess.run for security
        # Example styling: gravity middle, black text on green background, long duration for detailed messages
        cmd_list = ['termux-toast', '-g', 'middle', '-c', 'black', '-b', 'green', '-s', 'long', full_message]
        # Ensure cmd_list contains only strings
        cmd_list = [str(arg) for arg in cmd_list]

        # Run the command non-blocking or with a short timeout
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=5)  # Add timeout

        if result.returncode != 0:
            # Log stderr if available
            stderr_msg = result.stderr.strip()
            logger.warning(f"termux-toast command failed with code {result.returncode}" + (f": {stderr_msg}" if stderr_msg else ""))
        # No else needed, success is silent

    except FileNotFoundError:
        logger.debug("termux-toast command not found (FileNotFoundError). Skipping notification.")
    except subprocess.TimeoutExpired:
        logger.warning("termux-toast command timed out. Skipping notification.")
    except Exception as e:
        # Catch other potential exceptions during subprocess execution
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}", exc_info=True)

# --- Precision Casting Spells ---

# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# Modified to accept symbol argument and use MARKET_INFOS
def get_market_info(symbol: str) -> Optional[Dict]:
    """Safely retrieve market info for a given symbol."""
    global MARKET_INFOS
    market_info = MARKET_INFOS.get(symbol)
    if market_info is None:
        logger.error(f"{Fore.RED}Market info not found for symbol {symbol}.")
    return market_info

def format_price(symbol: str, price: Union[Decimal, str, float, int]) -> str:
    """Formats price according to market precision rules using exchange's method."""
    global EXCHANGE
    market_info = get_market_info(symbol)
    if market_info is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format price.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            price_dec = safe_decimal(price) # Use safe_decimal
            if price_dec.is_nan(): return str(price) # Return original if cannot convert
            # Use enough decimal places for a fallback, maybe based on typical crypto prices
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))  # Quantize to 8 decimal places
        except Exception:
            logger.error(f"Fallback price formatting failed for {price}.")
            return str(price)  # Last resort

    # Ensure input is Decimal first for internal consistency, then float for CCXT methods
    price_dec = safe_decimal(price) # Use safe_decimal
    if price_dec.is_nan():
         logger.error(f"Cannot format price '{price}': Invalid number format after safe_decimal.")
         return str(price)  # Return original input if cannot even convert to Decimal

    try:
        # CCXT's price_to_precision handles rounding/truncation based on market rules (tick size).
        # Ensure input is float as expected by CCXT methods.
        price_float = float(price_dec)
        return EXCHANGE.price_to_precision(symbol, price_float)
    except (AttributeError, KeyError, InvalidOperation) as e:
        logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid price format: {e}. Using fallback formatting.")
        try:
            # Fallback with Decimal quantize based on a reasonable number of decimal places
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))
        except Exception:
            logger.error(f"Fallback price formatting failed for {price_dec}.")
            return str(price_dec)  # Last resort (Decimal as string)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price_dec} for {symbol}: {e}. Using fallback.")
        try:
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))
        except Exception:
            logger.error(f"Fallback price formatting failed for {price_dec}.")
            return str(price_dec)

def format_amount(symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
    """Formats amount according to market precision rules (step size) using exchange's method."""
    global EXCHANGE
    market_info = get_market_info(symbol)
    if market_info is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format amount.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            amount_dec = safe_decimal(amount) # Use safe_decimal
            if amount_dec.is_nan(): return str(amount) # Return original if cannot convert
            # Use quantize for fallback if Decimal input (e.g., 8 decimal places)
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            logger.error(f"Fallback amount formatting failed for {amount}.")
            return str(amount)  # Last resort

    # Ensure input is Decimal first for internal consistency, then float for CCXT methods
    amount_dec = safe_decimal(amount) # Use safe_decimal
    if amount_dec.is_nan():
         logger.error(f"Cannot format amount '{amount}': Invalid number format after safe_decimal.")
         return str(amount)  # Return original input if cannot even convert to Decimal


    try:
        # CCXT's amount_to_precision handles step size and rounding.
        # Map Python Decimal rounding modes to CCXT rounding modes if needed.
        # CCXT primarily supports ROUND (nearest) and TRUNCATE (down).
        ccxt_rounding_mode = ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND  # Basic mapping
        # Ensure input is float as expected by CCXT methods.
        amount_float = float(amount_dec)
        return EXCHANGE.amount_to_precision(symbol, amount_float, rounding_mode=ccxt_rounding_mode)
    except (AttributeError, KeyError, InvalidOperation) as e:
        logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid amount format: {e}. Using fallback formatting.")
        try:
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            logger.error(f"Fallback amount formatting failed for {amount_dec}.")
            return str(amount_dec)  # Last resort (Decimal as string)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount_dec} for {symbol}: {e}. Using fallback.")
        try:
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            logger.error(f"Fallback amount formatting failed for {amount_dec}.")
            return str(amount_dec)
# </xaiArtifact>

# --- Core Spell Functions ---

def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    """Generic wrapper to fetch data with retries and exponential backoff."""
    global EXCHANGE
    if EXCHANGE is None:
        logger.critical("Exchange object is None, cannot fetch data.")
        return None  # Indicate critical failure

    last_exception = None
    # Add category param automatically for V5 if not already present in kwargs['params']
    # Only attempt this if EXCHANGE and its options structure exist
    # This logic needs to be symbol-aware if category differs by symbol (it shouldn't for V5 unified)
    # The category is set in the config, so use that.
    if 'params' not in kwargs:
        kwargs['params'] = {}
    # Check if category is *already* in params before adding
    if 'category' not in kwargs['params']:
        kwargs['params']['category'] = CONFIG.market_type # Use category from config
        # logger.debug(f"Auto-added category '{kwargs['params']['category']}' to params for {fetch_function.__name__}")


    for attempt in range(CONFIG.max_fetch_retries + 1):  # +1 to allow logging final failure
        try:
            # Log the attempt number and function being called at DEBUG level
            # Be cautious not to log sensitive parameters like API keys if they were somehow passed directly
            log_kwargs = {k: ('****' if 'secret' in str(k).lower() or 'key' in str(k).lower() else v) for k, v in kwargs.items()}
            log_args = tuple('****' if isinstance(arg, str) and ('secret' in arg.lower() or 'key' in arg.lower()) else arg for arg in args)  # Basic sanitization for args too
            # Limit logging of large data args like OHLCV lists
            log_args_repr = [repr(a)[:100] + '...' if isinstance(a, list) and len(repr(a)) > 100 else repr(a) for a in log_args]
            log_kwargs_repr = {k: (repr(v)[:100] + '...' if isinstance(v, (list, dict)) and len(repr(v)) > 100 else repr(v)) for k, v in log_kwargs.items()}

            logger.debug(f"Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}: Calling {fetch_function.__name__} with args=[{', '.join(log_args_repr)}], kwargs={log_kwargs_repr}")
            result = fetch_function(*args, **kwargs)
            # <xaiArtifact artifact_id="8a9b0c1d-2e3f-4a5b-6c7d-8e9f0a1b2c3d" title="Validate API Rate Limits" contentType="python">Snippet 5: API Rate Limits</xaiArtifact>
            # Check response for rate limit headers if available (Bybit might not always provide these explicitly in unified responses)
            # CCXT's enableRateLimit=True handles most cases internally, but explicit check is defensive.
            # CCXT usually puts rate limit info in the 'info' dict of the response.
            if isinstance(result, dict) and 'info' in result and isinstance(result['info'], dict):
                # Bybit V5 rate limit info is often in headers, which CCXT might expose via 'info'
                # e.g., response.headers.get('x-rate-limit-remaining')
                # CCXT response['info'] might contain parsed headers or specific rate limit fields.
                # Checking a common pattern like 'RateLimit-Remaining' in the response info is a guess based on snippet.
                # A more reliable check would be to examine raw headers or CCXT's internal state if exposed.
                # Sticking to the snippet's check pattern:
                rate_limit_remaining = safe_decimal(result['info'].get('RateLimit-Remaining', result['info'].get('X-RateLimit-Remaining'))) # Try both keys
                if not rate_limit_remaining.is_nan() and rate_limit_remaining < 10:
                    logger.warning(f"{Fore.YELLOW}API Rate Limit Low: {rate_limit_remaining.normalize()} requests remaining. Adding a small delay.")
                    time.sleep(2) # Add a small delay to avoid hitting limit hard
            # </xaiArtifact>
            return result  # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = 2 ** attempt  # Exponential backoff (1, 2, 4, 8...)
            logger.warning(Fore.YELLOW + f"{fetch_function.__name__}: Network issue (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}). Retrying in {wait_time}s... Error: {e}")
            if attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            else:
                logger.error(Fore.RED + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to network issues.")
                break # Exit retry loop
        except ccxt.ExchangeNotAvailable as e:
            last_exception = e
            logger.error(Fore.RED + f"{fetch_function.__name__}: Exchange not available: {e}. Stopping retries.")
            return None # Indicate hard failure
        except ccxt.AuthenticationError as e:
            last_exception = e
            logger.critical(Fore.RED + Style.BRIGHT + f"{fetch_function.__name__}: Authentication error: {e}. Halting script.")
            send_email("Bot Critical Error", f"{fetch_function.__name__}: Authentication error: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
            sys.exit(1)  # Exit immediately on auth failure
        except (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied) as e:
            # These are typically non-retryable errors related to the request parameters or exchange state.
            last_exception = e
            error_type = type(e).__name__
            logger.error(Fore.RED + f"{fetch_function.__name__}: Non-retryable error ({error_type}): {e}. Stopping retries for this call.")
            # Re-raise these specific errors so the caller can handle them appropriately
            raise e
        except ccxt.ExchangeError as e:
            # Includes rate limit errors, potentially invalid requests etc.
            last_exception = e
            # Check for specific retryable Bybit error codes if needed (e.g., 10006=timeout, 10016=internal error)
            # Bybit V5 Rate limit codes: 10018 (IP), 10017 (Key), 10009 (Frequency)
            # Bybit V5 Invalid Parameter codes often start with 11xxxx.
            # General internal errors: 10006, 10016, 10020, 10030
            # Other: 30034 (Position status not normal), 110025 (SL/TP order not found or completed)
            # Check for 'info' dict and 'retCode' for Bybit V5 errors
            error_info = getattr(e, 'info', {}) # CCXT might store raw response info here
            error_code = error_info.get('retCode', getattr(e, 'code', None)) # Prioritize V5 retCode
            error_message = str(e)
            should_retry = True
            wait_time = 2 * (attempt + 1)  # Default backoff

            # Check for common rate limit patterns / codes
            if "Rate limit exceeded" in error_message or error_code in [10017, 10018, 10009]:
                wait_time = 5 * (attempt + 1)  # Longer wait for rate limits
                logger.warning(f"{Fore.YELLOW}{fetch_function.__name__}: Rate limit hit (Code: {error_code}). Retrying in {wait_time}s... Error: {e}")
            # Check for specific non-retryable errors (e.g., invalid parameter codes, insufficient funds codes)
            # These often start with 11xxxx for Bybit V5
            elif error_code is not None and (110000 <= int(safe_decimal(error_code, default=Decimal('110000'))) <= 110100 or error_code in [30034]):
                 logger.error(Fore.RED + f"{fetch_function.__name__}: Non-retryable parameter/logic/state exchange error (Code: {error_code}): {e}. Stopping retries.")
                 should_retry = False
            else:
                # General exchange error, apply default backoff
                logger.warning(f"{Fore.YELLOW}{fetch_function.__name__}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}, Code: {error_code}). Retrying in {wait_time}s... Error: {e}")

            if should_retry and attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            elif should_retry:  # Final attempt failed
                logger.error(Fore.RED + f"{fetch_function.__name__}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to exchange errors.")
                break  # Exit retry loop
            else:  # Non-retryable error encountered
                break  # Exit retry loop

        except Exception as e:
            # Catch-all for unexpected errors
            last_exception = e
            logger.error(Fore.RED + f"{fetch_function.__name__}: Unexpected shadow encountered: {e}", exc_info=True)
            send_email("Bot Unexpected Error", f"{fetch_function.__name__} failed: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
            break  # Stop on unexpected errors

    # If loop finished without returning, it means all retries failed or a break occurred
    # Re-raise the last specific non-retryable exception if it wasn't already (e.g. OrderNotFound, InsufficientFunds)
    if isinstance(last_exception, (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied)):
        raise last_exception  # Propagate specific non-retryable errors

    # For other failures (Network, ExchangeNotAvailable, general ExchangeError after retries), return None
    if last_exception:
        logger.error(f"{fetch_function.__name__} ultimately failed after {CONFIG.max_fetch_retries + 1} attempts or encountered a non-retryable error type not explicitly re-raised.")
        return None  # Indicate failure

    # Should not reach here if successful, but defensive return None
    # If we reached the end of the loop without returning or raising, it implies failure.
    # The function should only return the result on success.
    # So if we are here, it's a failure case.
    return None


# <xaiArtifact artifact_id="061781c2-1141-436d-a49a-47006d3c4902" title="Optimize OHLCV Fetch with Cache" contentType="python">Snippet 11: OHLCV Cache</xaiArtifact>
# OHLCV Cache (per symbol, timeframe, limit)
# Max 10 cache entries (e.g., 10 symbols x 1 timeframe)
# TTL 60 seconds (adjust based on interval and loop sleep)
ohlcv_cache = TTLCache(maxsize=10, ttl=60)

# Rename the original fetch_market_data logic
def _fetch_market_data_original(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Original function to fetch OHLCV data without caching."""
    global EXCHANGE
    logger.info(Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})...")

    if EXCHANGE is None or not hasattr(EXCHANGE, 'fetch_ohlcv'):
        logger.error(Fore.RED + "Exchange object not properly initialized or missing fetch_ohlcv.")
        return None

    # Ensure limit is positive (already validated in config, but double check)
    if limit <= 0:
        logger.error(f"Invalid OHLCV limit requested: {limit}. Using default 100.")
        limit = 100

    ohlcv_data = None
    try:
        # fetch_with_retries handles category param automatically
        ohlcv_data = fetch_with_retries(EXCHANGE.fetch_ohlcv, symbol, timeframe, limit=limit)
    except Exception as e:
        # fetch_with_retries should handle most errors, but catch any unexpected ones here
        logger.error(Fore.RED + f"Unhandled exception during fetch_ohlcv call via fetch_with_retries: {e}", exc_info=True)
        return None

    if ohlcv_data is None:
        # fetch_with_retries already logged the failure reason
        logger.error(Fore.RED + f"Failed to fetch OHLCV data for {symbol}.")
        return None
    if not isinstance(ohlcv_data, list) or not ohlcv_data:
        logger.error(Fore.RED + f"Received empty or invalid OHLCV data type: {type(ohlcv_data)}. Content: {str(ohlcv_data)[:100]}")
        return None

    try:
        # Use Decimal for numeric columns directly during DataFrame creation where possible
        # However, pandas expects floats for most calculations, so convert to float for calculations
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp immediately to UTC datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True)  # Drop rows where timestamp conversion failed

        # Convert numeric columns to float first for pandas/numpy compatibility
        for col in ["open", "high", "low", "close", "volume"]:
            # Ensure column exists before attempting conversion
            if col in df.columns:
                # Use errors='coerce' to turn unparseable values into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.warning(f"OHLCV data is missing expected column: {col}")

        # Check for NaNs in critical price columns *after* conversion
        initial_len = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len:
            dropped_count = initial_len - len(df)
            logger.warning(f"Dropped {dropped_count} rows with missing essential price data from OHLCV for {symbol}.")

        if df.empty:
            logger.error(Fore.RED + f"DataFrame is empty after processing OHLCV data for {symbol} (all rows dropped?).")
            return None

        df = df.set_index("timestamp")
        # Ensure data is sorted chronologically (fetch_ohlcv usually guarantees this, but verify)
        if not df.index.is_monotonic_increasing:
            logger.warning(f"OHLCV data for {symbol} was not sorted chronologically. Sorting now.")
            df.sort_index(inplace=True)

        # Check for duplicate timestamps (can indicate data issues)
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].unique()
            logger.warning(Fore.YELLOW + f"Duplicate timestamps found in OHLCV data for {symbol} ({len(duplicates)} unique duplicates). Keeping last entry for each.")
            df = df[~df.index.duplicated(keep='last')]

        # Check time difference between last two candles vs expected interval
        if len(df) > 1:
            time_diff = df.index.to_series().diff().iloc[-1] # Calculate difference between last two timestamps
            try:
                # Use pandas to parse timeframe string robustly
                expected_interval_ms = EXCHANGE.parse_timeframe(timeframe) * 1000  # Convert to milliseconds
                expected_interval_td = pd.Timedelta(expected_interval_ms, unit='ms')
                # Allow some tolerance (e.g., 20% of interval + 10 seconds) for minor timing differences/API lag
                tolerance = expected_interval_td * 0.2 + pd.Timedelta(seconds=10)
                if abs(time_diff.total_seconds() - expected_interval_td.total_seconds()) > tolerance.total_seconds():
                    logger.warning(f"[{symbol}] Unexpected large time gap between last two candles: {time_diff} (expected ~{expected_interval_td}). Difference: {abs(time_diff.total_seconds() - expected_interval_td.total_seconds()):.2f}s")
            except ValueError:
                logger.warning(f"[{symbol}] Could not parse timeframe '{timeframe}' to calculate expected interval for time gap check.")
            except Exception as time_check_e:
                logger.warning(f"[{symbol}] Error during time difference check: {time_check_e}")

        logger.info(Fore.GREEN + f"[{symbol}] Market whispers received ({len(df)} candles). Latest: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return df
    except Exception as e:
        logger.error(Fore.RED + f"[{symbol}] Error processing OHLCV data into DataFrame: {e}", exc_info=True)
        return None

def fetch_market_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data using the retry wrapper and perform validation, with caching."""
    cache_key = (symbol, timeframe, limit)
    if cache_key in ohlcv_cache:
        logger.debug(f"[{symbol}] Returning cached OHLCV data.")
        return ohlcv_cache[cache_key]

    # Call the original fetch function
    df = _fetch_market_data_original(symbol, timeframe, limit)

    if df is not None and not df.empty:
        ohlcv_cache[cache_key] = df
        logger.debug(f"[{symbol}] Cached OHLCV data.")
    return df
# </xaiArtifact>


# --- ECC Calculation (from Block 1, adapted) ---
def calculate_ehlers_cyber_cycle(prices: np.ndarray, alpha: Decimal, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Ehlers Cyber Cycle and its trigger line.

    Args:
        prices: Numpy array of closing prices (float).
        alpha: Smoothing factor (Decimal).
        lookback: Number of bars for cycle calculation (int).

    Returns:
        Tuple of (cyber_cycle, trigger_line) arrays (float).
        Arrays will contain NaNs for initial bars where calculation is not possible.
    """
    # Ensure enough data for calculation based on lookback + initial smoothing
    # The formula needs prices[i-3], smooth[i], smooth[i-1], smooth[i-2], cycle[i-1], cycle[i-2]
    # The smooth calculation needs 4 prior points (i, i-1, i-2, i-3). First valid smooth at index 3.
    # The cycle calculation needs smooth[i-2] and cycle[i-2]. It starts from index 4.
    # Minimum length to get *any* cycle value is 5.
    # A safe minimum length is roughly `lookback + initial points`. Let's enforce min 15 or lookback+5, whichever is larger.
    min_safe_len = max(lookback + 5, 15)

    if len(prices) < min_safe_len:
        logger.warning(f"Insufficient data ({len(prices)}) for ECC calculation (minimum safe: {min_safe_len}). Returning arrays with NaNs.")
        # Return arrays of NaNs matching input length if insufficient data
        return np.full(len(prices), np.nan), np.full(len(prices), np.nan)

    try:
        # Convert alpha to float for numpy calculation
        alpha_float = float(alpha)
        alpha_sq = (1 - 0.5 * alpha_float) ** 2
        alpha_minus_1 = 2 * (1 - alpha_float)
        alpha_minus_2 = (1 - alpha_float) ** 2

        # Initialize arrays with NaNs to reflect warm-up period
        smooth = np.full(len(prices), np.nan, dtype=float)
        cycle = np.full(len(prices), np.nan, dtype=float)
        trigger = np.full(len(prices), np.nan, dtype=float)

        # Smooth the price series (from original script logic)
        # This smoothing uses a fixed 4-bar window (i, i-1, i-2, i-3)
        # Loop starts from index 3 to ensure prices[i-3] is valid (0)
        for i in range(3, len(prices)):
            # Add a small epsilon to divisor to prevent DivisionByZero if prices are identical
            smooth[i] = (prices[i] + 2 * prices[i-1] + 2 * prices[i-2] + prices[i-3]) / 6.0 # Use float 6.0

        # Calculate Cyber Cycle (from original script logic)
        # Loop starts from index 4 to ensure smooth[i-2] and cycle[i-2] are valid
        # Initialize first two cycle values (index 2 and 3) to zero or small values if needed by formula?
        # Ehlers' original code initializes cycle[2] and cycle[3] to zero. Let's follow that.
        cycle[2] = 0.0
        cycle[3] = 0.0

        for i in range(4, len(prices)):
            # Use the pre-calculated alpha terms
            # Ensure smooth values are not NaN before using
            if not np.isnan(smooth[i]) and not np.isnan(smooth[i-1]) and not np.isnan(smooth[i-2]) and \
               not np.isnan(cycle[i-1]) and not np.isnan(cycle[i-2]): # Check dependent cycle values too
                 cycle[i] = alpha_sq * (smooth[i] - 2 * smooth[i-1] + smooth[i-2]) + alpha_minus_1 * cycle[i-1] - alpha_minus_2 * cycle[i-2]
            # trigger calculation needs cycle[i-1]
            if not np.isnan(cycle[i-1]):
                 trigger[i] = cycle[i-1]

        # Return the calculated values. NaNs will be present at the beginning where calculation wasn't possible.
        return cycle, trigger

    except Exception as e:
        logger.error(f"{Fore.RED}Error during Ehlers Cyber Cycle calculation: {e}", exc_info=True)
        # Return arrays of NaNs matching input length on calculation error
        return np.full(len(prices), np.nan), np.full(len(prices), np.nan)


def calculate_indicators(symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Decimal]]:
    """Calculate technical indicators for a symbol, returning results as Decimals for precision."""
    logger.info(Fore.CYAN + f"[{symbol}] # Weaving indicator patterns...")
    if df is None or df.empty:
        logger.error(Fore.RED + f"[{symbol}] Cannot calculate indicators on missing or empty DataFrame.")
        return None
    try:
        # Ensure data is float for TA-Lib / Pandas calculations, convert to Decimal at the end
        # Defensive: Make a copy to avoid modifying the original DataFrame if it's used elsewhere
        df_calc = df.copy()
        # Ensure necessary columns exist before accessing
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df_calc.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_calc.columns]
            logger.error(f"{Fore.RED}[{symbol}] DataFrame is missing required columns for indicator calculation: {missing}")
            return None

        # Convert columns to float, coercing errors to NaN
        close = pd.to_numeric(df_calc["close"], errors='coerce').astype(float)
        high = pd.to_numeric(df_calc["high"], errors='coerce').astype(float)
        low = pd.to_numeric(df_calc["low"], errors='coerce').astype(float)
        # volume = pd.to_numeric(df_calc["volume"], errors='coerce').astype(float) # Volume not used in indicators yet

        # Drop rows where critical price data became NaN after conversion
        initial_len_after_convert = len(df_calc)
        df_calc = df_calc[close.notna() & high.notna() & low.notna()]
        close = close[df_calc.index] # Re-align series with cleaned df index
        high = high[df_calc.index]
        low = low[df_calc.index]

        if len(df_calc) < initial_len_after_convert:
             dropped_count = initial_len_after_convert - len(df_calc)
             logger.warning(f"[{symbol}] Dropped {dropped_count} rows with NaN price data after initial conversion.")

        if df_calc.empty:
            logger.error(Fore.RED + f"[{symbol}] DataFrame is empty after cleaning NaN prices.")
            return None

        current_data_length = len(df_calc)

        # --- Calculate ATR ---
        # ATR needs `period + 1` data points (for the first TR calculation involving previous close).
        min_required_atr = CONFIG.atr_period + 1
        atr_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)  # Initialize with NaNs
        if current_data_length < min_required_atr:
            logger.warning(f"{Fore.YELLOW}[{symbol}] Insufficient data ({current_data_length}) for ATR calculation (minimum required: {min_required_atr}). ATR will be NaN.")
        else:
            # Average True Range (ATR) - Wilder's smoothing matches TradingView standard
            tr_df = pd.DataFrame(index=df_calc.index)
            tr_df["hl"] = high - low
            tr_df["hc"] = (high - close.shift(1)).abs() # Shift by 1 for previous close
            tr_df["lc"] = (low - close.shift(1)).abs()
            tr = tr_df[["hl", "hc", "lc"]].max(axis=1)
            # Use ewm with alpha = 1/period for Wilder's smoothing, adjust=False
            atr_series = tr.ewm(alpha=1/float(CONFIG.atr_period), adjust=False).mean()


        # --- Calculate Strategy Specific Indicators ---
        ecc_cycle_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index) # Initialize with NaNs
        ecc_trigger_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index) # Initialize with NaNs
        fast_ema_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        slow_ema_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        trend_ema_series = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        stoch_k = pd.Series([np.nan] * len(df_calc), index=df_calc.index)
        stoch_d = pd.Series([np.nan] * len(df_calc), index=df_calc.index)

        # Trend EMA is used by both strategies for filtering
        min_required_trend_ema = CONFIG.trend_ema_period
        if current_data_length < min_required_trend_ema:
             logger.warning(f"{Fore.YELLOW}[{symbol}] Insufficient data ({current_data_length}) for Trend EMA calculation (minimum required: {min_required_trend_ema}). Trend EMA will be NaN.")
        else:
             # Use adjust=False for EWMA to match standard EMA calculation
             trend_ema_series = close.ewm(span=CONFIG.trend_ema_period, adjust=False).mean()


        if CONFIG.strategy == 'ecc_scalp':
            # ECC calculation requires numpy array of prices
            prices_np = close.values
            # ECC needs enough data based on its internal formula structure (min 5) and lookback
            min_required_ecc = CONFIG.ecc_lookback + 5 # Need min 5 for calculation to start
            if current_data_length < min_required_ecc:
                 logger.warning(f"{Fore.YELLOW}[{symbol}] Insufficient data ({current_data_length}) for ECC calculation (minimum required: {min_required_ecc}). ECC will be NaN.")
            else:
                 ecc_cycle_np, ecc_trigger_np = calculate_ehlers_cyber_cycle(prices_np, CONFIG.ecc_alpha, CONFIG.ecc_lookback)
                 # Convert numpy arrays back to pandas Series for consistent handling
                 # Align the calculated values with the end of the DataFrame
                 if len(ecc_cycle_np) > 0:
                      # Use iloc[-len(ecc_cycle_np):] to get the correct index slice from the end of the DataFrame
                      ecc_cycle_series = pd.Series(ecc_cycle_np, index=df_calc.index.iloc[-len(ecc_cycle_np):])
                 if len(ecc_trigger_np) > 0:
                      ecc_trigger_series = pd.Series(ecc_trigger_np, index=df_calc.index.iloc[-len(ecc_trigger_np):])


        elif CONFIG.strategy == 'ema_stoch':
            # EMA calculation
            min_required_ema = max(CONFIG.ema_fast_period, CONFIG.ema_slow_period)
            if current_data_length < min_required_ema:
                logger.warning(f"{Fore.YELLOW}[{symbol}] Insufficient data ({current_data_length}) for Fast/Slow EMA calculation (minimum required: {min_required_ema}). EMAs will be NaN.")
            else:
                 fast_ema_series = close.ewm(span=CONFIG.ema_fast_period, adjust=False).mean()
                 slow_ema_series = close.ewm(span=CONFIG.ema_slow_period, adjust=False).mean()

            # Stochastic Oscillator %K and %D
            # Ensure enough data for rolling window calculation
            min_required_stoch = CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 2
            if current_data_length < min_required_stoch:
                 logger.warning(f"{Fore.YELLOW}[{symbol}] Insufficient data ({current_data_length}) for Stochastic calculation (minimum required: {min_required_stoch}). Stoch will be NaN.")
            elif current_data_length >= CONFIG.stoch_period: # Period check for %K base
                low_min = low.rolling(window=CONFIG.stoch_period).min()
                high_max = high.rolling(window=CONFIG.stoch_period).max()
                # Add epsilon to prevent division by zero if high == low over the period
                # Use float epsilon for float calculation
                # Ensure low_min and high_max are not all NaN before division
                stoch_k_raw = pd.Series([np.nan] * len(df_calc), index=df_calc.index) # Initialize with NaN
                valid_indices = (high_max - low_min).notna() & ((high_max - low_min).abs() > 1e-9) # Avoid division by zero
                stoch_k_raw[valid_indices] = 100 * (close[valid_indices] - low_min[valid_indices]) / (high_max[valid_indices] - low_min[valid_indices])
                # Ensure enough data for smoothing windows
                if current_data_length >= CONFIG.stoch_period + CONFIG.stoch_smooth_k - 1: # Smooth K check
                     stoch_k = stoch_k_raw.rolling(window=CONFIG.stoch_smooth_k).mean()
                     if current_data_length >= CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 2: # Smooth D check
                          stoch_d = stoch_k.rolling(window=CONFIG.stoch_smooth_d).mean()
                     else:
                          logger.warning(f"{Fore.YELLOW}[{symbol}] Not enough data for Stochastic %D calculation.")
                 else:
                      logger.warning(f"{Fore.YELLOW}[{symbol}] Not enough data for Stochastic %K smoothing.")
            else:
                 logger.warning(f"{Fore.YELLOW}[{symbol}] Not enough data for Stochastic period calculation.")


        # --- Extract Latest Values & Convert to Decimal ---
        # Define quantizers for consistent decimal places (adjust as needed)
        # These are for *internal* Decimal representation, not API formatting.
        # Use enough precision to avoid rounding errors before API formatting.
        price_quantizer = Decimal("1E-8")  # 8 decimal places for price-like values
        percent_quantizer = Decimal("1E-4")  # 4 decimal places for Stoch, ECC percentages
        atr_quantizer = Decimal("1E-8")  # 8 decimal places for ATR

        # Helper to safely get latest non-NaN value, convert to Decimal, and handle errors
        def get_latest_decimal(series: Union[pd.Series, np.ndarray], quantizer: Decimal, name: str, default_val: Decimal = Decimal("NaN")) -> Decimal:
            # Handle numpy arrays passed directly
            if isinstance(series, np.ndarray):
                 # Convert to Series for consistent handling of index/dropna/iloc
                 if len(series) > 0:
                      series = pd.Series(series)
                 else:
                      # logger.debug(f"Indicator series '{name}' (numpy) is empty.") # Too noisy
                      return default_val

            if series.empty or series.isna().all():
                # logger.debug(f"Indicator series '{name}' is empty or all NaN.") # Too noisy
                return default_val
            # Get the last valid (non-NaN) value
            latest_valid_val = series.dropna().iloc[-1] if not series.dropna().empty else None

            if latest_valid_val is None or pd.isna(latest_valid_val):
                # logger.debug(f"Indicator calculation for '{name}' resulted in NaN or only NaNs.") # Too noisy
                return default_val
            try:
                # Convert via string for precision, then quantize
                return Decimal(str(latest_valid_val)).quantize(quantizer, rounding=ROUND_DOWN)  # Use ROUND_DOWN for consistency
            except (InvalidOperation, TypeError, DecimalException) as e: # Added DecimalException
                logger.error(f"[{symbol}] Could not convert indicator '{name}' value {latest_valid_val} to Decimal: {e}. Returning default.")
                return default_val
            except Exception as e:
                 logger.error(f"[{symbol}] Unexpected error converting indicator '{name}' value {latest_valid_val} to Decimal: {e}", exc_info=True)
                 return default_val


        indicators_out = {
            "atr": get_latest_decimal(atr_series, atr_quantizer, "atr", default_val=Decimal("0.0")),  # Default zero, but check is_nan before using for calc
            "atr_period": CONFIG.atr_period # Store period for display
        }

        # Add strategy specific indicators
        indicators_out["trend_ema"] = get_latest_decimal(trend_ema_series, price_quantizer, "trend_ema") # Trend filter EMA
        indicators_out["trend_ema_period"] = CONFIG.trend_ema_period # Store period for display


        if CONFIG.strategy == 'ecc_scalp':
             indicators_out["ecc_cycle"] = get_latest_decimal(ecc_cycle_series, percent_quantizer, "ecc_cycle")
             indicators_out["ecc_trigger"] = get_latest_decimal(ecc_trigger_series, percent_quantizer, "ecc_trigger")
             indicators_out["ecc_alpha"] = CONFIG.ecc_alpha # Store params for display
             indicators_out["ecc_lookback"] = CONFIG.ecc_lookback # Store params for display

        elif CONFIG.strategy == 'ema_stoch':
             indicators_out["fast_ema"] = get_latest_decimal(fast_ema_series, price_quantizer, "fast_ema")
             indicators_out["slow_ema"] = get_latest_decimal(slow_ema_series, price_quantizer, "slow_ema")
             indicators_out["stoch_k"] = get_latest_decimal(stoch_k, percent_quantizer, "stoch_k", default_val=Decimal("50.00"))  # Default neutral if NaN
             indicators_out["stoch_d"] = get_latest_decimal(stoch_d, percent_quantizer, "stoch_d", default_val=Decimal("50.00"))  # Default neutral if NaN
             indicators_out["ema_fast_period"] = CONFIG.ema_fast_period # Store params for display
             indicators_out["ema_slow_period"] = CONFIG.ema_slow_period # Store params for display
             indicators_out["stoch_period"] = CONFIG.stoch_period # Store params for display
             indicators_out["stoch_smooth_k"] = CONFIG.stoch_smooth_k # Store params for display
             indicators_out["stoch_smooth_d"] = CONFIG.stoch_smooth_d # Store params for display

        # Check if any crucial indicator calculation failed (returned NaN default)
        # This check is now done during signal generation/trade execution,
        # as we return the dict even if some values are NaN.
        # We only return None from calculate_indicators if the *initial* dataframe checks fail.

        logger.info(Fore.GREEN + f"[{symbol}] Indicator patterns woven successfully.")
        # logger.debug(f"[{symbol}] Latest Indicators: { {k: str(v) for k, v in indicators_out.items()} }") # Log values at debug, convert Decimal to str for clean log
        return indicators_out

    except Exception as e:
        logger.error(Fore.RED + f"[{symbol}] Failed to weave indicator patterns: {e}", exc_info=True)
        return None


# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# Modified to accept symbol argument and use MARKET_INFOS
def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch current positions for a symbol using retry wrapper, returning quantities and prices as Decimals."""
    global EXCHANGE, MARKET_INFOS
    logger.info(Fore.CYAN + f"[{symbol}] # Consulting position spirits...")

    market_info = get_market_info(symbol)
    if EXCHANGE is None or market_info is None:
        logger.error(f"[{symbol}] Exchange object or Market Info not available for fetching positions.")
        return None

    # Initialize with Decimal zero/NaN for clarity
    # Note: 'pnl' here refers to unrealized PnL from the position data
    pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN"), "open_time": None}, # Added open_time
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN"), "open_time": None} # Added open_time
    }

    positions_data = None
    try:
        # fetch_with_retries handles category param automatically
        # Use exchange's internal ID for symbol
        symbol_id = market_info.get('id')
        if symbol_id is None:
             logger.error(f"[{symbol}] Exchange symbol ID not available from Market Info. Cannot fetch positions.")
             return None
        positions_data = fetch_with_retries(EXCHANGE.fetch_positions, symbols=[symbol_id])
    except Exception as e:
        # Handle potential exceptions raised by fetch_with_retries itself (e.g., AuthenticationError, Non-retryable ExchangeError)
        logger.error(Fore.RED + f"[{symbol}] Unhandled exception during fetch_positions call via fetch_with_retries: {e}", exc_info=True)
        return None  # Indicate failure

    if positions_data is None:
        # fetch_with_retries already logged the failure reason
        logger.error(Fore.RED + f"[{symbol}] Failed to fetch positions.")
        return None  # Indicate failure

    if not isinstance(positions_data, list):
        logger.error(f"[{symbol}] Unexpected data type received from fetch_positions: {type(positions_data)}. Expected list. Data: {str(positions_data)[:200]}")
        return None

    if not positions_data:
        logger.info(Fore.BLUE + f"[{symbol}] No open positions reported by exchange.")
        return pos_dict  # Return the initialized zero dictionary

    # Process the fetched positions - find the primary long/short position for the symbol
    # In one-way mode, there should be at most one long and one short position per symbol.
    # Aggregate quantities if multiple entries exist for the same side (e.g., if both isolated/cross were returned, though unlikely with V5 category filter).
    aggregated_positions: Dict[str, Dict[str, Decimal]] = {
        "long": {"qty": Decimal("0.0"), "entry_price_sum_qty": Decimal("0.0")},  # Use weighted average sum for entry price
        "short": {"qty": Decimal("0.0"), "entry_price_sum_qty": Decimal("0.0")}
    }
    # Store other details from the *first* significant position found for each side
    other_pos_details: Dict[str, Dict[str, Any]] = { # Use Any for open_time
        "long": {"liq_price": Decimal("NaN"), "pnl": Decimal("NaN"), "open_time": None}, # Added open_time
        "short": {"liq_price": Decimal("NaN"), "pnl": Decimal("NaN"), "open_time": None} # Added open_time
    }
    first_long_found = False
    first_short_found = False

    for pos in positions_data:
        # Ensure pos is a dictionary
        if not isinstance(pos, dict):
            logger.warning(f"[{symbol}] Skipping non-dictionary item in positions data: {pos}")
            continue

        pos_symbol = pos.get('symbol')
        # Compare using CCXT symbol format
        if pos_symbol != symbol:
            logger.debug(f"[{symbol}] Ignoring position data for different symbol: {pos_symbol}")
            continue

        # Use info dictionary for safer access to raw exchange data if needed
        pos_info = pos.get('info', {})
        if not isinstance(pos_info, dict):  # Ensure info is a dict
            pos_info = {}

        # Determine side ('long' or 'short') - check unified field first, then 'info'
        side = pos.get("side")  # Unified field
        if side not in ["long", "short"]:
            # Fallback for Bybit V5 'info' field if unified 'side' is missing/invalid
            # Bybit V5 info side is "Buy" or "Sell"
            side_raw = pos_info.get("side", "").lower()
            if side_raw == "buy": side = "long"
            elif side_raw == "sell": side = "short"
            else:
                logger.warning(f"[{symbol}] Could not determine side for position: Info={str(pos_info)[:100]}. Skipping.")
                continue

        # Get quantity ('contracts' or 'size') - Use unified field first, fallback to info
        contracts_str = pos.get("contracts")  # Unified field ('contracts' seems standard)
        if contracts_str is None:
            contracts_str = pos_info.get("size")  # Common Bybit V5 field in 'info'

        # Get entry price - Use unified field first, fallback to info
        entry_price_str = pos.get("entryPrice")
        if entry_price_str is None:
            # Check 'avgPrice' (common in V5) or 'entryPrice' in info
            entry_price_str = pos_info.get("avgPrice", pos_info.get("entryPrice"))

        # Get Liq Price and PnL (these are less standardized, rely more on unified fields if available)
        liq_price_str = pos.get("liquidationPrice")
        if liq_price_str is None:
            liq_price_str = pos_info.get("liqPrice")

        pnl_str = pos.get("unrealizedPnl")
        if pnl_str is None:
            # Check Bybit specific info fields (V5 uses 'unrealisedPnl')
            pnl_str = pos_info.get("unrealisedPnl", pos_info.get("unrealizedPnl"))

        # <xaiArtifact artifact_id="1f2e3d4c-5b6a-7f8e-9d0c-1b2a3c4d5e6f" title="Add Position Age Timeout" contentType="python">Snippet 18: Position Age Timeout</xaiArtifact>
        # Get Open Time - Bybit V5 info.createTime (milliseconds)
        open_time_ms = pos_info.get('createTime')
        open_time_dt = None
        if open_time_ms is not None:
             try:
                  open_time_dt = pd.to_datetime(open_time_ms, unit='ms', utc=True)
             except (ValueError, TypeError):
                  logger.warning(f"[{symbol}] Could not parse open time '{open_time_ms}' for {side} position.")
        # </xaiArtifact>


        # --- Convert to Decimal and Aggregate/Store ---
        if side in aggregated_positions and contracts_str is not None:
            try:
                # Convert via string for precision
                contracts = safe_decimal(contracts_str) # Use safe_decimal

                # Use epsilon check for effectively zero positions
                if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                    logger.debug(f"[{symbol}] Ignoring effectively zero size {side} position entry (Qty: {contracts.normalize()}).")
                    continue  # Skip processing this entry

                # Aggregate quantity
                aggregated_positions[side]["qty"] += contracts

                # Aggregate for weighted average entry price if entry price is available
                entry_price = safe_decimal(entry_price_str, default=Decimal("NaN")) # Use safe_decimal with NaN default
                if not entry_price.is_nan():
                    # Weighted average: sum(entry_price * abs(qty)) / sum(abs(qty))
                    # Store sum(entry_price * abs(qty))
                    aggregated_positions[side]["entry_price_sum_qty"] += entry_price * contracts.copy_abs()
                else:
                     # If entry price is missing for a non-zero position entry, warn
                     logger.warning(f"[{symbol}] Missing entry price for non-zero position entry ({contracts.normalize()}) for {side} side.")


                # Store other details from the *first* significant entry found for this side
                # Note: Liq Price and PnL might differ across entries if account modes differ.
                # We just pick the first one found for display simplicity.
                # Also store the open_time from the first entry.
                if (side == "long" and not first_long_found) or (side == "short" and not first_short_found):
                    liq_price = safe_decimal(liq_price_str, default=Decimal("NaN")) # Use safe_decimal
                    pnl = safe_decimal(pnl_str, default=Decimal("NaN")) # Use safe_decimal
                    other_pos_details[side]["liq_price"] = liq_price
                    other_pos_details[side]["pnl"] = pnl
                    other_pos_details[side]["open_time"] = open_time_dt # Store parsed datetime or None
                    if side == "long": first_long_found = True
                    else: first_short_found = True

                    logger.debug(f"[{symbol}] Processing first significant {side.upper()} entry: Qty={contracts.normalize()}, Entry={entry_price}, Liq={liq_price}, PnL={pnl}, OpenTime={open_time_dt}")

            except (InvalidOperation, TypeError, DecimalException) as e: # Added DecimalException
                logger.error(f"[{symbol}] Could not parse position data for {side} side from entry (Qty:'{contracts_str}', Entry:'{entry_price_str}', Liq:'{liq_price_str}', Pnl:'{pnl_str}'). Error: {e}")
                # Do not continue here, this specific position entry is problematic.
                # The pos_dict[side] will retain its default NaN/0 values.
                continue
            except Exception as e: # Catch other unexpected errors during parsing
                 logger.error(f"[{symbol}] Unexpected error parsing position data for {side} side: {e}", exc_info=True)
                 continue
        elif side not in aggregated_positions:
            logger.warning(f"[{symbol}] Position data found for unknown side '{side}'. Skipping.")

    # --- Finalize Position Dictionary ---
    final_pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN"), "open_time": None}, # Added open_time
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN"), "open_time": None} # Added open_time
    }

    for side in ["long", "short"]:
        total_qty = aggregated_positions[side]["qty"]
        weighted_sum = aggregated_positions[side]["entry_price_sum_qty"]

        if total_qty.copy_abs() >= CONFIG.position_qty_epsilon:
            final_pos_dict[side]["qty"] = total_qty
            # Calculate weighted average entry price
            # Use absolute total_qty for division, handle division by zero
            if weighted_sum.copy_abs() > Decimal("0") and total_qty.copy_abs() > Decimal("0"):
                try:
                    final_pos_dict[side]["entry_price"] = weighted_sum / total_qty.copy_abs()
                except DivisionByZero:
                     logger.error(f"[{symbol}] Division by zero calculating weighted entry price for {side} side (total_qty_abs is zero).")
                     final_pos_dict[side]["entry_price"] = Decimal("NaN")
            else:
                # If sum is zero or total qty is zero/negative (unexpected), entry price is NaN
                final_pos_dict[side]["entry_price"] = Decimal("NaN")

            # Use the stored other details from the first significant position
            final_pos_dict[side]["liq_price"] = other_pos_details[side]["liq_price"]
            final_pos_dict[side]["pnl"] = other_pos_details[side]["pnl"]
            final_pos_dict[side]["open_time"] = other_pos_details[side]["open_time"] # Store open_time

            # Log with formatted decimals (for display)
            entry_log = f"{final_pos_dict[side]['entry_price']:.4f}" if not final_pos_dict[side]['entry_price'].is_nan() else "N/A"
            liq_log = f"{final_pos_dict[side]['liq_price']:.4f}" if not final_pos_dict[side]['liq_price'].is_nan() else "N/A"
            pnl_log = f"{final_pos_dict[side]['pnl']:+.4f}" if not final_pos_dict[side]['pnl'].is_nan() else "N/A"
            opentime_log = final_pos_dict[side]['open_time'].strftime('%Y-%m-%d %H:%M:%S %Z') if final_pos_dict[side]['open_time'] else 'N/A'

            logger.info(Fore.YELLOW + f"[{symbol}] Aggregated active {side.upper()} position: Qty={total_qty.normalize()}, Entry={entry_log}, Liq≈{liq_log}, PnL≈{pnl_log}, OpenTime={opentime_log}")

        else:
            logger.info(Fore.BLUE + f"[{symbol}] No significant {side.upper()} position found.")

    if final_pos_dict["long"]["qty"].copy_abs() > CONFIG.position_qty_epsilon and \
       final_pos_dict["short"]["qty"].copy_abs() > CONFIG.position_qty_epsilon:
        logger.warning(Fore.YELLOW + f"[{symbol}] Both LONG ({final_pos_dict['long']['qty'].normalize()}) and SHORT ({final_pos_dict['short']['qty'].normalize()}) positions found. Pyrmethus assumes one-way mode and will manage these independently. Please ensure your exchange account is configured for one-way trading if this is unexpected.")

    logger.info(Fore.GREEN + f"[{symbol}] Position spirits consulted.")
    return final_pos_dict
# </xaiArtifact>


# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# Modified to accept currency argument (though usually fixed per symbol)
def get_balance(currency: str = "USDT") -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Fetches the free and total balance for a specific currency using retry wrapper, returns Decimals."""
    global EXCHANGE
    # Note: Balance is usually account-wide, not per-symbol.
    # If using Unified account, balance applies to all markets.
    # If using Inverse/Linear accounts separately, need to specify account type.
    # This function assumes a single account type relevant for the configured symbols.
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching balance.")
        return None, None

    balance_data = None
    try:
        # Bybit V5 fetch_balance might need accountType (UNIFIED/CONTRACT) or coin.
        # CCXT's defaultType/SubType and category *should* handle this, but params might be needed.
        # Let's rely on fetch_with_retries to add category if configured.
        # Example params if needed for specific account types (adjust as per Bybit V5 docs):
        params = {'accountType': 'UNIFIED'} # Assume Unified for V5
        balance_data = fetch_with_retries(EXCHANGE.fetch_balance, params=params)
    except Exception as e:
        logger.error(Fore.RED + f"Unhandled exception during fetch_balance call via fetch_with_retries: {e}", exc_info=True)
        return None, None

    if balance_data is None:
        # fetch_with_retries already logged the failure
        logger.error(Fore.RED + f"Failed to fetch balance after retries. Cannot assess risk capital.")
        return None, None

    # --- Parse Balance Data ---
    # Initialize with NaN Decimals to indicate failure to find/parse
    free_balance = Decimal("NaN")
    total_balance = Decimal("NaN")  # Represents Equity for futures/swaps

    try:
        # Attempt to parse using standard CCXT structure
        # This often works for account-wide balance summary
        if currency in balance_data and isinstance(balance_data[currency], dict):
            currency_balance = balance_data[currency]
            free_str = currency_balance.get('free')
            total_str = currency_balance.get('total')  # 'total' usually represents equity in futures

            free_balance = safe_decimal(free_str, default=Decimal("NaN")) # Use safe_decimal
            total_balance = safe_decimal(total_str, default=Decimal("NaN")) # Use safe_decimal
            if not free_balance.is_nan() or not total_balance.is_nan():
                 logger.debug(f"Parsed standard CCXT balance for {currency}: Free={free_balance.normalize()}, Total={total_balance.normalize()}")


        # Fallback: Check 'info' for exchange-specific structure (Bybit V5 example)
        # This is the most reliable for Bybit V5 Unified Margin/Contract accounts
        # Check if standard parsing yielded NaN for total_balance before trying info fallback
        # Also check if standard parsing yielded NaN for free_balance
        if (free_balance.is_nan() or total_balance.is_nan()) and 'info' in balance_data and isinstance(balance_data['info'], dict):
            info_data = balance_data['info']
            # V5 structure: result -> list -> account objects
            if 'result' in info_data and isinstance(info_data['result'], dict) and \
               'list' in info_data['result'] and isinstance(info_data['result']['list'], list):
                for account in info_data['result']['list']:
                    # Find the account object for the target currency
                    if isinstance(account, dict) and account.get('coin') == currency:
                        # Bybit V5 Unified Margin fields (check docs):
                        # 'walletBalance': Total assets in wallet
                        # 'availableToWithdraw': Amount withdrawable
                        # 'equity': Account equity (often the most relevant for risk calculation in futures)
                        # 'availableToBorrow': Margin specific
                        # 'totalPerpUPL': Unrealized PnL (already included in equity)
                        equity_str = account.get('equity')  # Use equity as 'total' for risk calculation
                        free_str_info = account.get('availableToWithdraw')  # Use availableToWithdraw as 'free'

                        # Overwrite if found in info, as info is often more granular/accurate for V5
                        free_balance = safe_decimal(free_str_info, default=free_balance) # Use safe_decimal, keep old value if info fails
                        total_balance = safe_decimal(equity_str, default=total_balance) # Use safe_decimal, keep old value if info fails

                        logger.debug(f"Parsed Bybit V5 info structure for {currency}: Free={free_balance.normalize()}, Equity={total_balance.normalize()}")
                        break  # Found the currency account
            else:
                logger.warning("Bybit V5 info structure missing 'result' or 'list'.")

        # If parsing failed, balances will remain NaN
        if free_balance.is_nan():
            logger.warning(f"Could not find or parse free balance for {currency} in balance data.")
        if total_balance.is_nan():
            logger.warning(f"Could not find or parse total/equity balance for {currency} in balance data.")
            # Critical if equity is needed for risk calc
            logger.error(Fore.RED + "Failed to determine account equity. Cannot proceed safely.")
            return free_balance, None  # Indicate equity failure specifically

        # Use 'total' balance (Equity) as the primary value for risk calculation
        equity = total_balance

        logger.info(Fore.GREEN + f"Vault contains {free_balance:.4f} free {currency} (Equity/Total: {equity:.4f}).")
        return free_balance, equity  # Return free and total (equity)

    except (InvalidOperation, TypeError, KeyError, DecimalException) as e: # Added DecimalException
        logger.error(Fore.RED + f"Error parsing balance data for {currency}: {e}. Raw keys: {list(balance_data.keys()) if isinstance(balance_data, dict) else 'N/A'}")
        logger.debug(f"Raw balance data: {balance_data}")
        return None, None  # Indicate parsing failure
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected shadow encountered querying vault: {e}", exc_info=True)
        return None, None
# </xaiArtifact>


# --- PnL Fetching (from Block 1, adapted) ---
# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# Modified to accept symbol argument and use MARKET_INFOS
def fetch_trade_history_wrapper(symbol: Optional[str] = None, lookback_days: int = CONFIG.pnl_lookback_days) -> Optional[List[Dict]]:
    """
    Fetches trade execution history for realized PnL calculation using retry wrapper.
    Handles pagination. Returns Decimals for numeric fields where possible.
    """
    global EXCHANGE, MARKET_INFOS
    logger.debug(f"Fetching trade history for symbol {symbol or 'All'} for last {lookback_days} days...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching trade history.")
        return None

    # Use exchange's internal ID for symbol if available
    symbol_id = None
    if symbol:
        market_info = get_market_info(symbol)
        if market_info:
            symbol_id = market_info.get('id')
            if symbol_id is None:
                logger.error(f"[{symbol}] Exchange symbol ID not available from Market Info. Cannot fetch trade history.")
                return None

    params = {
        "category": CONFIG.market_type,
        "startTime": str(int((time.time() - lookback_days * 86400) * 1000)),
        "limit": "100" # Max limit allowed by Bybit V5 execution list endpoint
    }
    if symbol_id:
        params["symbol"] = symbol_id

    trades = []
    next_page_cursor = None

    while True:
        if next_page_cursor:
             params["cursor"] = next_page_cursor

        # Use fetch_with_retries for the API call
        response = None
        try:
            # Bybit V5 Execution List endpoint: GET /v5/execution/list
            response = fetch_with_retries(EXCHANGE.private_get_execution_list, params=params) # Use implicit method
        except Exception as e:
            logger.error(Fore.RED + f"Unhandled exception during fetch_trade_history call via fetch_with_retries: {e}", exc_info=True)
            return None # Indicate failure

        if response is None:
            # fetch_with_retries already logged the failure
            logger.error(Fore.RED + f"Failed to fetch trade history after retries for symbol {symbol or 'All'}.")
            return None

        # Check Bybit V5 response structure (retCode 0 means success)
        if isinstance(response, dict) and isinstance(response.get('info'), dict) and response['info'].get('retCode') == 0:
            result = response['info'].get('result', {})
            trade_list = result.get('list', [])
            next_page_cursor = result.get('nextPageCursor')

            # Process and add trades to the list, converting numeric fields to Decimal
            for trade in trade_list:
                 processed_trade = {}
                 # Copy essential fields, converting numeric ones
                 # Bybit V5 fields: symbol, side, orderId, execTime, execType, price, qty, execFee, feeRate, closedPnl (or realisedPnl?)
                 for key in ['symbol', 'side', 'orderId', 'execTime', 'execType', 'price', 'qty', 'execFee', 'feeRate', 'closedPnl', 'realisedPnl']:
                      raw_value = trade.get(key)
                      if raw_value is not None:
                           if key in ['price', 'qty', 'execFee', 'feeRate', 'closedPnl', 'realisedPnl']:
                                processed_trade[key] = safe_decimal(raw_value, default=Decimal('NaN')) # Use safe_decimal
                                if processed_trade[key].is_nan():
                                     logger.warning(f"[{symbol}] Could not parse trade field '{key}' with value '{raw_value}' to Decimal for trade {trade.get('orderId', 'N/A')}. Storing as string.")
                                     processed_trade[key] = str(raw_value) # Store as string if Decimal conversion fails
                           else:
                                processed_trade[key] = raw_value # Keep other fields as they are
                 trades.append(processed_trade)

            if not next_page_cursor:
                break # No more pages

            logger.debug(f"[{symbol or 'All'}] Fetched {len(trade_list)} trades, next cursor: {next_page_cursor[:10] if next_page_cursor else 'None'}. Total fetched so far: {len(trades)}")
            # Optional: add a small delay between paginated calls if rate limits are hit
            time.sleep(0.1)

        else:
            error_msg = response['info'].get('retMsg', 'Unknown error') if isinstance(response.get('info'), dict) else str(response)
            error_code = response['info'].get('retCode', 'N/A') if isinstance(response.get('info'), dict) else 'N/A'
            logger.error(Fore.RED + f"[{symbol or 'All'}] Failed to fetch trade history page (Code: {error_code}). Exchange message: {error_msg}. Stopping pagination.")
            return None # Indicate failure

    logger.debug(f"[{symbol or 'All'}] Finished fetching trade history. Total trades: {len(trades)}")
    return trades


def fetch_realized_pnl_summary(symbol: Optional[str] = None) -> Optional[Dict]:
    """
    Fetches trade history and calculates total realized PnL for the lookback period.
    Returns Decimal for total PnL.
    """
    logger.info(Fore.CYAN + f"[{symbol or 'All Symbols'}] # Consulting Realized PnL scrolls (Last {CONFIG.pnl_lookback_days} Days)...")

    # Fetch Realized PnL from Trade History
    # Fetch history for the specified symbol or all if None
    trades = fetch_trade_history_wrapper(symbol, CONFIG.pnl_lookback_days)
    if trades is None:
        logger.error(Fore.RED + f"[{symbol or 'All Symbols'}] Failed to fetch trade history for Realized PnL summary.")
        return None # Indicate failure

    # Aggregate realized PnL
    total_realized = Decimal("0.0")
    trade_count_with_pnl = 0 # Count trades where PnL was successfully parsed and is non-zero

    for trade in trades:
        try:
            # Use the Decimal values already parsed by fetch_trade_history_wrapper
            # Check both potential keys for Bybit V5
            realized_pnl = trade.get('closedPnl', trade.get('realisedPnl', Decimal('NaN')))

            # Ensure realized_pnl is Decimal and not NaN
            if isinstance(realized_pnl, Decimal) and not realized_pnl.is_nan():
                 # Only add non-zero PnL to the total
                 if realized_pnl.copy_abs() > CONFIG.position_qty_epsilon: # Use epsilon for zero check
                      total_realized += realized_pnl
                      trade_count_with_pnl += 1
            else:
                 logger.debug(f"[{symbol}] Skipping trade with invalid or missing PnL value for {trade.get('symbol', 'Unknown')}: {realized_pnl}")

        except Exception as e:
            logger.error(f"[{symbol}] Error processing trade for realized PnL {trade.get('symbol', 'Unknown')}: {e}")

    logger.info(f"[{symbol or 'All Symbols'}] Realized PnL Summary (Last {CONFIG.pnl_lookback_days} Days): Total Realized = {total_realized:+.4f} ({trade_count_with_pnl} trades contributing PnL)")

    return {"total_realized": total_realized}
# </xaiArtifact>


# <xaiArtifact artifact_id="e9f0a1b2-c3d4-5e6f-7a8b-9c0d1e2f3a4b" title="Enhance Order Confirmation" contentType="python">Snippet 19: Enhance Order Confirmation</xaiArtifact>
# Modified to accept symbol argument and use MARKET_INFOS
def check_order_status(order_id: str, symbol: str, timeout: int = CONFIG.order_check_timeout_seconds) -> Optional[Dict]:
    """Checks order status with retries and timeout. Returns the final order dict or None."""
    global EXCHANGE, MARKET_INFOS
    logger.info(Fore.CYAN + f"[{symbol}] Verifying final status of order {order_id} (Timeout: {timeout}s)...")
    market_info = get_market_info(symbol)
    if EXCHANGE is None or market_info is None:
        logger.error(f"[{symbol}] Exchange object or Market Info not available for checking order status.")
        return None
    if order_id is None or order_id == "":
        logger.warning(f"[{symbol}] Received None or empty order_id to check status. Skipping check.")
        return None  # Cannot check status for a None ID

    start_time = time.time()
    last_status = 'unknown'
    attempt = 0
    check_interval = CONFIG.order_check_delay_seconds  # Start with configured delay

    # Use exchange's internal ID for symbol
    symbol_id = market_info.get('id')
    if symbol_id is None:
         logger.error(f"[{symbol}] Market Info not available, cannot get exchange symbol ID for order status check.")
         return None

    while time.time() - start_time < timeout:
        attempt += 1
        logger.debug(f"[{symbol}] Checking order {order_id}, attempt {attempt}...")
        order_status_data = None
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            # Category param should be handled automatically by fetch_with_retries
            # Bybit V5 fetch_order requires category and symbol (exchange ID)
            fetch_params = {'category': CONFIG.market_type, 'symbol': symbol_id}
            order_status_data = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=fetch_params)

            if order_status_data and isinstance(order_status_data, dict):
                last_status = order_status_data.get('status', 'unknown')
                # Use Decimal for precision comparison
                filled_qty = safe_decimal(order_status_data.get('filled', 0.0)) # Use safe_decimal

                logger.info(f"[{symbol}] Order {order_id} status check: {last_status}, Filled: {filled_qty.normalize()}")  # Use normalize for cleaner log

                # Check for terminal states (fully filled, canceled, rejected, expired)
                # 'closed' usually means fully filled for market/limit orders on Bybit.
                if last_status in ['closed', 'canceled', 'rejected', 'expired']:
                    logger.info(f"[{symbol}] Order {order_id} reached terminal state: {last_status}.")
                    return order_status_data  # Return the final order dict
                # If 'open' but fully filled (can happen briefly), treat as terminal 'closed'
                # Check remaining amount using epsilon
                remaining_qty = safe_decimal(order_status_data.get('remaining', 0.0)) # Use safe_decimal

                # Check if filled >= original amount using epsilon (more reliable than remaining == 0)
                original_amount = safe_decimal(order_status_data.get('amount', 0.0)) # Use safe_decimal

                # Consider order fully filled if filled amount is very close to original amount
                # Use a tolerance based on configured epsilon or a small fraction of original amount
                fill_tolerance = max(CONFIG.position_qty_epsilon, original_amount.copy_abs() * Decimal('1E-6')) if not original_amount.is_nan() else CONFIG.position_qty_epsilon

                # Check if filled quantity is effectively equal to the original amount
                if not original_amount.is_nan() and (original_amount - filled_qty).copy_abs() < fill_tolerance:
                    if last_status == 'open':
                        logger.info(f"[{symbol}] Order {order_id} is 'open' but appears fully filled ({filled_qty.normalize()}/{original_amount.normalize()}). Treating as 'closed'.")
                        order_status_data['status'] = 'closed'  # Update status locally for clarity
                        return order_status_data
                    elif last_status in ['partially_filled']:
                        logger.info(f"[{symbol}] Order {order_id} is '{last_status}' but appears fully filled ({filled_qty.normalize()}/{original_amount.normalize()}). Treating as 'closed'.")
                        order_status_data['status'] = 'closed'  # Update status locally for clarity
                        return order_status_data

            else:
                # fetch_with_retries failed or returned unexpected data
                # Error logged within fetch_with_retries, just note it here
                logger.warning(f"[{symbol}] fetch_order call failed or returned invalid data for {order_id}. Continuing check loop.")
                # Continue the loop to retry check_order_status itself

        except ccxt.OrderNotFound:
            # Order is definitively not found. This is a terminal state indicating it never existed or was fully purged after fill/cancel.
            # For market orders, this often means it filled and was purged quickly.
            logger.info(f"[{symbol}] Order {order_id} confirmed NOT FOUND by exchange. Assuming filled/cancelled and purged.")
            # Cannot get fill details, but assume it's gone. For market orders, this often means success.
            # It's better to return None to indicate we couldn't verify the *final* state definitively,
            # unless we can infer fill from context (e.g., it was a market order).
            # For robustness, returning None when status cannot be confirmed is safer.
            # However, if it was a market order entry, 'NotFound' implies success.
            # Let's return a synthesized 'closed' status if the order type was market,
            # otherwise return None. Need order type... this is not readily available here.
            # The safest is to return None, and the caller must handle 'None' meaning 'cannot confirm'.
            return None  # Explicitly indicate not found / unable to verify

        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
            # Critical non-retryable errors
            logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] Authentication/Permission error during order status check for {order_id}: {e}. Halting.")
            send_email("Bot Critical Error", f"[{symbol}] Order check auth/perm error: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
            sys.exit(1)
        except Exception as e:
            # Catch any other unexpected error during the check itself
            logger.error(f"[{symbol}] Unexpected error during order status check loop for {order_id}: {e}", exc_info=True)
            # Decide whether to retry or fail; retrying is part of the loop.

        # Wait before the next check_order_status attempt
        time_elapsed = time.time() - start_time
        if time_elapsed + check_interval < timeout:
            logger.debug(f"[{symbol}] Order {order_id} status ({last_status}) not terminal, sleeping {check_interval:.1f}s...")
            time.sleep(check_interval)
            check_interval = min(check_interval * 1.2, 5)  # Slightly increase interval up to 5s, max 5s
        else:
            break  # Exit loop if next sleep would exceed timeout

    # --- Timeout Reached ---
    logger.error(Fore.RED + f"[{symbol}] Timed out checking status for order {order_id} after {timeout} seconds. Last known status: {last_status}.")
    # Attempt one final fetch outside the loop to get the very last state if possible
    final_check_status = None
    try:
        logger.info(f"[{symbol}] Performing final status check for order {order_id} after timeout...")
        # Use fetch_with_retries for the final check too
        fetch_params = {'category': CONFIG.market_type, 'symbol': symbol_id}
        final_check_status = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=fetch_params)

        if final_check_status and isinstance(final_check_status, dict):
            final_status = final_check_status.get('status', 'unknown')
            final_filled = final_check_status.get('filled', 'N/A')
            logger.info(f"[{symbol}] Final status after timeout: {final_status}, Filled: {final_filled}")
            # Return this final status even if timed out earlier
            return final_check_status
        else:
            logger.error(f"[{symbol}] Final status check for order {order_id} also failed or returned invalid data.")
            # If the final check also fails, we cannot confirm status.
            return None  # Indicate persistent failure to get status
    except ccxt.OrderNotFound:
        logger.error(Fore.RED + f"[{symbol}] Order {order_id} confirmed NOT FOUND on final check after timeout.")
        return None  # Still cannot confirm final state details
    except Exception as e:
        logger.error(f"[{symbol}] Error during final status check for order {order_id}: {e}", exc_info=True)
        return None
# </xaiArtifact>


# <xaiArtifact artifact_id="1f2e3d4c-5b6a-7f8e-9d0c-1b2a3c4d5e6f" title="Add Position Age Timeout" contentType="python">Snippet 18: Position Age Timeout</xaiArtifact>
def close_position(symbol: str, position_side: str, quantity: Optional[Decimal] = None) -> bool:
    """
    Closes an open position (long or short) for a symbol using a market order.
    Optionally takes a specific quantity to close (partial close).
    Uses fetch_with_retries and ensures reduceOnly.
    """
    global EXCHANGE, MARKET_INFOS, order_tracker

    logger.warning(Fore.YELLOW + f"[{symbol}] Attempting to close {position_side.upper()} position...")

    market_info = get_market_info(symbol)
    if EXCHANGE is None or market_info is None:
        logger.error(f"[{symbol}] Exchange object or Market Info not available. Cannot close position.")
        termux_notify("Closure Failed!", f"[{symbol}] Cannot close {position_side.upper()} - Exchange not ready.")
        return False

    # Determine closure side (sell for long, buy for short)
    close_side = "sell" if position_side == "long" else "buy"
    if position_side not in ["long", "short"]:
        logger.error(f"[{symbol}] Invalid position side '{position_side}' specified for closure.")
        return False

    try:
        # If quantity is not provided, fetch current position to get size
        if quantity is None:
            logger.debug(f"[{symbol}] Fetching current position size for closure...")
            positions = get_current_position(symbol)
            if positions is None:
                logger.error(f"[{symbol}] Failed to fetch current position size. Cannot close {position_side.upper()} position.")
                termux_notify("Closure Failed!", f"[{symbol}] Cannot close {position_side.upper()} - failed to fetch size.")
                return False

            pos_data = positions.get(position_side, {})
            qty_to_close_raw = pos_data.get('qty', Decimal("0.0"))

            # Ensure quantity is Decimal and positive for closing (absolute value)
            qty_to_close = safe_decimal(qty_to_close_raw).copy_abs()
            if qty_to_close.is_nan() or qty_to_close < CONFIG.position_qty_epsilon:
                logger.warning(f"[{symbol}] No significant {position_side.upper()} position found to close (Qty: {qty_to_close.normalize()}).")
                # Even if no position found, ensure tracker is clear
                if symbol in order_tracker and position_side in order_tracker[symbol]:
                     order_tracker[symbol][position_side] = {"sl_id": None, "tsl_id": None, "tp_id": None}
                     save_state() # Save state after clearing tracker
                     logger.info(f"[{symbol}] Cleared tracker for {position_side} as position is negligible.")
                return True # Consider successful if position is already gone or negligible

        else: # Quantity was provided (e.g., for partial close or known quantity)
            qty_to_close = safe_decimal(quantity).copy_abs() # Use safe_decimal and absolute value

        if qty_to_close.is_nan() or qty_to_close < CONFIG.position_qty_epsilon:
             logger.warning(f"[{symbol}] Specified closure quantity {qty_to_close.normalize()} is negligible or invalid. Skipping closure order.")
             return False # Cannot place order with negligible quantity


        # Format quantity precisely for closure order (round down)
        close_qty_str = format_amount(symbol, qty_to_close, ROUND_DOWN)
        try:
            close_qty_decimal = Decimal(close_qty_str)
        except InvalidOperation:
            logger.critical(f"[{symbol}] Failed to parse formatted closure quantity '{close_qty_str}' for {position_side} position. Cannot attempt closure.")
            termux_notify("Closure Failed!", f"[{symbol}] Cannot close {position_side.upper()} - qty parse failed!")
            return False  # Indicate failure

        # Validate against minimum quantity before attempting closure
        min_qty_dec = safe_decimal(market_info['limits'].get('amount', {}).get('min'), default=Decimal("0")) # Use safe_decimal with default
        if close_qty_decimal.copy_abs() < min_qty_dec: # Use absolute for comparison
            logger.critical(f"[{symbol}] Closure quantity {close_qty_decimal.normalize()} for {position_side} position is below exchange minimum {min_qty_dec.normalize()}. MANUAL CLOSURE REQUIRED!")
            termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS < MIN QTY! Close manually!")
            return False  # Indicate failure

        logger.warning(Fore.YELLOW + f"[{symbol}] Placing {close_side.upper()} market order to close {position_side.upper()} position ({close_qty_decimal.normalize()})...")

        # Place the closure market order
        close_params = {'reduceOnly': True}  # Crucial: Only close, don't open new position
        # fetch_with_retries handles category param
        close_order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol,
            side=close_side,
            amount=float(close_qty_decimal),  # CCXT needs float
            params=close_params
        )

        # Check response for success (basic check, full fill confirmed later by state check)
        if close_order and (close_order.get('id') or close_order.get('info', {}).get('retCode') == 0):
            close_id = close_order.get('id', 'N/A (retCode 0)')
            logger.trade(Fore.GREEN + f"[{symbol}] Position closure order placed successfully: ID {close_id}")
            termux_notify("Closure Order Sent", f"[{symbol}] {position_side.upper()} closure order sent.")

            # Optional: Wait and check status to confirm fill? For graceful shutdown, just placing is often enough.
            # For timeout/age closure, maybe check status. Let's add a quick check.
            logger.debug(f"[{symbol}] Waiting briefly for closure order {close_id} fill confirmation...")
            # Use check_order_status with a short timeout
            fill_confirmed = False
            if close_id != 'N/A (retCode 0)': # Only check status if we have an ID
                 try:
                      # Use a short timeout (e.g., half the normal timeout)
                      confirmed_order = check_order_status(close_id, symbol, timeout=CONFIG.order_check_timeout_seconds // 2)
                      if confirmed_order and confirmed_order.get('status') in ['closed', 'filled'] and safe_decimal(confirmed_order.get('filled', '0')).copy_abs() >= close_qty_decimal.copy_abs() * Decimal('0.99'): # Check for nearly full fill
                           logger.debug(f"[{symbol}] Closure order {close_id} confirmed filled.")
                           fill_confirmed = True
                      elif confirmed_order:
                           logger.warning(f"[{symbol}] Closure order {close_id} status: {confirmed_order.get('status')}, Filled: {safe_decimal(confirmed_order.get('filled', '0')).normalize()}. Fill not fully confirmed.")
                      else:
                           logger.warning(f"[{symbol}] Could not confirm status of closure order {close_id}.")

                 except Exception as status_err:
                      logger.warning(f"[{symbol}] Error checking status of closure order {close_id}: {status_err}")

            else:
                 logger.warning(f"[{symbol}] No order ID for closure order, skipping status check.")


            # If order placed successfully (or retCode 0), assume closure is in progress or complete.
            # Clear local tracker state for this side.
            if symbol in order_tracker and position_side in order_tracker[symbol]:
                 order_tracker[symbol][position_side] = {"sl_id": None, "tsl_id": None, "tp_id": None} # Clear all stop markers
                 save_state() # Save state after clearing tracker
                 logger.info(f"[{symbol}] Cleared tracker for {position_side} after closure order.")

            return True # Indicate success in placing closure order (best effort)

        else:
            # Log critical error if closure order placement fails
            error_msg = close_order.get('info', {}).get('retMsg', 'No ID and no success code.') if isinstance(close_order, dict) else str(close_order)
            logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] FAILED TO PLACE closure order for {position_side} position ({qty_to_close.normalize()}): {error_msg}. MANUAL INTERVENTION REQUIRED!")
            termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS CLOSURE FAILED! Manual action!")
            return False

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied) as e:
        logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] FAILED TO CLOSE {position_side} position ({qty_to_close.normalize() if 'qty_to_close' in locals() else 'N/A'}): {e}. MANUAL INTERVENTION REQUIRED!")
        termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS CLOSURE FAILED! Manual action!")
        return False
    except Exception as e:
        logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] Unexpected error closing {position_side} position: {e}. MANUAL INTERVENTION REQUIRED!", exc_info=True)
        termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS CLOSURE FAILED! Manual action!")
        return False
# </xaiArtifact>


# <xaiArtifact artifact_id="e9f0a1b2-c3d4-5e6f-7a8b-9c0d1e2f3a4b" title="Enhance Order Confirmation" contentType="python">Snippet 19: Enhance Order Confirmation</xaiArtifact>
# check_order_filled is replaced by check_order_status function logic within place_risked_market_order
# The snippet added the check_order_filled function, but place_risked_market_order already had inline logic.
# The separate check_order_status function is a cleaner way to implement the snippet's goal.
# No need to add a separate check_order_filled function. The logic is integrated.
# </xaiArtifact>


# <xaiArtifact artifact_id="03f4e5d6-a7b8-9c0d-1e2f-3a4b5c6d7e8f" title="Add Performance Metrics" contentType="python">Snippet 20: Performance Metrics</xaiArtifact>
# Function to record trade entry (basic version)
def record_trade_entry(symbol: str, side: str, qty: Decimal, entry_price: Decimal) -> None:
    """Records a trade entry event."""
    global trades
    # NOTE: This is a basic record. Realized PnL needs to be tracked on exit.
    # This current implementation records PnL as 0 at entry, which is incorrect for performance calculation.
    # A more robust solution would track unique open positions and update this record on closure with realized PnL.
    trade_record = {
        'timestamp': pd.Timestamp.now(tz='UTC').isoformat(), # Store as ISO string for JSON serialization
        'symbol': symbol,
        'side': side,
        'qty': float(qty), # Store as float for simplicity in this list
        'entry_price': float(entry_price) if not entry_price.is_nan() else None, # Store as float
        'pnl': 0.0 # Placeholder for realized PnL (needs update on exit)
        # Add unique trade ID if available from exchange
    }
    trades.append(trade_record)
    # Optional: Save trades list to file periodically or on shutdown
    # save_state() # This function doesn't save 'trades' yet, could add it.

# Function to attempt to update trade record with exit details (needs more complex logic to match entry/exit)
# This is NOT implemented in the snippet, just noting the requirement for full performance tracking.
# def update_trade_exit(symbol: str, side: str, exit_price: Decimal, realized_pnl: Decimal) -> None:
#     """Attempts to find and update a trade record with exit details and realized PnL."""
#     # This requires matching an open trade record to a closed position.
#     # Complex logic needed, likely involving position ID or unique trade IDs.
#     pass
# </xaiArtifact>


# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# Modified to accept symbol argument and use MARKET_INFOS
def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    """Places a market order for a symbol with calculated size and initial ATR-based stop-loss/take-profit, using Decimal precision."""
    trade_action = f"{side.upper()} Market Entry"
    logger.trade(Style.BRIGHT + f"[{symbol}] Attempting {trade_action}...")

    market_info = get_market_info(symbol)
    if market_info is None or EXCHANGE is None:
        logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Market info or Exchange not available.")
        return False

    # Use exchange's internal ID for symbol
    market_id = market_info.get('id')
    if market_id is None:
         logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Exchange symbol ID not available from Market Info.")
         return False

    # <xaiArtifact artifact_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890" title="Add Dry Run Mode" contentType="python">Snippet 25: Dry Run Mode</xaiArtifact>
    if CONFIG.dry_run:
        logger.info(Fore.CYAN + f"[{symbol}] DRY RUN: Would place {side} order with risk {risk_percentage*100:.2f}% based on ATR {atr:.6f}.")
        # Simulate a successful order placement for the dry run
        # Need to calculate size and SL/TP prices to log them in the dry run message
        try:
            # Fetch current price for dry run calc
            ticker_data = fetch_with_retries(EXCHANGE.fetch_ticker, symbol)
            if not ticker_data or ticker_data.get("last") is None:
                 logger.warning(f"[{symbol}] DRY RUN: Failed to fetch ticker for detailed dry run log.")
                 price = Decimal("NaN")
            else:
                 price = safe_decimal(ticker_data["last"])

            if not price.is_nan() and price > Decimal("0") and not atr.is_nan() and atr > Decimal("0"):
                 # Calculate SL price
                 sl_distance_points = CONFIG.sl_atr_multiplier * atr
                 sl_price_raw = price - sl_distance_points if side == "buy" else price + sl_distance_points
                 sl_price_str = format_price(symbol, sl_price_raw)

                 # Calculate TP price (if multiplier is set)
                 tp_price_str = "N/A"
                 if CONFIG.tp_atr_multiplier is not None and CONFIG.tp_atr_multiplier > Decimal("0"):
                      tp_distance_points = CONFIG.tp_atr_multiplier * atr
                      tp_price_raw = price + tp_distance_points if side == "buy" else price - tp_distance_points
                      tp_price_str = format_price(symbol, tp_price_raw)

                 # Calculate estimated quantity (needs balance)
                 quote_currency = market_info.get('settle', 'USDT')
                 _, total_equity = get_balance(quote_currency)
                 qty_est_str = "N/A"
                 if total_equity is not None and not total_equity.is_nan() and total_equity > Decimal("0") and (price - sl_price_raw).copy_abs() > Decimal("0"):
                      risk_amount_quote = total_equity * risk_percentage
                      stop_distance_quote = (price - sl_price_raw).copy_abs()
                      contract_size_dec = safe_decimal(market_info.get('contractSize', '1'))
                      qty_raw = Decimal('0')
                      if CONFIG.market_type == 'linear':
                           qty_raw = risk_amount_quote / stop_distance_quote
                      elif CONFIG.market_type == 'inverse' and not contract_size_dec.is_nan() and contract_size_dec > Decimal("0"):
                           qty_raw = risk_amount_quote / (stop_distance_quote * contract_size_dec)
                      qty_est_str = format_amount(symbol, qty_raw, ROUND_DOWN)

                 logger.info(Fore.CYAN + f"[{symbol}] DRY RUN: Would place {side.upper()} order for est. {qty_est_str} @ market price (est. {price:.4f}) with SL @ {sl_price_str} and TP @ {tp_price_str}.")
            else:
                 logger.info(Fore.CYAN + f"[{symbol}] DRY RUN: Would place {side.upper()} order. Insufficient data for detailed calculation log.")

        except Exception as dry_run_calc_err:
             logger.warning(f"[{symbol}] Error during dry run calculation/logging: {dry_run_calc_err}")

        # In dry run, simulate success to allow the rest of the cycle to proceed as if a trade happened
        return True
    # </xaiArtifact>


    # --- Pre-computation & Validation ---
    quote_currency = market_info.get('settle', 'USDT')  # Use settle currency (e.g., USDT)
    free_balance, total_equity = get_balance(quote_currency)  # Fetch balance using the function
    if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
        logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Invalid, NaN, or zero account equity ({total_equity}). Cannot calculate risk capital.")
        return False

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Invalid ATR value ({atr}). Check indicator calculation.")
        return False

    # Fetch current ticker price using fetch_ticker with retries
    ticker_data = None
    try:
        ticker_data = fetch_with_retries(EXCHANGE.fetch_ticker, symbol)
    except Exception as e:
        logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Unhandled exception fetching ticker: {e}")
        return False

    if not ticker_data or ticker_data.get("last") is None:
        logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Cannot fetch current ticker price for sizing/SL calculation. Ticker data: {ticker_data}")
        # fetch_with_retries should have logged details if it failed
        return False

    try:
        # Use 'last' price as current price estimate, convert to Decimal
        price = safe_decimal(ticker_data["last"]) # Use safe_decimal
        if price.is_nan() or price <= Decimal(0):
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Fetched current price ({price}) is invalid, zero or negative. Aborting.")
            return False
        logger.debug(f"[{symbol}] Current ticker price: {price:.8f} {quote_currency}")  # Log with high precision for debug

        # --- Calculate Stop Loss Price ---
        sl_distance_points = CONFIG.sl_atr_multiplier * atr
        if sl_distance_points.is_nan() or sl_distance_points <= Decimal("0"):  # Use Decimal zero
            logger.error(f"{Fore.RED}[{symbol}] {trade_action} failed: Stop distance calculation resulted in invalid, zero or negative value ({sl_distance_points}). Check ATR ({atr:.6f}) and multiplier ({CONFIG.sl_atr_multiplier}).")
            return False

        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else:  # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations/API call
        sl_price_formatted_str = format_price(symbol, sl_price_raw)
        # Convert back to Decimal *after* formatting for consistent internal representation
        sl_price = safe_decimal(sl_price_formatted_str) # Use safe_decimal
        if sl_price.is_nan():
             logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Formatted SL price '{sl_price_formatted_str}' is invalid Decimal. Aborting.")
             return False


        logger.debug(f"[{symbol}] ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier.normalize()}, SL Distance Points: {sl_distance_points:.6f}")
        logger.debug(f"[{symbol}] Raw SL Price: {sl_price_raw:.8f}, Formatted SL Price for API: {sl_price_formatted_str} (Decimal: {sl_price.normalize()})")

        # Sanity check SL placement relative to current price
        # Use a small multiple of price tick size for tolerance if available, else a tiny Decimal
        try:
            price_precision_info = market_info['precision'].get('price')
            # If precision is number of decimals (int)
            if isinstance(price_precision_info, int):
                price_tick_size = Decimal(1) / (Decimal(10) ** price_precision_info)
            # If precision is tick size (string or Decimal)
            elif isinstance(price_precision_info, (str, Decimal)):
                price_tick_size = safe_decimal(price_precision_info, default=Decimal("1E-8"))
                if price_tick_size.is_nan() or price_tick_size <= Decimal("0"): price_tick_size = Decimal("1E-8") # Fallback
            else:
                price_tick_size = Decimal("1E-8")  # Fallback tiny Decimal
        except Exception:
            price_tick_size = Decimal("1E-8")  # Fallback tiny Decimal

        tolerance_ticks = price_tick_size * Decimal('5')  # Allow a few ticks tolerance

        if side == "buy" and sl_price >= price - tolerance_ticks:
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Calculated SL price ({sl_price.normalize()}) is too close to or above current price ({price.normalize()}) [Tolerance: {tolerance_ticks.normalize()}]. Check ATR/multiplier or market precision. Aborting.")
            return False
        if side == "sell" and sl_price <= price + tolerance_ticks:
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Calculated SL price ({sl_price.normalize()}) is too close to or below current price ({price.normalize()}) [Tolerance: {tolerance_ticks.normalize()}]. Check ATR/multiplier or market precision. Aborting.")
            return False

        # --- Calculate Take Profit Price (if enabled) ---
        tp_price = Decimal("NaN") # Default to NaN
        tp_price_str = "N/A"
        # <xaiArtifact artifact_id="f8e9d0c1-a2b3-4c5d-6e7f-8a9b0c1d2e3a" title="Add Support for Take-Profit Orders" contentType="python">Snippet 6: Take-Profit Orders</xaiArtifact>
        if CONFIG.tp_atr_multiplier is not None and CONFIG.tp_atr_multiplier > Decimal("0"):
            tp_distance_points = CONFIG.tp_atr_multiplier * atr
            if not tp_distance_points.is_nan() and tp_distance_points > Decimal("0"):
                 tp_price_raw = price + tp_distance_points if side == "buy" else price - tp_distance_points
                 tp_price_str = format_price(symbol, tp_price_raw)
                 tp_price = safe_decimal(tp_price_str) # Convert back to Decimal after formatting

                 if tp_price.is_nan():
                      logger.warning(f"[{symbol}] Could not parse formatted TP price '{tp_price_str}' to Decimal.")
                 elif (side == "buy" and tp_price <= price + tolerance_ticks) or (side == "sell" and tp_price >= price - tolerance_ticks):
                      logger.warning(f"[{symbol}] Calculated TP price ({tp_price.normalize()}) is too close to or inside current price ({price.normalize()}) [Tolerance: {tolerance_ticks.normalize()}]. Skipping TP.")
                      tp_price = Decimal("NaN") # Invalidate TP if too close
                      tp_price_str = "N/A (Too Close)"
                 else:
                      logger.debug(f"[{symbol}] Raw TP Price: {tp_price_raw:.8f}, Formatted TP Price for API: {tp_price_str} (Decimal: {tp_price.normalize()})")
            else:
                 logger.warning(f"[{symbol}] TP distance calculation resulted in invalid, zero or negative value ({tp_distance_points}). Skipping TP.")
        # </xaiArtifact>


        # --- Calculate Position Size ---
        risk_amount_quote = total_equity * risk_percentage
        # Stop distance in quote currency (use absolute difference, ensure Decimals)
        stop_distance_quote = (price - sl_price).copy_abs()

        if stop_distance_quote.is_nan() or stop_distance_quote <= Decimal("0"):
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Stop distance in quote currency is invalid, zero or negative ({stop_distance_quote.normalize()}). Check ATR, multiplier, or market precision. Cannot calculate size.")
            return False

        # Calculate quantity based on contract size and linear/inverse type
        # Ensure contract_size is a Decimal
        contract_size_dec = safe_decimal(market_info.get('contractSize', '1'))  # Ensure Decimal
        if contract_size_dec.is_nan() or contract_size_dec <= Decimal("0"):
             logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Invalid contract size ({contract_size_dec.normalize()}). Cannot calculate size.")
             return False


        qty_raw = Decimal('0')

        # --- Sizing Logic ---
        # Bybit uses size in Base currency for Linear (e.g., BTC for BTC/USDT)
        # Bybit uses size in Contracts (which represent USD value for BTC/USD inverse)
        # Risk Amount (Quote) = (Entry Price (Quote/Base) - SL Price (Quote/Base)) * Qty (Base)
        # Qty (Base) = Risk Amount (Quote) / (Entry Price (Quote/Base) - SL Price (Quote/Base))
        # Qty (Base) = Risk Amount (Quote) / Stop Distance (Quote)
        if CONFIG.market_type == 'linear':
            qty_raw = risk_amount_quote / stop_distance_quote
            logger.debug(f"[{symbol}] Linear Sizing: Qty (Base) = {risk_amount_quote:.8f} {quote_currency} / {stop_distance_quote:.8f} {quote_currency} = {qty_raw:.8f}")

        elif CONFIG.market_type == 'inverse':
            # Qty (in Contracts) = Total Risk (in Quote) / (Stop Distance (in Quote/Base) * Contract Size (in Base/Contract))
            qty_raw = risk_amount_quote / (stop_distance_quote * contract_size_dec)
            logger.debug(f"[{symbol}] Inverse Sizing (Contract Size = {contract_size_dec.normalize()} Base/Contract): Qty (Contracts) = {risk_amount_quote:.8f} {quote_currency} / ({stop_distance_quote:.8f} {quote_currency} * {contract_size_dec.normalize()}) = {qty_raw:.8f}")


        else:
            logger.error(f"[{symbol}] {trade_action} failed: Unsupported market type for sizing: {CONFIG.market_type}")
            return False

        # <xaiArtifact artifact_id="e2d1c1a2-1a2b-3c4d-5e6f-7a8b9c0d1e2f" title="Add Configuration for Maximum Position Size" contentType="python">Snippet 1: Max Position Size</xaiArtifact>
        # Calculate max allowed position size based on equity percentage
        # Max position value in quote currency = Total Equity * Max Position Percentage
        max_pos_value_quote = total_equity * CONFIG.max_position_percentage
        logger.debug(f"[{symbol}] Max allowed position value: {max_pos_value_quote.normalize()} {quote_currency} ({CONFIG.max_position_percentage*100:.2f}% of equity)")

        # Convert max position value back to quantity based on market type and current price
        # This is an *estimate* as market price can change before order fills.
        max_qty_allowed_raw = Decimal('Infinity')
        if price > Decimal("0"): # Avoid division by zero
            if CONFIG.market_type == 'linear':
                 # Max Value (Quote) = Max Qty (Base) * Price (Quote/Base)
                 max_qty_allowed_raw = max_pos_value_quote / price
                 logger.debug(f"[{symbol}] Max Qty (Base) allowed by max position percentage: {max_qty_allowed_raw.normalize()}")
            elif CONFIG.market_type == 'inverse' and not contract_size_dec.is_nan() and contract_size_dec > Decimal("0"):
                 # Max Value (Quote) = Max Qty (Contracts) * Contract Size (Base/Contract) * Price (Quote/Base)
                 # Max Qty (Contracts) = Max Value (Quote) / (Contract Size (Base/Contract) * Price (Quote/Base))
                 max_qty_allowed_raw = max_pos_value_quote / (contract_size_dec * price)
                 logger.debug(f"[{symbol}] Max Qty (Contracts) allowed by max position percentage: {max_qty_allowed_raw.normalize()}")

        # Cap the calculated quantity if it exceeds the max allowed based on equity percentage
        if qty_raw > max_qty_allowed_raw:
             logger.warning(Fore.YELLOW + f"[{symbol}] Calculated quantity {qty_raw.normalize()} exceeds max allowed quantity {max_qty_allowed_raw.normalize()} based on MAX_POSITION_PERCENTAGE. Capping order size.")
             qty_raw = max_qty_allowed_raw # Cap to the max allowed quantity
        # </xaiArtifact>


        # --- Format and Validate Quantity ---
        # Format quantity according to market precision (ROUND_DOWN to be conservative)
        qty_formatted_str = format_amount(symbol, qty_raw, ROUND_DOWN)
        qty = safe_decimal(qty_formatted_str) # Use safe_decimal
        if qty.is_nan():
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Formatted quantity '{qty_formatted_str}' is invalid Decimal. Aborting.")
            return False


        logger.debug(f"[{symbol}] Risk Amount: {risk_amount_quote:.8f} {quote_currency}, Stop Distance: {stop_distance_quote:.8f} {quote_currency}")
        logger.debug(f"[{symbol}] Raw Qty: {qty_raw:.12f}, Formatted Qty (Rounded Down): {qty.normalize()}")  # Use normalize for cleaner log

        # Validate Quantity Against Market Limits
        min_qty = safe_decimal(market_info['limits'].get('amount', {}).get('min'), default=Decimal("0")) # Use safe_decimal with default
        max_qty = safe_decimal(market_info['limits'].get('amount', {}).get('max'), default=Decimal('Infinity')) # Use safe_decimal with default

        # Use epsilon for zero check
        if qty.copy_abs() < min_qty or qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Calculated quantity ({qty.normalize()}) is zero or below minimum ({min_qty.normalize()}, epsilon {CONFIG.position_qty_epsilon:.1E}). Risk amount ({risk_amount_quote:.4f}), stop distance ({stop_distance_quote:.4f}), or equity might be too small. Cannot place order.")
            return False
        if max_qty != Decimal('Infinity') and qty > max_qty:
            logger.warning(Fore.YELLOW + f"[{symbol}] Calculated quantity {qty.normalize()} exceeds market maximum {max_qty.normalize()}. Capping order size to {max_qty.normalize()}.")
            qty = max_qty  # Use the Decimal max_qty
            # Re-format capped amount - crucial! Use ROUND_DOWN again.
            qty_formatted_str = format_amount(symbol, qty, ROUND_DOWN)
            qty = safe_decimal(qty_formatted_str) # Use safe_decimal again
            if qty.is_nan():
                 logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Re-formatted capped quantity '{qty_formatted_str}' is invalid Decimal. Aborting.")
                 return False

            logger.info(f"[{symbol}] Re-formatted capped Qty: {qty.normalize()}")
            # Double check if capped value is now below min (unlikely but possible with large steps)
            if qty.copy_abs() < min_qty or qty.copy_abs() < CONFIG.position_qty_epsilon:
                logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Capped quantity ({qty.normalize()}) is now below minimum ({min_qty.normalize()}) or zero. Aborting.")
                return False

        # Validate minimum cost if available
        min_cost = safe_decimal(market_info['limits'].get('cost', {}).get('min')) # Use safe_decimal
        if not min_cost.is_nan():
            try:
                estimated_cost = Decimal('0')
                # Estimate cost based on market type (Approximate!)
                if CONFIG.market_type == 'linear':
                    # Cost = Qty (Base) * Price (Quote/Base) = Quote
                    estimated_cost = qty * price
                elif CONFIG.market_type == 'inverse' and not contract_size_dec.is_nan() and contract_size_dec > Decimal('0'):
                    # Cost = Qty (Contracts) * Contract Size (Base/Contract) * Price (Quote/Base)
                    estimated_cost = qty * contract_size_dec * price
                    logger.debug(f"[{symbol}] Inverse cost estimation: Qty({qty.normalize()}) * ContractSize({contract_size_dec.normalize()}) * Price({price.normalize()}) = {estimated_cost.normalize()}")
                else:
                    estimated_cost = Decimal('NaN')  # Cannot estimate cost for unknown type

                # Check if estimated_cost is valid before comparison
                if not estimated_cost.is_nan() and estimated_cost.copy_abs() < min_cost:
                    logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Estimated order cost/value ({estimated_cost:.4f} {quote_currency}) is below minimum required ({min_cost:.4f} {quote_currency}). Increase risk or equity. Cannot place order.")
                    return False
                elif estimated_cost.is_nan():
                    logger.warning(Fore.YELLOW + f"[{symbol}] Estimated order cost could not be determined, skipping min cost check.")

            except (InvalidOperation, TypeError, KeyError, DecimalException) as cost_err: # Added DecimalException
                logger.warning(f"[{symbol}] Could not estimate order cost: {cost_err}. Skipping min cost check.")
            except Exception as cost_err:
                logger.warning(f"[{symbol}] Unexpected error during cost estimation: {cost_err}. Skipping min cost check.", exc_info=True)

        logger.info(Fore.YELLOW + f"[{symbol}] Calculated Order: Side={side.upper()}, Qty={qty.normalize()}, Entry≈{price:.4f}, SL={sl_price_formatted_str} (ATR={atr:.4f}), TP={tp_price_str}")

    except (InvalidOperation, TypeError, DivisionByZero, KeyError, DecimalException) as e: # Added DecimalException
        logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] {trade_action} failed: Error during pre-calculation/validation: {e}", exc_info=True)
        return False
    except Exception as e:  # Catch any other unexpected errors
        logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] {trade_action} failed: Unexpected error during pre-calculation: {e}", exc_info=True)
        return False

    # --- Cast the Market Order Spell ---
    order = None
    order_id = None
    filled_qty = Decimal("0.0")  # Initialize filled_qty for later use
    average_price = price  # Initialize average_price with estimated price, update if actual fill price is available

    try:
        logger.trade(f"[{symbol}] Submitting {side.upper()} market order for {qty.normalize()}...")
        # fetch_with_retries handles category param
        # CCXT expects float amount
        order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol,
            side=side,
            amount=float(qty)  # Explicitly cast Decimal qty to float for CCXT API
        )

        if order is None:
            # fetch_with_retries logged the error
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Market order placement failed after retries.")
            return False

        logger.debug(f"[{symbol}] Market order raw response: {order}")

        # --- Verify Order Fill (Crucial Step) ---
        # Bybit V5 create_market_order might return retCode=0 without an order ID in the standard field immediately.
        # The actual filled order details might be in 'info'->'result'->'list'.
        # Let's check retCode first if ID is missing, and try to extract details.

        # Default assumption: need to check status later
        needs_status_check = True
        order_status_data = None  # Initialize as None

        # Try to get order ID and fill details from the initial response first
        if order and isinstance(order, dict):
            order_id = order.get('id')  # Standard CCXT field

            # Check Bybit V5 specific response structure in 'info'
            if 'info' in order and isinstance(order['info'], dict):
                info = order['info']
                ret_code = info.get('retCode')
                ret_msg = info.get('retMsg')

                if ret_code == 0:  # Bybit V5 success code
                    logger.debug(f"[{symbol}] Market order initial response retCode 0 ({ret_msg}).")
                    # Try to extract order ID or details from the V5 result list if available
                    if 'result' in info and isinstance(info['result'], dict) and 'list' in info['result'] and isinstance(info['result']['list'], list) and info['result']['list']:
                        # For market orders, the list should contain the immediately filled order(s)
                        first_order_info = info['result']['list'][0]
                        # Prioritize Order ID from the standard field, fallback to info field
                        order_id = order_id or first_order_info.get('orderId')  # Use standard ID if present, otherwise info ID

                        # Also capture filled details from response if possible (more accurate than check_order_status if available)
                        filled_qty_from_response = safe_decimal(first_order_info.get('cumExecQty', '0'))  # Bybit V5 field, use safe_decimal
                        avg_price_from_response = safe_decimal(first_order_info.get('avgPrice', 'NaN'))  # Bybit V5 field, use safe_decimal
                        order_status_from_response = first_order_info.get('orderStatus')  # Bybit V5 field e.g. "Filled"

                        if filled_qty_from_response.is_nan(): filled_qty_from_response = Decimal('0')
                        if avg_price_from_response.is_nan(): avg_price_from_response = Decimal('NaN')


                        logger.debug(f"[{symbol}] Extracted details from V5 response: ID={order_id}, Status={order_status_from_response}, Filled={filled_qty_from_response.normalize()}, AvgPrice={avg_price_from_response.normalize()}")

                        # If response indicates "Filled" and filled quantity is significant
                        if order_status_from_response == 'Filled' and filled_qty_from_response.copy_abs() >= CONFIG.position_qty_epsilon:
                            filled_qty = filled_qty_from_response
                            average_price = avg_price_from_response if not avg_price_from_response.is_nan() else price  # Use estimated price if avgPrice is NaN
                            logger.trade(Fore.GREEN + Style.BRIGHT + f"[{symbol}] Market order confirmed FILLED from response: {filled_qty.normalize()} @ {average_price:.4f}")
                            # Synthesize a CCXT-like dict for consistency
                            order_status_data = {'status': 'closed', 'filled': float(filled_qty), 'average': float(average_price) if not average_price.is_nan() else None, 'id': order_id}
                            needs_status_check = False  # No need to check status later, already confirmed filled
                        else:
                            logger.warning(f"[{symbol}] {trade_action}: Order ID found ({order_id}) but filled quantity from response ({filled_qty_from_response.normalize()}) is zero or negligible, or status is not 'Filled'. Will proceed with check_order_status.")
                            # order_id is set, needs_status_check remains True -> check_order_status will be called
                    else:
                        logger.warning(f"[{symbol}] {trade_action}: Market order submitted (retCode 0) but no Order ID or fill details found in V5 result list. Cannot reliably track status immediately.")
                        # needs_status_check remains True, order_id might be None -> check_order_status might fail without ID
                        # If order_id is None here, we cannot proceed safely.
                        if order_id is None:
                            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Market order submitted but no Order ID was obtained from response. Aborting.")
                            return False
                else:  # Non-zero retCode indicates failure
                    logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Market order submission failed. Exchange message: {ret_msg} (Code: {ret_code})")
                    return False  # Submission failed

            # If we get here and needs_status_check is still True, proceed with status check
            if needs_status_check:
                if order_id is None:
                    logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Market order submission response processed, but no order ID could be identified. Cannot check status. Aborting.")
                    return False  # Cannot proceed safely without order ID

                logger.info(f"[{symbol}] Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}...")
                time.sleep(CONFIG.order_check_delay_seconds)

                # Use the dedicated check_order_status function
                order_status_data = check_order_status(order_id, symbol, timeout=CONFIG.order_check_timeout_seconds)

        # Evaluate the result of the status check (either from initial response or fetch_order)
        order_final_status = 'unknown'
        if order_status_data and isinstance(order_status_data, dict):
            order_final_status = order_status_data.get('status', 'unknown')
            filled_str = order_status_data.get('filled')
            average_str = order_status_data.get('average')  # Average fill price

            # Update filled_qty and average_price based on status check result
            filled_qty = safe_decimal(filled_str, default=filled_qty) # Use safe_decimal, keep previous estimate if check fails
            avg_price_decimal = safe_decimal(average_str) # Use safe_decimal
            if not avg_price_decimal.is_nan() and avg_price_decimal > Decimal(0):  # Use actual fill price only if valid
                 average_price = avg_price_decimal

            logger.debug(f"[{symbol}] Order {order_id} status check result: Status='{order_final_status}', Filled='{filled_qty.normalize()}', AvgPrice='{average_price:.4f}'")

            # 'closed' means fully filled for market orders on Bybit
            if order_final_status == 'closed' and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"[{symbol}] Order {order_id} confirmed FILLED: {filled_qty.normalize()} @ {average_price:.4f}")
            # Handle partial fills (less common for market, but possible during high volatility)
            # Bybit V5 market orders typically fill fully or are rejected. If partially filled, something unusual is happening.
            elif order_final_status in ['open', 'partially_filled'] and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                logger.warning(Fore.YELLOW + f"[{symbol}] Market Order {order_id} status is '{order_final_status}' but partially/fully filled ({filled_qty.normalize()}). This is unusual for market orders. Proceeding with filled amount.")
                # Assume the filled quantity is the position size and proceed.
            elif order_final_status in ['open', 'partially_filled'] and filled_qty.copy_abs() < CONFIG.position_qty_epsilon:
                logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Order {order_id} has status '{order_final_status}' but filled quantity is effectively zero ({filled_qty.normalize()}). Aborting SL/TP placement.")
                # Attempt to cancel just in case it's stuck (defensive)
                try:
                    logger.info(f"[{symbol}] Attempting cancellation of stuck/unfilled order {order_id}.")
                    # Bybit V5 cancel_order requires category and symbol
                    cancel_params = {'category': CONFIG.market_type, 'symbol': market_id}
                    fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)  # Use fetch_with_retries
                except Exception as cancel_err: logger.warning(f"[{symbol}] Failed to cancel stuck order {order_id}: {cancel_err}")
                return False
            else:  # canceled, rejected, expired, failed, unknown, or closed with zero fill
                logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] {trade_action} failed: Order {order_id} did not fill successfully: Status '{order_final_status}', Filled Qty: {filled_qty.normalize()}. Aborting SL/TP placement.")
                # Attempt to cancel if not already in a terminal state (defensive)
                if order_final_status not in ['canceled', 'rejected', 'expired']:
                    try:
                        logger.info(f"[{symbol}] Attempting cancellation of failed/unknown status order {order_id}.")
                        # Bybit V5 cancel_order requires category and symbol
                        cancel_params = {'category': CONFIG.market_type, 'symbol': market_id}
                        fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)  # Use fetch_with_retries
                    except Exception: pass  # Ignore errors here, main goal failed anyway
                return False
        else:
            # check_order_status returned None (timeout, not found, or final check failed)
            # If check_order_status returns None, we cannot confirm successful fill. Assume failure.
            logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] {trade_action} failed: Could not determine final status for order {order_id} (timeout or not found). Assuming failure. Aborting SL/TP placement.")
            # Attempt to cancel just in case it's stuck somehow (defensive)
            try:
                logger.info(f"[{symbol}] Attempting cancellation of unknown status order {order_id}.")
                # If order_id was None earlier, this will fail. check_order_status should handle None ID internally if possible, but better to have ID.
                if order_id:
                    # Bybit V5 cancel_order requires category and symbol
                    cancel_params = {'category': CONFIG.market_type, 'symbol': market_id}
                    fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)
                else:
                    logger.warning(f"[{symbol}] Cannot attempt cancellation: No order ID available.")
            except Exception: pass
            return False

        # Final check on filled quantity after status check
        if filled_qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.error(Fore.RED + f"[{symbol}] {trade_action} failed: Order {order_id} resulted in effectively zero filled quantity ({filled_qty.normalize()}) after status check. No position opened.")
            return False

        # --- Record Trade Entry (Basic) ---
        record_trade_entry(symbol, side, filled_qty, average_price) # Use filled_qty and average_price

        # --- Place Initial Stop-Loss & Take-Profit Orders (Set on Position for Bybit V5) ---
        position_side = "long" if side == "buy" else "short"
        logger.trade(f"[{symbol}] Setting initial SL/TP for new {position_side.upper()} position (filled qty: {filled_qty.normalize()})...")

        # Use the SL price calculated earlier, already formatted string
        sl_price_str_for_api = sl_price_formatted_str

        # Define parameters for setting the stop-loss/take-profit on the position (Bybit V5 specific)
        # We use the `private_post_position_set_trading_stop` implicit method via CCXT
        # This endpoint applies to the *entire* position for the symbol/side/category.
        set_stp_params: Dict[str, Any] = {
            'category': CONFIG.market_type,  # Required
            'symbol': market_id,  # Use exchange-specific market ID
            'tpslMode': 'Full',  # Apply SL/TP/TSL to the entire position ('Partial' also possible)
            # 'positionIdx': 0 # For Bybit unified: 0 for one-way mode (default), 1/2 for hedge mode
            # Note: We don't need quantity here as it applies to the existing position matching symbol/category/side.
            # No need to specify side in params for V5 set-trading-stop, it's determined by the symbol and position context.
            # Wait, the Bybit V5 docs *do* show 'side' as a parameter for set-trading-stop. Let's add it for clarity and correctness.
            'side': 'Buy' if position_side == 'long' else 'Sell'  # Add side parameter (Bybit V5 expects "Buy"/"Sell" for side)
        }

        # Add SL parameters
        if sl_price is not None and not sl_price.is_nan():
             set_stp_params['stopLoss'] = sl_price_str_for_api
             set_stp_params['slTriggerBy'] = CONFIG.sl_trigger_by
        else:
             logger.warning(f"[{symbol}] Calculated SL price is invalid. Will not set Stop Loss on position.")


        # Add TP parameters if calculated and valid
        # <xaiArtifact artifact_id="f8e9d0c1-a2b3-4c5d-6e7f-8a9b0c1d2e3a" title="Add Support for Take-Profit Orders" contentType="python">Snippet 6: Take-Profit Orders</xaiArtifact>
        if tp_price is not None and not tp_price.is_nan():
             set_stp_params['takeProfit'] = tp_price_str
             set_stp_params['tpTriggerBy'] = CONFIG.sl_trigger_by # Use SL trigger by for TP trigger by, or add separate config
        else:
             logger.info(f"[{symbol}] TP not calculated or invalid. Will not set Take Profit on position.")
        # </xaiArtifact>

        # Check if at least SL or TP parameters are included before sending the request
        if 'stopLoss' not in set_stp_params and 'takeProfit' not in set_stp_params:
             logger.error(Fore.RED + f"[{symbol}] Neither Stop Loss nor Take Profit parameters are valid or present. Cannot set trading stop. Position is UNPROTECTED.")
             # Critical: Position is open without SL/TP. Attempt emergency close.
             raise ccxt.InvalidOrder("No valid SL or TP parameters to set.")


        logger.trade(f"[{symbol}] Setting Position SL/TP: SL Trigger={set_stp_params.get('stopLoss', 'N/A')}, SL TriggerBy={set_stp_params.get('slTriggerBy', 'N/A')}, TP Trigger={set_stp_params.get('takeProfit', 'N/A')}, TP TriggerBy={set_stp_params.get('tpTriggerBy', 'N/A')}, Side={set_stp_params['side']}")
        logger.debug(f"[{symbol}] Set SL/TP Params (for setTradingStop): {set_stp_params}")

        stp_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods if available
            # Endpoint: POST /v5/position/set-trading-stop
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                stp_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_stp_params)
            else:
                # Fallback: Raise error if specific method missing.
                logger.error(Fore.RED + f"[{symbol}] Cannot set SL/TP: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                # Critical: Position is open without SL. Attempt emergency close.
                raise ccxt.NotSupported("SL/TP setting method not available via CCXT.")

            logger.debug(f"[{symbol}] Set SL/TP raw response: {stp_set_response}")

            # Handle potential failure from fetch_with_retries
            if stp_set_response is None:
                # fetch_with_retries already logged the failure
                raise ccxt.ExchangeError("Set SL/TP request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(stp_set_response, dict) and isinstance(stp_set_response.get('info'), dict) and stp_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"[{symbol}] Stop Loss/Take Profit successfully set directly on the {position_side.upper()} position.")
                # --- Update Global State ---
                # CRITICAL: Clear any previous tracker state for this side (should be clear from check before entry, but defensive)
                # Use placeholders to indicate SL/TP are active on the position
                sl_marker_id = f"POS_SL_{position_side.upper()}" if 'stopLoss' in set_stp_params else None
                tp_marker_id = f"POS_TP_{position_side.upper()}" if 'takeProfit' in set_stp_params else None
                order_tracker[symbol][position_side] = {"sl_id": sl_marker_id, "tsl_id": None, "tp_id": tp_marker_id} # Update tracker with TP ID
                save_state() # <xaiArtifact artifact_id="c2a3767e-3119-420d-92a6-05533a61856c" title="Persist State to File" contentType="python">Snippet 9: Persist State</xaiArtifact>
                logger.info(f"[{symbol}] Updated order tracker: {order_tracker[symbol]}")

                # <xaiArtifact artifact_id="f3d2c1b0-2b1a-4d3c-6e5f-8a9b0c1d2e3f" title="Implement Cooldown Period After Trades" contentType="python">Snippet 2: Cooldown Period</xaiArtifact>
                # Update cooldown time after successful entry
                update_cooldown_time(symbol)
                # </xaiArtifact>

                # Use actual average fill price in notification
                entry_msg = (
                    f"ENTERED {side.upper()} {filled_qty.normalize()} {symbol.split('/')[0]} @ {average_price:.4f}. "
                    f"Initial SL @ {sl_price_str_for_api}. "
                    f"TP @ {tp_price_str}. " # Include TP in message
                    f"TSL pending profit threshold."
                )
                logger.trade(Back.GREEN + Fore.BLACK + Style.BRIGHT + entry_msg)
                termux_notify("Trade Entry", f"[{symbol}] {side.upper()} @ {average_price:.4f}, SL: {sl_price_str_for_api}, TP: {tp_price_str}") # Include TP in notification
                return True  # SUCCESS!

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                error_code = None
                if isinstance(stp_set_response.get('info'), dict):
                    error_msg = stp_set_response['info'].get('retMsg', error_msg)
                    error_code = stp_set_response['info'].get('retCode')
                    error_msg += f" (Code: {error_code})"
                raise ccxt.ExchangeError(f"[{symbol}] Stop loss/Take Profit setting failed. Exchange message: {error_msg}")

        # --- Handle SL/TP Setting Failures ---
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied, DecimalException) as e: # Added DecimalException
            # This is critical - position opened but SL/TP setting failed. Emergency close needed.
            logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] CRITICAL: Failed to set stop-loss/take-profit on position after entry: {e}. Position is UNPROTECTED or partially protected.")
            logger.warning(Fore.YELLOW + f"[{symbol}] Attempting emergency market closure of unprotected position...")
            send_email("Bot Critical Error", f"[{symbol}] Failed to set SL/TP after entry: {e}. Position unprotected. Attempting emergency close.") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>

            try:
                # Use the dedicated close_position function
                if close_position(symbol, position_side, quantity=filled_qty): # Pass filled quantity for precision
                    logger.info(f"[{symbol}] Emergency closure initiated successfully.")
                    # close_position updates tracker and saves state
                else:
                    logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] EMERGENCY CLOSURE FAILED. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!")
                    termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                    send_email("Bot EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>

            except Exception as close_err:
                logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] EMERGENCY CLOSURE FAILED (Exception during closure attempt): {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!", exc_info=True)
                termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                send_email("Bot EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>


            return False  # Signal overall failure of the entry process due to SL/TP failure

        except Exception as e:
            logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] Unexpected error setting SL/TP: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + f"[{symbol}] Position may be open without Stop Loss/Take Profit due to unexpected SL/TP setting error. MANUAL INTERVENTION ADVISED.")
            send_email("Bot Unexpected Error", f"[{symbol}] Unexpected error setting SL/TP: {e}. Position potentially unprotected.") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
            # Consider emergency closure here too? Yes, safer. Re-use the emergency closure logic.
            try:
                position_side = "long" if side == "buy" else "short"
                if close_position(symbol, position_side, quantity=filled_qty): # Pass filled quantity
                    logger.info(f"[{symbol}] Emergency closure initiated successfully after unexpected error.")
                    # close_position updates tracker and saves state
                else:
                    logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] EMERGENCY CLOSURE FAILED after unexpected SL/TP error. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!")
                    termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                    send_email("Bot EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>

            except Exception as close_err:
                logger.critical(Fore.RED + Style.BRIGHT + f"[{symbol}] EMERGENCY CLOSURE FAILED (Exception during closure attempt) after unexpected SL/TP error: {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!", exc_info=True)
                termux_notify("EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                send_email("Bot EMERGENCY!", f"[{symbol}] {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>


            return False  # Signal overall failure

    # --- Handle Initial Market Order Failures ---
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied, DecimalException) as e: # Added DecimalException
        # Error placing the initial market order itself (handled by fetch_with_retries re-raising)
        logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] {trade_action} failed: Exchange error placing market order: {e}")
        send_email("Bot Trade Entry Failed", f"[{symbol}] {trade_action} failed: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
        # The exception message itself is usually sufficient.
        return False
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] {trade_action} failed: Unexpected error during market order placement: {e}", exc_info=True)
        send_email("Bot Unexpected Error", f"[{symbol}] {trade_action} failed: {e}") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
        return False


# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# Modified to accept symbol argument and use MARKET_INFOS, order_tracker[symbol]
def manage_trailing_stop(
    symbol: str,
    position_side: str,  # 'long' or 'short'
    position_qty: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    atr: Decimal
) -> None:
    """Manages the activation and setting of a trailing stop loss on the position, using Decimal."""
    global order_tracker, EXCHANGE, MARKET_INFOS

    logger.debug(f"[{symbol}] Checking TSL status for {position_side.upper()} position...")

    market_info = get_market_info(symbol)
    if EXCHANGE is None or market_info is None:
        logger.error(f"[{symbol}] Exchange or Market Info not available, cannot manage TSL.")
        return

    # Use exchange's internal ID for symbol
    market_id = market_info.get('id')
    if market_id is None:
         logger.error(f"[{symbol}] Cannot manage TSL: Exchange symbol ID not available from Market Info.")
         return

    # Ensure tracker entry exists for the symbol/side
    if symbol not in order_tracker or position_side not in order_tracker[symbol]:
        logger.warning(f"[{symbol}] Order tracker entry missing for {position_side}. Initializing.")
        if symbol not in order_tracker: order_tracker[symbol] = {}
        if position_side not in order_tracker[symbol]: order_tracker[symbol][position_side] = {"sl_id": None, "tsl_id": None, "tp_id": None} # Added tp_id


    # --- Initial Checks ---
    if position_qty.copy_abs() < CONFIG.position_qty_epsilon or entry_price.is_nan() or entry_price <= Decimal("0"):
        # If position seems closed or invalid, ensure tracker is clear.
        if order_tracker[symbol][position_side]["sl_id"] or order_tracker[symbol][position_side]["tsl_id"] or order_tracker[symbol][position_side]["tp_id"]: # Check tp_id too
            logger.info(f"[{symbol}] Position {position_side} appears closed or invalid (Qty: {position_qty.normalize()}, Entry: {entry_price.normalize() if not entry_price.is_nan() else 'NaN'}). Clearing stale order trackers.")
            order_tracker[symbol][position_side] = {"sl_id": None, "tsl_id": None, "tp_id": None} # Clear all
            save_state() # <xaiArtifact artifact_id="c2a3767e-3119-420d-92a6-05533a61856c" title="Persist State to File" contentType="python">Snippet 9: Persist State</xaiArtifact>
        return  # No position to manage TSL for

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.warning(Fore.YELLOW + f"[{symbol}] Cannot evaluate TSL activation: Invalid ATR value ({atr.normalize()}).")
        return

    if current_price is None or current_price.is_nan() or current_price <= Decimal("0"):
        logger.warning(Fore.YELLOW + f"[{symbol}] Cannot evaluate TSL activation: Invalid current price ({current_price.normalize()}).")
        return


    # --- Get Current Tracker State ---
    initial_sl_marker = order_tracker[symbol][position_side]["sl_id"]  # Could be ID or placeholder "POS_SL_..."
    active_tsl_marker = order_tracker[symbol][position_side]["tsl_id"]  # Could be ID or placeholder "POS_TSL_..."
    active_tp_marker = order_tracker[symbol][position_side]["tp_id"] # Check TP too


    # If TSL is already active (has a marker), assume exchange handles the trail.
    if active_tsl_marker:
        log_msg = f"[{symbol}] {position_side.upper()} TSL ({active_tsl_marker}) is already active. Exchange is managing the trail."
        logger.debug(log_msg)
        # Sanity check: Ensure initial SL and TP markers are None if TSL is active
        if initial_sl_marker or active_tp_marker:
            logger.warning(f"[{symbol}] Inconsistent state: TSL active ({active_tsl_marker}) but initial SL ({initial_sl_marker}) or TP ({active_tp_marker}) markers are also present in tracker. Clearing them.")
            order_tracker[symbol][position_side]["sl_id"] = None
            order_tracker[symbol][position_side]["tp_id"] = None
            save_state() # <xaiArtifact artifact_id="c2a3767e-3119-420d-92a6-05533a61856c" title="Persist State to File" contentType="python">Snippet 9: Persist State</xaiArtifact>
        return  # TSL is already active, nothing more to do here

    # If TSL is not active, check if we *should* activate it.
    # Requires an initial SL marker to be present to indicate the position is at least protected by a fixed SL.
    # Note: Bybit V5 allows setting TSL without removing SL in the same call,
    # but clearing the SL when activating TSL is the intended behavior for this strategy.
    # If there's no SL marker here, it implies either SL wasn't set, was hit, or was manually removed.
    # We proceed cautiously: if initial SL marker is missing, we *could* attempt to set TSL,
    # but it might overwrite an existing manual SL/TP.
    # For robustness, let's *only* attempt TSL activation if the initial SL marker is present,
    # assuming the marker indicates the position is protected by the initial SL we set.
    # If the marker is missing, assume the position is either already managed, or unprotected.
    if not initial_sl_marker:
        logger.warning(f"[{symbol}] Cannot activate TSL for {position_side.upper()}: Initial SL protection marker is missing from tracker ({initial_sl_marker}). Position might be unprotected or already managed externally. Skipping TSL activation.")
        return  # Cannot activate TSL if initial SL state is unknown/missing

    # --- Check TSL Activation Condition ---
    profit = Decimal("NaN")
    try:
        if position_side == "long":
            profit = current_price - entry_price
        else:  # short
            profit = entry_price - current_price
    except (TypeError, InvalidOperation, DecimalException):  # Handle potential NaN in prices
        logger.warning(f"[{symbol}] Cannot calculate profit for TSL check due to NaN price(s).")
        return

    # Activation threshold in price points
    activation_threshold_points = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(f"[{symbol}] {position_side.upper()} Profit: {profit.normalize() if not profit.is_nan() else 'NaN'}, TSL Activation Threshold (Points): {activation_threshold_points.normalize() if not activation_threshold_points.is_nan() else 'NaN'} ({CONFIG.tsl_activation_atr_multiplier.normalize()} * ATR)")

    # Use a small buffer (e.g., 0.01% of price) for threshold comparison to avoid flickering near the threshold
    # A fixed tiny Decimal might be better than a percentage buffer for volatile pairs
    threshold_buffer = current_price.copy_abs() * Decimal('0.0001') if not current_price.is_nan() else Decimal('0')  # 0.01% of current price as buffer

    # Activate TSL only if profit exceeds the threshold (use Decimal comparison) AND threshold is valid
    if not profit.is_nan() and not activation_threshold_points.is_nan() and profit > activation_threshold_points + threshold_buffer:
        logger.trade(Fore.GREEN + Style.BRIGHT + f"[{symbol}] Profit threshold reached for {position_side.upper()} position (Profit {profit:.4f} > Threshold {activation_threshold_points:.4f}). Activating TSL.")

        # --- Set Trailing Stop Loss on Position ---
        # Bybit V5 sets TSL directly on the position using specific parameters.
        # We use the same `set_trading_stop` endpoint as the initial SL, but provide TSL params.

        # TSL distance as percentage string (e.g., "0.5" for 0.5%)
        # Ensure correct formatting for the API (string representation with sufficient precision)
        # Quantize to a reasonable number of decimal places for percentage (e.g., 3-4)
        trail_percent_str = str(CONFIG.trailing_stop_percent.quantize(Decimal("0.001")))  # Format to 3 decimal places

        # Bybit V5 Parameters for setting TSL on position:
        # Endpoint: POST /v5/position/set-trading-stop
        set_tsl_params: Dict[str, Any] = {
            'category': CONFIG.market_type,  # Required
            'symbol': market_id,  # Use exchange-specific market ID
            'trailingStop': trail_percent_str,  # Trailing distance percentage (as string)
            'tpslMode': 'Full',  # Apply to the whole position
            'slTriggerBy': CONFIG.tsl_trigger_by,  # Trigger type for the trail (LastPrice, MarkPrice, IndexPrice)
            # 'activePrice': format_price(symbol, current_price), # Optional: Price to activate the trail immediately. If omitted, Bybit activates when price moves favorably by trail %. Check docs.
            # Recommended: Don't set activePrice here. Let Bybit handle the initial activation based on the best price.
            # To remove the fixed SL and TP when activating TSL, Bybit V5 documentation indicates setting 'stopLoss' and 'takeProfit' to "" (empty string).
            # Setting to "" is often safer to explicitly indicate removal.
            'stopLoss': '',  # Remove the fixed SL when activating TSL
            'takeProfit': '', # Remove the fixed TP when activating TSL (assuming fixed TP should be removed)
            'side': 'Buy' if position_side == 'long' else 'Sell'  # Add side parameter (Bybit V5 expects "Buy"/"Sell" for side)
            # 'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(f"[{symbol}] Setting Position TSL: Trail={trail_percent_str}%, TriggerBy={CONFIG.tsl_trigger_by}, Side={set_tsl_params['side']}, Removing Fixed SL/TP")
        logger.debug(f"[{symbol}] Set TSL Params (for setTradingStop): {set_tsl_params}")

        tsl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                tsl_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_tsl_params)
            else:
                logger.error(Fore.RED + f"[{symbol}] Cannot set TSL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                # Cannot proceed safely, log failure but don't trigger emergency close (position is still protected by initial SL if it was set)
                raise ccxt.NotSupported("TSL setting method not available.")

            logger.debug(f"[{symbol}] Set TSL raw response: {tsl_set_response}")

            # Handle potential failure from fetch_with_retries
            if tsl_set_response is None:
                # fetch_with_retries already logged the failure
                raise ccxt.ExchangeError("Set TSL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(tsl_set_response, dict) and isinstance(tsl_set_response.get('info'), dict) and tsl_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"[{symbol}] Trailing Stop Loss successfully activated for {position_side.upper()} position. Trail: {trail_percent_str}%")
                # --- Update Global State ---
                # Set TSL active marker and clear the initial SL/TP markers
                tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                order_tracker[symbol][position_side]["tsl_id"] = tsl_marker_id
                order_tracker[symbol][position_side]["sl_id"] = None  # Remove initial SL marker marker from tracker
                order_tracker[symbol][position_side]["tp_id"] = None # Remove initial TP marker from tracker
                save_state() # <xaiArtifact artifact_id="c2a3767e-3119-420d-92a6-05533a61856c" title="Persist State to File" contentType="python">Snippet 9: Persist State</xaiArtifact>
                logger.info(f"[{symbol}] Updated order tracker: {order_tracker[symbol]}")
                termux_notify("TSL Activated", f"[{symbol}] {position_side.upper()} TSL active.")
                return  # Success

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                error_code = None
                if isinstance(tsl_set_response.get('info'), dict):
                    error_msg = tsl_set_response['info'].get('retMsg', error_msg)
                    error_code = tsl_set_response['info'].get('retCode')
                    error_msg += f" (Code: {error_code})"
                # Check if error was due to trying to remove non-existent SL/TP (might be benign, e.g., SL/TP already hit)
                # Example Bybit code: 110025 = SL/TP order not found or completed
                if error_code == 110025:
                    logger.warning(f"[{symbol}] TSL activation may have succeeded, but received code 110025 (SL/TP not found/completed) when trying to clear fixed SL/TP. Assuming TSL is active and fixed SL/TP was already gone.")
                    # Proceed as if successful, update tracker
                    tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                    order_tracker[symbol][position_side]["tsl_id"] = tsl_marker_id
                    order_tracker[symbol][position_side]["sl_id"] = None
                    order_tracker[symbol][position_side]["tp_id"] = None
                    save_state() # <xaiArtifact artifact_id="c2a3767e-3119-420d-92a6-05533a61856c" title="Persist State to File" contentType="python">Snippet 9: Persist State</xaiArtifact>
                    logger.info(f"[{symbol}] Updated order tracker (assuming TSL active despite code 110025): {order_tracker[symbol]}")
                    termux_notify("TSL Activated*", f"[{symbol}] {position_side.upper()} TSL active (check exchange).")
                    return  # Treat as success for now
                else:
                    raise ccxt.ExchangeError(f"[{symbol}] Failed to activate trailing stop loss. Exchange message: {error_msg}")

        # --- Handle TSL Setting Failures ---
        except (ccxt.ExchangeError, ccxt.InvalidOrder, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied, DecimalException) as e: # Added DecimalException
            # TSL setting failed. Initial SL/TP markers *should* still be in the tracker if they were set initially.
            # Position might be protected by the initial SL/TP, or might be unprotected if initial SL/TP failed.
            logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] Failed to activate TSL: {e}")
            logger.warning(Fore.YELLOW + f"[{symbol}] Position continues with initial SL/TP (if successfully set) or may be UNPROTECTED if initial SL/TP failed. MANUAL INTERVENTION ADVISED if initial SL/TP state is uncertain.")
            send_email("Bot TSL Failed", f"[{symbol}] TSL activation failed: {e}. Position state uncertain.") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
            # Do NOT clear the initial SL/TP markers here. Do not set TSL marker.
            termux_notify("TSL Activation FAILED!", f"[{symbol}] TSL activation failed. Check logs/position.")
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"[{symbol}] Unexpected error activating TSL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + f"[{symbol}] Position continues with initial SL/TP (if successfully set) or may be UNPROTECTED. MANUAL INTERVENTION ADVISED if initial SL/TP state is uncertain.")
            send_email("Bot Unexpected Error", f"[{symbol}] Unexpected error activating TSL: {e}. Position state uncertain.") # <xaiArtifact artifact_id="a9642f64-73c0-41c6-8673-0f018603c872" title="Add Email Notifications" contentType="python">Snippet 10: Email Notifications</xaiArtifact>
            termux_notify("TSL Activation FAILED!", f"[{symbol}] TSL activation failed (unexpected). Check logs/position.")

    else:
        # Profit threshold not met
        sl_status_log = f"SL({initial_sl_marker})" if initial_sl_marker else ""
        tp_status_log = f" TP({active_tp_marker})" if active_tp_marker else ""
        stop_status_log = f"({sl_status_log}{tp_status_log})" if sl_status_log or tp_status_log else "(None!)"
        logger.debug(f"[{symbol}] {position_side.upper()} profit ({profit:.4f} vs Threshold {activation_threshold_points:.4f}) has not crossed TSL activation threshold. Keeping initial stops {stop_status_log}.")
# </xaiArtifact>


# <xaiArtifact artifact_id="11b1b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b" title="Add Support for Multiple Symbols" contentType="python">Snippet 12: Multiple Symbols</xaiArtifact>
# Modified to accept symbol argument and use MARKET_INFOS, order_tracker[symbol]
def print_status_panel(
    symbol: str, # Added symbol argument
    cycle: int, timestamp: Optional[pd.Timestamp], price: Optional[Decimal], indicators: Optional[Dict[str, Decimal]],
    positions: Optional[Dict[str, Dict[str, Any]]], equity: Optional[Decimal], signals: Dict[str, Union[bool, str]],
    total_realized_pnl: Optional[Decimal],
    order_tracker_state: Dict[str, Dict[str, Optional[str]]],  # Pass tracker state snapshot explicitly for this symbol
    # <xaiArtifact artifact_id="03f4e5d6-a7b8-9c0d-1e2f-3a4b5c6d7e8f" title="Add Performance Metrics" contentType="python">Snippet 20: Performance Metrics</xaiArtifact>
    trades_list: List[Dict] # Pass the global trades list
    # </xaiArtifact>
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""

    header_color = Fore.MAGENTA + Style.BRIGHT
    section_color = Fore.CYAN
    value_color = Fore.WHITE
    reset_all = Style.RESET_ALL

    print(header_color + "\n" + "=" * 80)
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else f"{Fore.YELLOW}N/A"
    print(f" {header_color}Symbol: {value_color}{symbol}{header_color} | Cycle: {value_color}{cycle}{header_color} | Timestamp: {value_color}{ts_str}")
    # Equity is account-wide, only show once per cycle or adapt if per-symbol equity tracking is added
    # For multi-symbol, maybe show total equity once, and per-symbol details below.
    # Display equity here for the primary symbol or just show N/A for others if not fetching per symbol.
    # Let's show equity only once per cycle, perhaps tied to the first symbol processed or as a separate call.
    # Passing equity here means we'll show it for each symbol's panel, which is redundant if it's account-wide.
    # Let's keep it as is for now, acknowledging this redundancy in multi-symbol view.
    market_info = get_market_info(symbol)
    quote_currency = market_info.get('settle', 'Quote') if market_info else 'Quote'
    equity_str = f"{equity:.4f} {quote_currency}" if equity is not None and not equity.is_nan() else f"{Fore.YELLOW}N/A"
    print(f" Equity: {Fore.GREEN}{equity_str}" + reset_all)

    # PnL Summary - also account-wide usually, but maybe filtered by symbol if symbol argument was used
    # Display total realized PnL for the lookback period.
    realized_pnl_str = f"{total_realized_pnl:+.4f}" if total_realized_pnl is not None and not total_realized_pnl.is_nan() else f"{Fore.YELLOW}N/A"
    realized_pnl_color = Fore.GREEN if total_realized_pnl is not None and not total_realized_pnl.is_nan() and total_realized_pnl >= 0 else Fore.RED
    print(f" {section_color}Realized PnL (Last {CONFIG.pnl_lookback_days} Days): {realized_pnl_color}{realized_pnl_str}{reset_all}")
    print(header_color + "-" * 80)


    # --- Market & Indicators ---
    # Use .get(..., Decimal('NaN')) for safe access to indicator values
    price_str = f"{price:.4f}" if price is not None and not price.is_nan() else f"{Fore.YELLOW}N/A"
    atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')
    atr_str = f"{atr:.6f}" if not atr.is_nan() else f"{Fore.YELLOW}N/A"

    trend_ema = indicators.get('trend_ema', Decimal('NaN')) if indicators else Decimal('NaN')
    trend_ema_str = f"{trend_ema:.4f}" if not trend_ema.is_nan() else f"{Fore.YELLOW}N/A"

    price_color = Fore.WHITE
    trend_desc = f"{Fore.YELLOW}Trend N/A"
    if price is not None and not price.is_nan() and not trend_ema.is_nan():
        # Use a small buffer for display consistency (e.g., 0.01% of price)
        trend_buffer_display = price.copy_abs() * Decimal('0.0001')  # 0.01% of current price
        if price > trend_ema + trend_buffer_display: price_color = Fore.GREEN; trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema - trend_buffer_display: price_color = Fore.RED; trend_desc = f"{price_color}(Below Trend)"
        else: price_color = Fore.YELLOW; trend_desc = f"{price_color}(At Trend)"

    status_data = [
        [section_color + "Market", value_color + symbol, f"{price_color}{price_str}"],
        [section_color + f"Trend EMA ({indicators.get('trend_ema_period', CONFIG.trend_ema_period)})", f"{value_color}{trend_ema_str}", trend_desc],
        [section_color + f"ATR ({indicators.get('atr_period', CONFIG.atr_period)})", f"{value_color}{atr_str}", ""], # Display ATR period if stored in indicators
    ]

    if CONFIG.strategy == 'ecc_scalp':
        ecc_cycle = indicators.get('ecc_cycle', Decimal('NaN')) if indicators else Decimal('NaN')
        ecc_trigger = indicators.get('ecc_trigger', Decimal('NaN')) if indicators else Decimal('NaN')
        ecc_cycle_str = f"{ecc_cycle:.4f}" if not ecc_cycle.is_nan() else f"{Fore.YELLOW}N/A"
        ecc_trigger_str = f"{ecc_trigger:.4f}" if not ecc_trigger.is_nan() else f"{Fore.YELLOW}N/A"
        ecc_color = Fore.WHITE
        ecc_desc = f"{Fore.YELLOW}ECC N/A"
        if not ecc_cycle.is_nan() and not ecc_trigger.is_nan():
             if ecc_cycle > ecc_trigger: ecc_color = Fore.GREEN; ecc_desc = f"{ecc_color}Bullish Cross"
             elif ecc_cycle < ecc_trigger: ecc_color = Fore.RED; ecc_desc = f"{ecc_color}Bearish Cross"
             else: ecc_color = Fore.YELLOW; ecc_desc = f"{Fore.YELLOW}Aligned"

        status_data.extend([
             [section_color + f"ECC ({indicators.get('ecc_lookback', CONFIG.ecc_lookback)}, {indicators.get('ecc_alpha', CONFIG.ecc_alpha).normalize()})", f"{ecc_color}{ecc_cycle_str} / {ecc_trigger_str}", ecc_desc], # Normalize alpha for cleaner display
        ])

    elif CONFIG.strategy == 'ema_stoch':
        fast_ema = indicators.get('fast_ema', Decimal('NaN')) if indicators else Decimal('NaN')
        slow_ema = indicators.get('slow_ema', Decimal('NaN')) if indicators else Decimal('NaN')
        stoch_k = indicators.get('stoch_k', Decimal('NaN')) if indicators else Decimal('NaN')
        stoch_d = indicators.get('stoch_d', Decimal('NaN')) if indicators else Decimal('NaN')

        fast_ema_str = f"{fast_ema:.4f}" if not fast_ema.is_nan() else f"{Fore.YELLOW}N/A"
        slow_ema_str = f"{slow_ema:.4f}" if not slow_ema.is_nan() else f"{Fore.YELLOW}N/A"
        stoch_k_str = f"{stoch_k:.2f}" if not stoch_k.is_nan() else f"{Fore.YELLOW}N/A"
        stoch_d_str = f"{stoch_d:.2f}" if not stoch_d.is_nan() else f"{Fore.YELLOW}N/A"

        ema_cross_color = Fore.WHITE
        ema_desc = f"{Fore.YELLOW}EMA N/A"
        if not fast_ema.is_nan() and not slow_ema.is_nan():
            if fast_ema > slow_ema: ema_cross_color = Fore.GREEN; ema_desc = f"{ema_cross_color}Bullish Cross"
            elif fast_ema < slow_ema: ema_cross_color = Fore.RED; ema_desc = f"{ema_cross_color}Bearish Cross"
            else: ema
