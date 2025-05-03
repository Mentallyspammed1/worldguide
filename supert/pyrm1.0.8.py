```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.3.0 (Fortified + TP + Stop Confirmation + pandas_ta Fix)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Perpetual Futures
Version: 2.3.0 (Enhanced: Selectable Strategies + Precision + Native SL/TP/TSL + Fortified Config + Pyrmethus Enhancements + Robustness + pandas_ta Fix)

Purpose:
Automates scalping strategies on Bybit USDT Perpetual Futures markets. This script is intended
for educational and experimental purposes, demonstrating concepts like API interaction,
indicator calculation, risk management, and automated order placement.

Key Features:
- Strategy Flexibility: Select from multiple trading strategies via configuration:
    - "DUAL_SUPERTREND": Uses two Supertrend indicators for trend confirmation.
    - "STOCHRSI_MOMENTUM": Combines Stochastic RSI for overbought/oversold signals with a Momentum indicator.
    - "EHLERS_FISHER": Implements the Ehlers Fisher Transform for identifying cyclical turning points.
    - "EMA_CROSS": Uses Exponential Moving Average crossovers (NOTE: Renamed from EHLERS_MA_CROSS to clarify it uses standard EMAs, not true Ehlers MAs).
- Enhanced Precision: Leverages Python's `Decimal` type for critical financial calculations, minimizing floating-point inaccuracies. Robust `safe_decimal_conversion` helper is used.
- Fortified Configuration: Robust loading of settings from environment variables (.env file) with strict type casting, validation, default handling, and clear logging for improved reliability.
- Native Stop-Loss, Take-Profit, & Trailing Stop-Loss: Utilizes Bybit V5 API's exchange-native Stop Loss (fixed, ATR-based), Take Profit (fixed, ATR-based), and Trailing Stop Loss capabilities (percentage-based), placed immediately upon position entry for faster reaction times. Includes post-entry verification that stops are attached.
- Volatility Adaptation: Employs the Average True Range (ATR) indicator to measure market volatility and dynamically adjust the initial Stop Loss and Take Profit distances based on multipliers.
- Optional Confirmation Filters: Includes optional filters based on Volume Spikes (relative to a moving average) and Order Book Pressure (Bid/Ask volume ratio) to potentially improve entry signal quality.
- Sophisticated Risk Management: Implements risk-based position sizing (percentage of equity per trade), incorporates exchange margin requirements checks (estimated, with configurable buffer), and allows setting a maximum position value cap (USDT). Checks against minimum order value.
- Termux Integration: Provides optional SMS alerts via Termux:API for critical events like initialization, errors, order placements, and shutdowns. Includes checks for command availability and robust error handling.
- Robust Operation: Features comprehensive error handling for common CCXT exceptions (network issues, authentication failures, rate limits, exchange errors), data validation (NaN handling in OHLCV and indicators), and detailed logging with vibrant console colors via Colorama. Includes robust `pandas_ta` column identification to prevent indicator calculation errors. Includes retry mechanisms for key API calls.
- Graceful Shutdown: Designed to handle interruptions (Ctrl+C) or critical errors by attempting to cancel open orders and close any existing positions before exiting.
- Bybit V5 API Focused: Tailored logic for interacting with the Bybit V5 API, particularly regarding position detection (One-Way Mode), order parameters (e.g., 'category'), and native stop placement/confirmation.

##################################################################################
#                                  !!! WARNING !!!                               #
#                                                                                #
#        ~~~ TRADING CRYPTOCURRENCY FUTURES WITH LEVERAGE IS EXTREMELY RISKY ~~~ #
#                                                                                #
# - HIGH RISK OF LOSS: You can lose your entire investment rapidly, and          #
#   potentially more than your initial deposit (liquidation).                    #
# - EDUCATIONAL USE ONLY: This script is provided for educational and            #
#   experimental purposes ONLY. It is NOT financial advice.                      #
# - NO GUARANTEES: Past performance is not indicative of future results.         #
#   Profitability is NOT guaranteed. Default parameters are EXAMPLES ONLY        #
#   and likely NOT profitable without extensive testing and optimization.        #
# - USE AT YOUR OWN RISK: The authors and contributors assume NO responsibility  #
#   for any financial losses incurred using this software.                       #
# - NATIVE STOP RELIANCE: Protective stops (SL/TP/TSL) rely on Bybit's           #
#   exchange execution, which is subject to slippage, market volatility,         #
#   liquidity, and API/exchange performance. Execution at the exact trigger      #
#   price is NOT GUARANTEED.                                                     #
# - TEST THOROUGHLY: **NEVER** run this bot with real funds without extensive    #
#   and successful testing on a TESTNET or DEMO account. Understand the code,    #
#   its logic, risks, and dependencies fully before ANY live deployment.         #
# - API LIMITS & BANS: Excessive API usage can lead to temporary or permanent    #
#   bans. Monitor usage and adjust sleep times if necessary.                     #
# - TERMUX DEPENDENCY: SMS alerts require Termux and Termux:API on Android.      #
# - UPDATES REQUIRED: Exchange APIs and libraries (CCXT, pandas_ta) change.      #
#   This code may require updates to remain functional. Keep libraries updated.  #
#                                                                                #
#                  ~~~ PROCEED WITH EXTREME CAUTION AND AWARENESS ~~~            #
##################################################################################

Disclaimer:
(Content moved into the WARNING block above for higher visibility)
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import sys
import time
import traceback
import subprocess
import shlex # For safe command argument parsing in logging
import shutil # For checking command existence (e.g., termux-sms-send)
from typing import Dict, Optional, Any, Tuple, List, Union, Callable, Type # Enhanced type hinting
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation, DivisionByZero, ROUND_DOWN

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta # type: ignore[import] # pandas_ta might lack complete type hints
    from dotenv import load_dotenv
    from colorama import init as colorama_init, Fore, Style, Back
except ImportError as e:
    missing_pkg = e.name
    # Use Colorama's raw codes here as it might not be initialized yet
    print(f"\033[91m\033[1mCRITICAL ERROR: Missing essential spell component: {missing_pkg}\033[0m") # Bold Bright Red
    print("\033[93mTo conjure it, cast the following spell in your terminal (e.g., Termux, Linux, macOS):\033[0m") # Bright Yellow
    print(f"\033[1m\033[96mpip install {missing_pkg}\033[0m") # Bold Bright Cyan
    print("\n\033[96mOr, to ensure all scrolls are present, cast:\033[0m") # Bright Cyan
    print("\033[1m\033[96mpip install ccxt pandas pandas_ta python-dotenv colorama\033[0m")
    print("\n\033[93mVerify installation with: \033[1m\033[96mpip list | grep -E 'ccxt|pandas|pandas-ta|python-dotenv|colorama'\033[0m")
    sys.exit(1) # Exit immediately if core dependencies are missing

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True) # Activate Colorama's magic for vibrant logs
load_dotenv() # Load secrets from the hidden .env scroll (if present)
getcontext().prec = 30 # Set Decimal precision for financial exactitude (increased for intermediate calcs, adjust if needed)

# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """
    Loads, validates, and stores configuration parameters from environment variables.
    Provides robust type casting, default value handling, validation, and logging.
    Ensures critical parameters are present and valid before proceeding.
    """
    # --- API Credentials ---
    api_key: Optional[str]
    api_secret: Optional[str]

    # --- Trading Parameters ---
    symbol: str
    interval: str
    leverage: int
    sleep_seconds: int

    # --- Strategy Selection ---
    strategy_name: str
    valid_strategies: List[str]

    # --- Risk Management ---
    risk_per_trade_percentage: Decimal
    atr_stop_loss_multiplier: Decimal
    atr_take_profit_multiplier: Decimal
    max_order_usdt_amount: Decimal
    required_margin_buffer: Decimal

    # --- Native Stop-Loss & Trailing Stop-Loss ---
    trailing_stop_percentage: Decimal
    trailing_stop_activation_offset_percent: Decimal

    # --- Strategy-Specific Parameters ---
    # Dual Supertrend
    st_atr_length: int
    st_multiplier: Decimal
    confirm_st_atr_length: int
    confirm_st_multiplier: Decimal
    # StochRSI + Momentum
    stochrsi_rsi_length: int
    stochrsi_stoch_length: int
    stochrsi_k_period: int
    stochrsi_d_period: int
    stochrsi_overbought: Decimal
    stochrsi_oversold: Decimal
    momentum_length: int
    # Ehlers Fisher Transform
    ehlers_fisher_length: int
    ehlers_fisher_signal_length: int
    # EMA Cross
    ema_fast_period: int
    ema_slow_period: int

    # --- Confirmation Filters ---
    # Volume Analysis
    volume_ma_period: int
    volume_spike_threshold: Decimal
    require_volume_spike_for_entry: bool
    # Order Book Analysis
    order_book_depth: int
    order_book_ratio_threshold_long: Decimal
    order_book_ratio_threshold_short: Decimal
    fetch_order_book_per_cycle: bool

    # --- ATR Calculation ---
    atr_calculation_period: int

    # --- Termux SMS Alerts ---
    enable_sms_alerts: bool
    sms_recipient_number: Optional[str]
    sms_timeout_seconds: int

    # --- CCXT / API Parameters ---
    default_recv_window: int
    order_book_fetch_limit: int
    order_fill_timeout_seconds: int
    stop_attach_confirm_attempts: int
    stop_attach_confirm_delay_seconds: int

    # --- Internal Constants ---
    side_buy: str
    side_sell: str
    pos_long: str
    pos_short: str
    pos_none: str
    usdt_symbol: str
    retry_count: int
    retry_delay_seconds: int
    api_fetch_limit_buffer: int
    position_qty_epsilon: Decimal
    post_close_delay_seconds: int
    min_order_value_usdt: Decimal

    def __init__(self):
        """Initializes configuration by loading and validating environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED, secret=True)
        self.api_secret = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED, secret=True)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW) # Target market (CCXT unified format, e.g., 'BTC/USDT:USDT')
        self.interval = self._get_env("INTERVAL", "1m", color=Fore.YELLOW) # Chart timeframe (e.g., '1m', '5m', '1h')
        self.leverage = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW) # Desired leverage multiplier
        self.sleep_seconds = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW) # Pause between trading cycles (seconds)

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        # NOTE: Renamed EHLERS_MA_CROSS to EMA_CROSS for clarity as it uses standard EMAs.
        self.valid_strategies = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EMA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}")
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}")
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.005 = 0.5% of equity per trade
        self.atr_stop_loss_multiplier = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN) # Multiplier for ATR to set initial fixed SL distance
        self.atr_take_profit_multiplier = self._get_env("ATR_TAKE_PROFIT_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN) # Multiplier for ATR to set initial fixed TP distance
        self.max_order_usdt_amount = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN) # Maximum position value in USDT (overrides risk calc if needed)
        self.required_margin_buffer = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN) # e.g., 1.05 = Require 5% more free margin than estimated for order placement

        # --- Native Stop-Loss & Trailing Stop-Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        self.trailing_stop_percentage = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.005 = 0.5% trailing distance from high/low water mark
        self.trailing_stop_activation_offset_percent = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.001 = 0.1% price movement in profit before TSL becomes active (using 'activePrice' param)

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN) # Primary Supertrend ATR period
        self.st_multiplier = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN) # Primary Supertrend ATR multiplier
        self.confirm_st_atr_length = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN) # Confirmation Supertrend ATR period
        self.confirm_st_multiplier = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN) # Confirmation Supertrend ATR multiplier
        # StochRSI + Momentum
        self.stochrsi_rsi_length = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN) # StochRSI: RSI period
        self.stochrsi_stoch_length = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN) # StochRSI: Stochastic period
        self.stochrsi_k_period = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN) # StochRSI: %K smoothing period
        self.stochrsi_d_period = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN) # StochRSI: %D smoothing period (signal line)
        self.stochrsi_overbought = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN) # StochRSI overbought threshold
        self.stochrsi_oversold = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN) # StochRSI oversold threshold
        self.momentum_length = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN) # Momentum indicator period
        # Ehlers Fisher Transform
        self.ehlers_fisher_length = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN) # Fisher Transform calculation period
        self.ehlers_fisher_signal_length = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN) # Fisher Transform signal line period (1 usually means no separate signal line smoothing)
        # EMA Cross (Uses standard EMAs, NOT Ehlers Super Smoother)
        self.ema_fast_period = self._get_env("EMA_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN) # Fast EMA period
        self.ema_slow_period = self._get_env("EMA_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN) # Slow EMA period

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        # Volume Analysis
        self.volume_ma_period = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW) # Moving average period for volume
        self.volume_spike_threshold = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW) # Multiplier over MA to consider a 'spike' (e.g., 1.5 = 150% of MA)
        self.require_volume_spike_for_entry = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW) # Require volume spike for entry signal?
        # Order Book Analysis
        self.order_book_depth = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW) # Number of bid/ask levels to analyze for ratio
        self.order_book_ratio_threshold_long = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW) # Min Bid/Ask volume ratio for long confirmation
        self.order_book_ratio_threshold_short = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW) # Max Bid/Ask volume ratio for short confirmation (ratio = Total Bid Vol / Total Ask Vol within depth)
        self.fetch_order_book_per_cycle = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW) # Fetch OB every cycle (more API calls) or only when needed for entry confirmation?

        # --- ATR Calculation (for Initial SL/TP) ---
        self.atr_calculation_period = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN) # Period for ATR calculation used in SL/TP

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA) # Enable/disable SMS alerts globally
        self.sms_recipient_number = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA, required=False) # Recipient phone number for alerts (optional)
        self.sms_timeout_seconds = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA) # Max time to wait for SMS command execution (seconds)

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window = self._get_env("CCXT_RECV_WINDOW", 10000, cast_type=int, color=Fore.WHITE) # Milliseconds for API request validity (Bybit default 5000, increased for potential latency)
        self.order_book_fetch_limit = max(25, self.order_book_depth) # How many levels to fetch (ensure >= depth needed, common limits are 25, 50, 100, 200)
        self.order_fill_timeout_seconds = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW) # Max time to wait for market order fill confirmation (seconds)
        self.stop_attach_confirm_attempts = self._get_env("STOP_ATTACH_CONFIRM_ATTEMPTS", 3, cast_type=int, color=Fore.YELLOW) # Attempts to confirm native stops are attached to position after entry
        self.stop_attach_confirm_delay_seconds = self._get_env("STOP_ATTACH_CONFIRM_DELAY_SECONDS", 1, cast_type=int, color=Fore.YELLOW) # Delay between attempts to confirm stops

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy = "buy"       # CCXT standard side for buying
        self.side_sell = "sell"     # CCXT standard side for selling
        self.pos_long = "Long"      # Internal representation for a long position
        self.pos_short = "Short"    # Internal representation for a short position
        self.pos_none = "None"      # Internal representation for no position (flat)
        self.usdt_symbol = "USDT"   # The stablecoin quote currency symbol used by Bybit
        self.retry_count = 3        # Default attempts for certain retryable API calls (e.g., setting leverage)
        self.retry_delay_seconds = 2 # Default pause between retries (seconds)
        self.api_fetch_limit_buffer = 50 # Extra candles to fetch beyond strict indicator needs, providing a safety margin (Increased buffer)
        self.position_qty_epsilon = Decimal("1e-9") # Small value for float/decimal comparisons involving position size to handle precision issues
        self.post_close_delay_seconds = 3 # Brief pause after successfully closing a position (seconds) to allow exchange state to potentially settle
        self.min_order_value_usdt = Decimal("1.0") # Minimum order value in USDT (Bybit default is often 1 USDT for perpetuals, adjust if needed based on market)

        # --- Post-Initialization Validation ---
        self._validate_parameters()

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    def _validate_parameters(self):
        """Performs comprehensive validation checks on loaded parameters."""
        logger.debug("Validating configuration parameters...")
        errors = []
        warnings = []

        # Type checks are implicitly handled by _get_env's casting, focus on logical constraints
        if self.leverage <= 0:
            errors.append("LEVERAGE must be a positive integer.")
        if self.risk_per_trade_percentage <= 0 or self.risk_per_trade_percentage >= 1:
            errors.append("RISK_PER_TRADE_PERCENTAGE must be between 0 and 1 (exclusive, e.g., 0.005 for 0.5%).")
        if self.atr_stop_loss_multiplier <= 0:
            errors.append("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if self.atr_take_profit_multiplier <= 0:
             errors.append("ATR_TAKE_PROFIT_MULTIPLIER must be positive.")
        if self.trailing_stop_percentage < 0 or self.trailing_stop_percentage >= 1: # TSL can be 0 to disable
            errors.append("TRAILING_STOP_PERCENTAGE must be between 0 (disabled) and 1 (exclusive).")
        if self.trailing_stop_activation_offset_percent < 0:
             errors.append("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT cannot be negative.")
        if self.max_order_usdt_amount < 0:
             errors.append("MAX_ORDER_USDT_AMOUNT cannot be negative.")
        if self.required_margin_buffer < 1:
            errors.append("REQUIRED_MARGIN_BUFFER must be >= 1 (e.g., 1.05 for 5% buffer).")
        if self.stop_attach_confirm_attempts < 1:
             errors.append("STOP_ATTACH_CONFIRM_ATTEMPTS must be at least 1.")
        if self.stop_attach_confirm_delay_seconds < 0:
             errors.append("STOP_ATTACH_CONFIRM_DELAY_SECONDS cannot be negative.")
        if self.order_fill_timeout_seconds < 1:
             errors.append("ORDER_FILL_TIMEOUT_SECONDS must be at least 1.")
        if self.atr_calculation_period <= 0:
             errors.append("ATR_CALCULATION_PERIOD must be positive.")
        if self.volume_ma_period <= 0:
             errors.append("VOLUME_MA_PERIOD must be positive.")
        if self.order_book_depth <= 0:
             errors.append("ORDER_BOOK_DEPTH must be positive.")
        if self.sms_timeout_seconds < 1:
             errors.append("SMS_TIMEOUT_SECONDS must be positive.")
        if self.sleep_seconds <= 0:
             errors.append("SLEEP_SECONDS must be positive.")

        # Warnings for potentially problematic settings
        if self.enable_sms_alerts and not self.sms_recipient_number:
             warnings.append("SMS alerts enabled (ENABLE_SMS_ALERTS=true) but SMS_RECIPIENT_NUMBER is not set. Alerts will not be sent.")
        if self.sleep_seconds < 5:
             warnings.append(f"SLEEP_SECONDS ({self.sleep_seconds}) is very low (< 5s). This significantly increases the risk of hitting API rate limits. Monitor usage closely.")
        if not self.symbol or "/" not in self.symbol or ":" not in self.symbol:
             errors.append(f"SYMBOL '{self.symbol}' does not appear to be a valid CCXT unified symbol format (e.g., BASE/QUOTE:SETTLE like BTC/USDT:USDT).")
        # Strategy specific validations
        if self.strategy_name == "EMA_CROSS" and self.ema_fast_period >= self.ema_slow_period:
             errors.append(f"EMA_CROSS strategy requires EMA_FAST_PERIOD ({self.ema_fast_period}) to be less than EMA_SLOW_PERIOD ({self.ema_slow_period}).")
        if self.strategy_name == "STOCHRSI_MOMENTUM" and self.stochrsi_oversold >= self.stochrsi_overbought:
             errors.append(f"STOCHRSI_MOMENTUM strategy requires STOCHRSI_OVERSOLD ({self.stochrsi_oversold}) to be less than STOCHRSI_OVERBOUGHT ({self.stochrsi_overbought}).")

        # Log warnings
        for warning in warnings:
            logger.warning(f"{Fore.YELLOW}Config Validation Warning: {warning}{Style.RESET_ALL}")

        # Log errors and raise if any critical issues found
        if errors:
            for error in errors:
                logger.critical(f"{Back.RED}{Fore.WHITE}Config Validation Error: {error}{Style.RESET_ALL}")
            raise ValueError("Configuration validation failed. Please check the .env file and error messages above.")

        logger.debug("Configuration parameter validation complete.")

    def _get_env(self, key: str, default: Any = None, cast_type: Type = str, required: bool = False, color: str = Fore.WHITE, secret: bool = False) -> Any:
        """
        Fetches an environment variable, performs robust type casting (including defaults),
        logs the process, handles required variables, and masks secrets in logs.

        Args:
            key: The environment variable name.
            default: The default value to use if the variable is not set.
            cast_type: The target type to cast the value to (e.g., int, Decimal, bool, str).
            required: If True, raises ValueError if the variable is not set AND no default is provided.
            color: Colorama Fore color for logging this parameter.
            secret: If True, masks the value in log messages.

        Returns:
            The value from the environment variable or default, cast to the specified type.

        Raises:
            ValueError: If a required variable is missing and no default, or if casting fails critically.
        """
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None

        log_value = lambda v: "*******" if secret and v is not None else v

        if value_str is None:
            if required and default is None:
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' not found and no default specified.{Style.RESET_ALL}")
                raise ValueError(f"Required environment variable '{key}' not set and no default provided.")
            elif default is not None:
                # Use the default (whether required or not) if env var not set
                log_prefix = "Required rune" if required else "Summoning"
                log_reason = "Not Set. Using Required Default:" if required else "Not Set. Using Default:"
                logger.debug(f"{color}{log_prefix} {key}: {log_reason} '{log_value(default)}'{Style.RESET_ALL}")
                value_to_cast = default
                source = "Default" # Simplified source name
            else: # Not required and no default
                 value_to_cast = None
                 source = "None (Not Required, No Default)"
                 logger.debug(f"{color}Summoning {key}: Not Set. No Default Provided.{Style.RESET_ALL}")

        else:
            # Env var was found
            logger.debug(f"{color}Summoning {key}: Found Env Value: '{log_value(value_str)}'{Style.RESET_ALL}")
            value_to_cast = value_str

        # --- Attempt Casting (applies to both env var value and default value) ---
        if value_to_cast is None and not required and default is None:
            # This handles cases where: Not required, env var not set, no default -> returns None
            logger.debug(f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}")
            return None

        final_value: Any = None
        try:
            # Use str() first for consistent handling before type-specific parsing
            raw_value_str = str(value_to_cast).strip()

            if cast_type == bool:
                if raw_value_str.lower() in ['true', '1', 'yes', 'y', 'on']:
                    final_value = True
                elif raw_value_str.lower() in ['false', '0', 'no', 'n', 'off']:
                    final_value = False
                else:
                    raise ValueError(f"Invalid boolean value '{raw_value_str}'. Expected true/false, yes/no, 1/0, on/off.")
            elif cast_type == Decimal:
                # Handle potential empty string after strip
                if raw_value_str == "": raise InvalidOperation(f"Empty string cannot be converted to Decimal for key '{key}'.")
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                # Handle potential empty string after strip
                if raw_value_str == "": raise ValueError(f"Empty string cannot be converted to int for key '{key}'.")
                # Cast via Decimal first to handle potential float strings like "10.0" -> 10 gracefully
                # Also handles scientific notation like "1e-5" which int() would fail on directly
                try:
                    dec_val = Decimal(raw_value_str)
                    # Check if it represents a whole number
                    if dec_val == dec_val.to_integral_value(rounding=ROUND_HALF_UP):
                        final_value = int(dec_val)
                    else:
                        raise ValueError(f"Value '{raw_value_str}' is not a whole number for int conversion (Key: '{key}').")
                except InvalidOperation:
                    raise ValueError(f"Invalid numeric value '{raw_value_str}' for int conversion (Key: '{key}').")
            elif cast_type == float:
                 # Handle potential empty string after strip
                 if raw_value_str == "": raise ValueError(f"Empty string cannot be converted to float for key '{key}'.")
                 final_value = float(raw_value_str) # Use float directly
            elif cast_type == str:
                final_value = raw_value_str # Keep as string
            else:
                # Should not happen if cast_type is a standard type, but handles extensions
                logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw value from {source}.")
                final_value = value_to_cast # Return original value if type is unknown

        except (ValueError, TypeError, InvalidOperation) as e:
            # Casting failed! Log error and attempt to use default, casting it carefully.
            logger.error(f"{Fore.RED}Invalid type/value for {key}: '{log_value(value_to_cast)}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Attempting to use default '{log_value(default)}'.{Style.RESET_ALL}")

            if default is None:
                 if required:
                     logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast value for required key '{key}' and default is None.{Style.RESET_ALL}")
                     raise ValueError(f"Required env var '{key}' failed casting and has no valid default.")
                 else:
                     logger.warning(f"{Fore.YELLOW}Casting failed for optional {key}, default is None. Final value: None{Style.RESET_ALL}")
                     return None # Return None if casting fails and no default for optional var
            else:
                # Try casting the default value itself
                source = "Default (Fallback)"
                logger.debug(f"Attempting to cast fallback default value '{log_value(default)}' for key '{key}' to {cast_type.__name__}")
                try:
                    # Repeat the casting logic for the default value
                    default_str = str(default).strip()
                    if cast_type == bool:
                        if default_str.lower() in ['true', '1', 'yes', 'y', 'on']: final_value = True
                        elif default_str.lower() in ['false', '0', 'no', 'n', 'off']: final_value = False
                        else: raise ValueError(f"Invalid boolean default value '{default_str}'")
                    elif cast_type == Decimal:
                         if default_str == "": raise InvalidOperation("Empty string cannot be converted to Decimal from default.")
                         final_value = Decimal(default_str)
                    elif cast_type == int:
                         if default_str == "": raise ValueError("Empty string cannot be converted to int from default.")
                         try:
                             dec_val = Decimal(default_str)
                             if dec_val == dec_val.to_integral_value(rounding=ROUND_HALF_UP): final_value = int(dec_val)
                             else: raise ValueError(f"Default value '{default_str}' is not a whole number for int conversion.")
                         except InvalidOperation: raise ValueError(f"Invalid numeric default value '{default_str}' for int conversion.")
                    elif cast_type == float:
                         if default_str == "": raise ValueError("Empty string cannot be converted to float from default.")
                         final_value = float(default_str)
                    elif cast_type == str: final_value = default_str
                    else: final_value = default # Fallback to raw default if type unknown

                    logger.warning(f"{Fore.YELLOW}Successfully used casted default value for {key}: '{log_value(final_value)}'{Style.RESET_ALL}")
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast BOTH provided value ('{log_value(value_to_cast)}') AND default value ('{log_value(default)}') for key '{key}' to {cast_type.__name__}. Error on default: {e_default}{Style.RESET_ALL}")
                    raise ValueError(f"Configuration error: Cannot cast value or default for key '{key}' to {cast_type.__name__}.")

        # Log the final type and value being used
        logger.debug(f"{color}Using final value for {key}: {log_value(final_value)} (Type: {type(final_value).__name__}) (Source: {source}){Style.RESET_ALL}")
        return final_value


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL_STR: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL_MAP: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": 25, # Custom level (between INFO and WARNING)
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Set default level to INFO if the env var value is invalid
LOGGING_LEVEL: int = LOGGING_LEVEL_MAP.get(LOGGING_LEVEL_STR, logging.INFO)

# Define custom SUCCESS level if it doesn't exist
SUCCESS_LEVEL: int = LOGGING_LEVEL_MAP["SUCCESS"]
if logging.getLevelName(SUCCESS_LEVEL) == f"Level {SUCCESS_LEVEL}": # Check if name is default
    logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# Define the success method for the logger instance
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a message with severity 'SUCCESS'."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

# Add the method to the Logger class (careful with type hinting here)
# Check if method already exists to prevent potential issues if run multiple times
if not hasattr(logging.Logger, 'success'):
    logging.Logger.success = log_success # type: ignore[attr-defined]

# Custom Formatter with Colors
class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds colors based on log level."""
    level_colors = {
        logging.DEBUG: Fore.CYAN + Style.DIM,
        logging.INFO: Fore.BLUE,
        SUCCESS_LEVEL: Fore.MAGENTA + Style.BRIGHT, # Use Bright Magenta for Success
        logging.WARNING: Fore.YELLOW + Style.BRIGHT,
        logging.ERROR: Fore.RED + Style.BRIGHT,
        logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
    }
    base_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    def format(self, record):
        log_color = self.level_colors.get(record.levelno, Fore.WHITE) # Default to white
        level_name = record.levelname
        # Apply color only to level name part for better readability
        record.levelname = f"{log_color}{level_name}{Style.RESET_ALL}"
        # Format the entire message using the base formatter
        formatter = logging.Formatter(self.base_format, datefmt=self.date_format)
        formatted_message = formatter.format(record)
        # Restore original levelname in case record is reused (though unlikely here)
        record.levelname = level_name
        # Add reset code at the end of the message to ensure color doesn't leak
        return formatted_message + Style.RESET_ALL

# Configure logging using the custom formatter
logger: logging.Logger = logging.getLogger(__name__) # Get the root logger
logger.setLevel(LOGGING_LEVEL)

# Create handler (StreamHandler for console output)
handler = logging.StreamHandler(sys.stdout)

# Set formatter based on whether output is a TTY
if sys.stdout.isatty():
    formatter = ColorFormatter()
else:
    formatter = logging.Formatter(ColorFormatter.base_format, datefmt=ColorFormatter.date_format)

handler.setFormatter(formatter)

# Add handler to the logger (remove existing default handlers if any)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)
# Prevent propagation to root logger if it has default handlers
logger.propagate = False


# --- Global Objects - Instantiated Arcana ---
# Define CONFIG as Optional initially, will be assigned after successful init
CONFIG: Optional[Config] = None
# Define exchange as Optional for graceful shutdown handling
exchange_instance: Optional[ccxt.Exchange] = None

try:
    # --- Termux SMS Alert Function (Defined early for use in Config error) ---
    _termux_sms_command_exists: Optional[bool] = None # Cache the result of checking command existence

    def send_sms_alert(message: str) -> bool:
        """
        Sends an SMS alert using the 'termux-sms-send' command, if enabled and available.
        Checks for command existence once and caches the result.
        Safely accesses necessary config values even if CONFIG is partially loaded or None.

        Args:
            message: The text message to send.

        Returns:
            True if the SMS command was executed successfully (return code 0), False otherwise.
        """
        global _termux_sms_command_exists, CONFIG # Allow modification of cache flag

        # Safely get config values needed for SMS, using defaults if CONFIG is missing/partial
        enable_sms = False
        recipient = None
        timeout = 30
        symbol_for_msg = "UNKNOWN_SYMBOL"
        strategy_for_msg = "UNKNOWN_STRATEGY"

        if CONFIG is not None:
            # Use getattr for safety in case CONFIG init was partial (though unlikely now)
            enable_sms = getattr(CONFIG, 'enable_sms_alerts', False)
            recipient = getattr(CONFIG, 'sms_recipient_number', None)
            timeout = getattr(CONFIG, 'sms_timeout_seconds', 30)
            symbol_for_msg = getattr(CONFIG, 'symbol', "UNKNOWN_SYMBOL")
            strategy_for_msg = getattr(CONFIG, 'strategy_name', "UNKNOWN_STRATEGY")
        else:
            # Fallback to direct environment variable access if CONFIG failed very early
            enable_sms_str = os.getenv("ENABLE_SMS_ALERTS", "false").lower()
            enable_sms = enable_sms_str in ['true', '1', 'yes', 'y', 'on']
            recipient = os.getenv("SMS_RECIPIENT_NUMBER")
            timeout_str = os.getenv("SMS_TIMEOUT_SECONDS", "30")
            try:
                timeout = int(timeout_str)
            except (ValueError, TypeError):
                timeout = 30 # Fallback to default if env var is invalid
            symbol_for_msg = os.getenv("SYMBOL", "UNKNOWN_SYMBOL")
            strategy_for_msg = os.getenv("STRATEGY_NAME", "UNKNOWN_STRATEGY")


        if not enable_sms:
            logger.debug("SMS alerts disabled by configuration.")
            return False

        # Check for command existence only once per script run
        if _termux_sms_command_exists is None:
            termux_command_path = shutil.which('termux-sms-send')
            _termux_sms_command_exists = termux_command_path is not None
            if not _termux_sms_command_exists:
                # Log warning only once
                logger.warning(f"{Fore.YELLOW}SMS alerts enabled, but 'termux-sms-send' command not found in PATH. "
                               f"Ensure Termux:API is installed (`pkg install termux-api`) and PATH is configured correctly.{Style.RESET_ALL}")
            else:
                logger.info(f"SMS alerts enabled. Found 'termux-sms-send' command at: {termux_command_path}")

        if not _termux_sms_command_exists:
            return False # Don't proceed if command is missing

        # Ensure recipient number is configured
        if not recipient:
            # Warning should have been logged during config validation if number is missing while enabled
            logger.debug("SMS recipient number not configured, cannot send alert.")
            return False

        try:
            # Prepare the command spell. Using a list of arguments is safer.
            # Prepend bot identifier to the message for context
            full_message = f"[Pyrmethus/{strategy_for_msg}/{symbol_for_msg}] {message}"
            # Limit message length if necessary (SMS limits vary, ~160 chars is safe)
            max_sms_len = 155 # Leave room for identifier
            if len(full_message) > max_sms_len:
                full_message = full_message[:max_sms_len] + "..."

            command: List[str] = ['termux-sms-send', '-n', recipient, full_message]
            logger.info(f"{Fore.MAGENTA}Dispatching SMS whisper to {recipient} (Timeout: {timeout}s)...{Style.RESET_ALL}")
            logger.debug(f"Executing command: {' '.join(shlex.quote(arg) for arg in command)}") # Log the command safely

            # Execute the spell via subprocess with timeout and output capture
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,          # Decode stdout/stderr as text
                check=False,        # Don't raise exception on non-zero exit code
                timeout=timeout
            )

            if result.returncode == 0:
                logger.success(f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}")
                if result.stdout: logger.debug(f"SMS Send stdout: {result.stdout.strip()}")
                if result.stderr: logger.warning(f"{Fore.YELLOW}SMS Send stderr (on success): {result.stderr.strip()}{Style.RESET_ALL}")
                return True
            else:
                error_details = result.stderr.strip() if result.stderr else "No stderr output"
                logger.error(f"{Fore.RED}SMS whisper failed. Return Code: {result.returncode}, Stderr: {error_details}{Style.RESET_ALL}")
                if result.stdout: logger.error(f"SMS Send stdout (on error): {result.stdout.strip()}")
                return False
        except FileNotFoundError:
            logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' command vanished unexpectedly? Ensure Termux:API is installed.{Style.RESET_ALL}")
            _termux_sms_command_exists = False # Update cache
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"{Fore.RED}SMS failed: Command timed out after {timeout}s.{Style.RESET_ALL}")
            return False
        except Exception as e:
            logger.error(f"{Fore.RED}SMS failed: Unexpected disturbance during dispatch: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            return False

    # --- Initialize Configuration ---
    CONFIG = Config() # Forge the configuration object from environment variables

except (ValueError, Exception) as config_error:
    # Error should have been logged within Config init or _get_env
    logger.critical(f"{Back.RED}{Fore.WHITE}Configuration loading failed. Cannot continue spellcasting. Error: {config_error}{Style.RESET_ALL}")
    logger.debug(traceback.format_exc()) # Log full traceback for debugging

    # Attempt SMS alert about the config error using the defined function
    send_sms_alert(f"CRITICAL CONFIG ERROR: {config_error}. Bot failed to start.")
    sys.exit(1)

# Check if CONFIG is None after try-except (shouldn't happen if Config() succeeded)
if CONFIG is None:
     logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Configuration object is None after initialization block. Exiting.{Style.RESET_ALL}")
     sys.exit(1)


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Optional[Decimal] = Decimal("NaN")) -> Optional[Decimal]:
    """
    Safely converts a value to a Decimal, handling None, pandas NA, empty strings,
    and potential errors. Returns the specified default value on failure.

    Args:
        value: The value to convert (can be string, float, int, Decimal, None, pandas NA, etc.).
        default: The Decimal value (or None) to return if conversion fails or input is
                 None/NA/empty. Defaults to Decimal('NaN'). Use Decimal('0.0') if
                 zero fallback is desired, or None if None should be returned on failure.

    Returns:
        The converted Decimal value or the default (Decimal('NaN'), Decimal('0.0'), or None).
    """
    if value is None or pd.isna(value):
        # logger.debug(f"safe_decimal_conversion: Input is None/NA, returning default {default}")
        return default
    try:
        # Using str(value) handles various input types more reliably before Decimal conversion
        # Strip whitespace in case value is a string from env var or similar
        str_value = str(value).strip()
        if str_value == "":
             # logger.debug(f"safe_decimal_conversion: Input is empty string, returning default {default}")
             return default # Treat empty string like None/NA

        # Attempt conversion
        result = Decimal(str_value)

        # Check for infinity/NaN explicitly as they can cause issues downstream
        if result.is_infinite() or result.is_nan():
             # logger.warning(f"safe_decimal_conversion: Converted '{value}' resulted in {result}. Returning default {default}.")
             return default
        return result
    except (InvalidOperation, TypeError, ValueError) as e:
        # Log a debug message, as this can happen frequently with non-numeric data
        # logger.debug(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}. Error: {e}")
        return default
    except Exception as e_unexp: # Catch unexpected errors during conversion
        logger.warning(f"Unexpected error converting '{value}' to Decimal: {e_unexp}. Returning default {default}.")
        return default


def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """Returns the last 6 characters of an order ID for concise logging, or 'N/A'."""
    if order_id:
        order_id_str = str(order_id)
        # Handle potential UUIDs or other long IDs, take last 6 chars
        return f"...{order_id_str[-6:]}" if len(order_id_str) > 6 else order_id_str
    return "N/A"

# --- Precision Formatting - Shaping the Numbers for the Exchange ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal, int, str]) -> str:
    """
    Formats a price according to the exchange's market precision rules using CCXT.
    Includes safety checks and fallbacks.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        price: The price value (float, Decimal, int, or string representation).

    Returns:
        The price formatted as a string according to market precision.
        Returns 'NaN' string on critical failure or if input converts to NaN.
        Uses Decimal.normalize() as a fallback if CCXT formatting fails subtly.
    """
    price_decimal = safe_decimal_conversion(price, default=Decimal('NaN')) # Ensure default is NaN

    # Handle NaN or invalid inputs early
    if price_decimal is None or price_decimal.is_nan():
        logger.error(f"{Fore.RED}Error shaping price: Input '{price}' converted to NaN or None. Cannot format.{Style.RESET_ALL}")
        return 'NaN'

    # Handle exact zero case explicitly
    if price_decimal.is_zero():
         # Check market precision for zero formatting if available (though "0" is usually safe)
         try:
            if symbol in exchange.markets:
                # Use price_to_precision even for zero to get correct decimal places (e.g., "0.00")
                return exchange.price_to_precision(symbol, 0.0)
         except Exception:
             pass # Fallback to simple "0" if precision check fails
         return "0"

    try:
        # Ensure the market is loaded
        if symbol not in exchange.markets:
            logger.warning(f"Market {symbol} not loaded in format_price. Attempting to load.")
            exchange.load_markets(True) # Force reload
        if symbol not in exchange.markets:
             raise ccxt.BadSymbol(f"Market {symbol} could not be loaded for formatting.")

        # CCXT formatting methods typically expect float input
        # Use float conversion carefully, potential precision loss for very large/small numbers
        try:
            price_float = float(price_decimal)
        except (OverflowError, ValueError):
            logger.warning(f"Could not convert Decimal {price_decimal} to float for CCXT price formatting. Using Decimal normalize.")
            return str(price_decimal.normalize()) # Fallback to normalized Decimal string

        formatted_price = exchange.price_to_precision(symbol, price_float)

        # --- Post-Formatting Sanity Checks ---
        # Check if formatting resulted in NaN or Zero unexpectedly
        formatted_decimal = safe_decimal_conversion(formatted_price, default=Decimal('NaN'))

        if formatted_decimal is not None and formatted_decimal.is_nan() and not price_decimal.is_nan():
            logger.warning(f"Price formatting resulted in NaN ('{formatted_price}') for valid input {price_decimal}. Using Decimal normalize.")
            return str(price_decimal.normalize())

        if formatted_decimal is not None and formatted_decimal.is_zero() and not price_decimal.is_zero():
             logger.warning(f"Price formatting resulted in zero ('{formatted_price}') for non-zero input {price_decimal}. Using Decimal normalize.")
             return str(price_decimal.normalize())

        # Check if formatted price is drastically different from original (e.g., > 1% deviation) - indicates potential issue
        try:
             if formatted_decimal is not None and not formatted_decimal.is_nan() and not price_decimal.is_zero():
                 deviation = abs(formatted_decimal - price_decimal) / price_decimal
                 if deviation > Decimal("0.01"): # 1% deviation threshold
                      logger.warning(f"Significant deviation ({deviation:.2%}) between original price {price_decimal} and formatted price {formatted_decimal}. Check precision rules.")
        except (DivisionByZero, InvalidOperation):
             pass # Ignore check if original price was zero or calculation fails

        return formatted_price

    except (ccxt.ExchangeError, ccxt.BadSymbol, ValueError, TypeError, KeyError) as e:
        logger.error(f"{Fore.RED}Error shaping price {price_decimal} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}")
        return str(price_decimal.normalize())
    except Exception as e_unexp:
        logger.error(f"{Fore.RED}Unexpected error shaping price {price_decimal} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return str(price_decimal.normalize())


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal, int, str]) -> str:
    """
    Formats an amount (quantity) according to the exchange's market precision rules using CCXT.
    Includes safety checks and fallbacks.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        amount: The amount value (float, Decimal, int, or string representation).

    Returns:
        The amount formatted as a string according to market precision.
        Returns 'NaN' string on critical failure or if input converts to NaN.
        Uses Decimal.normalize() as a fallback if CCXT formatting fails subtly.
    """
    amount_decimal = safe_decimal_conversion(amount, default=Decimal('NaN')) # Ensure default is NaN

    # Handle NaN or invalid inputs early
    if amount_decimal is None or amount_decimal.is_nan():
        logger.error(f"{Fore.RED}Error shaping amount: Input '{amount}' converted to NaN or None. Cannot format.{Style.RESET_ALL}")
        return 'NaN'

    # Handle exact zero case explicitly
    if amount_decimal.is_zero():
        try:
            if symbol in exchange.markets:
                # Use amount_to_precision for zero to get correct decimal places (e.g., "0.000")
                return exchange.amount_to_precision(symbol, 0.0)
        except Exception:
            pass # Fallback to simple "0"
        return "0"

    try:
        # Ensure the market is loaded
        if symbol not in exchange.markets:
            logger.warning(f"Market {symbol} not loaded in format_amount. Attempting to load.")
            exchange.load_markets(True)
        if symbol not in exchange.markets:
             raise ccxt.BadSymbol(f"Market {symbol} could not be loaded for formatting.")

        # CCXT formatting methods typically expect float input
        try:
            amount_float = float(amount_decimal)
        except (OverflowError, ValueError):
            logger.warning(f"Could not convert Decimal {amount_decimal} to float for CCXT amount formatting. Using Decimal normalize.")
            return str(amount_decimal.normalize())

        # Use amount_to_precision for formatting
        formatted_amount = exchange
