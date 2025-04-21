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
Version: 2.3.0 (Unified: Selectable Strategies + Precision + Native SL/TP/TSL + Fortified Config + Pyrmethus Enhancements + Robustness + pandas_ta Fix)

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
- Robust Operation: Features comprehensive error handling for common CCXT exceptions (network issues, authentication failures, rate limits, exchange errors), data validation (NaN handling in OHLCV and indicators), and detailed logging with vibrant console colors via Colorama. Includes robust `pandas_ta` column identification to prevent indicator calculation errors.
- Graceful Shutdown: Designed to handle interruptions (Ctrl+C) or critical errors by attempting to cancel open orders and close any existing positions before exiting.
- Bybit V5 API Focused: Tailored logic for interacting with the Bybit V5 API, particularly regarding position detection (One-Way Mode), order parameters (e.g., 'category'), and native stop placement/confirmation.

Disclaimer:
- **EXTREME RISK**: Trading cryptocurrencies, especially futures contracts with leverage and automated systems, involves substantial risk of financial loss. This script is provided for EDUCATIONAL PURPOSES ONLY. You could lose your entire investment and potentially more. Use this software entirely at your own risk. The authors and contributors assume NO responsibility for any trading losses.
- **NATIVE SL/TP/TSL RELIANCE**: The bot's protective stop mechanisms rely entirely on Bybit's exchange-native order execution. Their performance is subject to exchange conditions, potential slippage during volatile periods, API reliability, order book liquidity, and specific exchange rules. These orders are NOT GUARANTEED to execute at the precise trigger price specified. Slippage can occur, especially with market-type stop orders during high volatility.
- **PARAMETER SENSITIVITY & OPTIMIZATION**: The performance of this bot is HIGHLY dependent on the chosen strategy parameters (indicator settings, risk levels, SL/TP/TSL percentages, filter thresholds). These parameters require extensive backtesting, optimization, and forward testing on a TESTNET environment before considering ANY live deployment. Default parameters are unlikely to be profitable and serve only as examples.
- **API RATE LIMITS & BANS**: Excessive API requests can lead to temporary or permanent bans from the exchange. Monitor API usage and adjust script timing (`SLEEP_SECONDS`) accordingly. CCXT's built-in rate limiter is enabled but may not prevent all issues under heavy load or rapid market conditions.
- **SLIPPAGE**: Market orders, used for entry and potentially for SL/TP/TSL execution by the exchange, are susceptible to slippage. This means the actual execution price may differ significantly from the price observed when the order was placed, especially during high volatility or low liquidity. This impacts Profit & Loss (PnL) and the effective distance of stop-loss orders.
- **TEST THOROUGHLY**: **DO NOT RUN THIS SCRIPT WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTING ON A TESTNET OR DEMO ACCOUNT.** Ensure you fully understand every part of the code, its logic, its dependencies, and its potential risks before any live deployment. Verify indicator calculations and order placement logic carefully.
- **TERMUX DEPENDENCY**: SMS alert functionality requires a Termux environment on an Android device with the Termux:API package installed (`pkg install termux-api`). Ensure it is correctly installed and configured if you enable SMS alerts. The bot will function without it, but alerts will be unavailable.
- **API & LIBRARY UPDATES**: This script targets the Bybit V5 API via the CCXT library. Future updates to the exchange API or the CCXT library may introduce breaking changes that require code modifications. Keep CCXT updated (`pip install -U ccxt`). Verify `pandas_ta` behavior if updated, as column naming conventions can change.
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
from typing import Dict, Optional, Any, Tuple, List, Union, Callable # Enhanced type hinting
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation, DivisionByZero

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
getcontext().prec = 28 # Set Decimal precision for financial exactitude (increased for intermediate calcs, adjust if needed)

# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """
    Loads, validates, and stores configuration parameters from environment variables.
    Provides robust type casting, default value handling, validation, and logging.
    Ensures critical parameters are present and valid before proceeding.
    """
    def __init__(self):
        """Initializes configuration by loading and validating environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED, secret=True)
        self.api_secret: Optional[str] = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED, secret=True)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW) # Target market (CCXT unified format, e.g., 'BTC/USDT:USDT')
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW) # Chart timeframe (e.g., '1m', '5m', '1h')
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW) # Desired leverage multiplier
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW) # Pause between trading cycles (seconds)

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        # NOTE: Renamed EHLERS_MA_CROSS to EMA_CROSS for clarity as it uses standard EMAs.
        self.valid_strategies: List[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EMA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}")
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}")
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.005 = 0.5% of equity per trade
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN) # Multiplier for ATR to set initial fixed SL distance
        self.atr_take_profit_multiplier: Decimal = self._get_env("ATR_TAKE_PROFIT_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN) # Multiplier for ATR to set initial fixed TP distance
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN) # Maximum position value in USDT (overrides risk calc if needed)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN) # e.g., 1.05 = Require 5% more free margin than estimated for order placement

        # --- Native Stop-Loss & Trailing Stop-Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        # Note: Native SL/TP for MARKET entry orders on Bybit V5 are submitted as fixed PRICES.
        # TSL is submitted as a PERCENTAGE * 100 in the parameters (e.g., 0.5 for 0.5%).
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.005 = 0.5% trailing distance from high/low water mark
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.001 = 0.1% price movement in profit before TSL becomes active (using 'activePrice' param)

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN) # Primary Supertrend ATR period
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN) # Primary Supertrend ATR multiplier
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN) # Confirmation Supertrend ATR period
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN) # Confirmation Supertrend ATR multiplier
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN) # StochRSI: RSI period
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN) # StochRSI: Stochastic period
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN) # StochRSI: %K smoothing period
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN) # StochRSI: %D smoothing period (signal line)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN) # StochRSI overbought threshold
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN) # StochRSI oversold threshold
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN) # Momentum indicator period
        # Ehlers Fisher Transform
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN) # Fisher Transform calculation period
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN) # Fisher Transform signal line period (1 usually means no separate signal line smoothing)
        # EMA Cross (Uses standard EMAs, NOT Ehlers Super Smoother)
        self.ema_fast_period: int = self._get_env("EMA_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN) # Fast EMA period
        self.ema_slow_period: int = self._get_env("EMA_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN) # Slow EMA period

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        # Volume Analysis
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW) # Moving average period for volume
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW) # Multiplier over MA to consider a 'spike' (e.g., 1.5 = 150% of MA)
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW) # Require volume spike for entry signal?
        # Order Book Analysis
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW) # Number of bid/ask levels to analyze for ratio
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW) # Min Bid/Ask volume ratio for long confirmation
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW) # Max Bid/Ask volume ratio for short confirmation (ratio = Total Bid Vol / Total Ask Vol within depth)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW) # Fetch OB every cycle (more API calls) or only when needed for entry confirmation?

        # --- ATR Calculation (for Initial SL/TP) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN) # Period for ATR calculation used in SL/TP

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA) # Enable/disable SMS alerts globally
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA, required=False) # Recipient phone number for alerts (optional)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA) # Max time to wait for SMS command execution (seconds)

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = self._get_env("CCXT_RECV_WINDOW", 10000, cast_type=int, color=Fore.WHITE) # Milliseconds for API request validity (Bybit default 5000, increased for potential latency)
        self.order_book_fetch_limit: int = max(25, self.order_book_depth) # How many levels to fetch (ensure >= depth needed, common limits are 25, 50, 100, 200)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW) # Max time to wait for market order fill confirmation (seconds)
        self.stop_attach_confirm_attempts: int = self._get_env("STOP_ATTACH_CONFIRM_ATTEMPTS", 3, cast_type=int, color=Fore.YELLOW) # Attempts to confirm native stops are attached to position after entry
        self.stop_attach_confirm_delay_seconds: int = self._get_env("STOP_ATTACH_CONFIRM_DELAY_SECONDS", 1, cast_type=int, color=Fore.YELLOW) # Delay between attempts to confirm stops

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy: str = "buy"       # CCXT standard side for buying
        self.side_sell: str = "sell"     # CCXT standard side for selling
        self.pos_long: str = "Long"      # Internal representation for a long position
        self.pos_short: str = "Short"    # Internal representation for a short position
        self.pos_none: str = "None"      # Internal representation for no position (flat)
        self.usdt_symbol: str = "USDT"   # The stablecoin quote currency symbol used by Bybit
        self.retry_count: int = 3        # Default attempts for certain retryable API calls (e.g., setting leverage)
        self.retry_delay_seconds: int = 2 # Default pause between retries (seconds)
        self.api_fetch_limit_buffer: int = 50 # Extra candles to fetch beyond strict indicator needs, providing a safety margin (Increased buffer)
        self.position_qty_epsilon: Decimal = Decimal("1e-9") # Small value for float/decimal comparisons involving position size to handle precision issues
        self.post_close_delay_seconds: int = 3 # Brief pause after successfully closing a position (seconds) to allow exchange state to potentially settle
        self.min_order_value_usdt: Decimal = Decimal("1.0") # Minimum order value in USDT (Bybit default is often 1 USDT for perpetuals, adjust if needed based on market)

        # --- Post-Initialization Validation ---
        self._validate_parameters()

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    def _validate_parameters(self):
        """Performs comprehensive validation checks on loaded parameters."""
        logger.debug("Validating configuration parameters...")
        errors = []
        warnings = []

        if self.leverage <= 0:
            errors.append("LEVERAGE must be a positive integer.")
        if self.risk_per_trade_percentage <= 0 or self.risk_per_trade_percentage >= 1:
            errors.append("RISK_PER_TRADE_PERCENTAGE must be between 0 and 1 (exclusive, e.g., 0.005 for 0.5%).")
        if self.atr_stop_loss_multiplier <= 0:
            errors.append("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if self.atr_take_profit_multiplier <= 0:
             errors.append("ATR_TAKE_PROFIT_MULTIPLIER must be positive.")
        if self.trailing_stop_percentage < 0 or self.trailing_stop_percentage >= 1: # TSL can be 0 to disable
            errors.append("TRAILING_STOP_PERCENTAGE must be between 0 and 1 (inclusive of 0 for disabling).")
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

        # Warnings for potentially problematic settings
        if self.enable_sms_alerts and not self.sms_recipient_number:
             warnings.append("SMS alerts enabled (ENABLE_SMS_ALERTS=true) but SMS_RECIPIENT_NUMBER is not set. Alerts will not be sent.")
        if self.sleep_seconds < 5:
             warnings.append(f"SLEEP_SECONDS ({self.sleep_seconds}) is very low (< 5s). This significantly increases the risk of hitting API rate limits. Monitor usage closely.")
        if not self.symbol or "/" not in self.symbol or ":" not in self.symbol:
             errors.append(f"SYMBOL '{self.symbol}' does not appear to be a valid CCXT unified symbol format (e.g., BASE/QUOTE:SETTLE like BTC/USDT:USDT).")
        # Add more specific strategy parameter validations if needed (e.g., fast period < slow period)
        if self.strategy_name == "EMA_CROSS" and self.ema_fast_period >= self.ema_slow_period:
             errors.append(f"EMA_CROSS strategy requires EMA_FAST_PERIOD ({self.ema_fast_period}) to be less than EMA_SLOW_PERIOD ({self.ema_slow_period}).")

        # Log warnings
        for warning in warnings:
            logger.warning(f"{Fore.YELLOW}Config Validation Warning: {warning}{Style.RESET_ALL}")

        # Log errors and raise if any critical issues found
        if errors:
            for error in errors:
                logger.critical(f"{Back.RED}{Fore.WHITE}Config Validation Error: {error}{Style.RESET_ALL}")
            raise ValueError("Configuration validation failed. Please check the .env file and error messages above.")

        logger.debug("Configuration parameter validation complete.")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE, secret: bool = False) -> Any:
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
            elif required and default is not None:
                # Use the default if required and not set
                logger.debug(f"{color}Required rune {key}: Not Set. Using Required Default: '{log_value(default)}'{Style.RESET_ALL}")
                value_to_cast = default
                source = "Required Default"
            elif not required:
                 # Use default if not required and not set
                 logger.debug(f"{color}Summoning {key}: Not Set. Using Default: '{log_value(default)}'{Style.RESET_ALL}")
                 value_to_cast = default
                 source = "Default"
            # If not required and no default, value_to_cast remains None
            else:
                 value_to_cast = None
                 source = "None (Not Required, No Default)"

        else:
            # Env var was found
            logger.debug(f"{color}Summoning {key}: Found Env Value: '{log_value(value_str)}'{Style.RESET_ALL}")
            value_to_cast = value_str

        # --- Attempt Casting (applies to both env var value and default value) ---
        if value_to_cast is None:
            # This handles cases where:
            # 1. Not required, env var not set, no default -> returns None
            if not required:
                 logger.debug(f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}")
                 return None
            else:
                 # This state should not be reached due to the check above, but handle defensively.
                 logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' resolved to None unexpectedly after initial checks.{Style.RESET_ALL}")
                 raise ValueError(f"Required environment variable '{key}' resolved to None unexpectedly.")


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
                    raise ValueError(f"Invalid boolean value '{raw_value_str}'")
            elif cast_type == Decimal:
                # Handle potential empty string after strip
                if raw_value_str == "": raise InvalidOperation("Empty string cannot be converted to Decimal.")
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                # Handle potential empty string after strip
                if raw_value_str == "": raise ValueError("Empty string cannot be converted to int.")
                # Cast via Decimal first to handle potential float strings like "10.0" -> 10 gracefully
                # Also handles scientific notation like "1e-5" which int() would fail on directly
                try:
                    dec_val = Decimal(raw_value_str)
                    # Check if it represents a whole number
                    if dec_val == dec_val.to_integral_value(rounding=ROUND_HALF_UP):
                        final_value = int(dec_val)
                    else:
                        raise ValueError(f"Value '{raw_value_str}' is not a whole number for int conversion.")
                except InvalidOperation:
                    raise ValueError(f"Invalid numeric value '{raw_value_str}' for int conversion.")
            elif cast_type == float:
                 # Handle potential empty string after strip
                 if raw_value_str == "": raise ValueError("Empty string cannot be converted to float.")
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
                     logger.warning(f"{Fore.YELLOW}Casting failed for {key}, default is None. Final value: None{Style.RESET_ALL}")
                     return None # Return None if casting fails and no default
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

# Configure basic logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)] # Output to console
)
logger: logging.Logger = logging.getLogger(__name__) # Get the root logger

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

# Apply colors if outputting to a TTY (like Termux console or standard terminal)
if sys.stdout.isatty():
    # Define color mappings for levels
    level_colors = {
        logging.DEBUG: f"{Fore.CYAN}{Style.DIM}",
        logging.INFO: f"{Fore.BLUE}",
        SUCCESS_LEVEL: f"{Fore.MAGENTA}{Style.BRIGHT}", # Use Bright Magenta for Success
        logging.WARNING: f"{Fore.YELLOW}{Style.BRIGHT}",
        logging.ERROR: f"{Fore.RED}{Style.BRIGHT}",
        logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}"
    }
    # Apply colors to level names by replacing the formatter's levelname part
    # This requires a custom Formatter or careful manipulation of existing handlers/formatters.
    # A simpler approach is to just color the level name when adding it.
    for level, color_style in level_colors.items():
        level_name = logging.getLevelName(level)
        # Check if it's the custom level or standard level not already colored
        if level == SUCCESS_LEVEL or (level in logging._levelToName and not logging.getLevelName(level).startswith('\033')):
             logging.addLevelName(level, f"{color_style}{level_name}{Style.RESET_ALL}")
else:
    # If not a TTY, ensure SUCCESS level name is still registered without color codes
    # This might already be handled above, but double-check
    if logging.getLevelName(SUCCESS_LEVEL) == f"Level {SUCCESS_LEVEL}":
        logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


# --- Global Objects - Instantiated Arcana ---
# Define CONFIG as Optional initially
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
            enable_sms = getattr(CONFIG, 'enable_sms_alerts', False)
            recipient = getattr(CONFIG, 'sms_recipient_number', None)
            timeout = getattr(CONFIG, 'sms_timeout_seconds', 30)
            symbol_for_msg = getattr(CONFIG, 'symbol', "UNKNOWN_SYMBOL")
            strategy_for_msg = getattr(CONFIG, 'strategy_name', "UNKNOWN_STRATEGY")
        else:
            # Fallback to direct environment variable access if CONFIG failed early
            enable_sms_str = os.getenv("ENABLE_SMS_ALERTS", "false").lower()
            enable_sms = enable_sms_str in ['true', '1', 'yes', 'y', 'on']
            recipient = os.getenv("SMS_RECIPIENT_NUMBER")
            timeout_str = os.getenv("SMS_TIMEOUT_SECONDS", "30")
            try:
                timeout = int(timeout_str)
            except ValueError:
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
            command: List[str] = ['termux-sms-send', '-n', recipient, message]
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
    send_sms_alert(f"[{os.getenv('STRATEGY_NAME', 'CONFIG_ERR')}] CRITICAL CONFIG ERROR: {config_error}. Bot failed to start.")
    sys.exit(1)

# Check if CONFIG is None after try-except (shouldn't happen if Config() succeeded)
if CONFIG is None:
     logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Configuration object is None after initialization block. Exiting.{Style.RESET_ALL}")
     sys.exit(1)


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("NaN")) -> Decimal:
    """
    Safely converts a value to a Decimal, handling None, pandas NA, empty strings,
    and potential errors. Returns the specified default value on failure.

    Args:
        value: The value to convert (can be string, float, int, Decimal, None, pandas NA, etc.).
        default: The Decimal value to return if conversion fails or input is None/NA/empty.
                 Defaults to Decimal('NaN'). Use Decimal('0.0') if zero fallback is desired.

    Returns:
        The converted Decimal value or the default.
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
    if price_decimal.is_nan():
        logger.error(f"{Fore.RED}Error shaping price: Input '{price}' converted to NaN. Cannot format.{Style.RESET_ALL}")
        return 'NaN'

    # Handle exact zero case explicitly
    if price_decimal.is_zero():
         # Check market precision for zero formatting if available (though "0" is usually safe)
         try:
            if symbol in exchange.markets:
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

        if formatted_decimal.is_nan() and not price_decimal.is_nan():
            logger.warning(f"Price formatting resulted in NaN ('{formatted_price}') for valid input {price_decimal}. Using Decimal normalize.")
            return str(price_decimal.normalize())

        if formatted_decimal.is_zero() and not price_decimal.is_zero():
             logger.warning(f"Price formatting resulted in zero ('{formatted_price}') for non-zero input {price_decimal}. Using Decimal normalize.")
             return str(price_decimal.normalize())

        # Check if formatted price is drastically different from original (e.g., > 1% deviation) - indicates potential issue
        try:
             if not formatted_decimal.is_nan() and not price_decimal.is_zero():
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
    if amount_decimal.is_nan():
        logger.error(f"{Fore.RED}Error shaping amount: Input '{amount}' converted to NaN. Cannot format.{Style.RESET_ALL}")
        return 'NaN'

    # Handle exact zero case explicitly
    if amount_decimal.is_zero():
        try:
            if symbol in exchange.markets:
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

        formatted_amount = exchange.amount_to_precision(symbol, amount_float)

        # --- Post-Formatting Sanity Checks ---
        formatted_decimal = safe_decimal_conversion(formatted_amount, default=Decimal('NaN'))

        if formatted_decimal.is_nan() and not amount_decimal.is_nan():
            logger.warning(f"Amount formatting resulted in NaN ('{formatted_amount}') for valid input {amount_decimal}. Using Decimal normalize.")
            return str(amount_decimal.normalize())

        if formatted_decimal.is_zero() and not amount_decimal.is_zero():
             logger.warning(f"Amount formatting resulted in zero ('{formatted_amount}') for non-zero input {amount_decimal}. Using Decimal normalize.")
             return str(amount_decimal.normalize())

        # Check for significant deviation (e.g., > 0.1% for amount)
        try:
             if not formatted_decimal.is_nan() and not amount_decimal.is_zero():
                 deviation = abs(formatted_decimal - amount_decimal) / amount_decimal
                 if deviation > Decimal("0.001"): # 0.1% deviation threshold
                      logger.warning(f"Significant deviation ({deviation:.2%}) between original amount {amount_decimal} and formatted amount {formatted_decimal}. Check precision rules.")
        except (DivisionByZero, InvalidOperation):
             pass # Ignore check if original amount was zero

        return formatted_amount

    except (ccxt.ExchangeError, ccxt.BadSymbol, ValueError, TypeError, KeyError) as e:
        logger.error(f"{Fore.RED}Error shaping amount {amount_decimal} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}")
        return str(amount_decimal.normalize())
    except Exception as e_unexp:
        logger.error(f"{Fore.RED}Unexpected error shaping amount {amount_decimal} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return str(amount_decimal.normalize())


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """
    Initializes and returns the CCXT Bybit exchange instance.

    Handles authentication, loads markets, performs basic connectivity checks,
    and configures necessary options for Bybit V5 API interaction.

    Returns:
        A configured CCXT Bybit exchange instance, or None if initialization fails.
    """
    global CONFIG # Ensure access to the global CONFIG object
    if CONFIG is None: # Should not happen if called after global init, but check
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: CONFIG object not initialized before initialize_exchange call.{Style.RESET_ALL}")
        return None

    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        # This should technically be caught by Config validation, but double-check
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret runes missing. Cannot open portal.{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: API keys missing. Spell failed on {CONFIG.symbol}.")
        return None
    try:
        # Forging the connection with Bybit V5 defaults
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True, # Respect the exchange spirits' limits
            "options": {
                "defaultType": "linear", # Assume USDT perpetuals unless symbol specifies otherwise
                "adjustForTimeDifference": True, # Sync client time with server time for request validity
                # Explicitly request V5 API and set default category for V5 operations
                'api-version': 'v5',
                'defaultCategory': 'linear', # Set default category for V5 operations (linear for USDT perps)
            },
            'recvWindow': CONFIG.default_recv_window, # Set custom receive window
        })

        # --- Testnet Configuration ---
        # Uncomment the following line to use Bybit's testnet environment
        # exchange.set_sandbox_mode(True)

        # Use getattr defensively as 'sandbox' might not be present on all exchange objects
        is_sandbox = getattr(exchange, 'sandbox', False)
        # Also check URLs as a backup detection method
        if not is_sandbox and hasattr(exchange, 'urls') and 'test' in exchange.urls.get('api', ''):
             is_sandbox = True # Infer from URL if sandbox attribute is False/missing

        if is_sandbox:
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}!!! TESTNET MODE ACTIVE !!!{Style.RESET_ALL}")
        else:
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! LIVE TRADING MODE ACTIVE - EXTREME CAUTION ADVISED !!!{Style.RESET_ALL}")


        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True) # Force reload for fresh market data and limits
        logger.debug(f"Loaded {len(exchange.markets)} market structures.")

        # --- Initial Authentication & Connectivity Check ---
        logger.debug("Performing initial balance check for authentication and V5 connectivity...")
        try:
            # Fetch balance using V5 specific parameters to confirm keys and API version access
            # Specify account type if needed (e.g., UNIFIED, CONTRACT) - 'CONTRACT' is typical for perpetuals
            # Note: 'defaultCategory' in options should handle the category requirement for fetch_balance
            params = {'accountType': 'CONTRACT'} # Or 'UNIFIED' depending on user's account setup
            balance = exchange.fetch_balance(params=params)
            logger.success(f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (Targeting V5 API, {'Testnet' if is_sandbox else 'Live'}).{Style.RESET_ALL}")

            # Basic check for sufficient funds (optional, but good practice)
            # Check total equity or available balance depending on risk model
            # Use safe_decimal_conversion for robustness
            # Bybit V5 balance structure: balance['USDT']['total'] or ['free'] or ['used'] or ['equity']
            total_equity = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol, {}).get('equity'), default=Decimal('NaN'))
            free_balance = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol, {}).get('free'), default=Decimal('NaN'))

            if total_equity.is_nan() or free_balance.is_nan():
                 logger.warning(f"{Fore.YELLOW}Initial Balance Check: Could not parse balance details (Equity/Free) from response. Check API response structure or permissions.{Style.RESET_ALL}")
                 logger.debug(f"Raw Balance Response Snippet: {balance.get(CONFIG.usdt_symbol, {})}")
            else:
                 logger.info(f"Initial Balance Check: Equity: {total_equity:.4f} {CONFIG.usdt_symbol}, Free Margin: {free_balance:.4f} {CONFIG.usdt_symbol}")
                 if total_equity <= CONFIG.min_order_value_usdt: # Use min order value as a low threshold
                     logger.warning(f"{Fore.YELLOW}Initial Balance Check: Total equity ({total_equity:.2f} {CONFIG.usdt_symbol}) appears very low. Ensure sufficient funds for trading.{Style.RESET_ALL}")

            send_sms_alert(f"[{CONFIG.strategy_name}] Portal opened & authenticated on {CONFIG.symbol} ({'Testnet' if is_sandbox else 'Live'}). Equity: {total_equity:.2f}")
            return exchange

        except ccxt.AuthenticationError as auth_err:
            # Specific handling for auth errors after initial connection
            logger.critical(f"{Back.RED}{Fore.WHITE}Authentication failed during balance check: {auth_err}. Check API keys, permissions, IP whitelist, and account status on Bybit.{Style.RESET_ALL}")
            send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Authentication FAILED ({auth_err}). Spell failed on {CONFIG.symbol}.")
            return None
        except ccxt.ExchangeError as ex_err:
            # Catch V5 specific errors like invalid category/accountType if API setup is wrong
            logger.critical(f"{Back.RED}{Fore.WHITE}Exchange error during initial balance check (V5 connectivity issue? Invalid params? Account type?): {ex_err}.{Style.RESET_ALL}")
            logger.debug(f"Params used for balance check: {params}")
            send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Exchange Error on Init ({ex_err}). Spell failed on {CONFIG.symbol}.")
            return None
        except Exception as balance_err:
             logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error during initial balance check: {balance_err}.{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Unexpected Error on Init ({type(balance_err).__name__}). Spell failed on {CONFIG.symbol}.")
             return None


    # --- Broader Error Handling for Initialization ---
    except ccxt.AuthenticationError as e: # Catch auth error during initial setup
        logger.critical(f"{Back.RED}{Fore.WHITE}Authentication failed during initial connection setup: {e}.{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Authentication FAILED on setup ({e}). Spell failed on {CONFIG.symbol}.")
    except ccxt.NetworkError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Network disturbance during portal opening: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Network Error on Init ({e}). Spell failed on {CONFIG.symbol}.")
    except ccxt.ExchangeError as e: # Catch other exchange errors during setup
        logger.critical(f"{Back.RED}{Fore.WHITE}Exchange spirit rejected portal opening: {e}. Check Bybit status, API documentation, or account status.{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Exchange Error on Init ({e}). Spell failed on {CONFIG.symbol}.")
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected chaos during portal opening: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Unexpected Init Error: {type(e).__name__}. Spell failed on {CONFIG.symbol}.")

    return None # Return None if any initialization step failed

# --- Robust pandas_ta Column Identification Helper ---
def find_pandas_ta_column(df: pd.DataFrame, prefix_hint: str, suffix_hint: str = "", expected_count: int = 1) -> Optional[str]:
    """
    Finds a column in a DataFrame added by pandas_ta, based on prefix/suffix hints.
    Designed to be more robust than guessing the exact name by checking for containment and ending.

    Args:
        df: The DataFrame *after* running df.ta.indicator(append=True).
        prefix_hint: The expected start or contained part of the column name (case-insensitive check).
        suffix_hint: The expected end of the column name (case-insensitive check). Can be empty.
        expected_count: The expected number of columns matching the pattern (usually 1).

    Returns:
        The name of the found column, or None if not found or ambiguous (unless exactly expected_count match).
    """
    prefix_hint_lower = prefix_hint.lower()
    suffix_hint_lower = suffix_hint.lower()

    # Look for columns that contain the prefix hint and end with the suffix hint (case-insensitive)
    matching_cols = [
        col for col in df.columns
        if prefix_hint_lower in col.lower() and col.lower().endswith(suffix_hint_lower)
    ]

    if len(matching_cols) == expected_count:
        # logger.debug(f"pandas_ta Finder: Found unique column matching prefix '{prefix_hint}' and suffix '{suffix_hint}': {matching_cols[0]}.")
        return matching_cols[0] # Return the single unique match

    elif not matching_cols:
        # logger.debug(f"pandas_ta Finder: No columns found strictly matching prefix '{prefix_hint}' and suffix '{suffix_hint}'.")
        # Try slightly looser match: contains prefix OR ends with suffix (if suffix provided)
        # Only use looser match if it finds exactly the expected count
        if suffix_hint:
             looser_match = [col for col in df.columns if prefix_hint_lower in col.lower() or col.lower().endswith(suffix_hint_lower)]
             if len(looser_match) == expected_count:
                 logger.debug(f"pandas_ta Finder: Found unique column via looser match (contains prefix OR ends suffix): {looser_match[0]}")
                 return looser_match[0]
             # else: logger.debug(f"Looser match found {len(looser_match)} columns, expected {expected_count}. Not using.")
        # Try matching only prefix if no suffix was provided
        elif not suffix_hint:
             prefix_only_match = [col for col in df.columns if prefix_hint_lower in col.lower()]
             if len(prefix_only_match) == expected_count:
                  logger.debug(f"pandas_ta Finder: Found unique column via prefix-only match: {prefix_only_match[0]}")
                  return prefix_only_match[0]
             # else: logger.debug(f"Prefix-only match found {len(prefix_only_match)} columns, expected {expected_count}. Not using.")

        logger.warning(f"pandas_ta Finder: No reliable column found for prefix '{prefix_hint}', suffix '{suffix_hint}'. Available columns: {list(df.columns)}")
        return None # Still not found or ambiguous

    elif len(matching_cols) > expected_count:
        # Ambiguous: multiple columns match strictly. Log warning and return None.
        logger.warning(f"pandas_ta Finder: Found MULTIPLE columns matching prefix '{prefix_hint}' and suffix '{suffix_hint}': {matching_cols}. Expected {expected_count}. Cannot reliably identify.")
        # Could try returning the *last* one as a heuristic, but safer to return None.
        # logger.debug(f"Heuristically choosing the last matching column: {matching_cols[-1]}")
        # return matching_cols[-1] # Heuristic: assume the most recent is the intended one
        return None
    else: # len(matching_cols) < expected_count (but > 0)
         logger.warning(f"pandas_ta Finder: Found {len(matching_cols)} columns matching prefix '{prefix_hint}' and suffix '{suffix_hint}': {matching_cols}. Expected {expected_count}. Cannot reliably identify.")
         return None # Not enough columns found


# --- Indicator Calculation Functions - Scrying the Market ---

def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """
    Calculates the Supertrend indicator using pandas_ta, using robust column identification.

    Args:
        df: Pandas DataFrame with 'high', 'low', 'close' columns.
        length: The ATR lookback period for the Supertrend calculation.
        multiplier: The ATR multiplier for the Supertrend calculation (Decimal).
        prefix: A string prefix to add to the resulting column names (e.g., "confirm_").

    Returns:
        The DataFrame with added Supertrend columns:
        - f'{prefix}supertrend': The Supertrend line value (Decimal).
        - f'{prefix}trend': Boolean indicating uptrend (True) or downtrend (False) (Nullable).
        - f'{prefix}st_long': Boolean, True if trend flipped to Long on this candle.
        - f'{prefix}st_short': Boolean, True if trend flipped to Short on this candle.
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]

    required_input_cols = ["high", "low", "close"]
    # Estimate minimum data length needed for pandas_ta Supertrend
    min_len = length + 50 # Increased buffer for stability

    # Input validation
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): DataFrame is None or empty. Cannot calculate.{Style.RESET_ALL}")
        return df # Return original (potentially None)
    if not all(c in df.columns for c in required_input_cols):
        logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Missing required columns {required_input_cols}. Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA # Assign NA to expected output columns if df exists
        return df
    if len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df)}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA # Assign NA to expected output columns
        return df

    try:
        # Store columns before calculation to identify new ones added by pandas_ta
        initial_columns = set(df.columns)

        # pandas_ta expects float multiplier for calculation
        multiplier_float = float(multiplier)
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={multiplier_float}")

        # Calculate using pandas_ta, appending results to the DataFrame
        try:
            df.ta.supertrend(length=length, multiplier=multiplier_float, append=True)
        except Exception as ta_calc_error:
             logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): pandas_ta calculation failed: {ta_calc_error}{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             for col in target_cols: df[col] = pd.NA # Nullify results
             return df # Return early on calculation failure


        # --- Robust Verification and Renaming/Conversion ---
        # Identify new columns added by pandas_ta
        new_columns = list(set(df.columns) - initial_columns)
        # logger.debug(f"pandas_ta added columns for {col_prefix}ST: {new_columns}")

        # Programmatically find the correct columns based on typical pandas_ta names
        # Multiplier formatting in pandas_ta column names can vary (e.g., 2.5 -> _2.5 or _2_5)
        # Construct hint carefully, trying common formats
        mult_str_simple = str(multiplier_float)
        mult_str_underscore = mult_str_simple.replace('.', '_')
        # Try both suffix formats
        st_col_hint_suffix1 = f"_{length}_{mult_str_simple}"
        st_col_hint_suffix2 = f"_{length}_{mult_str_underscore}"

        st_col = find_pandas_ta_column(df, "SUPERT", suffix_hint=st_col_hint_suffix1) or \
                 find_pandas_ta_column(df, "SUPERT", suffix_hint=st_col_hint_suffix2)
        st_trend_col = find_pandas_ta_column(df, "SUPERTd", suffix_hint=st_col_hint_suffix1) or \
                       find_pandas_ta_column(df, "SUPERTd", suffix_hint=st_col_hint_suffix2)
        st_long_col = find_pandas_ta_column(df, "SUPERTl", suffix_hint=st_col_hint_suffix1) or \
                      find_pandas_ta_column(df, "SUPERTl", suffix_hint=st_col_hint_suffix2)
        st_short_col = find_pandas_ta_column(df, "SUPERTs", suffix_hint=st_col_hint_suffix1) or \
                       find_pandas_ta_column(df, "SUPERTs", suffix_hint=st_col_hint_suffix2)


        # Check if all essential columns were found
        essential_cols_found = st_col and st_trend_col
        signal_cols_found = st_long_col and st_short_col

        if not essential_cols_found:
            missing_details = f"ST Value Col: {st_col is None}, Trend Col: {st_trend_col is None}"
            logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Failed to find essential Supertrend columns (Value/Trend) from pandas_ta. Missing: {missing_details}. Check pandas_ta version or data.{Style.RESET_ALL}")
            # Attempt to clean up any partial columns found among new columns
            partial_cols_found = [c for c in [st_col, st_trend_col, st_long_col, st_short_col] if c]
            if partial_cols_found: df.drop(columns=partial_cols_found, errors='ignore', inplace=True)
            for col in target_cols: df[col] = pd.NA # Nullify results
            return df

        # Convert Supertrend value to Decimal
        df[f"{col_prefix}supertrend"] = df[st_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))

        # Interpret trend: 1 = Uptrend, -1 = Downtrend. Convert to boolean: True for Up, False for Down.
        # Handle potential NaN in trend column safely -> Use nullable Boolean type
        df[f"{col_prefix}trend"] = df[st_trend_col].apply(lambda x: bool(x == 1) if pd.notna(x) and x == 1 else (False if pd.notna(x) and x == -1 else pd.NA)).astype('boolean')

        # Interpret Flip Signals:
        # SUPERTl: Non-NaN (often 1.0 or entry price) when trend flips Long.
        # SUPERTs: Non-NaN (often -1.0 or entry price) when trend flips Short.
        if signal_cols_found:
            # Check for non-NaN signals
            df[f"{col_prefix}st_long"] = df[st_long_col].notna()
            df[f"{col_prefix}st_short"] = df[st_short_col].notna()
        else:
            logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Long/Short signal columns (SUPERTl/s) not found. Deriving flips from trend changes.{Style.RESET_ALL}")
            trend_series = df[f"{col_prefix}trend"]
            # Shifted trend: Compare current trend with previous trend
            prev_trend = trend_series.shift(1)
            # Long flip: Trend is True (Up) now, and was False (Down) previously (ignore NA transitions initially for safety)
            df[f"{col_prefix}st_long"] = (trend_series == True) & (prev_trend == False)
            # Short flip: Trend is False (Down) now, and was True (Up) previously
            df[f"{col_prefix}st_short"] = (trend_series == False) & (prev_trend == True)
            # Ensure boolean type (non-nullable boolean is okay here, as comparison results in True/False)
            df[f"{col_prefix}st_long"] = df[f"{col_prefix}st_long"].astype(bool)
            df[f"{col_prefix}st_short"] = df[f"{col_prefix}st_short"].astype(bool)


        # Check for NaNs in critical output columns (last row)
        last_row = df.iloc[-1]
        nan_check_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend"] # Check only essential outputs
        if last_row[nan_check_cols].isnull().any():
            nan_cols = last_row[nan_check_cols].isnull()
            nan_details = ', '.join([col for col in nan_check_cols if nan_cols[col]])
            logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Calculation resulted in NaN/NA(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}")

        # Clean up raw columns created by pandas_ta (if they exist and aren't target columns)
        raw_cols_to_drop = [c for c in [st_col, st_trend_col, st_long_col, st_short_col] if c is not None and c not in target_cols]
        if raw_cols_to_drop:
            # logger.debug(f"Dropping raw pandas_ta columns: {raw_cols_to_drop}")
            df.drop(columns=raw_cols_to_drop, errors='ignore', inplace=True)

        # Log the latest reading for debugging
        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        last_trend_bool = df[f'{col_prefix}trend'].iloc[-1] # Nullable boolean

        if pd.notna(last_st_val) and pd.notna(last_trend_bool):
            last_trend = 'Up' if last_trend_bool else 'Down'
            signal = 'LONG FLIP' if df[f'{col_prefix}st_long'].iloc[-1] else ('SHORT FLIP' if df[f'{col_prefix}st_short'].iloc[-1] else 'Hold')
            trend_color = Fore.GREEN if last_trend == 'Up' else Fore.RED
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier_float})): Trend={trend_color}{last_trend}{Style.RESET_ALL}, Val={last_st_val:.4f}, Signal={signal}")
        else:
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier_float})): Resulted in NA for Trend or Value on last candle.")

    except KeyError as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Error accessing column - likely pandas_ta issue, data problem, or naming mismatch: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df

def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> Dict[str, Optional[Decimal]]:
    """
    Calculates ATR, Volume Simple Moving Average (SMA), and Volume Ratio.

    Args:
        df: Pandas DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len: The lookback period for ATR calculation.
        vol_ma_len: The lookback period for Volume SMA calculation.

    Returns:
        A dictionary containing:
        - 'atr': The latest ATR value (Decimal) or Decimal('NaN').
        - 'volume_ma': The latest Volume SMA value (Decimal) or Decimal('NaN').
        - 'last_volume': The latest volume value (Decimal) or Decimal('NaN').
        - 'volume_ratio': The ratio of last volume to volume SMA (Decimal) or Decimal('NaN').
        Returns NaN values if calculation fails or data is insufficient/invalid.
    """
    results: Dict[str, Optional[Decimal]] = {"atr": Decimal('NaN'), "volume_ma": Decimal('NaN'), "last_volume": Decimal('NaN'), "volume_ratio": Decimal('NaN')}
    required_cols = ["high", "low", "close", "volume"]
    # Need sufficient data for both ATR and Volume MA calculations
    min_len = max(atr_len, vol_ma_len) + 50 # Increased buffer

    # Input validation
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): DataFrame is None or empty. Cannot calculate.{Style.RESET_ALL}")
        return results
    if not all(c in df.columns for c in required_cols):
        logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Missing required columns {required_cols}. Cannot calculate.{Style.RESET_ALL}")
        return results
    if len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Len: {len(df)}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        return results

    # Create a copy to avoid modifying the original DataFrame directly within this function
    df_copy = df.copy()

    try:
        # --- Calculate ATR ---
        logger.debug(f"Scrying (ATR): Calculating with length={atr_len}")
        try:
            df_copy.ta.atr(length=atr_len, append=True)
        except Exception as ta_atr_error:
             logger.error(f"{Fore.RED}Scrying (ATR): pandas_ta calculation failed: {ta_atr_error}{Style.RESET_ALL}")
             # Don't return yet, try volume calculation
             results["atr"] = Decimal('NaN')
        else:
            atr_col_name_hint = f"ATRr_{atr_len}" # Typical pandas_ta name for ATR
            atr_col = find_pandas_ta_column(df_copy, atr_col_name_hint)

            if atr_col and atr_col in df_copy.columns:
                last_atr = df_copy[atr_col].iloc[-1]
                results["atr"] = safe_decimal_conversion(last_atr, default=Decimal('NaN'))
                # Clean up the raw ATR column added by pandas_ta
                df_copy.drop(columns=[atr_col], errors='ignore', inplace=True)
            else:
                logger.warning(f"ATR column matching hint '{atr_col_name_hint}' not found or resulted in NA after calculation. Check pandas_ta behavior.")
                results["atr"] = Decimal('NaN')

        # --- Calculate Volume SMA and Ratio ---
        logger.debug(f"Scrying (Volume): Calculating SMA with length={vol_ma_len}")
        # Ensure volume is numeric, coercing errors and handling NaNs
        df_copy['volume_numeric'] = pd.to_numeric(df_copy['volume'], errors='coerce')
        initial_nan_count = df_copy['volume_numeric'].isnull().sum()
        if initial_nan_count > 0:
            logger.warning(f"{Fore.YELLOW}Scrying (Volume): Found {initial_nan_count} NaNs in volume data before SMA calculation. Using ffill/bfill.{Style.RESET_ALL}")
            df_copy['volume_numeric'].ffill(inplace=True)
            df_copy['volume_numeric'].bfill(inplace=True)
            final_nan_count = df_copy['volume_numeric'].isnull().sum()
            if final_nan_count > 0:
                 logger.error(f"{Fore.RED}Scrying (Volume): Cannot fill all NaNs ({final_nan_count}) in volume. Volume calculations may be unreliable.{Style.RESET_ALL}")
                 # Proceed, but MA/Ratio will likely be NaN

        volume_ma_col_name = f'volume_sma_{vol_ma_len}'
        # min_periods ensures we get a value even if window isn't full at the start
        df_copy[volume_ma_col_name] = df_copy['volume_numeric'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()

        last_vol_ma = df_copy[volume_ma_col_name].iloc[-1]
        last_vol = df_copy['volume_numeric'].iloc[-1] # Get the most recent numeric volume bar

        # Convert results to Decimal
        results["volume_ma"] = safe_decimal_conversion(last_vol_ma, default=Decimal('NaN'))
        results["last_volume"] = safe_decimal_conversion(last_vol, default=Decimal('NaN'))

        # Calculate Volume Ratio (Last Volume / Volume MA) safely
        vol_ma_val = results["volume_ma"]
        last_vol_val = results["last_volume"]

        if not vol_ma_val.is_nan() and vol_ma_val > CONFIG.position_qty_epsilon and not last_vol_val.is_nan():
            try:
                ratio = last_vol_val / vol_ma_val
                if ratio.is_infinite() or ratio.is_nan():
                     logger.warning(f"Volume ratio calculation resulted in {ratio}. Setting to NaN.")
                     results["volume_ratio"] = Decimal('NaN')
                else:
                    results["volume_ratio"] = ratio
            except (DivisionByZero, InvalidOperation) as ratio_err:
                 logger.warning(f"Division by zero or invalid op encountered calculating volume ratio (Volume MA likely zero/negligible). Error: {ratio_err}")
                 results["volume_ratio"] = Decimal('NaN')
            except Exception as ratio_err:
                 logger.warning(f"Unexpected error calculating volume ratio: {ratio_err}")
                 results["volume_ratio"] = Decimal('NaN')
        else:
            # logger.debug(f"Scrying (Volume): Cannot calculate ratio (VolMA={vol_ma_val}, LastVol={last_vol_val})")
            results["volume_ratio"] = Decimal('NaN') # Set explicitly to NaN

        # Clean up temporary columns from the copy
        if volume_ma_col_name in df_copy.columns: df_copy.drop(columns=[volume_ma_col_name], errors='ignore', inplace=True)
        if 'volume_numeric' in df_copy.columns: df_copy.drop(columns=['volume_numeric'], errors='ignore', inplace=True)
        del df_copy # Explicitly delete copy to potentially free memory sooner

        # Check for NaNs in critical output values
        if results["atr"].is_nan(): logger.warning(f"{Fore.YELLOW}Scrying (ATR): Final ATR result is NaN.{Style.RESET_ALL}")
        if results["volume_ma"].is_nan(): logger.warning(f"{Fore.YELLOW}Scrying (Volume MA): Final Volume MA result is NaN.{Style.RESET_ALL}")
        # No warning for last_volume NaN as it's just reporting raw data
        # No warning for ratio NaN as it depends on the others

        # Log calculated results
        atr_str = f"{results['atr']:.5f}" if not results['atr'].is_nan() else 'NaN'
        vol_ma_str = f"{results['volume_ma']:.2f}" if not results['volume_ma'].is_nan() else 'NaN'
        last_vol_str = f"{results['last_volume']:.2f}" if not results['last_volume'].is_nan() else 'NaN'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if not results['volume_ratio'].is_nan() else 'NaN'
        logger.debug(f"Scrying Results: ATR({atr_len})={Fore.CYAN}{atr_str}{Style.RESET_ALL}, "
                     f"LastVol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: Decimal('NaN') for key in results} # Nullify all results on error
    return results

def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """
    Calculates StochRSI (%K and %D) and Momentum indicator using pandas_ta, robustly.

    Args:
        df: Pandas DataFrame with 'close' column.
        rsi_len: The lookback period for the RSI component of StochRSI.
        stoch_len: The lookback period for the Stochastic component of StochRSI.
        k: The smoothing period for the %K line of StochRSI.
        d: The smoothing period for the %D (signal) line of StochRSI.
        mom_len: The lookback period for the Momentum indicator.

    Returns:
        The DataFrame with added columns:
        - 'stochrsi_k': The StochRSI %K value (Decimal or NaN).
        - 'stochrsi_d': The StochRSI %D value (Decimal or NaN).
        - 'momentum': The Momentum value (Decimal or NaN).
        Returns original DataFrame with NA/NaN columns if calculation fails or data is insufficient.
    """
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    # Estimate minimum length: StochRSI needs roughly RSI + Stoch + D periods. Momentum needs its own period.
    min_len_stochrsi = rsi_len + stoch_len + d + 50 # Increased buffer
    min_len_mom = mom_len + 1
    min_len = max(min_len_stochrsi, min_len_mom) + 5 # Extra safety buffer

    # Input validation
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): DataFrame is None or empty. Cannot calculate.{Style.RESET_ALL}")
        return df
    if not all(c in df.columns for c in ["close"]):
        logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): Missing required column 'close'. Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    if len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Len: {len(df)}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        initial_columns = set(df.columns)

        # --- Calculate StochRSI ---
        logger.debug(f"Scrying (StochRSI): Calculating with RSI={rsi_len}, Stoch={stoch_len}, K={k}, D={d}")
        try:
            df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=True)
        except Exception as ta_stoch_err:
             logger.error(f"{Fore.RED}Scrying (StochRSI): pandas_ta calculation failed: {ta_stoch_err}{Style.RESET_ALL}")
             # Nullify relevant columns if calculation fails
             df['stochrsi_k'] = pd.NA
             df['stochrsi_d'] = pd.NA
             # Continue to momentum calculation

        # Find the actual StochRSI columns using hints
        # Typical pandas_ta names: STOCHRSIk_stoch_rsi_k_d, STOCHRSId_stoch_rsi_k_d
        stoch_suffix_hint = f"_{stoch_len}_{rsi_len}_{k}_{d}"
        stochrsi_k_col = find_pandas_ta_column(df, "STOCHRSIk", suffix_hint=stoch_suffix_hint)
        stochrsi_d_col = find_pandas_ta_column(df, "STOCHRSId", suffix_hint=stoch_suffix_hint)

        # Assign results to main DataFrame and convert to Decimal
        if stochrsi_k_col and stochrsi_k_col in df.columns:
            df['stochrsi_k'] = df[stochrsi_k_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            # Drop raw column if it's different from target and not already dropped
            if stochrsi_k_col != 'stochrsi_k' and stochrsi_k_col in df.columns:
                df.drop(columns=[stochrsi_k_col], errors='ignore', inplace=True)
        else:
            if 'stochrsi_k' not in df.columns: # Only log/set NA if not already set by error handling
                logger.warning(f"StochRSI K column matching hint '{stoch_suffix_hint}' not found after calculation. Check pandas_ta naming/behavior.")
                df['stochrsi_k'] = pd.NA

        if stochrsi_d_col and stochrsi_d_col in df.columns:
             df['stochrsi_d'] = df[stochrsi_d_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
             if stochrsi_d_col != 'stochrsi_d' and stochrsi_d_col in df.columns:
                 df.drop(columns=[stochrsi_d_col], errors='ignore', inplace=True)
        else:
             if 'stochrsi_d' not in df.columns:
                logger.warning(f"StochRSI D column matching hint '{stoch_suffix_hint}' not found after calculation. Check pandas_ta naming/behavior.")
                df['stochrsi_d'] = pd.NA


        # --- Calculate Momentum ---
        logger.debug(f"Scrying (Momentum): Calculating with length={mom_len}")
        try:
            df.ta.mom(length=mom_len, append=True)
        except Exception as ta_mom_err:
            logger.error(f"{Fore.RED}Scrying (Momentum): pandas_ta calculation failed: {ta_mom_err}{Style.RESET_ALL}")
            df['momentum'] = pd.NA # Nullify momentum column

        mom_col_name_hint = f"MOM_{mom_len}" # Standard pandas_ta name
        mom_col = find_pandas_ta_column(df, mom_col_name_hint)

        if mom_col and mom_col in df.columns:
            df['momentum'] = df[mom_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            # Clean up raw momentum column if different
            if mom_col != 'momentum' and mom_col in df.columns:
                df.drop(columns=[mom_col], errors='ignore', inplace=True)
        else:
            if 'momentum' not in df.columns: # Only log/set NA if not already set by error handling
                logger.warning(f"Momentum column matching hint '{mom_col_name_hint}' not found after calculation. Check pandas_ta naming/behavior.")
                df['momentum'] = pd.NA

        # Check for NaNs in critical output columns (last row)
        last_row = df.iloc[-1]
        if last_row[target_cols].isnull().any():
            nan_cols = last_row[target_cols].isnull()
            nan_details = ', '.join([col for col in target_cols if nan_cols[col]])
            logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}")

        # Clean up any other unexpected raw columns created by pandas_ta
        # cols_to_drop = [c for c in set(df.columns) - initial_columns if c not in target_cols and c in new_columns]
        # if cols_to_drop:
        #     logger.debug(f"Dropping potentially orphaned pandas_ta columns: {cols_to_drop}")
        #     df.drop(columns=cols_to_drop, errors='ignore', inplace=True)


        # Log latest values for debugging
        k_val = df['stochrsi_k'].iloc[-1]
        d_val = df['stochrsi_d'].iloc[-1]
        mom_val = df['momentum'].iloc[-1]

        # Check for NaN before formatting/coloring
        k_str = f"{k_val:.2f}" if not k_val.is_nan() else "NaN"
        d_str = f"{d_val:.2f}" if not d_val.is_nan() else "NaN"
        mom_str = f"{mom_val:.4f}" if not mom_val.is_nan() else "NaN"

        k_color = Style.RESET_ALL
        d_color = Style.RESET_ALL
        mom_color = Style.RESET_ALL

        if not k_val.is_nan():
            k_color = Fore.RED if k_val > CONFIG.stochrsi_overbought else (Fore.GREEN if k_val < CONFIG.stochrsi_oversold else Fore.CYAN)
        if not d_val.is_nan():
            d_color = Fore.RED if d_val > CONFIG.stochrsi_overbought else (Fore.GREEN if d_val < CONFIG.stochrsi_oversold else Fore.CYAN)
        if not mom_val.is_nan():
            mom_color = Fore.GREEN if mom_val > CONFIG.position_qty_epsilon else (Fore.RED if mom_val < -CONFIG.position_qty_epsilon else Fore.WHITE)

        logger.debug(f"Scrying (StochRSI/Mom): K={k_color}{k_str}{Style.RESET_ALL}, D={d_color}{d_str}{Style.RESET_ALL}, Mom({mom_len})={mom_color}{mom_str}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df

def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """
    Calculates the Ehlers Fisher Transform indicator using pandas_ta, robustly.

    Args:
        df: Pandas DataFrame with 'high', 'low' columns.
        length: The lookback period for the Fisher Transform calculation.
        signal: The smoothing period for the signal line (often 1 for trigger-only).

    Returns:
        The DataFrame with added columns:
        - 'ehlers_fisher': The Fisher Transform value (Decimal or NaN).
        - 'ehlers_signal': The Fisher Transform signal line value (Decimal or NaN).
        Returns original DataFrame with NA/NaN columns if calculation fails or data is insufficient.
    """
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    required_input_cols = ["high", "low"]
    min_len = length + signal + 50 # Increased buffer

    # Input validation
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): DataFrame is None or empty. Cannot calculate.{Style.RESET_ALL}")
        return df
    if not all(c in df.columns for c in required_input_cols):
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): Missing required columns {required_input_cols}. Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    if len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Len: {len(df)}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        initial_columns = set(df.columns)

        logger.debug(f"Scrying (EhlersFisher): Calculating with length={length}, signal={signal}")
        try:
            # Ensure high/low are float for pandas_ta
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df.ta.fisher(length=length, signal=signal, append=True)
        except Exception as ta_fisher_err:
             logger.error(f"{Fore.RED}Scrying (EhlersFisher): pandas_ta calculation failed: {ta_fisher_err}{Style.RESET_ALL}")
             df['ehlers_fisher'] = pd.NA
             df['ehlers_signal'] = pd.NA
             return df

        # Find the actual Fisher columns using hints
        # Typical pandas_ta names: FISHERT_length_signal, FISHERTs_length_signal
        fisher_suffix_hint = f"_{length}_{signal}"
        fisher_col = find_pandas_ta_column(df, "FISHERT", suffix_hint=fisher_suffix_hint)
        signal_col = find_pandas_ta_column(df, "FISHERTs", suffix_hint=fisher_suffix_hint)

        # Assign results and convert to Decimal
        if fisher_col and fisher_col in df.columns:
            df['ehlers_fisher'] = df[fisher_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            if fisher_col != 'ehlers_fisher' and fisher_col in df.columns:
                 df.drop(columns=[fisher_col], errors='ignore', inplace=True)
        else:
            if 'ehlers_fisher' not in df.columns:
                logger.warning(f"Ehlers Fisher column matching hint '{fisher_suffix_hint}' not found after calculation. Check pandas_ta naming/behavior.")
                df['ehlers_fisher'] = pd.NA

        if signal_col and signal_col in df.columns:
            df['ehlers_signal'] = df[signal_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
            if signal_col != 'ehlers_signal' and signal_col in df.columns:
                 df.drop(columns=[signal_col], errors='ignore', inplace=True)
        else:
            # If signal=1, pandas_ta might not create a separate signal column, often it's the same as the fisher line shifted by 1
            if signal == 1 and 'ehlers_fisher' in df.columns:
                 logger.debug(f"Ehlers Fisher signal length is 1, using previous Fisher value as signal.")
                 # Use the previous Fisher value as the signal for signal=1
                 df['ehlers_signal'] = df['ehlers_fisher'].shift(1)
            elif 'ehlers_signal' not in df.columns:
                 logger.warning(f"Ehlers Signal column matching hint '{fisher_suffix_hint}' not found after calculation. Check pandas_ta naming/behavior.")
                 df['ehlers_signal'] = pd.NA

        # Check for NaNs in critical output columns (last row)
        last_row = df.iloc[-1]
        if last_row[target_cols].isnull().any():
            nan_cols = last_row[target_cols].isnull()
            nan_details = ', '.join([col for col in target_cols if nan_cols[col]])
            logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}")

        # Clean up any other unexpected raw columns
        # cols_to_drop = [c for c in set(df.columns) - initial_columns if c not in target_cols and c in new_columns]
        # if cols_to_drop: df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        # Log latest values for debugging
        fish_val = df['ehlers_fisher'].iloc[-1]
        sig_val = df['ehlers_signal'].iloc[-1]
        fish_str = f"{fish_val:.4f}" if not fish_val.is_nan() else "NaN"
        sig_str = f"{sig_val:.4f}" if not sig_val.is_nan() else "NaN"

        logger.debug(f"Scrying (EhlersFisher({length},{signal})): Fisher={Fore.CYAN}{fish_str}{Style.RESET_ALL}, Signal={Fore.MAGENTA}{sig_str}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df

def calculate_ema_cross(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """
    Calculates standard Exponential Moving Averages (EMA) for the EMA Cross strategy.

    *** WARNING: This uses standard EMAs and is NOT an Ehlers Super Smoother filter. ***
    The strategy name 'EMA_CROSS' reflects this. If you require true Ehlers filters,
    replace this calculation logic.

    Args:
        df: Pandas DataFrame with 'close' column.
        fast_len: The period for the fast EMA.
        slow_len: The period for the slow EMA.

    Returns:
        The DataFrame with added columns:
        - 'fast_ema': The fast EMA value (Decimal or NaN).
        - 'slow_ema': The slow EMA value (Decimal or NaN).
        Returns original DataFrame with NA/NaN columns if calculation fails or data is insufficient.
    """
    target_cols = ['fast_ema', 'slow_ema']
    required_input_cols = ["close"]
    # EMA needs buffer for stability, especially the slower one
    min_len = slow_len + 50 # Increased buffer

    # Input validation
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Scrying (EMA Cross): DataFrame is None or empty. Cannot calculate.{Style.RESET_ALL}")
        return df
    if not all(c in df.columns for c in required_input_cols):
        logger.warning(f"{Fore.YELLOW}Scrying (EMA Cross): Missing required column 'close'. Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    if len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EMA Cross): Insufficient data (Len: {len(df)}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # *** PYRMETHUS NOTE / WARNING ***
        logger.warning(f"{Fore.YELLOW}{Style.DIM}Scrying (EMA Cross): Using standard EMA. "
                       f"This strategy path ('EMA_CROSS') uses standard EMAs and may not perform as a true Ehlers MA strategy. "
                       f"Verify indicator suitability or implement actual Ehlers Super Smoother if needed.{Style.RESET_ALL}")

        initial_columns = set(df.columns)

        logger.debug(f"Scrying (EMA Cross): Calculating Fast EMA({fast_len}), Slow EMA({slow_len})")
        # Use pandas_ta standard EMA calculation
        # Calculate separately to find specific column names more easily
        try:
            df.ta.ema(length=fast_len, append=True)
            df.ta.ema(length=slow_len, append=True)
        except Exception as ta_ema_err:
             logger.error(f"{Fore.RED}Scrying (EMA Cross): pandas_ta calculation failed: {ta_ema_err}{Style.RESET_ALL}")
             df['fast_ema'] = pd.NA
             df['slow_ema'] = pd.NA
             return df


        # Find the actual EMA columns using hints
        # Typical pandas_ta names: EMA_length
        fast_ema_col = find_pandas_ta_column(df, "EMA", suffix_hint=f"_{fast_len}")
        slow_ema_col = find_pandas_ta_column(df, "EMA", suffix_hint=f"_{slow_len}")

        # Assign results and convert to Decimal
        if fast_ema_col and fast_ema_col in df.columns:
             df['fast_ema'] = df[fast_ema_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
             if fast_ema_col != 'fast_ema' and fast_ema_col in df.columns:
                 df.drop(columns=[fast_ema_col], errors='ignore', inplace=True)
        else:
             if 'fast_ema' not in df.columns:
                 logger.warning(f"Fast EMA column matching hint 'EMA_{fast_len}' not found after calculation. Check pandas_ta naming/behavior.")
                 df['fast_ema'] = pd.NA

        if slow_ema_col and slow_ema_col in df.columns:
             df['slow_ema'] = df[slow_ema_col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))
             if slow_ema_col != 'slow_ema' and slow_ema_col in df.columns:
                 df.drop(columns=[slow_ema_col], errors='ignore', inplace=True)
        else:
             if 'slow_ema' not in df.columns:
                 logger.warning(f"Slow EMA column matching hint 'EMA_{slow_len}' not found after calculation. Check pandas_ta naming/behavior.")
                 df['slow_ema'] = pd.NA

        # Check for NaNs in critical output columns (last row)
        last_row = df.iloc[-1]
        if last_row[target_cols].isnull().any():
            nan_cols = last_row[target_cols].isnull()
            nan_details = ', '.join([col for col in target_cols if nan_cols[col]])
            logger.warning(f"{Fore.YELLOW}Scrying (EMA Cross): Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}")

        # Clean up any other unexpected raw columns
        # cols_to_drop = [c for c in set(df.columns) - initial_columns if c not in target_cols and c in new_columns]
        # if cols_to_drop: df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        # Log latest values for debugging
        fast_val = df['fast_ema'].iloc[-1]
        slow_val = df['slow_ema'].iloc[-1]
        fast_str = f"{fast_val:.4f}" if not fast_val.is_nan() else "NaN"
        slow_str = f"{slow_val:.4f}" if not slow_val.is_nan() else "NaN"

        cross_color = Style.RESET_ALL
        if not fast_val.is_nan() and not slow_val.is_nan():
            cross_color = Fore.GREEN if fast_val > slow_val else Fore.RED

        logger.debug(f"Scrying (EMA Cross({fast_len},{slow_len})): Fast={cross_color}{fast_str}{Style.RESET_ALL}, Slow={cross_color}{slow_str}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EMA Cross): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Nullify results on error
    return df


def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> Dict[str, Any]:
    """
    Fetches and analyzes the L2 order book to calculate bid/ask volume ratio and spread.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        depth: The number of price levels on each side (bids/asks) to include in the volume ratio calculation.
        fetch_limit: The number of price levels to request from the exchange API (should be >= depth).

    Returns:
        A dictionary containing:
        - 'bid_ask_ratio': Ratio of total bid volume to total ask volume within the specified depth (Decimal) or Decimal('NaN').
        - 'spread': Difference between best ask and best bid (Decimal) or Decimal('NaN').
        - 'best_bid': The highest bid price (Decimal) or Decimal('NaN').
        - 'best_ask': The lowest ask price (Decimal) or Decimal('NaN').
        - 'fetched_this_cycle': True (internal tracking, not strictly OB data)
        Returns NaN values if analysis fails or data is unavailable.
    """
    # Ensure CONFIG is accessible
    if CONFIG is None:
        logger.error("analyze_order_book: CONFIG not loaded!")
        # Return NaNs as we cannot proceed without config (e.g., for epsilon)
        return {"bid_ask_ratio": Decimal('NaN'), "spread": Decimal('NaN'), "best_bid": Decimal('NaN'), "best_ask": Decimal('NaN'), "fetched_this_cycle": True}

    results: Dict[str, Any] = {"bid_ask_ratio": Decimal('NaN'), "spread": Decimal('NaN'), "best_bid": Decimal('NaN'), "best_ask": Decimal('NaN'), "fetched_this_cycle": True}
    logger.debug(f"Order Book Scrying: Fetching L2 {symbol} (Depth:{depth}, Request Limit:{fetch_limit})...")

    # Check if the exchange supports fetching L2 order book
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}Order Book Scrying: Exchange '{exchange.id}' does not support fetchL2OrderBook method. Cannot analyze depth.{Style.RESET_ALL}")
        results["fetched_this_cycle"] = False # Mark as not fetched if not supported
        return results # Return defaults (all NaN)

    try:
        # Fetching the order book's current state
        # Bybit V5 requires 'category' param for futures (should be handled by default options)
        params = {} # Keep params empty, rely on default options set in initialize_exchange
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit, params=params)
        bids: List[List[Union[float, str]]] = order_book.get('bids', []) # List of [price, amount]
        asks: List[List[Union[float, str]]] = order_book.get('asks', []) # List of [price, amount]

        if not bids or not asks:
             logger.warning(f"{Fore.YELLOW}Order Book Scrying: Empty bids or asks received for {symbol}. Cannot analyze.{Style.RESET_ALL}")
             return results # Return defaults (all NaN)

        # Extract best bid/ask with Decimal precision
        # Ensure lists are not empty and contain price/amount pairs, then safely convert
        best_bid = safe_decimal_conversion(bids[0][0], default=Decimal('NaN')) if bids and len(bids[0]) > 0 else Decimal('NaN')
        best_ask = safe_decimal_conversion(asks[0][0], default=Decimal('NaN')) if asks and len(asks[0]) > 0 else Decimal('NaN')
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Calculate spread
        if not best_bid.is_nan() and not best_ask.is_nan() and best_bid > 0 and best_ask > 0:
            try:
                spread_val = best_ask - best_bid
                # Check if spread is reasonable (e.g., not negative)
                if spread_val >= 0:
                    results["spread"] = spread_val
                    logger.debug(f"OB Scrying: Best Bid={Fore.GREEN}{best_bid:.4f}{Style.RESET_ALL}, Best Ask={Fore.RED}{best_ask:.4f}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results['spread']:.4f}{Style.RESET_ALL}")
                else:
                    logger.warning(f"OB Scrying: Negative spread calculated ({spread_val:.4f}). Bid={best_bid:.4f}, Ask={best_ask:.4f}. Check OB data. Setting spread to NaN.")
                    results["spread"] = Decimal('NaN')
            except (InvalidOperation, TypeError) as e:
                 logger.warning(f"{Fore.YELLOW}Error calculating spread: {e}. Skipping spread.{Style.RESET_ALL}")
                 results["spread"] = Decimal('NaN')
        else:
            logger.debug(f"OB Scrying: Best Bid={best_bid if not best_bid.is_nan() else 'N/A'}, Best Ask={best_ask if not best_ask.is_nan() else 'N/A'} (Spread calculation skipped due to invalid bid/ask)")
            results["spread"] = Decimal('NaN') # Ensure it's NaN if not calculated

        # Sum total volume within the specified depth using Decimal for precision
        # Ensure list slicing doesn't go out of bounds and elements are valid pairs
        # Use generator expression with safe conversion, defaulting to 0 for invalid entries
        # Depth should be min(configured_depth, actual_levels_received)
        actual_bid_depth = min(depth, len(bids))
        actual_ask_depth = min(depth, len(asks))

        bid_vol = sum(safe_decimal_conversion(bid[1], default=Decimal("0.0")) for bid in bids[:actual_bid_depth] if len(bid) >= 2)
        ask_vol = sum(safe_decimal_conversion(ask[1], default=Decimal("0.0")) for ask in asks[:actual_ask_depth] if len(ask) >= 2)
        logger.debug(f"OB Scrying (Depth {depth}, Actual Bids:{actual_bid_depth}, Asks:{actual_ask_depth}): Total BidVol={Fore.GREEN}{bid_vol:.4f}{Style.RESET_ALL}, Total AskVol={Fore.RED}{ask_vol:.4f}{Style.RESET_ALL}")

        # Calculate Bid/Ask Volume Ratio (Total Bid Volume / Total Ask Volume)
        if ask_vol > CONFIG.position_qty_epsilon: # Avoid division by zero or near-zero
            try:
                ratio = bid_vol / ask_vol
                if ratio.is_infinite() or ratio.is_nan():
                     logger.warning(f"Order Book ratio calculation resulted in {ratio}. Setting to NaN.")
                     results["bid_ask_ratio"] = Decimal('NaN')
                else:
                    results["bid_ask_ratio"] = ratio
                    # Determine color based on configured thresholds for logging
                    ratio_color = Fore.YELLOW # Default neutral
                    if not ratio.is_nan():
                        if ratio >= CONFIG.order_book_ratio_threshold_long: ratio_color = Fore.GREEN
                        elif ratio <= CONFIG.order_book_ratio_threshold_short: ratio_color = Fore.RED
                    logger.debug(f"OB Scrying Ratio (Bids/Asks): {ratio_color}{results['bid_ask_ratio']:.3f}{Style.RESET_ALL}")

            except (DivisionByZero, InvalidOperation, Exception) as e:
                logger.warning(f"{Fore.YELLOW}Error calculating OB ratio: {e}{Style.RESET_ALL}")
                results["bid_ask_ratio"] = Decimal('NaN')
        else:
            logger.debug(f"OB Scrying Ratio: N/A (Ask volume within depth {depth} is zero or negligible: {ask_vol:.4f})")
            results["bid_ask_ratio"] = Decimal('NaN') # Set explicitly to NaN

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        results["fetched_this_cycle"] = False # Mark as not successfully fetched
        # Keep results as NaN on API errors
    except IndexError:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error: Index out of bounds accessing bids/asks for {symbol}. Order book data might be malformed or incomplete.{Style.RESET_ALL}")
        results["fetched_this_cycle"] = False
        # Keep results as NaN
    except Exception as e:
        logger.warning(f"{Fore.YELLOW}Unexpected Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results["fetched_this_cycle"] = False
        # Keep results as NaN

    # Ensure results dictionary keys exist even if errors occurred
    results.setdefault("bid_ask_ratio", Decimal('NaN'))
    results.setdefault("spread", Decimal('NaN'))
    results.setdefault("best_bid", Decimal('NaN'))
    results.setdefault("best_ask", Decimal('NaN'))
    # fetched_this_cycle is already set correctly based on success/failure
    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV data, prepares it as a pandas DataFrame, ensures numeric types,
    and handles missing values (NaNs) robustly.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        interval: The timeframe interval (e.g., '1m', '5m').
        limit: The number of candles to fetch.

    Returns:
        A pandas DataFrame containing the OHLCV data with a datetime index (UTC),
        or None if fetching or processing fails or data is unusable.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("get_market_data: CONFIG not loaded!")
        return None

    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})...")
        # Channeling the data stream from the exchange
        # Bybit V5 requires 'category' param for futures (should be handled by default options)
        params = {} # Rely on default category set in exchange options
        ohlcv: List[List[Union[int, float, str]]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit, params=params)

        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Market might be inactive, symbol incorrect, or API issue.{Style.RESET_ALL}")
            return None

        if len(ohlcv) < limit:
             logger.warning(f"{Fore.YELLOW}Data Fetch: Only received {len(ohlcv)} candles for {symbol} ({interval}) instead of requested {limit}. Data history might be limited for this symbol/interval.{Style.RESET_ALL}")
             # Check if received data is critically low compared to buffer needed
             min_required_for_indicators = CONFIG.api_fetch_limit_buffer # Use buffer as minimum indicator need proxy
             if len(ohlcv) < min_required_for_indicators:
                  logger.error(f"{Fore.RED}Data Fetch: Received critically low amount of data ({len(ohlcv)} < {min_required_for_indicators}). Cannot proceed reliably.{Style.RESET_ALL}")
                  return None


        # Weaving data into a DataFrame structure
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # --- Data Cleaning and Preparation ---
        # Convert timestamp to datetime objects (UTC) and set as index
        try:
            # Ensure timestamp is treated as int64 before conversion if coming directly from JSON/API
            # Handle potential string timestamps as well
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str), unit="ms", utc=True, errors='coerce')
            if df["timestamp"].isnull().any():
                 raise ValueError("Timestamp conversion resulted in NaT values.")
            df.set_index("timestamp", inplace=True)
            # Sort index just in case data isn't perfectly ordered
            df.sort_index(inplace=True)
        except (ValueError, TypeError, OverflowError, Exception) as time_e:
             logger.error(f"{Fore.RED}Data Fetch: Error converting timestamp column: {time_e}{Style.RESET_ALL}")
             logger.debug(f"Problematic timestamp data (first 5): {df['timestamp'].head().tolist()}")
             return None # Cannot proceed without valid timestamps

        # Ensure OHLCV columns are numeric, coercing errors to NaN
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Robust NaN Handling ---
        initial_nan_count = df[numeric_cols].isnull().sum().sum()
        if initial_nan_count > 0:
            nan_counts_per_col = df[numeric_cols].isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: Found {initial_nan_count} NaN values in OHLCV data after conversion:\n"
                           f"{nan_counts_per_col[nan_counts_per_col > 0]}\nAttempting forward fill (ffill)...{Style.RESET_ALL}")
            df.ffill(inplace=True) # Fill NaNs with the previous valid observation

            # Check if NaNs remain (likely at the beginning of the series if data history is short)
            remaining_nan_count = df[numeric_cols].isnull().sum().sum()
            if remaining_nan_count > 0:
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill ({remaining_nan_count}). Attempting backward fill (bfill)...{Style.RESET_ALL}")
                df.bfill(inplace=True) # Fill remaining NaNs with the next valid observation

                # Final check: if NaNs still exist, data is likely too gappy at start/end or completely invalid
                final_nan_count = df[numeric_cols].isnull().sum().sum()
                if final_nan_count > 0:
                    logger.error(f"{Fore.RED}Data Fetch: Unfillable NaN values ({final_nan_count}) remain after ffill and bfill. "
                                 f"Data quality insufficient for {symbol}. Columns with NaNs:\n{df[numeric_cols].isnull().sum()[df[numeric_cols].isnull().sum() > 0]}\nSkipping cycle.{Style.RESET_ALL}")
                    return None # Cannot proceed with unreliable data

        # Check if the last candle is potentially incomplete (timestamp is very close to now)
        # This check is heuristic.
        if len(df) > 0:
            last_candle_time_utc = df.index[-1]
            now_utc = pd.Timestamp.now(tz='UTC')
            try:
                interval_seconds = exchange.parse_timeframe(interval) # Convert interval string to seconds
                # Allow a small buffer (e.g., 5 seconds or 10% of interval, whichever is smaller) for processing/latency
                buffer_seconds = min(5, interval_seconds * 0.1)
                time_diff_seconds = (now_utc - last_candle_time_utc).total_seconds()

                # If the difference is less than the interval minus the buffer, it's likely incomplete
                if time_diff_seconds < (interval_seconds - buffer_seconds):
                    logger.warning(f"{Fore.YELLOW}Data Fetch: Last candle timestamp ({last_candle_time_utc}) for {symbol} ({interval}) "
                                   f"is very recent ({time_diff_seconds:.1f}s ago, interval is {interval_seconds}s). "
                                   f"It might be incomplete. Using it anyway as typical for scalping, but be aware.{Style.RESET_ALL}")
            except Exception as tf_parse_err:
                logger.warning(f"Could not parse timeframe '{interval}' for completeness check: {tf_parse_err}")


        logger.debug(f"Data Fetch: Successfully woven and cleaned {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: Disturbance gathering OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return None # Return None if any error occurred

# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches current position details using Bybit V5 API specifics (`fetchPositions`).
    Assumes One-Way Mode (looks for positionIdx=0). Includes parsing attached SL/TP/TSL.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').

    Returns:
        A dictionary containing:
        - 'side': Position side ('Long', 'Short', or 'None').
        - 'qty': Position quantity as Decimal (0.0 if flat, NaN on error).
        - 'entry_price': Average entry price as Decimal (0.0 if flat, NaN on error).
        - 'leverage': Effective leverage for the symbol (Decimal) or NaN.
        - 'stop_loss': Native Stop Loss price (Decimal) or None if not set/found.
        - 'take_profit': Native Take Profit price (Decimal) or None if not set/found.
        - 'trailing_stop_price': Native Trailing Stop trigger price (Decimal) or None if not set/found.
        - 'unrealized_pnl': Unrealized PNL (Decimal) or NaN.
        - 'liquidation_price': Liquidation Price (Decimal) or NaN.
        Returns default flat state dictionary with NaNs/None on error or if no position found.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("get_current_position: CONFIG not loaded!")
        # Return default state with NaNs/None to indicate error
        return {
            'side': "None", 'qty': Decimal("NaN"), 'entry_price': Decimal("NaN"),
            'leverage': Decimal("NaN"), 'stop_loss': None, 'take_profit': None,
            'trailing_stop_price': None, 'unrealized_pnl': Decimal("NaN"), 'liquidation_price': Decimal("NaN"),
        }

    # Initialize default structure with NaNs/None for fields
    default_pos: Dict[str, Any] = {
        'side': CONFIG.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0"),
        'leverage': Decimal("NaN"), 'stop_loss': None, 'take_profit': None,
        'trailing_stop_price': None, 'unrealized_pnl': Decimal("NaN"), 'liquidation_price': Decimal("NaN"),
    }
    market_id: Optional[str] = None
    market: Optional[Dict[str, Any]] = None
    category: Optional[str] = None # Will rely on defaultCategory from exchange options

    try:
        # Get market details (primarily for logging/context, category is default)
        market = exchange.market(symbol)
        market_id = market['id'] # The exchange's specific ID (e.g., BTCUSDT)
        category = exchange.options.get('defaultCategory', 'linear') # Get category from options

    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(f"{Fore.RED}Position Check: Failed to identify market structure for '{symbol}': {e}. Cannot check position.{Style.RESET_ALL}")
        # Return default state with NaNs/None to signal error
        default_pos.update({k: Decimal("NaN") for k in default_pos if isinstance(default_pos[k], Decimal)})
        default_pos.update({k: None for k in default_pos if default_pos[k] is None})
        return default_pos
    except Exception as e_market:
        logger.error(f"{Fore.RED}Position Check: Unexpected error getting market info for '{symbol}': {e_market}. Cannot check position.{Style.RESET_ALL}")
        default_pos.update({k: Decimal("NaN") for k in default_pos if isinstance(default_pos[k], Decimal)})
        default_pos.update({k: None for k in default_pos if default_pos[k] is None})
        return default_pos

    try:
        # Check if the exchange instance supports fetchPositions (should for Bybit V5 via CCXT)
        if not exchange.has.get('fetchPositions'):
            logger.error(f"{Fore.RED}Position Check: Exchange '{exchange.id}' CCXT instance does not support fetchPositions method. Cannot get V5 position data.{Style.RESET_ALL}")
            default_pos.update({k: Decimal("NaN") for k in default_pos if isinstance(default_pos[k], Decimal)})
            default_pos.update({k: None for k in default_pos if default_pos[k] is None})
            return default_pos # Return default state with NaNs/None

        # Fetch positions for the specific symbol and category
        # CCXT fetchPositions should use the defaultCategory set in options
        params = {} # No extra params needed if defaultCategory is set
        positions = exchange.fetch_positions(symbols=[symbol], params=params) # Filter by unified symbol

        # Filter for the relevant position in One-Way mode (positionIdx=0)
        # Ensure positionIdx is compared as integer 0 as API might return it as int or string
        relevant_position = next(
            (p for p in positions if p.get('info', {}).get('positionIdx', -1) == 0 and p.get('symbol') == symbol),
            None
        )

        # Use safe_decimal_conversion with NaN default for numeric fields, None for stops initially
        if relevant_position:
            pos_info = relevant_position.get('info', {}) # Access raw info dictionary
            pos_size = safe_decimal_conversion(pos_info.get('size', 'NaN'), default=Decimal('NaN'))

            if not pos_size.is_nan() and pos_size > CONFIG.position_qty_epsilon:
                # Active position found
                pos_side_raw = pos_info.get('side', '').capitalize() # "Buy" or "Sell"
                pos_avg_entry = safe_decimal_conversion(pos_info.get('avgPrice', pos_info.get('entryPrice', 'NaN')), default=Decimal('NaN')) # Use avgPrice first
                pos_leverage = safe_decimal_conversion(pos_info.get('leverage', 'NaN'), default=Decimal('NaN'))
                # Native stops - use None default in conversion to distinguish not set vs. failed conversion
                pos_stop_loss = safe_decimal_conversion(pos_info.get('stopLoss'), default=None)
                pos_take_profit = safe_decimal_conversion(pos_info.get('takeProfit'), default=None)
                pos_trailing_stop_price = safe_decimal_conversion(pos_info.get('trailingStop'), default=None) # TSL trigger price
                pos_unrealized_pnl = safe_decimal_conversion(pos_info.get('unrealisedPnl', 'NaN'), default=Decimal('NaN'))
                pos_liquidation_price = safe_decimal_conversion(pos_info.get('liqPrice', 'NaN'), default=Decimal('NaN'))

                # Map Bybit side ("Buy"/"Sell") to internal representation
                pos_side = CONFIG.pos_long if pos_side_raw == "Buy" else (CONFIG.pos_short if pos_side_raw == "Sell" else CONFIG.pos_none)

                if pos_side == CONFIG.pos_none:
                     logger.warning(f"Position Check: Found position with size {pos_size} but unknown side '{pos_side_raw}'. Treating as flat.")
                     return default_pos # Return flat state if side is unclear

                logger.info(f"{Fore.CYAN}Position Check: Active {pos_side} position found for {symbol}. "
                            f"Qty: {pos_size.normalize()}, Entry: {pos_avg_entry.normalize() if not pos_avg_entry.is_nan() else 'N/A'}{Style.RESET_ALL}")

                # Log native stops if attached (non-None and positive)
                stop_details = []
                if pos_stop_loss is not None and pos_stop_loss > CONFIG.position_qty_epsilon:
                     stop_details.append(f"SL: {pos_stop_loss.normalize()}")
                if pos_take_profit is not None and pos_take_profit > CONFIG.position_qty_epsilon:
                     stop_details.append(f"TP: {pos_take_profit.normalize()}")
                if pos_trailing_stop_price is not None and pos_trailing_stop_price > CONFIG.position_qty_epsilon:
                     stop_details.append(f"TSL Trigger: {pos_trailing_stop_price.normalize()}")

                if stop_details:
                     logger.info(f"{Fore.CYAN}Position Check: Attached Stops -> {' | '.join(stop_details)}{Style.RESET_ALL}")
                else:
                     logger.info(f"{Fore.CYAN}Position Check: No active native SL/TP/TSL detected on the position (or values are zero/None).{Style.RESET_ALL}")

                # Return the detailed position state
                return {
                    'side': pos_side, 'qty': pos_size, 'entry_price': pos_avg_entry,
                    'leverage': pos_leverage, 'stop_loss': pos_stop_loss,
                    'take_profit': pos_take_profit, 'trailing_stop_price': pos_trailing_stop_price,
                    'unrealized_pnl': pos_unrealized_pnl, 'liquidation_price': pos_liquidation_price,
                }
            else:
                # Position found but size is zero or negligible, treat as flat
                logger.info(f"{Fore.BLUE}Position Check: One-Way (positionIdx=0) position found for {symbol} but size is {pos_size}. Currently Flat.{Style.RESET_ALL}")
                return default_pos
        else:
            # No active position found (either list is empty, or no position with positionIdx=0)
            logger.info(f"{Fore.BLUE}Position Check: No active One-Way (positionIdx=0) position found for {symbol}. Currently Flat.{Style.RESET_ALL}")
            return default_pos

    except ccxt.BadSymbol:
        logger.error(f"{Fore.RED}Position Check Error: Invalid symbol '{symbol}' during fetchPositions.{Style.RESET_ALL}")
        default_pos.update({k: Decimal("NaN") for k in default_pos if isinstance(default_pos[k], Decimal)})
        default_pos.update({k: None for k in default_pos if default_pos[k] is None})
        return default_pos
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Position Check Error for {symbol}: {type(e).__name__} - {e}. Cannot get current position state.{Style.RESET_ALL}")
        # Return default state with NaNs/None on temporary API/network errors
        default_pos.update({k: Decimal("NaN") for k in default_pos if isinstance(default_pos[k], Decimal)})
        default_pos.update({k: None for k in default_pos if default_pos[k] is None})
        return default_pos
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check Unexpected Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # Return default state with NaNs/None on unexpected errors
        default_pos.update({k: Decimal("NaN") for k in default_pos if isinstance(default_pos[k], Decimal)})
        default_pos.update({k: None for k in default_pos if default_pos[k] is None})
        return default_pos


def calculate_order_quantity(exchange: ccxt.Exchange, symbol: str, account_equity: Decimal, current_price: Decimal, stop_loss_price: Decimal, side: str, market_data: Dict[str, Any]) -> Optional[Decimal]:
    """
    Calculates the order quantity based on risk percentage, account equity,
    estimated stop loss distance, current price, leverage, and market limits.
    Includes margin checks.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        account_equity: The total account equity in the quote currency (USDT). Use Decimal.
        current_price: The current market price for the symbol. Use Decimal.
        stop_loss_price: The calculated price where the stop loss would trigger. Use Decimal.
        side: The trade side ('buy' or 'sell').
        market_data: Dictionary containing market details for the symbol (from exchange.market(symbol)).

    Returns:
        The calculated order quantity (Decimal) formatted to market precision,
        or None if calculation is not possible or results in zero/negative/invalid quantity or fails margin check.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("calculate_order_quantity: CONFIG not loaded!")
        return None

    # --- Input Validation ---
    if account_equity.is_nan() or current_price.is_nan() or stop_loss_price.is_nan():
         logger.warning(f"{Fore.YELLOW}Qty Calc: Invalid input values (NaN detected: Equity={account_equity}, Price={current_price}, SL={stop_loss_price}). Cannot calculate quantity.{Style.RESET_ALL}")
         return None
    if account_equity <= CONFIG.min_order_value_usdt or current_price <= CONFIG.position_qty_epsilon or stop_loss_price <= CONFIG.position_qty_epsilon or not market_data:
        logger.warning(f"{Fore.YELLOW}Qty Calc: Insufficient equity ({account_equity:.4f}), invalid price ({current_price:.4f}, SL:{stop_loss_price:.4f}), or market data missing. Cannot calculate quantity.{Style.RESET_ALL}")
        return None

    try:
        # Ensure Decimal types
        account_equity_dec = account_equity
        current_price_dec = current_price
        stop_loss_price_dec = stop_loss_price
        leverage_dec = Decimal(str(CONFIG.leverage)) # Ensure leverage from config is Decimal
        risk_percentage_dec = CONFIG.risk_per_trade_percentage
        max_order_usdt_dec = CONFIG.max_order_usdt_amount
        min_order_usdt_value = CONFIG.min_order_value_usdt # From config/constants

        # --- 1. Calculate quantity based on Risk % and SL distance ---
        risk_amount_usdt = account_equity_dec * risk_percentage_dec
        logger.debug(f"Qty Calc: Account Equity: {account_equity_dec:.4f} {CONFIG.usdt_symbol}, Risk %: {risk_percentage_dec:.2%}, Risk Amount: {risk_amount_usdt:.4f} {CONFIG.usdt_symbol}")

        price_diff = (current_price_dec - stop_loss_price_dec).abs()
        if price_diff <= CONFIG.position_qty_epsilon:
            logger.warning(f"{Fore.YELLOW}Qty Calc: Stop Loss price ({stop_loss_price_dec}) is too close or equal to current price ({current_price_dec}). Risk calculation requires a non-zero price difference. Cannot calculate quantity.{Style.RESET_ALL}")
            return None

        try:
             quantity_from_risk = risk_amount_usdt / price_diff
             logger.debug(f"Qty Calc: Price Diff (Entry vs SL): {price_diff.normalize()}, Qty based on Risk: {quantity_from_risk.normalize()} {market_data.get('base', '')}")
        except (DivisionByZero, InvalidOperation) as e:
             logger.error(f"{Fore.RED}Qty Calc: Error calculating quantity from risk: {e}. Price difference likely zero or invalid.{Style.RESET_ALL}")
             return None


        # --- 2. Calculate quantity based on Maximum Order Value ---
        quantity_from_max_value = Decimal('Infinity') # Default to infinity if max value check is disabled or very high
        if max_order_usdt_dec > CONFIG.position_qty_epsilon:
             if current_price_dec > CONFIG.position_qty_epsilon:
                 try:
                     quantity_from_max_value = max_order_usdt_dec / current_price_dec
                     logger.debug(f"Qty Calc: Max Order Value: {max_order_usdt_dec.normalize()} {CONFIG.usdt_symbol}, Qty based on Max Value: {quantity_from_max_value.normalize()} {market_data.get('base', '')}")
                 except (DivisionByZero, InvalidOperation) as e:
                     logger.error(f"{Fore.RED}Qty Calc: Error calculating quantity from max value: {e}. Current price likely zero or invalid.{Style.RESET_ALL}")
                     quantity_from_max_value = Decimal("0.0") # Treat as zero if error
             else:
                 quantity_from_max_value = Decimal("0.0") # Treat as zero if price is invalid
        else:
             logger.debug("Qty Calc: Max Order Value check disabled or set to zero.")


        # --- 3. Determine the minimum quantity (most conservative) ---
        calculated_quantity_dec = min(quantity_from_risk, quantity_from_max_value)
        logger.debug(f"Qty Calc: Calculated Quantity (Min of Risk/MaxValue): {calculated_quantity_dec.normalize()} {market_data.get('base', '')}")

        # --- 4. Apply exchange minimums and step size ---
        limits = market_data.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {}) # For minimum order value

        min_amount = safe_decimal_conversion(amount_limits.get('min'), default=Decimal('0.0'))
        max_amount = safe_decimal_conversion(amount_limits.get('max'), default=Decimal('Infinity'))
        # Get step size (amount precision) - CCXT usually provides this in precision or limits
        amount_precision_str = market_data.get('precision', {}).get('amount') # Number of decimal places
        amount_step = Decimal('NaN')
        if amount_precision_str is not None:
            try:
                # Convert precision (decimal places) to step size (e.g., 3 -> 0.001)
                amount_step = Decimal('1') / (Decimal('10') ** int(amount_precision_str))
            except (ValueError, InvalidOperation):
                logger.warning(f"Could not interpret amount precision '{amount_precision_str}' as integer.")
        # Fallback to limits if precision not available/parsable
        if amount_step.is_nan():
            amount_step = safe_decimal_conversion(amount_limits.get('step'), default=Decimal('NaN'))

        min_cost = safe_decimal_conversion(cost_limits.get('min'), default=CONFIG.min_order_value_usdt) # Use config default if market doesn't specify min cost

        if amount_step.is_nan() or amount_step <= CONFIG.position_qty_epsilon:
             logger.error(f"{Fore.RED}Qty Calc: Market amount step size ('precision.amount' or 'limits.amount.step') not found or invalid ({amount_step}) for {symbol}. Cannot format quantity correctly.{Style.RESET_ALL}")
             return None # Cannot proceed without step size


        # Adjust quantity DOWN to be a multiple of the step size
        if calculated_quantity_dec > CONFIG.position_qty_epsilon:
            # Use floor division logic with Decimals: floor(value / step) * step
            adjusted_quantity_dec = (calculated_quantity_dec // amount_step) * amount_step
            if adjusted_quantity_dec != calculated_quantity_dec:
                 logger.debug(f"Qty Calc: Adjusted quantity {calculated_quantity_dec.normalize()} down to market step {amount_step.normalize()}: {adjusted_quantity_dec.normalize()} {market_data.get('base', '')}")
            calculated_quantity_dec = adjusted_quantity_dec # Update with adjusted value
        else:
             # Quantity was already zero or negative before adjustment
             logger.debug(f"Qty Calc: Quantity {calculated_quantity_dec.normalize()} is zero or negative before step adjustment.")
             calculated_quantity_dec = Decimal("0.0")


        # Check against Min/Max Amount Limits *after* step adjustment
        if calculated_quantity_dec < min_amount and min_amount > CONFIG.position_qty_epsilon:
             logger.warning(f"{Fore.YELLOW}Qty Calc: Adjusted quantity {calculated_quantity_dec.normalize()} is below market minimum amount {min_amount.normalize()}. Cannot place order.{Style.RESET_ALL}")
             return None
        if calculated_quantity_dec > max_amount:
             logger.warning(f"{Fore.YELLOW}Qty Calc: Calculated quantity {calculated_quantity_dec.normalize()} exceeds market maximum amount {max_amount.normalize()}. Clamping to max.{Style.RESET_ALL}")
             calculated_quantity_dec = max_amount
             # Re-adjust to step size after clamping
             calculated_quantity_dec = (calculated_quantity_dec // amount_step) * amount_step


        # Final check on calculated quantity before margin check
        if calculated_quantity_dec <= CONFIG.position_qty_epsilon:
            logger.warning(f"{Fore.YELLOW}Qty Calc: Final calculated quantity {calculated_quantity_dec.normalize()} is zero or negligible after adjustments. Cannot place order.{Style.RESET_ALL}")
            return None

        # --- 5. Estimate Initial Margin Requirement and Check Against Free Margin ---
        leverage_limits = limits.get('leverage', {})
        max_market_leverage = safe_decimal_conversion(leverage_limits.get('max'), default=leverage_dec) # Default to configured if market doesn't specify max
        effective_leverage_for_margin = min(leverage_dec, max_market_leverage)

        if effective_leverage_for_margin <= CONFIG.position_qty_epsilon:
             logger.error(f"{Fore.RED}Qty Calc: Effective leverage zero or invalid ({effective_leverage_for_margin}). Cannot estimate margin.{Style.RESET_ALL}")
             return None

        estimated_position_value = calculated_quantity_dec * current_price_dec
        estimated_margin_required = estimated_position_value / effective_leverage_for_margin
        logger.debug(f"Qty Calc: Estimated Position Value: {estimated_position_value:.4f} {CONFIG.usdt_symbol}")
        logger.debug(f"Qty Calc: Estimated Initial Margin Required (using {effective_leverage_for_margin}x leverage): {estimated_margin_required:.4f} {CONFIG.usdt_symbol}")

        # Fetch *free* balance specifically (usable margin balance)
        try:
            # Use the same params as in initialize_exchange for consistency
            balance_params = {'accountType': 'CONTRACT'} # Or 'UNIFIED'
            balance = exchange.fetch_balance(params=balance_params)
            # Structure might be nested: balance['USDT']['free']
            free_balance_usdt = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol, {}).get('free'), default=Decimal('NaN'))

            if free_balance_usdt.is_nan():
                 logger.error(f"{Fore.RED}Qty Calc: Failed to parse free balance from API response. Cannot perform margin check.{Style.RESET_ALL}")
                 return None

            logger.debug(f"Qty Calc: Available Free Balance: {free_balance_usdt:.4f} {CONFIG.usdt_symbol}")
        except Exception as bal_err:
             logger.error(f"{Fore.RED}Qty Calc: Failed to fetch free balance: {bal_err}. Cannot perform margin check.{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             return None

        # Check if free balance is sufficient with the configured buffer
        required_free_margin = estimated_margin_required * CONFIG.required_margin_buffer
        if free_balance_usdt < required_free_margin:
            logger.warning(f"{Fore.YELLOW}Qty Calc: Insufficient Free Margin. Need ~{required_free_margin:.4f} {CONFIG.usdt_symbol} "
                           f"(includes {CONFIG.required_margin_buffer - 1:.1%} buffer) but have {free_balance_usdt:.4f} {CONFIG.usdt_symbol}. "
                           f"Cannot place order of size {calculated_quantity_dec.normalize()}.{Style.RESET_ALL}")
            send_sms_alert(f"[{CONFIG.strategy_name}] Insufficient Margin for {symbol}. Need {required_free_margin:.2f}, Have {free_balance_usdt:.2f}. Qty calc failed.")
            return None
        else:
             logger.debug(f"Qty Calc: Free margin ({free_balance_usdt:.4f}) sufficient for estimated requirement ({required_free_margin:.4f}).")


        # --- 6. Final Quantity Validation (Min Cost/Value) ---
        # Check if the estimated order value meets the minimum cost requirement
        if estimated_position_value < min_cost:
             logger.warning(f"{Fore.YELLOW}Qty Calc: Estimated order value {estimated_position_value:.4f} {CONFIG.usdt_symbol} is below market minimum cost {min_cost.normalize()} {CONFIG.usdt_symbol}. Cannot place order.{Style.RESET_ALL}")
             return None

        final_quantity_dec = calculated_quantity_dec # Assign to final variable

        logger.info(f"{Fore.GREEN}Qty Calc: Final Calculated Quantity: {final_quantity_dec.normalize()} {market_data.get('base', '')}. "
                    f"Estimated Order Value: {estimated_position_value:.4f} {CONFIG.usdt_symbol}.{Style.RESET_ALL}")

        # Return the final quantity, ensuring it's positive
        return final_quantity_dec.normalize() if final_quantity_dec > CONFIG.position_qty_epsilon else None

    except Exception as e:
        logger.error(f"{Fore.RED}Qty Calc: Unexpected error during quantity calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return None # Return None on unexpected errors


def create_order(exchange: ccxt.Exchange, symbol: str, type: str, side: str, amount: Decimal, price: Optional[Decimal] = None, stop_loss: Optional[Decimal] = None, take_profit: Optional[Decimal] = None, trailing_stop_percentage: Optional[Decimal] = None) -> Optional[Dict[str, Any]]:
    """
    Places an order with native Stop Loss, Take Profit, and Trailing Stop Loss via Bybit V5 API.
    Includes market order fill confirmation.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        type: Order type ('market', 'limit', etc.). This bot primarily uses 'market'.
        side: Order side ('buy' or 'sell').
        amount: The quantity to trade (Decimal). Should be positive.
        price: The price for limit orders (Optional, Decimal). For market orders, this is typically ignored by the exchange execution but used here for calculating TSL activation price.
        stop_loss: Native Stop Loss trigger price (Optional, Decimal).
        take_profit: Native Take Profit trigger price (Optional, Decimal).
        trailing_stop_percentage: Native Trailing Stop percentage (Optional, Decimal, e.g., 0.005 for 0.5%).

    Returns:
        The CCXT order response dictionary if successful (including fill info for market orders), None otherwise.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("create_order: CONFIG not loaded!")
        return None

    if amount.is_nan() or amount <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Create Order: Cannot place order with zero, negative, or NaN amount ({amount}).{Style.RESET_ALL}")
        return None

    # --- Format Parameters ---
    formatted_amount = format_amount(exchange, symbol, amount)
    if formatted_amount == 'NaN':
        logger.error(f"{Fore.RED}Create Order: Failed to format amount {amount}. Cannot proceed.{Style.RESET_ALL}")
        return None

    formatted_price: Optional[str] = None
    if type == 'limit' and price is not None and not price.is_nan():
        formatted_price = format_price(exchange, symbol, price)
        if formatted_price == 'NaN':
             logger.error(f"{Fore.RED}Create Order: Failed to format limit price {price}. Cannot proceed.{Style.RESET_ALL}")
             return None
    elif type == 'limit':
        logger.error(f"{Fore.RED}Create Order: Limit order type specified but price is None or NaN. Cannot proceed.{Style.RESET_ALL}")
        return None

    # Format SL/TP, ensuring they are valid prices
    formatted_stop_loss: Optional[str] = None
    if stop_loss is not None and not stop_loss.is_nan() and stop_loss > CONFIG.position_qty_epsilon:
        formatted_stop_loss = format_price(exchange, symbol, stop_loss)
        if formatted_stop_loss == 'NaN':
            logger.warning(f"{Fore.YELLOW}Create Order: Failed to format Stop Loss price {stop_loss}. SL will not be sent.{Style.RESET_ALL}")
            formatted_stop_loss = None # Do not send invalid SL

    formatted_take_profit: Optional[str] = None
    if take_profit is not None and not take_profit.is_nan() and take_profit > CONFIG.position_qty_epsilon:
        formatted_take_profit = format_price(exchange, symbol, take_profit)
        if formatted_take_profit == 'NaN':
            logger.warning(f"{Fore.YELLOW}Create Order: Failed to format Take Profit price {take_profit}. TP will not be sent.{Style.RESET_ALL}")
            formatted_take_profit = None # Do not send invalid TP


    # --- Prepare Bybit V5 Params ---
    # Category should be handled by default options, but can be explicit if needed
    params: Dict[str, Any] = {} # Rely on default category

    # Add native SL/TP prices to params
    if formatted_stop_loss:
        params['stopLoss'] = formatted_stop_loss
    if formatted_take_profit:
        params['takeProfit'] = formatted_take_profit

    # Add Trailing Stop Loss percentage and optional activation price
    if trailing_stop_percentage is not None and not trailing_stop_percentage.is_nan() and trailing_stop_percentage > CONFIG.position_qty_epsilon:
        # Bybit V5 linear TSL requires percentage * 100 passed as 'trailingStop'
        # Ensure conversion to float is safe
        try:
            tsl_param_value = float(trailing_stop_percentage * 100)
            params['trailingStop'] = tsl_param_value
            logger.debug(f"Create Order: Adding TSL param: trailingStop={tsl_param_value}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting TSL percentage {trailing_stop_percentage} to float: {e}. TSL not sent.")
            if 'trailingStop' in params: del params['trailingStop'] # Remove potentially invalid param


        # Calculate and add activation price if offset is configured and TSL was added
        if 'trailingStop' in params and CONFIG.trailing_stop_activation_offset_percent > CONFIG.position_qty_epsilon:
            # Use the 'price' argument (estimated entry for market orders)
            entry_price_for_activation = price if price is not None and not price.is_nan() else None

            if entry_price_for_activation:
                offset_factor = Decimal("1.0")
                if side == CONFIG.side_buy:
                    offset_factor += CONFIG.trailing_stop_activation_offset_percent
                elif side == CONFIG.side_sell:
                    offset_factor -= CONFIG.trailing_stop_activation_offset_percent

                activation_price = entry_price_for_activation * offset_factor
                formatted_activation_price = format_price(exchange, symbol, activation_price)

                if formatted_activation_price != 'NaN':
                    # Bybit V5 uses 'activePrice' for TSL activation price
                    params['activePrice'] = formatted_activation_price
                    logger.debug(f"Create Order: Calculated TSL activation price: {formatted_activation_price} (based on entry {entry_price_for_activation:.4f})")
                else:
                    logger.warning(f"{Fore.YELLOW}Create Order: Failed to format TSL activation price {activation_price}. Activation price will not be sent.{Style.RESET_ALL}")
            else:
                 logger.warning(f"{Fore.YELLOW}Create Order: TSL activation offset configured, but no valid entry price provided for calculation (using '{price}'). Activation price will not be sent.{Style.RESET_ALL}")


    # --- Log Order Intent ---
    order_intent = f"{Fore.YELLOW}Conjuring Order | Symbol: {symbol}, Type: {type}, Side: {side}, Amount: {formatted_amount}"
    if type == 'limit' and formatted_price:
        order_intent += f", Price: {formatted_price}"
    logger.info(order_intent)
    if formatted_stop_loss: logger.info(f"  Native Stop Loss: {formatted_stop_loss}")
    if formatted_take_profit: logger.info(f"  Native Take Profit: {formatted_take_profit}")
    if 'trailingStop' in params:
         activation_log = f"(Activation Price: {params.get('activePrice', 'Immediate')})" if 'activePrice' in params else "(Activation: Immediate)"
         logger.info(f"  Native Trailing Stop %: {params['trailingStop'] / 100:.2%} {activation_log}")


    # --- Place Order ---
    try:
        # CCXT expects float for amount/price in the main call signature
        amount_float = float(amount)
        price_float = float(price) if type == 'limit' and price is not None and not price.is_nan() else None

        order = exchange.create_order(
            symbol=symbol,
            type=type,
            side=side,
            amount=amount_float,
            price=price_float, # None for market orders
            params=params # Pass SL/TP/TSL/Category here
        )

        order_id = order.get('id')
        order_status = order.get('status', 'unknown') # Default to unknown if status missing

        logger.info(f"{Fore.GREEN}Order Conjured! | ID: {format_order_id(order_id)}, Status: {order_status}{Style.RESET_ALL}")
        # Use more detailed SMS
        sms_msg = f"[{CONFIG.strategy_name}/{symbol}] {side.upper()} Order Placed. ID: {format_order_id(order_id)}, Qty: {formatted_amount}, Status: {order_status}."
        if formatted_stop_loss: sms_msg += f" SL:{formatted_stop_loss}"
        if formatted_take_profit: sms_msg += f" TP:{formatted_take_profit}"
        if 'trailingStop' in params: sms_msg += f" TSL:{params['trailingStop']/100:.1f}%"
        send_sms_alert(sms_msg)


        # --- Wait for Market Order Fill Confirmation ---
        if type == 'market':
            logger.debug(f"Waiting up to {CONFIG.order_fill_timeout_seconds}s for market order {format_order_id(order_id)} fill confirmation...")
            filled_order = None
            start_time = time.time()
            last_status_checked = order_status

            while time.time() - start_time < CONFIG.order_fill_timeout_seconds:
                try:
                    # Fetch order details, rely on default category
                    fetch_params = {}
                    fetched_order = exchange.fetch_order(order_id, symbol, params=fetch_params)
                    current_status = fetched_order.get('status')

                    if current_status != last_status_checked:
                        logger.debug(f"Order {format_order_id(order_id)} status changed: {last_status_checked} -> {current_status}")
                        last_status_checked = current_status

                    if fetched_order and current_status == 'closed':
                        filled_order = fetched_order
                        logger.debug(f"Market order {format_order_id(order_id)} detected as 'closed'.")
                        break # Exit wait loop

                    # Small delay before next poll
                    # Use shorter initial delays, longer later delays
                    elapsed = time.time() - start_time
                    delay = 0.5 if elapsed < 5 else (1.0 if elapsed < 10 else 2.0)
                    time.sleep(min(delay, CONFIG.order_fill_timeout_seconds - elapsed))


                except ccxt.OrderNotFound:
                     # This might happen briefly after submission before it's fully registered
                     logger.debug(f"Order {format_order_id(order_id)} not found yet, retrying check...")
                     time.sleep(1)
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as poll_err:
                     logger.warning(f"{Fore.YELLOW}Error polling order status for {format_order_id(order_id)}: {poll_err}. Continuing wait...{Style.RESET_ALL}")
                     time.sleep(2) # Wait longer on error
                except Exception as poll_fatal_err:
                     logger.error(f"{Fore.RED}Fatal error polling order status for {format_order_id(order_id)}: {poll_fatal_err}{Style.RESET_ALL}")
                     logger.debug(traceback.format_exc())
                     # Stop polling on unexpected errors
                     break


            if filled_order:
                filled_qty = safe_decimal_conversion(filled_order.get('filled', '0.0'), default=Decimal('0.0'))
                avg_price = safe_decimal_conversion(filled_order.get('average', '0.0'), default=Decimal('0.0'))
                fee = safe_decimal_conversion(filled_order.get('fee', {}).get('cost', '0.0'), default=Decimal('0.0'))
                fee_currency = filled_order.get('fee', {}).get('currency', CONFIG.usdt_symbol)

                # Check if filled quantity is reasonably close to requested amount (e.g., > 99%)
                if filled_qty >= amount * Decimal("0.99") and avg_price > CONFIG.position_qty_epsilon:
                    logger.success(f"{Fore.GREEN}Market order {format_order_id(order_id)} confirmed filled! Filled Qty: {filled_qty.normalize()}, Avg Price: {avg_price.normalize()}, Fee: {fee:.4f} {fee_currency}{Style.RESET_ALL}")
                    # Update the original order object with fill details for consistency
                    order.update(filled_order)
                    return order # Return the updated order dictionary
                else:
                    logger.warning(f"{Fore.YELLOW}Market order {format_order_id(order_id)} status is 'closed' but filled quantity ({filled_qty.normalize()}) is significantly less than requested ({amount.normalize()}) or avg price is zero ({avg_price.normalize()}). Potential partial fill or issue.{Style.RESET_ALL}")
                    # Return the fetched order even if partial fill, caller should verify position state
                    order.update(filled_order)
                    return order
            else:
                # Timeout occurred or fatal polling error
                logger.error(f"{Fore.RED}Market order {format_order_id(order_id)} did not confirm 'closed' status after {CONFIG.order_fill_timeout_seconds}s (Last status: {last_status_checked}). Status may be unknown. Manual check advised!{Style.RESET_ALL}")
                send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] WARNING: Market order {format_order_id(order_id)} fill confirmation timed out!")
                # Return the initial order response, status might be 'open' or 'unknown'
                # Caller MUST check position state after this.
                return order

        else: # For non-market orders (limit etc.), just return the initial response
            return order

    # --- Error Handling for Order Creation ---
    except ccxt.InsufficientFunds as e:
        logger.error(f"{Fore.RED}Order Failed ({symbol}): Insufficient funds - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Insufficient Funds.")
    except ccxt.InvalidOrder as e:
        # Provide more specific details if possible from the error message
        logger.error(f"{Fore.RED}Order Failed ({symbol}): Invalid order request - {e}. Check parameters (qty precision={format_amount(exchange, symbol, amount)}, price={formatted_price}, stops, limits, value).{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Invalid Request ({e}).")
    except ccxt.DDoSProtection as e:
        logger.warning(f"{Fore.YELLOW}Order Failed ({symbol}): Rate limit hit - {e}. Backing off.{Style.RESET_ALL}")
        # Let the main loop handle sleep/backoff based on rateLimit property
    except ccxt.RequestTimeout as e:
        logger.warning(f"{Fore.YELLOW}Order Failed ({symbol}): Request timed out - {e}. Network issue or high load. Order status unknown.{Style.RESET_ALL}")
        # Treat as failure, but state is uncertain
    except ccxt.NetworkError as e:
        logger.warning(f"{Fore.YELLOW}Order Failed ({symbol}): Network error - {e}. Check connection. Order status unknown.{Style.RESET_ALL}")
        # Treat as failure, but state is uncertain
    except ccxt.ExchangeError as e:
        logger.error(f"{Fore.RED}Order Failed ({symbol}): Exchange error - {e}. Check account status, symbol status, Bybit system status.{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Exchange Error ({e}).")
    except Exception as e:
        logger.error(f"{Fore.RED}Order Failed ({symbol}): Unexpected error during creation/confirmation - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Unexpected Error: {type(e).__name__}.")

    return None # Return None if order placement or critical confirmation failed


def confirm_stops_attached(exchange: ccxt.Exchange, symbol: str, expected_sl_price: Optional[Decimal], expected_tp_price: Optional[Decimal], expected_tsl_active: bool, attempts: int, delay: int) -> bool:
    """
    Fetches the current position multiple times to confirm native stops are attached.
    Checks for non-zero/non-None SL/TP prices and a non-zero/non-None TSL trigger price if TSL was requested.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        expected_sl_price: The SL price expected to be attached (Decimal) or None.
        expected_tp_price: The TP price expected to be attached (Decimal) or None.
        expected_tsl_active: Boolean indicating if a Trailing Stop was requested.
        attempts: Number of times to check.
        delay: Delay in seconds between checks.

    Returns:
        True if all requested stops were confirmed attached (non-zero/non-None values returned by API)
        within the attempts, False otherwise.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("confirm_stops_attached: CONFIG not loaded!")
        return False

    logger.debug(f"Confirming native stops attached to position ({symbol})...")

    # Determine which stops need confirmation based on non-None/non-zero inputs
    needs_sl_confirm = expected_sl_price is not None and expected_sl_price > CONFIG.position_qty_epsilon
    needs_tp_confirm = expected_tp_price is not None and expected_tp_price > CONFIG.position_qty_epsilon
    needs_tsl_confirm = expected_tsl_active

    # Initialize confirmation flags based on whether confirmation is needed
    sl_attached = not needs_sl_confirm
    tp_attached = not needs_tp_confirm
    tsl_attached = not needs_tsl_confirm

    if sl_attached and tp_attached and tsl_attached:
        logger.debug("No stops required confirmation based on input.")
        return True # No stops needed confirmation

    for attempt in range(1, attempts + 1):
        logger.debug(f"Confirm Stops: Attempt {attempt}/{attempts}...")
        position_state = get_current_position(exchange, symbol)

        # If position check fails (returns NaNs/None), we cannot confirm stops reliably
        if position_state['qty'].is_nan():
             logger.warning(f"{Fore.YELLOW}Confirm Stops: Failed to fetch valid position state for {symbol} on attempt {attempt}. Cannot confirm stops.{Style.RESET_ALL}")
             if attempt < attempts:
                 time.sleep(delay)
                 continue
             else:
                 return False # Failed after all attempts

        if position_state['side'] == CONFIG.pos_none:
             logger.warning(f"{Fore.YELLOW}Confirm Stops: Position for {symbol} disappeared or became zero during confirmation check (Attempt {attempt}). Stops likely triggered or order failed.{Style.RESET_ALL}")
             # If the position is gone, we can't confirm stops were attached as intended post-entry.
             return False

        # Check if fetched position details contain valid values for the requested stops
        # get_current_position returns Decimal or None for stops
        current_sl = position_state.get('stop_loss')
        current_tp = position_state.get('take_profit')
        current_tsl_price = position_state.get('trailing_stop_price') # Trigger price

        # Update confirmation status if needed and corresponding value is valid (non-None and positive)
        if needs_sl_confirm and not sl_attached:
            if current_sl is not None and current_sl > CONFIG.position_qty_epsilon:
                 sl_attached = True
                 logger.debug(f"Confirm Stops: SL ({current_sl.normalize()}) confirmed attached.")
                 # Optional: Check if current_sl matches expected_sl_price approximately?
                 # deviation = abs(current_sl - expected_sl_price) / expected_sl_price if expected_sl_price > 0 else abs(current_sl)
                 # if deviation > Decimal("0.001"): # 0.1% tolerance
                 #    logger.warning(f"SL attached ({current_sl}) differs significantly from expected ({expected_sl_price}).")
            else:
                logger.debug(f"Confirm Stops: SL not yet confirmed (Current: {current_sl})")

        if needs_tp_confirm and not tp_attached:
            if current_tp is not None and current_tp > CONFIG.position_qty_epsilon:
                 tp_attached = True
                 logger.debug(f"Confirm Stops: TP ({current_tp.normalize()}) confirmed attached.")
            else:
                logger.debug(f"Confirm Stops: TP not yet confirmed (Current: {current_tp})")

        if needs_tsl_confirm and not tsl_attached:
             # Check if the 'trailing_stop_price' field is valid (non-None and positive)
             if current_tsl_price is not None and current_tsl_price > CONFIG.position_qty_epsilon:
                  tsl_attached = True
                  logger.debug(f"Confirm Stops: TSL (Trigger Price: {current_tsl_price.normalize()}) confirmed attached.")
             else:
                 logger.debug(f"Confirm Stops: TSL not yet confirmed (Current Trigger: {current_tsl_price})")


        # Check if all required confirmations are met
        if sl_attached and tp_attached and tsl_attached:
            confirmed_stops_list = []
            if needs_sl_confirm: confirmed_stops_list.append("SL")
            if needs_tp_confirm: confirmed_stops_list.append("TP")
            if needs_tsl_confirm: confirmed_stops_list.append("TSL")
            logger.success(f"{Fore.GREEN}Confirm Stops: All requested stops ({', '.join(confirmed_stops_list)}) confirmed attached!{Style.RESET_ALL}")
            return True

        # If not all confirmed, wait and try again (unless this is the last attempt)
        if attempt < attempts:
            time.sleep(delay)

    # If loop finishes and not all stops are confirmed attached
    logger.error(f"{Fore.RED}Confirm Stops: Failed to confirm all requested stops attached after {attempts} attempts.{Style.RESET_ALL}")
    missing_stops = []
    if needs_sl_confirm and not sl_attached: missing_stops.append("SL")
    if needs_tp_confirm and not tp_attached: missing_stops.append("TP")
    if needs_tsl_confirm and not tsl_attached: missing_stops.append("TSL")

    if missing_stops:
         logger.error(f"{Fore.RED}Confirm Stops: Potentially missing stops: {', '.join(missing_stops)}{Style.RESET_ALL}")
         send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] WARNING: Failed to confirm {', '.join(missing_stops)} attached after entry.")

    return False


def close_position(exchange: ccxt.Exchange, symbol: str, current_position: Dict[str, Any]) -> bool:
    """
    Closes the current active position using a market order with reduceOnly flag.
    Includes confirmation of closure by checking position state afterwards.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        current_position: Dictionary containing details of the current position (needs 'side' and 'qty').

    Returns:
        True if the position is confirmed closed after the attempt, False otherwise.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("close_position: CONFIG not loaded!")
        return False

    pos_side = current_position.get('side')
    # Use safe conversion for quantity, default to NaN
    pos_qty = safe_decimal_conversion(current_position.get('qty'), default=Decimal('NaN'))

    if pos_side == CONFIG.pos_none or pos_qty.is_nan() or pos_qty <= CONFIG.position_qty_epsilon:
        logger.info("Close Position: No active position found or quantity is invalid/zero. Assuming already flat.")
        return True # Already flat or cannot determine state

    close_side = CONFIG.side_sell if pos_side == CONFIG.pos_long else CONFIG.side_buy
    logger.warning(f"{Fore.YELLOW}Initiating Position Closure: Closing {pos_side} position for {symbol} (Qty: {pos_qty.normalize()}) with market order ({close_side})...{Style.RESET_ALL}")

    try:
        # Bybit V5 closing uses opposite side Market order with reduceOnly.
        # positionIdx=0 is for One-Way mode. category should be handled by defaults.
        params = {'reduceOnly': True, 'positionIdx': 0}

        # Format quantity using the exact position quantity fetched
        formatted_qty = format_amount(exchange, symbol, pos_qty)
        if formatted_qty == 'NaN':
            logger.error(f"{Fore.RED}Close Position: Failed to format position quantity {pos_qty}. Cannot place close order.{Style.RESET_ALL}")
            return False

        logger.debug(f"Placing market order to close position. Symbol: {symbol}, Side: {close_side}, Quantity: {formatted_qty}, Params: {params}")

        # Use float for amount in create_order call
        close_order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=close_side,
            amount=float(pos_qty), # Use exact position quantity
            params=params
        )

        order_id = close_order.get('id')
        order_status = close_order.get('status', 'unknown')

        logger.info(f"{Fore.GREEN}Position Close Order Conjured! | ID: {format_order_id(order_id)}, Status: {order_status}{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Closing {pos_side} position. Qty: {pos_qty.normalize()}. Order ID: {format_order_id(order_id)}. Status: {order_status}.")

        # --- Wait for Close Order Fill & Position Update ---
        # Market orders should fill fast, but confirmation involves checking the position state.
        logger.debug(f"Waiting up to {CONFIG.order_fill_timeout_seconds}s for position to reflect closure after order {format_order_id(order_id)}...")
        closed_confirmed = False
        start_time = time.time()
        last_pos_check_qty = pos_qty # Store initial qty for comparison

        while time.time() - start_time < CONFIG.order_fill_timeout_seconds:
            try:
                # Re-fetch position state
                post_close_pos = get_current_position(exchange, symbol)
                post_close_qty = post_close_pos['qty'] # Already Decimal or NaN

                # Check if position is flat (side is None or qty is zero/negligible)
                if post_close_pos['side'] == CONFIG.pos_none or (not post_close_qty.is_nan() and post_close_qty <= CONFIG.position_qty_epsilon):
                     logger.success(f"{Fore.GREEN}Position close confirmed: Position is now flat for {symbol}.{Style.RESET_ALL}")
                     closed_confirmed = True
                     break # Exit wait loop

                # Optional: Check if quantity has decreased (indicates partial fill of close order)
                if not post_close_qty.is_nan() and post_close_qty < last_pos_check_qty:
                     logger.debug(f"Position quantity reduced: {last_pos_check_qty.normalize()} -> {post_close_qty.normalize()}. Close order partially filled or still processing.")
                     last_pos_check_qty = post_close_qty # Update last known quantity

                # Small delay before next check
                elapsed = time.time() - start_time
                delay = 0.5 if elapsed < 5 else (1.0 if elapsed < 10 else 2.0)
                time.sleep(min(delay, CONFIG.order_fill_timeout_seconds - elapsed))

            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as poll_err:
                 logger.warning(f"{Fore.YELLOW}Error checking position status after close order: {poll_err}. Continuing wait...{Style.RESET_ALL}")
                 time.sleep(2)
            except Exception as poll_fatal_err:
                 logger.error(f"{Fore.RED}Fatal error checking position status after close order: {poll_fatal_err}{Style.RESET_ALL}")
                 logger.debug(traceback.format_exc())
                 break # Stop checking on unexpected errors

        # --- Final Outcome ---
        if closed_confirmed:
            time.sleep(CONFIG.post_close_delay_seconds) # Brief pause after confirmation
            return True # Successfully closed
        else:
            # Timeout occurred or fatal polling error or position didn't close
            logger.error(f"{Fore.RED}Position close FAILED or timed out after {CONFIG.order_fill_timeout_seconds}s. Position may still be active! Manual check required.{Style.RESET_ALL}")
            # Re-check position one last time just in case
            final_pos_check = get_current_position(exchange, symbol)
            final_pos_qty = final_pos_check['qty'] # Decimal or NaN
            if final_pos_check['side'] != CONFIG.pos_none and not final_pos_qty.is_nan() and final_pos_qty > CONFIG.position_qty_epsilon:
                 logger.error(f"Final Check: Position still active! Side: {final_pos_check['side']}, Qty: {final_pos_qty.normalize()}")
                 send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close order timed out/failed, pos still active. Qty: {final_pos_qty.normalize()}. MANUAL CHECK!")
            elif final_pos_qty.is_nan():
                 logger.error("Final Check: Could not determine final position state.")
                 send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close order timed out/failed, final position state UNKNOWN. MANUAL CHECK!")
            else:
                 # Position seems closed now, despite timeout/earlier issues
                 logger.info("Final Check: Position appears closed despite timeout/earlier issues.")
                 # Consider returning True here, but reporting failure due to timeout is safer
                 # return True
            return False # Indicate failure


    # --- Error Handling for Close Order Creation ---
    except ccxt.InsufficientFunds as e:
        # This might happen if margin changes drastically or due to fees
        logger.error(f"{Fore.RED}Close Order Failed ({symbol}): Insufficient funds (during close?) - {e}. Check margin.{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Insufficient Funds.")
    except ccxt.InvalidOrder as e:
        # Could be due to position already closed, or incorrect params
        logger.error(f"{Fore.RED}Close Order Failed ({symbol}): Invalid order request - {e}. Position might already be closed or params issue.{Style.RESET_ALL}")
        # Check if position is actually closed now
        time.sleep(1) # Brief pause
        check_pos = get_current_position(exchange, symbol)
        check_qty = check_pos['qty']
        if check_pos['side'] == CONFIG.pos_none or (not check_qty.is_nan() and check_qty <= CONFIG.position_qty_epsilon):
             logger.info(f"Position confirmed closed after InvalidOrder error during close attempt.")
             return True # Treat as success if position is gone
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Invalid Request ({e}).")
    except ccxt.DDoSProtection as e:
        logger.warning(f"{Fore.YELLOW}Close Order Failed ({symbol}): Rate limit hit - {e}. Backing off.{Style.RESET_ALL}")
        # Let main loop handle backoff
    except ccxt.RequestTimeout as e:
        logger.warning(f"{Fore.YELLOW}Close Order Failed ({symbol}): Request timed out - {e}. Network issue. Close status unknown.{Style.RESET_ALL}")
    except ccxt.NetworkError as e:
        logger.warning(f"{Fore.YELLOW}Close Order Failed ({symbol}): Network error - {e}. Check connection. Close status unknown.{Style.RESET_ALL}")
    except ccxt.ExchangeError as e:
        # Specific Bybit errors might indicate already closed, e.g., "position size is zero"
        err_msg = str(e).lower()
        # Check for common Bybit V5 error codes/messages indicating no position
        # Example codes (verify these with Bybit docs): 110044 (Position size is zero), 110025 (Position does not exist)
        if "position size is zero" in err_msg or "position does not exist" in err_msg or "110044" in err_msg or "110025" in err_msg:
            logger.info(f"{Fore.GREEN}Close Order ExchangeError indicates position already closed: {e}. Verifying...{Style.RESET_ALL}")
            time.sleep(1)
            check_pos = get_current_position(exchange, symbol)
            check_qty = check_pos['qty']
            if check_pos['side'] == CONFIG.pos_none or (not check_qty.is_nan() and check_qty <= CONFIG.position_qty_epsilon):
                 logger.info(f"Position confirmed closed after ExchangeError during close attempt.")
                 return True # Success
            else:
                 logger.error(f"Position still exists after 'position size is zero/not exist' error? State: {check_pos}")
                 send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Exchange Error ({e}), but pos still exists?")
        else:
            logger.error(f"{Fore.RED}Close Order Failed ({symbol}): Exchange error - {e}.{Style.RESET_ALL}")
            send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Exchange Error ({e}).")
    except Exception as e:
        logger.error(f"{Fore.RED}Close Order Failed ({symbol}): Unexpected error - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Unexpected Error: {type(e).__name__}.")

    # If any exception occurred or confirmation failed, return False
    return False


def cancel_all_orders_for_symbol(exchange: ccxt.Exchange, symbol: str, reason: str) -> int:
    """
    Attempts to cancel all open orders for a specific symbol using Bybit V5 specifics.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        reason: A string indicating why cancellation is being attempted (for logging).

    Returns:
        The number of cancellation requests sent (or confirmed cancelled if possible).
        Returns 0 on failure or if method not supported.
    """
    logger.info(f"{Fore.BLUE}Order Cleanup Ritual: Initiating for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    cancel_count = 0 # Track successful cancellations reported by exchange if possible

    if not exchange.has.get('cancelAllOrders'):
         logger.error(f"{Fore.RED}Order Cleanup Error: Exchange '{exchange.id}' CCXT instance does not support cancelAllOrders method. Cannot perform cleanup.{Style.RESET_ALL}")
         return 0

    try:
        # Bybit V5 cancelAllOrders requires 'category' parameter for futures
        # Rely on default category set in exchange options
        category = exchange.options.get('defaultCategory', 'linear')
        params = {'category': category}

        logger.warning(f"{Fore.YELLOW}Order Cleanup: Attempting to cancel ALL open orders for {symbol} (Category: {category})...{Style.RESET_ALL}")

        # Use cancel_all_orders with symbol and category params
        response = exchange.cancel_all_orders(symbol=symbol, params=params)

        # Bybit V5 response format for cancelAllOrders (linear):
        # { "retCode": 0, "retMsg": "OK", "result": { "list": [ { "orderId": "...", "orderLinkId": "..." }, ... ] }, ... }
        logger.info(f"{Fore.GREEN}Order Cleanup: cancel_all_orders request sent for {symbol}. Full Response: {response}{Style.RESET_ALL}")

        # Attempt to parse response for number of cancellations
        if isinstance(response, dict) and response.get('retCode') == 0:
             result_list = response.get('result', {}).get('list', [])
             if isinstance(result_list, list):
                 cancel_count = len(result_list)
                 logger.info(f"Order Cleanup: Successfully cancelled {cancel_count} orders for {symbol}.")
             else:
                 logger.info("Order Cleanup: Request OK, but could not parse list of cancelled orders from response.")
                 cancel_count = 1 # Indicate success, but count unknown
        elif isinstance(response, list): # Handle potential direct list response from CCXT unification layer
             cancel_count = len(response)
             logger.info(f"Order Cleanup: Successfully cancelled {cancel_count} orders for {symbol} (parsed from list response).")
        else:
             logger.warning("Order Cleanup: Request sent, but response format unexpected or indicated failure (retCode != 0). Cannot confirm cancellation count.")
             cancel_count = 0 # Indicate potential failure or unknown status

        return cancel_count

    except ccxt.NotSupported as e:
        logger.error(f"{Fore.RED}Order Cleanup Error: Exchange method not supported - {e}. Cannot perform cleanup.{Style.RESET_ALL}")
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Order Cleanup Error for {symbol}: {type(e).__name__} - {e}. Could not cancel orders.{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order Cancel FAILED: {type(e).__name__}.")
    except Exception as e:
        logger.error(f"{Fore.RED}Order Cleanup Unexpected Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order Cancel FAILED: Unexpected Error: {type(e).__name__}.")

    return 0 # Return 0 on failure

# --- Strategy Signal Generation - The Oracle's Prophecy ---
def generate_trading_signal(df: pd.DataFrame, current_position: Dict[str, Any], vol_atr_data: Dict[str, Optional[Decimal]], order_book_data: Dict[str, Any]) -> Optional[str]:
    """
    Analyzes indicator data based on the selected strategy and generates a trade signal,
    applying filters for entry signals.

    Args:
        df: DataFrame with OHLCV and calculated indicator columns.
        current_position: Dictionary with current position details ('side', 'qty').
        vol_atr_data: Dictionary with volume/ATR analysis results ('volume_ratio').
        order_book_data: Dictionary with order book analysis results ('bid_ask_ratio', 'fetched_this_cycle').

    Returns:
        'buy', 'sell', 'close_long', 'close_short', or None for no signal.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("generate_trading_signal: CONFIG not loaded!")
        return None

    # --- Basic Data Validation ---
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Signal Gen: DataFrame is missing or empty. Cannot generate signal.{Style.RESET_ALL}")
        return None

    # Estimate minimum candles needed based on the *selected* strategy's max lookback
    min_candles_strategy = 0
    if CONFIG.strategy_name == "DUAL_SUPERTREND":
        min_candles_strategy = max(CONFIG.st_atr_length, CONFIG.confirm_st_atr_length) + 2 # Need prev candle too
    elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
        min_candles_strategy = max(CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + CONFIG.stochrsi_d_period, CONFIG.momentum_length) + 2
    elif CONFIG.strategy_name == "EHLERS_FISHER":
        min_candles_strategy = CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length + 2
    elif CONFIG.strategy_name == "EMA_CROSS":
        min_candles_strategy = CONFIG.ema_slow_period + 2

    # Use a safe buffer, ensure at least 2 candles exist for prev_candle access
    min_len_required = max(2, min_candles_strategy + 5)
    if len(df) < min_len_required:
         logger.warning(f"{Fore.YELLOW}Signal Gen: Insufficient data ({len(df)} candles) for {CONFIG.strategy_name} (needs ~{min_len_required}). Cannot generate signal.{Style.RESET_ALL}")
         return None

    # Access last and previous candles safely
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2] # Guaranteed to exist due to length check above

    position_side = current_position['side']
    position_qty = current_position['qty'] # Decimal or NaN
    last_price = safe_decimal_conversion(last_candle.get('close'), default=Decimal('NaN'))

    if last_price.is_nan() or last_price <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Signal Gen: Last price is invalid ({last_price}). Cannot generate signal.{Style.RESET_ALL}")
        return None

    # --- Signal Generation Logic based on Strategy ---
    entry_signal: Optional[str] = None # 'buy' or 'sell' (use CONFIG constants)
    exit_signal: Optional[str] = None # 'close_long' or 'close_short'

    # Helper to safely get values from candles (Decimal or NaN)
    def get_val(candle: pd.Series, col: str) -> Decimal:
        return safe_decimal_conversion(candle.get(col), default=Decimal('NaN'))
    # Helper to safely get boolean values (True, False, or None if NA/missing)
    def get_bool(candle: pd.Series, col: str) -> Optional[bool]:
         val = candle.get(col)
         if pd.isna(val): return None
         # Handle potential non-boolean types that evaluate truthiness (e.g., 1.0/0.0)
         try:
             return bool(val)
         except:
             return None # Cannot determine boolean value

    # --- DUAL_SUPERTREND Logic ---
    if CONFIG.strategy_name == "DUAL_SUPERTREND":
        primary_trend = get_bool(last_candle, 'trend')
        primary_flip_long = get_bool(last_candle, 'st_long')
        primary_flip_short = get_bool(last_candle, 'st_short')
        confirm_trend = get_bool(last_candle, 'confirm_trend')
        confirm_st_val = get_val(last_candle, 'confirm_supertrend')
        primary_st_val = get_val(last_candle, 'supertrend') # For exit check

        # Check if required values are valid (bools can be None, Decimals check for NaN)
        if primary_trend is None or confirm_trend is None or confirm_st_val.is_nan() or primary_st_val.is_nan():
            logger.warning(f"{Fore.YELLOW}Signal Gen (DUAL_SUPERTREND): Skipping due to missing/NA indicator values (PrimaryTrend:{primary_trend}, ConfirmTrend:{confirm_trend}, ConfirmVal:{confirm_st_val}, PrimaryVal:{primary_st_val}).{Style.RESET_ALL}")
            return None
        # Flip signals can be None if column derived, handle as False
        primary_flip_long = primary_flip_long or False
        primary_flip_short = primary_flip_short or False

        # Entry Signal: Primary ST flips AND Confirmation ST agrees AND Price confirms breach of confirmation ST
        if primary_flip_long and confirm_trend is True and last_price > confirm_st_val:
            entry_signal = CONFIG.side_buy
            logger.debug(f"DUAL_ST: Primary ST Long Flip + Confirm ST Up + Price ({last_price:.4f}) > Confirm ST ({confirm_st_val:.4f}). Signal: {entry_signal}")
        elif primary_flip_short and confirm_trend is False and last_price < confirm_st_val:
            entry_signal = CONFIG.side_sell
            logger.debug(f"DUAL_ST: Primary ST Short Flip + Confirm ST Down + Price ({last_price:.4f}) < Confirm ST ({confirm_st_val:.4f}). Signal: {entry_signal}")

        # Exit Signal: Price crosses Primary ST against the trend direction
        if position_side == CONFIG.pos_long and last_price < primary_st_val:
            exit_signal = 'close_long'
            reason = f"Price < Primary ST ({primary_st_val:.4f})"
            logger.debug(f"DUAL_ST Exit: {reason}. Signal: {exit_signal}")
        elif position_side == CONFIG.pos_short and last_price > primary_st_val:
            exit_signal = 'close_short'
            reason = f"Price > Primary ST ({primary_st_val:.4f})"
            logger.debug(f"DUAL_ST Exit: {reason}. Signal: {exit_signal}")

    # --- STOCHRSI_MOMENTUM Logic ---
    elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
        k = get_val(last_candle, 'stochrsi_k')
        d = get_val(last_candle, 'stochrsi_d')
        mom = get_val(last_candle, 'momentum')
        prev_k = get_val(prev_candle, 'stochrsi_k')
        prev_d = get_val(prev_candle, 'stochrsi_d')

        if k.is_nan() or d.is_nan() or mom.is_nan() or prev_k.is_nan() or prev_d.is_nan():
            logger.warning(f"{Fore.YELLOW}Signal Gen (STOCHRSI_MOMENTUM): Skipping due to missing/NaN indicator values.{Style.RESET_ALL}")
            return None

        # Entry Signal: Crossover below/above threshold + Momentum confirmation
        # Long Entry: K crosses above D, both below oversold, Momentum positive
        if k > d and prev_k <= prev_d and k < CONFIG.stochrsi_oversold and d < CONFIG.stochrsi_oversold and mom > CONFIG.position_qty_epsilon:
            entry_signal = CONFIG.side_buy
            logger.debug(f"STOCH_MOM: K ({k:.2f}) crossed D ({d:.2f}) below OS ({CONFIG.stochrsi_oversold}), Mom ({mom:.4f}) > 0. Signal: {entry_signal}")
        # Short Entry: K crosses below D, both above overbought, Momentum negative
        elif k < d and prev_k >= prev_d and k > CONFIG.stochrsi_overbought and d > CONFIG.stochrsi_overbought and mom < -CONFIG.position_qty_epsilon:
            entry_signal = CONFIG.side_sell
            logger.debug(f"STOCH_MOM: K ({k:.2f}) crossed D ({d:.2f}) above OB ({CONFIG.stochrsi_overbought}), Mom ({mom:.4f}) < 0. Signal: {entry_signal}")

        # Exit Signal: Stoch crossover against position in the *opposite* extreme zone
        if position_side == CONFIG.pos_long and k < d and prev_k >= prev_d and k > CONFIG.stochrsi_overbought:
            exit_signal = 'close_long'
            reason = f"K ({k:.2f}) crossed D ({d:.2f}) above OB ({CONFIG.stochrsi_overbought})"
            logger.debug(f"STOCH_MOM Exit: {reason}. Signal: {exit_signal}")
        elif position_side == CONFIG.pos_short and k > d and prev_k <= prev_d and k < CONFIG.stochrsi_oversold:
            exit_signal = 'close_short'
            reason = f"K ({k:.2f}) crossed D ({d:.2f}) below OS ({CONFIG.stochrsi_oversold})"
            logger.debug(f"STOCH_MOM Exit: {reason}. Signal: {exit_signal}")

    # --- EHLERS_FISHER Logic ---
    elif CONFIG.strategy_name == "EHLERS_FISHER":
        fisher = get_val(last_candle, 'ehlers_fisher')
        signal_line = get_val(last_candle, 'ehlers_signal')
        prev_fisher = get_val(prev_candle, 'ehlers_fisher')
        prev_signal_line = get_val(prev_candle, 'ehlers_signal')

        if fisher.is_nan() or signal_line.is_nan() or prev_fisher.is_nan() or prev_signal_line.is_nan():
            logger.warning(f"{Fore.YELLOW}Signal Gen (EHLERS_FISHER): Skipping due to missing/NaN indicator values.{Style.RESET_ALL}")
            return None

        # Entry Signal: Fisher crosses Signal line
        if fisher > signal_line and prev_fisher <= prev_signal_line:
            entry_signal = CONFIG.side_buy
            logger.debug(f"EHLERS: Fisher ({fisher:.4f}) crossed above Signal ({signal_line:.4f}). Signal: {entry_signal}")
        elif fisher < signal_line and prev_fisher >= prev_signal_line:
            entry_signal = CONFIG.side_sell
            logger.debug(f"EHLERS: Fisher ({fisher:.4f}) crossed below Signal ({signal_line:.4f}). Signal: {entry_signal}")

        # Exit Signal: Fisher crosses Signal line back against position
        if position_side == CONFIG.pos_long and fisher < signal_line and prev_fisher >= prev_signal_line:
            exit_signal = 'close_long'
            logger.debug(f"EHLERS Exit: Fisher ({fisher:.4f}) crossed below Signal ({signal_line:.4f}). Signal: {exit_signal}")
        elif position_side == CONFIG.pos_short and fisher > signal_line and prev_fisher <= prev_signal_line:
            exit_signal = 'close_short'
            logger.debug(f"EHLERS Exit: Fisher ({fisher:.4f}) crossed above Signal ({signal_line:.4f}). Signal: {exit_signal}")

    # --- EMA_CROSS Logic ---
    elif CONFIG.strategy_name == "EMA_CROSS":
        fast_ema = get_val(last_candle, 'fast_ema')
        slow_ema = get_val(last_candle, 'slow_ema')
        prev_fast_ema = get_val(prev_candle, 'fast_ema')
        prev_slow_ema = get_val(prev_candle, 'slow_ema')

        if fast_ema.is_nan() or slow_ema.is_nan() or prev_fast_ema.is_nan() or prev_slow_ema.is_nan():
            logger.warning(f"{Fore.YELLOW}Signal Gen (EMA_CROSS): Skipping due to missing/NaN indicator values.{Style.RESET_ALL}")
            return None

        # Entry Signal: Fast EMA crosses Slow EMA
        if fast_ema > slow_ema and prev_fast_ema <= prev_slow_ema:
            entry_signal = CONFIG.side_buy
            logger.debug(f"EMA_CROSS: Fast ({fast_ema:.4f}) crossed above Slow ({slow_ema:.4f}). Signal: {entry_signal}")
        elif fast_ema < slow_ema and prev_fast_ema >= prev_slow_ema:
            entry_signal = CONFIG.side_sell
            logger.debug(f"EMA_CROSS: Fast ({fast_ema:.4f}) crossed below Slow ({slow_ema:.4f}). Signal: {entry_signal}")

        # Exit Signal: Fast EMA crosses back over Slow EMA against position
        if position_side == CONFIG.pos_long and fast_ema < slow_ema and prev_fast_ema >= prev_slow_ema:
            exit_signal = 'close_long'
            logger.debug(f"EMA_CROSS Exit: Fast ({fast_ema:.4f}) crossed below Slow ({slow_ema:.4f}). Signal: {exit_signal}")
        elif position_side == CONFIG.pos_short and fast_ema > slow_ema and prev_fast_ema <= prev_slow_ema:
            exit_signal = 'close_short'
            logger.debug(f"EMA_CROSS Exit: Fast ({fast_ema:.4f}) crossed above Slow ({slow_ema:.4f}). Signal: {exit_signal}")

    else:
        # Should not be reached due to config validation
        logger.error(f"{Fore.RED}Signal Gen: Unknown strategy name '{CONFIG.strategy_name}'. No signal generated.{Style.RESET_ALL}")
        return None

    # --- Apply Exit Signal Priority ---
    if exit_signal:
        # Only return exit signal if currently in a position that matches the signal
        if (exit_signal == 'close_long' and position_side == CONFIG.pos_long) or \
           (exit_signal == 'close_short' and position_side == CONFIG.pos_short):
            logger.info(f"{Fore.YELLOW}Signal Gen: Exit signal generated ({exit_signal}). Prioritizing exit.{Style.RESET_ALL}")
            return exit_signal
        else:
            # Log if exit signal doesn't match position, but don't return it
            logger.debug(f"Signal Gen: Ignoring irrelevant exit signal ({exit_signal}) for current {position_side} position.")
            # Continue to check entry signal if applicable

    # --- Apply Entry Signal with Filters ---
    # Only consider entry if currently flat (no active position)
    if position_side == CONFIG.pos_none and entry_signal:
        logger.debug(f"Signal Gen: Potential {entry_signal} entry signal generated by strategy. Checking filters...")
        # Pass necessary filter data to check_entry_filters
        filters_passed = check_entry_filters(vol_atr_data, order_book_data, entry_signal)

        if filters_passed:
             logger.info(f"{Fore.GREEN}Signal Gen: {entry_signal.capitalize()} entry signal confirmed by filters!{Style.RESET_ALL}")
             return entry_signal # Return CONFIG.side_buy or CONFIG.side_sell
        else:
             logger.info(f"{Fore.YELLOW}Signal Gen: {entry_signal.capitalize()} entry signal generated but rejected by filters.{Style.RESET_ALL}")
             return None # Filters failed, no entry signal

    # If no exit signal and either no entry signal or not flat or filters failed
    return None

def check_entry_filters(vol_atr_data: Dict[str, Optional[Decimal]], order_book_data: Dict[str, Any], signal_side: str) -> bool:
    """
    Applies configured entry filters (Volume, Order Book) to validate a signal.

    Args:
        vol_atr_data: Dictionary containing 'volume_ratio'.
        order_book_data: Dictionary containing 'bid_ask_ratio' and 'fetched_this_cycle'.
        signal_side: The potential entry side ('buy' or 'sell').

    Returns:
        True if all required filters pass or are not enabled, False otherwise.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.error("check_entry_filters: CONFIG not loaded!")
        return False

    # --- Volume Spike Filter ---
    if CONFIG.require_volume_spike_for_entry:
        volume_ratio = vol_atr_data.get('volume_ratio') # Decimal or NaN
        if volume_ratio is None or volume_ratio.is_nan():
            logger.debug(f"Filters: Volume spike required, but volume ratio is N/A. Filter FAIL.")
            return False # Cannot check if data is missing

        volume_spike_threshold = CONFIG.volume_spike_threshold
        volume_spike_detected = volume_ratio >= volume_spike_threshold

        if not volume_spike_detected:
            logger.debug(f"Filters: Volume spike required (Ratio={volume_ratio:.2f} < Threshold={volume_spike_threshold}). Filter FAIL.")
            return False
        else:
            logger.debug(f"Filters: Volume spike required (Ratio={volume_ratio:.2f} >= Threshold={volume_spike_threshold}). Filter PASS.")
    else:
        logger.debug("Filters: Volume spike filter not required.")


    # --- Order Book Pressure Filter ---
    # Check if OB filtering is enabled by checking if thresholds are meaningful
    ob_filter_active = CONFIG.order_book_ratio_threshold_long > CONFIG.position_qty_epsilon or \
                       CONFIG.order_book_ratio_threshold_short < Decimal('Infinity') # Check if short threshold is restrictive

    if ob_filter_active:
        order_book_ratio = order_book_data.get('bid_ask_ratio') # Decimal or NaN
        # Check if OB was actually fetched this cycle (important if fetch_per_cycle is false)
        ob_fetched = order_book_data.get('fetched_this_cycle', False)

        if not ob_fetched:
             logger.warning(f"{Fore.YELLOW}Filters: Order Book filter enabled, but data was not fetched this cycle (likely due to config). Filter FAIL.{Style.RESET_ALL}")
             return False # Cannot check if not fetched when needed
        if order_book_ratio is None or order_book_ratio.is_nan():
            logger.debug(f"Filters: Order Book filter enabled, but ratio is N/A. Filter FAIL.")
            return False # Cannot check if data is missing

        # Apply thresholds
        if signal_side == CONFIG.side_buy:
            if order_book_ratio < CONFIG.order_book_ratio_threshold_long:
                logger.debug(f"Filters: Long OB ratio required (Ratio={order_book_ratio:.3f} < Threshold={CONFIG.order_book_ratio_threshold_long}). Filter FAIL.")
                return False
            else:
                logger.debug(f"Filters: Long OB ratio required (Ratio={order_book_ratio:.3f} >= Threshold={CONFIG.order_book_ratio_threshold_long}). Filter PASS.")
        elif signal_side == CONFIG.side_sell:
            # For short, we need more ask volume, meaning the ratio (Bid/Ask) should be *below* the short threshold.
            if order_book_ratio > CONFIG.order_book_ratio_threshold_short:
                logger.debug(f"Filters: Short OB ratio required (Ratio={order_book_ratio:.3f} > Threshold={CONFIG.order_book_ratio_threshold_short}). Filter FAIL.")
                return False
            else:
                logger.debug(f"Filters: Short OB ratio required (Ratio={order_book_ratio:.3f} <= Threshold={CONFIG.order_book_ratio_threshold_short}). Filter PASS.")
    else:
        logger.debug("Filters: Order Book filter not enabled (thresholds indicate disabled).")


    # If all enabled filters passed
    logger.debug("Filters: All enabled filters passed.")
    return True


# --- Main Trading Logic - The Core Spell Loop ---
def main_trade_logic(exchange: ccxt.Exchange):
    """
    The main trading loop that fetches data, calculates indicators,
    generates signals, and executes trades based on the chosen strategy.
    Includes robust error handling and state management.
    """
    global CONFIG # Access global CONFIG
    if CONFIG is None:
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: CONFIG object not available in main_trade_logic. Exiting.{Style.RESET_ALL}")
        return # Cannot run without config

    logger.info(f"{Fore.BLUE}--- Pyrmethus Bybit Scalping Spell v2.3.0 Initializing ({time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.BLUE}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.BLUE}--- Protective Wards Activated: Initial ATR-Stop, ATR-TakeProfit + Exchange Trailing Stop (Bybit V5 Native) ---{Style.RESET_ALL}")


    # --- Initial Setup: Market Info & Leverage ---
    try:
        logger.info(f"Attempting to focus spell on symbol: {CONFIG.symbol}")
        market = exchange.market(CONFIG.symbol)
        if not market: # Check if market info was loaded
             raise ccxt.BadSymbol(f"Market {CONFIG.symbol} not found in loaded markets.")

        logger.info(f"Market Details | ID: {market.get('id')}, Type: {market.get('type')}, Base: {market.get('base')}, Quote: {market.get('quote')}, Settle: {market.get('settle')}")
        logger.info(f"Market Details | Linear: {market.get('linear')}, Inverse: {market.get('inverse')}, Contract: {market.get('contract')}")
        logger.info(f"Market Details | Precision: Amount={market.get('precision', {}).get('amount')}, Price={market.get('precision', {}).get('price')}")
        logger.info(f"Market Details | Limits: Amount(Min={market.get('limits',{}).get('amount',{}).get('min')}, Max={market.get('limits',{}).get('amount',{}).get('max')}), Cost(Min={market.get('limits',{}).get('cost',{}).get('min')})")

        # Verify it's a linear contract as expected
        if not market.get('linear'):
             logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Symbol '{CONFIG.symbol}' is not identified as a linear (USDT/USDC settled) contract. This bot requires linear contracts. Exiting.{Style.RESET_ALL}")
             send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Symbol {CONFIG.symbol} is not LINEAR. Bot stopped.")
             raise SystemExit("Unsupported contract type")

        # Set leverage (attempt retry)
        leverage_set = False
        for attempt in range(CONFIG.retry_count + 1): # Allow one extra attempt message
            try:
                if attempt > 0: logger.info(f"{Fore.YELLOW}Leverage Conjuring: Retrying ({attempt}/{CONFIG.retry_count})...{Style.RESET_ALL}")
                else: logger.info(f"{Fore.YELLOW}Leverage Conjuring: Attempting to set {CONFIG.leverage}x for {CONFIG.symbol}...{Style.RESET_ALL}")

                # CCXT set_leverage for Bybit V5 requires 'category' and 'symbol' in params
                # Rely on default category from exchange options
                response = exchange.set_leverage(CONFIG.leverage, CONFIG.symbol, params={})
                # Check response if possible (Bybit might return leverage info)
                logger.debug(f"Set Leverage Response: {response}")
                # Bybit response structure example: {'info': {'leverage': '10', ...}, 'leverage': 10.0}
                confirmed_leverage = safe_decimal_conversion(response.get('leverage'), default=Decimal('NaN'))
                if not confirmed_leverage.is_nan() and confirmed_leverage == Decimal(CONFIG.leverage):
                    logger.info(f"{Fore.GREEN}Leverage Conjuring: Leverage set and confirmed to {CONFIG.leverage}x for {CONFIG.symbol}.{Style.RESET_ALL}")
                    leverage_set = True
                    break # Exit retry loop on success
                else:
                    # Check if maybe it was already set (some exchanges don't return full info on no-op)
                    # Re-fetch position to check current leverage (might be slightly delayed)
                    time.sleep(1)
                    pos_check = get_current_position(exchange, CONFIG.symbol)
                    current_lev = pos_check.get('leverage') # Decimal or NaN
                    if not current_lev.is_nan() and current_lev == Decimal(CONFIG.leverage):
                        logger.info(f"{Fore.GREEN}Leverage Conjuring: Leverage confirmed to be {CONFIG.leverage}x for {CONFIG.symbol} via position check.{Style.RESET_ALL}")
                        leverage_set = True
                        break
                    else:
                         logger.warning(f"Leverage set response ({response}) or position check ({current_lev}) did not confirm target leverage ({CONFIG.leverage}).")
                         # Continue retrying

            except ccxt.NotSupported as e:
                 logger.error(f"{Fore.RED}Leverage Conjuring Failed: Exchange or symbol does not support setting leverage via this method: {e}{Style.RESET_ALL}")
                 leverage_set = False
                 break # No point retrying if not supported
            except ccxt.ExchangeError as e:
                # Check for specific "leverage not modified" messages or codes
                err_str = str(e).lower()
                # Bybit V5 error code for no modification: 110026
                if "leverage not modified" in err_str or "same leverage" in err_str or "110026" in err_str:
                    logger.info(f"{Fore.GREEN}Leverage Conjuring: Leverage already set to {CONFIG.leverage}x for {CONFIG.symbol} (or not modified).{Style.RESET_ALL}")
                    leverage_set = True
                    break # Already set, no need to retry
                else:
                    logger.warning(f"{Fore.YELLOW}Leverage Conjuring Failed (Attempt {attempt + 1}): Exchange error - {e}.{Style.RESET_ALL}")
                    if attempt < CONFIG.retry_count: time.sleep(CONFIG.retry_delay_seconds)
            except Exception as e:
                logger.warning(f"{Fore.YELLOW}Leverage Conjuring Failed (Attempt {attempt + 1}): Unexpected error - {e}.{Style.RESET_ALL}")
                if attempt < CONFIG.retry_count: time.sleep(CONFIG.retry_delay_seconds)

        if not leverage_set:
             logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to set or confirm leverage after {CONFIG.retry_count} attempts. Cannot continue.{Style.RESET_ALL}")
             send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: Failed to set leverage for {CONFIG.symbol}. Bot stopped.")
             raise SystemExit("Failed to set leverage")


        logger.success(f"{Fore.GREEN}Spell successfully focused on Symbol: {CONFIG.symbol}{Style.RESET_ALL}")

        # --- Log Configuration Summary ---
        logger.info(f"{Fore.BLUE}--- Spell Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Symbol: {CONFIG.symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Strategy Path: {CONFIG.strategy_name}{Style.RESET_ALL}")
        # Log strategy specific parameters
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
             logger.info(f"{Fore.BLUE}  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}{Style.RESET_ALL}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
             logger.info(f"{Fore.BLUE}  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} ({CONFIG.stochrsi_oversold}-{CONFIG.stochrsi_overbought}), Momentum={CONFIG.momentum_length}{Style.RESET_ALL}")
        elif CONFIG.strategy_name == "EHLERS_FISHER":
             logger.info(f"{Fore.BLUE}  Params: Fisher={CONFIG.ehlers_fisher_length}/{CONFIG.ehlers_fisher_signal_length}{Style.RESET_ALL}")
        elif CONFIG.strategy_name == "EMA_CROSS":
             logger.info(f"{Fore.BLUE}  Params: EMA Fast={CONFIG.ema_fast_period}, Slow={CONFIG.ema_slow_period}{Style.RESET_ALL}")

        logger.info(f"{Fore.BLUE}Risk Ward: {CONFIG.risk_per_trade_percentage:.3%} equity/trade, Max Pos Value: {CONFIG.max_order_usdt_amount.normalize()} {CONFIG.usdt_symbol}{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Initial SL Ward: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period}){Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Initial TP Enchantment: {CONFIG.atr_take_profit_multiplier} * ATR({CONFIG.atr_calculation_period}){Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Trailing SL Shield: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Volume Filter: EntryRequiresSpike={CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, SpikeThr={CONFIG.volume_spike_threshold}x){Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Order Book Filter: FetchPerCycle={CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short}){Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Timing: Sleep={CONFIG.sleep_seconds}s | API: RecvWin={CONFIG.default_recv_window}ms, FillTimeout={CONFIG.order_fill_timeout_seconds}s, StopConfirmAttempts={CONFIG.stop_attach_confirm_attempts}, StopConfirmDelay={CONFIG.stop_attach_confirm_delay_seconds}s{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Other: Margin Buffer={CONFIG.required_margin_buffer - 1:.1%}, SMS Alerts={CONFIG.enable_sms_alerts}{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}------------------------------{Style.RESET_ALL}")

        # Send initial configuration SMS alert
        sms_config_summary = f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Spell Initialized. Lvg:{CONFIG.leverage}x. Strat:{CONFIG.strategy_name}. Risk:{CONFIG.risk_per_trade_percentage:.2%}. SL/TP/TSL Active."
        send_sms_alert(sms_config_summary)


    except (ccxt.BadSymbol, ccxt.ExchangeError, SystemExit) as setup_err:
        # Catch specific errors during setup that should halt execution
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Setup failed: {setup_err}{Style.RESET_ALL}")
        # Attempt graceful shutdown (might not do much if exchange isn't fully working)
        graceful_shutdown(exchange) # Pass potentially partially initialized exchange
        sys.exit(1) # Ensure exit
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Unexpected error during setup: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL SETUP FAILED for {CONFIG.symbol}. Error: {type(e).__name__}.")
        graceful_shutdown(exchange)
        sys.exit(1)


    # --- Determine required candle history ---
    # Calculate maximum lookback needed by *any* potentially used indicator (including filters/ATR)
    max_lookback_periods = [
        CONFIG.st_atr_length, CONFIG.confirm_st_atr_length, # DUAL_SUPERTREND
        CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length, # STOCHRSI_MOMENTUM
        CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length, # EHLERS_FISHER
        CONFIG.ema_fast_period, CONFIG.ema_slow_period, # EMA_CROSS
        CONFIG.atr_calculation_period, # For SL/TP
        CONFIG.volume_ma_period # For Volume Filter
    ]
    # Filter out zero or negative periods if any defaults were unusual
    max_indicator_length = max(p for p in max_lookback_periods if p > 0) if max_lookback_periods else 1

    # Total candles needed = max lookback + buffer + safety margin + 1 (for prev candle access)
    # pandas_ta often needs more than just the period length for stable calculation.
    # Use a generous buffer (e.g., 50-100 candles) plus the API fetch buffer
    candles_needed = max_indicator_length + 100 + CONFIG.api_fetch_limit_buffer # Increased buffer
    logger.info(f"Estimating {candles_needed} candles needed (Max Lookback: {max_indicator_length} + Buffers).")


    # --- Main Cycle Loop ---
    running = True
    while running:
        try:
            # --- 1. Fetch Market Data ---
            df = get_market_data(exchange, CONFIG.symbol, CONFIG.interval, candles_needed)
            if df is None or df.empty:
                logger.warning(f"{Fore.YELLOW}Cycle Skip: Failed to get market data. Waiting {CONFIG.sleep_seconds}s...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds)
                continue
            # Check if sufficient length AFTER fetch and cleaning
            min_len_for_calc = max(2, max_indicator_length + 50) # Use the increased buffer here too
            if len(df) < min_len_for_calc:
                logger.warning(f"{Fore.YELLOW}Cycle Skip: Insufficient market data length ({len(df)} < {min_len_for_calc}) after fetch/cleaning. Waiting {CONFIG.sleep_seconds}s...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds)
                continue

            latest_candle_time = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"\n========== New Weaving Cycle ({CONFIG.strategy_name}): {CONFIG.symbol} | Candle: {latest_candle_time} =========={Style.RESET_ALL}")


            # --- 2. Calculate Base Indicators (ATR, Volume) ---
            # These are often needed regardless of strategy (SL/TP, filters)
            vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
            current_atr: Optional[Decimal] = vol_atr_data.get('atr') # Decimal or NaN

            # Check if latest ATR is valid (needed for SL/TP calculation)
            if current_atr is None or current_atr.is_nan() or current_atr <= CONFIG.position_qty_epsilon:
                 logger.warning(f"{Fore.YELLOW}Cycle Skip: Calculated ATR ({current_atr}) is invalid or zero. Cannot calculate dynamic SL/TP. Waiting {CONFIG.sleep_seconds}s...{Style.RESET_ALL}")
                 time.sleep(CONFIG.sleep_seconds)
                 continue


            # --- 3. Fetch Order Book (Conditional) ---
            order_book_data: Dict[str, Any] = {"bid_ask_ratio": Decimal('NaN'), "spread": Decimal('NaN'), "best_bid": Decimal('NaN'), "best_ask": Decimal('NaN'), "fetched_this_cycle": False}
            # Determine if OB filter is actually enabled based on thresholds
            ob_filter_active = CONFIG.order_book_ratio_threshold_long > CONFIG.position_qty_epsilon or \
                               CONFIG.order_book_ratio_threshold_short < Decimal('Infinity')
            # Fetch if configured per cycle OR if filter is active AND we are flat (potential entry)
            # Get position state early to decide on fetching OB
            current_position = get_current_position(exchange, CONFIG.symbol)
            position_side = current_position['side']

            should_fetch_ob = CONFIG.fetch_order_book_per_cycle or (ob_filter_active and position_side == CONFIG.pos_none)
            if should_fetch_ob:
                 order_book_data = analyze_order_book(exchange, CONFIG.symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            ob_fetched = order_book_data.get('fetched_this_cycle', False)


            # --- 4. Calculate Strategy-Specific Indicators ---
            # Pass the DataFrame, let functions modify it
            if CONFIG.strategy_name == "DUAL_SUPERTREND":
                df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
                df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
            elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
                 df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
            elif CONFIG.strategy_name == "EHLERS_FISHER":
                 df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
            elif CONFIG.strategy_name == "EMA_CROSS":
                 df = calculate_ema_cross(df, CONFIG.ema_fast_period, CONFIG.ema_slow_period)
            # Add other strategies here...


            # --- 5. State Logging (Position already fetched) ---
            position_qty = current_position['qty'] # Decimal or NaN
            position_entry_price = current_position['entry_price'] # Decimal or NaN

            # Get the *most recent* closing price
            last_price = safe_decimal_conversion(df['close'].iloc[-1], default=Decimal('NaN'))
            if last_price.is_nan():
                 logger.error(f"{Fore.RED}Cycle Skip: Last closing price is NaN. Cannot proceed.{Style.RESET_ALL}")
                 time.sleep(CONFIG.sleep_seconds)
                 continue

            # Log current state for context
            logger.info(f"State | Price: {last_price:.4f}, ATR({CONFIG.atr_calculation_period}): {current_atr:.5f}")

            # Volume Filter state logging
            vol_ratio = vol_atr_data.get('volume_ratio')
            vol_filter_state = f"Ratio={vol_ratio:.2f}" if vol_ratio and not vol_ratio.is_nan() else "Ratio=N/A"
            vol_spike_check = "N/A"
            if vol_ratio and not vol_ratio.is_nan():
                vol_spike_check = "YES" if vol_ratio >= CONFIG.volume_spike_threshold else "NO"
            logger.info(f"State | Volume: {vol_filter_state}, Spike={vol_spike_check} (Threshold={CONFIG.volume_spike_threshold}, Required={CONFIG.require_volume_spike_for_entry})")

            # Order Book Filter state logging
            ob_ratio = order_book_data.get('bid_ask_ratio')
            ob_spread = order_book_data.get('spread')
            ob_filter_state = f"Ratio(B/A)={ob_ratio:.3f}" if ob_ratio and not ob_ratio.is_nan() else "Ratio=N/A"
            ob_spread_state = f"Spread={ob_spread:.4f}" if ob_spread and not ob_spread.is_nan() else "Spread=N/A"
            logger.info(f"State | OrderBook: {ob_filter_state} (L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short}), {ob_spread_state} (Fetched={ob_fetched})")

            # Position state logging
            pos_log_color = Fore.BLUE # Default for flat
            if position_side == CONFIG.pos_long: pos_log_color = Fore.GREEN
            elif position_side == CONFIG.pos_short: pos_log_color = Fore.RED

            pos_details = f"Side={position_side}, Qty={position_qty.normalize() if not position_qty.is_nan() else 'NaN'}, Entry={position_entry_price.normalize() if not position_entry_price.is_nan() else 'NaN'}"
            if position_side != CONFIG.pos_none:
                 pos_stops_details = []
                 sl = current_position.get('stop_loss') # Decimal or None
                 tp = current_position.get('take_profit') # Decimal or None
                 tsl = current_position.get('trailing_stop_price') # Decimal or None
                 if sl is not None: pos_stops_details.append(f"SL={sl.normalize():.4f}")
                 if tp is not None: pos_stops_details.append(f"TP={tp.normalize():.4f}")
                 if tsl is not None: pos_stops_details.append(f"TSL(Trig)={tsl.normalize():.4f}")

                 if pos_stops_details: pos_details += f" | Stops: {' | '.join(pos_stops_details)}"
                 else: pos_details += " | Stops: None Active"

                 pos_pnl = current_position.get('unrealized_pnl') # Decimal or NaN
                 if pos_pnl is not None and not pos_pnl.is_nan():
                     pnl_color = Fore.GREEN if pos_pnl > 0 else (Fore.RED if pos_pnl < 0 else Fore.WHITE)
                     # Apply color only to PNL value, then reset, then apply position color
                     pos_details += f" | UPNL: {pnl_color}{pos_pnl:.4f}{Style.RESET_ALL}{pos_log_color}"

                 pos_liq_price = current_position.get('liquidation_price') # Decimal or NaN
                 if pos_liq_price is not None and not pos_liq_price.is_nan() and pos_liq_price > CONFIG.position_qty_epsilon:
                     pos_details += f" | Liq: {pos_liq_price:.4f}"

            logger.info(f"{pos_log_color}State | Position: {pos_details}{Style.RESET_ALL}")


            # --- 6. Generate Trading Signal ---
            # Pass necessary data for signal generation and filtering
            signal = generate_trading_signal(df, current_position, vol_atr_data, order_book_data)


            # --- 7. Execute Trade based on Signal ---
            if signal == CONFIG.side_buy and position_side == CONFIG.pos_none:
                logger.info(f"{Fore.GREEN}Entry Signal: LONG! Preparing order...{Style.RESET_ALL}")

                # Calculate SL/TP prices based on current price and ATR
                sl_distance = current_atr * CONFIG.atr_stop_loss_multiplier
                tp_distance = current_atr * CONFIG.atr_take_profit_multiplier
                stop_loss_price = last_price - sl_distance
                take_profit_price = last_price + tp_distance

                # Validate calculated SL/TP (ensure SL below entry, TP above entry)
                if stop_loss_price <= CONFIG.position_qty_epsilon or stop_loss_price >= last_price:
                     logger.warning(f"{Fore.YELLOW}Calculated Long SL price ({stop_loss_price:.4f}) is invalid relative to current price ({last_price:.4f}). Skipping order.{Style.RESET_ALL}")
                     send_sms_alert(f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Long SL calc invalid ({stop_loss_price:.2f}). Skipping order.")
                     time.sleep(CONFIG.sleep_seconds) # Prevent immediate retry on calc error
                     continue
                if take_profit_price <= CONFIG.position_qty_epsilon or take_profit_price <= last_price:
                     logger.warning(f"{Fore.YELLOW}Calculated Long TP price ({take_profit_price:.4f}) is invalid relative to current price ({last_price:.4f}). Skipping order.{Style.RESET_ALL}")
                     send_sms_alert(f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Long TP calc invalid ({take_profit_price:.2f}). Skipping order.")
                     time.sleep(CONFIG.sleep_seconds)
                     continue

                # Calculate Order Quantity
                # Fetch fresh equity balance before calculation
                account_equity = Decimal('NaN')
                try:
                    balance_params = {'accountType': 'CONTRACT'} # Or 'UNIFIED'
                    balance_info = exchange.fetch_balance(params=balance_params)
                    account_equity = safe_decimal_conversion(balance_info.get(CONFIG.usdt_symbol, {}).get('equity'), default=Decimal('NaN')) # Use equity for risk calc
                    if account_equity.is_nan():
                         logger.error(f"{Fore.RED}Failed to fetch valid account equity. Cannot calculate quantity.{Style.RESET_ALL}")
                         time.sleep(CONFIG.sleep_seconds)
                         continue
                except Exception as bal_err:
                    logger.error(f"{Fore.RED}Error fetching balance for quantity calculation: {bal_err}{Style.RESET_ALL}")
                    time.sleep(CONFIG.sleep_seconds)
                    continue

                order_quantity = calculate_order_quantity(
                    exchange, CONFIG.symbol, account_equity,
                    last_price, stop_loss_price, CONFIG.side_buy, market
                )

                if order_quantity is not None and order_quantity > CONFIG.position_qty_epsilon:
                    # Place the market order with native SL/TP/TSL
                    # Pass last_price as the reference for TSL activation calculation if needed
                    created_order = create_order(
                        exchange, CONFIG.symbol, 'market', CONFIG.side_buy, order_quantity,
                        price=last_price, # Provide price for TSL activation calc
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        trailing_stop_percentage=CONFIG.trailing_stop_percentage
                    )

                    if created_order:
                        logger.success(f"{Fore.GREEN}Long entry order placed (ID: {format_order_id(created_order.get('id'))}). Confirming position and stops...{Style.RESET_ALL}")
                        # Wait briefly before checking state
                        time.sleep(CONFIG.stop_attach_confirm_delay_seconds)
                        # Verify position and stops are active
                        pos_after_entry = get_current_position(exchange, CONFIG.symbol)
                        qty_after_entry = pos_after_entry['qty'] # Decimal or NaN

                        if pos_after_entry['side'] == CONFIG.pos_long and \
                           not qty_after_entry.is_nan() and \
                           qty_after_entry >= order_quantity * Decimal("0.99"): # Allow slight difference due to fees/precision
                             logger.success(f"{Fore.GREEN}Position confirmed active after long entry (Qty: {qty_after_entry.normalize()}).{Style.RESET_ALL}")
