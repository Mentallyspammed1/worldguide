#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.3.0 (Fortified + TP + Stop Confirmation + pandas_ta Fix)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.3.0 (Unified: Selectable Strategies + Precision + Native SL/TP/TSL + Fortified Config + Pyrmethus Enhancements + Robustness + pandas_ta Fix).

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
- Enhanced Precision: Leverages Python's `Decimal` type for critical financial calculations, minimizing floating-point inaccuracies.
- Fortified Configuration: Robust loading of settings from environment variables (.env file) with strict type casting and validation for improved reliability.
- Native Stop-Loss, Take-Profit, & Trailing Stop-Loss: Utilizes Bybit V5 API's exchange-native Stop Loss (fixed, ATR-based), Take Profit (fixed, percentage-based), and Trailing Stop Loss capabilities, placed immediately upon position entry for faster reaction times. Includes post-entry verification that stops are attached.
- Volatility Adaptation: Employs the Average True Range (ATR) indicator to measure market volatility and dynamically adjust the initial Stop Loss and Take Profit distances.
- Optional Confirmation Filters: Includes optional filters based on Volume Spikes (relative to a moving average) and Order Book Pressure (Bid/Ask volume ratio) to potentially improve entry signal quality.
- Sophisticated Risk Management: Implements risk-based position sizing (percentage of equity per trade), incorporates exchange margin requirements checks with a configurable buffer, and allows setting a maximum position value cap (USDT).
- Termux Integration: Provides optional SMS alerts via Termux:API for critical events like initialization, errors, order placements, and shutdowns. Includes checks for command availability.
- Robust Operation: Features comprehensive error handling for common CCXT exceptions (network issues, authentication failures, rate limits, exchange errors), data validation (NaN handling), and detailed logging with vibrant console colors via Colorama. Includes robust `pandas_ta` column identification to prevent indicator calculation errors.
- Graceful Shutdown: Designed to handle interruptions (Ctrl+C) or critical errors by attempting to cancel open orders and close any existing positions before exiting.
- Bybit V5 API Focused: Tailored logic for interacting with the Bybit V5 API, particularly regarding position detection (One-Way Mode), order parameters, and native stop placement.

Disclaimer:
- **EXTREME RISK**: Trading cryptocurrencies, especially futures contracts with leverage and automated systems, involves substantial risk of financial loss. This script is provided for EDUCATIONAL PURPOSES ONLY. You could lose your entire investment and potentially more. Use this software entirely at your own risk. The authors and contributors assume NO responsibility for any trading losses.
- **NATIVE SL/TP/TSL RELIANCE**: The bot's protective stop mechanisms rely entirely on Bybit's exchange-native order execution. Their performance is subject to exchange conditions, potential slippage during volatile periods, API reliability, order book liquidity, and specific exchange rules. These orders are NOT GUARANTEED to execute at the precise trigger price specified.
- **PARAMETER SENSITIVITY & OPTIMIZATION**: The performance of this bot is highly dependent on the chosen strategy parameters (indicator settings, risk levels, SL/TP/TSL percentages, filter thresholds). These parameters require extensive backtesting, optimization, and forward testing on a TESTNET environment before considering any live deployment. Default parameters are unlikely to be profitable.
- **API RATE LIMITS & BANS**: Excessive API requests can lead to temporary or permanent bans from the exchange. Monitor API usage and adjust script timing (`SLEEP_SECONDS`) accordingly. CCXT's built-in rate limiter is enabled but may not prevent all issues under heavy load.
- **SLIPPAGE**: Market orders, used for entry and potentially for SL/TP/TSL execution by the exchange, are susceptible to slippage. This means the actual execution price may differ from the price observed when the order was placed, especially during high volatility or low liquidity.
- **TEST THOROUGHLY**: **DO NOT RUN THIS SCRIPT WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTING ON A TESTNET OR DEMO ACCOUNT.** Ensure you fully understand every part of the code, its logic, and its potential risks before any live deployment.
- **TERMUX DEPENDENCY**: SMS alert functionality requires a Termux environment on an Android device with the Termux:API package installed (`pkg install termux-api`). Ensure it is correctly installed and configured if you enable SMS alerts.
- **API & LIBRARY UPDATES**: This script targets the Bybit V5 API via the CCXT library. Future updates to the exchange API or the CCXT library may introduce breaking changes that require code modifications. Keep CCXT updated (`pip install -U ccxt`).
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import shlex  # For safe command argument parsing
import shutil  # For checking command existence (e.g., termux-sms-send)
import subprocess
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Any

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import] # pandas_ta might lack complete type hints
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = e.name
    # Use Colorama's raw codes here as it might not be initialized yet
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)  # Activate Colorama's magic for vibrant logs
load_dotenv()  # Load secrets from the hidden .env scroll (if present)
getcontext().prec = 18  # Set Decimal precision for financial exactitude (adjust if needed, 18 is often sufficient)


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads, validates, and stores configuration parameters from environment variables.
    Provides robust type casting and default value handling.
    """

    def __init__(self) -> None:
        """Initializes configuration by loading and validating environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED, secret=True)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED, secret=True)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env(
            "SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW
        )  # Target market (CCXT unified format, e.g., 'BTC/USDT:USDT')
        self.interval: str = self._get_env(
            "INTERVAL", "1m", color=Fore.YELLOW
        )  # Chart timeframe (e.g., '1m', '5m', '1h')
        self.leverage: int = self._get_env(
            "LEVERAGE", 10, cast_type=int, color=Fore.YELLOW
        )  # Desired leverage multiplier
        self.sleep_seconds: int = self._get_env(
            "SLEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW
        )  # Pause between trading cycles (seconds)

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        # NOTE: Renamed EHLERS_MA_CROSS to EMA_CROSS for clarity as it uses standard EMAs.
        self.valid_strategies: list[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EMA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}"
            )
            raise ValueError(
                f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}"
            )
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.005 = 0.5% of equity per trade
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN
        )  # Multiplier for ATR to set initial fixed SL distance
        self.atr_take_profit_multiplier: Decimal = self._get_env(
            "ATR_TAKE_PROFIT_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN
        )  # Multiplier for ATR to set initial fixed TP distance
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN
        )  # Maximum position value in USDT (overrides risk calc if needed)
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 1.05 = Require 5% more free margin than estimated for order placement

        # --- Native Stop-Loss & Trailing Stop-Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        # Note: Native SL/TP for MARKET entry orders on Bybit V5 are submitted as fixed PRICES.
        # TSL is submitted as a PERCENTAGE in the parameters.
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.005 = 0.5% trailing distance from high/low water mark
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN
        )  # e.g., 0.001 = 0.1% price movement in profit before TSL becomes active

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN
        )  # Primary Supertrend ATR period
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN
        )  # Primary Supertrend ATR multiplier
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )  # Confirmation Supertrend ATR period
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN
        )  # Confirmation Supertrend ATR multiplier
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env(
            "STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )  # StochRSI: RSI period
        self.stochrsi_stoch_length: int = self._get_env(
            "STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )  # StochRSI: Stochastic period
        self.stochrsi_k_period: int = self._get_env(
            "STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )  # StochRSI: %K smoothing period
        self.stochrsi_d_period: int = self._get_env(
            "STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )  # StochRSI: %D smoothing period (signal line)
        self.stochrsi_overbought: Decimal = self._get_env(
            "STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN
        )  # StochRSI overbought threshold
        self.stochrsi_oversold: Decimal = self._get_env(
            "STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN
        )  # StochRSI oversold threshold
        self.momentum_length: int = self._get_env(
            "MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )  # Momentum indicator period
        # Ehlers Fisher Transform
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN
        )  # Fisher Transform calculation period
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN
        )  # Fisher Transform signal line period (1 usually means no separate signal line smoothing)
        # EMA Cross (Placeholder for Ehlers Super Smoother - Renamed)
        self.ema_fast_period: int = self._get_env(
            "EMA_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN
        )  # Fast EMA period
        self.ema_slow_period: int = self._get_env(
            "EMA_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN
        )  # Slow EMA period

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        # Volume Analysis
        self.volume_ma_period: int = self._get_env(
            "VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW
        )  # Moving average period for volume
        self.volume_spike_threshold: Decimal = self._get_env(
            "VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW
        )  # Multiplier over MA to consider a 'spike' (e.g., 1.5 = 150% of MA)
        self.require_volume_spike_for_entry: bool = self._get_env(
            "REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW
        )  # Require volume spike for entry signal?
        # Order Book Analysis
        self.order_book_depth: int = self._get_env(
            "ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW
        )  # Number of bid/ask levels to analyze for ratio
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW
        )  # Min Bid/Ask volume ratio for long confirmation
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW
        )  # Max Bid/Ask volume ratio for short confirmation (ratio = Total Bid Vol / Total Ask Vol within depth)
        self.fetch_order_book_per_cycle: bool = self._get_env(
            "FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW
        )  # Fetch OB every cycle (more API calls) or only when needed for entry confirmation?

        # --- ATR Calculation (for Initial SL/TP) ---
        self.atr_calculation_period: int = self._get_env(
            "ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN
        )  # Period for ATR calculation used in SL/TP

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env(
            "ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA
        )  # Enable/disable SMS alerts globally
        self.sms_recipient_number: str | None = self._get_env(
            "SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA, required=False
        )  # Recipient phone number for alerts (optional)
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA
        )  # Max time to wait for SMS command execution (seconds)

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = self._get_env(
            "CCXT_RECV_WINDOW", 10000, cast_type=int, color=Fore.WHITE
        )  # Milliseconds for API request validity (Bybit default 5000, increased for potential latency)
        self.order_book_fetch_limit: int = max(
            25, self.order_book_depth
        )  # How many levels to fetch (ensure >= depth needed, common limits are 25, 50, 100, 200)
        self.shallow_ob_fetch_depth: int = 5  # Depth for quick price estimates (used in order placement estimate)
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW
        )  # Max time to wait for market order fill confirmation (seconds)
        self.stop_attach_confirm_attempts: int = self._get_env(
            "STOP_ATTACH_CONFIRM_ATTEMPTS", 3, cast_type=int, color=Fore.YELLOW
        )  # Attempts to confirm native stops are attached to position after entry
        self.stop_attach_confirm_delay_seconds: int = self._get_env(
            "STOP_ATTACH_CONFIRM_DELAY_SECONDS", 1, cast_type=int, color=Fore.YELLOW
        )  # Delay between attempts to confirm stops

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy: str = "buy"  # CCXT standard side for buying
        self.side_sell: str = "sell"  # CCXT standard side for selling
        self.pos_long: str = "Long"  # Internal representation for a long position
        self.pos_short: str = "Short"  # Internal representation for a short position
        self.pos_none: str = "None"  # Internal representation for no position (flat)
        self.usdt_symbol: str = "USDT"  # The stablecoin quote currency symbol used by Bybit
        self.retry_count: int = 3  # Default attempts for certain retryable API calls (e.g., setting leverage)
        self.retry_delay_seconds: int = 2  # Default pause between retries (seconds)
        self.api_fetch_limit_buffer: int = (
            10  # Extra candles to fetch beyond strict indicator needs, providing a safety margin
        )
        self.position_qty_epsilon: Decimal = Decimal(
            "1e-9"
        )  # Small value for float/decimal comparisons involving position size to handle precision issues
        self.post_close_delay_seconds: int = 3  # Brief pause after successfully closing a position (seconds) to allow exchange state to potentially settle
        self.min_order_value_usdt: Decimal = Decimal(
            "1.0"
        )  # Minimum order value in USDT (Bybit default is often 1 USDT for perpetuals)

        # --- Post-Initialization Validation ---
        self._validate_parameters()

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    def _validate_parameters(self) -> None:
        """Performs basic validation checks on loaded parameters."""
        if self.leverage <= 0:
            raise ValueError("LEVERAGE must be a positive integer.")
        if self.risk_per_trade_percentage <= 0 or self.risk_per_trade_percentage >= 1:
            raise ValueError("RISK_PER_TRADE_PERCENTAGE must be between 0 and 1 (exclusive).")
        if self.atr_stop_loss_multiplier <= 0:
            raise ValueError("ATR_STOP_LOSS_MULTIPLIER must be positive.")
        if self.atr_take_profit_multiplier <= 0:
            raise ValueError("ATR_TAKE_PROFIT_MULTIPLIER must be positive.")
        if self.trailing_stop_percentage < 0 or self.trailing_stop_percentage >= 1:  # TSL can be 0 to disable
            raise ValueError("TRAILING_STOP_PERCENTAGE must be between 0 and 1 (inclusive of 0).")
        if self.trailing_stop_activation_offset_percent < 0:
            raise ValueError("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT cannot be negative.")
        if self.max_order_usdt_amount < 0:
            raise ValueError("MAX_ORDER_USDT_AMOUNT cannot be negative.")
        if self.required_margin_buffer < 1:
            raise ValueError("REQUIRED_MARGIN_BUFFER must be >= 1.")
        if self.enable_sms_alerts and not self.sms_recipient_number:
            logger.warning(
                f"{Fore.YELLOW}SMS alerts enabled (ENABLE_SMS_ALERTS=true) but SMS_RECIPIENT_NUMBER is not set. Alerts will not be sent.{Style.RESET_ALL}"
            )
        if self.stop_attach_confirm_attempts < 1:
            raise ValueError("STOP_ATTACH_CONFIRM_ATTEMPTS must be at least 1.")
        if self.stop_attach_confirm_delay_seconds < 0:
            raise ValueError("STOP_ATTACH_CONFIRM_DELAY_SECONDS cannot be negative.")
        # Add more validation as needed for strategy parameters, etc.

    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        color: str = Fore.WHITE,
        secret: bool = False,
    ) -> Any:
        """Fetches an environment variable, performs robust type casting (including defaults),
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

        def log_value(v):
            return "*******" if secret and v is not None else v

        if value_str is None:
            if required and default is None:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' not found and no default specified.{Style.RESET_ALL}"
                )
                raise ValueError(f"Required environment variable '{key}' not set and no default provided.")
            elif required and default is not None:
                logger.debug(
                    f"{color}Required rune {key}: Not Set. Using Required Default: '{log_value(default)}'{Style.RESET_ALL}"
                )
                value_to_cast = default
                source = "Required Default"
            elif not required:
                logger.debug(f"{color}Summoning {key}: Not Set. Using Default: '{log_value(default)}'{Style.RESET_ALL}")
                value_to_cast = default
                source = "Default"
        else:
            logger.debug(f"{color}Summoning {key}: Found Env Value: '{log_value(value_str)}'{Style.RESET_ALL}")
            value_to_cast = value_str

        # --- Attempt Casting (applies to both env var value and default value) ---
        if value_to_cast is None:
            if required:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required configuration rune '{key}' resolved to None unexpectedly during casting preparation.{Style.RESET_ALL}"
                )
                raise ValueError(f"Required environment variable '{key}' resolved to None during casting preparation.")
            else:
                logger.debug(f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}")
                return None

        final_value: Any = None
        try:
            raw_value_str = str(
                value_to_cast
            ).strip()  # Ensure string representation and remove leading/trailing whitespace
            if cast_type == bool:
                final_value = raw_value_str.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                if raw_value_str == "":
                    raise InvalidOperation("Empty string cannot be converted to Decimal.")
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                if raw_value_str == "":
                    raise ValueError("Empty string cannot be converted to int.")
                # Cast via Decimal first to handle potential float strings like "10.0" -> 10 gracefully
                final_value = int(Decimal(raw_value_str))
            elif cast_type == float:
                if raw_value_str == "":
                    raise ValueError("Empty string cannot be converted to float.")
                final_value = float(raw_value_str)
            elif cast_type == str:
                final_value = raw_value_str  # Keep as string
            else:
                logger.warning(f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw value.")
                final_value = value_to_cast  # Return original value if type is unknown

        except (ValueError, TypeError, InvalidOperation) as e:
            # Casting failed! Log error and attempt to use default, casting it carefully.
            logger.error(
                f"{Fore.RED}Invalid type/value for {key}: '{log_value(value_to_cast)}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Attempting to use default '{log_value(default)}'.{Style.RESET_ALL}"
            )
            if default is None:
                if required:
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast value for required key '{key}' and default is None.{Style.RESET_ALL}"
                    )
                    raise ValueError(f"Required env var '{key}' failed casting and has no valid default.")
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Casting failed for {key}, default is None. Final value: None{Style.RESET_ALL}"
                    )
                    return None
            else:
                # Try casting the default value itself
                source = "Default (Fallback)"
                logger.debug(
                    f"Attempting to cast fallback default value '{log_value(default)}' for key '{key}' to {cast_type.__name__}"
                )
                try:
                    default_str = str(default).strip()
                    if cast_type == bool:
                        final_value = default_str.lower() in ["true", "1", "yes", "y", "on"]
                    elif cast_type == Decimal:
                        if default_str == "":
                            raise InvalidOperation("Empty string cannot be converted to Decimal.")
                        final_value = Decimal(default_str)
                    elif cast_type == int:
                        if default_str == "":
                            raise ValueError("Empty string cannot be converted to int.")
                        final_value = int(Decimal(default_str))
                    elif cast_type == float:
                        if default_str == "":
                            raise ValueError("Empty string cannot be converted to float.")
                        final_value = float(default_str)
                    elif cast_type == str:
                        final_value = default_str
                    else:
                        final_value = default  # Fallback to raw default if type unknown

                    logger.warning(
                        f"{Fore.YELLOW}Successfully used casted default value for {key}: '{log_value(final_value)}'{Style.RESET_ALL}"
                    )
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to cast BOTH provided value ('{log_value(value_to_cast)}') AND default value ('{log_value(default)}') for key '{key}' to {cast_type.__name__}. Error on default: {e_default}{Style.RESET_ALL}"
                    )
                    raise ValueError(
                        f"Configuration error: Cannot cast value or default for key '{key}' to {cast_type.__name__}."
                    )

        # Log the final type and value being used
        logger.debug(
            f"{color}Using final value for {key}: {log_value(final_value)} (Type: {type(final_value).__name__}) (Source: {source}){Style.RESET_ALL}"
        )
        return final_value


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL_STR: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": 25,  # Custom level
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Set default level to INFO if the env var value is invalid
LOGGING_LEVEL: int = LOGGING_LEVEL_MAP.get(LOGGING_LEVEL_STR, logging.INFO)

# Define custom SUCCESS level
SUCCESS_LEVEL: int = LOGGING_LEVEL_MAP["SUCCESS"]
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# Configure basic logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],  # Output to console
)
logger: logging.Logger = logging.getLogger(__name__)  # Get the root logger


# Define the success method for the logger instance
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a message with severity 'SUCCESS'."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add the method to the Logger class (careful with type hinting here)
logging.Logger.success = log_success  # type: ignore[attr-defined]

# Apply colors if outputting to a TTY (like Termux console or standard terminal)
if sys.stdout.isatty():
    # Define color mappings for levels
    level_colors = {
        logging.DEBUG: f"{Fore.CYAN}{Style.DIM}",
        logging.INFO: f"{Fore.BLUE}",
        SUCCESS_LEVEL: f"{Fore.MAGENTA}{Style.BRIGHT}",
        logging.WARNING: f"{Fore.YELLOW}{Style.BRIGHT}",
        logging.ERROR: f"{Fore.RED}{Style.BRIGHT}",
        logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
    }
    # Apply colors to level names
    for level, color_style in level_colors.items():
        level_name = logging.getLevelName(level)
        logging.addLevelName(level, f"{color_style}{level_name}{Style.RESET_ALL}")
else:
    # If not a TTY, ensure SUCCESS level name is still registered without color codes
    logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()  # Forge the configuration object from environment variables
except ValueError as config_error:
    # Error should have been logged within Config init or _get_env
    logger.critical(
        f"{Back.RED}{Fore.WHITE}Configuration loading failed. Cannot continue spellcasting. Error: {config_error}{Style.RESET_ALL}"
    )
    # Attempt SMS alert if possible (basic config might be loaded for SMS settings)
    # Use raw os.getenv for sms settings here as CONFIG might not be fully initialized
    if os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true" and os.getenv("SMS_RECIPIENT_NUMBER"):
        try:
            # Manually construct minimal needed parts for alert
            temp_config_for_sms = type(
                "obj",
                (object,),
                {
                    "enable_sms_alerts": True,
                    "sms_recipient_number": os.getenv("SMS_RECIPIENT_NUMBER"),
                    "sms_timeout_seconds": int(os.getenv("SMS_TIMEOUT_SECONDS", "30")),
                    "symbol": os.getenv("SYMBOL", "UNKNOWN_SYMBOL"),  # Use fallback for symbol in SMS
                    "strategy_name": os.getenv("STRATEGY_NAME", "UNKNOWN_STRATEGY"),  # Use fallback
                },
            )()
            # Need to temporarily assign to CONFIG for send_sms_alert to work
            _original_config = globals().get("CONFIG")
            globals()["CONFIG"] = temp_config_for_sms
            send_sms_alert(
                f"[{CONFIG.strategy_name}] CRITICAL CONFIG ERROR: {config_error}. Bot failed to start on {CONFIG.symbol}."
            )
            if "_original_config" in locals():
                globals()["CONFIG"] = _original_config  # Restore original (likely None)
            else:
                del globals()["CONFIG"]  # Clean up temp if original didn't exist
        except Exception as sms_err:
            logger.error(f"Failed to send SMS alert about config error: {sms_err}")
    sys.exit(1)
except Exception as general_config_error:
    # Catch any other unexpected error during config initialization
    logger.critical(
        f"{Back.RED}{Fore.WHITE}Unexpected critical error during configuration loading: {general_config_error}{Style.RESET_ALL}"
    )
    logger.debug(traceback.format_exc())
    # Attempt SMS alert similarly
    if os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true" and os.getenv("SMS_RECIPIENT_NUMBER"):
        try:
            temp_config_for_sms = type(
                "obj",
                (object,),
                {
                    "enable_sms_alerts": True,
                    "sms_recipient_number": os.getenv("SMS_RECIPIENT_NUMBER"),
                    "sms_timeout_seconds": int(os.getenv("SMS_TIMEOUT_SECONDS", "30")),
                    "symbol": os.getenv("SYMBOL", "UNKNOWN_SYMBOL"),
                    "strategy_name": os.getenv("STRATEGY_NAME", "UNKNOWN_STRATEGY"),
                },
            )()
            _original_config = globals().get("CONFIG")
            globals()["CONFIG"] = temp_config_for_sms
            send_sms_alert(
                f"[{CONFIG.strategy_name}] UNEXPECTED CONFIG ERROR: {type(general_config_error).__name__}. Bot failed on {CONFIG.symbol}."
            )
            if "_original_config" in locals():
                globals()["CONFIG"] = _original_config
            else:
                del globals()["CONFIG"]
        except Exception as sms_err:
            logger.error(f"Failed to send SMS alert about unexpected config error: {sms_err}")
    sys.exit(1)


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to a Decimal, handling None, pandas NA, and potential errors.

    Args:
        value: The value to convert (can be string, float, int, Decimal, None, pandas NA, etc.).
        default: The Decimal value to return if conversion fails or input is None/NA.

    Returns:
        The converted Decimal value or the default.
    """
    if pd.isna(value) or value is None:
        return default
    try:
        # Using str(value) handles various input types more reliably before Decimal conversion
        # Strip whitespace in case value is a string from env var or similar
        str_value = str(value).strip()
        if str_value == "":  # Handle empty strings explicitly
            raise InvalidOperation("Cannot convert empty string to Decimal.")
        return Decimal(str_value)
    except (InvalidOperation, TypeError, ValueError) as e:
        # Log a warning, but only if the value was not None/NA initially
        if not (pd.isna(value) or value is None):
            logger.warning(
                f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}. Error: {e}"
            )
        return default


def format_order_id(order_id: str | int | None) -> str:
    """Returns the last 6 characters of an order ID for concise logging, or 'N/A'."""
    if order_id:
        order_id_str = str(order_id)
        # Handle potential UUIDs or other long IDs, take last 6 chars
        return f"...{order_id_str[-6:]}" if len(order_id_str) > 6 else order_id_str
    return "N/A"


# --- Precision Formatting - Shaping the Numbers for the Exchange ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal | int) -> str:
    """Formats a price according to the exchange's market precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        price: The price value (float, Decimal, or int).

    Returns:
        The price formatted as a string according to market precision.
        Returns a normalized Decimal string as fallback on error.
    """
    price_decimal = safe_decimal_conversion(price)
    if price_decimal.is_zero() and Decimal(str(price)).is_zero():  # Handle price is actually 0
        return "0"

    try:
        # Ensure the market is loaded
        if symbol not in exchange.markets:
            logger.warning(f"Market {symbol} not loaded in format_price. Attempting to load.")
            exchange.load_markets()
        if symbol not in exchange.markets:
            raise ccxt.BadSymbol(f"Market {symbol} could not be loaded for formatting.")

        # CCXT formatting methods typically expect float input
        price_float = float(price_decimal)
        formatted_price = exchange.price_to_precision(symbol, price_float)

        # Extra check: Ensure the formatted price isn't zero if the input wasn't,
        # which could happen with extremely small prices and precision rules.
        # Use Decimal comparison to avoid float issues
        if safe_decimal_conversion(formatted_price).is_zero() and not price_decimal.is_zero():
            logger.warning(
                f"Price formatting resulted in zero ({formatted_price}) for non-zero input {price_decimal}. Using Decimal normalize."
            )
            # Fallback to normalized Decimal string if CCXT result is zero for a non-zero input
            return str(price_decimal.normalize())
        return formatted_price

    except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
        logger.error(
            f"{Fore.RED}Error shaping price {price_decimal} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        # Fallback: return a normalized string representation of the Decimal
        return str(price_decimal.normalize())
    except Exception as e_unexp:
        logger.error(
            f"{Fore.RED}Unexpected error shaping price {price_decimal} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return str(price_decimal.normalize())


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal | int) -> str:
    """Formats an amount (quantity) according to the exchange's market precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        amount: The amount value (float, Decimal, or int).

    Returns:
        The amount formatted as a string according to market precision.
        Returns a normalized Decimal string as fallback on error.
    """
    amount_decimal = safe_decimal_conversion(amount)
    if amount_decimal.is_zero() and Decimal(str(amount)).is_zero():  # Handle amount is actually 0
        return "0"

    try:
        # Ensure the market is loaded
        if symbol not in exchange.markets:
            logger.warning(f"Market {symbol} not loaded in format_amount. Attempting to load.")
            exchange.load_markets()
        if symbol not in exchange.markets:
            raise ccxt.BadSymbol(f"Market {symbol} could not be loaded for formatting.")

        # CCXT formatting methods typically expect float input
        amount_float = float(amount_decimal)
        formatted_amount = exchange.amount_to_precision(symbol, amount_float)

        # Extra check: Ensure the formatted amount isn't zero if the input wasn't
        # Use Decimal comparison to avoid float issues
        if safe_decimal_conversion(formatted_amount).is_zero() and not amount_decimal.is_zero():
            logger.warning(
                f"Amount formatting resulted in zero ({formatted_amount}) for non-zero input {amount_decimal}. Using Decimal normalize."
            )
            # Fallback to normalized Decimal string if CCXT result is zero for a non-zero input
            return str(amount_decimal.normalize())
        return formatted_amount

    except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
        logger.error(
            f"{Fore.RED}Error shaping amount {amount_decimal} for {symbol}: {e}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        # Fallback: return a normalized string representation of the Decimal
        return str(amount_decimal.normalize())
    except Exception as e_unexp:
        logger.error(
            f"{Fore.RED}Unexpected error shaping amount {amount_decimal} for {symbol}: {e_unexp}. Using raw Decimal string.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return str(amount_decimal.normalize())


# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: bool | None = None  # Cache the result of checking command existence


def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using the 'termux-sms-send' command, if enabled and available.

    Checks for command existence once and caches the result.

    Args:
        message: The text message to send.

    Returns:
        True if the SMS command was executed successfully (return code 0), False otherwise.
    """
    global _termux_sms_command_exists

    # Ensure CONFIG exists before accessing its attributes
    if "CONFIG" not in globals() or not hasattr(CONFIG, "enable_sms_alerts") or not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled by configuration or CONFIG not ready.")
        return False

    # Check for command existence only once per script run
    if _termux_sms_command_exists is None:
        termux_command_path = shutil.which("termux-sms-send")
        _termux_sms_command_exists = termux_command_path is not None
        if not _termux_sms_command_exists:
            logger.warning(
                f"{Fore.YELLOW}SMS alerts enabled, but 'termux-sms-send' command not found in PATH. "
                f"Ensure Termux:API is installed (`pkg install termux-api`) and PATH is configured correctly.{Style.RESET_ALL}"
            )
        else:
            logger.debug(f"Found 'termux-sms-send' command at: {termux_command_path}")

    if not _termux_sms_command_exists:
        return False  # Don't proceed if command is missing

    # Ensure recipient number is configured
    if not hasattr(CONFIG, "sms_recipient_number") or not CONFIG.sms_recipient_number:
        # Warning already logged during config validation if number is missing while enabled
        logger.debug("SMS recipient number not configured, cannot send alert.")
        return False

    # Ensure timeout is configured
    sms_timeout = getattr(CONFIG, "sms_timeout_seconds", 30)

    try:
        # Prepare the command spell. The message should be the last argument(s).
        # No special quoting needed by termux-sms-send usually, it takes the rest as the message.
        command: list[str] = ["termux-sms-send", "-n", CONFIG.sms_recipient_number, message]
        logger.info(
            f"{Fore.MAGENTA}Dispatching SMS whisper to {CONFIG.sms_recipient_number} (Timeout: {sms_timeout}s)...{Style.RESET_ALL}"
        )
        logger.debug(f"Executing command: {' '.join(shlex.quote(arg) for arg in command)}")  # Log the command safely

        # Execute the spell via subprocess with timeout and output capture
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,  # Decode stdout/stderr as text
            check=False,  # Don't raise exception on non-zero exit code
            timeout=sms_timeout,
        )

        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}")
            if result.stdout:
                logger.debug(f"SMS Send stdout: {result.stdout.strip()}")
            return True
        else:
            # Log error details from stderr if available
            error_details = result.stderr.strip() if result.stderr else "No stderr output"
            logger.error(
                f"{Fore.RED}SMS whisper failed. Return Code: {result.returncode}, Stderr: {error_details}{Style.RESET_ALL}"
            )
            if result.stdout:
                logger.error(f"SMS Send stdout (on error): {result.stdout.strip()}")
            return False
    except FileNotFoundError:
        # This shouldn't happen due to the check above, but handle defensively
        logger.error(
            f"{Fore.RED}SMS failed: 'termux-sms-send' command vanished unexpectedly? Ensure Termux:API is installed.{Style.RESET_ALL}"
        )
        _termux_sms_command_exists = False  # Update cache
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: Command timed out after {sms_timeout}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected disturbance during dispatch: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance.

    Handles authentication, loads markets, performs basic connectivity checks,
    and configures necessary options for Bybit V5 API interaction.

    Returns:
        A configured CCXT Bybit exchange instance, or None if initialization fails.
    """
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        # This should technically be caught by Config validation, but double-check
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret runes missing. Cannot open portal.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{CONFIG.strategy_name}] CRITICAL: API keys missing. Spell failed on {CONFIG.symbol}.")
        return None
    try:
        # Forging the connection with Bybit V5 defaults
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,  # Respect the exchange spirits' limits
                "options": {
                    "defaultType": "linear",  # Assume USDT perpetuals unless symbol specifies otherwise
                    "adjustForTimeDifference": True,  # Sync client time with server time for request validity
                    # V5 API specific options might be added here if needed, but CCXT handles most.
                    # Example: Set default category if needed globally (though usually better per-call)
                    # 'defaultCategory': 'linear',
                },
                # Explicitly set API version if CCXT default changes or issues arise
                # 'options': {'api-version': 'v5'}, # Uncomment if explicit V5 needed, CCXT usually handles it
                "recvWindow": CONFIG.default_recv_window,  # Set custom receive window
            }
        )

        # --- Testnet Configuration ---
        # Uncomment the following line to use Bybit's testnet environment
        # exchange.set_sandbox_mode(True)
        # logger.warning(f"{Back.YELLOW}{Fore.BLACK}!!! TESTNET MODE ACTIVE !!!{Style.RESET_ALL}")

        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True)  # Force reload for fresh market data and limits
        logger.debug(f"Loaded {len(exchange.markets)} market structures.")

        # --- Initial Authentication & Connectivity Check ---
        logger.debug("Performing initial balance check for authentication and V5 connectivity...")
        try:
            # Fetch balance using V5 specific parameters to confirm keys and API version access
            # CCXT's fetchBalance for Bybit V5 requires category in params
            balance = exchange.fetch_balance(params={"category": "linear"})
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (Targeting V5 API).{Style.RESET_ALL}"
            )
            # Display warning only if NOT in sandbox mode
            # Use getattr defensively as 'sandbox' might not be present
            if not getattr(exchange, "sandbox", False):
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! LIVE SCALPING MODE ACTIVE - EXTREME CAUTION ADVISED !!!{Style.RESET_ALL}"
                )

            # Basic check for sufficient funds (optional, but good practice)
            total_equity = safe_decimal_conversion(balance.get("total", {}).get(CONFIG.usdt_symbol))
            if total_equity <= CONFIG.min_order_value_usdt:  # Use min order value as a low threshold
                logger.warning(
                    f"{Fore.YELLOW}Initial Balance Check: Total equity ({total_equity:.2f} {CONFIG.usdt_symbol}) appears low. Ensure sufficient funds for trading.{Style.RESET_ALL}"
                )

            send_sms_alert(f"[{CONFIG.strategy_name}] Portal opened & authenticated on {CONFIG.symbol}.")
            return exchange
        except ccxt.AuthenticationError as auth_err:
            # Specific handling for auth errors after initial connection
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Authentication failed during balance check: {auth_err}. Check API keys, permissions, IP whitelist, and account status on Bybit.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{CONFIG.strategy_name}] CRITICAL: Authentication FAILED ({auth_err}). Spell failed on {CONFIG.symbol}."
            )
            return None
        except ccxt.ExchangeError as ex_err:
            # Catch V5 specific errors like invalid category if API setup is wrong
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Exchange error during initial balance check (V5 connectivity issue?): {ex_err}.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{CONFIG.strategy_name}] CRITICAL: Exchange Error on Init ({ex_err}). Spell failed on {CONFIG.symbol}."
            )
            return None

    # --- Broader Error Handling for Initialization ---
    except ccxt.AuthenticationError as e:  # Catch auth error during initial setup
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Authentication failed during initial connection setup: {e}.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{CONFIG.strategy_name}] CRITICAL: Authentication FAILED on setup ({e}). Spell failed on {CONFIG.symbol}."
        )
    except ccxt.NetworkError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Network disturbance during portal opening: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{CONFIG.strategy_name}] CRITICAL: Network Error on Init ({e}). Spell failed on {CONFIG.symbol}."
        )
    except ccxt.ExchangeError as e:  # Catch other exchange errors during setup
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Exchange spirit rejected portal opening: {e}. Check Bybit status, API documentation, or account status.{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{CONFIG.strategy_name}] CRITICAL: Exchange Error on Init ({e}). Spell failed on {CONFIG.symbol}."
        )
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected chaos during portal opening: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{CONFIG.strategy_name}] CRITICAL: Unexpected Init Error: {type(e).__name__}. Spell failed on {CONFIG.symbol}."
        )

    return None  # Return None if any initialization step failed


# --- Robust pandas_ta Column Identification Helper ---
def find_pandas_ta_column(
    df: pd.DataFrame, prefix_hint: str, suffix_hint: str = "", expected_count: int = 1
) -> str | None:
    """Finds a column in a DataFrame added by pandas_ta, based on prefix/suffix hints.
    Designed to be more robust than guessing the exact name.

    Args:
        df: The DataFrame *after* running df.ta.indicator(append=True).
        prefix_hint: The expected start of the column name (e.g., "SUPERT", "STOCHRSIk").
        suffix_hint: The expected end of the column name (e.g., "d", "l", "s"). Can be empty.
        expected_count: The expected number of columns matching the pattern (usually 1).

    Returns:
        The name of the found column, or None if not found or multiple matches.
    """
    # Get the column names added in the last pandas_ta call
    # This is a bit of a heuristic - pandas_ta usually adds columns last
    # Or we could compare columns before and after the call
    # For simplicity and common use case, let's assume new columns related
    # to the indicator are added *during* the call and match hints.

    # Look for columns that contain the prefix hint and end with the suffix hint
    matching_cols = [col for col in df.columns if prefix_hint in col and col.endswith(suffix_hint)]

    if not matching_cols:
        logger.debug(f"pandas_ta Finder: No columns found matching prefix '{prefix_hint}' and suffix '{suffix_hint}'.")
        return None
    elif len(matching_cols) > expected_count:
        # This might happen if multiple indicators with similar names/params were added or hints are too broad
        logger.warning(
            f"pandas_ta Finder: Found multiple columns matching prefix '{prefix_hint}' and suffix '{suffix_hint}': {matching_cols}. Expected {expected_count}. Cannot reliably identify."
        )
        # Attempt to return the *last* one added, which is often the target, but log the ambiguity
        return matching_cols[-1]  # Heuristic: assume the most recent is the intended one
    elif len(matching_cols) < expected_count:
        logger.warning(
            f"pandas_ta Finder: Found {len(matching_cols)} columns matching prefix '{prefix_hint}' and suffix '{suffix_hint}': {matching_cols}. Expected {expected_count}. Cannot reliably identify."
        )
        return None  # Not enough columns found
    else:  # len(matching_cols) == expected_count
        logger.debug(
            f"pandas_ta Finder: Found unique column matching prefix '{prefix_hint}' and suffix '{suffix_hint}': {matching_cols[0]}."
        )
        return matching_cols[0]  # Return the single match


# --- Indicator Calculation Functions - Scrying the Market ---


def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, using robust column identification.

    Args:
        df: Pandas DataFrame with 'high', 'low', 'close' columns.
        length: The ATR lookback period for the Supertrend calculation.
        multiplier: The ATR multiplier for the Supertrend calculation.
        prefix: A string prefix to add to the resulting column names (e.g., "confirm_").

    Returns:
        The DataFrame with added Supertrend columns:
        - f'{prefix}supertrend': The Supertrend line value (Decimal).
        - f'{prefix}trend': Boolean indicating uptrend (True) or downtrend (False).
        - f'{prefix}st_long': Boolean, True if trend flipped to Long on this candle.
        - f'{prefix}st_short': Boolean, True if trend flipped to Short on this candle.
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]

    required_input_cols = ["high", "low", "close"]
    # Estimate minimum data length needed for pandas_ta Supertrend
    # Needs ATR length + buffer for initial calculations.
    min_len = length + 15  # Adding a more generous buffer

    # Input validation
    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA  # Assign NA to expected output columns
        return df

    try:
        # Store columns before calculation to identify new ones added by pandas_ta
        initial_columns = set(df.columns)

        # pandas_ta expects float multiplier for calculation
        logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={float(multiplier)}")
        # Calculate using pandas_ta, appending results to the DataFrame
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # --- Robust Verification and Renaming/Conversion ---
        # Identify new columns added by pandas_ta
        new_columns = list(set(df.columns) - initial_columns)
        logger.debug(f"pandas_ta added columns: {new_columns}")

        # Programmatically find the correct columns based on typical pandas_ta names
        # Supertrend line: Starts with SUPERT_, contains length and multiplier
        st_col = find_pandas_ta_column(df, "SUPERT_", suffix_hint=str(float(multiplier)).replace(".", "_"))
        # Trend direction: Starts with SUPERTd_, contains length and multiplier
        st_trend_col = find_pandas_ta_column(df, "SUPERTd_", suffix_hint=str(float(multiplier)).replace(".", "_"))
        # Long flip signal: Starts with SUPERTl_, contains length and multiplier
        st_long_col = find_pandas_ta_column(df, "SUPERTl_", suffix_hint=str(float(multiplier)).replace(".", "_"))
        # Short flip signal: Starts with SUPERTs_, contains length and multiplier
        st_short_col = find_pandas_ta_column(df, "SUPERTs_", suffix_hint=str(float(multiplier)).replace(".", "_"))

        if not all([st_col, st_trend_col, st_long_col, st_short_col]):
            # Find which specific columns are missing
            missing_details = f"ST: {st_col is None}, Trend: {st_trend_col is None}, Long: {st_long_col is None}, Short: {st_short_col is None}"
            logger.error(
                f"{Fore.RED}Scrying ({col_prefix}ST): Failed to find all expected output columns from pandas_ta after calculation. Missing: {missing_details}. Check pandas_ta version or symbol data.{Style.RESET_ALL}"
            )
            # Attempt to clean up any partial columns found among new columns
            partial_cols_found = [c for c in [st_col, st_trend_col, st_long_col, st_short_col] if c]
            if partial_cols_found:
                df.drop(columns=partial_cols_found, errors="ignore", inplace=True)
            for col in target_cols:
                df[col] = pd.NA  # Nullify results
            return df

        # Convert Supertrend value to Decimal, interpret trend and flips
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        # Trend: 1 = Uptrend, -1 = Downtrend. Convert to boolean: True for Up, False for Down.
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1
        # Flip Signals:
        # SUPERTl: Non-NaN (often 1.0) when trend flips Long.
        # SUPERTs: Non-NaN (often -1.0) when trend flips Short.
        df[f"{col_prefix}st_long"] = df[st_long_col].notna()  # True if flipped Long this candle
        df[f"{col_prefix}st_short"] = df[st_short_col].notna()  # True if flipped Short this candle

        # Check for NaNs in critical output columns (last row)
        if df[target_cols].iloc[-1].isnull().any():
            nan_cols = df[target_cols].iloc[-1].isnull()
            nan_details = ", ".join([col for col in target_cols if nan_cols[col]])
            logger.warning(
                f"{Fore.YELLOW}Scrying ({col_prefix}ST): Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}"
            )

        # Clean up raw columns created by pandas_ta
        raw_cols_to_drop = [c for c in [st_col, st_trend_col, st_long_col, st_short_col] if c is not None]
        df.drop(columns=raw_cols_to_drop, errors="ignore", inplace=True)

        # Log the latest reading for debugging
        last_st_val = df[f"{col_prefix}supertrend"].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = "Up" if df[f"{col_prefix}trend"].iloc[-1] else "Down"
            signal = (
                "LONG FLIP"
                if df[f"{col_prefix}st_long"].iloc[-1]
                else ("SHORT FLIP" if df[f"{col_prefix}st_short"].iloc[-1] else "Hold")
            )
            trend_color = Fore.GREEN if last_trend == "Up" else Fore.RED
            logger.debug(
                f"Scrying ({col_prefix}ST({length},{multiplier})): Trend={trend_color}{last_trend}{Style.RESET_ALL}, Val={last_st_val:.4f}, Signal={signal}"
            )
        else:
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except KeyError as e:
        logger.error(
            f"{Fore.RED}Scrying ({col_prefix}ST): Error accessing column - likely pandas_ta issue, data problem, or naming mismatch: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume Simple Moving Average (SMA), and Volume Ratio.

    Args:
        df: Pandas DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len: The lookback period for ATR calculation.
        vol_ma_len: The lookback period for Volume SMA calculation.

    Returns:
        A dictionary containing:
        - 'atr': The latest ATR value (Decimal) or None.
        - 'volume_ma': The latest Volume SMA value (Decimal) or None.
        - 'last_volume': The latest volume value (Decimal) or None.
        - 'volume_ratio': The ratio of last volume to volume SMA (Decimal) or None.
        Returns None values if calculation fails or data is insufficient/invalid.
    """
    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    # Need sufficient data for both ATR and Volume MA calculations
    min_len = max(atr_len, vol_ma_len) + 15  # Add buffer for stability

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}"
        )
        return results

    try:
        # Store columns before calculation
        set(df.columns)

        # Calculate ATR (Average True Range) - Measure of volatility
        logger.debug(f"Scrying (ATR): Calculating with length={atr_len}")
        # Use pandas_ta for ATR calculation
        df.ta.atr(length=atr_len, append=True)
        atr_col_name_hint = f"ATRr_{atr_len}"  # Typical pandas_ta name for ATR

        # Find the actual ATR column
        atr_col = find_pandas_ta_column(df, atr_col_name_hint)

        if atr_col and atr_col in df.columns:
            # Convert last ATR value to Decimal
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr):
                results["atr"] = safe_decimal_conversion(last_atr)
            # Clean up the raw ATR column added by pandas_ta
            df.drop(columns=[atr_col], errors="ignore", inplace=True)
        else:
            logger.warning(
                f"ATR column matching hint '{atr_col_name_hint}' not found after calculation or resulted in NA. Check pandas_ta behavior."
            )

        # Calculate Volume Moving Average (SMA) and Ratio - Measure of market energy
        logger.debug(f"Scrying (Volume): Calculating SMA with length={vol_ma_len}")
        # Use pandas rolling mean for SMA of volume, handle potential NaNs in volume first
        df_cleaned_vol = df.copy()
        df_cleaned_vol["volume"] = pd.to_numeric(df_cleaned_vol["volume"], errors="coerce")
        if df_cleaned_vol["volume"].isnull().any():
            logger.warning(
                f"{Fore.YELLOW}Scrying (Volume): Found NaNs in volume data before SMA calculation. Using ffill/bfill on volume.{Style.RESET_ALL}"
            )
            df_cleaned_vol["volume"].ffill(inplace=True)
            df_cleaned_vol["volume"].bfill(inplace=True)
            if df_cleaned_vol["volume"].isnull().any():
                logger.error(
                    f"{Fore.RED}Scrying (Volume): Cannot fill all NaNs in volume. Volume calculations may be unreliable.{Style.RESET_ALL}"
                )

        volume_ma_col_name = f"volume_sma_{vol_ma_len}"
        # min_periods ensures we get a value even if window isn't full at the start
        df_cleaned_vol[volume_ma_col_name] = (
            df_cleaned_vol["volume"].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        )

        last_vol_ma = df_cleaned_vol[volume_ma_col_name].iloc[-1]
        last_vol = df_cleaned_vol["volume"].iloc[-1]  # Get the most recent volume bar

        # Convert results to Decimal
        if pd.notna(last_vol_ma):
            results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol):
            results["last_volume"] = safe_decimal_conversion(last_vol)

        # Calculate Volume Ratio (Last Volume / Volume MA) safely
        if (
            results["volume_ma"] is not None
            and results["volume_ma"] > CONFIG.position_qty_epsilon
            and results["last_volume"] is not None
        ):
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except (DivisionByZero, InvalidOperation) as ratio_err:
                logger.warning(
                    f"Division by zero or invalid op encountered calculating volume ratio (Volume MA likely zero/negligible). Error: {ratio_err}"
                )
                results["volume_ratio"] = None
            except Exception as ratio_err:
                logger.warning(f"Unexpected error calculating volume ratio: {ratio_err}")
                results["volume_ratio"] = None
        else:
            logger.debug(
                f"Scrying (Volume): Cannot calculate ratio (VolMA={results['volume_ma']}, LastVol={results['last_volume']})"
            )
            results["volume_ratio"] = None  # Set explicitly to None

        # Clean up the temporary volume MA column
        df.drop(columns=[volume_ma_col_name], errors="ignore", inplace=True)  # Drop from original df
        del df_cleaned_vol  # Clean up temporary df

        # Check for NaNs in critical output values
        if results["atr"] is None:
            logger.warning(f"{Fore.YELLOW}Scrying (ATR): Final ATR result is NA.{Style.RESET_ALL}")
        if results["volume_ma"] is None:
            logger.warning(f"{Fore.YELLOW}Scrying (Volume MA): Final Volume MA result is NA.{Style.RESET_ALL}")
        if results["last_volume"] is None:
            logger.warning(f"{Fore.YELLOW}Scrying (Volume): Final Last Volume result is NA.{Style.RESET_ALL}")
        if results["volume_ratio"] is None:
            logger.warning(f"{Fore.YELLOW}Scrying (Volume Ratio): Final Volume Ratio result is NA.{Style.RESET_ALL}")

        # Log calculated results
        atr_str = f"{results['atr']:.5f}" if results["atr"] is not None else "N/A"
        vol_ma_str = f"{results['volume_ma']:.2f}" if results["volume_ma"] is not None else "N/A"
        last_vol_str = f"{results['last_volume']:.2f}" if results["last_volume"] is not None else "N/A"
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results["volume_ratio"] is not None else "N/A"
        logger.debug(
            f"Scrying Results: ATR({atr_len})={Fore.CYAN}{atr_str}{Style.RESET_ALL}, "
            f"LastVol={last_vol_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}"
        )

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results)  # Nullify all results on error
    return results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    """Calculates StochRSI (%K and %D) and Momentum indicator using pandas_ta, robustly.

    Args:
        df: Pandas DataFrame with 'close' column.
        rsi_len: The lookback period for the RSI component of StochRSI.
        stoch_len: The lookback period for the Stochastic component of StochRSI.
        k: The smoothing period for the %K line of StochRSI.
        d: The smoothing period for the %D (signal) line of StochRSI.
        mom_len: The lookback period for the Momentum indicator.

    Returns:
        The DataFrame with added columns:
        - 'stochrsi_k': The StochRSI %K value (Decimal).
        - 'stochrsi_d': The StochRSI %D value (Decimal).
        - 'momentum': The Momentum value (Decimal).
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    # Estimate minimum length: StochRSI needs roughly RSI + Stoch + D periods. Momentum needs its own period.
    min_len_stochrsi = rsi_len + stoch_len + d + 15  # Add buffer
    min_len_mom = mom_len + 1
    min_len = max(min_len_stochrsi, min_len_mom)

    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df

    try:
        initial_columns = set(df.columns)

        # Calculate StochRSI using pandas_ta
        logger.debug(f"Scrying (StochRSI): Calculating with RSI={rsi_len}, Stoch={stoch_len}, K={k}, D={d}")
        df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=True)

        # Find the actual StochRSI columns using hints
        # Typical pandas_ta names: STOCHRSIk_stoch_rsi_k_d, STOCHRSId_stoch_rsi_k_d
        stochrsi_k_col = find_pandas_ta_column(df, "STOCHRSIk_", suffix_hint=f"_{stoch_len}_{rsi_len}_{k}_{d}")
        stochrsi_d_col = find_pandas_ta_column(df, "STOCHRSId_", suffix_hint=f"_{stoch_len}_{rsi_len}_{k}_{d}")

        # Assign results to main DataFrame and convert to Decimal
        if stochrsi_k_col and stochrsi_k_col in df.columns:
            df["stochrsi_k"] = df[stochrsi_k_col].apply(safe_decimal_conversion)
        else:
            logger.warning(
                "StochRSI K column matching hint not found after calculation or resulted in NA. Check pandas_ta naming/behavior."
            )
            df["stochrsi_k"] = pd.NA
        if stochrsi_d_col and stochrsi_d_col in df.columns:
            df["stochrsi_d"] = df[stochrsi_d_col].apply(safe_decimal_conversion)
        else:
            logger.warning(
                "StochRSI D column matching hint not found after calculation or resulted in NA. Check pandas_ta naming/behavior."
            )
            df["stochrsi_d"] = pd.NA

        # Calculate Momentum using pandas_ta
        logger.debug(f"Scrying (Momentum): Calculating with length={mom_len}")
        df.ta.mom(length=mom_len, append=True)
        mom_col_name_hint = f"MOM_{mom_len}"  # Standard pandas_ta name

        # Find the actual Momentum column
        mom_col = find_pandas_ta_column(df, mom_col_name_hint)

        if mom_col and mom_col in df.columns:
            df["momentum"] = df[mom_col].apply(safe_decimal_conversion)
            # Clean up raw momentum column
            df.drop(columns=[mom_col], errors="ignore", inplace=True)
        else:
            logger.warning(
                f"Momentum column matching hint '{mom_col_name_hint}' not found after calculation or resulted in NA. Check pandas_ta naming/behavior."
            )
            df["momentum"] = pd.NA

        # Check for NaNs in critical output columns (last row)
        if df[target_cols].iloc[-1].isnull().any():
            nan_cols = df[target_cols].iloc[-1].isnull()
            nan_details = ", ".join([col for col in target_cols if nan_cols[col]])
            logger.warning(
                f"{Fore.YELLOW}Scrying (StochRSI/Mom): Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}"
            )

        # Clean up raw columns created by pandas_ta that were not explicitly dropped
        cols_to_drop = [c for c in set(df.columns) - initial_columns if c not in target_cols]
        df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

        # Log latest values for debugging
        k_val = df["stochrsi_k"].iloc[-1]
        d_val = df["stochrsi_d"].iloc[-1]
        mom_val = df["momentum"].iloc[-1]

        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            k_color = (
                Fore.RED
                if k_val > CONFIG.stochrsi_overbought
                else (Fore.GREEN if k_val < CONFIG.stochrsi_oversold else Fore.CYAN)
            )
            d_color = (
                Fore.RED
                if d_val > CONFIG.stochrsi_overbought
                else (Fore.GREEN if d_val < CONFIG.stochrsi_oversold else Fore.CYAN)
            )
            mom_color = (
                Fore.GREEN
                if mom_val > CONFIG.position_qty_epsilon
                else (Fore.RED if mom_val < -CONFIG.position_qty_epsilon else Fore.WHITE)
            )
            logger.debug(
                f"Scrying (StochRSI/Mom): K={k_color}{k_val:.2f}{Style.RESET_ALL}, D={d_color}{d_val:.2f}{Style.RESET_ALL}, Mom({mom_len})={mom_color}{mom_val:.4f}{Style.RESET_ALL}"
            )
        else:
            logger.debug("Scrying (StochRSI/Mom): Resulted in NA for one or more values on last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates the Ehlers Fisher Transform indicator using pandas_ta, robustly.

    Args:
        df: Pandas DataFrame with 'high', 'low' columns.
        length: The lookback period for the Fisher Transform calculation.
        signal: The smoothing period for the signal line (often 1 for trigger-only).

    Returns:
        The DataFrame with added columns:
        - 'ehlers_fisher': The Fisher Transform value (Decimal).
        - 'ehlers_signal': The Fisher Transform signal line value (Decimal).
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    required_input_cols = ["high", "low"]
    min_len = length + signal + 15  # Add buffer for calculation stability

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df
    try:
        initial_columns = set(df.columns)

        logger.debug(f"Scrying (EhlersFisher): Calculating with length={length}, signal={signal}")
        df.ta.fisher(length=length, signal=signal, append=True)

        # Find the actual Fisher columns using hints
        # Typical pandas_ta names: FISHERT_length_signal, FISHERTs_length_signal
        fisher_col = find_pandas_ta_column(df, "FISHERT_", suffix_hint=f"_{length}_{signal}")
        signal_col = find_pandas_ta_column(df, "FISHERTs_", suffix_hint=f"_{length}_{signal}")

        # Assign results and convert to Decimal
        if fisher_col and fisher_col in df.columns:
            df["ehlers_fisher"] = df[fisher_col].apply(safe_decimal_conversion)
        else:
            logger.warning(
                "Ehlers Fisher column matching hint not found after calculation or resulted in NA. Check pandas_ta naming/behavior."
            )
            df["ehlers_fisher"] = pd.NA

        if signal_col and signal_col in df.columns:
            df["ehlers_signal"] = df[signal_col].apply(safe_decimal_conversion)
        else:
            # If signal=1, pandas_ta might not create a separate signal column, often it's the same as the fisher line
            if signal == 1 and fisher_col and fisher_col in df.columns and pd.notna(df[fisher_col].iloc[-1]):
                logger.debug(f"Ehlers Fisher signal length is 1, using Fisher line '{fisher_col}' as signal.")
                df["ehlers_signal"] = df["ehlers_fisher"]  # Use Fisher line itself as signal if Fisher line is valid
            else:
                logger.warning(
                    "Ehlers Signal column matching hint not found after calculation or resulted in NA. Check pandas_ta naming/behavior."
                )
                df["ehlers_signal"] = pd.NA

        # Check for NaNs in critical output columns (last row)
        if df[target_cols].iloc[-1].isnull().any():
            nan_cols = df[target_cols].iloc[-1].isnull()
            nan_details = ", ".join([col for col in target_cols if nan_cols[col]])
            logger.warning(
                f"{Fore.YELLOW}Scrying (EhlersFisher): Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}"
            )

        # Clean up raw columns created by pandas_ta that were not explicitly dropped
        cols_to_drop = [c for c in set(df.columns) - initial_columns if c not in target_cols]
        df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

        # Log latest values for debugging
        fish_val = df["ehlers_fisher"].iloc[-1]
        sig_val = df["ehlers_signal"].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
            logger.debug(
                f"Scrying (EhlersFisher({length},{signal})): Fisher={Fore.CYAN}{fish_val:.4f}{Style.RESET_ALL}, Signal={Fore.MAGENTA}{sig_val:.4f}{Style.RESET_ALL}"
            )
        else:
            logger.debug("Scrying (EhlersFisher): Resulted in NA for one or more values on last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def calculate_ema_cross(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates standard Exponential Moving Averages (EMA) for the EMA Cross strategy.

    *** WARNING: This uses standard EMAs and is NOT an Ehlers Super Smoother filter. ***
    The strategy name 'EMA_CROSS' reflects this. If you require true Ehlers filters,
    replace this calculation logic.

    Args:
        df: Pandas DataFrame with 'close' column.
        fast_len: The period for the fast EMA.
        slow_len: The period for the slow EMA.

    Returns:
        The DataFrame with added columns:
        - 'fast_ema': The fast EMA value (Decimal).
        - 'slow_ema': The slow EMA value (Decimal).
        Returns original DataFrame with NA columns if calculation fails or data is insufficient.
    """
    target_cols = ["fast_ema", "slow_ema"]
    required_input_cols = ["close"]
    # EMA needs buffer for stability, especially the slower one
    min_len = slow_len + 15

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (EMA Cross): Insufficient data (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Cannot calculate.{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df
    try:
        # *** PYRMETHUS NOTE / WARNING ***
        logger.warning(
            f"{Fore.YELLOW}{Style.DIM}Scrying (EMA Cross): Using standard EMA as placeholder for Ehlers Super Smoother. "
            f"This strategy path ('EMA_CROSS') uses standard EMAs and may not perform as a true Ehlers MA strategy. "
            f"Verify indicator suitability or implement actual Ehlers Super Smoother if needed.{Style.RESET_ALL}"
        )

        initial_columns = set(df.columns)

        logger.debug(f"Scrying (EMA Cross): Calculating Fast EMA({fast_len}), Slow EMA({slow_len})")
        # Use pandas_ta standard EMA calculation and convert to Decimal
        # Calculate separately to find specific column names
        df.ta.ema(length=fast_len, append=True)
        df.ta.ema(length=slow_len, append=True)

        # Find the actual EMA columns using hints
        # Typical pandas_ta names: EMA_length
        fast_ema_col = find_pandas_ta_column(df, "EMA_", suffix_hint=str(fast_len))
        slow_ema_col = find_pandas_ta_column(df, "EMA_", suffix_hint=str(slow_len))

        if fast_ema_col and fast_ema_col in df.columns:
            df["fast_ema"] = df[fast_ema_col].apply(safe_decimal_conversion)
        else:
            logger.warning(
                "Fast EMA column matching hint not found after calculation or resulted in NA. Check pandas_ta naming/behavior."
            )
            df["fast_ema"] = pd.NA

        if slow_ema_col and slow_ema_col in df.columns:
            df["slow_ema"] = df[slow_ema_col].apply(safe_decimal_conversion)
        else:
            logger.warning(
                "Slow EMA column matching hint not found after calculation or resulted in NA. Check pandas_ta naming/behavior."
            )
            df["slow_ema"] = pd.NA

        # Check for NaNs in critical output columns (last row)
        if df[target_cols].iloc[-1].isnull().any():
            nan_cols = df[target_cols].iloc[-1].isnull()
            nan_details = ", ".join([col for col in target_cols if nan_cols[col]])
            logger.warning(
                f"{Fore.YELLOW}Scrying (EMA Cross): Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{Style.RESET_ALL}"
            )

        # Clean up raw columns created by pandas_ta that were not explicitly dropped
        cols_to_drop = [c for c in set(df.columns) - initial_columns if c not in target_cols]
        df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

        # Log latest values for debugging
        fast_val = df["fast_ema"].iloc[-1]
        slow_val = df["slow_ema"].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            cross_color = Fore.GREEN if fast_val > slow_val else Fore.RED
            logger.debug(
                f"Scrying (EMA Cross({fast_len},{slow_len})): Fast={cross_color}{fast_val:.4f}{Style.RESET_ALL}, Slow={cross_color}{slow_val:.4f}{Style.RESET_ALL}"
            )
        else:
            logger.debug("Scrying (EMA Cross): Resulted in NA for one or more values on last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EMA Cross): Unexpected error during calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA  # Nullify results on error
    return df


def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes the L2 order book to calculate bid/ask volume ratio and spread.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        depth: The number of price levels on each side (bids/asks) to include in the volume ratio calculation.
        fetch_limit: The number of price levels to request from the exchange API (should be >= depth).

    Returns:
        A dictionary containing:
        - 'bid_ask_ratio': Ratio of total bid volume to total ask volume within the specified depth (Decimal) or None.
        - 'spread': Difference between best ask and best bid (Decimal) or None.
        - 'best_bid': The highest bid price (Decimal) or None.
        - 'best_ask': The lowest ask price (Decimal) or None.
        Returns None values if analysis fails or data is unavailable.
    """
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book Scrying: Fetching L2 {symbol} (Depth:{depth}, Request Limit:{fetch_limit})...")

    # Check if the exchange supports fetching L2 order book
    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying: Exchange '{exchange.id}' does not support fetchL2OrderBook method. Cannot analyze depth.{Style.RESET_ALL}"
        )
        return results

    try:
        # Fetching the order book's current state
        # Bybit V5 requires 'category' param for futures
        params = {"category": "linear"}  # Add category for V5 consistency
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit, params=params)
        bids: list[list[float | str]] = order_book.get("bids", [])  # List of [price, amount]
        asks: list[list[float | str]] = order_book.get("asks", [])  # List of [price, amount]

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}Order Book Scrying: Empty bids or asks received for {symbol}. Cannot analyze.{Style.RESET_ALL}"
            )
            return results  # Return defaults (all None)

        # Extract best bid/ask with Decimal precision
        # Ensure lists are not empty and contain price/amount pairs, then safely convert
        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        # Calculate spread
        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            try:
                results["spread"] = best_ask - best_bid
                logger.debug(
                    f"OB Scrying: Best Bid={Fore.GREEN}{best_bid:.4f}{Style.RESET_ALL}, Best Ask={Fore.RED}{best_ask:.4f}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results['spread']:.4f}{Style.RESET_ALL}"
                )
            except (InvalidOperation, TypeError) as e:
                logger.warning(f"{Fore.YELLOW}Error calculating spread: {e}. Skipping spread.{Style.RESET_ALL}")
                results["spread"] = None
        else:
            logger.debug(
                f"OB Scrying: Best Bid={best_bid or 'N/A'}, Best Ask={best_ask or 'N/A'} (Spread calculation skipped due to invalid bid/ask)"
            )

        # Sum total volume within the specified depth using Decimal for precision
        # Ensure list slicing doesn't go out of bounds and elements are valid pairs
        # Use generator expression with safe conversion
        bid_vol = sum(
            safe_decimal_conversion(bid[1], Decimal("0.0")) for bid in bids[: min(depth, len(bids))] if len(bid) >= 2
        )
        ask_vol = sum(
            safe_decimal_conversion(ask[1], Decimal("0.0")) for ask in asks[: min(depth, len(asks))] if len(ask) >= 2
        )
        logger.debug(
            f"OB Scrying (Depth {depth}): Total BidVol={Fore.GREEN}{bid_vol:.4f}{Style.RESET_ALL}, Total AskVol={Fore.RED}{ask_vol:.4f}{Style.RESET_ALL}"
        )

        # Calculate Bid/Ask Volume Ratio (Total Bid Volume / Total Ask Volume)
        if ask_vol > CONFIG.position_qty_epsilon:  # Avoid division by zero or near-zero
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                # Determine color based on configured thresholds for logging
                ratio_color = (
                    Fore.GREEN
                    if results["bid_ask_ratio"] >= CONFIG.order_book_ratio_threshold_long
                    else (
                        Fore.RED if results["bid_ask_ratio"] <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW
                    )
                )
                logger.debug(
                    f"OB Scrying Ratio (Bids/Asks): {ratio_color}{results['bid_ask_ratio']:.3f}{Style.RESET_ALL}"
                )
            except (DivisionByZero, InvalidOperation, Exception) as e:
                logger.warning(f"{Fore.YELLOW}Error calculating OB ratio: {e}{Style.RESET_ALL}")
                results["bid_ask_ratio"] = None
        else:
            logger.debug(
                f"OB Scrying Ratio: N/A (Ask volume within depth {depth} is zero or negligible: {ask_vol:.4f})"
            )
            results["bid_ask_ratio"] = None  # Set explicitly to None

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except IndexError:
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying Error: Index out of bounds accessing bids/asks for {symbol}. Order book data might be malformed or incomplete.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Unexpected Order Book Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())

    # Ensure results dictionary keys exist even if errors occurred (avoids KeyErrors later)
    # (Redundant now with initial dict definition, but good practice)
    results.setdefault("bid_ask_ratio", None)
    results.setdefault("spread", None)
    results.setdefault("best_bid", None)
    results.setdefault("best_ask", None)
    return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches OHLCV data, prepares it as a pandas DataFrame, ensures numeric types,
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
    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.{Style.RESET_ALL}"
        )
        return None
    try:
        logger.debug(f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})...")
        # Channeling the data stream from the exchange
        # Bybit V5 requires 'category' param for futures
        params = {"category": "linear"}  # Add category for V5 consistency
        ohlcv: list[list[int | float | str]] = exchange.fetch_ohlcv(
            symbol, timeframe=interval, limit=limit, params=params
        )

        if not ohlcv:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Market might be inactive, symbol incorrect, or API issue.{Style.RESET_ALL}"
            )
            return None

        if len(ohlcv) < limit:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: Only received {len(ohlcv)} candles for {symbol} ({interval}) instead of requested {limit}. Data history might be limited for this symbol/interval.{Style.RESET_ALL}"
            )
            if len(ohlcv) < CONFIG.api_fetch_limit_buffer:  # Basic check if data is critically short
                logger.error(
                    f"{Fore.RED}Data Fetch: Received critically low amount of data ({len(ohlcv)}). Cannot proceed.{Style.RESET_ALL}"
                )
                return None

        # Weaving data into a DataFrame structure
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # --- Data Cleaning and Preparation ---
        # Convert timestamp to datetime objects (UTC) and set as index
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
        except Exception as time_e:
            logger.error(f"{Fore.RED}Data Fetch: Error converting timestamp column: {time_e}{Style.RESET_ALL}")
            return None  # Cannot proceed without valid timestamps

        # Ensure OHLCV columns are numeric, coercing errors to NaN
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Robust NaN Handling ---
        initial_nan_count = df[numeric_cols].isnull().sum().sum()
        if initial_nan_count > 0:
            nan_counts_per_col = df[numeric_cols].isnull().sum()
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: Found {initial_nan_count} NaN values in OHLCV data after conversion:\n"
                f"{nan_counts_per_col[nan_counts_per_col > 0]}\nAttempting forward fill (ffill)...{Style.RESET_ALL}"
            )
            df.ffill(inplace=True)  # Fill NaNs with the previous valid observation

            # Check if NaNs remain (likely at the beginning of the series if data history is short)
            remaining_nan_count = df[numeric_cols].isnull().sum().sum()
            if remaining_nan_count > 0:
                logger.warning(
                    f"{Fore.YELLOW}NaNs remain after ffill ({remaining_nan_count}). Attempting backward fill (bfill)...{Style.RESET_ALL}"
                )
                df.bfill(inplace=True)  # Fill remaining NaNs with the next valid observation

                # Final check: if NaNs still exist, data is likely too gappy at start/end or completely invalid
                final_nan_count = df[numeric_cols].isnull().sum().sum()
                if final_nan_count > 0:
                    logger.error(
                        f"{Fore.RED}Data Fetch: Unfillable NaN values ({final_nan_count}) remain after ffill and bfill. "
                        f"Data quality insufficient for {symbol}. Columns with NaNs:\n{df[numeric_cols].isnull().sum()[df[numeric_cols].isnull().sum() > 0]}\nSkipping cycle.{Style.RESET_ALL}"
                    )
                    return None  # Cannot proceed with unreliable data

        # Check if the last candle is complete (timestamp is slightly in the past)
        # This is an approximation; exact candle close time is better but harder across exchanges
        if len(df) > 0:
            last_candle_time_utc = df.index[-1]
            now_utc = pd.Timestamp.now(tz="UTC")
            # A completed candle's timestamp should be before the current time,
            # typically by at least the interval duration. Allow a small buffer.
            # This check is heuristic and depends on exchange timestamp precision.
            # A common pattern is to fetch N+1 candles and drop the potentially incomplete last one,
            # but this bot fetches exactly what's needed + buffer. Relying on the exchange sending completed
            # candles is typical for high-frequency bots on low intervals.
            # Let's just log a warning if the last candle timestamp is suspiciously close to now.
            time_diff_seconds = (now_utc - last_candle_time_utc).total_seconds()
            interval_seconds = exchange.parse_timeframe(interval)  # Convert interval string to seconds
            if time_diff_seconds < interval_seconds * 0.8:  # If last candle is less than 80% through interval ago
                logger.warning(
                    f"{Fore.YELLOW}Data Fetch: Last candle timestamp ({last_candle_time_utc}) for {symbol} ({interval}) "
                    f"is very recent ({time_diff_seconds:.1f}s ago, interval is {interval_seconds}s). "
                    f"It might be incomplete. Using it anyway as typical for scalping, but be aware.{Style.RESET_ALL}"
                )

        logger.debug(f"Data Fetch: Successfully woven and cleaned {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(
            f"{Fore.YELLOW}Data Fetch: Disturbance gathering OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return None  # Return None if any error occurred


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position details using Bybit V5 API specifics (`fetchPositions`).
    Assumes One-Way Mode (looks for positionIdx=0). Includes parsing attached SL/TP/TSL.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').

    Returns:
        A dictionary containing:
        - 'side': Position side ('Long', 'Short', or 'None').
        - 'qty': Position quantity as Decimal (0.0 if flat).
        - 'entry_price': Average entry price as Decimal (0.0 if flat).
        - 'initial_margin_rate': Initial margin rate for the symbol (Decimal) or None.
        - 'leverage': Effective leverage for the symbol (Decimal) or None.
        - 'stop_loss': Native Stop Loss price (Decimal) or None.
        - 'take_profit': Native Take Profit price (Decimal) or None.
        - 'trailing_stop_price': Native Trailing Stop trigger price (Decimal) or None.
        - 'unrealized_pnl': Unrealized PNL (Decimal) or None.
        - 'liquidation_price': Liquidation Price (Decimal) or None.
        Returns default flat state dictionary on error or if no position found.
    """
    # Initialize default structure with all possible keys
    default_pos: dict[str, Any] = {
        "side": CONFIG.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
        "initial_margin_rate": None,  # Added
        "leverage": None,  # Added
        "stop_loss": None,  # Added
        "take_profit": None,  # Added
        "trailing_stop_price": None,  # Added (Note: This is the *trigger price*, not the activation price)
        "unrealized_pnl": None,  # Added
        "liquidation_price": None,  # Added
    }
    market_id: str | None = None
    market: dict[str, Any] | None = None
    category: str | None = None

    try:
        # Get market details to determine category (linear/inverse) and the exchange's specific ID
        market = exchange.market(symbol)
        market_id = market["id"]  # The exchange's specific ID (e.g., BTCUSDT)
        # Determine category based on market properties (linear = USDT margined)
        if market.get("linear"):
            category = "linear"
        elif market.get("inverse"):
            category = "inverse"
        else:
            # Fallback or error if category cannot be determined (shouldn't happen for loaded futures markets)
            logger.warning(
                f"{Fore.YELLOW}Position Check: Could not determine category (linear/inverse) for market '{symbol}'. Assuming 'linear'.{Style.RESET_ALL}"
            )
            category = "linear"  # Default assumption for this bot for this bot

    except (ccxt.BadSymbol, KeyError) as e:
        logger.error(
            f"{Fore.RED}Position Check: Failed to identify market structure for '{symbol}': {e}. Cannot check position.{Style.RESET_ALL}"
        )
        return default_pos
    except Exception as e_market:
        logger.error(
            f"{Fore.RED}Position Check: Unexpected error getting market info for '{symbol}': {e_market}. Cannot check position.{Style.RESET_ALL}"
        )
        return default_pos

    try:
        # Check if the exchange instance supports fetchPositions (should for Bybit V5 via CCXT)
        if not exchange.has.get("fetchPositions"):
            logger.error(
                f"{Fore.RED}Position Check: Exchange '{exchange.id}' CCXT instance does not support fetchPositions method. Cannot get V5 position data.{Style.RESET_ALL}"
            )
            # This indicates a potential issue with the CCXT version or exchange setup
            return default_pos  # Return default state on critical method absence

        # Fetch positions for the specific symbol and category
        # Bybit V5 fetchPositions requires params={'category': 'linear'} and optionally symbol
        # We only care about the One-Way position (positionIdx=0) for the target symbol
        params = {"category": category}
        if market_id:
            params["symbol"] = market_id  # Use exchange-specific ID if available

        # CCXT fetchPositions returns a list of positions. For One-Way, we expect max one per symbol.
        positions = exchange.fetch_positions(symbols=[symbol], params=params)  # Filter by unified symbol

        # Filter for the relevant position in One-Way mode (positionIdx=0)
        relevant_position = next(
            (
                p
                for p in positions
                if str(p.get("info", {}).get("positionIdx", "")) == "0" and p.get("symbol") == symbol
            ),
            None,
        )

        if (
            relevant_position
            and safe_decimal_conversion(relevant_position.get("info", {}).get("size", "0"))
            > CONFIG.position_qty_epsilon
        ):
            # Active position found
            pos_info = relevant_position.get("info", {})  # Access raw info dictionary
            pos_side_raw = pos_info.get("side", "").capitalize()  # "Buy" or "Sell"
            pos_size = safe_decimal_conversion(pos_info.get("size", "0"))  # Quantity
            pos_avg_entry = safe_decimal_conversion(pos_info.get("entryPrice", "0"))  # Entry price
            pos_leverage = safe_decimal_conversion(pos_info.get("leverage", "0"))  # Effective leverage
            pos_initial_margin_rate = safe_decimal_conversion(
                pos_info.get("initialMarginRate", "0")
            )  # Initial margin rate for asset
            pos_stop_loss = safe_decimal_conversion(pos_info.get("stopLoss", None))  # Native SL price
            pos_take_profit = safe_decimal_conversion(pos_info.get("takeProfit", None))  # Native TP price
            pos_trailing_stop_price = safe_decimal_conversion(
                pos_info.get("trailingStop", None)
            )  # Native TSL trigger price
            pos_unrealized_pnl = safe_decimal_conversion(pos_info.get("unrealisedPnl", None))  # Unrealized PNL
            pos_liquidation_price = safe_decimal_conversion(pos_info.get("liqPrice", None))  # Liquidation price

            # Map Bybit side ("Buy"/"Sell") to internal representation
            pos_side = (
                CONFIG.pos_long
                if pos_side_raw == "Buy"
                else (CONFIG.pos_short if pos_side_raw == "Sell" else CONFIG.pos_none)
            )

            logger.info(
                f"{Fore.CYAN}Position Check: Active {pos_side} position found for {symbol}. "
                f"Qty: {pos_size.normalize()}, Entry: {pos_avg_entry.normalize()}{Style.RESET_ALL}"
            )

            # Log native stops if attached
            stop_details = []
            if pos_stop_loss is not None and not pos_stop_loss.is_zero():
                stop_details.append(f"SL: {pos_stop_loss.normalize()}")
            if pos_take_profit is not None and not pos_take_profit.is_zero():
                stop_details.append(f"TP: {pos_take_profit.normalize()}")
            # Note: Bybit API returns 'trailingStop' as the *trigger price*, not the percentage or activation price
            if pos_trailing_stop_price is not None and not pos_trailing_stop_price.is_zero():
                stop_details.append(f"TSL Trigger: {pos_trailing_stop_price.normalize()}")
                # To get the TSL activation price, we'd need the position's high/low watermark and the percentage.
                # CCXT position structure doesn't always expose high/low watermark directly.
                # We'll rely on checking if 'trailingStop' is non-zero as a proxy for TSL being active.
            if stop_details:
                logger.info(f"{Fore.CYAN}Position Check: Attached Stops -> {' | '.join(stop_details)}{Style.RESET_ALL}")
            else:
                # This should ideally not happen if the bot placed the order with stops
                logger.warning(
                    f"{Fore.YELLOW}Position Check: No native SL/TP/TSL detected on the active position!{Style.RESET_ALL}"
                )
                # TODO: Potentially handle this by attempting to attach stops dynamically? (Adds complexity)

            # Return the detailed position state
            return {
                "side": pos_side,
                "qty": pos_size,
                "entry_price": pos_avg_entry,
                "initial_margin_rate": pos_initial_margin_rate,
                "leverage": pos_leverage,
                "stop_loss": pos_stop_loss,
                "take_profit": pos_take_profit,
                "trailing_stop_price": pos_trailing_stop_price,
                "unrealized_pnl": pos_unrealized_pnl,
                "liquidation_price": pos_liquidation_price,
            }
        else:
            # No active position found (either list is empty, or the One-Way position has size 0)
            logger.info(
                f"{Fore.BLUE}Position Check: No active One-Way (positionIdx=0) position found for {market_id}. Currently Flat.{Style.RESET_ALL}"
            )
            return default_pos

    except ccxt.BadSymbol:
        logger.error(
            f"{Fore.RED}Position Check Error: Invalid symbol '{symbol}' during fetchPositions.{Style.RESET_ALL}"
        )
        return default_pos
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(
            f"{Fore.YELLOW}Position Check Error for {symbol}: {type(e).__name__} - {e}. Cannot get current position state.{Style.RESET_ALL}"
        )
        # Return default state on temporary API/network errors
        return default_pos
    except Exception as e:
        logger.error(
            f"{Fore.RED}Position Check Unexpected Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        # Return default state on unexpected errors
        return default_pos


def calculate_order_quantity(
    exchange: ccxt.Exchange,
    symbol: str,
    account_balance: Decimal,
    current_price: Decimal,
    stop_loss_price: Decimal,
    side: str,
    market_data: dict[str, Any],
) -> Decimal | None:
    """Calculates the order quantity based on risk percentage, account equity,
    estimated stop loss distance, current price, leverage, and market limits.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        account_balance: The total available equity in the quote currency (USDT).
        current_price: The current market price for the symbol.
        stop_loss_price: The calculated price where the stop loss would trigger.
        side: The trade side ('buy' or 'sell').
        market_data: Dictionary containing market details for the symbol.

    Returns:
        The calculated order quantity (Decimal) formatted to market precision,
        or None if calculation is not possible or results in zero/negative quantity.
    """
    if (
        account_balance <= CONFIG.min_order_value_usdt
        or current_price <= CONFIG.position_qty_epsilon
        or stop_loss_price <= CONFIG.position_qty_epsilon
        or not market_data
    ):
        logger.warning(
            f"{Fore.YELLOW}Qty Calc: Insufficient funds ({account_balance}), invalid price ({current_price}, SL:{stop_loss_price}), or market data missing. Cannot calculate quantity.{Style.RESET_ALL}"
        )
        return None

    try:
        # Ensure Decimal types
        account_balance_dec = safe_decimal_conversion(account_balance)
        current_price_dec = safe_decimal_conversion(current_price)
        stop_loss_price_dec = safe_decimal_conversion(stop_loss_price)
        leverage_dec = safe_decimal_conversion(CONFIG.leverage)
        risk_percentage_dec = CONFIG.risk_per_trade_percentage
        max_order_usdt_dec = CONFIG.max_order_usdt_amount
        min_order_usdt_value = CONFIG.min_order_value_usdt  # From config/constants

        # --- 1. Calculate quantity based on Risk % and SL distance ---
        # Risk Amount = Total Equity * Risk Percentage per trade
        risk_amount_usdt = account_balance_dec * risk_percentage_dec
        logger.debug(
            f"Qty Calc: Account Equity: {account_balance_dec.normalize()} {CONFIG.usdt_symbol}, Risk %: {risk_percentage_dec}, Risk Amount: {risk_amount_usdt.normalize()} {CONFIG.usdt_symbol}"
        )

        # Price difference between entry and stop loss
        price_diff = (current_price_dec - stop_loss_price_dec).abs()
        if price_diff <= CONFIG.position_qty_epsilon:
            logger.warning(
                f"{Fore.YELLOW}Qty Calc: Stop Loss price ({stop_loss_price_dec}) is too close or equal to current price ({current_price_dec}). Risk calculation requires a price difference. Cannot calculate quantity.{Style.RESET_ALL}"
            )
            return None

        # Calculate quantity based on (Risk Amount / Price Difference)
        # This gives the quantity where the loss at SL price equals the risk amount.
        # Example: Risk 10 USDT, Price diff 1 USDT -> Qty = 10 coins/contracts
        # The calculation is slightly more complex due to leverage and contract value
        # Position Value = Quantity * Entry Price
        # PnL (at SL) = (Exit Price - Entry Price) * Quantity * Contract Multiplier (often 1 for USDT)
        # For USDT pairs where quantity is in coins: Loss = abs(SL_Price - Entry_Price) * Quantity
        # So, Quantity = Risk Amount / abs(SL_Price - Entry_Price)
        try:
            quantity_from_risk = risk_amount_usdt / price_diff
            logger.debug(
                f"Qty Calc: Price Diff (Entry vs SL): {price_diff.normalize()}, Qty based on Risk: {quantity_from_risk.normalize()} coins/contracts"
            )
        except (DivisionByZero, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Qty Calc: Error calculating quantity from risk: {e}. Price difference likely zero or invalid.{Style.RESET_ALL}"
            )
            return None

        # --- 2. Calculate quantity based on Maximum Order Value ---
        # Max Quantity by Value = Max Order Value (USDT) / Current Price
        if current_price_dec > CONFIG.position_qty_epsilon:
            try:
                quantity_from_max_value = max_order_usdt_dec / current_price_dec
                logger.debug(
                    f"Qty Calc: Max Order Value: {max_order_usdt_dec.normalize()} {CONFIG.usdt_symbol}, Qty based on Max Value: {quantity_from_max_value.normalize()} coins/contracts"
                )
            except (DivisionByZero, InvalidOperation) as e:
                logger.error(
                    f"{Fore.RED}Qty Calc: Error calculating quantity from max value: {e}. Current price likely zero or invalid.{Style.RESET_ALL}"
                )
                quantity_from_max_value = Decimal("0.0")  # Set to zero if error
        else:
            quantity_from_max_value = Decimal("0.0")  # Set to zero if price is invalid

        # --- 3. Determine the minimum quantity (most conservative) ---
        # Use the smaller of the two calculated quantities
        # If max_order_usdt_dec is 0, quantity_from_max_value will be 0 or error, min() handles this.
        calculated_quantity_dec = (
            min(quantity_from_risk, quantity_from_max_value)
            if quantity_from_max_value > CONFIG.position_qty_epsilon
            else quantity_from_risk
        )

        logger.debug(
            f"Qty Calc: Calculated Quantity (Min of Risk/MaxValue): {calculated_quantity_dec.normalize()} coins/contracts"
        )

        # --- 4. Apply exchange minimums and step size ---
        market = exchange.market(symbol)  # Market details already loaded in initialize_exchange
        min_amount = safe_decimal_conversion(market.get("limits", {}).get("amount", {}).get("min", "0"))
        amount_step = safe_decimal_conversion(market.get("limits", {}).get("amount", {}).get("step", "0"))
        safe_decimal_conversion(
            market.get("limits", {}).get("price", {}).get("step", "0")
        )  # Also need price step for SL/TP formatting

        if calculated_quantity_dec < min_amount and min_amount > CONFIG.position_qty_epsilon:
            logger.warning(
                f"{Fore.YELLOW}Qty Calc: Calculated quantity {calculated_quantity_dec.normalize()} is below market minimum {min_amount.normalize()}. Adjusting up to minimum.{Style.RESET_ALL}"
            )
            final_quantity_dec = min_amount
        else:
            final_quantity_dec = calculated_quantity_dec

        # Adjust quantity to be a multiple of the step size
        if amount_step > CONFIG.position_qty_epsilon and final_quantity_dec > CONFIG.position_qty_epsilon:
            # Round down to the nearest multiple of amount_step
            final_quantity_dec = (final_quantity_dec / amount_step).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            ) * amount_step
            logger.debug(
                f"Qty Calc: Adjusted quantity to market step {amount_step.normalize()}: {final_quantity_dec.normalize()} coins/contracts"
            )

        # Final check on calculated quantity
        if final_quantity_dec <= CONFIG.position_qty_epsilon:
            logger.warning(
                f"{Fore.YELLOW}Qty Calc: Final calculated quantity {final_quantity_dec.normalize()} is zero or negligible after adjustments. Cannot place order.{Style.RESET_ALL}"
            )
            return None

        # --- 5. Estimate Initial Margin Requirement and Check Against Free Margin ---
        # This is a crucial step for Bybit V5, which uses initial margin rate per asset.
        # Estimated Initial Margin = Quantity * Entry Price * Initial Margin Rate (for asset)
        # CCXT's market structure *might* contain `info.initial_margin_rate` or similar,
        # or we can use the `initial_margin_rate` from `fetch_positions` if we were already in a position.
        # For simplicity here, we will approximate initial margin using 1/leverage (or 1/max leverage if less than desired)
        # and compare against *free* balance, adding a buffer.
        # A more accurate method would require fetching instrument-specific margin tiers.
        # Let's use the provided `initial_margin_rate` from `fetch_positions` if available from a prior position check,
        # or fall back to 1/leverage if not.

        # Get the actual initial margin rate for the symbol.
        # CCXT's market['info'] might have initial_margin_rate, or the position info might have it.
        # A reliable way is often to fetch this specifically or use the position data if available.
        # Let's enhance get_current_position to return this if the bot was previously in a position.
        # If the bot is flat, we might need a dedicated call or rely on market data if available.
        # For now, we'll rely on a fallback to 1/leverage for the estimate if the specific rate isn't easily found.
        # Note: Bybit V5 API has a 'get-instruments-info' endpoint that provides this. CCXT might expose it.
        # For simplicity, let's use the 1/leverage heuristic with a buffer.

        # Use MIN(desired_leverage, market_max_leverage)
        max_market_leverage = safe_decimal_conversion(
            market.get("limits", {}).get("leverage", {}).get("max", Decimal("100.0"))
        )  # Default to 100 if not found
        effective_leverage_for_margin = min(leverage_dec, max_market_leverage)

        # Estimated initial margin rate heuristic: 1 / effective leverage
        # Note: This is an approximation. Actual rate might differ based on tiers/risk limits.
        if effective_leverage_for_margin > CONFIG.position_qty_epsilon:
            estimated_margin_rate = Decimal("1") / effective_leverage_for_margin
        else:
            logger.error(
                f"{Fore.RED}Qty Calc: Effective leverage zero or invalid ({effective_leverage_for_margin}). Cannot estimate margin.{Style.RESET_ALL}"
            )
            return None

        # Estimated Initial Margin Required for this order: Quantity * Price * Margin Rate
        estimated_margin_required = final_quantity_dec * current_price_dec * estimated_margin_rate
        logger.debug(
            f"Qty Calc: Estimated Initial Margin Required ({estimated_margin_rate.normalize()} rate): {estimated_margin_required.normalize()} {CONFIG.usdt_symbol}"
        )

        # Fetch *free* balance specifically (usable margin balance)
        try:
            balance = exchange.fetch_balance(params={"category": "linear"})
            free_balance_usdt = safe_decimal_conversion(balance.get("free", {}).get(CONFIG.usdt_symbol))
            logger.debug(f"Qty Calc: Available Free Balance: {free_balance_usdt.normalize()} {CONFIG.usdt_symbol}")
        except Exception as bal_err:
            logger.error(
                f"{Fore.RED}Qty Calc: Failed to fetch free balance: {bal_err}. Cannot perform margin check.{Style.RESET_ALL}"
            )
            # Decide whether to stop or proceed without margin check - stopping is safer.
            return None

        # Check if free balance is sufficient with the configured buffer
        required_free_margin = estimated_margin_required * CONFIG.required_margin_buffer
        if free_balance_usdt < required_free_margin:
            logger.warning(
                f"{Fore.YELLOW}Qty Calc: Insufficient Free Margin. Need {required_free_margin.normalize()} {CONFIG.usdt_symbol} "
                f"(includes {CONFIG.required_margin_buffer}x buffer) but have {free_balance_usdt.normalize()} {CONFIG.usdt_symbol}. "
                f"Cannot place order of size {final_quantity_dec.normalize()}.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{CONFIG.strategy_name}] Insufficient Margin for {symbol}. Need {required_free_margin:.2f}, Have {free_balance_usdt:.2f}. Qty calc failed."
            )
            return None
        else:
            logger.debug(
                f"Qty Calc: Free margin ({free_balance_usdt.normalize()}) sufficient for estimated requirement ({required_free_margin.normalize()})."
            )

        # --- 6. Final Quantity Validation ---
        # Ensure the calculated quantity meets the minimum value requirement as well (qty * price)
        estimated_order_value_usdt = final_quantity_dec * current_price_dec
        if estimated_order_value_usdt < min_order_usdt_value:
            logger.warning(
                f"{Fore.YELLOW}Qty Calc: Calculated order value {estimated_order_value_usdt.normalize()} {CONFIG.usdt_symbol} is below market minimum value {min_order_usdt_value.normalize()} {CONFIG.usdt_symbol}. Cannot place order.{Style.RESET_ALL}"
            )
            return None

        logger.info(
            f"{Fore.GREEN}Qty Calc: Final Calculated Quantity: {final_quantity_dec.normalize()} {market.get('base')}. "
            f"Estimated Order Value: {estimated_order_value_usdt.normalize()} {CONFIG.usdt_symbol}.{Style.RESET_ALL}"
        )

        return final_quantity_dec.normalize()  # Return normalized Decimal quantity

    except Exception as e:
        logger.error(f"{Fore.RED}Qty Calc: Unexpected error during quantity calculation: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return None  # Return None on unexpected errors


def create_order(
    exchange: ccxt.Exchange,
    symbol: str,
    type: str,
    side: str,
    amount: Decimal,
    price: Decimal | None = None,
    stop_loss: Decimal | None = None,
    take_profit: Decimal | None = None,
    trailing_stop_percentage: Decimal | None = None,
) -> dict[str, Any] | None:
    """Places an order with native Stop Loss, Take Profit, and Trailing Stop Loss via Bybit V5 API.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        type: Order type ('market', 'limit', etc.). This bot primarily uses 'market'.
        side: Order side ('buy' or 'sell').
        amount: The quantity to trade (Decimal).
        price: The price for limit orders (Optional).
        stop_loss: Native Stop Loss trigger price (Optional, Decimal).
        take_profit: Native Take Profit trigger price (Optional, Decimal).
        trailing_stop_percentage: Native Trailing Stop percentage (Optional, Decimal, e.g., 0.005 for 0.5%).

    Returns:
        The CCXT order response dictionary if successful, None otherwise.
    """
    if amount <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Create Order: Cannot place order with zero or negligible amount ({amount}).{Style.RESET_ALL}"
        )
        return None

    formatted_amount = format_amount(exchange, symbol, amount)
    formatted_price = format_price(exchange, symbol, price) if price is not None else None
    formatted_stop_loss = (
        format_price(exchange, symbol, stop_loss)
        if stop_loss is not None and stop_loss > CONFIG.position_qty_epsilon
        else None
    )
    formatted_take_profit = (
        format_price(exchange, symbol, take_profit)
        if take_profit is not None and take_profit > CONFIG.position_qty_epsilon
        else None
    )

    # Bybit V5 requires 'category': 'linear' for perpetuals
    params: dict[str, Any] = {"category": "linear"}

    # Add Trailing Stop Loss percentage to params for Bybit V5
    if trailing_stop_percentage is not None and trailing_stop_percentage > CONFIG.position_qty_epsilon:
        # Bybit V5 TSL is often set as 'trailingStopP' in raw params, representing a percentage.
        # CCXT abstracts this via the standard 'trailingStop' parameter in createOrder,
        # which it then translates to the correct V5 param ('trailingStop', 'trailingStopP').
        # Let's try the standard CCXT 'trailingStop' parameter first, formatted as a percentage string if needed,
        # or pass 'trailingStopP' directly in params.
        # Bybit V5 'trailingStop' param in create-order is in USD units for inverse, or % for linear.
        # CCXT expects 'trailingStop' as a float price/percentage based on market type.
        # For LINEAR, it expects the percentage as float.
        params["trailingStop"] = (
            float(trailing_stop_percentage) * 100
        )  # Bybit V5 expects percentage * 100 for trailingStop (linear)
        # Bybit V5 also allows setting 'trailingStop' as an *absolute* offset if 'isLeverage': False,
        # but for leverage > 1, it must be a percentage. Our bot uses leverage > 1.
        # Let's double check Bybit V5 API docs and CCXT's Bybit implementation for exact TSL param name.
        # CCXT documentation for Bybit v5 `createOrder` suggests `trailingStop` param itself is used for this:
        # https://github.com/ccxt/ccxt/wiki/Manual#trailing-stop-orders
        # "Trailing Stop Orders: Some exchanges allow placing trailing stops with market or limit orders.
        # CCXT supports the `trailingStop` parameter for this purpose. The value usually represents
        # the distance from the current price in quote currency or a percentage."
        # For Bybit V5 LINEAR, 'trailingStop' in create-order params should be the percentage * 100.
        # Okay, setting `params['trailingStop'] = float(trailing_stop_percentage) * 100` is the correct approach for Bybit V5 linear.

        # Add TSL activation price offset if configured (Optional, Bybit V5 specific)
        # Bybit V5 allows a `activationPrice` or `activePrice` param.
        # The code has `trailing_stop_activation_offset_percent`.
        # Activation Price = Entry Price * (1 + offset_percent) for long
        # Activation Price = Entry Price * (1 - offset_percent) for short
        # This assumes the TSL only activates after being in profit by the offset percentage.
        # Let's calculate this price and add it as 'activationPrice' to params.
        if CONFIG.trailing_stop_activation_offset_percent > CONFIG.position_qty_epsilon and price is not None:
            offset_factor = Decimal("1") + CONFIG.trailing_stop_activation_offset_percent
            if side == CONFIG.side_sell:
                offset_factor = Decimal("1") - CONFIG.trailing_stop_activation_offset_percent
            activation_price = price * offset_factor
            params["activationPrice"] = format_price(exchange, symbol, activation_price)
            logger.debug(f"Create Order: Calculated TSL activation price: {params['activationPrice']}")
        else:
            logger.debug("Create Order: TSL activation offset zero or disabled.")

    logger.info(
        f"{Fore.YELLOW}Conjuring Order | Symbol: {symbol}, Type: {type}, Side: {side}, Amount: {formatted_amount}..."
    )
    if price is not None:
        logger.info(f"  Price: {formatted_price}")
    if formatted_stop_loss is not None:
        logger.info(f"  Native Stop Loss: {formatted_stop_loss}")
    if formatted_take_profit is not None:
        logger.info(f"  Native Take Profit: {formatted_take_profit}")
    if "trailingStop" in params:
        logger.info(
            f"  Native Trailing Stop %: {params['trailingStop'] / 100:.2%} (Activation Price: {params.get('activationPrice', 'Immediate')})"
        )

    try:
        # Place the order using CCXT's createOrder, including native SL/TP/TSL parameters
        # CCXT maps stopLoss and takeProfit directly for native Bybit V5 orders
        order = exchange.create_order(
            symbol=symbol,
            type=type,
            side=side,
            amount=float(amount),  # CCXT expects float for amount/price in the main call
            price=float(price) if price is not None else None,  # CCXT expects float
            params={
                **params,  # Include the category and TSL params
                "stopLoss": formatted_stop_loss,  # Pass SL price as string
                "takeProfit": formatted_take_profit,  # Pass TP price as string
                # Note: Bybit V5 requires stopLoss/takeProfit as *price* for Market orders.
                # The CCXT createOrder params map handles this translation.
            },
        )

        order_id = order.get("id")
        order_status = order.get("status")

        logger.info(
            f"{Fore.GREEN}Order Conjured! | ID: {format_order_id(order_id)}, Status: {order_status}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{CONFIG.strategy_name}/{symbol}] Order Placed (ID: {format_order_id(order_id)}, Side: {side}, Qty: {amount.normalize()}, Status: {order_status})"
        )

        # --- Wait for Market Order Fill (if applicable) ---
        # Market orders should fill quickly, but waiting and fetching details is crucial
        # to get the *actual* average entry price and confirm the order is closed.
        # For limit orders, this wait logic would need adjustment or removal.
        if type == "market":
            logger.debug(
                f"Waiting up to {CONFIG.order_fill_timeout_seconds}s for market order {format_order_id(order_id)} fill..."
            )
            filled_order = None
            try:
                # Use CCXT's fetchOrder or fetchOpenOrders and wait/poll
                # fetchOrder is better if we have the ID
                retries = 0
                while retries < CONFIG.order_fill_timeout_seconds / 2:  # Poll approx every 2 seconds
                    time.sleep(2)
                    fetched_order = exchange.fetch_order(
                        order_id, symbol, params={"category": params["category"]}
                    )  # Specify category for V5
                    if fetched_order and fetched_order.get("status") == "closed":
                        filled_order = fetched_order
                        logger.debug(f"Market order {format_order_id(order_id)} detected as 'closed'.")
                        break
                    retries += 1

                if filled_order:
                    filled_qty = safe_decimal_conversion(filled_order.get("filled", "0"))
                    avg_price = safe_decimal_conversion(filled_order.get("average", "0"))
                    if (
                        filled_qty >= amount * (Decimal("1") - CONFIG.position_qty_epsilon)
                        and avg_price > CONFIG.position_qty_epsilon
                    ):  # Check if filled significantly close to requested amount
                        logger.success(
                            f"{Fore.GREEN}Market order {format_order_id(order_id)} filled! Filled Qty: {filled_qty.normalize()}, Avg Price: {avg_price.normalize()}{Style.RESET_ALL}"
                        )
                        # Update the returned order object with potentially more accurate filled info
                        order["filled"] = filled_qty
                        order["average"] = avg_price
                        order["status"] = "closed"  # Ensure status is marked closed
                        return order  # Return the updated order dictionary
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Market order {format_order_id(order_id)} status is 'closed' but filled quantity ({filled_qty.normalize()}) is less than requested ({amount.normalize()}) or avg price zero. Potential partial fill or data issue.{Style.RESET_ALL}"
                        )
                        # Return the fetched order even if partial fill
                        return filled_order
                else:
                    logger.error(
                        f"{Fore.RED}Market order {format_order_id(order_id)} did not report 'closed' status after {CONFIG.order_fill_timeout_seconds}s. Potential issue.{Style.RESET_ALL}"
                    )
                    # Return the last fetched state, even if not closed
                    return fetched_order if "fetched_order" in locals() and fetched_order else order

            except Exception as fill_check_err:
                logger.error(
                    f"{Fore.RED}Error while waiting/checking market order fill for {format_order_id(order_id)}: {fill_check_err}{Style.RESET_ALL}"
                )
                logger.debug(traceback.format_exc())
                # Return the initial order response as fallback
                return order

        else:  # For non-market orders (limit etc.), just return the initial response
            return order

    except ccxt.InsufficientFunds as e:
        logger.error(f"{Fore.RED}Order Failed ({symbol}): Insufficient funds - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Insufficient Funds.")
    except ccxt.InvalidOrder as e:
        logger.error(
            f"{Fore.RED}Order Failed ({symbol}): Invalid order request - {e}. Check quantity precision, price, stop/TP params, min/max limits.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Invalid Request ({e}).")
    except ccxt.DDoSProtection as e:
        logger.warning(f"{Fore.YELLOW}Order Failed ({symbol}): Rate limit hit - {e}. Backing off.{Style.RESET_ALL}")
        time.sleep(exchange.rateLimit / 1000 + 1)  # Wait a bit longer than rate limit
    except ccxt.RequestTimeout as e:
        logger.warning(
            f"{Fore.YELLOW}Order Failed ({symbol}): Request timed out - {e}. Network issue or high load.{Style.RESET_ALL}"
        )
    except ccxt.NetworkError as e:
        logger.warning(f"{Fore.YELLOW}Order Failed ({symbol}): Network error - {e}. Check connection.{Style.RESET_ALL}")
    except ccxt.ExchangeError as e:
        logger.error(
            f"{Fore.RED}Order Failed ({symbol}): Exchange error - {e}. Check account status, symbol status, Bybit system status.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Exchange Error ({e}).")
    except Exception as e:
        logger.error(f"{Fore.RED}Order Failed ({symbol}): Unexpected error - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order FAILED: Unexpected Error: {type(e).__name__}.")

    return None  # Return None if order placement or fill confirmation failed


def confirm_stops_attached(
    exchange: ccxt.Exchange,
    symbol: str,
    expected_sl_price: Decimal | None,
    expected_tp_price: Decimal | None,
    expected_tsl_percentage: Decimal | None,
    attempts: int,
    delay: int,
) -> bool:
    """Fetches the current position multiple times to confirm native stops are attached.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        expected_sl_price: The SL price expected to be attached (Decimal) or None.
        expected_tp_price: The TP price expected to be attached (Decimal) or None.
        expected_tsl_percentage: The TSL percentage expected (Decimal) or None.
        attempts: Number of times to check.
        delay: Delay in seconds between checks.

    Returns:
        True if stops were confirmed attached (non-zero values returned by API)
        within the attempts, False otherwise.
        Note: This doesn't verify the *exact* price/percentage matches, only that
        non-zero values are present for SL/TP/TSL in the position details.
    """
    logger.debug(f"Confirming native stops attached to position ({symbol})...")
    sl_attached = (
        expected_sl_price is None or expected_sl_price <= CONFIG.position_qty_epsilon
    )  # Assume SL is attached if not requested
    tp_attached = (
        expected_tp_price is None or expected_tp_price <= CONFIG.position_qty_epsilon
    )  # Assume TP is attached if not requested
    tsl_attached = (
        expected_tsl_percentage is None or expected_tsl_percentage <= CONFIG.position_qty_epsilon
    )  # Assume TSL is attached if not requested

    if sl_attached and tp_attached and tsl_attached:
        logger.debug("No stops were requested, confirmation is trivially true.")
        return True  # No stops to confirm

    for attempt in range(1, attempts + 1):
        logger.debug(f"Confirm Stops: Attempt {attempt}/{attempts}...")
        position_state = get_current_position(exchange, symbol)

        if position_state["side"] == CONFIG.pos_none:
            logger.warning(
                f"{Fore.YELLOW}Confirm Stops: Position for {symbol} disappeared during confirmation check. Stops likely not attached.{Style.RESET_ALL}"
            )
            return False  # Position is gone

        # Check if fetched position details contain non-zero values for the requested stops
        current_sl = position_state.get("stop_loss")
        current_tp = position_state.get("take_profit")
        current_tsl_price = position_state.get("trailing_stop_price")  # V5 returns trigger price

        # Only update status if a stop was *requested* and is now seen as non-zero
        if not sl_attached and expected_sl_price is not None and expected_sl_price > CONFIG.position_qty_epsilon:
            if current_sl is not None and current_sl > CONFIG.position_qty_epsilon:
                sl_attached = True
                logger.debug(f"Confirm Stops: SL ({current_sl.normalize()}) confirmed attached.")
            elif current_sl is not None and current_sl <= CONFIG.position_qty_epsilon:
                # This case suggests the API returned SL=0 even if we requested it non-zero.
                logger.warning(
                    f"{Fore.YELLOW}Confirm Stops: API returned SL as zero ({current_sl}) despite non-zero request. Check Bybit logs/API behavior.{Style.RESET_ALL}"
                )

        if not tp_attached and expected_tp_price is not None and expected_tp_price > CONFIG.position_qty_epsilon:
            if current_tp is not None and current_tp > CONFIG.position_qty_epsilon:
                tp_attached = True
                logger.debug(f"Confirm Stops: TP ({current_tp.normalize()}) confirmed attached.")
            elif current_tp is not None and current_tp <= CONFIG.position_qty_epsilon:
                logger.warning(
                    f"{Fore.YELLOW}Confirm Stops: API returned TP as zero ({current_tp}) despite non-zero request. Check Bybit logs/API behavior.{Style.RESET_ALL}"
                )

        # For TSL, we check if the 'trailing_stop_price' field is non-zero, as Bybit V5 populates this
        # when TSL is active, showing the current trigger price.
        if (
            not tsl_attached
            and expected_tsl_percentage is not None
            and expected_tsl_percentage > CONFIG.position_qty_epsilon
        ):
            if current_tsl_price is not None and current_tsl_price > CONFIG.position_qty_epsilon:
                tsl_attached = True
                logger.debug(f"Confirm Stops: TSL (Trigger Price: {current_tsl_price.normalize()}) confirmed attached.")
            elif current_tsl_price is not None and current_tsl_price <= CONFIG.position_qty_epsilon:
                logger.warning(
                    f"{Fore.YELLOW}Confirm Stops: API returned TSL Trigger Price as zero ({current_tsl_price}) despite non-zero request. Check Bybit logs/API behavior.{Style.RESET_ALL}"
                )

        if sl_attached and tp_attached and tsl_attached:
            logger.success(
                f"{Fore.GREEN}Confirm Stops: All requested stops ({'SL' if expected_sl_price is not None and expected_sl_price > 0 else ''}{', ' if (expected_sl_price is not None and expected_sl_price > 0) and (expected_tp_price is not None and expected_tp_price > 0) else ''}{'TP' if expected_tp_price is not None and expected_tp_price > 0 else ''}{', ' if ((expected_sl_price is not None and expected_sl_price > 0) or (expected_tp_price is not None and expected_tp_price > 0)) and (expected_tsl_percentage is not None and expected_tsl_percentage > 0) else ''}{'TSL' if expected_tsl_percentage is not None and expected_tsl_percentage > 0 else ''}) confirmed attached!{Style.RESET_ALL}"
            )
            return True

        # If not all requested stops are attached, wait and try again (unless this is the last attempt)
        if attempt < attempts:
            time.sleep(delay)

    # If loop finishes and not all stops are confirmed attached
    logger.error(
        f"{Fore.RED}Confirm Stops: Failed to confirm all requested stops attached after {attempts} attempts.{Style.RESET_ALL}"
    )
    missing_stops = []
    if expected_sl_price is not None and expected_sl_price > 0 and not sl_attached:
        missing_stops.append("SL")
    if expected_tp_price is not None and expected_tp_price > 0 and not tp_attached:
        missing_stops.append("TP")
    # TSL is slightly harder to confirm by percentage after the fact, but checking the trigger price is a proxy
    if expected_tsl_percentage is not None and expected_tsl_percentage > 0 and not tsl_attached:
        missing_stops.append("TSL")

    if missing_stops:
        logger.error(f"{Fore.RED}Confirm Stops: Missing stops: {', '.join(missing_stops)}{Style.RESET_ALL}")
        send_sms_alert(
            f"[{CONFIG.strategy_name}/{symbol}] WARNING: Failed to confirm {', '.join(missing_stops)} attached after entry."
        )

    return False


def close_position(exchange: ccxt.Exchange, symbol: str, current_position: dict[str, Any]) -> bool:
    """Closes the current active position using a market order.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        current_position: Dictionary containing details of the current position.

    Returns:
        True if the close order is successfully placed (and assumed filled as market),
        False otherwise.
    """
    pos_side = current_position.get("side")
    pos_qty = safe_decimal_conversion(current_position.get("qty"))

    if pos_side == CONFIG.pos_none or pos_qty <= CONFIG.position_qty_epsilon:
        logger.info("Close Position: No active position to close.")
        return True  # Already flat

    close_side = CONFIG.side_sell if pos_side == CONFIG.pos_long else CONFIG.side_buy
    logger.warning(
        f"{Fore.YELLOW}Initiating Position Closure: Closing {pos_side} position for {symbol} (Qty: {pos_qty.normalize()}) with market order ({close_side})...{Style.RESET_ALL}"
    )

    try:
        # Bybit V5 closing a position uses the *opposite* side Market order.
        # Quantity should be the full position quantity.
        # CCXT's createOrder with 'reduceOnly': True is the standard way.
        # Bybit V5 also supports 'positionIdx' in params to specify which position to close in Hedge mode,
        # but for One-Way mode (positionIdx=0), it's usually not strictly necessary if using reduceOnly.
        # Let's add 'positionIdx': 0 to be explicit for One-Way.
        params = {"category": "linear", "reduceOnly": True, "positionIdx": 0}  # Explicit for One-Way

        # Use the exact position quantity
        formatted_qty = format_amount(exchange, symbol, pos_qty)

        logger.debug(
            f"Placing market order to close position. Symbol: {symbol}, Side: {close_side}, Quantity: {formatted_qty}, Params: {params}"
        )

        close_order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=close_side,
            amount=float(pos_qty),  # CCXT expects float
            params=params,
        )

        order_id = close_order.get("id")
        order_status = close_order.get("status")

        logger.info(
            f"{Fore.GREEN}Position Close Order Conjured! | ID: {format_order_id(order_id)}, Status: {order_status}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{CONFIG.strategy_name}/{symbol}] Closing {pos_side} position. Qty: {pos_qty.normalize()}. Order ID: {format_order_id(order_id)}. Status: {order_status}."
        )

        # For Market Close orders, wait for confirmation of fill
        logger.debug(
            f"Waiting up to {CONFIG.order_fill_timeout_seconds}s for close market order {format_order_id(order_id)} fill..."
        )
        filled_close_order = None
        try:
            retries = 0
            while retries < CONFIG.order_fill_timeout_seconds / 2:  # Poll approx every 2 seconds
                time.sleep(2)
                # Use fetchOrder with category and positionIdx if needed for this specific order on Bybit V5
                fetched_order = exchange.fetch_order(order_id, symbol, params={"category": params["category"]})
                if fetched_order and fetched_order.get("status") == "closed":
                    filled_close_order = fetched_order
                    logger.debug(f"Close market order {format_order_id(order_id)} detected as 'closed'.")
                    break
                retries += 1

            if filled_close_order:
                filled_qty = safe_decimal_conversion(filled_close_order.get("filled", "0"))
                avg_price = safe_decimal_conversion(filled_close_order.get("average", "0"))
                if (
                    filled_qty >= pos_qty * (Decimal("1") - CONFIG.position_qty_epsilon)
                    and avg_price > CONFIG.position_qty_epsilon
                ):  # Check if filled close to expected qty
                    logger.success(
                        f"{Fore.GREEN}Position close order {format_order_id(order_id)} filled! Filled Qty: {filled_qty.normalize()}, Avg Price: {avg_price.normalize()}{Style.RESET_ALL}"
                    )
                    # Give the exchange a moment to update the position state after fill
                    time.sleep(CONFIG.post_close_delay_seconds)
                    return True  # Successfully placed and filled close order
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Position close order {format_order_id(order_id)} status is 'closed' but filled quantity ({filled_qty.normalize()}) is less than expected ({pos_qty.normalize()}) or avg price zero. Position may not be fully closed.{Style.RESET_ALL}"
                    )
                    # It might still be partially closed, return True to avoid infinite loop, but log warning.
                    time.sleep(CONFIG.post_close_delay_seconds)
                    return True
            else:
                logger.error(
                    f"{Fore.RED}Position close market order {format_order_id(order_id)} did not report 'closed' status after {CONFIG.order_fill_timeout_seconds}s. Position may not be closed.{Style.RESET_ALL}"
                )
                # Decide how to handle - returning False might trigger another close attempt if position still exists.
                # Let's re-check position state immediately after timeout
                time.sleep(CONFIG.post_close_delay_seconds)  # Small delay then check
                post_close_pos = get_current_position(exchange, symbol)
                if post_close_pos["side"] == CONFIG.pos_none or post_close_pos["qty"] <= CONFIG.position_qty_epsilon:
                    logger.success(f"{Fore.GREEN}Position confirmed closed after fill check timeout.{Style.RESET_ALL}")
                    return True  # Position is indeed closed
                else:
                    logger.error(
                        f"{Fore.RED}Position still active after fill check timeout. Quantity remaining: {post_close_pos['qty'].normalize()}.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{CONFIG.strategy_name}/{symbol}] Close order timed out, pos still active. Qty: {post_close_pos['qty'].normalize()}."
                    )
                    return False  # Position still active, indicate failure
        except Exception as fill_check_err:
            logger.error(
                f"{Fore.RED}Error while waiting/checking close market order fill for {format_order_id(order_id)}: {fill_check_err}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            # Assume failure if check fails, let the main loop potentially retry closing
            return False

    except ccxt.InsufficientFunds as e:
        # This can happen if trying to close with wrong side/params, or margin call issues
        logger.error(
            f"{Fore.RED}Close Order Failed ({symbol}): Insufficient funds (during close?) - {e}. Check position, margin, order params.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Insufficient Funds.")
    except ccxt.InvalidOrder as e:
        logger.error(
            f"{Fore.RED}Close Order Failed ({symbol}): Invalid order request - {e}. Check quantity, side, reduceOnly param, positionIdx.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Invalid Request ({e}).")
    except ccxt.DDoSProtection as e:
        logger.warning(
            f"{Fore.YELLOW}Close Order Failed ({symbol}): Rate limit hit - {e}. Backing off.{Style.RESET_ALL}"
        )
        time.sleep(exchange.rateLimit / 1000 + 1)
    except ccxt.RequestTimeout as e:
        logger.warning(
            f"{Fore.YELLOW}Close Order Failed ({symbol}): Request timed out - {e}. Network issue.{Style.RESET_ALL}"
        )
    except ccxt.NetworkError as e:
        logger.warning(
            f"{Fore.YELLOW}Close Order Failed ({symbol}): Network error - {e}. Check connection.{Style.RESET_ALL}"
        )
    except ccxt.ExchangeError as e:
        # Bybit error like "Order quantity below minimum" or "Position size is 0" can happen if already closed
        logger.error(
            f"{Fore.RED}Close Order Failed ({symbol}): Exchange error - {e}. Position might already be closed or another issue.{Style.RESET_ALL}"
        )
        # Re-check position immediately if exchange error might indicate already closed state
        time.sleep(CONFIG.post_close_delay_seconds)
        post_close_pos = get_current_position(exchange, symbol)
        if post_close_pos["side"] == CONFIG.pos_none or post_close_pos["qty"] <= CONFIG.position_qty_epsilon:
            logger.success(
                f"{Fore.GREEN}Position confirmed closed after ExchangeError during close attempt.{Style.RESET_ALL}"
            )
            return True  # Position is indeed closed
        else:
            send_sms_alert(
                f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Exchange Error ({e}). Pos still active Qty: {post_close_pos['qty'].normalize()}."
            )
            return False
    except Exception as e:
        logger.error(f"{Fore.RED}Close Order Failed ({symbol}): Unexpected error - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Close Order FAILED: Unexpected Error: {type(e).__name__}.")

    return False  # Return False if order placement or confirmation failed


def cancel_all_orders_for_symbol(exchange: ccxt.Exchange, symbol: str, reason: str) -> int:
    """Attempts to cancel all open orders for a specific symbol.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The market symbol.
        reason: A string indicating why cancellation is being attempted (for logging).

    Returns:
        The number of cancellation attempts made (not necessarily successful cancellations).
    """
    logger.info(f"{Fore.BLUE}Order Cleanup Ritual: Initiating for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    attempts = 0
    try:
        # Bybit V5 cancelAllOrders requires 'category' parameter for futures
        # It also *strongly* prefers the exchange-specific market ID.
        market = exchange.market(symbol)  # Market details already loaded
        market["id"]
        category = "linear"  # Assuming linear based on bot purpose

        # CCXT's cancel_all_orders method
        logger.warning(
            f"{Fore.YELLOW}Order Cleanup: Attempting to cancel ALL open orders for {symbol} (Category: {category})...{Style.RESET_ALL}"
        )
        attempts += 1
        # Bybit V5 cancelAllOrders takes symbol and category in params
        response = exchange.cancel_all_orders(symbol=symbol, params={"category": category})

        # Bybit V5 cancelAllOrders response structure can vary, often empty or confirms actions.
        # Success is generally indicated by no exception and a non-error response structure.
        logger.info(
            f"{Fore.GREEN}Order Cleanup: cancel_all_orders request sent for {symbol}. Response: {response}{Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.GREEN}Order Cleanup Ritual Finished for {symbol}. Attempt successful (reported {attempts} actions/attempts).{Style.RESET_ALL}"
        )
        return attempts  # Report attempts made

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logger.warning(
            f"{Fore.YELLOW}Order Cleanup Error for {symbol}: {type(e).__name__} - {e}. Could not cancel orders.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order Cancel FAILED: {type(e).__name__}.")
    except ccxt.NotSupported:
        logger.error(
            f"{Fore.RED}Order Cleanup Error: Exchange '{exchange.id}' does not support cancelAllOrders method. Cannot perform cleanup.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Order Cleanup Unexpected Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{CONFIG.strategy_name}/{symbol}] Order Cancel FAILED: Unexpected Error: {type(e).__name__}.")

    return attempts  # Return attempts even on failure


# --- Strategy Signal Generation - The Oracle's Prophecy ---
def generate_trading_signal(
    df: pd.DataFrame, current_position: dict[str, Any], order_book_data: dict[str, Decimal | None]
) -> str | None:
    """Analyzes indicator data based on the selected strategy and generates a trade signal.

    Args:
        df: DataFrame with OHLCV and calculated indicator columns.
        current_position: Dictionary with current position details.
        order_book_data: Dictionary with order book analysis results.

    Returns:
        'long', 'short', 'close_long', 'close_short', or None for no signal.
    """
    last_candle = df.iloc[-1]
    position_side = current_position["side"]
    current_position["qty"]
    last_price = safe_decimal_conversion(last_candle["close"])

    # Check for sufficient data and valid last candle
    if (
        len(df)
        < max(
            CONFIG.st_atr_length,
            CONFIG.confirm_st_atr_length,
            CONFIG.stochrsi_stoch_length,
            CONFIG.momentum_length,
            CONFIG.ehlers_fisher_length,
            CONFIG.ema_slow_period,
        )
        + CONFIG.api_fetch_limit_buffer
    ):
        logger.warning(
            f"{Fore.YELLOW}Signal Gen: Insufficient data ({len(df)} candles) for indicators or buffer. Cannot generate signal.{Style.RESET_ALL}"
        )
        return None

    if last_price <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Signal Gen: Last price is zero or invalid ({last_price}). Cannot generate signal.{Style.RESET_ALL}"
        )
        return None

    # --- Signal Generation Logic based on Strategy ---
    entry_signal: str | None = None  # 'long' or 'short'
    exit_signal: str | None = None  # 'close_long' or 'close_short'

    if CONFIG.strategy_name == "DUAL_SUPERTREND":
        # Requires 'supertrend', 'trend', 'st_long', 'st_short' from calculate_supertrend
        # Requires 'confirm_supertrend', 'confirm_trend' from calculate_supertrend (with prefix)
        primary_flip_long = last_candle.get("st_long", pd.NA)
        primary_flip_short = last_candle.get("st_short", pd.NA)
        confirm_is_uptrend = last_candle.get("confirm_trend", pd.NA)
        confirm_st_val = last_candle.get("confirm_supertrend", pd.NA)

        # Check if required indicator values are available for the last candle
        if (
            pd.isna(primary_flip_long)
            or pd.isna(primary_flip_short)
            or pd.isna(confirm_is_uptrend)
            or pd.isna(confirm_st_val)
        ):
            missing = []
            if pd.isna(primary_flip_long):
                missing.append("PrimaryFlipL")
            if pd.isna(primary_flip_short):
                missing.append("PrimaryFlipS")
            if pd.isna(confirm_is_uptrend):
                missing.append("ConfirmUp")
            if pd.isna(confirm_st_val):
                missing.append("ConfirmSTVal")
            logger.warning(
                f"{Fore.YELLOW}Signal Gen (DUAL_SUPERTREND): Skipping due to missing indicator values ({', '.join(missing)}).{Style.RESET_ALL}"
            )
            return None  # Cannot generate signal if indicators are missing

        # Entry Signal: Primary ST flips AND Confirmation ST is in agreement
        # Long Entry: Primary ST flips long AND Confirmation ST is in uptrend AND price is above Confirmation ST
        if primary_flip_long and confirm_is_uptrend and last_price > safe_decimal_conversion(confirm_st_val):
            entry_signal = CONFIG.side_buy
            logger.debug(
                f"DUAL_ST: Primary ST flipped Long, Confirm ST is Up, Price ({last_price:.4f}) > Confirm ST ({safe_decimal_conversion(confirm_st_val):.4f}). Long entry signal generated."
            )

        # Short Entry: Primary ST flips short AND Confirmation ST is in downtrend AND price is below Confirmation ST
        elif primary_flip_short and not confirm_is_uptrend and last_price < safe_decimal_conversion(confirm_st_val):
            entry_signal = CONFIG.side_sell
            logger.debug(
                f"DUAL_ST: Primary ST flipped Short, Confirm ST is Down, Price ({last_price:.4f}) < Confirm ST ({safe_decimal_conversion(confirm_st_val):.4f}). Short entry signal generated."
            )

        # Exit Signal: Primary ST flips against the current position
        primary_is_uptrend = last_candle.get("trend", pd.NA)  # Get primary trend for exit check
        primary_st_val = last_candle.get("supertrend", pd.NA)  # Get primary ST value for exit check

        if pd.notna(primary_is_uptrend) and pd.notna(primary_st_val):
            # Close Long: Have Long position AND Primary ST flips short OR price crosses below Primary ST
            if position_side == CONFIG.pos_long and (
                primary_flip_short or last_price < safe_decimal_conversion(primary_st_val)
            ):
                exit_signal = "close_long"
                if primary_flip_short:
                    logger.debug("DUAL_ST: Primary ST flipped Short. Close Long signal generated.")
                else:
                    logger.debug(
                        f"DUAL_ST: Price ({last_price:.4f}) crossed below Primary ST ({safe_decimal_conversion(primary_st_val):.4f}). Close Long signal generated."
                    )

            # Close Short: Have Short position AND Primary ST flips long OR price crosses above Primary ST
            elif position_side == CONFIG.pos_short and (
                primary_flip_long or last_price > safe_decimal_conversion(primary_st_val)
            ):
                exit_signal = "close_short"
                if primary_flip_long:
                    logger.debug("DUAL_ST: Primary ST flipped Long. Close Short signal generated.")
                else:
                    logger.debug(
                        f"DUAL_ST: Price ({last_price:.4f}) crossed above Primary ST ({safe_decimal_conversion(primary_st_val):.4f}). Close Short signal generated."
                    )
        else:
            logger.warning(
                f"{Fore.YELLOW}Signal Gen (DUAL_SUPERTREND Exit): Primary ST values missing ({pd.isna(primary_is_uptrend)=}, {pd.isna(primary_st_val)=}). Cannot check exit conditions.{Style.RESET_ALL}"
            )

    elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
        # Requires 'stochrsi_k', 'stochrsi_d', 'momentum' from calculate_stochrsi_momentum
        stoch_k = last_candle.get("stochrsi_k", pd.NA)
        stoch_d = last_candle.get("stochrsi_d", pd.NA)
        momentum = last_candle.get("momentum", pd.NA)

        # Check for missing indicators
        if pd.isna(stoch_k) or pd.isna(stoch_d) or pd.isna(momentum):
            missing = []
            if pd.isna(stoch_k):
                missing.append("StochK")
            if pd.isna(stoch_d):
                missing.append("StochD")
            if pd.isna(momentum):
                missing.append("Momentum")
            logger.warning(
                f"{Fore.YELLOW}Signal Gen (STOCHRSI_MOMENTUM): Skipping due to missing indicator values ({', '.join(missing)}).{Style.RESET_ALL}"
            )
            return None

        # Entry Signal:
        # Long Entry: Stoch %K crosses above %D AND both are below oversold threshold AND Momentum is positive
        if (
            stoch_k > stoch_d
            and stoch_k.shift(1, fill_value=stoch_k) <= stoch_d.shift(1, fill_value=stoch_d)
            and stoch_k < CONFIG.stochrsi_oversold
            and stoch_d < CONFIG.stochrsi_oversold
            and momentum > CONFIG.position_qty_epsilon
        ):
            entry_signal = CONFIG.side_buy
            logger.debug(
                f"STOCHRSI_MOM: K ({stoch_k:.2f}) crossed above D ({stoch_d:.2f}), both below Oversold ({CONFIG.stochrsi_oversold}), Momentum ({momentum:.4f}) positive. Long entry signal generated."
            )

        # Short Entry: Stoch %K crosses below %D AND both are above overbought threshold AND Momentum is negative
        elif (
            stoch_k < stoch_d
            and stoch_k.shift(1, fill_value=stoch_k) >= stoch_d.shift(1, fill_value=stoch_d)
            and stoch_k > CONFIG.stochrsi_overbought
            and stoch_d > CONFIG.stochrsi_overbought
            and momentum < -CONFIG.position_qty_epsilon
        ):
            entry_signal = CONFIG.side_sell
            logger.debug(
                f"STOCHRSI_MOM: K ({stoch_k:.2f}) crossed below D ({stoch_d:.2f}), both above Overbought ({CONFIG.stochrsi_overbought}), Momentum ({momentum:.4f}) negative. Short entry signal generated."
            )

        # Exit Signal (example: exit on momentum reversal or STOCHRSI extreme crossover):
        # Close Long: Have Long position AND (Momentum turns negative OR Stoch %K crosses below %D above overbought)
        if position_side == CONFIG.pos_long and (
            momentum < -CONFIG.position_qty_epsilon
            or (
                stoch_k < stoch_d
                and stoch_k.shift(1, fill_value=stoch_k) >= stoch_d.shift(1, fill_value=stoch_d)
                and stoch_k > CONFIG.stochrsi_overbought
            )
        ):
            exit_signal = "close_long"
            if momentum < -CONFIG.position_qty_epsilon:
                logger.debug(f"STOCHRSI_MOM: Momentum ({momentum:.4f}) turned negative. Close Long signal generated.")
            else:
                logger.debug(
                    f"STOCHRSI_MOM: K ({stoch_k:.2f}) crossed below D ({stoch_d:.2f}) above Overbought ({CONFIG.stochrsi_overbought}). Close Long signal generated."
                )

        # Close Short: Have Short position AND (Momentum turns positive OR Stoch %K crosses above %D below oversold)
        elif position_side == CONFIG.pos_short and (
            momentum > CONFIG.position_qty_epsilon
            or (
                stoch_k > stoch_d
                and stoch_k.shift(1, fill_value=stoch_k) <= stoch_d.shift(1, fill_value=stoch_d)
                and stoch_k < CONFIG.stochrsi_oversold
            )
        ):
            exit_signal = "close_short"
            if momentum > CONFIG.position_qty_epsilon:
                logger.debug(f"STOCHRSI_MOM: Momentum ({momentum:.4f}) turned positive. Close Short signal generated.")
            else:
                logger.debug(
                    f"STOCHRSI_MOM: K ({stoch_k:.2f}) crossed above D ({stoch_d:.2f}) below Oversold ({CONFIG.stochrsi_oversold}). Close Short signal generated."
                )

    elif CONFIG.strategy_name == "EHLERS_FISHER":
        # Requires 'ehlers_fisher', 'ehlers_signal' from calculate_ehlers_fisher
        fisher = last_candle.get("ehlers_fisher", pd.NA)
        signal = last_candle.get("ehlers_signal", pd.NA)
        prev_fisher = df["ehlers_fisher"].iloc[-2] if len(df) >= 2 else pd.NA
        prev_signal = df["ehlers_signal"].iloc[-2] if len(df) >= 2 else pd.NA

        # Check for missing indicators
        if pd.isna(fisher) or pd.isna(signal) or pd.isna(prev_fisher) or pd.isna(prev_signal):
            missing = []
            if pd.isna(fisher):
                missing.append("Fisher")
            if pd.isna(signal):
                missing.append("Signal")
            if pd.isna(prev_fisher):
                missing.append("PrevFisher")
            if pd.isna(prev_signal):
                missing.append("PrevSignal")
            logger.warning(
                f"{Fore.YELLOW}Signal Gen (EHLERS_FISHER): Skipping due to missing indicator values ({', '.join(missing)}).{Style.RESET_ALL}"
            )
            return None

        # Entry Signal: Fisher crosses above Signal
        # Long Entry: Fisher crosses above Signal line
        if fisher > signal and prev_fisher <= prev_signal:
            entry_signal = CONFIG.side_buy
            logger.debug(
                f"EHLERS_FISHER: Fisher ({fisher:.4f}) crossed above Signal ({signal:.4f}). Long entry signal generated."
            )

        # Short Entry: Fisher crosses below Signal
        elif fisher < signal and prev_fisher >= prev_signal:
            entry_signal = CONFIG.side_sell
            logger.debug(
                f"EHLERS_FISHER: Fisher ({fisher:.4f}) crossed below Signal ({signal:.4f}). Short entry signal generated."
            )

        # Exit Signal (example: exit on Fisher turning back towards zero or crossing signal again)
        # A simple exit could be when Fisher crosses the Signal line *against* the position direction,
        # or when Fisher moves back across zero. Let's use crossing the signal line for exit.
        # Close Long: Have Long position AND Fisher crosses below Signal
        if position_side == CONFIG.pos_long and fisher < signal and prev_fisher >= prev_signal:
            exit_signal = "close_long"
            logger.debug(
                f"EHLERS_FISHER: Fisher ({fisher:.4f}) crossed below Signal ({signal:.4f}). Close Long signal generated."
            )

        # Close Short: Have Short position AND Fisher crosses above Signal
        elif position_side == CONFIG.pos_short and fisher > signal and prev_fisher <= prev_signal:
            exit_signal = "close_short"
            logger.debug(
                f"EHLERS_FISHER: Fisher ({fisher:.4f}) crossed above Signal ({signal:.4f}). Close Short signal generated."
            )

    elif CONFIG.strategy_name == "EMA_CROSS":
        # Requires 'fast_ema', 'slow_ema' from calculate_ema_cross
        # REMINDER: This uses standard EMA, NOT Ehlers Super Smoother.
        fast_ema = last_candle.get("fast_ema", pd.NA)
        slow_ema = last_candle.get("slow_ema", pd.NA)
        prev_fast_ema = df["fast_ema"].iloc[-2] if len(df) >= 2 else pd.NA
        prev_slow_ema = df["slow_ema"].iloc[-2] if len(df) >= 2 else pd.NA

        # Check for missing indicators
        if pd.isna(fast_ema) or pd.isna(slow_ema) or pd.isna(prev_fast_ema) or pd.isna(prev_slow_ema):
            missing = []
            if pd.isna(fast_ema):
                missing.append("FastEMA")
            if pd.isna(slow_ema):
                missing.append("SlowEMA")
            if pd.isna(prev_fast_ema):
                missing.append("PrevFastEMA")
            if pd.isna(prev_slow_ema):
                missing.append("PrevSlowEMA")
            logger.warning(
                f"{Fore.YELLOW}Signal Gen (EMA_CROSS): Skipping due to missing indicator values ({', '.join(missing)}).{Style.RESET_ALL}"
            )
            return None

        # Entry Signal: Fast EMA crosses above Slow EMA
        # Long Entry: Fast EMA crosses above Slow EMA
        if fast_ema > slow_ema and prev_fast_ema <= prev_slow_ema:
            entry_signal = CONFIG.side_buy
            logger.debug(
                f"EMA_CROSS: Fast EMA ({fast_ema:.4f}) crossed above Slow EMA ({slow_ema:.4f}). Long entry signal generated."
            )

        # Short Entry: Fast EMA crosses below Slow EMA
        elif fast_ema < slow_ema and prev_fast_ema >= prev_slow_ema:
            entry_signal = CONFIG.side_sell
            logger.debug(
                f"EMA_CROSS: Fast EMA ({fast_ema:.4f}) crossed below Slow EMA ({slow_ema:.4f}). Short entry signal generated."
            )

        # Exit Signal: Fast EMA crosses back over Slow EMA
        # Close Long: Have Long position AND Fast EMA crosses below Slow EMA
        if position_side == CONFIG.pos_long and fast_ema < slow_ema and prev_fast_ema >= prev_slow_ema:
            exit_signal = "close_long"
            logger.debug(
                f"EMA_CROSS: Fast EMA ({fast_ema:.4f}) crossed below Slow EMA ({slow_ema:.4f}). Close Long signal generated."
            )

        # Close Short: Have Short position AND Fast EMA crosses above Slow EMA
        elif position_side == CONFIG.pos_short and fast_ema > slow_ema and prev_fast_ema <= prev_slow_ema:
            exit_signal = "close_short"
            logger.debug(
                f"EMA_CROSS: Fast EMA ({fast_ema:.4f}) crossed above Slow EMA ({slow_ema:.4f}). Close Short signal generated."
            )

    else:
        # This case should not be reached due to config validation, but handle defensively
        logger.error(
            f"{Fore.RED}Signal Gen: Unknown strategy name '{CONFIG.strategy_name}'. No signal generated.{Style.RESET_ALL}"
        )
        return None  # No signal for unknown strategy

    # --- Apply Exit Signal Priority ---
    # If an exit signal is generated, it overrides any potential entry signal
    if exit_signal:
        # Only return exit signal if currently in a position that matches the signal
        if (exit_signal == "close_long" and position_side == CONFIG.pos_long) or (
            exit_signal == "close_short" and position_side == CONFIG.pos_short
        ):
            logger.info(
                f"{Fore.YELLOW}Signal Gen: Exit signal generated ({exit_signal}). Prioritizing exit.{Style.RESET_ALL}"
            )
            return exit_signal
        else:
            # This shouldn't happen with correct strategy logic (exit signal for wrong position type)
            logger.warning(
                f"{Fore.YELLOW}Signal Gen: Ignoring irrelevant exit signal ({exit_signal}) for current {position_side} position.{Style.RESET_ALL}"
            )
            exit_signal = None  # Ignore the signal

    # --- Apply Entry Signal with Filters ---
    # Only consider entry if currently flat
    if position_side == CONFIG.pos_none and entry_signal:
        logger.debug(f"Signal Gen: Potential {entry_signal} entry signal generated by strategy. Checking filters...")
        if check_entry_filters(df, entry_signal, order_book_data):
            logger.info(
                f"{Fore.GREEN}Signal Gen: {entry_signal.capitalize()} entry signal confirmed by filters!{Style.RESET_ALL}"
            )
            return entry_signal
        else:
            logger.info(
                f"{Fore.YELLOW}Signal Gen: {entry_signal.capitalize()} entry signal rejected by filters.{Style.RESET_ALL}"
            )
            return None  # Filters failed, no entry signal

    # If no signal generated or filters failed, return None
    return None


def check_entry_filters(df: pd.DataFrame, signal_side: str, order_book_data: dict[str, Decimal | None]) -> bool:
    """Applies configured entry filters (Volume, Order Book) to validate a signal.

    Args:
        df: DataFrame with calculated indicator columns including volume analysis.
        signal_side: The potential entry side ('buy' or 'sell').
        order_book_data: Dictionary with order book analysis results.

    Returns:
        True if all required filters pass or are not enabled, False otherwise.
    """
    df.iloc[-1]

    # --- Volume Spike Filter ---
    if CONFIG.require_volume_spike_for_entry:
        volume_ratio = order_book_data.get("volume_ratio")  # Volume ratio is calculated with ATR/Vol analysis
        if volume_ratio is None:
            logger.debug("Filters: Volume spike required, but volume ratio is NA. Filter FAIL.")
            return False  # Cannot check if data is missing

        volume_spike_threshold = CONFIG.volume_spike_threshold
        volume_spike_detected = volume_ratio >= volume_spike_threshold

        if not volume_spike_detected:
            logger.debug(
                f"Filters: Volume spike required ({volume_ratio:.2f} < {volume_spike_threshold}). Filter FAIL."
            )
            return False
        else:
            logger.debug(
                f"Filters: Volume spike required ({volume_ratio:.2f} >= {volume_spike_threshold}). Filter PASS."
            )

    # --- Order Book Pressure Filter ---
    # Check Order Book ratio only if fetched (either per cycle or just now)
    order_book_ratio = order_book_data.get("bid_ask_ratio")
    if (
        CONFIG.order_book_ratio_threshold_long > CONFIG.position_qty_epsilon
        or CONFIG.order_book_ratio_threshold_short > CONFIG.position_qty_epsilon
    ):
        if order_book_ratio is None:
            logger.debug("Filters: Order Book filter enabled, but ratio is NA. Filter FAIL.")
            return False  # Cannot check if data is missing

        if signal_side == CONFIG.side_buy:
            if order_book_ratio < CONFIG.order_book_ratio_threshold_long:
                logger.debug(
                    f"Filters: Long OB ratio required ({order_book_ratio:.3f} < {CONFIG.order_book_ratio_threshold_long}). Filter FAIL."
                )
                return False
            else:
                logger.debug(
                    f"Filters: Long OB ratio required ({order_book_ratio:.3f} >= {CONFIG.order_book_ratio_threshold_long}). Filter PASS."
                )
        elif signal_side == CONFIG.side_sell:
            # For short, we need more ask volume, meaning the ratio (Bid/Ask) should be *below* the threshold.
            if order_book_ratio > CONFIG.order_book_ratio_threshold_short:
                logger.debug(
                    f"Filters: Short OB ratio required ({order_book_ratio:.3f} > {CONFIG.order_book_ratio_threshold_short}). Filter FAIL."
                )
                return False
            else:
                logger.debug(
                    f"Filters: Short OB ratio required ({order_book_ratio:.3f} <= {CONFIG.order_book_ratio_threshold_short}). Filter PASS."
                )

    # If all enabled filters passed or no filters are enabled
    logger.debug("Filters: All enabled filters passed.")
    return True


# --- Main Trading Logic - The Core Spell Loop ---
def main_trade_logic(exchange: ccxt.Exchange) -> None:
    """The main trading loop that fetches data, calculates indicators,
    generates signals, and executes trades based on the chosen strategy.
    """
    logger.info(
        f"{Fore.BLUE}--- Pyrmethus Bybit Scalping Spell v2.3.0 Initializing ({time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{Style.RESET_ALL}"
    )
    logger.info(f"{Fore.BLUE}--- Strategy Enchantment Selected: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(
        f"{Fore.BLUE}--- Protective Wards Activated: Initial ATR-Stop, ATR-TakeProfit + Exchange Trailing Stop (Bybit V5 Native) ---{Style.RESET_ALL}"
    )

    # Focus the spell on the symbol (check market exists, get its properties)
    try:
        logger.info(f"Attempting to focus spell on symbol: {CONFIG.symbol}")
        market = exchange.market(CONFIG.symbol)
        logger.info(
            f"Market Details: Type={market.get('type')}, Contract={market.get('contract')}, Linear={market.get('linear')}, Inverse={market.get('inverse')}"
        )
        if not market.get("spot") and not market.get("margin") and not market.get("futures") and not market.get("swap"):
            logger.warning(
                f"{Fore.YELLOW}Warning: Symbol '{CONFIG.symbol}' market type is unusual ({market.get('type')}). Ensure it is a supported futures/swap contract.{Style.RESET_ALL}"
            )
        if not market.get("linear"):
            logger.warning(
                f"{Fore.YELLOW}Warning: Symbol '{CONFIG.symbol}' is not identified as a linear (USDT) contract. Ensure CONFIG.symbol is correct.{Style.RESET_ALL}"
            )

        # Set leverage (attempt retry)
        leverage_set = False
        for attempt in range(CONFIG.retry_count):
            try:
                logger.info(
                    f"{Fore.YELLOW}Leverage Conjuring: Attempt {attempt + 1}/{CONFIG.retry_count}: Attempting to set {CONFIG.leverage}x for {CONFIG.symbol}...{Style.RESET_ALL}"
                )
                # CCXT set_leverage for Bybit V5 requires 'category' and 'symbol' in params
                exchange.set_leverage(CONFIG.leverage, CONFIG.symbol, params={"category": "linear"})
                logger.info(
                    f"{Fore.GREEN}Leverage Conjuring: Leverage set to {CONFIG.leverage}x for {CONFIG.symbol}.{Style.RESET_ALL}"
                )
                leverage_set = True
                break  # Exit retry loop on success
            except ccxt.NotSupported:
                logger.error(
                    f"{Fore.RED}Leverage Conjuring Failed: Exchange or symbol does not support setting leverage via this method.{Style.RESET_ALL}"
                )
                leverage_set = False
                break  # No point retrying if not supported
            except ccxt.ExchangeError as e:
                if "leverage setting is not modified" in str(e):
                    logger.info(
                        f"{Fore.GREEN}Leverage Conjuring: Leverage already set to {CONFIG.leverage}x for {CONFIG.symbol} (or not modified).{Style.RESET_ALL}"
                    )
                    leverage_set = True
                    break  # Already set, no need to retry
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Leverage Conjuring Failed (Attempt {attempt + 1}): Exchange error - {e}. Retrying...{Style.RESET_ALL}"
                    )
                    time.sleep(CONFIG.retry_delay_seconds)
            except Exception as e:
                logger.warning(
                    f"{Fore.YELLOW}Leverage Conjuring Failed (Attempt {attempt + 1}): Unexpected error - {e}. Retrying...{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.retry_delay_seconds)

        if not leverage_set:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to set leverage after {CONFIG.retry_count} attempts. Cannot continue.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{CONFIG.strategy_name}] CRITICAL: Failed to set leverage for {CONFIG.symbol}. Bot failed."
            )
            raise SystemExit("Failed to set leverage")  # Exit if leverage cannot be set

        logger.success(f"{Fore.GREEN}Spell successfully focused on Symbol: {CONFIG.symbol}{Style.RESET_ALL}")

        logger.info(f"{Fore.BLUE}--- Spell Configuration Summary ---{Style.RESET_ALL}")
        logger.info(
            f"{Fore.BLUE}Symbol: {CONFIG.symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x{Style.RESET_ALL}"
        )
        logger.info(f"{Fore.BLUE}Strategy Path: {CONFIG.strategy_name}{Style.RESET_ALL}")
        # Log strategy specific parameters (add more as needed)
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            logger.info(
                f"{Fore.BLUE}  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}{Style.RESET_ALL}"
            )
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            logger.info(
                f"{Fore.BLUE}  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} ({CONFIG.stochrsi_oversold}-{CONFIG.stochrsi_overbought}), Momentum={CONFIG.momentum_length}{Style.RESET_ALL}"
            )
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            logger.info(
                f"{Fore.BLUE}  Params: Fisher={CONFIG.ehlers_fisher_length}/{CONFIG.ehlers_fisher_signal_length}{Style.RESET_ALL}"
            )
        elif CONFIG.strategy_name == "EMA_CROSS":
            logger.info(
                f"{Fore.BLUE}  Params: EMA Fast={CONFIG.ema_fast_period}, Slow={CONFIG.ema_slow_period}{Style.RESET_ALL}"
            )  # Note renamed config variables

        logger.info(
            f"{Fore.BLUE}Risk Ward: {CONFIG.risk_per_trade_percentage:.3%} equity/trade, Max Pos Value: {CONFIG.max_order_usdt_amount.normalize()} {CONFIG.usdt_symbol}{Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.BLUE}Initial SL Ward: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period}){Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.BLUE}Initial TP Enchantment: {CONFIG.atr_take_profit_multiplier} * ATR({CONFIG.atr_calculation_period}){Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.BLUE}Trailing SL Shield: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}{Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.BLUE}Volume Filter: EntryRequiresSpike={CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, SpikeThr={CONFIG.volume_spike_threshold}x){Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.BLUE}Order Book Filter: FetchPerCycle={CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short}){Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.BLUE}Timing: Sleep={CONFIG.sleep_seconds}s | API: RecvWin={CONFIG.default_recv_window}ms, FillTimeout={CONFIG.order_fill_timeout_seconds}s, StopConfirmAttempts={CONFIG.stop_attach_confirm_attempts}, StopConfirmDelay={CONFIG.stop_attach_confirm_delay_seconds}s{Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.BLUE}Other: Margin Buffer={CONFIG.required_margin_buffer - 1:.1%}, SMS Alerts={CONFIG.enable_sms_alerts}{Style.RESET_ALL}"
        )
        logger.info(f"{Fore.BLUE}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}------------------------------{Style.RESET_ALL}")

        # Send initial configuration SMS alert
        sms_config_summary = f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Spell Initialized. Leverage: {CONFIG.leverage}x. Strategy: {CONFIG.strategy_name}. Risk: {CONFIG.risk_per_trade_percentage:.2%}. SL/TP/TSL Active."
        send_sms_alert(sms_config_summary)

    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: Setup failed during focus/leverage setting: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{CONFIG.strategy_name}] CRITICAL SETUP FAILED for {CONFIG.symbol}. Error: {type(e).__name__}."
        )
        # Attempt graceful shutdown actions before exiting
        graceful_shutdown(exchange)
        sys.exit(1)

    # --- Determine required candle history for indicators + buffer ---
    # Calculate maximum lookback needed by any configured indicator
    max_indicator_length = 0
    if "DUAL_SUPERTREND" in CONFIG.valid_strategies:  # Check against all valid strategies, not just selected
        max_indicator_length = max(max_indicator_length, CONFIG.st_atr_length, CONFIG.confirm_st_atr_length)
    if "STOCHRSI_MOMENTUM" in CONFIG.valid_strategies:
        max_indicator_length = max(
            max_indicator_length,
            CONFIG.stochrsi_rsi_length,
            CONFIG.stochrsi_stoch_length,
            CONFIG.stochrsi_k_period,
            CONFIG.stochrsi_d_period,
            CONFIG.momentum_length,
        )
    if "EHLERS_FISHER" in CONFIG.valid_strategies:
        max_indicator_length = max(
            max_indicator_length, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length
        )
    # Note: EMA cross also needs sufficient history, use the slow period
    if "EMA_CROSS" in CONFIG.valid_strategies:  # Check against renamed strategy name
        max_indicator_length = max(max_indicator_length, CONFIG.ema_slow_period)

    # Add ATR calculation period (used for dynamic SL/TP)
    max_indicator_length = max(max_indicator_length, CONFIG.atr_calculation_period)
    # Add Volume MA period (used for Volume filter)
    max_indicator_length = max(max_indicator_length, CONFIG.volume_ma_period)

    # Total candles needed = max lookback + buffer + a few extra for safety/derivations
    # Estimate minimum candles needed for indicators + buffer
    # A common rule of thumb is max(longest_period * 2, 100) + buffer, but let's be more specific
    candles_needed = max_indicator_length + CONFIG.api_fetch_limit_buffer + 10  # Add extra buffer

    logger.info(
        f"Estimating {candles_needed} candles needed for indicators ({max_indicator_length} max lookback + {CONFIG.api_fetch_limit_buffer} buffer + 10 safety)."
    )

    # --- Main Cycle Loop ---
    running = True
    while running:
        try:
            # Fetch OHLCV Data
            df = get_market_data(exchange, CONFIG.symbol, CONFIG.interval, candles_needed)
            if (
                df is None or df.empty or len(df) < max_indicator_length
            ):  # Re-check length against strict min for indicators
                logger.warning(
                    f"{Fore.YELLOW}Cycle Skip: Failed to get sufficient market data. Waiting for next cycle.{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds)
                continue  # Skip to next cycle if data fetch failed or insufficient

            # Log the timestamp of the latest candle
            latest_candle_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")
            logger.info(
                f"\n========== New Weaving Cycle ({CONFIG.strategy_name}): {CONFIG.symbol} | Candle: {latest_candle_time} =========={Style.RESET_ALL}"
            )

            # Calculate ATR and Volume indicators (needed for dynamic SL/TP and filters)
            vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
            current_atr: Decimal | None = vol_atr_data.get("atr")
            last_volume_ratio: Decimal | None = vol_atr_data.get("volume_ratio")
            vol_atr_data.get("volume_ma")

            # Check if latest ATR is valid (needed for SL/TP calculation)
            if current_atr is None or current_atr <= CONFIG.position_qty_epsilon:
                logger.warning(
                    f"{Fore.YELLOW}Cycle Skip: Calculated ATR ({current_atr}) is invalid or zero. Cannot calculate dynamic SL/TP. Waiting for next cycle.{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds)
                continue  # Skip if ATR is bad

            # Fetch Order Book data if needed per cycle
            order_book_data: dict[str, Decimal | None] = {
                "bid_ask_ratio": None,
                "spread": None,
                "best_bid": None,
                "best_ask": None,
            }
            if CONFIG.fetch_order_book_per_cycle or (
                CONFIG.order_book_ratio_threshold_long > CONFIG.position_qty_epsilon
                or CONFIG.order_book_ratio_threshold_short > CONFIG.position_qty_epsilon
            ):
                order_book_data = analyze_order_book(
                    exchange, CONFIG.symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
                )
                # Log whether OB was fetched this cycle for clarity
                order_book_data["fetched_this_cycle"] = True  # Custom key for tracking
            else:
                order_book_data["fetched_this_cycle"] = False

            ob_ratio = order_book_data.get("bid_ask_ratio")
            ob_spread = order_book_data.get("spread")
            ob_fetched = order_book_data.get("fetched_this_cycle")

            # Calculate Strategy-Specific Indicators
            if CONFIG.strategy_name == "DUAL_SUPERTREND":
                df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
                df = calculate_supertrend(
                    df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_"
                )
            elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
                df = calculate_stochrsi_momentum(
                    df,
                    CONFIG.stochrsi_rsi_length,
                    CONFIG.stochrsi_stoch_length,
                    CONFIG.stochrsi_k_period,
                    CONFIG.stochrsi_d_period,
                    CONFIG.momentum_length,
                )
            elif CONFIG.strategy_name == "EHLERS_FISHER":
                df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
            elif CONFIG.strategy_name == "EMA_CROSS":  # Note renamed strategy
                df = calculate_ema_cross(df, CONFIG.ema_fast_period, CONFIG.ema_slow_period)
            # Add other strategies here...

            # Fetch Current Position State
            current_position = get_current_position(exchange, CONFIG.symbol)
            position_side = current_position["side"]
            position_qty = current_position["qty"]
            position_entry_price = current_position["entry_price"]

            # Get the *most recent* price from the fetched data
            last_price = safe_decimal_conversion(df["close"].iloc[-1])
            # Get the timestamp of the last candle's close
            df.index[-1]

            # Log current state for context
            logger.info(
                f"State | Price: {last_price.normalize():.4f}, ATR({CONFIG.atr_calculation_period}): {current_atr.normalize():.5f}"
            )

            # Add Volume Filter state logging
            vol_filter_state = f"Ratio={last_volume_ratio:.2f}" if last_volume_ratio is not None else "Ratio=N/A"
            vol_spike_check = (
                "YES" if last_volume_ratio is not None and last_volume_ratio >= CONFIG.volume_spike_threshold else "NO"
            )
            logger.info(
                f"State | Volume: {vol_filter_state}, Spike={vol_spike_check} (Threshold={CONFIG.volume_spike_threshold}, RequiredForEntry={CONFIG.require_volume_spike_for_entry})"
            )

            # Add Order Book Filter state logging
            ob_filter_state = f"Ratio(B/A)={ob_ratio:.3f}" if ob_ratio is not None else "Ratio=N/A"
            ob_spread_state = f"Spread={ob_spread:.4f}" if ob_spread is not None else "Spread=N/A"
            logger.info(
                f"State | OrderBook: {ob_filter_state} (L >= {CONFIG.order_book_ratio_threshold_long}, S <= {CONFIG.order_book_ratio_threshold_short}), {ob_spread_state} (Fetched This Cycle={ob_fetched})"
            )

            # Add Position state logging
            pos_details = f"Side={position_side}, Qty={position_qty.normalize():.8f}, Entry={position_entry_price.normalize():.4f}"
            if position_side != CONFIG.pos_none:
                pos_stops_details = []
                if current_position.get("stop_loss") is not None:
                    pos_stops_details.append(f"SL={current_position['stop_loss'].normalize():.4f}")
                if current_position.get("take_profit") is not None:
                    pos_stops_details.append(f"TP={current_position['take_profit'].normalize():.4f}")
                if current_position.get("trailing_stop_price") is not None:
                    pos_stops_details.append(f"TSL(Trig)={current_position['trailing_stop_price'].normalize():.4f}")
                if pos_stops_details:
                    pos_details += f" | Stops: {' | '.join(pos_stops_details)}"
                pos_pnl = current_position.get("unrealized_pnl")
                if pos_pnl is not None:
                    pnl_color = Fore.GREEN if pos_pnl > 0 else (Fore.RED if pos_pnl < 0 else Fore.WHITE)
                    pos_details += f" | UPNL: {pnl_color}{pos_pnl.normalize():.4f}{Style.RESET_ALL}"
                pos_liq_price = current_position.get("liquidation_price")
                if pos_liq_price is not None and pos_liq_price > CONFIG.position_qty_epsilon:
                    pos_details += f" | Liq Price: {pos_liq_price.normalize():.4f}"

            logger.info(f"State | Position: {pos_details}{Style.RESET_ALL}")

            # Generate Trading Signal
            signal = generate_trading_signal(
                df, current_position, vol_atr_data if CONFIG.require_volume_spike_for_entry else order_book_data
            )  # Pass volume ratio via vol_atr_data if needed

            # --- Execute Trade based on Signal ---
            if signal == "long" and position_side == CONFIG.pos_none:
                logger.info(f"{Fore.GREEN}Entry Signal: LONG! Preparing order...{Style.RESET_ALL}")

                # Calculate SL/TP prices based on current price and ATR
                # SL Price = Current Price - (ATR * ATR_STOP_LOSS_MULTIPLIER) for Long
                stop_loss_price = last_price - (current_atr * CONFIG.atr_stop_loss_multiplier)
                # TP Price = Current Price + (ATR * ATR_TAKE_PROFIT_MULTIPLIER) for Long
                take_profit_price = last_price + (current_atr * CONFIG.atr_take_profit_multiplier)

                # Ensure SL/TP prices are non-negative and valid relative to current price
                if stop_loss_price <= CONFIG.position_qty_epsilon or stop_loss_price >= last_price:
                    logger.warning(
                        f"{Fore.YELLOW}Calculated Long SL price ({stop_loss_price.normalize()}) is invalid or not below current price ({last_price.normalize()}). Skipping order placement.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Long SL calc invalid ({stop_loss_price.normalize()}). Skipping order."
                    )
                    # Consider calculating quantity based on a *minimum* acceptable SL distance if ATR yields a bad one
                    continue  # Skip this cycle

                if take_profit_price <= CONFIG.position_qty_epsilon or take_profit_price <= last_price:
                    logger.warning(
                        f"{Fore.YELLOW}Calculated Long TP price ({take_profit_price.normalize()}) is invalid or not above current price ({last_price.normalize()}). Skipping order placement.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Long TP calc invalid ({take_profit_price.normalize()}). Skipping order."
                    )
                    continue  # Skip this cycle

                # Calculate Order Quantity
                # Note: Quantity is calculated based on the *potential* SL price for risk management.
                # The actual entry price might differ slightly due to market order slippage.
                order_quantity = calculate_order_quantity(
                    exchange,
                    CONFIG.symbol,
                    safe_decimal_conversion(
                        exchange.fetch_balance(params={"category": "linear"})
                        .get("total", {})
                        .get(CONFIG.usdt_symbol, "0.0")
                    ),  # Fetch fresh equity
                    last_price,
                    stop_loss_price,
                    CONFIG.side_buy,
                    market,
                )

                if order_quantity is not None:
                    # Place the market order with native SL/TP/TSL
                    created_order = create_order(
                        exchange,
                        CONFIG.symbol,
                        "market",
                        CONFIG.side_buy,
                        order_quantity,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        trailing_stop_percentage=CONFIG.trailing_stop_percentage,
                    )

                    if created_order:
                        logger.success(
                            f"{Fore.GREEN}Long entry order placed successfully! Waiting for position confirmation...{Style.RESET_ALL}"
                        )
                        # Wait a moment for the exchange to register the position and stops
                        time.sleep(CONFIG.stop_attach_confirm_delay_seconds)
                        # Verify position and stops are active
                        current_position_after_entry = get_current_position(exchange, CONFIG.symbol)
                        if current_position_after_entry["side"] == CONFIG.pos_long and current_position_after_entry[
                            "qty"
                        ] >= order_quantity * (
                            Decimal("1") - CONFIG.position_qty_epsilon
                        ):  # Check if position quantity is close to order quantity
                            logger.success(f"{Fore.GREEN}Position confirmed active after long entry.{Style.RESET_ALL}")
                            # Confirm native stops are attached
                            confirm_stops_attached(
                                exchange,
                                CONFIG.symbol,
                                stop_loss_price,
                                take_profit_price,
                                CONFIG.trailing_stop_percentage,
                                CONFIG.stop_attach_confirm_attempts,
                                CONFIG.stop_attach_confirm_delay_seconds,
                            )
                            # Once in a position, we enter a different state or just wait for the next cycle's exit signal
                            # The main loop structure naturally handles waiting for the exit signal in the next cycle.
                            pass  # Successfully entered and confirmed
                        else:
                            logger.error(
                                f"{Fore.RED}Position confirmation failed after long entry order. Current state: {current_position_after_entry}. Manual check required!{Style.RESET_ALL}"
                            )
                            send_sms_alert(
                                f"[{CONFIG.strategy_name}/{CONFIG.symbol}] WARNING: Long entry order placed but position NOT confirmed active."
                            )
                    else:
                        logger.error(
                            f"{Fore.RED}Long entry order failed. Order placement returned None.{Style.RESET_ALL}"
                        )
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Long entry signal generated, but quantity calculation failed. Skipping order.{Style.RESET_ALL}"
                    )

            elif signal == "short" and position_side == CONFIG.pos_none:
                logger.info(f"{Fore.RED}Entry Signal: SHORT! Preparing order...{Style.RESET_ALL}")

                # Calculate SL/TP prices based on current price and ATR
                # SL Price = Current Price + (ATR * ATR_STOP_LOSS_MULTIPLIER) for Short
                stop_loss_price = last_price + (current_atr * CONFIG.atr_stop_loss_multiplier)
                # TP Price = Current Price - (ATR * ATR_TAKE_PROFIT_MULTIPLIER) for Short
                take_profit_price = last_price - (current_atr * CONFIG.atr_take_profit_multiplier)

                # Ensure SL/TP prices are non-negative and valid relative to current price
                if stop_loss_price <= CONFIG.position_qty_epsilon or stop_loss_price <= last_price:
                    logger.warning(
                        f"{Fore.YELLOW}Calculated Short SL price ({stop_loss_price.normalize()}) is invalid or not above current price ({last_price.normalize()}). Skipping order placement.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Short SL calc invalid ({stop_loss_price.normalize()}). Skipping order."
                    )
                    continue  # Skip this cycle

                if take_profit_price <= CONFIG.position_qty_epsilon or take_profit_price >= last_price:
                    logger.warning(
                        f"{Fore.YELLOW}Calculated Short TP price ({take_profit_price.normalize()}) is invalid or not below current price ({last_price.normalize()}). Skipping order placement.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Short TP calc invalid ({take_profit_price.normalize()}). Skipping order."
                    )
                    continue  # Skip this cycle

                # Calculate Order Quantity
                order_quantity = calculate_order_quantity(
                    exchange,
                    CONFIG.symbol,
                    safe_decimal_conversion(
                        exchange.fetch_balance(params={"category": "linear"})
                        .get("total", {})
                        .get(CONFIG.usdt_symbol, "0.0")
                    ),  # Fetch fresh equity
                    last_price,
                    stop_loss_price,
                    CONFIG.side_sell,
                    market,
                )

                if order_quantity is not None:
                    # Place the market order with native SL/TP/TSL
                    created_order = create_order(
                        exchange,
                        CONFIG.symbol,
                        "market",
                        CONFIG.side_sell,
                        order_quantity,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        trailing_stop_percentage=CONFIG.trailing_stop_percentage,
                    )

                    if created_order:
                        logger.success(
                            f"{Fore.GREEN}Short entry order placed successfully! Waiting for position confirmation...{Style.RESET_ALL}"
                        )
                        # Wait a moment for the exchange to register the position and stops
                        time.sleep(CONFIG.stop_attach_confirm_delay_seconds)
                        # Verify position and stops are active
                        current_position_after_entry = get_current_position(exchange, CONFIG.symbol)
                        if current_position_after_entry["side"] == CONFIG.pos_short and current_position_after_entry[
                            "qty"
                        ] >= order_quantity * (
                            Decimal("1") - CONFIG.position_qty_epsilon
                        ):  # Check if position quantity is close to order quantity
                            logger.success(f"{Fore.GREEN}Position confirmed active after short entry.{Style.RESET_ALL}")
                            # Confirm native stops are attached
                            confirm_stops_attached(
                                exchange,
                                CONFIG.symbol,
                                stop_loss_price,
                                take_profit_price,
                                CONFIG.trailing_stop_percentage,
                                CONFIG.stop_attach_confirm_attempts,
                                CONFIG.stop_attach_confirm_delay_seconds,
                            )
                            # Once in a position, we enter a different state or just wait for the next cycle's exit signal
                            pass  # Successfully entered and confirmed
                        else:
                            logger.error(
                                f"{Fore.RED}Position confirmation failed after short entry order. Current state: {current_position_after_entry}. Manual check required!{Style.RESET_ALL}"
                            )
                            send_sms_alert(
                                f"[{CONFIG.strategy_name}/{CONFIG.symbol}] WARNING: Short entry order placed but position NOT confirmed active."
                            )
                    else:
                        logger.error(
                            f"{Fore.RED}Short entry order failed. Order placement returned None.{Style.RESET_ALL}"
                        )
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Short entry signal generated, but quantity calculation failed. Skipping order.{Style.RESET_ALL}"
                    )

            elif signal in ["close_long", "close_short"]:
                # This signal indicates the strategy wants to exit the current position
                # Note: Native SL/TP/TSL should ideally handle most exits automatically.
                # This manual close is a fallback based on the primary strategy logic's exit signal.
                # It's important to ensure this manual close doesn't interfere with native stops.
                # Using 'reduceOnly' on a market order should close the position without interfering
                # with active stop orders on the position (Bybit V5 behavior).

                if (signal == "close_long" and position_side == CONFIG.pos_long) or (
                    signal == "close_short" and position_side == CONFIG.pos_short
                ):
                    # Attempt to close the position
                    close_success = close_position(exchange, CONFIG.symbol, current_position)

                    if close_success:
                        logger.success(
                            f"{Fore.GREEN}Position closed successfully via strategy exit signal ({signal}).{Style.RESET_ALL}"
                        )
                        # No need to wait for next cycle; the close order is assumed fast.
                        # But let's briefly pause to allow state to update before the next cycle.
                        time.sleep(CONFIG.post_close_delay_seconds)
                    else:
                        logger.error(
                            f"{Fore.RED}Position closure failed for {CONFIG.symbol}. Manual intervention may be required.{Style.RESET_ALL}"
                        )
                        # The loop will continue and attempt closure again if position is still open
                else:
                    # This case was handled in generate_trading_signal, should not be reached here
                    logger.warning(
                        f"{Fore.YELLOW}Received unexpected exit signal ({signal}) for current {position_side} position. Ignoring.{Style.RESET_ALL}"
                    )

            elif signal is None:
                logger.info("Holding Cash. No entry signal generated by strategy or filters, or no exit signal needed.")

            # --- End of Cycle ---
            logger.info(f"========== Cycle Weaving End: {CONFIG.symbol} =========={Style.RESET_ALL}")

        except ccxt.NetworkError as e:
            logger.error(
                f"{Fore.RED}Major Network Disturbance during cycle: {e}. Retrying after delay...{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Major Network Error: {type(e).__name__}. Retrying."
            )
            time.sleep(CONFIG.retry_delay_seconds * 2)  # Longer delay for network issues
            continue  # Continue loop
        except ccxt.ExchangeError as e:
            logger.error(f"{Fore.RED}Major Exchange Error during cycle: {e}. Retrying after delay...{Style.RESET_ALL}")
            send_sms_alert(
                f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Major Exchange Error: {type(e).__name__}. Retrying."
            )
            time.sleep(CONFIG.retry_delay_seconds * 2)  # Longer delay for exchange errors
            continue  # Continue loop
        except ccxt.DDoSProtection as e:
            logger.warning(
                f"{Fore.YELLOW}Rate Limit / DDoS Protection triggered during cycle: {e}. Backing off...{Style.RESET_ALL}"
            )
            send_sms_alert(f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Rate Limit Hit. Backing off.")
            time.sleep(exchange.rateLimit / 1000 + 5)  # Wait longer than rate limit
            continue  # Continue loop
        except Exception as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Unexpected and unhandled chaos during cycle: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{CONFIG.strategy_name}/{CONFIG.symbol}] CRITICAL UNHANDLED ERROR: {type(e).__name__}. Attempting graceful shutdown."
            )
            running = False  # Set running to False to trigger shutdown sequence
            # Do NOT `continue` here, let the loop end naturally to go to shutdown.

        # --- Sleep before next cycle ---
        if running:  # Only sleep if not shutting down due to error
            logger.debug(f"Pausing spell for {CONFIG.sleep_seconds} seconds...")
            time.sleep(CONFIG.sleep_seconds)

    # Loop ends when running is False (due to critical error) or interrupted (handled by main __main__ block)


def graceful_shutdown(exchange: ccxt.Exchange | None) -> None:
    """Attempts to gracefully shut down the bot by cancelling orders and closing positions."""
    logger.warning("\nShutdown requested. Withdrawing arcane energies gracefully...{Style.RESET_ALL}")
    send_sms_alert(f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Initiating graceful shutdown.")

    # Indicate if running in live mode for shutdown messages
    is_live = (
        not getattr(exchange, "sandbox", False) if exchange else True
    )  # Assume live if exchange object is None or sandbox status unavailable
    if is_live:
        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Shutdown operating in LIVE mode.{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.BLUE}Shutdown operating in TESTNET mode.{Style.RESET_ALL}")

    # --- Step 1: Cancel Open Orders ---
    logger.warning(f"Shutdown Step 1: Cancelling all open orders for {CONFIG.symbol}...{Style.RESET_ALL}")
    if exchange:
        try:
            cancel_attempts = cancel_all_orders_for_symbol(exchange, CONFIG.symbol, "Graceful Shutdown")
            logger.info(
                f"Shutdown Step 1: Cancel order attempt finished. ({cancel_attempts} attempts){Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Shutdown Step 1 Error: Unexpected error during order cancellation attempt: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
    else:
        logger.warning("Shutdown Step 1: Exchange object not initialized. Cannot cancel orders.{Style.RESET_ALL}")

    # --- Step 2: Close Active Position ---
    logger.warning(f"Shutdown Step 2: Checking for active position to close...{Style.RESET_ALL}")
    if exchange:
        try:
            # Fetch current position first
            current_position = get_current_position(exchange, CONFIG.symbol)

            if current_position["side"] != CONFIG.pos_none and current_position["qty"] > CONFIG.position_qty_epsilon:
                logger.info(
                    f"Shutdown Step 2: Active {current_position['side']} position found (Qty: {current_position['qty'].normalize()}). Attempting to close...{Style.RESET_ALL}"
                )
                close_success = close_position(exchange, CONFIG.symbol, current_position)

                if close_success:
                    logger.success(
                        f"{Fore.GREEN}Shutdown Step 2: Active position for {CONFIG.symbol} successfully closed.{Style.RESET_ALL}"
                    )
                else:
                    logger.error(
                        f"{Fore.RED}Shutdown Step 2 Error: Failed to close active position for {CONFIG.symbol}. Manual intervention required!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{CONFIG.strategy_name}/{CONFIG.symbol}] SHUTDOWN WARNING: Failed to close position. Manual check required."
                    )
            else:
                logger.info(
                    f"{Fore.BLUE}Shutdown Step 2: No active position found for {CONFIG.symbol}. Clean exit state.{Style.RESET_ALL}"
                )

        except Exception as e:
            logger.error(
                f"{Fore.RED}Shutdown Step 2 Error: Unexpected error during position closing attempt: {e}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{CONFIG.strategy_name}/{CONFIG.symbol}] SHUTDOWN ERROR: Unexpected error closing position: {type(e).__name__}."
            )
    else:
        logger.warning(
            "Shutdown Step 2: Exchange object not initialized. Cannot check/close position.{Style.RESET_ALL}"
        )

    # --- Final Shutdown Message ---
    shutdown_msg = f"--- Pyrmethus Scalping Spell Shutdown Sequence Complete ({'LIVE' if is_live else 'TESTNET'}) ---"
    logger.info(f"{Fore.BLUE}{shutdown_msg}{Style.RESET_ALL}")
    logger.info(f"{Fore.BLUE}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")
    send_sms_alert(f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Shutdown complete.")


# --- Entry Point - Igniting the Spell ---
if __name__ == "__main__":
    exchange: ccxt.Exchange | None = None
    try:
        exchange = initialize_exchange()
        if exchange:
            main_trade_logic(exchange)
        else:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Exchange initialization failed. Cannot run main trade logic.{Style.RESET_ALL}"
            )
            # Initialization failure should have already triggered SMS/exit

    except KeyboardInterrupt:
        logger.warning("\nKeyboardInterrupt detected. User requests withdrawal of arcane energies...{Style.RESET_ALL}")
        # Graceful shutdown will be triggered by the `finally` block

    except SystemExit:
        logger.info("SystemExit requested, terminating.")
        # SystemExit is raised intentionally, typically after critical errors handled within functions.
        # The finally block will still attempt shutdown, but it might be dealing with a partially initialized state.
        pass

    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}An unhandled exception caused the spell to fail: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # The finally block will handle shutdown

    finally:
        # Ensure graceful shutdown is attempted even if errors occur
        if exchange:
            graceful_shutdown(exchange)
        else:
            logger.warning(
                "Cannot attempt graceful shutdown: Exchange object was not successfully initialized.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{CONFIG.strategy_name}/{CONFIG.symbol}] Shutdown attempt failed: Exchange not initialized."
            )
