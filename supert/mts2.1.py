#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.3.0 (Enhanced Modularity, Caching, Metrics & Health Checks)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.3.0 (Unified: Selectable Strategies + Precision + Native SL/TSL + Fortified Config + Pyrmethus Enhancements + Robustness + Ehlers SSF MA + Enhanced Indicator Logging + Corrected ST Flips + Strategy ABC + Data Caching + Trade Metrics + Account Health).

Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS" (using Ehlers Super Smoother Filter).
- Modular strategy implementation using Abstract Base Class (ABC).
- Enhanced Precision: Uses Decimal for critical financial calculations.
- Fortified Configuration Loading: Correctly handles type casting for environment variables and default values.
- Exchange-native Trailing Stop Loss (TSL) placed immediately after entry (Bybit V5).
- Exchange-native fixed Stop Loss (based on ATR) placed immediately after entry (Bybit V5).
- ATR for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation filters.
- Risk-based position sizing with margin checks and configurable cap.
- Termux SMS alerts for critical events and trade actions (with Termux:API command check).
- Robust error handling (CCXT exceptions, validation) and detailed logging with vibrant Neon color support via Colorama and File logging.
- Graceful shutdown on KeyboardInterrupt or critical errors, attempting position/order closing.
- Stricter position detection logic tailored for Bybit V5 API (One-Way Mode).
- OHLCV Data Caching: Reduces API calls by caching market data within a candle's duration.
- Trade Metrics: Basic tracking of trade P/L and win rate.
- Account Health Check: Monitors margin ratio to prevent excessive risk.
- NaN handling in fetched OHLCV data.
- Re-validation of position state before closing.
- Enhanced Indicator Logging: Comprehensive output of key indicator values each cycle.
- Corrected SuperTrend flip signal generation for improved DUAL_SUPERTREND strategy accuracy.
- Enhanced robustness in order placement: Attempts emergency close if essential stop-loss orders fail after entry.

Disclaimer:
- **EXTREME RISK**: Trading futures, especially with leverage and automation, is extremely risky. This script is for EDUCATIONAL PURPOSES ONLY. You can lose all your capital and more. Use at your own absolute risk.
- **EXCHANGE-NATIVE SL/TSL DEPENDENCE**: Relies entirely on Bybit's native SL/TSL order execution. Performance is subject to exchange conditions, potential slippage, API reliability, and order book liquidity. These orders are NOT guaranteed to execute at the exact trigger price.
- **PARAMETER SENSITIVITY**: Bot performance is highly sensitive to parameter tuning (strategy settings, risk, SL/TSL percentages, filters). Requires significant backtesting and forward testing on TESTNET.
- **API RATE LIMITS**: Monitor API usage. Excessive requests can lead to temporary or permanent bans from the exchange.
- **SLIPPAGE**: Market orders used for entry and potentially for SL/TSL execution are prone to slippage, especially during volatile market conditions.
- **TEST THOROUGHLY**: **DO NOT RUN WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTNET/DEMO TESTING.** Understand every part of the code before considering live deployment.
- **TERMUX DEPENDENCY**: Requires Termux environment and Termux:API package (`pkg install termux-api`) for SMS alerts. Ensure it's correctly installed and configured.
- **API CHANGES**: This code targets the Bybit V5 API via CCXT. Exchange API updates may break functionality. Keep CCXT updated (`pip install -U ccxt`).
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import shutil  # For checking command existence
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any

import pytz  # For timezone-aware datetimes

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
    from retry import retry  # For safe_api_call decorator
except ImportError as e:
    missing_pkg = e.name
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing required Python package: '{missing_pkg}'.\033[0m\n"
    )
    sys.stderr.write(
        f"\033[91mPlease install it by running: pip install {missing_pkg}\033[0m\n"
    )
    if missing_pkg == "pandas_ta":
        sys.stderr.write(
            f"\033[91mFor pandas_ta, you might also need TA-Lib. See pandas_ta documentation.\033[0m\n"
        )
    if missing_pkg == "retry":
        sys.stderr.write(
            f"\033[91mFor retry decorator, install with: pip install retry\033[0m\n"
        )
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18


# --- Enums ---
class StrategyName(str, Enum):
    DUAL_SUPERTREND = "DUAL_SUPERTREND"
    STOCHRSI_MOMENTUM = "STOCHRSI_MOMENTUM"
    EHLERS_FISHER = "EHLERS_FISHER"
    EHLERS_MA_CROSS = "EHLERS_MA_CROSS"


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads and validates configuration parameters from environment variables."""

    def __init__(self) -> None:
        # Logger might not be fully configured yet when Config is initialized first.
        # Use print for initial config messages or configure logger very early.
        # For now, assuming logger is available at least for info level.
        _pre_logger = logging.getLogger(__name__)  # Temp logger for config phase
        _pre_logger.info(
            f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}"
        )
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: str | None = self._get_env(
            "BYBIT_API_KEY", required=True, color=Fore.RED
        )
        self.api_secret: str | None = self._get_env(
            "BYBIT_API_SECRET", required=True, color=Fore.RED
        )

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env(
            "LEVERAGE", 25, cast_type=int, color=Fore.YELLOW
        )
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW
        )

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: StrategyName = StrategyName(
            self._get_env(
                "STRATEGY_NAME", StrategyName.DUAL_SUPERTREND.value, color=Fore.CYAN
            ).upper()
        )
        self.valid_strategies: list[str] = [s.value for s in StrategyName]
        if self.strategy_name.value not in self.valid_strategies:
            _pre_logger.critical(
                f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name.value}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}"
            )
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name.value}'.")
        _pre_logger.info(
            f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name.value}{Style.RESET_ALL}"
        )
        self.strategy_instance: (
            "TradingStrategy"  # Forward declaration, will be set after CONFIG is loaded
        )

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.01", cast_type=Decimal, color=Fore.GREEN
        )
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.3", cast_type=Decimal, color=Fore.GREEN
        )
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "50.0", cast_type=Decimal, color=Fore.GREEN
        )
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN
        )
        self.max_account_margin_ratio: Decimal = (
            self._get_env(  # For check_account_health
                "MAX_ACCOUNT_MARGIN_RATIO",
                "0.8",
                cast_type=Decimal,
                color=Fore.GREEN,  # 80%
            )
        )

        # --- Trailing Stop Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE",
            "0.005",
            cast_type=Decimal,
            color=Fore.GREEN,  # e.g., 0.5%
        )
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT",
            "0.001",  # e.g., 0.1%
            cast_type=Decimal,
            color=Fore.GREEN,
        )

        # --- Strategy-Specific Parameters - Tuning the Chosen Path ---
        # Dual Supertrend
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "1.0", cast_type=Decimal, color=Fore.CYAN
        )
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 3, cast_type=int, color=Fore.CYAN
        )
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "0.6", cast_type=Decimal, color=Fore.CYAN
        )
        # StochRSI + Momentum
        self.stochrsi_rsi_length: int = self._get_env(
            "STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_stoch_length: int = self._get_env(
            "STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_k_period: int = self._get_env(
            "STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_d_period: int = self._get_env(
            "STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_overbought: Decimal = self._get_env(
            "STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN
        )
        self.stochrsi_oversold: Decimal = self._get_env(
            "STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN
        )
        self.momentum_length: int = self._get_env(
            "MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )
        # Ehlers Fisher Transform
        self.ehlers_fisher_length: int = self._get_env(
            "EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN
        )
        self.ehlers_fisher_signal_length: int = self._get_env(
            "EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN
        )
        # Ehlers MA Cross (Super Smoother Filter)
        self.ehlers_fast_period: int = self._get_env(
            "EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN
        )
        self.ehlers_slow_period: int = self._get_env(
            "EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN
        )
        self.ehlers_ssf_poles: int = self._get_env(
            "EHLERS_SSF_POLES", 2, cast_type=int, color=Fore.CYAN
        )

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        self.volume_ma_period: int = self._get_env(
            "VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW
        )
        self.volume_spike_threshold: Decimal = self._get_env(
            "VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW
        )
        self.require_volume_spike_for_entry: bool = self._get_env(
            "REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW
        )
        self.order_book_depth: int = self._get_env(
            "ORDER_BOOK_DEPTH", 20, cast_type=int, color=Fore.YELLOW
        )
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG",
            "1.2",
            cast_type=Decimal,
            color=Fore.YELLOW,
        )
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT",
            "0.8",
            cast_type=Decimal,
            color=Fore.YELLOW,
        )
        self.fetch_order_book_per_cycle: bool = self._get_env(
            "FETCH_ORDER_BOOK_PER_CYCLE", "true", cast_type=bool, color=Fore.YELLOW
        )

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env(
            "ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN
        )

        # --- Termux SMS Alerts - Whispers Through the Digital Veil ---
        self.enable_sms_alerts: bool = self._get_env(
            "ENABLE_SMS_ALERTS", "true", cast_type=bool, color=Fore.MAGENTA
        )
        self.sms_recipient_number: str | None = self._get_env(
            "SMS_RECIPIENT_NUMBER", None, cast_type=str, color=Fore.MAGENTA
        )
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA
        )

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = 13000
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW
        )

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 3
        self.retry_delay_seconds: int = 2
        self.api_fetch_limit_buffer: int = 10
        self.position_qty_epsilon: Decimal = Decimal("1e-9")
        self.post_close_delay_seconds: int = 3
        self.cache_candle_duration_multiplier: Decimal = Decimal(
            "0.95"
        )  # For data cache validity, e.g., 95% of candle duration

        _pre_logger.info(
            f"{Fore.MAGENTA}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}"
        )

    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        color: str = Fore.WHITE,
    ) -> Any:
        _pre_logger = logging.getLogger(__name__)  # Use temp logger
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None

        if value_str is None:
            if required:
                _pre_logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required config '{key}' not in .env.{Style.RESET_ALL}"
                )
                raise ValueError(f"Required environment variable '{key}' not set.")
            _pre_logger.debug(
                f"{color}Config {key}: Not Set. Using Default: '{default}'{Style.RESET_ALL}"
            )
            value_to_cast = default
            source = "Default"
        else:
            _pre_logger.debug(
                f"{color}Config {key}: Found Env Value: '{value_str}'{Style.RESET_ALL}"
            )
            value_to_cast = value_str

        if value_to_cast is None:  # Catches if default was None and env var not set
            if required:
                _pre_logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required config '{key}' has no value (env/default).{Style.RESET_ALL}"
                )
                raise ValueError(
                    f"Required environment variable '{key}' resolved to None."
                )
            else:
                # This debug log is important for optional None values like SMS_RECIPIENT_NUMBER
                _pre_logger.debug(
                    f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}"
                )
                return None

        final_value: Any = None
        try:
            raw_value_str = str(
                value_to_cast
            )  # Ensure it's a string before type-specific casting
            if cast_type == bool:
                final_value = raw_value_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal:
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                # Allow casting from float-like strings, e.g., "25.0" to int 25
                final_value = int(Decimal(raw_value_str))
            elif cast_type == float:
                final_value = float(raw_value_str)
            elif cast_type == str:
                final_value = raw_value_str  # Already a string, or cast from default
            else:  # Should not happen if cast_type is one of the above
                _pre_logger.warning(
                    f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw."
                )
                final_value = value_to_cast  # Return as is

        except (ValueError, TypeError, InvalidOperation) as e:
            _pre_logger.error(
                f"{Fore.RED}Invalid type/value for {key}: '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Trying default '{default}'.{Style.RESET_ALL}"
            )
            if default is None:  # If default is also None, and casting failed
                if required:
                    _pre_logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed cast for required key '{key}', default is None.{Style.RESET_ALL}"
                    )
                    raise ValueError(
                        f"Required env var '{key}' failed casting, no valid default."
                    )
                else:
                    _pre_logger.warning(
                        f"{Fore.YELLOW}Casting failed for {key}, default is None. Final value: None{Style.RESET_ALL}"
                    )
                    return None  # Return None if default is None and casting failed
            else:
                # Try casting the default value
                source = "Default (Fallback)"
                _pre_logger.debug(
                    f"Casting fallback default '{default}' for '{key}' to {cast_type.__name__}"
                )
                try:
                    default_str = str(
                        default
                    )  # Ensure default is string before type-specific casting
                    if cast_type == bool:
                        final_value = default_str.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal:
                        final_value = Decimal(default_str)
                    elif cast_type == int:
                        final_value = int(Decimal(default_str))
                    elif cast_type == float:
                        final_value = float(default_str)
                    elif cast_type == str:
                        final_value = default_str
                    else:
                        final_value = default  # Should not happen
                    _pre_logger.warning(
                        f"{Fore.YELLOW}Used casted default for {key}: '{final_value}'{Style.RESET_ALL}"
                    )
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _pre_logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed cast for BOTH value ('{value_to_cast}') AND default ('{default}') for key '{key}' to {cast_type.__name__}. Error on default: {e_default}{Style.RESET_ALL}"
                    )
                    raise ValueError(
                        f"Config error: Cannot cast value or default for '{key}' to {cast_type.__name__}."
                    )
        _pre_logger.debug(
            f"{color}Using final value for {key}: {final_value} (Type: {type(final_value).__name__}) (Source: {source}){Style.RESET_ALL}"
        )
        return final_value


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL: int = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

log_file_name = f"logs/pyrmethus_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)-25s %(message)s",  # Increased name width
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_name),  # Added file handler
    ],
)
logger: logging.Logger = logging.getLogger("PyrmethusCore")  # Main logger for the bot

# Custom SUCCESS level and Neon Color Formatting for the Oracle
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)  # type: ignore[attr-defined]


logging.Logger.success = log_success  # type: ignore[attr-defined]

if sys.stdout.isatty():  # Apply colors only if output is a TTY
    logging.addLevelName(
        logging.DEBUG,
        f"{Fore.CYAN}{Style.DIM}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.INFO,
        f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        SUCCESS_LEVEL,
        f"{Fore.MAGENTA}{Style.BRIGHT}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.WARNING,
        f"{Fore.YELLOW}{Style.BRIGHT}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.ERROR,
        f"{Fore.RED}{Style.BRIGHT}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}",
    )
    logging.addLevelName(
        logging.CRITICAL,
        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}",
    )

# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()
except ValueError as config_error:
    # Use root logger if PyrmethusCore logger failed or not fully set up
    logging.getLogger().critical(  # Fallback to root logger
        f"{Back.RED}{Fore.WHITE}Configuration loading failed. Error: {config_error}{Style.RESET_ALL}"
    )
    sys.exit(1)
except Exception as general_config_error:
    logging.getLogger().critical(  # Fallback to root logger
        f"{Back.RED}{Fore.WHITE}Unexpected critical error during configuration: {general_config_error}{Style.RESET_ALL}"
    )
    logging.getLogger().debug(traceback.format_exc())
    sys.exit(1)


# --- Trading Strategy Abstract Base Class & Implementations ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: list[str] | None = None):
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{self.__class__.__name__}")
        self.required_columns = df_columns if df_columns else []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        pass

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(
                f"Insufficient data (Rows: {len(df) if df is not None else 0}, Need: {min_rows})."
            )
            return False
        if self.required_columns and not all(
            col in df.columns for col in self.required_columns
        ):
            missing_cols = [
                col for col in self.required_columns if col not in df.columns
            ]
            self.logger.warning(f"Missing required columns: {missing_cols}.")
            return False
        # Check for NaNs in the last row for required columns
        if self.required_columns and df.iloc[-1][self.required_columns].isnull().any():
            nan_cols_last_row = df.iloc[-1][self.required_columns].isnull()
            nan_cols_last_row = nan_cols_last_row[nan_cols_last_row].index.tolist()
            self.logger.debug(
                f"NaN values in last row for required columns: {nan_cols_last_row}"
            )
            # Allow strategies to handle this if they wish, or return False here to be stricter
        return True

    def _get_default_signals(self) -> dict[str, Any]:
        return {
            "enter_long": False,
            "enter_short": False,
            "exit_long": False,
            "exit_short": False,
            "exit_reason": "Strategy Exit Signal",
        }


class DualSupertrendStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(
            config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend"]
        )

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df):
            return signals
        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up = last.get(
            "confirm_trend", pd.NA
        )  # pd.NA is better than None for pandas boolean series

        if pd.isna(confirm_is_up):  # Explicit check for pd.NA
            self.logger.debug("Confirmation trend is NA. No signal.")
            return signals
        if primary_long_flip and confirm_is_up is True:
            signals["enter_long"] = True
        if primary_short_flip and confirm_is_up is False:
            signals["enter_short"] = True
        # Exit signals
        if primary_short_flip:  # If primary ST flips short, exit any long.
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary ST Flipped Short"
        if primary_long_flip:  # If primary ST flips long, exit any short.
            signals["exit_short"] = True
            signals["exit_reason"] = "Primary ST Flipped Long"
        return signals


class StochRsiMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["stochrsi_k", "stochrsi_d", "momentum"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2):
            return signals  # Need prev and current
        last, prev = df.iloc[-1], df.iloc[-2]

        # Use .get() with default pd.NA to handle potentially missing columns gracefully
        k_now, d_now, mom_now = (
            last.get("stochrsi_k", pd.NA),
            last.get("stochrsi_d", pd.NA),
            last.get("momentum", pd.NA),
        )
        k_prev, d_prev = prev.get("stochrsi_k", pd.NA), prev.get("stochrsi_d", pd.NA)

        if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]):
            self.logger.debug("Skipping due to NA StochRSI/Mom values.")
            return signals

        # Convert to Decimal for precise comparison with config thresholds
        k_now_dec = safe_decimal_conversion(k_now, Decimal("NaN"))
        # d_now_dec = safe_decimal_conversion(d_now, Decimal('NaN')) # Not directly compared with threshold
        mom_now_dec = safe_decimal_conversion(mom_now, Decimal("NaN"))
        # k_prev_dec = safe_decimal_conversion(k_prev, Decimal('NaN')) # Not directly compared
        # d_prev_dec = safe_decimal_conversion(d_prev, Decimal('NaN')) # Not directly compared

        if k_now_dec.is_nan() or mom_now_dec.is_nan():  # Check converted Decimals
            self.logger.debug(
                "Skipping due to NA StochRSI/Mom Decimal values after conversion."
            )
            return signals

        # Entry signals
        if (
            k_prev <= d_prev
            and k_now > d_now
            and k_now_dec < self.config.stochrsi_oversold
            and mom_now_dec > Decimal("0")
        ):
            signals["enter_long"] = True
        if (
            k_prev >= d_prev
            and k_now < d_now
            and k_now_dec > self.config.stochrsi_overbought
            and mom_now_dec < Decimal("0")
        ):
            signals["enter_short"] = True

        # Exit signals (based on StochRSI cross, simpler exit)
        if k_prev >= d_prev and k_now < d_now:  # K crosses below D
            signals["exit_long"] = True
            signals["exit_reason"] = "StochRSI K crossed below D"
        if k_prev <= d_prev and k_now > d_now:  # K crosses above D
            signals["exit_short"] = True
            signals["exit_reason"] = "StochRSI K crossed above D"
        return signals


class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2):
            return signals
        last, prev = df.iloc[-1], df.iloc[-2]

        fish_now, sig_now = (
            last.get("ehlers_fisher", pd.NA),
            last.get("ehlers_signal", pd.NA),
        )
        fish_prev, sig_prev = (
            prev.get("ehlers_fisher", pd.NA),
            prev.get("ehlers_signal", pd.NA),
        )

        if any(pd.isna(v) for v in [fish_now, sig_now, fish_prev, sig_prev]):
            self.logger.debug("Skipping due to NA Ehlers Fisher values.")
            return signals

        # Entry signals
        if fish_prev <= sig_prev and fish_now > sig_now:
            signals["enter_long"] = True
        if fish_prev >= sig_prev and fish_now < sig_now:
            signals["enter_short"] = True

        # Exit signals
        if fish_prev >= sig_prev and fish_now < sig_now:  # Fisher crosses below Signal
            signals["exit_long"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed Short"
        if fish_prev <= sig_prev and fish_now > sig_now:  # Fisher crosses above Signal
            signals["exit_short"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed Long"
        return signals


class EhlersMaCrossStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_ssf_fast", "ehlers_ssf_slow"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2):
            return signals
        last, prev = df.iloc[-1], df.iloc[-2]

        fast_ma_now, slow_ma_now = (
            last.get("ehlers_ssf_fast", pd.NA),
            last.get("ehlers_ssf_slow", pd.NA),
        )
        fast_ma_prev, slow_ma_prev = (
            prev.get("ehlers_ssf_fast", pd.NA),
            prev.get("ehlers_ssf_slow", pd.NA),
        )

        if any(
            pd.isna(v) for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]
        ):
            self.logger.debug("Skipping due to NA Ehlers SSF MA values.")
            return signals

        # Entry signals
        if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
            signals["enter_long"] = True
        if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
            signals["enter_short"] = True

        # Exit signals
        if (
            fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now
        ):  # Fast MA crosses below Slow MA
            signals["exit_long"] = True
            signals["exit_reason"] = "Fast Ehlers SSF MA crossed below Slow"
        if (
            fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now
        ):  # Fast MA crosses above Slow MA
            signals["exit_short"] = True
            signals["exit_reason"] = "Fast Ehlers SSF MA crossed above Slow"
        return signals


# Initialize strategy instance in CONFIG after it's loaded
strategy_map = {
    StrategyName.DUAL_SUPERTREND: DualSupertrendStrategy,
    StrategyName.STOCHRSI_MOMENTUM: StochRsiMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy,
    StrategyName.EHLERS_MA_CROSS: EhlersMaCrossStrategy,
}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass:
    CONFIG.strategy_instance = StrategyClass(CONFIG)
else:
    logger.critical(
        f"Failed to find strategy class for {CONFIG.strategy_name.value}. Exiting."
    )
    sys.exit(1)


# --- Trade Metrics Tracking ---
class TradeMetrics:
    def __init__(self):
        self.trades = []
        self.logger = logging.getLogger("TradeMetrics")

    def log_trade(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        exit_price: Decimal,
        qty: Decimal,
        entry_time_ms: int,
        exit_time_ms: int,
        reason: str,
    ):
        if not all(
            [
                entry_price > 0,
                exit_price > 0,
                qty > 0,
                entry_time_ms > 0,
                exit_time_ms > 0,
            ]
        ):
            self.logger.warning(
                f"Trade log skipped: Invalid params. EntryPx:{entry_price}, ExitPx:{exit_price}, Qty:{qty}, EntryT:{entry_time_ms}, ExitT:{exit_time_ms}"
            )
            return

        profit_per_unit = exit_price - entry_price
        # Adjust profit calculation for short trades
        if side.lower() == CONFIG.side_sell or side.lower() == CONFIG.pos_short.lower():
            profit_per_unit = (
                entry_price - exit_price
            )  # For shorts, profit if exit_price < entry_price

        profit = profit_per_unit * qty
        entry_dt = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration = exit_dt - entry_dt

        self.trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "profit": profit,
                "entry_time": entry_dt,
                "exit_time": exit_dt,
                "duration_seconds": duration.total_seconds(),
                "exit_reason": reason,
            }
        )
        pnl_color = (
            Fore.GREEN if profit > 0 else (Fore.RED if profit < 0 else Fore.YELLOW)
        )
        self.logger.success(
            f"{Fore.MAGENTA}Trade Recorded: {side.upper()} {qty} {symbol.split('/')[0]} | Entry: {entry_price:.4f}, Exit: {exit_price:.4f} | P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{Style.RESET_ALL} | Duration: {duration} | Reason: {reason}"
        )

    def summary(self) -> str:
        if not self.trades:
            return "No trades recorded."
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t["profit"] > 0)
        losses = sum(1 for t in self.trades if t["profit"] < 0)
        breakeven = total_trades - wins - losses
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_profit = sum(t["profit"] for t in self.trades)
        avg_profit = total_profit / total_trades if total_trades > 0 else Decimal(0)

        summary_str = (
            f"\n{Fore.CYAN}--- Trade Metrics Summary ---\n"
            f"Total Trades: {total_trades} | Wins: {Fore.GREEN}{wins}{Style.RESET_ALL}, Losses: {Fore.RED}{losses}{Style.RESET_ALL}, Breakeven: {Fore.YELLOW}{breakeven}{Style.RESET_ALL}\n"
            f"Win Rate: {win_rate:.2f}% | Total P/L: {(Fore.GREEN if total_profit > 0 else Fore.RED)}{total_profit:.2f} {CONFIG.usdt_symbol}{Style.RESET_ALL}\n"
            f"Avg P/L per Trade: {(Fore.GREEN if avg_profit > 0 else Fore.RED)}{avg_profit:.2f} {CONFIG.usdt_symbol}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}--- End Summary ---{Style.RESET_ALL}"
        )
        self.logger.info(summary_str)  # Log with colors
        # Return a plain string version if needed elsewhere
        plain_summary_str = (
            f"\n--- Trade Metrics Summary ---\n"
            f"Total Trades: {total_trades} | Wins: {wins}, Losses: {losses}, Breakeven: {breakeven}\n"
            f"Win Rate: {win_rate:.2f}% | Total P/L: {total_profit:.2f} {CONFIG.usdt_symbol}\n"
            f"Avg P/L per Trade: {avg_profit:.2f} {CONFIG.usdt_symbol}\n"
            f"--- End Summary ---"
        )
        return plain_summary_str


trade_metrics = TradeMetrics()
_active_trade_details: dict[str, Any] = {
    "entry_price": None,
    "entry_time_ms": None,
    "side": None,
    "qty": None,
}


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, handling None, pd.NA, and conversion errors."""
    if pd.isna(value) or value is None:  # pd.isna handles None, np.nan, pd.NaT, pd.NA
        return default
    try:
        return Decimal(str(value))  # Convert to string first to handle floats correctly
    except (InvalidOperation, TypeError, ValueError):
        logger.debug(  # Changed to debug to reduce noise for common indicator NaNs at start
            f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}"
        )
        return default


def format_order_id(order_id: str | int | None) -> str:
    return str(order_id)[-6:] if order_id else "N/A"


def _format_for_log(value: Any, precision: int = 4, is_bool_trend: bool = False) -> str:
    if pd.isna(value) or value is None:
        return "N/A"
    if is_bool_trend:
        if value is True:
            return f"{Fore.GREEN}Up{Style.RESET_ALL}"
        if value is False:
            return f"{Fore.RED}Down{Style.RESET_ALL}"
        return "N/A (Trend)"  # Should not happen if pd.NA handled above
    if isinstance(value, Decimal):
        return f"{value:.{precision}f}"
    if isinstance(value, (float, int)):
        return f"{float(value):.{precision}f}"  # Ensure it's float for formatting
    if isinstance(value, bool):
        return str(value)
    return str(value)


def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal) -> str:
    try:
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error shaping price {price} for {symbol}: {e}. Using raw Decimal.{Style.RESET_ALL}"
        )
        # Fallback to string representation of Decimal, potentially with rounding
        return str(
            Decimal(str(price))
            .quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
            .normalize()
        )


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal) -> str:
    try:
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error shaping amount {amount} for {symbol}: {e}. Using raw Decimal.{Style.RESET_ALL}"
        )
        return str(
            Decimal(str(amount))
            .quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
            .normalize()
        )


# --- Retry Decorator for API calls ---
@retry(
    tries=CONFIG.retry_count,
    delay=CONFIG.retry_delay_seconds,
    backoff=2,
    logger=logger,
    exceptions=(
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
    ),
)
def safe_api_call(func, *args, **kwargs):
    """Wraps an API call with retry logic for common transient errors."""
    return func(*args, **kwargs)


# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: bool | None = None


def send_sms_alert(message: str) -> bool:
    global _termux_sms_command_exists
    if not CONFIG.enable_sms_alerts:
        return False
    if _termux_sms_command_exists is None:  # Check only once
        _termux_sms_command_exists = shutil.which("termux-sms-send") is not None
        if not _termux_sms_command_exists:
            logger.warning(
                f"{Fore.YELLOW}SMS: 'termux-sms-send' command not found. Install Termux:API and ensure termux-api package is installed in Termux.{Style.RESET_ALL}"
            )

    if not _termux_sms_command_exists or not CONFIG.sms_recipient_number:
        if CONFIG.enable_sms_alerts and not CONFIG.sms_recipient_number:
            logger.debug("SMS sending skipped: Recipient number not configured.")
        return False

    try:
        # Sanitize message slightly for command line (basic)
        safe_message = message.replace('"', "'").replace("`", "'").replace("$", "")
        command: list[str] = [
            "termux-sms-send",
            "-n",
            CONFIG.sms_recipient_number,
            safe_message,
        ]
        logger.info(
            f"{Fore.MAGENTA}Dispatching SMS to {CONFIG.sms_recipient_number}...{Style.RESET_ALL}"
        )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=CONFIG.sms_timeout_seconds,
        )

        if result.returncode == 0:
            logger.success(
                f"{Fore.MAGENTA}SMS dispatched successfully.{Style.RESET_ALL}"
            )
            return True
        else:
            logger.error(
                f"{Fore.RED}SMS dispatch failed. Code: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"{Fore.RED}SMS dispatch timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}"
        )
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}SMS dispatch failed with exception: {e}{Style.RESET_ALL}"
        )
        return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> ccxt.Exchange | None:
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API Key and/or Secret are missing in configuration.{Style.RESET_ALL}"
        )
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing. Bot cannot start.")
        return None
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",  # For USDT perpetuals
                    "adjustForTimeDifference": True,
                    # "brokerId": "YOUR_BROKER_ID" # If using via a broker affiliate program
                },
                "recvWindow": CONFIG.default_recv_window,
            }
        )
        # exchange.set_sandbox_mode(True) # Uncomment for Testnet trading

        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True)  # Force reload
        logger.debug(f"Loaded {len(exchange.markets)} market structures from Bybit.")

        logger.debug(
            "Performing initial balance check to verify API key permissions..."
        )
        # For V5 API, category is often needed for balance and positions
        balance_params = (
            {"category": "linear"}
            if exchange.options.get("defaultType") == "linear"
            else {}
        )
        exchange.fetch_balance(params=balance_params)

        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (V5 API).{Style.RESET_ALL}"
        )
        if exchange.sandbox:
            logger.warning(
                f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! TESTNET MODE ACTIVE !!!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[Pyrmethus/{CONFIG.strategy_name.value}] Portal opened (TESTNET)."
            )
        else:
            logger.warning(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! LIVE TRADING MODE ACTIVE - EXTREME CAUTION !!!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[Pyrmethus/{CONFIG.strategy_name.value}] Portal opened (LIVE)."
            )
        return exchange
    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Portal opening FAILED (Authentication Error): {e}{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Auth FAILED: {e}. Check API keys.")
    except ccxt.NetworkError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Portal opening FAILED (Network Error): {e}{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Network FAILED: {e}.")
    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Portal opening FAILED (Unexpected Error): {e}{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Init FAILED: {type(e).__name__}.")
        logger.debug(traceback.format_exc())
    return None


# --- Indicator Calculation Functions - Scrying the Market ---
def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = ""
) -> pd.DataFrame:
    col_prefix = f"{prefix}" if prefix else ""
    out_supertrend_val = (
        f"{col_prefix}supertrend"  # This column from pandas_ta is the ST line value
    )
    out_trend_direction = f"{col_prefix}trend"  # Boolean: True for Up, False for Down
    out_long_flip = f"{col_prefix}st_long_flip"  # Boolean: True if just flipped to Long
    out_short_flip = (
        f"{col_prefix}st_short_flip"  # Boolean: True if just flipped to Short
    )
    target_cols = [
        out_supertrend_val,
        out_trend_direction,
        out_long_flip,
        out_short_flip,
    ]

    # pandas_ta column names (standardized)
    pta_st_val_col = f"SUPERT_{length}_{float(multiplier)}"  # SuperTrend line value
    pta_st_dir_col = (
        f"SUPERTd_{length}_{float(multiplier)}"  # Direction (1 for long, -1 for short)
    )
    # pta_st_long_level_col = f"SUPERTl_{length}_{float(multiplier)}" # ST line when in long trend
    # pta_st_short_level_col = f"SUPERTs_{length}_{float(multiplier)}"# ST line when in short trend

    min_len_needed_for_pta = (
        length + 1
    )  # ATR period + 1 for pandas_ta to start calculating

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low", "close"])
        or len(df) < min_len_needed_for_pta
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Rows: {len(df) if df is not None else 0}, Need approx: {min_len_needed_for_pta}). Populating with NAs.{Style.RESET_ALL}"
        )
        if df is not None:  # Ensure df exists to add NA columns
            for col in target_cols:
                df[col] = pd.NA
        return (
            df if df is not None else pd.DataFrame(columns=target_cols)
        )  # Return empty DF if input was None

    try:
        # Work on a copy for pandas_ta to avoid SettingWithCopyWarning if df is a slice
        temp_df = df[["high", "low", "close"]].copy()
        temp_df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the expected columns
        pta_cols_to_check = [
            pta_st_val_col,
            pta_st_dir_col,
        ]  # pta_st_long_level_col, pta_st_short_level_col are also created
        if not all(c in temp_df.columns for c in pta_cols_to_check):
            missing_cols = [c for c in pta_cols_to_check if c not in temp_df.columns]
            logger.error(
                f"{Fore.RED}Scrying ({col_prefix}ST): pandas_ta failed to create expected SuperTrend columns: {missing_cols}. Populating with NAs.{Style.RESET_ALL}"
            )
            for col in target_cols:
                df[col] = pd.NA
            return df

        df[out_supertrend_val] = temp_df[pta_st_val_col].apply(
            lambda x: safe_decimal_conversion(x, pd.NA)
        )  # Keep NA for Decimal

        # Trend: True for up (1), False for down (-1), pd.NA for no trend/undefined
        df[out_trend_direction] = pd.NA  # Default to NA
        df.loc[temp_df[pta_st_dir_col] == 1, out_trend_direction] = True
        df.loc[temp_df[pta_st_dir_col] == -1, out_trend_direction] = False

        # Flips: A flip occurs when the direction changes from the previous candle
        # Ensure series align by using .values if mixing pandas and numpy operations or direct assignment
        prev_dir = temp_df[pta_st_dir_col].shift(1)
        df[out_long_flip] = (temp_df[pta_st_dir_col].values == 1) & (
            prev_dir.values == -1
        )
        df[out_short_flip] = (temp_df[pta_st_dir_col].values == -1) & (
            prev_dir.values == 1
        )

        if (
            not df.empty and not df.iloc[-1].isnull().all()
        ):  # Check last row has some data
            last_val = df[out_supertrend_val].iloc[-1]
            last_trend_bool = df[out_trend_direction].iloc[-1]
            last_l_flip = df[out_long_flip].iloc[-1]
            last_s_flip = df[out_short_flip].iloc[-1]

            trend_str = _format_for_log(last_trend_bool, is_bool_trend=True)
            flip_str = (
                "L"
                if last_l_flip
                else (
                    "S"
                    if last_s_flip
                    else (
                        "None"
                        if not pd.isna(last_l_flip) and not pd.isna(last_s_flip)
                        else "N/A"
                    )
                )
            )  # Handle NA in flip bools

            logger.debug(
                f"Scrying ({col_prefix}ST({length},{multiplier})): Val={_format_for_log(last_val)}, Trend={trend_str}, Flip={flip_str}"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying ({col_prefix}ST): Error during calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        if df is not None:  # Ensure df exists to add NA columns
            for col in target_cols:
                df[col] = pd.NA
    return df


def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> dict[str, Decimal | pd.NA | None]:
    results: dict[str, Decimal | pd.NA | None] = {
        "atr": pd.NA,
        "volume_ma": pd.NA,
        "last_volume": pd.NA,
        "volume_ratio": pd.NA,
    }
    min_len_needed = max(atr_len, vol_ma_len) + 1  # General heuristic

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low", "close", "volume"])
        or len(df) < min_len_needed
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data (Rows: {len(df) if df is not None else 0}, Need approx: {min_len_needed}).{Style.RESET_ALL}"
        )
        return results  # Return dict with NAs

    try:
        temp_df = df.copy()  # Work on a copy

        # ATR
        atr_col = f"ATRr_{atr_len}"  # pandas_ta default ATR column name
        temp_df.ta.atr(length=atr_len, append=True)
        if atr_col in temp_df.columns and not temp_df[atr_col].empty:
            results["atr"] = safe_decimal_conversion(temp_df[atr_col].iloc[-1], pd.NA)

        # Volume Analysis
        volume_ma_col = f"volume_sma_{vol_ma_len}"
        temp_df["volume"] = pd.to_numeric(
            temp_df["volume"], errors="coerce"
        )  # Ensure numeric
        temp_df[volume_ma_col] = (
            temp_df["volume"]
            .rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2))
            .mean()
        )

        if not temp_df.empty:
            results["volume_ma"] = safe_decimal_conversion(
                temp_df[volume_ma_col].iloc[-1], pd.NA
            )
            results["last_volume"] = safe_decimal_conversion(
                temp_df["volume"].iloc[-1], pd.NA
            )

            if (
                not pd.isna(results["volume_ma"])
                and not pd.isna(results["last_volume"])
                and results["volume_ma"] > CONFIG.position_qty_epsilon
            ):
                try:
                    results["volume_ratio"] = (
                        results["last_volume"] / results["volume_ma"]
                    )  # type: ignore
                except (DivisionByZero, InvalidOperation):
                    results["volume_ratio"] = pd.NA  # Keep as NA if calculation fails

        log_parts = [
            f"ATR({atr_len})={Fore.CYAN}{_format_for_log(results['atr'], 5)}{Style.RESET_ALL}"
        ]
        if not pd.isna(results["last_volume"]):
            log_parts.append(f"LastVol={_format_for_log(results['last_volume'], 2)}")
        if not pd.isna(results["volume_ma"]):
            log_parts.append(
                f"VolMA({vol_ma_len})={_format_for_log(results['volume_ma'], 2)}"
            )
        if not pd.isna(results["volume_ratio"]):
            log_parts.append(
                f"VolRatio={Fore.YELLOW}{_format_for_log(results['volume_ratio'], 2)}{Style.RESET_ALL}"
            )
        logger.debug(f"Scrying Results (Vol/ATR): {', '.join(log_parts)}")

    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (Vol/ATR): Error during calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        results = {key: pd.NA for key in results}  # Reset to NAs on error
    return results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    # StochRSI needs RSI_len + Stoch_len + d_smooth for full calculation. Momentum needs mom_len.
    min_len_needed = (
        max(rsi_len + stoch_len + d, mom_len) + 10
    )  # Add buffer for stabilization

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len_needed:
        logger.warning(
            f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data (Rows: {len(df) if df is not None else 0}, Need approx: {min_len_needed}). Populating NAs.{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        # StochRSI
        # pandas_ta appends columns like STOCHRSIk_14_14_3_3 and STOCHRSId_14_14_3_3
        stochrsi_df = df.ta.stochrsi(
            length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False
        )  # Calculate separately
        k_col_ta, d_col_ta = (
            f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}",
            f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}",
        )

        if stochrsi_df is not None and not stochrsi_df.empty:
            df["stochrsi_k"] = (
                stochrsi_df[k_col_ta].apply(lambda x: safe_decimal_conversion(x, pd.NA))
                if k_col_ta in stochrsi_df
                else pd.NA
            )
            df["stochrsi_d"] = (
                stochrsi_df[d_col_ta].apply(lambda x: safe_decimal_conversion(x, pd.NA))
                if d_col_ta in stochrsi_df
                else pd.NA
            )
        else:
            df["stochrsi_k"], df["stochrsi_d"] = pd.NA, pd.NA

        # Momentum
        temp_df_mom = df[["close"]].copy()  # Operate on copy for momentum
        mom_col_ta = f"MOM_{mom_len}"  # pandas_ta momentum column name
        temp_df_mom.ta.mom(length=mom_len, append=True)
        df["momentum"] = (
            temp_df_mom[mom_col_ta].apply(lambda x: safe_decimal_conversion(x, pd.NA))
            if mom_col_ta in temp_df_mom
            else pd.NA
        )

        if not df.empty and not df.iloc[-1].isnull().all():
            k_v, d_v, m_v = (
                df["stochrsi_k"].iloc[-1],
                df["stochrsi_d"].iloc[-1],
                df["momentum"].iloc[-1],
            )
            logger.debug(
                f"Scrying (StochRSI/Mom): K={_format_for_log(k_v, 2)}, D={_format_for_log(d_v, 2)}, Mom={_format_for_log(m_v, 4)}"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (StochRSI/Mom): Error during calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    min_len_needed = length + signal + 5  # Heuristic for Ehlers Fisher

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low"])
        or len(df) < min_len_needed
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data (Rows: {len(df) if df is not None else 0}, Need approx: {min_len_needed}). Populating NAs.{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        # Ehlers Fisher Transform
        # pandas_ta appends FISHERT_10_1 and FISHERTs_10_1
        fisher_df = df.ta.fisher(
            length=length, signal=signal, append=False
        )  # Calculate separately
        fish_col_ta, signal_col_ta = (
            f"FISHERT_{length}_{signal}",
            f"FISHERTs_{length}_{signal}",
        )

        if fisher_df is not None and not fisher_df.empty:
            df["ehlers_fisher"] = (
                fisher_df[fish_col_ta].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if fish_col_ta in fisher_df
                else pd.NA
            )
            df["ehlers_signal"] = (
                fisher_df[signal_col_ta].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if signal_col_ta in fisher_df
                else pd.NA
            )
        else:
            df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA

        if not df.empty and not df.iloc[-1].isnull().all():
            f_v, s_v = df["ehlers_fisher"].iloc[-1], df["ehlers_signal"].iloc[-1]
            logger.debug(
                f"Scrying (EhlersFisher({length},{signal})): Fisher={_format_for_log(f_v)}, Signal={_format_for_log(s_v)}"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (EhlersFisher): Error during calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def calculate_ehlers_ma(
    df: pd.DataFrame, fast_len: int, slow_len: int, poles: int
) -> pd.DataFrame:
    target_cols = ["ehlers_ssf_fast", "ehlers_ssf_slow"]
    min_len_needed = max(fast_len, slow_len) + poles + 5  # Heuristic for SSF

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len_needed:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Ehlers SSF MA): Insufficient data (Rows: {len(df) if df is not None else 0}, Need approx: {min_len_needed}). Populating NAs.{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        # Ehlers Super Smoother Filter (as MA)
        # pandas_ta ssf returns a Series, column name typically 'SSF_length_poles'
        ssf_fast_series = df.ta.ssf(length=fast_len, poles=poles, append=False)  # type: ignore
        df["ehlers_ssf_fast"] = (
            ssf_fast_series.apply(lambda x: safe_decimal_conversion(x, pd.NA))
            if ssf_fast_series is not None
            else pd.NA
        )

        ssf_slow_series = df.ta.ssf(length=slow_len, poles=poles, append=False)  # type: ignore
        df["ehlers_ssf_slow"] = (
            ssf_slow_series.apply(lambda x: safe_decimal_conversion(x, pd.NA))
            if ssf_slow_series is not None
            else pd.NA
        )

        if not df.empty and not df.iloc[-1].isnull().all():
            fast_v, slow_v = (
                df["ehlers_ssf_fast"].iloc[-1],
                df["ehlers_ssf_slow"].iloc[-1],
            )
            logger.debug(
                f"Scrying (Ehlers SSF MA({fast_len},{slow_len},p{poles})): Fast={_format_for_log(fast_v)}, Slow={_format_for_log(slow_v)}"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (Ehlers SSF MA): Error during calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> dict[str, Decimal | pd.NA | None]:
    results: dict[str, Decimal | pd.NA | None] = {
        "bid_ask_ratio": pd.NA,
        "spread": pd.NA,
        "best_bid": pd.NA,
        "best_ask": pd.NA,
    }
    if not exchange.has.get(
        "fetchL2OrderBook"
    ):  # fetchL2OrderBook is generally preferred over fetchOrderBook
        logger.warning(
            f"{Fore.YELLOW}OB Scrying: Exchange does not support fetchL2OrderBook for {symbol}.{Style.RESET_ALL}"
        )
        return results

    try:
        # fetch_limit for fetchL2OrderBook is the number of bids/asks levels to fetch
        order_book = safe_api_call(
            exchange.fetch_l2_order_book, symbol, limit=fetch_limit
        )
        bids, asks = order_book.get("bids", []), order_book.get("asks", [])

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}OB Scrying: Empty bids or asks array for {symbol}. Market might be thin or API issue.{Style.RESET_ALL}"
            )
            return results

        results["best_bid"] = (
            safe_decimal_conversion(bids[0][0], pd.NA)
            if bids and len(bids[0]) > 0
            else pd.NA
        )
        results["best_ask"] = (
            safe_decimal_conversion(asks[0][0], pd.NA)
            if asks and len(asks[0]) > 0
            else pd.NA
        )

        if (
            not pd.isna(results["best_bid"])
            and not pd.isna(results["best_ask"])
            and results["best_bid"] > 0
            and results["best_ask"] > 0
        ):  # type: ignore
            results["spread"] = results["best_ask"] - results["best_bid"]  # type: ignore

        # Sum volume up to specified depth
        bid_vol = sum(
            safe_decimal_conversion(b[1], Decimal(0))
            for b in bids[: min(depth, len(bids))]
            if len(b) > 1
        )
        ask_vol = sum(
            safe_decimal_conversion(a[1], Decimal(0))
            for a in asks[: min(depth, len(asks))]
            if len(a) > 1
        )

        if ask_vol > CONFIG.position_qty_epsilon:  # Avoid division by zero or near-zero
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
            except (DivisionByZero, InvalidOperation):
                results["bid_ask_ratio"] = pd.NA  # Keep as NA
        else:
            results["bid_ask_ratio"] = pd.NA  # Undefined if ask volume is negligible

        log_parts = [
            f"BestBid={Fore.GREEN}{_format_for_log(results['best_bid'], 4)}{Style.RESET_ALL}",
            f"BestAsk={Fore.RED}{_format_for_log(results['best_ask'], 4)}{Style.RESET_ALL}",
        ]
        if not pd.isna(results["spread"]):
            log_parts.append(
                f"Spread={Fore.YELLOW}{_format_for_log(results['spread'], 4)}{Style.RESET_ALL}"
            )
        if not pd.isna(results["bid_ask_ratio"]):
            log_parts.append(
                f"Ratio(B/A)={Fore.CYAN}{_format_for_log(results['bid_ask_ratio'], 3)}{Style.RESET_ALL}"
            )
        logger.debug(
            f"OB Scrying (Depth {depth}, Fetched {fetch_limit}): {', '.join(log_parts)}"
        )

    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}OB Scrying Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        # Reset to NAs on error
        results = {key: pd.NA for key in results}
    return results


# --- Data Fetching & Caching - Gathering Etheric Data Streams ---
_last_market_data: pd.DataFrame | None = None
_last_fetch_timestamp: float = 0.0


def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int
) -> pd.DataFrame | None:
    global _last_market_data, _last_fetch_timestamp
    current_time = time.time()

    candle_duration_seconds: int = 0
    try:
        candle_duration_seconds = exchange.parse_timeframe(interval)  # in seconds
    except Exception as e:
        logger.warning(
            f"Could not parse timeframe '{interval}' for caching via exchange.parse_timeframe: {e}. Cache duration check might be affected."
        )
        # Fallback logic or disable cache for this call if duration is critical and unknown
        # For simplicity, if parse_timeframe fails, we might fetch fresh data more often or disable cache.
        # Setting to 0 effectively disables time-based cache validation for this call.
        candle_duration_seconds = 0

    cache_is_valid = False
    if (
        _last_market_data is not None
        and not _last_market_data.empty
        and len(_last_market_data) >= limit
    ):
        if candle_duration_seconds > 0:
            # Check if current time is within the (almost) full duration of the last fetched candle
            # This assumes the last candle in _last_market_data is the *current forming* candle
            # A more robust cache would check the timestamp of the last candle.
            time_since_last_fetch = current_time - _last_fetch_timestamp
            if time_since_last_fetch < (
                candle_duration_seconds * float(CONFIG.cache_candle_duration_multiplier)
            ):
                cache_is_valid = True
        # If candle_duration_seconds is 0 (e.g. parse_timeframe failed), cache won't be time-valid, forces fetch.

    if cache_is_valid:
        logger.debug(
            f"Data Fetch: Using CACHED market data ({len(_last_market_data)} candles) for {symbol}. Last fetch: {time.strftime('%H:%M:%S', time.localtime(_last_fetch_timestamp))}"
        )
        return (
            _last_market_data.copy()
        )  # Return a copy to prevent modification of cached data

    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV method.{Style.RESET_ALL}"
        )
        return None

    try:
        logger.debug(
            f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} (Timeframe: {interval})..."
        )
        # Bybit V5 API might need since/limit or specific params for pagination if limit is very large.
        # CCXT usually handles this.
        params = (
            {"category": "linear"}
            if exchange.options.get("defaultType") == "linear"
            else {}
        )
        ohlcv = safe_api_call(
            exchange.fetch_ohlcv, symbol, timeframe=interval, limit=limit, params=params
        )

        if not ohlcv:  # Empty list if no data
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Market may be inactive or API issue.{Style.RESET_ALL}"
            )
            return None  # Return None, not empty DataFrame, to distinguish from valid but empty data.

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        )  # Ensure UTC for consistency
        df.set_index("timestamp", inplace=True)

        # Convert OHLCV columns to numeric, coercing errors to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle NaNs that might have resulted from coercion or were in original data
        if df.isnull().values.any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: NaNs found in OHLCV data for columns: {nan_cols}. Applying ffill then bfill.{Style.RESET_ALL}"
            )
            df.ffill(inplace=True)  # Forward fill first
            df.bfill(
                inplace=True
            )  # Backward fill remaining NaNs (usually at the beginning)

            if df.isnull().values.any():  # Check again after filling
                logger.error(
                    f"{Fore.RED}Data Fetch: Unfillable NaNs remain in OHLCV data after ffill/bfill. Data quality issue for {symbol}.{Style.RESET_ALL}"
                )
                return None  # Data is unusable

        _last_market_data = df.copy()  # Cache the clean data
        _last_fetch_timestamp = current_time
        logger.debug(
            f"Data Fetch: Successfully woven {len(df)} OHLCV candles for {symbol}. Data cached."
        )
        return df.copy()  # Return a copy

    except Exception as e:
        logger.error(
            f"{Fore.RED}Data Fetch: Error processing OHLCV for {symbol} ({interval}): {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return None


# --- Account Health Check ---
def check_account_health(exchange: ccxt.Exchange, config: Config) -> bool:
    logger.debug("Performing account health check...")
    try:
        # For Bybit V5, 'category' is needed for linear/inverse specific balances.
        balance_params = (
            {"category": "linear"}
            if exchange.options.get("defaultType") == "linear"
            else {}
        )
        balance = safe_api_call(exchange.fetch_balance, params=balance_params)

        # Assuming USDT is the collateral currency from CONFIG.usdt_symbol
        usdt_balance = balance.get(config.usdt_symbol, {})
        if not usdt_balance:  # Check if USDT key exists and has data
            logger.error(
                f"Account Health: '{config.usdt_symbol}' balance data not found in response. Full balance: {balance}"
            )
            return False  # Cannot assess health

        # Bybit V5 keys for linear might be different, e.g. 'equity', 'unrealisedPnl', 'availableBalance', 'usedMargin'
        # CCXT abstracts some of this into 'total', 'free', 'used'.
        # For Bybit V5 linear, 'total' usually means walletBalance + unrealisedPnl (equity).
        # 'used' is position margin + order margin.
        total_equity = safe_decimal_conversion(
            usdt_balance.get("total")
        )  # Total equity
        used_margin = safe_decimal_conversion(
            usdt_balance.get("used")
        )  # Margin used for positions/orders

        if total_equity.is_nan() or used_margin.is_nan():  # Check if conversion failed
            logger.warning(
                f"Account Health: Could not determine equity or used margin. Total: {usdt_balance.get('total')}, Used: {usdt_balance.get('used')}"
            )
            return False  # Unhealthy or unknown

        if total_equity <= Decimal("0"):
            logger.warning(
                f"Account Health: Total equity is {_format_for_log(total_equity, 2)} {config.usdt_symbol}. Margin ratio calculation skipped."
            )
            if used_margin > Decimal("0"):
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL HEALTH: Zero/Negative Equity ({_format_for_log(total_equity, 2)}) with Used Margin ({_format_for_log(used_margin, 2)})! Halting trading logic.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{config.symbol.split('/')[0]}/{config.strategy_name.value}] CRITICAL: Zero/Neg Equity with Used Margin. BOT PAUSED."
                )
                return False  # Critically unhealthy
            return True  # No used margin, so technically "healthy" but no funds.

        margin_ratio = (
            used_margin / total_equity if total_equity > 0 else Decimal("Infinity")
        )  # Avoid DivByZero
        health_color = (
            Fore.GREEN if margin_ratio <= config.max_account_margin_ratio else Fore.RED
        )

        logger.info(
            f"Account Health: Equity={_format_for_log(total_equity, 2)}, UsedMargin={_format_for_log(used_margin, 2)}, MarginRatio={health_color}{margin_ratio:.2%}{Style.RESET_ALL} (Max: {config.max_account_margin_ratio:.0%})"
        )

        if margin_ratio > config.max_account_margin_ratio:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL HEALTH: High margin ratio {margin_ratio:.2%} exceeds max {config.max_account_margin_ratio:.0%}. Halting trading logic.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{config.symbol.split('/')[0]}/{config.strategy_name.value}] CRITICAL: High margin ratio {margin_ratio:.2%}. BOT PAUSED."
            )
            return False  # Unhealthy
        return True  # Healthy

    except Exception as e:
        logger.error(
            f"Account health check failed with exception: {type(e).__name__} - {e}"
        )
        logger.debug(traceback.format_exc())
        # On error, assume unhealthy or uncertain to be safe
        send_sms_alert(
            f"[{config.symbol.split('/')[0]}/{config.strategy_name.value}] WARNING: Account health check FAILED: {type(e).__name__}."
        )
        return False


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    default_pos: dict[str, Any] = {
        "side": CONFIG.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
    }
    try:
        market = exchange.market(symbol)
        market_id = market["id"]  # Use 'id' for API calls

        # Determine category for Bybit V5 API (linear/inverse)
        category = "linear"  # Default based on script's focus
        if market.get("linear"):
            category = "linear"
        elif market.get("inverse"):
            category = "inverse"
        else:
            # Fallback if market type not explicitly linear/inverse in CCXT market structure
            # This might happen for older CCXT versions or if market data is unusual
            logger.warning(
                f"{Fore.YELLOW}Pos Check: Market type for {symbol} not explicitly linear/inverse. Assuming 'linear'.{Style.RESET_ALL}"
            )

        params = {"category": category, "symbol": market_id}
        logger.debug(f"Pos Check: Fetching positions with params: {params}")
        fetched_positions = safe_api_call(
            exchange.fetch_positions, symbols=[symbol], params=params
        )

        if not fetched_positions:  # Empty list if no positions
            logger.info(
                f"{Fore.BLUE}Pos Check: No positions returned for {market_id} with category '{category}'. Assumed Flat.{Style.RESET_ALL}"
            )
            return default_pos

        # Iterate through positions; Bybit V5 returns a list.
        # For One-Way mode, there should be only one relevant entry per symbol.
        for pos in fetched_positions:
            pos_info = pos.get("info", {})  # Raw exchange data

            # Match symbol (important if fetch_positions was called without symbols=[symbol])
            if pos_info.get("symbol") != market_id:
                continue

            # Bybit V5 One-Way mode: positionIdx is 0 for buy/sell combined position.
            # positionIdx 1 for Buy side hedge mode, 2 for Sell side hedge mode.
            if (
                int(pos_info.get("positionIdx", -1)) == 0
            ):  # Ensure One-Way mode position
                size_str = pos_info.get("size", "0")
                size_dec = safe_decimal_conversion(size_str)

                if (
                    size_dec > CONFIG.position_qty_epsilon
                ):  # If position has significant size
                    entry_price_str = pos_info.get("avgPrice")  # or "entryPrice"
                    entry_price = safe_decimal_conversion(entry_price_str)

                    # 'side' in Bybit V5 position info: "Buy" for Long, "Sell" for Short
                    bybit_side = pos_info.get("side")
                    if bybit_side == "Buy":
                        side = CONFIG.pos_long
                    elif bybit_side == "Sell":
                        side = CONFIG.pos_short
                    else:  # Should not happen for an active position
                        logger.warning(
                            f"{Fore.YELLOW}Pos Check: Unknown side '{bybit_side}' for {market_id}. Treating as flat.{Style.RESET_ALL}"
                        )
                        continue  # Check next entry in fetched_positions if any

                    pos_color = Fore.GREEN if side == CONFIG.pos_long else Fore.RED
                    logger.info(
                        f"{pos_color}Pos Check: ACTIVE {side} position found. Qty={_format_for_log(size_dec, 8)} @ Entry={_format_for_log(entry_price, 4)}{Style.RESET_ALL}"
                    )
                    return {"side": side, "qty": size_dec, "entry_price": entry_price}

        logger.info(
            f"{Fore.BLUE}Pos Check: No active One-Way position for {market_id} (category '{category}'). Flat.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Pos Check: Error fetching or processing position for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return default_pos  # Return default (flat) on error


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    logger.info(
        f"{Fore.CYAN}Leverage: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}"
    )
    try:
        market = exchange.market(symbol)
        if not market or not market.get("contract"):
            logger.error(
                f"{Fore.RED}Leverage: Market '{symbol}' is not a contract market or not found.{Style.RESET_ALL}"
            )
            return False

        # Bybit V5 set_leverage requires buyLeverage and sellLeverage, and category.
        # CCXT abstracts this; leverage applies to both.
        # Params might be needed if symbol is ambiguous or for specific modes.
        category = "linear"  # As per bot's focus
        params = {
            "category": category,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }

        response = safe_api_call(
            exchange.set_leverage, leverage=leverage, symbol=symbol, params=params
        )
        # Successful response might be empty or contain confirmation.
        logger.success(
            f"{Fore.GREEN}Leverage: Successfully set to {leverage}x for {symbol} (Category: {category}). Response: {response}{Style.RESET_ALL}"
        )
        return True
    except ccxt.ExchangeError as e:
        # Check for "leverage not modified" or similar messages
        err_str = str(e).lower()
        # Bybit V5 error codes: 110044 for "Set leverage not modified"
        if any(
            sub in err_str
            for sub in ["leverage not modified", "same leverage", "110044"]
        ):
            logger.info(
                f"{Fore.CYAN}Leverage: Already set to {leverage}x for {symbol} (or no change needed).{Style.RESET_ALL}"
            )
            return True
        logger.error(
            f"{Fore.RED}Leverage: Exchange error setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}"
        )
    except Exception as e_unexp:
        logger.error(
            f"{Fore.RED}Leverage: Unexpected error setting leverage for {symbol}: {e_unexp}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return False


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    position_to_close: dict[str, Any],
    reason: str = "Signal",
) -> dict[str, Any] | None:
    global _active_trade_details  # Used for logging trade P/L via TradeMetrics

    initial_side = position_to_close.get("side", CONFIG.pos_none)
    initial_qty = position_to_close.get("qty", Decimal("0.0"))
    market_base = symbol.split("/")[0].split(":")[0]  # e.g., BTC from BTC/USDT:USDT

    logger.info(
        f"{Fore.YELLOW}Banish Position: Attempting to close {symbol} ({reason}). Initial details: Side={initial_side}, Qty={_format_for_log(initial_qty, 8)}{Style.RESET_ALL}"
    )

    # Re-validate the current live position before acting
    live_position = get_current_position(exchange, symbol)
    if (
        live_position["side"] == CONFIG.pos_none
        or live_position["qty"] <= CONFIG.position_qty_epsilon
    ):
        logger.warning(
            f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position for {symbol}. Aborting close attempt.{Style.RESET_ALL}"
        )
        # If we thought there was a trade (_active_trade_details set), but now it's gone, clear details.
        # This might happen if SL/TSL triggered externally or due to race condition.
        if _active_trade_details.get("entry_price") is not None:
            logger.info(
                "Clearing potentially stale active trade details as position is now confirmed flat."
            )
            _active_trade_details = {
                "entry_price": None,
                "entry_time_ms": None,
                "side": None,
                "qty": None,
            }
        return None  # No position to close

    # Determine side for close order (opposite of current live position)
    side_to_execute = (
        CONFIG.side_sell
        if live_position["side"] == CONFIG.pos_long
        else CONFIG.side_buy
    )
    amount_to_close_str = format_amount(
        exchange, symbol, live_position["qty"]
    )  # Use live qty

    try:
        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}Banish Position: Sending CLOSE order for {live_position['side']} ({reason}): {side_to_execute.upper()} MARKET {amount_to_close_str} {symbol}...{Style.RESET_ALL}"
        )

        # Bybit V5: 'reduceOnly' is key for closing. 'category' for linear/inverse.
        # 'positionIdx: 0' for One-Way mode.
        params = {"reduceOnly": True, "category": "linear", "positionIdx": 0}
        order = safe_api_call(
            exchange.create_market_order,
            symbol=symbol,
            side=side_to_execute,
            amount=float(amount_to_close_str),
            params=params,
        )

        # Market orders are often filled immediately. Check status and filled amount.
        # CCXT might normalize 'status' to 'closed' if fully filled.
        status = order.get("status", "unknown")
        filled_qty_closed = safe_decimal_conversion(order.get("filled", "0"))
        avg_fill_price_closed = safe_decimal_conversion(
            order.get("average", "0")
        )  # Average fill price of this close order

        # Check if the close order fully filled the intended quantity
        if (
            status == "closed"
            and abs(filled_qty_closed - live_position["qty"])
            < CONFIG.position_qty_epsilon
        ):
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}Banish Position: CONFIRMED FILLED for {symbol} ({reason}). Order ID: ...{format_order_id(order.get('id'))}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name.value}] BANISHED {live_position['side']} {amount_to_close_str} ({reason}). ExitPx: {_format_for_log(avg_fill_price_closed, 4)}"
            )

            # Log to TradeMetrics if we have entry details
            if (
                _active_trade_details.get("entry_price") is not None
                and order.get("timestamp") is not None
            ):
                # Ensure all required details for TradeMetrics are valid
                if _active_trade_details.get("side") and _active_trade_details.get(
                    "qty"
                ):
                    trade_metrics.log_trade(
                        symbol=symbol,
                        side=_active_trade_details[
                            "side"
                        ],  # Original side of the trade
                        entry_price=_active_trade_details["entry_price"],
                        exit_price=avg_fill_price_closed,  # Price from this close order
                        qty=_active_trade_details["qty"],  # Original entry quantity
                        entry_time_ms=_active_trade_details["entry_time_ms"],
                        exit_time_ms=order[
                            "timestamp"
                        ],  # Timestamp of this close order
                        reason=reason,
                    )
                else:
                    logger.warning(
                        f"TradeMetrics: Skipping log for {symbol} due to missing original side/qty in _active_trade_details."
                    )
            else:
                logger.info(
                    f"TradeMetrics: Skipping log for {symbol} as no active entry details were found or exit order timestamp missing."
                )

            # Reset active trade details as position is now closed
            _active_trade_details = {
                "entry_price": None,
                "entry_time_ms": None,
                "side": None,
                "qty": None,
            }
            return order  # Return the successful close order
        else:
            # Partial fill or unexpected status
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Fill status uncertain for {symbol}. Expected Qty: {live_position['qty']}, Filled: {filled_qty_closed}. Order ID: ...{format_order_id(order.get('id'))}, Status: {status}. Manual check may be needed.{Style.RESET_ALL}"
            )
            # Consider _active_trade_details state here. If partially closed, it's complex.
            # For simplicity, assume full close or full failure of this attempt.
            # If it's a critical failure to close, further action might be needed (e.g. retry, alert more strongly).
            return order  # Return order for further checks by caller if needed

    except Exception as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Exception during close attempt for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] ERROR Banishing {initial_side} ({reason}): {type(e).__name__}."
        )
        logger.debug(traceback.format_exc())
    return None  # Return None if close attempt failed due to exception


def calculate_position_size(
    equity: Decimal,
    risk_pct: Decimal,
    entry_px: Decimal,
    sl_px: Decimal,
    lev: int,
    sym: str,
    ex: ccxt.Exchange,
) -> tuple[Decimal | None, Decimal | None]:
    if not (entry_px > 0 and sl_px > 0 and 0 < risk_pct < 1 and equity > 0 and lev > 0):
        logger.error(
            f"{Fore.RED}RiskCalc: Invalid inputs for position sizing (Equity: {equity}, Risk%: {risk_pct}, Entry: {entry_px}, SL: {sl_px}, Lev: {lev}).{Style.RESET_ALL}"
        )
        return None, None

    price_diff_per_unit = abs(entry_px - sl_px)  # Potential loss per unit if SL hits
    if (
        price_diff_per_unit < CONFIG.position_qty_epsilon
    ):  # Avoid division by zero if entry and SL are identical
        logger.error(
            f"{Fore.RED}RiskCalc: Entry price and Stop Loss price are too close or identical. Cannot calculate position size.{Style.RESET_ALL}"
        )
        return None, None

    try:
        # Amount of USDT to risk on this trade
        risk_amt_usdt = equity * risk_pct

        # Quantity based on risk amount and price difference (stop distance)
        # Qty = Risk_USDT / (Price_Diff_Per_Unit)
        qty_raw = risk_amt_usdt / price_diff_per_unit

        # Adjust quantity to exchange's precision rules
        qty_prec_str = format_amount(
            ex, sym, qty_raw
        )  # Uses exchange.amount_to_precision
        qty_prec = Decimal(qty_prec_str)

        if (
            qty_prec <= CONFIG.position_qty_epsilon
        ):  # If calculated quantity is too small (e.g., due to high price_diff or low risk_amt)
            logger.warning(
                f"{Fore.YELLOW}RiskCalc: Calculated quantity ({_format_for_log(qty_prec, 8)}) is negligible or zero after precision adjustment. Min risk not met or stop too wide.{Style.RESET_ALL}"
            )
            return None, None

        # Calculate estimated position value and margin required
        position_value_usdt = qty_prec * entry_px
        margin_required = position_value_usdt / Decimal(
            lev
        )  # Margin = (Qty * EntryPx) / Leverage

        logger.debug(
            f"RiskCalc: Equity={_format_for_log(equity, 2)}, Risk%={risk_pct:.2%}, RiskAmt={_format_for_log(risk_amt_usdt, 2)}"
        )
        logger.debug(
            f"RiskCalc: EntryPx={_format_for_log(entry_px, 4)}, SLPx={_format_for_log(sl_px, 4)}, PriceDiff={_format_for_log(price_diff_per_unit, 4)}"
        )
        logger.debug(
            f"RiskCalc: RawQty={_format_for_log(qty_raw, 8)}, PreciseQty={Fore.CYAN}{_format_for_log(qty_prec, 8)}{Style.RESET_ALL}, PosValue={_format_for_log(position_value_usdt, 2)}, MarginReq={_format_for_log(margin_required, 4)}"
        )

        return qty_prec, margin_required
    except (DivisionByZero, InvalidOperation, Exception) as e:
        logger.error(
            f"{Fore.RED}RiskCalc: Error during position size calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return None, None


def wait_for_order_fill(
    exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_s: int
) -> dict[str, Any] | None:
    start_time = time.time()
    oid_short = format_order_id(order_id)
    logger.info(
        f"{Fore.CYAN}Observing order ...{oid_short} ({symbol}) for fill (Timeout: {timeout_s}s)...{Style.RESET_ALL}"
    )

    # Bybit V5 may need category for fetch_order
    params = (
        {"category": "linear"}
        if exchange.options.get("defaultType") == "linear"
        else {}
    )

    while time.time() - start_time < timeout_s:
        try:
            order = safe_api_call(exchange.fetch_order, order_id, symbol, params=params)
            status = order.get(
                "status"
            )  # CCXT normalized status: 'open', 'closed', 'canceled', etc.

            if (
                status == "closed"
            ):  # 'closed' usually means fully filled for market/limit orders that executed
                logger.success(
                    f"{Fore.GREEN}Order ...{oid_short} ({symbol}) confirmed FILLED/CLOSED.{Style.RESET_ALL}"
                )
                return order
            if status in ["canceled", "rejected", "expired"]:
                logger.error(
                    f"{Fore.RED}Order ...{oid_short} ({symbol}) FAILED with status: {status}.{Style.RESET_ALL}"
                )
                return order

            # If still 'open' or other transient status, wait and retry
            time.sleep(0.75)  # Polling interval

        except ccxt.OrderNotFound:
            logger.warning(
                f"{Fore.YELLOW}Order ...{oid_short} ({symbol}) not found. Possible propagation delay. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(1.5)  # Longer sleep if OrderNotFound, might be propagation
        except Exception as e:  # Catch other potential errors during fetch_order
            logger.warning(
                f"{Fore.YELLOW}Error checking order ...{oid_short} ({symbol}): {type(e).__name__}. Retrying...{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            time.sleep(2)  # Sleep on other errors before retrying fetch

    logger.error(
        f"{Fore.RED}Order ...{oid_short} ({symbol}) fill check TIMEOUT after {timeout_s}s.{Style.RESET_ALL}"
    )
    try:  # One final attempt to fetch the order status after timeout
        return safe_api_call(exchange.fetch_order, order_id, symbol, params=params)
    except Exception as e_final:
        logger.error(
            f"{Fore.RED}Final check for order ...{oid_short} ({symbol}) also failed: {type(e_final).__name__}{Style.RESET_ALL}"
        )
        return None


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: Decimal,
    current_atr: Decimal | pd.NA | None,
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal,
) -> dict[str, Any] | None:
    global _active_trade_details
    market_base = symbol.split("/")[0].split(":")[0]
    order_side_color = Fore.GREEN if side == CONFIG.side_buy else Fore.RED
    logger.info(
        f"{order_side_color}{Style.BRIGHT}Place Order Ritual: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}"
    )

    if pd.isna(current_atr) or current_atr is None or current_atr <= 0:
        logger.error(
            f"{Fore.RED}Place Order Error: Invalid ATR ({current_atr}). Cannot calculate Stop Loss or position size.{Style.RESET_ALL}"
        )
        return None

    v5_category = "linear"  # Assuming linear contracts as per bot's focus

    try:
        # --- Pre-computation and Checks ---
        balance = safe_api_call(
            exchange.fetch_balance, params={"category": v5_category}
        )
        market = exchange.market(symbol)
        min_qty_allowed = safe_decimal_conversion(
            market.get("limits", {}).get("amount", {}).get("min"), Decimal("0")
        )

        usdt_equity = safe_decimal_conversion(
            balance.get(CONFIG.usdt_symbol, {}).get("total")
        )
        usdt_free = safe_decimal_conversion(
            balance.get(CONFIG.usdt_symbol, {}).get("free")
        )

        if (
            usdt_equity <= 0 or usdt_free < 0
        ):  # free can be slightly negative due to fees sometimes
            logger.error(
                f"{Fore.RED}Place Order Error: Invalid equity ({_format_for_log(usdt_equity, 2)}) or free margin ({_format_for_log(usdt_free, 2)}).{Style.RESET_ALL}"
            )
            return None

        # Estimate entry price (use OB or last price)
        ob_data = analyze_order_book(
            exchange,
            symbol,
            CONFIG.shallow_ob_fetch_depth,
            CONFIG.shallow_ob_fetch_depth,
        )
        entry_px_est = (
            ob_data.get("best_ask")
            if side == CONFIG.side_buy
            else ob_data.get("best_bid")
        )
        if pd.isna(entry_px_est) or entry_px_est is None or entry_px_est <= 0:
            ticker = safe_api_call(
                exchange.fetch_ticker, symbol
            )  # Fallback to last price
            entry_px_est = safe_decimal_conversion(ticker.get("last"))
        if pd.isna(entry_px_est) or entry_px_est is None or entry_px_est <= 0:
            logger.error(
                f"{Fore.RED}Place Order Error: Failed to get a valid entry price estimate for {symbol}.{Style.RESET_ALL}"
            )
            return None

        # Calculate initial SL price based on ATR and estimated entry
        sl_distance = current_atr * sl_atr_multiplier
        sl_px_raw_est = (
            (entry_px_est - sl_distance)
            if side == CONFIG.side_buy
            else (entry_px_est + sl_distance)
        )
        sl_px_est_str = format_price(
            exchange, symbol, sl_px_raw_est
        )  # Format to exchange precision
        sl_px_est = Decimal(sl_px_est_str)
        if sl_px_est <= 0:
            logger.error(
                f"{Fore.RED}Place Order Error: Invalid estimated SL price ({_format_for_log(sl_px_est, 4)}) after calculation.{Style.RESET_ALL}"
            )
            return None

        # Calculate position size and margin
        final_qty, margin_est = calculate_position_size(
            usdt_equity,
            risk_percentage,
            entry_px_est,
            sl_px_est,
            leverage,
            symbol,
            exchange,
        )
        if final_qty is None or margin_est is None or final_qty <= 0 or margin_est <= 0:
            logger.error(
                f"{Fore.RED}Place Order Error: Position size calculation failed or resulted in zero/negative qty/margin.{Style.RESET_ALL}"
            )
            return None

        # Apply max order cap
        estimated_pos_value = final_qty * entry_px_est
        if estimated_pos_value > max_order_cap_usdt:
            logger.warning(
                f"{Fore.YELLOW}Place Order: Estimated position value {estimated_pos_value:.2f} USDT exceeds cap {max_order_cap_usdt:.2f} USDT. Adjusting quantity.{Style.RESET_ALL}"
            )
            final_qty = Decimal(
                format_amount(exchange, symbol, max_order_cap_usdt / entry_px_est)
            )  # Recalculate qty based on cap
            margin_est = (final_qty * entry_px_est) / Decimal(
                leverage
            )  # Recalculate margin
            if final_qty <= 0:
                logger.error(
                    f"{Fore.RED}Place Order: Quantity became zero after cap adjustment.{Style.RESET_ALL}"
                )
                return None
            logger.info(
                f"Place Order: Quantity adjusted to {_format_for_log(final_qty, 8)} due to max cap."
            )

        # Check against min quantity and available margin
        if min_qty_allowed > 0 and final_qty < min_qty_allowed:
            logger.error(
                f"{Fore.RED}Place Order Error: Calculated qty {_format_for_log(final_qty, 8)} is less than min allowed {_format_for_log(min_qty_allowed, 8)} for {symbol}.{Style.RESET_ALL}"
            )
            return None
        if usdt_free < margin_est * margin_check_buffer:
            logger.error(
                f"{Fore.RED}Place Order Error: Insufficient FREE margin. Need ~{_format_for_log(margin_est * margin_check_buffer, 2)}, Have {_format_for_log(usdt_free, 2)}.{Style.RESET_ALL}"
            )
            return None

        # --- Place Entry Order ---
        entry_params = {
            "reduceOnly": False,
            "category": v5_category,
            "positionIdx": 0,
        }  # For One-Way mode entry
        entry_order = safe_api_call(
            exchange.create_market_order,
            symbol=symbol,
            side=side,
            amount=float(final_qty),
            params=entry_params,
        )
        order_id = entry_order.get("id")
        if not order_id:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Entry order for {symbol} did NOT return an ID! Order details: {entry_order}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL: Entry order NO ID for {symbol}!"
            )
            return None  # Cannot proceed without order ID

        filled_entry_order = wait_for_order_fill(
            exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds
        )
        if not filled_entry_order or filled_entry_order.get("status") != "closed":
            logger.error(
                f"{Fore.RED}Entry order ...{format_order_id(order_id)} for {symbol} not filled or failed. Status: {filled_entry_order.get('status') if filled_entry_order else 'timeout'}.{Style.RESET_ALL}"
            )
            try:  # Attempt to cancel if it's somehow still open and not filled
                if filled_entry_order and filled_entry_order.get("status") == "open":
                    safe_api_call(
                        exchange.cancel_order,
                        order_id,
                        symbol,
                        params={"category": v5_category},
                    )
                    logger.info(
                        f"Attempted to cancel unfilled/open entry order ...{format_order_id(order_id)}."
                    )
            except Exception as e_cancel:
                logger.warning(
                    f"Could not cancel potentially unfilled order ...{format_order_id(order_id)}: {e_cancel}"
                )
            return None

        avg_fill_px = safe_decimal_conversion(filled_entry_order.get("average"))
        filled_qty_val = safe_decimal_conversion(filled_entry_order.get("filled"))
        entry_timestamp_ms = filled_entry_order.get("timestamp")

        if (
            filled_qty_val <= CONFIG.position_qty_epsilon
            or avg_fill_px <= CONFIG.position_qty_epsilon
            or entry_timestamp_ms is None
        ):
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill data for entry order ...{format_order_id(order_id)}. Qty: {filled_qty_val}, Px: {avg_fill_px}, Ts: {entry_timestamp_ms}.{Style.RESET_ALL}"
            )
            return None  # Cannot proceed with invalid fill data

        logger.success(
            f"{order_side_color}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {_format_for_log(filled_qty_val, 8)} {symbol.split('/')[0]} @ AvgPx: {_format_for_log(avg_fill_px, 4)}{Style.RESET_ALL}"
        )

        # IMPORTANT: Update active trade details now that entry is confirmed
        _active_trade_details = {
            "entry_price": avg_fill_px,
            "entry_time_ms": entry_timestamp_ms,
            "side": side,
            "qty": filled_qty_val,
        }

        # --- Place Protection Orders (SL & TSL) ---
        sl_placed_successfully = False
        tsl_placed_successfully = (
            False  # Assume TSL is optional or best-effort if fixed SL is primary
        )

        # Calculate actual SL price based on actual average fill price
        actual_sl_px_raw = (
            (avg_fill_px - sl_distance)
            if side == CONFIG.side_buy
            else (avg_fill_px + sl_distance)
        )
        actual_sl_px_str = format_price(exchange, symbol, actual_sl_px_raw)
        if Decimal(actual_sl_px_str) <= 0:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Actual SL price ({actual_sl_px_str}) is zero or negative! Cannot place SL.{Style.RESET_ALL}"
            )
            # This is a critical failure after entry. Trigger emergency close.
        else:
            sl_order_side = (
                CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            )
            # For Bybit V5 Stop Market: type='StopMarket', stopPrice is trigger. Amount is contract qty.
            sl_params = {
                "category": v5_category,
                "stopPrice": float(actual_sl_px_str),  # Trigger price for stop market
                "reduceOnly": True,
                "positionIdx": 0,  # For One-Way mode
                # "tpslMode": "Full" # or "Partial" - for TP/SL on entire position
                # "slOrderType": "Market" (default for stopMarket) or "Limit"
            }
            try:
                logger.info(
                    f"Placing Fixed SL order: Side={sl_order_side}, Qty={float(filled_qty_val)}, TriggerPx={actual_sl_px_str}"
                )
                sl_order = safe_api_call(
                    exchange.create_order,
                    symbol,
                    "StopMarket",
                    sl_order_side,
                    float(filled_qty_val),
                    price=None,
                    params=sl_params,
                )
                logger.success(
                    f"{Fore.GREEN}Fixed SL placed successfully. ID:...{format_order_id(sl_order.get('id'))}, TriggerPx: {actual_sl_px_str}{Style.RESET_ALL}"
                )
                sl_placed_successfully = True
            except Exception as e_sl:
                logger.error(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: FAILED to place Fixed SL order: {e_sl}{Style.RESET_ALL}"
                )
                logger.debug(traceback.format_exc())
                # sl_placed_successfully remains False

        # Place Trailing Stop Loss (TSL) if configured
        if (
            CONFIG.trailing_stop_percentage > 0 and sl_placed_successfully
        ):  # Only place TSL if fixed SL was attempted (even if failed, to see if TSL works) or if SL succeeded
            tsl_activation_offset_val = avg_fill_px * tsl_activation_offset_percent
            tsl_activation_price_raw = (
                (avg_fill_px + tsl_activation_offset_val)
                if side == CONFIG.side_buy
                else (avg_fill_px - tsl_activation_offset_val)
            )
            tsl_activation_price_str = format_price(
                exchange, symbol, tsl_activation_price_raw
            )

            # Bybit V5 TSL: trailingStop is percentage (e.g., "0.1" for 0.1%). activePrice is trigger for TSL to activate.
            # Ensure trailing_stop_percentage (e.g. 0.005 for 0.5%) is converted to string "0.5" for API.
            tsl_value_for_api = str(
                (tsl_percent * Decimal("100")).normalize()
            )  # e.g., 0.005 -> "0.5"

            if Decimal(tsl_activation_price_str) <= 0:
                logger.error(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: TSL Activation Price ({tsl_activation_price_str}) is zero or negative! Cannot place TSL.{Style.RESET_ALL}"
                )
            else:
                tsl_params = {
                    "category": v5_category,
                    "trailingStop": tsl_value_for_api,  # Percentage value as string, e.g. "0.5" for 0.5%
                    "activePrice": float(
                        tsl_activation_price_str
                    ),  # Price at which TSL activates
                    "reduceOnly": True,
                    "positionIdx": 0,  # For One-Way mode
                }
                try:
                    tsl_order_side = sl_order_side  # Same side as fixed SL
                    logger.info(
                        f"Placing Trailing SL order: Side={tsl_order_side}, Qty={float(filled_qty_val)}, TrailValue={tsl_value_for_api}%, ActivationPx={tsl_activation_price_str}"
                    )
                    tsl_order = safe_api_call(
                        exchange.create_order,
                        symbol,
                        "StopMarket",
                        tsl_order_side,
                        float(filled_qty_val),
                        price=None,
                        params=tsl_params,
                    )
                    logger.success(
                        f"{Fore.GREEN}Trailing SL placed successfully. ID:...{format_order_id(tsl_order.get('id'))}, Trail: {tsl_value_for_api}%, ActivateAt: {tsl_activation_price_str}{Style.RESET_ALL}"
                    )
                    tsl_placed_successfully = (
                        True  # Or some other logic if TSL is optional
                    )
                except Exception as e_tsl:
                    logger.error(
                        f"{Back.YELLOW}WARNING: FAILED to place Trailing SL order: {e_tsl}{Style.RESET_ALL}"
                    )  # Warning as TSL might be secondary
                    logger.debug(traceback.format_exc())
                    # tsl_placed_successfully remains False

        # --- Final Check and Emergency Close if Protections Failed ---
        if not sl_placed_successfully:  # Fixed SL is considered essential
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: ESSENTIAL Fixed Stop-Loss placement FAILED for new {side.upper()} position on {symbol}. Attempting EMERGENCY CLOSE of position.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL: SL FAILED for {side.upper()} {filled_qty_val} @ {avg_fill_px}. ATTEMPTING EMERGENCY CLOSE."
            )

            # _active_trade_details is already set. close_position will use it and then clear it.
            emergency_close_reason = "EMERGENCY CLOSE - FIXED SL FAILED"
            close_order_details = close_position(
                exchange,
                symbol,
                _active_trade_details.copy(),
                reason=emergency_close_reason,
            )  # Pass copy

            if close_order_details and close_order_details.get("status") == "closed":
                logger.warning(
                    f"{Fore.YELLOW}Emergency close successful for {symbol} due to SL failure.{Style.RESET_ALL}"
                )
            else:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}EMERGENCY CLOSE FAILED or status uncertain for {symbol}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL: EMERGENCY CLOSE FAILED for {symbol}. MANUAL CHECK!"
                )
            return None  # Indicate overall entry operation failed due to protection failure

        # All good, entry and protections (at least fixed SL) are in place
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] ENTERED {side.upper()} {_format_for_log(filled_qty_val, 8)} @ {_format_for_log(avg_fill_px, 4)}. SL:~{actual_sl_px_str}, TSL:{tsl_value_for_api if CONFIG.trailing_stop_percentage > 0 else 'N/A'}%@~{tsl_activation_price_str if CONFIG.trailing_stop_percentage > 0 else 'N/A'}. EntryID:...{format_order_id(order_id)}"
        )
        return filled_entry_order

    except Exception as e_main:
        logger.error(
            f"{Back.RED}{Fore.WHITE}Place Order Ritual FAILED with unhandled exception: {type(e_main).__name__} - {e_main}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] ORDER FAIL ({side.upper()}): {type(e_main).__name__}"
        )
        # If an error occurred after _active_trade_details was set but before protections,
        # it's possible a position is open without SL. This is a risk.
        # However, the emergency close logic above should handle SL placement failures specifically.
        # This catch-all is for other unexpected issues.
        # Consider if _active_trade_details needs clearing here if an order might be partially open.
        # For now, assume if it gets here, the state is uncertain, and _active_trade_details might be stale.
        # A robust solution would re-check position status.
    return None


def cancel_open_orders(
    exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup"
) -> int:
    logger.info(
        f"{Fore.CYAN}Order Cleanup: Cancelling ALL open orders for {symbol} (Reason: {reason})...{Style.RESET_ALL}"
    )
    cancelled_count, failed_count = 0, 0

    # Bybit V5: 'category' needed for fetch_open_orders
    v5_category = "linear" if exchange.options.get("defaultType") == "linear" else {}
    params = {"category": v5_category}

    try:
        # Fetch all open orders for the symbol
        open_orders = safe_api_call(exchange.fetch_open_orders, symbol, params=params)

        if not open_orders:
            logger.info(
                f"{Fore.CYAN}Order Cleanup: No open orders found for {symbol} (Category: {v5_category}).{Style.RESET_ALL}"
            )
            return 0

        logger.warning(
            f"{Fore.YELLOW}Order Cleanup: Found {len(open_orders)} open orders for {symbol}. Attempting to cancel...{Style.RESET_ALL}"
        )
        for order in open_orders:
            order_id = order.get("id")
            order_type = order.get("type", "N/A")
            order_side = order.get("side", "N/A")
            order_price = order.get("price", "N/A")

            if order_id:
                try:
                    logger.info(
                        f"Cancelling order ...{format_order_id(order_id)} ({order_type} {order_side} @ {order_price})"
                    )
                    # cancel_order also needs category for Bybit V5
                    safe_api_call(
                        exchange.cancel_order, order_id, symbol, params=params
                    )
                    cancelled_count += 1
                    logger.debug(
                        f"Order ...{format_order_id(order_id)} cancellation request sent."
                    )
                except ccxt.OrderNotFound:
                    logger.info(
                        f"{Fore.GREEN}Order ...{format_order_id(order_id)} already gone (not found). Considered handled.{Style.RESET_ALL}"
                    )
                    cancelled_count += 1  # Effectively cancelled or closed
                except Exception as e_cancel:
                    logger.error(
                        f"{Fore.RED}Order Cleanup: FAILED to cancel order ...{format_order_id(order_id)}: {e_cancel}{Style.RESET_ALL}"
                    )
                    failed_count += 1
            else:
                logger.error(
                    f"{Fore.RED}Order Cleanup: Found an open order without an ID. Order data: {order}{Style.RESET_ALL}"
                )
                failed_count += 1

        if failed_count > 0:
            send_sms_alert(
                f"[{symbol.split('/')[0]}/{CONFIG.strategy_name.value}] WARNING: Failed to cancel {failed_count} orders during cleanup ({reason}). Manual check recommended."
            )

    except Exception as e_fetch:
        logger.error(
            f"{Fore.RED}Order Cleanup: Error fetching/processing open orders for {symbol}: {e_fetch}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        # Return -1 or raise to indicate a problem with the cleanup process itself?
        # For now, just log and report counts based on what was processed.

    log_msg_color = (
        Fore.GREEN
        if failed_count == 0 and cancelled_count > 0
        else (Fore.YELLOW if failed_count > 0 else Fore.CYAN)
    )
    logger.info(
        f"{log_msg_color}Order Cleanup for {symbol} (Reason: {reason}): Cancelled/Handled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}"
    )
    return cancelled_count


# --- Strategy Signal Generation Wrapper ---
def generate_strategy_signals(
    df: pd.DataFrame, strategy_instance: TradingStrategy
) -> dict[str, Any]:
    if strategy_instance:
        return strategy_instance.generate_signals(df)
    # This case should ideally not be reached if CONFIG.strategy_instance is always set
    logger.error(
        "Unknown or uninitialized strategy instance provided for signal generation."
    )
    # Fallback to a default "no signal" structure
    return TradingStrategy(CONFIG)._get_default_signals()  # type: ignore


# --- All Indicator Calculations ---
def calculate_all_indicators(
    df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculates all indicators required by any strategy and returns the DataFrame and volume/ATR analysis."""
    # Always calculate all indicators that *might* be used by *any* strategy.
    # This simplifies the logic, though slightly less efficient if only a subset is needed.
    # Strategies will pick the columns they need.

    df = calculate_supertrend(
        df, config.st_atr_length, config.st_multiplier
    )  # Primary ST
    df = calculate_supertrend(
        df,
        config.confirm_st_atr_length,
        config.confirm_st_multiplier,
        prefix="confirm_",
    )  # Confirmation ST
    df = calculate_stochrsi_momentum(
        df,
        config.stochrsi_rsi_length,
        config.stochrsi_stoch_length,
        config.stochrsi_k_period,
        config.stochrsi_d_period,
        config.momentum_length,
    )
    df = calculate_ehlers_fisher(
        df, config.ehlers_fisher_length, config.ehlers_fisher_signal_length
    )
    df = calculate_ehlers_ma(
        df,
        config.ehlers_fast_period,
        config.ehlers_slow_period,
        config.ehlers_ssf_poles,
    )

    # Volume and ATR analysis (ATR for SL, Volume for filter)
    vol_atr_data = analyze_volume_atr(
        df, config.atr_calculation_period, config.volume_ma_period
    )

    return df, vol_atr_data


# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    cycle_time_str = (
        df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")
        if not df.empty and df.index[-1]
        else "N/A"
    )
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Start ({CONFIG.strategy_name.value}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}"
    )

    # Ensure DataFrame has enough rows for indicator calculations.
    # This check is somewhat redundant if get_market_data provides enough, but good as a safeguard.
    # The actual min rows needed depends on the longest lookback of all indicators.
    # A general high number like 50-100 is a safe bet before detailed calculations.
    required_rows_for_any_indicator = 100  # A generous baseline
    if df is None or len(df) < required_rows_for_any_indicator:
        logger.warning(
            f"{Fore.YELLOW}Trade Logic: Insufficient data for {symbol} ({len(df) if df is not None else 0} rows, need ~{required_rows_for_any_indicator}). Skipping logic cycle.{Style.RESET_ALL}"
        )
        return

    try:
        # Calculate all indicators and get ATR/Volume data
        # Pass a copy of df to avoid modifying the one potentially cached in get_market_data
        df_with_indicators, vol_atr_data = calculate_all_indicators(df.copy(), CONFIG)

        current_atr = vol_atr_data.get("atr")  # This is Decimal or pd.NA
        last_candle_data = (
            df_with_indicators.iloc[-1]
            if not df_with_indicators.empty
            else pd.Series(dtype="object")
        )
        current_price = safe_decimal_conversion(
            last_candle_data.get("close"), pd.NA
        )  # Decimal or pd.NA

        if pd.isna(current_price) or current_price <= 0:
            logger.warning(
                f"{Fore.YELLOW}Trade Logic: Invalid or missing last close price for {symbol}. Last candle: {last_candle_data.to_dict()}{Style.RESET_ALL}"
            )
            return

        # Condition for being able to place an order (mainly for ATR-based SL)
        can_place_order_with_atr_sl = (
            not pd.isna(current_atr) and current_atr is not None and current_atr > 0
        )

        # Get current position status
        position = get_current_position(exchange, symbol)
        pos_side, pos_qty, pos_entry = (
            position["side"],
            position["qty"],
            position["entry_price"],
        )

        # Fetch order book if configured, or if flat and might enter (for price estimation)
        ob_data = None
        if CONFIG.fetch_order_book_per_cycle or (
            pos_side == CONFIG.pos_none and can_place_order_with_atr_sl
        ):
            ob_data = analyze_order_book(
                exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
            )

        # --- Log Current State Snapshot ---
        logger.info(
            f"{Fore.MAGENTA}--- Indicator & Market Snapshot for {symbol} ---{Style.RESET_ALL}"
        )
        logger.info(
            f"  Market: Close={_format_for_log(current_price, 4)}, ATR({CONFIG.atr_calculation_period})={_format_for_log(current_atr, 5)}"
        )

        volume_ratio_val = vol_atr_data.get("volume_ratio")
        is_vol_spike = False
        if not pd.isna(volume_ratio_val) and volume_ratio_val is not None:
            is_vol_spike = volume_ratio_val > CONFIG.volume_spike_threshold
        logger.info(
            f"  Volume: Ratio={_format_for_log(volume_ratio_val, 2)}, SpikeThreshold={CONFIG.volume_spike_threshold}, IsSpike={is_vol_spike}"
        )

        if ob_data:
            logger.info(
                f"  OrderBook: Ratio(B/A)={_format_for_log(ob_data.get('bid_ask_ratio'), 3)}, Spread={_format_for_log(ob_data.get('spread'), 4)}"
            )

        # Log strategy-specific values from the last candle
        strategy_logger = (
            CONFIG.strategy_instance.logger
        )  # Use the strategy's own logger instance
        strategy_logger.info(f"  Strategy Values ({CONFIG.strategy_name.value}):")
        for col_name in CONFIG.strategy_instance.required_columns:
            if col_name in last_candle_data.index:
                is_trend_col = (
                    "trend" in col_name.lower()
                )  # Simple check for ST trend columns
                strategy_logger.info(
                    f"    {col_name}: {_format_for_log(last_candle_data[col_name], is_bool_trend=is_trend_col)}"
                )
            else:
                strategy_logger.info(
                    f"    {col_name}: N/A (Not found in DataFrame's last row)"
                )

        pos_color = (
            Fore.GREEN
            if pos_side == CONFIG.pos_long
            else (Fore.RED if pos_side == CONFIG.pos_short else Fore.BLUE)
        )
        logger.info(
            f"  Position: Side={pos_color}{pos_side}{Style.RESET_ALL}, Qty={_format_for_log(pos_qty, 8)}, EntryPx={_format_for_log(pos_entry, 4)}"
        )
        logger.info(
            f"{Fore.MAGENTA}{'-' * (len(f'--- Indicator & Market Snapshot for {symbol} ---') - 2)}{Style.RESET_ALL}"
        )
        # --- End Snapshot ---

        # Generate signals from the chosen strategy
        strategy_signals = generate_strategy_signals(
            df_with_indicators, CONFIG.strategy_instance
        )

        # --- Exit Logic ---
        should_exit_long = pos_side == CONFIG.pos_long and strategy_signals.get(
            "exit_long", False
        )
        should_exit_short = pos_side == CONFIG.pos_short and strategy_signals.get(
            "exit_short", False
        )

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals.get("exit_reason", "Strategy Exit Signal")
            logger.warning(
                f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}*** EXIT SIGNAL TRIGGERED: Attempting to close {pos_side} position for {symbol} (Reason: {exit_reason}) ***{Style.RESET_ALL}"
            )
            cancel_open_orders(exchange, symbol, f"Pre-Exit ({exit_reason})")
            time.sleep(0.5)  # Cancel stops before market close
            close_result = close_position(
                exchange, symbol, position, reason=exit_reason
            )  # position dict from get_current_position
            if close_result:  # If close order was successfully processed (not necessarily instant fill confirmation for all exchanges)
                logger.info(
                    f"Position close order for {symbol} processed. Waiting for post-close delay."
                )
                time.sleep(CONFIG.post_close_delay_seconds)
            return  # End cycle after attempting close

        # --- Hold Logic (If already in position and no exit signal) ---
        if pos_side != CONFIG.pos_none:
            logger.info(
                f"Holding {pos_color}{pos_side}{Style.RESET_ALL} position for {symbol}. Awaiting SL/TSL trigger or explicit exit signal."
            )
            return  # End cycle, continue holding

        # --- Entry Logic (If flat) ---
        if not can_place_order_with_atr_sl:  # Check if ATR is valid for SL placement
            logger.warning(
                f"{Fore.YELLOW}Holding Cash for {symbol}. Cannot consider entry: Invalid ATR ({current_atr}) for SL calculation.{Style.RESET_ALL}"
            )
            return

        # Check for entry signals from strategy
        potential_enter_long = strategy_signals.get("enter_long", False)
        potential_enter_short = strategy_signals.get("enter_short", False)

        if not (potential_enter_long or potential_enter_short):
            logger.info(
                f"Holding Cash for {symbol}. No entry signal from strategy {CONFIG.strategy_name.value}."
            )
            return

        # Apply Confirmation Filters (Order Book, Volume Spike)
        # Order Book Filter
        ob_confirm_long, ob_confirm_short = (
            True,
            True,
        )  # Default to true if filter not used or data unavailable
        if (
            ob_data
            and not pd.isna(ob_data.get("bid_ask_ratio"))
            and ob_data.get("bid_ask_ratio") is not None
        ):
            ratio = ob_data["bid_ask_ratio"]  # This is a Decimal
            # Check if thresholds are finite (i.e., filter is active)
            if CONFIG.order_book_ratio_threshold_long < Decimal("Infinity"):
                ob_confirm_long = ratio >= CONFIG.order_book_ratio_threshold_long
            if CONFIG.order_book_ratio_threshold_short > Decimal(
                0
            ):  # Assuming short ratio is < 1
                ob_confirm_short = ratio <= CONFIG.order_book_ratio_threshold_short
        elif CONFIG.order_book_ratio_threshold_long < Decimal(
            "Infinity"
        ) or CONFIG.order_book_ratio_threshold_short > Decimal(0):
            # If filter is active but OB data was not available/valid
            logger.warning(
                f"{Fore.YELLOW}Order book filter active, but no valid OB data. Entry blocked by OB filter.{Style.RESET_ALL}"
            )
            ob_confirm_long, ob_confirm_short = False, False

        # Volume Spike Filter
        vol_confirm = not CONFIG.require_volume_spike_for_entry or is_vol_spike

        # Final decision to enter
        final_enter_long = potential_enter_long and ob_confirm_long and vol_confirm
        final_enter_short = potential_enter_short and ob_confirm_short and vol_confirm

        entry_params_dict = {
            "exchange": exchange,
            "symbol": symbol,
            "risk_percentage": CONFIG.risk_per_trade_percentage,
            "current_atr": current_atr,
            "sl_atr_multiplier": CONFIG.atr_stop_loss_multiplier,
            "leverage": CONFIG.leverage,
            "max_order_cap_usdt": CONFIG.max_order_usdt_amount,
            "margin_check_buffer": CONFIG.required_margin_buffer,
            "tsl_percent": CONFIG.trailing_stop_percentage,
            "tsl_activation_offset_percent": CONFIG.trailing_stop_activation_offset_percent,
        }

        if final_enter_long:
            logger.success(
                f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** CONFIRMED LONG ENTRY SIGNAL for {symbol} (Strategy: {CONFIG.strategy_name.value}) ***{Style.RESET_ALL}"
            )
            logger.info(
                f"Filter status: OB Long Confirm: {ob_confirm_long}, Volume Confirm: {vol_confirm}"
            )
            cancel_open_orders(exchange, symbol, "Pre-Long Entry")
            time.sleep(0.5)  # Cancel any existing stops/limits
            place_risked_market_order(side=CONFIG.side_buy, **entry_params_dict)  # type: ignore
        elif final_enter_short:
            logger.success(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** CONFIRMED SHORT ENTRY SIGNAL for {symbol} (Strategy: {CONFIG.strategy_name.value}) ***{Style.RESET_ALL}"
            )
            logger.info(
                f"Filter status: OB Short Confirm: {ob_confirm_short}, Volume Confirm: {vol_confirm}"
            )
            cancel_open_orders(exchange, symbol, "Pre-Short Entry")
            time.sleep(0.5)
            place_risked_market_order(side=CONFIG.side_sell, **entry_params_dict)  # type: ignore
        elif (
            potential_enter_long or potential_enter_short
        ):  # Signal was there, but filters blocked
            logger.info(
                f"Holding Cash for {symbol}. Entry signal present but confirmation filters not met. Long Filter: OB({ob_confirm_long}) Vol({vol_confirm}). Short Filter: OB({ob_confirm_short}) Vol({vol_confirm})."
            )
        # else: No signal, already handled. Implicitly Holding Cash.

    except (
        Exception
    ) as e:  # Catch-all for unexpected errors within a single trade logic cycle
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL ERROR in trade_logic cycle for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Full traceback for debugging
        market_base = symbol.split("/")[0].split(":")[0]
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL trade_logic ERROR: {type(e).__name__}. Cycle skipped."
        )
    finally:
        logger.info(
            f"{Fore.BLUE}{Style.BRIGHT}========== Cycle End: {symbol} =========={Style.RESET_ALL}\n"
        )


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    logger.warning(
        f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing Pyrmethus's arcane energies...{Style.RESET_ALL}"
    )

    # Determine market base for SMS, handle None symbol
    market_base = "Bot"  # Default if symbol is None
    if symbol:
        parts = symbol.split("/")
        if parts:
            market_base = parts[0].split(":")[0]

    # Determine strategy name for SMS, handle if CONFIG is not fully available
    strat_name_val = "N/A"
    if (
        "CONFIG" in globals()
        and hasattr(CONFIG, "strategy_name")
        and CONFIG.strategy_name
    ):
        strat_name_val = CONFIG.strategy_name.value

    send_sms_alert(
        f"[{market_base}/{strat_name_val}] Shutdown initiated. Attempting cleanup..."
    )

    # Log final trade metrics summary if available
    if "trade_metrics" in globals() and hasattr(trade_metrics, "summary"):
        logger.info("Attempting to log final trade metrics summary...")
        trade_metrics.summary()

    if not exchange or not symbol:
        logger.warning(
            f"{Fore.YELLOW}Shutdown: Exchange instance or trading symbol not defined. Automated cleanup might be limited.{Style.RESET_ALL}"
        )
        # Even if exchange/symbol is None, we still want to log the shutdown message at the end.
    else:
        try:
            logger.warning(f"Shutdown: Cancelling any open orders for {symbol}...")
            # Pass a specific reason for cancellation
            cancel_open_orders(exchange, symbol, "Bot Shutdown")
            time.sleep(1.5)  # Allow time for cancellations to process

            logger.warning(f"Shutdown: Checking for active position on {symbol}...")
            position = get_current_position(exchange, symbol)  # Re-check position

            if (
                position["side"] != CONFIG.pos_none
                and position["qty"] > CONFIG.position_qty_epsilon
            ):
                logger.warning(
                    f"{Fore.YELLOW}Shutdown: Active {position['side']} position found for {symbol}. Attempting to close...{Style.RESET_ALL}"
                )
                # Pass a specific reason for closing
                close_result = close_position(
                    exchange, symbol, position, "Bot Shutdown Emergency Close"
                )

                if close_result:  # If close order was placed
                    logger.info(
                        f"{Fore.CYAN}Shutdown: Close order for {symbol} placed. Performing final position check after delay...{Style.RESET_ALL}"
                    )
                    time.sleep(
                        CONFIG.post_close_delay_seconds * 2
                    )  # Longer delay to allow fill
                    final_pos_check = get_current_position(exchange, symbol)
                    if final_pos_check["side"] == CONFIG.pos_none:
                        logger.success(
                            f"{Fore.GREEN}Shutdown: Position for {symbol} confirmed CLOSED successfully.{Style.RESET_ALL}"
                        )
                        send_sms_alert(
                            f"[{market_base}/{strat_name_val}] Position on {symbol} successfully closed during shutdown."
                        )
                    else:
                        logger.critical(
                            f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN ERROR: FAILED TO CONFIRM position closure for {symbol}! Final State: {final_pos_check['side']} Qty={final_pos_check['qty']}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                        )
                        send_sms_alert(
                            f"[{market_base}/{strat_name_val}] CRITICAL: FAILED to confirm {symbol} position closure on shutdown. MANUAL CHECK!"
                        )
                else:  # If close_position call itself failed (e.g., API error placing close order)
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN ERROR: FAILED TO PLACE close order for active position on {symbol}! MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}/{strat_name_val}] CRITICAL: FAILED to place close order for {symbol} on shutdown. MANUAL CHECK!"
                    )
            else:
                logger.info(
                    f"{Fore.GREEN}Shutdown: No active position found for {symbol}. Clean exit regarding positions.{Style.RESET_ALL}"
                )
        except Exception as e_cleanup:
            logger.error(
                f"{Fore.RED}Shutdown Error: Exception during cleanup for {symbol}: {e_cleanup}{Style.RESET_ALL}"
            )
            logger.debug(traceback.format_exc())
            send_sms_alert(
                f"[{market_base}/{strat_name_val}] ERROR during shutdown cleanup for {symbol}: {type(e_cleanup).__name__}. MANUAL CHECK MAY BE NEEDED."
            )

    logger.info(
        f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Spell Shutdown Sequence Complete ---{Style.RESET_ALL}"
    )


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.3.0 Initializing ({start_time_str}) ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}--- Active Strategy Path: {CONFIG.strategy_name.value} ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.GREEN}--- Primary Protections: ATR-based Fixed Stop Loss + Exchange-Native Trailing Stop Loss (Bybit V5) ---{Style.RESET_ALL}"
    )
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING - EXTREME RISK - EDUCATIONAL USE ONLY !!! ---{Style.RESET_ALL}"
    )

    exchange: ccxt.Exchange | None = (
        None  # Ensure it's defined in the broader scope for finally block
    )
    symbol_unified: str | None = None  # Same for symbol
    run_bot: bool = True
    cycle_count: int = 0

    try:
        exchange = initialize_exchange()
        if not exchange:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Exchange initialization failed. Pyrmethus cannot proceed. Exiting.{Style.RESET_ALL}"
            )
            return  # Exit main if exchange setup fails

        # Validate and unify symbol format using exchange.market()
        market = exchange.market(CONFIG.symbol)
        if not market:
            raise ValueError(
                f"Market for symbol '{CONFIG.symbol}' not found. Ensure it's a valid symbol for {exchange.id}."
            )
        symbol_unified = market["symbol"]  # Use the CCXT unified symbol string

        if not market.get("contract", False):  # Check if it's a futures/contract market
            raise ValueError(
                f"Market '{symbol_unified}' is not a contract market. This bot is for futures trading."
            )

        logger.info(
            f"{Fore.GREEN}Spell focused on symbol: {symbol_unified} (Type: {market.get('type', 'N/A')}, ID: {market.get('id', 'N/A')}){Style.RESET_ALL}"
        )

        if not set_leverage(exchange, symbol_unified, CONFIG.leverage):
            # set_leverage logs its own errors. If it returns False, it's a critical setup failure.
            raise RuntimeError(
                f"Failed to set leverage to {CONFIG.leverage}x for {symbol_unified}. Halting."
            )

        # Log key configuration details for this run
        logger.info(
            f"{Fore.MAGENTA}--- Key Spell Configuration Summary ---{Style.RESET_ALL}"
        )
        logger.info(
            f"Trading Symbol: {symbol_unified}, Candle Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x"
        )
        logger.info(f"Strategy: {CONFIG.strategy_name.value}")
        logger.info(
            f"Risk Per Trade: {CONFIG.risk_per_trade_percentage:.2%}, Max Position Value (USDT): {CONFIG.max_order_usdt_amount}"
        )
        logger.info(
            f"ATR SL Multiplier: {CONFIG.atr_stop_loss_multiplier}, TSL Percent: {CONFIG.trailing_stop_percentage:.3%} (Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.3%})"
        )
        logger.info(
            f"Account Max Margin Ratio Threshold: {CONFIG.max_account_margin_ratio:.0%}"
        )
        logger.info(
            f"Volume Spike Filter: {'Enabled' if CONFIG.require_volume_spike_for_entry else 'Disabled'}, OB Filter Long Thresh: {CONFIG.order_book_ratio_threshold_long}, OB Filter Short Thresh: {CONFIG.order_book_ratio_threshold_short}"
        )
        logger.info(f"{Fore.MAGENTA}{'-' * 35}{Style.RESET_ALL}")

        market_base_name = symbol_unified.split("/")[0].split(":")[0]
        send_sms_alert(
            f"[{market_base_name}/{CONFIG.strategy_name.value}] Pyrmethus v2.3.0 Initialized. Symbol: {symbol_unified}. Starting trading loop."
        )

        # --- Main Trading Loop ---
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(
                f"{Fore.CYAN}--- Cycle {cycle_count} Start ({time.strftime('%H:%M:%S %Z')}) ---{Style.RESET_ALL}"
            )

            # Account Health Check at the start of each cycle
            if not check_account_health(exchange, CONFIG):
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}ACCOUNT HEALTH CHECK FAILED! Bot operations paused for safety. Will re-check next cycle.{Style.RESET_ALL}"
                )
                # SMS for health failure is sent from check_account_health
                time.sleep(
                    CONFIG.sleep_seconds * 10
                )  # Extended pause on health failure
                continue  # Skip to next cycle for re-check

            try:
                # Determine data limit: max lookback of all indicators + buffer
                # This should be generous enough for all indicators to initialize properly.
                indicator_lookbacks = [
                    CONFIG.st_atr_length,
                    CONFIG.confirm_st_atr_length,
                    CONFIG.stochrsi_rsi_length
                    + CONFIG.stochrsi_stoch_length
                    + CONFIG.stochrsi_d_period,  # StochRSI needs sum of lengths
                    CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length,
                    max(CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
                    + CONFIG.ehlers_ssf_poles,
                    CONFIG.atr_calculation_period,
                    CONFIG.volume_ma_period,
                ]
                # A base of 100, plus max lookback, plus buffer for API and calculations.
                data_limit = (
                    max(100, max(indicator_lookbacks) if indicator_lookbacks else 100)
                    + CONFIG.api_fetch_limit_buffer
                    + 20
                )

                df_market_data = get_market_data(
                    exchange, symbol_unified, CONFIG.interval, limit=data_limit
                )

                if df_market_data is not None and not df_market_data.empty:
                    trade_logic(exchange, symbol_unified, df_market_data)
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Skipping trade logic for cycle {cycle_count}: Invalid or missing market data for {symbol_unified}.{Style.RESET_ALL}"
                    )

            # --- Main Loop Exception Handling ---
            except ccxt.RateLimitExceeded as e_ratelimit:
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e_ratelimit}. Sleeping for an extended duration...{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds * 6)  # Longer sleep for rate limits
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
            ) as e_network:
                logger.warning(
                    f"{Fore.YELLOW}Network/Exchange Connectivity Issue: {e_network}. Retrying after a pause.{Style.RESET_ALL}"
                )
                sleep_multiplier = (
                    6 if isinstance(e_network, ccxt.ExchangeNotAvailable) else 3
                )
                time.sleep(CONFIG.sleep_seconds * sleep_multiplier)
            except ccxt.AuthenticationError as e_auth:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}FATAL: Authentication Error: {e_auth}. API keys may be invalid or revoked. Stopping bot.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base_name}/{CONFIG.strategy_name.value}] CRITICAL: Auth Error! Bot stopping. Check API keys."
                )
                run_bot = False  # Stop the bot on auth errors
            except (
                Exception
            ) as e_mainloop:  # Catch any other unexpected errors in the loop
                logger.exception(
                    f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL ERROR in Main Loop (Cycle {cycle_count}): {e_mainloop} !!! Attempting to stop gracefully.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base_name}/{CONFIG.strategy_name.value}] CRITICAL UNEXPECTED ERROR: {type(e_mainloop).__name__}! Bot stopping."
                )
                run_bot = False  # Stop the bot on critical unexpected errors

            # --- Cycle Sleep ---
            if run_bot:  # Only sleep if bot is still supposed to run
                elapsed_cycle_time = time.monotonic() - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed_cycle_time)
                logger.debug(
                    f"Cycle {cycle_count} processed in {elapsed_cycle_time:.2f}s. Sleeping for {sleep_duration:.2f}s until next cycle."
                )
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.warning(
            f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt detected. Initiating graceful shutdown...{Style.RESET_ALL}"
        )
        run_bot = False  # Signal loop to stop
    except (
        Exception
    ) as startup_error:  # Catch errors during initial setup (before main loop)
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL STARTUP ERROR: {startup_error}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Full traceback for startup errors
        # Try to send SMS if config might be partially loaded
        sms_alert_msg = f"[Pyrmethus] CRITICAL STARTUP ERROR: {type(startup_error).__name__}. Bot failed to start."
        if (
            "CONFIG" in globals()
            and hasattr(CONFIG, "enable_sms_alerts")
            and CONFIG.enable_sms_alerts
            and hasattr(CONFIG, "sms_recipient_number")
            and CONFIG.sms_recipient_number
        ):
            send_sms_alert(sms_alert_msg)
        else:  # Log to console if SMS can't be sent
            print(f"CRITICAL STARTUP SMS (Simulated): {sms_alert_msg}")
        run_bot = False  # Ensure bot doesn't attempt to run
    finally:
        # Graceful shutdown will be called here, regardless of how the try block was exited.
        # It handles cancelling orders and closing positions if exchange and symbol are available.
        graceful_shutdown(exchange, symbol_unified)
        logger.info(
            f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    main()
