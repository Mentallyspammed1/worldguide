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
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT",
            "0.001",
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

        if value_to_cast is None:
            if required:
                _pre_logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required config '{key}' has no value (env/default).{Style.RESET_ALL}"
                )
                raise ValueError(
                    f"Required environment variable '{key}' resolved to None."
                )
            else:
                _pre_logger.debug(
                    f"{color}Final value for {key}: None (Type: NoneType) (Source: {source}){Style.RESET_ALL}"
                )
                return None

        final_value: Any = None
        try:
            raw_value_str = str(value_to_cast)
            if cast_type == bool:
                final_value = raw_value_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal:
                final_value = Decimal(raw_value_str)
            elif cast_type == int:
                final_value = int(Decimal(raw_value_str))
            elif cast_type == float:
                final_value = float(raw_value_str)
            elif cast_type == str:
                final_value = raw_value_str
            else:
                _pre_logger.warning(
                    f"Unsupported cast_type '{cast_type.__name__}' for key '{key}'. Returning raw."
                )
                final_value = value_to_cast

        except (ValueError, TypeError, InvalidOperation) as e:
            _pre_logger.error(
                f"{Fore.RED}Invalid type/value for {key}: '{value_to_cast}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Trying default '{default}'.{Style.RESET_ALL}"
            )
            if default is None:
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
                    return None
            else:
                source = "Default (Fallback)"
                _pre_logger.debug(
                    f"Casting fallback default '{default}' for '{key}' to {cast_type.__name__}"
                )
                try:
                    default_str = str(default)
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
                        final_value = default
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
    format="%(asctime)s [%(levelname)-8s] %(name)-15s %(message)s",  # Added logger name
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

if sys.stdout.isatty():
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
    logger.critical(
        f"{Back.RED}{Fore.WHITE}Configuration loading failed. Error: {config_error}{Style.RESET_ALL}"
    )
    sys.exit(1)
except Exception as general_config_error:
    logger.critical(
        f"{Back.RED}{Fore.WHITE}Unexpected critical error during configuration: {general_config_error}{Style.RESET_ALL}"
    )
    logger.debug(traceback.format_exc())
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
        confirm_is_up = last.get("confirm_trend", pd.NA)

        if pd.isna(confirm_is_up):
            self.logger.debug("Confirmation trend is NA. No signal.")
            return signals
        if primary_long_flip and confirm_is_up is True:
            signals["enter_long"] = True
        if primary_short_flip and confirm_is_up is False:
            signals["enter_short"] = True
        if primary_short_flip:
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary ST Flipped Short"
        if primary_long_flip:
            signals["exit_short"] = True
            signals["exit_reason"] = "Primary ST Flipped Long"
        return signals


class StochRsiMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["stochrsi_k", "stochrsi_d", "momentum"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2):
            return signals
        last, prev = df.iloc[-1], df.iloc[-2]
        k_now, d_now, mom_now = (
            last.get("stochrsi_k"),
            last.get("stochrsi_d"),
            last.get("momentum"),
        )
        k_prev, d_prev = prev.get("stochrsi_k"), prev.get("stochrsi_d")

        if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]):
            self.logger.debug("Skipping due to NA StochRSI/Mom values.")
            return signals

        k_now_dec = safe_decimal_conversion(k_now, Decimal("NaN"))
        mom_now_dec = safe_decimal_conversion(mom_now, Decimal("NaN"))

        if k_now_dec.is_nan() or mom_now_dec.is_nan():
            self.logger.debug(
                "Skipping due to NA StochRSI/Mom Decimal values after conversion."
            )
            return signals

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
        if k_prev >= d_prev and k_now < d_now:
            signals["exit_long"] = True
            signals["exit_reason"] = "StochRSI K crossed below D"
        if k_prev <= d_prev and k_now > d_now:
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
        fish_now, sig_now = last.get("ehlers_fisher"), last.get("ehlers_signal")
        fish_prev, sig_prev = prev.get("ehlers_fisher"), prev.get("ehlers_signal")

        if any(pd.isna(v) for v in [fish_now, sig_now, fish_prev, sig_prev]):
            self.logger.debug("Skipping due to NA Ehlers Fisher values.")
            return signals
        if fish_prev <= sig_prev and fish_now > sig_now:
            signals["enter_long"] = True
        if fish_prev >= sig_prev and fish_now < sig_now:
            signals["enter_short"] = True
        if fish_prev >= sig_prev and fish_now < sig_now:
            signals["exit_long"] = True
            signals["exit_reason"] = "Ehlers Fisher crossed Short"
        if fish_prev <= sig_prev and fish_now > sig_now:
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
            last.get("ehlers_ssf_fast"),
            last.get("ehlers_ssf_slow"),
        )
        fast_ma_prev, slow_ma_prev = (
            prev.get("ehlers_ssf_fast"),
            prev.get("ehlers_ssf_slow"),
        )

        if any(
            pd.isna(v) for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]
        ):
            self.logger.debug("Skipping due to NA Ehlers SSF MA values.")
            return signals
        if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
            signals["enter_long"] = True
        if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
            signals["enter_short"] = True
        if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
            signals["exit_long"] = True
            signals["exit_reason"] = "Fast Ehlers SSF MA crossed below Slow"
        if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
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
    logger.critical(f"Failed to find strategy class for {CONFIG.strategy_name.value}")
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
        if side.lower() == CONFIG.side_sell or side.lower() == CONFIG.pos_short.lower():
            profit_per_unit = entry_price - exit_price

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
        self.logger.success(
            f"{Fore.MAGENTA}Trade Recorded: {side.upper()} {qty} {symbol.split('/')[0]} | Entry: {entry_price:.4f}, Exit: {exit_price:.4f} | P/L: {profit:.2f} {CONFIG.usdt_symbol} | Duration: {duration} | Reason: {reason}{Style.RESET_ALL}"
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
            f"\n--- Trade Metrics Summary ---\n"
            f"Total Trades: {total_trades} | Wins: {wins}, Losses: {losses}, Breakeven: {breakeven}\n"
            f"Win Rate: {win_rate:.2f}% | Total P/L: {total_profit:.2f} {CONFIG.usdt_symbol}\n"
            f"Avg P/L per Trade: {avg_profit:.2f} {CONFIG.usdt_symbol}\n"
            f"--- End Summary ---"
        )
        self.logger.info(summary_str)
        return summary_str


trade_metrics = TradeMetrics()
_active_trade_details: dict[str, Any] = {
    "entry_price": None,
    "entry_time_ms": None,
    "side": None,
    "qty": None,
}


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    if value is None or (
        isinstance(value, float) and pd.isna(value)
    ):  # Handle pd.NA passed as float NaN
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(
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
        return "N/A (Trend)"
    if isinstance(value, Decimal):
        return f"{value:.{precision}f}"
    if isinstance(value, (float, int)):
        return f"{float(value):.{precision}f}"
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
        return str(Decimal(str(price)).normalize())


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal) -> str:
    try:
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error shaping amount {amount} for {symbol}: {e}. Using raw Decimal.{Style.RESET_ALL}"
        )
        return str(Decimal(str(amount)).normalize())


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
    if _termux_sms_command_exists is None:
        _termux_sms_command_exists = shutil.which("termux-sms-send") is not None
        if not _termux_sms_command_exists:
            logger.warning(
                f"{Fore.YELLOW}SMS: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}"
            )
    if not _termux_sms_command_exists or not CONFIG.sms_recipient_number:
        return False
    try:
        command: list[str] = [
            "termux-sms-send",
            "-n",
            CONFIG.sms_recipient_number,
            message,
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
            logger.success(f"{Fore.MAGENTA}SMS dispatched.{Style.RESET_ALL}")
            return True
        else:
            logger.error(
                f"{Fore.RED}SMS failed. Code: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}"
            )
            return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: {e}{Style.RESET_ALL}")
        return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> ccxt.Exchange | None:
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret missing.{Style.RESET_ALL}"
        )
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing.")
        return None
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "linear", "adjustForTimeDifference": True},
                "recvWindow": CONFIG.default_recv_window,
            }
        )
        # exchange.set_sandbox_mode(True) # Testnet
        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True)
        logger.debug(f"Loaded {len(exchange.markets)} market structures.")
        logger.debug("Performing initial balance check...")
        exchange.fetch_balance(params={"category": "linear"})
        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (V5 API).{Style.RESET_ALL}"
        )
        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! LIVE SCALPING MODE ACTIVE - EXTREME CAUTION !!!{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus/{CONFIG.strategy_name.value}] Portal opened.")
        return exchange
    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Portal opening failed: {e}{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Init FAILED: {type(e).__name__}.")
        logger.debug(traceback.format_exc())
    return None


# --- Indicator Calculation Functions - Scrying the Market ---
def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = ""
) -> pd.DataFrame:
    col_prefix = f"{prefix}" if prefix else ""
    out_supertrend_val = f"{col_prefix}supertrend"
    out_trend_direction = f"{col_prefix}trend"
    out_long_flip = f"{col_prefix}st_long_flip"
    out_short_flip = f"{col_prefix}st_short_flip"
    target_cols = [
        out_supertrend_val,
        out_trend_direction,
        out_long_flip,
        out_short_flip,
    ]

    pta_st_val_col = f"SUPERT_{length}_{float(multiplier)}"
    pta_st_dir_col = f"SUPERTd_{length}_{float(multiplier)}"
    pta_st_long_level_col = (
        f"SUPERTl_{length}_{float(multiplier)}"  # Actual ST line when long
    )
    pta_st_short_level_col = (
        f"SUPERTs_{length}_{float(multiplier)}"  # Actual ST line when short
    )

    min_len = length + 2  # pandas_ta might need length + lookback for direction change

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low", "close"])
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        # Work on a copy to avoid SettingWithCopyWarning if df is a slice
        temp_df = df[["high", "low", "close"]].copy()
        temp_df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the columns
        pta_cols_to_check = [
            pta_st_val_col,
            pta_st_dir_col,
            pta_st_long_level_col,
            pta_st_short_level_col,
        ]
        if not all(c in temp_df.columns for c in pta_cols_to_check):
            missing_cols = [c for c in pta_cols_to_check if c not in temp_df.columns]
            logger.error(
                f"{Fore.RED}Scrying ({col_prefix}ST): pandas_ta failed to create columns: {missing_cols}.{Style.RESET_ALL}"
            )
            for col in target_cols:
                df[col] = pd.NA
            return df

        df[out_supertrend_val] = temp_df[pta_st_val_col].apply(safe_decimal_conversion)
        # Trend: True for up (1), False for down (-1), pd.NA for no trend
        df[out_trend_direction] = pd.NA  # Default to NA
        df.loc[temp_df[pta_st_dir_col] == 1, out_trend_direction] = True
        df.loc[temp_df[pta_st_dir_col] == -1, out_trend_direction] = False

        # Flips
        prev_dir_temp = temp_df[pta_st_dir_col].shift(1)
        df[out_long_flip] = (temp_df[pta_st_dir_col] == 1) & (prev_dir_temp == -1)
        df[out_short_flip] = (temp_df[pta_st_dir_col] == -1) & (prev_dir_temp == 1)

        if not df.empty:
            last_val = df[out_supertrend_val].iloc[-1]
            last_trend_bool = df[out_trend_direction].iloc[-1]
            last_l_flip = df[out_long_flip].iloc[-1]
            last_s_flip = df[out_short_flip].iloc[-1]
            trend_str = _format_for_log(last_trend_bool, is_bool_trend=True)
            flip_str = "L" if last_l_flip else ("S" if last_s_flip else "None")
            logger.debug(
                f"Scrying ({col_prefix}ST({length},{multiplier})): Val={_format_for_log(last_val)}, Trend={trend_str}, Flip={flip_str}"
            )
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> dict[str, Decimal | None]:
    results: dict[str, Decimal | None] = {
        "atr": None,
        "volume_ma": None,
        "last_volume": None,
        "volume_ratio": None,
    }
    min_len = max(atr_len, vol_ma_len) + 1
    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low", "close", "volume"])
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data.{Style.RESET_ALL}"
        )
        return results
    try:
        temp_df = df.copy()  # Work on a copy
        atr_col = f"ATRr_{atr_len}"
        temp_df.ta.atr(length=atr_len, append=True)
        if atr_col in temp_df.columns and not temp_df.empty:
            results["atr"] = safe_decimal_conversion(temp_df[atr_col].iloc[-1])

        volume_ma_col = f"volume_sma_{vol_ma_len}"
        temp_df["volume"] = pd.to_numeric(temp_df["volume"], errors="coerce")
        temp_df[volume_ma_col] = (
            temp_df["volume"]
            .rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2))
            .mean()
        )
        if not temp_df.empty:
            results["volume_ma"] = safe_decimal_conversion(
                temp_df[volume_ma_col].iloc[-1]
            )
            results["last_volume"] = safe_decimal_conversion(temp_df["volume"].iloc[-1])
            if (
                results["volume_ma"]
                and results["volume_ma"] > CONFIG.position_qty_epsilon
                and results["last_volume"]
            ):
                try:
                    results["volume_ratio"] = (
                        results["last_volume"] / results["volume_ma"]
                    )
                except DivisionByZero:
                    results["volume_ratio"] = None

        log_parts = [
            f"ATR({atr_len})={Fore.CYAN}{_format_for_log(results['atr'], 5)}{Style.RESET_ALL}"
        ]
        if results["last_volume"] is not None:
            log_parts.append(f"LastVol={_format_for_log(results['last_volume'], 2)}")
        if results["volume_ma"] is not None:
            log_parts.append(
                f"VolMA({vol_ma_len})={_format_for_log(results['volume_ma'], 2)}"
            )
        if results["volume_ratio"] is not None:
            log_parts.append(
                f"VolRatio={Fore.YELLOW}{_format_for_log(results['volume_ratio'], 2)}{Style.RESET_ALL}"
            )
        logger.debug(f"Scrying Results: {', '.join(log_parts)}")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results}
    return results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    min_len = max(rsi_len + stoch_len + d, mom_len) + 10
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data.{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)
    try:
        stochrsi_df = df.ta.stochrsi(
            length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False
        )
        k_col_ta, d_col_ta = (
            f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}",
            f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}",
        )
        if stochrsi_df is not None and not stochrsi_df.empty:
            df["stochrsi_k"] = (
                stochrsi_df[k_col_ta].apply(safe_decimal_conversion)
                if k_col_ta in stochrsi_df
                else pd.NA
            )
            df["stochrsi_d"] = (
                stochrsi_df[d_col_ta].apply(safe_decimal_conversion)
                if d_col_ta in stochrsi_df
                else pd.NA
            )
        else:
            df["stochrsi_k"], df["stochrsi_d"] = pd.NA, pd.NA

        temp_df_mom = df[["close"]].copy()  # Operate on copy for momentum
        mom_col_ta = f"MOM_{mom_len}"
        temp_df_mom.ta.mom(length=mom_len, append=True)
        df["momentum"] = (
            temp_df_mom[mom_col_ta].apply(safe_decimal_conversion)
            if mom_col_ta in temp_df_mom
            else pd.NA
        )

        if not df.empty:
            k_v, d_v, m_v = (
                df["stochrsi_k"].iloc[-1],
                df["stochrsi_d"].iloc[-1],
                df["momentum"].iloc[-1],
            )
            logger.debug(
                f"Scrying (StochRSI/Mom): K={_format_for_log(k_v, 2)}, D={_format_for_log(d_v, 2)}, Mom={_format_for_log(m_v, 4)}"
            )
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    min_len = length + signal + 5
    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low"])
        or len(df) < min_len
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data.{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)
    try:
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col_ta, signal_col_ta = (
            f"FISHERT_{length}_{signal}",
            f"FISHERTs_{length}_{signal}",
        )
        if fisher_df is not None and not fisher_df.empty:
            df["ehlers_fisher"] = (
                fisher_df[fish_col_ta].apply(safe_decimal_conversion)
                if fish_col_ta in fisher_df
                else pd.NA
            )
            df["ehlers_signal"] = (
                fisher_df[signal_col_ta].apply(safe_decimal_conversion)
                if signal_col_ta in fisher_df
                else pd.NA
            )
        else:
            df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA

        if not df.empty:
            f_v, s_v = df["ehlers_fisher"].iloc[-1], df["ehlers_signal"].iloc[-1]
            logger.debug(
                f"Scrying (EhlersFisher({length},{signal})): Fisher={_format_for_log(f_v)}, Signal={_format_for_log(s_v)}"
            )
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def calculate_ehlers_ma(
    df: pd.DataFrame, fast_len: int, slow_len: int, poles: int
) -> pd.DataFrame:
    target_cols = ["ehlers_ssf_fast", "ehlers_ssf_slow"]
    min_len = max(fast_len, slow_len) + poles + 5
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Ehlers SSF MA): Insufficient data.{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)
    try:
        ssf_fast_series = df.ta.ssf(length=fast_len, poles=poles, append=False)
        df["ehlers_ssf_fast"] = (
            ssf_fast_series.apply(safe_decimal_conversion)
            if ssf_fast_series is not None
            else pd.NA
        )
        ssf_slow_series = df.ta.ssf(length=slow_len, poles=poles, append=False)
        df["ehlers_ssf_slow"] = (
            ssf_slow_series.apply(safe_decimal_conversion)
            if ssf_slow_series is not None
            else pd.NA
        )

        if not df.empty:
            fast_v, slow_v = (
                df["ehlers_ssf_fast"].iloc[-1],
                df["ehlers_ssf_slow"].iloc[-1],
            )
            logger.debug(
                f"Scrying (Ehlers SSF MA({fast_len},{slow_len},p{poles})): Fast={_format_for_log(fast_v)}, Slow={_format_for_log(slow_v)}"
            )
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Ehlers SSF MA): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> dict[str, Decimal | None]:
    results: dict[str, Decimal | None] = {
        "bid_ask_ratio": None,
        "spread": None,
        "best_bid": None,
        "best_ask": None,
    }
    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(
            f"{Fore.YELLOW}OB Scrying: fetchL2OrderBook not supported.{Style.RESET_ALL}"
        )
        return results
    try:
        order_book = safe_api_call(
            exchange.fetch_l2_order_book, symbol, limit=fetch_limit
        )
        bids, asks = order_book.get("bids", []), order_book.get("asks", [])
        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}OB Scrying: Empty bids/asks for {symbol}.{Style.RESET_ALL}"
            )
            return results

        results["best_bid"] = (
            safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        )
        results["best_ask"] = (
            safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        )
        if (
            results["best_bid"]
            and results["best_ask"]
            and results["best_bid"] > 0
            and results["best_ask"] > 0
        ):
            results["spread"] = results["best_ask"] - results["best_bid"]

        bid_vol = sum(
            safe_decimal_conversion(b[1])
            for b in bids[: min(depth, len(bids))]
            if len(b) > 1
        )
        ask_vol = sum(
            safe_decimal_conversion(a[1])
            for a in asks[: min(depth, len(asks))]
            if len(a) > 1
        )
        if ask_vol > CONFIG.position_qty_epsilon:
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
            except (DivisionByZero, InvalidOperation):
                results["bid_ask_ratio"] = None

        log_parts = [
            f"BestBid={Fore.GREEN}{_format_for_log(results['best_bid'], 4)}{Style.RESET_ALL}",
            f"BestAsk={Fore.RED}{_format_for_log(results['best_ask'], 4)}{Style.RESET_ALL}",
            f"Spread={Fore.YELLOW}{_format_for_log(results['spread'], 4)}{Style.RESET_ALL}",
        ]
        if results["bid_ask_ratio"] is not None:
            log_parts.append(
                f"Ratio(B/A)={Fore.CYAN}{_format_for_log(results['bid_ask_ratio'], 3)}{Style.RESET_ALL}"
            )
        logger.debug(f"OB Scrying (Depth {depth}): {', '.join(log_parts)}")

    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}OB Scrying Error: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return results


# --- Data Fetching & Caching - Gathering Etheric Data Streams ---
_last_market_data: pd.DataFrame | None = None
_last_fetch_timestamp: float = 0.0


def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int
) -> pd.DataFrame | None:
    global _last_market_data, _last_fetch_timestamp
    current_time = time.time()

    try:
        candle_duration_seconds = exchange.parse_timeframe(interval)  # in seconds
    except Exception as e:  # Fallback if parse_timeframe fails or not supported
        logger.warning(
            f"Could not parse timeframe '{interval}' for caching: {e}. Cache disabled for this call."
        )
        candle_duration_seconds = 0  # Disables cache effectively

    cache_is_valid = (
        _last_market_data is not None
        and candle_duration_seconds > 0  # Ensure candle_duration is valid
        and (current_time - _last_fetch_timestamp)
        < (candle_duration_seconds * float(CONFIG.cache_candle_duration_multiplier))
        and len(_last_market_data) >= limit  # Ensure cache has enough data
    )

    if cache_is_valid:
        logger.debug(
            f"Data Fetch: Using CACHED market data ({len(_last_market_data)} candles). Last fetch: {time.strftime('%H:%M:%S', time.localtime(_last_fetch_timestamp))}"
        )
        return _last_market_data.copy()

    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}"
        )
        return None
    try:
        logger.debug(
            f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})..."
        )
        ohlcv = safe_api_call(
            exchange.fetch_ohlcv, symbol, timeframe=interval, limit=limit
        )
        if not ohlcv:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: No OHLCV data for {symbol}. Market inactive or API issue.{Style.RESET_ALL}"
            )
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.isnull().values.any():
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: NaNs found. Ffilling/Bfilling...{Style.RESET_ALL}"
            )
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            if df.isnull().values.any():
                logger.error(
                    f"{Fore.RED}Data Fetch: Unfillable NaNs remain. Data quality insufficient.{Style.RESET_ALL}"
                )
                return None

        _last_market_data = df.copy()
        _last_fetch_timestamp = current_time
        logger.debug(
            f"Data Fetch: Successfully woven {len(df)} OHLCV candles for {symbol}. Cached."
        )
        return df
    except Exception as e:
        logger.error(
            f"{Fore.RED}Data Fetch: Error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return None


# --- Account Health Check ---
def check_account_health(exchange: ccxt.Exchange, config: Config) -> bool:
    logger.debug("Performing account health check...")
    try:
        balance_params = {"category": "linear"}  # For Bybit V5 USDT futures
        balance = safe_api_call(exchange.fetch_balance, params=balance_params)

        total_equity = safe_decimal_conversion(
            balance.get(config.usdt_symbol, {}).get("total")
        )
        used_margin = safe_decimal_conversion(
            balance.get(config.usdt_symbol, {}).get("used")
        )

        if total_equity <= Decimal("0"):
            logger.warning(
                f"Account Health: Total equity {_format_for_log(total_equity)}. Margin ratio calc skipped."
            )
            if used_margin > Decimal("0"):
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Zero/Negative Equity ({_format_for_log(total_equity)}) with Used Margin ({_format_for_log(used_margin)})! Halting.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"CRITICAL: Zero/Neg Equity with Used Margin. Bot paused."
                )
                return False
            return True

        margin_ratio = used_margin / total_equity
        logger.info(
            f"Account Health: Equity={_format_for_log(total_equity, 2)}, UsedMargin={_format_for_log(used_margin, 2)}, MarginRatio={margin_ratio:.2%}"
        )

        if margin_ratio > config.max_account_margin_ratio:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: High margin ratio {margin_ratio:.2%} > {config.max_account_margin_ratio:.0%}. Halting.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"CRITICAL: High margin ratio {margin_ratio:.2%}. Bot paused."
            )
            return False
        return True
    except Exception as e:
        logger.error(f"Account health check failed: {type(e).__name__} - {e}")
        logger.debug(traceback.format_exc())
        return False  # Treat as uncertain/unhealthy on error


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    default_pos: dict[str, Any] = {
        "side": CONFIG.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
    }
    try:
        market = exchange.market(symbol)
        market_id = market["id"]
        category = (
            "linear"
            if market.get("linear")
            else ("inverse" if market.get("inverse") else None)
        )
        if not category:
            logger.error(
                f"{Fore.RED}Pos Check: No category for {symbol}.{Style.RESET_ALL}"
            )
            return default_pos

        params = {"category": category, "symbol": market_id}
        fetched_positions = safe_api_call(
            exchange.fetch_positions, symbols=[symbol], params=params
        )

        if not fetched_positions:
            return default_pos
        for pos in fetched_positions:
            pos_info = pos.get("info", {})
            if pos_info.get("symbol") != market_id:
                continue
            if int(pos_info.get("positionIdx", -1)) == 0:  # One-Way Mode
                size_dec = safe_decimal_conversion(pos_info.get("size", "0"))
                if size_dec > CONFIG.position_qty_epsilon and pos_info.get("side") in [
                    "Buy",
                    "Sell",
                ]:
                    entry_price = safe_decimal_conversion(pos_info.get("avgPrice"))
                    side = (
                        CONFIG.pos_long
                        if pos_info.get("side") == "Buy"
                        else CONFIG.pos_short
                    )
                    pos_color = Fore.GREEN if side == CONFIG.pos_long else Fore.RED
                    logger.info(
                        f"{pos_color}Pos Check: ACTIVE {side} Qty={size_dec:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}"
                    )
                    return {"side": side, "qty": size_dec, "entry_price": entry_price}
        logger.info(
            f"{Fore.BLUE}Pos Check: No active One-Way position for {market_id}. Flat.{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Pos Check: Error: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return default_pos


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    logger.info(
        f"{Fore.CYAN}Leverage: Setting {leverage}x for {symbol}...{Style.RESET_ALL}"
    )
    try:
        market = exchange.market(symbol)
        if not market.get("contract"):
            logger.error(
                f"{Fore.RED}Leverage: Not a contract market: {symbol}.{Style.RESET_ALL}"
            )
            return False

        # For Bybit V5, CCXT handles mapping to buyLeverage/sellLeverage.
        # Category might be needed if symbol alone is ambiguous, but usually not for set_leverage.
        # params = {"category": "linear"} # Usually not needed for set_leverage by symbol
        response = safe_api_call(
            exchange.set_leverage, leverage=leverage, symbol=symbol
        )  # params=params
        logger.success(
            f"{Fore.GREEN}Leverage: Set to {leverage}x for {symbol}. Resp: {response}{Style.RESET_ALL}"
        )
        return True
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        if any(
            sub in err_str
            for sub in ["leverage not modified", "same leverage", "110044"]
        ):
            logger.info(
                f"{Fore.CYAN}Leverage: Already {leverage}x for {symbol}.{Style.RESET_ALL}"
            )
            return True
        logger.error(f"{Fore.RED}Leverage: Failed: {e}{Style.RESET_ALL}")
    except Exception as e_unexp:
        logger.error(
            f"{Fore.RED}Leverage: Unexpected error: {e_unexp}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return False


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    position_to_close: dict[str, Any],
    reason: str = "Signal",
) -> dict[str, Any] | None:
    global _active_trade_details  # To log trade via TradeMetrics
    initial_side, initial_qty = (
        position_to_close.get("side", CONFIG.pos_none),
        position_to_close.get("qty", Decimal("0.0")),
    )
    market_base = symbol.split("/")[0].split(":")[0]
    logger.info(
        f"{Fore.YELLOW}Banish Position: {symbol} ({reason}). Initial: {initial_side} Qty={_format_for_log(initial_qty, 8)}{Style.RESET_ALL}"
    )

    live_position = get_current_position(exchange, symbol)
    if (
        live_position["side"] == CONFIG.pos_none
        or live_position["qty"] <= CONFIG.position_qty_epsilon
    ):
        logger.warning(
            f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position. Aborting.{Style.RESET_ALL}"
        )
        # If we thought there was a trade, but now it's gone, clear details
        if _active_trade_details["entry_price"] is not None:
            logger.info(
                "Clearing potentially stale active trade details as position is now flat."
            )
            _active_trade_details = {
                "entry_price": None,
                "entry_time_ms": None,
                "side": None,
                "qty": None,
            }
        return None

    side_to_execute = (
        CONFIG.side_sell
        if live_position["side"] == CONFIG.pos_long
        else CONFIG.side_buy
    )
    amount_to_close_str = format_amount(exchange, symbol, live_position["qty"])

    try:
        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}Banish Position: CLOSE {live_position['side']} ({reason}): {side_to_execute.upper()} MARKET {amount_to_close_str} {symbol}...{Style.RESET_ALL}"
        )
        params = {"reduceOnly": True, "category": "linear"}  # Bybit V5
        order = safe_api_call(
            exchange.create_market_order,
            symbol=symbol,
            side=side_to_execute,
            amount=float(amount_to_close_str),
            params=params,
        )

        status = order.get("status", "unknown")
        # For market orders, 'closed' implies filled. Check 'filled' amount.
        filled_qty_closed = safe_decimal_conversion(order.get("filled"))
        avg_fill_price_closed = safe_decimal_conversion(order.get("average"))

        if (
            status == "closed"
            and abs(filled_qty_closed - live_position["qty"])
            < CONFIG.position_qty_epsilon
        ):
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}Banish Position: CONFIRMED FILLED ({reason}). ID:...{format_order_id(order.get('id'))}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}/{CONFIG.strategy_name.value}] BANISHED {live_position['side']} {amount_to_close_str} ({reason})."
            )

            # Log to TradeMetrics
            if (
                _active_trade_details["entry_price"] is not None
                and order.get("timestamp") is not None
            ):
                trade_metrics.log_trade(
                    symbol=symbol,
                    side=_active_trade_details["side"],
                    entry_price=_active_trade_details["entry_price"],
                    exit_price=avg_fill_price_closed,
                    qty=_active_trade_details["qty"],  # Use original entry qty
                    entry_time_ms=_active_trade_details["entry_time_ms"],
                    exit_time_ms=order["timestamp"],
                    reason=reason,
                )
            _active_trade_details = {
                "entry_price": None,
                "entry_time_ms": None,
                "side": None,
                "qty": None,
            }  # Reset
            return order
        else:
            logger.warning(
                f"{Fore.YELLOW}Banish Position: Fill uncertain. Expected {live_position['qty']}, Filled {filled_qty_closed}. ID:...{format_order_id(order.get('id'))}, Status: {status}.{Style.RESET_ALL}"
            )
            return order  # Return for potential checks

    except Exception as e:
        logger.error(
            f"{Fore.RED}Banish Position ({reason}): Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] ERROR Banishing ({reason}): {type(e).__name__}."
        )
        logger.debug(traceback.format_exc())
    return None


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
        logger.error(f"{Fore.RED}RiskCalc: Invalid inputs.{Style.RESET_ALL}")
        return None, None
    price_diff = abs(entry_px - sl_px)
    if price_diff < CONFIG.position_qty_epsilon:
        logger.error(f"{Fore.RED}RiskCalc: Entry/SL too close.{Style.RESET_ALL}")
        return None, None
    try:
        risk_amt_usdt = equity * risk_pct
        qty_raw = risk_amt_usdt / price_diff
        qty_prec_str = format_amount(ex, sym, qty_raw)
        qty_prec = Decimal(qty_prec_str)
        if qty_prec <= CONFIG.position_qty_epsilon:
            logger.warning(
                f"{Fore.YELLOW}RiskCalc: Qty negligible ({qty_prec}).{Style.RESET_ALL}"
            )
            return None, None
        pos_val_usdt = qty_prec * entry_px
        margin_req = pos_val_usdt / Decimal(lev)
        logger.debug(
            f"RiskCalc: Qty={Fore.CYAN}{_format_for_log(qty_prec, 8)}{Style.RESET_ALL}, MarginReq={_format_for_log(margin_req, 4)}"
        )
        return qty_prec, margin_req
    except Exception as e:
        logger.error(f"{Fore.RED}RiskCalc: Error: {e}{Style.RESET_ALL}")
        return None, None


def wait_for_order_fill(
    exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_s: int
) -> dict[str, Any] | None:
    start_time = time.time()
    oid_short = format_order_id(order_id)
    logger.info(
        f"{Fore.CYAN}Observing order ...{oid_short} ({symbol}) for fill (Timeout: {timeout_s}s)...{Style.RESET_ALL}"
    )
    params = {"category": "linear"}  # Bybit V5 may need category
    while time.time() - start_time < timeout_s:
        try:
            order = safe_api_call(exchange.fetch_order, order_id, symbol, params=params)
            status = order.get("status")
            if status == "closed":
                logger.success(
                    f"{Fore.GREEN}Order ...{oid_short} FILLED/CLOSED.{Style.RESET_ALL}"
                )
                return order
            if status in ["canceled", "rejected", "expired"]:
                logger.error(
                    f"{Fore.RED}Order ...{oid_short} FAILED: {status}.{Style.RESET_ALL}"
                )
                return order
            time.sleep(0.75)
        except ccxt.OrderNotFound:
            time.sleep(1.5)  # Propagation delay
        except Exception as e:
            logger.warning(
                f"{Fore.YELLOW}Error checking order ...{oid_short}: {type(e).__name__}. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(2)
    logger.error(
        f"{Fore.RED}Order ...{oid_short} TIMEOUT after {timeout_s}s.{Style.RESET_ALL}"
    )
    try:
        return safe_api_call(
            exchange.fetch_order, order_id, symbol, params=params
        )  # Final check
    except:
        return None


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: Decimal,
    current_atr: Decimal | None,
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal,
) -> dict[str, Any] | None:
    global _active_trade_details
    market_base = symbol.split("/")[0].split(":")[0]
    logger.info(
        f"{Fore.CYAN if side == CONFIG.side_buy else Fore.MAGENTA}{Style.BRIGHT}Place Order: {side.upper()} for {symbol}...{Style.RESET_ALL}"
    )
    if current_atr is None or current_atr <= 0:
        logger.error(
            f"{Fore.RED}Place Order Error: Invalid ATR ({current_atr}).{Style.RESET_ALL}"
        )
        return None
    v5_category = "linear"
    try:
        balance = safe_api_call(
            exchange.fetch_balance, params={"category": v5_category}
        )
        market = exchange.market(symbol)
        min_qty = safe_decimal_conversion(
            market.get("limits", {}).get("amount", {}).get("min")
        )
        usdt_equity = safe_decimal_conversion(
            balance.get(CONFIG.usdt_symbol, {}).get("total")
        )
        usdt_free = safe_decimal_conversion(
            balance.get(CONFIG.usdt_symbol, {}).get("free")
        )
        if usdt_equity <= 0 or usdt_free < 0:
            logger.error(
                f"{Fore.RED}Place Order Error: Invalid equity/free margin.{Style.RESET_ALL}"
            )
            return None

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
        if not entry_px_est or entry_px_est <= 0:
            ticker = safe_api_call(exchange.fetch_ticker, symbol)
            entry_px_est = safe_decimal_conversion(ticker.get("last"))
        if not entry_px_est or entry_px_est <= 0:
            logger.error(
                f"{Fore.RED}Place Order Error: Failed to get entry price estimate.{Style.RESET_ALL}"
            )
            return None

        sl_dist = current_atr * sl_atr_multiplier
        sl_px_raw = (
            (entry_px_est - sl_dist)
            if side == CONFIG.side_buy
            else (entry_px_est + sl_dist)
        )
        sl_px_est_str = format_price(exchange, symbol, sl_px_raw)
        sl_px_est = Decimal(sl_px_est_str)
        if sl_px_est <= 0:
            logger.error(
                f"{Fore.RED}Place Order: Invalid SL estimate.{Style.RESET_ALL}"
            )
            return None

        final_qty, margin_est = calculate_position_size(
            usdt_equity,
            risk_percentage,
            entry_px_est,
            sl_px_est,
            leverage,
            symbol,
            exchange,
        )
        if final_qty is None or margin_est is None:
            return None

        pos_val_est = final_qty * entry_px_est
        if pos_val_est > max_order_cap_usdt:
            final_qty = Decimal(
                format_amount(exchange, symbol, max_order_cap_usdt / entry_px_est)
            )
            margin_est = (final_qty * entry_px_est) / Decimal(leverage)
        if min_qty and final_qty < min_qty:
            logger.error(
                f"{Fore.RED}Place Order: Qty {_format_for_log(final_qty, 8)} < Min Qty {_format_for_log(min_qty, 8)}.{Style.RESET_ALL}"
            )
            return None
        if usdt_free < margin_est * margin_check_buffer:
            logger.error(
                f"{Fore.RED}Place Order: Insufficient FREE margin.{Style.RESET_ALL}"
            )
            return None

        entry_params = {"reduceOnly": False, "category": v5_category}
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
                f"{Back.RED}{Fore.WHITE}CRITICAL: Entry order NO ID!{Style.RESET_ALL}"
            )
            return None

        filled_entry = wait_for_order_fill(
            exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds
        )
        if not filled_entry or filled_entry.get("status") != "closed":
            logger.error(
                f"{Fore.RED}Entry order ...{format_order_id(order_id)} not filled. Status: {filled_entry.get('status') if filled_entry else 'timeout'}.{Style.RESET_ALL}"
            )
            # Attempt to cancel if not filled
            try:
                safe_api_call(
                    exchange.cancel_order,
                    order_id,
                    symbol,
                    params={"category": v5_category},
                )
            except Exception as e_cancel:
                logger.warning(
                    f"Could not cancel unfilled order {order_id}: {e_cancel}"
                )
            return None

        avg_fill_px = safe_decimal_conversion(filled_entry.get("average"))
        filled_qty_val = safe_decimal_conversion(filled_entry.get("filled"))
        if (
            filled_qty_val <= CONFIG.position_qty_epsilon
            or avg_fill_px <= CONFIG.position_qty_epsilon
        ):
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill qty/price for ...{format_order_id(order_id)}.{Style.RESET_ALL}"
            )
            return filled_entry

        logger.success(
            f"{Fore.GREEN if side == CONFIG.side_buy else Fore.RED}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {_format_for_log(filled_qty_val, 8)} @ Avg: {_format_for_log(avg_fill_px, 4)}{Style.RESET_ALL}"
        )
        _active_trade_details = {
            "entry_price": avg_fill_px,
            "entry_time_ms": filled_entry.get("timestamp"),
            "side": side,
            "qty": filled_qty_val,
        }

        # Place Fixed SL
        actual_sl_px_raw = (
            (avg_fill_px - sl_dist)
            if side == CONFIG.side_buy
            else (avg_fill_px + sl_dist)
        )
        actual_sl_px_str = format_price(exchange, symbol, actual_sl_px_raw)
        if Decimal(actual_sl_px_str) <= 0:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL: ACTUAL SL invalid!{Style.RESET_ALL}"
            )
            return filled_entry  # Potentially close pos
        sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
        sl_params = {
            "stopPrice": float(actual_sl_px_str),
            "reduceOnly": True,
            "category": v5_category,
            "positionIdx": 0,
        }
        try:
            sl_order = safe_api_call(
                exchange.create_order,
                symbol,
                "stopMarket",
                sl_side,
                float(filled_qty_val),
                params=sl_params,
            )
            logger.success(
                f"{Fore.GREEN}Fixed SL placed. ID:...{format_order_id(sl_order.get('id'))}, Trigger:{actual_sl_px_str}{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}FAILED Fixed SL: {e}{Style.RESET_ALL}")

        # Place TSL
        act_offset = avg_fill_px * tsl_activation_offset_percent
        act_price_raw = (
            (avg_fill_px + act_offset)
            if side == CONFIG.side_buy
            else (avg_fill_px - act_offset)
        )
        tsl_act_px_str = format_price(exchange, symbol, act_price_raw)
        if Decimal(tsl_act_px_str) <= 0:
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL: TSL Act Price invalid!{Style.RESET_ALL}"
            )
            return filled_entry
        tsl_trail_val_str = str((tsl_percent * Decimal("100")).normalize())
        tsl_params = {
            "trailingStop": tsl_trail_val_str,
            "activePrice": float(tsl_act_px_str),
            "reduceOnly": True,
            "category": v5_category,
            "positionIdx": 0,
        }
        try:
            tsl_order = safe_api_call(
                exchange.create_order,
                symbol,
                "stopMarket",
                sl_side,
                float(filled_qty_val),
                params=tsl_params,
            )  # sl_side is same for TSL
            logger.success(
                f"{Fore.GREEN}Trailing SL placed. ID:...{format_order_id(tsl_order.get('id'))}, Trail%:{tsl_trail_val_str}, ActPx:{tsl_act_px_str}{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}FAILED TSL: {e}{Style.RESET_ALL}")

        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] ENTERED {side.upper()} {_format_for_log(filled_qty_val, 8)} @ {_format_for_log(avg_fill_px, 4)}. SL:~{actual_sl_px_str}, TSL:{tsl_percent:.2%}@~{tsl_act_px_str}. EntryID:...{format_order_id(order_id)}"
        )
        return filled_entry
    except Exception as e:
        logger.error(
            f"{Back.RED}{Fore.WHITE}Place Order Ritual FAILED: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] ORDER FAIL ({side.upper()}): {type(e).__name__}"
        )
    return None


def cancel_open_orders(
    exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup"
) -> int:
    logger.info(
        f"{Fore.CYAN}Order Cleanup: {symbol} (Reason: {reason})...{Style.RESET_ALL}"
    )
    cancelled_count, failed_count = 0, 0
    v5_category = "linear"
    try:
        open_orders = safe_api_call(
            exchange.fetch_open_orders, symbol, params={"category": v5_category}
        )
        if not open_orders:
            logger.info(
                f"{Fore.CYAN}Order Cleanup: No open orders for {symbol}.{Style.RESET_ALL}"
            )
            return 0
        logger.warning(
            f"{Fore.YELLOW}Order Cleanup: Found {len(open_orders)} open orders. Cancelling...{Style.RESET_ALL}"
        )
        for order in open_orders:
            order_id = order.get("id")
            if order_id:
                try:
                    safe_api_call(
                        exchange.cancel_order,
                        order_id,
                        symbol,
                        params={"category": v5_category},
                    )
                    cancelled_count += 1
                except ccxt.OrderNotFound:
                    cancelled_count += 1  # Already gone
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Order Cleanup: FAILED cancel ...{format_order_id(order_id)}: {e}{Style.RESET_ALL}"
                    )
                    failed_count += 1
            else:
                logger.error(
                    f"{Fore.RED}Order Cleanup: Open order with no ID.{Style.RESET_ALL}"
                )
                failed_count += 1
        if failed_count > 0:
            send_sms_alert(
                f"[{symbol.split('/')[0]}/{CONFIG.strategy_name.value}] WARNING: Failed cancel {failed_count} orders ({reason})."
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Order Cleanup: Error fetching/processing: {e}{Style.RESET_ALL}"
        )
    logger.info(
        f"{Fore.CYAN}Order Cleanup: Cancelled/Handled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}"
    )
    return cancelled_count


# --- Strategy Signal Generation Wrapper ---
def generate_strategy_signals(
    df: pd.DataFrame, strategy_instance: TradingStrategy
) -> dict[str, Any]:
    if strategy_instance:
        return strategy_instance.generate_signals(df)
    logger.error("Unknown strategy instance provided for signal generation.")
    return strategy_instance._get_default_signals()  # type: ignore


# --- All Indicator Calculations ---
def calculate_all_indicators(
    df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculates all indicators and returns the DataFrame and volume/ATR analysis."""
    df = calculate_supertrend(df, config.st_atr_length, config.st_multiplier)
    df = calculate_supertrend(
        df,
        config.confirm_st_atr_length,
        config.confirm_st_multiplier,
        prefix="confirm_",
    )
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
    vol_atr_data = analyze_volume_atr(
        df, config.atr_calculation_period, config.volume_ma_period
    )
    return df, vol_atr_data


# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    cycle_time_str = (
        df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z") if not df.empty else "N/A"
    )
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== Cycle ({CONFIG.strategy_name.value}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}"
    )

    # Calculate required rows dynamically - simplified, ensure data_limit in main loop is generous
    required_rows = 50  # A general baseline, specific strategies might need more
    if df is None or len(df) < required_rows:
        logger.warning(
            f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0} rows).{Style.RESET_ALL}"
        )
        return

    try:
        df, vol_atr_data = calculate_all_indicators(df.copy(), CONFIG)  # Use copy
        current_atr = vol_atr_data.get("atr")
        last_candle = df.iloc[-1] if not df.empty else pd.Series(dtype="object")
        current_price = safe_decimal_conversion(last_candle.get("close"))

        if pd.isna(current_price) or current_price <= 0:
            logger.warning(
                f"{Fore.YELLOW}Trade Logic: Invalid last close price.{Style.RESET_ALL}"
            )
            return
        can_place_order = current_atr is not None and current_atr > 0

        position = get_current_position(exchange, symbol)
        pos_side, pos_qty, pos_entry = (
            position["side"],
            position["qty"],
            position["entry_price"],
        )

        ob_data = None
        if CONFIG.fetch_order_book_per_cycle or (
            pos_side == CONFIG.pos_none and can_place_order
        ):  # Fetch if flat and might enter
            ob_data = analyze_order_book(
                exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
            )

        # Log Snapshot
        logger.info(f"{Fore.MAGENTA}--- Indicator Snapshot ---{Style.RESET_ALL}")
        logger.info(
            f"  Market: Close={_format_for_log(current_price, 4)}, ATR({CONFIG.atr_calculation_period})={_format_for_log(current_atr, 5)}"
        )
        is_vol_spike = (
            vol_atr_data.get("volume_ratio") is not None
            and vol_atr_data["volume_ratio"] > CONFIG.volume_spike_threshold
        )  # type: ignore
        logger.info(
            f"  Volume: Ratio={_format_for_log(vol_atr_data.get('volume_ratio'), 2)}, Spike={is_vol_spike}"
        )
        if ob_data:
            logger.info(
                f"  OrderBook: Ratio(B/A)={_format_for_log(ob_data.get('bid_ask_ratio'), 3)}, Spread={_format_for_log(ob_data.get('spread'), 4)}"
            )
        # Dynamic strategy indicator logging
        CONFIG.strategy_instance.logger.info(
            f"  Strategy Values ({CONFIG.strategy_name.value}):"
        )  # Use strategy's logger
        for col_name in CONFIG.strategy_instance.required_columns:
            if col_name in last_candle.index:
                is_trend = "trend" in col_name.lower()
                CONFIG.strategy_instance.logger.info(
                    f"    {col_name}: {_format_for_log(last_candle[col_name], is_bool_trend=is_trend)}"
                )
        pos_color = (
            Fore.GREEN
            if pos_side == CONFIG.pos_long
            else (Fore.RED if pos_side == CONFIG.pos_short else Fore.BLUE)
        )
        logger.info(
            f"  Position: Side={pos_color}{pos_side}{Style.RESET_ALL}, Qty={_format_for_log(pos_qty, 8)}, Entry={_format_for_log(pos_entry, 4)}"
        )
        logger.info(f"{Fore.MAGENTA}{'-' * 26}{Style.RESET_ALL}")

        strategy_signals = generate_strategy_signals(df, CONFIG.strategy_instance)
        should_exit_long = pos_side == CONFIG.pos_long and strategy_signals["exit_long"]
        should_exit_short = (
            pos_side == CONFIG.pos_short and strategy_signals["exit_short"]
        )

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals["exit_reason"]
            logger.warning(
                f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}*** EXIT SIGNAL: Closing {pos_side} ({exit_reason}) ***{Style.RESET_ALL}"
            )
            cancel_open_orders(exchange, symbol, f"Pre-Exit ({exit_reason})")
            time.sleep(0.5)
            close_result = close_position(
                exchange, symbol, position, reason=exit_reason
            )
            if close_result:
                time.sleep(CONFIG.post_close_delay_seconds)
            return

        if pos_side != CONFIG.pos_none:
            logger.info(
                f"Holding {pos_color}{pos_side}{Style.RESET_ALL}. Awaiting SL/TSL or Exit Signal."
            )
            return
        if not can_place_order:
            logger.warning(
                f"{Fore.YELLOW}Holding Cash. Cannot enter: Invalid ATR.{Style.RESET_ALL}"
            )
            return

        potential_entry = (
            strategy_signals["enter_long"] or strategy_signals["enter_short"]
        )
        if not potential_entry:
            logger.info("Holding Cash. No entry signal.")
            return

        ob_confirm_long, ob_confirm_short = True, True
        if ob_data and ob_data.get("bid_ask_ratio") is not None:
            ratio = ob_data["bid_ask_ratio"]
            if CONFIG.order_book_ratio_threshold_long < Decimal("Infinity"):
                ob_confirm_long = ratio >= CONFIG.order_book_ratio_threshold_long  # type: ignore
            if CONFIG.order_book_ratio_threshold_short > Decimal(0):
                ob_confirm_short = ratio <= CONFIG.order_book_ratio_threshold_short  # type: ignore
        elif CONFIG.order_book_ratio_threshold_long < Decimal(
            "Infinity"
        ) or CONFIG.order_book_ratio_threshold_short > Decimal(0):
            ob_confirm_long, ob_confirm_short = False, False  # Required but no data
        vol_confirm = not CONFIG.require_volume_spike_for_entry or is_vol_spike

        final_enter_long = (
            strategy_signals["enter_long"] and ob_confirm_long and vol_confirm
        )
        final_enter_short = (
            strategy_signals["enter_short"] and ob_confirm_short and vol_confirm
        )

        entry_params = {
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
                f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** CONFIRMED LONG ENTRY ({CONFIG.strategy_name.value}) ***{Style.RESET_ALL}"
            )
            cancel_open_orders(exchange, symbol, "Pre-Long Entry")
            time.sleep(0.5)
            place_risked_market_order(side=CONFIG.side_buy, **entry_params)  # type: ignore
        elif final_enter_short:
            logger.success(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** CONFIRMED SHORT ENTRY ({CONFIG.strategy_name.value}) ***{Style.RESET_ALL}"
            )
            cancel_open_orders(exchange, symbol, "Pre-Short Entry")
            time.sleep(0.5)
            place_risked_market_order(side=CONFIG.side_sell, **entry_params)  # type: ignore
        elif potential_entry:
            logger.info("Holding Cash. Signal present but filters not met.")

    except Exception as e:
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL ERROR in trade_logic: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        send_sms_alert(
            f"[{symbol.split('/')[0]}/{CONFIG.strategy_name.value}] CRITICAL trade_logic ERROR: {type(e).__name__}."
        )
    finally:
        logger.info(
            f"{Fore.BLUE}{Style.BRIGHT}========== Cycle End: {symbol} =========={Style.RESET_ALL}\n"
        )


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    logger.warning(
        f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing energies...{Style.RESET_ALL}"
    )
    market_base = (
        symbol.split("/")[0].split(":")[0]
        if symbol and "/" in symbol and ":" in symbol
        else (symbol if symbol else "Bot")
    )
    strat_name_val = (
        CONFIG.strategy_name.value
        if "CONFIG" in globals() and hasattr(CONFIG, "strategy_name")
        else "N/A"
    )
    send_sms_alert(
        f"[{market_base}/{strat_name_val}] Shutdown initiated. Cleanup attempt..."
    )

    if trade_metrics and hasattr(trade_metrics, "summary"):
        trade_metrics.summary()  # Log final metrics

    if not exchange or not symbol:
        logger.warning(
            f"{Fore.YELLOW}Shutdown: Exchange/Symbol not defined. No automated cleanup.{Style.RESET_ALL}"
        )
        return
    try:
        logger.warning("Shutdown: Cancelling open orders...")
        cancel_open_orders(exchange, symbol, "Shutdown")
        time.sleep(1.5)
        position = get_current_position(exchange, symbol)
        if position["side"] != CONFIG.pos_none and position["qty"] > 0:
            logger.warning(
                f"{Fore.YELLOW}Shutdown: Active {position['side']} position. Closing...{Style.RESET_ALL}"
            )
            close_result = close_position(exchange, symbol, position, "Shutdown")
            if close_result:
                logger.info(
                    f"{Fore.CYAN}Shutdown: Close order placed. Final check after delay...{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.post_close_delay_seconds * 2)
                final_pos = get_current_position(exchange, symbol)
                if final_pos["side"] == CONFIG.pos_none:
                    logger.success(
                        f"{Fore.GREEN}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}"
                    )
                else:
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN: FAILED CONFIRM closure! Final: {final_pos['side']} Qty={final_pos['qty']}. MANUAL CHECK!{Style.RESET_ALL}"
                    )
            else:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN: FAILED PLACE close order! MANUAL CHECK!{Style.RESET_ALL}"
                )
        else:
            logger.info(
                f"{Fore.GREEN}Shutdown: No active position. Clean exit.{Style.RESET_ALL}"
            )
    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    logger.info(
        f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Spell Shutdown Complete ---{Style.RESET_ALL}"
    )


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.3.0 Initializing ({start_time_str}) ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}--- Strategy: {CONFIG.strategy_name.value} ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.GREEN}--- Protections: ATR-Stop + Exchange TSL (Bybit V5) ---{Style.RESET_ALL}"
    )
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING - EXTREME RISK - EDUCATIONAL USE ONLY !!! ---{Style.RESET_ALL}"
    )

    exchange: ccxt.Exchange | None = None
    symbol_unified: str | None = None
    run_bot: bool = True
    cycle_count: int = 0

    try:
        exchange = initialize_exchange()
        if not exchange:
            return
        market = exchange.market(CONFIG.symbol)
        symbol_unified = market["symbol"]
        if not market.get("contract"):
            raise ValueError(f"Market '{symbol_unified}' not a contract market.")
        logger.info(
            f"{Fore.GREEN}Spell focused on: {symbol_unified} (Type: {market.get('type', 'N/A')}){Style.RESET_ALL}"
        )
        if not set_leverage(exchange, symbol_unified, CONFIG.leverage):
            raise RuntimeError(f"Leverage setting failed for {symbol_unified}.")

        # Log key config details
        logger.info(f"{Fore.MAGENTA}--- Spell Config Summary ---{Style.RESET_ALL}")
        logger.info(
            f"Symbol: {symbol_unified}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x, Strategy: {CONFIG.strategy_name.value}"
        )
        logger.info(
            f"Risk: {CONFIG.risk_per_trade_percentage:.2%}/trade, MaxCap: {CONFIG.max_order_usdt_amount} USDT, ATR SL Mult: {CONFIG.atr_stop_loss_multiplier}"
        )
        logger.info(
            f"TSL: {CONFIG.trailing_stop_percentage:.2%}, TSL Act. Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}"
        )
        logger.info(f"Account Max Margin Ratio: {CONFIG.max_account_margin_ratio:.0%}")
        logger.info(f"{Fore.MAGENTA}{'-' * 26}{Style.RESET_ALL}")

        market_base = symbol_unified.split("/")[0].split(":")[0]
        send_sms_alert(
            f"[{market_base}/{CONFIG.strategy_name.value}] Pyrmethus v2.3.0 Initialized. Symbol: {symbol_unified}. Starting loop."
        )

        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(
                f"{Fore.CYAN}--- Cycle {cycle_count} Start ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}"
            )

            if not check_account_health(exchange, CONFIG):
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}ACCOUNT HEALTH CHECK FAILED! Pausing bot for safety.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL: Account health fail! Bot paused."
                )
                time.sleep(CONFIG.sleep_seconds * 10)  # Long pause
                continue  # Skip to next cycle for re-check

            try:
                # More generous data limit for indicators
                data_limit = (
                    max(
                        CONFIG.st_atr_length,
                        CONFIG.confirm_st_atr_length,
                        CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length,
                        CONFIG.ehlers_fisher_length,
                        CONFIG.ehlers_fast_period,
                        CONFIG.ehlers_slow_period,
                        CONFIG.atr_calculation_period,
                        100,
                    )
                    + CONFIG.api_fetch_limit_buffer
                    + 20
                )

                df = get_market_data(
                    exchange, symbol_unified, CONFIG.interval, limit=data_limit
                )
                if df is not None and not df.empty:
                    trade_logic(exchange, symbol_unified, df)
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Skipping logic: invalid/missing market data.{Style.RESET_ALL}"
                    )

            except ccxt.RateLimitExceeded as e:
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}Rate Limit: {e}. Sleeping longer...{Style.RESET_ALL}"
                )
                time.sleep(CONFIG.sleep_seconds * 6)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                logger.warning(
                    f"{Fore.YELLOW}Network/Exchange issue: {e}. Retrying after pause.{Style.RESET_ALL}"
                )
                time.sleep(
                    CONFIG.sleep_seconds
                    * (6 if isinstance(e, ccxt.ExchangeNotAvailable) else 2)
                )
            except ccxt.AuthenticationError as e:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}FATAL: Auth Error: {e}. Stopping.{Style.RESET_ALL}"
                )
                run_bot = False
            except Exception as e:
                logger.exception(
                    f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL ERROR in Main Loop: {e} !!! Stopping!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Bot stopping."
                )
                run_bot = False

            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(
                    f"Cycle {cycle_count} duration: {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s."
                )
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.warning(
            f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt. Withdrawing...{Style.RESET_ALL}"
        )
        run_bot = False
    except Exception as startup_error:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL STARTUP ERROR: {startup_error}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        if (
            "CONFIG" in globals()
            and hasattr(CONFIG, "enable_sms_alerts")
            and CONFIG.enable_sms_alerts
        ):
            send_sms_alert(
                f"[Pyrmethus] CRITICAL STARTUP ERROR: {type(startup_error).__name__}."
            )
        run_bot = False
    finally:
        graceful_shutdown(exchange, symbol_unified)
        logger.info(
            f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    main()
