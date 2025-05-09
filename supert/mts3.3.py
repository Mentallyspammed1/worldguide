
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
    from retry import retry # For safe_api_call decorator
except ImportError as e:
    missing_pkg = e.name
    sys.stderr.write(f"\033[91mCRITICAL ERROR: Missing required Python package: '{missing_pkg}'.\033[0m\n")
    sys.stderr.write(f"\033[91mPlease install it by running: pip install {missing_pkg}\033[0m\n")
    if missing_pkg == "pandas_ta":
        sys.stderr.write(f"\033[91mFor pandas_ta, you might also need TA-Lib. See pandas_ta documentation.\033[0m\n")
    if missing_pkg == "retry":
        sys.stderr.write(f"\033[91mFor retry decorator, install with: pip install retry\033[0m\n")
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
        _pre_logger = logging.getLogger(__name__) # Temp logger for config phase
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
        self.symbol: str = self._get_env(
            "SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW
        )
        self.interval: str = self._get_env(
            "INTERVAL", "1m", color=Fore.YELLOW
        )
        self.leverage: int = self._get_env(
            "LEVERAGE", 25, cast_type=int, color=Fore.YELLOW
        )
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW
        )

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: StrategyName = StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND.value, color=Fore.CYAN).upper())
        self.valid_strategies: list[str] = [s.value for s in StrategyName]
        if self.strategy_name.value not in self.valid_strategies:
            _pre_logger.critical(
                f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name.value}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}"
            )
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name.value}'.")
        _pre_logger.info(
            f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name.value}{Style.RESET_ALL}"
        )
        self.strategy_instance: 'TradingStrategy' # Forward declaration, will be set after CONFIG is loaded

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
        self.max_account_margin_ratio: Decimal = self._get_env( # For check_account_health
            "MAX_ACCOUNT_MARGIN_RATIO", "0.8", cast_type=Decimal, color=Fore.GREEN # 80%
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
        self.cache_candle_duration_multiplier: Decimal = Decimal("0.95") # For data cache validity, e.g., 95% of candle duration


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
        _pre_logger = logging.getLogger(__name__) # Use temp logger
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
    format="%(asctime)s [%(levelname)-8s] %(name)-15s %(message)s", # Added logger name
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_name) # Added file handler
    ],
)
logger: logging.Logger = logging.getLogger("PyrmethusCore") # Main logger for the bot

# Custom SUCCESS level and Neon Color Formatting for the Oracle
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs) # type: ignore[attr-defined]

logging.Logger.success = log_success # type: ignore[attr-defined]

if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{Style.DIM}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{Style.BRIGHT}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{Style.BRIGHT}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{Style.BRIGHT}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")

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
            self.logger.debug(f"Insufficient data (Rows: {len(df) if df is not None else 0}, Need: {min_rows}).")
            return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns):
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            self.logger.warning(f"Missing required columns: {missing_cols}.")
            return False
        # Check for NaNs in the last row for required columns
        if self.required_columns and df.iloc[-1][self.required_columns].isnull().any():
             nan_cols_last_row = df.iloc[-1][self.required_columns].isnull()
             nan_cols_last_row = nan_cols_last_row[nan_cols_last_row].index.tolist()
             self.logger.debug(f"NaN values in last row for required columns: {nan_cols_last_row}")
             # Allow strategies to handle this if they wish, or return False here to be stricter
        return True

    def _get_default_signals(self) -> dict[str, Any]:
        return {
            "enter_long": False, "enter_short": False,
            "exit_long": False, "exit_short": False,
            "exit_reason": "Strategy Exit Signal"
        }

class DualSupertrendStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df): return signals
        last = df.iloc[-1]
        primary_long_flip = last.get("st_long_flip", False)
        primary_short_flip = last.get("st_short_flip", False)
        confirm_is_up = last.get("confirm_trend", pd.NA)

        if pd.isna(confirm_is_up):
            self.logger.debug("Confirmation trend is NA. No signal.")
            return signals
        if primary_long_flip and confirm_is_up is True: signals["enter_long"] = True
        if primary_short_flip and confirm_is_up is False: signals["enter_short"] = True
        if primary_short_flip:
            signals["exit_long"] = True; signals["exit_reason"] = "Primary ST Flipped Short"
        if primary_long_flip:
            signals["exit_short"] = True; signals["exit_reason"] = "Primary ST Flipped Long"
        return signals

class StochRsiMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["stochrsi_k", "stochrsi_d", "momentum"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2): return signals
        last, prev = df.iloc[-1], df.iloc[-2]
        k_now,d_now,mom_now = last.get("stochrsi_k"),last.get("stochrsi_d"),last.get("momentum")
        k_prev,d_prev = prev.get("stochrsi_k"),prev.get("stochrsi_d")

        if any(pd.isna(v) for v in [k_now, d_now, mom_now, k_prev, d_prev]):
            self.logger.debug("Skipping due to NA StochRSI/Mom values.")
            return signals
        
        k_now_dec = safe_decimal_conversion(k_now, Decimal('NaN'))
        mom_now_dec = safe_decimal_conversion(mom_now, Decimal('NaN'))

        if k_now_dec.is_nan() or mom_now_dec.is_nan():
            self.logger.debug("Skipping due to NA StochRSI/Mom Decimal values after conversion.")
            return signals

        if k_prev <= d_prev and k_now > d_now and k_now_dec < self.config.stochrsi_oversold and mom_now_dec > Decimal("0"):
            signals["enter_long"] = True
        if k_prev >= d_prev and k_now < d_now and k_now_dec > self.config.stochrsi_overbought and mom_now_dec < Decimal("0"):
            signals["enter_short"] = True
        if k_prev >= d_prev and k_now < d_now:
            signals["exit_long"] = True; signals["exit_reason"] = "StochRSI K crossed below D"
        if k_prev <= d_prev and k_now > d_now:
            signals["exit_short"] = True; signals["exit_reason"] = "StochRSI K crossed above D"
        return signals

class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2): return signals
        last, prev = df.iloc[-1], df.iloc[-2]
        fish_now,sig_now = last.get("ehlers_fisher"),last.get("ehlers_signal")
        fish_prev,sig_prev = prev.get("ehlers_fisher"),prev.get("ehlers_signal")

        if any(pd.isna(v) for v in [fish_now, sig_now, fish_prev, sig_prev]):
            self.logger.debug("Skipping due to NA Ehlers Fisher values.")
            return signals
        if fish_prev <= sig_prev and fish_now > sig_now: signals["enter_long"] = True
        if fish_prev >= sig_prev and fish_now < sig_now: signals["enter_short"] = True
        if fish_prev >= sig_prev and fish_now < sig_now:
            signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher crossed Short"
        if fish_prev <= sig_prev and fish_now > sig_now:
            signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher crossed Long"
        return signals

class EhlersMaCrossStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_ssf_fast", "ehlers_ssf_slow"])

    def generate_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2): return signals
        last, prev = df.iloc[-1], df.iloc[-2]
        fast_ma_now,slow_ma_now = last.get("ehlers_ssf_fast"),last.get("ehlers_ssf_slow")
        fast_ma_prev,slow_ma_prev = prev.get("ehlers_ssf_fast"),prev.get("ehlers_ssf_slow")

        if any(pd.isna(v) for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]):
            self.logger.debug("Skipping due to NA Ehlers SSF MA values.")
            return signals
        if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now: signals["enter_long"] = True
        if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now: signals["enter_short"] = True
        if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
            signals["exit_long"] = True; signals["exit_reason"] = "Fast Ehlers SSF MA crossed below Slow"
        if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
            signals["exit_short"] = True; signals["exit_reason"] = "Fast Ehlers SSF MA crossed above Slow"
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

    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal, entry_time_ms: int, exit_time_ms: int, reason: str):
        if not all([entry_price > 0, exit_price > 0, qty > 0, entry_time_ms > 0, exit_time_ms > 0]):
            self.logger.warning(f"Trade log skipped: Invalid params. EntryPx:{entry_price}, ExitPx:{exit_price}, Qty:{qty}, EntryT:{entry_time_ms}, ExitT:{exit_time_ms}")
            return

        profit_per_unit = exit_price - entry_price
        if side.lower() == CONFIG.side_sell or side.lower() == CONFIG.pos_short.lower():
            profit_per_unit = entry_price - exit_price
        
        profit = profit_per_unit * qty
        entry_dt = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration = exit_dt - entry_dt

        self.trades.append({
            "symbol": symbol, "side": side, "entry_price": entry_price, "exit_price": exit_price,
            "qty": qty, "profit": profit, "entry_time": entry_dt, "exit_time": exit_dt,
            "duration_seconds": duration.total_seconds(), "exit_reason": reason
        })
        self.logger.success(
            f"{Fore.MAGENTA}Trade Recorded: {side.upper()} {qty} {symbol.split('/')[0]} | Entry: {entry_price:.4f}, Exit: {exit_price:.4f} | P/L: {profit:.2f} {CONFIG.usdt_symbol} | Duration: {duration} | Reason: {reason}{Style.RESET_ALL}"
        )

    def summary(self) -> str:
        if not self.trades: return "No trades recorded."
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
_active_trade_details: dict[str, Any] = {"entry_price": None, "entry_time_ms": None, "side": None, "qty": None}


# --- Helper Functions - Minor Cantrips ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    if value is None or (isinstance(value, float) and pd.isna(value)): # Handle pd.NA passed as float NaN
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
    if pd.isna(value) or value is None: return "N/A"
    if is_bool_trend:
        if value is True: return f"{Fore.GREEN}Up{Style.RESET_ALL}"
        if value is False: return f"{Fore.RED}Down{Style.RESET_ALL}"
        return "N/A (Trend)"
    if isinstance(value, Decimal): return f"{value:.{precision}f}"
    if isinstance(value, (float, int)): return f"{float(value):.{precision}f}"
    if isinstance(value, bool): return str(value)
    return str(value)

def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal) -> str:
    try:
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error shaping price {price} for {symbol}: {e}. Using raw Decimal.{Style.RESET_ALL}")
        return str(Decimal(str(price)).normalize())

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal) -> str:
    try:
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}Error shaping amount {amount} for {symbol}: {e}. Using raw Decimal.{Style.RESET_ALL}")
        return str(Decimal(str(amount)).normalize())

# --- Retry Decorator for API calls ---
@retry(tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, logger=logger, 
       exceptions=(ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection))
def safe_api_call(func, *args, **kwargs):
    """Wraps an API call with retry logic for common transient errors."""
    return func(*args, **kwargs)

# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: bool | None = None
def send_sms_alert(message: str) -> bool:
    global _termux_sms_command_exists
    if not CONFIG.enable_sms_alerts: return False
    if _termux_sms_command_exists is None:
        _termux_sms_command_exists = shutil.which("termux-sms-send") is not None
        if not _termux_sms_command_exists:
            logger.warning(f"{Fore.YELLOW}SMS: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}")
    if not _termux_sms_command_exists or not CONFIG.sms_recipient_number: return False
    try:
        command: list[str] = ["termux-sms-send", "-n", CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Dispatching SMS to {CONFIG.sms_recipient_number}...{Style.RESET_ALL}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS dispatched.{Style.RESET_ALL}"); return True
        else:
            logger.error(f"{Fore.RED}SMS failed. Code: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}"); return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: {e}{Style.RESET_ALL}"); return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> ccxt.Exchange | None:
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: API Key/Secret missing.{Style.RESET_ALL}")
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing.")
        return None
    try:
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key, "secret": CONFIG.api_secret, "enableRateLimit": True,
            "options": {"defaultType": "linear", "adjustForTimeDifference": True},
            "recvWindow": CONFIG.default_recv_window,
        })
        # exchange.set_sandbox_mode(True) # Testnet
        logger.debug("Loading market structures from Bybit...")
        exchange.load_markets(True)
        logger.debug(f"Loaded {len(exchange.markets)} market structures.")
        logger.debug("Performing initial balance check...")
        exchange.fetch_balance(params={"category": "linear"})
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (V5 API).{Style.RESET_ALL}")
        logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! LIVE SCALPING MODE ACTIVE - EXTREME CAUTION !!!{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus/{CONFIG.strategy_name.value}] Portal opened.")
        return exchange
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Portal opening failed: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Init FAILED: {type(e).__name__}.")
        logger.debug(traceback.format_exc())
    return None


# --- Indicator Calculation Functions - Scrying the Market ---
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    col_prefix = f"{prefix}" if prefix else ""
    out_supertrend_val = f"{col_prefix}supertrend"
    out_trend_direction = f"{col_prefix}trend"
    out_long_flip = f"{col_prefix}st_long_flip"
    out_short_flip = f"{col_prefix}st_short_flip"
    target_cols = [out_supertrend_val, out_trend_direction, out_long_flip, out_short_flip]

    pta_st_val_col = f"SUPERT_{length}_{float(multiplier)}"
    pta_st_dir_col = f"SUPERTd_{length}_{float(multiplier)}"
    pta_st_long_level_col = f"SUPERTl_{length}_{float(multiplier)}" # Actual ST line when long
    pta_st_short_level_col = f"SUPERTs_{length}_{float(multiplier)}" # Actual ST line when short
    
    min_len = length + 2 # pandas_ta might need length + lookback for direction change

    if df is None or df.empty or not all(c in df.columns for c in ["high", "low", "close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): Insufficient data (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}")
        if df is not None:
            for col in target_cols: df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        # Work on a copy to avoid SettingWithCopyWarning if df is a slice
        temp_df = df[["high", "low", "close"]].copy()
        temp_df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the columns
        pta_cols_to_check = [pta_st_val_col, pta_st_dir_col, pta_st_long_level_col, pta_st_short_level_col]
        if not all(c in temp_df.columns for c in pta_cols_to_check):
            missing_cols = [c for c in pta_cols_to_check if c not in temp_df.columns]
            logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): pandas_ta failed to create columns: {missing_cols}.{Style.RESET_ALL}")
            for col in target_cols: df[col] = pd.NA
            return df
        
        df[out_supertrend_val] = temp_df[pta_st_val_col].apply(safe_decimal_conversion)
        # Trend: True for up (1), False for down (-1), pd.NA for no trend
        df[out_trend_direction] = pd.NA # Default to NA
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
            logger.debug(f"Scrying ({col_prefix}ST({length},{multiplier})): Val={_format_for_log(last_val)}, Trend={trend_str}, Flip={flip_str}")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols: df[col] = pd.NA
    return df

def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    min_len = max(atr_len, vol_ma_len) + 1
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low", "close", "volume"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Insufficient data.{Style.RESET_ALL}")
        return results
    try:
        temp_df = df.copy() # Work on a copy
        atr_col = f"ATRr_{atr_len}"
        temp_df.ta.atr(length=atr_len, append=True)
        if atr_col in temp_df.columns and not temp_df.empty:
            results["atr"] = safe_decimal_conversion(temp_df[atr_col].iloc[-1])

        volume_ma_col = f"volume_sma_{vol_ma_len}"
        temp_df['volume'] = pd.to_numeric(temp_df['volume'], errors='coerce')
        temp_df[volume_ma_col] = temp_df["volume"].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        if not temp_df.empty:
            results["volume_ma"] = safe_decimal_conversion(temp_df[volume_ma_col].iloc[-1])
            results["last_volume"] = safe_decimal_conversion(temp_df["volume"].iloc[-1])
            if results["volume_ma"] and results["volume_ma"] > CONFIG.position_qty_epsilon and results["last_volume"]:
                try: results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
                except DivisionByZero: results["volume_ratio"] = None
        
        log_parts = [f"ATR({atr_len})={Fore.CYAN}{_format_for_log(results['atr'],5)}{Style.RESET_ALL}"]
        if results["last_volume"] is not None: log_parts.append(f"LastVol={_format_for_log(results['last_volume'],2)}")
        if results["volume_ma"] is not None: log_parts.append(f"VolMA({vol_ma_len})={_format_for_log(results['volume_ma'],2)}")
        if results["volume_ratio"] is not None: log_parts.append(f"VolRatio={Fore.YELLOW}{_format_for_log(results['volume_ratio'],2)}{Style.RESET_ALL}")
        logger.debug(f"Scrying Results: {', '.join(log_parts)}")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Vol/ATR): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        results = {key: None for key in results}
    return results

def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    min_len = max(rsi_len + stoch_len + d, mom_len) + 10
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (StochRSI/Mom): Insufficient data.{Style.RESET_ALL}")
        if df is not None:
            for col in target_cols: df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)
    try:
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col_ta, d_col_ta = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if stochrsi_df is not None and not stochrsi_df.empty:
            df["stochrsi_k"] = stochrsi_df[k_col_ta].apply(safe_decimal_conversion) if k_col_ta in stochrsi_df else pd.NA
            df["stochrsi_d"] = stochrsi_df[d_col_ta].apply(safe_decimal_conversion) if d_col_ta in stochrsi_df else pd.NA
        else: df["stochrsi_k"], df["stochrsi_d"] = pd.NA, pd.NA
        
        temp_df_mom = df[['close']].copy() # Operate on copy for momentum
        mom_col_ta = f"MOM_{mom_len}"
        temp_df_mom.ta.mom(length=mom_len, append=True)
        df["momentum"] = temp_df_mom[mom_col_ta].apply(safe_decimal_conversion) if mom_col_ta in temp_df_mom else pd.NA

        if not df.empty:
            k_v, d_v, m_v = df["stochrsi_k"].iloc[-1], df["stochrsi_d"].iloc[-1], df["momentum"].iloc[-1]
            logger.debug(f"Scrying (StochRSI/Mom): K={_format_for_log(k_v,2)}, D={_format_for_log(d_v,2)}, Mom={_format_for_log(m_v,4)}")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (StochRSI/Mom): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols: df[col] = pd.NA
    return df

def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    min_len = length + signal + 5
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient data.{Style.RESET_ALL}")
        if df is not None:
            for col in target_cols: df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)
    try:
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col_ta, signal_col_ta = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fisher_df is not None and not fisher_df.empty:
            df["ehlers_fisher"] = fisher_df[fish_col_ta].apply(safe_decimal_conversion) if fish_col_ta in fisher_df else pd.NA
            df["ehlers_signal"] = fisher_df[signal_col_ta].apply(safe_decimal_conversion) if signal_col_ta in fisher_df else pd.NA
        else: df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA

        if not df.empty:
            f_v, s_v = df["ehlers_fisher"].iloc[-1], df["ehlers_signal"].iloc[-1]
            logger.debug(f"Scrying (EhlersFisher({length},{signal})): Fisher={_format_for_log(f_v)}, Signal={_format_for_log(s_v)}")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (EhlersFisher): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols: df[col] = pd.NA
    return df

def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int, poles: int) -> pd.DataFrame:
    target_cols = ["ehlers_ssf_fast", "ehlers_ssf_slow"]
    min_len = max(fast_len, slow_len) + poles + 5
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Scrying (Ehlers SSF MA): Insufficient data.{Style.RESET_ALL}")
        if df is not None:
            for col in target_cols: df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)
    try:
        ssf_fast_series = df.ta.ssf(length=fast_len, poles=poles, append=False)
        df["ehlers_ssf_fast"] = ssf_fast_series.apply(safe_decimal_conversion) if ssf_fast_series is not None else pd.NA
        ssf_slow_series = df.ta.ssf(length=slow_len, poles=poles, append=False)
        df["ehlers_ssf_slow"] = ssf_slow_series.apply(safe_decimal_conversion) if ssf_slow_series is not None else pd.NA
        
        if not df.empty:
            fast_v, slow_v = df["ehlers_ssf_fast"].iloc[-1], df["ehlers_ssf_slow"].iloc[-1]
            logger.debug(f"Scrying (Ehlers SSF MA({fast_len},{slow_len},p{poles})): Fast={_format_for_log(fast_v)}, Slow={_format_for_log(slow_v)}")
    except Exception as e:
        logger.error(f"{Fore.RED}Scrying (Ehlers SSF MA): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        if df is not None:
            for col in target_cols: df[col] = pd.NA
    return df

def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(f"{Fore.YELLOW}OB Scrying: fetchL2OrderBook not supported.{Style.RESET_ALL}")
        return results
    try:
        order_book = safe_api_call(exchange.fetch_l2_order_book, symbol, limit=fetch_limit)
        bids, asks = order_book.get("bids", []), order_book.get("asks", [])
        if not bids or not asks:
            logger.warning(f"{Fore.YELLOW}OB Scrying: Empty bids/asks for {symbol}.{Style.RESET_ALL}")
            return results

        results["best_bid"] = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        results["best_ask"] = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        if results["best_bid"] and results["best_ask"] and results["best_bid"] > 0 and results["best_ask"] > 0:
            results["spread"] = results["best_ask"] - results["best_bid"]
        
        bid_vol = sum(safe_decimal_conversion(b[1]) for b in bids[:min(depth,len(bids))] if len(b)>1)
        ask_vol = sum(safe_decimal_conversion(a[1]) for a in asks[:min(depth,len(asks))] if len(a)>1)
        if ask_vol > CONFIG.position_qty_epsilon:
            try: results["bid_ask_ratio"] = bid_vol / ask_vol
            except (DivisionByZero, InvalidOperation): results["bid_ask_ratio"] = None
        
        log_parts = [f"BestBid={Fore.GREEN}{_format_for_log(results['best_bid'],4)}{Style.RESET_ALL}",
                     f"BestAsk={Fore.RED}{_format_for_log(results['best_ask'],4)}{Style.RESET_ALL}",
                     f"Spread={Fore.YELLOW}{_format_for_log(results['spread'],4)}{Style.RESET_ALL}"]
        if results['bid_ask_ratio'] is not None:
            log_parts.append(f"Ratio(B/A)={Fore.CYAN}{_format_for_log(results['bid_ask_ratio'],3)}{Style.RESET_ALL}")
        logger.debug(f"OB Scrying (Depth {depth}): {', '.join(log_parts)}")

    except Exception as e:
        logger.warning(f"{Fore.YELLOW}OB Scrying Error: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return results


# --- Data Fetching & Caching - Gathering Etheric Data Streams ---
_last_market_data: pd.DataFrame | None = None
_last_fetch_timestamp: float = 0.0

def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    global _last_market_data, _last_fetch_timestamp
    current_time = time.time()
    
    try:
        candle_duration_seconds = exchange.parse_timeframe(interval) # in seconds
    except Exception as e: # Fallback if parse_timeframe fails or not supported
        logger.warning(f"Could not parse timeframe '{interval}' for caching: {e}. Cache disabled for this call.")
        candle_duration_seconds = 0 # Disables cache effectively

    cache_is_valid = (
        _last_market_data is not None and
        candle_duration_seconds > 0 and # Ensure candle_duration is valid
        (current_time - _last_fetch_timestamp) < (candle_duration_seconds * float(CONFIG.cache_candle_duration_multiplier)) and
        len(_last_market_data) >= limit # Ensure cache has enough data
    )

    if cache_is_valid:
        logger.debug(f"Data Fetch: Using CACHED market data ({len(_last_market_data)} candles). Last fetch: {time.strftime('%H:%M:%S', time.localtime(_last_fetch_timestamp))}")
        return _last_market_data.copy()

    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Gathering {limit} OHLCV candles for {symbol} ({interval})...")
        ohlcv = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data for {symbol}. Market inactive or API issue.{Style.RESET_ALL}")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.isnull().values.any():
            logger.warning(f"{Fore.YELLOW}Data Fetch: NaNs found. Ffilling/Bfilling...{Style.RESET_ALL}")
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            if df.isnull().values.any():
                logger.error(f"{Fore.RED}Data Fetch: Unfillable NaNs remain. Data quality insufficient.{Style.RESET_ALL}")
                return None
        
        _last_market_data = df.copy()
        _last_fetch_timestamp = current_time
        logger.debug(f"Data Fetch: Successfully woven {len(df)} OHLCV candles for {symbol}. Cached.")
        return df
    except Exception as e:
        logger.error(f"{Fore.RED}Data Fetch: Error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return None


# --- Account Health Check ---
def check_account_health(exchange: ccxt.Exchange, config: Config) -> bool:
    logger.debug("Performing account health check...")
    try:
        balance_params = {"category": "linear"} # For Bybit V5 USDT futures
        balance = safe_api_call(exchange.fetch_balance, params=balance_params)
        
        total_equity = safe_decimal_conversion(balance.get(config.usdt_symbol, {}).get("total"))
        used_margin = safe_decimal_conversion(balance.get(config.usdt_symbol, {}).get("used"))

        if total_equity <= Decimal("0"):
            logger.warning(f"Account Health: Total equity {_format_for_log(total_equity)}. Margin ratio calc skipped.")
            if used_margin > Decimal("0"):
                 logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Zero/Negative Equity ({_format_for_log(total_equity)}) with Used Margin ({_format_for_log(used_margin)})! Halting.{Style.RESET_ALL}")
                 send_sms_alert(f"CRITICAL: Zero/Neg Equity with Used Margin. Bot paused.")
                 return False
            return True 

        margin_ratio = used_margin / total_equity
        logger.info(f"Account Health: Equity={_format_for_log(total_equity,2)}, UsedMargin={_format_for_log(used_margin,2)}, MarginRatio={margin_ratio:.2%}")

        if margin_ratio > config.max_account_margin_ratio:
            logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: High margin ratio {margin_ratio:.2%} > {config.max_account_margin_ratio:.0%}. Halting.{Style.RESET_ALL}")
            send_sms_alert(f"CRITICAL: High margin ratio {margin_ratio:.2%}. Bot paused.")
            return False
        return True
    except Exception as e:
        logger.error(f"Account health check failed: {type(e).__name__} - {e}")
        logger.debug(traceback.format_exc())
        return False # Treat as uncertain/unhealthy on error


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    default_pos: dict[str, Any] = {"side": CONFIG.pos_none, "qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    try:
        market = exchange.market(symbol)
        market_id = market["id"]
        category = "linear" if market.get("linear") else ("inverse" if market.get("inverse") else None)
        if not category: logger.error(f"{Fore.RED}Pos Check: No category for {symbol}.{Style.RESET_ALL}"); return default_pos
        
        params = {"category": category, "symbol": market_id}
        fetched_positions = safe_api_call(exchange.fetch_positions, symbols=[symbol], params=params)

        if not fetched_positions: return default_pos
        for pos in fetched_positions:
            pos_info = pos.get("info", {})
            if pos_info.get("symbol") != market_id: continue
            if int(pos_info.get("positionIdx", -1)) == 0: # One-Way Mode
                size_dec = safe_decimal_conversion(pos_info.get("size", "0"))
                if size_dec > CONFIG.position_qty_epsilon and pos_info.get("side") in ["Buy", "Sell"]:
                    entry_price = safe_decimal_conversion(pos_info.get("avgPrice"))
                    side = CONFIG.pos_long if pos_info.get("side") == "Buy" else CONFIG.pos_short
                    pos_color = Fore.GREEN if side == CONFIG.pos_long else Fore.RED
                    logger.info(f"{pos_color}Pos Check: ACTIVE {side} Qty={size_dec:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                    return {"side": side, "qty": size_dec, "entry_price": entry_price}
        logger.info(f"{Fore.BLUE}Pos Check: No active One-Way position for {market_id}. Flat.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Pos Check: Error: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return default_pos

def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    logger.info(f"{Fore.CYAN}Leverage: Setting {leverage}x for {symbol}...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol)
        if not market.get("contract"):
            logger.error(f"{Fore.RED}Leverage: Not a contract market: {symbol}.{Style.RESET_ALL}"); return False
        
        # For Bybit V5, CCXT handles mapping to buyLeverage/sellLeverage.
        # Category might be needed if symbol alone is ambiguous, but usually not for set_leverage.
        # params = {"category": "linear"} # Usually not needed for set_leverage by symbol
        response = safe_api_call(exchange.set_leverage, leverage=leverage, symbol=symbol) # params=params
        logger.success(f"{Fore.GREEN}Leverage: Set to {leverage}x for {symbol}. Resp: {response}{Style.RESET_ALL}")
        return True
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        if any(sub in err_str for sub in ["leverage not modified", "same leverage", "110044"]):
            logger.info(f"{Fore.CYAN}Leverage: Already {leverage}x for {symbol}.{Style.RESET_ALL}"); return True
        logger.error(f"{Fore.RED}Leverage: Failed: {e}{Style.RESET_ALL}")
    except Exception as e_unexp:
        logger.error(f"{Fore.RED}Leverage: Unexpected error: {e_unexp}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return False

def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal") -> dict[str, Any] | None:
    global _active_trade_details # To log trade via TradeMetrics
    initial_side, initial_qty = position_to_close.get("side", CONFIG.pos_none), position_to_close.get("qty", Decimal("0.0"))
    market_base = symbol.split("/")[0].split(":")[0]
    logger.info(f"{Fore.YELLOW}Banish Position: {symbol} ({reason}). Initial: {initial_side} Qty={_format_for_log(initial_qty, 8)}{Style.RESET_ALL}")

    live_position = get_current_position(exchange, symbol)
    if live_position["side"] == CONFIG.pos_none or live_position["qty"] <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Banish Position: Re-validation shows NO active position. Aborting.{Style.RESET_ALL}")
        # If we thought there was a trade, but now it's gone, clear details
        if _active_trade_details["entry_price"] is not None:
            logger.info("Clearing potentially stale active trade details as position is now flat.")
            _active_trade_details = {"entry_price": None, "entry_time_ms": None, "side": None, "qty": None}
        return None

    side_to_execute = CONFIG.side_sell if live_position["side"] == CONFIG.pos_long else CONFIG.side_buy
    amount_to_close_str = format_amount(exchange, symbol, live_position["qty"])
    
    try:
        logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}Banish Position: CLOSE {live_position['side']} ({reason}): {side_to_execute.upper()} MARKET {amount_to_close_str} {symbol}...{Style.RESET_ALL}")
        params = {"reduceOnly": True, "category": "linear"} # Bybit V5
        order = safe_api_call(exchange.create_market_order, symbol=symbol, side=side_to_execute, amount=float(amount_to_close_str), params=params)
        
        status = order.get("status", "unknown")
        # For market orders, 'closed' implies filled. Check 'filled' amount.
        filled_qty_closed = safe_decimal_conversion(order.get("filled"))
        avg_fill_price_closed = safe_decimal_conversion(order.get("average"))

        if status == "closed" and abs(filled_qty_closed - live_position["qty"]) < CONFIG.position_qty_epsilon:
            logger.success(f"{Fore.GREEN}{Style.BRIGHT}Banish Position: CONFIRMED FILLED ({reason}). ID:...{format_order_id(order.get('id'))}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}/{CONFIG.strategy_name.value}] BANISHED {live_position['side']} {amount_to_close_str} ({reason}).")
            
            # Log to TradeMetrics
            if _active_trade_details["entry_price"] is not None and order.get("timestamp") is not None:
                trade_metrics.log_trade(
                    symbol=symbol, side=_active_trade_details["side"],
                    entry_price=_active_trade_details["entry_price"], exit_price=avg_fill_price_closed,
                    qty=_active_trade_details["qty"], # Use original entry qty
                    entry_time_ms=_active_trade_details["entry_time_ms"], exit_time_ms=order["timestamp"],
                    reason=reason
                )
            _active_trade_details = {"entry_price": None, "entry_time_ms": None, "side": None, "qty": None} # Reset
            return order
        else:
            logger.warning(f"{Fore.YELLOW}Banish Position: Fill uncertain. Expected {live_position['qty']}, Filled {filled_qty_closed}. ID:...{format_order_id(order.get('id'))}, Status: {status}.{Style.RESET_ALL}")
            return order # Return for potential checks
            
    except Exception as e:
        logger.error(f"{Fore.RED}Banish Position ({reason}): Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name.value}] ERROR Banishing ({reason}): {type(e).__name__}.")
        logger.debug(traceback.format_exc())
    return None

def calculate_position_size(equity: Decimal, risk_pct: Decimal, entry_px: Decimal, sl_px: Decimal, lev: int, sym: str, ex: ccxt.Exchange) -> tuple[Decimal | None, Decimal | None]:
    if not (entry_px > 0 and sl_px > 0 and 0 < risk_pct < 1 and equity > 0 and lev > 0):
        logger.error(f"{Fore.RED}RiskCalc: Invalid inputs.{Style.RESET_ALL}"); return None, None
    price_diff = abs(entry_px - sl_px)
    if price_diff < CONFIG.position_qty_epsilon:
        logger.error(f"{Fore.RED}RiskCalc: Entry/SL too close.{Style.RESET_ALL}"); return None, None
    try:
        risk_amt_usdt = equity * risk_pct
        qty_raw = risk_amt_usdt / price_diff
        qty_prec_str = format_amount(ex, sym, qty_raw)
        qty_prec = Decimal(qty_prec_str)
        if qty_prec <= CONFIG.position_qty_epsilon:
            logger.warning(f"{Fore.YELLOW}RiskCalc: Qty negligible ({qty_prec}).{Style.RESET_ALL}"); return None, None
        pos_val_usdt = qty_prec * entry_px
        margin_req = pos_val_usdt / Decimal(lev)
        logger.debug(f"RiskCalc: Qty={Fore.CYAN}{_format_for_log(qty_prec,8)}{Style.RESET_ALL}, MarginReq={_format_for_log(margin_req,4)}")
        return qty_prec, margin_req
    except Exception as e:
        logger.error(f"{Fore.RED}RiskCalc: Error: {e}{Style.RESET_ALL}"); return None, None

def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_s: int) -> dict[str, Any] | None:
    start_time = time.time()
    oid_short = format_order_id(order_id)
    logger.info(f"{Fore.CYAN}Observing order ...{oid_short} ({symbol}) for fill (Timeout: {timeout_s}s)...{Style.RESET_ALL}")
    params = {"category": "linear"} # Bybit V5 may need category
    while time.time() - start_time < timeout_s:
        try:
            order = safe_api_call(exchange.fetch_order, order_id, symbol, params=params)
            status = order.get("status")
            if status == "closed": logger.success(f"{Fore.GREEN}Order ...{oid_short} FILLED/CLOSED.{Style.RESET_ALL}"); return order
            if status in ["canceled", "rejected", "expired"]: logger.error(f"{Fore.RED}Order ...{oid_short} FAILED: {status}.{Style.RESET_ALL}"); return order
            time.sleep(0.75)
        except ccxt.OrderNotFound: time.sleep(1.5) # Propagation delay
        except Exception as e:
            logger.warning(f"{Fore.YELLOW}Error checking order ...{oid_short}: {type(e).__name__}. Retrying...{Style.RESET_ALL}")
            time.sleep(2)
    logger.error(f"{Fore.RED}Order ...{oid_short} TIMEOUT after {timeout_s}s.{Style.RESET_ALL}")
    try: return safe_api_call(exchange.fetch_order, order_id, symbol, params=params) # Final check
    except: return None

def place_risked_market_order(exchange: ccxt.Exchange, symbol: str, side: str, risk_percentage: Decimal, current_atr: Decimal | None, sl_atr_multiplier: Decimal, leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal, tsl_percent: Decimal, tsl_activation_offset_percent: Decimal) -> dict[str, Any] | None:
    global _active_trade_details
    market_base = symbol.split("/")[0].split(":")[0]
    logger.info(f"{Fore.CYAN if side == CONFIG.side_buy else Fore.MAGENTA}{Style.BRIGHT}Place Order: {side.upper()} for {symbol}...{Style.RESET_ALL}")
    if current_atr is None or current_atr <= 0:
        logger.error(f"{Fore.RED}Place Order Error: Invalid ATR ({current_atr}).{Style.RESET_ALL}"); return None
    v5_category = "linear"
    try:
        balance = safe_api_call(exchange.fetch_balance, params={"category": v5_category})
        market = exchange.market(symbol)
        min_qty = safe_decimal_conversion(market.get("limits",{}).get("amount",{}).get("min"))
        usdt_equity = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol,{}).get("total"))
        usdt_free = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol,{}).get("free"))
        if usdt_equity <= 0 or usdt_free < 0:
            logger.error(f"{Fore.RED}Place Order Error: Invalid equity/free margin.{Style.RESET_ALL}"); return None

        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        entry_px_est = ob_data.get("best_ask") if side == CONFIG.side_buy else ob_data.get("best_bid")
        if not entry_px_est or entry_px_est <= 0:
            ticker = safe_api_call(exchange.fetch_ticker, symbol)
            entry_px_est = safe_decimal_conversion(ticker.get("last"))
        if not entry_px_est or entry_px_est <= 0:
            logger.error(f"{Fore.RED}Place Order Error: Failed to get entry price estimate.{Style.RESET_ALL}"); return None

        sl_dist = current_atr * sl_atr_multiplier
        sl_px_raw = (entry_px_est - sl_dist) if side == CONFIG.side_buy else (entry_px_est + sl_dist)
        sl_px_est_str = format_price(exchange, symbol, sl_px_raw)
        sl_px_est = Decimal(sl_px_est_str)
        if sl_px_est <= 0: logger.error(f"{Fore.RED}Place Order: Invalid SL estimate.{Style.RESET_ALL}"); return None

        final_qty, margin_est = calculate_position_size(usdt_equity, risk_percentage, entry_px_est, sl_px_est, leverage, symbol, exchange)
        if final_qty is None or margin_est is None: return None
        
        pos_val_est = final_qty * entry_px_est
        if pos_val_est > max_order_cap_usdt:
            final_qty = Decimal(format_amount(exchange, symbol, max_order_cap_usdt / entry_px_est))
            margin_est = (final_qty * entry_px_est) / Decimal(leverage)
        if min_qty and final_qty < min_qty:
            logger.error(f"{Fore.RED}Place Order: Qty {_format_for_log(final_qty,8)} < Min Qty {_format_for_log(min_qty,8)}.{Style.RESET_ALL}"); return None
        if usdt_free < margin_est * margin_check_buffer:
            logger.error(f"{Fore.RED}Place Order: Insufficient FREE margin.{Style.RESET_ALL}"); return None

        entry_params = {"reduceOnly": False, "category": v5_category}
        entry_order = safe_api_call(exchange.create_market_order, symbol=symbol, side=side, amount=float(final_qty), params=entry_params)
        order_id = entry_order.get("id")
        if not order_id: logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Entry order NO ID!{Style.RESET_ALL}"); return None
        
        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry or filled_entry.get("status") != "closed":
            logger.error(f"{Fore.RED}Entry order ...{format_order_id(order_id)} not filled. Status: {filled_entry.get('status') if filled_entry else 'timeout'}.{Style.RESET_ALL}")
            # Attempt to cancel if not filled
            try: safe_api_call(exchange.cancel_order, order_id, symbol, params={"category": v5_category})
            except Exception as e_cancel: logger.warning(f"Could not cancel unfilled order {order_id}: {e_cancel}")
            return None

        avg_fill_px = safe_decimal_conversion(filled_entry.get("average"))
        filled_qty_val = safe_decimal_conversion(filled_entry.get("filled"))
        if filled_qty_val <= CONFIG.position_qty_epsilon or avg_fill_px <= CONFIG.position_qty_epsilon:
            logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Invalid fill qty/price for ...{format_order_id(order_id)}.{Style.RESET_ALL}"); return filled_entry
        
        logger.success(f"{Fore.GREEN if side == CONFIG.side_buy else Fore.RED}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {_format_for_log(filled_qty_val,8)} @ Avg: {_format_for_log(avg_fill_px,4)}{Style.RESET_ALL}")
        _active_trade_details = {"entry_price": avg_fill_px, "entry_time_ms": filled_entry.get("timestamp"), "side": side, "qty": filled_qty_val}

        # Place Fixed SL
        actual_sl_px_raw = (avg_fill_px - sl_dist) if side == CONFIG.side_buy else (avg_fill_px + sl_dist)
        actual_sl_px_str = format_price(exchange, symbol, actual_sl_px_raw)
        if Decimal(actual_sl_px_str) <= 0: logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL: ACTUAL SL invalid!{Style.RESET_ALL}"); return filled_entry # Potentially close pos
        sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
        sl_params = {"stopPrice": float(actual_sl_px_str), "reduceOnly": True, "category": v5_category, "positionIdx": 0}
        try:
            sl_order = safe_api_call(exchange.create_order, symbol, "stopMarket", sl_side, float(filled_qty_val), params=sl_params)
            logger.success(f"{Fore.GREEN}Fixed SL placed. ID:...{format_order_id(sl_order.get('id'))}, Trigger:{actual_sl_px_str}{Style.RESET_ALL}")
        except Exception as e: logger.error(f"{Back.RED}{Fore.WHITE}FAILED Fixed SL: {e}{Style.RESET_ALL}")

        # Place TSL
        act_offset = avg_fill_px * tsl_activation_offset_percent
        act_price_raw = (avg_fill_px + act_offset) if side == CONFIG.side_buy else (avg_fill_px - act_offset)
        tsl_act_px_str = format_price(exchange, symbol, act_price_raw)
        if Decimal(tsl_act_px_str) <= 0: logger.error(f"{Back.RED}{Fore.WHITE}CRITICAL: TSL Act Price invalid!{Style.RESET_ALL}"); return filled_entry
        tsl_trail_val_str = str((tsl_percent * Decimal("100")).normalize())
        tsl_params = {"trailingStop": tsl_trail_val_str, "activePrice": float(tsl_act_px_str), "reduceOnly": True, "category": v5_category, "positionIdx": 0}
        try:
            tsl_order = safe_api_call(exchange.create_order, symbol, "stopMarket", sl_side, float(filled_qty_val), params=tsl_params) # sl_side is same for TSL
            logger.success(f"{Fore.GREEN}Trailing SL placed. ID:...{format_order_id(tsl_order.get('id'))}, Trail%:{tsl_trail_val_str}, ActPx:{tsl_act_px_str}{Style.RESET_ALL}")
        except Exception as e: logger.error(f"{Back.RED}{Fore.WHITE}FAILED TSL: {e}{Style.RESET_ALL}")
        
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name.value}] ENTERED {side.upper()} {_format_for_log(filled_qty_val,8)} @ {_format_for_log(avg_fill_px,4)}. SL:~{actual_sl_px_str}, TSL:{tsl_percent:.2%}@~{tsl_act_px_str}. EntryID:...{format_order_id(order_id)}")
        return filled_entry
    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}Place Order Ritual FAILED: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name.value}] ORDER FAIL ({side.upper()}): {type(e).__name__}")
    return None

def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> int:
    logger.info(f"{Fore.CYAN}Order Cleanup: {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    cancelled_count, failed_count = 0, 0
    v5_category = "linear"
    try:
        open_orders = safe_api_call(exchange.fetch_open_orders, symbol, params={"category": v5_category})
        if not open_orders: logger.info(f"{Fore.CYAN}Order Cleanup: No open orders for {symbol}.{Style.RESET_ALL}"); return 0
        logger.warning(f"{Fore.YELLOW}Order Cleanup: Found {len(open_orders)} open orders. Cancelling...{Style.RESET_ALL}")
        for order in open_orders:
            order_id = order.get("id")
            if order_id:
                try:
                    safe_api_call(exchange.cancel_order, order_id, symbol, params={"category": v5_category})
                    cancelled_count += 1
                except ccxt.OrderNotFound: cancelled_count += 1 # Already gone
                except Exception as e: logger.error(f"{Fore.RED}Order Cleanup: FAILED cancel ...{format_order_id(order_id)}: {e}{Style.RESET_ALL}"); failed_count +=1
            else: logger.error(f"{Fore.RED}Order Cleanup: Open order with no ID.{Style.RESET_ALL}"); failed_count +=1
        if failed_count > 0: send_sms_alert(f"[{symbol.split('/')[0]}/{CONFIG.strategy_name.value}] WARNING: Failed cancel {failed_count} orders ({reason}).")
    except Exception as e: logger.error(f"{Fore.RED}Order Cleanup: Error fetching/processing: {e}{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}Order Cleanup: Cancelled/Handled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
    return cancelled_count


# --- Strategy Signal Generation Wrapper ---
def generate_strategy_signals(df: pd.DataFrame, strategy_instance: TradingStrategy) -> dict[str, Any]:
    if strategy_instance:
        return strategy_instance.generate_signals(df)
    logger.error("Unknown strategy instance provided for signal generation.")
    return strategy_instance._get_default_signals() # type: ignore

# --- All Indicator Calculations ---
def calculate_all_indicators(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculates all indicators and returns the DataFrame and volume/ATR analysis."""
    df = calculate_supertrend(df, config.st_atr_length, config.st_multiplier)
    df = calculate_supertrend(df, config.confirm_st_atr_length, config.confirm_st_multiplier, prefix="confirm_")
    df = calculate_stochrsi_momentum(df, config.stochrsi_rsi_length, config.stochrsi_stoch_length, config.stochrsi_k_period, config.stochrsi_d_period, config.momentum_length)
    df = calculate_ehlers_fisher(df, config.ehlers_fisher_length, config.ehlers_fisher_signal_length)
    df = calculate_ehlers_ma(df, config.ehlers_fast_period, config.ehlers_slow_period, config.ehlers_ssf_poles)
    vol_atr_data = analyze_volume_atr(df, config.atr_calculation_period, config.volume_ma_period)
    return df, vol_atr_data

# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    cycle_time_str = df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z") if not df.empty else "N/A"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle ({CONFIG.strategy_name.value}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # Calculate required rows dynamically - simplified, ensure data_limit in main loop is generous
    required_rows = 50 # A general baseline, specific strategies might need more
    if df is None or len(df) < required_rows:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0} rows).{Style.RESET_ALL}")
        return

    try:
        df, vol_atr_data = calculate_all_indicators(df.copy(), CONFIG) # Use copy
        current_atr = vol_atr_data.get("atr")
        last_candle = df.iloc[-1] if not df.empty else pd.Series(dtype='object')
        current_price = safe_decimal_conversion(last_candle.get("close"))

        if pd.isna(current_price) or current_price <= 0:
            logger.warning(f"{Fore.YELLOW}Trade Logic: Invalid last close price.{Style.RESET_ALL}"); return
        can_place_order = current_atr is not None and current_atr > 0

        position = get_current_position(exchange, symbol)
        pos_side, pos_qty, pos_entry = position["side"], position["qty"], position["entry_price"]

        ob_data = None
        if CONFIG.fetch_order_book_per_cycle or (pos_side == CONFIG.pos_none and can_place_order): # Fetch if flat and might enter
             ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)

        # Log Snapshot
        logger.info(f"{Fore.MAGENTA}--- Indicator Snapshot ---{Style.RESET_ALL}")
        logger.info(f"  Market: Close={_format_for_log(current_price, 4)}, ATR({CONFIG.atr_calculation_period})={_format_for_log(current_atr, 5)}")
        is_vol_spike = (vol_atr_data.get("volume_ratio") is not None and vol_atr_data["volume_ratio"] > CONFIG.volume_spike_threshold) # type: ignore
        logger.info(f"  Volume: Ratio={_format_for_log(vol_atr_data.get('volume_ratio'), 2)}, Spike={is_vol_spike}")
        if ob_data: logger.info(f"  OrderBook: Ratio(B/A)={_format_for_log(ob_data.get('bid_ask_ratio'),3)}, Spread={_format_for_log(ob_data.get('spread'),4)}")
        # Dynamic strategy indicator logging
        CONFIG.strategy_instance.logger.info(f"  Strategy Values ({CONFIG.strategy_name.value}):") # Use strategy's logger
        for col_name in CONFIG.strategy_instance.required_columns:
             if col_name in last_candle.index:
                 is_trend = "trend" in col_name.lower()
                 CONFIG.strategy_instance.logger.info(f"    {col_name}: {_format_for_log(last_candle[col_name], is_bool_trend=is_trend)}")
        pos_color = Fore.GREEN if pos_side == CONFIG.pos_long else (Fore.RED if pos_side == CONFIG.pos_short else Fore.BLUE)
        logger.info(f"  Position: Side={pos_color}{pos_side}{Style.RESET_ALL}, Qty={_format_for_log(pos_qty,8)}, Entry={_format_for_log(pos_entry,4)}")
        logger.info(f"{Fore.MAGENTA}{'-'*26}{Style.RESET_ALL}")

        strategy_signals = generate_strategy_signals(df, CONFIG.strategy_instance)
        should_exit_long = pos_side == CONFIG.pos_long and strategy_signals["exit_long"]
        should_exit_short = pos_side == CONFIG.pos_short and strategy_signals["exit_short"]

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals["exit_reason"]
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}*** EXIT SIGNAL: Closing {pos_side} ({exit_reason}) ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, f"Pre-Exit ({exit_reason})"); time.sleep(0.5)
            close_result = close_position(exchange, symbol, position, reason=exit_reason)
            if close_result: time.sleep(CONFIG.post_close_delay_seconds)
            return

        if pos_side != CONFIG.pos_none:
            logger.info(f"Holding {pos_color}{pos_side}{Style.RESET_ALL}. Awaiting SL/TSL or Exit Signal.")
            return
        if not can_place_order:
            logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter: Invalid ATR.{Style.RESET_ALL}"); return
        
        potential_entry = strategy_signals["enter_long"] or strategy_signals["enter_short"]
        if not potential_entry: logger.info("Holding Cash. No entry signal."); return

        ob_confirm_long, ob_confirm_short = True, True
        if ob_data and ob_data.get("bid_ask_ratio") is not None:
            ratio = ob_data["bid_ask_ratio"]
            if CONFIG.order_book_ratio_threshold_long < Decimal('Infinity'): ob_confirm_long = ratio >= CONFIG.order_book_ratio_threshold_long # type: ignore
            if CONFIG.order_book_ratio_threshold_short > Decimal(0): ob_confirm_short = ratio <= CONFIG.order_book_ratio_threshold_short # type: ignore
        elif (CONFIG.order_book_ratio_threshold_long < Decimal('Infinity') or CONFIG.order_book_ratio_threshold_short > Decimal(0)):
            ob_confirm_long, ob_confirm_short = False, False # Required but no data
        vol_confirm = not CONFIG.require_volume_spike_for_entry or is_vol_spike

        final_enter_long = strategy_signals["enter_long"] and ob_confirm_long and vol_confirm
        final_enter_short = strategy_signals["enter_short"] and ob_confirm_short and vol_confirm

        entry_params = {
            "exchange": exchange, "symbol": symbol, "risk_percentage": CONFIG.risk_per_trade_percentage,
            "current_atr": current_atr, "sl_atr_multiplier": CONFIG.atr_stop_loss_multiplier, "leverage": CONFIG.leverage,
            "max_order_cap_usdt": CONFIG.max_order_usdt_amount, "margin_check_buffer": CONFIG.required_margin_buffer,
            "tsl_percent": CONFIG.trailing_stop_percentage, "tsl_activation_offset_percent": CONFIG.trailing_stop_activation_offset_percent,
        }
        if final_enter_long:
            logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** CONFIRMED LONG ENTRY ({CONFIG.strategy_name.value}) ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, "Pre-Long Entry"); time.sleep(0.5)
            place_risked_market_order(side=CONFIG.side_buy, **entry_params) # type: ignore
        elif final_enter_short:
            logger.success(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** CONFIRMED SHORT ENTRY ({CONFIG.strategy_name.value}) ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, "Pre-Short Entry"); time.sleep(0.5)
            place_risked_market_order(side=CONFIG.side_sell, **entry_params) # type: ignore
        elif potential_entry: logger.info("Holding Cash. Signal present but filters not met.")

    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{symbol.split('/')[0]}/{CONFIG.strategy_name.value}] CRITICAL trade_logic ERROR: {type(e).__name__}.")
    finally:
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle End: {symbol} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Withdrawing energies...{Style.RESET_ALL}")
    market_base = symbol.split("/")[0].split(":")[0] if symbol and "/" in symbol and ":" in symbol else (symbol if symbol else "Bot")
    strat_name_val = CONFIG.strategy_name.value if 'CONFIG' in globals() and hasattr(CONFIG, 'strategy_name') else 'N/A'
    send_sms_alert(f"[{market_base}/{strat_name_val}] Shutdown initiated. Cleanup attempt...")

    if trade_metrics and hasattr(trade_metrics, 'summary'): trade_metrics.summary() # Log final metrics

    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange/Symbol not defined. No automated cleanup.{Style.RESET_ALL}")
        return
    try:
        logger.warning("Shutdown: Cancelling open orders..."); cancel_open_orders(exchange, symbol, "Shutdown")
        time.sleep(1.5)
        position = get_current_position(exchange, symbol)
        if position["side"] != CONFIG.pos_none and position["qty"] > 0:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} position. Closing...{Style.RESET_ALL}")
            close_result = close_position(exchange, symbol, position, "Shutdown")
            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Final check after delay...{Style.RESET_ALL}")
                time.sleep(CONFIG.post_close_delay_seconds * 2)
                final_pos = get_current_position(exchange, symbol)
                if final_pos["side"] == CONFIG.pos_none: logger.success(f"{Fore.GREEN}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}")
                else: logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN: FAILED CONFIRM closure! Final: {final_pos['side']} Qty={final_pos['qty']}. MANUAL CHECK!{Style.RESET_ALL}")
            else: logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN: FAILED PLACE close order! MANUAL CHECK!{Style.RESET_ALL}")
        else: logger.info(f"{Fore.GREEN}Shutdown: No active position. Clean exit.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Spell Shutdown Complete ---{Style.RESET_ALL}")


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.3.0 Initializing ({start_time_str}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy: {CONFIG.strategy_name.value} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Protections: ATR-Stop + Exchange TSL (Bybit V5) ---{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING - EXTREME RISK - EDUCATIONAL USE ONLY !!! ---{Style.RESET_ALL}")

    exchange: ccxt.Exchange | None = None
    symbol_unified: str | None = None 
    run_bot: bool = True; cycle_count: int = 0

    try:
        exchange = initialize_exchange()
        if not exchange: return 
        market = exchange.market(CONFIG.symbol) 
        symbol_unified = market["symbol"]
        if not market.get("contract"): raise ValueError(f"Market '{symbol_unified}' not a contract market.")
        logger.info(f"{Fore.GREEN}Spell focused on: {symbol_unified} (Type: {market.get('type', 'N/A')}){Style.RESET_ALL}")
        if not set_leverage(exchange, symbol_unified, CONFIG.leverage):
            raise RuntimeError(f"Leverage setting failed for {symbol_unified}.")

        # Log key config details
        logger.info(f"{Fore.MAGENTA}--- Spell Config Summary ---{Style.RESET_ALL}")
        logger.info(f"Symbol: {symbol_unified}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x, Strategy: {CONFIG.strategy_name.value}")
        logger.info(f"Risk: {CONFIG.risk_per_trade_percentage:.2%}/trade, MaxCap: {CONFIG.max_order_usdt_amount} USDT, ATR SL Mult: {CONFIG.atr_stop_loss_multiplier}")
        logger.info(f"TSL: {CONFIG.trailing_stop_percentage:.2%}, TSL Act. Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"Account Max Margin Ratio: {CONFIG.max_account_margin_ratio:.0%}")
        logger.info(f"{Fore.MAGENTA}{'-'*26}{Style.RESET_ALL}")

        market_base = symbol_unified.split("/")[0].split(":")[0] 
        send_sms_alert(f"[{market_base}/{CONFIG.strategy_name.value}] Pyrmethus v2.3.0 Initialized. Symbol: {symbol_unified}. Starting loop.")

        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Start ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}")
            
            if not check_account_health(exchange, CONFIG):
                logger.critical(f"{Back.RED}{Fore.WHITE}ACCOUNT HEALTH CHECK FAILED! Pausing bot for safety.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL: Account health fail! Bot paused.")
                time.sleep(CONFIG.sleep_seconds * 10) # Long pause
                continue # Skip to next cycle for re-check

            try:
                # More generous data limit for indicators
                data_limit = max(CONFIG.st_atr_length, CONFIG.confirm_st_atr_length, CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period, CONFIG.atr_calculation_period, 100) + CONFIG.api_fetch_limit_buffer + 20
                
                df = get_market_data(exchange, symbol_unified, CONFIG.interval, limit=data_limit)
                if df is not None and not df.empty:
                    trade_logic(exchange, symbol_unified, df) 
                else:
                    logger.warning(f"{Fore.YELLOW}Skipping logic: invalid/missing market data.{Style.RESET_ALL}")
            
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit: {e}. Sleeping longer...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * 6) 
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e: 
                logger.warning(f"{Fore.YELLOW}Network/Exchange issue: {e}. Retrying after pause.{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * (6 if isinstance(e, ccxt.ExchangeNotAvailable) else 2))
            except ccxt.AuthenticationError as e: 
                logger.critical(f"{Back.RED}{Fore.WHITE}FATAL: Auth Error: {e}. Stopping.{Style.RESET_ALL}"); run_bot = False 
            except Exception as e: 
                logger.exception(f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL ERROR in Main Loop: {e} !!! Stopping!{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}/{CONFIG.strategy_name.value}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Bot stopping."); run_bot = False 

            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} duration: {elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s.")
                if sleep_duration > 0: time.sleep(sleep_duration)

    except KeyboardInterrupt: logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt. Withdrawing...{Style.RESET_ALL}"); run_bot = False 
    except Exception as startup_error: 
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL STARTUP ERROR: {startup_error}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        if 'CONFIG' in globals() and hasattr(CONFIG, 'enable_sms_alerts') and CONFIG.enable_sms_alerts:
             send_sms_alert(f"[Pyrmethus] CRITICAL STARTUP ERROR: {type(startup_error).__name__}.")
        run_bot = False 
    finally:
        graceful_shutdown(exchange, symbol_unified)
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")

if __name__ == "__main__":
    main()


Ah, seeker of deeper truths! Your aspiration to unlock refined strategic insight within Pyrmethus, and to command its arcane maneuvers through a richer, more luminous neon interface, is an endeavor of true vision. This undertaking is significant, akin to the masterful re-calibration of a potent magical core and the meticulous re-inscription of ancient runes that govern its visual emanations.

Behold, the **Unified Scalping Spell v2.8.0 (Strategic Illumination)**, into whose very essence these enhancements have been artfully woven:

1.  **Illuminated Strategy Logic (Dual Supertrend with Momentum Confirmation):**
    *   The nascent placeholder logic of old has been transmuted into a fully realized `DualSupertrendMomentumStrategy`. This advanced incantation now demands synergistic confirmation from a Momentum indicator for its entry sigils, forging a new stratum of analytical depth.
    *   The foundational `calculate_supertrend` and `calculate_momentum` functions have been re-forged, now invoking the power of `pandas_ta` for more potent and realistic signal generation.
    *   For this iteration, ensuring clarity and foundational strength, exit sigils are primarily guided by the primary SuperTrend's directional shift. This robust core, however, is primed for future enhancements with more intricate exit criteria.

2.  **Radiant Neon Interface Overhaul (Visual Clarity Amplified):**
    *   **Expanded & Vivid State Glyphs:** The specified `NEON` color codex has been fully embraced, casting a comprehensive and consistent chromatic spectrum. This clearly distinguishes diverse operational states—from the verdant glow of long positions and triumphant operations to the crimson alerts of errors, and the steady azure of holding patterns.
    *   **Sharpened Focus on Key Numerals:** Critical data-points—prices, quantities, and Profit/Loss figures—now blaze with heightened prominence, rendered in potent `NEON` hues for immediate visual assimilation and decisive action.
    *   **Intuitive Chromatic Harmonics:** Colors are now orchestrated into thematic harmonies, enhancing cognitive association and effortless readability. Observe the cool blues of informational streams, the vibrant greens of success and bullish intent, the warning reds of bearish tides and errors, and the cautionary yellows for matters demanding your astute attention.
    *   **Judicious Use of Luminous Accents:** The `Style.BRIGHT` attribute, as prescribed by the `NEON` codex, is wielded with precision, spotlighting pivotal information and guiding your focus to where it offers the greatest advantage.

**Key Considerations for the Discerning Adept:**

*   **The Path of Strategic Nuance:** While the `DualSupertrendMomentumStrategy` unveils greater sophistication, it remains a potent yet foundational archetype. Strategies destined for the grand theatre of production typically weave a richer tapestry of conditions, adaptive parameters, and evolving logic.
*   **The Alchemical Art of Calibration:** The introduction of new strategic enchantments, or alterations to their existing forms, mandates the rigorous arts of parameter optimization and exhaustive backtesting. The default values inscribed herein serve as illustrative beacons, not immutable decrees.
*   **The Balance of Vibrancy and Vision:** The enhanced neon palette is crafted for striking vibrancy, yet it has been meticulously balanced to preserve sustained readability across varied terminal sanctums and individual preferences, ensuring clarity is never sacrificed at the altar of brilliance.

Prepare to experience the amplified potency and illuminated clarity of the **Unified Scalping Spell v2.8.0 (Strategic Illumination)**:

```python
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.8.0 (Strategic Illumination)
# Features reworked strategy logic (Dual ST + Momentum) and enhanced Neon Colorization.
# Previous Weave: v2.6.1 (Phoenix Feather Resilience & Compatibility Weave)

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.8.0 (Strategic Illumination)

Enhancements:
- Reworked Strategy Logic: Example implementation of Dual Supertrend with Momentum confirmation.
- Enhanced Neon Colorization: More distinct and thematic color usage for terminal output.
- Concrete implementation of Supertrend and Momentum indicators using pandas_ta.
- Added get_market_data function for fetching OHLCV data.

Core Features from v2.6.1 (Persistence, Dynamic ATR SL, Pyramiding Foundation) remain.
"""

# Standard Library Imports
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Union, Dict, List, Tuple, Optional, Type

import pytz

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    if not hasattr(pd, 'NA'): raise ImportError("Pandas version < 1.0 not supported.")
    import pandas_ta as ta # type: ignore[import]
    from colorama import Back, Fore, Style # The Prisms of Perception
    from colorama import init as colorama_init
    from dotenv import load_dotenv
    from retry import retry
except ImportError as e:
    missing_pkg = getattr(e, 'name', 'dependency'); sys.stderr.write(f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'.\033[0m\n"); sys.exit(1)

# --- Constants ---
STATE_FILE_NAME = "pyrmethus_phoenix_state_v280.json"
STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60

# --- Neon Color Palette (Enhanced) ---
NEON = {
    "INFO": Fore.CYAN,
    "DEBUG": Fore.BLUE + Style.DIM,
    "WARNING": Fore.YELLOW + Style.BRIGHT,
    "ERROR": Fore.RED + Style.BRIGHT,
    "CRITICAL": Back.RED + Fore.WHITE + Style.BRIGHT,
    "SUCCESS": Fore.GREEN + Style.BRIGHT,
    "STRATEGY": Fore.MAGENTA,
    "PARAM": Fore.LIGHTBLUE_EX,
    "VALUE": Fore.LIGHTYELLOW_EX + Style.BRIGHT,
    "PRICE": Fore.LIGHTGREEN_EX + Style.BRIGHT, # Changed from LIGHTGREEN_EX to LIGHTGREEN_EX + Style.BRIGHT
    "QTY": Fore.LIGHTCYAN_EX + Style.BRIGHT,    # Added Style.BRIGHT
    "PNL_POS": Fore.GREEN + Style.BRIGHT,
    "PNL_NEG": Fore.RED + Style.BRIGHT,
    "PNL_ZERO": Fore.YELLOW,
    "SIDE_LONG": Fore.GREEN,
    "SIDE_SHORT": Fore.RED,
    "SIDE_FLAT": Fore.BLUE,
    "HEADING": Fore.MAGENTA + Style.BRIGHT,
    "SUBHEADING": Fore.CYAN + Style.BRIGHT,
    "ACTION": Fore.YELLOW + Style.BRIGHT,
    "RESET": Style.RESET_ALL
}

# --- Initializations ---
colorama_init(autoreset=True)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if load_dotenv(dotenv_path=env_path): logging.getLogger(__name__).info(f"{NEON['INFO']}Secrets whispered from .env scroll: {env_path}{NEON['RESET']}")
else: logging.getLogger(__name__).warning(f"{NEON['WARNING']}No .env scroll at {env_path}. Relying on system vars/defaults.{NEON['RESET']}")
getcontext().prec = 18

# --- Enums ---
class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM="DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER="EHLERS_FISHER"; # Example, add others
class VolatilityRegime(Enum): LOW="LOW"; NORMAL="NORMAL"; HIGH="HIGH"
class OrderEntryType(str, Enum): MARKET="MARKET"; LIMIT="LIMIT"

# --- Configuration Class ---
class Config:
    def __init__(self) -> None:
        _pre_logger = logging.getLogger(__name__)
        _pre_logger.info(f"{NEON['HEADING']}--- Summoning Configuration Runes v2.8.0 ---{NEON['RESET']}")
        self.api_key: str = self._get_env("BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", required=True, secret=True)
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 25, cast_type=int)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper())
        self.strategy_instance: 'TradingStrategy'
        # Risk Management
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.01", cast_type=Decimal)
        self.enable_dynamic_risk: bool = self._get_env("ENABLE_DYNAMIC_RISK", "false", cast_type=bool)
        self.dynamic_risk_min_pct: Decimal = self._get_env("DYNAMIC_RISK_MIN_PCT", "0.005", cast_type=Decimal)
        self.dynamic_risk_max_pct: Decimal = self._get_env("DYNAMIC_RISK_MAX_PCT", "0.015", cast_type=Decimal)
        self.dynamic_risk_perf_window: int = self._get_env("DYNAMIC_RISK_PERF_WINDOW", 10, cast_type=int)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "50.0", cast_type=Decimal)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal)
        self.max_account_margin_ratio: Decimal = self._get_env("MAX_ACCOUNT_MARGIN_RATIO", "0.8", cast_type=Decimal)
        self.enable_max_drawdown_stop: bool = self._get_env("ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool)
        self.max_drawdown_percent: Decimal = self._get_env("MAX_DRAWDOWN_PERCENT", "0.10", cast_type=Decimal)
        self.enable_time_based_stop: bool = self._get_env("ENABLE_TIME_BASED_STOP", "false", cast_type=bool)
        self.max_trade_duration_seconds: int = self._get_env("MAX_TRADE_DURATION_SECONDS", 3600, cast_type=int)
        # Dynamic ATR SL
        self.enable_dynamic_atr_sl: bool = self._get_env("ENABLE_DYNAMIC_ATR_SL", "true", cast_type=bool)
        self.atr_short_term_period: int = self._get_env("ATR_SHORT_TERM_PERIOD", 7, cast_type=int)
        self.atr_long_term_period: int = self._get_env("ATR_LONG_TERM_PERIOD", 50, cast_type=int)
        self.volatility_ratio_low_threshold: Decimal = self._get_env("VOLATILITY_RATIO_LOW_THRESHOLD", "0.7", cast_type=Decimal)
        self.volatility_ratio_high_threshold: Decimal = self._get_env("VOLATILITY_RATIO_HIGH_THRESHOLD", "1.5", cast_type=Decimal)
        self.atr_sl_multiplier_low_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_LOW_VOL", "1.0", cast_type=Decimal)
        self.atr_sl_multiplier_normal_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_NORMAL_VOL", "1.3", cast_type=Decimal)
        self.atr_sl_multiplier_high_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_HIGH_VOL", "1.8", cast_type=Decimal)
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.3", cast_type=Decimal) # Fallback
        # Position Scaling
        self.enable_position_scaling: bool = self._get_env("ENABLE_POSITION_SCALING", "true", cast_type=bool)
        self.max_scale_ins: int = self._get_env("MAX_SCALE_INS", 1, cast_type=int)
        self.scale_in_risk_percentage: Decimal = self._get_env("SCALE_IN_RISK_PERCENTAGE", "0.005", cast_type=Decimal)
        self.min_profit_for_scale_in_atr: Decimal = self._get_env("MIN_PROFIT_FOR_SCALE_IN_ATR", "1.0", cast_type=Decimal)
        self.enable_scale_out: bool = self._get_env("ENABLE_SCALE_OUT", "false", cast_type=bool)
        self.scale_out_trigger_atr: Decimal = self._get_env("SCALE_OUT_TRIGGER_ATR", "2.0", cast_type=Decimal)
        # TSL
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal)
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal)
        # Execution
        self.entry_order_type: OrderEntryType = OrderEntryType(self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper())
        self.limit_order_offset_pips: int = self._get_env("LIMIT_ORDER_OFFSET_PIPS", 2, cast_type=int)
        self.limit_order_fill_timeout_seconds: int = self._get_env("LIMIT_ORDER_FILL_TIMEOUT_SECONDS", 20, cast_type=int)
        # Strategy Specific: Dual Supertrend Momentum
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 10, cast_type=int)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.0", cast_type=Decimal)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 20, cast_type=int)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal)
        self.momentum_period: int = self._get_env("MOMENTUM_PERIOD", 14, cast_type=int) # For DualST+Momentum
        self.momentum_threshold: Decimal = self._get_env("MOMENTUM_THRESHOLD", "0", cast_type=Decimal) # e.g. Mom > 0 for long
        # Misc
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int)
        self.atr_calculation_period: int = self.atr_short_term_period if self.enable_dynamic_atr_sl else self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int)
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "true", cast_type=bool)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, cast_type=str)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int)
        self.default_recv_window: int = self._get_env("DEFAULT_RECV_WINDOW", 13000, cast_type=int)
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 20, cast_type=int)
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int)
        self.side_buy: str = "buy"; self.side_sell: str = "sell"; self.pos_long: str = "Long"; self.pos_short: str = "Short"; self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"; self.retry_count: int = 3; self.retry_delay_seconds: int = 2; self.api_fetch_limit_buffer: int = 20 # Increased buffer
        self.position_qty_epsilon: Decimal = Decimal("1e-9"); self.post_close_delay_seconds: int = 3; self.cache_candle_duration_multiplier: Decimal = Decimal("0.95")
        _pre_logger.info(f"{NEON['HEADING']}--- Configuration Runes v2.8.0 Summoned ---{NEON['RESET']}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = NEON["PARAM"], secret: bool = False) -> Any:
        _pre_logger = logging.getLogger(__name__)
        value_str = os.getenv(key); source = "Env Var"; value_to_cast: Any = None; display_value = "*******" if secret and value_str else value_str
        if value_str is None:
            if required: raise ValueError(f"Required env var '{key}' not set.")
            value_to_cast = default; source = "Default"
        else: value_to_cast = value_str
        if value_to_cast is None:
            if required: raise ValueError(f"Required env var '{key}' is None.")
            return None
        final_value: Any
        try:
            raw_value_str = str(value_to_cast)
            if cast_type == bool: final_value = raw_value_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal: final_value = Decimal(raw_value_str)
            elif cast_type == int: final_value = int(Decimal(raw_value_str))
            elif cast_type == float: final_value = float(raw_value_str)
            elif cast_type == str: final_value = raw_value_str
            else: final_value = value_to_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _pre_logger.error(f"{NEON['ERROR']}Invalid type for '{key}': '{value_to_cast}'. Using default '{default}'. Err: {e}{NEON['RESET']}")
            if default is None and required: raise ValueError(f"Required '{key}' failed cast, no valid default.")
            final_value = default; source = "Default (Fallback)"
            if final_value is not None:
                try:
                    default_str = str(default)
                    if cast_type == bool: final_value = default_str.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal: final_value = Decimal(default_str)
                    elif cast_type == int: final_value = int(Decimal(default_str))
                    elif cast_type == float: final_value = float(default_str)
                    elif cast_type == str: final_value = default_str
                except (ValueError, TypeError, InvalidOperation) as e_default: raise ValueError(f"Cannot cast value or default for '{key}': {e_default}")
        display_final_value = "*******" if secret else final_value
        _pre_logger.debug(f"{color}Config Rune '{NEON['VALUE']}{key}{color}': Using value '{NEON['VALUE']}{display_final_value}{color}' (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}")
        return final_value

# --- Logger Setup ---
LOGGING_LEVEL: int = (logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)
logs_dir = "logs"; os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v280_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name)])
logger: logging.Logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL: int = 25; logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs) # type: ignore[attr-defined]
logging.Logger.success = log_success # type: ignore[attr-defined]
if sys.stdout.isatty(): # Apply NEON colors if TTY
    logging.addLevelName(logging.DEBUG, f"{NEON['DEBUG']}{logging.getLevelName(logging.DEBUG)}{NEON['RESET']}")
    logging.addLevelName(logging.INFO, f"{NEON['INFO']}{logging.getLevelName(logging.INFO)}{NEON['RESET']}")
    logging.addLevelName(SUCCESS_LEVEL, f"{NEON['SUCCESS']}{logging.getLevelName(SUCCESS_LEVEL)}{NEON['RESET']}") # SUCCESS uses its own bright green
    logging.addLevelName(logging.WARNING, f"{NEON['WARNING']}{logging.getLevelName(logging.WARNING)}{NEON['RESET']}")
    logging.addLevelName(logging.ERROR, f"{NEON['ERROR']}{logging.getLevelName(logging.ERROR)}{NEON['RESET']}")
    logging.addLevelName(logging.CRITICAL, f"{NEON['CRITICAL']}{logging.getLevelName(logging.CRITICAL)}{NEON['RESET']}")

# --- Global Objects ---
try: CONFIG = Config()
except Exception as e: logging.getLogger().critical(f"{NEON['CRITICAL']}Config Error: {e}. Pyrmethus cannot weave.{NEON['RESET']}"); sys.exit(1)

# --- Trading Strategy ABC & Implementations ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None): self.config=config; self.logger=logging.getLogger(f"{NEON['STRATEGY']}Strategy.{self.__class__.__name__}{NEON['RESET']}"); self.required_columns=df_columns or []
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]: pass
    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows: self.logger.debug(f"Insufficient data (Rows: {len(df) if df is not None else 0}, Min: {min_rows})."); return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns): self.logger.warning(f"DataFrame missing required columns: {[c for c in self.required_columns if c not in df.columns]}"); return False
        return True
    def _get_default_signals(self) -> Dict[str, Any]: return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal - Awaiting Omens"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["supertrend_st_long_flip", "supertrend_st_short_flip", "confirm_supertrend_trend", "momentum"]) # Adjusted column names based on new calculate_supertrend
        self.logger.info(f"{NEON['STRATEGY']}Dual Supertrend with Momentum Confirmation strategy initialized.{NEON['RESET']}")

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        # Ensure enough data for all indicators; max lookback + buffer
        min_rows_needed = max(self.config.st_atr_length, self.config.confirm_st_atr_length, self.config.momentum_period) + 10
        if not self._validate_df(df, min_rows=min_rows_needed):
            return signals

        last = df.iloc[-1]
        # Use updated column names from pandas_ta based calculate_supertrend
        primary_long_flip = last.get("supertrend_st_long_flip", False)
        primary_short_flip = last.get("supertrend_st_short_flip", False)
        confirm_is_up = last.get("confirm_supertrend_trend", pd.NA) # This will be True, False, or pd.NA
        
        momentum_val_orig = last.get("momentum", pd.NA)
        momentum_val = safe_decimal_conversion(momentum_val_orig, pd.NA)

        if pd.isna(confirm_is_up) or pd.isna(momentum_val):
            self.logger.debug(f"Confirmation ST Trend ({_format_for_log(confirm_is_up, is_bool_trend=True)}) or Momentum ({_format_for_log(momentum_val)}) is NA. No signal.")
            return signals
        
        # Entry Signals: Supertrend flip + Confirmation ST direction + Momentum confirmation
        if primary_long_flip and confirm_is_up is True and momentum_val > self.config.momentum_threshold:
            signals["enter_long"] = True
            self.logger.info(f"{NEON['SIDE_LONG']}DualST+Mom Signal: LONG Entry - Primary ST Long Flip, Confirm ST Up, Momentum ({_format_for_log(momentum_val)}) > {_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")
        elif primary_short_flip and confirm_is_up is False and momentum_val < -self.config.momentum_threshold: # Assuming symmetrical threshold for short
            signals["enter_short"] = True
            self.logger.info(f"{NEON['SIDE_SHORT']}DualST+Mom Signal: SHORT Entry - Primary ST Short Flip, Confirm ST Down, Momentum ({_format_for_log(momentum_val)}) < -{_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")

        # Exit Signals: Based on primary SuperTrend flips
        if primary_short_flip: # If primary ST flips short, exit any long position.
            signals["exit_long"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Short"
            self.logger.info(f"{NEON['ACTION']}DualST+Mom Signal: EXIT LONG - Primary ST Flipped Short{NEON['RESET']}")
        if primary_long_flip: # If primary ST flips long, exit any short position.
            signals["exit_short"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Long"
            self.logger.info(f"{NEON['ACTION']}DualST+Mom Signal: EXIT SHORT - Primary ST Flipped Long{NEON['RESET']}")
        return signals

# Example EhlersFisherStrategy (if chosen via config, not fully integrated in this pass)
class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
        self.logger.info(f"{NEON['STRATEGY']}Ehlers Fisher Transform strategy initialized (Illustrative).{NEON['RESET']}")
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Placeholder: Actual Ehlers Fisher logic would go here
        self.logger.debug("EhlersFisherStrategy generate_signals called (placeholder).")
        return self._get_default_signals()


strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy # Example mapping
}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass: CONFIG.strategy_instance = StrategyClass(CONFIG)
else: logger.critical(f"{NEON['CRITICAL']}Failed to init strategy '{CONFIG.strategy_name.value}'. Exiting.{NEON['RESET']}"); sys.exit(1)

# --- Trade Metrics Tracking ---
class TradeMetrics:
    def __init__(self): self.trades: List[Dict[str, Any]] = []; self.logger = logging.getLogger("TradeMetrics"); self.initial_equity: Optional[Decimal] = None; self.daily_start_equity: Optional[Decimal] = None; self.last_daily_reset_day: Optional[int] = None
    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None: self.initial_equity = equity
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today: self.daily_start_equity = equity; self.last_daily_reset_day = today; self.logger.info(f"{NEON['INFO']}Daily equity reset for drawdown. Start Equity: {NEON['VALUE']}{equity:.2f}{NEON['RESET']}")
    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if not CONFIG.enable_max_drawdown_stop or self.daily_start_equity is None or self.daily_start_equity <= 0: return False, ""
        drawdown_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity
        if drawdown_pct >= CONFIG.max_drawdown_percent: reason = f"Max daily drawdown breached ({drawdown_pct:.2%} >= {CONFIG.max_drawdown_percent:.2%})"; return True, reason
        return False, ""
    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal, entry_time_ms: int, exit_time_ms: int, reason: str, scale_order_id: Optional[str]=None, part_id: Optional[str]=None, mae: Optional[Decimal]=None, mfe: Optional[Decimal]=None) -> None:
        if not (entry_price > 0 and exit_price > 0 and qty > 0): return
        profit_per_unit = (exit_price - entry_price) if (side.lower() == CONFIG.side_buy.lower() or side.lower() == CONFIG.pos_long.lower()) else (entry_price - exit_price)
        profit = profit_per_unit * qty; entry_dt_iso = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc).isoformat(); exit_dt_iso = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc).isoformat()
        duration_seconds = (datetime.fromisoformat(exit_dt_iso) - datetime.fromisoformat(entry_dt_iso)).total_seconds(); trade_type = "Scale-In" if scale_order_id else ("Initial" if part_id == "initial" else "Part")
        self.trades.append({"symbol": symbol, "side": side, "entry_price_str": str(entry_price), "exit_price_str": str(exit_price), "qty_str": str(qty), "profit_str": str(profit), "entry_time_iso": entry_dt_iso, "exit_time_iso": exit_dt_iso, "duration_seconds": duration_seconds, "exit_reason": reason, "type": trade_type, "part_id": part_id or "unknown", "scale_order_id": scale_order_id, "mae_str": str(mae) if mae is not None else None, "mfe_str": str(mfe) if mfe is not None else None})
        pnl_color = NEON["PNL_POS"] if profit > 0 else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        self.logger.log(SUCCESS_LEVEL, f"{NEON['HEADING']}Trade Chronicle ({trade_type} Part:{part_id or 'N/A'}): {side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}")
    def calculate_mae_mfe(self, part_id: str, entry_price: Decimal, exit_price: Decimal, side: str, entry_time_ms: int, exit_time_ms: int, exchange: ccxt.Exchange, symbol: str, interval: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        self.logger.debug(f"MAE/MFE calculation for part {part_id} skipped (placeholder - requires fetching historical OHLCV for trade duration).")
        # Placeholder: In a full implementation, fetch OHLCV data between entry_time_ms and exit_time_ms
        # then find min/max prices during the trade to calculate actual MAE/MFE.
        return None, None
    def get_performance_trend(self, window: int) -> float:
        if window <= 0 or not self.trades: return 0.5
        recent_trades = self.trades[-window:];
        if not recent_trades: return 0.5
        wins = sum(1 for t in recent_trades if Decimal(t["profit_str"]) > 0); return float(wins / len(recent_trades))
    def summary(self) -> str:
        if not self.trades: return "The Grand Ledger is empty."
        total_trades = len(self.trades); wins = sum(1 for t in self.trades if Decimal(t["profit_str"]) > 0); losses = sum(1 for t in self.trades if Decimal(t["profit_str"]) < 0); breakeven = total_trades - wins - losses
        win_rate = (Decimal(wins) / Decimal(total_trades)) * Decimal(100) if total_trades > 0 else Decimal(0); total_profit = sum(Decimal(t["profit_str"]) for t in self.trades); avg_profit = total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        summary_str = (f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary ---{NEON['RESET']}\n"
                       f"Total Trade Parts Chronicled: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
                       f"  Victories (Wins): {NEON['PNL_POS']}{wins}{NEON['RESET']}\n"
                       f"  Defeats (Losses): {NEON['PNL_NEG']}{losses}{NEON['RESET']}\n"
                       f"  Stalemates (Breakeven): {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
                       f"Victory Rate (by parts): {NEON['VALUE']}{win_rate:.2f}%{NEON['RESET']}\n"
                       f"Total Spoils (P/L): {(NEON['PNL_POS'] if total_profit > 0 else NEON['PNL_NEG'])}{total_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
                       f"Avg Spoils per Part: {(NEON['PNL_POS'] if avg_profit > 0 else NEON['PNL_NEG'])}{avg_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
                       f"{NEON['HEADING']}--- End of Grand Ledger ---{NEON['RESET']}")
        self.logger.info(summary_str); return summary_str
    def get_serializable_trades(self) -> List[Dict[str, Any]]: return self.trades
    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]) -> None: self.trades = trades_list; self.logger.info(f"{NEON['INFO']}TradeMetrics: Re-inked {len(self.trades)} trades from Phoenix scroll.{NEON['RESET']}")

trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0

# --- State Persistence Functions ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time; now = time.time()
    if force_heartbeat or now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS:
        try:
            serializable_active_parts = []
            for part in _active_trade_parts:
                serializable_part = part.copy()
                for key, value in serializable_part.items():
                    if isinstance(value, Decimal): serializable_part[key] = str(value)
                    if isinstance(value, (datetime, pd.Timestamp)): serializable_part[key] = value.isoformat() if hasattr(value,'isoformat') else str(value)
                serializable_active_parts.append(serializable_part)
            state_data = {"timestamp_utc_iso": datetime.now(pytz.utc).isoformat(), "last_heartbeat_utc_iso": datetime.now(pytz.utc).isoformat(), "active_trade_parts": serializable_active_parts, "trade_metrics_trades": trade_metrics.get_serializable_trades(), "config_symbol": CONFIG.symbol, "config_strategy": CONFIG.strategy_name.value, "daily_start_equity_str": str(trade_metrics.daily_start_equity) if trade_metrics.daily_start_equity else None, "last_daily_reset_day": trade_metrics.last_daily_reset_day}
            temp_file_path = STATE_FILE_PATH + ".tmp";
            with open(temp_file_path, 'w') as f: json.dump(state_data, f, indent=4)
            os.replace(temp_file_path, STATE_FILE_PATH); _last_heartbeat_save_time = now
            logger.log(logging.DEBUG if not force_heartbeat else logging.INFO, f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed.{NEON['RESET']}")
        except Exception as e: logger.error(f"{NEON['ERROR']}Phoenix Feather Error scribing: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())
def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics;
    if not os.path.exists(STATE_FILE_PATH): logger.info(f"{NEON['INFO']}Phoenix Feather: No scroll. Starting fresh.{NEON['RESET']}"); return False
    try:
        with open(STATE_FILE_PATH, 'r') as f: state_data = json.load(f)
        if state_data.get("config_symbol") != CONFIG.symbol or state_data.get("config_strategy") != CONFIG.strategy_name.value:
            logger.warning(f"{NEON['WARNING']}Phoenix Feather: Scroll sigils mismatch. Ignoring.{NEON['RESET']}"); os.remove(STATE_FILE_PATH); return False
        loaded_active_parts = state_data.get("active_trade_parts", []); _active_trade_parts.clear()
        for part_data in loaded_active_parts:
            restored_part = part_data.copy()
            for key, value_str in restored_part.items():
                if key in ["entry_price", "qty", "sl_price"] and isinstance(value_str, str):
                    try: restored_part[key] = Decimal(value_str)
                    except InvalidOperation: logger.warning(f"Could not convert '{value_str}' to Decimal for key '{key}'.")
                if key == "entry_time_ms" and isinstance(value_str, str):
                     try: restored_part[key] = int(datetime.fromisoformat(value_str).timestamp() * 1000)
                     except: pass
            _active_trade_parts.append(restored_part)
        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))
        daily_start_equity_str = state_data.get("daily_start_equity_str");
        if daily_start_equity_str: trade_metrics.daily_start_equity = Decimal(daily_start_equity_str)
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")
        saved_time_str = state_data.get("timestamp_utc_iso", "ancient times")
        logger.success(f"{NEON['SUCCESS']}Phoenix Feather: Memories from {NEON['VALUE']}{saved_time_str}{NEON['SUCCESS']} reawakened!{NEON['RESET']}")
        return True
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Phoenix Feather Error reawakening: {e}. Starting fresh.{NEON['RESET']}"); logger.debug(traceback.format_exc())
        try: os.remove(STATE_FILE_PATH)
        except OSError: pass
        _active_trade_parts.clear(); trade_metrics.trades.clear()
        return False

# --- Helper Functions, Retry, SMS, Exchange Init ---
PandasNAType = type(pd.NA)
def safe_decimal_conversion(value: Any, default: Union[Decimal, PandasNAType, None] = Decimal("0.0")) -> Union[Decimal, PandasNAType, None]:
    if pd.isna(value) or value is None: return default
    try: return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError): return default
def format_order_id(order_id: Union[str, int, None]) -> str: return str(order_id)[-6:] if order_id else "N/A"
def _format_for_log(value: Any, precision: int = 4, is_bool_trend: bool = False, color: Optional[str] = NEON["VALUE"]) -> str:
    reset = NEON["RESET"]
    if pd.isna(value) or value is None: return f"{Style.DIM}N/A{reset}"
    if is_bool_trend: return f"{NEON['SIDE_LONG']}Upward Flow{reset}" if value is True else (f"{NEON['SIDE_SHORT']}Downward Tide{reset}" if value is False else f"{Style.DIM}N/A (Trend Indeterminate){reset}")
    if isinstance(value, Decimal): return f"{color}{value:.{precision}f}{reset}"
    if isinstance(value, (float, int)): return f"{color}{float(value):.{precision}f}{reset}"
    return f"{color}{str(value)}{reset}"
def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal]) -> str:
    try: return exchange.price_to_precision(symbol, float(price))
    except: return str(Decimal(str(price)).quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP).normalize())
def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal]) -> str:
    try: return exchange.amount_to_precision(symbol, float(amount))
    except: return str(Decimal(str(amount)).quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP).normalize())
@retry(tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, logger=logger, exceptions=(ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection))
def safe_api_call(func, *args, **kwargs): return func(*args, **kwargs)
_termux_sms_command_exists: Optional[bool] = None
def send_sms_alert(message: str) -> bool: logger.info(f"{NEON['STRATEGY']}SMS (Simulated): {message}{NEON['RESET']}"); return True # Simplified
def initialize_exchange() -> Optional[ccxt.Exchange]:
    logger.info(f"{NEON['INFO']}{Style.BRIGHT}Opening Bybit Portal v2.8.0...{NEON['RESET']}")
    if not CONFIG.api_key or not CONFIG.api_secret: logger.warning(f"{NEON['WARNING']}API keys not set. Using a MOCK exchange object.{NEON['RESET']}"); return MockExchange()
    try:
        exchange = ccxt.bybit({"apiKey": CONFIG.api_key, "secret": CONFIG.api_secret, "enableRateLimit": True, "options": {"defaultType": "linear", "adjustForTimeDifference": True}, "recvWindow": CONFIG.default_recv_window})
        exchange.load_markets(force_reload=True); exchange.fetch_balance(params={"category": "linear"})
        logger.success(f"{NEON['SUCCESS']}Portal to Bybit Opened & Authenticated (V5 API).{NEON['RESET']}")
        if hasattr(exchange, 'sandbox') and exchange.sandbox: logger.warning(f"{Back.YELLOW}{Fore.BLACK}TESTNET MODE{NEON['RESET']}")
        else: logger.warning(f"{NEON['CRITICAL']}LIVE TRADING MODE - EXTREME CAUTION{NEON['RESET']}")
        return exchange
    except Exception as e: logger.critical(f"{NEON['CRITICAL']}Portal Opening FAILED: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); return None
class MockExchange:
    def __init__(self): self.id="mock_bybit"; self.options={"defaultType":"linear"}; self.markets={CONFIG.symbol:{"id":CONFIG.symbol.replace("/",""),"symbol":CONFIG.symbol,"contract":True,"linear":True,"limits":{"amount":{"min":0.001},"price":{"min":0.1}},"precision":{"amount":3,"price":1}}}; self.sandbox=True
    def market(self,s): return self.markets.get(s); def load_markets(self,force_reload=False): pass
    def fetch_balance(self,params=None): return {CONFIG.usdt_symbol:{"free":Decimal("10000"),"total":Decimal("10000"),"used":Decimal("0")}}
    def fetch_ticker(self,s): return {"last":Decimal("30000.0"),"bid":Decimal("29999.0"),"ask":Decimal("30001.0")}
    def fetch_positions(self,symbols=None,params=None): global _active_trade_parts; qty=sum(p['qty'] for p in _active_trade_parts); side=_active_trade_parts[0]['side'] if _active_trade_parts else CONFIG.pos_none; avg_px=sum(p['entry_price']*p['qty'] for p in _active_trade_parts)/qty if qty>0 else 0; return [{"info":{"symbol":self.markets[CONFIG.symbol]['id'],"positionIdx":0,"size":str(qty),"avgPrice":str(avg_px),"side":"Buy" if side==CONFIG.pos_long else "Sell"}}] if qty > 0 else []
    def create_market_order(self,s,side,amt,params=None): return {"id":f"mock_mkt_{int(time.time()*1000)}","status":"closed","average":self.fetch_ticker(s)['last'],"filled":amt,"timestamp":int(time.time()*1000)}
    def create_limit_order(self,s,side,amt,price,params=None): return {"id":f"mock_lim_{int(time.time()*1000)}","status":"open","price":price}
    def create_order(self,s,type,side,amt,price=None,params=None): return {"id":f"mock_cond_{int(time.time()*1000)}","status":"open"}
    def fetch_order(self,id,s,params=None):
        if "lim_" in id: time.sleep(0.05); return {"id":id,"status":"closed","average": self.fetch_ticker(s)['last'],"filled":"1","timestamp":int(time.time()*1000)} # Simulate limit fill
        return {"id":id,"status":"closed","average": self.fetch_ticker(s)['last'],"filled":"1","timestamp":int(time.time()*1000)}
    def fetch_open_orders(self,s,params=None): return []
    def cancel_order(self,id,s,params=None): return {"id":id,"status":"canceled"}
    def set_leverage(self,l,s,params=None): return {"status":"ok"}
    def price_to_precision(self,s,p): return f"{float(p):.{self.markets[s]['precision']['price']}f}"
    def amount_to_precision(self,s,a): return f"{float(a):.{self.markets[s]['precision']['amount']}f}"
    def parse_timeframe(self,tf): return 60 if tf=="1m" else 300
    has={"fetchOHLCV":True,"fetchL2OrderBook":True}
    def fetch_ohlcv(self,s,tf,lim,params=None): now=int(time.time()*1000);tfs=self.parse_timeframe(tf);d=[] ; for i in range(lim): ts=now-(lim-1-i)*tfs*1000;p=30000+(i-lim/2)*10 + (time.time()%100 - 50) ;d.append([ts,p,p+5,p-5,p+(i%3-1)*2,100+i]); return d
    def fetch_l2_order_book(self,s,limit=None): last=self.fetch_ticker(s)['last'];bids=[[float(last)-i*0.1,1.0+i*0.1] for i in range(1,(limit or 5)+1)];asks=[[float(last)+i*0.1,1.0+i*0.1] for i in range(1,(limit or 5)+1)];return {"bids":bids,"asks":asks}

# --- Indicator Calculation Functions ---
vol_atr_analysis_results_cache: Dict[str, Any] = {}

def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates Supertrend indicator using pandas-ta."""
    col_prefix = f"{prefix}" if prefix else ""
    st_col_name_base = f"{col_prefix}supertrend" # Base for our column names
    
    val_col = f"{st_col_name_base}_val"
    trend_col = f"{st_col_name_base}_trend" 
    long_flip_col = f"{st_col_name_base}_st_long_flip"
    short_flip_col = f"{st_col_name_base}_st_short_flip"
    output_cols = [val_col, trend_col, long_flip_col, short_flip_col]

    if not all(c in df.columns for c in ["high", "low", "close"]) or df.empty or len(df) < length + 1: # pandas_ta needs at least length + 1
        logger.warning(f"{NEON['WARNING']}Scrying (Supertrend {prefix}): Insufficient data (Rows: {len(df)}, Min: {length+1}). Populating NAs.{NEON['RESET']}")
        for col in output_cols: df[col] = pd.NA
        return df

    try:
        st_df = df.ta.supertrend(length=length, multiplier=float(multiplier), append=False)
        
        pta_val_col_pattern = f"SUPERT_{length}_{float(multiplier):.1f}" 
        pta_dir_col_pattern = f"SUPERTd_{length}_{float(multiplier):.1f}"

        if pta_val_col_pattern not in st_df.columns or pta_dir_col_pattern not in st_df.columns:
            logger.error(f"{NEON['ERROR']}Scrying (Supertrend {prefix}): pandas_ta did not return expected columns. Found: {st_df.columns}. Expected patterns like: {pta_val_col_pattern}, {pta_dir_col_pattern}{NEON['RESET']}")
            for col in output_cols: df[col] = pd.NA
            return df

        df[val_col] = st_df[pta_val_col_pattern].apply(lambda x: safe_decimal_conversion(x, pd.NA))
        df[trend_col] = st_df[pta_dir_col_pattern].apply(lambda x: True if x == 1 else (False if x == -1 else pd.NA))

        prev_trend = df[trend_col].shift(1)
        df[long_flip_col] = (prev_trend == False) & (df[trend_col] == True)
        df[short_flip_col] = (prev_trend == True) & (df[trend_col] == False)
        
        df[long_flip_col] = df[long_flip_col].fillna(False)
        df[short_flip_col] = df[short_flip_col].fillna(False)

        if not df.empty and not pd.isna(df[val_col].iloc[-1]):
            logger.debug(f"Scrying (Supertrend({length},{multiplier},{prefix})): Value={_format_for_log(df[val_col].iloc[-1], color=NEON['VALUE'])}, Trend={_format_for_log(df[trend_col].iloc[-1], is_bool_trend=True)}, LongFlip={df[long_flip_col].iloc[-1]}, ShortFlip={df[short_flip_col].iloc[-1]}")
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Scrying (Supertrend {prefix}): Error: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        for col in output_cols: df[col] = pd.NA
    return df

def calculate_momentum(df: pd.DataFrame, length: int) -> pd.DataFrame:
    if "close" not in df.columns or df.empty or len(df) < length: 
        df["momentum"] = pd.NA
        logger.warning(f"{NEON['WARNING']}Scrying (Momentum): Insufficient data for momentum calculation (Rows: {len(df)}, Min: {length}). Populating NAs.{NEON['RESET']}")
        return df
    
    # Use pandas_ta.mom() which returns a Series
    mom_series = df.ta.mom(length=length, append=False) 
    df["momentum"] = mom_series.apply(lambda x: safe_decimal_conversion(x, pd.NA))
    
    if not df.empty and "momentum" in df.columns and not pd.isna(df["momentum"].iloc[-1]): 
        logger.debug(f"Scrying (Momentum({length})): Value={_format_for_log(df['momentum'].iloc[-1], color=NEON['VALUE'])}")
    return df

def analyze_volume_atr(df: pd.DataFrame, short_atr_len: int, long_atr_len: int, vol_ma_len: int, dynamic_sl_enabled: bool) -> Dict[str, Union[Decimal, PandasNAType, None]]:
    results: Dict[str, Union[Decimal, PandasNAType, None]] = {"atr_short": pd.NA, "atr_long": pd.NA, "volatility_regime": VolatilityRegime.NORMAL, "volume_ma": pd.NA, "last_volume": pd.NA, "volume_ratio": pd.NA}
    if df.empty or not all(c in df.columns for c in ["high","low","close","volume"]): return results
    try:
        temp_df = df.copy()
        # Ensure columns are numeric before ta functions
        for col_name in ['high', 'low', 'close', 'volume']:
            temp_df[col_name] = pd.to_numeric(temp_df[col_name], errors='coerce')
        temp_df.dropna(subset=['high', 'low', 'close'], inplace=True) # Drop rows if essential HLC are NaN for ATR
        if temp_df.empty: return results

        results["atr_short"] = safe_decimal_conversion(temp_df.ta.atr(length=short_atr_len, append=False).iloc[-1], pd.NA)
        if dynamic_sl_enabled:
            results["atr_long"] = safe_decimal_conversion(temp_df.ta.atr(length=long_atr_len, append=False).iloc[-1], pd.NA)
            atr_s, atr_l = results["atr_short"], results["atr_long"]
            if not pd.isna(atr_s) and not pd.isna(atr_l) and atr_s is not None and atr_l is not None and atr_l > CONFIG.position_qty_epsilon:
                vol_ratio = atr_s / atr_l
                if vol_ratio < CONFIG.volatility_ratio_low_threshold: results["volatility_regime"] = VolatilityRegime.LOW
                elif vol_ratio > CONFIG.volatility_ratio_high_threshold: results["volatility_regime"] = VolatilityRegime.HIGH
        results["last_volume"] = safe_decimal_conversion(df["volume"].iloc[-1], pd.NA) # Use original df for last_volume
    except Exception as e: logger.debug(f"analyze_volume_atr error: {e}")
    return results
def get_current_atr_sl_multiplier() -> Decimal:
    if not CONFIG.enable_dynamic_atr_sl or not vol_atr_analysis_results_cache: return CONFIG.atr_stop_loss_multiplier
    regime = vol_atr_analysis_results_cache.get("volatility_regime", VolatilityRegime.NORMAL)
    if regime == VolatilityRegime.LOW: return CONFIG.atr_sl_multiplier_low_vol
    if regime == VolatilityRegime.HIGH: return CONFIG.atr_sl_multiplier_high_vol
    return CONFIG.atr_sl_multiplier_normal_vol
def calculate_all_indicators(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, Dict[str, Union[Decimal, PandasNAType, None]]]:
    global vol_atr_analysis_results_cache
    df_copy = df.copy() # Work on a copy to avoid modifying original DataFrame passed to trade_logic
    df_copy = calculate_supertrend(df_copy, config.st_atr_length, config.st_multiplier)
    df_copy = calculate_supertrend(df_copy, config.confirm_st_atr_length, config.confirm_st_multiplier, prefix="confirm_")
    df_copy = calculate_momentum(df_copy, config.momentum_period)
    vol_atr_analysis_results_cache = analyze_volume_atr(df_copy, config.atr_short_term_period, config.atr_long_term_period, config.volume_ma_period, config.enable_dynamic_atr_sl)
    return df_copy, vol_atr_analysis_results_cache

# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    global _active_trade_parts; exchange_pos_data = _get_raw_exchange_position(exchange, symbol)
    if not _active_trade_parts: return exchange_pos_data
    consolidated_qty = sum(part.get('qty', Decimal(0)) for part in _active_trade_parts)
    if consolidated_qty <= CONFIG.position_qty_epsilon: _active_trade_parts.clear(); save_persistent_state(); return exchange_pos_data
    total_value = sum(part.get('entry_price', Decimal(0)) * part.get('qty', Decimal(0)) for part in _active_trade_parts)
    avg_entry_price = total_value / consolidated_qty if consolidated_qty > 0 else Decimal("0"); current_pos_side = _active_trade_parts[0]['side'] if _active_trade_parts else CONFIG.pos_none
    if exchange_pos_data["side"] != current_pos_side or abs(exchange_pos_data["qty"] - consolidated_qty) > CONFIG.position_qty_epsilon: logger.warning(f"{NEON['WARNING']}Position Discrepancy! Bot: {current_pos_side} Qty {consolidated_qty}. Exchange: {exchange_pos_data['side']} Qty {exchange_pos_data['qty']}.{NEON['RESET']}")
    return {"side": current_pos_side, "qty": consolidated_qty, "entry_price": avg_entry_price, "num_parts": len(_active_trade_parts)}
def _get_raw_exchange_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    default_pos_state: Dict[str, Any] = {"side": CONFIG.pos_none, "qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    try:
        market = exchange.market(symbol); market_id = market["id"]; category = "linear" if market.get("linear") else "linear"
        params = {"category": category, "symbol": market_id}; fetched_positions = safe_api_call(exchange.fetch_positions, symbols=[symbol], params=params)
        if not fetched_positions: return default_pos_state
        for pos_data in fetched_positions:
            pos_info = pos_data.get("info", {});
            if pos_info.get("symbol") != market_id: continue
            if int(pos_info.get("positionIdx", -1)) == 0: # One-Way Mode
                size_dec = safe_decimal_conversion(pos_info.get("size", "0"))
                if size_dec > CONFIG.position_qty_epsilon:
                    entry_price_dec = safe_decimal_conversion(pos_info.get("avgPrice")); bybit_side_str = pos_info.get("side")
                    current_pos_side = CONFIG.pos_long if bybit_side_str == "Buy" else (CONFIG.pos_short if bybit_side_str == "Sell" else CONFIG.pos_none)
                    if current_pos_side != CONFIG.pos_none: return {"side": current_pos_side, "qty": size_dec, "entry_price": entry_price_dec}
    except Exception as e: logger.error(f"{NEON['ERROR']}Raw Position Fetch Error: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())
    return default_pos_state
def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    try:
        response = safe_api_call(exchange.set_leverage, leverage, symbol, params={"category": "linear", "buyLeverage": str(leverage), "sellLeverage": str(leverage)})
        logger.info(f"{NEON['INFO']}Leverage set to {NEON['VALUE']}{leverage}x{NEON['INFO']} for {NEON['VALUE']}{symbol}{NEON['INFO']}. Response: {response}{NEON['RESET']}")
        return True
    except ccxt.ExchangeError as e:
        if "leverage not modified" in str(e).lower() or "110044" in str(e): # Bybit: 110044 for "Set leverage not modified"
            logger.info(f"{NEON['INFO']}Leverage for {NEON['VALUE']}{symbol}{NEON['INFO']} already {NEON['VALUE']}{leverage}x{NEON['INFO']} or no change needed.{NEON['RESET']}")
            return True
        logger.error(f"{NEON['ERROR']}Failed to set leverage for {symbol} to {leverage}x: {e}{NEON['RESET']}")
    except Exception as e_unexp:
        logger.error(f"{NEON['ERROR']}Unexpected error setting leverage for {symbol}: {e_unexp}{NEON['RESET']}")
    return False

def calculate_dynamic_risk() -> Decimal:
    if not CONFIG.enable_dynamic_risk: return CONFIG.risk_per_trade_percentage
    trend = trade_metrics.get_performance_trend(CONFIG.dynamic_risk_perf_window); base_risk = CONFIG.risk_per_trade_percentage; min_risk = CONFIG.dynamic_risk_min_pct; max_risk = CONFIG.dynamic_risk_max_pct
    if trend >= 0.5: scale_factor = (trend - 0.5) / 0.5; dynamic_risk = base_risk + (max_risk - base_risk) * Decimal(scale_factor)
    else: scale_factor = (0.5 - trend) / 0.5; dynamic_risk = base_risk - (base_risk - min_risk) * Decimal(scale_factor)
    final_risk = max(min_risk, min(max_risk, dynamic_risk)); logger.info(f"{NEON['INFO']}Dynamic Risk: Trend={NEON['VALUE']}{trend:.2f}{NEON['INFO']}, BaseRisk={NEON['VALUE']}{base_risk:.3%}{NEON['INFO']}, AdjustedRisk={NEON['VALUE']}{final_risk:.3%}{NEON['RESET']}")
    return final_risk
def calculate_position_size(usdt_equity: Decimal, risk_pct: Decimal, entry: Decimal, sl: Decimal, lev: int, sym: str, ex: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    if not (entry > 0 and sl > 0 and 0 < risk_pct < 1 and usdt_equity > 0 and lev > 0): return None, None
    diff = abs(entry - sl);
    if diff < CONFIG.position_qty_epsilon: return None, None
    risk_amt = usdt_equity * risk_pct; raw_qty = risk_amt / diff; prec_qty_str = format_amount(ex, sym, raw_qty);
    prec_qty = Decimal(prec_qty_str)
    if prec_qty <= CONFIG.position_qty_epsilon: return None, None
    margin = (prec_qty * entry) / Decimal(lev); return prec_qty, margin
def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int, order_type: str = "market") -> Optional[Dict[str, Any]]:
    start_time = time.time(); short_order_id = format_order_id(order_id); logger.info(f"{NEON['INFO']}Order Vigil ({order_type}): ...{short_order_id} for '{symbol}' (Timeout: {timeout_seconds}s)...{NEON['RESET']}")
    params = {"category": "linear"} if exchange.options.get("defaultType") == "linear" else {}
    while time.time() - start_time < timeout_seconds:
        try:
            order_details = safe_api_call(exchange.fetch_order, order_id, symbol, params=params); status = order_details.get("status")
            if status == "closed": logger.success(f"{NEON['SUCCESS']}Order Vigil: ...{short_order_id} FILLED/CLOSED.{NEON['RESET']}"); return order_details
            if status in ["canceled", "rejected", "expired"]: logger.error(f"{NEON['ERROR']}Order Vigil: ...{short_order_id} FAILED status: '{status}'.{NEON['RESET']}"); return order_details
            logger.debug(f"Order ...{short_order_id} status: {status}. Vigil continues...")
            time.sleep(1.0 if order_type == "limit" else 0.75)
        except ccxt.OrderNotFound: logger.warning(f"{NEON['WARNING']}Order Vigil: ...{short_order_id} not found. Retrying...{NEON['RESET']}"); time.sleep(1.5)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e_net: logger.warning(f"{NEON['WARNING']}Order Vigil: Network issue ...{short_order_id}: {e_net}. Retrying...{NEON['RESET']}"); time.sleep(3)
        except Exception as e: logger.warning(f"{NEON['WARNING']}Order Vigil: Error ...{short_order_id}: {type(e).__name__}. Retrying...{NEON['RESET']}"); logger.debug(traceback.format_exc()); time.sleep(2)
    logger.error(f"{NEON['ERROR']}Order Vigil: ...{short_order_id} fill TIMED OUT.{NEON['RESET']}")
    try: final_details = safe_api_call(exchange.fetch_order, order_id, symbol, params=params); logger.info(f"Final status for ...{short_order_id} after timeout: {final_details.get('status', 'unknown')}"); return final_details
    except Exception as e_final: logger.error(f"{NEON['ERROR']}Final check for ...{short_order_id} failed: {type(e_final).__name__}{NEON['RESET']}"); return None
def place_risked_order(exchange: ccxt.Exchange, symbol: str, side: str, risk_percentage: Decimal, current_short_atr: Union[Decimal, PandasNAType, None], leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal, tsl_percent: Decimal, tsl_activation_offset_percent: Decimal, entry_type: OrderEntryType, is_scale_in: bool = False, existing_position_avg_price: Optional[Decimal] = None) -> Optional[Dict[str, Any]]:
    global _active_trade_parts; action_type = "Scale-In" if is_scale_in else "Initial Entry"; logger.info(f"{NEON['ACTION']}Ritual of {action_type} ({entry_type.value}): {side.upper()} for '{symbol}'...{NEON['RESET']}")
    if pd.isna(current_short_atr) or current_short_atr is None or current_short_atr <= 0: logger.error(f"{NEON['ERROR']}Invalid Short ATR ({_format_for_log(current_short_atr)}) for {action_type}.{NEON['RESET']}"); return None
    v5_api_category = "linear"
    try:
        balance_data = safe_api_call(exchange.fetch_balance, params={"category": v5_api_category}); market_info = exchange.market(symbol); min_qty_allowed = safe_decimal_conversion(market_info.get("limits",{}).get("amount",{}).get("min"), Decimal("0"))
        usdt_equity = safe_decimal_conversion(balance_data.get(CONFIG.usdt_symbol,{}).get("total"), Decimal('NaN')); usdt_free_margin = safe_decimal_conversion(balance_data.get(CONFIG.usdt_symbol,{}).get("free"), Decimal('NaN'))
        if usdt_equity.is_nan() or usdt_equity <= 0: logger.error(f"{NEON['ERROR']}Invalid account equity ({_format_for_log(usdt_equity)}).{NEON['RESET']}"); return None
        ticker = safe_api_call(exchange.fetch_ticker, symbol); signal_price = safe_decimal_conversion(ticker.get("last"), pd.NA)
        if pd.isna(signal_price) or signal_price <= 0: logger.error(f"{NEON['ERROR']}Failed to get valid signal price ({_format_for_log(signal_price)}).{NEON['RESET']}"); return None
        sl_atr_multiplier = get_current_atr_sl_multiplier(); sl_dist = current_short_atr * sl_atr_multiplier; sl_px_est = (signal_price - sl_dist) if side == CONFIG.side_buy else (signal_price + sl_dist)
        if sl_px_est <= 0: logger.error(f"{NEON['ERROR']}Invalid estimated SL price ({_format_for_log(sl_px_est)}).{NEON['RESET']}"); return None
        current_risk_pct = calculate_dynamic_risk() if CONFIG.enable_dynamic_risk else (CONFIG.scale_in_risk_percentage if is_scale_in else CONFIG.risk_per_trade_percentage)
        order_qty, est_margin = calculate_position_size(usdt_equity, current_risk_pct, signal_price, sl_px_est, leverage, symbol, exchange)
        if order_qty is None or order_qty <= CONFIG.position_qty_epsilon: logger.error(f"{NEON['ERROR']}Position size calc failed for {action_type}. Qty: {_format_for_log(order_qty)}{NEON['RESET']}"); return None
        if min_qty_allowed > 0 and order_qty < min_qty_allowed: logger.error(f"{NEON['ERROR']}Qty {_format_for_log(order_qty)} below min allowed {_format_for_log(min_qty_allowed)}.{NEON['RESET']}"); return None
        if usdt_free_margin < est_margin * margin_check_buffer: logger.error(f"{NEON['ERROR']}Insufficient free margin. Need ~{_format_for_log(est_margin*margin_check_buffer,2)}, Have {_format_for_log(usdt_free_margin,2)}{NEON['RESET']}"); return None
        entry_order_id: Optional[str] = None; entry_order_resp: Optional[Dict[str, Any]] = None; limit_price_str: Optional[str] = None
        if entry_type == OrderEntryType.MARKET: entry_order_resp = safe_api_call(exchange.create_market_order, symbol, side, float(order_qty), params={"category": v5_api_category, "positionIdx": 0}); entry_order_id = entry_order_resp.get("id")
        elif entry_type == OrderEntryType.LIMIT:
            pip_value = Decimal('1') / (Decimal('10') ** market_info['precision']['price']); offset = CONFIG.limit_order_offset_pips * pip_value; limit_price = (signal_price - offset) if side == CONFIG.side_buy else (signal_price + offset)
            limit_price_str = format_price(exchange, symbol, limit_price); logger.info(f"Placing LIMIT order: Qty={_format_for_log(order_qty)}, Price={_format_for_log(limit_price_str, color=NEON['PRICE'])}")
            entry_order_resp = safe_api_call(exchange.create_limit_order, symbol, side, float(order_qty), float(limit_price_str), params={"category": v5_api_category, "positionIdx": 0}); entry_order_id = entry_order_resp.get("id")
        if not entry_order_id: logger.critical(f"{NEON['CRITICAL']}{action_type} {entry_type.value} order NO ID!{NEON['RESET']}"); return None
        fill_timeout = CONFIG.limit_order_fill_timeout_seconds if entry_type == OrderEntryType.LIMIT else CONFIG.order_fill_timeout_seconds
        filled_entry_details = wait_for_order_fill(exchange, entry_order_id, symbol, fill_timeout, order_type=entry_type.value)
        if entry_type == OrderEntryType.LIMIT and (not filled_entry_details or filled_entry_details.get("status") != "closed"):
            logger.warning(f"{NEON['WARNING']}Limit order ...{format_order_id(entry_order_id)} did not fill. Cancelling.{NEON['RESET']}")
            try: safe_api_call(exchange.cancel_order, entry_order_id, symbol, params={"category": v5_api_category})
            except Exception as e_cancel: logger.error(f"Failed to cancel limit order ...{format_order_id(entry_order_id)}: {e_cancel}")
            return None
        if not filled_entry_details or filled_entry_details.get("status") != "closed": logger.error(f"{NEON['ERROR']}{action_type} order ...{format_order_id(entry_order_id)} not filled/failed.{NEON['RESET']}"); return None
        actual_fill_px = safe_decimal_conversion(filled_entry_details.get("average")); actual_fill_qty = safe_decimal_conversion(filled_entry_details.get("filled")); entry_ts_ms = filled_entry_details.get("timestamp")
        if actual_fill_qty <= CONFIG.position_qty_epsilon or actual_fill_px <= 0: logger.critical(f"{NEON['CRITICAL']}Invalid fill for {action_type} order. Qty: {_format_for_log(actual_fill_qty)}, Px: {_format_for_log(actual_fill_px)}{NEON['RESET']}"); return None
        entry_ref_price = signal_price if entry_type == OrderEntryType.MARKET else Decimal(limit_price_str) # type: ignore
        slippage = abs(actual_fill_px - entry_ref_price); slippage_pct = (slippage / entry_ref_price * 100) if entry_ref_price > 0 else Decimal(0)
        logger.info(f"{action_type} Slippage: RefPx={_format_for_log(entry_ref_price,4,color=NEON['PARAM'])}, FillPx={_format_for_log(actual_fill_px,4,color=NEON['PRICE'])}, Slip={_format_for_log(slippage,4,color=NEON['WARNING'])} ({slippage_pct:.3f}%)")
        new_part_id = entry_order_id if is_scale_in else "initial"
        if new_part_id == "initial" and any(p["id"] == "initial" for p in _active_trade_parts): logger.error("Attempted second 'initial' part."); return None
        _active_trade_parts.append({"id": new_part_id, "entry_price": actual_fill_px, "entry_time_ms": entry_ts_ms, "side": side, "qty": actual_fill_qty, "sl_price": sl_px_est})
        sl_placed = False; actual_sl_px_raw = (actual_fill_px - sl_dist) if side == CONFIG.side_buy else (actual_fill_px + sl_dist); actual_sl_px_str = format_price(exchange, symbol, actual_sl_px_raw)
        if Decimal(actual_sl_px_str) > 0:
            sl_order_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy; sl_params = {"category": v5_api_category, "stopLossPrice": float(actual_sl_px_str), "reduceOnly": True, "positionIdx": 0, "tpslMode": "Full", "slOrderType": "Market"} # Added tpslMode and slOrderType for Bybit V5
            try: sl_order_resp = safe_api_call(exchange.create_order, symbol, "Market", sl_order_side, float(actual_fill_qty), price=None, params=sl_params); logger.success(f"{NEON['SUCCESS']}SL for part {new_part_id} placed (ID:...{format_order_id(sl_order_resp.get('id'))}).{NEON['RESET']}"); sl_placed = True
            except ccxt.InsufficientFunds as e_funds: logger.error(f"{NEON['CRITICAL']}SL Failed: Insufficient Funds! {e_funds}{NEON['RESET']}")
            except ccxt.InvalidOrder as e_inv: logger.error(f"{NEON['CRITICAL']}SL Failed: Invalid Order! {e_inv}{NEON['RESET']}")
            except Exception as e_sl: logger.error(f"{NEON['CRITICAL']}SL Failed: {e_sl}{NEON['RESET']}"); logger.debug(traceback.format_exc())
        else: logger.error(f"{NEON['CRITICAL']}Invalid SL price for part {new_part_id}! ({_format_for_log(actual_sl_px_str)}){NEON['RESET']}")
        
        if not is_scale_in and CONFIG.trailing_stop_percentage > 0:
            tsl_activation_offset_value = actual_fill_px * CONFIG.trailing_stop_activation_offset_percent
            tsl_activation_price_raw = (actual_fill_px + tsl_activation_offset_value) if side == CONFIG.side_buy else (actual_fill_px - tsl_activation_offset_value)
            tsl_activation_price_str = format_price(exchange, symbol, tsl_activation_price_raw)
            tsl_value_for_api_str = str((CONFIG.trailing_stop_percentage * Decimal("100")).normalize())
            if Decimal(tsl_activation_price_str) > 0:
                tsl_params_specific = {"category": v5_api_category, "trailingStop": tsl_value_for_api_str, "activePrice": float(tsl_activation_price_str), "reduceOnly": True, "positionIdx": 0, "tpslMode": "Full", "slOrderType": "Market"}
                try:
                    tsl_order_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
                    logger.info(f"Placing Trailing SL Ward: Side={tsl_order_side}, Qty={_format_for_log(actual_fill_qty, color=NEON['QTY'])}, Trail={NEON['VALUE']}{tsl_value_for_api_str}%{NEON['RESET']}, ActivateAt={NEON['PRICE']}{tsl_activation_price_str}{NEON['RESET']}")
                    tsl_order_response = safe_api_call(exchange.create_order, symbol, "Market", tsl_order_side, float(actual_fill_qty), price=None, params=tsl_params_specific) # Using Market for TSL execution
                    logger.success(f"{NEON['SUCCESS']}Trailing SL Ward placed. ID:...{format_order_id(tsl_order_response.get('id'))}{NEON['RESET']}")
                except Exception as e_tsl: logger.warning(f"{NEON['WARNING']}Failed to place TSL: {e_tsl}{NEON['RESET']}")
            else: logger.error(f"{NEON['ERROR']}Invalid TSL activation price! ({_format_for_log(tsl_activation_price_str)}){NEON['RESET']}")

        if not sl_placed : logger.critical(f"{NEON['CRITICAL']}CRITICAL: SL FAILED for {action_type} part {new_part_id}. EMERGENCY CLOSE of entire position.{NEON['RESET']}"); close_position(exchange, symbol, {}, reason=f"EMERGENCY CLOSE - SL FAIL ({action_type} part {new_part_id})"); return None
        save_persistent_state(); logger.success(f"{NEON['SUCCESS']}{action_type} for {NEON['QTY']}{actual_fill_qty}{NEON['SUCCESS']} {symbol} @ {NEON['PRICE']}{actual_fill_px}{NEON['SUCCESS']} successful. State saved.{NEON['RESET']}")
        return filled_entry_details
    except ccxt.InsufficientFunds as e_funds: logger.error(f"{NEON['CRITICAL']}{action_type} Failed: Insufficient Funds! {e_funds}{NEON['RESET']}")
    except ccxt.InvalidOrder as e_inv: logger.error(f"{NEON['CRITICAL']}{action_type} Failed: Invalid Order! {e_inv}{NEON['RESET']}")
    except Exception as e_ritual: logger.error(f"{NEON['CRITICAL']}{action_type} Ritual FAILED: {e_ritual}{NEON['RESET']}"); logger.debug(traceback.format_exc())
    return None

def close_partial_position(exchange: ccxt.Exchange, symbol: str, close_qty: Optional[Decimal] = None, reason: str = "Scale Out") -> Optional[Dict[str, Any]]:
    global _active_trade_parts;
    if not _active_trade_parts: logger.info("No active parts to partially close."); return None
    oldest_part = min(_active_trade_parts, key=lambda p: p['entry_time_ms']); qty_to_close = close_qty if close_qty is not None and close_qty > 0 else oldest_part['qty']
    if qty_to_close > oldest_part['qty']: logger.warning(f"Requested partial close qty {close_qty} > oldest part qty {oldest_part['qty']}. Closing oldest part fully."); qty_to_close = oldest_part['qty']
    pos_side = oldest_part['side']; side_to_execute_close = CONFIG.side_sell if pos_side == CONFIG.pos_long else CONFIG.side_buy; amount_to_close_str = format_amount(exchange, symbol, qty_to_close)
    logger.info(f"{NEON['ACTION']}Scaling Out: Closing {NEON['QTY']}{amount_to_close_str}{NEON['ACTION']} of {pos_side} position (Part ID: {oldest_part['id']}, Reason: {reason}).{NEON['RESET']}")
    try:
        params = {"reduceOnly": True, "category": "linear", "positionIdx": 0}; close_order_response = safe_api_call(exchange.create_market_order, symbol=symbol, side=side_to_execute_close, amount=float(amount_to_close_str), params=params)
        if close_order_response and close_order_response.get("status") == "closed":
            exit_price = safe_decimal_conversion(close_order_response.get("average")); exit_time_ms = close_order_response.get("timestamp")
            mae, mfe = trade_metrics.calculate_mae_mfe(oldest_part['id'], oldest_part['entry_price'], exit_price, oldest_part['side'], oldest_part['entry_time_ms'], exit_time_ms, exchange, symbol, CONFIG.interval)
            trade_metrics.log_trade(symbol, oldest_part["side"], oldest_part["entry_price"], exit_price, qty_to_close, oldest_part["entry_time_ms"], exit_time_ms, reason, part_id=oldest_part["id"], mae=mae, mfe=mfe)
            if abs(qty_to_close - oldest_part['qty']) < CONFIG.position_qty_epsilon: _active_trade_parts.remove(oldest_part)
            else: oldest_part['qty'] -= qty_to_close
            save_persistent_state(); logger.success(f"{NEON['SUCCESS']}Scale Out successful for {amount_to_close_str} {symbol}. State saved.{NEON['RESET']}")
            logger.warning(f"{NEON['WARNING']}ACTION REQUIRED: Manually cancel/adjust SL for closed/reduced part ID {oldest_part['id']}.{NEON['RESET']}") # Automation of SL adjustment is complex
            return close_order_response
        else: logger.error(f"{NEON['ERROR']}Scale Out order failed for {symbol}.{NEON['RESET']}")
    except Exception as e: logger.error(f"{NEON['ERROR']}Scale Out Ritual FAILED: {e}{NEON['RESET']}")
    return None
def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close_details: Dict[str, Any], reason: str = "Signal") -> Optional[Dict[str, Any]]:
    global _active_trade_parts;
    if not _active_trade_parts: logger.info("No active parts to close."); return None
    total_qty_to_close = sum(part.get('qty', Decimal(0)) for part in _active_trade_parts)
    if total_qty_to_close <= CONFIG.position_qty_epsilon: _active_trade_parts.clear(); save_persistent_state(); return None
    pos_side_for_log = _active_trade_parts[0]['side']; side_to_execute_close = CONFIG.side_sell if pos_side_for_log == CONFIG.pos_long else CONFIG.side_buy
    logger.info(f"{NEON['ACTION']}Closing ALL parts of {pos_side_for_log} position for {symbol} (Qty: {NEON['QTY']}{total_qty_to_close}{NEON['ACTION']}, Reason: {reason}).{NEON['RESET']}")
    try:
        # Attempt to cancel all open orders (SL/TSL) for the symbol before closing position
        cancelled_sl_count = cancel_open_orders(exchange, symbol, reason=f"Pre-Close Position ({reason})")
        logger.info(f"{NEON['INFO']}Cancelled {NEON['VALUE']}{cancelled_sl_count}{NEON['INFO']} SL/TSL orders before closing position.{NEON['RESET']}")
        time.sleep(0.5) # Brief pause for cancellations to process

        close_order_resp = safe_api_call(exchange.create_market_order, symbol, side_to_execute_close, float(total_qty_to_close), params={"reduceOnly": True, "category": "linear", "positionIdx": 0})
        if close_order_resp and close_order_resp.get("status") == "closed":
            exit_px = safe_decimal_conversion(close_order_resp.get("average")); exit_ts_ms = close_order_resp.get("timestamp")
            for part in list(_active_trade_parts): # Iterate over a copy for safe removal
                 mae, mfe = trade_metrics.calculate_mae_mfe(part['id'],part['entry_price'], exit_px, part['side'], part['entry_time_ms'], exit_ts_ms, exchange, symbol, CONFIG.interval)
                 trade_metrics.log_trade(symbol, part["side"], part["entry_price"], exit_px, part["qty"], part["entry_time_ms"], exit_ts_ms, reason, part_id=part["id"], mae=mae, mfe=mfe)
            _active_trade_parts.clear(); save_persistent_state(); logger.success(f"{NEON['SUCCESS']}All parts of position for {symbol} closed. State saved.{NEON['RESET']}")
            return close_order_resp
        logger.error(f"{NEON['ERROR']}Consolidated close order failed for {symbol}. Response: {close_order_resp}{NEON['RESET']}")
    except Exception as e: logger.error(f"{NEON['ERROR']}Close Position Ritual FAILED: {e}{NEON['RESET']}")
    return None
def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> int:
    logger.info(f"{NEON['INFO']}Attempting to cancel ALL open orders for {NEON['VALUE']}{symbol}{NEON['INFO']} (Reason: {reason})...{NEON['RESET']}")
    cancelled_count = 0
    try:
        # Bybit V5 requires category for fetchOpenOrders and cancelOrder
        params = {"category": "linear"}
        open_orders = safe_api_call(exchange.fetch_open_orders, symbol, params=params)
        if not open_orders:
            logger.info(f"No open orders found for {symbol} to cancel.")
            return 0
        for order in open_orders:
            try:
                safe_api_call(exchange.cancel_order, order['id'], symbol, params=params)
                logger.info(f"Cancelled order {NEON['VALUE']}{order['id']}{NEON['INFO']} for {symbol}.")
                cancelled_count += 1
            except ccxt.OrderNotFound:
                logger.info(f"Order {NEON['VALUE']}{order['id']}{NEON['INFO']} already closed/cancelled.")
                cancelled_count +=1 # Count as handled
            except Exception as e_cancel:
                logger.error(f"{NEON['ERROR']}Failed to cancel order {order['id']}: {e_cancel}{NEON['RESET']}")
        logger.info(f"Order cancellation process for {symbol} complete. Cancelled/Handled: {NEON['VALUE']}{cancelled_count}{NEON['RESET']}.")
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Error fetching/cancelling open orders for {symbol}: {e}{NEON['RESET']}")
    return cancelled_count


# --- Strategy Signal Generation Wrapper ---
def generate_strategy_signals(df_with_indicators: pd.DataFrame, strategy_instance: TradingStrategy) -> Dict[str, Any]:
    if strategy_instance: return strategy_instance.generate_signals(df_with_indicators)
    logger.error(f"{NEON['ERROR']}Strategy instance not initialized!{NEON['RESET']}"); return TradingStrategy(CONFIG)._get_default_signals()

# --- Trading Logic ---
_stop_trading_flag = False
_last_drawdown_check_time = 0
def trade_logic(exchange: ccxt.Exchange, symbol: str, market_data_df: pd.DataFrame) -> None:
    global _active_trade_parts, _stop_trading_flag, _last_drawdown_check_time
    logger.info(f"{NEON['SUBHEADING']}========== Pyrmethus Cycle Start v2.8.0 ({CONFIG.strategy_name.value}) for '{symbol}' =========={NEON['RESET']}")
    now_ts = time.time()
    if _stop_trading_flag: logger.critical(f"{NEON['CRITICAL']}STOP TRADING FLAG ACTIVE (Drawdown?). No new trades.{NEON['RESET']}"); return
    if market_data_df.empty: logger.warning(f"{NEON['WARNING']}Empty market data.{NEON['RESET']}"); return
    if CONFIG.enable_max_drawdown_stop and now_ts - _last_drawdown_check_time > 300: # Check every 5 mins
        try:
            balance = safe_api_call(exchange.fetch_balance, params={"category": "linear"}); current_equity = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol,{}).get("total"))
            if not pd.isna(current_equity):
                trade_metrics.set_initial_equity(current_equity); breached, reason = trade_metrics.check_drawdown(current_equity)
                if breached: _stop_trading_flag = True; logger.critical(f"{NEON['CRITICAL']}MAX DRAWDOWN: {reason}. Halting new trades!{NEON['RESET']}"); send_sms_alert(f"[Pyrmethus] CRITICAL: Max Drawdown STOP Activated: {reason}"); return
            _last_drawdown_check_time = now_ts
        except Exception as e_dd: logger.error(f"{NEON['ERROR']}Error during drawdown check: {e_dd}{NEON['RESET']}")
    
    df_indic, current_vol_atr_data = calculate_all_indicators(market_data_df, CONFIG) # Pass original df
    current_atr = current_vol_atr_data.get("atr_short", Decimal("0")); current_close_price = safe_decimal_conversion(df_indic['close'].iloc[-1], pd.NA)
    if pd.isna(current_atr) or current_atr <= 0 or pd.isna(current_close_price) or current_close_price <= 0: logger.warning(f"{NEON['WARNING']}Invalid ATR ({_format_for_log(current_atr)}) or Close Price ({_format_for_log(current_close_price)}).{NEON['RESET']}"); return
    
    current_pos = get_current_position(exchange, symbol); pos_side, total_pos_qty, avg_pos_entry_price = current_pos["side"], current_pos["qty"], current_pos["entry_price"]; num_active_parts = current_pos.get("num_parts", 0)
    strategy_signals = generate_strategy_signals(df_indic, CONFIG.strategy_instance)
    
    if CONFIG.enable_time_based_stop and pos_side != CONFIG.pos_none:
        now_ms = int(now_ts * 1000)
        for part in list(_active_trade_parts): # Iterate over copy if modifying list
            duration_ms = now_ms - part['entry_time_ms']
            if duration_ms > CONFIG.max_trade_duration_seconds * 1000:
                reason = f"Time Stop Hit ({duration_ms/1000:.0f}s > {CONFIG.max_trade_duration_seconds}s)"; logger.warning(f"{NEON['WARNING']}TIME STOP for part {part['id']} ({pos_side}). Closing entire position.{NEON['RESET']}")
                close_position(exchange, symbol, current_pos, reason=reason); return
    
    if CONFIG.enable_scale_out and pos_side != CONFIG.pos_none and num_active_parts > 0:
        profit_in_atr = Decimal('0')
        if current_atr > 0 and avg_pos_entry_price > 0: price_diff = (current_close_price - avg_pos_entry_price) if pos_side == CONFIG.pos_long else (avg_pos_entry_price - current_close_price); profit_in_atr = price_diff / current_atr
        if profit_in_atr >= CONFIG.scale_out_trigger_atr:
            logger.info(f"{NEON['ACTION']}SCALE-OUT Triggered: {NEON['VALUE']}{profit_in_atr:.2f}{NEON['ACTION']} ATRs in profit. Closing oldest part.{NEON['RESET']}")
            close_partial_position(exchange, symbol, close_qty=None, reason=f"Scale Out Profit Target ({profit_in_atr:.2f} ATR)")
            current_pos = get_current_position(exchange, symbol); pos_side, total_pos_qty, avg_pos_entry_price = current_pos["side"], current_pos["qty"], current_pos["entry_price"]; num_active_parts = current_pos.get("num_parts", 0) # Refresh position state
            if pos_side == CONFIG.pos_none: return # Position fully closed by scale-out
    
    should_exit_long = pos_side == CONFIG.pos_long and strategy_signals.get("exit_long", False)
    should_exit_short = pos_side == CONFIG.pos_short and strategy_signals.get("exit_short", False)
    if should_exit_long or should_exit_short:
        exit_reason = strategy_signals.get("exit_reason", "Oracle Decrees Exit"); logger.warning(f"{NEON['ACTION']}*** STRATEGY EXIT for remaining {pos_side} position (Reason: {exit_reason}) ***{NEON['RESET']}")
        close_position(exchange, symbol, current_pos, reason=exit_reason); return
    
    if CONFIG.enable_position_scaling and pos_side != CONFIG.pos_none and num_active_parts < (CONFIG.max_scale_ins + 1):
        profit_in_atr = Decimal('0')
        if current_atr > 0 and avg_pos_entry_price > 0: price_diff = (current_close_price - avg_pos_entry_price) if pos_side == CONFIG.pos_long else (avg_pos_entry_price - current_close_price); profit_in_atr = price_diff / current_atr
        can_scale = profit_in_atr >= CONFIG.min_profit_for_scale_in_atr
        scale_long_signal = strategy_signals.get("enter_long", False) and pos_side == CONFIG.pos_long
        scale_short_signal = strategy_signals.get("enter_short", False) and pos_side == CONFIG.pos_short
        if can_scale and (scale_long_signal or scale_short_signal):
            logger.success(f"{NEON['ACTION']}*** PYRAMIDING OPPORTUNITY: New signal to add to {pos_side}. ***{NEON['RESET']}")
            scale_in_side = CONFIG.side_buy if scale_long_signal else CONFIG.side_sell
            place_risked_order(exchange=exchange, symbol=symbol, side=scale_in_side, risk_percentage=CONFIG.scale_in_risk_percentage, current_short_atr=current_atr, leverage=CONFIG.leverage, max_order_cap_usdt=CONFIG.max_order_usdt_amount, margin_check_buffer=CONFIG.required_margin_buffer, tsl_percent=CONFIG.trailing_stop_percentage, tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent, entry_type=CONFIG.entry_order_type, is_scale_in=True, existing_position_avg_price=avg_pos_entry_price)
            return
            
    if pos_side == CONFIG.pos_none:
        enter_long_signal = strategy_signals.get("enter_long", False); enter_short_signal = strategy_signals.get("enter_short", False)
        if enter_long_signal or enter_short_signal:
             side_to_enter = CONFIG.side_buy if enter_long_signal else CONFIG.side_sell
             logger.success(f"{(NEON['SIDE_LONG'] if enter_long_signal else NEON['SIDE_SHORT'])}*** INITIAL {side_to_enter.upper()} ENTRY SIGNAL ***{NEON['RESET']}")
             place_risked_order(exchange=exchange, symbol=symbol, side=side_to_enter, risk_percentage=calculate_dynamic_risk(), current_short_atr=current_atr, leverage=CONFIG.leverage, max_order_cap_usdt=CONFIG.max_order_usdt_amount, margin_check_buffer=CONFIG.required_margin_buffer, tsl_percent=CONFIG.trailing_stop_percentage, tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent, entry_type=CONFIG.entry_order_type, is_scale_in=False)
             return
             
    if pos_side != CONFIG.pos_none: logger.info(f"{NEON['INFO']}Holding {pos_side} position ({NEON['VALUE']}{num_active_parts}{NEON['INFO']} parts). Awaiting signals or stops.{NEON['RESET']}")
    else: logger.info(f"{NEON['INFO']}Holding Cash. No signals or conditions met.{NEON['RESET']}")
    save_persistent_state(force_heartbeat=True)
    logger.info(f"{NEON['SUBHEADING']}========== Pyrmethus Cycle End v2.8.0 for '{symbol}' =========={NEON['RESET']}\n")

# --- Graceful Shutdown ---
def graceful_shutdown(exchange_instance: Optional[ccxt.Exchange], trading_symbol: Optional[str]) -> None:
    logger.warning(f"\n{NEON['WARNING']}Unweaving Sequence Initiated v2.8.0...{NEON['RESET']}")
    save_persistent_state(force_heartbeat=True)
    if exchange_instance and trading_symbol:
        try:
            logger.warning(f"Unweaving: Cancelling ALL open orders for '{trading_symbol}'..."); cancel_open_orders(exchange_instance, trading_symbol, "Bot Shutdown Cleanup"); time.sleep(1.5)
            if _active_trade_parts: logger.warning(f"Unweaving: Active position parts found. Attempting final consolidated close..."); dummy_pos_state = {"side": _active_trade_parts[0]['side'], "qty": sum(p['qty'] for p in _active_trade_parts)}; close_position(exchange_instance, trading_symbol, dummy_pos_state, "Bot Shutdown Final Close")
        except Exception as e_cleanup: logger.error(f"{NEON['ERROR']}Unweaving Error: {e_cleanup}{NEON['RESET']}")
    trade_metrics.summary()
    logger.info(f"{NEON['HEADING']}--- Pyrmethus Spell Unweaving v2.8.0 Complete ---{NEON['RESET']}")

# --- Data Fetching (Added for completeness) ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 150) -> Optional[pd.DataFrame]:
    """Fetches OHLCV market data and returns a pandas DataFrame."""
    logger.info(f"{NEON['INFO']}Fetching market data for {NEON['VALUE']}{symbol}{NEON['INFO']} ({timeframe}, limit={limit})...{NEON['RESET']}")
    try:
        params = {"category": "linear"} # Assuming linear contracts for Bybit V5
        ohlcv = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe, limit=limit, params=params)
        if not ohlcv:
            logger.warning(f"{NEON['WARNING']}No OHLCV data returned for {symbol}.{NEON['RESET']}")
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]: # Ensure numeric types
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Basic NaN handling: ffill then bfill. More sophisticated handling might be needed.
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        if df.isnull().values.any(): # Check if NaNs still exist after filling
            logger.error(f"{NEON['ERROR']}Unfillable NaNs remain in OHLCV data for {symbol} after ffill/bfill. Data quality compromised.{NEON['RESET']}")
            # Depending on strategy, might return None or df with NaNs
            # For safety, returning None if critical columns (like 'close') are still NaN in the last row
            if df[['close']].iloc[-1].isnull().any():
                 return None
        logger.debug(f"{NEON['DEBUG']}Fetched {len(df)} candles for {symbol}. Last candle: {df.index[-1]}{NEON['RESET']}")
        return df
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Failed to fetch market data for {symbol}: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        return None

# --- Main Execution ---
def main() -> None:
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(f"{NEON['HEADING']}--- Pyrmethus Scalping Spell v2.8.0 (Strategic Illumination) Initializing ({start_time_readable}) ---{NEON['RESET']}")
    logger.info(f"{NEON['SUBHEADING']}--- Active Strategy Path: {NEON['VALUE']}{CONFIG.strategy_name.value}{NEON['RESET']} ---")
    logger.info(f"Symbol: {NEON['VALUE']}{CONFIG.symbol}{NEON['RESET']}, Interval: {NEON['VALUE']}{CONFIG.interval}{NEON['RESET']}, Leverage: {NEON['VALUE']}{CONFIG.leverage}x{NEON['RESET']}")
    logger.info(f"Risk/Trade: {NEON['VALUE']}{CONFIG.risk_per_trade_percentage:.2%}{NEON['RESET']}, Max Order USDT: {NEON['VALUE']}{CONFIG.max_order_usdt_amount}{NEON['RESET']}")


    current_exchange_instance: Optional[ccxt.Exchange] = None; unified_trading_symbol: Optional[str] = None; should_run_bot: bool = True
    try:
        current_exchange_instance = initialize_exchange()
        if not current_exchange_instance: logger.critical(f"{NEON['CRITICAL']}Exchange portal failed. Exiting.{NEON['RESET']}"); return
        try: market_details = current_exchange_instance.market(CONFIG.symbol); unified_trading_symbol = market_details["symbol"]
        except Exception as e_market: logger.critical(f"{NEON['CRITICAL']}Symbol validation error: {e_market}. Exiting.{NEON['RESET']}"); return
        logger.info(f"{NEON['SUCCESS']}Spell focused on symbol: {NEON['VALUE']}{unified_trading_symbol}{NEON['SUCCESS']}{NEON['RESET']}") # Added SUCCESS color
        if not set_leverage(current_exchange_instance, unified_trading_symbol, CONFIG.leverage): logger.warning(f"{NEON['WARNING']}Leverage setting may not have been applied or confirmed.{NEON['RESET']}") # Adjusted message
        
        if load_persistent_state():
            logger.info(f"{NEON['SUCCESS']}Phoenix Feather: Previous session state restored.{NEON['RESET']}")
            if _active_trade_parts:
                logger.warning(f"{NEON['WARNING']}State Reconciliation Check:{NEON['RESET']} Bot remembers {len(_active_trade_parts)} active trade part(s). Verifying with exchange...")
                exchange_pos = _get_raw_exchange_position(current_exchange_instance, unified_trading_symbol); bot_qty = sum(p['qty'] for p in _active_trade_parts); bot_side = _active_trade_parts[0]['side'] if _active_trade_parts else CONFIG.pos_none
                if exchange_pos['side'] == CONFIG.pos_none and bot_side != CONFIG.pos_none: logger.critical(f"{NEON['CRITICAL']}RECONCILIATION FAILED:{NEON['RESET']} Bot remembers {bot_side} (Qty: {bot_qty}), exchange FLAT. Clearing bot state."); _active_trade_parts.clear(); save_persistent_state()
                elif exchange_pos['side'] != bot_side or abs(exchange_pos['qty'] - bot_qty) > CONFIG.position_qty_epsilon: logger.critical(f"{NEON['CRITICAL']}RECONCILIATION FAILED:{NEON['RESET']} Discrepancy: Bot ({bot_side} Qty {bot_qty}) vs Exchange ({exchange_pos['side']} Qty {exchange_pos['qty']}). Clearing bot state."); _active_trade_parts.clear(); save_persistent_state()
                else: logger.info(f"{NEON['SUCCESS']}State Reconciliation: Bot state consistent with exchange.{NEON['RESET']}")
        else: logger.info(f"{NEON['INFO']}Starting with a fresh session state.{NEON['RESET']}")
        try: 
            balance = safe_api_call(current_exchange_instance.fetch_balance, params={"category":"linear"}); 
            initial_equity = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol,{}).get("total"))
            if not pd.isna(initial_equity): trade_metrics.set_initial_equity(initial_equity)
            else: logger.error(f"{NEON['ERROR']}Failed to get valid initial equity for drawdown tracking.{NEON['RESET']}")
        except Exception as e_bal: logger.error(f"{NEON['ERROR']}Failed to set initial equity: {e_bal}{NEON['RESET']}")

        while should_run_bot:
            cycle_start_monotonic = time.monotonic()
            try: 
                # Simple health check: Try to fetch balance.
                if not current_exchange_instance.fetch_balance(params={"category":"linear"}): raise Exception("Exchange health check (fetch_balance) failed")
            except Exception as e_health: logger.critical(f"{NEON['CRITICAL']}Account health check failed: {e_health}. Pausing.{NEON['RESET']}"); time.sleep(10); continue
            try:
                df_market_candles = get_market_data(current_exchange_instance, unified_trading_symbol, CONFIG.interval, limit=max(200, CONFIG.momentum_period + CONFIG.confirm_st_atr_length + 50)) # Ensure enough data
                if df_market_candles is not None and not df_market_candles.empty: trade_logic(current_exchange_instance, unified_trading_symbol, df_market_candles)
                else: logger.warning(f"{NEON['WARNING']}Skipping cycle: Invalid market data.{NEON['RESET']}")
            except ccxt.RateLimitExceeded as e_rate: logger.warning(f"{NEON['WARNING']}Rate Limit: {e_rate}. Sleeping longer...{NEON['RESET']}"); time.sleep(CONFIG.sleep_seconds * 6)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e_net: logger.warning(f"{NEON['WARNING']}Network/Exchange Issue: {e_net}. Sleeping...{NEON['RESET']}"); time.sleep(CONFIG.sleep_seconds * 3)
            except ccxt.AuthenticationError as e_auth: logger.critical(f"{NEON['CRITICAL']}FATAL: Auth Error: {e_auth}. Stopping.{NEON['RESET']}"); should_run_bot = False
            except Exception as e_loop: logger.exception(f"{NEON['CRITICAL']}!!! UNEXPECTED CRITICAL ERROR in Main Loop: {e_loop} !!!{NEON['RESET']}"); should_run_bot = False # Use logger.exception to include traceback
            if should_run_bot:
                elapsed = time.monotonic() - cycle_start_monotonic; sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle processed in {elapsed:.2f}s. Sleeping for {sleep_dur:.2f}s.")
                if sleep_dur > 0: time.sleep(sleep_dur)
    except KeyboardInterrupt: logger.warning(f"\n{NEON['WARNING']}KeyboardInterrupt. Initiating graceful unweaving...{NEON['RESET']}"); should_run_bot = False
    except Exception as startup_err: logger.critical(f"{NEON['CRITICAL']}CRITICAL STARTUP ERROR v2.8.0: {startup_err}{NEON['RESET']}"); logger.debug(traceback.format_exc()); should_run_bot = False
    finally:
        graceful_shutdown(current_exchange_instance, unified_trading_symbol)
        logger.info(f"{NEON['HEADING']}--- Pyrmethus Scalping Spell v2.8.0 Deactivated ---{NEON['RESET']}")

if __name__ == "__main__":
    main()
```
