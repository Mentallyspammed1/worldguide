#
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

This script implements Pyrmethus, a high-frequency trading bot designed for scalping
on Bybit USDT Futures. It features a modular architecture, allowing for the selection
of different trading strategies, robust risk management, and various operational enhancements.
The bot is imbued with a "spellcasting" theme for its operations and logging.

Core Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS" (using Ehlers Super Smoother Filter).
- Modular strategy implementation using Abstract Base Class (ABC).
- Precision Handling: Employs `Decimal` for critical financial calculations. Indicator and market data values can be `Decimal`, `pd.NAType` (for pandas-specific missing data), or `None` to accurately represent their state.
- Fortified Configuration Loading: Correctly handles type casting for environment variables and default values, with clear, thematic logging.
- Exchange-native Trailing Stop Loss (TSL) placed immediately after entry (Bybit V5 API).
- Exchange-native fixed Stop Loss (based on ATR) placed immediately after entry (Bybit V5 API).
- Average True Range (ATR) for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation filters for trade entries.
- Risk-based position sizing with margin checks and configurable maximum position cap.
- Termux SMS alerts for critical events and trade actions (with Termux:API command check).
- Robust error handling (CCXT exceptions, validation) and detailed logging with vibrant Neon color support via Colorama, plus file logging. Retry mechanism for API calls.
- Graceful shutdown on KeyboardInterrupt or critical errors, attempting to close open positions and orders.
- Stricter position detection logic tailored for Bybit V5 API (One-Way Mode).
- OHLCV Data Caching: Reduces API calls by caching market data within a candle's duration, improving efficiency.
- Trade Metrics: Basic tracking of trade Profit/Loss (P/L) and win rate, chronicling each "battle."
- Account Health Check: Monitors margin ratio to prevent excessive risk and potential liquidation, acting as a "ward."
- NaN handling in fetched OHLCV data using forward and backward fill.
- Re-validation of position state before attempting to close a position.
- Enhanced Indicator Logging: Comprehensive output of key indicator values during each trading cycle's "scrying ritual."
- Corrected SuperTrend flip signal generation for improved DUAL_SUPERTREND strategy accuracy.
- Enhanced robustness in order placement: Attempts an emergency market close if essential stop-loss orders fail after entry.

Disclaimer:
- **EXTREME RISK**: Trading futures, especially with leverage and automation, is extremely risky. This script is provided for EDUCATIONAL PURPOSES ONLY. You can lose all your capital and potentially more. Use at your own absolute risk. No liability is assumed for any financial losses incurred.
- **EXCHANGE-NATIVE SL/TSL DEPENDENCE**: The bot relies entirely on Bybit's native Stop Loss (SL) and Trailing Stop Loss (TSL) order execution. Performance is subject to exchange conditions, potential slippage, API reliability, and order book liquidity. These orders are NOT guaranteed to execute at the exact trigger price.
- **PARAMETER SENSITIVITY**: Bot performance is highly sensitive to parameter tuning (strategy settings, risk parameters, SL/TSL percentages, confirmation filters). Requires significant backtesting and forward testing on a TESTNET environment before any consideration of live trading.
- **API RATE LIMITS**: Monitor API usage. Excessive requests can lead to temporary or permanent bans from the exchange. This script includes rate limiting via CCXT and a retry decorator, but careful configuration is still necessary.
- **SLIPPAGE**: Market orders, used for entry and potentially for SL/TSL execution, are prone to slippage, especially during volatile market conditions. This can affect entry and exit prices.
- **TEST THOROUGHLY**: **DO NOT RUN WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTNET/DEMO TESTING.** Understand every part of the code and its potential risks before considering live deployment.
- **TERMUX DEPENDENCY**: SMS alert functionality requires a Termux environment and the Termux:API package (`pkg install termux-api`). Ensure it's correctly installed and configured on your Termux device if using this feature.
- **API CHANGES**: This code targets the Bybit V5 API via the CCXT library. Future updates to the exchange API may break functionality. Keep CCXT updated (`pip install -U ccxt`) and monitor for breaking changes.
"""

# Standard Library Imports - The Foundational Runes
import logging
import os
import shutil  # For checking command existence (e.g., termux-sms-send)
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Union, Dict, List, Tuple, Optional

import pytz  # For timezone-aware datetimes

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
    from retry import retry  # For the safe_api_call decorator
except ImportError as e:
    missing_pkg = e.name
    sys.stderr.write(
        f"\033[91mCRITICAL ERROR: Missing required Python package: '{missing_pkg}'. Pyrmethus cannot awaken.\033[0m\n"
    )
    sys.stderr.write(
        f"\033[91mPlease install it by running: 'pip install {missing_pkg}'\033[0m\n"
    )
    if missing_pkg == "pandas_ta":
        sys.stderr.write(
            f"\033[91mFor pandas_ta, you might also need TA-Lib. Consult the scrolls (pandas_ta documentation) for installation instructions.\033[0m\n"
        )
    if missing_pkg == "retry":
        sys.stderr.write(
            f"\033[91mFor the retry decorator, install with: 'pip install retry'\033[0m\n"
        )
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)  # Initialize Colorama for vibrant console output
load_dotenv()  # Load arcane secrets from .env file
getcontext().prec = 18  # Set Decimal precision for all financial calculations


# --- Enums - Defining Arcane Categories ---
class StrategyName(str, Enum):
    """Enumeration of available trading strategies, or 'Paths of Magic'."""

    DUAL_SUPERTREND = "DUAL_SUPERTREND"
    STOCHRSI_MOMENTUM = "STOCHRSI_MOMENTUM"
    EHLERS_FISHER = "EHLERS_FISHER"
    EHLERS_MA_CROSS = "EHLERS_MA_CROSS"


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """
    Loads, validates, and stores configuration parameters (runes) from environment
    variables or specified defaults. Provides typed access to all bot settings.
    """

    def __init__(self) -> None:
        # A temporary logger for use during the configuration summoning phase,
        # before the main Oracle's Voice (logger) is fully attuned.
        _pre_logger = logging.getLogger(__name__ + ".ConfigSummoner")
        _pre_logger.info(
            f"{Fore.MAGENTA}--- Summoning Configuration Runes from the Environment Scroll (.env) ---{Style.RESET_ALL}"
        )
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: str = self._get_env(
            "BYBIT_API_KEY", required=True, color=Fore.RED, secret=True
        )
        self.api_secret: str = self._get_env(
            "BYBIT_API_SECRET", required=True, color=Fore.RED, secret=True
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
        self.valid_strategies: List[str] = [s.value for s in StrategyName]
        if self.strategy_name.value not in self.valid_strategies:
            _pre_logger.critical(
                f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME rune '{self.strategy_name.value}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}"
            )
            raise ValueError(
                f"Invalid STRATEGY_NAME '{self.strategy_name.value}'. Valid options are: {self.valid_strategies}"
            )
        _pre_logger.info(
            f"{Fore.CYAN}Chosen Path of Magic: {self.strategy_name.value}{Style.RESET_ALL}"
        )
        self.strategy_instance: "TradingStrategy"  # Forward declaration; will be set after CONFIG is fully loaded

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE",
            "0.01",
            cast_type=Decimal,
            color=Fore.GREEN,  # e.g., 1% of equity risked per trade
        )
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.3", cast_type=Decimal, color=Fore.GREEN
        )
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT",
            "50.0",
            cast_type=Decimal,
            color=Fore.GREEN,  # Max position size in USDT (value)
        )
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER",
            "1.05",
            cast_type=Decimal,
            color=Fore.GREEN,  # e.g., 5% buffer for margin checks
        )
        self.max_account_margin_ratio: Decimal = (
            self._get_env(  # For check_account_health ward
                "MAX_ACCOUNT_MARGIN_RATIO",
                "0.8",
                cast_type=Decimal,
                color=Fore.GREEN,  # e.g., 80% of equity used as margin
            )
        )

        # --- Trailing Stop Loss (Exchange Native - Bybit V5) - The Adaptive Shield ---
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE",
            "0.005",
            cast_type=Decimal,
            color=Fore.GREEN,  # e.g., 0.5% (0.005) trail distance
        )
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT",
            "0.001",  # e.g., 0.1% (0.001) move in profit before TSL activates
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
            "EHLERS_SSF_POLES",
            2,
            cast_type=int,
            color=Fore.CYAN,  # 2 or 3 poles are typical for SSF
        )

        # --- Confirmation Filters - Seeking Concordance in the Ether ---
        self.volume_ma_period: int = self._get_env(
            "VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW
        )
        self.volume_spike_threshold: Decimal = self._get_env(
            "VOLUME_SPIKE_THRESHOLD",
            "1.5",
            cast_type=Decimal,
            color=Fore.YELLOW,  # e.g., current volume > 1.5x its MA
        )
        self.require_volume_spike_for_entry: bool = self._get_env(
            "REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW
        )
        self.order_book_depth: int = self._get_env(
            "ORDER_BOOK_DEPTH",
            20,
            cast_type=int,
            color=Fore.YELLOW,  # Number of order book levels to sum volume from for ratio
        )
        self.order_book_ratio_threshold_long: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_LONG",
            "1.2",  # Bid volume must be >= 1.2 * Ask volume for long confirmation
            cast_type=Decimal,
            color=Fore.YELLOW,
        )
        self.order_book_ratio_threshold_short: Decimal = self._get_env(
            "ORDER_BOOK_RATIO_THRESHOLD_SHORT",
            "0.8",  # Bid volume must be <= 0.8 * Ask volume for short confirmation (or ask/bid ratio >= 1/0.8 = 1.25)
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
        self.sms_recipient_number: Optional[str] = (
            self._get_env(  # Corrected type (already correct in provided code)
                "SMS_RECIPIENT_NUMBER",
                None,
                cast_type=str,
                color=Fore.MAGENTA,  # Optional, defaults to None if not set
            )
        )
        self.sms_timeout_seconds: int = self._get_env(
            "SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA
        )

        # --- CCXT / API Parameters - Tuning the Connection ---
        self.default_recv_window: int = 13000  # Bybit max is 20000ms; default 5000ms can be too short for some connections
        self.order_book_fetch_limit: int = max(
            25, self.order_book_depth
        )  # How many levels to fetch from API for OB analysis
        self.shallow_ob_fetch_depth: int = (
            5  # For quick price estimation if needed (e.g., before placing an order)
        )
        self.order_fill_timeout_seconds: int = self._get_env(
            "ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW
        )

        # --- Internal Constants - Fixed Arcane Symbols ---
        self.side_buy: str = "buy"  # Standardized 'buy' side string
        self.side_sell: str = "sell"  # Standardized 'sell' side string
        self.pos_long: str = "Long"  # Standardized 'Long' position string
        self.pos_short: str = "Short"  # Standardized 'Short' position string
        self.pos_none: str = "None"  # Standardized 'No position' string
        self.usdt_symbol: str = "USDT"  # Standard symbol for USDT currency
        self.retry_count: int = 3  # Default retry attempts for API calls
        self.retry_delay_seconds: int = 2  # Initial delay for retries
        self.api_fetch_limit_buffer: int = (
            10  # Extra candles to fetch beyond minimum indicator requirements
        )
        self.position_qty_epsilon: Decimal = Decimal(
            "1e-9"
        )  # Small value for float comparisons with position sizes
        self.post_close_delay_seconds: int = (
            3  # Wait after closing a position before next potential action
        )
        self.cache_candle_duration_multiplier: Decimal = Decimal(
            "0.95"
        )  # For data cache validity, e.g., 95% of candle duration

        _pre_logger.info(
            f"{Fore.GREEN}{Style.BRIGHT}--- All Configuration Runes Summoned and Verified. The Spell is Ready to be Woven. ---{Style.RESET_ALL}"
        )

    def _get_env(
        self,
        key: str,
        default: Any = None,
        cast_type: type = str,
        required: bool = False,
        color: str = Fore.WHITE,
        secret: bool = False,  # New parameter to obfuscate secret values in logs
    ) -> Any:
        """
        Retrieves an environment variable, casts it to the specified type,
        and handles default values or requirement checks. Thematic logging included.
        If `secret` is True, the value is obfuscated in logs.
        """
        _pre_logger = logging.getLogger(
            __name__ + ".ConfigSummoner"
        )  # Use temp logger for config phase
        value_str = os.getenv(key)
        source = "Env Var"
        value_to_cast: Any = None
        display_value = "***SECRET***" if secret and value_str else value_str
        display_default = "***SECRET***" if secret and default is not None else default

        if value_str is None:
            if required:
                _pre_logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Essential rune '{key}' missing from the environment scroll (.env). Spell cannot be woven without it.{Style.RESET_ALL}"
                )
                raise ValueError(f"Required environment variable '{key}' not set.")
            _pre_logger.debug(
                f"{color}Rune '{key}': Undefined in environment. Conjuring default: '{display_default}'{Style.RESET_ALL}"
            )
            value_to_cast = default
            source = "Default"
        else:
            _pre_logger.debug(
                f"{color}Rune '{key}': Sourced from environment: '{display_value}'{Style.RESET_ALL}"
            )
            value_to_cast = value_str

        if value_to_cast is None:  # Catches if default was None and env var not set
            if required:  # Should have been caught above, but as a safeguard
                _pre_logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Required rune '{key}' resolved to None (env/default). Spellcrafting halted.{Style.RESET_ALL}"
                )
                raise ValueError(
                    f"Required environment variable '{key}' resolved to None."
                )
            else:  # Not required, value is None
                _pre_logger.debug(
                    f"{color}Rune '{key}': Final value is None (Type: NoneType, Source: {source}){Style.RESET_ALL}"
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
                final_value = raw_value_str
            else:  # Should not happen if cast_type is one of the above
                _pre_logger.warning(
                    f"Unsupported cast_type '{cast_type.__name__}' for rune '{key}'. Returning raw value."
                )
                final_value = value_to_cast

        except (ValueError, TypeError, InvalidOperation) as e:
            _pre_logger.error(
                f"{Fore.RED}Invalid type/value for rune '{key}': '{display_value if source == 'Env Var' else display_default}' (Source: {source}). Expected {cast_type.__name__}. Error: {e}. Attempting to use default '{display_default}'.{Style.RESET_ALL}"
            )
            if default is None:  # If default is also None, and casting failed
                if required:
                    _pre_logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed cast for required rune '{key}', and its default is None. Spellcrafting cannot proceed.{Style.RESET_ALL}"
                    )
                    raise ValueError(
                        f"Required env var '{key}' failed casting, and no valid default was provided."
                    )
                else:  # Not required, default is None, cast failed
                    _pre_logger.warning(
                        f"{Fore.YELLOW}Casting failed for rune '{key}', default is None. Final value: None{Style.RESET_ALL}"
                    )
                    return None  # Return None if default is None and casting failed
            else:  # Default is not None, try casting the default
                source = "Default (Fallback after Cast Error)"
                _pre_logger.debug(
                    f"Casting fallback default '{display_default}' for rune '{key}' to {cast_type.__name__}"
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
                        final_value = (
                            default  # Should not happen with supported cast_types
                        )
                    _pre_logger.warning(
                        f"{Fore.YELLOW}Used casted default for rune '{key}': '{display_value if secret else final_value}'{Style.RESET_ALL}"
                    )
                except (ValueError, TypeError, InvalidOperation) as e_default:
                    _pre_logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed cast for BOTH value ('{display_value if source == 'Default (Fallback after Cast Error)' else value_to_cast}') AND default ('{display_default}') for rune '{key}' to {cast_type.__name__}. Error on default: {e_default}. Spellcrafting halted.{Style.RESET_ALL}"
                    )
                    raise ValueError(
                        f"Configuration error: Cannot cast value or default for '{key}' to {cast_type.__name__}."
                    )
        _pre_logger.debug(
            f"{color}Rune '{key}': Using value '{display_value if secret else final_value}' (Type: {type(final_value).__name__}, Source: {source}){Style.RESET_ALL}"
        )
        return final_value


# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL: int = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)
# Create logs directory if it doesn't exist, the sacred ground for chronicles
if not os.path.exists("logs"):
    os.makedirs("logs")

log_file_name = f"logs/pyrmethus_chronicle_{time.strftime('%Y%m%d_%H%M%S')}.log"  # Timestamped log file for unique chronicles
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",  # Increased name width for better readability of sources
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Oracle's voice to the console
        logging.FileHandler(log_file_name),  # Oracle's voice to the chronicle file
    ],
)
logger: logging.Logger = logging.getLogger(
    "PyrmethusCore"
)  # Main logger for Pyrmethus's core operations

# Custom SUCCESS level and Neon Color Formatting for the Oracle's pronouncements
SUCCESS_LEVEL: int = 25  # Between INFO and WARNING, for positive, significant events
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logs a message with SUCCESS level, for triumphant spell outcomes."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)  # type: ignore[attr-defined]


logging.Logger.success = log_success  # type: ignore[attr-defined] # Add success method to Logger instances for easy use

if (
    sys.stdout.isatty()
):  # Apply colors only if output is a TTY (e.g., a terminal), not a pipe or file
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
    )  # Custom SUCCESS color
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
    CONFIG = Config()  # Instantiate the spell's configuration
except ValueError as config_error:
    # Use root logger if PyrmethusCore logger failed or not fully set up
    # This ensures critical configuration errors are always seen.
    logging.getLogger().critical(
        f"{Back.RED}{Fore.WHITE}Fatal Configuration Error: {config_error}. Pyrmethus cannot awaken.{Style.RESET_ALL}"
    )
    sys.exit(1)
except (
    Exception
) as general_config_error:  # Catch any other unexpected error during config
    logging.getLogger().critical(
        f"{Back.RED}{Fore.WHITE}Unexpected critical error during configuration summoning: {general_config_error}{Style.RESET_ALL}"
    )
    logging.getLogger().debug(
        traceback.format_exc()
    )  # Log stack trace for detailed debugging
    sys.exit(1)


# --- Trading Strategy Abstract Base Class & Implementations ---
class TradingStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Each strategy is a 'Path of Magic' that Pyrmethus can follow.
    Subclasses must implement `generate_signals` to define their unique spellcraft.
    """

    def __init__(self, config: Config, df_columns: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(
            f"Strategy.{self.__class__.__name__}"
        )  # Dedicated logger for each strategy path
        self.required_columns = (
            df_columns if df_columns else []
        )  # Columns needed from DataFrame for this strategy

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates trading signals based on the input DataFrame with indicator data.
        This is the core spell of the strategy, determining when to act.

        Args:
            df: Pandas DataFrame with OHLCV data and pre-calculated indicators.

        Returns:
            A dictionary containing boolean signals: "enter_long", "enter_short",
            "exit_long", "exit_short", and "exit_reason" (str).
        """
        pass

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        """
        Validates the input DataFrame for common issues before signal generation.
        Ensures the scrolls (data) are sufficient and legible.
        """
        if df is None or df.empty or len(df) < min_rows:
            self.logger.debug(
                f"Insufficient data for signal generation (Rows: {len(df) if df is not None else 0}, Min Required: {min_rows}). Spell cannot be cast."
            )
            return False
        if self.required_columns and not all(
            col in df.columns for col in self.required_columns
        ):
            missing_cols = [
                col for col in self.required_columns if col not in df.columns
            ]
            self.logger.warning(
                f"DataFrame is missing runes (required columns) for {self.__class__.__name__}: {missing_cols}. Spell may be weakened."
            )
            return False
        # Check for NaNs in the last row for required columns, which might indicate calculation issues or insufficient leading data.
        if self.required_columns and df.iloc[-1][self.required_columns].isnull().any():
            nan_cols_last_row_series = df.iloc[-1][self.required_columns].isnull()
            nan_cols_last_row = nan_cols_last_row_series[
                nan_cols_last_row_series
            ].index.tolist()
            self.logger.debug(
                f"Faded runes (NaN values) detected in the latest data for required columns: {nan_cols_last_row}. Strategy must interpret carefully."
            )
            # Strategies are expected to handle potential NaNs from indicators, often by not generating a signal.
        return True

    def _get_default_signals(self) -> Dict[str, Any]:
        """Returns a default signal dictionary (all signals false), a state of magical neutrality."""
        return {
            "enter_long": False,
            "enter_short": False,
            "exit_long": False,
            "exit_short": False,
            "exit_reason": "Strategy Default Exit Signal - No specific omen",  # Generic reason
        }


class DualSupertrendStrategy(TradingStrategy):
    """
    Trading strategy based on two SuperTrend indicators:
    A primary SuperTrend for main trend direction and flips, and a
    confirmation SuperTrend for filtering signals, like two guiding stars.
    """

    def __init__(self, config: Config):
        super().__init__(
            config, df_columns=["st_long_flip", "st_short_flip", "confirm_trend"]
        )  # Internal column names representing SuperTrend states

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates signals based on SuperTrend flips and confirmation trend alignment."""
        signals = self._get_default_signals()
        if not self._validate_df(df):
            return signals

        last_reading = df.iloc[-1]  # The most recent omen
        primary_long_flip = last_reading.get("st_long_flip", False)
        primary_short_flip = last_reading.get("st_short_flip", False)
        confirm_is_up = last_reading.get(
            "confirm_trend", pd.NA
        )  # pd.NA if trend is indeterminate

        if pd.isna(
            confirm_is_up
        ):  # Explicit check for pd.NA (indeterminate confirmation trend)
            self.logger.debug(
                "Confirmation SuperTrend's guidance is unclear (NA). No signal divined."
            )
            return signals

        if primary_long_flip and confirm_is_up is True:
            signals["enter_long"] = True
            self.logger.info(
                "DualSupertrend: Primary ST flipped LONG, confirmed UP. Long entry signal divined."
            )
        if primary_short_flip and confirm_is_up is False:
            signals["enter_short"] = True
            self.logger.info(
                "DualSupertrend: Primary ST flipped SHORT, confirmed DOWN. Short entry signal divined."
            )

        # Exit signals based on primary SuperTrend flips (a change in the main guiding star)
        if primary_short_flip:  # If primary ST flips short, exit any long position.
            signals["exit_long"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Short (Bearish Omen)"
            self.logger.info(
                "DualSupertrend: Primary ST flipped SHORT. Long exit signal divined."
            )
        if primary_long_flip:  # If primary ST flips long, exit any short position.
            signals["exit_short"] = True
            signals["exit_reason"] = "Primary SuperTrend Flipped Long (Bullish Omen)"
            self.logger.info(
                "DualSupertrend: Primary ST flipped LONG. Short exit signal divined."
            )
        return signals


class StochRsiMomentumStrategy(TradingStrategy):
    """
    Trading strategy combining Stochastic RSI for overbought/oversold conditions
    (the market's breath) and a Momentum indicator for trend confirmation (its stride).
    """

    def __init__(self, config: Config):
        super().__init__(config, df_columns=["stochrsi_k", "stochrsi_d", "momentum"])

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates signals based on StochRSI K/D crosses and momentum value."""
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2):
            return signals  # Requires current and previous candle's omens

        last_omen, prev_omen = df.iloc[-1], df.iloc[-2]

        # Fetch raw values, which might be float, Decimal, or pd.NA
        k_now_orig, d_now_orig, mom_now_orig = (
            last_omen.get("stochrsi_k", pd.NA),
            last_omen.get("stochrsi_d", pd.NA),
            last_omen.get("momentum", pd.NA),
        )
        k_prev_orig, d_prev_orig = (
            prev_omen.get("stochrsi_k", pd.NA),
            prev_omen.get("stochrsi_d", pd.NA),
        )

        if any(
            pd.isna(v)
            for v in [k_now_orig, d_now_orig, mom_now_orig, k_prev_orig, d_prev_orig]
        ):
            self.logger.debug(
                "Skipping signal divination due to initial NA values in StochRSI/Momentum runes."
            )
            return signals

        # Convert to Decimal for precise comparison with config thresholds.
        # safe_decimal_conversion handles original float/Decimal/str values, returns Decimal('NaN') if problematic.
        k_now_dec = safe_decimal_conversion(k_now_orig, Decimal("NaN"))
        d_now_dec = safe_decimal_conversion(d_now_orig, Decimal("NaN"))
        mom_now_dec = safe_decimal_conversion(mom_now_orig, Decimal("NaN"))
        k_prev_dec = safe_decimal_conversion(k_prev_orig, Decimal("NaN"))
        d_prev_dec = safe_decimal_conversion(d_prev_orig, Decimal("NaN"))

        if any(
            v.is_nan()
            for v in [k_now_dec, d_now_dec, mom_now_dec, k_prev_dec, d_prev_dec]
        ):
            self.logger.debug(
                "Skipping signal divination due to NA Decimal values after StochRSI/Momentum rune conversion."
            )
            return signals

        # Entry signals: K crosses D in oversold/overbought, confirmed by momentum's stride
        if (
            k_prev_dec <= d_prev_dec
            and k_now_dec > d_now_dec
            and k_now_dec < self.config.stochrsi_oversold
            and mom_now_dec > Decimal("0")
        ):
            signals["enter_long"] = True
            self.logger.info(
                "StochRSI_Momentum: K crossed D UP from OVERSOLD, Momentum POSITIVE. Long entry signal divined."
            )
        if (
            k_prev_dec >= d_prev_dec
            and k_now_dec < d_now_dec
            and k_now_dec > self.config.stochrsi_overbought
            and mom_now_dec < Decimal("0")
        ):
            signals["enter_short"] = True
            self.logger.info(
                "StochRSI_Momentum: K crossed D DOWN from OVERBOUGHT, Momentum NEGATIVE. Short entry signal divined."
            )

        # Exit signals (based on StochRSI cross, a simpler exit omen)
        if k_prev_dec >= d_prev_dec and k_now_dec < d_now_dec:  # K crosses below D
            signals["exit_long"] = True
            signals["exit_reason"] = "StochRSI K crossed below D (Bearish Breath)"
            self.logger.info(
                "StochRSI_Momentum: K crossed D DOWN. Long exit signal divined."
            )
        if k_prev_dec <= d_prev_dec and k_now_dec > d_now_dec:  # K crosses above D
            signals["exit_short"] = True
            signals["exit_reason"] = "StochRSI K crossed above D (Bullish Breath)"
            self.logger.info(
                "StochRSI_Momentum: K crossed D UP. Short exit signal divined."
            )
        return signals


class EhlersFisherStrategy(TradingStrategy):
    """
    Trading strategy using the Ehlers Fisher Transform indicator.
    Signals are generated on crossovers between the Fisher line and its signal line,
    like divining trends from the ebb and flow of transformed market energies.
    """

    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates signals based on Ehlers Fisher Transform crossovers."""
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2):
            return signals

        last_omen, prev_omen = df.iloc[-1], df.iloc[-2]

        fish_now_orig, sig_now_orig = (
            last_omen.get("ehlers_fisher", pd.NA),
            last_omen.get("ehlers_signal", pd.NA),
        )
        fish_prev_orig, sig_prev_orig = (
            prev_omen.get("ehlers_fisher", pd.NA),
            prev_omen.get("ehlers_signal", pd.NA),
        )

        if any(
            pd.isna(v)
            for v in [fish_now_orig, sig_now_orig, fish_prev_orig, sig_prev_orig]
        ):
            self.logger.debug(
                "Skipping signal divination due to NA values in Ehlers Fisher Transform runes."
            )
            return signals

        fish_now = safe_decimal_conversion(fish_now_orig, Decimal("NaN"))
        sig_now = safe_decimal_conversion(sig_now_orig, Decimal("NaN"))
        fish_prev = safe_decimal_conversion(fish_prev_orig, Decimal("NaN"))
        sig_prev = safe_decimal_conversion(sig_prev_orig, Decimal("NaN"))

        if any(v.is_nan() for v in [fish_now, sig_now, fish_prev, sig_prev]):
            self.logger.debug(
                "Skipping signal divination due to NA Decimal values after Ehlers Fisher rune conversion."
            )
            return signals

        # Entry signals: Fisher line crosses its signal line
        if fish_prev <= sig_prev and fish_now > sig_now:  # Fisher crosses above Signal
            signals["enter_long"] = True
            self.logger.info(
                "EhlersFisher: Fisher crossed ABOVE Signal. Long entry signal divined."
            )
        if fish_prev >= sig_prev and fish_now < sig_now:  # Fisher crosses below Signal
            signals["enter_short"] = True
            self.logger.info(
                "EhlersFisher: Fisher crossed BELOW Signal. Short entry signal divined."
            )

        # Exit signals (opposite crossover indicates trend change)
        if fish_prev >= sig_prev and fish_now < sig_now:  # Fisher crosses below Signal
            signals["exit_long"] = True
            signals["exit_reason"] = (
                "Ehlers Fisher crossed below Signal Line (Bearish Energy Shift)"
            )
            self.logger.info(
                "EhlersFisher: Fisher crossed BELOW Signal. Long exit signal divined."
            )
        if fish_prev <= sig_prev and fish_now > sig_now:  # Fisher crosses above Signal
            signals["exit_short"] = True
            signals["exit_reason"] = (
                "Ehlers Fisher crossed above Signal Line (Bullish Energy Shift)"
            )
            self.logger.info(
                "EhlersFisher: Fisher crossed ABOVE Signal. Short exit signal divined."
            )
        return signals


class EhlersMaCrossStrategy(TradingStrategy):
    """
    Trading strategy using a crossover of two Ehlers Super Smoother Filter (SSF)
    Moving Averages (a fast MA and a slow MA). The SSF provides smoother MAs
    with reduced lag, like purified streams of market momentum.
    """

    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_ssf_fast", "ehlers_ssf_slow"])

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates signals based on Ehlers SSF MA crossovers."""
        signals = self._get_default_signals()
        if not self._validate_df(df, min_rows=2):
            return signals

        last_omen, prev_omen = df.iloc[-1], df.iloc[-2]

        fast_ma_now_orig, slow_ma_now_orig = (
            last_omen.get("ehlers_ssf_fast", pd.NA),
            last_omen.get("ehlers_ssf_slow", pd.NA),
        )
        fast_ma_prev_orig, slow_ma_prev_orig = (
            prev_omen.get("ehlers_ssf_fast", pd.NA),
            prev_omen.get("ehlers_ssf_slow", pd.NA),
        )

        if any(
            pd.isna(v)
            for v in [
                fast_ma_now_orig,
                slow_ma_now_orig,
                fast_ma_prev_orig,
                slow_ma_prev_orig,
            ]
        ):
            self.logger.debug(
                "Skipping signal divination due to NA values in Ehlers SSF MA runes."
            )
            return signals

        fast_ma_now = safe_decimal_conversion(fast_ma_now_orig, Decimal("NaN"))
        slow_ma_now = safe_decimal_conversion(slow_ma_now_orig, Decimal("NaN"))
        fast_ma_prev = safe_decimal_conversion(fast_ma_prev_orig, Decimal("NaN"))
        slow_ma_prev = safe_decimal_conversion(slow_ma_prev_orig, Decimal("NaN"))

        if any(
            v.is_nan() for v in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]
        ):
            self.logger.debug(
                "Skipping signal divination due to NA Decimal values after Ehlers SSF MA rune conversion."
            )
            return signals

        # Entry signals: Fast MA crosses Slow MA
        if (
            fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now
        ):  # Fast MA crosses above Slow MA
            signals["enter_long"] = True
            self.logger.info(
                "EhlersMaCross: Fast SSF MA crossed ABOVE Slow SSF MA. Long entry signal divined."
            )
        if (
            fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now
        ):  # Fast MA crosses below Slow MA
            signals["enter_short"] = True
            self.logger.info(
                "EhlersMaCross: Fast SSF MA crossed BELOW Slow SSF MA. Short entry signal divined."
            )

        # Exit signals (opposite crossover indicates trend change)
        if (
            fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now
        ):  # Fast MA crosses below Slow MA
            signals["exit_long"] = True
            signals["exit_reason"] = (
                "Fast Ehlers SSF MA crossed below Slow MA (Bearish Momentum Shift)"
            )
            self.logger.info(
                "EhlersMaCross: Fast SSF MA crossed BELOW Slow. Long exit signal divined."
            )
        if (
            fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now
        ):  # Fast MA crosses above Slow MA
            signals["exit_short"] = True
            signals["exit_reason"] = (
                "Fast Ehlers SSF MA crossed above Slow MA (Bullish Momentum Shift)"
            )
            self.logger.info(
                "EhlersMaCross: Fast SSF MA crossed ABOVE Slow. Short exit signal divined."
            )
        return signals


# Initialize strategy instance in CONFIG after it's loaded and validated
# This binds the chosen Path of Magic to the spell's configuration.
strategy_map = {
    StrategyName.DUAL_SUPERTREND: DualSupertrendStrategy,
    StrategyName.STOCHRSI_MOMENTUM: StochRsiMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy,
    StrategyName.EHLERS_MA_CROSS: EhlersMaCrossStrategy,
}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass:
    CONFIG.strategy_instance = StrategyClass(CONFIG)
    logger.info(
        f"Path of Magic '{CONFIG.strategy_name.value}' has been chosen and its avatar initialized."
    )
else:
    # This case should ideally be caught by Config validation, but as a final safeguard:
    logger.critical(
        f"Failed to find and initialize the avatar for Path of Magic '{CONFIG.strategy_name.value}'. Pyrmethus cannot proceed."
    )
    sys.exit(1)


# --- Trade Metrics Tracking - Chronicling the Battles ---
class TradeMetrics:
    """Tracks and summarizes trading performance, the 'Chronicle of Battles'."""

    def __init__(self):
        self.trades: List[
            Dict[str, Any]
        ] = []  # A list to store details of each skirmish
        self.logger = logging.getLogger(
            "TradeMetrics"
        )  # Dedicated logger for battle reports

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
    ) -> None:
        """
        Logs a completed trade (a concluded battle), calculates its P/L, and stores it in the chronicle.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT:USDT").
            side: Side of the trade ("Long" or "Short").
            entry_price: Average entry price of the trade.
            exit_price: Average exit price of the trade.
            qty: Quantity of the trade.
            entry_time_ms: Timestamp of trade entry (milliseconds).
            exit_time_ms: Timestamp of trade exit (milliseconds).
            reason: Reason for exiting the trade (the battle's end).
        """
        if not (
            entry_price > 0
            and exit_price > 0
            and qty > 0
            and entry_time_ms > 0
            and exit_time_ms > 0
        ):
            self.logger.warning(
                f"Battle log skipped due to invalid parameters: EntryPx={entry_price}, ExitPx={exit_price}, Qty={qty}, EntryTime={entry_time_ms}, ExitTime={exit_time_ms}. Such a skirmish cannot be chronicled."
            )
            return

        profit_per_unit: Decimal
        # Adjust profit calculation for short trades (profit if exit_price < entry_price)
        if (
            side.lower() == CONFIG.side_sell.lower()
            or side.lower() == CONFIG.pos_short.lower()
        ):  # Short battle
            profit_per_unit = entry_price - exit_price
        else:  # Long battle
            profit_per_unit = exit_price - entry_price

        profit = profit_per_unit * qty  # Total spoils of war (or losses)
        # Convert timestamps to timezone-aware datetime objects for the chronicle
        entry_dt = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc)
        exit_dt = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc)
        duration = exit_dt - entry_dt  # Duration of the engagement

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
        )  # Color for victory, defeat, or stalemate
        self.logger.success(  # Using custom success level for positive trade logging
            f"{Fore.MAGENTA}Battle Chronicle: {side.upper()} {qty} {symbol.split('/')[0]} | Entry: {entry_price:.4f}, Exit: {exit_price:.4f} | P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{Style.RESET_ALL} | Duration: {duration} | End Reason: {reason}"
        )

    def summary(self) -> str:
        """Generates, logs,and returns a summary of all recorded trades - the 'Campaign Report'."""
        if not self.trades:
            no_trades_msg = "No battles chronicled yet. The campaign is young."
            self.logger.info(no_trades_msg)
            return no_trades_msg

        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t["profit"] > 0)
        losses = sum(1 for t in self.trades if t["profit"] < 0)
        breakeven = total_trades - wins - losses
        win_rate = (
            (Decimal(wins) / Decimal(total_trades)) * Decimal(100)
            if total_trades > 0
            else Decimal(0)
        )
        total_profit = sum(
            t["profit"] for t in self.trades
        )  # Sum of Decimals for precise accounting of spoils
        avg_profit_per_trade = (
            total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        )

        summary_str_colored = (
            f"\n{Fore.CYAN}--- Campaign Report (Trade Metrics Summary) ---\n"
            f"Total Battles: {total_trades} | Victories (Wins): {Fore.GREEN}{wins}{Style.RESET_ALL}, Defeats (Losses): {Fore.RED}{losses}{Style.RESET_ALL}, Stalemates (Breakeven): {Fore.YELLOW}{breakeven}{Style.RESET_ALL}\n"
            f"Victory Rate: {win_rate:.2f}% | Total Spoils (P/L): {(Fore.GREEN if total_profit > 0 else (Fore.RED if total_profit < 0 else Fore.YELLOW))}{total_profit:.2f} {CONFIG.usdt_symbol}{Style.RESET_ALL}\n"
            f"Avg. Spoils per Battle (P/L): {(Fore.GREEN if avg_profit_per_trade > 0 else (Fore.RED if avg_profit_per_trade < 0 else Fore.YELLOW))}{avg_profit_per_trade:.2f} {CONFIG.usdt_symbol}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}--- End of Campaign Report ---{Style.RESET_ALL}"
        )
        self.logger.info(summary_str_colored)  # Log with colors to the Oracle's console

        # Plain string version for non-console outputs or other uses
        plain_summary_str = (
            f"\n--- Campaign Report (Trade Metrics Summary) ---\n"
            f"Total Battles: {total_trades} | Victories (Wins): {wins}, Defeats (Losses): {losses}, Stalemates (Breakeven): {breakeven}\n"
            f"Victory Rate: {win_rate:.2f}% | Total Spoils (P/L): {total_profit:.2f} {CONFIG.usdt_symbol}\n"
            f"Avg. Spoils per Battle (P/L): {avg_profit_per_trade:.2f} {CONFIG.usdt_symbol}\n"
            f"--- End of Campaign Report ---"
        )
        return plain_summary_str


trade_metrics = TradeMetrics()  # Instantiate the chronicler
# Stores details of the currently active trade (battle) for P/L calculation upon closing.
_active_trade_details: Dict[str, Union[Decimal, int, str, None]] = {
    "entry_price": None,
    "entry_time_ms": None,
    "side": None,
    "qty": None,
}


# --- Helper Functions - Minor Cantrips & Utilities ---
def safe_decimal_conversion(
    value: Any, default: Union[Decimal, pd.NAType, None] = Decimal("0.0")
) -> Union[Decimal, pd.NAType, None]:
    """
    Safely converts a value to Decimal. Handles None, pd.NA, and conversion errors.
    If conversion fails or input is NA/None, returns the provided default value.
    The type of the returned value will match the type of the `default` argument in such cases.
    A small cantrip for ensuring numerical stability.

    Args:
        value: The value to convert (potentially a faded rune).
        default: The value to return on failure or NA/None input.
                 Can be Decimal, pd.NA, or None.

    Returns:
        The converted Decimal, or the default value if the rune was unreadable.
    """
    if pd.isna(value) or value is None:  # pd.isna handles None, np.nan, pd.NaT, pd.NA
        return default
    try:
        return Decimal(
            str(value)
        )  # Convert to string first to handle floats and other types correctly
    except (InvalidOperation, TypeError, ValueError):
        # Log as debug to reduce noise, as this can happen with initial indicator NaNs (fading signals)
        logger.debug(
            f"Could not transmute '{value}' (type: {type(value).__name__}) to Decimal. Using default: {default}"
        )
        return default


def format_order_id(order_id: Union[str, int, None]) -> str:
    """Formats an order ID for concise logging, typically showing its mystical suffix."""
    return str(order_id)[-6:] if order_id else "N/A (No ID)"


def _format_for_log(value: Any, precision: int = 4, is_bool_trend: bool = False) -> str:
    """
    Helper cantrip to format various data types for readable logging output,
    with precision control for numbers and special handling for boolean trends.
    """
    if pd.isna(value) or value is None:
        return "N/A (Faded)"
    if (
        is_bool_trend
    ):  # Special formatting for boolean trend values (omens of direction)
        if value is True:
            return f"{Fore.GREEN}Upward{Style.RESET_ALL}"
        if value is False:
            return f"{Fore.RED}Downward{Style.RESET_ALL}"
        return "N/A (Trend Unclear)"  # Should ideally be caught by pd.isna above
    if isinstance(value, Decimal):
        return f"{value:.{precision}f}"
    if isinstance(value, (float, int)):
        return f"{float(value):.{precision}f}"  # Ensure it's float for formatting
    if isinstance(value, bool):
        return str(value)
    return str(value)  # Fallback for other arcane symbols


def format_price(
    exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal]
) -> str:
    """Formats a price value to the precision required by the exchange for the given symbol. A shaping spell."""
    try:
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error shaping price '{price}' for symbol '{symbol}': {e}. Using raw Decimal form.{Style.RESET_ALL}"
        )
        # Fallback to string representation of Decimal, with a reasonable number of decimal places for prices
        return str(
            Decimal(str(price))
            .quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
            .normalize()
        )


def format_amount(
    exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal]
) -> str:
    """Formats an amount (quantity) to the precision required by the exchange for the given symbol. Another shaping spell."""
    try:
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error shaping amount '{amount}' for symbol '{symbol}': {e}. Using raw Decimal form.{Style.RESET_ALL}"
        )
        # Fallback to string representation of Decimal, with a reasonable number of decimal places for amounts
        return str(
            Decimal(str(amount))
            .quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
            .normalize()
        )


# --- Retry Decorator for API calls - Fortifying Connections ---
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
    """
    Wraps an API call with retry logic for common transient network or exchange errors.
    Uses exponential backoff between retries. A ward against fickle connections.
    """
    return func(*args, **kwargs)


# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: Optional[bool] = (
    None  # Cache for command existence check (the 'Whisperwind' spell availability)
)


def send_sms_alert(message: str) -> bool:
    """
    Sends an SMS alert using Termux:API if enabled and configured.
    This is the 'Whisperwind' spell, carrying urgent messages through the digital aether.

    Args:
        message: The message content for the SMS.

    Returns:
        True if SMS was dispatched successfully, False otherwise.
    """
    global _termux_sms_command_exists
    if not CONFIG.enable_sms_alerts:
        logger.debug(
            "SMS alerts (Whisperwind spell) are disabled in configuration. Skipping."
        )
        return False

    if _termux_sms_command_exists is None:  # Check for command existence only once
        _termux_sms_command_exists = shutil.which("termux-sms-send") is not None
        if not _termux_sms_command_exists:
            logger.warning(
                f"{Fore.YELLOW}Whisperwind Alerting: 'termux-sms-send' command not found. "
                f"Ensure Termux:API is installed and the 'termux-api' package is available in Termux's grimoire.{Style.RESET_ALL}"
            )

    if not _termux_sms_command_exists:
        return False  # Command not found, cannot send whisper

    if not CONFIG.sms_recipient_number:
        logger.debug(
            "Whisperwind spell skipped: Recipient number (SMS_RECIPIENT_NUMBER rune) is not configured."
        )
        return False

    try:
        # Sanitize message slightly for command line (basic warding against injection)
        safe_message = message.replace('"', "'").replace("`", "'").replace("$", "")
        command: List[str] = [
            "termux-sms-send",
            "-n",
            CONFIG.sms_recipient_number,
            safe_message,
        ]

        logger.info(
            f'{Fore.MAGENTA}Dispatching Whisperwind (SMS) to {CONFIG.sms_recipient_number} with message: "{safe_message[:50]}..."{Style.RESET_ALL}'
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
                f"{Fore.MAGENTA}Whisperwind (SMS) dispatched successfully to {CONFIG.sms_recipient_number}.{Style.RESET_ALL}"
            )
            return True
        else:
            logger.error(
                f"{Fore.RED}Whisperwind (SMS) dispatch failed. Return Code: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"{Fore.RED}Whisperwind (SMS) dispatch timed out after {CONFIG.sms_timeout_seconds} seconds. The aether was too slow.{Style.RESET_ALL}"
        )
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}Whisperwind (SMS) dispatch encountered an unexpected disturbance: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """
    Initializes and configures the CCXT exchange object for Bybit.
    Performs basic checks like API key validity and market loading.
    This is the 'Portal Opening' ritual to connect to the exchange's realm.

    Returns:
        A configured ccxt.Exchange object if successful, None otherwise.
    """
    logger.info(
        f"{Fore.BLUE}Opening Bybit Portal via CCXT (Targeting V5 API)... The ritual begins.{Style.RESET_ALL}"
    )
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL: API Key and/or Secret (the Portal Keys) are missing. Cannot connect to Bybit's realm.{Style.RESET_ALL}"
        )
        send_sms_alert(
            "[Pyrmethus] CRITICAL: API Portal Keys missing. Bot cannot start."
        )
        return None
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": CONFIG.api_key,
                "secret": CONFIG.api_secret,
                "enableRateLimit": True,  # CCXT's built-in rate limiter (a mana governor)
                "options": {
                    "defaultType": "linear",  # For USDT perpetuals (the chosen market plane)
                    "adjustForTimeDifference": True,  # Adjust for clock drift (temporal synchronization)
                    # "brokerId": "YOUR_BROKER_ID" # Uncomment and set if using via a broker familiar program
                },
                "recvWindow": CONFIG.default_recv_window,
            }
        )
        # exchange.set_sandbox_mode(True) # Uncomment for Testnet trading (the Astral Plane)

        logger.debug(
            "Loading market structures from Bybit... (Mapping the trading realm)"
        )
        exchange.load_markets(
            force_reload=True
        )  # Force reload to get latest market data
        logger.debug(
            f"Successfully mapped {len(exchange.markets)} market structures from Bybit."
        )

        logger.debug(
            "Performing initial balance check to verify API key permissions and portal stability..."
        )
        # For V5 API, 'category' is often needed for balance and positions
        balance_params = (
            {"category": "linear"}
            if exchange.options.get("defaultType") == "linear"
            else {}
        )
        # Not using safe_api_call here; want to fail fast on initialization if basic auth/conn fails.
        exchange.fetch_balance(params=balance_params)  # A test whisper to the vault

        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened & Authenticated (V5 API). The connection is stable.{Style.RESET_ALL}"
        )

        # Log and alert if running in sandbox (testnet) mode
        if hasattr(exchange, "sandbox") and exchange.sandbox:  # type: ignore
            logger.warning(
                f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}!!! PYRMETHUS IS OPERATING IN THE ASTRAL PLANE (TESTNET MODE) !!!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[Pyrmethus/{CONFIG.strategy_name.value}] Portal opened (ASTRAL/TESTNET). No real mana (funds) at risk."
            )
        else:
            logger.warning(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! PYRMETHUS IS OPERATING IN THE MATERIAL PLANE (LIVE TRADING MODE) - EXTREME CAUTION ADVISED !!!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[Pyrmethus/{CONFIG.strategy_name.value}] Portal opened (MATERIAL/LIVE). Real mana (funds) is active."
            )
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Portal Opening FAILED (Authentication Error): {e}. Check your Portal Keys (API key/secret).{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Portal Authentication FAILED: {e}. Check API keys."
        )
    except ccxt.NetworkError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Portal Opening FAILED (Network Disturbance): {e}. Check internet connection and Bybit's realm status.{Style.RESET_ALL}"
        )
        send_sms_alert(f"[Pyrmethus] CRITICAL: Network Disturbance FAILED: {e}.")
    except (
        Exception
    ) as e:  # Catch-all for other unexpected errors during initialization
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Portal Opening FAILED (Unexpected Anomaly): {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[Pyrmethus] CRITICAL: Exchange initialization FAILED: {type(e).__name__}."
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
    return None


# --- Indicator Calculation Functions - Scrying the Market's Intent ---
def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = ""
) -> pd.DataFrame:
    """
    Calculates the SuperTrend indicator and trend flip signals using pandas_ta.
    This is a scrying technique to divine the market's underlying trend.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns (the raw market energies).
        length: ATR period for SuperTrend (the scrying crystal's focus).
        multiplier: Multiplier for ATR to define SuperTrend bands (the crystal's sensitivity).
        prefix: Optional prefix for output column names (e.g., "confirm_" for a secondary scrying).

    Returns:
        DataFrame augmented with SuperTrend columns:
        '{prefix}supertrend_val', '{prefix}trend' (True for up, False for down, pd.NA if unclear),
        '{prefix}st_long_flip' (omen of bullish reversal), '{prefix}st_short_flip' (omen of bearish reversal).
    """
    col_prefix = f"{prefix}" if prefix else ""
    # Standardized internal column names for consistency within the bot's grimoire
    out_supertrend_val = f"{col_prefix}supertrend_val"
    out_trend_direction = (
        f"{col_prefix}trend"  # True for up, False for down, pd.NA if indeterminate
    )
    out_long_flip = (
        f"{col_prefix}st_long_flip"  # True if trend flipped to long on this candle
    )
    out_short_flip = (
        f"{col_prefix}st_short_flip"  # True if trend flipped to short on this candle
    )
    target_cols = [
        out_supertrend_val,
        out_trend_direction,
        out_long_flip,
        out_short_flip,
    ]

    # pandas_ta column names (standardized by the library's own grimoire)
    pta_st_val_col = f"SUPERT_{length}_{float(multiplier)}"  # SuperTrend line value
    pta_st_dir_col = (
        f"SUPERTd_{length}_{float(multiplier)}"  # Direction (1 for long, -1 for short)
    )

    # Minimum length of DataFrame needed for pandas_ta to start calculating ATR and then SuperTrend
    min_len_needed_for_pta = (
        length + 1
    )  # ATR period + 1 initial value for the spell to take hold

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low", "close"])
        or len(df) < min_len_needed_for_pta
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying ({col_prefix}SuperTrend): Insufficient historical energies (Rows: {len(df) if df is not None else 0}, Min Approx: {min_len_needed_for_pta}). Populating with NAs (unclear visions).{Style.RESET_ALL}"
        )
        if df is not None:  # Ensure df exists to add NA columns
            for col in target_cols:
                df[col] = pd.NA
        return (
            df if df is not None else pd.DataFrame(columns=target_cols)
        )  # Return empty DF if input was None

    try:
        # Work on a copy of relevant columns for pandas_ta to avoid SettingWithCopyWarning if df is a slice
        temp_df = df[["high", "low", "close"]].copy()
        # Calculate SuperTrend using pandas_ta; it appends columns to temp_df
        temp_df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)

        # Check if pandas_ta created the expected columns (runes)
        pta_cols_to_check = [pta_st_val_col, pta_st_dir_col]
        if not all(c in temp_df.columns for c in pta_cols_to_check):
            missing_cols_pta = [
                c for c in pta_cols_to_check if c not in temp_df.columns
            ]
            logger.error(
                f"{Fore.RED}Scrying ({col_prefix}SuperTrend): pandas_ta failed to etch expected SuperTrend runes: {missing_cols_pta}. Populating with NAs.{Style.RESET_ALL}"
            )
            for col in target_cols:
                df[col] = pd.NA  # Assign NAs to original df
            return df

        df[out_supertrend_val] = temp_df[pta_st_val_col].apply(
            lambda x: safe_decimal_conversion(x, pd.NA)
        )

        df[out_trend_direction] = pd.NA  # Default to NA (indeterminate vision)
        df.loc[temp_df[pta_st_dir_col] == 1, out_trend_direction] = (
            True  # Up-trend (bullish energies)
        )
        df.loc[temp_df[pta_st_dir_col] == -1, out_trend_direction] = (
            False  # Down-trend (bearish energies)
        )

        # Detect flips by comparing current direction with previous direction (a shift in the prevailing winds)
        prev_dir = temp_df[pta_st_dir_col].shift(1)
        # Ensure boolean results, handle potential NaNs from shift or original data
        df[out_long_flip] = (temp_df[pta_st_dir_col] == 1) & (
            prev_dir == -1
        )  # Flipped from short to long
        df[out_short_flip] = (temp_df[pta_st_dir_col] == -1) & (
            prev_dir == 1
        )  # Flipped from long to short

        # Fill NaNs in boolean flip columns with False after comparison, as NA means no flip occurred (no change in omen)
        df[out_long_flip] = df[out_long_flip].fillna(False)
        df[out_short_flip] = df[out_short_flip].fillna(False)

        if (
            not df.empty and not df.iloc[-1].isnull().all()
        ):  # Log last calculated values if available
            last_val = df[out_supertrend_val].iloc[-1]
            last_trend_bool = df[out_trend_direction].iloc[
                -1
            ]  # This can be True, False, or pd.NA
            last_l_flip = df[out_long_flip].iloc[-1]
            last_s_flip = df[out_short_flip].iloc[-1]

            trend_str = _format_for_log(last_trend_bool, is_bool_trend=True)
            flip_str = (
                "Bullish Reversal"
                if last_l_flip
                else ("Bearish Reversal" if last_s_flip else "No Reversal")
            )

            logger.debug(
                f"Scrying ({col_prefix}SuperTrend({length},{multiplier})): Value={_format_for_log(last_val)}, Trend Vision={trend_str}, Reversal Omen={flip_str}"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying ({col_prefix}SuperTrend): Error during divination: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        if df is not None:  # Populate with NAs if calculation failed
            for col in target_cols:
                df[col] = pd.NA
    return df


def analyze_volume_atr(
    df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> Dict[str, Union[Decimal, pd.NAType, None]]:
    """
    Calculates Average True Range (ATR - market's breath) and volume moving average (market's pulse),
    then analyzes volume for spikes (surges of energy).

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        atr_len: Period for ATR calculation.
        vol_ma_len: Period for volume moving average.

    Returns:
        Dictionary with 'atr', 'volume_ma', 'last_volume', 'volume_ratio' (last_volume / volume_ma).
        Values can be Decimal or pd.NA if calculation fails or data is insufficient.
    """
    results: Dict[str, Union[Decimal, pd.NAType, None]] = {
        "atr": pd.NA,
        "volume_ma": pd.NA,
        "last_volume": pd.NA,
        "volume_ratio": pd.NA,
    }
    min_len_needed = (
        max(atr_len, vol_ma_len) + 1
    )  # Approx min rows for MA/ATR calculations to stabilize

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low", "close", "volume"])
        or len(df) < min_len_needed
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (Volume/ATR): Insufficient historical energies (Rows: {len(df) if df is not None else 0}, Min Approx: {min_len_needed}). Results will be NA (unclear visions).{Style.RESET_ALL}"
        )
        return results

    try:
        temp_df = df.copy()  # Work on a copy to avoid altering the original scroll

        # ATR Calculation (Market's Breath)
        atr_col_name = f"ATRr_{atr_len}"  # pandas_ta ATR column name
        temp_df.ta.atr(length=atr_len, append=True)
        if atr_col_name in temp_df.columns and not temp_df[atr_col_name].empty:
            results["atr"] = safe_decimal_conversion(
                temp_df[atr_col_name].iloc[-1], pd.NA
            )

        # Volume Analysis (Market's Pulse)
        volume_ma_col_name = f"volume_sma_{vol_ma_len}"
        temp_df["volume"] = pd.to_numeric(
            temp_df["volume"], errors="coerce"
        )  # Ensure volume is numeric
        temp_df[volume_ma_col_name] = (
            temp_df["volume"]
            .rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2))
            .mean()
        )  # SMA of volume

        if not temp_df.empty:
            results["volume_ma"] = safe_decimal_conversion(
                temp_df[volume_ma_col_name].iloc[-1], pd.NA
            )
            results["last_volume"] = safe_decimal_conversion(
                temp_df["volume"].iloc[-1], pd.NA
            )

            vol_ma_val = results["volume_ma"]
            last_vol_val = results["last_volume"]
            # Calculate volume ratio if data is valid and MA is not zero (avoid dividing by ethereal silence)
            if (
                not pd.isna(vol_ma_val)
                and not pd.isna(last_vol_val)
                and vol_ma_val is not None
                and last_vol_val is not None
                and vol_ma_val > CONFIG.position_qty_epsilon
            ):
                try:
                    results["volume_ratio"] = last_vol_val / vol_ma_val
                except (DivisionByZero, InvalidOperation):
                    results["volume_ratio"] = (
                        pd.NA
                    )  # Or some indicator of extreme ratio if preferred

        log_parts = [
            f"ATR({atr_len})={Fore.CYAN}{_format_for_log(results['atr'], 5)}{Style.RESET_ALL} (Breath)"
        ]
        if not pd.isna(results["last_volume"]):
            log_parts.append(f"LastVol={_format_for_log(results['last_volume'], 2)}")
        if not pd.isna(results["volume_ma"]):
            log_parts.append(
                f"VolMA({vol_ma_len})={_format_for_log(results['volume_ma'], 2)} (Pulse)"
            )
        if not pd.isna(results["volume_ratio"]):
            log_parts.append(
                f"VolRatio={Fore.YELLOW}{_format_for_log(results['volume_ratio'], 2)}{Style.RESET_ALL} (Energy Surge)"
            )
        logger.debug(f"Scrying Results (Volume/ATR Divination): {', '.join(log_parts)}")

    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (Volume/ATR): Error during divination: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        results = {key: pd.NA for key in results}  # Ensure all results are NA on error
    return results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    """Calculates Stochastic RSI (K and D lines - market's overextension) and Momentum indicator (market's stride)."""
    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    # Estimate minimum length needed; StochRSI needs RSI length + Stoch length + D period, plus some buffer for the spell to mature.
    min_len_needed = max(rsi_len + stoch_len + d, mom_len) + 10

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len_needed:
        logger.warning(
            f"{Fore.YELLOW}Scrying (StochRSI/Momentum): Insufficient historical energies (Rows: {len(df) if df is not None else 0}, Min Approx: {min_len_needed}). Populating NAs (unclear visions).{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        # StochRSI calculation (pandas_ta returns a DataFrame)
        stochrsi_df_pta = df.ta.stochrsi(
            length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False
        )
        k_col_pta, d_col_pta = (
            f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}",
            f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}",
        )

        if stochrsi_df_pta is not None and not stochrsi_df_pta.empty:
            df["stochrsi_k"] = (
                stochrsi_df_pta[k_col_pta].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if k_col_pta in stochrsi_df_pta
                else pd.NA
            )
            df["stochrsi_d"] = (
                stochrsi_df_pta[d_col_pta].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if d_col_pta in stochrsi_df_pta
                else pd.NA
            )
        else:
            df["stochrsi_k"], df["stochrsi_d"] = pd.NA, pd.NA

        # Momentum calculation
        temp_df_mom = (
            df[["close"]].copy()
        )  # Use a temporary DataFrame for momentum to avoid issues with original df
        mom_col_pta = f"MOM_{mom_len}"
        temp_df_mom.ta.mom(length=mom_len, append=True)  # Appends to temp_df_mom
        df["momentum"] = (
            temp_df_mom[mom_col_pta].apply(lambda x: safe_decimal_conversion(x, pd.NA))
            if mom_col_pta in temp_df_mom
            else pd.NA
        )

        if not df.empty and not df.iloc[-1].isnull().all():
            k_val, d_val, m_val = (
                df["stochrsi_k"].iloc[-1],
                df["stochrsi_d"].iloc[-1],
                df["momentum"].iloc[-1],
            )
            logger.debug(
                f"Scrying (StochRSI/Momentum): K={_format_for_log(k_val, 2)} (Fast Breath), D={_format_for_log(d_val, 2)} (Slow Breath), Momentum({mom_len})={_format_for_log(m_val, 4)} (Stride)"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (StochRSI/Momentum): Error during divination: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates the Ehlers Fisher Transform and its signal line - a spell to reveal sharp turning points."""
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    min_len_needed = (
        length + signal + 5
    )  # Approximate minimum rows for the Fisher spell to stabilize

    if (
        df is None
        or df.empty
        or not all(c in df.columns for c in ["high", "low"])
        or len(df) < min_len_needed
    ):
        logger.warning(
            f"{Fore.YELLOW}Scrying (EhlersFisher): Insufficient historical energies (Rows: {len(df) if df is not None else 0}, Min Approx: {min_len_needed}). Populating NAs (unclear visions).{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        fisher_df_pta = df.ta.fisher(
            length=length, signal=signal, append=False
        )  # pandas_ta returns a DataFrame
        fish_col_pta, signal_col_pta = (
            f"FISHERT_{length}_{signal}",
            f"FISHERTs_{length}_{signal}",
        )

        if fisher_df_pta is not None and not fisher_df_pta.empty:
            df["ehlers_fisher"] = (
                fisher_df_pta[fish_col_pta].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if fish_col_pta in fisher_df_pta
                else pd.NA
            )
            df["ehlers_signal"] = (
                fisher_df_pta[signal_col_pta].apply(
                    lambda x: safe_decimal_conversion(x, pd.NA)
                )
                if signal_col_pta in fisher_df_pta
                else pd.NA
            )
        else:
            df["ehlers_fisher"], df["ehlers_signal"] = pd.NA, pd.NA

        if not df.empty and not df.iloc[-1].isnull().all():
            f_val, s_val = df["ehlers_fisher"].iloc[-1], df["ehlers_signal"].iloc[-1]
            logger.debug(
                f"Scrying (EhlersFisher({length},{signal})): Fisher Line={_format_for_log(f_val)}, Signal Line={_format_for_log(s_val)}"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (EhlersFisher): Error during divination: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def calculate_ehlers_ma(
    df: pd.DataFrame, fast_len: int, slow_len: int, poles: int
) -> pd.DataFrame:
    """Calculates Fast and Slow Ehlers Super Smoother Filter (SSF) Moving Averages - a spell for smoother trend insights."""
    target_cols = ["ehlers_ssf_fast", "ehlers_ssf_slow"]
    min_len_needed = (
        max(fast_len, slow_len) + poles + 5
    )  # Approximate minimum rows for SSF spell to gain clarity

    if df is None or df.empty or "close" not in df.columns or len(df) < min_len_needed:
        logger.warning(
            f"{Fore.YELLOW}Scrying (Ehlers SSF MA): Insufficient historical energies (Rows: {len(df) if df is not None else 0}, Min Approx: {min_len_needed}). Populating NAs (unclear visions).{Style.RESET_ALL}"
        )
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
        return df if df is not None else pd.DataFrame(columns=target_cols)

    try:
        # Calculate SSF Fast MA (pandas_ta returns a Series for ssf)
        ssf_fast_series = df.ta.ssf(length=fast_len, poles=poles, append=False)  # type: ignore
        df["ehlers_ssf_fast"] = (
            ssf_fast_series.apply(lambda x: safe_decimal_conversion(x, pd.NA))
            if ssf_fast_series is not None and not ssf_fast_series.empty
            else pd.NA
        )

        # Calculate SSF Slow MA
        ssf_slow_series = df.ta.ssf(length=slow_len, poles=poles, append=False)  # type: ignore
        df["ehlers_ssf_slow"] = (
            ssf_slow_series.apply(lambda x: safe_decimal_conversion(x, pd.NA))
            if ssf_slow_series is not None and not ssf_slow_series.empty
            else pd.NA
        )

        if not df.empty and not df.iloc[-1].isnull().all():
            fast_val, slow_val = (
                df["ehlers_ssf_fast"].iloc[-1],
                df["ehlers_ssf_slow"].iloc[-1],
            )
            logger.debug(
                f"Scrying (Ehlers SSF MA({fast_len},{slow_len}, Poles:{poles})): Fast MA={_format_for_log(fast_val)}, Slow MA={_format_for_log(slow_val)}"
            )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Scrying (Ehlers SSF MA): Error during divination: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        if df is not None:
            for col in target_cols:
                df[col] = pd.NA
    return df


def analyze_order_book(
    exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int
) -> Dict[str, Union[Decimal, pd.NAType, None]]:
    """
    Fetches and analyzes the L2 order book for the given symbol.
    This is like scrying the collective intent of buyers and sellers.

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.
        depth: Number of bid/ask levels to sum for volume ratio (depth of scrying).
        fetch_limit: Number of levels to fetch from the API (scope of scrying).

    Returns:
        Dictionary with 'bid_ask_ratio', 'spread', 'best_bid', 'best_ask'.
        Values can be Decimal or pd.NA if the vision is unclear.
    """
    results: Dict[str, Union[Decimal, pd.NAType, None]] = {
        "bid_ask_ratio": pd.NA,
        "spread": pd.NA,
        "best_bid": pd.NA,
        "best_ask": pd.NA,
    }
    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying: Exchange '{exchange.id}' does not support fetchL2OrderBook for symbol '{symbol}'. This divination is not possible.{Style.RESET_ALL}"
        )
        return results

    try:
        order_book = safe_api_call(
            exchange.fetch_l2_order_book, symbol, limit=fetch_limit
        )
        bids, asks = order_book.get("bids", []), order_book.get("asks", [])

        if not bids or not asks:
            logger.warning(
                f"{Fore.YELLOW}Order Book Scrying: Empty bid or ask arrays returned for '{symbol}'. Market might be thin, or the aether is disturbed (API issue).{Style.RESET_ALL}"
            )
            return results

        # Best bid and ask prices (the leading edges of buying/selling pressure)
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

        best_bid_val = results["best_bid"]
        best_ask_val = results["best_ask"]
        # Calculate spread if best bid/ask are valid (the gap between forces)
        if (
            not pd.isna(best_bid_val)
            and not pd.isna(best_ask_val)
            and best_bid_val is not None
            and best_ask_val is not None
            and best_bid_val > 0
            and best_ask_val > 0
        ):
            results["spread"] = best_ask_val - best_bid_val

        # Calculate cumulative volume for specified depth (the weight of buying/selling intent)
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

        # Calculate bid/ask volume ratio (the balance of power)
        if (
            ask_vol > CONFIG.position_qty_epsilon
        ):  # Avoid division by zero (if selling intent is non-existent)
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
            except (DivisionByZero, InvalidOperation):
                results["bid_ask_ratio"] = pd.NA  # Unclear balance
        else:  # Ask volume is zero or negligible
            results["bid_ask_ratio"] = (
                pd.NA
            )  # Or Decimal('inf') if bids exist, or 0 if bids also zero. pd.NA is safer.

        log_parts = [
            f"BestBid={Fore.GREEN}{_format_for_log(results['best_bid'], 4)}{Style.RESET_ALL} (Leading Buy)",
            f"BestAsk={Fore.RED}{_format_for_log(results['best_ask'], 4)}{Style.RESET_ALL} (Leading Sell)",
        ]
        if not pd.isna(results["spread"]):
            log_parts.append(
                f"Spread={Fore.YELLOW}{_format_for_log(results['spread'], 4)}{Style.RESET_ALL} (The Gap)"
            )
        if not pd.isna(results["bid_ask_ratio"]):
            log_parts.append(
                f"Ratio(Bid/Ask Vol)={Fore.CYAN}{_format_for_log(results['bid_ask_ratio'], 3)}{Style.RESET_ALL} (Balance of Power)"
            )
        logger.debug(
            f"Order Book Scrying (Depth {depth}, Fetched {fetch_limit}): {', '.join(log_parts)}"
        )

    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Order Book Scrying Error for '{symbol}': {type(e).__name__} - {e}. The vision was obscured.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        results = {key: pd.NA for key in results}  # Ensure all results are NA on error
    return results


# --- Data Fetching & Caching - Gathering Etheric Data Streams ---
_last_market_data: Optional[pd.DataFrame] = (
    None  # Cache for market data DataFrame (the latest scroll of omens)
)
_last_fetch_timestamp: float = (
    0.0  # Timestamp of the last successful fetch (when the scroll was last updated)
)


def get_market_data(
    exchange: ccxt.Exchange, symbol: str, interval: str, limit: int
) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV market data for the given symbol and interval.
    Utilizes a time-based cache: if the current time is within the same candle
    as the last fetch, cached data (the existing scroll) is returned to reduce API calls.

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.
        interval: Timeframe string (e.g., "1m", "5m").
        limit: Number of candles to fetch.

    Returns:
        A Pandas DataFrame with OHLCV data, or None if fetching fails or data is invalid.
    """
    global _last_market_data, _last_fetch_timestamp
    current_time = time.time()

    candle_duration_seconds: int = 0
    try:
        candle_duration_seconds = exchange.parse_timeframe(
            interval
        )  # Duration of one candle in seconds (the rhythm of the market's breath)
    except Exception as e:
        logger.warning(
            f"Could not parse timeframe '{interval}' for caching via exchange.parse_timeframe: {e}. Cache duration check might be affected; fetching fresh data stream."
        )
        candle_duration_seconds = 0  # Force fetch if duration unknown

    cache_is_valid = False
    if (
        _last_market_data is not None
        and not _last_market_data.empty
        and len(_last_market_data) >= limit
    ):
        if candle_duration_seconds > 0:
            time_since_last_fetch = current_time - _last_fetch_timestamp
            # Cache is valid if last fetch was within a configurable percentage of the current candle's duration
            if time_since_last_fetch < (
                candle_duration_seconds * float(CONFIG.cache_candle_duration_multiplier)
            ):
                cache_is_valid = True
        # If candle_duration_seconds is 0 (e.g. parse_timeframe failed), cache is not considered valid based on time.

    if cache_is_valid:
        logger.debug(
            f"Data Stream: Using CACHED market scroll ({len(_last_market_data)} candles) for '{symbol}'. Last fetch: {time.strftime('%H:%M:%S', time.localtime(_last_fetch_timestamp))}"
        )
        return (
            _last_market_data.copy()
        )  # Return a copy to prevent modification of cached data

    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}Data Stream Error: Exchange '{exchange.id}' does not support fetchOHLCV method. Cannot gather etheric data.{Style.RESET_ALL}"
        )
        return None

    try:
        logger.debug(
            f"Data Stream: Gathering {limit} OHLCV candles (etheric patterns) for '{symbol}' (Timeframe: {interval})..."
        )
        # Bybit V5 API requires 'category' for linear contracts
        params = (
            {"category": "linear"}
            if exchange.options.get("defaultType") == "linear"
            else {}
        )
        ohlcv = safe_api_call(
            exchange.fetch_ohlcv, symbol, timeframe=interval, limit=limit, params=params
        )

        if not ohlcv:
            logger.warning(
                f"{Fore.YELLOW}Data Stream: No OHLCV data returned for '{symbol}' ({interval}). Market energies may be dormant or API disturbance.{Style.RESET_ALL}"
            )
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        )  # Convert to UTC datetime for universal timing
        df.set_index("timestamp", inplace=True)

        # Ensure numeric types for OHLCV columns, coercing errors to NaN (handling faded runes)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle NaNs that might appear from API or coercion (mending gaps in the scroll)
        if df.isnull().values.any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            logger.warning(
                f"{Fore.YELLOW}Data Stream: Faded runes (NaNs) found in fetched OHLCV data for columns: {nan_cols}. Applying ffill then bfill to mend the scroll.{Style.RESET_ALL}"
            )
            df.ffill(inplace=True)  # Forward fill NaNs
            df.bfill(
                inplace=True
            )  # Backward fill remaining NaNs (e.g., at the beginning)

            if (
                df.isnull().values.any()
            ):  # If NaNs still exist (e.g., all data was NaN for a column)
                logger.error(
                    f"{Fore.RED}Data Stream: Unfillable NaNs remain in OHLCV scroll for '{symbol}'. Data quality is compromised.{Style.RESET_ALL}"
                )
                return None

        _last_market_data = (
            df.copy()
        )  # Cache the freshly fetched and cleaned data (update the scroll)
        _last_fetch_timestamp = current_time
        logger.debug(
            f"Data Stream: Successfully woven {len(df)} OHLCV candles for '{symbol}'. Data scroll cached."
        )
        return df.copy()  # Return a copy

    except Exception as e:
        logger.error(
            f"{Fore.RED}Data Stream: Error fetching or processing OHLCV for '{symbol}' ({interval}): {e}. The etheric stream was disrupted.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
    return None


# --- Account Health Check - Warding Against Overexposure ---
def check_account_health(exchange: ccxt.Exchange, config: Config) -> bool:
    """
    Checks the account's health by comparing the current margin ratio against a configured maximum.
    This is a 'Vitality Ward' to prevent trading if the account is at excessive risk.

    Args:
        exchange: CCXT exchange instance.
        config: Configuration object containing the spell's parameters.

    Returns:
        True if account health is acceptable, False otherwise (ward is breached).
    """
    logger.debug("Performing Account Vitality Ward assessment...")
    try:
        balance_params = (
            {"category": "linear"}
            if exchange.options.get("defaultType") == "linear"
            else {}
        )
        balance = safe_api_call(
            exchange.fetch_balance, params=balance_params
        )  # Glimpse into the mana reserves

        usdt_balance_data = balance.get(config.usdt_symbol, {})
        if not usdt_balance_data:
            logger.error(
                f"Vitality Ward: '{config.usdt_symbol}' balance data not found in API response. Full balance response: {balance}. Cannot assess vitality."
            )
            return False  # Cannot assess health without USDT balance info

        total_equity = safe_decimal_conversion(
            usdt_balance_data.get("total"), Decimal("NaN")
        )  # Total mana
        used_margin = safe_decimal_conversion(
            usdt_balance_data.get("used"), Decimal("NaN")
        )  # Mana currently bound in spells

        if total_equity.is_nan() or used_margin.is_nan():
            logger.warning(
                f"Vitality Ward: Could not determine total equity or used margin from API response. Total: {usdt_balance_data.get('total')}, Used: {usdt_balance_data.get('used')}. Vitality unclear."
            )
            return False  # Cannot assess health if values are indeterminate

        if total_equity <= Decimal("0"):
            logger.warning(
                f"Vitality Ward: Total equity (mana) is {_format_for_log(total_equity, 2)} {config.usdt_symbol}. Margin ratio calculation skipped."
            )
            if used_margin > Decimal(
                "0"
            ):  # This is a critical state: negative/zero equity but still using margin
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL VITALITY ALERT: Zero/Negative Equity ({_format_for_log(total_equity, 2)}) with Used Margin ({_format_for_log(used_margin, 2)})! Halting spellcasting.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{config.symbol.split('/')[0].split(':')[0]}/{config.strategy_name.value}] CRITICAL: Zero/Negative Mana with Bound Mana! BOT PAUSED."
                )
                return False
            return True  # No used margin and zero/negative equity, technically "healthy" by ratio but implies no mana.

        margin_ratio = (
            used_margin / total_equity if total_equity > 0 else Decimal("Infinity")
        )  # Handle division by zero if equity is zero
        health_color = (
            Fore.GREEN if margin_ratio <= config.max_account_margin_ratio else Fore.RED
        )

        logger.info(
            f"Vitality Ward: Equity={_format_for_log(total_equity, 2)}, UsedMargin={_format_for_log(used_margin, 2)}, "
            f"MarginRatio={health_color}{margin_ratio:.2%}{Style.RESET_ALL} (Configured Max Ward Threshold: {config.max_account_margin_ratio:.0%})"
        )

        if margin_ratio > config.max_account_margin_ratio:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL VITALITY ALERT: Current margin ratio {margin_ratio:.2%} exceeds maximum ward threshold {config.max_account_margin_ratio:.0%}. Halting spellcasting to prevent mana drain.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{config.symbol.split('/')[0].split(':')[0]}/{config.strategy_name.value}] CRITICAL: High margin ratio {margin_ratio:.2%}. BOT PAUSED."
            )
            return False  # Ward breached
        return True  # Vitality acceptable

    except Exception as e:
        logger.error(
            f"Vitality Ward assessment failed with an unexpected disturbance: {type(e).__name__} - {e}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        send_sms_alert(
            f"[{config.symbol.split('/')[0].split(':')[0]}/{config.strategy_name.value}] WARNING: Vitality Ward FAILED: {type(e).__name__}."
        )
        return False  # Assume unhealthy if check fails (fail-safe)


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches and determines the current position (long, short, or flat) for the given symbol.
    Tailored for Bybit V5 API (One-Way Mode). This is a 'Presence Scrying' spell.

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.

    Returns:
        A dictionary: {"side": "Long"|"Short"|"None", "qty": Decimal, "entry_price": Decimal}.
    """
    default_pos_state: Dict[str, Any] = {
        "side": CONFIG.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
    }  # Ethereal state (no presence)
    try:
        market = exchange.market(symbol)
        market_id = market[
            "id"
        ]  # Exchange-specific market ID (the true name of the market)

        # Determine category based on market type (linear/inverse) - the plane of trading
        category = "linear"  # Default for USDT perpetuals
        if market.get("linear"):
            category = "linear"
        elif market.get("inverse"):
            category = "inverse"
        else:
            logger.warning(
                f"{Fore.YELLOW}Presence Scrying: Market type for '{symbol}' not explicitly linear/inverse. Assuming 'linear' plane for API params.{Style.RESET_ALL}"
            )

        params = {"category": category, "symbol": market_id}
        logger.debug(
            f"Presence Scrying: Fetching positions for '{market_id}' on plane '{category}' with params: {params}"
        )
        fetched_positions = safe_api_call(
            exchange.fetch_positions, symbols=[symbol], params=params
        )

        if not fetched_positions:
            logger.info(
                f"{Fore.BLUE}Presence Scrying: No position array returned for '{market_id}' (Plane: '{category}'). Assumed Ethereal (Flat).{Style.RESET_ALL}"
            )
            return default_pos_state

        for pos_data in fetched_positions:
            pos_info = pos_data.get(
                "info", {}
            )  # Raw exchange response (the direct vision)

            # Ensure this position entry is for the correct symbol (API might return all under category if symbol not specified in fetch, though we do specify)
            if pos_info.get("symbol") != market_id:
                continue

            # For Bybit V5 One-Way Mode, positionIdx is 0.
            # Hedge mode would use 1 for Buy side, 2 for Sell side. This bot assumes One-Way Mode (singular focus).
            if int(pos_info.get("positionIdx", -1)) == 0:  # One-Way mode position
                size_str = pos_info.get("size", "0")
                size_dec = safe_decimal_conversion(size_str)

                if (
                    size_dec > CONFIG.position_qty_epsilon
                ):  # If position size is significant (a tangible presence)
                    entry_price_str = pos_info.get(
                        "avgPrice"
                    )  # Average entry price (the price of manifestation)
                    entry_price_dec = safe_decimal_conversion(entry_price_str)

                    bybit_side_str = pos_info.get(
                        "side"
                    )  # "Buy" for Long, "Sell" for Short in Bybit's terms
                    current_pos_side = CONFIG.pos_none
                    if bybit_side_str == "Buy":
                        current_pos_side = CONFIG.pos_long
                    elif bybit_side_str == "Sell":
                        current_pos_side = CONFIG.pos_short
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Presence Scrying: Unknown side '{bybit_side_str}' reported by Bybit for '{market_id}'. Treating as Ethereal.{Style.RESET_ALL}"
                        )
                        continue  # Skip this entry if side is unrecognized

                    pos_color = (
                        Fore.GREEN if current_pos_side == CONFIG.pos_long else Fore.RED
                    )
                    logger.info(
                        f"{pos_color}Presence Scrying: ACTIVE {current_pos_side} presence found for '{market_id}'. Qty={_format_for_log(size_dec, 8)} @ EntryPx={_format_for_log(entry_price_dec, 4)}{Style.RESET_ALL}"
                    )
                    return {
                        "side": current_pos_side,
                        "qty": size_dec,
                        "entry_price": entry_price_dec,
                    }

        # If loop completes without returning, no active One-Way position was found for the symbol
        logger.info(
            f"{Fore.BLUE}Presence Scrying: No active One-Way presence found for '{market_id}' (Plane: '{category}'). Assumed Ethereal (Flat).{Style.RESET_ALL}"
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}Presence Scrying: Error fetching or processing position for '{symbol}': {type(e).__name__} - {e}. The vision was disturbed.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
    return default_pos_state  # Return default flat state on error


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets the leverage for the specified symbol on the exchange. A 'Power Amplification' spell."""
    logger.info(
        f"{Fore.CYAN}Leverage Control: Attempting to amplify power to {leverage}x for '{symbol}'...{Style.RESET_ALL}"
    )
    try:
        market = exchange.market(symbol)
        if not market or not market.get("contract"):
            logger.error(
                f"{Fore.RED}Leverage Control: Market '{symbol}' is not a contract market or not found. Cannot amplify power.{Style.RESET_ALL}"
            )
            return False

        # Bybit V5 API requires category and separate buy/sell leverage params, though they can be the same for cross margin.
        category = "linear"  # Assuming linear contracts (the chosen plane)
        params = {
            "category": category,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }

        response = safe_api_call(
            exchange.set_leverage, leverage=leverage, symbol=symbol, params=params
        )
        logger.success(
            f"{Fore.GREEN}Leverage Control: Power successfully amplified to {leverage}x for '{symbol}' (Plane: {category}). Exchange Response: {response}{Style.RESET_ALL}"
        )
        return True
    except ccxt.ExchangeError as e:
        # Bybit V5 "Set leverage not modified" error code is 110044.
        # Other exchanges/APIs might have different messages for "already set" or "no change".
        err_str = str(e).lower()
        if any(
            sub_err_str in err_str
            for sub_err_str in ["leverage not modified", "same leverage", "110044"]
        ):
            logger.info(
                f"{Fore.CYAN}Leverage Control: Power for '{symbol}' is already amplified to {leverage}x (or no change needed).{Style.RESET_ALL}"
            )
            return True
        logger.error(
            f"{Fore.RED}Leverage Control: Exchange error amplifying power for '{symbol}' to {leverage}x: {e}{Style.RESET_ALL}"
        )
    except Exception as e_unexp:  # Catch any other unexpected errors
        logger.error(
            f"{Fore.RED}Leverage Control: Unexpected disturbance while amplifying power for '{symbol}': {e_unexp}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
    return False


def close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    position_to_close_details: Dict[str, Any],
    reason: str = "Signal",
) -> Optional[Dict[str, Any]]:
    """
    Closes the current active position for the symbol using a market order.
    Re-validates the position before attempting closure. This is a 'Banishment Ritual'.

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.
        position_to_close_details: Dictionary containing details of the position to close (e.g., from get_current_position).
        reason: Reason for closing the position (for logging and metrics).

    Returns:
        The CCXT order dictionary if the close order was placed, None otherwise.
    """
    global _active_trade_details  # To log the trade (battle) and clear details

    initial_side = position_to_close_details.get("side", CONFIG.pos_none)
    initial_qty = position_to_close_details.get("qty", Decimal("0.0"))
    market_base_currency = symbol.split("/")[0].split(":")[
        0
    ]  # e.g., "BTC" from "BTC/USDT:USDT"

    logger.info(
        f"{Fore.YELLOW}Position Banishment Ritual: Attempting to close '{symbol}' presence (Reason: {reason}). "
        f"Initial known details: Side={initial_side}, Qty={_format_for_log(initial_qty, 8)}{Style.RESET_ALL}"
    )

    # Re-validate current position state directly from exchange before closing (confirm presence)
    live_position_state = get_current_position(exchange, symbol)
    if (
        live_position_state["side"] == CONFIG.pos_none
        or live_position_state["qty"] <= CONFIG.position_qty_epsilon
    ):
        logger.warning(
            f"{Fore.YELLOW}Position Banishment: Re-validation shows NO active presence for '{symbol}'. Aborting banishment.{Style.RESET_ALL}"
        )
        if (
            _active_trade_details.get("entry_price") is not None
        ):  # If there were active trade details
            logger.info(
                "Clearing potentially stale active battle details as presence is now confirmed Ethereal."
            )
            _active_trade_details = {
                "entry_price": None,
                "entry_time_ms": None,
                "side": None,
                "qty": None,
            }
        return None  # No position to close

    # Determine side for the closing order (opposite of current position's alignment)
    side_to_execute_close = (
        CONFIG.side_sell
        if live_position_state["side"] == CONFIG.pos_long
        else CONFIG.side_buy
    )
    amount_to_close_str = format_amount(
        exchange, symbol, live_position_state["qty"]
    )  # Format quantity to exchange precision

    try:
        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}Position Banishment: Sending CLOSE order for {live_position_state['side']} presence on '{symbol}' (Reason: {reason}): "
            f"{side_to_execute_close.upper()} MARKET {amount_to_close_str} units...{Style.RESET_ALL}"
        )

        # Bybit V5 API: reduceOnly=True, category="linear", positionIdx=0 for one-way mode close (ensure it only banishes)
        params = {"reduceOnly": True, "category": "linear", "positionIdx": 0}
        close_order_response = safe_api_call(
            exchange.create_market_order,
            symbol=symbol,
            side=side_to_execute_close,
            amount=float(amount_to_close_str),
            params=params,
        )

        # Process the response (read the outcome of the ritual)
        order_status = close_order_response.get("status", "unknown")
        filled_qty_closed = safe_decimal_conversion(
            close_order_response.get("filled", "0")
        )
        avg_fill_price_closed = safe_decimal_conversion(
            close_order_response.get("average", "0")
        )
        close_order_timestamp_ms = close_order_response.get(
            "timestamp"
        )  # Timestamp of the close order itself

        if (
            order_status == "closed"
            and abs(filled_qty_closed - live_position_state["qty"])
            < CONFIG.position_qty_epsilon
        ):
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}Position Banishment CONFIRMED: '{symbol}' presence closed (Reason: {reason}). "
                f"Order ID: ...{format_order_id(close_order_response.get('id'))}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base_currency}/{CONFIG.strategy_name.value}] BANISHED {live_position_state['side']} {amount_to_close_str} units of {market_base_currency} (Reason: {reason}). ExitPx: {_format_for_log(avg_fill_price_closed, 4)}"
            )

            # Log trade to metrics if entry details are available (chronicle the battle)
            if (
                _active_trade_details.get("entry_price") is not None
                and close_order_timestamp_ms is not None
            ):
                # Ensure all required details for trade logging are present
                if all(
                    val is not None
                    for val in [
                        _active_trade_details.get("side"),
                        _active_trade_details.get("qty"),
                        _active_trade_details.get("entry_time_ms"),
                    ]
                ):
                    trade_metrics.log_trade(
                        symbol=symbol,
                        side=str(_active_trade_details["side"]),  # Ensure string type
                        entry_price=Decimal(
                            str(_active_trade_details["entry_price"])
                        ),  # Ensure Decimal
                        exit_price=avg_fill_price_closed,
                        qty=Decimal(
                            str(_active_trade_details["qty"])
                        ),  # Ensure Decimal
                        entry_time_ms=int(
                            str(_active_trade_details["entry_time_ms"])
                        ),  # Ensure int
                        exit_time_ms=close_order_timestamp_ms,
                        reason=reason,
                    )
                else:
                    logger.warning(
                        f"TradeMetrics: Skipping chronicle for '{symbol}' due to missing original side/qty/entry_time in _active_trade_details. The battle's beginning is unclear."
                    )
            else:
                logger.info(
                    f"TradeMetrics: Skipping chronicle for '{symbol}' as no active battle details were found or exit order timestamp was missing. The battle was not fully recorded."
                )

            _active_trade_details = {
                "entry_price": None,
                "entry_time_ms": None,
                "side": None,
                "qty": None,
            }  # Clear active battle details
            return close_order_response
        else:
            # If not fully closed or status uncertain
            logger.warning(
                f"{Fore.YELLOW}Position Banishment: Fill status uncertain for close order on '{symbol}'. "
                f"Expected Qty: {live_position_state['qty']}, Filled Qty: {filled_qty_closed}. Order ID: ...{format_order_id(close_order_response.get('id'))}, Status: {order_status}. "
                f"Manual scrying may be needed.{Style.RESET_ALL}"
            )
            return close_order_response  # Return order details even if uncertain

    except Exception as e:
        logger.error(
            f"{Fore.RED}Position Banishment Ritual FAILED for '{symbol}' (Reason: {reason}): Disturbance during close attempt: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base_currency}/{CONFIG.strategy_name.value}] ERROR Banishing {initial_side} presence ({reason}): {type(e).__name__}."
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
    return None  # Return None if close attempt failed catastrophically


def calculate_position_size(
    usdt_equity: Decimal,
    risk_per_trade_pct: Decimal,
    estimated_entry_price: Decimal,
    estimated_sl_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Calculates the position size based on risk percentage, equity, entry/SL prices, and leverage.
    This is the 'Mana Allocation' cantrip for a trade.

    Args:
        usdt_equity: Total USDT equity in the account (total available mana).
        risk_per_trade_pct: Percentage of equity to risk per trade (mana commitment limit).
        estimated_entry_price: Estimated entry price for the trade.
        estimated_sl_price: Estimated stop-loss price for the trade.
        leverage: Leverage to be used (power amplification factor).
        symbol: Trading symbol.
        exchange: CCXT exchange instance.

    Returns:
        A tuple (quantity, margin_required). Both can be None if calculation fails.
        Quantity is formatted to exchange precision.
    """
    if not (
        estimated_entry_price > 0
        and estimated_sl_price > 0
        and 0 < risk_per_trade_pct < 1
        and usdt_equity > 0
        and leverage > 0
    ):
        logger.error(
            f"{Fore.RED}Mana Allocation Error: Invalid inputs for position sizing. "
            f"Equity: {usdt_equity}, Risk%: {risk_per_trade_pct}, EntryPx: {estimated_entry_price}, SLPx: {estimated_sl_price}, Leverage: {leverage}. Cannot allocate mana.{Style.RESET_ALL}"
        )
        return None, None

    price_difference_per_unit = abs(
        estimated_entry_price - estimated_sl_price
    )  # Potential mana loss per unit if SL ward is breached
    if (
        price_difference_per_unit < CONFIG.position_qty_epsilon
    ):  # Avoid division by zero or near-zero (if entry and SL are too close, the ward is illusory)
        logger.error(
            f"{Fore.RED}Mana Allocation Error: Entry price and Stop Loss price are too close or identical ({price_difference_per_unit}). Cannot calculate position size.{Style.RESET_ALL}"
        )
        return None, None

    try:
        # Calculate total USDT amount to risk (max mana to commit to this spell)
        risk_amount_usdt = usdt_equity * risk_per_trade_pct

        # Calculate raw quantity based on risk amount and price difference
        raw_quantity = risk_amount_usdt / price_difference_per_unit

        # Format quantity to exchange's precision rules (shaping the mana)
        precise_quantity_str = format_amount(exchange, symbol, raw_quantity)
        precise_quantity = Decimal(precise_quantity_str)

        if precise_quantity <= CONFIG.position_qty_epsilon:
            logger.warning(
                f"{Fore.YELLOW}Mana Allocation: Calculated quantity ({_format_for_log(precise_quantity, 8)}) is negligible or zero after precision shaping. "
                f"This might be due to minimum risk not being met or stop-loss ward being too wide relative to risk percentage. No mana allocated.{Style.RESET_ALL}"
            )
            return None, None

        # Calculate position value and required margin (total mana value of the spell and mana cost)
        position_value_usdt = precise_quantity * estimated_entry_price
        margin_required = position_value_usdt / Decimal(leverage)

        logger.debug(f"Mana Allocation Details:")
        logger.debug(
            f"  Equity: {_format_for_log(usdt_equity, 2)} USDT, Risk Per Trade: {risk_per_trade_pct:.2%}, Risk Amount: {_format_for_log(risk_amount_usdt, 2)} USDT"
        )
        logger.debug(
            f"  Est. Entry Px: {_format_for_log(estimated_entry_price, 4)}, Est. SL Px: {_format_for_log(estimated_sl_price, 4)}, Price Diff/Unit: {_format_for_log(price_difference_per_unit, 4)}"
        )
        logger.debug(
            f"  Raw Qty: {_format_for_log(raw_quantity, 8)}, Precise Qty (Shaped Mana): {Fore.CYAN}{_format_for_log(precise_quantity, 8)}{Style.RESET_ALL} units"
        )
        logger.debug(
            f"  Est. Position Value (Spell Value): {_format_for_log(position_value_usdt, 2)} USDT, Est. Margin Req. (Mana Cost): {_format_for_log(margin_required, 4)} USDT (at {leverage}x amplification)"
        )

        return precise_quantity, margin_required
    except (DivisionByZero, InvalidOperation, Exception) as e:
        logger.error(
            f"{Fore.RED}Mana Allocation Error: Disturbance during position size calculation: {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        return None, None


def wait_for_order_fill(
    exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int
) -> Optional[Dict[str, Any]]:
    """
    Waits for a specified order to be filled (status 'closed'), with a timeout.
    This is an 'Order Vigil' to observe its fate.

    Args:
        exchange: CCXT exchange instance.
        order_id: ID of the order to monitor.
        symbol: Trading symbol.
        timeout_seconds: Maximum time to wait for the order to fill.

    Returns:
        The CCXT order dictionary if filled/failed, or None on timeout/unrecoverable error.
    """
    start_time = time.time()
    short_order_id = format_order_id(
        order_id
    )  # A shorter incantation for the order's name
    logger.info(
        f"{Fore.CYAN}Order Vigil: Monitoring order ...{short_order_id} for '{symbol}' for fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}"
    )

    # Bybit V5 API requires category for fetching orders
    params = (
        {"category": "linear"}
        if exchange.options.get("defaultType") == "linear"
        else {}
    )

    while time.time() - start_time < timeout_seconds:
        try:
            order_details = safe_api_call(
                exchange.fetch_order, order_id, symbol, params=params
            )
            status = order_details.get("status")

            if status == "closed":  # Order is filled (the spell has taken full effect)
                logger.success(
                    f"{Fore.GREEN}Order Vigil: Order ...{short_order_id} ('{symbol}') confirmed FILLED/CLOSED.{Style.RESET_ALL}"
                )
                return order_details
            if status in [
                "canceled",
                "rejected",
                "expired",
            ]:  # Order failed or was cancelled (the spell was countered or fizzled)
                logger.error(
                    f"{Fore.RED}Order Vigil: Order ...{short_order_id} ('{symbol}') FAILED with status: '{status}'.{Style.RESET_ALL}"
                )
                return order_details  # Return details of the failed order

            # If order is still open or partially filled, continue polling
            logger.debug(
                f"Order ...{short_order_id} status: {status}. Vigil continues..."
            )
            time.sleep(0.75)  # Brief pause before next observation

        except ccxt.OrderNotFound:
            # This can happen due to propagation delay, especially immediately after placing an order (the echo of the spell)
            logger.warning(
                f"{Fore.YELLOW}Order Vigil: Order ...{short_order_id} ('{symbol}') not found. Possible propagation delay in the aether. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(1.5)  # Longer pause for propagation
        except Exception as e:  # Handle other potential errors during fetch
            logger.warning(
                f"{Fore.YELLOW}Order Vigil: Error checking order ...{short_order_id} ('{symbol}'): {type(e).__name__}. Retrying vigil...{Style.RESET_ALL}"
            )
            logger.debug(
                traceback.format_exc()
            )  # Log the full incantation of the error
            time.sleep(2)  # Pause after other disturbances

    logger.error(
        f"{Fore.RED}Order Vigil: Order ...{short_order_id} ('{symbol}') fill check TIMED OUT after {timeout_seconds}s. Its fate is uncertain.{Style.RESET_ALL}"
    )
    # Attempt one final fetch after timeout
    try:
        final_order_details = safe_api_call(
            exchange.fetch_order, order_id, symbol, params=params
        )
        logger.info(
            f"Final status for order ...{short_order_id} after timeout: {final_order_details.get('status', 'unknown')}"
        )
        return final_order_details
    except Exception as e_final_check:
        logger.error(
            f"{Fore.RED}Final check for order ...{short_order_id} ('{symbol}') also failed: {type(e_final_check).__name__}{Style.RESET_ALL}"
        )
        return None  # Return None if final check also fails


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: Decimal,
    current_atr_value: Union[Decimal, pd.NAType, None],
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal,
) -> Optional[Dict[str, Any]]:
    """
    Places a market order with calculated risk, including setting up Stop Loss (SL)
    and Trailing Stop Loss (TSL) using Bybit V5 API features.
    This is the 'Grand Summoning Ritual' for entering a trade.

    This is a critical and complex function. It involves:
    1. Fetching account balance (mana reserves) and market info (realm details).
    2. Estimating entry price (divining the point of manifestation).
    3. Calculating SL price based on ATR (setting the primary protective ward).
    4. Calculating position size (allocating mana via `calculate_position_size`).
    5. Performing margin and cap checks (ensuring mana limits are respected).
    6. Placing the market entry order (the summoning itself).
    7. Waiting for entry order fill (confirming manifestation).
    8. Placing fixed SL order (binding the primary ward).
    9. Optionally placing TSL order (binding the adaptive shield).
    10. Handling failures, including emergency close if SL placement fails (a critical safety unbinding).

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.
        side: "buy" or "sell" (alignment of the summoning).
        risk_percentage: Risk per trade as a decimal (e.g., 0.01 for 1%).
        current_atr_value: Current ATR value (market's breath).
        sl_atr_multiplier: Multiplier for ATR to set SL distance.
        leverage: Leverage to use (power amplification).
        max_order_cap_usdt: Maximum position value in USDT (mana cap for the spell).
        margin_check_buffer: Buffer for margin check (safety margin for mana cost).
        tsl_percent: Trailing stop percentage (e.g., 0.005 for 0.5%).
        tsl_activation_offset_percent: TSL activation offset from entry, as percentage.

    Returns:
        CCXT order dictionary for the filled entry order if successful, None otherwise.
    """
    global _active_trade_details
    market_base_currency = symbol.split("/")[0].split(":")[0]
    order_side_color = Fore.GREEN if side == CONFIG.side_buy else Fore.RED
    logger.info(
        f"{order_side_color}{Style.BRIGHT}Grand Summoning Ritual: Initiating {side.upper()} market order for '{symbol}'...{Style.RESET_ALL}"
    )

    if (
        pd.isna(current_atr_value)
        or current_atr_value is None
        or current_atr_value <= 0
    ):
        logger.error(
            f"{Fore.RED}Summoning Error: Invalid ATR value ({current_atr_value}). Market's breath is unreadable. Cannot calculate Stop Loss ward or allocate mana.{Style.RESET_ALL}"
        )
        return None

    v5_api_category = (
        "linear"  # Assuming linear contracts for Bybit V5 (the chosen plane)
    )

    try:
        # --- 1. Pre-computation and Checks (Gathering Ritual Components) ---
        balance_data = safe_api_call(
            exchange.fetch_balance, params={"category": v5_api_category}
        )  # Check mana reserves
        market_info = exchange.market(symbol)  # Consult realm map
        min_qty_allowed_by_exchange = safe_decimal_conversion(
            market_info.get("limits", {}).get("amount", {}).get("min"), Decimal("0")
        )  # Smallest permissible manifestation

        usdt_equity = safe_decimal_conversion(
            balance_data.get(CONFIG.usdt_symbol, {}).get("total"), Decimal("NaN")
        )  # Total mana
        usdt_free_margin = safe_decimal_conversion(
            balance_data.get(CONFIG.usdt_symbol, {}).get("free"), Decimal("NaN")
        )  # Unbound mana

        if (
            usdt_equity.is_nan()
            or usdt_equity <= 0
            or usdt_free_margin.is_nan()
            or usdt_free_margin < 0
        ):
            logger.error(
                f"{Fore.RED}Summoning Error: Invalid account equity ({_format_for_log(usdt_equity, 2)}) or free margin ({_format_for_log(usdt_free_margin, 2)}). Insufficient mana.{Style.RESET_ALL}"
            )
            return None

        # --- 2. Estimate Entry Price (Divining Point of Manifestation) ---
        # Try order book first for more current price, fallback to ticker
        order_book_snapshot = analyze_order_book(
            exchange,
            symbol,
            CONFIG.shallow_ob_fetch_depth,
            CONFIG.shallow_ob_fetch_depth,
        )
        estimated_entry_px_orig = (
            order_book_snapshot.get("best_ask")
            if side == CONFIG.side_buy
            else order_book_snapshot.get("best_bid")
        )

        estimated_entry_px: Union[Decimal, pd.NAType, None] = pd.NA
        if (
            not pd.isna(estimated_entry_px_orig)
            and estimated_entry_px_orig is not None
            and estimated_entry_px_orig > 0
        ):
            estimated_entry_px = (
                estimated_entry_px_orig  # Vision from order book is clear
            )
        else:  # Fallback to last traded price from ticker (a more general omen)
            ticker_data = safe_api_call(exchange.fetch_ticker, symbol)
            estimated_entry_px = safe_decimal_conversion(ticker_data.get("last"), pd.NA)

        if (
            pd.isna(estimated_entry_px)
            or estimated_entry_px is None
            or estimated_entry_px <= 0
        ):
            logger.error(
                f"{Fore.RED}Summoning Error: Failed to obtain a valid estimated entry price for '{symbol}'. Point of manifestation unclear.{Style.RESET_ALL}"
            )
            return None

        # --- 3. Calculate Estimated SL Price (Setting the Primary Protective Ward) ---
        sl_distance_from_entry = (
            current_atr_value * sl_atr_multiplier
        )  # Ward's distance based on market's breath
        sl_price_raw_estimate = (
            (estimated_entry_px - sl_distance_from_entry)
            if side == CONFIG.side_buy
            else (estimated_entry_px + sl_distance_from_entry)
        )
        sl_price_estimate_str = format_price(
            exchange, symbol, sl_price_raw_estimate
        )  # Format to exchange precision
        sl_price_estimate = Decimal(sl_price_estimate_str)
        if sl_price_estimate <= 0:  # A ward set at or below zero is no ward at all
            logger.error(
                f"{Fore.RED}Summoning Error: Invalid estimated SL price ({_format_for_log(sl_price_estimate, 4)}) after calculation. Ward cannot be placed.{Style.RESET_ALL}"
            )
            return None

        # --- 4. Calculate Position Size (Mana Allocation) ---
        final_order_qty, estimated_margin_needed = calculate_position_size(
            usdt_equity,
            risk_percentage,
            estimated_entry_px,
            sl_price_estimate,
            leverage,
            symbol,
            exchange,
        )
        if (
            final_order_qty is None
            or estimated_margin_needed is None
            or final_order_qty <= 0
            or estimated_margin_needed <= 0
        ):
            logger.error(
                f"{Fore.RED}Summoning Error: Mana allocation failed or resulted in zero/negative quantity or margin. Spell cannot be cast.{Style.RESET_ALL}"
            )
            return None

        # --- 5. Perform Margin and Cap Checks (Respecting Mana Limits) ---
        estimated_position_value_usdt = (
            final_order_qty * estimated_entry_px
        )  # Total value of the summoned entity
        if estimated_position_value_usdt > CONFIG.max_order_cap_usdt:
            logger.warning(
                f"{Fore.YELLOW}Summoning Warning: Estimated position value {estimated_position_value_usdt:.2f} USDT exceeds configured mana cap {CONFIG.max_order_cap_usdt:.2f} USDT. Adjusting quantity down.{Style.RESET_ALL}"
            )
            final_order_qty = Decimal(
                format_amount(
                    exchange, symbol, CONFIG.max_order_cap_usdt / estimated_entry_px
                )
            )  # Recalculate qty based on cap
            estimated_margin_needed = (final_order_qty * estimated_entry_px) / Decimal(
                leverage
            )  # Recalculate mana cost
            if final_order_qty <= 0:
                logger.error(
                    f"{Fore.RED}Summoning Error: Quantity became zero or negative after applying mana cap. Spell too weak.{Style.RESET_ALL}"
                )
                return None
            logger.info(
                f"Summoning: Quantity (mana shaped) adjusted to {_format_for_log(final_order_qty, 8)} due to max order cap."
            )

        if (
            min_qty_allowed_by_exchange > 0
            and final_order_qty < min_qty_allowed_by_exchange
        ):  # Check against smallest permissible manifestation
            logger.error(
                f"{Fore.RED}Summoning Error: Calculated quantity {_format_for_log(final_order_qty, 8)} is less than minimum allowed by exchange ({_format_for_log(min_qty_allowed_by_exchange, 8)}) for '{symbol}'. Spell too small to manifest.{Style.RESET_ALL}"
            )
            return None
        if (
            usdt_free_margin < estimated_margin_needed * CONFIG.required_margin_buffer
        ):  # Ensure enough unbound mana
            logger.error(
                f"{Fore.RED}Summoning Error: Insufficient FREE mana for this spell. "
                f"Need approx. {_format_for_log(estimated_margin_needed * CONFIG.required_margin_buffer, 2)} USDT (incl. buffer), Have {_format_for_log(usdt_free_margin, 2)} USDT.{Style.RESET_ALL}"
            )
            return None

        # --- 6. Place Entry Order (The Summoning) ---
        entry_order_params = {
            "reduceOnly": False,
            "category": v5_api_category,
            "positionIdx": 0,
        }  # For Bybit V5 one-way mode
        logger.info(
            f"Casting Summoning Spell: {side.upper()} MARKET entry for {float(final_order_qty)} units of '{symbol}'."
        )
        entry_order_response = safe_api_call(
            exchange.create_market_order,
            symbol=symbol,
            side=side,
            amount=float(final_order_qty),
            params=entry_order_params,
        )

        entry_order_id = entry_order_response.get("id")
        if not entry_order_id:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL SUMMONING FAILURE: Entry order for '{symbol}' did NOT return an Order ID! Exchange response: {entry_order_response}. The spell left no trace!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base_currency}/{CONFIG.strategy_name.value}] CRITICAL: Summoning for {symbol} returned NO ID!"
            )
            return None

        # --- 7. Wait for Entry Order Fill (Confirming Manifestation) ---
        filled_entry_order_details = wait_for_order_fill(
            exchange, entry_order_id, symbol, CONFIG.order_fill_timeout_seconds
        )
        if (
            not filled_entry_order_details
            or filled_entry_order_details.get("status") != "closed"
        ):
            status_msg = (
                filled_entry_order_details.get("status")
                if filled_entry_order_details
                else "timeout or error"
            )
            logger.error(
                f"{Fore.RED}Summoning Failed: Entry order ...{format_order_id(entry_order_id)} for '{symbol}' was not filled or failed. Status: {status_msg}. Manifestation incomplete.{Style.RESET_ALL}"
            )
            # Attempt to cancel if it's still open (though market orders usually fill or fail quickly)
            try:
                if (
                    filled_entry_order_details
                    and filled_entry_order_details.get("status") == "open"
                ):
                    safe_api_call(
                        exchange.cancel_order,
                        entry_order_id,
                        symbol,
                        params={"category": v5_api_category},
                    )
                    logger.info(
                        f"Attempted to dispel potentially unmanifested/open entry order ...{format_order_id(entry_order_id)}."
                    )
            except Exception as e_cancel_entry:
                logger.warning(
                    f"Could not dispel potentially unmanifested entry order ...{format_order_id(entry_order_id)}: {e_cancel_entry}"
                )
            return None  # Do not proceed if entry failed

        actual_avg_fill_price = safe_decimal_conversion(
            filled_entry_order_details.get("average"), Decimal("0")
        )  # Price of manifestation
        actual_filled_qty = safe_decimal_conversion(
            filled_entry_order_details.get("filled"), Decimal("0")
        )  # Magnitude of manifestation
        entry_order_timestamp_ms = filled_entry_order_details.get(
            "timestamp"
        )  # Moment of manifestation

        if (
            actual_filled_qty <= CONFIG.position_qty_epsilon
            or actual_avg_fill_price <= CONFIG.position_qty_epsilon
            or entry_order_timestamp_ms is None
        ):
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL SUMMONING FLAW: Invalid fill data for entry order ...{format_order_id(entry_order_id)}. "
                f"Qty: {actual_filled_qty}, AvgPx: {actual_avg_fill_price}, Timestamp: {entry_order_timestamp_ms}. Cannot proceed with protective wards.{Style.RESET_ALL}"
            )
            # Position might be partially open; this state requires manual intervention or more complex handling.
            return None

        logger.success(
            f"{order_side_color}{Style.BRIGHT}MANIFESTATION CONFIRMED: Order ...{format_order_id(entry_order_id)} filled. "
            f"Qty: {_format_for_log(actual_filled_qty, 8)} {market_base_currency} @ AvgPx: {_format_for_log(actual_avg_fill_price, 4)}{Style.RESET_ALL}"
        )

        # Store active trade details (begin chronicling the battle)
        _active_trade_details = {
            "entry_price": actual_avg_fill_price,
            "entry_time_ms": entry_order_timestamp_ms,
            "side": side,
            "qty": actual_filled_qty,
        }

        # --- 8. Place Fixed Stop Loss (SL) Order (Binding the Primary Ward) ---
        sl_placed_successfully = False
        # Recalculate SL price based on actual fill price (attune ward to true manifestation price)
        actual_sl_price_raw = (
            (actual_avg_fill_price - sl_distance_from_entry)
            if side == CONFIG.side_buy
            else (actual_avg_fill_price + sl_distance_from_entry)
        )
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)

        if Decimal(actual_sl_price_str) <= 0:  # Ensure ward is not illusory
            logger.error(
                f"{Back.RED}{Fore.WHITE}CRITICAL WARDING ERROR: Actual SL price ({actual_sl_price_str}) based on fill price is zero or negative! Cannot place fixed SL ward.{Style.RESET_ALL}"
            )
        else:
            sl_order_side = (
                CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            )  # Ward acts opposite to entry
            # Bybit V5 API params for stop market order (fixed SL ward)
            sl_params = {
                "category": v5_api_category,
                "stopPrice": float(actual_sl_price_str),  # Trigger price for SL ward
                "reduceOnly": True,  # Ensures it only reduces position (banishes, not reverses)
                "positionIdx": 0,  # For one-way mode
            }
            try:
                logger.info(
                    f"Binding Fixed SL Ward: Side={sl_order_side}, Qty={float(actual_filled_qty)}, TriggerPx={actual_sl_price_str}"
                )
                sl_order_response = safe_api_call(
                    exchange.create_order,
                    symbol,
                    "StopMarket",
                    sl_order_side,
                    float(actual_filled_qty),
                    price=None,
                    params=sl_params,
                )
                logger.success(
                    f"{Fore.GREEN}Fixed SL Ward bound successfully. ID:...{format_order_id(sl_order_response.get('id'))}, TriggerPx: {actual_sl_price_str}{Style.RESET_ALL}"
                )
                sl_placed_successfully = True
            except Exception as e_sl_placement:
                logger.error(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: FAILED to bind Fixed SL Ward: {e_sl_placement}{Style.RESET_ALL}"
                )
                logger.debug(
                    traceback.format_exc()
                )  # Log the full incantation of the failed warding

        # --- 9. Optionally Place Trailing Stop Loss (TSL) Order (Binding the Adaptive Shield) ---
        tsl_placed_successfully = (
            False  # Not strictly required for emergency close logic, but good to track
        )
        if CONFIG.trailing_stop_percentage > 0:
            # Calculate TSL activation price (when the shield materializes)
            tsl_activation_offset_value = (
                actual_avg_fill_price * CONFIG.trailing_stop_activation_offset_percent
            )  # Use config for TSL activation
            tsl_activation_price_raw = (
                (actual_avg_fill_price + tsl_activation_offset_value)
                if side == CONFIG.side_buy
                else (actual_avg_fill_price - tsl_activation_offset_value)
            )
            tsl_activation_price_str = format_price(
                exchange, symbol, tsl_activation_price_raw
            )

            # Bybit API expects TSL value as percentage string, e.g., "0.5" for 0.5%
            tsl_value_for_api_str = str(
                (tsl_percent * Decimal("100")).normalize()
            )  # E.g., 0.005 -> "0.5"

            if (
                Decimal(tsl_activation_price_str) <= 0
            ):  # Check for valid activation price for the shield
                logger.error(
                    f"{Back.RED}{Fore.WHITE}CRITICAL SHIELDING ERROR: TSL Activation Price ({tsl_activation_price_str}) is zero or negative! Cannot bind Adaptive Shield.{Style.RESET_ALL}"
                )
            else:
                # Bybit V5 API params for Trailing Stop order (placed as a conditional StopMarket order)
                tsl_params_specific = {
                    "category": v5_api_category,
                    "trailingStop": tsl_value_for_api_str,  # Percentage string for Bybit (shield's adaptiveness)
                    "activePrice": float(
                        tsl_activation_price_str
                    ),  # Price at which TSL becomes active
                    "reduceOnly": True,
                    "positionIdx": 0,
                }
                try:
                    tsl_order_side = (
                        CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
                    )  # Shield acts same direction as fixed SL
                    logger.info(
                        f"Binding Adaptive Shield (TSL): Side={tsl_order_side}, Qty={float(actual_filled_qty)}, TrailValue={tsl_value_for_api_str}%, ActivationPx={tsl_activation_price_str}"
                    )
                    tsl_order_response = safe_api_call(
                        exchange.create_order,
                        symbol,
                        "StopMarket",
                        tsl_order_side,
                        float(actual_filled_qty),
                        price=None,
                        params=tsl_params_specific,
                    )
                    logger.success(
                        f"{Fore.GREEN}Adaptive Shield (TSL) bound successfully. ID:...{format_order_id(tsl_order_response.get('id'))}, Trail: {tsl_value_for_api_str}%, ActivateAt: {tsl_activation_price_str}{Style.RESET_ALL}"
                    )
                    tsl_placed_successfully = True
                except Exception as e_tsl_placement:
                    # TSL failure might be less critical than fixed SL, log as warning
                    logger.warning(
                        f"{Back.YELLOW}{Fore.BLACK}WARNING: FAILED to bind Adaptive Shield (TSL): {e_tsl_placement}{Style.RESET_ALL}"
                    )
                    logger.debug(
                        traceback.format_exc()
                    )  # Log the full incantation of the failed shielding

        # --- 10. Handle SL Placement Failure (Emergency Unbinding/Close) ---
        if not sl_placed_successfully:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}CRITICAL SAFETY BREACH: ESSENTIAL Fixed Stop-Loss Ward binding FAILED for new {side.upper()} presence on '{symbol}'. "
                f"Attempting EMERGENCY MARKET BANISHMENT of the presence to mitigate risk.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base_currency}/{CONFIG.strategy_name.value}] CRITICAL: SL Ward FAILED for {side.upper()} {actual_filled_qty} @ {actual_avg_fill_price}. ATTEMPTING EMERGENCY BANISHMENT."
            )

            emergency_close_reason = (
                "EMERGENCY BANISHMENT - FIXED SL WARD BINDING FAILED"
            )
            # _active_trade_details is set; close_position will use it and then clear it.
            # Pass a representation of the current position based on fill data.
            current_pos_details_for_emergency_close = {
                "side": CONFIG.pos_long
                if side == CONFIG.side_buy
                else CONFIG.pos_short,
                "qty": actual_filled_qty,
                "entry_price": actual_avg_fill_price,
            }
            closed_order_info = close_position(
                exchange,
                symbol,
                current_pos_details_for_emergency_close,
                reason=emergency_close_reason,
            )

            if closed_order_info and closed_order_info.get("status") == "closed":
                logger.warning(
                    f"{Fore.YELLOW}Emergency market banishment successful for '{symbol}' due to SL ward failure.{Style.RESET_ALL}"
                )
            else:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}EMERGENCY BANISHMENT FAILED or status uncertain for '{symbol}'. MANUAL INTERVENTION REQUIRED IMMEDIATELY! The entity might still linger!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base_currency}/{CONFIG.strategy_name.value}] CRITICAL: EMERGENCY BANISHMENT FAILED for {symbol}. MANUAL CHECK REQUIRED!"
                )
            return None  # Return None as the original trade attempt effectively failed safely

        # If all protection orders (at least fixed SL) are placed:
        sl_info = f"SL Ward@~{actual_sl_price_str}"
        tsl_info = (
            f"AdaptiveShield:{tsl_value_for_api_str}%@~{tsl_activation_price_str}"
            if CONFIG.trailing_stop_percentage > 0 and tsl_placed_successfully
            else "AdaptiveShield:N/A"
        )
        send_sms_alert(
            f"[{market_base_currency}/{CONFIG.strategy_name.value}] MANIFESTED {side.upper()} {_format_for_log(actual_filled_qty, 8)} @ {_format_for_log(actual_avg_fill_price, 4)}. {sl_info}, {tsl_info}. EntryID:...{format_order_id(entry_order_id)}"
        )
        return (
            filled_entry_order_details  # Return details of the successful entry order
        )

    except (
        Exception
    ) as e_main_order_ritual:  # Catch-all for the entire summoning ritual
        logger.error(
            f"{Back.RED}{Fore.WHITE}Grand Summoning Ritual FAILED with an unhandled disturbance: {type(e_main_order_ritual).__name__} - {e_main_order_ritual}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error
        send_sms_alert(
            f"[{market_base_currency}/{CONFIG.strategy_name.value}] SUMMONING FAIL ({side.upper()}): {type(e_main_order_ritual).__name__}"
        )
    return None  # Return None if any part of the ritual fails significantly


def cancel_open_orders(
    exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup"
) -> int:
    """
    Cancels all open orders for the specified symbol. A 'Dispelling Ritual' for lingering spells.

    Args:
        exchange: CCXT exchange instance.
        symbol: Trading symbol.
        reason: Reason for cancellation (for logging).

    Returns:
        The number of orders successfully cancelled or confirmed gone.
    """
    logger.info(
        f"{Fore.CYAN}Order Dispelling Ritual: Cancelling ALL open orders (lingering spells) for '{symbol}' (Reason: {reason})...{Style.RESET_ALL}"
    )
    cancelled_count, failed_count = 0, 0

    # Bybit V5 API requires category for fetching/cancelling orders
    v5_api_category = (
        "linear" if exchange.options.get("defaultType") == "linear" else {}
    )
    params_for_api = {"category": v5_api_category} if v5_api_category else {}

    try:
        open_orders_list = safe_api_call(
            exchange.fetch_open_orders, symbol, params=params_for_api
        )

        if not open_orders_list:
            logger.info(
                f"{Fore.CYAN}Order Dispelling: No open orders (lingering spells) found for '{symbol}' (Plane: {v5_api_category or 'N/A'}).{Style.RESET_ALL}"
            )
            return 0

        logger.warning(
            f"{Fore.YELLOW}Order Dispelling: Found {len(open_orders_list)} open orders for '{symbol}'. Attempting to dispel...{Style.RESET_ALL}"
        )
        for order_data in open_orders_list:
            order_id = order_data.get("id")
            order_type = order_data.get("type", "N/A")
            order_side = order_data.get("side", "N/A")
            order_price = order_data.get("price", "N/A")

            if order_id:
                try:
                    logger.info(
                        f"Dispelling order ...{format_order_id(order_id)} (Type: {order_type}, Side: {order_side}, Price: {order_price})"
                    )
                    safe_api_call(
                        exchange.cancel_order, order_id, symbol, params=params_for_api
                    )
                    cancelled_count += 1
                    logger.debug(
                        f"Order ...{format_order_id(order_id)} dispelling request sent successfully."
                    )
                except (
                    ccxt.OrderNotFound
                ):  # If order was already filled or cancelled (already dispelled)
                    logger.info(
                        f"{Fore.GREEN}Order ...{format_order_id(order_id)} already dispelled (not found). Considered handled.{Style.RESET_ALL}"
                    )
                    cancelled_count += 1
                except Exception as e_cancel_single:
                    logger.error(
                        f"{Fore.RED}Order Dispelling: FAILED to dispel order ...{format_order_id(order_id)}: {e_cancel_single}{Style.RESET_ALL}"
                    )
                    failed_count += 1
            else:  # Should not happen with valid API responses
                logger.error(
                    f"{Fore.RED}Order Dispelling: Found an open order without an ID (an unnamed spell). Order data: {order_data}{Style.RESET_ALL}"
                )
                failed_count += 1

        if failed_count > 0:
            send_sms_alert(
                f"[{symbol.split('/')[0].split(':')[0]}/{CONFIG.strategy_name.value}] WARNING: Failed to dispel {failed_count} orders during cleanup for {symbol} (Reason: {reason}). Manual scrying recommended."
            )

    except Exception as e_fetch_open:
        logger.error(
            f"{Fore.RED}Order Dispelling: Error fetching or processing open orders for '{symbol}': {e_fetch_open}. The aether is disturbed.{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())  # Log the full incantation of the error

    log_msg_color = (
        Fore.GREEN
        if failed_count == 0 and cancelled_count > 0
        else (Fore.YELLOW if failed_count > 0 else Fore.CYAN)
    )
    logger.info(
        f"{log_msg_color}Order Dispelling for '{symbol}' (Reason: {reason}): Dispelled/Handled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}"
    )
    return cancelled_count


# --- Strategy Signal Generation Wrapper - The Strategy's Decree ---
def generate_strategy_signals(
    df_with_indicators: pd.DataFrame, strategy_instance: TradingStrategy
) -> Dict[str, Any]:
    """
    Wrapper to call the `generate_signals` method of the currently active strategy instance.
    This is where the chosen Path of Magic issues its decree based on the omens.
    """
    if strategy_instance:
        return strategy_instance.generate_signals(df_with_indicators)

    # Fallback if strategy_instance is somehow None (should be caught earlier by CONFIG initialization)
    logger.error(
        "CRITICAL OMEN: Strategy instance (Path of Magic) is not initialized. Cannot generate signals from the void."
    )
    # Return a default "no signal" structure using a temporary base strategy instance
    # This is a safety net and should ideally never be reached if CONFIG.strategy_instance is properly set.
    return TradingStrategy(CONFIG)._get_default_signals()


# --- All Indicator Calculations - The Grand Scrying Ritual ---
def calculate_all_indicators(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, Dict[str, Union[Decimal, pd.NAType, None]]]:
    """
    Calculates all indicators required by any strategy and returns the
    DataFrame augmented with these indicators, plus volume/ATR analysis.
    This is the 'Grand Scrying Ritual' where all omens are gathered and interpreted.
    """
    logger.debug(
        "Performing Grand Scrying Ritual: Calculating all indicators from the market's etheric patterns..."
    )

    # Dual SuperTrend (Primary and Confirmation Guiding Stars)
    df = calculate_supertrend(
        df, config.st_atr_length, config.st_multiplier
    )  # Primary ST
    df = calculate_supertrend(
        df,
        config.confirm_st_atr_length,
        config.confirm_st_multiplier,
        prefix="confirm_",
    )  # Confirmation ST

    # StochRSI + Momentum (Market's Breath and Stride)
    df = calculate_stochrsi_momentum(
        df,
        config.stochrsi_rsi_length,
        config.stochrsi_stoch_length,
        config.stochrsi_k_period,
        config.stochrsi_d_period,
        config.momentum_length,
    )
    # Ehlers Fisher Transform (Divining Turning Points)
    df = calculate_ehlers_fisher(
        df, config.ehlers_fisher_length, config.ehlers_fisher_signal_length
    )

    # Ehlers MA Cross (Super Smoother Filter - Purified Momentum Streams)
    df = calculate_ehlers_ma(
        df,
        config.ehlers_fast_period,
        config.ehlers_slow_period,
        config.ehlers_ssf_poles,
    )

    # Volume and ATR analysis (Market's Pulse and Overall Volatility/Breath)
    # ATR is also a key component for SL ward calculations.
    vol_atr_analysis_results = analyze_volume_atr(
        df, config.atr_calculation_period, config.volume_ma_period
    )

    logger.debug(
        "Grand Scrying Ritual complete. Indicators (omens) woven into the data stream (scroll)."
    )
    return df, vol_atr_analysis_results


# --- Trading Logic - The Core Spell Weaving ---
def trade_logic(
    exchange: ccxt.Exchange, symbol: str, market_data_df: pd.DataFrame
) -> None:
    """
    Main trading logic function executed each cycle.
    It fetches data, calculates indicators (scries for omens), generates signals (interprets decrees),
    applies filters (tests concordance), and manages trades (summons, banishes, or holds presence).
    This is the heart of Pyrmethus's spell-weaving.
    """
    candle_time_str = "N/A (Time Unclear)"
    if not market_data_df.empty and isinstance(market_data_df.index[-1], pd.Timestamp):
        candle_time_str = market_data_df.index[-1].strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )  # Time of the latest omen

    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== Pyrmethus Spell Cycle Start ({CONFIG.strategy_name.value}) for '{symbol}' | Candle Omen Time: {candle_time_str} =========={Style.RESET_ALL}"
    )

    # Ensure sufficient data for indicator calculation (enough history for reliable omens)
    # This value should be at least the longest lookback period of any indicator.
    min_rows_for_indicators = (
        100  # A general safe minimum; can be fine-tuned based on strategy needs
    )
    if market_data_df is None or len(market_data_df) < min_rows_for_indicators:
        logger.warning(
            f"{Fore.YELLOW}Spell Weaving Halted: Insufficient historical energies for '{symbol}' "
            f"({len(market_data_df) if market_data_df is not None else 0} rows, need approx. {min_rows_for_indicators}). Omens would be unreliable. Skipping logic cycle.{Style.RESET_ALL}"
        )
        return

    try:
        # Perform the Grand Scrying Ritual: Calculate all indicators and get volume/ATR data
        df_with_all_indicators, vol_atr_data = calculate_all_indicators(
            market_data_df.copy(), CONFIG
        )

        # Extract key current market values from the scrying
        current_atr_val_orig = vol_atr_data.get("atr")  # Market's current breath
        current_atr_value: Union[Decimal, pd.NAType, None] = pd.NA  # Default to NA
        if not pd.isna(current_atr_val_orig) and current_atr_val_orig is not None:
            current_atr_value = current_atr_val_orig  # Already a Decimal or pd.NA from analyze_volume_atr

        last_candle_info = (
            df_with_all_indicators.iloc[-1]
            if not df_with_all_indicators.empty
            else pd.Series(dtype="object")
        )  # Most recent full set of omens
        current_close_price_orig = last_candle_info.get(
            "close"
        )  # Current focal point of market energy
        current_close_price: Union[Decimal, pd.NAType, None] = pd.NA  # Default to NA
        if (
            not pd.isna(current_close_price_orig)
            and current_close_price_orig is not None
        ):
            current_close_price = safe_decimal_conversion(
                current_close_price_orig, pd.NA
            )  # Ensure Decimal or pd.NA

        if (
            pd.isna(current_close_price)
            or current_close_price is None
            or current_close_price <= 0
        ):
            logger.warning(
                f"{Fore.YELLOW}Spell Weaving Halted: Invalid or missing last close price for '{symbol}'. Focal energy point unclear. Last candle omens: "
                f"{last_candle_info.to_dict() if not last_candle_info.empty else 'empty/unavailable'}{Style.RESET_ALL}"
            )
            return

        # Check if ATR (market's breath) is valid for SL ward calculation (needed for new summonings)
        can_place_new_order_with_atr_sl = (
            not pd.isna(current_atr_value)
            and current_atr_value is not None
            and current_atr_value > 0
        )

        # Get current position status (Pyrmethus's current market presence)
        current_position_state = get_current_position(exchange, symbol)
        pos_side, pos_qty, pos_entry_price = (
            current_position_state["side"],
            current_position_state["qty"],
            current_position_state["entry_price"],
        )

        # Fetch order book data (scry collective intent) if configured or if considering a new summoning
        order_book_analysis_data: Optional[Dict[str, Any]] = None
        if CONFIG.fetch_order_book_per_cycle or (
            pos_side == CONFIG.pos_none and can_place_new_order_with_atr_sl
        ):
            order_book_analysis_data = analyze_order_book(
                exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit
            )

        # --- Log Current State Snapshot (The Oracle's Vision) ---
        logger.info(
            f"{Fore.MAGENTA}--- Indicator & Market Snapshot for {symbol} ({CONFIG.strategy_name.value}) ---{Style.RESET_ALL}"
        )
        logger.info(
            f"  Market State: ClosePx={_format_for_log(current_close_price, 4)} (Focal Energy), ATR({CONFIG.atr_calculation_period})={_format_for_log(current_atr_value, 5)} (Breath)"
        )

        volume_ratio_val = vol_atr_data.get("volume_ratio")
        is_volume_spike = False  # Default assumption of no energy surge
        if not pd.isna(volume_ratio_val) and volume_ratio_val is not None:
            is_volume_spike = volume_ratio_val > CONFIG.volume_spike_threshold
        logger.info(
            f"  Volume Analysis: Ratio={_format_for_log(volume_ratio_val, 2)} (Energy Surge), SpikeThreshold={CONFIG.volume_spike_threshold}, IsSpike={is_volume_spike}"
        )

        if order_book_analysis_data and isinstance(order_book_analysis_data, dict):
            logger.info(
                f"  Order Book (Collective Intent): Ratio(Bid/Ask Vol)={_format_for_log(order_book_analysis_data.get('bid_ask_ratio'), 3)}, Spread={_format_for_log(order_book_analysis_data.get('spread'), 4)}"
            )

        # Log strategy-specific indicator values from the last candle (omens for the chosen Path of Magic)
        strategy_logger = (
            CONFIG.strategy_instance.logger
        )  # Use the strategy's own logger for its omens
        strategy_logger.info(f"  Strategy Omens ({CONFIG.strategy_name.value}):")
        for col_name in (
            CONFIG.strategy_instance.required_columns
        ):  # Log only columns relevant to the active strategy
            if col_name in last_candle_info.index:
                is_trend_col = (
                    "trend" in col_name.lower()
                )  # Heuristic for special boolean trend formatting (directional omens)
                strategy_logger.info(
                    f"    {col_name}: {_format_for_log(last_candle_info[col_name], is_bool_trend=is_trend_col)}"
                )
            else:  # Should not happen if df validation is robust
                strategy_logger.warning(
                    f"    {col_name}: N/A (Omen not found in DataFrame's last reading - check indicator scrying)"
                )

        pos_color = (
            Fore.GREEN
            if pos_side == CONFIG.pos_long
            else (Fore.RED if pos_side == CONFIG.pos_short else Fore.BLUE)
        )
        logger.info(
            f"  Current Presence: Side={pos_color}{pos_side}{Style.RESET_ALL}, Qty={_format_for_log(pos_qty, 8)}, EntryPx={_format_for_log(pos_entry_price, 4)}"
        )
        logger.info(
            f"{Fore.MAGENTA}{'-' * 70}{Style.RESET_ALL}"
        )  # Dynamic separator for clarity
        # --- End Snapshot ---

        # Generate signals from the active strategy (Interpret the Path of Magic's decree)
        strategy_signals = generate_strategy_signals(
            df_with_all_indicators, CONFIG.strategy_instance
        )

        # --- Handle Exits (Banishment Rituals if Decreed) ---
        should_exit_long_position = (
            pos_side == CONFIG.pos_long and strategy_signals.get("exit_long", False)
        )
        should_exit_short_position = (
            pos_side == CONFIG.pos_short and strategy_signals.get("exit_short", False)
        )

        if should_exit_long_position or should_exit_short_position:
            exit_reason_from_strategy = strategy_signals.get(
                "exit_reason", "Strategy Exit Omen"
            )
            logger.warning(
                f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}*** STRATEGY EXIT DECREE: Attempting to banish {pos_side} presence for '{symbol}' (Reason: {exit_reason_from_strategy}) ***{Style.RESET_ALL}"
            )
            cancel_open_orders(
                exchange, symbol, f"Pre-Banishment ({exit_reason_from_strategy})"
            )  # Dispel other spells before banishing main presence
            time.sleep(0.5)  # Small delay for order dispelling to process
            close_order_result = close_position(
                exchange,
                symbol,
                current_position_state,
                reason=exit_reason_from_strategy,
            )
            if close_order_result:
                logger.info(
                    f"Position banishment order for '{symbol}' processed due to strategy decree. Waiting for post-banishment stillness before next cycle actions."
                )
                time.sleep(CONFIG.post_close_delay_seconds)
            return  # End cycle after attempting banishment

        # --- Handle Holding Position (Maintaining Presence) ---
        if pos_side != CONFIG.pos_none:
            logger.info(
                f"Maintaining {pos_color}{pos_side}{Style.RESET_ALL} presence for '{symbol}'. Awaiting SL/TSL ward trigger or explicit strategy banishment decree."
            )
            return  # End cycle, no new summoning if already present in market

        # --- Handle Entries (Summoning Rituals, If Flat and Conditions Met) ---
        if not can_place_new_order_with_atr_sl:
            logger.warning(
                f"{Fore.YELLOW}Holding Mana (Cash) for '{symbol}'. Cannot consider new summoning: Invalid ATR ({current_atr_value}) for SL ward calculation. Market's breath unclear.{Style.RESET_ALL}"
            )
            return

        potential_enter_long_signal = strategy_signals.get(
            "enter_long", False
        )  # Potential bullish decree
        potential_enter_short_signal = strategy_signals.get(
            "enter_short", False
        )  # Potential bearish decree

        if not (potential_enter_long_signal or potential_enter_short_signal):
            logger.info(
                f"Holding Mana (Cash) for '{symbol}'. No summoning decree from strategy '{CONFIG.strategy_name.value}'."
            )
            return

        # Apply Confirmation Filters (Seeking Concordance in Etheric Energies)
        ob_confirms_long, ob_confirms_short = (
            True,
            True,
        )  # Default to true if filter not active or data unavailable
        if (
            order_book_analysis_data
            and isinstance(order_book_analysis_data, dict)
            and not pd.isna(order_book_analysis_data.get("bid_ask_ratio"))
            and order_book_analysis_data.get("bid_ask_ratio") is not None
        ):
            ob_ratio = order_book_analysis_data[
                "bid_ask_ratio"
            ]  # Collective intent ratio
            if isinstance(ob_ratio, Decimal):  # Ensure ratio is Decimal for comparison
                if CONFIG.order_book_ratio_threshold_long < Decimal(
                    "Infinity"
                ):  # Check if long OB filter is active
                    ob_confirms_long = (
                        ob_ratio >= CONFIG.order_book_ratio_threshold_long
                    )
                if CONFIG.order_book_ratio_threshold_short > Decimal(
                    0
                ):  # Check if short OB filter is active
                    ob_confirms_short = (
                        ob_ratio <= CONFIG.order_book_ratio_threshold_short
                    )
        elif CONFIG.order_book_ratio_threshold_long < Decimal(
            "Infinity"
        ) or CONFIG.order_book_ratio_threshold_short > Decimal(0):
            # If OB filters are active but data was not available/valid
            logger.warning(
                f"{Fore.YELLOW}Order book filter is active, but no valid OB data or ratio was divined. Summoning blocked by unclear collective intent.{Style.RESET_ALL}"
            )
            ob_confirms_long, ob_confirms_short = (
                False,
                False,
            )  # Block entry if filter active and data missing

        volume_confirms_entry = (
            not CONFIG.require_volume_spike_for_entry or is_volume_spike
        )  # Volume energy surge confirmation

        # Final decision for entry (all omens and concordances align)
        final_enter_long = (
            potential_enter_long_signal and ob_confirms_long and volume_confirms_entry
        )
        final_enter_short = (
            potential_enter_short_signal and ob_confirms_short and volume_confirms_entry
        )

        # Prepare parameters for place_risked_market_order (the Grand Summoning Ritual)
        entry_order_common_params: Dict[str, Any] = {
            "exchange": exchange,
            "symbol": symbol,
            "risk_percentage": CONFIG.risk_per_trade_percentage,
            "current_atr_value": current_atr_value,
            "sl_atr_multiplier": CONFIG.atr_stop_loss_multiplier,
            "leverage": CONFIG.leverage,
            "max_order_cap_usdt": CONFIG.max_order_usdt_amount,
            "margin_check_buffer": CONFIG.required_margin_buffer,
            "tsl_percent": CONFIG.trailing_stop_percentage,
            "tsl_activation_offset_percent": CONFIG.trailing_stop_activation_offset_percent,
        }

        if final_enter_long:
            logger.success(
                f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** CONFIRMED LONG SUMMONING DECREE for '{symbol}' (Strategy: {CONFIG.strategy_name.value}) ***{Style.RESET_ALL}"
            )
            logger.info(
                f"Concordance Status: OrderBook Long Confirm: {ob_confirms_long}, Volume Confirm: {volume_confirms_entry}"
            )
            cancel_open_orders(
                exchange, symbol, "Pre-Long Summoning Cleanup"
            )  # Dispel lingering spells
            time.sleep(0.5)
            place_risked_market_order(side=CONFIG.side_buy, **entry_order_common_params)
        elif final_enter_short:
            logger.success(
                f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** CONFIRMED SHORT SUMMONING DECREE for '{symbol}' (Strategy: {CONFIG.strategy_name.value}) ***{Style.RESET_ALL}"
            )
            logger.info(
                f"Concordance Status: OrderBook Short Confirm: {ob_confirms_short}, Volume Confirm: {volume_confirms_entry}"
            )
            cancel_open_orders(
                exchange, symbol, "Pre-Short Summoning Cleanup"
            )  # Dispel lingering spells
            time.sleep(0.5)
            place_risked_market_order(
                side=CONFIG.side_sell, **entry_order_common_params
            )
        elif (
            potential_enter_long_signal or potential_enter_short_signal
        ):  # Decree present but filters blocked
            logger.info(
                f"Holding Mana (Cash) for '{symbol}'. Summoning decree was present but concordance filters were not met. "
                f"Long Filters: OB({ob_confirms_long}), Vol({volume_confirms_entry}). "
                f"Short Filters: OB({ob_confirms_short}), Vol({volume_confirms_entry}). The stars are not aligned."
            )

    except (
        Exception
    ) as e:  # Catch-all for unexpected errors within the trade logic cycle
        logger.error(
            f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL ERROR in spell-weaving cycle for '{symbol}': {e}{Style.RESET_ALL}"
        )
        logger.debug(
            traceback.format_exc()
        )  # Log full traceback for debugging (the failed incantation)
        market_base_currency = symbol.split("/")[0].split(":")[0]
        send_sms_alert(
            f"[{market_base_currency}/{CONFIG.strategy_name.value}] CRITICAL spell-weaving ERROR: {type(e).__name__}. Cycle disrupted."
        )
    finally:
        logger.info(
            f"{Fore.BLUE}{Style.BRIGHT}========== Pyrmethus Spell Cycle End for '{symbol}' =========={Style.RESET_ALL}\n"
        )


# --- Graceful Shutdown - Withdrawing the Arcane Energies ---
def graceful_shutdown(
    exchange_instance: Optional[ccxt.Exchange], trading_symbol: Optional[str]
) -> None:
    """
    Handles graceful shutdown of the bot. Attempts to cancel open orders (dispel lingering spells),
    close any active position (banish summoned entities), and log final trade metrics (the campaign's end).
    """
    logger.warning(
        f"\n{Fore.YELLOW}{Style.BRIGHT}Shutdown Sequence Initiated. Withdrawing Pyrmethus's arcane energies...{Style.RESET_ALL}"
    )

    market_base_name = "Bot"  # Default for SMS if symbol not available
    if trading_symbol:
        parts = trading_symbol.split("/")
        if parts:
            market_base_name = parts[0].split(":")[0]

    strategy_name_for_sms = "N/A"
    if (
        "CONFIG" in globals()
        and hasattr(CONFIG, "strategy_name")
        and CONFIG.strategy_name
    ):
        strategy_name_for_sms = CONFIG.strategy_name.value

    send_sms_alert(
        f"[{market_base_name}/{strategy_name_for_sms}] Pyrmethus shutdown initiated. Attempting cleanup ritual..."
    )

    # Log final trade metrics summary if trade_metrics object exists (final chapter of the chronicle)
    if "trade_metrics" in globals() and hasattr(trade_metrics, "summary"):
        logger.info(
            "Attempting to log final Campaign Report (trade metrics) before Pyrmethus sleeps..."
        )
        trade_metrics.summary()

    if not exchange_instance or not trading_symbol:
        logger.warning(
            f"{Fore.YELLOW}Shutdown: Exchange portal or trading symbol not defined. Automated cleanup ritual might be limited.{Style.RESET_ALL}"
        )
    else:
        try:
            logger.warning(
                f"Shutdown: Dispelling any lingering spells (open orders) for '{trading_symbol}'..."
            )
            cancel_open_orders(
                exchange_instance, trading_symbol, "Bot Shutdown Cleanup"
            )
            time.sleep(1.5)  # Allow time for dispelling to process

            logger.warning(
                f"Shutdown: Scrying for active presence on '{trading_symbol}' to banish..."
            )
            current_pos_state = get_current_position(exchange_instance, trading_symbol)

            if (
                current_pos_state["side"] != CONFIG.pos_none
                and current_pos_state["qty"] > CONFIG.position_qty_epsilon
            ):  # If a presence is found
                logger.warning(
                    f"{Fore.YELLOW}Shutdown: Active {current_pos_state['side']} presence found for '{trading_symbol}'. Attempting emergency market banishment...{Style.RESET_ALL}"
                )
                close_order_details = close_position(
                    exchange_instance,
                    trading_symbol,
                    current_pos_state,
                    "Bot Shutdown Emergency Banishment",
                )

                if close_order_details:
                    logger.info(
                        f"{Fore.CYAN}Shutdown: Banishment order for '{trading_symbol}' placed. Performing final presence check after delay...{Style.RESET_ALL}"
                    )
                    time.sleep(
                        CONFIG.post_close_delay_seconds * 2
                    )  # Longer delay for final check
                    final_pos_check_state = get_current_position(
                        exchange_instance, trading_symbol
                    )
                    if final_pos_check_state["side"] == CONFIG.pos_none:
                        logger.success(
                            f"{Fore.GREEN}Shutdown: Presence for '{trading_symbol}' confirmed BANISHED successfully.{Style.RESET_ALL}"
                        )
                        send_sms_alert(
                            f"[{market_base_name}/{strategy_name_for_sms}] Presence on {trading_symbol} successfully banished during shutdown."
                        )
                    else:
                        logger.critical(
                            f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN ERROR: FAILED TO CONFIRM presence banishment for '{trading_symbol}'! "
                            f"Final State: {final_pos_check_state['side']} Qty={final_pos_check_state['qty']}. MANUAL INTERVENTION REQUIRED! The entity may linger!{Style.RESET_ALL}"
                        )
                        send_sms_alert(
                            f"[{market_base_name}/{strategy_name_for_sms}] CRITICAL: FAILED to confirm {trading_symbol} presence banishment on shutdown. MANUAL CHECK REQUIRED!"
                        )
                else:
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL SHUTDOWN ERROR: FAILED TO PLACE banishment order for active presence on '{trading_symbol}'! MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base_name}/{strategy_name_for_sms}] CRITICAL: FAILED to place banishment order for {trading_symbol} on shutdown. MANUAL CHECK REQUIRED!"
                    )
            else:
                logger.info(
                    f"{Fore.GREEN}Shutdown: No active presence found for '{trading_symbol}'. Clean exit regarding market positions.{Style.RESET_ALL}"
                )
        except Exception as e_cleanup_error:
            logger.error(
                f"{Fore.RED}Shutdown Error: Disturbance during cleanup ritual for '{trading_symbol}': {e_cleanup_error}{Style.RESET_ALL}"
            )
            logger.debug(
                traceback.format_exc()
            )  # Log the full incantation of the error
            send_sms_alert(
                f"[{market_base_name}/{strategy_name_for_sms}] ERROR during shutdown cleanup for {trading_symbol}: {type(e_cleanup_error).__name__}. MANUAL SCRYING MAY BE NEEDED."
            )

    logger.info(
        f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Spell Shutdown Sequence Complete. Energies Withdrawn. ---{Style.RESET_ALL}"
    )


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    """
    Main function to initialize and run the Pyrmethus trading bot.
    Manages the main trading loop, exception handling, and graceful shutdown.
    This is where the Pyrmethus spell is ignited.
    """
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.3.0 Awakening ({start_time_readable}) ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}--- Active Path of Magic Conjured: {CONFIG.strategy_name.value} ---{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.GREEN}--- Primary Protective Wards Enchanted: ATR-based Fixed Stop Loss + Exchange-Native Adaptive Shield (TSL - Bybit V5) ---{Style.RESET_ALL}"
    )
    logger.warning(
        f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING - EXTREME RISK - EDUCATIONAL PURPOSES ONLY - HANDLE WITH CARE !!! ---{Style.RESET_ALL}"
    )

    # Initialize to None, will be set in try block
    current_exchange_instance: Optional[ccxt.Exchange] = (
        None  # The portal to the exchange realm
    )
    unified_trading_symbol: Optional[str] = (
        None  # The true name of the trading instrument
    )
    should_run_bot: bool = True  # Flag to keep the spell active
    trading_cycle_count: int = 0  # Counter for spell cycles

    try:
        current_exchange_instance = initialize_exchange()  # Open the portal
        if not current_exchange_instance:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Exchange portal initialization failed. Pyrmethus cannot proceed with spellcasting. Returning to slumber.{Style.RESET_ALL}"
            )
            return  # Exit if exchange cannot be initialized

        # Validate and unify symbol format using CCXT market data (confirm true name)
        market_details = current_exchange_instance.market(CONFIG.symbol)
        if not market_details:
            raise ValueError(
                f"Market for symbol '{CONFIG.symbol}' not found on {current_exchange_instance.id}. Ensure it's a valid realm name."
            )
        unified_trading_symbol = market_details["symbol"]  # Use CCXT's unified symbol

        if not market_details.get(
            "contract", False
        ):  # Ensure it's a futures/contract market (the correct plane of existence)
            raise ValueError(
                f"Market '{unified_trading_symbol}' is not a contract market. Pyrmethus is designed for futures trading on this plane."
            )

        logger.info(
            f"{Fore.GREEN}Spell focused on symbol: {unified_trading_symbol} (Type: {market_details.get('type', 'N/A')}, ID: {market_details.get('id', 'N/A')}){Style.RESET_ALL}"
        )

        # Set leverage (amplify power); halt if unsuccessful
        if not set_leverage(
            current_exchange_instance, unified_trading_symbol, CONFIG.leverage
        ):
            raise RuntimeError(
                f"Failed to amplify power (leverage) to {CONFIG.leverage}x for {unified_trading_symbol}. Halting spell."
            )

        # Log key configuration summary (the spell's main parameters)
        logger.info(
            f"{Fore.MAGENTA}--- Key Spell Configuration Runes ---{Style.RESET_ALL}"
        )
        logger.info(
            f"Trading Symbol: {unified_trading_symbol}, Candle Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x"
        )
        logger.info(f"Active Path of Magic: {CONFIG.strategy_name.value}")
        logger.info(
            f"Risk Per Trade (Mana Commitment): {CONFIG.risk_per_trade_percentage:.2%}, Max Position Value (USDT): {CONFIG.max_order_usdt_amount}"
        )
        logger.info(
            f"ATR SL Ward Multiplier: {CONFIG.atr_stop_loss_multiplier}, Adaptive Shield (TSL) Percent: {CONFIG.trailing_stop_percentage:.3%} (Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.3%})"
        )
        logger.info(
            f"Account Vitality Ward (Max Margin Ratio): {CONFIG.max_account_margin_ratio:.0%}"
        )
        logger.info(
            f"Volume Surge Filter: {'Enabled' if CONFIG.require_volume_spike_for_entry else 'Disabled'}, OB Concordance Long Thresh: {CONFIG.order_book_ratio_threshold_long}, OB Concordance Short Thresh: {CONFIG.order_book_ratio_threshold_short}"
        )
        logger.info(f"{Fore.MAGENTA}{'-' * 70}{Style.RESET_ALL}")  # Separator

        market_base_name_for_sms = unified_trading_symbol.split("/")[0].split(":")[0]
        send_sms_alert(
            f"[{market_base_name_for_sms}/{CONFIG.strategy_name.value}] Pyrmethus v2.3.0 Awakened. Symbol: {unified_trading_symbol}. Commencing spell cycles."
        )

        # --- Main Trading Loop (The Continuous Weaving of the Spell) ---
        while should_run_bot:
            cycle_start_time_monotonic = time.monotonic()
            trading_cycle_count += 1
            logger.debug(
                f"{Fore.CYAN}--- Pyrmethus Spell Cycle {trading_cycle_count} Commencing ({time.strftime('%H:%M:%S %Z')}) ---{Style.RESET_ALL}"
            )

            # Perform account health check (Vitality Ward) at the start of each cycle
            if not check_account_health(current_exchange_instance, CONFIG):
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}ACCOUNT VITALITY WARD BREACHED! Bot operations paused for safety. Will re-assess vitality next cycle.{Style.RESET_ALL}"
                )
                time.sleep(
                    CONFIG.sleep_seconds * 10
                )  # Longer sleep if health check fails, allowing mana to recover
                continue  # Skip to next cycle iteration

            try:
                # Determine data limit based on longest indicator lookback (how far back to scry for reliable omens)
                # This ensures enough data for all indicators to calculate properly.
                indicator_lookback_periods = [
                    CONFIG.st_atr_length,
                    CONFIG.confirm_st_atr_length,
                    CONFIG.stochrsi_rsi_length
                    + CONFIG.stochrsi_stoch_length
                    + CONFIG.stochrsi_d_period,
                    CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length,
                    max(CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
                    + CONFIG.ehlers_ssf_poles,
                    CONFIG.atr_calculation_period,
                    CONFIG.volume_ma_period,
                ]
                # Add buffer for API fetch limits and initial NaNs from indicators (extra scroll length)
                data_fetch_limit = (
                    max(
                        100,
                        max(indicator_lookback_periods)
                        if indicator_lookback_periods
                        else 100,
                    )
                    + CONFIG.api_fetch_limit_buffer
                    + 20
                )

                # Get market data (potentially cached scroll of omens)
                df_market_candles = get_market_data(
                    current_exchange_instance,
                    unified_trading_symbol,
                    CONFIG.interval,
                    limit=data_fetch_limit,
                )

                if df_market_candles is not None and not df_market_candles.empty:
                    trade_logic(
                        current_exchange_instance,
                        unified_trading_symbol,
                        df_market_candles,
                    )  # Weave the core spell
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Skipping spell-weaving for cycle {trading_cycle_count}: Invalid or missing market scroll for '{unified_trading_symbol}'. The omens are unreadable.{Style.RESET_ALL}"
                    )

            # Handle specific CCXT exceptions for robustness against exchange disturbances
            except ccxt.RateLimitExceeded as e_rate_limit:
                logger.warning(
                    f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded by Exchange (Mana Overload): {e_rate_limit}. Sleeping for an extended duration to let the aether calm...{Style.RESET_ALL}"
                )
                time.sleep(
                    CONFIG.sleep_seconds * 6
                )  # Longer sleep for rate limit issues
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
            ) as e_network_conn:
                logger.warning(
                    f"{Fore.YELLOW}Network/Exchange Connectivity Disturbance: {e_network_conn}. Retrying after a pause to re-attune.{Style.RESET_ALL}"
                )
                sleep_multiplier = (
                    6 if isinstance(e_network_conn, ccxt.ExchangeNotAvailable) else 3
                )
                time.sleep(CONFIG.sleep_seconds * sleep_multiplier)
            except ccxt.AuthenticationError as e_auth_runtime:  # Catch auth errors that might occur during runtime (portal keys revoked)
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}FATAL RUNTIME ERROR: Authentication Disturbance: {e_auth_runtime}. API Portal Keys may have been revoked or expired. Stopping spell.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base_name_for_sms}/{CONFIG.strategy_name.value}] CRITICAL: Runtime Authentication Error! Bot stopping. Check API Portal Keys/permissions."
                )
                should_run_bot = False  # Stop the bot
            except Exception as e_main_loop_unexpected:  # Catch any other unexpected errors in the main loop
                logger.exception(
                    f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL DISTURBANCE in Main Spell Loop (Cycle {trading_cycle_count}): {e_main_loop_unexpected} !!! Attempting to stop gracefully.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base_name_for_sms}/{CONFIG.strategy_name.value}] CRITICAL UNEXPECTED ERROR: {type(e_main_loop_unexpected).__name__}! Bot stopping."
                )
                should_run_bot = False  # Stop the bot

            # Calculate sleep duration for the end of the cycle (allow Pyrmethus to rest and mana to regenerate)
            if should_run_bot:
                elapsed_cycle_processing_time = (
                    time.monotonic() - cycle_start_time_monotonic
                )
                sleep_duration_this_cycle = max(
                    0, CONFIG.sleep_seconds - elapsed_cycle_processing_time
                )
                logger.debug(
                    f"Spell Cycle {trading_cycle_count} processed in {elapsed_cycle_processing_time:.2f}s. Resting for {sleep_duration_this_cycle:.2f}s until next cycle."
                )
                if sleep_duration_this_cycle > 0:
                    time.sleep(sleep_duration_this_cycle)

    except (
        KeyboardInterrupt
    ):  # Handle manual shutdown (Ctrl+C - the master's call to cease)
        logger.warning(
            f"\n{Fore.YELLOW}{Style.BRIGHT}Master's Call (KeyboardInterrupt) detected. Initiating graceful shutdown of Pyrmethus...{Style.RESET_ALL}"
        )
        should_run_bot = False  # Signal loop to stop
    except Exception as startup_or_config_error:  # Catch errors during initial setup before main loop (flaw in initial incantation)
        logger.critical(
            f"{Back.RED}{Fore.WHITE}CRITICAL AWAKENING ERROR: {startup_or_config_error}{Style.RESET_ALL}"
        )
        logger.debug(
            traceback.format_exc()
        )  # Log full traceback of the flawed incantation
        sms_alert_msg_startup_fail = f"[Pyrmethus] CRITICAL AWAKENING ERROR: {type(startup_or_config_error).__name__}. Bot failed to awaken properly."
        # Attempt to send SMS even if full config not loaded, if basic SMS settings might be available
        if (
            "CONFIG" in globals()
            and hasattr(CONFIG, "enable_sms_alerts")
            and CONFIG.enable_sms_alerts
            and hasattr(CONFIG, "sms_recipient_number")
            and CONFIG.sms_recipient_number
        ):
            send_sms_alert(sms_alert_msg_startup_fail)
        else:  # Fallback if SMS cannot be sent
            print(
                f"CRITICAL AWAKENING SMS (Simulated as print): {sms_alert_msg_startup_fail}"
            )
        should_run_bot = False  # Ensure bot doesn't run
    finally:
        # Graceful shutdown attempts to close positions/orders (withdraw energies and seal the portal)
        graceful_shutdown(current_exchange_instance, unified_trading_symbol)
        logger.info(
            f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated. Session Ended. Pyrmethus returns to slumber. ---{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    main()  # Ignite the spell
