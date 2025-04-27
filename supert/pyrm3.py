#!/usr/bin/env python

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.3.0 (Refactored & Robust)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and improved robustness.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.3.0 (Unified: Selectable Strategies + Precision + Native SL/TSL + Refactoring + Robustness).

Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS".
- Optimized Indicator Calculation: Only calculates indicators needed for the selected strategy (+ATR/Vol).
- Enhanced Precision: Uses Decimal for critical financial calculations with explicit rounding.
- Native SL/TSL Placement: Places exchange-native orders immediately after entry.
- Robust SL/TSL Handling: Improved checks and alerts for SL/TSL placement success/failure.
- Refactored Order Placement: `place_risked_market_order` broken into helper functions.
- ATR for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation.
- Risk-based position sizing with margin checks and explicit ROUND_DOWN for quantity.
- Termux SMS alerts for critical events and trade actions.
- Robust error handling (NaN checks, DivisionByZero, API errors) and logging with Neon color support.
- Graceful shutdown on KeyboardInterrupt with position/order closing attempt.
- Stricter position detection logic (Bybit V5 API).
- Improved clarity via constants and type hints. Configurable SL/TSL trigger types.

Disclaimer:
- **EXTREME RISK**: Educational purposes ONLY. High-risk. Use at own absolute risk.
- **NATIVE SL/TSL DEPENDENCE**: Relies on exchange-native orders. Subject to exchange performance, slippage, API reliability.
- Parameter Sensitivity: Requires significant tuning and testing.
- API Rate Limits: Monitor usage.
- Slippage: Market orders are prone to slippage.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: Requires Termux:API app and package.
- API Changes: Code targets Bybit V5 via CCXT, updates may be needed.
"""

# Standard Library Imports
import logging
import os
import subprocess
import sys
import time
import traceback
from decimal import (
    ROUND_DOWN,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    getcontext,
)

# import shlex # No longer needed with basic sanitization
from typing import Any, Literal

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = e.name
    # Corrected dependency list (removed requests)
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 28  # Set Decimal precision globally (adjust if needed)
DECIMAL_QUANTIZE_EXP = Decimal('1e-8')  # Default quantization exponent (8 decimal places)


# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables."""
    # Define constants for sides and position status
    SIDE_BUY: Literal['buy'] = "buy"
    SIDE_SELL: Literal['sell'] = "sell"
    POS_LONG: Literal['Long'] = "Long"
    POS_SHORT: Literal['Short'] = "Short"
    POS_NONE: Literal['None'] = "None"
    # Trigger types (aligned with Bybit)
    TRIGGER_BY_LAST: Literal['LastPrice'] = "LastPrice"
    TRIGGER_BY_MARK: Literal['MarkPrice'] = "MarkPrice"
    TRIGGER_BY_INDEX: Literal['IndexPrice'] = "IndexPrice"
    VALID_TRIGGER_TYPES: list[str] = [TRIGGER_BY_LAST, TRIGGER_BY_MARK, TRIGGER_BY_INDEX]

    def __init__(self) -> None:
        """Initializes the configuration by loading environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes (v2.3.0) ---{Style.RESET_ALL}")
        # --- API Credentials ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW, min_val=1, max_val=100)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW, min_val=1)
        self.order_retry_delay: int = self._get_env("ORDER_RETRY_DELAY", 2, cast_type=int, color=Fore.YELLOW, min_val=1)  # For potential future retries

        # --- Strategy Selection ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: list[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options: {self.valid_strategies}")
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid: {self.valid_strategies}")
        logger.info(f"Selected Strategy: {Fore.CYAN}{self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.0001"), max_val=Decimal("0.1"))  # 0.01% to 10%
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.1"))
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("1.0"))
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("1.0"))  # Must be >= 1

        # --- SL/TSL Configuration ---
        self.sl_trigger_by: str = self._get_env("SL_TRIGGER_BY", self.TRIGGER_BY_LAST, color=Fore.GREEN)
        if self.sl_trigger_by not in self.VALID_TRIGGER_TYPES: raise ValueError(f"Invalid SL_TRIGGER_BY: {self.sl_trigger_by}")
        self.tsl_trigger_by: str = self._get_env("TSL_TRIGGER_BY", self.TRIGGER_BY_LAST, color=Fore.GREEN)
        if self.tsl_trigger_by not in self.VALID_TRIGGER_TYPES: raise ValueError(f"Invalid TSL_TRIGGER_BY: {self.tsl_trigger_by}")

        # --- Trailing Stop Loss (Exchange Native) ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.0001"), max_val=Decimal("0.1"))  # 0.01% to 10% trail
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN, min_val=Decimal("0.0001"), max_val=Decimal("0.05"))  # 0.01% to 5% activation offset

        # --- Indicator Parameters (Common) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN, min_val=2)
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW, min_val=2)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW, min_val=Decimal("0.1"))
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW, min_val=1, max_val=50)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW, min_val=Decimal("0"))
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW, min_val=Decimal("0"))
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW)

        # --- Strategy Specific Parameters ---
        # Only load parameters for the *selected* strategy to avoid clutter and potential errors
        if self.strategy_name == "DUAL_SUPERTREND":
            self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN, min_val=2)
            self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN, min_val=Decimal("0.1"))
            self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN, min_val=2)
            self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN, min_val=Decimal("0.1"))
        elif self.strategy_name == "STOCHRSI_MOMENTUM":
            self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN, min_val=2)
            self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN, min_val=2)
            self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN, min_val=1)
            self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN, min_val=1)
            self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN, min_val=Decimal("50"), max_val=Decimal("100"))
            self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN, min_val=Decimal("0"), max_val=Decimal("50"))
            self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN, min_val=1)
        elif self.strategy_name == "EHLERS_FISHER":
            self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN, min_val=2)
            self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN, min_val=1)
        elif self.strategy_name == "EHLERS_MA_CROSS":
            self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN, min_val=2)
            self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN, min_val=3)
            if self.ehlers_fast_period >= self.ehlers_slow_period: raise ValueError("EHLERS_FAST_PERIOD must be less than EHLERS_SLOW_PERIOD")

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: str | None = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA, min_val=5)

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = 10000
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW, min_val=5)

        # --- Internal Constants ---
        self.USDT_SYMBOL: str = "USDT"
        self.RETRY_COUNT: int = 3
        self.RETRY_DELAY_SECONDS: int = self.order_retry_delay  # Use configured delay
        self.API_FETCH_LIMIT_BUFFER: int = 50
        self.POSITION_QTY_EPSILON: Decimal = Decimal("1e-9")
        self.POST_CLOSE_DELAY_SECONDS: int = 3
        self.MARKET_ORDER_FILL_CHECK_INTERVAL: float = 0.5  # Seconds between fill checks

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned Successfully ---{Style.RESET_ALL}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE, min_val: int | Decimal | None = None, max_val: int | Decimal | None = None) -> Any:
        """Fetches env var, casts, validates ranges, logs, handles defaults/errors."""
        value_str = os.getenv(key)
        value: Any = None
        log_source = ""

        if value_str is not None:
            log_source = f"(from env: '{value_str}')"
            try:
                if cast_type == bool: value = value_str.lower() in ['true', '1', 'yes', 'y']
                elif cast_type == Decimal: value = Decimal(value_str)
                elif cast_type is int: value = int(value_str)
                elif cast_type is float: value = float(value_str)  # Add float handling if needed
                elif cast_type is str: value = str(value_str)
                else: value = cast_type(value_str)  # For other types if necessary

                # Validate numeric ranges after successful casting
                if value is not None and cast_type in [int, Decimal, float]:
                    is_adjusted = False
                    original_value = value
                    if min_val is not None and value < min_val: value = min_val; is_adjusted = True
                    if max_val is not None and value > max_val: value = max_val; is_adjusted = True
                    if is_adjusted: logger.warning(f"{Fore.YELLOW}Value '{original_value}' for {key} out of range [{min_val}, {max_val}]. Adjusted to {value}.{Style.RESET_ALL}")

            except (ValueError, TypeError, InvalidOperation) as e:
                logger.error(f"{Fore.RED}Invalid type/value for {key}: '{value_str}'. Expected {cast_type.__name__}. Error: {e}. Using default: '{default}'{Style.RESET_ALL}")
                value = default
                log_source = f"(env parse error, using default: '{default}')"
        else:
            value = default
            log_source = f"(not set, using default: '{default}')" if default is not None else "(not set, no default)"

        if value is None and required:
            critical_msg = f"CRITICAL: Required env var '{key}' not set and no default."
            logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{critical_msg}{Style.RESET_ALL}")
            raise ValueError(critical_msg)

        # Final log of the processed value
        # Redact secret key from logs
        log_display_value = "****" if "SECRET" in key.upper() else value
        logger.debug(f"{color}Config {key}: {log_display_value} {log_source}{Style.RESET_ALL}")
        return value


# --- Logger Setup ---
# (Keep logger setup from v2.2.0)
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger: logging.Logger = logging.getLogger(__name__)
logger.propagate = False

SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs)  # type: ignore


logging.Logger.success = log_success  # type: ignore[attr-defined]

if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")

# --- Global Objects ---
try:
    CONFIG = Config()
except ValueError as e:
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Configuration Error: {e}{Style.RESET_ALL}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Unexpected Error initializing configuration: {e}{Style.RESET_ALL}")
    logger.debug(traceback.format_exc())
    sys.exit(1)


# --- Helper & Utility Functions ---
# (Keep safe_decimal_conversion, format_order_id, get_market_base_currency, format_price, format_amount, send_sms_alert, initialize_exchange as refined in v2.2.0 analysis)
# --- safe_decimal_conversion ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("NaN")) -> Decimal:
    """Safely converts a value to Decimal, returning Decimal('NaN') if conversion fails."""
    if value is None: return default
    try: return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, returning {default}")
        return default


# --- format_order_id ---
def format_order_id(order_id: str | int | None) -> str:
    """Returns a shortened representation of an order ID or 'N/A'."""
    return f"...{str(order_id)[-6:]}" if order_id else "N/A"


# --- get_market_base_currency ---
def get_market_base_currency(symbol: str) -> str:
    """Extracts the base currency from a symbol (e.g., 'BTC' from 'BTC/USDT:USDT')."""
    try: return symbol.split('/')[0]
    except IndexError: logger.warning(f"Could not parse base currency from symbol: {symbol}"); return symbol


# --- format_price ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal | str) -> str:
    """Formats a price string according to market precision rules using CCXT."""
    try: return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price '{price}' for {symbol}: {e}{Style.RESET_ALL}")
        try: return str(Decimal(str(price)).normalize())
        except: return str(price)


# --- format_amount ---
def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal | str, round_direction=ROUND_DOWN) -> str:
    """Formats an amount (quantity) string according to market precision rules."""
    try: return exchange.amount_to_precision(symbol, float(amount))  # CCXT usually handles rounding correctly
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount '{amount}' for {symbol}: {e}{Style.RESET_ALL}")
        try:  # Fallback with explicit rounding
            precision_str = exchange.markets[symbol].get('precision', {}).get('amount')
            if precision_str:
                decimal_places = Decimal(str(precision_str)).normalize().as_tuple().exponent * -1
                return str(Decimal(str(amount)).quantize(Decimal('1e-' + str(decimal_places)), rounding=round_direction))
            return str(Decimal(str(amount)).quantize(DECIMAL_QUANTIZE_EXP, rounding=round_direction))
        except: return str(amount)


# --- send_sms_alert ---
def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux:API if enabled and configured."""
    if not CONFIG.enable_sms_alerts: logger.debug("SMS alerts disabled."); return False
    if not CONFIG.sms_recipient_number: logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set."); return False
    try:
        safe_message = message.replace('"', "'").replace("`", "'").replace("$", "")
        command: list[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, safe_message]
        logger.info(f"{Fore.MAGENTA}Attempting SMS to {'*****' + CONFIG.sms_recipient_number[-4:]} (Timeout: {CONFIG.sms_timeout_seconds}s): \"{safe_message[:60]}...\"{Style.RESET_ALL}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if result.returncode == 0: logger.success(f"{Fore.MAGENTA}SMS command executed successfully.{Style.RESET_ALL}"); return True
        else: logger.error(f"{Fore.RED}SMS command failed. RC: {result.returncode}, Stderr: {result.stderr.strip() or 'N/A'}{Style.RESET_ALL}"); return False
    except FileNotFoundError: logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}"); return False
    except subprocess.TimeoutExpired: logger.error(f"{Fore.RED}SMS failed: Command timed out ({CONFIG.sms_timeout_seconds}s).{Style.RESET_ALL}"); return False
    except Exception as e: logger.error(f"{Fore.RED}SMS failed: Unexpected error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); return False


# --- initialize_exchange ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance."""
    logger.info(f"{Fore.BLUE}Initializing CCXT Bybit connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret: logger.critical("API Key or Secret missing."); send_sms_alert("[Pyrmethus] CRITICAL: API keys missing."); return None
    try:
        exchange = ccxt.bybit({'apiKey': CONFIG.api_key, 'secret': CONFIG.api_secret, 'enableRateLimit': True, 'options': {'defaultType': 'linear', 'recvWindow': CONFIG.default_recv_window, 'adjustForTimeDifference': True, 'verbose': LOGGING_LEVEL <= logging.DEBUG}})
        logger.debug("Loading markets (forced reload)..."); exchange.load_markets(True)
        logger.debug("Performing initial balance check..."); exchange.fetch_balance()
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session Initialized (LIVE SCALPING MODE!).{Style.RESET_ALL}"); send_sms_alert("[Pyrmethus] Initialized & auth OK.")
        return exchange
    except ccxt.AuthenticationError as e: logger.critical(f"Authentication failed: {e}. Check keys/IP/perms."); send_sms_alert(f"[Pyrmethus] CRITICAL: Auth FAILED: {e}.")
    except ccxt.NetworkError as e: logger.critical(f"Network error on init: {e}. Check connection."); send_sms_alert(f"[Pyrmethus] CRITICAL: Network Error Init: {e}.")
    except ccxt.ExchangeError as e: logger.critical(f"Exchange error on init: {e}. Check Bybit status."); send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error Init: {e}.")
    except Exception as e: logger.critical(f"Unexpected error during init: {e}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}.")
    return None


# --- Indicator Calculation Functions ---
# (Using refined versions from v2.2.0 analysis)
# --- calculate_supertrend ---
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta."""
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    st_col = f"SUPERT_{length}_{float(multiplier)}"
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]
    min_required_len = length + 1
    for col in target_cols: df[col] = pd.NA  # Initialize cols
    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_required_len:
        logger.warning(f"{Fore.YELLOW}Indicator ({col_prefix}ST): Input invalid/short (Len: {len(df) if df is not None else 0}, Need: {min_required_len}).{Style.RESET_ALL}")
        return df
    try:
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if st_col not in df.columns or st_trend_col not in df.columns: raise KeyError(f"Missing raw columns: {st_col}, {st_trend_col}")
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        df[f"{col_prefix}trend"] = (df[st_trend_col] == 1).astype('boolean')
        prev_trend = df[st_trend_col].shift(1)
        df[f"{col_prefix}st_long"] = ((prev_trend == -1) & (df[st_trend_col] == 1)).astype('boolean')
        df[f"{col_prefix}st_short"] = ((prev_trend == 1) & (df[st_trend_col] == -1)).astype('boolean')
        raw_cols = [st_col, st_trend_col, st_long_col, st_short_col]
        df.drop(columns=[col for col in raw_cols if col in df.columns], inplace=True)
        # ... (optional logging of last valid value) ...
    except Exception as e: logger.error(f"{Fore.RED}Indicator ({col_prefix}ST): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols]  # type: ignore
    return df


# --- analyze_volume_atr ---
def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume MA, and volume ratio."""
    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]; min_len = max(atr_len, vol_ma_len) + 1
    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator (Vol/ATR): Input invalid/short (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}"); return results
    try:
        atr_col = f"ATRr_{atr_len}"; df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns: results["atr"] = safe_decimal_conversion(df[atr_col].iloc[-1]); df.drop(columns=[atr_col], errors='ignore', inplace=True)
        else: logger.warning(f"ATR col '{atr_col}' not found.")
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        results["volume_ma"] = safe_decimal_conversion(df[volume_ma_col].iloc[-1])
        results["last_volume"] = safe_decimal_conversion(df['volume'].iloc[-1])
        df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)
        if not results["volume_ma"].is_nan() and results["volume_ma"] > CONFIG.POSITION_QTY_EPSILON and not results["last_volume"].is_nan():
            try: results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except (DivisionByZero, InvalidOperation): results["volume_ratio"] = Decimal('NaN')
        else: results["volume_ratio"] = Decimal('NaN')
        for k, v in results.items():  # Convert NaN to None for easier checks
            if isinstance(v, Decimal) and v.is_nan(): results[k] = None
        # ... (debug logging) ...
    except Exception as e: logger.error(f"{Fore.RED}Indicator (Vol/ATR): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); results = dict.fromkeys(results)
    return results


# --- calculate_stochrsi_momentum ---
def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """Calculates StochRSI (K and D lines) and Momentum indicator."""
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']; min_len = max(rsi_len + stoch_len + max(k, d), mom_len) + 5
    for col in target_cols: df[col] = pd.NA
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator (StochRSI/Mom): Input invalid/short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"); return df
    try:
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col, d_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if k_col in stochrsi_df.columns: df['stochrsi_k'] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else: logger.warning(f"StochRSI K col '{k_col}' not found.")
        if d_col in stochrsi_df.columns: df['stochrsi_d'] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else: logger.warning(f"StochRSI D col '{d_col}' not found.")
        mom_col = f"MOM_{mom_len}"; df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns: df['momentum'] = df[mom_col].apply(safe_decimal_conversion); df.drop(columns=[mom_col], errors='ignore', inplace=True)
        else: logger.warning(f"Momentum col '{mom_col}' not found.")
        # ... (debug logging) ...
    except Exception as e: logger.error(f"{Fore.RED}Indicator (StochRSI/Mom): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols]  # type: ignore
    return df


# --- calculate_ehlers_fisher ---
def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates the Ehlers Fisher Transform indicator."""
    target_cols = ['ehlers_fisher', 'ehlers_signal']; min_len = length + signal + 5
    for col in target_cols: df[col] = pd.NA
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator (EhlersFisher): Input invalid/short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"); return df
    try:
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col, sig_col = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fish_col in fisher_df.columns: df['ehlers_fisher'] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else: logger.warning(f"Ehlers Fisher col '{fish_col}' not found.")
        if sig_col in fisher_df.columns: df['ehlers_signal'] = fisher_df[sig_col].apply(safe_decimal_conversion)
        # If signal=1, pandas_ta might not create signal col, default is NaN which is handled
        # ... (debug logging) ...
    except Exception as e: logger.error(f"{Fore.RED}Indicator (EhlersFisher): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols]  # type: ignore
    return df


# --- calculate_ehlers_ma ---
def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers MAs (Placeholder using EMA)."""
    target_cols = ['fast_ema', 'slow_ema']; min_len = max(fast_len, slow_len) + 5
    for col in target_cols: df[col] = pd.NA
    if df is None or df.empty or "close" not in df.columns or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator (EhlersMA): Input invalid/short (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"); return df
    try:
        logger.warning(f"{Fore.YELLOW}Using standard EMA as placeholder for Ehlers MAs. Review if accurate Ehlers needed.{Style.RESET_ALL}")
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)
        # ... (debug logging) ...
    except Exception as e: logger.error(f"{Fore.RED}Indicator (EhlersMA Placeholder): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); [df.__setitem__(c, pd.NA) for c in target_cols]  # type: ignore
    return df


# --- Order Book Analysis ---
# (Keep analyze_order_book refined version from v2.2.0)
def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes the L2 order book for bid/ask pressure and spread."""
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}; log_prefix = "Order Book"
    logger.debug(f"{log_prefix}: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")
    if not exchange.has.get('fetchL2OrderBook'): logger.warning(f"{Fore.YELLOW}{log_prefix}: fetchL2OrderBook not supported.{Style.RESET_ALL}"); return results
    try:
        ob = exchange.fetch_l2_order_book(symbol, limit=fetch_limit); bids, asks = ob.get('bids', []), ob.get('asks', [])
        if not bids or not asks: logger.warning(f"{Fore.YELLOW}{log_prefix}: Fetched empty bids/asks for {symbol}.{Style.RESET_ALL}"); return results
        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else Decimal('NaN')
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else Decimal('NaN')
        results["best_bid"] = None if best_bid.is_nan() else best_bid
        results["best_ask"] = None if best_ask.is_nan() else best_ask
        if results["best_bid"] and results["best_ask"]:
            if results["best_ask"] > results["best_bid"]: results["spread"] = results["best_ask"] - results["best_bid"]; logger.debug(f"{log_prefix}: Bid={results['best_bid']:.4f}, Ask={results['best_ask']:.4f}, Spread={results['spread']:.4f}")
            else: logger.warning(f"{Fore.YELLOW}{log_prefix}: Bid >= Ask. Spread invalid.{Style.RESET_ALL}"); results["spread"] = None
        else: logger.debug(f"{log_prefix}: Bid={results['best_bid'] or 'N/A'}, Ask={results['best_ask'] or 'N/A'} (Spread N/A)")
        bid_vol = sum(safe_decimal_conversion(b[1], default=Decimal('0')) for b in bids[:depth] if len(b) > 1)
        ask_vol = sum(safe_decimal_conversion(a[1], default=Decimal('0')) for a in asks[:depth] if len(a) > 1)
        logger.debug(f"{log_prefix} (Depth {depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")
        if ask_vol > CONFIG.POSITION_QTY_EPSILON:
            try: results["bid_ask_ratio"] = bid_vol / ask_vol
            except: results["bid_ask_ratio"] = None
        else: results["bid_ask_ratio"] = None
        if results["bid_ask_ratio"]: logger.debug(f"{log_prefix} Ratio: {results['bid_ask_ratio']:.3f}")
        else: logger.debug(f"{log_prefix} Ratio: N/A (AskVol={ask_vol:.4f})")
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.warning(f"{Fore.YELLOW}{log_prefix}: API Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except (IndexError, TypeError, KeyError) as e: logger.warning(f"{Fore.YELLOW}{log_prefix}: Error parsing data for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Unexpected error for {symbol}: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); results = dict.fromkeys(results)
    return results


# --- Data Fetching ---
# (Keep get_market_data refined version from v2.2.0)
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data from the exchange."""
    log_prefix = "Data Fetch"
    if not exchange.has.get("fetchOHLCV"): logger.error(f"{Fore.RED}{log_prefix}: Exchange '{exchange.id}' no fetchOHLCV.{Style.RESET_ALL}"); return None
    try:
        logger.debug(f"{log_prefix}: Fetching {limit} OHLCV for {symbol} ({interval})..."); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv: logger.warning(f"{Fore.YELLOW}{log_prefix}: No OHLCV data returned for {symbol} ({interval}).{Style.RESET_ALL}"); return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty: logger.warning(f"{Fore.YELLOW}{log_prefix}: OHLCV for {symbol} empty DataFrame.{Style.RESET_ALL}"); return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True); df.set_index("timestamp", inplace=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        if df.isnull().values.any():
            nan_counts = df.isnull().sum(); logger.warning(f"{Fore.YELLOW}{log_prefix}: OHLCV has NaNs:\n{nan_counts[nan_counts > 0]}\nAttempting ffill...{Style.RESET_ALL}")
            df.ffill(inplace=True)
            if df.isnull().values.any(): logger.warning(f"{Fore.YELLOW}NaNs remain post-ffill, trying bfill...{Style.RESET_ALL}"); df.bfill(inplace=True)
            if df.isnull().values.any(): logger.error(f"{Fore.RED}{log_prefix}: NaNs persist post-fill. Cannot use.{Style.RESET_ALL}"); return None
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in numeric_cols): logger.error(f"{Fore.RED}{log_prefix}: Non-numeric data found post-process.{Style.RESET_ALL}"); return None
        logger.debug(f"{log_prefix}: Processed {len(df)} candles for {symbol}. Last: {df.index[-1]}"); return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.warning(f"{Fore.YELLOW}{log_prefix}: API Error fetching OHLCV: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Error processing OHLCV: {type(e).__name__} - {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
    return None


# --- Position & Order Management ---
# (Keep get_current_position, set_leverage, close_position, calculate_position_size, wait_for_order_fill refined versions from v2.2.0)
# --- get_current_position ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position details using Bybit V5 API specifics via CCXT."""
    default_pos: dict[str, Any] = {'side': CONFIG.POS_NONE, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market: dict | None = None; market_id: str | None = None; log_prefix = "Position Check"
    try: market = exchange.market(symbol); market_id = market['id']
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Failed get market info '{symbol}': {e}{Style.RESET_ALL}"); return default_pos
    if not market: return default_pos
    try:
        if not exchange.has.get('fetchPositions'): logger.warning(f"{Fore.YELLOW}{log_prefix}: fetchPositions not supported. Assuming no pos.{Style.RESET_ALL}"); return default_pos
        category = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else 'linear'); params = {'category': category}
        logger.debug(f"{log_prefix}: Fetching for {symbol} (ID: {market_id}, Cat: {category})...")
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        active_pos_data = None
        for pos in fetched_positions:
            info = pos.get('info', {}); pos_market_id = info.get('symbol'); idx = int(info.get('positionIdx', -1)); side_v5 = info.get('side', 'None').strip(); size_str = info.get('size')
            if pos_market_id == market_id and idx == 0 and side_v5 in ['Buy', 'Sell']:
                size = safe_decimal_conversion(size_str)
                if not size.is_nan() and abs(size) > CONFIG.POSITION_QTY_EPSILON: active_pos_data = pos; break
        if active_pos_data:
            try:
                info = active_pos_data.get('info', {}); size = safe_decimal_conversion(info.get('size')); entry_price = safe_decimal_conversion(info.get('avgPrice')); side_v5 = info.get('side')
                side = CONFIG.POS_LONG if side_v5 == 'Buy' else CONFIG.POS_SHORT
                if not size.is_nan() and not entry_price.is_nan() and entry_price >= 0:
                    pos_qty = abs(size)
                    if pos_qty > CONFIG.POSITION_QTY_EPSILON: logger.info(f"{Fore.YELLOW}{log_prefix}: Found ACTIVE {side}: Qty={pos_qty:.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}"); return {'side': side, 'qty': pos_qty, 'entry_price': entry_price}
                    else: logger.info(f"{log_prefix}: Pos size negligible ({pos_qty:.8f}). Flat.{Style.RESET_ALL}"); return default_pos
                else: logger.warning(f"{Fore.YELLOW}{log_prefix}: Invalid size/entry ({size}/{entry_price}). Flat.{Style.RESET_ALL}"); return default_pos
            except Exception as parse_err: logger.warning(f"{Fore.YELLOW}{log_prefix}: Error parsing pos data: {parse_err}. Data: {active_pos_data}{Style.RESET_ALL}"); return default_pos
        else: logger.info(f"{log_prefix}: No active One-Way pos found for {market_id}."); return default_pos
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.warning(f"{Fore.YELLOW}{log_prefix}: API Error fetching positions: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Unexpected error fetching positions: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
    return default_pos


# --- set_leverage ---
def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol using Bybit V5 specifics via CCXT."""
    log_prefix = "Leverage Setting"; logger.info(f"{Fore.CYAN}{log_prefix}: Attempting {leverage}x for {symbol}...{Style.RESET_ALL}")
    try: market = exchange.market(symbol); assert market.get('contract')
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Failed get market info/not contract '{symbol}': {e}{Style.RESET_ALL}"); return False
    for attempt in range(CONFIG.RETRY_COUNT):
        try:
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            logger.debug(f"{log_prefix}: Calling set_leverage: lev={leverage}, sym={symbol}, params={params}")
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}{log_prefix}: Set OK {leverage}x for {symbol}. Resp: {str(response)[:100]}...{Style.RESET_ALL}"); return True
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            if "leverage not modified" in err_str or "same as requested" in err_str or "110044" in err_str:
                logger.info(f"{Fore.CYAN}{log_prefix}: Already set {leverage}x for {symbol}.{Style.RESET_ALL}"); return True
            logger.warning(f"{Fore.YELLOW}{log_prefix}: Exchange error (Try {attempt + 1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
        except Exception as e: logger.warning(f"{Fore.YELLOW}{log_prefix}: Network/Other error (Try {attempt + 1}/{CONFIG.RETRY_COUNT}): {e}{Style.RESET_ALL}")
        if attempt < CONFIG.RETRY_COUNT - 1: time.sleep(CONFIG.RETRY_DELAY_SECONDS)
        else: logger.error(f"{Fore.RED}{log_prefix}: Failed after {CONFIG.RETRY_COUNT} attempts.{Style.RESET_ALL}")
    return False


# --- close_position ---
def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal") -> dict[str, Any] | None:
    """Closes the specified active position by placing a market order with reduceOnly=True."""
    initial_side = position_to_close.get('side', CONFIG.POS_NONE); initial_qty = position_to_close.get('qty', Decimal("0.0")); market_base = get_market_base_currency(symbol); log_prefix = "Close Position"
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{log_prefix}: Initiated for {symbol}. Reason: {reason}. Init State: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")
    logger.debug(f"{log_prefix}: Re-validating current position state...")
    live_position = get_current_position(exchange, symbol); live_pos_side = live_position['side']; live_qty = live_position['qty']
    if live_pos_side == CONFIG.POS_NONE or live_qty <= CONFIG.POSITION_QTY_EPSILON:
        logger.warning(f"{Fore.YELLOW}{log_prefix}: Re-validation shows NO active position. Aborting close.{Style.RESET_ALL}")
        if initial_side != CONFIG.POS_NONE: logger.warning(f"{Fore.YELLOW}{log_prefix}: Discrepancy: Bot thought {initial_side}, exchange reports None/Zero.{Style.RESET_ALL}")
        return None
    if live_pos_side != initial_side: logger.warning(f"{Fore.YELLOW}{log_prefix}: Discrepancy! Initial={initial_side}, Live={live_pos_side}. Closing live pos.{Style.RESET_ALL}")
    close_exec_side = CONFIG.SIDE_SELL if live_pos_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY
    try:
        amount_str = format_amount(exchange, symbol, live_qty); amount_dec = safe_decimal_conversion(amount_str); amount_float = float(amount_dec)
        if amount_dec <= CONFIG.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}{log_prefix}: Closing amount '{amount_str}' negligible. Aborting.{Style.RESET_ALL}"); return None
        logger.warning(f"{Back.YELLOW}{Fore.BLACK}{log_prefix}: Attempting CLOSE {live_pos_side} ({reason}): Exec {close_exec_side.upper()} MARKET {amount_str} {symbol} (reduceOnly=True)...{Style.RESET_ALL}")
        params = {'reduceOnly': True}; order = exchange.create_market_order(symbol=symbol, side=close_exec_side, amount=amount_float, params=params)
        order_id = order.get('id'); status = order.get('status', 'unknown'); filled_qty = safe_decimal_conversion(order.get('filled')); fill_price = safe_decimal_conversion(order.get('average')); cost = safe_decimal_conversion(order.get('cost')); fee = safe_decimal_conversion(order.get('fee', {}).get('cost', '0'))
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}{log_prefix}: Order ({reason}) PLACED/FILLED(?). ID: {format_order_id(order_id)}, Status: {status}. Filled: {filled_qty:.8f}/{amount_str}, AvgFill: {fill_price:.4f}, Cost: {cost:.2f}, Fee: {fee:.4f}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] Closed {live_pos_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:{format_order_id(order_id)}")
        return order
    except ccxt.InsufficientFunds as e: logger.error(f"{Fore.RED}{log_prefix} ({reason}): Insufficient Funds: {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insufficient Funds.")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        if "position size is zero" in err_str or "order would not reduce position size" in err_str or "110025" in err_str or "110045" in err_str:  # Example checks
             logger.warning(f"{Fore.YELLOW}{log_prefix} ({reason}): Exchange indicates position already closed/zero: {e}. Assuming closed.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] Close ({reason}): Already closed/zero reported."); return None
        else: logger.error(f"{Fore.RED}{log_prefix} ({reason}): Exchange error: {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Exchange Error: {type(e).__name__}.")
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} ({reason}): Failed: {type(e).__name__} - {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}.")
    return None


# --- calculate_position_size ---
def calculate_position_size(equity: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, stop_loss_price: Decimal, leverage: int, symbol: str, exchange: ccxt.Exchange) -> tuple[Decimal | None, Decimal | None]:
    """Calculates position size (base currency) and estimated margin based on risk."""
    log_prefix = "Risk Calc"; logger.debug(f"{log_prefix}: Eq={equity:.4f}, Risk%={(risk_per_trade_pct * 100):.3f}%, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")
    if not (entry_price > 0 and stop_loss_price > 0): logger.error(f"{Fore.RED}{log_prefix} Error: Invalid entry/SL price.{Style.RESET_ALL}"); return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}{log_prefix} Error: Entry/SL too close (Diff: {price_diff:.8f}).{Style.RESET_ALL}"); return None, None
    if not (0 < risk_per_trade_pct < 1): logger.error(f"{Fore.RED}{log_prefix} Error: Invalid risk %: {risk_per_trade_pct:.4%}.{Style.RESET_ALL}"); return None, None
    if equity <= 0: logger.error(f"{Fore.RED}{log_prefix} Error: Invalid equity: {equity:.4f}.{Style.RESET_ALL}"); return None, None
    if leverage <= 0: logger.error(f"{Fore.RED}{log_prefix} Error: Invalid leverage: {leverage}.{Style.RESET_ALL}"); return None, None
    try:
        risk_amount_usdt = equity * risk_per_trade_pct; quantity_raw = risk_amount_usdt / price_diff
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)  # Rounds down
        quantity_precise = safe_decimal_conversion(quantity_precise_str)
        if quantity_precise.is_nan() or quantity_precise <= CONFIG.POSITION_QTY_EPSILON:
            logger.warning(f"{Fore.YELLOW}{log_prefix} Warning: Qty negligible/zero after precision ({quantity_precise:.8f}). RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}.{Style.RESET_ALL}"); return None, None
        pos_value_usdt = quantity_precise * entry_price; required_margin = pos_value_usdt / Decimal(leverage)
        logger.debug(f"{log_prefix} Result: Qty={quantity_precise:.8f}, RiskAmt={risk_amount_usdt:.4f}, EstValue={pos_value_usdt:.4f}, EstMargin={required_margin:.4f}")
        return quantity_precise, required_margin
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Error: Calculation failed: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); return None, None


# --- wait_for_order_fill ---
def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> dict[str, Any] | None:
    """Waits for a specific order to reach 'closed' (filled) status."""
    start_time = time.monotonic(); order_id_short = format_order_id(order_id); log_prefix = f"Wait Fill ({order_id_short})"
    logger.info(f"{Fore.CYAN}{log_prefix}: Waiting for {symbol} order (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")
    while time.monotonic() - start_time < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol); status = order.get('status')
            logger.debug(f"{log_prefix} Status: {status}")
            if status == 'closed': logger.success(f"{Fore.GREEN}{log_prefix}: Confirmed FILLED.{Style.RESET_ALL}"); return order
            elif status in ['canceled', 'rejected', 'expired']: logger.error(f"{Fore.RED}{log_prefix}: Failed status '{status}'.{Style.RESET_ALL}"); return None
            time.sleep(CONFIG.MARKET_ORDER_FILL_CHECK_INTERVAL)
        except ccxt.OrderNotFound:
            elapsed = time.monotonic() - start_time
            logger.warning(f"{Fore.YELLOW}{log_prefix}: Not found yet ({elapsed:.1f}s). Retrying...{Style.RESET_ALL}"); time.sleep(0.5)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e: elapsed = time.monotonic() - start_time; logger.warning(f"{Fore.YELLOW}{log_prefix}: API Error ({elapsed:.1f}s): {type(e).__name__}. Retrying...{Style.RESET_ALL}"); time.sleep(CONFIG.RETRY_DELAY_SECONDS)
        except Exception as e: elapsed = time.monotonic() - start_time; logger.error(f"{Fore.RED}{log_prefix}: Unexpected error ({elapsed:.1f}s): {e}. Stop wait.{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); return None
    logger.error(f"{Fore.RED}{log_prefix}: Timeout ({timeout_seconds}s) waiting for fill.{Style.RESET_ALL}")
    try: final_check = exchange.fetch_order(order_id, symbol); logger.warning(f"{log_prefix}: Final status check on timeout: {final_check.get('status')}")
    except Exception as final_e: logger.warning(f"{log_prefix}: Final status check failed: {final_e}")
    return None


# --- Order Placement Refactoring ---
def _calculate_initial_sl_price(entry_price: Decimal, atr: Decimal, multiplier: Decimal, side: str, min_price: Decimal, symbol: str, exchange: ccxt.Exchange) -> Decimal | None:
    """Helper to calculate and validate the initial SL price."""
    log_prefix = "Place Order (Calc SL)"
    sl_distance = atr * multiplier
    if side == CONFIG.SIDE_BUY: raw_sl = entry_price - sl_distance
    else: raw_sl = entry_price + sl_distance

    if min_price > 0 and raw_sl < min_price:
        logger.warning(f"{Fore.YELLOW}{log_prefix}: Raw SL {raw_sl:.4f} below min {min_price:.4f}. Adjusting.{Style.RESET_ALL}")
        raw_sl = min_price
    elif raw_sl <= 0:
        logger.error(f"{Fore.RED}{log_prefix}: Calculated SL zero/negative ({raw_sl:.4f}). Cannot proceed.{Style.RESET_ALL}")
        return None

    sl_str = format_price(exchange, symbol, raw_sl)
    sl_price = safe_decimal_conversion(sl_str)
    if sl_price.is_nan():
         logger.error(f"{Fore.RED}{log_prefix}: Failed to format/convert SL price {sl_str}. Cannot proceed.{Style.RESET_ALL}")
         return None
    logger.info(f"{log_prefix}: Initial SL Price ~ {sl_price:.4f} (ATR Dist: {sl_distance:.4f})")
    return sl_price


def _validate_and_cap_quantity(
    calculated_qty: Decimal, entry_price: Decimal, leverage: int, max_cap_usdt: Decimal,
    min_qty_limit: Decimal, max_qty_limit: Decimal | None, symbol: str, exchange: ccxt.Exchange
) -> tuple[Decimal | None, Decimal | None]:
    """Helper to validate quantity against limits and cap, returns (final_qty, final_margin_est)."""
    log_prefix = "Place Order (Validate Qty)"
    final_qty = calculated_qty
    req_margin_est = (final_qty * entry_price) / Decimal(leverage)  # Initial estimate

    # Apply Max Value Cap
    pos_value_est = final_qty * entry_price
    if pos_value_est > max_cap_usdt:
        logger.warning(f"{Fore.YELLOW}{log_prefix}: Est. value {pos_value_est:.2f} USDT > Cap {max_cap_usdt:.2f}. Capping.{Style.RESET_ALL}")
        try: capped_raw = max_cap_usdt / entry_price
        except DivisionByZero: logger.error(f"{log_prefix}: Cannot cap qty, entry price estimate is zero."); return None, None
        capped_str = format_amount(exchange, symbol, capped_raw)  # Rounds down
        final_qty = safe_decimal_conversion(capped_str)
        if final_qty.is_nan(): logger.error(f"{log_prefix}: Failed convert capped qty str '{capped_str}'."); return None, None
        req_margin_est = (final_qty * entry_price) / Decimal(leverage)  # Recalculate margin
        logger.info(f"{log_prefix}: Qty capped to {final_qty:.8f}. New Est. Margin ~{req_margin_est:.4f} USDT")

    # Check Limits
    if final_qty <= CONFIG.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}{log_prefix}: Final qty ({final_qty:.8f}) negligible/zero. Abort.{Style.RESET_ALL}"); return None, None
    if min_qty_limit > 0 and final_qty < min_qty_limit: logger.error(f"{Fore.RED}{log_prefix}: Final qty {final_qty:.8f} < Min {min_qty_limit:.8f}. Abort.{Style.RESET_ALL}"); return None, None
    if max_qty_limit is not None and final_qty > max_qty_limit:
        logger.warning(f"{Fore.YELLOW}{log_prefix}: Final qty {final_qty:.8f} > Max {max_qty_limit:.8f}. Capping to max.{Style.RESET_ALL}")
        final_qty = max_qty_limit  # Already a Decimal
        final_qty = safe_decimal_conversion(format_amount(exchange, symbol, final_qty))  # Reformat
        if final_qty.is_nan(): logger.error(f"{log_prefix}: Failed convert max qty."); return None, None
        req_margin_est = (final_qty * entry_price) / Decimal(leverage)  # Recalculate margin

    return final_qty, req_margin_est


def _check_margin_availability(req_margin_est: Decimal, free_balance: Decimal, buffer: Decimal, side: str, market_base: str) -> bool:
    """Helper to check if sufficient free margin is available."""
    log_prefix = "Place Order (Margin Check)"
    buffered_req_margin = req_margin_est * buffer
    if free_balance < buffered_req_margin:
        logger.error(f"{Fore.RED}{log_prefix}: Insufficient FREE margin. Need ~{buffered_req_margin:.4f} USDT, Have {free_balance:.4f} USDT.{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{buffered_req_margin:.2f})")
        return False
    logger.info(f"{Fore.GREEN}{log_prefix}: OK. EstMargin={req_margin_est:.4f}, BufferedReq={buffered_req_margin:.4f}, Free={free_balance:.4f}{Style.RESET_ALL}")
    return True


def _place_and_confirm_entry_order(exchange: ccxt.Exchange, symbol: str, side: str, quantity: Decimal) -> dict[str, Any] | None:
    """Places the market entry order and waits for fill confirmation."""
    log_prefix = f"Place Order ({side.upper()} Entry)"
    entry_order_id: str | None = None
    try:
        qty_float = float(quantity)
        logger.warning(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** {log_prefix}: Placing MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
        entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params={'reduce_only': False})
        entry_order_id = entry_order.get('id')
        if not entry_order_id: raise ValueError("Entry order placement returned no ID.")
        logger.success(f"{Fore.GREEN}{log_prefix}: Order submitted. ID: {format_order_id(entry_order_id)}. Waiting for fill...{Style.RESET_ALL}")

        filled_entry = wait_for_order_fill(exchange, entry_order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry:
            logger.error(f"{Fore.RED}{log_prefix}: Order {format_order_id(entry_order_id)} did NOT fill or failed.{Style.RESET_ALL}")
            try: logger.warning(f"{log_prefix}: Attempting cancel for {format_order_id(entry_order_id)}..."); exchange.cancel_order(entry_order_id, symbol)
            except Exception as cancel_err: logger.warning(f"{log_prefix}: Cancel failed for {format_order_id(entry_order_id)}: {cancel_err}")
            return None
        return filled_entry
    except Exception as e:
        logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: FAILED TO PLACE/CONFIRM ENTRY: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        # If order ID was obtained but fill failed, try cancelling
        if entry_order_id:
            try: logger.warning(f"{log_prefix}: Attempting cancel after failure for {format_order_id(entry_order_id)}..."); exchange.cancel_order(entry_order_id, symbol)
            except Exception as cancel_err: logger.warning(f"{log_prefix}: Cancel failed: {cancel_err}")
        return None


def _place_stop_order(exchange: ccxt.Exchange, symbol: str, pos_side: str, qty: Decimal, stop_price_str: str, trigger_by: str, is_tsl: bool = False, tsl_params: dict | None = None) -> tuple[str | None, str]:
    """Helper to place either fixed SL or TSL order. Returns (order_id, status_message)."""
    order_type = "Trailing SL" if is_tsl else "Fixed SL"
    log_prefix = f"Place Order ({order_type})"
    sl_exec_side = CONFIG.SIDE_SELL if pos_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY
    order_id: str | None = None
    status_msg: str = "Placement FAILED"

    try:
        qty_str = format_amount(exchange, symbol, qty)
        qty_float = float(qty_str)
        stop_price_float = float(stop_price_str)  # stopPrice for fixed SL, activePrice for TSL

        params = {'reduceOnly': True, 'triggerBy': trigger_by}
        if is_tsl and tsl_params:
            # Specific TSL params for Bybit V5: trailingStop (%), activePrice
            params['trailingStop'] = tsl_params['trailingStop']
            params['activePrice'] = stop_price_float  # TSL activation price
            logger.info(f"{Fore.CYAN}{log_prefix} ({tsl_params['trail_percent_str']:.2%})... Side: {sl_exec_side.upper()}, Qty: {qty_float:.8f}, Trail%: {params['trailingStop']}, ActPx: {stop_price_str}{Style.RESET_ALL}")
        else:
            # Fixed SL params: stopPrice
            params['stopPrice'] = stop_price_float
            logger.info(f"{Fore.CYAN}{log_prefix}... Side: {sl_exec_side.upper()}, Qty: {qty_float:.8f}, StopPx: {stop_price_str}{Style.RESET_ALL}")

        # Use 'stopMarket' type for both fixed and trailing stops on Bybit V5 via CCXT
        stop_order = exchange.create_order(symbol, 'stopMarket', sl_exec_side, qty_float, params=params)
        order_id = stop_order.get('id')
        if not order_id: raise ValueError(f"{order_type} order placement returned no ID.")

        order_id_short = format_order_id(order_id)
        if is_tsl: status_msg = f"Placed (ID: ...{order_id_short}, Trail: {params['trailingStop']}%, ActPx: {stop_price_str})"
        else: status_msg = f"Placed (ID: ...{order_id_short}, Trigger: {stop_price_str})"
        logger.success(f"{Fore.GREEN}{log_prefix}: {status_msg}{Style.RESET_ALL}")
        return order_id, status_msg

    except Exception as e:
        status_msg = f"Placement FAILED: {type(e).__name__} - {e}"
        logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: {status_msg}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return None, status_msg


def place_risked_market_order(
    exchange: ccxt.Exchange, symbol: str, side: str, risk_percentage: Decimal,
    current_atr: Decimal, sl_atr_multiplier: Decimal, leverage: int, max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal, tsl_percent: Decimal, tsl_activation_offset_percent: Decimal
) -> dict[str, Any] | None:
    """Orchestrates placing a market entry order with risk management, SL, and TSL."""
    market_base = get_market_base_currency(symbol)
    log_prefix = f"Place Order ({side.upper()})"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}{log_prefix}: Full process start for {symbol}...{Style.RESET_ALL}")

    # --- Pre-computation ---
    if current_atr.is_nan() or current_atr <= 0: logger.error(f"{log_prefix}: Invalid ATR. Cannot start."); return None

    try:
        # Fetch needed info
        balance = exchange.fetch_balance()
        market = exchange.market(symbol)
        limits = market.get('limits', {})
        min_qty = safe_decimal_conversion(limits.get('amount', {}).get('min'), default=Decimal('0'))
        max_qty = safe_decimal_conversion(limits.get('amount', {}).get('max')) if limits.get('amount', {}).get('max') else None
        min_price = safe_decimal_conversion(limits.get('price', {}).get('min'), default=Decimal('0'))
        usdt_balance = balance.get(CONFIG.USDT_SYMBOL, {})
        usdt_total = safe_decimal_conversion(usdt_balance.get('total'))
        usdt_free = safe_decimal_conversion(usdt_balance.get('free'))
        usdt_equity = usdt_total if usdt_total > 0 else usdt_free
        if usdt_equity <= 0 or usdt_free < 0: logger.error(f"{log_prefix}: Invalid balance Eq={usdt_equity}, Free={usdt_free}."); return None

        # Estimate entry price
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        entry_price_estimate = ob_data.get("best_ask") if side == CONFIG.SIDE_BUY else ob_data.get("best_bid")
        if not entry_price_estimate: entry_price_estimate = safe_decimal_conversion(exchange.fetch_ticker(symbol).get('last'))
        if entry_price_estimate is None or entry_price_estimate.is_nan() or entry_price_estimate <= 0: logger.error(f"{log_prefix}: Failed get valid entry price estimate."); return None
        logger.info(f"{log_prefix}: Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # Calculate estimated SL
        initial_sl_price = _calculate_initial_sl_price(entry_price_estimate, current_atr, sl_atr_multiplier, side, min_price, symbol, exchange)
        if not initial_sl_price: return None  # Error logged in helper

        # Calculate size
        calc_qty, req_margin_est = calculate_position_size(usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price, leverage, symbol, exchange)
        if calc_qty is None or req_margin_est is None: return None  # Error logged in helper

        # Validate/Cap quantity
        final_qty, final_margin_est = _validate_and_cap_quantity(calc_qty, entry_price_estimate, leverage, max_order_cap_usdt, min_qty, max_qty, symbol, exchange)
        if final_qty is None or final_margin_est is None: return None  # Error logged in helper

        # Check margin
        if not _check_margin_availability(final_margin_est, usdt_free, margin_check_buffer, side, market_base): return None

        # --- Execute Entry ---
        filled_entry_order = _place_and_confirm_entry_order(exchange, symbol, side, final_qty)
        if not filled_entry_order:
             send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement/confirmation failed.")
             # Check if position exists despite failure (edge case)
             time.sleep(1)
             current_pos = get_current_position(exchange, symbol)
             if current_pos['side'] != CONFIG.POS_NONE:
                 logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix} POSITION OPENED despite entry failure! Qty: {current_pos['qty']}. Emergency Closing!{Style.RESET_ALL}")
                 send_sms_alert(f"[{market_base}] CRITICAL: Position opened on FAILED entry! Closing NOW.")
                 close_position(exchange, symbol, current_pos, reason="Emergency Close - Failed Entry Confirm")
             return None  # Entry failed

        # --- Post-Entry SL/TSL Placement ---
        avg_fill_price = safe_decimal_conversion(filled_entry_order.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry_order.get('filled'))
        pos_side = CONFIG.POS_LONG if side == CONFIG.SIDE_BUY else CONFIG.POS_SHORT

        # Calculate actual SL price based on fill
        sl_distance = current_atr * sl_atr_multiplier  # Use same ATR/mult as for sizing
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.SIDE_BUY else (avg_fill_price + sl_distance)
        if min_price > 0 and actual_sl_price_raw < min_price: actual_sl_price_raw = min_price
        elif actual_sl_price_raw <= 0:
             logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: Invalid ACTUAL SL price ({actual_sl_price_raw:.4f}) post-fill. Cannot place SL!{Style.RESET_ALL}")
             send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price ({actual_sl_price_raw:.4f})! Emergency Closing.")
             close_position(exchange, symbol, {'side': pos_side, 'qty': filled_qty}, reason="Invalid SL Calc Post-Entry")
             return filled_entry_order  # Return filled entry, signal overall failure state

        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)

        # Place Fixed SL
        sl_order_id, sl_status_msg = _place_stop_order(exchange, symbol, pos_side, filled_qty, actual_sl_price_str, CONFIG.sl_trigger_by)
        if not sl_order_id:
             send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement! Check position protection.")
             # High risk: Position open without fixed SL

        # Place TSL
        tsl_order_id, tsl_status_msg = "N/A", "Not Configured"
        tsl_trail_val_str = "OFF"
        if tsl_percent > CONFIG.POSITION_QTY_EPSILON:
            try:
                act_offset = avg_fill_price * tsl_activation_offset_percent
                act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.SIDE_BUY else (avg_fill_price - act_offset)
                if min_price > 0 and act_price_raw < min_price: act_price_raw = min_price
                if act_price_raw <= 0: raise ValueError(f"Invalid TSL activation price {act_price_raw:.4f}")

                tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
                tsl_trail_val_str = str((tsl_percent * 100).quantize(Decimal("0.01")))  # e.g., "0.50"
                tsl_params_dict = {'trailingStop': tsl_trail_val_str, 'trail_percent_str': tsl_percent}

                tsl_order_id, tsl_status_msg = _place_stop_order(exchange, symbol, pos_side, filled_qty, tsl_act_price_str, CONFIG.tsl_trigger_by, is_tsl=True, tsl_params=tsl_params_dict)
                if not tsl_order_id:
                    send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement! Relying on fixed SL if placed.")
            except Exception as tsl_calc_err:  # Catch errors in TSL activation price calc
                 logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: FAILED to calculate/place TSL: {tsl_calc_err}{Style.RESET_ALL}")
                 tsl_status_msg = f"Calculation/Placement FAILED: {tsl_calc_err}"
                 send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL setup ({type(tsl_calc_err).__name__}).")

        # Final Summary Log & SMS
        logger.info(f"{Back.BLUE}{Fore.WHITE}--- ORDER PLACEMENT SUMMARY ({side.upper()} {symbol}) ---{Style.RESET_ALL}")
        logger.info(f"  Entry: {format_order_id(filled_entry_order.get('id'))} | Filled: {filled_qty:.8f} @ {avg_fill_price:.4f}")
        logger.info(f"  Fixed SL: {sl_status_msg}")
        logger.info(f"  Trailing SL: {tsl_status_msg}")
        logger.info(f"{Back.BLUE}{Fore.WHITE}--- END SUMMARY ---{Style.RESET_ALL}")

        # Send summary SMS regardless of SL/TSL success (as entry succeeded)
        sms_summary = (f"[{market_base}] {side.upper()} {filled_qty:.6f}@{avg_fill_price:.3f}. "
                       f"SL:{('~' + actual_sl_price_str if sl_order_id else 'FAIL')}. "
                       f"TSL:{('%' + tsl_trail_val_str if tsl_order_id else ('FAIL' if tsl_percent > 0 else 'OFF'))}. "
                       f"EID:{format_order_id(filled_entry_order.get('id'))}")
        send_sms_alert(sms_summary)

        return filled_entry_order  # Return successful entry order

    except Exception as e:
        logger.error(f"{Fore.RED}{Style.BRIGHT}{log_prefix}: Overall process FAILED: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Overall process failed: {type(e).__name__}")
        return None


# --- cancel_open_orders --- (Keep refined version from v2.2.0)
def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Attempts to cancel all open orders for the specified symbol."""
    log_prefix = "Order Cancel"; logger.info(f"{Fore.CYAN}{log_prefix}: Attempting for {symbol} ({reason})...{Style.RESET_ALL}")
    cancelled_count, failed_count = 0, 0; market_base = get_market_base_currency(symbol)
    try:
        if not exchange.has.get('fetchOpenOrders'): logger.warning(f"{Fore.YELLOW}{log_prefix}: fetchOpenOrders not supported.{Style.RESET_ALL}"); return
        logger.debug(f"{log_prefix}: Fetching open orders..."); open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders: logger.info(f"{Fore.CYAN}{log_prefix}: No open orders found.{Style.RESET_ALL}"); return
        logger.warning(f"{Fore.YELLOW}{log_prefix}: Found {len(open_orders)} open orders. Cancelling...{Style.RESET_ALL}")
        for order in open_orders:
            order_id = order.get('id'); order_info_str = f"ID:{format_order_id(order_id)} ({order.get('type', '?')} {order.get('side', '?')})"
            if order_id:
                try: logger.debug(f"{log_prefix}: Cancelling {order_info_str}"); exchange.cancel_order(order_id, symbol); logger.info(f"{Fore.CYAN}{log_prefix}: Success for {order_info_str}{Style.RESET_ALL}"); cancelled_count += 1; time.sleep(0.1)
                except ccxt.OrderNotFound: logger.warning(f"{Fore.YELLOW}{log_prefix}: Not found (already closed?): {order_info_str}{Style.RESET_ALL}"); cancelled_count += 1
                except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.error(f"{Fore.RED}{log_prefix}: FAILED for {order_info_str}: {e}{Style.RESET_ALL}"); failed_count += 1
                except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Unexpected error cancelling {order_info_str}: {e}{Style.RESET_ALL}"); failed_count += 1
            else: logger.warning(f"{log_prefix}: Found order without ID: {order}")
        log_level = logging.INFO if failed_count == 0 else logging.WARNING
        logger.log(log_level, f"{Fore.CYAN}{log_prefix}: Finished. Cancelled/NotFound: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
        if failed_count > 0: send_sms_alert(f"[{market_base}] WARNING: Failed cancel {failed_count} orders during {reason}.")
    except (ccxt.NetworkError, ccxt.ExchangeError) as e: logger.error(f"{Fore.RED}{log_prefix}: Failed fetching open orders: {e}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix}: Unexpected error during cancel process: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())


# --- Strategy Signal Generation ---
# (Keep generate_signals refined version from v2.2.0)
def generate_signals(df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    """Generates entry/exit signals based on the selected strategy and DataFrame columns."""
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit"}
    log_prefix = f"Signal Gen ({strategy_name})"
    if df is None or len(df) < 2: logger.debug(f"{log_prefix}: Insufficient data length."); return signals
    last, prev = df.iloc[-1], df.iloc[-2]
    try:  # Wrap entire strategy block
        if strategy_name == "DUAL_SUPERTREND":
            st_l, st_s, cf_t = last.get('st_long'), last.get('st_short'), last.get('confirm_trend')
            if pd.notna(st_l) and pd.notna(st_s) and pd.notna(cf_t):  # Check indicators are valid
                if st_l and cf_t: signals['enter_long'] = True
                if st_s and not cf_t: signals['enter_short'] = True
                if st_s: signals['exit_long'] = True; signals['exit_reason'] = "Primary ST Short Flip"
                if st_l: signals['exit_short'] = True; signals['exit_reason'] = "Primary ST Long Flip"
            else: logger.debug(f"{log_prefix}: Skipping due to missing ST columns.")
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k, d, m = last.get('stochrsi_k'), last.get('stochrsi_d'), last.get('momentum')
            kp, dp = prev.get('stochrsi_k'), prev.get('stochrsi_d')
            if any(v is None or v.is_nan() for v in [k, d, m, kp, dp]): logger.debug(f"{log_prefix}: Skipping due to missing StochRSI/Mom values.")
            else:
                if kp <= dp and k > d and k < CONFIG.stochrsi_oversold and m > CONFIG.POSITION_QTY_EPSILON: signals['enter_long'] = True
                if kp >= dp and k < d and k > CONFIG.stochrsi_overbought and m < -CONFIG.POSITION_QTY_EPSILON: signals['enter_short'] = True
                if kp >= dp and k < d: signals['exit_long'] = True; signals['exit_reason'] = "StochRSI K crossed below D"
                if kp <= dp and k > d: signals['exit_short'] = True; signals['exit_reason'] = "StochRSI K crossed above D"
        elif strategy_name == "EHLERS_FISHER":
            f, s = last.get('ehlers_fisher'), last.get('ehlers_signal')
            fp, sp = prev.get('ehlers_fisher'), prev.get('ehlers_signal')
            if any(v is None or v.is_nan() for v in [f, s, fp, sp]): logger.debug(f"{log_prefix}: Skipping due to missing Ehlers Fisher values.")
            else:
                if fp <= sp and f > s: signals['enter_long'] = True
                if fp >= sp and f < s: signals['enter_short'] = True
                if fp >= sp and f < s: signals['exit_long'] = True; signals['exit_reason'] = "Ehlers Fisher crossed below Signal"
                if fp <= sp and f > s: signals['exit_short'] = True; signals['exit_reason'] = "Ehlers Fisher crossed above Signal"
        elif strategy_name == "EHLERS_MA_CROSS":
            fm, sm = last.get('fast_ema'), last.get('slow_ema')
            fmp, smp = prev.get('fast_ema'), prev.get('slow_ema')
            if any(v is None or v.is_nan() for v in [fm, sm, fmp, smp]): logger.debug(f"{log_prefix}: Skipping due to missing Ehlers MA (EMA) values.")
            else:
                if fmp <= smp and fm > sm: signals['enter_long'] = True
                if fmp >= smp and fm < sm: signals['enter_short'] = True
                if fmp >= smp and fm < sm: signals['exit_long'] = True; signals['exit_reason'] = "Fast MA crossed below Slow MA"
                if fmp <= smp and fm > sm: signals['exit_short'] = True; signals['exit_reason'] = "Fast MA crossed above Slow MA"
        else: logger.warning(f"{log_prefix}: Unknown strategy name '{strategy_name}'.")
    except KeyError as e: logger.error(f"{Fore.RED}{log_prefix} Error: Missing indicator column: {e}.{Style.RESET_ALL}"); signals = {k: (v if k == 'exit_reason' else False) for k, v in signals.items()}
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Error: Unexpected: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); signals = {k: (v if k == 'exit_reason' else False) for k, v in signals.items()}
    return signals


# --- Trading Logic ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame, cycle_count: int) -> None:
    """Executes the main trading logic for one cycle."""
    # ... (Setup log_prefix, cycle_time_str, market_base as before) ...
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    market_base = get_market_base_currency(symbol)
    log_prefix = f"[Cycle {cycle_count} | {market_base}]"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== {log_prefix} Check Start ({CONFIG.strategy_name}) | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # --- Data Sufficiency Check ---
    # (Keep required_rows check from v2.2.0)
    required_rows = max(150, CONFIG.st_atr_length * 2, CONFIG.confirm_st_atr_length * 2, CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 5, CONFIG.momentum_length * 2, CONFIG.ehlers_fisher_length * 2, CONFIG.ehlers_fisher_signal_length * 2, CONFIG.ehlers_fast_period * 2, CONFIG.ehlers_slow_period * 2, CONFIG.atr_calculation_period * 2, CONFIG.volume_ma_period * 2) + CONFIG.API_FETCH_LIMIT_BUFFER
    if df is None or len(df) < required_rows: logger.warning(f"{Fore.YELLOW}{log_prefix} Insufficient data ({len(df) if df is not None else 0} rows, need ~{required_rows}). Skipping cycle.{Style.RESET_ALL}"); return

    try:
        # === Step 1: Calculate Necessary Indicators ===
        logger.debug(f"{log_prefix} Calculating indicators...")
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")

        # OPTIMIZATION: Only calculate strategy indicators if needed
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
            df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
            df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM":
            df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        elif CONFIG.strategy_name == "EHLERS_FISHER":
            df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS":
            df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)

        # === Step 2: Validate Base Requirements ===
        last_candle = df.iloc[-1]
        current_price = safe_decimal_conversion(last_candle.get('close'))
        if current_price.is_nan() or current_price <= 0: logger.warning(f"{Fore.YELLOW}{log_prefix} Last candle close invalid ({current_price}). Skipping.{Style.RESET_ALL}"); return
        can_place_new_order = current_atr is not None  # Check if None, not NaN check needed after analyze_volume_atr
        if not can_place_new_order: logger.warning(f"{Fore.YELLOW}{log_prefix} Invalid ATR ({current_atr}). New order placement disabled.{Style.RESET_ALL}")

        # === Step 3: Get Position & Optional OB Data ===
        position = get_current_position(exchange, symbol)
        position_side = position['side']; position_qty = position['qty']; position_entry_price = position['entry_price']
        ob_data: dict[str, Decimal | None] | None = None
        if CONFIG.fetch_order_book_per_cycle: ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)

        # === Step 4: Log State (with PnL Estimate) ===
        atr_str = f"{current_atr:.5f}" if current_atr else 'N/A'
        logger.info(f"{log_prefix} State | Price: {current_price:.4f}, ATR: {atr_str}")
        # ... Volume and OB logging ...
        vol_ratio = vol_atr_data.get("volume_ratio"); vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        vol_ratio_str = f"{vol_ratio:.2f}" if vol_ratio is not None else 'N/A'
        logger.info(f"{log_prefix} State | Volume: Ratio={vol_ratio_str}, Spike={vol_spike} (Req:{CONFIG.require_volume_spike_for_entry})")
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None; spread = ob_data.get("spread") if ob_data else None
        ob_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else 'N/A'; spread_str = f"{spread:.4f}" if spread is not None else 'N/A'
        logger.info(f"{log_prefix} State | OrderBook: Ratio={ob_ratio_str}, Spread={spread_str} (Fetched={ob_data is not None})")

        # Estimated PnL Logging
        pnl_str = "N/A"
        if position_side != CONFIG.POS_NONE and not position_entry_price.is_nan() and position_entry_price > 0:
             price_diff = current_price - position_entry_price
             if position_side == CONFIG.POS_SHORT: price_diff = -price_diff
             # Simple PnL estimate (doesn't account for fees/funding)
             # Assume linear contract: PnL = Qty * Price Difference
             est_pnl = position_qty * price_diff
             pnl_color = Fore.GREEN if est_pnl >= 0 else Fore.RED
             pnl_str = f"{pnl_color}{est_pnl:+.4f} USDT{Style.RESET_ALL}"
        logger.info(f"{log_prefix} State | Position: Side={position_side}, Qty={position_qty:.8f}, Entry={position_entry_price:.4f}, Est. PnL: {pnl_str}")

        # === Step 5: Generate Signals ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name)  # Pass df directly
        logger.debug(f"{log_prefix} Signals ({CONFIG.strategy_name}): {strategy_signals}")

        # === Step 6: Handle Exits ===
        # (Keep exit logic from v2.2.0)
        if position_side != CONFIG.POS_NONE:
            should_exit = (position_side == CONFIG.POS_LONG and strategy_signals['exit_long']) or \
                          (position_side == CONFIG.POS_SHORT and strategy_signals['exit_short'])
            if should_exit:
                exit_reason = strategy_signals['exit_reason']
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}{log_prefix} *** STRATEGY EXIT: Closing {position_side} ({exit_reason}) ***{Style.RESET_ALL}")
                cancel_open_orders(exchange, symbol, f"Pre-{exit_reason} Exit")
                time.sleep(0.5)
                close_result = close_position(exchange, symbol, position, reason=exit_reason)
                if close_result:
                    logger.info(f"{log_prefix} Pausing {CONFIG.POST_CLOSE_DELAY_SECONDS}s post-close..."); time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS)
                else: logger.error(f"{Fore.RED}{log_prefix} Failed execute close for {position_side} exit! Check Manually!{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CRITICAL: Failed CLOSE {position_side} on signal! Check!")
                return  # Exit cycle
            else: logger.info(f"{log_prefix} Holding {position_side}. No strategy exit. Waiting SL/TSL."); return

        # === Step 7: Handle Entries (If Flat) ===
        if position_side == CONFIG.POS_NONE:
            if not can_place_new_order: logger.warning(f"{Fore.YELLOW}{log_prefix} Holding Cash. Cannot enter: Invalid ATR.{Style.RESET_ALL}"); return
            logger.debug(f"{log_prefix} Holding Cash. Checking entry signals...")
            enter_long_signal = strategy_signals['enter_long']; enter_short_signal = strategy_signals['enter_short']; potential_entry = enter_long_signal or enter_short_signal
            if not potential_entry: logger.info(f"{log_prefix} Holding Cash. No entry signal from strategy."); return

            # --- Confirmations ---
            passes_volume = not CONFIG.require_volume_spike_for_entry or (vol_atr_data.get("volume_ratio") is not None and vol_atr_data["volume_ratio"] > CONFIG.volume_spike_threshold)
            vol_log = f"VolConf:{passes_volume}(Req:{CONFIG.require_volume_spike_for_entry},Spike:{vol_atr_data.get('volume_ratio'):.2f if vol_atr_data.get('volume_ratio') else 'N/A'}>Thr:{CONFIG.volume_spike_threshold})"
            if not CONFIG.fetch_order_book_per_cycle and ob_data is None:
                 logger.debug(f"{log_prefix} Fetching OB for entry confirm..."); ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None; ob_available = bid_ask_ratio is not None
            passes_long_ob = not ob_available or (bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
            passes_short_ob = not ob_available or (bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
            ob_log = f"OBConf:LPass:{passes_long_ob},SPass:{passes_short_ob}(Avail:{ob_available},Ratio:{bid_ask_ratio:.3f if ob_available else 'N/A'})"

            # --- Combine ---
            final_enter_long = enter_long_signal and passes_volume and passes_long_ob
            final_enter_short = enter_short_signal and passes_volume and passes_short_ob
            logger.debug(f"{log_prefix} Final Entry Check | Long: {final_enter_long} ({vol_log}, {ob_log})")
            logger.debug(f"{log_prefix} Final Entry Check | Short: {final_enter_short} ({vol_log}, {ob_log})")

            # --- Execute ---
            entry_side: str | None = None
            if final_enter_long: entry_side = CONFIG.SIDE_BUY; logger.success(f"{Back.GREEN}{Fore.BLACK}{log_prefix} *** CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) ***{Style.RESET_ALL}")
            elif final_enter_short: entry_side = CONFIG.SIDE_SELL; logger.success(f"{Back.RED}{Fore.WHITE}{log_prefix} *** CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) ***{Style.RESET_ALL}")

            if entry_side:
                cancel_open_orders(exchange, symbol, f"Pre-{entry_side.upper()} Entry"); time.sleep(0.5)
                # Pass validated current_atr to placement function
                place_result = place_risked_market_order(exchange, symbol, entry_side, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier, CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer, CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
                if place_result: pass
                return  # End cycle
            else: logger.info(f"{log_prefix} Holding Cash. Entry signal failed confirmation checks.")

    # ... (rest of error handling as before) ...
    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{log_prefix} CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!")
    finally:
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== {log_prefix} Cycle Check End =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown ---
# (Keep graceful_shutdown refined version from v2.2.0)
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to gracefully shut down the bot."""
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit...{Style.RESET_ALL}")
    market_base = get_market_base_currency(symbol) if symbol else "Bot"; send_sms_alert(f"[{market_base}] Shutdown initiated. Cleanup...")
    if not exchange or not symbol: logger.warning(f"{Fore.YELLOW}Shutdown: Exchange/Symbol missing. Cannot cleanup.{Style.RESET_ALL}"); return
    try:
        logger.info(f"{Fore.CYAN}Shutdown: Cancelling open orders for {symbol}...{Style.RESET_ALL}"); cancel_open_orders(exchange, symbol, reason="Graceful Shutdown"); time.sleep(1)
        logger.info(f"{Fore.CYAN}Shutdown: Checking active position for {symbol}...{Style.RESET_ALL}"); position = get_current_position(exchange, symbol)
        if position['side'] != CONFIG.POS_NONE and position['qty'] > CONFIG.POSITION_QTY_EPSILON:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} pos Qty: {position['qty']:.8f}. Closing...{Style.RESET_ALL}"); close_result = close_position(exchange, symbol, position, reason="Shutdown Request")
            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.POST_CLOSE_DELAY_SECONDS * 2}s confirm...{Style.RESET_ALL}"); time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS * 2)
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.POS_NONE or final_pos['qty'] <= CONFIG.POSITION_QTY_EPSILON: logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else: logger.error(f"{Back.RED}{Fore.WHITE}Shutdown CRITICAL: FAILED CONFIRM closure. Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CRITICAL ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. CHECK MANUALLY!")
            else: logger.error(f"{Back.RED}{Fore.WHITE}Shutdown CRITICAL: Failed PLACE close order. MANUAL INTERVENTION!{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CRITICAL ERROR: Failed PLACE close order! MANUAL CHECK!")
        else: logger.info(f"{Fore.GREEN}Shutdown: No active position found.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] No active position found on shutdown.")
    except Exception as e: logger.error(f"{Fore.RED}Shutdown Error: Unexpected during cleanup: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] Error during shutdown cleanup: {type(e).__name__}")
    finally: logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Shutdown Sequence Complete ---{Style.RESET_ALL}")


# --- Main Execution ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    global cycle_count  # Allow modification of global cycle count
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}{'--- Pyrmethus Bybit Scalping Spell v2.3.0 Initializing ---':^80}{Style.RESET_ALL}")
    logger.info(f"{'Timestamp:':<20} {start_time_str}")
    logger.info(f"{'Selected Strategy:':<20} {Fore.CYAN}{CONFIG.strategy_name}{Style.RESET_ALL}")
    logger.info(f"{'Risk Management:':<20} {Fore.GREEN}Initial ATR SL + Exchange TSL{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}{'--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---':^80}{Style.RESET_ALL}")

    exchange: ccxt.Exchange | None = None
    symbol: str | None = None
    market_base: str = "Bot"
    run_bot: bool = True

    try:
        # === Initialize Exchange ===
        exchange = initialize_exchange()
        if not exchange: return

        # === Setup Symbol and Leverage ===
        try:
            symbol_to_use = CONFIG.symbol
            logger.info(f"Attempting to use symbol: {symbol_to_use}")
            market = exchange.market(symbol_to_use)
            symbol = market['symbol']
            market_base = get_market_base_currency(symbol)
            if not market.get('contract'): raise ValueError("Not a contract market")
            logger.info(f"{Fore.GREEN}Using Symbol: {symbol} (Type: {market.get('type', 'N/A')}, ID: {market.get('id')}){Style.RESET_ALL}")
            if not set_leverage(exchange, symbol, CONFIG.leverage): raise RuntimeError("Leverage setup failed")
        except Exception as e:  # Catch all setup errors
            logger.critical(f"Symbol/Leverage setup failed for '{CONFIG.symbol}': {e}")
            send_sms_alert(f"[{market_base or 'Bot'}] CRITICAL: Symbol/Leverage setup FAILED ({type(e).__name__}). Exiting.")
            return

        # === Log Configuration Summary ===
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}{'--- Configuration Summary ---':^80}{Style.RESET_ALL}")
        logger.info(f"  {'Symbol:':<25} {symbol}")
        logger.info(f"  {'Interval:':<25} {CONFIG.interval}")
        logger.info(f"  {'Leverage:':<25} {CONFIG.leverage}x")
        logger.info(f"  {'Strategy:':<25} {CONFIG.strategy_name}")
        # Log relevant strategy params concisely
        if CONFIG.strategy_name == "DUAL_SUPERTREND": logger.info(f"    Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": logger.info(f"    Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER": logger.info(f"    Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": logger.info(f"    Params: FastMA={CONFIG.ehlers_fast_period}, SlowMA={CONFIG.ehlers_slow_period} (EMA Placeholder)")
        logger.info(f"{Fore.GREEN}  {'Risk % / Trade:':<25} {(CONFIG.risk_per_trade_percentage * 100):.3f}%")
        logger.info(f"  {'Max Position Value Cap:':<25} {CONFIG.max_order_usdt_amount:.2f} USDT")
        logger.info(f"  {'Initial SL:':<25} {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period}) | Trigger: {CONFIG.sl_trigger_by}")
        logger.info(f"  {'Trailing SL:':<25} {(CONFIG.trailing_stop_percentage * 100):.2f}% | Act. Offset: {(CONFIG.trailing_stop_activation_offset_percent * 100):.2f}% | Trigger: {CONFIG.tsl_trigger_by}")
        logger.info(f"{Fore.YELLOW}  {'Volume Confirmation:':<25} {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"  {'Order Book Confirmation:':<25} {CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        logger.info(f"{Fore.WHITE}  {'Loop Sleep:':<25} {CONFIG.sleep_seconds}s")
        logger.info(f"  {'Margin Check Buffer:':<25} {((CONFIG.required_margin_buffer - 1) * 100):.1f}%")
        logger.info(f"  {'Order Fill Timeout:':<25} {CONFIG.order_fill_timeout_seconds}s")
        logger.info(f"{Fore.MAGENTA}  {'SMS Alerts:':<25} {CONFIG.enable_sms_alerts} (To: {'*****' + CONFIG.sms_recipient_number[-4:] if CONFIG.sms_recipient_number else 'Not Set'})")
        logger.info(f"{Fore.CYAN}  {'Logging Level:':<25} {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}{'---------------------------------------------------':^80}{Style.RESET_ALL}")

        send_sms_alert(f"[{market_base}] Pyrmethus v2.3 Started. Strat:{CONFIG.strategy_name}|{symbol}|{CONFIG.interval}|{CONFIG.leverage}x. Live Trading!")

        # === Main Trading Loop ===
        logger.info(f"{Fore.GREEN}{Style.BRIGHT}--- Starting Main Trading Loop ({time.strftime('%H:%M:%S')}) ---{Style.RESET_ALL}")
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            log_prefix_cycle = f"[Cycle {cycle_count} | {market_base}]"
            logger.debug(f"{Fore.CYAN}--- {log_prefix_cycle} Start ---{Style.RESET_ALL}")

            try:
                # Determine required data length dynamically
                data_limit = max(150, CONFIG.atr_calculation_period * 2, CONFIG.volume_ma_period * 2,  # Base + common
                                # Strategy specific lookbacks
                                (CONFIG.st_atr_length * 2 + CONFIG.confirm_st_atr_length * 2) if CONFIG.strategy_name == "DUAL_SUPERTREND" else 0,
                                (CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + max(CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period) + CONFIG.momentum_length) if CONFIG.strategy_name == "STOCHRSI_MOMENTUM" else 0,
                                (CONFIG.ehlers_fisher_length * 2 + CONFIG.ehlers_fisher_signal_length) if CONFIG.strategy_name == "EHLERS_FISHER" else 0,
                                (CONFIG.ehlers_slow_period * 2) if CONFIG.strategy_name == "EHLERS_MA_CROSS" else 0
                                ) + CONFIG.API_FETCH_LIMIT_BUFFER

                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                if df is not None and not df.empty:
                    trade_logic(exchange, symbol, df, cycle_count)  # Pass cycle count for logging
                else:
                    logger.warning(f"{Fore.YELLOW}{log_prefix_cycle} No valid market data. Skipping logic.{Style.RESET_ALL}")

            # ... (Robust error handling within loop - keep from v2.2.0) ...
            except ccxt.RateLimitExceeded as e: logger.warning(f"{Back.YELLOW}{Fore.BLACK}{log_prefix_cycle} Rate Limit Exceeded: {e}. Sleeping {CONFIG.sleep_seconds * 5}s...{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] WARNING: Rate limit hit!"); time.sleep(CONFIG.sleep_seconds * 5)
            except ccxt.NetworkError as e: logger.warning(f"{Fore.YELLOW}{log_prefix_cycle} Network error: {e}. Retrying next cycle.{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds)
            except ccxt.ExchangeNotAvailable as e: logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix_cycle} Exchange unavailable: {e}. Sleeping {CONFIG.sleep_seconds * 10}s...{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable!"); time.sleep(CONFIG.sleep_seconds * 10)
            except ccxt.AuthenticationError as e: logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{log_prefix_cycle} Auth Error: {e}. Stopping NOW.{Style.RESET_ALL}"); run_bot = False; send_sms_alert(f"[{market_base}] CRITICAL: Auth Error! Stopping NOW.")
            except ccxt.ExchangeError as e: logger.error(f"{Fore.RED}{log_prefix_cycle} Unhandled Exchange error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs."); time.sleep(CONFIG.sleep_seconds)
            except Exception as e: logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{log_prefix_cycle} !!! UNEXPECTED CRITICAL ERROR: {e} !!!{Style.RESET_ALL}"); run_bot = False; send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping NOW.")

            # --- Loop Delay ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"{log_prefix_cycle} Cycle time: {elapsed:.2f}s. Sleeping: {sleep_dur:.2f}s.")
                if sleep_dur > 0: time.sleep(sleep_dur)

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Initiating shutdown...{Style.RESET_ALL}")
        run_bot = False
    except Exception as e:  # Catch setup errors
         logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Critical error during bot setup phase: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
         send_sms_alert(f"[{market_base or 'Bot'}] CRITICAL SETUP ERROR: {type(e).__name__}! Bot failed start.")
         run_bot = False
    finally:
        # --- Graceful Shutdown ---
        graceful_shutdown(exchange, symbol)
        final_alert_market = market_base if market_base != "Bot" else "Bot"
        send_sms_alert(f"[{final_alert_market}] Pyrmethus v2.3 process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}{'--- Pyrmethus Scalping Spell Deactivated ---':^80}{Style.RESET_ALL}")


if __name__ == "__main__":
    try: main()
    except Exception as e: logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Unhandled exception at top level: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); sys.exit(1)
    sys.exit(0)
