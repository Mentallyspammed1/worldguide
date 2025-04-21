# --- START OF FILE kbot3_enhanced.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.1 (Precision, Strategy Selection, Refined Robustness)
# Conjures high-frequency trades on Bybit Futures with enhanced precision, adaptable strategies, and improved resilience.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.1.0 (Unified: Selectable Strategies + Precision + Native SL/TSL + Refinements)

Features:
- Multiple strategies selectable via config: "DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS".
- Enhanced Precision: Uses Decimal for critical financial calculations.
- Exchange-native Trailing Stop Loss (TSL) placed immediately after entry.
- Exchange-native fixed Stop Loss placed immediately after entry.
- ATR for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation.
- Risk-based position sizing with margin checks.
- Termux SMS alerts for critical events and trade actions.
- Robust error handling and logging with Neon color support.
- Graceful shutdown on KeyboardInterrupt with position/order closing attempt.
- Stricter position detection logic (Bybit V5 API).
- Refined error handling, logging, and robustness checks.

Disclaimer:
- **EXTREME RISK**: Educational purposes ONLY. High-risk. Use at own absolute risk.
- **EXCHANGE-NATIVE SL/TSL DEPENDENCE**: Relies on exchange-native orders. Subject to exchange performance, slippage, API reliability.
- Parameter Sensitivity: Requires significant tuning and testing.
- API Rate Limits: Monitor usage.
- Slippage: Market orders are prone to slippage.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: Requires Termux:API.
- API Changes: Code targets Bybit V5 via CCXT, updates may be needed.
"""

# Standard Library Imports
import logging
import os
import sys
import time
import traceback
import subprocess
import shlex
from typing import Dict, Optional, Any, Tuple, List, Union
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN, InvalidOperation

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta # type: ignore[import]
    from dotenv import load_dotenv
    from colorama import init as colorama_init, Fore, Style, Back
except ImportError as e:
    missing_pkg = e.name
    print(f"\033[91mMissing essential spell component: \033[1m{missing_pkg}\033[0m") # Bright Red
    print("\033[93mTo conjure it, cast the following spell in your Termux terminal:\033[0m") # Bright Yellow
    print(f"\033[1m\033[96mpip install {missing_pkg}\033[0m") # Bold Bright Cyan
    print("\n\033[96mOr, to ensure all scrolls are present, cast:\033[0m") # Bright Cyan
    print("\033[1m\033[96mpip install ccxt pandas pandas_ta python-dotenv colorama\033[0m")
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18 # Set Decimal precision globally

# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables."""
    def __init__(self):
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials ---
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: Optional[str] = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW)

        # --- Strategy Selection ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: List[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid: {self.valid_strategies}")

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # 0.5% risk per trade
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN) # Multiplier for ATR-based initial SL
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN) # Max position value in USDT
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN) # e.g., 1.05 means 5% buffer on required margin

        # --- Trailing Stop Loss (Exchange Native) ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.005 = 0.5% trail distance
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN) # e.g., 0.001 = 0.1% offset from entry to activate TSL

        # --- Dual Supertrend Parameters ---
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN)

        # --- StochRSI + Momentum Parameters ---
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN)
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN)
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN)

        # --- Ehlers Fisher Transform Parameters ---
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN) # Signal = 1 means Fisher line only

        # --- Ehlers MA Cross Parameters ---
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN)

        # --- Volume Analysis ---
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW) # Ratio of current vol to MA
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW) # Levels to sum for ratio
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW) # Bid/Ask ratio >= this for long
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW) # Bid/Ask ratio <= this for short
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW) # If false, fetch only on potential entry

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN)

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA)

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = 10000 # milliseconds
        self.order_book_fetch_limit: int = max(25, self.order_book_depth) # Min limit often 25 for L2
        self.shallow_ob_fetch_depth: int = 5 # For quick price estimate
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW) # Wait time for market order fill confirmation

        # --- Internal Constants ---
        self.side_buy: str = "buy"
        self.side_sell: str = "sell"
        self.pos_long: str = "Long"
        self.pos_short: str = "Short"
        self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"
        self.retry_count: int = 3
        self.retry_delay_seconds: int = 2
        self.api_fetch_limit_buffer: int = 10 # Extra candles to fetch beyond indicator needs
        self.position_qty_epsilon: Decimal = Decimal("1e-9") # Small value to check against zero qty
        self.post_close_delay_seconds: int = 3 # Pause after closing before allowing new entry
        self.market_order_fill_check_interval: float = 0.5 # Seconds between checks for fill

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned ---{Style.RESET_ALL}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE) -> Any:
        """Fetches env var, casts type, logs, handles defaults/errors."""
        value = os.getenv(key)
        log_value = f"'{value}'" if value is not None else f"Not Set (Using Default: '{default}')"
        logger.debug(f"{color}Summoning {key}: {log_value}{Style.RESET_ALL}")

        if value is None:
            if required:
                raise ValueError(f"CRITICAL: Required environment variable '{key}' not set.")
            value = default
        elif cast_type == bool:
            value = value.lower() in ['true', '1', 'yes', 'y']
        elif cast_type == Decimal:
            try:
                value = Decimal(value)
            except InvalidOperation:
                logger.error(f"{Fore.RED}Invalid Decimal value for {key}: '{value}'. Using default: '{default}'{Style.RESET_ALL}")
                value = Decimal(str(default)) if default is not None else None
                if required and value is None: # Ensure required Decimal has a valid default
                    raise ValueError(f"CRITICAL: Required Decimal env var '{key}' had invalid value and no valid default.")
        elif cast_type is not None:
            try:
                value = cast_type(value)
            except (ValueError, TypeError):
                logger.error(f"{Fore.RED}Invalid type for {key}: '{value}'. Expected {cast_type.__name__}. Using default: '{default}'{Style.RESET_ALL}")
                value = default

        if value is None and required: # Check again if default was None
             raise ValueError(f"CRITICAL: Required environment variable '{key}' has no value or default.")

        return value

# --- Logger Setup ---
LOGGING_LEVEL: int = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger: logging.Logger = logging.getLogger(__name__)

# Custom SUCCESS level and Neon Color Formatting
SUCCESS_LEVEL: int = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs) # pylint: disable=protected-access
logging.Logger.success = log_success # type: ignore

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

# --- Helper Functions ---
def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning default if conversion fails or value is None."""
    if value is None:
        return default
    try:
        # Convert potential floats or other numeric types to string first for Decimal
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}")
        return default

def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """Returns the last 6 characters of an order ID or 'N/A'."""
    return f"...{str(order_id)[-6:]}" if order_id else "N/A"

def get_market_base_currency(symbol: str) -> str:
    """Extracts the base currency from a symbol like 'BTC/USDT:USDT'."""
    try:
        return symbol.split('/')[0]
    except IndexError:
        return symbol # Fallback if format is unexpected

# --- Precision Formatting ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal, str]) -> str:
    """Formats price according to market precision rules, returning string."""
    try:
        # Convert Decimal to float for ccxt, handle potential exceptions during conversion
        price_float = float(price)
        return exchange.price_to_precision(symbol, price_float)
    except (ValueError, TypeError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}{Style.RESET_ALL}")
        # Fallback: attempt to quantize Decimal if input was Decimal, else return as string
        if isinstance(price, Decimal):
            return str(price.normalize())
        return str(price)

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal, str]) -> str:
    """Formats amount according to market precision rules, returning string."""
    try:
        # Convert Decimal to float for ccxt, handle potential exceptions during conversion
        amount_float = float(amount)
        return exchange.amount_to_precision(symbol, amount_float)
    except (ValueError, TypeError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}{Style.RESET_ALL}")
         # Fallback: attempt to quantize Decimal if input was Decimal, else return as string
        if isinstance(amount, Decimal):
            # Get market precision if possible for a better fallback
            precision = exchange.markets[symbol].get('precision', {}).get('amount')
            if precision:
                 # Precision might be 0.001, 1e-8 etc. Convert to Decimal places.
                decimal_places = Decimal(str(precision)).normalize().as_tuple().exponent * -1
                return str(amount.quantize(Decimal('1e-' + str(decimal_places)), rounding=ROUND_DOWN)) # Always round down amount
            return str(amount.normalize()) # Basic normalize if precision unknown
        return str(amount)

# --- Termux SMS Alert Function ---
def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API if enabled."""
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled.")
        return False
    if not CONFIG.sms_recipient_number:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")
        return False
    try:
        # Use shlex.quote for safety if message content could be complex, but direct passing often works
        # quoted_message = shlex.quote(message)
        command: List[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Attempting SMS to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS command executed successfully.{Style.RESET_ALL}")
            return True
        else:
            logger.error(f"{Fore.RED}SMS command failed. RC: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}")
            return False
    except FileNotFoundError:
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' not found. Ensure Termux:API package and app are installed and configured.{Style.RESET_ALL}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False

# --- Exchange Initialization ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """Initializes and returns the CCXT Bybit exchange instance."""
    logger.info(f"{Fore.BLUE}Initializing CCXT Bybit connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret:
        logger.critical("API keys (BYBIT_API_KEY, BYBIT_API_SECRET) missing in environment.")
        send_sms_alert("[Pyrmethus] CRITICAL: API keys missing. Bot stopped.")
        return None
    try:
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear", # Explicitly set for USDT perpetuals
                "recvWindow": CONFIG.default_recv_window,
                "adjustForTimeDifference": True,
                'verbose': LOGGING_LEVEL == logging.DEBUG, # Enable verbose CCXT logging if bot is in debug
            },
        })
        logger.debug("Loading markets (forced)...")
        exchange.load_markets(True) # Force reload to get latest info
        logger.debug("Performing initial balance check...")
        exchange.fetch_balance() # Initial connectivity and auth check
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session Initialized (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}")
        send_sms_alert("[Pyrmethus] Initialized & authenticated successfully.")
        return exchange
    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}. Check API keys, IP whitelist, and permissions.")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error during initialization: {e}. Check connection and Bybit status.")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Network Error on Init: {e}. Bot stopped.")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error during initialization: {e}. Check Bybit status and API documentation.")
        send_sms_alert(f"[Pyrmethus] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
    except Exception as e:
        logger.critical(f"Unexpected error during exchange initialization: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[Pyrmethus] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")
    return None

# --- Indicator Calculation Functions ---
# Note: These functions now return the modified DataFrame and handle NA values more explicitly.
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returns Decimal where applicable."""
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    st_col = f"SUPERT_{length}_{float(multiplier)}" # pandas_ta uses float in name
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]

    # Initialize target columns with NA
    for col in target_cols: df[col] = pd.NA

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Indicator Calc ({col_prefix}ST): Invalid input (Len: {len(df) if df is not None else 0}, Need: {length + 1}). Setting cols to NA.{Style.RESET_ALL}")
        return df

    try:
        # pandas_ta expects float multiplier
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if st_col not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col}, {st_trend_col}")

        # Convert Supertrend value to Decimal
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1 # Boolean (True for uptrend, False for downtrend)
        prev_trend = df[st_trend_col].shift(1)
        # Boolean flags for trend changes
        df[f"{col_prefix}st_long"] = (prev_trend == -1) & (df[st_trend_col] == 1) # Downtrend to Uptrend
        df[f"{col_prefix}st_short"] = (prev_trend == 1) & (df[st_trend_col] == -1) # Uptrend to Downtrend

        # Clean up raw pandas_ta columns
        raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col]
        df.drop(columns=raw_st_cols, errors='ignore', inplace=True)

        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
            signal = 'LONG_FLIP' if df[f'{col_prefix}st_long'].iloc[-1] else ('SHORT_FLIP' if df[f'{col_prefix}st_short'].iloc[-1] else 'No Flip')
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Trend={last_trend}, Val={last_st_val:.4f}, Signal={signal}")
        else:
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Ensure reset on error
    return df

def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> Tuple[pd.DataFrame, Dict[str, Optional[Decimal]]]:
    """Calculates ATR, Volume MA, checks spikes. Returns modified DF and Decimals in a Dict."""
    results: Dict[str, Optional[Decimal]] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len) + 1 # Need +1 for calculations like rolling MA

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (Vol/ATR): Invalid input (Len: {len(df) if df is not None else 0}, Need: {min_len}). Returning N/A.{Style.RESET_ALL}")
        return df, results

    try:
        # Calculate ATR
        atr_col = f"ATRr_{atr_len}"
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr): results["atr"] = safe_decimal_conversion(last_atr)
            df.drop(columns=[atr_col], errors='ignore', inplace=True) # Clean up raw column
        else:
            logger.warning(f"ATR column '{atr_col}' not found after calculation.")

        # Calculate Volume MA
        volume_ma_col = f'volume_ma_{vol_ma_len}'
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df['volume'].iloc[-1]

        if pd.notna(last_vol_ma): results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol): results["last_volume"] = safe_decimal_conversion(last_vol)

        # Calculate volume ratio safely
        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.position_qty_epsilon and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except Exception as ratio_err:
                 logger.warning(f"Error calculating volume ratio: {ratio_err}")
                 results["volume_ratio"] = None

        # Optionally keep the MA column in df or drop it
        # df.drop(columns=[volume_ma_col], errors='ignore', inplace=True) # Uncomment to drop

        # Log results
        atr_str = f"{results['atr']:.5f}" if results['atr'] else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results['volume_ratio'] else 'N/A'
        logger.debug(f"Indicator Calc: ATR({atr_len})={atr_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={vol_ratio_str}")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results} # Reset results on error
    return df, results

def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, returns modified DF with Decimals."""
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    min_len = max(rsi_len + stoch_len, mom_len) + k + d + 5 # Conservative buffer
    required_input_cols = ["close"]

    # Initialize target columns with NA
    for col in target_cols: df[col] = pd.NA

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (StochRSI/Mom): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Setting cols to NA.{Style.RESET_ALL}")
        return df

    try:
        # StochRSI
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col, d_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if k_col in stochrsi_df.columns: df['stochrsi_k'] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else: logger.warning(f"StochRSI K column '{k_col}' not found"); df['stochrsi_k'] = pd.NA
        if d_col in stochrsi_df.columns: df['stochrsi_d'] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else: logger.warning(f"StochRSI D column '{d_col}' not found"); df['stochrsi_d'] = pd.NA

        # Momentum
        mom_col = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns:
            df['momentum'] = df[mom_col].apply(safe_decimal_conversion)
            df.drop(columns=[mom_col], errors='ignore', inplace=True) # Clean up raw column
        else: logger.warning(f"Momentum column '{mom_col}' not found"); df['momentum'] = pd.NA

        # Log last values
        k_val, d_val, mom_val = df['stochrsi_k'].iloc[-1], df['stochrsi_d'].iloc[-1], df['momentum'].iloc[-1]
        k_str = f"{k_val:.2f}" if pd.notna(k_val) else "NA"
        d_str = f"{d_val:.2f}" if pd.notna(d_val) else "NA"
        mom_str = f"{mom_val:.4f}" if pd.notna(mom_val) else "NA"
        logger.debug(f"Indicator Calc (StochRSI/Mom): K={k_str}, D={d_str}, Mom={mom_str}")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Ensure reset on error
    return df

def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform, returns modified DF with Decimals."""
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    required_input_cols = ["high", "low"]
    min_len = length + signal + 5 # Conservative buffer

    # Initialize target columns with NA
    for col in target_cols: df[col] = pd.NA

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersFisher): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Setting cols to NA.{Style.RESET_ALL}")
        return df

    try:
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col, signal_col = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fish_col in fisher_df.columns: df['ehlers_fisher'] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else: logger.warning(f"Ehlers Fisher column '{fish_col}' not found"); df['ehlers_fisher'] = pd.NA
        # Only add signal if length > 1, pandas_ta might not create it otherwise
        if signal > 0 and signal_col in fisher_df.columns: df['ehlers_signal'] = fisher_df[signal_col].apply(safe_decimal_conversion)
        else: df['ehlers_signal'] = pd.NA # Set to NA if signal length is 0 or column missing

        # Log last values
        fish_val, sig_val = df['ehlers_fisher'].iloc[-1], df['ehlers_signal'].iloc[-1]
        fish_str = f"{fish_val:.4f}" if pd.notna(fish_val) else "NA"
        sig_str = f"{sig_val:.4f}" if pd.notna(sig_val) else "NA"
        logger.debug(f"Indicator Calc (EhlersFisher({length},{signal})): Fisher={fish_str}, Signal={sig_str}")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Ensure reset on error
    return df

def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers Super Smoother Moving Averages (using EMA as placeholder), returns modified DF with Decimals."""
    target_cols = ['fast_ema', 'slow_ema']
    required_input_cols = ["close"]
    min_len = max(fast_len, slow_len) + 5 # Add buffer

    # Initialize target columns with NA
    for col in target_cols: df[col] = pd.NA

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersMA): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}). Setting cols to NA.{Style.RESET_ALL}")
        return df

    try:
        # Placeholder: Using standard EMA. Replace with a proper Ehlers Super Smoother implementation if available/needed.
        # Standard libraries like pandas_ta might not have Ehlers Super Smoother directly.
        # Common libraries like `talib` might, or require custom implementation based on Ehlers' formula.
        logger.warning(f"{Fore.YELLOW}Using EMA as placeholder for Ehlers Super Smoother. Verify if this is intended or replace with actual Ehlers implementation.{Style.RESET_ALL}")
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        # Log last values
        fast_val, slow_val = df['fast_ema'].iloc[-1], df['slow_ema'].iloc[-1]
        fast_str = f"{fast_val:.4f}" if pd.notna(fast_val) else "NA"
        slow_str = f"{slow_val:.4f}" if pd.notna(slow_val) else "NA"
        logger.debug(f"Indicator Calc (EhlersMA({fast_len},{slow_len}) - Placeholder): Fast={fast_str}, Slow={slow_str}")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA Placeholder): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA # Ensure reset on error
    return df

def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> Dict[str, Optional[Decimal]]:
    """Fetches and analyzes L2 order book pressure and spread. Returns Decimals."""
    results: Dict[str, Optional[Decimal]] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}fetchL2OrderBook not supported by {exchange.id}. Cannot analyze order book.{Style.RESET_ALL}")
        return results
    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: List[List[Union[float, str]]] = order_book.get('bids', [])
        asks: List[List[Union[float, str]]] = order_book.get('asks', [])

        if not bids or not asks:
            logger.warning(f"Order Book: Empty bids or asks returned for {symbol}.")
            return results

        best_bid = safe_decimal_conversion(bids[0][0]) if len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
            logger.debug(f"OB: Best Bid={best_bid:.4f}, Best Ask={best_ask:.4f}, Spread={results['spread']:.4f}")
        else:
            logger.debug(f"OB: Best Bid={best_bid or 'N/A'}, Best Ask={best_ask or 'N/A'} (Spread N/A)")

        # Sum volumes within the specified depth using Decimal for precision
        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:depth] if len(ask) > 1)
        logger.debug(f"OB (Depth {depth}): Total BidVol={bid_vol:.4f}, Total AskVol={ask_vol:.4f}")

        # Calculate ratio safely
        if ask_vol > CONFIG.position_qty_epsilon: # Avoid division by zero or near-zero
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                logger.debug(f"OB Ratio (Bid/Ask): {results['bid_ask_ratio']:.3f}")
            except Exception as ratio_err:
                logger.warning(f"Error calculating OB ratio: {ratio_err}")
                results["bid_ask_ratio"] = None
        else:
            logger.debug("OB Ratio: N/A (Ask volume zero or negligible)")

    except (ccxt.NetworkError, ccxt.ExchangeError, IndexError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Analysis Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results} # Reset results on error
    return results

# --- Data Fetching ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches and prepares OHLCV data, ensuring numeric types and handling NaNs."""
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} ({interval})...")
        # FetchOHLCV params: symbol, timeframe, since=None, limit=None, params={}
        ohlcv: List[List[Union[int, float, str]]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}). Could be an API issue or incorrect symbol/interval.{Style.RESET_ALL}")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Convert to numeric, coercing errors to NaN
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for and handle NaNs introduced by coercion or missing data
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV contains NaNs after conversion:\n{nan_counts[nan_counts > 0]}\nAttempting ffill...{Style.RESET_ALL}")
            df.ffill(inplace=True) # Forward fill first
            if df.isnull().values.any(): # Check again
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill, attempting bfill...{Style.RESET_ALL}")
                df.bfill(inplace=True) # Back fill if ffill wasn't enough (e.g., NaNs at the start)
                if df.isnull().values.any():
                    logger.error(f"{Fore.RED}Data Fetch: NaNs persist after ffill/bfill. Cannot use this data.{Style.RESET_ALL}")
                    return None # Unrecoverable NaNs

        # Final check for valid data types (ensure they are numeric)
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in numeric_cols):
             logger.error(f"{Fore.RED}Data Fetch: Non-numeric data found in OHLCV columns after processing. Cannot proceed.{Style.RESET_ALL}")
             return None

        logger.debug(f"Data Fetch: Processed {len(df)} OHLCV candles for {symbol}. Last candle: {df.index[-1]}")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: CCXT Error fetching OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Data Fetch: Unexpected error processing OHLCV for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return None

# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches current position details using Bybit V5 API specifics via CCXT.
    Returns a dictionary with 'side' (Config.pos_long/pos_short/pos_none),
    'qty' (Decimal, absolute value), and 'entry_price' (Decimal).
    Handles One-Way mode (positionIdx=0).
    """
    default_pos: Dict[str, Any] = {'side': CONFIG.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market: Optional[Dict] = None
    market_id: Optional[str] = None

    try:
        market = exchange.market(symbol)
        market_id = market['id']
        if not market or not market_id:
             raise ValueError("Market info not found in CCXT.")
    except (ccxt.BadSymbol, KeyError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}Position Check: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
        return default_pos

    try:
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"{Fore.YELLOW}Position Check: fetchPositions capability not reported by CCXT for {exchange.id}. Assuming no position.{Style.RESET_ALL}")
            return default_pos

        # Bybit V5 requires 'category' parameter: 'linear' or 'inverse'
        params = {'category': 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else 'linear')} # Default linear if unsure
        logger.debug(f"Position Check: Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # Filter for the specific symbol and One-Way mode (positionIdx=0)
        active_pos = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {})
            pos_market_id = pos_info.get('symbol')
            position_idx = int(pos_info.get('positionIdx', -1)) # Default to -1 if missing
            pos_side_v5 = pos_info.get('side', 'None').strip() # 'Buy', 'Sell', or 'None'
            size_str = pos_info.get('size')

            # Strict check for matching symbol, one-way mode, and an actual side
            if pos_market_id == market_id and position_idx == 0 and pos_side_v5 in ['Buy', 'Sell']:
                size = safe_decimal_conversion(size_str)
                # Check if size is significantly different from zero
                if abs(size) > CONFIG.position_qty_epsilon:
                    active_pos = pos # Found the likely active position
                    logger.debug(f"Found potential active position entry: {pos_info}")
                    break # Assume only one active position in One-Way mode

        if active_pos:
            try:
                size = safe_decimal_conversion(active_pos.get('info', {}).get('size'))
                # Use 'avgPrice' from V5 info dict for entry price
                entry_price = safe_decimal_conversion(active_pos.get('info', {}).get('avgPrice'))
                pos_side_v5 = active_pos.get('info', {}).get('side')

                # Determine bot's side representation
                side = CONFIG.pos_long if pos_side_v5 == 'Buy' else (CONFIG.pos_short if pos_side_v5 == 'Sell' else CONFIG.pos_none)

                if side != CONFIG.pos_none and abs(size) > CONFIG.position_qty_epsilon and entry_price >= Decimal("0"): # Check entry price validity
                    position_details = {'side': side, 'qty': abs(size), 'entry_price': entry_price}
                    logger.info(f"{Fore.YELLOW}Position Check: Found ACTIVE {side} position: Qty={position_details['qty']:.8f} @ Entry={position_details['entry_price']:.4f}{Style.RESET_ALL}")
                    return position_details
                else:
                    logger.warning(f"{Fore.YELLOW}Position Check: Found position data but parsed as invalid (Side:{side}, Qty:{size}, Entry:{entry_price}). Treating as flat.{Style.RESET_ALL}")
                    return default_pos

            except Exception as parse_err:
                 logger.warning(f"{Fore.YELLOW}Position Check: Error parsing active position data: {parse_err}. Data: {active_pos}{Style.RESET_ALL}")
                 return default_pos # Return default on parsing error
        else:
            logger.info(f"Position Check: No active One-Way position found for {market_id}.")
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(f"{Fore.YELLOW}Position Check: CCXT Error fetching positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check: Unexpected error fetching positions for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return default_pos # Return default if any error occurs

def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol using Bybit V5 specifics via CCXT."""
    logger.info(f"{Fore.CYAN}Leverage Setting: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    market: Optional[Dict] = None
    try:
        market = exchange.market(symbol)
        if not market or not market.get('contract'):
            logger.error(f"{Fore.RED}Leverage Setting: Cannot set for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False
    except (ccxt.BadSymbol, KeyError, Exception) as e:
         logger.error(f"{Fore.RED}Leverage Setting: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
         return False

    for attempt in range(CONFIG.retry_count):
        try:
            # Bybit V5 'setLeverage' potentially needs 'buyLeverage' and 'sellLeverage' in params for unified margin
            # CCXT might handle this abstraction, but providing explicitly can be safer.
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            # The `leverage` argument to ccxt's `set_leverage` is usually sufficient for V5 via recent CCXT versions.
            # Params might be redundant but included for robustness demonstration.
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            # Response parsing can be tricky, often just checking for exception is enough
            logger.success(f"{Fore.GREEN}Leverage Setting: Successfully set to {leverage}x for {symbol}. Response sample: {str(response)[:100]}...{Style.RESET_ALL}")
            return True
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            # Bybit V5 might return specific codes or messages for no modification needed
            # Example error codes (check Bybit docs): 110044 (leverage not modified)
            if "leverage not modified" in err_str or "leverage is same as requested" in err_str or "110044" in err_str:
                logger.info(f"{Fore.CYAN}Leverage Setting: Already set to {leverage}x for {symbol}.{Style.RESET_ALL}")
                return True
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Exchange error (Attempt {attempt+1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
        except (ccxt.NetworkError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Network/Other error (Attempt {attempt+1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")

        if attempt < CONFIG.retry_count - 1:
            logger.debug(f"Retrying leverage setting in {CONFIG.retry_delay_seconds}s...")
            time.sleep(CONFIG.retry_delay_seconds)
        else:
            logger.error(f"{Fore.RED}Leverage Setting: Failed to set leverage for {symbol} after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")

    return False

def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal") -> Optional[Dict[str, Any]]:
    """
    Closes the specified active position with re-validation using a market order.
    Uses Decimal for quantity, handles precision, and Bybit V5 `reduceOnly`.
    Returns the executed order dict on success, None on failure or if no position exists.
    """
    initial_side = position_to_close.get('side', CONFIG.pos_none)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = get_market_base_currency(symbol)
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}Close Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # --- Re-validate the position just before closing ---
    live_position = get_current_position(exchange, symbol)
    if live_position['side'] == CONFIG.pos_none:
        logger.warning(f"{Fore.YELLOW}Close Position: Re-validation shows NO active position for {symbol}. Aborting closure attempt.{Style.RESET_ALL}")
        if initial_side != CONFIG.pos_none:
            logger.info(f"{Fore.CYAN}Close Position: Discrepancy noted (Bot thought {initial_side}, exchange reports None). State corrected.{Style.RESET_ALL}")
        return None # Nothing to close

    live_amount_to_close = live_position['qty']
    live_position_side = live_position['side']

    # Determine the side needed to close the position
    side_to_execute_close = CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy

    try:
        # Format amount according to market rules BEFORE converting to float for CCXT
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_to_close_precise = Decimal(amount_str)
        amount_float = float(amount_to_close_precise) # CCXT create order often expects float

        if amount_to_close_precise <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Close Position: Closing amount after precision ({amount_str}) is negligible or zero. Aborting.{Style.RESET_ALL}")
            # This might happen if the position is extremely small or due to precision issues
            return None

        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Close Position: Attempting to CLOSE {live_position_side} ({reason}): "
                       f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}")

        # Bybit V5 uses 'reduceOnly' parameter for closing orders
        params = {'reduceOnly': True}
        order = exchange.create_market_order(symbol=symbol, side=side_to_execute_close, amount=amount_float, params=params)

        # --- Parse order response safely using Decimal ---
        order_id = order.get('id')
        status = order.get('status', 'unknown') # Check status if available
        fill_price = safe_decimal_conversion(order.get('average')) # Avg fill price
        filled_qty = safe_decimal_conversion(order.get('filled')) # Amount filled
        cost = safe_decimal_conversion(order.get('cost')) # Total cost in quote currency
        fee = safe_decimal_conversion(order.get('fee', {}).get('cost', '0.0')) # Fee if available

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Close Position: Order ({reason}) for {symbol} PLACED/FILLED(?). "
                       f"ID: {format_order_id(order_id)}, Status: {status}. "
                       f"Filled Qty: {filled_qty:.8f}/{amount_str}, AvgFill: {fill_price:.4f}, Cost: {cost:.2f}, Fee: {fee:.4f}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:{format_order_id(order_id)}")

        # Optional: Wait briefly and re-check position to be absolutely sure it's closed
        # time.sleep(1)
        # final_pos_check = get_current_position(exchange, symbol)
        # if final_pos_check['side'] == CONFIG.pos_none: logger.info("Post-close check confirms position is flat.")
        # else: logger.warning("Post-close check shows position might still exist!")

        return order # Return the order details

    except ccxt.InsufficientFunds as e:
         logger.error(f"{Fore.RED}Close Position ({reason}): Failed for {symbol} - Insufficient Funds: {e}{Style.RESET_ALL}")
         send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insufficient Funds. Check Margin!")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Check for Bybit errors indicating already closed or reduce-only conflict
        # Example codes/messages: 110025 (position is zero), 110045 (order would not reduce position size)
        if ("order would not reduce position size" in err_str or
            "position is zero" in err_str or
            "size is zero" in err_str or
            "110025" in err_str or "110045" in err_str):
             logger.warning(f"{Fore.YELLOW}Close Position: Exchange indicates position already closed/closing ({e}). Assuming successful closure.{Style.RESET_ALL}")
             # Send SMS that it was likely already closed if bot initiated it
             send_sms_alert(f"[{market_base}] Close ({reason}): Exchange reported position already closed/zero.")
             return None # Treat as effectively closed in this case
        else:
            logger.error(f"{Fore.RED}Close Position ({reason}): Exchange error for {symbol}: {e}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Exchange error: {type(e).__name__}. Check logs.")
    except (ccxt.NetworkError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}Close Position ({reason}): Failed for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}. Check logs.")
    return None

def calculate_position_size(equity: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, stop_loss_price: Decimal,
                            leverage: int, symbol: str, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Calculates position size (in base currency) and estimated margin based on risk, using Decimal.
    Returns (quantity_precise, required_margin) or (None, None) on error.
    """
    logger.debug(f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")

    # Validate inputs
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(f"{Fore.RED}Risk Calc: Invalid entry ({entry_price}) or SL price ({stop_loss_price}).{Style.RESET_ALL}")
        return None, None
    if entry_price == stop_loss_price:
         logger.error(f"{Fore.RED}Risk Calc: Entry price cannot equal SL price.{Style.RESET_ALL}")
         return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff <= CONFIG.position_qty_epsilon:
        logger.error(f"{Fore.RED}Risk Calc: Entry and SL prices are too close ({price_diff:.8f}). Increase ATR multiplier or check price data.{Style.RESET_ALL}")
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(f"{Fore.RED}Risk Calc: Invalid risk percentage: {risk_per_trade_pct:.4%}. Must be between 0 and 1.{Style.RESET_ALL}")
        return None, None
    if equity <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.4f}{Style.RESET_ALL}")
        return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}{Style.RESET_ALL}")
        return None, None

    # --- Calculate Risk Amount and Quantity ---
    risk_amount_usdt = equity * risk_per_trade_pct
    # For linear contracts (like BTC/USDT:USDT), risk per unit of base currency = price_diff in USDT
    # Quantity = Total Risk Amount / Risk per Unit
    quantity_raw = risk_amount_usdt / price_diff

    try:
        # Format the raw quantity according to market precision RULES (rounding down)
        # Then convert back to Decimal for internal use
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw) # format_amount should handle rounding down
        quantity_precise = Decimal(quantity_precise_str)
    except (ValueError, InvalidOperation, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Failed precision formatting for quantity {quantity_raw:.8f}. Using raw quantized. Error: {e}{Style.RESET_ALL}")
        # Fallback: Quantize manually if formatting fails, rounding down
        # Determine decimal places from market if possible
        precision_str = exchange.markets[symbol].get('precision', {}).get('amount')
        decimal_places = 8 # Default fallback
        if precision_str:
            try: decimal_places = Decimal(str(precision_str)).normalize().as_tuple().exponent * -1
            except: pass
        quantity_precise = quantity_raw.quantize(Decimal('1e-' + str(decimal_places)), rounding=ROUND_DOWN)

    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Calculated quantity negligible or zero after precision ({quantity_precise:.8f}). RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}{Style.RESET_ALL}")
        return None, None

    # --- Calculate Estimated Margin ---
    position_value_usdt = quantity_precise * entry_price
    required_margin = position_value_usdt / Decimal(leverage)

    logger.debug(f"Risk Calc Result: Qty={quantity_precise:.8f}, RiskAmt={risk_amount_usdt:.4f}, EstValue={position_value_usdt:.4f}, EstMargin={required_margin:.4f}")
    return quantity_precise, required_margin

def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
    """
    Waits for a specific order to reach a 'closed' (filled) status.
    Returns the filled order dict or None if timeout or failed status.
    """
    start_time = time.time()
    order_id_short = format_order_id(order_id)
    logger.info(f"{Fore.CYAN}Waiting for order {order_id_short} ({symbol}) fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")

    while time.time() - start_time < timeout_seconds:
        try:
            # fetch_order is preferred over fetch_orders for specific ID check
            order = exchange.fetch_order(order_id, symbol)
            status = order.get('status')
            logger.debug(f"Order {order_id_short} status check: {status}")

            if status == 'closed':
                logger.success(f"{Fore.GREEN}Order {order_id_short} confirmed FILLED.{Style.RESET_ALL}")
                return order
            elif status in ['canceled', 'rejected', 'expired']:
                logger.error(f"{Fore.RED}Order {order_id_short} reached final FAILED status: '{status}'.{Style.RESET_ALL}")
                return None # Failed state

            # Continue polling if 'open', 'partially_filled' (for market usually goes straight to closed), or None/unknown
            time.sleep(CONFIG.market_order_fill_check_interval) # Check frequently

        except ccxt.OrderNotFound:
            # Can happen briefly after placing or if already closed/canceled and pruned by exchange
            elapsed_time = time.time() - start_time
            if elapsed_time < 5: # Tolerate 'not found' for a short period
                logger.warning(f"{Fore.YELLOW}Order {order_id_short} not found yet (after {elapsed_time:.1f}s). Retrying...{Style.RESET_ALL}")
                time.sleep(0.5) # Slightly longer wait if not found initially
            else:
                logger.error(f"{Fore.RED}Order {order_id_short} not found after {elapsed_time:.1f}s. Assuming failed/pruned.{Style.RESET_ALL}")
                return None # Assume failed if not found after a reasonable time

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(f"{Fore.YELLOW}Error checking order {order_id_short}: {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}")
            time.sleep(1) # Wait longer on API errors
        except Exception as e:
             logger.error(f"{Fore.RED}Unexpected error checking order {order_id_short}: {e}{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             time.sleep(1) # Wait longer on unexpected errors

    # Loop finished without success
    logger.error(f"{Fore.RED}Order {order_id_short} did not fill within {timeout_seconds}s timeout.{Style.RESET_ALL}")
    # Attempt to fetch one last time to see final status
    try:
        final_order_check = exchange.fetch_order(order_id, symbol)
        logger.warning(f"Final status check for timed-out order {order_id_short}: {final_order_check.get('status')}")
    except Exception as final_e:
        logger.warning(f"Could not perform final status check for timed-out order {order_id_short}: {final_e}")
    return None # Timeout

def place_risked_market_order(exchange: ccxt.Exchange, symbol: str, side: str,
                            risk_percentage: Decimal, current_atr: Optional[Decimal], sl_atr_multiplier: Decimal,
                            leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal,
                            tsl_percent: Decimal, tsl_activation_offset_percent: Decimal) -> Optional[Dict[str, Any]]:
    """
    Handles the complete process of placing a risk-calculated market entry order,
    waiting for fill, and then placing exchange-native fixed SL and TSL orders.
    Uses Decimal precision throughout calculations.
    Returns the filled entry order dict on success (even if SL/TSL placement partially fails), None on major failure.
    """
    market_base = get_market_base_currency(symbol)
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}Place Order: Initiating {side.upper()} for {symbol} with Risk/SL/TSL...{Style.RESET_ALL}")

    # --- Pre-computation Checks ---
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid ATR ({current_atr}). Cannot calculate SL or place order.{Style.RESET_ALL}")
        return None

    market: Optional[Dict] = None
    entry_price_estimate: Optional[Decimal] = None
    initial_sl_price_estimate: Optional[Decimal] = None
    final_quantity: Optional[Decimal] = None
    entry_order_id: Optional[str] = None
    filled_entry_order: Optional[Dict[str, Any]] = None
    sl_order_id: Optional[str] = None
    tsl_order_id: Optional[str] = None
    sl_status: str = "Not Placed"
    tsl_status: str = "Not Placed"

    try:
        # === 1. Get Balance, Market Info, Limits ===
        logger.debug("Fetching balance & market details...")
        balance = exchange.fetch_balance()
        market = exchange.market(symbol)
        if not market: raise ValueError(f"Market {symbol} not found in loaded markets.")

        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        min_qty = safe_decimal_conversion(amount_limits.get('min')) if amount_limits.get('min') else None
        max_qty = safe_decimal_conversion(amount_limits.get('max')) if amount_limits.get('max') else None
        min_price = safe_decimal_conversion(price_limits.get('min')) if price_limits.get('min') else None
        # Log limits for debugging
        logger.debug(f"Market Limits: MinQty={min_qty}, MaxQty={max_qty}, MinPrice={min_price}")


        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        # Use 'total' for equity calculation, 'free' for margin check
        usdt_total = safe_decimal_conversion(usdt_balance.get('total'))
        usdt_free = safe_decimal_conversion(usdt_balance.get('free'))
        # Bybit V5 might have different structures, check fetch_balance response structure if needed
        # For simplicity, assume 'total' represents equity if available, else 'free'
        usdt_equity = usdt_total if usdt_total > 0 else usdt_free

        if usdt_equity <= Decimal("0"):
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Zero or invalid equity ({usdt_equity:.4f}). Cannot place order.{Style.RESET_ALL}")
            return None
        if usdt_free < Decimal("0"): # Free shouldn't be negative, but check anyway
             logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid free margin ({usdt_free:.4f}).{Style.RESET_ALL}")
             return None
        logger.debug(f"Balance: Equity={usdt_equity:.4f} {CONFIG.usdt_symbol}, Free={usdt_free:.4f} {CONFIG.usdt_symbol}")

        # === 2. Estimate Entry Price (using shallow OB or ticker) ===
        logger.debug("Estimating entry price...")
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        if side == CONFIG.side_buy and best_ask: entry_price_estimate = best_ask
        elif side == CONFIG.side_sell and best_bid: entry_price_estimate = best_bid
        else:
            try:
                ticker = exchange.fetch_ticker(symbol)
                entry_price_estimate = safe_decimal_conversion(ticker.get('last'))
                if not entry_price_estimate or entry_price_estimate <= 0:
                     raise ValueError(f"Invalid ticker price: {ticker.get('last')}")
            except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
                 logger.error(f"{Fore.RED}Failed to fetch ticker or get valid price for estimation: {e}{Style.RESET_ALL}")
                 return None # Cannot proceed without a price estimate
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === 3. Calculate Initial Stop Loss Price (Estimate) ===
        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (entry_price_estimate - sl_distance) if side == CONFIG.side_buy else (entry_price_estimate + sl_distance)

        # Ensure SL is not below minimum price if limit exists
        if min_price is not None and initial_sl_price_raw < min_price:
            logger.warning(f"{Fore.YELLOW}Calculated SL price {initial_sl_price_raw:.4f} is below min price {min_price}. Adjusting SL to min price.{Style.RESET_ALL}")
            initial_sl_price_raw = min_price

        if initial_sl_price_raw <= 0:
             logger.error(f"{Fore.RED}Invalid Initial SL price calculated ({initial_sl_price_raw:.4f}). Cannot place order.{Style.RESET_ALL}")
             return None

        # Format the estimated SL price using market precision
        initial_sl_price_str_estimate = format_price(exchange, symbol, initial_sl_price_raw)
        initial_sl_price_estimate = Decimal(initial_sl_price_str_estimate)
        logger.info(f"Calculated Initial SL Price (Estimate) ~ {initial_sl_price_estimate:.4f} (ATR Dist: {sl_distance:.4f})")

        # === 4. Calculate Position Size based on Risk & Estimated SL ===
        calc_qty, req_margin_estimate = calculate_position_size(usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate, leverage, symbol, exchange)
        if calc_qty is None or req_margin_estimate is None:
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Failed risk calculation. Cannot determine position size.{Style.RESET_ALL}")
            return None
        final_quantity = calc_qty # Start with risk-based quantity

        # === 5. Apply Max Order Value Cap ===
        pos_value_estimate = final_quantity * entry_price_estimate
        if pos_value_estimate > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}Calculated order value {pos_value_estimate:.4f} USDT exceeds cap {max_order_cap_usdt:.4f}. Capping quantity.{Style.RESET_ALL}")
            capped_quantity_raw = max_order_cap_usdt / entry_price_estimate
            # Format capped quantity according to market rules (round down)
            capped_quantity_str = format_amount(exchange, symbol, capped_quantity_raw)
            final_quantity = Decimal(capped_quantity_str)
            # Recalculate estimated margin based on capped quantity
            req_margin_estimate = (final_quantity * entry_price_estimate) / Decimal(leverage)
            logger.info(f"Quantity capped to {final_quantity:.8f}. New Est Margin ~ {req_margin_estimate:.4f}")

        # === 6. Check Quantity Limits & Margin Availability ===
        if final_quantity <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Final quantity ({final_quantity:.8f}) is negligible or zero after calculations/capping. Aborting.{Style.RESET_ALL}")
            return None
        if min_qty is not None and final_quantity < min_qty:
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Final quantity {final_quantity:.8f} is below market minimum {min_qty}. Aborting.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Qty {final_quantity:.8f} < Min {min_qty}")
            return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"{Fore.YELLOW}Final quantity {final_quantity:.8f} exceeds market maximum {max_qty}. Adjusting to max.{Style.RESET_ALL}")
            final_quantity = max_qty # Use the absolute max allowed by exchange
            # Re-format just in case max_qty needs formatting (unlikely but safe)
            final_quantity = Decimal(format_amount(exchange, symbol, final_quantity))
            # Recalculate final margin estimate
            req_margin_estimate = (final_quantity * entry_price_estimate) / Decimal(leverage)

        # Final margin check with buffer
        req_margin_buffered = req_margin_estimate * margin_check_buffer
        if usdt_free < req_margin_buffered:
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Insufficient FREE margin. Need ~{req_margin_buffered:.4f} (incl. buffer), Have {usdt_free:.4f}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin (Need ~{req_margin_buffered:.2f})")
            return None

        logger.info(f"{Fore.GREEN}Pre-order Checks Passed. Final Qty: {final_quantity:.8f}, Est Margin: {req_margin_estimate:.4f}, Buffered Margin: {req_margin_buffered:.4f}{Style.RESET_ALL}")

        # === 7. Place Entry Market Order ===
        entry_order_details: Optional[Dict[str, Any]] = None
        try:
            qty_float = float(final_quantity) # CCXT requires float for amount
            logger.warning(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
            # For Bybit V5, ensure `reduceOnly` is false or omitted for entry orders
            entry_params = {'reduceOnly': False}
            entry_order_details = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params=entry_params)
            entry_order_id = entry_order_details.get('id')
            if not entry_order_id:
                logger.error(f"{Fore.RED}{Style.BRIGHT}Entry order placed but NO ID returned! Response: {entry_order_details}{Style.RESET_ALL}")
                # Attempt to check position just in case it somehow opened without ID
                time.sleep(1)
                current_pos = get_current_position(exchange, symbol)
                if current_pos['side'] != CONFIG.pos_none:
                     logger.warning(f"{Fore.YELLOW}Position exists despite missing entry ID! Qty: {current_pos['qty']}. Manual check needed!{Style.RESET_ALL}")
                     # Cannot proceed with SL/TSL reliably
                     return None
                raise ValueError("Entry order placement failed to return an ID.") # Critical failure

            logger.success(f"{Fore.GREEN}Market Entry Order submitted. ID: {format_order_id(entry_order_id)}. Waiting for fill confirmation...{Style.RESET_ALL}")

        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError, Exception) as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}")
            return None # Stop the process if entry fails

        # === 8. Wait for Entry Order Fill Confirmation ===
        filled_entry_order = wait_for_order_fill(exchange, entry_order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry_order:
            logger.error(f"{Fore.RED}{Style.BRIGHT}Entry order {format_order_id(entry_order_id)} did NOT fill or failed.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry {format_order_id(entry_order_id)} fill timeout/fail.")
            # Attempt to cancel the potentially stuck order (might fail if already filled/gone)
            try:
                logger.warning(f"Attempting to cancel unfilled/failed entry order {format_order_id(entry_order_id)}...")
                exchange.cancel_order(entry_order_id, symbol)
                logger.info(f"Cancel request sent for {format_order_id(entry_order_id)}.")
            except (ccxt.OrderNotFound, ccxt.ExchangeError) as cancel_e:
                logger.warning(f"Could not cancel order {format_order_id(entry_order_id)} (may already be filled/cancelled): {cancel_e}")
            except Exception as cancel_e:
                logger.error(f"Error cancelling order {format_order_id(entry_order_id)}: {cancel_e}")
            # Check position status again after failed fill/cancel attempt
            time.sleep(1)
            current_pos = get_current_position(exchange, symbol)
            if current_pos['side'] != CONFIG.pos_none:
                 logger.error(f"{Back.RED}{Fore.WHITE}POSITION OPENED despite entry fill failure! Qty: {current_pos['qty']}. Closing immediately!{Style.RESET_ALL}")
                 send_sms_alert(f"[{market_base}] CRITICAL: Position opened on FAILED entry fill! Closing NOW.")
                 close_position(exchange, symbol, current_pos, reason="Emergency Close - Failed Entry Fill")
            return None # Stop process

        # === 9. Extract Actual Fill Details (Crucial: Use Actual Fill Info) ===
        avg_fill_price = safe_decimal_conversion(filled_entry_order.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry_order.get('filled'))
        cost = safe_decimal_conversion(filled_entry_order.get('cost'))
        fee = safe_decimal_conversion(filled_entry_order.get('fee', {}).get('cost', '0.0'))

        # Validate fill details
        if avg_fill_price <= 0 or filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}{Style.BRIGHT}Invalid fill details received for order {format_order_id(entry_order_id)}: Price={avg_fill_price}, Qty={filled_qty}. Cannot proceed with SL/TSL.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid fill details {format_order_id(entry_order_id)}.")
            # Position might be open with bad data, attempt emergency close
            logger.error(f"{Back.RED}{Fore.WHITE}Attempting emergency close due to invalid fill details...{Style.RESET_ALL}")
            # We need quantity to close, if filled_qty is bad, re-fetch position
            current_pos = get_current_position(exchange, symbol)
            if current_pos['side'] != CONFIG.pos_none:
                 close_position(exchange, symbol, current_pos, reason="Emergency Close - Invalid Fill Data")
            else:
                 logger.warning("Position seems already closed or wasn't opened despite invalid fill.")
            return filled_entry_order # Return the problematic order, but signal failure upstream

        logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}ENTRY CONFIRMED: {format_order_id(entry_order_id)}. "
                       f"Filled Qty: {filled_qty:.8f} @ AvgPrice: {avg_fill_price:.4f}. Cost: {cost:.4f}, Fee: {fee:.4f}{Style.RESET_ALL}")

        # --- Post-Fill Actions: Place SL and TSL ---
        # Use the ACTUAL filled quantity and average price for SL/TSL calculations

        # === 10. Calculate ACTUAL Stop Loss Price based on Actual Fill ===
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.side_buy else (avg_fill_price + sl_distance)
        if min_price is not None and actual_sl_price_raw < min_price: actual_sl_price_raw = min_price
        if actual_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}{Style.BRIGHT}Invalid ACTUAL SL price calculated ({actual_sl_price_raw:.4f}) based on fill price {avg_fill_price:.4f}. CANNOT PLACE SL!{Style.RESET_ALL}")
            # CRITICAL: Position is open without SL protection. Attempt emergency close.
            send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price ({actual_sl_price_raw:.4f})! Attempting emergency close.")
            close_position(exchange, symbol, {'side': side, 'qty': filled_qty}, reason="Emergency Close - Invalid SL Calc")
            return filled_entry_order # Return filled entry, but indicate failure state

        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        actual_sl_price_float = float(actual_sl_price_str) # For CCXT param requiring float

        # === 11. Place Initial Fixed Stop Loss (Stop Market Order) ===
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            # Use the actual filled quantity, formatted
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)

            logger.info(f"{Fore.CYAN}Placing Initial Fixed SL ({sl_atr_multiplier}*ATR)... TriggerPx: {actual_sl_price_str}, Qty: {sl_qty_float:.8f}{Style.RESET_ALL}")
            # Bybit V5 stop market order params via CCXT:
            # - type='StopMarket' (or 'Stop')
            # - params={'stopPrice': trigger_price, 'reduceOnly': True, 'basePrice': mark/index/last (optional trigger base)}
            sl_params = {
                'stopPrice': actual_sl_price_float, # The trigger price
                'reduceOnly': True
                # 'triggerDirection': 1 if sl_price > current_price else 2 # Optional: 1=above, 2=below (check CCXT/Bybit docs if needed)
                # 'tpslMode': 'Partial' # Or 'Full' - might be needed depending on account settings
                # 'slOrderType': 'Market' # Ensure it's a market stop loss
            }
            # Using exchange.create_stop_market_order if available and handles params correctly, else create_order
            # Note: CCXT might abstract this; create_order with type='stopMarket' might be preferred.
            if hasattr(exchange, 'create_stop_market_order'):
                 sl_order = exchange.create_stop_market_order(symbol, sl_side, sl_qty_float, actual_sl_price_float, params=sl_params)
            else:
                 sl_order = exchange.create_order(symbol=symbol, type='Stop', side=sl_side, amount=sl_qty_float, params=sl_params) # Check if 'Stop' or 'StopMarket' is correct type alias

            sl_order_id = sl_order.get('id')
            if sl_order_id:
                sl_status = f"Placed (ID: {format_order_id(sl_order_id)}, Trigger: {actual_sl_price_str})"
                logger.success(f"{Fore.GREEN}{sl_status}{Style.RESET_ALL}")
            else:
                 sl_status = f"Placement FAILED (No ID returned). Response: {sl_order}"
                 logger.error(f"{Fore.RED}{sl_status}{Style.RESET_ALL}")
                 send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement (NO ID).")

        except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError, Exception) as e:
            sl_status = f"Placement FAILED: {type(e).__name__} - {e}"
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Initial Fixed SL order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement: {type(e).__name__}")
            # Decide if this is critical enough to close the position. For now, log and continue to TSL attempt.

        # === 12. Place Trailing Stop Loss (if percentage > 0) ===
        if tsl_percent > CONFIG.position_qty_epsilon: # Only place if TSL percentage is meaningful
            try:
                # Calculate TSL activation price based on actual fill
                act_offset = avg_fill_price * tsl_activation_offset_percent
                act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.side_buy else (avg_fill_price - act_offset)
                if min_price is not None and act_price_raw < min_price: act_price_raw = min_price
                if act_price_raw <= 0: raise ValueError(f"Invalid TSL activation price {act_price_raw:.4f}")

                tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
                tsl_act_price_float = float(tsl_act_price_str)
                tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
                # Bybit V5 uses 'trailingStop' for percentage distance, requires value as string percentage (e.g., "0.5" for 0.5%)
                tsl_trail_value_str = str((tsl_percent * Decimal("100")).quantize(Decimal("0.01"))) # Format to standard percentage string e.g. "0.50"
                tsl_qty_str = format_amount(exchange, symbol, filled_qty)
                tsl_qty_float = float(tsl_qty_str)

                logger.info(f"{Fore.CYAN}Placing Trailing SL ({tsl_percent:.2%})... Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}, Qty: {tsl_qty_float:.8f}{Style.RESET_ALL}")
                # Bybit V5 TSL params via CCXT:
                # - type='StopMarket' (or 'Stop') seems correct
                # - params={'trailingStop': percentage_string, 'activePrice': activation_trigger_price, 'reduceOnly': True}
                tsl_params = {
                    'trailingStop': tsl_trail_value_str, # e.g., '0.5' for 0.5%
                    'activePrice': tsl_act_price_float, # Price at which the TSL becomes active
                    'reduceOnly': True,
                    # 'tpslMode': 'Partial' # Or 'Full'
                    # 'slOrderType': 'Market'
                }
                # Use create_order with appropriate type and params
                tsl_order = exchange.create_order(symbol=symbol, type='Stop', side=tsl_side, amount=tsl_qty_float, params=tsl_params) # Check if 'Stop' or 'StopMarket' type

                tsl_order_id = tsl_order.get('id')
                if tsl_order_id:
                     tsl_status = f"Placed (ID: {format_order_id(tsl_order_id)}, Trail: {tsl_trail_value_str}%, ActPx: {tsl_act_price_str})"
                     logger.success(f"{Fore.GREEN}{tsl_status}{Style.RESET_ALL}")
                else:
                    tsl_status = f"Placement FAILED (No ID returned). Response: {tsl_order}"
                    logger.error(f"{Fore.RED}{tsl_status}{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement (NO ID).")

            except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError, Exception) as e:
                tsl_status = f"Placement FAILED: {type(e).__name__} - {e}"
                logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Trailing SL order: {e}{Style.RESET_ALL}")
                logger.debug(traceback.format_exc())
                send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}")
        else:
            tsl_status = "Not Configured (Percentage Zero)"
            logger.info(f"{Fore.CYAN}Trailing SL not configured (percentage is zero or less). Skipping placement.{Style.RESET_ALL}")


        # === 13. Final Summary Log & SMS ===
        logger.info(f"{Back.BLUE}{Fore.WHITE}--- ORDER PLACEMENT SUMMARY ({side.upper()} {symbol}) ---{Style.RESET_ALL}")
        logger.info(f"  Entry: {format_order_id(entry_order_id)} | Filled: {filled_qty:.8f} @ {avg_fill_price:.4f}")
        logger.info(f"  Fixed SL: {sl_status}")
        logger.info(f"  Trailing SL: {tsl_status}")
        logger.info(f"{Back.BLUE}{Fore.WHITE}--- END SUMMARY ---{Style.RESET_ALL}")

        # Send comprehensive SMS only if entry was successful
        sms_summary = (f"[{market_base}] {side.upper()} {filled_qty:.6f}@{avg_fill_price:.3f}. "
                       f"SL:{(actual_sl_price_str if sl_order_id else 'FAIL')}. "
                       f"TSL:{('%' + tsl_trail_value_str if tsl_order_id else ('FAIL' if tsl_percent > 0 else 'OFF'))}. "
                       f"E:{format_order_id(entry_order_id)}")
        send_sms_alert(sms_summary)

        # Return the filled entry order details, signalling overall success of getting into position
        return filled_entry_order

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        # Catch errors occurring before entry placement or during setup phases
        logger.error(f"{Fore.RED}{Style.BRIGHT}Place Order ({side.upper()}): Overall process failed before/during entry: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Pre-entry setup/checks failed: {type(e).__name__}")
        # Ensure we didn't somehow open a position
        if entry_order_id and not filled_entry_order:
             logger.warning("Checking position status after setup failure...")
             time.sleep(1)
             current_pos = get_current_position(exchange, symbol)
             if current_pos['side'] != CONFIG.pos_none:
                  logger.error(f"{Back.RED}{Fore.WHITE}POSITION OPENED despite setup failure! Qty: {current_pos['qty']}. Closing immediately!{Style.RESET_ALL}")
                  send_sms_alert(f"[{market_base}] CRITICAL: Position opened on FAILED setup! Closing NOW.")
                  close_position(exchange, symbol, current_pos, reason="Emergency Close - Failed Order Setup")
    return None # Indicate overall failure

def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Attempts to cancel all open orders (limit, stop, etc.) for the specified symbol."""
    logger.info(f"{Fore.CYAN}Order Cancel: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    cancelled_count, failed_count = 0, 0
    market_base = get_market_base_currency(symbol)
    try:
        # Use fetch_open_orders (might need params for stops on some exchanges/ccxt versions)
        # Bybit V5 might require category, let CCXT handle default if possible
        if not exchange.has.get('fetchOpenOrders'):
            logger.warning(f"{Fore.YELLOW}Order Cancel: fetchOpenOrders not supported by {exchange.id}. Cannot automatically cancel.{Style.RESET_ALL}")
            return

        logger.debug("Fetching open orders...")
        # Add params if needed, e.g., {'category': 'linear'} or potentially type filters
        open_orders = exchange.fetch_open_orders(symbol) # Add params={} if needed

        if not open_orders:
            logger.info(f"{Fore.CYAN}Order Cancel: No open orders found for {symbol}.{Style.RESET_ALL}")
            return

        logger.warning(f"{Fore.YELLOW}Order Cancel: Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...{Style.RESET_ALL}")

        for order in open_orders:
            order_id = order.get('id')
            order_info = f"ID: {format_order_id(order_id)} ({order.get('type', 'N/A')} {order.get('side', 'N/A')} Qty:{order.get('amount', 'N/A')} Px:{order.get('price', 'N/A')} StopPx:{order.get('stopPrice', 'N/A')})"
            if order_id:
                try:
                    logger.debug(f"Cancelling order: {order_info}")
                    exchange.cancel_order(order_id, symbol)
                    logger.info(f"{Fore.CYAN}Order Cancel: Success for {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1
                    time.sleep(0.1) # Small delay between cancellations to avoid rate limits
                except ccxt.OrderNotFound:
                    logger.warning(f"{Fore.YELLOW}Order Cancel: Order not found (already closed/cancelled?): {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1 # Treat as cancelled if not found during batch cancel
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.error(f"{Fore.RED}Order Cancel: FAILED for {order_info}: {type(e).__name__} - {e}{Style.RESET_ALL}")
                    failed_count += 1
                except Exception as e:
                     logger.error(f"{Fore.RED}Order Cancel: Unexpected error for {order_info}: {e}{Style.RESET_ALL}")
                     failed_count += 1
            else:
                logger.warning(f"Order Cancel: Found order without ID: {order}") # Should not happen

        log_level = logging.INFO if failed_count == 0 else logging.WARNING
        logger.log(log_level, f"{Fore.CYAN}Order Cancel: Finished for {symbol}. Cancelled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
        if failed_count > 0:
            send_sms_alert(f"[{market_base}] WARNING: Failed to cancel {failed_count} order(s) during {reason}. Check manually.")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(f"{Fore.RED}Order Cancel: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
    except Exception as e:
         logger.error(f"{Fore.RED}Order Cancel: Unexpected error during cancel process: {e}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())

# --- Strategy Signal Generation ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """
    Generates entry/exit signals based on the selected strategy and indicator columns in the DataFrame.
    Returns a dict: {'enter_long': bool, 'enter_short': bool, 'exit_long': bool, 'exit_short': bool, 'exit_reason': str}
    """
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit Signal"}
    required_rows = 2 # Need at least current and previous row for comparisons/crosses

    if df is None or len(df) < required_rows:
        logger.warning(f"Signal Gen: Insufficient data ({len(df) if df is not None else 0} rows, need {required_rows})")
        return signals # Not enough data to generate signals

    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            # Entry: Primary ST flips long AND confirmation ST is also long
            if pd.notna(last['st_long']) and last['st_long'] and pd.notna(last['confirm_trend']) and last['confirm_trend']:
                signals['enter_long'] = True
            # Entry: Primary ST flips short AND confirmation ST is also short
            if pd.notna(last['st_short']) and last['st_short'] and pd.notna(last['confirm_trend']) and not last['confirm_trend']:
                signals['enter_short'] = True
            # Exit Long: Primary ST flips short
            if pd.notna(last['st_short']) and last['st_short']:
                signals['exit_long'] = True
                signals['exit_reason'] = "Primary ST Short Flip"
            # Exit Short: Primary ST flips long
            if pd.notna(last['st_long']) and last['st_long']:
                signals['exit_short'] = True
                signals['exit_reason'] = "Primary ST Long Flip"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now, d_now, mom_now = last.get('stochrsi_k'), last.get('stochrsi_d'), last.get('momentum')
            k_prev, d_prev = prev.get('stochrsi_k'), prev.get('stochrsi_d')
            # Check if all needed values are valid Decimals
            if not all(isinstance(val, Decimal) for val in [k_now, d_now, mom_now, k_prev, d_prev]):
                 logger.debug("Signal Gen (StochRSI/Mom): Skipping due to missing/invalid indicator data.")
                 return signals

            # Entry Long: K crosses above D in oversold zone, with positive momentum
            if k_prev <= d_prev and k_now > d_now and k_now < CONFIG.stochrsi_oversold and mom_now > CONFIG.position_qty_epsilon:
                signals['enter_long'] = True
            # Entry Short: K crosses below D in overbought zone, with negative momentum
            if k_prev >= d_prev and k_now < d_now and k_now > CONFIG.stochrsi_overbought and mom_now < -CONFIG.position_qty_epsilon:
                signals['enter_short'] = True
            # Exit Long: K crosses below D (anywhere)
            if k_prev >= d_prev and k_now < d_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "StochRSI K crossed below D"
            # Exit Short: K crosses above D (anywhere)
            if k_prev <= d_prev and k_now > d_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "StochRSI K crossed above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            fish_now, sig_now = last.get('ehlers_fisher'), last.get('ehlers_signal')
            fish_prev, sig_prev = prev.get('ehlers_fisher'), prev.get('ehlers_signal')
            # Signal line might be NA if signal length is 1 or less
            use_signal = isinstance(sig_now, Decimal) and isinstance(sig_prev, Decimal)

            if not isinstance(fish_now, Decimal) or not isinstance(fish_prev, Decimal):
                 logger.debug("Signal Gen (EhlersFisher): Skipping due to missing/invalid Fisher line data.")
                 return signals

            if use_signal:
                # Entry Long: Fisher crosses above Signal
                if fish_prev <= sig_prev and fish_now > sig_now: signals['enter_long'] = True
                # Entry Short: Fisher crosses below Signal
                if fish_prev >= sig_prev and fish_now < sig_now: signals['enter_short'] = True
                # Exit Long: Fisher crosses below Signal
                if fish_prev >= sig_prev and fish_now < sig_now: signals['exit_long'] = True; signals['exit_reason'] = "Ehlers Fisher crossed below Signal"
                # Exit Short: Fisher crosses above Signal
                if fish_prev <= sig_prev and fish_now > sig_now: signals['exit_short'] = True; signals['exit_reason'] = "Ehlers Fisher crossed above Signal"
            else: # Strategy using Fisher crossing zero (or previous value) if no signal line
                 logger.debug("Signal Gen (EhlersFisher): Using Fisher line crossover (no signal line).")
                 # Entry Long: Fisher crosses above previous Fisher (or zero)
                 if fish_prev <= Decimal("0") and fish_now > Decimal("0"): signals['enter_long'] = True # Example: Zero cross
                 # Entry Short: Fisher crosses below previous Fisher (or zero)
                 if fish_prev >= Decimal("0") and fish_now < Decimal("0"): signals['enter_short'] = True # Example: Zero cross
                 # Exit Long: Fisher crosses below previous Fisher (or zero)
                 if fish_prev >= Decimal("0") and fish_now < Decimal("0"): signals['exit_long'] = True; signals['exit_reason'] = "Ehlers Fisher crossed below previous/zero"
                 # Exit Short: Fisher crosses above previous Fisher (or zero)
                 if fish_prev <= Decimal("0") and fish_now > Decimal("0"): signals['exit_short'] = True; signals['exit_reason'] = "Ehlers Fisher crossed above previous/zero"

        # --- Ehlers MA Cross Logic (Placeholder EMA Cross) ---
        elif strategy_name == "EHLERS_MA_CROSS":
            fast_ma_now, slow_ma_now = last.get('fast_ema'), last.get('slow_ema')
            fast_ma_prev, slow_ma_prev = prev.get('fast_ema'), prev.get('slow_ema')
            if not all(isinstance(val, Decimal) for val in [fast_ma_now, slow_ma_now, fast_ma_prev, slow_ma_prev]):
                 logger.debug("Signal Gen (EhlersMA Cross): Skipping due to missing/invalid MA data.")
                 return signals

            # Entry Long: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['enter_long'] = True
            # Entry Short: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['enter_short'] = True
            # Exit Long: Fast MA crosses below Slow MA
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
                signals['exit_long'] = True
                signals['exit_reason'] = "Fast MA crossed below Slow MA"
            # Exit Short: Fast MA crosses above Slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
                signals['exit_short'] = True
                signals['exit_reason'] = "Fast MA crossed above Slow MA"

    except KeyError as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Missing expected indicator column in DataFrame: {e}. Strategy '{strategy_name}' cannot run.{Style.RESET_ALL}")
        # Prevent signals if data is missing
        signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Missing Indicator Data"}
    except Exception as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Unexpected exception during signal calculation for strategy '{strategy_name}': {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        # Prevent signals on unexpected error
        signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Calculation Error"}

    return signals

# --- Trading Logic ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """
    Executes the main trading logic for one cycle based on the selected strategy.
    1. Calculates indicators.
    2. Checks position status.
    3. Generates strategy signals.
    4. Handles exits based on signals.
    5. Handles entries based on signals and confirmations (Volume, Order Book).
    """
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    market_base = get_market_base_currency(symbol)
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Check Cycle [{cycle_count}] ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # --- Data Sufficiency Check ---
    # Determine required rows based on the longest lookback needed by *any* indicator potentially used
    # This is a simplification; could be dynamic based on selected strategy, but safer this way.
    all_lookbacks = [
        CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + CONFIG.stochrsi_k_period + CONFIG.stochrsi_d_period, # Sum for full StochRSI calc depth
        CONFIG.momentum_length,
        CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length,
        CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
        CONFIG.atr_calculation_period, CONFIG.volume_ma_period
    ]
    required_rows = max(all_lookbacks) + CONFIG.api_fetch_limit_buffer # Add buffer for stability

    if df is None or len(df) < required_rows:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0} rows, need ~{required_rows}). Skipping cycle.{Style.RESET_ALL}")
        return

    action_taken_this_cycle: bool = False
    try:
        # === 1. Calculate ALL Indicators ===
        # Calculate all indicators potentially needed by any strategy.
        # The signal generation function will pick the ones relevant to the selected strategy.
        logger.debug("Calculating indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period) # Placeholder EMA
        df, vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")

        # === 2. Validate Base Requirements for Trading ===
        last = df.iloc[-1]
        current_price = safe_decimal_conversion(last['close'])
        if pd.isna(current_price) or current_price <= 0:
            logger.warning(f"{Fore.YELLOW}Trade Logic: Last candle close price is invalid ({current_price}). Skipping cycle.{Style.RESET_ALL}")
            return

        # Check if ATR is valid for SL calculation (needed for *new* entries)
        can_calculate_sl = current_atr is not None and current_atr > CONFIG.position_qty_epsilon
        if not can_calculate_sl:
             # Log warning but allow cycle to continue for potential position management/exits
            logger.warning(f"{Fore.YELLOW}Trade Logic: Invalid ATR ({current_atr}). Cannot calculate SL for new entries.{Style.RESET_ALL}")

        # === 3. Get Position & Analyze Order Book (if needed) ===
        position = get_current_position(exchange, symbol)
        position_side = position['side']
        position_qty = position['qty']
        position_entry_price = position['entry_price']

        # Fetch OB per cycle only if configured, otherwise fetch later if needed for entry confirmation
        ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit) if CONFIG.fetch_order_book_per_cycle else None

        # === 4. Log Current State ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        logger.info(f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): {current_atr:.5f}" if current_atr else f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): N/A")
        logger.info(f"State | Volume: Ratio={vol_ratio:.2f if vol_ratio else 'N/A'}, Spike={vol_spike} (EntryReq={CONFIG.require_volume_spike_for_entry})")
        # Log key indicators for the *active* strategy
        # Example for Dual Supertrend:
        if CONFIG.strategy_name == "DUAL_SUPERTREND":
             st_trend = last.get('trend')
             st_confirm_trend = last.get('confirm_trend')
             st_log = f"ST Trend: {'Up' if st_trend else ('Down' if st_trend == False else 'NA')}"
             st_confirm_log = f"Confirm Trend: {'Up' if st_confirm_trend else ('Down' if st_confirm_trend == False else 'NA')}"
             logger.info(f"State | Strategy ({CONFIG.strategy_name}): {st_log}, {st_confirm_log}")
        # Add similar specific logging for other strategies if desired

        logger.info(f"State | OrderBook: Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'}, Spread={spread:.4f if spread else 'N/A'} (Fetched={ob_data is not None})")
        logger.info(f"State | Position: Side={position_side}, Qty={position_qty:.8f}, Entry={position_entry_price:.4f}")

        # === 5. Generate Strategy Signals ===
        strategy_signals = generate_signals(df.copy(), CONFIG.strategy_name) # Pass copy to avoid mutation issues
        logger.debug(f"Strategy Signals ({CONFIG.strategy_name}): {strategy_signals}")

        # === 6. Execute Exit Actions (If in Position) ===
        if position_side != CONFIG.pos_none:
            should_exit = (position_side == CONFIG.pos_long and strategy_signals['exit_long']) or \
                          (position_side == CONFIG.pos_short and strategy_signals['exit_short'])

            if should_exit:
                exit_reason = strategy_signals['exit_reason']
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}*** STRATEGY EXIT SIGNAL: Closing {position_side} due to {exit_reason} ***{Style.RESET_ALL}")
                # Cancel existing SL/TSL orders BEFORE sending the closing market order
                cancel_open_orders(exchange, symbol, reason=f"Cancel SL/TSL before {exit_reason} Exit")
                time.sleep(0.5) # Brief pause after cancel before closing
                close_result = close_position(exchange, symbol, position, reason=exit_reason)

                if close_result:
                    action_taken_this_cycle = True
                    logger.info(f"Pausing for {CONFIG.post_close_delay_seconds}s after closing position...")
                    time.sleep(CONFIG.post_close_delay_seconds)
                else:
                    logger.error(f"{Fore.RED}Failed to execute close order for {position_side} exit signal! Manual check required.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] CRITICAL: Failed to CLOSE {position_side} on signal! Check position!")
                return # Exit cycle after attempting close (successful or not)
            else:
                 # Still in position, no strategy exit signal
                 logger.info(f"Holding {position_side} position. No strategy exit signal. Relying on exchange SL/TSL.")
                 return # No further action needed this cycle

        # === 7. Check & Execute Entry Actions (Only if Flat) ===
        # If we reach here, position_side == CONFIG.pos_none

        # Check if we can place new orders (ATR must be valid for SL calc)
        if not can_calculate_sl:
             logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter trade because ATR is invalid for SL calculation.{Style.RESET_ALL}")
             return

        logger.debug("Checking entry signals (currently flat)...")
        enter_long_signal = strategy_signals['enter_long']
        enter_short_signal = strategy_signals['enter_short']
        potential_entry = enter_long_signal or enter_short_signal

        if not potential_entry:
            logger.info("No entry signal from strategy. Holding cash.")
            return

        # --- Check Confirmation Conditions ---
        # Fetch OB now if not fetched per cycle and a potential entry exists
        if not CONFIG.fetch_order_book_per_cycle and potential_entry and ob_data is None:
            logger.debug("Potential entry signal detected, fetching OB for confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None # Update ratio if fetched
            logger.info(f"State Update | OrderBook: Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'} (Fetched on demand)")


        # Evaluate Order Book confirmation
        ob_available = ob_data is not None and bid_ask_ratio is not None
        passes_long_ob = not potential_entry or (ob_available and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
        passes_short_ob = not potential_entry or (ob_available and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
        # Note: The logic `not potential_entry or (...)` means OB check is effectively skipped if no entry signal exists.
        # If OB is required, this check should be tied to the specific long/short signal. Let's refine:
        ob_confirm_needed = True # Assume confirmation is always desired if signal exists (can be made configurable)
        passes_long_ob_final = not ob_confirm_needed or (ob_available and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
        passes_short_ob_final = not ob_confirm_needed or (ob_available and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
        ob_log = f"OB Check: Needed={ob_confirm_needed}, Avail={ob_available}, Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'} -> LongOK={passes_long_ob_final}, ShortOK={passes_short_ob_final}"
        logger.debug(ob_log)

        # Evaluate Volume confirmation
        vol_confirm_needed = CONFIG.require_volume_spike_for_entry
        passes_volume = not vol_confirm_needed or vol_spike # Passes if not needed OR spike occurred
        vol_log = f"Vol Check: Needed={vol_confirm_needed}, SpikeMet={vol_spike} -> OK={passes_volume}"
        logger.debug(vol_log)

        # --- Combine Strategy Signal with Confirmations ---
        execute_long_entry = enter_long_signal and passes_long_ob_final and passes_volume
        execute_short_entry = enter_short_signal and passes_short_ob_final and passes_volume

        logger.info(f"Final Entry Decision | Long: Signal={enter_long_signal}, OB OK={passes_long_ob_final}, Vol OK={passes_volume} => EXECUTE={execute_long_entry}")
        logger.info(f"Final Entry Decision | Short: Signal={enter_short_signal}, OB OK={passes_short_ob_final}, Vol OK={passes_volume} => EXECUTE={execute_short_entry}")

        # --- Execute Entry ---
        if execute_long_entry:
            logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            # Cancel any lingering orders (shouldn't exist if flat, but safety check)
            cancel_open_orders(exchange, symbol, "Before Long Entry")
            time.sleep(0.5)
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_buy, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        elif execute_short_entry:
            logger.success(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, "Before Short Entry")
            time.sleep(0.5)
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_sell, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        else:
             # Log if there was a signal but confirmations failed
             if potential_entry and not (execute_long_entry or execute_short_entry):
                 logger.warning(f"{Fore.YELLOW}Entry signal received but FAILED confirmation checks (Vol/OB). Holding cash.{Style.RESET_ALL}")
             elif not action_taken_this_cycle:
                 # This case should have been caught earlier if no potential_entry signal
                 logger.info("No confirmed entry signal. Holding cash.")

    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] CRITICAL ERROR in trade_logic: {type(e).__name__}. Check logs!")
    finally:
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Check End [{cycle_count}]: {symbol} =========={Style.RESET_ALL}\n")

# --- Graceful Shutdown ---
def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str]) -> None:
    """Attempts to close any open position and cancel all orders before exiting."""
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit sequence...{Style.RESET_ALL}")
    market_base = get_market_base_currency(symbol) if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown initiated. Attempting cleanup...")

    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange instance or symbol not available. Cannot perform cleanup.{Style.RESET_ALL}")
        return

    try:
        # 1. Cancel All Open Orders First
        # This prevents SL/TSL orders from potentially triggering while we try to close manually
        logger.info("Shutdown: Cancelling all open orders...")
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(1) # Allow time for cancellations to be processed by the exchange

        # 2. Check Current Position Status
        logger.info("Shutdown: Checking for active position...")
        position = get_current_position(exchange, symbol)

        # 3. Close Position if Active
        if position['side'] != CONFIG.pos_none:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} position found (Qty: {position['qty']:.8f}). Attempting to close...{Style.RESET_ALL}")
            close_result = close_position(exchange, symbol, position, reason="Shutdown Request")

            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s for confirmation...{Style.RESET_ALL}")
                time.sleep(CONFIG.post_close_delay_seconds * 2) # Wait longer to allow market closure
                # Final check to confirm closure
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.pos_none:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position successfully confirmed CLOSED.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] Position confirmed CLOSED during shutdown.")
                else:
                    # This is a critical situation
                    logger.error(f"{Back.RED}{Fore.WHITE}Shutdown Error: FAILED TO CONFIRM position closure after placing order. "
                                 f"Final state: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] CRITICAL ERROR: Failed to CONFIRM closure on shutdown! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. CHECK MANUALLY!")
            else:
                # Failed to even place the close order
                logger.error(f"{Back.RED}{Fore.WHITE}Shutdown Error: Failed to PLACE close order for active position. MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL ERROR: Failed to PLACE close order on shutdown. CHECK MANUALLY!")
        else:
            logger.info(f"{Fore.GREEN}Shutdown: No active position found. No closure needed.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] No active position found during shutdown.")

    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown: Error during cleanup sequence: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] Error during shutdown cleanup: {type(e).__name__}")
    finally:
        logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Pyrmethus Shutdown Sequence Complete ---{Style.RESET_ALL}")

# --- Main Execution ---
# Global variable to track cycle count for logging
cycle_count: int = 0

def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    global cycle_count # Allow modification of global cycle count
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.1 Initializing ({start_time_str}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Warding Runes: Initial ATR Stop Loss + Exchange Native Trailing Stop ---{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE ENGAGED - EXTREME RISK INVOLVED !!! ---{Style.RESET_ALL}")

    exchange: Optional[ccxt.Exchange] = None
    symbol: Optional[str] = None
    run_bot: bool = True

    try:
        # === Initialization ===
        exchange = initialize_exchange()
        if not exchange:
            logger.critical("Exchange initialization failed. Exiting.")
            return # Exit if exchange setup fails

        # === Symbol and Leverage Setup ===
        try:
            # Use configured symbol directly, ensuring it's loaded correctly
            symbol_to_use = CONFIG.symbol
            logger.info(f"Attempting to use symbol: {symbol_to_use}")
            market = exchange.market(symbol_to_use) # Raises BadSymbol if not found
            symbol = market['symbol'] # Use the unified symbol from CCXT (e.g., BTC/USDT:USDT)
            market_base = get_market_base_currency(symbol) # For alerts

            if not market.get('contract'):
                raise ValueError(f"Market '{symbol}' is not a contract/futures market.")
            logger.info(f"{Fore.GREEN}Using Symbol: {symbol} (Type: {market.get('type', 'N/A')}, ID: {market.get('id')}){Style.RESET_ALL}")

            # Set leverage (crucial for futures)
            if not set_leverage(exchange, symbol, CONFIG.leverage):
                 raise RuntimeError("Leverage setup failed after retries.")

        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"Symbol/Leverage setup failed for '{CONFIG.symbol}': {e}")
            send_sms_alert(f"[Pyrmethus] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return
        except Exception as e:
            logger.critical(f"Unexpected error during symbol/leverage setup: {e}")
            logger.debug(traceback.format_exc())
            send_sms_alert("[Pyrmethus] CRITICAL: Unexpected setup error. Exiting.")
            return

        # === Log Configuration Summary ===
        logger.info(f"{Fore.MAGENTA}--- Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"  Symbol: {symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        logger.info(f"  Strategy: {CONFIG.strategy_name}")
        # Log relevant strategy params concisely
        if CONFIG.strategy_name == "DUAL_SUPERTREND": logger.info(f"    Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": logger.info(f"    Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER": logger.info(f"    Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": logger.info(f"    Params: FastMA={CONFIG.ehlers_fast_period}, SlowMA={CONFIG.ehlers_slow_period} (Placeholder EMA)")
        logger.info(f"  Risk: {CONFIG.risk_per_trade_percentage:.3%} / trade")
        logger.info(f"  Sizing: MaxPosValue={CONFIG.max_order_usdt_amount:.2f} USDT, MarginBuffer={CONFIG.required_margin_buffer:.1%}")
        logger.info(f"  Initial SL: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})")
        logger.info(f"  Trailing SL: {CONFIG.trailing_stop_percentage:.2%} (Trail), {CONFIG.trailing_stop_activation_offset_percent:.2%} (Activation Offset)")
        logger.info(f"  Confirmations:")
        logger.info(f"    Vol: Required={CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"    OB: FetchPerCycle={CONFIG.fetch_order_book_per_cycle}, Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short}")
        logger.info(f"  Timing: Sleep={CONFIG.sleep_seconds}s, FillTimeout={CONFIG.order_fill_timeout_seconds}s")
        logger.info(f"  Alerts: SMS={CONFIG.enable_sms_alerts} (To: {CONFIG.sms_recipient_number or 'Not Set'})")
        logger.info(f"  Logging Level: {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'-'*30}{Style.RESET_ALL}")

        send_sms_alert(f"[{market_base}] Pyrmethus Bot started. Strategy: {CONFIG.strategy_name}. SL: ATR+TSL. Risk: {CONFIG.risk_per_trade_percentage:.2%}. Live Trading!")

        # === Main Trading Loop ===
        logger.info(f"{Fore.GREEN}{Style.BRIGHT}--- Starting Main Trading Loop ---{Style.RESET_ALL}")
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Start ---{Style.RESET_ALL}")

            try:
                # Determine required data length dynamically based on *all* potential indicators
                all_lookbacks = [
                    CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
                    CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + CONFIG.stochrsi_k_period + CONFIG.stochrsi_d_period,
                    CONFIG.momentum_length,
                    CONFIG.ehlers_fisher_length + CONFIG.ehlers_fisher_signal_length,
                    CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
                    CONFIG.atr_calculation_period, CONFIG.volume_ma_period
                ]
                # Fetch enough data for the longest lookback plus some buffer
                data_limit = max(all_lookbacks) + CONFIG.api_fetch_limit_buffer + 50 # Increased buffer

                # Fetch Market Data
                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                if df is not None and not df.empty:
                    # Execute Trade Logic
                    trade_logic(exchange, symbol, df.copy()) # Pass a copy to prevent accidental mutation
                else:
                    logger.warning(f"{Fore.YELLOW}No valid market data returned for {symbol} ({CONFIG.interval}). Skipping trade logic for cycle {cycle_count}. Check connection or symbol.{Style.RESET_ALL}")
                    # Optional: Add a longer sleep here if data fetching fails repeatedly
                    # time.sleep(CONFIG.sleep_seconds * 2)

            # --- Robust Error Handling within the Loop ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. Sleeping for {CONFIG.sleep_seconds * 5}s...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * 5)
                send_sms_alert(f"[{market_base}] WARNING: Rate limit hit!")
            except ccxt.NetworkError as e:
                logger.warning(f"{Fore.YELLOW}Network Error in main loop: {e}. Will retry next cycle.{Style.RESET_ALL}")
                # Consider a slightly longer sleep or connection check here if persistent
                time.sleep(CONFIG.sleep_seconds) # Standard sleep before next attempt
            except ccxt.ExchangeNotAvailable as e:
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange Not Available: {e}. Sleeping for {CONFIG.sleep_seconds * 10}s...{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds * 10)
                send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable!")
            except ccxt.AuthenticationError as e:
                # This is critical and should stop the bot
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Authentication Error during operation: {e}. Stopping bot NOW.{Style.RESET_ALL}")
                run_bot = False # Signal loop termination
                send_sms_alert(f"[{market_base}] CRITICAL: Authentication Error! Bot stopping NOW.")
            except ccxt.ExchangeError as e: # Catch other specific exchange errors
                logger.error(f"{Fore.RED}Unhandled Exchange Error in main loop: {e}{Style.RESET_ALL}")
                logger.debug(f"Exchange Error Details: {traceback.format_exc()}")
                send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}. Check logs.")
                time.sleep(CONFIG.sleep_seconds) # Sleep before retrying
            except Exception as e:
                # Catch-all for truly unexpected issues
                logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL ERROR IN MAIN LOOP: {e} !!!{Style.RESET_ALL}")
                run_bot = False # Stop bot on unknown critical errors
                send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Bot stopping NOW.")

            # --- Loop Delay Calculation ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_duration = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} duration: {elapsed:.2f}s. Sleeping for: {sleep_duration:.2f}s.")
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Shutting down gracefully...{Style.RESET_ALL}")
        run_bot = False # Ensure loop terminates cleanly
    except Exception as e:
         # Catch errors during initial setup phase (before main loop)
         logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Critical error during bot initialization phase: {e}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())
         send_sms_alert(f"[Pyrmethus] CRITICAL SETUP ERROR: {type(e).__name__}! Bot failed to start.")
         run_bot = False # Ensure shutdown sequence runs even if loop never started
    finally:
        # --- Graceful Shutdown Sequence ---
        if exchange and symbol: # Only run cleanup if exchange and symbol were initialized
            graceful_shutdown(exchange, symbol)
        else:
            logger.warning("Shutdown: Exchange or symbol not initialized, skipping cleanup.")

        market_base_final = get_market_base_currency(symbol) if symbol else "Bot"
        send_sms_alert(f"[{market_base_final}] Pyrmethus bot process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ({time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

