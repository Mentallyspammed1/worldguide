#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.0 (Precision & Strategy Selection)
# Conjures high-frequency trades on Bybit Futures with enhanced precision and adaptable strategies.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.0.0 (Unified: Selectable Strategies + Precision + Native SL/TSL)

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
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

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
getcontext().prec = 18 # Set Decimal precision (adjust as needed)

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
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # 0.5%
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN) # 5% buffer

        # --- Trailing Stop Loss (Exchange Native) ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # 0.5% trail
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN) # 0.1% offset

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
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN) # Default to 1

        # --- Ehlers MA Cross Parameters ---
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN)

        # --- Volume Analysis ---
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW)
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW)
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW)

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN)

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA)

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = 10000
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW)

        # --- Internal Constants ---
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
logging.Logger.success = log_success

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
    """Safely converts a value to Decimal, returning default if conversion fails."""
    try:
        return Decimal(str(value)) if value is not None else default
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' to Decimal, using default {default}")
        return default

def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """Returns the last 6 characters of an order ID or 'N/A'."""
    return str(order_id)[-6:] if order_id else "N/A"

# --- Precision Formatting ---
def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal]) -> str:
    """Formats price according to market precision rules."""
    try:
        # CCXT formatting methods often expect float input
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(price)).normalize()) # Fallback to Decimal string

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal]) -> str:
    """Formats amount according to market precision rules."""
    try:
        # CCXT formatting methods often expect float input
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(amount)).normalize()) # Fallback to Decimal string

# --- Termux SMS Alert Function ---
def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API."""
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled.")
        return False
    if not CONFIG.sms_recipient_number:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")
        return False
    try:
        # Use shlex.quote for message safety, though direct passing is usually fine
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
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}")
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
        logger.critical("API keys missing in .env file.")
        send_sms_alert("[ScalpBot] CRITICAL: API keys missing. Bot stopped.")
        return None
    try:
        exchange = ccxt.bybit({
            "apiKey": CONFIG.api_key,
            "secret": CONFIG.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear", # Assuming USDT perpetuals
                "recvWindow": CONFIG.default_recv_window,
                "adjustForTimeDifference": True,
            },
        })
        logger.debug("Loading markets...")
        exchange.load_markets(True) # Force reload
        logger.debug("Fetching initial balance...")
        exchange.fetch_balance() # Initial check
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}CCXT Bybit Session Initialized (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}")
        send_sms_alert("[ScalpBot] Initialized & authenticated successfully.")
        return exchange
    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}. Check keys/IP/permissions.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error on init: {e}. Check connection/Bybit status.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Bot stopped.")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error on init: {e}. Check Bybit status/API docs.")
        send_sms_alert(f"[ScalpBot] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
    except Exception as e:
        logger.critical(f"Unexpected error during init: {e}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")
    return None

# --- Indicator Calculation Functions ---
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returns Decimal where applicable."""
    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    st_col = f"SUPERT_{length}_{float(multiplier)}" # pandas_ta uses float in name
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Indicator Calc ({col_prefix}ST): Invalid input (Len: {len(df) if df is not None else 0}, Need: {length + 1}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df

    try:
        # pandas_ta expects float multiplier
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if st_col not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col}, {st_trend_col}")

        # Convert Supertrend value to Decimal
        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1 # Boolean
        prev_trend = df[st_trend_col].shift(1)
        df[f"{col_prefix}st_long"] = (prev_trend == -1) & (df[st_trend_col] == 1) # Boolean
        df[f"{col_prefix}st_short"] = (prev_trend == 1) & (df[st_trend_col] == -1) # Boolean

        raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col]
        df.drop(columns=raw_st_cols, errors='ignore', inplace=True)

        last_st_val = df[f'{col_prefix}supertrend'].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = 'Up' if df[f'{col_prefix}trend'].iloc[-1] else 'Down'
            signal = 'LONG' if df[f'{col_prefix}st_long'].iloc[-1] else ('SHORT' if df[f'{col_prefix}st_short'].iloc[-1] else 'None')
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Trend={last_trend}, Val={last_st_val:.4f}, Signal={signal}")
        else:
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df

def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> Dict[str, Optional[Decimal]]:
    """Calculates ATR, Volume MA, checks spikes. Returns Decimals."""
    results: Dict[str, Optional[Decimal]] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len)

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (Vol/ATR): Invalid input (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}")
        return results

    try:
        # Calculate ATR
        atr_col = f"ATRr_{atr_len}"
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr): results["atr"] = safe_decimal_conversion(last_atr)
            df.drop(columns=[atr_col], errors='ignore', inplace=True)

        # Calculate Volume MA
        volume_ma_col = 'volume_ma'
        df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df['volume'].iloc[-1]

        if pd.notna(last_vol_ma): results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol): results["last_volume"] = safe_decimal_conversion(last_vol)

        if results["volume_ma"] is not None and results["volume_ma"] > CONFIG.position_qty_epsilon and results["last_volume"] is not None:
            try:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except Exception: # Handles potential division by zero if MA is epsilon
                 results["volume_ratio"] = None

        if volume_ma_col in df.columns: df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)

        # Log results
        atr_str = f"{results['atr']:.5f}" if results['atr'] else 'N/A'
        vol_ma_str = f"{results['volume_ma']:.2f}" if results['volume_ma'] else 'N/A'
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results['volume_ratio'] else 'N/A'
        logger.debug(f"Indicator Calc: ATR({atr_len})={atr_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={vol_ratio_str}")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results}
    return results

def calculate_stochrsi_momentum(df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, returns Decimals."""
    target_cols = ['stochrsi_k', 'stochrsi_d', 'momentum']
    min_len = max(rsi_len + stoch_len, mom_len) + 5 # Add buffer
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (StochRSI/Mom): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        # StochRSI
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col, d_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if k_col in stochrsi_df.columns: df['stochrsi_k'] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI K column not found"); df['stochrsi_k'] = pd.NA
        if d_col in stochrsi_df.columns: df['stochrsi_d'] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI D column not found"); df['stochrsi_d'] = pd.NA

        # Momentum
        mom_col = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns:
            df['momentum'] = df[mom_col].apply(safe_decimal_conversion)
            df.drop(columns=[mom_col], errors='ignore', inplace=True)
        else: logger.warning("Momentum column not found"); df['momentum'] = pd.NA

        k_val, d_val, mom_val = df['stochrsi_k'].iloc[-1], df['stochrsi_d'].iloc[-1], df['momentum'].iloc[-1]
        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            logger.debug(f"Indicator Calc (StochRSI/Mom): K={k_val:.2f}, D={d_val:.2f}, Mom={mom_val:.4f}")
        else:
            logger.debug("Indicator Calc (StochRSI/Mom): Resulted in NA for last candle.")

    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df

def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform, returns Decimals."""
    target_cols = ['ehlers_fisher', 'ehlers_signal']
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < length + 1:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersFisher): Invalid input (Len: {len(df) if df is not None else 0}, Need {length + 1}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col, signal_col = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fish_col in fisher_df.columns: df['ehlers_fisher'] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Fisher column not found"); df['ehlers_fisher'] = pd.NA
        if signal_col in fisher_df.columns: df['ehlers_signal'] = fisher_df[signal_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Signal column not found"); df['ehlers_signal'] = pd.NA

        fish_val, sig_val = df['ehlers_fisher'].iloc[-1], df['ehlers_signal'].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
             logger.debug(f"Indicator Calc (EhlersFisher({length},{signal})): Fisher={fish_val:.4f}, Signal={sig_val:.4f}")
        else:
             logger.debug("Indicator Calc (EhlersFisher): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df

def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers Super Smoother Moving Averages, returns Decimals."""
    target_cols = ['fast_ema', 'slow_ema']
    min_len = max(fast_len, slow_len) + 5 # Add buffer
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(f"{Fore.YELLOW}Indicator Calc (EhlersMA): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}")
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        # pandas_ta.supersmoother might not exist, use custom or alternative like Ehlers Filter if needed
        # Assuming ta.ema as a placeholder if supersmoother is unavailable or buggy
        # Replace with actual Ehlers filter implementation if required
        logger.warning(f"{Fore.YELLOW}Using EMA as placeholder for Ehlers Super Smoother. Replace with actual implementation if needed.{Style.RESET_ALL}")
        df['fast_ema'] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df['slow_ema'] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        fast_val, slow_val = df['fast_ema'].iloc[-1], df['slow_ema'].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            logger.debug(f"Indicator Calc (EhlersMA({fast_len},{slow_len})): Fast={fast_val:.4f}, Slow={slow_val:.4f}")
        else:
             logger.debug("Indicator Calc (EhlersMA): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA): Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df

def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> Dict[str, Optional[Decimal]]:
    """Fetches and analyzes L2 order book pressure and spread. Returns Decimals."""
    results: Dict[str, Optional[Decimal]] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")
    if not exchange.has.get('fetchL2OrderBook'):
        logger.warning(f"{Fore.YELLOW}fetchL2OrderBook not supported by {exchange.id}.{Style.RESET_ALL}")
        return results
    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: List[List[Union[float, str]]] = order_book.get('bids', [])
        asks: List[List[Union[float, str]]] = order_book.get('asks', [])

        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
            logger.debug(f"OB: Bid={best_bid:.4f}, Ask={best_ask:.4f}, Spread={results['spread']:.4f}")
        else:
            logger.debug(f"OB: Bid={best_bid or 'N/A'}, Ask={best_ask or 'N/A'} (Spread N/A)")

        # Sum volumes using Decimal
        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:depth] if len(ask) > 1)
        logger.debug(f"OB (Depth {depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")

        if ask_vol > CONFIG.position_qty_epsilon:
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                logger.debug(f"OB Ratio: {results['bid_ask_ratio']:.3f}")
            except Exception:
                logger.warning("Error calculating OB ratio.")
                results["bid_ask_ratio"] = None
        else:
            logger.debug("OB Ratio: N/A (Ask volume zero or negligible)")

    except (ccxt.NetworkError, ccxt.ExchangeError, IndexError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = {key: None for key in results} # Reset on error
    return results

# --- Data Fetching ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches and prepares OHLCV data, ensuring numeric types."""
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} ({interval})...")
        ohlcv: List[List[Union[int, float, str]]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}).{Style.RESET_ALL}")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Convert to numeric, coercing errors, check NaNs robustly
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV contains NaNs after conversion:\n{nan_counts[nan_counts > 0]}\nAttempting ffill...{Style.RESET_ALL}")
            df.ffill(inplace=True) # Forward fill first
            if df.isnull().values.any(): # Check again, maybe backfill needed?
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill, attempting bfill...{Style.RESET_ALL}")
                df.bfill(inplace=True)
                if df.isnull().values.any():
                    logger.error(f"{Fore.RED}Data Fetch: NaNs persist after ffill/bfill. Cannot proceed.{Style.RESET_ALL}")
                    return None

        logger.debug(f"Data Fetch: Processed {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Data Fetch: Error fetching OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return None

# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """Fetches current position details (Bybit V5 focus), returns Decimals."""
    default_pos: Dict[str, Any] = {'side': CONFIG.pos_none, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market_id = None
    market = None
    try:
        market = exchange.market(symbol)
        market_id = market['id']
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
        return default_pos

    try:
        if not exchange.has.get('fetchPositions'):
            logger.warning(f"{Fore.YELLOW}Position Check: fetchPositions not supported by {exchange.id}.{Style.RESET_ALL}")
            return default_pos

        # Bybit V5 uses 'category' parameter
        params = {'category': 'linear'} if market.get('linear') else ({'category': 'inverse'} if market.get('inverse') else {})
        logger.debug(f"Position Check: Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")

        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        # Bybit V5 might return multiple entries even for one-way mode sometimes, find the active one
        active_pos = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {})
            pos_market_id = pos_info.get('symbol')
            position_idx = pos_info.get('positionIdx', 0) # 0 for One-Way mode
            pos_side_v5 = pos_info.get('side', 'None') # 'Buy' for long, 'Sell' for short
            size_str = pos_info.get('size')

            # Filter for the correct symbol and One-Way mode active position
            if pos_market_id == market_id and position_idx == 0 and pos_side_v5 != 'None':
                size = safe_decimal_conversion(size_str)
                if abs(size) > CONFIG.position_qty_epsilon:
                    active_pos = pos # Found the active position
                    break # Assume only one active position in One-Way mode

        if active_pos:
            try:
                size = safe_decimal_conversion(active_pos.get('info', {}).get('size'))
                # Use 'avgPrice' from info for V5 entry price
                entry_price = safe_decimal_conversion(active_pos.get('info', {}).get('avgPrice'))
                # Determine side based on V5 'side' field
                side = CONFIG.pos_long if active_pos.get('info', {}).get('side') == 'Buy' else CONFIG.pos_short

                logger.info(f"{Fore.YELLOW}Position Check: Found ACTIVE {side} position: Qty={abs(size):.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}")
                return {'side': side, 'qty': abs(size), 'entry_price': entry_price}
            except Exception as parse_err:
                 logger.warning(f"{Fore.YELLOW}Position Check: Error parsing active position data: {parse_err}. Data: {active_pos}{Style.RESET_ALL}")
                 return default_pos
        else:
            logger.info(f"Position Check: No active One-Way position found for {market_id}.")
            return default_pos

    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Position Check: Error fetching positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
    return default_pos

def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol (Bybit V5 focus)."""
    logger.info(f"{Fore.CYAN}Leverage Setting: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol)
        if not market.get('contract'):
            logger.error(f"{Fore.RED}Leverage Setting: Cannot set for non-contract market: {symbol}.{Style.RESET_ALL}")
            return False
    except Exception as e:
         logger.error(f"{Fore.RED}Leverage Setting: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
         return False

    for attempt in range(CONFIG.retry_count):
        try:
            # Bybit V5 requires setting buy and sell leverage separately
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}Leverage Setting: Set to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True
        except ccxt.ExchangeError as e:
            # Check for common "already set" messages
            err_str = str(e).lower()
            if "leverage not modified" in err_str or "leverage is same as requested" in err_str:
                logger.info(f"{Fore.CYAN}Leverage Setting: Already set to {leverage}x for {symbol}.{Style.RESET_ALL}")
                return True
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Exchange error (Attempt {attempt+1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1: time.sleep(CONFIG.retry_delay_seconds)
            else: logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")
        except (ccxt.NetworkError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Network/Other error (Attempt {attempt+1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1: time.sleep(CONFIG.retry_delay_seconds)
            else: logger.error(f"{Fore.RED}Leverage Setting: Failed after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")
    return False

def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal") -> Optional[Dict[str, Any]]:
    """Closes the specified active position with re-validation, uses Decimal."""
    initial_side = position_to_close.get('side', CONFIG.pos_none)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0]
    logger.info(f"{Fore.YELLOW}Close Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # Re-validate the position just before closing
    live_position = get_current_position(exchange, symbol)
    if live_position['side'] == CONFIG.pos_none:
        logger.warning(f"{Fore.YELLOW}Close Position: Re-validation shows NO active position for {symbol}. Aborting.{Style.RESET_ALL}")
        if initial_side != CONFIG.pos_none: logger.warning(f"{Fore.YELLOW}Close Position: Discrepancy detected (was {initial_side}, now None).{Style.RESET_ALL}")
        return None

    live_amount_to_close = live_position['qty']
    live_position_side = live_position['side']
    side_to_execute_close = CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy

    try:
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_float = float(amount_str) # CCXT create order expects float
        if amount_float <= float(CONFIG.position_qty_epsilon):
            logger.error(f"{Fore.RED}Close Position: Closing amount after precision is negligible ({amount_str}). Aborting.{Style.RESET_ALL}")
            return None

        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Close Position: Attempting to CLOSE {live_position_side} ({reason}): "
                       f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}")
        params = {'reduceOnly': True}
        order = exchange.create_market_order(symbol=symbol, side=side_to_execute_close, amount=amount_float, params=params)

        # Parse order response safely using Decimal
        fill_price = safe_decimal_conversion(order.get('average'))
        filled_qty = safe_decimal_conversion(order.get('filled'))
        cost = safe_decimal_conversion(order.get('cost'))
        order_id_short = format_order_id(order.get('id'))

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Close Position: Order ({reason}) placed for {symbol}. "
                       f"Filled: {filled_qty:.8f}/{amount_str}, AvgFill: {fill_price:.4f}, Cost: {cost:.2f} USDT. ID:...{order_id_short}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:...{order_id_short}")
        return order

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}Close Position ({reason}): Failed for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        # Check for specific Bybit errors indicating already closed
        err_str = str(e).lower()
        if isinstance(e, ccxt.ExchangeError) and ("order would not reduce position size" in err_str or "position is zero" in err_str or "position size is zero" in err_str):
             logger.warning(f"{Fore.YELLOW}Close Position: Exchange indicates position already closed/closing. Assuming closed.{Style.RESET_ALL}")
             return None # Treat as success in this case
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}. Check logs.")
    return None

def calculate_position_size(equity: Decimal, risk_per_trade_pct: Decimal, entry_price: Decimal, stop_loss_price: Decimal,
                            leverage: int, symbol: str, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Calculates position size and estimated margin based on risk, using Decimal."""
    logger.debug(f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x")
    if not (entry_price > 0 and stop_loss_price > 0): logger.error(f"{Fore.RED}Risk Calc: Invalid entry/SL price.{Style.RESET_ALL}"); return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.position_qty_epsilon: logger.error(f"{Fore.RED}Risk Calc: Entry/SL prices too close ({price_diff:.8f}).{Style.RESET_ALL}"); return None, None
    if not 0 < risk_per_trade_pct < 1: logger.error(f"{Fore.RED}Risk Calc: Invalid risk %: {risk_per_trade_pct:.4%}.{Style.RESET_ALL}"); return None, None
    if equity <= 0: logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.4f}{Style.RESET_ALL}"); return None, None
    if leverage <= 0: logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}{Style.RESET_ALL}"); return None, None

    risk_amount_usdt = equity * risk_per_trade_pct
    # Assuming linear contract where 1 unit = 1 base currency (e.g., 1 BTC)
    # Risk per unit = price_diff
    quantity_raw = risk_amount_usdt / price_diff

    try:
        # Format according to market precision *then* convert back to Decimal
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = Decimal(quantity_precise_str)
    except Exception as e:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Failed precision formatting for quantity {quantity_raw:.8f}. Using raw. Error: {e}{Style.RESET_ALL}")
        quantity_precise = quantity_raw.quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP) # Fallback quantization

    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Risk Calc: Calculated quantity negligible ({quantity_precise:.8f}). RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}{Style.RESET_ALL}")
        return None, None

    pos_value_usdt = quantity_precise * entry_price
    required_margin = pos_value_usdt / Decimal(leverage)
    logger.debug(f"Risk Calc Result: Qty={quantity_precise:.8f}, EstValue={pos_value_usdt:.4f}, EstMargin={required_margin:.4f}")
    return quantity_precise, required_margin

def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
    """Waits for a specific order to be filled (status 'closed')."""
    start_time = time.time()
    logger.info(f"{Fore.CYAN}Waiting for order ...{format_order_id(order_id)} ({symbol}) fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")
    while time.time() - start_time < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get('status')
            logger.debug(f"Order ...{format_order_id(order_id)} status: {status}")
            if status == 'closed':
                logger.success(f"{Fore.GREEN}Order ...{format_order_id(order_id)} confirmed FILLED.{Style.RESET_ALL}")
                return order
            elif status in ['canceled', 'rejected', 'expired']:
                logger.error(f"{Fore.RED}Order ...{format_order_id(order_id)} failed with status '{status}'.{Style.RESET_ALL}")
                return None # Failed state
            # Continue polling if 'open' or 'partially_filled' or None
            time.sleep(0.5) # Check every 500ms
        except ccxt.OrderNotFound:
            # This might happen briefly after placing, keep trying
            logger.warning(f"{Fore.YELLOW}Order ...{format_order_id(order_id)} not found yet. Retrying...{Style.RESET_ALL}")
            time.sleep(1)
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.warning(f"{Fore.YELLOW}Error checking order ...{format_order_id(order_id)}: {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}")
            time.sleep(1) # Wait longer on error
    logger.error(f"{Fore.RED}Order ...{format_order_id(order_id)} did not fill within {timeout_seconds}s timeout.{Style.RESET_ALL}")
    return None # Timeout

def place_risked_market_order(exchange: ccxt.Exchange, symbol: str, side: str,
                            risk_percentage: Decimal, current_atr: Optional[Decimal], sl_atr_multiplier: Decimal,
                            leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal,
                            tsl_percent: Decimal, tsl_activation_offset_percent: Decimal) -> Optional[Dict[str, Any]]:
    """Places market entry, waits for fill, then places exchange-native fixed SL and TSL using Decimal."""
    market_base = symbol.split('/')[0]
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}Place Order: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}")
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid ATR ({current_atr}). Cannot place order.{Style.RESET_ALL}")
        return None

    entry_price_estimate: Optional[Decimal] = None
    initial_sl_price_estimate: Optional[Decimal] = None
    final_quantity: Optional[Decimal] = None
    market: Optional[Dict] = None

    try:
        # === 1. Get Balance, Market Info, Limits ===
        logger.debug("Fetching balance & market details...")
        balance = exchange.fetch_balance()
        market = exchange.market(symbol)
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        price_limits = limits.get('price', {})
        min_qty_str = amount_limits.get('min')
        max_qty_str = amount_limits.get('max')
        min_price_str = price_limits.get('min')
        min_qty = safe_decimal_conversion(min_qty_str) if min_qty_str else None
        max_qty = safe_decimal_conversion(max_qty_str) if max_qty_str else None
        min_price = safe_decimal_conversion(min_price_str) if min_price_str else None

        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        usdt_total = safe_decimal_conversion(usdt_balance.get('total'))
        usdt_free = safe_decimal_conversion(usdt_balance.get('free'))
        usdt_equity = usdt_total if usdt_total > 0 else usdt_free # Use total if available, else free

        if usdt_equity <= Decimal("0"): logger.error(f"{Fore.RED}Place Order ({side.upper()}): Zero/Invalid equity ({usdt_equity:.4f}).{Style.RESET_ALL}"); return None
        if usdt_free < Decimal("0"): logger.error(f"{Fore.RED}Place Order ({side.upper()}): Invalid free margin ({usdt_free:.4f}).{Style.RESET_ALL}"); return None
        logger.debug(f"Equity={usdt_equity:.4f}, Free={usdt_free:.4f} USDT")

        # === 2. Estimate Entry Price ===
        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask = ob_data.get("best_ask")
        best_bid = ob_data.get("best_bid")
        if side == CONFIG.side_buy and best_ask: entry_price_estimate = best_ask
        elif side == CONFIG.side_sell and best_bid: entry_price_estimate = best_bid
        else:
            try: entry_price_estimate = safe_decimal_conversion(exchange.fetch_ticker(symbol).get('last'))
            except Exception as e: logger.error(f"{Fore.RED}Failed to fetch ticker price: {e}{Style.RESET_ALL}"); return None
        if not entry_price_estimate or entry_price_estimate <= 0: logger.error(f"{Fore.RED}Invalid entry price estimate ({entry_price_estimate}).{Style.RESET_ALL}"); return None
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        # === 3. Calculate Initial Stop Loss Price (Estimate) ===
        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (entry_price_estimate - sl_distance) if side == CONFIG.side_buy else (entry_price_estimate + sl_distance)
        if min_price is not None and initial_sl_price_raw < min_price: initial_sl_price_raw = min_price
        if initial_sl_price_raw <= 0: logger.error(f"{Fore.RED}Invalid Initial SL price calc: {initial_sl_price_raw:.4f}{Style.RESET_ALL}"); return None
        initial_sl_price_estimate = safe_decimal_conversion(format_price(exchange, symbol, initial_sl_price_raw)) # Format estimate
        logger.info(f"Calculated Initial SL Price (Estimate) ~ {initial_sl_price_estimate:.4f} (Dist: {sl_distance:.4f})")

        # === 4. Calculate Position Size ===
        calc_qty, req_margin = calculate_position_size(usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate, leverage, symbol, exchange)
        if calc_qty is None or req_margin is None: logger.error(f"{Fore.RED}Failed risk calculation.{Style.RESET_ALL}"); return None
        final_quantity = calc_qty

        # === 5. Apply Max Order Cap ===
        pos_value = final_quantity * entry_price_estimate
        if pos_value > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}Order value {pos_value:.4f} > Cap {max_order_cap_usdt:.4f}. Capping qty.{Style.RESET_ALL}")
            final_quantity = max_order_cap_usdt / entry_price_estimate
            # Format capped quantity
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))
            req_margin = (max_order_cap_usdt / Decimal(leverage)) # Recalculate margin based on cap

        # === 6. Check Limits & Margin ===
        if final_quantity <= CONFIG.position_qty_epsilon: logger.error(f"{Fore.RED}Final Qty negligible: {final_quantity:.8f}{Style.RESET_ALL}"); return None
        if min_qty is not None and final_quantity < min_qty: logger.error(f"{Fore.RED}Final Qty {final_quantity:.8f} < Min {min_qty}{Style.RESET_ALL}"); return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"{Fore.YELLOW}Final Qty {final_quantity:.8f} > Max {max_qty}. Capping.{Style.RESET_ALL}")
            final_quantity = max_qty
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity)) # Re-format capped amount

        final_req_margin = (final_quantity * entry_price_estimate) / Decimal(leverage) # Final margin estimate
        req_margin_buffered = final_req_margin * margin_check_buffer

        if usdt_free < req_margin_buffered:
            logger.error(f"{Fore.RED}Insufficient FREE margin. Need ~{req_margin_buffered:.4f}, Have {usdt_free:.4f}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin")
            return None
        logger.info(f"{Fore.GREEN}Final Order: Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={final_req_margin:.4f}. Margin check OK.{Style.RESET_ALL}")

        # === 7. Place Entry Market Order ===
        entry_order: Optional[Dict[str, Any]] = None
        order_id: Optional[str] = None
        try:
            qty_float = float(final_quantity) # CCXT expects float
            logger.warning(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
            entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params={'reduce_only': False})
            order_id = entry_order.get('id')
            if not order_id:
                logger.error(f"{Fore.RED}Entry order placed but no ID returned! Response: {entry_order}{Style.RESET_ALL}")
                return None
            logger.success(f"{Fore.GREEN}Market Entry Order submitted. ID: ...{format_order_id(order_id)}. Waiting for fill...{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}")
            return None

        # === 8. Wait for Entry Fill ===
        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry:
            logger.error(f"{Fore.RED}Entry order ...{format_order_id(order_id)} did not fill/failed.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry ...{format_order_id(order_id)} fill timeout/fail.")
            # Try to cancel the potentially stuck order
            try: exchange.cancel_order(order_id, symbol) except Exception: pass
            return None

        # === 9. Extract Fill Details (Crucial: Use Actual Fill) ===
        avg_fill_price = safe_decimal_conversion(filled_entry.get('average'))
        filled_qty = safe_decimal_conversion(filled_entry.get('filled'))
        cost = safe_decimal_conversion(filled_entry.get('cost'))

        if avg_fill_price <= 0 or filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Invalid fill details for ...{format_order_id(order_id)}: Price={avg_fill_price}, Qty={filled_qty}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid fill details ...{format_order_id(order_id)}.")
            return filled_entry # Return problematic order

        logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {filled_qty:.8f} @ {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}")

        # === 10. Calculate ACTUAL Stop Loss Price based on Fill ===
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.side_buy else (avg_fill_price + sl_distance)
        if min_price is not None and actual_sl_price_raw < min_price: actual_sl_price_raw = min_price
        if actual_sl_price_raw <= 0:
            logger.error(f"{Fore.RED}Invalid ACTUAL SL price calc based on fill: {actual_sl_price_raw:.4f}. Cannot place SL!{Style.RESET_ALL}")
            # CRITICAL: Position is open without SL. Attempt emergency close.
            send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price! Attempting emergency close.")
            close_position(exchange, symbol, {'side': side, 'qty': filled_qty}, reason="Invalid SL Calc")
            return filled_entry # Return filled entry, but indicate failure state
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        actual_sl_price_float = float(actual_sl_price_str) # For CCXT param

        # === 11. Place Initial Fixed Stop Loss ===
        sl_order_id = "N/A"
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)

            logger.info(f"{Fore.CYAN}Placing Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, StopPx: {actual_sl_price_str}{Style.RESET_ALL}")
            # Bybit V5 stop order params: stopPrice (trigger), reduceOnly
            sl_params = {'stopPrice': actual_sl_price_float, 'reduceOnly': True}
            sl_order = exchange.create_order(symbol, 'stopMarket', sl_side, sl_qty_float, params=sl_params)
            sl_order_id = format_order_id(sl_order.get('id'))
            logger.success(f"{Fore.GREEN}Initial Fixed SL order placed. ID: ...{sl_order_id}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Initial Fixed SL order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement: {type(e).__name__}")
            # Don't necessarily close here, TSL might still work, or user might want manual intervention

        # === 12. Place Trailing Stop Loss ===
        tsl_order_id = "N/A"
        tsl_act_price_str = "N/A"
        try:
            # Calculate TSL activation price based on actual fill
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.side_buy else (avg_fill_price - act_offset)
            if min_price is not None and act_price_raw < min_price: act_price_raw = min_price
            if act_price_raw <= 0: raise ValueError(f"Invalid TSL activation price {act_price_raw:.4f}")

            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            # Bybit V5 uses 'trailingStop' for percentage distance (e.g., "0.5" for 0.5%)
            tsl_trail_value_str = str(tsl_percent * Decimal("100"))
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)
            tsl_qty_float = float(tsl_qty_str)

            logger.info(f"{Fore.CYAN}Placing Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")
            # Bybit V5 TSL params: trailingStop (percent string), activePrice (activation trigger), reduceOnly
            tsl_params = {
                'trailingStop': tsl_trail_value_str,
                'activePrice': tsl_act_price_float,
                'reduceOnly': True,
            }
            # Use 'stopMarket' type with TSL params for Bybit V5 via CCXT
            tsl_order = exchange.create_order(symbol, 'stopMarket', tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id = format_order_id(tsl_order.get('id'))
            logger.success(f"{Fore.GREEN}Trailing SL order placed. ID: ...{tsl_order_id}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")

            # Final comprehensive SMS
            sms_msg = (f"[{market_base}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                       f"Init SL ~{actual_sl_price_str}. TSL {tsl_percent:.2%} act@{tsl_act_price_str}. "
                       f"IDs E:...{format_order_id(order_id)}, SL:...{sl_order_id}, TSL:...{tsl_order_id}")
            send_sms_alert(sms_msg)

        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Trailing SL order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}")
            # If TSL fails but initial SL was placed, the position is still protected initially.

        return filled_entry # Return filled entry order details regardless of SL/TSL placement success

    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}{Style.BRIGHT}Place Order ({side.upper()}): Overall process failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Overall process failed: {type(e).__name__}")
    return None

def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Attempts to cancel all open orders for the specified symbol."""
    logger.info(f"{Fore.CYAN}Order Cancel: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get('fetchOpenOrders'):
            logger.warning(f"{Fore.YELLOW}Order Cancel: fetchOpenOrders not supported.{Style.RESET_ALL}")
            return
        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders:
            logger.info(f"{Fore.CYAN}Order Cancel: No open orders found for {symbol}.{Style.RESET_ALL}")
            return

        logger.warning(f"{Fore.YELLOW}Order Cancel: Found {len(open_orders)} open orders for {symbol}. Cancelling...{Style.RESET_ALL}")
        cancelled_count, failed_count = 0, 0
        for order in open_orders:
            order_id = order.get('id')
            order_info = f"...{format_order_id(order_id)} ({order.get('type')} {order.get('side')})"
            if order_id:
                try:
                    exchange.cancel_order(order_id, symbol)
                    logger.info(f"{Fore.CYAN}Order Cancel: Success for {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1
                    time.sleep(0.1) # Small delay between cancels
                except ccxt.OrderNotFound:
                    logger.warning(f"{Fore.YELLOW}Order Cancel: Not found (already closed/cancelled?): {order_info}{Style.RESET_ALL}")
                    cancelled_count += 1 # Treat as cancelled if not found
                except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
                    logger.error(f"{Fore.RED}Order Cancel: FAILED for {order_info}: {type(e).__name__} - {e}{Style.RESET_ALL}")
                    failed_count += 1
        logger.info(f"{Fore.CYAN}Order Cancel: Finished. Cancelled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
        if failed_count > 0: send_sms_alert(f"[{symbol.split('/')[0]}] WARNING: Failed to cancel {failed_count} orders during {reason}.")
    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Order Cancel: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")

# --- Strategy Signal Generation ---
def generate_signals(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """Generates entry/exit signals based on the selected strategy."""
    signals = {'enter_long': False, 'enter_short': False, 'exit_long': False, 'exit_short': False, 'exit_reason': "Strategy Exit"}
    if len(df) < 2: return signals # Need previous candle for comparisons/crosses

    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        # --- Dual Supertrend Logic ---
        if strategy_name == "DUAL_SUPERTREND":
            if pd.notna(last['st_long']) and last['st_long'] and pd.notna(last['confirm_trend']) and last['confirm_trend']: signals['enter_long'] = True
            if pd.notna(last['st_short']) and last['st_short'] and pd.notna(last['confirm_trend']) and not last['confirm_trend']: signals['enter_short'] = True
            if pd.notna(last['st_short']) and last['st_short']: signals['exit_long'] = True; signals['exit_reason'] = "Primary ST Short Flip"
            if pd.notna(last['st_long']) and last['st_long']: signals['exit_short'] = True; signals['exit_reason'] = "Primary ST Long Flip"

        # --- StochRSI + Momentum Logic ---
        elif strategy_name == "STOCHRSI_MOMENTUM":
            k_now, d_now, mom_now = last['stochrsi_k'], last['stochrsi_d'], last['momentum']
            k_prev, d_prev = prev['stochrsi_k'], prev['stochrsi_d']
            if pd.isna(k_now) or pd.isna(d_now) or pd.isna(mom_now) or pd.isna(k_prev) or pd.isna(d_prev): return signals # Not enough data

            if k_prev <= d_prev and k_now > d_now and k_now < CONFIG.stochrsi_oversold and mom_now > CONFIG.position_qty_epsilon: signals['enter_long'] = True
            if k_prev >= d_prev and k_now < d_now and k_now > CONFIG.stochrsi_overbought and mom_now < -CONFIG.position_qty_epsilon: signals['enter_short'] = True
            if k_prev >= d_prev and k_now < d_now: signals['exit_long'] = True; signals['exit_reason'] = "StochRSI K below D"
            if k_prev <= d_prev and k_now > d_now: signals['exit_short'] = True; signals['exit_reason'] = "StochRSI K above D"

        # --- Ehlers Fisher Logic ---
        elif strategy_name == "EHLERS_FISHER":
            fish_now, sig_now = last['ehlers_fisher'], last['ehlers_signal']
            fish_prev, sig_prev = prev['ehlers_fisher'], prev['ehlers_signal']
            if pd.isna(fish_now) or pd.isna(sig_now) or pd.isna(fish_prev) or pd.isna(sig_prev): return signals

            if fish_prev <= sig_prev and fish_now > sig_now: signals['enter_long'] = True
            if fish_prev >= sig_prev and fish_now < sig_now: signals['enter_short'] = True
            if fish_prev >= sig_prev and fish_now < sig_now: signals['exit_long'] = True; signals['exit_reason'] = "Ehlers Fisher Short Cross"
            if fish_prev <= sig_prev and fish_now > sig_now: signals['exit_short'] = True; signals['exit_reason'] = "Ehlers Fisher Long Cross"

        # --- Ehlers MA Cross Logic ---
        elif strategy_name == "EHLERS_MA_CROSS":
            fast_ma_now, slow_ma_now = last['fast_ema'], last['slow_ema']
            fast_ma_prev, slow_ma_prev = prev['fast_ema'], prev['slow_ema']
            if pd.isna(fast_ma_now) or pd.isna(slow_ma_now) or pd.isna(fast_ma_prev) or pd.isna(slow_ma_prev): return signals

            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now: signals['enter_long'] = True
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now: signals['enter_short'] = True
            if fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now: signals['exit_long'] = True; signals['exit_reason'] = "Ehlers MA Short Cross"
            if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now: signals['exit_short'] = True; signals['exit_reason'] = "Ehlers MA Long Cross"

    except KeyError as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Missing expected column in DataFrame: {e}{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Unexpected exception: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())

    return signals

# --- Trading Logic ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle based on selected strategy."""
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Check Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    # Determine required rows based on the longest lookback needed by any indicator used
    required_rows = max(
        CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length, CONFIG.momentum_length, # Estimate
        CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length,
        CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
        CONFIG.atr_calculation_period, CONFIG.volume_ma_period
    ) + 10 # Add buffer

    if df is None or len(df) < required_rows:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0}, need ~{required_rows}). Skipping.{Style.RESET_ALL}")
        return

    action_taken_this_cycle: bool = False
    try:
        # === 1. Calculate ALL Indicators ===
        # It's often simpler to calculate all potential indicators needed by any strategy
        # and let the signal generation function pick the ones it needs.
        logger.debug("Calculating indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")

        # === 2. Validate Base Requirements ===
        last = df.iloc[-1]
        current_price = safe_decimal_conversion(last['close'])
        if pd.isna(current_price) or current_price <= 0:
            logger.warning(f"{Fore.YELLOW}Last candle close price is invalid ({current_price}). Skipping.{Style.RESET_ALL}")
            return
        can_place_order = current_atr is not None and current_atr > Decimal("0")
        if not can_place_order:
            logger.warning(f"{Fore.YELLOW}Invalid ATR ({current_atr}). Cannot calculate SL or place new orders.{Style.RESET_ALL}")

        # === 3. Get Position & Analyze OB ===
        position = get_current_position(exchange, symbol)
        position_side = position['side']
        ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit) if CONFIG.fetch_order_book_per_cycle else None

        # === 4. Log State ===
        vol_ratio = vol_atr_data.get("volume_ratio")
        vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
        spread = ob_data.get("spread") if ob_data else None

        logger.info(f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): {current_atr:.5f}" if current_atr else f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): N/A")
        logger.info(f"State | Volume: Ratio={vol_ratio:.2f if vol_ratio else 'N/A'}, Spike={vol_spike} (Req={CONFIG.require_volume_spike_for_entry})")
        # Log specific strategy indicators
        # ... (Add logging for relevant indicators based on CONFIG.strategy_name if needed, or rely on debug logs from calc functions) ...
        logger.info(f"State | OrderBook: Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'}, Spread={spread:.4f if spread else 'N/A'}")
        logger.info(f"State | Position: Side={position_side}, Qty={position['qty']:.8f}, Entry={position['entry_price']:.4f}")

        # === 5. Generate Strategy Signals ===
        strategy_signals = generate_signals(df, CONFIG.strategy_name)
        logger.debug(f"Strategy Signals ({CONFIG.strategy_name}): {strategy_signals}")

        # === 6. Execute Exit Actions ===
        should_exit_long = position_side == CONFIG.pos_long and strategy_signals['exit_long']
        should_exit_short = position_side == CONFIG.pos_short and strategy_signals['exit_short']

        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals['exit_reason']
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}*** TRADE EXIT SIGNAL: Closing {position_side} due to {exit_reason} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, f"SL/TSL before {exit_reason} Exit")
            close_result = close_position(exchange, symbol, position, reason=exit_reason)
            if close_result: action_taken_this_cycle = True
            # Add delay after closing before allowing new entry
            if action_taken_this_cycle:
                logger.info(f"Pausing for {CONFIG.post_close_delay_seconds}s after closing position...")
                time.sleep(CONFIG.post_close_delay_seconds)
            return # Exit cycle after attempting close

        # === 7. Check & Execute Entry Actions (Only if Flat & Can Place Order) ===
        if position_side != CONFIG.pos_none:
             logger.info(f"Holding {position_side} position. Waiting for SL/TSL or Strategy Exit.")
             return
        if not can_place_order:
             logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter due to invalid ATR for SL calculation.{Style.RESET_ALL}")
             return

        logger.debug("Checking entry signals...")
        # --- Define Confirmation Conditions ---
        potential_entry = strategy_signals['enter_long'] or strategy_signals['enter_short']
        if not CONFIG.fetch_order_book_per_cycle and potential_entry and ob_data is None:
            logger.debug("Potential entry signal, fetching OB for confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None # Update ratio

        # Check OB confirmation only if required
        ob_check_required = potential_entry # Always check OB if entry signal exists? Or make configurable? Let's assume yes for now.
        ob_available = ob_data is not None and bid_ask_ratio is not None
        passes_long_ob = not ob_check_required or (ob_available and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
        passes_short_ob = not ob_check_required or (ob_available and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
        ob_log = f"OB OK (L:{passes_long_ob},S:{passes_short_ob}, Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'}, Req={ob_check_required})"

        # Check Volume confirmation only if required
        vol_check_required = CONFIG.require_volume_spike_for_entry
        passes_volume = not vol_check_required or (vol_spike)
        vol_log = f"Vol OK (Pass:{passes_volume}, Spike={vol_spike}, Req={vol_check_required})"

        # --- Combine Strategy Signal with Confirmations ---
        enter_long = strategy_signals['enter_long'] and passes_long_ob and passes_volume
        enter_short = strategy_signals['enter_short'] and passes_short_ob and passes_volume
        logger.debug(f"Final Entry Check (Long): Strategy={strategy_signals['enter_long']}, {ob_log}, {vol_log} => Enter={enter_long}")
        logger.debug(f"Final Entry Check (Short): Strategy={strategy_signals['enter_short']}, {ob_log}, {vol_log} => Enter={enter_short}")

        # --- Execute ---
        if enter_long:
            logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, "Before Long Entry") # Cancel previous SL/TSL just in case
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_buy, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        elif enter_short:
            logger.success(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, "Before Short Entry")
            place_result = place_risked_market_order(
                exchange, symbol, CONFIG.side_sell, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
            if place_result: action_taken_this_cycle = True

        else:
             if not action_taken_this_cycle: logger.info("No confirmed entry signal. Holding cash.")

    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL ERROR in trade_logic: {type(e).__name__}")
    finally:
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Check End: {symbol} =========={Style.RESET_ALL}\n")

# --- Graceful Shutdown ---
def graceful_shutdown(exchange: Optional[ccxt.Exchange], symbol: Optional[str]) -> None:
    """Attempts to close position and cancel orders before exiting."""
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit...{Style.RESET_ALL}")
    market_base = symbol.split('/')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown requested. Attempting cleanup...")
    if not exchange or not symbol:
        logger.warning(f"{Fore.YELLOW}Shutdown: Exchange/Symbol not available.{Style.RESET_ALL}")
        return

    try:
        # 1. Cancel All Open Orders
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(1) # Allow cancellations to process

        # 2. Check and Close Position
        position = get_current_position(exchange, symbol)
        if position['side'] != CONFIG.pos_none:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} position found (Qty: {position['qty']:.8f}). Closing...{Style.RESET_ALL}")
            close_result = close_position(exchange, symbol, position, reason="Shutdown")
            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.post_close_delay_seconds * 2}s for confirmation...{Style.RESET_ALL}")
                time.sleep(CONFIG.post_close_delay_seconds * 2)
                # Final check
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.pos_none:
                    logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else:
                    logger.error(f"{Back.RED}{Fore.WHITE}Shutdown: FAILED TO CONFIRM closure. Final state: {final_pos['side']} Qty={final_pos['qty']:.8f}{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] ERROR: Failed confirm closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}. MANUAL CHECK!")
            else:
                logger.error(f"{Back.RED}{Fore.WHITE}Shutdown: Failed to place close order. MANUAL INTERVENTION NEEDED.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] ERROR: Failed PLACE close order on shutdown. MANUAL CHECK!")
        else:
            logger.info(f"{Fore.GREEN}Shutdown: No active position found.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] No active position found on shutdown.")
    except Exception as e:
        logger.error(f"{Fore.RED}Shutdown: Error during cleanup: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] Error during shutdown sequence: {type(e).__name__}")
    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")

# --- Main Execution ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    start_time = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.0 Initializing ({start_time}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment: {CONFIG.strategy_name} ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Warding Rune: Initial ATR + Exchange Trailing Stop ---{Style.RESET_ALL}")
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK INVOLVED !!! ---{Style.RESET_ALL}")

    exchange: Optional[ccxt.Exchange] = None
    symbol: Optional[str] = None
    run_bot: bool = True
    cycle_count: int = 0

    try:
        # Initialize Exchange
        exchange = initialize_exchange()
        if not exchange: return

        # Setup Symbol and Leverage
        try:
            # Allow user input or use default from config
            sym_input = input(f"{Fore.YELLOW}Enter symbol {Style.DIM}(Default [{CONFIG.symbol}]){Style.NORMAL}: {Style.RESET_ALL}").strip()
            symbol_to_use = sym_input or CONFIG.symbol
            market = exchange.market(symbol_to_use)
            symbol = market['symbol'] # Use the unified symbol from CCXT
            if not market.get('contract'): raise ValueError("Not a contract/futures market")
            logger.info(f"{Fore.GREEN}Using Symbol: {symbol} (Type: {market.get('type')}){Style.RESET_ALL}")
            if not set_leverage(exchange, symbol, CONFIG.leverage): raise RuntimeError("Leverage setup failed")
        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e:
            logger.critical(f"Symbol/Leverage setup failed: {e}")
            send_sms_alert(f"[ScalpBot] CRITICAL: Symbol/Leverage setup FAILED ({e}). Exiting.")
            return
        except Exception as e:
            logger.critical(f"Unexpected error during setup: {e}")
            send_sms_alert("[ScalpBot] CRITICAL: Unexpected setup error. Exiting.")
            return

        # Log Config Summary
        logger.info(f"{Fore.MAGENTA}--- Configuration Summary ---{Style.RESET_ALL}")
        logger.info(f"{Fore.WHITE}Symbol: {symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
        logger.info(f"{Fore.CYAN}Strategy: {CONFIG.strategy_name}")
        # Log relevant strategy params
        if CONFIG.strategy_name == "DUAL_SUPERTREND": logger.info(f"  Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
        elif CONFIG.strategy_name == "STOCHRSI_MOMENTUM": logger.info(f"  Params: StochRSI={CONFIG.stochrsi_rsi_length}/{CONFIG.stochrsi_stoch_length}/{CONFIG.stochrsi_k_period}/{CONFIG.stochrsi_d_period} (OB={CONFIG.stochrsi_overbought},OS={CONFIG.stochrsi_oversold}), Mom={CONFIG.momentum_length}")
        elif CONFIG.strategy_name == "EHLERS_FISHER": logger.info(f"  Params: Fisher={CONFIG.ehlers_fisher_length}, Signal={CONFIG.ehlers_fisher_signal_length}")
        elif CONFIG.strategy_name == "EHLERS_MA_CROSS": logger.info(f"  Params: FastMA={CONFIG.ehlers_fast_period}, SlowMA={CONFIG.ehlers_slow_period}")
        logger.info(f"{Fore.GREEN}Risk: {CONFIG.risk_per_trade_percentage:.3%}/trade, MaxPosValue: {CONFIG.max_order_usdt_amount:.4f} USDT")
        logger.info(f"{Fore.GREEN}Initial SL: {CONFIG.atr_stop_loss_multiplier} * ATR({CONFIG.atr_calculation_period})")
        logger.info(f"{Fore.GREEN}Trailing SL: {CONFIG.trailing_stop_percentage:.2%}, Activation Offset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f"{Fore.YELLOW}Vol Confirm: {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
        logger.info(f"{Fore.YELLOW}OB Confirm: {CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
        logger.info(f"{Fore.WHITE}Sleep: {CONFIG.sleep_seconds}s, Margin Buffer: {CONFIG.required_margin_buffer:.1%}, SMS: {CONFIG.enable_sms_alerts}")
        logger.info(f"{Fore.CYAN}Logging Level: {logging.getLevelName(logger.level)}")
        logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")
        market_base = symbol.split('/')[0]
        send_sms_alert(f"[{market_base}] Bot configured ({CONFIG.strategy_name}). SL: ATR+TSL. Starting loop.")

        # Main Trading Loop
        while run_bot:
            cycle_start_time = time.monotonic()
            cycle_count += 1
            logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Start ---{Style.RESET_ALL}")
            try:
                # Determine required data length based on longest possible indicator lookback
                data_limit = max(100, CONFIG.st_atr_length*2, CONFIG.confirm_st_atr_length*2,
                                 CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 5, CONFIG.momentum_length*2,
                                 CONFIG.ehlers_fisher_length*2, CONFIG.ehlers_fisher_signal_length*2,
                                 CONFIG.ehlers_fast_period*2, CONFIG.ehlers_slow_period*2,
                                 CONFIG.atr_calculation_period*2, CONFIG.volume_ma_period*2) + CONFIG.api_fetch_limit_buffer

                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)

                if df is not None and not df.empty:
                    trade_logic(exchange, symbol, df.copy()) # Pass copy to avoid modifying original in logic
                else:
                    logger.warning(f"{Fore.YELLOW}No valid market data for {symbol}. Skipping cycle.{Style.RESET_ALL}")

            # --- Robust Error Handling ---
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit Exceeded: {e}. Sleeping longer...{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds * 5); send_sms_alert(f"[{market_base}] WARNING: Rate limit hit!")
            except ccxt.NetworkError as e:
                logger.warning(f"{Fore.YELLOW}Network error: {e}. Retrying next cycle.{Style.RESET_ALL}")
                time.sleep(CONFIG.sleep_seconds) # Standard sleep on recoverable network errors
            except ccxt.ExchangeNotAvailable as e:
                logger.error(f"{Back.RED}{Fore.WHITE}Exchange unavailable: {e}. Sleeping much longer...{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds * 10); send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable!")
            except ccxt.AuthenticationError as e:
                logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Authentication Error: {e}. Stopping NOW.{Style.RESET_ALL}"); run_bot = False; send_sms_alert(f"[{market_base}] CRITICAL: Authentication Error! Stopping NOW.")
            except ccxt.ExchangeError as e: # Catch broader exchange errors
                logger.error(f"{Fore.RED}Unhandled Exchange error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}")
                time.sleep(CONFIG.sleep_seconds) # Sleep before retrying after general exchange error
            except Exception as e:
                logger.exception(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}!!! UNEXPECTED CRITICAL ERROR: {e} !!!{Style.RESET_ALL}"); run_bot = False; send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping NOW.")

            # --- Loop Delay ---
            if run_bot:
                elapsed = time.monotonic() - cycle_start_time
                sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle {cycle_count} time: {elapsed:.2f}s. Sleeping: {sleep_dur:.2f}s.")
                if sleep_dur > 0: time.sleep(sleep_dur)

    except KeyboardInterrupt:
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Arcane energies withdrawing...{Style.RESET_ALL}")
        run_bot = False # Ensure loop terminates
    finally:
        # --- Graceful Shutdown ---
        graceful_shutdown(exchange, symbol)
        market_base_final = symbol.split('/')[0] if symbol else "Bot"
        send_sms_alert(f"[{market_base_final}] Bot process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")

if __name__ == "__main__":
    main()