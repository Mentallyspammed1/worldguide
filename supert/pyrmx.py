#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.1.0 (Integrated Functions)
# Conjures high-frequency trades on Bybit Futures with enhanced precision and adaptable strategies.

"""
High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.1.0 (Unified: Selectable Strategies + Precision + Native SL/TP/TSL + Integrated Functions)

Features:
- Multiple strategies selectable via config: "DUAL_EHLERS_VOLUMETRIC", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS".
- Enhanced Precision: Uses Decimal for critical financial calculations.
- Exchange-native Trailing Stop Loss (TSL) placed immediately after entry.
- Exchange-native fixed Stop Loss (SL) placed immediately after entry.
- Exchange-native Take Profit (TP) limit order placed immediately after entry.
- ATR for volatility measurement and initial Stop-Loss calculation.
- Optional Volume spike and Order Book pressure confirmation.
- Risk-based position sizing with margin checks (including taker fee estimate).
- Termux SMS alerts for critical events and trade actions.
- Robust error handling and logging with Neon color support.
- Graceful shutdown on KeyboardInterrupt with position/order closing attempt.
- Stricter position detection logic (Bybit V5 API).
- Integrated previously missing helper functions.

Disclaimer:
- **EXTREME RISK**: Educational purposes ONLY. High-risk. Use at own absolute risk.
- **EXCHANGE-NATIVE SL/TP/TSL DEPENDENCE**: Relies on exchange-native orders. Subject to exchange performance, slippage, API reliability.
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
import subprocess
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Any

# Third-party Libraries
try:
    import ccxt
    import numpy as np
    import pandas as pd
    import pandas_ta as ta # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = e.name
    print(f"Error: Missing required library '{missing_pkg}'. Please run: pip install ccxt pandas numpy pandas_ta PyYAML python-dotenv colorama")
    sys.exit(1)

# --- Initializations ---
colorama_init(autoreset=True)
load_dotenv()
getcontext().prec = 18 # Set Decimal precision (adjust as needed)

# --- Logger Setup ---
LOGGING_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
)
logger = logging.getLogger(__name__)

# Custom SUCCESS level and Neon Color Formatting
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log level."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

logging.Logger.success = log_success # type: ignore[attr-defined]

if sys.stdout.isatty():
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")

# --- Configuration Class ---
class Config:
    """Loads and validates configuration parameters from environment variables."""
    def __init__(self) -> None:
        """Initializes the configuration by loading environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED)

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW)
        self.interval: str = self._get_env("INTERVAL", "1m", color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", 10, cast_type=int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW)

        # --- Strategy Selection ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_EHLERS_VOLUMETRIC", color=Fore.CYAN).upper()
        self.valid_strategies: list[str] = ["DUAL_EHLERS_VOLUMETRIC", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EHLERS_MA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}")
            raise ValueError(f"Invalid STRATEGY_NAME '{self.strategy_name}'")
        logger.info(f"Selected Strategy: {Fore.CYAN}{self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # 0.5% risk
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN)
        self.take_profit_multiplier: Decimal = self._get_env("TAKE_PROFIT_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.GREEN) # Default to 1:2 R:R
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN) # 5% buffer
        self.min_sl_distance_percent: Decimal = self._get_env("MIN_SL_DISTANCE_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN) # Min 0.1% SL distance from entry

        # --- Trailing Stop Loss (Exchange Native) ---
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN) # 0.5% trail distance
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN) # 0.1% offset from entry

        # --- Ehlers Volumetric Trend Parameters --- (DUAL_EHLERS_VOLUMETRIC strategy)
        self.evt_length: int = self._get_env("EVT_LENGTH", 7, cast_type=int, color=Fore.CYAN)
        self.evt_multiplier: Decimal = self._get_env("EVT_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN)
        self.confirm_evt_length: int = self._get_env("CONFIRM_EVT_LENGTH", 5, cast_type=int, color=Fore.CYAN)
        self.confirm_evt_multiplier: Decimal = self._get_env("CONFIRM_EVT_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN)

        # --- StochRSI + Momentum Parameters --- (STOCHRSI_MOMENTUM strategy)
        self.stochrsi_rsi_length: int = self._get_env("STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_stoch_length: int = self._get_env("STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN)
        self.stochrsi_k_period: int = self._get_env("STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_d_period: int = self._get_env("STOCHRSI_D_PERIOD", 3, cast_type=int, color=Fore.CYAN)
        self.stochrsi_overbought: Decimal = self._get_env("STOCHRSI_OVERBOUGHT", "80.0", cast_type=Decimal, color=Fore.CYAN)
        self.stochrsi_oversold: Decimal = self._get_env("STOCHRSI_OVERSOLD", "20.0", cast_type=Decimal, color=Fore.CYAN)
        self.momentum_length: int = self._get_env("MOMENTUM_LENGTH", 5, cast_type=int, color=Fore.CYAN)

        # --- Ehlers Fisher Transform Parameters --- (EHLERS_FISHER strategy)
        self.ehlers_fisher_length: int = self._get_env("EHLERS_FISHER_LENGTH", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_fisher_signal_length: int = self._get_env("EHLERS_FISHER_SIGNAL_LENGTH", 1, cast_type=int, color=Fore.CYAN) # Default to 1 (often just used as Fisher value)

        # --- Ehlers MA Cross Parameters --- (EHLERS_MA_CROSS strategy)
        self.ehlers_fast_period: int = self._get_env("EHLERS_FAST_PERIOD", 10, cast_type=int, color=Fore.CYAN)
        self.ehlers_slow_period: int = self._get_env("EHLERS_SLOW_PERIOD", 30, cast_type=int, color=Fore.CYAN)

        # --- Volume Analysis ---
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", "1.5", cast_type=Decimal, color=Fore.YELLOW) # Vol > 1.5x MA
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "false", cast_type=bool, color=Fore.YELLOW)

        # --- Order Book Analysis ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 10, cast_type=int, color=Fore.YELLOW)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", "1.2", cast_type=Decimal, color=Fore.YELLOW) # Bid/Ask ratio >= 1.2 for Long confirm
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", "0.8", cast_type=Decimal, color=Fore.YELLOW) # Bid/Ask ratio <= 0.8 for Short confirm
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", "false", cast_type=bool, color=Fore.YELLOW)

        # --- ATR Calculation (for Initial SL) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int, color=Fore.GREEN)

        # --- Termux SMS Alerts ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "false", cast_type=bool, color=Fore.MAGENTA)
        self.sms_recipient_number: str | None = self._get_env("SMS_RECIPIENT_NUMBER", None, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int, color=Fore.MAGENTA)

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = 10000
        self.order_book_fetch_limit: int = max(25, self.order_book_depth) # Ensure fetch limit is sufficient for Bybit L2
        self.shallow_ob_fetch_depth: int = 5 # For quick price estimates
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int, color=Fore.YELLOW)

        # --- Internal Constants --- (Moved from ps.py global scope to Config)
        self.SIDE_BUY: str = "buy"
        self.SIDE_SELL: str = "sell"
        self.POS_LONG: str = "Long"
        self.POS_SHORT: str = "Short"
        self.POS_NONE: str = "None"
        self.USDT_SYMBOL: str = "USDT" # Assume USDT quote, adjust if needed
        self.RETRY_COUNT: int = 3
        self.RETRY_DELAY_SECONDS: int = 2
        self.API_FETCH_LIMIT_BUFFER: int = 10 # Extra candles to fetch beyond indicator needs
        self.POSITION_QTY_EPSILON: Decimal = Decimal("1e-9") # Small value to treat quantities near zero
        self.POST_CLOSE_DELAY_SECONDS: int = 3 # Wait time after closing position before next action
        self.TAKER_FEE_RATE: Decimal = Decimal("0.0006") # Default Bybit taker fee (0.06%), adjust if needed

        logger.info(f"{Fore.MAGENTA}--- Configuration Runes Summoned Successfully ---{Style.RESET_ALL}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = Fore.WHITE) -> Any:
        """Fetches an environment variable, casts its type, logs the value, and handles defaults or errors."""
        value_str = os.getenv(key)
        value = None
        log_source = ""

        if value_str is not None:
            # Log the raw value fetched from env at DEBUG level
            logger.debug(f"{color}Env Var Found: {key}='{value_str}'{Style.RESET_ALL}")
            log_source = f"(from env)"
            try:
                if cast_type == bool:
                    value = value_str.lower() in ['true', '1', 'yes', 'y']
                elif cast_type == Decimal:
                    # Use safe_decimal_conversion to handle potential errors gracefully during init
                    value = safe_decimal_conversion(value_str, default=None) # Return None on failure here
                    if value is None: # If conversion failed
                        raise InvalidOperation(f"Invalid Decimal value: '{value_str}'")
                elif cast_type is not None:
                    value = cast_type(value_str)
                else:
                    value = value_str # Keep as string if cast_type is None
            except (ValueError, TypeError, InvalidOperation) as e:
                logger.error(f"{Fore.RED}Invalid type/value for env var {key}: '{value_str}'. Expected {cast_type.__name__}. Error: {e}. Using default: '{default}'{Style.RESET_ALL}")
                value = default # Fallback to default on casting error
                log_source = f"(env parse error, using default: '{default}')"
        else:
            value = default
            log_source = f"(not set, using default: '{default}')" if default is not None else "(not set, no default)"

        if value is None and required:
            critical_msg = f"CRITICAL: Required environment variable '{key}' not set and no default value provided."
            logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{critical_msg}{Style.RESET_ALL}")
            raise ValueError(critical_msg)

        # Log the final assigned value at DEBUG level
        logger.debug(f"{color}Config Final: {key}: {value} {log_source}{Style.RESET_ALL}")
        return value

# --- Global CONFIG object ---
try:
    CONFIG = Config()
except ValueError as e:
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Configuration Error: {e}{Style.RESET_ALL}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}Unexpected Error initializing configuration: {e}{Style.RESET_ALL}")
    logger.debug(traceback.format_exc())
    sys.exit(1)

# --- Helper Functions --- (Decimal conversions, formatting, SMS)
# (Implementations are the same as provided in ps.py)

def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning default if conversion fails."""
    if value is None: return default
    try:
        # Handle potential numpy types explicitly
        if isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
             if np.isnan(value): return default
             value = str(value) # Convert numpy float to string first
        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
             value = str(value)
        return Decimal(str(value)) # Convert string to Decimal
    except (InvalidOperation, TypeError, ValueError):
        # logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}") # Too verbose for DEBUG
        return default

def format_order_id(order_id: str | int | None) -> str:
    """Returns last 6 chars of order ID or 'N/A'."""
    return str(order_id)[-6:] if order_id else "N/A"

def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal | str | None) -> str:
    """Formats price according to market precision rules using Decimal for input."""
    if price is None: return "N/A"
    try:
        # Convert to float for CCXT formatter, handle potential errors
        price_decimal = safe_decimal_conversion(price, default=Decimal('NaN'))
        if price_decimal.is_nan(): return str(price) # Fallback if conversion failed
        return exchange.price_to_precision(symbol, float(price_decimal))
    except (ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Error formatting price '{price}' for {symbol}: {e}{Style.RESET_ALL}")
        return str(price) # Absolute fallback

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal | str | None) -> str:
    """Formats amount according to market precision rules using Decimal for input."""
    if amount is None: return "N/A"
    try:
        # Convert to float for CCXT formatter
        amount_decimal = safe_decimal_conversion(amount, default=Decimal('NaN'))
        if amount_decimal.is_nan(): return str(amount) # Fallback
        return exchange.amount_to_precision(symbol, float(amount_decimal))
    except (ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Error formatting amount '{amount}' for {symbol}: {e}{Style.RESET_ALL}")
        return str(amount) # Absolute fallback

def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API if enabled."""
    if not CONFIG.enable_sms_alerts: return False
    if not CONFIG.sms_recipient_number: logger.warning("SMS alerts enabled, but recipient number missing."); return False
    try:
        command = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Attempting SMS to {'*****' + CONFIG.sms_recipient_number[-4:]}: \"{message[:50]}...\"{Style.RESET_ALL}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if result.returncode == 0: logger.success(f"{Fore.MAGENTA}SMS sent successfully.{Style.RESET_ALL}"); return True
        else: logger.error(f"{Fore.RED}SMS failed. RC:{result.returncode}, Err:{result.stderr.strip()}{Style.RESET_ALL}"); return False
    except FileNotFoundError: logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}"); return False
    except subprocess.TimeoutExpired: logger.error(f"{Fore.RED}SMS failed: Timeout after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}"); return False
    except Exception as e: logger.error(f"{Fore.RED}SMS failed: Unexpected error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc()); return False

# --- Exchange Initialization ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance."""
    # (Using logic from ps.py)
    logger.info(f"{Fore.BLUE}Initializing CCXT {CONFIG.exchange_id} connection...{Style.RESET_ALL}")
    if not CONFIG.api_key or not CONFIG.api_secret: logger.critical("API keys missing."); send_sms_alert("[ScalpBot] CRITICAL: API keys missing."); return None
    try:
        exchange = getattr(ccxt, CONFIG.exchange_id)({ # Use getattr for flexibility
            "apiKey": CONFIG.api_key, "secret": CONFIG.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "linear", "recvWindow": CONFIG.default_recv_window, "adjustForTimeDifference": True},
        })
        # Note: Testnet mode needs to be handled if exchange_id is not 'bybit'
        if CONFIG.exchange_id == 'bybit' and CONFIG.testnet_mode:
            logger.info("Setting sandbox mode for Bybit...")
            exchange.set_sandbox_mode(True)
        logger.debug("Loading markets (forced reload)...")
        exchange.load_markets(True)
        logger.debug("Performing initial balance check...")
        exchange.fetch_balance()
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}CCXT {CONFIG.exchange_id} Session Initialized (Testnet: {CONFIG.testnet_mode}).{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot] Initialized & authenticated ({'Testnet' if CONFIG.testnet_mode else 'LIVE'}).")
        return exchange
    except (ccxt.AuthenticationError, ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.critical(f"Initialization failed: {e}"); send_sms_alert(f"[ScalpBot] CRITICAL: Init FAILED: {type(e).__name__}."); return None

# --- Indicator Calculation Functions ---
# (Using implementations from ps.py: ehlers_volumetric_trend, calculate_indicators, etc.)

def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: Decimal) -> pd.DataFrame:
    """Calculate Ehlers Volumetric Trend indicator."""
    try:
        vwma = ta.vwma(df['close'], df['volume'], length=length)
        if vwma is None or vwma.isnull().all(): raise ValueError("VWMA calculation failed or returned all NaNs")
        df['vwma'] = vwma

        a = np.exp(-1.414 * np.pi / length); b = 2 * a * np.cos(1.414 * np.pi / length)
        c2 = b; c3 = -a * a; c1 = 1 - c2 - c3

        smoothed = np.zeros(len(df))
        vwma_values = vwma.values # Use numpy array for faster access
        for i in range(2, len(df)):
            # Use np.isnan for checking, faster than pd.notna on numpy arrays
            if not np.isnan(vwma_values[i]):
                smoothed[i] = c1 * vwma_values[i] + c2 * smoothed[i-1] + c3 * smoothed[i-2]
            else: # Handle NaN in input VWMA
                smoothed[i] = smoothed[i-1] if i > 0 else 0 # Carry forward previous smoothed value
        df['smoothed_vwma'] = smoothed

        trend = np.zeros(len(df), dtype=int)
        mult_factor_high = 1 + float(multiplier) / 100
        mult_factor_low = 1 - float(multiplier) / 100
        smoothed_vwma_shifted = df['smoothed_vwma'].shift(1).values # Shifted numpy array

        # Vectorized trend calculation where possible
        trend = np.where(smoothed > smoothed_vwma_shifted * mult_factor_high, 1, trend)
        trend = np.where(smoothed < smoothed_vwma_shifted * mult_factor_low, -1, trend)
        df['trend'] = trend
        df['trend'] = df['trend'].ffill().fillna(0).astype(int) # Forward fill initial NaNs

        trend_shifted = df['trend'].shift(1).values
        df['evt_buy'] = (df['trend'] == 1) & (trend_shifted != 1)
        df['evt_sell'] = (df['trend'] == -1) & (trend_shifted != -1)

        logger.debug("Ehlers Volumetric Trend calculated.")
        return df
    except Exception as e:
        logger.error(f"{Fore.RED}Error calculating Ehlers Volumetric Trend: {e}{Style.RESET_ALL}", exc_info=True)
        # Ensure columns exist even on error
        df[['vwma', 'smoothed_vwma', 'trend', 'evt_buy', 'evt_sell']] = pd.NA
        return df


def calculate_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, Decimal | None]:
    """Calculates indicators based on config, using pandas_ta."""
    try:
        logger.debug("Calculating Indicators...")
        df = df.copy() # Avoid modifying original DataFrame
        # Always calculate ATR
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=CONFIG.atr_calculation_period)
        current_atr = safe_decimal_conversion(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else None

        strategy = CONFIG.strategy_name
        if strategy == "DUAL_EHLERS_VOLUMETRIC":
            # Calculate primary EVT
            df = ehlers_volumetric_trend(df, CONFIG.evt_length, CONFIG.evt_multiplier)
            # Rename primary columns before calculating confirmation
            df = df.rename(columns={'vwma':'primary_vwma', 'smoothed_vwma':'primary_smoothed_vwma', 'trend':'primary_trend', 'evt_buy':'primary_evt_buy', 'evt_sell':'primary_evt_sell'})
            # Calculate confirmation EVT (will overwrite 'trend', 'evt_buy', 'evt_sell' from primary calc)
            df = ehlers_volumetric_trend(df, CONFIG.confirm_evt_length, CONFIG.confirm_evt_multiplier)
            # Rename confirmation columns
            df = df.rename(columns={'vwma':'confirm_vwma', 'smoothed_vwma':'confirm_smoothed_vwma', 'trend':'confirm_trend', 'evt_buy':'confirm_evt_buy', 'evt_sell':'confirm_evt_sell'})

        elif strategy == "STOCHRSI_MOMENTUM":
            # Note: pandas_ta stochrsi might have different parameter names or defaults. Verify implementation.
            # Let's assume default pandas_ta stochrsi calculation method
            stochrsi = ta.stochrsi(df['close'], length=CONFIG.stochrsi_stoch_length, rsi_length=CONFIG.stochrsi_rsi_length, k=CONFIG.stochrsi_k_period, d=CONFIG.stochrsi_d_period)
            if stochrsi is not None and not stochrsi.empty:
                # Adjust column names based on actual pandas_ta output
                k_col = stochrsi.columns[0] # Usually the K column
                d_col = stochrsi.columns[1] # Usually the D column
                df['stochrsi_k'] = stochrsi[k_col].apply(safe_decimal_conversion)
                df['stochrsi_d'] = stochrsi[d_col].apply(safe_decimal_conversion)
            else: df['stochrsi_k'], df['stochrsi_d'] = pd.NA, pd.NA
            momentum = ta.mom(df['close'], length=CONFIG.momentum_length)
            df['momentum'] = momentum.apply(safe_decimal_conversion) if momentum is not None else pd.NA

        elif strategy == "EHLERS_FISHER":
            # pandas_ta fisher transform
            fisher = ta.fisher(df['high'], df['low'], length=CONFIG.ehlers_fisher_length, signal=CONFIG.ehlers_fisher_signal_length)
            if fisher is not None and not fisher.empty:
                # Adjust column names based on actual pandas_ta output
                fish_col = fisher.columns[0] # Usually Fisher value
                sig_col = fisher.columns[1] # Usually Signal value
                df['fisher'] = fisher[fish_col].apply(safe_decimal_conversion)
                df['fisher_signal'] = fisher[sig_col].apply(safe_decimal_conversion)
            else: df['fisher'], df['fisher_signal'] = pd.NA, pd.NA

        elif strategy == "EHLERS_MA_CROSS":
            # Using EMA as placeholder per ps.py comment
            logger.warning(f"{Fore.YELLOW}Using EMA as placeholder for Ehlers MA Cross. Verify if suitable.{Style.RESET_ALL}")
            fast_ma = ta.ema(df['close'], length=CONFIG.ehlers_fast_period)
            slow_ma = ta.ema(df['close'], length=CONFIG.ehlers_slow_period)
            df['fast_ma'] = fast_ma.apply(safe_decimal_conversion) if fast_ma is not None else pd.NA
            df['slow_ma'] = slow_ma.apply(safe_decimal_conversion) if slow_ma is not None else pd.NA

        logger.debug(f"Indicator calculation complete. Current ATR: {current_atr}")
        return df, current_atr if current_atr and not current_atr.is_nan() else None
    except Exception as e:
        logger.error(f"{Fore.RED}Error calculating indicators: {e}{Style.RESET_ALL}", exc_info=True)
        # Return original df and None for ATR on failure
        return df, None


def generate_signals(df: pd.DataFrame, current_position: str) -> tuple[str | None, str | None]:
    """Generates entry/exit signals based on selected strategy and DataFrame columns."""
    entry_signal: str | None = None
    exit_signal: str | None = None
    exit_reason = "Strategy Exit" # Default exit reason

    if len(df) < 2: return None, None # Need previous candle

    last = df.iloc[-1]; prev = df.iloc[-2]

    try:
        strategy = CONFIG.strategy_name
        if strategy == "DUAL_EHLERS_VOLUMETRIC":
            # Check required columns exist and are not NA
            primary_cols = ['primary_evt_buy', 'primary_evt_sell', 'primary_trend']
            confirm_cols = ['confirm_trend'] # Only need confirm trend value
            if not all(c in last and pd.notna(last[c]) for c in primary_cols + confirm_cols):
                logger.debug(f"Signals ({strategy}): Skipping due to NA indicator values.")
                return None, None

            if current_position == CONFIG.POS_NONE:
                if last['primary_evt_buy'] and last['confirm_trend'] == 1:
                    entry_signal = CONFIG.SIDE_BUY
                elif last['primary_evt_sell'] and last['confirm_trend'] == -1:
                    entry_signal = CONFIG.SIDE_SELL
            elif current_position == CONFIG.POS_LONG:
                if last['primary_trend'] != 1: # Exit if primary trend is no longer 1
                    exit_signal = CONFIG.SIDE_SELL
                    exit_reason = "Primary EVT Trend Exit"
            elif current_position == CONFIG.POS_SHORT:
                if last['primary_trend'] != -1: # Exit if primary trend is no longer -1
                    exit_signal = CONFIG.SIDE_BUY
                    exit_reason = "Primary EVT Trend Exit"

        elif strategy == "STOCHRSI_MOMENTUM":
            req = ['stochrsi_k', 'stochrsi_d', 'momentum']
            k_n,d_n,m_n = last.get(req[0]), last.get(req[1]), last.get(req[2])
            k_p,d_p = prev.get(req[0]), prev.get(req[1])
            if any(v is None or not isinstance(v,Decimal) or v.is_nan() for v in [k_n,d_n,m_n,k_p,d_p]): return None, None

            if current_position == CONFIG.POS_NONE:
                if k_p<=d_p and k_n>d_n and k_n<CONFIG.stochrsi_oversold and m_n>CONFIG.POSITION_QTY_EPSILON: entry_signal=CONFIG.SIDE_BUY
                elif k_p>=d_p and k_n<d_n and k_n>CONFIG.stochrsi_overbought and m_n<-CONFIG.POSITION_QTY_EPSILON: entry_signal=CONFIG.SIDE_SELL
            elif current_position == CONFIG.POS_LONG:
                if k_p>=d_p and k_n<d_n: exit_signal=CONFIG.SIDE_SELL; exit_reason="StochRSI K<D"
            elif current_position == CONFIG.POS_SHORT:
                if k_p<=d_p and k_n>d_n: exit_signal=CONFIG.SIDE_BUY; exit_reason="StochRSI K>D"

        elif strategy == "EHLERS_FISHER":
            req=['fisher','fisher_signal']
            f_n,s_n=last.get(req[0]),last.get(req[1]); f_p,s_p=prev.get(req[0]),prev.get(req[1])
            if any(v is None or not isinstance(v,Decimal) or v.is_nan() for v in [f_n,s_n,f_p,s_p]): return None, None

            if current_position == CONFIG.POS_NONE:
                if f_p<=s_p and f_n>s_n: entry_signal=CONFIG.SIDE_BUY
                elif f_p>=s_p and f_n<s_n: entry_signal=CONFIG.SIDE_SELL
            elif current_position == CONFIG.POS_LONG:
                if f_p>=s_p and f_n<s_n: exit_signal=CONFIG.SIDE_SELL; exit_reason="Fisher Cross Down"
            elif current_position == CONFIG.POS_SHORT:
                if f_p<=s_p and f_n>s_n: exit_signal=CONFIG.SIDE_BUY; exit_reason="Fisher Cross Up"

        elif strategy == "EHLERS_MA_CROSS":
            req=['fast_ma','slow_ma']
            f_n,s_n=last.get(req[0]),last.get(req[1]); f_p,s_p=prev.get(req[0]),prev.get(req[1])
            if any(v is None or not isinstance(v,Decimal) or v.is_nan() for v in [f_n,s_n,f_p,s_p]): return None, None

            if current_position == CONFIG.POS_NONE:
                if f_p<=s_p and f_n>s_n: entry_signal=CONFIG.SIDE_BUY
                elif f_p>=s_p and f_n<s_n: entry_signal=CONFIG.SIDE_SELL
            elif current_position == CONFIG.POS_LONG:
                if f_p>=s_p and f_n<s_n: exit_signal=CONFIG.SIDE_SELL; exit_reason="MA Cross Down"
            elif current_position == CONFIG.POS_SHORT:
                if f_p<=s_p and f_n>s_n: exit_signal=CONFIG.SIDE_BUY; exit_reason="MA Cross Up"

        if entry_signal: logger.info(f"{Fore.GREEN}Signal: {entry_signal.upper()} ({CONFIG.strategy_name}){Style.RESET_ALL}")
        if exit_signal: logger.info(f"{Fore.YELLOW}Signal: Exit {'Long' if exit_signal==CONFIG.SIDE_SELL else 'Short'} ({exit_reason}){Style.RESET_ALL}")

        return entry_signal, exit_signal if exit_signal else None # Return reason only if exit is triggered
    except KeyError as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Missing expected indicator column: {e}{Style.RESET_ALL}")
        return None, None
    except Exception as e:
        logger.error(f"{Fore.RED}Signal Generation Error: Unexpected exception: {e}{Style.RESET_ALL}", exc_info=True)
        return None, None


# --- Integrated Helper Functions (analyze_order_book, get_market_data, get_current_position, set_leverage, wait_for_order_fill, close_position, cancel_open_orders) ---
# Using implementations adapted from previous integration efforts

def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes L2 order book, returns Decimals."""
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    # logger.debug(f"OB Fetch: L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})") # Reduce log noise
    if not exchange.has.get('fetchL2OrderBook'): return results
    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids = order_book.get('bids', []); asks = order_book.get('asks', [])
        if not bids or not asks: return results

        best_bid = safe_decimal_conversion(bids[0][0], default=None) if len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0], default=None) if len(asks[0]) > 0 else None
        results["best_bid"] = best_bid; results["best_ask"] = best_ask

        if best_bid and best_ask and best_ask > best_bid:
            results["spread"] = best_ask - best_bid
            # logger.debug(f"OB: Bid={best_bid:.4f}, Ask={best_ask:.4f}, Spread={results['spread']:.4f}")
        # else: logger.debug(f"OB: Bid={best_bid or 'N/A'}, Ask={best_ask or 'N/A'} (Spread N/A)")

        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:depth] if len(ask) > 1)
        # logger.debug(f"OB (Depth {depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")

        if ask_vol > CONFIG.POSITION_QTY_EPSILON:
            try: results["bid_ask_ratio"] = bid_vol / ask_vol # Bid / Ask Ratio
            except Exception: results["bid_ask_ratio"] = None
        # else: logger.debug("OB Ratio: N/A (Ask volume zero)")

    except Exception as e:
        logger.warning(f"{Fore.YELLOW}OB Fetch/Parse Error: {type(e).__name__}{Style.RESET_ALL}")
        results = dict.fromkeys(results) # Reset on error
    return results


def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data, ensuring numeric types and handling NaNs."""
    # (Using implementation from ps.py, adapted slightly)
    if not exchange.has.get("fetchOHLCV"): logger.error(f"{Fore.RED}Data Fetch: fetchOHLCV not supported.{Style.RESET_ALL}"); return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} ({interval})...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv: logger.warning(f"{Fore.YELLOW}Data Fetch: No OHLCV data returned.{Style.RESET_ALL}"); return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
        df.set_index("timestamp", inplace=True)
        df.dropna(subset=[df.index.name], inplace=True) # Drop invalid timestamps

        for col in ["open", "high", "low", "close", "volume"]: df[col] = pd.to_numeric(df[col], errors='coerce')
        if df.isnull().values.any():
            nan_counts = df.isnull().sum(); logger.warning(f"{Fore.YELLOW}Data Fetch: OHLCV has NaNs:\n{nan_counts[nan_counts > 0]}\nAttempting ffill...{Style.RESET_ALL}")
            df.ffill(inplace=True)
            if df.isnull().values.any(): logger.warning(f"{Fore.YELLOW}NaNs remain after ffill, attempting bfill...{Style.RESET_ALL}"); df.bfill(inplace=True)
            if df.isnull().values.any(): logger.error(f"{Fore.RED}Data Fetch: NaNs persist after fill. Cannot proceed.{Style.RESET_ALL}"); return None
        logger.debug(f"Data Fetch: Processed {len(df)} OHLCV candles.")
        return df
    except Exception as e: logger.warning(f"{Fore.YELLOW}Data Fetch: Error fetching/processing OHLCV: {e}{Style.RESET_ALL}"); return None

def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position using Bybit V5 logic, returns Decimals."""
    # (Using implementation from ps.py)
    default_pos = {'side': CONFIG.POS_NONE, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0")}
    market_id = None; market = None
    try: market = exchange.market(symbol); market_id = market['id']
    except Exception as e: logger.error(f"{Fore.RED}Position Check: Failed market info '{symbol}': {e}{Style.RESET_ALL}"); return default_pos
    try:
        if not exchange.has.get('fetchPositions'): logger.warning(f"{Fore.YELLOW}fetchPositions not supported.{Style.RESET_ALL}"); return default_pos
        params = {'category': 'linear'} if market.get('linear') else ({'category': 'inverse'} if market.get('inverse') else {})
        logger.debug(f"Position Check: Fetching positions for {symbol} (ID: {market_id}), params: {params}")
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        active_pos = None
        for pos in fetched_positions:
            info=pos.get('info',{}); pos_mid=info.get('symbol'); pos_idx=info.get('positionIdx',0); side_v5=info.get('side','None'); size_str=info.get('size')
            if pos_mid == market_id and pos_idx == 0 and side_v5 != 'None':
                size = safe_decimal_conversion(size_str)
                if abs(size) > CONFIG.POSITION_QTY_EPSILON: active_pos = pos; break
        if active_pos:
            try:
                size=safe_decimal_conversion(active_pos.get('info',{}).get('size')); entry=safe_decimal_conversion(active_pos.get('info',{}).get('avgPrice'))
                side=CONFIG.POS_LONG if active_pos.get('info',{}).get('side')=='Buy' else CONFIG.POS_SHORT; qty=abs(size)
                logger.info(f"{Fore.YELLOW}Position Check: Found ACTIVE {side} Qty={qty:.8f} @ Entry={entry:.4f}{Style.RESET_ALL}")
                return {'side': side, 'qty': qty, 'entry_price': entry}
            except Exception as parse_err: logger.warning(f"{Fore.YELLOW}Position Check: Error parsing active pos: {parse_err}. Data: {active_pos}{Style.RESET_ALL}"); return default_pos
        else: logger.info(f"Position Check: No active position found for {market_id}."); return default_pos
    except Exception as e: logger.warning(f"{Fore.YELLOW}Position Check: Error fetching positions: {e}{Style.RESET_ALL}"); return default_pos

def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage using Bybit V5 specific params."""
    # (Using implementation from ps.py)
    logger.info(f"{Fore.CYAN}Leverage Setting: Attempting {leverage}x for {symbol}...{Style.RESET_ALL}")
    try: market = exchange.market(symbol); assert market.get('contract')
    except Exception as e: logger.error(f"{Fore.RED}Leverage Setting: Failed market info/check for '{symbol}': {e}{Style.RESET_ALL}"); return False
    for attempt in range(CONFIG.RETRY_COUNT):
        try:
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            resp = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
            logger.success(f"{Fore.GREEN}Leverage Setting: Set to {leverage}x. Response: {resp}{Style.RESET_ALL}"); return True
        except ccxt.ExchangeError as e:
            err = str(e).lower(); if "leverage not modified" in err or "same as requested" in err or "110044" in str(e): logger.info(f"{Fore.CYAN}Leverage already set to {leverage}x.{Style.RESET_ALL}"); return True
            logger.warning(f"{Fore.YELLOW}Leverage Setting: Exchange error (Try {attempt+1}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.RETRY_COUNT - 1: time.sleep(CONFIG.RETRY_DELAY_SECONDS); else: logger.error(f"{Fore.RED}Leverage Setting: Failed after retries.{Style.RESET_ALL}")
        except Exception as e: logger.warning(f"{Fore.YELLOW}Leverage Setting: Network/Other error (Try {attempt+1}): {e}{Style.RESET_ALL}"); if attempt < CONFIG.RETRY_COUNT - 1: time.sleep(CONFIG.RETRY_DELAY_SECONDS); else: logger.error(f"{Fore.RED}Leverage Setting: Failed after retries.{Style.RESET_ALL}")
    return False

def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int) -> dict[str, Any] | None:
    """Waits for order fill (status 'closed'), returns filled order or None."""
    # (Using implementation from ps.py)
    start = time.monotonic(); oid_short = format_order_id(order_id)
    logger.info(f"{Fore.CYAN}Waiting for order ...{oid_short} ({symbol}) fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}")
    while time.monotonic() - start < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get('status')
            logger.debug(f"Order ...{oid_short} status: {status}")
            if status == 'closed': logger.success(f"{Fore.GREEN}Order ...{oid_short} confirmed FILLED.{Style.RESET_ALL}"); return order
            elif status in ['canceled', 'rejected', 'expired']: logger.error(f"{Fore.RED}Order ...{oid_short} FAILED status: '{status}'.{Style.RESET_ALL}"); return None
            time.sleep(0.5)
        except ccxt.OrderNotFound: time.sleep(1); logger.warning(f"{Fore.YELLOW}Order ...{oid_short} not found yet. Retrying...{Style.RESET_ALL}")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e: time.sleep(CONFIG.RETRY_DELAY_SECONDS); logger.warning(f"{Fore.YELLOW}API Error checking order ...{oid_short}: {e}. Retrying...{Style.RESET_ALL}")
    logger.error(f"{Fore.RED}Order ...{oid_short} TIMEOUT after {timeout_seconds}s.{Style.RESET_ALL}"); return None

def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal") -> dict[str, Any] | None:
    """Closes position with re-validation, uses Decimal, returns close order or None."""
    # (Using implementation from ps.py, adapted for Decimal)
    initial_side = position_to_close.get('side', CONFIG.POS_NONE)
    initial_qty = position_to_close.get('qty', Decimal("0.0"))
    market_base = symbol.split('/')[0]
    logger.info(f"{Fore.YELLOW}Close Position: Init {symbol}. Reason: {reason}. State: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")
    live_position = get_current_position(exchange, symbol)
    live_side = live_position['side']; live_qty = live_position['qty']
    if live_side == CONFIG.POS_NONE: logger.warning(f"{Fore.YELLOW}Close Position: Re-validation shows NO position. Aborting.{Style.RESET_ALL}"); return None
    close_exec_side = CONFIG.SIDE_SELL if live_side == CONFIG.POS_LONG else CONFIG.SIDE_BUY
    try:
        qty_str = format_amount(exchange, symbol, live_qty); qty_float = float(qty_str)
        if qty_float <= float(CONFIG.POSITION_QTY_EPSILON): logger.error(f"{Fore.RED}Close Position: Qty negligible ({qty_str}). Aborting.{Style.RESET_ALL}"); return None
        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Close Position: Closing {live_side} ({reason}): Exec {close_exec_side.upper()} MARKET {qty_str} {symbol} (reduceOnly)...{Style.RESET_ALL}")
        params = {'reduceOnly': True}
        # Add category for Bybit V5 close order
        market = exchange.market(symbol) # Reuse market info if possible
        if market and CONFIG.exchange_id == 'bybit':
            cat = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
            if cat: params['category'] = cat
        order = exchange.create_market_order(symbol=symbol, side=close_exec_side, amount=qty_float, params=params)
        fill_p=safe_decimal_conversion(order.get('average')); fill_q=safe_decimal_conversion(order.get('filled')); cost=safe_decimal_conversion(order.get('cost')); oid_short=format_order_id(order.get('id'))
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Close Position: Order ({reason}) submitted {symbol}. ID:...{oid_short}, Status:{order.get('status')}, Filled:{fill_q:.8f}/{qty_str}, AvgFill:{fill_p:.4f}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] Closed {live_side} {qty_str} @ ~{fill_p:.4f} ({reason}). ID:...{oid_short}")
        return order
    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}Close Position ({reason}) FAILED {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        err=str(e).lower(); if isinstance(e,ccxt.ExchangeError) and any(p in err for p in ["order would not reduce","position is zero","position size is zero"]): logger.warning(f"{Fore.YELLOW}Close Pos: Exchange indicates already closed. Assuming closed.{Style.RESET_ALL}"); return None
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}.")
    return None

def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Cancels all open orders for the symbol."""
    # (Using implementation from ps.py)
    logger.info(f"{Fore.CYAN}Order Cancel: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get('fetchOpenOrders'): logger.warning(f"{Fore.YELLOW}Order Cancel: fetchOpenOrders not supported.{Style.RESET_ALL}"); return
        params = {} # Add category hint if needed for fetchOpenOrders
        market = exchange.market(symbol) if symbol in exchange.markets else None
        if market and CONFIG.exchange_id == 'bybit':
            cat = 'linear' if market.get('linear') else ('inverse' if market.get('inverse') else None)
            if cat: params['category'] = cat; logger.debug(f"Cancel Orders: Using category {cat} for fetch.")
        open_orders = exchange.fetch_open_orders(symbol, params=params)
        if not open_orders: logger.info(f"{Fore.CYAN}Order Cancel: No open orders found.{Style.RESET_ALL}"); return
        logger.warning(f"{Fore.YELLOW}Order Cancel: Found {len(open_orders)} orders. Cancelling...{Style.RESET_ALL}"); cancelled_c, failed_c = 0, 0
        for order in open_orders:
            oid = order.get('id'); oinfo = f"...{format_order_id(oid)} ({order.get('type')} {order.get('side')})"
            if oid:
                try: exchange.cancel_order(oid, symbol, params=params); logger.info(f"{Fore.CYAN}Order Cancel: Success for {oinfo}{Style.RESET_ALL}"); cancelled_c += 1; time.sleep(0.1)
                except ccxt.OrderNotFound: logger.warning(f"{Fore.YELLOW}Order Cancel: Not found (already gone?): {oinfo}{Style.RESET_ALL}"); cancelled_c += 1
                except Exception as e: logger.error(f"{Fore.RED}Order Cancel: FAILED for {oinfo}: {e}{Style.RESET_ALL}"); failed_c += 1
            else: logger.warning(f"{Fore.YELLOW}Order Cancel: Skipping order with no ID.{Style.RESET_ALL}")
        log_level = logging.INFO if failed_c == 0 else logging.WARNING; logger.log(log_level, f"{Fore.CYAN}Order Cancel: Finished. Cancelled:{cancelled_c}, Failed:{failed_c}.{Style.RESET_ALL}")
        if failed_c > 0: send_sms_alert(f"[{symbol.split('/')[0]}] WARNING: Failed to cancel {failed_c} orders during {reason}.")
    except Exception as e: logger.error(f"{Fore.RED}Order Cancel: Error during fetch/cancel: {e}{Style.RESET_ALL}", exc_info=True)

# --- Trading Logic --- (Remains largely the same as ps.py, using the integrated functions)
def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle."""
    cycle_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Check Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}")

    required_rows = max(100, CONFIG.evt_length*2, CONFIG.confirm_evt_length*2, CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length + 5, CONFIG.atr_calculation_period*2, CONFIG.volume_ma_period*2) + CONFIG.API_FETCH_LIMIT_BUFFER
    if df is None or len(df) < required_rows: logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0}, need ~{required_rows}). Skipping.{Style.RESET_ALL}"); return

    action_taken_this_cycle = False
    try:
        # === 1. Calculate Indicators ===
        df, current_atr = calculate_indicators(df) # Pass df, returns df with indicators and current_atr
        if current_atr is None: logger.warning(f"{Fore.YELLOW}ATR calculation failed. SL/TP/Risk logic will be affected.{Style.RESET_ALL}")

        # === 2. Get Position & Market State ===
        position = get_current_position(exchange, symbol)
        position_side = position['side']; position_qty = position['qty']
        last = df.iloc[-1]; current_price = safe_decimal_conversion(last.get('close'))
        if current_price.is_nan() or current_price <= 0: logger.warning(f"{Fore.YELLOW}Invalid last close price ({current_price}). Skipping.{Style.RESET_ALL}"); return

        # Fetch OB if needed for confirmation logic
        ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit) if CONFIG.fetch_order_book_per_cycle else None

        # === 3. Log State ===
        vol_ratio = safe_decimal_conversion(last.get('volume_ratio')) if pd.notna(last.get('volume_ratio')) else None
        vol_spike = vol_ratio is not None and vol_ratio > CONFIG.volume_spike_threshold
        logger.info(f"State | Price: {format_price(exchange, symbol, current_price)}, ATR: {format_price(exchange, symbol, current_atr) if current_atr else 'N/A'}")
        logger.info(f"State | VolRatio: {vol_ratio:.2f if vol_ratio else 'N/A'}, Spike: {vol_spike} (Req={CONFIG.require_volume_spike_for_entry})")
        logger.info(f"State | Position: {position_side}, Qty: {format_amount(exchange, symbol, position_qty)}, Entry: {format_price(exchange, symbol, position['entry_price'])}")

        # === 4. Generate Signals ===
        entry_signal, exit_signal = generate_signals(df, position_side)

        # === 5. Execute Exits ===
        if position_side != CONFIG.POS_NONE and exit_signal is not None:
            exit_reason = "Strategy Exit" # Refine reason based on signal function if needed
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}*** TRADE EXIT SIGNAL: Closing {position_side} ({exit_reason}) ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, f"SL/TP/TSL before {exit_reason} Exit")
            close_result = close_position(exchange, symbol, position, reason=exit_reason)
            if close_result: time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS); return # Pause after close attempt

        # === 6. Execute Entries ===
        elif position_side == CONFIG.POS_NONE and entry_signal is not None:
            if current_atr is None: logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter due to invalid ATR.{Style.RESET_ALL}"); return

            # --- Confirmation Logic (Volume/OB) ---
            passes_volume = not CONFIG.require_volume_spike_for_entry or vol_spike
            # Fetch OB now if needed
            if not CONFIG.fetch_order_book_per_cycle and ob_data is None:
                 ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None
            ob_available = ob_data is not None and bid_ask_ratio is not None
            passes_long_ob = not ob_available or (bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
            passes_short_ob = not ob_available or (bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)

            confirmed_entry = (entry_signal == CONFIG.SIDE_BUY and passes_volume and passes_long_ob) or \
                              (entry_signal == CONFIG.SIDE_SELL and passes_volume and passes_short_ob)

            if confirmed_entry:
                side = CONFIG.SIDE_BUY if entry_signal == CONFIG.SIDE_BUY else CONFIG.SIDE_SELL
                sig_type = "LONG" if side == CONFIG.SIDE_BUY else "SHORT"
                logger.success(f"{Back.GREEN if side == CONFIG.SIDE_BUY else Back.RED}{Fore.WHITE}{Style.BRIGHT}*** CONFIRMED {sig_type} ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}")
                cancel_open_orders(exchange, symbol, f"Pre-{sig_type}-Entry Cleanup")
                place_result = place_risked_market_order(
                    exchange, symbol, side, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                    CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                    CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent
                )
                # If entry was placed, cycle logic ends here
                if place_result: return
            else:
                logger.info(f"Entry signal ({entry_signal}) present but confirmation failed (Vol:{passes_volume}, OB:{passes_long_ob if entry_signal == CONFIG.SIDE_BUY else passes_short_ob}). Holding cash.")

        # If no action taken (no exit, no entry)
        elif position_side == CONFIG.POS_NONE:
             logger.info("Holding Cash. No entry signal this cycle.")

    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL ERROR in trade_logic: {type(e).__name__}")
    finally:
        logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Check End: {symbol} =========={Style.RESET_ALL}\n")


# --- Graceful Shutdown ---
def graceful_shutdown(exchange: ccxt.Exchange | None, symbol: str | None) -> None:
    """Attempts to close position and cancel orders before exiting."""
    # (Using implementation from ps.py)
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Shutdown requested. Initiating graceful exit...{Style.RESET_ALL}")
    market_base = symbol.split('/')[0] if symbol else "Bot"
    send_sms_alert(f"[{market_base}] Shutdown requested. Attempting cleanup...")
    if not exchange or not symbol: logger.warning(f"{Fore.YELLOW}Shutdown: Exchange/Symbol not available.{Style.RESET_ALL}"); return
    try:
        cancel_open_orders(exchange, symbol, reason="Graceful Shutdown")
        time.sleep(1)
        position = get_current_position(exchange, symbol)
        if position['side'] != CONFIG.POS_NONE:
            logger.warning(f"{Fore.YELLOW}Shutdown: Active {position['side']} position found. Closing...{Style.RESET_ALL}")
            close_result = close_position(exchange, symbol, position, reason="Shutdown")
            if close_result:
                logger.info(f"{Fore.CYAN}Shutdown: Close order placed. Waiting {CONFIG.POST_CLOSE_DELAY_SECONDS * 2}s for confirmation...{Style.RESET_ALL}")
                time.sleep(CONFIG.POST_CLOSE_DELAY_SECONDS * 2)
                final_pos = get_current_position(exchange, symbol)
                if final_pos['side'] == CONFIG.POS_NONE: logger.success(f"{Fore.GREEN}{Style.BRIGHT}Shutdown: Position confirmed CLOSED.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] Position confirmed CLOSED on shutdown.")
                else: logger.error(f"{Back.RED}Shutdown: FAILED TO CONFIRM closure! Final: {final_pos['side']} Qty={final_pos['qty']:.8f}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR: Failed confirm closure! MANUAL CHECK!")
            else: logger.error(f"{Back.RED}Shutdown: Failed to place close order. MANUAL CHECK NEEDED.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR: Failed PLACE close on shutdown. MANUAL CHECK!")
        else: logger.info(f"{Fore.GREEN}Shutdown: No active position found.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] No active position found on shutdown.")
    except Exception as e: logger.error(f"{Fore.RED}Shutdown: Error during cleanup: {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] Error during shutdown sequence: {type(e).__name__}")
    logger.info(f"{Fore.YELLOW}{Style.BRIGHT}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")


# --- Main Execution ---
def main() -> None:
    """Main function to initialize, set up, and run the trading loop."""
    # (Using implementation from ps.py)
    start_time = time.strftime('%Y-%m-%d %H:%M:%S %Z'); logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v2.1.0 Initializing ({start_time}) ---{Style.RESET_ALL}"); logger.info(f"{Fore.CYAN}--- Strategy: {CONFIG.strategy_name} ---{Style.RESET_ALL}"); logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK !!! ---{Style.RESET_ALL}")
    exchange: ccxt.Exchange | None = None; symbol: str | None = None; run_bot: bool = True; cycle_count: int = 0
    try:
        exchange = initialize_exchange();
        if not exchange: return
        try:
            sym_input = input(f"{Fore.YELLOW}Enter symbol {Style.DIM}(Default [{CONFIG.symbol}]){Style.NORMAL}: {Style.RESET_ALL}").strip()
            symbol_to_use = sym_input or CONFIG.symbol; market = exchange.market(symbol_to_use); symbol = market['symbol']
            if not market.get('contract'): raise ValueError("Not a contract/futures market")
            logger.info(f"{Fore.GREEN}Using Symbol: {symbol} (Type: {market.get('type')}){Style.RESET_ALL}")
            if not set_leverage(exchange, symbol, CONFIG.leverage): raise RuntimeError("Leverage setup failed")
        except (ccxt.BadSymbol, KeyError, ValueError, RuntimeError) as e: logger.critical(f"Symbol/Leverage setup failed: {e}"); send_sms_alert(f"[{CONFIG.symbol.split('/')[0]}] CRITICAL: Symbol/Leverage FAILED ({e})."); return
        except Exception as e: logger.critical(f"Unexpected setup error: {e}"); send_sms_alert("[ScalpBot] CRITICAL: Unexpected setup error."); return

        logger.info(f"{Fore.MAGENTA}--- Configuration Summary ---{Style.RESET_ALL}") # Log summary
        logger.info(f" Symbol: {symbol}, TF: {CONFIG.interval}, Lev: {CONFIG.leverage}x, Strategy: {CONFIG.strategy_name}")
        logger.info(f" Risk: {CONFIG.risk_per_trade_percentage:.3%}/trade, MaxPosVal: {CONFIG.max_order_usdt_amount:.2f} USDT")
        logger.info(f" SL: {CONFIG.atr_stop_loss_multiplier}*ATR({CONFIG.atr_calculation_period}), MinDist: {CONFIG.min_sl_distance_percent:.2%}")
        logger.info(f" TP: {CONFIG.take_profit_multiplier}x SL Distance")
        logger.info(f" TSL: {CONFIG.trailing_stop_percentage:.2%}, ActOffset: {CONFIG.trailing_stop_activation_offset_percent:.2%}")
        logger.info(f" Confirm: Vol={CONFIG.require_volume_spike_for_entry}, OB={CONFIG.fetch_order_book_per_cycle}")
        logger.info(f"{Fore.MAGENTA}{'-'*27}{Style.RESET_ALL}")
        market_base = symbol.split('/')[0]; send_sms_alert(f"[{market_base}] Bot configured ({CONFIG.strategy_name}). Starting loop.")

        while run_bot:
            cycle_start = time.monotonic(); cycle_count += 1; logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Start ---{Style.RESET_ALL}")
            try:
                data_limit = max(100, CONFIG.evt_length*2, CONFIG.confirm_evt_length*2, CONFIG.stochrsi_rsi_length+CONFIG.stochrsi_stoch_length+5, CONFIG.atr_calculation_period*2, CONFIG.volume_ma_period*2) + CONFIG.API_FETCH_LIMIT_BUFFER
                df = get_market_data(exchange, symbol, CONFIG.interval, limit=data_limit)
                if df is not None and not df.empty: trade_logic(exchange, symbol, df.copy())
                else: logger.warning(f"{Fore.YELLOW}No valid market data. Skipping cycle.{Style.RESET_ALL}")
            except ccxt.RateLimitExceeded as e: logger.warning(f"{Back.YELLOW}{Fore.BLACK}Rate Limit: {e}. Sleeping longer...{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds*5); send_sms_alert(f"[{market_base}] WARNING: Rate limit hit!")
            except ccxt.NetworkError as e: logger.warning(f"{Fore.YELLOW}Network error: {e}. Retrying.{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds)
            except ccxt.ExchangeNotAvailable as e: logger.error(f"{Back.RED}Exchange unavailable: {e}. Sleeping much longer...{Style.RESET_ALL}"); time.sleep(CONFIG.sleep_seconds*10); send_sms_alert(f"[{market_base}] ERROR: Exchange unavailable!")
            except ccxt.AuthenticationError as e: logger.critical(f"{Back.RED}Auth Error: {e}. Stopping NOW.{Style.RESET_ALL}"); run_bot=False; send_sms_alert(f"[{market_base}] CRITICAL: Auth Error! Stopping NOW.")
            except ccxt.ExchangeError as e: logger.error(f"{Fore.RED}Unhandled Exchange Error: {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR: Unhandled Exchange error: {type(e).__name__}"); time.sleep(CONFIG.sleep_seconds)
            except Exception as e: logger.exception(f"{Back.RED}!!! UNEXPECTED CRITICAL ERROR: {e} !!!{Style.RESET_ALL}"); run_bot=False; send_sms_alert(f"[{market_base}] CRITICAL UNEXPECTED ERROR: {type(e).__name__}! Stopping NOW.")
            if run_bot: elapsed=time.monotonic()-cycle_start; sleep_dur=max(0,CONFIG.sleep_seconds-elapsed); logger.debug(f"Cycle {cycle_count} time: {elapsed:.2f}s. Sleeping: {sleep_dur:.2f}s."); time.sleep(sleep_dur)
    except KeyboardInterrupt: logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt received. Stopping...{Style.RESET_ALL}"); run_bot=False
    finally:
        graceful_shutdown(exchange, symbol)
        market_base_final = symbol.split('/')[0] if symbol else "Bot"; send_sms_alert(f"[{market_base_final}] Bot process terminated.")
        logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

