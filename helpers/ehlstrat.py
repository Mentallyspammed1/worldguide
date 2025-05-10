#!/usr/bin/env python

"""Ehlers Volumetric Trend Strategy for Bybit V5 (v1.4 - Robust Setup & Cleanup)

Handles potential synchronous nature of set_leverage, ensures
proper error handling during setup, conditional strategy start,
and reliable exchange closing.
"""

import asyncio
import logging
import os
import sys
from decimal import ROUND_DOWN, Decimal

# Third-party libraries
import ccxt
import pandas as pd
from dotenv import load_dotenv

# --- Import Colorama ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Provide dummy Fore, Style, Back if colorama is not installed
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()
    COLORAMA_AVAILABLE = False
    print(
        "Warning: 'colorama' library not found. Run 'pip install colorama' for vibrant logs.",
        file=sys.stderr,
    )


# --- Import Custom Modules ---
try:
    # Ensure bybit_helpers has all required functions like initialize_bybit, set_leverage, validate_market etc.
    import bybit_helpers as bybit
    import indicators as ind
    from bybit_utils import (
        format_amount,
        format_order_id,
        format_price,
        safe_decimal_conversion,
        send_sms_alert,
    )
    from neon_logger import setup_logger
except ImportError as e:
    # Use dummy colors if colorama failed but we reach here
    err_back = Back.RED if COLORAMA_AVAILABLE else ""
    err_fore = Fore.WHITE if COLORAMA_AVAILABLE else ""
    warn_fore = Fore.YELLOW if COLORAMA_AVAILABLE else ""
    reset_all = Style.RESET_ALL if COLORAMA_AVAILABLE else ""
    print(f"{err_back}{err_fore}Error importing helper modules: {e}{reset_all}")
    print(
        f"{warn_fore}Ensure bybit_helpers.py, indicators.py, neon_logger.py, and bybit_utils.py are accessible.{reset_all}"
    )
    sys.exit(1)

# --- Load Environment Variables ---
load_dotenv()


# --- Configuration Class ---
class Config:
    def __init__(self):
        # Exchange & API
        self.EXCHANGE_ID: str = "bybit"
        self.API_KEY: str | None = os.getenv("BYBIT_API_KEY")
        self.API_SECRET: str | None = os.getenv("BYBIT_API_SECRET")
        self.TESTNET_MODE: bool = (
            os.getenv("BYBIT_TESTNET_MODE", "true").lower() == "true"
        )
        self.DEFAULT_RECV_WINDOW: int = int(os.getenv("DEFAULT_RECV_WINDOW", 10000))

        # Symbol & Market
        self.SYMBOL: str = os.getenv(
            "SYMBOL", "BTC/USDT:USDT"
        )  # Example: BTC/USDT Perpetual
        self.USDT_SYMBOL: str = "USDT"
        self.EXPECTED_MARKET_TYPE: str = "swap"  # e.g., 'swap', 'future', 'spot'
        self.EXPECTED_MARKET_LOGIC: str = "linear"  # e.g., 'linear', 'inverse'
        self.TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")
        self.OHLCV_LIMIT: int = int(
            os.getenv("OHLCV_LIMIT", 200)
        )  # Candles for indicators

        # Account & Position Settings
        self.DEFAULT_LEVERAGE: int = int(os.getenv("LEVERAGE", 10))
        self.DEFAULT_MARGIN_MODE: str = os.getenv(
            "MARGIN_MODE", "cross"
        ).lower()  # 'cross' or 'isolated'
        self.DEFAULT_POSITION_MODE: str = os.getenv(
            "POSITION_MODE", "one-way"
        ).lower()  # 'one-way' or 'hedge'
        self.RISK_PER_TRADE: Decimal = Decimal(
            os.getenv("RISK_PER_TRADE", "0.01")
        )  # 1% risk

        # Order Settings
        self.DEFAULT_SLIPPAGE_PCT: Decimal = Decimal(
            os.getenv("DEFAULT_SLIPPAGE_PCT", "0.005")
        )  # 0.5%
        self.ORDER_BOOK_FETCH_LIMIT: int = 25
        self.SHALLOW_OB_FETCH_DEPTH: int = 5

        # Fees
        self.TAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_TAKER_FEE", "0.00055"))
        self.MAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_MAKER_FEE", "0.0002"))

        # Strategy Parameters (Ehlers Volumetric Trend)
        self.EVT_ENABLED: bool = (
            os.getenv("EVT_ENABLED", "true").lower() == "true"
        )  # Master switch
        self.EVT_LENGTH: int = int(os.getenv("EVT_LENGTH", 7))
        self.EVT_MULTIPLIER: float = float(os.getenv("EVT_MULTIPLIER", 2.5))
        self.STOP_LOSS_ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", 14))
        self.STOP_LOSS_ATR_MULTIPLIER: Decimal = Decimal(
            os.getenv("ATR_MULTIPLIER", "2.5")
        )

        # Retry & Timing
        self.RETRY_COUNT: int = int(os.getenv("RETRY_COUNT", 3))
        self.RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY", 2.0))
        self.LOOP_DELAY_SECONDS: int = int(
            os.getenv("LOOP_DELAY", 60)
        )  # Wait time between cycles

        # Logging & Alerts
        self.LOG_CONSOLE_LEVEL: str = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
        self.LOG_FILE_LEVEL: str = os.getenv("LOG_FILE_LEVEL", "DEBUG").upper()
        self.LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "ehlers_strategy.log")
        self.ENABLE_SMS_ALERTS: bool = (
            os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"
        )
        self.SMS_RECIPIENT_NUMBER: str | None = os.getenv("SMS_RECIPIENT_NUMBER")
        self.SMS_TIMEOUT_SECONDS: int = 30
        self.TWILIO_ACCOUNT_SID: str | None = os.getenv(
            "TWILIO_ACCOUNT_SID"
        )  # Needed for SMS
        self.TWILIO_AUTH_TOKEN: str | None = os.getenv(
            "TWILIO_AUTH_TOKEN"
        )  # Needed for SMS
        self.TWILIO_PHONE_NUMBER: str | None = os.getenv(
            "TWILIO_PHONE_NUMBER"
        )  # Needed for SMS

        # Constants
        self.SIDE_BUY: str = "buy"
        self.SIDE_SELL: str = "sell"
        self.POS_LONG: str = "LONG"
        self.POS_SHORT: str = "SHORT"
        self.POS_NONE: str = "NONE"
        self.POSITION_QTY_EPSILON: Decimal = Decimal(
            "1e-9"
        )  # Small value for float comparisons

        # --- Derived/Helper Attributes ---
        self.indicator_settings = {
            "atr_period": self.STOP_LOSS_ATR_PERIOD,
            "evt_length": self.EVT_LENGTH,
            "evt_multiplier": self.EVT_MULTIPLIER,
        }
        self.analysis_flags = {
            "use_atr": True,
            "use_evt": self.EVT_ENABLED,
            # Add other flags if needed by indicators.py
        }
        self.strategy_params = {
            "ehlers_volumetric": {
                "evt_length": self.EVT_LENGTH,
                "evt_multiplier": self.EVT_MULTIPLIER,
            }
            # Add other strategy params if needed
        }
        self.strategy = {"name": "ehlers_volumetric"}

        # --- Validate SMS Config if enabled ---
        if self.ENABLE_SMS_ALERTS and not all(
            [
                self.SMS_RECIPIENT_NUMBER,
                self.TWILIO_ACCOUNT_SID,
                self.TWILIO_AUTH_TOKEN,
                self.TWILIO_PHONE_NUMBER,
            ]
        ):
            print(
                f"{Fore.YELLOW}Warning: SMS alerts enabled, but one or more Twilio config variables (SID, TOKEN, NUMBER) or recipient number are missing.{Style.RESET_ALL}"
            )
            self.ENABLE_SMS_ALERTS = False  # Disable if config is incomplete


# --- Global Variables ---
# Use type hints for better clarity and safety
logger: logging.Logger | None = None
exchange: ccxt.Exchange | None = None
CONFIG: Config | None = None

# --- Core Strategy Functions (Example implementations assumed present) ---
# These functions should ideally take `exchange` and `config` as arguments
# rather than relying solely on globals, but we'll work with the existing pattern.


def calculate_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame | None:
    """Calculates indicators needed for the strategy (Synchronous)."""
    if logger is None:
        print("Logger not initialized in calculate_indicators")
        return None  # Early exit if logger missing
    if df is None or df.empty:
        logger.error(
            f"{Fore.RED}Cannot calculate indicators: Input DataFrame is empty.{Style.RESET_ALL}"
        )
        return None
    try:
        indicator_config = {
            "indicator_settings": config.indicator_settings,
            "analysis_flags": config.analysis_flags,
            "strategy_params": config.strategy_params,
            "strategy": config.strategy,
        }
        # Assuming ind.calculate_all_indicators is synchronous and robust
        df_with_indicators = ind.calculate_all_indicators(
            df.copy(), indicator_config
        )  # Use copy to avoid modifying original

        if df_with_indicators is None or df_with_indicators.empty:
            logger.error(
                f"{Fore.RED}Indicator calculation returned empty or None DataFrame.{Style.RESET_ALL}"
            )
            return None

        # Validate required columns exist
        required_cols = []
        if config.EVT_ENABLED:
            required_cols.extend(
                [
                    f"evt_trend_{config.EVT_LENGTH}",
                    f"evt_buy_{config.EVT_LENGTH}",
                    f"evt_sell_{config.EVT_LENGTH}",
                ]
            )
        if config.analysis_flags.get("use_atr", False):
            required_cols.append(f"ATRr_{config.STOP_LOSS_ATR_PERIOD}")

        missing_cols = [
            col for col in required_cols if col not in df_with_indicators.columns
        ]
        if missing_cols:
            logger.error(
                f"{Fore.RED}Required indicator columns missing after calculation: {', '.join(missing_cols)}{Style.RESET_ALL}"
            )
            # Attempt to calculate ATR if missing (as an example fix)
            if f"ATRr_{config.STOP_LOSS_ATR_PERIOD}" in missing_cols and all(
                c in df_with_indicators for c in ["high", "low", "close"]
            ):
                try:
                    atr_result = df_with_indicators.ta.atr(
                        length=config.STOP_LOSS_ATR_PERIOD, append=False
                    )
                    if atr_result is not None:
                        df_with_indicators[atr_result.name] = atr_result
                        logger.info(
                            f"{Fore.CYAN}Calculated missing ATR column: {atr_result.name}{Style.RESET_ALL}"
                        )
                        missing_cols.remove(
                            f"ATRr_{config.STOP_LOSS_ATR_PERIOD}"
                        )  # Remove if added
                    else:
                        logger.error(
                            f"{Fore.RED}Failed fallback calculation for missing ATR.{Style.RESET_ALL}"
                        )
                except Exception as atr_err:
                    logger.error(
                        f"{Fore.RED}Error during fallback ATR calculation: {atr_err}{Style.RESET_ALL}",
                        exc_info=True,
                    )

            # If columns are still missing after potential fixes, fail
            if missing_cols:
                return None

        logger.debug(
            f"Indicators calculated. DataFrame shape: {df_with_indicators.shape}"
        )
        return df_with_indicators
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error calculating indicators: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


def generate_signals(df_ind: pd.DataFrame, config: Config) -> str | None:
    """Generates trading signals ('buy', 'sell', or None) based on the last row (Synchronous)."""
    if logger is None:
        print("Logger not initialized in generate_signals")
        return None
    if df_ind is None or df_ind.empty:
        return None
    if not config.EVT_ENABLED:
        logger.debug("EVT indicator is disabled, no signals generated.")
        return None
    try:
        latest = df_ind.iloc[-1]
        trend_col = f"evt_trend_{config.EVT_LENGTH}"
        buy_col = f"evt_buy_{config.EVT_LENGTH}"
        sell_col = f"evt_sell_{config.EVT_LENGTH}"

        # Check required columns for EVT exist
        if not all(col in latest.index for col in [trend_col, buy_col, sell_col]):
            logger.warning(
                f"{Fore.YELLOW}EVT signal columns missing ({trend_col}, etc.) in latest data. Index: {latest.name}{Style.RESET_ALL}"
            )
            return None

        # Check for NaN values which can cause issues
        if (
            pd.isna(latest[trend_col])
            or pd.isna(latest[buy_col])
            or pd.isna(latest[sell_col])
        ):
            logger.debug(
                f"Latest indicator data contains NaN values. Trend={latest[trend_col]}, Buy={latest[buy_col]}, Sell={latest[sell_col]}. No signal."
            )
            return None

        trend = int(latest[trend_col])  # Assuming trend is numerical (-1, 0, 1)
        buy_signal = bool(latest[buy_col])  # Assuming buy/sell are boolean or 0/1
        sell_signal = bool(latest[sell_col])

        logger.debug(
            f"Signal Check: Index={latest.name}, Close={latest.get('close', 'N/A'):.4f}, "
            f"{trend_col}={trend}, {buy_col}={buy_signal}, {sell_col}={sell_signal}"
        )

        # Simple Logic: Enter on explicit buy/sell flag if trend agrees (optional)
        # Refine this logic based on your exact strategy rules
        if buy_signal:  # and trend == 1: # Optional: Add trend confirmation
            logger.info(
                f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}"
            )
            return config.SIDE_BUY
        elif sell_signal:  # and trend == -1: # Optional: Add trend confirmation
            logger.info(
                f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}"
            )
            return config.SIDE_SELL

        return None  # No signal
    except IndexError:
        logger.warning(
            f"{Fore.YELLOW}Could not access latest indicator data (IndexError), DataFrame might be too short.{Style.RESET_ALL}"
        )
        return None
    except KeyError as e:
        logger.error(
            f"{Fore.RED}Missing expected column in latest data for signal generation: {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error generating signals: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


def calculate_stop_loss(
    df_ind: pd.DataFrame, side: str, entry_price: Decimal, config: Config
) -> Decimal | None:
    """Calculates the initial stop-loss price based on ATR (Synchronous)."""
    global exchange  # Access global exchange object
    if logger is None:
        print("Logger not initialized in calculate_stop_loss")
        return None
    if exchange is None:  # Critical check
        logger.error(
            f"{Fore.RED}Cannot calculate stop-loss: Exchange object is None.{Style.RESET_ALL}"
        )
        return None
    if df_ind is None or df_ind.empty:
        logger.error(
            f"{Fore.RED}Cannot calculate stop-loss: Indicator DataFrame is missing or empty.{Style.RESET_ALL}"
        )
        return None
    if not config.analysis_flags.get("use_atr", False):
        logger.warning(
            f"{Fore.YELLOW}ATR usage is disabled in config, cannot calculate ATR-based stop loss.{Style.RESET_ALL}"
        )
        return None  # Or implement a fallback SL method

    try:
        atr_col = f"ATRr_{config.STOP_LOSS_ATR_PERIOD}"
        if atr_col not in df_ind.columns:
            logger.error(
                f"{Fore.RED}ATR column '{atr_col}' not found for stop-loss calculation.{Style.RESET_ALL}"
            )
            return None

        latest_atr_raw = df_ind.iloc[-1][atr_col]
        if pd.isna(latest_atr_raw):
            logger.warning(
                f"{Fore.YELLOW}Latest ATR value is NaN, cannot calculate stop-loss.{Style.RESET_ALL}"
            )
            return None  # Or use fallback

        latest_atr = safe_decimal_conversion(latest_atr_raw)

        if latest_atr is None or latest_atr <= Decimal(0):
            logger.warning(
                f"{Fore.YELLOW}Invalid ATR value ({latest_atr_raw} -> {latest_atr}), cannot calculate stop-loss accurately.{Style.RESET_ALL}"
            )
            # Consider a simple percentage-based fallback?
            # sl_fallback = entry_price * (Decimal(1) - Decimal("0.01")) if side == config.SIDE_BUY else entry_price * (Decimal(1) + Decimal("0.01"))
            # logger.info(f"{Fore.CYAN}Using fixed percentage fallback SL: {format_price(exchange, config.SYMBOL, sl_fallback)}{Style.RESET_ALL}")
            # return format_price(exchange, config.SYMBOL, sl_fallback, precise=True) # Return precise Decimal
            return None  # Current: Fail if ATR is invalid

        stop_offset = latest_atr * config.STOP_LOSS_ATR_MULTIPLIER
        stop_loss_price_raw = (
            entry_price - stop_offset
            if side == config.SIDE_BUY
            else entry_price + stop_offset
        )

        # Format price according to market precision *before* sanity checks
        formatted_sl_str = format_price(exchange, config.SYMBOL, stop_loss_price_raw)
        stop_loss_price = safe_decimal_conversion(formatted_sl_str)

        if stop_loss_price is None:
            logger.error(
                f"{Fore.RED}Failed to format/convert calculated SL price {stop_loss_price_raw} precisely.{Style.RESET_ALL}"
            )
            return None

        # Sanity checks after formatting
        if side == config.SIDE_BUY and stop_loss_price >= entry_price:
            logger.warning(
                f"{Fore.YELLOW}Calculated Buy SL ({formatted_sl_str}) >= Entry ({format_price(exchange, config.SYMBOL, entry_price)}). Adjusting slightly below.{Style.RESET_ALL}"
            )
            # Adjust based on tick size if possible, otherwise small percentage
            tick_size = safe_decimal_conversion(
                exchange.market(config.SYMBOL).get("precision", {}).get("price"),
                Decimal("0.000001"),
            )
            stop_loss_price = entry_price - tick_size
            stop_loss_price = safe_decimal_conversion(
                format_price(exchange, config.SYMBOL, stop_loss_price)
            )  # Reformat after adjustment

        elif side == config.SIDE_SELL and stop_loss_price <= entry_price:
            logger.warning(
                f"{Fore.YELLOW}Calculated Sell SL ({formatted_sl_str}) <= Entry ({format_price(exchange, config.SYMBOL, entry_price)}). Adjusting slightly above.{Style.RESET_ALL}"
            )
            tick_size = safe_decimal_conversion(
                exchange.market(config.SYMBOL).get("precision", {}).get("price"),
                Decimal("0.000001"),
            )
            stop_loss_price = entry_price + tick_size
            stop_loss_price = safe_decimal_conversion(
                format_price(exchange, config.SYMBOL, stop_loss_price)
            )  # Reformat

        if stop_loss_price is None:  # Check again after potential adjustment
            logger.error(
                f"{Fore.RED}Failed to get valid SL price after adjustment.{Style.RESET_ALL}"
            )
            return None

        logger.info(
            f"Calculated SL for {side.upper()} at {format_price(exchange, config.SYMBOL, stop_loss_price)} (Entry: {format_price(exchange, config.SYMBOL, entry_price)}, ATR: {latest_atr:.4f}, Mult: {config.STOP_LOSS_ATR_MULTIPLIER})"
        )
        return stop_loss_price  # Return the precise Decimal value

    except IndexError:
        logger.warning(
            f"{Fore.YELLOW}Could not access latest indicator data (IndexError) for SL calculation.{Style.RESET_ALL}"
        )
        return None
    except KeyError as e:
        logger.error(
            f"{Fore.RED}Missing expected column in indicator data for SL calculation: {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error calculating stop-loss: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


async def calculate_position_size(
    exchange: ccxt.Exchange,
    symbol: str,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    config: Config,
) -> Decimal | None:
    """Calculates position size based on risk percentage and stop-loss distance (Asynchronous due to balance fetch)."""
    if logger is None:
        print("Logger not initialized in calculate_position_size")
        return None
    # No need to check exchange is None here, as it's passed directly
    try:
        # Assuming bybit.fetch_usdt_balance IS ASYNC
        logger.debug(f"{Fore.CYAN}# Awaiting USDT balance...{Style.RESET_ALL}")
        total_balance, available_balance = await bybit.fetch_usdt_balance(
            exchange, config
        )

        if available_balance is None or available_balance <= Decimal("0"):
            logger.error(
                f"{Fore.RED}Cannot calculate position size: Zero or invalid available balance ({available_balance}).{Style.RESET_ALL}"
            )
            return None

        risk_amount_usd = available_balance * config.RISK_PER_TRADE
        price_diff = abs(entry_price - stop_loss_price)

        if price_diff <= Decimal("1e-9"):  # Use epsilon for safety
            logger.error(
                f"{Fore.RED}Cannot calculate position size: Entry price ({entry_price}) and SL price ({stop_loss_price}) are too close or invalid (Diff: {price_diff}).{Style.RESET_ALL}"
            )
            return None

        # Position size = (Amount willing to risk) / (Risk per unit in quote currency)
        position_size_base = risk_amount_usd / price_diff

        # Get market details for precision and limits
        market = exchange.market(symbol)
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        precision = market.get("precision", {})

        min_qty_str = amount_limits.get("min")
        min_qty = safe_decimal_conversion(
            min_qty_str, default=None
        )  # Default to None if conversion fails

        qty_precision_digits = precision.get(
            "amount"
        )  # Number of decimal places for amount

        if qty_precision_digits is None:
            logger.warning(
                f"{Fore.YELLOW}Could not determine quantity precision digits for {symbol}. Using raw calculation and minimal step size.{Style.RESET_ALL}"
            )
            step_size = Decimal("1e-8")  # Fallback step size
        else:
            # Calculate step size from precision digits (e.g., 3 digits -> 0.001 step)
            step_size = Decimal("1") / (Decimal("10") ** int(qty_precision_digits))

        # Adjust position size down to the nearest valid step size
        # Use Decimal's quantize method for proper rounding based on step size
        position_size_adjusted = (position_size_base // step_size) * step_size
        # Alternative using quantize: position_size_adjusted = position_size_base.quantize(step_size, rounding=ROUND_DOWN)

        if position_size_adjusted <= Decimal(0):
            logger.warning(
                f"{Fore.YELLOW}Calculated position size is zero or negative after adjusting for step size {step_size}. Original: {position_size_base:.8f}{Style.RESET_ALL}"
            )
            return None

        # Check against minimum order size *after* adjusting for step size
        if min_qty is not None and position_size_adjusted < min_qty:
            logger.warning(
                f"{Fore.YELLOW}Calculated position size ({position_size_adjusted}) is below exchange minimum ({min_qty}). Cannot place trade.{Style.RESET_ALL}"
            )
            return None

        # Optional: Check against maximum order size
        max_qty_str = amount_limits.get("max")
        max_qty = safe_decimal_conversion(max_qty_str, default=None)
        if max_qty is not None and position_size_adjusted > max_qty:
            logger.warning(
                f"{Fore.YELLOW}Calculated position size ({position_size_adjusted}) exceeds exchange maximum ({max_qty}). Capping at maximum.{Style.RESET_ALL}"
            )
            position_size_adjusted = max_qty.quantize(
                step_size, rounding=ROUND_DOWN
            )  # Adjust max qty to step size too

        # Optional: Check if size exceeds available margin (simple check, doesn't account for existing positions perfectly in cross mode)
        # cost = position_size_adjusted * entry_price / config.DEFAULT_LEVERAGE
        # if cost > available_balance:
        #     logger.warning(f"Calculated position cost ({cost:.2f}) exceeds available balance ({available_balance:.2f}) with {config.DEFAULT_LEVERAGE}x leverage. Reducing size.")
        #     # Reduce size proportionally or recalculate based on available balance (more complex)
        #     # For simplicity, we might just prevent the trade here or let the exchange reject it.
        #     # return None

        logger.info(
            f"Calculated position size: {format_amount(exchange, symbol, position_size_adjusted)} {symbol.split('/')[0]} "
            f"(Risk: {risk_amount_usd:.2f} {config.USDT_SYMBOL}, Balance: {available_balance:.2f} {config.USDT_SYMBOL}, Price Diff: {price_diff:.4f})"
        )
        return position_size_adjusted

    except ccxt.NetworkError as e:
        logger.warning(
            f"{Fore.YELLOW}Network error fetching balance during position size calculation: {e}{Style.RESET_ALL}"
        )
        return None
    except ccxt.ExchangeError as e:
        logger.error(
            f"{Fore.RED}Exchange error fetching balance during position size calculation: {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error calculating position size: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


async def run_strategy(config: Config, current_exchange: ccxt.bybit):
    """Main asynchronous trading loop."""
    global exchange, logger  # Use globals established in main()
    if logger is None:
        print("Logger not initialized in run_strategy")
        return
    if not current_exchange:  # Check the passed exchange object
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Strategy cannot run: Invalid exchange object received.{Style.RESET_ALL}"
        )
        return
    exchange = current_exchange  # Assign to global for helper functions that might still use it

    logger.info(
        f"{Fore.MAGENTA}--- Starting Ehlers Volumetric Strategy Loop for {config.SYMBOL} on {config.TIMEFRAME} ---{Style.RESET_ALL}"
    )
    logger.info(
        f"Risk: {config.RISK_PER_TRADE:.2%}, Leverage: {config.DEFAULT_LEVERAGE}x, Margin: {config.DEFAULT_MARGIN_MODE}, Position: {config.DEFAULT_POSITION_MODE}"
    )
    logger.info(
        f"EVT Params: Enabled={config.EVT_ENABLED}, Length={config.EVT_LENGTH}, Multiplier={config.EVT_MULTIPLIER}"
    )
    logger.info(
        f"ATR Params: Period={config.STOP_LOSS_ATR_PERIOD}, Multiplier={config.STOP_LOSS_ATR_MULTIPLIER}"
    )

    # Persists across loops to track the SL order associated with the current position
    stop_loss_orders: dict[str, str] = {}

    while True:
        try:
            cycle_start_time = pd.Timestamp.now(tz="UTC")
            logger.info(
                f"{Fore.BLUE}{Style.BRIGHT}"
                + "-" * 25
                + f" Cycle Start: {cycle_start_time.isoformat()} "
                + "-" * 25
                + f"{Style.RESET_ALL}"
            )

            # --- 1. Fetch Current State (Requires async calls) ---
            logger.debug(
                f"{Fore.CYAN}# Awaiting current position state...{Style.RESET_ALL}"
            )
            # Assuming get_current_position_bybit_v5 IS async
            current_position = await bybit.get_current_position_bybit_v5(
                exchange, config.SYMBOL, config
            )
            if current_position is None:
                logger.warning(
                    f"{Fore.YELLOW}Failed to get current position state. Retrying next cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue  # Skip rest of the cycle

            current_side = current_position["side"]  # POS_LONG, POS_SHORT, POS_NONE
            current_qty = current_position["qty"]  # Decimal
            entry_price = current_position.get("entry_price")  # Decimal or None
            logger.info(
                f"Current Position: Side={current_side}, Qty={format_amount(exchange, config.SYMBOL, current_qty)}, Entry={format_price(exchange, config.SYMBOL, entry_price) if entry_price else 'N/A'}"
            )

            # --- 2. Fetch Data & Calculate Indicators (Requires async fetches) ---
            logger.debug(f"{Fore.CYAN}# Awaiting OHLCV data...{Style.RESET_ALL}")
            # Assuming fetch_ohlcv_paginated IS async
            ohlcv_df = await bybit.fetch_ohlcv_paginated(
                exchange,
                config.SYMBOL,
                config.TIMEFRAME,
                limit_per_req=1000,
                max_total_candles=config.OHLCV_LIMIT
                + 50,  # Fetch bit extra for indicator stability
                config=config,
            )
            if (
                ohlcv_df is None
                or ohlcv_df.empty
                or len(ohlcv_df) < config.OHLCV_LIMIT // 2
            ):  # Check length
                logger.warning(
                    f"{Fore.YELLOW}Could not fetch sufficient OHLCV data ({len(ohlcv_df) if ohlcv_df is not None else 0} candles). Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue

            logger.debug(f"{Fore.CYAN}# Awaiting ticker data...{Style.RESET_ALL}")
            # Assuming fetch_ticker_validated IS async
            ticker = await bybit.fetch_ticker_validated(exchange, config.SYMBOL, config)
            if ticker is None or ticker.get("last") is None:
                logger.warning(
                    f"{Fore.YELLOW}Could not fetch valid ticker data. Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue
            current_price = safe_decimal_conversion(ticker["last"])
            if current_price is None:
                logger.error(
                    f"{Fore.RED}Could not convert ticker price '{ticker['last']}' to Decimal. Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue
            logger.debug(
                f"Current Price: {format_price(exchange, config.SYMBOL, current_price)}"
            )

            # Indicator calculation is synchronous (CPU-bound)
            df_with_indicators = calculate_indicators(ohlcv_df, config)
            if df_with_indicators is None:
                logger.warning(
                    f"{Fore.YELLOW}Failed to calculate indicators. Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue

            # --- 3. Generate Trading Signal (Synchronous) ---
            signal = generate_signals(
                df_with_indicators, config
            )  # Returns 'buy', 'sell', or None
            logger.info(f"Generated Signal: {signal if signal else 'None'}")

            # --- 4. Handle Exits (Requires async order calls) ---
            if current_side != config.POS_NONE:
                should_exit = False
                exit_reason = ""

                # Exit based on EVT Trend flip (Example logic)
                if config.EVT_ENABLED:
                    latest_trend_val = df_with_indicators.iloc[-1].get(
                        f"evt_trend_{config.EVT_LENGTH}"
                    )
                    if not pd.isna(latest_trend_val):
                        latest_trend = int(latest_trend_val)
                        if current_side == config.POS_LONG and latest_trend == -1:
                            should_exit = True
                            exit_reason = "EVT Trend flipped Short"
                        elif current_side == config.POS_SHORT and latest_trend == 1:
                            should_exit = True
                            exit_reason = "EVT Trend flipped Long"
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Latest EVT trend is NaN, cannot use for exit signal.{Style.RESET_ALL}"
                        )

                # Add other exit conditions here (e.g., Take Profit, Trailing Stop logic)

                if should_exit:
                    logger.warning(
                        f"{Fore.YELLOW}{Style.BRIGHT}Exit condition met for {current_side} position: {exit_reason}. Attempting to close.{Style.RESET_ALL}"
                    )

                    # Cancel existing SL order *before* closing position
                    sl_order_id = stop_loss_orders.pop(
                        config.SYMBOL, None
                    )  # Remove ID if found
                    if sl_order_id:
                        try:
                            logger.info(
                                f"Attempting to cancel existing SL order: {format_order_id(sl_order_id)}"
                            )
                            logger.debug(
                                f"{Fore.CYAN}# Awaiting cancellation of SL order {sl_order_id}...{Style.RESET_ALL}"
                            )
                            # Assuming cancel_order IS async
                            cancelled = await bybit.cancel_order(
                                exchange, config.SYMBOL, sl_order_id, config=config
                            )
                            if cancelled:
                                logger.info(
                                    f"{Fore.GREEN}Successfully cancelled SL order {format_order_id(sl_order_id)} before closing.{Style.RESET_ALL}"
                                )
                            else:
                                # If cancel returns False, it might mean the order was already filled/cancelled. Log warning but proceed.
                                logger.warning(
                                    f"{Fore.YELLOW}Attempt to cancel SL order {format_order_id(sl_order_id)} returned non-True. It might have already executed or been cancelled.{Style.RESET_ALL}"
                                )
                        except ccxt.OrderNotFound:
                            logger.warning(
                                f"{Fore.YELLOW}SL order {format_order_id(sl_order_id)} not found (already filled/cancelled?). Proceeding with close.{Style.RESET_ALL}"
                            )
                        except NameError:
                            logger.error(
                                f"{Fore.RED}bybit_helpers.cancel_order function not found/imported correctly.{Style.RESET_ALL}"
                            )
                        except Exception as e:
                            logger.error(
                                f"{Fore.RED}Failed to cancel SL order {format_order_id(sl_order_id)}: {e}{Style.RESET_ALL}",
                                exc_info=True,
                            )
                            # Decide whether to proceed with closing or wait. Proceeding is usually safer.
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}No tracked SL order ID found to cancel for the existing {current_side} position.{Style.RESET_ALL}"
                        )

                    # Close the position using a reduce-only market order
                    logger.debug(
                        f"{Fore.CYAN}# Awaiting position close market order...{Style.RESET_ALL}"
                    )
                    # Assuming close_position_reduce_only IS async
                    close_order = await bybit.close_position_reduce_only(
                        exchange,
                        config.SYMBOL,
                        config,
                        position_to_close=current_position,  # Pass full details if needed by helper
                        reason=exit_reason,
                    )

                    if close_order and close_order.get("id"):
                        logger.success(
                            f"{Fore.GREEN}Position successfully closed via market order {format_order_id(close_order['id'])} based on: {exit_reason}.{Style.RESET_ALL}"
                        )
                        if config.ENABLE_SMS_ALERTS:
                            alert_msg = f"[{config.SYMBOL}] {current_side} Position Closed ({format_amount(exchange, config.SYMBOL, current_qty)}). Reason: {exit_reason}"
                            send_sms_alert(alert_msg, config)
                        # Optional: Short delay to allow position update propagation
                        await asyncio.sleep(10)
                        continue  # Skip entry logic for this cycle as we just exited
                    else:
                        logger.error(
                            f"{Back.RED}{Fore.WHITE}Failed to submit position close order for exit signal! Manual intervention likely required.{Style.RESET_ALL}"
                        )
                        if config.ENABLE_SMS_ALERTS:
                            alert_msg = f"[{config.SYMBOL}] URGENT: Failed to submit close order for {current_side} position ({format_amount(exchange, config.SYMBOL, current_qty)}) on signal: {exit_reason}!"
                            send_sms_alert(alert_msg, config)
                        # Consider adding logic to retry closing or halt trading here
                        await asyncio.sleep(
                            config.LOOP_DELAY_SECONDS
                        )  # Wait before next cycle attempt
                        continue  # Skip entry logic

            # --- 5. Handle Entries (Requires async order calls) ---
            elif (
                current_side == config.POS_NONE and signal
            ):  # Only enter if flat and signal exists
                logger.info(
                    f"{Fore.CYAN}{Style.BRIGHT}Attempting to enter {signal.upper()} position based on signal...{Style.RESET_ALL}"
                )

                # Cancel any potentially lingering orders before entering
                logger.debug(
                    f"{Fore.CYAN}# Awaiting pre-entry order cleanup...{Style.RESET_ALL}"
                )
                # Assuming cancel_all_orders IS async
                cancelled_count = await bybit.cancel_all_orders(
                    exchange, config.SYMBOL, config, reason="Pre-Entry Cleanup"
                )
                if cancelled_count is not None and cancelled_count > 0:
                    logger.info(
                        f"Pre-entry cleanup: Cancelled {cancelled_count} potential lingering order(s)."
                    )
                elif cancelled_count == 0:
                    logger.info("Pre-entry cleanup: No lingering orders found.")
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Pre-entry order cleanup potentially failed or could not determine cancelled count. Proceeding with caution.{Style.RESET_ALL}"
                    )

                # Calculate Stop Loss (Synchronous, uses latest data)
                # Use current_price as the estimated entry for SL calculation
                stop_loss_price = calculate_stop_loss(
                    df_with_indicators, signal, current_price, config
                )
                if not stop_loss_price:  # Check if None or potentially zero/invalid
                    logger.error(
                        f"{Fore.RED}Could not calculate a valid stop-loss based on current data. Cannot enter trade.{Style.RESET_ALL}"
                    )
                    await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                    continue

                # Calculate Position Size (Asynchronous, fetches balance)
                logger.debug(
                    f"{Fore.CYAN}# Awaiting position size calculation...{Style.RESET_ALL}"
                )
                position_size = await calculate_position_size(
                    exchange, config.SYMBOL, current_price, stop_loss_price, config
                )
                if not position_size:  # Check if None or zero
                    logger.error(
                        f"{Fore.RED}Could not calculate a valid position size. Cannot enter trade.{Style.RESET_ALL}"
                    )
                    await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                    continue

                # Place Market Entry Order (Asynchronous)
                logger.info(
                    f"Placing {signal.upper()} market order for {format_amount(exchange, config.SYMBOL, position_size)} {config.SYMBOL.split('/')[0]}..."
                )
                logger.debug(
                    f"{Fore.CYAN}# Awaiting market entry order placement...{Style.RESET_ALL}"
                )
                # Assuming place_market_order_slippage_check IS async and returns order dict or None
                entry_order = await bybit.place_market_order_slippage_check(
                    exchange, config.SYMBOL, signal, position_size, config
                )

                if entry_order and entry_order.get("id"):
                    order_id_short = format_order_id(entry_order["id"])
                    avg_fill_price = safe_decimal_conversion(entry_order.get("average"))
                    filled_qty_order = safe_decimal_conversion(
                        entry_order.get("filled", 0)
                    )

                    logger.success(
                        f"{Fore.GREEN}Entry market order {order_id_short} submitted ({signal.upper()} {format_amount(exchange, config.SYMBOL, position_size)}). "
                        f"Filled Qty (from order): {format_amount(exchange, config.SYMBOL, filled_qty_order)}, AvgPrice: {format_price(exchange, config.SYMBOL, avg_fill_price) if avg_fill_price else 'N/A'}{Style.RESET_ALL}"
                    )

                    # Wait briefly for order fill and position update
                    logger.debug(
                        f"{Fore.CYAN}# Waiting briefly for position confirmation...{Style.RESET_ALL}"
                    )
                    await asyncio.sleep(7)  # Adjust delay as needed

                    # Verify position opened correctly
                    logger.debug(
                        f"{Fore.CYAN}# Awaiting position confirmation after entry...{Style.RESET_ALL}"
                    )
                    pos_after_entry = await bybit.get_current_position_bybit_v5(
                        exchange, config.SYMBOL, config
                    )

                    if (
                        pos_after_entry
                        and pos_after_entry["side"] == signal.upper()
                        and pos_after_entry["qty"] > config.POSITION_QTY_EPSILON
                    ):
                        # Use the actual filled quantity from the position info for SL
                        actual_filled_qty = pos_after_entry["qty"]
                        actual_entry_price = pos_after_entry.get(
                            "entry_price", current_price
                        )  # Use actual entry if available
                        pos_qty_formatted = format_amount(
                            exchange, config.SYMBOL, actual_filled_qty
                        )
                        entry_price_formatted = format_price(
                            exchange, config.SYMBOL, actual_entry_price
                        )

                        logger.info(
                            f"{Fore.GREEN}{Style.BRIGHT}Position confirmed OPEN: {pos_after_entry['side']} {pos_qty_formatted} @ ~{entry_price_formatted}{Style.RESET_ALL}"
                        )

                        # Place Stop Loss order using actual filled quantity
                        sl_side = (
                            config.SIDE_SELL
                            if signal == config.SIDE_BUY
                            else config.SIDE_BUY
                        )
                        logger.info(
                            f"Placing {sl_side.upper()} stop-loss order for {pos_qty_formatted} at {format_price(exchange, config.SYMBOL, stop_loss_price)}..."
                        )
                        logger.debug(
                            f"{Fore.CYAN}# Awaiting native stop-loss placement...{Style.RESET_ALL}"
                        )
                        # Assuming place_native_stop_loss IS async
                        sl_order = await bybit.place_native_stop_loss(
                            exchange,
                            config.SYMBOL,
                            sl_side,
                            qty=actual_filled_qty,  # Use actual quantity
                            stop_price=stop_loss_price,  # Use originally calculated SL price
                            config=config,
                        )

                        if sl_order and sl_order.get("id"):
                            sl_id_short = format_order_id(sl_order["id"])
                            logger.success(
                                f"{Fore.GREEN}Native stop-loss order {sl_id_short} placed successfully for {pos_qty_formatted} at {format_price(exchange, config.SYMBOL, stop_loss_price)}.{Style.RESET_ALL}"
                            )
                            stop_loss_orders[config.SYMBOL] = sl_order[
                                "id"
                            ]  # Track the SL order ID

                            if config.ENABLE_SMS_ALERTS:
                                sl_price_fmt = format_price(
                                    exchange, config.SYMBOL, stop_loss_price
                                )
                                alert_msg = f"[{config.SYMBOL}] Entered {signal.upper()} {pos_qty_formatted} @ ~{entry_price_formatted}. SL {sl_id_short} @ {sl_price_fmt}"
                                send_sms_alert(alert_msg, config)
                        else:
                            # CRITICAL: Failed to place SL after opening position
                            logger.error(
                                f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to place stop-loss order after entering {signal.upper()} position! Attempting emergency close.{Style.RESET_ALL}"
                            )
                            if config.ENABLE_SMS_ALERTS:
                                alert_msg = f"[{config.SYMBOL}] URGENT: Failed to place SL after {signal.upper()} entry ({pos_qty_formatted})! Closing position."
                                send_sms_alert(alert_msg, config)

                            logger.debug(
                                f"{Fore.CYAN}# Awaiting emergency position close due to failed SL placement...{Style.RESET_ALL}"
                            )
                            # Use the latest position info for closing
                            close_order = await bybit.close_position_reduce_only(
                                exchange,
                                config.SYMBOL,
                                config,
                                position_to_close=pos_after_entry,
                                reason="Failed SL Placement",
                            )
                            if close_order and close_order.get("id"):
                                logger.warning(
                                    f"{Fore.YELLOW}Position closed via order {format_order_id(close_order['id'])} due to failed SL placement.{Style.RESET_ALL}"
                                )
                            else:
                                logger.critical(
                                    f"{Back.RED}{Fore.WHITE}EMERGENCY FAILED: FAILED TO CLOSE POSITION ({pos_qty_formatted}) AFTER FAILED SL PLACEMENT! MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                                )
                                if config.ENABLE_SMS_ALERTS:
                                    send_sms_alert(
                                        f"[{config.SYMBOL}] !!! CRITICAL MANUAL ACTION NEEDED: Failed to close position {pos_qty_formatted} after failed SL placement !!!",
                                        config,
                                    )
                            # Clear SL tracking just in case
                            stop_loss_orders.pop(config.SYMBOL, None)
                    else:
                        # Position not confirmed or quantity is zero
                        pos_side_report = (
                            pos_after_entry["side"] if pos_after_entry else "N/A"
                        )
                        pos_qty_report = (
                            pos_after_entry["qty"] if pos_after_entry else "N/A"
                        )
                        logger.error(
                            f"{Fore.RED}Entry order {order_id_short} submitted, but position confirmation failed or quantity is insufficient. "
                            f"Position State: Side={pos_side_report}, Qty={pos_qty_report}. Manual check advised.{Style.RESET_ALL}"
                        )
                        if config.ENABLE_SMS_ALERTS:
                            alert_msg = f"[{config.SYMBOL}] URGENT: Entry order {order_id_short} confirmation failed! Pos Side: {pos_side_report}, Qty: {pos_qty_report}. Check manually!"
                            send_sms_alert(alert_msg, config)
                        # Consider cancelling the submitted order if possible, though it might have partially filled.

                else:
                    # Market order placement itself failed
                    logger.error(
                        f"{Fore.RED}Entry market order placement failed. No order ID received. Check exchange status and API logs.{Style.RESET_ALL}"
                    )
                    # No alert here unless it persists, as it might be a temporary issue.

            # --- 6. Wait for next cycle ---
            cycle_end_time = pd.Timestamp.now(tz="UTC")
            cycle_duration = (cycle_end_time - cycle_start_time).total_seconds()
            wait_time = max(0, config.LOOP_DELAY_SECONDS - cycle_duration)
            logger.info(
                f"Cycle complete. Duration: {cycle_duration:.2f}s. Waiting {wait_time:.2f}s for next cycle..."
            )
            await asyncio.sleep(wait_time)

        # --- Exception Handling for the Main Loop ---
        except ccxt.NetworkError as e:
            logger.warning(
                f"{Fore.YELLOW}Network Error in main loop: {e}. Retrying after {config.LOOP_DELAY_SECONDS * 2}s delay...{Style.RESET_ALL}"
            )
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 2)
        except ccxt.ExchangeNotAvailable as e:
            logger.error(
                f"{Fore.RED}Exchange Not Available: {e}. Possibly maintenance. Waiting longer ({config.LOOP_DELAY_SECONDS * 5}s)...{Style.RESET_ALL}"
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] Exchange Not Available: {e}. Pausing.", config
                )
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 5)
        except ccxt.RateLimitExceeded as e:
            logger.warning(
                f"{Fore.YELLOW}Rate Limit Exceeded: {e}. Waiting {config.LOOP_DELAY_SECONDS * 3}s...{Style.RESET_ALL}"
            )
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 3)
        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Authentication Error during runtime: {e}. API keys might be invalid or expired. Stopping strategy.{Style.RESET_ALL}",
                exc_info=True,
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] CRITICAL: Authentication Error during runtime! Strategy stopping.",
                    config,
                )
            break  # Exit the loop on auth errors
        except ccxt.ExchangeError as e:
            # Catch broader exchange errors not handled specifically above
            logger.error(
                f"{Fore.RED}Unhandled Exchange Error in main loop: {e}. Retrying after {config.LOOP_DELAY_SECONDS}s delay...{Style.RESET_ALL}",
                exc_info=True,
            )
            # Optional: Add SMS for persistent or specific exchange errors
            await asyncio.sleep(config.LOOP_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.warning(
                f"{Fore.YELLOW}{Style.BRIGHT}Keyboard interrupt received. Initiating graceful shutdown...{Style.RESET_ALL}"
            )
            break  # Exit the loop
        except NameError as e:
            # Often happens if colorama failed import and dummy Fore/Style weren't used everywhere
            logger.critical(
                f"{Back.RED}{Fore.WHITE}A NameError occurred: {e}. This might be due to failed library imports (like colorama).{Style.RESET_ALL}",
                exc_info=True,
            )
            if not COLORAMA_AVAILABLE and (
                "Fore" in str(e) or "Back" in str(e) or "Style" in str(e)
            ):
                logger.critical(
                    f"{Fore.YELLOW}Suggestion: Ensure {Style.BRIGHT}'pip install colorama'{Style.RESET_ALL}{Fore.YELLOW} is installed and the import succeeded.{Style.RESET_ALL}"
                )
            # Consider breaking the loop if critical functionality is affected
            break
        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.critical(
                f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL ERROR IN MAIN LOOP !!!{Style.RESET_ALL}",
                exc_info=True,
            )
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Error Type: {type(e).__name__}, Message: {e}{Style.RESET_ALL}"
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] CRITICAL ERROR: {type(e).__name__} in main loop. Check logs!",
                    config,
                )
            logger.info(
                f"{Fore.YELLOW}Attempting to continue after critical error... pausing for {config.LOOP_DELAY_SECONDS * 3}s{Style.RESET_ALL}"
            )
            await asyncio.sleep(
                config.LOOP_DELAY_SECONDS * 3
            )  # Longer pause after unexpected error

    logger.info(
        f"{Fore.MAGENTA}--- Ehlers Volumetric Strategy Loop Stopped ---{Style.RESET_ALL}"
    )


# --- Asynchronous Main Function (Setup & Execution) ---
async def main():
    global logger, exchange, CONFIG  # Allow assignment to globals

    # --- Initialize Logger (Must happen first) ---
    log_file_path = os.getenv(
        "LOG_FILE_PATH", "ehlers_strategy.log"
    )  # Get path here for early use
    try:
        logger = setup_logger(
            logger_name="EhlersStrategy",
            log_file=log_file_path,
            console_level=logging.getLevelName(
                os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
            ),
            file_level=logging.getLevelName(
                os.getenv("LOG_FILE_LEVEL", "DEBUG").upper()
            ),
            third_party_log_level=logging.WARNING,  # Suppress noisy logs from libraries
        )
    except Exception as e:
        # Use print because logger failed
        print(f"FATAL: Failed to initialize logger: {e}", file=sys.stderr)
        print(f"Log file path attempted: {log_file_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load Configuration ---
    try:
        CONFIG = Config()
    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Failed to load or initialize configuration: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        sys.exit(1)

    # --- Validate Core Config ---
    if not CONFIG.API_KEY or not CONFIG.API_SECRET:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}API Key or Secret not found in environment variables. Grant the script access! Exiting.{Style.RESET_ALL}"
        )
        sys.exit(1)
    logger.info(
        f"Configuration loaded. Strategy: {CONFIG.strategy['name']}, Symbol: {CONFIG.SYMBOL}, Timeframe: {CONFIG.TIMEFRAME}, Testnet: {CONFIG.TESTNET_MODE}"
    )
    if CONFIG.ENABLE_SMS_ALERTS:
        logger.info(f"SMS Alerts Enabled for: {CONFIG.SMS_RECIPIENT_NUMBER}")
    else:
        logger.info("SMS Alerts Disabled.")

    # --- Setup Phase (Initialize Exchange, Set Leverage, Validate Market) ---
    setup_success = False
    try:
        # --- 1. Initialize Exchange (Synchronous/blocking call typically) ---
        logger.info(f"Initializing connection to {CONFIG.EXCHANGE_ID}...")
        exchange = bybit.initialize_bybit(CONFIG)  # Assuming this is synchronous
        if not exchange:
            # Error logged within initialize_bybit hopefully
            raise ConnectionError(
                "Failed to initialize Bybit exchange object."
            )  # Raise error to enter except block

        logger.success(
            f"{Fore.GREEN}Exchange connection initialized successfully.{Style.RESET_ALL}"
        )

        # --- 2. Load Markets (Crucial before validation or setting leverage) ---
        logger.info("Loading markets from exchange...")
        await exchange.load_markets()  # Use await, load_markets is async
        logger.info(f"Markets loaded. Found {len(exchange.markets)} markets.")

        # --- 3. Validate Market (Synchronous check after loading) ---
        logger.info(f"Validating market configuration for {CONFIG.SYMBOL}...")
        market_details = bybit.validate_market(
            exchange, CONFIG.SYMBOL, CONFIG
        )  # Assuming synchronous
        if not market_details:
            # Error logged within validate_market
            raise ValueError(
                f"Market validation failed for {CONFIG.SYMBOL}."
            )  # Raise error

        logger.success(
            f"{Fore.GREEN}Market {CONFIG.SYMBOL} validated: Type={market_details.get('type')}, Logic={market_details.get('linear', 'N/A')}/{market_details.get('inverse', 'N/A')}{Style.RESET_ALL}"
        )

        # --- 4. Set Leverage (Handle sync/async and return value carefully) ---
        logger.info(
            f"Attempting to set leverage for {CONFIG.SYMBOL} to {CONFIG.DEFAULT_LEVERAGE}x..."
        )

        # --- YOU MUST CHECK YOUR `bybit.set_leverage` DEFINITION ---
        # Option A: If `bybit.set_leverage` is defined as `async def`:
        # try:
        #     leverage_set_result = await bybit.set_leverage(exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_LEVERAGE, CONFIG)
        #     if not leverage_set_result: # Check if helper returns False/None on failure
        #          raise ccxt.ExchangeError("Helper function set_leverage returned failure")
        # except TypeError as e:
        #     logger.critical(f"{Back.RED}{Fore.WHITE}TypeError calling set_leverage: Is it truly async? {e}{Style.RESET_ALL}", exc_info=True)
        #     raise # Re-raise to be caught by outer block
        # except Exception as e: # Catch other potential errors from the awaitable call
        #      logger.critical(f"{Back.RED}{Fore.WHITE}Error during awaited set_leverage call: {e}{Style.RESET_ALL}", exc_info=True)
        #      raise # Re-raise

        # Option B: If `bybit.set_leverage` is defined as `def` (synchronous):
        try:
            # DO NOT use await here
            leverage_set_result = bybit.set_leverage(
                exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_LEVERAGE, CONFIG
            )
            # Check the return value explicitly for failure indication (e.g., False, None)
            # The helper should log the specific API error, this check confirms failure.
            if (
                not leverage_set_result
            ):  # Adjust condition based on what your helper returns on failure
                # Helper function should have logged the specific Bybit error
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}Helper function set_leverage indicated failure (returned: {leverage_set_result}). See previous logs for API error.{Style.RESET_ALL}"
                )
                raise ccxt.ExchangeError(
                    "Leverage setting failed as reported by helper function."
                )
        except TypeError as e:
            # This specific TypeError *shouldn't* happen if we don't use await, but catch defensively
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Unexpected TypeError calling synchronous set_leverage: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            raise  # Re-raise
        except Exception as e:  # Catch other potential errors from the synchronous call
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Error during synchronous set_leverage call: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            raise  # Re-raise

        # If we reach here without raising an error, leverage setting was successful
        logger.success(
            f"{Fore.GREEN}Leverage successfully set to {CONFIG.DEFAULT_LEVERAGE}x for {CONFIG.SYMBOL}.{Style.RESET_ALL}"
        )

        # --- 5. (Optional) Set Margin Mode / Position Mode ---
        # Add similar calls here if needed, e.g.,
        # logger.info(f"Setting margin mode to {CONFIG.DEFAULT_MARGIN_MODE}...")
        # mode_set_ok = await bybit.set_margin_mode(exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_MARGIN_MODE, CONFIG) # Assuming async
        # if not mode_set_ok: raise ccxt.ExchangeError("Failed to set margin mode.")
        # logger.success(f"Margin mode set to {CONFIG.DEFAULT_MARGIN_MODE}.")

        # --- Mark Setup as Successful ---
        setup_success = True
        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}Setup phase completed successfully.{Style.RESET_ALL}"
        )

    # --- Catch Specific Setup Errors ---
    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Authentication Error during setup: {e}. Check API keys/permissions. Exiting.{Style.RESET_ALL}",
            exc_info=True,
        )
        if CONFIG and CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert("Strategy EXIT: Authentication error during setup.", CONFIG)
    except (
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ConnectionError,
        ValueError,
        TypeError,
    ) as e:
        # Catches failures from initialize, validate, set_leverage (if they raise properly or via our raise)
        # Also catches the specific TypeError we were originally debugging if it occurred here
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Exchange/Network/Configuration Error during setup: {e}. Exiting.{Style.RESET_ALL}",
            exc_info=True,
        )
        if CONFIG and CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"Strategy EXIT: Setup failed ({type(e).__name__}). Check logs.", CONFIG
            )
    except Exception as e:
        # Catch any other unexpected errors during setup
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Unexpected critical error during setup phase: {e}. Exiting.{Style.RESET_ALL}",
            exc_info=True,
        )
        if CONFIG and CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"Strategy EXIT: Unexpected critical error during setup ({type(e).__name__}). Check logs.",
                CONFIG,
            )

    # --- Run Strategy only if setup succeeded ---
    if setup_success and exchange and CONFIG:
        try:
            # Pass the validated exchange object and config to the main loop
            await run_strategy(CONFIG, exchange)
        except Exception as e:
            # Catch errors specifically from run_strategy if they aren't handled internally
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Strategy execution loop terminated abnormally: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            if CONFIG.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{CONFIG.SYMBOL}] CRITICAL FAILURE: Strategy loop terminated unexpectedly ({type(e).__name__}). Check logs!",
                    CONFIG,
                )
    elif not setup_success:
        logger.error(
            f"{Fore.RED}Strategy execution skipped due to setup failure.{Style.RESET_ALL}"
        )

    # --- Cleanup Phase: Close Exchange Connection ---
    # This block runs regardless of whether setup_success was True or False,
    # or if run_strategy finished normally or with an error,
    # ensuring the connection is closed if it was ever opened.
    logger.info(f"{Fore.CYAN}# Initiating cleanup phase...{Style.RESET_ALL}")
    if exchange and hasattr(exchange, "close") and callable(exchange.close):
        try:
            logger.info(
                f"{Fore.CYAN}# Closing connection to the exchange realm...{Style.RESET_ALL}"
            )
            await exchange.close()  # exchange.close() IS typically async in ccxt
            logger.info(
                f"{Fore.GREEN}Exchange connection closed gracefully.{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error occurred while closing the exchange connection: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
    else:
        logger.warning(
            f"{Fore.YELLOW}Exchange object was not initialized or has no close method; skipping explicit close.{Style.RESET_ALL}"
        )

    # Final exit status
    if not setup_success:
        logger.warning(
            f"{Fore.YELLOW}Exiting script with error status due to setup failure.{Style.RESET_ALL}"
        )
        sys.exit(1)  # Ensure exit code reflects failure if setup didn't complete

    logger.info(f"{Fore.MAGENTA}--- Strategy Shutdown Complete ---{Style.RESET_ALL}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Pyrmethus initiates the asynchronous ritual...
    try:
        # Setup asyncio event loop and run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully if it happens outside the main loop's catch
        print(
            f"\n{Fore.YELLOW}Asyncio execution interrupted by user (KeyboardInterrupt). Exiting.{Style.RESET_ALL}"
        )
        # Allow finally block in main() to attempt cleanup if it got that far
        sys.exit(0)  # Exit cleanly on user interrupt
    except Exception as e:
        # Catch errors during asyncio.run() itself if main() fails very early or setup fails catastrophically
        print(
            f"{Back.RED}{Fore.WHITE}Fatal error during asyncio setup or execution: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        # Attempt to log if logger was initialized, otherwise just print
        if logger:
            logger.critical(
                "Fatal error during asyncio setup or execution", exc_info=True
            )
        sys.exit(1)  # Exit with error status
