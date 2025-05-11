#!/usr/bin/env python

"""Enhanced Ehlers Volumetric Trend Strategy for Bybit V5 (v1.5 - Async Focus & Robustness)

Improves upon v1.4 by:
- Ensuring correct async/await usage, especially around potentially blocking calls.
- Refining error handling across setup, execution, and cleanup phases.
- Implementing robust position confirmation after entry before placing Stop Loss.
- Adding emergency position closing if Stop Loss placement fails after entry.
- Enhancing logging clarity with consistent formatting and color usage.
- Strengthening validation of market data, indicators, and order results.
- Improving Stop Loss and Position Size calculation precision and edge case handling.
- Managing Stop Loss order state reliably throughout the strategy lifecycle.
"""

import asyncio
import logging
import os
import sys
from decimal import (  # Import ROUND_UP for SL
    ROUND_DOWN,
    ROUND_UP,
    Decimal,
    InvalidOperation,
)

# Third-party libraries
import ccxt
import ccxt.async_support as ccxt_async  # Use async support explicitly
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
    # Ensure bybit_helpers has all required functions (async where appropriate)
    # Assume helpers are adapted for async where needed, explicitly mark assumptions
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
    err_back = Back.RED if COLORAMA_AVAILABLE else ""
    err_fore = Fore.WHITE if COLORAMA_AVAILABLE else ""
    warn_fore = Fore.YELLOW if COLORAMA_AVAILABLE else ""
    reset_all = Style.RESET_ALL if COLORAMA_AVAILABLE else ""
    print(f"{err_back}{err_fore}Error importing helper modules: {e}{reset_all}")
    print(
        f"{warn_fore}Ensure bybit_helpers.py, indicators.py, neon_logger.py, and bybit_utils.py are accessible, compatible, and contain necessary async functions.{reset_all}"
    )
    sys.exit(1)

# --- Load Environment Variables ---
load_dotenv()


# --- Configuration Class ---
class Config:
    """Holds all configuration parameters for the strategy."""

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
        self.SYMBOL: str = os.getenv("SYMBOL", "BTC/USDT:USDT")
        self.USDT_SYMBOL: str = "USDT"  # Base currency for balance checks
        self.EXPECTED_MARKET_TYPE: str = "swap"  # e.g., 'spot', 'swap'
        self.EXPECTED_MARKET_LOGIC: str = "linear"  # 'linear' or 'inverse'
        self.TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")
        self.OHLCV_LIMIT: int = int(
            os.getenv("OHLCV_LIMIT", 200)
        )  # Min candles for indicators

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
        )  # e.g., 0.01 = 1% risk

        # Order Settings
        self.DEFAULT_SLIPPAGE_PCT: Decimal = Decimal(
            os.getenv("DEFAULT_SLIPPAGE_PCT", "0.005")
        )  # Allowed slippage for market orders
        self.ORDER_BOOK_FETCH_LIMIT: int = 25  # Depth for slippage check
        self.SHALLOW_OB_FETCH_DEPTH: int = 5  # Less used here, but good to have

        # Fees (Note: Fetching fees dynamically via `fetchTradingFees` is preferred)
        self.TAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_TAKER_FEE", "0.00055"))
        self.MAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_MAKER_FEE", "0.0002"))

        # Strategy Parameters (Ehlers Volumetric Trend)
        self.EVT_ENABLED: bool = os.getenv("EVT_ENABLED", "true").lower() == "true"
        self.EVT_LENGTH: int = int(os.getenv("EVT_LENGTH", 7))
        self.EVT_MULTIPLIER: float = float(
            os.getenv("EVT_MULTIPLIER", 2.5)
        )  # Note: float, might be better as Decimal if precision critical in indicator
        self.STOP_LOSS_ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", 14))
        self.STOP_LOSS_ATR_MULTIPLIER: Decimal = Decimal(
            os.getenv("ATR_MULTIPLIER", "2.5")
        )

        # Retry & Timing
        self.RETRY_COUNT: int = int(os.getenv("RETRY_COUNT", 3))
        self.RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY", 2.0))
        self.LOOP_DELAY_SECONDS: int = int(
            os.getenv("LOOP_DELAY", 60)
        )  # Target time per loop cycle
        self.POSITION_CONFIRM_DELAY_SECONDS: int = int(
            os.getenv("POSITION_CONFIRM_DELAY", 7)
        )  # Wait after entry order before checking position
        self.POST_CLOSE_DELAY_SECONDS: int = int(
            os.getenv("POST_CLOSE_DELAY", 10)
        )  # Wait after closing position

        # Logging & Alerts
        self.LOG_CONSOLE_LEVEL: str = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
        self.LOG_FILE_LEVEL: str = os.getenv("LOG_FILE_LEVEL", "DEBUG").upper()
        self.LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "ehlers_strategy.log")
        self.ENABLE_SMS_ALERTS: bool = (
            os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"
        )
        self.SMS_RECIPIENT_NUMBER: str | None = os.getenv("SMS_RECIPIENT_NUMBER")
        self.SMS_TIMEOUT_SECONDS: int = 30
        self.TWILIO_ACCOUNT_SID: str | None = os.getenv("TWILIO_ACCOUNT_SID")
        self.TWILIO_AUTH_TOKEN: str | None = os.getenv("TWILIO_AUTH_TOKEN")
        self.TWILIO_PHONE_NUMBER: str | None = os.getenv("TWILIO_PHONE_NUMBER")

        # Constants
        self.SIDE_BUY: str = "buy"
        self.SIDE_SELL: str = "sell"
        self.POS_LONG: str = "LONG"  # Standardize case for position side reporting
        self.POS_SHORT: str = "SHORT"
        self.POS_NONE: str = "NONE"
        self.POSITION_QTY_EPSILON: Decimal = Decimal(
            "1e-9"
        )  # Small value to compare position quantities

        # --- Derived/Helper Attributes ---
        # Bundled for easier passing to indicator calculation
        self.indicator_settings = {
            "atr_period": self.STOP_LOSS_ATR_PERIOD,
            "evt_length": self.EVT_LENGTH,
            "evt_multiplier": self.EVT_MULTIPLIER,
        }
        self.analysis_flags = {
            "use_atr": True,  # Assumed required for SL calculation
            "use_evt": self.EVT_ENABLED,
        }
        self.strategy_params = {
            "ehlers_volumetric": {
                "evt_length": self.EVT_LENGTH,
                "evt_multiplier": self.EVT_MULTIPLIER,
            }
        }
        self.strategy = {
            "name": "ehlers_volumetric"
        }  # For potential use in shared functions

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
            self.ENABLE_SMS_ALERTS = False


# --- Global Variables ---
logger: logging.Logger | None = None
# Explicitly type hint async exchange object
exchange: ccxt_async.bybit | None = None
CONFIG: Config | None = None

# --- Core Strategy Functions ---


def calculate_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame | None:
    """Calculates indicators needed for the strategy (Synchronous, CPU-bound).
    Validates that required indicator columns are present in the output.
    Attempts fallback calculation for missing ATR if possible.
    """
    if logger is None:
        print("Logger not initialized in calculate_indicators", file=sys.stderr)
        return None
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
        # ASSUMPTION: ind.calculate_all_indicators is a synchronous function
        df_with_indicators = ind.calculate_all_indicators(df.copy(), indicator_config)

        if df_with_indicators is None or df_with_indicators.empty:
            logger.error(
                f"{Fore.RED}Indicator calculation returned empty or None DataFrame.{Style.RESET_ALL}"
            )
            return None

        # --- Validate required columns exist robustly ---
        required_cols = []
        if config.EVT_ENABLED:
            required_cols.extend(
                [
                    f"evt_trend_{config.EVT_LENGTH}",
                    f"evt_buy_{config.EVT_LENGTH}",
                    f"evt_sell_{config.EVT_LENGTH}",
                ]
            )
        if config.analysis_flags.get("use_atr", False):  # Check if ATR is expected
            required_cols.append(f"ATRr_{config.STOP_LOSS_ATR_PERIOD}")

        missing_cols = [
            col for col in required_cols if col not in df_with_indicators.columns
        ]

        # --- Attempt Fallback Calculation for Missing ATR ---
        atr_col_name = f"ATRr_{config.STOP_LOSS_ATR_PERIOD}"
        if atr_col_name in missing_cols and config.analysis_flags.get("use_atr", False):
            logger.warning(
                f"{Fore.YELLOW}ATR column '{atr_col_name}' missing. Attempting fallback calculation...{Style.RESET_ALL}"
            )
            if all(c in df_with_indicators for c in ["high", "low", "close"]):
                try:
                    # Assuming 'ta' library is used within indicators.py or available here
                    import pandas_ta as ta  # Ensure pandas_ta is available if used directly

                    atr_result = df_with_indicators.ta.atr(
                        length=config.STOP_LOSS_ATR_PERIOD, append=False
                    )
                    if atr_result is not None and not atr_result.empty:
                        df_with_indicators[atr_result.name] = atr_result
                        logger.info(
                            f"{Fore.CYAN}Calculated missing ATR column '{atr_result.name}' using pandas_ta.{Style.RESET_ALL}"
                        )
                        missing_cols.remove(
                            atr_col_name
                        )  # Remove if successfully added
                    else:
                        logger.error(
                            f"{Fore.RED}Fallback ATR calculation failed or returned empty.{Style.RESET_ALL}"
                        )
                except ImportError:
                    logger.error(
                        f"{Fore.RED}Cannot perform fallback ATR calculation: 'pandas_ta' not installed. Run 'pip install pandas_ta'.{Style.RESET_ALL}"
                    )
                except Exception as atr_err:
                    logger.error(
                        f"{Fore.RED}Error during fallback ATR calculation: {atr_err}{Style.RESET_ALL}",
                        exc_info=True,
                    )
            else:
                logger.error(
                    f"{Fore.RED}Cannot calculate fallback ATR: Missing 'high', 'low', or 'close' columns.{Style.RESET_ALL}"
                )

        # --- Final Check for Missing Columns ---
        if missing_cols:  # Check again after potential fallback
            logger.error(
                f"{Fore.RED}Required indicator columns remain missing after calculation: {', '.join(missing_cols)}. Cannot proceed with signal generation or SL calculation.{Style.RESET_ALL}"
            )
            return None

        logger.debug(
            f"Indicators calculated. DataFrame shape: {df_with_indicators.shape}. Columns: {', '.join(df_with_indicators.columns)}"
        )
        return df_with_indicators
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error calculating indicators: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


def generate_signals(df_ind: pd.DataFrame, config: Config) -> str | None:
    """Generates trading signals ('buy', 'sell', or None) based on the last row (Synchronous).
    Checks for NaN values and conflicting signals.
    """
    if logger is None:
        print("Logger not initialized in generate_signals", file=sys.stderr)
        return None
    if df_ind is None or df_ind.empty:
        logger.debug("Cannot generate signals: Input DataFrame is None or empty.")
        return None
    if not config.EVT_ENABLED:
        logger.debug("EVT indicator is disabled, no signals generated.")
        return None

    try:
        if len(df_ind) == 0:
            logger.warning(
                f"{Fore.YELLOW}Cannot generate signals: DataFrame has 0 rows after indicator calculation.{Style.RESET_ALL}"
            )
            return None

        latest = df_ind.iloc[-1]
        trend_col = f"evt_trend_{config.EVT_LENGTH}"
        buy_col = f"evt_buy_{config.EVT_LENGTH}"
        sell_col = f"evt_sell_{config.EVT_LENGTH}"

        # Check required columns for EVT exist in the latest row's index
        required_evt_cols = [trend_col, buy_col, sell_col]
        if not all(col in latest.index for col in required_evt_cols):
            missing_evt_cols = [c for c in required_evt_cols if c not in latest.index]
            logger.warning(
                f"{Fore.YELLOW}EVT signal columns missing ({', '.join(missing_evt_cols)}) in latest data row (Index: {latest.name}). Cannot generate EVT signal.{Style.RESET_ALL}"
            )
            return None

        # Check for NaN values which can cause issues
        if any(pd.isna(latest[col]) for col in required_evt_cols):
            trend_val = latest.get(trend_col, "NaN")
            buy_val = latest.get(buy_col, "NaN")
            sell_val = latest.get(sell_col, "NaN")
            logger.debug(
                f"Latest EVT data contains NaN values. Trend={trend_val}, Buy={buy_val}, Sell={sell_val}. No signal."
            )
            return None

        # Assuming trend is numerical (-1, 0, 1) and buy/sell are boolean or 0/1
        try:
            trend = int(latest[trend_col])
            buy_signal = bool(latest[buy_col])
            sell_signal = bool(latest[sell_col])
        except (ValueError, TypeError) as conv_err:
            logger.error(
                f"{Fore.RED}Error converting EVT signal data to expected types: {conv_err}. Data: Trend={latest.get(trend_col)}, Buy={latest.get(buy_col)}, Sell={latest.get(sell_col)}.{Style.RESET_ALL}"
            )
            return None

        logger.debug(
            f"Signal Check: Index={latest.name}, Close={latest.get('close', 'N/A'):.4f}, "
            f"{trend_col}={trend}, {buy_col}={buy_signal}, {sell_col}={sell_signal}"
        )

        # Refined logic: Ensure only one signal is active to avoid ambiguity
        if buy_signal and not sell_signal:  # Explicit buy signal
            # Optional: Add trend confirmation: if trend == 1: ...
            logger.info(
                f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}"
            )
            return config.SIDE_BUY
        elif sell_signal and not buy_signal:  # Explicit sell signal
            # Optional: Add trend confirmation: if trend == -1: ...
            logger.info(
                f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}"
            )
            return config.SIDE_SELL
        elif buy_signal and sell_signal:
            logger.warning(
                f"{Fore.YELLOW}Conflicting signals: Both Buy and Sell flags are active in latest data. No signal generated.{Style.RESET_ALL}"
            )
            return None
        else:
            # Neither buy nor sell flag is active
            logger.debug("No active Buy or Sell flag in latest EVT data.")
            return None

    except IndexError:
        logger.warning(
            f"{Fore.YELLOW}Could not access latest indicator data (IndexError), DataFrame might be too short ({len(df_ind)} rows).{Style.RESET_ALL}"
        )
        return None
    except (
        KeyError,
        ValueError,
        TypeError,
    ) as e:  # Catch potential issues if columns exist but data is wrong type
        logger.error(
            f"{Fore.RED}Error processing latest data for signal generation: {e}. Data: {latest.to_dict()}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}Unexpected error generating signals: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


def calculate_stop_loss(
    current_exchange: ccxt_async.bybit,
    df_ind: pd.DataFrame,
    side: str,
    entry_price: Decimal,
    config: Config,
) -> Decimal | None:
    """Calculates the initial stop-loss price based on ATR (Synchronous).
    Uses market precision (tick size) for accurate formatting.
    Includes sanity checks to prevent SL from being on the wrong side of entry.
    """
    if logger is None:
        print("Logger not initialized in calculate_stop_loss", file=sys.stderr)
        return None
    if current_exchange is None:
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
        return None  # Or implement alternative SL logic (e.g., fixed percentage)

    try:
        market = current_exchange.market(config.SYMBOL)
        if not market:
            logger.error(
                f"{Fore.RED}Cannot calculate stop-loss: Market details for {config.SYMBOL} not loaded.{Style.RESET_ALL}"
            )
            return None

        # Determine tick size for price formatting
        price_precision_str = market.get("precision", {}).get("price")
        if price_precision_str is None:
            logger.warning(
                f"{Fore.YELLOW}Could not determine price precision (tick size) for {config.SYMBOL}. Using fallback 1e-8. SL accuracy might be affected.{Style.RESET_ALL}"
            )
            tick_size = Decimal("1e-8")
        else:
            # Precision might be number of decimal places or the tick size itself
            try:
                # Handles '0.01' style precision
                tick_size = Decimal(str(price_precision_str))
            except InvalidOperation:
                try:
                    # Handles integer style precision (number of decimal places)
                    tick_size = Decimal("1") / (
                        Decimal("10") ** int(price_precision_str)
                    )
                except (ValueError, TypeError, InvalidOperation):
                    logger.warning(
                        f"{Fore.YELLOW}Could not parse price precision '{price_precision_str}' for {config.SYMBOL}. Using fallback 1e-8.{Style.RESET_ALL}"
                    )
                    tick_size = Decimal("1e-8")

        atr_col = f"ATRr_{config.STOP_LOSS_ATR_PERIOD}"
        if atr_col not in df_ind.columns:
            logger.error(
                f"{Fore.RED}ATR column '{atr_col}' not found for stop-loss calculation.{Style.RESET_ALL}"
            )
            return None

        if len(df_ind) == 0:
            logger.warning(
                f"{Fore.YELLOW}Cannot calculate SL: Indicator DataFrame has 0 rows.{Style.RESET_ALL}"
            )
            return None

        latest_atr_raw = df_ind.iloc[-1][atr_col]
        if pd.isna(latest_atr_raw):
            logger.warning(
                f"{Fore.YELLOW}Latest ATR value is NaN (Index: {df_ind.index[-1]}), cannot calculate stop-loss.{Style.RESET_ALL}"
            )
            return None

        latest_atr = safe_decimal_conversion(latest_atr_raw)
        if latest_atr is None or latest_atr <= Decimal(0):
            logger.warning(
                f"{Fore.YELLOW}Invalid ATR value ({latest_atr_raw} -> {latest_atr}), cannot calculate stop-loss accurately.{Style.RESET_ALL}"
            )
            return None

        stop_offset = latest_atr * config.STOP_LOSS_ATR_MULTIPLIER
        stop_loss_price_raw = (
            entry_price - stop_offset
            if side == config.SIDE_BUY
            else entry_price + stop_offset
        )

        # --- Format price according to market precision (tick size) ---
        # Use quantize for correct rounding based on tick size.
        # Round "away" from the entry price for safety.
        rounding_mode = ROUND_DOWN if side == config.SIDE_BUY else ROUND_UP
        stop_loss_price = stop_loss_price_raw.quantize(
            tick_size, rounding=rounding_mode
        )

        # --- Sanity checks AFTER precise formatting ---
        if side == config.SIDE_BUY:
            if stop_loss_price >= entry_price:
                logger.warning(
                    f"{Fore.YELLOW}Calculated Buy SL ({format_price(current_exchange, config.SYMBOL, stop_loss_price)}) is >= Entry ({format_price(current_exchange, config.SYMBOL, entry_price)}). Adjusting one tick lower.{Style.RESET_ALL}"
                )
                stop_loss_price = (stop_loss_price - tick_size).quantize(
                    tick_size, rounding=ROUND_DOWN
                )
            # Ensure it didn't become negative after adjustment
            if stop_loss_price <= Decimal(0):
                logger.error(
                    f"{Fore.RED}Adjusted Buy SL resulted in zero or negative price ({stop_loss_price}). Cannot set SL.{Style.RESET_ALL}"
                )
                return None
        elif side == config.SIDE_SELL:
            if stop_loss_price <= entry_price:
                logger.warning(
                    f"{Fore.YELLOW}Calculated Sell SL ({format_price(current_exchange, config.SYMBOL, stop_loss_price)}) is <= Entry ({format_price(current_exchange, config.SYMBOL, entry_price)}). Adjusting one tick higher.{Style.RESET_ALL}"
                )
                stop_loss_price = (stop_loss_price + tick_size).quantize(
                    tick_size, rounding=ROUND_UP
                )

        logger.info(
            f"Calculated SL for {side.upper()} at {format_price(current_exchange, config.SYMBOL, stop_loss_price)} "
            f"(Entry: {format_price(current_exchange, config.SYMBOL, entry_price)}, ATR: {latest_atr:.5f}, Mult: {config.STOP_LOSS_ATR_MULTIPLIER}, Tick: {tick_size})"
        )
        return stop_loss_price  # Return the precise Decimal value

    except IndexError:
        logger.warning(
            f"{Fore.YELLOW}Could not access latest indicator data (IndexError) for SL calculation.{Style.RESET_ALL}"
        )
        return None
    except (KeyError, ValueError, TypeError, InvalidOperation) as e:
        logger.error(
            f"{Fore.RED}Error processing data or market info for SL calculation: {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}Unexpected error calculating stop-loss: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


async def calculate_position_size(
    current_exchange: ccxt_async.bybit,
    symbol: str,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    config: Config,
) -> Decimal | None:
    """Calculates position size based on risk percentage and stop-loss distance.
    Fetches available balance asynchronously.
    Adjusts size based on market quantity precision (step size) and limits.
    """
    if logger is None:
        print("Logger not initialized in calculate_position_size", file=sys.stderr)
        return None
    if not current_exchange:
        logger.error(
            f"{Fore.RED}Cannot calculate position size: Exchange object is None.{Style.RESET_ALL}"
        )
        return None

    try:
        # --- Fetch Available Balance (Asynchronous) ---
        logger.debug(
            f"{Fore.CYAN}# Awaiting {config.USDT_SYMBOL} balance for position sizing...{Style.RESET_ALL}"
        )
        # ASSUMPTION: bybit.fetch_usdt_balance IS ASYNC and returns (total, available) Decimals or Nones
        total_balance, available_balance = await bybit.fetch_usdt_balance(
            current_exchange, config
        )

        if available_balance is None or available_balance <= Decimal("0"):
            logger.error(
                f"{Fore.RED}Cannot calculate position size: Zero, None, or invalid available balance ({available_balance}).{Style.RESET_ALL}"
            )
            return None
        logger.debug(
            f"Available balance for sizing: {available_balance:.4f} {config.USDT_SYMBOL}"
        )

        # --- Calculate Risk ---
        risk_amount_usd = available_balance * config.RISK_PER_TRADE
        price_diff = abs(entry_price - stop_loss_price)

        # Use epsilon for safety, especially if entry/SL calculation had rounding
        if price_diff <= config.POSITION_QTY_EPSILON:
            logger.error(
                f"{Fore.RED}Cannot calculate position size: Entry price ({format_price(current_exchange, symbol, entry_price)}) and SL price ({format_price(current_exchange, symbol, stop_loss_price)}) are identical or too close (Diff: {price_diff}).{Style.RESET_ALL}"
            )
            return None

        # Position size in base currency = (Amount willing to risk in quote) / (Risk per unit in quote)
        position_size_base = risk_amount_usd / price_diff

        # --- Get Market Limits & Precision (Step Size) ---
        market = current_exchange.market(symbol)
        if not market:
            logger.error(
                f"{Fore.RED}Market {symbol} not loaded, cannot get limits/precision for position size.{Style.RESET_ALL}"
            )
            return None

        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        precision = market.get("precision", {})

        min_qty_str = amount_limits.get("min")
        max_qty_str = amount_limits.get("max")
        # Amount precision (step size) can be tricky - might be number of digits or step value string
        qty_precision_value = precision.get(
            "amount"
        )  # Can be int (digits) or float/str ('0.001')

        # --- Determine Step Size (amount precision) Robustly ---
        step_size: Decimal | None = None
        if qty_precision_value is not None:
            try:
                # Handles numeric step size directly (e.g., 0.001)
                step_size = Decimal(str(qty_precision_value))
                if step_size <= 0:
                    logger.warning(
                        f"{Fore.YELLOW}Market precision 'amount' is zero or negative ({qty_precision_value}). Using fallback 1e-8.{Style.RESET_ALL}"
                    )
                    step_size = Decimal("1e-8")
            except (InvalidOperation, ValueError, TypeError):
                try:
                    # Assume it's number of decimal places (integer)
                    num_digits = int(qty_precision_value)
                    if num_digits == 0:
                        step_size = Decimal("1")  # Whole numbers
                    elif num_digits > 0:
                        step_size = Decimal("1") / (Decimal("10") ** num_digits)
                    else:  # Negative precision doesn't make sense for amount steps
                        logger.warning(
                            f"{Fore.YELLOW}Market precision 'amount' is negative ({num_digits}). Using fallback 1e-8.{Style.RESET_ALL}"
                        )
                        step_size = Decimal("1e-8")
                except (ValueError, TypeError):
                    logger.warning(
                        f"{Fore.YELLOW}Could not determine valid quantity step size from precision '{qty_precision_value}' for {symbol}. Using fallback 1e-8.{Style.RESET_ALL}"
                    )
                    step_size = Decimal("1e-8")
        else:
            logger.warning(
                f"{Fore.YELLOW}Could not determine quantity precision/step size for {symbol}. Using fallback 1e-8.{Style.RESET_ALL}"
            )
            step_size = Decimal("1e-8")  # Fallback step size

        # --- Adjust position size down to the nearest valid step size ---
        position_size_adjusted = position_size_base.quantize(
            step_size, rounding=ROUND_DOWN
        )

        if position_size_adjusted <= Decimal(0):
            logger.warning(
                f"{Fore.YELLOW}Calculated position size is zero or negative ({position_size_adjusted}) after adjusting for step size {step_size}. "
                f"Original Raw: {position_size_base:.8f}, Risk Amt: {risk_amount_usd:.4f}, Price Diff: {price_diff:.8f}{Style.RESET_ALL}"
            )
            return None

        # --- Check Against Min/Max Limits ---
        min_qty = safe_decimal_conversion(min_qty_str, default=None)
        if min_qty is not None and position_size_adjusted < min_qty:
            logger.warning(
                f"{Fore.YELLOW}Calculated position size ({format_amount(current_exchange, symbol, position_size_adjusted)}) is below exchange minimum ({format_amount(current_exchange, symbol, min_qty)}). Cannot place trade.{Style.RESET_ALL}"
            )
            return None

        max_qty = safe_decimal_conversion(max_qty_str, default=None)
        if max_qty is not None and position_size_adjusted > max_qty:
            logger.warning(
                f"{Fore.YELLOW}Calculated position size ({format_amount(current_exchange, symbol, position_size_adjusted)}) exceeds exchange maximum ({format_amount(current_exchange, symbol, max_qty)}). Capping at maximum.{Style.RESET_ALL}"
            )
            # Adjust max qty to step size too, just in case max isn't a multiple of step
            position_size_adjusted = max_qty.quantize(step_size, rounding=ROUND_DOWN)
            # Re-check if capping made it too small or zero
            if position_size_adjusted <= Decimal(0) or (
                min_qty is not None and position_size_adjusted < min_qty
            ):
                logger.error(
                    f"{Fore.RED}Position size after capping at max ({format_amount(current_exchange, symbol, position_size_adjusted)}) became invalid (zero or below min). Cannot place trade.{Style.RESET_ALL}"
                )
                return None

        # --- Optional: Check Available Margin (Simplified) ---
        # This is a rough check. Real margin calculations are complex (initial vs maintenance, cross vs isolated).
        # cost_estimate = position_size_adjusted * entry_price / config.DEFAULT_LEVERAGE
        # if cost_estimate > available_balance:
        #     logger.warning(f"{Fore.YELLOW}Estimated position cost (~{cost_estimate:.2f} {config.USDT_SYMBOL}) may exceed available balance ({available_balance:.2f} {config.USDT_SYMBOL}) with {config.DEFAULT_LEVERAGE}x leverage. Position size may be too large.{Style.RESET_ALL}")
        # Consider reducing size further or returning None
        # return None # Safer option if estimated cost exceeds available balance

        base_currency = symbol.split("/")[0]
        logger.info(
            f"Calculated position size: {format_amount(current_exchange, symbol, position_size_adjusted)} {base_currency} "
            f"(Risk: {risk_amount_usd:.2f} {config.USDT_SYMBOL}, Balance: {available_balance:.2f} {config.USDT_SYMBOL}, Price Diff: {price_diff:.5f}, Step: {step_size})"
        )
        return position_size_adjusted

    except ccxt.NetworkError as e:
        logger.warning(
            f"{Fore.YELLOW}Network error during position size calculation (fetching balance?): {e}{Style.RESET_ALL}"
        )
        return None
    except ccxt.ExchangeError as e:
        logger.error(
            f"{Fore.RED}Exchange error during position size calculation (fetching balance?): {e}{Style.RESET_ALL}"
        )
        return None
    except (KeyError, ValueError, TypeError, InvalidOperation) as e:
        logger.error(
            f"{Fore.RED}Error processing data/market info during position size calculation: {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}Unexpected error calculating position size: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- Main Strategy Loop ---
async def run_strategy(config: Config, current_exchange: ccxt_async.bybit):
    """Main asynchronous trading loop."""
    global logger  # Use global logger implicitly
    if logger is None:
        print("Logger not initialized in run_strategy", file=sys.stderr)
        return
    if not current_exchange:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Strategy cannot run: Invalid exchange object received.{Style.RESET_ALL}"
        )
        return

    logger.info(
        f"{Fore.MAGENTA}{Style.BRIGHT}--- Starting Ehlers Volumetric Strategy Loop for {config.SYMBOL} on {config.TIMEFRAME} ---{Style.RESET_ALL}"
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
    if config.ENABLE_SMS_ALERTS:
        logger.info(f"SMS Alerts: ENABLED for {config.SMS_RECIPIENT_NUMBER}")

    # Persists across loops to track the SL order associated with the current position for the symbol
    # Key: Symbol (e.g., "BTC/USDT:USDT"), Value: Order ID (string)
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
                f"{Fore.CYAN}# Awaiting current position state for {config.SYMBOL}...{Style.RESET_ALL}"
            )
            # ASSUMPTION: get_current_position_bybit_v5 IS ASYNC and returns dict or None
            current_position = await bybit.get_current_position_bybit_v5(
                current_exchange, config.SYMBOL, config
            )
            if current_position is None:
                logger.warning(
                    f"{Fore.YELLOW}Failed to get current position state for {config.SYMBOL}. Retrying next cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue

            # Extract position details (POS_LONG, POS_SHORT, POS_NONE)
            current_side = current_position.get(
                "side", config.POS_NONE
            )  # Default to NONE if missing
            current_qty = current_position.get(
                "qty", Decimal(0)
            )  # Decimal, default to 0
            entry_price = current_position.get("entry_price")  # Decimal or None
            logger.info(
                f"Current Position ({config.SYMBOL}): Side={current_side}, "
                f"Qty={format_amount(current_exchange, config.SYMBOL, current_qty)}, "
                f"Entry={format_price(current_exchange, config.SYMBOL, entry_price) if entry_price else 'N/A'}"
            )

            # --- 2. Fetch Data & Calculate Indicators (Requires async fetches) ---
            logger.debug(
                f"{Fore.CYAN}# Awaiting OHLCV data for {config.SYMBOL}...{Style.RESET_ALL}"
            )
            # ASSUMPTION: fetch_ohlcv_paginated IS ASYNC
            ohlcv_df = await bybit.fetch_ohlcv_paginated(
                current_exchange,
                config.SYMBOL,
                config.TIMEFRAME,
                limit_per_req=1000,  # Bybit V5 default/max
                max_total_candles=config.OHLCV_LIMIT
                + 50,  # Fetch slightly more for indicator stability
                config=config,
            )
            # Robust check for sufficient data
            min_required_candles = max(
                config.OHLCV_LIMIT,
                config.STOP_LOSS_ATR_PERIOD + 1,
                config.EVT_LENGTH + 1,
            )
            if (
                ohlcv_df is None
                or ohlcv_df.empty
                or len(ohlcv_df) < min_required_candles
            ):
                logger.warning(
                    f"{Fore.YELLOW}Could not fetch sufficient OHLCV data for {config.SYMBOL} "
                    f"({len(ohlcv_df) if ohlcv_df is not None else 0} candles, needed ~{min_required_candles}). Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue

            logger.debug(
                f"{Fore.CYAN}# Awaiting ticker data for {config.SYMBOL}...{Style.RESET_ALL}"
            )
            # ASSUMPTION: fetch_ticker_validated IS ASYNC
            ticker = await bybit.fetch_ticker_validated(
                current_exchange, config.SYMBOL, config
            )
            if ticker is None or ticker.get("last") is None:
                logger.warning(
                    f"{Fore.YELLOW}Could not fetch valid ticker data for {config.SYMBOL}. Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue
            current_price = safe_decimal_conversion(ticker["last"])
            if current_price is None:
                logger.error(
                    f"{Fore.RED}Could not convert ticker price '{ticker['last']}' to Decimal for {config.SYMBOL}. Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue
            logger.debug(
                f"Current Price ({config.SYMBOL}): {format_price(current_exchange, config.SYMBOL, current_price)}"
            )

            # Indicator calculation is synchronous (CPU-bound), run it directly
            df_with_indicators = calculate_indicators(ohlcv_df, config)
            if df_with_indicators is None:
                logger.warning(
                    f"{Fore.YELLOW}Failed to calculate indicators for {config.SYMBOL}. Skipping cycle.{Style.RESET_ALL}"
                )
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue

            # --- 3. Generate Trading Signal (Synchronous) ---
            signal = generate_signals(df_with_indicators, config)
            logger.info(
                f"Generated Signal ({config.SYMBOL}): {signal if signal else 'None'}"
            )

            # --- 4. Handle Exits (Requires async order calls) ---
            # Only process exits if we are currently in a position
            if (
                current_side != config.POS_NONE
                and current_qty > config.POSITION_QTY_EPSILON
            ):
                should_exit = False
                exit_reason = ""

                # --- Define Exit Conditions ---
                # Exit Condition 1: EVT Trend Reversal Signal
                if config.EVT_ENABLED:
                    try:
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
                                f"{Fore.YELLOW}Latest EVT trend is NaN for {config.SYMBOL}, cannot use for exit signal.{Style.RESET_ALL}"
                            )
                    except (IndexError, KeyError, ValueError, TypeError) as trend_err:
                        logger.warning(
                            f"{Fore.YELLOW}Could not evaluate EVT trend for exit condition: {trend_err}.{Style.RESET_ALL}"
                        )

                # Exit Condition 2: Opposite EVT Signal (e.g., Long position sees a SELL signal)
                if (
                    not should_exit and signal
                ):  # Only check if not already exiting based on trend
                    if (
                        current_side == config.POS_LONG
                        and signal == config.SIDE_SELL
                        or current_side == config.POS_SHORT
                        and signal == config.SIDE_BUY
                    ):
                        should_exit = True
                        exit_reason = f"Opposing EVT Signal ({signal.upper()})"

                # --- Add other exit conditions here ---
                # Example: Take Profit (requires fetching order book or using TP orders)
                # Example: Trailing Stop Loss (requires tracking SL and updating it)

                # --- Execute Exit if Triggered ---
                if should_exit:
                    logger.warning(
                        f"{Fore.YELLOW}{Style.BRIGHT}Exit condition met for {current_side} {config.SYMBOL} position: {exit_reason}. Attempting to close.{Style.RESET_ALL}"
                    )

                    # --- Cancel Existing SL Order FIRST ---
                    sl_order_id = stop_loss_orders.pop(
                        config.SYMBOL, None
                    )  # Get and remove ID if found
                    if sl_order_id:
                        try:
                            logger.info(
                                f"Attempting to cancel existing SL order {format_order_id(sl_order_id)} for {config.SYMBOL} before closing position..."
                            )
                            logger.debug(
                                f"{Fore.CYAN}# Awaiting cancellation of SL order {sl_order_id}...{Style.RESET_ALL}"
                            )
                            # ASSUMPTION: cancel_order IS async
                            cancelled = await bybit.cancel_order(
                                current_exchange,
                                config.SYMBOL,
                                sl_order_id,
                                config=config,
                            )
                            if cancelled:
                                logger.info(
                                    f"{Fore.GREEN}Successfully cancelled SL order {format_order_id(sl_order_id)} for {config.SYMBOL}.{Style.RESET_ALL}"
                                )
                            else:
                                # This might happen if the order was already filled/cancelled. Log as warning.
                                logger.warning(
                                    f"{Fore.YELLOW}Attempt to cancel SL order {format_order_id(sl_order_id)} returned non-True or None. It might have already executed or been cancelled.{Style.RESET_ALL}"
                                )
                        except ccxt.OrderNotFound:
                            logger.warning(
                                f"{Fore.YELLOW}SL order {format_order_id(sl_order_id)} not found when trying to cancel (already filled/cancelled?). Proceeding with position close.{Style.RESET_ALL}"
                            )
                        except ccxt.NetworkError as e:
                            logger.warning(
                                f"{Fore.YELLOW}Network error cancelling SL order {format_order_id(sl_order_id)}: {e}. Proceeding with position close.{Style.RESET_ALL}"
                            )
                        except ccxt.ExchangeError as e:
                            # Log error but proceed with closing the position - better to close than be stuck.
                            logger.error(
                                f"{Fore.RED}Exchange error cancelling SL order {format_order_id(sl_order_id)}: {e}. Proceeding with position close with caution.{Style.RESET_ALL}"
                            )
                        except Exception as e:
                            logger.error(
                                f"{Fore.RED}Unexpected error cancelling SL order {format_order_id(sl_order_id)}: {e}{Style.RESET_ALL}",
                                exc_info=True,
                            )
                            # Proceeding is usually better than being stuck.
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}No tracked SL order ID found to cancel for the existing {current_side} {config.SYMBOL} position. Ensure SL was placed and tracked correctly.{Style.RESET_ALL}"
                        )

                    # --- Close the Position (Market Order, Reduce Only) ---
                    logger.info(
                        f"Attempting to close {current_side} position ({format_amount(current_exchange, config.SYMBOL, current_qty)} {config.SYMBOL}) via market order..."
                    )
                    logger.debug(
                        f"{Fore.CYAN}# Awaiting position close market order for {config.SYMBOL}...{Style.RESET_ALL}"
                    )
                    # ASSUMPTION: close_position_reduce_only IS async and returns order dict or None
                    close_order = await bybit.close_position_reduce_only(
                        current_exchange,
                        config.SYMBOL,
                        config,
                        position_to_close=current_position,  # Pass full details if helper needs them
                        reason=exit_reason,
                    )

                    if close_order and close_order.get("id"):
                        close_id_short = format_order_id(close_order["id"])
                        # Try to get actual filled qty/price from order, fallback to initial qty
                        filled_qty_close = safe_decimal_conversion(
                            close_order.get("filled", current_qty)
                        )
                        avg_close_price = safe_decimal_conversion(
                            close_order.get("average")
                        )
                        close_price_formatted = (
                            format_price(
                                current_exchange, config.SYMBOL, avg_close_price
                            )
                            if avg_close_price
                            else "N/A"
                        )

                        logger.success(
                            f"{Fore.GREEN}{Style.BRIGHT}Position ({config.SYMBOL}) CLOSE order {close_id_short} submitted. "
                            f"Qty: ~{format_amount(current_exchange, config.SYMBOL, filled_qty_close)}, AvgPrice: {close_price_formatted}. Reason: {exit_reason}.{Style.RESET_ALL}"
                        )

                        if config.ENABLE_SMS_ALERTS:
                            alert_msg = f"[{config.SYMBOL}] {current_side} Pos Closed (~{format_amount(current_exchange, config.SYMBOL, filled_qty_close)} @ {close_price_formatted}). Reason: {exit_reason}"
                            send_sms_alert(alert_msg, config)

                        # Optional: Short delay to allow position update propagation before next cycle checks
                        logger.debug(
                            f"Waiting {config.POST_CLOSE_DELAY_SECONDS}s after closing position..."
                        )
                        await asyncio.sleep(config.POST_CLOSE_DELAY_SECONDS)
                        continue  # Skip entry logic for this cycle as we just exited

                    else:
                        # Log critical error and potentially alert - Failure to submit close order is serious.
                        logger.critical(
                            f"{Back.RED}{Fore.WHITE}CRITICAL: FAILED TO SUBMIT POSITION CLOSE ORDER for {config.SYMBOL} ({current_side}, {format_amount(current_exchange, config.SYMBOL, current_qty)})! Reason: {exit_reason}. Manual intervention likely required!{Style.RESET_ALL}"
                        )
                        if config.ENABLE_SMS_ALERTS:
                            alert_msg = f"[{config.SYMBOL}] URGENT: Failed submit close order for {current_side} pos ({format_amount(current_exchange, config.SYMBOL, current_qty)})! Reason: {exit_reason}! Check account!"
                            send_sms_alert(alert_msg, config)
                        # Consider halting strategy or specific symbol trading here
                        # Wait longer before next cycle attempt after critical failure
                        await asyncio.sleep(config.LOOP_DELAY_SECONDS * 3)
                        continue  # Skip entry logic

            # --- 5. Handle Entries (Requires async order calls) ---
            # Only enter if we are flat (no position) and there is a valid signal
            elif current_side == config.POS_NONE and signal:
                logger.info(
                    f"{Fore.CYAN}{Style.BRIGHT}Attempting to enter {signal.upper()} position for {config.SYMBOL} based on signal...{Style.RESET_ALL}"
                )

                # --- Pre-entry Cleanup: Cancel any existing orders for the symbol ---
                logger.debug(
                    f"{Fore.CYAN}# Awaiting pre-entry order cleanup for {config.SYMBOL}...{Style.RESET_ALL}"
                )
                # ASSUMPTION: cancel_all_orders IS async and returns count or None
                cancelled_count = await bybit.cancel_all_orders(
                    current_exchange, config.SYMBOL, config, reason="Pre-Entry Cleanup"
                )
                if cancelled_count is not None and cancelled_count > 0:
                    logger.info(
                        f"Pre-entry cleanup ({config.SYMBOL}): Cancelled {cancelled_count} potential lingering order(s)."
                    )
                elif cancelled_count == 0:
                    logger.info(
                        f"Pre-entry cleanup ({config.SYMBOL}): No lingering orders found."
                    )
                # If cancelled_count is None, helper likely logged an error, proceed with caution

                # --- Calculate Stop Loss (Synchronous, uses latest data) ---
                # Use current_price as the estimated entry for initial SL calculation
                stop_loss_price = calculate_stop_loss(
                    current_exchange, df_with_indicators, signal, current_price, config
                )
                if (
                    not stop_loss_price or stop_loss_price <= 0
                ):  # Check if None or invalid (zero/negative)
                    logger.error(
                        f"{Fore.RED}Could not calculate a valid stop-loss for {config.SYMBOL} based on current data (Result: {stop_loss_price}). Cannot enter trade.{Style.RESET_ALL}"
                    )
                    await asyncio.sleep(
                        config.LOOP_DELAY_SECONDS
                    )  # Wait before retrying cycle
                    continue

                # --- Calculate Position Size (Asynchronous, fetches balance) ---
                logger.debug(
                    f"{Fore.CYAN}# Awaiting position size calculation for {config.SYMBOL}...{Style.RESET_ALL}"
                )
                position_size = await calculate_position_size(
                    current_exchange,
                    config.SYMBOL,
                    current_price,
                    stop_loss_price,
                    config,
                )
                if (
                    not position_size or position_size <= 0
                ):  # Check if None or zero/negative
                    logger.error(
                        f"{Fore.RED}Could not calculate a valid position size for {config.SYMBOL} (Result: {position_size}). Cannot enter trade.{Style.RESET_ALL}"
                    )
                    await asyncio.sleep(
                        config.LOOP_DELAY_SECONDS
                    )  # Wait before retrying cycle
                    continue

                # --- Place Market Entry Order (Asynchronous) ---
                base_currency = config.SYMBOL.split("/")[0]
                logger.info(
                    f"Placing {signal.upper()} market entry order for {format_amount(current_exchange, config.SYMBOL, position_size)} {base_currency}..."
                )
                logger.debug(
                    f"{Fore.CYAN}# Awaiting market entry order placement for {config.SYMBOL}...{Style.RESET_ALL}"
                )
                # ASSUMPTION: place_market_order_slippage_check IS async and returns order dict or None
                entry_order = await bybit.place_market_order_slippage_check(
                    current_exchange, config.SYMBOL, signal, position_size, config
                )

                if entry_order and entry_order.get("id"):
                    order_id_short = format_order_id(entry_order["id"])
                    # Log preliminary info from the submitted order
                    avg_fill_price_order = safe_decimal_conversion(
                        entry_order.get("average")
                    )  # Might be None immediately
                    filled_qty_order = safe_decimal_conversion(
                        entry_order.get("filled", 0)
                    )  # Might be 0 immediately

                    logger.success(
                        f"{Fore.GREEN}Entry market order {order_id_short} submitted for {config.SYMBOL} ({signal.upper()} {format_amount(current_exchange, config.SYMBOL, position_size)}). "
                        f"Initial Order Info: Filled Qty={format_amount(current_exchange, config.SYMBOL, filled_qty_order)}, "
                        f"AvgPrice={format_price(current_exchange, config.SYMBOL, avg_fill_price_order) if avg_fill_price_order else 'N/A'}{Style.RESET_ALL}"
                    )

                    # --- CRITICAL STEP: Wait and Confirm Position Update ---
                    logger.info(
                        f"Waiting {config.POSITION_CONFIRM_DELAY_SECONDS}s for position update after entry order {order_id_short}..."
                    )
                    await asyncio.sleep(config.POSITION_CONFIRM_DELAY_SECONDS)

                    logger.debug(
                        f"{Fore.CYAN}# Awaiting position confirmation after entry ({config.SYMBOL})...{Style.RESET_ALL}"
                    )
                    pos_after_entry = await bybit.get_current_position_bybit_v5(
                        current_exchange, config.SYMBOL, config
                    )

                    # --- Validate the confirmed position ---
                    # Check side matches signal, and quantity is positive (using epsilon)
                    expected_pos_side = (
                        config.POS_LONG
                        if signal == config.SIDE_BUY
                        else config.POS_SHORT
                    )
                    if (
                        pos_after_entry
                        and pos_after_entry.get("side") == expected_pos_side
                        and pos_after_entry.get("qty", Decimal(0))
                        > config.POSITION_QTY_EPSILON
                    ):
                        # --- Position Confirmed! ---
                        actual_filled_qty = pos_after_entry["qty"]
                        # Use actual entry price from position if available, fallback to order's avg or original estimate
                        actual_entry_price = (
                            pos_after_entry.get("entry_price")
                            or avg_fill_price_order
                            or current_price
                        )
                        pos_qty_formatted = format_amount(
                            current_exchange, config.SYMBOL, actual_filled_qty
                        )
                        entry_price_formatted = format_price(
                            current_exchange, config.SYMBOL, actual_entry_price
                        )

                        logger.info(
                            f"{Fore.GREEN}{Style.BRIGHT}Position confirmed OPEN ({config.SYMBOL}): {pos_after_entry['side']} {pos_qty_formatted} @ ~{entry_price_formatted}{Style.RESET_ALL}"
                        )

                        # Optional: Re-calculate SL price based on actual_entry_price if it differs significantly?
                        # stop_loss_price = calculate_stop_loss(current_exchange, df_with_indicators, signal, actual_entry_price, config)
                        # if not stop_loss_price:
                        #     logger.error(f"{Fore.RED}Failed to recalculate SL based on actual entry price {entry_price_formatted}. Using original SL calculation.{Style.RESET_ALL}")
                        #     # Need to decide if we proceed without a recalculated SL or emergency close.
                        #     # For simplicity here, we proceed with the original SL price.

                        # --- Place Stop Loss Order ---
                        sl_side = (
                            config.SIDE_SELL
                            if signal == config.SIDE_BUY
                            else config.SIDE_BUY
                        )
                        sl_price_formatted = format_price(
                            current_exchange, config.SYMBOL, stop_loss_price
                        )
                        logger.info(
                            f"Placing {sl_side.upper()} native stop-loss order for {pos_qty_formatted} {config.SYMBOL} at {sl_price_formatted}..."
                        )
                        logger.debug(
                            f"{Fore.CYAN}# Awaiting native stop-loss placement for {config.SYMBOL}...{Style.RESET_ALL}"
                        )
                        # ASSUMPTION: place_native_stop_loss IS async
                        sl_order = await bybit.place_native_stop_loss(
                            current_exchange,
                            config.SYMBOL,
                            sl_side,
                            qty=actual_filled_qty,  # Use the actual confirmed quantity
                            stop_price=stop_loss_price,  # Use the SL price calculated earlier
                            config=config,
                        )

                        if sl_order and sl_order.get("id"):
                            sl_id_short = format_order_id(sl_order["id"])
                            logger.success(
                                f"{Fore.GREEN}Native stop-loss order {sl_id_short} placed successfully for {pos_qty_formatted} {config.SYMBOL} at {sl_price_formatted}.{Style.RESET_ALL}"
                            )
                            # Track the SL order ID for this symbol
                            stop_loss_orders[config.SYMBOL] = sl_order["id"]

                            if config.ENABLE_SMS_ALERTS:
                                alert_msg = f"[{config.SYMBOL}] Entered {signal.upper()} {pos_qty_formatted} @ ~{entry_price_formatted}. SL {sl_id_short} @ {sl_price_formatted}"
                                send_sms_alert(alert_msg, config)
                        else:
                            # --- CRITICAL: Failed to place SL after position confirmed open ---
                            logger.critical(
                                f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to place stop-loss order for {config.SYMBOL} after entering {signal.upper()} position ({pos_qty_formatted})! Attempting EMERGENCY CLOSE!{Style.RESET_ALL}"
                            )
                            if config.ENABLE_SMS_ALERTS:
                                alert_msg = f"[{config.SYMBOL}] URGENT: Failed SL place after {signal.upper()} entry ({pos_qty_formatted})! Closing pos NOW."
                                send_sms_alert(alert_msg, config)

                            logger.debug(
                                f"{Fore.CYAN}# Awaiting emergency position close for {config.SYMBOL} due to failed SL placement...{Style.RESET_ALL}"
                            )
                            # Use the confirmed position info for closing
                            emergency_close_order = await bybit.close_position_reduce_only(
                                current_exchange,
                                config.SYMBOL,
                                config,
                                position_to_close=pos_after_entry,  # Use the confirmed details
                                reason="Emergency Close - Failed SL Placement",
                            )
                            if emergency_close_order and emergency_close_order.get(
                                "id"
                            ):
                                logger.warning(
                                    f"{Fore.YELLOW}Position ({config.SYMBOL}) emergency closed via order {format_order_id(emergency_close_order['id'])} due to failed SL placement.{Style.RESET_ALL}"
                                )
                            else:
                                logger.critical(
                                    f"{Back.RED}{Fore.WHITE}EMERGENCY FAILED: FAILED TO CLOSE POSITION ({pos_qty_formatted} {config.SYMBOL}) AFTER FAILED SL PLACEMENT! MANUAL INTERVENTION REQUIRED IMMEDIATELY!{Style.RESET_ALL}"
                                )
                                if config.ENABLE_SMS_ALERTS:
                                    send_sms_alert(
                                        f"[{config.SYMBOL}] !!! CRITICAL MANUAL ACTION: Failed emergency close pos {pos_qty_formatted} after failed SL place !!!",
                                        config,
                                    )
                            # Clear SL tracking just in case something went wrong
                            stop_loss_orders.pop(config.SYMBOL, None)
                            # Wait longer after critical failure before next cycle
                            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 3)

                    else:
                        # --- Position confirmation failed ---
                        pos_side_report = (
                            pos_after_entry.get("side", "N/A")
                            if pos_after_entry
                            else "Fetch Failed"
                        )
                        pos_qty_report = (
                            format_amount(
                                current_exchange,
                                config.SYMBOL,
                                pos_after_entry.get("qty", 0),
                            )
                            if pos_after_entry
                            else "N/A"
                        )
                        logger.error(
                            f"{Fore.RED}Entry order {order_id_short} for {config.SYMBOL} submitted, but position confirmation failed or quantity is zero/insufficient. "
                            f"Expected Side: {expected_pos_side}, Confirmed State: Side={pos_side_report}, Qty={pos_qty_report}. Manual check advised.{Style.RESET_ALL}"
                        )
                        if config.ENABLE_SMS_ALERTS:
                            alert_msg = f"[{config.SYMBOL}] URGENT: Entry order {order_id_short} confirm failed! Pos Side: {pos_side_report}, Qty: {pos_qty_report}. Check manual!"
                            send_sms_alert(alert_msg, config)
                        # What to do here?
                        # Option 1: Assume order failed or only partially filled, do nothing, wait for next cycle. (Safer)
                        # Option 2: Try to cancel the entry order ID. Risky if partially filled. Requires checking order status first.
                        # Option 3: Try to close any potential small position. Risky.
                        # Safest is likely Option 1: Log and wait.

                else:
                    # --- Market entry order placement itself failed ---
                    logger.error(
                        f"{Fore.RED}Entry market order placement FAILED for {config.SYMBOL}. No valid order dictionary received from helper. Check exchange status and API logs.{Style.RESET_ALL}"
                    )
                    # Optional: Send SMS if this persists?

            # --- 6. Wait for next cycle ---
            cycle_end_time = pd.Timestamp.now(tz="UTC")
            cycle_duration = (cycle_end_time - cycle_start_time).total_seconds()
            wait_time = max(0, config.LOOP_DELAY_SECONDS - cycle_duration)
            logger.info(
                f"Cycle complete ({config.SYMBOL}). Duration: {cycle_duration:.2f}s. Waiting {wait_time:.2f}s..."
            )
            await asyncio.sleep(wait_time)

        # --- Exception Handling for the Main Loop ---
        except ccxt.NetworkError as e:
            logger.warning(
                f"{Fore.YELLOW}Network Error in main loop ({config.SYMBOL}): {e}. Retrying after {config.RETRY_DELAY_SECONDS * 5:.1f}s delay...{Style.RESET_ALL}"
            )
            await asyncio.sleep(
                config.RETRY_DELAY_SECONDS * 5
            )  # Longer delay for network issues
        except ccxt.ExchangeNotAvailable as e:
            logger.error(
                f"{Fore.RED}Exchange Not Available ({config.SYMBOL}): {e}. Possibly maintenance. Waiting longer ({config.LOOP_DELAY_SECONDS * 5}s)...{Style.RESET_ALL}"
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] Exchange Not Available: {e}. Pausing.", config
                )
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 5)
        except ccxt.RateLimitExceeded as e:
            logger.warning(
                f"{Fore.YELLOW}Rate Limit Exceeded ({config.SYMBOL}): {e}. Waiting {config.RETRY_DELAY_SECONDS * 10:.1f}s...{Style.RESET_ALL}"
            )  # Longer delay for rate limits
            await asyncio.sleep(config.RETRY_DELAY_SECONDS * 10)
        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Authentication Error during runtime ({config.SYMBOL}): {e}. API keys might be invalid or expired. Stopping strategy.{Style.RESET_ALL}",
                exc_info=True,
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] CRITICAL: Auth Error runtime! Strategy stopping.",
                    config,
                )
            break  # Exit the loop on auth errors
        except ccxt.OrderNotFound as e:
            # This might occur if trying to cancel an already filled/cancelled order - often benign
            logger.warning(
                f"{Fore.YELLOW}OrderNotFound error encountered in loop: {e}. This might be okay if cancelling an order that already executed/cancelled.{Style.RESET_ALL}"
            )
            # Check if the missing order ID was a tracked SL order
            order_id_in_error = None
            try:  # Try to parse the order ID from the error message
                parts = str(e).split("order id ")
                if len(parts) > 1:
                    order_id_in_error = parts[1].split(" ")[0]
            except Exception:
                pass  # Ignore parsing errors

            if order_id_in_error and order_id_in_error in stop_loss_orders.values():
                logger.info(
                    f"OrderNotFound likely relates to tracked SL order {order_id_in_error}. Removing from tracking."
                )
                # Find the key (symbol) for the value (order id) and pop it
                key_to_remove = next(
                    (k for k, v in stop_loss_orders.items() if v == order_id_in_error),
                    None,
                )
                if key_to_remove:
                    stop_loss_orders.pop(key_to_remove, None)
            await asyncio.sleep(
                config.LOOP_DELAY_SECONDS
            )  # Normal delay after OrderNotFound
        except ccxt.InsufficientFunds as e:
            logger.error(
                f"{Back.RED}{Fore.WHITE}Insufficient Funds Error ({config.SYMBOL}): {e}. Check balance and open orders/positions. Pausing loop significantly.{Style.RESET_ALL}"
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] Insufficient Funds Error! Check account. Strategy Paused.",
                    config,
                )
            # Long pause - needs manual intervention usually
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 10)
        except ccxt.ExchangeError as e:
            # Catch other specific exchange errors (e.g., invalid parameters, margin errors)
            logger.error(
                f"{Fore.RED}Unhandled Exchange Error in main loop ({config.SYMBOL}): {e}. Retrying after {config.LOOP_DELAY_SECONDS}s delay...{Style.RESET_ALL}",
                exc_info=True,
            )
            # Optional: Add SMS for specific error codes if needed (e.g., margin call)
            await asyncio.sleep(config.LOOP_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.warning(
                f"{Fore.YELLOW}{Style.BRIGHT}Keyboard interrupt received. Initiating graceful shutdown...{Style.RESET_ALL}"
            )
            break  # Exit the loop
        except NameError as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}A NameError occurred: {e}. Check imports (e.g., colorama, pandas_ta) and variable names.{Style.RESET_ALL}",
                exc_info=True,
            )
            if not COLORAMA_AVAILABLE and (
                "Fore" in str(e) or "Back" in str(e) or "Style" in str(e)
            ):
                logger.critical(
                    f"{Fore.YELLOW}Suggestion: Ensure {Style.BRIGHT}'pip install colorama'{Style.RESET_ALL}{Fore.YELLOW} is installed and the import succeeded.{Style.RESET_ALL}"
                )
            if "pandas_ta" in str(e):
                logger.critical(
                    f"{Fore.YELLOW}Suggestion: Ensure {Style.BRIGHT}'pip install pandas_ta'{Style.RESET_ALL}{Fore.YELLOW} is installed for fallback ATR calculation.{Style.RESET_ALL}"
                )
            break  # Often critical, stop the strategy
        except Exception as e:
            # Catch-all for any other unexpected errors during the loop
            logger.critical(
                f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL ERROR IN MAIN LOOP ({config.SYMBOL}) !!!{Style.RESET_ALL}",
                exc_info=True,
            )
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Error Type: {type(e).__name__}, Message: {e}{Style.RESET_ALL}"
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] CRITICAL ERROR: {type(e).__name__} in loop! Check logs!",
                    config,
                )
            logger.info(
                f"{Fore.YELLOW}Attempting to continue after critical error... pausing for {config.LOOP_DELAY_SECONDS * 3}s{Style.RESET_ALL}"
            )
            # Longer pause after unknown critical error
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 3)

    logger.info(
        f"{Fore.MAGENTA}{Style.BRIGHT}--- Ehlers Volumetric Strategy Loop Stopped for {config.SYMBOL} ---{Style.RESET_ALL}"
    )


# --- Asynchronous Main Function (Setup & Execution) ---
async def main():
    """Initializes resources, runs setup checks, executes the strategy loop, and handles cleanup."""
    global logger, exchange, CONFIG  # Allow assignment to globals
    setup_success = False  # Track if setup completes without critical errors
    exit_code = 1  # Default to error exit code

    # --- 0. Initialize Logger (Must happen first) ---
    log_file_path = os.getenv("LOG_FILE_PATH", "ehlers_strategy.log")
    console_log_level_str = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
    file_log_level_str = os.getenv("LOG_FILE_LEVEL", "DEBUG").upper()
    try:
        logger = setup_logger(
            logger_name="EhlersStrategy",
            log_file=log_file_path,
            console_level=logging.getLevelName(
                console_log_level_str
            ),  # Handles invalid level names
            file_level=logging.getLevelName(file_log_level_str),
            third_party_log_level=logging.WARNING,  # Reduce noise from ccxt/requests
        )
    except Exception as e:
        # Use print as logger isn't working
        print(
            f"{Back.RED}{Fore.WHITE}FATAL: Failed to initialize logger: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        print(f"Log file path attempted: {log_file_path}", file=sys.stderr)
        print(
            f"Console Level: {console_log_level_str}, File Level: {file_log_level_str}",
            file=sys.stderr,
        )
        sys.exit(exit_code)  # Exit with error

    # --- 1. Load Configuration ---
    try:
        CONFIG = Config()
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Failed to load or initialize configuration: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        sys.exit(exit_code)  # Exit with error

    # --- 2. Validate Core Config (API Keys) ---
    if not CONFIG.API_KEY or not CONFIG.API_SECRET:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}API Key or Secret not found in environment variables (BYBIT_API_KEY, BYBIT_API_SECRET). Grant the script access! Exiting.{Style.RESET_ALL}"
        )
        sys.exit(exit_code)  # Exit with error
    logger.info(
        f"Strategy: {CONFIG.strategy['name']}, Symbol: {CONFIG.SYMBOL}, Timeframe: {CONFIG.TIMEFRAME}, Testnet: {CONFIG.TESTNET_MODE}"
    )
    if CONFIG.ENABLE_SMS_ALERTS:
        logger.info(f"SMS Alerts Enabled for: {CONFIG.SMS_RECIPIENT_NUMBER}")
    else:
        logger.info("SMS Alerts Disabled.")

    # --- 3. Setup Phase (Exchange Connection, Market Validation, Settings) ---
    try:
        logger.info("--- Starting Setup Phase ---")
        # --- 3a. Initialize Async Exchange ---
        logger.info(f"Initializing async connection to {CONFIG.EXCHANGE_ID}...")
        # ASSUMPTION: initialize_bybit is adapted to return an async exchange instance AND handles testnet mode.
        exchange = bybit.initialize_bybit(CONFIG, use_async=True)  # Pass async flag

        # Fallback: Direct initialization if helper fails or isn't adapted
        if not exchange:
            logger.warning(
                f"{Fore.YELLOW}Helper 'initialize_bybit' did not return an exchange object. Attempting direct ccxt_async initialization...{Style.RESET_ALL}"
            )
            try:
                exchange_class = getattr(ccxt_async, CONFIG.EXCHANGE_ID)
                exchange = exchange_class(
                    {
                        "apiKey": CONFIG.API_KEY,
                        "secret": CONFIG.API_SECRET,
                        "options": {
                            "defaultType": CONFIG.EXPECTED_MARKET_TYPE,
                            # Add V5 specific options if needed, e.g., unified margin?
                            # 'defaultMarginMode': CONFIG.DEFAULT_MARGIN_MODE.lower(), # Usually set via API call, not option
                            # 'brokerId': 'YOUR_BROKER_ID', # If applicable
                        },
                        "enableRateLimit": True,  # Let ccxt handle basic rate limiting
                        "adjustForTimeDifference": True,  # Recommended
                        "recvWindow": CONFIG.DEFAULT_RECV_WINDOW,
                    }
                )
                if CONFIG.TESTNET_MODE:
                    logger.info("Setting sandbox mode (Testnet).")
                    exchange.set_sandbox_mode(True)
            except Exception as direct_init_err:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}Direct ccxt_async initialization failed: {direct_init_err}{Style.RESET_ALL}",
                    exc_info=True,
                )
                raise ConnectionError(
                    "Failed to initialize Bybit async exchange object via helper or directly."
                ) from direct_init_err

        if not exchange:  # Check again after fallback attempt
            raise ConnectionError("Failed to initialize Bybit async exchange object.")

        logger.success(
            f"{Fore.GREEN}Async Exchange connection initialized successfully ({'Testnet' if CONFIG.TESTNET_MODE else 'Mainnet'}).{Style.RESET_ALL}"
        )

        # --- 3b. Load Markets (Async) ---
        logger.info("Loading markets from exchange...")
        logger.debug(
            f"{Fore.CYAN}# Awaiting exchange.load_markets()...{Style.RESET_ALL}"
        )
        await exchange.load_markets()
        logger.info(f"Markets loaded. Found {len(exchange.markets)} markets.")
        if CONFIG.SYMBOL not in exchange.markets:
            # Log available symbols if feasible (might be very long)
            # available_symbols = list(exchange.markets.keys())
            # logger.debug(f"Available symbols: {available_symbols[:20]}...") # Log first few
            raise ValueError(
                f"Symbol {CONFIG.SYMBOL} not found in loaded markets from {CONFIG.EXCHANGE_ID}."
            )

        # --- 3c. Validate Market (Sync check after loading) ---
        logger.info(f"Validating market configuration for {CONFIG.SYMBOL}...")
        # ASSUMPTION: validate_market is synchronous
        market_details = bybit.validate_market(exchange, CONFIG.SYMBOL, CONFIG)
        if not market_details:
            # Error should have been logged within the helper function
            raise ValueError(
                f"Market validation failed for {CONFIG.SYMBOL}. Check logs above for details."
            )

        market_type = market_details.get("type", "N/A")
        market_logic = (
            "linear"
            if market_details.get("linear", False)
            else ("inverse" if market_details.get("inverse", False) else "N/A")
        )
        logger.success(
            f"{Fore.GREEN}Market {CONFIG.SYMBOL} validated: Type={market_type}, Logic={market_logic}{Style.RESET_ALL}"
        )

        # --- 3d. Set Leverage ---
        # CRITICAL: Understand if the `bybit.set_leverage` helper is async or sync.
        # ASSUMPTION HERE: `bybit.set_leverage` helper is ASYNCHRONOUS (uses `await` internally).
        # If it were synchronous, we would NOT use `await` here, but it could block the event loop.
        logger.info(
            f"Attempting to set leverage for {CONFIG.SYMBOL} to {CONFIG.DEFAULT_LEVERAGE}x..."
        )
        try:
            logger.debug(
                f"{Fore.CYAN}# Awaiting bybit.set_leverage(...)...{Style.RESET_ALL}"
            )
            # Use await IF the helper function is defined with `async def`
            leverage_set_result = await bybit.set_leverage(
                exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_LEVERAGE, CONFIG
            )

            # Check the return value explicitly for failure indication (e.g., False, None, or exception)
            if (
                not leverage_set_result
            ):  # Adjust condition based on helper's success/failure return convention
                # Helper function should have logged the specific Bybit API error.
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}Helper function 'set_leverage' indicated failure (returned: {leverage_set_result}). Check previous logs for API errors.{Style.RESET_ALL}"
                )
                # Raise an exception to trigger setup failure handling
                raise ccxt.ExchangeError(
                    "Leverage setting failed as reported by helper function."
                )

            logger.success(
                f"{Fore.GREEN}Leverage successfully set to {CONFIG.DEFAULT_LEVERAGE}x for {CONFIG.SYMBOL}.{Style.RESET_ALL}"
            )

        except ccxt.ExchangeError as leverage_error:
            # Catch specific exchange errors potentially raised BY the helper OR by our check above
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Failed to set leverage due to exchange error: {leverage_error}{Style.RESET_ALL}",
                exc_info=True,
            )
            # Log specific details if available in the error (e.g., error code, message from Bybit)
            logger.critical(
                f"Check if leverage {CONFIG.DEFAULT_LEVERAGE}x is valid for {CONFIG.SYMBOL} and if API keys have trade permissions."
            )
            raise  # Re-raise to be caught by the outer setup error handler
        except Exception as e:  # Catch other unexpected errors from the async call
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Unexpected error during set_leverage call: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            raise  # Re-raise

        # --- 3e. (Optional but Recommended) Set Margin Mode / Position Mode ---
        # ASSUMPTION: These helpers are ASYNCHRONOUS
        # logger.info(f"Setting margin mode to {CONFIG.DEFAULT_MARGIN_MODE} for {CONFIG.SYMBOL}...")
        # logger.debug(f"{Fore.CYAN}# Awaiting bybit.set_margin_mode(...)...{Style.RESET_ALL}")
        # mode_set_ok = await bybit.set_margin_mode(exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_MARGIN_MODE, CONFIG)
        # if not mode_set_ok: raise ccxt.ExchangeError(f"Failed to set margin mode to {CONFIG.DEFAULT_MARGIN_MODE}.")
        # logger.success(f"{Fore.GREEN}Margin mode set to {CONFIG.DEFAULT_MARGIN_MODE}.{Style.RESET_ALL}")

        # logger.info(f"Setting position mode to {CONFIG.DEFAULT_POSITION_MODE} for {CONFIG.SYMBOL}...")
        # logger.debug(f"{Fore.CYAN}# Awaiting bybit.set_position_mode(...)...{Style.RESET_ALL}")
        # pos_mode_set_ok = await bybit.set_position_mode(exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_POSITION_MODE, CONFIG) # Hedge mode might require symbol suffix _hedge
        # if not pos_mode_set_ok: raise ccxt.ExchangeError(f"Failed to set position mode to {CONFIG.DEFAULT_POSITION_MODE}.")
        # logger.success(f"{Fore.GREEN}Position mode set to {CONFIG.DEFAULT_POSITION_MODE}.{Style.RESET_ALL}")

        # --- Mark Setup as Successful ---
        setup_success = True
        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}--- Setup Phase Completed Successfully ---{Style.RESET_ALL}"
        )

    # --- Catch Specific Setup Errors ---
    except ccxt.AuthenticationError as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Authentication Error during setup: {e}. Check API keys/permissions. Exiting.{Style.RESET_ALL}",
            exc_info=True,
        )
        if CONFIG and CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"[{CONFIG.SYMBOL if CONFIG else 'N/A'}] EXIT: Auth error setup.",
                CONFIG,
            )
    except (ccxt.ExchangeError, ccxt.NetworkError, ConnectionError, ValueError) as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Exchange/Network/Configuration Error during setup: {e}. Exiting.{Style.RESET_ALL}",
            exc_info=True,
        )
        if CONFIG and CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"[{CONFIG.SYMBOL if CONFIG else 'N/A'}] EXIT: Setup fail ({type(e).__name__}). Logs.",
                CONFIG,
            )
    except Exception as e:
        # Catch any other unexpected errors during setup
        logger.critical(
            f"{Back.RED}{Fore.WHITE}Unexpected critical error during setup phase: {e}. Exiting.{Style.RESET_ALL}",
            exc_info=True,
        )
        if CONFIG and CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"[{CONFIG.SYMBOL if CONFIG else 'N/A'}] EXIT: Unexpected critical setup error ({type(e).__name__}). Logs.",
                CONFIG,
            )

    # --- 4. Run Strategy only if setup succeeded ---
    if setup_success and exchange and CONFIG:
        try:
            await run_strategy(CONFIG, exchange)
            # If run_strategy finishes normally (e.g., loop broken by condition, not error)
            logger.info("Strategy execution loop finished normally.")
            exit_code = (
                0  # Success exit code if loop finishes without unhandled exception
            )
        except Exception as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Strategy execution loop terminated abnormally by unhandled exception: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            if CONFIG.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{CONFIG.SYMBOL}] CRITICAL FAILURE: Loop terminated ({type(e).__name__}). Logs!",
                    CONFIG,
                )
            exit_code = 1  # Ensure error exit code
    elif not setup_success:
        logger.error(
            f"{Fore.RED}Strategy execution skipped due to setup failure.{Style.RESET_ALL}"
        )
        exit_code = 1  # Ensure error exit code

    # --- 5. Cleanup Phase: Close Exchange Connection ---
    logger.info(f"{Fore.CYAN}--- Initiating Cleanup Phase ---{Style.RESET_ALL}")
    if exchange and hasattr(exchange, "close") and callable(exchange.close):
        try:
            logger.info(
                f"{Fore.CYAN}# Closing connection to the exchange...{Style.RESET_ALL}"
            )
            # exchange.close() IS async in ccxt.async_support
            await exchange.close()
            logger.info(
                f"{Fore.GREEN}Exchange connection closed gracefully.{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error occurred while closing the exchange connection: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            # Don't change exit code here, primary error was likely earlier
    else:
        logger.warning(
            f"{Fore.YELLOW}Exchange object was not initialized or has no close method; skipping explicit close.{Style.RESET_ALL}"
        )

    # Final log message indicating shutdown status
    final_message = f"--- Strategy Shutdown Complete --- Exit Code: {exit_code}"
    if exit_code == 0:
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}{final_message}{Style.RESET_ALL}")
    else:
        logger.warning(
            f"{Fore.YELLOW}{Style.BRIGHT}{final_message} (Indicates setup failure or abnormal termination){Style.RESET_ALL}"
        )

    sys.exit(exit_code)


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        # Python 3.7+
        asyncio.run(main())
    except KeyboardInterrupt:
        # Use print because logger might not be available if interrupt happens very early
        print(
            f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt detected. Exiting script...{Style.RESET_ALL}"
        )
        # Cleanup (like exchange.close) should happen within main()'s final section
        sys.exit(0)  # Clean exit on user interrupt
    except Exception as e:
        # Catch errors during asyncio.run() itself if main() fails before logger setup
        print(
            f"{Back.RED}{Fore.WHITE}Fatal error during asyncio setup or top-level execution: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        # Attempt to log if logger was initialized (might not work)
        if logger:
            logger.critical(
                "Fatal error during asyncio setup or execution", exc_info=True
            )
        else:
            import traceback

            traceback.print_exc()  # Print traceback if logger failed
        sys.exit(1)  # Error exit code
