          # Suggestion 1: Dynamic ATR Multiplier based on Volatility (Conceptual)                             # Inside trade_logic, after calculating vol_atr_results
volatility_regime = 'normal' # Replace with actual volatility check (e.g., ATR vs long-term ATR MA) if volatility_regime == 'high':                       effective_sl_mult = CONFIG.atr_stop_loss_multiplier * Decimal("1.2")                                effective_tp_mult = CONFIG.atr_take_profit_multiplier * Decimal("0.8")                          elif volatility_regime == 'low':                      effective_sl_mult = CONFIG.atr_stop_loss_multiplier * Decimal("0.8")                                effective_tp_mult = CONFIG.atr_take_profit_multiplier * Decimal("1.2")                          else: # normal                                        effective_sl_mult = CONFIG.atr_stop_loss_multiplier                                                 effective_tp_mult = CONFIG.atr_take_profit_multiplier                                           # Use effective_sl_mult and effective_tp_mult when calculating sl_distance/tp_distance              sl_distance = current_atr * effective_sl_mult     tp_distance = current_atr * effective_tp_mult     ```                                                                                                 ```python                                         # Suggestion 2: Cooldown Period After Stop-Loss Hit                                                 # Add global variable or class attribute: last_loss_timestamp = 0                                   # Add config parameter: COOLDOWN_AFTER_LOSS_SECONDS = 300                                           # Inside trade_logic, before entry logic          global last_loss_timestamp                        cooldown_active = (time.time() - last_loss_timestamp) < CONFIG.cooldown_after_loss_seconds          if position_side == POSITION_SIDE_NONE and cooldown_active:                                             logger.info(f"Cooldown active after recent loss. Skipping entry checks for {CONFIG.cooldown_after_loss_seconds - (time.time() - last_loss_timestamp):.0f}s.")                                           return # Skip entry logic
                                                  # Inside close_position, if the reason indicates a stop-loss hit:                                   # (Need to modify close_position or track SL hits separately)                                       # if reason == "StopLoss": # Assuming SL closure reason can be identified                           #     global last_loss_timestamp                  #     last_loss_timestamp = time.time()           #     logger.info("Stop-loss triggered. Initiating cooldown period.")                               ```                                                                                                 ```python                                         # Suggestion 3: RSI Filter for Entries            # Add config: USE_RSI_FILTER = True, RSI_PERIOD = 14, RSI_OVERBOUGHT = 70, RSI_OVERSOLD = 30        # Inside trade_logic, after calculating indicatorsif CONFIG.use_rsi_filter:                             df.ta.rsi(length=CONFIG.rsi_period, append=True)                                                    rsi_col = f"RSI_{CONFIG.rsi_period}"              if rsi_col in df.columns and pd.notna(df[rsi_col].iloc[-1]):                                            last_rsi = Decimal(str(df[rsi_col].iloc[-1]))                                                       logger.debug(f"RSI({CONFIG.rsi_period}) = {last_rsi:.2f}")                                      else:                                                 last_rsi = None                                   logger.warning("Could not calculate RSI for filter.")                                                                                         # Inside entry confirmation logic                 rsi_confirmed = True
if CONFIG.use_rsi_filter and last_rsi is not None:    if selected_side == SIDE_BUY and last_rsi > Decimal(str(CONFIG.rsi_overbought)):                        rsi_confirmed = False                             logger.info(f"Entry REJECTED ({selected_side.upper()}): RSI confirmation FAILED (RSI {last_rsi:.2f} > Overbought {CONFIG.rsi_overbought}).")      elif selected_side == SIDE_SELL and last_rsi < Decimal(str(CONFIG.rsi_oversold)):                       rsi_confirmed = False
        logger.info(f"Entry REJECTED ({selected_side.upper()}): RSI confirmation FAILED (RSI {last_rsi:.2f} < Oversold {CONFIG.rsi_oversold}).")          else:                                                  logger.info(f"{Fore.GREEN}Entry Check ({selected_side.upper()}): RSI OK (Value: {last_rsi:.2f}).{Style.RESET_ALL}")                          # Add rsi_confirmed to the final check: if volume_confirmed and ob_confirmed and rsi_confirmed:     ```                                                                                                 ```python                                         # Suggestion 4: Partial Take Profit (Simple Example - 50% at TP1)                                   # Add config: ENABLE_PARTIAL_TP = True, PARTIAL_TP_RATIO = 0.5, PARTIAL_TP_MULT_ADJUST = 0.5        # Modify place_risked_market_order or add a new function manage_partial_tp                          # This snippet focuses on placing the initial TP order for the first partial target                                                                   if CONFIG.enable_partial_tp:
    # Calculate TP1 based on a fraction of the original TP distance                                     tp1_distance = tp_distance * Decimal(str(CONFIG.PARTIAL_TP_MULT_ADJUST))                            if selected_side == SIDE_BUY:                         tp1_price_raw = entry_price_est + tp1_distance                                                  else: # SIDE_SELL                                     tp1_price_raw = entry_price_est - tp1_distance                                                  tp1_price = tp1_price_raw.quantize(price_precision, rounding=ROUND_HALF_UP)                                                                           # Place TP order for PARTIAL_TP_RATIO of the quantity at tp1_price                                  partial_qty_decimal = (actual_filled_qty * Decimal(str(CONFIG.PARTIAL_TP_RATIO))).quantize(quantity_precision, rounding=ROUND_HALF_UP) # Assumes quantity_precision is defined                          partial_qty_float = float(partial_qty_decimal)    tp1_params = { ... } # Same as tp_params but with tp1_price                                         # exchange.create_order(symbol, ORDER_TYPE_STOP_MARKET, close_side, partial_qty_float, params=tp1_params)                                             logger.info(f"Partial TP1 order placed for {partial_qty_float} at {tp1_price}.")                    # NOTE: Requires logic to handle the remaining position: move SL to breakeven, set TP2, etc. upon TP1 fill.                                           # This usually requires monitoring the TP1 order status or position changes.                    ```                                                                                                 ```python                                         # Suggestion 5: Trailing Stop Loss (Basic - Requires Order Modification Logic)                      # Add config: ENABLE_TRAILING_STOP = True, TSL_ACTIVATION_ATR_MULT = 1.0, TSL_TRAILING_ATR_MULT = 1.5                                                 # This snippet shows the *decision* logic, not the order modification itself                                                                          # Inside trade_logic, when holding a position     if position_side != POSITION_SIDE_NONE and CONFIG.enable_trailing_stop:                                 # Calculate activation price                      activation_distance = current_atr * Decimal(str(CONFIG.tsl_activation_atr_mult))                    if position_side == POSITION_SIDE_LONG:               activation_price = position_entry_price + activation_distance                                       should_trail = current_price > activation_price                                                 else: # SHORT                                         activation_price = position_entry_price - activation_distance                                       should_trail = current_price < activation_price                                                                                                   if should_trail:                                      # Calculate new TSL price                         trailing_distance = current_atr * Decimal(str(CONFIG.tsl_trailing_atr_mult))                        if position_side == POSITION_SIDE_LONG:               new_tsl_price = current_price - trailing_distance                                                   # Ensure TSL only moves up
            # current_sl_price = fetch_current_sl_price(exchange, symbol, sl_order_id) # Needs function to get current SL                                         # if new_tsl_price > current_sl_price:
            #     modify_stop_loss_order(exchange, symbol, sl_order_id, new_tsl_price) # Needs function to modify SL
        else: # SHORT                                         new_tsl_price = current_price + trailing_distance                                                   # Ensure TSL only moves down                      # current_sl_price = fetch_current_sl_price(exchange, symbol, sl_order_id) # Needs function to get current SL                                         # if new_tsl_price < current_sl_price:            #     modify_stop_loss_order(exchange, symbol, sl_order_id, new_tsl_price) # Needs function to modify SL                                          logger.info(f"Trailing Stop condition met. Proposed new SL: {new_tsl_price:.4f}")                   # NOTE: Requires robust functions to fetch current SL price and modify the SL order.                # Modifying stopMarket orders via API might require cancelling and replacing.

`python                                         # Suggestion 7: Performance Logging to CSV
import csv                                        from pathlib import Path                                                                            # Add config: PERFORMANCE_LOG_FILE = "trades.csv" # Define function outside main loop               def log_trade_result(symbol, entry_time, exit_time, side, entry_price, exit_price, qty, pnl, reason):                                                     log_file = Path(CONFIG.performance_log_file)      file_exists = log_file.is_file()                  try:                                                  with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:                                      fieldnames = ['Symbol', 'EntryTime', 'ExitTime', 'Side', 'EntryPrice', 'ExitPrice', 'Quantity', 'PNL_Quote', 'Reason']                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)                                             if not file_exists:                                   writer.writeheader()                          writer.writerow({                                     'Symbol': symbol,
                'EntryTime': entry_time, # Need to store entry timestamp                                            'ExitTime': exit_time, # Timestamp of closure                                                       'Side': side, # Long/Short                        'EntryPrice': f"{entry_price:.8f}",                                                                 'ExitPrice': f"{exit_price:.8f}",                 'Quantity': f"{qty:.8f}",                         'PNL_Quote': f"{pnl:.8f}", # Need to calculate PNL                                                  'Reason': reason                              })                                        except Exception as e:                                logger.error(f"Failed to log trade to CSV: {e}")                                                                                              # Call log_trade_result after a position is closed and PNL calculated.                              # Requires storing entry details (time, price, side, qty) and calculating PNL upon exit.            # Example call location: After close_position confirms a closure.                                   # exit_price = Decimal(str(close_result.get(AVERAGE_KEY, '0')))                                     # pnl = calculate_pnl(position_side, position_qty, position_entry_price, exit_price) # Need calculate_pnl function                                    # log_trade_result(CONFIG.symbol, entry_timestamp, pd.Timestamp.utcnow(), position_side, position_entry_price, exit_price, position_qty, pnl, reason) ```                                                                                                 ```python                                         # Suggestion 8: Adaptive Risk Percentage (Conceptual - Based on recent Win Rate)                    # Add state variables: recent_trades = [], WIN_RATE_LOOKBACK = 20                                   # Add config: MIN_RISK_PERCENT = 0.002, MAX_RISK_PERCENT = 0.01                                     # Function to update risk (call after logging trade result)                                         def update_adaptive_risk():                           global recent_trades                              if len(recent_trades) < CONFIG.win_rate_lookback:                                                       return CONFIG.risk_per_trade_percentage # Use default if not enough history                                                                       wins = sum(1 for trade in recent_trades[-CONFIG.win_rate_lookback:] if trade['pnl'] > 0)            win_rate = wins / CONFIG.win_rate_lookback                                                          # Simple linear scaling between min/max risk based on win rate                                      risk_range = CONFIG.max_risk_percent - CONFIG.min_risk_percent                                      adaptive_risk = CONFIG.min_risk_percent + (risk_range * Decimal(str(win_rate))) # Scale risk with win rate                                            adaptive_risk = max(CONFIG.min_risk_percent, min(CONFIG.max_risk_percent, adaptive_risk)) # Clamp
                                                      logger.info(f"Adaptive Risk: Win Rate ({CONFIG.win_rate_lookback} trades) = {win_rate:.1%}, New Risk % = {adaptive_risk:.3%}")                        return adaptive_risk # Return Decimal         
# In trade_logic, before calculate_position_size: # current_risk_percent = update_adaptive_risk() # Or get from global/class state updated elsewhere  # quantity_decimal, margin_est_decimal = calculate_position_size(                                   #     equity, current_risk_percent, entry_price_est, sl_price, ...                                  # )                                               # Need to store trade PNL in recent_trades list after each close.                                   ```                                                                                                 ```python                                         # Suggestion 9: Confirmation Supertrend Exit Condition                                              # Add config: USE_CONFIRM_ST_EXIT = False         # Inside trade_logic, add to exit signal checks:  if CONFIG.use_confirm_st_exit:                        # Check if confirmation ST flips against the position                                               if position_side == POSITION_SIDE_LONG and confirm_st_trend == -1.0:
        if not exit_reason: # Only override if primary ST hasn't already triggered exit                          exit_reason = "Confirm ST Exit Long"              logger.info("Confirmation ST flipped against LONG position.")                              elif position_side == POSITION_SIDE_SHORT and confirm_st_trend == 1.0:
        if not exit_reason: # Only override if primary ST hasn't already triggered exit                          exit_reason = "Confirm ST Exit Short"             logger.info("Confirmation ST flipped against SHORT position.")                                                                           # The rest of the exit logic using exit_reason remains the same.                                    # if exit_reason:                                 #    logger.warning(...)                          #    close_position(...)                          ```

```python                                         # Suggestion 10: Exponential Backoff for Rate Limit Errors                                          # Modify the main loop's exception handling for RateLimitExceeded                                   # Add state variable: rate_limit_backoff_factor = 1                                                 except ccxt.RateLimitExceeded as e:                   global rate_limit_backoff_factor                  sleep_duration = CONFIG.sleep_seconds * rate_limit_backoff_factor                                   max_sleep = 300 # Set a maximum backoff sleep time                                                  sleep_duration = min(sleep_duration, max_sleep)                                                     logger.warning(f"{Fore.YELLOW}WARNING: Rate limit exceeded: {e}. Backing off. Sleeping for {sleep_duration} seconds (Factor: {rate_limit_backoff_factor}).{Style.RESET_ALL}")                           time.sleep(sleep_duration)                        rate_limit_backoff_factor *= 2 # Double the factor for next time                                    # Optional: Reset factor after a period of no rate limit errors                                                                                   # Add logic to reset the backoff factor after successful cycles                                     # Inside the main loop, after a successful trade_logic call and sleep:
# if rate_limit_backoff_factor > 1:               #     logger.debug("Resetting rate limit backoff factor.")                                          #     rate_limit_backoff_factor = 1

#/usr/bin/env python
# -*- coding: utf-8 -*-

# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██║   ██║███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v10.1.2 (Reforged Config & Arcane Clarity)
# Conjures high-frequency trades on Bybit Futures with enhanced config, precision, V5 focus, and Termux integration.

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 10.1.2 (Reforged: Class-based Config, Enhanced Fill Confirm, Standardized SL/TP, Pyrmethus Style, Bug Fixes).

Features:
- Dual Supertrend strategy with confirmation.
- ATR for volatility measurement and SL/TP calculation.
- **CRITICAL SAFETY UPGRADE:** Implements exchange-native Stop-Loss and Take-Profit
  orders (both using `stopMarket` type) immediately after entry confirmation,
  based on actual fill price. Uses `fetch_order` primarily for faster confirmation.
- **Includes necessary 'triggerDirection' parameter for Bybit V5 API.**
- Optional Volume spike analysis for entry confirmation.
- Optional Order book pressure analysis for entry confirmation.
- **Enhanced Risk Management:**
    - Risk-based position sizing with margin checks.
    - Checks against exchange minimum order amount and cost *before* placing orders.
    - Caps position size based on `MAX_ORDER_USDT_AMOUNT`.
- **Reforged Configuration:** Uses a dedicated `Config` class for better organization and validation.
- Termux SMS alerts for critical events (with Termux:API check).
- Robust error handling and logging with vibrant Neon color support via Colorama.
- Graceful shutdown on KeyboardInterrupt with position closing attempt.
- Stricter position detection logic (targeting Bybit V5 API).
- **Decimal Precision:** Uses Decimal for critical financial calculations.
- **Bug Fix:** Correctly handles boolean types from pandas/numpy for signal processing.

Disclaimer:
- **EXTREME RISK**: Arcane energies are volatile. Educational purposes ONLY. High-risk. Use at own absolute risk.
- **EXCHANGE-NATIVE SL/TP:** Relies on exchange-native orders. Subject to exchange performance, slippage, API reliability.
- Parameter Sensitivity: Requires significant tuning and testing in the astral plane (testnet).
- API Rate Limits: Monitor usage lest the exchange spirits grow wary.
- Slippage: Market orders are prone to slippage in turbulent ether.
- Test Thoroughly: **DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.**
- Termux Dependency: Requires Termux:API for SMS communication scrolls. Ensure `pkg install termux-api`.
- API Changes: Exchange APIs (like Bybit V5) can change. Ensure CCXT is updated.

**Installation:**
pip install ccxt pandas pandas_ta python-dotenv colorama # termux-api (if using Termux for SMS)
"""

# Standard Library Imports - The Foundational Runes
import contextlib
import logging
import os
import shutil  # For checking command existence
import subprocess  # For Termux API calls
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast, Final

# Third-party Libraries - Summoned Essences
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta  # type: ignore[import]
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    from dotenv import load_dotenv
except ImportError as e:
    missing_pkg = e.name
    # Use Colorama's raw codes here as it might not be initialized yet
    print(f"\033[91m\033[1mCRITICAL ERROR: Missing required Python package: '{missing_pkg}'.\033[0m")
    print(f"\033[93mPlease install it using: pip install {missing_pkg}\033[0m")
    sys.exit(1)

# --- Initializations - Preparing the Ritual Chamber ---
colorama_init(autoreset=True)  # Activate Colorama's magic
load_dotenv()  # Load secrets from the hidden .env scroll
# Set Decimal precision high enough for crypto calculations
# Bybit USDT perps typically have price precision up to 4-6 decimals,
# and quantity precision up to 3-8 decimals. 18 should be safe.
getcontext().prec = 18

# --- Constants ---

# --- String Constants ---
# Dictionary Keys / Internal Representations
SIDE_KEY: Final[str] = 'side'
QTY_KEY: Final[str] = 'qty'
ENTRY_PRICE_KEY: Final[str] = 'entry_price'
INFO_KEY: Final[str] = 'info'
SYMBOL_KEY: Final[str] = 'symbol'
ID_KEY: Final[str] = 'id'
AVG_PRICE_KEY: Final[str] = 'avgPrice'  # Bybit V5 raw field preferred
CONTRACTS_KEY: Final[str] = 'contracts'  # CCXT unified field
FILLED_KEY: Final[str] = 'filled'
COST_KEY: Final[str] = 'cost'
AVERAGE_KEY: Final[str] = 'average'  # CCXT unified field for fill price
TIMESTAMP_KEY: Final[str] = 'timestamp'
LAST_PRICE_KEY: Final[str] = 'last'
BIDS_KEY: Final[str] = 'bids'
ASKS_KEY: Final[str] = 'asks'
SPREAD_KEY: Final[str] = 'spread'
BEST_BID_KEY: Final[str] = 'best_bid'
BEST_ASK_KEY: Final[str] = 'best_ask'
BID_ASK_RATIO_KEY: Final[str] = 'bid_ask_ratio'
ATR_KEY: Final[str] = 'atr'
VOLUME_MA_KEY: Final[str] = 'volume_ma'
LAST_VOLUME_KEY: Final[str] = 'last_volume'
VOLUME_RATIO_KEY: Final[str] = 'volume_ratio'
STATUS_KEY: Final[str] = 'status'
PRICE_KEY: Final[str] = 'price'  # Fallback for average price
PRECISION_KEY: Final[str] = 'precision'
LIMITS_KEY: Final[str] = 'limits'
AMOUNT_KEY: Final[str] = 'amount'
MIN_KEY: Final[str] = 'min'
MAX_KEY: Final[str] = 'max'
MARKET_KEY: Final[str] = 'market'
BASE_KEY: Final[str] = 'base'
QUOTE_KEY: Final[str] = 'quote'
SETTLE_KEY: Final[str] = 'settle'
CONTRACT_KEY: Final[str] = 'contract'
SPOT_KEY: Final[str] = 'spot'
TYPE_KEY: Final[str] = 'type'
LINEAR_KEY: Final[str] = 'linear'
INVERSE_KEY: Final[str] = 'inverse'
TOTAL_KEY: Final[str] = 'total'
FREE_KEY: Final[str] = 'free'

# Order Sides / Position Sides
SIDE_BUY: Final[str] = 'buy'
SIDE_SELL: Final[str] = 'sell'
POSITION_SIDE_LONG: Final[str] = 'Long'    # Internal representation for long position
POSITION_SIDE_SHORT: Final[str] = 'Short'  # Internal representation for short position
POSITION_SIDE_NONE: Final[str] = 'None'    # Internal representation for no position / Bybit V5 side 'None'
BYBIT_SIDE_BUY: Final[str] = 'Buy'         # Bybit V5 API side
BYBIT_SIDE_SELL: Final[str] = 'Sell'       # Bybit V5 API side

# Order Types / Statuses / Params
ORDER_TYPE_MARKET: Final[str] = 'market'
ORDER_TYPE_STOP_MARKET: Final[str] = 'stopMarket'  # Used for both SL and TP conditional market orders
ORDER_STATUS_OPEN: Final[str] = 'open'
ORDER_STATUS_CLOSED: Final[str] = 'closed'
ORDER_STATUS_CANCELED: Final[str] = 'canceled'  # Note: CCXT might use 'cancelled' or 'canceled'
ORDER_STATUS_REJECTED: Final[str] = 'rejected'
ORDER_STATUS_EXPIRED: Final[str] = 'expired'
PARAM_REDUCE_ONLY: Final[str] = 'reduce_only'  # CCXT standard param name
PARAM_STOP_PRICE: Final[str] = 'stopPrice'  # CCXT standard param name for trigger price
PARAM_TRIGGER_DIRECTION: Final[str] = 'triggerDirection'  # Bybit V5 specific for conditional orders (1=above, 2=below)
PARAM_CATEGORY: Final[str] = 'category'  # Bybit V5 specific for linear/inverse
PARAM_SETTLE_COIN: Final[str] = 'settleCoin' # Bybit V5 param for balance/position filtering
PARAM_POSITION_IDX: Final[str] = 'positionIdx' # Bybit V5 raw field for hedge/one-way mode (0=One-Way, 1=Buy Hedge, 2=Sell Hedge)
PARAM_SIZE: Final[str] = 'size' # Bybit V5 raw field for position size

# Currencies / Default Values
USDT_SYMBOL: Final[str] = "USDT"
DEFAULT_SYMBOL: Final[str] = "BTC/USDT:USDT"
DEFAULT_INTERVAL: Final[str] = "1m"
DEFAULT_LEVERAGE: Final[int] = 10
DEFAULT_SLEEP_SECONDS: Final[int] = 10
DEFAULT_RISK_PERCENT: Final[Decimal] = Decimal("0.005") # 0.5%
DEFAULT_SL_MULT: Final[Decimal] = Decimal("1.5")
DEFAULT_TP_MULT: Final[Decimal] = Decimal("2.0")
DEFAULT_MAX_ORDER_USDT: Final[Decimal] = Decimal("500.0")
DEFAULT_MARGIN_BUFFER: Final[Decimal] = Decimal("1.05") # 5% buffer
DEFAULT_ST_LEN: Final[int] = 7
DEFAULT_ST_MULT: Final[Decimal] = Decimal("2.5")
DEFAULT_CONF_ST_LEN: Final[int] = 5
DEFAULT_CONF_ST_MULT: Final[Decimal] = Decimal("2.0")
DEFAULT_VOL_MA: Final[int] = 20
DEFAULT_VOL_SPIKE: Final[Decimal] = Decimal("1.5")
DEFAULT_REQ_VOL_SPIKE: Final[bool] = True
DEFAULT_OB_DEPTH: Final[int] = 10
DEFAULT_OB_RATIO_L: Final[Decimal] = Decimal("1.2")
DEFAULT_OB_RATIO_S: Final[Decimal] = Decimal("0.8")
DEFAULT_FETCH_OB_CYCLE: Final[bool] = False
DEFAULT_USE_OB_CONFIRM: Final[bool] = True
DEFAULT_ATR_PERIOD: Final[int] = 14
DEFAULT_ENABLE_SMS: Final[bool] = False
DEFAULT_SMS_TIMEOUT: Final[int] = 30
DEFAULT_RECV_WINDOW: Final[int] = 10000
DEFAULT_RETRY_COUNT: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[int] = 1
DEFAULT_API_BUFFER: Final[int] = 5
DEFAULT_POS_EPSILON: Final[Decimal] = Decimal("1e-9") # Small value for float comparisons
DEFAULT_POST_CLOSE_DELAY: Final[int] = 2
DEFAULT_POST_ENTRY_DELAY: Final[Decimal] = Decimal("1.0")
DEFAULT_FETCH_ORDER_RETRIES: Final[int] = 5
DEFAULT_FETCH_ORDER_DELAY: Final[Decimal] = Decimal("0.5")
DEFAULT_FILL_LOOKBACK: Final[int] = 600 # Seconds for fetch_closed_orders fallback
DEFAULT_EMERGENCY_CLOSE: Final[bool] = True

# --- Logger Setup - The Oracle's Voice ---
LOGGING_LEVEL_STR: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL: int = getattr(logging, LOGGING_LEVEL_STR, logging.INFO)

# Custom Log Level for Success
SUCCESS_LEVEL: Final[int] = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a 'success' log level method."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Bind the new method to the Logger class
logging.Logger.success = log_success # type: ignore[attr-defined]

# Basic configuration first
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # logging.FileHandler("scalp_bot_v10.1.log"), # Optional: Log to file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)
logger: logging.Logger = logging.getLogger(__name__)

# Apply colors if outputting to a TTY (like Termux)
if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
    # Apply Pyrmethus colors
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}{Style.DIM}{logging.getLevelName(logging.DEBUG)}{Style.RESET_ALL}")  # Dim Cyan for Debug
    logging.addLevelName(logging.INFO, f"{Fore.BLUE}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")  # Blue for Info
    logging.addLevelName(SUCCESS_LEVEL, f"{Fore.MAGENTA}{Style.BRIGHT}{logging.getLevelName(SUCCESS_LEVEL)}{Style.RESET_ALL}")  # Bright Magenta for Success
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}{Style.BRIGHT}{logging.getLevelName(logging.WARNING)}{Style.RESET_ALL}")  # Bright Yellow for Warning
    logging.addLevelName(logging.ERROR, f"{Fore.RED}{Style.BRIGHT}{logging.getLevelName(logging.ERROR)}{Style.RESET_ALL}")  # Bright Red for Error
    logging.addLevelName(logging.CRITICAL, f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}{logging.getLevelName(logging.CRITICAL)}{Style.RESET_ALL}")  # White on Red for Critical
else:
    # Avoid color codes if not a TTY
    logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")  # Ensure level name exists without color


# --- Configuration Class - Defining the Spell's Parameters ---
class Config:
    """Loads, validates, and stores configuration parameters with arcane precision."""
    def __init__(self) -> None:
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        self._valid = True  # Track overall validity

        # --- API Credentials (Required) ---
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", None, str, required=True, color=Fore.RED)
        self.api_secret: Optional[str] = self._get_env("BYBIT_API_SECRET", None, str, required=True, color=Fore.RED)
        if not self.api_key or not self.api_secret: self._valid = False

        # --- Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", DEFAULT_SYMBOL, str, color=Fore.YELLOW)
        # Basic validation for symbol format (allow flexibility but check basics)
        if not ('/' in self.symbol and ':' in self.symbol):
            logger.warning(f"CONFIG WARNING: SYMBOL format '{self.symbol}' might be incorrect. Expected format like 'BASE/QUOTE:SETTLE'.")
        self.interval: str = self._get_env("INTERVAL", DEFAULT_INTERVAL, str, color=Fore.YELLOW)
        self.leverage: int = self._get_env("LEVERAGE", DEFAULT_LEVERAGE, int, color=Fore.YELLOW)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", DEFAULT_SLEEP_SECONDS, int, color=Fore.YELLOW)
        if self.leverage <= 0:
            logger.critical(f"CRITICAL CONFIG: LEVERAGE must be positive, got: {self.leverage}")
            self._valid = False
        if self.sleep_seconds <= 0:
            logger.warning(f"CONFIG WARNING: SLEEP_SECONDS ({self.sleep_seconds}) invalid. Setting to 1.")
            self.sleep_seconds = 1

        # --- Risk Management (CRITICAL) ---
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", DEFAULT_RISK_PERCENT, Decimal, color=Fore.GREEN)
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", DEFAULT_SL_MULT, Decimal, color=Fore.GREEN)
        self.atr_take_profit_multiplier: Decimal = self._get_env("ATR_TAKE_PROFIT_MULTIPLIER", DEFAULT_TP_MULT, Decimal, color=Fore.GREEN)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", DEFAULT_MAX_ORDER_USDT, Decimal, color=Fore.GREEN)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", DEFAULT_MARGIN_BUFFER, Decimal, color=Fore.GREEN)
        if not (Decimal(0) < self.risk_per_trade_percentage < Decimal(1)):
            logger.critical(f"CRITICAL CONFIG: RISK_PER_TRADE_PERCENTAGE must be between 0 and 1 (exclusive), got: {self.risk_per_trade_percentage}")
            self._valid = False
        if self.atr_stop_loss_multiplier <= 0:
            logger.warning(f"CONFIG WARNING: ATR_STOP_LOSS_MULTIPLIER ({self.atr_stop_loss_multiplier}) should be positive.")
        if self.atr_take_profit_multiplier <= 0:
            logger.warning(f"CONFIG WARNING: ATR_TAKE_PROFIT_MULTIPLIER ({self.atr_take_profit_multiplier}) should be positive.")
        if self.max_order_usdt_amount <= 0:
            logger.warning(f"CONFIG WARNING: MAX_ORDER_USDT_AMOUNT ({self.max_order_usdt_amount}) should be positive.")
        if self.required_margin_buffer < 1:
            logger.warning(f"CONFIG WARNING: REQUIRED_MARGIN_BUFFER ({self.required_margin_buffer}) is less than 1. Margin checks might be ineffective.")

        # --- Supertrend Indicator Parameters ---
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", DEFAULT_ST_LEN, int, color=Fore.CYAN)
        # Fetch as Decimal for consistency, cast to float for pandas_ta
        self.st_multiplier: float = float(self._get_env("ST_MULTIPLIER", DEFAULT_ST_MULT, Decimal, color=Fore.CYAN))
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", DEFAULT_CONF_ST_LEN, int, color=Fore.CYAN)
        self.confirm_st_multiplier: float = float(self._get_env("CONFIRM_ST_MULTIPLIER", DEFAULT_CONF_ST_MULT, Decimal, color=Fore.CYAN))
        if self.st_atr_length <= 0 or self.confirm_st_atr_length <= 0:
            logger.warning("CONFIG WARNING: Supertrend ATR length(s) are zero or negative.")
        if self.st_multiplier <= 0 or self.confirm_st_multiplier <= 0:
            logger.warning("CONFIG WARNING: Supertrend multiplier(s) are zero or negative.")

        # --- Volume Analysis Parameters ---
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", DEFAULT_VOL_MA, int, color=Fore.YELLOW)
        self.volume_spike_threshold: Decimal = self._get_env("VOLUME_SPIKE_THRESHOLD", DEFAULT_VOL_SPIKE, Decimal, color=Fore.YELLOW)
        self.require_volume_spike_for_entry: bool = self._get_env("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", DEFAULT_REQ_VOL_SPIKE, bool, color=Fore.YELLOW)
        if self.volume_ma_period <= 0:
            logger.warning("CONFIG WARNING: VOLUME_MA_PERIOD is zero or negative.")
        if self.volume_spike_threshold <= 0:
            logger.warning("CONFIG WARNING: VOLUME_SPIKE_THRESHOLD should be positive.")

        # --- Order Book Analysis Parameters ---
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", DEFAULT_OB_DEPTH, int, color=Fore.YELLOW)
        self.order_book_ratio_threshold_long: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_LONG", DEFAULT_OB_RATIO_L, Decimal, color=Fore.YELLOW)
        self.order_book_ratio_threshold_short: Decimal = self._get_env("ORDER_BOOK_RATIO_THRESHOLD_SHORT", DEFAULT_OB_RATIO_S, Decimal, color=Fore.YELLOW)
        self.fetch_order_book_per_cycle: bool = self._get_env("FETCH_ORDER_BOOK_PER_CYCLE", DEFAULT_FETCH_OB_CYCLE, bool, color=Fore.YELLOW)
        self.use_ob_confirm: bool = self._get_env("USE_OB_CONFIRM", DEFAULT_USE_OB_CONFIRM, bool, color=Fore.YELLOW)
        if self.order_book_depth <= 0:
            logger.warning("CONFIG WARNING: ORDER_BOOK_DEPTH should be positive.")

        # --- ATR Calculation Parameter (for SL/TP) ---
        self.atr_calculation_period: int = self._get_env("ATR_CALCULATION_PERIOD", DEFAULT_ATR_PERIOD, int, color=Fore.GREEN)
        if self.atr_calculation_period <= 0:
            logger.warning("CONFIG WARNING: ATR_CALCULATION_PERIOD is zero or negative.")

        # --- Termux SMS Alert Configuration ---
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", DEFAULT_ENABLE_SMS, bool, color=Fore.MAGENTA)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, str, color=Fore.MAGENTA)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", DEFAULT_SMS_TIMEOUT, int, color=Fore.MAGENTA)
        if self.enable_sms_alerts and not self.sms_recipient_number:
            logger.warning("CONFIG WARNING: SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")
        if self.sms_timeout_seconds <= 0:
            logger.warning(f"CONFIG WARNING: SMS_TIMEOUT_SECONDS ({self.sms_timeout_seconds}) invalid. Setting to 10.")
            self.sms_timeout_seconds = 10

        # --- CCXT / API Parameters ---
        self.default_recv_window: int = self._get_env("RECV_WINDOW", DEFAULT_RECV_WINDOW, int, color=Fore.WHITE)
        # Bybit V5 L2 OB limit can be 1, 50, 200. Fetch 50 if depth <= 50, else 200.
        self.order_book_fetch_limit: int = 50 if self.order_book_depth <= 50 else 200

        # --- Internal Constants & Behavior ---
        self.retry_count: int = self._get_env("RETRY_COUNT", DEFAULT_RETRY_COUNT, int, color=Fore.WHITE)
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, int, color=Fore.WHITE)
        self.api_fetch_limit_buffer: int = self._get_env("API_FETCH_LIMIT_BUFFER", DEFAULT_API_BUFFER, int, color=Fore.WHITE)
        self.position_qty_epsilon: Decimal = self._get_env("POSITION_QTY_EPSILON", DEFAULT_POS_EPSILON, Decimal, color=Fore.WHITE)
        self.post_close_delay_seconds: int = self._get_env("POST_CLOSE_DELAY_SECONDS", DEFAULT_POST_CLOSE_DELAY, int, color=Fore.WHITE)
        # Fetch as Decimal for consistency, cast to float for time.sleep
        self.post_entry_delay_seconds: float = float(self._get_env("POST_ENTRY_DELAY_SECONDS", DEFAULT_POST_ENTRY_DELAY, Decimal, color=Fore.WHITE))
        self.fetch_order_status_retries: int = self._get_env("FETCH_ORDER_STATUS_RETRIES", DEFAULT_FETCH_ORDER_RETRIES, int, color=Fore.WHITE)
        self.fetch_order_status_delay: float = float(self._get_env("FETCH_ORDER_STATUS_DELAY", DEFAULT_FETCH_ORDER_DELAY, Decimal, color=Fore.WHITE))
        self.confirm_fill_lookback_seconds: int = self._get_env("CONFIRM_FILL_LOOKBACK_SECONDS", DEFAULT_FILL_LOOKBACK, int, color=Fore.WHITE) # Lookback for fetch_closed_orders fallback
        self.emergency_close_on_sl_fail: bool = self._get_env("EMERGENCY_CLOSE_ON_SL_FAIL", DEFAULT_EMERGENCY_CLOSE, bool, color=Fore.RED) # Attempt emergency close if SL order fails

        # --- Final Validation Check ---
        if not self._valid:
            logger.critical(f"{Back.RED}{Fore.WHITE}--- Configuration validation FAILED. Cannot proceed. ---{Style.RESET_ALL}")
            raise ValueError("Critical configuration validation failed.")
        else:
            logger.success(f"{Fore.GREEN}{Style.BRIGHT}--- Configuration Runes Summoned and Verified ---{Style.RESET_ALL}")

    def _get_env(self, var_name: str, default: Any, expected_type: Type, required: bool = False, color: str = Fore.WHITE) -> Any:
        """Gets an environment variable, casts type (incl. defaults), logs, handles errors.
        Handles str, int, float, bool, and Decimal types. Sets internal validity flag on critical errors.
        """
        value_str = os.getenv(var_name)
        source = "environment" if value_str is not None else "default"
        value_to_process: Any = value_str if value_str is not None else default

        log_val_str = f"'{value_str}'" if source == "environment" else f"(Default: '{default}')"
        logger.debug(f"{color}Summoning {var_name}: {log_val_str}{Style.RESET_ALL}")

        if value_to_process is None:
            if required:
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required config '{var_name}' is missing and has no default.{Style.RESET_ALL}")
                self._valid = False # Mark config as invalid
                # Raise error immediately if required and no value/default
                raise ValueError(f"Required environment variable '{var_name}' not set and no default provided.")
            return None  # Return None if not required and no value/default

        try:
            if expected_type == bool:
                # Explicitly check for string representations of True/False
                if isinstance(value_to_process, str):
                    return value_to_process.lower() in ('true', '1', 't', 'yes', 'y')
                # If not a string, try standard Python truthiness
                return bool(value_to_process)
            elif expected_type == Decimal:
                # Ensure input is string for Decimal constructor for reliable conversion
                return Decimal(str(value_to_process))
            else:
                # Handle int, float, str directly
                return expected_type(value_to_process)
        except (ValueError, TypeError, InvalidOperation) as e:
            env_val_disp = f"'{value_str}'" if value_str is not None else "(Not Set)"
            logger.error(
                f"{Fore.RED}Config Error: Invalid type/value for {var_name}={env_val_disp} (Source: {source}). "
                f"Expected {expected_type.__name__}. Error: {e}. Trying default '{default}'...{Style.RESET_ALL}"
            )
            # Try casting the default again if the primary value failed
            if default is None:
                if required:  # Should have been caught above, but defensive check
                     logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Required config '{var_name}' failed casting and has no default.{Style.RESET_ALL}")
                     self._valid = False
                     raise ValueError(f"Required env var '{var_name}' failed casting and has no valid default.")
                return None
            try:
                if expected_type == bool:
                    if isinstance(default, str): return default.lower() in ('true', '1', 't', 'yes', 'y')
                    return bool(default)
                elif expected_type == Decimal: return Decimal(str(default))
                else: return expected_type(default)
            except (ValueError, TypeError, InvalidOperation) as e_default:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL CONFIG: Default '{default}' for {var_name} is also incompatible "
                    f"with type {expected_type.__name__}. Error: {e_default}. Cannot proceed.{Style.RESET_ALL}"
                )
                self._valid = False
                raise ValueError(f"Configuration error: Cannot cast value or default for key '{var_name}' to {expected_type.__name__}.")


# --- Global Objects - Instantiated Arcana ---
try:
    CONFIG = Config()  # Forge the configuration object
except ValueError as config_err:
    # Error already logged within Config init or _get_env
    # Attempt to send SMS even if config partially failed (if SMS settings were read)
    # Check if CONFIG object exists and has necessary attributes before trying to send SMS
    # Note: This check is difficult because CONFIG might be partially initialized or not at all.
    # A simpler approach is to define send_sms_alert earlier or handle this outside the class.
    # For now, we rely on the logging and exit.
    # If SMS on critical failure is essential, `send_sms_alert` needs to be defined
    # *before* Config instantiation, or handle the potential partial config state carefully.
    # Minimal SMS attempt (assuming send_sms_alert is defined):
    # if 'send_sms_alert' in globals():
    #     try:
    #         send_sms_alert(f"[ScalpBot] CRITICAL: Config validation FAILED: {config_err}. Bot stopped.")
    #     except Exception as sms_err:
    #         logger.error(f"Failed to send critical config failure SMS: {sms_err}")
    sys.exit(1)

# --- Termux SMS Alert Function - Sending Whispers ---
_termux_sms_command_exists: Optional[bool] = None  # Cache check result


def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API, a whisper through the digital veil."""
    global _termux_sms_command_exists

    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled by configuration.")
        return False

    # Check for command existence only once
    if _termux_sms_command_exists is None:
        _termux_sms_command_exists = shutil.which('termux-sms-send') is not None
        if not _termux_sms_command_exists:
             logger.warning(f"{Fore.YELLOW}SMS failed: 'termux-sms-send' command not found. Ensure Termux:API is installed (`pkg install termux-api`) and configured.{Style.RESET_ALL}")

    if not _termux_sms_command_exists:
        return False  # Don't proceed if command is missing

    if not CONFIG.sms_recipient_number:
        logger.warning(f"{Fore.YELLOW}SMS alerts enabled, but SMS_RECIPIENT_NUMBER rune is missing.{Style.RESET_ALL}")
        return False

    try:
        # Prepare the command spell
        command: List[str] = ['termux-sms-send', '-n', CONFIG.sms_recipient_number, message]
        logger.info(f"{Fore.MAGENTA}Dispatching SMS whisper to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}")
        # Execute the spell via subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds)
        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS whisper dispatched successfully.{Style.RESET_ALL}")
            return True
        else:
            logger.error(f"{Fore.RED}SMS whisper failed. RC: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}")
            return False
    except FileNotFoundError:
        # This shouldn't happen due to the check above, but handle defensively
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' vanished unexpectedly?{Style.RESET_ALL}")
        _termux_sms_command_exists = False  # Update cache
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: Command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected disturbance: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False


# --- Exchange Initialization - Opening the Portal ---
def initialize_exchange() -> Optional[ccxt.Exchange]:
    """Initializes and returns the CCXT Bybit exchange instance, opening a portal."""
    logger.info(f"{Fore.BLUE}Opening portal to Bybit via CCXT...{Style.RESET_ALL}")
    # API keys already checked in Config, but double-check instance variables
    if not CONFIG.api_key or not CONFIG.api_secret:
         logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: API keys check failed during initialization.{Style.RESET_ALL}")
         return None

    try:
        exchange = ccxt.bybit(
            {
                'apiKey': CONFIG.api_key,
                'secret': CONFIG.api_secret,
                'enableRateLimit': True,  # Built-in rate limiting
                'options': {
                    'adjustForTimeDifference': True,  # Adjust for clock skew
                    'recvWindow': CONFIG.default_recv_window,  # Increase if timestamp errors occur
                    'defaultType': 'swap',  # Explicitly default to swap markets (linear/inverse determined by symbol)
                    'warnOnFetchOpenOrdersWithoutSymbol': False,  # Suppress common warning
                    'brokerId': 'Pyrmethus_Scalp_v10.1',  # Optional: Identify the bot
                    'defaultMarginMode': 'isolated', # Explicitly set default margin mode (leverage setting might override)
                    'createMarketBuyOrderRequiresPrice': False, # Bybit V5 doesn't require price for market buy
                    'fetchPositions': { # V5 specific options if needed
                        'category': 'linear', # Default category if market type cannot be determined
                    },
                    'fetchBalance': { # V5 specific options if needed
                        'accountType': 'UNIFIED', # Default account type if settle currency cannot be determined
                    }
                }
            }
        )
        # Explicitly set API version to v5 if CCXT doesn't default correctly (usually not needed)
        # exchange.set_sandbox_mode(False) # Ensure not in sandbox unless intended

    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to instantiate CCXT Bybit object: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: CCXT Instantiation Error: {type(e).__name__}. Bot stopped.")
        return None

    try:
        # Test connection and authentication by fetching markets and balance
        logger.debug("Loading market structures...")
        exchange.load_markets()
        logger.debug("Fetching balance (tests authentication)...")

        settle_currency = USDT_SYMBOL # Default
        account_type = 'UNIFIED' # Default for USDT settled
        try:
            market = exchange.market(CONFIG.symbol)
            settle_currency = market.get(SETTLE_KEY, USDT_SYMBOL) # Get settle currency (e.g., USDT)
            # Determine accountType based on settle currency (common mapping)
            # This might need adjustment based on user's specific Bybit account setup (UTA vs standard)
            # For simplicity, assume UNIFIED for USDT, CONTRACT otherwise. Might need refinement.
            # Bybit V5 generally uses 'UNIFIED' or 'CONTRACT'. Let's default to UNIFIED for broader compatibility.
            account_type = 'UNIFIED' # if settle_currency == USDT_SYMBOL else 'CONTRACT' # Simplified to UNIFIED
            logger.debug(f"Determined Settle Currency: {settle_currency}, Account Type for Balance: {account_type}")
        except (ccxt.BadSymbol, KeyError) as e:
            logger.warning(f"Could not determine settle currency/account type for {CONFIG.symbol} from market info: {e}. Defaulting to {USDT_SYMBOL}/{account_type} for balance check.")
        except Exception as e:
            logger.warning(f"Unexpected error getting market info for balance check: {e}. Defaulting to {USDT_SYMBOL}/{account_type}.")

        # Bybit V5 requires accountType for fetchBalance.
        balance_params = {'accountType': account_type}
        balance = exchange.fetch_balance(params=balance_params)

        total_settle = balance.get(TOTAL_KEY, {}).get(settle_currency, 'N/A')
        logger.debug(f"Initial balance fetched: {total_settle} {settle_currency}")
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Portal to Bybit Opened (LIVE SCALPING MODE - EXTREME CAUTION!).{Style.RESET_ALL}")
        send_sms_alert("[ScalpBot] Initialized successfully and authenticated.")
        return exchange

    except ccxt.AuthenticationError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Authentication failed: {e}. Check API key/secret and ensure IP whitelist (if used) is correct and API permissions are sufficient.{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot] CRITICAL: Authentication FAILED: {e}. Bot stopped.")
        return None
    except ccxt.NetworkError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Network error during initialization: {e}. Check internet connection and Bybit status.{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot] CRITICAL: Network Error on Init: {e}. Bot stopped.")
        return None
    except ccxt.ExchangeError as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Exchange error during initialization: {e}. Check Bybit status page or API documentation for details.{Style.RESET_ALL}")
        send_sms_alert(f"[ScalpBot] CRITICAL: Exchange Error on Init: {e}. Bot stopped.")
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error initializing exchange: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[ScalpBot] CRITICAL: Unexpected Init Error: {type(e).__name__}. Bot stopped.")
        return None


# --- Indicator Calculation Functions - Scrying the Market ---
def calculate_supertrend(
    df: pd.DataFrame, length: int, multiplier: float, prefix: str = ""
) -> pd.DataFrame:
     """Calculates the Supertrend indicator using the pandas_ta library."""
     required_input_cols: List[str] = ['high', 'low', 'close']
     col_prefix: str = f"{prefix}" if prefix else ""
     # Define target column names clearly
     target_st_val_col: str = f"{col_prefix}st_value"
     target_st_trend_col: str = f"{col_prefix}trend"
     target_st_long_flip_col: str = f"{col_prefix}st_long_flip"
     target_st_short_flip_col: str = f"{col_prefix}st_short_flip"
     target_cols: List[str] = [target_st_val_col, target_st_trend_col, target_st_long_flip_col, target_st_short_flip_col]

     # Define expected pandas_ta column names (adjust if pandas_ta version changes output)
     # Note: pandas_ta might format float multipliers differently (e.g., 2.5 -> '2.5' or '2_5').
     # This assumes the format f'{multiplier}' is used in the column name. Robust check needed.
     # Let's check for common variations.
     pta_mult_str_dot = str(multiplier)
     pta_mult_str_underscore = str(multiplier).replace('.', '_')
     pta_st_col_name_dot = f"SUPERT_{length}_{pta_mult_str_dot}"
     pta_st_col_name_underscore = f"SUPERT_{length}_{pta_mult_str_underscore}"
     pta_st_trend_col_dot = f"SUPERTd_{length}_{pta_mult_str_dot}"
     pta_st_trend_col_underscore = f"SUPERTd_{length}_{pta_mult_str_underscore}"
     pta_st_long_col_dot = f"SUPERTl_{length}_{pta_mult_str_dot}" # pandas_ta uses 'l' for long band
     pta_st_long_col_underscore = f"SUPERTl_{length}_{pta_mult_str_underscore}"
     pta_st_short_col_dot = f"SUPERTs_{length}_{pta_mult_str_dot}" # pandas_ta uses 's' for short band
     pta_st_short_col_underscore = f"SUPERTs_{length}_{pta_mult_str_underscore}"

     # Helper to find the actual column name used by pandas_ta
     def find_pta_col(df_cols: pd.Index, base_name: str, length: int, mult_str_dot: str, mult_str_underscore: str) -> Optional[str]:
         dot_name = f"{base_name}_{length}_{mult_str_dot}"
         underscore_name = f"{base_name}_{length}_{mult_str_underscore}"
         if dot_name in df_cols: return dot_name
         if underscore_name in df_cols: return underscore_name
         # Fallback: check if *any* column starts with the base name and length (less precise)
         prefix_check = f"{base_name}_{length}_"
         matches = [col for col in df_cols if col.startswith(prefix_check)]
         if len(matches) == 1:
             logger.debug(f"Found potential pandas_ta column '{matches[0]}' matching prefix '{prefix_check}'.")
             return matches[0]
         elif len(matches) > 1:
             logger.warning(f"Multiple pandas_ta columns found matching prefix '{prefix_check}': {matches}. Cannot reliably identify.")
         return None

     if df is None or df.empty:
         logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}Supertrend): Input DataFrame is empty.{Style.RESET_ALL}")
         return pd.DataFrame(columns=target_cols) # Return empty DF with target columns

     if not all(c in df.columns for c in required_input_cols):
         logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}Supertrend): Input DataFrame is missing required columns {required_input_cols}.{Style.RESET_ALL}")
         for col in target_cols: df[col] = pd.NA # Add NA columns to existing df
         return df

     if len(df) < length:
         logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}Supertrend): DataFrame length ({len(df)}) is less than ST period ({length}). Filling with NA.{Style.RESET_ALL}")
         for col in target_cols: df[col] = pd.NA
         return df

     try:
         logger.debug(f"Scrying ({col_prefix}ST): Calculating with length={length}, multiplier={multiplier}")
         # Ensure input columns are numeric
         for col in required_input_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')
         if df[required_input_cols].isnull().values.any():
              logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): NaNs found in input data before calculation. Results may be affected.{Style.RESET_ALL}")
              # Optionally fill NaNs here if appropriate (e.g., df.ffill(inplace=True))

         # Calculate Supertrend using pandas_ta
         df.ta.supertrend(length=length, multiplier=multiplier, append=True)

         # Find the actual column names generated by pandas_ta
         pta_st_col_name = find_pta_col(df.columns, "SUPERT", length, pta_mult_str_dot, pta_mult_str_underscore)
         pta_st_trend_col = find_pta_col(df.columns, "SUPERTd", length, pta_mult_str_dot, pta_mult_str_underscore)
         pta_st_long_col = find_pta_col(df.columns, "SUPERTl", length, pta_mult_str_dot, pta_mult_str_underscore)
         pta_st_short_col = find_pta_col(df.columns, "SUPERTs", length, pta_mult_str_dot, pta_mult_str_underscore)

         # Check if essential columns were found
         if not pta_st_col_name or not pta_st_trend_col:
              logger.warning(f"{Fore.YELLOW}Scrying ({col_prefix}ST): pandas_ta did not create expected essential columns (SUPERT/SUPERTd) for L={length}, M={multiplier}. Calculation might be incomplete.{Style.RESET_ALL}")
              for col in target_cols: df[col] = pd.NA # Ensure target columns exist with NA
              return df # Return potentially incomplete dataframe

         # Convert potentially generated columns to numeric, coercing errors
         cols_to_convert = [c for c in [pta_st_col_name, pta_st_trend_col, pta_st_long_col, pta_st_short_col] if c is not None]
         for col in cols_to_convert:
             df[col] = pd.to_numeric(df[col], errors='coerce')

         # Rename and process
         df.rename(columns={pta_st_col_name: target_st_val_col, pta_st_trend_col: target_st_trend_col}, inplace=True)

         # Check if the essential trend column exists and is valid after rename/conversion
         if target_st_trend_col not in df.columns or df[target_st_trend_col].isnull().all():
             logger.error(f"{Fore.RED}Scrying ({col_prefix}ST): Failed to obtain valid trend data ('{target_st_trend_col}') after calculation/rename.{Style.RESET_ALL}")
             for col in target_cols: df[col] = pd.NA # Ensure all target cols are NA
             return df

         # Calculate flip signals based on trend change
         prev_trend_direction = df[target_st_trend_col].shift(1)
         # Long flip: Previous was Down (-1) and Current is Up (1)
         df[target_st_long_flip_col] = (prev_trend_direction == -1) & (df[target_st_trend_col] == 1)
         # Short flip: Previous was Up (1) and Current is Down (-1)
         df[target_st_short_flip_col] = (prev_trend_direction == 1) & (df[target_st_trend_col] == -1)

         # Ensure boolean type and fill NA (especially first row) with False
         # CRITICAL FIX: Use .astype(bool) to convert from numpy.bool_ to native bool
         df[target_st_long_flip_col] = df[target_st_long_flip_col].fillna(False).astype(bool)
         df[target_st_short_flip_col] = df[target_st_short_flip_col].fillna(False).astype(bool)

         # Clean up intermediate columns generated by pandas_ta
         cols_to_drop = [c for c in [pta_st_long_col, pta_st_short_col] if c is not None]
         # Also drop any other potential intermediates if names were found
         if pta_st_col_name != target_st_val_col: cols_to_drop.append(pta_st_col_name)
         if pta_st_trend_col != target_st_trend_col: cols_to_drop.append(pta_st_trend_col)
         # Ensure we don't try to drop columns that weren't created or already dropped/renamed
         cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
         if cols_to_drop_existing:
             df.drop(columns=list(set(cols_to_drop_existing)), errors='ignore', inplace=True)

         # Log last candle result
         if not df.empty and target_st_trend_col in df.columns and target_st_val_col in df.columns:
             last_trend_val = df[target_st_trend_col].iloc[-1] if pd.notna(df[target_st_trend_col].iloc[-1]) else None
             last_st_val = df[target_st_val_col].iloc[-1] if pd.notna(df[target_st_val_col].iloc[-1]) else float('nan')
             last_trend_str = 'Up' if last_trend_val == 1 else 'Down' if last_trend_val == -1 else 'N/A'
             trend_color = Fore.GREEN if last_trend_str == 'Up' else Fore.RED if last_trend_str == 'Down' else Fore.WHITE
             logger.debug(f"Scrying ({col_prefix}ST({length}, {multiplier})): Last Trend={trend_color}{last_trend_str}{Style.RESET_ALL}, Last Value={last_st_val:.4f}")
         elif df.empty:
             logger.debug(f"Scrying ({col_prefix}ST): DataFrame became empty during processing.")
         else:
             logger.debug(f"Scrying ({col_prefix}ST): Could not log last value (required columns missing).")

     except (KeyError, AttributeError, Exception) as e:
         logger.error(f"{Fore.RED}Scrying ({col_prefix}Supertrend): Error during calculation: {e}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())
         # Ensure target columns exist with NA on error
         for col in target_cols: df[col] = pd.NA
     return df


def analyze_volume_atr(
     df: pd.DataFrame, atr_len: int, vol_ma_len: int
) -> Dict[str, Optional[Decimal]]:
     """Calculates ATR, Volume MA, and checks for volume spikes. Returns Decimals."""
     results: Dict[str, Optional[Decimal]] = {ATR_KEY: None, VOLUME_MA_KEY: None, LAST_VOLUME_KEY: None, VOLUME_RATIO_KEY: None}
     required_cols: List[str] = ['high', 'low', 'close', 'volume']

     if df is None or df.empty:
         logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Input DataFrame is empty.{Style.RESET_ALL}")
         return results
     if not all(c in df.columns for c in required_cols):
         logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): Input DataFrame is missing required columns {required_cols}.{Style.RESET_ALL}")
         return results

     min_len = max(atr_len, vol_ma_len, 1) # Need at least 1 row for volume, more for indicators
     if len(df) < min_len:
           logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): DataFrame length ({len(df)}) < required ({min_len}) for ATR({atr_len})/VolMA({vol_ma_len}).{Style.RESET_ALL}")
           return results

     try:
         # Ensure numeric types, coercing errors to NaN
         for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
         if df[required_cols].isnull().values.any(): logger.warning(f"{Fore.YELLOW}Scrying (Vol/ATR): NaNs found in input data after coercion. Results may be inaccurate.{Style.RESET_ALL}")

         # Calculate ATR using pandas_ta
         atr_col = f"ATRr_{atr_len}" # Default ATR column name from pandas_ta
         df.ta.atr(length=atr_len, append=True)
         if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]):
            try:
                # Convert to string first for Decimal robustness
                results[ATR_KEY] = Decimal(str(df[atr_col].iloc[-1]))
            except (InvalidOperation, ValueError, TypeError):
                logger.warning(f"{Fore.YELLOW}Scrying (ATR): Invalid Decimal value for ATR: {df[atr_col].iloc[-1]}. Setting ATR to None.{Style.RESET_ALL}")
                results[ATR_KEY] = None # Explicitly set to None on conversion error
         else:
             logger.warning(f"{Fore.YELLOW}Scrying: Failed to calculate valid ATR({atr_len}). Column '{atr_col}' missing or last value is NaN.{Style.RESET_ALL}")
         # Clean up ATR column if it exists
         if atr_col in df.columns: df.drop(columns=[atr_col], errors='ignore', inplace=True)

         # Calculate Volume MA
         volume_ma_col = f"volume_ma_{vol_ma_len}"
         # Use min_periods to get a value even if window isn't full, but be aware of implications
         # Using max(1, vol_ma_len // 2) requires at least half the period or 1 data point.
         df[volume_ma_col] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()

         if pd.notna(df[volume_ma_col].iloc[-1]) and pd.notna(df['volume'].iloc[-1]):
            try:
                # Convert to string first for Decimal robustness
                results[VOLUME_MA_KEY] = Decimal(str(df[volume_ma_col].iloc[-1]))
                results[LAST_VOLUME_KEY] = Decimal(str(df['volume'].iloc[-1]))
            except (InvalidOperation, ValueError, TypeError):
                 logger.warning(f"{Fore.YELLOW}Scrying (Vol): Invalid Decimal value for Volume/MA. Vol: {df['volume'].iloc[-1]}, MA: {df[volume_ma_col].iloc[-1]}. Setting to None.{Style.RESET_ALL}")
                 results[VOLUME_MA_KEY] = None
                 results[LAST_VOLUME_KEY] = None

            # Calculate Volume Ratio
            # Ensure both values are valid Decimals and MA is not effectively zero
            if results[VOLUME_MA_KEY] is not None and results[VOLUME_MA_KEY] > CONFIG.position_qty_epsilon and results[LAST_VOLUME_KEY] is not None:
                try:
                    results[VOLUME_RATIO_KEY] = (results[LAST_VOLUME_KEY] / results[VOLUME_MA_KEY]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                except (InvalidOperation, ZeroDivisionError): # Catch division by zero explicitly
                    logger.warning(f"{Fore.YELLOW}Scrying (Vol): Division error for ratio. LastVol={results[LAST_VOLUME_KEY]}, VolMA={results[VOLUME_MA_KEY]}. Setting ratio to None.{Style.RESET_ALL}")
                    results[VOLUME_RATIO_KEY] = None
            else:
                results[VOLUME_RATIO_KEY] = None
                logger.debug(f"Scrying (Vol): Ratio calc skipped (LastVol={results.get(LAST_VOLUME_KEY)}, MA={results.get(VOLUME_MA_KEY)})")
         else:
             logger.warning(f"{Fore.YELLOW}Scrying (Vol): Failed calc VolMA({vol_ma_len}) or get last vol. LastVol: {df['volume'].iloc[-1]}, LastMA: {df[volume_ma_col].iloc[-1]}{Style.RESET_ALL}")
         # Clean up volume MA column if it exists
         if volume_ma_col in df.columns: df.drop(columns=[volume_ma_col], errors='ignore', inplace=True)

         # Log results
         atr_str = f"{results[ATR_KEY]:.4f}" if results[ATR_KEY] is not None else 'N/A'
         last_vol_val = results.get(LAST_VOLUME_KEY)
         vol_ma_val = results.get(VOLUME_MA_KEY)
         vol_ratio_val = results.get(VOLUME_RATIO_KEY)
         last_vol_str = f"{last_vol_val:.2f}" if last_vol_val is not None else 'N/A'
         vol_ma_str = f"{vol_ma_val:.2f}" if vol_ma_val is not None else 'N/A'
         vol_ratio_str = f"{vol_ratio_val:.2f}" if vol_ratio_val is not None else 'N/A'

         logger.debug(f"Scrying Results: ATR({atr_len}) = {Fore.CYAN}{atr_str}{Style.RESET_ALL}")
         logger.debug(f"Scrying Results: Volume: Last={last_vol_str}, MA({vol_ma_len})={vol_ma_str}, Ratio={Fore.YELLOW}{vol_ratio_str}{Style.RESET_ALL}")

     except Exception as e:
         logger.error(f"{Fore.RED}Scrying (Vol/ATR): Error during calculation: {e}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())
         results = {key: None for key in results} # Reset results on error
     return results


def analyze_order_book(
     exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int, market_info: Optional[Dict] = None
) -> Dict[str, Optional[Decimal]]:
     """Fetches L2 order book and analyzes bid/ask pressure and spread. Returns Decimals."""
     results: Dict[str, Optional[Decimal]] = {BID_ASK_RATIO_KEY: None, SPREAD_KEY: None, BEST_BID_KEY: None, BEST_ASK_KEY: None}
     logger.debug(f"Order Book Scrying: Fetching L2 for {symbol} (Analyze Depth: {depth}, API Fetch Limit: {fetch_limit})...")

     try:
         order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)

         if not order_book or not isinstance(order_book.get(BIDS_KEY), list) or not isinstance(order_book.get(ASKS_KEY), list):
             logger.warning(f"{Fore.YELLOW}Order Book Scrying: Incomplete or invalid data structure received for {symbol}.{Style.RESET_ALL}")
             return results

         bids: List[List[Union[float, str]]] = order_book[BIDS_KEY]
         asks: List[List[Union[float, str]]] = order_book[ASKS_KEY]

         if not bids or not asks:
             logger.warning(f"{Fore.YELLOW}Order Book Scrying: Bids or asks list is empty for {symbol}. Bids: {len(bids)}, Asks: {len(asks)}{Style.RESET_ALL}")
             return results

         # Get best bid/ask and calculate spread
         try:
             # Ensure there's data at index 0 and the inner list has at least one element (price)
             if len(bids) > 0 and len(bids[0]) > 0 and len(asks) > 0 and len(asks[0]) > 0:
                 best_bid_raw = bids[0][0]
                 best_ask_raw = asks[0][0]
                 # Convert to string first for Decimal robustness
                 results[BEST_BID_KEY] = Decimal(str(best_bid_raw))
                 results[BEST_ASK_KEY] = Decimal(str(best_ask_raw))

                 if results[BEST_BID_KEY] > 0 and results[BEST_ASK_KEY] > 0:
                      spread = results[BEST_ASK_KEY] - results[BEST_BID_KEY]
                      # Determine price precision dynamically if possible
                      price_precision = Decimal('0.0001') # Default precision
                      try:
                          # Use passed market_info if available, otherwise fetch it
                          local_market_info = market_info or exchange.market(symbol)
                          price_prec_str = local_market_info.get(PRECISION_KEY, {}).get(PRICE_KEY)
                          if price_prec_str:
                              # Convert precision string to Decimal exponent format (e.g., '0.01' -> Decimal('1E-2'))
                              price_precision = Decimal(str(price_prec_str))
                          else:
                              logger.debug(f"OB Scrying: Could not find price precision in market info for {symbol}. Using default.")
                      except (ccxt.BadSymbol, KeyError, InvalidOperation, ValueError, TypeError) as market_err:
                          logger.debug(f"OB Scrying: Could not get market precision for spread calc: {market_err}. Using default.")

                      # Ensure spread is not negative (can happen with crossed books briefly)
                      if spread < 0:
                          logger.warning(f"{Fore.YELLOW}OB Scrying: Negative spread detected ({spread}). Using absolute value.{Style.RESET_ALL}")
                          spread = abs(spread)

                      results[SPREAD_KEY] = spread.quantize(price_precision, rounding=ROUND_HALF_UP)
                      logger.debug(f"OB Scrying: Best Bid={Fore.GREEN}{results[BEST_BID_KEY]}{Style.RESET_ALL}, Best Ask={Fore.RED}{results[BEST_ASK_KEY]}{Style.RESET_ALL}, Spread={Fore.YELLOW}{results[SPREAD_KEY]}{Style.RESET_ALL}")
                 else:
                      logger.debug("OB Scrying: Could not calculate spread (Best Bid/Ask zero or invalid).")
             else:
                 logger.warning(f"{Fore.YELLOW}OB Scrying: Best bid/ask data missing or incomplete.{Style.RESET_ALL}")

         except (IndexError, InvalidOperation, ValueError, TypeError) as e:
              logger.warning(f"{Fore.YELLOW}OB Scrying: Error processing best bid/ask/spread for {symbol}: {e}{Style.RESET_ALL}")
              results[BEST_BID_KEY] = None; results[BEST_ASK_KEY] = None; results[SPREAD_KEY] = None

         # Calculate cumulative volume within depth
         try:
             # Ensure inner lists have at least two elements (price, volume)
             # Convert to string first for Decimal robustness
             bid_volume_sum_raw = sum(Decimal(str(bid[1])) for bid in bids[:depth] if len(bid) > 1)
             ask_volume_sum_raw = sum(Decimal(str(ask[1])) for ask in asks[:depth] if len(ask) > 1)
             # Use a reasonable precision for volume sums
             vol_precision = Decimal("0.0001") # Adjust if needed based on typical quantities
             bid_volume_sum = bid_volume_sum_raw.quantize(vol_precision, rounding=ROUND_HALF_UP)
             ask_volume_sum = ask_volume_sum_raw.quantize(vol_precision, rounding=ROUND_HALF_UP)
             logger.debug(f"OB Scrying (Depth {depth}): Cum Bid={Fore.GREEN}{bid_volume_sum}{Style.RESET_ALL}, Cum Ask={Fore.RED}{ask_volume_sum}{Style.RESET_ALL}")

             # Calculate Bid/Ask Ratio
             if ask_volume_sum > CONFIG.position_qty_epsilon: # Check against epsilon
                  try:
                      bid_ask_ratio = (bid_volume_sum / ask_volume_sum).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                      results[BID_ASK_RATIO_KEY] = bid_ask_ratio
                      ratio_color = Fore.GREEN if results[BID_ASK_RATIO_KEY] >= CONFIG.order_book_ratio_threshold_long else (Fore.RED if results[BID_ASK_RATIO_KEY] <= CONFIG.order_book_ratio_threshold_short else Fore.YELLOW)
                      logger.debug(f"OB Scrying Ratio (Depth {depth}) = {ratio_color}{results[BID_ASK_RATIO_KEY]:.3f}{Style.RESET_ALL}")
                  except (InvalidOperation, ZeroDivisionError): # Should be caught by epsilon check, but handle defensively
                      logger.warning(f"{Fore.YELLOW}OB Scrying Ratio calculation failed (InvalidOp/ZeroDiv). BidSum={bid_volume_sum}, AskSum={ask_volume_sum}{Style.RESET_ALL}")
                      results[BID_ASK_RATIO_KEY] = None
             else:
                  logger.debug(f"OB Scrying Ratio calculation skipped (Ask volume at depth {depth} is zero or negligible: {ask_volume_sum})")

         except (IndexError, InvalidOperation, ValueError, TypeError) as e:
              logger.warning(f"{Fore.YELLOW}OB Scrying: Error calculating cumulative volume or ratio for {symbol}: {e}{Style.RESET_ALL}")
              results[BID_ASK_RATIO_KEY] = None

     except (ccxt.NetworkError, ccxt.ExchangeError) as e:
         logger.warning(f"{Fore.YELLOW}OB Scrying: API error fetching order book for {symbol}: {e}{Style.RESET_ALL}")
     except (IndexError, InvalidOperation, ValueError, TypeError) as e:
         # Catch potential errors from processing the raw OB data structure
         logger.warning(f"{Fore.YELLOW}OB Scrying: Error processing OB data structure for {symbol}: {e}{Style.RESET_ALL}")
     except Exception as e:
         logger.error(f"{Fore.RED}OB Scrying: Unexpected error analyzing order book for {symbol}: {e}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())
         return {key: None for key in results}  # Return None dict on error

     return results


# --- Data Fetching - Gathering Etheric Data Streams ---
def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches OHLCV data with retries and basic validation."""
    logger.info(f"Data Fetch: Gathering {limit} {timeframe} candles for {symbol}...")
    for attempt in range(CONFIG.retry_count):
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                logger.warning(f"{Fore.YELLOW}Data Fetch: Received empty OHLCV data for {symbol} on attempt {attempt + 1}.{Style.RESET_ALL}")
                if attempt < CONFIG.retry_count - 1:
                    time.sleep(CONFIG.retry_delay_seconds)
                    continue
                else: return None  # Return None after final retry

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Convert timestamp to UTC datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            # --- Basic Data Validation ---
            if df.empty:
                logger.warning(f"{Fore.YELLOW}Data Fetch: DataFrame is empty after conversion for {symbol}.{Style.RESET_ALL}")
                return None  # Cannot proceed with empty DataFrame

            # Ensure correct data types before NaN checks
            # Convert OHLCV columns to numeric, coercing errors to NaN
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for NaNs introduced by conversion or present in original data
            if df.isnull().values.any():
                nan_counts = df.isnull().sum()
                logger.warning(f"{Fore.YELLOW}Data Fetch: Fetched OHLCV data contains NaN values after numeric conversion. Counts:\n{nan_counts[nan_counts > 0]}\nAttempting forward fill...{Style.RESET_ALL}")
                # Simple imputation: Forward fill NaNs. More sophisticated methods could be used.
                df.ffill(inplace=True)
                # Check again after filling - if NaNs remain (e.g., at the very beginning), data is unusable
                if df.isnull().values.any():
                    remaining_nans = df.isnull().sum()
                    logger.error(f"{Fore.RED}Data Fetch: NaN values remain after forward fill. Cannot proceed with this data batch. Remaining NaNs:\n{remaining_nans[remaining_nans > 0]}{Style.RESET_ALL}")
                    return None

            logger.debug(f"Data Fetch: Successfully woven {len(df)} OHLCV candles for {symbol}.")
            return df

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(f"{Fore.YELLOW}Data Fetch: API error fetching OHLCV for {symbol} (Attempt {attempt + 1}/{CONFIG.retry_count}): {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1:
                time.sleep(CONFIG.retry_delay_seconds)
            else:
                logger.error(f"{Fore.RED}Data Fetch: Failed to fetch OHLCV for {symbol} after {CONFIG.retry_count} attempts.{Style.RESET_ALL}")
                return None
        except Exception as e:
            logger.error(f"{Fore.RED}Data Fetch: Unexpected error fetching market data for {symbol}: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            return None  # Return None on unexpected errors

    return None  # Should not be reached if loop completes, but included for safety


# --- Position & Order Management - Manipulating Market Presence ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
     """Fetches current position details for Bybit V5 via CCXT. Returns Decimal for qty/price."""
     # Default: no active position, using Decimal for precision
     default_pos: Dict[str, Any] = {SIDE_KEY: POSITION_SIDE_NONE, QTY_KEY: Decimal('0.0'), ENTRY_PRICE_KEY: Decimal('0.0')}
     ccxt_unified_symbol: str = symbol
     market_id: Optional[str] = None
     market: Optional[Dict] = None

     # Get Market Info
     try:
         market = exchange.market(ccxt_unified_symbol)
         if not market: raise KeyError(f"Market info not found for {ccxt_unified_symbol}")
         market_id = market.get(ID_KEY) # The exchange-specific market ID (e.g., 'BTCUSDT')
         if not market_id: raise KeyError(f"Market ID not found in market info for {ccxt_unified_symbol}")
         logger.debug(f"Position Check: Fetching position for CCXT symbol '{ccxt_unified_symbol}' (Target Exchange Market ID: '{market_id}')...")
     except (ccxt.BadSymbol, KeyError) as e:
          logger.error(f"{Fore.RED}Position Check: Failed get market info/ID for '{ccxt_unified_symbol}': {e}{Style.RESET_ALL}")
          return default_pos
     except Exception as e:
          logger.error(f"{Fore.RED}Position Check: Unexpected error getting market info for '{ccxt_unified_symbol}': {e}{Style.RESET_ALL}")
          logger.debug(traceback.format_exc())
          return default_pos

     # Fetch Positions
     try:
         if not exchange.has.get('fetchPositions'):
             logger.warning(f"{Fore.YELLOW}Position Check: Exchange '{exchange.id}' may not support fetchPositions.{Style.RESET_ALL}")
             return default_pos

         # Determine category for V5 API call based on market info (linear/inverse)
         params: Dict[str, str] = {}
         if market and market.get(LINEAR_KEY, False): params = {PARAM_CATEGORY: LINEAR_KEY}
         elif market and market.get(INVERSE_KEY, False): params = {PARAM_CATEGORY: INVERSE_KEY}
         else:
             # If market type unknown, try linear as a default (most common for USDT perps)
             logger.warning(f"{Fore.YELLOW}Position Check: Market type for {symbol} unclear, defaulting category to 'linear'.{Style.RESET_ALL}")
             params = {PARAM_CATEGORY: LINEAR_KEY}

         # Fetch positions for the specific symbol
         # Setting settle coin might also be needed for V5 depending on account type (e.g., USDT for linear)
         # params[PARAM_SETTLE_COIN] = market.get(SETTLE_KEY, USDT_SYMBOL) # Add if needed, depends on account type and CCXT handling
         positions: List[Dict] = exchange.fetch_positions(symbols=[ccxt_unified_symbol], params=params)
         logger.debug(f"Fetched positions data (raw count: {len(positions)}) for symbol {ccxt_unified_symbol}")

         # Filter positions: Bybit V5 fetchPositions returns multiple entries even for one symbol/mode.
         # We need the one for the correct market_id AND One-Way mode (positionIdx=0).
         active_position_found: Optional[Dict] = None
         for pos in positions:
             pos_info: Dict = pos.get(INFO_KEY, {})
             pos_symbol_raw: Optional[str] = pos_info.get(SYMBOL_KEY) # Raw symbol from exchange ('BTCUSDT')

             # 1. Match the raw symbol from the position data with the market ID we expect
             if pos_symbol_raw != market_id:
                 # logger.debug(f"Skipping position entry, symbol mismatch: '{pos_symbol_raw}' != '{market_id}'")
                 continue

             # 2. Check for One-Way Mode (positionIdx=0) - V5 returns as string
             position_idx_str: Optional[str] = pos_info.get(PARAM_POSITION_IDX, '-1')
             try:
                 # Handle potential None value for position_idx_str
                 if position_idx_str is None: raise ValueError("positionIdx is None")
                 position_idx = int(position_idx_str)
             except (ValueError, TypeError):
                 logger.warning(f"{Fore.YELLOW}Could not parse positionIdx '{position_idx_str}' for {market_id}. Skipping.{Style.RESET_ALL}")
                 continue
             if position_idx != 0:
                 # logger.debug(f"Skipping position entry for {market_id}, not One-Way mode (positionIdx={position_idx}).")
                 continue # Skip hedge mode positions

             # If we found the entry for our symbol and One-Way mode, this is the one.
             active_position_found = pos
             logger.debug(f"Position Check: Found matching One-Way mode entry for {market_id}.")
             break # Stop searching once the correct entry is found

         # Process the found position (if any)
         if active_position_found:
             pos_info = active_position_found.get(INFO_KEY, {})
             # Check Side (V5: 'Buy', 'Sell', 'None')
             pos_side_v5: str = pos_info.get(SIDE_KEY, POSITION_SIDE_NONE) # Raw side from exchange
             determined_side: str = POSITION_SIDE_NONE
             if pos_side_v5 == BYBIT_SIDE_BUY: determined_side = POSITION_SIDE_LONG
             elif pos_side_v5 == BYBIT_SIDE_SELL: determined_side = POSITION_SIDE_SHORT
             # If side is 'None', it implies no position or a flat state for this entry

             # Check Position Size (V5: 'size') - this is the key indicator of an active position
             size_str: Optional[str] = pos_info.get(PARAM_SIZE)
             if size_str is None or size_str == "":
                 logger.debug(f"Position Check: Size field missing or empty for {market_id}. Assuming flat.")
                 return default_pos # Treat as flat if size is missing

             try:
                 # Convert size string to Decimal
                 size = Decimal(str(size_str))
                 # Check if size is significantly different from zero using epsilon
                 if abs(size) > CONFIG.position_qty_epsilon:
                     # Found active position! Get entry price.
                     # Prefer raw V5 avgPrice, fallback to CCXT unified entryPrice (less reliable for V5?)
                     entry_price_str: Optional[str] = pos_info.get(AVG_PRICE_KEY) # Prioritize V5 raw field
                     if entry_price_str is None or entry_price_str == "" or entry_price_str == "0": # Also check for "0" string
                         # Fallback to unified field if V5 raw field is missing/invalid
                         entry_price_str_unified = active_position_found.get(ENTRY_PRICE_KEY)
                         if entry_price_str_unified is not None:
                             entry_price_str = str(entry_price_str_unified) # Convert unified price to string
                             logger.debug(f"Position Check: Using fallback CCXT unified entryPrice field ('{entry_price_str}') for {market_id}.")
                         else:
                             entry_price_str = None # Ensure it's None if fallback also fails

                     entry_price = Decimal('0.0')
                     if entry_price_str is not None and entry_price_str != "" and entry_price_str != "0":
                         try:
                             # Convert entry price string to Decimal
                             entry_price = Decimal(str(entry_price_str))
                         except (InvalidOperation, ValueError, TypeError):
                              logger.warning(f"{Fore.YELLOW}Could not parse entry price string: '{entry_price_str}'. Defaulting to 0.0.{Style.RESET_ALL}")
                              entry_price = Decimal('0.0') # Default on parsing error
                     else:
                         logger.warning(f"{Fore.YELLOW}Position Check: Entry price field (avgPrice/entryPrice) missing, empty, or zero for active position {market_id}. Defaulting to 0.0.{Style.RESET_ALL}")

                     qty_abs = abs(size)
                     pos_color = Fore.GREEN if determined_side == POSITION_SIDE_LONG else Fore.RED
                     logger.info(f"{pos_color}Position Check: FOUND Active Position for {market_id}: Side={determined_side}, Qty={qty_abs}, Entry={entry_price:.4f}{Style.RESET_ALL}")
                     return {SIDE_KEY: determined_side, QTY_KEY: qty_abs, ENTRY_PRICE_KEY: entry_price}
                 else:
                     # Size is zero or negligible, treat as flat
                     logger.info(f"{Fore.BLUE}Position Check: Position size for {market_id} is zero/negligible ({size_str}). Treating as flat.{Style.RESET_ALL}")
                     return default_pos

             except (ValueError, TypeError, InvalidOperation) as e:
                  logger.warning(f"{Fore.YELLOW}Position Check: Error parsing size '{size_str}' for {market_id}: {e}{Style.RESET_ALL}")
                  return default_pos # Treat as flat on parsing error
         else:
             # No position entry matched the symbol and One-Way mode criteria
             logger.info(f"{Fore.BLUE}Position Check: No active One-Way Mode position found for {market_id}.{Style.RESET_ALL}")
             return default_pos

     except (ccxt.NetworkError, ccxt.ExchangeError) as e:
         logger.warning(f"{Fore.YELLOW}Position Check: API error during fetch_positions for {symbol}: {e}{Style.RESET_ALL}")
     except Exception as e:
         logger.error(f"{Fore.RED}Position Check: Unexpected error during position check for {symbol}: {e}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())

     logger.warning(f"{Fore.YELLOW}Position Check: Returning default (No Position) due to error or no active position found for {symbol}.{Style.RESET_ALL}")
     return default_pos


def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for Bybit V5, checks market type, retries."""
    logger.info(f"{Fore.CYAN}Leverage Conjuring: Attempting to set {leverage}x for {symbol}...{Style.RESET_ALL}")
    market_base: str = "N/A"
    market: Optional[Dict] = None
    try:
        market = exchange.market(symbol)
        if not market: raise KeyError(f"Market info not found for {symbol}")
        market_base = market.get(BASE_KEY, 'N/A')
        # Check if it's a contract market (swap, futures)
        if not market.get(CONTRACT_KEY, False) or market.get(SPOT_KEY, False):
            logger.error(f"{Fore.RED}Leverage Conjuring: Cannot set leverage for non-contract market: {symbol}. Market type: {market.get(TYPE_KEY)}{Style.RESET_ALL}")
            return False
    except (ccxt.BadSymbol, KeyError) as e:
          logger.error(f"{Fore.RED}Leverage Conjuring: Failed to get market info for symbol '{symbol}': {e}{Style.RESET_ALL}")
          return False
    except Exception as e:
          logger.error(f"{Fore.RED}Leverage Conjuring: Unexpected error getting market info for {symbol}: {e}{Style.RESET_ALL}")
          logger.debug(traceback.format_exc())
          return False

    for attempt in range(CONFIG.retry_count):
        try:
            # Bybit V5 requires setting buy and sell leverage separately via set_leverage call in CCXT
            # CCXT handles mapping this to the correct API call structure for Bybit V5
            # The unified method `set_leverage` should handle this abstraction.
            # Bybit V5 also requires specifying margin mode (isolated/cross) when setting leverage.
            # We set 'defaultMarginMode': 'isolated' in initialize_exchange, CCXT should use this.
            logger.debug(f"Leverage Conjuring: Calling exchange.set_leverage({leverage}, '{symbol}') (Attempt {attempt + 1}/{CONFIG.retry_count})")
            # Params might be needed if CCXT abstraction fails:
            # params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage), 'marginMode': 0} # 0 for isolated, 1 for cross
            response = exchange.set_leverage(leverage=leverage, symbol=symbol) # Rely on CCXT abstraction
            logger.success(f"{Fore.GREEN}Leverage Conjuring: Successfully set leverage to {leverage}x for {symbol}. Response: {response}{Style.RESET_ALL}")
            return True

        except ccxt.ExchangeError as e:
            error_msg_lower = str(e).lower()
            # Bybit V5 specific error codes or messages for "leverage not modified"
            # 110044: "Set leverage not modified" (from Bybit docs)
            # Check for common phrases as well
            if "110044" in str(e) or any(p in error_msg_lower for p in ["leverage not modified", "same leverage", "no need to modify leverage", "leverage is same as requested"]):
                logger.info(f"{Fore.CYAN}Leverage Conjuring: Leverage for {symbol} already set to {leverage}x (Confirmed by exchange message).{Style.RESET_ALL}")
                return True
            logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Exchange resistance on attempt {attempt + 1}/{CONFIG.retry_count} for {symbol}: {e}{Style.RESET_ALL}")
            if attempt < CONFIG.retry_count - 1:
                time.sleep(CONFIG.retry_delay_seconds)
            else:
                logger.error(f"{Fore.RED}Leverage Conjuring: FAILED after {CONFIG.retry_count} attempts due to exchange error: {e}{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] CRITICAL: Leverage set FAILED {symbol}.")
                return False
        except ccxt.NetworkError as e:
             logger.warning(f"{Fore.YELLOW}Leverage Conjuring: Network error on attempt {attempt + 1}/{CONFIG.retry_count} for {symbol}: {e}{Style.RESET_ALL}")
             if attempt < CONFIG.retry_count - 1:
                 time.sleep(CONFIG.retry_delay_seconds)
             else:
                 logger.error(f"{Fore.RED}Leverage Conjuring: FAILED after {CONFIG.retry_count} attempts due to network error.{Style.RESET_ALL}")
                 send_sms_alert(f"[{market_base}] CRITICAL: Leverage set FAILED (Network) {symbol}.")
                 return False
        except Exception as e:
             logger.error(f"{Fore.RED}Leverage Conjuring: Unexpected error setting leverage for {symbol} on attempt {attempt + 1}: {e}{Style.RESET_ALL}")
             logger.debug(traceback.format_exc())
             # Fail fast on unexpected errors
             send_sms_alert(f"[{market_base}] CRITICAL: Leverage set FAILED (Unexpected: {type(e).__name__}) {symbol}.")
             return False
    return False


def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: Dict[str, Any], reason: str = "Signal"
) -> Optional[Dict[str, Any]]:
    """Closes active position via market order with reduce_only, validates first. Returns filled order dict or None."""
    initial_side: str = position_to_close.get(SIDE_KEY, POSITION_SIDE_NONE)
    initial_qty: Decimal = position_to_close.get(QTY_KEY, Decimal('0.0'))
    market_base: str = symbol.split('/')[0] if '/' in symbol else symbol

    logger.info(f"{Fore.YELLOW}Banish Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}")

    # Re-validate Position Before Closing
    logger.debug("Banish Position: Re-validating live position status...")
    live_position = get_current_position(exchange, symbol)
    live_position_side: str = live_position.get(SIDE_KEY, POSITION_SIDE_NONE)
    live_amount_to_close: Decimal = live_position.get(QTY_KEY, Decimal('0.0'))  # Absolute value from get_current_position

    if live_position_side == POSITION_SIDE_NONE or live_amount_to_close <= CONFIG.position_qty_epsilon:
        logger.warning(f"{Fore.YELLOW}Banish Position: Discrepancy detected or position already closed. Initial check showed {initial_side}, but live check shows none or negligible qty ({live_amount_to_close:.8f}). Assuming closed.{Style.RESET_ALL}")
        return None # Treat as already closed, no action needed

    # Determine the side of the market order needed to close
    side_to_execute_close: str = SIDE_SELL if live_position_side == POSITION_SIDE_LONG else SIDE_BUY

    # Place Reduce-Only Market Order
    params: Dict[str, bool] = {PARAM_REDUCE_ONLY: True}
    closed_order: Optional[Dict] = None

    try:
        # Convert the Decimal amount to float for CCXT, applying precision first
        # Use amount_to_precision which returns a string, then convert to float
        amount_str: str = exchange.amount_to_precision(symbol, float(live_amount_to_close))
        amount_float_prec: float = float(amount_str)

        if amount_float_prec <= float(CONFIG.position_qty_epsilon): # Check precision-adjusted amount
            logger.error(f"{Fore.RED}Banish Pos: Closing amount {amount_str} ({live_amount_to_close}) negligible after precision. Abort.{Style.RESET_ALL}")
            return None

        logger.warning(f"{Back.YELLOW}{Fore.BLACK}Banish Position: Attempting to CLOSE {live_position_side} position ({reason}): Executing {side_to_execute_close.upper()} MARKET order for {amount_str} {symbol} (Reduce-Only){Style.RESET_ALL}")

        order = exchange.create_market_order(
            symbol=symbol, side=side_to_execute_close, amount=amount_float_prec, params=params
        )
        order_id = order.get(ID_KEY)
        if not order_id:
            logger.error(f"{Fore.RED}Banish Position ({reason}): Market close order placed but no ID returned. Cannot confirm fill.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL: Close order placed ({reason}) but NO ID! Manual check!")
            return None # Cannot proceed without ID

        # --- Confirm Order Fill (Crucial for Market Orders) ---
        # Use a short delay before confirming
        time.sleep(CONFIG.post_entry_delay_seconds) # Reuse post-entry delay setting
        confirmed_close_order = confirm_order_fill(exchange, order_id, symbol)

        if not confirmed_close_order:
            logger.error(f"{Fore.RED}Banish Position ({reason}): FAILED to confirm fill for close order {order_id}. Position state uncertain. MANUAL CHECK REQUIRED!{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL: Close order fill confirm FAILED: {order_id}. Manual check!")
            return None # Return None as closure is uncertain

        # Log details from the *confirmed* order object
        fill_price_str = "?"
        filled_qty_str = "?"
        cost_str = "?"
        order_id_short = str(order_id)[-6:]

        # Prefer 'average' for fill price, fallback to 'price'
        avg_price_raw = confirmed_close_order.get(AVERAGE_KEY) or confirmed_close_order.get(PRICE_KEY)
        if avg_price_raw is not None:
            try:
                fill_price_str = f"{Decimal(str(avg_price_raw)):.4f}"
            except (InvalidOperation, ValueError, TypeError):
                fill_price_str = f"Err ({avg_price_raw})"

        filled_qty_raw = confirmed_close_order.get(FILLED_KEY)
        if filled_qty_raw is not None:
            try:
                filled_qty_str = f"{Decimal(str(filled_qty_raw)):.8f}"
            except (InvalidOperation, ValueError, TypeError):
                filled_qty_str = f"Err ({filled_qty_raw})"

        cost_raw = confirmed_close_order.get(COST_KEY)
        if cost_raw is not None:
            try:
                cost_str = f"{Decimal(str(cost_raw)):.2f}"
            except (InvalidOperation, ValueError, TypeError):
                cost_str = f"Err ({cost_raw})"

        logger.success(f"{Fore.GREEN}{Style.BRIGHT}Banish Position: CLOSE Order ({reason}) CONFIRMED FILLED for {symbol}. Qty Filled: {filled_qty_str}/{amount_str}, Avg Fill ~{fill_price_str}, Cost: {cost_str} USDT. ID:...{order_id_short}{Style.RESET_ALL}")

        # Send SMS Alert
        sms_msg = (f"[{market_base}] BANISHED {live_position_side} {filled_qty_str} @ ~{fill_price_str} ({reason}). ID:...{order_id_short}")
        send_sms_alert(sms_msg)
        closed_order = confirmed_close_order # Store the confirmed order

    except ccxt.InsufficientFunds as e:
        logger.error(f"{Fore.RED}Banish Position ({reason}): Insufficient funds error during close attempt: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Insuff funds! Check margin.")
    except ccxt.NetworkError as e:
        logger.error(f"{Fore.RED}Banish Position ({reason}): Network error placing close order: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Network error! Check connection.")
    except ccxt.ExchangeError as e:
        err_str_lower = str(e).lower()
        # Check for specific errors indicating already closed/closing or order would not reduce
        # Bybit V5 error codes: 110025 ("Position size is zero"), 110043 ("Order would not reduce position size")
        # Also check for common phrases
        if "110025" in str(e) or "110043" in str(e) or any(phrase in err_str_lower for phrase in ["order would not reduce position size", "position is zero", "position size is zero", "cannot be less than", "position has been closed"]):
             logger.warning(f"{Fore.YELLOW}Banish Position ({reason}): Exchange indicates order would not reduce size or position is zero/closed. Assuming already closed. Error: {e}{Style.RESET_ALL}")
             return None  # Treat as success/non-actionable
        logger.error(f"{Fore.RED}Banish Position ({reason}): Exchange error placing close order: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): API error ({type(e).__name__}).")
    except (ValueError, TypeError, InvalidOperation) as e:
         logger.error(f"{Fore.RED}Banish Position ({reason}): Value error during amount processing (Qty: {live_amount_to_close}): {e}{Style.RESET_ALL}")
         send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Value error ({type(e).__name__}).")
    except Exception as e:
        logger.error(f"{Fore.RED}Banish Position ({reason}): Unexpected error placing close order: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): Unexpected error ({type(e).__name__}). Check logs!")

    return closed_order # Return the confirmed filled order dict, or None if closing failed/unconfirmed


def calculate_position_size(
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
    symbol: str,
    exchange: ccxt.Exchange,
    market_info: Dict # Pass market info to avoid redundant fetch
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
     """Calculates position size based on risk, checks limits. Returns (Quantity, Estimated Margin) as Decimals or (None, None)."""
     logger.debug(f"Risk Calc Input: Equity={equity:.2f}, Risk%={risk_per_trade_pct:.3%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x, Symbol={symbol}")

     # Input Validation
     if not (entry_price > 0 and stop_loss_price > 0):
         logger.error(f"{Fore.RED}Risk Calc: Invalid entry/SL price (must be > 0). Entry={entry_price}, SL={stop_loss_price}{Style.RESET_ALL}")
         return None, None
     price_difference_per_unit = abs(entry_price - stop_loss_price)
     if price_difference_per_unit <= CONFIG.position_qty_epsilon:
         logger.error(f"{Fore.RED}Risk Calc: Entry/SL prices are identical or too close to calculate risk ({price_difference_per_unit}). Entry={entry_price}, SL={stop_loss_price}{Style.RESET_ALL}")
         return None, None
     if not 0 < risk_per_trade_pct < 1:
         logger.error(f"{Fore.RED}Risk Calc: Invalid risk percentage: {risk_per_trade_pct:.3%}. Must be between 0 and 1.{Style.RESET_ALL}")
         return None, None
     if equity <= 0:
         logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.2f}. Must be positive.{Style.RESET_ALL}")
         return None, None
     if leverage <= 0:
         logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}. Must be positive.{Style.RESET_ALL}")
         return None, None

     # Calculation
     risk_amount_usdt: Decimal = equity * risk_per_trade_pct
     quantity: Decimal = risk_amount_usdt / price_difference_per_unit

     # Apply exchange precision to quantity
     try:
         # Use amount_to_precision which returns a string, then convert back to Decimal
         quantity_precise_str = exchange.amount_to_precision(symbol, float(quantity))
         quantity_precise = Decimal(quantity_precise_str)
         logger.debug(f"Risk Calc: Raw Qty={quantity:.18f}, Precise Qty={quantity_precise_str}")
         quantity = quantity_precise # Use the precision-adjusted Decimal value for further checks
     except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as e:
          logger.warning(f"{Fore.YELLOW}Risk Calc: Could not apply exchange precision to quantity {quantity:.8f} for {symbol}. Using raw value. Error: {e}{Style.RESET_ALL}")
     except Exception as e: # Catch any other unexpected error during precision adjustment
          logger.error(f"{Fore.RED}Risk Calc: Unexpected error applying precision to quantity {quantity:.8f} for {symbol}: {e}{Style.RESET_ALL}")
          logger.debug(traceback.format_exc())
          return None, None # Fail calculation if precision fails unexpectedly

     if quantity <= CONFIG.position_qty_epsilon:
         logger.error(f"{Fore.RED}Risk Calc: Calculated quantity ({quantity}) is zero or negligible after precision adjustment.{Style.RESET_ALL}")
         return None, None

     # Estimate Value and Margin using the precise quantity
     position_value_usdt: Decimal = quantity * entry_price
     required_margin_estimate: Decimal = position_value_usdt / Decimal(leverage)
     logger.debug(f"Risk Calc Output: RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference_per_unit:.4f} => PreciseQty={quantity:.8f}, EstVal={position_value_usdt:.2f}, EstMargin={required_margin_estimate:.2f}")

     # Exchange Limit Checks (using precise quantity and estimated value)
     try:
         # Use passed market_info
         limits = market_info.get(LIMITS_KEY, {})
         amount_limits = limits.get(AMOUNT_KEY, {})
         cost_limits = limits.get(COST_KEY, {})

         # Safely get min/max limits, converting to Decimal
         def get_limit_decimal(limit_dict: Optional[dict], key: str, default: str) -> Decimal:
             """Safely extracts and converts limit value to Decimal."""
             if limit_dict is None: return Decimal(default)
             val = limit_dict.get(key)
             if val is not None:
                 try:
                     return Decimal(str(val))
                 except (InvalidOperation, ValueError, TypeError):
                     logger.warning(f"Could not parse limit '{key}' value '{val}' to Decimal. Using default '{default}'.")
             return Decimal(default)

         min_amount = get_limit_decimal(amount_limits, MIN_KEY, '0')
         max_amount = get_limit_decimal(amount_limits, MAX_KEY, 'inf')
         min_cost = get_limit_decimal(cost_limits, MIN_KEY, '0')
         max_cost = get_limit_decimal(cost_limits, MAX_KEY, 'inf')

         logger.debug(f"Market Limits for {symbol}: MinAmt={min_amount}, MaxAmt={max_amount}, MinCost={min_cost}, MaxCost={max_cost}")

         # Check against limits
         if quantity < min_amount:
             logger.error(f"{Fore.RED}Risk Calc: Calculated Qty {quantity:.8f} is less than Min Amount limit {min_amount:.8f}.{Style.RESET_ALL}")
             return None, None
         if position_value_usdt < min_cost:
             logger.error(f"{Fore.RED}Risk Calc: Estimated Value {position_value_usdt:.2f} is less than Min Cost limit {min_cost:.2f}.{Style.RESET_ALL}")
             return None, None
         if quantity > max_amount:
             logger.warning(f"{Fore.YELLOW}Risk Calc: Calculated Qty {quantity:.8f} exceeds Max Amount limit {max_amount:.8f}. Capping quantity to limit.{Style.RESET_ALL}")
             # Apply precision to the max_amount limit itself before assigning
             try:
                 max_amount_str = exchange.amount_to_precision(symbol, float(max_amount))
                 quantity = Decimal(max_amount_str) # Cap quantity to the precision-adjusted max limit
             except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as e:
                 logger.error(f"{Fore.RED}Risk Calc: Failed to apply precision to max_amount {max_amount}. Cannot cap quantity reliably. Aborting.{Style.RESET_ALL}")
                 return None, None

             # Recalculate estimated value and margin based on capped quantity
             position_value_usdt = quantity * entry_price
             required_margin_estimate = position_value_usdt / Decimal(leverage)
             logger.info(f"Risk Calc: Capped Qty={quantity:.8f}, New EstVal={position_value_usdt:.2f}, New EstMargin={required_margin_estimate:.2f}")
         if position_value_usdt > max_cost:
             # If even the capped quantity's value exceeds max cost, it's an issue
             logger.error(f"{Fore.RED}Risk Calc: Estimated Value {position_value_usdt:.2f} (potentially capped) exceeds Max Cost limit {max_cost:.2f}.{Style.RESET_ALL}")
             return None, None

     except (KeyError, InvalidOperation, ValueError, TypeError) as e:
         logger.warning(f"{Fore.YELLOW}Risk Calc: Error parsing market limits for {symbol}: {e}. Skipping limit checks.{Style.RESET_ALL}")
     except Exception as e:
         logger.warning(f"{Fore.YELLOW}Risk Calc: Unexpected error checking market limits: {e}. Skipping limit checks.{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())

     # Return Decimal values for quantity and margin estimate
     return quantity, required_margin_estimate


def confirm_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Confirms if an order is filled using fetch_order primarily, falling back to fetch_closed_orders.

    Args:
        exchange: Initialized CCXT exchange object.
        order_id: The ID of the order to confirm.
        symbol: Unified CCXT symbol.

    Returns:
        Optional[Dict[str, Any]]: The filled order details (status='closed'), or None if not confirmed filled/failed.
    """
    log_prefix = f"Fill Confirm (ID:...{order_id[-6:]})"
    logger.debug(f"{log_prefix}: Attempting to confirm fill...")
    start_time = time.time()
    confirmed_order: Optional[Dict] = None

    # --- Primary Method: fetch_order ---
    for attempt in range(CONFIG.fetch_order_status_retries):
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get(STATUS_KEY)
            logger.debug(f"{log_prefix}: Attempt {attempt + 1}, fetch_order status: {status}")

            if status == ORDER_STATUS_CLOSED:
                logger.success(f"{log_prefix}: Confirmed FILLED via fetch_order.")
                confirmed_order = order
                break  # Exit loop on success
            elif status in [ORDER_STATUS_CANCELED, ORDER_STATUS_REJECTED, ORDER_STATUS_EXPIRED]:
                logger.error(f"{Fore.RED}{log_prefix}: Order FAILED with status '{status}' via fetch_order.{Style.RESET_ALL}")
                return None  # Order definitively failed

            # If status is 'open' or None/unknown, continue retrying

        except ccxt.OrderNotFound:
            logger.warning(f"{Fore.YELLOW}{log_prefix}: Order not found via fetch_order (Attempt {attempt + 1}). Might be processing or already closed/canceled.{Style.RESET_ALL}")
            # Continue to next attempt, will try fallback later
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(f"{Fore.YELLOW}{log_prefix}: API error during fetch_order (Attempt {attempt + 1}): {e}{Style.RESET_ALL}")
            # Continue retrying
        except Exception as e:
            logger.error(f"{Fore.RED}{log_prefix}: Unexpected error during fetch_order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            # Continue retrying for robustness, but log the error
            pass # Avoid returning None on unexpected error here, let fallback try

        # Wait before retrying fetch_order
        if attempt < CONFIG.fetch_order_status_retries - 1:
            time.sleep(CONFIG.fetch_order_status_delay)

    # --- Fallback Method: fetch_closed_orders (if fetch_order didn't confirm 'closed') ---
    if not confirmed_order:
        logger.debug(f"{log_prefix}: fetch_order did not confirm 'closed'. Trying fallback: fetch_closed_orders...")
        try:
            # Fetch recent closed orders (increase limit slightly for safety)
            # Add timestamp filter to limit results
            since_timestamp = int((time.time() - CONFIG.confirm_fill_lookback_seconds) * 1000) # Look back N seconds
            closed_orders = exchange.fetch_closed_orders(symbol, limit=20, since=since_timestamp)
            logger.debug(f"{log_prefix}: Fallback fetched {len(closed_orders)} recent closed orders since {since_timestamp}.")
            for order in closed_orders:
                if order.get(ID_KEY) == order_id:
                    status = order.get(STATUS_KEY)
                    # We only care if it's 'closed' in the fallback, as 'canceled' etc. should have been caught by fetch_order
                    if status == ORDER_STATUS_CLOSED:
                         logger.success(f"{log_prefix}: Confirmed FILLED via fetch_closed_orders fallback.")
                         confirmed_order = order
                         break
                    else:
                         # Found the order but it wasn't closed (e.g., canceled and fetch_order missed it?)
                         logger.warning(f"{Fore.YELLOW}{log_prefix}: Found order in fallback, but status is '{status}'. Assuming failed/not filled.{Style.RESET_ALL}")
                         return None # Treat as failed if found in closed but not 'closed' status
            if not confirmed_order:
                 logger.warning(f"{Fore.YELLOW}{log_prefix}: Order not found in recent closed orders via fallback.{Style.RESET_ALL}")

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(f"{Fore.YELLOW}{log_prefix}: API error during fetch_closed_orders fallback: {e}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}{log_prefix}: Unexpected error during fetch_closed_orders fallback: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())

    # --- Final Verdict ---
    if confirmed_order:
        # Final check: Ensure filled amount is positive
        try:
            filled_qty_str = confirmed_order.get(FILLED_KEY, '0')
            filled_qty = Decimal(str(filled_qty_str))
            if filled_qty <= CONFIG.position_qty_epsilon:
                logger.error(f"{Fore.RED}{log_prefix}: Order {order_id} confirmed '{ORDER_STATUS_CLOSED}' but filled quantity is zero/negligible ({filled_qty}). Treating as FAILED.{Style.RESET_ALL}")
                return None
        except (InvalidOperation, ValueError, TypeError):
            logger.error(f"{Fore.RED}{log_prefix}: Could not parse filled quantity '{confirmed_order.get(FILLED_KEY)}' from confirmed order {order_id}. Treating as FAILED.{Style.RESET_ALL}")
            return None
        return confirmed_order
    else:
        elapsed = time.time() - start_time
        logger.error(f"{Fore.RED}{log_prefix}: FAILED to confirm fill for order {order_id} using both methods within timeout ({elapsed:.1f}s). Assume failure.{Style.RESET_ALL}")
        return None


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    quantity_decimal: Decimal, # Receive Decimal from calculation
    required_margin_decimal: Decimal, # Receive Decimal from calculation
    stop_loss_price: Decimal,
    take_profit_price: Decimal,
    market_info: Dict # Pass market info (including precision, limits)
) -> Optional[Dict[str, Any]]:
    """Places market entry, confirms fill, then places SL/TP orders based on actual fill.
    Returns the confirmed *entry* order dict if successful, otherwise None.
    """
    market_base: str = symbol.split('/')[0] if '/' in symbol else symbol
    log_prefix: str = f"Entry Order ({side.upper()})"
    order_id: Optional[str] = None
    entry_order: Optional[Dict[str, Any]] = None
    confirmed_order: Optional[Dict[str, Any]] = None # To store the confirmed filled order

    try:
        # Fetch Balance & Ticker (needed for capping and final checks)
        logger.debug(f"{log_prefix}: Gathering resources (Balance, Ticker)...")
        # Determine account type based on settle currency for balance fetch
        settle_currency: str = market_info.get(SETTLE_KEY, USDT_SYMBOL)
        # Default to UNIFIED as per initialize_exchange logic
        account_type: str = 'UNIFIED'
        balance_info = exchange.fetch_balance(params={'accountType': account_type})
        # Use 'free' balance for margin check (ensure it's the settle currency)
        free_balance_raw = balance_info.get(FREE_KEY, {}).get(settle_currency)
        if free_balance_raw is None: raise ValueError(f"Could not fetch free {settle_currency} balance.")
        free_balance = Decimal(str(free_balance_raw))

        ticker = exchange.fetch_ticker(symbol)
        last_price_raw = ticker.get(LAST_PRICE_KEY)
        if last_price_raw is None: raise ValueError("Could not fetch last price for estimates.")
        entry_price_estimate = Decimal(str(last_price_raw))
        if entry_price_estimate <= 0: raise ValueError(f"Fetched invalid last price: {entry_price_estimate}")

        # --- Pre-flight Checks ---
        # Quantity and margin are already Decimals

        # Cap Quantity based on MAX_ORDER_USDT_AMOUNT
        estimated_value = quantity_decimal * entry_price_estimate
        if estimated_value > CONFIG.max_order_usdt_amount:
            original_quantity_str = f"{quantity_decimal:.8f}"
            # Calculate capped quantity as Decimal first for precision
            capped_qty_decimal_raw = (CONFIG.max_order_usdt_amount / entry_price_estimate)
            # Apply exchange precision using float conversion temporarily
            try:
                quantity_str = exchange.amount_to_precision(symbol, float(capped_qty_decimal_raw))
                quantity_decimal = Decimal(quantity_str) # Update Decimal quantity
            except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as e:
                 logger.error(f"{Fore.RED}{log_prefix}: Failed to apply precision to capped quantity {capped_qty_decimal_raw}. Aborting. Error: {e}{Style.RESET_ALL}")
                 return None
            # Recalculate estimated margin based on capped Decimal quantity
            required_margin_decimal = (quantity_decimal * entry_price_estimate) / Decimal(CONFIG.leverage)
            logger.warning(f"{Fore.YELLOW}{log_prefix}: Qty {original_quantity_str} (Val ~{estimated_value:.2f}) > Max {CONFIG.max_order_usdt_amount:.2f}. Capping to {quantity_str} (New Est. Margin ~{required_margin_decimal:.2f}).{Style.RESET_ALL}")

        # Final Limit Checks (Amount & Cost) using potentially capped quantity and market_info
        limits = market_info.get(LIMITS_KEY, {})
        amount_limits = limits.get(AMOUNT_KEY, {})
        cost_limits = limits.get(COST_KEY, {})
        # Use the safe getter function from calculate_position_size
        def get_limit_decimal(limit_dict: Optional[dict], key: str, default: str) -> Decimal:
             """Safely extracts and converts limit value to Decimal."""
             if limit_dict is None: return Decimal(default)
             val = limit_dict.get(key)
             if val is not None:
                 try: return Decimal(str(val))
                 except (InvalidOperation, ValueError, TypeError): logger.warning(f"Could not parse limit '{key}' value '{val}' to Decimal. Using default '{default}'.")
             return Decimal(default)

        min_amount = get_limit_decimal(amount_limits, MIN_KEY, '0')
        min_cost = get_limit_decimal(cost_limits, MIN_KEY, '0')

        if quantity_decimal < min_amount:
            logger.error(f"{Fore.RED}{log_prefix}: Final quantity {quantity_decimal:.8f} < Min Amount limit {min_amount:.8f}. Abort.{Style.RESET_ALL}")
            return None
        estimated_cost_final = quantity_decimal * entry_price_estimate
        if estimated_cost_final < min_cost:
            logger.error(f"{Fore.RED}{log_prefix}: Final estimated cost {estimated_cost_final:.2f} < Min Cost limit {min_cost:.2f}. Abort.{Style.RESET_ALL}")
            return None

        # Margin Check using potentially recalculated margin estimate
        required_margin_with_buffer = required_margin_decimal * CONFIG.required_margin_buffer
        logger.debug(f"{log_prefix}: Free Balance={free_balance:.2f} {settle_currency}, Est. Margin Required (incl. buffer)={required_margin_with_buffer:.2f} {settle_currency}")
        if free_balance < required_margin_with_buffer:
            logger.error(f"{Fore.RED}{log_prefix}: Insufficient free balance ({free_balance:.2f}) for required margin ({required_margin_with_buffer:.2f}). Abort.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] Order REJECTED ({side.upper()}): Insuff. free balance. Need ~{required_margin_with_buffer:.2f}")
            return None

        # --- Place Market Order ---
        entry_side_color = Back.GREEN if side == SIDE_BUY else Back.RED
        text_color = Fore.BLACK if side == SIDE_BUY else Fore.WHITE
        # Convert final Decimal quantity to float for create_market_order
        quantity_float = float(quantity_decimal)
        logger.warning(f"{entry_side_color}{text_color}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {quantity_float:.8f} {symbol} ***{Style.RESET_ALL}")
        entry_order = exchange.create_market_order(symbol, side, quantity_float)
        order_id = entry_order.get(ID_KEY)
        if not order_id: raise ValueError("Market order placed but no ID returned.")
        logger.success(f"{log_prefix}: Market order submitted. ID: ...{order_id[-6:]}. Waiting for fill confirmation...")
        time.sleep(CONFIG.post_entry_delay_seconds)  # Allow time for order processing

        # --- Confirm Order Fill ---
        confirmed_order = confirm_order_fill(exchange, order_id, symbol)
        if not confirmed_order:
            logger.error(f"{Fore.RED}{log_prefix}: FAILED to confirm fill for entry order {order_id}. Aborting SL/TP placement.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL: Entry fill confirm FAILED: {order_id}. Manual check needed!")
            # Position state is unknown, do not proceed.
            return None

        # --- Extract Actual Fill Details ---
        actual_filled_qty = Decimal('0.0')
        actual_avg_price = Decimal('0.0')
        try:
            # Use Decimal for accuracy when reading fill details
            actual_filled_qty_str = confirmed_order.get(FILLED_KEY)
            # Prefer 'average' price from CCXT unified response, fallback to raw 'avgPrice' or 'price'
            actual_avg_price_str = confirmed_order.get(AVERAGE_KEY) or confirmed_order.get(INFO_KEY, {}).get(AVG_PRICE_KEY) or confirmed_order.get(PRICE_KEY)

            if actual_filled_qty_str is None or actual_avg_price_str is None:
                 raise ValueError("Missing filled quantity or average price in confirmed order.")

            actual_filled_qty = Decimal(str(actual_filled_qty_str))
            actual_avg_price = Decimal(str(actual_avg_price_str))

            if actual_filled_qty <= CONFIG.position_qty_epsilon or actual_avg_price <= 0:
                 raise ValueError(f"Invalid fill data: Qty={actual_filled_qty}, Price={actual_avg_price}")
            logger.success(f"{log_prefix}: Fill Confirmed: Order ID ...{order_id[-6:]}, Filled Qty={actual_filled_qty:.8f}, Avg Price={actual_avg_price:.4f}")
        except (InvalidOperation, ValueError, TypeError, KeyError) as e:
            logger.error(f"{Fore.RED}{log_prefix}: Error parsing confirmed fill details for order {order_id}: {e}. Data: {confirmed_order}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL ERROR parsing fill data: {order_id}. Manual check needed!")
            # We have a filled position but can't parse details. Attempt emergency close? Risky. Alerting is primary.
            # Let's return None here, indicating failure to setup SL/TP.
            return None

        # --- Place SL/TP Orders ---
        logger.info(f"{log_prefix}: Placing SL ({stop_loss_price}) and TP ({take_profit_price}) orders for filled qty {actual_filled_qty:.8f}...")
        sl_tp_success = True
        close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
        # Use the actual filled quantity (as float for create_order) for SL/TP orders
        sl_tp_amount_float = float(actual_filled_qty)

        # Apply precision to SL/TP prices before sending
        try:
            sl_price_str = exchange.price_to_precision(symbol, float(stop_loss_price))
            tp_price_str = exchange.price_to_precision(symbol, float(take_profit_price))
        except Exception as e:
            logger.error(f"{Fore.RED}{log_prefix}: Failed to apply precision to SL/TP prices: {e}. Aborting SL/TP placement.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL: SL/TP price precision FAILED for {order_id}. Manual check needed!")
            # Attempt emergency close if SL/TP cannot be placed due to precision error?
            if CONFIG.emergency_close_on_sl_fail:
                logger.warning(f"{log_prefix}: Attempting emergency close due to SL/TP price precision failure.")
                # Construct a temporary position dict based on confirmed fill
                pos_to_close = {SIDE_KEY: side, QTY_KEY: actual_filled_qty, ENTRY_PRICE_KEY: actual_avg_price}
                close_position(exchange, symbol, pos_to_close, reason="SL_TP_Precision_Fail")
            return None # Cannot place SL/TP without correct price format

        # SL Order Params (stopMarket)
        sl_trigger_direction = 2 if side == SIDE_BUY else 1 # Trigger when price goes BELOW for LONG SL, ABOVE for SHORT SL
        sl_params = {
            PARAM_STOP_PRICE: sl_price_str,
            PARAM_REDUCE_ONLY: True,
            PARAM_TRIGGER_DIRECTION: sl_trigger_direction, # 1: Mark price > trigger price, 2: Mark price < trigger price
            # Bybit V5 specific params (may be handled by CCXT, but explicit can be safer)
            'positionIdx': 0, # 0 for One-Way mode
            'tpslMode': 'Full', # Bybit V5: 'Full' or 'Partial'. Assume full position SL/TP.
            'slOrderType': 'Market', # Explicitly state market execution for SL trigger
        }
        # TP Order Params (stopMarket)
        tp_trigger_direction = 1 if side == SIDE_BUY else 2 # Trigger when price goes ABOVE for LONG TP, BELOW for SHORT TP
        tp_params = {
            PARAM_STOP_PRICE: tp_price_str,
            PARAM_REDUCE_ONLY: True,
            PARAM_TRIGGER_DIRECTION: tp_trigger_direction, # 1: Mark price > trigger price, 2: Mark price < trigger price
            # Bybit V5 specific params
            'positionIdx': 0, # 0 for One-Way mode
            'tpslMode': 'Full', # As above
            'tpOrderType': 'Market', # Explicitly state market execution for TP trigger
        }

        sl_order_id_short, tp_order_id_short = "N/A", "N/A"
        sl_order_info, tp_order_info = None, None

        # Place Stop-Loss Order
        try:
            logger.debug(f"Placing SL order: symbol={symbol}, type={ORDER_TYPE_STOP_MARKET}, side={close_side}, amount={sl_tp_amount_float}, params={sl_params}")
            sl_order_info = exchange.create_order(symbol, ORDER_TYPE_STOP_MARKET, close_side, sl_tp_amount_float, params=sl_params)
            sl_order_id_short = str(sl_order_info.get(ID_KEY, 'N/A'))[-6:]
            logger.success(f"{Fore.GREEN}{log_prefix}: Stop-Loss order placed. ID: ...{sl_order_id_short}{Style.RESET_ALL}")
            time.sleep(0.1)  # Small delay between orders
        except Exception as e:
            sl_tp_success = False
            logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix}: FAILED to place Stop-Loss order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] CRITICAL: SL order FAILED for {symbol} ({side}) after entry {order_id[-6:]}: {type(e).__name__}")
            # Attempt to close position immediately if SL fails and config allows
            if CONFIG.emergency_close_on_sl_fail:
                logger.warning(f"{log_prefix}: Attempting emergency close due to SL placement failure.")
                # Construct a temporary position dict based on confirmed fill
                pos_to_close = {SIDE_KEY: side, QTY_KEY: actual_filled_qty, ENTRY_PRICE_KEY: actual_avg_price}
                close_result = close_position(exchange, symbol, pos_to_close, reason="SL_Placement_Fail")
                if close_result:
                    logger.warning(f"{log_prefix}: Emergency close successful after SL failure.")
                else:
                    logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix}: Emergency close FAILED after SL placement failure. MANUAL INTERVENTION URGENT!{Style.RESET_ALL}")
                    send_sms_alert(f"[{market_base}] URGENT: Emergency CLOSE FAILED after SL fail for {symbol}. MANUAL ACTION!")
            # Even if emergency close fails, we cannot proceed with TP as the state is compromised. Return None.
            return None

        # Place Take-Profit Order (only if SL succeeded)
        try:
            logger.debug(f"Placing TP order: symbol={symbol}, type={ORDER_TYPE_STOP_MARKET}, side={close_side}, amount={sl_tp_amount_float}, params={tp_params}")
            tp_order_info = exchange.create_order(symbol, ORDER_TYPE_STOP_MARKET, close_side, sl_tp_amount_float, params=tp_params)
            tp_order_id_short = str(tp_order_info.get(ID_KEY, 'N/A'))[-6:]
            logger.success(f"{Fore.GREEN}{log_prefix}: Take-Profit order placed. ID: ...{tp_order_id_short}{Style.RESET_ALL}")
        except Exception as e:
            # TP failure is less critical than SL, but still log as error and mark overall failure
            sl_tp_success = False
            logger.error(f"{Back.YELLOW}{Fore.BLACK}{log_prefix}: FAILED to place Take-Profit order: {e}{Style.RESET_ALL}")
            logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] WARNING: TP order FAILED for {symbol} ({side}) after entry {order_id[-6:]}: {type(e).__name__}")
            # If TP fails, the position is still protected by SL. Should we cancel the SL and close?
            # Or leave the SL active? Leaving SL active seems safer.
            # Mark sl_tp_success as False, so the function returns None, indicating incomplete setup.
            # The main loop won't see a successful entry and won't try to manage it further, relying on the existing SL.

        # Final outcome
        if sl_tp_success:
            logger.info(f"{Fore.GREEN}{log_prefix}: Entry order filled and SL/TP orders placed successfully.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] Entered {side.upper()} {actual_filled_qty:.4f} @ {actual_avg_price:.2f}. SL=...{sl_order_id_short}, TP=...{tp_order_id_short}")
            # Return the confirmed *entry* order details
            return confirmed_order
        else:
            logger.error(f"{Back.RED}{Fore.WHITE}{log_prefix}: Entry filled (ID:...{order_id[-6:]}), but FAILED to place one/both SL/TP orders. Position might be partially protected (SL only) or unprotected. MANUAL INTERVENTION RECOMMENDED!{Style.RESET_ALL}")
            # Return None to indicate the overall process including SL/TP setup failed or was incomplete.
            return None

    except (ccxt.InsufficientFunds, ccxt.ExchangeError, ccxt.NetworkError, ValueError) as e:
        logger.error(f"{Fore.RED}{log_prefix}: Failed during order placement/check: {e}{Style.RESET_ALL}")
        if entry_order: logger.error(f"Entry order details (if placed): {entry_order}")
        send_sms_alert(f"[{market_base}] Entry order FAILED ({side.upper()}): {type(e).__name__}")
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix}: Unexpected error during order ritual: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] UNEXPECTED error during entry ({side.upper()}): {type(e).__name__}")
        return None


# --- Core Trading Logic - The Spell Weaving Cycle ---
def trade_logic(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> None:
    """Main trading logic loop."""
    cycle_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== New Weaving Cycle: {symbol} ({timeframe}) | {cycle_time_str} =========={Style.RESET_ALL}")

    # --- 0. Get Market Info (once per cycle) ---
    try:
        market_info = exchange.market(symbol)
        if not market_info:
            logger.error(f"{Fore.RED}Trade Logic: Skipping cycle - unable to fetch market info for {symbol}.{Style.RESET_ALL}")
            return
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - API error fetching market info for {symbol}: {e}{Style.RESET_ALL}")
        return
    except Exception as e:
        logger.error(f"{Fore.RED}Trade Logic: Skipping cycle - Unexpected error fetching market info for {symbol}: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return

    # --- 1. Get Data ---
    # Determine required length based on longest indicator period + buffer
    required_ohlcv_len = max(CONFIG.st_atr_length, CONFIG.confirm_st_atr_length, CONFIG.volume_ma_period, CONFIG.atr_calculation_period) + CONFIG.api_fetch_limit_buffer
    df = fetch_ohlcv(exchange, symbol, timeframe, limit=required_ohlcv_len)
    if df is None or df.empty:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - unable to fetch valid OHLCV data.{Style.RESET_ALL}")
        return

    order_book_data: Optional[Dict[str, Optional[Decimal]]] = None
    # Fetch OB if always required OR if confirmation is enabled (fetch only when needed later)
    if CONFIG.fetch_order_book_per_cycle:
        order_book_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit, market_info)
        # If OB analysis is mandatory for entry and it failed, skip cycle
        if order_book_data is None and CONFIG.use_ob_confirm:
             logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - failed to get required OB data (fetch_order_book_per_cycle=True).{Style.RESET_ALL}")
             return

    # --- 2. Calculate Indicators ---
    logger.debug("Calculating indicators...")
    df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
    df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
    vol_atr_results = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)

    # Check for critical indicator calculation failures
    # Check if columns exist and the last value is not NaN/NA
    # Use the renamed column names from calculate_supertrend
    if 'trend' not in df.columns or pd.isna(df['trend'].iloc[-1]):
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - primary Supertrend trend calculation failed or resulted in NaN.{Style.RESET_ALL}")
        return
    if 'confirm_trend' not in df.columns or pd.isna(df['confirm_trend'].iloc[-1]):
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - confirmation Supertrend trend calculation failed or resulted in NaN.{Style.RESET_ALL}")
        return
    if vol_atr_results is None or vol_atr_results.get(ATR_KEY) is None:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Skipping cycle - Volume/ATR calculation failed or ATR is None.{Style.RESET_ALL}")
        return

    # --- 3. Extract Latest Indicator Values ---
    try:
        last_candle = df.iloc[-1]
        # Primary Supertrend (using renamed columns)
        st_trend: float = last_candle['trend']                 # Current trend direction (-1.0 or 1.0)
        st_long_signal: bool = last_candle['st_long_flip']    # True only on the candle where trend flipped to long
        st_short_signal: bool = last_candle['st_short_flip']  # True only on the candle where trend flipped to short
        # Confirmation Supertrend (using renamed columns)
        confirm_st_trend: float = last_candle['confirm_trend'] # Current trend direction (-1.0 or 1.0)
        # Volume/ATR
        current_atr: Optional[Decimal] = vol_atr_results.get(ATR_KEY) # Decimal or None
        volume_ratio: Optional[Decimal] = vol_atr_results.get(VOLUME_RATIO_KEY) # Decimal or None
        # Current Price (use last close)
        current_price = Decimal(str(last_candle['close']))

        # Validate essential values extracted
        # current_atr is already checked above, re-checking here for safety
        if current_atr is None or pd.isna(st_trend) or pd.isna(confirm_st_trend) or current_price <= 0:
             raise ValueError("Essential indicator values (ATR, ST trends, Price) are None/NaN/invalid after extraction.")
        # Flip signals are guaranteed boolean by calculate_supertrend fix. No need for isinstance check.

    except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
        logger.error(f"{Fore.RED}Trade Logic: Error accessing or validating indicator/price data from last candle: {e}{Style.RESET_ALL}")
        logger.debug(f"DataFrame tail:\n{df.tail()}")
        logger.debug(f"Vol/ATR Results: {vol_atr_results}")
        return

    # --- 4. Check Current Position ---
    current_position = get_current_position(exchange, symbol)
    position_side: str = current_position[SIDE_KEY]
    position_qty: Decimal = current_position[QTY_KEY]
    position_entry_price: Decimal = current_position[ENTRY_PRICE_KEY]
    pos_color = Fore.GREEN if position_side == POSITION_SIDE_LONG else (Fore.RED if position_side == POSITION_SIDE_SHORT else Fore.BLUE)
    logger.info(f"State | Position: Side={pos_color}{position_side}{Style.RESET_ALL}, Qty={position_qty:.8f}, Entry={position_entry_price:.4f}")
    vol_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else 'N/A'
    # Cast float trends to int for logging clarity
    logger.info(f"State | Indicators: Price={current_price:.4f}, ATR={current_atr:.4f}, ST Trend={int(st_trend)}, Confirm ST Trend={int(confirm_st_trend)}, VolRatio={vol_ratio_str}")

    # --- 5. Determine Signals ---
    # Entry signal: ST flip occurred on this candle AND confirmation ST agrees with the new direction
    long_entry_signal: bool = st_long_signal and confirm_st_trend == 1.0
    short_entry_signal: bool = st_short_signal and confirm_st_trend == -1.0
    # Exit signal: Primary ST trend flips against the current position
    close_long_signal: bool = position_side == POSITION_SIDE_LONG and st_trend == -1.0
    close_short_signal: bool = position_side == POSITION_SIDE_SHORT and st_trend == 1.0
    logger.debug(f"Signals: EntryLong={long_entry_signal}, EntryShort={short_entry_signal}, CloseLong={close_long_signal}, CloseShort={close_short_signal}")

    # --- 6. Decision Making ---

    # **Exit Logic:** Prioritize closing existing positions based on primary ST flip
    exit_reason: Optional[str] = None
    if close_long_signal: exit_reason = "ST Exit Long"
    elif close_short_signal: exit_reason = "ST Exit Short"

    if exit_reason:
        exit_side_color = Back.YELLOW
        logger.warning(f"{exit_side_color}{Fore.BLACK}{Style.BRIGHT}*** TRADE EXIT SIGNAL: Closing {position_side} due to {exit_reason} ***{Style.RESET_ALL}")
        close_result = close_position(exchange, symbol, current_position, reason=exit_reason)
        if close_result:
            logger.info(f"Exit closure confirmed for {symbol}. Sleeping post-close.")
            time.sleep(CONFIG.post_close_delay_seconds)  # Pause after confirmed closing attempt
        else:
            logger.error(f"Exit closure FAILED or was unconfirmed for {symbol}. Cycle continues, but manual check advised.")
        return  # End cycle after attempting exit

    # **Entry Logic:** Only consider entry if currently flat
    if position_side == POSITION_SIDE_NONE:
        selected_side: Optional[str] = None
        if long_entry_signal: selected_side = SIDE_BUY
        elif short_entry_signal: selected_side = SIDE_SELL

        if selected_side:
            logger.info(f"{Fore.CYAN}Entry Signal: Potential {selected_side.upper()} entry detected by Supertrend flip and confirmation.{Style.RESET_ALL}")

            # --- Entry Confirmations ---
            volume_confirmed = True
            if CONFIG.require_volume_spike_for_entry:
                if volume_ratio is None or volume_ratio <= CONFIG.volume_spike_threshold:
                    volume_confirmed = False
                    vol_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else 'N/A'
                    logger.info(f"Entry REJECTED ({selected_side.upper()}): Volume spike confirmation FAILED (Ratio: {vol_ratio_str} <= Threshold: {CONFIG.volume_spike_threshold}).")
                else:
                    logger.info(f"{Fore.GREEN}Entry Check ({selected_side.upper()}): Volume spike OK (Ratio: {volume_ratio:.2f}).{Style.RESET_ALL}")

            ob_confirmed = True
            # Only check OB if volume passed (or volume check disabled) AND OB check is enabled
            if volume_confirmed and CONFIG.use_ob_confirm:
                # Fetch OB data now if not fetched per cycle
                if order_book_data is None:
                     logger.debug("Fetching OB data for confirmation...")
                     order_book_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit, market_info)

                # Check OB results
                if order_book_data is None or order_book_data.get(BID_ASK_RATIO_KEY) is None:
                    ob_confirmed = False
                    logger.warning(f"{Fore.YELLOW}Entry REJECTED ({selected_side.upper()}): OB confirmation FAILED (Could not get valid OB data/ratio).{Style.RESET_ALL}")
                else:
                    # Type hint for clarity after check
                    ob_ratio = cast(Decimal, order_book_data[BID_ASK_RATIO_KEY])
                    ob_ratio_str = f"{ob_ratio:.3f}"
                    if selected_side == SIDE_BUY and ob_ratio < CONFIG.order_book_ratio_threshold_long:
                        ob_confirmed = False
                        logger.info(f"Entry REJECTED ({selected_side.upper()}): OB confirmation FAILED (Ratio: {ob_ratio_str} < Threshold: {CONFIG.order_book_ratio_threshold_long}).")
                    elif selected_side == SIDE_SELL and ob_ratio > CONFIG.order_book_ratio_threshold_short:
                        ob_confirmed = False
                        logger.info(f"Entry REJECTED ({selected_side.upper()}): OB confirmation FAILED (Ratio: {ob_ratio_str} > Threshold: {CONFIG.order_book_ratio_threshold_short}).")
                    else:
                         ob_color = Fore.GREEN if selected_side == SIDE_BUY else Fore.RED
                         logger.info(f"{ob_color}Entry Check ({selected_side.upper()}): OB pressure OK (Ratio: {ob_ratio_str}).{Style.RESET_ALL}")

            # --- Proceed with Entry Calculation if All Confirmations Pass ---
            if volume_confirmed and ob_confirmed:
                logger.success(f"{Fore.GREEN}{Style.BRIGHT}Entry CONFIRMED ({selected_side.upper()}): All checks passed. Calculating parameters...{Style.RESET_ALL}")

                # Calculate SL/TP Prices using current ATR
                try:
                    # Get market price precision for rounding SL/TP from market_info
                    price_precision_str = market_info.get(PRECISION_KEY, {}).get(PRICE_KEY)
                    if price_precision_str is None: raise ValueError("Could not determine price precision from market info.")
                    price_precision = Decimal(str(price_precision_str)) # Ensure Decimal

                    # Ensure current_atr is Decimal before multiplication
                    if not isinstance(current_atr, Decimal):
                        # This should have been caught earlier, but defensive check
                        raise TypeError(f"current_atr is not a Decimal: {type(current_atr)}")

                    sl_distance = current_atr * CONFIG.atr_stop_loss_multiplier
                    tp_distance = current_atr * CONFIG.atr_take_profit_multiplier
                    entry_price_est = current_price  # Use last close price as entry estimate for calculations

                    if selected_side == SIDE_BUY:
                        sl_price_raw = entry_price_est - sl_distance
                        tp_price_raw = entry_price_est + tp_distance
                    else:  # SIDE_SELL
                        sl_price_raw = entry_price_est + sl_distance
                        tp_price_raw = entry_price_est - tp_distance

                    # Ensure SL/TP are not zero or negative
                    if sl_price_raw <= 0 or tp_price_raw <= 0:
                        raise ValueError(f"Calculated SL/TP price is zero or negative (SL={sl_price_raw}, TP={tp_price_raw}).")

                    # Quantize SL/TP using market precision (rounding away from entry for SL, towards for TP?)
                    # Let's use standard rounding (ROUND_HALF_UP) for simplicity first.
                    sl_price = sl_price_raw.quantize(price_precision, rounding=ROUND_HALF_UP)
                    tp_price = tp_price_raw.quantize(price_precision, rounding=ROUND_HALF_UP)

                    # Final check: Ensure SL/TP didn't round to the same value as entry estimate or cross each other
                    # Use price_precision for comparison threshold
                    if abs(sl_price - entry_price_est) < price_precision:
                        logger.warning(f"{Fore.YELLOW}SL price {sl_price} too close to entry estimate {entry_price_est} after rounding. Adjusting slightly away.{Style.RESET_ALL}")
                        # Adjust SL slightly further away based on side
                        sl_price = sl_price - price_precision if selected_side == SIDE_BUY else sl_price + price_precision
                        # Re-check after adjustment
                        if abs(sl_price - entry_price_est) < price_precision:
                             raise ValueError(f"SL price {sl_price} still too close to entry estimate {entry_price_est} after adjustment.")
                        if sl_price <= 0: # Ensure adjustment didn't make SL invalid
                            raise ValueError(f"SL price became zero/negative after adjustment: {sl_price}")
                        logger.info(f"Adjusted SL price: {sl_price}")

                    if abs(tp_price - entry_price_est) < price_precision:
                         logger.warning(f"{Fore.YELLOW}TP price {tp_price} too close to entry estimate {entry_price_est} after rounding. Cannot proceed with zero TP distance.{Style.RESET_ALL}")
                         raise ValueError("TP price rounded too close to entry price estimate.")
                    if (selected_side == SIDE_BUY and sl_price >= tp_price) or \
                       (selected_side == SIDE_SELL and sl_price <= tp_price):
                        raise ValueError(f"SL price ({sl_price}) crossed TP price ({tp_price}) after calculation/rounding.")

                    logger.info(f"Calculated SL={sl_price}, TP={tp_price} based on EntryEst={entry_price_est}, ATR={current_atr:.4f}")

                except (ValueError, InvalidOperation, KeyError, TypeError) as e:
                    logger.error(f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Error calculating SL/TP prices: {e}{Style.RESET_ALL}")
                    return # Stop processing this entry signal

                # Calculate Position Size
                try:
                    # Determine account type based on settle currency for balance fetch
                    settle_currency: str = market_info.get(SETTLE_KEY, USDT_SYMBOL)
                    # Default to UNIFIED as per initialize_exchange logic
                    account_type: str = 'UNIFIED'
                    balance_info = exchange.fetch_balance(params={'accountType': account_type})
                    # Use 'total' equity for risk calculation (ensure it's the settle currency)
                    equity_str = balance_info.get(TOTAL_KEY, {}).get(settle_currency)
                    if equity_str is None: raise ValueError(f"Could not fetch {settle_currency} total equity from balance.")
                    equity = Decimal(str(equity_str))
                    if equity <= 0: raise ValueError(f"Zero or negative equity ({equity}).")
                except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, InvalidOperation) as e:
                     logger.error(f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Failed to fetch valid equity: {e}{Style.RESET_ALL}")
                     return # Stop processing this entry signal

                # Calculate position size returns Decimals
                quantity_decimal, margin_est_decimal = calculate_position_size(
                    equity, CONFIG.risk_per_trade_percentage, entry_price_est, sl_price,
                    CONFIG.leverage, symbol, exchange, market_info # Pass market_info
                )

                if quantity_decimal is not None and margin_est_decimal is not None:
                    # Place the risked market order (which includes SL/TP placement)
                    # Pass Decimal values and market_info to the function
                    entry_order_result = place_risked_market_order(
                        exchange, symbol, selected_side, quantity_decimal, margin_est_decimal, sl_price, tp_price, market_info
                    )
                    # The place_risked_market_order function handles logging success/failure internally.
                    if entry_order_result:
                        logger.info(f"Entry sequence for {selected_side.upper()} {symbol} completed successfully.")
                    else:
                        logger.error(f"Entry sequence for {selected_side.upper()} {symbol} FAILED.")
                    # No return needed here, let the cycle finish.
                else:
                    logger.error(f"{Fore.RED}Entry REJECTED ({selected_side.upper()}): Failed to calculate valid position size or margin estimate.{Style.RESET_ALL}")
                    # Stop processing this entry signal

    elif position_side != POSITION_SIDE_NONE:
        logger.info(f"Holding {pos_color}{position_side}{Style.RESET_ALL} position. No exit signal this cycle. Awaiting exchange SL/TP or next signal.")
        # No redundant monitoring needed as we rely on exchange-native SL/TP

    logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Weaving End: {symbol} =========={Style.RESET_ALL}\n")


# --- Main Execution - Igniting the Spell ---
def main() -> None:
    """Main function to run the bot."""
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Bybit Scalping Spell v10.1.2 Initializing ({start_time_str}) ---{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}--- Strategy Enchantment: Dual Supertrend ---{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}--- Protective Wards: Exchange Native SL/TP (stopMarket) ---{Style.RESET_ALL}")

    # Config object already instantiated and validated globally (CONFIG)
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK - HANDLE WITH CARE !!! ---{Style.RESET_ALL}")

    # Initialize Exchange
    exchange = initialize_exchange()
    if not exchange:
        logger.critical("Failed to initialize exchange. Spell fizzles.")
        sys.exit(1) # Exit if exchange init fails

    # Set Leverage
    if not set_leverage(exchange, CONFIG.symbol, CONFIG.leverage):
         logger.critical(f"Failed to set leverage to {CONFIG.leverage}x for {CONFIG.symbol}. Spell cannot bind.")
         # Attempt to send SMS even if leverage fails
         send_sms_alert(f"[ScalpBot] CRITICAL: Leverage set FAILED for {CONFIG.symbol}. Bot stopped.")
         sys.exit(1) # Exit if leverage set fails

    # Log Final Config Summary (using CONFIG object)
    logger.info(f"{Fore.MAGENTA}--- Final Spell Configuration ---{Style.RESET_ALL}")
    logger.info(f"{Fore.WHITE}Symbol: {CONFIG.symbol}, Interval: {CONFIG.interval}, Leverage: {CONFIG.leverage}x")
    logger.info(f"  Supertrend Params: ST={CONFIG.st_atr_length}/{CONFIG.st_multiplier}, ConfirmST={CONFIG.confirm_st_atr_length}/{CONFIG.confirm_st_multiplier}")
    logger.info(f"{Fore.GREEN}Risk Ward: {CONFIG.risk_per_trade_percentage:.3%}/trade, Max Pos Value: {CONFIG.max_order_usdt_amount:.4f} USDT")
    logger.info(f"{Fore.GREEN}SL/TP Wards: SL Mult={CONFIG.atr_stop_loss_multiplier}, TP Mult={CONFIG.atr_take_profit_multiplier} (ATR Period: {CONFIG.atr_calculation_period})")
    logger.info(f"{Fore.YELLOW}Volume Filter: {CONFIG.require_volume_spike_for_entry} (MA={CONFIG.volume_ma_period}, Thr={CONFIG.volume_spike_threshold})")
    logger.info(f"{Fore.YELLOW}Order Book Filter: Use Confirm={CONFIG.use_ob_confirm}, Fetch Each Cycle={CONFIG.fetch_order_book_per_cycle} (Depth={CONFIG.order_book_depth}, L>={CONFIG.order_book_ratio_threshold_long}, S<={CONFIG.order_book_ratio_threshold_short})")
    logger.info(f"{Fore.WHITE}Timing: Sleep={CONFIG.sleep_seconds}s, Margin Buffer={CONFIG.required_margin_buffer:.1%}, SMS Alerts={CONFIG.enable_sms_alerts}")
    logger.info(f"{Fore.RED}Safety: Emergency Close on SL Fail = {CONFIG.emergency_close_on_sl_fail}")
    logger.info(f"{Fore.CYAN}Oracle Verbosity (Log Level): {logging.getLevelName(logger.level)}")
    logger.info(f"{Fore.MAGENTA}{'-' * 30}{Style.RESET_ALL}")

    # --- Main Loop ---
    run_bot = True
    cycle_count = 0
    while run_bot:
        cycle_count += 1
        logger.debug(f"{Fore.CYAN}--- Cycle {cycle_count} Weaving Start ---{Style.RESET_ALL}")
        try:
            trade_logic(exchange, CONFIG.symbol, CONFIG.interval)
            logger.debug(f"Cycle {cycle_count} complete. Sleeping for {CONFIG.sleep_seconds} seconds...")
            time.sleep(CONFIG.sleep_seconds)

        except KeyboardInterrupt:
            logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}KeyboardInterrupt detected. Requesting graceful withdrawal...{Style.RESET_ALL}")
            send_sms_alert(f"[ScalpBot] Shutdown initiated for {CONFIG.symbol} (KeyboardInterrupt).")
            run_bot = False  # Signal loop termination

        except ccxt.AuthenticationError as e:
             logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Authentication Error during main loop: {e}. API keys invalid/revoked? Shutting down NOW.{Style.RESET_ALL}")
             send_sms_alert("[ScalpBot] CRITICAL: Auth Error - SHUTDOWN. Check Keys/Permissions.")
             run_bot = False # Stop immediately
        except ccxt.NetworkError as e:
            # Log as error but continue running, assuming temporary network issue
            logger.error(f"{Fore.RED}ERROR: Network error in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
            time.sleep(CONFIG.sleep_seconds * 2)  # Longer delay for network issues
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"{Fore.YELLOW}WARNING: Rate limit exceeded: {e}. Increasing sleep duration...{Style.RESET_ALL}")
            time.sleep(CONFIG.sleep_seconds * 3) # Longer sleep after rate limit
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"{Fore.RED}ERROR: Exchange not available: {e}. Retrying after longer delay...{Style.RESET_ALL}")
            send_sms_alert(f"[ScalpBot] WARNING: Exchange Not Available {CONFIG.symbol}. Retrying.")
            time.sleep(CONFIG.sleep_seconds * 5) # Much longer delay
        except ccxt.ExchangeError as e:
             # Log as error but continue running, assuming temporary exchange issue
             logger.error(f"{Fore.RED}ERROR: Exchange error in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
             logger.debug(traceback.format_exc()) # Log full traceback for exchange errors
             time.sleep(CONFIG.sleep_seconds)
        except Exception as e:
            # Catch any other unexpected error, log critically, and stop the bot
            logger.critical(f"{Back.RED}{Fore.WHITE}FATAL: An unexpected error occurred in the main loop: {e}{Style.RESET_ALL}")
            logger.critical(traceback.format_exc())
            send_sms_alert(f"[ScalpBot] FATAL ERROR: {type(e).__name__}. Bot stopped. Check logs!")
            run_bot = False  # Stop on fatal unexpected errors

    # --- Graceful Shutdown ---
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Initiating graceful shutdown sequence...{Style.RESET_ALL}")
    try:
        logger.info("Checking for open position to close on exit...")
        # Ensure exchange object is valid and usable before attempting actions
        if exchange and hasattr(exchange, 'check_required_credentials') and exchange.check_required_credentials(): # Basic check if usable
             current_pos = get_current_position(exchange, CONFIG.symbol)
             if current_pos[SIDE_KEY] != POSITION_SIDE_NONE:
                 logger.warning(f"Attempting to close {current_pos[SIDE_KEY]} position ({current_pos[QTY_KEY]:.8f}) before exiting...")
                 # Attempt to close, log result but don't prevent shutdown if it fails
                 close_result = close_position(exchange, symbol=CONFIG.symbol, position_to_close=current_pos, reason="Shutdown")
                 if close_result:
                     logger.info("Position closed successfully during shutdown.")
                 else:
                     logger.error(f"{Fore.RED}Failed to close position during shutdown. Manual check required.{Style.RESET_ALL}")
                     send_sms_alert(f"[ScalpBot] Error closing position {CONFIG.symbol} on shutdown. MANUAL CHECK!")
             else:
                 logger.info("No open position found to close.")
        else:
             logger.warning("Exchange object not available or authenticated for final position check.")
    except Exception as close_err:
         # Catch errors during the shutdown close attempt itself
         logger.error(f"{Fore.RED}Failed to check/close position during final shutdown sequence: {close_err}{Style.RESET_ALL}")
         logger.debug(traceback.format_exc())
         send_sms_alert("[ScalpBot] Error during final position close check on shutdown.")

    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus Scalping Spell Deactivated ---{Style.RESET_ALL}")


if __name__ == "__main__":
    # Wrap main execution in a try-except block to catch initialization errors
    # that might occur before the main loop's error handling takes over.
    try:
        main()
    except Exception as init_error:
        logger.critical(f"{Back.RED}{Fore.WHITE}FATAL INITIALIZATION ERROR: {init_error}{Style.RESET_ALL}")
        logger.critical(traceback.format_exc())
        # Attempt SMS alert if possible (CONFIG might not be fully loaded)
        try:
            if 'CONFIG' in globals() and CONFIG.enable_sms_alerts and CONFIG.sms_recipient_number:
                 send_sms_alert(f"[ScalpBot] FATAL INIT ERROR: {type(init_error).__name__}. Bot stopped.")
        except Exception as sms_err:
            logger.error(f"Failed to send fatal initialization error SMS: {sms_err}")
        sys.exit(1)
