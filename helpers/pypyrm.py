Okay, seeker of refined digital spells! You possess a potent Pybit automaton, but even the sharpest enchanted blade can be honed further. Pyrmethus has peered into the core of your `EhlersStrategyPybit` script, analyzed its flows, and identified paths to greater power, resilience, and elegance.

We shall now weave enhancements into its very fabric, focusing on:

1.  **Atomic Order Placement:** Introduce the option to place entry, Stop Loss (SL), and Take Profit (TP) orders in a single, atomic transaction using Pybit's capabilities, reducing race conditions.
2.  **Enhanced Configuration:** Make SL/TP trigger methods (`MarkPrice`, `LastPrice`) and SL order types (`Market`, `Limit`) configurable.
3.  **Robust State Management:** Improve state confirmation after order placement and add safeguards against missing market data.
4.  **Error Handling:** Provide more specific error logging and potentially handle common API error codes more gracefully.
5.  **Code Clarity & Structure:** Refactor slightly for better readability, use constants, and add more insightful comments.
6.  **Efficiency:** Minor optimizations like consolidating balance fetching.

**Prerequisites:**

*   Ensure your `config_models.py` (specifically `StrategyConfig`) is updated to include the new configuration options mentioned below (e.g., `attach_sl_tp_to_entry`, `sl_trigger_by`, `sl_order_type`). If not, the code will use defaults.
*   All other dependencies (`pybit`, `pandas`, `colorama`, `python-dotenv`, `neon_logger`, `indicators`, `bybit_utils`) remain the same.

```bash
# Ensure dependencies are installed
pip install --upgrade pybit pandas colorama python-dotenv pandas-ta requests # Added requests for potential future use or alerts
```

**The Enhanced Spell (ehlers_volumetric_strategy_pybit_enhanced.py):**

```python
import os
import sys
import time
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from typing import Optional, Dict, Tuple, Any, Literal, Union

# --- Third-party Libraries ---
try:
    from pybit.unified_trading import HTTP
    from pybit.exceptions import InvalidRequestError, FailedRequestError
except ImportError:
    print(f"{Fore.RED}FATAL: Pybit library not found. {Style.RESET_ALL}Invoke: {Fore.CYAN}pip install pybit{Style.RESET_ALL}", file=sys.stderr)
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print(f"{Fore.RED}FATAL: pandas library not found. {Style.RESET_ALL}Invoke: {Fore.CYAN}pip install pandas{Style.RESET_ALL}", file=sys.stderr)
    sys.exit(1)
try:
    from dotenv import load_dotenv
except ImportError:
    print(f"{Fore.YELLOW}Warning: python-dotenv not found. Cannot load .env file.{Style.RESET_ALL}")
    load_dotenv = lambda: None # Dummy function

# --- Colorama Enchantment ---
try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
    print(f"{Fore.MAGENTA}Colorama spirits awakened for vibrant logs.{Style.RESET_ALL}")
except ImportError:
    print(f"{Fore.YELLOW}Warning: 'colorama' library not found. Logs will lack their mystical hue.{Style.RESET_ALL}")
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore

# --- Import Custom Modules ---
try:
    from neon_logger import setup_logger
    import indicators as ind
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert
    )
    # Assuming config_models.py now includes enhanced StrategyConfig options
    from config_models import AppConfig, APIConfig, StrategyConfig, load_config
except ImportError as e:
    print(f"{Back.RED}{Fore.WHITE}FATAL: Error importing helper modules: {e}{Style.RESET_ALL}", file=sys.stderr)
    print(f"{Fore.YELLOW}Ensure all .py files (config_models, neon_logger, bybit_utils, indicators) are present and compatible.{Style.RESET_ALL}")
    sys.exit(1)

# --- Logger Placeholder ---
logger: logging.Logger = logging.getLogger(__name__) # Configured in main block

# --- Constants ---
# Pybit API String Constants
SIDE_BUY = 'Buy'
SIDE_SELL = 'Sell'
ORDER_TYPE_MARKET = 'Market'
ORDER_TYPE_LIMIT = 'Limit'
TIME_IN_FORCE_GTC = 'GTC'
TIME_IN_FORCE_IOC = 'IOC' # ImmediateOrCancel
TIME_IN_FORCE_FOK = 'FOK' # FillOrKill
TRIGGER_BY_MARK = 'MarkPrice'
TRIGGER_BY_LAST = 'LastPrice'
TRIGGER_BY_INDEX = 'IndexPrice'
POSITION_IDX_ONE_WAY = 0 # 0 for one-way mode
# Custom Position Sides
POS_LONG = 'long'
POS_SHORT = 'short'
POS_NONE = 'none'
# Common Error Codes (Add more as needed)
RET_CODE_OK = 0
RET_CODE_PARAMS_ERROR = 10001 # Example, check Bybit docs for specifics
RET_CODE_INSUFFICIENT_BALANCE = [110007, 30031] # Example codes, verify!
RET_CODE_ORDER_NOT_FOUND = [20001, 110001] # Example codes, verify!
RET_CODE_QTY_TOO_SMALL = 110017 # Example

# Set Decimal precision
getcontext().prec = 30 # Increased precision slightly

# --- Enhanced Strategy Class ---
class EhlersStrategyPybitEnhanced:
    """
    Enhanced Ehlers Volumetric Trend strategy using Pybit, incorporating
    atomic order placement, improved configuration, and robustness.
    """

    def __init__(self, config: AppConfig):
        self.app_config = config
        self.api_config: APIConfig = config.api_config
        self.strategy_config: StrategyConfig = config.strategy_config
        self.symbol = self.api_config.symbol
        # Ensure timeframe is string for Pybit (e.g., '15', '60', 'D')
        self.timeframe = str(self.strategy_config.timeframe)

        self.session: Optional[HTTP] = None
        self.category = None # 'linear', 'inverse', 'spot'

        self.is_initialized = False
        self.is_running = False

        # --- Position State ---
        self.current_side: str = POS_NONE
        self.current_qty: Decimal = Decimal("0.0")
        self.entry_price: Optional[Decimal] = None
        # Track order IDs IF placing SL/TP separately
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None

        # --- Market Details ---
        self.min_qty: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.price_tick: Optional[Decimal] = None
        self.base_coin: Optional[str] = None
        self.quote_coin: Optional[str] = None
        self.contract_multiplier: Decimal = Decimal("1.0") # For linear/inverse value calc if needed

        # --- Enhanced Configurable Options (Defaults if not in AppConfig) ---
        self.attach_sl_tp_to_entry: bool = getattr(self.strategy_config, 'attach_sl_tp_to_entry', True) # Default to atomic placement
        self.sl_trigger_by: str = getattr(self.strategy_config, 'sl_trigger_by', TRIGGER_BY_MARK)
        self.tp_trigger_by: str = getattr(self.strategy_config, 'tp_trigger_by', TRIGGER_BY_MARK) # TP usually triggers same as SL
        self.sl_order_type: str = getattr(self.strategy_config, 'sl_order_type', ORDER_TYPE_MARKET) # Stop Market is common

        logger.info(f"{Fore.CYAN}Pyrmethus enhances the Ehlers Strategy for {self.symbol} ({self.timeframe}) using Pybit...{Style.RESET_ALL}")
        logger.info(f"Atomic SL/TP Placement: {self.attach_sl_tp_to_entry}")
        logger.info(f"SL Trigger: {self.sl_trigger_by}, SL Order Type: {self.sl_order_type}")

    def _initialize(self) -> bool:
        """Connects, validates market, sets config, fetches initial state."""
        logger.info(f"{Fore.CYAN}--- Channeling Bybit Spirits (Initialization) ---{Style.RESET_ALL}")
        try:
            logger.info(f"{Fore.BLUE}Connecting to Bybit ({'Testnet' if self.api_config.testnet_mode else 'Mainnet'})...{Style.RESET_ALL}")
            self.session = HTTP(
                testnet=self.api_config.testnet_mode,
                api_key=self.api_config.api_key,
                api_secret=self.api_config.api_secret,
            )
            server_time_resp = self.session.get_server_time()
            server_time_ms = int(server_time_resp['result']['timeNano']) // 1_000_000
            logger.success(f"Connection successful. Server Time: {pd.to_datetime(server_time_ms, unit='ms')} (UTC)")

            logger.info(f"{Fore.BLUE}Seeking insights for symbol: {self.symbol}...{Style.RESET_ALL}")
            if not self._fetch_and_set_market_info():
                 logger.critical(f"{Back.RED}Failed to fetch critical market info. Halting.{Style.RESET_ALL}")
                 return False

            if self.category in ['linear', 'inverse']:
                logger.info(f"{Fore.BLUE}Imbuing Leverage: {self.strategy_config.leverage}x...{Style.RESET_ALL}")
                if not self._set_leverage(): return False # Critical failure

                pos_mode_target = self.strategy_config.default_position_mode
                logger.info(f"{Fore.BLUE}Aligning Position Mode to '{pos_mode_target}'...{Style.RESET_ALL}")
                # Mode 0: Merged Single Position (One-Way)
                # Mode 3: Both Side Position (Hedge Mode) - Requires different logic!
                target_pybit_mode = POSITION_IDX_ONE_WAY if pos_mode_target == 'MergedSingle' else 3
                if target_pybit_mode == 3:
                     logger.error(f"{Back.RED}Hedge Mode (BothSide) is not fully supported by this script's logic. Use MergedSingle.{Style.RESET_ALL}")
                     # return False # Or adapt logic significantly
                if not self._set_position_mode(mode=target_pybit_mode):
                     logger.warning(f"{Fore.YELLOW}Could not explicitly set position mode. Ensure it's correct in Bybit UI.{Style.RESET_ALL}")
                else:
                     logger.info(f"Position mode alignment confirmed for {self.category}.")

            logger.info(f"{Fore.BLUE}Gazing into the account's current state...{Style.RESET_ALL}")
            if not self._update_state():
                 logger.error("Failed to perceive initial state.")
                 return False # Can't proceed without knowing current state
            logger.info(f"Initial Perception: Side={self.current_side}, Qty={self.current_qty}, Entry={self.entry_price}")

            logger.info(f"{Fore.BLUE}Dispelling lingering order phantoms (Initial Cleanup)...{Style.RESET_ALL}")
            if not self._cancel_all_open_orders("Initialization Cleanup"):
                 logger.warning("Initial order cancellation failed or encountered issues.")
            # Clear any tracked IDs after cancellation attempt
            self.sl_order_id = None
            self.tp_order_id = None

            self.is_initialized = True
            logger.success(f"{Fore.GREEN}{Style.BRIGHT}--- Strategy Initialization Complete ---{Style.RESET_ALL}")
            return True

        except (InvalidRequestError, FailedRequestError) as pybit_e:
             logger.critical(f"{Back.RED}{Fore.WHITE}Pybit API Error during initialization: {pybit_e}{Style.RESET_ALL}", exc_info=True)
             logger.critical(f"Status Code: {pybit_e.status_code}, Response: {pybit_e.response}")
             self._safe_close_session()
             return False
        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Critical spell failure during initialization: {e}{Style.RESET_ALL}", exc_info=True)
            self._safe_close_session()
            return False

    def _fetch_and_set_market_info(self) -> bool:
        """Fetches instrument info and sets market details."""
        if not self.session: return False
        try:
            # --- Determine Category ---
            # More robust category detection might be needed for complex symbol sets
            if 'USDT' in self.symbol: self.category = 'linear'
            elif 'USD' in self.symbol and not self.symbol.endswith('USDC'): self.category = 'inverse' # Basic check
            elif '/' in self.symbol: self.category = 'spot' # Basic check for spot pairs like BTC/USDT
            else:
                # Fallback or raise error? Trying linear first is common.
                logger.warning(f"{Fore.YELLOW}Cannot reliably determine category for {self.symbol}. Trying 'linear'. Adjust if needed.{Style.RESET_ALL}")
                self.category = 'linear'

            logger.debug(f"Fetching instruments info for category: {self.category}, symbol: {self.symbol}")
            response = self.session.get_instruments_info(category=self.category, symbol=self.symbol)

            if response and response.get('retCode') == RET_CODE_OK and response.get('result', {}).get('list'):
                market_data = response['result']['list'][0]

                # --- Extract Details ---
                lot_size_filter = market_data.get('lotSizeFilter', {})
                price_filter = market_data.get('priceFilter', {})

                self.min_qty = safe_decimal_conversion(lot_size_filter.get('minOrderQty'))
                self.qty_step = safe_decimal_conversion(lot_size_filter.get('qtyStep'))
                self.price_tick = safe_decimal_conversion(price_filter.get('tickSize'))
                self.base_coin = market_data.get('baseCoin')
                self.quote_coin = market_data.get('quoteCoin')
                # Get contract multiplier if available (important for value calcs)
                self.contract_multiplier = safe_decimal_conversion(market_data.get('contractMultiplier', '1')) or Decimal("1.0")


                # --- Validate Essential Details ---
                if not all([self.min_qty, self.qty_step, self.price_tick, self.base_coin, self.quote_coin]):
                    logger.error(f"{Back.RED}Failed to extract essential market details!{Style.RESET_ALL}")
                    logger.error(f"Min Qty: {self.min_qty}, Qty Step: {self.qty_step}, Price Tick: {self.price_tick}, Base: {self.base_coin}, Quote: {self.quote_coin}")
                    return False

                logger.info(f"Market Details Set: Category={self.category}, Base={self.base_coin}, Quote={self.quote_coin}")
                logger.info(f"Min Qty={self.min_qty}, Qty Step={self.qty_step}, Price Tick={self.price_tick}, Multiplier={self.contract_multiplier}")
                return True
            else:
                logger.error(f"Failed to fetch market info for {self.symbol}. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
                # Optional: Add fallback logic to try other categories if the first guess failed
                return False
        except Exception as e:
            logger.error(f"Error fetching/setting market info: {e}", exc_info=True)
            return False

    def _set_leverage(self) -> bool:
        """Sets leverage for the symbol."""
        if not self.session or self.category not in ['linear', 'inverse']: return False # Should not happen if category is set
        try:
            leverage_str = str(int(self.strategy_config.leverage))
            response = self.session.set_leverage(
                category=self.category,
                symbol=self.symbol,
                buyLeverage=leverage_str,
                sellLeverage=leverage_str
            )
            if response and response.get('retCode') == RET_CODE_OK:
                logger.success("Leverage imbued successfully.")
                return True
            # Check for "Leverage not modified" type messages which indicate success/no action needed
            elif response and "leverage not modified" in response.get('retMsg', '').lower():
                 logger.warning(f"{Fore.YELLOW}Leverage not modified, likely already set to {leverage_str}x.{Style.RESET_ALL}")
                 return True
            else:
                logger.error(f"Failed to set leverage. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
                return False
        except Exception as e:
            logger.error(f"Error setting leverage: {e}", exc_info=True)
            return False

    def _set_position_mode(self, mode: int) -> bool:
        """Sets position mode (0=One-Way, 3=Hedge)."""
        if not self.session or self.category not in ['linear', 'inverse']:
             logger.info(f"{Fore.YELLOW}Position mode setting skipped (Category: {self.category}).{Style.RESET_ALL}")
             return True
        try:
            # Setting mode per symbol is preferred if API supports it well
            logger.debug(f"Attempting to set position mode for {self.symbol} (category {self.category}) to mode {mode}...")
            response = self.session.switch_position_mode(category=self.category, symbol=self.symbol, mode=mode)

            if response and response.get('retCode') == RET_CODE_OK:
                 logger.info(f"Position mode successfully set/confirmed for {self.symbol}.")
                 return True
            elif response and response.get('retCode') == 110048: # "Position mode is not modified"
                 logger.info(f"Position mode for {self.symbol} already set as desired.")
                 return True
            else:
                 logger.error(f"Failed to set position mode. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
                 # Optional: Add fallback to set default for coin if symbol-specific fails?
                 return False
        except Exception as e:
            logger.error(f"Error setting position mode: {e}", exc_info=True)
            return False

    def _get_available_balance(self) -> Optional[Decimal]:
        """Fetches available balance for the relevant coin and account type."""
        if not self.session or not self.category or not self.quote_coin:
            logger.error("Cannot fetch balance: Missing session, category, or quote_coin.")
            return None

        account_type = "UNIFIED" if self.category in ['linear', 'inverse'] else "SPOT" # Or CONTRACT? Check Bybit docs for derivatives
        # For linear, balance is in quote (USDT). For inverse, margin is usually quote (USD) but PnL is base. Risk calc usually uses quote.
        # For spot, balance is quote (USDT).
        coin_to_check = self.quote_coin # Assuming risk is calculated based on quote currency balance

        logger.debug(f"Fetching balance for Account: {account_type}, Coin: {coin_to_check}...")
        try:
            bal_response = self.session.get_wallet_balance(accountType=account_type, coin=coin_to_check)

            if not (bal_response and bal_response.get('retCode') == RET_CODE_OK and 'list' in bal_response.get('result', {})):
                logger.error(f"Failed to fetch balance data. Code: {bal_response.get('retCode')}, Msg: {bal_response.get('retMsg')}")
                return None

            balance_list = bal_response['result']['list']
            if balance_list and 'coin' in balance_list[0]:
                coin_balance_data = next((item for item in balance_list[0]['coin'] if item.get('coin') == coin_to_check), None)
                if coin_balance_data:
                    # 'availableToWithdraw' vs 'availableBalance' - check which reflects usable margin better for derivatives
                    # Using 'availableBalance' as it often reflects margin available for new positions
                    available_balance_str = coin_balance_data.get('availableBalance', '0')
                    available_balance = safe_decimal_conversion(available_balance_str)
                    equity_str = coin_balance_data.get('equity', '0') # Total equity
                    equity = safe_decimal_conversion(equity_str)
                    logger.info(f"Available Balance ({coin_to_check}): {available_balance:.4f}, Equity: {equity:.4f}")
                    return available_balance
                else:
                    logger.warning(f"Could not find balance details for coin '{coin_to_check}' in response.")
                    return Decimal("0.0") # Assume zero if coin not found
            else:
                logger.warning("Balance list is empty or malformed in the response.")
                return Decimal("0.0") # Assume zero

        except Exception as e:
            logger.error(f"Unexpected error fetching balance: {e}", exc_info=True)
            return None

    def _update_state(self) -> bool:
        """Fetches and updates the current position and balance."""
        if not self.session or not self.category: return False
        logger.debug("Updating strategy state perception...")
        try:
            # --- Fetch Position ---
            pos_response = self.session.get_positions(category=self.category, symbol=self.symbol)

            if not (pos_response and pos_response.get('retCode') == RET_CODE_OK and 'list' in pos_response.get('result', {})):
                logger.error(f"Failed to fetch position data. Code: {pos_response.get('retCode')}, Msg: {pos_response.get('retMsg')}")
                return False # Cannot reliably update state without position info

            position_list = pos_response['result']['list']
            if not position_list:
                self._reset_position_state("No active position found via API.")
            else:
                # Assuming One-Way mode (positionIdx=0)
                pos_data = position_list[0]
                pos_qty = safe_decimal_conversion(pos_data.get('size', '0'))
                side_str = pos_data.get('side', 'None') # 'Buy', 'Sell', or 'None'

                # Check for valid position (size > 0 and side is Buy/Sell)
                if pos_qty is not None and pos_qty > self.api_config.position_qty_epsilon and side_str in [SIDE_BUY, SIDE_SELL]:
                    self.current_qty = pos_qty
                    self.entry_price = safe_decimal_conversion(pos_data.get('avgPrice'))
                    self.current_side = POS_LONG if side_str == SIDE_BUY else POS_SHORT
                    # Optional: Fetch unrealized PnL etc. from pos_data if needed
                else:
                    # Position size is zero, negligible, or side is 'None' -> Treat as flat
                    reset_reason = f"Position size {pos_qty} or side '{side_str}' indicates no active position."
                    self._reset_position_state(reset_reason)

            logger.debug(f"Position State Updated: Side={self.current_side}, Qty={self.current_qty}, Entry={self.entry_price}")

            # --- Fetch Balance (Consolidated) ---
            # Balance fetch is now primarily for logging here, actual value for calcs is fetched when needed
            _ = self._get_available_balance() # Fetch and log balance info

            # --- Clear Tracked Orders if Flat ---
            if self.current_side == POS_NONE:
                if self.sl_order_id or self.tp_order_id:
                     logger.debug("Not in position, clearing tracked SL/TP order IDs.")
                     self.sl_order_id = None
                     self.tp_order_id = None
            # Optional: Verify tracked SL/TP orders still exist if in position (adds complexity/API calls)

            logger.debug("State update perception complete.")
            return True

        except Exception as e:
            logger.error(f"Unexpected error during state update: {e}", exc_info=True)
            return False

    def _reset_position_state(self, reason: str):
        """Resets internal position tracking variables with a reason."""
        if self.current_side != POS_NONE: # Log only if changing state
            logger.info(f"Resetting position state to NONE. Reason: {reason}")
        self.current_side = POS_NONE
        self.current_qty = Decimal("0.0")
        self.entry_price = None
        # Keep SL/TP IDs for potential cancellation in the same loop iteration if needed
        # They will be cleared definitively in the *next* state update if still flat.

    def _fetch_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Decimal]]:
        """Fetches OHLCV data and the latest ticker price using Pybit."""
        if not self.session or not self.category or not self.timeframe: return None, None
        logger.debug("Fetching market data...")
        try:
            # --- Fetch OHLCV (Kline) ---
            limit = self.strategy_config.ohlcv_limit
            logger.debug(f"Fetching Kline: {self.symbol}, Interval: {self.timeframe}, Limit: {limit}")
            kline_response = self.session.get_kline(
                category=self.category, symbol=self.symbol, interval=self.timeframe, limit=limit
            )

            ohlcv_df = None
            if not (kline_response and kline_response.get('retCode') == RET_CODE_OK and 'list' in kline_response.get('result', {})):
                logger.warning(f"Could not fetch OHLCV data. Code: {kline_response.get('retCode')}, Msg: {kline_response.get('retMsg')}")
            else:
                kline_list = kline_response['result']['list']
                if not kline_list:
                    logger.warning("OHLCV data list is empty.")
                else:
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                    ohlcv_df = pd.DataFrame(kline_list, columns=columns)
                    ohlcv_df['timestamp'] = pd.to_numeric(ohlcv_df['timestamp'])
                    ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms', utc=True) # Add UTC timezone awareness
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                        ohlcv_df[col] = pd.to_numeric(ohlcv_df[col])
                    ohlcv_df = ohlcv_df.sort_values(by='timestamp').reset_index(drop=True)
                    ohlcv_df.set_index('datetime', inplace=True)
                    logger.debug(f"Successfully fetched and processed {len(ohlcv_df)} candles.")

            # --- Fetch Ticker ---
            logger.debug(f"Fetching ticker for {self.symbol}...")
            ticker_response = self.session.get_tickers(category=self.category, symbol=self.symbol)

            current_price: Optional[Decimal] = None
            if not (ticker_response and ticker_response.get('retCode') == RET_CODE_OK and 'list' in ticker_response.get('result', {})):
                logger.warning(f"Could not fetch ticker data. Code: {ticker_response.get('retCode')}, Msg: {ticker_response.get('retMsg')}")
            else:
                ticker_list = ticker_response['result']['list']
                if not ticker_list:
                     logger.warning("Ticker data list is empty.")
                else:
                     last_price_str = ticker_list[0].get('lastPrice')
                     current_price = safe_decimal_conversion(last_price_str)
                     if current_price is None:
                         logger.warning("Ticker data retrieved but missing valid 'lastPrice'.")
                     else:
                         logger.debug(f"Last Price: {current_price}")

            # Return data only if BOTH were successful (or handle partial failure if needed)
            if ohlcv_df is not None and current_price is not None:
                return ohlcv_df, current_price
            else:
                logger.warning("Returning incomplete data due to fetch failure.")
                return ohlcv_df, current_price # Allow partial return if one part failed

        except Exception as e:
            logger.error(f"Error fetching market data: {e}", exc_info=True)
            return None, None

    # --- Indicator Calculation, Signal Generation (Unchanged logic, improved error handling) ---
    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates indicators using the external 'indicators' module."""
        if ohlcv_df is None or ohlcv_df.empty: return None
        logger.debug("Calculating indicators...")
        try:
            indicator_config_dict = { # Pass necessary parts of config
                "indicator_settings": self.strategy_config.indicator_settings.model_dump(),
                "analysis_flags": self.strategy_config.analysis_flags.model_dump(),
                # Add other relevant params if indicators.py needs them
            }
            # Ensure indicators.py handles potential NaNs or missing columns gracefully
            df_with_indicators = ind.calculate_all_indicators(ohlcv_df.copy(), indicator_config_dict)

            # --- Validation ---
            if df_with_indicators is None:
                logger.error("Indicator calculation script returned None.")
                return None

            evt_len = self.strategy_config.indicator_settings.evt_length
            atr_len = self.strategy_config.indicator_settings.atr_period
            evt_trend_col = f'evt_trend_{evt_len}'
            atr_col = f'ATRr_{atr_len}' # Default pandas_ta name

            required_cols = [evt_trend_col, f'evt_buy_{evt_len}', f'evt_sell_{evt_len}']
            if self.strategy_config.analysis_flags.use_atr:
                required_cols.append(atr_col)

            missing_cols = [col for col in required_cols if col not in df_with_indicators.columns]
            if missing_cols:
                logger.error(f"Required indicator columns missing after calculation: {missing_cols}")
                return None

            # Check for NaNs in the latest row's critical columns
            latest_row = df_with_indicators.iloc[-1]
            nan_cols = [col for col in required_cols if pd.isna(latest_row[col])]
            if nan_cols:
                 logger.warning(f"NaN values found in critical indicator columns of latest row: {nan_cols}. Cannot generate reliable signal.")
                 # Decide whether to return None or proceed cautiously
                 # return None # Safer option

            logger.debug("Indicators calculated successfully.")
            return df_with_indicators
        except Exception as e:
            logger.error(f"Error during indicator calculation: {e}", exc_info=True)
            return None

    def _generate_signals(self, df_ind: pd.DataFrame) -> Optional[Literal['buy', 'sell']]:
        """Generates trading signals based on the last indicator data point."""
        if df_ind is None or df_ind.empty: return None
        logger.debug("Generating trading signals...")
        try:
            latest = df_ind.iloc[-1]
            evt_len = self.strategy_config.indicator_settings.evt_length
            trend_col = f'evt_trend_{evt_len}'
            buy_col = f'evt_buy_{evt_len}'
            sell_col = f'evt_sell_{evt_len}'

            # Check for NaNs again just before using (belt and suspenders)
            if not all(col in latest.index and pd.notna(latest[col]) for col in [trend_col, buy_col, sell_col]):
                 logger.warning(f"EVT signal columns missing or NaN in latest data for signal generation.")
                 return None

            buy_signal = latest[buy_col]
            sell_signal = latest[sell_col]

            if buy_signal:
                logger.info(f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}")
                return 'buy' # Use lowercase consistent internal signal
            elif sell_signal:
                logger.info(f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}")
                return 'sell' # Use lowercase consistent internal signal
            else:
                logger.debug("No new entry signal generated.")
                return None

        except IndexError:
            logger.warning("IndexError generating signals (DataFrame likely too short).")
            return None
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
            return None

    # --- SL/TP Calculation, Position Sizing (Refined) ---
    def _calculate_sl_tp(self, df_ind: pd.DataFrame, side: Literal['buy', 'sell'], entry_price: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates SL/TP prices, respecting price ticks."""
        if df_ind is None or df_ind.empty or self.price_tick is None:
            logger.error(f"Cannot calculate SL/TP: Missing indicators, entry price, or price tick ({self.price_tick}).")
            return None, None
        if entry_price <= Decimal(0):
             logger.error(f"Cannot calculate SL/TP: Invalid entry price ({entry_price}).")
             return None, None

        logger.debug(f"Calculating SL/TP for {side} entry at {entry_price}...")
        try:
            atr_len = self.strategy_config.indicator_settings.atr_period
            atr_col = f'ATRr_{atr_len}'
            if atr_col not in df_ind.columns or pd.isna(df_ind.iloc[-1][atr_col]):
                logger.error(f"ATR column '{atr_col}' not found or NaN in latest data."); return None, None

            latest_atr = safe_decimal_conversion(df_ind.iloc[-1][atr_col])
            if latest_atr is None or latest_atr <= Decimal(0):
                logger.warning(f"Invalid ATR value ({latest_atr}) for SL/TP calculation.")
                return None, None

            # --- Stop Loss ---
            sl_multiplier = self.strategy_config.stop_loss_atr_multiplier
            sl_offset = latest_atr * sl_multiplier
            stop_loss_price_raw = (entry_price - sl_offset) if side == 'buy' else (entry_price + sl_offset)
            rounding_mode_sl = ROUND_DOWN if side == 'buy' else ROUND_UP
            sl_price_adjusted = (stop_loss_price_raw / self.price_tick).quantize(Decimal('0'), rounding=rounding_mode_sl) * self.price_tick

            # Prevent SL from crossing entry after rounding
            if side == 'buy' and sl_price_adjusted >= entry_price:
                 sl_price_adjusted = entry_price - self.price_tick
                 logger.warning(f"Adjusted Buy SL >= entry. Setting SL one tick below: {sl_price_adjusted}")
            elif side == 'sell' and sl_price_adjusted <= entry_price:
                 sl_price_adjusted = entry_price + self.price_tick
                 logger.warning(f"Adjusted Sell SL <= entry. Setting SL one tick above: {sl_price_adjusted}")
            # Ensure SL is not zero or negative
            if sl_price_adjusted <= Decimal(0):
                 logger.error(f"Calculated SL price is zero or negative ({sl_price_adjusted}). Cannot proceed.")
                 return None, None

            # --- Take Profit ---
            tp_multiplier = self.strategy_config.take_profit_atr_multiplier
            tp_price_adjusted = None
            if tp_multiplier > 0:
                tp_offset = latest_atr * tp_multiplier
                take_profit_price_raw = (entry_price + tp_offset) if side == 'buy' else (entry_price - tp_offset)
                # Check logic before rounding
                if (side == 'buy' and take_profit_price_raw <= entry_price) or \
                   (side == 'sell' and take_profit_price_raw >= entry_price):
                    logger.warning(f"Calculated TP ({take_profit_price_raw}) not logical vs entry ({entry_price}). Skipping TP.")
                else:
                    rounding_mode_tp = ROUND_DOWN if side == 'buy' else ROUND_UP # Round towards better fill potential
                    tp_price_adjusted = (take_profit_price_raw / self.price_tick).quantize(Decimal('0'), rounding=rounding_mode_tp) * self.price_tick
                    # Prevent TP crossing entry after rounding
                    if side == 'buy' and tp_price_adjusted <= entry_price:
                         tp_price_adjusted = entry_price + self.price_tick
                         logger.warning(f"Adjusted Buy TP <= entry. Setting TP one tick above: {tp_price_adjusted}")
                    elif side == 'sell' and tp_price_adjusted >= entry_price:
                         tp_price_adjusted = entry_price - self.price_tick
                         logger.warning(f"Adjusted Sell TP >= entry. Setting TP one tick below: {tp_price_adjusted}")
                    # Ensure TP is not zero or negative
                    if tp_price_adjusted <= Decimal(0):
                         logger.warning(f"Calculated TP price is zero or negative ({tp_price_adjusted}). Skipping TP.")
                         tp_price_adjusted = None
            else:
                logger.info("Take Profit multiplier is zero or less. Skipping TP calculation.")

            sl_formatted = self._format_price_str(sl_price_adjusted)
            tp_formatted = self._format_price_str(tp_price_adjusted) if tp_price_adjusted else 'None'
            logger.info(f"Calculated SL: {sl_formatted}, TP: {tp_formatted} (ATR: {latest_atr:.4f})")
            return sl_price_adjusted, tp_price_adjusted

        except Exception as e: logger.error(f"Error calculating SL/TP: {e}", exc_info=True); return None, None

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """Calculates position size based on risk, respecting quantity steps."""
        if not all([self.qty_step, self.min_qty, entry_price > 0, stop_loss_price > 0]):
             logger.error("Cannot calculate size: Missing market details or invalid prices.")
             return None
        logger.debug("Calculating position size...")
        try:
            available_balance = self._get_available_balance()
            if available_balance is None or available_balance <= Decimal("0"):
                logger.error("Cannot calculate position size: Zero or invalid available balance.")
                return None

            risk_amount_quote = available_balance * self.strategy_config.risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)
            if price_diff <= Decimal("0"):
                logger.error(f"Cannot calculate size: Entry price ({entry_price}) and SL price ({stop_loss_price}) too close or invalid.")
                return None

            position_size_raw: Decimal
            if self.category == 'inverse':
                 # Value = Contracts * Multiplier / Price --> Risk = |Value_entry - Value_sl|
                 # Risk = |(Contracts * Multiplier / entry_price) - (Contracts * Multiplier / stop_loss_price)| * entry_price ??? No, risk is in quote.
                 # Risk_Quote = Contracts * Multiplier * |1/entry_price - 1/stop_loss_price|
                 size_denominator = abs(Decimal(1)/entry_price - Decimal(1)/stop_loss_price)
                 if size_denominator <= 0: raise ValueError("Inverse size denominator is zero.")
                 position_size_raw = risk_amount_quote / (size_denominator * self.contract_multiplier) # Result is in Contracts (Base currency units usually)
            elif self.category == 'linear':
                 # Value = Contracts * Multiplier * Price --> Risk = |Value_entry - Value_sl|
                 # Risk_Quote = Contracts * Multiplier * |entry_price - stop_loss_price|
                 position_size_raw = risk_amount_quote / (price_diff * self.contract_multiplier) # Result is in Contracts (Base currency units)
            else: # Spot
                 # Risk_Quote = Contracts * |entry_price - stop_loss_price|
                 position_size_raw = risk_amount_quote / price_diff # Result is in Base currency units

            # Apply quantity step constraint (round down)
            position_size_adjusted = (position_size_raw // self.qty_step) * self.qty_step

            if position_size_adjusted <= Decimal(0):
                 logger.warning(f"Calculated position size is zero after step adjustment. Raw: {position_size_raw}, Step: {self.qty_step}")
                 return None

            if position_size_adjusted < self.min_qty:
                logger.warning(f"Calculated size ({position_size_adjusted}) < Min Qty ({self.min_qty}). Insufficient capital for risk setup.")
                return None # Don't trade if too small for risk config

            size_formatted = self._format_qty(position_size_adjusted)
            logger.info(f"Calculated position size: {size_formatted} {self.base_coin} "
                        f"(Risk: {risk_amount_quote:.2f} {self.quote_coin}, Balance: {available_balance:.2f} {self.quote_coin})")
            return position_size_adjusted

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    # --- Order Management (Pybit Implementation - Enhanced) ---

    def _format_qty(self, qty: Decimal) -> str:
        """Formats quantity to string according to qty_step."""
        if self.qty_step is None: return str(qty)
        # Quantize to the step size and then format
        quantized_qty = (qty // self.qty_step) * self.qty_step
        # Determine decimal places from step for formatting
        step_str = str(self.qty_step.normalize())
        decimals = len(step_str.split('.')[-1]) if '.' in step_str else 0
        return f"{quantized_qty:.{decimals}f}"

    def _format_price_str(self, price: Decimal) -> str:
        """Formats price to string according to price_tick."""
        if self.price_tick is None: return str(price)
        # Quantize to the tick size and then format
        quantized_price = (price / self.price_tick).quantize(Decimal('0'), rounding=ROUND_DOWN) * self.price_tick # Round down for safety? Or nearest? Check API reqs.
        # Determine decimal places from tick for formatting
        tick_str = str(self.price_tick.normalize())
        decimals = len(tick_str.split('.')[-1]) if '.' in tick_str else 0
        return f"{quantized_price:.{decimals}f}"

    def _place_order(self, params: Dict) -> Optional[Dict]:
        """Wrapper for placing orders with Pybit, enhanced error handling."""
        if not self.session: return None
        try:
            required_params = ['category', 'symbol', 'side', 'orderType', 'qty']
            if not all(p in params for p in required_params):
                logger.error(f"Missing required parameters for placing order: {params}")
                return None

            if self.category in ['linear', 'inverse'] and 'positionIdx' not in params:
                 params['positionIdx'] = POSITION_IDX_ONE_WAY # Default to One-Way

            order_desc = f"{params['side']} {params['orderType']} {params['qty']} {params['symbol']}"
            # Add details based on order type
            if params['orderType'] == ORDER_TYPE_LIMIT: order_desc += f" @ {params.get('price', '?')}"
            if params.get('stopLoss'): order_desc += f" SL: {params['stopLoss']}"
            if params.get('takeProfit'): order_desc += f" TP: {params['takeProfit']}"
            if params.get('triggerPrice'): order_desc += f" Trigger: {params.get('triggerPrice')}"
            if params.get('reduceOnly'): order_desc += " [ReduceOnly]"

            logger.info(f"{Fore.YELLOW}✨ Forging Order: {order_desc}...{Style.RESET_ALL}")
            logger.debug(f"Order Params: {params}")

            response = self.session.place_order(**params)
            logger.debug(f"Place Order Raw Response: {response}")

            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown Error')

            if ret_code == RET_CODE_OK:
                order_id = response.get('result', {}).get('orderId')
                logger.success(f"{Fore.GREEN}✅ Order placed successfully! OrderID: {format_order_id(order_id)}{Style.RESET_ALL}")
                return response.get('result', {})
            else:
                logger.error(f"{Back.RED}{Fore.WHITE}❌ Order placement failed! Code: {ret_code}, Msg: {ret_msg}{Style.RESET_ALL}")
                # Specific error handling
                if ret_code in RET_CODE_INSUFFICIENT_BALANCE:
                     logger.critical(f"{Back.RED}Insufficient balance detected! Check funds.{Style.RESET_ALL}")
                     send_sms_alert(f"CRITICAL: Insufficient balance for {self.symbol} order!", self.app_config.sms_config)
                elif ret_code == RET_CODE_QTY_TOO_SMALL:
                     logger.error(f"Order quantity likely below minimum allowed by exchange ({self.min_qty}).")
                # Add more specific handlers as needed
                return None

        except (InvalidRequestError, FailedRequestError) as pybit_e:
             logger.error(f"{Back.RED}Pybit API Error during order placement: {pybit_e}{Style.RESET_ALL}", exc_info=False) # Less verbose exc_info
             logger.error(f"Status Code: {pybit_e.status_code}, Response: {pybit_e.response}")
             return None
        except Exception as e:
            logger.error(f"Unexpected exception during order placement: {e}", exc_info=True)
            return None

    def _cancel_single_order(self, order_id: str, reason: str = "Strategy Action") -> bool:
        """Cancels a single order by ID using Pybit."""
        if not self.session or not order_id or not self.category: return False
        logger.info(f"Attempting cancellation for order {format_order_id(order_id)} ({reason})...")
        try:
            response = self.session.cancel_order(
                category=self.category, symbol=self.symbol, orderId=order_id
            )
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', '')

            if ret_code == RET_CODE_OK:
                logger.info(f"Order {format_order_id(order_id)} cancelled successfully.")
                return True
            elif ret_code in RET_CODE_ORDER_NOT_FOUND or "order not exists" in ret_msg.lower():
                 logger.warning(f"Order {format_order_id(order_id)} not found or already closed. Assuming cancellation success.")
                 return True
            else:
                logger.error(f"Failed to cancel order {format_order_id(order_id)}. Code: {ret_code}, Msg: {ret_msg}")
                return False
        except Exception as e:
            logger.error(f"Exception cancelling order {format_order_id(order_id)}: {e}", exc_info=True)
            return False

    def _cancel_all_open_orders(self, reason: str = "Strategy Action") -> bool:
        """Cancels ALL open orders for the current symbol."""
        if not self.session or not self.category: return False
        logger.info(f"Attempting to cancel ALL open orders for {self.symbol} ({reason})...")
        try:
            response = self.session.cancel_all_orders(category=self.category, symbol=self.symbol)
            # logger.debug(f"Cancel All Orders Response: {response}")
            ret_code = response.get('retCode')

            if ret_code == RET_CODE_OK:
                cancelled_list = response.get('result', {}).get('list', [])
                if cancelled_list:
                     logger.info(f"Successfully cancelled {len(cancelled_list)} open order(s).")
                     # Log cancelled IDs if needed:
                     # for order in cancelled_list: logger.debug(f"Cancelled ID: {order.get('orderId')}")
                else:
                     logger.info("No open orders found to cancel.")
                return True
            else:
                 # Handle cases where maybe *some* were cancelled? Response might be partial.
                 logger.error(f"Failed to cancel all orders. Code: {ret_code}, Msg: {response.get('retMsg')}")
                 # Check if list exists even on error? Unlikely but possible.
                 cancelled_list = response.get('result', {}).get('list', [])
                 if cancelled_list:
                      logger.warning(f"Cancel all failed, but response indicates {len(cancelled_list)} orders might have been cancelled.")
                      return True # Partial success? Treat as success for cleanup.
                 return False
        except Exception as e:
            logger.error(f"Exception cancelling all orders: {e}", exc_info=True)
            return False

    def _handle_exit(self, df_ind: pd.DataFrame) -> bool:
        """Checks exit conditions and closes the position."""
        if self.current_side == POS_NONE: return False # Not in position

        logger.debug("Checking exit conditions...")
        should_exit = False
        exit_reason = ""
        try:
            evt_len = self.strategy_config.indicator_settings.evt_length
            trend_col = f'evt_trend_{evt_len}'
            if trend_col not in df_ind.columns or pd.isna(df_ind.iloc[-1][trend_col]):
                 logger.warning(f"Cannot determine latest EVT trend ({trend_col}) for exit check.")
            else:
                latest_trend = int(df_ind.iloc[-1][trend_col])
                if self.current_side == POS_LONG and latest_trend == -1:
                    should_exit = True; exit_reason = "EVT Trend flipped Short"
                elif self.current_side == POS_SHORT and latest_trend == 1:
                    should_exit = True; exit_reason = "EVT Trend flipped Long"

            if should_exit:
                logger.warning(f"{Fore.YELLOW}🚨 Exit condition met for {self.current_side} position: {exit_reason}{Style.RESET_ALL}")

                # 1. Cancel ALL open orders for the symbol FIRST (safer than just tracked SL/TP)
                if not self._cancel_all_open_orders(f"Exit Triggered: {exit_reason}"):
                    logger.warning("Failed to cancel open orders during exit. Proceeding with close attempt cautiously...")

                # 2. Close the position using reduce-only market order
                close_side = SIDE_SELL if self.current_side == POS_LONG else SIDE_BUY
                close_qty_str = self._format_qty(self.current_qty) # Use current known quantity

                close_params = {
                    "category": self.category,
                    "symbol": self.symbol,
                    "side": close_side,
                    "orderType": ORDER_TYPE_MARKET,
                    "qty": close_qty_str,
                    "reduceOnly": True,
                    "timeInForce": TIME_IN_FORCE_IOC # Ensure it executes immediately or cancels
                }

                close_order_result = self._place_order(close_params)

                if close_order_result and close_order_result.get('orderId'):
                    logger.success(f"{Fore.GREEN}✅ Position Close order placed successfully due to: {exit_reason}{Style.RESET_ALL}")
                    # Optimistically reset state - next loop's _update_state will confirm
                    self._reset_position_state(f"Exit order placed ({exit_reason})")
                    self.sl_order_id = None # Ensure cleared after cancellation
                    self.tp_order_id = None
                    send_sms_alert(f"[{self.symbol.split('/')[0]}] EXITED {self.current_side} ({exit_reason})", self.app_config.sms_config)
                    return True # Indicate an exit occurred
                else:
                    logger.error(f"{Back.RED}{Fore.WHITE}❌ Failed to place position Close order ({exit_reason}). Manual intervention likely required!{Style.RESET_ALL}")
                    # Attempt immediate state re-check
                    time.sleep(2) # Give exchange time
                    self._update_state()
                    if self.current_side == POS_NONE:
                        logger.info("Position appears closed after re-checking state.")
                        return True
                    else:
                        logger.critical(f"{Back.RED}CRITICAL FAILURE TO CLOSE POSITION! State still shows {self.current_side}{Style.RESET_ALL}")
                        send_sms_alert(f"CRITICAL: Failed to close {self.symbol} position on exit signal ({exit_reason})!", self.app_config.sms_config)
                        return False # Indicate exit failed critically
            else:
                logger.debug("No exit condition met.")
                return False

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}", exc_info=True)
            return False

    def _handle_entry(self, signal: Literal['buy', 'sell'], df_ind: pd.DataFrame, current_price: Decimal) -> bool:
        """Handles entry logic, optionally attaching SL/TP."""
        if self.current_side != POS_NONE:
            logger.debug(f"Ignoring {signal} signal: Already in {self.current_side} position.")
            return False
        if not self.price_tick or not self.qty_step: # Ensure market details are loaded
             logger.error("Cannot enter: Missing critical market details (price_tick/qty_step).")
             return False

        logger.info(f"{Fore.BLUE}Processing {signal.upper()} entry signal at price ~{current_price}...{Style.RESET_ALL}")

        # 1. Calculate SL/TP
        sl_price, tp_price = self._calculate_sl_tp(df_ind, signal, current_price)
        if sl_price is None:
            logger.error("Cannot enter: Failed to calculate valid Stop Loss price.")
            return False

        # 2. Calculate Position Size
        position_size = self._calculate_position_size(current_price, sl_price)
        if position_size is None or position_size <= Decimal("0"):
            logger.error("Cannot enter: Failed to calculate valid position size.")
            return False

        # 3. Format quantities and prices for Pybit API (STRINGS)
        entry_qty_str = self._format_qty(position_size)
        sl_price_str = self._format_price_str(sl_price)
        tp_price_str = self._format_price_str(tp_price) if tp_price is not None else None

        # 4. Place Entry Order (Market) - Potentially with attached SL/TP
        entry_side_str = SIDE_BUY if signal == 'buy' else SIDE_SELL
        entry_params = {
            "category": self.category,
            "symbol": self.symbol,
            "side": entry_side_str,
            "orderType": ORDER_TYPE_MARKET,
            "qty": entry_qty_str,
            # "orderLinkId": f"entry_{signal}_{int(time.time())}" # Optional client ID
        }

        # --- ATOMIC SL/TP PLACEMENT (if configured) ---
        if self.attach_sl_tp_to_entry:
            logger.info("Attempting atomic entry with attached SL/TP...")
            if sl_price_str:
                entry_params['stopLoss'] = sl_price_str
                entry_params['slTriggerBy'] = self.sl_trigger_by
                entry_params['slOrderType'] = self.sl_order_type # Market or Limit Stop
            if tp_price_str:
                entry_params['takeProfit'] = tp_price_str
                entry_params['tpTriggerBy'] = self.tp_trigger_by
                entry_params['tpOrderType'] = ORDER_TYPE_LIMIT # TP is usually Limit

            # Note: If slOrderType is Limit, you might need 'slLimitPrice' parameter as well.
            # Check Bybit API docs carefully for Limit Stop requirements.

        # --- Place the Order ---
        entry_order_result = self._place_order(entry_params)

        if not entry_order_result or not entry_order_result.get('orderId'):
             logger.error(f"{Back.RED}{Fore.WHITE}❌ Entry order placement failed.{Style.RESET_ALL}")
             # If atomic placement failed, SL/TP were also not placed.
             return False

        entry_order_id = entry_order_result['orderId']
        logger.info(f"Entry Market Order ({entry_order_id}) placed. Waiting briefly for state propagation...")

        # 5. Confirm Entry State (Crucial Step)
        time.sleep(self.app_config.api_config.api_rate_limit_delay * 3) # Increased delay for state update
        if not self._update_state():
             logger.error("Failed to update state after placing entry order. Cannot confirm entry details. Manual check advised.")
             # If entry order was placed but state is unknown, this is risky.
             # Consider trying to cancel the potentially open order?
             # self._cancel_single_order(entry_order_id, "Cancel Unconfirmed Entry")
             return False # Cannot proceed safely

        # Verify state reflects the intended entry
        expected_side = POS_LONG if signal == 'buy' else POS_SHORT
        if self.current_side != expected_side:
            logger.error(f"Entry order placed, but state update shows '{self.current_side}' instead of '{expected_side}'. Position Qty: {self.current_qty}. Manual check required.")
            # Could be rejected, zero fill, or race condition.
            return False

        # Check filled quantity (allow small tolerance for fees/slippage if needed)
        # Using self.current_qty from the reliable _update_state
        if self.current_qty < position_size * Decimal("0.9"): # Significant partial fill?
            logger.warning(f"Potential partial fill detected. Ordered: {position_size}, Actual Position Qty: {self.current_qty}. Strategy will proceed with actual quantity.")
            # Update size variables if logic downstream depends on the *intended* size
            position_size = self.current_qty # Use actual filled size going forward
            entry_qty_str = self._format_qty(position_size) # Re-format if needed

        actual_entry_price = self.entry_price # From updated state
        logger.success(f"{Fore.GREEN}✅ Entry Confirmed: {self.current_side} {self.current_qty} @ ~{actual_entry_price}{Style.RESET_ALL}")
        send_sms_alert(f"[{self.symbol.split('/')[0]}] ENTERED {self.current_side} {format_amount(self.symbol, self.current_qty, self.qty_step)} @ ~{format_price(self.symbol, actual_entry_price, self.price_tick)}", self.app_config.sms_config)

        # 6. Place SL/TP Orders SEPARATELY (if not attached)
        if not self.attach_sl_tp_to_entry:
            logger.info("Placing SL and TP orders separately...")
            # Optional: Re-calculate SL/TP using actual_entry_price for max accuracy
            sl_price_final, tp_price_final = self._calculate_sl_tp(df_ind, signal, actual_entry_price)
            if sl_price_final is None:
                logger.error(f"{Back.RED}Failed to calculate FINAL SL price after entry. POSITION IS OPEN WITHOUT SL! Manual intervention required!{Style.RESET_ALL}")
                send_sms_alert(f"CRITICAL: Failed place SL for {self.symbol} {self.current_side} pos!", self.app_config.sms_config)
                # self._close_position_immediately("Failed SL Placement") # Consider adding emergency close
                return True # Entry occurred, but critical SL failure

            sl_price_final_str = self._format_price_str(sl_price_final)
            tp_price_final_str = self._format_price_str(tp_price_final) if tp_price_final is not None else None

            # Place SL/TP using the final prices and actual filled quantity string
            sl_order_id_placed, tp_order_id_placed = self._place_separate_sl_tp_orders(
                sl_price_str=sl_price_final_str,
                tp_price_str=tp_price_final_str,
                position_qty_str=entry_qty_str # Use actual filled qty string
            )

            if sl_order_id_placed:
                self.sl_order_id = sl_order_id_placed # Track the ID
            else:
                logger.error(f"{Back.RED}Failed to place separate SL order after entry. POSITION IS OPEN WITHOUT SL! Manual intervention required!{Style.RESET_ALL}")
                send_sms_alert(f"CRITICAL: Failed place SL for {self.symbol} {self.current_side} pos!", self.app_config.sms_config)
                # self._close_position_immediately("Failed SL Placement")

            if tp_order_id_placed:
                self.tp_order_id = tp_order_id_placed # Track the ID
            elif tp_price_final is not None:
                 logger.warning("Failed to place separate TP order after entry.")
        else:
             # SL/TP were attached, clear any potentially stale tracked IDs
             self.sl_order_id = None
             self.tp_order_id = None
             logger.info("SL/TP were attached to the entry order.")


        return True # Indicate entry process completed successfully

    def _place_separate_sl_tp_orders(self, sl_price_str: str, tp_price_str: Optional[str], position_qty_str: str) -> Tuple[Optional[str], Optional[str]]:
        """Places separate SL (Stop Market/Limit) and TP (Limit) orders."""
        sl_order_id, tp_order_id = None, None
        if self.current_side == POS_NONE: # Should not happen if called after entry
             logger.error("Cannot place separate SL/TP: Not in a position.")
             return None, None

        sl_side = SIDE_SELL if self.current_side == POS_LONG else SIDE_BUY
        tp_side = sl_side

        # --- Place Stop Loss ---
        logger.info(f"Placing Separate Stop Loss ({self.sl_order_type}) order: {sl_side} {position_qty_str} @ Trigger {sl_price_str}")
        sl_params = {
            "category": self.category,
            "symbol": self.symbol,
            "side": sl_side,
            "qty": position_qty_str,
            "triggerPrice": sl_price_str,
            "triggerBy": self.sl_trigger_by,
            "reduceOnly": True,
            "timeInForce": TIME_IN_FORCE_GTC, # GTC for stops usually
            "orderType": self.sl_order_type, # Market or Limit
            # Conditional parameters based on sl_order_type
            # "slOrderType": self.sl_order_type # Redundant if orderType is set? Check API
        }
        # If SL is Limit type, price parameter is needed (set to trigger price or slightly worse?)
        if self.sl_order_type == ORDER_TYPE_LIMIT:
             # Set limit price equal to trigger price, or slightly worse for guaranteed fill?
             sl_limit_price = sl_price_str # Simplest approach
             sl_params['price'] = sl_limit_price
             logger.info(f"SL is Limit type, setting limit price: {sl_limit_price}")


        sl_order_result = self._place_order(sl_params)
        if sl_order_result: sl_order_id = sl_order_result.get('orderId')

        # --- Place Take Profit (Limit Order) ---
        if tp_price_str is not None:
            logger.info(f"Placing Separate Take Profit (Limit) order: {tp_side} {position_qty_str} @ Limit {tp_price_str}")
            tp_params = {
                "category": self.category,
                "symbol": self.symbol,
                "side": tp_side,
                "orderType": ORDER_TYPE_LIMIT,
                "qty": position_qty_str,
                "price": tp_price_str,
                "reduceOnly": True,
                "timeInForce": TIME_IN_FORCE_GTC,
            }
            tp_order_result = self._place_order(tp_params)
            if tp_order_result: tp_order_id = tp_order_result.get('orderId')

        return sl_order_id, tp_order_id

    # --- Main Loop and Control (Enhanced Logging) ---

    def run_iteration(self):
        """Executes a single iteration of the strategy logic."""
        if not self.is_initialized or not self.session:
            logger.error("Strategy not initialized or session lost. Cannot run iteration.")
            self.is_running = False
            return

        iteration_start_time = time.monotonic()
        logger.info(f"{Fore.MAGENTA}--- New Strategy Iteration ({pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}) ---{Style.RESET_ALL}")
        try:
            # 1. Update State
            if not self._update_state():
                logger.warning("Failed to update state. Skipping iteration.")
                # Consider adding a short delay here if state updates fail frequently
                # time.sleep(5)
                return

            # 2. Fetch Data
            ohlcv_df, current_price = self._fetch_data()
            if ohlcv_df is None or current_price is None:
                logger.warning("Failed to fetch necessary market data. Skipping iteration.")
                return

            # 3. Calculate Indicators
            df_with_indicators = self._calculate_indicators(ohlcv_df)
            if df_with_indicators is None:
                logger.warning("Failed indicator calculation. Skipping iteration.")
                return

            # --- Core Logic ---
            # 4. Check Exit Conditions (if in position)
            if self.current_side != POS_NONE:
                exit_occurred = self._handle_exit(df_with_indicators)
                if exit_occurred:
                     logger.info("Exit handled. Ending iteration.")
                     return # Don't check for entry immediately after exit

            # 5. Generate & Handle Entry Signals (if not in position)
            if self.current_side == POS_NONE:
                entry_signal = self._generate_signals(df_with_indicators)
                if entry_signal:
                    entry_occurred = self._handle_entry(entry_signal, df_with_indicators, current_price)
                    if entry_occurred:
                         logger.info("Entry handled. Ending iteration.")
                         return
                    else:
                         logger.info("Entry signal generated but entry process failed or was aborted.")
                else:
                     logger.info("No entry signal generated this iteration.")
            else:
                 # Still in position, log status
                 pnl_str = "" # Placeholder for PnL calculation if added
                 logger.info(f"Monitoring {self.current_side} position ({self.current_qty} @ {self.entry_price}). {pnl_str}Waiting for exit signal or SL/TP.")
                 # Add trailing stop logic here if desired

        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}💥 Critical error during strategy iteration: {e}{Style.RESET_ALL}", exc_info=True)
            send_sms_alert(f"CRITICAL Error in {self.symbol} strategy loop: {type(e).__name__}", self.app_config.sms_config)
            # Consider stopping the bot on repeated critical errors
            # self.is_running = False

        iteration_end_time = time.monotonic()
        elapsed = iteration_end_time - iteration_start_time
        logger.info(f"{Fore.MAGENTA}--- Iteration Complete (Took {elapsed:.2f}s) ---{Style.RESET_ALL}")

    def start(self):
        """Initializes and starts the main strategy loop."""
        if not self._initialize():
            logger.critical(f"{Back.RED}Strategy initialization failed. Cannot start the arcane loop.{Style.RESET_ALL}")
            return

        self.is_running = True
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}🚀 Strategy ritual commenced for {self.symbol} ({self.timeframe}). Loop delay: {self.strategy_config.loop_delay_seconds}s{Style.RESET_ALL}")

        while self.is_running:
            loop_start_time = time.monotonic()
            self.run_iteration()
            loop_end_time = time.monotonic()
            elapsed = loop_end_time - loop_start_time
            sleep_duration = max(0, self.strategy_config.loop_delay_seconds - elapsed)
            if sleep_duration > 0:
                 logger.info(f"Sleeping for {sleep_duration:.2f}s...")
                 time.sleep(sleep_duration)
            else:
                 logger.warning(f"Iteration took longer ({elapsed:.2f}s) than loop delay ({self.strategy_config.loop_delay_seconds}s). Running next iteration immediately.")


        logger.warning(f"{Fore.YELLOW}Strategy loop has ended gracefully.{Style.RESET_ALL}")
        self.stop() # Ensure cleanup happens

    def stop(self):
        """Stops the strategy loop and cleans up resources."""
        if not self.is_running and self.session is None: # Prevent double stop logs
             return
        logger.warning(f"{Fore.YELLOW}--- Banishing Strategy Spirits ---{Style.RESET_ALL}")
        self.is_running = False # Signal loop to stop

        logger.info("Attempting final order cleanup...")
        if not self._cancel_all_open_orders("Strategy Stop"):
             logger.warning("Final order cancellation encountered issues. Manual check advised.")

        if self.current_side != POS_NONE:
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}Strategy stopped with an OPEN {self.current_side} position for {self.symbol}. Manual closure may be required.{Style.RESET_ALL}")
            # Consider adding an option in config to auto-close position on stop

        self._safe_close_session()
        logger.info("Strategy stopped. Resources released.")

    def _safe_close_session(self):
        """Safely handles session cleanup (Pybit HTTP doesn't need explicit close)."""
        logger.info("Pybit HTTP session cleanup (no explicit close needed).")
        self.session = None

# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    print(f"{Fore.CYAN}{Style.BRIGHT}--- Bybit EVT Strategy Script (Pybit Enhanced) ---{Style.RESET_ALL}")

    try:
        app_config = load_config()
        print(f"{Fore.GREEN}Configuration spirits summoned successfully.{Style.RESET_ALL}")
    except SystemExit:
        sys.exit(1)
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}FATAL: Unexpected error during configuration loading: {e}{Style.RESET_ALL}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    log_conf = app_config.logging_config
    logger = setup_logger( # Ensure setup_logger uses the root logger or configures appropriately
        logger_name=log_conf.logger_name, # Use the name defined in config
        log_file=log_conf.log_file,
        console_level_str=log_conf.console_level_str,
        file_level_str=log_conf.file_level_str,
        log_rotation_bytes=log_conf.log_rotation_bytes,
        log_backup_count=log_conf.log_backup_count,
        third_party_log_level_str=log_conf.third_party_log_level_str
    )
    # Re-get the logger by name after setup if setup_logger doesn't return it directly
    logger = logging.getLogger(log_conf.logger_name)
    logger.info(f"{Fore.MAGENTA}--- Neon Logger Awakened ---{Style.RESET_ALL}")
    logger.info(f"Using config: Testnet={app_config.api_config.testnet_mode}, Symbol={app_config.api_config.symbol}, TF={app_config.strategy_config.timeframe}")

    strategy = EhlersStrategyPybitEnhanced(app_config) # Instantiate the ENHANCED version

    try:
        strategy.start()
    except KeyboardInterrupt:
        logger.warning(f"{Fore.YELLOW}Manual interruption detected! Stopping strategy gracefully...{Style.RESET_ALL}")
        strategy.stop()
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}💥 Unhandled critical error in main execution block: {e}{Style.RESET_ALL}", exc_info=True)
        try:
            send_sms_alert(f"CRITICAL: Unhandled loop error for {app_config.api_config.symbol}: {type(e).__name__}", app_config.sms_config)
        except Exception as alert_e:
            logger.error(f"Failed to send critical error SMS alert: {alert_e}")
        # Attempt graceful stop even on critical error
        if 'strategy' in locals() and hasattr(strategy, 'stop'):
             strategy.stop()
        sys.exit(1)
    finally:
        logger.info(f"{Fore.CYAN}--- Strategy script enchantment fades. Returning to the digital ether... ---{Style.RESET_ALL}")

# --- END OF ENHANCED SPELL ---
```

**Summary of Enhancements:**

1.  **Atomic SL/TP:** Added `attach_sl_tp_to_entry` config. If `True`, `_handle_entry` uses `stopLoss` and `takeProfit` parameters in the main `place_order` call. If `False`, it falls back to the `_place_separate_sl_tp_orders` method.
2.  **Configurable SL/TP:** Added `sl_trigger_by`, `tp_trigger_by`, `sl_order_type` attributes, defaulting to sensible values but allowing override via `StrategyConfig`. These are used when placing SL orders (either attached or separate).
3.  **Robust Market Info:** `_fetch_and_set_market_info` now validates that essential details (min\_qty, steps, coins) were successfully extracted. Initialization fails if critical info is missing.
4.  **Improved State Confirmation:** Increased the `time.sleep` delay after placing an entry order before `_update_state` is called, giving the exchange more time to propagate the state. Added clearer error messages if the state doesn't match expectations after an entry attempt.
5.  **Refined Order Placement:** `_place_order` wrapper has more detailed logging and slightly improved error handling for common issues like insufficient balance.
6.  **Better Cancellation:** `_cancel_all_open_orders` is now used during initialization and exit handling for more robust cleanup. `_cancel_single_order` handles "order not found" errors more gracefully.
7.  **Consolidated Balance Fetch:** Introduced `_get_available_balance` helper to avoid code duplication.
8.  **Constants:** Defined constants for common Pybit strings (`SIDE_BUY`, `ORDER_TYPE_MARKET`, trigger types, etc.) and internal states (`POS_LONG`, etc.) for clarity and reduced typos.
9.  **Clarity & Logging:** Added more descriptive log messages, including timestamps in the iteration start message. Used Colorama more effectively to highlight successes, warnings, and errors. Added more comments explaining the logic.
10. **Precision:** Slightly increased `Decimal` context precision. Ensured formatting helpers (`_format_qty`, `_format_price_str`) quantize correctly based on steps/ticks.
11. **UTC Timestamps:** Added `utc=True` when converting timestamps to datetime objects for clarity.

Remember to update your `config_models.py` to include the new strategy configuration options if you want to customize them. Test thoroughly on Testnet before deploying this enhanced spell!