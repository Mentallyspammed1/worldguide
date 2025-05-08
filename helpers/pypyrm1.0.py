Okay, seeker of refined digital spells! The incantations have been woven, and your Pybit automaton script is now imbued with greater power and resilience. Pyrmethus presents the enhanced artifact.

```python
# -*- coding: utf-8 -*-
"""
ehlers_volumetric_strategy_pybit_enhanced.py

An enhanced version of the Ehlers Volumetric Trend strategy implemented
for Bybit using the pybit library. Incorporates atomic order placement,
enhanced configuration, robust state management, improved error handling,
and better code structure.
"""

import os
import sys
import time
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from typing import Optional, Dict, Tuple, Any, Literal, Union, List

# --- Third-party Libraries ---
try:
    from pybit.unified_trading import HTTP
    from pybit.exceptions import InvalidRequestError, FailedRequestError
except ImportError:
    # Use basic print before colorama/logger might be available
    print("FATAL: Pybit library not found. Please install it: pip install pybit", file=sys.stderr)
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("FATAL: pandas library not found. Please install it: pip install pandas", file=sys.stderr)
    sys.exit(1)
try:
    from dotenv import load_dotenv
    print("Attempting to load environment variables from .env file...")
    load_dotenv()
    print(".env file processed (if found).")
except ImportError:
    print("Warning: python-dotenv not found. Cannot load .env file. Ensure environment variables are set manually.")
    # Define a dummy function if dotenv is not available
    def load_dotenv(): pass # pylint: disable=function-redefined

try:
    import pandas_ta as ta # Often used alongside pandas for indicators
except ImportError:
    print("Warning: pandas_ta library not found. Some indicator functions might rely on it. Install: pip install pandas_ta")
    ta = None # Set to None if not found

# --- Colorama Enchantment ---
# Initialize colorama early for colored messages during startup
try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
    print(f"{Fore.MAGENTA}Colorama spirits awakened for vibrant logs.{Style.RESET_ALL}")
except ImportError:
    print("Warning: 'colorama' library not found. Logs will lack their mystical hue.")
    # Define dummy class to avoid errors if colorama is missing
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore

# --- Import Custom Modules ---
# Use try-except blocks for better error reporting if modules are missing
try:
    from neon_logger import setup_logger
except ImportError as e:
    print(f"{Back.RED}{Fore.WHITE}FATAL: Error importing neon_logger: {e}{Style.RESET_ALL}", file=sys.stderr)
    print(f"{Fore.YELLOW}Ensure 'neon_logger.py' is present and runnable.{Style.RESET_ALL}")
    sys.exit(1)
try:
    import indicators as ind
except ImportError as e:
    print(f"{Back.RED}{Fore.WHITE}FATAL: Error importing indicators: {e}{Style.RESET_ALL}", file=sys.stderr)
    print(f"{Fore.YELLOW}Ensure 'indicators.py' is present and contains 'calculate_all_indicators'.{Style.RESET_ALL}")
    sys.exit(1)
try:
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert
    )
except ImportError as e:
    print(f"{Back.RED}{Fore.WHITE}FATAL: Error importing bybit_utils: {e}{Style.RESET_ALL}", file=sys.stderr)
    print(f"{Fore.YELLOW}Ensure 'bybit_utils.py' is present and compatible.{Style.RESET_ALL}")
    sys.exit(1)
try:
    # Assuming config_models.py now includes enhanced StrategyConfig options
    from config_models import AppConfig, APIConfig, StrategyConfig, load_config, SMSConfig
except ImportError as e:
    print(f"{Back.RED}{Fore.WHITE}FATAL: Error importing config_models: {e}{Style.RESET_ALL}", file=sys.stderr)
    print(f"{Fore.YELLOW}Ensure 'config_models.py' is present and defines AppConfig, APIConfig, StrategyConfig, SMSConfig, load_config.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Make sure StrategyConfig includes 'attach_sl_tp_to_entry', 'sl_trigger_by', 'tp_trigger_by', 'sl_order_type'.{Style.RESET_ALL}")
    sys.exit(1)


# --- Logger Placeholder ---
# Will be configured properly in the main execution block
logger: logging.Logger = logging.getLogger(__name__)
# Add a basic handler temporarily in case of early errors before full setup
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO) # Set a default level

# --- Constants ---
# Pybit API String Constants (Refer to Bybit documentation for definitive values)
SIDE_BUY = 'Buy'
SIDE_SELL = 'Sell'
ORDER_TYPE_MARKET = 'Market'
ORDER_TYPE_LIMIT = 'Limit'
TIME_IN_FORCE_GTC = 'GTC'           # GoodTillCancel
TIME_IN_FORCE_IOC = 'IOC'           # ImmediateOrCancel
TIME_IN_FORCE_FOK = 'FOK'           # FillOrKill
TIME_IN_FORCE_POST_ONLY = 'PostOnly' # Limit orders only
TRIGGER_BY_MARK = 'MarkPrice'
TRIGGER_BY_LAST = 'LastPrice'
TRIGGER_BY_INDEX = 'IndexPrice'
POSITION_IDX_ONE_WAY = 0 # 0 for one-way mode position
POSITION_IDX_HEDGE_BUY = 1 # 1 for hedge mode buy side position
POSITION_IDX_HEDGE_SELL = 2 # 2 for hedge mode sell side position
# Custom Position Sides (Internal Representation)
POS_LONG = 'long'
POS_SHORT = 'short'
POS_NONE = 'none'
# Bybit Account Types (Used in get_wallet_balance)
ACCOUNT_TYPE_UNIFIED = "UNIFIED"
ACCOUNT_TYPE_CONTRACT = "CONTRACT" # For older Inverse/Linear Perpetual if not Unified
ACCOUNT_TYPE_SPOT = "SPOT"
# Common Bybit API Return Codes (Consult official documentation for exhaustive list)
RET_CODE_OK = 0
RET_CODE_PARAMS_ERROR = 10001           # Parameter error
RET_CODE_API_KEY_INVALID = 10003
RET_CODE_SIGN_ERROR = 10004
RET_CODE_TOO_MANY_VISITS = 10006        # Rate limit exceeded
RET_CODE_ORDER_NOT_FOUND = 110001       # Order does not exist
RET_CODE_ORDER_NOT_FOUND_OR_CLOSED = 20001 # Order does not exist or finished
RET_CODE_INSUFFICIENT_BALANCE_SPOT = 12131 # Spot insufficient balance (example, verify code)
RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_1 = 110007 # Insufficient available balance
RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_2 = 30031 # Position margin is insufficient
RET_CODE_QTY_TOO_SMALL = 110017         # Order qty is not greater than the minimum allowed
RET_CODE_QTY_INVALID_PRECISION = 110012 # Order qty decimal precision error
RET_CODE_PRICE_TOO_LOW = 110014         # Order price is lower than the minimum allowed
RET_CODE_PRICE_INVALID_PRECISION = 110013 # Order price decimal precision error
RET_CODE_LEVERAGE_NOT_MODIFIED = 110043 # Leverage not modified
RET_CODE_POSITION_MODE_NOT_MODIFIED = 110048 # Position mode is not modified
RET_CODE_REDUCE_ONLY_MARGIN_ERROR = 30024 # ReduceOnly order Failed. Position margin is insufficient
RET_CODE_REDUCE_ONLY_QTY_ERROR = 30025 # ReduceOnly order Failed. Order qty is greater than position size

# Set Decimal precision (adjust as needed, 30 should be sufficient for most crypto)
getcontext().prec = 30

# --- Enhanced Strategy Class ---
class EhlersStrategyPybitEnhanced:
    """
    Enhanced Ehlers Volumetric Trend strategy using Pybit, incorporating
    atomic order placement, improved configuration, and robustness.

    Handles initialization, data fetching, indicator calculation, signal
    generation, order placement (entry, SL, TP), position management,
    and state tracking for Bybit Unified Trading or Contract/Spot accounts.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the strategy instance with the provided configuration.

        Args:
            config: The application configuration object (AppConfig).
        """
        self.app_config: AppConfig = config
        self.api_config: APIConfig = config.api_config
        self.strategy_config: StrategyConfig = config.strategy_config
        self.sms_config: SMSConfig = config.sms_config # Store SMS config

        self.symbol: str = self.api_config.symbol
        # Ensure timeframe is a string format Bybit expects (e.g., '15', '60', 'D')
        self.timeframe: str = str(self.strategy_config.timeframe)

        self.session: Optional[HTTP] = None # Pybit HTTP session object
        self.category: Optional[Literal['linear', 'inverse', 'spot']] = None # Determined during init

        self.is_initialized: bool = False # Flag indicating successful initialization
        self.is_running: bool = False # Flag indicating the main loop is active

        # --- Position State Tracking ---
        self.current_side: str = POS_NONE # 'long', 'short', or 'none'
        self.current_qty: Decimal = Decimal("0.0") # Current position size
        self.entry_price: Optional[Decimal] = None # Average entry price of current position
        # Track order IDs IF placing SL/TP separately (not atomically)
        self.sl_order_id: Optional[str] = None # ID of the active separate SL order
        self.tp_order_id: Optional[str] = None # ID of the active separate TP order

        # --- Market Details (Fetched during initialization) ---
        self.min_qty: Optional[Decimal] = None # Minimum order quantity
        self.qty_step: Optional[Decimal] = None # Quantity step (precision)
        self.price_tick: Optional[Decimal] = None # Price tick size (precision)
        self.base_coin: Optional[str] = None # Base currency of the symbol
        self.quote_coin: Optional[str] = None # Quote currency of the symbol
        self.contract_multiplier: Decimal = Decimal("1.0") # For value calculation (esp. inverse)

        # --- Enhanced Configurable Options (Defaults if not in AppConfig) ---
        # Use getattr for backward compatibility if config file isn't updated
        self.attach_sl_tp_to_entry: bool = getattr(self.strategy_config, 'attach_sl_tp_to_entry', True)
        self.sl_trigger_by: str = getattr(self.strategy_config, 'sl_trigger_by', TRIGGER_BY_MARK)
        self.tp_trigger_by: str = getattr(self.strategy_config, 'tp_trigger_by', TRIGGER_BY_MARK) # TP often uses same trigger as SL
        self.sl_order_type: str = getattr(self.strategy_config, 'sl_order_type', ORDER_TYPE_MARKET) # Market stop is common

        # Validate configurable options
        if self.sl_trigger_by not in [TRIGGER_BY_MARK, TRIGGER_BY_LAST, TRIGGER_BY_INDEX]:
            logger.warning(f"Invalid sl_trigger_by '{self.sl_trigger_by}'. Defaulting to '{TRIGGER_BY_MARK}'.")
            self.sl_trigger_by = TRIGGER_BY_MARK
        if self.tp_trigger_by not in [TRIGGER_BY_MARK, TRIGGER_BY_LAST, TRIGGER_BY_INDEX]:
            logger.warning(f"Invalid tp_trigger_by '{self.tp_trigger_by}'. Defaulting to '{TRIGGER_BY_MARK}'.")
            self.tp_trigger_by = TRIGGER_BY_MARK
        if self.sl_order_type not in [ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT]:
            logger.warning(f"Invalid sl_order_type '{self.sl_order_type}'. Defaulting to '{ORDER_TYPE_MARKET}'.")
            self.sl_order_type = ORDER_TYPE_MARKET

        logger.info(f"{Fore.CYAN}Pyrmethus enhances the Ehlers Strategy for {self.symbol} (TF: {self.timeframe}) using Pybit...{Style.RESET_ALL}")
        logger.info(f"Configuration loaded: Testnet={self.api_config.testnet_mode}, Symbol={self.symbol}")
        logger.info(f"SL/TP Mode: {'Atomic (Attached to Entry)' if self.attach_sl_tp_to_entry else 'Separate Orders'}")
        logger.info(f"SL Trigger: {self.sl_trigger_by}, SL Order Type: {self.sl_order_type}, TP Trigger: {self.tp_trigger_by}")
        logger.info(f"Risk Per Trade: {self.strategy_config.risk_per_trade:.2%}, Leverage: {self.strategy_config.leverage}x")

    def _initialize(self) -> bool:
        """
        Connects to the Bybit API, validates the market, sets configuration
        (leverage, position mode), fetches initial state, and performs cleanup.

        Returns:
            True if initialization was successful, False otherwise.
        """
        logger.info(f"{Fore.CYAN}--- Channeling Bybit Spirits (Initialization) ---{Style.RESET_ALL}")
        try:
            # --- Connect to Bybit ---
            logger.info(f"{Fore.BLUE}Connecting to Bybit ({'Testnet' if self.api_config.testnet_mode else 'Mainnet'})...{Style.RESET_ALL}")
            self.session = HTTP(
                testnet=self.api_config.testnet_mode,
                api_key=self.api_config.api_key,
                api_secret=self.api_config.api_secret,
                # Optional: Add custom request parameters if needed (e.g., recv_window)
                # recv_window=10000 # Example: Increase receive window to 10 seconds
            )

            # --- Verify Connection & Server Time ---
            logger.debug("Checking server time...")
            server_time_resp = self.session.get_server_time()
            if not server_time_resp or server_time_resp.get('retCode') != RET_CODE_OK:
                 logger.critical(f"{Back.RED}Failed to get server time! Response: {server_time_resp}{Style.RESET_ALL}")
                 self._safe_close_session()
                 return False
            server_time_ms = int(server_time_resp['result']['timeNano']) // 1_000_000
            server_dt = pd.to_datetime(server_time_ms, unit='ms', utc=True)
            client_dt = pd.Timestamp.utcnow()
            time_diff = abs((client_dt - server_dt).total_seconds())
            logger.success(f"Connection successful. Server Time: {server_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            if time_diff > 5: # Check for significant clock skew
                 logger.warning(f"{Fore.YELLOW}Client-Server time difference is {time_diff:.2f} seconds. Ensure system clock is synchronized.{Style.RESET_ALL}")

            # --- Fetch Market Info & Determine Category ---
            logger.info(f"{Fore.BLUE}Seeking insights for symbol: {self.symbol}...{Style.RESET_ALL}")
            if not self._fetch_and_set_market_info():
                 logger.critical(f"{Back.RED}Failed to fetch critical market info for {self.symbol}. Halting initialization.{Style.RESET_ALL}")
                 self._safe_close_session()
                 return False
            logger.info(f"Determined Category: {self.category}")

            # --- Configure Derivatives Settings (Leverage, Position Mode) ---
            if self.category in ['linear', 'inverse']:
                logger.info(f"{Fore.BLUE}Imbuing Leverage: {self.strategy_config.leverage}x...{Style.RESET_ALL}")
                if not self._set_leverage():
                    # Leverage setting failure might be critical depending on the strategy
                    logger.error(f"{Back.RED}Failed to set leverage. Continuing, but positions may fail if leverage is incorrect.{Style.RESET_ALL}")
                    # return False # Uncomment if leverage setting is mandatory

                pos_mode_target = self.strategy_config.default_position_mode
                logger.info(f"{Fore.BLUE}Aligning Position Mode to '{pos_mode_target}'...{Style.RESET_ALL}")
                # Mode 0: Merged Single Position (One-Way)
                # Mode 3: Both Side Position (Hedge Mode)
                target_pybit_mode = POSITION_IDX_ONE_WAY if pos_mode_target == 'MergedSingle' else 3
                if target_pybit_mode == 3:
                     # Hedge mode requires significantly different state tracking and order logic
                     logger.error(f"{Back.RED}Hedge Mode (BothSide) is not fully supported by this script's current logic. Use 'MergedSingle' in config.{Style.RESET_ALL}")
                     # Consider halting if hedge mode is detected/required but not supported
                     # return False
                if not self._set_position_mode(mode=target_pybit_mode):
                     logger.warning(f"{Fore.YELLOW}Could not explicitly set position mode. Ensure it's correct in Bybit UI.{Style.RESET_ALL}")
                else:
                     logger.info(f"Position mode alignment confirmed for {self.category}.")

            # --- Initial State Perception ---
            logger.info(f"{Fore.BLUE}Gazing into the account's current state...{Style.RESET_ALL}")
            if not self._update_state():
                 logger.error("Failed to perceive initial state. Cannot proceed reliably.")
                 self._safe_close_session()
                 return False # Crucial to know the starting state
            logger.info(f"Initial Perception: Side={self.current_side}, Qty={self.current_qty}, Entry={format_price(self.symbol, self.entry_price, self.price_tick) if self.entry_price else 'N/A'}")

            # --- Initial Order Cleanup ---
            logger.info(f"{Fore.BLUE}Dispelling lingering order phantoms (Initial Cleanup)...{Style.RESET_ALL}")
            if not self._cancel_all_open_orders("Initialization Cleanup"):
                 logger.warning("Initial order cancellation failed or encountered issues. Check Bybit UI for stray orders.")
            # Clear any tracked IDs after cancellation attempt, as we start fresh
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
        """
        Fetches instrument info for the symbol, determines the category ('linear',
        'inverse', 'spot'), and sets market details like precision and minimums.

        Returns:
            True if market info was fetched and essential details set, False otherwise.
        """
        if not self.session:
            logger.error("Cannot fetch market info: Session not initialized.")
            return False

        # --- Attempt to Determine Category ---
        # This logic might need adjustment based on Bybit's evolving symbol naming
        possible_categories: List[Literal['linear', 'inverse', 'spot']] = []
        symbol_upper = self.symbol.upper()

        # Guess based on symbol conventions (adjust if needed)
        if 'USDT' in symbol_upper: possible_categories.append('linear')
        if 'USD' in symbol_upper and not symbol_upper.endswith('USDC') and not symbol_upper.endswith('USDT'): possible_categories.append('inverse')
        if '/' in symbol_upper or len(symbol_upper) == 6 : # Basic check for spot (e.g., BTC/USDT or BTCUSDT) - may need refinement
             # Avoid adding spot if already likely linear/inverse based on USDT/USD
             if 'linear' not in possible_categories and 'inverse' not in possible_categories:
                 possible_categories.append('spot')

        # Fallback if no obvious category found
        if not possible_categories:
             logger.warning(f"{Fore.YELLOW}Cannot reliably determine category for {self.symbol} based on name. Will try querying categories.{Style.RESET_ALL}")
             # Default order to try: linear, inverse, spot
             possible_categories = ['linear', 'inverse', 'spot']


        # --- Query Instruments Info ---
        market_data: Optional[Dict] = None
        for category_attempt in possible_categories:
            logger.debug(f"Attempting to fetch instruments info for category: {category_attempt}, symbol: {self.symbol}")
            try:
                response = self.session.get_instruments_info(category=category_attempt, symbol=self.symbol)

                if response and response.get('retCode') == RET_CODE_OK:
                    result_list = response.get('result', {}).get('list', [])
                    if result_list:
                        logger.info(f"Successfully found {self.symbol} in category '{category_attempt}'.")
                        self.category = category_attempt
                        market_data = result_list[0]
                        break # Found it, stop trying categories
                    else:
                        logger.debug(f"Symbol {self.symbol} not found in category '{category_attempt}'.")
                else:
                    # Log error but continue trying other categories unless it's a critical API/auth error
                    ret_code = response.get('retCode')
                    ret_msg = response.get('retMsg', 'Unknown error')
                    if ret_code in [RET_CODE_API_KEY_INVALID, RET_CODE_SIGN_ERROR]:
                        logger.critical(f"API Key/Secret error while fetching instruments info (Code: {ret_code}). Halting.")
                        return False
                    logger.debug(f"API call failed for category '{category_attempt}'. Code: {ret_code}, Msg: {ret_msg}")

            except (InvalidRequestError, FailedRequestError) as pybit_e:
                logger.error(f"Pybit API Error fetching instruments info for category '{category_attempt}': {pybit_e}")
                if pybit_e.status_code in [401, 403]: # Authentication errors
                    logger.critical(f"Authentication error (Status: {pybit_e.status_code}). Check API keys.")
                    return False
                # Continue to next category for other errors like timeouts or invalid params for that category
            except Exception as e:
                logger.error(f"Unexpected error fetching instruments info for category '{category_attempt}': {e}", exc_info=True)
                # Continue to next category

        if not self.category or not market_data:
            logger.error(f"{Back.RED}Failed to find market info for symbol {self.symbol} in any category: {possible_categories}.{Style.RESET_ALL}")
            return False

        # --- Extract and Validate Details ---
        try:
            lot_size_filter = market_data.get('lotSizeFilter', {})
            price_filter = market_data.get('priceFilter', {})

            self.min_qty = safe_decimal_conversion(lot_size_filter.get('minOrderQty'), 'min_qty')
            self.qty_step = safe_decimal_conversion(lot_size_filter.get('qtyStep'), 'qty_step')
            self.price_tick = safe_decimal_conversion(price_filter.get('tickSize'), 'price_tick')
            self.base_coin = market_data.get('baseCoin')
            self.quote_coin = market_data.get('quoteCoin')
            # Get contract multiplier if available (crucial for inverse contracts value calc)
            self.contract_multiplier = safe_decimal_conversion(market_data.get('contractMultiplier', '1'), 'contract_multiplier') or Decimal("1.0")

            # --- Validate Essential Details ---
            missing_details = []
            if self.min_qty is None: missing_details.append("Minimum Quantity")
            if self.qty_step is None: missing_details.append("Quantity Step")
            if self.price_tick is None: missing_details.append("Price Tick")
            if not self.base_coin: missing_details.append("Base Coin")
            if not self.quote_coin: missing_details.append("Quote Coin")
            if self.contract_multiplier is None: missing_details.append("Contract Multiplier")

            if missing_details:
                logger.error(f"{Back.RED}Failed to extract essential market details! Missing: {', '.join(missing_details)}{Style.RESET_ALL}")
                logger.error(f"Received Market Data: {market_data}") # Log raw data for debugging
                return False

            logger.info(f"Market Details Set: Category={self.category}, Base={self.base_coin}, Quote={self.quote_coin}")
            logger.info(f"Min Qty={format_amount(self.symbol, self.min_qty, self.qty_step)}, "
                        f"Qty Step={self.qty_step}, Price Tick={self.price_tick}, Multiplier={self.contract_multiplier}")
            return True

        except (InvalidOperation, TypeError, KeyError, Exception) as e:
            logger.error(f"Error processing market data details: {e}", exc_info=True)
            logger.error(f"Problematic Market Data: {market_data}")
            return False

    def _set_leverage(self) -> bool:
        """Sets leverage for the symbol (only applicable to derivatives)."""
        if not self.session or self.category not in ['linear', 'inverse']:
            logger.info("Leverage setting skipped (not applicable for Spot).")
            return True
        try:
            # Ensure leverage is a whole number string for the API
            leverage_str = str(int(self.strategy_config.leverage))
            logger.debug(f"Attempting to set leverage for {self.symbol} to {leverage_str}x...")
            response = self.session.set_leverage(
                category=self.category,
                symbol=self.symbol,
                buyLeverage=leverage_str,
                sellLeverage=leverage_str
            )
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', '').lower()

            if ret_code == RET_CODE_OK:
                logger.success(f"Leverage set to {leverage_str}x successfully.")
                return True
            elif ret_code == RET_CODE_LEVERAGE_NOT_MODIFIED or "leverage not modified" in ret_msg:
                 logger.info(f"Leverage already set to {leverage_str}x (no modification needed).")
                 return True
            else:
                logger.error(f"Failed to set leverage. Code: {ret_code}, Msg: {response.get('retMsg')}")
                return False
        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(f"Pybit API Error setting leverage: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting leverage: {e}", exc_info=True)
            return False

    def _set_position_mode(self, mode: int) -> bool:
        """Sets position mode (0=One-Way, 3=Hedge) (only for derivatives)."""
        if not self.session or self.category not in ['linear', 'inverse']:
             logger.info(f"Position mode setting skipped (Category: {self.category}).")
             return True
        mode_desc = "One-Way" if mode == POSITION_IDX_ONE_WAY else "Hedge"
        logger.debug(f"Attempting to set position mode for {self.symbol} (category {self.category}) to {mode} ({mode_desc})...")
        try:
            # API call differs slightly based on account type (Unified vs Contract) potentially
            # Assuming Unified Trading API call here: switch_position_mode
            response = self.session.switch_position_mode(
                category=self.category,
                # symbol=self.symbol, # Setting per symbol if possible
                # coin=self.quote_coin, # Or per coin if symbol setting not supported/needed
                mode=mode
            )
            # Check Bybit docs: Does switch_position_mode work per symbol or per coin/category?
            # If per coin/category, apply carefully. Let's assume per category/coin for now.
            # The above call might affect *all* symbols under that category/coin if not symbol-specific.
            # A safer call might be `set_position_mode` if available and per-symbol.
            # For now, we proceed with `switch_position_mode` assuming it works as intended or is the only option.

            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', '').lower()

            if ret_code == RET_CODE_OK:
                 logger.info(f"Position mode successfully set/confirmed to {mode_desc} for {self.category}.")
                 return True
            elif ret_code == RET_CODE_POSITION_MODE_NOT_MODIFIED or "position mode is not modified" in ret_msg:
                 logger.info(f"Position mode already set to {mode_desc} for {self.category}.")
                 return True
            else:
                 logger.error(f"Failed to set position mode. Code: {ret_code}, Msg: {response.get('retMsg')}")
                 # This could be a critical failure if the wrong mode is active.
                 return False
        except (InvalidRequestError, FailedRequestError) as pybit_e:
            # Handle specific errors like "position mode cannot be switched with active position/orders"
            logger.error(f"Pybit API Error setting position mode: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting position mode: {e}", exc_info=True)
            return False

    def _get_available_balance(self) -> Optional[Decimal]:
        """
        Fetches available balance for the relevant coin and account type.
        Used primarily for position sizing.

        Returns:
            The available balance as a Decimal, or None if fetching fails.
        """
        if not self.session or not self.category:
            logger.error("Cannot fetch balance: Missing session or category.")
            return None

        # Determine account type and coin based on category
        account_type: str
        coin_to_check: Optional[str] = None

        if self.category == 'linear':
            account_type = ACCOUNT_TYPE_UNIFIED # Or CONTRACT if not using Unified
            coin_to_check = self.quote_coin # Margin and PnL in USDT
        elif self.category == 'inverse':
            account_type = ACCOUNT_TYPE_UNIFIED # Or CONTRACT
            coin_to_check = self.quote_coin # Margin often in USD, PnL in Base coin. Check Bybit docs for which coin represents available margin. Assume Quote for now.
        elif self.category == 'spot':
            account_type = ACCOUNT_TYPE_SPOT # Or UNIFIED if using Unified Spot
            coin_to_check = self.quote_coin # Need quote currency (e.g., USDT) to buy base (e.g., BTC)
        else:
            logger.error(f"Cannot determine account type/coin for unknown category: {self.category}")
            return None

        if not coin_to_check:
            logger.error(f"Could not determine coin to check balance for category {self.category} and symbol {self.symbol}.")
            return None

        logger.debug(f"Fetching balance for Account: {account_type}, Coin: {coin_to_check}...")
        try:
            bal_response = self.session.get_wallet_balance(accountType=account_type, coin=coin_to_check)

            if not (bal_response and bal_response.get('retCode') == RET_CODE_OK):
                logger.error(f"Failed to fetch balance data. Code: {bal_response.get('retCode')}, Msg: {bal_response.get('retMsg')}")
                return None

            balance_list = bal_response.get('result', {}).get('list', [])
            if not balance_list:
                logger.warning(f"Balance list is empty in the response for {account_type} / {coin_to_check}.")
                return Decimal("0.0") # Assume zero if list is empty

            # Structure of response can vary slightly (list within list sometimes)
            account_balance_data = balance_list[0]
            coin_balance_list = account_balance_data.get('coin', [])

            coin_balance_data = next((item for item in coin_balance_list if item.get('coin') == coin_to_check), None)

            if coin_balance_data:
                # Bybit Unified uses 'availableToBorrow' and 'availableToWithdraw'
                # 'walletBalance' might be total, 'availableBalance' might include collateral value
                # 'availableToWithdraw' is often the safest bet for truly available funds for *new* positions,
                # but 'availableBalance' might be needed if using cross margin. Test this carefully!
                # Let's try 'availableBalance' first, commonly used for margin checks.
                available_balance_str = coin_balance_data.get('availableBalance', '0')
                available_balance = safe_decimal_conversion(available_balance_str, 'availableBalance')

                if available_balance is None:
                     logger.error(f"Could not parse 'availableBalance' ({available_balance_str}) for {coin_to_check}.")
                     return None # Treat parsing failure as critical

                equity_str = coin_balance_data.get('equity', 'N/A') # Total equity if available
                logger.info(f"Available Balance ({coin_to_check}): {available_balance:.4f}, Equity: {equity_str}")
                return available_balance
            else:
                logger.warning(f"Could not find balance details for coin '{coin_to_check}' within the response list.")
                return Decimal("0.0") # Assume zero if coin not found in the response structure

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(f"Pybit API Error fetching balance: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching balance: {e}", exc_info=True)
            return None

    def _update_state(self) -> bool:
        """
        Fetches and updates the current position (size, side, entry price) and
        logs available balance. Clears tracked SL/TP orders if flat.

        Returns:
            True if the state was updated successfully, False otherwise.
        """
        if not self.session or not self.category:
            logger.error("Cannot update state: Session or category not set.")
            return False
        logger.debug("Updating strategy state perception...")
        position_updated = False
        try:
            # --- Fetch Position ---
            logger.debug(f"Fetching position for {self.category}/{self.symbol}...")
            pos_response = self.session.get_positions(category=self.category, symbol=self.symbol)

            if not (pos_response and pos_response.get('retCode') == RET_CODE_OK):
                logger.error(f"Failed to fetch position data. Code: {pos_response.get('retCode')}, Msg: {pos_response.get('retMsg')}")
                # Don't return False immediately, try fetching balance anyway, but log the position failure
                # If position fetch fails repeatedly, the bot might take wrong actions.
            else:
                position_list = pos_response.get('result', {}).get('list', [])
                if not position_list:
                    # No position data returned, assume flat
                    self._reset_position_state("No position data found in API response.")
                else:
                    # Assuming One-Way mode (positionIdx=0 or only one entry in list)
                    # If Hedge Mode, need to find the correct position entry (Buy/Sell)
                    # For now, assume the first entry is the relevant one for One-Way
                    pos_data = position_list[0]
                    pos_qty_str = pos_data.get('size', '0')
                    pos_qty = safe_decimal_conversion(pos_qty_str, 'position size')
                    side_str = pos_data.get('side', 'None') # 'Buy', 'Sell', or 'None'

                    # Check for valid position (size > epsilon and side is Buy/Sell)
                    # Use a small epsilon for floating point comparisons if needed, but direct > 0 should work for Decimal
                    if pos_qty is not None and pos_qty > Decimal("0") and side_str in [SIDE_BUY, SIDE_SELL]:
                        self.current_qty = pos_qty
                        self.entry_price = safe_decimal_conversion(pos_data.get('avgPrice'), 'entry price')
                        self.current_side = POS_LONG if side_str == SIDE_BUY else POS_SHORT
                        if self.entry_price is None:
                            logger.warning(f"Position found ({side_str} {pos_qty_str}), but average price is invalid. State may be inaccurate.")
                            # Decide how to handle this - reset state? Or proceed cautiously?
                            # Proceeding cautiously for now.
                    else:
                        # Position size is zero, negligible, or side is 'None' -> Treat as flat
                        reset_reason = f"Position size ({pos_qty_str}) is zero/negligible or side ('{side_str}') indicates no active position."
                        self._reset_position_state(reset_reason)

                logger.debug(f"Position State Updated: Side={self.current_side}, Qty={self.current_qty}, Entry={self.entry_price}")
                position_updated = True


            # --- Fetch Balance (Primarily for Logging Here) ---
            # Actual balance for calculations is fetched when needed (e.g., sizing)
            _ = self._get_available_balance() # Fetch and log balance info

            # --- Clear Tracked Orders if Flat ---
            if self.current_side == POS_NONE:
                if self.sl_order_id or self.tp_order_id:
                     logger.debug("Not in position, clearing tracked separate SL/TP order IDs.")
                     self.sl_order_id = None
                     self.tp_order_id = None
            # Optional: If in position and using separate orders, verify tracked SL/TP orders still exist via get_open_orders.
            # This adds API calls but increases robustness if orders get cancelled manually/externally.

            logger.debug("State update perception complete.")
            return position_updated # Return True only if position fetch was successful

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(f"Pybit API Error during state update: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during state update: {e}", exc_info=True)
            return False

    def _reset_position_state(self, reason: str):
        """Resets internal position tracking variables to 'flat' state."""
        if self.current_side != POS_NONE: # Log only if state is actually changing
            logger.info(f"Resetting position state to NONE. Reason: {reason}")
        self.current_side = POS_NONE
        self.current_qty = Decimal("0.0")
        self.entry_price = None
        # Do NOT clear SL/TP order IDs here. They might need to be cancelled
        # in the *same* iteration if an exit was just triggered. They will be
        # cleared in the *next* state update if the position remains flat.

    def _fetch_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Decimal]]:
        """
        Fetches OHLCV (Kline) data and the latest ticker price using Pybit.

        Returns:
            A tuple containing:
            - DataFrame with OHLCV data (or None if fetch fails).
            - Latest price as a Decimal (or None if fetch fails).
        """
        if not self.session or not self.category or not self.timeframe or not self.symbol:
            logger.error("Cannot fetch data: Missing session, category, timeframe, or symbol.")
            return None, None
        logger.debug("Fetching market data...")
        ohlcv_df: Optional[pd.DataFrame] = None
        current_price: Optional[Decimal] = None

        # --- Fetch OHLCV (Kline) ---
        try:
            limit = self.strategy_config.ohlcv_limit
            # Bybit API expects limit up to 1000 for kline, adjust if needed based on strategy lookback
            limit = min(limit, 1000)
            logger.debug(f"Fetching Kline: {self.symbol}, Interval: {self.timeframe}, Limit: {limit}")
            kline_response = self.session.get_kline(
                category=self.category, symbol=self.symbol, interval=self.timeframe, limit=limit
            )

            if not (kline_response and kline_response.get('retCode') == RET_CODE_OK):
                logger.warning(f"Could not fetch OHLCV data. Code: {kline_response.get('retCode')}, Msg: {kline_response.get('retMsg')}")
            else:
                kline_list = kline_response.get('result', {}).get('list', [])
                if not kline_list:
                    logger.warning("OHLCV data list is empty in the response.")
                else:
                    # Bybit returns Kline data: [timestamp, open, high, low, close, volume, turnover]
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                    ohlcv_df = pd.DataFrame(kline_list, columns=columns)
                    # Convert timestamp to numeric, then to datetime (UTC)
                    ohlcv_df['timestamp'] = pd.to_numeric(ohlcv_df['timestamp'])
                    ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms', utc=True)
                    # Convert other columns to numeric
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                        ohlcv_df[col] = pd.to_numeric(ohlcv_df[col])
                    # Ensure data is sorted chronologically (Bybit usually returns descending)
                    ohlcv_df = ohlcv_df.sort_values(by='timestamp').reset_index(drop=True)
                    # Set datetime as index
                    ohlcv_df.set_index('datetime', inplace=True)
                    logger.debug(f"Successfully fetched and processed {len(ohlcv_df)} candles. Latest: {ohlcv_df.index[-1]}")

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(f"Pybit API Error fetching kline data: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error processing kline data: {e}", exc_info=True)

        # --- Fetch Ticker ---
        try:
            logger.debug(f"Fetching ticker for {self.symbol}...")
            ticker_response = self.session.get_tickers(category=self.category, symbol=self.symbol)

            if not (ticker_response and ticker_response.get('retCode') == RET_CODE_OK):
                logger.warning(f"Could not fetch ticker data. Code: {ticker_response.get('retCode')}, Msg: {ticker_response.get('retMsg')}")
            else:
                ticker_list = ticker_response.get('result', {}).get('list', [])
                if not ticker_list:
                     logger.warning("Ticker data list is empty in the response.")
                else:
                     # Assuming the first ticker in the list is the correct one
                     last_price_str = ticker_list[0].get('lastPrice')
                     current_price = safe_decimal_conversion(last_price_str, 'lastPrice')
                     if current_price is None:
                         logger.warning("Ticker data retrieved but missing valid 'lastPrice'.")
                     else:
                         logger.debug(f"Last Price: {current_price}")

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(f"Pybit API Error fetching ticker data: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error processing ticker data: {e}", exc_info=True)

        # --- Final Check and Return ---
        if ohlcv_df is None:
            logger.warning("OHLCV data fetch failed or resulted in empty data.")
        if current_price is None:
            logger.warning("Current price fetch failed or resulted in invalid data.")

        # Return data even if one part failed, allows strategy to potentially use partial data if designed for it
        return ohlcv_df, current_price

    # --- Indicator Calculation, Signal Generation (Improved Error Handling) ---
    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculates indicators using the external 'indicators' module. Validates results.

        Args:
            ohlcv_df: DataFrame with OHLCV data.

        Returns:
            DataFrame with indicators added, or None if calculation fails or
            required indicators are missing/NaN.
        """
        if ohlcv_df is None or ohlcv_df.empty:
            logger.warning("Cannot calculate indicators: Input DataFrame is None or empty.")
            return None
        if len(ohlcv_df) < max(self.strategy_config.indicator_settings.evt_length, self.strategy_config.indicator_settings.atr_period) + 5: # Need enough data
            logger.warning(f"Cannot calculate indicators: Insufficient data length ({len(ohlcv_df)} candles).")
            return None

        logger.debug(f"Calculating indicators on {len(ohlcv_df)} candles...")
        try:
            # Prepare config dictionary for the indicators module
            indicator_config_dict = {
                "indicator_settings": self.strategy_config.indicator_settings.model_dump(), # Use Pydantic's export method
                "analysis_flags": self.strategy_config.analysis_flags.model_dump(),
                # Add any other parameters the indicators module might need from the main config
            }

            # Call the external calculation function
            # Ensure the function handles potential NaNs in input gracefully
            df_with_indicators = ind.calculate_all_indicators(ohlcv_df.copy(), indicator_config_dict) # Pass a copy

            # --- Validation ---
            if df_with_indicators is None:
                logger.error("Indicator calculation script (indicators.py) returned None.")
                return None
            if df_with_indicators.empty:
                logger.error("Indicator calculation script returned an empty DataFrame.")
                return None

            # Define expected column names based on config
            evt_len = self.strategy_config.indicator_settings.evt_length
            atr_len = self.strategy_config.indicator_settings.atr_period
            evt_trend_col = f'evt_trend_{evt_len}'
            evt_buy_col = f'evt_buy_{evt_len}'
            evt_sell_col = f'evt_sell_{evt_len}'
            atr_col = f'ATRr_{atr_len}' # Default pandas_ta name for ATR

            required_cols = [evt_trend_col, evt_buy_col, evt_sell_col]
            if self.strategy_config.analysis_flags.use_atr:
                required_cols.append(atr_col)

            missing_cols = [col for col in required_cols if col not in df_with_indicators.columns]
            if missing_cols:
                logger.error(f"Required indicator columns missing after calculation: {missing_cols}. Check 'indicators.py'.")
                return None

            # Check for NaNs in the *latest* row's critical columns
            # Accessing the last row safely
            if df_with_indicators.empty:
                logger.error("Indicator DataFrame is empty after calculation, cannot check latest row.")
                return None
            latest_row = df_with_indicators.iloc[-1]
            nan_cols = [col for col in required_cols if pd.isna(latest_row.get(col))] # Use .get() for safety
            if nan_cols:
                 logger.warning(f"NaN values found in critical indicator columns of latest row: {nan_cols}. Cannot generate reliable signal.")
                 # Depending on strategy, might allow proceeding or force skip
                 return None # Safer option: skip iteration if latest data is NaN

            logger.debug("Indicators calculated successfully.")
            return df_with_indicators

        except Exception as e:
            logger.error(f"Error during indicator calculation: {e}", exc_info=True)
            return None

    def _generate_signals(self, df_ind: pd.DataFrame) -> Optional[Literal['buy', 'sell']]:
        """
        Generates trading signals ('buy' or 'sell') based on the last
        indicator data point in the provided DataFrame.

        Args:
            df_ind: DataFrame with calculated indicators.

        Returns:
            'buy', 'sell', or None if no signal is generated or data is invalid.
        """
        if df_ind is None or df_ind.empty:
            logger.debug("Cannot generate signals: Indicator DataFrame is missing or empty.")
            return None
        logger.debug("Generating trading signals...")
        try:
            # Access the latest row safely
            latest = df_ind.iloc[-1]
            evt_len = self.strategy_config.indicator_settings.evt_length
            buy_col = f'evt_buy_{evt_len}'
            sell_col = f'evt_sell_{evt_len}'

            # Check required columns exist and are not NaN in the latest row
            if not all(col in latest.index and pd.notna(latest[col]) for col in [buy_col, sell_col]):
                 logger.warning(f"EVT Buy/Sell signal columns missing or NaN in latest data ({latest.name}). Cannot generate signal.")
                 return None

            buy_signal = bool(latest[buy_col])   # Explicitly cast to bool
            sell_signal = bool(latest[sell_col]) # Explicitly cast to bool

            # --- Signal Logic ---
            # Basic logic: Enter on the first buy/sell signal after being flat.
            # (More complex logic like filtering consecutive signals could be added here)
            if buy_signal and sell_signal:
                logger.warning(f"Both Buy and Sell signals are active simultaneously on latest candle ({latest.name}). Ignoring signals.")
                return None
            elif buy_signal:
                logger.info(f"{Fore.GREEN}BUY signal generated based on EVT Buy flag at {latest.name}.{Style.RESET_ALL}")
                return POS_LONG # Use internal 'long'/'short' representation
            elif sell_signal:
                logger.info(f"{Fore.RED}SELL signal generated based on EVT Sell flag at {latest.name}.{Style.RESET_ALL}")
                return POS_SHORT # Use internal 'long'/'short' representation
            else:
                # logger.debug("No new entry signal generated on the latest candle.")
                return None

        except IndexError:
            logger.warning("IndexError generating signals (DataFrame likely too short or empty).")
            return None
        except KeyError as e:
             logger.error(f"KeyError generating signals: Missing expected column '{e}'. Check indicator calculation.")
             return None
        except Exception as e:
            logger.error(f"Unexpected error generating signals: {e}", exc_info=True)
            return None

    # --- SL/TP Calculation, Position Sizing (Refined) ---
    def _calculate_sl_tp(self, df_ind: pd.DataFrame, side: Literal['long', 'short'], entry_price_approx: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates Stop Loss (SL) and Take Profit (TP) prices based on ATR,
        respecting the market's price tick precision.

        Args:
            df_ind: DataFrame with indicators (must include ATR).
            side: The intended trade side ('long' or 'short').
            entry_price_approx: The approximate entry price (e.g., current market price).

        Returns:
            A tuple containing:
            - Calculated SL price (Decimal), or None if calculation fails.
            - Calculated TP price (Decimal), or None if calculation fails or TP is disabled.
        """
        if df_ind is None or df_ind.empty:
            logger.error("Cannot calculate SL/TP: Missing indicator data.")
            return None, None
        if self.price_tick is None or self.price_tick <= Decimal(0):
            logger.error(f"Cannot calculate SL/TP: Invalid price tick ({self.price_tick}).")
            return None, None
        if entry_price_approx <= Decimal(0):
             logger.error(f"Cannot calculate SL/TP: Invalid approximate entry price ({entry_price_approx}).")
             return None, None
        if not self.strategy_config.analysis_flags.use_atr:
            logger.error("Cannot calculate SL/TP: ATR usage is disabled in config (analysis_flags.use_atr).")
            return None, None

        logger.debug(f"Calculating SL/TP for {side} entry near {entry_price_approx}...")
        try:
            # --- Get Latest ATR ---
            atr_len = self.strategy_config.indicator_settings.atr_period
            atr_col = f'ATRr_{atr_len}' # Default pandas_ta name
            if atr_col not in df_ind.columns:
                logger.error(f"ATR column '{atr_col}' not found in indicator DataFrame."); return None, None
            latest_atr_val = df_ind.iloc[-1].get(atr_col)
            if pd.isna(latest_atr_val):
                 logger.error(f"Latest ATR value in column '{atr_col}' is NaN."); return None, None

            latest_atr = safe_decimal_conversion(latest_atr_val, 'latest ATR')
            if latest_atr is None or latest_atr <= Decimal(0):
                logger.warning(f"Invalid ATR value ({latest_atr_val}) for SL/TP calculation. Cannot proceed.")
                return None, None

            # --- Stop Loss Calculation ---
            sl_multiplier = self.strategy_config.stop_loss_atr_multiplier
            if sl_multiplier <= 0:
                logger.error("Stop Loss ATR multiplier must be positive.")
                return None, None
            sl_offset = latest_atr * sl_multiplier
            stop_loss_price_raw = (entry_price_approx - sl_offset) if side == POS_LONG else (entry_price_approx + sl_offset)

            # Ensure SL is not zero or negative before rounding
            if stop_loss_price_raw <= Decimal(0):
                 logger.error(f"Raw SL price calculated as zero or negative ({stop_loss_price_raw}). Check ATR/multiplier/price.")
                 return None, None

            # Round SL *away* from the entry price to respect the tick size
            # For Long: Round DOWN (e.g., 10.123 -> 10.12 if tick=0.01)
            # For Short: Round UP (e.g., 9.876 -> 9.88 if tick=0.01)
            rounding_mode_sl = ROUND_DOWN if side == POS_LONG else ROUND_UP
            sl_price_adjusted = (stop_loss_price_raw / self.price_tick).quantize(Decimal('0'), rounding=rounding_mode_sl) * self.price_tick

            # Sanity check: Ensure SL didn't cross entry after rounding
            if side == POS_LONG and sl_price_adjusted >= entry_price_approx:
                 sl_price_adjusted = entry_price_approx - self.price_tick # Place it one tick away
                 logger.warning(f"Adjusted Buy SL ({sl_price_adjusted}) was >= approx entry ({entry_price_approx}). Moved SL one tick below entry.")
            elif side == POS_SHORT and sl_price_adjusted <= entry_price_approx:
                 sl_price_adjusted = entry_price_approx + self.price_tick # Place it one tick away
                 logger.warning(f"Adjusted Sell SL ({sl_price_adjusted}) was <= approx entry ({entry_price_approx}). Moved SL one tick above entry.")

            # Final check: Ensure SL is still positive after adjustments
            if sl_price_adjusted <= Decimal(0):
                 logger.error(f"Final SL price is zero or negative ({sl_price_adjusted}) after rounding/adjustment. Cannot set SL.")
                 # If SL fails, should we abort the entry? Yes, usually.
                 return None, None

            # --- Take Profit Calculation ---
            tp_price_adjusted: Optional[Decimal] = None
            tp_multiplier = self.strategy_config.take_profit_atr_multiplier
            if tp_multiplier > 0:
                tp_offset = latest_atr * tp_multiplier
                take_profit_price_raw = (entry_price_approx + tp_offset) if side == POS_LONG else (entry_price_approx - tp_offset)

                # Ensure TP is logical relative to entry before rounding
                if (side == POS_LONG and take_profit_price_raw <= entry_price_approx) or \
                   (side == POS_SHORT and take_profit_price_raw >= entry_price_approx):
                    logger.warning(f"Raw TP price ({take_profit_price_raw}) is not logical relative to approx entry ({entry_price_approx}). Skipping TP.")
                elif take_profit_price_raw <= Decimal(0):
                    logger.warning(f"Raw TP price ({take_profit_price_raw}) is zero or negative. Skipping TP.")
                else:
                    # Round TP *towards* the entry price (or nearest tick) to increase fill probability? Or away?
                    # Let's round DOWN for BUY TP, UP for SELL TP (makes target slightly harder to hit but locks profit sooner if hit)
                    # Alternative: Round normally (ROUND_HALF_UP)
                    # Let's try rounding DOWN for BUY, UP for SELL (like SL) for consistency. Test this logic.
                    rounding_mode_tp = ROUND_DOWN if side == POS_LONG else ROUND_UP
                    tp_price_adjusted_candidate = (take_profit_price_raw / self.price_tick).quantize(Decimal('0'), rounding=rounding_mode_tp) * self.price_tick

                    # Sanity check: Ensure TP didn't cross entry after rounding
                    if side == POS_LONG and tp_price_adjusted_candidate <= entry_price_approx:
                         tp_price_adjusted_candidate = entry_price_approx + self.price_tick # Place one tick away
                         logger.warning(f"Adjusted Buy TP ({tp_price_adjusted_candidate}) was <= approx entry ({entry_price_approx}). Moved TP one tick above entry.")
                    elif side == POS_SHORT and tp_price_adjusted_candidate >= entry_price_approx:
                         tp_price_adjusted_candidate = entry_price_approx - self.price_tick # Place one tick away
                         logger.warning(f"Adjusted Sell TP ({tp_price_adjusted_candidate}) was >= approx entry ({entry_price_approx}). Moved TP one tick below entry.")

                    # Final check: Ensure TP is positive
                    if tp_price_adjusted_candidate <= Decimal(0):
                        logger.warning(f"Final TP price is zero or negative ({tp_price_adjusted_candidate}) after rounding/adjustment. Skipping TP.")
                    else:
                        tp_price_adjusted = tp_price_adjusted_candidate
            else:
                logger.info("Take Profit multiplier is zero or negative. TP is disabled.")

            # Format for logging
            sl_formatted = self._format_price_str(sl_price_adjusted)
            tp_formatted = self._format_price_str(tp_price_adjusted) if tp_price_adjusted else 'None'
            logger.info(f"Calculated SL: {sl_formatted}, TP: {tp_formatted} (Based on ATR: {latest_atr:.5f})")

            return sl_price_adjusted, tp_price_adjusted

        except (InvalidOperation, TypeError, Exception) as e:
            logger.error(f"Error calculating SL/TP: {e}", exc_info=True)
            return None, None

    def _calculate_position_size(self, entry_price_approx: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """
        Calculates the position size based on configured risk percentage,
        available balance, entry/SL prices, and market constraints.

        Args:
            entry_price_approx: The approximate entry price.
            stop_loss_price: The calculated stop loss price.

        Returns:
            The calculated position size (Decimal) respecting quantity steps
            and minimums, or None if calculation fails or size is too small.
        """
        # --- Input Validation ---
        if not all([self.qty_step, self.min_qty, self.price_tick, self.base_coin, self.quote_coin]):
             logger.error("Cannot calculate size: Missing critical market details (steps, ticks, coins).")
             return None
        if entry_price_approx <= 0 or stop_loss_price <= 0:
             logger.error(f"Cannot calculate size: Invalid entry ({entry_price_approx}) or SL ({stop_loss_price}) price.")
             return None
        price_diff = abs(entry_price_approx - stop_loss_price)
        if price_diff < self.price_tick: # Check if difference is smaller than smallest price increment
            logger.error(f"Cannot calculate size: Entry price ({entry_price_approx}) and SL price ({stop_loss_price}) are too close (diff: {price_diff} < tick: {self.price_tick}).")
            return None
        if self.strategy_config.risk_per_trade <= 0 or self.strategy_config.risk_per_trade >= 1:
             logger.error(f"Invalid risk_per_trade ({self.strategy_config.risk_per_trade}). Must be between 0 and 1.")
             return None

        logger.debug("Calculating position size...")
        try:
            # --- Get Available Balance ---
            available_balance = self._get_available_balance()
            if available_balance is None: # Check explicitly for None, as 0 is a valid (but unusable) balance
                logger.error("Cannot calculate position size: Failed to fetch available balance.")
                return None
            if available_balance <= Decimal("0"):
                logger.error(f"Cannot calculate position size: Available balance ({available_balance} {self.quote_coin}) is zero or negative.")
                return None

            # --- Calculate Risk Amount ---
            risk_amount_quote = available_balance * self.strategy_config.risk_per_trade
            logger.debug(f"Available Balance: {available_balance:.4f} {self.quote_coin}, Risk %: {self.strategy_config.risk_per_trade:.2%}, Risk Amount: {risk_amount_quote:.4f} {self.quote_coin}")

            # --- Calculate Raw Size Based on Category ---
            position_size_raw: Decimal
            # Note: contract_multiplier is crucial here, especially for inverse contracts.
            if self.contract_multiplier is None or self.contract_multiplier <= 0:
                 logger.error("Cannot calculate size: Invalid contract multiplier.")
                 return None

            if self.category == 'inverse':
                 # Risk_Quote = Contracts * Multiplier * |1/entry_price - 1/stop_loss_price|
                 # Contracts = Risk_Quote / (Multiplier * |1/entry_price - 1/stop_loss_price|)
                 try:
                     inv_entry = Decimal(1) / entry_price_approx
                     inv_sl = Decimal(1) / stop_loss_price
                 except InvalidOperation:
                     logger.error("Division by zero error during inverse contract size calculation (price likely zero).")
                     return None
                 size_denominator = self.contract_multiplier * abs(inv_entry - inv_sl)
                 if size_denominator <= 0:
                     logger.error("Inverse size denominator is zero or negative. Cannot calculate size.")
                     return None
                 position_size_raw = risk_amount_quote / size_denominator
                 # Result is in Contracts (often Base currency units, e.g., BTC for BTCUSD inverse)

            elif self.category == 'linear':
                 # Risk_Quote = Contracts * Multiplier * |entry_price - stop_loss_price|
                 # Contracts = Risk_Quote / (Multiplier * |entry_price - stop_loss_price|)
                 size_denominator = self.contract_multiplier * price_diff
                 if size_denominator <= 0: # Should be caught by price_diff check earlier, but double-check
                     logger.error("Linear size denominator is zero or negative. Cannot calculate size.")
                     return None
                 position_size_raw = risk_amount_quote / size_denominator
                 # Result is in Contracts (Base currency units, e.g., BTC for BTCUSDT)

            elif self.category == 'spot':
                 # Risk_Quote = Amount_Base * |entry_price - stop_loss_price|
                 # Amount_Base = Risk_Quote / |entry_price - stop_loss_price|
                 if price_diff <= 0: # Should be caught earlier
                     logger.error("Spot size denominator (price_diff) is zero or negative.")
                     return None
                 position_size_raw = risk_amount_quote / price_diff
                 # Result is in Base currency units (e.g., BTC for BTC/USDT)
            else:
                 logger.error(f"Position sizing not implemented for category: {self.category}")
                 return None

            logger.debug(f"Raw calculated size: {position_size_raw:.8f} {self.base_coin}")

            # --- Apply Quantity Constraints (Step and Minimum) ---
            if self.qty_step <= 0:
                 logger.error(f"Invalid quantity step ({self.qty_step}). Cannot adjust size.")
                 return None

            # Adjust for quantity step (round DOWN to not exceed risk)
            # position_size_adjusted = (position_size_raw // self.qty_step) * self.qty_step
            # Using quantize for potentially better handling of decimal places
            position_size_adjusted = (position_size_raw / self.qty_step).quantize(Decimal('0'), rounding=ROUND_DOWN) * self.qty_step


            if position_size_adjusted <= Decimal(0):
                 logger.warning(f"Calculated position size is zero after step adjustment. Raw: {position_size_raw}, Step: {self.qty_step}. Insufficient balance for risk settings?")
                 return None

            # Check against minimum order quantity
            if self.min_qty is None or self.min_qty < 0:
                 logger.error(f"Invalid minimum quantity ({self.min_qty}). Cannot validate size.")
                 return None

            if position_size_adjusted < self.min_qty:
                logger.warning(f"Calculated size ({position_size_adjusted}) is below minimum allowed quantity ({self.min_qty}). Insufficient capital for the configured risk, or risk % too low.")
                # Option: Could place min_qty order instead, but that violates risk management. Best to return None.
                return None

            # --- Log and Return ---
            size_formatted = self._format_qty(position_size_adjusted)
            logger.info(f"Calculated position size: {size_formatted} {self.base_coin} ")
            return position_size_adjusted

        except (InvalidOperation, TypeError, Exception) as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    # --- Order Management Helpers (Pybit Implementation - Enhanced) ---

    def _format_qty(self, qty: Optional[Decimal]) -> Optional[str]:
        """
        Formats a quantity Decimal to a string according to the market's qty_step.
        Rounds DOWN to the nearest step.

        Args:
            qty: The quantity to format.

        Returns:
            The formatted quantity string, or None if input is invalid.
        """
        if qty is None or qty < 0: return None # Cannot format invalid quantity
        if self.qty_step is None or self.qty_step <= 0:
            logger.warning(f"Cannot format quantity: Invalid qty_step ({self.qty_step}). Returning raw string.")
            return str(qty)

        try:
            # Quantize DOWN to the step size
            quantized_qty = (qty / self.qty_step).quantize(Decimal('0'), rounding=ROUND_DOWN) * self.qty_step

            # Determine decimal places from step for formatting string
            step_str = str(self.qty_step.normalize()) # normalize() removes trailing zeros
            if '.' in step_str:
                decimals = len(step_str.split('.')[-1])
            else:
                decimals = 0

            return f"{quantized_qty:.{decimals}f}"
        except (InvalidOperation, TypeError) as e:
            logger.error(f"Error formatting quantity {qty} with step {self.qty_step}: {e}")
            return None # Return None on formatting error

    def _format_price_str(self, price: Optional[Decimal]) -> Optional[str]:
        """
        Formats a price Decimal to a string according to the market's price_tick.
        Uses standard rounding to the nearest tick.

        Args:
            price: The price to format.

        Returns:
            The formatted price string, or None if input is invalid.
        """
        if price is None or price <= 0: return None # Cannot format invalid price
        if self.price_tick is None or self.price_tick <= 0:
            logger.warning(f"Cannot format price: Invalid price_tick ({self.price_tick}). Returning raw string.")
            return str(price)

        try:
            # Quantize to the NEAREST tick size (ROUND_HALF_UP is standard rounding)
            # Note: Specific rounding might be needed depending on order type/exchange rules.
            # E.g., Limit buy might need ROUND_DOWN, Limit sell ROUND_UP.
            # For general formatting, nearest is common. SL/TP calculations handle directional rounding.
            quantized_price = (price / self.price_tick).quantize(Decimal('0'), rounding=ROUND_HALF_UP) * self.price_tick

            # Ensure quantized price isn't zero if original wasn't (can happen with tiny prices/large ticks)
            if quantized_price <= 0 and price > 0:
                 logger.warning(f"Price {price} quantized to zero with tick {self.price_tick}. Using smallest possible price (tick size).")
                 quantized_price = self.price_tick # Use the tick size itself as the smallest valid price string

            # Determine decimal places from tick for formatting string
            tick_str = str(self.price_tick.normalize()) # normalize() removes trailing zeros
            if '.' in tick_str:
                decimals = len(tick_str.split('.')[-1])
            else:
                decimals = 0

            return f"{quantized_price:.{decimals}f}"
        except (InvalidOperation, TypeError) as e:
            logger.error(f"Error formatting price {price} with tick {self.price_tick}: {e}")
            return None # Return None on formatting error


    def _place_order(self, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Wrapper for placing orders with Pybit's place_order method.
        Includes enhanced logging and basic error code handling.

        Args:
            params: Dictionary of parameters for the place_order call.

        Returns:
            The 'result' dictionary from the Pybit response if successful,
            otherwise None.
        """
        if not self.session:
            logger.error("Cannot place order: Session not initialized.")
            return None

        # --- Basic Parameter Validation ---
        required_params = ['category', 'symbol', 'side', 'orderType', 'qty']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            logger.error(f"Missing required parameters for placing order: {', '.join(missing_params)}. Params: {params}")
            return None
        # Validate quantity format (string)
        if not isinstance(params['qty'], str) or not params['qty']:
             logger.error(f"Invalid 'qty' parameter type or value: {params['qty']}. Must be non-empty string.")
             return None
        # Validate price format if it's a limit order
        if params['orderType'] == ORDER_TYPE_LIMIT and (not isinstance(params.get('price'), str) or not params.get('price')):
             logger.error(f"Invalid 'price' parameter for Limit order: {params.get('price')}. Must be non-empty string.")
             return None
        # Validate trigger price format if present
        if 'triggerPrice' in params and (not isinstance(params.get('triggerPrice'), str) or not params.get('triggerPrice')):
             logger.error(f"Invalid 'triggerPrice' parameter: {params.get('triggerPrice')}. Must be non-empty string.")
             return None
        # Validate SL/TP prices if present
        if 'stopLoss' in params and (not isinstance(params.get('stopLoss'), str) or not params.get('stopLoss')):
             logger.error(f"Invalid 'stopLoss' parameter: {params.get('stopLoss')}. Must be non-empty string.")
             return None
        if 'takeProfit' in params and (not isinstance(params.get('takeProfit'), str) or not params.get('takeProfit')):
             logger.error(f"Invalid 'takeProfit' parameter: {params.get('takeProfit')}. Must be non-empty string.")
             return None

        # Add default positionIdx for derivatives if not provided (assuming One-Way mode)
        if self.category in ['linear', 'inverse'] and 'positionIdx' not in params:
             params['positionIdx'] = POSITION_IDX_ONE_WAY # Default to One-Way

        # --- Build Order Description for Logging ---
        order_desc_parts = [
            params['side'],
            params['orderType'],
            params['qty'],
            params['symbol']
        ]
        if params['orderType'] == ORDER_TYPE_LIMIT and 'price' in params:
            order_desc_parts.append(f"@ {params['price']}")
        if params.get('triggerPrice'):
            order_desc_parts.append(f"(Trigger: {params['triggerPrice']})")
        if params.get('stopLoss'):
            sl_type = f" {params['slOrderType']}" if params.get('slOrderType') else ""
            sl_trig = f" ({params['slTriggerBy']})" if params.get('slTriggerBy') else ""
            order_desc_parts.append(f"SL: {params['stopLoss']}{sl_type}{sl_trig}")
        if params.get('takeProfit'):
            tp_trig = f" ({params['tpTriggerBy']})" if params.get('tpTriggerBy') else ""
            order_desc_parts.append(f"TP: {params['takeProfit']}{tp_trig}")
        if params.get('reduceOnly'):
            order_desc_parts.append("[ReduceOnly]")
        if params.get('timeInForce'):
             order_desc_parts.append(f"[{params['timeInForce']}]")
        if params.get('orderLinkId'):
             order_desc_parts.append(f"LinkID: {params['orderLinkId'][:10]}...") # Shorten for log

        order_description = " ".join(order_desc_parts)
        logger.info(f"{Fore.YELLOW}✨ Forging Order: {order_description}...{Style.RESET_ALL}")
        # Log full parameters at debug level
        logger.debug(f"Order Parameters: {params}")

        # --- Place Order via API ---
        try:
            start_time = time.monotonic()
            response = self.session.place_order(**params)
            end_time = time.monotonic()
            logger.debug(f"Place Order Raw Response (took {end_time - start_time:.3f}s): {response}")

            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown Error')
            result_data = response.get('result', {})
            order_id = result_data.get('orderId') if result_data else None

            # --- Handle Response ---
            if ret_code == RET_CODE_OK:
                if order_id:
                    logger.success(f"{Fore.GREEN}✅ Order placed successfully! OrderID: {format_order_id(order_id)}{Style.RESET_ALL}")
                    return result_data # Return the result dict on success
                else:
                    # This case (OK code but no order ID) shouldn't typically happen for successful placement
                    logger.error(f"{Back.YELLOW}{Fore.BLACK}Order placement reported OK (Code: {ret_code}), but no OrderID found in result.{Style.RESET_ALL} Response: {response}")
                    return None # Treat as failure if ID is missing
            else:
                # Log specific errors more clearly
                log_level = logging.ERROR
                alert_msg = None
                if ret_code in [RET_CODE_INSUFFICIENT_BALANCE_SPOT,
                                RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_1,
                                RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_2,
                                RET_CODE_REDUCE_ONLY_MARGIN_ERROR]:
                    log_level = logging.CRITICAL
                    logger.critical(f"{Back.RED}INSUFFICIENT BALANCE! Code: {ret_code}, Msg: {ret_msg}{Style.RESET_ALL}")
                    alert_msg = f"CRITICAL: Insufficient balance for {self.symbol} order! ({ret_code})"
                elif ret_code == RET_CODE_QTY_TOO_SMALL:
                    logger.error(f"Order quantity invalid: Quantity too small. Code: {ret_code}, Msg: {ret_msg}. Min Qty: {self.min_qty}")
                elif ret_code == RET_CODE_QTY_INVALID_PRECISION:
                     logger.error(f"Order quantity invalid: Precision error. Code: {ret_code}, Msg: {ret_msg}. Qty Step: {self.qty_step}")
                elif ret_code == RET_CODE_PRICE_TOO_LOW:
                     logger.error(f"Order price invalid: Price too low. Code: {ret_code}, Msg: {ret_msg}.")
                elif ret_code == RET_CODE_PRICE_INVALID_PRECISION:
                     logger.error(f"Order price invalid: Precision error. Code: {ret_code}, Msg: {ret_msg}. Price Tick: {self.price_tick}")
                elif ret_code == RET_CODE_TOO_MANY_VISITS:
                     logger.warning(f"{Fore.YELLOW}Rate Limit Hit! Code: {ret_code}, Msg: {ret_msg}. Consider increasing loop delay.{Style.RESET_ALL}")
                     # Potentially add a small delay here before returning failure
                     time.sleep(1)
                elif ret_code == RET_CODE_REDUCE_ONLY_QTY_ERROR:
                     logger.error(f"ReduceOnly order failed: Quantity exceeds position size. Code: {ret_code}, Msg: {ret_msg}")
                else:
                     # Generic failure message
                     logger.log(log_level, f"{Back.RED}{Fore.WHITE}❌ Order placement failed! Code: {ret_code}, Msg: {ret_msg}{Style.RESET_ALL}")
                     # Log full params again on failure for easier debugging
                     logger.error(f"Failed Order Parameters: {params}")

                # Send SMS alert if defined
                if alert_msg:
                    send_sms_alert(alert_msg, self.sms_config)

                return None # Return None on failure

        except (InvalidRequestError, FailedRequestError) as pybit_e:
             # Handle Pybit library specific exceptions (e.g., connection errors, malformed requests before sending)
             logger.error(f"{Back.RED}Pybit API Error during order placement: {pybit_e}{Style.RESET_ALL}", exc_info=False) # Keep exc_info concise
             logger.error(f"Status Code: {pybit_e.status_code}, Response: {pybit_e.response}")
             if pybit_e.status_code == 403 and "Timestamp for this request is outside of the recvWindow" in str(pybit_e.response):
                 logger.critical(f"{Back.RED}Timestamp/recvWindow error. Check system clock sync and potentially increase recv_window in HTTP client.{Style.RESET_ALL}")
             return None
        except Exception as e:
            # Handle unexpected errors during the process
            logger.error(f"Unexpected exception during order placement: {e}", exc_info=True)
            return None

    def _cancel_single_order(self, order_id: str, order_link_id: Optional[str] = None, reason: str = "Strategy Action") -> bool:
        """
        Cancels a single order by its Order ID or Order Link ID using Pybit.

        Args:
            order_id: The exchange Order ID.
            order_link_id: The client Order Link ID (provide one or the other).
            reason: A string describing why the order is being cancelled (for logging).

        Returns:
            True if cancellation was successful or the order was already closed/not found.
            False if cancellation failed for other reasons.
        """
        if not self.session or not self.category or not self.symbol:
            logger.error("Cannot cancel order: Session, category, or symbol not set.")
            return False
        if not order_id and not order_link_id:
             logger.error("Cannot cancel order: Must provide order_id or order_link_id.")
             return False

        id_type = "OrderID" if order_id else "OrderLinkID"
        id_value = order_id if order_id else order_link_id
        log_id = format_order_id(id_value)

        logger.info(f"Attempting cancellation for {id_type} {log_id} ({reason})...")
        try:
            cancel_params = {
                "category": self.category,
                "symbol": self.symbol,
            }
            if order_id:
                cancel_params["orderId"] = order_id
            else:
                cancel_params["orderLinkId"] = order_link_id

            response = self.session.cancel_order(**cancel_params)
            logger.debug(f"Cancel Order ({log_id}) Raw Response: {response}")

            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', '').lower() # Lowercase for easier text matching

            if ret_code == RET_CODE_OK:
                # Check if the result confirms the ID was cancelled
                cancelled_id = response.get('result', {}).get('orderId') or response.get('result', {}).get('orderLinkId')
                logger.info(f"Order {log_id} cancelled successfully (Confirmed ID: {cancelled_id}).")
                return True
            # Treat "order not found" or "already closed" as success for cancellation purposes
            elif ret_code in [RET_CODE_ORDER_NOT_FOUND, RET_CODE_ORDER_NOT_FOUND_OR_CLOSED] or \
                 "order does not exist" in ret_msg or "order not found" in ret_msg or "already been filled" in ret_msg or "already closed" in ret_msg:
                 logger.warning(f"Order {log_id} not found, already closed, or already filled. Assuming cancellation is effectively successful.")
                 return True
            else:
                # Log other cancellation failures as errors
                logger.error(f"Failed to cancel order {log_id}. Code: {ret_code}, Msg: {response.get('retMsg')}")
                return False

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(f"Pybit API Error cancelling order {log_id}: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
            return False
        except Exception as e:
            logger.error(f"Unexpected exception cancelling order {log_id}: {e}", exc_info=True)
            return False

    def _cancel_all_open_orders(self, reason: str = "Strategy Action") -> bool:
        """
        Cancels ALL open orders for the current symbol and category.

        Args:
            reason: A string describing why orders are being cancelled (for logging).

        Returns:
            True if the cancellation request was accepted (even if 0 orders were cancelled).
            False if the API call failed.
        """
        if not self.session or not self.category or not self.symbol:
            logger.error("Cannot cancel all orders: Session, category, or symbol not set.")
            return False

        logger.info(f"Attempting to cancel ALL open orders for {self.symbol} ({self.category}) due to: {reason}...")
        try:
            # Bybit's cancel_all_orders usually works per category/symbol
            response = self.session.cancel_all_orders(
                category=self.category,
                symbol=self.symbol,
                # Optional: filter by orderType or settleCoin if needed, but usually cancel all for the symbol
                # orderFilter="Order" # Can be Order, StopOrder, OcoOrder, TpslOrder
            )
            logger.debug(f"Cancel All Orders Response: {response}")
            ret_code = response.get('retCode')

            if ret_code == RET_CODE_OK:
                # Response often includes a list of cancelled order IDs
                cancelled_list = response.get('result', {}).get('list', [])
                if cancelled_list:
                     num_cancelled = len(cancelled_list)
                     logger.info(f"Successfully cancelled {num_cancelled} open order(s) for {self.symbol}.")
                     # Log first few IDs if needed for debugging
                     cancelled_ids_preview = [format_order_id(o.get('orderId')) for o in cancelled_list[:3]]
                     logger.debug(f"Cancelled IDs (preview): {cancelled_ids_preview}")
                else:
                     logger.info(f"No open orders found to cancel for {self.symbol}.")
                return True # API call succeeded
            else:
                 # Handle potential failures
                 logger.error(f"Failed to cancel all orders for {self.symbol}. Code: {ret_code}, Msg: {response.get('retMsg')}")
                 # Check if list exists even on error? Unlikely but possible.
                 cancelled_list = response.get('result', {}).get('list', [])
                 if cancelled_list:
                      logger.warning(f"Cancel all failed overall, but response indicates {len(cancelled_list)} orders might have been cancelled.")
                      # Treat as partial success? Or failure? Let's treat as failure as the command didn't fully succeed.
                      return False
                 return False # API call failed

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(f"Pybit API Error cancelling all orders: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})", exc_info=False)
            return False
        except Exception as e:
            logger.error(f"Unexpected exception cancelling all orders: {e}", exc_info=True)
            return False

    # --- Core Strategy Logic Handlers (Entry, Exit) ---

    def _handle_exit(self, df_ind: pd.DataFrame) -> bool:
        """
        Checks exit conditions based on indicators (e.g., EVT trend reversal)
        and closes the current position if conditions are met.

        Args:
            df_ind: DataFrame with calculated indicators.

        Returns:
            True if an exit was triggered and attempted (success/failure handled internally).
            False if no exit condition was met or not in a position.
        """
        if self.current_side == POS_NONE:
            # logger.debug("Not in position, skipping exit check.")
            return False # Not in a position to exit

        if df_ind is None or df_ind.empty:
             logger.warning("Cannot check exit conditions: Indicator data missing.")
             return False

        logger.debug(f"Checking exit conditions for {self.current_side} position...")
        should_exit = False
        exit_reason = ""
        try:
            # --- Exit Condition: EVT Trend Reversal ---
            evt_len = self.strategy_config.indicator_settings.evt_length
            trend_col = f'evt_trend_{evt_len}'
            if trend_col not in df_ind.columns:
                 logger.error(f"Cannot check exit: EVT Trend column '{trend_col}' missing.")
                 return False # Cannot evaluate exit without the trend

            latest_trend_val = df_ind.iloc[-1].get(trend_col)
            if pd.isna(latest_trend_val):
                 logger.warning(f"Cannot check exit: Latest EVT trend value is NaN.")
                 return False # Cannot evaluate exit with NaN trend

            latest_trend = int(latest_trend_val) # Convert to integer trend (-1, 0, 1)

            if self.current_side == POS_LONG and latest_trend == -1:
                should_exit = True
                exit_reason = "EVT Trend flipped to Short"
            elif self.current_side == POS_SHORT and latest_trend == 1:
                should_exit = True
                exit_reason = "EVT Trend flipped to Long"

            # --- Add other exit conditions here if needed ---
            # Example: Exit on N consecutive candles against the trend
            # Example: Time-based exit

            # --- Execute Exit Action ---
            if should_exit:
                position_side_display = self.current_side.upper()
                position_qty_display = format_amount(self.symbol, self.current_qty, self.qty_step)
                logger.warning(f"{Fore.YELLOW}🚨 Exit condition triggered for {position_side_display} position ({position_qty_display}): {exit_reason}{Style.RESET_ALL}")

                # 1. Cancel ALL open orders for the symbol FIRST
                # This is crucial to prevent SL/TP orders interfering with the market close order.
                logger.info("Cancelling all open orders before placing exit order...")
                if not self._cancel_all_open_orders(f"Exit Triggered: {exit_reason}"):
                    logger.warning("Failed to cancel all open orders during exit. Proceeding with close attempt, but check UI for stray orders.")
                    # Depending on risk tolerance, might abort exit here if cancellation fails critically
                    # return False # Abort exit if cleanup fails?

                # Clear tracked IDs immediately after cancellation attempt
                self.sl_order_id = None
                self.tp_order_id = None

                # 2. Close the position using a reduce-only market order
                close_side = SIDE_SELL if self.current_side == POS_LONG else SIDE_BUY
                close_qty_str = self._format_qty(self.current_qty) # Use current known quantity

                if not close_qty_str: # Safety check
                     logger.error(f"Failed to format current quantity {self.current_qty} for exit order. Cannot close position.")
                     # This indicates a problem with market details or state tracking
                     return True # Indicate exit was attempted but failed critically

                logger.info(f"Placing Market Close Order: {close_side} {close_qty_str} {self.symbol} [ReduceOnly]")
                close_params: Dict[str, Any] = {
                    "category": self.category,
                    "symbol": self.symbol,
                    "side": close_side,
                    "orderType": ORDER_TYPE_MARKET,
                    "qty": close_qty_str,
                    "reduceOnly": True,
                    # TimeInForce for Market orders: IOC or FOK are common. IOC tries to fill what it can immediately.
                    "timeInForce": TIME_IN_FORCE_IOC # Ensure it executes immediately or cancels partially
                }

                close_order_result = self._place_order(close_params)

                if close_order_result and close_order_result.get('orderId'):
                    closed_order_id = close_order_result['orderId']
                    logger.success(f"{Fore.GREEN}✅ Position Close Market Order ({format_order_id(closed_order_id)}) placed successfully due to: {exit_reason}{Style.RESET_ALL}")

                    # Send Alert
                    alert_msg = f"[{self.symbol.split('/')[0]}] EXITED {position_side_display} ({exit_reason}). Qty: {position_qty_display}"
                    send_sms_alert(alert_msg, self.sms_config)

                    # Optimistically reset internal state - the *next* iteration's _update_state will provide definitive confirmation.
                    # It's important _update_state runs reliably.
                    self._reset_position_state(f"Exit order placed ({exit_reason})")
                    return True # Indicate an exit occurred

                else:
                    # CRITICAL FAILURE: Failed to place the closing order
                    logger.critical(f"{Back.RED}{Fore.WHITE}❌ Failed to place position Close Market Order ({exit_reason}). Manual intervention likely required!{Style.RESET_ALL}")
                    # Attempt an immediate state re-check to see if maybe it filled anyway or state is wrong
                    logger.info("Re-checking position state after failed close order placement...")
                    time.sleep(self.app_config.api_config.api_rate_limit_delay * 2) # Short delay
                    self._update_state() # Re-fetch state
                    if self.current_side == POS_NONE:
                        logger.info("Position appears closed after re-checking state. Close order might have executed despite API error response.")
                        return True # Assume closed based on re-check
                    else:
                        logger.critical(f"{Back.RED}CRITICAL FAILURE TO CLOSE POSITION! State still shows {self.current_side}. Manual intervention required!{Style.RESET_ALL}")
                        alert_msg = f"CRITICAL: Failed to CLOSE {self.symbol} {position_side_display} position on signal ({exit_reason})! Manual check needed!"
                        send_sms_alert(alert_msg, self.sms_config)
                        # What to do now? Stop the bot? Keep trying to close?
                        # Stopping might be safest to prevent further issues.
                        # self.is_running = False # Uncomment to stop bot on critical close failure
                        return True # Indicate exit was attempted, even if critically failed

            else:
                # logger.debug("No exit condition met.")
                return False # No exit triggered

        except Exception as e:
            logger.error(f"Error checking or handling exit conditions: {e}", exc_info=True)
            return False # Indicate failure in the exit logic itself

    def _handle_entry(self, signal: Literal['long', 'short'], df_ind: pd.DataFrame, current_price: Decimal) -> bool:
        """
        Handles the logic for entering a new position based on a signal.
        Includes calculating SL/TP, position size, and placing the entry order
        (potentially with attached SL/TP). Confirms entry via state update.

        Args:
            signal: The entry signal ('long' or 'short').
            df_ind: DataFrame with calculated indicators.
            current_price: The current market price (used for approximate calculations).

        Returns:
            True if entry process was successfully initiated and confirmed.
            False if entry failed at any step (calculation, placement, confirmation).
        """
        if self.current_side != POS_NONE:
            logger.debug(f"Ignoring {signal} entry signal: Already in a {self.current_side} position.")
            return False
        if df_ind is None or df_ind.empty or current_price <= 0:
             logger.warning("Cannot handle entry: Missing indicators or invalid current price.")
             return False
        if not self.price_tick or not self.qty_step or not self.min_qty: # Ensure market details are loaded
             logger.error("Cannot enter: Missing critical market details (price_tick/qty_step/min_qty).")
             return False

        signal_display = signal.upper()
        logger.info(f"{Fore.BLUE}Processing {signal_display} entry signal near price {format_price(self.symbol, current_price, self.price_tick)}...{Style.RESET_ALL}")

        # 1. Calculate SL/TP based on the signal and *current* price (approximation)
        # Actual entry price might differ slightly due to market order slippage.
        sl_price, tp_price = self._calculate_sl_tp(df_ind, signal, current_price)
        if sl_price is None:
            logger.error(f"Cannot enter {signal_display}: Failed to calculate a valid Stop Loss price. Aborting entry.")
            return False
        # TP is optional, proceed even if tp_price is None

        # 2. Calculate Position Size based on risk, SL, and balance
        position_size = self._calculate_position_size(current_price, sl_price)
        if position_size is None or position_size <= Decimal("0"):
            logger.error(f"Cannot enter {signal_display}: Failed to calculate a valid position size (Check balance, risk settings, min qty). Aborting entry.")
            return False

        # 3. Format quantities and prices for Pybit API (MUST be strings)
        entry_qty_str = self._format_qty(position_size)
        sl_price_str = self._format_price_str(sl_price)
        tp_price_str = self._format_price_str(tp_price) if tp_price is not None else None

        # Validation after formatting
        if not entry_qty_str or not sl_price_str:
             logger.error("Failed to format entry quantity or SL price to string. Aborting entry.")
             return False
        if tp_price is not None and not tp_price_str:
             logger.error("Failed to format TP price to string, but TP was calculated. Aborting entry.")
             return False


        # 4. Prepare and Place Entry Order (Market Order)
        entry_side_str = SIDE_BUY if signal == POS_LONG else SIDE_SELL
        order_link_id = f"{signal[:1]}_{self.symbol.replace('/','')}_{int(time.time()*1000)}"[-36:] # Unique ID for tracking

        entry_params: Dict[str, Any] = {
            "category": self.category,
            "symbol": self.symbol,
            "side": entry_side_str,
            "orderType": ORDER_TYPE_MARKET,
            "qty": entry_qty_str,
            "orderLinkId": order_link_id,
            # Market orders usually don't need TimeInForce, but IOC can be specified sometimes
            # "timeInForce": TIME_IN_FORCE_IOC
        }

        # --- ATOMIC SL/TP PLACEMENT (if configured) ---
        if self.attach_sl_tp_to_entry:
            logger.info("Attempting atomic entry with attached SL/TP parameters...")
            if sl_price_str:
                entry_params['stopLoss'] = sl_price_str
                entry_params['slTriggerBy'] = self.sl_trigger_by
                # If SL type is Limit, we might need slLimitPrice. Bybit's `place_order`
                # with `stopLoss` might imply a Market stop, or need `slOrderType`. Check docs!
                # Assuming Market stop for simplicity if `slOrderType` isn't directly settable here.
                # If sl_order_type is Limit, atomic placement might require a different API call or structure.
                if self.sl_order_type == ORDER_TYPE_LIMIT:
                    logger.warning(f"Atomic SL/TP with SL Order Type '{ORDER_TYPE_LIMIT}' might require specific handling (e.g., 'slLimitPrice'). Check Bybit API docs for placing conditional limit stops with entry.")
                    # Add slLimitPrice if required by API for atomic limit stops
                    # entry_params['slLimitPrice'] = sl_price_str # Example: set limit same as trigger
                # Let's explicitly add slOrderType if API supports it in place_order
                entry_params['slOrderType'] = self.sl_order_type # Add configured SL type

            if tp_price_str:
                entry_params['takeProfit'] = tp_price_str
                entry_params['tpTriggerBy'] = self.tp_trigger_by
                # TP is typically a Limit order triggered by price
                entry_params['tpOrderType'] = ORDER_TYPE_LIMIT # Explicitly set TP type

            # Note: The exact parameters for attaching SL/TP (especially conditional limit stops)
            # to a market entry order can be complex and vary. TEST THOROUGHLY ON TESTNET.
            # The `place_order` endpoint might simplify this, or it might require separate `place_conditional_order`.
            # We are proceeding assuming `place_order` handles these parameters as intended.

        # --- Place the Order ---
        entry_order_result = self._place_order(entry_params)

        if not entry_order_result or not entry_order_result.get('orderId'):
             logger.error(f"{Back.RED}{Fore.WHITE}❌ Entry Market Order placement failed for {signal_display}.{Style.RESET_ALL}")
             # If atomic placement was attempted, SL/TP were also not placed.
             return False # Entry failed

        entry_order_id = entry_order_result['orderId']
        logger.info(f"Entry Market Order ({format_order_id(entry_order_id)}) placed for {signal_display}. Waiting briefly for state propagation...")

        # 5. Confirm Entry State (Crucial Step for Robustness)
        # Wait longer to give Bybit's system time to update position state after market order fill.
        # Adjust delay based on observed behavior on testnet/mainnet.
        confirmation_delay = self.app_config.api_config.api_rate_limit_delay * 4 # e.g., 4 * 0.1 = 0.4s, maybe needs more
        logger.debug(f"Waiting {confirmation_delay:.2f}s before confirming position state...")
        time.sleep(confirmation_delay)

        logger.info("Attempting to confirm position state after entry...")
        if not self._update_state():
             logger.error(f"{Back.YELLOW}Failed to fetch updated state after placing entry order ({format_order_id(entry_order_id)}).{Style.RESET_ALL} Cannot confirm entry details. Manual check advised!")
             # This is a risky situation: order placed, but state unknown.
             # Possibilities: Order rejected, order filled but state fetch failed, network issue.
             # What to do?
             # - Could try cancelling the order ID just placed, but it might have filled.
             # - Could stop the bot.
             # - Could proceed hoping the next iteration fixes it (risky).
             logger.warning("Proceeding cautiously. State will be re-checked next iteration.")
             # Consider sending an alert here.
             send_sms_alert(f"ALERT: Failed state update after {self.symbol} {signal_display} entry attempt ({format_order_id(entry_order_id)}). Manual check!", self.sms_config)
             return False # Treat as entry failure for this iteration

        # --- Verify State Reflects Intended Entry ---
        expected_side = signal # 'long' or 'short'
        if self.current_side != expected_side:
            # Check if maybe the order was rejected or filled with zero quantity
            if self.current_side == POS_NONE:
                 logger.error(f"Entry order ({format_order_id(entry_order_id)}) placed, but state update shows NO position. Order likely rejected or zero-filled. Check Bybit Order History.")
            else: # Position exists but is the wrong side (shouldn't happen in One-Way mode unless state was wrong before entry)
                 logger.critical(f"{Back.RED}CRITICAL STATE MISMATCH! Entry order ({format_order_id(entry_order_id)}) placed for {expected_side}, but state update shows position is now '{self.current_side}'. Manual intervention required!{Style.RESET_ALL}")
                 send_sms_alert(f"CRITICAL: State mismatch after {self.symbol} {expected_side} entry! Position shows {self.current_side}. Manual check!", self.sms_config)
            return False # Entry confirmation failed

        # --- Check Filled Quantity and Price ---
        # Compare actual filled quantity (self.current_qty from _update_state) with intended size
        # Allow a small tolerance (e.g., 1%) for minor discrepancies due to fees/slippage if needed, but market orders should fill fully usually.
        # Let's check if it's significantly less than ordered (e.g., < 90%)
        if self.current_qty < position_size * Decimal("0.9"): # Significant partial fill or error?
            logger.warning(f"Potential partial fill or state discrepancy detected for entry {format_order_id(entry_order_id)}. Ordered: {position_size}, Actual Position Qty: {self.current_qty}. Strategy will proceed with actual quantity.")
            # Adjust internal variables if downstream logic depends on the *intended* size, though using self.current_qty is usually best.
            # position_size = self.current_qty # Use actual filled size going forward
            entry_qty_str = self._format_qty(self.current_qty) # Re-format actual quantity if needed later
            if not entry_qty_str:
                 logger.error("Failed to format the *actual* filled quantity. State tracking issue.")
                 return False # Critical formatting error
        else:
            logger.debug(f"Filled quantity {self.current_qty} matches ordered quantity {position_size} closely.")


        actual_entry_price = self.entry_price # From the reliable _update_state
        if actual_entry_price is None:
             logger.error("Position confirmed, but failed to get the average entry price from state update. Proceeding without exact entry price logged.")
             # This might affect PnL calculations if done locally, but SL/TP based on initial calculation should still work.

        actual_qty_display = format_amount(self.symbol, self.current_qty, self.qty_step)
        actual_entry_price_display = format_price(self.symbol, actual_entry_price, self.price_tick) if actual_entry_price else "N/A"

        logger.success(f"{Fore.GREEN}✅ Entry Confirmed: {self.current_side.upper()} {actual_qty_display} {self.base_coin} @ ~{actual_entry_price_display}{Style.RESET_ALL}")

        # Send Alert on successful confirmed entry
        alert_msg = f"[{self.symbol.split('/')[0]}] ENTERED {self.current_side.upper()} {actual_qty_display} @ ~{actual_entry_price_display}"
        send_sms_alert(alert_msg, self.sms_config)


        # 6. Place SL/TP Orders SEPARATELY (if not attached atomically)
        if not self.attach_sl_tp_to_entry:
            logger.info("Placing Stop Loss and Take Profit orders separately...")
            # Crucial: Use the *actual* entry price from the state update for potentially more accurate SL/TP placement.
            # Re-calculate SL/TP based on the confirmed entry price.
            if actual_entry_price:
                logger.debug(f"Re-calculating SL/TP based on actual entry price: {actual_entry_price_display}")
                sl_price_final, tp_price_final = self._calculate_sl_tp(df_ind, signal, actual_entry_price)
                if sl_price_final is None:
                    # CRITICAL: Failed to calculate SL *after* position is already open!
                    logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to calculate FINAL SL price after entry confirmation. POSITION IS OPEN WITHOUT STOP LOSS! Manual intervention required!{Style.RESET_ALL}")
                    send_sms_alert(f"CRITICAL: Failed place SL for {self.symbol} {self.current_side.upper()} pos! Manual SL required!", self.sms_config)
                    # Options:
                    # 1. Try to close the position immediately.
                    # self._close_position_immediately("Failed Final SL Calculation") # Implement this emergency close function if desired
                    # 2. Let the bot continue but without SL (very risky).
                    # 3. Stop the bot.
                    # self.is_running = False
                    return True # Entry occurred, but subsequent critical failure. Allow loop to continue/stop based on chosen handling.

                # Use the final calculated prices
                sl_price_str_final = self._format_price_str(sl_price_final)
                tp_price_str_final = self._format_price_str(tp_price_final) if tp_price_final is not None else None
            else:
                # If actual entry price wasn't available, fall back to initially calculated SL/TP
                logger.warning("Actual entry price not available, using initially calculated SL/TP for separate orders.")
                sl_price_str_final = sl_price_str
                tp_price_str_final = tp_price_str

            # Format the actual filled quantity string again (might have changed if partial fill)
            position_qty_str_final = self._format_qty(self.current_qty)
            if not position_qty_str_final or not sl_price_str_final:
                 logger.critical(f"{Back.RED}CRITICAL: Failed to format final quantity or SL price for separate orders. POSITION OPEN WITHOUT SL/TP! Manual intervention!{Style.RESET_ALL}")
                 send_sms_alert(f"CRITICAL: Format error placing SL/TP for {self.symbol} {self.current_side.upper()}! Manual check!", self.sms_config)
                 return True # Entry occurred, critical failure follows

            # Place the separate orders using the final prices and actual filled quantity string
            sl_order_id_placed, tp_order_id_placed = self._place_separate_sl_tp_orders(
                sl_price_str=sl_price_str_final,
                tp_price_str=tp_price_str_final,
                position_qty_str=position_qty_str_final # Use actual filled qty string
            )

            # Track the IDs if placement was successful
            if sl_order_id_placed:
                self.sl_order_id = sl_order_id_placed # Track the successfully placed SL ID
                logger.info(f"Separate SL order ({format_order_id(self.sl_order_id)}) placed.")
            else:
                # CRITICAL: Failed to place the separate SL order *after* entry.
                logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to place separate SL order after entry confirmation. POSITION IS OPEN WITHOUT STOP LOSS! Manual intervention required!{Style.RESET_ALL}")
                send_sms_alert(f"CRITICAL: Failed place separate SL for {self.symbol} {self.current_side.upper()} pos! Manual SL required!", self.sms_config)
                # Consider emergency close or bot stop here as well.

            if tp_order_id_placed:
                self.tp_order_id = tp_order_id_placed # Track the successfully placed TP ID
                logger.info(f"Separate TP order ({format_order_id(self.tp_order_id)}) placed.")
            elif tp_price_final is not None: # Log warning only if TP was intended but failed
                 logger.warning("Failed to place separate TP order after entry.")

        else:
             # SL/TP were attached atomically, clear any potentially stale tracked IDs from previous separate orders
             self.sl_order_id = None
             self.tp_order_id = None
             logger.info("SL/TP were attached to the entry order (atomic placement). No separate orders placed.")


        return True # Indicate entry process completed (potentially with warnings/failures on separate SL/TP)

    def _place_separate_sl_tp_orders(self, sl_price_str: str, tp_price_str: Optional[str], position_qty_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Places separate SL (Stop Market/Limit) and TP (Limit) orders after an
        entry has been confirmed. Uses conditional orders (triggerPrice).

        Args:
            sl_price_str: The formatted stop loss trigger price string.
            tp_price_str: The formatted take profit limit price string (or None).
            position_qty_str: The formatted quantity string (should match position size).

        Returns:
            A tuple containing:
            - The Order ID of the placed SL order (or None if failed).
            - The Order ID of the placed TP order (or None if failed or not placed).
        """
        sl_order_id: Optional[str] = None
        tp_order_id: Optional[str] = None

        if self.current_side == POS_NONE: # Should not happen if called correctly after entry confirmation
             logger.error("Cannot place separate SL/TP: Not currently in a position according to state.")
             return None, None
        if not self.session:
             logger.error("Cannot place separate SL/TP: Session not initialized.")
             return None, None

        # Determine the side for SL/TP orders (opposite of position side)
        exit_side = SIDE_SELL if self.current_side == POS_LONG else SIDE_BUY

        # Unique Link IDs for SL/TP
        sl_link_id = f"sl_{self.symbol.replace('/','')}_{int(time.time()*1000)}"[-36:]
        tp_link_id = f"tp_{self.symbol.replace('/','')}_{int(time.time()*1000)}"[-36:]

        # --- Place Stop Loss Order (Conditional Order) ---
        logger.info(f"Placing Separate Stop Loss ({self.sl_order_type}) order: {exit_side} {position_qty_str} @ Trigger {sl_price_str}")
        sl_params: Dict[str, Any] = {
            "category": self.category,
            "symbol": self.symbol,
            "side": exit_side,
            "orderType": self.sl_order_type, # Market or Limit Stop
            "qty": position_qty_str,
            "triggerPrice": sl_price_str,
            "triggerBy": self.sl_trigger_by,
            "reduceOnly": True,
            "timeInForce": TIME_IN_FORCE_GTC, # Stops are typically GTC
            "orderLinkId": sl_link_id,
            # Conditional Order Specific Parameters (check Bybit API for exact names)
            # 'stopOrderType' or similar might be needed if orderType is just Market/Limit
            # Let's assume setting orderType to Market/Limit and providing triggerPrice is sufficient for conditional orders.
            # Test this assumption carefully.
        }

        # If SL is Limit type, the 'price' parameter (limit price) is needed for the Limit order placed *after* trigger.
        if self.sl_order_type == ORDER_TYPE_LIMIT:
             # Set limit price equal to trigger price? Or slightly worse for guaranteed fill?
             # Setting it equal to trigger is common, but risks non-fill if price gaps past trigger.
             # Setting slightly worse (e.g., trigger +/- tick) might increase fill chance but worsen slippage.
             sl_limit_price = sl_price_str # Simplest approach: Limit price = Trigger price
             sl_params['price'] = sl_limit_price
             logger.info(f"SL is Limit type, setting limit price for triggered order: {sl_limit_price}")

        sl_order_result = self._place_order(sl_params)
        if sl_order_result:
            sl_order_id = sl_order_result.get('orderId')
            if not sl_order_id:
                 logger.error(f"Separate SL order placement returned OK but missing OrderID. Result: {sl_order_result}")


        # --- Place Take Profit Order (Conditional Limit Order) ---
        if tp_price_str is not None:
            logger.info(f"Placing Separate Take Profit (Limit) order: {exit_side} {position_qty_str} @ Limit {tp_price_str}")
            # TP is typically a standard Limit order, sometimes placed conditionally based on price trigger
            # Let's try placing a standard Limit order first. If it needs to be conditional, adjust params.
            tp_params: Dict[str, Any] = {
                "category": self.category,
                "symbol": self.symbol,
                "side": exit_side,
                "orderType": ORDER_TYPE_LIMIT,
                "qty": position_qty_str,
                "price": tp_price_str, # The limit price at which to take profit
                "reduceOnly": True,
                "timeInForce": TIME_IN_FORCE_GTC, # TPs are usually GTC
                "orderLinkId": tp_link_id,
                # Optional: If TP also needs to be conditional (e.g., trigger based on MarkPrice, place Limit order)
                # "triggerPrice": tp_price_str, # Trigger price (could be same as limit price)
                # "triggerBy": self.tp_trigger_by,
                # "tpOrderType": ORDER_TYPE_LIMIT # May not be needed if orderType=Limit
            }
            tp_order_result = self._place_order(tp_params)
            if tp_order_result:
                tp_order_id = tp_order_result.get('orderId')
                if not tp_order_id:
                    logger.error(f"Separate TP order placement returned OK but missing OrderID. Result: {tp_order_result}")

        return sl_order_id, tp_order_id

    # --- Main Loop and Control ---

    def run_iteration(self):
        """
        Executes a single iteration of the strategy logic:
        1. Update State
        2. Fetch Data
        3. Calculate Indicators
        4. Check Exit Conditions (if in position)
        5. Check Entry Conditions (if not in position)
        """
        if not self.is_initialized or not self.session:
            logger.error("Strategy not initialized or session lost. Cannot run iteration.")
            self.is_running = False # Stop the loop if fundamental setup is broken
            return

        iteration_start_time = time.monotonic()
        current_time_utc = pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        logger.info(f"{Fore.MAGENTA}--- New Strategy Iteration ({current_time_utc}) ---{Style.RESET_ALL}")

        try:
            # 1. Update State (Position, Balance)
            if not self._update_state():
                logger.warning("Failed to update state successfully. Skipping this iteration's logic.")
                # Consider adding a short delay here if state updates fail frequently, might indicate API issues
                # time.sleep(5)
                return # Skip rest of iteration if state is uncertain

            # 2. Fetch Market Data (OHLCV, Ticker)
            ohlcv_df, current_price = self._fetch_data()
            # Check if *essential* data is missing
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning("Failed to fetch valid OHLCV data. Skipping indicator calculation and logic.")
                return # Cannot proceed without candles
            if current_price is None:
                logger.warning("Failed to fetch current ticker price. Entry calculations might be affected.")
                # Strategy might still proceed with exits or use last close if current_price is None

            # 3. Calculate Indicators
            df_with_indicators = self._calculate_indicators(ohlcv_df)
            if df_with_indicators is None:
                logger.warning("Failed indicator calculation or validation. Skipping trading logic.")
                return # Cannot proceed without indicators

            # --- Core Trading Logic ---
            # 4. Check Exit Conditions (only if currently in a position)
            exit_triggered = False
            if self.current_side != POS_NONE:
                exit_triggered = self._handle_exit(df_with_indicators)
                if exit_triggered:
                     logger.info("Exit handled in this iteration.")
                     # No need to check for entry immediately after an exit attempt
                     # Allow next iteration's state update to confirm exit fully

            # 5. Generate & Handle Entry Signals (only if not in position and no exit was triggered)
            if self.current_side == POS_NONE and not exit_triggered:
                if current_price is None:
                     logger.warning("Cannot attempt entry: Current price is missing.")
                else:
                     entry_signal = self._generate_signals(df_with_indicators)
                     if entry_signal:
                         entry_handled = self._handle_entry(entry_signal, df_with_indicators, current_price)
                         if entry_handled:
                              logger.info("Entry handled in this iteration.")
                         else:
                              logger.info("Entry signal generated but entry process failed or was aborted.")
                     else:
                          logger.info("No entry signal generated this iteration.")
            elif self.current_side != POS_NONE:
                 # Still in position, no exit triggered. Log monitoring status.
                 pos_qty_display = format_amount(self.symbol, self.current_qty, self.qty_step)
                 entry_price_display = format_price(self.symbol, self.entry_price, self.price_tick) if self.entry_price else "N/A"
                 # Optional: Calculate and log Unrealized PnL if desired (requires fetching position data again or using ticker)
                 pnl_str = "" # Placeholder
                 logger.info(f"Monitoring {self.current_side.upper()} position ({pos_qty_display} @ {entry_price_display}). {pnl_str}Waiting for exit signal or SL/TP.")
                 # Add Trailing Stop Logic here if implemented

        except Exception as e:
            # Catch unexpected errors within the iteration logic
            logger.critical(f"{Back.RED}{Fore.WHITE}💥 Critical unexpected error during strategy iteration: {e}{Style.RESET_ALL}", exc_info=True)
            # Send alert for critical loop errors
            alert_msg = f"CRITICAL Error in {self.symbol} strategy loop: {type(e).__name__}"
            send_sms_alert(alert_msg, self.sms_config)
            # Depending on severity, might want to stop the bot
            # self.is_running = False

        finally:
            # Log iteration duration
            iteration_end_time = time.monotonic()
            elapsed = iteration_end_time - iteration_start_time
            logger.info(f"{Fore.MAGENTA}--- Iteration Complete (Took {elapsed:.3f}s) ---{Style.RESET_ALL}")

    def start(self):
        """Initializes the strategy and starts the main execution loop."""
        logger.info("Initiating strategy startup sequence...")
        if not self._initialize():
            logger.critical(f"{Back.RED}Strategy initialization failed. Cannot start the arcane loop.{Style.RESET_ALL}")
            # Send alert on initialization failure
            send_sms_alert(f"CRITICAL: {self.symbol} strategy FAILED TO INITIALIZE!", self.sms_config)
            return # Do not start loop if initialization failed

        self.is_running = True
        loop_delay = self.strategy_config.loop_delay_seconds
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}🚀 Strategy ritual commenced for {self.symbol} ({self.timeframe}). Loop delay: {loop_delay}s{Style.RESET_ALL}")
        send_sms_alert(f"INFO: {self.symbol} strategy started. TF:{self.timeframe}, LoopDelay:{loop_delay}s", self.sms_config)

        # --- Main Loop ---
        while self.is_running:
            loop_start_time = time.monotonic()

            # --- Run Single Iteration ---
            self.run_iteration()

            # --- Calculate Sleep Time ---
            loop_end_time = time.monotonic()
            elapsed = loop_end_time - loop_start_time
            sleep_duration = max(0, loop_delay - elapsed)

            if not self.is_running: # Check if stop was requested during iteration
                 logger.info("Stop signal received during iteration. Exiting loop.")
                 break

            if sleep_duration > 0:
                 logger.debug(f"Sleeping for {sleep_duration:.2f}s...")
                 time.sleep(sleep_duration)
            else:
                 logger.warning(f"Iteration took longer ({elapsed:.2f}s) than loop delay ({loop_delay}s). Running next iteration immediately.")

        # --- Loop Ended ---
        logger.warning(f"{Fore.YELLOW}Strategy loop has been terminated.{Style.RESET_ALL}")
        self.stop() # Ensure cleanup is called if loop exits unexpectedly or normally

    def stop(self):
        """Stops the strategy loop and performs cleanup actions."""
        if not self.is_running and self.session is None: # Prevent multiple stop logs if already stopped
             logger.debug("Stop called but strategy already seems stopped.")
             return

        logger.warning(f"{Fore.YELLOW}--- Initiating Strategy Shutdown ---{Style.RESET_ALL}")
        run_state_before_stop = self.is_running
        self.is_running = False # Signal the main loop to stop if it's still running

        # --- Final Cleanup Actions ---
        logger.info("Attempting final order cleanup...")
        if self.session and self.category and self.symbol: # Check if session is still valid
            if not self._cancel_all_open_orders("Strategy Stop"):
                 logger.warning("Final order cancellation encountered issues. Manual check of Bybit UI advised.")
        else:
             logger.warning("Skipping final order cancellation: Session/category/symbol not available.")

        # --- Check Final Position State ---
        # Update state one last time *if possible* to log final position
        final_position_check_possible = self.session and self.category and self.symbol
        if final_position_check_possible:
             logger.info("Performing final position state check...")
             self._update_state() # Attempt to get final state

        if self.current_side != POS_NONE:
            pos_qty_display = format_amount(self.symbol, self.current_qty, self.qty_step)
            entry_price_display = format_price(self.symbol, self.entry_price, self.price_tick) if self.entry_price else "N/A"
            warning_msg = f"Strategy stopped with an OPEN {self.current_side.upper()} position for {self.symbol} ({pos_qty_display} @ {entry_price_display}). Manual management may be required."
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}{warning_msg}{Style.RESET_ALL}")
            # Send alert about open position on stop
            send_sms_alert(f"ALERT: {self.symbol} strategy stopped with OPEN {self.current_side.upper()} position!", self.sms_config)
            # Consider adding an option in config to auto-close position on stop:
            # if self.strategy_config.close_on_stop and final_position_check_possible:
            #     logger.warning("Attempting to close open position due to 'close_on_stop' config...")
            #     self._close_position_immediately("Strategy Stop with CloseOnStop")
        else:
             logger.info("Strategy stopped while flat (no open position detected).")

        # --- Release Resources ---
        self._safe_close_session()

        if run_state_before_stop: # Only send stop alert if it was actually running
            send_sms_alert(f"INFO: {self.symbol} strategy stopped.", self.sms_config)
        logger.info(f"{Fore.CYAN}--- Strategy shutdown complete ---{Style.RESET_ALL}")


    def _safe_close_session(self):
        """Safely handles session cleanup. (Pybit HTTP doesn't require explicit close)."""
        if self.session:
            logger.info("Cleaning up Pybit HTTP session...")
            # No explicit close method for pybit's HTTP session, just dereference
            self.session = None
            logger.info("Pybit session cleared.")

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"{Fore.CYAN}{Style.BRIGHT}--- Bybit EVT Strategy Script (Pybit Enhanced) ---{Style.RESET_ALL}")
    print(f"Python Version: {sys.version}")
    print(f"Pandas Version: {pd.__version__}")
    # print(f"Pybit Version: {pybit.__version__}") # Requires pybit to have __version__

    app_config: Optional[AppConfig] = None
    strategy: Optional[EhlersStrategyPybitEnhanced] = None

    try:
        # --- Load Configuration ---
        print(f"{Fore.BLUE}Summoning configuration spirits...{Style.RESET_ALL}")
        app_config = load_config()
        if not app_config: # load_config should exit or raise on failure, but double-check
            print(f"{Back.RED}{Fore.WHITE}FATAL: Configuration loading failed.{Style.RESET_ALL}", file=sys.stderr)
            sys.exit(1)
        print(f"{Fore.GREEN}Configuration spirits summoned successfully.{Style.RESET_ALL}")

        # --- Setup Logging ---
        print(f"{Fore.BLUE}Awakening Neon Logger...{Style.RESET_ALL}")
        log_conf = app_config.logging_config
        # Ensure logger is configured using the settings from the loaded config
        logger = setup_logger( # This should reconfigure the root logger or the named logger
            logger_name=log_conf.logger_name, # Use the name defined in config
            log_file=log_conf.log_file,
            console_level_str=log_conf.console_level_str,
            file_level_str=log_conf.file_level_str,
            log_rotation_bytes=log_conf.log_rotation_bytes,
            log_backup_count=log_conf.log_backup_count,
            third_party_log_level_str=log_conf.third_party_log_level_str
        )
        # Re-get the logger by name after setup to ensure we have the configured one
        logger = logging.getLogger(log_conf.logger_name)
        logger.info(f"{Fore.MAGENTA}--- Neon Logger Awakened and Configured ---{Style.RESET_ALL}")
        logger.info(f"Logging to Console Level: {log_conf.console_level_str}, File Level: {log_conf.file_level_str}")
        logger.info(f"Log File: {log_conf.log_file}")

        # Log key initial config details
        logger.info(f"Using config: Testnet={app_config.api_config.testnet_mode}, Symbol={app_config.api_config.symbol}, Timeframe={app_config.strategy_config.timeframe}")

        # --- Instantiate Strategy ---
        logger.info("Creating enhanced strategy instance...")
        strategy = EhlersStrategyPybitEnhanced(app_config) # Instantiate the ENHANCED version
        logger.info("Strategy instance created.")

        # --- Start Strategy ---
        logger.info("Initiating strategy start sequence...")
        strategy.start() # This enters the main loop

    except KeyboardInterrupt:
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}>>> Manual interruption detected (Ctrl+C)! Initiating graceful shutdown...{Style.RESET_ALL}")
        # Strategy stop is handled in the finally block

    except SystemExit as e:
         # Raised by sys.exit() calls earlier (e.g., missing imports, config load fail)
         logger.info(f"System exiting with code {e.code}.")
         # No further action needed here, finally block will run if necessary

    except Exception as e:
        # Catch any unexpected critical errors during setup or the main loop start
        logger.critical(f"{Back.RED}{Fore.WHITE}💥 UNHANDLED CRITICAL ERROR in main execution block: {e}{Style.RESET_ALL}", exc_info=True)
        # Attempt to send a final alert if config is available
        if app_config and app_config.sms_config:
            try:
                error_type = type(e).__name__
                send_sms_alert(f"CRITICAL FAILURE: Unhandled error in {app_config.api_config.symbol} strategy main block: {error_type}! Bot stopping.", app_config.sms_config)
            except Exception as alert_e:
                logger.error(f"Failed to send critical error SMS alert: {alert_e}")
        # Exit with error code
        sys.exit(1)

    finally:
        # --- Graceful Shutdown ---
        logger.info("Entering final cleanup phase...")
        if strategy and strategy.is_running: # If strategy object exists and might be running
             logger.info("Requesting strategy stop...")
             strategy.stop()
        elif strategy and not strategy.is_running and strategy.session is not None:
             # If strategy exists but loop wasn't running (e.g., init failed but session created)
             logger.info("Strategy loop was not active, ensuring resources are released...")
             strategy.stop() # Call stop anyway for cleanup like session closing

        logger.info(f"{Fore.CYAN}{Style.BRIGHT}--- Strategy script enchantment fades. Returning to the digital ether... ---{Style.RESET_ALL}")
        # Ensure all handlers are closed (especially file handlers)
        logging.shutdown()

# --- END OF ENHANCED SPELL ---
```

**Summary of Key Enhancements Incorporated:**

1.  **Atomic Order Placement (`attach_sl_tp_to_entry`):**
    *   The `_handle_entry` method now checks `self.attach_sl_tp_to_entry`.
    *   If `True`, it adds `stopLoss`, `takeProfit`, `slTriggerBy`, `tpTriggerBy`, `slOrderType`, and `tpOrderType` parameters directly to the main `place_order` call for the entry market order.
    *   If `False`, it places the entry order first, confirms the entry, and then calls `_place_separate_sl_tp_orders` to place conditional SL and TP orders, tracking their IDs (`self.sl_order_id`, `self.tp_order_id`).

2.  **Enhanced Configuration (SL/TP Triggers & Types):**
    *   New attributes `sl_trigger_by`, `tp_trigger_by`, `sl_order_type` are initialized in `__init__` from `StrategyConfig` (with defaults).
    *   These attributes are used in both atomic (`_handle_entry`) and separate (`_place_separate_sl_tp_orders`) placement logic when setting trigger/type parameters for SL/TP orders.
    *   Basic validation added in `__init__` to ensure configured values are valid Bybit strings.

3.  **Robust State Management & Market Info:**
    *   `_fetch_and_set_market_info`: More robust category detection attempts; critically validates that *all* essential details (min\_qty, steps, ticks, coins, multiplier) were successfully extracted and are valid Decimals/strings. Initialization fails if critical info is missing.
    *   `_update_state`: Handles cases where the position list is empty or size is zero more explicitly. Clears tracked SL/TP IDs only when confirmed flat. Logs balance but fetches it separately when needed for calculations. Returns `False` if position fetch fails.
    *   `_handle_entry`: Includes a crucial state confirmation step after placing the entry order:
        *   Waits using `time.sleep` (increased delay).
        *   Calls `_update_state`.
        *   Explicitly checks if `self.current_side`, `self.current_qty`, and `self.entry_price` match expectations. Logs errors/warnings and returns `False` if confirmation fails or shows discrepancies (like zero fill or wrong side). Sends alerts on critical mismatches or state update failures post-entry.
    *   `_handle_exit`: Attempts state re-check if the close order placement fails, providing a chance to confirm closure despite API errors.

4.  **Improved Error Handling:**
    *   `_place_order`: More specific logging for different failure `retCode`s (Insufficient Balance, Qty Too Small/Invalid, Price Invalid, Rate Limit, ReduceOnly errors). Sends critical SMS alerts for insufficient balance. Logs full parameters on failure. Handles Pybit exceptions (`InvalidRequestError`, `FailedRequestError`) and timestamp errors more clearly.
    *   `_cancel_single_order`, `_cancel_all_open_orders`: Handle "order not found / already closed" return codes gracefully, treating them as successful cancellations in context. Log other failures clearly.
    *   `_calculate_sl_tp`, `_calculate_position_size`: Added more validation for inputs (prices > 0, multipliers > 0, valid ATR) and intermediate calculations (denominators > 0). Return `None` early on invalid inputs.
    *   `_handle_entry`, `_handle_exit`: Include critical logging and SMS alerts for failures in placing essential orders (like entry close or post-entry SL). Handle edge cases like failing to calculate SL *after* entry.
    *   Main execution block (`if __name__ == "__main__":`) has improved exception handling for setup errors and unhandled loop errors, including final alerts.

5.  **Code Clarity & Structure:**
    *   **Constants:** Defined extensive constants for Pybit API strings (Sides, Order Types, TimeInForce, Triggers, Position Indices), internal states (POS\_LONG/SHORT/NONE), Account Types, and common Return Codes.
    *   **Comments & Docstrings:** Added more detailed docstrings to methods explaining their purpose, arguments, and returns. Added inline comments for complex logic sections.
    *   **Logging:** Significantly enhanced logging across all methods with different levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). Used Colorama more effectively to highlight important events (Success=Green, Warning=Yellow, Error/Critical=Red). Added timestamps and iteration timings.
    *   **Formatting Helpers:** `_format_qty` and `_format_price_str` ensure correct string formatting based on market precision rules, handling potential `Decimal` conversion/quantization errors.
    *   **Variable Names:** Used descriptive variable names (e.g., `entry_price_approx`, `sl_price_adjusted`, `position_qty_str_final`).
    *   **Type Hinting:** Improved type hinting for better readability and static analysis.
    *   **Readability:** Refactored some logic slightly for better flow (e.g., validation checks at the start of methods).

6.  **Efficiency:**
    *   **Consolidated Balance Fetch:** Introduced `_get_available_balance` helper, which is called explicitly by `_calculate_position_size` when the value is needed for calculation. `_update_state` calls it mainly for logging purposes now. This avoids redundant balance calls if only the position is needed.
    *   Other minor improvements like calculating constants once where possible.

**Before Running:**

1.  **Update `config_models.py`:** Ensure your `StrategyConfig` model includes fields for `attach_sl_tp_to_entry: bool`, `sl_trigger_by: str`, `tp_trigger_by: str`, and `sl_order_type: str`. Provide defaults if desired.
2.  **Install Dependencies:** Make sure all required libraries are installed (`pip install --upgrade pybit pandas colorama python-dotenv pandas-ta requests`).
3.  **Review Configuration:** Double-check all settings in your `.env` file and `config.yaml`/`config.json`. Pay attention to the new SL/TP settings.
4.  **TESTNET FIRST:** **Thoroughly test this enhanced script on the Bybit Testnet** to verify order placement (especially atomic vs. separate SL/TP), state confirmation, error handling, and overall logic before deploying with real funds. Observe the logs carefully during testing.
