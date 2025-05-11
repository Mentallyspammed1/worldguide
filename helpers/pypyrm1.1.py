# -*- coding: utf-8 -*-
"""
ehlers_volumetric_strategy_pybit_enhanced.py

An enhanced version of the Ehlers Volumetric Trend strategy implemented
for Bybit using the pybit library. Incorporates atomic order placement,
enhanced configuration, robust state management, improved error handling,
and better code structure.
"""

import sys
import time
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation, Context
from typing import Optional, Dict, Tuple, Any, Literal, List

# --- Third-party Libraries ---
try:
    from pybit.unified_trading import HTTP
    from pybit.exceptions import InvalidRequestError, FailedRequestError

    # Attempt to get pybit version if available
    try:
        from pybit import __version__ as pybit_version
    except ImportError:
        pybit_version = "unknown"
except ImportError:
    # Use basic print before colorama/logger might be available
    print(
        "FATAL: Pybit library not found. Please install it: pip install pybit",
        file=sys.stderr,
    )
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print(
        "FATAL: pandas library not found. Please install it: pip install pandas",
        file=sys.stderr,
    )
    sys.exit(1)
try:
    from dotenv import load_dotenv

    print("Attempting to load environment variables from .env file...")
    dotenv_loaded = load_dotenv()
    print(f".env file {'processed' if dotenv_loaded else 'not found or empty'}.")
except ImportError:
    print(
        "Warning: python-dotenv not found. Cannot load .env file. Ensure environment variables are set manually."
    )

    # Define a dummy function if dotenv is not available
    def load_dotenv():
        pass  # pylint: disable=function-redefined


try:
    import pandas_ta as ta  # Often used alongside pandas for indicators
except ImportError:
    print(
        "Warning: pandas_ta library not found. Some indicator functions might rely on it. Install: pip install pandas_ta"
    )
    ta = None  # Set to None if not found

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
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()  # type: ignore

# --- Import Custom Modules ---
# Use try-except blocks for better error reporting if modules are missing
try:
    from neon_logger import setup_logger
except ImportError as e:
    print(
        f"{Back.RED}{Fore.WHITE}FATAL: Error importing neon_logger: {e}{Style.RESET_ALL}",
        file=sys.stderr,
    )
    print(
        f"{Fore.YELLOW}Ensure 'neon_logger.py' is present and runnable.{Style.RESET_ALL}"
    )
    sys.exit(1)
try:
    import indicators as ind
except ImportError as e:
    print(
        f"{Back.RED}{Fore.WHITE}FATAL: Error importing indicators: {e}{Style.RESET_ALL}",
        file=sys.stderr,
    )
    print(
        f"{Fore.YELLOW}Ensure 'indicators.py' is present and contains 'calculate_all_indicators'.{Style.RESET_ALL}"
    )
    sys.exit(1)
try:
    from bybit_utils import (
        safe_decimal_conversion,
        format_price,
        format_amount,
        format_order_id,
        send_sms_alert,
    )
except ImportError as e:
    print(
        f"{Back.RED}{Fore.WHITE}FATAL: Error importing bybit_utils: {e}{Style.RESET_ALL}",
        file=sys.stderr,
    )
    print(
        f"{Fore.YELLOW}Ensure 'bybit_utils.py' is present and compatible.{Style.RESET_ALL}"
    )
    sys.exit(1)
try:
    # Assuming config_models.py now includes enhanced StrategyConfig options
    from config_models import (
        AppConfig,
        APIConfig,
        StrategyConfig,
        load_config,
        SMSConfig,
        LoggingConfig,
    )
except ImportError as e:
    print(
        f"{Back.RED}{Fore.WHITE}FATAL: Error importing config_models: {e}{Style.RESET_ALL}",
        file=sys.stderr,
    )
    print(
        f"{Fore.YELLOW}Ensure 'config_models.py' is present and defines AppConfig, APIConfig, StrategyConfig, SMSConfig, LoggingConfig, load_config.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}Make sure StrategyConfig includes 'attach_sl_tp_to_entry', 'sl_trigger_by', 'tp_trigger_by', 'sl_order_type', 'close_on_stop'.{Style.RESET_ALL}"
    )
    sys.exit(1)


# --- Logger Placeholder ---
# Will be configured properly in the main execution block
logger: logging.Logger = logging.getLogger(__name__)
# Add a basic handler temporarily in case of early errors before full setup
temp_handler = logging.StreamHandler(sys.stdout)
temp_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
temp_handler.setFormatter(temp_formatter)
logger.addHandler(temp_handler)
logger.setLevel(logging.INFO)  # Set a default level

# --- Constants ---
# Pybit API String Constants (Refer to Bybit documentation for definitive values)
SIDE_BUY = "Buy"
SIDE_SELL = "Sell"
ORDER_TYPE_MARKET = "Market"
ORDER_TYPE_LIMIT = "Limit"
TIME_IN_FORCE_GTC = "GTC"  # GoodTillCancel
TIME_IN_FORCE_IOC = "IOC"  # ImmediateOrCancel
TIME_IN_FORCE_FOK = "FOK"  # FillOrKill
TIME_IN_FORCE_POST_ONLY = "PostOnly"  # Limit orders only
TRIGGER_BY_MARK = "MarkPrice"
TRIGGER_BY_LAST = "LastPrice"
TRIGGER_BY_INDEX = "IndexPrice"
POSITION_IDX_ONE_WAY = 0  # 0 for one-way mode position
POSITION_IDX_HEDGE_BUY = 1  # 1 for hedge mode buy side position
POSITION_IDX_HEDGE_SELL = 2  # 2 for hedge mode sell side position
POSITION_MODE_ONE_WAY = 0  # Alias for clarity
POSITION_MODE_HEDGE = 3  # Corresponds to BothSidePosition mode in API v5
# Custom Position Sides (Internal Representation)
POS_LONG = "long"
POS_SHORT = "short"
POS_NONE = "none"
# Bybit Account Types (Used in get_wallet_balance)
ACCOUNT_TYPE_UNIFIED = "UNIFIED"
ACCOUNT_TYPE_CONTRACT = "CONTRACT"  # For older Inverse/Linear Perpetual if not Unified
ACCOUNT_TYPE_SPOT = "SPOT"
# Common Bybit API Return Codes (Consult official documentation for exhaustive list)
RET_CODE_OK = 0
RET_CODE_PARAMS_ERROR = 10001  # Parameter error
RET_CODE_API_KEY_INVALID = 10003
RET_CODE_SIGN_ERROR = 10004
RET_CODE_TOO_MANY_VISITS = 10006  # Rate limit exceeded
RET_CODE_ORDER_NOT_FOUND = 110001  # Order does not exist
RET_CODE_ORDER_NOT_FOUND_OR_CLOSED = 20001  # Order does not exist or finished (Unified)
RET_CODE_INSUFFICIENT_BALANCE_SPOT = (
    12131  # Spot insufficient balance (example, verify code)
)
RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_1 = (
    110007  # Insufficient available balance (Unified)
)
RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_2 = (
    30031  # Position margin is insufficient (Classic)
)
RET_CODE_QTY_TOO_SMALL = 110017  # Order qty is not greater than the minimum allowed
RET_CODE_QTY_INVALID_PRECISION = 110012  # Order qty decimal precision error
RET_CODE_PRICE_TOO_LOW = 110014  # Order price is lower than the minimum allowed
RET_CODE_PRICE_INVALID_PRECISION = 110013  # Order price decimal precision error
RET_CODE_LEVERAGE_NOT_MODIFIED = 110043  # Leverage not modified
RET_CODE_POSITION_MODE_NOT_MODIFIED = 110048  # Position mode is not modified (Unified)
RET_CODE_REDUCE_ONLY_MARGIN_ERROR = (
    30024  # ReduceOnly order Failed. Position margin is insufficient (Classic)
)
RET_CODE_REDUCE_ONLY_QTY_ERROR = (
    30025  # ReduceOnly order Failed. Order qty is greater than position size (Classic)
)
RET_CODE_REDUCE_ONLY_QTY_ERROR_UNIFIED = 110025  # Reduce only order qty error (Unified)
RET_CODE_ORDER_CANCELLED_OR_REJECTED = (
    110010  # Order has been cancelled or rejected (Unified)
)
RET_CODE_ORDER_FILLED = 110011  # Order has been filled (Unified)
RET_CODE_NO_NEED_TO_SET_MARGIN_MODE = (
    110026  # Cross/isolated margin mode is not modified (Unified)
)

# Set Decimal precision (adjust as needed, 30 should be sufficient for most crypto)
DECIMAL_CONTEXT = Context(prec=30)
getcontext().prec = 30


# --- Enhanced Strategy Class ---
class EhlersStrategyPybitEnhanced:
    """
    Enhanced Ehlers Volumetric Trend strategy using Pybit HTTP API, incorporating
    atomic order placement, improved configuration, state management, and robustness.

    Handles initialization, data fetching, indicator calculation, signal
    generation, order placement (entry, SL, TP), position management,
    and state tracking for Bybit Unified Trading accounts.
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
        self.sms_config: SMSConfig = config.sms_config  # Store SMS config

        self.symbol: str = self.api_config.symbol
        # Ensure timeframe is a string format Bybit expects (e.g., '15', '60', 'D')
        self.timeframe: str = str(self.strategy_config.timeframe)

        self.session: Optional[HTTP] = None  # Pybit HTTP session object
        # Category determined during init, crucial for API calls
        self.category: Optional[Literal["linear", "inverse", "spot"]] = None

        self.is_initialized: bool = False  # Flag indicating successful initialization
        self.is_running: bool = False  # Flag indicating the main loop is active

        # --- Position State Tracking ---
        self.current_side: str = POS_NONE  # 'long', 'short', or 'none'
        self.current_qty: Decimal = DECIMAL_CONTEXT.create_decimal(
            "0.0"
        )  # Current position size
        self.entry_price: Optional[Decimal] = (
            None  # Average entry price of current position
        )
        # Track order IDs IF placing SL/TP separately (not atomically)
        self.sl_order_id: Optional[str] = None  # ID of the active separate SL order
        self.tp_order_id: Optional[str] = None  # ID of the active separate TP order

        # --- Market Details (Fetched during initialization) ---
        self.min_qty: Optional[Decimal] = None  # Minimum order quantity
        self.qty_step: Optional[Decimal] = None  # Quantity step (precision)
        self.price_tick: Optional[Decimal] = None  # Price tick size (precision)
        self.base_coin: Optional[str] = None  # Base currency of the symbol
        self.quote_coin: Optional[str] = None  # Quote currency of the symbol
        self.contract_multiplier: Decimal = DECIMAL_CONTEXT.create_decimal(
            "1.0"
        )  # For value calculation (esp. inverse)

        # --- Enhanced Configurable Options (Defaults if not in AppConfig) ---
        # Use getattr for backward compatibility if config file isn't updated
        self.attach_sl_tp_to_entry: bool = getattr(
            self.strategy_config, "attach_sl_tp_to_entry", True
        )
        self.sl_trigger_by: str = getattr(
            self.strategy_config, "sl_trigger_by", TRIGGER_BY_MARK
        )
        self.tp_trigger_by: str = getattr(
            self.strategy_config, "tp_trigger_by", TRIGGER_BY_MARK
        )  # TP often uses same trigger as SL
        self.sl_order_type: str = getattr(
            self.strategy_config, "sl_order_type", ORDER_TYPE_MARKET
        )  # Market stop is common
        self.close_on_stop: bool = getattr(
            self.strategy_config, "close_on_stop", False
        )  # Option to close position on bot stop

        # Validate configurable options
        valid_triggers = [TRIGGER_BY_MARK, TRIGGER_BY_LAST, TRIGGER_BY_INDEX]
        valid_sl_types = [ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT]
        if self.sl_trigger_by not in valid_triggers:
            logger.warning(
                f"Invalid sl_trigger_by '{self.sl_trigger_by}'. Defaulting to '{TRIGGER_BY_MARK}'. Valid: {valid_triggers}"
            )
            self.sl_trigger_by = TRIGGER_BY_MARK
        if self.tp_trigger_by not in valid_triggers:
            logger.warning(
                f"Invalid tp_trigger_by '{self.tp_trigger_by}'. Defaulting to '{TRIGGER_BY_MARK}'. Valid: {valid_triggers}"
            )
            self.tp_trigger_by = TRIGGER_BY_MARK
        if self.sl_order_type not in valid_sl_types:
            logger.warning(
                f"Invalid sl_order_type '{self.sl_order_type}'. Defaulting to '{ORDER_TYPE_MARKET}'. Valid: {valid_sl_types}"
            )
            self.sl_order_type = ORDER_TYPE_MARKET

        logger.info(
            f"{Fore.CYAN}Pyrmethus enhances the Ehlers Strategy for {self.symbol} (TF: {self.timeframe}) using Pybit HTTP...{Style.RESET_ALL}"
        )
        logger.info(
            f"Configuration loaded: Testnet={self.api_config.testnet_mode}, Symbol={self.symbol}"
        )
        logger.info(
            f"SL/TP Mode: {'Atomic (Attached to Entry)' if self.attach_sl_tp_to_entry else 'Separate Orders'}"
        )
        logger.info(
            f"SL Trigger: {self.sl_trigger_by}, SL Order Type: {self.sl_order_type}, TP Trigger: {self.tp_trigger_by}"
        )
        logger.info(
            f"Risk Per Trade: {self.strategy_config.risk_per_trade:.2%}, Leverage: {self.strategy_config.leverage}x"
        )
        logger.info(f"Close Position on Stop: {self.close_on_stop}")

    def _initialize(self) -> bool:
        """
        Connects to the Bybit API, validates the market, sets configuration
        (leverage, position mode), fetches initial state, and performs cleanup.

        Returns:
            True if initialization was successful, False otherwise.
        """
        logger.info(
            f"{Fore.CYAN}--- Channeling Bybit Spirits (Initialization) ---{Style.RESET_ALL}"
        )
        try:
            # --- Connect to Bybit ---
            logger.info(
                f"{Fore.BLUE}Connecting to Bybit ({'Testnet' if self.api_config.testnet_mode else 'Mainnet'})...{Style.RESET_ALL}"
            )
            self.session = HTTP(
                testnet=self.api_config.testnet_mode,
                api_key=self.api_config.api_key,
                api_secret=self.api_config.api_secret,
                recv_window=self.api_config.recv_window,  # Use configured recv_window
            )

            # --- Verify Connection & Server Time ---
            logger.debug("Checking server time...")
            server_time_resp = self.session.get_server_time()
            if not server_time_resp or server_time_resp.get("retCode") != RET_CODE_OK:
                logger.critical(
                    f"{Back.RED}Failed to get server time! Response: {server_time_resp}{Style.RESET_ALL}"
                )
                self._safe_close_session()
                return False
            server_time_ms = int(server_time_resp["result"]["timeNano"]) // 1_000_000
            server_dt = pd.to_datetime(server_time_ms, unit="ms", utc=True)
            client_dt = pd.Timestamp.utcnow()
            time_diff = abs((client_dt - server_dt).total_seconds())
            logger.success(
                f"Connection successful. Server Time: {server_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            if (
                time_diff > self.api_config.max_time_sync_diff
            ):  # Check against configured max diff
                logger.critical(
                    f"{Back.RED}CRITICAL: Client-Server time difference ({time_diff:.2f}s) exceeds allowed limit ({self.api_config.max_time_sync_diff}s). Ensure system clock is synchronized via NTP. Halting.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"CRITICAL: Clock skew error ({time_diff:.1f}s) for {self.symbol} bot! Halting.",
                    self.sms_config,
                )
                self._safe_close_session()
                return False
            elif time_diff > 5:  # Warn if moderately high
                logger.warning(
                    f"{Fore.YELLOW}Client-Server time difference is {time_diff:.2f} seconds. Monitor clock synchronization.{Style.RESET_ALL}"
                )

            # --- Fetch Market Info & Determine Category ---
            logger.info(
                f"{Fore.BLUE}Seeking insights for symbol: {self.symbol}...{Style.RESET_ALL}"
            )
            if not self._fetch_and_set_market_info():
                logger.critical(
                    f"{Back.RED}Failed to fetch critical market info for {self.symbol}. Halting initialization.{Style.RESET_ALL}"
                )
                self._safe_close_session()
                return False
            logger.info(f"Determined Category: {self.category}")

            # --- Configure Derivatives Settings (Leverage, Position Mode, Margin Mode) ---
            if self.category in ["linear", "inverse"]:
                logger.info(
                    f"{Fore.BLUE}Configuring derivatives settings for {self.category}...{Style.RESET_ALL}"
                )

                # Set Leverage
                if not self._set_leverage():
                    logger.error(
                        f"{Back.RED}Failed to set leverage. Continuing, but positions may fail if leverage is incorrect.{Style.RESET_ALL}"
                    )
                    # Consider making this fatal depending on strategy needs
                    # return False

                # Set Position Mode (One-Way / Hedge)
                pos_mode_target_str = self.strategy_config.default_position_mode
                # Convert string from config ('MergedSingle'/'BothSide') to Bybit mode (0/3)
                target_pybit_pos_mode = (
                    POSITION_MODE_ONE_WAY
                    if pos_mode_target_str == "MergedSingle"
                    else POSITION_MODE_HEDGE
                )
                if target_pybit_pos_mode == POSITION_MODE_HEDGE:
                    logger.error(
                        f"{Back.RED}Hedge Mode (BothSide) is configured but NOT fully supported by this script's current logic. Ensure strategy accounts for separate Buy/Sell positions. Use 'MergedSingle' unless logic is adapted.{Style.RESET_ALL}"
                    )
                    # return False # Consider making this fatal

                if not self._set_position_mode(mode=target_pybit_pos_mode):
                    logger.warning(
                        f"{Fore.YELLOW}Could not explicitly set position mode to '{pos_mode_target_str}'. Ensure it's correct in Bybit UI.{Style.RESET_ALL}"
                    )
                else:
                    logger.info(
                        f"Position mode alignment to '{pos_mode_target_str}' confirmed for {self.category}."
                    )

                # Set Margin Mode (Isolated / Cross)
                margin_mode_target = (
                    self.strategy_config.default_margin_mode
                )  # 'ISOLATED' or 'CROSS'
                leverage_str = str(
                    int(self.strategy_config.leverage)
                )  # Leverage needed for isolated
                if not self._set_margin_mode(
                    mode=margin_mode_target, leverage=leverage_str
                ):
                    logger.warning(
                        f"{Fore.YELLOW}Could not explicitly set margin mode to '{margin_mode_target}'. Ensure it's correct in Bybit UI.{Style.RESET_ALL}"
                    )
                else:
                    logger.info(
                        f"Margin mode alignment to '{margin_mode_target}' confirmed for {self.category}."
                    )

            # --- Initial State Perception ---
            logger.info(
                f"{Fore.BLUE}Gazing into the account's current state...{Style.RESET_ALL}"
            )
            if not self._update_state():
                logger.error(
                    "Failed to perceive initial state. Cannot proceed reliably."
                )
                self._safe_close_session()
                return False  # Crucial to know the starting state
            logger.info(
                f"Initial Perception: Side={self.current_side}, Qty={self.current_qty}, Entry={format_price(self.symbol, self.entry_price, self.price_tick) if self.entry_price else 'N/A'}"
            )

            # --- Initial Order Cleanup ---
            logger.info(
                f"{Fore.BLUE}Dispelling lingering order phantoms (Initial Cleanup)...{Style.RESET_ALL}"
            )
            if not self._cancel_all_open_orders("Initialization Cleanup"):
                logger.warning(
                    "Initial order cancellation failed or encountered issues. Check Bybit UI for stray orders."
                )
            # Clear any tracked IDs after cancellation attempt, as we start fresh
            self.sl_order_id = None
            self.tp_order_id = None

            self.is_initialized = True
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}--- Strategy Initialization Complete ---{Style.RESET_ALL}"
            )
            return True

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Pybit API Error during initialization: {pybit_e}{Style.RESET_ALL}",
                exc_info=True,
            )
            logger.critical(
                f"Status Code: {pybit_e.status_code}, Response: {pybit_e.response}"
            )
            self._safe_close_session()
            return False
        except Exception as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Critical spell failure during initialization: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
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

        # Define categories to try
        possible_categories: List[Literal["linear", "inverse", "spot"]] = [
            "linear",
            "inverse",
            "spot",
        ]
        market_data: Optional[Dict] = None
        found_category: Optional[Literal["linear", "inverse", "spot"]] = None

        # --- Query Instruments Info ---
        for category_attempt in possible_categories:
            logger.debug(
                f"Attempting to fetch instruments info for category: {category_attempt}, symbol: {self.symbol}"
            )
            try:
                response = self.session.get_instruments_info(
                    category=category_attempt, symbol=self.symbol
                )

                if response and response.get("retCode") == RET_CODE_OK:
                    result_list = response.get("result", {}).get("list", [])
                    if result_list:
                        # Ensure the response is for the correct symbol (API might return multiple if symbol param is ignored/fuzzy)
                        matched_data = next(
                            (
                                item
                                for item in result_list
                                if item.get("symbol") == self.symbol
                            ),
                            None,
                        )
                        if matched_data:
                            logger.info(
                                f"Successfully found {self.symbol} in category '{category_attempt}'."
                            )
                            found_category = category_attempt
                            market_data = matched_data
                            break  # Found it, stop trying categories
                        else:
                            logger.debug(
                                f"Response received for category '{category_attempt}', but symbol {self.symbol} not in the list."
                            )
                    else:
                        logger.debug(
                            f"Symbol {self.symbol} not found in category '{category_attempt}'."
                        )
                else:
                    ret_code = response.get("retCode")
                    ret_msg = response.get("retMsg", "Unknown error")
                    if ret_code in [RET_CODE_API_KEY_INVALID, RET_CODE_SIGN_ERROR]:
                        logger.critical(
                            f"API Key/Secret error while fetching instruments info (Code: {ret_code}). Halting."
                        )
                        return False
                    logger.debug(
                        f"API call failed for category '{category_attempt}'. Code: {ret_code}, Msg: {ret_msg}"
                    )

            except (InvalidRequestError, FailedRequestError) as pybit_e:
                logger.error(
                    f"Pybit API Error fetching instruments info for category '{category_attempt}': {pybit_e}"
                )
                if pybit_e.status_code in [401, 403]:  # Authentication errors
                    logger.critical(
                        f"Authentication error (Status: {pybit_e.status_code}). Check API keys."
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching instruments info for category '{category_attempt}': {e}",
                    exc_info=True,
                )

        if not found_category or not market_data:
            logger.error(
                f"{Back.RED}Failed to find market info for symbol {self.symbol} in any tried category: {possible_categories}.{Style.RESET_ALL}"
            )
            return False

        self.category = found_category  # Set the determined category

        # --- Extract and Validate Details ---
        try:
            lot_size_filter = market_data.get("lotSizeFilter", {})
            price_filter = market_data.get("priceFilter", {})

            # Use Decimal context for precision
            ctx = DECIMAL_CONTEXT

            self.min_qty = safe_decimal_conversion(
                lot_size_filter.get("minOrderQty"), "min_qty", ctx
            )
            self.qty_step = safe_decimal_conversion(
                lot_size_filter.get("qtyStep"), "qty_step", ctx
            )
            self.price_tick = safe_decimal_conversion(
                price_filter.get("tickSize"), "price_tick", ctx
            )
            self.base_coin = market_data.get("baseCoin")
            self.quote_coin = market_data.get("quoteCoin")
            # Get contract multiplier (defaults to 1 if not present or invalid)
            self.contract_multiplier = safe_decimal_conversion(
                market_data.get("contractMultiplier", "1"), "contract_multiplier", ctx
            ) or ctx.create_decimal("1.0")

            # --- Validate Essential Details ---
            missing_details = []
            if self.min_qty is None or self.min_qty <= 0:
                missing_details.append(f"Minimum Quantity ({self.min_qty})")
            if self.qty_step is None or self.qty_step <= 0:
                missing_details.append(f"Quantity Step ({self.qty_step})")
            if self.price_tick is None or self.price_tick <= 0:
                missing_details.append(f"Price Tick ({self.price_tick})")
            if not self.base_coin:
                missing_details.append("Base Coin")
            if not self.quote_coin:
                missing_details.append("Quote Coin")
            if self.contract_multiplier is None or self.contract_multiplier <= 0:
                missing_details.append(
                    f"Contract Multiplier ({self.contract_multiplier})"
                )

            if missing_details:
                logger.error(
                    f"{Back.RED}Failed to extract or validate essential market details! Issues: {'; '.join(missing_details)}{Style.RESET_ALL}"
                )
                logger.error(
                    f"Received Market Data: {market_data}"
                )  # Log raw data for debugging
                return False

            logger.info(
                f"Market Details Set: Category={self.category}, Base={self.base_coin}, Quote={self.quote_coin}"
            )
            logger.info(
                f"Min Qty={format_amount(self.symbol, self.min_qty, self.qty_step)}, "
                f"Qty Step={self.qty_step}, Price Tick={self.price_tick}, Multiplier={self.contract_multiplier}"
            )
            return True

        except (InvalidOperation, TypeError, KeyError, Exception) as e:
            logger.error(f"Error processing market data details: {e}", exc_info=True)
            logger.error(f"Problematic Market Data: {market_data}")
            return False

    def _set_leverage(self) -> bool:
        """Sets leverage for the symbol (only applicable to derivatives)."""
        if not self.session or self.category not in ["linear", "inverse"]:
            logger.info(f"Leverage setting skipped (Category: {self.category}).")
            return True
        try:
            # Ensure leverage is a whole number string for the API
            leverage_str = str(int(self.strategy_config.leverage))
            logger.info(
                f"Attempting to set leverage for {self.symbol} to {leverage_str}x..."
            )
            response = self.session.set_leverage(
                category=self.category,
                symbol=self.symbol,
                buyLeverage=leverage_str,
                sellLeverage=leverage_str,
            )
            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "").lower()

            if ret_code == RET_CODE_OK:
                logger.success(f"Leverage set to {leverage_str}x successfully.")
                return True
            elif (
                ret_code == RET_CODE_LEVERAGE_NOT_MODIFIED
                or "leverage not modified" in ret_msg
            ):
                logger.info(
                    f"Leverage already set to {leverage_str}x (no modification needed)."
                )
                return True
            else:
                logger.error(
                    f"Failed to set leverage. Code: {ret_code}, Msg: {response.get('retMsg')}"
                )
                return False
        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error setting leverage: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting leverage: {e}", exc_info=True)
            return False

    def _set_position_mode(self, mode: int) -> bool:
        """Sets position mode (0=One-Way, 3=Hedge) (only for derivatives)."""
        if not self.session or self.category not in ["linear", "inverse"]:
            logger.info(f"Position mode setting skipped (Category: {self.category}).")
            return True
        mode_desc = (
            "One-Way (MergedSingle)"
            if mode == POSITION_MODE_ONE_WAY
            else "Hedge (BothSide)"
        )
        logger.info(
            f"Attempting to set position mode for {self.symbol} (category {self.category}) to {mode} ({mode_desc})..."
        )
        try:
            # Unified Trading API: switch_position_mode affects the *entire category* (e.g., all linear)
            # It's generally better to set this once manually or ensure it matches config.
            # We attempt to set it here but log warnings if it fails.
            response = self.session.switch_position_mode(
                category=self.category,
                # symbol=self.symbol, # Symbol is NOT applicable for switch_position_mode
                # coin=self.quote_coin, # Coin is NOT applicable for switch_position_mode
                mode=mode,
            )

            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "").lower()

            if ret_code == RET_CODE_OK:
                logger.info(
                    f"Position mode successfully set/confirmed to {mode_desc} for {self.category}."
                )
                return True
            elif (
                ret_code == RET_CODE_POSITION_MODE_NOT_MODIFIED
                or "position mode is not modified" in ret_msg
            ):
                logger.info(
                    f"Position mode already set to {mode_desc} for {self.category}."
                )
                return True
            else:
                # Common failure: Cannot switch mode with active positions/orders
                logger.error(
                    f"Failed to set position mode. Code: {ret_code}, Msg: {response.get('retMsg')}"
                )
                return False
        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error setting position mode: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
            # If error indicates cannot switch due to active state, treat as non-fatal warning?
            if "cannot be switched" in str(pybit_e.response).lower():
                logger.warning(
                    "Could not switch position mode, likely due to active positions/orders. Ensure mode is correct."
                )
                return True  # Treat as success (already set or cannot change)
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting position mode: {e}", exc_info=True)
            return False

    def _set_margin_mode(self, mode: str, leverage: str) -> bool:
        """Sets margin mode (ISOLATED/CROSS) for the symbol (derivatives only)."""
        if not self.session or self.category not in ["linear", "inverse"]:
            logger.info(f"Margin mode setting skipped (Category: {self.category}).")
            return True

        # API requires integer for mode: 0 = CROSS, 1 = ISOLATED
        trade_mode = 1 if mode.upper() == "ISOLATED" else 0
        mode_desc = "Isolated" if trade_mode == 1 else "Cross"

        logger.info(
            f"Attempting to set margin mode for {self.symbol} to {mode_desc}..."
        )
        try:
            response = self.session.switch_margin_mode(
                category=self.category,
                symbol=self.symbol,
                tradeMode=trade_mode,  # 0 for Cross, 1 for Isolated
                buyLeverage=leverage,  # Required even for cross sometimes, definitely for isolated
                sellLeverage=leverage,  # Required even for cross sometimes, definitely for isolated
            )
            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "").lower()

            if ret_code == RET_CODE_OK:
                logger.success(f"Margin mode set to {mode_desc} successfully.")
                return True
            elif (
                ret_code == RET_CODE_NO_NEED_TO_SET_MARGIN_MODE
                or "not modified" in ret_msg
            ):
                logger.info(
                    f"Margin mode already set to {mode_desc} (no modification needed)."
                )
                return True
            else:
                # Common failure: Cannot switch with active position/orders
                logger.error(
                    f"Failed to set margin mode. Code: {ret_code}, Msg: {response.get('retMsg')}"
                )
                return False
        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error setting margin mode: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
            if "cannot be modified" in str(pybit_e.response).lower():
                logger.warning(
                    "Could not switch margin mode, likely due to active positions/orders. Ensure mode is correct."
                )
                return True  # Treat as success (already set or cannot change)
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting margin mode: {e}", exc_info=True)
            return False

    def _get_available_balance(self) -> Optional[Decimal]:
        """
        Fetches available balance for the relevant quote coin and account type.
        Used primarily for position sizing.

        Returns:
            The available balance as a Decimal, or None if fetching fails.
        """
        if not self.session or not self.category:
            logger.error("Cannot fetch balance: Missing session or category.")
            return None
        if not self.quote_coin:  # Should be set during init
            logger.error("Cannot fetch balance: Quote coin not determined.")
            return None

        # Determine account type based on category (primarily for Unified)
        # Bybit V5 often uses 'UNIFIED' for linear/inverse, 'SPOT' for spot, 'CONTRACT' for classic accounts
        account_type: str = (
            ACCOUNT_TYPE_UNIFIED  # Assume unified by default for linear/inverse
        )
        if self.category == "spot":
            account_type = (
                ACCOUNT_TYPE_SPOT  # Or UNIFIED if spot is under unified wallet
            )

        coin_to_check: str = self.quote_coin  # Margin/PnL currency (USDT, USD, etc.)

        logger.debug(
            f"Fetching balance for Account: {account_type}, Coin: {coin_to_check}..."
        )
        try:
            # Use get_wallet_balance for V5 Unified/Spot
            bal_response = self.session.get_wallet_balance(
                accountType=account_type, coin=coin_to_check
            )

            if not (bal_response and bal_response.get("retCode") == RET_CODE_OK):
                logger.error(
                    f"Failed to fetch balance data. Code: {bal_response.get('retCode')}, Msg: {bal_response.get('retMsg')}"
                )
                return None

            balance_list = bal_response.get("result", {}).get("list", [])
            if not balance_list:
                logger.warning(
                    f"Balance list is empty in the response for {account_type}."
                )
                return DECIMAL_CONTEXT.create_decimal(
                    "0.0"
                )  # Assume zero if list is empty

            # The structure might contain account info first, then coin list
            account_balance_data = balance_list[
                0
            ]  # Assuming the first element contains the relevant account type data
            coin_balance_list = account_balance_data.get("coin", [])

            coin_balance_data = next(
                (
                    item
                    for item in coin_balance_list
                    if item.get("coin") == coin_to_check
                ),
                None,
            )

            if coin_balance_data:
                # V5 Unified uses 'availableToWithdraw' or 'availableBalance'
                # 'availableBalance' usually represents the balance usable for margin/new trades
                available_balance_str = coin_balance_data.get("availableBalance", "0")
                available_balance = safe_decimal_conversion(
                    available_balance_str, "availableBalance", DECIMAL_CONTEXT
                )

                if available_balance is None:
                    logger.error(
                        f"Could not parse 'availableBalance' ({available_balance_str}) for {coin_to_check}."
                    )
                    return None  # Treat parsing failure as critical

                equity_str = coin_balance_data.get(
                    "equity", "N/A"
                )  # Total equity if available
                logger.info(
                    f"Available Balance ({coin_to_check}): {available_balance:.4f}, Equity: {equity_str}"
                )
                return available_balance
            else:
                logger.warning(
                    f"Could not find balance details for coin '{coin_to_check}' within the response list for account {account_type}."
                )
                return DECIMAL_CONTEXT.create_decimal(
                    "0.0"
                )  # Assume zero if coin not found

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error fetching balance: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching balance: {e}", exc_info=True)
            return None

    def _update_state(self) -> bool:
        """
        Fetches and updates the current position (size, side, entry price) and
        logs available balance. Clears tracked SL/TP orders if flat.

        Returns:
            True if the position state was updated successfully, False otherwise.
            (Balance fetch failure is logged but doesn't cause False return).
        """
        if not self.session or not self.category:
            logger.error("Cannot update state: Session or category not set.")
            return False
        logger.debug("Updating strategy state perception...")
        position_updated_successfully = False
        ctx = DECIMAL_CONTEXT
        try:
            # --- Fetch Position ---
            logger.debug(f"Fetching position for {self.category}/{self.symbol}...")
            pos_response = self.session.get_positions(
                category=self.category, symbol=self.symbol
            )

            if not (pos_response and pos_response.get("retCode") == RET_CODE_OK):
                logger.error(
                    f"Failed to fetch position data. Code: {pos_response.get('retCode')}, Msg: {pos_response.get('retMsg')}"
                )
                # Don't reset state here, just report failure to fetch
                self._reset_position_state(
                    "Position fetch API call failed."
                )  # Reset to be safe
                return (
                    False  # Treat position fetch failure as critical for state update
                )
            else:
                position_list = pos_response.get("result", {}).get("list", [])
                if not position_list:
                    # No position data returned, assume flat
                    self._reset_position_state(
                        "No position data found in API response."
                    )
                else:
                    # Assuming One-Way mode: use the first entry.
                    # If Hedge Mode, logic would need to find correct entry based on side or idx.
                    pos_data = position_list[0]
                    pos_qty_str = pos_data.get("size", "0")
                    pos_qty = safe_decimal_conversion(pos_qty_str, "position size", ctx)
                    side_str = pos_data.get("side", "None")  # 'Buy', 'Sell', or 'None'
                    avg_price_str = pos_data.get("avgPrice")

                    # Check for valid position (size > epsilon and side is Buy/Sell)
                    if (
                        pos_qty is not None
                        and pos_qty > ctx.create_decimal("0.0")
                        and side_str in [SIDE_BUY, SIDE_SELL]
                    ):
                        old_side = self.current_side
                        old_qty = self.current_qty

                        self.current_qty = pos_qty
                        self.entry_price = safe_decimal_conversion(
                            avg_price_str, "entry price", ctx
                        )
                        self.current_side = (
                            POS_LONG if side_str == SIDE_BUY else POS_SHORT
                        )

                        if self.entry_price is None:
                            logger.warning(
                                f"Position found ({side_str} {pos_qty_str}), but average price ('{avg_price_str}') is invalid. State may be inaccurate."
                            )
                            # Decide handling: reset state? Proceed cautiously? Resetting might be safer.
                            # self._reset_position_state("Invalid average price in position data.")
                            # return False # Treat invalid avgPrice as critical state failure? Maybe not, SL/TP might still work. Proceed with warning.

                        # Log state change if detected
                        if self.current_side != old_side or self.current_qty != old_qty:
                            logger.info(
                                f"Position State Changed: Side={self.current_side}, Qty={self.current_qty}, Entry={format_price(self.symbol, self.entry_price, self.price_tick) if self.entry_price else 'N/A'}"
                            )

                    else:
                        # Position size is zero, negligible, or side is 'None' -> Treat as flat
                        reset_reason = f"Position size ('{pos_qty_str}') is zero/negligible or side ('{side_str}') indicates no active position."
                        self._reset_position_state(reset_reason)

                logger.debug(
                    f"Position State Updated: Side={self.current_side}, Qty={self.current_qty}, Entry={self.entry_price}"
                )
                position_updated_successfully = True

            # --- Fetch Balance (Primarily for Logging Here) ---
            _ = self._get_available_balance()  # Fetch and log balance info

            # --- Clear Tracked Orders if Flat ---
            if self.current_side == POS_NONE:
                if self.sl_order_id or self.tp_order_id:
                    logger.debug(
                        "Not in position, clearing tracked separate SL/TP order IDs."
                    )
                    self.sl_order_id = None
                    self.tp_order_id = None

            logger.debug("State update perception complete.")
            return position_updated_successfully  # Return True only if position fetch was successful

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error during state update: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
            self._reset_position_state(
                "API error during state update."
            )  # Reset to be safe
            return False
        except Exception as e:
            logger.error(f"Unexpected error during state update: {e}", exc_info=True)
            self._reset_position_state(
                "Exception during state update."
            )  # Reset to be safe
            return False

    def _reset_position_state(self, reason: str):
        """Resets internal position tracking variables to 'flat' state."""
        ctx = DECIMAL_CONTEXT
        if self.current_side != POS_NONE:  # Log only if state is actually changing
            logger.info(f"Resetting position state to NONE. Reason: {reason}")
        self.current_side = POS_NONE
        self.current_qty = ctx.create_decimal("0.0")
        self.entry_price = None
        # SL/TP order IDs are cleared in _update_state *if* the position is found to be flat.
        # Do not clear them here, as this function might be called before cancellation attempts.

    def _fetch_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Decimal]]:
        """
        Fetches OHLCV (Kline) data and the latest ticker price using Pybit.

        Returns:
            A tuple containing:
            - DataFrame with OHLCV data (or None if fetch fails).
            - Latest price as a Decimal (or None if fetch fails).
        """
        if (
            not self.session
            or not self.category
            or not self.timeframe
            or not self.symbol
        ):
            logger.error(
                "Cannot fetch data: Missing session, category, timeframe, or symbol."
            )
            return None, None
        logger.debug("Fetching market data...")
        ohlcv_df: Optional[pd.DataFrame] = None
        current_price: Optional[Decimal] = None
        ctx = DECIMAL_CONTEXT

        # --- Fetch OHLCV (Kline) ---
        try:
            limit = self.strategy_config.ohlcv_limit
            limit = min(limit, 1000)  # Bybit API limit for kline
            logger.debug(
                f"Fetching Kline: {self.symbol}, Interval: {self.timeframe}, Limit: {limit}"
            )
            kline_response = self.session.get_kline(
                category=self.category,
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit,
            )

            if not (kline_response and kline_response.get("retCode") == RET_CODE_OK):
                logger.warning(
                    f"Could not fetch OHLCV data. Code: {kline_response.get('retCode')}, Msg: {kline_response.get('retMsg')}"
                )
            else:
                kline_list = kline_response.get("result", {}).get("list", [])
                if not kline_list:
                    logger.warning("OHLCV data list is empty in the response.")
                else:
                    # Bybit V5 Kline: [timestamp, open, high, low, close, volume, turnover]
                    columns = [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "turnover",
                    ]
                    ohlcv_df = pd.DataFrame(kline_list, columns=columns)
                    ohlcv_df["timestamp"] = pd.to_numeric(ohlcv_df["timestamp"])
                    ohlcv_df["datetime"] = pd.to_datetime(
                        ohlcv_df["timestamp"], unit="ms", utc=True
                    )
                    # Convert OHLCV to numeric, handling potential errors
                    for col in ["open", "high", "low", "close", "volume", "turnover"]:
                        ohlcv_df[col] = pd.to_numeric(ohlcv_df[col], errors="coerce")
                    # Drop rows with NaN in essential columns (open, high, low, close, volume)
                    ohlcv_df.dropna(
                        subset=["open", "high", "low", "close", "volume"], inplace=True
                    )
                    # Ensure data is sorted chronologically (Bybit usually returns descending)
                    ohlcv_df = ohlcv_df.sort_values(by="timestamp").reset_index(
                        drop=True
                    )
                    ohlcv_df.set_index("datetime", inplace=True)
                    if not ohlcv_df.empty:
                        logger.debug(
                            f"Successfully fetched and processed {len(ohlcv_df)} candles. Latest: {ohlcv_df.index[-1]}"
                        )
                    else:
                        logger.warning(
                            "OHLCV data processing resulted in an empty DataFrame after handling errors/NaNs."
                        )
                        ohlcv_df = None  # Set back to None if empty

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error fetching kline data: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
        except Exception as e:
            logger.error(f"Unexpected error processing kline data: {e}", exc_info=True)

        # --- Fetch Ticker ---
        try:
            logger.debug(f"Fetching ticker for {self.symbol}...")
            ticker_response = self.session.get_tickers(
                category=self.category, symbol=self.symbol
            )

            if not (ticker_response and ticker_response.get("retCode") == RET_CODE_OK):
                logger.warning(
                    f"Could not fetch ticker data. Code: {ticker_response.get('retCode')}, Msg: {ticker_response.get('retMsg')}"
                )
            else:
                ticker_list = ticker_response.get("result", {}).get("list", [])
                if not ticker_list:
                    logger.warning("Ticker data list is empty in the response.")
                else:
                    # Assuming the first ticker in the list is the correct one
                    last_price_str = ticker_list[0].get("lastPrice")
                    current_price = safe_decimal_conversion(
                        last_price_str, "lastPrice", ctx
                    )
                    if current_price is None:
                        logger.warning(
                            f"Ticker data retrieved but missing valid 'lastPrice' ('{last_price_str}')."
                        )
                    else:
                        logger.debug(f"Last Price: {current_price}")

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error fetching ticker data: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
        except Exception as e:
            logger.error(f"Unexpected error processing ticker data: {e}", exc_info=True)

        # --- Final Check and Return ---
        if ohlcv_df is None:
            logger.warning("OHLCV data fetch failed or resulted in empty data.")
        if current_price is None:
            logger.warning("Current price fetch failed or resulted in invalid data.")

        return ohlcv_df, current_price

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
            logger.warning(
                "Cannot calculate indicators: Input DataFrame is None or empty."
            )
            return None
        # Ensure enough data for lookbacks + some buffer
        required_len = (
            max(
                self.strategy_config.indicator_settings.evt_length,
                self.strategy_config.indicator_settings.atr_period,
            )
            + 5
        )
        if len(ohlcv_df) < required_len:
            logger.warning(
                f"Cannot calculate indicators: Insufficient data length ({len(ohlcv_df)} candles). Need at least {required_len}."
            )
            return None

        logger.debug(f"Calculating indicators on {len(ohlcv_df)} candles...")
        try:
            # Prepare config dictionary for the indicators module
            indicator_config_dict = {
                "indicator_settings": self.strategy_config.indicator_settings.model_dump(),
                "analysis_flags": self.strategy_config.analysis_flags.model_dump(),
            }

            # Call the external calculation function
            df_with_indicators = ind.calculate_all_indicators(
                ohlcv_df.copy(), indicator_config_dict
            )  # Pass a copy

            # --- Validation ---
            if df_with_indicators is None:
                logger.error(
                    "Indicator calculation script (indicators.py) returned None."
                )
                return None
            if df_with_indicators.empty:
                logger.error(
                    "Indicator calculation script returned an empty DataFrame."
                )
                return None

            # Define expected column names based on config
            evt_len = self.strategy_config.indicator_settings.evt_length
            atr_len = self.strategy_config.indicator_settings.atr_period
            evt_trend_col = f"evt_trend_{evt_len}"
            evt_buy_col = f"evt_buy_{evt_len}"
            evt_sell_col = f"evt_sell_{evt_len}"
            atr_col = f"ATRr_{atr_len}"  # Default pandas_ta name for ATR

            required_cols = [evt_trend_col, evt_buy_col, evt_sell_col]
            if self.strategy_config.analysis_flags.use_atr:
                required_cols.append(atr_col)

            missing_cols = [
                col for col in required_cols if col not in df_with_indicators.columns
            ]
            if missing_cols:
                logger.error(
                    f"Required indicator columns missing after calculation: {missing_cols}. Check 'indicators.py'."
                )
                return None

            # Check for NaNs in the *latest* row's critical columns
            if df_with_indicators.empty:
                logger.error(
                    "Indicator DataFrame is empty after calculation, cannot check latest row."
                )
                return None
            latest_row = df_with_indicators.iloc[-1]
            nan_cols = [col for col in required_cols if pd.isna(latest_row.get(col))]
            if nan_cols:
                logger.warning(
                    f"NaN values found in critical indicator columns of latest row ({latest_row.name}): {nan_cols}. Cannot generate reliable signal."
                )
                return None  # Skip iteration if latest data is NaN

            logger.debug("Indicators calculated successfully.")
            return df_with_indicators

        except Exception as e:
            logger.error(f"Error during indicator calculation: {e}", exc_info=True)
            return None

    def _generate_signals(
        self, df_ind: pd.DataFrame
    ) -> Optional[Literal["long", "short"]]:
        """
        Generates trading signals ('long' or 'short') based on the last
        indicator data point in the provided DataFrame.

        Args:
            df_ind: DataFrame with calculated indicators.

        Returns:
            'long', 'short', or None if no signal is generated or data is invalid.
        """
        if df_ind is None or df_ind.empty:
            logger.debug(
                "Cannot generate signals: Indicator DataFrame is missing or empty."
            )
            return None
        logger.debug("Generating trading signals...")
        try:
            # Access the latest row safely
            latest = df_ind.iloc[-1]
            latest_time = latest.name  # Timestamp of the latest candle
            evt_len = self.strategy_config.indicator_settings.evt_length
            buy_col = f"evt_buy_{evt_len}"
            sell_col = f"evt_sell_{evt_len}"

            # Check required columns exist and are not NaN in the latest row
            if not all(
                col in latest.index and pd.notna(latest[col])
                for col in [buy_col, sell_col]
            ):
                logger.warning(
                    f"EVT Buy/Sell signal columns missing or NaN in latest data ({latest_time}). Cannot generate signal."
                )
                return None

            buy_signal = bool(latest[buy_col])
            sell_signal = bool(latest[sell_col])

            # --- Signal Logic ---
            if buy_signal and sell_signal:
                logger.warning(
                    f"Both Buy and Sell signals are active simultaneously on latest candle ({latest_time}). Ignoring signals."
                )
                return None
            elif buy_signal:
                logger.info(
                    f"{Fore.GREEN}BUY signal generated based on EVT Buy flag at {latest_time}.{Style.RESET_ALL}"
                )
                return POS_LONG
            elif sell_signal:
                logger.info(
                    f"{Fore.RED}SELL signal generated based on EVT Sell flag at {latest_time}.{Style.RESET_ALL}"
                )
                return POS_SHORT
            else:
                return None

        except IndexError:
            logger.warning(
                "IndexError generating signals (DataFrame likely too short or empty)."
            )
            return None
        except KeyError as e:
            logger.error(
                f"KeyError generating signals: Missing expected column '{e}'. Check indicator calculation."
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating signals: {e}", exc_info=True)
            return None

    def _calculate_sl_tp(
        self,
        df_ind: pd.DataFrame,
        side: Literal["long", "short"],
        entry_price_approx: Decimal,
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
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
        ctx = DECIMAL_CONTEXT
        if df_ind is None or df_ind.empty:
            logger.error("Cannot calculate SL/TP: Missing indicator data.")
            return None, None
        if self.price_tick is None or self.price_tick <= ctx.create_decimal(0):
            logger.error(
                f"Cannot calculate SL/TP: Invalid price tick ({self.price_tick})."
            )
            return None, None
        if entry_price_approx <= ctx.create_decimal(0):
            logger.error(
                f"Cannot calculate SL/TP: Invalid approximate entry price ({entry_price_approx})."
            )
            return None, None
        if not self.strategy_config.analysis_flags.use_atr:
            logger.error(
                "Cannot calculate SL/TP: ATR usage is disabled in config (analysis_flags.use_atr)."
            )
            return None, None

        logger.debug(f"Calculating SL/TP for {side} entry near {entry_price_approx}...")
        try:
            # --- Get Latest ATR ---
            atr_len = self.strategy_config.indicator_settings.atr_period
            atr_col = f"ATRr_{atr_len}"  # Default pandas_ta name
            if atr_col not in df_ind.columns:
                logger.error(
                    f"ATR column '{atr_col}' not found in indicator DataFrame."
                )
                return None, None
            latest_atr_val = df_ind.iloc[-1].get(atr_col)
            if pd.isna(latest_atr_val):
                logger.error(f"Latest ATR value in column '{atr_col}' is NaN.")
                return None, None

            latest_atr = safe_decimal_conversion(latest_atr_val, "latest ATR", ctx)
            if latest_atr is None or latest_atr <= ctx.create_decimal(0):
                logger.warning(
                    f"Invalid ATR value ({latest_atr_val}) for SL/TP calculation. Cannot proceed."
                )
                return None, None

            # --- Stop Loss Calculation ---
            sl_multiplier = ctx.create_decimal(
                str(self.strategy_config.stop_loss_atr_multiplier)
            )
            if sl_multiplier <= 0:
                logger.error("Stop Loss ATR multiplier must be positive.")
                return None, None
            sl_offset = latest_atr * sl_multiplier
            stop_loss_price_raw = (
                (entry_price_approx - sl_offset)
                if side == POS_LONG
                else (entry_price_approx + sl_offset)
            )

            if stop_loss_price_raw <= ctx.create_decimal(0):
                logger.error(
                    f"Raw SL price calculated as zero or negative ({stop_loss_price_raw}). Check ATR/multiplier/price."
                )
                return None, None

            # Round SL *away* from the entry price to respect the tick size
            rounding_mode_sl = ROUND_DOWN if side == POS_LONG else ROUND_UP
            sl_price_adjusted = (stop_loss_price_raw / self.price_tick).quantize(
                ctx.create_decimal("0"), rounding=rounding_mode_sl
            ) * self.price_tick

            # Sanity check: Ensure SL didn't cross entry after rounding
            if side == POS_LONG and sl_price_adjusted >= entry_price_approx:
                sl_price_adjusted = entry_price_approx - self.price_tick
                logger.warning(
                    f"Adjusted Buy SL ({sl_price_adjusted}) was >= approx entry ({entry_price_approx}). Moved SL one tick below entry."
                )
            elif side == POS_SHORT and sl_price_adjusted <= entry_price_approx:
                sl_price_adjusted = entry_price_approx + self.price_tick
                logger.warning(
                    f"Adjusted Sell SL ({sl_price_adjusted}) was <= approx entry ({entry_price_approx}). Moved SL one tick above entry."
                )

            if sl_price_adjusted <= ctx.create_decimal(0):
                logger.error(
                    f"Final SL price is zero or negative ({sl_price_adjusted}) after rounding/adjustment. Cannot set SL."
                )
                return None, None

            # --- Take Profit Calculation ---
            tp_price_adjusted: Optional[Decimal] = None
            tp_multiplier = ctx.create_decimal(
                str(self.strategy_config.take_profit_atr_multiplier)
            )
            if tp_multiplier > 0:
                tp_offset = latest_atr * tp_multiplier
                take_profit_price_raw = (
                    (entry_price_approx + tp_offset)
                    if side == POS_LONG
                    else (entry_price_approx - tp_offset)
                )

                if (
                    side == POS_LONG and take_profit_price_raw <= entry_price_approx
                ) or (
                    side == POS_SHORT and take_profit_price_raw >= entry_price_approx
                ):
                    logger.warning(
                        f"Raw TP price ({take_profit_price_raw}) is not logical relative to approx entry ({entry_price_approx}). Skipping TP."
                    )
                elif take_profit_price_raw <= ctx.create_decimal(0):
                    logger.warning(
                        f"Raw TP price ({take_profit_price_raw}) is zero or negative. Skipping TP."
                    )
                else:
                    # Round TP *away* from entry to be conservative? Or towards?
                    # Let's round DOWN for BUY TP, UP for SELL TP (makes target slightly harder to hit but ensures profit if hit)
                    rounding_mode_tp = ROUND_DOWN if side == POS_LONG else ROUND_UP
                    tp_price_adjusted_candidate = (
                        take_profit_price_raw / self.price_tick
                    ).quantize(
                        ctx.create_decimal("0"), rounding=rounding_mode_tp
                    ) * self.price_tick

                    # Sanity check: Ensure TP didn't cross entry after rounding
                    if (
                        side == POS_LONG
                        and tp_price_adjusted_candidate <= entry_price_approx
                    ):
                        tp_price_adjusted_candidate = (
                            entry_price_approx + self.price_tick
                        )
                        logger.warning(
                            f"Adjusted Buy TP ({tp_price_adjusted_candidate}) was <= approx entry ({entry_price_approx}). Moved TP one tick above entry."
                        )
                    elif (
                        side == POS_SHORT
                        and tp_price_adjusted_candidate >= entry_price_approx
                    ):
                        tp_price_adjusted_candidate = (
                            entry_price_approx - self.price_tick
                        )
                        logger.warning(
                            f"Adjusted Sell TP ({tp_price_adjusted_candidate}) was >= approx entry ({entry_price_approx}). Moved TP one tick below entry."
                        )

                    if tp_price_adjusted_candidate <= ctx.create_decimal(0):
                        logger.warning(
                            f"Final TP price is zero or negative ({tp_price_adjusted_candidate}) after rounding/adjustment. Skipping TP."
                        )
                    else:
                        tp_price_adjusted = tp_price_adjusted_candidate
            else:
                logger.info(
                    "Take Profit multiplier is zero or negative. TP is disabled."
                )

            sl_formatted = self._format_price_str(sl_price_adjusted)
            tp_formatted = (
                self._format_price_str(tp_price_adjusted)
                if tp_price_adjusted
                else "None"
            )
            logger.info(
                f"Calculated SL: {sl_formatted}, TP: {tp_formatted} (Based on ATR: {latest_atr:.5f})"
            )

            return sl_price_adjusted, tp_price_adjusted

        except (InvalidOperation, TypeError, Exception) as e:
            logger.error(f"Error calculating SL/TP: {e}", exc_info=True)
            return None, None

    def _calculate_position_size(
        self, entry_price_approx: Decimal, stop_loss_price: Decimal
    ) -> Optional[Decimal]:
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
        ctx = DECIMAL_CONTEXT
        # --- Input Validation ---
        if not all(
            [
                self.qty_step,
                self.min_qty,
                self.price_tick,
                self.base_coin,
                self.quote_coin,
                self.contract_multiplier,
            ]
        ):
            logger.error(
                "Cannot calculate size: Missing critical market details (steps, ticks, coins, multiplier)."
            )
            return None
        if entry_price_approx <= 0 or stop_loss_price <= 0:
            logger.error(
                f"Cannot calculate size: Invalid entry ({entry_price_approx}) or SL ({stop_loss_price}) price."
            )
            return None
        price_diff = abs(entry_price_approx - stop_loss_price)
        if price_diff < self.price_tick:
            logger.error(
                f"Cannot calculate size: Entry price ({entry_price_approx}) and SL price ({stop_loss_price}) are too close (diff: {price_diff} < tick: {self.price_tick})."
            )
            return None
        risk_percent = ctx.create_decimal(str(self.strategy_config.risk_per_trade))
        if risk_percent <= 0 or risk_percent >= 1:
            logger.error(
                f"Invalid risk_per_trade ({risk_percent}). Must be between 0 and 1."
            )
            return None

        logger.debug("Calculating position size...")
        try:
            # --- Get Available Balance ---
            available_balance = self._get_available_balance()
            if available_balance is None:
                logger.error(
                    "Cannot calculate position size: Failed to fetch available balance."
                )
                return None
            if available_balance <= ctx.create_decimal("0"):
                logger.error(
                    f"Cannot calculate position size: Available balance ({available_balance} {self.quote_coin}) is zero or negative."
                )
                return None

            # --- Calculate Risk Amount ---
            risk_amount_quote = available_balance * risk_percent
            logger.debug(
                f"Available Balance: {available_balance:.4f} {self.quote_coin}, Risk %: {risk_percent:.2%}, Risk Amount: {risk_amount_quote:.4f} {self.quote_coin}"
            )

            # --- Calculate Raw Size Based on Category ---
            position_size_raw: Decimal
            if self.contract_multiplier <= 0:
                logger.error("Cannot calculate size: Invalid contract multiplier.")
                return None

            if self.category == "inverse":
                try:
                    inv_entry = ctx.power(entry_price_approx, -1)
                    inv_sl = ctx.power(stop_loss_price, -1)
                except InvalidOperation:
                    logger.error(
                        "Division by zero error during inverse contract size calculation (price likely zero)."
                    )
                    return None
                size_denominator = self.contract_multiplier * abs(inv_entry - inv_sl)
                if size_denominator <= 0:
                    logger.error(
                        "Inverse size denominator is zero or negative. Cannot calculate size."
                    )
                    return None
                position_size_raw = risk_amount_quote / size_denominator
            elif self.category == "linear":
                size_denominator = self.contract_multiplier * price_diff
                if size_denominator <= 0:
                    logger.error(
                        "Linear size denominator is zero or negative. Cannot calculate size."
                    )
                    return None
                position_size_raw = risk_amount_quote / size_denominator
            elif self.category == "spot":
                if price_diff <= 0:
                    logger.error(
                        "Spot size denominator (price_diff) is zero or negative."
                    )
                    return None
                position_size_raw = risk_amount_quote / price_diff
            else:
                logger.error(
                    f"Position sizing not implemented for category: {self.category}"
                )
                return None

            logger.debug(
                f"Raw calculated size: {position_size_raw:.8f} {self.base_coin}"
            )

            # --- Apply Quantity Constraints (Step and Minimum) ---
            if self.qty_step <= 0:
                logger.error(
                    f"Invalid quantity step ({self.qty_step}). Cannot adjust size."
                )
                return None

            # Adjust for quantity step (round DOWN to not exceed risk)
            position_size_adjusted = (position_size_raw / self.qty_step).quantize(
                ctx.create_decimal("0"), rounding=ROUND_DOWN
            ) * self.qty_step

            if position_size_adjusted <= ctx.create_decimal(0):
                logger.warning(
                    f"Calculated position size is zero after step adjustment. Raw: {position_size_raw}, Step: {self.qty_step}. Insufficient balance or risk settings too low?"
                )
                return None

            if self.min_qty is None or self.min_qty < 0:
                logger.error(
                    f"Invalid minimum quantity ({self.min_qty}). Cannot validate size."
                )
                return None

            if position_size_adjusted < self.min_qty:
                logger.warning(
                    f"Calculated size ({position_size_adjusted}) is below minimum allowed quantity ({self.min_qty}). Insufficient capital for configured risk, or risk % too low."
                )
                return None

            # --- Log and Return ---
            size_formatted = self._format_qty(position_size_adjusted)
            logger.info(f"Calculated position size: {size_formatted} {self.base_coin}")
            return position_size_adjusted

        except (InvalidOperation, TypeError, Exception) as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    def _format_qty(self, qty: Optional[Decimal]) -> Optional[str]:
        """
        Formats a quantity Decimal to a string according to the market's qty_step.
        Rounds DOWN to the nearest step. Returns None if input/step is invalid.
        """
        ctx = DECIMAL_CONTEXT
        if qty is None or qty < ctx.create_decimal(0):
            return None
        if self.qty_step is None or self.qty_step <= ctx.create_decimal(0):
            logger.warning(
                f"Cannot format quantity: Invalid qty_step ({self.qty_step}). Returning raw string."
            )
            return str(qty)

        try:
            quantized_qty = (qty / self.qty_step).quantize(
                ctx.create_decimal("0"), rounding=ROUND_DOWN
            ) * self.qty_step
            # Format to string without scientific notation and respecting step decimals
            step_str = str(self.qty_step.normalize())
            decimals = len(step_str.split(".")[-1]) if "." in step_str else 0
            # Use f-string formatting for fixed-point notation
            return f"{quantized_qty:.{decimals}f}"
        except (InvalidOperation, TypeError) as e:
            logger.error(
                f"Error formatting quantity {qty} with step {self.qty_step}: {e}"
            )
            return None

    def _format_price_str(self, price: Optional[Decimal]) -> Optional[str]:
        """
        Formats a price Decimal to a string according to the market's price_tick.
        Uses standard rounding (ROUND_HALF_UP) to the nearest tick. Returns None if input/tick is invalid.
        """
        ctx = DECIMAL_CONTEXT
        if price is None or price <= ctx.create_decimal(0):
            return None
        if self.price_tick is None or self.price_tick <= ctx.create_decimal(0):
            logger.warning(
                f"Cannot format price: Invalid price_tick ({self.price_tick}). Returning raw string."
            )
            return str(price)

        try:
            quantized_price = (price / self.price_tick).quantize(
                ctx.create_decimal("0"), rounding=ROUND_HALF_UP
            ) * self.price_tick
            if quantized_price <= 0 and price > 0:
                logger.warning(
                    f"Price {price} quantized to zero with tick {self.price_tick}. Using smallest possible price (tick size)."
                )
                quantized_price = self.price_tick

            tick_str = str(self.price_tick.normalize())
            decimals = len(tick_str.split(".")[-1]) if "." in tick_str else 0
            return f"{quantized_price:.{decimals}f}"
        except (InvalidOperation, TypeError) as e:
            logger.error(
                f"Error formatting price {price} with tick {self.price_tick}: {e}"
            )
            return None

    def _place_order(self, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Wrapper for placing orders using Pybit's place_order method.
        Includes parameter validation, enhanced logging, and error code handling.

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
        required_params = ["category", "symbol", "side", "orderType", "qty"]
        missing_params = [
            p for p in required_params if p not in params or params[p] is None
        ]
        if missing_params:
            logger.error(
                f"Missing required parameters for placing order: {', '.join(missing_params)}. Params: {params}"
            )
            return None
        # Validate quantity format (string) and value
        qty_str = params["qty"]
        if not isinstance(qty_str, str) or not qty_str:
            logger.error(
                f"Invalid 'qty' parameter type or empty value: {qty_str}. Must be non-empty string."
            )
            return None
        try:
            qty_val = DECIMAL_CONTEXT.create_decimal(qty_str)
            if qty_val <= 0:
                logger.error(f"Invalid 'qty' value: {qty_str}. Must be positive.")
                return None
        except InvalidOperation:
            logger.error(
                f"Invalid 'qty' value format: {qty_str}. Must be a valid number string."
            )
            return None

        # Validate price format if it's a limit order or has trigger/sl/tp
        price_params = [
            "price",
            "triggerPrice",
            "stopLoss",
            "takeProfit",
            "slLimitPrice",
            "tpLimitPrice",
        ]  # Add limit prices if used
        for p_name in price_params:
            if p_name in params and params[p_name] is not None:
                price_str = params[p_name]
                if not isinstance(price_str, str) or not price_str:
                    logger.error(
                        f"Invalid '{p_name}' parameter type or empty value: {price_str}. Must be non-empty string."
                    )
                    return None
                try:
                    price_val = DECIMAL_CONTEXT.create_decimal(price_str)
                    if price_val <= 0:
                        logger.error(
                            f"Invalid '{p_name}' value: {price_str}. Must be positive."
                        )
                        return None
                except InvalidOperation:
                    logger.error(
                        f"Invalid '{p_name}' value format: {price_str}. Must be a valid number string."
                    )
                    return None

        # Add default positionIdx for derivatives if not provided (assuming One-Way mode)
        if self.category in ["linear", "inverse"] and "positionIdx" not in params:
            params["positionIdx"] = POSITION_IDX_ONE_WAY

        # --- Build Order Description for Logging ---
        order_desc_parts = [
            params["side"],
            params["orderType"],
            params["qty"],
            params["symbol"],
        ]
        if params["orderType"] == ORDER_TYPE_LIMIT and "price" in params:
            order_desc_parts.append(f"@ {params['price']}")
        if params.get("triggerPrice"):
            order_desc_parts.append(
                f"(Trig: {params['triggerPrice']} {params.get('triggerBy', '')})"
            )
        if params.get("stopLoss"):
            sl_type = f" {params['slOrderType']}" if params.get("slOrderType") else ""
            sl_trig = f" ({params['slTriggerBy']})" if params.get("slTriggerBy") else ""
            sl_limit = (
                f" Limit:{params['slLimitPrice']}" if params.get("slLimitPrice") else ""
            )
            order_desc_parts.append(
                f"SL: {params['stopLoss']}{sl_type}{sl_trig}{sl_limit}"
            )
        if params.get("takeProfit"):
            tp_trig = f" ({params['tpTriggerBy']})" if params.get("tpTriggerBy") else ""
            tp_limit = (
                f" Limit:{params['tpLimitPrice']}" if params.get("tpLimitPrice") else ""
            )  # Assuming tpOrderType is Limit implicitly or set
            order_desc_parts.append(f"TP: {params['takeProfit']}{tp_trig}{tp_limit}")
        if params.get("reduceOnly"):
            order_desc_parts.append("[ReduceOnly]")
        if params.get("timeInForce"):
            order_desc_parts.append(f"[{params['timeInForce']}]")
        if params.get("orderLinkId"):
            order_desc_parts.append(f"LinkID: {params['orderLinkId'][:10]}...")

        order_description = " ".join(order_desc_parts)
        logger.info(
            f"{Fore.YELLOW}✨ Forging Order: {order_description}...{Style.RESET_ALL}"
        )
        logger.debug(f"Order Parameters: {params}")

        # --- Place Order via API ---
        try:
            start_time = time.monotonic()
            response = self.session.place_order(**params)
            end_time = time.monotonic()
            latency = end_time - start_time
            logger.debug(f"Place Order Raw Response (took {latency:.3f}s): {response}")

            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "Unknown Error")
            result_data = response.get("result", {})
            order_id = result_data.get("orderId") if result_data else None

            # --- Handle Response ---
            if ret_code == RET_CODE_OK:
                if order_id:
                    logger.success(
                        f"{Fore.GREEN}✅ Order placed successfully! OrderID: {format_order_id(order_id)} (Latency: {latency:.3f}s){Style.RESET_ALL}"
                    )
                    return result_data
                else:
                    logger.error(
                        f"{Back.YELLOW}{Fore.BLACK}Order placement reported OK (Code: {ret_code}), but no OrderID found in result.{Style.RESET_ALL} Response: {response}"
                    )
                    return None
            else:
                # Log specific errors more clearly
                log_level = logging.ERROR
                alert_msg = None
                if ret_code in [
                    RET_CODE_INSUFFICIENT_BALANCE_SPOT,
                    RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_1,
                    RET_CODE_INSUFFICIENT_BALANCE_DERIVATIVES_2,
                    RET_CODE_REDUCE_ONLY_MARGIN_ERROR,
                ]:
                    log_level = logging.CRITICAL
                    alert_msg = f"CRITICAL: Insufficient balance for {self.symbol} {params['side']} order! ({ret_code})"
                elif ret_code == RET_CODE_QTY_TOO_SMALL:
                    alert_msg = f"ERROR: Order qty {params['qty']} too small for {self.symbol}. Min: {self.min_qty}"
                elif ret_code == RET_CODE_QTY_INVALID_PRECISION:
                    alert_msg = f"ERROR: Order qty {params['qty']} precision invalid for {self.symbol}. Step: {self.qty_step}"
                elif ret_code == RET_CODE_PRICE_TOO_LOW:
                    alert_msg = (
                        f"ERROR: Order price invalid (too low) for {self.symbol}."
                    )
                elif ret_code == RET_CODE_PRICE_INVALID_PRECISION:
                    alert_msg = f"ERROR: Order price precision invalid for {self.symbol}. Tick: {self.price_tick}"
                elif ret_code == RET_CODE_TOO_MANY_VISITS:
                    log_level = logging.WARNING
                    alert_msg = (
                        f"WARNING: Rate Limit Hit for {self.symbol} order! ({ret_code})"
                    )
                    time.sleep(1)  # Back off slightly
                elif ret_code in [
                    RET_CODE_REDUCE_ONLY_QTY_ERROR,
                    RET_CODE_REDUCE_ONLY_QTY_ERROR_UNIFIED,
                ]:
                    alert_msg = f"ERROR: ReduceOnly order failed for {self.symbol}: Qty {params['qty']} exceeds position size. ({ret_code})"
                else:
                    # Generic failure message
                    alert_msg = f"ERROR: Order placement failed for {self.symbol}! Code: {ret_code}, Msg: {ret_msg}"

                logger.log(
                    log_level, f"{Back.RED}{Fore.WHITE}❌ {alert_msg}{Style.RESET_ALL}"
                )
                logger.error(
                    f"Failed Order Parameters: {params}"
                )  # Log params again on failure
                if log_level >= logging.ERROR:  # Send SMS for errors/critical
                    send_sms_alert(alert_msg, self.sms_config)

                return None  # Return None on failure

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"{Back.RED}Pybit API Error during order placement: {pybit_e}{Style.RESET_ALL}",
                exc_info=False,
            )
            logger.error(
                f"Status Code: {pybit_e.status_code}, Response: {pybit_e.response}"
            )
            alert_msg = f"CRITICAL: Pybit API Error placing {self.symbol} order! Status:{pybit_e.status_code}"
            if pybit_e.status_code == 403 and "Timestamp" in str(pybit_e.response):
                logger.critical(
                    f"{Back.RED}Timestamp/recvWindow error. Check system clock sync and recv_window ({self.api_config.recv_window}ms).{Style.RESET_ALL}"
                )
                alert_msg += " (Timestamp/RecvWindow Error)"
            send_sms_alert(alert_msg, self.sms_config)
            return None
        except Exception as e:
            logger.error(
                f"Unexpected exception during order placement: {e}", exc_info=True
            )
            send_sms_alert(
                f"CRITICAL: Unexpected error placing {self.symbol} order: {type(e).__name__}",
                self.sms_config,
            )
            return None

    def _cancel_single_order(
        self,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        reason: str = "Strategy Action",
    ) -> bool:
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
            cancel_params: Dict[str, Any] = {
                "category": self.category,
                "symbol": self.symbol,
            }
            if order_id:
                cancel_params["orderId"] = order_id
            if order_link_id:  # Can provide both, API usually prioritizes orderId
                cancel_params["orderLinkId"] = order_link_id

            response = self.session.cancel_order(**cancel_params)
            logger.debug(f"Cancel Order ({log_id}) Raw Response: {response}")

            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "").lower()

            if ret_code == RET_CODE_OK:
                cancelled_id = response.get("result", {}).get(
                    "orderId"
                ) or response.get("result", {}).get("orderLinkId")
                logger.info(
                    f"Order {log_id} cancelled successfully (Confirmed ID: {cancelled_id})."
                )
                return True
            # Treat "order not found", "already closed/filled", "already cancelled" as success
            elif ret_code in [
                RET_CODE_ORDER_NOT_FOUND,
                RET_CODE_ORDER_NOT_FOUND_OR_CLOSED,
                RET_CODE_ORDER_CANCELLED_OR_REJECTED,
                RET_CODE_ORDER_FILLED,
            ] or any(
                msg in ret_msg
                for msg in [
                    "order does not exist",
                    "order not found",
                    "already been filled",
                    "already closed",
                    "has been cancelled or rejected",
                ]
            ):
                logger.warning(
                    f"Order {log_id} not found or already inactive (Code: {ret_code}, Msg: {ret_msg}). Assuming cancellation is effectively successful."
                )
                return True
            else:
                logger.error(
                    f"Failed to cancel order {log_id}. Code: {ret_code}, Msg: {response.get('retMsg')}"
                )
                send_sms_alert(
                    f"ERROR: Failed to cancel {self.symbol} order {log_id}! Code:{ret_code}",
                    self.sms_config,
                )
                return False

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error cancelling order {log_id}: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
            send_sms_alert(
                f"CRITICAL: API Error cancelling {self.symbol} order {log_id}! Status:{pybit_e.status_code}",
                self.sms_config,
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected exception cancelling order {log_id}: {e}", exc_info=True
            )
            send_sms_alert(
                f"CRITICAL: Unexpected error cancelling {self.symbol} order {log_id}: {type(e).__name__}",
                self.sms_config,
            )
            return False

    def _cancel_all_open_orders(self, reason: str = "Strategy Action") -> bool:
        """
        Cancels ALL open orders (regular + conditional) for the current symbol and category.

        Args:
            reason: A string describing why orders are being cancelled (for logging).

        Returns:
            True if the cancellation request was accepted (even if 0 orders were cancelled).
            False if the API call failed.
        """
        if not self.session or not self.category or not self.symbol:
            logger.error(
                "Cannot cancel all orders: Session, category, or symbol not set."
            )
            return False

        logger.info(
            f"Attempting to cancel ALL open orders for {self.symbol} ({self.category}) due to: {reason}..."
        )
        all_cancelled = True  # Assume success unless an API call fails

        # --- Cancel Regular Orders ---
        try:
            logger.debug("Cancelling regular open orders...")
            response_reg = self.session.cancel_all_orders(
                category=self.category,
                symbol=self.symbol,
                orderFilter="Order",  # Explicitly target regular orders
            )
            logger.debug(f"Cancel All Regular Orders Response: {response_reg}")
            ret_code_reg = response_reg.get("retCode")

            if ret_code_reg == RET_CODE_OK:
                cancelled_list_reg = response_reg.get("result", {}).get("list", [])
                num_cancelled = len(cancelled_list_reg) if cancelled_list_reg else 0
                logger.info(
                    f"Cancelled {num_cancelled} regular open order(s) for {self.symbol}."
                )
            else:
                logger.error(
                    f"Failed to cancel regular orders for {self.symbol}. Code: {ret_code_reg}, Msg: {response_reg.get('retMsg')}"
                )
                all_cancelled = False  # Mark failure

        except (InvalidRequestError, FailedRequestError) as pybit_e:
            logger.error(
                f"Pybit API Error cancelling regular orders: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                exc_info=False,
            )
            all_cancelled = False
        except Exception as e:
            logger.error(
                f"Unexpected exception cancelling regular orders: {e}", exc_info=True
            )
            all_cancelled = False

        # --- Cancel Conditional Orders (Stop Orders, TP/SL) ---
        # Only applicable for derivatives
        if self.category in ["linear", "inverse"]:
            try:
                logger.debug("Cancelling conditional (Stop/TP/SL) open orders...")
                response_cond = self.session.cancel_all_orders(
                    category=self.category,
                    symbol=self.symbol,
                    orderFilter="StopOrder",  # Target conditional orders (includes TP/SL set via conditional orders)
                )
                logger.debug(f"Cancel All Conditional Orders Response: {response_cond}")
                ret_code_cond = response_cond.get("retCode")

                if ret_code_cond == RET_CODE_OK:
                    cancelled_list_cond = response_cond.get("result", {}).get(
                        "list", []
                    )
                    num_cancelled = (
                        len(cancelled_list_cond) if cancelled_list_cond else 0
                    )
                    logger.info(
                        f"Cancelled {num_cancelled} conditional open order(s) for {self.symbol}."
                    )
                else:
                    logger.error(
                        f"Failed to cancel conditional orders for {self.symbol}. Code: {ret_code_cond}, Msg: {response_cond.get('retMsg')}"
                    )
                    all_cancelled = False  # Mark failure

            except (InvalidRequestError, FailedRequestError) as pybit_e:
                logger.error(
                    f"Pybit API Error cancelling conditional orders: {pybit_e} (Status: {pybit_e.status_code}, Response: {pybit_e.response})",
                    exc_info=False,
                )
                all_cancelled = False
            except Exception as e:
                logger.error(
                    f"Unexpected exception cancelling conditional orders: {e}",
                    exc_info=True,
                )
                all_cancelled = False

        # --- Final Result ---
        if not all_cancelled:
            logger.warning(
                "Cancellation of all orders encountered one or more failures."
            )
            send_sms_alert(
                f"WARNING: Failure cancelling some {self.symbol} orders during '{reason}'. Check UI!",
                self.sms_config,
            )

        # Clear tracked IDs regardless of API success, as we intended to cancel them
        self.sl_order_id = None
        self.tp_order_id = None
        logger.debug("Cleared tracked SL/TP order IDs after cancel all attempt.")

        return all_cancelled

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
            return False  # Not in a position to exit

        if df_ind is None or df_ind.empty:
            logger.warning("Cannot check exit conditions: Indicator data missing.")
            return False

        logger.debug(f"Checking exit conditions for {self.current_side} position...")
        should_exit = False
        exit_reason = ""
        try:
            # --- Exit Condition: EVT Trend Reversal ---
            evt_len = self.strategy_config.indicator_settings.evt_length
            trend_col = f"evt_trend_{evt_len}"
            if trend_col not in df_ind.columns:
                logger.error(
                    f"Cannot check exit: EVT Trend column '{trend_col}' missing."
                )
                return False

            latest_trend_val = df_ind.iloc[-1].get(trend_col)
            if pd.isna(latest_trend_val):
                logger.warning("Cannot check exit: Latest EVT trend value is NaN.")
                return False

            latest_trend = int(latest_trend_val)

            if self.current_side == POS_LONG and latest_trend == -1:
                should_exit = True
                exit_reason = "EVT Trend flipped to Short"
            elif self.current_side == POS_SHORT and latest_trend == 1:
                should_exit = True
                exit_reason = "EVT Trend flipped to Long"

            # --- Execute Exit Action ---
            if should_exit:
                position_side_display = self.current_side.upper()
                position_qty_display = format_amount(
                    self.symbol, self.current_qty, self.qty_step
                )
                logger.warning(
                    f"{Fore.YELLOW}🚨 Exit condition triggered for {position_side_display} position ({position_qty_display}): {exit_reason}{Style.RESET_ALL}"
                )

                # 1. Cancel ALL open orders for the symbol FIRST
                logger.info("Cancelling all open orders before placing exit order...")
                if not self._cancel_all_open_orders(f"Exit Triggered: {exit_reason}"):
                    logger.warning(
                        "Failed to cancel all open orders during exit. Proceeding with close attempt, but check UI for stray orders."
                    )
                    # Potentially abort exit if cancellation fails critically
                    # return False # Abort exit if cleanup fails?

                # 2. Close the position using a reduce-only market order
                close_side = SIDE_SELL if self.current_side == POS_LONG else SIDE_BUY
                close_qty_str = self._format_qty(self.current_qty)

                if not close_qty_str:
                    logger.error(
                        f"Failed to format current quantity {self.current_qty} for exit order. Cannot close position."
                    )
                    return True  # Indicate exit was attempted but failed critically

                logger.info(
                    f"Placing Market Close Order: {close_side} {close_qty_str} {self.symbol} [ReduceOnly]"
                )
                close_params: Dict[str, Any] = {
                    "category": self.category,
                    "symbol": self.symbol,
                    "side": close_side,
                    "orderType": ORDER_TYPE_MARKET,
                    "qty": close_qty_str,
                    "reduceOnly": True,
                    "timeInForce": TIME_IN_FORCE_IOC,  # Ensure it executes immediately
                }

                close_order_result = self._place_order(close_params)

                if close_order_result and close_order_result.get("orderId"):
                    closed_order_id = close_order_result["orderId"]
                    logger.success(
                        f"{Fore.GREEN}✅ Position Close Market Order ({format_order_id(closed_order_id)}) placed successfully due to: {exit_reason}{Style.RESET_ALL}"
                    )

                    alert_msg = f"[{self.symbol.split('/')[0]}] EXITED {position_side_display} ({exit_reason}). Qty: {position_qty_display}"
                    send_sms_alert(alert_msg, self.sms_config)

                    # Optimistically reset internal state - next iteration's _update_state confirms
                    self._reset_position_state(f"Exit order placed ({exit_reason})")
                    return True  # Indicate an exit occurred

                else:
                    # CRITICAL FAILURE: Failed to place the closing order
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}❌ Failed to place position Close Market Order ({exit_reason}). Manual intervention likely required!{Style.RESET_ALL}"
                    )
                    logger.info(
                        "Re-checking position state after failed close order placement..."
                    )
                    time.sleep(self.app_config.api_config.api_rate_limit_delay * 2)
                    state_updated = self._update_state()  # Re-fetch state
                    if state_updated and self.current_side == POS_NONE:
                        logger.info(
                            "Position appears closed after re-checking state. Close order might have executed despite API error response."
                        )
                        return True  # Assume closed based on re-check
                    else:
                        logger.critical(
                            f"{Back.RED}CRITICAL FAILURE TO CLOSE POSITION! State still shows {self.current_side} or failed to update. Manual intervention required!{Style.RESET_ALL}"
                        )
                        alert_msg = f"CRITICAL: Failed to CLOSE {self.symbol} {position_side_display} position on signal ({exit_reason})! Manual check needed!"
                        send_sms_alert(alert_msg, self.sms_config)
                        # Stop the bot?
                        # self.is_running = False
                        return True  # Indicate exit was attempted, even if critically failed

            else:
                return False  # No exit triggered

        except Exception as e:
            logger.error(
                f"Error checking or handling exit conditions: {e}", exc_info=True
            )
            return False

    def _handle_entry(
        self,
        signal: Literal["long", "short"],
        df_ind: pd.DataFrame,
        current_price: Decimal,
    ) -> bool:
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
            logger.debug(
                f"Ignoring {signal} entry signal: Already in a {self.current_side} position."
            )
            return False
        if df_ind is None or df_ind.empty or current_price <= 0:
            logger.warning(
                "Cannot handle entry: Missing indicators or invalid current price."
            )
            return False
        if not self.price_tick or not self.qty_step or not self.min_qty:
            logger.error(
                "Cannot enter: Missing critical market details (price_tick/qty_step/min_qty)."
            )
            return False

        signal_display = signal.upper()
        logger.info(
            f"{Fore.BLUE}Processing {signal_display} entry signal near price {format_price(self.symbol, current_price, self.price_tick)}...{Style.RESET_ALL}"
        )

        # 1. Calculate SL/TP
        sl_price, tp_price = self._calculate_sl_tp(df_ind, signal, current_price)
        if sl_price is None:
            logger.error(
                f"Cannot enter {signal_display}: Failed to calculate a valid Stop Loss price. Aborting entry."
            )
            return False

        # 2. Calculate Position Size
        position_size = self._calculate_position_size(current_price, sl_price)
        if position_size is None or position_size <= DECIMAL_CONTEXT.create_decimal(
            "0"
        ):
            logger.error(
                f"Cannot enter {signal_display}: Failed to calculate a valid position size. Aborting entry."
            )
            return False

        # 3. Format quantities and prices for API (strings)
        entry_qty_str = self._format_qty(position_size)
        sl_price_str = self._format_price_str(sl_price)
        tp_price_str = (
            self._format_price_str(tp_price) if tp_price is not None else None
        )

        if not entry_qty_str or not sl_price_str:
            logger.error(
                "Failed to format entry quantity or SL price to string. Aborting entry."
            )
            return False
        if tp_price is not None and not tp_price_str:
            logger.error(
                "Failed to format TP price to string, but TP was calculated. Aborting entry."
            )
            return False

        # 4. Prepare and Place Entry Order (Market Order)
        entry_side_str = SIDE_BUY if signal == POS_LONG else SIDE_SELL
        order_link_id = (
            f"{signal[:1]}_{self.symbol.replace('/', '')}_{int(time.time() * 1000)}"[
                -36:
            ]
        )  # Max 36 chars for V5

        entry_params: Dict[str, Any] = {
            "category": self.category,
            "symbol": self.symbol,
            "side": entry_side_str,
            "orderType": ORDER_TYPE_MARKET,
            "qty": entry_qty_str,
            "orderLinkId": order_link_id,
            # "timeInForce": TIME_IN_FORCE_IOC # Optional for market orders
        }

        # --- ATOMIC SL/TP PLACEMENT (if configured) ---
        if self.attach_sl_tp_to_entry:
            logger.info("Attempting atomic entry with attached SL/TP parameters...")
            if sl_price_str:
                entry_params["stopLoss"] = sl_price_str
                entry_params["slTriggerBy"] = self.sl_trigger_by
                entry_params["slOrderType"] = self.sl_order_type  # Market or Limit
                # If SL type is Limit, API might require slLimitPrice (often same as trigger)
                if self.sl_order_type == ORDER_TYPE_LIMIT:
                    entry_params["slLimitPrice"] = (
                        sl_price_str  # Set limit price = trigger price
                    )
                    logger.debug(
                        f"Atomic SL is Limit, setting slLimitPrice: {sl_price_str}"
                    )

            if tp_price_str:
                entry_params["takeProfit"] = tp_price_str
                entry_params["tpTriggerBy"] = self.tp_trigger_by
                entry_params["tpOrderType"] = ORDER_TYPE_LIMIT  # TP is typically Limit
                # If TP type is Limit, API might require tpLimitPrice (usually same as trigger)
                entry_params["tpLimitPrice"] = (
                    tp_price_str  # Set limit price = trigger price
                )
                logger.debug(f"Atomic TP active, setting tpLimitPrice: {tp_price_str}")

            # Note: Verify exact parameters required by Bybit V5 `place_order` for atomic SL/TP (esp. limit types)

        # --- Place the Order ---
        entry_order_result = self._place_order(entry_params)

        if not entry_order_result or not entry_order_result.get("orderId"):
            logger.error(
                f"{Back.RED}{Fore.WHITE}❌ Entry Market Order placement failed for {signal_display}.{Style.RESET_ALL}"
            )
            return False  # Entry failed

        entry_order_id = entry_order_result["orderId"]
        logger.info(
            f"Entry Market Order ({format_order_id(entry_order_id)}) placed for {signal_display}. Waiting briefly for state propagation..."
        )

        # 5. Confirm Entry State (Crucial Step)
        confirmation_delay = (
            self.app_config.api_config.api_rate_limit_delay * 5
        )  # Increased delay
        logger.debug(
            f"Waiting {confirmation_delay:.2f}s before confirming position state..."
        )
        time.sleep(confirmation_delay)

        logger.info("Attempting to confirm position state after entry...")
        state_updated = self._update_state()
        if not state_updated:
            logger.error(
                f"{Back.YELLOW}Failed to fetch updated state after placing entry order ({format_order_id(entry_order_id)}).{Style.RESET_ALL} Cannot confirm entry details. Manual check advised!"
            )
            alert_msg = f"ALERT: Failed state update after {self.symbol} {signal_display} entry attempt ({format_order_id(entry_order_id)}). Manual check!"
            send_sms_alert(alert_msg, self.sms_config)
            # Treat as failure? Order might be open without confirmation. Risky.
            # Let's assume failure for this iteration. Next iteration might recover state.
            return False

        # --- Verify State Reflects Intended Entry ---
        expected_side = signal
        if self.current_side != expected_side:
            if self.current_side == POS_NONE:
                logger.error(
                    f"Entry order ({format_order_id(entry_order_id)}) placed, but state update shows NO position. Order likely rejected, zero-filled, or closed instantly. Check Bybit Order History."
                )
            else:
                logger.critical(
                    f"{Back.RED}CRITICAL STATE MISMATCH! Entry order ({format_order_id(entry_order_id)}) placed for {expected_side}, but state update shows position is now '{self.current_side}'. Manual intervention required!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"CRITICAL: State mismatch after {self.symbol} {expected_side} entry! Position shows {self.current_side}. Manual check!",
                    self.sms_config,
                )
            # Reset state to be safe if mismatch detected
            self._reset_position_state("State mismatch after entry attempt.")
            return False  # Entry confirmation failed

        # --- Check Filled Quantity and Price ---
        # Use a small tolerance for quantity comparison if needed
        qty_tolerance = DECIMAL_CONTEXT.create_decimal("0.01")  # 1% tolerance
        min_expected_qty = position_size * (Decimal(1) - qty_tolerance)
        if self.current_qty < min_expected_qty:
            logger.warning(
                f"Potential partial fill or state discrepancy for entry {format_order_id(entry_order_id)}. Ordered: {position_size}, Actual Position Qty: {self.current_qty}. Proceeding with actual quantity."
            )

        actual_entry_price = self.entry_price
        if actual_entry_price is None:
            logger.error(
                "Position confirmed, but failed to get the average entry price from state update. SL/TP accuracy might be slightly affected if placed separately."
            )

        actual_qty_display = format_amount(self.symbol, self.current_qty, self.qty_step)
        actual_entry_price_display = (
            format_price(self.symbol, actual_entry_price, self.price_tick)
            if actual_entry_price
            else "N/A"
        )

        logger.success(
            f"{Fore.GREEN}✅ Entry Confirmed: {self.current_side.upper()} {actual_qty_display} {self.base_coin} @ ~{actual_entry_price_display}{Style.RESET_ALL}"
        )

        alert_msg = f"[{self.symbol.split('/')[0]}] ENTERED {self.current_side.upper()} {actual_qty_display} @ ~{actual_entry_price_display}"
        send_sms_alert(alert_msg, self.sms_config)

        # 6. Place SL/TP Orders SEPARATELY (if not attached atomically)
        if not self.attach_sl_tp_to_entry:
            logger.info("Placing Stop Loss and Take Profit orders separately...")
            # Re-calculate SL/TP based on the *actual* confirmed entry price for better accuracy
            if actual_entry_price:
                logger.debug(
                    f"Re-calculating SL/TP based on actual entry price: {actual_entry_price_display}"
                )
                sl_price_final, tp_price_final = self._calculate_sl_tp(
                    df_ind, signal, actual_entry_price
                )
                if sl_price_final is None:
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to calculate FINAL SL price after entry confirmation. POSITION IS OPEN WITHOUT STOP LOSS! Manual intervention required!{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"CRITICAL: Failed place SL for {self.symbol} {self.current_side.upper()} pos! Manual SL required!",
                        self.sms_config,
                    )
                    # Attempt emergency close?
                    # self._close_position_immediately("Failed Final SL Calculation")
                    return True  # Entry occurred, but subsequent critical failure.

                sl_price_str_final = self._format_price_str(sl_price_final)
                tp_price_str_final = (
                    self._format_price_str(tp_price_final)
                    if tp_price_final is not None
                    else None
                )
            else:
                logger.warning(
                    "Actual entry price not available, using initially calculated SL/TP for separate orders."
                )
                sl_price_str_final = sl_price_str
                tp_price_str_final = tp_price_str

            position_qty_str_final = self._format_qty(
                self.current_qty
            )  # Use actual filled qty
            if not position_qty_str_final or not sl_price_str_final:
                logger.critical(
                    f"{Back.RED}CRITICAL: Failed to format final quantity or SL price for separate orders. POSITION OPEN WITHOUT SL/TP! Manual intervention!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"CRITICAL: Format error placing SL/TP for {self.symbol} {self.current_side.upper()}! Manual check!",
                    self.sms_config,
                )
                return True  # Entry occurred, critical failure follows

            sl_order_id_placed, tp_order_id_placed = self._place_separate_sl_tp_orders(
                sl_price_str=sl_price_str_final,
                tp_price_str=tp_price_str_final,
                position_qty_str=position_qty_str_final,
            )

            if sl_order_id_placed:
                self.sl_order_id = sl_order_id_placed
                logger.info(
                    f"Separate SL order ({format_order_id(self.sl_order_id)}) placed."
                )
            else:
                logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL: Failed to place separate SL order after entry confirmation. POSITION IS OPEN WITHOUT STOP LOSS! Manual intervention required!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"CRITICAL: Failed place separate SL for {self.symbol} {self.current_side.upper()} pos! Manual SL required!",
                    self.sms_config,
                )

            if tp_order_id_placed:
                self.tp_order_id = tp_order_id_placed
                logger.info(
                    f"Separate TP order ({format_order_id(self.tp_order_id)}) placed."
                )
            elif tp_price_final is not None:
                logger.warning("Failed to place separate TP order after entry.")

        else:
            self.sl_order_id = None  # Ensure cleared if atomic
            self.tp_order_id = None
            logger.info("SL/TP were attached to the entry order (atomic placement).")

        return True  # Indicate entry process completed

    def _place_separate_sl_tp_orders(
        self, sl_price_str: str, tp_price_str: Optional[str], position_qty_str: str
    ) -> Tuple[Optional[str], Optional[str]]:
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

        if self.current_side == POS_NONE:
            logger.error("Cannot place separate SL/TP: Not currently in a position.")
            return None, None
        if not self.session:
            logger.error("Cannot place separate SL/TP: Session not initialized.")
            return None, None

        exit_side = SIDE_SELL if self.current_side == POS_LONG else SIDE_BUY
        sl_link_id = f"sl_{self.symbol.replace('/', '')}_{int(time.time() * 1000)}"[
            -36:
        ]
        tp_link_id = f"tp_{self.symbol.replace('/', '')}_{int(time.time() * 1000)}"[
            -36:
        ]

        # --- Place Stop Loss Order (Conditional) ---
        logger.info(
            f"Placing Separate Stop Loss ({self.sl_order_type}) order: {exit_side} {position_qty_str} @ Trigger {sl_price_str}"
        )
        sl_params: Dict[str, Any] = {
            "category": self.category,
            "symbol": self.symbol,
            "side": exit_side,
            "orderType": self.sl_order_type,  # Market or Limit
            "qty": position_qty_str,
            "triggerPrice": sl_price_str,
            "triggerBy": self.sl_trigger_by,
            "reduceOnly": True,
            # "timeInForce": TIME_IN_FORCE_GTC, # GTC often implied for stops, check API
            "orderLinkId": sl_link_id,
            # For V5, conditional orders might be placed via place_order with trigger fields
            # Or via place_conditional_order endpoint (check pybit implementation)
            # Let's assume place_order works with triggerPrice for now.
            # Need to potentially add `stopOrderType` or `tpslMode`/`slOrderType` if required by V5 place_order
            "slOrderType": self.sl_order_type,  # Explicitly add if supported by place_order
        }
        if self.sl_order_type == ORDER_TYPE_LIMIT:
            sl_params["price"] = sl_price_str  # Set limit price for the triggered order
            sl_params["slLimitPrice"] = (
                sl_price_str  # Also set the specific SL limit price field if needed
            )
            logger.info(f"SL is Limit type, setting limit price: {sl_price_str}")

        # Use place_order assuming it handles conditional logic via triggerPrice
        sl_order_result = self._place_order(sl_params)
        if sl_order_result and sl_order_result.get("orderId"):
            sl_order_id = sl_order_result["orderId"]
        elif sl_order_result:  # OK response but no ID?
            logger.error(
                f"Separate SL order placement returned OK but missing OrderID. Result: {sl_order_result}"
            )

        # --- Place Take Profit Order (Conditional Limit) ---
        if tp_price_str is not None:
            logger.info(
                f"Placing Separate Take Profit (Limit) order: {exit_side} {position_qty_str} @ Limit {tp_price_str}"
            )
            # TP is often a conditional limit order triggered by price
            tp_params: Dict[str, Any] = {
                "category": self.category,
                "symbol": self.symbol,
                "side": exit_side,
                "orderType": ORDER_TYPE_LIMIT,  # TP order itself is Limit
                "qty": position_qty_str,
                "price": tp_price_str,  # The limit price for the TP order
                "triggerPrice": tp_price_str,  # Trigger at the TP price
                "triggerBy": self.tp_trigger_by,
                "reduceOnly": True,
                # "timeInForce": TIME_IN_FORCE_GTC, # GTC often implied
                "orderLinkId": tp_link_id,
                # Add specific TP fields if required by place_order for conditional TP
                "tpOrderType": ORDER_TYPE_LIMIT,  # Explicitly add if supported
                "tpLimitPrice": tp_price_str,  # Explicitly add if supported
            }
            tp_order_result = self._place_order(tp_params)
            if tp_order_result and tp_order_result.get("orderId"):
                tp_order_id = tp_order_result["orderId"]
            elif tp_order_result:
                logger.error(
                    f"Separate TP order placement returned OK but missing OrderID. Result: {tp_order_result}"
                )

        return sl_order_id, tp_order_id

    def _close_position_immediately(self, reason: str):
        """Places an immediate market order to close the current position."""
        if self.current_side == POS_NONE:
            logger.info(
                f"Request to close position immediately, but already flat ({reason})."
            )
            return True
        if not self.session or not self.category or not self.symbol:
            logger.error(
                f"Cannot close position immediately: Session/category/symbol missing ({reason})."
            )
            return False

        logger.warning(
            f"{Fore.YELLOW}🚨 Attempting IMMEDIATE MARKET CLOSE of {self.current_side} position due to: {reason}{Style.RESET_ALL}"
        )

        # 1. Cancel all orders first
        logger.info("Cancelling all open orders before emergency close...")
        if not self._cancel_all_open_orders(f"Emergency Close: {reason}"):
            logger.warning(
                "Failed to cancel all orders during emergency close. Proceeding with close attempt anyway."
            )

        # 2. Place ReduceOnly Market Order
        close_side = SIDE_SELL if self.current_side == POS_LONG else SIDE_BUY
        close_qty_str = self._format_qty(self.current_qty)
        if not close_qty_str:
            logger.critical(
                f"Failed to format current quantity {self.current_qty} for emergency close order ({reason}). Cannot close."
            )
            send_sms_alert(
                f"CRITICAL: Format error during emergency close of {self.symbol}! Manual check!",
                self.sms_config,
            )
            return False

        logger.info(
            f"Placing Emergency Market Close Order: {close_side} {close_qty_str} {self.symbol} [ReduceOnly]"
        )
        close_params: Dict[str, Any] = {
            "category": self.category,
            "symbol": self.symbol,
            "side": close_side,
            "orderType": ORDER_TYPE_MARKET,
            "qty": close_qty_str,
            "reduceOnly": True,
            "timeInForce": TIME_IN_FORCE_IOC,
        }
        close_order_result = self._place_order(close_params)

        if close_order_result and close_order_result.get("orderId"):
            closed_order_id = close_order_result["orderId"]
            logger.success(
                f"{Fore.GREEN}✅ Emergency Close Market Order ({format_order_id(closed_order_id)}) placed successfully ({reason}).{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"INFO: Emergency close of {self.symbol} {self.current_side} position executed ({reason}).",
                self.sms_config,
            )
            self._reset_position_state(f"Emergency close order placed ({reason})")
            # Re-check state immediately to confirm closure
            time.sleep(self.app_config.api_config.api_rate_limit_delay * 2)
            self._update_state()
            if self.current_side == POS_NONE:
                logger.info("Emergency close confirmed by state update.")
                return True
            else:
                logger.critical(
                    f"{Back.RED}Emergency close order placed, but state still shows position {self.current_side}! Manual check required!{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"CRITICAL: Emergency close of {self.symbol} FAILED TO CONFIRM! Manual check!",
                    self.sms_config,
                )
                return False
        else:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}❌ Failed to place Emergency Close Market Order ({reason}). Manual intervention required!{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"CRITICAL: Failed to place emergency close order for {self.symbol}! Manual check!",
                self.sms_config,
            )
            return False

    def run_iteration(self):
        """
        Executes a single iteration of the strategy logic.
        """
        if not self.is_initialized or not self.session:
            logger.error(
                "Strategy not initialized or session lost. Cannot run iteration."
            )
            self.is_running = False
            return

        iteration_start_time = time.monotonic()
        current_time_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.info(
            f"{Fore.MAGENTA}--- New Strategy Iteration ({current_time_utc}) ---{Style.RESET_ALL}"
        )

        try:
            # 1. Update State
            if not self._update_state():
                logger.warning(
                    "Failed to update state successfully. Skipping this iteration's logic."
                )
                return

            # 2. Fetch Market Data
            ohlcv_df, current_price = self._fetch_data()
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(
                    "Failed to fetch valid OHLCV data. Skipping indicator calculation and logic."
                )
                return
            # current_price can be None, handled downstream

            # 3. Calculate Indicators
            df_with_indicators = self._calculate_indicators(ohlcv_df)
            if df_with_indicators is None:
                logger.warning(
                    "Failed indicator calculation or validation. Skipping trading logic."
                )
                return

            # --- Core Trading Logic ---
            # 4. Check Exit Conditions (if in position)
            exit_triggered = False
            if self.current_side != POS_NONE:
                exit_triggered = self._handle_exit(df_with_indicators)
                if exit_triggered:
                    logger.info("Exit handled in this iteration.")

            # 5. Generate & Handle Entry Signals (if flat and no exit triggered)
            if self.current_side == POS_NONE and not exit_triggered:
                if current_price is None:
                    logger.warning("Cannot attempt entry: Current price is missing.")
                else:
                    entry_signal = self._generate_signals(df_with_indicators)
                    if entry_signal:
                        entry_handled = self._handle_entry(
                            entry_signal, df_with_indicators, current_price
                        )
                        if entry_handled:
                            logger.info("Entry handled in this iteration.")
                        # else: # Failure logged within _handle_entry
                        # logger.info("Entry signal generated but entry process failed or was aborted.")
                    # else: # No signal generated is normal, reduce log noise
                    # logger.info("No entry signal generated this iteration.")
            elif self.current_side != POS_NONE and not exit_triggered:
                # Monitor existing position
                pos_qty_display = format_amount(
                    self.symbol, self.current_qty, self.qty_step
                )
                entry_price_display = (
                    format_price(self.symbol, self.entry_price, self.price_tick)
                    if self.entry_price
                    else "N/A"
                )
                logger.info(
                    f"Monitoring {self.current_side.upper()} position ({pos_qty_display} @ {entry_price_display}). Waiting for exit signal or SL/TP."
                )
                # Add Trailing Stop Logic here if implemented

        except Exception as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}💥 Critical unexpected error during strategy iteration: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            alert_msg = (
                f"CRITICAL Error in {self.symbol} strategy loop: {type(e).__name__}"
            )
            send_sms_alert(alert_msg, self.sms_config)
            # Stop the bot on critical loop errors?
            # self.is_running = False

        finally:
            iteration_end_time = time.monotonic()
            elapsed = iteration_end_time - iteration_start_time
            logger.info(
                f"{Fore.MAGENTA}--- Iteration Complete (Took {elapsed:.3f}s) ---{Style.RESET_ALL}"
            )

    def start(self):
        """Initializes the strategy and starts the main execution loop."""
        logger.info("Initiating strategy startup sequence...")
        if not self._initialize():
            logger.critical(
                f"{Back.RED}Strategy initialization failed. Cannot start the arcane loop.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"CRITICAL: {self.symbol} strategy FAILED TO INITIALIZE!",
                self.sms_config,
            )
            return

        self.is_running = True
        loop_delay = self.strategy_config.loop_delay_seconds
        logger.success(
            f"{Fore.GREEN}{Style.BRIGHT}🚀 Strategy ritual commenced for {self.symbol} ({self.timeframe}). Loop delay: {loop_delay}s{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"INFO: {self.symbol} strategy started. TF:{self.timeframe}, LoopDelay:{loop_delay}s",
            self.sms_config,
        )

        while self.is_running:
            loop_start_time = time.monotonic()
            self.run_iteration()
            loop_end_time = time.monotonic()
            elapsed = loop_end_time - loop_start_time
            sleep_duration = max(0, loop_delay - elapsed)

            if not self.is_running:
                logger.info("Stop signal received during iteration. Exiting loop.")
                break

            if sleep_duration > 0:
                logger.debug(f"Sleeping for {sleep_duration:.2f}s...")
                time.sleep(sleep_duration)
            else:
                logger.warning(
                    f"Iteration took longer ({elapsed:.2f}s) than loop delay ({loop_delay}s). Running next iteration immediately."
                )

        logger.warning(
            f"{Fore.YELLOW}Strategy loop has been terminated.{Style.RESET_ALL}"
        )
        self.stop(
            initiated_by_user=False
        )  # Ensure cleanup, mark as not user-initiated stop

    def stop(self, initiated_by_user: bool = True):
        """Stops the strategy loop and performs cleanup actions."""
        if not self.is_running and self.session is None:
            logger.debug("Stop called but strategy already seems stopped.")
            return

        logger.warning(
            f"{Fore.YELLOW}--- Initiating Strategy Shutdown ({'User Request' if initiated_by_user else 'Internal Stop'}) ---{Style.RESET_ALL}"
        )
        run_state_before_stop = self.is_running
        self.is_running = False  # Signal the main loop to stop

        # --- Final Cleanup Actions ---
        logger.info("Attempting final cleanup...")
        final_position_check_possible = self.session and self.category and self.symbol

        # Option to close position on stop
        if self.close_on_stop and final_position_check_possible:
            logger.warning(
                "Config 'close_on_stop' is True. Checking final position state..."
            )
            self._update_state()  # Get latest state before deciding to close
            if self.current_side != POS_NONE:
                logger.warning(
                    f"Attempting to close open {self.current_side} position due to 'close_on_stop' config..."
                )
                self._close_position_immediately("Strategy Stop with CloseOnStop")
            else:
                logger.info("No open position found to close on stop.")
        else:
            # If not closing on stop, still cancel orders and log final state
            if final_position_check_possible:
                logger.info("Cancelling any remaining open orders...")
                if not self._cancel_all_open_orders("Strategy Stop"):
                    logger.warning(
                        "Final order cancellation encountered issues. Manual check of Bybit UI advised."
                    )
                # Check final state after cancellations
                logger.info("Performing final position state check...")
                self._update_state()
                if self.current_side != POS_NONE:
                    pos_qty_display = format_amount(
                        self.symbol, self.current_qty, self.qty_step
                    )
                    entry_price_display = (
                        format_price(self.symbol, self.entry_price, self.price_tick)
                        if self.entry_price
                        else "N/A"
                    )
                    warning_msg = f"Strategy stopped with an OPEN {self.current_side.upper()} position for {self.symbol} ({pos_qty_display} @ {entry_price_display}). Manual management may be required."
                    logger.warning(
                        f"{Back.YELLOW}{Fore.BLACK}{warning_msg}{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"ALERT: {self.symbol} strategy stopped with OPEN {self.current_side.upper()} position!",
                        self.sms_config,
                    )
                else:
                    logger.info(
                        "Strategy stopped while flat (no open position detected)."
                    )
            else:
                logger.warning(
                    "Skipping final order cancellation/state check: Session/category/symbol not available."
                )

        # --- Release Resources ---
        self._safe_close_session()

        if run_state_before_stop:  # Only send stop alert if it was actually running
            stop_reason = "User Request" if initiated_by_user else "Internal Stop"
            send_sms_alert(
                f"INFO: {self.symbol} strategy stopped ({stop_reason}).",
                self.sms_config,
            )
        logger.info(f"{Fore.CYAN}--- Strategy shutdown complete ---{Style.RESET_ALL}")

    def _safe_close_session(self):
        """Safely handles session cleanup."""
        if self.session:
            logger.info("Cleaning up Pybit HTTP session...")
            self.session = None
            logger.info("Pybit session cleared.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print(
        f"{Fore.CYAN}{Style.BRIGHT}--- Bybit EVT Strategy Script (Pybit Enhanced - HTTP) ---{Style.RESET_ALL}"
    )
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"Pybit Version: {pybit_version}")

    app_config: Optional[AppConfig] = None
    strategy: Optional[EhlersStrategyPybitEnhanced] = None

    try:
        # --- Load Configuration ---
        print(f"{Fore.BLUE}Summoning configuration spirits...{Style.RESET_ALL}")
        app_config = load_config()
        if not app_config:
            print(
                f"{Back.RED}{Fore.WHITE}FATAL: Configuration loading failed.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"{Fore.GREEN}Configuration spirits summoned successfully.{Style.RESET_ALL}"
        )

        # --- Setup Logging ---
        print(f"{Fore.BLUE}Awakening Neon Logger...{Style.RESET_ALL}")
        log_conf = app_config.logging_config
        # Remove the temporary handler added earlier
        logger.removeHandler(temp_handler)
        # Configure logger using settings from the loaded config
        logger = setup_logger(
            logger_name=log_conf.logger_name,
            log_file=log_conf.log_file,
            console_level_str=log_conf.console_level_str,
            file_level_str=log_conf.file_level_str,
            log_rotation_bytes=log_conf.log_rotation_bytes,
            log_backup_count=log_conf.log_backup_count,
            third_party_log_level_str=log_conf.third_party_log_level_str,
        )
        # Re-get the logger by name after setup to ensure we have the configured one
        logger = logging.getLogger(log_conf.logger_name)
        logger.info(
            f"{Fore.MAGENTA}--- Neon Logger Awakened and Configured ---{Style.RESET_ALL}"
        )
        logger.info(
            f"Logging to Console Level: {log_conf.console_level_str}, File Level: {log_conf.file_level_str}"
        )
        logger.info(f"Log File: {log_conf.log_file}")

        logger.info(
            f"Using config: Testnet={app_config.api_config.testnet_mode}, Symbol={app_config.api_config.symbol}, Timeframe={app_config.strategy_config.timeframe}"
        )

        # --- Instantiate Strategy ---
        logger.info("Creating enhanced strategy instance...")
        strategy = EhlersStrategyPybitEnhanced(app_config)
        logger.info("Strategy instance created.")

        # --- Start Strategy ---
        logger.info("Initiating strategy start sequence...")
        strategy.start()  # Enters the main loop

    except KeyboardInterrupt:
        logger.warning(
            f"{Fore.YELLOW}{Style.BRIGHT}>>> Manual interruption detected (Ctrl+C)! Initiating graceful shutdown...{Style.RESET_ALL}"
        )
        # Strategy stop is handled in the finally block, initiated_by_user=True

    except SystemExit as e:
        logger.info(f"System exiting with code {e.code}.")
        # No further action needed here, finally block runs if necessary

    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}💥 UNHANDLED CRITICAL ERROR in main execution block: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        if app_config and app_config.sms_config:
            try:
                error_type = type(e).__name__
                send_sms_alert(
                    f"CRITICAL FAILURE: Unhandled error in {app_config.api_config.symbol} strategy main block: {error_type}! Bot stopping.",
                    app_config.sms_config,
                )
            except Exception as alert_e:
                logger.error(f"Failed to send critical error SMS alert: {alert_e}")
        sys.exit(1)

    finally:
        # --- Graceful Shutdown ---
        logger.info("Entering final cleanup phase...")
        if strategy:  # Check if strategy object was successfully created
            logger.info("Requesting strategy stop...")
            strategy.stop(
                initiated_by_user=isinstance(sys.exc_info()[1], KeyboardInterrupt)
            )  # Pass True if stopped by Ctrl+C
        else:
            logger.info("Strategy object not created, skipping strategy stop call.")

        logger.info(
            f"{Fore.CYAN}{Style.BRIGHT}--- Strategy script enchantment fades. Returning to the digital ether... ---{Style.RESET_ALL}"
        )
        logging.shutdown()
