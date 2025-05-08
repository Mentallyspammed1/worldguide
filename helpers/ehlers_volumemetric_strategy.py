# ehlers_volumetric_strategy.py
import asyncio
import logging
import time
from decimal import Decimal

import pandas as pd  # Required by indicators.py

# --- Import Custom Modules & Helpers ---
try:
    # Exchange Interaction Helpers (Async)
    import bybit_helpers as bybit

    # Indicator Calculations (Sync)
    import indicators as ind
    from bybit_helpers import (
        Category,
        OrderFilter,
        PositionIdx,
        Side,
        TimeInForce,
        TriggerBy,
        TriggerDirection,
    )
    from bybit_helpers import (
        Config as HelperConfig,  # Avoid clash if needed
    )

    # Utility Functions (Sync)
    from bybit_utils import (
        format_amount,
        format_order_id,
        format_price,
        safe_decimal_conversion,
        send_sms_alert,  # Use cautiously in async code
    )

    # Colorama for logging (check availability)
    try:
        from colorama import Back, Fore, Style

        COLORAMA_AVAILABLE = True
    except ImportError:

        class DummyColor:  # type: ignore
            def __getattr__(self, name: str) -> str:
                return ""

        Fore = Style = Back = DummyColor()  # type: ignore
        COLORAMA_AVAILABLE = False

except ImportError as e:
    # Define dummies here too in case colorama itself failed but helpers are needed
    class DummyColorImportError:  # type: ignore
        def __getattr__(self, name: str) -> str:
            return ""

    ForeImportError = StyleImportError = BackImportError = DummyColorImportError()  # type: ignore
    err_back = BackImportError.RED if COLORAMA_AVAILABLE else ""  # Use dummies if needed
    err_fore = ForeImportError.WHITE if COLORAMA_AVAILABLE else ""
    reset_all = StyleImportError.RESET_ALL if COLORAMA_AVAILABLE else ""
    print(f"{err_back}{err_fore}Strategy FATAL: Failed to import helper modules: {e}{reset_all}")
    print("Ensure neon_logger.py, bybit_helpers.py, indicators.py, bybit_utils.py are present.")
    exit(1)

# --- Strategy Class ---


class EhlersVolumetricStrategy:
    """A trading strategy using the Ehlers Volumetric Trend (EVT) indicator.
    - Enters LONG on EVT Buy signal (`evt_buy_LEN` == True).
    - Enters SHORT on EVT Sell signal (`evt_sell_LEN` == True).
    - Exits position when the EVT trend (`evt_trend_LEN`) reverses against the position.
    - Uses ATR-based stop loss.
    - Calculates position size based on risk percentage.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        """Initializes the strategy instance.

        Args:
            config: A dictionary containing all configuration sections (API, Strategy, Logging, SMS).
            logger: An initialized logging.Logger instance.
        """
        self.config = config
        self.logger = logger
        self.api_config = config.get("API_CONFIG", {})
        self.strategy_config = config.get("STRATEGY_CONFIG", {})
        self.sms_config = config.get("SMS_CONFIG", {})  # For alerts

        # --- Essential Config Validation ---
        required_api = ["SYMBOL", "TESTNET_MODE", "POS_NONE", "POS_LONG", "POS_SHORT", "POSITION_QTY_EPSILON"]
        required_strategy = [
            "timeframe",
            "leverage",
            "indicator_settings",
            "analysis_flags",
            "risk_per_trade",
            "stop_loss_atr_multiplier",
            "position_idx",
            "EVT_ENABLED",
            "EVT_LENGTH",
            "STOP_LOSS_ATR_PERIOD",
        ]
        if not all(k in self.api_config for k in required_api):
            missing = [k for k in required_api if k not in self.api_config]
            self.logger.critical(f"Missing essential API configuration keys: {missing}. Check config.py.")
            raise ValueError("Incomplete API configuration.")
        if not all(k in self.strategy_config for k in required_strategy):
            missing = [k for k in required_strategy if k not in self.strategy_config]
            self.logger.critical(f"Missing essential Strategy configuration keys: {missing}. Check config.py.")
            raise ValueError("Incomplete Strategy configuration.")
        if not self.strategy_config.get("EVT_ENABLED", False) or not self.strategy_config["analysis_flags"].get(
            "use_evt"
        ):
            self.logger.critical(
                "Ehlers Volumetric strategy requires 'EVT_ENABLED' in strategy_config AND 'use_evt' in analysis_flags to be True."
            )
            raise ValueError("EVT indicator not enabled in configuration.")
        if not self.strategy_config["analysis_flags"].get("use_atr"):
            self.logger.critical("ATR-based stop loss requires 'use_atr' flag to be True in analysis_flags.")
            raise ValueError("ATR required for SL but not enabled in config.")

        self.symbol = self.api_config["SYMBOL"]
        self.timeframe = self.strategy_config["timeframe"]
        self.leverage = self.strategy_config["leverage"]
        # Ensure position_idx is the correct Enum type if helpers expect it
        self.position_idx = self.strategy_config.get("position_idx", PositionIdx.ONE_WAY)
        if not isinstance(self.position_idx, PositionIdx):
            try:
                self.position_idx = PositionIdx(int(self.position_idx))  # Try converting from int
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid position_idx format '{self.strategy_config.get('position_idx')}'. Defaulting to ONE_WAY (0)."
                )
                self.position_idx = PositionIdx.ONE_WAY

        # --- Exchange and Market Info (Initialized Async) ---
        self.exchange: bybit.ccxt.bybit | None = None
        self.market_info: dict | None = None
        self.min_qty = Decimal("0.000001")  # Smallest possible default
        self.qty_step = Decimal("0.000001")
        self.price_tick = Decimal("0.000001")

        # --- Strategy State ---
        self.current_position: dict | None = None
        self.open_orders: dict[str, dict] = {}
        self.last_known_price = Decimal("0")
        self.available_balance = Decimal("0")
        self.is_running = False
        self.stop_loss_order_id: str | None = None

        # --- Indicator Config Convenience & Column Names ---
        self.indicator_settings = self.strategy_config.get("indicator_settings", {})
        self.analysis_flags = self.strategy_config.get("analysis_flags", {})
        self.min_data_periods = self.indicator_settings.get("min_data_periods", 100)
        self.evt_length = self.indicator_settings.get("evt_length", 7)
        self.atr_period = self.indicator_settings.get("atr_period", 14)

        # Define expected column names based on config and indicators.py naming convention
        self.evt_trend_col = f"evt_trend_{self.evt_length}"
        self.evt_buy_col = f"evt_buy_{self.evt_length}"
        self.evt_sell_col = f"evt_sell_{self.evt_length}"
        self.atr_col = f"ATRr_{self.atr_period}"

        self.required_indicators = []
        if self.analysis_flags.get("use_evt"):
            self.required_indicators.extend([self.evt_trend_col, self.evt_buy_col, self.evt_sell_col])
        if self.analysis_flags.get("use_atr"):
            self.required_indicators.append(self.atr_col)

        self.logger.info(f"Strategy {self.strategy_config.get('name', 'EhlersVolumetric')} initialized.")
        self.logger.debug(f"Required indicators: {self.required_indicators}")

    async def _initialize(self) -> bool:
        """Asynchronously initializes the exchange connection, loads market data,
        sets leverage, fetches initial state, and performs pre-run cleanup.

        Returns:
            bool: True if initialization succeeded, False otherwise.
        """
        self.logger.info(f"{Style.BRIGHT}--- Strategy Initialization Phase ---{Style.RESET_ALL}")
        try:
            # 1. Initialize Exchange Connection (Async)
            self.exchange = await bybit.initialize_bybit(self.api_config)
            if not self.exchange:
                self.logger.critical("Failed to initialize Bybit exchange connection. Stopping.")
                return False

            # 2. Load/Verify Market Info (Uses cache populated by initialize_bybit)
            self.market_info = bybit.market_cache.get_market(self.symbol)
            if not self.market_info:
                self.logger.critical(f"Market info not found for {self.symbol}. Was it loaded? Stopping.")
                return False
            self._extract_market_details()  # Set precision, limits etc.

            # 3. Set Leverage (Async)
            if self.leverage > 0:
                self.logger.info(f"Setting leverage for {self.symbol} to {self.leverage}x...")
                leverage_set = await bybit.set_leverage(self.exchange, self.symbol, self.leverage, self.api_config)
                if not leverage_set:
                    self.logger.warning(
                        f"{Fore.YELLOW}Failed to set leverage to {self.leverage}x. Check API permissions or existing position/orders. Strategy will proceed with current setting.{Style.RESET_ALL}"
                    )
                else:
                    self.logger.success(f"{Fore.GREEN}Leverage set confirmed.{Style.RESET_ALL}")

            # 4. Fetch Initial State (Async)
            self.logger.info("Fetching initial account state (position, orders, balance)...")
            await self._update_state()  # Handles internal errors
            pos_side = (
                self.current_position.get("side", self.api_config["POS_NONE"])
                if self.current_position
                else self.api_config["POS_NONE"]
            )
            pos_qty = self.current_position.get("qty", Decimal(0)) if self.current_position else Decimal(0)
            self.logger.info(f"Initial Position: Side={pos_side}, Qty={pos_qty}")
            self.logger.info(f"Initial Open Orders: {len(self.open_orders)}")
            self.logger.info(f"Initial Available Balance: {self.available_balance:.4f} USDT")

            # 5. Initial Order Cleanup (Optional but Recommended)
            self.logger.info("Performing initial cleanup: cancelling existing orders...")
            category = bybit.market_cache.get_category(self.symbol)  # Needed for V5
            if category:  # Proceed only if category is known
                cancelled_count = await bybit.cancel_all_orders(
                    self.exchange, symbol=self.symbol, category=category, config=self.api_config, reason="Init Cleanup"
                )
                if cancelled_count is not None:
                    self.logger.info(f"Cancelled {cancelled_count} existing orders for {self.symbol}.")
                    await asyncio.sleep(1)  # Brief pause for exchange to process
                    await self._update_state()  # Refresh state after cleanup
                else:
                    self.logger.warning("Failed to execute cancel_all_orders during init (check logs).")
            else:
                self.logger.error(f"Cannot perform initial order cleanup: Category unknown for {self.symbol}.")

            self.logger.success(f"{Style.BRIGHT}--- Strategy Initialization Complete ---{Style.RESET_ALL}")
            return True

        except Exception as e:
            self.logger.critical(
                f"{Back.RED}{Fore.WHITE}Critical error during strategy initialization: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            # Ensure exchange is closed if partially initialized
            if self.exchange and hasattr(self.exchange, "close") and not self.exchange.closed:
                try:
                    await self.exchange.close()
                except Exception:
                    pass  # Ignore errors during cleanup
            return False

    def _extract_market_details(self):
        """Extracts and sets precision and limit values from market info."""
        if not self.market_info:
            self.logger.error("Cannot extract market details: market_info is None.")
            return

        try:
            limits = self.market_info.get("limits", {})
            amount_limits = limits.get("amount", {})
            precision = self.market_info.get("precision", {})

            min_qty_str = amount_limits.get("min")
            # CCXT precision['amount'] for Bybit V5 IS the step size (as string)
            qty_step_str = precision.get("amount")
            price_tick_str = precision.get("price")  # This IS the tick size (as string)

            self.min_qty = max(
                Decimal("1E-8"),  # Absolute minimum > 0
                safe_decimal_conversion(min_qty_str, Decimal("0.000001")),
            )
            self.qty_step = safe_decimal_conversion(qty_step_str, Decimal("0.000001"))
            self.price_tick = safe_decimal_conversion(
                price_tick_str, Decimal("0.01")
            )  # Default tick if conversion fails

            if self.qty_step <= 0 or self.price_tick <= 0:
                raise ValueError(
                    f"Parsed invalid step/tick size (QtyStep: {self.qty_step}, PriceTick: {self.price_tick})"
                )

            self.logger.info(
                f"Market Details Set: Min Qty={self.min_qty}, Qty Step={self.qty_step}, Price Tick={self.price_tick}"
            )

        except Exception as e:
            self.logger.error(f"Failed to parse market details for {self.symbol}: {e}", exc_info=True)
            self.logger.warning("Using default market details, order sizing/pricing may be inaccurate.")
            # Keep the small defaults set in __init__

    async def _update_state(self):
        """Fetches current position, open orders, balance, and ticker price."""
        self.logger.debug("Updating strategy state...")
        if not self.exchange:  # Safety check
            self.logger.error("Cannot update state: Exchange not initialized.")
            return
        try:
            # Use asyncio.gather for concurrent fetching
            results = await asyncio.gather(
                bybit.get_current_position_bybit_v5(self.exchange, self.symbol, self.api_config),
                self._fetch_all_open_orders(),  # Custom helper to get all order types
                bybit.fetch_usdt_balance(self.exchange, self.api_config),
                bybit.fetch_ticker_validated(self.exchange, self.symbol, self.api_config),
                return_exceptions=True,  # Don't let one failure stop others
            )

            # Process results, handling potential exceptions
            pos_data, open_orders_list, balance_tuple, ticker = results

            # Position
            if isinstance(pos_data, Exception):
                self.logger.error(f"Failed to fetch position state: {pos_data}")
            elif isinstance(pos_data, list):  # Hedge mode
                self.logger.debug("Hedge mode detected by position fetch.")
                found_pos = next(
                    (p for p in pos_data if p.get("misc", {}).get("positionIdx") == self.position_idx.value), None
                )
                self.current_position = found_pos  # Assign found pos or None
            else:  # One-way mode or None returned
                self.current_position = pos_data

            # Open Orders
            if isinstance(open_orders_list, Exception):
                self.logger.error(f"Failed to fetch open orders: {open_orders_list}")
            else:
                self.open_orders = {o["id"]: o for o in open_orders_list}
                # Prune tracked SL order ID if it's no longer open
                if self.stop_loss_order_id and self.stop_loss_order_id not in self.open_orders:
                    self.logger.info(
                        f"Tracked SL order ...{format_order_id(self.stop_loss_order_id)} no longer found in open orders."
                    )
                    self.stop_loss_order_id = None

            # Balance
            if isinstance(balance_tuple, Exception):
                self.logger.error(f"Failed to fetch balance: {balance_tuple}")
            elif balance_tuple is not None and len(balance_tuple) == 2:
                _, fetched_available = balance_tuple  # (total, available)
                self.available_balance = fetched_available if fetched_available is not None else Decimal(0)
            else:
                self.logger.error(
                    f"Failed to fetch balance (fetch_usdt_balance returned unexpected value: {balance_tuple})."
                )

            # Ticker
            if isinstance(ticker, Exception):
                self.logger.warning(f"Failed to fetch ticker: {ticker}")
            elif ticker and ticker.get("last"):
                new_price = safe_decimal_conversion(ticker["last"])
                if new_price and new_price > 0:  # Ensure price is valid
                    self.last_known_price = new_price
                else:
                    self.logger.warning(
                        f"Fetched ticker price '{ticker['last']}' is invalid. Keeping previous price {self.last_known_price}"
                    )

            # Log summary of state
            pos_side = (
                self.current_position.get("side", self.api_config["POS_NONE"])
                if self.current_position
                else self.api_config["POS_NONE"]
            )
            pos_qty = self.current_position.get("qty", Decimal(0)) if self.current_position else Decimal(0)
            self.logger.debug(
                f"State Updated: Pos Side={pos_side}, Qty={pos_qty:.8f}, "
                f"Orders={len(self.open_orders)}, Avail Bal={self.available_balance:.4f}, Last Px={self.last_known_price:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Unexpected error during state update: {e}", exc_info=True)

    async def _fetch_all_open_orders(self) -> list[dict]:
        """Fetches all types of open orders (Regular, Stop, TP/SL) concurrently."""
        if not self.exchange:
            return []
        category = bybit.market_cache.get_category(self.symbol)
        if not category:
            self.logger.error(f"Cannot fetch orders: Category unknown for {self.symbol}")
            return []

        tasks = []
        # Define filters based on Bybit V5 types
        order_filters_to_check = [
            bybit.OrderFilter.ORDER,  # Regular limit/market orders
            bybit.OrderFilter.STOP_ORDER,  # Conditional orders (includes SL/TP placed this way)
            # bybit.OrderFilter.TPSL_ORDER, # Specific UTA TP/SL orders attached to positions (uncomment if needed)
        ]

        for order_filter in order_filters_to_check:
            tasks.append(
                bybit.fetch_open_orders_filtered(
                    self.exchange, self.symbol, category=category, order_filter=order_filter, config=self.api_config
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_orders = []
        filter_names = [f.value for f in order_filters_to_check]
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.warning(f"Failed to fetch open orders with filter '{filter_names[i]}': {res}")
            elif isinstance(res, list):
                all_orders.extend(res)
                self.logger.debug(f"Fetched {len(res)} open orders with filter '{filter_names[i]}'.")

        # Deduplicate based on order ID
        unique_orders = {o["id"]: o for o in all_orders}
        return list(unique_orders.values())

    async def _fetch_and_calculate_indicators(self) -> pd.DataFrame | None:
        """Fetches OHLCV data and calculates indicators."""
        self.logger.debug(f"Fetching OHLCV data ({self.timeframe})...")
        if not self.exchange:
            return None
        try:
            # Fetch enough data for indicators + warm-up
            limit = self.min_data_periods + 50  # Add buffer
            ohlcv_data = await bybit.fetch_ohlcv_paginated(
                self.exchange,
                self.symbol,
                self.timeframe,
                limit=limit,
                config=self.api_config,
            )

            if ohlcv_data is None or ohlcv_data.empty or len(ohlcv_data) < self.min_data_periods:
                self.logger.warning(
                    f"Insufficient OHLCV data ({len(ohlcv_data) if ohlcv_data is not None else 0} candles < {self.min_data_periods})."
                )
                return None

            if not isinstance(ohlcv_data, pd.DataFrame):
                # Handle list conversion if helper returns list
                if isinstance(ohlcv_data, list) and len(ohlcv_data) > 0 and len(ohlcv_data[0]) == 6:
                    try:
                        ohlcv_data = pd.DataFrame(
                            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
                        )
                        ohlcv_data["datetime"] = pd.to_datetime(ohlcv_data["timestamp"], unit="ms", utc=True)
                        ohlcv_data.set_index("datetime", inplace=True)
                        for col in ["open", "high", "low", "close", "volume"]:
                            ohlcv_data[col] = pd.to_numeric(ohlcv_data[col], errors="coerce")
                    except Exception as conv_err:
                        self.logger.error(f"Failed to convert OHLCV list to DataFrame: {conv_err}")
                        return None
                else:
                    self.logger.error("OHLCV data is not a DataFrame and could not be converted.")
                    return None

            # Combine strategy config for calculation
            indicator_config = {
                "indicator_settings": self.indicator_settings,
                "analysis_flags": self.analysis_flags,
                # Pass strategy name/params if indicators.py needs them
                "strategy_params": self.strategy_config.get("strategy_params", {}),
                "strategy": self.strategy_config.get("strategy", {}),
            }
            self.logger.debug("Calculating indicators...")
            # Run synchronous calculation in executor
            loop = asyncio.get_running_loop()
            df_with_indicators = await loop.run_in_executor(
                None, ind.calculate_all_indicators, ohlcv_data.copy(), indicator_config
            )

            # Validate required columns
            if df_with_indicators is None:
                self.logger.error("Indicator calculation function returned None.")
                return None
            missing_cols = [col for col in self.required_indicators if col not in df_with_indicators.columns]
            if missing_cols:
                self.logger.error(f"Required indicator columns missing after calculation: {missing_cols}")
                return None

            return df_with_indicators

        except Exception as e:
            self.logger.error(f"Error fetching or calculating indicators: {e}", exc_info=True)
            return None

    def _check_signals(self, df: pd.DataFrame) -> tuple[Side | None, bool]:
        """Evaluates entry/exit signals based on the EVT indicator.

        Returns:
            tuple: (entry_side, should_exit)
                   entry_side: Side.BUY, Side.SELL, or None
                   should_exit: True if current position should be closed, False otherwise
        """
        if df is None or df.empty or len(df) < 2:  # Need at least 2 rows for prev trend check
            self.logger.debug("Indicator DataFrame is empty or too short for signal generation.")
            return None, False
        if not self.analysis_flags.get("use_evt"):
            self.logger.debug("EVT indicator is disabled, no EVT signals generated.")
            return None, False

        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]
        except IndexError:
            self.logger.warning("Could not access latest/previous rows in indicator DataFrame.")
            return None, False

        # --- Safely get required values ---
        trend_latest = latest.get(self.evt_trend_col)
        trend_prev = previous.get(self.evt_trend_col)
        buy_signal_latest = latest.get(self.evt_buy_col)
        sell_signal_latest = latest.get(self.evt_sell_col)

        # Check if essential EVT indicators are available and not NaN
        # Allow trend to be 0 (neutral state)
        if (
            trend_latest is None
            or pd.isna(trend_latest)
            or trend_prev is None
            or pd.isna(trend_prev)
            or buy_signal_latest is None
            or pd.isna(buy_signal_latest)
            or sell_signal_latest is None
            or pd.isna(sell_signal_latest)
        ):
            self.logger.debug(
                f"EVT Indicators missing or NaN in latest data (Trend:{trend_latest}, Buy:{buy_signal_latest}, Sell:{sell_signal_latest}). No signal."
            )
            return None, False

        # Convert to expected types
        current_trend = int(trend_latest)
        previous_trend = int(trend_prev)
        is_buy_signal = bool(buy_signal_latest)
        is_sell_signal = bool(sell_signal_latest)

        self.logger.debug(
            f"Signal Check Values: Trend={current_trend} (Prev:{previous_trend}), BuySig={is_buy_signal}, SellSig={is_sell_signal}"
        )

        # --- Define Entry/Exit Logic ---
        entry_side: Side | None = None
        should_exit: bool = False
        current_pos_side = self.current_position.get("side") if self.current_position else self.api_config["POS_NONE"]

        # **Entry Conditions (Only if flat)**
        if current_pos_side == self.api_config["POS_NONE"]:
            if is_buy_signal:
                entry_side = Side.BUY
                self.logger.info(f"{Fore.GREEN}ENTRY Signal: BUY triggered by EVT Buy flag.{Style.RESET_ALL}")
            elif is_sell_signal:
                entry_side = Side.SELL
                self.logger.info(f"{Fore.RED}ENTRY Signal: SELL triggered by EVT Sell flag.{Style.RESET_ALL}")

        # **Exit Conditions (Only if in position)**
        elif current_pos_side != self.api_config["POS_NONE"]:
            # Exit Long: EVT Trend flips from 1 (or 0) to -1
            if current_pos_side == self.api_config["POS_LONG"] and current_trend == -1 and previous_trend != -1:
                should_exit = True
                self.logger.info(
                    f"{Fore.YELLOW}EXIT Signal: Close LONG triggered (EVT Trend flipped to -1 from {previous_trend}){Style.RESET_ALL}"
                )

            # Exit Short: EVT Trend flips from -1 (or 0) to 1
            elif current_pos_side == self.api_config["POS_SHORT"] and current_trend == 1 and previous_trend != 1:
                should_exit = True
                self.logger.info(
                    f"{Fore.YELLOW}EXIT Signal: Close SHORT triggered (EVT Trend flipped to 1 from {previous_trend}){Style.RESET_ALL}"
                )

        return entry_side, should_exit

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Decimal | None:
        """Calculates position size based on risk percentage, stop-loss distance, and available balance.
        (Relies on self.available_balance which is updated asynchronously).
        """
        risk_pct = self.strategy_config.get("risk_per_trade", Decimal("0.01"))
        balance = self.available_balance

        if balance <= 0:
            self.logger.error("Cannot calculate position size: Available balance is zero or negative.")
            return None
        if entry_price <= 0 or stop_loss_price <= 0:
            self.logger.error("Cannot calculate position size: Entry or SL price is zero or negative.")
            return None

        risk_amount_usd = balance * risk_pct
        price_diff = abs(entry_price - stop_loss_price)

        if price_diff < self.price_tick:
            self.logger.error(
                f"Cannot calculate position size: Price difference ({price_diff:.8f}) between entry ({entry_price:.4f}) and SL ({stop_loss_price:.4f}) is smaller than tick size ({self.price_tick:.8f}). SL too tight."
            )
            return None

        # Assume Linear contract for position sizing calculation
        position_size_base = risk_amount_usd / price_diff

        # Adjust for Step Size and Limits
        position_size_adjusted = (position_size_base // self.qty_step) * self.qty_step

        if position_size_adjusted <= Decimal(0):
            self.logger.warning(
                f"Calculated position size is zero or negative after adjusting for step size {self.qty_step}. "
                f"Original: {position_size_base:.8f}, Risk Amount: {risk_amount_usd:.4f}, Price Diff: {price_diff:.4f}"
            )
            return None

        # Check against minimum order size
        if position_size_adjusted < self.min_qty:
            self.logger.warning(
                f"Calculated position size ({position_size_adjusted}) is below exchange minimum ({self.min_qty}). Trying minimum size."
            )
            min_qty_risk_usd = self.min_qty * price_diff
            if min_qty_risk_usd <= risk_amount_usd * Decimal("1.05"):  # Allow slightly exceeding risk for min qty
                self.logger.info(
                    f"Using minimum order size ({self.min_qty}) as its risk ({min_qty_risk_usd:.4f}) is close to budget ({risk_amount_usd:.4f})."
                )
                position_size_adjusted = self.min_qty
            else:
                self.logger.error(
                    f"Minimum order size ({self.min_qty}) risk ({min_qty_risk_usd:.4f}) significantly exceeds budget ({risk_amount_usd:.4f}). Cannot place trade."
                )
                return None

        # Basic Margin Check
        cost_estimate = (
            (position_size_adjusted * entry_price) / Decimal(str(self.leverage))
            if self.leverage > 0
            else position_size_adjusted * entry_price
        )
        if cost_estimate > balance * Decimal("0.98"):  # Leave 2% buffer
            self.logger.warning(
                f"Estimated cost ({cost_estimate:.4f}) exceeds 98% of available balance ({balance:.4f}) with {self.leverage}x leverage. Cannot place trade."
            )
            return None

        self.logger.info(
            f"Calculated Position Size: {format_amount(self.exchange, self.symbol, position_size_adjusted)} "
            f"(Risk: {risk_amount_usd:.4f} {self.api_config.get('USDT_SYMBOL', 'USDT')}, Balance: {balance:.4f}, Price Diff: {price_diff:.4f})"
        )
        return position_size_adjusted

    def _calculate_stop_loss_price(self, df: pd.DataFrame, side: Side, entry_price: Decimal) -> Decimal | None:
        """Calculates ATR based stop loss price (Synchronous)."""
        if not self.analysis_flags.get("use_atr"):
            return None
        if df is None or df.empty:
            return None
        if not self.exchange:  # Need exchange for formatting
            self.logger.error("Cannot calculate SL price: Exchange not initialized.")
            return None

        try:
            latest_atr_raw = df.iloc[-1].get(self.atr_col)
            if pd.isna(latest_atr_raw):
                self.logger.warning("Latest ATR value is NaN.")
                return None
            latest_atr = safe_decimal_conversion(latest_atr_raw)
            if latest_atr is None or latest_atr <= 0:
                self.logger.warning(f"Invalid ATR value ({latest_atr}) for SL.")
                return None

            multiplier = self.strategy_config.get("stop_loss_atr_multiplier", Decimal("2.0"))
            stop_offset = latest_atr * multiplier

            if side == Side.BUY:
                sl_price_raw = entry_price - stop_offset
            elif side == Side.SELL:
                sl_price_raw = entry_price + stop_offset
            else:
                return None

            # Format and apply sanity checks
            sl_price_fmt = format_price(self.exchange, self.symbol, sl_price_raw)
            sl_price = safe_decimal_conversion(sl_price_fmt)
            if sl_price is None or sl_price <= 0:
                self.logger.error(f"Formatted SL price is invalid: {sl_price_fmt}")
                return None

            # Sanity check and adjustment using price_tick
            if side == Side.BUY and sl_price >= entry_price:
                self.logger.warning(f"Adjusting Buy SL {sl_price} >= Entry {entry_price}")
                sl_price = entry_price - self.price_tick
            elif side == Side.SELL and sl_price <= entry_price:
                self.logger.warning(f"Adjusting Sell SL {sl_price} <= Entry {entry_price}")
                sl_price = entry_price + self.price_tick

            # Re-format after potential adjustment
            sl_price_final_fmt = format_price(self.exchange, self.symbol, sl_price)
            sl_price_final = safe_decimal_conversion(sl_price_final_fmt)

            if sl_price_final and sl_price_final > 0:
                self.logger.info(
                    f"Calculated SL Price: {sl_price_final_fmt} (Entry: {entry_price:.4f}, ATR: {latest_atr:.5f}, Mult: {multiplier})"
                )
            else:
                self.logger.error(f"Final SL price after adjustment is invalid: {sl_price_final}")
                sl_price_final = None  # Ensure None is returned

            return sl_price_final

        except Exception as e:
            self.logger.error(f"Error calculating stop loss price: {e}", exc_info=True)
            return None

    async def _place_stop_loss(self, entry_side: Side, qty: Decimal, sl_price: Decimal):
        """Places the native stop loss order (Async)."""
        if not self.exchange:
            return
        if not sl_price or sl_price <= 0:
            self.logger.error("Invalid SL price provided. Cannot place SL.")
            return

        sl_order_side = Side.SELL if entry_side == Side.BUY else Side.BUY
        trigger_direction = TriggerDirection.FALL if sl_order_side == Side.SELL else TriggerDirection.RISE

        self.logger.info(
            f"Placing {sl_order_side.value.upper()} native Stop Loss for {qty:.8f} at trigger price {sl_price:.4f}..."
        )

        # Use the helper function
        base_price_for_sl = self.last_known_price
        if base_price_for_sl <= 0:
            # Fetch ticker again if last price is bad? Or fail? Fail safer.
            self.logger.error("Last known price is invalid for SL basePrice. Cannot place SL.")
            # Consider emergency exit here?
            return

        sl_order = await bybit.place_native_stop_loss(
            exchange=self.exchange,
            symbol=self.symbol,
            side=sl_order_side,  # Side of the SL order itself
            amount=qty,
            stop_price=sl_price,
            base_price=base_price_for_sl,  # V5 often requires this
            config=self.api_config,
            trigger_direction=trigger_direction,
            is_reduce_only=True,
            order_type="Market",
            position_idx=self.position_idx,
            trigger_by=TriggerBy.MARK,  # Defaulting to Mark Price for SL
        )

        if sl_order and sl_order.get("id"):
            self.stop_loss_order_id = sl_order["id"]
            self.logger.success(
                f"{Fore.GREEN}Native Stop Loss placed successfully. Order ID: ...{format_order_id(self.stop_loss_order_id)}{Style.RESET_ALL}"
            )
        else:
            self.logger.error(
                f"{Back.RED}{Fore.WHITE}Failed to place native Stop Loss order! Position is unprotected.{Style.RESET_ALL}"
            )
            self.stop_loss_order_id = None
            if self.sms_config.get("ENABLE_SMS_ALERTS"):
                send_sms_alert(
                    f"[{self.symbol}] URGENT: Failed SL placement after {entry_side.value} entry! Pos UNPROTECTED.",
                    self.config,
                )
                # Consider adding emergency exit logic here

    async def _cancel_stop_loss(self):
        """Cancels the tracked stop loss order if it exists (Async)."""
        if not self.exchange:
            return
        if self.stop_loss_order_id:
            sl_id_short = format_order_id(self.stop_loss_order_id)
            self.logger.info(f"Attempting to cancel existing Stop Loss order: ...{sl_id_short}")
            try:
                category = bybit.market_cache.get_category(self.symbol)
                if not category:
                    self.logger.error(f"Cannot cancel SL {sl_id_short}: Category unknown for {self.symbol}")
                    return  # Don't clear ID if we couldn't even try cancelling

                success = await bybit.cancel_order(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    order_id=self.stop_loss_order_id,
                    config=self.api_config,
                    order_filter=bybit.OrderFilter.STOP_ORDER,  # Crucial for V5 conditional order cancellation
                )
                if success:
                    self.logger.info(f"Stop Loss ...{sl_id_short} cancelled successfully (or was already gone).")
                    self.stop_loss_order_id = None  # Clear ID on success/confirmation gone
                else:
                    self.logger.warning(
                        f"Failed attempt to cancel Stop Loss order ...{sl_id_short}. It might remain active or already be filled/cancelled."
                    )
                    # Re-check state to be sure it's gone before clearing ID
                    await self._update_state()
                    if self.stop_loss_order_id and self.stop_loss_order_id not in self.open_orders:
                        self.logger.info(f"Re-checked state: SL order ...{sl_id_short} is confirmed gone.")
                        self.stop_loss_order_id = None

            except Exception as e:
                self.logger.error(f"Error cancelling Stop Loss ...{sl_id_short}: {e}", exc_info=True)
                # Don't clear self.stop_loss_order_id here, might need retry or manual check
        else:
            self.logger.debug("No active Stop Loss order ID tracked to cancel.")

    async def _manage_position(self, entry_side: Side | None, should_exit: bool, df_indicators: pd.DataFrame):
        """Handles placing entry/exit orders and managing SL based on signals."""
        current_pos_side = self.current_position.get("side") if self.current_position else self.api_config["POS_NONE"]
        current_qty = self.current_position.get("qty", Decimal(0)) if self.current_position else Decimal(0)

        # --- Exit Logic ---
        if should_exit and current_pos_side != self.api_config["POS_NONE"]:
            self.logger.warning(
                f"{Fore.YELLOW}{Style.BRIGHT}Exit signal received for {current_pos_side} position. Closing...{Style.RESET_ALL}"
            )
            # 1. Cancel existing SL order FIRST
            await self._cancel_stop_loss()
            await asyncio.sleep(0.5)  # Short pause after cancel request

            # 2. Close the position with reduce-only market order
            self.logger.info(f"Submitting market order to close {current_pos_side} position of {current_qty}...")
            close_order = await bybit.close_position_reduce_only(
                self.exchange,
                self.symbol,
                self.api_config,
                position_to_close=self.current_position,
                reason="Strategy Exit Signal",
            )

            if close_order and close_order.get("id"):
                self.logger.success(
                    f"{Fore.GREEN}Position close order {format_order_id(close_order['id'])} submitted successfully.{Style.RESET_ALL}"
                )
                if self.sms_config.get("ENABLE_SMS_ALERTS"):
                    send_sms_alert(
                        f"[{self.symbol}] Closed {current_pos_side} Pos ({current_qty:.8f}). Reason: EVT Exit",
                        self.config,
                    )
            else:
                self.logger.error(
                    f"{Back.RED}{Fore.WHITE}Failed to submit position close order! Manual intervention likely needed.{Style.RESET_ALL}"
                )
                if self.sms_config.get("ENABLE_SMS_ALERTS"):
                    send_sms_alert(
                        f"[{self.symbol}] URGENT: Failed CLOSE order for {current_pos_side} ({current_qty:.8f})!",
                        self.config,
                    )

            # Update state after attempting closure
            await asyncio.sleep(5)  # Allow time for order processing/state update
            await self._update_state()
            return  # Prevent entry on the same tick

        # --- Entry Logic ---
        if entry_side is not None and current_pos_side == self.api_config["POS_NONE"]:
            self.logger.info(
                f"{Style.BRIGHT}Entry signal: {entry_side.value.upper()}. Preparing to enter...{Style.RESET_ALL}"
            )

            # 1. Cancel any other pending orders (optional, defensive)
            if self.open_orders:  # Check if any orders exist (SL ID is handled separately)
                non_sl_orders = {oid: o for oid, o in self.open_orders.items() if oid != self.stop_loss_order_id}
                if non_sl_orders:
                    self.logger.warning(
                        f"Found {len(non_sl_orders)} non-SL open orders before entry. Cancelling them..."
                    )
                    category = bybit.market_cache.get_category(self.symbol)
                    # Cancel only regular orders, leave stops? Safer to cancel all non-SL.
                    await bybit.cancel_all_orders(
                        self.exchange,
                        self.symbol,
                        category=category,
                        order_filter=OrderFilter.ORDER,
                        config=self.api_config,
                        reason="Pre-Entry Cleanup",
                    )
                    await asyncio.sleep(1)

            # 2. Calculate SL Price (using current price as estimate)
            entry_price_estimate = self.last_known_price
            if entry_price_estimate <= 0:
                self.logger.error("Cannot calculate SL: Invalid estimated entry price.")
                return

            sl_price = self._calculate_stop_loss_price(df_indicators, entry_side, entry_price_estimate)
            if sl_price is None:
                self.logger.error("Failed to calculate valid SL price. Cannot enter trade.")
                return

            # 3. Calculate Position Size (based on estimated entry and SL)
            qty_to_order = self._calculate_position_size(entry_price_estimate, sl_price)
            if qty_to_order is None or qty_to_order <= 0:
                self.logger.error(f"Failed to calculate valid order quantity ({qty_to_order}). Cannot enter trade.")
                return

            # 4. Place Market Entry Order
            entry_order_side_str = entry_side.value  # Convert Enum to string for helper
            self.logger.info(f"Placing {entry_order_side_str.upper()} market order for {qty_to_order:.8f}...")
            entry_order = await bybit.place_market_order_slippage_check(
                exchange=self.exchange,
                symbol=self.symbol,
                side=entry_order_side_str,
                amount=qty_to_order,
                config=self.api_config,
                is_reduce_only=False,
                time_in_force=TimeInForce.IOC,
                position_idx=self.position_idx,  # Ensure correct index passed
            )

            # 5. Handle Entry Order Result & Place SL
            if entry_order and entry_order.get("status") in ["closed", "filled"]:
                # Use filled price/qty from order receipt if available, else re-fetch position state
                filled_price_entry = safe_decimal_conversion(entry_order.get("average"))
                filled_qty_entry = safe_decimal_conversion(entry_order.get("filled"))

                # If receipt lacks details, wait and fetch position state
                if not filled_price_entry or not filled_qty_entry or filled_qty_entry <= 0:
                    self.logger.warning("Entry order receipt lacks fill details, re-fetching position state...")
                    await asyncio.sleep(5)  # Wait longer for state update
                    await self._update_state()
                    if (
                        self.current_position and self.current_position.get("side") == entry_side.name.upper()
                    ):  # Check Enum name vs POS_LONG/SHORT
                        filled_price_entry = self.current_position.get("entry_price", entry_price_estimate)  # Fallback
                        filled_qty_entry = self.current_position.get("qty", qty_to_order)  # Fallback
                        self.logger.info("Using position state for fill details.")
                    else:
                        self.logger.error("Failed to confirm position opening after entry order.")
                        return  # Cannot proceed without confirmed entry

                if filled_qty_entry <= 0:  # Final check
                    self.logger.error("Confirmed filled quantity is zero. Cannot place SL.")
                    return

                self.logger.success(
                    f"{Fore.GREEN}Entry order filled: Side={entry_side.value}, Qty={filled_qty_entry:.8f}, AvgPx={filled_price_entry:.4f}{Style.RESET_ALL}"
                )

                # IMPORTANT: Recalculate SL price based on ACTUAL fill price
                sl_price_actual = self._calculate_stop_loss_price(df_indicators, entry_side, filled_price_entry)
                if sl_price_actual is None:
                    self.logger.error(
                        f"{Back.RED}{Fore.WHITE}Failed to recalculate SL price based on actual fill price ({filled_price_entry:.4f}). Cannot place SL. POSITION UNPROTECTED.{Style.RESET_ALL}"
                    )
                    if self.sms_config.get("ENABLE_SMS_ALERTS"):
                        send_sms_alert(
                            f"[{self.symbol}] URGENT: Failed SL RECALC after {entry_side.value} entry! Pos UNPROTECTED.",
                            self.config,
                        )
                    # Add emergency close logic here? Very important.
                    # Example: await self._emergency_close("Failed SL Recalculation")
                    return

                await asyncio.sleep(0.5)  # Brief pause before placing SL

                # Place the SL using actual filled quantity and recalculated SL price
                await self._place_stop_loss(entry_side, filled_qty_entry, sl_price_actual)

                # Final state update after entry and SL placement
                await asyncio.sleep(2)
                await self._update_state()

            elif entry_order:  # Order placed but not confirmed filled
                status = entry_order.get("status", "unknown")
                order_id = entry_order.get("id", "N/A")
                self.logger.warning(
                    f"Entry market order ({format_order_id(order_id)}) status is '{status}'. Position may not have been opened."
                )
                await self._update_state()  # Check if position opened unexpectedly
            else:  # place_market_order returned None
                self.logger.error("Failed to place entry market order. Check logs from bybit_helpers.")

    async def run_loop(self):
        """Main strategy execution loop."""
        if not await self._initialize():
            self.logger.critical("Strategy initialization failed. Shutting down.")
            return

        self.is_running = True
        self.logger.info(f"{Fore.CYAN}{Style.BRIGHT}=== Starting Ehlers Volumetric Strategy Trading Loop ===")

        while self.is_running:
            try:
                loop_start_time = time.monotonic()
                self.logger.info(
                    "-" * 30 + f" Tick Start ({pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')}) " + "-" * 30
                )

                # 1. Update current state (position, orders, balance, price)
                await self._update_state()
                if self.exchange is None:
                    self.logger.critical("Exchange object became None. Stopping loop.")
                    self.is_running = False
                    continue

                # 2. Fetch data and calculate indicators
                df_indicators = await self._fetch_and_calculate_indicators()
                if df_indicators is None:
                    self.logger.warning("Failed to get indicator data for this tick.")
                    await asyncio.sleep(self.strategy_config.get("polling_interval_seconds", 60))
                    continue

                # 3. Check for entry/exit signals
                entry_side, should_exit = self._check_signals(df_indicators)

                # 4. Manage position based on signals
                await self._manage_position(entry_side, should_exit, df_indicators)

                # --- Loop Timing ---
                loop_end_time = time.monotonic()
                elapsed = loop_end_time - loop_start_time
                poll_interval = self.strategy_config.get("polling_interval_seconds", 60)
                sleep_time = max(0.1, poll_interval - elapsed)  # Ensure minimum sleep
                self.logger.info(f"Tick processed in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s.")
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                self.logger.info("Strategy loop cancellation requested.")
                self.is_running = False  # Ensure loop terminates
            except Exception as e:
                self.logger.critical(
                    f"{Back.RED}{Fore.WHITE}CRITICAL UNHANDLED ERROR in strategy loop: {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                if self.sms_config.get("ENABLE_SMS_ALERTS"):
                    send_sms_alert(f"[{self.symbol}] CRITICAL LOOP ERROR: {type(e).__name__}. Check Logs!", self.config)
                self.logger.info("Pausing for 1 minute after critical loop error...")
                await asyncio.sleep(60)  # Pause longer after error

        self.logger.info("--- Strategy Loop Finished ---")
        await self._cleanup()

    async def stop(self):
        """Signals the strategy loop to stop gracefully."""
        if self.is_running:
            self.logger.warning("Stop signal received. Strategy loop will terminate after current cycle.")
            self.is_running = False
        else:
            self.logger.info("Stop signal received, but loop wasn't running.")

    async def _cleanup(self):
        """Performs cleanup actions on shutdown."""
        self.logger.info("--- Initiating Strategy Cleanup ---")
        if self.exchange:
            # Cancel all remaining open orders (including SL)
            self.logger.info("Cancelling all remaining open orders...")
            category = bybit.market_cache.get_category(self.symbol)
            if category:
                cancelled_count = await bybit.cancel_all_orders(
                    self.exchange,
                    symbol=self.symbol,
                    category=category,
                    config=self.api_config,
                    reason="Shutdown Cleanup",
                )
                if cancelled_count is not None:
                    self.logger.info(f"Cancelled {cancelled_count} orders during cleanup.")
            else:
                self.logger.error("Cannot perform cleanup order cancellation: Category unknown.")

            # Exchange closing is handled in main.py's finally block
        self.logger.info("--- Strategy Cleanup Complete ---")

    async def _emergency_close(self, reason: str):
        """Attempts to immediately close the current position in an emergency."""
        self.logger.critical(f"{Back.RED}{Fore.WHITE}EMERGENCY CLOSE triggered! Reason: {reason}{Style.RESET_ALL}")
        await self._update_state()  # Get latest position info
        if self.current_position and self.current_position.get("side") != self.api_config["POS_NONE"]:
            pos_side = self.current_position["side"]
            pos_qty = self.current_position["qty"]
            self.logger.warning(f"Attempting to close {pos_side} position of {pos_qty}...")
            await self._cancel_stop_loss()  # Cancel SL first
            await asyncio.sleep(0.5)
            close_order = await bybit.close_position_reduce_only(
                self.exchange,
                self.symbol,
                self.api_config,
                position_to_close=self.current_position,
                reason=f"Emergency Close: {reason}",
            )
            if close_order and close_order.get("id"):
                self.logger.warning(
                    f"{Fore.YELLOW}Emergency close order submitted: {format_order_id(close_order['id'])}"
                )
                if self.sms_config.get("ENABLE_SMS_ALERTS"):
                    send_sms_alert(
                        f"[{self.symbol}] EMERGENCY Closing {pos_side} ({pos_qty:.8f}). Reason: {reason}", self.config
                    )
            else:
                self.logger.critical(
                    f"{Back.RED}{Fore.WHITE}FAILED to submit emergency close order! MANUAL INTERVENTION REQUIRED!{Style.RESET_ALL}"
                )
                if self.sms_config.get("ENABLE_SMS_ALERTS"):
                    send_sms_alert(
                        f"[{self.symbol}] !!! CRITICAL: FAILED EMERGENCY CLOSE for {pos_side} ({pos_qty:.8f}) !!!",
                        self.config,
                    )
        else:
            self.logger.info("Emergency close triggered, but no open position found.")
