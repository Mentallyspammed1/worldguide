#!/usr/bin/env python
"""
Ehlers Volumetric Trend Strategy (v2.1 - Integrated with Bybit Trading Enchanted)

An advanced trading strategy using the Ehlers Volumetric Trend (EVT) indicator,
integrated with Bybit Trading Enchanced for Bybit V5 API interaction in Termux.

Strategy Logic:
- Uses EVT from indicators.py for bullish/bearish trend entry signals.
- Confirms signals with multi-timeframe analysis.
- Enters LONG/SHORT with ATR-based SL/TP (reduce-only limit orders).
- Exits on trend reversal, SL/TP hit, or trailing stop.
- Dynamic position sizing based on risk and volatility.
- Real-time updates via WebSocket (kline, order streams).
- Termux-optimized with SMS alerts and robust logging.

Usage:
- Ensure bybit_trading_enchanced.py and indicators.py are in the same directory.
- Configure .env via `python bybit_trading_enchanced.py --setup`.
- Run: `python ehlers_volumetric_strategy.py`
"""

import logging
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional, Dict, Tuple, List, Literal
from datetime import datetime
import pandas as pd

try:
    from bybit_trading_enchanced import load_config, BybitHelper
except ImportError as e:
    print(f"Error: Could not import bybit_trading_enchanced: {e}", file=sys.stderr)
    print("Ensure bybit_trading_enchanced.py is in the same directory.", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas library not found. Install: pip install pandas", file=sys.stderr)
    sys.exit(1)

# --- Strategy Class ---
class EhlersStrategy:
    """Encapsulates the Ehlers Volumetric Trend trading strategy."""

    def __init__(self, helper: BybitHelper):
        self.helper = helper
        self.config = helper.config
        self.logger = helper.logger
        self.symbol = self.config.api_config.symbol
        self.timeframe = self.config.strategy_config.timeframe
        self.confirmation_timeframe = "15m"
        self.is_running = False

        # Position State
        self.current_side: Literal["none", "long", "short"] = "none"
        self.current_qty: Decimal = Decimal("0.0")
        self.entry_price: Optional[Decimal] = None
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.trailing_stop_price: Optional[Decimal] = None
        self.position_start_time: float = 0

        # Market Details
        self.min_qty: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.price_tick: Optional[Decimal] = None

        # Strategy Parameters
        self.evt_length = self.config.strategy_config.indicator_settings.get("evt_length", 7)
        self.atr_period = self.config.strategy_config.indicator_settings.get("atr_period", 14)
        self.sl_atr_multiplier = Decimal("2.0")
        self.tp_atr_multiplier = Decimal("4.0")
        self.trailing_stop_atr_multiplier = Decimal("1.5")
        self.volatility_threshold = Decimal("0.5")
        self.max_position_age_seconds = 3600

        # Data Cache
        self.last_kline_time: Dict[str, float] = {}

        self.logger.info(f"EhlersStrategy initialized for {self.symbol} on {self.timeframe}")

    def _initialize(self) -> bool:
        """Initialize market details and validate connection."""
        self.logger.info("Initializing Ehlers Strategy...")
        try:
            market = self.helper.exchange.markets.get(self.symbol)
            if not market:
                self.logger.critical(f"Market {self.symbol} not found.")
                return False
            self.min_qty = Decimal(str(market['limits']['amount']['min']))
            self.qty_step = Decimal(str(market['precision']['amount']))
            self.price_tick = Decimal(str(market['precision']['price']))
            self.logger.info(
                f"Market Details: Min Qty={self.min_qty}, Qty Step={self.qty_step}, Price Tick={self.price_tick}"
            )

            if not self.helper.diagnose_connection():
                self.logger.critical("Connection diagnostics failed.")
                return False

            def kline_callback(data: Dict):
                df_updated = self.helper.process_kline_update(data)
                if df_updated is not None:
                    self._process_kline(df_updated)

            def order_callback(data: Dict):
                order = data.get('data', [{}])[0]
                if order:
                    self.logger.info(
                        f"Order update: {order['orderId']} - {order['side']} {order['qty']} @ {order['price']}"
                    )
                    self._check_order_status()

            self.helper.subscribe_to_stream(
                [f"kline.{self.timeframe}.{self.symbol}", f"kline.{self.confirmation_timeframe}.{self.symbol}"],
                kline_callback,
            )
            self.helper.subscribe_to_stream(["order"], order_callback, channel_type="private")

            return True
        except Exception as e:
            self.logger.critical(f"Initialization error: {e}", exc_info=True)
            return False

    def _process_kline(self, df: pd.DataFrame):
        """Process updated kline data with indicators."""
        try:
            self.helper.ohlcv_cache[self.timeframe] = df
            self.last_kline_time[self.timeframe] = time.time()
            self.logger.debug(f"Processed {self.timeframe} kline update")
        except Exception as e:
            self.logger.error(f"Error processing kline: {e}", exc_info=True)

    def _fetch_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch or use cached OHLCV data with indicators."""
        if (
            timeframe in self.helper.ohlcv_cache
            and time.time() - self.last_kline_time.get(timeframe, 0) < 300
        ):
            return self.helper.ohlcv_cache[timeframe]
        ohlcv = self.helper.fetch_ohlcv(timeframe, limit=200)
        if not ohlcv:
            self.logger.warning(f"Could not fetch {timeframe} OHLCV data")
            return None
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = self.helper.calculate_indicators(df)
        self.helper.ohlcv_cache[timeframe] = df
        self.last_kline_time[timeframe] = time.time()
        return df

    def _generate_signals(self, df_ind: pd.DataFrame, df_confirm: pd.DataFrame) -> Optional[Literal["Buy", "Sell"]]:
        """Generate trading signals with multi-timeframe confirmation."""
        if df_ind is None or df_ind.empty or df_confirm is None or df_confirm.empty:
            return None
        try:
            latest = df_ind.iloc[-1]
            latest_confirm = df_confirm.iloc[-1]
            evt_col = f"evt_trend_{self.evt_length}"
            buy_col = f"evt_buy_{self.evt_length}"
            sell_col = f"evt_sell_{self.evt_length}"

            if not all(col in latest for col in [evt_col, buy_col, sell_col]):
                self.logger.warning(f"Missing EVT columns: {latest.to_dict()}")
                return None

            confirm_trend = latest_confirm.get(evt_col, 0)
            buy_signal = latest[buy_col] and confirm_trend >= 0
            sell_signal = latest[sell_col] and confirm_trend <= 0

            if buy_signal:
                self.logger.info("BUY signal generated (confirmed by higher timeframe)")
                return "Buy"
            elif sell_signal:
                self.logger.info("SELL signal generated (confirmed by higher timeframe)")
                return "Sell"
            return None
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}", exc_info=True)
            return None

    def _calculate_sl_tp(self, df_ind: pd.DataFrame, side: Literal["Buy", "Sell"], entry_price: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate stop-loss and take-profit prices."""
        if df_ind is None or df_ind.empty:
            return None, None
        try:
            atr = Decimal(str(df_ind.iloc[-1][f"ATRr_{self.atr_period}"]))
            if atr <= 0:
                self.logger.warning(f"Invalid ATR: {atr}")
                return None, None

            sl_offset = atr * self.sl_atr_multiplier
            tp_offset = atr * self.tp_atr_multiplier if self.tp_atr_multiplier > 0 else None
            if side == "Buy":
                sl_price = entry_price - sl_offset
                tp_price = entry_price + tp_offset if tp_offset else None
            else:
                sl_price = entry_price + sl_offset
                tp_price = entry_price - tp_offset if tp_offset else None

            if self.price_tick:
                rounding_sl = ROUND_DOWN if side == "Buy" else ROUND_UP
                sl_price = (sl_price / self.price_tick).quantize(Decimal("1"), rounding=rounding_sl) * self.price_tick
                if tp_price:
                    rounding_tp = ROUND_DOWN if side == "Buy" else ROUND_UP
                    tp_price = (tp_price / self.price_tick).quantize(Decimal("1"), rounding=rounding_tp) * self.price_tick

            if side == "Buy" and (sl_price >= entry_price or (tp_price and tp_price <= entry_price)):
                self.logger.warning("Invalid SL/TP for Buy")
                return None, None
            if side == "Sell" and (sl_price <= entry_price or (tp_price and tp_price >= entry_price)):
                self.logger.warning("Invalid SL/TP for Sell")
                return None, None

            self.logger.info(f"SL: {sl_price}, TP: {tp_price or 'None'} (ATR: {atr})")
            return sl_price, tp_price
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP: {e}", exc_info=True)
            return None, None

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """Calculate position size based on risk and volatility."""
        try:
            balance = Decimal(str(self.helper.fetch_balance().get("total", {}).get("USDT", 0)))
            if balance <= 0:
                self.logger.error("Zero or invalid balance")
                return None

            risk_amount = balance * self.config.strategy_config.risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)
            if price_diff <= 0:
                self.logger.error("Invalid price difference")
                return None

            size = risk_amount / price_diff
            if self.qty_step:
                size = (size // self.qty_step) * self.qty_step
            if self.min_qty and size < self.min_qty:
                self.logger.warning(f"Size {size} below min qty {self.min_qty}")
                return None
            if size <= 0:
                self.logger.warning("Calculated size is zero")
                return None

            self.logger.info(f"Position size: {size} (Risk: {risk_amount}, Balance: {balance})")
            return size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    def _cancel_open_orders(self, reason: str = "Strategy Action") -> bool:
        """Cancel tracked SL/TP orders."""
        success = True
        for order_id in [self.sl_order_id, self.tp_order_id]:
            if order_id:
                try:
                    self.helper.session.cancel_order(category="linear", orderId=order_id)
                    self.logger.info(f"Cancelled order {order_id}: {reason}")
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order_id}: {e}")
                    success = False
        self.sl_order_id = self.tp_order_id = None
        return success

    def _check_order_status(self):
        """Check if SL/TP orders have been filled."""
        if not self.sl_order_id and not self.tp_order_id:
            return
        try:
            orders = self.helper.get_open_orders().get("result", {}).get("list", [])
            open_ids = {order["orderId"] for order in orders}
            if self.sl_order_id and self.sl_order_id not in open_ids:
                self.logger.info("SL order filled or cancelled")
                self.sl_order_id = None
                self._update_state()
            if self.tp_order_id and self.tp_order_id not in open_ids:
                self.logger.info("TP order filled or cancelled")
                self.tp_order_id = None
                self._update_state()
        except Exception as e:
            self.logger.error(f"Error checking order status: {e}", exc_info=True)

    def _update_state(self):
        """Update position and order state."""
        try:
            pos_data = self.helper.get_position_info().get("result", {}).get("list", [{}])[0]
            self.current_side = pos_data.get("side", "none").lower()
            self.current_qty = Decimal(str(pos_data.get("size", "0")))
            self.entry_price = Decimal(str(pos_data.get("avgPrice", "0"))) if self.current_side != "none" else None
            if self.current_side == "none":
                self.sl_order_id = self.tp_order_id = self.trailing_stop_price = None
            self.logger.debug(f"State: Side={self.current_side}, Qty={self.current_qty}, Entry={self.entry_price}")
        except Exception as e:
            self.logger.error(f"Error updating state: {e}", exc_info=True)

    def _place_sl_tp_orders(self, side: Literal["Buy", "Sell"], qty: Decimal, sl_price: Decimal, tp_price: Optional[Decimal]):
        """Place SL/TP as reduce-only limit orders."""
        try:
            if sl_price:
                sl_side = "Sell" if side == "Buy" else "Buy"
                sl_result = self.helper.session.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side=sl_side,
                    orderType="Limit",
                    qty=str(qty),
                    price=str(sl_price),
                    reduceOnly=True,
                    triggerPrice=str(sl_price),
                    orderFilter="StopOrder"
                )
                if sl_result.get("retCode") == 0:
                    self.sl_order_id = sl_result["result"]["orderId"]
                    self.logger.info(f"Placed SL order: {sl_side} @ {sl_price}")
                else:
                    self.logger.error(f"Failed to place SL order: {sl_result.get('retMsg')}")
                    return False

            if tp_price:
                tp_side = "Sell" if side == "Buy" else "Buy"
                tp_result = self.helper.session.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side=tp_side,
                    orderType="Limit",
                    qty=str(qty),
                    price=str(tp_price),
                    reduceOnly=True
                )
                if tp_result.get("retCode") == 0:
                    self.tp_order_id = tp_result["result"]["orderId"]
                    self.logger.info(f"Placed TP order: {tp_side} @ {tp_price}")
                else:
                    self.logger.error(f"Failed to place TP order: {tp_result.get('retMsg')}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error placing SL/TP orders: {e}", exc_info=True)
            return False

    def _update_trailing_stop(self, current_price: Decimal, df_ind: pd.DataFrame):
        """Update trailing stop based on ATR."""
        if not self.entry_price or not self.current_side or df_ind is None:
            return
        try:
            atr = Decimal(str(df_ind.iloc[-1][f"ATRr_{self.atr_period}"]))
            trail_offset = atr * self.trailing_stop_atr_multiplier
            if self.current_side == "long":
                new_trail = current_price - trail_offset
                if not self.trailing_stop_price or new_trail > self.trailing_stop_price:
                    self.trailing_stop_price = new_trail
                    self.logger.info(f"Updated trailing stop: {new_trail}")
            else:
                new_trail = current_price + trail_offset
                if not self.trailing_stop_price or new_trail < self.trailing_stop_price:
                    self.trailing_stop_price = new_trail
                    self.logger.info(f"Updated trailing stop: {new_trail}")
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}", exc_info=True)

    def run(self):
        """Main strategy loop."""
        self.is_running = True
        if not self._initialize():
            self.logger.critical("Initialization failed. Exiting.")
            self.helper.send_sms_alert("CRITICAL: Ehlers Strategy failed to start", priority="critical")
            return

        self.logger.info(f"Starting Ehlers Strategy for {self.symbol}")

        while self.is_running:
            try:
                # Fetch data
                df_ind = self._fetch_data(self.timeframe)
                df_confirm = self._fetch_data(self.confirmation_timeframe)
                if not df_ind or not df_confirm:
                    time.sleep(self.config.strategy_config.loop_delay_seconds)
                    continue

                # Update state
                self._update_state()
                current_price = Decimal(str(self.helper.fetch_ticker().get("last", "0")))
                if current_price <= 0:
                    self.logger.warning("Invalid current price")
                    time.sleep(self.config.strategy_config.loop_delay_seconds)
                    continue

                # Update trailing stop
                self._update_trailing_stop(current_price, df_ind)

                # Check position timeout
                if self.current_side != "none" and self.position_start_time:
                    if time.time() - self.position_start_time > self.max_position_age_seconds:
                        self.logger.info("Position timeout reached. Closing position.")
                        self._cancel_open_orders("Position Timeout")
                        self.helper.place_market_order(
                            "Sell" if self.current_side == "long" else "Buy",
                            float(self.current_qty)
                        )
                        self._update_state()
                        self.position_start_time = 0

                # Check exit conditions
                if self.current_side != "none":
                    latest_trend = df_ind.iloc[-1].get(f"evt_trend_{self.evt_length}", 0)
                    should_exit = False
                    exit_reason = ""
                    if self.current_side == "long" and latest_trend == -1:
                        should_exit = True
                        exit_reason = "EVT Trend flipped Short"
                    elif self.current_side == "short" and latest_trend == 1:
                        should_exit = True
                        exit_reason = "EVT Trend flipped Long"
                    elif self.trailing_stop_price and (
                        (self.current_side == "long" and current_price <= self.trailing_stop_price) or
                        (self.current_side == "short" and current_price >= self.trailing_stop_price)
                    ):
                        should_exit = True
                        exit_reason = "Trailing Stop Hit"

                    if should_exit:
                        self.logger.info(f"Exiting position: {exit_reason}")
                        self._cancel_open_orders(exit_reason)
                        self.helper.place_market_order(
                            "Sell" if self.current_side == "long" else "Buy",
                            float(self.current_qty)
                        )
                        self._update_state()
                        self.position_start_time = 0
                        continue

                # Generate signals
                signal = self._generate_signals(df_ind, df_confirm)
                if signal and self.current_side == "none":
                    atr = Decimal(str(df_ind.iloc[-1][f"ATRr_{self.atr_period}"]))
                    atr_percent = atr / current_price * 100
                    if atr_percent < self.volatility_threshold:
                        self.logger.info(f"Volatility too low ({atr_percent:.2f}%). Skipping signal.")
                        time.sleep(self.config.strategy_config.loop_delay_seconds)
                        continue

                    sl_price, tp_price = self._calculate_sl_tp(df_ind, signal, current_price)
                    if not sl_price:
                        self.logger.warning("Invalid SL price. Skipping signal.")
                        time.sleep(self.config.strategy_config.loop_delay_seconds)
                        continue

                    qty = self._calculate_position_size(current_price, sl_price)
                    if not qty:
                        self.logger.warning("Invalid position size. Skipping signal.")
                        time.sleep(self.config.strategy_config.loop_delay_seconds)
                        continue

                    order = self.helper.place_market_order(signal, float(qty))
                    if order.get("retCode") != 0:
                        self.logger.error(f"Failed to place order: {order.get('retMsg')}")
                        time.sleep(self.config.strategy_config.loop_delay_seconds)
                        continue

                    if self._place_sl_tp_orders(signal, qty, sl_price, tp_price):
                        self._update_state()
                        self.position_start_time = time.time()
                        self.helper.send_sms_alert(
                            f"Entered {signal} position: {qty} {self.symbol} @ {current_price}",
                            priority="normal"
                        )

                time.sleep(self.config.strategy_config.loop_delay_seconds)

            except Exception as e:
                self.logger.error(f"Strategy loop error: {e}", exc_info=True)
                self.helper.send_sms_alert(f"Ehlers Strategy error: {e}", priority="critical")
                time.sleep(60)

    def stop(self):
        """Stop the strategy."""
        self.is_running = False
        self._cancel_open_orders("Strategy Stop")
        if self.current_side != "none":
            self.helper.place_market_order(
                "Sell" if self.current_side == "long" else "Buy",
                float(self.current_qty)
            )
        self.logger.info("Ehlers Strategy stopped")

def main():
    """Main entry point."""
    config = load_config()
    helper = BybitHelper(config)
    strategy = EhlersStrategy(helper)
    try:
        strategy.run()
    except KeyboardInterrupt:
        strategy.stop()
        helper.stop()

if __name__ == "__main__":
    main()