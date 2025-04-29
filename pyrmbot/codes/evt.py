#!/usr/bin/env python
"""
Ehlers Volumetric Strategy (v1.0)

A trading strategy using the Ehlers Volumetric Trend (EVT) indicator for signal generation.
Integrates with BybitHelper for API interactions and indicators.py for technical analysis.
Designed for Termux with real-time WebSocket data and SMS alerts.

Key Features:
- Uses EVT for trend-based entry signals.
- ATR-based stop-loss and take-profit.
- Risk management with configurable risk per trade.
- WebSocket-driven real-time updates.
- Logging and SMS alerts for trade events.

Usage:
- Ensure bybit_trading_enchanced.py and indicators.py are in the same directory.
- Configure .env via: python bybit_trading_enchanced.py --setup
- Run: python ehlers_volumetric_strategy.py
"""

import logging
import time
from decimal import Decimal
import pandas as pd
from bybit_trading_enchanced import BybitHelper, load_config

class EhlersVolumetricStrategy:
    """Trading strategy using Ehlers Volumetric Trend indicator."""

    def __init__(self, helper: BybitHelper):
        self.helper = helper
        self.config = helper.config
        self.logger = logging.getLogger(__name__)
        self.position_side = None
        self.position_qty = Decimal("0")
        self.evt_length = self.config.strategy_config.indicator_settings.get("evt_length", 7)
        self.atr_multiplier = Decimal("2.0")  # For SL/TP
        self._initialize()

    def _initialize(self):
        """Initialize WebSocket subscriptions and check connection."""
        if not self.helper.diagnose_connection():
            self.logger.critical("Connection diagnostics failed. Exiting.")
            raise SystemExit(1)
        topics = [f"kline.{self.config.strategy_config.timeframe}.{self.config.api_config.symbol}"]
        self.helper.subscribe_to_stream(topics, self._kline_callback, channel_type="public")
        self.logger.info("Strategy initialized and subscribed to WebSocket streams.")

    def _kline_callback(self, data: dict):
        """Handle WebSocket kline updates."""
        df_updated = self.helper.process_kline_update(data)
        if df_updated is not None:
            self._generate_signals(df_updated)

    def _generate_signals(self, df: pd.DataFrame):
        """Generate trading signals based on EVT."""
        try:
            evt_buy_col = f"evt_buy_{self.evt_length}"
            evt_sell_col = f"evt_sell_{self.evt_length}"
            atr_col = f"ATRr_{self.config.strategy_config.indicator_settings.get('atr_period', 14)}"

            if evt_buy_col not in df.columns or evt_sell_col not in df.columns or atr_col not in df.columns:
                self.logger.warning("Required indicator columns missing.")
                return

            latest = df.iloc[-1]
            current_price = Decimal(str(latest["close"]))
            atr = Decimal(str(latest[atr_col])) if pd.notna(latest[atr_col]) else Decimal("0")

            # Calculate position size
            balance = self.helper.fetch_balance()
            usdt_balance = Decimal(str(balance.get("list", [{}])[0].get("totalEquity", "0")))
            risk_amount = usdt_balance * self.config.strategy_config.risk_per_trade
            position_qty = (risk_amount / (atr * self.atr_multiplier)).quantize(Decimal("0.001"))

            # Signals
            if latest[evt_buy_col] and self.position_side != "Buy":
                self._execute_trade("Buy", position_qty, current_price, atr)
            elif latest[evt_sell_col] and self.position_side != "Sell":
                self._execute_trade("Sell", position_qty, current_price, atr)

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}", exc_info=True)

    def _execute_trade(self, side: str, qty: Decimal, price: Decimal, atr: Decimal):
        """Execute trade with SL/TP."""
        try:
            # Close existing position if opposite
            if self.position_side and self.position_side != side:
                self.helper.place_market_order(
                    side="Sell" if self.position_side == "Buy" else "Buy",
                    qty=float(self.position_qty),
                    reduce_only=True
                )
                self.logger.info(f"Closed {self.position_side} position: {self.position_qty}")
                self.position_side = None
                self.position_qty = Decimal("0")

            # Place new market order
            response = self.helper.place_market_order(side, float(qty))
            if response.get("retCode") == 0:
                self.position_side = side
                self.position_qty = qty
                # Set SL/TP
                sl_price = price - (atr * self.atr_multiplier) if side == "Buy" else price + (atr * self.atr_multiplier)
                tp_price = price + (atr * self.atr_multiplier) if side == "Buy" else price - (atr * self.atr_multiplier)
                self.helper.place_limit_order(
                    side="Sell" if side == "Buy" else "Buy",
                    qty=float(qty),
                    price=float(sl_price),
                    reduce_only=True
                )
                self.helper.place_limit_order(
                    side="Sell" if side == "Buy" else "Buy",
                    qty=float(qty),
                    price=float(tp_price),
                    reduce_only=True
                )
                self.logger.info(f"Set SL: {sl_price}, TP: {tp_price}")
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)

    def run(self):
        """Main strategy loop."""
        self.logger.info("Starting Ehlers Volumetric Strategy...")
        try:
            while True:
                time.sleep(self.config.strategy_config.loop_delay_seconds)
        except KeyboardInterrupt:
            self.logger.info("Strategy stopped by user.")
            self.helper.stop()

def main():
    config = load_config()
    helper = BybitHelper(config)
    strategy = EhlersVolumetricStrategy(helper)
    strategy.run()

if __name__ == "__main__":
    main()