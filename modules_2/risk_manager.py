# File: risk_manager.py
import logging
import pandas as pd # For pd.Series type hint
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
from typing import Dict, Any, Optional, Tuple

# Assuming app_config.py, trading_enums.py, exchange_interface.py are in the same directory
from app_config import CALCULATION_PRECISION, DECIMAL_DISPLAY_PRECISION
from trading_enums import PositionSide, OrderSide
from exchange_interface import BybitV5Wrapper # For type hinting

class PositionManager:
    """Handles position sizing, stop-loss, take-profit, and exit logic."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, exchange_wrapper: BybitV5Wrapper):
        self.logger = logger
        self.config = config
        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading_settings", {})
        self.indicator_config = config.get("indicator_settings", {})
        self.exchange = exchange_wrapper
        self.symbol = self.trading_config.get("symbol")
        if not self.symbol:
             logger.critical("Trading symbol not defined. Cannot init PositionManager.")
             raise ValueError("Trading symbol is required.")
        self.hedge_mode = self.trading_config.get("hedge_mode", False)

    def get_base_quote(self, market: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        if market:
            base, quote = market.get('base'), market.get('quote')
            if base and quote: return base, quote
            else: self.logger.error(f"Base/Quote asset missing in market data for {self.symbol}.")
        return None, None

    def calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal,
                                available_equity: Decimal, quote_asset: str) -> Optional[Decimal]:
        risk_percent = self.risk_config.get("risk_per_trade_percent", Decimal("1.0")) / Decimal("100")
        if not all(isinstance(val, Decimal) and val.is_finite() and val > 0 for val in [entry_price, stop_loss_price, available_equity]):
             self.logger.error(f"Invalid inputs for size calc: E={entry_price}, SL={stop_loss_price}, EQ={available_equity}"); return None
        if entry_price == stop_loss_price: self.logger.error("Entry and SL price cannot be same."); return None
        
        market = self.exchange.get_market(self.symbol)
        if not market: self.logger.error(f"Market data for {self.symbol} not found for size calc."); return None
        base_asset, market_quote_asset = self.get_base_quote(market)
        if not base_asset or not market_quote_asset: self.logger.error(f"Could not get base/quote for {self.symbol}."); return None
        if market_quote_asset != quote_asset: self.logger.warning(f"Config quote '{quote_asset}' differs from market '{market_quote_asset}'. Using market quote."); quote_asset = market_quote_asset
        
        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION
            risk_amount_quote = available_equity * risk_percent
            self.logger.info(f"Risk: {risk_percent:.2%}, Equity ({quote_asset}): {available_equity:.2f}, Risk Amount: {risk_amount_quote:.2f} {quote_asset}")
            stop_loss_distance = abs(entry_price - stop_loss_price)
            if stop_loss_distance <= Decimal(0): self.logger.error(f"SL distance non-positive ({stop_loss_distance})."); return None
            
            contract_type, is_inverse = market.get('type', 'spot'), market.get('inverse', False)
            is_linear = market.get('linear', True) if contract_type == 'swap' and not is_inverse else False
            contract_size = market.get('contractSize', Decimal('1.0'))
            if contract_size is None or not contract_size.is_finite() or contract_size <= 0: self.logger.error(f"Invalid contract size ({contract_size})."); return None

            position_size_base = (risk_amount_quote * entry_price) / (contract_size * stop_loss_distance) if is_inverse else risk_amount_quote / (contract_size * stop_loss_distance)
            if position_size_base is None or not position_size_base.is_finite() or position_size_base <= 0: self.logger.error(f"Calculated size invalid: {position_size_base}"); return None
            self.logger.debug(f"Raw position size: {position_size_base:.{DECIMAL_DISPLAY_PRECISION+4}f} {base_asset}")
            
            try:
                limits = market.get('limits', {})
                min_amount_dec = Decimal(str(limits.get('amount', {}).get('min'))) if limits.get('amount', {}).get('min') is not None else None
                max_amount_dec = Decimal(str(limits.get('amount', {}).get('max'))) if limits.get('amount', {}).get('max') is not None else None
                
                quantized_size = self.exchange.quantize_value(position_size_base, 'amount', market, rounding_mode=ROUND_DOWN) # Using exchange's quantize
                if quantized_size is None: self.logger.error("Failed to quantize position size."); return None
                if quantized_size <= 0: self.logger.warning(f"Size {position_size_base} became zero ({quantized_size}) after quantization."); return None
                self.logger.debug(f"Size after step-quantization (ROUND_DOWN): {quantized_size} {base_asset}")
                
                if min_amount_dec is not None and quantized_size < min_amount_dec:
                    self.logger.warning(f"Quantized size {quantized_size} < min order size {min_amount_dec}."); return None
                
                final_size_base = quantized_size
                if max_amount_dec is not None and final_size_base > max_amount_dec:
                    self.logger.warning(f"Quantized size {final_size_base} > max limit {max_amount_dec}. Capping.")
                    final_size_base = self.exchange.quantize_value(max_amount_dec, 'amount', market, rounding_mode=ROUND_DOWN)
                    if final_size_base is None or final_size_base <= 0: self.logger.error("Failed to quantize capped max size."); return None
                
                if final_size_base <= 0: self.logger.error(f"Final size non-positive ({final_size_base})."); return None
                self.logger.info(f"Calculated Final Position Size: {final_size_base} {base_asset}")
                return final_size_base
            except (ValueError, Exception) as e: self.logger.error(f"Error applying market limits/precision to size: {e}", exc_info=True); return None

    def quantize_price(self, price: Decimal, side_for_conservative_rounding: Optional[PositionSide] = None) -> Optional[Decimal]:
         if not isinstance(price, Decimal) or not price.is_finite(): self.logger.error(f"Invalid price for quantization: {price}"); return None
         market = self.exchange.get_market(self.symbol)
         if not market: self.logger.error(f"Market data not found for {self.symbol} for price quantization."); return None
         
         rounding_mode = ROUND_HALF_UP
         if side_for_conservative_rounding:
              if side_for_conservative_rounding == PositionSide.LONG: rounding_mode = ROUND_DOWN
              elif side_for_conservative_rounding == PositionSide.SHORT: rounding_mode = ROUND_UP
         
         quantized_price = self.exchange.quantize_value(price, 'price', market, rounding_mode=rounding_mode) # Using exchange's quantize
         if quantized_price is None: return None
         self.logger.debug(f"Quantized price {price} to {quantized_price} (Rounding: {rounding_mode}, Side: {side_for_conservative_rounding.name if side_for_conservative_rounding else 'Default'})")
         return quantized_price

    def calculate_stop_loss(self, entry_price: Decimal, side: PositionSide, latest_indicators: pd.Series) -> Optional[Decimal]:
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0: self.logger.error(f"Invalid entry price for SL: {entry_price}"); return None
        sl_method = self.risk_config.get("stop_loss_method", "atr").lower()
        stop_loss_price_raw = None
        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION
            if sl_method == "atr":
                atr_multiplier = self.risk_config.get("atr_multiplier", Decimal("1.5"))
                atr_period = self.indicator_config.get("atr_period", 14)
                atr_col = f"ATR_{atr_period}" if isinstance(atr_period, int) else None
                if not atr_col or atr_col not in latest_indicators or pd.isna(latest_indicators[atr_col]): self.logger.error(f"ATR '{atr_col}' not found/NaN."); return None
                atr_value = latest_indicators[atr_col]
                if not isinstance(atr_value, Decimal) or not atr_value.is_finite() or atr_value <= 0: self.logger.error(f"Invalid ATR value ({atr_value})."); return None
                stop_distance = atr_value * atr_multiplier
                if side == PositionSide.LONG: stop_loss_price_raw = entry_price - stop_distance
                elif side == PositionSide.SHORT: stop_loss_price_raw = entry_price + stop_distance
                else: self.logger.error("Invalid side for SL calc."); return None
                self.logger.debug(f"ATR SL: Entry={entry_price}, Side={side.name}, ATR={atr_value}, Multiplier={atr_multiplier}, Raw SL={stop_loss_price_raw}")
            elif sl_method == "fixed_percent":
                fixed_percent = self.risk_config.get("fixed_stop_loss_percent", Decimal("2.0")) / Decimal("100")
                if side == PositionSide.LONG: stop_loss_price_raw = entry_price * (Decimal("1") - fixed_percent)
                elif side == PositionSide.SHORT: stop_loss_price_raw = entry_price * (Decimal("1") + fixed_percent)
                else: self.logger.error("Invalid side for SL calc."); return None
                self.logger.debug(f"Fixed % SL: Entry={entry_price}, Side={side.name}, Percent={fixed_percent*100}%, Raw SL={stop_loss_price_raw}")
            else: self.logger.error(f"Unknown SL method: {sl_method}"); return None

        if stop_loss_price_raw is None or not stop_loss_price_raw.is_finite() or stop_loss_price_raw <= 0: self.logger.error(f"Raw SL price ({stop_loss_price_raw}) invalid."); return None
        quantized_sl = self.quantize_price(stop_loss_price_raw, side_for_conservative_rounding=side)
        if quantized_sl is None: self.logger.error("Failed to quantize SL price."); return None
        
        quantized_entry = self.quantize_price(entry_price) # For fair comparison
        if quantized_entry:
            if side == PositionSide.LONG and quantized_sl >= quantized_entry: self.logger.error(f"Quantized SL {quantized_sl} >= entry {quantized_entry}. Invalid."); return None
            if side == PositionSide.SHORT and quantized_sl <= quantized_entry: self.logger.error(f"Quantized SL {quantized_sl} <= entry {quantized_entry}. Invalid."); return None
        
        self.logger.info(f"Calculated Initial SL Price (Quantized): {quantized_sl}")
        return quantized_sl

    def check_ma_cross_exit(self, indicators_df: pd.DataFrame, position_side: PositionSide) -> bool:
        if not self.trading_config.get("use_ma_cross_exit", False): return False
        ema_s_p, ema_l_p = self.indicator_config.get("ema_short_period"), self.indicator_config.get("ema_long_period")
        ema_s_col, ema_l_col = (f"EMA_{ema_s_p}" if isinstance(ema_s_p, int) else None), (f"EMA_{ema_l_p}" if isinstance(ema_l_p, int) else None)
        if not ema_s_col or not ema_l_col: self.logger.warning("MA cross exit enabled, but EMA periods not configured."); return False
        if len(indicators_df) < 2: self.logger.debug("Not enough data for MA cross check."); return False
        try: latest_data, prev_data = indicators_df.iloc[-1], indicators_df.iloc[-2]
        except IndexError: self.logger.warning("Could not get latest/prev indicator data for MA cross."); return False
        if not all(c in latest_data and c in prev_data for c in [ema_s_col, ema_l_col]): self.logger.warning("EMA columns missing for MA cross."); return False
        
        ema_s_now, ema_l_now, ema_s_prev, ema_l_prev = latest_data[ema_s_col], latest_data[ema_l_col], prev_data[ema_s_col], prev_data[ema_l_col]
        if not all(isinstance(e, Decimal) and e.is_finite() for e in [ema_s_now, ema_l_now, ema_s_prev, ema_l_prev]): self.logger.warning("Invalid EMA values for MA cross."); return False
        
        exit_signal = False
        if position_side == PositionSide.LONG and ema_s_prev >= ema_l_prev and ema_s_now < ema_l_now:
            self.logger.info("MA Cross Exit for LONG (Short EMA crossed below Long EMA)."); exit_signal = True
        elif position_side == PositionSide.SHORT and ema_s_prev <= ema_l_prev and ema_s_now > ema_l_now:
            self.logger.info("MA Cross Exit for SHORT (Short EMA crossed above Long EMA)."); exit_signal = True
        return exit_signal

    def manage_stop_loss(self, position: Dict[str, Any], latest_indicators: pd.Series, current_state: Dict[str, Any]) -> Optional[Decimal]:
        if not position or position.get('side') == 'none' or position.get('side') is None: self.logger.debug("No active position/side for SL management."); return None
        if not current_state.get('active_position'): self.logger.warning("manage_stop_loss called but state has no active position."); return None

        entry_price, current_sl_state = position.get('entryPrice'), current_state.get('stop_loss_price')
        position_side_str, mark_price = position.get('side'), position.get('markPrice')
        if not all(isinstance(val, Decimal) and val.is_finite() and val > 0 for val in [entry_price, current_sl_state, mark_price] if val is not None):
            self.logger.warning(f"Invalid E/SL/Mark for SL mgmt: E={entry_price}, SL_state={current_sl_state}, Mark={mark_price}"); return None
        try: position_side = PositionSide(position_side_str)
        except ValueError: self.logger.error(f"Invalid side string '{position_side_str}'."); return None

        new_sl_price_proposal, proposed_sl_type, state_updated_this_cycle = None, "", False
        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION
            atr_value = None
            atr_period, atr_col = self.indicator_config.get("atr_period", 14), (f"ATR_{self.indicator_config.get('atr_period', 14)}" if isinstance(self.indicator_config.get('atr_period', 14), int) else None)
            if atr_col and atr_col in latest_indicators and pd.notna(latest_indicators[atr_col]):
                 val = latest_indicators[atr_col]
                 if isinstance(val, Decimal) and val.is_finite() and val > 0: atr_value = val
                 else: self.logger.warning(f"ATR value ({val}) invalid/non-positive.")
            else: self.logger.warning(f"ATR column '{atr_col}' not found/NaN.")

            use_be = self.risk_config.get("use_break_even_sl", False)
            if use_be and atr_value is not None and not current_state.get('break_even_achieved', False):
                be_trigger_atr, be_offset_atr = self.risk_config.get("break_even_trigger_atr", Decimal("1.0")), self.risk_config.get("break_even_offset_atr", Decimal("0.1"))
                profit_target_dist, offset_dist = atr_value * be_trigger_atr, atr_value * be_offset_atr
                target_be_price_raw = entry_price + offset_dist if position_side == PositionSide.LONG else entry_price - offset_dist
                quantized_be_price = self.quantize_price(target_be_price_raw, side_for_conservative_rounding=position_side)
                if quantized_be_price:
                    be_triggered = (position_side == PositionSide.LONG and mark_price >= (entry_price + profit_target_dist)) or \
                                   (position_side == PositionSide.SHORT and mark_price <= (entry_price - profit_target_dist))
                    if be_triggered:
                        is_better = (position_side == PositionSide.LONG and quantized_be_price > current_sl_state) or \
                                    (position_side == PositionSide.SHORT and quantized_be_price < current_sl_state)
                        if is_better:
                            self.logger.info(f"BE Triggered ({position_side.name}): Proposing SL update from {current_sl_state} to BE {quantized_be_price}")
                            new_sl_price_proposal, proposed_sl_type, state_updated_this_cycle = quantized_be_price, "BE", True
                        else: self.logger.debug(f"BE triggered, but proposed BE {quantized_be_price} not better than current SL {current_sl_state}.")
                else: self.logger.warning("Failed to quantize BE price.")
            
            use_tsl = self.risk_config.get("use_trailing_sl", False)
            if use_tsl and atr_value is not None and not state_updated_this_cycle:
                 tsl_atr_mult = self.risk_config.get("trailing_sl_atr_multiplier", Decimal("2.0"))
                 trail_dist = atr_value * tsl_atr_mult
                 potential_tsl_raw = mark_price - trail_dist if position_side == PositionSide.LONG else mark_price + trail_dist
                 if potential_tsl_raw:
                     quantized_tsl = self.quantize_price(potential_tsl_raw, side_for_conservative_rounding=position_side)
                     if quantized_tsl:
                         is_better = (position_side == PositionSide.LONG and quantized_tsl > current_sl_state) or \
                                     (position_side == PositionSide.SHORT and quantized_tsl < current_sl_state)
                         if is_better:
                             self.logger.debug(f"TSL Update ({position_side.name}): Potential TSL {quantized_tsl} better than Current SL {current_sl_state}")
                             if current_state.get('break_even_achieved', False): # Clamp to entry after BE
                                  quantized_entry = self.quantize_price(entry_price)
                                  if quantized_entry:
                                       if position_side == PositionSide.LONG and quantized_tsl < quantized_entry: quantized_tsl = quantized_entry
                                       elif position_side == PositionSide.SHORT and quantized_tsl > quantized_entry: quantized_tsl = quantized_entry
                             new_sl_price_proposal, proposed_sl_type = quantized_tsl, "TSL"
                     else: self.logger.warning(f"Failed to quantize potential TSL price ({potential_tsl_raw}).")

        if new_sl_price_proposal:
            if not new_sl_price_proposal.is_finite() or new_sl_price_proposal <= 0: self.logger.warning(f"Proposed SL ({new_sl_price_proposal}) invalid."); return None
            price_tick_size = None
            market = self.exchange.get_market(self.symbol)
            if market and 'precision' in market and 'price' in market['precision']:
                 tick_str = market['precision']['price'];
                 if tick_str: try: price_tick_size = Decimal(str(tick_str)); except: pass
            tolerance = (price_tick_size / Decimal(2)) if price_tick_size else Decimal('1e-9')
            
            sl_invalid = (position_side == PositionSide.LONG and new_sl_price_proposal > (mark_price + tolerance)) or \
                         (position_side == PositionSide.SHORT and new_sl_price_proposal < (mark_price - tolerance))
            if sl_invalid: self.logger.warning(f"Proposed SL {new_sl_price_proposal} invalid vs mark price {mark_price}."); return None
            
            if abs(new_sl_price_proposal - current_sl_state) > tolerance:
                 self.logger.info(f"Proposing {proposed_sl_type} SL update from {current_sl_state} to {new_sl_price_proposal}")
                 return new_sl_price_proposal
            else: self.logger.debug(f"New {proposed_sl_type} SL {new_sl_price_proposal} not significantly different from current {current_sl_state}."); return None
        return None
```

```python
