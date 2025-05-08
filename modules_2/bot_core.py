# File: bot_core.py
import sys
import time
import logging
import pandas as pd # For type hints and Timedelta
import ccxt # For ccxt.AuthenticationError, ccxt.NetworkError
from pathlib import Path
from decimal import Decimal # For type hints

# Assuming other modules are in the same directory
from app_config import load_config, load_state, save_state, STATE_FILE
from logger_utils import setup_logger
from trading_enums import Signal, PositionSide, OrderSide
from exchange_interface import BybitV5Wrapper
from indicator_analyzer import TradingAnalyzer
from risk_manager import PositionManager

class TradingBot:
    """
    The main trading bot class orchestrating the fetch-analyze-execute loop.
    Manages state, interacts with exchange wrapper, analyzer, and position manager.
    """
    def __init__(self, config_path: Path, state_path: Path):
        self.config_path = config_path
        self.state_path = state_path
        self.logger = setup_logger() # Initial setup, level may change

        self.config = load_config(config_path, self.logger)
        if not self.config:
            self.logger.critical("Failed to load/validate configuration. Exiting.")
            sys.exit(1)

        self._apply_log_level_from_config()
        self.state = load_state(state_path, self.logger)
        self._initialize_default_state()

        try:
            self.exchange = BybitV5Wrapper(self.config, self.logger)
            self.analyzer = TradingAnalyzer(self.config, self.logger)
            self.position_manager = PositionManager(self.config, self.logger, self.exchange)
        except Exception as e:
             self.logger.critical(f"Failed to initialize core components: {e}. Exiting.", exc_info=True)
             sys.exit(1)

        self._load_trading_parameters()
        self.is_running = True

    def _apply_log_level_from_config(self):
        log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
        log_level_enum = getattr(logging, log_level_str, None)
        if isinstance(log_level_enum, int) and self.logger.level != log_level_enum:
             self.logger.setLevel(log_level_enum)
             for handler in self.logger.handlers: handler.setLevel(log_level_enum)
             self.logger.info(f"Log level set to {log_level_str} from config.")

    def _initialize_default_state(self):
        self.state.setdefault('active_position', None)
        self.state.setdefault('stop_loss_price', None)
        self.state.setdefault('take_profit_price', None)
        self.state.setdefault('break_even_achieved', False)
        self.state.setdefault('last_order_id', None)
        self.state.setdefault('last_sync_time', None)

    def _load_trading_parameters(self):
        settings = self.config.get('trading_settings', {})
        self.symbol = settings.get('symbol')
        self.timeframe = settings.get('timeframe')
        self.leverage = settings.get('leverage')
        self.quote_asset = settings.get('quote_asset')
        self.category = settings.get('category')
        self.poll_interval = settings.get('poll_interval_seconds', 60)
        self.hedge_mode = settings.get('hedge_mode', False)
        self.exit_on_opposing_signal = settings.get('exit_on_opposing_signal', True)
        self.use_ma_cross_exit = settings.get('use_ma_cross_exit', False)
        self.post_order_verify_delay = settings.get('post_order_verify_delay_seconds', 5)
        if not all([self.symbol, self.timeframe, self.leverage, self.quote_asset, self.category]):
             self.logger.critical("Essential trading settings missing. Exiting.")
             sys.exit(1)

    def run(self):
        self.logger.info(f"--- Starting Trading Bot ---")
        self.logger.info(f"Symbol: {self.symbol}, TF: {self.timeframe}, Cat: {self.category}, Quote: {self.quote_asset}, Lev: {self.leverage}x, Hedge: {self.hedge_mode}, Poll: {self.poll_interval}s")
        if not self.initialize_exchange_settings(): self.logger.critical("Failed to init exchange settings. Exiting."); sys.exit(1)
        
        self.logger.info("Performing initial position sync...")
        initial_position = self.get_current_position()
        self.sync_bot_state_with_position(initial_position)
        save_state(self.state, self.state_path, self.logger)

        while self.is_running:
            try:
                self.logger.info(f"--- New Trading Cycle ({pd.Timestamp.utcnow()}) ---")
                start_time = time.time()
                ohlcv_limit = self.config.get("indicator_settings", {}).get("ohlcv_fetch_limit", 250)
                ohlcv_data = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=ohlcv_limit)
                if ohlcv_data is None or ohlcv_data.empty: self._wait_for_next_cycle(start_time); continue
                
                indicators_df = self.analyzer.calculate_indicators(ohlcv_data)
                if indicators_df is None or indicators_df.empty: self._wait_for_next_cycle(start_time); continue
                try:
                     if indicators_df.iloc[-1].name < (pd.Timestamp.utcnow() - pd.Timedelta(minutes=15)):
                          self.logger.warning(f"Latest indicator data ({indicators_df.iloc[-1].name}) stale. Skipping."); self._wait_for_next_cycle(start_time); continue
                     latest_indicators = indicators_df.iloc[-1]
                except IndexError: self.logger.warning("Indicators DataFrame empty. Skipping."); self._wait_for_next_cycle(start_time); continue
                
                signal, signal_details = self.analyzer.generate_signal(indicators_df)
                current_position_on_exchange = self.get_current_position()
                self.sync_bot_state_with_position(current_position_on_exchange)

                if self.state.get('active_position'):
                    self.manage_existing_position(indicators_df, signal, current_position_on_exchange)
                else:
                    self.attempt_new_entry(signal, latest_indicators, signal_details)
                
                save_state(self.state, self.state_path, self.logger)
                self._wait_for_next_cycle(start_time)
            except KeyboardInterrupt: self.logger.info("KeyboardInterrupt. Stopping bot..."); self.is_running = False
            except ccxt.AuthenticationError: self.logger.critical("Authentication failed. Stopping bot.", exc_info=True); self.is_running = False
            except ccxt.NetworkError as e: self.logger.error(f"Network error: {e}. Retrying after poll interval."); time.sleep(self.poll_interval)
            except Exception as e:
                self.logger.critical(f"Critical unexpected error in main loop: {e}", exc_info=True)
                self.logger.info("Attempting to continue. Manual check advised."); time.sleep(self.poll_interval * 2)
        self.logger.info("--- Trading Bot stopped ---")

    def initialize_exchange_settings(self) -> bool:
        self.logger.info("Initializing exchange settings...")
        success = True
        if self.category in ['linear', 'inverse']:
             if not self.exchange.set_leverage(self.symbol, self.leverage):
                  self.logger.error(f"Failed to set initial leverage to {self.leverage}x."); success = False
             else: self.logger.info(f"Leverage set request to {self.leverage}x sent.")
        
        if self.exchange.exchange_id == 'bybit' and self.category in ['linear', 'inverse']:
             try:
                  target_mode, mode_name = (3, 'Hedge') if self.hedge_mode else (0, 'One-way')
                  self.logger.info(f"Checking/Setting position mode (Target: {mode_name})...")
                  params = {'symbol': self.symbol, 'mode': target_mode}
                  method_name = 'private_post_position_switch_mode'
                  if hasattr(self.exchange.exchange, method_name):
                       result = self.exchange.safe_ccxt_call(method_name, params=params)
                       if result and result.get('retCode') == 0: self.logger.info(f"Position mode set to {mode_name}.")
                       elif result and result.get('retCode') == 110025: self.logger.info(f"Position mode already {mode_name}.")
                       elif result: self.logger.error(f"Failed to set position mode. Code: {result.get('retCode')}, Msg: {result.get('retMsg')}"); success = False
                       else: self.logger.error("API call failed for position mode."); success = False
                  else: self.logger.warning(f"CCXT method '{method_name}' not found. Cannot set position mode.")
             except Exception as e: self.logger.error(f"Error setting position mode: {e}", exc_info=True); success = False
        return success

    def get_current_position(self) -> Optional[dict]:
        positions = self.exchange.fetch_positions(self.symbol)
        if positions is None: self.logger.warning(f"Could not fetch positions for {self.symbol}. Assuming none."); return None
        if not positions: self.logger.info(f"No active position found on exchange for {self.symbol}."); return None

        if self.hedge_mode:
            state_pos = self.state.get('active_position')
            if state_pos:
                 target_idx, target_side_val = state_pos.get('position_idx'), state_pos.get('side', PositionSide.NONE).value
                 if target_idx is None or target_side_val == PositionSide.NONE.value:
                      self.logger.error("Hedge mode active, but bot state inconsistent. Cannot ID position."); return None
                 for p in positions:
                      if p.get('positionIdx') == target_idx and p.get('side') == target_side_val:
                           self.logger.info(f"Found active hedge position matching state: Side {p.get('side')}, Idx {p.get('positionIdx')}, Size {p.get('contracts')}")
                           return p
                 self.logger.warning(f"Bot state indicates hedge position (Idx: {target_idx}, Side: {target_side_val}), but no match found."); return None
            else:
                 pos_details = [f"Idx:{p.get('positionIdx')}, Side:{p.get('side')}, Size:{p.get('contracts')}" for p in positions]
                 self.logger.info(f"Ignoring {len(positions)} active hedge position(s) as bot state is empty: [{'; '.join(pos_details)}]"); return None
        else: # One-way
            if len(positions) > 1: self.logger.error(f"CRITICAL: Found {len(positions)} active positions for {self.symbol} in non-hedge mode!"); return None
            pos = positions[0]
            if pos.get('positionIdx') != 0: self.logger.warning(f"One-way mode, but found pos with non-zero index ({pos.get('positionIdx')}).")
            self.logger.info(f"Found active non-hedge position: Side {pos.get('side')}, Size {pos.get('contracts')}, Idx {pos.get('positionIdx')}")
            return pos

    def sync_bot_state_with_position(self, current_pos_on_exchange: Optional[dict]):
        bot_state_pos = self.state.get('active_position')
        bot_thinks_has_pos = bot_state_pos is not None
        state_changed = False

        if current_pos_on_exchange:
            ex_sym, ex_side_str, ex_size, ex_entry, ex_idx = current_pos_on_exchange.get('symbol'), current_pos_on_exchange.get('side'), current_pos_on_exchange.get('contracts'), current_pos_on_exchange.get('entryPrice'), current_pos_on_exchange.get('positionIdx')
            ex_sl_dec = current_pos_on_exchange.get('info', {}).get('stopLoss')
            if not all([ex_sym, ex_side_str != PositionSide.NONE.value, ex_size is not None, ex_entry is not None]):
                 self.logger.error("Fetched exchange position data incomplete/invalid. Cannot sync.")
                 if bot_thinks_has_pos: self._clear_position_state("Incomplete exchange data"); state_changed = True
                 return
            if ex_sl_dec is not None and (not ex_sl_dec.is_finite() or ex_sl_dec <= 0): ex_sl_dec = None

            if not bot_thinks_has_pos:
                matches_mode = (self.hedge_mode and ex_idx in [1, 2]) or (not self.hedge_mode and ex_idx == 0)
                if matches_mode:
                     self.logger.warning(f"Found unexpected active position ({ex_side_str}, Idx:{ex_idx}). Adopting.");
                     try: adopted_side = PositionSide(ex_side_str)
                     except ValueError: self.logger.error(f"Invalid side '{ex_side_str}'. Cannot adopt."); return
                     self.state['active_position'] = {'symbol': ex_sym, 'side': adopted_side, 'entry_price': ex_entry, 'size': ex_size, 'position_idx': ex_idx, 'order_id': None}
                     self.state['stop_loss_price'], self.state['take_profit_price'], self.state['break_even_achieved'] = ex_sl_dec, None, False; state_changed = True
                     if ex_sl_dec: self.logger.info(f"Adopted SL {ex_sl_dec} from exchange.")
                else: self.logger.warning(f"Found active position ({ex_side_str}, Idx:{ex_idx}) NOT matching bot mode. Ignoring.")
            else: # Bot knew about a position
                state_side_enum, state_idx = bot_state_pos['side'], bot_state_pos.get('position_idx')
                match = (ex_sym == bot_state_pos['symbol'] and ex_side_str == state_side_enum.value) and \
                        ((self.hedge_mode and ex_idx == state_idx) or (not self.hedge_mode and ex_idx == 0))
                if match:
                     self.logger.debug("Exchange position matches state. Syncing details...")
                     if bot_state_pos['size'] != ex_size: self.logger.info(f"Pos size changed: State={bot_state_pos['size']}, Ex={ex_size}. Updating."); bot_state_pos['size'] = ex_size; state_changed = True
                     if bot_state_pos['entry_price'] != ex_entry: self.logger.info(f"Pos entry changed: State={bot_state_pos['entry_price']}, Ex={ex_entry}. Updating."); bot_state_pos['entry_price'] = ex_entry; state_changed = True
                     
                     current_state_sl, tolerance = self.state.get('stop_loss_price'), Decimal('1e-9')
                     if ex_sl_dec is not None:
                          if current_state_sl is None or abs(ex_sl_dec - current_state_sl) > tolerance:
                               self.logger.info(f"Updating state SL from {current_state_sl} to exchange SL {ex_sl_dec}."); self.state['stop_loss_price'] = ex_sl_dec; state_changed = True
                               self._check_and_update_be_status(bot_state_pos, ex_sl_dec)
                     elif current_state_sl is not None:
                          self.logger.warning(f"Bot state SL {current_state_sl}, but no SL on exchange. Clearing state SL."); self.state['stop_loss_price'] = None; state_changed = True
                          if self.state.get('break_even_achieved', False): self.state['break_even_achieved'] = False
                     self.state['last_sync_time'] = time.time()
                else:
                     self.logger.warning(f"Found active pos ({ex_sym},{ex_side_str},Idx:{ex_idx}) NOT matching bot state ({bot_state_pos.get('symbol')},{state_side_enum.value},Idx:{state_idx}). Clearing bot state.")
                     self._clear_position_state("Mismatch with exchange position"); state_changed = True
        else: # No position on exchange
            if bot_thinks_has_pos:
                self.logger.info(f"Position ({bot_state_pos.get('side', PositionSide.NONE).name}, Size:{bot_state_pos.get('size')}) no longer on exchange. Clearing state.")
                self._clear_position_state("Position closed/missing on exchange"); state_changed = True
        if state_changed: save_state(self.state, self.state_path, self.logger)

    def _check_and_update_be_status(self, position_state: dict, current_sl: Decimal):
        if not self.state.get('break_even_achieved', False):
            entry, side = position_state.get('entry_price'), position_state.get('side')
            if entry is None or side is None or current_sl is None: return
            quantized_entry = self.position_manager.quantize_price(entry)
            if quantized_entry is None: return
            be_achieved = (side == PositionSide.LONG and current_sl >= quantized_entry) or \
                          (side == PositionSide.SHORT and current_sl <= quantized_entry)
            if be_achieved: self.logger.info(f"Marking BE achieved (SL {current_sl} vs Entry {quantized_entry})."); self.state['break_even_achieved'] = True

    def _clear_position_state(self, reason: str):
        self.logger.info(f"Clearing position state. Reason: {reason}")
        self.state['active_position'] = None
        self.state['stop_loss_price'] = None
        self.state['take_profit_price'] = None
        self.state['break_even_achieved'] = False

    def manage_existing_position(self, indicators_df: pd.DataFrame, signal: Signal, live_pos_data: Optional[dict]):
        pos_state = self.state.get('active_position')
        if not pos_state: self.logger.warning("manage_existing_position called but no active state position."); return
        if not live_pos_data: self.logger.error("manage_existing_position: no live position data. Sync failed or pos closed."); return

        pos_side, pos_idx = pos_state['side'], pos_state.get('position_idx')
        self.logger.info(f"Managing existing {pos_side.name} position (Idx: {pos_idx if self.hedge_mode else 'N/A'})...")

        if self.use_ma_cross_exit and self.position_manager.check_ma_cross_exit(indicators_df, pos_side):
            self.logger.info("MA Cross exit condition met."); self.close_position("MA Cross Exit"); return
        
        exit_on_signal = (self.exit_on_opposing_signal and 
                         ((pos_side == PositionSide.LONG and signal in [Signal.SELL, Signal.STRONG_SELL]) or \
                          (pos_side == PositionSide.SHORT and signal in [Signal.BUY, Signal.STRONG_BUY])))
        if exit_on_signal: self.logger.info(f"Opposing signal ({signal.name}) received."); self.close_position(f"Opposing Signal ({signal.name})"); return

        current_sl_in_state = self.state.get('stop_loss_price')
        if current_sl_in_state is None: self.logger.warning("Cannot manage SL: SL price missing from state."); return
        
        latest_indicators = indicators_df.iloc[-1]
        new_sl_price = self.position_manager.manage_stop_loss(live_pos_data, latest_indicators, self.state)
        if new_sl_price:
            self.logger.info(f"Attempting to update SL on exchange to: {new_sl_price}")
            protection_params = {'stop_loss': new_sl_price}
            if self.hedge_mode: protection_params['position_idx'] = pos_idx
            if self.exchange.set_protection(self.symbol, **protection_params):
                 self.logger.info(f"SL update request successful for {new_sl_price}.")
                 self.state['stop_loss_price'] = new_sl_price
                 self._check_and_update_be_status(pos_state, new_sl_price)
                 save_state(self.state, self.state_path, self.logger)
            else: self.logger.error(f"Failed to update SL to {new_sl_price}. API call failed. State SL remains {current_sl_in_state}.")
        self.logger.debug("Position management cycle finished.")

    def attempt_new_entry(self, signal: Signal, latest_indicators: pd.Series, signal_details: dict):
        if self.state.get('active_position'): self.logger.debug("Skipping entry: Position already active."); return
        
        target_side, order_side = None, None
        if signal in [Signal.BUY, Signal.STRONG_BUY]: target_side, order_side = PositionSide.LONG, OrderSide.BUY
        elif signal in [Signal.SELL, Signal.STRONG_SELL]: target_side, order_side = PositionSide.SHORT, OrderSide.SELL
        if not target_side: self.logger.info(f"Signal {signal.name}. No entry condition."); return
        self.logger.info(f"Entry signal {signal.name}. Preparing {target_side.name} entry...")

        entry_price_est = latest_indicators.get('close')
        if not isinstance(entry_price_est, Decimal) or not entry_price_est.is_finite() or entry_price_est <= 0:
             ticker = self.exchange.safe_ccxt_call('fetch_ticker', self.symbol)
             if ticker and ticker.get('last'): try: entry_price_est = Decimal(str(ticker['last'])); except: entry_price_est = None
             if not isinstance(entry_price_est, Decimal) or not entry_price_est.is_finite() or entry_price_est <= 0:
                  self.logger.error("Failed to get valid entry price estimate. Cannot enter."); return
        
        stop_loss_price = self.position_manager.calculate_stop_loss(entry_price_est, target_side, latest_indicators)
        if stop_loss_price is None: self.logger.error("Failed to calculate SL. Cannot enter."); return
        
        balance_info = self.exchange.fetch_balance()
        if not balance_info or self.quote_asset not in balance_info: self.logger.error(f"Failed to fetch balance or quote '{self.quote_asset}' missing."); return
        available_equity = balance_info[self.quote_asset]
        if not isinstance(available_equity, Decimal) or not available_equity.is_finite() or available_equity <= 0: self.logger.error(f"Invalid equity {available_equity}."); return
        
        position_size = self.position_manager.calculate_position_size(entry_price_est, stop_loss_price, available_equity, self.quote_asset)
        if position_size is None or position_size <= 0: self.logger.warning("Position size calc failed or non-positive. No entry."); return

        self.logger.info(f"Attempting {order_side.value} market order: Size={position_size}, Est.Entry={entry_price_est}, Calc.SL={stop_loss_price}")
        sl_price_str = self.exchange.format_value_for_api(self.symbol, 'price', stop_loss_price)
        order_params = {'stopLoss': sl_price_str}
        pos_idx = (1 if order_side == OrderSide.BUY else 2) if self.hedge_mode else 0
        order_params['positionIdx'] = pos_idx
        
        order_result = self.exchange.create_order(symbol=self.symbol, order_type='market', side=order_side, amount=position_size, price=None, params=order_params)
        if order_result and order_result.get('id'):
            order_id = order_result['id']
            self.logger.info(f"Entry order placed. ID: {order_id}, Status: {order_result.get('status', 'unknown')}. Intended SL: {stop_loss_price}")
            self.state['last_order_id'] = order_id
            self.state['active_position'] = {'symbol': self.symbol, 'side': target_side, 'entry_price': entry_price_est, 'size': position_size, 'order_id': order_id, 'position_idx': pos_idx}
            self.state['stop_loss_price'], self.state['take_profit_price'], self.state['break_even_achieved'] = stop_loss_price, None, False
            save_state(self.state, self.state_path, self.logger)
            self.logger.info(f"Bot state updated optimistically for new {target_side.name} position.")

            if self.post_order_verify_delay > 0:
                 self.logger.info(f"Waiting {self.post_order_verify_delay}s before verifying entry..."); time.sleep(self.post_order_verify_delay)
                 self.logger.info("Verifying position status after entry...")
                 final_pos = self.get_current_position(); self.sync_bot_state_with_position(final_pos)
                 if self.state.get('active_position'):
                      state_sl = self.state.get('stop_loss_price')
                      if state_sl and abs(state_sl - stop_loss_price) < Decimal('1e-9'): self.logger.info(f"SL {stop_loss_price} confirmed active post-verification.")
                      elif state_sl: self.logger.warning(f"State SL ({state_sl}) differs from intended ({stop_loss_price}) post-sync.")
                      else: # No SL found
                           self.logger.warning("SL NOT found on exchange post-order. Attempting to set SL again.")
                           protect_params = {'stop_loss': stop_loss_price}
                           if self.hedge_mode: protect_params['position_idx'] = self.state['active_position'].get('position_idx')
                           if self.exchange.set_protection(self.symbol, **protect_params):
                                self.logger.info("Successfully set SL via set_protection after initial attempt failed."); self.state['stop_loss_price'] = stop_loss_price; save_state(self.state, self.state_path, self.logger)
                           else: self.logger.error("Retry failed to set SL after entry.")
                 else: self.logger.error("Position not found during verification post-entry. State cleared by sync.")
        else:
            self.logger.error("Failed to place entry order."); self._clear_position_state("Entry order placement failed"); save_state(self.state, self.state_path, self.logger)

    def close_position(self, reason: str):
        pos_state = self.state.get('active_position')
        if not pos_state: self.logger.warning("Close position called but no active state position."); return
        size_to_close, current_side, pos_idx = pos_state.get('size'), pos_state.get('side'), pos_state.get('position_idx')
        if not isinstance(size_to_close, Decimal) or size_to_close <= 0: self.logger.error(f"Cannot close: Invalid size {size_to_close}."); self._clear_position_state(f"Invalid size during close"); save_state(self.state, self.state_path, self.logger); return
        if current_side is None or current_side == PositionSide.NONE: self.logger.error(f"Cannot close: Invalid side {current_side}."); self._clear_position_state(f"Invalid side during close"); save_state(self.state, self.state_path, self.logger); return
        if self.hedge_mode and pos_idx is None: self.logger.error("Cannot close hedge: posIdx missing."); self._clear_position_state("Missing hedge idx during close"); save_state(self.state, self.state_path, self.logger); return

        close_side = OrderSide.SELL if current_side == PositionSide.LONG else OrderSide.BUY
        self.logger.info(f"Attempting to close {current_side.name} position (Idx:{pos_idx if self.hedge_mode else 'N/A'}, Size:{size_to_close}) via market. Reason: {reason}")
        close_params = {'reduceOnly': True, 'positionIdx': pos_idx if self.hedge_mode else 0}
        
        order_result = self.exchange.create_order(symbol=self.symbol, order_type='market', side=close_side, amount=size_to_close, price=None, params=close_params)
        if order_result and order_result.get('id'):
            self.logger.info(f"Position close order placed. ID: {order_result['id']}, Status: {order_result.get('status', 'unknown')}. Reason: {reason}")
            self._clear_position_state(f"Close order placed (ID: {order_result['id']}, Reason: {reason})")
            self.state['last_order_id'] = order_result['id']; save_state(self.state, self.state_path, self.logger)
        else: self.logger.error(f"Failed to place position close order. Reason: {reason}. Position state unchanged.")

    def _wait_for_next_cycle(self, cycle_start_time: float):
        execution_time = time.time() - cycle_start_time
        wait_time = max(0, self.poll_interval - execution_time)
        self.logger.debug(f"Cycle exec time: {execution_time:.2f}s. Waiting {wait_time:.2f}s...")
        if wait_time > 0: time.sleep(wait_time)

```

```python
